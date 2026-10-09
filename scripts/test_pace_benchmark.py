"""Contract tests with deliberately corrupted measurements and detection values."""
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_pace_benchmark as runner


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root / name / "perception.csv" for name in ("pipes", "dora")]
        for path in self.paths:
            path.parent.mkdir()
            rows = []
            for i in range(6):
                t = 1_000_000_000 + i * 1000
                r = dict.fromkeys(runner.FIELDS, "1")
                r.update(timestamp_micros=str(i), status="delivered", camera_started_ns=t,
                         lidar_started_ns=t+1, camera_sent_ns=t+2, lidar_sent_ns=t+3,
                         camera_received_ns=t+4, lidar_received_ns=t+6, pair_ready_ns=t+8,
                         fusion_finished_ns=t+10, camera_queue_wait_ns=2, lidar_queue_wait_ns=3,
                         camera_pair_wait_ns=4, lidar_pair_wait_ns=2, end_to_end_ns=10,
                         camera_stage_ns=2, lidar_stage_ns=2, inference_ns=1, fusion_ns=2)
                rows.append(r)
            self.write_rows(path, rows)
            path.with_suffix(".setup.txt").write_text("submitted=6\ndelivered=6\ndropped=0\n")
            values = [dict(timestamp_micros=i, points=1, points_crc32="deadbeef", detections=[dict(
                id="onnx-0", kind="vehicle", depth_points=2, x=1.0, y=2.0,
                width=3.0, height=4.0, score=0.75, depth_m=10.0)]) for i in range(6)]
            path.with_suffix(".validation.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))

    @staticmethod
    def write_rows(path, rows):
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def change_values(self, fn):
        path = self.paths[1].with_suffix(".validation.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        fn(rows)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def test_matching_values_and_accounting(self):
        self.assertEqual(runner.compare(*self.paths), 6)
        self.assertEqual(runner.compare_values(*self.paths), 6)
        runner.validate_accounting(self.paths[0].parent, runner.read_metrics(self.paths[0]))

    def test_changed_depth_rejected_despite_equal_counts(self):
        self.change_values(lambda rows: rows[0]["detections"][0].update(depth_m=20.0))
        with self.assertRaisesRegex(RuntimeError, "depth_m mismatch"):
            runner.compare_values(*self.paths)

    def test_small_numeric_roundoff_accepted(self):
        self.change_values(lambda rows: rows[0]["detections"][0].update(x=1.000001))
        runner.compare_values(*self.paths)

    def test_point_field_corruption_rejected(self):
        self.change_values(lambda rows: rows[0].update(points_crc32="bad"))
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            runner.compare_values(*self.paths)

    def test_missing_validation_frame_rejected(self):
        self.change_values(lambda rows: rows.pop())
        with self.assertRaisesRegex(RuntimeError, "validation timestamps"):
            runner.compare_values(*self.paths)

    def test_invalid_clock_order_rejected(self):
        rows = runner.read_metrics(self.paths[0])
        rows[0]["camera_received_ns"] = 0
        self.write_rows(self.paths[0], rows)
        with self.assertRaisesRegex(RuntimeError, "timing order"):
            runner.read_metrics(self.paths[0])

    def test_misreported_accounting_rejected(self):
        self.paths[0].with_suffix(".setup.txt").write_text("submitted=7\ndelivered=6\ndropped=1\n")
        with self.assertRaisesRegex(RuntimeError, "incomplete delivery"):
            runner.validate_accounting(self.paths[0].parent, runner.read_metrics(self.paths[0]))

    def test_excess_outstanding_processing_rejected(self):
        rows = runner.read_metrics(self.paths[0])
        for row in rows:
            row["fusion_finished_ns"] = 1_000_010_000
            row["end_to_end_ns"] = 1_000_010_000 - int(row["camera_started_ns"])
        self.write_rows(self.paths[0], rows)
        with self.assertRaisesRegex(RuntimeError, "concurrency exceeds"):
            runner.read_metrics(self.paths[0])

    def test_warmup_within_run_and_wall_time_separate(self):
        summary = runner.summarize(self.paths[0].parent, "pipes-rs", runner.read_metrics(self.paths[0]),
                                   2_000_000_000, 900_000_000, 2)
        self.assertEqual(summary["steady_frames"], 4)
        self.assertEqual(summary["wall_seconds"], 2)
        self.assertEqual(summary["post_warmup_active_seconds"], 3010 / 1e9)

    def test_subprocess_runner_separates_validation_and_timing(self):
        executable = self.root / "fake-framework"
        executable.write_text('''#!/usr/bin/env python3
import csv, os, pathlib, shutil, time
source = pathlib.Path(os.environ["PILOT_TEST_FIXTURE"])
target = pathlib.Path(os.environ["OUTPUT_CSV"])
with source.open() as handle:
    rows = list(csv.DictReader(handle))
offset = time.monotonic_ns() - 1_000_000_000
for row in rows:
    for key in ("camera_started_ns", "lidar_started_ns", "camera_sent_ns", "lidar_sent_ns", "camera_received_ns", "lidar_received_ns", "pair_ready_ns", "fusion_finished_ns"):
        row[key] = int(row[key]) + offset
with target.open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
shutil.copyfile(source.with_suffix(".setup.txt"), target.with_suffix(".setup.txt"))
if os.environ["PILOT_VALIDATE"] == "1":
    shutil.copyfile(source.with_suffix(".validation.jsonl"), target.with_suffix(".validation.jsonl"))
''')
        executable.chmod(0o755)
        segment = dict(split="training", context_name="test")
        data = self.root / "data"
        for component in runner.COMPONENTS:
            path = data / "training" / component / "test.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"test input")
        model = self.root / "model"
        model.write_bytes(b"test model")
        args = SimpleNamespace(data_root=data, model=model, warmup_frames=2)
        binaries = {k: str(executable) for k in ("pipes-rs", "dora", "node")}
        with patch.dict(os.environ, PILOT_TEST_FIXTURE=str(self.paths[0])):
            a = runner.run("pipes-rs", self.root / "validation-pipes", args, segment, binaries, 6, True)
            b = runner.run("dora", self.root / "validation-dora", args, segment, binaries, 6, True)
            self.assertEqual(runner.compare_values(a, b), 6)
            timed = runner.run("pipes-rs", self.root / "timed", args, segment, binaries, 6)
        self.assertFalse(timed.with_suffix(".validation.jsonl").exists())
        summary = json.loads((timed.parent / "pilot-summary.json").read_text())
        self.assertEqual(summary["steady_frames"], 4)
        self.assertGreater(summary["launch_to_first_result_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
