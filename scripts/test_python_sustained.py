"""Queue overload, failure propagation, and mapped-buffer ownership checks."""

import copy
import gc
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from python_boundary import Frame, Library
from python_sustained import Work, WorkQueue, measure, queue_statistics
from test_python_boundary import LIBRARY


class QueueTests(unittest.TestCase):
    def item(self, sequence):
        return Work(sequence, {"payload_bytes": 100}, None, 0, 0)

    def test_drop_oldest_keeps_latest_and_drains_after_close(self):
        queue = WorkQueue(2, "drop-oldest")
        self.assertIsNone(queue.put(self.item(0)))
        queue.put(self.item(1))
        self.assertEqual(queue.put(self.item(2)).sequence, 0)
        queue.close()
        self.assertEqual([queue.get()[0].sequence for _ in range(2)], [1, 2])
        self.assertIsNone(queue.get())
        self.assertTrue(all(e["depth"] <= 2 for e in queue.events))
        self.assertEqual(queue.bytes, 0)

    def test_backpressure_and_cancellation_unblock_producer(self):
        queue = WorkQueue(1, "backpressure")
        queue.put(self.item(0))
        waiting, finished = threading.Event(), threading.Event()
        errors = []

        def producer():
            waiting.set()
            try:
                queue.put(self.item(1))
            except RuntimeError as error:
                errors.append(str(error))
            finally:
                finished.set()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        self.assertTrue(waiting.wait(1))
        self.assertFalse(finished.wait(0.02))
        queue.close()
        self.assertTrue(finished.wait(1))
        thread.join()
        self.assertEqual(len(errors), 1)

    def test_queue_integral_includes_idle_and_full_intervals(self):
        events = [
            {"time_ns": 10, "depth": 1, "queued_payload_bytes": 100},
            {"time_ns": 20, "depth": 2, "queued_payload_bytes": 200},
            {"time_ns": 40, "depth": 0, "queued_payload_bytes": 0},
        ]
        result = queue_statistics(events, 0, 50, 2)
        self.assertEqual(result["mean_queue_depth"], 1)
        self.assertEqual(result["mean_queued_payload_bytes"], 100)
        self.assertEqual(result["fraction_queue_full"], 0.4)


class FakeFrame:
    def clone(self):
        return FakeFrame()

    def arrays(self, variant):
        return {}


class RunnerTests(unittest.TestCase):
    def run_measure(
        self,
        policy="backpressure",
        rate=0,
        processor=None,
        base=None,
        variant="native_shared",
    ):
        a = SimpleNamespace(
            queue_size=1,
            queue_policy=policy,
            passes=20,
            rate=rate,
            variant=variant,
            condition="conversion_inclusive"
            if variant == "native_producer_copy"
            else "prepared",
            cameras=[1],
        )
        entry = {
            "payload_bytes": 100,
            "expected_crc32": 7,
            "segment": "test",
            "split": "training",
            "metadata": {"timestamp": 1},
        }

        def slow_process(*_):
            time.sleep(0.002)
            return [], {"fusion_ns": 1}

        return measure(
            [entry],
            [base or FakeFrame()],
            SimpleNamespace(consume=lambda _: 7),
            None,
            a,
            processor=processor or slow_process,
        )

    def test_saturation_has_backpressure_without_lost_items(self):
        rows, events, start, *_ = self.run_measure(rate=10000)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["status"] == "delivered" for r in rows))
        self.assertGreater(sum(r["backpressure_ns"] for r in rows), 0)
        self.assertEqual(max(e["depth"] for e in events), 1)
        for row in rows:
            self.assertEqual(row["due_ns"], start + row["sequence"] * 100000)

    def test_overload_drops_frames_without_resetting_arrival_schedule(self):
        rows, _, start, *_ = self.run_measure("drop-oldest", rate=10000)
        self.assertTrue(any(r["status"] == "dropped" for r in rows))
        self.assertTrue(any(r["status"] == "delivered" for r in rows))
        for row in rows:
            self.assertEqual(row["due_ns"], start + row["sequence"] * 100000)

    def test_consumer_failure_does_not_hang_blocked_producer(self):
        def fail(*_):
            raise ValueError("model failed")

        with self.assertRaisesRegex(ValueError, "model failed"):
            self.run_measure(processor=fail)

    def test_producer_failure_reaches_caller(self):
        class BrokenFrame:
            def clone(self):
                raise ValueError("bad source")

        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            self.run_measure(base=BrokenFrame())

    def test_native_copy_runs_on_producer_before_consumer(self):
        copied_on, consumed_on = [], []

        class TrackedFrame:
            copied = False

            def clone(self):
                return TrackedFrame()

            def prepare_native_copy(self):
                copied_on.append(threading.current_thread().name)
                self.copied = True

            def arrays(self, variant):
                if not self.copied or variant != "native_producer_copy":
                    raise AssertionError("consumer received unprepared native input")
                consumed_on.append(threading.current_thread().name)
                return {}

        rows, *_ = self.run_measure(base=TrackedFrame(), variant="native_producer_copy")
        self.assertEqual(copied_on, ["sensor-producer"] * 20)
        self.assertEqual(consumed_on, [threading.current_thread().name] * 20)
        self.assertTrue(all(r["producer_copy_ns"] > 0 for r in rows))
        self.assertEqual(sum(r["producer_copy_bytes"] for r in rows), 2000)


@unittest.skipUnless(LIBRARY.exists(), "build Rust library for mapped ABI tests")
class MappedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.bin"
        self.expected = np.arange(16, dtype=np.float32)
        self.expected.tofile(self.path)
        self.meta = {
            "buffers": [
                {"name": "range", "dtype": "float32", "shape": [16], "offset": 0}
            ]
        }
        self.lib = Library(LIBRARY)

    def test_shared_arrow_survives_all_frame_handles(self):
        base = Frame.cached(self.lib, self.path, self.meta)
        base.prepare_arrow_shared()
        frame = base.clone()
        native = frame.arrays("native_shared")["range"]
        arrow = frame.arrays("arrow")["range"]
        self.assertEqual(native.ctypes.data, arrow.ctypes.data)
        del native, base, frame
        gc.collect()
        np.testing.assert_array_equal(arrow, self.expected)
        self.assertFalse(arrow.flags.writeable)

    def test_materialized_arrow_copies_and_native_view_retains_mapping(self):
        base = Frame.cached(self.lib, self.path, self.meta)
        frame = base.clone()
        frame.prepare_arrow()
        native = np.asarray(frame.arrays("native_shared")["range"])
        arrow = frame.arrays("arrow")["range"]
        self.assertNotEqual(native.ctypes.data, arrow.ctypes.data)
        del base, frame
        gc.collect()
        np.testing.assert_array_equal(native, self.expected)
        np.testing.assert_array_equal(arrow, self.expected)

    def test_invalid_mapping_descriptors_are_rejected(self):
        for changes in (
            {"offset": 1},
            {"shape": [17]},
            {"dtype": "int64"},
            {"shape": [2**63, 4]},
        ):
            meta = copy.deepcopy(self.meta)
            meta["buffers"][0].update(changes)
            with self.assertRaises(RuntimeError):
                Frame.cached(self.lib, self.path, meta)

    def test_native_producer_copy_is_independent_and_handoff_does_not_copy(self):
        # Exercise float, RGB byte, and empty buffers in one mapped frame.
        self.path.write_bytes(self.expected.tobytes() + bytes(range(64)))
        self.meta["buffers"].extend(
            [
                {"name": "camera", "dtype": "uint8", "shape": [4, 4, 4], "offset": 64},
                {"name": "empty", "dtype": "float32", "shape": [0], "offset": 128},
            ]
        )
        base = Frame.cached(self.lib, self.path, self.meta)
        source = base.arrays("native_shared")
        frame = base.clone()
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            frame.arrays("native_producer_copy")
        frame.prepare_native_copy()
        copied = frame.arrays("native_producer_copy")
        again = frame.arrays("native_producer_copy")
        for key, value in copied.items():
            np.testing.assert_array_equal(value, source[key])
            self.assertEqual(value.ctypes.data, again[key].ctypes.data)
            self.assertFalse(value.flags.writeable)
            self.assertEqual(value.ctypes.data % 64, 0)
            if value.size:
                self.assertNotEqual(value.ctypes.data, source[key].ctypes.data)
        with self.assertRaises(RuntimeError):
            frame.prepare_native_copy()
        with self.assertRaises(RuntimeError):
            frame.prepare_arrow()
        with self.assertRaises(RuntimeError):
            frame.clone()
        view = np.asarray(copied["range"])
        del base, frame, source, copied, again
        gc.collect()
        np.testing.assert_array_equal(view, self.expected)


if __name__ == "__main__":
    unittest.main()
