"""Unit tests plus optional real-data ABI/ownership integration checks."""

import ctypes as C
import gc
import sys
import unittest
import weakref
from pathlib import Path

import numpy as np
from python_boundary import ArrowArray, ArrowSchema, Library, Source
from python_perception import fuse
from run_python_boundary import summarize

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (
    ROOT
    / "target/release"
    / ("libpipes_rs.dylib" if sys.platform == "darwin" else "libpipes_rs.so")
)
CONFIG = {
    "data_root": str(ROOT / "data/waymo-v2-sample"),
    "split": "training",
    "segment": "10023947602400723454_1120_000_1140_000",
    "cameras": [1, 2, 3, 4, 5],
    "lidars": [1, 2, 3, 4, 5],
    "returns": [1, 2],
    "frame_limit": 1,
}


class FusionTests(unittest.TestCase):
    def test_two_projection_slots_do_not_double_count(self):
        arrays = {
            k: np.array(v, dtype=np.float32)
            for k, v in {
                "camera1": [1, 2, 1, 1],
                "camera2": [2, 2, 0, 0],
                "u1": [5, 5, 20, 5],
                "v1": [5, 5, 20, 5],
                "u2": [5, 5, 0, 0],
                "v2": [5, 5, 0, 0],
                "range": [2, 4, 10, float("nan")],
            }.items()
        }
        box = np.array([[0, 0, 10, 10, 0.9, 2]], dtype=np.float32)
        output = fuse(arrays, 2, box)
        self.assertEqual(output[0, 7], 4)
        self.assertEqual(output[0, 8], 2)
        self.assertEqual(fuse(arrays, 1, box)[0, 8], 1)
        self.assertTrue(np.isnan(fuse(arrays, 3, box)[0, 7]))
        self.assertEqual(fuse(arrays, 1, np.empty((0, 6))).shape, (0, 9))

    def test_summary_pairs_by_frame_before_combining(self):
        rows = []
        for frame, native, arrow in [(1, 10, 20), (2, 100, 100)]:
            for variant, total in [("native_shared", native), ("arrow", arrow)]:
                for repeat in range(2):
                    rows.append(
                        dict(
                            segment="test",
                            timestamp=frame,
                            condition="prepared",
                            variant=variant,
                            **{
                                m: total
                                for m in [
                                    "conversion_ns",
                                    "boundary_ns",
                                    "preprocess_ns",
                                    "inference_ns",
                                    "postprocess_ns",
                                    "fusion_ns",
                                    "return_ns",
                                    "total_ns",
                                ]
                            },
                        )
                    )
        result = summarize(rows)
        self.assertEqual(
            next(s for s in result if s["variant"] == "arrow")[
                "paired_total_over_native_shared"
            ],
            1.5,
        )


@unittest.skipUnless(
    LIBRARY.exists() and (ROOT / "data/waymo-v2-sample/training/lidar").exists(),
    "build Rust library and supply local Waymo data for integration tests",
)
class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = Library(LIBRARY)

    def frame(self):
        with Source(self.library, CONFIG) as source:
            frame = source.next()
            self.assertIsNone(source.next())
        return frame

    def test_sensor_coverage_and_all_three_representations(self):
        frame = self.frame()
        meta = frame.metadata
        self.assertEqual(meta["cameras"], [1, 2, 3, 4, 5])
        self.assertEqual(
            {(s["lidar"], s["return"]) for s in meta["sensors"]},
            {(i, r) for i in range(1, 6) for r in [1, 2]},
        )
        native = frame.arrays("native_shared")
        copied = frame.arrays("native_copy")
        frame.prepare_arrow()
        arrow = frame.arrays("arrow")
        frame.clear_arrow()
        for key in native:
            np.testing.assert_array_equal(native[key], copied[key])
            np.testing.assert_array_equal(native[key], arrow[key])
            self.assertFalse(native[key].flags.writeable)
            self.assertFalse(np.shares_memory(native[key], copied[key]))
        self.assertGreater(native["range"].size, 0)
        self.assertEqual(C.sizeof(ArrowArray), 80)
        self.assertEqual(C.sizeof(ArrowSchema), 72)

    def test_native_numpy_views_retain_owner(self):
        frame = self.frame()
        ref = weakref.ref(frame)
        array = np.asarray(frame.arrays("native_shared")["range"])[::2]
        expected = array.copy()
        del frame
        gc.collect()
        self.assertIsNotNone(ref())
        np.testing.assert_array_equal(array, expected)
        del array
        gc.collect()
        self.assertIsNone(ref())

    def test_arrow_import_owns_buffers_after_rust_frame_dies(self):
        frame = self.frame()
        ref = weakref.ref(frame)
        frame.prepare_arrow()
        array = frame.arrays("arrow")["camera_5"]
        expected = array.copy()
        frame.clear_arrow()
        del frame
        gc.collect()
        self.assertIsNone(ref())
        np.testing.assert_array_equal(array, expected)

    def test_missing_component_and_invalid_sensor_fail(self):
        for config in [
            dict(CONFIG, data_root="/nonexistent-waymo-data"),
            dict(CONFIG, cameras=[6]),
        ]:
            with self.assertRaises(RuntimeError):
                Source(self.library, config)

    def test_arrow_export_requires_preparation(self):
        frame = self.frame()
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            frame.arrays("arrow")


if __name__ == "__main__":
    unittest.main()
