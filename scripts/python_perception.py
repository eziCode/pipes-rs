"""One Python model and depth-fusion implementation for every representation."""

import numpy as np
from PIL import Image


class Yolox:
    def __init__(self, model, threads=1, confidence=0.3, nms=0.45):
        import onnxruntime as ort

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.confidence = confidence
        self.nms = nms
        self.grids = []
        self.strides = []
        for stride in (8, 16, 32):
            yy, xx = np.mgrid[: 416 // stride, : 416 // stride]
            self.grids.append(np.stack([xx, yy], axis=-1).reshape(-1, 2))
            self.strides.append(np.full((xx.size, 1), stride))
        self.grids = np.concatenate(self.grids).astype(np.float32)
        self.strides = np.concatenate(self.strides).astype(np.float32)

    def preprocess(self, rgb):
        h, w = rgb.shape[:2]
        ratio = min(416 / w, 416 / h)
        # Same unavoidable resize/layout conversion in all three paths.
        resized = np.asarray(
            Image.fromarray(np.asarray(rgb)).resize(
                (int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR
            )
        )
        tensor = np.full((1, 3, 416, 416), 114, dtype=np.float32)
        tensor[0, :, : resized.shape[0], : resized.shape[1]] = resized[
            :, :, ::-1
        ].transpose(2, 0, 1)
        return tensor, ratio

    def infer(self, tensor):
        raw = self.session.run(None, {self.input: tensor})[0]
        if raw.shape != (1, len(self.grids), 85):
            raise ValueError(f"Expected raw YOLOX Nano output, got {raw.shape}")
        return raw[0]

    def postprocess(self, raw, ratio):
        classes = raw[:, 5:].argmax(axis=1)
        score = raw[:, 4] * raw[np.arange(len(raw)), 5 + classes]
        keep = (score >= self.confidence) & np.isin(classes, [0, 1, 2, 3, 5, 7, 11])
        values = raw[keep]
        scores = score[keep]
        cls = classes[keep]
        center = (values[:, :2] + self.grids[keep]) * self.strides[keep] / ratio
        size = np.exp(values[:, 2:4]) * self.strides[keep] / ratio
        boxes = np.concatenate([center - size / 2, size], axis=1)
        order = np.argsort(-scores, kind="stable")
        selected = []
        while len(order):
            i = order[0]
            selected.append(i)
            rest = order[1:]
            lo = np.maximum(boxes[i, :2], boxes[rest, :2])
            hi = np.minimum(
                boxes[i, :2] + boxes[i, 2:], boxes[rest, :2] + boxes[rest, 2:]
            )
            intersection = np.maximum(hi - lo, 0).prod(axis=1)
            union = boxes[i, 2:].prod() + boxes[rest, 2:].prod(axis=1) - intersection
            order = rest[
                intersection / np.maximum(union, np.finfo(np.float32).eps) <= self.nms
            ]
        # x,y,width,height,score,COCO class. No representation-specific code here.
        return np.column_stack(
            [boxes[selected], scores[selected], cls[selected]]
        ).astype(np.float32)


def fuse(arrays, camera, boxes):
    """Use both projection slots, but count each physical return at most once."""
    first = arrays["camera1"] == camera
    second = (~first) & (arrays["camera2"] == camera)
    visible = first | second
    u = np.where(first, arrays["u1"], arrays["u2"])[visible].astype(np.float64)
    v = np.where(first, arrays["v1"], arrays["v2"])[visible].astype(np.float64)
    ranges = arrays["range"][visible]
    output = np.empty((len(boxes), 9), dtype=np.float32)
    for i, box in enumerate(boxes):
        x, y, w, h = map(float, box[:4])
        matched = ranges[
            (u >= x) & (u <= x + w) & (v >= y) & (v <= y + h) & np.isfinite(ranges)
        ]
        depth = np.sort(matched)[len(matched) // 2] if len(matched) else np.nan
        output[i] = [camera, *box, depth, len(matched)]
    return output


def process(arrays, cameras, model):
    from time import perf_counter_ns

    timings = {
        "preprocess_ns": 0,
        "inference_ns": 0,
        "postprocess_ns": 0,
        "fusion_ns": 0,
    }
    results = []
    for camera in cameras:
        start = perf_counter_ns()
        tensor, ratio = model.preprocess(arrays[f"camera_{camera}"])
        ready = perf_counter_ns()
        raw = model.infer(tensor)
        inferred = perf_counter_ns()
        boxes = model.postprocess(raw, ratio)
        detected = perf_counter_ns()
        results.append(fuse(arrays, camera, boxes))
        finished = perf_counter_ns()
        for key, delta in zip(
            timings,
            [ready - start, inferred - ready, detected - inferred, finished - detected],
        ):
            timings[key] += delta
    return np.concatenate(results), timings
