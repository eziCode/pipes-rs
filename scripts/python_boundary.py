"""Ownership-safe ctypes bridge to the Rust-owned multisensor source.

Each arm uses an in-process Rust/Python boundary; sustained arms run independently.
Arrow's low-level C import is isolated here and covered by lifetime tests.
"""

import ctypes as C
import json

import numpy as np
import pyarrow as pa


class ArrowArray(C.Structure):
    _fields_ = [
        (n, C.c_int64)
        for n in ("length", "null_count", "offset", "n_buffers", "n_children")
    ] + [
        ("buffers", C.POINTER(C.c_void_p)),
        ("children", C.c_void_p),
        ("dictionary", C.c_void_p),
        ("release", C.c_void_p),
        ("private_data", C.c_void_p),
    ]


class ArrowSchema(C.Structure):
    _fields_ = [
        ("format", C.c_char_p),
        ("name", C.c_char_p),
        ("metadata", C.c_char_p),
        ("flags", C.c_int64),
        ("n_children", C.c_int64),
        ("children", C.c_void_p),
        ("dictionary", C.c_void_p),
        ("release", C.c_void_p),
        ("private_data", C.c_void_p),
    ]


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("length", C.c_size_t), ("kind", C.c_uint32)]


class OwnedArray(np.ndarray):
    """Keep the Rust frame alive for all derived native NumPy views."""

    def __array_finalize__(self, original):
        self._owner = getattr(original, "_owner", None)


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        signatures = {
            "pb_consume_result": ([C.c_void_p, C.c_size_t], C.c_uint64),
            "pb_error": ([], C.c_char_p),
            "pb_cached_open": ([C.c_char_p, C.c_char_p], C.c_void_p),
            "pb_frame_clone": ([C.c_void_p], C.c_void_p),
            "pb_prepare_native_copy": ([C.c_void_p], C.c_int),
            "pb_prepare_arrow_shared": ([C.c_void_p], C.c_int),
            "pb_open": ([C.c_char_p], C.c_void_p),
            "pb_source_free": ([C.c_void_p], None),
            "pb_next": ([C.c_void_p], C.c_void_p),
            "pb_frame_free": ([C.c_void_p], None),
            "pb_metadata": ([C.c_void_p], C.c_char_p),
            "pb_view": ([C.c_void_p, C.c_size_t, C.POINTER(View)], C.c_int),
            "pb_prepare_arrow": ([C.c_void_p], C.c_int),
            "pb_clear_arrow": ([C.c_void_p], None),
            "pb_export": (
                [C.c_void_p, C.c_size_t, C.POINTER(ArrowArray), C.POINTER(ArrowSchema)],
                C.c_int,
            ),
            "pb_export_release": (
                [C.POINTER(ArrowArray), C.POINTER(ArrowSchema)],
                None,
            ),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result

    def consume(self, result):
        result = np.ascontiguousarray(result, dtype=np.float32)
        checksum = self.lib.pb_consume_result(result.ctypes.data, result.size)
        if checksum == 2**64 - 1:
            raise RuntimeError(self.lib.pb_error().decode())
        return checksum

    def check(self, code):
        if code != 0:
            raise RuntimeError(self.lib.pb_error().decode())


class Source:
    def __init__(self, library, config):
        self.library = library
        self.handle = library.lib.pb_open(json.dumps(config).encode())
        if not self.handle:
            raise RuntimeError(library.lib.pb_error().decode())

    def next(self):
        handle = self.library.lib.pb_next(self.handle)
        if not handle:
            error = self.library.lib.pb_error().decode()
            if error:
                raise RuntimeError(error)
            return None
        return Frame(self.library, handle)

    def close(self):
        if self.handle:
            self.library.lib.pb_source_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Frame:
    def __init__(self, library, handle):
        self.library = library
        self.handle = handle
        self.metadata = json.loads(library.lib.pb_metadata(handle))

    def __del__(self):
        if getattr(self, "handle", None):
            self.library.lib.pb_frame_free(self.handle)

    @classmethod
    def cached(cls, library, path, metadata):
        handle = library.lib.pb_cached_open(
            str(path).encode(), json.dumps(metadata).encode()
        )
        if not handle:
            raise RuntimeError(library.lib.pb_error().decode())
        return cls(library, handle)

    def clone(self):
        handle = self.library.lib.pb_frame_clone(self.handle)
        if not handle:
            raise RuntimeError(self.library.lib.pb_error().decode())
        return Frame(self.library, handle)

    def prepare_arrow_shared(self):
        self.library.check(self.library.lib.pb_prepare_arrow_shared(self.handle))

    def prepare_arrow(self):
        self.library.check(self.library.lib.pb_prepare_arrow(self.handle))

    def prepare_native_copy(self):
        self.library.check(self.library.lib.pb_prepare_native_copy(self.handle))
        self.native_copy_prepared = True

    def clear_arrow(self):
        self.library.lib.pb_clear_arrow(self.handle)

    def arrays(self, variant):
        if variant == "native_producer_copy" and not getattr(
            self, "native_copy_prepared", False
        ):
            raise RuntimeError("native producer copy not prepared")
        result = {}
        for i, descriptor in enumerate(self.metadata["buffers"]):
            if variant == "arrow":
                a, s = ArrowArray(), ArrowSchema()
                self.library.check(
                    self.library.lib.pb_export(self.handle, i, C.byref(a), C.byref(s))
                )
                address = a.buffers[1]
                try:
                    # Import consumes the release callbacks; PyArrow then owns the buffers.
                    column = pa.Array._import_from_c(C.addressof(a), C.addressof(s))
                    array = column.to_numpy(zero_copy_only=True)
                finally:
                    self.library.lib.pb_export_release(C.byref(a), C.byref(s))
                if array.size and array.ctypes.data != address:
                    raise AssertionError("Arrow import unexpectedly copied a buffer")
            else:
                view = View()
                self.library.check(
                    self.library.lib.pb_view(self.handle, i, C.byref(view))
                )
                scalar = {1: C.c_float, 2: C.c_uint8}[view.kind]
                raw = (scalar * view.length).from_address(view.data)
                # Keep the owner on the ctypes object as well as ndarray subclasses:
                # np.asarray may strip the subclass but retains this buffer owner.
                raw._owner = self
                array = np.ctypeslib.as_array(raw).view(OwnedArray)
                array._owner = self
                if array.size and array.ctypes.data != view.data:
                    raise AssertionError("native view copied")
                if variant == "native_copy":
                    copied = np.array(array, copy=True, subok=False)
                    if array.size and copied.ctypes.data == array.ctypes.data:
                        raise AssertionError("copy aliased input")
                    array = copied
                elif variant not in ("native_shared", "native_producer_copy"):
                    raise ValueError(variant)
            array = array.reshape(descriptor["shape"])
            array.flags.writeable = False
            result[descriptor["name"]] = array
        return result
