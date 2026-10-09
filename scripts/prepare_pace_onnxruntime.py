#!/usr/bin/env python3
"""Extract the CPU C library from Microsoft's pinned manylinux ONNX Runtime wheel.

The Python extension is not used; no Python package or environment is modified.
The wheel targets glibc 2.27/2.28, compatible with PACE's glibc 2.34.
"""
import argparse
import ctypes
import hashlib
from pathlib import Path
import platform
import subprocess
import zipfile

VERSION = "1.27.0"
URL = "https://files.pythonhosted.org/packages/e0/cd/74bb804170ceb622fda9111df31a07b3024f7491472256d3a90b5391a4d2/onnxruntime-1.27.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
SHA256 = "e4f7b0e90d2d212e2c2deaa6c8291616183ab815d3ec558ea12d3ac8b26d36f4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this pinned runtime is for Linux x86_64 only")
    args.output.mkdir(parents=True, exist_ok=True)
    wheel = args.output / "onnxruntime.whl"
    if not wheel.exists():
        temporary = wheel.with_suffix(".download")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "-o", str(temporary), URL], check=True)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            temporary.unlink()
            raise RuntimeError("ONNX Runtime wheel checksum mismatch")
        temporary.rename(wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(f"Cached wheel checksum mismatch: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("onnxruntime/capi/libonnxruntime") and ".so" in name:
                target = args.output / Path(name).name
                contents = archive.read(name)
                if target.exists():
                    if target.read_bytes() != contents:
                        raise RuntimeError(f"Cached runtime checksum mismatch: {target}")
                else:
                    temporary = target.with_suffix(target.suffix + ".partial")
                    temporary.write_bytes(contents)
                    temporary.rename(target)
    library = args.output / f"libonnxruntime.so.{VERSION}"
    for name in ("libonnxruntime.so", "libonnxruntime.so.1"):
        link = args.output / name
        if not link.exists():
            link.symlink_to(library.name)

    # Fail before compilation if the host cannot load the library or API 27.
    runtime = ctypes.CDLL(str(library.resolve()))
    get_api = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
    get_version = ctypes.CFUNCTYPE(ctypes.c_char_p)
    class ApiBase(ctypes.Structure):
        _fields_ = [("GetApi", get_api), ("GetVersionString", get_version)]
    runtime.OrtGetApiBase.restype = ctypes.POINTER(ApiBase)
    api = runtime.OrtGetApiBase().contents
    version = api.GetVersionString().decode()
    if version != VERSION or not api.GetApi(27):
        raise RuntimeError(f"Unexpected runtime version or missing API 27: {version}")
    print(f"ONNX Runtime {version}: host load and API 27 check passed")
    (args.output / "runtime.sha256").write_text(
        f"{hashlib.sha256(library.read_bytes()).hexdigest()}  {library.name}\n")


if __name__ == "__main__":
    main()
