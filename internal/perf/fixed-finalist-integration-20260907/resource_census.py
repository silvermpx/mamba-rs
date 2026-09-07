#!/usr/bin/env python3
"""Read raw CUDA function resources from the frozen Fixed PTX cache."""

import ctypes
import hashlib
import json
import os
from pathlib import Path


SYMBOLS = {
    "rna_n96": (
        b"gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3",
        256,
        86_016,
    ),
    "half_m64n64_s3_f16": (
        b"gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16",
        128,
        49_152,
    ),
    "half_m128n64_s2_f16": (
        b"gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16",
        128,
        49_152,
    ),
}


def check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUresult {result}")


def main() -> None:
    cache_dir = Path(os.environ["CENSUS_CACHE_DIR"])
    output = Path(os.environ["CENSUS_OUTPUT"])
    candidates = []
    for path in cache_dir.glob("mamba-kernels-v1-*.bin"):
        raw = path.read_bytes()
        if all(symbol in raw for symbol, _, _ in SYMBOLS.values()):
            candidates.append((path, raw))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Fixed cache artifact, found {len(candidates)}")
    path, raw = candidates[0]
    if raw[:16] != b"MAMBA-PTX-CACHE\0" or int.from_bytes(raw[16:18], "little") != 1:
        raise RuntimeError("cache envelope magic/version mismatch")
    if raw[50] != 1:
        raise RuntimeError("cache payload is not PTX")
    payload_len = int.from_bytes(raw[51:59], "little")
    if len(raw) != 91 + payload_len:
        raise RuntimeError("cache payload length mismatch")
    ptx = raw[91:]

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cuda.cuFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

    check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    try:
        module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(ptx + b"\0")
        check(cuda.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        try:
            driver = ctypes.c_int()
            check(cuda.cuDriverGetVersion(ctypes.byref(driver)), "cuDriverGetVersion")
            records = {}
            for label, (symbol, threads, dynamic_shared) in SYMBOLS.items():
                function = ctypes.c_void_p()
                check(
                    cuda.cuModuleGetFunction(ctypes.byref(function), module, symbol),
                    f"cuModuleGetFunction({symbol.decode()})",
                )
                check(
                    cuda.cuFuncSetAttribute(function, 8, dynamic_shared),
                    f"cuFuncSetAttribute({symbol.decode()})",
                )
                attributes = {}
                for name, attribute in [
                    ("max_threads", 0),
                    ("static_shared_bytes", 1),
                    ("local_bytes", 3),
                    ("registers", 4),
                ]:
                    value = ctypes.c_int()
                    check(
                        cuda.cuFuncGetAttribute(ctypes.byref(value), attribute, function),
                        f"cuFuncGetAttribute({symbol.decode()},{attribute})",
                    )
                    attributes[name] = value.value
                active = ctypes.c_int()
                check(
                    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        ctypes.byref(active), function, threads, dynamic_shared
                    ),
                    f"cuOccupancyMaxActiveBlocksPerMultiprocessor({symbol.decode()})",
                )
                records[label] = {
                    "symbol": symbol.decode(),
                    "threads": threads,
                    "dynamic_shared_bytes": dynamic_shared,
                    "active_blocks": active.value,
                    **attributes,
                }
        finally:
            check(cuda.cuModuleUnload(module), "cuModuleUnload")
    finally:
        check(cuda.cuCtxDestroy_v2(context), "cuCtxDestroy")

    result = {
        "cache_path": str(path),
        "cache_sha256": hashlib.sha256(raw).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
        "driver_version": driver.value,
        "records": records,
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
