import hashlib
import json
import pathlib
import re
import struct
import sys

cache, ptx_path, sass_path, resources_path, compile_path, cap = sys.argv[1:]
envelope = pathlib.Path(cache).read_bytes()
assert envelope[:16] == b"MAMBA-PTX-CACHE\0"
assert struct.unpack("<H", envelope[16:18])[0] == 1
assert envelope[50] == 1, "cache must contain PTX"
payload = envelope[91:]
assert struct.unpack("<Q", envelope[51:59])[0] == len(payload)
assert envelope[59:91] == hashlib.sha256(payload).digest()
assert payload == pathlib.Path(ptx_path).read_bytes()
sass = pathlib.Path(sass_path).read_text()
resources = pathlib.Path(resources_path).read_text()
compile_log = pathlib.Path(compile_path).read_text()
proofs = []
for dtype in ["bf16", "f16"]:
    symbol = f"gemm_bi_nn_fixed_sm89_tc128_s3_v1_{dtype}"
    body = next(block for block in re.split(r"\n\s*Function : ", sass) if block.startswith(symbol + "\n"))
    resource = re.search(rf"Function {symbol}:\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
    assert resource, symbol
    values = list(map(int, resource.groups()))
    assert 1 <= values[0] <= int(cap) and values[1:] == [0, 0, 0], values
    info = re.search(rf"Function properties for {symbol}\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s+: Used (\d+) registers", compile_log)
    assert info and list(map(int, info.groups()[:3])) == [0, 0, 0], symbol
    assert not re.search(r"\b(?:LDL|STL|ATOM|RED|REDUX)\b", body)
    patterns = {"commit": r"\bLDGDEPBAR\b", "wait0": r"DEPBAR\.LE SB0, 0x0", "wait1": r"DEPBAR\.LE SB0, 0x1", "mma": r"\bHMMA\.16816", "matrix_load": r"\bLDSM\b", "async_copy": r"\bLDGSTS\b", "barrier": r"\bBAR\.SYNC"}
    counts = {key: len(re.findall(pattern, body)) for key, pattern in patterns.items()}
    assert all(counts.values()), counts
    proofs.append(dict(symbol=symbol, registers=values[0], stack=0, static_shared=0, local=0, spill_stores=0, spill_loads=0, sass_counts=counts))
print(json.dumps(dict(verdict="PASS", cache=cache, cache_sha256=hashlib.sha256(envelope).hexdigest(), compile_key=envelope[18:50].hex(), artifact_sha256=hashlib.sha256(payload).hexdigest(), proofs=proofs), indent=2))
