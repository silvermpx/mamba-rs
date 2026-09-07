import hashlib
import json
import pathlib
import re
import tarfile

root = pathlib.Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def manifest(path):
    return {match[2]: match[1] for line in path.read_text().splitlines() if (match := re.fullmatch(r"([a-f0-9]{64})  (.+)", line))}

results = []
for tag, attempt, archive_name, binary_count in [
    ("128", "cuda128-continuation1", "cuda128-final-binaries-cache-v2.tar.gz", 6),
    ("130", "cuda130-attempt1", "cuda130-final-binaries-cache.tar.gz", 7),
    ("132", "cuda132-attempt1", "cuda132-final-binaries-cache.tar.gz", 7),
]:
    identities = manifest(root / attempt / "identities-final.log")
    control = json.loads((root / f"identity-cuda{tag}.json").read_text())
    source_count = 0
    with tarfile.open(root / "source-final-v2.tar.gz") as archive:
        for member in archive.getmembers():
            # macOS tar carries AppleDouble metadata; it is not compiler input.
            if not member.isfile() or pathlib.PurePosixPath(member.name).name.startswith("._"):
                continue
            actual = digest(archive.extractfile(member).read())
            assert identities.get(member.name) == actual, (tag, "source mismatch", member.name)
            source_count += 1
    checked = 0
    fixed_found = 0
    with tarfile.open(root / archive_name) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            data = archive.extractfile(member).read()
            actual = digest(data)
            assert identities.get("/root/" + member.name) == actual, (tag, "binary/cache mismatch", member.name)
            checked += 1
            if member.name.endswith(".bin") and b"gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16" in data:
                assert digest(data[91:]) == control["fixed_artifact_digest"]
                assert data[18:50].hex() == control["fixed_invocation_digest"]
                fixed_found += 1
    assert checked == binary_count + 3 and fixed_found == 1, (tag, checked, fixed_found)
    results.append(dict(toolkit=tag, archived_sources=source_count, archived_test_binaries=binary_count, archived_cache_envelopes=3, fixed_identity_match=True))
print(json.dumps(dict(verdict="PASS", results=results), indent=2))
