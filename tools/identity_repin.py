#!/usr/bin/env python3
"""identity_repin.py <old_mint_dir> <new_mint_dir> [--apply]

Re-pin every frozen compile identity after a source change that moves no
bits (a kernel rename, a moved file, a comment).  Each pinned digest in the
tree is the OLD value of some (module, target, capacity, toolkit, field).
Run `identity_mint` (tools/qualification/identity_mint.rs) on the reference
tree and on this tree, once per toolkit, keeping the logs of each tree in
its own directory as mint-<toolkit>.log; this script maps key -> old value
and key -> new value from those logs and substitutes old -> new everywhere,
in both spellings the sources use: 64-hex strings and 32-byte decimal
arrays.  Run from the crate root.  A dry run prints what would change;
--apply writes it.  Digests it cannot map (retired cohorts, log hashes in
comments, file-level owner digests) are left alone and listed.
"""
import glob, json, os, re, sys

old_dir, new_dir = sys.argv[1], sys.argv[2]
APPLY = "--apply" in sys.argv
FIELDS = ["source", "compile_key", "invocation", "artifact", "header", "library"]

def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "mint-*.log")):
        for line in open(f):
            at = line.find("IDENTITY {")
            if at < 0:
                continue
            rec = json.loads(line[at + len("IDENTITY "):])
            key = (rec["module"], rec["target"], rec["cap"], tuple(rec["nvrtc"]))
            out[key] = rec
    return out

old, new = load(old_dir), load(new_dir)
print(f"old identities: {len(old)}  new identities: {len(new)}  shared keys: {len(set(old)&set(new))}")
hex_map = {}
conflicts = []
for key in sorted(set(old) & set(new)):
    for field in FIELDS:
        o, n = old[key][field], new[key][field]
        if o in (None, "null") or n in (None, "null"):
            continue
        if o == n:
            continue
        if o in hex_map and hex_map[o] != n:
            conflicts.append((key, field, o, hex_map[o], n))
        hex_map[o] = n
if conflicts:
    print("CONFLICTS (same old value, different new values):")
    for c in conflicts:
        print("  ", c)
    sys.exit(2)
print(f"distinct old->new digest pairs: {len(hex_map)}")

def to_bytes(h):
    return tuple(int(h[i:i + 2], 16) for i in range(0, 64, 2))
bytes_map = {to_bytes(o): to_bytes(n) for o, n in hex_map.items()}
ARRAY = re.compile(r"\[\s*((?:\d{1,3}\s*,\s*){31}\d{1,3})\s*,?\s*\]")
HEX = re.compile(r"\b[0-9a-f]{64}\b")

targets = []
for pat in ["src/**/*.rs", "tests/**/*.rs", "tools/**/*.rs", "benches/**/*.rs", "tests/**/*.jsonl", "*.md", "docs/**/*.md", "tools/**/*.md"]:
    targets += glob.glob(pat, recursive=True)
targets = sorted(set(targets))
used = {}
per_file = {}
def sub_hex(m):
    h = m.group(0)
    if h in hex_map:
        used[h] = used.get(h, 0) + 1
        return hex_map[h]
    return h
def sub_arr(m):
    vals = tuple(int(v) for v in re.split(r"\s*,\s*", m.group(1).strip()))
    if vals in bytes_map:
        h = "".join(f"{b:02x}" for b in vals)
        used[h] = used.get(h, 0) + 1
        return "[" + ", ".join(str(b) for b in bytes_map[vals]) + "]"
    return m.group(0)
for f in targets:
    text = open(f).read()
    n1 = HEX.sub(sub_hex, text)
    n2 = ARRAY.sub(sub_arr, n1)
    if n2 != text:
        per_file[f] = per_file.get(f, 0) + 1
        if APPLY:
            open(f, "w").write(n2)
print("files touched:", len(per_file))
for f in sorted(per_file):
    print("  ", f)
unused = [o for o in hex_map if o not in used]
print(f"old digests never found in the tree: {len(unused)} of {len(hex_map)}")
# which keys/fields were they?
rev = {}
for key in old:
    for field in FIELDS:
        rev.setdefault(old[key][field], []).append((key, field))
for o in unused[:40]:
    print("   unused:", o[:16], rev.get(o, [])[:3])
print("applied" if APPLY else "dry run")
