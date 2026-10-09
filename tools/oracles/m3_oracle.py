#!/usr/bin/env python3
"""Float64 oracle for the official Mamba-3 SISO checkpoints.

Runs a state-spaces/mamba3-siso-* checkpoint in float64 on the CPU with
numpy and writes the goldens the Rust parity tests compare against. The
math is transcribed from the reference, not from mamba-rs:

- the mixer from `mamba3_siso_step_ref` (tests/ops/triton/test_mamba3_siso.py)
  and `Mamba3.forward` (mamba_ssm/modules/mamba3.py) at state-spaces/mamba
  e9594ce;
- the block from mamba_ssm/modules/block.py and the gated MLP from
  mamba_ssm/modules/mlp.py;
- the model from mamba_ssm/models/mixer_seq_simple.py.

The upstream kernels are not used as goldens: they run bf16 operands and
approximate transcendentals and are tested to a few percent.

The checkpoint is read without torch. `pytorch_model.bin` is a zip
archive whose `data.pkl` is unpickled with a whitelist of exactly the
globals a torch state dict needs, and the tensor bytes are read from
the archive's `data/<key>` entries.

Usage:
    m3_oracle.py --model DIR --tokenizer tokenizer.json --out GOLDEN_DIR
                 [--prompt TEXT] [--greedy N] [--a-activation heavy_tail|softplus]
"""

import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import sys
import zipfile

import numpy as np


# ---------------------------------------------------------------------------
# Reading pytorch_model.bin without torch.
# ---------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "BFloat16Storage": ("bf16", 2),
    "HalfStorage": (np.float16, 2),
    "FloatStorage": (np.float32, 4),
}


class _StorageType:
    def __init__(self, name):
        self.name = name


class _Storage:
    def __init__(self, archive, prefix, key, kind):
        self.archive = archive
        self.prefix = prefix
        self.key = key
        self.kind = kind


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    kind, width = _STORAGE_DTYPES[storage.kind]
    raw = storage.archive.read(f"{storage.prefix}/data/{storage.key}")
    numel = 1
    for s in size:
        numel *= s
    if kind == "bf16":
        words = np.frombuffer(raw, dtype=np.uint16)
        flat = (words.astype(np.uint32) << 16).view(np.float32)
    else:
        flat = np.frombuffer(raw, dtype=kind)
    # Only contiguous row-major tensors occur in these checkpoints; any
    # other stride is refused rather than guessed.
    expect = []
    acc = 1
    for s in reversed(size):
        expect.append(acc)
        acc *= s
    expect = tuple(reversed(expect))
    if tuple(stride) != expect and numel > 1:
        raise ValueError(f"non-contiguous tensor (size {size}, stride {stride})")
    if offset + numel > flat.size:
        raise ValueError("tensor runs past its storage")
    return flat[offset:offset + numel].astype(np.float64).reshape(size)


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, fh, archive, prefix):
        super().__init__(fh)
        self.archive = archive
        self.prefix = prefix

    # The checkpoint is untrusted input. Every global the pickle may name is
    # on this list and none of them runs code beyond building a dict or
    # slicing an array, so a hostile data.pkl can at worst fail to load.
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")

    def persistent_load(self, pid):
        tag, storage_type, key, _location, _numel = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"unexpected persistent id {tag}")
        return _Storage(self.archive, self.prefix, key, storage_type.name)


def load_state_dict(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise ValueError("expected one data.pkl in the archive")
        prefix = pkl[0][: -len("/data.pkl")]
        with archive.open(pkl[0]) as fh:
            return _RestrictedUnpickler(fh, archive, prefix).load()


# ---------------------------------------------------------------------------
# The model, in float64.
# ---------------------------------------------------------------------------

def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def heavy_tail(x):
    neg = np.minimum(x, 0.0)
    pos = np.maximum(x, 0.0)
    return pos + 1.0 / (1.0 - neg)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rotate_interleaved(t, cos, sin):
    """Rotate pairs (2a, 2a+1) for a < len(cos); the rest pass through."""
    out = t.copy()
    na = cos.shape[-1]
    t0 = t[..., 0:2 * na:2]
    t1 = t[..., 1:2 * na:2]
    out[..., 0:2 * na:2] = t0 * cos - t1 * sin
    out[..., 1:2 * na:2] = t0 * sin + t1 * cos
    return out


class Mamba3Siso:
    def __init__(self, model_dir, a_activation):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        ssm = cfg["ssm_cfg"]
        if ssm.get("layer") != "Mamba3" or ssm.get("is_mimo"):
            raise ValueError("this oracle runs Mamba-3 SISO checkpoints")
        if cfg.get("attn_layer_idx"):
            raise ValueError("hybrid attention layers are not modelled")
        self.cfg = cfg
        self.d_model = cfg["d_model"]
        self.n_layer = cfg["n_layer"]
        self.d_state = ssm.get("d_state", 128)
        self.expand = ssm.get("expand", 2)
        self.headdim = ssm.get("headdim", 64)
        self.ngroups = ssm.get("ngroups", 1)
        self.a_floor = ssm.get("A_floor", 1e-4)
        rope_fraction = ssm.get("rope_fraction", 0.5)
        if ssm.get("is_outproj_norm", False):
            raise ValueError("is_outproj_norm checkpoints are not modelled here")
        self.eps = cfg.get("norm_epsilon", 1e-5)
        self.d_inner = self.expand * self.d_model
        self.nheads = self.d_inner // self.headdim
        split = int(self.d_state * rope_fraction)
        if split % 2:
            split -= 1
        self.n_angles = split // 2
        self.a_activation = a_activation
        sd = load_state_dict(os.path.join(model_dir, "pytorch_model.bin"))
        self.emb = sd["backbone.embedding.weight"]
        self.norm_f = sd["backbone.norm_f.weight"]
        head = sd.get("lm_head.weight")
        if head is not None and not np.array_equal(head, self.emb):
            raise ValueError("lm_head is not tied to the embedding")
        self.layers = []
        for i in range(self.n_layer):
            p = f"backbone.layers.{i}."
            layer = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            self.layers.append(layer)
        self.vocab = self.emb.shape[0]

    def init_state(self):
        H, P, N = self.nheads, self.headdim, self.d_state
        return [
            {
                "angle": np.zeros((H, self.n_angles)),
                "ssm": np.zeros((H, P, N)),
                "k": np.zeros((H, N)),
                "v": np.zeros((H, P)),
            }
            for _ in range(self.n_layer)
        ]

    def mixer_step(self, L, u, st):
        """One token through one Mamba-3 SISO mixer (the step reference)."""
        di, H, P, N, G = self.d_inner, self.nheads, self.headdim, self.d_state, self.ngroups
        proj = u @ L["mixer.in_proj.weight"].T
        o = 0
        z = proj[o:o + di].reshape(H, P); o += di
        x = proj[o:o + di].reshape(H, P); o += di
        B = proj[o:o + G * N].reshape(G, N); o += G * N
        C = proj[o:o + G * N].reshape(G, N); o += G * N
        dd_dt = proj[o:o + H]; o += H
        dd_a = proj[o:o + H]; o += H
        trap_raw = proj[o:o + H]; o += H
        ang_raw = proj[o:o + self.n_angles]; o += self.n_angles
        assert o == proj.shape[0]

        if self.a_activation == "heavy_tail":
            a = -heavy_tail(dd_a)
        else:
            a = -softplus(dd_a)
        a = np.minimum(a, -self.a_floor)
        dt = softplus(dd_dt + L["mixer.dt_bias"])
        adt = a * dt

        B = rms_norm(B, L["mixer.B_norm.weight"], 1e-5)
        C = rms_norm(C, L["mixer.C_norm.weight"], 1e-5)
        rep = H // G
        Bh = np.repeat(B, rep, axis=0)
        Ch = np.repeat(C, rep, axis=0)
        q = Ch + L["mixer.C_bias"].reshape(H, N)
        k = Bh + L["mixer.B_bias"].reshape(H, N)

        angles = np.tanh(ang_raw) * math.pi
        st["angle"] = st["angle"] + angles[None, :] * dt[:, None]
        st["angle"] = st["angle"] - 2 * math.pi * np.floor(st["angle"] / (2 * math.pi))
        cos, sin = np.cos(st["angle"]), np.sin(st["angle"])
        q_rot = rotate_interleaved(q, cos, sin)
        k_rot = rotate_interleaved(k, cos, sin)

        trap = 1.0 / (1.0 + np.exp(-trap_raw))
        alpha = np.exp(adt)
        beta = (1.0 - trap) * dt * alpha
        gamma = trap * dt
        ssm = alpha[:, None, None] * st["ssm"]
        ssm = ssm + beta[:, None, None] * (st["k"][:, None, :] * st["v"][:, :, None])
        ssm = ssm + gamma[:, None, None] * (k_rot[:, None, :] * x[:, :, None])
        y = np.einsum("hpn,hn->hp", ssm, q_rot)
        y = y + L["mixer.D"][:, None] * x
        y = y * silu(z)
        st["ssm"], st["k"], st["v"] = ssm, k_rot, x
        return y.reshape(di) @ L["mixer.out_proj.weight"].T

    def mlp(self, L, h):
        up = h @ L["mlp.fc1.weight"].T
        hidden = up.shape[0] // 2
        val, gate = up[:hidden], up[hidden:]
        return (val * silu(gate)) @ L["mlp.fc2.weight"].T

    def step(self, token, state):
        """One token through the whole model; returns the logits."""
        x = self.emb[token].copy()
        for L, st in zip(self.layers, state):
            x = x + self.mixer_step(L, rms_norm(x, L["norm.weight"], self.eps), st)
            if "mlp.fc1.weight" in L:
                x = x + self.mlp(L, rms_norm(x, L["norm2.weight"], self.eps))
        hidden = rms_norm(x, self.norm_f, self.eps)
        return hidden @ self.emb.T, hidden


def logsumexp(v):
    m = np.max(v)
    return m + math.log(np.sum(np.exp(v - m)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    ap.add_argument("--prompt-file", help="read the prompt from this file instead")
    ap.add_argument("--greedy", type=int, default=32)
    ap.add_argument("--a-activation", choices=["heavy_tail", "softplus"], default="heavy_tail")
    ap.add_argument("--full-rows", type=int, default=16)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            args.prompt = f.read()
    ids = [128000] + tok.encode(args.prompt, add_special_tokens=False).ids
    model = Mamba3Siso(args.model, args.a_activation)
    state = model.init_state()

    T = len(ids)
    lse, target_lp, top_ids, top_vals, rows = [], [], [], [], []
    for t, token in enumerate(ids):
        logits, hidden = model.step(token, state)
        lse.append(logsumexp(logits))
        if t + 1 < T:
            target_lp.append(logits[ids[t + 1]] - lse[-1])
        order = np.argsort(-logits)[:32]
        top_ids.append(order)
        top_vals.append(logits[order])
        if t >= T - args.full_rows:
            rows.append(logits.copy())

    greedy, margins = [], []
    logits = None
    next_tok = int(top_ids[-1][0])
    margins.append(float(top_vals[-1][0] - top_vals[-1][1]))
    for _ in range(args.greedy):
        greedy.append(next_tok)
        logits, _ = model.step(next_tok, state)
        order = np.argsort(-logits)[:2]
        margins.append(float(logits[order[0]] - logits[order[1]]))
        next_tok = int(order[0])

    meta = {
        "model_dir": os.path.abspath(args.model),
        "weights_sha256": sha256(os.path.join(args.model, "pytorch_model.bin")),
        "tokenizer_sha256": sha256(args.tokenizer),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "numpy": np.__version__,
        "prompt": args.prompt,
        "a_activation": args.a_activation,
        "reference": "state-spaces/mamba e9594ce",
    }
    # A directory of plain files the Rust tests read without a numpy
    # dependency: golden.json for the scalars and token lists, raw
    # little-endian float64 for the last rows' full logits.
    os.makedirs(args.out, exist_ok=True)
    golden = {
        "meta": meta,
        "vocab_size": int(model.vocab),
        "ids": [int(i) for i in ids],
        "logsumexp": [float(v) for v in lse],
        "target_logprob": [float(v) for v in target_lp],
        "top_ids": [[int(i) for i in row] for row in top_ids],
        "top_logits": [[float(v) for v in row] for row in top_vals],
        "last_rows_from": T - len(rows),
        "greedy": [int(t) for t in greedy],
        "greedy_margin": margins,
    }
    with open(os.path.join(args.out, "golden.json"), "w") as f:
        json.dump(golden, f)
    np.asarray(rows, dtype="<f8").tofile(os.path.join(args.out, "last_logits.f64"))
    nll = -float(np.mean(target_lp))
    print(f"T={T} nll={nll:.6f} greedy={tok.decode(greedy)!r}")
    print(json.dumps(meta))


if __name__ == "__main__":
    sys.exit(main())
