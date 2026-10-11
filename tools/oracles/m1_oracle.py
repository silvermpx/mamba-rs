#!/usr/bin/env python3
"""Float64 oracle for the official Mamba-1 checkpoints.

Runs a state-spaces/mamba-*-hf checkpoint (the transformers layout, one
safetensors file) in float64 on the CPU with numpy and writes goldens in
the format `m3_oracle.py` writes. The math is transcribed from the
reference at state-spaces/mamba e9594ce, not from mamba-rs:

- the mixer from `Mamba.step` (mamba_ssm/modules/mamba_simple.py): x and z
  from in_proj, the conv over the last d_conv inputs plus bias, SiLU,
  dt, B and C from x_proj, dt = softplus(dt_proj(dt) + dt_bias),
  dA = exp(dt A) with A = -exp(A_log), state = state dA + x dt B,
  y = state C + D x, gated by SiLU(z), then out_proj;
- the block from mamba_ssm/modules/block.py and the model from
  mamba_ssm/models/mixer_seq_simple.py: the residual stream adds each
  mixer's output to its own input, every mixer reads the RMS-normalized
  stream, and norm_f closes the stack; the head is tied to the embedding.

Usage:
    m1_oracle.py --model DIR --tokenizer tokenizer.json --out GOLDEN_DIR
                 [--prompt TEXT | --prompt-file FILE] [--greedy N] [--bos ID]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_oracle import logsumexp, rms_norm, sha256, silu, softplus  # noqa: E402

DTYPES = {"F32": "<f4", "F64": "<f8", "F16": "<f2"}


def load_safetensors(path):
    """Every tensor of one safetensors file as float64 arrays."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"{name}: dtype {info['dtype']} is not read here")
        start, end = info["data_offsets"]
        raw = np.frombuffer(data[start:end], dtype=dtype)
        tensors[name] = raw.astype(np.float64).reshape(info["shape"])
    return tensors


class Mamba1:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("model_type") != "mamba":
            raise ValueError("this oracle runs Mamba-1 checkpoints")
        self.d_model = cfg["hidden_size"]
        self.n_layer = cfg["num_hidden_layers"]
        self.d_state = cfg["state_size"]
        self.d_conv = cfg["conv_kernel"]
        self.d_inner = cfg["intermediate_size"]
        self.dt_rank = cfg["time_step_rank"]
        self.eps = cfg.get("layer_norm_epsilon", 1e-5)
        if not cfg.get("rms_norm", True):
            raise ValueError("LayerNorm models are not modelled")
        sd = load_safetensors(os.path.join(model_dir, "model.safetensors"))
        self.emb = sd["backbone.embeddings.weight"]
        self.norm_f = sd["backbone.norm_f.weight"]
        head = sd.get("lm_head.weight")
        if head is not None and not np.array_equal(head, self.emb):
            raise ValueError("lm_head is not tied to the embedding")
        self.layers = []
        for i in range(self.n_layer):
            p = f"backbone.layers.{i}."
            self.layers.append({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
        self.vocab = cfg["vocab_size"]

    def init_state(self):
        return [
            {
                "conv": np.zeros((self.d_inner, self.d_conv)),
                "ssm": np.zeros((self.d_inner, self.d_state)),
            }
            for _ in range(self.n_layer)
        ]

    def mixer_step(self, L, u, st):
        """One token through one Mamba-1 mixer (the step reference)."""
        xz = u @ L["mixer.in_proj.weight"].T
        if "mixer.in_proj.bias" in L:
            xz = xz + L["mixer.in_proj.bias"]
        x, z = xz[: self.d_inner], xz[self.d_inner:]

        # Conv step: roll the window, append this input, weight and sum.
        st["conv"] = np.roll(st["conv"], -1, axis=-1)
        st["conv"][:, -1] = x
        w = L["mixer.conv1d.weight"][:, 0, :]
        x = np.sum(st["conv"] * w, axis=-1)
        if "mixer.conv1d.bias" in L:
            x = x + L["mixer.conv1d.bias"]
        x = silu(x)

        x_db = x @ L["mixer.x_proj.weight"].T
        dt = x_db[: self.dt_rank]
        B = x_db[self.dt_rank:self.dt_rank + self.d_state]
        C = x_db[self.dt_rank + self.d_state:]
        dt = dt @ L["mixer.dt_proj.weight"].T
        dt = softplus(dt + L["mixer.dt_proj.bias"])
        A = -np.exp(L["mixer.A_log"])
        dA = np.exp(dt[:, None] * A)
        dB = dt[:, None] * B[None, :]
        st["ssm"] = st["ssm"] * dA + x[:, None] * dB
        y = st["ssm"] @ C + L["mixer.D"] * x
        y = y * silu(z)
        out = y @ L["mixer.out_proj.weight"].T
        if "mixer.out_proj.bias" in L:
            out = out + L["mixer.out_proj.bias"]
        return out

    def step(self, token, state):
        x = self.emb[token].copy()
        for L, st in zip(self.layers, state):
            x = x + self.mixer_step(L, rms_norm(x, L["norm.weight"], self.eps), st)
        hidden = rms_norm(x, self.norm_f, self.eps)
        return hidden @ self.emb[: self.vocab].T, hidden


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    ap.add_argument("--prompt-file")
    ap.add_argument("--greedy", type=int, default=32)
    ap.add_argument("--full-rows", type=int, default=16)
    ap.add_argument("--bos", type=int, default=None,
                    help="prepend this token id (the NeoX-trained models take none)")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            args.prompt = f.read()
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    if args.bos is not None:
        ids = [args.bos] + ids
    model = Mamba1(args.model)
    state = model.init_state()

    T = len(ids)
    lse, target_lp, top_ids, top_vals, rows = [], [], [], [], []
    for t, token in enumerate(ids):
        logits, _ = model.step(token, state)
        lse.append(logsumexp(logits))
        if t + 1 < T:
            target_lp.append(logits[ids[t + 1]] - lse[-1])
        order = np.argsort(-logits)[:32]
        top_ids.append(order)
        top_vals.append(logits[order])
        if t >= T - args.full_rows:
            rows.append(logits.copy())

    greedy, margins = [], []
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
        "weights_sha256": sha256(os.path.join(args.model, "model.safetensors")),
        "tokenizer_sha256": sha256(args.tokenizer),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "numpy": np.__version__,
        "prompt": args.prompt,
        "reference": "state-spaces/mamba e9594ce",
    }
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
    nll = -float(np.mean(target_lp)) if target_lp else float("nan")
    print(f"T={T} nll={nll:.6f} greedy={tok.decode(greedy)!r}")
    print(json.dumps(meta))


if __name__ == "__main__":
    sys.exit(main())
