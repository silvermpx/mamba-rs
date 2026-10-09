#!/usr/bin/env python3
"""Float64 oracle for the official Mamba-2 checkpoints.

Runs a state-spaces/mamba2-* checkpoint in float64 on the CPU with numpy and
writes goldens in the format `m3_oracle.py` writes. The math is transcribed
from the reference at state-spaces/mamba e9594ce, not from mamba-rs:

- the mixer from `Mamba2.step` (mamba_ssm/modules/mamba2.py) and
  `selective_state_update` (mamba_ssm/ops/triton/selective_state_update.py):
  conv over the last d_conv inputs plus bias, SiLU, dt = softplus(dt +
  dt_bias) clamped to dt_limit, dA = exp(dt A) with A = -exp(A_log),
  state = state dA + dt B x, y = state C + D x;
- the gated norm from `RMSNormGated` (mamba_ssm/ops/triton/layernorm_gated.py)
  with group_size = d_ssm / ngroups: with norm_before_gate false the gate
  comes first, rmsnorm(y * silu(z)) * w per group;
- the block from mamba_ssm/modules/block.py and the model from
  mamba_ssm/models/mixer_seq_simple.py.

The checkpoint is read with the restricted unpickler in m3_oracle.py.

Usage:
    m2_oracle.py --model DIR --tokenizer tokenizer.json --out GOLDEN_DIR
                 [--prompt TEXT | --prompt-file FILE] [--greedy N] [--bos ID]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_oracle import load_state_dict, logsumexp, rms_norm, sha256, silu, softplus  # noqa: E402


class Mamba2:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        ssm = cfg.get("ssm_cfg", {})
        if ssm.get("layer") != "Mamba2":
            raise ValueError("this oracle runs Mamba-2 checkpoints")
        if cfg.get("attn_layer_idx"):
            raise ValueError("hybrid attention layers are not modelled")
        if cfg.get("d_intermediate", 0):
            raise ValueError("an MLP after the mixer is not modelled here")
        self.d_model = cfg["d_model"]
        self.n_layer = cfg["n_layer"]
        self.d_state = ssm.get("d_state", 128)
        self.d_conv = ssm.get("d_conv", 4)
        self.expand = ssm.get("expand", 2)
        self.headdim = ssm.get("headdim", 64)
        self.ngroups = ssm.get("ngroups", 1)
        self.d_inner = self.expand * self.d_model
        self.d_ssm = ssm.get("d_ssm") or self.d_inner
        self.nheads = self.d_ssm // self.headdim
        self.rmsnorm = ssm.get("rmsnorm", True)
        self.norm_before_gate = ssm.get("norm_before_gate", False)
        self.d_has_hdim = ssm.get("D_has_hdim", False)
        lo, hi = ssm.get("dt_limit", (0.0, float("inf")))
        self.dt_limit = (float(lo), float(hi))
        self.eps = cfg.get("norm_epsilon", 1e-5)
        sd = load_state_dict(os.path.join(model_dir, "pytorch_model.bin"))
        self.emb = sd["backbone.embedding.weight"]
        self.norm_f = sd["backbone.norm_f.weight"]
        head = sd.get("lm_head.weight")
        if head is not None and not np.array_equal(head, self.emb):
            raise ValueError("lm_head is not tied to the embedding")
        self.layers = []
        for i in range(self.n_layer):
            p = f"backbone.layers.{i}."
            self.layers.append({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
        # The padded rows past the tokenizer's vocabulary are never tokens;
        # the logits cover the real vocabulary only.
        self.vocab = cfg["vocab_size"]

    def init_state(self):
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        return [
            {
                "conv": np.zeros((conv_dim, self.d_conv)),
                "ssm": np.zeros((self.nheads, self.headdim, self.d_state)),
            }
            for _ in range(self.n_layer)
        ]

    def mixer_step(self, L, u, st):
        """One token through one Mamba-2 mixer (the step reference)."""
        H, P, N, G = self.nheads, self.headdim, self.d_state, self.ngroups
        d_ssm = self.d_ssm
        proj = u @ L["mixer.in_proj.weight"].T
        if "mixer.in_proj.bias" in L:
            proj = proj + L["mixer.in_proj.bias"]
        d_mlp = (proj.shape[0] - 2 * d_ssm - 2 * G * N - H) // 2
        o = 0
        z0 = proj[o:o + d_mlp]; o += d_mlp
        x0 = proj[o:o + d_mlp]; o += d_mlp
        z = proj[o:o + d_ssm]; o += d_ssm
        xbc = proj[o:o + d_ssm + 2 * G * N]; o += d_ssm + 2 * G * N
        dt = proj[o:o + H]; o += H
        assert o == proj.shape[0]

        # Conv step: roll the window, append this input, weight and sum.
        st["conv"] = np.roll(st["conv"], -1, axis=-1)
        st["conv"][:, -1] = xbc
        w = L["mixer.conv1d.weight"][:, 0, :]
        xbc = np.sum(st["conv"] * w, axis=-1)
        if "mixer.conv1d.bias" in L:
            xbc = xbc + L["mixer.conv1d.bias"]
        xbc = silu(xbc)
        x = xbc[:d_ssm].reshape(H, P)
        B = xbc[d_ssm:d_ssm + G * N].reshape(G, N)
        C = xbc[d_ssm + G * N:].reshape(G, N)

        A = -np.exp(L["mixer.A_log"])
        dt = softplus(dt + L["mixer.dt_bias"])
        dt = np.clip(dt, self.dt_limit[0], self.dt_limit[1])
        dA = np.exp(dt * A)
        rep = H // G
        Bh = np.repeat(B, rep, axis=0)
        Ch = np.repeat(C, rep, axis=0)
        ssm = st["ssm"] * dA[:, None, None] + (dt[:, None, None] * x[:, :, None] * Bh[:, None, :])
        st["ssm"] = ssm
        y = np.einsum("hpn,hn->hp", ssm, Ch)
        D = L["mixer.D"]
        y = y + (D.reshape(H, P) if self.d_has_hdim else D[:, None]) * x
        y = y.reshape(d_ssm)

        if self.rmsnorm:
            w_n = L["mixer.norm.weight"]
            gs = d_ssm // G
            out = np.empty(d_ssm)
            for g in range(G):
                sl = slice(g * gs, (g + 1) * gs)
                if self.norm_before_gate:
                    out[sl] = rms_norm(y[sl], w_n[sl], 1e-5) * silu(z[sl])
                else:
                    out[sl] = rms_norm(y[sl] * silu(z[sl]), w_n[sl], 1e-5)
            y = out
        else:
            y = y * silu(z)
        if d_mlp > 0:
            y = np.concatenate([silu(z0) * x0, y])
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
    model = Mamba2(args.model)
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
        "weights_sha256": sha256(os.path.join(args.model, "pytorch_model.bin")),
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
