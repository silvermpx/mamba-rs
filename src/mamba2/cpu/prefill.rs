//! Mamba-2 prefill: a prompt window through the backbone in one pass,
//! carrying the recurrent state in and out.
//!
//! The projections run as GEMMs over every row of the window, the conv
//! walks the window token by token with the decode's function, and the
//! heads scan in parallel, each over the whole window with the decode's
//! per-token scan. The BLAS-free GEMM adds every output in ascending k
//! whatever the number of rows, so in the default build the prefill gives
//! the decode chain's bits, for either [`PrefillMode`].

use rayon::prelude::*;

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::{Mamba2LayerWeights, Mamba2Weights};
use super::core::{
    DSkip, HeadScan, block_norm_row, conv_hist_push, conv_silu_token, dt_da, gate_token, head_scan,
};
use crate::mamba1::cpu::prefill::{PrefillMode, zip_rows};
use crate::mlp::cpu::{GatedMlpRowsScratch, gated_mlp_rows, mlp_norm_row};
use crate::ops::blas::{sgemm_forward, sgemm_forward_par};
use crate::ops::lane::LaneGemm;
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

/// Reusable prefill buffers, grown to the longest window seen.
pub struct Mamba2PrefillScratch {
    rows: usize,
    norm: Vec<f32>,
    proj: Vec<f32>,
    xbc: Vec<f32>,
    /// Head-major scan outputs `[nheads, T, headdim]`.
    y_hm: Vec<f32>,
    /// Row-major scan outputs `[T, d_ssm]`.
    y_rm: Vec<f32>,
    gated: Vec<f32>,
    out: Vec<f32>,
    mlp: GatedMlpRowsScratch,
    /// The precision of the windows (see `CpuOptions`).
    dtype: WeightDtype,
    gemm: LaneGemm,
}

impl Mamba2PrefillScratch {
    pub fn new() -> Self {
        Self {
            rows: 0,
            norm: Vec::new(),
            proj: Vec::new(),
            xbc: Vec::new(),
            y_hm: Vec::new(),
            y_rm: Vec::new(),
            gated: Vec::new(),
            out: Vec::new(),
            mlp: GatedMlpRowsScratch::default(),
            dtype: WeightDtype::F32,
            gemm: LaneGemm::default(),
        }
    }

    /// Run the windows in `dtype`: the weights must hold the bulk matrices
    /// rounded as that precision keeps them.
    pub(crate) fn set_dtype(&mut self, dtype: WeightDtype) {
        self.dtype = dtype;
        self.mlp.dtype = dtype;
    }

    fn ensure(&mut self, cfg: &Mamba2Config, rows: usize) {
        if rows <= self.rows {
            return;
        }
        let dm = cfg.d_model;
        self.norm.resize(rows * dm, 0.0);
        self.proj.resize(rows * cfg.in_proj_out_dim(), 0.0);
        self.xbc.resize(rows * cfg.conv_dim(), 0.0);
        self.y_hm.resize(rows * cfg.ssm_dim(), 0.0);
        self.y_rm.resize(rows * cfg.ssm_dim(), 0.0);
        self.gated.resize(rows * cfg.d_inner(), 0.0);
        self.out.resize(rows * dm, 0.0);
        self.rows = rows;
    }
}

impl Default for Mamba2PrefillScratch {
    fn default() -> Self {
        Self::new()
    }
}

type Gemm = fn(&mut [f32], &[f32], &[f32], Option<&[f32]>, usize, usize, usize);

fn gemm_for(mode: PrefillMode) -> Gemm {
    match mode {
        PrefillMode::Single => sgemm_forward,
        PrefillMode::Parallel => sgemm_forward_par,
    }
}

fn bias(b: &[f32]) -> Option<&[f32]> {
    if b.is_empty() { None } else { Some(b) }
}

/// The input projection's precision: f32 in the half lanes, as the GPU
/// engine keeps it, TF32 operands in the TF32 lane.
fn input_proj_dtype(dtype: WeightDtype) -> WeightDtype {
    if dtype == WeightDtype::Tf32 {
        dtype
    } else {
        WeightDtype::F32
    }
}

/// One Mamba-2 mixer over a window of `t_len` rows of `temporal`
/// (`[t_len, d_model]`), residual included, advancing `state`.
fn mamba2_layer_prefill(
    temporal: &mut [f32],
    t_len: usize,
    (lw, state): (&Mamba2LayerWeights, &mut Mamba2LayerState),
    s: &mut Mamba2PrefillScratch,
    (cfg, mode): (&Mamba2Config, PrefillMode),
) {
    let (dm, di, ip, cd) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.conv_dim(),
    );
    let (d_mlp, d_ssm, n, p_len, g_n, nh) = (
        cfg.d_mlp(),
        cfg.ssm_dim(),
        cfg.d_state,
        cfg.headdim,
        cfg.ngroups,
        cfg.nheads(),
    );
    let lane = (gemm_for(mode), s.dtype);

    zip_rows(
        mode,
        (&mut s.norm[..t_len * dm], dm),
        (&temporal[..t_len * dm], dm),
        |d, x| block_norm_row(d, x, (&lw.norm_weight, &lw.norm_bias), cfg.rms_norm_eps),
    );
    store(s.dtype, &mut s.norm[..t_len * dm]);
    s.gemm.forward(
        lane,
        (&mut s.proj[..t_len * ip], &s.norm[..t_len * dm]),
        (&lw.in_proj_w, bias(&lw.in_proj_b)),
        (t_len, dm, ip),
    );
    // The conv walks the window in order through the carried history.
    let xbc_col = 2 * d_mlp + d_ssm;
    for (proj, xbc) in s.proj[..t_len * ip]
        .chunks_exact(ip)
        .zip(s.xbc.chunks_exact_mut(cd))
    {
        let raw = &proj[xbc_col..xbc_col + cd];
        conv_silu_token(xbc, raw, &state.conv_hist, lw);
        conv_hist_push(&mut state.conv_hist, raw);
    }
    store(s.dtype, &mut s.xbc[..t_len * cd]);

    // Each head scans the window on its own state.
    let (proj, xbc) = (&s.proj[..t_len * ip], &s.xbc[..t_len * cd]);
    let run_head = |(h, (st, y_h)): (usize, (&mut [f32], &mut [f32]))| {
        let g = h / (nh / g_n);
        for (t, y_t) in y_h.chunks_exact_mut(p_len).enumerate() {
            let row = &xbc[t * cd..(t + 1) * cd];
            let (dt, da) = dt_da(
                proj[t * ip + xbc_col + cd + h],
                lw.dt_bias[h],
                lw.a_log[h],
                cfg.dt_limit,
            );
            head_scan(HeadScan {
                state: &mut *st,
                x: &row[h * p_len..(h + 1) * p_len],
                b: &row[d_ssm + g * n..d_ssm + (g + 1) * n],
                c: &row[d_ssm + (g_n + g) * n..d_ssm + (g_n + g + 1) * n],
                dt,
                da,
                d: if cfg.d_has_hdim {
                    DSkip::Channel(&lw.d_param[h * p_len..(h + 1) * p_len])
                } else {
                    DSkip::Head(lw.d_param[h])
                },
                y: y_t,
            });
        }
    };
    let work = state
        .ssm
        .chunks_exact_mut(n * p_len)
        .zip(s.y_hm[..t_len * d_ssm].chunks_exact_mut(t_len * p_len))
        .enumerate();
    match mode {
        PrefillMode::Single => work.for_each(run_head),
        PrefillMode::Parallel => work.collect::<Vec<_>>().into_par_iter().for_each(run_head),
    }
    for (h, y_h) in s.y_hm[..t_len * d_ssm]
        .chunks_exact(t_len * p_len)
        .enumerate()
    {
        for (t, y_t) in y_h.chunks_exact(p_len).enumerate() {
            s.y_rm[t * d_ssm + h * p_len..t * d_ssm + (h + 1) * p_len].copy_from_slice(y_t);
        }
    }

    // Gate rows, out_proj, residual.
    let y_rm = &s.y_rm[..t_len * d_ssm];
    let gate_row = |(t, g): (usize, &mut [f32])| {
        gate_token(
            g,
            &y_rm[t * d_ssm..(t + 1) * d_ssm],
            &proj[t * ip..(t + 1) * ip],
            lw,
            cfg,
        );
    };
    let gated = s.gated[..t_len * di].chunks_exact_mut(di).enumerate();
    match mode {
        PrefillMode::Single => gated.for_each(gate_row),
        PrefillMode::Parallel => gated.collect::<Vec<_>>().into_par_iter().for_each(gate_row),
    }
    store(s.dtype, &mut s.gated[..t_len * di]);
    s.gemm.forward(
        lane,
        (&mut s.out[..t_len * dm], &s.gated[..t_len * di]),
        (&lw.out_proj_w, bias(&lw.out_proj_b)),
        (t_len, di, dm),
    );
    for (t, &o) in temporal[..t_len * dm].iter_mut().zip(&s.out[..t_len * dm]) {
        *t += o;
    }
}

/// A prompt window through the whole Mamba-2 backbone, the decode's math on
/// every row. `input_flat` is `[seq_len, input_dim]`; `temporal_out`
/// (`[seq_len, d_model]`) receives every position's final-norm output, and
/// `states` carry in and out.
pub fn forward_mamba2_backbone_prefill(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    scratch: &mut Mamba2PrefillScratch,
    (cfg, seq_len): (&Mamba2Config, usize),
) {
    forward_mamba2_backbone_prefill_mode(
        temporal_out,
        input_flat,
        w,
        states,
        scratch,
        (cfg, seq_len, PrefillMode::Parallel),
    );
}

/// [`forward_mamba2_backbone_prefill`] with an explicit execution mode;
/// both modes give the same bits.
pub fn forward_mamba2_backbone_prefill_mode(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    scratch: &mut Mamba2PrefillScratch,
    (cfg, seq_len, mode): (&Mamba2Config, usize, PrefillMode),
) {
    let dm = cfg.d_model;
    if seq_len == 0 {
        return;
    }
    scratch.ensure(cfg, seq_len);
    let out = &mut temporal_out[..seq_len * dm];
    if w.input_proj_w.is_empty() {
        out.copy_from_slice(&input_flat[..seq_len * dm]);
    } else {
        let input_dim = input_flat.len() / seq_len;
        scratch.gemm.forward(
            (gemm_for(mode), input_proj_dtype(scratch.dtype)),
            (out, &input_flat[..seq_len * input_dim]),
            (&w.input_proj_w, bias(&w.input_proj_b)),
            (seq_len, input_dim, dm),
        );
    }
    for (l, (lw, st)) in w.layers.iter().zip(states.iter_mut()).enumerate() {
        mamba2_layer_prefill(out, seq_len, (lw, st), scratch, (cfg, mode));
        if let Some(mlp) = w.mlp.get(l) {
            let eps = cfg.rms_norm_eps;
            gated_mlp_rows(
                out,
                seq_len,
                mlp,
                &mut scratch.mlp,
                gemm_for(mode),
                |dst, src| {
                    zip_rows(mode, (dst, dm), (src, dm), |d, x| {
                        mlp_norm_row(d, x, mlp, eps);
                    });
                },
            );
        }
    }
    let norm = &mut scratch.norm[..seq_len * dm];
    norm.copy_from_slice(out);
    zip_rows(mode, (out, dm), (norm, dm), |d, x| {
        block_norm_row(d, x, (&w.norm_f_weight, &w.norm_f_bias), cfg.rms_norm_eps);
    });
}
