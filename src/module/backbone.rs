use crate::mamba1::config::MambaConfig;
use crate::mamba1::cpu::inference::{
    MambaPackedLinears, MambaStepScratch, mamba_step, mamba_step_no_proj,
    mamba_step_no_proj_packed, mamba_step_packed,
};
use crate::mamba1::cpu::prefill::{
    MambaPrefillScratch, PrefillMode, forward_mamba_backbone_prefill_mode,
};
use crate::mamba1::state::MambaState;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::ops::dims::MambaDims;
use crate::precision::{CpuOptions, WeightDtype};

/// Mamba backbone for CPU inference: input projection, the layers and the
/// final norm. The recurrent state is owned by the caller, so several
/// sessions can share one set of weights.
///
/// ```rust
/// use mamba_rs::{MambaBackbone, MambaConfig};
///
/// let cfg = MambaConfig::default();
/// let backbone = MambaBackbone::init(cfg, 128, 42)?;
///
/// let mut state = backbone.alloc_state();
/// let mut scratch = backbone.alloc_scratch();
/// let mut output = vec![0.0f32; backbone.d_model()];
///
/// let input = vec![0.1f32; 128];
/// backbone.forward_step(&input, &mut output, &mut state, &mut scratch);
/// # Ok::<(), String>(())
/// ```
pub struct MambaBackbone {
    weights: MambaWeights,
    cfg: MambaConfig,
    input_dim: usize,
    identity_proj: bool,
    /// The projections packed for decode by [`Self::pack_for_decode`]: a
    /// copy of `weights`, dropped by every mutable access to them so a step
    /// never reads stale panels.
    packed: Option<MambaPackedLinears>,
    /// The precision the backbone computes in.
    dtype: WeightDtype,
}

impl MambaBackbone {
    /// Fresh weights with the reference init: Kaiming uniform projections,
    /// `A` in log space and the inverse-softplus dt bias (Gu and Dao,
    /// section 3.5). Errs on a configuration `validate` refuses.
    pub fn init(cfg: MambaConfig, input_dim: usize, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        Self::new(MambaWeights::init(&cfg, input_dim, seed), cfg, input_dim)
    }

    /// Wrap loaded weights for input rows `input_dim` wide, after checking
    /// every tensor against `cfg`. Weights without an input projection (an
    /// empty `input_proj_w`, as the checkpoint loader gives) read rows
    /// `d_model` wide unchanged, so `input_dim` is then `d_model`.
    ///
    /// Recomputes `a_neg = -exp(a_log)` for every layer: inference reads
    /// `a_neg` alone, and loaded weights may carry a stale one.
    pub fn new(weights: MambaWeights, cfg: MambaConfig, input_dim: usize) -> Result<Self, String> {
        Self::new_with(weights, cfg, input_dim, CpuOptions::default())
    }

    /// [`MambaBackbone::new`] in the precision `opts` names: the
    /// projections rounded once here as the GPU engine of that precision
    /// holds them, and the values between the kernels rounded where it
    /// stores them. [`MambaBackbone::weights`] then gives the rounded
    /// weights, the ones the backbone computes with.
    pub fn new_with(
        weights: MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        let mut backbone = if weights.input_proj_w.is_empty() {
            if input_dim != cfg.d_model {
                return Err(format!(
                    "weights without an input projection read rows d_model ({}) wide, \
                     not {input_dim}",
                    cfg.d_model
                ));
            }
            Self::without_projection(cfg, weights)?
        } else {
            Self::with_projection(cfg, weights, input_dim)?
        };
        backbone.weights.round_bulk(opts.dtype);
        backbone.dtype = opts.dtype;
        Ok(backbone)
    }

    /// The precision the backbone computes in.
    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    fn with_projection(
        cfg: MambaConfig,
        mut weights: MambaWeights,
        input_dim: usize,
    ) -> Result<Self, String> {
        weights.validate(&cfg, input_dim)?;
        for lw in &mut weights.layers {
            lw.compute_a_neg();
        }
        Ok(Self {
            weights,
            cfg,
            input_dim,
            identity_proj: false,
            packed: None,
            dtype: WeightDtype::F32,
        })
    }

    /// Weights without an input projection. The checks of `validate()`
    /// except the projection's, which it requires, and the `a_neg`
    /// refresh the SSM reads.
    fn without_projection(cfg: MambaConfig, mut weights: MambaWeights) -> Result<Self, String> {
        cfg.validate()?;
        if weights.layers.len() != cfg.n_layers {
            return Err(format!(
                "expected {} layers, got {}",
                cfg.n_layers,
                weights.layers.len()
            ));
        }
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dc = cfg.d_conv;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();
        let nb = cfg.block_norm.bias_len(d);
        for (i, lw) in weights.layers.iter().enumerate() {
            let check = |name: &str, actual: usize, expected: usize| -> Result<(), String> {
                if actual != expected {
                    return Err(format!(
                        "layer[{i}].{name}: expected {expected}, got {actual}"
                    ));
                }
                Ok(())
            };
            check("norm_weight", lw.norm_weight.len(), d)?;
            check("norm_bias", lw.norm_bias.len(), nb)?;
            check("in_proj_w", lw.in_proj_w.len(), d * 2 * di)?;
            check("in_proj_b", lw.in_proj_b.len(), cfg.in_proj_bias_len())?;
            check("conv1d_weight", lw.conv1d_weight.len(), di * dc)?;
            check("conv1d_bias", lw.conv1d_bias.len(), cfg.conv_bias_len())?;
            check("x_proj_w", lw.x_proj_w.len(), di * xd)?;
            check("dt_proj_w", lw.dt_proj_w.len(), dr * di)?;
            check("dt_proj_b", lw.dt_proj_b.len(), di)?;
            check("a_log", lw.a_log.len(), di * ds)?;
            check("d_param", lw.d_param.len(), di)?;
            check("out_proj_w", lw.out_proj_w.len(), di * d)?;
            check("out_proj_b", lw.out_proj_b.len(), cfg.out_proj_bias_len())?;
        }
        let hidden = cfg.mlp_hidden();
        let mlp_layers = if hidden > 0 { cfg.n_layers } else { 0 };
        if weights.mlp.len() != mlp_layers {
            return Err(format!(
                "mlp: expected {mlp_layers} layers, got {}",
                weights.mlp.len()
            ));
        }
        for (i, m) in weights.mlp.iter().enumerate() {
            m.validate(d, hidden, cfg.block_norm, &format!("mlp[{i}]"))?;
        }
        if weights.norm_f_weight.len() != d {
            return Err(format!(
                "norm_f_weight: expected {d}, got {}",
                weights.norm_f_weight.len()
            ));
        }
        if weights.norm_f_bias.len() != nb {
            return Err(format!(
                "norm_f_bias: expected {nb}, got {}",
                weights.norm_f_bias.len()
            ));
        }
        for lw in &mut weights.layers {
            lw.compute_a_neg();
        }
        Ok(Self {
            weights,
            cfg,
            input_dim: d,
            identity_proj: true,
            packed: None,
            dtype: WeightDtype::F32,
        })
    }

    /// The weights, consuming the backbone.
    pub fn into_weights(self) -> MambaWeights {
        self.weights
    }

    /// The weights.
    pub fn weights(&self) -> &MambaWeights {
        &self.weights
    }

    /// The weights, to change in place. Drops the packed projections; pack
    /// again with [`Self::pack_for_decode`]. In a reduced precision the
    /// backbone computes with the values written here as they are;
    /// [`Self::new_with`] rounds a set of weights as the GPU engine of that
    /// precision holds them.
    pub fn weights_mut(&mut self) -> &mut MambaWeights {
        self.packed = None;
        &mut self.weights
    }

    /// One layer's weights.
    pub fn layer(&self, index: usize) -> &MambaLayerWeights {
        &self.weights.layers[index]
    }

    /// One layer's weights, to change in place, as with
    /// [`Self::weights_mut`].
    pub fn layer_mut(&mut self, index: usize) -> &mut MambaLayerWeights {
        self.packed = None;
        &mut self.weights.layers[index]
    }

    /// Pack the projections in panels for [`Self::forward_step`]: the same
    /// outputs as the row-major product without a BLAS, faster on large
    /// layers, at the memory of a second copy of the projections. A mutable
    /// access to the weights drops the copy.
    pub fn pack_for_decode(&mut self) {
        self.packed = Some(MambaPackedLinears::pack(
            &self.weights,
            &self.cfg,
            self.input_dim,
        ));
    }

    /// Whether the projections are packed for decode.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    /// Total parameter count: every tensor the family's checkpoint holds.
    pub fn param_count(&self) -> usize {
        // A built backbone's weights always pass the checkpoint's checks.
        crate::mamba1::serialize::checkpoint_parts(&self.weights, &self.cfg, self.input_dim)
            .map_or(0, |(list, _)| list.element_count())
    }

    /// The model's configuration.
    pub fn config(&self) -> &MambaConfig {
        &self.cfg
    }

    /// Width of an input row.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Width of an output row.
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// One token: reads `input` (`input_dim` wide), writes `output`
    /// (`d_model` wide) and advances every layer's state. No allocation.
    ///
    /// # Panics
    ///
    /// For a bidirectional model, which has no decode step: it reads whole
    /// sequences with [`Self::forward_sequence`].
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut MambaState,
        scratch: &mut MambaScratch,
    ) {
        let scratch = &mut scratch.step;
        scratch.set_dtype(self.dtype);
        match (&self.packed, self.identity_proj) {
            (Some(p), true) => mamba_step_no_proj_packed(
                input,
                output,
                (&self.weights, p),
                &mut state.layers,
                scratch,
                &self.cfg,
            ),
            (Some(p), false) => mamba_step_packed(
                input,
                output,
                (&self.weights, p),
                &mut state.layers,
                scratch,
                &self.cfg,
                self.input_dim,
            ),
            (None, true) => mamba_step_no_proj(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
            ),
            (None, false) => mamba_step(
                input,
                output,
                &self.weights,
                &mut state.layers,
                scratch,
                &self.cfg,
                self.input_dim,
            ),
        }
    }

    /// A window of `seq_len` tokens in one pass: `inputs` is
    /// `[seq_len * input_dim]`, every position's output goes to `outputs`
    /// (`[seq_len * d_model]`) and the state carries in and out, so
    /// [`Self::forward_step`] continues the sequence. The window's matrix
    /// products run batched over its rows on the thread pool, with the same
    /// bits on any pool.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut MambaState,
        scratch: &mut MambaScratch,
        seq_len: usize,
    ) {
        self.forward_sequence_with(
            inputs,
            outputs,
            state,
            scratch,
            seq_len,
            PrefillMode::Parallel,
        );
    }

    /// [`Self::forward_sequence`] in an explicit execution mode, for the
    /// tests that hold the parallel window to the serial one bit for bit.
    #[doc(hidden)]
    pub fn forward_sequence_with(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut MambaState,
        scratch: &mut MambaScratch,
        seq_len: usize,
        mode: PrefillMode,
    ) {
        if seq_len == 0 {
            return;
        }
        let dims = MambaDims::from_config(&self.cfg, seq_len, self.input_dim);
        let prefill = scratch
            .prefill
            .get_or_insert_with(|| MambaPrefillScratch::new(&dims));
        prefill.set_dtype(self.dtype);
        forward_mamba_backbone_prefill_mode(
            outputs,
            inputs,
            &self.weights,
            state,
            prefill,
            &dims,
            mode,
        );
    }

    /// One token of `B` independent sequences with the same weights, each
    /// through [`Self::forward_step`]: `inputs` is `[B * input_dim]`,
    /// `outputs` `[B * d_model]`, one state and one scratch a sequence.
    pub fn forward_step_batch(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        states: &mut [MambaState],
        scratches: &mut [MambaScratch],
    ) {
        crate::ops::step_batch::step_batch_by(
            (inputs, outputs),
            (states, scratches),
            (self.cfg.d_model, self.input_dim),
            |inp, out, state, scratch| self.forward_step(inp, out, state, scratch),
        );
    }

    /// A zeroed recurrent state, the start of a sequence.
    pub fn alloc_state(&self) -> MambaState {
        MambaState::zeros(&self.cfg)
    }

    /// Buffers for single steps and for windows.
    pub fn alloc_scratch(&self) -> MambaScratch {
        MambaScratch {
            step: MambaStepScratch::new(&self.cfg),
            prefill: None,
        }
    }
}

/// The step and window buffers of one Mamba session. The window buffers
/// are sized on the first window and follow its length.
pub struct MambaScratch {
    step: MambaStepScratch,
    prefill: Option<MambaPrefillScratch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter count is every element of the checkpoint's tensors,
    /// the gated MLP's included.
    #[test]
    fn the_parameter_count_is_the_checkpoints() {
        for d_intermediate in [0, 300] {
            let cfg = MambaConfig {
                d_model: 64,
                d_state: 16,
                n_layers: 3,
                d_intermediate,
                ..MambaConfig::default()
            };
            let b = MambaBackbone::init(cfg, 24, 5).unwrap();
            let (list, _) =
                crate::mamba1::serialize::checkpoint_parts(b.weights(), b.config(), 24).unwrap();
            assert_eq!(b.n_layers(), 3);
            assert_eq!(b.param_count(), list.element_count());
        }
    }

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// Packed projections give the row-major step's bits over a chain of
    /// steps (without a BLAS), and a mutable access to the weights drops
    /// them, so a step never reads panels of weights that changed.
    #[test]
    fn packed_decode_keeps_the_bits_and_drops_on_mutation() {
        let cfg = MambaConfig {
            d_model: 96,
            n_layers: 2,
            ..MambaConfig::default()
        };
        let input_dim = 40;
        let plain = MambaBackbone::init(cfg, input_dim, 7).unwrap();
        let mut packed = MambaBackbone::init(cfg, input_dim, 7).unwrap();
        packed.pack_for_decode();
        assert!(packed.is_packed());
        let (mut s0, mut s1) = (plain.alloc_state(), packed.alloc_state());
        let (mut c0, mut c1) = (plain.alloc_scratch(), packed.alloc_scratch());
        let (mut o0, mut o1) = (vec![0f32; cfg.d_model], vec![0f32; cfg.d_model]);
        for t in 0..12u64 {
            let x = det(input_dim, 100 + t);
            plain.forward_step(&x, &mut o0, &mut s0, &mut c0);
            packed.forward_step(&x, &mut o1, &mut s1, &mut c1);
            if cfg!(not(any(
                all(feature = "accelerate", target_os = "macos"),
                feature = "gemm-blas"
            ))) {
                assert_eq!(o0, o1, "step {t}");
            }
        }
        packed.weights_mut().layers[0].in_proj_w[0] += 1.0;
        assert!(!packed.is_packed());
        packed.pack_for_decode();
        packed.layer_mut(1).out_proj_w[0] += 1.0;
        assert!(!packed.is_packed());
    }
}
