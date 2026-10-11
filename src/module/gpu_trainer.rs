//! One surface over the GPU trainers of the four families.
//!
//! Every family's trainer has these methods of its own; [`GpuTrainer`]
//! names them for a training loop that takes any family. The master
//! weights come back through each trainer's `snapshot_master`, typed by its
//! family.

use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::trainer::{BackwardMetrics, BackwardOpts, MambaGpuTrainer, StepMetrics};
use crate::mamba2::gpu::Mamba2GpuTrainer;
use crate::mamba3_mimo::gpu::Mamba3MimoGpuTrainer;
use crate::mamba3_siso::gpu::trainer::Mamba3GpuTrainer;

/// What the GPU trainer of every family does.
///
/// ```rust,no_run
/// use mamba_rs::{GpuOptions, GpuTrainer, Mamba2Config, Mamba2GpuTrainer, Mamba2Weights};
/// use mamba_rs::TrainSessionCfg;
///
/// fn train(t: &mut impl GpuTrainer, input: &[f32], target: &[f32]) -> f32 {
///     let mut out = vec![0.0f32; t.batch() * t.seq_len() * t.d_model()];
///     t.forward(input, &mut out).unwrap();
///     let n = out.len() as f32;
///     let d: Vec<f32> = out.iter().zip(target).map(|(o, y)| 2.0 * (o - y) / n).collect();
///     t.backward_step(&d, Default::default()).unwrap();
///     out.iter().zip(target).map(|(o, y)| (o - y) * (o - y)).sum::<f32>() / n
/// }
///
/// let cfg = Mamba2Config::default();
/// let weights = Mamba2Weights::init(&cfg, 16, 7);
/// let session = TrainSessionCfg::new(16, 2, 64);
/// let mut trainer = Mamba2GpuTrainer::new(&weights, cfg, session, GpuOptions::default()).unwrap();
/// let input = vec![0.1f32; 2 * 64 * 16];
/// let target = vec![0.0f32; 2 * 64 * cfg.d_model];
/// let loss = train(&mut trainer, &input, &target);
/// ```
pub trait GpuTrainer {
    /// The training forward over one window: `input` is `[batch, seq_len,
    /// input_dim]`, and `temporal_out` (`[batch, seq_len, d_model]`)
    /// receives every position's final-norm rows, f32 in every storage
    /// precision. The recurrent state carries from the last window.
    fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String>;

    /// The backward of the last forward for `d_temporal` (d loss / d its
    /// rows), then the optimizer tail `opts` asks for: the global-norm
    /// clip, the skip of a spiking window, an accumulating call.
    fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String>;

    /// Forward, backward and the optimizer step in one call.
    fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String>;

    /// Zero the recurrent state, the start of new sequences.
    fn reset_state(&mut self) -> Result<(), String>;

    /// Set the AdamW learning rate for the next steps.
    fn set_lr(&mut self, lr: f32) -> Result<(), String>;

    /// The AdamW learning rate.
    fn lr(&self) -> f32;

    /// Sequences per window.
    fn batch(&self) -> usize;

    /// Tokens per window.
    fn seq_len(&self) -> usize;

    /// Width of an input row.
    fn input_dim(&self) -> usize;

    /// Width of the model, the rows `forward` writes.
    fn d_model(&self) -> usize;

    /// Storage precision of the weights.
    fn dtype(&self) -> WeightDtype;

    /// The CUDA context: the GEMM mode and route live here.
    fn ctx(&self) -> &GpuCtx;

    /// The f16 loss scale and its growth counter; `None` in the other
    /// precisions.
    fn scaler_state(&self) -> Option<(f32, u32)>;

    /// Restore the f16 loss scale and its growth counter; no effect in the
    /// other precisions.
    fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32);
}

// Each family's trainer has these methods of its own (inherent methods win
// method resolution), so the trait methods hand straight to them.
macro_rules! gpu_trainer_by_inherent_methods {
    ($($ty:ty),+ $(,)?) => {$(
        impl GpuTrainer for $ty {
            fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
                <$ty>::forward(self, input, temporal_out)
            }
            fn backward_step(
                &mut self,
                d_temporal: &[f32],
                opts: BackwardOpts,
            ) -> Result<BackwardMetrics, String> {
                <$ty>::backward_step(self, d_temporal, opts)
            }
            fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
                <$ty>::step(self, input, d_temporal)
            }
            fn reset_state(&mut self) -> Result<(), String> {
                <$ty>::reset_state(self)
            }
            fn set_lr(&mut self, lr: f32) -> Result<(), String> {
                <$ty>::set_lr(self, lr)
            }
            fn lr(&self) -> f32 {
                <$ty>::lr(self)
            }
            fn batch(&self) -> usize {
                <$ty>::batch(self)
            }
            fn seq_len(&self) -> usize {
                <$ty>::seq_len(self)
            }
            fn input_dim(&self) -> usize {
                <$ty>::input_dim(self)
            }
            fn d_model(&self) -> usize {
                <$ty>::d_model(self)
            }
            fn dtype(&self) -> WeightDtype {
                <$ty>::dtype(self)
            }
            fn ctx(&self) -> &GpuCtx {
                <$ty>::ctx(self)
            }
            fn scaler_state(&self) -> Option<(f32, u32)> {
                <$ty>::scaler_state(self)
            }
            fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
                <$ty>::load_scaler_state(self, scale, growth_tracker)
            }
        }
    )+};
}

gpu_trainer_by_inherent_methods!(
    MambaGpuTrainer,
    Mamba2GpuTrainer,
    Mamba3GpuTrainer,
    Mamba3MimoGpuTrainer,
);
