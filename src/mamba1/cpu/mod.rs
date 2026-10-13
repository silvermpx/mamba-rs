//! CPU backend for Mamba SSM.
//!
//! - `inference` — zero-alloc T=1 step (single + batch + sequence)
//! - `forward` — training forward pass (batched SGEMM)
//! - `backward` — training backward pass (BPTT)
//! - `target` — target network forward (no activation saves)

pub mod backward;
pub mod backward_ops;
pub mod flat;
pub mod forward;
pub mod inference;
pub mod parallel;
pub mod prefill;
pub mod scratch;
pub mod target;
pub mod trainer;
pub mod weights;

/// A layer's optional bias as the products take it: `None` when the config
/// has none and the tensor is empty.
#[inline]
pub(crate) fn bias(b: &[f32]) -> Option<&[f32]> {
    if b.is_empty() { None } else { Some(b) }
}

/// [`bias`] for a gradient the backward accumulates into.
#[inline]
pub(crate) fn bias_mut(b: &mut [f32]) -> Option<&mut [f32]> {
    if b.is_empty() { None } else { Some(b) }
}

/// Panics unless `w` carries the optional tensors `dims` sizes the buffers
/// for. [`MambaDims::new`](crate::ops::dims::MambaDims::new) describes the
/// reference's default options; a model with others takes
/// [`MambaDims::from_config`](crate::ops::dims::MambaDims::from_config),
/// since buffers sized for the defaults would leave a bias without its
/// gradient.
pub(crate) fn assert_dims_fit(dims: &crate::ops::dims::MambaDims, w: &weights::TrainMambaWeights) {
    let fits = w.layers.iter().all(|l| {
        l.in_proj_b.is_empty() != dims.proj_bias
            && l.out_proj_b.is_empty() != dims.proj_bias
            && l.conv1d_bias.is_empty() != dims.conv_bias
    });
    assert!(
        fits,
        "the dims describe other options than the weights: build them with MambaDims::from_config"
    );
}
