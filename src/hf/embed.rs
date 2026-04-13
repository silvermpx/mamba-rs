//! Embedding lookup and lm_head projection for LM inference.

use crate::ops::blas::{matvec_forward, sgemm_forward};

/// Look up one token's embedding vector.
///
/// Returns a slice `[d_model]` into the embedding table.
/// Panics with a descriptive message if `token_id >= vocab_size`.
#[inline]
pub fn embed_lookup(embed: &[f32], token_id: u32, d_model: usize, vocab_size: usize) -> &[f32] {
    let id = token_id as usize;
    assert!(
        id < vocab_size,
        "token_id {id} out of range (vocab_size={vocab_size})"
    );
    &embed[id * d_model..(id + 1) * d_model]
}

/// Compute logits = embed @ hidden (tied lm_head).
///
/// `embed`: `[vocab_size_padded * d_model]` row-major — each row is one token's embedding.
/// `hidden`: `[d_model]`.
/// `logits`: `[vocab_size]` output (clamped to real vocab, not padded).
///
/// Mathematically: `logits[v] = dot(embed[v, :], hidden)` = `embed[V,D] @ hidden[D,1]`.
/// This is `sgemm_forward(batch=V, n_in=D, n_out=1)` with embed as "X" and hidden as "W".
/// Compute logits with tied weights. `scratch` must be `[vocab_size_padded]` (pre-allocated, reused).
pub fn lm_head_logits(
    logits: &mut [f32],
    hidden: &[f32],
    embed: &[f32],
    vocab_size: usize,
    vocab_size_padded: usize,
    d_model: usize,
    scratch: &mut [f32],
) {
    debug_assert!(scratch.len() >= vocab_size_padded);
    sgemm_forward(
        &mut scratch[..vocab_size_padded],
        embed,
        hidden,
        None,
        vocab_size_padded,
        d_model,
        1,
    );
    logits[..vocab_size].copy_from_slice(&scratch[..vocab_size]);
}

/// Compute logits using a separate lm_head weight matrix (untied weights).
///
/// `lm_head_w`: `[vocab_size * d_model]` row-major (PyTorch layout: [out, in]).
/// After transpose during HF loading, this becomes `[d_model, vocab_size]`.
/// So we use matvec: `logits[V] = hidden[D] @ W[D,V]`.
pub fn lm_head_logits_untied(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head_w: &[f32],
    vocab_size: usize,
    d_model: usize,
) {
    matvec_forward(logits, hidden, lm_head_w, None, d_model, vocab_size);
}
