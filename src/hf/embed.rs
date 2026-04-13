//! Embedding lookup and lm_head projection for LM inference.

use crate::ops::blas::matvec_forward;

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

/// Compute logits = embed @ hidden (row-major gemv).
///
/// `embed`: `[vocab_size_padded * d_model]` row-major.
/// `hidden`: `[d_model]`.
/// `logits`: `[vocab_size]` output (clamped to real vocab, not padded).
pub fn lm_head_logits(
    logits: &mut [f32],
    hidden: &[f32],
    embed: &[f32],
    vocab_size: usize,
    vocab_size_padded: usize,
    d_model: usize,
) {
    if vocab_size == vocab_size_padded {
        matvec_forward(logits, hidden, embed, None, d_model, vocab_size);
    } else {
        let mut padded_logits = vec![0.0f32; vocab_size_padded];
        matvec_forward(
            &mut padded_logits,
            hidden,
            embed,
            None,
            d_model,
            vocab_size_padded,
        );
        logits[..vocab_size].copy_from_slice(&padded_logits[..vocab_size]);
    }
}

/// Compute logits using a separate lm_head weight matrix (untied weights).
///
/// `lm_head_w`: `[vocab_size * d_model]` row-major.
pub fn lm_head_logits_untied(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head_w: &[f32],
    vocab_size: usize,
    d_model: usize,
) {
    matvec_forward(logits, hidden, lm_head_w, None, d_model, vocab_size);
}
