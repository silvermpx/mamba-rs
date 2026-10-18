//! One token of several independent sequences through the same weights,
//! shared by the families' CPU backbones.

/// B independent samples through `step`, one T=1 step each: in parallel
/// when the batch fills the pool, one after another otherwise. Each sample
/// runs the same arithmetic either way.
pub(crate) fn step_batch_by<S: Send, C: Send>(
    (inputs, outputs): (&[f32], &mut [f32]),
    (states, scratches): (&mut [S], &mut [C]),
    (d_model, input_dim): (usize, usize),
    step: impl Fn(&[f32], &mut [f32], &mut S, &mut C) + Sync,
) {
    let batch = states.len();
    assert_eq!(
        inputs.len(),
        batch * input_dim,
        "inputs size mismatch: expected {}, got {}",
        batch * input_dim,
        inputs.len()
    );
    assert_eq!(
        outputs.len(),
        batch * d_model,
        "outputs size mismatch: expected {}, got {}",
        batch * d_model,
        outputs.len()
    );
    assert_eq!(
        scratches.len(),
        batch,
        "scratches count mismatch: expected {batch}, got {}",
        scratches.len()
    );
    if batch >= rayon::current_num_threads().max(2) {
        use rayon::prelude::*;
        outputs
            .par_chunks_mut(d_model)
            .zip(states.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .enumerate()
            .for_each(|(b, ((out, state), scratch))| {
                step(
                    &inputs[b * input_dim..(b + 1) * input_dim],
                    out,
                    state,
                    scratch,
                );
            });
    } else {
        for (b, ((out, state), scratch)) in outputs
            .chunks_mut(d_model)
            .zip(states.iter_mut())
            .zip(scratches.iter_mut())
            .enumerate()
        {
            step(
                &inputs[b * input_dim..(b + 1) * input_dim],
                out,
                state,
                scratch,
            );
        }
    }
}
