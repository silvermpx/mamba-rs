//! AdamW: the state every trainer exports for a bit-continuous resume, and
//! the host optimizer of the CPU trainers, whose update is the GPU
//! kernel's (`kernels/adamw.cu`), the arithmetic of
//! `torch.optim.AdamW`'s single-tensor path.

use super::tensors::TensorList;

/// The full AdamW state for a bit-continuous resume: without the moments
/// and the step counter, a resumed run re-warms Adam from zero and diverges
/// from the unbroken run. The learning rate is not part of it: it belongs
/// to the caller's schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWStateBlob {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: u64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub reference_no_decay: bool,
}

/// The bias corrections of step `step`: `1 / (1 - beta^t)` for each beta,
/// computed in f64 as the GPU trainers compute them. The exponent stops at
/// 2^30 steps, past which `beta^t` lies below f64's range anyway.
pub(crate) fn bias_corrections(step: u64, beta1: f32, beta2: f32) -> (f32, f32) {
    let t = step.min(1 << 30) as i32;
    let denom1 = 1.0 - f64::from(beta1).powi(t);
    let denom2 = 1.0 - f64::from(beta2).powi(t);
    (
        (1.0 / denom1.max(1e-30)) as f32,
        (1.0 / denom2.max(1e-30)) as f32,
    )
}

/// AdamW on the host over a flat parameter vector, with the decay of each
/// element set by the caller's mask (the no-decay groups decay at zero).
pub(crate) struct HostAdamW {
    pub(crate) m: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) step: u64,
    pub(crate) lr: f32,
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) eps: f32,
    pub(crate) weight_decay: f32,
    /// Whether the no-decay groups (the A and D parameters, biases, norm
    /// scales) decay at zero, as the reference trains them.
    pub(crate) reference_no_decay: bool,
}

impl HostAdamW {
    /// An optimizer for `n` parameters at learning rate `lr` and weight
    /// decay `weight_decay`, with torch's betas and epsilon.
    pub(crate) fn new(n: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            reference_no_decay: true,
        }
    }

    /// One AdamW step of `params` on `grads`, tensor by tensor in the
    /// list's order, the tensors laid end to end in the moments; a tensor
    /// `decays(i)` rejects takes no weight decay while `reference_no_decay`
    /// is on. The step counter moves first, as the GPU trainers move
    /// theirs.
    pub(crate) fn step<W: TensorList>(
        &mut self,
        params: &mut W,
        grads: &W,
        decays: impl Fn(usize) -> bool,
    ) -> Result<(), String> {
        let n = params.n_elements();
        if n != self.m.len() {
            return Err(format!(
                "AdamW holds moments for {} parameters, the model has {n}",
                self.m.len()
            ));
        }
        self.step += 1;
        let (bc1, bc2) = bias_corrections(self.step, self.beta1, self.beta2);
        let (b1, b2, eps, lr) = (self.beta1, self.beta2, self.eps, self.lr);
        let (one_minus_b1, one_minus_b2) = (1.0 - b1, 1.0 - b2);
        let mut at = 0;
        for i in 0..params.n_tensors() {
            let wd = if decays(i) || !self.reference_no_decay {
                self.weight_decay
            } else {
                0.0
            };
            let decay_factor = 1.0 - lr * wd;
            let (p_t, g_t) = (params.tensor_mut(i), grads.tensor(i));
            let len = p_t.len();
            let (m, v) = (&mut self.m[at..at + len], &mut self.v[at..at + len]);
            for (((p, &g), mi), vi) in p_t.iter_mut().zip(g_t).zip(m).zip(v) {
                let m_new = *mi * b1 + one_minus_b1 * g;
                let v_new = *vi * b2 + one_minus_b2 * g * g;
                *mi = m_new;
                *vi = v_new;
                let m_hat = m_new * bc1;
                let v_hat = v_new * bc2;
                *p = decay_factor * *p - lr * m_hat / (v_hat.sqrt() + eps);
            }
            at += len;
        }
        Ok(())
    }

    /// The state for a bit-continuous resume.
    pub(crate) fn export_state(&self) -> AdamWStateBlob {
        AdamWStateBlob {
            m: self.m.clone(),
            v: self.v.clone(),
            step: self.step,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            reference_no_decay: self.reference_no_decay,
        }
    }

    /// Load a state [`Self::export_state`] gave; a state of another
    /// parameter count is an error.
    pub(crate) fn import_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
        if blob.m.len() != self.m.len() || blob.v.len() != self.v.len() {
            return Err(format!(
                "AdamW state of {} and {} moments for a model of {} parameters",
                blob.m.len(),
                blob.v.len(),
                self.m.len()
            ));
        }
        self.m.copy_from_slice(&blob.m);
        self.v.copy_from_slice(&blob.v);
        self.step = blob.step;
        self.beta1 = blob.beta1;
        self.beta2 = blob.beta2;
        self.eps = blob.eps;
        self.weight_decay = blob.weight_decay;
        self.reference_no_decay = blob.reference_no_decay;
        Ok(())
    }
}
