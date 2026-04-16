//! Weight storage dtype for GPU inference (f32/f16/bf16).
//!
//! Compute always stays f32 (CUBLAS_COMPUTE_32F for GEMMs, f32 for all custom kernels).
//! Only bulk linear weights (in_proj, out_proj, x_proj, dt_proj, embed, lm_head) and
//! activations between GEMMs use reduced precision. Norms, biases, a_log, D stay f32.
//!
//! Based on official state-spaces/mamba design and NVIDIA cuBLAS best practices.

use cudarc::cublas::sys as cublas_sys;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum WeightDtype {
    /// IEEE 754 single precision. Default. Largest memory footprint and
    /// the safe choice for math sensitive to precision (full pre-training,
    /// long-horizon RL). Compute is always f32 internally regardless.
    #[default]
    F32,
    /// IEEE 754 half precision (`half::f16`). Tightest dynamic range —
    /// requires the dynamic loss scaler in training to avoid gradient
    /// underflow. ~2× memory savings and ~1.3× tok/s speedup at
    /// inference vs f32 on Ada / Hopper.
    F16,
    /// Brain float 16 (`half::bf16`). Same exponent range as f32 with
    /// reduced mantissa — no loss scaler needed in training. The
    /// recommended default for mixed-precision: ~2× memory, ~1.3×
    /// tok/s, no overflow regime to manage.
    Bf16,
}

impl WeightDtype {
    /// Byte size of one element.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// cuBLAS/CUDA data type identifier for cublasGemmEx.
    pub fn cuda_data_type(self) -> cublas_sys::cudaDataType {
        match self {
            Self::F32 => cublas_sys::cudaDataType::CUDA_R_32F,
            Self::F16 => cublas_sys::cudaDataType::CUDA_R_16F,
            Self::Bf16 => cublas_sys::cudaDataType::CUDA_R_16BF,
        }
    }

    /// cuBLAS compute type — always f32 for inference stability.
    ///
    /// For bf16/f16 inputs we explicitly use `CUBLAS_COMPUTE_32F_PEDANTIC`
    /// rather than `CUBLAS_COMPUTE_32F`. The non-pedantic variant is
    /// allowed by cuBLAS to pick a TF32/faster-than-f32 accumulation on
    /// Ampere+ GPUs, and in practice on RTX 6000 Ada + CUDA 12.8 this chose
    /// a kernel that silently reduced precision on large bf16 GEMMs
    /// (mamba-1.4b, d_inner=4096) — greedy decode on bf16 diverged from
    /// bf16 HF reference. Pedantic forces true f32 accumulation with a
    /// modest perf cost on small-M GEMMs.
    ///
    /// **Trade-off (audit Agent 2/5)**: PEDANTIC may forfeit Tensor Core
    /// HMMA/BMMA acceleration for bf16/f16 GEMMs on Ada/Hopper — possibly
    /// 2–4× speed regression on large GEMMs. For models you have validated
    /// not to regress under non-pedantic accumulation (e.g. small models
    /// where the rounding bias cancels), use [`compute_type_fast`] instead.
    /// For mamba-1.4b and similar, keep PEDANTIC — it is the only mode
    /// known to bit-match the HF reference at greedy decode.
    pub fn compute_type(self) -> cublas_sys::cublasComputeType_t {
        match self {
            Self::F32 => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Self::Bf16 | Self::F16 => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
        }
    }

    /// Opt-in faster cuBLAS compute mode — `CUBLAS_COMPUTE_32F` (non-PEDANTIC)
    /// for bf16/f16, allowing cuBLAS to pick BMMA/HMMA Tensor Core kernels
    /// with f32 accumulate. Up to 2–4× faster than [`compute_type`] on
    /// large bf16/f16 GEMMs, but cuBLAS is permitted to use approximate
    /// accumulation that may lose precision on large reductions. For small
    /// RL-scope models (d_inner ≤ 512) the difference is typically below
    /// f32-eps and safe to use.
    ///
    /// **VALIDATE PER MODEL** before switching production training to this.
    /// mamba-1.4b regressed under this mode on Ada + CUDA 12.8 — that is
    /// why [`compute_type`] defaults to PEDANTIC.
    pub fn compute_type_fast(self) -> cublas_sys::cublasComputeType_t {
        // Same f32 mode for all dtypes — non-pedantic lets cuBLAS choose
        // the best Tensor Core kernel for bf16/f16 inputs.
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }

    pub fn is_f32(self) -> bool {
        matches!(self, Self::F32)
    }

    pub fn is_half(self) -> bool {
        !self.is_f32()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(WeightDtype::F32.size_bytes(), 4);
        assert_eq!(WeightDtype::F16.size_bytes(), 2);
        assert_eq!(WeightDtype::Bf16.size_bytes(), 2);
    }

    #[test]
    fn is_half() {
        assert!(!WeightDtype::F32.is_half());
        assert!(WeightDtype::F16.is_half());
        assert!(WeightDtype::Bf16.is_half());
    }
}
