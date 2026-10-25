//! Weight storage precision for the GPU engines (f32 / tf32 / bf16 / f16).
//!
//! Every kernel accumulates in f32. Only bulk linear weights (in_proj,
//! out_proj, x_proj, dt_proj, embed, lm_head) and activations between GEMMs
//! use reduced precision; norms, biases, a_log and D stay f32. `Tf32` is f32
//! storage whose deterministic GEMM products all run on the tensor cores in
//! TF32, each operand rounded with `cvt.rna` and each forward or input
//! gradient cut by its layer alone.
//!
//! Based on official state-spaces/mamba design and NVIDIA cuBLAS best practices.

#[cfg(feature = "cublas")]
use cudarc::cublas::sys as cublas_sys;

pub use crate::precision::WeightDtype;

/// The cuBLAS names of a storage dtype, for the vendor GEMM modes.
#[cfg(feature = "cublas")]
impl WeightDtype {
    /// cuBLAS/CUDA data type identifier for cublasGemmEx.
    pub fn cuda_data_type(self) -> cublas_sys::cudaDataType {
        match self {
            Self::F32 | Self::Tf32 => cublas_sys::cudaDataType::CUDA_R_32F,
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
    /// **Trade-off**: PEDANTIC may forfeit Tensor Core
    /// HMMA/BMMA acceleration for bf16/f16 GEMMs on Ada/Hopper — possibly
    /// 2–4× speed regression on large GEMMs. For models you have validated
    /// not to regress under non-pedantic accumulation (e.g. small models
    /// where the rounding bias cancels), use [`Self::compute_type_fast`] instead.
    /// For mamba-1.4b and similar, keep PEDANTIC — it is the only mode
    /// known to bit-match the HF reference at greedy decode.
    pub fn compute_type(self) -> cublas_sys::cublasComputeType_t {
        match self {
            Self::F32 | Self::Tf32 => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Self::Bf16 | Self::F16 => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
        }
    }

    /// Opt-in faster cuBLAS compute mode — `CUBLAS_COMPUTE_32F` (non-PEDANTIC)
    /// for bf16/f16, allowing cuBLAS to pick BMMA/HMMA Tensor Core kernels
    /// with f32 accumulate. Up to 2–4× faster than [`Self::compute_type`] on
    /// large bf16/f16 GEMMs, but cuBLAS is permitted to use approximate
    /// accumulation that may lose precision on large reductions. For small
    /// RL-scope models (d_inner ≤ 512) the difference is typically below
    /// f32-eps and safe to use.
    ///
    /// **VALIDATE PER MODEL** before switching production training to this.
    /// mamba-1.4b regressed under this mode on Ada + CUDA 12.8 — that is
    /// why [`Self::compute_type`] defaults to PEDANTIC.
    pub fn compute_type_fast(self) -> cublas_sys::cublasComputeType_t {
        // Same f32 mode for all dtypes — non-pedantic lets cuBLAS choose
        // the best Tensor Core kernel for bf16/f16 inputs.
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

#[cfg(all(test, feature = "cublas"))]
mod tests {
    use super::*;

    #[test]
    fn tf32_has_the_f32_data_type() {
        assert_eq!(
            WeightDtype::Tf32.cuda_data_type(),
            WeightDtype::F32.cuda_data_type()
        );
    }
}
