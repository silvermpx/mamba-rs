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
    #[default]
    F32,
    F16,
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
    pub fn compute_type(self) -> cublas_sys::cublasComputeType_t {
        match self {
            Self::F32 => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Self::Bf16 | Self::F16 => {
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
            }
        }
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
