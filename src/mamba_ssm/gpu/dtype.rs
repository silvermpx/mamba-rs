//! Weight storage dtype for GPU inference (f32/f16/bf16).
//!
//! Compute always stays f32 (CUBLAS_COMPUTE_32F for GEMMs, f32 for all custom kernels).
//! Only bulk linear weights (in_proj, out_proj, x_proj, dt_proj, embed, lm_head) and
//! activations between GEMMs use reduced precision. Norms, biases, a_log, D stay f32.
//!
//! Based on official state-spaces/mamba design and NVIDIA cuBLAS best practices.

use cudarc::cublas::sys as cublas_sys;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightDtype {
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
    pub fn compute_type(self) -> cublas_sys::cublasComputeType_t {
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

impl Default for WeightDtype {
    fn default() -> Self {
        Self::F32
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
