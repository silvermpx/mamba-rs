//! HuggingFace checkpoint loading for Mamba LLM inference.
//!
//! Supports both HF-native (`-hf`) and original (`mamba_ssm`) config formats.
//! Handles safetensors (single + multi-shard), the `pytorch_model.bin`
//! archives `torch.save` writes, bf16 upcast and key remapping.

pub mod bf16;
pub mod config_json;
pub mod embed;
pub mod keys;
pub mod load;
pub mod load_mamba2;
pub mod load_mamba3;
pub mod pickle;
pub mod source;
pub mod torch_zip;

pub use config_json::{ModelFamily, read_family};
pub use load::load_hf;
