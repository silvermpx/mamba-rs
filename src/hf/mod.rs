//! HuggingFace checkpoint loading for Mamba LLM inference.
//!
//! Supports both HF-native (`-hf`) and original (`mamba_ssm`) config formats.
//! Handles safetensors (single + multi-shard), bf16 upcast, key remapping.

pub mod bf16;
pub mod config_json;
pub mod embed;
pub mod keys;
pub mod load;

pub use config_json::ModelFamily;
pub use load::load_hf;
