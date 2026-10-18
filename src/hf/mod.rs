//! HuggingFace checkpoint loading for Mamba LLM inference.
//!
//! Supports both HF-native (`-hf`) and original (`mamba_ssm`) config formats.
//! Handles safetensors (single + multi-shard), the `pytorch_model.bin`
//! archives `torch.save` writes, bf16 upcast and key remapping.

pub mod bf16;
pub mod config_json;
pub mod embed;
pub mod keys;
mod mamba1;
mod mamba2;
mod mamba3;
pub mod pickle;
pub mod source;
pub mod torch_zip;

pub use config_json::{ModelFamily, read_family};
// One loader a family, each returning the configuration, the backbone
// weights, the embedding and the head.
pub use mamba1::{HfMambaModel, load_mamba};
pub use mamba2::{HfMamba2Config, HfMamba2Model, load_mamba2};
pub use mamba3::{HfMamba3Config, HfMamba3MimoModel, HfMamba3Model, load_mamba3, load_mamba3_mimo};
// The config readers the loaders run, for tests that check a config alone.
#[doc(hidden)]
pub use mamba2::{is_mamba2_config, parse_mamba2_config};
#[doc(hidden)]
pub use mamba3::{parse_mamba3_config, read_mamba3_config};
