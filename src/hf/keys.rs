//! HuggingFace tensor key remapping for Mamba SSM.

/// Result of remapping an HF safetensors key.
#[derive(Debug, PartialEq)]
pub enum RemappedKey {
    /// Layer weight: (layer_index, field_name)
    Layer(usize, &'static str),
    /// Final RMSNorm weight
    NormF,
    /// Embedding matrix (not a backbone weight)
    Embedding,
    /// LM head (separate from embedding when not tied)
    LmHead,
    /// Unknown key — skip silently
    Unknown,
}

/// Remap an HF Mamba SSM safetensors key to our internal field name.
pub fn remap_m1_hf_key(hf_key: &str) -> RemappedKey {
    if hf_key == "backbone.embeddings.weight" || hf_key == "backbone.embedding.weight" {
        return RemappedKey::Embedding;
    }
    if hf_key == "lm_head.weight" {
        return RemappedKey::LmHead;
    }
    if hf_key == "backbone.norm_f.weight" {
        return RemappedKey::NormF;
    }

    if let Some(rest) = hf_key.strip_prefix("backbone.layers.") {
        let dot = rest.find('.').unwrap_or(rest.len());
        let layer_idx: usize = match rest[..dot].parse() {
            Ok(i) => i,
            Err(_) => return RemappedKey::Unknown,
        };
        let suffix = &rest[dot + 1..];

        let field = match suffix {
            "mixer.in_proj.weight" => "in_proj_w",
            "mixer.conv1d.weight" => "conv1d_weight",
            "mixer.conv1d.bias" => "conv1d_bias",
            "mixer.x_proj.weight" => "x_proj_w",
            "mixer.dt_proj.weight" => "dt_proj_w",
            "mixer.dt_proj.bias" => "dt_proj_b",
            "mixer.A_log" => "a_log",
            "mixer.D" => "d_param",
            "mixer.out_proj.weight" => "out_proj_w",
            "norm.weight" => "norm_weight",
            _ => return RemappedKey::Unknown,
        };
        return RemappedKey::Layer(layer_idx, field);
    }

    RemappedKey::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_m1_hf_layer_key() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.3.mixer.in_proj.weight"),
            RemappedKey::Layer(3, "in_proj_w")
        );
    }

    #[test]
    fn test_remap_a_log_case() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.0.mixer.A_log"),
            RemappedKey::Layer(0, "a_log")
        );
    }

    #[test]
    fn test_remap_d_to_d_param() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.5.mixer.D"),
            RemappedKey::Layer(5, "d_param")
        );
    }

    #[test]
    fn test_remap_embeddings_plural() {
        assert_eq!(
            remap_m1_hf_key("backbone.embeddings.weight"),
            RemappedKey::Embedding
        );
    }

    #[test]
    fn test_remap_lm_head_present() {
        assert_eq!(remap_m1_hf_key("lm_head.weight"), RemappedKey::LmHead);
    }

    #[test]
    fn test_remap_norm_f() {
        assert_eq!(
            remap_m1_hf_key("backbone.norm_f.weight"),
            RemappedKey::NormF
        );
    }

    #[test]
    fn test_remap_unknown() {
        assert_eq!(
            remap_m1_hf_key("something.else.weight"),
            RemappedKey::Unknown
        );
    }

    #[test]
    fn test_remap_conv1d_shape_identity() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.0.mixer.conv1d.weight"),
            RemappedKey::Layer(0, "conv1d_weight")
        );
    }
}
