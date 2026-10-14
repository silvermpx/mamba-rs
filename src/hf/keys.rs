//! HuggingFace tensor key remapping for Mamba SSM.

/// Result of remapping an HF safetensors key.
#[derive(Debug, PartialEq)]
pub enum RemappedKey {
    /// Layer weight: (layer_index, field_name)
    Layer(usize, &'static str),
    /// A tensor of a layer's gated MLP: (layer_index, `norm2`,
    /// `norm2_bias`, `fc1` or `fc2`)
    Mlp(usize, &'static str),
    /// The final norm's weight
    NormF,
    /// The final LayerNorm's bias
    NormFBias,
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
    if hf_key == "backbone.norm_f.bias" {
        return RemappedKey::NormFBias;
    }

    if let Some(rest) = hf_key.strip_prefix("backbone.layers.") {
        // A key like "backbone.layers.0" (numeric tail, no field suffix) is
        // not ours — the old `&rest[dot + 1..]` slicing panicked on it.
        let Some((idx_str, suffix)) = rest.split_once('.') else {
            return RemappedKey::Unknown;
        };
        let layer_idx: usize = match idx_str.parse() {
            Ok(i) => i,
            Err(_) => return RemappedKey::Unknown,
        };

        let mlp = match suffix {
            "norm2.weight" => Some("norm2"),
            "norm2.bias" => Some("norm2_bias"),
            "mlp.fc1.weight" => Some("fc1"),
            "mlp.fc2.weight" => Some("fc2"),
            _ => None,
        };
        if let Some(part) = mlp {
            return RemappedKey::Mlp(layer_idx, part);
        }
        let field = match suffix {
            "mixer.in_proj.weight" => "in_proj_w",
            "mixer.in_proj.bias" => "in_proj_b",
            "mixer.conv1d.weight" => "conv1d_weight",
            "mixer.conv1d.bias" => "conv1d_bias",
            "mixer.x_proj.weight" => "x_proj_w",
            "mixer.dt_proj.weight" => "dt_proj_w",
            "mixer.dt_proj.bias" => "dt_proj_b",
            "mixer.A_log" => "a_log",
            "mixer.D" => "d_param",
            "mixer.out_proj.weight" => "out_proj_w",
            "mixer.out_proj.bias" => "out_proj_b",
            "norm.weight" => "norm_weight",
            "norm.bias" => "norm_bias",
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
    fn test_remap_numeric_tail_without_suffix_is_unknown() {
        // Regression: a checkpoint tensor literally named "backbone.layers.0"
        // used to panic via out-of-range slicing (`&rest[dot + 1..]`).
        assert_eq!(remap_m1_hf_key("backbone.layers.0"), RemappedKey::Unknown);
        assert_eq!(remap_m1_hf_key("backbone.layers."), RemappedKey::Unknown);
    }

    #[test]
    fn test_remap_mlp_keys() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.2.norm2.weight"),
            RemappedKey::Mlp(2, "norm2")
        );
        assert_eq!(
            remap_m1_hf_key("backbone.layers.0.mlp.fc1.weight"),
            RemappedKey::Mlp(0, "fc1")
        );
        assert_eq!(
            remap_m1_hf_key("backbone.layers.1.mlp.fc2.weight"),
            RemappedKey::Mlp(1, "fc2")
        );
        assert_eq!(
            remap_m1_hf_key("backbone.layers.1.norm2.bias"),
            RemappedKey::Mlp(1, "norm2_bias")
        );
    }

    /// A LayerNorm model's block norm biases.
    #[test]
    fn test_remap_layer_norm_biases() {
        assert_eq!(
            remap_m1_hf_key("backbone.layers.4.norm.bias"),
            RemappedKey::Layer(4, "norm_bias")
        );
        assert_eq!(
            remap_m1_hf_key("backbone.norm_f.bias"),
            RemappedKey::NormFBias
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
