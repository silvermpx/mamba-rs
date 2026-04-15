//! HuggingFace config.json parsing for Mamba models.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Mamba1,
    Mamba2,
}

/// Raw HF config.json — supports both HF-native and original formats.
#[derive(Deserialize)]
struct RawConfig {
    // HF-native fields
    model_type: Option<String>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    state_size: Option<usize>,
    conv_kernel: Option<usize>,
    time_step_rank: Option<usize>,
    // Original fields
    d_model: Option<usize>,
    n_layer: Option<usize>,
    d_state: Option<usize>,
    d_conv: Option<usize>,
    // Shared fields
    expand: Option<usize>,
    vocab_size: Option<usize>,
    // HF-specific flags
    tie_word_embeddings: Option<bool>,
    use_bias: Option<bool>,
    use_conv_bias: Option<bool>,
    #[allow(dead_code)]
    residual_in_fp32: Option<bool>,
    // RMSNorm epsilon — state-spaces/mamba-*-hf uses 1e-5 (default);
    // FalconMamba uses 1e-6. Reading this from the config lets both
    // loader paths produce numerically correct norms on the native
    // architecture's scale. Name mirrors the HF field.
    rms_norm_eps: Option<f32>,
    layer_norm_epsilon: Option<f32>,
}

/// Parsed and validated Mamba config for HF loading.
#[derive(Debug, Clone)]
pub struct HfMambaConfig {
    pub family: ModelFamily,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub use_bias: bool,
    pub use_conv_bias: bool,
    /// RMSNorm epsilon from the HF config. `1e-5` is the state-spaces
    /// default; FalconMamba ships `1e-6`. Callers should plumb this into
    /// every norm kernel instead of hardcoding the constant so both
    /// checkpoint families produce numerically correct outputs.
    pub rms_norm_eps: f32,
}

pub fn parse_config_json(json_bytes: &[u8]) -> Result<HfMambaConfig, String> {
    let raw: RawConfig =
        serde_json::from_slice(json_bytes).map_err(|e| format!("config.json parse error: {e}"))?;

    let family = detect_family(&raw)?;
    let (d_model, n_layers, d_state, d_conv) = extract_dims(&raw)?;
    let expand = raw.expand.unwrap_or(2);
    let vocab_size = raw.vocab_size.ok_or("missing vocab_size in config.json")?;

    if let Some(tsr) = raw.time_step_rank {
        let expected = d_model.div_ceil(16);
        if tsr != expected {
            return Err(format!(
                "time_step_rank mismatch: config has {tsr}, expected d_model.div_ceil(16) = {expected}"
            ));
        }
    }

    // Accept either name. HF-native Mamba uses `rms_norm_eps`; some
    // older forks / Mamba2 configs use `layer_norm_epsilon`. Default
    // 1e-5 matches state-spaces/mamba reference.
    let rms_norm_eps = raw
        .rms_norm_eps
        .or(raw.layer_norm_epsilon)
        .unwrap_or(1e-5);

    Ok(HfMambaConfig {
        family,
        d_model,
        n_layers,
        d_state,
        d_conv,
        expand,
        vocab_size,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(true),
        use_bias: raw.use_bias.unwrap_or(false),
        use_conv_bias: raw.use_conv_bias.unwrap_or(true),
        rms_norm_eps,
    })
}

fn detect_family(raw: &RawConfig) -> Result<ModelFamily, String> {
    if let Some(mt) = &raw.model_type {
        match mt.as_str() {
            "mamba" | "falcon_mamba" => Ok(ModelFamily::Mamba1),
            "mamba2" => Ok(ModelFamily::Mamba2),
            other => Err(format!("unsupported model_type: {other}")),
        }
    } else if raw.d_model.is_some() {
        Ok(ModelFamily::Mamba1)
    } else {
        Err("cannot detect model family: no model_type or d_model field".into())
    }
}

fn extract_dims(raw: &RawConfig) -> Result<(usize, usize, usize, usize), String> {
    if raw.model_type.is_some() || raw.hidden_size.is_some() {
        let d_model = raw.hidden_size.ok_or("HF config missing hidden_size")?;
        let n_layers = raw
            .num_hidden_layers
            .ok_or("HF config missing num_hidden_layers")?;
        let d_state = raw.state_size.ok_or("HF config missing state_size")?;
        let d_conv = raw.conv_kernel.ok_or("HF config missing conv_kernel")?;
        Ok((d_model, n_layers, d_state, d_conv))
    } else {
        let d_model = raw.d_model.ok_or("original config missing d_model")?;
        let n_layers = raw.n_layer.ok_or("original config missing n_layer")?;
        let d_state = raw.d_state.unwrap_or(16);
        let d_conv = raw.d_conv.unwrap_or(4);
        Ok((d_model, n_layers, d_state, d_conv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hf_native_config() {
        let json = br#"{
            "model_type": "mamba",
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "state_size": 16,
            "conv_kernel": 4,
            "expand": 2,
            "vocab_size": 50280,
            "time_step_rank": 48
        }"#;
        let cfg = parse_config_json(json).unwrap();
        assert_eq!(cfg.family, ModelFamily::Mamba1);
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.expand, 2);
        assert_eq!(cfg.vocab_size, 50280);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn test_parse_original_config() {
        let json = br#"{
            "d_model": 768,
            "n_layer": 24,
            "vocab_size": 50257
        }"#;
        let cfg = parse_config_json(json).unwrap();
        assert_eq!(cfg.family, ModelFamily::Mamba1);
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.vocab_size, 50257);
    }

    #[test]
    fn test_auto_detect_format() {
        let hf = br#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(parse_config_json(hf).unwrap().family, ModelFamily::Mamba1);

        let orig = br#"{"d_model":64,"n_layer":2,"vocab_size":100}"#;
        assert_eq!(parse_config_json(orig).unwrap().family, ModelFamily::Mamba1);
    }

    #[test]
    fn test_model_family_dispatch() {
        let m1 = br#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(parse_config_json(m1).unwrap().family, ModelFamily::Mamba1);

        let m2 = br#"{"model_type":"mamba2","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(parse_config_json(m2).unwrap().family, ModelFamily::Mamba2);

        let falcon = br#"{"model_type":"falcon_mamba","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(
            parse_config_json(falcon).unwrap().family,
            ModelFamily::Mamba1
        );
    }

    #[test]
    fn test_time_step_rank_mismatch_err() {
        let json = br#"{
            "model_type": "mamba",
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "state_size": 16,
            "conv_kernel": 4,
            "vocab_size": 50280,
            "time_step_rank": 999
        }"#;
        let err = parse_config_json(json).unwrap_err();
        assert!(err.contains("time_step_rank mismatch"), "{err}");
    }

    #[test]
    fn test_missing_field_returns_err() {
        let json = br#"{"model_type": "mamba", "hidden_size": 768}"#;
        let err = parse_config_json(json).unwrap_err();
        assert!(
            err.contains("missing") || err.contains("num_hidden_layers"),
            "{err}"
        );
    }

    #[test]
    fn test_tie_word_embeddings_default() {
        let json = br#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        let cfg = parse_config_json(json).unwrap();
        assert!(cfg.tie_word_embeddings);
    }
}
