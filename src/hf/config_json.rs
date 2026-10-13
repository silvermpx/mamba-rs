//! HuggingFace config.json parsing for Mamba models.

use std::path::Path;

use serde::Deserialize;

pub use crate::config::ModelFamily;

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
    // The original format spells the same flag `tie_embeddings` and pads
    // the vocabulary itself; `save_pretrained` writes both keys.
    tie_embeddings: Option<bool>,
    pad_vocab_size_multiple: Option<usize>,
    use_bias: Option<bool>,
    use_conv_bias: Option<bool>,
    // (residual_in_fp32 intentionally not parsed — serde skips unknown
    // fields, and nothing reads it; our mixed pipeline is always
    // residual-in-f32.)
    // RMSNorm epsilon — state-spaces/mamba-*-hf uses 1e-5 (default);
    // FalconMamba uses 1e-6. Reading this from the config lets both
    // loader paths produce numerically correct norms on the native
    // architecture's scale. Name mirrors the HF field.
    rms_norm_eps: Option<f32>,
    layer_norm_epsilon: Option<f32>,
    // The HF format sizes the mixer by `intermediate_size`.
    intermediate_size: Option<usize>,
    // Backbone parts the original format can carry. Each one this build
    // does not model is refused rather than loaded without it.
    d_intermediate: Option<usize>,
    rms_norm: Option<bool>,
    attn_layer_idx: Option<Vec<usize>>,
    // The original format names the mixer under `ssm_cfg.layer`; a config
    // without it is Mamba-1.
    ssm_cfg: Option<RawSsmCfg>,
}

/// The mixer arguments of an original-format config. A Mamba-1 checkpoint
/// keeps its own `d_state`, `d_conv`, `expand` and `dt_rank` here, so they
/// are read here before the top-level fields.
#[derive(Deserialize)]
struct RawSsmCfg {
    layer: Option<String>,
    is_mimo: Option<bool>,
    d_state: Option<usize>,
    d_conv: Option<usize>,
    expand: Option<usize>,
    dt_rank: Option<serde_json::Value>,
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

/// A `config.json` as Python's `json.dump` writes it, which can hold the
/// bare tokens `Infinity`, `-Infinity` and `NaN` strict JSON lacks (a
/// transformers Mamba-2 config carries `"time_step_limit": [0.0,
/// Infinity]`). Each such token outside a string is read as a string of
/// its own name, which [`json_f32`] reads back as the float it names.
pub(crate) fn config_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("config.json is not UTF-8: {e}"))?;
    let mut out = String::with_capacity(text.len() + 16);
    let (mut in_string, mut escaped) = (false, false);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if let Some(token) = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|t| rest.starts_with(t))
        {
            out.push('"');
            out.push_str(token);
            out.push('"');
            rest = &rest[token.len()..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    serde_json::from_str(&out).map_err(|e| format!("config.json parse error: {e}"))
}

/// A config number, or one of the bare non-finite tokens [`config_json`]
/// keeps as strings.
pub(crate) fn json_f32(v: &serde_json::Value) -> Option<f32> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().map(|x| x as f32),
        serde_json::Value::String(s) => match s.as_str() {
            "Infinity" => Some(f32::INFINITY),
            "-Infinity" => Some(f32::NEG_INFINITY),
            "NaN" => Some(f32::NAN),
            _ => None,
        },
        _ => None,
    }
}

pub fn parse_config_json(json_bytes: &[u8]) -> Result<HfMambaConfig, String> {
    let raw: RawConfig = config_json(json_bytes)?;

    let family = detect_family(&raw)?;
    if raw.model_type.as_deref() == Some("falcon_mamba") {
        // FalconMamba RMS-normalises B, C and dt inside every mixer. Read as
        // plain Mamba-1 it would load and give wrong outputs.
        return Err(
            "falcon_mamba checkpoints need the B, C and dt mixer norms, \
                    which this build does not implement"
                .into(),
        );
    }
    if raw.d_intermediate.is_some_and(|d| d > 0) {
        return Err(
            "config.json: d_intermediate > 0 (an MLP after every mixer) \
                    is not supported for Mamba-1 yet"
                .into(),
        );
    }
    if raw.rms_norm == Some(false) {
        return Err("config.json: rms_norm = false (LayerNorm) is not supported yet".into());
    }
    if raw.attn_layer_idx.as_ref().is_some_and(|a| !a.is_empty()) {
        return Err("config.json: attention layers (attn_layer_idx) are not supported yet".into());
    }
    let (d_model, n_layers, d_state, d_conv) = extract_dims(&raw)?;
    let ssm_cfg = raw.ssm_cfg.as_ref();
    let expand = match raw.intermediate_size {
        // The HF format sizes the mixer by intermediate_size, and its
        // `expand` can disagree (FalconMamba carries 16 against 8192 / 4096).
        Some(inner) => {
            if d_model == 0 || inner % d_model != 0 {
                return Err(format!(
                    "intermediate_size {inner} is not a whole multiple of hidden_size \
                     {d_model}"
                ));
            }
            inner / d_model
        }
        None => ssm_cfg.and_then(|c| c.expand).or(raw.expand).unwrap_or(2),
    };
    if let Some(rank) = ssm_cfg.and_then(|c| c.dt_rank.as_ref()) {
        let expected = d_model.div_ceil(16);
        let explicit = match rank {
            serde_json::Value::String(s) if s == "auto" => None,
            serde_json::Value::Number(n) => n.as_u64().map(|v| v as usize),
            other => {
                return Err(format!(
                    "config.json: ssm_cfg.dt_rank {other} is not understood"
                ));
            }
        };
        if explicit.is_some_and(|r| r != expected) {
            return Err(format!(
                "ssm_cfg.dt_rank {rank} differs from d_model.div_ceil(16) = {expected}, \
                 the only rank this build supports"
            ));
        }
    }
    let vocab_size = raw.vocab_size.ok_or("missing vocab_size in config.json")?;
    // The original format stores the UNPADDED vocabulary and rounds it up
    // before allocating the embedding and the head, so a checkpoint saved
    // with the default 50277 carries 50280 rows. The HF format stores the
    // final row count, so it is left alone.
    let is_original = raw.model_type.is_none() && raw.hidden_size.is_none();
    let vocab_size = if is_original {
        let multiple = raw.pad_vocab_size_multiple.unwrap_or(8);
        match multiple {
            0 | 1 => vocab_size,
            m => vocab_size.div_ceil(m) * m,
        }
    } else {
        vocab_size
    };

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
    let rms_norm_eps = raw.rms_norm_eps.or(raw.layer_norm_epsilon).unwrap_or(1e-5);

    Ok(HfMambaConfig {
        family,
        d_model,
        n_layers,
        d_state,
        d_conv,
        expand,
        vocab_size,
        tie_word_embeddings: raw
            .tie_word_embeddings
            .or(raw.tie_embeddings)
            .unwrap_or(true),
        use_bias: raw.use_bias.unwrap_or(false),
        use_conv_bias: raw.use_conv_bias.unwrap_or(true),
        rms_norm_eps,
    })
}

/// The family a checkpoint directory's `config.json` names, without the
/// checks a loader makes on the rest of the file.
pub fn read_family(dir: &Path) -> Result<ModelFamily, String> {
    let json = std::fs::read(dir.join("config.json"))
        .map_err(|e| format!("cannot read config.json: {e}"))?;
    let raw: RawConfig = config_json(&json)?;
    detect_family(&raw)
}

fn detect_family(raw: &RawConfig) -> Result<ModelFamily, String> {
    // A Mamba-3 config marks a MIMO mixer under ssm_cfg.
    let mamba3 = || {
        if raw.ssm_cfg.as_ref().and_then(|c| c.is_mimo) == Some(true) {
            ModelFamily::Mamba3Mimo
        } else {
            ModelFamily::Mamba3
        }
    };
    if let Some(mt) = &raw.model_type {
        match mt.as_str() {
            "mamba" | "falcon_mamba" => Ok(ModelFamily::Mamba1),
            "mamba2" => Ok(ModelFamily::Mamba2),
            "mamba3" | "mamba3_siso" => Ok(mamba3()),
            other => Err(format!("unsupported model_type: {other}")),
        }
    } else if raw.d_model.is_some() {
        match raw.ssm_cfg.as_ref().and_then(|c| c.layer.as_deref()) {
            None | Some("Mamba1") => Ok(ModelFamily::Mamba1),
            Some("Mamba2") => Ok(ModelFamily::Mamba2),
            Some("Mamba3") => Ok(mamba3()),
            Some(other) => Err(format!("unsupported ssm_cfg.layer: {other}")),
        }
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
        let ssm = raw.ssm_cfg.as_ref();
        let d_state = ssm.and_then(|c| c.d_state).or(raw.d_state).unwrap_or(16);
        let d_conv = ssm.and_then(|c| c.d_conv).or(raw.d_conv).unwrap_or(4);
        Ok((d_model, n_layers, d_state, d_conv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An original-format config names its mixer under ssm_cfg; a Mamba-2
    /// or Mamba-3 one is not read as Mamba-1.
    #[test]
    fn original_format_mixer_is_detected() {
        let m2 = br#"{"d_model":768,"n_layer":24,"vocab_size":50277,"ssm_cfg":{"layer":"Mamba2"}}"#;
        assert_eq!(parse_config_json(m2).unwrap().family, ModelFamily::Mamba2);
        let m3 =
            br#"{"d_model":768,"n_layer":12,"vocab_size":128256,"ssm_cfg":{"layer":"Mamba3"}}"#;
        assert_eq!(parse_config_json(m3).unwrap().family, ModelFamily::Mamba3);
        let m1 = br#"{"d_model":768,"n_layer":24,"vocab_size":50277,"ssm_cfg":{}}"#;
        assert_eq!(parse_config_json(m1).unwrap().family, ModelFamily::Mamba1);
        let mimo = br#"{"d_model":768,"n_layer":12,"vocab_size":128256,"ssm_cfg":{"layer":"Mamba3","is_mimo":true,"mimo_rank":4}}"#;
        assert_eq!(
            parse_config_json(mimo).unwrap().family,
            ModelFamily::Mamba3Mimo
        );
        let siso = br#"{"d_model":768,"n_layer":12,"vocab_size":128256,"ssm_cfg":{"layer":"Mamba3","is_mimo":false}}"#;
        assert_eq!(parse_config_json(siso).unwrap().family, ModelFamily::Mamba3);
    }

    #[test]
    fn original_format_tie_embeddings_is_honoured() {
        // The original format spells the flag `tie_embeddings`; reading
        // only the HF spelling defaulted an untied checkpoint to tied and
        // dropped the lm_head that had just been read.
        let json = br#"{"d_model":64,"n_layer":2,"vocab_size":128,"tie_embeddings":false}"#;
        let cfg = parse_config_json(json).unwrap();
        assert!(!cfg.tie_word_embeddings);

        let tied = br#"{"d_model":64,"n_layer":2,"vocab_size":128,"tie_embeddings":true}"#;
        assert!(parse_config_json(tied).unwrap().tie_word_embeddings);

        // Absent in both spellings: the reference default is tied.
        let silent = br#"{"d_model":64,"n_layer":2,"vocab_size":128}"#;
        assert!(parse_config_json(silent).unwrap().tie_word_embeddings);
    }

    #[test]
    fn original_format_pads_the_vocabulary() {
        // The original format stores the unpadded vocabulary and rounds it
        // up before allocating the embedding, so the reference's own
        // default lands 50280 rows on disk against a config that says
        // 50277.
        let json = br#"{"d_model":768,"n_layer":24,"vocab_size":50277}"#;
        assert_eq!(parse_config_json(json).unwrap().vocab_size, 50280);

        let explicit =
            br#"{"d_model":768,"n_layer":24,"vocab_size":50277,"pad_vocab_size_multiple":16}"#;
        assert_eq!(parse_config_json(explicit).unwrap().vocab_size, 50288);

        // The HF format stores the final row count and must not be padded.
        let hf = br#"{"model_type":"mamba","hidden_size":768,"num_hidden_layers":24,"state_size":16,"conv_kernel":4,"vocab_size":50277}"#;
        assert_eq!(parse_config_json(hf).unwrap().vocab_size, 50277);
    }

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
        // 50257 is not a multiple of the default 8: the reference model
        // built from this config holds 50264 embedding rows.
        assert_eq!(cfg.vocab_size, 50264);
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

        // M3 is DETECTED (not an "unsupported model_type" error) so the
        // loader can reject it with a precise family message.
        let m3 = br#"{"model_type":"mamba3","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(parse_config_json(m3).unwrap().family, ModelFamily::Mamba3);
        let m3s = br#"{"model_type":"mamba3_siso","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        assert_eq!(parse_config_json(m3s).unwrap().family, ModelFamily::Mamba3);
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

    /// An original-format Mamba-1 config keeps its mixer arguments under
    /// ssm_cfg, and they win over the defaults.
    #[test]
    fn original_format_mixer_arguments_come_from_ssm_cfg() {
        let json = br#"{"d_model":256,"n_layer":4,"vocab_size":1000,
            "ssm_cfg":{"d_state":32,"d_conv":3,"expand":3,"dt_rank":"auto"}}"#;
        let cfg = parse_config_json(json).unwrap();
        assert_eq!((cfg.d_state, cfg.d_conv, cfg.expand), (32, 3, 3));
    }

    /// The HF format sizes the mixer by intermediate_size, whatever its
    /// `expand` field says.
    #[test]
    fn hf_format_inner_width_comes_from_intermediate_size() {
        let json = br#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2,
            "state_size":8,"conv_kernel":4,"vocab_size":100,"expand":16,
            "intermediate_size":128}"#;
        assert_eq!(parse_config_json(json).unwrap().expand, 2);
    }

    /// Parts this build does not model are refused, never dropped: a
    /// checkpoint must not load and then run without them.
    #[test]
    fn unmodelled_backbone_parts_are_refused() {
        for (json, what) in [
            (
                &br#"{"model_type":"falcon_mamba","hidden_size":64,"num_hidden_layers":2,
                "state_size":8,"conv_kernel":4,"vocab_size":100}"#[..],
                "falcon_mamba",
            ),
            (
                br#"{"d_model":64,"n_layer":2,"vocab_size":100,"d_intermediate":128}"#,
                "d_intermediate",
            ),
            (
                br#"{"d_model":64,"n_layer":2,"vocab_size":100,"rms_norm":false}"#,
                "LayerNorm",
            ),
            (
                br#"{"d_model":64,"n_layer":2,"vocab_size":100,"attn_layer_idx":[0]}"#,
                "attn_layer_idx",
            ),
            (
                br#"{"d_model":64,"n_layer":2,"vocab_size":100,"ssm_cfg":{"dt_rank":7}}"#,
                "dt_rank",
            ),
        ] {
            let err = parse_config_json(json).unwrap_err();
            assert!(err.contains(what), "{what}: {err}");
        }
    }

    #[test]
    fn test_tie_word_embeddings_default() {
        let json = br#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2,"state_size":8,"conv_kernel":4,"vocab_size":100}"#;
        let cfg = parse_config_json(json).unwrap();
        assert!(cfg.tie_word_embeddings);
    }

    /// Python's `json.dump` writes the non-finite floats as bare tokens;
    /// they read as the floats they name, and the same words inside a
    /// string stay text.
    #[test]
    fn python_non_finite_tokens_are_read() {
        let json = br#"{"time_step_limit": [0.0, Infinity], "low": -Infinity, "odd": NaN,
            "name": "Infinity and \"NaN\" stay text", "model_type": "mamba2"}"#;
        let v: serde_json::Value = config_json(json).unwrap();
        let limit = v["time_step_limit"].as_array().unwrap();
        assert_eq!(json_f32(&limit[0]), Some(0.0));
        assert_eq!(json_f32(&limit[1]), Some(f32::INFINITY));
        assert_eq!(json_f32(&v["low"]), Some(f32::NEG_INFINITY));
        assert!(json_f32(&v["odd"]).is_some_and(f32::is_nan));
        assert_eq!(v["name"], "Infinity and \"NaN\" stay text");
        assert_eq!(json_f32(&v["name"]), None);
        assert_eq!(
            read_family_of(json).unwrap(),
            ModelFamily::Mamba2,
            "the family reads through the same parser"
        );
    }

    fn read_family_of(json: &[u8]) -> Result<ModelFamily, String> {
        detect_family(&config_json::<RawConfig>(json)?)
    }
}
