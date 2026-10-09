//! The float64 oracles' goldens for the official checkpoints
//! (`tools/oracles/m{2,3}_oracle.py`), shared by the checkpoint parity tests.

use std::path::{Path, PathBuf};

/// Where a family's checkpoints and goldens live: the directory the
/// environment variable `env` names, or `$HOME/<home_sub>`.
pub fn official_dir(env: &str, home_sub: &str) -> PathBuf {
    std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(home_sub))
}

pub struct Golden {
    pub ids: Vec<u32>,
    pub logsumexp: Vec<f64>,
    pub target_logprob: Vec<f64>,
    pub top_ids: Vec<Vec<u32>>,
    pub last_rows_from: usize,
    pub last_logits: Vec<f64>,
    pub greedy: Vec<u32>,
    pub greedy_margin: Vec<f64>,
}

pub fn load_golden(dir: &Path) -> Golden {
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let u32s = |k: &str| -> Vec<u32> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let f64s = |k: &str| -> Vec<f64> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let raw = std::fs::read(dir.join("last_logits.f64")).unwrap();
    Golden {
        ids: u32s("ids"),
        logsumexp: f64s("logsumexp"),
        target_logprob: f64s("target_logprob"),
        top_ids: json["top_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect(),
        last_rows_from: json["last_rows_from"].as_u64().unwrap() as usize,
        last_logits: raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        greedy: u32s("greedy"),
        greedy_margin: f64s("greedy_margin"),
    }
}

pub fn logsumexp(v: &[f32]) -> f64 {
    let m = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    m + v
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln()
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0 as u32
}
