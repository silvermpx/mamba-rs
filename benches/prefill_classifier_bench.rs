//! CPU prefill latency at the vision-classifier page shape (T = 4621,
//! d384 x 24 layers) for the single and parallel prefill modes.
//! The prefill contracts are tested in tests/prefill.rs.
use mamba_rs::config::MambaConfig;
use mamba_rs::inference::{PrefillMode, PrefillScratch, forward_mamba_backbone_prefill_mode};
use mamba_rs::ops::dims::MambaDims;
use mamba_rs::state::MambaState;
use mamba_rs::weights::MambaWeights;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn init_weights(cfg: &MambaConfig, input_dim: usize, seed: u64) -> MambaWeights {
    let mut w = MambaWeights::init(cfg, input_dim, seed);
    for lw in w.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    w
}

/// Latency probe at the classifier center shape (manual: --ignored).
/// Prints ms/page for Single and Parallel modes plus the per-step loop
/// extrapolation — the first measured numbers for the doc-101 serve tier.
fn prefill_bench_classifier_shape() {
    let cfg = MambaConfig {
        d_model: 384,
        n_layers: 24,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let input_dim = 1024usize;
    // Production patchify T_TOTAL: 57x81 patches + 4 register tokens.
    let seq_len = 4621usize;
    let w = init_weights(&cfg, input_dim, 0xC0FFEE);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
    let (di, ds, dc, nl) = (cfg.d_inner(), cfg.d_state, cfg.d_conv, cfg.n_layers);
    let input = det(seq_len * input_dim, 0xAF, 0.05);
    let mut out = vec![0.0f32; seq_len * cfg.d_model];

    for mode in [PrefillMode::Single, PrefillMode::Parallel] {
        let mut state = MambaState::zeros(nl, di, ds, dc);
        let mut scratch = PrefillScratch::new(&dims);
        // Warm-up.
        forward_mamba_backbone_prefill_mode(
            &mut out,
            &input,
            &w,
            &mut state,
            &mut scratch,
            &dims,
            mode,
        );
        let reps = 3;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            state.reset();
            forward_mamba_backbone_prefill_mode(
                &mut out,
                &input,
                &w,
                &mut state,
                &mut scratch,
                &dims,
                mode,
            );
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        eprintln!("prefill {mode:?}: {ms:.1} ms/page (T={seq_len}, d384x24)");
    }
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target>  -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("prefill_bench_classifier_shape") {
        prefill_bench_classifier_shape();
    }
}
