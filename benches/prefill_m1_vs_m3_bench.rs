//! M1 against M3 CPU prefill at the classifier page shape, with the
//! parameter counts printed beside the times. The M3 prefill contracts
//! are tested in tests/prefill_m3.rs.
use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::prefill::{
    Mamba3PrefillScratch, forward_mamba3_backbone_prefill_mode,
};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

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

/// M1-vs-M3 prefill head-to-head at the classifier center shape (manual:
/// --ignored). Same d_model / n_layers / input_dim / T; parameter counts are
/// printed alongside the times so the comparison is param-labeled, not
/// param-blind (M3 layers carry a different budget than M1 layers).
fn m1_vs_m3_prefill_head_to_head() {
    let seq_len = 4617usize;
    let input_dim = 1024usize;
    let reps = 3;

    // M1 at the classify center shape.
    {
        use mamba_rs::config::MambaConfig;
        use mamba_rs::inference::{PrefillScratch, forward_mamba_backbone_prefill_mode};
        use mamba_rs::ops::dims::MambaDims;
        use mamba_rs::state::MambaState;
        use mamba_rs::weights::MambaWeights;
        let cfg = MambaConfig {
            d_model: 384,
            n_layers: 24,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            scan_mode: mamba_rs::config::ScanMode::Sequential,
            rms_norm_eps: 1e-5,
        };
        let mut w = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
        for lw in w.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        let dims = MambaDims::from_config(&cfg, seq_len, input_dim);
        let input = det(seq_len * input_dim, 0xAF, 0.05);
        let mut out = vec![0.0f32; seq_len * cfg.d_model];
        let n_params = w.param_count(input_dim, &cfg);
        for mode in [PrefillMode::Single, PrefillMode::Parallel] {
            let mut state = MambaState::zeros(cfg.n_layers, cfg.d_inner(), cfg.d_state, cfg.d_conv);
            let mut scratch = PrefillScratch::new(&dims);
            forward_mamba_backbone_prefill_mode(
                &mut out,
                &input,
                &w,
                &mut state,
                &mut scratch,
                &dims,
                mode,
            );
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
            eprintln!("M1 d384x24 ({n_params} params) prefill {mode:?}: {ms:.1} ms/page");
        }
    }

    // M3 at the matched outer shape (d384 x 24 layers).
    {
        let cfg = Mamba3Config {
            d_model: 384,
            d_state: 32,
            expand: 2,
            headdim: 32,
            ngroups: 1,
            n_layers: 24,
            rope_fraction: 0.5,
            a_floor: 0.0625,
            is_outproj_norm: true,
            ..Mamba3Config::default()
        };
        let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
        let dims = Mamba3Dims::from_config(&cfg, seq_len);
        let input = det(seq_len * input_dim, 0xAF, 0.05);
        let mut out = vec![0.0f32; seq_len * cfg.d_model];
        let n_params: usize = w.input_proj_w.len()
            + w.input_proj_b.len()
            + w.norm_f_weight.len()
            + w.layers
                .iter()
                .map(|l| {
                    l.norm_weight.len()
                        + l.in_proj_w.len()
                        + l.dt_bias.len()
                        + l.b_norm_weight.len()
                        + l.c_norm_weight.len()
                        + l.b_bias.len()
                        + l.c_bias.len()
                        + l.d_param.len()
                        + l.norm_gate_weight.len()
                        + l.out_proj_w.len()
                })
                .sum::<usize>();
        for mode in [PrefillMode::Single, PrefillMode::Parallel] {
            let mut state = Mamba3State::zeros(&cfg);
            let mut scratch = Mamba3PrefillScratch::new(&dims, input_dim);
            forward_mamba3_backbone_prefill_mode(
                &mut out,
                &input,
                &w,
                &mut state.layers,
                &mut scratch,
                &dims,
                mode,
            );
            let start = std::time::Instant::now();
            for _ in 0..reps {
                state.reset();
                forward_mamba3_backbone_prefill_mode(
                    &mut out,
                    &input,
                    &w,
                    &mut state.layers,
                    &mut scratch,
                    &dims,
                    mode,
                );
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            eprintln!("M3 d384x24 ({n_params} params) prefill {mode:?}: {ms:.1} ms/page");
        }
    }
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target>  -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("m1_vs_m3_prefill_head_to_head") {
        m1_vs_m3_prefill_head_to_head();
    }
}
