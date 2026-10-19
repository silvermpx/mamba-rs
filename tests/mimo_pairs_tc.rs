//! The MIMO pair terms on the tensor cores (`m3mb_pairs_tc`, what the TF32,
//! bf16 and f16 trainers run) against the CUDA-core pair kernels the f32
//! trainer runs. `MAMBA_RS_PAIRS_TC=off` is read once per process, so a
//! child process trains the same steps on the CUDA-core kernels, and every
//! weight gradient and the input gradient agree to the rounding of TF32
//! operands: over every rank's own chunk, heads and states off the slices of
//! 32, partial last chunks, groups, a full rope, the released head and
//! packings whose sequences start inside a chunk. The tensor-core gradients
//! are not the CUDA-core bits (the kernel ran) and repeat their bits from a
//! fresh trainer; the f32 trainer gives the same bits both ways.
//!
//! Run on a GPU box (without `MAMBA_RS_PAIRS_TC` set):
//!   cargo test --release --features cuda --test mimo_pairs_tc -- --test-threads=1
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::Command;

use mamba_rs::Varlen;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba3_mimo::gpu::Mamba3MimoGpuTrainer;
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};

/// The directory the child writes its gradients to.
const CHILD_OUT: &str = "MAMBA_RS_PAIRS_TC_TEST_OUT";
const INPUT_DIM: usize = 40;
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Case {
    label: &'static str,
    cfg: Mamba3MimoConfig,
    batch: usize,
    seq: usize,
    /// Where sequences start in each row, for a packing.
    starts: Option<Vec<Vec<usize>>>,
}

fn cases() -> Vec<Case> {
    let base = Mamba3MimoConfig {
        d_model: 64,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let case = |label, cfg, starts| Case {
        label,
        cfg,
        batch: 2,
        seq: 100,
        starts,
    };
    vec![
        case(
            "rank 1, full rope",
            Mamba3MimoConfig {
                mimo_rank: 1,
                rope_fraction: 1.0,
                ..base
            },
            None,
        ),
        case(
            "rank 2, groups 2, 24 states",
            Mamba3MimoConfig {
                mimo_rank: 2,
                ngroups: 2,
                d_state: 24,
                ..base
            },
            None,
        ),
        case(
            "rank 3, head 40",
            Mamba3MimoConfig {
                d_model: 120,
                headdim: 40,
                mimo_rank: 3,
                ..base
            },
            None,
        ),
        case(
            "rank 5, head 24, 40 states",
            Mamba3MimoConfig {
                d_model: 96,
                headdim: 24,
                d_state: 40,
                mimo_rank: 5,
                ..base
            },
            None,
        ),
        case(
            "rank 8, head 32, 128 states",
            Mamba3MimoConfig {
                headdim: 32,
                d_state: 128,
                mimo_rank: 8,
                ..base
            },
            None,
        ),
        case(
            "rank 16",
            Mamba3MimoConfig {
                mimo_rank: 16,
                ..base
            },
            None,
        ),
        case(
            "released head",
            Mamba3MimoConfig {
                headdim: 64,
                d_state: 128,
                ..base
            },
            None,
        ),
        case(
            "rank 1, a packing inside the chunks",
            Mamba3MimoConfig {
                mimo_rank: 1,
                ..base
            },
            Some(vec![vec![5, 50], vec![64, 90]]),
        ),
        case(
            "rank 2, a packing inside the chunks",
            Mamba3MimoConfig {
                mimo_rank: 2,
                d_state: 24,
                ..base
            },
            Some(vec![vec![10, 37, 70], vec![0, 45]]),
        ),
    ]
}

/// The case's weights, every vector drawn away from its default so the
/// biases, scales and rotations all reach the gradients.
fn drawn(cfg: &Mamba3MimoConfig, seed: u64) -> Mamba3MimoWeights {
    let mut w = Mamba3MimoWeights::init(cfg, INPUT_DIM, seed);
    for (l, lw) in w.layers.iter_mut().enumerate() {
        let s = seed * 100 + l as u64 * 20;
        for (v, k, base, scale) in [
            (&mut lw.b_norm_weight, 1, 1.0, 0.3),
            (&mut lw.c_norm_weight, 2, 1.0, 0.3),
            (&mut lw.b_bias, 3, 0.0, 0.5),
            (&mut lw.c_bias, 4, 0.0, 0.5),
            (&mut lw.d_param, 5, 1.0, 0.5),
            (&mut lw.mimo_x, 6, 0.0, 0.8),
            (&mut lw.mimo_z, 7, 1.0, 0.5),
            (&mut lw.mimo_o, 8, 0.0, 0.8),
        ] {
            let d = det(v.len(), s + k, scale);
            for (x, dv) in v.iter_mut().zip(d) {
                *x = base + dv;
            }
        }
        for v in &mut lw.out_proj_w {
            *v *= 3.0;
        }
    }
    w
}

/// Every weight gradient by name, then the input gradient.
fn named(g: Mamba3MimoWeights, d_in: Vec<f32>) -> Vec<(String, Vec<f32>)> {
    let mut out = vec![
        ("input_proj_w".to_string(), g.input_proj_w),
        ("input_proj_b".to_string(), g.input_proj_b),
        ("norm_f".to_string(), g.norm_f_weight),
    ];
    for (l, lw) in g.layers.into_iter().enumerate() {
        for (name, v) in [
            ("norm", lw.norm_weight),
            ("in_proj_w", lw.in_proj_w),
            ("dt_bias", lw.dt_bias),
            ("b_norm", lw.b_norm_weight),
            ("c_norm", lw.c_norm_weight),
            ("b_bias", lw.b_bias),
            ("c_bias", lw.c_bias),
            ("d", lw.d_param),
            ("mimo_x", lw.mimo_x),
            ("mimo_z", lw.mimo_z),
            ("mimo_o", lw.mimo_o),
            ("out_proj_w", lw.out_proj_w),
        ] {
            out.push((format!("layer{l}.{name}"), v));
        }
    }
    out.push(("input".to_string(), d_in));
    out
}

/// One forward and backward of the case in `dtype` from a fresh trainer.
fn gradients(case: &Case, dtype: WeightDtype) -> Vec<(String, Vec<f32>)> {
    let (batch, seq, dm) = (case.batch, case.seq, case.cfg.d_model);
    case.cfg.validate().unwrap();
    let w = drawn(&case.cfg, 11);
    let mut tr = Mamba3MimoGpuTrainer::new(
        &w,
        case.cfg,
        TrainSessionCfg::new(INPUT_DIM, batch, seq),
        GpuOptions::default().dtype(dtype),
    )
    .unwrap();
    if let Some(starts) = &case.starts {
        tr.set_varlen(Some(&Varlen::from_starts(starts, seq).unwrap()))
            .unwrap();
    }
    tr.load_scaler_state(256.0, 0).unwrap();
    let inputs = det(batch * seq * INPUT_DIM, 500, 1.0);
    let d_out = det(batch * seq * dm, 501, 1.0);
    let mut rows = vec![0f32; batch * seq * dm];
    tr.forward(&inputs, &mut rows).unwrap();
    let mut d_in = vec![0f32; batch * seq * INPUT_DIM];
    tr.backward(&d_out, Some(&mut d_in)).unwrap();
    assert!(!tr.overflowed(), "{dtype:?} {}: overflow", case.label);
    named(tr.grads().unwrap(), d_in)
}

fn out_file(dir: &Path, case: usize, dtype: WeightDtype) -> PathBuf {
    dir.join(format!("{case}_{dtype:?}.bin"))
}

/// The gradients of every case and dtype on the CUDA-core pair kernels,
/// written for the parent.
#[test]
#[ignore = "the child of tensor_core_pairs_match_the_cuda_core_pairs"]
fn pairs_child() {
    let dir = PathBuf::from(std::env::var(CHILD_OUT).expect("the parent's output directory"));
    for (i, case) in cases().iter().enumerate() {
        for dtype in DTYPES {
            let values: Vec<u8> = gradients(case, dtype)
                .into_iter()
                .flat_map(|(_, v)| v)
                .flat_map(f32::to_le_bytes)
                .collect();
            std::fs::write(out_file(&dir, i, dtype), values).unwrap();
        }
    }
}

// The tensor cores round each operand of the pair products to TF32 (11
// significant bits) where the CUDA-core kernels sum in f32; the bounds hold
// about ten times what these shapes measure, and a wrong index or mask
// fails by orders of magnitude.
#[test]
fn tensor_core_pairs_match_the_cuda_core_pairs() {
    let dir = std::env::temp_dir().join(format!("mamba-rs-pairs-tc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            "pairs_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MAMBA_RS_PAIRS_TC", "off")
        .env(CHILD_OUT, &dir)
        .output()
        .expect("spawn the CUDA-core child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed; 0 ignored"),
        "the CUDA-core child failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for (i, case) in cases().iter().enumerate() {
        for dtype in DTYPES {
            let got = gradients(case, dtype);
            let raw = std::fs::read(out_file(&dir, i, dtype)).unwrap();
            let all: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            let total: usize = got.iter().map(|(_, v)| v.len()).sum();
            assert_eq!(
                all.len(),
                total,
                "{dtype:?} {}: the child's sizes",
                case.label
            );
            let mut at = 0;
            let mut worst = (0f64, String::new());
            let mut moved = false;
            for (name, v) in &got {
                let want = &all[at..at + v.len()];
                at += v.len();
                moved |= bits(v) != bits(want);
                let e = rel_l2(v, want);
                if e > worst.0 {
                    worst = (e, name.clone());
                }
            }
            eprintln!(
                "{dtype:?} {}: worst gradient {:.2e} ({})",
                case.label, worst.0, worst.1
            );
            match dtype {
                WeightDtype::F32 => assert!(!moved, "f32 {}: the CUDA-core bits moved", case.label),
                _ => {
                    assert!(
                        moved,
                        "{dtype:?} {}: the CUDA-core bits, so the tensor cores never ran",
                        case.label
                    );
                    let bound = match dtype {
                        WeightDtype::Bf16 => 6e-2,
                        WeightDtype::F16 => 1.5e-2,
                        _ => 3e-2,
                    };
                    assert!(
                        worst.0 < bound,
                        "{dtype:?} {}: gradient {} {:.3e}",
                        case.label,
                        worst.1,
                        worst.0
                    );
                    let again = gradients(case, dtype);
                    for ((name, a), (_, b)) in got.iter().zip(&again) {
                        assert!(
                            bits(a) == bits(b),
                            "{dtype:?} {}: {name} moved from a fresh trainer",
                            case.label
                        );
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
