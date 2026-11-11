//! The MIMO chunked scan on the tensor cores (`m3m_chunk_scan_tc`) against
//! the CUDA-core scans it stands in for on the TF32, bf16 and f16 lanes. Its
//! products round their operands to TF32, so the training forward's rows,
//! every weight gradient, the input gradient and a prompt's decode steps
//! after it stay within that rounding of the CUDA-core scans', and repeat
//! their own bits; the f32 lane keeps the CUDA-core scans and their bits.
//! `MAMBA_RS_MIMO_SCAN_TC` is read once per process, so a child process runs
//! the same steps on the CUDA-core scans (`off`). The cases: every rank's
//! own chunk, heads and states off the tiles, a head wider than a block's
//! channels, partial last chunks, groups, a full rope, the released head
//! over a long window and packings whose sequences start inside a chunk.
//!
//! Run on a GPU box (without `MAMBA_RS_MIMO_SCAN_TC` set):
//!   cargo test --release --features cuda --test mimo_scan_tc -- --test-threads=1
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::Command;

use mamba_rs::Varlen;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba3_mimo::gpu::{Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer};
use mamba_rs::mamba3_mimo::{Mamba3MimoConfig, Mamba3MimoWeights};

/// The directory the child writes its values to.
const CHILD_OUT: &str = "MAMBA_RS_MIMO_SCAN_TC_TEST_OUT";

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Case {
    label: &'static str,
    cfg: Mamba3MimoConfig,
    batch: usize,
    seq: usize,
    /// Where sequences start in each row, for a packing of the training
    /// window.
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
    let case = |label, cfg, seq, starts| Case {
        label,
        cfg,
        batch: 2,
        seq,
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
            700,
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
            300,
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
            150,
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
            150,
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
            100,
            None,
        ),
        case(
            "rank 16",
            Mamba3MimoConfig {
                mimo_rank: 16,
                ..base
            },
            100,
            None,
        ),
        case(
            "head 96, wider than a block's channels",
            Mamba3MimoConfig {
                d_model: 96,
                headdim: 96,
                d_state: 32,
                ..base
            },
            100,
            None,
        ),
        case(
            "released head",
            Mamba3MimoConfig {
                headdim: 64,
                d_state: 128,
                ..base
            },
            100,
            None,
        ),
        Case {
            label: "released head, a long window",
            cfg: Mamba3MimoConfig {
                headdim: 64,
                d_state: 128,
                ..base
            },
            batch: 1,
            seq: 700,
            starts: None,
        },
        case(
            "rank 1, a packing inside the chunks",
            Mamba3MimoConfig {
                mimo_rank: 1,
                ..base
            },
            700,
            Some(vec![vec![5, 50, 333], vec![64, 90, 512, 600]]),
        ),
        case(
            "rank 4, a packing inside the chunks",
            Mamba3MimoConfig {
                headdim: 32,
                d_state: 64,
                ..base
            },
            300,
            Some(vec![vec![10, 37, 129, 200], vec![0, 45, 128]]),
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

/// The case in `dtype`: a forward and backward from a fresh trainer (the
/// forward rows, every weight gradient and the input gradient), then a
/// prompt of the window on a fresh backbone and three decode steps after
/// it.
fn outputs(case: &Case, dtype: WeightDtype) -> Vec<(String, Vec<f32>)> {
    let (batch, seq, dm) = (case.batch, case.seq, case.cfg.d_model);
    case.cfg.validate().unwrap();
    let w = drawn(&case.cfg, 11);
    let opts = GpuOptions::default().dtype(dtype);
    let mut tr = Mamba3MimoGpuTrainer::new(
        &w,
        case.cfg,
        TrainSessionCfg::new(INPUT_DIM, batch, seq),
        opts,
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
    let mut all = vec![("rows".to_string(), rows)];
    all.extend(named(tr.grads().unwrap(), d_in));
    let mut bb = Mamba3MimoGpuBackbone::new(&w, case.cfg, INPUT_DIM, batch, opts).unwrap();
    bb.prefill(&inputs, seq).unwrap();
    for k in 0..3 {
        let step_in = det(batch * INPUT_DIM, 600 + k, 1.0);
        let mut step_out = vec![0f32; batch * dm];
        bb.step(&step_in, &mut step_out).unwrap();
        all.push((format!("decode{k}"), step_out));
    }
    all
}

fn out_file(dir: &Path, case: usize, dtype: WeightDtype) -> PathBuf {
    dir.join(format!("{case}_{dtype:?}.bin"))
}

/// Every case and storage under the child's `MAMBA_RS_MIMO_SCAN_TC`,
/// written for the parent.
#[test]
#[ignore = "invoked by the_tensor_core_scan_tracks_the_cuda_core_scans"]
fn child_outputs() {
    let dir = PathBuf::from(std::env::var(CHILD_OUT).expect("the parent's output directory"));
    for (i, case) in cases().iter().enumerate() {
        for dtype in DTYPES {
            let values: Vec<u8> = outputs(case, dtype)
                .into_iter()
                .flat_map(|(_, v)| v)
                .flat_map(f32::to_le_bytes)
                .collect();
            std::fs::write(out_file(&dir, i, dtype), values).unwrap();
        }
    }
}

/// The child's values under `setting`, case by case and storage by storage.
fn child(setting: &str) -> Vec<Vec<Vec<f32>>> {
    let dir = std::env::temp_dir().join(format!(
        "mamba-rs-mimo-scan-tc-{setting}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let output = Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            "child_outputs",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MAMBA_RS_MIMO_SCAN_TC", setting)
        .env(CHILD_OUT, &dir)
        .output()
        .expect("spawn the child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed; 0 ignored"),
        "the {setting} child failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let all = cases()
        .iter()
        .enumerate()
        .map(|(i, _)| {
            DTYPES
                .iter()
                .map(|&dtype| {
                    std::fs::read(out_file(&dir, i, dtype))
                        .unwrap()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect()
                })
                .collect()
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    all
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

// The tensor-core products round each operand to TF32, a relative step of
// 2^-11; through the scan, its gate and the layers that is a few times that
// at most, the bounds above it and far below what a wrong index or mask
// gives. The f32 lane takes the CUDA-core scans, so its bits are the
// child's.
#[test]
fn the_tensor_core_scan_tracks_the_cuda_core_scans() {
    let cuda_cores = child("off");
    let mut checked = 0;
    for (i, case) in cases().iter().enumerate() {
        for (d, &dtype) in DTYPES.iter().enumerate() {
            let want = &cuda_cores[i][d];
            let got = outputs(case, dtype);
            let again = outputs(case, dtype);
            let total: usize = got.iter().map(|(_, v)| v.len()).sum();
            assert_eq!(
                want.len(),
                total,
                "{dtype:?} {}: the child's sizes",
                case.label
            );
            let bound = match dtype {
                WeightDtype::F32 => 0.0,
                WeightDtype::Bf16 => 2e-2,
                _ => 5e-3,
            };
            let mut at = 0;
            let mut worst = (0f64, String::new());
            for ((name, v), (_, v2)) in got.iter().zip(&again) {
                assert_eq!(
                    bits(v),
                    bits(v2),
                    "{dtype:?} {}: {name} repeats",
                    case.label
                );
                let w = &want[at..at + v.len()];
                at += v.len();
                if dtype == WeightDtype::F32 {
                    assert_eq!(bits(v), bits(w), "{dtype:?} {}: {name} moved", case.label);
                    continue;
                }
                let e = rel_l2(v, w);
                if e > worst.0 {
                    worst = (e, name.clone());
                }
                checked += v.len();
            }
            assert!(
                worst.0 <= bound,
                "{dtype:?} {}: {} {:.3e}",
                case.label,
                worst.1,
                worst.0
            );
            eprintln!(
                "{dtype:?} {}: worst {:.2e} ({})",
                case.label, worst.0, worst.1
            );
        }
    }
    eprintln!("the tensor-core scan checked {checked} values against the CUDA-core scans");
}
