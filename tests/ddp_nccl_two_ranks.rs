//! Live two-process NCCL data parallelism: the supervisor self-spawns
//! two ranks on two GPUs, each trains the same model on its own
//! micro-batches through `backward_step_dist`, and the final weights
//! must match the single-process emulated-world oracle BIT FOR BIT.
//! (NCCL hard-refuses two ranks on one device — "Duplicate GPU
//! detected" — so this needs a genuinely multi-GPU box; on a single-GPU
//! machine the test reports itself skipped before spawning anything.
//! (The path up to the communicator init — spawn, environment contract,
//! rendezvous, unique-id exchange, version preflight — was exercised on
//! a one-GPU box during development by letting NCCL itself reject the
//! duplicate device; this test does not repeat that probe.)
//! The bitwise claim is exact at world size 2: a two-addend sum has one
//! association, and IEEE-754 addition is commutative, so the library
//! collective cannot produce different bits than the house fold.
//!
//! Run manually (spawns processes, needs libnccl + a GPU):
//! `cargo test --features cuda,hf,gemm-blas,nccl --test ddp_nccl_two_ranks -- --ignored --nocapture`

#![cfg(all(feature = "cuda", feature = "nccl"))]

use mamba_rs::dist::{Bootstrap, Devices, DistConfig, EmulatedWorld, Rendezvous, bootstrap};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::BackwardOpts;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

const ROUNDS: usize = 3;
const WORLD: usize = 2;

fn cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    }
}

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn weight_bits(w: &Mamba3Weights) -> Vec<u32> {
    let mut out = Vec::new();
    let mut push = |v: &[f32]| out.extend(v.iter().map(|x| x.to_bits()));
    push(&w.input_proj_w);
    push(&w.input_proj_b);
    for lw in &w.layers {
        push(&lw.norm_weight);
        push(&lw.in_proj_w);
        push(&lw.dt_bias);
        push(&lw.b_norm_weight);
        push(&lw.c_norm_weight);
        push(&lw.b_bias);
        push(&lw.c_bias);
        push(&lw.d_param);
        push(&lw.norm_gate_weight);
        push(&lw.out_proj_w);
    }
    push(&w.norm_f_weight);
    out
}

fn base_weights() -> Mamba3Weights {
    let c = cfg();
    let mut w = Mamba3Weights::init(&c, c.d_model, 0xC0FFEE);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w
}

fn build_trainer(device: usize) -> Mamba3Trainer {
    let c = cfg();
    Mamba3Trainer::new_with_dtype(
        device,
        &base_weights(),
        c.clone(),
        c.d_model,
        1,
        64,
        WeightDtype::Bf16,
    )
    .unwrap()
}

fn micro_inputs(round: usize, rank: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        det(n, 0x1000 + (round * WORLD + rank) as u32),
        det(n, 0x2000 + (round * WORLD + rank) as u32),
    )
}

/// The single-process oracle: the same schedule through the emulated
/// world's fold (accumulate, sum ascending, mean, apply).
fn emulated_final_bits() -> Vec<u32> {
    let c = cfg();
    let n = 64 * c.d_model;
    let ew = EmulatedWorld::new(WORLD).unwrap();
    let mut ranks: Vec<Mamba3Trainer> = (0..WORLD).map(|_| build_trainer(0)).collect();
    let mut out = vec![0.0f32; n];
    for round in 0..ROUNDS {
        let mut arenas = Vec::with_capacity(WORLD);
        for (r, t) in ranks.iter_mut().enumerate() {
            let (input, d_temporal) = micro_inputs(round, r, n);
            t.forward(&input, &mut out).unwrap();
            t.backward_step(
                &d_temporal,
                BackwardOpts::default().with_accumulate_only(true),
            )
            .unwrap();
            let stream = t.ctx().stream.clone();
            arenas.push(t.grad_arena().to_cpu(&stream).unwrap());
        }
        ew.all_reduce_mean(&mut arenas, None).unwrap();
        for (t, reduced) in ranks.iter_mut().zip(&arenas) {
            let stream = t.ctx().stream.clone();
            t.grad_arena().upload(&stream, reduced).unwrap();
            t.apply_step(None).unwrap();
        }
    }
    weight_bits(&ranks[0].snapshot_master().unwrap())
}

fn digest_path(rank: usize) -> std::path::PathBuf {
    let dir = std::env::var("MAMBA_RS_RENDEZVOUS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mamba-rs-ddp2"));
    let job = std::env::var("MAMBA_RS_JOB_ID").unwrap_or_else(|_| "job".into());
    dir.join(job).join(format!("digest-rank-{rank}"))
}

#[test]
#[ignore]
fn ddp_two_ranks_one_gpu_matches_emulated() {
    // Supervisor-side skip: children inherit the rank contract and never
    // take this branch.
    if std::env::var("MAMBA_RS_RANK").is_err() {
        let n = cudarc::driver::CudaContext::device_count().unwrap_or(0);
        if n < 2 {
            eprintln!("SKIPPED: needs 2 GPUs (found {n}) — NCCL refuses duplicate devices");
            return;
        }
    }
    let dir = std::env::temp_dir().join("mamba-rs-ddp2");
    let job = format!("run-{}", std::process::id());
    let dist_cfg = DistConfig::default()
        .with_devices(Devices::List(vec![0, 1]))
        .with_seed(7)
        .with_rendezvous(Rendezvous::File {
            dir: dir.clone(),
            job_id: job.clone(),
        });

    match bootstrap(dist_cfg).unwrap() {
        Bootstrap::Rank(ctx) => {
            // Child process: train through the live communicator, write
            // the final weight bits for the supervisor, and exit before
            // the harness tries to run anything else.
            let n = 64 * cfg().d_model;
            let mut t = build_trainer(ctx.device_ordinal());
            let mut out = vec![0.0f32; n];
            for round in 0..ROUNDS {
                let (input, d_temporal) = micro_inputs(round, ctx.rank(), n);
                t.forward(&input, &mut out).unwrap();
                t.backward_step_dist(&d_temporal, BackwardOpts::default(), &ctx)
                    .unwrap();
            }
            let bits = weight_bits(&t.snapshot_master().unwrap());
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            std::fs::write(digest_path(ctx.rank()), bytes).unwrap();
            ctx.barrier().unwrap();
            std::process::exit(0);
        }
        Bootstrap::Supervisor(status) => {
            assert!(status.all_ok(), "rank exit codes: {:?}", status.exit_codes);
            let read = |rank: usize| -> Vec<u32> {
                let p = dir.join(&job).join(format!("digest-rank-{rank}"));
                let bytes =
                    std::fs::read(&p).unwrap_or_else(|e| panic!("digest {}: {e}", p.display()));
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let r0 = read(0);
            let r1 = read(1);
            assert_eq!(r0, r1, "ranks diverged");
            let oracle = emulated_final_bits();
            assert_eq!(
                r0, oracle,
                "live NCCL world diverged from the emulated oracle \
                 (at world size 2 the sum has one association and must match bitwise)"
            );
            let _ = std::fs::remove_dir_all(dir.join(&job));
        }
    }
}
