//! The denominator probe: what is this box's tensor-core ceiling, and
//! where does everything actually sit against it?
//!
//! The program's efficiency claims have been divided by numbers that
//! were never measured here. This instrument settles the denominator
//! from first principles on the box itself:
//!
//! - a null-memory mma.sync issue-rate kernel (register fragments, no
//!   loads in the loop) measures the pure tensor pipe rate for
//!   bf16 with f32 accumulate - the contract's datapath - and its
//!   f16-with-f16-accumulate twin. The RATIO is the model decider: 2x
//!   means f32 accumulation halves throughput on this part (the
//!   512 FLOP/clk/SM model), 1x means it does not;
//! - event-timed arms at the fat training shapes: cuBLAS PEDANTIC,
//!   cuBLAS fast (tensor cores - the first such measurement on this
//!   box), and the forced deterministic Tile128;
//! - every rate printed against both candidate ceilings.
#![cfg(feature = "cuda")]

mod common;

use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    TcFwdOperands, TcTile, gemm_bi_forward_tc_with_tile,
};

const RATE_SRC: &str = r#"
// Null-memory tensor-pipe rate kernels: eight independent accumulator
// sets per warp, operands live in registers, nothing is loaded in the
// loop. The final store keeps the chains observable.
extern "C" __global__ void mma_rate_bf16_f32acc(float* out, int iters) {
    unsigned t = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned a0 = t * 0x9E3779B9u + 1u;
    unsigned a1 = a0 ^ 0x55555555u;
    unsigned a2 = a0 + 3u;
    unsigned a3 = a1 + 7u;
    unsigned b0 = a2 ^ 0x33333333u;
    unsigned b1 = a3 ^ 0x0F0F0F0Fu;
    float d[8][4];
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 4; e++) d[s][e] = (float)((t + s + e) & 7);
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(d[s][0]), "+f"(d[s][1]), "+f"(d[s][2]), "+f"(d[s][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float acc = 0.0f;
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 4; e++) acc += d[s][e];
    out[t] = acc;
}

extern "C" __global__ void mma_rate_f16_f16acc(float* out, int iters) {
    unsigned t = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned a0 = t * 0x9E3779B9u + 1u;
    unsigned a1 = a0 ^ 0x55555555u;
    unsigned a2 = a0 + 3u;
    unsigned a3 = a1 + 7u;
    unsigned b0 = a2 ^ 0x33333333u;
    unsigned b1 = a3 ^ 0x0F0F0F0Fu;
    unsigned d[8][2];
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 2; e++) d[s][e] = (t + s + e) & 0x00070007u;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                : "+r"(d[s][0]), "+r"(d[s][1])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    unsigned acc = 0;
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 2; e++) acc ^= d[s][e];
    out[t] = (float)(acc & 0xFF);
}

extern "C" __global__ void mma_rate_f16_f32acc(float* out, int iters) {
    unsigned t = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned a0 = t * 0x9E3779B9u + 1u;
    unsigned a1 = a0 ^ 0x55555555u;
    unsigned a2 = a0 + 3u;
    unsigned a3 = a1 + 7u;
    unsigned b0 = a2 ^ 0x33333333u;
    unsigned b1 = a3 ^ 0x0F0F0F0Fu;
    float d[8][4];
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 4; e++) d[s][e] = (float)((t + s + e) & 7);
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(d[s][0]), "+f"(d[s][1]), "+f"(d[s][2]), "+f"(d[s][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float acc = 0.0f;
    for (int s = 0; s < 8; s++)
        for (int e = 0; e < 4; e++) acc += d[s][e];
    out[t] = acc;
}
"#;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 0.5 - 0.25
        })
        .collect()
}

#[test]
#[ignore = "record-lane instrument (GPU, quiet card)"]
fn denominator_probe() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    let st = &ctx.stream;
    let cuda = st.context();

    // --- Arm 1: the null-memory tensor-pipe rate ---
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(RATE_SRC, opts).expect("rate probe compiles");
    let module = cuda.load_module(ptx).expect("rate module");
    let blocks = 142u32 * 3;
    let threads = 256u32;
    let warps = (blocks * threads / 32) as f64;
    let iters: i32 = 60_000;
    let flop_per_mma = 2.0 * 16.0 * 8.0 * 16.0;
    let total_flop = warps * iters as f64 * 8.0 * flop_per_mma;
    let out = DtypedBuf::zeros(st, (blocks * threads) as usize, WeightDtype::F32).unwrap();
    let mk_ev = || {
        cuda.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let mut rates = Vec::new();
    for name in [
        "mma_rate_bf16_f32acc",
        "mma_rate_f16_f32acc",
        "mma_rate_f16_f16acc",
    ] {
        let f = module.load_function(name).expect("rate kernel");
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let op = out.cached_ptr();
        // warm
        for _ in 0..2 {
            let mut b = st.launch_builder(&f);
            b.arg(&op);
            b.arg(&iters);
            unsafe { b.launch(cfg) }.expect("rate launch");
        }
        st.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let s_ev = mk_ev();
            let e_ev = mk_ev();
            s_ev.record(st).unwrap();
            let mut b = st.launch_builder(&f);
            b.arg(&op);
            b.arg(&iters);
            unsafe { b.launch(cfg) }.expect("rate launch");
            e_ev.record(st).unwrap();
            st.synchronize().unwrap();
            let ms = f64::from(s_ev.elapsed_ms(&e_ev).unwrap());
            best = best.min(ms);
        }
        let tflops = total_flop / (best / 1e3) / 1e12;
        println!("RATE {name}: {tflops:.1} TFLOPS ({best:.2} ms)");
        common::evidence::record(
            "gemm_denominator_probe",
            "mma_rate",
            name,
            &format!("{tflops:.1}"),
        )
        .expect("acceptance evidence");
        rates.push((name, tflops));
    }
    let f32acc = rates[0].1.max(rates[1].1);
    let f16acc = rates[2].1;
    println!(
        "RATIO f16acc/f32acc = {:.2} -> f32 accumulation {} throughput on this part",
        f16acc / f32acc,
        if f16acc / f32acc > 1.5 {
            "HALVES"
        } else {
            "does NOT halve"
        }
    );

    // --- Arm 2: the fat shapes, event-timed, three routes ---
    let shapes = [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (4096, 768, 3072),
        (2048, 1536, 1536),
        (2048, 2304, 768),
        (2048, 768, 2304),
    ];
    let dt = WeightDtype::Bf16;
    println!(
        "\nshape                | tile128    ped       fast-TC  | TFLOPS t128/fast | % of {f32acc:.0}"
    );
    for (m, k, n) in shapes {
        let a = DtypedBuf::zeros(st, m * k, dt).unwrap();
        a.upload_f32(st, &det(m * k, 11)).unwrap();
        let w = DtypedBuf::zeros(st, k * n, dt).unwrap();
        w.upload_f32(st, &det(k * n, 22)).unwrap();
        let c = DtypedBuf::zeros(st, m * n, dt).unwrap();
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let tp = |b: &DtypedBuf| TypedPtr {
            ptr: b.cached_ptr(),
            dtype: dt,
        };
        let time_us = |run: &dyn Fn()| -> f64 {
            for _ in 0..10 {
                run();
            }
            st.synchronize().unwrap();
            let mut best = f64::MAX;
            for _ in 0..4 {
                let s_ev = mk_ev();
                let e_ev = mk_ev();
                s_ev.record(st).unwrap();
                for _ in 0..30 {
                    run();
                }
                e_ev.record(st).unwrap();
                st.synchronize().unwrap();
                best = best.min(f64::from(s_ev.elapsed_ms(&e_ev).unwrap()) * 1000.0 / 30.0);
            }
            best
        };
        let t128 = time_us(&|| {
            gemm_bi_forward_tc_with_tile(
                st,
                &ctx.kernels,
                &TcFwdOperands {
                    y: tp(&c),
                    x: tp(&a),
                    w: tp(&w),
                    bias_ptr: 0,
                },
                (m, k, n),
                TcTile::Tile128,
            )
            .expect("tile128");
        });
        let cublas = |fast: bool| -> f64 {
            ctx.set_batch_invariant(false);
            ctx.set_fast_gemm(fast);
            let t = time_us(&|| {
                gpu_gemm_typed_forward_raw(&ctx, tp(&c), tp(&a), tp(&w), None, (m, k, n))
                    .expect("cublas");
            });
            ctx.set_fast_gemm(false);
            ctx.set_batch_invariant(true);
            t
        };
        let t_ped = cublas(false);
        let t_fast = cublas(true);
        let tf = |t: f64| flop / (t / 1e6) / 1e12;
        println!(
            "M{m:<5} K{k:<5} N{n:<5} | {t128:8.1} {t_ped:8.1} {t_fast:8.1} us | {:6.1} / {:6.1} | {:4.0}% / {:4.0}%",
            tf(t128),
            tf(t_fast),
            tf(t128) / f32acc * 100.0,
            tf(t_fast) / f32acc * 100.0,
        );
        for (arm, t) in [("tile128", t128), ("ped", t_ped), ("fast_tc", t_fast)] {
            common::evidence::record(
                "gemm_denominator_probe",
                arm,
                &format!("M{m}K{k}N{n}"),
                &format!("{:.1}", tf(t)),
            )
            .expect("acceptance evidence");
        }
    }
}
