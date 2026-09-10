//! Isolated timing of the M3 backward kernels at the production shape
//! (B = 8, T = 1300, d_model 384). The kernels' parity and bit hashes are
//! tested in tests/m3_final_grads_unit_parity.rs.
#![cfg(feature = "cuda")]

#[path = "../tests/common/arch.rs"]
mod arch;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;

fn det_rand(n: usize, seed: u32) -> Vec<f32> {
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

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn upload_f32(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn m3_kernels_isolated_bench() {
    use std::time::Instant;

    const CB: usize = 8;
    const CT: usize = 1300;
    const CNH: usize = 48;
    const CHD: usize = 16;
    const CDS: usize = 16;
    const CCS: usize = 64;
    const CNA: usize = 4;
    let n_chunks = CT.div_ceil(CCS);
    let d_inner = CNH * CHD;
    let layers = 24f64;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    // Production JIT injects the 16-granular state cap; the default-64
    // helper would inflate the per-thread register arrays and overstate
    // these kernels vs the real trainer.
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), CDS.max(16))
            .unwrap();

    let n_q = CB * CT * CNH * CDS;
    let n_v = CB * CT * d_inner;
    let n_th = CB * CT * CNH;

    let q_f32 = upload_f32(&ctx, &det_rand(n_q, 0xA001));
    let ks_f32 = upload_f32(&ctx, &det_rand(n_q, 0xA002));
    let v_f32 = upload_f32(&ctx, &det_rand(n_v, 0xA003));
    let do_f32 = upload_f32(&ctx, &det_rand(n_v, 0xA004));
    let dcs_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH * CCS, 0xA005));
    let dcs_sum_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH, 0xA006));
    let qk_buf = upload_f32(&ctx, &det_rand(n_th, 0xA007));
    let ssm_buf = upload_f32(&ctx, &det_rand(CB * n_chunks * CNH * CHD * CDS, 0xA008));
    let d_buf = upload_f32(&ctx, &det_rand(CNH, 0xA009));

    let dq = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dk = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dv = GpuBuffer::zeros(&ctx.stream, n_v).unwrap();
    let dadt = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dqk = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dd = GpuBuffer::zeros(&ctx.stream, CB * n_chunks * CNH).unwrap();
    ctx.stream.synchronize().unwrap();

    // Production tier ladder (mirrors backward.rs).
    let legacy_floats = 2 * CCS * CDS + 2 * CCS * CHD + 4 * CCS + 2 * CHD * CDS;
    let mats_floats = legacy_floats + CCS * (CCS - 1) * 3 / 2 + 2 * CCS;
    let cap_floats = 99 * 1024 / 4;
    let (per_head_floats, use_mats_i): (usize, i32) = if mats_floats <= cap_floats {
        (mats_floats, 1)
    } else {
        (legacy_floats, 0)
    };
    let t_split = 16u32.min((1024 / CHD) as u32);
    let enter = GpuBuffer::zeros(&ctx.stream, CB * n_chunks * CNH * CHD * CDS).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: (CNH as u32, CB as u32, n_chunks as u32),
        block_dim: (CHD as u32, t_split, 1),
        shared_mem_bytes: (per_head_floats * 4) as u32,
    };
    eprintln!(
        "tier: t_split={t_split} use_pair_mats={use_mats_i} smem={}B",
        per_head_floats * 4
    );

    let (bi, ti, nhi, hdi, dsi, csi) = (
        CB as i32, CT as i32, CNH as i32, CHD as i32, CDS as i32, CCS as i32,
    );

    let time_it = |label: &str, launch: &dyn Fn()| {
        for _ in 0..3 {
            launch();
        }
        ctx.stream.synchronize().unwrap();
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            launch();
        }
        ctx.stream.synchronize().unwrap();
        let ms = 1e3 * t0.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "{label}: {ms:.3} ms/launch -> {:.1} ms/step(24L)",
            ms * layers
        );
    };

    // m3_dqkv f32
    time_it("m3_dqkv f32", &|| {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_dqkv);
        let args = [
            dq.cached_ptr(),
            dk.cached_ptr(),
            dv.cached_ptr(),
            dadt.cached_ptr(),
            dqk.cached_ptr(),
            dd.cached_ptr(),
            q_f32.cached_ptr(),
            ks_f32.cached_ptr(),
            v_f32.cached_ptr(),
            dcs_buf.cached_ptr(),
            dcs_sum_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            ssm_buf.cached_ptr(),
            do_f32.cached_ptr(),
            d_buf.cached_ptr(),
            enter.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&use_mats_i);
        unsafe { bld.launch(cfg) }.unwrap();
    });

    // m3_dqkv bf16 typed (production dtype)
    let q_t = upload_typed(&ctx, &det_rand(n_q, 0xA001), WeightDtype::Bf16);
    let ks_t = upload_typed(&ctx, &det_rand(n_q, 0xA002), WeightDtype::Bf16);
    let v_t = upload_typed(&ctx, &det_rand(n_v, 0xA003), WeightDtype::Bf16);
    let do_t = upload_typed(&ctx, &det_rand(n_v, 0xA004), WeightDtype::Bf16);
    time_it("m3_dqkv bf16", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_dqkv_typed.get(WeightDtype::Bf16));
        let args = [
            dq.cached_ptr(),
            dk.cached_ptr(),
            dv.cached_ptr(),
            dadt.cached_ptr(),
            dqk.cached_ptr(),
            dd.cached_ptr(),
            q_t.cached_ptr(),
            ks_t.cached_ptr(),
            v_t.cached_ptr(),
            dcs_buf.cached_ptr(),
            dcs_sum_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            ssm_buf.cached_ptr(),
            do_t.cached_ptr(),
            d_buf.cached_ptr(),
            enter.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&use_mats_i);
        unsafe { bld.launch(cfg) }.unwrap();
    });

    // m3_dqktheta f32
    let n_ang = CB * CT * CNH * CNA;
    let scale_buf = upload_f32(&ctx, &det_rand(n_th, 0xA00A));
    let gamma_buf = upload_f32(&ctx, &det_rand(n_th, 0xA00B));
    let angle_buf = upload_f32(&ctx, &det_rand(n_ang, 0xA00C));
    let dqk_dot_buf = upload_f32(&ctx, &det_rand(n_th, 0xA00D));
    let dqpre = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dkpre = GpuBuffer::zeros(&ctx.stream, n_q).unwrap();
    let dang = GpuBuffer::zeros(&ctx.stream, n_ang).unwrap();
    let dscale = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    let dgamma = GpuBuffer::zeros(&ctx.stream, n_th).unwrap();
    ctx.stream.synchronize().unwrap();
    let th_cfg = LaunchConfig {
        grid_dim: ((CB * n_chunks) as u32, CNH as u32, 1),
        block_dim: (CCS as u32, 1, 1),
        shared_mem_bytes: (6 * CCS * CDS * 4) as u32,
    };
    let nai = CNA as i32;
    let th_staging: i32 = 1;
    time_it("m3_dqktheta f32", &|| {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_dqktheta);
        let args = [
            dqpre.cached_ptr(),
            dkpre.cached_ptr(),
            dang.cached_ptr(),
            dscale.cached_ptr(),
            dgamma.cached_ptr(),
            q_f32.cached_ptr(),
            ks_f32.cached_ptr(),
            scale_buf.cached_ptr(),
            gamma_buf.cached_ptr(),
            angle_buf.cached_ptr(),
            dq.cached_ptr(),
            dk.cached_ptr(),
            dqk_dot_buf.cached_ptr(),
        ];
        for a in &args {
            bld.arg(a);
        }
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&dsi);
        bld.arg(&nai);
        bld.arg(&csi);
        bld.arg(&th_staging);
        unsafe { bld.launch(th_cfg) }.unwrap();
    });

    // colsum_accumulate over dQ_pre (per layer the trainer runs two of
    // these for dQ_bias/dK_bias) — the colsum-retirement cost question.
    let dqb = GpuBuffer::zeros(&ctx.stream, CNH * CDS).unwrap();
    ctx.stream.synchronize().unwrap();
    let cs_grid = LaunchConfig {
        grid_dim: (((CNH * CDS) as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows = (CB * CT) as i32;
    let cols = (CNH * CDS) as i32;
    time_it("colsum_accumulate (1 of 2/layer)", &|| {
        let mut bld = ctx.stream.launch_builder(&m3k.colsum_accumulate);
        let dqb_p = dqb.cached_ptr();
        bld.arg(&dqb_p);
        bld.arg(dqpre.inner());
        bld.arg(&rows);
        bld.arg(&cols);
        unsafe { bld.launch(cs_grid) }.unwrap();
    });

    // Forward chunk kernels (bf16, production geometry): grid covers
    // (b*chunk, head-pair), so occupancy is structurally different from
    // the backward dqkv — measure, do not assume.
    let cst_out = GpuBuffer::zeros(&ctx.stream, CB * n_chunks * CNH * CHD * CDS).unwrap();
    let y_t = upload_typed(&ctx, &det_rand(n_v, 0xA010), WeightDtype::Bf16);
    ctx.stream.synchronize().unwrap();
    let fwd_cfg = LaunchConfig {
        grid_dim: ((CB * n_chunks) as u32, (CNH as u32).div_ceil(2), 1),
        block_dim: (CHD as u32, 2, 1),
        shared_mem_bytes: 0,
    };
    time_it("m3_chunk_state_fwd bf16", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_chunk_state_fwd_typed.get(WeightDtype::Bf16));
        let xp = v_t.cached_ptr();
        let ksp = ks_t.cached_ptr();
        let cst_p = cst_out.cached_ptr();
        bld.arg(&cst_p);
        bld.arg(&xp);
        bld.arg(&ksp);
        bld.arg(dcs_buf.inner());
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        unsafe { bld.launch(fwd_cfg) }.unwrap();
    });
    time_it("m3_chunk_scan_fwd bf16", &|| {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_chunk_scan_fwd_typed.get(WeightDtype::Bf16));
        let yp = y_t.cached_ptr();
        let xp = v_t.cached_ptr();
        let qp = q_t.cached_ptr();
        let ksp = ks_t.cached_ptr();
        bld.arg(&yp);
        bld.arg(&xp);
        bld.arg(&qp);
        bld.arg(&ksp);
        bld.arg(qk_buf.inner());
        bld.arg(dcs_buf.inner());
        bld.arg(ssm_buf.inner());
        bld.arg(d_buf.inner());
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        unsafe { bld.launch(fwd_cfg) }.unwrap();
    });
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    let selected: Vec<String> = std::env::args().skip(1).collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("m3_kernels_isolated_bench") {
        m3_kernels_isolated_bench();
    }
}
