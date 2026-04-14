//! Step 9a — per-kernel parity tests for the 4 new typed M3 backward
//! kernels (`bcnorm_bwd_typed`, `bc_bias_add_bwd_typed`, `rope_bwd_typed`,
//! `m3_split_bwd_typed`). Runs each typed variant against its f32 oracle
//! on random inputs and asserts cosine + norm-ratio parity.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::launch::grid_1d;
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

fn cos_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let ratio = if na > 0.0 {
        ((nb.sqrt()) / (na.sqrt())) as f32
    } else {
        1.0
    };
    (cos, ratio)
}

fn assert_close(label: &str, a: &[f32], b: &[f32], cos_min: f32, norm_tol: f32) {
    let (cos, ratio) = cos_norm(a, b);
    eprintln!("  {label}: cos={cos:.6} norm={ratio:.4} n={}", a.len());
    assert!(cos >= cos_min, "{label}: cos {cos} < {cos_min}");
    assert!(
        (ratio - 1.0).abs() <= norm_tol,
        "{label}: norm {ratio} outside [1 ± {norm_tol}]"
    );
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn download_typed(ctx: &GpuCtx, buf: &DtypedBuf) -> Vec<f32> {
    let mut out = vec![0f32; buf.len_elems()];
    buf.download_f32(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    out
}

// Small config shared across tests.
const N_SAMPLES: usize = 32;
const NG: usize = 1;
const DS: usize = 8;
const NH: usize = 8;
const HD: usize = 8;
const N_ANGLES: usize = 2;
const DI: usize = NH * HD;

fn make_m3k(ctx: &GpuCtx) -> Mamba3Kernels {
    Mamba3Kernels::compile(ctx.stream.context(), "sm_89").unwrap()
}

// ─── bcnorm_bwd ───────────────────────────────────────────────────────

fn run_bcnorm_bwd_f32(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_out: &GpuBuffer,
    b_raw: &GpuBuffer,
    rms: &GpuBuffer,
    weight: &GpuBuffer,
) -> (Vec<f32>, Vec<f32>) {
    let d_b = GpuBuffer::zeros(&ctx.stream, N_SAMPLES * NG * DS).unwrap();
    let d_w = GpuBuffer::zeros(&ctx.stream, N_SAMPLES * NG * DS).unwrap();
    ctx.stream.synchronize().unwrap();
    let ni = N_SAMPLES as i32;
    let ngi = NG as i32;
    let dsi = DS as i32;
    let cfg = LaunchConfig {
        grid_dim: ((N_SAMPLES * NG) as u32, 1, 1),
        block_dim: (DS as u32, 1, 1),
        shared_mem_bytes: DS as u32 * 4,
    };
    let mut bld = ctx.stream.launch_builder(&m3k.bcnorm_bwd);
    let dbp = d_b.cached_ptr();
    let dwp = d_w.cached_ptr();
    let dop = d_out.cached_ptr();
    let brp = b_raw.cached_ptr();
    let rp = rms.cached_ptr();
    let wp = weight.cached_ptr();
    bld.arg(&dbp);
    bld.arg(&dwp);
    bld.arg(&dop);
    bld.arg(&brp);
    bld.arg(&rp);
    bld.arg(&wp);
    bld.arg(&ni);
    bld.arg(&ngi);
    bld.arg(&dsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let mut d_b_out = vec![0f32; N_SAMPLES * NG * DS];
    let mut d_w_out = vec![0f32; N_SAMPLES * NG * DS];
    d_b.download(&ctx.stream, &mut d_b_out).unwrap();
    d_w.download(&ctx.stream, &mut d_w_out).unwrap();
    (d_b_out, d_w_out)
}

fn run_bcnorm_bwd_typed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    d_out: &DtypedBuf,
    b_raw: &DtypedBuf,
    rms: &GpuBuffer,
    weight: &GpuBuffer,
) -> (Vec<f32>, Vec<f32>) {
    let d_b = DtypedBuf::zeros(&ctx.stream, N_SAMPLES * NG * DS, dtype).unwrap();
    let d_w = GpuBuffer::zeros(&ctx.stream, N_SAMPLES * NG * DS).unwrap();
    ctx.stream.synchronize().unwrap();
    let ni = N_SAMPLES as i32;
    let ngi = NG as i32;
    let dsi = DS as i32;
    let cfg = LaunchConfig {
        grid_dim: ((N_SAMPLES * NG) as u32, 1, 1),
        block_dim: (DS as u32, 1, 1),
        shared_mem_bytes: DS as u32 * 4,
    };
    let mut bld = ctx.stream.launch_builder(m3k.bcnorm_bwd_typed.get(dtype));
    let dbp = d_b.cached_ptr();
    let dwp = d_w.cached_ptr();
    let dop = d_out.cached_ptr();
    let brp = b_raw.cached_ptr();
    let rp = rms.cached_ptr();
    let wp = weight.cached_ptr();
    bld.arg(&dbp);
    bld.arg(&dwp);
    bld.arg(&dop);
    bld.arg(&brp);
    bld.arg(&rp);
    bld.arg(&wp);
    bld.arg(&ni);
    bld.arg(&ngi);
    bld.arg(&dsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let d_b_out = download_typed(ctx, &d_b);
    let mut d_w_out = vec![0f32; N_SAMPLES * NG * DS];
    d_w.download(&ctx.stream, &mut d_w_out).unwrap();
    (d_b_out, d_w_out)
}

fn check_bcnorm_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let n_elems = N_SAMPLES * NG * DS;
    let d_out_f = det_rand(n_elems, 0xB1);
    let b_raw_f = det_rand(n_elems, 0xB2);
    let rms_f: Vec<f32> = (0..N_SAMPLES * NG)
        .map(|i| 1.0 + 0.1 * (i as f32))
        .collect();
    let weight_f: Vec<f32> = (0..DS).map(|i| 1.0 + 0.01 * (i as f32)).collect();

    let mut d_out_f32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let mut b_raw_f32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let mut rms_buf = GpuBuffer::zeros(&ctx.stream, N_SAMPLES * NG).unwrap();
    let mut weight_buf = GpuBuffer::zeros(&ctx.stream, DS).unwrap();
    ctx.stream.synchronize().unwrap();
    d_out_f32.upload(&ctx.stream, &d_out_f).unwrap();
    b_raw_f32.upload(&ctx.stream, &b_raw_f).unwrap();
    rms_buf.upload(&ctx.stream, &rms_f).unwrap();
    weight_buf.upload(&ctx.stream, &weight_f).unwrap();
    ctx.stream.synchronize().unwrap();

    let (d_b_ref, d_w_ref) =
        run_bcnorm_bwd_f32(&ctx, &m3k, &d_out_f32, &b_raw_f32, &rms_buf, &weight_buf);
    let d_out_t = upload_typed(&ctx, &d_out_f, dtype);
    let b_raw_t = upload_typed(&ctx, &b_raw_f, dtype);
    let (d_b_typ, d_w_typ) =
        run_bcnorm_bwd_typed(&ctx, &m3k, dtype, &d_out_t, &b_raw_t, &rms_buf, &weight_buf);

    eprintln!("bcnorm_bwd {dtype:?}:");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.03_f32),
        WeightDtype::F16 => (0.9995_f32, 0.01_f32),
        _ => unreachable!(),
    };
    assert_close("d_B", &d_b_ref, &d_b_typ, cos_min, norm_tol);
    assert_close(
        "d_weight (f32 master)",
        &d_w_ref,
        &d_w_typ,
        cos_min,
        norm_tol,
    );
}

#[test]
fn bcnorm_bwd_bf16() {
    check_bcnorm_bwd(WeightDtype::Bf16);
}
#[test]
fn bcnorm_bwd_f16() {
    check_bcnorm_bwd(WeightDtype::F16);
}

// ─── bc_bias_add_bwd ───────────────────────────────────────────────────

fn check_bc_bias_add_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let n_in = N_SAMPLES * NH * DS;
    let n_out = N_SAMPLES * NG * DS;
    let d_bias = det_rand(n_in, 0xC1);

    // f32 oracle.
    let mut dbb_f32 = GpuBuffer::zeros(&ctx.stream, n_in).unwrap();
    ctx.stream.synchronize().unwrap();
    dbb_f32.upload(&ctx.stream, &d_bias).unwrap();
    let dbn_f32 = GpuBuffer::zeros(&ctx.stream, n_out).unwrap();
    ctx.stream.synchronize().unwrap();
    let ni = N_SAMPLES as i32;
    let nhi = NH as i32;
    let ngi = NG as i32;
    let dsi = DS as i32;
    let mut bld = ctx.stream.launch_builder(&m3k.bc_bias_add_bwd);
    let a = dbn_f32.cached_ptr();
    let b = dbb_f32.cached_ptr();
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&ni);
    bld.arg(&nhi);
    bld.arg(&ngi);
    bld.arg(&dsi);
    unsafe { bld.launch(grid_1d(n_out)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dbn_ref = vec![0f32; n_out];
    dbn_f32.download(&ctx.stream, &mut dbn_ref).unwrap();

    // typed
    let dbb_t = upload_typed(&ctx, &d_bias, dtype);
    let dbn_t = DtypedBuf::zeros(&ctx.stream, n_out, dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx
        .stream
        .launch_builder(m3k.bc_bias_add_bwd_typed.get(dtype));
    let a = dbn_t.cached_ptr();
    let b = dbb_t.cached_ptr();
    bld.arg(&a);
    bld.arg(&b);
    bld.arg(&ni);
    bld.arg(&nhi);
    bld.arg(&ngi);
    bld.arg(&dsi);
    unsafe { bld.launch(grid_1d(n_out)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let dbn_typ = download_typed(&ctx, &dbn_t);

    eprintln!("bc_bias_add_bwd {dtype:?}:");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.03_f32),
        WeightDtype::F16 => (0.9995_f32, 0.01_f32),
        _ => unreachable!(),
    };
    assert_close("d_B_normed", &dbn_ref, &dbn_typ, cos_min, norm_tol);
}

#[test]
fn bc_bias_add_bwd_bf16() {
    check_bc_bias_add_bwd(WeightDtype::Bf16);
}
#[test]
fn bc_bias_add_bwd_f16() {
    check_bc_bias_add_bwd(WeightDtype::F16);
}

// ─── m3_split_bwd ──────────────────────────────────────────────────────

fn check_m3_split_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let n = N_SAMPLES;
    let in_proj_dim = 2 * DI + 2 * NG * DS + 3 * NH + N_ANGLES;

    let d_z = det_rand(n * DI, 0xD1);
    let d_x = det_rand(n * DI, 0xD2);
    let d_b_raw = det_rand(n * NG * DS, 0xD3);
    let d_c_raw = det_rand(n * NG * DS, 0xD4);
    let d_dd_dt = det_rand(n * NH, 0xD5);
    let d_dd_a = det_rand(n * NH, 0xD6);
    let d_trap = det_rand(n * NH, 0xD7);
    let d_angles = det_rand(n * N_ANGLES, 0xD8);

    // f32 oracle
    let up = |v: &[f32]| {
        let mut b = GpuBuffer::zeros(&ctx.stream, v.len()).unwrap();
        ctx.stream.synchronize().unwrap();
        b.upload(&ctx.stream, v).unwrap();
        b
    };
    let dz32 = up(&d_z);
    let dx32 = up(&d_x);
    let db32 = up(&d_b_raw);
    let dc32 = up(&d_c_raw);
    let ddt32 = up(&d_dd_dt);
    let dda32 = up(&d_dd_a);
    let dtr32 = up(&d_trap);
    let dan32 = up(&d_angles);
    let dp32 = GpuBuffer::zeros(&ctx.stream, n * in_proj_dim).unwrap();
    ctx.stream.synchronize().unwrap();
    let ni = n as i32;
    let dii = DI as i32;
    let ngi = NG as i32;
    let dsi = DS as i32;
    let nhi = NH as i32;
    let nai = N_ANGLES as i32;
    let mut bld = ctx.stream.launch_builder(&m3k.m3_split_bwd);
    let p = dp32.cached_ptr();
    let z = dz32.cached_ptr();
    let x = dx32.cached_ptr();
    let br = db32.cached_ptr();
    let cr = dc32.cached_ptr();
    let dt = ddt32.cached_ptr();
    let da = dda32.cached_ptr();
    let tr = dtr32.cached_ptr();
    let an = dan32.cached_ptr();
    bld.arg(&p);
    bld.arg(&z);
    bld.arg(&x);
    bld.arg(&br);
    bld.arg(&cr);
    bld.arg(&dt);
    bld.arg(&da);
    bld.arg(&tr);
    bld.arg(&an);
    bld.arg(&ni);
    bld.arg(&dii);
    bld.arg(&ngi);
    bld.arg(&dsi);
    bld.arg(&nhi);
    bld.arg(&nai);
    unsafe { bld.launch(grid_1d(n * in_proj_dim)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dp_ref = vec![0f32; n * in_proj_dim];
    dp32.download(&ctx.stream, &mut dp_ref).unwrap();

    // typed
    let dzt = upload_typed(&ctx, &d_z, dtype);
    let dxt = upload_typed(&ctx, &d_x, dtype);
    let dbt = upload_typed(&ctx, &d_b_raw, dtype);
    let dct = upload_typed(&ctx, &d_c_raw, dtype);
    let dpt = DtypedBuf::zeros(&ctx.stream, n * in_proj_dim, dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx.stream.launch_builder(m3k.m3_split_bwd_typed.get(dtype));
    let p = dpt.cached_ptr();
    let z = dzt.cached_ptr();
    let x = dxt.cached_ptr();
    let br = dbt.cached_ptr();
    let cr = dct.cached_ptr();
    let dt = ddt32.cached_ptr();
    let da = dda32.cached_ptr();
    let tr = dtr32.cached_ptr();
    let an = dan32.cached_ptr();
    bld.arg(&p);
    bld.arg(&z);
    bld.arg(&x);
    bld.arg(&br);
    bld.arg(&cr);
    bld.arg(&dt);
    bld.arg(&da);
    bld.arg(&tr);
    bld.arg(&an);
    bld.arg(&ni);
    bld.arg(&dii);
    bld.arg(&ngi);
    bld.arg(&dsi);
    bld.arg(&nhi);
    bld.arg(&nai);
    unsafe { bld.launch(grid_1d(n * in_proj_dim)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let dp_typ = download_typed(&ctx, &dpt);

    eprintln!("m3_split_bwd {dtype:?}:");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.03_f32),
        WeightDtype::F16 => (0.9995_f32, 0.01_f32),
        _ => unreachable!(),
    };
    assert_close("d_proj", &dp_ref, &dp_typ, cos_min, norm_tol);
}

#[test]
fn m3_split_bwd_bf16() {
    check_m3_split_bwd(WeightDtype::Bf16);
}
#[test]
fn m3_split_bwd_f16() {
    check_m3_split_bwd(WeightDtype::F16);
}

// ─── rope_bwd ──────────────────────────────────────────────────────────

fn check_rope_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let n_elems = N_SAMPLES * NH * DS;
    let n_angles_total = N_SAMPLES * NH * N_ANGLES;

    let d_b_rot = det_rand(n_elems, 0xE1);
    let d_c_rot = det_rand(n_elems, 0xE2);
    let b_biased = det_rand(n_elems, 0xE3);
    let c_biased = det_rand(n_elems, 0xE4);
    let angle: Vec<f32> = (0..n_angles_total).map(|i| 0.1 * (i as f32)).collect();

    // f32 oracle
    let up = |v: &[f32]| {
        let mut b = GpuBuffer::zeros(&ctx.stream, v.len()).unwrap();
        ctx.stream.synchronize().unwrap();
        b.upload(&ctx.stream, v).unwrap();
        b
    };
    let dbr32 = up(&d_b_rot);
    let dcr32 = up(&d_c_rot);
    let bb32 = up(&b_biased);
    let cb32 = up(&c_biased);
    let ang32 = up(&angle);
    let dbp32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let dcp32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let dang32 = GpuBuffer::zeros(&ctx.stream, n_angles_total).unwrap();
    ctx.stream.synchronize().unwrap();
    let ni = N_SAMPLES as i32;
    let nhi = NH as i32;
    let dsi = DS as i32;
    let nai = N_ANGLES as i32;
    let mut bld = ctx.stream.launch_builder(&m3k.rope_bwd);
    let dp = dbp32.cached_ptr();
    let dpp = dcp32.cached_ptr();
    let dap = dang32.cached_ptr();
    let dbr = dbr32.cached_ptr();
    let dcr = dcr32.cached_ptr();
    let bb = bb32.cached_ptr();
    let cb = cb32.cached_ptr();
    let ap = ang32.cached_ptr();
    bld.arg(&dp);
    bld.arg(&dpp);
    bld.arg(&dap);
    bld.arg(&dbr);
    bld.arg(&dcr);
    bld.arg(&bb);
    bld.arg(&cb);
    bld.arg(&ap);
    bld.arg(&ni);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&nai);
    unsafe { bld.launch(grid_1d(n_elems)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dbp_ref = vec![0f32; n_elems];
    let mut dcp_ref = vec![0f32; n_elems];
    let mut dang_ref = vec![0f32; n_angles_total];
    dbp32.download(&ctx.stream, &mut dbp_ref).unwrap();
    dcp32.download(&ctx.stream, &mut dcp_ref).unwrap();
    dang32.download(&ctx.stream, &mut dang_ref).unwrap();

    // typed
    let dbrt = upload_typed(&ctx, &d_b_rot, dtype);
    let dcrt = upload_typed(&ctx, &d_c_rot, dtype);
    let bbt = upload_typed(&ctx, &b_biased, dtype);
    let cbt = upload_typed(&ctx, &c_biased, dtype);
    let dbpt = DtypedBuf::zeros(&ctx.stream, n_elems, dtype).unwrap();
    let dcpt = DtypedBuf::zeros(&ctx.stream, n_elems, dtype).unwrap();
    let dang_t = GpuBuffer::zeros(&ctx.stream, n_angles_total).unwrap();
    ctx.stream.synchronize().unwrap();
    let mut bld = ctx.stream.launch_builder(m3k.rope_bwd_typed.get(dtype));
    let dp = dbpt.cached_ptr();
    let dpp = dcpt.cached_ptr();
    let dap = dang_t.cached_ptr();
    let dbr = dbrt.cached_ptr();
    let dcr = dcrt.cached_ptr();
    let bb = bbt.cached_ptr();
    let cb = cbt.cached_ptr();
    let ap = ang32.cached_ptr();
    bld.arg(&dp);
    bld.arg(&dpp);
    bld.arg(&dap);
    bld.arg(&dbr);
    bld.arg(&dcr);
    bld.arg(&bb);
    bld.arg(&cb);
    bld.arg(&ap);
    bld.arg(&ni);
    bld.arg(&nhi);
    bld.arg(&dsi);
    bld.arg(&nai);
    unsafe { bld.launch(grid_1d(n_elems)) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let dbp_typ = download_typed(&ctx, &dbpt);
    let dcp_typ = download_typed(&ctx, &dcpt);
    let mut dang_typ = vec![0f32; n_angles_total];
    dang_t.download(&ctx.stream, &mut dang_typ).unwrap();
    // Silence unused warning on host-side dang_typ variable.
    let _ = &dang_t;

    eprintln!("rope_bwd {dtype:?}:");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.03_f32),
        WeightDtype::F16 => (0.9995_f32, 0.01_f32),
        _ => unreachable!(),
    };
    assert_close("d_B_pre_rope", &dbp_ref, &dbp_typ, cos_min, norm_tol);
    assert_close("d_C_pre_rope", &dcp_ref, &dcp_typ, cos_min, norm_tol);
    assert_close(
        "d_angle_cumsum (f32)",
        &dang_ref,
        &dang_typ,
        cos_min,
        norm_tol,
    );
}

#[test]
fn rope_bwd_bf16() {
    check_rope_bwd(WeightDtype::Bf16);
}
#[test]
fn rope_bwd_f16() {
    check_rope_bwd(WeightDtype::F16);
}

// ─── rmsnorm_gated_bwd (Step 9c) ───────────────────────────────────────

fn check_rmsnorm_gated_bwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    // group_size == headdim (M3 convention) → n_groups == nheads.
    let group_size = HD;
    let n_groups = DI / group_size;
    let n = N_SAMPLES;
    let n_elems = n * DI;
    let n_rms = n * n_groups;

    let d_out = det_rand(n_elems, 0xF1);
    let y_f = det_rand(n_elems, 0xF2);
    let z_f = det_rand(n_elems, 0xF3);
    let weight: Vec<f32> = (0..DI).map(|i| 1.0 + 0.01 * (i as f32)).collect();
    let rms: Vec<f32> = (0..n_rms).map(|i| 0.5 + 0.1 * (i as f32)).collect();

    // f32 oracle
    let up = |v: &[f32]| {
        let mut b = GpuBuffer::zeros(&ctx.stream, v.len()).unwrap();
        ctx.stream.synchronize().unwrap();
        b.upload(&ctx.stream, v).unwrap();
        b
    };
    let do32 = up(&d_out);
    let y32 = up(&y_f);
    let z32 = up(&z_f);
    let w32 = up(&weight);
    let r32 = up(&rms);
    let dy32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let dz32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    let dw32 = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    ctx.stream.synchronize().unwrap();

    let ni = n as i32;
    let dii = DI as i32;
    let gsi = group_size as i32;
    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (DI as u32, 1, 1),
        shared_mem_bytes: (DI * std::mem::size_of::<f32>()) as u32,
    };
    let mut bld = ctx.stream.launch_builder(&m3k.rmsnorm_gated_bwd);
    let dy = dy32.cached_ptr();
    let dz = dz32.cached_ptr();
    let dw = dw32.cached_ptr();
    let do_p = do32.cached_ptr();
    let y_p = y32.cached_ptr();
    let z_p = z32.cached_ptr();
    let w_p = w32.cached_ptr();
    let r_p = r32.cached_ptr();
    bld.arg(&dy);
    bld.arg(&dz);
    bld.arg(&dw);
    bld.arg(&do_p);
    bld.arg(&y_p);
    bld.arg(&z_p);
    bld.arg(&w_p);
    bld.arg(&r_p);
    bld.arg(&ni);
    bld.arg(&dii);
    bld.arg(&gsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let mut dy_ref = vec![0f32; n_elems];
    let mut dz_ref = vec![0f32; n_elems];
    let mut dw_ref = vec![0f32; n_elems];
    dy32.download(&ctx.stream, &mut dy_ref).unwrap();
    dz32.download(&ctx.stream, &mut dz_ref).unwrap();
    dw32.download(&ctx.stream, &mut dw_ref).unwrap();

    // typed
    let do_t = upload_typed(&ctx, &d_out, dtype);
    let y_t = upload_typed(&ctx, &y_f, dtype);
    let z_t = upload_typed(&ctx, &z_f, dtype);
    let dy_t = DtypedBuf::zeros(&ctx.stream, n_elems, dtype).unwrap();
    let dz_t = DtypedBuf::zeros(&ctx.stream, n_elems, dtype).unwrap();
    let dw_t = GpuBuffer::zeros(&ctx.stream, n_elems).unwrap();
    ctx.stream.synchronize().unwrap();

    let mut bld = ctx
        .stream
        .launch_builder(m3k.rmsnorm_gated_bwd_typed.get(dtype));
    let dy = dy_t.cached_ptr();
    let dz = dz_t.cached_ptr();
    let dw = dw_t.cached_ptr();
    let do_p = do_t.cached_ptr();
    let y_p = y_t.cached_ptr();
    let z_p = z_t.cached_ptr();
    let w_p = w32.cached_ptr();
    let r_p = r32.cached_ptr();
    bld.arg(&dy);
    bld.arg(&dz);
    bld.arg(&dw);
    bld.arg(&do_p);
    bld.arg(&y_p);
    bld.arg(&z_p);
    bld.arg(&w_p);
    bld.arg(&r_p);
    bld.arg(&ni);
    bld.arg(&dii);
    bld.arg(&gsi);
    unsafe { bld.launch(cfg) }.unwrap();
    ctx.stream.synchronize().unwrap();
    let dy_typ = download_typed(&ctx, &dy_t);
    let dz_typ = download_typed(&ctx, &dz_t);
    let mut dw_typ = vec![0f32; n_elems];
    dw_t.download(&ctx.stream, &mut dw_typ).unwrap();

    eprintln!("rmsnorm_gated_bwd {dtype:?}:");
    let (cos_min, norm_tol) = match dtype {
        WeightDtype::Bf16 => (0.995_f32, 0.03_f32),
        WeightDtype::F16 => (0.9995_f32, 0.01_f32),
        _ => unreachable!(),
    };
    assert_close("d_y", &dy_ref, &dy_typ, cos_min, norm_tol);
    assert_close("d_z", &dz_ref, &dz_typ, cos_min, norm_tol);
    assert_close("d_weight (f32 master)", &dw_ref, &dw_typ, cos_min, norm_tol);
}

#[test]
fn rmsnorm_gated_bwd_bf16() {
    check_rmsnorm_gated_bwd(WeightDtype::Bf16);
}
#[test]
fn rmsnorm_gated_bwd_f16() {
    check_rmsnorm_gated_bwd(WeightDtype::F16);
}
