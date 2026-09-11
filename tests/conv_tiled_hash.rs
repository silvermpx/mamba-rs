//! Bit gate for the tiled depthwise conv kernels of the training path, the
//! forward and the two backward halves: they had no digest of their own,
//! only tolerance parity. Record once on a good build; compare after any
//! conv edit.
//!
//!   cargo test --release --features cuda --test conv_tiled_hash -- --ignored --nocapture

#![cfg(feature = "cuda")]

#[path = "common/digest.rs"]
mod digest;
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/evidence_digest.rs"]
mod evidence_digest;
#[path = "common/hash_outputs.rs"]
mod hash_outputs;

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::launch::grid_conv_tiled;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn run(dtype: WeightDtype) {
    let (b, t, di, dc) = (8usize, 1300usize, 768usize, 4usize);
    let bt = b * t;
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let k = &ctx.kernels;
    let lane = match dtype {
        WeightDtype::F32 => "f32",
        WeightDtype::Bf16 => "bf16",
        WeightDtype::F16 => "f16",
    };
    let upload_typed = |data: &[f32]| -> DtypedBuf {
        let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
        ctx.stream.synchronize().unwrap();
        buf.upload_f32(&ctx.stream, data).unwrap();
        ctx.stream.synchronize().unwrap();
        buf
    };
    let upload_f32 = |data: &[f32]| -> GpuBuffer {
        let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
        ctx.stream.synchronize().unwrap();
        buf.upload(&ctx.stream, data).unwrap();
        ctx.stream.synchronize().unwrap();
        buf
    };
    let x_branch = upload_typed(&det(bt * di, 21, 0.5));
    let d_u = upload_typed(&det(bt * di, 22, 0.1));
    let weight = upload_f32(&det(di * dc, 23, 0.4));
    let bias = upload_f32(&det(di, 24, 0.1));
    let state = upload_f32(&det(b * di * dc, 25, 0.3));
    let conv_states = GpuBuffer::zeros(&ctx.stream, b * di * dc).unwrap();
    let u = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    // The backward writes the x half of a [bt, 2*di] projection gradient.
    let d_proj = DtypedBuf::zeros(&ctx.stream, bt * 2 * di, dtype).unwrap();
    let n_tiles = t.div_ceil(128);
    let wp = GpuBuffer::zeros(&ctx.stream, b * n_tiles * di * dc).unwrap();
    let bp = GpuBuffer::zeros(&ctx.stream, b * n_tiles * di).unwrap();
    ctx.stream.synchronize().unwrap();
    let (bi, ti, dii, dci) = (b as i32, t as i32, di as i32, dc as i32);

    {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_burnin_fwd_tiled_typed.get(dtype));
        let up = u.cached_ptr();
        let sp = state.cached_ptr();
        let csp = conv_states.cached_ptr();
        let xp = x_branch.cached_ptr();
        let wpt = weight.cached_ptr();
        let bpt = bias.cached_ptr();
        bld.arg(&up);
        bld.arg(&sp);
        bld.arg(&csp);
        bld.arg(&xp);
        bld.arg(&wpt);
        bld.arg(&bpt);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dci);
        bld.arg(&dii); // x row stride: a contiguous x here
        unsafe { bld.launch(grid_conv_tiled(b, di, t)) }.unwrap();
    }
    {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_bwd_tiled_typed.get(dtype));
        let dxp = d_proj.cached_ptr();
        let wpp = wp.cached_ptr();
        let bpp = bp.cached_ptr();
        let dup = d_u.cached_ptr();
        let xbp = x_branch.cached_ptr();
        let cip = conv_states.cached_ptr();
        let wpt = weight.cached_ptr();
        let bpt = bias.cached_ptr();
        let stride = (2 * di) as i32;
        let off = 0i32;
        bld.arg(&dxp);
        bld.arg(&wpp);
        bld.arg(&bpp);
        bld.arg(&dup);
        bld.arg(&xbp);
        bld.arg(&cip);
        bld.arg(&wpt);
        bld.arg(&bpt);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dci);
        bld.arg(&stride);
        bld.arg(&off);
        bld.arg(&dii); // x row stride
        unsafe { bld.launch(grid_conv_tiled(b, di, t)) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    for (name, buf, n) in [("u", &u, bt * di), ("d_x", &d_proj, bt * 2 * di)] {
        let mut v = vec![0f32; n];
        buf.download_f32(&ctx.stream, &mut v).unwrap();
        ctx.stream.synchronize().unwrap();
        eprintln!("HASH conv:{lane}:{name} {:016x}", digest::fnv1a_f32(&v));
    }
    let state_name = format!("conv:{lane}:state_out");
    let saved_name = format!("conv:{lane}:conv_states_saved");
    let wp_name = format!("conv:{lane}:d_weight_partials");
    let bp_name = format!("conv:{lane}:d_bias_partials");
    hash_outputs::hash_outputs(
        &ctx,
        &[
            (&state_name, &state, b * di * dc),
            (&saved_name, &conv_states, b * di * dc),
            (&wp_name, &wp, b * n_tiles * di * dc),
            (&bp_name, &bp, b * n_tiles * di),
        ],
    );
}

#[test]
#[ignore = "bit-gate recorder"]
fn conv_tiled_output_hashes_f32() {
    run(WeightDtype::F32);
}

#[test]
#[ignore = "bit-gate recorder"]
fn conv_tiled_output_hashes_bf16() {
    run(WeightDtype::Bf16);
}
