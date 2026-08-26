//! State-capacity bit-invariance arm for the SEQUENTIAL / STEP kernel
//! family (`kernels/mamba_ssm.cu`): the compile-time MAMBA_RS_STATE_CAP
//! only sizes the per-thread register arrays, so two builds at different
//! capacities must produce identical bits for the same `d_state` inputs.
//! The parallel-scan family needs no probe - its kernels take no
//! capacity-dependent code path (cap-invariance is provable by
//! inspection); the sequential family's register arrays are exactly
//! where a capacity-dependent spill or loop change would hide.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::launch::grid_1d;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn buf(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    b.upload(&ctx.stream, data).unwrap();
    b
}

fn dl(ctx: &GpuCtx, b: &GpuBuffer) -> Vec<f32> {
    let mut out = vec![0.0f32; b.len()];
    b.download(&ctx.stream, &mut out).unwrap();
    out
}

fn assert_bits(tag: &str, cap: usize, got: &[f32], want: &[f32]) {
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        bad,
        0,
        "{tag}: cap={cap} diverges from cap=16 in {bad}/{} values",
        got.len()
    );
}

/// Sequential burnin (prefill fallback) + decode step at d_state=16,
/// compiled at capacities 16 / 64 / 256 - outputs and states must match
/// the cap=16 build bit for bit.
#[test]
fn sequential_family_is_state_cap_invariant() {
    let (b, t, di, ds) = (2usize, 96usize, 64usize, 16usize);
    let bt = b * t;
    let device = GpuDevice::new(0).unwrap();

    let delta = det(bt * di, 11, 1.0);
    let u = det(bt * di, 12, 1.0);
    let bmat = det(bt * ds, 13, 0.5);
    let cmat = det(bt * ds, 14, 0.5);
    let a_neg: Vec<f32> = det(di * ds, 15, 1.0)
        .iter()
        .map(|v| -v.abs() - 0.05)
        .collect();
    let dpar = det(di, 16, 0.5);
    let h0 = det(b * di * ds, 17, 0.1);
    let sd = det(b * di, 21, 1.0);
    let su = det(b * di, 22, 1.0);
    let sb = det(b * ds, 23, 0.5);
    let sc = det(b * ds, 24, 0.5);

    type StepOutputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let mut reference: Option<StepOutputs> = None;
    for cap in [16usize, 64, 256] {
        let ctx = GpuCtx::new_with_state_cap(&device, cap).unwrap();
        let k = &ctx.kernels;
        let mut h = buf(&ctx, &h0);
        let mut y = GpuBuffer::zeros(&ctx.stream, bt * di).unwrap();
        let delta_b = buf(&ctx, &delta);
        let u_b = buf(&ctx, &u);
        let bb = buf(&ctx, &bmat);
        let cb = buf(&ctx, &cmat);
        let ab = buf(&ctx, &a_neg);
        let db = buf(&ctx, &dpar);
        let (b_i, t_i, di_i, ds_i) = (b as i32, t as i32, di as i32, ds as i32);

        // Burnin nosave (the sequential prefill route). Gate stride 0 =
        // plain y store; the gate pointer is never dereferenced.
        {
            let gate_stride = 0i32;
            let mut bld = ctx.stream.launch_builder(&k.ssm_burnin_fwd_nosave);
            bld.arg(h.inner_mut());
            bld.arg(y.inner_mut());
            bld.arg(delta_b.inner());
            bld.arg(u_b.inner());
            bld.arg(bb.inner());
            bld.arg(cb.inner());
            bld.arg(ab.inner());
            bld.arg(db.inner());
            bld.arg(delta_b.inner());
            bld.arg(&gate_stride);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&ds_i);
            unsafe { bld.launch(grid_1d(b * di)) }.unwrap();
        }

        // Decode step on a fresh state.
        let mut hs = buf(&ctx, &h0);
        let mut ys = GpuBuffer::zeros(&ctx.stream, b * di).unwrap();
        {
            let sdb = buf(&ctx, &sd);
            let sub = buf(&ctx, &su);
            let sbb = buf(&ctx, &sb);
            let scb = buf(&ctx, &sc);
            let mut bld = ctx.stream.launch_builder(&k.ssm_step_fwd);
            bld.arg(hs.inner_mut());
            bld.arg(ys.inner_mut());
            bld.arg(sdb.inner());
            bld.arg(sub.inner());
            bld.arg(sbb.inner());
            bld.arg(scb.inner());
            bld.arg(ab.inner());
            bld.arg(db.inner());
            bld.arg(&b_i);
            bld.arg(&di_i);
            bld.arg(&ds_i);
            unsafe { bld.launch(grid_1d(b * di)) }.unwrap();
        }
        ctx.stream.synchronize().unwrap();

        let got = (dl(&ctx, &y), dl(&ctx, &h), dl(&ctx, &ys), dl(&ctx, &hs));
        match &reference {
            None => reference = Some(got),
            Some(r) => {
                assert_bits("burnin y", cap, &got.0, &r.0);
                assert_bits("burnin h", cap, &got.1, &r.1);
                assert_bits("step y", cap, &got.2, &r.2);
                assert_bits("step h", cap, &got.3, &r.3);
            }
        }
    }
}
