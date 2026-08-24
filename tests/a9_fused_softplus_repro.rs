//! Focused reproduction for the fused-softplus forward at the exact shape
//! where the fold parity harness faults (b=1, T=512, d_inner=4, d_state=8,
//! f32): launches ONE forward and checks y and the written delta save
//! against a serial CPU reference. Distinguishes a kernel defect from a
//! harness defect in one run.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::launch::grid_parallel_scan;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn fbuf(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    b.upload(&ctx.stream, data).unwrap();
    b
}

#[test]
fn fused_softplus_fwd_matches_cpu() {
    let shape: (usize, usize, usize, usize) = match std::env::var("A9_SHAPE").as_deref() {
        Ok("big") => (1, 512, 768, 16),
        Ok("smalldi") => (1, 512, 4, 16),
        Ok("shortt") => (1, 64, 4, 8),
        _ => (1, 512, 4, 8),
    };
    let (b, t, di, ds) = shape;
    let bt = b * t;
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&device).unwrap();

    let raw = det(bt * di, 11, 0.5);
    let u = det(bt * di, 12, 0.5);
    // The parallel kernel reads B/C in the T-MAJOR layout the production
    // gather emits: [b][n][t].
    let bm_tn = det(bt * ds, 13, 0.5);
    let cm_tn = det(bt * ds, 14, 0.5);
    let mut bm = vec![0.0f32; bt * ds];
    let mut cm = vec![0.0f32; bt * ds];
    for tt in 0..t {
        for n in 0..ds {
            bm[n * t + tt] = bm_tn[tt * ds + n];
            cm[n * t + tt] = cm_tn[tt * ds + n];
        }
    }
    let a_neg: Vec<f32> = det(di * ds, 15, 1.0)
        .iter()
        .map(|v| -v.abs() - 0.05)
        .collect();
    let dpar = det(di, 16, 0.5);
    let h0 = vec![0.0f32; b * di * ds];

    let mut h = fbuf(&ctx, &h0);
    let mut y = GpuBuffer::zeros(&ctx.stream, bt * di).unwrap();
    let mut h_saved = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).unwrap();
    let raw_b = fbuf(&ctx, &raw);
    let mut dsave = GpuBuffer::zeros(&ctx.stream, bt * di).unwrap();
    let u_b = fbuf(&ctx, &u);
    let bb = fbuf(&ctx, &bm);
    let cb = fbuf(&ctx, &cm);
    let ab = fbuf(&ctx, &a_neg);
    let db = fbuf(&ctx, &dpar);
    let (b_i, t_i, di_i, ds_i) = (b as i32, t as i32, di as i32, ds as i32);
    let slim_mode = std::env::var("A9_SLIM").is_ok();
    let tape_buf = GpuBuffer::zeros(
        &ctx.stream,
        mamba_rs::mamba_ssm::gpu::launch::scan_tape_len(b, t, di, ds),
    )
    .unwrap();
    let tape = if slim_mode {
        tape_buf.cached_ptr()
    } else {
        h_saved.cached_ptr()
    };
    let slim = i32::from(slim_mode);
    {
        let mut bld = ctx.stream.launch_builder(&ctx.kernels.ssm_parallel_fwd);
        bld.arg(h.inner_mut());
        bld.arg(y.inner_mut());
        bld.arg(h_saved.inner_mut());
        bld.arg(raw_b.inner());
        bld.arg(dsave.inner_mut());
        bld.arg(u_b.inner());
        bld.arg(bb.inner());
        bld.arg(cb.inner());
        bld.arg(ab.inner());
        bld.arg(db.inner());
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&ds_i);
        bld.arg(&tape);
        bld.arg(&slim);
        unsafe { bld.launch(grid_parallel_scan(b, di, ds)) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let mut y_gpu = vec![0.0f32; bt * di];
    y.download(&ctx.stream, &mut y_gpu).unwrap();
    let mut ds_gpu = vec![0.0f32; bt * di];
    dsave.download(&ctx.stream, &mut ds_gpu).unwrap();

    // CPU reference: softplus then the serial scan.
    let sp: Vec<f32> = raw
        .iter()
        .map(|&x| {
            if x > 20.0 {
                x
            } else {
                (x * std::f32::consts::LOG2_E).exp2().ln_1p()
            }
        })
        .collect();
    eprintln!("y_gpu[0..8]  = {:?}", &y_gpu[..8.min(y_gpu.len())]);
    eprintln!("save[0..8]   = {:?}", &ds_gpu[..8.min(ds_gpu.len())]);
    eprintln!("cpu_sp[0..8] = {:?}", &sp[..8.min(sp.len())]);
    let mut bad_save = 0usize;
    for i in 0..bt * di {
        if (ds_gpu[i] - sp[i]).abs() > 1e-6 {
            bad_save += 1;
        }
    }
    assert_eq!(bad_save, 0, "delta save diverges from CPU softplus");

    let mut h_cpu = vec![0.0f32; di * ds];
    let mut worst = 0.0f32;
    for tt in 0..t {
        for d in 0..di {
            let idx = tt * di + d;
            let dd = sp[idx];
            let du = dd * u[idx];
            let mut yv = dpar[d] * u[idx];
            for n in 0..ds {
                let da = (dd * a_neg[d * ds + n] * std::f32::consts::LOG2_E).exp2();
                h_cpu[d * ds + n] = da * h_cpu[d * ds + n] + du * bm_tn[tt * ds + n];
                yv += h_cpu[d * ds + n] * cm_tn[tt * ds + n];
            }
            let diff = (yv - y_gpu[idx]).abs() / yv.abs().max(1e-3);
            if diff > worst {
                worst = diff;
            }
        }
    }
    assert!(worst < 1e-3, "y diverges from CPU scan: rel {worst}");
}
