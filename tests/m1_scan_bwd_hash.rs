//! Bit gate for the M1 parallel scan backward — whole-trainer digests
//! alone give the PRODUCTION fold kernel no per-kernel hash coverage. Record once on a good build; any later kernel edit that
//! moves a bit shows up here in seconds, at the exact kernel.
//!
//!   cargo test --release --features cuda --test m1_scan_bwd_hash -- --ignored --nocapture

#![cfg(feature = "cuda")]

mod common;

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::launch::{
    SCAN_BWD_DGROUP, grid_parallel_scan_bwd, grid_parallel_scan_bwd_fold, grid_parallel_scan_typed,
    scan_tape_len,
};

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

#[test]
#[ignore = "bit-gate recorder"]
fn m1_scan_bwd_output_hashes() {
    let (b, t, di, ds) = (8usize, 1300usize, 768usize, 16usize);
    let device = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&device, 16).unwrap();
    let k = &ctx.kernels;
    let dtype = WeightDtype::Bf16;
    let bt = b * t;

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

    let h = GpuBuffer::zeros(&ctx.stream, b * di * ds).unwrap();
    let y = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let delta = upload_typed(&det(bt * di, 11, 0.05));
    let delta_saved = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let u = upload_typed(&det(bt * di, 12, 0.5));
    let bb = upload_typed(&det(bt * ds, 13, 0.3));
    let cc = upload_typed(&det(bt * ds, 14, 0.3));
    let a_neg = upload_f32(
        &det(di * ds, 15, 0.2)
            .iter()
            .map(|x| -x.abs())
            .collect::<Vec<_>>(),
    );
    let dpar = upload_f32(&det(di, 16, 0.1));
    let h_saved = GpuBuffer::zeros(&ctx.stream, b * (t + 1) * di * ds).unwrap();
    let tape = GpuBuffer::zeros(&ctx.stream, scan_tape_len(b, t, di, ds)).unwrap();
    ctx.stream.synchronize().unwrap();

    let bi = b as i32;
    let ti = t as i32;
    let dii = di as i32;
    let dsi = ds as i32;

    // Forward twice: full tape (feeds the ungrouped/full backward) and
    // slim tape (feeds the production fold/slim backward).
    for (slim, tape_ptr) in [(0i32, h_saved.cached_ptr()), (1i32, tape.cached_ptr())] {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_parallel_fwd_typed.get(dtype));
        let hp = h.cached_ptr();
        let yp = y.cached_ptr();
        let hs = h_saved.cached_ptr();
        // The fused forward reads the PRE-softplus dt and writes the
        // post-softplus save itself; the backward replays from that save.
        let dp = delta.cached_ptr();
        let dsv = delta_saved.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        bld.arg(&hp);
        bld.arg(&yp);
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&dsv);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        bld.arg(&tape_ptr);
        bld.arg(&slim);
        unsafe { bld.launch(grid_parallel_scan_typed(b, di, dtype.size_bytes(), ds)) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    {
        let mut hv = vec![0f32; b * di * ds];
        h.download(&ctx.stream, &mut hv).unwrap();
        ctx.stream.synchronize().unwrap();
        eprintln!("HASH fwd_h_final {:016x}", common::bench::fnv1a_f32(&hv));
    }

    let d_y = upload_typed(&det(bt * di, 17, 0.1));
    let d_delta = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_u = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let d_b_full = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_c_full = DtypedBuf::zeros(&ctx.stream, bt * di * ds, dtype).unwrap();
    let d_b_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    let d_c_fold = DtypedBuf::zeros(&ctx.stream, bt * (di / SCAN_BWD_DGROUP) * ds, dtype).unwrap();
    let d_d_local = GpuBuffer::zeros(&ctx.stream, b * di).unwrap();
    let nc_fold = t
        .div_ceil(mamba_rs::mamba_ssm::gpu::launch::SCAN_CHUNK)
        .max(1);
    // Fold writes one partial row per chunk; hashed after a host fold in
    // the same walk order (bit-equal to the retired accumulator).
    let mut d_a_log_local = GpuBuffer::zeros(&ctx.stream, b * nc_fold * di * ds).unwrap();
    ctx.stream.synchronize().unwrap();

    for (route, fold) in [("fold_slim", true), ("ungrouped_full", false)] {
        // Distinct layouts share this buffer across arms (fold writes
        // chunk-slot rows with =, the ungrouped kernel accumulates the
        // per-sample prefix with +=) - reset between arms.
        d_a_log_local.zero(&ctx.stream).unwrap();
        ctx.stream.synchronize().unwrap();
        let mut bld = ctx.stream.launch_builder(if fold {
            k.ssm_parallel_bwd_fold_typed.get(dtype)
        } else {
            k.ssm_parallel_bwd_typed.get(dtype)
        });
        let hs = h_saved.cached_ptr();
        let tp = tape.cached_ptr();
        let dp = delta_saved.cached_ptr();
        let up = u.cached_ptr();
        let bp = bb.cached_ptr();
        let cp = cc.cached_ptr();
        let ap = a_neg.cached_ptr();
        let ddp = dpar.cached_ptr();
        let dyp = d_y.cached_ptr();
        let ddel = d_delta.cached_ptr();
        let dp_raw = delta.cached_ptr();
        let dup = d_u.cached_ptr();
        let dbl = if fold {
            d_b_fold.cached_ptr()
        } else {
            d_b_full.cached_ptr()
        };
        let dcl = if fold {
            d_c_fold.cached_ptr()
        } else {
            d_c_full.cached_ptr()
        };
        let ddl = d_d_local.cached_ptr();
        let dal = d_a_log_local.cached_ptr();
        bld.arg(&hs);
        bld.arg(&dp);
        bld.arg(&up);
        bld.arg(&bp);
        bld.arg(&cp);
        bld.arg(&ap);
        bld.arg(&ddp);
        bld.arg(&dyp);
        bld.arg(&ddel);
        if fold {
            // The fold takes the PRE-softplus dt right after its output
            // slot (it emits the raw-dt gradient inline).
            bld.arg(&dp_raw);
        }
        bld.arg(&dup);
        bld.arg(&dbl);
        bld.arg(&dcl);
        bld.arg(&ddl);
        bld.arg(&dal);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&dii);
        bld.arg(&dsi);
        let (tape_arg, slim) = if fold { (tp, 1i32) } else { (hs, 0i32) };
        bld.arg(&tape_arg);
        bld.arg(&slim);
        let cfg = if fold {
            grid_parallel_scan_bwd_fold(b, di, ds, dtype.size_bytes())
        } else {
            grid_parallel_scan_bwd(b, di)
        };
        unsafe { bld.launch(cfg) }.unwrap();
        ctx.stream.synchronize().unwrap();

        eprintln!(
            "{}",
            common::bench::bench_stamp(
                &device,
                &ctx,
                "B8 T1300 di768 ds16",
                route,
                if fold { di / SCAN_BWD_DGROUP } else { di }
            )
        );
        common::bench::hash_outputs(
            &ctx,
            &[
                ("d_D_local", &d_d_local, b * di),
                ("d_a_log_local", &d_a_log_local, b * nc_fold * di * ds),
            ],
        );
        // Typed (bf16) outputs hash through their raw f32 upcast download.
        for (name, buf, n) in [("d_delta", &d_delta, bt * di), ("d_u", &d_u, bt * di)] {
            let mut v = vec![0f32; n];
            buf.download_f32(&ctx.stream, &mut v).unwrap();
            ctx.stream.synchronize().unwrap();
            eprintln!("HASH {route}:{name} {:016x}", common::bench::fnv1a_f32(&v));
        }
        let (bname, bbuf, cname, cbuf, rows) = if fold {
            (
                "d_B_fold",
                &d_b_fold,
                "d_C_fold",
                &d_c_fold,
                di / SCAN_BWD_DGROUP,
            )
        } else {
            ("d_B_full", &d_b_full, "d_C_full", &d_c_full, di)
        };
        for (name, buf) in [(bname, bbuf), (cname, cbuf)] {
            // Full-buffer download: DtypedBuf::download_f32 asserts the
            // exact element count, and a recorder runs once per build.
            let n = bt * rows * ds;
            let mut v = vec![0f32; n];
            buf.download_f32(&ctx.stream, &mut v).unwrap();
            ctx.stream.synchronize().unwrap();
            eprintln!("HASH {route}:{name} {:016x}", common::bench::fnv1a_f32(&v));
        }
    }

    // Reducer at the production (fold) depth.
    let d_b_red = GpuBuffer::zeros(&ctx.stream, bt * ds).unwrap();
    let d_c_red = GpuBuffer::zeros(&ctx.stream, bt * ds).unwrap();
    let reduce_di = (di / SCAN_BWD_DGROUP) as i32;
    {
        let mut bld = ctx
            .stream
            .launch_builder(k.ssm_reduce_d_bc_tmajor_typed.get(dtype));
        let o1 = d_b_red.cached_ptr();
        let o2 = d_c_red.cached_ptr();
        let i1 = d_b_fold.cached_ptr();
        let i2 = d_c_fold.cached_ptr();
        bld.arg(&o1);
        bld.arg(&o2);
        bld.arg(&i1);
        bld.arg(&i2);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&reduce_di);
        bld.arg(&dsi);
        unsafe { bld.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(bt * ds)) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    common::bench::hash_outputs(
        &ctx,
        &[
            ("reduce_d_B", &d_b_red, bt * ds),
            ("reduce_d_C", &d_c_red, bt * ds),
        ],
    );
}
