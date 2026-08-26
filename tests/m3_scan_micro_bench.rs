//! Micro-instrument for the three SSM chunk kernels that dominate the
//! prism page (chunk_scan_fwd_coop / preprocess_chunks / chunk_state_fwd):
//! per-kernel wall time at the serve shape plus an FNV hash of the output
//! bits, so a scheduling/occupancy change can prove itself bit-preserving
//! against a pre-change run of this same test.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, chunk_scan_cfg};
use std::time::Instant;

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

mod common;
use common::digest::fnv1a_f32 as fnv;

#[test]
#[ignore = "needs a CUDA device"]
fn scan_trio_time_and_hash() {
    // Prism serve shape.
    let batch = 1usize;
    let t = 4621usize;
    let nh = 48usize;
    let hd = 16usize;
    let ds = 16usize;
    let cs = 64usize;
    let nc = t.div_ceil(cs);
    let d_inner = nh * hd;

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    let m3k = Mamba3Kernels::compile(dev.context(), arch).expect("m3 kernels");
    let st = &ctx.stream;

    let x = GpuBuffer::from_cpu(st, &det(batch * t * d_inner, 1)).unwrap();
    let q = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 2)).unwrap();
    let k = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 3)).unwrap();
    let dt_b = GpuBuffer::from_cpu(
        st,
        &det(batch * t * nh, 4)
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let trap = GpuBuffer::from_cpu(
        st,
        &det(batch * t * nh, 5)
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let da_cumsum = GpuBuffer::from_cpu(
        st,
        &det(batch * nc * nh * cs, 6)
            .iter()
            .map(|v| -v.abs())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let prev_states = GpuBuffer::from_cpu(st, &det(batch * nc * nh * hd * ds, 7)).unwrap();
    let d_param = GpuBuffer::from_cpu(st, &det(nh, 8)).unwrap();
    let qk_dot = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let scale = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let gamma = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let k_scaled = GpuBuffer::zeros(st, batch * t * nh * ds).unwrap();
    let y = GpuBuffer::zeros(st, batch * t * d_inner).unwrap();
    let chunk_states = GpuBuffer::zeros(st, batch * nc * nh * hd * ds).unwrap();

    let b_i = batch as i32;
    let t_i = t as i32;
    let nh_i = nh as i32;
    let hd_i = hd as i32;
    let ds_i = ds as i32;
    let cs_i = cs as i32;

    // K1 preprocess: writes k_scaled/qk_dot/scale/gamma consumed below.
    let preprocess = || {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch * nc) as u32, nh as u32, 1),
            block_dim: (cs as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
        b.arg(k_scaled.inner());
        b.arg(qk_dot.inner());
        b.arg(scale.inner());
        b.arg(gamma.inner());
        b.arg(k.inner());
        b.arg(q.inner());
        b.arg(dt_b.inner());
        b.arg(trap.inner());
        b.arg(&b_i);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        unsafe { b.launch(cfg) }.unwrap();
    };
    let chunk_state = || {
        let cfg = mamba_rs::mamba3_siso::gpu::kernels::chunk_state_cfg(batch, nc, nh, hd, ds, cs);
        let mut b = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd);
        b.arg(chunk_states.inner());
        b.arg(x.inner());
        b.arg(k_scaled.inner());
        b.arg(da_cumsum.inner());
        b.arg(&b_i);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&hd_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        unsafe { b.launch(cfg) }.unwrap();
    };
    let scan = || {
        let (coop, cfg) = chunk_scan_cfg(batch, nc, nh, hd, ds, cs);
        assert!(coop, "serve shape must ride the coop kernel");
        let mut b = ctx.stream.launch_builder(&m3k.m3_chunk_scan_fwd_coop);
        b.arg(y.inner());
        b.arg(x.inner());
        b.arg(q.inner());
        b.arg(k_scaled.inner());
        b.arg(qk_dot.inner());
        b.arg(da_cumsum.inner());
        b.arg(prev_states.inner());
        b.arg(d_param.inner());
        b.arg(&b_i);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&hd_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        unsafe { b.launch(cfg) }.unwrap();
    };

    preprocess();
    chunk_state();
    scan();
    ctx.stream.synchronize().unwrap();

    let mut host = vec![0.0f32; batch * t * d_inner];
    y.download(st, &mut host).unwrap();
    let nonzero = host.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > host.len() / 2,
        "scan output mostly zero: {nonzero}"
    );
    let mut ks_h = vec![0.0f32; batch * t * nh * ds];
    k_scaled.download(st, &mut ks_h).unwrap();
    let mut cst_h = vec![0.0f32; batch * nc * nh * hd * ds];
    chunk_states.download(st, &mut cst_h).unwrap();
    println!(
        "HASH y={:016x} k_scaled={:016x} chunk_states={:016x}",
        fnv(&host),
        fnv(&ks_h),
        fnv(&cst_h)
    );

    let iters = 100usize;
    let time = |f: &dyn Fn()| -> f64 {
        f();
        ctx.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        ctx.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    println!("preprocess  {:8.1} us", time(&preprocess));
    println!("chunk_state {:8.1} us", time(&chunk_state));
    println!("scan_coop   {:8.1} us", time(&scan));

    // The fused kernel must reproduce the pair's outputs BIT-EXACTLY:
    // zero the outputs, run fused once, re-hash.
    // Fresh zero buffers for the fused arm: a silent no-op cannot pass
    // by inheriting the pair's results.
    let k_scaled_f = GpuBuffer::zeros(st, batch * t * nh * ds).unwrap();
    let qk_dot_f = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let scale_f = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let gamma_f = GpuBuffer::zeros(st, batch * t * nh).unwrap();
    let chunk_states_f = GpuBuffer::zeros(st, batch * nc * nh * hd * ds).unwrap();
    let fused = || {
        let fcfg = mamba_rs::mamba3_siso::gpu::kernels::chunk_fused_cfg(batch, nc, nh, hd, ds, cs)
            .expect("serve shape must fit the fused kernel");
        let mut b = ctx
            .stream
            .launch_builder(&m3k.m3_chunk_pre_state_fused_typed.f32);
        b.arg(k_scaled_f.inner());
        b.arg(qk_dot_f.inner());
        b.arg(scale_f.inner());
        b.arg(gamma_f.inner());
        b.arg(chunk_states_f.inner());
        b.arg(k.inner());
        b.arg(q.inner());
        b.arg(dt_b.inner());
        b.arg(trap.inner());
        b.arg(x.inner());
        b.arg(da_cumsum.inner());
        b.arg(&b_i);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&hd_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        unsafe { b.launch(fcfg) }.unwrap();
    };
    fused();
    ctx.stream.synchronize().unwrap();
    let mut ks_f = vec![0.0f32; batch * t * nh * ds];
    k_scaled_f.download(st, &mut ks_f).unwrap();
    let mut cst_f = vec![0.0f32; batch * nc * nh * hd * ds];
    chunk_states_f.download(st, &mut cst_f).unwrap();
    println!(
        "FUSED HASH k_scaled={:016x} chunk_states={:016x}",
        fnv(&ks_f),
        fnv(&cst_f)
    );
    assert_eq!(fnv(&ks_f), fnv(&ks_h), "fused k_scaled diverged from pair");
    assert_eq!(
        fnv(&cst_f),
        fnv(&cst_h),
        "fused chunk_states diverged from pair"
    );
    println!("fused       {:8.1} us", time(&fused));

    // The typed scan macro instantiates an f32 variant with identity
    // conversions, so its output must match the plain coop kernel
    // bit-for-bit. This arm gates the shared macro body: an indexing or
    // layout slip in the typed twin shows up here even when every
    // bf16 parity test only measures within tolerance.
    let y_t = GpuBuffer::zeros(st, batch * t * d_inner).unwrap();
    let scan_typed = || {
        let (coop, cfg) = chunk_scan_cfg(batch, nc, nh, hd, ds, cs);
        assert!(coop, "serve shape must ride the coop kernel");
        let mut b = ctx
            .stream
            .launch_builder(&m3k.m3_chunk_scan_fwd_coop_typed.f32);
        b.arg(y_t.inner());
        b.arg(x.inner());
        b.arg(q.inner());
        b.arg(k_scaled.inner());
        b.arg(qk_dot.inner());
        b.arg(da_cumsum.inner());
        b.arg(prev_states.inner());
        b.arg(d_param.inner());
        b.arg(&b_i);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&hd_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        unsafe { b.launch(cfg) }.unwrap();
    };
    scan_typed();
    ctx.stream.synchronize().unwrap();
    let mut yt_h = vec![0.0f32; batch * t * d_inner];
    y_t.download(st, &mut yt_h).unwrap();
    assert_eq!(
        fnv(&yt_h),
        fnv(&host),
        "typed coop scan diverged from the plain coop kernel"
    );

    // The inter-chunk prefix scan mutates its states buffer in place, so
    // the hash arm runs once on a fresh copy of the pair's chunk_states;
    // the timing loop then reuses the (already transformed) buffer -
    // values drift there but the address pattern is identical.
    let states_sp = GpuBuffer::from_cpu(st, &cst_h).unwrap();
    let final_states = GpuBuffer::zeros(st, batch * nh * hd * ds).unwrap();
    let state_passing = || {
        let dim = hd * ds;
        let block_x = dim.min(256) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                batch as u32,
                nh as u32,
                dim.div_ceil(block_x as usize) as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = nc as i32;
        let init_states_null: u64 = 0;
        let mut b = ctx.stream.launch_builder(&m3k.m3_state_passing_fwd);
        b.arg(states_sp.inner());
        b.arg(final_states.inner());
        b.arg(da_cumsum.inner());
        b.arg(&init_states_null);
        b.arg(&b_i);
        b.arg(&nc_i);
        b.arg(&nh_i);
        b.arg(&hd_i);
        b.arg(&ds_i);
        b.arg(&cs_i);
        b.arg(&t_i);
        unsafe { b.launch(cfg) }.unwrap();
    };
    state_passing();
    ctx.stream.synchronize().unwrap();
    let mut sp_h = vec![0.0f32; batch * nc * nh * hd * ds];
    states_sp.download(st, &mut sp_h).unwrap();
    let mut fs_h = vec![0.0f32; batch * nh * hd * ds];
    final_states.download(st, &mut fs_h).unwrap();
    println!(
        "STATE_PASSING HASH entering={:016x} final={:016x}",
        fnv(&sp_h),
        fnv(&fs_h)
    );
    println!("state_pass  {:8.1} us", time(&state_passing));
}

/// The coefficient chain at the serve shape: split -> bcnorm -> angle
/// accumulation -> bias+rope -> abg, each timed and output-hashed. The
/// hashes are the bit gates for the dead-write elision and the angle
/// rework: every tensor a consumer actually reads must keep its exact
/// bits through those changes.
#[test]
#[ignore = "needs a CUDA device"]
fn coeff_chain_time_and_hash() {
    use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

    let batch = 1usize;
    let t = 4621usize;
    let nh = 48usize;
    let hd = 16usize;
    let ds = 16usize;
    let ng = 1usize;
    let na = 4usize;
    let di = nh * hd;
    let ip = 2 * di + 2 * ng * ds + 3 * nh + na;
    let bt = batch * t;
    let dt_ty = WeightDtype::Bf16;

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    let m3k = Mamba3Kernels::compile(dev.context(), arch).expect("m3 kernels");
    let st = &ctx.stream;

    // Exposing typed projection input + f32 coefficient weights.
    let mut proj_host = det(bt * ip, 11);
    for (i, v) in proj_host.iter_mut().enumerate() {
        match i % 4 {
            0 => *v = 4096.0,
            1 => *v = -4096.0,
            2 => *v *= 512.0,
            _ => {}
        }
    }
    let proj = DtypedBuf::zeros(st, bt * ip, dt_ty).unwrap();
    proj.upload_f32(st, &proj_host).unwrap();
    let dt_bias = GpuBuffer::from_cpu(st, &det(nh, 21)).unwrap();
    let bnw = GpuBuffer::from_cpu(st, &det(ds, 22)).unwrap();
    let cnw = GpuBuffer::from_cpu(st, &det(ds, 23)).unwrap();
    let b_bias = GpuBuffer::from_cpu(st, &det(nh * ds, 24)).unwrap();
    let c_bias = GpuBuffer::from_cpu(st, &det(nh * ds, 25)).unwrap();

    let z = DtypedBuf::zeros(st, bt * di, dt_ty).unwrap();
    let x = DtypedBuf::zeros(st, bt * di, dt_ty).unwrap();
    let b_raw = DtypedBuf::zeros(st, bt * ng * ds, dt_ty).unwrap();
    let c_raw = DtypedBuf::zeros(st, bt * ng * ds, dt_ty).unwrap();
    let dtc = GpuBuffer::zeros(st, bt * nh).unwrap();
    let a_val = GpuBuffer::zeros(st, bt * nh).unwrap();
    let trap = GpuBuffer::zeros(st, bt * nh).unwrap();
    let angles_raw = GpuBuffer::zeros(st, bt * na).unwrap();
    let dd_dt = GpuBuffer::zeros(st, bt * nh).unwrap();
    let dd_a = GpuBuffer::zeros(st, bt * nh).unwrap();
    let trap_raw = GpuBuffer::zeros(st, bt * nh).unwrap();
    let b_normed = DtypedBuf::zeros(st, bt * ng * ds, dt_ty).unwrap();
    let c_normed = DtypedBuf::zeros(st, bt * ng * ds, dt_ty).unwrap();
    let b_rms = GpuBuffer::zeros(st, bt * ng).unwrap();
    let c_rms = GpuBuffer::zeros(st, bt * ng).unwrap();
    let b_biased = DtypedBuf::zeros(st, bt * nh * ds, dt_ty).unwrap();
    let c_biased = DtypedBuf::zeros(st, bt * nh * ds, dt_ty).unwrap();
    let kk = DtypedBuf::zeros(st, bt * nh * ds, dt_ty).unwrap();
    let qq = DtypedBuf::zeros(st, bt * nh * ds, dt_ty).unwrap();
    let angle_state = GpuBuffer::zeros(st, batch * nh * na).unwrap();
    let nc = t.div_ceil(64);
    let sums =
        mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(st, batch * nc * nh * na * 8)
            .unwrap();
    let carries =
        mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(st, batch * nc * nh * na * 8)
            .unwrap();
    let alpha = GpuBuffer::zeros(st, bt * nh).unwrap();
    let beta = GpuBuffer::zeros(st, bt * nh).unwrap();
    let gamma = GpuBuffer::zeros(st, bt * nh).unwrap();

    let (di_i, ng_i, ds_i, nh_i, na_i) = (di as i32, ng as i32, ds as i32, nh as i32, na as i32);
    let a_floor: f32 = 1e-4;
    let eps: f32 = 1e-5;

    let split = || {
        let zc = z.cached_ptr();
        let xc = x.cached_ptr();
        let br = b_raw.cached_ptr();
        let cr = c_raw.cached_ptr();
        let pj = proj.cached_ptr();
        let db = dt_bias.cached_ptr();
        let n_i = bt as i32;
        let mut b = ctx.stream.launch_builder(m3k.m3_split_typed.get(dt_ty));
        b.arg(&zc);
        b.arg(&xc);
        b.arg(&br);
        b.arg(&cr);
        b.arg(dtc.inner());
        b.arg(a_val.inner());
        b.arg(trap.inner());
        b.arg(angles_raw.inner());
        b.arg(dd_dt.inner());
        b.arg(dd_a.inner());
        b.arg(trap_raw.inner());
        b.arg(&pj);
        b.arg(&db);
        b.arg(&a_floor);
        b.arg(&n_i);
        b.arg(&di_i);
        b.arg(&ng_i);
        b.arg(&ds_i);
        b.arg(&nh_i);
        b.arg(&na_i);
        unsafe { b.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(bt * ip)) }.unwrap();
    };
    let bcnorm = || {
        let n_i = bt as i32;
        let bn = b_normed.cached_ptr();
        let cn = c_normed.cached_ptr();
        let br = b_raw.cached_ptr();
        let cr = c_raw.cached_ptr();
        let bw = bnw.cached_ptr();
        let cw = cnw.cached_ptr();
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 2, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut b = ctx
            .stream
            .launch_builder(m3k.bcnorm_fwd_bc_typed.get(dt_ty));
        b.arg(&bn);
        b.arg(&cn);
        b.arg(b_rms.inner());
        b.arg(c_rms.inner());
        b.arg(&br);
        b.arg(&cr);
        b.arg(&bw);
        b.arg(&cw);
        b.arg(&n_i);
        b.arg(&ng_i);
        b.arg(&ds_i);
        b.arg(&eps);
        let src_stride = (ng * ds) as i32;
        b.arg(&src_stride);
        unsafe { b.launch(cfg) }.unwrap();
    };
    let angle_out = std::cell::RefCell::new(GpuBuffer::zeros(st, bt * nh * na).unwrap());
    let angles = || {
        mamba_rs::mamba3_siso::gpu::forward::gpu_angle_chunked_fwd(
            &ctx,
            &m3k,
            &mut angle_out.borrow_mut(),
            angle_state.cached_ptr(),
            &angles_raw,
            &dtc,
            &sums,
            &carries,
            batch,
            t,
            nh,
            na,
        )
        .unwrap();
    };
    let bias_rope = || {
        let n_i = bt as i32;
        let bb = b_biased.cached_ptr();
        let cb = c_biased.cached_ptr();
        let kp = kk.cached_ptr();
        let qp = qq.cached_ptr();
        let bn = b_normed.cached_ptr();
        let cn = c_normed.cached_ptr();
        let bbp = b_bias.cached_ptr();
        let cbp = c_bias.cached_ptr();
        let mut b = ctx
            .stream
            .launch_builder(m3k.m3_bias_rope_fwd_typed.get(dt_ty));
        b.arg(&bb);
        b.arg(&cb);
        b.arg(&kp);
        b.arg(&qp);
        b.arg(&bn);
        b.arg(&cn);
        b.arg(&bbp);
        b.arg(&cbp);
        let ac = angle_out.borrow().cached_ptr();
        b.arg(&ac);
        b.arg(&n_i);
        b.arg(&nh_i);
        b.arg(&ng_i);
        b.arg(&ds_i);
        b.arg(&na_i);
        unsafe { b.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(bt * nh * ds)) }.unwrap();
    };
    let abg = || {
        let n_total = (bt * nh) as i32;
        let mut b = ctx.stream.launch_builder(&m3k.m3_compute_abg);
        b.arg(alpha.inner());
        b.arg(beta.inner());
        b.arg(gamma.inner());
        b.arg(dtc.inner());
        b.arg(a_val.inner());
        b.arg(trap.inner());
        b.arg(&n_total);
        unsafe { b.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(bt * nh)) }.unwrap();
    };

    split();
    bcnorm();
    angles();
    bias_rope();
    abg();
    ctx.stream.synchronize().unwrap();

    let hash_typed = |b: &DtypedBuf, n: usize| -> u64 {
        let mut h = vec![0.0f32; n];
        b.download_f32(st, &mut h).unwrap();
        fnv(&h)
    };
    let hash_f32 = |b: &GpuBuffer, n: usize| -> u64 {
        let mut h = vec![0.0f32; n];
        b.download(st, &mut h).unwrap();
        fnv(&h)
    };
    println!(
        "COEFF HASH z={:016x} x={:016x} dt={:016x} a_val={:016x} trap={:016x} angles_raw={:016x}",
        hash_typed(&z, bt * di),
        hash_typed(&x, bt * di),
        hash_f32(&dtc, bt * nh),
        hash_f32(&a_val, bt * nh),
        hash_f32(&trap, bt * nh),
        hash_f32(&angles_raw, bt * na),
    );
    println!(
        "COEFF HASH bn={:016x} cn={:016x} cumsum={:016x} k={:016x} q={:016x} abg={:016x}",
        hash_typed(&b_normed, bt * ng * ds),
        hash_typed(&c_normed, bt * ng * ds),
        hash_f32(&angle_out.borrow(), bt * nh * na),
        hash_typed(&kk, bt * nh * ds),
        hash_typed(&qq, bt * nh * ds),
        hash_f32(&gamma, bt * nh),
    );

    let iters = 100usize;
    let time = |f: &dyn Fn()| -> f64 {
        f();
        ctx.stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        ctx.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    println!("split       {:8.1} us", time(&split));
    println!("bcnorm      {:8.1} us", time(&bcnorm));
    println!("angle_chain {:8.1} us", time(&angles));
    println!("bias_rope   {:8.1} us", time(&bias_rope));
    println!("abg         {:8.1} us", time(&abg));
}
