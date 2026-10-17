//! Micro-instrument for the SSM chunk kernels that dominate the classifier
//! page (the pre-state, the intra-chunk scan, the state pass), each on the
//! kernels its shapes take: per-kernel wall time at the serve shape and the
//! shapes where the kernels part, plus an FNV hash of the output bits, so a
//! scheduling/occupancy change can prove itself bit-preserving against a
//! pre-change run of this same test.
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::launch::y_runs;
use mamba_rs::mamba3_siso::gpu::kernels::{
    ChunkArm, Mamba3Kernels, bcnorm_fwd_bc_cfg, chunk_fused_cfg, chunk_pre_state_cfg,
    chunk_pre_state_sliced_cfg, chunk_scan_cfg, chunk_scan_sliced_cfg,
};
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

#[path = "../tests/common/digest.rs"]
mod digest;
#[path = "../tests/common/evidence.rs"]
mod evidence;
#[path = "../tests/common/evidence_digest.rs"]
mod evidence_digest;
use digest::fnv1a_f32 as fnv;

/// The chunked forward's stages at the serve shape and at the shapes where
/// their kernels part: the pre-state on its fused and its sliced kernel,
/// the intra-chunk scan on the kernel its route takes and on the sliced
/// one, each timed, the pre-state's arms held to each other bit for bit.
/// At the serve shape the outputs' hashes are the acceptance digests, and
/// the state pass runs on the pre-state's chunk states with its digests.
fn scan_trio_time_and_hash() {
    // The pre-state's outputs, one set per arm.
    struct PreState {
        k_scaled: GpuBuffer,
        qk_dot: GpuBuffer,
        scale: GpuBuffer,
        gamma: GpuBuffer,
        states: GpuBuffer,
        da: GpuBuffer,
    }
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    // (batch, t, nh, hd, ds): the vision-classifier serve shape first, then
    // the official head, a head of 32 over 64 states and the widest head.
    let shapes: [(usize, usize, usize, usize, usize); 4] = [
        (1, 4621, 48, 16, 16),
        (4, 512, 24, 64, 128),
        (4, 512, 48, 32, 64),
        (4, 512, 12, 128, 128),
    ];
    let cs = 64usize;
    for (serve, &(batch, t, nh, hd, ds)) in shapes.iter().enumerate().map(|(i, s)| (i == 0, s)) {
        let cap = mamba_rs::gpu::kernels::state_capacity(ds).expect("state capacity");
        let m3k =
            Mamba3Kernels::compile_with_state_cap(dev.context(), arch, cap).expect("m3 kernels");
        let st = &ctx.stream;
        let nc = t.div_ceil(cs);
        let d_inner = nh * hd;
        let x = GpuBuffer::from_cpu(st, &det(batch * t * d_inner, 1)).unwrap();
        let q = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 2)).unwrap();
        let k = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 3)).unwrap();
        let positive = |v: Vec<f32>| v.iter().map(|v| v.abs()).collect::<Vec<_>>();
        let dt_b = GpuBuffer::from_cpu(st, &positive(det(batch * t * nh, 4))).unwrap();
        let trap = GpuBuffer::from_cpu(st, &positive(det(batch * t * nh, 5))).unwrap();
        // The scan reads a decay cumsum and entering states of its own, as
        // it always has here, so its digest stays the scan's alone.
        let da_cumsum = GpuBuffer::from_cpu(
            st,
            &det(batch * nc * nh * cs, 6)
                .iter()
                .map(|v| -v.abs())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let prev_states = GpuBuffer::from_cpu(st, &det(batch * nc * nh * hd * ds, 7)).unwrap();
        let adt = GpuBuffer::from_cpu(
            st,
            &det(batch * t * nh, 9)
                .iter()
                .map(|v| -v.abs() * 0.05)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let d_param = GpuBuffer::from_cpu(st, &det(nh, 8)).unwrap();
        let (b_i, t_i, nh_i, hd_i, ds_i, cs_i) = (
            batch as i32,
            t as i32,
            nh as i32,
            hd as i32,
            ds as i32,
            cs as i32,
        );

        let outputs = || PreState {
            k_scaled: GpuBuffer::zeros(st, batch * t * nh * ds).unwrap(),
            qk_dot: GpuBuffer::zeros(st, batch * t * nh).unwrap(),
            scale: GpuBuffer::zeros(st, batch * t * nh).unwrap(),
            gamma: GpuBuffer::zeros(st, batch * t * nh).unwrap(),
            states: GpuBuffer::zeros(st, batch * nc * nh * hd * ds).unwrap(),
            da: GpuBuffer::zeros(st, batch * nc * nh * cs).unwrap(),
        };
        let pre_state = |kernel: &cudarc::driver::CudaFunction,
                         cfg: cudarc::driver::LaunchConfig,
                         out: &PreState| {
            for (y_first, cfg) in y_runs(cfg) {
                let mut b = ctx.stream.launch_builder(kernel);
                b.arg(out.k_scaled.inner());
                b.arg(out.qk_dot.inner());
                b.arg(out.scale.inner());
                b.arg(out.gamma.inner());
                b.arg(out.states.inner());
                b.arg(k.inner());
                b.arg(q.inner());
                b.arg(dt_b.inner());
                b.arg(trap.inner());
                b.arg(x.inner());
                b.arg(adt.inner());
                b.arg(out.da.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs_i);
                b.arg(&y_first);
                b.arg(&0u64);
                unsafe { b.launch(cfg) }.unwrap();
            }
        };
        let down = |buf: &GpuBuffer, n: usize| {
            let mut host = vec![0.0f32; n];
            buf.download(st, &mut host).unwrap();
            host
        };
        let hashes = |out: &PreState| {
            [
                fnv(&down(&out.k_scaled, batch * t * nh * ds)),
                fnv(&down(&out.qk_dot, batch * t * nh)),
                fnv(&down(&out.scale, batch * t * nh)),
                fnv(&down(&out.gamma, batch * t * nh)),
                fnv(&down(&out.states, batch * nc * nh * hd * ds)),
                fnv(&down(&out.da, batch * nc * nh * cs)),
            ]
        };
        let sliced_kernel = &m3k.m3_chunk_pre_state_sliced_typed.i32.f32;
        let sliced_cfg = chunk_pre_state_sliced_cfg(batch, nc, nh, hd);
        let sliced_out = outputs();
        pre_state(sliced_kernel, sliced_cfg, &sliced_out);
        ctx.stream.synchronize().unwrap();
        let want = hashes(&sliced_out);
        let fused_cfg = chunk_fused_cfg(batch, nc, nh, hd, ds, cs, m3k.smem_optin);
        let fused_out = outputs();
        if let Some(cfg) = fused_cfg {
            pre_state(&m3k.m3_chunk_pre_state_fused_typed.i32.f32, cfg, &fused_out);
            ctx.stream.synchronize().unwrap();
            assert_eq!(
                hashes(&fused_out),
                want,
                "B {batch} T {t} nh {nh} hd {hd} ds {ds}: the fused pre-state diverged from the sliced"
            );
        }

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
        println!("--- B {batch} T {t} nh {nh} hd {hd} ds {ds}");
        let (arm, _) = chunk_pre_state_cfg(batch, nc, nh, hd, ds, cs, m3k.smem_optin);
        println!("pre-state route      {arm:?}");
        println!(
            "pre-state sliced     {:8.1} us",
            time(&|| pre_state(sliced_kernel, sliced_cfg, &sliced_out))
        );
        if let Some(cfg) = fused_cfg {
            println!(
                "pre-state fused      {:8.1} us",
                time(&|| pre_state(&m3k.m3_chunk_pre_state_fused_typed.i32.f32, cfg, &fused_out))
            );
        }

        // The scan over the pre-state's K and dots, on its route's kernel
        // and on the sliced one.
        let y = GpuBuffer::zeros(st, batch * t * d_inner).unwrap();
        let scan = |kernel: &cudarc::driver::CudaFunction, cfg: cudarc::driver::LaunchConfig| {
            for (y_first, cfg) in y_runs(cfg) {
                let mut b = ctx.stream.launch_builder(kernel);
                b.arg(y.inner());
                b.arg(x.inner());
                b.arg(q.inner());
                b.arg(sliced_out.k_scaled.inner());
                b.arg(sliced_out.qk_dot.inner());
                b.arg(da_cumsum.inner());
                b.arg(prev_states.inner());
                b.arg(d_param.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs_i);
                b.arg(&y_first);
                b.arg(&0u64);
                unsafe { b.launch(cfg) }.unwrap();
            }
        };
        let (arm, route_cfg) = chunk_scan_cfg(batch, nc, nh, hd, ds, cs);
        let route_kernel = match arm {
            ChunkArm::Whole => &m3k.m3_chunk_scan_fwd_coop.i32,
            ChunkArm::Sliced => &m3k.m3_chunk_scan_fwd_sliced_typed.i32.f32,
        };
        scan(route_kernel, route_cfg);
        ctx.stream.synchronize().unwrap();
        let y_h = down(&y, batch * t * d_inner);
        let nonzero = y_h.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > y_h.len() / 2,
            "scan output mostly zero: {nonzero}"
        );
        println!("scan route           {arm:?}");
        println!(
            "scan route kernel    {:8.1} us",
            time(&|| scan(route_kernel, route_cfg))
        );
        println!(
            "scan sliced          {:8.1} us",
            time(&|| scan(
                &m3k.m3_chunk_scan_fwd_sliced_typed.i32.f32,
                chunk_scan_sliced_cfg(batch, nc, nh, hd)
            ))
        );
        if !serve {
            continue;
        }

        // The serve shape's digests: the scan's rows, the pre-state's K and
        // chunk states, then the state pass over those states.
        scan(route_kernel, route_cfg);
        ctx.stream.synchronize().unwrap();
        let y_h = down(&y, batch * t * d_inner);
        let ks_h = down(&sliced_out.k_scaled, batch * t * nh * ds);
        let cst_h = down(&sliced_out.states, batch * nc * nh * hd * ds);
        println!(
            "HASH y={:016x} k_scaled={:016x} chunk_states={:016x}",
            fnv(&y_h),
            fnv(&ks_h),
            fnv(&cst_h)
        );
        evidence_digest::record_digest("m3_scan_micro_bench", "scan_trio", "y", fnv(&y_h))
            .expect("acceptance evidence");
        evidence_digest::record_digest("m3_scan_micro_bench", "scan_trio", "k_scaled", fnv(&ks_h))
            .expect("acceptance evidence");
        evidence_digest::record_digest(
            "m3_scan_micro_bench",
            "scan_trio",
            "chunk_states",
            fnv(&cst_h),
        )
        .expect("acceptance evidence");

        // The state pass mutates its states buffer in place, so the hash
        // arm runs once on a fresh copy of the chunk states; the timing loop
        // then reuses the (already transformed) buffer - values drift there
        // but the address pattern is identical.
        let states_fp = GpuBuffer::from_cpu(st, &cst_h).unwrap();
        let ssm_out = GpuBuffer::zeros(st, batch * nh * hd * ds).unwrap();
        let k_out = GpuBuffer::zeros(st, batch * nh * ds).unwrap();
        let v_out = GpuBuffer::zeros(st, batch * nh * hd).unwrap();
        let state_pass_fused = || {
            let cfg = mamba_rs::mamba3_siso::gpu::kernels::state_pass_fused_cfg(batch, nh, hd, ds);
            let nc_i = nc as i32;
            let seed_null: u64 = 0;
            for (y_first, cfg) in y_runs(cfg) {
                let mut b = ctx
                    .stream
                    .launch_builder(&m3k.m3_state_pass_fused_typed.i32.f32);
                b.arg(states_fp.inner());
                b.arg(ssm_out.inner());
                b.arg(k_out.inner());
                b.arg(v_out.inner());
                b.arg(&seed_null);
                b.arg(k_out.inner());
                b.arg(v_out.inner());
                b.arg(dt_b.inner());
                b.arg(trap.inner());
                b.arg(da_cumsum.inner());
                b.arg(k.inner());
                b.arg(x.inner());
                b.arg(&b_i);
                b.arg(&nc_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs_i);
                b.arg(&t_i);
                b.arg(&y_first);
                b.arg(&0u64);
                unsafe { b.launch(cfg) }.unwrap();
            }
        };
        state_pass_fused();
        ctx.stream.synchronize().unwrap();
        let sp_h = down(&states_fp, batch * nc * nh * hd * ds);
        let fs_h = down(&ssm_out, batch * nh * hd * ds);
        println!(
            "STATE_PASSING HASH entering={:016x} final={:016x}",
            fnv(&sp_h),
            fnv(&fs_h)
        );
        evidence_digest::record_digest(
            "m3_scan_micro_bench",
            "state_passing",
            "entering",
            fnv(&sp_h),
        )
        .expect("acceptance evidence");
        evidence_digest::record_digest("m3_scan_micro_bench", "state_passing", "final", fnv(&fs_h))
            .expect("acceptance evidence");
        println!("state_pass_fused {:8.1} us", time(&state_pass_fused));
    }
}

/// The coefficient chain at the serve shape: split -> bcnorm -> angle
/// accumulation -> bias+rope -> abg, each timed and output-hashed. The
/// hashes are the bit gates for the dead-write elision and the angle
/// rework: every tensor a consumer actually reads must keep its exact
/// bits through those changes.
fn coeff_chain_time_and_hash() {
    use mamba_rs::gpu::buffers::DtypedBuf;
    use mamba_rs::gpu::dtype::WeightDtype;

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
    let sums = mamba_rs::gpu::buffers::GpuByteBuffer::zeros(st, batch * nc * nh * na * 8).unwrap();
    let carries =
        mamba_rs::gpu::buffers::GpuByteBuffer::zeros(st, batch * nc * nh * na * 8).unwrap();
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
        let mut b = ctx.stream.launch_builder(m3k.m3_split_typed.i32.get(dt_ty));
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
        unsafe { b.launch(mamba_rs::gpu::launch::grid_1d(bt * ip)) }.unwrap();
    };
    let bcnorm = || {
        let n_i = bt as i32;
        let bn = b_normed.cached_ptr();
        let cn = c_normed.cached_ptr();
        let br = b_raw.cached_ptr();
        let cr = c_raw.cached_ptr();
        let bw = bnw.cached_ptr();
        let cw = cnw.cached_ptr();
        let cfg = bcnorm_fwd_bc_cfg(bt * ng, ds);
        let mut b = ctx
            .stream
            .launch_builder(m3k.bcnorm_fwd_bc_typed.i32.get(dt_ty));
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
        // SAFETY: this benchmark owns distinct same-context buffers that
        // remain live until after every measured launch completes.
        unsafe {
            mamba_rs::mamba3_siso::gpu::forward::gpu_angle_chunked_fwd(
                &ctx,
                &m3k,
                mamba_rs::mamba3_siso::gpu::forward::AngleChunkedFwd {
                    angle_cumsum: &mut angle_out.borrow_mut(),
                    angle_state_ptr: angle_state.cached_ptr(),
                    angles_raw: &angles_raw,
                    dt: &dtc,
                    sums: &sums,
                    carries: &carries,
                    shape: mamba_rs::mamba3_siso::gpu::forward::AngleChunkedShape {
                        batch,
                        seq_len: t,
                        heads: nh,
                        angles: na,
                    },
                    seq_idx: 0,
                },
            )
        }
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
            .launch_builder(m3k.m3_bias_rope_fwd_typed.i32.get(dt_ty));
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
        // No angle state on this lane: the saved angles are read.
        let no_angle: cudarc::driver::sys::CUdeviceptr = 0;
        b.arg(&no_angle);
        b.arg(&no_angle);
        b.arg(&no_angle);
        unsafe { b.launch(mamba_rs::gpu::launch::grid_1d(bt * nh * ds)) }.unwrap();
    };
    let abg = || {
        let n_total = (bt * nh) as i32;
        let mut b = ctx.stream.launch_builder(&m3k.m3_compute_abg.i32);
        b.arg(alpha.inner());
        b.arg(beta.inner());
        b.arg(gamma.inner());
        b.arg(dtc.inner());
        b.arg(a_val.inner());
        b.arg(trap.inner());
        b.arg(&n_total);
        let (t_i, nh_i) = (t as i32, nh as i32);
        b.arg(&t_i);
        b.arg(&nh_i);
        b.arg(&0u64);
        unsafe { b.launch(mamba_rs::gpu::launch::grid_1d(bt * nh)) }.unwrap();
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

// The sequential target burn-in at the serve shape: wall time plus an FNV
// hash of everything it writes. This lane has no ledger of its own, so a
// scheduling or occupancy change could move its bits with nothing to say
// so; the chunk trio above has had one since the 0.6 pass and this gives
// the sequential twin the same footing. The serve prefill does NOT run
// here - it takes the chunked pipeline - so the number is the burn-in's
// own cost, not a prefill wall.
fn sequential_prefill_time_and_hash() {
    let batch = 1usize;
    let t = 4621usize;
    let nh = 48usize;
    let hd = 16usize;
    let ds = 16usize;
    let d_inner = nh * hd;

    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(dev.compute_capability);
    let m3k = Mamba3Kernels::compile(dev.context(), arch).expect("m3 kernels");
    let st = &ctx.stream;

    let x = GpuBuffer::from_cpu(st, &det(batch * t * d_inner, 11)).unwrap();
    let k = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 12)).unwrap();
    let q = GpuBuffer::from_cpu(st, &det(batch * t * nh * ds, 13)).unwrap();
    let alpha = GpuBuffer::from_cpu(st, &det(batch * t * nh, 14)).unwrap();
    let beta = GpuBuffer::from_cpu(st, &det(batch * t * nh, 15)).unwrap();
    let gamma = GpuBuffer::from_cpu(st, &det(batch * t * nh, 16)).unwrap();
    let d_skip = GpuBuffer::from_cpu(st, &det(nh, 17)).unwrap();

    let ssm_state = GpuBuffer::zeros(st, batch * nh * hd * ds).unwrap();
    let k_state = GpuBuffer::zeros(st, batch * nh * ds).unwrap();
    let v_state = GpuBuffer::zeros(st, batch * nh * hd).unwrap();
    let y_out = GpuBuffer::zeros(st, batch * t * d_inner).unwrap();

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, nh as u32, 1),
        block_dim: (hd as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (batch_i, t_i, nh_i, hd_i, ds_i) =
        (batch as i32, t as i32, nh as i32, hd as i32, ds as i32);
    let prefill = || {
        for (y_first, cfg) in y_runs(cfg) {
            let mut b = st.launch_builder(&m3k.m3_burnin_fwd_nosave.i32);
            b.arg(ssm_state.inner());
            b.arg(k_state.inner());
            b.arg(v_state.inner());
            b.arg(y_out.inner());
            b.arg(x.inner());
            b.arg(k.inner());
            b.arg(q.inner());
            b.arg(alpha.inner());
            b.arg(beta.inner());
            b.arg(gamma.inner());
            b.arg(d_skip.inner());
            b.arg(&batch_i);
            b.arg(&t_i);
            b.arg(&nh_i);
            b.arg(&hd_i);
            b.arg(&ds_i);
            b.arg(&y_first);
            b.arg(&0u64);
            unsafe { b.launch(cfg) }.unwrap();
        }
    };

    let iters = 20usize;
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
    let wall = time(&prefill);

    let mut y_h = vec![0.0f32; batch * t * d_inner];
    y_out.download(st, &mut y_h).unwrap();
    let mut ssm_h = vec![0.0f32; batch * nh * hd * ds];
    ssm_state.download(st, &mut ssm_h).unwrap();
    let mut k_h = vec![0.0f32; batch * nh * ds];
    k_state.download(st, &mut k_h).unwrap();
    let mut v_h = vec![0.0f32; batch * nh * hd];
    v_state.download(st, &mut v_h).unwrap();
    println!(
        "PREFILL HASH y={:016x} ssm={:016x} k={:016x} v={:016x}",
        fnv(&y_h),
        fnv(&ssm_h),
        fnv(&k_h),
        fnv(&v_h)
    );
    println!("prefill_seq {wall:8.1} us");
}

// Run every instrument, or only the ones named on the command line:
// `cargo bench --bench <target> --features cuda -- <name> [<name> ...]`.
fn main() {
    // cargo bench passes its own `--bench` flag; only the other arguments
    // name instruments.
    let selected: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"))
        .collect();
    let run = |name: &str| selected.is_empty() || selected.iter().any(|s| s == name);
    if run("scan_trio_time_and_hash") {
        scan_trio_time_and_hash();
    }
    if run("coeff_chain_time_and_hash") {
        coeff_chain_time_and_hash();
    }
    if run("sequential_prefill_time_and_hash") {
        sequential_prefill_time_and_hash();
    }
}
