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

fn fnv(bits: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in bits {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[test]
#[ignore = "needs a CUDA device"]
fn scan_trio_time_and_hash() {
    // Prism serve shape.
    let batch = 1usize;
    let t = 4621usize;
    let nh = 24usize;
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
        b.arg(q.inner());
        b.arg(k.inner());
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
}
