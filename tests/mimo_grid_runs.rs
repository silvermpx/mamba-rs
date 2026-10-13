//! Mamba-3 MIMO kernels over grid axes past the y extent of 65535 blocks.
//! Such an axis is launched in runs, each run passing its first block. A
//! state element of the chunked state pass, whose elements fill the y axis
//! in blocks of 256, must come out in a later run with the bits of its
//! twin, an element of the first run given the same inputs. Every other
//! split launch of the family is proven at ordinary shapes by the suites
//! run with `MAMBA_RS_GRID_RUN`.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::launch::{GRID_YZ_EXTENT, y_runs};
use mamba_rs::mamba3_mimo::gpu::kernels::Mamba3MimoKernels;

/// Elements `e < PAIRS` and their twins `e + SHIFT`, two full runs on.
const PAIRS: usize = 16;
const SHIFT: usize = 2 * GRID_YZ_EXTENT as usize * 256;
const CHUNK: usize = 64;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn floats(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).expect("alloc");
    ctx.stream.synchronize().expect("sync");
    buf.upload(&ctx.stream, data).expect("upload");
    buf
}

/// The words of a buffer, read once the stream drains.
fn words(ctx: &GpuCtx, buf: &GpuBuffer) -> Vec<u32> {
    ctx.stream.synchronize().expect("sync");
    let n = buf.size_bytes() / 4;
    let mut out = vec![0u32; n];
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), buf.cached_ptr(), n * 4)
    };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

/// Whether in `w`, `outer` rows of `stride` words, every element `e <
/// PAIRS` of a row holds its twin's bits.
fn twins_match(w: &[u32], outer: usize, stride: usize) -> bool {
    (0..outer).all(|o| (0..PAIRS).all(|e| w[o * stride + e] == w[o * stride + e + SHIFT]))
}

/// The chunked state pass over a head whose state elements fill more
/// blocks of 256 than the y extent holds: every twin element's entering
/// states and final state carry the bits of its first-run twin. An element
/// reads its channel's carried value and up-projection, so the channel of
/// the twins (the first, since the pairs lie inside one channel's states)
/// is copied onto the twins' channel.
#[test]
fn the_state_pass_gives_elements_past_the_y_extent_their_twins_bits() {
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let (batch, nh, rank, ds) = (1usize, 1usize, 1usize, 256usize);
    let k = Mamba3MimoKernels::compile(ctx.stream.context(), arch::arch0(), rank)
        .expect("MIMO kernels");
    assert!(PAIRS <= ds && SHIFT.is_multiple_of(ds));
    let hd = (SHIFT + PAIRS).div_ceil(ds);
    let pn = hd * ds;
    let (p0, p1) = (0, SHIFT / ds);
    let seq = 2 * CHUNK;
    let nc = seq.div_ceil(CHUNK);
    let twin = |v: &mut [f32], rows: usize, stride: usize| {
        for r in 0..rows {
            let at = r * stride;
            v.copy_within(at..at + PAIRS, at + SHIFT);
        }
    };
    let mut ssm = det(pn, 21, 0.5);
    twin(&mut ssm, 1, pn);
    let ssm = floats(&ctx, &ssm);
    let mut contrib = det(nc * pn, 22, 0.5);
    twin(&mut contrib, nc, pn);
    let contrib = floats(&ctx, &contrib);
    let coef = floats(
        &ctx,
        &det(batch * seq * nh * 4, 23, 0.5)
            .into_iter()
            .map(|v| -v.abs())
            .collect::<Vec<_>>(),
    );
    let e0 = floats(&ctx, &det(batch * nh, 24, 0.5));
    let kst = floats(&ctx, &det(batch * nh * rank * ds, 25, 0.5));
    let mut vst = det(batch * nh * hd, 26, 0.5);
    vst[p1] = vst[p0];
    let vst = floats(&ctx, &vst);
    let mut mx = det(nh * rank * hd, 27, 0.5);
    for r in 0..rank {
        mx[r * hd + p1] = mx[r * hd + p0];
    }
    let mx = floats(&ctx, &mx);
    let enter = GpuBuffer::zeros(&ctx.stream, nc * pn).expect("enter");
    let cfg = LaunchConfig {
        grid_dim: ((batch * nh) as u32, pn.div_ceil(256) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    assert!(cfg.grid_dim.1 > GRID_YZ_EXTENT);
    let ints = [batch, seq, nh, hd, ds, rank].map(|v| v as i32);
    let ptrs = [
        enter.cached_ptr(),
        ssm.cached_ptr(),
        contrib.cached_ptr(),
        coef.cached_ptr(),
        e0.cached_ptr(),
        kst.cached_ptr(),
        vst.cached_ptr(),
        mx.cached_ptr(),
    ];
    for (y_first, cfg) in y_runs(cfg) {
        let mut b = ctx.stream.launch_builder(&k.state_pass);
        for p in &ptrs {
            b.arg(p);
        }
        for v in &ints {
            b.arg(v);
        }
        b.arg(&y_first);
        unsafe { b.launch(cfg) }.expect("state pass");
    }
    assert!(twins_match(&words(&ctx, &enter), nc, pn), "entering states");
    assert!(twins_match(&words(&ctx, &ssm), 1, pn), "final state");
}
