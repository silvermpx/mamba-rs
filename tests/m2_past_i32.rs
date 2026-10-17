//! Mamba-2's chunked kernels past 2^31 elements. They index with 64 bits in
//! a single build, so a window whose conv rows or chunk states pass the
//! 32-bit range runs on the same kernels as a short one; these tests launch
//! such windows and compare their first and last chunks, and the last
//! sequence of a batch, with launches of those chunks or that sequence
//! alone.
//!
//!   cargo test --release --features cuda --test m2_past_i32 -- --ignored --nocapture
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use std::sync::Arc;

use cudarc::driver::sys::CUresult::CUDA_SUCCESS;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::launch::y_runs;
use mamba_rs::mamba2::gpu::kernels::{
    CHUNK, CHUNK_SCAN_TILES_SMEM_BYTES, CHUNK_SCAN_TT, Mamba2Kernels, bmm_smem_bytes,
    chunk_state_geom,
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

fn floats(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut buf = GpuBuffer::zeros(&ctx.stream, data.len()).expect("alloc");
    ctx.stream.synchronize().expect("sync");
    buf.upload(&ctx.stream, data).expect("upload");
    buf
}

fn zeros(ctx: &GpuCtx, n: usize) -> GpuBuffer {
    GpuBuffer::zeros(&ctx.stream, n).expect("alloc")
}

fn read(stream: &Arc<CudaStream>, ptr: u64, n: usize) -> Vec<u32> {
    stream.synchronize().expect("sync");
    let mut out = vec![0u32; n];
    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, n * 4) };
    assert_eq!(r, CUDA_SUCCESS, "read");
    out
}

fn copy(stream: &Arc<CudaStream>, dst: u64, src: u64, n: usize) {
    let r =
        unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, n * 4, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "copy");
}

fn fill(stream: &Arc<CudaStream>, ptr: u64, word: u32, n: usize) {
    let r = unsafe { cudarc::driver::sys::cuMemsetD32Async(ptr, word, n, stream.cu_stream()) };
    assert_eq!(r, CUDA_SUCCESS, "memset");
}

fn at(ptr: u64, n: usize) -> u64 {
    ptr + (n * 4) as u64
}

#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    I(i32),
    L(i64),
}

fn launch(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    let mut b = ctx.stream.launch_builder(f);
    for a in args {
        match a {
            Arg::P(p) => b.arg(p),
            Arg::I(i) => b.arg(i),
            Arg::L(l) => b.arg(l),
        };
    }
    unsafe { b.launch(cfg) }.expect("launch");
}

/// [`launch`] once per run of the grid's y axis, each run's first block
/// appended, then the packing the window kernels take last (none).
fn launch_runs(ctx: &GpuCtx, f: &CudaFunction, cfg: LaunchConfig, args: &[Arg]) {
    for (y_first, cfg) in y_runs(cfg) {
        let mut run = args.to_vec();
        run.extend([Arg::I(y_first), Arg::P(0)]);
        launch(ctx, f, cfg, &run);
    }
}

fn assert_bits(what: &str, a: &[u32], b: &[u32]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!(
            "{what}: element {i} of {} differs: {:#010x} != {:#010x}",
            a.len(),
            a[i],
            b[i]
        );
    }
}

/// A chunked window's shape and the conv row's columns.
#[derive(Clone, Copy)]
struct Win {
    b: usize,
    t: usize,
    nh: usize,
    ng: usize,
    hd: usize,
    ds: usize,
}

impl Win {
    fn nc(&self) -> usize {
        self.t.div_ceil(CHUNK)
    }
    fn di(&self) -> usize {
        self.nh * self.hd
    }
    /// The conv row: x, then B, then C.
    fn ld(&self) -> usize {
        self.di() + 2 * self.ng * self.ds
    }
    fn pairs(&self) -> usize {
        CHUNK * (CHUNK + 1) / 2
    }
}

/// The window's buffers from chunk `c` on.
#[derive(Clone, Copy)]
struct ChunkPtrs {
    xbc: u64,
    dt: u64,
    acs: u64,
    cb: u64,
    states: u64,
    y: u64,
    d: u64,
}

impl ChunkPtrs {
    fn past(&self, w: Win, c: usize) -> Self {
        let t0 = c * CHUNK;
        Self {
            xbc: at(self.xbc, t0 * w.ld()),
            dt: at(self.dt, c * w.nh * CHUNK),
            acs: at(self.acs, c * w.nh * CHUNK),
            cb: at(self.cb, c * w.ng * w.pairs()),
            states: at(self.states, c * w.nh * w.hd * w.ds),
            y: at(self.y, t0 * w.di()),
            d: self.d,
        }
    }
}

fn ints(w: Win) -> [Arg; 2] {
    [Arg::I(w.b as i32), Arg::I(w.t as i32)]
}

/// The chunk-local kernels of the window: the C B products, the chunk
/// states and the outputs.
fn run_chunks(ctx: &GpuCtx, k: &Mamba2Kernels, w: Win, p: ChunkPtrs) {
    let (ld, b_col, c_col) = (w.ld() as i64, w.di() as i32, (w.di() + w.ng * w.ds) as i32);
    let (nh, ng, hd, ds) = (
        Arg::I(w.nh as i32),
        Arg::I(w.ng as i32),
        Arg::I(w.hd as i32),
        Arg::I(w.ds as i32),
    );
    let [b, t] = ints(w);
    launch_runs(
        ctx,
        k.bmm_chunk.get(mamba_rs::gpu::dtype::WeightDtype::F32),
        LaunchConfig {
            grid_dim: ((w.b * w.nc()) as u32, w.ng as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: bmm_smem_bytes(w.ds) as u32,
        },
        &[
            Arg::P(p.cb),
            Arg::P(p.xbc),
            Arg::L(ld),
            Arg::I(b_col),
            Arg::I(c_col),
            b,
            t,
            ng,
            ds,
        ],
    );
    let (cols, pt, smem) = chunk_state_geom(w.hd, w.ds, k.smem_optin);
    launch_runs(
        ctx,
        k.chunk_state.get(mamba_rs::gpu::dtype::WeightDtype::F32),
        LaunchConfig {
            grid_dim: (
                (w.b * w.nc()) as u32,
                w.nh as u32,
                w.ds.div_ceil(cols) as u32,
            ),
            block_dim: ((32 * pt / 4).min(256) as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        },
        &[
            Arg::P(p.states),
            Arg::P(p.xbc),
            Arg::L(ld),
            Arg::I(b_col),
            Arg::P(p.dt),
            Arg::P(p.acs),
            b,
            t,
            nh,
            ng,
            hd,
            ds,
            Arg::I(pt as i32),
        ],
    );
    launch_runs(
        ctx,
        k.chunk_scan_tiles
            .get(mamba_rs::gpu::dtype::WeightDtype::F32),
        LaunchConfig {
            grid_dim: (
                (w.b * w.nc()) as u32,
                w.nh as u32,
                CHUNK.div_ceil(CHUNK_SCAN_TT) as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: CHUNK_SCAN_TILES_SMEM_BYTES as u32,
        },
        &[
            Arg::P(p.y),
            Arg::P(p.states),
            Arg::P(p.cb),
            Arg::P(p.xbc),
            Arg::L(ld),
            Arg::I(c_col),
            Arg::P(p.dt),
            Arg::P(p.acs),
            Arg::P(p.d),
            Arg::I(0),
            b,
            t,
            nh,
            ng,
            hd,
            ds,
        ],
    );
}

/// The decay sums of every chunk of `w`'s heads: nonpositive and falling
/// inside a chunk, as the dt kernel writes them.
fn decay_sums(w: Win, seed: u32) -> Vec<f32> {
    let steps = det(w.b * w.nc() * w.nh * CHUNK, seed, 0.05);
    let mut out = vec![0.0f32; steps.len()];
    for (row, src) in out.chunks_mut(CHUNK).zip(steps.chunks(CHUNK)) {
        let mut acc = 0.0f32;
        for (o, v) in row.iter_mut().zip(src) {
            acc -= v.abs();
            *o = acc;
        }
    }
    out
}

/// Past 2^31 elements of the conv rows the C B products, the chunk states
/// and the outputs of a window's first and last chunks are what those
/// chunks give in a launch of their own.
#[test]
#[ignore = "14 GiB of conv rows and states; run on a 32 GB card"]
fn the_chunk_kernels_give_chunks_past_i32_max_their_own_bits() {
    let w = Win {
        b: 1,
        t: 3_801_088,
        nh: 1,
        ng: 1,
        hd: 64,
        ds: 256,
    };
    assert!(w.t * w.ld() > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = Mamba2Kernels::compile(ctx.stream.context(), arch::arch0()).expect("M2 kernels");

    // The conv rows hold one word everywhere but in the chunks compared, so
    // a row read at a wrapped offset reads other bits than its own.
    let window = 64;
    let rows = w.t * w.ld();
    let xbc = zeros(&ctx, rows);
    fill(st, xbc.cached_ptr(), 0x3e80_0000, rows);
    let firsts = [0, w.nc() - window];
    for (i, &c) in firsts.iter().enumerate() {
        let n = window * CHUNK * w.ld();
        let vals = floats(&ctx, &det(n, 70 + i as u32, 1.0));
        copy(
            st,
            at(xbc.cached_ptr(), c * CHUNK * w.ld()),
            vals.cached_ptr(),
            n,
        );
    }
    let dt = floats(&ctx, &det(w.nc() * w.nh * CHUNK, 72, 0.5));
    let acs = floats(&ctx, &decay_sums(w, 73));
    let cb = zeros(&ctx, w.nc() * w.ng * w.pairs());
    let states = zeros(&ctx, w.nc() * w.nh * w.hd * w.ds);
    let y = zeros(&ctx, w.t * w.di());
    let d = floats(&ctx, &det(w.nh, 74, 1.0));
    let all = ChunkPtrs {
        xbc: xbc.cached_ptr(),
        dt: dt.cached_ptr(),
        acs: acs.cached_ptr(),
        cb: cb.cached_ptr(),
        states: states.cached_ptr(),
        y: y.cached_ptr(),
        d: d.cached_ptr(),
    };
    let own = Win {
        t: window * CHUNK,
        ..w
    };
    run_chunks(&ctx, &k, w, all);
    for &c in &firsts {
        let own_cb = zeros(&ctx, window * w.ng * w.pairs());
        let own_states = zeros(&ctx, window * w.nh * w.hd * w.ds);
        let own_y = zeros(&ctx, own.t * w.di());
        let p = ChunkPtrs {
            cb: own_cb.cached_ptr(),
            states: own_states.cached_ptr(),
            y: own_y.cached_ptr(),
            ..all.past(w, c)
        };
        run_chunks(&ctx, &k, own, p);
        let past = all.past(w, c);
        for (what, a, b, n) in [
            ("C B", past.cb, p.cb, window * w.ng * w.pairs()),
            ("states", past.states, p.states, window * w.nh * w.hd * w.ds),
            ("y", past.y, p.y, own.t * w.di()),
        ] {
            assert_bits(
                &format!("chunks from {c}: {what}"),
                &read(st, a, n),
                &read(st, b, n),
            );
        }
    }
}

/// Past 2^31 elements of the chunk states the state passing gives the last
/// sequence of a batch, whose states lie wholly past the range, what that
/// sequence gives alone.
#[test]
#[ignore = "14 GiB of chunk states; run on a 32 GB card"]
fn the_state_passing_gives_a_sequence_past_i32_max_its_own_bits() {
    let w = Win {
        b: 4,
        t: 2_801_664,
        nh: 1,
        ng: 1,
        hd: 64,
        ds: 256,
    };
    let pn = w.hd * w.ds;
    let per_seq = w.nc() * w.nh * pn;
    assert!((w.b - 1) * per_seq > i32::MAX as usize);
    let device = GpuDevice::new(0).expect("open CUDA device");
    let ctx = GpuCtx::new(&device).expect("context");
    let st = &ctx.stream;
    let k = Mamba2Kernels::compile(ctx.stream.context(), arch::arch0()).expect("M2 kernels");
    let last = w.b - 1;

    let states = zeros(&ctx, w.b * per_seq);
    fill(st, states.cached_ptr(), 0x3c00_0000, w.b * per_seq);
    let own_in = floats(&ctx, &det(per_seq, 80, 0.5));
    copy(
        st,
        at(states.cached_ptr(), last * per_seq),
        own_in.cached_ptr(),
        per_seq,
    );
    let ssm = floats(&ctx, &det(w.b * w.nh * pn, 81, 0.5));
    let acs = floats(&ctx, &decay_sums(w, 82));
    let pass = |w: Win, states: u64, ssm: u64, acs: u64| {
        launch_runs(
            &ctx,
            &k.state_passing,
            LaunchConfig {
                grid_dim: ((w.b * w.nh) as u32, pn.div_ceil(256) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &[
                Arg::P(states),
                Arg::P(ssm),
                Arg::P(acs),
                Arg::I(w.b as i32),
                Arg::I(w.t as i32),
                Arg::I(w.nh as i32),
                Arg::I(w.hd as i32),
                Arg::I(w.ds as i32),
            ],
        );
    };
    // The last sequence's carried state, before the batched pass moves it.
    let own_ssm = zeros(&ctx, w.nh * pn);
    copy(
        st,
        own_ssm.cached_ptr(),
        at(ssm.cached_ptr(), last * w.nh * pn),
        w.nh * pn,
    );
    pass(w, states.cached_ptr(), ssm.cached_ptr(), acs.cached_ptr());
    let one = Win { b: 1, ..w };
    pass(
        one,
        own_in.cached_ptr(),
        own_ssm.cached_ptr(),
        at(acs.cached_ptr(), last * w.nc() * w.nh * CHUNK),
    );
    assert_bits(
        "entering states",
        &read(st, at(states.cached_ptr(), last * per_seq), per_seq),
        &read(st, own_in.cached_ptr(), per_seq),
    );
    assert_bits(
        "carried state",
        &read(st, at(ssm.cached_ptr(), last * w.nh * pn), w.nh * pn),
        &read(st, own_ssm.cached_ptr(), w.nh * pn),
    );
}
