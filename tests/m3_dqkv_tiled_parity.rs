//! The tiled chunked dQKV kernels against the warp kernels they extend.
//!
//! Where the warp kernel runs a shape, every output of the tiled stage must
//! carry its bits: the per-chunk state terms, and dQ, dK, dV, dADT, dQK and
//! the dD partials, in f32, bf16 and f16, with the pair matrices and without.
//! At widths only the tiled kernels serve, the register-tiled stage must give
//! the bits of the staged stage it replaced (frozen in
//! `fixtures/m3_dqkv_tiled_reference.cu`) at every tile width that one took,
//! and the state terms must still match the original kernel wherever that
//! one runs. The training parity suite checks the values against the CPU.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::sys::CUfunction_attribute_enum as CUfunction_attribute;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use frozen_kernels::frozen_module;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::yz_runs;
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, dm_chain_cfg, state_terms_tiled_cfg};
use mamba_rs::mamba3_siso::route::{
    PLANNED_SMEM, chunked_backward_smem_bytes, chunked_backward_tiled_smem_bytes,
};

const B: usize = 2;
const NH: usize = 2;
const CS: usize = 64;

/// The staged dQKV stage as it was released before the register-tiled
/// stage replaced it, its tile width the caller's.
const FROZEN_TILED: &str = include_str!("fixtures/m3_dqkv_tiled_reference.cu");

/// The staged stage's shared memory for tiles `tile` columns wide: nine
/// per-step lanes, the three pair triangles, four chunk-by-tile operand
/// tiles and two tile-by-tile state tiles, each row padded by one float.
fn staged_smem_bytes(tile: usize) -> usize {
    let row = tile + 1;
    (9 * CS + 3 * (CS * (CS - 1) / 2) + 4 * CS * row + 2 * tile * row) * 4
}

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn upload(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    b.upload(&ctx.stream, data).unwrap();
    b
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    b.upload_f32(&ctx.stream, data).unwrap();
    b
}

fn download(ctx: &GpuCtx, buf: &GpuBuffer) -> Vec<f32> {
    ctx.stream.synchronize().unwrap();
    buf.to_cpu(&ctx.stream).unwrap()
}

fn blocks(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Every element bit for bit, naming the first difference.
fn assert_bits(label: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{label}: lengths");
    let differ: Vec<usize> = want
        .iter()
        .zip(got)
        .enumerate()
        .filter(|(_, (w, g))| w.to_bits() != g.to_bits())
        .map(|(i, _)| i)
        .collect();
    if let Some(&i) = differ.first() {
        panic!(
            "{label}: {} of {} elements differ, first at {i}: want {:e} ({:#010x}), \
             got {:e} ({:#010x})",
            differ.len(),
            want.len(),
            want[i],
            want[i].to_bits(),
            got[i],
            got[i].to_bits()
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    hd: usize,
    ds: usize,
    t: usize,
}

impl Shape {
    fn nc(&self) -> usize {
        self.t.div_ceil(CS)
    }
    fn dims(&self) -> [i32; 6] {
        [B, self.t, NH, self.hd, self.ds, CS].map(|v| v as i32)
    }
}

/// One layer's inputs to the stage.
struct Inputs {
    q: DtypedBuf,
    k: DtypedBuf,
    v: DtypedBuf,
    d_o: DtypedBuf,
    da_cs: GpuBuffer,
    da_cs_sum: GpuBuffer,
    qk: GpuBuffer,
    states: GpuBuffer,
    d_param: GpuBuffer,
}

impl Inputs {
    fn new(ctx: &GpuCtx, s: Shape, dtype: WeightDtype) -> Self {
        let (nc, di) = (s.nc(), NH * s.hd);
        let seed = (s.hd * 131 + s.ds * 17 + s.t) as u32;
        // A decay cumsum as the forward saves it: negative steps summed
        // within each chunk.
        let steps = det(B * nc * NH * CS, seed ^ 0x51, 1.0);
        let mut da_cs = vec![0f32; B * nc * NH * CS];
        let mut da_cs_sum = vec![0f32; B * nc * NH];
        for row in 0..B * nc * NH {
            let chunk = (row / NH) % nc;
            let len = CS.min(s.t - chunk * CS);
            let mut acc = 0f32;
            for t in 0..len {
                acc -= 0.03 + 0.02 * steps[row * CS + t];
                da_cs[row * CS + t] = acc;
            }
            da_cs_sum[row] = da_cs[row * CS + len - 1];
        }
        Self {
            q: upload_typed(ctx, &det(B * s.t * NH * s.ds, seed ^ 0x11, 1.0), dtype),
            k: upload_typed(ctx, &det(B * s.t * NH * s.ds, seed ^ 0x22, 1.0), dtype),
            v: upload_typed(ctx, &det(B * s.t * di, seed ^ 0x33, 1.0), dtype),
            d_o: upload_typed(ctx, &det(B * s.t * di, seed ^ 0x44, 1.0), dtype),
            da_cs: upload(ctx, &da_cs),
            da_cs_sum: upload(ctx, &da_cs_sum),
            qk: upload(ctx, &det(B * s.t * NH, seed ^ 0x55, 1.0)),
            states: upload(ctx, &det(B * nc * NH * s.hd * s.ds, seed ^ 0x66, 1.0)),
            d_param: upload(ctx, &[0.5, -0.25]),
        }
    }
}

/// The stage's six outputs.
struct Outputs {
    d_q: GpuBuffer,
    d_k: GpuBuffer,
    d_v: GpuBuffer,
    d_adt: GpuBuffer,
    d_qk: GpuBuffer,
    d_partials: GpuBuffer,
}

impl Outputs {
    fn new(ctx: &GpuCtx, s: Shape) -> Self {
        let z = |n: usize| GpuBuffer::zeros(&ctx.stream, n).unwrap();
        Self {
            d_q: z(B * s.t * NH * s.ds),
            d_k: z(B * s.t * NH * s.ds),
            d_v: z(B * s.t * NH * s.hd),
            d_adt: z(B * s.t * NH),
            d_qk: z(B * s.t * NH),
            d_partials: z(B * s.nc() * NH),
        }
    }

    fn ptrs(&self) -> [CUdeviceptr; 6] {
        [
            self.d_q.cached_ptr(),
            self.d_k.cached_ptr(),
            self.d_v.cached_ptr(),
            self.d_adt.cached_ptr(),
            self.d_qk.cached_ptr(),
            self.d_partials.cached_ptr(),
        ]
    }

    fn read(&self, ctx: &GpuCtx) -> [(&'static str, Vec<f32>); 6] {
        [
            ("dQ", download(ctx, &self.d_q)),
            ("dK", download(ctx, &self.d_k)),
            ("dV", download(ctx, &self.d_v)),
            ("dADT", download(ctx, &self.d_adt)),
            ("dQK", download(ctx, &self.d_qk)),
            ("dD partials", download(ctx, &self.d_partials)),
        ]
    }
}

fn state_terms_reference(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    s: Shape,
    inp: &Inputs,
) -> GpuBuffer {
    let out = GpuBuffer::zeros(&ctx.stream, B * s.nc() * NH * s.hd * s.ds).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, B as u32, s.nc() as u32),
        block_dim: (s.hd as u32, 4, 1),
        shared_mem_bytes: ((CS * s.ds + CS * s.hd + CS) * 4) as u32,
    };
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut b = ctx
            .stream
            .launch_builder(m3k.m3_dqkv_state_terms_typed.i32.get(dtype));
        let ptrs = [
            out.cached_ptr(),
            inp.q.cached_ptr(),
            inp.da_cs.cached_ptr(),
            inp.d_o.cached_ptr(),
        ];
        for p in &ptrs {
            b.arg(p);
        }
        let dims = s.dims();
        for d in &dims {
            b.arg(d);
        }
        b.arg(&y_first);
        b.arg(&z_first);
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    out
}

fn state_terms_tiled(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    s: Shape,
    inp: &Inputs,
) -> GpuBuffer {
    let out = GpuBuffer::zeros(&ctx.stream, B * s.nc() * NH * s.hd * s.ds).unwrap();
    let cfg = state_terms_tiled_cfg(B, s.nc(), NH, s.hd, s.ds);
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut b = ctx
            .stream
            .launch_builder(m3k.m3_dqkv_state_terms_tiled.get(dtype));
        let ptrs = [
            out.cached_ptr(),
            inp.q.cached_ptr(),
            inp.da_cs.cached_ptr(),
            inp.d_o.cached_ptr(),
        ];
        for p in &ptrs {
            b.arg(p);
        }
        let dims = s.dims();
        for d in &dims {
            b.arg(d);
        }
        b.arg(&y_first);
        b.arg(&z_first);
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    out
}

/// Every chunk's entering state from the per-chunk terms.
fn enter_state(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    s: Shape,
    terms: &GpuBuffer,
    inp: &Inputs,
) -> GpuBuffer {
    let out = GpuBuffer::zeros(&ctx.stream, B * s.nc() * NH * s.hd * s.ds).unwrap();
    let mut b = ctx.stream.launch_builder(&m3k.m3_dstate_passing_bwd.i32);
    let ptrs = [
        out.cached_ptr(),
        terms.cached_ptr(),
        inp.da_cs_sum.cached_ptr(),
    ];
    for p in &ptrs {
        b.arg(p);
    }
    let dims = [B, NH, s.hd, s.ds, s.nc()].map(|v| v as i32);
    for d in &dims {
        b.arg(d);
    }
    // No gradient of the window's entering state is asked for, and the
    // rows hold one sequence each.
    let no_seed_grad: u64 = 0;
    b.arg(&no_seed_grad);
    let (t, cs) = (s.t as i32, CS as i32);
    b.arg(&t);
    b.arg(&cs);
    b.arg(&0u64);
    unsafe { b.launch(blocks(B * NH * s.hd * s.ds)) }.unwrap();
    out
}

fn dqkv_warp(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    (s, pairs): (Shape, bool),
    inp: &Inputs,
    enter: &GpuBuffer,
) -> Outputs {
    let out = Outputs::new(ctx, s);
    let legacy = chunked_backward_smem_bytes(s.hd, s.ds);
    let smem = if pairs {
        legacy + (CS * (CS - 1) * 3 / 2 + 2 * CS) * 4
    } else {
        legacy
    };
    assert!(smem <= PLANNED_SMEM, "{s:?} pairs {pairs}");
    let t_split = 16.min(1024 / s.hd).max(1) as u32;
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, B as u32, s.nc() as u32),
        block_dim: (s.hd as u32, t_split, 1),
        shared_mem_bytes: smem as u32,
    };
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut b = ctx.stream.launch_builder(m3k.m3_dqkv_typed.i32.get(dtype));
        let outs = out.ptrs();
        let mut ptrs = outs.to_vec();
        ptrs.extend([
            inp.q.cached_ptr(),
            inp.k.cached_ptr(),
            inp.v.cached_ptr(),
            inp.da_cs.cached_ptr(),
            inp.da_cs_sum.cached_ptr(),
            inp.qk.cached_ptr(),
            inp.states.cached_ptr(),
            inp.d_o.cached_ptr(),
            inp.d_param.cached_ptr(),
            enter.cached_ptr(),
        ]);
        for p in &ptrs {
            b.arg(p);
        }
        let dims = s.dims();
        for d in &dims {
            b.arg(d);
        }
        let pairs_i = i32::from(pairs);
        b.arg(&pairs_i);
        b.arg(&y_first);
        b.arg(&z_first);
        // No packing: the rows hold one sequence each.
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    out
}

/// The dADT chain of every chunk's saved state, as the stage takes it.
fn dm_chain(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    s: Shape,
    inp: &Inputs,
    enter: &GpuBuffer,
) -> GpuBuffer {
    let rows = B * s.nc() * NH;
    let chain = GpuBuffer::zeros(&ctx.stream, rows).unwrap();
    let mut c = ctx.stream.launch_builder(&m3k.m3_dqkv_dm_chain);
    let ptrs = [
        chain.cached_ptr(),
        inp.states.cached_ptr(),
        enter.cached_ptr(),
    ];
    for p in &ptrs {
        c.arg(p);
    }
    let cdims = [rows, s.hd, s.ds].map(|v| v as i32);
    for d in &cdims {
        c.arg(d);
    }
    unsafe { c.launch(dm_chain_cfg(rows)) }.unwrap();
    chain
}

/// One layer's stage through `stage`: the register-tiled kernel (`tile`
/// none) or the frozen staged one at its tile width.
fn dqkv_stage(
    ctx: &GpuCtx,
    stage: &CudaFunction,
    (s, tile): (Shape, Option<usize>),
    inp: &Inputs,
    (enter, chain): (&GpuBuffer, &GpuBuffer),
) -> Outputs {
    let out = Outputs::new(ctx, s);
    let smem = tile.map_or_else(chunked_backward_tiled_smem_bytes, staged_smem_bytes);
    stage
        .set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem as i32,
        )
        .unwrap();
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, B as u32, s.nc() as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let tile_i = tile.map(|t| t as i32);
    for (y_first, z_first, cfg) in yz_runs(cfg) {
        let mut b = ctx.stream.launch_builder(stage);
        let outs = out.ptrs();
        let mut ptrs = outs.to_vec();
        ptrs.extend([
            inp.q.cached_ptr(),
            inp.k.cached_ptr(),
            inp.v.cached_ptr(),
            inp.da_cs.cached_ptr(),
            inp.da_cs_sum.cached_ptr(),
            inp.qk.cached_ptr(),
            inp.states.cached_ptr(),
            inp.d_o.cached_ptr(),
            inp.d_param.cached_ptr(),
            enter.cached_ptr(),
            chain.cached_ptr(),
        ]);
        for p in &ptrs {
            b.arg(p);
        }
        let dims = s.dims();
        for d in &dims {
            b.arg(d);
        }
        if let Some(tile) = &tile_i {
            b.arg(tile);
        }
        b.arg(&y_first);
        b.arg(&z_first);
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    out
}

/// One layer's register-tiled stage.
fn dqkv_tiled(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    s: Shape,
    inp: &Inputs,
    enter: &GpuBuffer,
) -> Outputs {
    let chain = dm_chain(ctx, m3k, s, inp, enter);
    dqkv_stage(
        ctx,
        m3k.m3_dqkv_tiled.get(dtype),
        (s, None),
        inp,
        (enter, &chain),
    )
}

fn setup() -> (GpuCtx, Mamba3Kernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(ctx.stream.context(), arch::arch0()).unwrap();
    (ctx, m3k)
}

/// Every warp-kernel shape: the tiled stage gives the warp stage's bits.
fn tiled_matches_warp(dtype: WeightDtype) {
    let (ctx, m3k) = setup();
    let mut cases = 0;
    for hd in [1usize, 4, 8, 16, 32] {
        for ds in [1usize, 16, 17, 32, 48] {
            for t in [7usize, 64, 100] {
                let s = Shape { hd, ds, t };
                let inp = Inputs::new(&ctx, s, dtype);
                let terms_ref = state_terms_reference(&ctx, &m3k, dtype, s, &inp);
                let terms = state_terms_tiled(&ctx, &m3k, dtype, s, &inp);
                let label = format!("{dtype:?} hd {hd} ds {ds} T {t}");
                assert_bits(
                    &format!("{label} state terms"),
                    &download(&ctx, &terms_ref),
                    &download(&ctx, &terms),
                );
                let enter = enter_state(&ctx, &m3k, s, &terms_ref, &inp);
                let tiled = dqkv_tiled(&ctx, &m3k, dtype, s, &inp, &enter).read(&ctx);
                for pairs in [true, false] {
                    let warp = dqkv_warp(&ctx, &m3k, dtype, (s, pairs), &inp, &enter).read(&ctx);
                    for ((name, want), (_, got)) in warp.iter().zip(&tiled) {
                        assert_bits(&format!("{label} pairs {pairs} {name}"), want, got);
                    }
                }
                cases += 1;
            }
        }
    }
    eprintln!("{dtype:?}: the tiled stage carries the warp stage's bits on {cases} shapes");
}

#[test]
fn m3_dqkv_tiled_matches_the_warp_kernel_f32() {
    tiled_matches_warp(WeightDtype::F32);
}

#[test]
fn m3_dqkv_tiled_matches_the_warp_kernel_bf16() {
    tiled_matches_warp(WeightDtype::Bf16);
}

#[test]
fn m3_dqkv_tiled_matches_the_warp_kernel_f16() {
    tiled_matches_warp(WeightDtype::F16);
}

/// Widths only the tiled kernels serve: the register-tiled stage gives the
/// staged stage's bits at every tile width that one took, and the state
/// terms match the original kernel where it runs.
#[test]
fn the_register_tiled_stage_keeps_the_staged_stage_s_bits() {
    let (ctx, m3k) = setup();
    let frozen = frozen_module(
        ctx.stream.context(),
        arch::arch0(),
        m3k.state_cap,
        FROZEN_TILED,
    );
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let suffix = match dtype {
            WeightDtype::Bf16 => "_bf16",
            WeightDtype::F16 => "_f16",
            _ => "",
        };
        let staged = frozen
            .load_function(&format!("m3_reference_dqkv_tiled{suffix}"))
            .unwrap();
        for (hd, ds, t) in [
            (48usize, 16usize, 100usize),
            (64, 64, 100),
            (64, 64, 128),
            (100, 17, 100),
            (33, 5, 7),
            (256, 3, 100),
            (128, 128, 65),
            (16, 256, 100),
        ] {
            let s = Shape { hd, ds, t };
            let label = format!("{dtype:?} hd {hd} ds {ds} T {t}");
            let inp = Inputs::new(&ctx, s, dtype);
            let terms = state_terms_tiled(&ctx, &m3k, dtype, s, &inp);
            let original_runs = s.hd * 4 <= 1024 && (CS * s.ds + CS * s.hd + CS) * 4 <= 48 * 1024;
            if original_runs {
                let terms_ref = state_terms_reference(&ctx, &m3k, dtype, s, &inp);
                assert_bits(
                    &format!("{label} state terms"),
                    &download(&ctx, &terms_ref),
                    &download(&ctx, &terms),
                );
            }
            let enter = enter_state(&ctx, &m3k, s, &terms, &inp);
            let chain = dm_chain(&ctx, &m3k, s, &inp, &enter);
            let tiled = dqkv_stage(
                &ctx,
                m3k.m3_dqkv_tiled.get(dtype),
                (s, None),
                &inp,
                (&enter, &chain),
            )
            .read(&ctx);
            for tile in [16usize, 1, 5, 32] {
                let want =
                    dqkv_stage(&ctx, &staged, (s, Some(tile)), &inp, (&enter, &chain)).read(&ctx);
                for ((name, want), (_, got)) in want.iter().zip(&tiled) {
                    assert_bits(&format!("{label} staged tile {tile} {name}"), want, got);
                }
            }
            for (name, values) in &tiled {
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{label} {name}: a value is not finite"
                );
            }
        }
    }
}

/// The dADT chain of a chunk's saved state, a warp a row, is one thread's
/// chain of fused multiply-adds over the row in order from +0: the same
/// chain on the CPU gives every bit, at rows whose length is and is not a
/// whole number of quads and of the warp's segments, and over more rows
/// than a block holds.
#[test]
fn the_dm_chain_is_each_row_s_ordered_chain() {
    let (ctx, m3k) = setup();
    for (rows, hd, ds) in [
        (9usize, 5usize, 7usize),
        (17, 64, 128),
        (3, 128, 256),
        (8, 16, 16),
        (1, 1, 1),
    ] {
        let n = rows * hd * ds;
        let s_vals = det(n, 71, 1.0);
        let e_vals = det(n, 72, 1.0);
        let (s_buf, e_buf) = (upload(&ctx, &s_vals), upload(&ctx, &e_vals));
        let out = GpuBuffer::zeros(&ctx.stream, rows).unwrap();
        let mut c = ctx.stream.launch_builder(&m3k.m3_dqkv_dm_chain);
        let ptrs = [out.cached_ptr(), s_buf.cached_ptr(), e_buf.cached_ptr()];
        for p in &ptrs {
            c.arg(p);
        }
        let dims = [rows, hd, ds].map(|v| v as i32);
        for d in &dims {
            c.arg(d);
        }
        unsafe { c.launch(dm_chain_cfg(rows)) }.unwrap();
        let got = download(&ctx, &out);
        let want: Vec<f32> = (0..rows)
            .map(|r| {
                let row = r * hd * ds..(r + 1) * hd * ds;
                s_vals[row.clone()]
                    .iter()
                    .zip(&e_vals[row])
                    .fold(0.0f32, |acc, (s, e)| s.mul_add(*e, acc))
            })
            .collect();
        assert_bits(&format!("rows {rows} hd {hd} ds {ds}"), &want, &got);
    }
}
