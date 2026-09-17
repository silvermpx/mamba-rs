//! Actual SM120 NVRTC exact-F32 N64 copy-plan force/ABI/numerical-contract gates.
//! AUTO admission and performance evidence live in the separate SM120 paired harness.
//! Finite numeric corpus is deliberately bounded: all-output CPU ascending FMA
//! plus independent f64 (exact int64 dyadic dot for Signed) truth. This is not
//! a general cuBLAS performance gate or a blanket exception to vendor tolerance.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaGraph, DeviceRepr, LaunchConfig, PushKernelArg, sys};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceTile, inference_forward,
    inference_forward_f32_legacy_baseline, inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

const SYMBOL: &str = "nn_sm120_f32_n64_sliced";
const LEGACY_SYMBOL: &str = "f32_f32_s2";
const CANDIDATE: InferenceTile = InferenceTile::F32Sm120N64Sliced;
const GUARD: usize = 64;
const POISON: u32 = 0xa5a5_a5a5;
const SPECIAL: [u32; 16] = [
    0,
    0x8000_0000,
    1,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc1_2345,
    0xffc5_4321,
    0x7f80_0001,
    0xff80_0001,
    0x3f80_0000,
    0xbf80_0000,
];

// Deliberately independent of the future production host type. The captured
// Driver ABI must agree with this literal layout, including m/n/k order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawParams {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}
unsafe impl DeviceRepr for RawParams {}

impl RawParams {
    fn words(self) -> [u32; 8] {
        [
            self.alpha.to_bits(),
            self.beta.to_bits(),
            self.m as u32,
            self.n as u32,
            self.k as u32,
            self.lda as u32,
            self.ldb as u32,
            self.ldc as u32,
        ]
    }
}

fn raw_params_match(actual: [u32; 8], expected: RawParams) -> bool {
    actual == expected.words()
}

#[test]
fn captured_exact_n64_parameters_reject_signed_zero_drift() {
    let expected = Fixture::contiguous(65, 96, 136, false, Corpus::Signed).params();
    assert!(raw_params_match(expected.words(), expected));
    assert!(!raw_params_match(
        RawParams {
            beta: -0.0,
            ..expected
        }
        .words(),
        expected
    ));
    let positive_zero_alpha = RawParams {
        alpha: 0.0,
        ..expected
    };
    assert!(!raw_params_match(
        RawParams {
            alpha: -0.0,
            ..positive_zero_alpha
        }
        .words(),
        positive_zero_alpha
    ));
    for index in 0..8 {
        let mut wrong = expected.words();
        wrong[index] ^= 1;
        assert!(
            !raw_params_match(wrong, expected),
            "raw parameter word {index}"
        );
    }
}
const _: () = {
    assert!(std::mem::size_of::<RawParams>() == 32);
    assert!(std::mem::align_of::<RawParams>() == 4);
    assert!(std::mem::offset_of!(RawParams, alpha) == 0);
    assert!(std::mem::offset_of!(RawParams, beta) == 4);
    assert!(std::mem::offset_of!(RawParams, m) == 8);
    assert!(std::mem::offset_of!(RawParams, n) == 12);
    assert!(std::mem::offset_of!(RawParams, k) == 16);
    assert!(std::mem::offset_of!(RawParams, lda) == 20);
    assert!(std::mem::offset_of!(RawParams, ldb) == 24);
    assert!(std::mem::offset_of!(RawParams, ldc) == 28);
};

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Signed,
    Positive,
    Order,
    BiasOrder,
    SpecialA,
    SpecialB,
    SpecialBias,
    Mixed,
}

impl Corpus {
    fn finite(self) -> bool {
        matches!(
            self,
            Self::Signed | Self::Positive | Self::Order | Self::BiasOrder
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Fixture {
    m: usize,
    k: usize,
    n: usize,
    storage_rows: usize,
    row: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a_shift: usize,
    b_shift: usize,
    c_shift: usize,
    bias: bool,
    alpha: f32,
    beta: f32,
    corpus: Corpus,
}

impl Fixture {
    fn contiguous(m: usize, k: usize, n: usize, bias: bool, corpus: Corpus) -> Self {
        Self {
            m,
            k,
            n,
            storage_rows: m,
            row: 0,
            lda: k,
            ldb: n,
            ldc: n,
            a_shift: 0,
            b_shift: 0,
            c_shift: 0,
            bias,
            alpha: 1.0,
            beta: 0.0,
            corpus,
        }
    }
    fn params(self) -> RawParams {
        RawParams {
            alpha: self.alpha,
            beta: self.beta,
            m: self.m as i32,
            n: self.n as i32,
            k: self.k as i32,
            lda: self.lda as i32,
            ldb: self.ldb as i32,
            ldc: self.ldc as i32,
        }
    }
    fn public(self) -> bool {
        self.lda == self.k
            && self.ldb == self.n
            && self.ldc == self.n
            && self.alpha.to_bits() == 1.0f32.to_bits()
            && self.beta.to_bits() == 0
    }
    fn full(self) -> Self {
        Self {
            m: self.storage_rows,
            row: 0,
            ..self
        }
    }
    fn validate(self) {
        assert!(self.m > 0 && self.n > 0 && self.row + self.m <= self.storage_rows);
        assert!(self.lda >= self.k && self.ldb >= self.n && self.ldc >= self.n);
        assert!(
            self.storage_rows <= 256 && self.k <= 192 && self.n <= 136,
            "this independent numeric oracle admits only its bounded corpus"
        );
        assert!(self.a_shift <= 3 && self.b_shift <= 3 && self.c_shift <= 3);
    }
}

fn seed(index: usize, salt: u64, corpus: Corpus) -> u32 {
    let mut x = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    if matches!(corpus, Corpus::Positive) {
        0x3f00_0000 | (x as u32 & 0x007f_ffff) | 1
    } else {
        ((x % 4093) as f32 / 1024.0 - 2046.0 / 1024.0).to_bits()
    }
}

struct Guarded {
    device: GpuByteBuffer,
    initial: Vec<u32>,
    start: usize,
}

impl Guarded {
    fn new(ctx: &GpuCtx, elements: usize, shift: usize) -> Self {
        let initial = vec![POISON; GUARD + shift + elements + GUARD];
        let device =
            GpuByteBuffer::zeros(&ctx.stream, initial.len() * 4).expect("guarded allocation");
        Self {
            device,
            initial,
            start: GUARD + shift,
        }
    }
    fn ptr(&self) -> u64 {
        self.device.cached_ptr() + (self.start * 4) as u64
    }
    fn reset(&mut self, ctx: &GpuCtx) {
        self.device
            .upload_bytes(&ctx.stream, bytemuck::cast_slice(&self.initial))
            .expect("guarded upload");
        ctx.stream
            .synchronize()
            .expect("finish upload before host storage can move");
    }
    fn read(&self, ctx: &GpuCtx) -> Vec<u32> {
        let mut bytes = vec![0; self.initial.len() * 4];
        ctx.stream
            .memcpy_dtoh(self.device.inner(), &mut bytes)
            .expect("raw read");
        ctx.stream
            .synchronize()
            .expect("finish asynchronous raw read");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect()
    }
    fn unchanged(&self, ctx: &GpuCtx, name: &str) {
        assert_words(&self.read(ctx), &self.initial, name);
    }
}

struct Inputs {
    a: Guarded,
    b: Guarded,
    bias: Guarded,
}

impl Inputs {
    fn new(ctx: &GpuCtx, f: Fixture) -> Self {
        let mut a = Guarded::new(ctx, f.storage_rows * f.lda, f.a_shift);
        let mut b = Guarded::new(ctx, f.k * f.ldb, f.b_shift);
        let mut bias = Guarded::new(ctx, f.n, 1);
        for r in 0..f.storage_rows {
            for k in 0..f.k {
                a.initial[a.start + r * f.lda + k] = match f.corpus {
                    Corpus::Order => [16777216.0f32, 1.0, -16777216.0][k].to_bits(),
                    Corpus::BiasOrder => [16777216.0f32, -16777216.0][k].to_bits(),
                    Corpus::SpecialA => SPECIAL[r % SPECIAL.len()],
                    Corpus::Mixed => SPECIAL[(r + k) % SPECIAL.len()],
                    Corpus::SpecialB | Corpus::SpecialBias => 1.0f32.to_bits(),
                    _ => seed(r * f.k + k, 0x0ada_a001, f.corpus),
                };
            }
        }
        for k in 0..f.k {
            for n in 0..f.n {
                b.initial[b.start + k * f.ldb + n] = match f.corpus {
                    Corpus::SpecialB => SPECIAL[n % SPECIAL.len()],
                    Corpus::Mixed => SPECIAL[(n + k + 3) % SPECIAL.len()],
                    Corpus::Signed | Corpus::Positive => seed(k * f.n + n, 0x0ada_b001, f.corpus),
                    _ => 1.0f32.to_bits(),
                };
            }
        }
        for n in 0..f.n {
            bias.initial[bias.start + n] = match f.corpus {
                Corpus::SpecialBias | Corpus::Mixed => SPECIAL[n % SPECIAL.len()],
                Corpus::Signed | Corpus::Positive => seed(n, 0x0ada_b1a5, f.corpus),
                _ => 1.0f32.to_bits(),
            };
        }
        a.reset(ctx);
        b.reset(ctx);
        bias.reset(ctx);
        Self { a, b, bias }
    }
    fn operands(&self, output: &Guarded, f: Fixture) -> InferenceFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        InferenceFwdOperands {
            c: typed(output.ptr() + (f.row * f.ldc * 4) as u64),
            x: typed(if f.k == 0 {
                0
            } else {
                self.a.ptr() + (f.row * f.lda * 4) as u64
            }),
            w: typed(if f.k == 0 { 0 } else { self.b.ptr() }),
            bias_ptr: f.bias.then_some(self.bias.ptr()),
        }
    }
    fn unchanged(&self, ctx: &GpuCtx) {
        self.a.unchanged(ctx, "A including prefix/suffix/padding");
        self.b.unchanged(ctx, "B including prefix/suffix/padding");
        self.bias.unchanged(ctx, "bias including prefix/suffix");
    }
}

fn output(ctx: &GpuCtx, f: Fixture) -> Guarded {
    let mut result = Guarded::new(ctx, f.storage_rows * f.ldc, f.c_shift);
    if f.beta != 0.0 {
        for r in 0..f.storage_rows {
            for n in 0..f.n {
                result.initial[result.start + r * f.ldc + n] =
                    ((r % 7) as f32 * 0.25 - (n % 5) as f32 * 0.125 + 0.5).to_bits();
            }
        }
    }
    result.reset(ctx);
    result
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    OldOracle,
    Legacy,
    N128,
    Candidate,
}

fn launch(
    ctx: &GpuCtx,
    inputs: &Inputs,
    out: &Guarded,
    f: Fixture,
    arm: Arm,
) -> Result<(), String> {
    let operands = inputs.operands(out, f);
    let shape = InferenceShape {
        m: f.m,
        k: f.k,
        n: f.n,
    };
    if f.public() {
        return match arm {
            Arm::OldOracle => inference_forward_f32_legacy_baseline(ctx, operands, shape),
            Arm::Legacy => inference_forward_with_tile(ctx, operands, shape, InferenceTile::Legacy),
            Arm::N128 => {
                inference_forward_with_tile(ctx, operands, shape, InferenceTile::F32N128S2)
            }
            Arm::Candidate => inference_forward_with_tile(ctx, operands, shape, CANDIDATE),
        };
    }
    // Private raw controls exercise existing arithmetic contracts (padded
    // strides and alpha/beta), not a new public production interface.
    let function = match arm {
        Arm::OldOracle => &ctx.kernels.gemm_bi_f32_f32,
        Arm::Legacy => &ctx.kernels.gemm_bi_f32_f32_s2,
        Arm::N128 => &ctx.kernels.gemm_bi_f32_f32_n128_s2,
        Arm::Candidate => ctx
            .kernels
            .fixed_sm120_f32_n64_sliced
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "missing exact N64 holder: {:?}",
                    ctx.kernels.fixed_sm120_f32_n64_sliced_rejection
                )
            })?,
    };
    let params = f.params();
    let bias = operands.bias_ptr.unwrap_or(0);
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&operands.c.ptr)
        .arg(&operands.x.ptr)
        .arg(&operands.w.ptr)
        .arg(&bias);
    if matches!(arm, Arm::Candidate) {
        builder.arg(&params);
    } else {
        builder
            .arg(&params.alpha)
            .arg(&params.beta)
            .arg(&params.m)
            .arg(&params.n)
            .arg(&params.k)
            .arg(&params.lda)
            .arg(&params.ldb)
            .arg(&params.ldc);
    }
    let (bn, threads) = match arm {
        Arm::N128 => (128, 256),
        Arm::OldOracle => (64, 256),
        _ => (64, 128),
    };
    let config = LaunchConfig {
        grid_dim: ((f.m.div_ceil(64) * f.n.div_ceil(bn)) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|error| format!("{arm:?} raw launch: {error:?}"))
}

fn assert_words(actual: &[u32], expected: &[u32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    if let Some((i, (got, want))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (got, want))| got != want)
    {
        panic!("{label} raw mismatch index={i} got=0x{got:08x} expected=0x{want:08x}");
    }
}

fn logical(out: &Guarded, all: &[u32], f: Fixture) -> Vec<u32> {
    let mut result = Vec::with_capacity(f.m * f.n);
    for (i, &word) in all.iter().enumerate() {
        let offset = i.checked_sub(out.start);
        let live = offset.is_some_and(|offset| {
            offset / f.ldc >= f.row && offset / f.ldc < f.row + f.m && offset % f.ldc < f.n
        });
        if live {
            result.push(word);
        } else {
            assert_eq!(
                word, out.initial[i],
                "output guard/padding/outside-row-view index={i} {f:?}"
            );
        }
    }
    assert_eq!(result.len(), f.m * f.n);
    result
}

fn numeric_close(actual: f64, exact: f64) -> bool {
    actual.is_finite()
        && exact.is_finite()
        && (actual - exact).abs() <= 0.0002 * (1.0 + exact.abs())
}

fn signed_integer(value: f32) -> Option<i64> {
    let scaled = value as f64 * 1024.0;
    (scaled.is_finite() && scaled.abs() <= 2046.0 && scaled == scaled.trunc())
        .then_some(scaled as i64)
}

fn numeric_gate(inputs: &Inputs, out: &Guarded, f: Fixture, words: &[u32]) {
    if !f.corpus.finite() {
        return;
    }
    assert_eq!(words.len(), f.m * f.n);
    for r in 0..f.m {
        for n in 0..f.n {
            let row = f.row + r;
            let mut ascending = 0.0f32;
            let mut exact = 0.0f64;
            let mut numerator = 0i64;
            for k in 0..f.k {
                let a = f32::from_bits(inputs.a.initial[inputs.a.start + row * f.lda + k]);
                let b = f32::from_bits(inputs.b.initial[inputs.b.start + k * f.ldb + n]);
                ascending = a.mul_add(b, ascending);
                exact += a as f64 * b as f64;
                if matches!(f.corpus, Corpus::Signed) {
                    numerator += signed_integer(a).expect("signed A corpus bound")
                        * signed_integer(b).expect("signed B corpus bound");
                }
            }
            for _ in f.k..f.k.div_ceil(32) * 32 {
                ascending = 0.0f32.mul_add(0.0, ascending);
            }
            if matches!(f.corpus, Corpus::Signed) {
                exact = numerator as f64 / 1048576.0;
            }
            ascending *= f.alpha;
            exact *= f.alpha as f64;
            if f.bias {
                let bias = f32::from_bits(inputs.bias.initial[inputs.bias.start + n]);
                ascending += bias;
                exact += bias as f64;
            }
            if f.beta != 0.0 {
                let initial = f32::from_bits(out.initial[out.start + row * f.ldc + n]);
                ascending = f.beta.mul_add(initial, ascending);
                exact += f.beta as f64 * initial as f64;
            }
            let got = words[r * f.n + n];
            assert_eq!(
                got,
                ascending.to_bits(),
                "independent ascending-FMA raw oracle r={r} n={n} {f:?}"
            );
            // Order/BiasOrder intentionally distinguish rounded serial arithmetic
            // from the real-valued dot; check their separate hand oracle instead.
            if matches!(f.corpus, Corpus::Signed | Corpus::Positive) {
                assert!(
                    numeric_close(f32::from_bits(got) as f64, exact),
                    "bounded independent numeric oracle r={r} n={n} got={} exact={exact} {f:?}",
                    f32::from_bits(got)
                );
            }
        }
    }
    if matches!(f.corpus, Corpus::Order) {
        assert!(words.iter().all(|&w| w == 0));
    }
    if matches!(f.corpus, Corpus::BiasOrder) {
        assert!(words.iter().all(|&w| w == 1.0f32.to_bits()));
    }
}

fn graph_parameter<T: Copy>(params: &sys::CUDA_KERNEL_NODE_PARAMS_v2, index: usize) -> T {
    assert!(
        !params.kernelParams.is_null() && params.extra.is_null(),
        "expected direct captured argument storage"
    );
    let pointer = unsafe { *params.kernelParams.add(index) };
    assert!(!pointer.is_null(), "captured argument {index} is null");
    unsafe { pointer.cast::<T>().read_unaligned() }
}

fn assert_candidate_graph(graph: &CudaGraph, operands: InferenceFwdOperands, f: Fixture) {
    assert_exact_n64_graph(graph, operands, f, CANDIDATE);
}

fn assert_exact_n64_graph(
    graph: &CudaGraph,
    operands: InferenceFwdOperands,
    f: Fixture,
    tile: InferenceTile,
) {
    let (symbol, compact, tma, tile_m, tile_n, threads, dynamic_shared) = match tile {
        CANDIDATE => (SYMBOL, true, false, 64, 64, 128, 0),
        InferenceTile::F32Sm120N64CopyPlan => {
            ("nn_sm120_f32_n64_copyplan", true, false, 64, 64, 128, 0)
        }
        InferenceTile::F32Sm120TmaFmaM128N64 => (
            "nn_sm120_tma_fma_m128n64_bk16_s2",
            false,
            true,
            128,
            64,
            128,
            24_592,
        ),
        InferenceTile::F32N128S2 => ("f32_f32_n128_s2", false, false, 64, 128, 256, 0),
        InferenceTile::Legacy => (LEGACY_SYMBOL, false, false, 64, 64, 128, 0),
        _ => panic!("unqualified exact N64 force graph tile: {tile:?}"),
    };
    let mut count = 0;
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        count, 1,
        "exact N64 must capture one kernel and no workspace/initialization nodes"
    );
    let mut node = std::ptr::null_mut();
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
    assert_eq!(
        unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(kind, sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL);
    let mut params = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut name = std::ptr::null();
    assert_eq!(
        unsafe { sys::cuFuncGetName(&mut name, params.func) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert!(!name.is_null());
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes(),
        symbol.as_bytes()
    );
    assert_eq!(
        (params.gridDimX, params.gridDimY, params.gridDimZ),
        ((f.m.div_ceil(tile_m) * f.n.div_ceil(tile_n)) as u32, 1, 1)
    );
    assert_eq!(
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        (threads, 1, 1)
    );
    assert_eq!(params.sharedMemBytes, dynamic_shared);
    let abi: Vec<(usize, usize)> = if tma {
        vec![
            (0, 8),
            (8, 8),
            (16, 8),
            (128, 128),
            (256, 128),
            (384, 8),
            (392, 32),
        ]
    } else if compact {
        vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
    } else {
        (0..4)
            .map(|i| (i * 8, 8))
            .chain((0..8).map(|i| (32 + i * 4, 4)))
            .collect()
    };
    for (i, expected) in abi.iter().copied().enumerate() {
        let (mut offset, mut size) = (usize::MAX, usize::MAX);
        assert_eq!(
            unsafe { sys::cuFuncGetParamInfo(params.func, i, &mut offset, &mut size) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!((offset, size), expected, "captured live ABI parameter {i}");
    }
    let (mut offset, mut size) = (0, 0);
    assert_eq!(
        unsafe { sys::cuFuncGetParamInfo(params.func, abi.len(), &mut offset, &mut size) },
        sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        "unexpected terminal parameter would permit an incorrect ABI or scratch"
    );
    if tma {
        assert_eq!(graph_parameter::<u64>(&params, 0), operands.c.ptr);
        assert_eq!(
            graph_parameter::<u64>(&params, 1),
            0,
            "single-split exact TMA must not bind scratch"
        );
        assert_eq!(
            graph_parameter::<u64>(&params, 2),
            0,
            "single-split exact TMA must not bind flags"
        );
        assert!(
            graph_parameter::<[u8; 128]>(&params, 3)
                .iter()
                .any(|byte| *byte != 0),
            "captured A tensor map is zero"
        );
        assert!(
            graph_parameter::<[u8; 128]>(&params, 4)
                .iter()
                .any(|byte| *byte != 0),
            "captured B tensor map is zero"
        );
        assert_eq!(
            graph_parameter::<u64>(&params, 5),
            operands.bias_ptr.unwrap_or(0)
        );
        assert_eq!(
            graph_parameter::<[u32; 8]>(&params, 6),
            [
                f.alpha.to_bits(),
                f.beta.to_bits(),
                f.m as u32,
                f.n as u32,
                f.k as u32,
                f.ldc as u32,
                1,
                f.k.div_ceil(16) as u32,
            ]
        );
        return;
    }
    for (i, expected) in [
        operands.c.ptr,
        operands.x.ptr,
        operands.w.ptr,
        operands.bias_ptr.unwrap_or(0),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            graph_parameter::<u64>(&params, i),
            expected,
            "captured pointer {i}"
        );
    }
    let actual = if compact {
        graph_parameter::<[u32; 8]>(&params, 4)
    } else {
        [
            graph_parameter(&params, 4),
            graph_parameter(&params, 5),
            graph_parameter(&params, 6),
            graph_parameter(&params, 7),
            graph_parameter(&params, 8),
            graph_parameter(&params, 9),
            graph_parameter(&params, 10),
            graph_parameter(&params, 11),
        ]
    };
    assert!(
        raw_params_match(actual, f.params()),
        "captured alpha/beta/m/n/k/strides"
    );
}

fn admit_context() -> GpuCtx {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (12, 0), "exact SM120 gate");
    assert_eq!(device.multiprocessor_count(), 170, "exact SM120 SM count");
    let ctx = GpuCtx::new(&device).expect("actual NVRTC Fixed context");
    let compiler = ctx.kernels.compiler_identity();
    assert_eq!(compiler.nvrtc_version, (13, 2), "qualified NVRTC version");
    assert!(compiler.nvrtc_library_known, "known NVRTC library required");
    assert_eq!(
        compiler.target.as_str(),
        "compute_120",
        "qualified production compiler target"
    );
    let function = ctx
        .kernels
        .fixed_sm120_f32_n64_sliced
        .as_ref()
        .unwrap_or_else(|| {
            panic!(
                "production optional holder declined: {:?}",
                ctx.kernels.fixed_sm120_f32_n64_sliced_rejection
            )
        });
    assert!(ctx.kernels.fixed_sm120_f32_n64_sliced_rejection.is_none());
    let local = function.local_size_bytes().unwrap();
    let registers = function.num_regs().unwrap();
    let shared = function.shared_size_bytes().unwrap();
    let threads = function.max_threads_per_block().unwrap();
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
        .unwrap();
    let carveout = function
        .get_attribute(
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        )
        .unwrap();
    assert_eq!(local, 0);
    assert!((1..=160).contains(&registers));
    assert_eq!(shared, 32768);
    assert!(threads >= 128);
    assert!(occupancy >= 3);
    assert_eq!(carveout, 100);
    let version = ctx.kernels.compiler_identity().nvrtc_version;
    println!(
        concat!(
            "{{\"record\":\"FixedSm120SlicedN64LiveAdmissionV1\",\"symbol\":\"{}\",",
            "\"cc\":[12,0],\"sm\":{},\"registers\":{},\"static_shared\":{},",
            "\"dynamic_shared\":0,\"local\":{},\"max_threads\":{},",
            "\"active_ctas\":{},\"carveout\":{},\"nvrtc\":[{},{}],\"nvrtc_library_known\":true,\"compiler_target\":\"compute_120\"}}"
        ),
        SYMBOL,
        device.multiprocessor_count(),
        registers,
        shared,
        local,
        threads,
        occupancy,
        carveout,
        version.0,
        version.1
    );
    ctx
}

fn run_fixture(ctx: &GpuCtx, f: Fixture) {
    f.validate();
    let inputs = Inputs::new(ctx, f);
    let full = f.full();
    let reference = output(ctx, full);
    launch(ctx, &inputs, &reference, full, Arm::OldOracle).expect("full old exact reference");
    let full_words = logical(&reference, &reference.read(ctx), full);
    numeric_gate(&inputs, &reference, full, &full_words);
    // Prove the complete full-batch reference numerically once. Every eager
    // and replay word below must then equal its exact slice, including views.
    let expected = &full_words[f.row * f.n..(f.row + f.m) * f.n];
    for arm in [Arm::OldOracle, Arm::Legacy, Arm::N128, Arm::Candidate] {
        if matches!(arm, Arm::Candidate) {
            println!(
                concat!(
                    "{{\"record\":\"FixedSm120SlicedN64ForcePreflightV1\",",
                    "\"dims_mkn\":[{},{},{}],\"three_incumbents_eager_graph_raw\":true,",
                    "\"independent_cpu_reference\":{},\"public_force\":{}}}"
                ),
                f.m,
                f.k,
                f.n,
                f.corpus.finite(),
                f.public()
            );
        }
        let mut out = output(ctx, f);
        for _ in 0..2 {
            out.reset(ctx);
            launch(ctx, &inputs, &out, f, arm)
                .unwrap_or_else(|error| panic!("{arm:?} eager launch {f:?}: {error}"));
            let got = logical(&out, &out.read(ctx), f);
            assert_words(
                &got,
                expected,
                &format!("{arm:?} eager/full-reference row view {f:?}"),
            );
        }
        let graph =
            unsafe { capture_into_graph(&ctx.stream, || launch(ctx, &inputs, &out, f, arm)) }
                .unwrap_or_else(|error| panic!("{arm:?} actual capture {f:?}: {error}"));
        if matches!(arm, Arm::Candidate) {
            assert_candidate_graph(&graph, inputs.operands(&out, f), f);
        }
        for _ in 0..2 {
            out.reset(ctx);
            graph.launch().expect("captured replay");
            let got = logical(&out, &out.read(ctx), f);
            assert_words(
                &got,
                expected,
                &format!("{arm:?} poisoned graph/full-reference row view {f:?}"),
            );
        }
        // Keep captured storage alive through inspection/replays, then prove
        // eager behavior survives graph execution with beta1 freshly reset.
        drop(graph);
        out.reset(ctx);
        launch(ctx, &inputs, &out, f, arm).expect("post-graph eager launch");
        assert_words(
            &logical(&out, &out.read(ctx), f),
            expected,
            "post-graph exact bits",
        );
        inputs.unchanged(ctx);
    }
    println!(
        concat!(
            "{{\"record\":\"FixedSm120SlicedN64CorrectnessV1\",\"dtype\":\"f32\",\"op\":\"nn\",",
            "\"requested\":\"{:?}\",\"actual\":\"{}\",\"dims_mkn\":[{},{},{}],",
            "\"strides_lda_ldb_ldc\":[{},{},{}],\"storage_rows\":{},\"row_view\":{},",
            "\"float_offsets_abc\":[{},{},{}],\"bias\":{},\"alpha_bits\":{},\"beta_bits\":{},",
            "\"corpus\":\"{:?}\",\"public_force\":{},\"full_raw_bits\":true,",
            "\"cpu_numeric\":{},\"eager_repeats\":3,\"captured_replays\":2,",
            "\"guards\":true,\"input_immutability\":true}}"
        ),
        CANDIDATE,
        SYMBOL,
        f.m,
        f.k,
        f.n,
        f.lda,
        f.ldb,
        f.ldc,
        f.storage_rows,
        f.row,
        f.a_shift,
        f.b_shift,
        f.c_shift,
        f.bias,
        f.alpha.to_bits(),
        f.beta.to_bits(),
        f.corpus,
        f.public(),
        f.corpus.finite()
    );
}

#[test]
#[ignore = "requires actual SM120 NVRTC exact N64 holder and implemented forced launcher"]
fn fixed_sm120_sliced_forced_bits_graph_views() {
    let ctx = admit_context();
    // First test reaches a real public forced launch only after all incumbent
    // controls and independent full-output numeric checks have passed.
    run_fixture(&ctx, Fixture::contiguous(65, 1, 65, false, Corpus::Signed));
    for corpus in [Corpus::Signed, Corpus::Positive] {
        for bias in [false, true] {
            for (k, n) in [
                (0, 65),
                (31, 65),
                (32, 136),
                (33, 131),
                (65, 131),
                (96, 136),
                (192, 136),
            ] {
                let base = Fixture::contiguous(129, k, n, bias, corpus);
                run_fixture(&ctx, base);
                for (m, row, a_shift, b_shift, c_shift) in [
                    (1, 0, 0, 0, 0),
                    (32, 0, 0, 0, 0),
                    (64, 0, 0, 0, 0),
                    (17, 17, 1, 0, 0),
                    (65, 17, 0, 1, 0),
                    (129, 17, 0, 0, 1),
                    (129, 17, 3, 3, 3),
                ] {
                    run_fixture(
                        &ctx,
                        Fixture {
                            m,
                            row,
                            storage_rows: 146,
                            a_shift,
                            b_shift,
                            c_shift,
                            ..base
                        },
                    );
                }
            }
        }
    }
    run_fixture(&ctx, Fixture::contiguous(65, 3, 65, false, Corpus::Order));
    run_fixture(
        &ctx,
        Fixture::contiguous(65, 2, 65, true, Corpus::BiasOrder),
    );
    for corpus in [
        Corpus::SpecialA,
        Corpus::SpecialB,
        Corpus::SpecialBias,
        Corpus::Mixed,
    ] {
        for k in [0, 1, 32, 33, 96] {
            for bias in [false, true] {
                let n = if matches!(k, 32 | 96) { 136 } else { 65 };
                run_fixture(&ctx, Fixture::contiguous(17, k, n, bias, corpus));
            }
        }
    }
    // Padded full-K path and scalar fallback have the same raw alpha/beta
    // arithmetic; beta1 always starts from nonzero, freshly reset C.
    for k in [0, 32, 65, 96] {
        for bias in [false, true] {
            let base = Fixture::contiguous(65, k, 65, bias, Corpus::Signed);
            run_fixture(
                &ctx,
                Fixture {
                    storage_rows: 82,
                    row: 17,
                    lda: k + 4,
                    ldb: 68,
                    ldc: 68,
                    alpha: 0.5,
                    beta: 1.0,
                    ..base
                },
            );
            run_fixture(
                &ctx,
                Fixture {
                    storage_rows: 82,
                    row: 17,
                    lda: k + 1,
                    ldb: 67,
                    ldc: 67,
                    a_shift: 1,
                    b_shift: 2,
                    c_shift: 3,
                    alpha: 0.5,
                    beta: 1.0,
                    ..base
                },
            );
        }
    }
}

#[test]
#[ignore = "bounded actual SM120 NVRTC exact N64 subset for four compute-sanitizer tools"]
fn fixed_sm120_sliced_sanitizer_smoke() {
    let ctx = admit_context();
    for bias in [false, true] {
        // Logical M/N tails with aligned physical B/C strides: this must use
        // the planned full-K ring, not the generic odd-stride path.
        let base = Fixture::contiguous(65, 96, 65, bias, Corpus::Signed);
        run_fixture(
            &ctx,
            Fixture {
                ldb: 68,
                ldc: 68,
                ..base
            },
        );
        run_fixture(
            &ctx,
            Fixture::contiguous(3, 0, 5, bias, Corpus::SpecialBias),
        );
    }
    for (a_shift, b_shift, c_shift, bias) in [(1, 0, 0, false), (0, 2, 0, true), (0, 0, 3, true)] {
        let base = Fixture::contiguous(17, 65, 131, bias, Corpus::Signed);
        run_fixture(
            &ctx,
            Fixture {
                storage_rows: 34,
                row: 17,
                a_shift,
                b_shift,
                c_shift,
                ..base
            },
        );
    }
    run_fixture(&ctx, Fixture::contiguous(17, 96, 136, true, Corpus::Mixed));
}

#[test]
#[ignore = "requires actual SM120 exact N64 force; rejects unsafe operands before enqueue"]
fn fixed_sm120_sliced_rejects_unsafe_inputs_and_empty_is_noop() {
    let ctx = admit_context();
    let f = Fixture::contiguous(17, 65, 131, false, Corpus::Signed);
    let inputs = Inputs::new(&ctx, f);
    let mut out = output(&ctx, f);
    let good = inputs.operands(&out, f);
    let shape = InferenceShape {
        m: f.m,
        k: f.k,
        n: f.n,
    };
    inference_forward_with_tile(&ctx, good, shape, CANDIDATE).expect("positive guard control");
    out.reset(&ctx);
    let f32ptr = |ptr| TypedPtr {
        ptr,
        dtype: WeightDtype::F32,
    };
    for bad in [
        InferenceFwdOperands {
            x: f32ptr(0),
            ..good
        },
        InferenceFwdOperands {
            w: f32ptr(0),
            ..good
        },
        InferenceFwdOperands {
            c: f32ptr(0),
            ..good
        },
        InferenceFwdOperands {
            x: f32ptr(good.x.ptr + 1),
            ..good
        },
        InferenceFwdOperands {
            w: f32ptr(good.w.ptr + 2),
            ..good
        },
        InferenceFwdOperands {
            c: f32ptr(good.c.ptr + 3),
            ..good
        },
        InferenceFwdOperands {
            bias_ptr: Some(1),
            ..good
        },
        InferenceFwdOperands {
            x: TypedPtr {
                dtype: WeightDtype::Bf16,
                ..good.x
            },
            ..good
        },
        InferenceFwdOperands {
            w: TypedPtr {
                dtype: WeightDtype::F16,
                ..good.w
            },
            ..good
        },
        InferenceFwdOperands {
            c: TypedPtr {
                dtype: WeightDtype::Bf16,
                ..good.c
            },
            ..good
        },
        InferenceFwdOperands {
            x: f32ptr(u64::MAX - 3),
            ..good
        },
        InferenceFwdOperands {
            w: f32ptr(u64::MAX - 3),
            ..good
        },
        InferenceFwdOperands {
            c: f32ptr(u64::MAX - 3),
            ..good
        },
        InferenceFwdOperands {
            bias_ptr: Some(u64::MAX - 3),
            ..good
        },
    ] {
        assert!(
            inference_forward_with_tile(&ctx, bad, shape, CANDIDATE).is_err(),
            "unsafe force operands were admitted"
        );
        out.unchanged(&ctx, "rejected operands enqueued output work");
        inputs.unchanged(&ctx);
    }
    for bad in [
        InferenceShape {
            m: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            k: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            n: i32::MAX as usize,
            ..shape
        },
        InferenceShape {
            m: i32::MAX as usize + 1,
            ..shape
        },
        InferenceShape {
            m: 1 << 20,
            n: 1 << 29,
            ..shape
        },
    ] {
        assert!(
            inference_forward_with_tile(&ctx, good, bad, CANDIDATE).is_err(),
            "unsafe dimensions were admitted {bad:?}"
        );
        out.unchanged(&ctx, "rejected dimensions enqueued output work");
    }
    let null = InferenceFwdOperands {
        c: f32ptr(0),
        x: f32ptr(0),
        w: f32ptr(0),
        bias_ptr: None,
    };
    for empty in [
        InferenceShape { m: 0, ..shape },
        InferenceShape { n: 0, ..shape },
    ] {
        inference_forward_with_tile(&ctx, null, empty, CANDIDATE)
            .expect("empty output must not launch");
    }
}

#[test]
fn bounded_numeric_oracle_rejects_wrong_results_and_unrelated_corpora() {
    assert!(numeric_close(6.0, 6.0));
    assert!(!numeric_close(0.0, 6.0));
    assert!(!numeric_close(f64::NAN, 6.0));
    assert!(!numeric_close(6.0, f64::INFINITY));
    assert_eq!(signed_integer(-2046.0 / 1024.0), Some(-2046));
    for bad in [0.1, f32::INFINITY, f32::from_bits(0x7fc1_2345), 3.0] {
        assert!(signed_integer(bad).is_none());
    }
    // A signed cancellation point from the independent standalone finding;
    // this bounded GPU suite does not admit a tolerance exception for it.
    let mut ascending = 0.0f32;
    let mut numerator = 0i64;
    for k in 0..1928 {
        let a = f32::from_bits(seed(549 * 1928 + k, 0x0ada_a001, Corpus::Signed));
        let b = f32::from_bits(seed(k * 384 + 319, 0x0ada_b001, Corpus::Signed));
        ascending = a.mul_add(b, ascending);
        numerator += signed_integer(a).unwrap() * signed_integer(b).unwrap();
    }
    assert_eq!(ascending.to_bits(), 0xbd2b_af00);
    assert_eq!(numerator, -43710);
    assert!(!numeric_close(ascending as f64, -43710.0 / 1048576.0));
}

fn launch_auto(
    ctx: &GpuCtx,
    inputs: &Inputs,
    out: &Guarded,
    f: Fixture,
) -> Result<InferenceTile, String> {
    assert!(
        f.public(),
        "AUTO exercises only public contiguous alpha1/beta+0"
    );
    let operands = inputs.operands(out, f);
    inference_forward(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (f.m, f.k, f.n),
    )
}

fn run_auto_view(
    ctx: &GpuCtx,
    inputs: &Inputs,
    f: Fixture,
    full_legacy: &[u32],
    expected_tile: InferenceTile,
) {
    assert_eq!(full_legacy.len(), f.storage_rows * f.n);
    let expected = &full_legacy[f.row * f.n..(f.row + f.m) * f.n];
    let mut out = output(ctx, f);
    let actual_tile = launch_auto(ctx, inputs, &out, f).expect("actual AUTO eager launch");
    assert_words(
        &logical(&out, &out.read(ctx), f),
        expected,
        "AUTO eager versus full old Legacy raw row slice",
    );
    out.reset(ctx);
    assert_eq!(launch_auto(ctx, inputs, &out, f).unwrap(), actual_tile);
    assert_words(
        &logical(&out, &out.read(ctx), f),
        expected,
        "AUTO reset eager versus full old Legacy raw row slice",
    );
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            let captured_tile = launch_auto(ctx, inputs, &out, f)?;
            if captured_tile != actual_tile {
                return Err(format!(
                    "AUTO eager/capture route drift: {actual_tile:?} -> {captured_tile:?}"
                ));
            }
            Ok(())
        })
    }
    .expect("actual AUTO capture");
    // Decode what actually launched before testing the expected promotion.
    // During RED this proves valid Legacy geometry/ABI and raw output, then
    // fails only at the final intended candidate-versus-Legacy assertion.
    assert_exact_n64_graph(&graph, inputs.operands(&out, f), f, actual_tile);
    for _ in 0..2 {
        out.reset(ctx);
        graph.launch().expect("AUTO poisoned graph replay");
        assert_words(
            &logical(&out, &out.read(ctx), f),
            expected,
            "AUTO poisoned graph versus full old Legacy raw row slice",
        );
    }
    drop(graph);
    inputs.unchanged(ctx);
    assert_eq!(ctx.f32_triad_policy(), F32TriadPolicy::ExactScalarFma);
    let actual_symbol = match actual_tile {
        CANDIDATE => SYMBOL,
        InferenceTile::Legacy => LEGACY_SYMBOL,
        InferenceTile::F32Sm120N64CopyPlan => "nn_sm120_f32_n64_copyplan",
        InferenceTile::F32Sm120TmaFmaM128N64 => "nn_sm120_tma_fma_m128n64_bk16_s2",
        InferenceTile::F32N128S2 => "f32_f32_n128_s2",
        _ => unreachable!("graph contract already rejects other AUTO symbols"),
    };
    println!(
        concat!(
            "{{\"record\":\"FixedSm120SlicedAutoBoundaryV1\",\"dtype\":\"f32\",\"op\":\"nn\",",
            "\"requested\":\"AUTO\",\"expected\":\"{:?}\",\"actual\":\"{:?}\",\"symbol\":\"{}\",",
            "\"dims_mkn\":[{},{},{}],\"storage_rows\":{},\"row_view\":{},\"float_offsets_abc\":[{},{},{}],",
            "\"bias\":{},\"alpha_bits\":1065353216,\"beta_bits\":0,\"full_old_legacy_reference\":true,",
            "\"raw_full_output_and_row_slice\":true,\"eager_repeats\":2,\"poisoned_graph_replays\":2,",
            "\"actual_graph_symbol_geometry_abi_words\":true,\"guards\":true,\"input_immutability\":true}}"
        ),
        expected_tile,
        actual_tile,
        actual_symbol,
        f.m,
        f.k,
        f.n,
        f.storage_rows,
        f.row,
        f.a_shift,
        f.b_shift,
        f.c_shift,
        f.bias
    );
    assert_eq!(
        actual_tile, expected_tile,
        "measured AUTO row or its exact M/pointer boundary {f:?}"
    );
}

#[test]
#[ignore = "actual SM120 CUDA13.2 exact-TMA/copyplan AUTO, full-hot Legacy prefix/view raw-bit boundaries"]
fn fixed_sm120_sliced_auto_prefix_view_graph_bits() {
    let ctx = admit_context();
    assert_eq!(
        GpuDevice::new(0)
            .expect("AUTO device identity")
            .multiprocessor_count(),
        170
    );
    assert_eq!(ctx.kernels.compiler_identity().nvrtc_version, (13, 2));
    assert!(ctx.kernels.compiler_identity().nvrtc_library_known);
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);
    // Sliced is force-only: full-hot raw batch/view gate uses exact-TMA at B0
    // and retains copyplan for E0/E1/B1.
    // This is not a replacement for independent numerical or paired timing gates.
    for (hot_m, k, n) in [(2048, 2304, 768), (4621, 768, 2304)] {
        for bias in [false, true] {
            for (a_shift, b_shift, c_shift) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                let full = Fixture {
                    storage_rows: hot_m + 18,
                    a_shift,
                    b_shift,
                    c_shift,
                    ..Fixture::contiguous(hot_m + 18, k, n, bias, Corpus::Signed)
                };
                let inputs = Inputs::new(&ctx, full);
                let mut reference = output(&ctx, full);
                launch(&ctx, &inputs, &reference, full, Arm::Legacy)
                    .expect("full old Legacy reference, never AUTO or candidate");
                let full_legacy = logical(&reference, &reference.read(&ctx), full);
                reference.reset(&ctx);
                launch(&ctx, &inputs, &reference, full, Arm::Legacy)
                    .expect("full old Legacy reference repeat");
                assert_words(
                    &logical(&reference, &reference.read(&ctx), full),
                    &full_legacy,
                    "full old Legacy reference repeat raw bits",
                );
                inputs.unchanged(&ctx);
                // Real row17 A/C subviews, not a repacked prefix. K and N are
                // multiples of four, so a whole-row subview preserves A16/C16.
                for row in [0, 17] {
                    let hot_expected = if (hot_m, k, n, bias) == (4621, 768, 2304, false) {
                        InferenceTile::F32Sm120TmaFmaM128N64
                    } else {
                        InferenceTile::F32Sm120N64CopyPlan
                    };
                    for (m, aligned_expected) in [
                        (hot_m, hot_expected),
                        (hot_m - 1, InferenceTile::Legacy),
                        (hot_m + 1, InferenceTile::Legacy),
                        (1, InferenceTile::Legacy),
                        (17, InferenceTile::Legacy),
                        (63, InferenceTile::Legacy),
                        (64, InferenceTile::Legacy),
                        (65, InferenceTile::Legacy),
                        (127, InferenceTile::Legacy),
                        (128, InferenceTile::Legacy),
                        (129, InferenceTile::Legacy),
                    ] {
                        let expected = if (a_shift, b_shift, c_shift) == (0, 0, 0) {
                            aligned_expected
                        } else if (m, k, n) == (4621, 768, 2304) {
                            InferenceTile::F32N128S2
                        } else {
                            InferenceTile::Legacy
                        };
                        run_auto_view(
                            &ctx,
                            &inputs,
                            Fixture { m, row, ..full },
                            &full_legacy,
                            expected,
                        );
                    }
                }
            }
        }
    }
}
