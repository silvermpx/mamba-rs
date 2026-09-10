#![cfg(feature = "cuda")]

use std::collections::{BTreeSet, VecDeque};
use std::ffi::CStr;
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, sys,
};
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceTile, inference_forward, inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    NUMERIC_ABI_REVISION, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest as _, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    N96,
    N128,
}

impl Width {
    const fn label(self) -> &'static str {
        match self {
            Self::N96 => "w4_n96",
            Self::N128 => "w4_n128",
        }
    }

    const fn symbol(self) -> &'static str {
        match self {
            Self::N96 => "gemm_bi_nn_fixed_rna_tf32_exp_m128n96_bk16_s5_w4",
            Self::N128 => "gemm_bi_nn_fixed_rna_tf32_exp_m128n128_bk16_s5_w4",
        }
    }

    const fn shared_bytes(self) -> usize {
        5 * 16 * (128 + self.columns()) * 4
    }
}

impl Width {
    const fn columns(self) -> usize {
        match self {
            Self::N96 => 96,
            Self::N128 => 128,
        }
    }
}

fn w4_a_index(row: usize, k: usize) -> usize {
    let chunk = (k >> 2) ^ (row & 3);
    row * 16 + chunk * 4 + (k & 3)
}

fn w4_b_index(width: Width, k: usize, column: usize) -> usize {
    let chunk = (column >> 2) ^ ((k & 3) << 1);
    k * width.columns() + chunk * 4 + (column & 3)
}

fn w4_a_copy(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 128;
    (linear / 4, (linear % 4) * 4)
}

fn w4_b_copy(width: Width, thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 128;
    let chunks = width.columns() / 4;
    (linear / chunks, (linear % chunks) * 4)
}

fn w4_thread_outputs(width: Width, warp: usize, lane: usize) -> Vec<(usize, usize)> {
    let columns = width.columns();
    let warp_m = (warp >> 1) * 64;
    let warp_n = (warp & 1) * (columns / 2);
    let group = lane >> 2;
    let thread = lane & 3;
    let mut outputs = Vec::with_capacity(columns);
    for m_atom in 0..4 {
        for n_atom in 0..columns / 16 {
            for element in 0..4 {
                outputs.push((
                    warp_m + m_atom * 16 + group + usize::from(element >= 2) * 8,
                    warp_n + n_atom * 8 + 2 * thread + (element & 1),
                ));
            }
        }
    }
    outputs
}

fn w4_ring_trace(tile_count: usize) -> Vec<(usize, usize, Option<(usize, usize)>)> {
    (0..tile_count)
        .map(|tile| {
            let next = tile + 4;
            (
                tile,
                tile % 5,
                (next < tile_count).then_some((next, next % 5)),
            )
        })
        .collect()
}

fn w4_schedule_is_safe(tile_count: usize, write_slot_offset: usize) -> bool {
    let mut slots = [None; 5];
    let mut groups = VecDeque::new();
    let mut complete = BTreeSet::new();
    for tile in 0..4.min(tile_count) {
        slots[tile % 5] = Some(tile);
        groups.push_back(Some(tile));
    }
    while groups.len() < 4 {
        groups.push_back(None);
    }
    for tile in 0..tile_count {
        while groups.len() > 3 {
            if let Some(Some(completed_tile)) = groups.pop_front() {
                complete.insert(completed_tile);
            }
        }
        if slots[tile % 5] != Some(tile) || !complete.contains(&tile) {
            return false;
        }
        let next = tile + 4;
        if next < tile_count {
            let slot = (next % 5 + write_slot_offset) % 5;
            slots[slot] = Some(next);
            groups.push_back(Some(next));
        } else {
            groups.push_back(None);
        }
    }
    true
}

fn w4_needs_input_pointers(k: usize) -> bool {
    k != 0
}

fn w4_bk16_slab_count(k: usize) -> usize {
    k.div_ceil(32) * 2
}

#[test]
fn bk16_a_mask3_layout_and_four_copy_slices_are_bijective() {
    let mut layout = BTreeSet::new();
    for row in 0..128 {
        for k in 0..16 {
            let index = w4_a_index(row, k);
            assert!(index < 128 * 16);
            assert!(layout.insert(index), "A layout alias at row={row} k={k}");
        }
    }
    assert_eq!(layout.len(), 128 * 16);
    assert_eq!(w4_a_index(0, 0), 0);
    assert_eq!(w4_a_index(3, 0), 60);
    assert_eq!(w4_a_index(4, 0), 64);
    assert_eq!(w4_a_index(127, 15), 2_035);

    let mut copies = BTreeSet::new();
    for slice in 0..4 {
        for thread in 0..128 {
            let (row, k) = w4_a_copy(thread, slice);
            assert!(row < 128);
            assert!(k < 16);
            assert_eq!(k % 4, 0);
            assert!(copies.insert((row, k)), "duplicate A copy {:?}", (row, k));
        }
    }
    assert_eq!(copies.len(), 128 * 4);
    assert_eq!(copies.first(), Some(&(0, 0)));
    assert_eq!(copies.last(), Some(&(127, 12)));
}

#[test]
fn asymmetric_b_layout_and_copy_slices_cover_n96_and_n128() {
    for (width, slices) in [(Width::N96, 3), (Width::N128, 4)] {
        let columns = width.columns();
        let mut layout = BTreeSet::new();
        for k in 0..16 {
            for column in 0..columns {
                let index = w4_b_index(width, k, column);
                assert!(index < 16 * columns);
                assert!(
                    layout.insert(index),
                    "B layout alias for {width:?} at k={k} column={column}"
                );
            }
        }
        assert_eq!(layout.len(), 16 * columns);

        let mut copies = BTreeSet::new();
        for slice in 0..slices {
            for thread in 0..128 {
                let (row, column) = w4_b_copy(width, thread, slice);
                assert!(row < 16);
                assert!(column < columns);
                assert_eq!(column % 4, 0);
                assert!(copies.insert((row, column)));
            }
        }
        assert_eq!(copies.len(), 16 * columns / 4);
        assert_eq!(copies.first(), Some(&(0, 0)));
        assert_eq!(copies.last(), Some(&(15, columns - 4)));
    }
    assert_eq!(w4_b_copy(Width::N96, 127, 0), (5, 28));
    assert_eq!(w4_b_copy(Width::N96, 0, 1), (5, 32));
    assert_eq!(w4_b_copy(Width::N128, 127, 3), (15, 124));
}

#[test]
fn four_warps_own_every_output_once_for_both_widths() {
    for width in [Width::N96, Width::N128] {
        let columns = width.columns();
        let expected_per_thread = columns;
        let mut outputs = BTreeSet::new();
        for warp in 0..4 {
            for lane in 0..32 {
                let owned = w4_thread_outputs(width, warp, lane);
                assert_eq!(owned.len(), expected_per_thread);
                for (row, column) in owned {
                    assert!(row < 128);
                    assert!(column < columns);
                    assert!(outputs.insert((row, column)));
                }
            }
        }
        assert_eq!(outputs.len(), 128 * columns);
        assert_eq!(outputs.first(), Some(&(0, 0)));
        assert_eq!(outputs.last(), Some(&(127, columns - 1)));
    }
}

#[test]
fn five_stage_ring_reads_each_tile_before_reusing_its_slot() {
    assert_eq!(
        w4_ring_trace(8),
        vec![
            (0, 0, Some((4, 4))),
            (1, 1, Some((5, 0))),
            (2, 2, Some((6, 1))),
            (3, 3, Some((7, 2))),
            (4, 4, None),
            (5, 0, None),
            (6, 1, None),
            (7, 2, None),
        ]
    );
    for tile_count in (1..20).chain([144]) {
        let trace = w4_ring_trace(tile_count);
        assert_eq!(trace.len(), tile_count);
        for (tile, read_slot, next) in trace {
            assert_eq!(read_slot, tile % 5);
            if let Some((next_tile, write_slot)) = next {
                assert_eq!(next_tile, tile + 4);
                assert_eq!(write_slot, next_tile % 5);
                assert_ne!(read_slot, write_slot);
            }
        }
    }
    assert!(w4_schedule_is_safe(17, 0));
    assert!(!w4_schedule_is_safe(17, 1));
}

#[test]
fn zero_reduction_does_not_require_operand_pointers() {
    assert!(!w4_needs_input_pointers(0));
    assert!(w4_needs_input_pointers(1));
    assert!(w4_needs_input_pointers(16));
}

#[test]
fn bk16_slab_count_preserves_production_k32_padding() {
    assert_eq!(w4_bk16_slab_count(0), 0);
    assert_eq!(w4_bk16_slab_count(1), 2);
    assert_eq!(w4_bk16_slab_count(16), 2);
    assert_eq!(w4_bk16_slab_count(32), 2);
    assert_eq!(w4_bk16_slab_count(33), 4);
    assert_eq!(w4_bk16_slab_count(36), 4);
    assert_eq!(w4_bk16_slab_count(64), 4);
}

const W4_CUDA: &str = include_str!("gemm_bi_fixed_tf32_w4_discovery.cu");
const N96_CUDA: &str = include_str!("gemm_bi_fixed_tf32_n96_discovery.cu");
const N96_SYMBOL: &str = "gemm_bi_nn_fixed_rna_tf32_exp_m128n96_bk32_s3";
const FAST_SYMBOL: &str =
    "_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4EEvNT_6ParamsE";
const GUARD: usize = 32;
const GUARD_BITS: u32 = 0x7fc0_4a11;
const POISON_BITS: u32 = 0x7fc0_4a12;

#[derive(Clone, Copy)]
#[repr(C)]
struct W4Params {
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for W4Params {}

const _: () = {
    assert!(std::mem::size_of::<W4Params>() == 32);
    assert!(std::mem::align_of::<W4Params>() == 4);
};

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    shape: InferenceShape,
    bias: bool,
}

const E0: Case = Case {
    label: "e0",
    shape: InferenceShape {
        m: 2048,
        k: 2304,
        n: 768,
    },
    bias: false,
};

const CASES: [Case; 5] = [
    E0,
    Case {
        label: "e1",
        bias: true,
        ..E0
    },
    Case {
        label: "tail0",
        shape: InferenceShape {
            m: 129,
            k: 36,
            n: 100,
        },
        bias: false,
    },
    Case {
        label: "tail1",
        shape: InferenceShape {
            m: 129,
            k: 36,
            n: 100,
        },
        bias: true,
    },
    Case {
        label: "k16_special",
        shape: InferenceShape {
            m: 129,
            k: 16,
            n: 100,
        },
        bias: true,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Candidate(Width),
    CommittedN96,
    ProductionRna,
    FastTf32,
}

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Candidate(width) => width.label(),
            Self::CommittedN96 => "committed_n96",
            Self::ProductionRna => "production_rna",
            Self::FastTf32 => "fast_tf32",
        }
    }
}

struct Runtime {
    _device: GpuDevice,
    ctx: GpuCtx,
    _module: Arc<CudaModule>,
    width: Width,
    candidate: CudaFunction,
    committed_n96: CudaFunction,
    source_sha: String,
    ptx_sha: String,
}

impl Runtime {
    fn candidate(&self, width: Width) -> &CudaFunction {
        assert_eq!(self.width, width);
        &self.candidate
    }
}

struct GuardedF32 {
    buffer: GpuBuffer,
    baseline: Vec<f32>,
    offset: usize,
    len: usize,
}

impl GuardedF32 {
    fn new(stream: &Arc<CudaStream>, active: Vec<f32>) -> Result<Self, String> {
        let len = active.len();
        let mut baseline = vec![f32::from_bits(GUARD_BITS); GUARD + len + GUARD];
        baseline[GUARD..GUARD + len].copy_from_slice(&active);
        Ok(Self {
            buffer: GpuBuffer::from_cpu(stream, &baseline)?,
            baseline,
            offset: GUARD,
            len,
        })
    }

    fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buffer.raw_ptr_at(stream, self.offset)
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        self.buffer.upload(stream, &self.baseline)
    }

    fn bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
        let values = self.buffer.to_cpu(stream)?;
        for (index, value) in values[..self.offset]
            .iter()
            .chain(&values[self.offset + self.len..])
            .enumerate()
        {
            if value.to_bits() != GUARD_BITS {
                return Err(format!("{label} changed guard {index}"));
            }
        }
        Ok(values[self.offset..self.offset + self.len]
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
        let values = self.buffer.to_cpu(stream)?;
        if values
            .iter()
            .zip(&self.baseline)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(format!("{label} input/guard changed"));
        }
        Ok(())
    }
}

struct Fixture {
    a: GuardedF32,
    b: GuardedF32,
    bias: GuardedF32,
    candidate: GuardedF32,
    production: GuardedF32,
    committed: GuardedF32,
    fast: GuardedF32,
}

impl Fixture {
    fn new(runtime: &Runtime, case: Case) -> Result<Self, String> {
        let output = vec![f32::from_bits(POISON_BITS); case.shape.m * case.shape.n];
        let values = |len, seed| {
            if case.label == "k16_special" {
                exceptional_values(len, seed)
            } else {
                finite_full_mantissa_values(len, seed)
            }
        };
        Ok(Self {
            a: GuardedF32::new(
                &runtime.ctx.stream,
                values(case.shape.m * case.shape.k, 0xa040_0001),
            )?,
            b: GuardedF32::new(
                &runtime.ctx.stream,
                values(case.shape.k * case.shape.n, 0xb040_0002),
            )?,
            bias: GuardedF32::new(&runtime.ctx.stream, values(case.shape.n, 0xb1a5_0003))?,
            candidate: GuardedF32::new(&runtime.ctx.stream, output.clone())?,
            production: GuardedF32::new(&runtime.ctx.stream, output.clone())?,
            committed: GuardedF32::new(&runtime.ctx.stream, output.clone())?,
            fast: GuardedF32::new(&runtime.ctx.stream, output)?,
        })
    }

    fn output(&self, arm: Arm) -> &GuardedF32 {
        match arm {
            Arm::Candidate(_) => &self.candidate,
            Arm::ProductionRna => &self.production,
            Arm::CommittedN96 => &self.committed,
            Arm::FastTf32 => &self.fast,
        }
    }

    fn output_mut(&mut self, arm: Arm) -> &mut GuardedF32 {
        match arm {
            Arm::Candidate(_) => &mut self.candidate,
            Arm::ProductionRna => &mut self.production,
            Arm::CommittedN96 => &mut self.committed,
            Arm::FastTf32 => &mut self.fast,
        }
    }

    fn operands(&self, runtime: &Runtime, case: Case, arm: Arm) -> InferenceFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        InferenceFwdOperands {
            c: typed(self.output(arm).ptr(&runtime.ctx.stream)),
            x: typed(self.a.ptr(&runtime.ctx.stream)),
            w: typed(self.b.ptr(&runtime.ctx.stream)),
            bias_ptr: case.bias.then(|| self.bias.ptr(&runtime.ctx.stream)),
        }
    }

    fn reset(&mut self, runtime: &Runtime, arm: Arm) -> Result<(), String> {
        self.output_mut(arm).reset(&runtime.ctx.stream)
    }

    fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
        self.a.unchanged(&runtime.ctx.stream, "A")?;
        self.b.unchanged(&runtime.ctx.stream, "B")?;
        self.bias.unchanged(&runtime.ctx.stream, "bias")
    }
}

fn finite_full_mantissa_values(len: usize, mut state: u64) -> Vec<f32> {
    const SPECIAL: [u32; 10] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_1000,
        0xbf80_1000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x3f80_0fff,
        0x3f80_1001,
    ];
    (0..len)
        .map(|index| {
            if index < SPECIAL.len() {
                return f32::from_bits(SPECIAL[index]);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = (state as u32) & 0x8000_0000;
            let exponent = (((state >> 32) as u32 % 48) + 103) << 23;
            let mantissa = (state as u32) & 0x007f_ffff;
            f32::from_bits(sign | exponent | mantissa)
        })
        .collect()
}

fn exceptional_values(len: usize, state: u64) -> Vec<f32> {
    const EXCEPTIONAL: [u32; 4] = [0x7f80_0000, 0xff80_0000, 0x7fc0_0041, 0xffc0_0042];
    let mut values = finite_full_mantissa_values(len, state);
    for (value, bits) in values.iter_mut().zip(EXCEPTIONAL) {
        *value = f32::from_bits(bits);
    }
    values
}

fn compose_source() -> String {
    let prelude = include_str!("../kernels/_typed_prelude.cuh");
    let common = include_str!("../kernels/gemm_bi_inference/common.cuh")
        .lines()
        .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
        .collect::<Vec<_>>()
        .join("\n");
    let tf32 = include_str!("../kernels/gemm_bi_inference/tf32.cu");
    [prelude, &common, tf32, N96_CUDA, W4_CUDA].join("\n")
}

fn new_runtime(width: Width) -> Result<Runtime, String> {
    if std::env::var("MAMBA_FIXED_W4_DISCOVERY").as_deref() != Ok("1") {
        return Err("set MAMBA_FIXED_W4_DISCOVERY=1".into());
    }
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "W4 discovery requires CC8.9/142SM, found {:?}/{}",
            device.compute_capability,
            device.multiprocessor_count()
        ));
    }
    let ctx = GpuCtx::new(&device)?;
    let compiler = ctx.kernels.compiler_identity();
    let artifact = ctx.kernels.artifact_set_identity().fixed;
    if compiler.nvrtc_version != (13, 2)
        || compiler.target.as_str() != "sm_89"
        || !compiler.nvrtc_library_known
        || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || compiler.schedule_revision != SCHEDULE_REVISION
        || TUNING_TABLE_REVISION != 44
        || NUMERIC_ABI_REVISION != 5
        || SCHEDULE_REVISION != 8
        || artifact.compile_key != compiler.invocation_digest
    {
        return Err(format!(
            "W4 lost accepted identity {compiler:?} {artifact:?}"
        ));
    }
    let source = compose_source();
    let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .map_err(|error| format!("compile W4 discovery: {error:?}"))?;
    let ptx_source = ptx.to_src();
    let ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
    let module = device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
        .map_err(|error| format!("load W4 module: {error:?}"))?;
    let load = |symbol| {
        module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))
    };
    let candidate = load(width.symbol())?;
    let committed_n96 = load(N96_SYMBOL)?;
    for (function, shared) in [(&candidate, width.shared_bytes()), (&committed_n96, 86_016)] {
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared as i32,
            )
            .map_err(|error| format!("set dynamic shared {shared}: {error:?}"))?;
    }
    let runtime = Runtime {
        _device: device,
        ctx,
        _module: module,
        width,
        candidate,
        committed_n96,
        source_sha,
        ptx_sha,
    };
    println!(
        concat!(
            "{{\"schema\":\"MambaBiFixedTf32W4IdentityV1\",",
            "\"fixed_source_sha\":\"{}\",\"fixed_invocation_sha\":\"{}\",",
            "\"fixed_artifact_sha\":\"{}\",\"candidate_source_sha\":\"{}\",",
            "\"candidate_ptx_sha\":\"{}\"}}"
        ),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&artifact.artifact_digest),
        runtime.source_sha,
        runtime.ptx_sha,
    );
    Ok(runtime)
}

fn configure(runtime: &Runtime, arm: Arm) -> Result<(), String> {
    runtime.ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    runtime.ctx.set_bi_tensor_cores(false);
    runtime
        .ctx
        .set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    if arm == Arm::FastTf32 {
        runtime.ctx.set_batch_invariant(false);
        runtime.ctx.set_fast_gemm(true);
        if !runtime.ctx.tf32() {
            return Err("FAST_TF32 comparator disabled".into());
        }
    } else {
        runtime.ctx.set_batch_invariant(true);
        runtime.ctx.set_fast_gemm(false);
    }
    Ok(())
}

fn launch_config(shape: InferenceShape, width: Width) -> Result<LaunchConfig, String> {
    let blocks = shape.m.div_ceil(128) * shape.n.div_ceil(width.columns());
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(blocks).map_err(|_| "W4 grid exceeds u32")?,
            1,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: width.shared_bytes() as u32,
    })
}

fn launch_candidate_raw(
    runtime: &Runtime,
    width: Width,
    output: u64,
    a: u64,
    b: u64,
    bias: u64,
    shape: InferenceShape,
    alpha: f32,
    beta: f32,
) -> Result<(), String> {
    let params = W4Params {
        alpha,
        beta,
        m: i32::try_from(shape.m).map_err(|_| "M exceeds i32")?,
        k: i32::try_from(shape.k).map_err(|_| "K exceeds i32")?,
        n: i32::try_from(shape.n).map_err(|_| "N exceeds i32")?,
        lda: i32::try_from(shape.k).map_err(|_| "lda exceeds i32")?,
        ldb: i32::try_from(shape.n).map_err(|_| "ldb exceeds i32")?,
        ldc: i32::try_from(shape.n).map_err(|_| "ldc exceeds i32")?,
    };
    let mut builder = runtime.ctx.stream.launch_builder(runtime.candidate(width));
    builder.arg(&output);
    builder.arg(&a);
    builder.arg(&b);
    builder.arg(&bias);
    builder.arg(&params);
    unsafe { builder.launch(launch_config(shape, width)?) }
        .map(|_| ())
        .map_err(|error| format!("launch {}: {error:?}", width.label()))
}

fn launch(runtime: &Runtime, fixture: &Fixture, case: Case, arm: Arm) -> Result<(), String> {
    configure(runtime, arm)?;
    let operands = fixture.operands(runtime, case, arm);
    match arm {
        Arm::Candidate(width) => launch_candidate_raw(
            runtime,
            width,
            operands.c.ptr,
            operands.x.ptr,
            operands.w.ptr,
            operands.bias_ptr.unwrap_or(0),
            case.shape,
            1.0,
            0.0,
        ),
        Arm::CommittedN96 => {
            let params = W4Params {
                alpha: 1.0,
                beta: 0.0,
                m: case.shape.m as i32,
                k: case.shape.k as i32,
                n: case.shape.n as i32,
                lda: case.shape.k as i32,
                ldb: case.shape.n as i32,
                ldc: case.shape.n as i32,
            };
            let mut builder = runtime.ctx.stream.launch_builder(&runtime.committed_n96);
            builder.arg(&operands.c.ptr);
            builder.arg(&operands.x.ptr);
            builder.arg(&operands.w.ptr);
            let bias = operands.bias_ptr.unwrap_or(0);
            builder.arg(&bias);
            builder.arg(&params);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (
                        (case.shape.m.div_ceil(128) * case.shape.n.div_ceil(96)) as u32,
                        1,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 86_016,
                })
            }
            .map(|_| ())
            .map_err(|error| format!("launch committed N96: {error:?}"))
        }
        Arm::ProductionRna => inference_forward_with_tile(
            &runtime.ctx,
            operands,
            case.shape,
            InferenceTile::Tf32RnaM128N128S3,
        ),
        Arm::FastTf32 => gpu_gemm_typed_forward_raw(
            &runtime.ctx,
            operands.c,
            operands.x,
            operands.w,
            operands.bias_ptr,
            (case.shape.m, case.shape.k, case.shape.n),
        ),
    }
}

fn output_bits(
    runtime: &Runtime,
    fixture: &Fixture,
    arm: Arm,
    label: &str,
) -> Result<Vec<u32>, String> {
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize {label}: {error:?}"))?;
    fixture.output(arm).bits(&runtime.ctx.stream, label)
}

fn capture(
    runtime: &Runtime,
    fixture: &Fixture,
    case: Case,
    arm: Arm,
) -> Result<CudaGraph, String> {
    unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, case, arm)) }
}

fn validate_candidate(runtime: &Runtime, width: Width, graph: &CudaGraph) -> Result<(), String> {
    let function = runtime.candidate(width);
    let registers = function
        .num_regs()
        .map_err(|error| format!("{} registers: {error:?}", width.label()))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("{} local: {error:?}", width.label()))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("{} static shared: {error:?}", width.label()))?;
    let occupancy = function
        .occupancy_max_active_blocks_per_multiprocessor(128, width.shared_bytes(), None)
        .map_err(|error| format!("{} occupancy: {error:?}", width.label()))?;
    if registers <= 0 || registers > 255 || local != 0 || static_shared != 0 || occupancy != 1 {
        return Err(format!(
            "{} resource reject regs={registers} local={local} static={static_shared} occupancy={occupancy}",
            width.label()
        ));
    }
    let mut count = 0usize;
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
        || count != 1
    {
        return Err(format!("{} graph count {count}", width.label()));
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err(format!("{} graph nodes failed", width.label()));
    }
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    if unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err(format!("{} graph params failed", width.label()));
    }
    let mut name = std::ptr::null();
    if unsafe { sys::cuFuncGetName(&mut name, params.func) } != sys::CUresult::CUDA_SUCCESS
        || name.is_null()
    {
        return Err(format!("{} graph symbol failed", width.label()));
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| format!("{} symbol UTF8: {error}", width.label()))?;
    let expected_grid = E0.shape.m.div_ceil(128) * E0.shape.n.div_ceil(width.columns());
    if symbol != width.symbol()
        || (params.gridDimX, params.gridDimY, params.gridDimZ) != (expected_grid as u32, 1, 1)
        || (params.blockDimX, params.blockDimY, params.blockDimZ) != (128, 1, 1)
        || params.sharedMemBytes != width.shared_bytes() as u32
    {
        return Err(format!("{} physical mismatch {symbol}", width.label()));
    }
    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut offset = 0;
        let mut size = 0;
        if unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) }
            != sys::CUresult::CUDA_SUCCESS
            || (offset, size) != expected
        {
            return Err(format!(
                "{} ABI {index}: {:?}",
                width.label(),
                (offset, size)
            ));
        }
    }
    let mut offset = 0;
    let mut size = 0;
    if unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
        != sys::CUresult::CUDA_ERROR_INVALID_VALUE
    {
        return Err(format!("{} ABI accepted parameter 5", width.label()));
    }
    println!(
        "{{\"schema\":\"MambaBiFixedTf32W4ResourceV1\",\"arm\":\"{}\",\"symbol\":\"{}\",\"registers\":{},\"local_bytes\":{},\"shared_bytes\":{},\"occupancy\":{}}}",
        width.label(),
        symbol,
        registers,
        local,
        width.shared_bytes(),
        occupancy
    );
    Ok(())
}

fn run_arm_bits(
    runtime: &Runtime,
    fixture: &mut Fixture,
    case: Case,
    arm: Arm,
) -> Result<Vec<u32>, String> {
    fixture.reset(runtime, arm)?;
    launch(runtime, fixture, case, arm)?;
    let bits = output_bits(runtime, fixture, arm, arm.label())?;
    if bits.iter().any(|word| *word == POISON_BITS) {
        return Err(format!("{} retained poison", arm.label()));
    }
    Ok(bits)
}

fn assert_actual_auto(runtime: &Runtime) -> Result<(), String> {
    let mut fixture = Fixture::new(runtime, E0)?;
    fixture.reset(runtime, Arm::ProductionRna)?;
    configure(runtime, Arm::ProductionRna)?;
    let operands = fixture.operands(runtime, E0, Arm::ProductionRna);
    let selected = inference_forward(
        &runtime.ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (E0.shape.m, E0.shape.k, E0.shape.n),
    )?;
    if selected != InferenceTile::Tf32RnaM128N128S3 {
        return Err(format!("E AUTO selected {selected:?}"));
    }
    output_bits(runtime, &fixture, Arm::ProductionRna, "actual AUTO")?;
    Ok(())
}

fn check_case(runtime: &Runtime, width: Width, case: Case) -> Result<Vec<u32>, String> {
    let mut fixture = Fixture::new(runtime, case)?;
    let production = run_arm_bits(runtime, &mut fixture, case, Arm::ProductionRna)?;
    let candidate = run_arm_bits(runtime, &mut fixture, case, Arm::Candidate(width))?;
    if candidate != production {
        return Err(format!("{} {} differs from RNA", width.label(), case.label));
    }
    for repeat in 0..3 {
        for arm in [Arm::ProductionRna, Arm::Candidate(width)] {
            if run_arm_bits(runtime, &mut fixture, case, arm)? != production {
                return Err(format!(
                    "{} {} eager {repeat} changed",
                    width.label(),
                    arm.label()
                ));
            }
        }
    }
    let production_graph = capture(runtime, &fixture, case, Arm::ProductionRna)?;
    let candidate_graph = capture(runtime, &fixture, case, Arm::Candidate(width))?;
    if case.label == "e0" {
        validate_candidate(runtime, width, &candidate_graph)?;
    }
    for repeat in 0..3 {
        for (arm, graph) in [
            (Arm::ProductionRna, &production_graph),
            (Arm::Candidate(width), &candidate_graph),
        ] {
            fixture.reset(runtime, arm)?;
            graph
                .launch()
                .map_err(|error| format!("{} graph {repeat}: {error:?}", arm.label()))?;
            if output_bits(runtime, &fixture, arm, arm.label())? != production {
                return Err(format!("{} {} graph changed", width.label(), arm.label()));
            }
        }
    }
    fixture.validate_inputs(runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32W4BitsV1\",\"arm\":\"{}\",\"case\":\"{}\",\"shape\":[{},{},{}],\"bias\":{},\"passed\":true}}",
        width.label(),
        case.label,
        case.shape.m,
        case.shape.k,
        case.shape.n,
        case.bias
    );
    Ok(production)
}

fn check_prefix(runtime: &Runtime, width: Width, expected: &[u32]) -> Result<(), String> {
    let prefix = Case {
        label: "e_prefix_m1",
        shape: InferenceShape { m: 1, ..E0.shape },
        bias: false,
    };
    let mut fixture = Fixture::new(runtime, E0)?;
    for arm in [Arm::ProductionRna, Arm::Candidate(width)] {
        fixture.reset(runtime, arm)?;
        launch(runtime, &fixture, prefix, arm)?;
        let bits = output_bits(runtime, &fixture, arm, arm.label())?;
        if bits[..prefix.shape.n] != expected[..prefix.shape.n]
            || bits[prefix.shape.n..]
                .iter()
                .any(|actual| *actual != POISON_BITS)
        {
            return Err(format!("{} prefix mismatch", arm.label()));
        }
    }
    fixture.validate_inputs(runtime)
}

fn check_zero_reduction(runtime: &Runtime, width: Width) -> Result<(), String> {
    let shape = InferenceShape { m: 3, k: 0, n: 12 };
    let bias_values = finite_full_mantissa_values(shape.n, 0xb1a5_0005);
    let mut output = GuardedF32::new(
        &runtime.ctx.stream,
        vec![f32::from_bits(POISON_BITS); shape.m * shape.n],
    )?;
    let bias = GuardedF32::new(&runtime.ctx.stream, bias_values.clone())?;
    launch_candidate_raw(
        runtime,
        width,
        output.ptr(&runtime.ctx.stream),
        0,
        0,
        bias.ptr(&runtime.ctx.stream),
        shape,
        1.0,
        0.0,
    )?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("K0 sync: {error:?}"))?;
    let actual = output.bits(&runtime.ctx.stream, "K0")?;
    let expected: Vec<_> = (0..shape.m)
        .flat_map(|_| bias_values.iter().map(|value| value.to_bits()))
        .collect();
    if actual != expected {
        return Err(format!("{} K0 bias mismatch", width.label()));
    }
    output.reset(&runtime.ctx.stream)?;
    launch_candidate_raw(
        runtime,
        width,
        output.ptr(&runtime.ctx.stream),
        0,
        0,
        0,
        shape,
        1.0,
        0.0,
    )?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("K0 null sync: {error:?}"))?;
    if output
        .bits(&runtime.ctx.stream, "K0 null")?
        .iter()
        .any(|bits| *bits != 0)
    {
        return Err(format!("{} K0 null bias not +0", width.label()));
    }
    bias.unchanged(&runtime.ctx.stream, "K0 bias")
}

fn check_alpha_beta_epilogue(runtime: &Runtime, width: Width) -> Result<(), String> {
    let case = CASES[2];
    let fixture = Fixture::new(runtime, case)?;
    let accumulator_output = GuardedF32::new(
        &runtime.ctx.stream,
        vec![f32::from_bits(POISON_BITS); case.shape.m * case.shape.n],
    )?;
    launch_candidate_raw(
        runtime,
        width,
        accumulator_output.ptr(&runtime.ctx.stream),
        fixture.a.ptr(&runtime.ctx.stream),
        fixture.b.ptr(&runtime.ctx.stream),
        0,
        case.shape,
        1.0,
        0.0,
    )?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{} alpha/beta accumulator sync: {error:?}", width.label()))?;
    let accumulator = accumulator_output.bits(&runtime.ctx.stream, "alpha/beta accumulator")?;
    let old_values = finite_full_mantissa_values(case.shape.m * case.shape.n, 0xc040_0004);
    let blended = GuardedF32::new(&runtime.ctx.stream, old_values.clone())?;
    launch_candidate_raw(
        runtime,
        width,
        blended.ptr(&runtime.ctx.stream),
        fixture.a.ptr(&runtime.ctx.stream),
        fixture.b.ptr(&runtime.ctx.stream),
        0,
        case.shape,
        0.75,
        -0.25,
    )?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{} alpha/beta blend sync: {error:?}", width.label()))?;
    let actual = blended.bits(&runtime.ctx.stream, "alpha/beta blend")?;
    let mut compared = 0usize;
    for ((actual, accumulator), old) in actual.iter().zip(&accumulator).zip(&old_values) {
        let accumulator = f32::from_bits(*accumulator);
        let scaled = 0.75_f32 * accumulator;
        let expected = (-0.25_f32).mul_add(*old, scaled);
        if accumulator.is_finite() && old.is_finite() && expected.is_finite() {
            if *actual != expected.to_bits() {
                return Err(format!("{} alpha/beta finite mismatch", width.label()));
            }
            compared += 1;
        }
    }
    if compared < 100 {
        return Err(format!(
            "{} alpha/beta finite coverage {compared}",
            width.label()
        ));
    }
    fixture.validate_inputs(runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32W4AlphaBetaV1\",\"arm\":\"{}\",\"finite_compared\":{},\"passed\":true}}",
        width.label(),
        compared
    );
    Ok(())
}

fn run_correctness(width: Width) -> Result<(), String> {
    let runtime = new_runtime(width)?;
    assert_actual_auto(&runtime)?;
    let e0 = check_case(&runtime, width, E0)?;
    check_prefix(&runtime, width, &e0)?;
    for case in &CASES[1..] {
        check_case(&runtime, width, *case)?;
    }
    check_zero_reduction(&runtime, width)?;
    check_alpha_beta_epilogue(&runtime, width)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32W4CorrectnessCompleteV1\",\"arm\":\"{}\",\"passed\":true}}",
        width.label()
    );
    Ok(())
}

fn measure(
    runtime: &Runtime,
    fixture: &Fixture,
    arm: Arm,
    iterations: usize,
) -> Result<f64, String> {
    let start = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("{} start: {error:?}", arm.label()))?;
    for _ in 0..iterations {
        launch(runtime, fixture, E0, arm)?;
    }
    let end = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("{} end: {error:?}", arm.label()))?;
    let us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("{} elapsed: {error:?}", arm.label()))?,
    ) * 1000.0
        / iterations as f64;
    if us.is_finite() && us > 0.0 {
        Ok(us)
    } else {
        Err(format!("{} invalid time {us}", arm.label()))
    }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn compare(
    runtime: &Runtime,
    fixture: &Fixture,
    width: Width,
    comparator: Arm,
    windows: usize,
) -> Result<(f64, f64), String> {
    let candidate = Arm::Candidate(width);
    for _ in 0..64 {
        launch(runtime, fixture, E0, candidate)?;
        launch(runtime, fixture, E0, comparator)?;
    }
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("warmup: {error:?}"))?;
    let candidate_pilot = measure(runtime, fixture, candidate, 16)?;
    let comparator_pilot = measure(runtime, fixture, comparator, 16)?;
    let iterations = |pilot: f64| (5000.0 / pilot).round().clamp(1.0, 4096.0) as usize;
    let ci = iterations(candidate_pilot);
    let xi = iterations(comparator_pilot);
    let mut all_ratios = Vec::with_capacity(windows * 2);
    for (order, candidate_first) in [("abba", true), ("baab", false)] {
        let mut ratios = Vec::with_capacity(windows);
        for window in 0..windows {
            let observations = if candidate_first {
                [
                    (candidate, measure(runtime, fixture, candidate, ci)?),
                    (comparator, measure(runtime, fixture, comparator, xi)?),
                    (comparator, measure(runtime, fixture, comparator, xi)?),
                    (candidate, measure(runtime, fixture, candidate, ci)?),
                ]
            } else {
                [
                    (comparator, measure(runtime, fixture, comparator, xi)?),
                    (candidate, measure(runtime, fixture, candidate, ci)?),
                    (candidate, measure(runtime, fixture, candidate, ci)?),
                    (comparator, measure(runtime, fixture, comparator, xi)?),
                ]
            };
            let candidate_us = observations
                .iter()
                .filter(|(arm, _)| *arm == candidate)
                .map(|(_, us)| us)
                .sum::<f64>()
                * 0.5;
            let comparator_us = observations
                .iter()
                .filter(|(arm, _)| *arm == comparator)
                .map(|(_, us)| us)
                .sum::<f64>()
                * 0.5;
            let ratio = candidate_us / comparator_us;
            ratios.push(ratio);
            all_ratios.push(ratio);
            println!(
                "{{\"schema\":\"MambaBiFixedTf32W4RawBracketV1\",\"arm\":\"{}\",\"comparator\":\"{}\",\"order\":\"{}\",\"window\":{},\"observations\":[[\"{}\",{}],[\"{}\",{}],[\"{}\",{}],[\"{}\",{}]],\"ratio\":{}}}",
                width.label(),
                comparator.label(),
                order,
                window,
                observations[0].0.label(),
                observations[0].1,
                observations[1].0.label(),
                observations[1].1,
                observations[2].0.label(),
                observations[2].1,
                observations[3].0.label(),
                observations[3].1,
                ratio
            );
        }
        println!(
            "{{\"schema\":\"MambaBiFixedTf32W4TimingSummaryV1\",\"arm\":\"{}\",\"comparator\":\"{}\",\"order\":\"{}\",\"windows\":{},\"ratio_p50\":{},\"ratio_p95\":{}}}",
            width.label(),
            comparator.label(),
            order,
            windows,
            percentile(&ratios, 0.5),
            percentile(&ratios, 0.95)
        );
    }
    Ok((percentile(&all_ratios, 0.5), percentile(&all_ratios, 0.95)))
}

fn validate_fast_graph(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
    let graph = capture(runtime, fixture, E0, Arm::FastTf32)?;
    let mut count = 0usize;
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
        || count != 1
    {
        return Err(format!("Fast graph count {count}"));
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err("Fast graph nodes failed".into());
    }
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    if unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err("Fast graph params failed".into());
    }
    let mut name = std::ptr::null();
    if unsafe { sys::cuFuncGetName(&mut name, params.func) } != sys::CUresult::CUDA_SUCCESS
        || name.is_null()
    {
        return Err("Fast graph symbol failed".into());
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| format!("Fast UTF8: {error}"))?;
    if symbol != FAST_SYMBOL {
        return Err(format!("unexpected Fast symbol {symbol}"));
    }
    Ok(())
}

fn run_timing(width: Width) -> Result<(), String> {
    let windows = std::env::var("MAMBA_FIXED_W4_WINDOWS")
        .unwrap_or_else(|_| "7".into())
        .parse::<usize>()
        .map_err(|error| format!("parse W4 windows: {error}"))?;
    if !matches!(windows, 7 | 21) {
        return Err(format!("W4 windows must be 7 or 21, got {windows}"));
    }
    let runtime = new_runtime(width)?;
    assert_actual_auto(&runtime)?;
    let fixture = Fixture::new(&runtime, E0)?;
    let graph = capture(&runtime, &fixture, E0, Arm::Candidate(width))?;
    validate_candidate(&runtime, width, &graph)?;
    validate_fast_graph(&runtime, &fixture)?;
    let own = compare(&runtime, &fixture, width, Arm::ProductionRna, windows)?;
    let fast = compare(&runtime, &fixture, width, Arm::FastTf32, windows)?;
    let n96 = compare(&runtime, &fixture, width, Arm::CommittedN96, windows)?;
    fixture.validate_inputs(&runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32W4TimingCompleteV1\",\"arm\":\"{}\",\"windows\":{},\"own_p50\":{},\"own_p95\":{},\"fast_p50\":{},\"fast_p95\":{},\"n96_p50\":{},\"n96_p95\":{},\"own_win\":{},\"passed\":true}}",
        width.label(),
        windows,
        own.0,
        own.1,
        fast.0,
        fast.1,
        n96.0,
        n96.1,
        own.0 < 1.0 && own.1 < 1.0
    );
    Ok(())
}

#[test]
#[ignore = "bounded CUDA13.2 W4 N96 correctness; set MAMBA_FIXED_W4_DISCOVERY=1"]
fn w4_n96_small_bits_and_resources() {
    run_correctness(Width::N96).unwrap();
}

#[test]
#[ignore = "bounded CUDA13.2 W4 N128 correctness; set MAMBA_FIXED_W4_DISCOVERY=1"]
fn w4_n128_small_bits_and_resources() {
    run_correctness(Width::N128).unwrap();
}

#[test]
#[ignore = "bounded CUDA13.2 W4 N96 timing; set MAMBA_FIXED_W4_DISCOVERY=1"]
fn w4_n96_short_timing() {
    run_timing(Width::N96).unwrap();
}

#[test]
#[ignore = "bounded CUDA13.2 W4 N128 timing; set MAMBA_FIXED_W4_DISCOVERY=1"]
fn w4_n128_short_timing() {
    run_timing(Width::N128).unwrap();
}
