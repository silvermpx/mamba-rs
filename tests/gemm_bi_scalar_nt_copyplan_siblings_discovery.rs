//! Discovery only: reuse unchanged production transpose + exact Fixed CopyPlan.
//! No dispatcher admission, reduced-precision replacement, or split reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    m: usize,
    out: usize,
    reduction: usize,
}

const CELLS: [Cell; 2] = [
    Cell {
        name: "d768_in_proj",
        m: 2048,
        out: 768,
        reduction: 3072,
    },
    Cell {
        name: "prism",
        m: 4621,
        out: 384,
        reduction: 1928,
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Params {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

impl Cell {
    fn params(self) -> Params {
        Params {
            alpha: 1.0,
            beta: 0.0,
            m: self.m as i32,
            n: self.out as i32,
            k: self.reduction as i32,
            lda: self.reduction as i32,
            ldb: self.out as i32,
            ldc: self.out as i32,
        }
    }
    fn transpose_grid(self) -> (u32, u32, u32) {
        (
            self.reduction.div_ceil(32) as u32,
            self.out.div_ceil(32) as u32,
            1,
        )
    }
    fn fixed_grid(self) -> (u32, u32, u32) {
        ((self.m.div_ceil(64) * self.out.div_ceil(64)) as u32, 1, 1)
    }
}

fn ratio(raw: [f64; 4], candidate_endpoints: bool) -> Result<f64, String> {
    if raw.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err("event observations must be positive finite".into());
    }
    let endpoints = raw[0] + raw[3];
    let middle = raw[1] + raw[2];
    Ok(if candidate_endpoints {
        endpoints / middle
    } else {
        middle / endpoints
    })
}

fn quantile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty() && (0.0..=1.0).contains(&q));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1)]
}

#[test]
fn nt_siblings_map_to_exact_nn_copyplan_abi() {
    assert_eq!((size_of::<Params>(), align_of::<Params>()), (32, 4));
    for cell in CELLS {
        let p = cell.params();
        assert_eq!(
            (p.alpha.to_bits(), p.beta.to_bits()),
            (1.0_f32.to_bits(), 0)
        );
        assert_eq!(
            (p.m, p.n, p.k),
            (cell.m as i32, cell.out as i32, cell.reduction as i32)
        );
        assert_eq!((p.lda, p.ldb, p.ldc), (p.k, p.n, p.n));
    }
    assert_eq!(CELLS[0].transpose_grid(), (96, 24, 1));
    assert_eq!(CELLS[0].fixed_grid(), (384, 1, 1));
    assert_eq!(CELLS[1].transpose_grid(), (61, 12, 1));
    assert_eq!(CELLS[1].fixed_grid(), (438, 1, 1));
}

#[test]
fn transpose_mapping_preserves_each_original_nt_dot_product() {
    for cell in CELLS {
        let p = cell.params();
        for row in [0, cell.m - 1] {
            for col in [0, cell.out - 1] {
                for k in [0, cell.reduction - 1] {
                    let original_a = row * cell.reduction + k;
                    let original_b = col * cell.reduction + k;
                    assert_eq!(row * p.lda as usize + k, original_a);
                    let scratch_index = k * p.ldb as usize + col;
                    assert_eq!(
                        (scratch_index % cell.out) * cell.reduction + scratch_index / cell.out,
                        original_b
                    );
                }
            }
        }
    }
}

#[test]
fn paired_ratio_and_nearest_rank_are_not_order_ambiguous() {
    assert_eq!(ratio([8., 10., 10., 8.], true).unwrap(), 0.8);
    assert_eq!(ratio([10., 8., 8., 10.], false).unwrap(), 0.8);
    assert!(ratio([0., 1., 1., 1.], true).is_err());
    assert!(ratio([f64::NAN, 1., 1., 1.], true).is_err());
    assert_eq!(quantile(&[7., 6., 5., 4., 3., 2., 1.], 0.5), 4.);
    assert_eq!(quantile(&[7., 6., 5., 4., 3., 2., 1.], 0.95), 7.);
}

#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;

#[cfg(feature = "cuda")]
mod cuda_suite {
    use super::*;
    use cudarc::cublas::{result as blas_result, sys as blas};
    use cudarc::driver::{CudaFunction, CudaGraph, DeviceRepr, LaunchConfig, PushKernelArg, sys};
    use mamba_rs::mamba_ssm::gpu::{
        blas::gpu_gemm_bi_backward_dx_raw,
        buffers::GpuBuffer,
        context::{BiGemmFamily, F32TriadPolicy, GpuCtx},
        device::GpuDevice,
        dtype::WeightDtype,
        graph_capture::capture_into_graph,
    };
    use serde_json::json;
    use std::ffi::{CStr, c_void};

    const FIXED: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
    const TRANSPOSE: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
    const GUARD: usize = 64;
    const GUARD_BITS: u32 = 0x7fc0_3189;
    const OPS: usize = 20;
    unsafe impl DeviceRepr for Params {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Auto,
        Fast,
        Generic,
    }

    struct Buffer {
        gpu: GpuBuffer,
        seed: Vec<f32>,
        len: usize,
    }
    impl Buffer {
        // All arms use origin-zero buffers: the public facade owns A and C.
        // Trailing guards and an explicit pointer check keep the timed cohort 256B aligned.
        fn new(ctx: &GpuCtx, mut values: Vec<f32>) -> Result<Self, String> {
            let len = values.len();
            values.resize(len + GUARD, f32::from_bits(GUARD_BITS));
            let gpu = GpuBuffer::from_cpu(&ctx.stream, &values)?;
            if gpu.cached_ptr() % 256 != 0 {
                return Err("timed buffer is not 256B aligned".into());
            }
            Ok(Self {
                gpu,
                seed: values,
                len,
            })
        }
        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.gpu.upload(&ctx.stream, &self.seed)
        }
        fn bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let data = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|e| format!("download sync: {e:?}"))?;
            if data[self.len..].iter().any(|x| x.to_bits() != GUARD_BITS) {
                return Err("output/input/scratch trailing guard changed".into());
            }
            Ok(data[..self.len].iter().map(|x| x.to_bits()).collect())
        }
        fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
            let actual = self.bits(ctx)?;
            if actual
                .iter()
                .zip(&self.seed)
                .any(|(a, b)| *a != b.to_bits())
            {
                return Err("input storage bits changed".into());
            }
            Ok(())
        }
    }
    struct Fixture {
        cell: Cell,
        a: Buffer,
        b: Buffer,
        scratch: Buffer,
        output: Buffer,
    }
    impl Fixture {
        fn new(ctx: &GpuCtx, cell: Cell, exceptional: bool) -> Result<Self, String> {
            let mut a =
                full_mantissa::finite_full_mantissa_values(cell.m * cell.reduction, 0x8931_a001);
            let mut b =
                full_mantissa::finite_full_mantissa_values(cell.out * cell.reduction, 0x8931_b002);
            if exceptional {
                for (i, bits) in [0x7fc1_2345, 0xffc2_3456, 0x7f80_0000, 0xff80_0000]
                    .into_iter()
                    .enumerate()
                {
                    if i < a.len() {
                        a[i] = f32::from_bits(bits);
                    }
                    if i + 8 < b.len() {
                        b[i + 8] = f32::from_bits(bits);
                    }
                }
            }
            Ok(Self {
                cell,
                a: Buffer::new(ctx, a)?,
                b: Buffer::new(ctx, b)?,
                scratch: Buffer::new(
                    ctx,
                    vec![f32::from_bits(0x7fc0_aaaa); cell.out * cell.reduction],
                )?,
                output: Buffer::new(ctx, vec![f32::from_bits(0x7fc0_bbbb); cell.m * cell.out])?,
            })
        }
        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.reset(ctx)?;
            self.scratch.reset(ctx)
        }
        fn validate(&self, ctx: &GpuCtx, golden: &[u32], scratch: bool) -> Result<(), String> {
            let actual = self.output.bits(ctx)?;
            if let Some(i) = actual.iter().zip(golden).position(|(a, b)| a != b) {
                return Err(format!(
                    "{} exact output mismatch {i}: {:08x} != {:08x}",
                    self.cell.name, actual[i], golden[i]
                ));
            }
            self.a.unchanged(ctx)?;
            self.b.unchanged(ctx)?;
            let transposed = self.scratch.bits(ctx)?;
            if scratch {
                for row in 0..self.cell.out {
                    for k in 0..self.cell.reduction {
                        if transposed[k * self.cell.out + row]
                            != self.b.seed[row * self.cell.reduction + k].to_bits()
                        {
                            return Err(format!("transpose changed B bits at ({row},{k})"));
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn config(grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared,
        }
    }
    fn launch(ctx: &GpuCtx, fixed: &CudaFunction, f: &mut Fixture, arm: Arm) -> Result<(), String> {
        let cell = f.cell;
        let a = f.a.gpu.cached_ptr();
        let b = f.b.gpu.cached_ptr();
        let output = f.output.gpu.cached_ptr();
        let p = cell.params();
        match arm {
            Arm::Auto => gpu_gemm_bi_backward_dx_raw(
                ctx,
                &mut f.output.gpu,
                &f.a.gpu,
                b,
                cell.m,
                cell.out,
                cell.reduction,
            ),
            Arm::Candidate => {
                let scratch = f.scratch.gpu.cached_ptr();
                if cell.reduction != 0 {
                    let mut t = ctx
                        .stream
                        .launch_builder(&ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1);
                    t.arg(&scratch);
                    t.arg(&b);
                    t.arg(&p.n);
                    t.arg(&p.k);
                    unsafe { t.launch(config(cell.transpose_grid(), (32, 16, 1), 0)) }
                        .map_err(|e| format!("transpose: {e:?}"))?;
                }
                let bias = 0_u64;
                let mut k = ctx.stream.launch_builder(fixed);
                k.arg(&output);
                k.arg(&a);
                k.arg(&scratch);
                k.arg(&bias);
                k.arg(&p);
                unsafe { k.launch(config(cell.fixed_grid(), (128, 1, 1), 0)) }
                    .map(|_| ())
                    .map_err(|e| format!("CopyPlan: {e:?}"))
            }
            Arm::Generic => {
                let mut k = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_nt);
                k.arg(&output);
                k.arg(&a);
                k.arg(&b);
                k.arg(&p.alpha);
                k.arg(&p.m);
                k.arg(&p.k);
                k.arg(&p.n);
                unsafe {
                    k.launch(config(
                        ((cell.m.div_ceil(128) * cell.out.div_ceil(128)) as u32, 1, 1),
                        (256, 1, 1),
                        33_376,
                    ))
                }
                .map(|_| ())
                .map_err(|e| format!("generic exact NT: {e:?}"))
            }
            Arm::Fast => {
                let dtype = WeightDtype::F32.cuda_data_type();
                // C^T = B * A^T in column-major storage: transpose physical B.
                unsafe {
                    blas_result::gemm_ex(
                        *ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_T,
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        p.n,
                        p.m,
                        p.k,
                        (&p.alpha as *const f32).cast::<c_void>(),
                        b as *const c_void,
                        dtype,
                        p.k.max(1),
                        a as *const c_void,
                        dtype,
                        p.k.max(1),
                        (&p.beta as *const f32).cast::<c_void>(),
                        output as *mut c_void,
                        dtype,
                        p.n,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|e| format!("explicit cuBLAS FAST_TF32 NT: {e:?}"))
            }
        }
    }

    fn capture(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        // Resolve public pointer-bound AUTO caches before entering capture.
        launch(ctx, fixed, f, arm)?;
        ctx.stream
            .synchronize()
            .map_err(|e| format!("capture warmup: {e:?}"))?;
        unsafe { capture_into_graph(&ctx.stream, || launch(ctx, fixed, f, arm)) }
    }

    fn graph_identity(graph: &CudaGraph, f: &Fixture, arm: Arm) -> Result<(), String> {
        unsafe {
            let mut count = 0;
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count == 0
            {
                return Err("empty/unqueryable graph".into());
            }
            if arm == Arm::Fast {
                println!(
                    "{}",
                    json!({"schema":"MambaBiNtCopyPlanSiblingsGraphV1","cell":f.cell.name,"arm":"Fast","node_count":count,"vendor_abi":"opaque","timing":"whole_graph"})
                );
                return Ok(());
            }
            let mut nodes = vec![std::ptr::null_mut(); count];
            if sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
            {
                return Err("graph node query failed".into());
            }
            let mut identities = Vec::new();
            let mut candidate_names = Vec::new();
            for node in nodes {
                let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                if sys::cuGraphNodeGetType(node, &mut ty) != sys::CUresult::CUDA_SUCCESS {
                    return Err("graph type query failed".into());
                }
                if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                    if arm != Arm::Fast {
                        return Err("custom graph contains a non-kernel node".into());
                    }
                    identities.push(json!({"type":format!("{ty:?}")}));
                    continue;
                }
                let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                if sys::cuGraphKernelNodeGetParams_v2(node, &mut params)
                    != sys::CUresult::CUDA_SUCCESS
                {
                    return Err("graph params query failed".into());
                }
                let mut name = std::ptr::null();
                // Vendor kernels may be opaque; our own kernels must have identities.
                let named = sys::cuFuncGetName(&mut name, params.func)
                    == sys::CUresult::CUDA_SUCCESS
                    && !name.is_null();
                if !named && arm != Arm::Fast {
                    return Err("custom graph symbol unavailable".into());
                }
                let name = if named {
                    CStr::from_ptr(name).to_string_lossy().into_owned()
                } else {
                    "<vendor-opaque>".into()
                };
                let grid = (params.gridDimX, params.gridDimY, params.gridDimZ);
                let block = (params.blockDimX, params.blockDimY, params.blockDimZ);
                if arm == Arm::Candidate {
                    let (expected_grid, expected_block) = match name.as_str() {
                        TRANSPOSE => (f.cell.transpose_grid(), (32, 16, 1)),
                        FIXED => (f.cell.fixed_grid(), (128, 1, 1)),
                        _ => return Err(format!("wrong candidate symbol {name}")),
                    };
                    if grid != expected_grid
                        || block != expected_block
                        || params.sharedMemBytes != 0
                    {
                        return Err(format!("candidate launch changed: {name}"));
                    }
                    candidate_names.push(name.clone());
                }
                identities.push(json!({"symbol":name,"grid":grid,"block":block,"dynamic_shared":params.sharedMemBytes}));
            }
            if arm == Arm::Candidate {
                candidate_names.sort();
                let mut expected = if f.cell.reduction == 0 {
                    vec![FIXED.to_owned()]
                } else {
                    vec![FIXED.to_owned(), TRANSPOSE.to_owned()]
                };
                expected.sort();
                if candidate_names != expected {
                    return Err("candidate must time the entire expected pipeline".into());
                }
            }
            println!(
                "{}",
                json!({"schema":"MambaBiNtCopyPlanSiblingsGraphV1","cell":f.cell.name,"arm":format!("{arm:?}"),"nodes":identities})
            );
        }
        Ok(())
    }

    fn resources(
        function: &CudaFunction,
        symbol: &str,
        threads: u32,
        shared: usize,
    ) -> Result<(), String> {
        let regs = function.num_regs().map_err(|e| format!("regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("local: {e:?}"))?;
        let actual_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("shared: {e:?}"))?;
        let occ = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .map_err(|e| format!("occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({"schema":"MambaBiNtCopyPlanSiblingsResourceV1","symbol":symbol,"threads":threads,"registers":regs,"local_bytes":local,"static_shared_bytes":actual_shared,"dynamic_shared_bytes":0,"occupancy":occ})
        );
        if local != 0 || actual_shared as usize != shared || occ == 0 {
            return Err(format!("{symbol} resource gate failed"));
        }
        Ok(())
    }

    fn check_bits(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        f.reset(ctx)?;
        launch(ctx, fixed, f, Arm::Generic)?;
        let exact = f.output.bits(ctx)?;
        let mut fast_golden = Vec::new();
        for arm in [Arm::Auto, Arm::Candidate, Arm::Fast] {
            if arm == Arm::Fast && f.cell.reduction == 0 {
                continue;
            }
            f.reset(ctx)?;
            launch(ctx, fixed, f, arm)?;
            let eager = f.output.bits(ctx)?;
            let golden = if arm == Arm::Fast {
                fast_golden = eager;
                &fast_golden
            } else {
                &exact
            };
            f.validate(ctx, golden, arm == Arm::Candidate)?;
            let graph = capture(ctx, fixed, f, arm)?;
            graph_identity(&graph, f, arm)?;
            for path in ["eager", "graph"] {
                for repeat in 0..2 {
                    f.reset(ctx)?;
                    if path == "graph" {
                        graph.launch().map_err(|e| format!("bits graph: {e:?}"))?;
                    } else {
                        launch(ctx, fixed, f, arm)?;
                    }
                    f.validate(ctx, golden, arm == Arm::Candidate)?;
                    println!(
                        "{}",
                        json!({"schema":"MambaBiNtCopyPlanSiblingsBitsV1","cell":f.cell.name,"arm":format!("{arm:?}"),"path":path,"repeat":repeat,"words":golden.len(),"oracle":if arm == Arm::Fast {"vendor_self"} else {"generic_exact"}})
                    );
                }
            }
        }
        Ok((exact, fast_golden))
    }

    fn measure(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
        arm: Arm,
        graph: &CudaGraph,
        path: &str,
        golden: &[u32],
    ) -> Result<f64, String> {
        f.reset(ctx)?;
        let start = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("start: {e:?}"))?;
        for _ in 0..OPS {
            if path == "graph" {
                graph.launch().map_err(|e| format!("timed graph: {e:?}"))?;
            } else {
                launch(ctx, fixed, f, arm)?;
            }
        }
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("end: {e:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("elapsed: {e:?}"))?,
        ) * 1000.0
            / OPS as f64;
        // Downloads and resets are outside both events; all logical output is overwritten.
        f.validate(ctx, golden, arm == Arm::Candidate)?;
        if !us.is_finite() || us <= 0.0 {
            return Err("invalid elapsed time".into());
        }
        Ok(us)
    }

    #[test]
    #[ignore = "Ada CUDA13.2 discovery: two F32 NT siblings, bits then short AUTO/Fast pairs"]
    fn ada_f32_nt_copyplan_in_prism_discovery_once7() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("performance requires release".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-copyplan-siblings/pre")?;
        let device = GpuDevice::new(0)?;
        let id = device.identity();
        if id.compute_capability != (8, 9) || id.multiprocessor_count != 142 {
            return Err("requires RTX6000Ada/142 SM".into());
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(false);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        for compiler in [
            ctx.kernels.compiler_identity(),
            ctx.kernels.triad_scalar_compiler_identity(),
        ] {
            if compiler.nvrtc_version != (13, 2)
                || compiler.target.as_str() != "sm_89"
                || !compiler.nvrtc_library_known
            {
                return Err(format!("wrong discovery compiler {compiler:?}"));
            }
            println!(
                "{}",
                json!({"schema":"MambaBiNtCopyPlanSiblingsCompilerV1","compiler":format!("{compiler:?}")})
            );
        }
        println!(
            "{}",
            json!({"schema":"MambaBiNtCopyPlanSiblingsArtifactsV1","artifacts":format!("{:?}",ctx.kernels.artifact_set_identity()),"fast_compute":"CUBLAS_COMPUTE_32F_FAST_TF32","algorithm":"CUBLAS_GEMM_DEFAULT","alpha":1,"beta":0,"bias":false,"promotion":false})
        );
        let fixed = ctx
            .kernels
            .fixed_sm89_f32_n64_copyplan
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "CopyPlan unavailable: {:?}",
                    ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                )
            })?
            .clone();
        let mut math = blas::cublasMath_t::CUBLAS_DEFAULT_MATH;
        let mut pointer = blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
        unsafe {
            if blas::cublasGetMathMode(*ctx.blas.handle(), &mut math)
                != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || blas::cublasGetPointerMode_v2(*ctx.blas.handle(), &mut pointer)
                    != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || math == blas::cublasMath_t::CUBLAS_PEDANTIC_MATH
                || pointer != blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST
            {
                return Err("Fast handle mode is not supported".into());
            }
        }
        resources(&fixed, FIXED, 128, 32768)?;
        resources(
            &ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            TRANSPOSE,
            512,
            4224,
        )?;
        for (cell, exceptional) in [
            (
                Cell {
                    name: "tail",
                    m: 67,
                    out: 68,
                    reduction: 36,
                },
                false,
            ),
            (
                Cell {
                    name: "exceptional",
                    m: 67,
                    out: 68,
                    reduction: 36,
                },
                true,
            ),
            (
                Cell {
                    name: "zero_reduction",
                    m: 67,
                    out: 68,
                    reduction: 0,
                },
                false,
            ),
        ] {
            check_bits(&ctx, &fixed, &mut Fixture::new(&ctx, cell, exceptional)?)?;
        }
        for cell in CELLS {
            let mut f = Fixture::new(&ctx, cell, false)?;
            let (exact, fast) = check_bits(&ctx, &fixed, &mut f)?;
            let candidate_graph = capture(&ctx, &fixed, &mut f, Arm::Candidate)?;
            for comparator in [Arm::Auto, Arm::Fast] {
                let comparator_graph = capture(&ctx, &fixed, &mut f, comparator)?;
                quiet.require_cohort("nt-copyplan-siblings/timed")?;
                let mut strata = Vec::new();
                for path in ["eager", "graph"] {
                    for candidate_endpoints in [true, false] {
                        let arms = if candidate_endpoints {
                            [Arm::Candidate, comparator, comparator, Arm::Candidate]
                        } else {
                            [comparator, Arm::Candidate, Arm::Candidate, comparator]
                        };
                        let mut raw = Vec::new();
                        for bracket in 0..9 {
                            let mut observations = [0.; 4];
                            for (i, arm) in arms.into_iter().enumerate() {
                                let graph = if arm == Arm::Candidate {
                                    &candidate_graph
                                } else {
                                    &comparator_graph
                                };
                                let golden = if arm == Arm::Fast { &fast } else { &exact };
                                observations[i] =
                                    measure(&ctx, &fixed, &mut f, arm, graph, path, golden)?;
                            }
                            if bracket >= 2 {
                                raw.push(observations);
                            }
                        }
                        let ratios = raw
                            .iter()
                            .map(|r| ratio(*r, candidate_endpoints))
                            .collect::<Result<Vec<_>, _>>()?;
                        let p50 = quantile(&ratios, 0.5);
                        let p95 = quantile(&ratios, 0.95);
                        strata.push([p50, p95]);
                        println!(
                            "{}",
                            json!({"schema":"MambaBiNtCopyPlanSiblingsScreenV1","cell":cell.name,"shape":[cell.m,cell.out,cell.reduction],"comparator":format!("{comparator:?}"),"path":path,"order":if candidate_endpoints {"ABBA"} else {"BAAB"},"observation_arms":arms.map(|a|format!("{a:?}")),"windows":7,"logical_gemms_per_observation":OPS,"raw_observations_us":raw,"ratio_direction":"candidate_over_comparator","ratio_p50":p50,"ratio_p95":p95})
                        );
                    }
                }
                let retain = strata.iter().all(|s| s[0] < 0.99 && s[1] < 0.99);
                println!(
                    "{}",
                    json!({"schema":"MambaBiNtCopyPlanSiblingsDecisionV1","cell":cell.name,"comparator":format!("{comparator:?}"),"strata":strata,"retain":retain,"promotion":false,"decision":if retain {"shortlist"} else {"stop_no_retry"}})
                );
            }
        }
        quiet.verify_post_cohort("nt-copyplan-siblings/post")?;
        Ok(())
    }
}
