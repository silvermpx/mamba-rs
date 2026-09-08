//! Discovery only: raw transpose X, unchanged exact Fixed CopyPlan, exact TN epilogue.
//! No dispatcher admission and no change to production kernels.

#[path = "support/triad_f32_tn_copyplan_epilogue_source.rs"]
mod epilogue_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    batch: usize,
    k_out: usize,
    n_out: usize,
}

const D768_IN: Cell = Cell {
    batch: 2048,
    k_out: 768,
    n_out: 3072,
};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct FixedParams {
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
    fn fixed_params(self) -> FixedParams {
        FixedParams {
            alpha: 1.0,
            beta: 0.0,
            m: self.k_out as i32,
            n: self.n_out as i32,
            k: self.batch as i32,
            lda: self.batch as i32,
            ldb: self.n_out as i32,
            ldc: self.n_out as i32,
        }
    }

    fn transpose_grid(self) -> (u32, u32, u32) {
        (
            self.k_out.div_ceil(32) as u32,
            self.batch.div_ceil(32) as u32,
            1,
        )
    }

    fn fixed_grid(self) -> (u32, u32, u32) {
        (
            (self.k_out.div_ceil(64) * self.n_out.div_ceil(64)) as u32,
            1,
            1,
        )
    }

    fn epilogue_grid(self) -> (u32, u32, u32) {
        ((self.k_out * self.n_out).div_ceil(256) as u32, 1, 1)
    }

    fn candidate_node_count(self) -> usize {
        if self.batch == 0 { 1 } else { 3 }
    }

    fn transposed_a_index(self, batch_row: usize, k_column: usize) -> usize {
        assert!(batch_row < self.batch && k_column < self.k_out);
        k_column * self.batch + batch_row
    }
}

fn exact_tn_oracle(cell: Cell, alpha: f32, output: &[f32], x: &[f32], dy: &[f32]) -> Vec<u32> {
    assert_eq!(output.len(), cell.k_out * cell.n_out);
    assert_eq!(x.len(), cell.batch * cell.k_out);
    assert_eq!(dy.len(), cell.batch * cell.n_out);
    let mut result = Vec::with_capacity(output.len());
    for k in 0..cell.k_out {
        for n in 0..cell.n_out {
            let mut accumulator = 0.0_f32;
            for m in 0..cell.batch {
                accumulator = x[m * cell.k_out + k].mul_add(dy[m * cell.n_out + n], accumulator);
            }
            result.push(
                alpha
                    .mul_add(accumulator, output[k * cell.n_out + n])
                    .to_bits(),
            );
        }
    }
    result
}

#[test]
fn tn_d768_in_maps_to_exact_three_node_copyplan_pipeline() {
    assert_eq!(
        (size_of::<FixedParams>(), align_of::<FixedParams>()),
        (32, 4)
    );
    let p = D768_IN.fixed_params();
    assert_eq!(
        (p.alpha.to_bits(), p.beta.to_bits()),
        (1.0_f32.to_bits(), 0)
    );
    assert_eq!((p.m, p.n, p.k), (768, 3072, 2048));
    assert_eq!((p.lda, p.ldb, p.ldc), (2048, 3072, 3072));
    assert_eq!(D768_IN.transpose_grid(), (24, 64, 1));
    assert_eq!(D768_IN.fixed_grid(), (576, 1, 1));
    assert_eq!(D768_IN.epilogue_grid(), (9216, 1, 1));
    assert_eq!(D768_IN.candidate_node_count(), 3);
    assert_eq!(
        Cell {
            batch: 0,
            k_out: 68,
            n_out: 69,
        }
        .candidate_node_count(),
        1
    );
    for (batch_row, k_column, expected) in [
        (0, 0, 0),
        (2047, 0, 2047),
        (0, 767, 767 * 2048),
        (2047, 767, 768 * 2048 - 1),
    ] {
        assert_eq!(D768_IN.transposed_a_index(batch_row, k_column), expected);
    }
    let source = epilogue_source::compose_source().unwrap();
    assert!(source.contains(epilogue_source::SYMBOL));
    assert!(
        source.contains("output[linear] = __fmaf_rn(alpha, accumulator[linear], output[linear]);")
    );
    assert!(!source.contains("__fmul_rn(alpha"));
}

#[test]
fn independent_literal_oracle_keeps_tn_reduction_then_single_fused_epilogue() {
    let cell = Cell {
        batch: 3,
        k_out: 2,
        n_out: 2,
    };
    let output = [0.25, -0.5, 0.75, -1.0];
    let x = [1.0, -2.0, 0.5, 4.0, -3.0, 0.25];
    let dy = [2.0, -1.0, 3.0, 0.5, -0.25, 8.0];
    assert_eq!(
        exact_tn_oracle(cell, -0.75, &output, &x, &dy),
        [0xc03c_0000, 0x4190_8000, 0xc0a6_8000, 0xc0b0_0000]
    );

    let zero = Cell {
        batch: 0,
        k_out: 1,
        n_out: 2,
    };
    let zero_seed = [-0.0_f32, f32::from_bits(0x7fc1_2345)];
    assert_eq!(
        exact_tn_oracle(zero, 1.0, &zero_seed, &[], &[]),
        [0x0000_0000, 0x7fc1_2345]
    );
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
fn paired_ratio_and_nearest_rank_are_order_unambiguous() {
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
    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg, sys,
    };
    use mamba_rs::mamba_ssm::gpu::{
        blas::gpu_gemm_bi_backward_dw_grad,
        buffers::{GpuBuffer, GradSlice},
        context::{BiGemmFamily, F32TriadPolicy, GpuCtx},
        device::GpuDevice,
        dtype::WeightDtype,
        graph_capture::capture_into_graph,
        kernels::cuda_include_paths,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        ffi::{CStr, c_void},
        sync::Arc,
    };

    const FIXED: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
    const TRANSPOSE: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
    const EPILOGUE: &str = epilogue_source::SYMBOL;
    const GUARD: usize = 64;
    const GUARD_BITS: u32 = 0x7fc0_3189;
    const OPS: usize = 20;
    unsafe impl DeviceRepr for FixedParams {}

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
        label: &'static str,
    }

    impl Buffer {
        fn new(ctx: &GpuCtx, mut values: Vec<f32>, label: &'static str) -> Result<Self, String> {
            let len = values.len();
            values.resize(len + GUARD, f32::from_bits(GUARD_BITS));
            let gpu = GpuBuffer::from_cpu(&ctx.stream, &values)?;
            if gpu.cached_ptr() % 256 != 0 {
                return Err(format!("{label} base is not 256B aligned"));
            }
            Ok(Self {
                gpu,
                seed: values,
                len,
                label,
            })
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.gpu.upload(&ctx.stream, &self.seed)
        }

        fn bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let data = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|e| format!("{} download sync: {e:?}", self.label))?;
            if data[self.len..].iter().any(|x| x.to_bits() != GUARD_BITS) {
                return Err(format!("{} trailing guard changed", self.label));
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
                return Err(format!("{} input bits changed", self.label));
            }
            Ok(())
        }
    }

    struct Fixture {
        cell: Cell,
        alpha: f32,
        output: Buffer,
        x: Buffer,
        dy: Buffer,
        transposed: Buffer,
        accumulator: Buffer,
    }

    impl Fixture {
        fn new(ctx: &GpuCtx, cell: Cell, alpha: f32, exceptional: bool) -> Result<Self, String> {
            let mut output =
                full_mantissa::finite_full_mantissa_values(cell.k_out * cell.n_out, 0x8931_c003);
            let mut x =
                full_mantissa::finite_full_mantissa_values(cell.batch * cell.k_out, 0x8931_a001);
            let mut dy =
                full_mantissa::finite_full_mantissa_values(cell.batch * cell.n_out, 0x8931_b002);
            if exceptional {
                let cases = [
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0001,
                    0x8000_0001,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc1_2345,
                    0x7fa1_2345,
                ];
                for (i, bits) in cases.into_iter().enumerate() {
                    if i < x.len() {
                        x[i] = f32::from_bits(bits);
                    }
                    if i + 16 < dy.len() {
                        dy[i + 16] = f32::from_bits(bits);
                    }
                    if i + 32 < output.len() {
                        output[i + 32] = f32::from_bits(bits);
                    }
                }
            }
            Ok(Self {
                cell,
                alpha,
                output: Buffer::new(ctx, output, "output")?,
                x: Buffer::new(ctx, x, "x")?,
                dy: Buffer::new(ctx, dy, "dy")?,
                transposed: Buffer::new(
                    ctx,
                    vec![f32::from_bits(0x7fc0_aaaa); cell.batch * cell.k_out],
                    "transposed_x",
                )?,
                accumulator: Buffer::new(ctx, vec![0.0; cell.k_out * cell.n_out], "accumulator")?,
            })
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.reset(ctx)?;
            self.transposed.reset(ctx)?;
            self.accumulator.reset(ctx)
        }

        fn validate(
            &self,
            ctx: &GpuCtx,
            golden: &[u32],
            candidate: bool,
            check_transpose_mapping: bool,
        ) -> Result<(), String> {
            let actual = self.output.bits(ctx)?;
            if let Some(i) = actual.iter().zip(golden).position(|(a, b)| a != b) {
                return Err(format!(
                    "exact TN mismatch {i}: {:08x} != {:08x}",
                    actual[i], golden[i]
                ));
            }
            self.x.unchanged(ctx)?;
            self.dy.unchanged(ctx)?;
            if candidate {
                let transposed = self.transposed.bits(ctx)?;
                self.accumulator.bits(ctx)?;
                if self.cell.batch > 0 && check_transpose_mapping {
                    for row in 0..self.cell.batch {
                        for column in 0..self.cell.k_out {
                            if transposed[self.cell.transposed_a_index(row, column)]
                                != self.x.seed[row * self.cell.k_out + column].to_bits()
                            {
                                return Err(format!(
                                    "raw X transpose changed bits at ({row},{column})"
                                ));
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _module: Arc<CudaModule>,
        epilogue: CudaFunction,
        source_sha: String,
    }

    fn config(grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared,
        }
    }

    fn new_runtime() -> Result<Runtime, String> {
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
        }
        let source = epilogue_source::compose_source()?;
        let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec!["--fmad=true".into(), "-DNDEBUG".into()],
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|e| format!("compile exact TN epilogue: {e:?}"))?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx.to_src()))
            .map_err(|e| format!("load exact TN epilogue module: {e:?}"))?;
        let epilogue = module
            .load_function(EPILOGUE)
            .map_err(|e| format!("load {EPILOGUE}: {e:?}"))?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            epilogue,
            source_sha,
        })
    }

    fn launch(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<(), String> {
        let ctx = &runtime.ctx;
        let cell = fixture.cell;
        let output = fixture.output.gpu.cached_ptr();
        let x = fixture.x.gpu.cached_ptr();
        let dy = fixture.dy.gpu.cached_ptr();
        match arm {
            Arm::Auto => gpu_gemm_bi_backward_dw_grad(
                ctx,
                &GradSlice::from_raw(output, cell.k_out * cell.n_out),
                &fixture.dy.gpu,
                &fixture.x.gpu,
                cell.batch,
                cell.k_out,
                cell.n_out,
            ),
            Arm::Candidate => {
                let transposed = fixture.transposed.gpu.cached_ptr();
                let accumulator = fixture.accumulator.gpu.cached_ptr();
                if cell.batch > 0 {
                    let rows = cell.batch as i32;
                    let columns = cell.k_out as i32;
                    let mut transpose = ctx
                        .stream
                        .launch_builder(&ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1);
                    transpose.arg(&transposed);
                    transpose.arg(&x);
                    transpose.arg(&rows);
                    transpose.arg(&columns);
                    unsafe { transpose.launch(config(cell.transpose_grid(), (32, 16, 1), 0)) }
                        .map_err(|e| format!("transpose X: {e:?}"))?;

                    let fixed = ctx
                        .kernels
                        .fixed_sm89_f32_n64_copyplan
                        .as_ref()
                        .ok_or_else(|| {
                            format!(
                                "CopyPlan unavailable: {:?}",
                                ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                            )
                        })?;
                    let null_bias = 0_u64;
                    let params = cell.fixed_params();
                    let mut gemm = ctx.stream.launch_builder(fixed);
                    gemm.arg(&accumulator);
                    gemm.arg(&transposed);
                    gemm.arg(&dy);
                    gemm.arg(&null_bias);
                    gemm.arg(&params);
                    unsafe { gemm.launch(config(cell.fixed_grid(), (128, 1, 1), 0)) }
                        .map_err(|e| format!("exact CopyPlan: {e:?}"))?;
                }
                let elements = (cell.k_out * cell.n_out) as u64;
                let mut epilogue = ctx.stream.launch_builder(&runtime.epilogue);
                epilogue.arg(&output);
                epilogue.arg(&accumulator);
                epilogue.arg(&fixture.alpha);
                epilogue.arg(&elements);
                unsafe { epilogue.launch(config(cell.epilogue_grid(), (256, 1, 1), 0)) }
                    .map(|_| ())
                    .map_err(|e| format!("exact TN epilogue: {e:?}"))
            }
            Arm::Generic => {
                if cell.batch == 0 {
                    return Err("generic non-unit probe requires positive reduction".into());
                }
                let m = cell.batch as i32;
                let k = cell.k_out as i32;
                let n = cell.n_out as i32;
                let mut kernel = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_tn);
                kernel.arg(&output);
                kernel.arg(&x);
                kernel.arg(&dy);
                kernel.arg(&fixture.alpha);
                kernel.arg(&m);
                kernel.arg(&k);
                kernel.arg(&n);
                unsafe {
                    kernel.launch(config(
                        (
                            (cell.k_out.div_ceil(128) * cell.n_out.div_ceil(128)) as u32,
                            1,
                            1,
                        ),
                        (256, 1, 1),
                        34 * 1024,
                    ))
                }
                .map(|_| ())
                .map_err(|e| format!("generic exact TN: {e:?}"))
            }
            Arm::Fast => {
                if cell.batch == 0 {
                    return Err("Fast comparator does not admit zero reduction".into());
                }
                let dtype = WeightDtype::F32.cuda_data_type();
                let alpha = 1.0_f32;
                let beta = 1.0_f32;
                unsafe {
                    blas_result::gemm_ex(
                        *ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        blas::cublasOperation_t::CUBLAS_OP_T,
                        cell.n_out as i32,
                        cell.k_out as i32,
                        cell.batch as i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        dy as *const c_void,
                        dtype,
                        cell.n_out as i32,
                        x as *const c_void,
                        dtype,
                        cell.k_out as i32,
                        (&beta as *const f32).cast::<c_void>(),
                        output as *mut c_void,
                        dtype,
                        cell.n_out as i32,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|e| format!("explicit cuBLAS Fast TN: {e:?}"))
            }
        }
    }

    fn capture(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<CudaGraph, String> {
        launch(runtime, fixture, arm)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("capture warmup: {e:?}"))?;
        fixture.reset(&runtime.ctx)?;
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, arm)) }
    }

    fn graph_identity(graph: &CudaGraph, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        unsafe {
            let mut count = 0;
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count == 0
            {
                return Err("empty/unqueryable graph".into());
            }
            if arm != Arm::Candidate {
                println!(
                    "{}",
                    json!({"schema":"MambaBiF32TnCopyPlanGraphV1","arm":format!("{arm:?}"),"node_count":count,"abi":"opaque","timing":"whole_graph"})
                );
                return Ok(());
            }
            let mut nodes = vec![std::ptr::null_mut(); count];
            if sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
            {
                return Err("candidate graph node query failed".into());
            }
            let mut identities = Vec::new();
            let mut names = Vec::new();
            for node in nodes {
                let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                if sys::cuGraphNodeGetType(node, &mut ty) != sys::CUresult::CUDA_SUCCESS
                    || ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL
                {
                    return Err("candidate graph contains non-kernel node".into());
                }
                let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                if sys::cuGraphKernelNodeGetParams_v2(node, &mut params)
                    != sys::CUresult::CUDA_SUCCESS
                {
                    return Err("candidate graph params unavailable".into());
                }
                let mut name_ptr = std::ptr::null();
                if sys::cuFuncGetName(&mut name_ptr, params.func) != sys::CUresult::CUDA_SUCCESS
                    || name_ptr.is_null()
                {
                    return Err("candidate graph symbol unavailable".into());
                }
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                let (grid, block) = match name.as_str() {
                    TRANSPOSE if fixture.cell.batch > 0 => {
                        (fixture.cell.transpose_grid(), (32, 16, 1))
                    }
                    FIXED if fixture.cell.batch > 0 => (fixture.cell.fixed_grid(), (128, 1, 1)),
                    EPILOGUE => (fixture.cell.epilogue_grid(), (256, 1, 1)),
                    _ => return Err(format!("wrong candidate graph symbol {name}")),
                };
                if (params.gridDimX, params.gridDimY, params.gridDimZ) != grid
                    || (params.blockDimX, params.blockDimY, params.blockDimZ) != block
                    || params.sharedMemBytes != 0
                {
                    return Err(format!("candidate launch changed for {name}"));
                }
                names.push(name.clone());
                identities
                    .push(json!({"symbol":name,"grid":grid,"block":block,"dynamic_shared":0}));
            }
            names.sort();
            let mut expected = if fixture.cell.batch == 0 {
                vec![EPILOGUE.to_owned()]
            } else {
                vec![EPILOGUE.to_owned(), FIXED.to_owned(), TRANSPOSE.to_owned()]
            };
            expected.sort();
            if names != expected {
                return Err(format!("candidate graph pipeline changed: {names:?}"));
            }
            println!(
                "{}",
                json!({"schema":"MambaBiF32TnCopyPlanGraphV1","arm":"Candidate","nodes":identities,"timing":"whole_graph"})
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
        let static_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("shared: {e:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .map_err(|e| format!("occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({"schema":"MambaBiF32TnCopyPlanResourceV1","symbol":symbol,"threads":threads,"registers":regs,"local_bytes":local,"static_shared_bytes":static_shared,"dynamic_shared_bytes":0,"occupancy":occupancy})
        );
        if local != 0 || static_shared as usize != shared || occupancy == 0 {
            return Err(format!("{symbol} resource gate failed"));
        }
        Ok(())
    }

    fn launch_path(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: Option<&CudaGraph>,
    ) -> Result<(), String> {
        if let Some(graph) = graph {
            graph.launch().map_err(|e| format!("graph launch: {e:?}"))
        } else {
            launch(runtime, fixture, arm)
        }
    }

    fn output_after(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: Option<&CudaGraph>,
        repeats: usize,
    ) -> Result<Vec<u32>, String> {
        fixture.reset(&runtime.ctx)?;
        for _ in 0..repeats {
            launch_path(runtime, fixture, arm, graph)?;
        }
        fixture.output.bits(&runtime.ctx)
    }

    fn check_exact_bits(
        runtime: &Runtime,
        fixture: &mut Fixture,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let auto_graph = capture(runtime, fixture, Arm::Auto)?;
        let candidate_graph = capture(runtime, fixture, Arm::Candidate)?;
        graph_identity(&auto_graph, fixture, Arm::Auto)?;
        let single = output_after(runtime, fixture, Arm::Auto, None, 1)?;
        let repeated = output_after(runtime, fixture, Arm::Auto, None, OPS)?;
        for (arm, graph) in [(Arm::Auto, &auto_graph), (Arm::Candidate, &candidate_graph)] {
            graph_identity(graph, fixture, arm)?;
            for path in ["eager", "graph"] {
                for repeat in 0..2 {
                    let actual =
                        output_after(runtime, fixture, arm, (path == "graph").then_some(graph), 1)?;
                    fixture.validate(&runtime.ctx, &single, arm == Arm::Candidate, true)?;
                    if actual != single {
                        return Err(format!("{arm:?} {path} repeat {repeat} changed bits"));
                    }
                    println!(
                        "{}",
                        json!({"schema":"MambaBiF32TnCopyPlanBitsV1","shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],"alpha_bits":fixture.alpha.to_bits(),"arm":format!("{arm:?}"),"path":path,"repeat":repeat,"words":single.len(),"oracle":"actual_auto_exact"})
                    );
                }
            }
        }
        Ok((single, repeated))
    }

    fn check_nonunit_bits(runtime: &Runtime, fixture: &mut Fixture) -> Result<(), String> {
        let generic_graph = capture(runtime, fixture, Arm::Generic)?;
        let candidate_graph = capture(runtime, fixture, Arm::Candidate)?;
        graph_identity(&candidate_graph, fixture, Arm::Candidate)?;
        let golden = output_after(runtime, fixture, Arm::Generic, None, 1)?;
        for (arm, graph) in [
            (Arm::Generic, &generic_graph),
            (Arm::Candidate, &candidate_graph),
        ] {
            for path in ["eager", "graph"] {
                for repeat in 0..2 {
                    let actual =
                        output_after(runtime, fixture, arm, (path == "graph").then_some(graph), 1)?;
                    fixture.validate(&runtime.ctx, &golden, arm == Arm::Candidate, true)?;
                    if actual != golden {
                        return Err(format!(
                            "non-unit {arm:?} {path} repeat {repeat} changed bits"
                        ));
                    }
                    println!(
                        "{}",
                        json!({"schema":"MambaBiF32TnCopyPlanNonunitBitsV1","shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],"alpha_bits":fixture.alpha.to_bits(),"arm":format!("{arm:?}"),"path":path,"repeat":repeat,"words":golden.len(),"oracle":"generic_exact_tn"})
                    );
                }
            }
        }
        Ok(())
    }

    fn fast_goldens(
        runtime: &Runtime,
        fixture: &mut Fixture,
        graph: &CudaGraph,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let single = output_after(runtime, fixture, Arm::Fast, None, 1)?;
        if single.is_empty()
            || single.iter().all(|x| x & 0x7fff_ffff == 0)
            || single.iter().any(|x| !f32::from_bits(*x).is_finite())
        {
            return Err("Fast comparator produced invalid output".into());
        }
        for path in ["eager", "graph"] {
            for repeat in 0..2 {
                let actual = output_after(
                    runtime,
                    fixture,
                    Arm::Fast,
                    (path == "graph").then_some(graph),
                    1,
                )?;
                if actual != single {
                    return Err(format!("Fast {path} repeat {repeat} changed its own bits"));
                }
            }
        }
        let repeated = output_after(runtime, fixture, Arm::Fast, None, OPS)?;
        Ok((single, repeated))
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: &CudaGraph,
        graph_path: bool,
        golden: &[u32],
    ) -> Result<f64, String> {
        fixture.reset(&runtime.ctx)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("start event: {e:?}"))?;
        for _ in 0..OPS {
            launch_path(runtime, fixture, arm, graph_path.then_some(graph))?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("end event: {e:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("elapsed: {e:?}"))?,
        ) * 1000.0
            / OPS as f64;
        fixture.validate(&runtime.ctx, golden, arm == Arm::Candidate, false)?;
        if !us.is_finite() || us <= 0.0 {
            return Err("invalid elapsed time".into());
        }
        Ok(us)
    }

    #[test]
    #[ignore = "Ada CUDA13.2 discovery: exact F32 TN d768-in three-node reuse"]
    fn ada_f32_tn_copyplan_d768_in_discovery_once7() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("performance requires release".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("f32-tn-copyplan/pre")?;
        let runtime = new_runtime()?;
        let ctx = &runtime.ctx;
        let fixed = ctx
            .kernels
            .fixed_sm89_f32_n64_copyplan
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "CopyPlan unavailable: {:?}",
                    ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                )
            })?;
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
        resources(fixed, FIXED, 128, 32768)?;
        resources(
            &ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            TRANSPOSE,
            512,
            4224,
        )?;
        resources(&runtime.epilogue, EPILOGUE, 256, 0)?;
        println!(
            "{}",
            json!({"schema":"MambaBiF32TnCopyPlanArtifactsV1","artifacts":format!("{:?}",ctx.kernels.artifact_set_identity()),"epilogue_source_sha":runtime.source_sha,"candidate_nodes":[TRANSPOSE,FIXED,EPILOGUE],"physical_nn_shape":[768,3072,2048],"fast_compute":"CUBLAS_COMPUTE_32F_FAST_TF32","algorithm":"CUBLAS_GEMM_DEFAULT","alpha":1,"beta":1,"promotion":false})
        );

        for (cell, exceptional) in [
            (
                Cell {
                    batch: 67,
                    k_out: 68,
                    n_out: 69,
                },
                false,
            ),
            (
                Cell {
                    batch: 67,
                    k_out: 68,
                    n_out: 69,
                },
                true,
            ),
            (
                Cell {
                    batch: 0,
                    k_out: 68,
                    n_out: 69,
                },
                false,
            ),
        ] {
            check_exact_bits(&runtime, &mut Fixture::new(ctx, cell, 1.0, exceptional)?)?;
        }
        check_nonunit_bits(
            &runtime,
            &mut Fixture::new(
                ctx,
                Cell {
                    batch: 67,
                    k_out: 68,
                    n_out: 69,
                },
                -0.75,
                false,
            )?,
        )?;

        let mut fixture = Fixture::new(ctx, D768_IN, 1.0, false)?;
        let (_, exact_repeated) = check_exact_bits(&runtime, &mut fixture)?;
        let candidate_graph = capture(&runtime, &mut fixture, Arm::Candidate)?;
        graph_identity(&candidate_graph, &fixture, Arm::Candidate)?;
        for comparator in [Arm::Auto, Arm::Fast] {
            let comparator_graph = capture(&runtime, &mut fixture, comparator)?;
            graph_identity(&comparator_graph, &fixture, comparator)?;
            let comparator_repeated = if comparator == Arm::Fast {
                fast_goldens(&runtime, &mut fixture, &comparator_graph)?.1
            } else {
                exact_repeated.clone()
            };
            quiet.require_cohort("f32-tn-copyplan/timed")?;
            let mut strata = Vec::new();
            for graph_path in [false, true] {
                for candidate_endpoints in [true, false] {
                    let arms = if candidate_endpoints {
                        [Arm::Candidate, comparator, comparator, Arm::Candidate]
                    } else {
                        [comparator, Arm::Candidate, Arm::Candidate, comparator]
                    };
                    let mut raw = Vec::new();
                    for bracket in 0..9 {
                        let mut observations = [0.0; 4];
                        for (i, arm) in arms.into_iter().enumerate() {
                            let (graph, golden) = if arm == Arm::Candidate {
                                (&candidate_graph, &exact_repeated)
                            } else {
                                (&comparator_graph, &comparator_repeated)
                            };
                            observations[i] =
                                measure(&runtime, &mut fixture, arm, graph, graph_path, golden)?;
                        }
                        if bracket >= 2 {
                            raw.push(observations);
                        }
                    }
                    let ratios = raw
                        .iter()
                        .map(|x| ratio(*x, candidate_endpoints))
                        .collect::<Result<Vec<_>, _>>()?;
                    let p50 = quantile(&ratios, 0.5);
                    let p95 = quantile(&ratios, 0.95);
                    strata.push([p50, p95]);
                    println!(
                        "{}",
                        json!({"schema":"MambaBiF32TnCopyPlanScreenV1","shape":[D768_IN.batch,D768_IN.k_out,D768_IN.n_out],"physical_nn_shape":[768,3072,2048],"comparator":format!("{comparator:?}"),"path":if graph_path {"graph"} else {"eager"},"order":if candidate_endpoints {"ABBA"} else {"BAAB"},"observation_arms":arms.map(|x|format!("{x:?}")),"windows":7,"logical_gemms_per_observation":OPS,"candidate_timed_nodes":[TRANSPOSE,FIXED,EPILOGUE],"raw_observations_us":raw,"ratio_direction":"candidate_over_comparator","ratio_p50":p50,"ratio_p95":p95})
                    );
                }
            }
            let retain = strata.iter().all(|x| x[0] < 0.99 && x[1] < 0.99);
            println!(
                "{}",
                json!({"schema":"MambaBiF32TnCopyPlanDecisionV1","shape":[D768_IN.batch,D768_IN.k_out,D768_IN.n_out],"comparator":format!("{comparator:?}"),"strata":strata,"retain":retain,"decision":if retain {"shortlist"} else {"stop_no_retry"},"promotion":false})
            );
        }
        quiet.verify_post_cohort("f32-tn-copyplan/post")?;
        Ok(())
    }
}
