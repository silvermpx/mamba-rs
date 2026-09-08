//! Discovery only: exact F32 TN Split-M CopyPlan reuse for d768-out and Prism.
//! No dispatcher admission and no production kernel changes.

#[path = "support/triad_tn_transpose_n96_source.rs"]
mod padded_transpose_source;
#[path = "support/triad_f32_tn_copyplan_raw_store_source.rs"]
mod raw_store_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    batch: usize,
    k_out: usize,
    n_out: usize,
    padded_transpose: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitPlan {
    m_chunk: usize,
    chunks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Target {
    cell: Cell,
    plan: SplitPlan,
}

const TARGETS: [Target; 2] = [
    Target {
        cell: Cell {
            name: "d768_out",
            batch: 2048,
            k_out: 1536,
            n_out: 768,
            padded_transpose: false,
        },
        plan: SplitPlan {
            m_chunk: 512,
            chunks: 4,
        },
    },
    Target {
        cell: Cell {
            name: "prism",
            batch: 4621,
            k_out: 384,
            n_out: 1928,
            padded_transpose: true,
        },
        plan: SplitPlan {
            m_chunk: 784,
            chunks: 6,
        },
    },
];

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chunk {
    begin: usize,
    len: usize,
    partial_offset: usize,
}

impl Cell {
    fn chunks(self, plan: SplitPlan) -> Result<Vec<Chunk>, String> {
        if self.batch == 0 {
            return Ok(Vec::new());
        }
        let actual = self.batch.div_ceil(plan.m_chunk);
        if actual != plan.chunks {
            return Err(format!(
                "{} plan declares {} chunks but covers {actual}",
                self.name, plan.chunks
            ));
        }
        Ok((0..plan.chunks)
            .map(|index| {
                let begin = index * plan.m_chunk;
                Chunk {
                    begin,
                    len: (self.batch - begin).min(plan.m_chunk),
                    partial_offset: index * self.k_out * self.n_out,
                }
            })
            .collect())
    }

    fn fixed_params(self, chunk: Chunk) -> FixedParams {
        FixedParams {
            alpha: 1.0,
            beta: 0.0,
            m: self.k_out as i32,
            n: self.n_out as i32,
            k: chunk.len as i32,
            lda: self.transpose_stride() as i32,
            ldb: self.n_out as i32,
            ldc: self.n_out as i32,
        }
    }

    fn transpose_grid(self) -> (u32, u32, u32) {
        (
            self.k_out.div_ceil(32) as u32,
            self.transpose_stride().div_ceil(32) as u32,
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

    fn partial_grid(self, plan: SplitPlan) -> (u32, u32, u32) {
        (
            (self.k_out.div_ceil(128) * self.n_out.div_ceil(128)) as u32,
            1,
            plan.chunks as u32,
        )
    }

    fn reducer_grid(self) -> (u32, u32, u32) {
        ((self.k_out * self.n_out).div_ceil(256) as u32, 1, 1)
    }

    fn transposed_a_index(self, batch_row: usize, k_column: usize) -> usize {
        k_column * self.transpose_stride() + batch_row
    }

    fn transpose_stride(self) -> usize {
        if self.padded_transpose {
            padded_transpose_source::padded_stride(self.batch)
                .expect("static sibling cell transpose stride")
        } else {
            self.batch
        }
    }

    fn transpose_symbol(self) -> &'static str {
        if self.padded_transpose {
            padded_transpose_source::TRANSPOSE_SYMBOL
        } else {
            "gemm_bi_transpose_f32_32x16_d768_v1"
        }
    }

    fn transpose_block(self) -> (u32, u32, u32) {
        if self.padded_transpose {
            (32, 8, 1)
        } else {
            (32, 16, 1)
        }
    }
}

fn ratio(raw: [f64; 4], candidate_endpoints: bool) -> Result<f64, String> {
    if raw.iter().any(|value| !value.is_finite() || *value <= 0.0) {
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
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1)]
}

#[test]
fn sibling_splitm_chunks_match_committed_ada_policy() {
    assert_eq!(
        (size_of::<FixedParams>(), align_of::<FixedParams>()),
        (32, 4)
    );
    let d768 = TARGETS[0];
    assert_eq!(
        d768.cell.chunks(d768.plan).unwrap(),
        [
            Chunk {
                begin: 0,
                len: 512,
                partial_offset: 0
            },
            Chunk {
                begin: 512,
                len: 512,
                partial_offset: 1536 * 768
            },
            Chunk {
                begin: 1024,
                len: 512,
                partial_offset: 2 * 1536 * 768
            },
            Chunk {
                begin: 1536,
                len: 512,
                partial_offset: 3 * 1536 * 768
            },
        ]
    );
    assert_eq!(d768.cell.partial_grid(d768.plan), (72, 1, 4));
    assert_eq!(d768.cell.fixed_grid(), (288, 1, 1));
    assert_eq!(d768.plan.chunks + 2, 6);

    let prism = TARGETS[1];
    let chunks = prism.cell.chunks(prism.plan).unwrap();
    assert_eq!(chunks.len(), 6);
    assert_eq!(
        chunks[0],
        Chunk {
            begin: 0,
            len: 784,
            partial_offset: 0
        }
    );
    assert_eq!(
        chunks[5],
        Chunk {
            begin: 3920,
            len: 701,
            partial_offset: 5 * 384 * 1928
        }
    );
    assert_eq!(prism.cell.partial_grid(prism.plan), (48, 1, 6));
    assert_eq!(prism.cell.fixed_grid(), (186, 1, 1));
    assert_eq!(prism.cell.transpose_grid(), (12, 145, 1));
    assert_eq!(prism.cell.reducer_grid(), (2892, 1, 1));
    assert_eq!(prism.plan.chunks + 2, 8);
    assert_eq!(prism.cell.transpose_stride(), 4624);
    assert_eq!(prism.cell.transposed_a_index(4620, 383), 383 * 4624 + 4620);
    assert_eq!(
        prism.cell.transpose_symbol(),
        padded_transpose_source::TRANSPOSE_SYMBOL
    );
    assert_eq!(prism.cell.transpose_block(), (32, 8, 1));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.len.next_multiple_of(32) - chunk.len)
            .collect::<Vec<_>>(),
        [16, 16, 16, 16, 16, 3]
    );
}

#[test]
fn sibling_chunk_pointer_offsets_and_fixed_abi_are_exact() {
    for target in TARGETS {
        for chunk in target.cell.chunks(target.plan).unwrap() {
            let params = target.cell.fixed_params(chunk);
            assert_eq!(
                (params.alpha.to_bits(), params.beta.to_bits()),
                (1.0_f32.to_bits(), 0)
            );
            assert_eq!(
                (params.m, params.n, params.k),
                (
                    target.cell.k_out as i32,
                    target.cell.n_out as i32,
                    chunk.len as i32
                )
            );
            assert_eq!(
                (params.lda, params.ldb, params.ldc),
                (
                    target.cell.transpose_stride() as i32,
                    target.cell.n_out as i32,
                    target.cell.n_out as i32
                )
            );
            assert_eq!(
                chunk.begin * target.cell.n_out,
                chunk.begin * params.ldb as usize
            );
        }
    }
}

#[test]
fn sibling_pairing_math_is_order_unambiguous() {
    assert_eq!(ratio([8., 10., 10., 8.], true).unwrap(), 0.8);
    assert_eq!(ratio([10., 8., 8., 10.], false).unwrap(), 0.8);
    assert!(ratio([0., 1., 1., 1.], true).is_err());
    assert_eq!(quantile(&[7., 6., 5., 4., 3., 2., 1.], 0.5), 4.);
    assert_eq!(quantile(&[7., 6., 5., 4., 3., 2., 1.], 0.95), 7.);
}

#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;
#[cfg(feature = "cuda")]
mod sibling_cuda_suite {
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

    const FIXED: &str = raw_store_source::PRODUCTION_SYMBOL;
    const RAW_FIXED: &str = raw_store_source::RAW_SYMBOL;
    const TRANSPOSE: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
    const PADDED_TRANSPOSE: &str = padded_transpose_source::TRANSPOSE_SYMBOL;
    const PARTIAL: &str = "gemm_bi_tn_splitm_partial_aligned";
    const REDUCER: &str = "gemm_bi_splitm_reduce";
    const ZERO: &str = "gemm_bi_tn_zero_reduction_v1";
    const GUARD: usize = 64;
    const GUARD_BITS: u32 = 0x7fc0_3189;
    const OPS: usize = 20;

    unsafe impl DeviceRepr for FixedParams {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(C)]
    struct PaddedTransposeParams {
        rows: i32,
        columns: i32,
        output_stride: i32,
    }

    unsafe impl DeviceRepr for PaddedTransposeParams {}

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct ZeroParams {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for ZeroParams {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CandidateFixed {
        Production,
        RawStore,
    }

    #[derive(Debug)]
    enum RawPartialError {
        Mismatch(String),
        Harness(String),
    }

    impl std::fmt::Display for RawPartialError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Mismatch(message) | Self::Harness(message) => formatter.write_str(message),
            }
        }
    }

    impl CandidateFixed {
        fn symbol(self) -> &'static str {
            match self {
                Self::Production => FIXED,
                Self::RawStore => RAW_FIXED,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Reference,
        Auto,
        Fast,
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

        fn ptr(&self) -> u64 {
            self.gpu.cached_ptr()
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.gpu.upload(&ctx.stream, &self.seed)
        }

        fn bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let data = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("{} download sync: {error:?}", self.label))?;
            if data[self.len..]
                .iter()
                .any(|value| value.to_bits() != GUARD_BITS)
            {
                return Err(format!("{} trailing guard changed", self.label));
            }
            Ok(data[..self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
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
        plan: SplitPlan,
        alpha: f32,
        output: Buffer,
        x: Buffer,
        dy: Buffer,
        transposed: Buffer,
        candidate_partial: Buffer,
        reference_partial: Buffer,
    }

    impl Fixture {
        fn new(
            ctx: &GpuCtx,
            cell: Cell,
            plan: SplitPlan,
            alpha: f32,
            exceptional: bool,
        ) -> Result<Self, String> {
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
                    0xffc5_4321,
                    0xffa5_4321,
                ];
                for (case, bits) in cases.into_iter().enumerate() {
                    for row in cell
                        .chunks(plan)?
                        .into_iter()
                        .map(|chunk| chunk.begin + case)
                    {
                        if row < cell.batch {
                            x[row * cell.k_out] = f32::from_bits(bits);
                            dy[row * cell.n_out] = f32::from_bits(bits.rotate_left(7));
                        }
                    }
                    if case < output.len() {
                        output[case] = f32::from_bits(bits);
                    }
                }
            }
            let partial_len = if cell.batch == 0 {
                0
            } else {
                plan.chunks * cell.k_out * cell.n_out
            };
            Ok(Self {
                cell,
                plan,
                alpha,
                output: Buffer::new(ctx, output, "output")?,
                x: Buffer::new(ctx, x, "x")?,
                dy: Buffer::new(ctx, dy, "dy")?,
                transposed: Buffer::new(
                    ctx,
                    vec![f32::from_bits(0x7fc0_aaaa); cell.transpose_stride() * cell.k_out],
                    "transposed_x",
                )?,
                candidate_partial: Buffer::new(
                    ctx,
                    vec![f32::from_bits(0x7fc0_bbbb); partial_len],
                    "candidate_partial",
                )?,
                reference_partial: Buffer::new(
                    ctx,
                    vec![f32::from_bits(0x7fc0_cccc); partial_len],
                    "reference_partial",
                )?,
            })
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.reset(ctx)?;
            self.transposed.reset(ctx)?;
            self.candidate_partial.reset(ctx)?;
            self.reference_partial.reset(ctx)
        }

        fn validate_inputs(&self, ctx: &GpuCtx) -> Result<(), String> {
            self.x.unchanged(ctx)?;
            self.dy.unchanged(ctx)
        }

        fn validate_output(&self, ctx: &GpuCtx, golden: &[u32]) -> Result<(), String> {
            let actual = self.output.bits(ctx)?;
            if let Some(index) = actual.iter().zip(golden).position(|(a, b)| a != b) {
                return Err(format!(
                    "exact TN mismatch {index}: {:08x} != {:08x}",
                    actual[index], golden[index]
                ));
            }
            self.validate_inputs(ctx)
        }

        fn validate_transpose(&self, ctx: &GpuCtx) -> Result<(), String> {
            let actual = self.transposed.bits(ctx)?;
            for row in 0..self.cell.batch {
                for column in 0..self.cell.k_out {
                    let destination = self.cell.transposed_a_index(row, column);
                    let source = row * self.cell.k_out + column;
                    if actual[destination] != self.x.seed[source].to_bits() {
                        return Err(format!("raw X transpose changed bits at ({row},{column})"));
                    }
                }
            }
            for column in 0..self.cell.k_out {
                for row in self.cell.batch..self.cell.transpose_stride() {
                    let destination = self.cell.transposed_a_index(row, column);
                    if actual[destination] != 0 {
                        return Err(format!(
                            "raw X transpose padding is nonzero at ({row},{column})"
                        ));
                    }
                }
            }
            Ok(())
        }
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _raw_module: Arc<CudaModule>,
        raw_fixed: CudaFunction,
        padded_transpose: CudaFunction,
        raw_source_sha: String,
        padded_transpose_source_sha: String,
    }

    fn config(grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared,
        }
    }

    fn new_runtime() -> Result<Runtime, String> {
        if size_of::<PaddedTransposeParams>() != 12 || align_of::<PaddedTransposeParams>() != 4 {
            return Err("padded transpose parameter ABI changed".into());
        }
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
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
        let raw_source = raw_store_source::compose_source()?;
        let padded_source = padded_transpose_source::transpose_source();
        let raw_source_sha = format!("{:x}", Sha256::digest(raw_source.as_bytes()));
        let padded_transpose_source_sha = format!("{:x}", Sha256::digest(padded_source.as_bytes()));
        let source = format!("{raw_source}\n{padded_source}");
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec!["--fmad=true".into(), "-DNDEBUG".into()],
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("compile raw-store CopyPlan: {error:?}"))?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx.to_src()))
            .map_err(|error| format!("load raw-store CopyPlan module: {error:?}"))?;
        let raw_fixed = module
            .load_function(RAW_FIXED)
            .map_err(|error| format!("load {RAW_FIXED}: {error:?}"))?;
        let padded_transpose = module
            .load_function(PADDED_TRANSPOSE)
            .map_err(|error| format!("load {PADDED_TRANSPOSE}: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            ctx,
            _raw_module: module,
            raw_fixed,
            padded_transpose,
            raw_source_sha,
            padded_transpose_source_sha,
        })
    }

    fn fixed_function(
        runtime: &Runtime,
        candidate: CandidateFixed,
    ) -> Result<&CudaFunction, String> {
        match candidate {
            CandidateFixed::Production => runtime
                .ctx
                .kernels
                .fixed_sm89_f32_n64_copyplan
                .as_ref()
                .ok_or_else(|| {
                    format!(
                        "CopyPlan unavailable: {:?}",
                        runtime.ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                    )
                }),
            CandidateFixed::RawStore => Ok(&runtime.raw_fixed),
        }
    }

    fn launch_transpose(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let output = fixture.transposed.ptr();
        let input = fixture.x.ptr();
        let rows = fixture.cell.batch as i32;
        let columns = fixture.cell.k_out as i32;
        if fixture.cell.padded_transpose {
            let params = PaddedTransposeParams {
                rows,
                columns,
                output_stride: fixture.cell.transpose_stride() as i32,
            };
            let mut builder = runtime.ctx.stream.launch_builder(&runtime.padded_transpose);
            builder.arg(&input);
            builder.arg(&output);
            builder.arg(&params);
            unsafe {
                builder.launch(config(
                    fixture.cell.transpose_grid(),
                    fixture.cell.transpose_block(),
                    0,
                ))
            }
            .map(|_| ())
            .map_err(|error| format!("padded transpose X: {error:?}"))
        } else {
            let mut builder = runtime
                .ctx
                .stream
                .launch_builder(&runtime.ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1);
            builder.arg(&output);
            builder.arg(&input);
            builder.arg(&rows);
            builder.arg(&columns);
            unsafe {
                builder.launch(config(
                    fixture.cell.transpose_grid(),
                    fixture.cell.transpose_block(),
                    0,
                ))
            }
            .map(|_| ())
            .map_err(|error| format!("transpose X: {error:?}"))
        }
    }

    fn launch_candidate_partials(
        runtime: &Runtime,
        fixture: &Fixture,
        candidate: CandidateFixed,
    ) -> Result<(), String> {
        if fixture.cell.batch == 0 {
            return Err("zero reduction has no partial launches".into());
        }
        launch_transpose(runtime, fixture)?;
        let fixed = fixed_function(runtime, candidate)?;
        let transposed = fixture.transposed.ptr();
        let dy = fixture.dy.ptr();
        let partial = fixture.candidate_partial.ptr();
        let null_bias = 0_u64;
        for chunk in fixture.cell.chunks(fixture.plan)? {
            let a = transposed + (chunk.begin * size_of::<f32>()) as u64;
            let b = dy + (chunk.begin * fixture.cell.n_out * size_of::<f32>()) as u64;
            let c = partial + (chunk.partial_offset * size_of::<f32>()) as u64;
            let params = fixture.cell.fixed_params(chunk);
            let mut builder = runtime.ctx.stream.launch_builder(fixed);
            builder.arg(&c);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&null_bias);
            builder.arg(&params);
            unsafe { builder.launch(config(fixture.cell.fixed_grid(), (128, 1, 1), 0)) }.map_err(
                |error| format!("{} chunk {}: {error:?}", candidate.symbol(), chunk.begin),
            )?;
        }
        Ok(())
    }

    fn launch_reference_partials(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        if fixture.cell.batch == 0 {
            return Err("zero reduction has no reference partial".into());
        }
        let partial = fixture.reference_partial.ptr();
        let x = fixture.x.ptr();
        let dy = fixture.dy.ptr();
        let m = fixture.cell.batch as i32;
        let k = fixture.cell.k_out as i32;
        let n = fixture.cell.n_out as i32;
        let m_chunk = fixture.plan.m_chunk as i32;
        let mut builder = runtime
            .ctx
            .stream
            .launch_builder(&runtime.ctx.kernels.gemm_bi_tn_splitm_partial_aligned);
        builder.arg(&partial);
        builder.arg(&x);
        builder.arg(&dy);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&m_chunk);
        unsafe {
            builder.launch(config(
                fixture.cell.partial_grid(fixture.plan),
                (256, 1, 1),
                0,
            ))
        }
        .map(|_| ())
        .map_err(|error| format!("launch raw {PARTIAL}: {error:?}"))
    }

    fn launch_reducer(runtime: &Runtime, fixture: &Fixture, partial: u64) -> Result<(), String> {
        let output = fixture.output.ptr();
        let k = fixture.cell.k_out as i32;
        let n = fixture.cell.n_out as i32;
        let chunks = fixture.plan.chunks as i32;
        let mut builder = runtime
            .ctx
            .stream
            .launch_builder(&runtime.ctx.kernels.gemm_bi_splitm_reduce);
        builder.arg(&output);
        builder.arg(&partial);
        builder.arg(&fixture.alpha);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&chunks);
        unsafe { builder.launch(config(fixture.cell.reducer_grid(), (256, 1, 1), 0)) }
            .map(|_| ())
            .map_err(|error| format!("launch {REDUCER}: {error:?}"))
    }

    fn launch_zero(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let output = fixture.output.ptr();
        let null_input = 0_u64;
        let null_bias = 0_u64;
        let params = ZeroParams {
            alpha: fixture.alpha,
            beta: 1.0,
            m: 0,
            k: fixture.cell.k_out as i32,
            n: fixture.cell.n_out as i32,
            lda: fixture.cell.k_out as i32,
            ldb: fixture.cell.n_out as i32,
            ldc: fixture.cell.n_out as i32,
        };
        let mut builder = runtime
            .ctx
            .stream
            .launch_builder(&runtime.ctx.kernels.gemm_bi_tn_zero_reduction);
        builder.arg(&output);
        builder.arg(&null_input);
        builder.arg(&null_input);
        builder.arg(&null_bias);
        builder.arg(&params);
        unsafe { builder.launch(config(fixture.cell.reducer_grid(), (256, 1, 1), 0)) }
            .map(|_| ())
            .map_err(|error| format!("launch {ZERO}: {error:?}"))
    }

    fn launch(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
        candidate: CandidateFixed,
    ) -> Result<(), String> {
        if fixture.cell.batch == 0 {
            return match arm {
                Arm::Candidate | Arm::Reference => launch_zero(runtime, fixture),
                Arm::Auto => gpu_gemm_bi_backward_dw_grad(
                    &runtime.ctx,
                    &GradSlice::from_raw(
                        fixture.output.ptr(),
                        fixture.cell.k_out * fixture.cell.n_out,
                    ),
                    &fixture.dy.gpu,
                    &fixture.x.gpu,
                    0,
                    fixture.cell.k_out,
                    fixture.cell.n_out,
                ),
                Arm::Fast => Err("Fast comparator does not admit K=0".into()),
            };
        }
        match arm {
            Arm::Candidate => {
                launch_candidate_partials(runtime, fixture, candidate)?;
                launch_reducer(runtime, fixture, fixture.candidate_partial.ptr())
            }
            Arm::Reference => {
                launch_reference_partials(runtime, fixture)?;
                launch_reducer(runtime, fixture, fixture.reference_partial.ptr())
            }
            Arm::Auto => gpu_gemm_bi_backward_dw_grad(
                &runtime.ctx,
                &GradSlice::from_raw(
                    fixture.output.ptr(),
                    fixture.cell.k_out * fixture.cell.n_out,
                ),
                &fixture.dy.gpu,
                &fixture.x.gpu,
                fixture.cell.batch,
                fixture.cell.k_out,
                fixture.cell.n_out,
            ),
            Arm::Fast => {
                let dtype = WeightDtype::F32.cuda_data_type();
                let alpha = 1.0_f32;
                let beta = 1.0_f32;
                unsafe {
                    blas_result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        blas::cublasOperation_t::CUBLAS_OP_T,
                        fixture.cell.n_out as i32,
                        fixture.cell.k_out as i32,
                        fixture.cell.batch as i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        fixture.dy.ptr() as *const c_void,
                        dtype,
                        fixture.cell.n_out as i32,
                        fixture.x.ptr() as *const c_void,
                        dtype,
                        fixture.cell.k_out as i32,
                        (&beta as *const f32).cast::<c_void>(),
                        fixture.output.ptr() as *mut c_void,
                        dtype,
                        fixture.cell.n_out as i32,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|error| format!("explicit cuBLAS Fast TN: {error:?}"))
            }
        }
    }

    fn compare_raw_partials(
        runtime: &Runtime,
        fixture: &mut Fixture,
        candidate: CandidateFixed,
        corpus: &str,
    ) -> Result<(), RawPartialError> {
        let harness = RawPartialError::Harness;
        fixture.reset(&runtime.ctx).map_err(&harness)?;
        launch_reference_partials(runtime, fixture).map_err(&harness)?;
        launch_candidate_partials(runtime, fixture, candidate).map_err(&harness)?;
        let reference = fixture
            .reference_partial
            .bits(&runtime.ctx)
            .map_err(&harness)?;
        let actual = fixture
            .candidate_partial
            .bits(&runtime.ctx)
            .map_err(&harness)?;
        let stride = fixture.cell.k_out * fixture.cell.n_out;
        for (index, chunk) in fixture
            .cell
            .chunks(fixture.plan)
            .map_err(&harness)?
            .into_iter()
            .enumerate()
        {
            let begin = index * stride;
            let end = begin + stride;
            if let Some(relative) = actual[begin..end]
                .iter()
                .zip(&reference[begin..end])
                .position(|(a, b)| a != b)
            {
                let word = begin + relative;
                return Err(RawPartialError::Mismatch(format!(
                    "{corpus} {} raw partial fc={index} m=[{},{}), word={relative}: {:08x} != {:08x}",
                    candidate.symbol(),
                    chunk.begin,
                    chunk.begin + chunk.len,
                    actual[word],
                    reference[word]
                )));
            }
            println!(
                "{}",
                json!({
                    "schema":"MambaBiF32TnCopyPlanRawPartialV2",
                    "corpus":corpus,
                    "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                    "candidate":candidate.symbol(),
                    "fc":index,
                    "m_begin":chunk.begin,
                    "m_end":chunk.begin+chunk.len,
                    "words":stride,
                    "oracle":PARTIAL,
                    "exact_bits":true
                })
            );
        }
        fixture.validate_transpose(&runtime.ctx).map_err(&harness)?;
        fixture.validate_inputs(&runtime.ctx).map_err(harness)
    }

    fn select_candidate(runtime: &Runtime, target: Target) -> Result<CandidateFixed, String> {
        let mut finite = Fixture::new(&runtime.ctx, target.cell, target.plan, 1.0, false)?;
        compare_raw_partials(
            runtime,
            &mut finite,
            CandidateFixed::Production,
            "full_mantissa",
        )
        .map_err(|error| error.to_string())?;
        let mut exceptional = Fixture::new(&runtime.ctx, target.cell, target.plan, 1.0, true)?;
        match compare_raw_partials(
            runtime,
            &mut exceptional,
            CandidateFixed::Production,
            "exceptional_payload",
        ) {
            Ok(()) => Ok(CandidateFixed::Production),
            Err(RawPartialError::Mismatch(production_error)) => {
                println!(
                    "{}",
                    json!({
                        "schema":"MambaBiF32TnCopyPlanRawStoreFallbackV1",
                        "production_error":production_error,
                        "fallback":RAW_FIXED,
                        "reason":"alpha1_store_changed_exceptional_raw_bits"
                    })
                );
                compare_raw_partials(
                    runtime,
                    &mut finite,
                    CandidateFixed::RawStore,
                    "full_mantissa",
                )
                .map_err(|error| error.to_string())?;
                compare_raw_partials(
                    runtime,
                    &mut exceptional,
                    CandidateFixed::RawStore,
                    "exceptional_payload",
                )
                .map_err(|error| error.to_string())?;
                Ok(CandidateFixed::RawStore)
            }
            Err(RawPartialError::Harness(error)) => Err(error),
        }
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    struct ObservedKernel {
        node: sys::CUgraphNode,
        symbol: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        kernel_params: *mut *mut c_void,
    }

    unsafe fn observed_kernels(graph: &CudaGraph) -> Result<Vec<ObservedKernel>, String> {
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "query graph node count",
        )?;
        if count == 0 {
            return Err("captured graph is empty".into());
        }
        let mut nodes = vec![std::ptr::null_mut(); count];
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) },
            "query graph nodes",
        )?;
        nodes
            .into_iter()
            .map(|node| {
                let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                cuda_ok(
                    unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
                    "query graph node type",
                )?;
                if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                    return Err(format!("graph contains non-kernel node {kind:?}"));
                }
                let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
                cuda_ok(
                    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
                    "query graph kernel params",
                )?;
                if params.kernelParams.is_null() {
                    return Err("graph kernel omitted packed parameters".into());
                }
                let mut name = std::ptr::null();
                cuda_ok(
                    unsafe { sys::cuFuncGetName(&mut name, params.func) },
                    "query graph kernel symbol",
                )?;
                if name.is_null() {
                    return Err("graph kernel omitted symbol".into());
                }
                Ok(ObservedKernel {
                    node,
                    symbol: unsafe { CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned(),
                    grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
                    block: (params.blockDimX, params.blockDimY, params.blockDimZ),
                    shared: params.sharedMemBytes,
                    kernel_params: params.kernelParams,
                })
            })
            .collect()
    }

    unsafe fn graph_arg<T: Copy>(node: &ObservedKernel, index: usize) -> Result<T, String> {
        let address = unsafe { *node.kernel_params.add(index) };
        if address.is_null() {
            return Err(format!("{} argument {index} is null storage", node.symbol));
        }
        Ok(unsafe { std::ptr::read_unaligned(address.cast::<T>()) })
    }

    unsafe fn graph_edges(
        graph: &CudaGraph,
    ) -> Result<Vec<(sys::CUgraphNode, sys::CUgraphNode)>, String> {
        let mut count = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph.cu_graph(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut count,
                )
            },
            "query graph edge count",
        )?;
        let mut from = vec![std::ptr::null_mut(); count];
        let mut to = vec![std::ptr::null_mut(); count];
        let mut data: Vec<sys::CUgraphEdgeData> =
            (0..count).map(|_| unsafe { std::mem::zeroed() }).collect();
        if count > 0 {
            cuda_ok(
                unsafe {
                    sys::cuGraphGetEdges_v2(
                        graph.cu_graph(),
                        from.as_mut_ptr(),
                        to.as_mut_ptr(),
                        data.as_mut_ptr(),
                        &mut count,
                    )
                },
                "query graph edges",
            )?;
        }
        for edge in data {
            if edge.from_port != 0
                || edge.to_port != 0
                || edge.type_ != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
                || edge.reserved != [0; 5]
            {
                return Err("graph edge descriptor is not CUDA default".into());
            }
        }
        Ok(from.into_iter().zip(to).collect())
    }

    fn require_launch(
        node: &ObservedKernel,
        symbol: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
    ) -> Result<(), String> {
        if node.symbol != symbol
            || node.grid != grid
            || node.block != block
            || node.shared != shared
        {
            return Err(format!(
                "graph launch drift: {} {:?} {:?} {}",
                node.symbol, node.grid, node.block, node.shared
            ));
        }
        Ok(())
    }

    fn capture(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        candidate: CandidateFixed,
    ) -> Result<CudaGraph, String> {
        launch(runtime, fixture, arm, candidate)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("capture warmup: {error:?}"))?;
        fixture.reset(&runtime.ctx)?;
        unsafe {
            capture_into_graph(&runtime.ctx.stream, || {
                launch(runtime, fixture, arm, candidate)
            })
        }
    }

    unsafe fn require_chain(
        graph: &CudaGraph,
        chain: &[sys::CUgraphNode],
        label: &str,
    ) -> Result<(), String> {
        let edges = unsafe { graph_edges(graph) }?;
        let expected = chain
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();
        if edges.len() != expected.len()
            || expected
                .iter()
                .any(|edge| !edges.iter().any(|actual| actual == edge))
        {
            return Err(format!("{label} graph is not the exact linear chain"));
        }
        Ok(())
    }

    unsafe fn require_auto_splitm_graph(
        graph: &CudaGraph,
        fixture: &Fixture,
    ) -> Result<(), String> {
        let nodes = unsafe { observed_kernels(graph) }?;
        if nodes.len() != 2 {
            return Err(format!(
                "AUTO has {} nodes, expected TnSplitM two",
                nodes.len()
            ));
        }
        let partial = nodes
            .iter()
            .find(|node| node.symbol == PARTIAL)
            .ok_or_else(|| format!("AUTO omitted {PARTIAL}"))?;
        let reducer = nodes
            .iter()
            .find(|node| node.symbol == REDUCER)
            .ok_or_else(|| format!("AUTO omitted {REDUCER}"))?;
        require_launch(
            partial,
            PARTIAL,
            fixture.cell.partial_grid(fixture.plan),
            (256, 1, 1),
            0,
        )?;
        require_launch(
            reducer,
            REDUCER,
            fixture.cell.reducer_grid(),
            (256, 1, 1),
            0,
        )?;
        let partial_ptr = unsafe { graph_arg::<u64>(partial, 0) }?;
        if unsafe { graph_arg::<u64>(partial, 1) }? != fixture.x.ptr()
            || unsafe { graph_arg::<u64>(partial, 2) }? != fixture.dy.ptr()
            || unsafe { graph_arg::<i32>(partial, 3) }? != fixture.cell.batch as i32
            || unsafe { graph_arg::<i32>(partial, 4) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(partial, 5) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(partial, 6) }? != fixture.plan.m_chunk as i32
        {
            return Err("AUTO TnSplitM partial ABI/arguments changed".into());
        }
        if unsafe { graph_arg::<u64>(reducer, 0) }? != fixture.output.ptr()
            || unsafe { graph_arg::<u64>(reducer, 1) }? != partial_ptr
            || unsafe { graph_arg::<f32>(reducer, 2) }?.to_bits() != 1.0_f32.to_bits()
            || unsafe { graph_arg::<i32>(reducer, 3) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(reducer, 4) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(reducer, 5) }? != fixture.plan.chunks as i32
        {
            return Err("AUTO TnSplitM reducer ABI/arguments changed".into());
        }
        unsafe { require_chain(graph, &[partial.node, reducer.node], "AUTO") }?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanAutoPlanV2",
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "plan":{"kind":"TnSplitM","m_chunk":fixture.plan.m_chunk,"chunks":fixture.plan.chunks},
                "nodes":[PARTIAL,REDUCER],
                "edge":[PARTIAL,REDUCER],
                "partial_layout":[fixture.plan.chunks,fixture.cell.k_out,fixture.cell.n_out]
            })
        );
        Ok(())
    }

    unsafe fn require_candidate_graph(
        graph: &CudaGraph,
        fixture: &Fixture,
        candidate: CandidateFixed,
    ) -> Result<(), String> {
        let nodes = unsafe { observed_kernels(graph) }?;
        if fixture.cell.batch == 0 {
            if nodes.len() != 1 {
                return Err(format!("K0 candidate has {} graph nodes", nodes.len()));
            }
            let zero = &nodes[0];
            require_launch(zero, ZERO, fixture.cell.reducer_grid(), (256, 1, 1), 0)?;
            let params = unsafe { graph_arg::<ZeroParams>(zero, 4) }?;
            if unsafe { graph_arg::<u64>(zero, 0) }? != fixture.output.ptr()
                || unsafe { graph_arg::<u64>(zero, 1) }? != 0
                || unsafe { graph_arg::<u64>(zero, 2) }? != 0
                || unsafe { graph_arg::<u64>(zero, 3) }? != 0
                || params.alpha.to_bits() != fixture.alpha.to_bits()
                || params.beta.to_bits() != 1.0_f32.to_bits()
                || (params.m, params.k, params.n)
                    != (0, fixture.cell.k_out as i32, fixture.cell.n_out as i32)
                || (params.lda, params.ldb, params.ldc)
                    != (
                        fixture.cell.k_out as i32,
                        fixture.cell.n_out as i32,
                        fixture.cell.n_out as i32,
                    )
            {
                return Err("K0 candidate ABI/arguments changed".into());
            }
            unsafe { require_chain(graph, &[zero.node], "K0 candidate") }?;
            return Ok(());
        }
        if nodes.len() != fixture.plan.chunks + 2 {
            return Err(format!(
                "candidate has {} nodes, expected {}",
                nodes.len(),
                fixture.plan.chunks + 2
            ));
        }
        let transpose = nodes
            .iter()
            .find(|node| node.symbol == fixture.cell.transpose_symbol())
            .ok_or_else(|| format!("candidate omitted {}", fixture.cell.transpose_symbol()))?;
        require_launch(
            transpose,
            fixture.cell.transpose_symbol(),
            fixture.cell.transpose_grid(),
            fixture.cell.transpose_block(),
            0,
        )?;
        if fixture.cell.padded_transpose {
            let params = unsafe { graph_arg::<PaddedTransposeParams>(transpose, 2) }?;
            if unsafe { graph_arg::<u64>(transpose, 0) }? != fixture.x.ptr()
                || unsafe { graph_arg::<u64>(transpose, 1) }? != fixture.transposed.ptr()
                || params
                    != (PaddedTransposeParams {
                        rows: fixture.cell.batch as i32,
                        columns: fixture.cell.k_out as i32,
                        output_stride: fixture.cell.transpose_stride() as i32,
                    })
            {
                return Err("candidate padded transpose ABI/arguments changed".into());
            }
        } else if unsafe { graph_arg::<u64>(transpose, 0) }? != fixture.transposed.ptr()
            || unsafe { graph_arg::<u64>(transpose, 1) }? != fixture.x.ptr()
            || unsafe { graph_arg::<i32>(transpose, 2) }? != fixture.cell.batch as i32
            || unsafe { graph_arg::<i32>(transpose, 3) }? != fixture.cell.k_out as i32
        {
            return Err("candidate transpose ABI/arguments changed".into());
        }
        let fixed_nodes = nodes
            .iter()
            .filter(|node| node.symbol == candidate.symbol())
            .collect::<Vec<_>>();
        if fixed_nodes.len() != fixture.plan.chunks {
            return Err(format!(
                "candidate has {} {} nodes",
                fixed_nodes.len(),
                candidate.symbol()
            ));
        }
        let mut ordered_fixed: Vec<Option<&ObservedKernel>> = vec![None; fixture.plan.chunks];
        for node in fixed_nodes {
            require_launch(
                node,
                candidate.symbol(),
                fixture.cell.fixed_grid(),
                (128, 1, 1),
                0,
            )?;
            let c = unsafe { graph_arg::<u64>(node, 0) }?;
            let index = fixture
                .cell
                .chunks(fixture.plan)?
                .into_iter()
                .position(|chunk| {
                    c == fixture.candidate_partial.ptr()
                        + (chunk.partial_offset * size_of::<f32>()) as u64
                })
                .ok_or_else(|| {
                    "candidate Fixed output is outside raw partial tensor".to_string()
                })?;
            if ordered_fixed[index].replace(node).is_some() {
                return Err(format!("duplicate candidate Fixed chunk {index}"));
            }
            let chunk = fixture.cell.chunks(fixture.plan)?[index];
            let expected_a = fixture.transposed.ptr() + (chunk.begin * size_of::<f32>()) as u64;
            let expected_b =
                fixture.dy.ptr() + (chunk.begin * fixture.cell.n_out * size_of::<f32>()) as u64;
            if unsafe { graph_arg::<u64>(node, 1) }? != expected_a
                || unsafe { graph_arg::<u64>(node, 2) }? != expected_b
                || unsafe { graph_arg::<u64>(node, 3) }? != 0
                || unsafe { graph_arg::<FixedParams>(node, 4) }? != fixture.cell.fixed_params(chunk)
            {
                return Err(format!(
                    "candidate Fixed chunk {index} ABI/arguments changed"
                ));
            }
        }
        let reducer = nodes
            .iter()
            .find(|node| node.symbol == REDUCER)
            .ok_or_else(|| format!("candidate omitted {REDUCER}"))?;
        require_launch(
            reducer,
            REDUCER,
            fixture.cell.reducer_grid(),
            (256, 1, 1),
            0,
        )?;
        if unsafe { graph_arg::<u64>(reducer, 0) }? != fixture.output.ptr()
            || unsafe { graph_arg::<u64>(reducer, 1) }? != fixture.candidate_partial.ptr()
            || unsafe { graph_arg::<f32>(reducer, 2) }?.to_bits() != fixture.alpha.to_bits()
            || unsafe { graph_arg::<i32>(reducer, 3) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(reducer, 4) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(reducer, 5) }? != fixture.plan.chunks as i32
        {
            return Err("candidate reducer ABI/arguments changed".into());
        }
        let mut chain = Vec::with_capacity(fixture.plan.chunks + 2);
        chain.push(transpose.node);
        for (index, fixed) in ordered_fixed.into_iter().enumerate() {
            chain.push(
                fixed
                    .ok_or_else(|| format!("missing candidate chunk{index}"))?
                    .node,
            );
        }
        chain.push(reducer.node);
        unsafe { require_chain(graph, &chain, "candidate") }?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanSiblingGraphV1",
                "arm":"Candidate",
                "cell":fixture.cell.name,
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "node_count":chain.len(),
                "fixed_nodes":fixture.plan.chunks,
                "order":"transpose_then_chunks_then_reduce"
            })
        );
        Ok(())
    }

    unsafe fn require_reference_graph(graph: &CudaGraph, fixture: &Fixture) -> Result<(), String> {
        let nodes = unsafe { observed_kernels(graph) }?;
        if fixture.cell.batch == 0 {
            return unsafe { require_candidate_graph(graph, fixture, CandidateFixed::Production) };
        }
        if nodes.len() != 2 {
            return Err(format!("reference has {} nodes", nodes.len()));
        }
        let partial = nodes
            .iter()
            .find(|node| node.symbol == PARTIAL)
            .ok_or_else(|| "reference partial missing".to_string())?;
        let reducer = nodes
            .iter()
            .find(|node| node.symbol == REDUCER)
            .ok_or_else(|| "reference reducer missing".to_string())?;
        require_launch(
            partial,
            PARTIAL,
            fixture.cell.partial_grid(fixture.plan),
            (256, 1, 1),
            0,
        )?;
        require_launch(
            reducer,
            REDUCER,
            fixture.cell.reducer_grid(),
            (256, 1, 1),
            0,
        )?;
        if unsafe { graph_arg::<u64>(partial, 0) }? != fixture.reference_partial.ptr()
            || unsafe { graph_arg::<u64>(reducer, 1) }? != fixture.reference_partial.ptr()
            || unsafe { graph_arg::<f32>(reducer, 2) }?.to_bits() != fixture.alpha.to_bits()
        {
            return Err("reference graph pointers/alpha changed".into());
        }
        unsafe { require_chain(graph, &[partial.node, reducer.node], "reference") }
    }

    unsafe fn require_fast_graph(graph: &CudaGraph) -> Result<(), String> {
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "query Fast graph nodes",
        )?;
        if count == 0 {
            return Err("cuBLAS Fast graph is empty".into());
        }
        println!(
            "{}",
            json!({"schema":"MambaBiF32TnCopyPlanGraphV2","arm":"Fast","node_count":count,"abi":"opaque","timing":"whole_graph"})
        );
        Ok(())
    }

    fn launch_path(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
        candidate: CandidateFixed,
        graph: Option<&CudaGraph>,
    ) -> Result<(), String> {
        if let Some(graph) = graph {
            graph
                .launch()
                .map_err(|error| format!("graph launch: {error:?}"))
        } else {
            launch(runtime, fixture, arm, candidate)
        }
    }

    fn output_after(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        candidate: CandidateFixed,
        graph: Option<&CudaGraph>,
        repeats: usize,
    ) -> Result<Vec<u32>, String> {
        fixture.reset(&runtime.ctx)?;
        for _ in 0..repeats {
            launch_path(runtime, fixture, arm, candidate, graph)?;
        }
        let output = fixture.output.bits(&runtime.ctx)?;
        fixture.validate_inputs(&runtime.ctx)?;
        match arm {
            Arm::Candidate if fixture.cell.batch > 0 => {
                fixture.transposed.bits(&runtime.ctx)?;
                fixture.candidate_partial.bits(&runtime.ctx)?;
            }
            Arm::Reference if fixture.cell.batch > 0 => {
                fixture.reference_partial.bits(&runtime.ctx)?;
            }
            _ => {}
        }
        Ok(output)
    }

    fn check_exact_case(
        runtime: &Runtime,
        fixture: &mut Fixture,
        candidate: CandidateFixed,
        oracle: Arm,
        label: &str,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let candidate_graph = capture(runtime, fixture, Arm::Candidate, candidate)?;
        let oracle_graph = capture(runtime, fixture, oracle, candidate)?;
        unsafe {
            require_candidate_graph(&candidate_graph, fixture, candidate)?;
            match oracle {
                Arm::Auto if fixture.cell.batch > 0 => {
                    require_auto_splitm_graph(&oracle_graph, fixture)?
                }
                Arm::Reference => require_reference_graph(&oracle_graph, fixture)?,
                Arm::Auto => {
                    // The public dW API fixes alpha=beta=1.0.  Checking only
                    // the symbol here previously let a non-unit manual K0
                    // launch masquerade as the AUTO oracle and fail only on
                    // signed zero.  Validate the complete packed ABI too.
                    require_candidate_graph(&oracle_graph, fixture, candidate)?;
                }
                _ => return Err("invalid exact oracle arm".into()),
            }
        }
        let single = output_after(runtime, fixture, oracle, candidate, None, 1)?;
        let repeated = output_after(runtime, fixture, oracle, candidate, None, OPS)?;
        for (arm, graph) in [(oracle, &oracle_graph), (Arm::Candidate, &candidate_graph)] {
            for (path, captured) in [("eager", None), ("graph", Some(graph))] {
                for repeat in 0..2 {
                    let actual = output_after(runtime, fixture, arm, candidate, captured, 1)?;
                    fixture.validate_output(&runtime.ctx, &single)?;
                    if actual != single {
                        return Err(format!(
                            "{label} {arm:?} {path} repeat{repeat} changed bits"
                        ));
                    }
                }
            }
        }
        for (arm, graph) in [(oracle, &oracle_graph), (Arm::Candidate, &candidate_graph)] {
            for (path, captured) in [("eager", None), ("graph", Some(graph))] {
                let actual = output_after(runtime, fixture, arm, candidate, captured, OPS)?;
                if actual != repeated {
                    return Err(format!("{label} {arm:?} {path} 20-op bits changed"));
                }
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanBitsV2",
                "case":label,
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "alpha_bits":fixture.alpha.to_bits(),
                "candidate":candidate.symbol(),
                "oracle":format!("{oracle:?}"),
                "words":single.len(),
                "paths":["eager","graph"],
                "single_repeats":2,
                "accumulation_ops":OPS,
                "exact_bits":true
            })
        );
        Ok((single, repeated))
    }

    fn fast_goldens(
        runtime: &Runtime,
        fixture: &mut Fixture,
        candidate: CandidateFixed,
        graph: &CudaGraph,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let single = output_after(runtime, fixture, Arm::Fast, candidate, None, 1)?;
        if single.is_empty()
            || single.iter().all(|word| word & 0x7fff_ffff == 0)
            || single.iter().any(|word| !f32::from_bits(*word).is_finite())
        {
            return Err("Fast comparator produced invalid output".into());
        }
        for (path, captured) in [("eager", None), ("graph", Some(graph))] {
            for repeat in 0..2 {
                let actual = output_after(runtime, fixture, Arm::Fast, candidate, captured, 1)?;
                if actual != single {
                    return Err(format!("Fast {path} repeat{repeat} changed its own bits"));
                }
            }
        }
        let repeated = output_after(runtime, fixture, Arm::Fast, candidate, None, OPS)?;
        for (path, captured) in [("eager", None), ("graph", Some(graph))] {
            let actual = output_after(runtime, fixture, Arm::Fast, candidate, captured, OPS)?;
            if actual != repeated {
                return Err(format!("Fast {path} 20-op bits changed"));
            }
        }
        Ok((single, repeated))
    }

    fn resources(
        function: &CudaFunction,
        symbol: &str,
        threads: u32,
        static_shared_expected: usize,
    ) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|error| format!("{symbol} regs: {error:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|error| format!("{symbol} local: {error:?}"))?;
        let shared = function
            .shared_size_bytes()
            .map_err(|error| format!("{symbol} shared: {error:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .map_err(|error| format!("{symbol} occupancy: {error:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanResourceV2",
                "symbol":symbol,
                "threads":threads,
                "registers":registers,
                "local_bytes":local,
                "static_shared_bytes":shared,
                "dynamic_shared_bytes":0,
                "occupancy":occupancy
            })
        );
        if local != 0 || shared as usize != static_shared_expected || occupancy == 0 {
            return Err(format!("{symbol} resource gate failed"));
        }
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        candidate: CandidateFixed,
        graph: &CudaGraph,
        graph_path: bool,
        golden: &[u32],
    ) -> Result<f64, String> {
        fixture.reset(&runtime.ctx)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("start event: {error:?}"))?;
        for _ in 0..OPS {
            launch_path(
                runtime,
                fixture,
                arm,
                candidate,
                graph_path.then_some(graph),
            )?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("end event: {error:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("elapsed event: {error:?}"))?,
        ) * 1000.0
            / OPS as f64;
        fixture.validate_output(&runtime.ctx, golden)?;
        if arm == Arm::Candidate {
            fixture.candidate_partial.bits(&runtime.ctx)?;
            fixture.transposed.bits(&runtime.ctx)?;
        }
        if !us.is_finite() || us <= 0.0 {
            return Err("invalid elapsed time".into());
        }
        Ok(us)
    }

    fn validate_fast_handle(ctx: &GpuCtx) -> Result<(), String> {
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
        Ok(())
    }

    fn run_target(
        runtime: &Runtime,
        quiet: &common::gpu_quiet::QuietGpu,
        target: Target,
    ) -> Result<(), String> {
        let mut auto_proof = Fixture::new(&runtime.ctx, target.cell, target.plan, 1.0, false)?;
        let auto_proof_graph = capture(
            runtime,
            &mut auto_proof,
            Arm::Auto,
            CandidateFixed::Production,
        )?;
        unsafe { require_auto_splitm_graph(&auto_proof_graph, &auto_proof) }?;

        let candidate = select_candidate(runtime, target)?;
        let fixed = fixed_function(runtime, candidate)?;
        resources(fixed, candidate.symbol(), 128, 32_768)?;
        let mut candidate_nodes = Vec::with_capacity(target.plan.chunks + 2);
        candidate_nodes.push(target.cell.transpose_symbol());
        candidate_nodes.extend(std::iter::repeat_n(candidate.symbol(), target.plan.chunks));
        candidate_nodes.push(REDUCER);
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanSiblingArtifactsV1",
                "cell":target.cell.name,
                "shape":[target.cell.batch,target.cell.k_out,target.cell.n_out],
                "plan":{"m_chunk":target.plan.m_chunk,"chunks":target.plan.chunks},
                "candidate_fixed":candidate.symbol(),
                "candidate_nodes":candidate_nodes,
                "raw_store_source_sha":runtime.raw_source_sha,
                "padded_transpose_source_sha":target.cell.padded_transpose.then_some(&runtime.padded_transpose_source_sha),
                "promotion":false
            })
        );

        let mut fixture = Fixture::new(&runtime.ctx, target.cell, target.plan, 1.0, false)?;
        let (_, exact_repeated) = check_exact_case(
            runtime,
            &mut fixture,
            candidate,
            Arm::Auto,
            target.cell.name,
        )?;

        let tail_cell = Cell {
            name: "tail",
            batch: target.cell.batch - 1,
            ..target.cell
        };
        let mut tail = Fixture::new(&runtime.ctx, tail_cell, target.plan, 1.0, false)?;
        compare_raw_partials(runtime, &mut tail, candidate, "tail_chunk")
            .map_err(|error| error.to_string())?;
        check_exact_case(
            runtime,
            &mut tail,
            candidate,
            Arm::Reference,
            &format!("{}_tail", target.cell.name),
        )?;

        let probe_cell = Cell {
            name: "probe",
            batch: target.cell.batch,
            k_out: 68,
            n_out: 129,
            padded_transpose: target.cell.padded_transpose,
        };
        let mut exceptional = Fixture::new(&runtime.ctx, probe_cell, target.plan, 1.0, true)?;
        compare_raw_partials(runtime, &mut exceptional, candidate, "exceptional_payload")
            .map_err(|error| error.to_string())?;
        check_exact_case(
            runtime,
            &mut exceptional,
            candidate,
            Arm::Reference,
            &format!("{}_exceptional", target.cell.name),
        )?;

        let mut nonunit = Fixture::new(&runtime.ctx, probe_cell, target.plan, -0.75, false)?;
        check_exact_case(
            runtime,
            &mut nonunit,
            candidate,
            Arm::Reference,
            &format!("{}_nonunit", target.cell.name),
        )?;

        let mut zero = Fixture::new(
            &runtime.ctx,
            Cell {
                name: "k0",
                batch: 0,
                k_out: 68,
                n_out: 129,
                padded_transpose: false,
            },
            target.plan,
            1.0,
            true,
        )?;
        check_exact_case(
            runtime,
            &mut zero,
            candidate,
            Arm::Auto,
            &format!("{}_k0", target.cell.name),
        )?;

        let candidate_graph = capture(runtime, &mut fixture, Arm::Candidate, candidate)?;
        unsafe { require_candidate_graph(&candidate_graph, &fixture, candidate) }?;
        let mut retained_win = None;
        let mut fast_win = None;
        for comparator in [Arm::Auto, Arm::Fast] {
            let comparator_graph = capture(runtime, &mut fixture, comparator, candidate)?;
            unsafe {
                match comparator {
                    Arm::Auto => require_auto_splitm_graph(&comparator_graph, &fixture)?,
                    Arm::Fast => require_fast_graph(&comparator_graph)?,
                    _ => unreachable!(),
                }
            }
            let comparator_repeated = if comparator == Arm::Fast {
                fast_goldens(runtime, &mut fixture, candidate, &comparator_graph)?.1
            } else {
                exact_repeated.clone()
            };
            quiet.require_cohort(&format!(
                "f32-tn-splitm-copyplan-siblings/{}/timed",
                target.cell.name
            ))?;
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
                        for (index, arm) in arms.into_iter().enumerate() {
                            let (graph, golden) = if arm == Arm::Candidate {
                                (&candidate_graph, &exact_repeated)
                            } else {
                                (&comparator_graph, &comparator_repeated)
                            };
                            observations[index] = measure(
                                runtime,
                                &mut fixture,
                                arm,
                                candidate,
                                graph,
                                graph_path,
                                golden,
                            )?;
                        }
                        if bracket >= 2 {
                            raw.push(observations);
                        }
                    }
                    let ratios = raw
                        .iter()
                        .map(|values| ratio(*values, candidate_endpoints))
                        .collect::<Result<Vec<_>, _>>()?;
                    let p50 = quantile(&ratios, 0.5);
                    let p95 = quantile(&ratios, 0.95);
                    strata.push([p50, p95]);
                    println!(
                        "{}",
                        json!({
                            "schema":"MambaBiF32TnCopyPlanSiblingScreenV1",
                            "cell":target.cell.name,
                            "shape":[target.cell.batch,target.cell.k_out,target.cell.n_out],
                            "plan":{"m_chunk":target.plan.m_chunk,"chunks":target.plan.chunks},
                            "comparator":format!("{comparator:?}"),
                            "path":if graph_path {"graph"} else {"eager"},
                            "order":if candidate_endpoints {"ABBA"} else {"BAAB"},
                            "observation_arms":arms.map(|arm|format!("{arm:?}")),
                            "windows":7,
                            "warmup_windows":2,
                            "logical_gemms_per_observation":OPS,
                            "raw_observations_us":raw,
                            "ratio_direction":"candidate_over_comparator",
                            "ratio_p50":p50,
                            "ratio_p95":p95
                        })
                    );
                }
            }
            let strict_win = strata
                .iter()
                .all(|values| values[0] < 0.99 && values[1] < 0.99);
            match comparator {
                Arm::Auto => retained_win = Some(strict_win),
                Arm::Fast => fast_win = Some(strict_win),
                _ => unreachable!(),
            }
            println!(
                "{}",
                json!({
                    "schema":"MambaBiF32TnCopyPlanSiblingDecisionV1",
                    "cell":target.cell.name,
                    "comparator":format!("{comparator:?}"),
                    "strata":strata,
                    "strict_win":strict_win,
                    "promotion":false
                })
            );
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanSiblingFinalDecisionV1",
                "cell":target.cell.name,
                "candidate":candidate.symbol(),
                "retain_against_actual_auto":retained_win.ok_or("missing AUTO decision")?,
                "fast_win":fast_win.ok_or("missing Fast decision")?,
                "promotion":false
            })
        );
        Ok(())
    }

    #[test]
    #[ignore = "Ada CUDA13.2 exact F32 TN d768-out+Prism Split-M CopyPlan siblings"]
    fn ada_f32_tn_splitm_copyplan_siblings_once7() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("performance requires release".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("f32-tn-splitm-copyplan-siblings/pre")?;
        let runtime = new_runtime()?;
        validate_fast_handle(&runtime.ctx)?;
        resources(
            &runtime.ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            TRANSPOSE,
            512,
            4_224,
        )?;
        resources(&runtime.padded_transpose, PADDED_TRANSPOSE, 256, 4_224)?;
        resources(
            &runtime.ctx.kernels.gemm_bi_tn_splitm_partial_aligned,
            PARTIAL,
            256,
            33_792,
        )?;
        resources(&runtime.ctx.kernels.gemm_bi_splitm_reduce, REDUCER, 256, 0)?;
        resources(&runtime.ctx.kernels.gemm_bi_tn_zero_reduction, ZERO, 256, 0)?;
        for target in TARGETS {
            run_target(&runtime, &quiet, target)?;
        }
        quiet.verify_post_cohort("f32-tn-splitm-copyplan-siblings/post")?;
        Ok(())
    }
}
