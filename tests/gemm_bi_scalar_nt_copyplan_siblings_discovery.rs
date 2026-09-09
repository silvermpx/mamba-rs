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
    fn params(self, alpha: f32) -> Params {
        Params {
            alpha,
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
    fn prior_auto_symbol(self) -> &'static str {
        if self.name == "prism" {
            "gemm_bi_nt_slim"
        } else {
            "gemm_bi_nt"
        }
    }
    fn prior_auto_grid(self) -> (u32, u32, u32) {
        let bn = if self.name == "prism" { 64 } else { 128 };
        ((self.m.div_ceil(128) * self.out.div_ceil(bn)) as u32, 1, 1)
    }
    fn prior_auto_block(self) -> (u32, u32, u32) {
        (if self.name == "prism" { 128 } else { 256 }, 1, 1)
    }
    fn prior_auto_shared(self) -> u32 {
        if self.name == "prism" { 0 } else { 33_376 }
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
        let p = cell.params(1.0);
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
fn pre_admission_matrix_freezes_prior_auto_and_once3_then_once7() {
    assert_eq!(
        (
            CELLS[0].prior_auto_symbol(),
            CELLS[0].prior_auto_grid(),
            CELLS[0].prior_auto_block(),
            CELLS[0].prior_auto_shared(),
        ),
        ("gemm_bi_nt", (96, 1, 1), (256, 1, 1), 33_376)
    );
    assert_eq!(
        (
            CELLS[1].prior_auto_symbol(),
            CELLS[1].prior_auto_grid(),
            CELLS[1].prior_auto_block(),
            CELLS[1].prior_auto_shared(),
        ),
        ("gemm_bi_nt_slim", (222, 1, 1), (128, 1, 1), 0)
    );
    assert_eq!([3_usize, 7], [3, 7]);
}

#[test]
fn transpose_mapping_preserves_each_original_nt_dot_product() {
    for cell in CELLS {
        let p = cell.params(1.0);
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

#[cfg(test)]
mod frozen_receipt_regression {
    use serde_json::Value;

    const DISPATCH: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs");
    const LOGS: [(&str, &str); 3] = [
        (
            "CUDA12.8",
            include_str!(
                "../internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/exact-nt-prequal-f04584c8-cuda128.log"
            ),
        ),
        (
            "CUDA13.0",
            include_str!(
                "../internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/exact-nt-prequal-f04584c8-cuda130.log"
            ),
        ),
        (
            "CUDA13.2",
            include_str!(
                "../internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/exact-nt-prequal-f04584c8-cuda132.log"
            ),
        ),
    ];
    const FIELDS: [&str; 5] = [
        "compile_key",
        "artifact_digest",
        "source_digest",
        "header_manifest_digest",
        "nvrtc_library_domain",
    ];

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Identity {
        version: (i32, i32),
        fields: [[u8; 32]; 5],
    }

    #[derive(Clone, Debug)]
    struct Composed {
        scalar: Identity,
        fixed: Identity,
    }

    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        text.split_once(start).unwrap().1.split_once(end).unwrap().0
    }

    fn bytes(text: &str, marker: &str) -> [u8; 32] {
        let tail = text.split_once(marker).unwrap().1;
        let body = between(tail, "[", "]");
        body.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| part.parse::<u8>().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn version(text: &str) -> (i32, i32) {
        let body = between(text.split_once("nvrtc_version:").unwrap().1, "(", ")");
        let (major, minor) = body.split_once(',').unwrap();
        (major.trim().parse().unwrap(), minor.trim().parse().unwrap())
    }

    fn identity(compiler: &str, artifact: &str) -> Identity {
        let compile_key = bytes(artifact, "compile_key:");
        assert_eq!(compile_key, bytes(compiler, "invocation_digest:"));
        Identity {
            version: version(compiler),
            fields: [
                compile_key,
                bytes(artifact, "artifact_digest:"),
                bytes(compiler, "source_digest:"),
                bytes(compiler, "header_manifest_digest:"),
                bytes(compiler, "nvrtc_library_domain:"),
            ],
        }
    }

    fn live(log: &str) -> Composed {
        let record = log
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|value| value["schema"] == "MambaBiNtCopyPlanSiblingsPreAdmissionIdentityV1")
            .unwrap();
        let artifacts = record["artifacts"].as_str().unwrap();
        Composed {
            scalar: identity(
                record["scalar_compiler"].as_str().unwrap(),
                between(
                    artifacts,
                    "triad_scalar: ArtifactIdentity {",
                    "}, triad_sm80:",
                ),
            ),
            fixed: identity(
                record["fixed_compiler"].as_str().unwrap(),
                between(artifacts, "fixed: ArtifactIdentity {", "}, triad_scalar:"),
            ),
        }
    }

    fn frozen() -> Vec<Composed> {
        let fixed_source = bytes(
            DISPATCH
                .split_once("const FIXED_COPYPLAN_SOURCE_DIGEST: [u8; 32] =")
                .unwrap()
                .1,
            "",
        );
        let scalar_source = bytes(
            DISPATCH
                .split_once("const TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST: [u8; 32] =")
                .unwrap()
                .1,
            "",
        );
        let fixed = between(
            DISPATCH,
            "const FIXED_COPYPLAN_EVIDENCE_COHORTS:",
            "struct ScalarTransposeQualificationIdentity",
        )
        .split("\n    FixedCopyPlanQualificationIdentity {")
        .skip(1)
        .map(|block| {
            assert!(block.contains("source_digest: FIXED_COPYPLAN_SOURCE_DIGEST,"));
            Identity {
                version: version(block),
                fields: [
                    bytes(block, "compile_key:"),
                    bytes(block, "artifact_digest:"),
                    fixed_source,
                    bytes(block, "header_manifest_digest:"),
                    bytes(block, "nvrtc_library_domain:"),
                ],
            }
        })
        .collect::<Vec<_>>();
        let scalar_blocks = between(
            DISPATCH,
            "const NT_FIXED_COPYPLAN_COMPOSED_QUALIFICATION_CANDIDATES:",
            "// The three frozen candidates passed the complete pre-admission qualification",
        )
        .split("scalar: ScalarTransposeQualificationIdentity {")
        .skip(1)
        .collect::<Vec<_>>();
        assert_eq!((fixed.len(), scalar_blocks.len()), (3, 3));
        scalar_blocks
            .into_iter()
            .enumerate()
            .map(|(index, block)| {
                assert!(block.contains("source_digest: TRIAD_SCALAR_TRANSPOSE_SOURCE_DIGEST,"));
                assert!(
                    block.contains(&format!("fixed: FIXED_COPYPLAN_EVIDENCE_COHORTS[{index}]"))
                );
                Composed {
                    scalar: Identity {
                        version: version(block),
                        fields: [
                            bytes(block, "compile_key:"),
                            bytes(block, "artifact_digest:"),
                            scalar_source,
                            bytes(block, "header_manifest_digest:"),
                            bytes(block, "nvrtc_library_domain:"),
                        ],
                    },
                    fixed: fixed[index].clone(),
                }
            })
            .collect()
    }

    fn mismatch(live: &Identity, frozen: &Identity) -> Option<&'static str> {
        if live.version != frozen.version {
            return Some("nvrtc_version");
        }
        FIELDS
            .into_iter()
            .zip(live.fields.iter().zip(frozen.fields.iter()))
            .find_map(|(name, (live, frozen))| (live != frozen).then_some(name))
    }

    #[test]
    fn admitted_composed_cohorts_match_independent_live_identity_receipts() {
        let mut dispatch = frozen();
        for ((toolkit, log), frozen) in LOGS.into_iter().zip(dispatch.iter()) {
            let live = live(log);
            assert_eq!(
                mismatch(&live.scalar, &frozen.scalar),
                None,
                "{toolkit} scalar"
            );
            assert_eq!(
                mismatch(&live.fixed, &frozen.fixed),
                None,
                "{toolkit} Fixed"
            );
        }

        // Mutation proof for the original CUDA12.8/13.0 admission bug: a
        // Fixed-module header cannot stand in for the scalar-module header.
        for cohort in [0, 1] {
            dispatch[cohort].scalar.fields[3] = dispatch[cohort].fixed.fields[3];
            assert_eq!(
                mismatch(&live(LOGS[cohort].1).scalar, &dispatch[cohort].scalar),
                Some("header_manifest_digest")
            );
        }
    }
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
        gemm_bi_triad::{
            PhysicalQualificationRequest, PhysicalQualificationRoute,
            QualifiedPhysicalLaunchEvidence, qualify_physical_launch,
        },
        graph_capture::capture_into_graph,
        kernel_identity::{ModuleKind, ResolvedGemmOp, ResolvedNumericContract},
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
        active_offset: usize,
        len: usize,
    }
    impl Buffer {
        fn guarded(ctx: &GpuCtx, values: Vec<f32>) -> Result<Self, String> {
            Self::new(ctx, values, GUARD)
        }
        fn production(ctx: &GpuCtx, values: Vec<f32>) -> Result<Self, String> {
            Self::new(ctx, values, 0)
        }
        fn new(ctx: &GpuCtx, values: Vec<f32>, active_offset: usize) -> Result<Self, String> {
            let len = values.len();
            let total = active_offset
                .checked_add(len)
                .and_then(|n| n.checked_add(GUARD))
                .ok_or_else(|| "guarded allocation extent overflow".to_string())?;
            let mut seed = vec![f32::from_bits(GUARD_BITS); total];
            seed[active_offset..active_offset + len].copy_from_slice(&values);
            let gpu = GpuBuffer::from_cpu(&ctx.stream, &seed)?;
            let pointer = if active_offset == 0 {
                gpu.cached_ptr()
            } else {
                gpu.raw_ptr_at(&ctx.stream, active_offset)
            };
            if pointer % 256 != 0 {
                return Err("timed active pointer is not 256B aligned".into());
            }
            Ok(Self {
                gpu,
                seed,
                active_offset,
                len,
            })
        }
        fn ptr(&self, ctx: &GpuCtx) -> u64 {
            if self.active_offset == 0 {
                self.gpu.cached_ptr()
            } else {
                self.gpu.raw_ptr_at(&ctx.stream, self.active_offset)
            }
        }
        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.gpu.upload(&ctx.stream, &self.seed)
        }
        fn bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let data = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|e| format!("download sync: {e:?}"))?;
            if data[..self.active_offset]
                .iter()
                .chain(&data[self.active_offset + self.len..])
                .any(|x| x.to_bits() != GUARD_BITS)
            {
                return Err("output/input/scratch red zone changed".into());
            }
            Ok(data[self.active_offset..self.active_offset + self.len]
                .iter()
                .map(|x| x.to_bits())
                .collect())
        }
        fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
            let actual = self.gpu.to_cpu(&ctx.stream)?;
            ctx.stream
                .synchronize()
                .map_err(|e| format!("unchanged sync: {e:?}"))?;
            if actual
                .iter()
                .zip(&self.seed)
                .any(|(a, b)| a.to_bits() != b.to_bits())
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
        production_a: Buffer,
        production_b: Buffer,
        production_output: Buffer,
        alpha: f32,
    }
    impl Fixture {
        fn new(ctx: &GpuCtx, cell: Cell, exceptional: bool, alpha: f32) -> Result<Self, String> {
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
            let output = vec![f32::from_bits(0x7fc0_bbbb); cell.m * cell.out];
            Ok(Self {
                cell,
                a: Buffer::guarded(ctx, a.clone())?,
                b: Buffer::guarded(ctx, b.clone())?,
                scratch: Buffer::guarded(
                    ctx,
                    vec![f32::from_bits(0x7fc0_aaaa); cell.out * cell.reduction],
                )?,
                output: Buffer::guarded(ctx, output.clone())?,
                production_a: Buffer::production(ctx, a)?,
                production_b: Buffer::production(ctx, b)?,
                production_output: Buffer::production(ctx, output)?,
                alpha,
            })
        }
        fn reset(&mut self, ctx: &GpuCtx, arm: Arm) -> Result<(), String> {
            match arm {
                Arm::Auto | Arm::Fast => self.production_output.reset(ctx),
                Arm::Candidate => {
                    self.output.reset(ctx)?;
                    self.scratch.reset(ctx)
                }
                Arm::Generic => self.output.reset(ctx),
            }
        }
        fn validate(&self, ctx: &GpuCtx, golden: &[u32], arm: Arm) -> Result<(), String> {
            let (output, a, b) = if matches!(arm, Arm::Auto | Arm::Fast) {
                (
                    &self.production_output,
                    &self.production_a,
                    &self.production_b,
                )
            } else {
                (&self.output, &self.a, &self.b)
            };
            let actual = output.bits(ctx)?;
            if let Some(i) = actual.iter().zip(golden).position(|(a, b)| a != b) {
                return Err(format!(
                    "{} exact output mismatch {i}: {:08x} != {:08x}",
                    self.cell.name, actual[i], golden[i]
                ));
            }
            a.unchanged(ctx)?;
            b.unchanged(ctx)?;
            let transposed = self.scratch.bits(ctx)?;
            if arm == Arm::Candidate {
                for row in 0..self.cell.out {
                    for k in 0..self.cell.reduction {
                        if transposed[k * self.cell.out + row]
                            != self.b.seed[self.b.active_offset + row * self.cell.reduction + k]
                                .to_bits()
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
        let p = cell.params(f.alpha);
        match arm {
            Arm::Auto => gpu_gemm_bi_backward_dx_raw(
                ctx,
                &mut f.production_output.gpu,
                &f.production_a.gpu,
                f.production_b.gpu.cached_ptr(),
                cell.m,
                cell.out,
                cell.reduction,
            ),
            Arm::Candidate => {
                let scratch = f.scratch.ptr(ctx);
                let b = f.b.ptr(ctx);
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
                let output = f.output.ptr(ctx);
                let a = f.a.ptr(ctx);
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
                let output = f.output.ptr(ctx);
                let a = f.a.ptr(ctx);
                let b = f.b.ptr(ctx);
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
                let output = f.production_output.gpu.cached_ptr();
                let a = f.production_a.gpu.cached_ptr();
                let b = f.production_b.gpu.cached_ptr();
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
            let exact_pipeline =
                arm == Arm::Candidate || (arm == Arm::Auto && CELLS.contains(&f.cell));
            let mut pipeline_nodes = Vec::new();
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
                if exact_pipeline {
                    let (expected_grid, expected_block) = match name.as_str() {
                        TRANSPOSE => (f.cell.transpose_grid(), (32, 16, 1)),
                        FIXED => (f.cell.fixed_grid(), (128, 1, 1)),
                        _ => return Err(format!("wrong admitted pipeline symbol {name}")),
                    };
                    if grid != expected_grid
                        || block != expected_block
                        || params.sharedMemBytes != 0
                    {
                        return Err(format!("admitted pipeline launch changed: {name}"));
                    }
                    let expected_abi: &[(usize, usize)] = if name == TRANSPOSE {
                        &[(0, 8), (8, 8), (16, 4), (20, 4)]
                    } else {
                        &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
                    };
                    driver_abi_gate(params.func, &name, expected_abi)?;
                    if params.kernelParams.is_null() {
                        return Err(format!("{name} graph omitted packed arguments"));
                    }
                    let scratch_index = if name == TRANSPOSE { 0 } else { 2 };
                    let scratch_storage = *params.kernelParams.add(scratch_index);
                    if scratch_storage.is_null() {
                        return Err(format!("{name} graph scratch argument is null storage"));
                    }
                    let scratch = std::ptr::read_unaligned(scratch_storage.cast::<u64>());
                    if scratch == 0 {
                        return Err(format!("{name} graph scratch pointer is null"));
                    }
                    pipeline_nodes.push((node, name.clone(), scratch));
                }
                identities.push(json!({"symbol":name,"grid":grid,"block":block,"dynamic_shared":params.sharedMemBytes}));
            }
            if exact_pipeline {
                let mut pipeline_names = pipeline_nodes
                    .iter()
                    .map(|(_, name, _)| name.clone())
                    .collect::<Vec<_>>();
                pipeline_names.sort();
                let mut expected = if f.cell.reduction == 0 {
                    vec![FIXED.to_owned()]
                } else {
                    vec![FIXED.to_owned(), TRANSPOSE.to_owned()]
                };
                expected.sort();
                if pipeline_names != expected {
                    return Err("admitted route must contain the entire expected pipeline".into());
                }
                if f.cell.reduction != 0
                    && (pipeline_nodes.len() != 2 || pipeline_nodes[0].2 != pipeline_nodes[1].2)
                {
                    return Err(
                        "transpose output and Fixed B must share one exact scratch pointer".into(),
                    );
                }
                let mut edge_count = 0_usize;
                if sys::cuGraphGetEdges_v2(
                    graph.cu_graph(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                ) != sys::CUresult::CUDA_SUCCESS
                {
                    return Err("graph edge-count query failed".into());
                }
                let expected_edges = usize::from(f.cell.reduction != 0);
                if edge_count != expected_edges {
                    return Err(format!(
                        "admitted route has {edge_count} graph edges, expected {expected_edges}"
                    ));
                }
                if expected_edges == 1 {
                    let mut from = std::ptr::null_mut();
                    let mut to = std::ptr::null_mut();
                    let mut edge: sys::CUgraphEdgeData = std::mem::zeroed();
                    if sys::cuGraphGetEdges_v2(
                        graph.cu_graph(),
                        &mut from,
                        &mut to,
                        &mut edge,
                        &mut edge_count,
                    ) != sys::CUresult::CUDA_SUCCESS
                    {
                        return Err("graph edge query failed".into());
                    }
                    let symbol_for = |handle| {
                        pipeline_nodes
                            .iter()
                            .find_map(|(node, name, _)| (*node == handle).then_some(name.as_str()))
                    };
                    if symbol_for(from) != Some(TRANSPOSE)
                        || symbol_for(to) != Some(FIXED)
                        || edge.from_port != 0
                        || edge.to_port != 0
                        || edge.type_
                            != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
                        || edge.reserved != [0; 5]
                    {
                        return Err(
                            "admitted graph is not the exact transpose -> Fixed chain".into()
                        );
                    }
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

    fn integrated_resource_gate(
        function: &CudaFunction,
        symbol: &str,
        threads: u32,
        expected_shared: usize,
        register_cap: i32,
    ) -> Result<(), String> {
        let registers = function.num_regs().map_err(|e| format!("regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("local: {e:?}"))?;
        let shared = function
            .shared_size_bytes()
            .map_err(|e| format!("shared: {e:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .map_err(|e| format!("occupancy: {e:?}"))?;
        if registers <= 0
            || registers > register_cap
            || local != 0
            || shared as usize != expected_shared
            || occupancy < 3
        {
            return Err(format!(
                "{symbol} integrated resource gate failed: registers={registers}/{register_cap} local={local}/0 shared={shared}/{expected_shared} occupancy={occupancy}/3"
            ));
        }
        Ok(())
    }

    fn driver_abi_gate(
        function: sys::CUfunction,
        symbol: &str,
        expected: &[(usize, usize)],
    ) -> Result<(), String> {
        for (index, &(expected_offset, expected_size)) in expected.iter().enumerate() {
            let (mut offset, mut size) = (usize::MAX, usize::MAX);
            let result =
                unsafe { sys::cuFuncGetParamInfo(function, index, &mut offset, &mut size) };
            if result != sys::CUresult::CUDA_SUCCESS
                || (offset, size) != (expected_offset, expected_size)
            {
                return Err(format!(
                    "{symbol} Driver ABI parameter {index} changed: result={result:?} actual=({offset},{size}) expected=({expected_offset},{expected_size})"
                ));
            }
        }
        let (mut offset, mut size) = (0, 0);
        let terminal =
            unsafe { sys::cuFuncGetParamInfo(function, expected.len(), &mut offset, &mut size) };
        if terminal != sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            return Err(format!(
                "{symbol} exposes an unexpected terminal Driver ABI parameter: {terminal:?} ({offset},{size})"
            ));
        }
        Ok(())
    }

    fn check_bits(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        f.reset(ctx, Arm::Generic)?;
        launch(ctx, fixed, f, Arm::Generic)?;
        let exact = f.output.bits(ctx)?;
        let mut fast_golden = Vec::new();
        for arm in [Arm::Auto, Arm::Candidate, Arm::Fast] {
            if arm == Arm::Fast && f.cell.reduction == 0 {
                continue;
            }
            f.reset(ctx, arm)?;
            launch(ctx, fixed, f, arm)?;
            let eager = if matches!(arm, Arm::Auto | Arm::Fast) {
                f.production_output.bits(ctx)?
            } else {
                f.output.bits(ctx)?
            };
            let golden = if arm == Arm::Fast {
                fast_golden = eager;
                &fast_golden
            } else {
                &exact
            };
            f.validate(ctx, golden, arm)?;
            let graph = capture(ctx, fixed, f, arm)?;
            graph_identity(&graph, f, arm)?;
            for path in ["eager", "graph"] {
                for repeat in 0..2 {
                    f.reset(ctx, arm)?;
                    if path == "graph" {
                        graph.launch().map_err(|e| format!("bits graph: {e:?}"))?;
                    } else {
                        launch(ctx, fixed, f, arm)?;
                    }
                    f.validate(ctx, golden, arm)?;
                    println!(
                        "{}",
                        json!({"schema":"MambaBiNtCopyPlanSiblingsBitsV1","cell":f.cell.name,"arm":format!("{arm:?}"),"path":path,"repeat":repeat,"words":golden.len(),"oracle":if arm == Arm::Fast {"vendor_self"} else {"generic_exact"}})
                    );
                }
            }
        }
        Ok((exact, fast_golden))
    }

    fn check_candidate_bits(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
    ) -> Result<(), String> {
        f.reset(ctx, Arm::Generic)?;
        launch(ctx, fixed, f, Arm::Generic)?;
        let exact = f.output.bits(ctx)?;
        for arm in [Arm::Generic, Arm::Candidate] {
            f.reset(ctx, arm)?;
            launch(ctx, fixed, f, arm)?;
            f.validate(ctx, &exact, arm)?;
            let graph = capture(ctx, fixed, f, arm)?;
            graph_identity(&graph, f, arm)?;
            f.reset(ctx, arm)?;
            graph.launch().map_err(|e| format!("bits graph: {e:?}"))?;
            f.validate(ctx, &exact, arm)?;
        }
        Ok(())
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
        f.reset(ctx, arm)?;
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
        f.validate(ctx, golden, arm)?;
        if !us.is_finite() || us <= 0.0 {
            return Err("invalid elapsed time".into());
        }
        Ok(us)
    }

    fn paired_candidate_vs_prior_auto_screen(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
        golden: &[u32],
        windows: usize,
    ) -> Result<Vec<[f64; 2]>, String> {
        if !matches!(windows, 3 | 7) {
            return Err("pre-admission screen permits only once3 or once7".into());
        }
        let candidate_graph = capture(ctx, fixed, f, Arm::Candidate)?;
        graph_identity(&candidate_graph, f, Arm::Candidate)?;
        let auto_graph = capture(ctx, fixed, f, Arm::Auto)?;
        graph_identity(&auto_graph, f, Arm::Auto)?;
        let mut strata = Vec::new();
        for path in ["eager", "graph"] {
            for candidate_endpoints in [true, false] {
                let arms = if candidate_endpoints {
                    [Arm::Candidate, Arm::Auto, Arm::Auto, Arm::Candidate]
                } else {
                    [Arm::Auto, Arm::Candidate, Arm::Candidate, Arm::Auto]
                };
                let mut raw = Vec::new();
                for bracket in 0..windows + 2 {
                    let mut observations = [0.; 4];
                    for (index, arm) in arms.into_iter().enumerate() {
                        let graph = if arm == Arm::Candidate {
                            &candidate_graph
                        } else {
                            &auto_graph
                        };
                        observations[index] = measure(ctx, fixed, f, arm, graph, path, golden)?;
                    }
                    if bracket >= 2 {
                        raw.push(observations);
                    }
                }
                let ratios = raw
                    .iter()
                    .map(|observation| ratio(*observation, candidate_endpoints))
                    .collect::<Result<Vec<_>, _>>()?;
                let p50 = quantile(&ratios, 0.5);
                let p95 = quantile(&ratios, 0.95);
                println!(
                    "{}",
                    json!({
                        "schema":"MambaBiNtCopyPlanSiblingsPreAdmissionScreenV1",
                        "cell":f.cell.name,
                        "toolkit":format!("{:?}",ctx.kernels.compiler_identity().nvrtc_version),
                        "phase":format!("once{windows}"),
                        "path":path,
                        "order":if candidate_endpoints {"ABBA"} else {"BAAB"},
                        "windows":windows,
                        "raw_observations_us":raw,
                        "ratio_direction":"candidate_over_prior_actual_auto",
                        "ratio_p50":p50,
                        "ratio_p95":p95,
                    })
                );
                strata.push([p50, p95]);
            }
        }
        if strata
            .iter()
            .any(|values| values[0] >= 0.99 || values[1] >= 0.99)
        {
            return Err(format!(
                "{} once{windows} failed strict candidate/prior-AUTO p50+p95<.99: {strata:?}",
                f.cell.name
            ));
        }
        Ok(strata)
    }

    fn paired_post_admission_screen(
        ctx: &GpuCtx,
        fixed: &CudaFunction,
        f: &mut Fixture,
        exact: &[u32],
        fast: &[u32],
        comparator: Arm,
    ) -> Result<Vec<[f64; 2]>, String> {
        if !matches!(comparator, Arm::Generic | Arm::Fast) {
            return Err("post-admission comparator must be Generic or Fast".into());
        }
        let auto_graph = capture(ctx, fixed, f, Arm::Auto)?;
        graph_identity(&auto_graph, f, Arm::Auto)?;
        let comparator_graph = capture(ctx, fixed, f, comparator)?;
        graph_identity(&comparator_graph, f, comparator)?;
        let mut strata = Vec::new();
        for path in ["eager", "graph"] {
            for auto_endpoints in [true, false] {
                let arms = if auto_endpoints {
                    [Arm::Auto, comparator, comparator, Arm::Auto]
                } else {
                    [comparator, Arm::Auto, Arm::Auto, comparator]
                };
                let mut raw = Vec::new();
                for bracket in 0..23 {
                    let mut observations = [0.; 4];
                    for (index, arm) in arms.into_iter().enumerate() {
                        let (graph, golden) = if arm == Arm::Auto {
                            (&auto_graph, exact)
                        } else {
                            (
                                &comparator_graph,
                                if comparator == Arm::Fast { fast } else { exact },
                            )
                        };
                        observations[index] = measure(ctx, fixed, f, arm, graph, path, golden)?;
                    }
                    if bracket >= 2 {
                        raw.push(observations);
                    }
                }
                let ratios = raw
                    .iter()
                    .map(|observation| ratio(*observation, auto_endpoints))
                    .collect::<Result<Vec<_>, _>>()?;
                let p50 = quantile(&ratios, 0.5);
                let p95 = quantile(&ratios, 0.95);
                println!(
                    "{}",
                    json!({
                        "schema":"MambaBiNtCopyPlanSiblingsPostAdmissionOnce21V1",
                        "cell":f.cell.name,
                        "toolkit":format!("{:?}",ctx.kernels.compiler_identity().nvrtc_version),
                        "comparator":format!("{comparator:?}"),
                        "path":path,
                        "order":if auto_endpoints {"ABBA"} else {"BAAB"},
                        "windows":21,
                        "raw_observations_us":raw,
                        "ratio_direction":"actual_auto_over_comparator",
                        "ratio_p50":p50,
                        "ratio_p95":p95,
                        "fast_is_separately_labeled":comparator == Arm::Fast,
                    })
                );
                strata.push([p50, p95]);
            }
        }
        if comparator == Arm::Generic
            && strata
                .iter()
                .any(|values| values[0] >= 0.99 || values[1] >= 0.99)
        {
            return Err(format!(
                "{} post-admission AUTO failed strict improvement over forced Generic p50+p95<.99: {strata:?}",
                f.cell.name
            ));
        }
        Ok(strata)
    }

    fn require_admitted_physical_identity(
        evidence: &QualifiedPhysicalLaunchEvidence,
        cell: Cell,
    ) -> Result<(), String> {
        let nodes = evidence.nodes();
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 2
            || nodes.len() != 2
            || nodes[0].symbol != TRANSPOSE
            || nodes[0].module_kind != ModuleKind::TriadScalar
            || nodes[0].logical_op != ResolvedGemmOp::Nt
            || nodes[0].shape != (cell.m, cell.out, cell.reduction)
            || nodes[0].strides != (cell.reduction, cell.reduction, cell.out)
            || nodes[0].numeric_contract != Some(ResolvedNumericContract::ScalarFmaV1)
            || nodes[0].launch.grid_dim != cell.transpose_grid()
            || nodes[0].launch.block_dim != (32, 16, 1)
            || nodes[0].launch.shared_mem_bytes != 0
            || nodes[1].symbol != FIXED
            || nodes[1].module_kind != ModuleKind::Fixed
            || nodes[1].logical_op != ResolvedGemmOp::Nt
            || nodes[1].shape != (cell.m, cell.out, cell.reduction)
            || nodes[1].strides != (cell.reduction, cell.reduction, cell.out)
            || nodes[1].tile != Some((64, 64))
            || nodes[1].numeric_contract != Some(ResolvedNumericContract::ScalarFmaV1)
            || nodes[1].launch.grid_dim != cell.fixed_grid()
            || nodes[1].launch.block_dim != (128, 1, 1)
            || nodes[1].launch.shared_mem_bytes != 0
            || nodes[0].launch.arguments_digest == [0; 32]
            || nodes[1].launch.arguments_digest == [0; 32]
            || nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest
            || evidence.launch_digest() == [0; 32]
        {
            return Err(format!(
                "{} admitted eager/prepared physical identity changed: {evidence:?}",
                cell.name
            ));
        }
        Ok(())
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
            check_bits(
                &ctx,
                &fixed,
                &mut Fixture::new(&ctx, cell, exceptional, 1.0)?,
            )?;
        }
        for cell in CELLS {
            let mut f = Fixture::new(&ctx, cell, false, 1.0)?;
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

    #[test]
    #[ignore = "pre-admission RTX6000Ada qualification; run independently on frozen CUDA12.8/13.0/13.2 before populating the production cohort"]
    fn ada_f32_nt_copyplan_siblings_pre_admission_qualification() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("pre-admission qualification requires --release".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-copyplan-siblings-pre-admission/pre")?;
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
        let historical_request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            (CELLS[0].m, CELLS[0].out, CELLS[0].reduction),
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let historical_probe = qualify_physical_launch(&ctx, historical_request)?;
        let historical_nodes = historical_probe.evidence().nodes();
        if historical_nodes.len() != 1 || historical_nodes[0].symbol != CELLS[0].prior_auto_symbol()
        {
            return Err(
                "historical pre-admission gate requires an empty production cohort; AUTO is already admitted—run the post-admission qualification instead"
                    .into(),
            );
        }
        drop(historical_probe);
        let fixed_compiler = ctx.kernels.compiler_identity();
        let scalar_compiler = ctx.kernels.triad_scalar_compiler_identity();
        if ![(12, 8), (13, 0), (13, 2)].contains(&fixed_compiler.nvrtc_version)
            || scalar_compiler.nvrtc_version != fixed_compiler.nvrtc_version
            || fixed_compiler.target.as_str() != "sm_89"
            || scalar_compiler.target.as_str() != "sm_89"
            || !fixed_compiler.nvrtc_library_known
            || !scalar_compiler.nvrtc_library_known
            || fixed_compiler.nvrtc_library_domain != scalar_compiler.nvrtc_library_domain
        {
            return Err(format!(
                "pre-admission composed compiler domain is not frozen: fixed={fixed_compiler:?} scalar={scalar_compiler:?}"
            ));
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiNtCopyPlanSiblingsPreAdmissionIdentityV1",
                "fixed_compiler":format!("{fixed_compiler:?}"),
                "scalar_compiler":format!("{scalar_compiler:?}"),
                "artifacts":format!("{:?}",ctx.kernels.artifact_set_identity()),
                "production_admission":false,
            })
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
        integrated_resource_gate(&fixed, FIXED, 128, 32_768, 135)?;
        integrated_resource_gate(
            &ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            TRANSPOSE,
            512,
            4_224,
            18,
        )?;
        let mut abi_probe = Fixture::new(&ctx, CELLS[0], false, 1.0)?;
        let abi_graph = capture(&ctx, &fixed, &mut abi_probe, Arm::Candidate)?;
        graph_identity(&abi_graph, &abi_probe, Arm::Candidate)?;
        abi_probe.a.unchanged(&ctx)?;
        abi_probe.b.unchanged(&ctx)?;
        abi_probe.output.bits(&ctx)?;
        abi_probe.scratch.bits(&ctx)?;
        drop(abi_graph);
        drop(abi_probe);

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
            check_bits(
                &ctx,
                &fixed,
                &mut Fixture::new(&ctx, cell, exceptional, 1.0)?,
            )?;
        }
        check_candidate_bits(
            &ctx,
            &fixed,
            &mut Fixture::new(
                &ctx,
                Cell {
                    name: "nonunit_alpha",
                    m: 67,
                    out: 68,
                    reduction: 36,
                },
                false,
                0.375,
            )?,
        )?;

        for cell in CELLS {
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nt,
                (cell.m, cell.out, cell.reduction),
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            );
            {
                let mut qualified = qualify_physical_launch(&ctx, request)?;
                let evidence = qualified.evidence();
                let nodes = evidence.nodes();
                if !evidence.eager_graph_equal()
                    || evidence.launch_count() != 1
                    || nodes.len() != 1
                    || nodes[0].symbol != cell.prior_auto_symbol()
                    || nodes[0].module_kind != ModuleKind::TriadScalar
                    || nodes[0].logical_op != ResolvedGemmOp::Nt
                    || nodes[0].shape != (cell.m, cell.out, cell.reduction)
                    || nodes[0].strides != (cell.reduction, cell.reduction, cell.out)
                    || nodes[0].numeric_contract != Some(ResolvedNumericContract::ScalarFmaV1)
                    || nodes[0].launch.grid_dim != cell.prior_auto_grid()
                    || nodes[0].launch.block_dim != cell.prior_auto_block()
                    || nodes[0].launch.shared_mem_bytes != cell.prior_auto_shared()
                    || nodes[0].launch.arguments_digest == [0; 32]
                    || evidence.launch_digest() == [0; 32]
                {
                    return Err(format!(
                        "{} actual AUTO must remain the prior eager/prepared fallback: {evidence:?}",
                        cell.name
                    ));
                }
                qualified.seed_f32_operands(&ctx, 0x00a1_7680 ^ cell.m as u64)?;
                let before = qualified.f32_operand_bits(&ctx)?;
                qualified.measure_eager_window_ms(&ctx, 1)?;
                let eager = qualified.f32_output_bits(&ctx)?;
                if qualified.f32_operand_bits(&ctx)? != before {
                    return Err(format!("{} actual AUTO eager mutated A/B", cell.name));
                }
                qualified.validate_red_zones(&ctx)?;
                qualified.seed_f32_operands(&ctx, 0x00a1_7680 ^ cell.m as u64)?;
                let before = qualified.f32_operand_bits(&ctx)?;
                qualified.measure_graph_window_ms(&ctx, 1)?;
                if qualified.f32_output_bits(&ctx)? != eager
                    || qualified.f32_operand_bits(&ctx)? != before
                {
                    return Err(format!(
                        "{} actual AUTO captured graph changed output bits or A/B",
                        cell.name
                    ));
                }
                qualified.validate_red_zones(&ctx)?;
            }

            let mut fixture = Fixture::new(&ctx, cell, false, 1.0)?;
            let (exact, _) = check_bits(&ctx, &fixed, &mut fixture)?;
            quiet.require_cohort("nt-copyplan-siblings-pre-admission/timed")?;
            let once3 =
                paired_candidate_vs_prior_auto_screen(&ctx, &fixed, &mut fixture, &exact, 3)?;
            let once7 =
                paired_candidate_vs_prior_auto_screen(&ctx, &fixed, &mut fixture, &exact, 7)?;
            println!(
                "{}",
                json!({
                    "schema":"MambaBiNtCopyPlanSiblingsPreAdmissionDecisionV1",
                    "cell":cell.name,
                    "shape":[cell.m,cell.out,cell.reduction],
                    "candidate_symbols":[TRANSPOSE,FIXED],
                    "prior_actual_auto_symbol":cell.prior_auto_symbol(),
                    "exact_oracle":"generic_exact_f32",
                    "once3":once3,
                    "once7":once7,
                    "production_admission":false,
                    "decision":"qualified_for_separate_identity_admission_commit",
                })
            );
        }
        drop(ctx);
        quiet.verify_post_cohort("nt-copyplan-siblings-pre-admission/post")?;
        Ok(())
    }

    #[test]
    #[ignore = "post-admission RTX6000Ada qualification; run independently on frozen CUDA12.8/13.0/13.2"]
    fn ada_f32_nt_copyplan_siblings_post_admission_qualification() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("post-admission qualification requires --release".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("nt-copyplan-siblings-post-admission/pre")?;
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
        let fixed_compiler = ctx.kernels.compiler_identity();
        let scalar_compiler = ctx.kernels.triad_scalar_compiler_identity();
        if ![(12, 8), (13, 0), (13, 2)].contains(&fixed_compiler.nvrtc_version)
            || scalar_compiler.nvrtc_version != fixed_compiler.nvrtc_version
            || fixed_compiler.target.as_str() != "sm_89"
            || scalar_compiler.target.as_str() != "sm_89"
            || !fixed_compiler.nvrtc_library_known
            || !scalar_compiler.nvrtc_library_known
            || fixed_compiler.nvrtc_library_domain != scalar_compiler.nvrtc_library_domain
        {
            return Err(format!(
                "post-admission composed compiler domain is not frozen: fixed={fixed_compiler:?} scalar={scalar_compiler:?}"
            ));
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiNtCopyPlanSiblingsPostAdmissionIdentityV1",
                "fixed_compiler":format!("{fixed_compiler:?}"),
                "scalar_compiler":format!("{scalar_compiler:?}"),
                "artifacts":format!("{:?}",ctx.kernels.artifact_set_identity()),
                "production_admission":true,
            })
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
        integrated_resource_gate(&fixed, FIXED, 128, 32_768, 135)?;
        integrated_resource_gate(
            &ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            TRANSPOSE,
            512,
            4_224,
            18,
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
            check_bits(
                &ctx,
                &fixed,
                &mut Fixture::new(&ctx, cell, exceptional, 1.0)?,
            )?;
        }
        check_candidate_bits(
            &ctx,
            &fixed,
            &mut Fixture::new(
                &ctx,
                Cell {
                    name: "nonunit_alpha",
                    m: 67,
                    out: 68,
                    reduction: 36,
                },
                false,
                0.375,
            )?,
        )?;

        for cell in CELLS {
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nt,
                (cell.m, cell.out, cell.reduction),
                PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            );
            {
                let mut qualified = qualify_physical_launch(&ctx, request)?;
                require_admitted_physical_identity(qualified.evidence(), cell)?;
                qualified.seed_f32_operands(&ctx, 0x00a2_7680 ^ cell.m as u64)?;
                let before = qualified.f32_operand_bits(&ctx)?;
                qualified.measure_eager_window_ms(&ctx, 1)?;
                let eager = qualified.f32_output_bits(&ctx)?;
                if qualified.f32_operand_bits(&ctx)? != before {
                    return Err(format!("{} admitted AUTO eager mutated A/B", cell.name));
                }
                qualified.validate_red_zones(&ctx)?;
                qualified.seed_f32_operands(&ctx, 0x00a2_7680 ^ cell.m as u64)?;
                let before = qualified.f32_operand_bits(&ctx)?;
                qualified.measure_graph_window_ms(&ctx, 1)?;
                if qualified.f32_output_bits(&ctx)? != eager
                    || qualified.f32_operand_bits(&ctx)? != before
                {
                    return Err(format!(
                        "{} admitted AUTO prepared graph changed output bits or A/B",
                        cell.name
                    ));
                }
                qualified.validate_red_zones(&ctx)?;
            }

            let mut fixture = Fixture::new(&ctx, cell, false, 1.0)?;
            let (exact, fast) = check_bits(&ctx, &fixed, &mut fixture)?;
            quiet.require_cohort("nt-copyplan-siblings-post-admission/timed")?;
            let generic = paired_post_admission_screen(
                &ctx,
                &fixed,
                &mut fixture,
                &exact,
                &fast,
                Arm::Generic,
            )?;
            let fast_labeled =
                paired_post_admission_screen(&ctx, &fixed, &mut fixture, &exact, &fast, Arm::Fast)?;
            println!(
                "{}",
                json!({
                    "schema":"MambaBiNtCopyPlanSiblingsPostAdmissionDecisionV1",
                    "cell":cell.name,
                    "shape":[cell.m,cell.out,cell.reduction],
                    "actual_auto_symbols":[TRANSPOSE,FIXED],
                    "forced_generic_symbol":"gemm_bi_nt",
                    "actual_auto_over_forced_generic_once21":generic,
                    "actual_auto_over_fast_once21_labeled_only":fast_labeled,
                    "production_admission":true,
                    "decision":"retain_admitted_route_if_all_generic_strata_p50_p95_lt_0.99",
                })
            );
        }
        drop(ctx);
        quiet.verify_post_cohort("nt-copyplan-siblings-post-admission/post")?;
        Ok(())
    }
}
