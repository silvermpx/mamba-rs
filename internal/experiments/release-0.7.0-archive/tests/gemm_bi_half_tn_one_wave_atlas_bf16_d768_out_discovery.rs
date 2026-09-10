//! Test-only Ada BF16 TN d768-out discovery for a heterogeneous one-wave atlas.

#[path = "support/triad_half_tn_one_wave_atlas_source.rs"]
#[allow(dead_code)]
mod candidate_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingPhase {
    Scout,
    Qualification,
}

const SCOUT_THRESHOLD: f64 = 0.99;
const QUALIFICATION_THRESHOLD: f64 = 0.99;

const fn windows_for(phase: TimingPhase) -> usize {
    match phase {
        TimingPhase::Scout => 3,
        TimingPhase::Qualification => 7,
    }
}

fn may_run_fast(scout_pass: bool, retained_once7_pass: bool) -> bool {
    scout_pass && retained_once7_pass
}

fn uses_target_atlas(reduction: usize, k_out: usize, n: usize) -> bool {
    (reduction, k_out, n) == candidate_source::TARGET
}

fn ptxas_static_shared_bytes(block: &str) -> u32 {
    block
        .lines()
        .find_map(|line| {
            let prefix = line.split_once(" bytes smem")?.0;
            prefix.split_whitespace().next_back()?.parse().ok()
        })
        .unwrap_or(0)
}

#[test]
fn ptxas_omitted_smem_metric_means_zero_static_shared() {
    let dynamic_only = "ptxas info    : Used 124 registers\nptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads";
    let retained_static = "ptxas info    : Used 118 registers, used 32768 bytes smem\nptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads";
    assert_eq!(ptxas_static_shared_bytes(dynamic_only), 0);
    assert_eq!(ptxas_static_shared_bytes(retained_static), 32_768);
}

#[test]
fn timing_plan_is_once3_then_gated_retained_once7_before_fast() {
    assert_eq!(windows_for(TimingPhase::Scout), 3);
    assert_eq!(windows_for(TimingPhase::Qualification), 7);
    assert!(!may_run_fast(false, false));
    assert!(!may_run_fast(true, false));
    assert!(may_run_fast(true, true));
}

#[test]
fn strict_gate_requires_every_p50_and_p95_below_point99() {
    assert_eq!(SCOUT_THRESHOLD, 0.99);
    assert_eq!(QUALIFICATION_THRESHOLD, 0.99);
    assert!(candidate_source::all_strata_below(
        &[[0.98, 0.989]; 4],
        SCOUT_THRESHOLD
    ));
    assert!(!candidate_source::all_strata_below(
        &[[0.98, 0.99]; 4],
        QUALIFICATION_THRESHOLD
    ));
}

#[test]
fn harness_scope_freezes_bf16_one_wave_atlas_physics_and_retained_comparator() {
    assert_eq!(candidate_source::TARGET, (2_048, 1_536, 768));
    assert_eq!(candidate_source::TARGET_GRID, 142);
    assert_eq!(candidate_source::BLOCK_THREADS, 288);
    assert_eq!(candidate_source::REGISTER_CAP, 128);
    assert_eq!(candidate_source::STATIC_SHARED_BYTES, 0);
    assert_eq!(candidate_source::DYNAMIC_SHARED_BYTES, 65_536);
    assert_eq!(candidate_source::REQUIRED_OCCUPANCY, 1);
    assert_eq!(candidate_source::RETAINED_BLOCK_THREADS, 128);
    assert_eq!(candidate_source::RETAINED_STATIC_SHARED_BYTES, 32_768);
    assert_eq!(candidate_source::RETAINED_REQUIRED_OCCUPANCY, 3);
    assert_eq!(candidate_source::target_staged_half_elements(), 1_658_880);
    assert_eq!(candidate_source::retained_staged_half_elements(), 2_359_296);
    assert_eq!(candidate_source::target_active_warps(), 1_152);
    assert!(uses_target_atlas(2_048, 1_536, 768));
    assert!(!uses_target_atlas(2_047, 1_536, 768));
    assert!(!uses_target_atlas(2_048, 1_535, 768));
    assert!(!uses_target_atlas(2_048, 1_536, 767));
}

#[cfg(feature = "cuda")]
mod common;

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::ffi::{CStr, c_void};
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaGraph, CudaModule, LaunchConfig, PushKernelArg, sys};
    use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use sha2::{Digest as _, Sha256};

    use super::candidate_source;
    use super::common::gpu_quiet::QuietGpu;
    use super::{
        QUALIFICATION_THRESHOLD, SCOUT_THRESHOLD, TimingPhase, may_run_fast, uses_target_atlas,
        windows_for,
    };

    const HALF_GUARD_WORDS: usize = 128;
    const F32_GUARD_WORDS: usize = 64;
    const HALF_GUARD_BITS: u16 = 0x7e4d;
    const F32_GUARD_BITS: u32 = 0x7fc0_51ad;
    const OPERATIONS: usize = 20;
    const WARMUPS: usize = 3;
    const TARGET: Shape = Shape::new(
        candidate_source::TARGET.0,
        candidate_source::TARGET.1,
        candidate_source::TARGET.2,
        1.0,
        "d768_out",
    );
    const ALIGNED_TAIL: Shape = Shape::new(67, 72, 72, -0.75, "aligned_tail_negative_alpha");
    const MISALIGNED_TAIL: Shape = Shape::new(67, 69, 71, -0.75, "misaligned_tail_negative_alpha");
    const EXCEPTIONAL: Shape = Shape::new(64, 64, 96, 1.0, "exceptional_full_tile");
    const K0: Shape = Shape::new(0, 97, 67, 1.0, "k0");

    #[derive(Clone, Copy)]
    struct Shape {
        reduction: usize,
        k_out: usize,
        n: usize,
        alpha: f32,
        label: &'static str,
    }

    impl Shape {
        const fn new(
            reduction: usize,
            k_out: usize,
            n: usize,
            alpha: f32,
            label: &'static str,
        ) -> Self {
            Self {
                reduction,
                k_out,
                n,
                alpha,
                label,
            }
        }

        fn uses_atlas(self) -> bool {
            uses_target_atlas(self.reduction, self.k_out, self.n)
        }

        fn candidate_grid(self) -> u32 {
            if self.uses_atlas() {
                candidate_source::TARGET_GRID
            } else {
                self.retained_grid()
            }
        }

        fn candidate_threads(self) -> u32 {
            if self.uses_atlas() {
                candidate_source::BLOCK_THREADS
            } else {
                candidate_source::RETAINED_BLOCK_THREADS
            }
        }

        fn candidate_dynamic_shared(self) -> u32 {
            if self.uses_atlas() {
                candidate_source::DYNAMIC_SHARED_BYTES
            } else {
                0
            }
        }

        fn retained_grid(self) -> u32 {
            self.k_out
                .div_ceil(64)
                .checked_mul(self.n.div_ceil(64))
                .unwrap() as u32
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Retained,
        Fast,
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Candidate => "one_wave_atlas_bk64_s2_regpipe_vec2",
                Self::Retained => "retained_regpipe_vec2",
                Self::Fast => "cublas_fast_bf16",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Order {
        Abba,
        Baab,
    }

    impl Order {
        const fn name(self) -> &'static str {
            match self {
                Self::Abba => "ABBA",
                Self::Baab => "BAAB",
            }
        }

        fn arms(self, comparator: Arm) -> [Arm; 4] {
            match self {
                Self::Abba => [Arm::Candidate, comparator, comparator, Arm::Candidate],
                Self::Baab => [comparator, Arm::Candidate, Arm::Candidate, comparator],
            }
        }
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _candidate_module: Arc<CudaModule>,
        _retained_module: Arc<CudaModule>,
        candidate: CudaFunction,
        retained: CudaFunction,
        candidate_source_sha: String,
        retained_source_sha: String,
        candidate_ptx_sha: String,
        retained_ptx_sha: String,
    }

    struct GuardedHalf {
        buffer: GpuByteBuffer,
        baseline: Vec<u16>,
    }

    impl GuardedHalf {
        fn new(ctx: &GpuCtx, words: &[u16]) -> Result<Self, String> {
            let mut baseline =
                vec![HALF_GUARD_BITS; HALF_GUARD_WORDS + words.len() + HALF_GUARD_WORDS];
            baseline[HALF_GUARD_WORDS..HALF_GUARD_WORDS + words.len()].copy_from_slice(words);
            let mut buffer = GpuByteBuffer::zeros(&ctx.stream, baseline.len() * 2)?;
            buffer.upload_bytes(&ctx.stream, bytemuck::cast_slice(&baseline))?;
            Ok(Self { buffer, baseline })
        }

        fn ptr(&self) -> u64 {
            self.buffer.cached_ptr() + (HALF_GUARD_WORDS * 2) as u64
        }

        fn raw(&self, ctx: &GpuCtx) -> Result<Vec<u16>, String> {
            let mut bytes = vec![0; self.buffer.len_bytes()];
            ctx.stream
                .memcpy_dtoh(self.buffer.inner(), &mut bytes)
                .map_err(|error| format!("half download: {error:?}"))?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("half download sync: {error:?}"))?;
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .copied()
                .map(u16::from_le_bytes)
                .collect())
        }

        fn unchanged(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
            if self.raw(ctx)? != self.baseline {
                return Err(format!("{label} input or redzone changed"));
            }
            Ok(())
        }
    }

    struct GuardedF32 {
        buffer: GpuByteBuffer,
        baseline: Vec<u32>,
        active: usize,
    }

    impl GuardedF32 {
        fn new(ctx: &GpuCtx, words: &[u32]) -> Result<Self, String> {
            let mut baseline =
                vec![F32_GUARD_BITS; F32_GUARD_WORDS + words.len() + F32_GUARD_WORDS];
            baseline[F32_GUARD_WORDS..F32_GUARD_WORDS + words.len()].copy_from_slice(words);
            let mut buffer = GpuByteBuffer::zeros(&ctx.stream, baseline.len() * 4)?;
            buffer.upload_bytes(&ctx.stream, bytemuck::cast_slice(&baseline))?;
            Ok(Self {
                buffer,
                baseline,
                active: words.len(),
            })
        }

        fn ptr(&self) -> u64 {
            self.buffer.cached_ptr() + (F32_GUARD_WORDS * 4) as u64
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.buffer
                .upload_bytes(&ctx.stream, bytemuck::cast_slice(&self.baseline))
        }

        fn raw(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let mut bytes = vec![0; self.buffer.len_bytes()];
            ctx.stream
                .memcpy_dtoh(self.buffer.inner(), &mut bytes)
                .map_err(|error| format!("f32 download: {error:?}"))?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("f32 download sync: {error:?}"))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect())
        }

        fn active_and_guards(&self, ctx: &GpuCtx, label: &str) -> Result<Vec<u32>, String> {
            let raw = self.raw(ctx)?;
            if raw[..F32_GUARD_WORDS]
                .iter()
                .chain(&raw[F32_GUARD_WORDS + self.active..])
                .any(|&word| word != F32_GUARD_BITS)
            {
                return Err(format!("{label} changed an f32 redzone"));
            }
            Ok(raw[F32_GUARD_WORDS..F32_GUARD_WORDS + self.active].to_vec())
        }
    }

    struct Fixture {
        shape: Shape,
        a: GuardedHalf,
        b: GuardedHalf,
        outputs: [GuardedF32; 3],
        seed: Vec<u32>,
    }

    impl Fixture {
        fn new(runtime: &Runtime, shape: Shape, exceptional: bool) -> Result<Self, String> {
            let a = half_corpus(
                shape.reduction * shape.k_out,
                0xa89a_9101 ^ shape.reduction as u64,
                exceptional,
            );
            let b = half_corpus(
                shape.reduction * shape.n,
                0xb89a_9102 ^ shape.n as u64,
                exceptional,
            );
            let seed = f32_seed(shape.k_out * shape.n, 0xc89a_9103 ^ shape.k_out as u64);
            let fixture = Self {
                shape,
                a: GuardedHalf::new(&runtime.ctx, &a)?,
                b: GuardedHalf::new(&runtime.ctx, &b)?,
                outputs: [
                    GuardedF32::new(&runtime.ctx, &seed)?,
                    GuardedF32::new(&runtime.ctx, &seed)?,
                    GuardedF32::new(&runtime.ctx, &seed)?,
                ],
                seed,
            };
            if fixture.a.ptr() % 256 != 0
                || fixture.b.ptr() % 256 != 0
                || fixture.outputs.iter().any(|output| output.ptr() % 256 != 0)
            {
                return Err(format!("{} fixture is not 256-byte aligned", shape.label));
            }
            Ok(fixture)
        }

        fn output(&self, arm: Arm) -> &GuardedF32 {
            &self.outputs[arm as usize]
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedF32 {
            &mut self.outputs[arm as usize]
        }

        fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
            self.a.unchanged(&runtime.ctx, "A")?;
            self.b.unchanged(&runtime.ctx, "B")
        }
    }

    fn half_corpus(len: usize, mut state: u64, exceptional: bool) -> Vec<u16> {
        const SPECIAL: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x7f7f,
            0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1,
        ];
        (0..len)
            .map(|index| {
                if exceptional && index < 256 {
                    return SPECIAL[index % SPECIAL.len()];
                }
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value =
                    ((state.wrapping_add(index as u64) % 4095) as i32 - 2047) as f32 / 8192.0;
                half::bf16::from_f32(value).to_bits()
            })
            .collect()
    }

    fn f32_seed(len: usize, mut state: u64) -> Vec<u32> {
        (0..len)
            .map(|index| match index {
                0 => 0.0f32.to_bits(),
                1 => (-0.0f32).to_bits(),
                _ => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (((state % 8191) as i32 - 4095) as f32 / 131_072.0).to_bits()
                }
            })
            .collect()
    }

    fn strip_typed_include(source: &str) -> String {
        source
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn composed_source(transformed: &str) -> String {
        [
            include_str!("../kernels/_typed_prelude.cuh").to_owned(),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/contract.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/common.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/epilogue.cuh")),
            strip_typed_include(include_str!("../kernels/gemm_bi_triad/mma16.cuh")),
            transformed.to_owned(),
        ]
        .join("\n")
    }

    fn compile_ptx_only(
        transformed: &str,
        label: &str,
    ) -> Result<(cudarc::nvrtc::Ptx, String), String> {
        let source = composed_source(transformed);
        let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("compute_89"),
                options: vec![
                    "--fmad=true".into(),
                    "--extra-device-vectorization".into(),
                    "-DNDEBUG".into(),
                    "-DGEMM_BI_GROUP_M=16".into(),
                    "-DMAMBA_RS_STATE_CAP=256".into(),
                    "--frandom-seed=1295072049".into(),
                ],
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("NVRTC compile {label}: {error:?}"))?;
        Ok((ptx, source_sha))
    }

    fn compile_source(
        device: &GpuDevice,
        transformed: &str,
        symbol: &str,
        dynamic_shared: u32,
    ) -> Result<(Arc<CudaModule>, CudaFunction, String, String), String> {
        let (ptx, source_sha) = compile_ptx_only(transformed, symbol)?;
        let ptx_sha = format!("{:x}", Sha256::digest(ptx.to_src().as_bytes()));
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load module {symbol}: {error:?}"))?;
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        if dynamic_shared != 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared as i32,
                )
                .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        }
        Ok((module, function, source_sha, ptx_sha))
    }

    fn metric_before(line: &str, suffix: &str) -> Option<u32> {
        let prefix = line.split_once(suffix)?.0;
        prefix.split_whitespace().next_back()?.parse().ok()
    }

    fn sass_entry<'a>(sass: &'a str, symbol: &str) -> Result<&'a str, String> {
        let function_marker = format!("Function : {symbol}");
        let text_marker = format!(".text.{symbol}:");
        let (start, marker_len) = if let Some(start) = sass.find(&function_marker) {
            (start, function_marker.len())
        } else if let Some(start) = sass.find(&text_marker) {
            (start, text_marker.len())
        } else {
            return Err(format!("nvdisasm omitted {symbol}"));
        };
        let tail = &sass[start..];
        let remainder = &tail[marker_len..];
        let end = ["Function : ", "//--------------------- .text."]
            .into_iter()
            .filter_map(|marker| remainder.find(marker))
            .min()
            .map(|offset| marker_len + offset)
            .unwrap_or(tail.len());
        Ok(&tail[..end])
    }

    struct SassStats {
        registers: u32,
        static_shared: u32,
        hmma: usize,
        ldsm: usize,
        ldgsts: usize,
        cubin_sha: String,
        sass_sha: String,
    }

    fn cuda_tool(name: &str) -> std::path::PathBuf {
        let configured = std::env::var("CUDA_HOME")
            .ok()
            .map(|root| std::path::PathBuf::from(root).join("bin").join(name));
        configured
            .into_iter()
            .chain([std::path::PathBuf::from("/usr/local/cuda-13.2/bin").join(name)])
            .find(|path| path.is_file())
            .unwrap_or_else(|| name.into())
    }

    fn validate_ptxas_and_sass(
        symbol: &str,
        ptx: &str,
        register_cap: u32,
        expected_static_shared: u32,
    ) -> Result<SassStats, String> {
        let stem = format!(
            "mamba-half-tn-one_wave_atlas-{}-{}",
            std::process::id(),
            symbol.len()
        );
        let directory = std::env::temp_dir();
        let ptx_path = directory.join(format!("{stem}.ptx"));
        let cubin_path = directory.join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
        let result = (|| {
            let version = std::process::Command::new(cuda_tool("ptxas"))
                .arg("--version")
                .output()
                .map_err(|error| format!("run ptxas --version: {error}"))?;
            let version_text = format!(
                "{}{}",
                String::from_utf8_lossy(&version.stdout),
                String::from_utf8_lossy(&version.stderr)
            );
            if !version.status.success() || !version_text.contains("release 13.2") {
                return Err(format!("requires exact ptxas 13.2, found {version_text}"));
            }
            let ptxas = std::process::Command::new(cuda_tool("ptxas"))
                .arg("--gpu-name=sm_89")
                .arg("--verbose")
                .arg(&ptx_path)
                .arg("--output-file")
                .arg(&cubin_path)
                .output()
                .map_err(|error| format!("run ptxas: {error}"))?;
            if !ptxas.status.success() {
                return Err(format!(
                    "ptxas failed: {}",
                    String::from_utf8_lossy(&ptxas.stderr)
                ));
            }
            let report = format!(
                "{}\n{}",
                String::from_utf8_lossy(&ptxas.stdout),
                String::from_utf8_lossy(&ptxas.stderr)
            );
            let marker = format!("Compiling entry function '{symbol}'");
            let block = report
                .split_once(&marker)
                .map(|(_, tail)| tail)
                .ok_or_else(|| format!("ptxas omitted {symbol} resource record"))?;
            let metric = |suffix: &str| -> Result<u32, String> {
                block
                    .lines()
                    .find_map(|line| metric_before(line, suffix))
                    .ok_or_else(|| format!("ptxas omitted {symbol} {suffix}"))
            };
            let registers = metric(" registers")?;
            let static_shared = super::ptxas_static_shared_bytes(block);
            for suffix in [
                " bytes stack frame",
                " bytes spill stores",
                " bytes spill loads",
            ] {
                if metric(suffix)? != 0 {
                    return Err(format!("{symbol} has nonzero {suffix}"));
                }
            }
            if registers > register_cap || static_shared != expected_static_shared {
                return Err(format!(
                    "{symbol} resource reject regs={registers}/{register_cap} static={static_shared}/{expected_static_shared}"
                ));
            }
            let cubin = std::fs::read(&cubin_path)
                .map_err(|error| format!("read {symbol} cubin: {error}"))?;
            let disassembly = std::process::Command::new(cuda_tool("nvdisasm"))
                .arg(&cubin_path)
                .output()
                .map_err(|error| format!("run nvdisasm: {error}"))?;
            if !disassembly.status.success() {
                return Err(format!(
                    "nvdisasm failed: {}",
                    String::from_utf8_lossy(&disassembly.stderr)
                ));
            }
            let sass = String::from_utf8_lossy(&disassembly.stdout);
            let entry = sass_entry(&sass, symbol)?;
            let hmma = entry
                .lines()
                .filter(|line| line.contains("HMMA.16816.F32"))
                .count();
            let ldsm = entry.lines().filter(|line| line.contains("LDSM")).count();
            let ldgsts = entry.lines().filter(|line| line.contains("LDGSTS")).count();
            if hmma == 0 || ldsm == 0 || ldgsts == 0 {
                return Err(format!("{symbol} SASS omitted HMMA, LDSM or LDGSTS"));
            }
            for forbidden in [" LDL", " STL", " ATOM", " RED", " REDUX"] {
                if entry.contains(forbidden) {
                    return Err(format!("{symbol} SASS contains {forbidden}"));
                }
            }
            Ok(SassStats {
                registers,
                static_shared,
                hmma,
                ldsm,
                ldgsts,
                cubin_sha: format!("{:x}", Sha256::digest(&cubin)),
                sass_sha: format!("{:x}", Sha256::digest(entry.as_bytes())),
            })
        })();
        let _ = std::fs::remove_file(ptx_path);
        let _ = std::fs::remove_file(cubin_path);
        result
    }

    #[test]
    #[ignore = "CUDA13.2 NVRTC compile-only; creates no device or CUDA context"]
    fn cuda132_nvrtc_source_compile_only() -> Result<(), String> {
        let production = include_str!("../kernels/gemm_bi_triad/sm80.cu");
        let candidate = candidate_source::candidate_source(production)?;
        let retained = candidate_source::retained_source(production)?;
        let (candidate_ptx, candidate_source_sha) =
            compile_ptx_only(&candidate, "OneWaveAtlas candidate compile-only")?;
        let (retained_ptx, retained_source_sha) =
            compile_ptx_only(&retained, "retained vec2 compile-only")?;
        let candidate_ptx = candidate_ptx.to_src();
        let retained_ptx = retained_ptx.to_src();
        let candidate_symbol = format!("{}bf16", candidate_source::SYMBOL_PREFIX);
        let retained_symbol = format!("{}bf16", candidate_source::RETAINED_SYMBOL_PREFIX);
        for anchor in [
            candidate_symbol.as_str(),
            "cp.async.commit_group",
            "cp.async.wait_group 0",
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        ] {
            if !candidate_ptx.contains(anchor) {
                return Err(format!("candidate PTX missing {anchor:?}"));
            }
        }
        for anchor in [
            retained_symbol.as_str(),
            "cp.async.commit_group",
            "cp.async.wait_group 0",
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        ] {
            if !retained_ptx.contains(anchor) {
                return Err(format!("retained PTX missing {anchor:?}"));
            }
        }
        let candidate_sass = validate_ptxas_and_sass(
            &candidate_symbol,
            &candidate_ptx,
            candidate_source::REGISTER_CAP as u32,
            candidate_source::STATIC_SHARED_BYTES as u32,
        )?;
        let retained_sass = validate_ptxas_and_sass(
            &retained_symbol,
            &retained_ptx,
            candidate_source::RETAINED_REGISTER_CAP as u32,
            candidate_source::RETAINED_STATIC_SHARED_BYTES as u32,
        )?;
        if candidate_sass.hmma != candidate_source::EXPECTED_HMMA
            || candidate_sass.ldsm != candidate_source::EXPECTED_LDSM
            || candidate_sass.ldgsts > candidate_source::MAX_LDGSTS
            || retained_sass.hmma != candidate_source::RETAINED_EXPECTED_HMMA
            || retained_sass.ldsm != candidate_source::RETAINED_EXPECTED_LDSM
            || retained_sass.ldgsts != candidate_source::RETAINED_EXPECTED_LDGSTS
        {
            return Err(format!(
                "SASS work reject candidate HMMA/LDSM/LDGSTS={}/{}/{} retained={}/{}/{}",
                candidate_sass.hmma,
                candidate_sass.ldsm,
                candidate_sass.ldgsts,
                retained_sass.hmma,
                retained_sass.ldsm,
                retained_sass.ldgsts,
            ));
        }
        println!(
            "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasCompileV1\",\"candidate_source_sha\":\"{candidate_source_sha}\",\"retained_source_sha\":\"{retained_source_sha}\",\"candidate_ptx_sha\":\"{:x}\",\"retained_ptx_sha\":\"{:x}\",\"candidate_cubin_sha\":\"{}\",\"retained_cubin_sha\":\"{}\",\"candidate_sass_sha\":\"{}\",\"retained_sass_sha\":\"{}\",\"candidate_registers\":{},\"retained_registers\":{},\"static_shared_bytes\":{},\"hmma\":{},\"ldsm\":{},\"ldgsts\":{}}}",
            Sha256::digest(candidate_ptx.as_bytes()),
            Sha256::digest(retained_ptx.as_bytes()),
            candidate_sass.cubin_sha,
            retained_sass.cubin_sha,
            candidate_sass.sass_sha,
            retained_sass.sass_sha,
            candidate_sass.registers,
            retained_sass.registers,
            candidate_sass.static_shared,
            candidate_sass.hmma,
            candidate_sass.ldsm,
            candidate_sass.ldgsts,
        );
        Ok(())
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var("MAMBA_TRIAD_BF16_TN_ONE_WAVE_ATLAS_DISCOVERY").as_deref() != Ok("1") {
            return Err("set MAMBA_TRIAD_BF16_TN_ONE_WAVE_ATLAS_DISCOVERY=1".into());
        }
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (8, 9) || identity.multiprocessor_count != 142 {
            return Err(format!(
                "half TN OneWaveAtlas discovery requires CC8.9/142-SM Ada, got {:?}/{}",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        let compiler = ctx.kernels.compiler_identity();
        if compiler.nvrtc_version != (13, 2) || compiler.target.as_str() != "sm_89" {
            return Err(format!(
                "half TN OneWaveAtlas discovery requires CUDA13.2/sm_89, got {compiler:?}"
            ));
        }
        let production = include_str!("../kernels/gemm_bi_triad/sm80.cu");
        let candidate_source = candidate_source::candidate_source(production)?;
        let retained_source = candidate_source::retained_source(production)?;
        let candidate_symbol = format!("{}bf16", candidate_source::SYMBOL_PREFIX);
        let retained_symbol = format!("{}bf16", candidate_source::RETAINED_SYMBOL_PREFIX);
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_source(
                &device,
                &candidate_source,
                &candidate_symbol,
                candidate_source::DYNAMIC_SHARED_BYTES,
            )?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) =
            compile_source(&device, &retained_source, &retained_symbol, 0)?;
        Ok(Runtime {
            _device: device,
            ctx,
            _candidate_module: candidate_module,
            _retained_module: retained_module,
            candidate,
            retained,
            candidate_source_sha,
            retained_source_sha,
            candidate_ptx_sha,
            retained_ptx_sha,
        })
    }

    fn launch(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let shape = fixture.shape;
        match arm {
            Arm::Candidate | Arm::Retained => {
                let output = fixture.output(arm).ptr();
                let a = fixture.a.ptr();
                let b = fixture.b.ptr();
                let (reduction, k_out, n) =
                    (shape.reduction as i32, shape.k_out as i32, shape.n as i32);
                let function = if arm == Arm::Candidate && shape.uses_atlas() {
                    &runtime.candidate
                } else {
                    &runtime.retained
                };
                let mut builder = runtime.ctx.stream.launch_builder(function);
                builder
                    .arg(&output)
                    .arg(&a)
                    .arg(&b)
                    .arg(&shape.alpha)
                    .arg(&reduction)
                    .arg(&k_out)
                    .arg(&n);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (
                            if arm == Arm::Candidate {
                                shape.candidate_grid()
                            } else {
                                shape.retained_grid()
                            },
                            1,
                            1,
                        ),
                        block_dim: (
                            if arm == Arm::Candidate {
                                shape.candidate_threads()
                            } else {
                                candidate_source::RETAINED_BLOCK_THREADS
                            },
                            1,
                            1,
                        ),
                        shared_mem_bytes: if arm == Arm::Candidate {
                            shape.candidate_dynamic_shared()
                        } else {
                            0
                        },
                    })
                }
                .map(|_| ())
                .map_err(|error| format!("{} launch: {error:?}", arm.name()))
            }
            Arm::Fast => {
                use cudarc::cublas::{result, sys as blas_sys};
                let beta = 1.0f32;
                unsafe {
                    result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas_sys::cublasOperation_t::CUBLAS_OP_N,
                        blas_sys::cublasOperation_t::CUBLAS_OP_T,
                        shape.n as i32,
                        shape.k_out as i32,
                        shape.reduction as i32,
                        (&shape.alpha as *const f32).cast(),
                        fixture.b.ptr() as *const c_void,
                        WeightDtype::Bf16.cuda_data_type(),
                        shape.n as i32,
                        fixture.a.ptr() as *const c_void,
                        WeightDtype::Bf16.cuda_data_type(),
                        shape.k_out as i32,
                        (&beta as *const f32).cast(),
                        fixture.output(arm).ptr() as *mut c_void,
                        blas_sys::cudaDataType::CUDA_R_32F,
                        shape.n as i32,
                        blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .map_err(|error| format!("cuBLAS Fast half TN: {error:?}"))
            }
        }
    }

    fn capture(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
        operations: usize,
    ) -> Result<CudaGraph, String> {
        unsafe {
            capture_into_graph(&runtime.ctx.stream, || {
                for _ in 0..operations {
                    launch(runtime, fixture, arm)?;
                }
                Ok(())
            })
        }
    }

    fn graph_node_count(graph: &CudaGraph, label: &str) -> Result<usize, String> {
        let mut count = 0usize;
        let result =
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
        if result != sys::CUresult::CUDA_SUCCESS || count == 0 {
            return Err(format!("{label} graph inventory {result:?}/{count}"));
        }
        Ok(count)
    }

    fn validate_kernel_graph(graph: &CudaGraph, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let mut count = graph_node_count(graph, arm.name())?;
        if count != 1 {
            return Err(format!("{} exact graph has {count} nodes", arm.name()));
        }
        let mut node = std::ptr::null_mut();
        let nodes = unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) };
        if nodes != sys::CUresult::CUDA_SUCCESS || node.is_null() {
            return Err(format!("{} graph node {nodes:?}", arm.name()));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        let params_result = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) };
        if params_result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("{} graph params {params_result:?}", arm.name()));
        }
        let mut name = std::ptr::null();
        let name_result = unsafe { sys::cuFuncGetName(&mut name, params.func) };
        if name_result != sys::CUresult::CUDA_SUCCESS || name.is_null() {
            return Err(format!("{} graph symbol {name_result:?}", arm.name()));
        }
        let actual = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph symbol UTF-8: {error}"))?;
        let expected = match arm {
            Arm::Candidate if fixture.shape.uses_atlas() => {
                format!("{}bf16", candidate_source::SYMBOL_PREFIX)
            }
            Arm::Candidate => format!("{}bf16", candidate_source::RETAINED_SYMBOL_PREFIX),
            Arm::Retained => format!("{}bf16", candidate_source::RETAINED_SYMBOL_PREFIX),
            Arm::Fast => return Err("Fast graph is vendor-owned".into()),
        };
        let (expected_grid, expected_threads, expected_dynamic_shared) = match arm {
            Arm::Candidate => (
                fixture.shape.candidate_grid(),
                fixture.shape.candidate_threads(),
                fixture.shape.candidate_dynamic_shared(),
            ),
            Arm::Retained => (
                fixture.shape.retained_grid(),
                candidate_source::RETAINED_BLOCK_THREADS,
                0,
            ),
            Arm::Fast => unreachable!(),
        };
        if actual != expected
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != (expected_grid, 1, 1)
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != (expected_threads, 1, 1)
            || params.sharedMemBytes != expected_dynamic_shared
        {
            return Err(format!(
                "{} graph physical mismatch symbol={actual} grid={:?} block={:?} dynamic_shared={}",
                arm.name(),
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (params.blockDimX, params.blockDimY, params.blockDimZ),
                params.sharedMemBytes
            ));
        }
        for (index, expected_info) in [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)]
            .into_iter()
            .enumerate()
        {
            let (mut offset, mut size) = (0usize, 0usize);
            let result =
                unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) };
            if result != sys::CUresult::CUDA_SUCCESS || (offset, size) != expected_info {
                return Err(format!(
                    "{} ABI argument {index}: {result:?} offset={offset} size={size}",
                    arm.name()
                ));
            }
        }
        let (mut offset, mut size) = (0usize, 0usize);
        if unsafe { sys::cuFuncGetParamInfo(params.func, 7, &mut offset, &mut size) }
            != sys::CUresult::CUDA_ERROR_INVALID_VALUE
        {
            return Err(format!("{} accepted an eighth argument", arm.name()));
        }
        if params.kernelParams.is_null() {
            return Err(format!("{} graph arguments are null", arm.name()));
        }
        for index in 0..7 {
            if unsafe { *params.kernelParams.add(index) }.is_null() {
                return Err(format!("{} graph argument {index} is null", arm.name()));
            }
        }
        let expected_pointers = [fixture.output(arm).ptr(), fixture.a.ptr(), fixture.b.ptr()];
        for (index, expected_pointer) in expected_pointers.into_iter().enumerate() {
            let storage = unsafe { *params.kernelParams.add(index) };
            if unsafe { storage.cast::<u64>().read_unaligned() } != expected_pointer {
                return Err(format!(
                    "{} graph pointer argument {index} changed",
                    arm.name()
                ));
            }
        }
        let alpha = unsafe { (*params.kernelParams.add(3)).cast::<f32>().read_unaligned() };
        let reduction = unsafe { (*params.kernelParams.add(4)).cast::<i32>().read_unaligned() };
        let k_out = unsafe { (*params.kernelParams.add(5)).cast::<i32>().read_unaligned() };
        let n = unsafe { (*params.kernelParams.add(6)).cast::<i32>().read_unaligned() };
        if alpha.to_bits() != fixture.shape.alpha.to_bits()
            || (reduction, k_out, n)
                != (
                    fixture.shape.reduction as i32,
                    fixture.shape.k_out as i32,
                    fixture.shape.n as i32,
                )
        {
            return Err(format!("{} graph scalar arguments changed", arm.name()));
        }
        Ok(())
    }

    fn validate_resources(runtime: &Runtime) -> Result<(), String> {
        for (arm, function) in [
            (Arm::Candidate, &runtime.candidate),
            (Arm::Retained, &runtime.retained),
        ] {
            let (expected_static_shared, dynamic_shared, required_occupancy, threads, register_cap) =
                match arm {
                    Arm::Candidate => (
                        candidate_source::STATIC_SHARED_BYTES,
                        candidate_source::DYNAMIC_SHARED_BYTES,
                        candidate_source::REQUIRED_OCCUPANCY,
                        candidate_source::BLOCK_THREADS,
                        candidate_source::REGISTER_CAP,
                    ),
                    Arm::Retained => (
                        candidate_source::RETAINED_STATIC_SHARED_BYTES,
                        0,
                        candidate_source::RETAINED_REQUIRED_OCCUPANCY,
                        candidate_source::RETAINED_BLOCK_THREADS,
                        candidate_source::RETAINED_REGISTER_CAP,
                    ),
                    Arm::Fast => unreachable!(),
                };
            let registers = function
                .num_regs()
                .map_err(|error| format!("{} registers: {error:?}", arm.name()))?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("{} local: {error:?}", arm.name()))?;
            let static_shared = function
                .shared_size_bytes()
                .map_err(|error| format!("{} static shared: {error:?}", arm.name()))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("{} max threads: {error:?}", arm.name()))?;
            let max_dynamic = function
                .get_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                )
                .map_err(|error| format!("{} max dynamic: {error:?}", arm.name()))?;
            let active = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    threads,
                    dynamic_shared as usize,
                    None,
                )
                .map_err(|error| format!("{} occupancy: {error:?}", arm.name()))?;
            if !(1..=register_cap).contains(&registers)
                || local != 0
                || static_shared != expected_static_shared
                || max_dynamic < dynamic_shared as i32
                || max_threads < threads as i32
                || active < required_occupancy
            {
                return Err(format!(
                    "{} resource reject regs={registers}/{} local={local} static={static_shared}/{} dynamic={dynamic_shared}/{max_dynamic} max_threads={max_threads}/{} active={active}/{}",
                    arm.name(),
                    register_cap,
                    expected_static_shared,
                    threads,
                    required_occupancy,
                ));
            }
            println!(
                "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasResourceV1\",\"arm\":\"{}\",\"registers\":{registers},\"register_cap\":{},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{dynamic_shared},\"max_dynamic_shared_bytes\":{max_dynamic},\"block_threads\":{},\"active_ctas\":{active},\"required_active_ctas\":{}}}",
                arm.name(),
                register_cap,
                threads,
                required_occupancy,
            );
        }
        Ok(())
    }

    fn run_and_read(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        graph: Option<&CudaGraph>,
        operations: usize,
    ) -> Result<Vec<u32>, String> {
        fixture.output_mut(arm).reset(&runtime.ctx)?;
        if let Some(graph) = graph {
            graph
                .launch()
                .map_err(|error| format!("{} graph launch: {error:?}", arm.name()))?;
        } else {
            for _ in 0..operations {
                launch(runtime, fixture, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("{} synchronize: {error:?}", arm.name()))?;
        fixture.validate_inputs(runtime)?;
        fixture
            .output(arm)
            .active_and_guards(&runtime.ctx, arm.name())
    }

    fn exact_case(runtime: &Runtime, shape: Shape, exceptional: bool) -> Result<(), String> {
        let mut fixture = Fixture::new(runtime, shape, exceptional)?;
        let retained = run_and_read(runtime, &mut fixture, Arm::Retained, None, 1)?;
        let candidate = run_and_read(runtime, &mut fixture, Arm::Candidate, None, 1)?;
        if candidate != retained {
            let mismatch = candidate
                .iter()
                .zip(&retained)
                .position(|(candidate, retained)| candidate != retained)
                .unwrap_or(candidate.len());
            return Err(format!(
                "{} eager exact mismatch at word {mismatch}",
                shape.label
            ));
        }
        if shape.label == K0.label {
            let expected = fixture
                .seed
                .iter()
                .map(|&bits| (f32::from_bits(bits) + 0.0).to_bits())
                .collect::<Vec<_>>();
            if retained != expected {
                return Err("K0 differs from the retained alpha*zero accumulation contract".into());
            }
        }
        for arm in [Arm::Candidate, Arm::Retained] {
            let graph = capture(runtime, &fixture, arm, 1)?;
            validate_kernel_graph(&graph, &fixture, arm)?;
            for repeat in 0..3 {
                if run_and_read(runtime, &mut fixture, arm, Some(&graph), 1)? != retained {
                    return Err(format!(
                        "{} {} graph exact mismatch repeat={repeat}",
                        shape.label,
                        arm.name()
                    ));
                }
            }
        }
        if shape.label == TARGET.label {
            let candidate_many =
                run_and_read(runtime, &mut fixture, Arm::Candidate, None, OPERATIONS)?;
            let retained_many =
                run_and_read(runtime, &mut fixture, Arm::Retained, None, OPERATIONS)?;
            if candidate_many != retained_many {
                return Err("target 20-op eager candidate/retained mismatch".into());
            }
            for arm in [Arm::Candidate, Arm::Retained] {
                let graph = capture(runtime, &fixture, arm, OPERATIONS)?;
                if graph_node_count(&graph, arm.name())? != OPERATIONS {
                    return Err(format!("{} 20-op graph node count changed", arm.name()));
                }
                let graphed = run_and_read(runtime, &mut fixture, arm, Some(&graph), OPERATIONS)?;
                if graphed != candidate_many {
                    return Err(format!("{} 20-op eager/graph mismatch", arm.name()));
                }
            }
        }
        println!(
            "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasBitsV1\",\"case\":\"{}\",\"shape\":[{},{},{}],\"alpha_bits\":{},\"exceptional\":{exceptional},\"candidate_vs_retained_exact\":true,\"eager_graph_repeats\":3,\"words\":{}}}",
            shape.label,
            shape.reduction,
            shape.k_out,
            shape.n,
            shape.alpha.to_bits(),
            retained.len(),
        );
        Ok(())
    }

    fn event_us(
        runtime: &Runtime,
        operations: usize,
        mut operation: impl FnMut() -> Result<(), String>,
    ) -> Result<f64, String> {
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("timing start: {error:?}"))?;
        operation()?;
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("timing end: {error:?}"))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("timing elapsed: {error:?}"))?,
        ) * 1_000.0
            / operations as f64;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("invalid timing {us}"));
        }
        Ok(us)
    }

    fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err("invalid timing ratios".into());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        Ok(sorted[index])
    }

    fn timed_pair(
        runtime: &Runtime,
        fixture: &mut Fixture,
        comparator: Arm,
        path: Path,
        order: Order,
        windows: usize,
        candidate_bits: &[u32],
        comparator_bits: &[u32],
    ) -> Result<[f64; 2], String> {
        let candidate_graph = match path {
            Path::Eager => None,
            Path::Graph => Some(capture(runtime, fixture, Arm::Candidate, OPERATIONS)?),
        };
        let comparator_graph = match path {
            Path::Eager => None,
            Path::Graph => Some(capture(runtime, fixture, comparator, OPERATIONS)?),
        };
        if let Some(graph) = &candidate_graph
            && graph_node_count(graph, Arm::Candidate.name())? != OPERATIONS
        {
            return Err("candidate timing graph node count changed".into());
        }
        if let Some(graph) = &comparator_graph {
            let count = graph_node_count(graph, comparator.name())?;
            if comparator != Arm::Fast && count != OPERATIONS {
                return Err(format!(
                    "{} timing graph node count changed",
                    comparator.name()
                ));
            }
        }
        for _ in 0..WARMUPS {
            for arm in [Arm::Candidate, comparator] {
                let graph = if arm == Arm::Candidate {
                    candidate_graph.as_ref()
                } else {
                    comparator_graph.as_ref()
                };
                let expected = if arm == Arm::Candidate {
                    candidate_bits
                } else {
                    comparator_bits
                };
                if run_and_read(runtime, fixture, arm, graph, OPERATIONS)? != expected {
                    return Err(format!("{} warmup bits changed", arm.name()));
                }
            }
        }
        let mut ratios = Vec::with_capacity(windows);
        for window in 0..windows {
            let mut raw = [0.0; 4];
            for (index, arm) in order.arms(comparator).into_iter().enumerate() {
                fixture.output_mut(arm).reset(&runtime.ctx)?;
                let graph = if arm == Arm::Candidate {
                    candidate_graph.as_ref()
                } else {
                    comparator_graph.as_ref()
                };
                raw[index] = event_us(runtime, OPERATIONS, || {
                    if let Some(graph) = graph {
                        graph
                            .launch()
                            .map_err(|error| format!("{} timing graph: {error:?}", arm.name()))
                    } else {
                        for _ in 0..OPERATIONS {
                            launch(runtime, fixture, arm)?;
                        }
                        Ok(())
                    }
                })?;
                let expected = if arm == Arm::Candidate {
                    candidate_bits
                } else {
                    comparator_bits
                };
                let actual = fixture
                    .output(arm)
                    .active_and_guards(&runtime.ctx, arm.name())?;
                if actual != expected {
                    return Err(format!("{} timing bits changed", arm.name()));
                }
                fixture.validate_inputs(runtime)?;
            }
            let ratio = match order {
                Order::Abba => (raw[0] + raw[3]) / (raw[1] + raw[2]),
                Order::Baab => (raw[1] + raw[2]) / (raw[0] + raw[3]),
            };
            ratios.push(ratio);
            println!(
                "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasTimingV1\",\"comparator\":\"{}\",\"path\":\"{}\",\"order\":\"{}\",\"windows\":{windows},\"window\":{window},\"operations\":{OPERATIONS},\"raw_us\":{:?},\"ratio\":{ratio}}}",
                comparator.name(),
                path.name(),
                order.name(),
                raw,
            );
        }
        Ok([percentile(&ratios, 0.5)?, percentile(&ratios, 0.95)?])
    }

    fn screen_comparator(
        runtime: &Runtime,
        fixture: &mut Fixture,
        comparator: Arm,
        windows: usize,
        candidate_bits: &[u32],
        comparator_bits: &[u32],
    ) -> Result<Vec<[f64; 2]>, String> {
        let mut strata = Vec::with_capacity(4);
        for path in [Path::Eager, Path::Graph] {
            for order in [Order::Abba, Order::Baab] {
                strata.push(timed_pair(
                    runtime,
                    fixture,
                    comparator,
                    path,
                    order,
                    windows,
                    candidate_bits,
                    comparator_bits,
                )?);
            }
        }
        Ok(strata)
    }

    fn finish_quiet(runtime: Runtime, quiet: &QuietGpu) -> Result<(), String> {
        drop(runtime);
        quiet
            .verify_post_cohort("half-tn-one_wave_atlas-regpipe-vec2/post")
            .map(|_| ())
    }

    #[test]
    #[ignore = "requires exclusive 142-SM Ada and CUDA13.2"]
    fn ada_bf16_tn_one_wave_atlas_regpipe_vec2_d768_out_scout3_then_once7() -> Result<(), String> {
        assert!(!cfg!(debug_assertions), "discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let _pre = quiet.require_pre_context("half-tn-one_wave_atlas-regpipe-vec2/pre-context")?;
        let runtime = new_runtime()?;
        let _cohort = quiet.require_cohort("half-tn-one_wave_atlas-regpipe-vec2/cohort")?;
        validate_resources(&runtime)?;
        exact_case(&runtime, TARGET, false)?;
        exact_case(&runtime, ALIGNED_TAIL, false)?;
        exact_case(&runtime, MISALIGNED_TAIL, false)?;
        exact_case(&runtime, EXCEPTIONAL, true)?;
        exact_case(&runtime, K0, false)?;

        let mut fixture = Fixture::new(&runtime, TARGET, false)?;
        let candidate_bits =
            run_and_read(&runtime, &mut fixture, Arm::Candidate, None, OPERATIONS)?;
        let retained_bits = run_and_read(&runtime, &mut fixture, Arm::Retained, None, OPERATIONS)?;
        if candidate_bits != retained_bits {
            return Err("target candidate differs from retained before timing".into());
        }

        let scout_windows = windows_for(TimingPhase::Scout);
        let scout = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Retained,
            scout_windows,
            &candidate_bits,
            &retained_bits,
        )?;
        let scout_pass = candidate_source::all_strata_below(&scout, SCOUT_THRESHOLD);
        println!(
            "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasScoutDecisionV1\",\"candidate\":\"{}\",\"comparator\":\"{}\",\"windows\":{scout_windows},\"threshold\":{SCOUT_THRESHOLD},\"strata\":{:?},\"pass\":{scout_pass},\"decision\":\"{}\"}}",
            Arm::Candidate.name(),
            Arm::Retained.name(),
            scout,
            if scout_pass {
                "advance_retained_once7"
            } else {
                "stop_no_retry"
            },
        );
        if !scout_pass {
            return finish_quiet(runtime, &quiet);
        }

        let qualification_windows = windows_for(TimingPhase::Qualification);
        let retained_once7 = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Retained,
            qualification_windows,
            &candidate_bits,
            &retained_bits,
        )?;
        let retained_pass =
            candidate_source::all_strata_below(&retained_once7, QUALIFICATION_THRESHOLD);
        println!(
            "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasRetainedDecisionV1\",\"candidate\":\"{}\",\"comparator\":\"{}\",\"windows\":{qualification_windows},\"threshold\":{QUALIFICATION_THRESHOLD},\"strata\":{:?},\"pass\":{retained_pass},\"decision\":\"{}\"}}",
            Arm::Candidate.name(),
            Arm::Retained.name(),
            retained_once7,
            if retained_pass {
                "advance_fast_once7"
            } else {
                "stop_no_fast"
            },
        );
        if !may_run_fast(scout_pass, retained_pass) {
            return finish_quiet(runtime, &quiet);
        }

        let fast_bits = run_and_read(&runtime, &mut fixture, Arm::Fast, None, OPERATIONS)?;
        if !fast_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .all(f32::is_finite)
            || !fast_bits.iter().any(|&bits| bits & 0x7fff_ffff != 0)
        {
            return Err("Fast target output is non-finite or all-zero".into());
        }
        let fast_graph = capture(&runtime, &fixture, Arm::Fast, OPERATIONS)?;
        graph_node_count(&fast_graph, Arm::Fast.name())?;
        for repeat in 0..3 {
            if run_and_read(
                &runtime,
                &mut fixture,
                Arm::Fast,
                Some(&fast_graph),
                OPERATIONS,
            )? != fast_bits
            {
                return Err(format!("Fast eager/graph bits changed repeat={repeat}"));
            }
        }
        let fast_once7 = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Fast,
            qualification_windows,
            &candidate_bits,
            &fast_bits,
        )?;
        let fast_pass = candidate_source::all_strata_below(&fast_once7, QUALIFICATION_THRESHOLD);
        println!(
            "{{\"schema\":\"MambaBiHalfTnOneWaveAtlasFinalDecisionV1\",\"shape\":[2048,1536,768],\"candidate_source_sha\":\"{}\",\"retained_source_sha\":\"{}\",\"candidate_ptx_sha\":\"{}\",\"retained_ptx_sha\":\"{}\",\"scout_strata\":{:?},\"retained_once7_strata\":{:?},\"fast_once7_strata\":{:?},\"strict_fast_p50_p95_lt_0_99\":{fast_pass},\"decision\":\"{}\",\"promotion\":false}}",
            runtime.candidate_source_sha,
            runtime.retained_source_sha,
            runtime.candidate_ptx_sha,
            runtime.retained_ptx_sha,
            scout,
            retained_once7,
            fast_once7,
            if fast_pass {
                "retain_test_only_fast_winner"
            } else {
                "retain_test_only_retained_winner"
            },
        );
        finish_quiet(runtime, &quiet)
    }
}
