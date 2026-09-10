#[path = "support/triad_half_nt_m64n192_s3_source.rs"]
#[allow(dead_code)]
mod candidate_source;

const SWIZZLE: &str = include_str!("../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
const S3: &str = include_str!("../kernels/gemm_bi_inference/sm89_half_s3.cu");

#[test]
fn candidate_source_has_unique_m64n192_geometry() {
    let source = candidate_source::candidate_source(SWIZZLE, S3).expect("compose candidate");
    assert!(source.contains("namespace sm89_test_half_nt_m64n192_s3"));
    assert!(source.contains("static constexpr int kSharedBytes = 98304;"));
    assert!(source.contains("T (*Bs)[192][64]"));
    assert!(source.contains("reinterpret_cast<T (*)[192][64]>"));
    assert!(source.contains("int num_pid_n = (K_out + 191) / 192;"));
    assert!(source.contains("int warpM = (warp / 6) * 32;"));
    assert!(source.contains("int warpN = (warp % 6) * 32;"));
    assert!(source.contains("for (int slice = 0; slice < 6; ++slice)"));
    assert!(source.contains("pid_n * 192 + 192 <= K_out"));
    assert!(source.contains("__launch_bounds__(384, 1)"));
    assert!(source.contains(candidate_source::SYMBOL));
    assert!(!source.contains("void gemm_bi_nt_test_fixed_s3_m64n128_f16"));
}

#[test]
fn candidate_resource_and_target_geometry_are_frozen() {
    assert_eq!(candidate_source::BLOCK_THREADS, 384);
    assert_eq!(candidate_source::DYNAMIC_SHARED_BYTES, 98_304);
    assert_eq!(candidate_source::MAX_REGISTERS, 168);
    let grid = 2_048_u32.div_ceil(64) * 1_536_u32.div_ceil(192);
    assert_eq!(grid, 256);
}

#[test]
fn strict_gate_requires_every_p50_and_p95_below_point99() {
    assert!(candidate_source::all_strata_pass(&[
        [0.98, 0.989],
        [0.981, 0.988],
        [0.982, 0.987],
        [0.983, 0.986],
    ]));
    assert!(!candidate_source::all_strata_pass(&[
        [0.98, 0.989],
        [0.981, 0.990],
        [0.982, 0.987],
        [0.983, 0.986],
    ]));
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

    const GUARD_WORDS: usize = 128;
    const GUARD_BITS: u16 = 0x7e4d;
    const POISON_BITS: u16 = 0x7e31;
    const OPERATIONS: usize = 20;
    const SCOUT_WINDOWS: usize = 3;
    const WINDOWS: usize = 7;
    const TARGET: Shape = Shape::new(2_048, 1_536, 768, 1.0, "d768_out");
    const M_TAIL: Shape = Shape::new(67, 192, 64, -0.75, "m_tail_negative_alpha");
    const N_TAIL: Shape = Shape::new(64, 197, 64, 1.0, "k_out_tail");
    const K_TAIL: Shape = Shape::new(64, 192, 69, 1.0, "reduction_tail");
    const EXCEPTIONAL: Shape = Shape::new(64, 192, 64, 1.0, "exceptional_full_tile");
    const K0: Shape = Shape::new(64, 192, 0, 1.0, "k0");

    #[derive(Clone, Copy)]
    struct Shape {
        m: usize,
        k_out: usize,
        reduction: usize,
        alpha: f32,
        label: &'static str,
    }

    impl Shape {
        const fn new(
            m: usize,
            k_out: usize,
            reduction: usize,
            alpha: f32,
            label: &'static str,
        ) -> Self {
            Self {
                m,
                k_out,
                reduction,
                alpha,
                label,
            }
        }

        fn candidate_grid(self) -> u32 {
            self.m
                .div_ceil(64)
                .checked_mul(self.k_out.div_ceil(192))
                .unwrap() as u32
        }

        fn retained_grid(self) -> u32 {
            self.m
                .div_ceil(64)
                .checked_mul(self.k_out.div_ceil(128))
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
                Self::Candidate => "m64n192_bk64_s3",
                Self::Retained => "measured_m64n128_bk64_s3",
                Self::Fast => "cublas_fast_f16",
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
    }

    struct GuardedHalf {
        buffer: GpuByteBuffer,
        baseline: Vec<u16>,
        active: usize,
    }

    impl GuardedHalf {
        fn new(ctx: &GpuCtx, words: &[u16]) -> Result<Self, String> {
            let mut baseline = vec![GUARD_BITS; GUARD_WORDS + words.len() + GUARD_WORDS];
            baseline[GUARD_WORDS..GUARD_WORDS + words.len()].copy_from_slice(words);
            let mut buffer = GpuByteBuffer::zeros(&ctx.stream, baseline.len() * 2)?;
            buffer.upload_bytes(&ctx.stream, bytemuck::cast_slice(&baseline))?;
            Ok(Self {
                buffer,
                baseline,
                active: words.len(),
            })
        }

        fn ptr(&self) -> u64 {
            self.buffer.cached_ptr() + (GUARD_WORDS * 2) as u64
        }

        fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.buffer
                .upload_bytes(&ctx.stream, bytemuck::cast_slice(&self.baseline))
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

        fn active_and_guards(&self, ctx: &GpuCtx, label: &str) -> Result<Vec<u16>, String> {
            let raw = self.raw(ctx)?;
            if raw[..GUARD_WORDS]
                .iter()
                .chain(&raw[GUARD_WORDS + self.active..])
                .any(|&word| word != GUARD_BITS)
            {
                return Err(format!("{label} changed a redzone"));
            }
            Ok(raw[GUARD_WORDS..GUARD_WORDS + self.active].to_vec())
        }

        fn unchanged(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
            if self.raw(ctx)? != self.baseline {
                return Err(format!("{label} or its redzone changed"));
            }
            Ok(())
        }
    }

    struct Fixture {
        shape: Shape,
        a: GuardedHalf,
        b: GuardedHalf,
        outputs: [GuardedHalf; 3],
    }

    impl Fixture {
        fn new(runtime: &Runtime, shape: Shape, exceptional: bool) -> Result<Self, String> {
            let a = half_corpus(
                shape.m * shape.reduction,
                0xa89a_8201 ^ shape.reduction as u64,
                exceptional,
            );
            let b = half_corpus(
                shape.k_out * shape.reduction,
                0xb89a_8202 ^ shape.k_out as u64,
                exceptional,
            );
            let poison = vec![POISON_BITS; shape.m * shape.k_out];
            let fixture = Self {
                shape,
                a: GuardedHalf::new(&runtime.ctx, &a)?,
                b: GuardedHalf::new(&runtime.ctx, &b)?,
                outputs: [
                    GuardedHalf::new(&runtime.ctx, &poison)?,
                    GuardedHalf::new(&runtime.ctx, &poison)?,
                    GuardedHalf::new(&runtime.ctx, &poison)?,
                ],
            };
            if fixture.a.ptr() % 256 != 0
                || fixture.b.ptr() % 256 != 0
                || fixture.outputs.iter().any(|output| output.ptr() % 256 != 0)
            {
                return Err(format!("{} fixture is not 256-byte aligned", shape.label));
            }
            Ok(fixture)
        }

        fn output(&self, arm: Arm) -> &GuardedHalf {
            &self.outputs[arm as usize]
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedHalf {
            &mut self.outputs[arm as usize]
        }

        fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
            self.a.unchanged(&runtime.ctx, "A")?;
            self.b.unchanged(&runtime.ctx, "B")
        }
    }

    fn half_corpus(len: usize, mut state: u64, exceptional: bool) -> Vec<u16> {
        const SPECIAL: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x3c01, 0xbc01, 0x3555, 0xb555, 0x7bff,
            0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe11,
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
                half::f16::from_f32(value).to_bits()
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
        let fixed_common = strip_typed_include(include_str!("../kernels/gemm_bi_inference/common.cuh"));
        [
            include_str!("../kernels/_typed_prelude.cuh").to_owned(),
            fixed_common,
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
        shared: usize,
    ) -> Result<(Arc<CudaModule>, CudaFunction, String), String> {
        let (ptx, source_sha) = compile_ptx_only(transformed, symbol)?;
        let module = device
            .context()
            .load_module(ptx)
            .map_err(|error| format!("load module {symbol}: {error:?}"))?;
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        Ok((module, function, source_sha))
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
        hmma: usize,
        ldgsts: usize,
        cubin_sha: String,
        sass_sha: String,
    }

    fn cuda_tool(name: &str) -> std::path::PathBuf {
        std::env::var("CUDA_HOME")
            .ok()
            .map(|root| std::path::PathBuf::from(root).join("bin").join(name))
            .into_iter()
            .chain([std::path::PathBuf::from("/usr/local/cuda-13.2/bin").join(name)])
            .find(|path| path.is_file())
            .unwrap_or_else(|| name.into())
    }

    fn validate_ptxas_and_sass(
        symbol: &str,
        ptx: &str,
        register_cap: u32,
    ) -> Result<SassStats, String> {
        let stem = format!(
            "mamba-half-nt-m64n192-{}-{}",
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
            for suffix in [
                " bytes stack frame",
                " bytes spill stores",
                " bytes spill loads",
            ] {
                if metric(suffix)? != 0 {
                    return Err(format!("{symbol} has nonzero {suffix}"));
                }
            }
            if registers > register_cap {
                return Err(format!(
                    "{symbol} register reject {registers}/{register_cap}"
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
            let ldgsts = entry.lines().filter(|line| line.contains("LDGSTS")).count();
            if hmma == 0 || ldgsts == 0 {
                return Err(format!("{symbol} SASS omitted HMMA or LDGSTS"));
            }
            for forbidden in [" LDL", " STL", " ATOM", " RED", " REDUX"] {
                if entry.contains(forbidden) {
                    return Err(format!("{symbol} SASS contains {forbidden}"));
                }
            }
            Ok(SassStats {
                registers,
                hmma,
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
        let swizzle = include_str!("../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
        let s3 = include_str!("../kernels/gemm_bi_inference/sm89_half_s3.cu");
        let candidate = candidate_source::candidate_source(swizzle, s3)?;
        let retained = candidate_source::measured_s3_source(swizzle, s3)?;
        let (candidate_ptx, candidate_source_sha) =
            compile_ptx_only(&candidate, "candidate compile-only")?;
        let (retained_ptx, retained_source_sha) =
            compile_ptx_only(&retained, "retained compile-only")?;
        let candidate_ptx = candidate_ptx.to_src();
        let retained_ptx = retained_ptx.to_src();
        let retained_symbol = "gemm_bi_nt_test_fixed_s3_m64n128_f16";
        if !candidate_ptx.contains(candidate_source::SYMBOL)
            || !retained_ptx.contains(retained_symbol)
        {
            return Err("NVRTC PTX lost candidate or retained export".into());
        }
        for (label, ptx) in [
            ("candidate", candidate_ptx.as_str()),
            ("retained", retained_ptx.as_str()),
        ] {
            for anchor in [
                "cp.async.commit_group",
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ] {
                if !ptx.contains(anchor) {
                    return Err(format!("{label} PTX missing {anchor:?}"));
                }
            }
        }
        let candidate_sass = validate_ptxas_and_sass(
            candidate_source::SYMBOL,
            &candidate_ptx,
            candidate_source::MAX_REGISTERS as u32,
        )?;
        let retained_sass = validate_ptxas_and_sass(retained_symbol, &retained_ptx, 123)?;
        if candidate_sass.hmma != retained_sass.hmma {
            return Err(format!(
                "candidate changed per-warp MMA program: {}/{}",
                candidate_sass.hmma, retained_sass.hmma
            ));
        }
        println!(
            "{{\"schema\":\"MambaBiHalfNtM64N192S3CompileV1\",\"candidate_source_sha\":\"{candidate_source_sha}\",\"retained_source_sha\":\"{retained_source_sha}\",\"candidate_ptx_sha\":\"{:x}\",\"retained_ptx_sha\":\"{:x}\",\"candidate_cubin_sha\":\"{}\",\"retained_cubin_sha\":\"{}\",\"candidate_sass_sha\":\"{}\",\"retained_sass_sha\":\"{}\",\"candidate_registers\":{},\"retained_registers\":{},\"candidate_hmma\":{},\"retained_hmma\":{},\"candidate_ldgsts\":{},\"retained_ldgsts\":{}}}",
            Sha256::digest(candidate_ptx.as_bytes()),
            Sha256::digest(retained_ptx.as_bytes()),
            candidate_sass.cubin_sha,
            retained_sass.cubin_sha,
            candidate_sass.sass_sha,
            retained_sass.sass_sha,
            candidate_sass.registers,
            retained_sass.registers,
            candidate_sass.hmma,
            retained_sass.hmma,
            candidate_sass.ldgsts,
            retained_sass.ldgsts,
        );
        Ok(())
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var("MAMBA_TRIAD_F16_NT_M64N192_S3_DISCOVERY").as_deref() != Ok("1") {
            return Err("set MAMBA_TRIAD_F16_NT_M64N192_S3_DISCOVERY=1".into());
        }
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err(format!(
                "M64N192/S3 discovery requires 142-SM Ada, got {:?}/{}",
                device.compute_capability,
                device.multiprocessor_count()
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        let compiler = ctx.kernels.compiler_identity();
        if compiler.nvrtc_version != (13, 2) || compiler.target.as_str() != "sm_89" {
            return Err(format!(
                "M64N192/S3 discovery requires CUDA13.2/sm_89, got {compiler:?}"
            ));
        }
        let swizzle = include_str!("../kernels/gemm_bi_inference/sm89_half_swizzle.cu");
        let s3 = include_str!("../kernels/gemm_bi_inference/sm89_half_s3.cu");
        let candidate_source = candidate_source::candidate_source(swizzle, s3)?;
        let retained_source = candidate_source::measured_s3_source(swizzle, s3)?;
        let (candidate_module, candidate, candidate_source_sha) = compile_source(
            &device,
            &candidate_source,
            candidate_source::SYMBOL,
            candidate_source::DYNAMIC_SHARED_BYTES,
        )?;
        let (retained_module, retained, retained_source_sha) = compile_source(
            &device,
            &retained_source,
            "gemm_bi_nt_test_fixed_s3_m64n128_f16",
            73_728,
        )?;
        Ok(Runtime {
            _device: device,
            ctx,
            _candidate_module: candidate_module,
            _retained_module: retained_module,
            candidate,
            retained,
            candidate_source_sha,
            retained_source_sha,
        })
    }

    fn launch(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let shape = fixture.shape;
        match arm {
            Arm::Candidate | Arm::Retained => {
                let output = fixture.output(arm).ptr();
                let a = fixture.a.ptr();
                let b = fixture.b.ptr();
                let (m, reduction, k_out) =
                    (shape.m as i32, shape.reduction as i32, shape.k_out as i32);
                let function = if arm == Arm::Candidate {
                    &runtime.candidate
                } else {
                    &runtime.retained
                };
                let shared = if arm == Arm::Candidate {
                    candidate_source::DYNAMIC_SHARED_BYTES
                } else {
                    73_728
                };
                let mut builder = runtime.ctx.stream.launch_builder(function);
                builder
                    .arg(&output)
                    .arg(&a)
                    .arg(&b)
                    .arg(&shape.alpha)
                    .arg(&m)
                    .arg(&reduction)
                    .arg(&k_out);
                let (grid, threads) = if arm == Arm::Candidate {
                    (shape.candidate_grid(), candidate_source::BLOCK_THREADS)
                } else {
                    (shape.retained_grid(), 256)
                };
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: shared as u32,
                    })
                }
                .map(|_| ())
                .map_err(|error| format!("{} launch: {error:?}", arm.name()))
            }
            Arm::Fast => {
                use cudarc::cublas::{result, sys as blas_sys};
                let beta = 0.0f32;
                unsafe {
                    result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas_sys::cublasOperation_t::CUBLAS_OP_T,
                        blas_sys::cublasOperation_t::CUBLAS_OP_N,
                        shape.k_out as i32,
                        shape.m as i32,
                        shape.reduction as i32,
                        (&shape.alpha as *const f32).cast(),
                        fixture.b.ptr() as *const c_void,
                        WeightDtype::F16.cuda_data_type(),
                        shape.reduction as i32,
                        fixture.a.ptr() as *const c_void,
                        WeightDtype::F16.cuda_data_type(),
                        shape.reduction as i32,
                        (&beta as *const f32).cast(),
                        fixture.output(arm).ptr() as *mut c_void,
                        WeightDtype::F16.cuda_data_type(),
                        shape.k_out as i32,
                        blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .map_err(|error| format!("cuBLAS Fast NT: {error:?}"))
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

    fn validate_kernel_graph(graph: &CudaGraph, fixture: &Fixture, arm: Arm) -> Result<(), String> {
        let mut count = 0usize;
        let first =
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
        if first != sys::CUresult::CUDA_SUCCESS || count != 1 {
            return Err(format!("{} graph inventory {first:?}/{count}", arm.name()));
        }
        let mut node = std::ptr::null_mut();
        let second = unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) };
        if second != sys::CUresult::CUDA_SUCCESS || node.is_null() {
            return Err(format!("{} graph node {second:?}", arm.name()));
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
        let expected_symbol = match arm {
            Arm::Candidate => candidate_source::SYMBOL.to_owned(),
            Arm::Retained => "gemm_bi_nt_test_fixed_s3_m64n128_f16".to_owned(),
            Arm::Fast => return Err("Fast graph is not a single candidate kernel".into()),
        };
        let shared = if arm == Arm::Candidate {
            candidate_source::DYNAMIC_SHARED_BYTES
        } else {
            73_728
        };
        let (expected_grid, expected_threads) = if arm == Arm::Candidate {
            (
                fixture.shape.candidate_grid(),
                candidate_source::BLOCK_THREADS,
            )
        } else {
            (fixture.shape.retained_grid(), 256)
        };
        if actual != expected_symbol
            || (params.gridDimX, params.gridDimY, params.gridDimZ) != (expected_grid, 1, 1)
            || (params.blockDimX, params.blockDimY, params.blockDimZ) != (expected_threads, 1, 1)
            || params.sharedMemBytes != shared as u32
        {
            return Err(format!(
                "{} graph physical mismatch symbol={actual} grid={:?} block={:?} shared={}",
                arm.name(),
                (params.gridDimX, params.gridDimY, params.gridDimZ),
                (params.blockDimX, params.blockDimY, params.blockDimZ),
                params.sharedMemBytes
            ));
        }
        for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)]
            .into_iter()
            .enumerate()
        {
            let (mut offset, mut size) = (0usize, 0usize);
            let result =
                unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) };
            if result != sys::CUresult::CUDA_SUCCESS || (offset, size) != expected {
                return Err(format!(
                    "{} ABI argument {index}: result={result:?} offset={offset} size={size}",
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
            return Err(format!("{} graph argument vector is null", arm.name()));
        }
        for index in 0..7 {
            if unsafe { *params.kernelParams.add(index) }.is_null() {
                return Err(format!("{} graph argument {index} is null", arm.name()));
            }
        }
        let expected_pointers = [fixture.output(arm).ptr(), fixture.a.ptr(), fixture.b.ptr()];
        for (index, expected) in expected_pointers.into_iter().enumerate() {
            let storage = unsafe { *params.kernelParams.add(index) };
            if unsafe { storage.cast::<u64>().read_unaligned() } != expected {
                return Err(format!(
                    "{} graph pointer argument {index} changed",
                    arm.name()
                ));
            }
        }
        let alpha = unsafe { (*params.kernelParams.add(3)).cast::<f32>().read_unaligned() };
        let m = unsafe { (*params.kernelParams.add(4)).cast::<i32>().read_unaligned() };
        let reduction = unsafe { (*params.kernelParams.add(5)).cast::<i32>().read_unaligned() };
        let k_out = unsafe { (*params.kernelParams.add(6)).cast::<i32>().read_unaligned() };
        if alpha.to_bits() != fixture.shape.alpha.to_bits()
            || (m, reduction, k_out)
                != (
                    fixture.shape.m as i32,
                    fixture.shape.reduction as i32,
                    fixture.shape.k_out as i32,
                )
        {
            return Err(format!("{} graph scalar arguments changed", arm.name()));
        }
        Ok(())
    }

    fn validate_resources(runtime: &Runtime) -> Result<(), String> {
        for (arm, function, shared, occupancy) in [
            (
                Arm::Candidate,
                &runtime.candidate,
                candidate_source::DYNAMIC_SHARED_BYTES,
                candidate_source::REQUIRED_OCCUPANCY,
            ),
            (Arm::Retained, &runtime.retained, 73_728, 1),
        ] {
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
            let threads = if arm == Arm::Candidate {
                candidate_source::BLOCK_THREADS
            } else {
                256
            };
            let active = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)
                .map_err(|error| format!("{} occupancy: {error:?}", arm.name()))?;
            let max_registers = if arm == Arm::Candidate {
                candidate_source::MAX_REGISTERS
            } else {
                123
            };
            if !(1..=max_registers).contains(&registers)
                || local != 0
                || static_shared != 0
                || max_threads < threads as i32
                || active < occupancy
            {
                return Err(format!(
                    "{} resource reject regs={registers} local={local} static={static_shared} max_threads={max_threads} active={active}/{occupancy}",
                    arm.name()
                ));
            }
            println!(
                "{{\"schema\":\"MambaBiHalfNtM64N192S3ResourceV1\",\"arm\":\"{}\",\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{shared},\"active_ctas\":{active},\"required_active_ctas\":{occupancy}}}",
                arm.name()
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
    ) -> Result<Vec<u16>, String> {
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
        let bits = fixture
            .output(arm)
            .active_and_guards(&runtime.ctx, arm.name())?;
        if fixture.shape.label != EXCEPTIONAL.label && bits.contains(&POISON_BITS) {
            return Err(format!("{} retained output poison", arm.name()));
        }
        Ok(bits)
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
        for arm in [Arm::Candidate, Arm::Retained] {
            let graph = capture(runtime, &fixture, arm, 1)?;
            validate_kernel_graph(&graph, &fixture, arm)?;
            for repeat in 0..3 {
                let bits = run_and_read(runtime, &mut fixture, arm, Some(&graph), 1)?;
                if bits != retained {
                    return Err(format!(
                        "{} {} graph exact mismatch repeat={repeat}",
                        shape.label,
                        arm.name()
                    ));
                }
            }
        }
        if shape.label == TARGET.label {
            for arm in [Arm::Candidate, Arm::Retained] {
                let eager = run_and_read(runtime, &mut fixture, arm, None, OPERATIONS)?;
                let graph = capture(runtime, &fixture, arm, OPERATIONS)?;
                let graphed = run_and_read(runtime, &mut fixture, arm, Some(&graph), OPERATIONS)?;
                if eager != retained || graphed != retained {
                    return Err(format!("{} 20-op repeat mismatch", arm.name()));
                }
            }
        }
        println!(
            "{{\"schema\":\"MambaBiHalfNtM64N192S3BitsV1\",\"case\":\"{}\",\"shape\":[{},{},{}],\"alpha\":{},\"exceptional\":{},\"candidate_vs_measured_s3_exact\":true,\"eager_graph_repeats\":3,\"words\":{}}}",
            shape.label,
            shape.m,
            shape.k_out,
            shape.reduction,
            shape.alpha,
            exceptional,
            retained.len()
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

    fn percentile(values: &[f64], q: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() - 1) as f64 * q).round() as usize]
    }

    fn timed_pair(
        runtime: &Runtime,
        fixture: &mut Fixture,
        comparator: Arm,
        path: Path,
        order: Order,
        candidate_bits: &[u16],
        comparator_bits: &[u16],
        windows: usize,
        phase: &str,
    ) -> Result<[f64; 2], String> {
        let candidate_graph = match path {
            Path::Eager => None,
            Path::Graph => Some(capture(runtime, fixture, Arm::Candidate, OPERATIONS)?),
        };
        let comparator_graph = match path {
            Path::Eager => None,
            Path::Graph => Some(capture(runtime, fixture, comparator, OPERATIONS)?),
        };
        for _ in 0..3 {
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
            let candidate_us = (raw[0] + raw[3]) * 0.5;
            let comparator_us = (raw[1] + raw[2]) * 0.5;
            let ratio = match order {
                Order::Abba => candidate_us / comparator_us,
                Order::Baab => (raw[1] + raw[2]) / (raw[0] + raw[3]),
            };
            ratios.push(ratio);
            println!(
                "{{\"schema\":\"MambaBiHalfNtM64N192S3TimingV1\",\"phase\":\"{phase}\",\"comparator\":\"{}\",\"path\":\"{}\",\"order\":\"{}\",\"window\":{window},\"operations\":{OPERATIONS},\"raw_us\":{:?},\"ratio\":{ratio}}}",
                comparator.name(),
                path.name(),
                order.name(),
                raw
            );
        }
        Ok([percentile(&ratios, 0.5), percentile(&ratios, 0.95)])
    }

    fn screen_comparator(
        runtime: &Runtime,
        fixture: &mut Fixture,
        comparator: Arm,
        candidate_bits: &[u16],
        comparator_bits: &[u16],
        windows: usize,
        phase: &str,
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
                    candidate_bits,
                    comparator_bits,
                    windows,
                    phase,
                )?);
            }
        }
        Ok(strata)
    }

    #[test]
    #[ignore = "requires exclusive 142-SM Ada and CUDA13.2"]
    fn ada_f16_nt_m64n192_bk64_s3_d768_out_scout_once7() -> Result<(), String> {
        assert!(!cfg!(debug_assertions), "discovery requires --release");
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let _pre = quiet.require_pre_context("half-nt-m64n192-s3/pre-context")?;
        let runtime = new_runtime()?;
        let _cohort = quiet.require_cohort("half-nt-m64n192-s3/cohort")?;
        validate_resources(&runtime)?;
        exact_case(&runtime, TARGET, false)?;
        exact_case(&runtime, M_TAIL, false)?;
        exact_case(&runtime, N_TAIL, false)?;
        exact_case(&runtime, K_TAIL, false)?;
        exact_case(&runtime, EXCEPTIONAL, true)?;
        exact_case(&runtime, K0, false)?;

        let mut fixture = Fixture::new(&runtime, TARGET, false)?;
        let candidate_bits = run_and_read(&runtime, &mut fixture, Arm::Candidate, None, 1)?;
        let retained_bits = run_and_read(&runtime, &mut fixture, Arm::Retained, None, 1)?;
        if candidate_bits != retained_bits {
            return Err("target candidate differs from measured S3 before timing".into());
        }
        let scout_strata = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Retained,
            &candidate_bits,
            &retained_bits,
            SCOUT_WINDOWS,
            "scout_once3",
        )?;
        let scout_pass = candidate_source::all_strata_pass(&scout_strata);
        println!(
            "{{\"schema\":\"MambaBiHalfNtM64N192S3ScoutDecisionV1\",\"shape\":[2048,1536,768],\"strata\":{:?},\"strict_p50_p95_lt_0_99\":{scout_pass},\"decision\":\"{}\"}}",
            scout_strata,
            if scout_pass {
                "advance_to_once7"
            } else {
                "stop_no_once7"
            }
        );
        if !scout_pass {
            drop(runtime);
            quiet.verify_post_cohort("half-nt-m64n192-s3/post")?;
            return Ok(());
        }
        let retained_strata = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Retained,
            &candidate_bits,
            &retained_bits,
            WINDOWS,
            "retained_once7",
        )?;
        let retained_pass = candidate_source::all_strata_pass(&retained_strata);
        println!(
            "{{\"schema\":\"MambaBiHalfNtM64N192S3RetainedDecisionV1\",\"shape\":[2048,1536,768],\"candidate\":\"{}\",\"comparator\":\"{}\",\"strata\":{:?},\"strict_p50_p95_lt_0_99\":{retained_pass},\"decision\":\"{}\"}}",
            Arm::Candidate.name(),
            Arm::Retained.name(),
            retained_strata,
            if retained_pass {
                "advance_to_fast"
            } else {
                "stop_no_fast"
            }
        );
        if !retained_pass {
            drop(runtime);
            quiet.verify_post_cohort("half-nt-m64n192-s3/post")?;
            return Ok(());
        }

        let fast_bits = run_and_read(&runtime, &mut fixture, Arm::Fast, None, 1)?;
        if !fast_bits
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .all(f32::is_finite)
            || !fast_bits.iter().any(|&bits| bits & 0x7fff != 0)
        {
            return Err("Fast output is non-finite or all-zero".into());
        }
        let fast_graph = capture(&runtime, &fixture, Arm::Fast, 1)?;
        for _ in 0..3 {
            if run_and_read(&runtime, &mut fixture, Arm::Fast, Some(&fast_graph), 1)? != fast_bits {
                return Err("Fast eager/graph output is not deterministic".into());
            }
        }
        let fast_strata = screen_comparator(
            &runtime,
            &mut fixture,
            Arm::Fast,
            &candidate_bits,
            &fast_bits,
            WINDOWS,
            "fast_once7",
        )?;
        let fast_pass = candidate_source::all_strata_pass(&fast_strata);
        println!(
            "{{\"schema\":\"MambaBiHalfNtM64N192S3FinalDecisionV1\",\"shape\":[2048,1536,768],\"candidate_source_sha\":\"{}\",\"retained_source_sha\":\"{}\",\"retained_strata\":{:?},\"fast_strata\":{:?},\"strict_fast_p50_p95_lt_0_99\":{fast_pass},\"decision\":\"{}\",\"promotion\":false}}",
            runtime.candidate_source_sha,
            runtime.retained_source_sha,
            retained_strata,
            fast_strata,
            if fast_pass {
                "retain_test_only"
            } else {
                "stop_no_retry"
            }
        );
        drop(runtime);
        quiet.verify_post_cohort("half-nt-m64n192-s3/post")?;
        Ok(())
    }
}
