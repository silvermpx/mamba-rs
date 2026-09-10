//! Test-only Ada TF32 NN canonical-Prism direct-N96 grid-constant discovery.
//! Production kernels and dispatch remain untouched.

#[path = "support/triad_tf32_nn_n96_direct_grid_constant_source.rs"]
#[allow(dead_code)]
mod candidate_source;
#[path = "support/triad_tf32_nn_n96_direct_epilogue_source.rs"]
#[allow(dead_code)]
mod direct_source;
#[path = "support/triad_nn_n96_source.rs"]
#[allow(dead_code)]
mod retained_source;

const FIXED_N96_SOURCE: &str = include_str!("../kernels/gemm_bi_inference/tf32_rna_n96.cu");

fn strict_once7_win(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .flatten()
            .all(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < 0.99)
}

fn strict_fast_win(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata
            .iter()
            .flatten()
            .all(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < 0.99)
}

#[test]
fn direct_float2_mapping_covers_full_m128n96_tile_once() {
    for (warp, m_atom, n_atom, half, lane, expected) in [
        (0, 0, 0, 0, 0, ((0, 0), (0, 1))),
        (0, 0, 0, 1, 31, ((15, 6), (15, 7))),
        (3, 2, 1, 0, 5, ((33, 82), (33, 83))),
        (7, 3, 2, 1, 30, ((127, 92), (127, 93))),
    ] {
        assert_eq!(
            candidate_source::direct_pair_coordinates(warp, m_atom, n_atom, half, lane),
            expected,
        );
    }

    let mut owners = vec![0_u8; 128 * 96];
    for warp in 0..8 {
        for m_atom in 0..4 {
            for n_atom in 0..3 {
                for half in 0..2 {
                    for lane in 0..32 {
                        let ((row, column0), (row1, column1)) =
                            candidate_source::direct_pair_coordinates(
                                warp, m_atom, n_atom, half, lane,
                            );
                        assert_eq!(row, row1);
                        assert_eq!(column1, column0 + 1);
                        for column in [column0, column1] {
                            assert!(row < 128 && column < 96);
                            owners[row * 96 + column] += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(owners.iter().all(|count| *count == 1));
}

fn direct_retained_source() -> String {
    candidate_source::retained_source(FIXED_N96_SOURCE).unwrap()
}

#[test]
fn source_adapter_is_reversible_and_changes_only_params_residency() {
    let retained = direct_retained_source();
    let candidate = candidate_source::candidate_source(FIXED_N96_SOURCE).unwrap();
    assert!(candidate.contains("const __grid_constant__ GbfTf32N96Params params"));
    assert!(candidate.contains("const GbfTf32N96Params& params"));
    assert!(candidate.contains("static_assert(sizeof(GbfTf32N96Params) == 32"));
    assert!(candidate.contains("static_assert(alignof(GbfTf32N96Params) == 4"));
    assert!(candidate.contains("bool direct_pair_epilogue ="));
    assert!(candidate.contains("return bits + 0x1000U;"));
    assert!(candidate.contains("make_float2("));
    assert_eq!(
        candidate_source::restore_retained_source(&candidate).unwrap(),
        retained,
    );
}

#[test]
fn prism_grid_params_abi_and_resource_budget_are_frozen() {
    assert_eq!(candidate_source::grid(4_621, 1_928), 777);
    assert_eq!(direct_source::n96_grid(4_621, 1_928), 777);
    assert_eq!(candidate_source::TILE, (128, 96, 32));
    assert_eq!(candidate_source::BLOCK_THREADS, 256);
    assert_eq!(candidate_source::STAGES, 3);
    assert_eq!(candidate_source::K8_ISSUE_OFFSETS, [0, 8, 16, 24]);
    assert_eq!(candidate_source::PARAM_BYTES, 32);
    assert_eq!(candidate_source::PARAM_ALIGNMENT, 4);
    assert_eq!(candidate_source::DYNAMIC_SHARED_BYTES, 86_016);
    assert!(candidate_source::DYNAMIC_SHARED_BYTES <= 101_376);
    assert_eq!(candidate_source::MAX_REGISTERS, 124);
    assert_eq!(candidate_source::REQUIRED_OCCUPANCY, 1);
}

#[test]
fn transform_rejects_non_parent_and_duplicate_seams() {
    assert!(candidate_source::candidate_source("not the frozen parent").is_err());
    assert!(candidate_source::restore_retained_source("not the frozen candidate").is_err());
}

#[test]
fn logical_nn_targets_have_the_frozen_n96_grids() {
    assert_eq!(direct_source::n96_grid(4_621, 1_928), 777);
}

#[test]
fn fast_screen_requires_four_strict_retained_strata() {
    assert!(strict_once7_win(&[[0.98, 0.989]; 4]));
    assert!(!strict_once7_win(&[[0.98, 0.99]; 4]));
    assert!(!strict_once7_win(&[[0.98, 0.989]; 3]));
    assert!(!strict_once7_win(&[
        [0.98, 0.989],
        [0.98, 0.989],
        [f64::NAN, 0.989],
        [0.98, 0.989],
    ]));
    assert!(strict_fast_win(&[[0.98, 0.989]; 4]));
    assert!(!strict_fast_win(&[[0.98, 0.99]; 4]));
}

#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;

#[cfg(feature = "cuda")]
mod cuda_suite {
    use super::*;
    use cudarc::{
        cublas::{result as blas_result, sys as blas},
        driver::{
            sys, CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig,
            PushKernelArg,
        },
    };
    use mamba_rs::mamba_ssm::gpu::{
        buffers::GpuBuffer,
        context::{BiGemmFamily, F32TriadPolicy, GpuCtx},
        device::GpuDevice,
        dtype::WeightDtype,
        graph_capture::capture_into_graph,
        kernels::cuda_include_paths,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        ffi::{c_void, CStr},
        sync::Arc,
    };

    const ENV: &str = "MAMBA_TRIAD_TF32_NN_N96_DIRECT_GRID_CONSTANT_PRISM_DISCOVERY";
    const CANDIDATE_SHARED: usize = candidate_source::DYNAMIC_SHARED_BYTES;
    const RETAINED_SHARED: usize = 86_016;
    const OPS: usize = 20;
    const WARMUPS: usize = 8;
    const GUARD: usize = 64;
    const ALIGNMENT: u64 = 256;
    const GUARD_BITS: u32 = 0x7fc0_b196;
    const POISON_BITS: u32 = 0x7fc0_c196;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Shape {
        m: usize,
        k: usize,
        n: usize,
    }

    impl Shape {
        fn grid(self, arm: Arm) -> Result<(u32, u32, u32), String> {
            let blocks = match arm {
                Arm::Candidate => candidate_source::grid(self.m, self.n),
                Arm::Retained => direct_source::n96_grid(self.m, self.n),
                Arm::Fast => return Err("Fast has no custom grid".into()),
            };
            Ok((
                u32::try_from(blocks).map_err(|_| "N96 grid exceeds u32")?,
                1,
                1,
            ))
        }

        fn config(self, arm: Arm) -> Result<LaunchConfig, String> {
            let (threads, shared) = match arm {
                Arm::Candidate => (candidate_source::BLOCK_THREADS, CANDIDATE_SHARED),
                Arm::Retained => (256, RETAINED_SHARED),
                Arm::Fast => return Err("Fast has no custom launch config".into()),
            };
            Ok(LaunchConfig {
                grid_dim: self.grid(arm)?,
                block_dim: (threads, 1, 1),
                shared_mem_bytes: shared as u32,
            })
        }
    }

    #[derive(Clone, Copy)]
    struct Case {
        label: &'static str,
        shape: Shape,
    }

    const PRISM: Case = Case {
        label: "prism_in_proj",
        shape: Shape {
            m: 4_621,
            k: 384,
            n: 1_928,
        },
    };
    const FULL_TILE: Case = Case {
        label: "full_tile_128x32x96",
        shape: Shape {
            m: 128,
            k: 32,
            n: 96,
        },
    };
    const TAIL: Case = Case {
        label: "tail_129x36x100",
        shape: Shape {
            m: 129,
            k: 36,
            n: 100,
        },
    };
    const K0: Case = Case {
        label: "k0_129x0x100",
        shape: Shape {
            m: 129,
            k: 0,
            n: 100,
        },
    };

    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Params {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }
    unsafe impl DeviceRepr for Params {}

    impl Params {
        fn new(shape: Shape) -> Result<Self, String> {
            Ok(Self {
                alpha: 1.0,
                beta: 0.0,
                m: i32::try_from(shape.m).map_err(|_| "M exceeds i32")?,
                k: i32::try_from(shape.k).map_err(|_| "K exceeds i32")?,
                n: i32::try_from(shape.n).map_err(|_| "N exceeds i32")?,
                lda: i32::try_from(shape.k).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(shape.n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(shape.n).map_err(|_| "ldc exceeds i32")?,
            })
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
                Self::Candidate => "n96_direct_grid_constant",
                Self::Retained => "direct_epilogue_n96",
                Self::Fast => "cublas_fast_tf32",
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

    struct GuardedF32 {
        buffer: GpuBuffer,
        baseline: Vec<f32>,
        len: usize,
        label: &'static str,
    }

    impl GuardedF32 {
        fn new(
            stream: &Arc<CudaStream>,
            active: Vec<f32>,
            label: &'static str,
        ) -> Result<Self, String> {
            let len = active.len();
            let mut baseline = vec![f32::from_bits(GUARD_BITS); GUARD + len + GUARD];
            baseline[GUARD..GUARD + len].copy_from_slice(&active);
            let buffer = GpuBuffer::from_cpu(stream, &baseline)?;
            let value = Self {
                buffer,
                baseline,
                len,
                label,
            };
            if value.buffer.cached_ptr() % ALIGNMENT != 0 || value.ptr(stream) % ALIGNMENT != 0 {
                return Err(format!(
                    "{} base/logical pointer is not 256B aligned",
                    value.label
                ));
            }
            Ok(value)
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, GUARD)
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.baseline)
        }

        fn bits(&self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            if values[..GUARD]
                .iter()
                .chain(&values[GUARD + self.len..])
                .any(|value| value.to_bits() != GUARD_BITS)
            {
                return Err(format!("{} red zone changed", self.label));
            }
            Ok(values[GUARD..GUARD + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn unchanged(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            let values = self.buffer.to_cpu(stream)?;
            if values
                .iter()
                .zip(&self.baseline)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("{} input or guard changed", self.label));
            }
            Ok(())
        }
    }

    struct Fixture {
        a: GuardedF32,
        b: GuardedF32,
        candidate: GuardedF32,
        retained: GuardedF32,
        fast: GuardedF32,
    }

    impl Fixture {
        fn new(runtime: &Runtime, case: Case, exceptional: bool) -> Result<Self, String> {
            let shape = case.shape;
            let mut a = full_mantissa::finite_full_mantissa_values(shape.m * shape.k, 0xb196_a001);
            let mut b = full_mantissa::finite_full_mantissa_values(shape.k * shape.n, 0xb196_b002);
            if exceptional && shape.k > 0 {
                a.fill(0.0);
                b.fill(0.0);
                for row in 0..shape.m {
                    a[row * shape.k] = 1.0;
                }
                for (column, bits) in [
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0001,
                    0x8000_0001,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7f80_0001,
                    0x7fc1_2345,
                    0x7fa1_2345,
                ]
                .into_iter()
                .enumerate()
                {
                    if column < shape.n {
                        b[column] = f32::from_bits(bits);
                    }
                }
            }
            let output = vec![f32::from_bits(POISON_BITS); shape.m * shape.n];
            Ok(Self {
                a: GuardedF32::new(&runtime.ctx.stream, a, "A")?,
                b: GuardedF32::new(&runtime.ctx.stream, b, "B")?,
                candidate: GuardedF32::new(&runtime.ctx.stream, output.clone(), "candidate C")?,
                retained: GuardedF32::new(&runtime.ctx.stream, output.clone(), "retained C")?,
                fast: GuardedF32::new(&runtime.ctx.stream, output, "Fast C")?,
            })
        }

        fn output(&self, arm: Arm) -> &GuardedF32 {
            match arm {
                Arm::Candidate => &self.candidate,
                Arm::Retained => &self.retained,
                Arm::Fast => &self.fast,
            }
        }

        fn output_mut(&mut self, arm: Arm) -> &mut GuardedF32 {
            match arm {
                Arm::Candidate => &mut self.candidate,
                Arm::Retained => &mut self.retained,
                Arm::Fast => &mut self.fast,
            }
        }

        fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
            self.a.unchanged(&runtime.ctx.stream)?;
            self.b.unchanged(&runtime.ctx.stream)
        }
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _retained_module: Arc<CudaModule>,
        _candidate_module: Arc<CudaModule>,
        retained: CudaFunction,
        candidate: CudaFunction,
        retained_source_sha: String,
        candidate_source_sha: String,
        retained_ptx_sha: String,
        candidate_ptx_sha: String,
    }

    fn module_source(body: &str) -> String {
        let prelude = include_str!("../kernels/_typed_prelude.cuh");
        let common = include_str!("../kernels/gemm_bi_inference/common.cuh")
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n");
        let tf32 = include_str!("../kernels/gemm_bi_inference/tf32.cu");
        [prelude, &common, tf32, body].join("\n")
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

    fn metric_before(line: &str, suffix: &str) -> Option<u32> {
        line.split_once(suffix)?
            .0
            .split_whitespace()
            .next_back()?
            .parse()
            .ok()
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

    #[derive(Clone, Debug)]
    struct SassStats {
        text_bytes: usize,
        instruction_sha: String,
        hmma: usize,
        ldgsts: usize,
        registers: u32,
        static_shared: u32,
        stack_bytes: u32,
        spill_store_bytes: u32,
        spill_load_bytes: u32,
        forbidden: bool,
    }

    fn ptxas_and_sass(symbol: &str, ptx: &str, shared: usize) -> Result<SassStats, String> {
        let stem = format!(
            "mamba-n96-grid-constant-{}-{}",
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
                .map_err(|error| format!("ptxas --version: {error}"))?;
            let version_text = format!(
                "{}{}",
                String::from_utf8_lossy(&version.stdout),
                String::from_utf8_lossy(&version.stderr),
            );
            if !version.status.success() || !version_text.contains("release 13.2") {
                return Err(format!("requires exact ptxas 13.2, found {version_text}"));
            }
            let output = std::process::Command::new(cuda_tool("ptxas"))
                .arg("--gpu-name=sm_89")
                .arg("--verbose")
                .arg(&ptx_path)
                .arg("--output-file")
                .arg(&cubin_path)
                .output()
                .map_err(|error| format!("run ptxas: {error}"))?;
            if !output.status.success() {
                return Err(format!(
                    "ptxas failed: {}",
                    String::from_utf8_lossy(&output.stderr),
                ));
            }
            let report = format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            let marker = format!("Compiling entry function '{symbol}'");
            if report.matches(&marker).count() != 1 {
                return Err(format!("ptxas ambiguous {symbol} resource record"));
            }
            let tail = report
                .split_once(&marker)
                .map(|(_, tail)| tail)
                .ok_or_else(|| format!("ptxas omitted {symbol} resource record"))?;
            let block = tail
                .split_once("Compiling entry function '")
                .map_or(tail, |(entry, _)| entry);
            let unique_metric = |metric: &str| -> Result<u32, String> {
                let values = block
                    .lines()
                    .filter_map(|line| metric_before(line, metric))
                    .collect::<Vec<_>>();
                match values.as_slice() {
                    [value] => Ok(*value),
                    _ => Err(format!("{symbol} ambiguous {metric}: {values:?}")),
                }
            };
            let registers = unique_metric(" registers")?;
            let static_shared_values = block
                .lines()
                .filter_map(|line| metric_before(line, " bytes smem"))
                .collect::<Vec<_>>();
            let static_shared = match static_shared_values.as_slice() {
                [] => 0,
                [value] => *value,
                _ => {
                    return Err(format!(
                        "{symbol} ambiguous static shared bytes: {static_shared_values:?}"
                    ));
                }
            };
            let disassembly = std::process::Command::new(cuda_tool("nvdisasm"))
                .arg(&cubin_path)
                .output()
                .map_err(|error| format!("run nvdisasm: {error}"))?;
            if !disassembly.status.success() {
                return Err(format!(
                    "nvdisasm failed: {}",
                    String::from_utf8_lossy(&disassembly.stderr),
                ));
            }
            let sass = String::from_utf8_lossy(&disassembly.stdout);
            let entry = sass_entry(&sass, symbol)?;
            let instructions = entry
                .lines()
                .filter(|line| line.trim_start().starts_with("/*"))
                .count();
            let instruction_text = entry
                .lines()
                .filter(|line| line.trim_start().starts_with("/*"))
                .collect::<Vec<_>>()
                .join("\n");
            let stats = SassStats {
                text_bytes: instructions * 16,
                instruction_sha: format!("{:x}", Sha256::digest(instruction_text.as_bytes())),
                hmma: entry.lines().filter(|line| line.contains("HMMA")).count(),
                ldgsts: entry.lines().filter(|line| line.contains("LDGSTS")).count(),
                registers,
                static_shared,
                stack_bytes: unique_metric(" bytes stack frame")?,
                spill_store_bytes: unique_metric(" bytes spill stores")?,
                spill_load_bytes: unique_metric(" bytes spill loads")?,
                forbidden: [" LDL", " STL", " SHFL", " ATOM", " RED", " REDUX"]
                    .into_iter()
                    .any(|opcode| entry.contains(opcode)),
            };
            if stats.hmma == 0 || stats.ldgsts == 0 || instructions == 0 {
                return Err(format!("{symbol} lost HMMA/LDGSTS/executable text"));
            }
            println!(
                "{}",
                json!({
                    "schema":"MambaBiTriadTf32NnN96DirectGridConstantSassV1",
                    "symbol":symbol,"target":"sm_89","text_bytes":stats.text_bytes,
                    "instruction_sha":stats.instruction_sha,
                    "hmma":stats.hmma,"ldgsts":stats.ldgsts,"registers":stats.registers,
                    "static_shared_bytes":stats.static_shared,"dynamic_shared_bytes":shared,
                    "stack_bytes":stats.stack_bytes,"spill_store_bytes":stats.spill_store_bytes,
                    "spill_load_bytes":stats.spill_load_bytes,"forbidden":stats.forbidden,
                }),
            );
            Ok(stats)
        })();
        let _ = std::fs::remove_file(ptx_path);
        let _ = std::fs::remove_file(cubin_path);
        result
    }

    fn compile_ptx(body: &str, label: &str) -> Result<String, String> {
        cudarc::nvrtc::compile_ptx_with_opts(
            module_source(body),
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec![
                    "--fmad=true".into(),
                    "--extra-device-vectorization".into(),
                    "-DNDEBUG".into(),
                ],
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| format!("compile {label}: {error:?}"))
    }

    fn compile_only_gate() -> Result<(), String> {
        let retained_body = candidate_source::retained_source(FIXED_N96_SOURCE)?;
        let candidate_body = candidate_source::candidate_source(FIXED_N96_SOURCE)?;
        let retained = ptxas_and_sass(
            direct_source::SYMBOL,
            &compile_ptx(&retained_body, "direct-float2 retained")?,
            RETAINED_SHARED,
        )?;
        let candidate = ptxas_and_sass(
            candidate_source::SYMBOL,
            &compile_ptx(&candidate_body, "N96 direct grid-constant candidate")?,
            CANDIDATE_SHARED,
        )?;
        let text_ratio = candidate.text_bytes as f64 / retained.text_bytes as f64;
        let hmma_ratio = candidate.hmma as f64 / retained.hmma as f64;
        let ldgsts_ratio = candidate.ldgsts as f64 / retained.ldgsts as f64;
        let sass_distinct = candidate.instruction_sha != retained.instruction_sha;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantCompileDecisionV1",
                "candidate_text_bytes":candidate.text_bytes,"retained_text_bytes":retained.text_bytes,
                "candidate_instruction_sha":candidate.instruction_sha,
                "retained_instruction_sha":retained.instruction_sha,
                "sass_distinct":sass_distinct,"text_ratio":text_ratio,"text_ratio_max":1.0,
                "hmma_ratio":hmma_ratio,"ldgsts_ratio":ldgsts_ratio,
                "register_limit":candidate_source::MAX_REGISTERS,"timing_started":false,
            }),
        );
        if !sass_distinct
            || candidate.registers == 0
            || candidate.registers > candidate_source::MAX_REGISTERS as u32
            || candidate.registers > retained.registers
            || candidate.static_shared != 0
            || candidate.stack_bytes != 0
            || candidate.spill_store_bytes != 0
            || candidate.spill_load_bytes != 0
            || candidate.forbidden
            || retained.static_shared != 0
            || retained.stack_bytes != 0
            || retained.spill_store_bytes != 0
            || retained.spill_load_bytes != 0
            || retained.forbidden
            || candidate.text_bytes > retained.text_bytes
            || candidate.hmma != retained.hmma
            || candidate.ldgsts != retained.ldgsts
        {
            return Err(format!(
                "compile-only stop: sass_distinct={sass_distinct}, text={text_ratio:.4}, HMMA={hmma_ratio:.4}, LDGSTS={ldgsts_ratio:.4}, regs={}/{}, shared={}, stack={}, spills={}/{}, forbidden={}",
                candidate.registers,
                retained.registers,
                candidate.static_shared,
                candidate.stack_bytes,
                candidate.spill_store_bytes,
                candidate.spill_load_bytes,
                candidate.forbidden,
            ));
        }
        Ok(())
    }

    fn compile_module(
        device: &GpuDevice,
        source: String,
        symbol: &str,
        shared: usize,
    ) -> Result<(Arc<CudaModule>, CudaFunction, String, String), String> {
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
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("compile {symbol}: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load {symbol} module: {error:?}"))?;
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        function
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared as i32,
            )
            .map_err(|error| format!("set {symbol} dynamic shared: {error:?}"))?;
        Ok((module, function, source_sha, ptx_sha))
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var(ENV).as_deref() != Ok("1") {
            return Err(format!("set {ENV}=1"));
        }
        if cfg!(debug_assertions) {
            return Err("N96 direct grid-constant timing requires --release".into());
        }
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err(format!(
                "requires CC8.9/142 SM, found {:?}/{}",
                device.compute_capability,
                device.multiprocessor_count()
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
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
                return Err("cuBLAS Fast handle mode is not supported".into());
            }
        }
        let compiler = ctx.kernels.compiler_identity();
        if compiler.nvrtc_version != (13, 2)
            || compiler.target.as_str() != "sm_89"
            || !compiler.nvrtc_library_known
        {
            return Err(format!("wrong N96 discovery compiler: {compiler:?}"));
        }
        let retained_body = candidate_source::retained_source(FIXED_N96_SOURCE)?;
        let candidate_body = candidate_source::candidate_source(FIXED_N96_SOURCE)?;
        let (retained_module, retained, retained_source_sha, retained_ptx_sha) = compile_module(
            &device,
            module_source(&retained_body),
            direct_source::SYMBOL,
            RETAINED_SHARED,
        )?;
        let (candidate_module, candidate, candidate_source_sha, candidate_ptx_sha) =
            compile_module(
                &device,
                module_source(&candidate_body),
                candidate_source::SYMBOL,
                CANDIDATE_SHARED,
            )?;
        let runtime = Runtime {
            _device: device,
            ctx,
            _retained_module: retained_module,
            _candidate_module: candidate_module,
            retained,
            candidate,
            retained_source_sha,
            candidate_source_sha,
            retained_ptx_sha,
            candidate_ptx_sha,
        };
        resource_gate(
            &runtime.retained,
            direct_source::SYMBOL,
            128,
            256,
            RETAINED_SHARED,
            1,
        )?;
        resource_gate(
            &runtime.candidate,
            candidate_source::SYMBOL,
            candidate_source::MAX_REGISTERS,
            candidate_source::BLOCK_THREADS,
            CANDIDATE_SHARED,
            candidate_source::REQUIRED_OCCUPANCY,
        )?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantIdentityV1",
                "retained_symbol":direct_source::SYMBOL,
                "candidate_symbol":candidate_source::SYMBOL,
                "retained_source_sha":runtime.retained_source_sha,
                "candidate_source_sha":runtime.candidate_source_sha,
                "retained_ptx_sha":runtime.retained_ptx_sha,
                "candidate_ptx_sha":runtime.candidate_ptx_sha,
                "conversion":"add_half_ulp_tf32_v1",
                "change":"grid_constant_params_with_const_ref_threading",
            })
        );
        Ok(runtime)
    }

    fn resource_gate(
        function: &CudaFunction,
        symbol: &str,
        register_limit: i32,
        threads: u32,
        shared: usize,
        required_occupancy: u32,
    ) -> Result<(), String> {
        let registers = function
            .num_regs()
            .map_err(|e| format!("{symbol} regs: {e:?}"))?;
        let local = function
            .local_size_bytes()
            .map_err(|e| format!("{symbol} local: {e:?}"))?;
        let static_shared = function
            .shared_size_bytes()
            .map_err(|e| format!("{symbol} shared: {e:?}"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|e| format!("{symbol} max threads: {e:?}"))?;
        let max_dynamic = function
            .get_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            )
            .map_err(|e| format!("{symbol} max dynamic: {e:?}"))?;
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)
            .map_err(|e| format!("{symbol} occupancy: {e:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantResourceV1","symbol":symbol,
                "threads":threads,"registers":registers,"local_bytes":local,
                "static_shared_bytes":static_shared,"dynamic_shared_bytes":shared,
                "max_threads_per_block":max_threads,"occupancy":occupancy,
            })
        );
        if registers <= 0
            || registers > register_limit
            || local != 0
            || static_shared != 0
            || max_threads < threads as i32
            || max_dynamic < shared as i32
            || occupancy < required_occupancy
        {
            return Err(format!(
                "{symbol} resource gate failed: regs={registers} local={local} static={static_shared} max_dynamic={max_dynamic} occupancy={occupancy}"
            ));
        }
        Ok(())
    }

    fn launch(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
    ) -> Result<(), String> {
        let shape = case.shape;
        let output = fixture.output(arm).ptr(&runtime.ctx.stream);
        let (a, b) = if shape.k == 0 {
            (0, 0)
        } else {
            (
                fixture.a.ptr(&runtime.ctx.stream),
                fixture.b.ptr(&runtime.ctx.stream),
            )
        };
        match arm {
            Arm::Candidate | Arm::Retained => {
                let function = if arm == Arm::Candidate {
                    &runtime.candidate
                } else {
                    &runtime.retained
                };
                let bias = 0_u64;
                let params = Params::new(shape)?;
                let mut builder = runtime.ctx.stream.launch_builder(function);
                builder.arg(&output).arg(&a).arg(&b).arg(&bias).arg(&params);
                unsafe { builder.launch(shape.config(arm)?) }
                    .map(|_| ())
                    .map_err(|e| format!("launch {}: {e:?}", arm.name()))
            }
            Arm::Fast => {
                if shape.k == 0 {
                    return Err("Fast comparator excludes K0".into());
                }
                let alpha = 1.0_f32;
                let beta = 0.0_f32;
                let dtype = WeightDtype::F32.cuda_data_type();
                unsafe {
                    blas_result::gemm_ex(
                        *runtime.ctx.blas.handle(),
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        blas::cublasOperation_t::CUBLAS_OP_N,
                        shape.n as i32,
                        shape.m as i32,
                        shape.k as i32,
                        (&alpha as *const f32).cast::<c_void>(),
                        b as *const c_void,
                        dtype,
                        shape.n as i32,
                        a as *const c_void,
                        dtype,
                        shape.k as i32,
                        (&beta as *const f32).cast::<c_void>(),
                        output as *mut c_void,
                        dtype,
                        shape.n as i32,
                        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                        blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
                .map_err(|e| format!("cuBLAS Fast TF32: {e:?}"))
            }
        }
    }

    fn capture(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        launch(runtime, fixture, case, arm)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} warmup sync: {e:?}", arm.name()))?;
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, case, arm)) }
    }

    fn graph_identity(graph: &CudaGraph, case: Case, arm: Arm) -> Result<(), String> {
        unsafe {
            let mut count = 0;
            if sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                != sys::CUresult::CUDA_SUCCESS
                || count == 0
            {
                return Err(format!("{} graph is empty/unqueryable", arm.name()));
            }
            if arm == Arm::Fast {
                println!(
                    "{}",
                    json!({"schema":"MambaBiTriadTf32NnN96DirectGridConstantGraphV1","arm":arm.name(),"nodes":count,"abi":"opaque","timing":"whole_graph"})
                );
                return Ok(());
            }
            if count != 1 {
                return Err(format!("{} graph has {count} nodes", arm.name()));
            }
            let mut node = std::ptr::null_mut();
            if sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count)
                != sys::CUresult::CUDA_SUCCESS
            {
                return Err(format!("{} graph node query failed", arm.name()));
            }
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
            if sys::cuGraphKernelNodeGetParams_v2(node, &mut params) != sys::CUresult::CUDA_SUCCESS
            {
                return Err(format!("{} graph params unavailable", arm.name()));
            }
            let mut name_ptr = std::ptr::null();
            if sys::cuFuncGetName(&mut name_ptr, params.func) != sys::CUresult::CUDA_SUCCESS
                || name_ptr.is_null()
            {
                return Err(format!("{} graph symbol unavailable", arm.name()));
            }
            let symbol = CStr::from_ptr(name_ptr).to_string_lossy();
            let expected_symbol = if arm == Arm::Candidate {
                candidate_source::SYMBOL
            } else {
                direct_source::SYMBOL
            };
            let expected_config = case.shape.config(arm)?;
            let expected_grid = expected_config.grid_dim;
            if symbol != expected_symbol
                || (params.gridDimX, params.gridDimY, params.gridDimZ) != expected_grid
                || (params.blockDimX, params.blockDimY, params.blockDimZ)
                    != expected_config.block_dim
                || params.sharedMemBytes != expected_config.shared_mem_bytes
            {
                return Err(format!(
                    "{} graph identity changed: {symbol} grid={:?} block={:?} shared={}",
                    arm.name(),
                    (params.gridDimX, params.gridDimY, params.gridDimZ),
                    (params.blockDimX, params.blockDimY, params.blockDimZ),
                    params.sharedMemBytes
                ));
            }
            println!(
                "{}",
                json!({"schema":"MambaBiTriadTf32NnN96DirectGridConstantGraphV1","arm":arm.name(),"symbol":symbol,"grid":expected_grid,"block":expected_config.block_dim,"shared":expected_config.shared_mem_bytes,"timing":"whole_graph"})
            );
        }
        Ok(())
    }

    fn output_after(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
        graph: Option<&CudaGraph>,
        repeats: usize,
    ) -> Result<Vec<u32>, String> {
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        for _ in 0..repeats {
            if let Some(graph) = graph {
                graph
                    .launch()
                    .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?;
            } else {
                launch(runtime, fixture, case, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} sync: {e:?}", arm.name()))?;
        let bits = fixture.output(arm).bits(&runtime.ctx.stream)?;
        if bits.iter().any(|word| *word == POISON_BITS) {
            return Err(format!("{} left an unwritten output word", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        Ok(bits)
    }

    fn check_exact_case(runtime: &Runtime, case: Case, exceptional: bool) -> Result<(), String> {
        let mut fixture = Fixture::new(runtime, case, exceptional)?;
        let golden = output_after(runtime, &mut fixture, case, Arm::Retained, None, 1)?;
        let candidate_graph = capture(runtime, &mut fixture, case, Arm::Candidate)?;
        let retained_graph = capture(runtime, &mut fixture, case, Arm::Retained)?;
        graph_identity(&candidate_graph, case, Arm::Candidate)?;
        graph_identity(&retained_graph, case, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (Arm::Candidate, &candidate_graph),
                (Arm::Retained, &retained_graph),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, case, arm, path, 1)? != golden {
                        return Err(format!(
                            "{} {} repeat {repeat} changed exact bits",
                            case.label,
                            arm.name()
                        ));
                    }
                }
            }
        }
        if case.shape.k == 0 && golden.iter().any(|word| *word != 0) {
            return Err(format!(
                "{} alpha1/beta0/no-bias K0 oracle is not positive zero",
                case.label
            ));
        }
        println!(
            "{}",
            json!({"schema":"MambaBiTriadTf32NnN96DirectGridConstantBitsV1","case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],"exceptional":exceptional,"candidate_retained_exact":true,"eager_repeats":2,"graph_repeats":2,"guards":true})
        );
        Ok(())
    }

    struct Prepared {
        fixture: Fixture,
        candidate_graph: CudaGraph,
        retained_graph: CudaGraph,
        fast_graph: Option<CudaGraph>,
        exact_bits: Vec<u32>,
        fast_bits: Option<Vec<u32>>,
    }

    fn prepare_target(runtime: &Runtime, case: Case) -> Result<Prepared, String> {
        let mut fixture = Fixture::new(runtime, case, false)?;
        let exact_bits = output_after(runtime, &mut fixture, case, Arm::Retained, None, 1)?;
        if output_after(runtime, &mut fixture, case, Arm::Candidate, None, 1)? != exact_bits {
            return Err(format!(
                "{} candidate differs from direct-float2 retained N96",
                case.label
            ));
        }
        let candidate_graph = capture(runtime, &mut fixture, case, Arm::Candidate)?;
        let retained_graph = capture(runtime, &mut fixture, case, Arm::Retained)?;
        graph_identity(&candidate_graph, case, Arm::Candidate)?;
        graph_identity(&retained_graph, case, Arm::Retained)?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (Arm::Candidate, &candidate_graph),
                (Arm::Retained, &retained_graph),
            ] {
                for path in [None, Some(graph)] {
                    if output_after(runtime, &mut fixture, case, arm, path, 1)? != exact_bits {
                        return Err(format!(
                            "{} {} {} repeat {repeat} changed bits",
                            case.label,
                            arm.name(),
                            if path.is_some() { "graph" } else { "eager" }
                        ));
                    }
                }
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantTargetBitsV1",
                "case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],
                "candidate_retained_exact":true,"fast_deferred_until_retained_win":true,
                "eager_repeats":2,"graph_repeats":2,"guards":true,
            })
        );
        Ok(Prepared {
            fixture,
            candidate_graph,
            retained_graph,
            fast_graph: None,
            exact_bits,
            fast_bits: None,
        })
    }

    fn prepare_fast(runtime: &Runtime, case: Case, prepared: &mut Prepared) -> Result<(), String> {
        let fast_bits = output_after(runtime, &mut prepared.fixture, case, Arm::Fast, None, 1)?;
        if fast_bits.is_empty()
            || fast_bits.iter().all(|word| word & 0x7fff_ffff == 0)
            || fast_bits
                .iter()
                .any(|word| !f32::from_bits(*word).is_finite())
        {
            return Err(format!("{} Fast comparator output invalid", case.label));
        }
        let fast_graph = capture(runtime, &mut prepared.fixture, case, Arm::Fast)?;
        graph_identity(&fast_graph, case, Arm::Fast)?;
        for repeat in 0..2 {
            for path in [None, Some(&fast_graph)] {
                if output_after(runtime, &mut prepared.fixture, case, Arm::Fast, path, 1)?
                    != fast_bits
                {
                    return Err(format!(
                        "{} Fast {} repeat {repeat} changed bits",
                        case.label,
                        if path.is_some() { "graph" } else { "eager" }
                    ));
                }
            }
        }
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantFastBitsV1",
                "case":case.label,"shape":[case.shape.m,case.shape.k,case.shape.n],
                "fast_self_consistent":true,"eager_repeats":2,"graph_repeats":2,
                "guards":true,"after_retained_win":true,
            })
        );
        prepared.fast_bits = Some(fast_bits);
        prepared.fast_graph = Some(fast_graph);
        Ok(())
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: Arm,
        graph: &CudaGraph,
        path: Path,
        golden: &[u32],
    ) -> Result<f64, String> {
        fixture.output_mut(arm).reset(&runtime.ctx.stream)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("{} start event: {e:?}", arm.name()))?;
        for _ in 0..OPS {
            match path {
                Path::Eager => launch(runtime, fixture, case, arm)?,
                Path::Graph => graph
                    .launch()
                    .map_err(|e| format!("{} graph launch: {e:?}", arm.name()))?,
            }
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("{} end event: {e:?}", arm.name()))?;
        let us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("{} elapsed: {e:?}", arm.name()))?,
        ) * 1_000.0
            / OPS as f64;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("{} timed sync: {e:?}", arm.name()))?;
        if fixture.output(arm).bits(&runtime.ctx.stream)? != golden {
            return Err(format!("{} timed bits changed", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        if !us.is_finite() || us <= 0.0 {
            return Err(format!("{} invalid sample {us}", arm.name()));
        }
        Ok(us)
    }

    fn quantile(values: &[f64], q: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() as f64 * q).ceil() as usize).saturating_sub(1)]
    }

    fn screen(
        runtime: &Runtime,
        case: Case,
        prepared: &mut Prepared,
        comparator: Arm,
        path: Path,
        order: retained_source::BracketOrder,
        windows: usize,
    ) -> Result<[f64; 2], String> {
        let (comparator_graph, comparator_bits) = match comparator {
            Arm::Retained => (&prepared.retained_graph, prepared.exact_bits.as_slice()),
            Arm::Fast => (
                prepared
                    .fast_graph
                    .as_ref()
                    .ok_or("Fast graph requested before retained win")?,
                prepared
                    .fast_bits
                    .as_deref()
                    .ok_or("Fast bits requested before retained win")?,
            ),
            Arm::Candidate => return Err("candidate cannot be its own comparator".into()),
        };
        for _ in 0..WARMUPS {
            measure(
                runtime,
                &mut prepared.fixture,
                case,
                Arm::Candidate,
                &prepared.candidate_graph,
                path,
                &prepared.exact_bits,
            )?;
            measure(
                runtime,
                &mut prepared.fixture,
                case,
                comparator,
                comparator_graph,
                path,
                comparator_bits,
            )?;
        }
        let candidate_slots = order.candidate_slots();
        let mut raw = Vec::with_capacity(windows);
        let mut ratios = Vec::with_capacity(windows);
        for _ in 0..windows {
            let mut observation = [0.0; 4];
            for (index, candidate_slot) in candidate_slots.into_iter().enumerate() {
                let (arm, graph, bits) = if candidate_slot {
                    (
                        Arm::Candidate,
                        &prepared.candidate_graph,
                        prepared.exact_bits.as_slice(),
                    )
                } else {
                    (comparator, comparator_graph, comparator_bits)
                };
                observation[index] =
                    measure(runtime, &mut prepared.fixture, case, arm, graph, path, bits)?;
            }
            ratios.push(retained_source::candidate_over_auto_ratio(
                order,
                observation,
            )?);
            raw.push(observation);
        }
        let result = [quantile(&ratios, 0.5), quantile(&ratios, 0.95)];
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantScreenV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],"grid":case.shape.grid(Arm::Candidate)?,
                "candidate":"n96_direct_grid_constant","comparator":comparator.name(),"path":path.name(),
                "order":order.name(),"windows":windows,"warmups_per_arm":WARMUPS,
                "logical_gemms_per_observation":OPS,"raw_observations_us":raw,
                "ratio_direction":"candidate_over_comparator","ratio_p50":result[0],"ratio_p95":result[1],
            })
        );
        Ok(result)
    }

    fn screen_all(
        runtime: &Runtime,
        case: Case,
        prepared: &mut Prepared,
        comparator: Arm,
        windows: usize,
    ) -> Result<Vec<[f64; 2]>, String> {
        let mut strata = Vec::with_capacity(4);
        for path in [Path::Eager, Path::Graph] {
            for order in [
                retained_source::BracketOrder::Abba,
                retained_source::BracketOrder::Baab,
            ] {
                strata.push(screen(
                    runtime, case, prepared, comparator, path, order, windows,
                )?);
            }
        }
        Ok(strata)
    }

    #[test]
    #[ignore = "requires CUDA13.2 NVRTC, ptxas, and nvdisasm; launches no GPU context or GPU work"]
    fn cuda132_compile_only_n96_direct_grid_constant_resources_and_sass() -> Result<(), String> {
        compile_only_gate()
    }

    #[test]
    #[ignore = "requires exclusive quiet CC8.9/142-SM CUDA13.2; isolated TF32 NN N96 direct grid-constant Prism once3 -> once7 -> Fast"]
    fn ada_tf32_nn_n96_direct_grid_constant_prism_protocol() -> Result<(), String> {
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        compile_only_gate()?;
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        let idle_resident_mode = common::gpu_quiet::idle_resident_mode_enabled();
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-nn-n96-direct-grid-constant/pre", 1_800)?;
        } else {
            quiet.require_pre_context("tf32-nn-n96-direct-grid-constant/pre")?;
        }
        let runtime = new_runtime()?;
        assert_eq!(PRISM.shape.grid(Arm::Candidate)?, (777, 1, 1));
        assert_eq!(PRISM.shape.grid(Arm::Retained)?, (777, 1, 1));
        check_exact_case(&runtime, FULL_TILE, false)?;
        check_exact_case(&runtime, FULL_TILE, true)?;
        check_exact_case(&runtime, TAIL, false)?;
        check_exact_case(&runtime, TAIL, true)?;
        check_exact_case(&runtime, K0, false)?;
        let case = PRISM;
        let mut prepared = prepare_target(&runtime, case)?;
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-nn-n96-direct-grid-constant/retained", 256)?;
        } else {
            quiet.require_cohort("tf32-nn-n96-direct-grid-constant/retained")?;
        }
        let scout_strata = screen_all(&runtime, case, &mut prepared, Arm::Retained, 3)?;
        let scout_win = strict_once7_win(&scout_strata);
        let retained_strata = if scout_win {
            screen_all(&runtime, case, &mut prepared, Arm::Retained, 7)?
        } else {
            Vec::new()
        };
        let retained_win = scout_win && strict_once7_win(&retained_strata);
        let fast_strata = if retained_win {
            prepare_fast(&runtime, case, &mut prepared)?;
            if idle_resident_mode {
                quiet.require_idle_resident("tf32-nn-n96-direct-grid-constant/fast", 256)?;
            } else {
                quiet.require_cohort("tf32-nn-n96-direct-grid-constant/fast")?;
            }
            screen_all(&runtime, case, &mut prepared, Arm::Fast, 7)?
        } else {
            Vec::new()
        };
        let fast_win = retained_win && strict_fast_win(&fast_strata);
        println!(
            "{}",
            json!({
                "schema":"MambaBiTriadTf32NnN96DirectGridConstantDecisionV1","cell":case.label,
                "shape":[case.shape.m,case.shape.k,case.shape.n],
                "candidate_grid":case.shape.grid(Arm::Candidate)?,
                "retained_grid":case.shape.grid(Arm::Retained)?,
                "scout_windows":3,"scout_strata":scout_strata,
                "once7_windows":7,"retained_strata":retained_strata,"fast_strata":fast_strata,
                "strata_order":["eager/ABBA","eager/BAAB","graph/ABBA","graph/BAAB"],
                "threshold":0.99,"scout_win":scout_win,"retained_win":retained_win,
                "fast_win":fast_win,"decision":if !scout_win {
                    "stop_after_once3"
                } else if !retained_win {
                    "stop_after_once7"
                } else if fast_win {
                    "shortlist_strict_fast_win"
                } else {
                    "retain_candidate_fast_miss"
                },
                "fast_screened":retained_win,"fast_qualified":fast_win,
                "idle_resident_mode":idle_resident_mode,"promotion":false,
            })
        );
        drop(prepared);
        if idle_resident_mode {
            quiet.require_idle_resident("tf32-nn-n96-direct-grid-constant/post", 256)?;
        } else {
            quiet.verify_post_cohort("tf32-nn-n96-direct-grid-constant/post")?;
        }
        Ok(())
    }
}
