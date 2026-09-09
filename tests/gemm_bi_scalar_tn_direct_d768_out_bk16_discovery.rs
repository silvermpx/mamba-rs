//! Discovery only: one direct-TN BK16 exact raw-partial launch for canonical
//! Prism plus the unchanged production split-M reducer. No dispatcher admission
//! or production changes.

#[path = "support/triad_f32_tn_direct_d768_out_bk16_source.rs"]
mod direct_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    batch: usize,
    k_out: usize,
    n_out: usize,
}

const PRISM: Cell = Cell {
    batch: 2048,
    k_out: 1536,
    n_out: 768,
};

const M_CHUNK: usize = 512;
const CHUNKS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chunk {
    begin: usize,
    len: usize,
    partial_offset: usize,
}

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
    fn chunks(self) -> Result<Vec<Chunk>, String> {
        if self.batch == 0 {
            return Ok(Vec::new());
        }
        let chunks = self.batch.div_ceil(M_CHUNK);
        if chunks != CHUNKS {
            return Err(format!(
                "split-M successor requires exactly {CHUNKS} chunks, got {chunks}"
            ));
        }
        Ok((0..chunks)
            .map(|index| {
                let begin = index * M_CHUNK;
                Chunk {
                    begin,
                    len: (self.batch - begin).min(M_CHUNK),
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

    fn partial_grid(self) -> (u32, u32, u32) {
        (
            (self.k_out.div_ceil(128) * self.n_out.div_ceil(128)) as u32,
            1,
            CHUNKS as u32,
        )
    }

    fn reducer_grid(self) -> (u32, u32, u32) {
        ((self.k_out * self.n_out).div_ceil(256) as u32, 1, 1)
    }

    fn candidate_node_count(self) -> usize {
        if self.batch == 0 {
            1
        } else {
            2
        }
    }

    fn retained_node_count(self) -> usize {
        if self.batch == 0 {
            1
        } else {
            CHUNKS + 2
        }
    }

    fn direct_grid(self) -> (u32, u32, u32) {
        let mut grid = self.fixed_grid();
        grid.2 = CHUNKS as u32;
        grid
    }

    fn transposed_a_index(self, batch_row: usize, k_column: usize) -> usize {
        assert!(batch_row < self.batch && k_column < self.k_out);
        k_column * self.batch + batch_row
    }
}

fn exact_splitm_oracle(cell: Cell, alpha: f32, output: &[f32], x: &[f32], dy: &[f32]) -> Vec<u32> {
    assert_eq!(output.len(), cell.k_out * cell.n_out);
    assert_eq!(x.len(), cell.batch * cell.k_out);
    assert_eq!(dy.len(), cell.batch * cell.n_out);
    let mut result = Vec::with_capacity(output.len());
    for k in 0..cell.k_out {
        for n in 0..cell.n_out {
            if cell.batch == 0 {
                result.push(alpha.mul_add(0.0, output[k * cell.n_out + n]).to_bits());
                continue;
            }
            let mut partials = Vec::new();
            for chunk in cell.chunks().unwrap() {
                let mut accumulator = 0.0_f32;
                for m in chunk.begin..chunk.begin + chunk.len {
                    accumulator =
                        x[m * cell.k_out + k].mul_add(dy[m * cell.n_out + n], accumulator);
                }
                partials.push(accumulator);
            }
            let sum = partials.into_iter().map(f64::from).sum::<f64>();
            let update = (f64::from(alpha) * sum) as f32;
            result.push((output[k * cell.n_out + n] + update).to_bits());
        }
    }
    result
}

#[test]
fn tn_d768_out_bk16_maps_four_exact_chunks_to_two_node_candidate() {
    assert_eq!(
        (size_of::<FixedParams>(), align_of::<FixedParams>()),
        (32, 4)
    );
    let chunks = PRISM.chunks().unwrap();
    assert_eq!(
        chunks,
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
    for p in chunks.into_iter().map(|chunk| PRISM.fixed_params(chunk)) {
        assert_eq!(
            (p.alpha.to_bits(), p.beta.to_bits()),
            (1.0_f32.to_bits(), 0)
        );
        assert_eq!((p.m, p.n, p.k), (1536, 768, 512));
        assert_eq!((p.lda, p.ldb, p.ldc), (2048, 768, 768));
    }
    assert_eq!(PRISM.transpose_grid(), (48, 64, 1));
    assert_eq!(PRISM.fixed_grid(), (288, 1, 1));
    assert_eq!(PRISM.partial_grid(), (72, 1, 4));
    assert_eq!(PRISM.reducer_grid(), (4608, 1, 1));
    assert_eq!(PRISM.direct_grid(), (288, 1, 4));
    assert_eq!(PRISM.candidate_node_count(), 2);
    assert_eq!(PRISM.retained_node_count(), 6);
    assert_eq!(
        Cell {
            batch: 0,
            k_out: 68,
            n_out: 132,
        }
        .candidate_node_count(),
        1
    );
    for (batch_row, k_column, expected) in [
        (0, 0, 0),
        (2047, 0, 2047),
        (0, 1535, 1535 * 2048),
        (2047, 1535, 1536 * 2048 - 1),
    ] {
        assert_eq!(PRISM.transposed_a_index(batch_row, k_column), expected);
    }
    let tail = Cell {
        batch: 2047,
        k_out: 68,
        n_out: 132,
    };
    assert_eq!(tail.chunks().unwrap()[3].len, 511);
    let direct = direct_source::compose_source();
    assert_eq!(direct.matches(direct_source::SYMBOL).count(), 1);
}

#[test]
fn independent_literal_oracle_keeps_four_f32_chains_then_f64_reduce() {
    let cell = Cell {
        batch: 2048,
        k_out: 2,
        n_out: 2,
    };
    let output = [0.25, -0.5, 0.75, -1.0];
    let mut x = vec![0.0; cell.batch * cell.k_out];
    let mut dy = vec![0.0; cell.batch * cell.n_out];
    for (chunk, value) in [1.0_f32, 2.0, 3.0, 4.0].into_iter().enumerate() {
        let row = chunk * M_CHUNK;
        x[row * 2] = value;
        dy[row * 2] = 1.0;
    }
    assert_eq!(
        exact_splitm_oracle(cell, -0.75, &output, &x, &dy),
        [
            (0.25_f32 + -0.75 * 10.0).to_bits(),
            (-0.5_f32).to_bits(),
            0.75_f32.to_bits(),
            (-1.0_f32).to_bits()
        ]
    );

    let zero = Cell {
        batch: 0,
        k_out: 1,
        n_out: 2,
    };
    let zero_seed = [-0.0_f32, f32::from_bits(0x7fc1_2345)];
    assert_eq!(
        exact_splitm_oracle(zero, 1.0, &zero_seed, &[], &[]),
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

#[test]
fn cuda_harness_stops_in_order_after_exact_resources_once3_and_once7() {
    let harness = include_str!("gemm_bi_scalar_tn_direct_d768_out_bk16_discovery.rs");
    assert!(harness.contains("validate_ptxas_and_sass(DIRECT"));
    assert!(harness.contains("registers > DIRECT_MAX_REGISTERS as i32"));
    assert!(harness.contains("occupancy < DIRECT_MIN_OCCUPANCY"));
    assert!(
        harness.contains("for forbidden in [\" LDL\", \" STL\", \" ATOM\", \" RED\", \" REDUX\"]")
    );
    assert!(harness.contains("let retained_once3_win = screen_comparator("));
    assert!(harness.contains("let retained_once7_win = screen_comparator("));
    assert!(harness.contains("Arm::Retained,\n            3,"));
    assert!(harness.contains("Arm::Retained,\n            7,"));
    assert!(harness.contains("if retained_once7_win"));
    assert!(harness.contains("Arm::Fast,\n                7,"));
}

#[test]
fn actual_auto_launch_uses_the_fixture_batch() {
    let harness = include_str!("gemm_bi_scalar_tn_direct_d768_out_bk16_discovery.rs");
    let launch = &harness[harness.find("    fn launch(\n").unwrap()
        ..harness.find("    fn compare_raw_partials(\n").unwrap()];
    assert_eq!(
        launch.matches("fixture.cell.batch,").count(),
        2,
        "both actual AUTO launches must forward the real fixture batch, including K0"
    );
}

#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
#[path = "support/fixed_full_mantissa.rs"]
mod full_mantissa;

#[cfg(feature = "cuda")]
mod splitm_cuda_suite {
    use super::*;
    use cudarc::cublas::{result as blas_result, sys as blas};
    use cudarc::driver::{
        sys, CudaFunction, CudaGraph, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg,
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
        ffi::{c_void, CStr},
        sync::Arc,
    };

    const FIXED: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1";
    const DIRECT: &str = direct_source::SYMBOL;
    const TRANSPOSE: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
    const PARTIAL: &str = "gemm_bi_tn_splitm_partial_aligned";
    const REDUCER: &str = "gemm_bi_splitm_reduce";
    const ZERO: &str = "gemm_bi_tn_zero_reduction_v1";
    const GUARD: usize = 64;
    const GUARD_BITS: u32 = 0x7fc0_3189;
    const OPS: usize = 20;
    const DIRECT_MAX_REGISTERS: u32 = 128;
    const DIRECT_STATIC_SHARED: usize = 16_384;
    const DIRECT_MIN_OCCUPANCY: u32 = 4;

    unsafe impl DeviceRepr for FixedParams {}

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
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Reference,
        Retained,
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
        alpha: f32,
        output: Buffer,
        x: Buffer,
        dy: Buffer,
        transposed: Buffer,
        candidate_partial: Buffer,
        reference_partial: Buffer,
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
                    0xffc5_4321,
                    0xffa5_4321,
                ];
                for (case, bits) in cases.into_iter().enumerate() {
                    for row in (0..CHUNKS).map(|chunk| chunk * M_CHUNK + case) {
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
                CHUNKS * cell.k_out * cell.n_out
            };
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
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _direct_module: Arc<CudaModule>,
        direct: CudaFunction,
        direct_source_sha: String,
    }

    fn config(grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared,
        }
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

    #[test]
    fn sass_entry_accepts_cuda132_text_labels() {
        let sass = concat!(
            "//--------------------- .text.target --------------------------\n",
            ".section .text.target,\"ax\",@progbits\n",
            ".text.target:\n",
            " FFMA R0, R1, R2, R3;\n",
            "//--------------------- .text.next --------------------------\n",
            ".text.next:\n",
            " EXIT;\n",
        );
        let entry = sass_entry(sass, "target").unwrap();
        assert!(entry.contains("FFMA"));
        assert!(!entry.contains(".text.next"));
    }

    fn validate_ptxas_and_sass(symbol: &str, ptx: &str) -> Result<(), String> {
        let stem = format!("mamba-f32-tn-prism-bk16-{}", std::process::id());
        let directory = std::env::temp_dir();
        let ptx_path = directory.join(format!("{stem}.ptx"));
        let cubin_path = directory.join(format!("{stem}.cubin"));
        std::fs::write(&ptx_path, ptx).map_err(|error| format!("write PTX: {error}"))?;
        let result = (|| {
            let ptxas = std::process::Command::new("ptxas")
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
            for metric in [
                " bytes stack frame",
                " bytes spill stores",
                " bytes spill loads",
            ] {
                let value = block
                    .lines()
                    .find_map(|line| metric_before(line, metric))
                    .ok_or_else(|| format!("ptxas omitted {metric}"))?;
                if value != 0 {
                    return Err(format!("{symbol} has {value}{metric}"));
                }
            }
            let used = block
                .lines()
                .find(|line| line.contains("Used ") && line.contains(" registers"))
                .ok_or_else(|| "ptxas omitted used-resource line".to_string())?;
            let registers = metric_before(used, " registers")
                .ok_or_else(|| "ptxas omitted register count".to_string())?;
            let shared = metric_before(used, " bytes smem").unwrap_or(0);
            if registers > DIRECT_MAX_REGISTERS || shared as usize != DIRECT_STATIC_SHARED {
                return Err(format!(
                    "{symbol} resource stop: regs={registers}, shared={shared}"
                ));
            }
            let disassembly = std::process::Command::new("nvdisasm")
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
            if !entry.contains("FFMA") || !entry.contains("LDGSTS") {
                return Err(format!("{symbol} SASS omitted FFMA or LDGSTS"));
            }
            for forbidden in [" LDL", " STL", " ATOM", " RED", " REDUX"] {
                if entry.contains(forbidden) {
                    return Err(format!("{symbol} SASS contains {forbidden}"));
                }
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(&ptx_path);
        let _ = std::fs::remove_file(&cubin_path);
        result
    }

    fn new_runtime() -> Result<Runtime, String> {
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
        let direct_source = direct_source::compose_source();
        let direct_source_sha = format!("{:x}", Sha256::digest(direct_source.as_bytes()));
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            direct_source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                options: vec!["--fmad=true".into(), "-DNDEBUG".into()],
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("compile direct TN CopyPlan: {error:?}"))?;
        validate_ptxas_and_sass(DIRECT, &ptx.to_src())?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx.to_src()))
            .map_err(|error| format!("load direct TN CopyPlan module: {error:?}"))?;
        let direct = module
            .load_function(DIRECT)
            .map_err(|error| format!("load {DIRECT}: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            ctx,
            _direct_module: module,
            direct,
            direct_source_sha,
        })
    }

    fn fixed_function(
        runtime: &Runtime,
        candidate: CandidateFixed,
    ) -> Result<&CudaFunction, String> {
        let CandidateFixed::Production = candidate;
        runtime
            .ctx
            .kernels
            .fixed_sm89_f32_n64_copyplan
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "CopyPlan unavailable: {:?}",
                    runtime.ctx.kernels.fixed_sm89_f32_n64_copyplan_rejection
                )
            })
    }

    fn launch_transpose(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let output = fixture.transposed.ptr();
        let input = fixture.x.ptr();
        let rows = fixture.cell.batch as i32;
        let columns = fixture.cell.k_out as i32;
        let mut builder = runtime
            .ctx
            .stream
            .launch_builder(&runtime.ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1);
        builder.arg(&output);
        builder.arg(&input);
        builder.arg(&rows);
        builder.arg(&columns);
        unsafe { builder.launch(config(fixture.cell.transpose_grid(), (32, 16, 1), 0)) }
            .map(|_| ())
            .map_err(|error| format!("transpose X: {error:?}"))
    }

    fn launch_retained_partials(
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
        for chunk in fixture.cell.chunks()? {
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

    fn launch_direct_partials(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        if fixture.cell.batch == 0 {
            return Err("zero reduction has no direct partial launch".into());
        }
        if fixture.cell.k_out & 3 != 0 || fixture.cell.n_out & 3 != 0 {
            return Err("direct TN candidate requires K_out/N strides divisible by four".into());
        }
        for (pointer, label) in [
            (fixture.x.ptr(), "X"),
            (fixture.dy.ptr(), "dY"),
            (fixture.candidate_partial.ptr(), "partial"),
        ] {
            if pointer & 15 != 0 {
                return Err(format!("direct TN {label} is not 16B aligned"));
            }
        }
        let partial = fixture.candidate_partial.ptr();
        let x = fixture.x.ptr();
        let dy = fixture.dy.ptr();
        let m = fixture.cell.batch as i32;
        let k = fixture.cell.k_out as i32;
        let n = fixture.cell.n_out as i32;
        let m_chunk = M_CHUNK as i32;
        let mut builder = runtime.ctx.stream.launch_builder(&runtime.direct);
        builder.arg(&partial);
        builder.arg(&x);
        builder.arg(&dy);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&m_chunk);
        unsafe { builder.launch(config(fixture.cell.direct_grid(), (128, 1, 1), 0)) }
            .map(|_| ())
            .map_err(|error| format!("launch raw {DIRECT}: {error:?}"))
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
        let m_chunk = M_CHUNK as i32;
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
        unsafe { builder.launch(config(fixture.cell.partial_grid(), (256, 1, 1), 0)) }
            .map(|_| ())
            .map_err(|error| format!("launch raw {PARTIAL}: {error:?}"))
    }

    fn launch_reducer(runtime: &Runtime, fixture: &Fixture, partial: u64) -> Result<(), String> {
        let output = fixture.output.ptr();
        let k = fixture.cell.k_out as i32;
        let n = fixture.cell.n_out as i32;
        let chunks = CHUNKS as i32;
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
                Arm::Candidate | Arm::Reference | Arm::Retained => launch_zero(runtime, fixture),
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
                Arm::Fast => Err("Fast comparator does not admit K=0".into()),
            };
        }
        match arm {
            Arm::Candidate => {
                launch_direct_partials(runtime, fixture)?;
                launch_reducer(runtime, fixture, fixture.candidate_partial.ptr())
            }
            Arm::Reference => {
                launch_reference_partials(runtime, fixture)?;
                launch_reducer(runtime, fixture, fixture.reference_partial.ptr())
            }
            Arm::Retained => {
                launch_retained_partials(runtime, fixture, candidate)?;
                launch_reducer(runtime, fixture, fixture.candidate_partial.ptr())
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
        _candidate: CandidateFixed,
        corpus: &str,
    ) -> Result<(), RawPartialError> {
        let harness = RawPartialError::Harness;
        fixture.reset(&runtime.ctx).map_err(&harness)?;
        launch_reference_partials(runtime, fixture).map_err(&harness)?;
        launch_direct_partials(runtime, fixture).map_err(&harness)?;
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
            .chunks()
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
                    DIRECT,
                    chunk.begin,
                    chunk.begin + chunk.len,
                    actual[word],
                    reference[word]
                )));
            }
            println!(
                "{}",
                json!({
                    "schema":"MambaBiF32TnDirectRawPartialV1",
                    "corpus":corpus,
                    "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                    "candidate":DIRECT,
                    "fc":index,
                    "m_begin":chunk.begin,
                    "m_end":chunk.begin+chunk.len,
                    "words":stride,
                    "oracle":PARTIAL,
                    "exact_bits":true
                })
            );
        }
        fixture.validate_inputs(&runtime.ctx).map_err(harness)
    }

    fn select_candidate(runtime: &Runtime) -> Result<CandidateFixed, String> {
        let mut finite = Fixture::new(&runtime.ctx, PRISM, 1.0, false)?;
        compare_raw_partials(
            runtime,
            &mut finite,
            CandidateFixed::Production,
            "full_mantissa",
        )
        .map_err(|error| error.to_string())?;
        let mut exceptional = Fixture::new(&runtime.ctx, PRISM, 1.0, true)?;
        compare_raw_partials(
            runtime,
            &mut exceptional,
            CandidateFixed::Production,
            "exceptional_payload",
        )
        .map_err(|error| error.to_string())?;
        Ok(CandidateFixed::Production)
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

    #[allow(dead_code)]
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
            fixture.cell.partial_grid(),
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
            || unsafe { graph_arg::<i32>(partial, 6) }? != M_CHUNK as i32
        {
            return Err("AUTO TnSplitM partial ABI/arguments changed".into());
        }
        if unsafe { graph_arg::<u64>(reducer, 0) }? != fixture.output.ptr()
            || unsafe { graph_arg::<u64>(reducer, 1) }? != partial_ptr
            || unsafe { graph_arg::<f32>(reducer, 2) }?.to_bits() != 1.0_f32.to_bits()
            || unsafe { graph_arg::<i32>(reducer, 3) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(reducer, 4) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(reducer, 5) }? != CHUNKS as i32
        {
            return Err("AUTO TnSplitM reducer ABI/arguments changed".into());
        }
        unsafe { require_chain(graph, &[partial.node, reducer.node], "AUTO") }?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanAutoPlanV2",
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "plan":{"kind":"TnSplitM","m_chunk":M_CHUNK,"chunks":CHUNKS},
                "nodes":[PARTIAL,REDUCER],
                "edge":[PARTIAL,REDUCER],
                "partial_layout":[CHUNKS,fixture.cell.k_out,fixture.cell.n_out]
            })
        );
        Ok(())
    }

    unsafe fn require_retained_graph(
        graph: &CudaGraph,
        fixture: &Fixture,
        candidate: CandidateFixed,
    ) -> Result<(), String> {
        let nodes = unsafe { observed_kernels(graph) }?;
        if fixture.cell.batch == 0 {
            if nodes.len() != 1 {
                return Err(format!("K0 retained has {} graph nodes", nodes.len()));
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
                return Err("K0 retained ABI/arguments changed".into());
            }
            unsafe { require_chain(graph, &[zero.node], "K0 retained") }?;
            return Ok(());
        }
        if nodes.len() != CHUNKS + 2 {
            return Err(format!(
                "retained has {} nodes, expected {}",
                nodes.len(),
                CHUNKS + 2
            ));
        }
        let transpose = nodes
            .iter()
            .find(|node| node.symbol == TRANSPOSE)
            .ok_or_else(|| format!("retained omitted {TRANSPOSE}"))?;
        require_launch(
            transpose,
            TRANSPOSE,
            fixture.cell.transpose_grid(),
            (32, 16, 1),
            0,
        )?;
        if unsafe { graph_arg::<u64>(transpose, 0) }? != fixture.transposed.ptr()
            || unsafe { graph_arg::<u64>(transpose, 1) }? != fixture.x.ptr()
            || unsafe { graph_arg::<i32>(transpose, 2) }? != fixture.cell.batch as i32
            || unsafe { graph_arg::<i32>(transpose, 3) }? != fixture.cell.k_out as i32
        {
            return Err("retained transpose ABI/arguments changed".into());
        }
        let fixed_nodes = nodes
            .iter()
            .filter(|node| node.symbol == candidate.symbol())
            .collect::<Vec<_>>();
        if fixed_nodes.len() != CHUNKS {
            return Err(format!(
                "retained has {} {} nodes",
                fixed_nodes.len(),
                candidate.symbol()
            ));
        }
        let mut ordered_fixed: Vec<Option<&ObservedKernel>> = vec![None; CHUNKS];
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
                .chunks()?
                .into_iter()
                .position(|chunk| {
                    c == fixture.candidate_partial.ptr()
                        + (chunk.partial_offset * size_of::<f32>()) as u64
                })
                .ok_or_else(|| "retained Fixed output is outside raw partial tensor".to_string())?;
            if ordered_fixed[index].replace(node).is_some() {
                return Err(format!("duplicate retained Fixed chunk {index}"));
            }
            let chunk = fixture.cell.chunks()?[index];
            let expected_a = fixture.transposed.ptr() + (chunk.begin * size_of::<f32>()) as u64;
            let expected_b =
                fixture.dy.ptr() + (chunk.begin * fixture.cell.n_out * size_of::<f32>()) as u64;
            if unsafe { graph_arg::<u64>(node, 1) }? != expected_a
                || unsafe { graph_arg::<u64>(node, 2) }? != expected_b
                || unsafe { graph_arg::<u64>(node, 3) }? != 0
                || unsafe { graph_arg::<FixedParams>(node, 4) }? != fixture.cell.fixed_params(chunk)
            {
                return Err(format!(
                    "retained Fixed chunk {index} ABI/arguments changed"
                ));
            }
        }
        let reducer = nodes
            .iter()
            .find(|node| node.symbol == REDUCER)
            .ok_or_else(|| format!("retained omitted {REDUCER}"))?;
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
            || unsafe { graph_arg::<i32>(reducer, 5) }? != CHUNKS as i32
        {
            return Err("retained reducer ABI/arguments changed".into());
        }
        let ordered_fixed = ordered_fixed
            .into_iter()
            .enumerate()
            .map(|(index, node)| node.ok_or_else(|| format!("missing retained chunk{index}")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut chain = vec![transpose.node];
        chain.extend(ordered_fixed.iter().map(|node| node.node));
        chain.push(reducer.node);
        unsafe { require_chain(graph, &chain, "retained") }?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnCopyPlanGraphV2",
                "arm":"Retained",
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "nodes":[TRANSPOSE,candidate.symbol(),candidate.symbol(),candidate.symbol(),candidate.symbol(),REDUCER],
                "order":["transpose","chunk0","chunk1","chunk2","chunk3","reduce"]
            })
        );
        Ok(())
    }

    unsafe fn require_candidate_graph(
        graph: &CudaGraph,
        fixture: &Fixture,
        candidate: CandidateFixed,
    ) -> Result<(), String> {
        if fixture.cell.batch == 0 {
            return unsafe { require_retained_graph(graph, fixture, candidate) };
        }
        let nodes = unsafe { observed_kernels(graph) }?;
        if nodes.len() != 2 {
            return Err(format!(
                "direct candidate has {} nodes, expected two",
                nodes.len()
            ));
        }
        let partial = nodes
            .iter()
            .find(|node| node.symbol == DIRECT)
            .ok_or_else(|| format!("direct candidate omitted {DIRECT}"))?;
        let reducer = nodes
            .iter()
            .find(|node| node.symbol == REDUCER)
            .ok_or_else(|| format!("direct candidate omitted {REDUCER}"))?;
        require_launch(partial, DIRECT, fixture.cell.direct_grid(), (128, 1, 1), 0)?;
        require_launch(
            reducer,
            REDUCER,
            fixture.cell.reducer_grid(),
            (256, 1, 1),
            0,
        )?;
        if unsafe { graph_arg::<u64>(partial, 0) }? != fixture.candidate_partial.ptr()
            || unsafe { graph_arg::<u64>(partial, 1) }? != fixture.x.ptr()
            || unsafe { graph_arg::<u64>(partial, 2) }? != fixture.dy.ptr()
            || unsafe { graph_arg::<i32>(partial, 3) }? != fixture.cell.batch as i32
            || unsafe { graph_arg::<i32>(partial, 4) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(partial, 5) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(partial, 6) }? != M_CHUNK as i32
        {
            return Err("direct partial ABI/arguments changed".into());
        }
        if unsafe { graph_arg::<u64>(reducer, 0) }? != fixture.output.ptr()
            || unsafe { graph_arg::<u64>(reducer, 1) }? != fixture.candidate_partial.ptr()
            || unsafe { graph_arg::<f32>(reducer, 2) }?.to_bits() != fixture.alpha.to_bits()
            || unsafe { graph_arg::<i32>(reducer, 3) }? != fixture.cell.k_out as i32
            || unsafe { graph_arg::<i32>(reducer, 4) }? != fixture.cell.n_out as i32
            || unsafe { graph_arg::<i32>(reducer, 5) }? != CHUNKS as i32
        {
            return Err("direct reducer ABI/arguments changed".into());
        }
        unsafe { require_chain(graph, &[partial.node, reducer.node], "direct candidate") }?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnDirectGraphV1",
                "arm":"Candidate",
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "nodes":[DIRECT,REDUCER],
                "edge":[DIRECT,REDUCER],
                "partial_layout":[CHUNKS,fixture.cell.k_out,fixture.cell.n_out]
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
            fixture.cell.partial_grid(),
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
            Arm::Candidate | Arm::Retained if fixture.cell.batch > 0 => {
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
                Arm::Retained if fixture.cell.batch > 0 => {
                    require_retained_graph(&oracle_graph, fixture, candidate)?
                }
                Arm::Reference => require_reference_graph(&oracle_graph, fixture)?,
                Arm::Retained | Arm::Auto => {
                    // The public dW API fixes alpha=beta=1.0.  Checking only
                    // the symbol here previously let a non-unit manual K0
                    // launch masquerade as the retained oracle and fail only on
                    // signed zero.  Validate the complete packed ABI too.
                    require_retained_graph(&oracle_graph, fixture, candidate)?;
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
                "schema":"MambaBiF32TnDirectBitsV1",
                "case":label,
                "shape":[fixture.cell.batch,fixture.cell.k_out,fixture.cell.n_out],
                "alpha_bits":fixture.alpha.to_bits(),
                "candidate":DIRECT,
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

    fn direct_resources(function: &CudaFunction, symbol: &str) -> Result<(), String> {
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
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .map_err(|error| format!("{symbol} occupancy: {error:?}"))?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnDirectPrismResourceV1",
                "symbol":symbol,
                "threads":128,
                "registers":registers,
                "register_cap":DIRECT_MAX_REGISTERS,
                "local_bytes":local,
                "static_shared_bytes":shared,
                "static_shared_expected":DIRECT_STATIC_SHARED,
                "occupancy":occupancy,
                "occupancy_min":DIRECT_MIN_OCCUPANCY
            })
        );
        if registers > DIRECT_MAX_REGISTERS as i32
            || local != 0
            || shared as usize != DIRECT_STATIC_SHARED
            || occupancy < DIRECT_MIN_OCCUPANCY
        {
            return Err(format!(
                "{symbol} hard resource stop: regs={registers}, local={local}, shared={shared}, occupancy={occupancy}"
            ));
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

    fn screen_comparator(
        quiet: &common::gpu_quiet::QuietGpu,
        runtime: &Runtime,
        fixture: &mut Fixture,
        comparator: Arm,
        windows: usize,
        candidate: CandidateFixed,
        candidate_graph: &CudaGraph,
        exact_repeated: &[u32],
    ) -> Result<bool, String> {
        if !matches!(comparator, Arm::Retained | Arm::Fast) || !matches!(windows, 3 | 7) {
            return Err("screen requires retained/Fast and once3/once7".into());
        }
        let comparator_graph = capture(runtime, fixture, comparator, candidate)?;
        unsafe {
            match comparator {
                Arm::Retained => require_retained_graph(&comparator_graph, fixture, candidate)?,
                Arm::Fast => require_fast_graph(&comparator_graph)?,
                _ => unreachable!(),
            }
        }
        let comparator_repeated = if comparator == Arm::Fast {
            fast_goldens(runtime, fixture, candidate, &comparator_graph)?.1
        } else {
            exact_repeated.to_vec()
        };
        quiet.require_cohort("f32-tn-direct-d768-out-bk16/timed")?;
        let mut strata = Vec::new();
        for graph_path in [false, true] {
            for candidate_endpoints in [true, false] {
                let arms = if candidate_endpoints {
                    [Arm::Candidate, comparator, comparator, Arm::Candidate]
                } else {
                    [comparator, Arm::Candidate, Arm::Candidate, comparator]
                };
                let mut raw = Vec::new();
                for bracket in 0..windows + 2 {
                    let mut observations = [0.0; 4];
                    for (index, arm) in arms.into_iter().enumerate() {
                        let (graph, golden) = if arm == Arm::Candidate {
                            (candidate_graph, exact_repeated)
                        } else {
                            (&comparator_graph, comparator_repeated.as_slice())
                        };
                        observations[index] =
                            measure(runtime, fixture, arm, candidate, graph, graph_path, golden)?;
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
                        "schema":"MambaBiF32TnDirectPrismBk16ScreenV1",
                        "shape":[PRISM.batch,PRISM.k_out,PRISM.n_out],
                        "comparator":format!("{comparator:?}"),
                        "path":if graph_path {"graph"} else {"eager"},
                        "order":if candidate_endpoints {"ABBA"} else {"BAAB"},
                        "observation_arms":arms.map(|arm|format!("{arm:?}")),
                        "windows":windows,
                        "warmup_windows":2,
                        "logical_gemms_per_observation":OPS,
                        "candidate_timed_nodes":[DIRECT,REDUCER],
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
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnDirectPrismBk16DecisionV1",
                "shape":[PRISM.batch,PRISM.k_out,PRISM.n_out],
                "comparator":format!("{comparator:?}"),
                "windows":windows,
                "strata":strata,
                "strict_win":strict_win,
                "promotion":false
            })
        );
        Ok(strict_win)
    }

    #[test]
    #[ignore = "Ada CUDA13.2 discovery: exact F32 TN d768-out direct BK16"]
    fn ada_f32_tn_direct_d768_out_bk16_discovery() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("performance requires release".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref() == Some("0") {
            return Err("cuBLAS Fast disabled".into());
        }
        let quiet = common::gpu_quiet::QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("f32-tn-direct-d768-out-bk16/pre")?;
        let runtime = new_runtime()?;
        validate_fast_handle(&runtime.ctx)?;

        let mut auto_proof = Fixture::new(&runtime.ctx, PRISM, 1.0, false)?;
        let auto_graph = capture(
            &runtime,
            &mut auto_proof,
            Arm::Auto,
            CandidateFixed::Production,
        )?;
        unsafe { require_auto_splitm_graph(&auto_graph, &auto_proof) }?;

        let candidate = select_candidate(&runtime)?;
        direct_resources(&runtime.direct, DIRECT)?;
        resources(fixed_function(&runtime, candidate)?, FIXED, 128, 32_768)?;
        resources(
            &runtime.ctx.kernels.gemm_bi_tn_splitm_partial_aligned,
            PARTIAL,
            256,
            33_792,
        )?;
        resources(&runtime.ctx.kernels.gemm_bi_splitm_reduce, REDUCER, 256, 0)?;
        resources(&runtime.ctx.kernels.gemm_bi_tn_zero_reduction, ZERO, 256, 0)?;
        println!(
            "{}",
            json!({
                "schema":"MambaBiF32TnDirectPrismBk16ArtifactsV1",
                "artifacts":format!("{:?}",runtime.ctx.kernels.artifact_set_identity()),
                "direct_source_sha":runtime.direct_source_sha,
                "candidate_nodes":[DIRECT,REDUCER],
                "retained_nodes":[TRANSPOSE,FIXED,FIXED,FIXED,FIXED,REDUCER],
                "actual_auto_nodes":[PARTIAL,REDUCER],
                "candidate_grid":[288,1,4],
                "retained_fixed_grid":[288,1,1],
                "chunks":[[0,512],[512,1024],[1024,1536],[1536,2048]],
                "fast_compute":"CUBLAS_COMPUTE_32F_FAST_TF32",
                "algorithm":"CUBLAS_GEMM_DEFAULT",
                "promotion":false
            })
        );

        let mut target = Fixture::new(&runtime.ctx, PRISM, 1.0, false)?;
        let (_, exact_repeated) =
            check_exact_case(&runtime, &mut target, candidate, Arm::Auto, "target")?;

        let tail_cell = Cell {
            batch: 2047,
            k_out: 68,
            n_out: 132,
        };
        let mut tail = Fixture::new(&runtime.ctx, tail_cell, 1.0, true)?;
        compare_raw_partials(&runtime, &mut tail, candidate, "exceptional_tail_chunk")
            .map_err(|error| error.to_string())?;
        check_exact_case(&runtime, &mut tail, candidate, Arm::Reference, "tail_chunk")?;

        let mut exceptional = Fixture::new(
            &runtime.ctx,
            Cell {
                batch: 2048,
                k_out: 68,
                n_out: 132,
            },
            1.0,
            true,
        )?;
        compare_raw_partials(&runtime, &mut exceptional, candidate, "exceptional_payload")
            .map_err(|error| error.to_string())?;
        check_exact_case(
            &runtime,
            &mut exceptional,
            candidate,
            Arm::Reference,
            "exceptional_payload",
        )?;

        let mut nonunit = Fixture::new(&runtime.ctx, tail_cell, -0.75, false)?;
        check_exact_case(
            &runtime,
            &mut nonunit,
            candidate,
            Arm::Reference,
            "nonunit_alpha",
        )?;

        let mut zero = Fixture::new(
            &runtime.ctx,
            Cell {
                batch: 0,
                k_out: 68,
                n_out: 132,
            },
            1.0,
            true,
        )?;
        check_exact_case(&runtime, &mut zero, candidate, Arm::Auto, "k0")?;

        let candidate_graph = capture(&runtime, &mut target, Arm::Candidate, candidate)?;
        unsafe { require_candidate_graph(&candidate_graph, &target, candidate) }?;
        let retained_once3_win = screen_comparator(
            &quiet,
            &runtime,
            &mut target,
            Arm::Retained,
            3,
            candidate,
            &candidate_graph,
            &exact_repeated,
        )?;
        if !retained_once3_win {
            quiet.verify_post_cohort("f32-tn-direct-d768-out-bk16/once3-stop")?;
            return Ok(());
        }
        let retained_once7_win = screen_comparator(
            &quiet,
            &runtime,
            &mut target,
            Arm::Retained,
            7,
            candidate,
            &candidate_graph,
            &exact_repeated,
        )?;
        if retained_once7_win {
            let fast_win = screen_comparator(
                &quiet,
                &runtime,
                &mut target,
                Arm::Fast,
                7,
                candidate,
                &candidate_graph,
                &exact_repeated,
            )?;
            println!(
                "{}",
                json!({
                    "schema":"MambaBiF32TnDirectPrismBk16FinalDecisionV1",
                    "shape":[PRISM.batch,PRISM.k_out,PRISM.n_out],
                    "candidate":DIRECT,
                    "retain_against_copyplan":true,
                    "fast_win":fast_win,
                    "decision":if fast_win {"retain_fast_winner"} else {"retain_best_not_fast"},
                    "promotion":false
                })
            );
        }
        quiet.verify_post_cohort("f32-tn-direct-d768-out-bk16/post")?;
        Ok(())
    }
}
