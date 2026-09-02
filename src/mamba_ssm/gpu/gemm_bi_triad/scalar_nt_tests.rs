use super::*;
use crate::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dx_raw;
use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
};
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba_ssm::gpu::kernel_identity::{ArtifactKind, ModuleKind};
use sha2::{Digest, Sha256};

const GUARD_WORDS: [u32; 4] = [0x7fc1_2345, 0xff81_3579, 0x8000_0000, 0x5a5a_a5a5];
const GUARD_FLOATS: usize = GUARD_WORDS.len();

#[derive(Clone, Copy)]
struct NtCase {
    label: &'static str,
    dims: (usize, usize, usize),
    misaligned_b: bool,
    exceptional: bool,
    rounding_sensitive: bool,
}

struct NtBuffers {
    a: GpuBuffer,
    b_storage: GpuBuffer,
    direct_output: GpuBuffer,
    live_output: GpuBuffer,
    a_host: Vec<f32>,
    a_storage_host: Vec<f32>,
    b_host: Vec<f32>,
    b_storage_host: Vec<f32>,
    b_offset: usize,
    output_offset: usize,
    live_elements: usize,
}

fn finite_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let signed = ((state >> 16) & 63) as i32 - 31;
            signed as f32 * (1.0 / 32.0)
        })
        .collect()
}

fn rounding_sensitive_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let magnitude = f32::from_bits(0x3f00_0000 | (state & 0x007f_ffff));
            if state & 0x8000_0000 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

fn guarded_storage(payload: &[f32], prefix: usize) -> Vec<f32> {
    let mut storage = (0..prefix + payload.len() + GUARD_FLOATS)
        .map(|index| f32::from_bits(GUARD_WORDS[index % GUARD_WORDS.len()]))
        .collect::<Vec<_>>();
    storage[prefix..prefix + payload.len()].copy_from_slice(payload);
    storage
}

fn guard_bits(start: usize, len: usize) -> Vec<u32> {
    (start..start + len)
        .map(|index| GUARD_WORDS[index % GUARD_WORDS.len()])
        .collect()
}

fn validate_immutable_storage(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} storage length changed: expected {}, got {}",
            expected.len(),
            actual.len()
        ));
    }
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
    {
        return Err(format!(
            "{label} storage changed at {index}: expected {:#010x}, got {:#010x}",
            expected.to_bits(),
            actual.to_bits()
        ));
    }
    Ok(())
}

fn validate_output_suffix(label: &str, storage: &[f32], active: usize) -> Result<(), String> {
    let suffix = storage
        .get(active..)
        .ok_or_else(|| format!("{label} active extent {active} exceeds {}", storage.len()))?;
    if suffix.len() != GUARD_FLOATS {
        return Err(format!(
            "{label} suffix length must be {GUARD_FLOATS}, got {}",
            suffix.len()
        ));
    }
    let expected = guard_bits(active, suffix.len());
    if let Some((index, (actual, expected))) = suffix
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual.to_bits() != **expected)
    {
        return Err(format!(
            "{label} suffix changed at {}: expected {:#010x}, got {:#010x}",
            active + index,
            expected,
            actual.to_bits()
        ));
    }
    Ok(())
}

fn case_inputs(case: NtCase) -> (Vec<f32>, Vec<f32>) {
    let (m, k_out, n) = case.dims;
    let values = if case.rounding_sensitive {
        rounding_sensitive_values
    } else {
        finite_values
    };
    let mut a = values(m * n, 0x1234_5678);
    let mut b = values(k_out * n, 0x9abc_def0);
    if case.exceptional {
        for row in 0..m {
            a[row * n + 17] = 1.0;
            a[row * n + 19] = 1.0;
        }
        b[17] = f32::from_bits(0x7fc0_55aa);
        b[n + 19] = f32::INFINITY;
        b[2 * n + 3] = -0.0;
        b[2 * n + 5] = f32::from_bits(1);
        b[3 * n + 7] = f32::MIN_POSITIVE;
        b[4 * n + 11] = f32::MAX;
    }
    (a, b)
}

impl NtBuffers {
    fn new(ctx: &GpuCtx, case: NtCase) -> Self {
        let (m, k_out, _) = case.dims;
        let (a_host, b_host) = case_inputs(case);
        let b_offset = if case.misaligned_b { 1 } else { 4 };
        let output_offset = 4;
        let live_elements = m * k_out;
        let a_storage_host = guarded_storage(&a_host, 0);
        let b_storage_host = guarded_storage(&b_host, b_offset);
        let output_storage = guarded_storage(&vec![0.0; live_elements], output_offset);
        let live_output_storage = guarded_storage(&vec![0.0; live_elements], 0);
        let a = GpuBuffer::from_cpu(&ctx.stream, &a_storage_host)
            .expect("allocate guarded scalar NT A");
        let b_storage = GpuBuffer::from_cpu(&ctx.stream, &b_storage_host)
            .expect("allocate guarded scalar NT B");
        let direct_output = GpuBuffer::from_cpu(&ctx.stream, &output_storage)
            .expect("allocate guarded direct scalar NT output");
        let live_output = GpuBuffer::from_cpu(&ctx.stream, &live_output_storage)
            .expect("allocate guarded live scalar NT output");
        ctx.stream
            .synchronize()
            .expect("finish scalar NT allocations");
        Self {
            a,
            b_storage,
            direct_output,
            live_output,
            a_host,
            a_storage_host,
            b_host,
            b_storage_host,
            b_offset,
            output_offset,
            live_elements,
        }
    }

    fn b_ptr(&self) -> CUptr {
        self.b_storage.inner_at(self.b_offset)
    }

    fn output_ptr(&self) -> CUptr {
        self.direct_output.inner_at(self.output_offset)
    }

    fn direct_bits(&self, ctx: &GpuCtx, elements: usize) -> Vec<u32> {
        let values = self
            .direct_output
            .to_cpu(&ctx.stream)
            .expect("download guarded direct scalar NT output");
        values[self.output_offset..self.output_offset + elements]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn live_bits(&self, ctx: &GpuCtx) -> Vec<u32> {
        let storage = self
            .live_output
            .to_cpu(&ctx.stream)
            .expect("download guarded live scalar NT output");
        validate_output_suffix("live scalar NT output", &storage, self.live_elements)
            .expect("validate live scalar NT output suffix");
        storage[..self.live_elements]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn assert_inputs_unchanged(&self, ctx: &GpuCtx, label: &str) {
        let a = self
            .a
            .to_cpu(&ctx.stream)
            .expect("download guarded scalar NT A");
        validate_immutable_storage(label, &a, &self.a_storage_host)
            .expect("validate scalar NT A snapshot");
        let b = self
            .b_storage
            .to_cpu(&ctx.stream)
            .expect("download guarded scalar NT B");
        validate_immutable_storage(label, &b, &self.b_storage_host)
            .expect("validate scalar NT B snapshot");
    }

    fn assert_red_zones(&self, ctx: &GpuCtx, elements: usize, label: &str) {
        let output = self
            .direct_output
            .to_cpu(&ctx.stream)
            .expect("download direct scalar NT red zones");
        let prefix = &output[..self.output_offset];
        let suffix = &output[self.output_offset + elements..];
        assert_eq!(
            prefix
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            guard_bits(0, prefix.len()),
            "{label} output prefix red zone"
        );
        assert_eq!(
            suffix
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            guard_bits(self.output_offset + elements, suffix.len()),
            "{label} output suffix red zone"
        );
        self.live_bits(ctx);
        self.assert_inputs_unchanged(ctx, label);
    }
}

fn oracle_bits(a: &[f32], b: &[f32], dims: (usize, usize, usize)) -> Vec<u32> {
    let (m, k_out, n) = dims;
    let mut output = Vec::with_capacity(m * k_out);
    for row in 0..m {
        for column in 0..k_out {
            let mut sum = 0.0_f32;
            for reduction in 0..n {
                sum = a[row * n + reduction].mul_add(b[column * n + reduction], sum);
            }
            output.push(sum.to_bits());
        }
    }
    output
}

fn splitk32_oracle_bits(a: &[f32], b: &[f32], dims: (usize, usize, usize)) -> Vec<u32> {
    let (m, k_out, n) = dims;
    assert!(n.is_multiple_of(32));
    let chunks = n / 32;
    let mut output = Vec::with_capacity(m * k_out);
    for row in 0..m {
        for column in 0..k_out {
            let mut reduced = 0.0_f32;
            for chunk in 0..chunks {
                let mut partial = 0.0_f32;
                for reduction in chunk * 32..(chunk + 1) * 32 {
                    partial = a[row * n + reduction].mul_add(b[column * n + reduction], partial);
                }
                if chunk == 0 {
                    reduced = partial;
                } else {
                    reduced += partial;
                }
            }
            output.push(reduced.to_bits());
        }
    }
    output
}

fn bit_digest(bits: &[u32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for word in bits {
        digest.update(word.to_le_bytes());
    }
    digest.finalize().into()
}

fn assert_oracle(actual: &[u32], expected: &[u32], case: NtCase) {
    let (_, k_out, _) = case.dims;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let column = index % k_out;
        if case.exceptional && column == 0 {
            assert!(
                f32::from_bits(actual).is_nan(),
                "{} output {index} must be NaN",
                case.label
            );
        } else if case.exceptional && column == 1 {
            assert_eq!(
                actual, expected,
                "{} output {index} infinity bits and sign",
                case.label
            );
        } else {
            assert_eq!(actual, expected, "{} output {index}", case.label);
        }
    }
}

unsafe fn launch_forced_big(ctx: &GpuCtx, buffers: &NtBuffers, dims: (usize, usize, usize)) {
    let (m, k_out, n) = dims;
    let output = buffers.output_ptr();
    let a = buffers.a.cached_ptr();
    let b = buffers.b_ptr();
    let alpha = 1.0_f32;
    let m = i32::try_from(m).expect("forced Big NT M");
    let k_out = i32::try_from(k_out).expect("forced Big NT K_out");
    let n = i32::try_from(n).expect("forced Big NT N");
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (
            u32::try_from(m).expect("positive M").div_ceil(128)
                * u32::try_from(k_out).expect("positive K_out").div_ceil(128),
            1,
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES,
    };
    let mut launch = ctx
        .stream
        .launch_builder(&ctx.kernels.triad_kernels().gemm_bi_nt);
    launch.arg(&output);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&alpha);
    launch.arg(&m);
    launch.arg(&n);
    launch.arg(&k_out);
    unsafe { launch.launch(config) }.expect("launch forced Big scalar NT");
}

fn launch_live(ctx: &GpuCtx, buffers: &mut NtBuffers, dims: (usize, usize, usize)) {
    let b = buffers.b_ptr();
    gpu_gemm_bi_backward_dx_raw(
        ctx,
        &mut buffers.live_output,
        &buffers.a,
        b,
        dims.0,
        dims.1,
        dims.2,
    )
    .expect("launch live scalar NT");
}

fn assert_repeated_outputs(
    ctx: &GpuCtx,
    case: NtCase,
    buffers: &mut NtBuffers,
    live_oracle: fn(&[f32], &[f32], (usize, usize, usize)) -> Vec<u32>,
) {
    let elements = case.dims.0 * case.dims.1;
    let forced_expected = oracle_bits(&buffers.a_host, &buffers.b_host, case.dims);
    let live_expected = live_oracle(&buffers.a_host, &buffers.b_host, case.dims);
    unsafe { launch_forced_big(ctx, buffers, case.dims) };
    ctx.stream.synchronize().expect("finish forced Big NT");
    let forced = buffers.direct_bits(ctx, elements);
    assert_oracle(&forced, &forced_expected, case);
    let forced_digest = bit_digest(&forced);
    for iteration in 0..3 {
        unsafe { launch_forced_big(ctx, buffers, case.dims) };
        ctx.stream.synchronize().expect("repeat forced Big NT");
        let repeated = buffers.direct_bits(ctx, elements);
        assert_eq!(repeated, forced, "{} forced repeat {iteration}", case.label);
        assert_eq!(bit_digest(&repeated), forced_digest);
    }

    launch_live(ctx, buffers, case.dims);
    ctx.stream.synchronize().expect("finish live scalar NT");
    let live = buffers.live_bits(ctx);
    assert_oracle(&live, &live_expected, case);
    let live_digest = bit_digest(&live);
    for iteration in 0..3 {
        launch_live(ctx, buffers, case.dims);
        ctx.stream.synchronize().expect("repeat live scalar NT");
        let repeated = buffers.live_bits(ctx);
        assert_eq!(repeated, live, "{} live repeat {iteration}", case.label);
        assert_eq!(bit_digest(&repeated), live_digest);
    }

    let forced_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_forced_big(ctx, buffers, case.dims);
            Ok(())
        })
    }
    .expect("capture forced Big scalar NT");
    for iteration in 0..3 {
        forced_graph.launch().expect("replay forced Big scalar NT");
        ctx.stream
            .synchronize()
            .expect("finish forced graph replay");
        let replay = buffers.direct_bits(ctx, elements);
        assert_eq!(replay, forced, "{} forced graph {iteration}", case.label);
        assert_eq!(bit_digest(&replay), forced_digest);
    }

    let live_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_live(ctx, buffers, case.dims);
            Ok(())
        })
    }
    .expect("capture live scalar NT");
    for iteration in 0..3 {
        live_graph.launch().expect("replay live scalar NT");
        ctx.stream.synchronize().expect("finish live graph replay");
        let replay = buffers.live_bits(ctx);
        assert_eq!(replay, live, "{} live graph {iteration}", case.label);
        assert_eq!(bit_digest(&replay), live_digest);
    }
    buffers.assert_red_zones(ctx, elements, case.label);
}

fn exact_m2n16_environment(ctx: &GpuCtx) -> bool {
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
    ctx.compute_capability() == (12, 0)
        && ctx.kernels.multiprocessor_count() == 170
        && compiler.target.as_str() == "compute_120"
        && compiler.nvrtc_version == (13, 2)
        && compiler.nvrtc_library_known
        && compiler.nvrtc_library_domain != [0; 32]
        && compiler.invocation_digest != [0; 32]
        && artifact.module_kind == ModuleKind::TriadScalar
        && artifact.artifact_kind == compiler.output_kind
        && artifact.compile_key == compiler.invocation_digest
        && artifact.artifact_digest != [0; 32]
}

fn assert_thin_physical_evidence(ctx: &GpuCtx, case: NtCase) -> [u8; 32] {
    let request = PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nt,
        case.dims,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
    );
    let qualified = qualify_physical_launch(ctx, request)
        .unwrap_or_else(|error| panic!("{} physical qualification: {error}", case.label));
    let evidence = qualified.evidence();
    assert!(
        evidence.eager_graph_equal(),
        "{} eager/graph nodes",
        case.label
    );
    let exact_m2n16 = case.dims == (512, 16, 2_048) && exact_m2n16_environment(ctx);
    if exact_m2n16 {
        assert_eq!(evidence.launch_count(), 1, "{} launch count", case.label);
        let node = evidence.nodes()[0];
        assert_eq!(node.symbol, "gemm_bi_nt_m2n16_bk64_splitk32_v1");
        assert_eq!(node.module_kind, ModuleKind::TriadScalar);
        assert_eq!(node.logical_op, ResolvedGemmOp::Nt);
        assert_eq!(node.shape, case.dims);
        assert_eq!(node.strides, (2_048, 2_048, 16));
        assert_eq!(node.tile, Some((2, 16)));
        assert_eq!(node.launch.grid_dim, (256, 1, 1));
        assert_eq!(node.launch.block_dim, (64, 1, 1));
        assert_eq!(node.launch.shared_mem_bytes, 17_984);
        assert_ne!(node.launch.arguments_digest, [0; 32]);
    } else {
        assert_eq!(evidence.launch_count(), 3, "{} launch count", case.label);
        assert_eq!(
            evidence
                .nodes()
                .iter()
                .map(|node| node.symbol)
                .collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_2d",
                "gemm_bi_nn_splitk32_partial",
                "gemm_bi_splitk_reduce",
            ],
            "{} physical nodes",
            case.label
        );
        let partial = evidence.nodes()[1];
        let reducer = evidence.nodes()[2];
        assert_ne!(partial.symbol, reducer.symbol, "{} node symbol", case.label);
        assert_ne!(partial.tile, reducer.tile, "{} node tile", case.label);
        assert_ne!(
            partial.launch.arguments_digest, reducer.launch.arguments_digest,
            "{} partial/reducer argument identity",
            case.label
        );
    }
    let digest = evidence.launch_digest();
    assert_ne!(digest, [0; 32], "{} launch digest", case.label);
    qualified
        .validate_red_zones(ctx)
        .unwrap_or_else(|error| panic!("{} physical red zones: {error}", case.label));
    digest
}

fn assert_repeated_live_outputs(
    ctx: &GpuCtx,
    case: NtCase,
    buffers: &mut NtBuffers,
    expected: &[u32],
) {
    let elements = case.dims.0 * case.dims.1;
    launch_live(ctx, buffers, case.dims);
    ctx.stream.synchronize().expect("finish live scalar NT");
    let live = buffers.live_bits(ctx);
    assert_oracle(&live, expected, case);
    let live_digest = bit_digest(&live);
    for iteration in 0..3 {
        launch_live(ctx, buffers, case.dims);
        ctx.stream.synchronize().expect("repeat live scalar NT");
        let repeated = buffers.live_bits(ctx);
        assert_eq!(repeated, live, "{} live repeat {iteration}", case.label);
        assert_eq!(bit_digest(&repeated), live_digest);
    }

    let live_graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_live(ctx, buffers, case.dims);
            Ok(())
        })
    }
    .expect("capture live scalar NT");
    for iteration in 0..3 {
        live_graph.launch().expect("replay live scalar NT");
        ctx.stream.synchronize().expect("finish live graph replay");
        let replay = buffers.live_bits(ctx);
        assert_eq!(replay, live, "{} live graph {iteration}", case.label);
        assert_eq!(bit_digest(&replay), live_digest);
    }
    buffers.assert_red_zones(ctx, elements, case.label);
}

fn assert_nt_transpose_scratch(ctx: &GpuCtx, buffers: &NtBuffers, case: NtCase) {
    let (_, rows, columns) = case.dims;
    let scratch = ctx
        .stream
        .clone_dtoh(
            ctx.kernels
                .transpose_scratch_buf(&ctx.stream)
                .expect("resolve scalar NT transpose scratch"),
        )
        .expect("download scalar NT transpose scratch");
    assert_eq!(
        scratch.len(),
        SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS,
        "{} shared transpose scratch allocation capacity",
        case.label
    );
    assert!(scratch.len() >= rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            assert_eq!(
                scratch[column * rows + row].to_bits(),
                buffers.b_host[row * columns + column].to_bits(),
                "{} transpose scratch row={row} column={column}",
                case.label
            );
        }
    }
}

#[test]
fn scalar_nt_live_canaries_detect_input_and_output_corruption() {
    let input = guarded_storage(&[1.0, -2.0, 3.0], 2);
    assert!(validate_immutable_storage("input", &input, &input).is_ok());
    for index in [0, 2, input.len() - 1] {
        let mut corrupted = input.clone();
        corrupted[index] = f32::from_bits(corrupted[index].to_bits() ^ 1);
        assert!(validate_immutable_storage("input", &corrupted, &input).is_err());
    }

    let mut output = guarded_storage(&[0.0; 3], 0);
    output[..3].copy_from_slice(&[4.0, 5.0, 6.0]);
    assert!(validate_output_suffix("output", &output, 3).is_ok());
    output[3] = f32::from_bits(output[3].to_bits() ^ 1);
    assert!(validate_output_suffix("output", &output, 3).is_err());
}

#[test]
#[ignore = "requires an exclusive SM120/170-SM/NVRTC-13.2 CUDA GPU"]
fn sm120_scalar_nt_d768_out_production_route_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (12, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(ctx.kernels.multiprocessor_count(), 170);
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    assert_eq!(compiler.target.as_str(), "compute_120");
    assert_eq!(compiler.nvrtc_version, (13, 2));
    assert!(compiler.nvrtc_library_known);
    assert_ne!(compiler.nvrtc_library_domain, [0; 32]);
    let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
    assert_eq!(artifact.module_kind, ModuleKind::TriadScalar);
    assert_eq!(artifact.artifact_kind, ArtifactKind::Ptx);
    assert_eq!(artifact.compile_key, compiler.invocation_digest);
    assert_ne!(artifact.artifact_digest, [0; 32]);

    let case = NtCase {
        label: "d768-out-transpose-m64n64",
        dims: (2_048, 1_536, 768),
        misaligned_b: false,
        exceptional: false,
        rounding_sensitive: true,
    };
    let request = PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nt,
        case.dims,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
    );
    let qualified = qualify_physical_launch(&ctx, request)
        .expect("qualify production d768-out scalar NT route");
    let evidence = qualified.evidence();
    assert_eq!(evidence.route_identity().artifacts.triad_scalar, artifact);
    assert_eq!(evidence.route_identity().device.compute_capability, (12, 0));
    assert_eq!(evidence.route_identity().device.multiprocessor_count, 170);
    assert_eq!(evidence.launch_count(), 2);
    assert!(evidence.eager_graph_equal());
    assert_eq!(
        evidence
            .nodes()
            .iter()
            .map(|node| node.symbol)
            .collect::<Vec<_>>(),
        [
            "gemm_bi_transpose_f32_32x16_d768_v1",
            "gemm_bi_nn_m64n64_bk16_s2_v1",
        ]
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.module_kind == ModuleKind::TriadScalar)
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.logical_op == ResolvedGemmOp::Nt)
    );
    assert!(evidence.nodes().iter().all(|node| node.shape == case.dims));
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.strides == (768, 768, 1_536))
    );
    assert_eq!(evidence.nodes()[0].tile, Some((32, 32)));
    assert_eq!(evidence.nodes()[1].tile, Some((64, 64)));
    assert_eq!(evidence.nodes()[0].launch.grid_dim, (24, 48, 1));
    assert_eq!(evidence.nodes()[0].launch.block_dim, (32, 16, 1));
    assert_eq!(evidence.nodes()[0].launch.shared_mem_bytes, 0);
    assert_eq!(evidence.nodes()[1].launch.grid_dim, (768, 1, 1));
    assert_eq!(evidence.nodes()[1].launch.block_dim, (128, 1, 1));
    assert_eq!(
        evidence.nodes()[1].launch.shared_mem_bytes,
        crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
    );
    assert_ne!(evidence.nodes()[0].launch.arguments_digest, [0; 32]);
    assert_ne!(evidence.nodes()[1].launch.arguments_digest, [0; 32]);
    assert_ne!(
        evidence.nodes()[0].launch.arguments_digest,
        evidence.nodes()[1].launch.arguments_digest
    );
    qualified
        .validate_red_zones(&ctx)
        .expect("validate production qualification red zones");

    let mut buffers = NtBuffers::new(&ctx, case);
    let elements = case.dims.0 * case.dims.1;
    unsafe { launch_forced_big(&ctx, &buffers, case.dims) };
    ctx.stream
        .synchronize()
        .expect("finish d768-out generic scalar NT reference");
    let expected = buffers.direct_bits(&ctx, elements);

    launch_live(&ctx, &mut buffers, case.dims);
    ctx.stream
        .synchronize()
        .expect("finish d768-out production eager warmup");
    let eager = buffers.live_bits(&ctx);
    assert_eq!(
        eager, expected,
        "production eager differs from generic exact NT"
    );
    for iteration in 0..3 {
        launch_live(&ctx, &mut buffers, case.dims);
        ctx.stream
            .synchronize()
            .expect("finish d768-out production eager repeat");
        assert_eq!(
            buffers.live_bits(&ctx),
            eager,
            "production eager repeat {iteration}"
        );
    }

    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_live(&ctx, &mut buffers, case.dims);
            Ok(())
        })
    }
    .expect("capture d768-out production graph");
    for iteration in 0..3 {
        graph.launch().expect("replay d768-out production graph");
        ctx.stream
            .synchronize()
            .expect("finish d768-out production graph replay");
        assert_eq!(
            buffers.live_bits(&ctx),
            eager,
            "production graph repeat {iteration}"
        );
    }
    assert_nt_transpose_scratch(&ctx, &buffers, case);
    buffers.assert_red_zones(&ctx, elements, case.label);
}

fn assert_transpose_m64n64_production_matches_generic(ctx: &GpuCtx, case: NtCase) {
    let mut buffers = NtBuffers::new(ctx, case);
    let elements = case.dims.0 * case.dims.1;
    unsafe { launch_forced_big(ctx, &buffers, case.dims) };
    ctx.stream
        .synchronize()
        .expect("finish qualified transpose generic scalar NT reference");
    let expected = buffers.direct_bits(ctx, elements);

    launch_live(ctx, &mut buffers, case.dims);
    ctx.stream
        .synchronize()
        .expect("finish qualified transpose production eager warmup");
    let eager = buffers.live_bits(ctx);
    assert_eq!(eager, expected, "{} eager versus generic", case.label);
    buffers.assert_inputs_unchanged(ctx, case.label);
    for iteration in 0..3 {
        launch_live(ctx, &mut buffers, case.dims);
        ctx.stream
            .synchronize()
            .expect("finish qualified transpose production eager repeat");
        assert_eq!(
            buffers.live_bits(ctx),
            eager,
            "{} eager repeat {iteration}",
            case.label
        );
    }
    buffers.assert_inputs_unchanged(ctx, case.label);

    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            launch_live(ctx, &mut buffers, case.dims);
            Ok(())
        })
    }
    .expect("capture qualified transpose production graph");
    for iteration in 0..3 {
        graph
            .launch()
            .expect("replay qualified transpose production graph");
        ctx.stream
            .synchronize()
            .expect("finish qualified transpose production graph replay");
        assert_eq!(
            buffers.live_bits(ctx),
            eager,
            "{} graph repeat {iteration}",
            case.label
        );
    }
    buffers.assert_inputs_unchanged(ctx, case.label);
    assert_nt_transpose_scratch(ctx, &buffers, case);
    buffers.assert_red_zones(ctx, elements, case.label);
}

#[test]
#[ignore = "requires an exclusive SM120/170-SM/NVRTC-13.2 CUDA GPU"]
fn sm120_scalar_nt_large_deep_production_route_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (12, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(ctx.kernels.multiprocessor_count(), 170);
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    assert_eq!(compiler.target.as_str(), "compute_120");
    assert_eq!(compiler.nvrtc_version, (13, 2));
    assert!(compiler.nvrtc_library_known);
    assert_ne!(compiler.nvrtc_library_domain, [0; 32]);
    assert_ne!(compiler.invocation_digest, [0; 32]);
    let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
    assert_eq!(artifact.module_kind, ModuleKind::TriadScalar);
    assert_eq!(artifact.artifact_kind, compiler.output_kind);
    assert_eq!(artifact.compile_key, compiler.invocation_digest);
    assert_ne!(artifact.artifact_digest, [0; 32]);

    let case = NtCase {
        label: "large-deep-transpose-m64n64",
        dims: (4_096, 3_072, 1_536),
        misaligned_b: false,
        exceptional: false,
        rounding_sensitive: true,
    };
    let request = PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nt,
        case.dims,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
    );
    let qualified = qualify_physical_launch(&ctx, request)
        .expect("qualify production large-deep scalar NT route");
    let evidence = qualified.evidence();
    assert_eq!(evidence.route_identity().artifacts.triad_scalar, artifact);
    assert_eq!(evidence.route_identity().device.compute_capability, (12, 0));
    assert_eq!(evidence.route_identity().device.multiprocessor_count, 170);
    assert_eq!(evidence.launch_count(), 2);
    assert!(evidence.eager_graph_equal());
    assert_eq!(
        evidence
            .nodes()
            .iter()
            .map(|node| node.symbol)
            .collect::<Vec<_>>(),
        [
            "gemm_bi_transpose_f32_32x16_d768_v1",
            "gemm_bi_nn_m64n64_bk16_s2_v1",
        ]
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.module_kind == ModuleKind::TriadScalar)
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.logical_op == ResolvedGemmOp::Nt)
    );
    assert!(evidence.nodes().iter().all(|node| node.shape == case.dims));
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.strides == (1_536, 1_536, 3_072))
    );
    assert_eq!(evidence.nodes()[0].launch.grid_dim, (48, 96, 1));
    assert_eq!(evidence.nodes()[0].launch.block_dim, (32, 16, 1));
    assert_eq!(evidence.nodes()[0].launch.shared_mem_bytes, 0);
    assert_eq!(evidence.nodes()[1].launch.grid_dim, (3_072, 1, 1));
    assert_eq!(evidence.nodes()[1].launch.block_dim, (128, 1, 1));
    assert_eq!(
        evidence.nodes()[1].launch.shared_mem_bytes,
        SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
    );
    assert_ne!(evidence.nodes()[0].launch.arguments_digest, [0; 32]);
    assert_ne!(evidence.nodes()[1].launch.arguments_digest, [0; 32]);
    assert_ne!(
        evidence.nodes()[0].launch.arguments_digest,
        evidence.nodes()[1].launch.arguments_digest
    );
    qualified
        .validate_red_zones(&ctx)
        .expect("validate production large-deep qualification red zones");

    let kernels = ctx.kernels.triad_kernels();
    let transpose = &kernels.gemm_bi_transpose_f32_32x16_d768_v1;
    assert!(transpose.num_regs().expect("transpose registers") <= 28);
    assert_eq!(transpose.local_size_bytes().expect("transpose local"), 0);
    assert_eq!(
        transpose
            .shared_size_bytes()
            .expect("transpose static shared") as usize,
        SCALAR_NT_D768_TRANSPOSE_STATIC_SHARED_BYTES
    );
    assert!(
        transpose
            .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
            .expect("transpose occupancy")
            >= 2
    );
    let m64 = &kernels.gemm_bi_nn_m64n64_bk16_s2_v1;
    assert!(m64.num_regs().expect("M64 registers") <= 103);
    assert_eq!(m64.local_size_bytes().expect("M64 local"), 0);
    assert_eq!(m64.shared_size_bytes().expect("M64 static shared"), 0);
    assert!(
        m64.occupancy_max_active_blocks_per_multiprocessor(
            128,
            SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .expect("M64 occupancy")
            >= 4
    );

    assert_transpose_m64n64_production_matches_generic(&ctx, case);
    assert_transpose_m64n64_production_matches_generic(
        &ctx,
        NtCase {
            label: "large-deep-transpose-m64n64-exceptional",
            exceptional: true,
            rounding_sensitive: false,
            ..case
        },
    );
}

#[test]
#[ignore = "requires an exclusive SM120/170-SM/NVRTC-13.2 CUDA GPU"]
fn sm120_scalar_nt_prism_production_route_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (12, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(ctx.kernels.multiprocessor_count(), 170);
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    assert_eq!(compiler.target.as_str(), "compute_120");
    assert_eq!(compiler.nvrtc_version, (13, 2));
    assert!(compiler.nvrtc_library_known);
    assert_ne!(compiler.nvrtc_library_domain, [0; 32]);
    assert_ne!(compiler.invocation_digest, [0; 32]);
    let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
    assert_eq!(artifact.module_kind, ModuleKind::TriadScalar);
    assert_eq!(artifact.artifact_kind, compiler.output_kind);
    assert_eq!(artifact.compile_key, compiler.invocation_digest);
    assert_ne!(artifact.artifact_digest, [0; 32]);

    let case = NtCase {
        label: "prism-transpose-m64n64",
        dims: (4_621, 384, 1_928),
        misaligned_b: false,
        exceptional: false,
        rounding_sensitive: true,
    };
    assert_eq!(case.dims.1.checked_mul(case.dims.2), Some(740_352));
    let request = PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nt,
        case.dims,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
    );
    let qualified =
        qualify_physical_launch(&ctx, request).expect("qualify production prism scalar NT route");
    let evidence = qualified.evidence();
    assert_eq!(evidence.route_identity().artifacts.triad_scalar, artifact);
    assert_eq!(evidence.route_identity().device.compute_capability, (12, 0));
    assert_eq!(evidence.route_identity().device.multiprocessor_count, 170);
    assert_eq!(evidence.launch_count(), 2);
    assert!(evidence.eager_graph_equal());
    assert_eq!(
        evidence
            .nodes()
            .iter()
            .map(|node| node.symbol)
            .collect::<Vec<_>>(),
        [
            "gemm_bi_transpose_f32_32x16_d768_v1",
            "gemm_bi_nn_prism_m64n64_bk16_s2_v1",
        ]
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.module_kind == ModuleKind::TriadScalar)
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.logical_op == ResolvedGemmOp::Nt)
    );
    assert!(evidence.nodes().iter().all(|node| node.shape == case.dims));
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.strides == (1_928, 1_928, 384))
    );
    assert_eq!(evidence.nodes()[0].tile, Some((32, 32)));
    assert_eq!(evidence.nodes()[1].tile, Some((64, 64)));
    assert_eq!(evidence.nodes()[0].launch.grid_dim, (61, 12, 1));
    assert_eq!(evidence.nodes()[0].launch.block_dim, (32, 16, 1));
    assert_eq!(evidence.nodes()[0].launch.shared_mem_bytes, 0);
    assert_eq!(evidence.nodes()[1].launch.grid_dim, (438, 1, 1));
    assert_eq!(evidence.nodes()[1].launch.block_dim, (128, 1, 1));
    assert_eq!(
        evidence.nodes()[1].launch.shared_mem_bytes,
        SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
    );
    assert_ne!(evidence.nodes()[0].launch.arguments_digest, [0; 32]);
    assert_ne!(evidence.nodes()[1].launch.arguments_digest, [0; 32]);
    assert_ne!(
        evidence.nodes()[0].launch.arguments_digest,
        evidence.nodes()[1].launch.arguments_digest
    );
    let kernels = ctx.kernels.triad_kernels();
    let transpose = &kernels.gemm_bi_transpose_f32_32x16_d768_v1;
    assert!(transpose.num_regs().expect("transpose registers") <= 28);
    assert_eq!(transpose.local_size_bytes().expect("transpose local"), 0);
    assert_eq!(
        transpose
            .shared_size_bytes()
            .expect("transpose static shared") as usize,
        SCALAR_NT_D768_TRANSPOSE_STATIC_SHARED_BYTES
    );
    assert!(
        transpose
            .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
            .expect("transpose occupancy")
            >= 2
    );
    let m64 = &kernels.gemm_bi_nn_prism_m64n64_bk16_s2_v1;
    assert!(m64.num_regs().expect("M64 registers") <= 103);
    assert_eq!(m64.local_size_bytes().expect("M64 local"), 0);
    assert_eq!(m64.shared_size_bytes().expect("M64 static shared"), 0);
    assert!(
        m64.occupancy_max_active_blocks_per_multiprocessor(
            128,
            SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .expect("M64 occupancy")
            >= 4
    );

    assert_transpose_m64n64_production_matches_generic(&ctx, case);
    assert_transpose_m64n64_production_matches_generic(
        &ctx,
        NtCase {
            label: "prism-transpose-m64n64-exceptional",
            exceptional: true,
            rounding_sensitive: false,
            ..case
        },
    );
}

#[test]
#[ignore = "requires an exclusive SM120/170-SM/NVRTC-13.2 CUDA GPU"]
fn sm120_scalar_nt_d128_out_production_route_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (12, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    assert_eq!(ctx.kernels.multiprocessor_count(), 170);
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    assert_eq!(compiler.target.as_str(), "compute_120");
    assert_eq!(compiler.nvrtc_version, (13, 2));
    assert!(compiler.nvrtc_library_known);
    assert_ne!(compiler.nvrtc_library_domain, [0; 32]);
    assert_ne!(compiler.invocation_digest, [0; 32]);
    let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
    assert_eq!(artifact.module_kind, ModuleKind::TriadScalar);
    assert_eq!(artifact.artifact_kind, compiler.output_kind);
    assert_eq!(artifact.compile_key, compiler.invocation_digest);
    assert_ne!(artifact.artifact_digest, [0; 32]);

    let case = NtCase {
        label: "d128-out-transpose-m64n64",
        dims: (1_024, 256, 128),
        misaligned_b: false,
        exceptional: false,
        rounding_sensitive: true,
    };
    let request = PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nt,
        case.dims,
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
    );
    let qualified = qualify_physical_launch(&ctx, request)
        .expect("qualify production d128-out scalar NT route");
    let evidence = qualified.evidence();
    assert_eq!(evidence.route_identity().artifacts.triad_scalar, artifact);
    assert_eq!(evidence.launch_count(), 2);
    assert!(evidence.eager_graph_equal());
    assert_eq!(
        evidence
            .nodes()
            .iter()
            .map(|node| node.symbol)
            .collect::<Vec<_>>(),
        [
            "gemm_bi_transpose_f32_32x16_d768_v1",
            "gemm_bi_nn_m64n64_bk16_s2_v1",
        ]
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.module_kind == ModuleKind::TriadScalar)
    );
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.logical_op == ResolvedGemmOp::Nt)
    );
    assert!(evidence.nodes().iter().all(|node| node.shape == case.dims));
    assert!(
        evidence
            .nodes()
            .iter()
            .all(|node| node.strides == (128, 128, 256))
    );
    assert_eq!(evidence.nodes()[0].launch.grid_dim, (4, 8, 1));
    assert_eq!(evidence.nodes()[0].launch.block_dim, (32, 16, 1));
    assert_eq!(evidence.nodes()[1].launch.grid_dim, (64, 1, 1));
    assert_eq!(evidence.nodes()[1].launch.block_dim, (128, 1, 1));
    assert_eq!(
        evidence.nodes()[1].launch.shared_mem_bytes,
        SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
    );
    assert_ne!(evidence.nodes()[0].launch.arguments_digest, [0; 32]);
    assert_ne!(evidence.nodes()[1].launch.arguments_digest, [0; 32]);
    assert_ne!(
        evidence.nodes()[0].launch.arguments_digest,
        evidence.nodes()[1].launch.arguments_digest
    );
    qualified
        .validate_red_zones(&ctx)
        .expect("validate production d128-out qualification red zones");

    let kernels = ctx.kernels.triad_kernels();
    let transpose = &kernels.gemm_bi_transpose_f32_32x16_d768_v1;
    assert!(transpose.num_regs().expect("transpose registers") <= 28);
    assert_eq!(transpose.local_size_bytes().expect("transpose local"), 0);
    assert_eq!(
        transpose
            .shared_size_bytes()
            .expect("transpose static shared") as usize,
        SCALAR_NT_D768_TRANSPOSE_STATIC_SHARED_BYTES
    );
    assert!(
        transpose
            .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
            .expect("transpose occupancy")
            >= 2
    );
    let m64 = &kernels.gemm_bi_nn_m64n64_bk16_s2_v1;
    assert!(m64.num_regs().expect("M64 registers") <= 103);
    assert_eq!(m64.local_size_bytes().expect("M64 local"), 0);
    assert_eq!(m64.shared_size_bytes().expect("M64 static shared"), 0);
    assert!(
        m64.occupancy_max_active_blocks_per_multiprocessor(
            128,
            SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .expect("M64 occupancy")
            >= 4
    );

    assert_transpose_m64n64_production_matches_generic(&ctx, case);
    assert_transpose_m64n64_production_matches_generic(
        &ctx,
        NtCase {
            label: "d128-out-transpose-m64n64-exceptional",
            exceptional: true,
            rounding_sensitive: false,
            ..case
        },
    );
}

#[test]
#[ignore = "requires an Ada GPU"]
fn ada_scalar_big_nt_runtime_matrix_is_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(device.compute_capability, (8, 9));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    for case in [
        NtCase {
            label: "aligned-live-big-vector",
            dims: (512, 513, 128),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "misaligned-forced-big-vector",
            dims: (129, 129, 128),
            misaligned_b: true,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "aligned-m127-k127-n129",
            dims: (127, 127, 129),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "aligned-m129-k129-n257",
            dims: (129, 129, 257),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "aligned-m257-k257-n129",
            dims: (257, 257, 129),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "misaligned-m127-k129-n257",
            dims: (127, 129, 257),
            misaligned_b: true,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "misaligned-m129-k257-n129",
            dims: (129, 257, 129),
            misaligned_b: true,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "misaligned-m257-k127-n257",
            dims: (257, 127, 257),
            misaligned_b: true,
            exceptional: false,
            rounding_sensitive: false,
        },
        NtCase {
            label: "misaligned-live-big-exceptional",
            dims: (513, 513, 129),
            misaligned_b: true,
            exceptional: true,
            rounding_sensitive: false,
        },
    ] {
        let mut buffers = NtBuffers::new(&ctx, case);
        assert_repeated_outputs(&ctx, case, &mut buffers, oracle_bits);
    }
}

#[test]
#[ignore = "requires an exclusive SM80+ CUDA GPU"]
fn sm80plus_scalar_nt_thin_outputs_are_exact_and_graph_stable() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(device.compute_capability >= (8, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    let mut prior_launch_digest = None;
    for case in [
        NtCase {
            label: "thin-output-columns",
            dims: (512, 16, 2048),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: true,
        },
        NtCase {
            label: "thin-output-rows",
            dims: (16, 512, 2048),
            misaligned_b: false,
            exceptional: false,
            rounding_sensitive: true,
        },
    ] {
        let launch_digest = assert_thin_physical_evidence(&ctx, case);
        if let Some(prior) = prior_launch_digest.replace(launch_digest) {
            assert_ne!(prior, launch_digest, "thin NT cells share a launch digest");
        }
        let mut buffers = NtBuffers::new(&ctx, case);
        assert_repeated_outputs(&ctx, case, &mut buffers, splitk32_oracle_bits);
    }
}

#[test]
#[ignore = "requires an exclusive SM80+ CUDA GPU"]
fn sm80plus_scalar_nt_splitk_matches_its_frozen_f32_tree() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(device.compute_capability >= (8, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    let case = NtCase {
        label: "splitk32-fixed-f32-tree",
        dims: (32, 96, 128),
        misaligned_b: false,
        exceptional: false,
        rounding_sensitive: true,
    };
    let mut buffers = NtBuffers::new(&ctx, case);
    let expected = splitk32_oracle_bits(&buffers.a_host, &buffers.b_host, case.dims);
    assert_repeated_live_outputs(&ctx, case, &mut buffers, &expected);
}

#[test]
#[ignore = "requires an SM80+ CUDA GPU"]
fn sm80plus_f32_preparation_rejects_output_resource_aliases() {
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(device.compute_capability >= (8, 0));
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    let dims = (128, 128, 128);
    let matrix_elements = dims.0 * dims.1;
    let shared = GpuBuffer::zeros(&ctx.stream, matrix_elements * 2)
        .expect("allocate shared output and operand storage");
    let a = GpuBuffer::zeros(&ctx.stream, matrix_elements).expect("allocate independent A");
    let b = GpuBuffer::zeros(&ctx.stream, matrix_elements).expect("allocate independent B");
    ctx.stream.synchronize().expect("finish alias allocations");

    let base = shared.cached_ptr();
    let adjacent = base + u64::try_from(matrix_elements * 4).expect("adjacent operand offset");
    for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
        let request = F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        };
        let beta = if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 };
        let operands = F32TriadOperands {
            output: base,
            a: a.cached_ptr(),
            b: b.cached_ptr(),
            bias: None,
            alpha: 1.0,
            beta,
        };

        let error = prepare_f32_triad(
            &ctx,
            request,
            F32TriadOperands {
                a: base + 4,
                ..operands
            },
        )
        .err()
        .unwrap_or_else(|| panic!("{op:?} output/A overlap must be rejected"));
        assert!(
            error.contains("output overlaps the requested A input range"),
            "{op:?}: {error}"
        );

        let error = prepare_f32_triad(
            &ctx,
            request,
            F32TriadOperands {
                b: base + 4,
                ..operands
            },
        )
        .err()
        .unwrap_or_else(|| panic!("{op:?} output/B overlap must be rejected"));
        assert!(
            error.contains("output overlaps the requested B input range"),
            "{op:?}: {error}"
        );

        prepare_f32_triad(
            &ctx,
            request,
            F32TriadOperands {
                a: adjacent,
                ..operands
            },
        )
        .unwrap_or_else(|error| panic!("{op:?} adjacent output/A spans: {error}"));
        prepare_f32_triad(
            &ctx,
            request,
            F32TriadOperands {
                b: adjacent,
                ..operands
            },
        )
        .unwrap_or_else(|error| panic!("{op:?} adjacent output/B spans: {error}"));

        if op == ResolvedGemmOp::Nn {
            let error = prepare_f32_triad(
                &ctx,
                request,
                F32TriadOperands {
                    bias: Some(base + 4),
                    ..operands
                },
            )
            .err()
            .expect("NN output/bias overlap must be rejected");
            assert!(
                error.contains("output overlaps the requested bias range"),
                "{error}"
            );
            prepare_f32_triad(
                &ctx,
                request,
                F32TriadOperands {
                    bias: Some(adjacent),
                    ..operands
                },
            )
            .unwrap_or_else(|error| panic!("NN adjacent output/bias spans: {error}"));
        }
    }

    let mut live_output = GpuBuffer::zeros(&ctx.stream, matrix_elements)
        .expect("allocate live aliased output/B storage");
    ctx.stream
        .synchronize()
        .expect("finish live alias allocation");
    let live_b = live_output.cached_ptr();
    let error =
        gpu_gemm_bi_backward_dx_raw(&ctx, &mut live_output, &a, live_b, dims.0, dims.1, dims.2)
            .expect_err("live NT output/B overlap must be rejected");
    assert!(
        error.contains("output overlaps the requested B input range"),
        "{error}"
    );
}
