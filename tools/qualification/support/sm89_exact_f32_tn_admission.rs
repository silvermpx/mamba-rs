//! Test-only raw qualification probes for the sealed SM89 exact-F32 TN routes.
//!
//! The launch shapes, reference Split-M kernel, exceptional corpus and guarded
//! buffers are narrowed adaptations of the frozen discovery harnesses.  This
//! file deliberately compiles no CUDA source and does not participate in AUTO.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::{
    blas::gpu_gemm_bi_backward_dw_grad,
    buffers::{GpuBuffer, GradSlice},
    context::GpuCtx,
    gemm_bi_triad::{
        D768_IN_FUSED_SYMBOL, D768_OUT_RAW_SYMBOL, PRISM_RAW_SYMBOL, Sm89ExactF32DualChunkParams,
        Sm89ExactF32TnRoute,
    },
    graph_capture::capture_into_graph,
};

const GUARD_ELEMENTS: usize = 64;
const GUARD_BITS: u32 = 0x7fc0_3189;
const POISON_BITS: u32 = 0x7fc0_bbbb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawCell {
    pub dims: (usize, usize, usize),
    pub chunks: usize,
    pub m_chunk: usize,
}

impl RawCell {
    fn partial_grid(self) -> (u32, u32, u32) {
        let (_, k, n) = self.dims;
        (
            (k.div_ceil(128) * n.div_ceil(128)) as u32,
            1,
            self.chunks as u32,
        )
    }

    fn candidate_grid(self) -> (u32, u32, u32) {
        let (_, k, n) = self.dims;
        (
            (k.div_ceil(64) * n.div_ceil(64)) as u32,
            1,
            self.chunks as u32,
        )
    }

    fn fused_grid(self) -> (u32, u32, u32) {
        let (_, k, n) = self.dims;
        ((k.div_ceil(64) * n.div_ceil(64)) as u32, 1, 1)
    }

    fn reducer_grid(self) -> (u32, u32, u32) {
        let (_, k, n) = self.dims;
        ((k * n).div_ceil(256) as u32, 1, 1)
    }

    fn transpose_grid(self) -> (u32, u32, u32) {
        let (m, k, _) = self.dims;
        (k.div_ceil(32) as u32, m.div_ceil(32) as u32, 1)
    }

    fn validate(self) -> Result<(), String> {
        let (m, k, n) = self.dims;
        if m == 0 || k == 0 || n == 0 || k & 3 != 0 || n & 3 != 0 {
            return Err(format!("invalid raw exact-F32 TN cell {:?}", self.dims));
        }
        if m.div_ceil(self.m_chunk) != self.chunks || self.chunks < 2 {
            return Err(format!(
                "raw cell {:?} does not match {}x{} Split-M partition",
                self.dims, self.chunks, self.m_chunk
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawProbeEvidence {
    pub compared_partial_words: usize,
    pub compared_transpose_words: usize,
    pub compared_output_words: usize,
    pub guarded_allocations: usize,
    pub guarded_elements: usize,
    pub output_bits: Vec<u32>,
}

struct GuardedBuffer {
    gpu: GpuBuffer,
    seed: Vec<f32>,
    active: usize,
    label: &'static str,
}

impl GuardedBuffer {
    fn new(ctx: &GpuCtx, values: Vec<f32>, label: &'static str) -> Result<Self, String> {
        let active = values.len();
        let mut seed = vec![f32::from_bits(GUARD_BITS); GUARD_ELEMENTS];
        seed.extend(values);
        seed.resize(
            GUARD_ELEMENTS + active + GUARD_ELEMENTS,
            f32::from_bits(GUARD_BITS),
        );
        let gpu = GpuBuffer::from_cpu(&ctx.stream, &seed)?;
        if (gpu.cached_ptr() + (GUARD_ELEMENTS * std::mem::size_of::<f32>()) as u64) & 255 != 0 {
            return Err(format!("{label} active origin is not 256-byte aligned"));
        }
        Ok(Self {
            gpu,
            seed,
            active,
            label,
        })
    }

    fn ptr(&self) -> u64 {
        self.gpu.cached_ptr() + (GUARD_ELEMENTS * std::mem::size_of::<f32>()) as u64
    }

    fn active_bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        let values = self.gpu.to_cpu(&ctx.stream)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize {} readback: {error:?}", self.label))?;
        let active_begin = GUARD_ELEMENTS;
        let active_end = active_begin + self.active;
        if values[..active_begin]
            .iter()
            .chain(&values[active_end..])
            .any(|value| value.to_bits() != GUARD_BITS)
        {
            return Err(format!("{} leading/trailing guard changed", self.label));
        }
        Ok(values[active_begin..active_end]
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    fn unchanged(&self, ctx: &GpuCtx) -> Result<(), String> {
        let actual = self.active_bits(ctx)?;
        if actual
            .iter()
            .zip(&self.seed[GUARD_ELEMENTS..GUARD_ELEMENTS + self.active])
            .any(|(left, right)| *left != right.to_bits())
        {
            return Err(format!("{} active input changed", self.label));
        }
        Ok(())
    }
}

fn config(grid: (u32, u32, u32), block: (u32, u32, u32)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    }
}

fn exceptional_values(values: &mut [f32], row_width: usize, row_starts: &[usize]) {
    let cases: [u32; 10] = [
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
        for &row in row_starts {
            let index = row.saturating_add(case).saturating_mul(row_width);
            if index < values.len() {
                values[index] = f32::from_bits(bits);
            }
        }
    }
}

fn launch_reference_partials(
    ctx: &GpuCtx,
    cell: RawCell,
    partial: u64,
    x: u64,
    dy: u64,
) -> Result<(), String> {
    let (m, k, n) = cell.dims;
    let (m, k, n, m_chunk) = (m as i32, k as i32, n as i32, cell.m_chunk as i32);
    let mut builder = ctx
        .stream
        .launch_builder(&ctx.kernels.gemm_bi_tn_splitm_partial_aligned);
    builder.arg(&partial);
    builder.arg(&x);
    builder.arg(&dy);
    builder.arg(&m);
    builder.arg(&k);
    builder.arg(&n);
    builder.arg(&m_chunk);
    unsafe { builder.launch(config(cell.partial_grid(), (256, 1, 1))) }
        .map(|_| ())
        .map_err(|error| format!("launch retained Split-M partial: {error:?}"))
}

fn launch_reducer(
    ctx: &GpuCtx,
    cell: RawCell,
    output: u64,
    partial: u64,
    alpha: f32,
) -> Result<(), String> {
    let (_, k, n) = cell.dims;
    let (k, n, chunks) = (k as i32, n as i32, cell.chunks as i32);
    let mut builder = ctx
        .stream
        .launch_builder(&ctx.kernels.gemm_bi_splitm_reduce);
    builder.arg(&output);
    builder.arg(&partial);
    builder.arg(&alpha);
    builder.arg(&k);
    builder.arg(&n);
    builder.arg(&chunks);
    unsafe { builder.launch(config(cell.reducer_grid(), (256, 1, 1))) }
        .map(|_| ())
        .map_err(|error| format!("launch retained Split-M reducer: {error:?}"))
}

fn launch_transpose(ctx: &GpuCtx, cell: RawCell, output: u64, input: u64) -> Result<(), String> {
    let (m, k, _) = cell.dims;
    let (rows, columns) = (m as i32, k as i32);
    let mut builder = ctx
        .stream
        .launch_builder(&ctx.kernels.gemm_bi_transpose_f32_32x16_d768_v1);
    builder.arg(&output);
    builder.arg(&input);
    builder.arg(&rows);
    builder.arg(&columns);
    unsafe { builder.launch(config(cell.transpose_grid(), (32, 16, 1))) }
        .map(|_| ())
        .map_err(|error| format!("launch retained F32 transpose: {error:?}"))
}

fn launch_candidate(
    ctx: &GpuCtx,
    route: Sm89ExactF32TnRoute,
    cell: RawCell,
    output: u64,
    candidate_partial: u64,
    transposed: u64,
    x: u64,
    dy: u64,
    alpha: f32,
) -> Result<(), String> {
    let (m, k, n) = cell.dims;
    match route {
        Sm89ExactF32TnRoute::D768InDualChunkFused => {
            launch_transpose(ctx, cell, transposed, x)?;
            let function = ctx
                .kernels
                .triad_sm89_exact_f32_function(D768_IN_FUSED_SYMBOL)
                .ok_or_else(|| format!("missing {D768_IN_FUSED_SYMBOL}"))?;
            let params = Sm89ExactF32DualChunkParams {
                alpha,
                m: k as i32,
                n: n as i32,
                k0: cell.m_chunk as i32,
                k1: (m - cell.m_chunk) as i32,
                lda: m as i32,
                ldb: n as i32,
                ldc: n as i32,
            };
            let mut builder = ctx.stream.launch_builder(function);
            builder.arg(&output);
            builder.arg(&transposed);
            builder.arg(&dy);
            builder.arg(&params);
            unsafe { builder.launch(config(cell.fused_grid(), (128, 1, 1))) }
                .map(|_| ())
                .map_err(|error| format!("launch {D768_IN_FUSED_SYMBOL}: {error:?}"))
        }
        Sm89ExactF32TnRoute::D768OutDirectBk16 | Sm89ExactF32TnRoute::PrismDirectBk16 => {
            let symbol = match route {
                Sm89ExactF32TnRoute::D768OutDirectBk16 => D768_OUT_RAW_SYMBOL,
                Sm89ExactF32TnRoute::PrismDirectBk16 => PRISM_RAW_SYMBOL,
                Sm89ExactF32TnRoute::D768InDualChunkFused => unreachable!(),
            };
            let function = ctx
                .kernels
                .triad_sm89_exact_f32_function(symbol)
                .ok_or_else(|| format!("missing {symbol}"))?;
            let (m, k, n, m_chunk) = (m as i32, k as i32, n as i32, cell.m_chunk as i32);
            let mut builder = ctx.stream.launch_builder(function);
            builder.arg(&candidate_partial);
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&m);
            builder.arg(&k);
            builder.arg(&n);
            builder.arg(&m_chunk);
            unsafe { builder.launch(config(cell.candidate_grid(), (128, 1, 1))) }
                .map_err(|error| format!("launch {symbol}: {error:?}"))?;
            launch_reducer(ctx, cell, output, candidate_partial, alpha)
        }
    }
}

fn compare_words(label: &str, actual: &[u32], expected: &[u32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} word count changed: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        return Err(format!(
            "{label} mismatch at {index}: {:08x} != {:08x}",
            actual[index], expected[index]
        ));
    }
    Ok(())
}

pub fn raw_seed_words(
    cell: RawCell,
    exceptional: bool,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), String> {
    cell.validate()?;
    let (m, k, n) = cell.dims;
    let mut x = crate::full_mantissa::finite_full_mantissa_values(m * k, 0x8931_a001);
    let mut dy = crate::full_mantissa::finite_full_mantissa_values(m * n, 0x8931_b002);
    let output_seed = crate::full_mantissa::finite_full_mantissa_values(k * n, 0x8931_c003);
    if exceptional {
        let rows = (0..cell.chunks)
            .map(|chunk| chunk * cell.m_chunk)
            .collect::<Vec<_>>();
        exceptional_values(&mut x, k, &rows);
        exceptional_values(&mut dy, n, &rows);
    }
    Ok((
        output_seed.into_iter().map(f32::to_bits).collect(),
        x.into_iter().map(f32::to_bits).collect(),
        dy.into_iter().map(f32::to_bits).collect(),
    ))
}

pub fn run_raw_probe(
    ctx: &GpuCtx,
    route: Sm89ExactF32TnRoute,
    cell: RawCell,
    alpha: f32,
    exceptional: bool,
) -> Result<RawProbeEvidence, String> {
    run_raw_probe_repeated(ctx, route, cell, alpha, exceptional, 1)
}

pub fn run_raw_probe_repeated(
    ctx: &GpuCtx,
    route: Sm89ExactF32TnRoute,
    cell: RawCell,
    alpha: f32,
    exceptional: bool,
    repeats: usize,
) -> Result<RawProbeEvidence, String> {
    if repeats == 0 {
        return Err("raw exact-F32 TN probe requires at least one repeat".into());
    }
    let (output_words, x_words, dy_words) = raw_seed_words(cell, exceptional)?;
    let x = x_words.iter().copied().map(f32::from_bits).collect();
    let dy = dy_words.iter().copied().map(f32::from_bits).collect();
    let output_seed = output_words
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let (m, k, n) = cell.dims;
    let x = GuardedBuffer::new(ctx, x, "raw X")?;
    let dy = GuardedBuffer::new(ctx, dy, "raw dY")?;
    let candidate_output = GuardedBuffer::new(ctx, output_seed.clone(), "candidate output")?;
    let reference_output = GuardedBuffer::new(ctx, output_seed, "reference output")?;
    let partial_len = cell.chunks * k * n;
    let candidate_partial = GuardedBuffer::new(
        ctx,
        vec![f32::from_bits(POISON_BITS); partial_len],
        "candidate scratch",
    )?;
    let reference_partial = GuardedBuffer::new(
        ctx,
        vec![f32::from_bits(POISON_BITS.rotate_left(3)); partial_len],
        "reference scratch",
    )?;
    let transposed = GuardedBuffer::new(
        ctx,
        vec![f32::from_bits(POISON_BITS.rotate_left(7)); m * k],
        "transpose scratch",
    )?;

    launch_reference_partials(ctx, cell, reference_partial.ptr(), x.ptr(), dy.ptr())?;
    for _ in 0..repeats {
        launch_candidate(
            ctx,
            route,
            cell,
            candidate_output.ptr(),
            candidate_partial.ptr(),
            transposed.ptr(),
            x.ptr(),
            dy.ptr(),
            alpha,
        )?;
    }

    let mut compared_partial_words = 0;
    let mut compared_transpose_words = 0;
    if route == Sm89ExactF32TnRoute::D768InDualChunkFused {
        let transposed_bits = transposed.active_bits(ctx)?;
        let x_bits = x.active_bits(ctx)?;
        let mut expected = vec![0_u32; x_bits.len()];
        for row in 0..m {
            for column in 0..k {
                expected[column * m + row] = x_bits[row * k + column];
            }
        }
        compare_words("transpose scratch", &transposed_bits, &expected)?;
        compared_transpose_words = expected.len();
    } else {
        let candidate_bits = candidate_partial.active_bits(ctx)?;
        let reference_bits = reference_partial.active_bits(ctx)?;
        compare_words("raw Split-M partial", &candidate_bits, &reference_bits)?;
        compared_partial_words = reference_bits.len();
    }

    for _ in 0..repeats {
        launch_reducer(
            ctx,
            cell,
            reference_output.ptr(),
            reference_partial.ptr(),
            alpha,
        )?;
    }
    let candidate_bits = candidate_output.active_bits(ctx)?;
    let reference_bits = reference_output.active_bits(ctx)?;
    compare_words(
        "raw candidate chain output",
        &candidate_bits,
        &reference_bits,
    )?;
    x.unchanged(ctx)?;
    dy.unchanged(ctx)?;

    // active_bits validates every trailing guard.  Count every allocation so
    // the caller can require the same two-sided inventory for every corpus.
    for buffer in [
        &candidate_output,
        &reference_output,
        &candidate_partial,
        &reference_partial,
        &transposed,
    ] {
        let _ = buffer.active_bits(ctx)?;
    }
    Ok(RawProbeEvidence {
        compared_partial_words,
        compared_transpose_words,
        compared_output_words: reference_bits.len(),
        guarded_allocations: 7,
        guarded_elements: 7 * 2 * GUARD_ELEMENTS,
        output_bits: candidate_bits,
    })
}

pub fn run_k0_auto_probe(ctx: &GpuCtx, k: usize, n: usize) -> Result<(), String> {
    let output_seed = crate::full_mantissa::finite_full_mantissa_values(k * n, 0x8931_c003);
    let expected = crate::tn_k0_expected_output_bits(1.0, &output_seed);
    let output = GuardedBuffer::new(ctx, output_seed, "K0 output")?;
    let x = GuardedBuffer::new(ctx, Vec::new(), "K0 X")?;
    let dy = GuardedBuffer::new(ctx, Vec::new(), "K0 dY")?;
    let launch = || {
        gpu_gemm_bi_backward_dw_grad(
            ctx,
            &GradSlice::from_raw(output.ptr(), k * n),
            &dy.gpu,
            &x.gpu,
            0,
            k,
            n,
        )
    };
    launch()?;
    compare_words("K0 eager output", &output.active_bits(ctx)?, &expected)?;
    x.unchanged(ctx)?;
    dy.unchanged(ctx)?;
    let graph = unsafe { capture_into_graph(&ctx.stream, launch) }?;
    graph
        .launch()
        .map_err(|error| format!("K0 graph launch: {error:?}"))?;
    compare_words("K0 graph output", &output.active_bits(ctx)?, &expected)?;
    x.unchanged(ctx)?;
    dy.unchanged(ctx)
}
