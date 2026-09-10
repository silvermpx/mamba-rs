//! Kernel-level GEMM timing on one board, one copy per endpoint (the old
//! main and the release), so the deterministic routes of both trees can be
//! compared with each other and with cuBLAS Fast and Pedantic under one
//! protocol. Every cell times three arms in the same process: the tree's
//! deterministic route, cuBLAS in the tree's fast setting and cuBLAS in the
//! tree's pedantic setting. The vendor arms double as a cross-tree control:
//! both trees call the same cuBLAS, so their vendor timings must agree.
//!
//! Cells: the seven training shapes of the release packets for the Triad
//! family (NN forward, TN weight gradient, NT input gradient) and the five
//! serving shapes for the forward-only family, in f32, bf16 and f16. Events
//! on the context stream time windows of calibrated launch counts; the arms
//! rotate inside every window in mirrored order so drift cancels.

#[cfg(feature = "cuda")]
mod bench {
    use mamba_rs::mamba_ssm::gpu::blas::{
        TypedPtr, gpu_gemm_bi_backward_dw_grad, gpu_gemm_bi_backward_dw_grad_typed,
        gpu_gemm_bi_backward_dx_raw, gpu_gemm_bi_forward_raw, gpu_gemm_ex_backward_dx_typed,
        gpu_gemm_typed_forward_raw,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::GemmMode;

    const WINDOWS: usize = 21;
    const WINDOW_TARGET_MS: f32 = 4.0;
    const MIN_ITERS: usize = 8;
    const WARMUP: usize = 3;

    /// (name, M, K, N): `Y[M,N] = X[M,K] W[K,N]`.
    const TRIAD_CELLS: [(&str, usize, usize, usize); 7] = [
        ("d128_in_proj", 1024, 128, 512),
        ("d128_out_proj", 1024, 256, 128),
        ("underfill", 256, 512, 384),
        ("d768_in_proj", 2048, 768, 3072),
        ("d768_out_proj", 2048, 1536, 768),
        ("large_deep", 4096, 3072, 1536),
        ("prism_in_proj", 4621, 384, 1928),
    ];
    const INFERENCE_CELLS: [(&str, usize, usize, usize); 5] = [
        ("hot_a", 4621, 384, 1928),
        ("hot_b", 4621, 768, 2304),
        ("hot_c", 4621, 1928, 384),
        ("hot_d", 2048, 768, 2304),
        ("hot_e", 2048, 2304, 768),
    ];
    const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];
    const GEMM_CONTROLS: [&str; 8] = [
        "MAMBA_RS_GEMM_MODE",
        "MAMBA_RS_BATCH_INVARIANT",
        "MAMBA_RS_BI_GEMM_FAMILY",
        "MAMBA_RS_BI_TENSOR_CORES",
        "MAMBA_RS_FAST_GEMM",
        "MAMBA_RS_BI_F32_POLICY",
        "MAMBA_RS_BI_HALF_POLICY",
        "MAMBA_RS_ARCH_RUNG",
    ];

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Arm {
        Deterministic,
        Fast,
        Pedantic,
    }

    impl Arm {
        fn label(self) -> &'static str {
            match self {
                Arm::Deterministic => "det",
                Arm::Fast => "fast",
                Arm::Pedantic => "pedantic",
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Op {
        Nn,
        Tn,
        Nt,
    }

    impl Op {
        fn label(self) -> &'static str {
            match self {
                Op::Nn => "nn",
                Op::Tn => "tn",
                Op::Nt => "nt",
            }
        }
    }

    fn dtype_label(dtype: WeightDtype) -> &'static str {
        match dtype {
            WeightDtype::F32 => "f32",
            WeightDtype::Bf16 => "bf16",
            WeightDtype::F16 => "f16",
        }
    }

    fn refuse_ambient_controls() {
        for name in GEMM_CONTROLS {
            assert!(
                std::env::var_os(name).is_none(),
                "{name} is set; the runner must clear every GEMM control so the adapter alone selects the route"
            );
        }
    }

    /// One context per arm: the mode is part of the context, and the old
    /// tree cannot re-enable TF32 on a handle once it was disabled.
    fn make_ctx(device: &GpuDevice, arm: Arm, family: BiGemmFamily, tensor_cores: bool) -> GpuCtx {
        let mode = match arm {
            Arm::Deterministic => GemmMode::Deterministic,
            Arm::Fast => GemmMode::CublasFast,
            Arm::Pedantic => GemmMode::CublasPedantic,
        };
        let ctx = GpuCtx::new_with_mode(device, mode).expect("context");
        if arm == Arm::Deterministic {
            ctx.set_bi_gemm_family(family);
            ctx.set_bi_tensor_cores(tensor_cores);
        }
        ctx
    }

    fn describe_ctx(ctx: &GpuCtx) -> String {
        format!(
            "batch_invariant={} family={:?} tensor_cores={} fast_gemm={} tf32={}",
            ctx.batch_invariant(),
            ctx.bi_gemm_family(),
            ctx.bi_tensor_cores(),
            ctx.fast_gemm(),
            ctx.tf32()
        )
    }

    fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
            })
            .collect()
    }

    fn f32_to_bf16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let round = 0x7FFF + ((bits >> 16) & 1);
        ((bits.wrapping_add(round)) >> 16) as u16
    }

    fn f32_to_f16_bits(v: f32) -> u16 {
        // Values here stay well inside the normal f16 range.
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = bits & 0x7F_FFFF;
        // Values below the f16 normal range flush to zero; the operands are
        // scaled so this touches only a negligible fraction of them, and every
        // arm of a cell sees the same rounded inputs.
        if v == 0.0 || exp < 1 {
            return sign;
        }
        assert!(exp < 31, "f16 conversion above the normal range");
        let mut h = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        let rem = mant & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h = h.wrapping_add(1);
        }
        h
    }

    /// A device operand in the requested storage: f32 buffers keep the
    /// crate's own buffer type, half buffers are raw u16 slices.
    enum Operand {
        F32(GpuBuffer),
        Half(cudarc::driver::CudaSlice<u16>),
    }

    impl Operand {
        fn upload(ctx: &GpuCtx, dtype: WeightDtype, values: &[f32]) -> Self {
            match dtype {
                WeightDtype::F32 => Operand::F32(GpuBuffer::from_cpu(&ctx.stream, values).expect("f32 upload")),
                WeightDtype::Bf16 | WeightDtype::F16 => {
                    let bits: Vec<u16> = values
                        .iter()
                        .map(|&v| {
                            if dtype == WeightDtype::Bf16 {
                                f32_to_bf16_bits(v)
                            } else {
                                f32_to_f16_bits(v)
                            }
                        })
                        .collect();
                    Operand::Half(ctx.stream.clone_htod(&bits).expect("half upload"))
                }
            }
        }

        fn ptr(&self, ctx: &GpuCtx) -> cudarc::driver::sys::CUdeviceptr {
            use cudarc::driver::DevicePtr;
            match self {
                Operand::F32(b) => b.cached_ptr(),
                Operand::Half(s) => {
                    let (p, _r) = s.device_ptr(&ctx.stream);
                    p
                }
            }
        }

        fn typed(&self, ctx: &GpuCtx, dtype: WeightDtype) -> TypedPtr {
            TypedPtr { ptr: self.ptr(ctx), dtype }
        }

        fn download(&self, ctx: &GpuCtx, dtype: WeightDtype) -> Vec<f32> {
            match self {
                Operand::F32(b) => b.to_cpu(&ctx.stream).expect("download"),
                Operand::Half(s) => {
                    let bits: Vec<u16> = ctx.stream.clone_dtoh(s).expect("download");
                    bits.iter()
                        .map(|&h| match dtype {
                            WeightDtype::Bf16 => f32::from_bits(u32::from(h) << 16),
                            _ => {
                                let sign = u32::from(h & 0x8000) << 16;
                                let exp = i32::from((h >> 10) & 0x1F);
                                let mant = u32::from(h & 0x3FF);
                                if exp == 0 {
                                    return f32::from_bits(sign);
                                }
                                f32::from_bits(sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
                            }
                        })
                        .collect()
                }
            }
        }
    }

    /// The operands of one cell in one storage; `a` is `X[M,K]`, `w` is
    /// `W[K,N]`, `dy` is `dY[M,N]`, and the outputs are sized per op.
    struct Cell {
        m: usize,
        k: usize,
        n: usize,
        x: Operand,
        w: Operand,
        dy: Operand,
        y: Operand,
        dx: Operand,
        dw: GpuBuffer,
    }

    impl Cell {
        fn new(ctx: &GpuCtx, dtype: WeightDtype, m: usize, k: usize, n: usize) -> Self {
            let x = det(m * k, 11, 1.0);
            let w = det(k * n, 13, 0.05);
            let dy = det(m * n, 17, 0.05);
            Cell {
                m,
                k,
                n,
                x: Operand::upload(ctx, dtype, &x),
                w: Operand::upload(ctx, dtype, &w),
                dy: Operand::upload(ctx, dtype, &dy),
                y: Operand::upload(ctx, dtype, &vec![0.0; m * n]),
                dx: Operand::upload(ctx, dtype, &vec![0.0; m * k]),
                dw: GpuBuffer::from_cpu(&ctx.stream, &vec![0.0; k * n]).expect("dw"),
            }
        }

        fn dims(&self) -> (usize, usize, usize) {
            (self.m, self.k, self.n)
        }

        fn launch(&mut self, ctx: &GpuCtx, dtype: WeightDtype, op: Op) {
            let dims = self.dims();
            match (op, dtype) {
                (Op::Nn, WeightDtype::F32) => {
                    let Operand::F32(y) = &mut self.y else { unreachable!() };
                    let Operand::F32(x) = &self.x else { unreachable!() };
                    gpu_gemm_bi_forward_raw(ctx, y, x, self.w.ptr(ctx), None, dims).expect("nn f32");
                }
                (Op::Nn, _) => {
                    gpu_gemm_typed_forward_raw(
                        ctx,
                        self.y.typed(ctx, dtype),
                        self.x.typed(ctx, dtype),
                        self.w.typed(ctx, dtype),
                        None,
                        dims,
                    )
                    .expect("nn typed");
                }
                (Op::Tn, WeightDtype::F32) => {
                    let Operand::F32(dy) = &self.dy else { unreachable!() };
                    let Operand::F32(x) = &self.x else { unreachable!() };
                    let dw = GradSlice::from_raw(self.dw.cached_ptr(), self.k * self.n);
                    gpu_gemm_bi_backward_dw_grad(ctx, &dw, dy, x, self.m, self.k, self.n).expect("tn f32");
                }
                (Op::Tn, _) => {
                    let dw = GradSlice::from_raw(self.dw.cached_ptr(), self.k * self.n);
                    gpu_gemm_bi_backward_dw_grad_typed(
                        ctx,
                        &dw,
                        self.dy.typed(ctx, dtype),
                        self.x.typed(ctx, dtype),
                        self.m,
                        self.k,
                        self.n,
                    )
                    .expect("tn typed");
                }
                (Op::Nt, WeightDtype::F32) => {
                    let Operand::F32(dx) = &mut self.dx else { unreachable!() };
                    let Operand::F32(dy) = &self.dy else { unreachable!() };
                    gpu_gemm_bi_backward_dx_raw(ctx, dx, dy, self.w.ptr(ctx), self.m, self.k, self.n)
                        .expect("nt f32");
                }
                (Op::Nt, _) => {
                    gpu_gemm_ex_backward_dx_typed(
                        ctx,
                        self.dx.typed(ctx, dtype),
                        self.dy.typed(ctx, dtype),
                        self.w.typed(ctx, dtype),
                        self.m,
                        self.k,
                        self.n,
                    )
                    .expect("nt typed");
                }
            }
        }

        /// The output of the op, downloaded as f32, after one launch from a
        /// zeroed accumulator (so the TN arm compares one product).
        fn output_once(&mut self, ctx: &GpuCtx, dtype: WeightDtype, op: Op) -> Vec<f32> {
            self.dw.zero(&ctx.stream).expect("zero dw");
            self.launch(ctx, dtype, op);
            ctx.stream.synchronize().expect("sync");
            match op {
                Op::Nn => self.y.download(ctx, dtype),
                Op::Tn => self.dw.to_cpu(&ctx.stream).expect("dw download"),
                Op::Nt => self.dx.download(ctx, dtype),
            }
        }
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            num += (f64::from(*x) - f64::from(*y)).powi(2);
            den += f64::from(*y).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    fn timed_window(ctx: &GpuCtx, cell: &mut Cell, dtype: WeightDtype, op: Op, iters: usize) -> f32 {
        let start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("start event");
        for _ in 0..iters {
            cell.launch(ctx, dtype, op);
        }
        let end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("end event");
        end.synchronize().expect("event sync");
        start.elapsed_ms(&end).expect("elapsed")
    }

    fn calibrate(ctx: &GpuCtx, cell: &mut Cell, dtype: WeightDtype, op: Op) -> usize {
        for _ in 0..WARMUP {
            cell.launch(ctx, dtype, op);
        }
        ctx.stream.synchronize().expect("sync");
        let probe = timed_window(ctx, cell, dtype, op, MIN_ITERS);
        let per_launch = probe / MIN_ITERS as f32;
        ((WINDOW_TARGET_MS / per_launch).round() as usize).max(MIN_ITERS)
    }

    fn quantile(sorted: &[f32], q: f64) -> f32 {
        let idx = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len()) - 1;
        sorted[idx]
    }

    struct Family {
        label: &'static str,
        family: BiGemmFamily,
        cells: &'static [(&'static str, usize, usize, usize)],
        ops: &'static [Op],
    }

    fn run_family(device: &GpuDevice, fam: &Family) {
        for dtype in DTYPES {
            let tensor_cores = dtype != WeightDtype::F32;
            let arms = [Arm::Deterministic, Arm::Fast, Arm::Pedantic];
            let ctxs: Vec<GpuCtx> = arms
                .iter()
                .map(|&arm| make_ctx(device, arm, fam.family, tensor_cores))
                .collect();
            for (arm, ctx) in arms.iter().zip(&ctxs) {
                println!(
                    "CONTEXT family={} dtype={} arm={} {}",
                    fam.label,
                    dtype_label(dtype),
                    arm.label(),
                    describe_ctx(ctx)
                );
            }
            for &(name, m, k, n) in fam.cells {
                for &op in fam.ops {
                    // One operand set per arm, uploaded through that arm's
                    // context; the values are identical.
                    let mut cells: Vec<Cell> = ctxs.iter().map(|ctx| Cell::new(ctx, dtype, m, k, n)).collect();
                    // Numeric sanity: every arm computes the same product.
                    let outputs: Vec<Vec<f32>> = cells
                        .iter_mut()
                        .zip(&ctxs)
                        .map(|(cell, ctx)| cell.output_once(ctx, dtype, op))
                        .collect();
                    let reference = &outputs[2];
                    for (arm, out) in arms.iter().zip(&outputs) {
                        assert!(out.iter().all(|v| v.is_finite()), "{name} {op:?} {arm:?}: non-finite output");
                        println!(
                            "CHECK family={} cell={} op={} dtype={} arm={} rel_l2_vs_pedantic={:.3e}",
                            fam.label,
                            name,
                            op.label(),
                            dtype_label(dtype),
                            arm.label(),
                            rel_l2(out, reference)
                        );
                    }
                    let iters: Vec<usize> = cells
                        .iter_mut()
                        .zip(&ctxs)
                        .map(|(cell, ctx)| calibrate(ctx, cell, dtype, op))
                        .collect();
                    let mut samples: Vec<Vec<f32>> = vec![Vec::new(); arms.len()];
                    for window in 0..WINDOWS {
                        let order: Vec<usize> = if window % 2 == 0 {
                            (0..arms.len()).collect()
                        } else {
                            (0..arms.len()).rev().collect()
                        };
                        for i in order {
                            let ms = timed_window(&ctxs[i], &mut cells[i], dtype, op, iters[i]);
                            samples[i].push(ms * 1000.0 / iters[i] as f32);
                        }
                    }
                    for (i, arm) in arms.iter().enumerate() {
                        let mut sorted = samples[i].clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        println!(
                            "TIMING family={} cell={} op={} dtype={} m={m} k={k} n={n} arm={} iters={} windows={} p05_us={:.3} p50_us={:.3} p95_us={:.3}",
                            fam.label,
                            name,
                            op.label(),
                            dtype_label(dtype),
                            arm.label(),
                            iters[i],
                            WINDOWS,
                            quantile(&sorted, 0.05),
                            quantile(&sorted, 0.50),
                            quantile(&sorted, 0.95),
                        );
                    }
                }
            }
        }
    }

    pub fn run() {
        refuse_ambient_controls();
        let device = GpuDevice::new(0).expect("device");
        println!("DEVICE {:?}", device.identity());
        run_family(
            &device,
            &Family {
                label: "triad",
                family: BiGemmFamily::Triad,
                cells: &TRIAD_CELLS,
                ops: &[Op::Nn, Op::Tn, Op::Nt],
            },
        );
        run_family(
            &device,
            &Family {
                label: "inference",
                family: BiGemmFamily::Inference,
                cells: &INFERENCE_CELLS,
                ops: &[Op::Nn],
            },
        );
        println!("DONE");
    }
}

#[cfg(feature = "cuda")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}
