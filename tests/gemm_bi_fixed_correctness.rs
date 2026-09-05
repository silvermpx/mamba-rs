//! Correctness of the FIXED-tile batch-invariant GEMM
//! (`kernels/gemm_bi_fixed/`, `BiGemmFamily::Fixed`) against a CPU
//! reference, across shapes that exercise the tile tails.
//!
//! The family had no direct test while it sat off every dispatch path;
//! it has one now, and it is the gate any tile-geometry change must
//! pass before a benchmark number means anything.
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_bi_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedTile, fixed_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 as f32 / u32::MAX as f32) * 0.4 - 0.2
        })
        .collect()
}

/// `y[m][n] = sum_k x[m][k] * w[k][n]`, ascending k.
fn cpu_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w[kk * n + j];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

#[test]
#[ignore = "needs a CUDA device"]
fn fixed_tile_matches_cpu_across_tails() {
    let dev = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&dev).expect("ctx");
    let stream = ctx.stream.clone();

    // (m, k, n): exact tile multiples first, then every tail combination
    // (m not a multiple of the M tile, n not of the N tile, k not of the
    // K tile), then a production-shaped case.
    let shapes: &[(usize, usize, usize)] = &[
        (64, 32, 64),
        (128, 64, 128),
        (127, 64, 128),
        (128, 64, 127),
        (128, 63, 128),
        (127, 63, 127),
        (321, 96, 193),
        (4621, 384, 384),
    ];

    for &(m, k, n) in shapes {
        let x_host = synth(m * k, 0xA11CE ^ m as u64);
        let w_host = synth(k * n, 0xB0B ^ n as u64);
        let want = cpu_ref(&x_host, &w_host, m, k, n);

        let x = GpuBuffer::from_cpu(&stream, &x_host).expect("x");
        let w = GpuBuffer::from_cpu(&stream, &w_host).expect("w");
        let mut y = GpuBuffer::zeros(&stream, m * n).expect("y");

        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
        gpu_gemm_bi_forward_raw(&ctx, &mut y, &x, w.raw_ptr(&stream), None, (m, k, n))
            .expect("fixed forward");
        let got = y.to_cpu(&stream).expect("d2h");
        ctx.set_batch_invariant(false);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);

        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        for i in 0..m {
            for j in 0..n {
                let d = (got[i * n + j] - want[i * n + j]).abs();
                if d > worst {
                    worst = d;
                    worst_at = (i, j);
                }
            }
        }
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        println!(
            "m={m:<5} k={k:<4} n={n:<5} worst |gpu-cpu| = {worst:.3e} at {worst_at:?} (scale {scale:.3e})"
        );
        assert!(
            worst <= 2e-4 * scale,
            "m={m} k={k} n={n}: fixed tile differs from the CPU reference by {worst:.3e} \
             at {worst_at:?} (scale {scale:.3e})"
        );
    }
}

fn f32_pointer(ptr: u64) -> TypedPtr {
    TypedPtr {
        ptr,
        dtype: WeightDtype::F32,
    }
}

fn output_bits(ctx: &GpuCtx, output: &GpuBuffer) -> Vec<u32> {
    output
        .to_cpu(&ctx.stream)
        .expect("output download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn assert_wide_graph_symbol(graph: &cudarc::driver::CudaGraph) {
    use cudarc::driver::sys;
    let mut count = 0;
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
        sys::CUresult::CUDA_SUCCESS,
    );
    assert_eq!(count, 1, "forced wide must capture its one tiled kernel");
    let mut node = std::ptr::null_mut();
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
        sys::CUresult::CUDA_SUCCESS,
    );
    let mut params = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
        sys::CUresult::CUDA_SUCCESS,
    );
    let mut name = std::ptr::null();
    assert_eq!(
        unsafe { sys::cuFuncGetName(&mut name, params.func) },
        sys::CUresult::CUDA_SUCCESS,
    );
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes(),
        b"gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
        "forced Fixed wide silently captured another route",
    );
}

#[test]
#[ignore = "requires a CUDA device with the Triad TF32 wide symbol bound"]
fn fixed_tf32_forced_wide_preserves_numeric_prefix_subview_and_graph_bits() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let tile = FixedTile::Tf32M128N128S3;
    let (rows, k, n) = (274, 36, 132);
    let sizes = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257];
    let finite_a: Vec<_> = synth(rows * k, 0xa128)
        .into_iter()
        .map(|x| x * 4.0 + 0.4)
        .collect();
    let b_host: Vec<_> = synth(k * n, 0xb128)
        .into_iter()
        .map(|x| x * 4.0 + 0.4)
        .collect();
    let bias_host = synth(n, 0xb1a5);
    let b = GpuBuffer::from_cpu(&ctx.stream, &b_host).expect("B upload");
    let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias upload");

    for exceptional in [false, true] {
        let mut a_host = finite_a.clone();
        if exceptional {
            // Payloads are compared within wide: the old Fixed converter has
            // a different NaN contract. Include signed zeros and both infinities.
            let values = [
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7f80_1000),
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0xff80_1000),
                0.125,
            ];
            for row in 0..rows {
                a_host[row * k] = values[row % values.len()];
            }
        }
        let a = GpuBuffer::from_cpu(&ctx.stream, &a_host).expect("A upload");
        for has_bias in [false, true] {
            let full = GpuBuffer::zeros(&ctx.stream, rows * n).expect("full output");
            let operands = FixedFwdOperands {
                c: f32_pointer(full.cached_ptr()),
                x: f32_pointer(a.cached_ptr()),
                w: f32_pointer(b.cached_ptr()),
                bias_ptr: has_bias.then_some(bias.cached_ptr()),
            };
            let full_shape = FixedShape { m: rows, k, n };
            fixed_forward_with_tile(&ctx, operands, full_shape, tile)
                .expect("forced wide full reference launch");
            let reference = output_bits(&ctx, &full);
            if exceptional {
                assert!(reference.iter().any(|bits| f32::from_bits(*bits).is_nan()));
                assert!(
                    reference
                        .iter()
                        .any(|bits| f32::from_bits(*bits).is_infinite())
                );
            } else {
                let baseline = GpuBuffer::zeros(&ctx.stream, rows * n).expect("Fixed baseline");
                fixed_forward_with_tile(
                    &ctx,
                    FixedFwdOperands {
                        c: f32_pointer(baseline.cached_ptr()),
                        ..operands
                    },
                    full_shape,
                    FixedTile::Tf32M64S2,
                )
                .expect("finite Fixed baseline launch");
                assert_eq!(
                    reference,
                    output_bits(&ctx, &baseline),
                    "finite wide/Fixed arithmetic differs; bias={has_bias}"
                );
                let want = cpu_ref(&a_host, &b_host, rows, k, n);
                for (index, (&bits, expected)) in reference.iter().zip(want).enumerate() {
                    let expected = expected + if has_bias { bias_host[index % n] } else { 0.0 };
                    let actual = f32::from_bits(bits);
                    assert!(
                        actual.is_finite()
                            && (actual - expected).abs() <= 2e-3 * expected.abs().max(1.0),
                        "wide numeric drift at {index}: {actual} vs {expected}, bias={has_bias}"
                    );
                }
            }

            for m in sizes {
                // The one-float C offset also forces scalar output stores;
                // whole-row A subviews retain the required vector alignment.
                for (row_offset, output_offset) in [(0, 4), (17, 1)] {
                    let initial = vec![-913.25; output_offset + m * n + 8];
                    let mut output = GpuBuffer::from_cpu(&ctx.stream, &initial).expect("guarded C");
                    let view_operands = FixedFwdOperands {
                        c: f32_pointer(output.cached_ptr() + (output_offset * 4) as u64),
                        x: f32_pointer(a.cached_ptr() + (row_offset * k * 4) as u64),
                        ..operands
                    };
                    let shape = FixedShape { m, k, n };
                    let mut expected: Vec<_> = initial.iter().map(|x| x.to_bits()).collect();
                    expected[output_offset..output_offset + m * n]
                        .copy_from_slice(&reference[row_offset * n..(row_offset + m) * n]);
                    let run = || fixed_forward_with_tile(&ctx, view_operands, shape, tile);
                    for repeat in 0..2 {
                        output
                            .upload(&ctx.stream, &initial)
                            .expect("poison eager C");
                        run().expect("forced wide view launch");
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "wide prefix/view/guard drift M={m}, row={row_offset}, bias={has_bias}, exceptional={exceptional}, repeat={repeat}"
                        );
                    }
                    // All captured buffers and the context outlive every replay.
                    let graph = unsafe { capture_into_graph(&ctx.stream, run) }
                        .expect("wide graph capture");
                    assert_wide_graph_symbol(&graph);
                    for replay in 0..2 {
                        output
                            .upload(&ctx.stream, &initial)
                            .expect("poison graph C");
                        graph.launch().expect("wide graph replay");
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "wide graph drift M={m}, row={row_offset}, bias={has_bias}, exceptional={exceptional}, replay={replay}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            output_bits(&ctx, &a),
            a_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            "wide modified A"
        );
    }
    assert_eq!(
        output_bits(&ctx, &b),
        b_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        "wide modified B"
    );
    assert_eq!(
        output_bits(&ctx, &bias),
        bias_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        "wide modified bias"
    );
}

#[test]
#[ignore = "requires a CUDA device with the Triad TF32 wide symbol bound"]
fn fixed_tf32_forced_wide_rejects_unsafe_loads_and_handles_zero_reduction() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let tile = FixedTile::Tf32M128N128S3;
    let (m, k, n) = (17, 36, 132);
    let a = GpuBuffer::zeros(&ctx.stream, m * k + 4).expect("A");
    let b = GpuBuffer::zeros(&ctx.stream, k * n + 4).expect("B");
    let mut output = GpuBuffer::zeros(&ctx.stream, m * n).expect("C");
    let operands = FixedFwdOperands {
        c: f32_pointer(output.cached_ptr()),
        x: f32_pointer(a.cached_ptr()),
        w: f32_pointer(b.cached_ptr()),
        bias_ptr: None,
    };
    let shape = FixedShape { m, k, n };
    for bad in [
        FixedFwdOperands {
            x: f32_pointer(a.cached_ptr() + 4),
            ..operands
        },
        FixedFwdOperands {
            w: f32_pointer(b.cached_ptr() + 4),
            ..operands
        },
        FixedFwdOperands {
            x: f32_pointer(0),
            ..operands
        },
        FixedFwdOperands {
            w: f32_pointer(0),
            ..operands
        },
        FixedFwdOperands {
            c: f32_pointer(0),
            ..operands
        },
        FixedFwdOperands {
            c: f32_pointer(output.cached_ptr() + 1),
            ..operands
        },
        FixedFwdOperands {
            bias_ptr: Some(1),
            ..operands
        },
    ] {
        assert!(
            fixed_forward_with_tile(&ctx, bad, shape, tile).is_err(),
            "unsafe pointer admitted"
        );
    }
    for bad in [
        FixedShape { k: k - 1, ..shape },
        FixedShape { n: n - 1, ..shape },
        FixedShape {
            n: i32::MAX as usize - 3,
            ..shape
        },
        FixedShape {
            m: i32::MAX as usize,
            n: 1 << 20,
            ..shape
        },
        FixedShape {
            m: 65_536,
            n: 1 << 29,
            ..shape
        },
    ] {
        assert!(
            fixed_forward_with_tile(&ctx, operands, bad, tile).is_err(),
            "unsafe shape admitted: {bad:?}"
        );
    }
    let bias_host: Vec<_> = (0..n).map(|i| i as f32 * 0.125 - 1.0).collect();
    let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("bias");
    for has_bias in [false, true] {
        output
            .upload(&ctx.stream, &vec![f32::NAN; m * n])
            .expect("poison zero-reduction C");
        fixed_forward_with_tile(
            &ctx,
            FixedFwdOperands {
                x: f32_pointer(0),
                w: f32_pointer(0),
                bias_ptr: has_bias.then_some(bias.cached_ptr()),
                ..operands
            },
            FixedShape { k: 0, ..shape },
            tile,
        )
        .expect("wide K=0 with null inputs");
        let expected: Vec<_> = (0..m * n)
            .map(|i| if has_bias { bias_host[i % n] } else { 0.0 }.to_bits())
            .collect();
        assert_eq!(
            output_bits(&ctx, &output),
            expected,
            "wide zero reduction bias={has_bias}"
        );
    }
}
