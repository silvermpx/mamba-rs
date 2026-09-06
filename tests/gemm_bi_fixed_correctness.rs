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
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedTile, fixed_forward, fixed_forward_with_tile,
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

#[test]
#[ignore = "requires exclusive CC8.9 Ada with the Fixed RNA-wide symbol admitted"]
fn fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let shape = FixedShape {
        m: 4621,
        k: 384,
        n: 1928,
    };
    let a = GpuBuffer::from_cpu(&ctx.stream, &synth(shape.m * shape.k, 71)).unwrap();
    let b = GpuBuffer::from_cpu(&ctx.stream, &synth(shape.k * shape.n, 73)).unwrap();
    let c = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).unwrap();
    let launch = || {
        let tile = fixed_forward(
            &ctx,
            f32_pointer(c.cached_ptr()),
            f32_pointer(a.cached_ptr()),
            f32_pointer(b.cached_ptr()),
            None,
            (shape.m, shape.k, shape.n),
        )?;
        assert_eq!(tile, FixedTile::Tf32RnaM128N128S3);
        Ok::<(), String>(())
    };
    launch().expect("hot A actual AUTO");
    let graph = unsafe { capture_into_graph(&ctx.stream, launch) }.expect("AUTO capture");
    assert_wide_graph(
        &graph,
        b"gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
        shape,
    );
}

fn assert_wide_graph(graph: &cudarc::driver::CudaGraph, symbol: &[u8], shape: FixedShape) {
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
        symbol,
        "forced Fixed wide silently captured another route",
    );
    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut offset = 0;
        let mut size = 0;
        assert_eq!(
            unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
            sys::CUresult::CUDA_SUCCESS,
            "wide Driver parameter {index}",
        );
        assert_eq!((offset, size), expected, "wide 32-byte Driver ABI");
    }
    let mut offset = 0;
    let mut size = 0;
    assert_eq!(
        unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) },
        sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        "wide launch must not have a sixth argument",
    );
    assert!(!params.kernelParams.is_null());
    let bundle_pointer = unsafe { *params.kernelParams.add(4) };
    assert!(!bundle_pointer.is_null());
    let bundle = unsafe { bundle_pointer.cast::<[u32; 8]>().read_unaligned() };
    assert_eq!(
        bundle,
        [
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            shape.m as u32,
            shape.k as u32,
            shape.n as u32,
            shape.k as u32,
            shape.n as u32,
            shape.n as u32,
        ],
        "wide graph captured the wrong 32-byte parameter bundle",
    );
    let grid = (shape.m as u32)
        .div_ceil(128)
        .checked_mul((shape.n as u32).div_ceil(128))
        .unwrap();
    assert_eq!(
        (params.gridDimX, params.gridDimY, params.gridDimZ),
        (grid, 1, 1)
    );
    assert_eq!(
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        (256, 1, 1)
    );
    assert_eq!(params.sharedMemBytes, 98_304);
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
                    assert_wide_graph(
                        &graph,
                        b"gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3",
                        shape,
                    );
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

#[test]
#[ignore = "requires exclusive CC8.9 Ada with the Fixed RNA-wide symbol admitted"]
fn fixed_tf32_rna_wide_matches_all_fixed_rungs_prefix_views_and_graph_bits() {
    check_rna_wide_prefix_views_and_graph_bits(false);
}

#[test]
#[ignore = "requires exclusive CC8.9 Ada with the Fixed RNA-wide symbol admitted"]
fn fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits() {
    check_rna_wide_prefix_views_and_graph_bits(true);
}

fn check_rna_wide_prefix_views_and_graph_bits(actual_auto: bool) {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    let tile = FixedTile::Tf32RnaM128N128S3;
    let mut shape_cases = vec![
        (
            "tail",
            FixedShape {
                m: 6018,
                k: 36,
                n: 132,
            },
        ),
        (
            "hot_a_boundary",
            FixedShape {
                m: 4622,
                k: 384,
                n: 1928,
            },
        ),
    ];
    if actual_auto {
        shape_cases.extend([
            (
                "hot_b_boundary",
                FixedShape {
                    m: 4622,
                    k: 768,
                    n: 2304,
                },
            ),
            (
                "hot_c_boundary",
                FixedShape {
                    m: 4622,
                    k: 1928,
                    n: 384,
                },
            ),
            (
                "hot_d_boundary",
                FixedShape {
                    m: 2049,
                    k: 768,
                    n: 2304,
                },
            ),
            (
                "hot_e_boundary",
                FixedShape {
                    m: 2049,
                    k: 2304,
                    n: 768,
                },
            ),
        ]);
    }
    let fixed_rungs = [
        FixedTile::Tf32M128S2,
        FixedTile::Tf32M128S3,
        FixedTile::Tf32M64S2,
        FixedTile::Tf32M64S3,
        FixedTile::Tf32M16S4,
    ];
    let special_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_1001,
        0xbf80_1001,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7f80_1000,
        0x7fff_ffff,
        0xffff_ffff,
    ];

    for (case, shape) in shape_cases {
        for exceptional in [false, true] {
            let mut a_host = synth(shape.m * shape.k, 0x89a0_0001);
            let mut b_host = synth(shape.k * shape.n, 0x89b0_0001);
            let mut bias_host = synth(shape.n, 0x89c0_0001);
            if exceptional {
                for (index, bits) in special_bits
                    .into_iter()
                    .cycle()
                    .take(a_host.len())
                    .enumerate()
                {
                    if index % 37 == 0 {
                        a_host[index] = f32::from_bits(bits);
                    }
                }
                for (index, bits) in special_bits
                    .into_iter()
                    .cycle()
                    .take(b_host.len())
                    .enumerate()
                {
                    if index % 29 == 0 {
                        b_host[index] = f32::from_bits(bits);
                    }
                }
                for (index, value) in bias_host.iter_mut().enumerate() {
                    *value = f32::from_bits(special_bits[index % special_bits.len()]);
                }
            } else {
                a_host[0] = f32::from_bits(0x3f80_1001);
                b_host[0] = f32::from_bits(0xbf80_1001);
            }
            let a = GpuBuffer::from_cpu(&ctx.stream, &a_host).expect("RNA A upload");
            let b = GpuBuffer::from_cpu(&ctx.stream, &b_host).expect("RNA B upload");
            let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("RNA bias upload");

            for has_bias in [false, true] {
                let reference =
                    GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("RNA reference output");
                let operands = FixedFwdOperands {
                    c: f32_pointer(reference.cached_ptr()),
                    x: f32_pointer(a.cached_ptr()),
                    w: f32_pointer(b.cached_ptr()),
                    bias_ptr: has_bias.then_some(bias.cached_ptr()),
                };
                fixed_forward_with_tile(&ctx, operands, shape, tile)
                    .expect("RNA-wide full reference launch");
                let reference_bits = output_bits(&ctx, &reference);
                for incumbent in fixed_rungs {
                    let output =
                        GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("incumbent output");
                    fixed_forward_with_tile(
                        &ctx,
                        FixedFwdOperands {
                            c: f32_pointer(output.cached_ptr()),
                            ..operands
                        },
                        shape,
                        incumbent,
                    )
                    .unwrap_or_else(|error| panic!("{incumbent:?} launch: {error}"));
                    assert_eq!(
                        output_bits(&ctx, &output),
                        reference_bits,
                        "RNA-wide differs from {incumbent:?}; case={case} bias={has_bias} exceptional={exceptional}"
                    );
                }

                let view_cases: Vec<_> = if case == "tail" {
                    vec![
                        (1, 0),
                        (15, 0),
                        (16, 0),
                        (17, 0),
                        (31, 0),
                        (32, 0),
                        (33, 0),
                        (63, 0),
                        (64, 0),
                        (65, 0),
                        (127, 0),
                        (128, 0),
                        (129, 0),
                        (255, 0),
                        (256, 0),
                        (257, 0),
                        (257, 17),
                        (4621, 0),
                        (6016, 0),
                        (6017, 0),
                        (6018, 0),
                    ]
                } else if actual_auto {
                    vec![
                        (1, 0),
                        (16, 0),
                        (17, 0),
                        (shape.m - 2, 0),
                        (shape.m - 1, 0),
                        (shape.m, 0),
                        (shape.m - 1, 1),
                    ]
                } else {
                    vec![(4620, 0), (4621, 0), (4622, 0)]
                };
                for (m, row_offset, output_offset) in view_cases.into_iter().flat_map(|(m, row)| {
                    if actual_auto {
                        vec![(m, row, 1), (m, row, 4)]
                    } else {
                        vec![(m, row, 1)]
                    }
                }) {
                    let initial = vec![-819.25; output_offset + m * shape.n + 7];
                    let mut output =
                        GpuBuffer::from_cpu(&ctx.stream, &initial).expect("guarded RNA output");
                    let view_shape = FixedShape {
                        m,
                        k: shape.k,
                        n: shape.n,
                    };
                    let view_operands = FixedFwdOperands {
                        c: f32_pointer(output.cached_ptr() + (output_offset * 4) as u64),
                        x: f32_pointer(a.cached_ptr() + (row_offset * shape.k * 4) as u64),
                        w: operands.w,
                        bias_ptr: operands.bias_ptr,
                    };
                    let mut expected: Vec<_> =
                        initial.iter().map(|value| value.to_bits()).collect();
                    expected[output_offset..output_offset + m * shape.n].copy_from_slice(
                        &reference_bits[row_offset * shape.n..(row_offset + m) * shape.n],
                    );
                    for incumbent in fixed_rungs {
                        output
                            .upload(&ctx.stream, &initial)
                            .expect("poison incumbent prefix/view C");
                        fixed_forward_with_tile(&ctx, view_operands, view_shape, incumbent)
                            .unwrap_or_else(|error| {
                                panic!("{incumbent:?} prefix/view launch: {error}")
                            });
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "RNA prefix differs from {incumbent:?}; case={case} M={m} row={row_offset} bias={has_bias} exceptional={exceptional}"
                        );
                    }
                    let launch = || fixed_forward_with_tile(&ctx, view_operands, view_shape, tile);
                    for repeat in 0..2 {
                        output
                            .upload(&ctx.stream, &initial)
                            .expect("poison RNA eager C");
                        launch().expect("RNA-wide eager view launch");
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "RNA eager prefix/view drift case={case} M={m} row={row_offset} bias={has_bias} exceptional={exceptional} repeat={repeat}"
                        );
                    }
                    let graph = unsafe { capture_into_graph(&ctx.stream, launch) }
                        .expect("RNA-wide graph capture");
                    assert_wide_graph(
                        &graph,
                        b"gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
                        view_shape,
                    );
                    for replay in 0..2 {
                        output
                            .upload(&ctx.stream, &initial)
                            .expect("poison RNA graph C");
                        graph.launch().expect("RNA-wide graph replay");
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "RNA graph prefix/view drift case={case} M={m} row={row_offset} bias={has_bias} exceptional={exceptional} replay={replay}"
                        );
                    }
                    if actual_auto {
                        let admitted = case != "tail" && m == shape.m - 1 && output_offset == 4;
                        let launch_auto = || {
                            let selected = fixed_forward(
                                &ctx,
                                view_operands.c,
                                view_operands.x,
                                view_operands.w,
                                view_operands.bias_ptr,
                                (m, shape.k, shape.n),
                            )?;
                            assert_eq!(
                                selected == tile,
                                admitted,
                                "actual AUTO case={case} M={m} row={row_offset} C-offset={output_offset}"
                            );
                            Ok::<(), String>(())
                        };
                        for repeat in 0..2 {
                            output
                                .upload(&ctx.stream, &initial)
                                .expect("poison AUTO eager");
                            launch_auto().expect("actual AUTO eager");
                            assert_eq!(
                                output_bits(&ctx, &output),
                                expected,
                                "AUTO eager bits case={case} M={m} bias={has_bias} exceptional={exceptional} repeat={repeat}"
                            );
                        }
                        let auto_graph = unsafe { capture_into_graph(&ctx.stream, launch_auto) }
                            .expect("actual AUTO graph");
                        if admitted {
                            assert_wide_graph(
                                &auto_graph,
                                b"gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
                                view_shape,
                            );
                        }
                        for replay in 0..2 {
                            output
                                .upload(&ctx.stream, &initial)
                                .expect("poison AUTO graph");
                            auto_graph.launch().expect("AUTO replay");
                            assert_eq!(
                                output_bits(&ctx, &output),
                                expected,
                                "AUTO graph bits case={case} M={m} bias={has_bias} exceptional={exceptional} replay={replay}"
                            );
                        }
                    }
                }
            }
            if actual_auto && case != "tail" {
                let hot = FixedShape {
                    m: shape.m - 1,
                    ..shape
                };
                for misalign_a in [false, true] {
                    let input_host = if misalign_a { &a_host } else { &b_host };
                    let mut guarded_input = vec![-917.25];
                    guarded_input.extend_from_slice(input_host);
                    guarded_input.push(-917.25);
                    let shifted = GpuBuffer::from_cpu(&ctx.stream, &guarded_input).unwrap();
                    for has_bias in [false, true] {
                        let initial = vec![-819.25; 4 + hot.m * hot.n + 7];
                        let mut output = GpuBuffer::from_cpu(&ctx.stream, &initial).unwrap();
                        let operands = FixedFwdOperands {
                            c: f32_pointer(output.cached_ptr() + 16),
                            x: f32_pointer(if misalign_a {
                                shifted.cached_ptr() + 4
                            } else {
                                a.cached_ptr()
                            }),
                            w: f32_pointer(if misalign_a {
                                b.cached_ptr()
                            } else {
                                shifted.cached_ptr() + 4
                            }),
                            bias_ptr: has_bias.then_some(bias.cached_ptr()),
                        };
                        fixed_forward_with_tile(
                            &ctx,
                            FixedFwdOperands {
                                x: f32_pointer(a.cached_ptr()),
                                w: f32_pointer(b.cached_ptr()),
                                ..operands
                            },
                            hot,
                            tile,
                        )
                        .expect("aligned RNA view reference");
                        let expected = output_bits(&ctx, &output);
                        output.upload(&ctx.stream, &initial).unwrap();
                        fixed_forward_with_tile(&ctx, operands, hot, FixedTile::Tf32M64S2)
                            .expect("misaligned old AUTO control");
                        assert_eq!(
                            output_bits(&ctx, &output),
                            expected,
                            "misaligned control/RNA bits"
                        );
                        let launch_auto = || {
                            let selected = fixed_forward(
                                &ctx,
                                operands.c,
                                operands.x,
                                operands.w,
                                operands.bias_ptr,
                                (hot.m, hot.k, hot.n),
                            )?;
                            assert_eq!(
                                selected,
                                FixedTile::Tf32M64S2,
                                "misaligned A/B must retain old AUTO"
                            );
                            Ok::<(), String>(())
                        };
                        for _ in 0..2 {
                            output.upload(&ctx.stream, &initial).unwrap();
                            launch_auto().expect("misaligned AUTO eager");
                            assert_eq!(
                                output_bits(&ctx, &output),
                                expected,
                                "misaligned AUTO/RNA eager bits"
                            );
                        }
                        let graph =
                            unsafe { capture_into_graph(&ctx.stream, launch_auto) }.unwrap();
                        for _ in 0..2 {
                            output.upload(&ctx.stream, &initial).unwrap();
                            graph.launch().unwrap();
                            assert_eq!(
                                output_bits(&ctx, &output),
                                expected,
                                "misaligned AUTO/RNA graph bits"
                            );
                        }
                    }
                    assert_eq!(
                        output_bits(&ctx, &shifted),
                        guarded_input
                            .into_iter()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                        "misaligned AUTO modified guarded input"
                    );
                }
            }
            assert_eq!(
                output_bits(&ctx, &a),
                a_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                "RNA-wide modified A"
            );
            assert_eq!(
                output_bits(&ctx, &b),
                b_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                "RNA-wide modified B"
            );
            assert_eq!(
                output_bits(&ctx, &bias),
                bias_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                "RNA-wide modified bias"
            );
        }
    }
}

#[test]
#[ignore = "requires exclusive CC8.9 Ada with the Fixed RNA-wide symbol admitted"]
fn fixed_tf32_rna_wide_rejects_unsafe_inputs_and_handles_k0() {
    let device = GpuDevice::new(0).expect("CUDA device");
    assert_eq!(device.compute_capability, (8, 9));
    assert_eq!(device.multiprocessor_count(), 142);
    let ctx = GpuCtx::new(&device).expect("GPU context");
    let tile = FixedTile::Tf32RnaM128N128S3;
    let null_operands = FixedFwdOperands {
        c: f32_pointer(0),
        x: f32_pointer(0),
        w: f32_pointer(0),
        bias_ptr: Some(1),
    };
    for empty in [
        FixedShape {
            m: 0,
            k: usize::MAX,
            n: usize::MAX,
        },
        FixedShape {
            m: usize::MAX,
            k: usize::MAX,
            n: 0,
        },
    ] {
        fixed_forward_with_tile(&ctx, null_operands, empty, tile)
            .unwrap_or_else(|error| panic!("RNA empty output {empty:?}: {error}"));
    }
    let shape = FixedShape {
        m: 17,
        k: 36,
        n: 132,
    };
    let a = GpuBuffer::zeros(&ctx.stream, shape.m * shape.k + 4).expect("RNA A");
    let b = GpuBuffer::zeros(&ctx.stream, shape.k * shape.n + 4).expect("RNA B");
    let output = GpuBuffer::zeros(&ctx.stream, shape.m * shape.n).expect("RNA C");
    let operands = FixedFwdOperands {
        c: f32_pointer(output.cached_ptr()),
        x: f32_pointer(a.cached_ptr()),
        w: f32_pointer(b.cached_ptr()),
        bias_ptr: None,
    };
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
            "unsafe RNA-wide pointer admitted"
        );
    }
    for bad in [
        FixedShape {
            k: shape.k - 1,
            ..shape
        },
        FixedShape {
            n: shape.n - 1,
            ..shape
        },
        FixedShape {
            n: i32::MAX as usize - 3,
            ..shape
        },
        FixedShape {
            m: i32::MAX as usize,
            n: 1 << 20,
            ..shape
        },
    ] {
        assert!(
            fixed_forward_with_tile(&ctx, operands, bad, tile).is_err(),
            "unsafe RNA-wide shape admitted: {bad:?}"
        );
    }

    let bias_host: Vec<_> = (0..shape.n)
        .map(|index| f32::from_bits([0, 0x8000_0000, 0x7f80_0001, 0xffff_ffff][index % 4]))
        .collect();
    let bias = GpuBuffer::from_cpu(&ctx.stream, &bias_host).expect("RNA K0 bias");
    let fixed_rungs = [
        FixedTile::Tf32M128S2,
        FixedTile::Tf32M128S3,
        FixedTile::Tf32M64S2,
        FixedTile::Tf32M64S3,
        FixedTile::Tf32M16S4,
    ];
    let k0_shape = FixedShape { k: 0, ..shape };
    for has_bias in [false, true] {
        let output_offset = 1;
        let output_elements = shape.m * shape.n;
        let mut initial = vec![-819.25; output_offset + output_elements + 7];
        initial[output_offset..output_offset + output_elements].fill(f32::from_bits(0x7fc0_1234));
        let mut k0_output =
            GpuBuffer::from_cpu(&ctx.stream, &initial).expect("guarded RNA K0 output");
        let k0_operands = FixedFwdOperands {
            c: f32_pointer(k0_output.cached_ptr() + (output_offset * 4) as u64),
            x: f32_pointer(0),
            w: f32_pointer(0),
            bias_ptr: has_bias.then_some(bias.cached_ptr()),
        };
        let mut expected: Vec<_> = initial.iter().map(|value| value.to_bits()).collect();
        for index in 0..output_elements {
            expected[output_offset + index] = if has_bias {
                bias_host[index % shape.n]
            } else {
                0.0
            }
            .to_bits();
        }

        for incumbent in fixed_rungs {
            for repeat in 0..2 {
                k0_output
                    .upload(&ctx.stream, &initial)
                    .expect("poison incumbent K0 output");
                fixed_forward_with_tile(&ctx, k0_operands, k0_shape, incumbent)
                    .unwrap_or_else(|error| panic!("{incumbent:?} K0 launch: {error}"));
                assert_eq!(
                    output_bits(&ctx, &k0_output),
                    expected,
                    "RNA K0 differs from {incumbent:?}; bias={has_bias} repeat={repeat}"
                );
            }
        }

        let launch = || fixed_forward_with_tile(&ctx, k0_operands, k0_shape, tile);
        for repeat in 0..2 {
            k0_output
                .upload(&ctx.stream, &initial)
                .expect("poison RNA K0 eager output");
            launch().expect("RNA-wide K0 eager with null inputs");
            assert_eq!(
                output_bits(&ctx, &k0_output),
                expected,
                "RNA K0 eager drift bias={has_bias} repeat={repeat}"
            );
        }
        let graph =
            unsafe { capture_into_graph(&ctx.stream, launch) }.expect("RNA-wide K0 graph capture");
        assert_wide_graph(
            &graph,
            b"gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
            k0_shape,
        );
        for replay in 0..2 {
            k0_output
                .upload(&ctx.stream, &initial)
                .expect("poison RNA K0 graph output");
            graph.launch().expect("RNA-wide K0 graph replay");
            assert_eq!(
                output_bits(&ctx, &k0_output),
                expected,
                "RNA K0 graph drift bias={has_bias} replay={replay}"
            );
        }
    }
    assert_eq!(
        output_bits(&ctx, &bias),
        bias_host.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        "RNA-wide K0 modified bias"
    );
}
