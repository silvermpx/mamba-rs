//! Exact B/C norm comparison against the frozen shared-memory reduction.
//! Checks activation storage, f32 RMS storage, guards, and source immutability
//! for all activation dtypes and both short-row and large-state launches.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, bcnorm_fwd_bc_cfg};
use std::sync::Arc;

const GUARD: usize = 1024;
const POISON: u8 = 0xa5;

struct Guarded {
    buffer: GpuByteBuffer,
    payload_bytes: usize,
}

impl Guarded {
    fn upload(stream: &Arc<CudaStream>, payload: &[u8]) -> Self {
        let mut host = vec![POISON; payload.len() + 2 * GUARD];
        host[GUARD..GUARD + payload.len()].copy_from_slice(payload);
        let mut buffer = GpuByteBuffer::zeros(stream, host.len()).unwrap();
        buffer.upload_bytes(stream, &host).unwrap();
        stream.synchronize().unwrap();
        Self {
            buffer,
            payload_bytes: payload.len(),
        }
    }

    fn ptr(&self) -> u64 {
        self.buffer.cached_ptr() + GUARD as u64
    }

    fn payload(&self, stream: &Arc<CudaStream>, label: &str) -> Vec<u8> {
        let mut host = vec![0; self.buffer.len_bytes()];
        stream.memcpy_dtoh(self.buffer.inner(), &mut host).unwrap();
        stream.synchronize().unwrap();
        assert!(
            host[..GUARD].iter().all(|&byte| byte == POISON),
            "{label}: prefix guard changed"
        );
        assert!(
            host[GUARD + self.payload_bytes..]
                .iter()
                .all(|&byte| byte == POISON),
            "{label}: suffix guard changed"
        );
        host[GUARD..GUARD + self.payload_bytes].to_vec()
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    dtype: WeightDtype,
    n: usize,
    ng: usize,
    ds: usize,
    projected: bool,
    special: bool,
}

struct Inputs {
    source: Guarded,
    b_weight: Guarded,
    c_weight: Guarded,
    b_offset: usize,
    c_offset: usize,
    stride: usize,
}

fn encode(values: &[f32], dtype: WeightDtype) -> Vec<u8> {
    match dtype {
        WeightDtype::F32 => bytemuck::cast_slice(values).to_vec(),
        WeightDtype::Bf16 => values
            .iter()
            .flat_map(|&value| half::bf16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        WeightDtype::F16 => values
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
    }
}

fn value(row: usize, d: usize, which: usize, special: bool) -> f32 {
    if special {
        match (row + 3 * which) % 7 {
            0 => return if d.is_multiple_of(2) { 0.0 } else { -0.0 },
            1 => return f32::from_bits(1 + (d % 31) as u32),
            2 if d == 0 => return f32::MAX,
            3 if d == 0 => return f32::INFINITY,
            4 if d == 0 => return f32::from_bits(0x7fc1_2345),
            5 => return 1e-30,
            6 => {
                return f32::from_bits([0x3f80_0001, 0x3f7f_fffe, 0x3980_0000, 0x3a35_04f3][d % 4]);
            }
            _ => {}
        }
    }
    let mut seed = 0x91e1_0da5_u32
        ^ ((row + 1) as u32).wrapping_mul(0x9e37_79b9)
        ^ ((d + 1) as u32).wrapping_mul(0x85eb_ca6b)
        ^ (which as u32).wrapping_mul(0xc2b2_ae35);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    let exponent = 116 + (seed >> 24) % 24;
    f32::from_bits((seed & 0x807f_ffff) | (exponent << 23))
}

fn enqueue(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    cfg: LaunchConfig,
    case: Case,
    inputs: &Inputs,
    outputs: &[Guarded; 4],
) {
    let elt = case.dtype.size_bytes() as u64;
    let pointers = [
        outputs[0].ptr(),
        outputs[1].ptr(),
        outputs[2].ptr(),
        outputs[3].ptr(),
        inputs.source.ptr() + inputs.b_offset as u64 * elt,
        inputs.source.ptr() + inputs.c_offset as u64 * elt,
        inputs.b_weight.ptr(),
        inputs.c_weight.ptr(),
    ];
    let n = case.n as i32;
    let ng = case.ng as i32;
    let ds = case.ds as i32;
    let eps = if case.special { 0.0_f32 } else { 1e-5_f32 };
    let stride = inputs.stride as i32;
    let mut builder = stream.launch_builder(kernel);
    for pointer in &pointers {
        builder.arg(pointer);
    }
    builder.arg(&n);
    builder.arg(&ng);
    builder.arg(&ds);
    builder.arg(&eps);
    builder.arg(&stride);
    // All source/output allocations are disjoint, live, and stream-ordered.
    unsafe { builder.launch(cfg) }.unwrap();
}

fn allocate_outputs(stream: &Arc<CudaStream>, case: Case) -> [Guarded; 4] {
    let rows = case.n * case.ng;
    let activation_bytes = rows * case.ds * case.dtype.size_bytes();
    [activation_bytes, activation_bytes, rows * 4, rows * 4]
        .map(|bytes| Guarded::upload(stream, &vec![POISON; bytes]))
}

fn launch(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    cfg: LaunchConfig,
    case: Case,
    inputs: &Inputs,
) -> [Vec<u8>; 4] {
    let outputs = allocate_outputs(stream, case);
    enqueue(stream, kernel, cfg, case, inputs, &outputs);
    stream.synchronize().unwrap();
    std::array::from_fn(|index| {
        outputs[index].payload(stream, &format!("{case:?}, output {index}"))
    })
}

fn make_inputs(stream: &Arc<CudaStream>, case: Case) -> (Inputs, [Vec<u8>; 3]) {
    let dense = case.ng * case.ds;
    let (stride, b_offset, c_offset, source_len) = if case.projected {
        let stride = 3 + dense + 5 + dense + 7;
        (stride, 3, 3 + dense + 5, case.n * stride)
    } else {
        (dense, 0, case.n * dense, 2 * case.n * dense)
    };
    let mut source = vec![f32::from_bits(0x7fc5_4321); source_len];
    for n in 0..case.n {
        for g in 0..case.ng {
            for d in 0..case.ds {
                let row = n * case.ng + g;
                source[b_offset + n * stride + g * case.ds + d] = value(row, d, 0, case.special);
                source[c_offset + n * stride + g * case.ds + d] = value(row, d, 1, case.special);
            }
        }
    }
    let b_weight: Vec<f32> = (0..case.ds)
        .map(|d| [-1.25, 0.0, f32::from_bits(0x3f80_0001), 0.375][d % 4])
        .collect();
    let c_weight: Vec<f32> = (0..case.ds)
        .map(|d| [0.875, -0.625, 1.5, -0.0][d % 4])
        .collect();
    let source_bytes = encode(&source, case.dtype);
    let b_weight_bytes = encode(&b_weight, WeightDtype::F32);
    let c_weight_bytes = encode(&c_weight, WeightDtype::F32);
    let inputs = Inputs {
        source: Guarded::upload(stream, &source_bytes),
        b_weight: Guarded::upload(stream, &b_weight_bytes),
        c_weight: Guarded::upload(stream, &c_weight_bytes),
        b_offset,
        c_offset,
        stride,
    };
    (inputs, [source_bytes, b_weight_bytes, c_weight_bytes])
}

fn assert_outputs(case: Case, actual: &[Vec<u8>; 4], expected: &[Vec<u8>; 4]) {
    for (index, label) in ["BN", "CN", "B RMS", "C RMS"].into_iter().enumerate() {
        assert_eq!(actual[index].len(), expected[index].len());
        for (byte, (&got, &want)) in actual[index].iter().zip(&expected[index]).enumerate() {
            assert_eq!(got, want, "{case:?}, {label}, byte {byte}");
        }
    }
}

fn legacy_cfg(case: Case) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((case.n * case.ng) as u32, 2, 1),
        block_dim: (case.ds as u32, 1, 1),
        shared_mem_bytes: (case.ds * 4) as u32,
    }
}

fn check_case(
    stream: &Arc<CudaStream>,
    candidate: &CudaFunction,
    reference: &CudaFunction,
    case: Case,
) {
    let (inputs, [source_bytes, b_weight_bytes, c_weight_bytes]) = make_inputs(stream, case);
    let rows = case.n * case.ng;
    let expected = launch(stream, reference, legacy_cfg(case), case, &inputs);
    let actual = launch(
        stream,
        candidate,
        bcnorm_fwd_bc_cfg(rows, case.ds),
        case,
        &inputs,
    );
    assert_outputs(case, &actual, &expected);
    assert_eq!(inputs.source.payload(stream, "source"), source_bytes);
    assert_eq!(inputs.b_weight.payload(stream, "B weight"), b_weight_bytes);
    assert_eq!(inputs.c_weight.payload(stream, "C weight"), c_weight_bytes);
}

#[test]
fn packed_bcnorm_matches_shared_tree_exactly() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let stream = device.fork_stream().unwrap();
    let reference_source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("fixtures/m3_bcnorm_shared_reference.cu"),
    ]
    .join("\n");
    let mut major = 0;
    let mut minor = 0;
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(status, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    let mut options = vec![
        "--fmad=true".to_owned(),
        "--extra-device-vectorization".to_owned(),
        "-DNDEBUG".to_owned(),
        "-DMAMBA_RS_STATE_CAP=64".to_owned(),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295203121".to_owned());
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &reference_source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(device.nvrtc_target()),
            options,
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .expect("shared B/C norm reference compilation");
    let reference_module = device.context().load_module(ptx).unwrap();
    let references = ["f32", "bf16", "f16"].map(|suffix| {
        reference_module
            .load_function(&format!("bcnorm_legacy_{suffix}"))
            .unwrap()
    });
    for state_cap in [16, 64] {
        let kernels = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            device.nvrtc_target(),
            state_cap,
        )
        .expect("production M3 module compilation");
        for (index, dtype) in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16]
            .into_iter()
            .enumerate()
        {
            let candidate = kernels.bcnorm_fwd_bc_typed.get(dtype);
            let reference = &references[index];
            let states = (1..=32).chain([33, 48, 64, 65, 96]);
            let mut cases = 0;
            for ds in states {
                for projected in [false, true] {
                    for special in [false, true] {
                        let case = Case {
                            dtype,
                            n: 11,
                            ng: 3,
                            ds,
                            projected,
                            special,
                        };
                        check_case(&stream, candidate, reference, case);
                        cases += 1;
                        if [1, 3, 8, 16, 32].contains(&ds) {
                            check_case(
                                &stream,
                                candidate,
                                reference,
                                Case {
                                    n: 129,
                                    ng: 1,
                                    ..case
                                },
                            );
                            cases += 1;
                        }
                        if [1, 2, 3, 8, 16, 32, 33, 64].contains(&ds) {
                            check_case(
                                &stream,
                                candidate,
                                reference,
                                Case {
                                    n: 1,
                                    ng: 1,
                                    ..case
                                },
                            );
                            cases += 1;
                        }
                    }
                }
            }
            assert_eq!(cases, 200);
            eprintln!("EXACT state_cap={state_cap} {dtype:?}: {cases} cases passed");
        }
    }
}
