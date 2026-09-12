//! Exact typed sequential burn-in comparison against the released recurrence.
//! Persistent state, typed output, and every activation save are compared as
//! storage bytes, including nonzero initial state and exceptional operands.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use std::sync::Arc;

const GUARD: usize = 1024;
const REDZONE: u8 = 0xc3;
const OUTPUT_NAMES: [&str; 7] = [
    "ssm",
    "k",
    "v",
    "y",
    "h_saved",
    "k_prev_saved",
    "v_prev_saved",
];

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    nh: usize,
    hd: usize,
    ds: usize,
}

impl Shape {
    fn cfg(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.b as u32, self.nh as u32, 1),
            block_dim: (self.hd as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn output_bytes(self, dtype: WeightDtype) -> [usize; 7] {
        let di = self.nh * self.hd;
        [
            self.b * di * self.ds * 4,
            self.b * self.nh * self.ds * 4,
            self.b * di * 4,
            self.b * self.t * di * dtype.size_bytes(),
            self.b * (self.t + 1) * di * self.ds * 4,
            self.b * self.t * self.nh * self.ds * 4,
            self.b * self.t * di * 4,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
enum Corpus {
    Ordinary,
    Cancellation,
    Range,
}

fn encode(values: &[f32], dtype: WeightDtype) -> Vec<u8> {
    match dtype {
        WeightDtype::F32 => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
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

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = ((state & 0xFFFFFF) as f32 / 16777216.0) * 0.5 - 0.25;
            if value == 0.0 { 0.125 } else { value }
        })
        .collect()
}

fn fill_pattern(values: &mut [f32], pattern: &[f32]) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = pattern[index % pattern.len()];
    }
}

fn apply_corpus(inputs: &mut [Vec<f32>; 7], state: &mut [Vec<f32>; 3], corpus: Corpus) {
    let above_one = f32::from_bits(0x3f80_0001);
    let below_one = f32::from_bits(0x3f7f_ffff);
    match corpus {
        Corpus::Ordinary => {}
        Corpus::Cancellation => {
            // Activation values straddle half-storage rounding boundaries;
            // f32 coefficients and initial state retain their low mantissa bits.
            fill_pattern(
                &mut inputs[0],
                &[
                    1.0,
                    -1.0,
                    f32::from_bits(0x3f80_8000),
                    -f32::from_bits(0x3f80_1000),
                ],
            );
            fill_pattern(
                &mut inputs[1],
                &[
                    1.0,
                    f32::from_bits(0x3f81_0000),
                    -1.0,
                    -f32::from_bits(0x3f80_2000),
                ],
            );
            fill_pattern(&mut inputs[2], &[1.0, -1.0, 0.5, -0.5]);
            fill_pattern(&mut inputs[3], &[above_one, below_one, -below_one]);
            fill_pattern(&mut inputs[4], &[above_one, -below_one, below_one]);
            fill_pattern(&mut inputs[5], &[-above_one, below_one, -below_one]);
            fill_pattern(&mut inputs[6], &[-1.0, above_one, -below_one]);
            fill_pattern(
                &mut state[0],
                &[above_one, -above_one, below_one, -below_one],
            );
            fill_pattern(
                &mut state[1],
                &[above_one, -below_one, -above_one, below_one],
            );
            fill_pattern(
                &mut state[2],
                &[above_one, -above_one, below_one, -below_one],
            );
        }
        Corpus::Range => {
            // Include underflow, signed zero, half overflow, and nonfinite
            // values without converting saved f32 state through a half dtype.
            let subnormal = f32::from_bits(1);
            fill_pattern(
                &mut inputs[0],
                &[
                    0.0,
                    -0.0,
                    subnormal,
                    -subnormal,
                    1e-30,
                    -1e-30,
                    f32::from_bits(0x3880_0000),
                    f32::from_bits(0x3380_0000),
                    65504.0,
                    65520.0,
                    -65520.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::from_bits(0x7fc1_2345),
                ],
            );
            fill_pattern(&mut inputs[1], &[1.0, -1.0, 0.5, -0.5]);
            fill_pattern(&mut inputs[2], &[1.0, -1.0, 0.25, -0.25]);
            fill_pattern(&mut inputs[3], &[1.0, 0.5, 1e-20]);
            fill_pattern(&mut inputs[4], &[f32::MIN_POSITIVE, subnormal, -1e-20]);
            fill_pattern(&mut inputs[5], &[1e-20, 1e20, 1.0]);
            fill_pattern(&mut inputs[6], &[above_one, -below_one, 0.0]);
            fill_pattern(
                &mut state[0],
                &[
                    f32::MIN_POSITIVE,
                    -f32::MIN_POSITIVE,
                    subnormal,
                    -subnormal,
                    1e-30,
                ],
            );
            fill_pattern(&mut state[1], &[above_one, -below_one, -above_one]);
            fill_pattern(
                &mut state[2],
                &[f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1e-10],
            );
        }
    }
}

struct Buffer {
    gpu: GpuByteBuffer,
    initial: Vec<u8>,
}

impl Buffer {
    fn new(stream: &Arc<CudaStream>, initial: Vec<u8>) -> Self {
        let mut guarded = vec![REDZONE; initial.len() + 2 * GUARD];
        guarded[GUARD..GUARD + initial.len()].copy_from_slice(&initial);
        let mut gpu = GpuByteBuffer::zeros(stream, guarded.len()).unwrap();
        gpu.upload_bytes(stream, &guarded).unwrap();
        stream.synchronize().unwrap();
        Self { gpu, initial }
    }

    fn ptr(&self) -> u64 {
        self.gpu.cached_ptr() + GUARD as u64
    }

    fn download(&self, stream: &Arc<CudaStream>, label: &str) -> Vec<u8> {
        let mut host = vec![0; self.gpu.len_bytes()];
        stream.memcpy_dtoh(self.gpu.inner(), &mut host).unwrap();
        stream.synchronize().unwrap();
        assert!(
            host[..GUARD].iter().all(|&byte| byte == REDZONE),
            "{label}: prefix redzone changed"
        );
        assert!(
            host[GUARD + self.initial.len()..]
                .iter()
                .all(|&byte| byte == REDZONE),
            "{label}: suffix redzone changed"
        );
        host[GUARD..GUARD + self.initial.len()].to_vec()
    }
}

struct Fixture {
    input: [Buffer; 7],
    state: [Buffer; 3],
}

impl Fixture {
    fn new(stream: &Arc<CudaStream>, shape: Shape, dtype: WeightDtype, corpus: Corpus) -> Self {
        let bt = shape.b * shape.t;
        let di = shape.nh * shape.hd;
        let mut values = [
            det(bt * di, 11),
            det(bt * shape.nh * shape.ds, 12),
            det(bt * shape.nh * shape.ds, 13),
            det(bt * shape.nh, 14)
                .into_iter()
                .map(|v| 0.9 + 0.2 * v.abs())
                .collect(),
            det(bt * shape.nh, 15)
                .into_iter()
                .map(|v| 0.01 + 0.1 * v.abs())
                .collect(),
            det(bt * shape.nh, 16)
                .into_iter()
                .map(|v| 0.01 + 0.1 * v.abs())
                .collect(),
            det(shape.nh, 17),
        ];
        let mut initial = [
            shape.b * di * shape.ds,
            shape.b * shape.nh * shape.ds,
            shape.b * di,
        ]
        .map(|n| det(n, 21 + n as u64));
        apply_corpus(&mut values, &mut initial, corpus);
        let input = std::array::from_fn(|index| {
            let storage = if index < 3 { dtype } else { WeightDtype::F32 };
            Buffer::new(stream, encode(&values[index], storage))
        });
        let state = initial.map(|values| Buffer::new(stream, encode(&values, WeightDtype::F32)));
        Self { input, state }
    }

    fn verify_immutable(&self, stream: &Arc<CudaStream>, label: &str) {
        for (index, buffer) in self.input.iter().chain(&self.state).enumerate() {
            let label = format!("{label}, immutable input {index}");
            assert_bytes(&label, &buffer.download(stream, &label), &buffer.initial);
        }
    }
}

fn outputs(stream: &Arc<CudaStream>, shape: Shape, dtype: WeightDtype, poison: u8) -> [Buffer; 7] {
    shape
        .output_bytes(dtype)
        .map(|bytes| Buffer::new(stream, vec![poison; bytes]))
}

fn enqueue(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    shape: Shape,
    fixture: &Fixture,
    output: &[Buffer; 7],
) {
    // Each arm gets a separate copy of the same immutable initial SSM/K/V.
    for (dst, src) in output[..3].iter().zip(&fixture.state) {
        assert_eq!(dst.initial.len(), src.initial.len());
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst.ptr(),
                src.ptr(),
                src.initial.len(),
                stream.cu_stream(),
            )
        }
        .expect("restore initial state");
    }
    let ptrs: [u64; 14] = std::array::from_fn(|index| {
        if index < 7 {
            output[index].ptr()
        } else {
            fixture.input[index - 7].ptr()
        }
    });
    let dims = [shape.b, shape.t, shape.nh, shape.hd, shape.ds].map(|value| value as i32);
    let mut builder = stream.launch_builder(function);
    for ptr in &ptrs {
        builder.arg(ptr);
    }
    for dim in &dims {
        builder.arg(dim);
    }
    // All pointers refer to disjoint, live allocations on this stream.
    unsafe { builder.launch(shape.cfg()) }.expect("burn-in launch");
}

fn assert_bytes(label: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len(), "{label}: byte length");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: byte {index} got={:02x} expected={:02x}",
            actual[index], expected[index]
        );
    }
}

fn check_case(
    stream: &Arc<CudaStream>,
    functions: [&CudaFunction; 2],
    shape: Shape,
    dtype: WeightDtype,
    corpus: Corpus,
    state_cap: usize,
) {
    let label = format!("cap={state_cap} {dtype:?} {corpus:?} {shape:?}");
    let fixture = Fixture::new(stream, shape, dtype, corpus);
    let arms = [
        outputs(stream, shape, dtype, 0xa5),
        outputs(stream, shape, dtype, 0x5a),
    ];
    for arm in 0..2 {
        enqueue(stream, functions[arm], shape, &fixture, &arms[arm]);
    }
    stream.synchronize().unwrap();
    for (index, name) in OUTPUT_NAMES.into_iter().enumerate() {
        let label = format!("{label}, {name}");
        let expected = arms[0][index].download(stream, &format!("{label}, reference"));
        let actual = arms[1][index].download(stream, &format!("{label}, production"));
        assert_bytes(&label, &actual, &expected);
    }
    fixture.verify_immutable(stream, &label);
}

fn reference_module(device: &GpuDevice, state_cap: usize) -> Arc<CudaModule> {
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("fixtures/m3_burnin_sequential_reference.cu"),
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
        format!("-DMAMBA_RS_STATE_CAP={state_cap}"),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295203121".to_owned());
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(device.nvrtc_target()),
            options,
            include_paths: cuda_include_paths(),
            ..Default::default()
        },
    )
    .expect("released sequential reference compilation");
    device
        .context()
        .load_module(ptx)
        .expect("load sequential reference")
}

#[test]
fn typed_burnin_matches_released_sequential_recurrence() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let stream = device.fork_stream().unwrap();
    for state_cap in [16, 32, 64] {
        let reference = reference_module(&device, state_cap);
        let production = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            device.nvrtc_target(),
            state_cap,
        )
        .expect("production M3 module compilation");
        let nvrtc = production.compiler_identity().nvrtc_version;
        let admits_specialization = device.context().compute_capability().unwrap() == (8, 9)
            && matches!(device.nvrtc_target(), "sm_89" | "compute_89")
            && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2));
        eprintln!("SELECTOR cap={state_cap} NVRTC={nvrtc:?} specialized={admits_specialization}");
        for ds in [3, 8, 16, 17, 32, 48, 64] {
            assert!(std::ptr::eq(
                production.burnin_fwd_typed_for_state(WeightDtype::F32, ds),
                production.burnin_fwd_for_state(ds),
            ));
        }
        for (dtype, suffix) in [(WeightDtype::Bf16, "bf16"), (WeightDtype::F16, "f16")] {
            let old = reference
                .load_function(&format!("m3_burnin_reference_{suffix}"))
                .unwrap();
            let mut cases = 0;
            for ds in [3, 8, 16, 17, 32, 48, 64]
                .into_iter()
                .filter(|&ds| ds <= state_cap)
            {
                let selected = production.burnin_fwd_typed_for_state(dtype, ds);
                let generic = if dtype == WeightDtype::Bf16 {
                    &production.m3_burnin_fwd_typed_bf16
                } else {
                    &production.m3_burnin_fwd_typed_f16
                };
                assert_eq!(
                    !std::ptr::eq(selected, generic),
                    admits_specialization && matches!(ds, 8 | 16 | 32 | 64),
                    "typed selector cap={state_cap} {dtype:?} ds={ds}",
                );
                for t in [1, 7, 33, 256] {
                    for hd in [1, 8, 16, 32] {
                        check_case(
                            &stream,
                            [&old, selected],
                            Shape {
                                b: 2,
                                t,
                                nh: 3,
                                hd,
                                ds,
                            },
                            dtype,
                            Corpus::Ordinary,
                            state_cap,
                        );
                        cases += 1;
                    }
                }
                if [3, 16, 64].contains(&ds) {
                    for corpus in [Corpus::Cancellation, Corpus::Range] {
                        check_case(
                            &stream,
                            [&old, selected],
                            Shape {
                                b: 2,
                                t: 7,
                                nh: 3,
                                hd: 16,
                                ds,
                            },
                            dtype,
                            corpus,
                            state_cap,
                        );
                        cases += 1;
                    }
                }
            }
            check_case(
                &stream,
                [&old, production.burnin_fwd_typed_for_state(dtype, 16)],
                Shape {
                    b: 1,
                    t: 256,
                    nh: 48,
                    hd: 16,
                    ds: 16,
                },
                dtype,
                Corpus::Ordinary,
                state_cap,
            );
            cases += 1;
            assert_eq!(
                cases,
                match state_cap {
                    16 => 53,
                    32 => 85,
                    64 => 119,
                    _ => unreachable!(),
                }
            );
            eprintln!("EXACT cap={state_cap} {dtype:?}: {cases} cases passed");
        }
    }
}
