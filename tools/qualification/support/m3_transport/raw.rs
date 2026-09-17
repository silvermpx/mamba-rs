//! Compare selected production functions with a frozen, pre-transport module.

#[path = "baseline.rs"]
mod baseline;
#[path = "cases.rs"]
mod cases;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba3_siso::gpu::Mamba3Kernels;
use std::sync::Arc;

const GUARD: usize = 256;

fn exact(label: &str, got: &[u8], expected: &[u8]) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    if let Some(i) = got.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: byte={i} got={:02x} expected={:02x}",
            got[i], expected[i]
        );
    }
}

fn values(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed & 0xffff) as f32 / 65536.0 - 0.5
        })
        .collect()
}

fn encode(values: &[f32], dtype: WeightDtype) -> Vec<u8> {
    match dtype {
        WeightDtype::F32 | WeightDtype::Tf32 => {
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        WeightDtype::Bf16 => values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect(),
        WeightDtype::F16 => values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect(),
    }
}

struct Buffer {
    gpu: GpuByteBuffer,
    initial: Vec<u8>,
}

impl Buffer {
    fn new(stream: &Arc<CudaStream>, payload: &[u8]) -> Self {
        let mut initial = vec![0xd3; payload.len() + 2 * GUARD];
        initial[GUARD..GUARD + payload.len()].copy_from_slice(payload);
        let mut gpu = GpuByteBuffer::zeros(stream, initial.len()).unwrap();
        gpu.upload_bytes(stream, &initial).unwrap();
        stream.synchronize().unwrap();
        Self { gpu, initial }
    }

    fn ptr(&self) -> u64 {
        self.gpu.cached_ptr() + GUARD as u64
    }

    fn download(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        let mut bytes = vec![0; self.initial.len()];
        stream.memcpy_dtoh(self.gpu.inner(), &mut bytes).unwrap();
        stream.synchronize().unwrap();
        bytes
    }

    fn payload(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        let bytes = self.download(stream);
        let end = bytes.len() - GUARD;
        exact("prefix guard", &bytes[..GUARD], &self.initial[..GUARD]);
        exact("suffix guard", &bytes[end..], &self.initial[end..]);
        bytes[GUARD..end].to_vec()
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) {
        self.gpu.upload_bytes(stream, &self.initial).unwrap();
        stream.synchronize().unwrap();
    }
}

struct Case<'a> {
    label: String,
    outputs: &'a [&'a str],
    lengths: Vec<usize>,
    inputs: Vec<Vec<u8>>,
    dims: Vec<i32>,
    functions: [&'a CudaFunction; 2],
    configs: [LaunchConfig; 2],
    // Accumulating reducers must start both arms from the same finite values.
    initial_output: Option<Vec<u8>>,
}

struct Arm {
    outputs: Vec<Buffer>,
    inputs: Vec<Buffer>,
}

impl Arm {
    fn new(stream: &Arc<CudaStream>, case: &Case<'_>, poison: u8) -> Self {
        Self {
            outputs: case
                .lengths
                .iter()
                .map(|&n| {
                    let bytes = case
                        .initial_output
                        .clone()
                        .unwrap_or_else(|| vec![poison; n * 4]);
                    assert_eq!(bytes.len(), n * 4);
                    Buffer::new(stream, &bytes)
                })
                .collect(),
            inputs: case
                .inputs
                .iter()
                .map(|bytes| Buffer::new(stream, bytes))
                .collect(),
        }
    }

    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        case: &Case<'_>,
        index: usize,
    ) -> Result<(), String> {
        let pointers: Vec<_> = self
            .outputs
            .iter()
            .chain(&self.inputs)
            .map(Buffer::ptr)
            .collect();
        let mut builder = stream.launch_builder(case.functions[index]);
        for ptr in &pointers {
            builder.arg(ptr);
        }
        for dim in &case.dims {
            builder.arg(dim);
        }
        unsafe { builder.launch(case.configs[index]) }
            .map(|_| ())
            .map_err(|error| format!("{}: {error:?}", case.label))
    }

    fn snapshot(&self, stream: &Arc<CudaStream>) -> Vec<Vec<u8>> {
        self.outputs
            .iter()
            .map(|buffer| buffer.payload(stream))
            .collect()
    }

    fn check(&self, stream: &Arc<CudaStream>, case: &Case<'_>, expected: &[Vec<u8>], stage: &str) {
        for (i, bytes) in self.snapshot(stream).iter().enumerate() {
            exact(
                &format!("{stage} {} {}", case.label, case.outputs[i]),
                bytes,
                &expected[i],
            );
        }
        for (i, input) in self.inputs.iter().enumerate() {
            exact(
                &format!("{stage} {} input={i}", case.label),
                &input.download(stream),
                &input.initial,
            );
        }
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) {
        for output in &mut self.outputs {
            output.reset(stream);
        }
    }
}

fn check_case(stream: &Arc<CudaStream>, case: Case<'_>) {
    assert_eq!(case.outputs.len(), case.lengths.len());
    let negative = std::env::var("MAMBA_RS_TRANSPORT_NEGATIVE").unwrap_or_default();
    assert!(["", "eager", "graph"].contains(&negative.as_str()));
    let mut arms = [Arm::new(stream, &case, 0xa5), Arm::new(stream, &case, 0x5a)];
    for (a, b) in arms[0].inputs.iter().zip(&arms[1].inputs) {
        assert_ne!(a.ptr(), b.ptr());
        exact("independent input fixture", &a.initial, &b.initial);
    }
    arms[0].enqueue(stream, &case, 0).unwrap();
    if negative != "eager" {
        arms[1].enqueue(stream, &case, 1).unwrap();
    }
    stream.synchronize().unwrap();
    let expected = arms[0].snapshot(stream);
    for arm in &arms {
        arm.check(stream, &case, &expected, "eager");
    }
    // A captured no-op must fail even after the eager comparison succeeded.
    let graphs: [_; 2] = std::array::from_fn(|i| unsafe {
        capture_into_graph(stream, || {
            if i == 1 && negative == "graph" {
                // Keep a valid graph but leave the candidate outputs untouched.
                return arms[0].enqueue(stream, &case, 0);
            }
            arms[i].enqueue(stream, &case, i)
        })
        .unwrap()
    });
    for _ in 0..2 {
        for arm in &mut arms {
            arm.reset(stream);
        }
        for graph in &graphs {
            graph.launch().unwrap();
        }
        stream.synchronize().unwrap();
        for arm in &arms {
            arm.check(stream, &case, &expected, "graph");
        }
    }
    eprintln!(
        "EXACT {} outputs={} eager=1 graph_fresh_replays=2 guards=1 immutable=1",
        case.label,
        case.outputs.len()
    );
}

pub(super) fn qualify(device: &GpuDevice, kernels: &Mamba3Kernels) {
    let baseline = baseline::compile(device, kernels.state_cap);
    let stream = device.context().new_stream().unwrap();
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        cases::dqkv(&stream, kernels, &baseline, dtype);
        cases::dqktheta(&stream, kernels, &baseline, dtype);
    }
    cases::axis0(&stream, kernels, &baseline);
    cases::angle(&stream, kernels, &baseline);
}
