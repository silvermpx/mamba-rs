#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

#[test]
fn panic_during_capture_restores_the_stream() {
    let device = GpuDevice::new(0).expect("CUDA device");
    let stream = device.fork_stream().expect("capture stream");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        capture_into_graph(&stream, || -> Result<(), String> {
            panic!("capture body panic")
        })
    }));
    assert!(panic.is_err(), "capture body panic was swallowed");
    stream
        .synchronize()
        .expect("stream remained stuck after capture panic");
}
