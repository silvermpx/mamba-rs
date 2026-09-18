//! Independent released fold oracle with guarded eager and graph comparisons.
use cudarc::driver::{CudaFunction, CudaGraph, CudaModule, CudaStream, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::GpuByteBuffer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{FramedSha256, canonical_ptx_image, digest_hex};
use mamba_rs::mamba_ssm::gpu::kernels::{MambaKernels, cuda_include_paths};
use mamba_rs::mamba_ssm::gpu::launch::{
    grid_parallel_scan, grid_parallel_scan_bwd_fold, scan_tape_len,
};
use std::sync::Arc;

const GUARD: usize = 256;
const OUTPUTS: [&str; 6] = [
    "d_delta_raw",
    "d_u",
    "d_B_local",
    "d_C_local",
    "d_D_local",
    "d_a_log_chunks",
];
const INPUTS: [&str; 10] = [
    "tape",
    "delta_saved",
    "u",
    "B",
    "C",
    "a_neg",
    "D",
    "dy",
    "dt_raw",
    "initial_h",
];

fn setting(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}
fn count(name: &str, default: usize) -> usize {
    let value = setting(name, &default.to_string()).parse().expect(name);
    assert!(value > 0, "{name} must be positive");
    value
}

#[derive(Clone, Copy, Debug)]
enum Tape {
    Slim,
    Full,
}
impl Tape {
    fn from_env() -> Self {
        match setting("M1_FOLD_TAPE", "slim").as_str() {
            "slim" => Self::Slim,
            "full" => Self::Full,
            other => panic!("M1_FOLD_TAPE={other:?} (use slim or full)"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Slim => "slim",
            Self::Full => "full",
        }
    }
    fn slim_flag(self) -> i32 {
        match self {
            Self::Slim => 1,
            Self::Full => 0,
        }
    }
    fn floats(self, s: Shape) -> usize {
        match self {
            Self::Slim => scan_tape_len(s.b, s.t, s.di, s.ds),
            Self::Full => s.b * s.di * s.ds * (s.t + 1),
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    digest_hex(&FramedSha256::bytes(bytes))
}
fn exact(label: &str, got: &[u8], expected: &[u8]) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    if let Some(i) = got.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label} byte={i} got={:02x} expected={:02x}",
            got[i], expected[i]
        );
    }
}
fn det(n: usize, mut seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
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

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    di: usize,
    ds: usize,
}
impl Shape {
    fn validate(self) {
        assert!(
            self.b > 0
                && self.t > 0
                && self.di > 0
                && self.di.is_multiple_of(4)
                && self.di <= 65535
        );
        assert!(self.ds > 0 && self.ds <= 256);
    }
    fn dims(self) -> [i32; 4] {
        [self.b, self.t, self.di, self.ds].map(|n| i32::try_from(n).unwrap())
    }
    fn output_bytes(self, dtype: WeightDtype) -> [usize; 6] {
        let typed = self.b * self.t * self.di * dtype.size_bytes();
        let bc = self.b * self.ds * (self.di / 4) * self.t * dtype.size_bytes();
        [
            typed,
            typed,
            bc,
            bc,
            self.b * self.di * 4,
            self.b * self.t.div_ceil(1024) * self.di * self.ds * 4,
        ]
    }
}

struct Buffer {
    gpu: GpuByteBuffer,
    initial: Vec<u8>,
}
impl Buffer {
    fn new(stream: &Arc<CudaStream>, data: &[u8]) -> Self {
        let mut initial = vec![0xd3; data.len() + 2 * GUARD];
        initial[GUARD..GUARD + data.len()].copy_from_slice(data);
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
    fn payload(&self, stream: &Arc<CudaStream>, name: &str) -> Vec<u8> {
        let bytes = self.download(stream);
        let end = bytes.len() - GUARD;
        exact(
            &format!("{name} prefix guard"),
            &bytes[..GUARD],
            &self.initial[..GUARD],
        );
        exact(
            &format!("{name} suffix guard"),
            &bytes[end..],
            &self.initial[end..],
        );
        bytes[GUARD..end].to_vec()
    }
    fn immutable(&self, stream: &Arc<CudaStream>, name: &str) {
        exact(name, &self.download(stream), &self.initial);
    }
    fn poison(&mut self, stream: &Arc<CudaStream>) {
        self.gpu.upload_bytes(stream, &self.initial).unwrap();
        stream.synchronize().unwrap();
    }
}

fn baseline_module(device: &GpuDevice) -> Arc<CudaModule> {
    // A frozen full composition keeps the oracle independent of current codegen.
    assert_eq!(
        device.nvrtc_target(),
        "sm_89",
        "this composition is Ada-only"
    );
    let source = include_str!("legacy_fixed.cu");
    assert_eq!(
        sha(source.as_bytes()),
        "cc63181f2f0376b79d8eb14a6f6c7d4a6d79b3de9e5a60a65e5423027241adf7"
    );
    let mut major = 0;
    let mut minor = 0;
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(status, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    assert!([(12, 8), (13, 0), (13, 2)].contains(&(major, minor)));
    let mut options = vec![
        "--fmad=true".to_owned(),
        "--extra-device-vectorization".to_owned(),
        "-DNDEBUG".to_owned(),
        "-DGEMM_BI_GROUP_M=16".to_owned(),
        "-DMAMBA_RS_STATE_CAP=16".to_owned(),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295072049".to_owned());
    }
    let include_paths = cuda_include_paths();
    assert!(!include_paths.is_empty(), "CUDA includes missing");
    eprintln!(
        "MODULE arm=legacy source_sha256={} NVRTC={major}.{minor} target={} flags={options:?} includes={include_paths:?}",
        sha(source.as_bytes()),
        device.nvrtc_target()
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(device.nvrtc_target()),
            options,
            include_paths,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("NVRTC legacy: {e:?}"));
    let canonical = canonical_ptx_image(ptx.as_bytes().expect("PTX bytes")).unwrap();
    eprintln!(
        "ARTIFACT arm=legacy canonical_ptx_sha256={}",
        sha(canonical.as_bytes())
    );
    device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(canonical))
        .unwrap()
}

// Forward is deliberately the unmodified baseline, executed once from nonzero h.
// Its post-softplus typed save and selected tape become immutable bwd inputs.
fn fixture(
    stream: &Arc<CudaStream>,
    forward: &CudaFunction,
    s: Shape,
    dtype: WeightDtype,
    tape_mode: Tape,
) -> [Vec<u8>; 10] {
    let n = s.b * s.t * s.di;
    let state = encode(&det(s.b * s.di * s.ds, 18, 0.2), WeightDtype::F32);
    let mut data = [
        vec![0; tape_mode.floats(s) * 4],
        vec![0; n * dtype.size_bytes()],
        encode(&det(n, 12, 0.5), dtype),
        encode(&det(s.b * s.t * s.ds, 13, 0.3), dtype),
        encode(&det(s.b * s.t * s.ds, 14, 0.3), dtype),
        encode(
            &det(s.di * s.ds, 15, 0.2)
                .iter()
                .map(|x| -x.abs())
                .collect::<Vec<_>>(),
            WeightDtype::F32,
        ),
        encode(&det(s.di, 16, 0.1), WeightDtype::F32),
        encode(&det(n, 17, 0.1), dtype),
        encode(&det(n, 11, 0.05), dtype),
        state,
    ];
    let input: [Buffer; 10] = std::array::from_fn(|i| Buffer::new(stream, &data[i]));
    let h = Buffer::new(stream, &data[9]);
    let y = Buffer::new(stream, &vec![0xa5; n * dtype.size_bytes()]);
    let pointers = [
        h.ptr(),
        y.ptr(),
        input[0].ptr(),
        input[8].ptr(),
        input[1].ptr(),
        input[2].ptr(),
        input[3].ptr(),
        input[4].ptr(),
        input[5].ptr(),
        input[6].ptr(),
    ];
    let dims = s.dims();
    let tape = input[0].ptr();
    let slim = tape_mode.slim_flag();
    let mut builder = stream.launch_builder(forward);
    for ptr in &pointers {
        builder.arg(ptr);
    }
    for dim in &dims {
        builder.arg(dim);
    }
    builder.arg(&tape).arg(&slim);
    unsafe { builder.launch(grid_parallel_scan(s.b, s.di, s.ds)) }.unwrap();
    stream.synchronize().unwrap();
    data[0] = input[0].payload(stream, "forward tape");
    data[1] = input[1].payload(stream, "forward delta_saved");
    for i in 2..10 {
        input[i].immutable(stream, INPUTS[i]);
    }
    let h_final = h.payload(stream, "forward h");
    let y_final = y.payload(stream, "forward y");
    eprintln!(
        "FIXTURE shape={s:?} dtype={dtype:?} tape={} slim={slim} tape_floats={} initial_h_sha={} final_h_sha={} y_sha={} tape_sha={} delta_saved_sha={} nonzero_h=1",
        tape_mode.label(),
        tape_mode.floats(s),
        sha(&data[9]),
        sha(&h_final),
        sha(&y_final),
        sha(&data[0]),
        sha(&data[1])
    );
    data
}

struct Arm {
    inputs: [Buffer; 10],
    outputs: [Buffer; 6],
    dtype: WeightDtype,
}
impl Arm {
    fn new(
        stream: &Arc<CudaStream>,
        data: &[Vec<u8>; 10],
        s: Shape,
        dtype: WeightDtype,
        poison: u8,
    ) -> Self {
        Self {
            inputs: std::array::from_fn(|i| Buffer::new(stream, &data[i])),
            outputs: std::array::from_fn(|i| {
                Buffer::new(stream, &vec![poison; s.output_bytes(dtype)[i]])
            }),
            dtype,
        }
    }
    fn snapshot(&self, stream: &Arc<CudaStream>) -> [Vec<u8>; 6] {
        std::array::from_fn(|i| self.outputs[i].payload(stream, OUTPUTS[i]))
    }
    fn immutable(&self, stream: &Arc<CudaStream>) {
        for (i, name) in INPUTS.iter().enumerate() {
            self.inputs[i].immutable(stream, name);
        }
    }
    fn poison(&mut self, stream: &Arc<CudaStream>) {
        for output in &mut self.outputs {
            output.poison(stream);
        }
    }
}

fn launch(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    s: Shape,
    arm: &Arm,
    tape_mode: Tape,
) -> Result<(), String> {
    let i = &arm.inputs;
    let o = &arm.outputs;
    // 15 pointers, four dimensions, run_tape pointer, slim flag: exact fold ABI.
    let pointers = [
        i[0].ptr(),
        i[1].ptr(),
        i[2].ptr(),
        i[3].ptr(),
        i[4].ptr(),
        i[5].ptr(),
        i[6].ptr(),
        i[7].ptr(),
        o[0].ptr(),
        i[8].ptr(),
        o[1].ptr(),
        o[2].ptr(),
        o[3].ptr(),
        o[4].ptr(),
        o[5].ptr(),
    ];
    let dims = s.dims();
    let tape = i[0].ptr();
    let slim = tape_mode.slim_flag();
    let mut builder = stream.launch_builder(function);
    for ptr in &pointers {
        builder.arg(ptr);
    }
    for dim in &dims {
        builder.arg(dim);
    }
    builder.arg(&tape).arg(&slim);
    unsafe {
        builder.launch(grid_parallel_scan_bwd_fold(
            s.b,
            s.di,
            s.ds,
            arm.dtype.size_bytes(),
        ))
    }
    .map(|_| ())
    .map_err(|e| format!("fold launch: {e:?}"))
}

fn one_node_graph(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    s: Shape,
    arm: &Arm,
    tape_mode: Tape,
) -> CudaGraph {
    unsafe {
        capture_into_graph(stream, || launch(stream, function, s, arm, tape_mode))
            .expect("capture one-node fold graph")
    }
}

fn compare(label: &str, got: &[Vec<u8>; 6], expected: &[Vec<u8>; 6]) {
    for i in 0..6 {
        exact(&format!("{label} {}", OUTPUTS[i]), &got[i], &expected[i]);
    }
}
fn event_us(stream: &Arc<CudaStream>, graph: &CudaGraph, nodes: usize, replays: usize) -> f64 {
    let start = stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..replays {
        graph.launch().unwrap();
    }
    let end = stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    stream.synchronize().unwrap();
    f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / (nodes * replays) as f64
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[m - 1] + values[m]) / 2.0
    } else {
        values[m]
    }
}
fn pair(
    stream: &Arc<CudaStream>,
    functions: [&CudaFunction; 2],
    forward: &CudaFunction,
    s: Shape,
    dtype: WeightDtype,
    timed: bool,
    tape_mode: Tape,
) {
    s.validate();
    let cfg = grid_parallel_scan_bwd_fold(s.b, s.di, s.ds, dtype.size_bytes());
    let slim = tape_mode.slim_flag();
    eprintln!(
        "CONFIG symbol=ssm_parallel_scan_bwd_fold_{} shape={s:?} grid={:?} block={:?} smem={} tape={} slim={slim} tape_floats={} G=4 NITEMS=8",
        dtype.as_str(),
        cfg.grid_dim,
        cfg.block_dim,
        cfg.shared_mem_bytes,
        tape_mode.label(),
        tape_mode.floats(s)
    );
    let data = fixture(stream, forward, s, dtype, tape_mode);
    let mut arms = [
        Arm::new(stream, &data, s, dtype, 0xa5),
        Arm::new(stream, &data, s, dtype, 0x5a),
    ];
    drop(data);
    for (i, name) in INPUTS.iter().enumerate() {
        assert_ne!(arms[0].inputs[i].ptr(), arms[1].inputs[i].ptr());
        exact(name, &arms[0].inputs[i].initial, &arms[1].inputs[i].initial);
    }
    for i in 0..6 {
        assert_ne!(arms[0].outputs[i].ptr(), arms[1].outputs[i].ptr());
    }
    launch(stream, functions[0], s, &arms[0], tape_mode).unwrap();
    if setting("M1_FOLD_NEGATIVE_SKIP", "0") != "1" {
        launch(stream, functions[1], s, &arms[1], tape_mode).unwrap();
    }
    stream.synchronize().unwrap();
    let expected = arms[0].snapshot(stream);
    compare("pre-graph", &arms[1].snapshot(stream), &expected);
    for arm in &arms {
        arm.immutable(stream);
    }
    eprintln!(
        "EXACT phase=pre-graph dtype={dtype:?} shape={s:?} tape={} {}",
        tape_mode.label(),
        (0..6)
            .map(|i| format!("{}={}", OUTPUTS[i], sha(&expected[i])))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let negative_graph = setting("M1_FOLD_NEGATIVE_GRAPH", "0") == "1";
    let exact_graphs = [
        one_node_graph(stream, functions[0], s, &arms[0], tape_mode),
        if negative_graph {
            // Capture a valid graph but deliberately leave candidate outputs
            // untouched. The post-reset exact check must catch the omission.
            one_node_graph(stream, functions[0], s, &arms[0], tape_mode)
        } else {
            one_node_graph(stream, functions[1], s, &arms[1], tape_mode)
        },
    ];
    for arm in &mut arms {
        arm.poison(stream);
    }
    exact_graphs[0].launch().unwrap();
    exact_graphs[1].launch().unwrap();
    stream.synchronize().unwrap();
    for arm in &arms {
        compare("one-node graph", &arm.snapshot(stream), &expected);
        arm.immutable(stream);
    }
    eprintln!(
        "EXACT phase=one-node-graph dtype={dtype:?} shape={s:?} tape={} fresh_poison=1 negative_graph={}",
        tape_mode.label(),
        i32::from(negative_graph)
    );
    if !timed {
        return;
    }
    let nodes = count("M1_FOLD_NODES", 2);
    let replays = count("M1_FOLD_REPLAYS", 3);
    let rounds = count("M1_FOLD_ROUNDS", 7);
    let graphs: [CudaGraph; 2] = std::array::from_fn(|i| unsafe {
        capture_into_graph(stream, || {
            for _ in 0..nodes {
                launch(stream, functions[i], s, &arms[i], tape_mode)?;
            }
            Ok(())
        })
        .expect("capture fold-only graph")
    });
    // Discard eager results so a graph that writes nothing cannot inherit a pass.
    for arm in &mut arms {
        arm.poison(stream);
    }
    for _ in 0..3 {
        graphs[0].launch().unwrap();
        graphs[1].launch().unwrap();
    }
    stream.synchronize().unwrap();
    for arm in &arms {
        compare("graph warmup", &arm.snapshot(stream), &expected);
        arm.immutable(stream);
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut ratios = Vec::new();
    eprintln!(
        "TIMING dtype={dtype:?} shape={s:?} tape={} nodes={nodes} replays={replays} rounds={rounds} fold_only=1 reset_copies=0",
        tape_mode.label()
    );
    for r in 0..rounds {
        let a1 = event_us(stream, &graphs[0], nodes, replays);
        let b1 = event_us(stream, &graphs[1], nodes, replays);
        let b2 = event_us(stream, &graphs[1], nodes, replays);
        let a2 = event_us(stream, &graphs[0], nodes, replays);
        let ratio = (a1 + a2) / (b1 + b2);
        a.extend([a1, a2]);
        b.extend([b1, b2]);
        ratios.push(ratio);
        eprintln!(
            "ABBA round={r} A1_us={a1:.4} B1_us={b1:.4} B2_us={b2:.4} A2_us={a2:.4} ratio={ratio:.6}"
        );
    }
    for arm in &arms {
        compare("post-timing", &arm.snapshot(stream), &expected);
        arm.immutable(stream);
    }
    eprintln!(
        "RESULT dtype={dtype:?} shape={s:?} tape={} baseline_us={:.4} candidate_us={:.4} paired_ratio={:.6} exact_post=1",
        tape_mode.label(),
        median(&mut a),
        median(&mut b),
        median(&mut ratios)
    );
}

pub(crate) fn qualify(device: &GpuDevice, kernels: &MambaKernels) {
    let mode = setting("M1_FOLD_MODE", "check");
    assert!(["compare", "check"].contains(&mode.as_str()));
    let tapes = match setting("M1_FOLD_TAPE", "both").as_str() {
        "both" => vec![Tape::Full, Tape::Slim],
        _ => vec![Tape::from_env()],
    };
    let dtypes = match setting("M1_FOLD_DTYPE", "all").as_str() {
        "bf16" => vec![WeightDtype::Bf16],
        "f16" => vec![WeightDtype::F16],
        "f32" => vec![WeightDtype::F32],
        "half" => vec![WeightDtype::Bf16, WeightDtype::F16],
        "all" => vec![WeightDtype::Bf16, WeightDtype::F16, WeightDtype::F32],
        other => panic!("M1_FOLD_DTYPE={other}"),
    };
    assert_eq!(kernels.state_cap, 16);
    let stream = device.fork_stream().unwrap();
    eprintln!("PRODUCTION compiler={:?}", kernels.compiler_identity());
    let baseline = baseline_module(device);
    let shapes = setting(
        "M1_FOLD_SHAPES",
        "1,1,4,8;2,7,8,16;1,33,12,16;8,33,768,16;1,256,4,8;1,256,4,16;8,256,768,16;8,257,768,16;1,1025,4,16;4,1024,768,16;8,1025,384,16;8,1300,768,16;8,2049,768,16;8,4096,768,16",
    );
    let shapes: Vec<Shape> = shapes
        .split(';')
        .map(|text| {
            let v: Vec<usize> = text
                .split(',')
                .map(|v| v.parse().expect("B,T,DI,DS"))
                .collect();
            assert_eq!(v.len(), 4, "shape B,T,DI,DS");
            let shape = Shape {
                b: v[0],
                t: v[1],
                di: v[2],
                ds: v[3],
            };
            shape.validate();
            assert!(shape.ds <= kernels.state_cap);
            shape
        })
        .collect();
    for dtype in dtypes {
        let symbol = format!("ssm_parallel_scan_bwd_fold_{}", dtype.as_str());
        let forward_symbol = if dtype == WeightDtype::F32 {
            "ssm_parallel_scan_fwd".to_owned()
        } else {
            format!("ssm_parallel_scan_fwd_{}", dtype.as_str())
        };
        let forward = baseline.load_function(&forward_symbol).unwrap();
        let legacy = baseline.load_function(&symbol).unwrap();
        legacy.set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            67584,
        ).unwrap();
        for &shape in &shapes {
            let selected = kernels
                .ssm_parallel_bwd_fold_for_shape(dtype, shape.b, shape.t, shape.di, shape.ds);
            let specialized =
                !std::ptr::eq(selected, kernels.ssm_parallel_bwd_fold_typed.get(dtype));
            eprintln!("ROUTE shape={shape:?} dtype={dtype:?} specialized={specialized}");
            use cudarc::driver::sys::CUfunction_attribute_enum as Attribute;
            let optin = selected
                .get_attribute(Attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
                .unwrap();
            let registers = selected
                .get_attribute(Attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)
                .unwrap();
            let local = selected
                .get_attribute(Attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
                .unwrap();
            let static_shared = selected
                .get_attribute(Attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
                .unwrap();
            assert_eq!(optin, 67584);
            eprintln!(
                "RESOURCE dtype={dtype:?} shape={shape:?} registers={registers} local_bytes={local} static_shared={static_shared} max_dynamic_shared={optin}"
            );
            for &tape in &tapes {
                pair(
                    &stream,
                    [&legacy, selected],
                    &forward,
                    shape,
                    dtype,
                    mode == "compare",
                    tape,
                );
            }
        }
    }
}
