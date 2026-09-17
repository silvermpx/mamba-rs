const ACTIVE_ELEMENTS: usize = 8095;
const GUARD_ELEMENTS: usize = 16;
const WINDOWS_PER_ORDER: usize = 101;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArenaCell {
    k: usize,
    n: usize,
    offset: usize,
    len: usize,
}

impl ArenaCell {
    const fn new(k: usize, n: usize, offset: usize, len: usize) -> Self {
        Self { k, n, offset, len }
    }

    const fn end(self) -> usize {
        self.offset + self.len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArenaLayout {
    guard_len: usize,
    active_len: usize,
    cells: [ArenaCell; 3],
}

impl ArenaLayout {
    const fn storage_len(self) -> usize {
        self.guard_len + self.active_len + self.guard_len
    }
}

const fn training_arena_layout() -> ArenaLayout {
    ArenaLayout {
        guard_len: GUARD_ELEMENTS,
        active_len: ACTIVE_ELEMENTS,
        cells: [
            ArenaCell::new(128, 25, 0, 3200),
            ArenaCell::new(64, 64, 3200, 4096),
            ArenaCell::new(47, 17, 7296, 799),
        ],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineArm {
    Production,
    DirectControl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedNode {
    MemcpyD2d {
        elements: usize,
    },
    Kernel {
        symbol: &'static str,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_bytes: u32,
        len: Option<usize>,
    },
}

impl ExpectedNode {
    const fn kernel(symbol: &'static str, grid: [u32; 3], block: [u32; 3]) -> Self {
        Self::Kernel {
            symbol,
            grid,
            block,
            dynamic_shared_bytes: 0,
            len: None,
        }
    }

    const fn adamw(len: usize) -> Self {
        Self::Kernel {
            symbol: "adamw_step_f32_capturable",
            grid: [32, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            len: Some(len),
        }
    }

    const fn symbol(self) -> Option<&'static str> {
        match self {
            Self::MemcpyD2d { .. } => None,
            Self::Kernel { symbol, .. } => Some(symbol),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpectedTopology {
    nodes: Vec<ExpectedNode>,
    edges: Vec<(usize, usize)>,
}

fn linear_edges(nodes: usize) -> Vec<(usize, usize)> {
    (0..nodes.saturating_sub(1))
        .map(|index| (index, index + 1))
        .collect()
}

fn expected_topology(arm: PipelineArm) -> ExpectedTopology {
    let nodes = match arm {
        PipelineArm::Production => vec![
            ExpectedNode::MemcpyD2d {
                elements: ACTIVE_ELEMENTS,
            },
            ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [1, 1, 32], [128, 1, 1]),
            ExpectedNode::kernel("splitm_reduce", [4, 1, 1], [256, 1, 1]),
            ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [2, 1, 32], [128, 1, 1]),
            ExpectedNode::kernel("splitm_reduce", [13, 1, 1], [256, 1, 1]),
            ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [1, 2, 86], [128, 1, 1]),
            ExpectedNode::kernel("splitm_reduce", [16, 1, 1], [256, 1, 1]),
            ExpectedNode::adamw(ACTIVE_ELEMENTS),
        ],
        PipelineArm::DirectControl => vec![
            ExpectedNode::MemcpyD2d {
                elements: ACTIVE_ELEMENTS,
            },
            ExpectedNode::kernel("tn_narrow", [1, 1, 1], [128, 1, 1]),
            ExpectedNode::kernel("tn_narrow", [2, 1, 1], [128, 1, 1]),
            ExpectedNode::kernel("tn_narrow", [2, 1, 1], [128, 1, 1]),
            ExpectedNode::adamw(ACTIVE_ELEMENTS),
        ],
    };
    ExpectedTopology {
        edges: linear_edges(nodes.len()),
        nodes,
    }
}

const fn ordinal_schedule() -> [u64; 10] {
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmOrder {
    ProductionFirst,
    DirectFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairedWindow {
    order: ArmOrder,
    window: usize,
}

fn paired_window_schedule() -> Vec<PairedWindow> {
    [ArmOrder::ProductionFirst, ArmOrder::DirectFirst]
        .into_iter()
        .flat_map(|order| (0..WINDOWS_PER_ORDER).map(move |window| PairedWindow { order, window }))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineMode {
    Eager,
    Graph,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PromotionCell {
    median_ratio: f64,
    p05_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PromotionEvidence {
    eager_production_first: PromotionCell,
    eager_direct_first: PromotionCell,
    graph_production_first: PromotionCell,
    graph_direct_first: PromotionCell,
}

impl PromotionEvidence {
    const fn uniform(median_ratio: f64, p05_ratio: f64) -> Self {
        let cell = PromotionCell {
            median_ratio,
            p05_ratio,
        };
        Self {
            eager_production_first: cell,
            eager_direct_first: cell,
            graph_production_first: cell,
            graph_direct_first: cell,
        }
    }

    fn cell_mut(&mut self, mode: PipelineMode, order: ArmOrder) -> &mut PromotionCell {
        match (mode, order) {
            (PipelineMode::Eager, ArmOrder::ProductionFirst) => &mut self.eager_production_first,
            (PipelineMode::Eager, ArmOrder::DirectFirst) => &mut self.eager_direct_first,
            (PipelineMode::Graph, ArmOrder::ProductionFirst) => &mut self.graph_production_first,
            (PipelineMode::Graph, ArmOrder::DirectFirst) => &mut self.graph_direct_first,
        }
    }

    fn promoted(self) -> bool {
        [
            self.eager_production_first,
            self.eager_direct_first,
            self.graph_production_first,
            self.graph_direct_first,
        ]
        .into_iter()
        .all(|cell| cell.median_ratio >= 1.01 && cell.p05_ratio > 1.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PairedAbsoluteSamples {
    production_device_us: Vec<f64>,
    direct_device_us: Vec<f64>,
    production_host_us: Vec<f64>,
    direct_host_us: Vec<f64>,
}

impl PairedAbsoluteSamples {
    fn ratios(numerator: &[f64], denominator: &[f64]) -> Result<Vec<f64>, String> {
        if numerator.len() != denominator.len()
            || numerator.is_empty()
            || denominator
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || numerator
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("paired samples require equal non-empty positive finite vectors".into());
        }
        Ok(numerator
            .iter()
            .zip(denominator)
            .map(|(n, d)| n / d)
            .collect())
    }

    fn device_ratios(&self) -> Result<Vec<f64>, String> {
        Self::ratios(&self.direct_device_us, &self.production_device_us)
    }

    fn host_ratios(&self) -> Result<Vec<f64>, String> {
        Self::ratios(&self.direct_host_us, &self.production_host_us)
    }
}

fn stable_schedule_canonical() -> String {
    [
        "task8_schedule",
        "modes=eager,graph",
        "orders=production_first,direct_first",
        "calls=1024:47:17:7296,1024:128:25:0,4096:64:64:3200",
        "arena=16:8095:16",
        "storage=128:25:0:3200,64:64:3200:4096,47:17:7296:799",
        "corpus_seed=57344",
        "calibration_seed=corpus_seed_xor_0xca11",
        "logical_steps_per_window=1",
        "bias_update=before_each_logical_step",
        "reset=active_only_d2d",
        "calibration=disposable:3:no_resize",
        "windows_per_order=101",
    ]
    .join("\n")
}

fn first_quiet_streak_end(
    samples: &[(u32, u32)],
    required: usize,
    threshold: u32,
) -> Option<usize> {
    let mut consecutive = 0usize;
    for (index, &(gpu_util, memory_util)) in samples.iter().enumerate() {
        if gpu_util <= threshold && memory_util <= threshold {
            consecutive += 1;
            if consecutive == required {
                return Some(index);
            }
        } else {
            consecutive = 0;
        }
    }
    None
}

#[test]
fn dense_arena_layout_is_exact_and_aligned() {
    let layout = training_arena_layout();
    assert_eq!(layout.active_len, 8095);
    assert_eq!(layout.guard_len, 16);
    assert_eq!(layout.storage_len(), 8127);
    assert_eq!(layout.cells.len(), 3);
    assert_eq!(layout.cells[0], ArenaCell::new(128, 25, 0, 3200));
    assert_eq!(layout.cells[1], ArenaCell::new(64, 64, 3200, 4096));
    assert_eq!(layout.cells[2], ArenaCell::new(47, 17, 7296, 799));
    assert!(
        layout
            .cells
            .windows(2)
            .all(|pair| pair[0].end() == pair[1].offset)
    );
    assert_eq!(layout.cells.last().unwrap().end(), layout.active_len);
    assert!(
        layout
            .cells
            .iter()
            .all(|cell| (layout.guard_len + cell.offset).is_multiple_of(4))
    );
}

#[test]
fn production_and_direct_topologies_are_exact_linear_contracts() {
    let production = expected_topology(PipelineArm::Production);
    assert_eq!(production.nodes.len(), 8);
    assert_eq!(production.edges, linear_edges(8));
    assert_eq!(
        production.nodes[0],
        ExpectedNode::MemcpyD2d { elements: 8095 }
    );
    assert_eq!(
        production.nodes[1],
        ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [1, 1, 32], [128, 1, 1])
    );
    assert_eq!(
        production.nodes[2],
        ExpectedNode::kernel("splitm_reduce", [4, 1, 1], [256, 1, 1])
    );
    assert_eq!(
        production.nodes[3],
        ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [2, 1, 32], [128, 1, 1])
    );
    assert_eq!(
        production.nodes[4],
        ExpectedNode::kernel("splitm_reduce", [13, 1, 1], [256, 1, 1])
    );
    assert_eq!(
        production.nodes[5],
        ExpectedNode::kernel("tn_narrow_splitm_partial_aligned", [1, 2, 86], [128, 1, 1])
    );
    assert_eq!(
        production.nodes[6],
        ExpectedNode::kernel("splitm_reduce", [16, 1, 1], [256, 1, 1])
    );
    assert_eq!(production.nodes[7], ExpectedNode::adamw(8095));

    let direct = expected_topology(PipelineArm::DirectControl);
    assert_eq!(direct.nodes.len(), 5);
    assert_eq!(direct.edges, linear_edges(5));
    assert_eq!(direct.nodes[0], ExpectedNode::MemcpyD2d { elements: 8095 });
    assert_eq!(direct.nodes[4], ExpectedNode::adamw(8095));
    assert!(
        direct.nodes[1..4]
            .iter()
            .all(|node| node.symbol() == Some("tn_narrow"))
    );
}

#[test]
fn ordinal_trajectory_schedule_is_one_through_ten() {
    assert_eq!(ordinal_schedule(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn paired_schedule_has_101_windows_in_both_orders() {
    let schedule = paired_window_schedule();
    assert_eq!(schedule.len(), 202);
    assert_eq!(
        schedule
            .iter()
            .filter(|entry| entry.order == ArmOrder::ProductionFirst)
            .count(),
        101
    );
    assert_eq!(
        schedule
            .iter()
            .filter(|entry| entry.order == ArmOrder::DirectFirst)
            .count(),
        101
    );
    for order in [ArmOrder::ProductionFirst, ArmOrder::DirectFirst] {
        let indices = schedule
            .iter()
            .filter(|entry| entry.order == order)
            .map(|entry| entry.window)
            .collect::<Vec<_>>();
        assert_eq!(indices, (0..101).collect::<Vec<_>>());
    }
}

#[test]
fn promotion_requires_every_mode_and_order_gate() {
    let passing = PromotionEvidence::uniform(1.01, 1.000_001);
    assert!(passing.promoted());
    assert!(!PromotionEvidence::uniform(1.009_999, 1.1).promoted());
    assert!(!PromotionEvidence::uniform(1.2, 1.0).promoted());
    for mode in [PipelineMode::Eager, PipelineMode::Graph] {
        for order in [ArmOrder::ProductionFirst, ArmOrder::DirectFirst] {
            let mut evidence = passing;
            evidence.cell_mut(mode, order).median_ratio = 1.0;
            assert!(!evidence.promoted());
        }
    }
}

#[test]
fn paired_absolute_samples_are_preserved_before_ratio_derivation() {
    let samples = PairedAbsoluteSamples {
        production_device_us: vec![2.0, 4.0],
        direct_device_us: vec![3.0, 8.0],
        production_host_us: vec![1.0, 2.0],
        direct_host_us: vec![2.0, 3.0],
    };
    assert_eq!(samples.device_ratios().unwrap(), vec![1.5, 2.0]);
    assert_eq!(samples.host_ratios().unwrap(), vec![2.0, 1.5]);
    assert_eq!(samples.production_device_us, vec![2.0, 4.0]);
    assert_eq!(samples.direct_host_us, vec![2.0, 3.0]);
}

#[test]
fn stable_schedule_identity_covers_every_protocol_dimension() {
    let canonical = stable_schedule_canonical();
    for required in [
        "modes=eager,graph",
        "orders=production_first,direct_first",
        "calls=1024:47:17:7296,1024:128:25:0,4096:64:64:3200",
        "arena=16:8095:16",
        "corpus_seed=57344",
        "logical_steps_per_window=1",
        "bias_update=before_each_logical_step",
        "reset=active_only_d2d",
        "calibration=disposable:3:no_resize",
    ] {
        assert!(canonical.contains(required), "missing {required}");
    }
}

#[test]
fn memcpy_graph_params_use_deferred_initialization() {
    let source = include_str!("training_pipeline_tests.rs");
    assert!(source.contains("MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit()"));
    let forbidden = ["zeroed::<sys::", "CUDA_MEMCPY3D>"].concat();
    assert!(!source.contains(&forbidden));
}

#[test]
fn quiet_gate_requires_five_consecutive_threshold_samples() {
    let samples = [
        (0, 0),
        (1, 1),
        (12, 0),
        (0, 1),
        (1, 0),
        (0, 0),
        (1, 1),
        (0, 0),
    ];
    assert_eq!(first_quiet_streak_end(&samples, 5, 1), Some(7));
    assert_eq!(first_quiet_streak_end(&samples[..7], 5, 1), None);
}

#[test]
fn quiet_gate_threshold_is_inclusive() {
    assert_eq!(first_quiet_streak_end(&[(1, 1); 5], 5, 1), Some(4));
    assert_eq!(
        first_quiet_streak_end(&[(0, 0), (2, 0), (0, 0)], 2, 1),
        None
    );
}

#[cfg(feature = "cuda")]
mod cuda_qualification {
    use std::ffi::CStr;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::Path;
    use std::process::Command;
    use std::sync::Arc;
    use std::time::Instant;

    use cudarc::driver::{CudaGraph, CudaStream, LaunchConfig, PushKernelArg, sys};
    use sha2::{Digest, Sha256};

    use super::{
        ACTIVE_ELEMENTS, ArmOrder, ExpectedNode, GUARD_ELEMENTS, PairedAbsoluteSamples,
        PipelineArm, PipelineMode, PromotionCell, PromotionEvidence, expected_topology,
        paired_window_schedule, stable_schedule_canonical, training_arena_layout,
    };
    use crate::mamba_ssm::gpu::adamw::{AdamWBiasFactors, AdamWParamPtrs, GpuAdamW};
    use crate::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dw_grad;
    use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::graph_capture::{
        capture_into_graph, capture_into_graph_with_gemm_plan,
    };
    use crate::mamba_ssm::gpu::kernel_identity::CapturedGemmGraphPlan;

    const GUARD_BITS: u32 = 0x7fc0_5a11;
    const LR: f32 = 1.0e-4;

    #[derive(Clone, Copy)]
    struct CellSpec {
        m: usize,
        k: usize,
        n: usize,
        offset: usize,
    }

    const CALLS: [CellSpec; 3] = [
        CellSpec {
            m: 1024,
            k: 47,
            n: 17,
            offset: 7296,
        },
        CellSpec {
            m: 1024,
            k: 128,
            n: 25,
            offset: 0,
        },
        CellSpec {
            m: 4096,
            k: 64,
            n: 64,
            offset: 3200,
        },
    ];

    struct CellInputs {
        x: GpuBuffer,
        dy: GpuBuffer,
        host_x: Vec<f32>,
        host_dy: Vec<f32>,
    }

    struct PipelineFixture {
        grad_storage: GpuBuffer,
        grad_seed: GpuBuffer,
        weight: GpuBuffer,
        adam: GpuAdamW,
        bias: AdamWBiasFactors,
        inputs: [CellInputs; 3],
        host_seed: Vec<f32>,
    }

    impl PipelineFixture {
        fn new(ctx: &GpuCtx, seed: u32) -> Result<Self, String> {
            let layout = training_arena_layout();
            let mut host_grad = deterministic_f32(layout.storage_len(), seed);
            host_grad[..layout.guard_len].fill(f32::from_bits(GUARD_BITS));
            host_grad[layout.guard_len + layout.active_len..].fill(f32::from_bits(GUARD_BITS));
            let grad_storage = GpuBuffer::from_cpu(&ctx.stream, &host_grad)?;
            let host_seed =
                host_grad[layout.guard_len..layout.guard_len + layout.active_len].to_vec();
            let grad_seed = GpuBuffer::from_cpu(&ctx.stream, &host_seed)?;
            let weight = GpuBuffer::from_cpu(
                &ctx.stream,
                &deterministic_f32(ACTIVE_ELEMENTS, seed ^ 0x51aa),
            )?;
            let adam = GpuAdamW::new(&ctx.stream, ACTIVE_ELEMENTS)?
                .with_lr(LR)
                .with_weight_decay(0.01);
            let mut bias = AdamWBiasFactors::new(&ctx.stream)?;
            bias.write(&ctx.stream, 1.0, 1.0, LR)?;
            let inputs = CALLS
                .into_iter()
                .enumerate()
                .map(|(index, spec)| {
                    let host_x = deterministic_f32(spec.m * spec.k, seed ^ (0x100 + index as u32));
                    let host_dy = deterministic_f32(spec.m * spec.n, seed ^ (0x200 + index as u32));
                    Ok::<_, String>(CellInputs {
                        x: GpuBuffer::from_cpu(&ctx.stream, &host_x)?,
                        dy: GpuBuffer::from_cpu(&ctx.stream, &host_dy)?,
                        host_x,
                        host_dy,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_: Vec<CellInputs>| "training fixture requires three inputs")?;
            Ok(Self {
                grad_storage,
                grad_seed,
                weight,
                adam,
                bias,
                inputs,
                host_seed,
            })
        }

        fn active_ptr(&self) -> sys::CUdeviceptr {
            self.grad_storage.cached_ptr() + (GUARD_ELEMENTS * 4) as u64
        }

        fn grad(&self, spec: CellSpec) -> GradSlice {
            GradSlice::from_offset(self.active_ptr(), spec.offset, spec.k * spec.n)
        }

        fn reset_active(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
            cuda_ok(
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        self.active_ptr(),
                        self.grad_seed.cached_ptr(),
                        ACTIVE_ELEMENTS * 4,
                        stream.cu_stream(),
                    )
                },
                "active gradient D2D reset",
            )
        }

        fn adam_ptrs(&self) -> AdamWParamPtrs {
            AdamWParamPtrs {
                weight: self.weight.cached_ptr(),
                grad: self.active_ptr(),
                m: self.adam.m.cached_ptr(),
                v: self.adam.v.cached_ptr(),
            }
        }

        fn write_next_bias(&mut self, ctx: &GpuCtx) -> Result<u64, String> {
            let (ordinal, bc1, bc2) = self.adam.advance();
            self.bias.write(&ctx.stream, bc1, bc2, LR)?;
            Ok(ordinal)
        }

        fn validate_guards(&self, ctx: &GpuCtx) -> Result<(), String> {
            let values = self.grad_storage.to_cpu(&ctx.stream)?;
            let layout = training_arena_layout();
            if values[..layout.guard_len]
                .iter()
                .any(|value| value.to_bits() != GUARD_BITS)
                || values[layout.guard_len + layout.active_len..]
                    .iter()
                    .any(|value| value.to_bits() != GUARD_BITS)
            {
                return Err("gradient arena red zone changed".into());
            }
            Ok(())
        }
    }

    struct CapturedPipeline {
        graph: CudaGraph,
        plan: Option<CapturedGemmGraphPlan>,
        _resources: std::rc::Rc<crate::mamba_ssm::gpu::context::GpuCtxResources>,
    }

    impl CapturedPipeline {
        fn replay(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
            match &self.plan {
                Some(plan) => plan.with_validated_launch(ctx, label, || {
                    self.graph
                        .launch()
                        .map_err(|error| format!("graph replay: {error:?}"))
                }),
                None => self
                    .graph
                    .launch()
                    .map_err(|error| format!("graph replay: {error:?}")),
            }
        }
    }

    fn deterministic_f32(len: usize, mut state: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xffff) as f32 / 65_536.0 - 0.5
            })
            .collect()
    }

    fn configured_ctx(device: &GpuDevice) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_gemm_mode(crate::mamba_ssm::gpu::GemmMode::Deterministic)
            .unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFma);
        Ok(ctx)
    }

    fn launch_production(ctx: &GpuCtx, fixture: &PipelineFixture) -> Result<(), String> {
        for (spec, inputs) in CALLS.into_iter().zip(&fixture.inputs) {
            gpu_gemm_bi_backward_dw_grad(
                ctx,
                &fixture.grad(spec),
                &inputs.dy,
                &inputs.x,
                spec.m,
                spec.k,
                spec.n,
            )?;
        }
        Ok(())
    }

    fn launch_direct(ctx: &GpuCtx, fixture: &PipelineFixture) -> Result<(), String> {
        for (spec, inputs) in CALLS.into_iter().zip(&fixture.inputs) {
            let output = fixture.grad(spec).ptr();
            let a = inputs.x.cached_ptr();
            let b = inputs.dy.cached_ptr();
            let alpha = 1.0f32;
            let m = i32::try_from(spec.m).map_err(|_| "M exceeds i32")?;
            let k = i32::try_from(spec.k).map_err(|_| "K exceeds i32")?;
            let n = i32::try_from(spec.n).map_err(|_| "N exceeds i32")?;
            let grid_m = u32::try_from(spec.k.div_ceil(64)).map_err(|_| "grid M exceeds u32")?;
            let grid_n = u32::try_from(spec.n.div_ceil(32)).map_err(|_| "grid N exceeds u32")?;
            let config = LaunchConfig {
                grid_dim: (grid_m * grid_n, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_tn_narrow);
            builder.arg(&output);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&alpha);
            builder.arg(&m);
            builder.arg(&k);
            builder.arg(&n);
            unsafe { builder.launch(config) }
                .map_err(|error| format!("direct TN launch: {error:?}"))?;
        }
        Ok(())
    }

    fn launch_adamw(ctx: &GpuCtx, fixture: &PipelineFixture) -> Result<(), String> {
        fixture.adam.step_one_capturable(
            ctx,
            &ctx.kernels.adamw_step_f32_capturable,
            fixture.adam_ptrs(),
            fixture.bias.ptr(),
            ACTIVE_ELEMENTS,
        )
    }

    fn launch_pipeline(
        ctx: &GpuCtx,
        fixture: &PipelineFixture,
        arm: PipelineArm,
    ) -> Result<(), String> {
        fixture.reset_active(&ctx.stream)?;
        match arm {
            PipelineArm::Production => launch_production(ctx, fixture)?,
            PipelineArm::DirectControl => launch_direct(ctx, fixture)?,
        }
        launch_adamw(ctx, fixture)
    }

    fn capture_pipeline(
        ctx: &GpuCtx,
        fixture: &PipelineFixture,
        arm: PipelineArm,
    ) -> Result<CapturedPipeline, String> {
        ctx.presize_bi_scratch()?;
        let resources = ctx.resource_anchor();
        let (graph, plan) = match arm {
            PipelineArm::Production => {
                let manifest =
                    ctx.record_eager_gemm_manifest(|| launch_production(ctx, fixture))?;
                ctx.stream
                    .synchronize()
                    .map_err(|error| format!("manifest sync: {error:?}"))?;
                ctx.freeze_graph_scratch();
                let (graph, plan) = unsafe {
                    capture_into_graph_with_gemm_plan(
                        ctx,
                        manifest.route_capacity,
                        &manifest,
                        || launch_pipeline(ctx, fixture, arm),
                    )
                }?;
                (
                    graph,
                    Some(
                        plan.ok_or_else(|| "production capture returned no GEMM plan".to_string())?,
                    ),
                )
            }
            PipelineArm::DirectControl => {
                ctx.freeze_graph_scratch();
                let graph = unsafe {
                    capture_into_graph(&ctx.stream, || launch_pipeline(ctx, fixture, arm))
                }?;
                (graph, None)
            }
        };
        ctx.note_graph_capture();
        Ok(CapturedPipeline {
            graph,
            plan,
            _resources: resources,
        })
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    #[derive(Debug)]
    enum ObservedNode {
        Memcpy {
            src: sys::CUdeviceptr,
            dst: sys::CUdeviceptr,
            bytes: usize,
            height: usize,
            depth: usize,
            device_to_device: bool,
            zero_offsets: bool,
        },
        Kernel {
            symbol: String,
            grid: [u32; 3],
            block: [u32; 3],
            shared: u32,
            arguments: *mut *mut std::ffi::c_void,
        },
    }

    fn observe_node(node: sys::CUgraphNode) -> Result<ObservedNode, String> {
        let mut node_type = std::mem::MaybeUninit::uninit();
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) },
            "node type",
        )?;
        match unsafe { node_type.assume_init() } {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
                let mut params = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit();
                cuda_ok(
                    unsafe { sys::cuGraphMemcpyNodeGetParams(node, params.as_mut_ptr()) },
                    "memcpy params",
                )?;
                let params = unsafe { params.assume_init() };
                Ok(ObservedNode::Memcpy {
                    src: params.srcDevice,
                    dst: params.dstDevice,
                    bytes: params.WidthInBytes,
                    height: params.Height,
                    depth: params.Depth,
                    device_to_device: params.srcMemoryType
                        == sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                        && params.dstMemoryType == sys::CUmemorytype::CU_MEMORYTYPE_DEVICE
                        && params.srcHost.is_null()
                        && params.dstHost.is_null()
                        && params.srcArray.is_null()
                        && params.dstArray.is_null(),
                    zero_offsets: params.srcXInBytes == 0
                        && params.srcY == 0
                        && params.srcZ == 0
                        && params.srcLOD == 0
                        && params.dstXInBytes == 0
                        && params.dstY == 0
                        && params.dstZ == 0
                        && params.dstLOD == 0,
                })
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                let mut params = unsafe { std::mem::zeroed() };
                cuda_ok(
                    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
                    "kernel params",
                )?;
                let mut name = std::ptr::null();
                cuda_ok(
                    unsafe { sys::cuFuncGetName(&mut name, params.func) },
                    "kernel name",
                )?;
                let symbol = unsafe { CStr::from_ptr(name) }
                    .to_str()
                    .map_err(|error| format!("kernel name UTF-8: {error}"))?
                    .to_owned();
                Ok(ObservedNode::Kernel {
                    symbol,
                    grid: [params.gridDimX, params.gridDimY, params.gridDimZ],
                    block: [params.blockDimX, params.blockDimY, params.blockDimZ],
                    shared: params.sharedMemBytes,
                    arguments: params.kernelParams,
                })
            }
            other => Err(format!("unexpected graph node type {other:?}")),
        }
    }

    fn validate_actual_topology(
        graph: &CudaGraph,
        fixture: &PipelineFixture,
        arm: PipelineArm,
    ) -> Result<String, String> {
        let expected = expected_topology(arm);
        let mut count = 0usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "node count",
        )?;
        if count != expected.nodes.len() {
            return Err(format!(
                "observed {count} nodes, expected {}",
                expected.nodes.len()
            ));
        }
        let mut nodes = vec![std::ptr::null_mut(); count];
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) },
            "nodes",
        )?;
        let mut edge_count = 0usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph.cu_graph(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                )
            },
            "edge count",
        )?;
        if edge_count != expected.edges.len() {
            return Err(format!(
                "observed {edge_count} edges, expected {}",
                expected.edges.len()
            ));
        }
        let mut from = vec![std::ptr::null_mut(); edge_count];
        let mut to = vec![std::ptr::null_mut(); edge_count];
        let mut data = vec![unsafe { std::mem::zeroed() }; edge_count];
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph.cu_graph(),
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    data.as_mut_ptr(),
                    &mut edge_count,
                )
            },
            "edges",
        )?;
        for edge in &data {
            if edge.from_port != 0
                || edge.to_port != 0
                || edge.type_ != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
                || edge.reserved != [0; 5]
            {
                return Err("graph edge data is not the CUDA default descriptor".into());
            }
        }
        let root = nodes
            .iter()
            .copied()
            .find(|node| !to.contains(node))
            .ok_or_else(|| "graph has no root".to_string())?;
        let mut ordered = Vec::with_capacity(count);
        let mut current = root;
        loop {
            ordered.push(current);
            let successors = from
                .iter()
                .zip(&to)
                .filter_map(|(source, target)| (*source == current).then_some(*target))
                .collect::<Vec<_>>();
            match successors.as_slice() {
                [] => break,
                [next] => current = *next,
                _ => return Err("graph is not a linear chain".into()),
            }
        }
        if ordered.len() != count {
            return Err("graph chain does not cover every node".into());
        }
        let arm_name = match arm {
            PipelineArm::Production => "production",
            PipelineArm::DirectControl => "direct_control",
        };
        let mut canonical = format!(
            "task8_topology\narm={arm_name}\nnode_count={count}\nedge_count={edge_count}\nedge_semantics=linear_default_zero_ports_reserved_zero\n"
        );
        for (index, (node, contract)) in ordered.into_iter().zip(expected.nodes).enumerate() {
            let observed = observe_node(node)?;
            let semantic = semantic_node_canonical(index, contract);
            match (observed, contract) {
                (
                    ObservedNode::Memcpy {
                        src,
                        dst,
                        bytes,
                        height,
                        depth,
                        device_to_device,
                        zero_offsets,
                    },
                    ExpectedNode::MemcpyD2d { elements },
                ) if src == fixture.grad_seed.cached_ptr()
                    && dst == fixture.active_ptr()
                    && bytes == elements * 4
                    && height == 1
                    && depth == 1
                    && device_to_device
                    && zero_offsets => {}
                (
                    ObservedNode::Kernel {
                        symbol,
                        grid,
                        block,
                        shared,
                        arguments,
                    },
                    ExpectedNode::Kernel {
                        symbol: expected_symbol,
                        grid: expected_grid,
                        block: expected_block,
                        dynamic_shared_bytes,
                        len,
                    },
                ) if symbol == expected_symbol
                    && grid == expected_grid
                    && block == expected_block
                    && shared == dynamic_shared_bytes =>
                {
                    if symbol == "adamw_step_f32_capturable" {
                        validate_adamw_arguments(
                            arguments,
                            fixture,
                            len.ok_or_else(|| "AdamW length is absent".to_string())?,
                        )?;
                    } else if arm == PipelineArm::DirectControl {
                        let call = index
                            .checked_sub(1)
                            .and_then(|call| CALLS.get(call))
                            .copied()
                            .ok_or_else(|| {
                                "direct kernel is outside the three-call segment".to_string()
                            })?;
                        validate_direct_arguments(arguments, fixture, call)?;
                    }
                }
                (observed, expected) => {
                    return Err(format!(
                        "node {index} mismatch: {observed:?} != {expected:?}"
                    ));
                }
            }
            canonical.push_str(&semantic);
            canonical.push('\n');
        }
        Ok(sha256_bytes(canonical.as_bytes()))
    }

    fn semantic_node_canonical(index: usize, node: ExpectedNode) -> String {
        match node {
            ExpectedNode::MemcpyD2d { elements } => {
                format!("node={index}:memcpy_d2d:elements={elements}")
            }
            ExpectedNode::Kernel {
                symbol,
                grid,
                block,
                dynamic_shared_bytes,
                len,
            } => format!(
                "node={index}:kernel={symbol}:grid={},{},{}:block={},{},{}:shared={dynamic_shared_bytes}:len={}",
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                len.map_or_else(|| "none".to_string(), |value| value.to_string()),
            ),
        }
    }

    fn kernel_argument<T: Copy>(
        arguments: *mut *mut std::ffi::c_void,
        index: usize,
    ) -> Result<T, String> {
        if arguments.is_null() {
            return Err("kernel argument table is null".into());
        }
        let slot = unsafe { *arguments.add(index) };
        if slot.is_null() {
            return Err(format!("kernel argument {index} is null"));
        }
        Ok(unsafe { *(slot.cast::<T>()) })
    }

    fn validate_adamw_arguments(
        arguments: *mut *mut std::ffi::c_void,
        fixture: &PipelineFixture,
        len: usize,
    ) -> Result<(), String> {
        let ptrs = fixture.adam_ptrs();
        let observed = [
            kernel_argument::<sys::CUdeviceptr>(arguments, 0)?,
            kernel_argument::<sys::CUdeviceptr>(arguments, 1)?,
            kernel_argument::<sys::CUdeviceptr>(arguments, 2)?,
            kernel_argument::<sys::CUdeviceptr>(arguments, 3)?,
        ];
        if observed != [ptrs.weight, ptrs.grad, ptrs.m, ptrs.v]
            || kernel_argument::<sys::CUdeviceptr>(arguments, 9)? != fixture.bias.ptr()
            || kernel_argument::<i32>(arguments, 10)?
                != i32::try_from(len).map_err(|_| "AdamW length exceeds i32")?
        {
            return Err("AdamW graph argument identity changed".into());
        }
        Ok(())
    }

    fn validate_direct_arguments(
        arguments: *mut *mut std::ffi::c_void,
        fixture: &PipelineFixture,
        call: CellSpec,
    ) -> Result<(), String> {
        let input_index = CALLS
            .iter()
            .position(|candidate| candidate.offset == call.offset)
            .ok_or_else(|| "direct call is absent from the schedule".to_string())?;
        let input = &fixture.inputs[input_index];
        let expected_ptrs = [
            fixture.grad(call).ptr(),
            input.x.cached_ptr(),
            input.dy.cached_ptr(),
        ];
        let observed_ptrs = [
            kernel_argument::<sys::CUdeviceptr>(arguments, 0)?,
            kernel_argument::<sys::CUdeviceptr>(arguments, 1)?,
            kernel_argument::<sys::CUdeviceptr>(arguments, 2)?,
        ];
        let observed_dims = [
            kernel_argument::<i32>(arguments, 4)?,
            kernel_argument::<i32>(arguments, 5)?,
            kernel_argument::<i32>(arguments, 6)?,
        ];
        let expected_dims = [
            i32::try_from(call.m).map_err(|_| "M exceeds i32")?,
            i32::try_from(call.k).map_err(|_| "K exceeds i32")?,
            i32::try_from(call.n).map_err(|_| "N exceeds i32")?,
        ];
        if observed_ptrs != expected_ptrs
            || observed_dims != expected_dims
            || kernel_argument::<f32>(arguments, 3)?.to_bits() != 1.0f32.to_bits()
        {
            return Err("direct TN graph arguments changed".into());
        }
        Ok(())
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TrajectorySnapshot {
        gradient: Vec<u32>,
        weight: Vec<u32>,
        m: Vec<u32>,
        v: Vec<u32>,
    }

    fn bits(values: Vec<f32>) -> Vec<u32> {
        values.into_iter().map(f32::to_bits).collect()
    }

    fn snapshot(ctx: &GpuCtx, fixture: &PipelineFixture) -> Result<TrajectorySnapshot, String> {
        let layout = training_arena_layout();
        let gradient = fixture.grad_storage.to_cpu(&ctx.stream)?;
        Ok(TrajectorySnapshot {
            gradient: gradient[layout.guard_len..layout.guard_len + layout.active_len]
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            weight: bits(fixture.weight.to_cpu(&ctx.stream)?),
            m: bits(fixture.adam.m.to_cpu(&ctx.stream)?),
            v: bits(fixture.adam.v.to_cpu(&ctx.stream)?),
        })
    }

    fn scheduled_gradient_oracle(fixture: &PipelineFixture, arm: PipelineArm) -> Vec<u32> {
        let mut output = fixture.host_seed.clone();
        for (index, call) in CALLS.into_iter().enumerate() {
            let input = &fixture.inputs[index];
            for row in 0..call.k {
                for column in 0..call.n {
                    let sum = match arm {
                        PipelineArm::DirectControl => {
                            let mut sum = 0.0f32;
                            for reduction in 0..call.m {
                                sum = input.host_x[reduction * call.k + row]
                                    .mul_add(input.host_dy[reduction * call.n + column], sum);
                            }
                            sum
                        }
                        PipelineArm::Production => {
                            let (chunk, chunks) = match (call.m, call.k, call.n) {
                                (1024, 47, 17) | (1024, 128, 25) => (32, 32),
                                (4096, 64, 64) => (48, 86),
                                _ => unreachable!("only admitted Task 8 cells are scheduled"),
                            };
                            let mut ordered = 0.0f64;
                            for partition in 0..chunks {
                                let mut partial = 0.0f32;
                                let begin = partition * chunk;
                                let end = (begin + chunk).min(call.m);
                                for reduction in begin..end {
                                    partial = input.host_x[reduction * call.k + row].mul_add(
                                        input.host_dy[reduction * call.n + column],
                                        partial,
                                    );
                                }
                                ordered += f64::from(partial);
                            }
                            ordered as f32
                        }
                    };
                    let target = call.offset + row * call.n + column;
                    output[target] += sum;
                }
            }
        }
        output.into_iter().map(f32::to_bits).collect()
    }

    fn run_trajectory_contract(ctx: &GpuCtx, arm: PipelineArm) -> Result<[String; 2], String> {
        let seed = 0x71a5_9000 | arm as u32;
        let mut eager = PipelineFixture::new(ctx, seed)?;
        let mut graph_a = PipelineFixture::new(ctx, seed)?;
        let mut graph_b = PipelineFixture::new(ctx, seed)?;
        let captured_a = capture_pipeline(ctx, &graph_a, arm)?;
        let captured_b = capture_pipeline(ctx, &graph_b, arm)?;
        let topology_a = validate_actual_topology(&captured_a.graph, &graph_a, arm)?;
        let topology_b = validate_actual_topology(&captured_b.graph, &graph_b, arm)?;
        if topology_a != topology_b {
            return Err(format!(
                "{arm:?} graph A/B semantic topology digests differ"
            ));
        }
        let oracle = scheduled_gradient_oracle(&eager, arm);

        for ordinal in super::ordinal_schedule() {
            if eager.write_next_bias(ctx)? != ordinal
                || graph_a.write_next_bias(ctx)? != ordinal
                || graph_b.write_next_bias(ctx)? != ordinal
            {
                return Err(format!("AdamW ordinal schedule diverged at step {ordinal}"));
            }
            launch_pipeline(ctx, &eager, arm)?;
            captured_a.replay(ctx, "Task 8 graph A")?;
            captured_b.replay(ctx, "Task 8 graph B")?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("trajectory sync: {error:?}"))?;
            let eager_snapshot = snapshot(ctx, &eager)?;
            let graph_a_snapshot = snapshot(ctx, &graph_a)?;
            let graph_b_snapshot = snapshot(ctx, &graph_b)?;
            if eager_snapshot.gradient != oracle {
                return Err(format!(
                    "scheduled {arm:?} gradient oracle failed at ordinal {ordinal}"
                ));
            }
            if eager_snapshot != graph_a_snapshot || eager_snapshot != graph_b_snapshot {
                return Err(format!(
                    "{arm:?} eager/graph trajectories differ at ordinal {ordinal}"
                ));
            }
            eager.validate_guards(ctx)?;
            graph_a.validate_guards(ctx)?;
            graph_b.validate_guards(ctx)?;
        }
        Ok([topology_a, topology_b])
    }

    struct TimedPipeline {
        captured: Option<CapturedPipeline>,
        fixture: PipelineFixture,
        arm: PipelineArm,
        topology_digest: Option<String>,
    }

    impl TimedPipeline {
        fn new(
            ctx: &GpuCtx,
            arm: PipelineArm,
            mode: PipelineMode,
            seed: u32,
        ) -> Result<Self, String> {
            let fixture = PipelineFixture::new(ctx, seed)?;
            let (captured, topology_digest) = match mode {
                PipelineMode::Eager => (None, None),
                PipelineMode::Graph => {
                    let captured = capture_pipeline(ctx, &fixture, arm)?;
                    let digest = validate_actual_topology(&captured.graph, &fixture, arm)?;
                    (Some(captured), Some(digest))
                }
            };
            Ok(Self {
                captured,
                fixture,
                arm,
                topology_digest,
            })
        }

        fn measure_step(&mut self, ctx: &GpuCtx) -> Result<TimingSample, String> {
            let start = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("pipeline start event: {error:?}"))?;
            let host_start = Instant::now();
            self.fixture.write_next_bias(ctx)?;
            match &self.captured {
                Some(captured) => captured.replay(ctx, "Task 8 timed graph")?,
                None => launch_pipeline(ctx, &self.fixture, self.arm)?,
            }
            let host_enqueue_us = host_start.elapsed().as_secs_f64() * 1_000_000.0;
            let end = ctx
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("pipeline end event: {error:?}"))?;
            let device_elapsed_us = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|error| format!("pipeline event elapsed: {error:?}"))?,
            ) * 1_000.0;
            if !device_elapsed_us.is_finite()
                || device_elapsed_us <= 0.0
                || !host_enqueue_us.is_finite()
                || host_enqueue_us <= 0.0
            {
                return Err("pipeline timing is not positive and finite".into());
            }
            Ok(TimingSample {
                device_elapsed_us,
                host_enqueue_us,
            })
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TimingSample {
        device_elapsed_us: f64,
        host_enqueue_us: f64,
    }

    fn percentile(mut values: Vec<f64>, quantile: f64) -> Result<f64, String> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err("percentile requires finite samples".into());
        }
        values.sort_by(f64::total_cmp);
        let index = ((values.len() - 1) as f64 * quantile).round() as usize;
        Ok(values[index])
    }

    fn cuda_uuid() -> Result<String, String> {
        let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
        cuda_ok(
            unsafe { sys::cuDeviceGetUuid_v2(&mut uuid, 0) },
            "query CUDA UUID",
        )?;
        let bytes = uuid.bytes.map(|byte| byte as u8);
        Ok(format!(
            "GPU-{}-{}-{}-{}-{}",
            bytes[..4]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            bytes[4..6]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            bytes[6..8]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            bytes[8..10]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            bytes[10..]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        ))
    }

    fn require_quiet_gpu() -> Result<String, String> {
        let uuid = cuda_uuid()?;
        let mut accepted = Vec::with_capacity(5);
        for attempt in 0..50 {
            let sample = query_quiet_sample(&uuid)?;
            if sample.gpu_util <= 1 && sample.memory_util <= 1 {
                accepted.push(sample.text);
                if accepted.len() == 5 {
                    return Ok(accepted.join(" | "));
                }
            } else {
                accepted.clear();
            }
            if attempt + 1 != 50 {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        Err("GPU did not produce five consecutive <=1% utilization samples in 50 attempts".into())
    }

    struct QuietSample {
        gpu_util: u32,
        memory_util: u32,
        text: String,
    }

    fn query_quiet_sample(uuid: &str) -> Result<QuietSample, String> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,pstate,clocks.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                uuid,
            ])
            .output()
            .map_err(|error| format!("run nvidia-smi quiet gate: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "nvidia-smi quiet gate failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let telemetry = String::from_utf8(output.stdout)
            .map_err(|error| format!("nvidia-smi output is not UTF-8: {error}"))?;
        let fields = telemetry
            .trim()
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        if fields.len() != 7 || fields[0] != uuid {
            return Err(format!("malformed GPU telemetry: {}", telemetry.trim()));
        }
        let gpu_util = fields[1]
            .parse::<u32>()
            .map_err(|error| format!("malformed GPU utilization {:?}: {error}", fields[1]))?;
        let memory_util = fields[2]
            .parse::<u32>()
            .map_err(|error| format!("malformed memory utilization {:?}: {error}", fields[2]))?;
        let processes = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                uuid,
            ])
            .output()
            .map_err(|error| format!("query competing GPU processes: {error}"))?;
        if !processes.status.success() {
            return Err(format!(
                "query competing GPU processes failed: {}",
                String::from_utf8_lossy(&processes.stderr)
            ));
        }
        let process_text = String::from_utf8(processes.stdout)
            .map_err(|error| format!("compute process output is not UTF-8: {error}"))?;
        let own_pid = std::process::id().to_string();
        for line in process_text.lines().filter(|line| !line.trim().is_empty()) {
            let process_fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            if process_fields.len() != 3
                || process_fields[0] != uuid
                || process_fields[1] != own_pid
            {
                return Err(format!("competing GPU process detected: {line}"));
            }
        }
        Ok(QuietSample {
            gpu_util,
            memory_util,
            text: format!(
                "telemetry={} compute_processes={}",
                telemetry.trim(),
                process_text.trim()
            ),
        })
    }

    fn measure_order(
        ctx: &GpuCtx,
        production: &mut TimedPipeline,
        direct: &mut TimedPipeline,
        order: ArmOrder,
    ) -> Result<(TimingSample, TimingSample), String> {
        match order {
            ArmOrder::ProductionFirst => {
                let production = production.measure_step(ctx)?;
                let direct = direct.measure_step(ctx)?;
                Ok((production, direct))
            }
            ArmOrder::DirectFirst => {
                let direct = direct.measure_step(ctx)?;
                let production = production.measure_step(ctx)?;
                Ok((production, direct))
            }
        }
    }

    fn measure_mode(
        ctx: &GpuCtx,
        mode: PipelineMode,
        order: ArmOrder,
        seed: u32,
    ) -> Result<ModeMeasurement, String> {
        // Calibration owns disposable mutable state and cannot influence routing.
        let mut calibration_production =
            TimedPipeline::new(ctx, PipelineArm::Production, mode, seed ^ 0xca11)?;
        let mut calibration_direct =
            TimedPipeline::new(ctx, PipelineArm::DirectControl, mode, seed ^ 0xca11)?;
        for _ in 0..3 {
            let _ = measure_order(
                ctx,
                &mut calibration_production,
                &mut calibration_direct,
                order,
            )?;
        }
        drop((calibration_production, calibration_direct));

        let mut production = TimedPipeline::new(ctx, PipelineArm::Production, mode, seed)?;
        let mut direct = TimedPipeline::new(ctx, PipelineArm::DirectControl, mode, seed)?;
        let topology_digests = [
            production.topology_digest.clone(),
            direct.topology_digest.clone(),
        ];
        let preflight = require_quiet_gpu()?;
        let mut timed_gate = None;
        let mut samples = PairedAbsoluteSamples {
            production_device_us: Vec::with_capacity(super::WINDOWS_PER_ORDER),
            direct_device_us: Vec::with_capacity(super::WINDOWS_PER_ORDER),
            production_host_us: Vec::with_capacity(super::WINDOWS_PER_ORDER),
            direct_host_us: Vec::with_capacity(super::WINDOWS_PER_ORDER),
        };
        for window in paired_window_schedule()
            .into_iter()
            .filter(|entry| entry.order == order)
        {
            if window.window >= super::WINDOWS_PER_ORDER {
                return Err("paired window index escaped its fixed schedule".into());
            }
            if window.window == super::WINDOWS_PER_ORDER / 2 {
                timed_gate = Some(require_quiet_gpu()?);
            }
            let (production_sample, direct_sample) =
                measure_order(ctx, &mut production, &mut direct, order)?;
            samples
                .production_device_us
                .push(production_sample.device_elapsed_us);
            samples
                .direct_device_us
                .push(direct_sample.device_elapsed_us);
            samples
                .production_host_us
                .push(production_sample.host_enqueue_us);
            samples.direct_host_us.push(direct_sample.host_enqueue_us);
        }
        production.fixture.validate_guards(ctx)?;
        direct.fixture.validate_guards(ctx)?;
        let production_snapshot = snapshot(ctx, &production.fixture)?;
        let direct_snapshot = snapshot(ctx, &direct.fixture)?;
        if production_snapshot.gradient
            != scheduled_gradient_oracle(&production.fixture, PipelineArm::Production)
            || direct_snapshot.gradient
                != scheduled_gradient_oracle(&direct.fixture, PipelineArm::DirectControl)
        {
            return Err("post-timing gradient oracle failed".into());
        }
        for values in [
            production.fixture.weight.to_cpu(&ctx.stream)?,
            production.fixture.adam.m.to_cpu(&ctx.stream)?,
            production.fixture.adam.v.to_cpu(&ctx.stream)?,
            direct.fixture.weight.to_cpu(&ctx.stream)?,
            direct.fixture.adam.m.to_cpu(&ctx.stream)?,
            direct.fixture.adam.v.to_cpu(&ctx.stream)?,
        ] {
            if values.iter().any(|value| !value.is_finite()) {
                return Err("post-timing optimizer state contains non-finite values".into());
            }
        }
        let postflight = require_quiet_gpu()?;
        let device_ratios = samples.device_ratios()?;
        let promotion = PromotionCell {
            median_ratio: percentile(device_ratios.clone(), 0.5)?,
            p05_ratio: percentile(device_ratios.clone(), 0.05)?,
        };
        Ok(ModeMeasurement {
            promotion,
            samples,
            topology_digests,
            telemetry: [
                preflight,
                timed_gate.ok_or_else(|| "timed quiet gate was not recorded".to_string())?,
                postflight,
            ],
        })
    }

    struct ModeMeasurement {
        promotion: PromotionCell,
        samples: PairedAbsoluteSamples,
        topology_digests: [Option<String>; 2],
        telemetry: [String; 3],
    }

    struct PerformanceMeasurements {
        eager_production_first: ModeMeasurement,
        eager_direct_first: ModeMeasurement,
        graph_production_first: ModeMeasurement,
        graph_direct_first: ModeMeasurement,
    }

    impl PerformanceMeasurements {
        fn promotion(&self) -> PromotionEvidence {
            PromotionEvidence {
                eager_production_first: self.eager_production_first.promotion,
                eager_direct_first: self.eager_direct_first.promotion,
                graph_production_first: self.graph_production_first.promotion,
                graph_direct_first: self.graph_direct_first.promotion,
            }
        }

        fn validate_timed_topologies(&self) -> Result<(), String> {
            if self.eager_production_first.topology_digests != [None, None]
                || self.eager_direct_first.topology_digests != [None, None]
            {
                return Err("eager measurement unexpectedly owns graph topology".into());
            }
            let first = &self.graph_production_first.topology_digests;
            let second = &self.graph_direct_first.topology_digests;
            if first.iter().any(Option::is_none) || first != second {
                return Err(
                    "fresh timed graph semantic topologies differ across paired orders".into(),
                );
            }
            Ok(())
        }
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read executable {}: {error}", path.display()))?;
        Ok(sha256_bytes(&bytes))
    }

    fn json_quote(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len() + 2);
        escaped.push('"');
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                control if control.is_control() => {
                    use std::fmt::Write as _;
                    let _ = write!(escaped, "\\u{:04x}", control as u32);
                }
                other => escaped.push(other),
            }
        }
        escaped.push('"');
        escaped
    }

    fn json_f64_array(values: &[f64]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn json_string_array(values: &[String]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|value| json_quote(value))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn mode_samples_json(mode: &ModeMeasurement) -> Result<String, String> {
        let topology = mode
            .topology_digests
            .iter()
            .map(|digest| {
                digest
                    .as_deref()
                    .map_or_else(|| "null".to_string(), json_quote)
            })
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!(
            concat!(
                "{{\"production_device_us\":{},\"direct_device_us\":{},",
                "\"production_host_us\":{},\"direct_host_us\":{},",
                "\"device_ratios\":{},\"host_ratios\":{},\"timed_topology_digests\":[{}]}}"
            ),
            json_f64_array(&mode.samples.production_device_us),
            json_f64_array(&mode.samples.direct_device_us),
            json_f64_array(&mode.samples.production_host_us),
            json_f64_array(&mode.samples.direct_host_us),
            json_f64_array(&mode.samples.device_ratios()?),
            json_f64_array(&mode.samples.host_ratios()?),
            topology,
        ))
    }

    fn open_append_only_regular(path: &Path) -> Result<std::fs::File, String> {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                return Err(format!(
                    "Task 8 evidence destination {} is not a regular file",
                    path.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "preflight Task 8 evidence {}: {error}",
                    path.display()
                ));
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
                .open(path)
                .map_err(|error| format!("open Task 8 evidence {}: {error}", path.display()))?;
            if !file
                .metadata()
                .map_err(|error| format!("fstat Task 8 evidence {}: {error}", path.display()))?
                .file_type()
                .is_file()
            {
                return Err(format!(
                    "Task 8 evidence destination {} is not a regular file after open",
                    path.display()
                ));
            }
            Ok(file)
        }
        #[cfg(not(unix))]
        {
            Err("Task 8 append-only evidence requires Unix O_NONBLOCK/O_NOFOLLOW".into())
        }
    }

    fn append_evidence(
        ctx: &GpuCtx,
        measurements: &PerformanceMeasurements,
        observed_topologies: &[String],
    ) -> Result<(), String> {
        let evidence = measurements.promotion();
        let path = std::env::var_os("MAMBA_RS_TASK8_EVIDENCE").ok_or_else(|| {
            "MAMBA_RS_TASK8_EVIDENCE must name the append-only JSONL output".to_string()
        })?;
        let executable = std::env::current_exe()
            .map_err(|error| format!("resolve qualification executable: {error}"))?;
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        let artifacts = ctx.kernels.artifact_set_identity();
        let selector_hash = sha256_bytes(include_bytes!("dispatch.rs"));
        let topology_hash = sha256_bytes(observed_topologies.join("").as_bytes());
        let schedule_hash = sha256_bytes(stable_schedule_canonical().as_bytes());
        let payload_text = format!(
            concat!(
                "{{\"schema\":\"MambaBiTnTrainingPipelineQualificationV1\",",
                "\"scope\":\"measurement_only_until_completion_validation\",",
                "\"timing_clock\":\"cuda_event_device_elapsed\",",
                "\"host_api_enqueue_time_recorded_separately\":true,",
                "\"windows_per_order\":{},\"logical_steps_per_window\":1,",
                "\"measurement_kind\":\"single_training_step_latency\",",
                "\"calibration_iterations\":3,\"calibration_resized_measurement_iterations\":false,",
                "\"compiler\":{},\"artifact_set\":{},\"executable_sha256\":{},",
                "\"selector_table_sha256\":{},\"graph_topology_sha256\":{},\"schedule_sha256\":{},",
                "\"eager_production_first\":{{\"median\":{},\"p05\":{}}},",
                "\"eager_direct_first\":{{\"median\":{},\"p05\":{}}},",
                "\"graph_production_first\":{{\"median\":{},\"p05\":{}}},",
                "\"graph_direct_first\":{{\"median\":{},\"p05\":{}}},",
                "\"absolute_paired_samples\":[{},{},{},{}],",
                "\"telemetry_pre_timed_post\":[{},{},{},{}],\"promoted\":{}}}"
            ),
            super::WINDOWS_PER_ORDER,
            json_quote(&format!("{compiler:?}")),
            json_quote(&format!("{artifacts:?}")),
            json_quote(&sha256_file(&executable)?),
            json_quote(&selector_hash),
            json_quote(&topology_hash),
            json_quote(&schedule_hash),
            evidence.eager_production_first.median_ratio,
            evidence.eager_production_first.p05_ratio,
            evidence.eager_direct_first.median_ratio,
            evidence.eager_direct_first.p05_ratio,
            evidence.graph_production_first.median_ratio,
            evidence.graph_production_first.p05_ratio,
            evidence.graph_direct_first.median_ratio,
            evidence.graph_direct_first.p05_ratio,
            mode_samples_json(&measurements.eager_production_first)?,
            mode_samples_json(&measurements.eager_direct_first)?,
            mode_samples_json(&measurements.graph_production_first)?,
            mode_samples_json(&measurements.graph_direct_first)?,
            json_string_array(&measurements.eager_production_first.telemetry),
            json_string_array(&measurements.eager_direct_first.telemetry),
            json_string_array(&measurements.graph_production_first.telemetry),
            json_string_array(&measurements.graph_direct_first.telemetry),
            evidence.promoted(),
        );
        let completion = format!(
            "{{\"schema\":\"MambaBiTnTrainingPipelineQualificationCompletionV1\",\"record_sha256\":{},\"validated\":{}}}",
            json_quote(&sha256_bytes(payload_text.as_bytes())),
            evidence.promoted(),
        );
        let mut output = open_append_only_regular(Path::new(&path))?;
        writeln!(output, "{payload_text}")
            .and_then(|_| writeln!(output, "{completion}"))
            .map_err(|error| format!("append Task 8 evidence {:?}: {error}", path))?;
        output
            .sync_all()
            .map_err(|error| format!("sync Task 8 evidence {:?}: {error}", path))
    }

    fn run_performance_contract(ctx: &GpuCtx) -> Result<PerformanceMeasurements, String> {
        let measurements = PerformanceMeasurements {
            eager_production_first: measure_mode(
                ctx,
                PipelineMode::Eager,
                ArmOrder::ProductionFirst,
                0xe000,
            )?,
            eager_direct_first: measure_mode(
                ctx,
                PipelineMode::Eager,
                ArmOrder::DirectFirst,
                0xe000,
            )?,
            graph_production_first: measure_mode(
                ctx,
                PipelineMode::Graph,
                ArmOrder::ProductionFirst,
                0xe000,
            )?,
            graph_direct_first: measure_mode(
                ctx,
                PipelineMode::Graph,
                ArmOrder::DirectFirst,
                0xe000,
            )?,
        };
        measurements.validate_timed_topologies()?;
        Ok(measurements)
    }

    #[test]
    #[ignore = "requires a quiet SM120/170-SM/NVRTC-13.2 qualification host"]
    fn captured_tn_narrow_training_pipeline_qualification() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        let ctx = configured_ctx(&device).expect("configure deterministic Triad context");
        assert_eq!(ctx.compute_capability(), (12, 0));
        assert_eq!(ctx.kernels.multiprocessor_count(), 170);
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        assert_eq!(compiler.target.as_str(), "compute_120");
        assert_eq!(compiler.nvrtc_version, (13, 2));
        assert!(compiler.nvrtc_library_known);

        let production_topologies = run_trajectory_contract(&ctx, PipelineArm::Production)
            .expect("validate production trajectories");
        let direct_topologies = run_trajectory_contract(&ctx, PipelineArm::DirectControl)
            .expect("validate direct trajectories");
        let performance = run_performance_contract(&ctx).expect("measure paired pipeline windows");
        let promotion = performance.promotion();
        assert!(
            promotion.promoted(),
            "Task 8 promotion gates failed: {promotion:?}"
        );
        let observed_topologies = [production_topologies, direct_topologies].concat();
        append_evidence(&ctx, &performance, &observed_topologies)
            .expect("append validated Task 8 evidence");
    }
}
