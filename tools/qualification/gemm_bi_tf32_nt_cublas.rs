#[cfg(feature = "cuda")]
#[path = "../../tests/common/gpu_quiet.rs"]
mod gpu_quiet;

const DISCOVERY_WINDOWS: usize = 21;
const FINAL_WINDOWS: usize = 101;
const SCHEMA: &str = "MambaBiTf32NtCublasComparatorV2";
const DECISION: &str = "measurement_only_no_dispatcher_admission";
const COMPARISON_SCOPE: &str = "candidate_vs_cublas_within_same_path_only";
const GRAPH_MEASUREMENT_KIND: &str = "standalone_single_gemm_graph_replay";
const TIMING_CLOCK: &str = "cuda_event_device_elapsed";
const CANDIDATE_MEMORY_SAFETY_SCOPE: &str =
    "test_owned_guarded_direct_launch_exact_candidate_symbol";
const COMPARATOR_MEMORY_SAFETY_SCOPE: &str = "test_owned_guarded_cublas_gemm_ex_launch";
const CANDIDATE_SYMBOL: &str = "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3";
const CUBLAS_GEMM_SYMBOL: &str =
    "_ZN7cutlass7Kernel2I50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE";
const CUBLAS_SPLIT_K_REDUCER_PREFIX: &str = "_ZN8cublasLt19splitKreduce_kernel";
const CUDA_KERNEL_NODE_KIND: &str = "CU_GRAPH_NODE_TYPE_KERNEL";
const NT_SEMANTICS: &str = "C[M,K]=A[M,N]*transpose(B[K,N])";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    dims: (usize, usize, usize),
}

const CELLS: [Cell; 2] = [
    Cell {
        name: "m64_k384_n1536",
        dims: (64, 384, 1_536),
    },
    Cell {
        name: "m384_k64_n1536",
        dims: (384, 64, 1_536),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateManifest {
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic_shared_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NtCublasGeometry {
    trans_a: &'static str,
    trans_b: &'static str,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairOrder {
    CandidateThenCublas,
    CublasThenCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathKind {
    Eager,
    Graph,
}

impl PathKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }

    const fn timing_measurement_kind(self) -> &'static str {
        match self {
            Self::Eager => "eager_single_gemm_call",
            Self::Graph => GRAPH_MEASUREMENT_KIND,
        }
    }

    const fn graph_launch_submission_per_logical_gemm(self) -> usize {
        match self {
            Self::Eager => 0,
            Self::Graph => 1,
        }
    }

    const fn timing_clock(self) -> &'static str {
        let _ = self;
        TIMING_CLOCK
    }

    const fn host_api_enqueue_time_included(self) -> bool {
        let _ = self;
        false
    }
}

const fn candidate_memory_safety_scope() -> &'static str {
    CANDIDATE_MEMORY_SAFETY_SCOPE
}

const fn comparator_memory_safety_scope() -> &'static str {
    COMPARATOR_MEMORY_SAFETY_SCOPE
}

fn validate_storage_red_zones(
    values: &[f32],
    active_len: usize,
    guard_elements: usize,
    guard_bits: u32,
    label: &str,
) -> Result<usize, String> {
    let guarded_total = guard_elements
        .checked_mul(2)
        .ok_or_else(|| format!("{label} guard length overflows usize"))?;
    let expected_len = active_len
        .checked_add(guarded_total)
        .ok_or_else(|| format!("{label} guarded length overflows usize"))?;
    if values.len() != expected_len {
        return Err(format!("{label} guarded allocation length changed"));
    }
    for (index, value) in values[..guard_elements].iter().enumerate() {
        if value.to_bits() != guard_bits {
            return Err(format!("{label} prefix guard changed at {index}"));
        }
    }
    let suffix = guard_elements + active_len;
    for (index, value) in values[suffix..].iter().enumerate() {
        if value.to_bits() != guard_bits {
            return Err(format!("{label} suffix guard changed at {index}"));
        }
    }
    Ok(guard_elements * 2)
}

fn validate_storage_active_unchanged(
    values: &[f32],
    expected_active: &[f32],
    guard_elements: usize,
    label: &str,
) -> Result<usize, String> {
    let end = guard_elements
        .checked_add(expected_active.len())
        .ok_or_else(|| format!("{label} active range overflows usize"))?;
    let actual = values
        .get(guard_elements..end)
        .ok_or_else(|| format!("{label} active range exceeds guarded storage"))?;
    for (index, (actual, expected)) in actual.iter().zip(expected_active).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!("{label} active input changed at element {index}"));
        }
    }
    Ok(actual.len())
}

impl PairOrder {
    const fn name(self) -> &'static str {
        match self {
            Self::CandidateThenCublas => "candidate_then_cublas",
            Self::CublasThenCandidate => "cublas_then_candidate",
        }
    }
}

fn qualified_cells() -> [(usize, usize, usize); 2] {
    CELLS.map(|cell| cell.dims)
}

fn candidate_symbol() -> &'static str {
    CANDIDATE_SYMBOL
}

fn nt_semantics() -> &'static str {
    NT_SEMANTICS
}

fn candidate_manifest(cell: Cell) -> CandidateManifest {
    CandidateManifest {
        grid: (
            (cell.dims.1 as u32).div_ceil(32),
            (cell.dims.0 as u32).div_ceil(16),
            4,
        ),
        block: (128, 1, 1),
        dynamic_shared_bytes: 20_736,
    }
}

fn nt_cublas_geometry(cell: Cell) -> NtCublasGeometry {
    let (m, k, n) = cell.dims;
    NtCublasGeometry {
        trans_a: "CUBLAS_OP_T",
        trans_b: "CUBLAS_OP_N",
        m: i32::try_from(k).expect("K fits i32"),
        n: i32::try_from(m).expect("M fits i32"),
        k: i32::try_from(n).expect("N fits i32"),
        lda: i32::try_from(n).expect("N fits i32"),
        ldb: i32::try_from(n).expect("N fits i32"),
        ldc: i32::try_from(k).expect("K fits i32"),
    }
}

fn window_count(value: Option<&str>) -> Result<usize, String> {
    match value {
        None | Some("discovery") | Some("21") => Ok(DISCOVERY_WINDOWS),
        Some("final") | Some("101") => Ok(FINAL_WINDOWS),
        Some(value) => Err(format!(
            "MAMBA_RS_TF32_NT_CUBLAS_FINAL must be discovery/21 or final/101, received {value:?}"
        )),
    }
}

fn validate_sealed_cohort(
    compute_capability: (u32, u32),
    driver_api_version: i32,
    nvrtc_version: (i32, i32),
) -> Result<(), String> {
    if compute_capability == (12, 0) && driver_api_version == 13_020 && nvrtc_version == (13, 2) {
        Ok(())
    } else {
        Err(format!(
            "cuBLAS split-K graph contract requires sealed SM120/CUDA13.2, received SM{}.{}, driver API {}, NVRTC {}.{}",
            compute_capability.0,
            compute_capability.1,
            driver_api_version,
            nvrtc_version.0,
            nvrtc_version.1,
        ))
    }
}

fn paired_orders(windows: usize) -> Vec<&'static str> {
    let mut orders = Vec::with_capacity(windows.saturating_mul(2));
    orders.extend((0..windows).map(|_| PairOrder::CandidateThenCublas.name()));
    orders.extend((0..windows).map(|_| PairOrder::CublasThenCandidate.name()));
    orders
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphNodeFact {
    node_kind: String,
    kernel_symbol: Option<String>,
}

impl GraphNodeFact {
    fn kernel(symbol: &str) -> Self {
        Self {
            node_kind: CUDA_KERNEL_NODE_KIND.to_owned(),
            kernel_symbol: Some(symbol.to_owned()),
        }
    }

    fn description(&self, index: usize) -> String {
        match self.kernel_symbol.as_deref() {
            Some(symbol) => format!("{index}:{}:function={symbol}", self.node_kind),
            None => format!("{index}:{}", self.node_kind),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CublasGraphRoles {
    gemm: usize,
    split_k_reducer: usize,
}

fn cublas_graph_contract_failure(nodes: &[GraphNodeFact], edges: &[(usize, usize)]) -> String {
    let descriptions = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| node.description(index))
        .collect::<Vec<_>>();
    graph_contract_failure(&descriptions, edges)
}

fn validate_cublas_graph_contract(
    nodes: &[GraphNodeFact],
    edges: &[(usize, usize)],
) -> Result<CublasGraphRoles, String> {
    if nodes.len() != 2
        || nodes
            .iter()
            .any(|node| node.node_kind != CUDA_KERNEL_NODE_KIND)
    {
        return Err(cublas_graph_contract_failure(nodes, edges));
    }
    let gemm = nodes
        .iter()
        .position(|node| node.kernel_symbol.as_deref() == Some(CUBLAS_GEMM_SYMBOL))
        .ok_or_else(|| cublas_graph_contract_failure(nodes, edges))?;
    let split_k_reducer = nodes
        .iter()
        .position(|node| {
            node.kernel_symbol
                .as_deref()
                .is_some_and(|symbol| symbol.starts_with(CUBLAS_SPLIT_K_REDUCER_PREFIX))
        })
        .ok_or_else(|| cublas_graph_contract_failure(nodes, edges))?;
    if edges != [(gemm, split_k_reducer)] {
        return Err(cublas_graph_contract_failure(nodes, edges));
    }
    Ok(CublasGraphRoles {
        gemm,
        split_k_reducer,
    })
}

fn graph_contract_failure(nodes: &[String], edges: &[(usize, usize)]) -> String {
    let nodes = nodes.join(",");
    let edges = edges
        .iter()
        .map(|(from, to)| format!("{from}->{to}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "cuBLAS graph violates sealed SM120/CUDA13.2 contract: expected exactly ",
            "one CUTLASS TF32 GEMM kernel, one cublasLt split-K reducer kernel, and one ",
            "GEMM-to-reducer edge; nodes=[{}]; edges=[{}]"
        ),
        nodes, edges,
    )
}

struct CompletionRecord<'a> {
    records_before_completion: usize,
    records_total: usize,
    windows_per_order: usize,
    content_sha256_before_completion: &'a str,
}

impl CompletionRecord<'_> {
    fn render(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"completion\",",
                "\"records_before_completion\":{},\"records_total\":{},",
                "\"windows_per_order\":{},",
                "\"content_sha256_before_completion\":\"{}\",",
                "\"dataset_complete\":true,\"measurement_only\":true,",
                "\"sealed_cohort\":\"sm120_cuda13_2\",",
                "\"cublas_graph_contract\":\"cutlass_tf32_gemm_to_cublaslt_split_k_reducer\",",
                "\"comparison_scope\":\"{}\",",
                "\"graph_measurement_kind\":\"{}\",",
                "\"timing_clock\":\"{}\",",
                "\"host_api_enqueue_time_included\":false,",
                "\"graph_launch_submission_per_logical_gemm\":1,",
                "\"candidate_memory_safety_scope\":\"{}\",",
                "\"comparator_memory_safety_scope\":\"{}\",",
                "\"cross_path_comparison_valid\":false,",
                "\"full_training_graph_projection_valid\":false,",
                "\"runtime_autotuning\":false,\"dispatcher_admitted\":false,",
                "\"decision\":\"{}\"}}"
            ),
            SCHEMA,
            self.records_before_completion,
            self.records_total,
            self.windows_per_order,
            self.content_sha256_before_completion,
            COMPARISON_SCOPE,
            GRAPH_MEASUREMENT_KIND,
            TIMING_CLOCK,
            CANDIDATE_MEMORY_SAFETY_SCOPE,
            COMPARATOR_MEMORY_SAFETY_SCOPE,
            DECISION,
        )
    }
}

fn completion_record(
    records_before_completion: usize,
    windows_per_order: usize,
    content_sha256_before_completion: &str,
) -> CompletionRecord<'_> {
    CompletionRecord {
        records_before_completion,
        records_total: records_before_completion + 1,
        windows_per_order,
        content_sha256_before_completion,
    }
}

#[test]
fn comparator_contract_freezes_shapes_route_and_explicit_nt_semantics() {
    assert_eq!(qualified_cells(), [(64, 384, 1_536), (384, 64, 1_536)]);
    assert_eq!(
        candidate_symbol(),
        "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3"
    );
    assert_eq!(nt_semantics(), "C[M,K]=A[M,N]*transpose(B[K,N])");
    assert_eq!(candidate_manifest(CELLS[0]).grid, (12, 4, 4));
    assert_eq!(candidate_manifest(CELLS[1]).grid, (2, 24, 4));
    for cell in CELLS {
        let manifest = candidate_manifest(cell);
        assert_eq!(manifest.block, (128, 1, 1));
        assert_eq!(manifest.dynamic_shared_bytes, 20_736);
    }
}

#[test]
fn cublas_geometry_implements_row_major_nt_without_role_aliasing() {
    assert_eq!(
        nt_cublas_geometry(Cell {
            name: "fixture",
            dims: (5, 7, 11),
        }),
        NtCublasGeometry {
            trans_a: "CUBLAS_OP_T",
            trans_b: "CUBLAS_OP_N",
            m: 7,
            n: 5,
            k: 11,
            lda: 11,
            ldb: 11,
            ldc: 7,
        }
    );
}

#[test]
fn discovery_and_final_modes_are_exact_and_fail_closed() {
    assert_eq!(window_count(None).unwrap(), 21);
    assert_eq!(window_count(Some("discovery")).unwrap(), 21);
    assert_eq!(window_count(Some("21")).unwrap(), 21);
    assert_eq!(window_count(Some("final")).unwrap(), 101);
    assert_eq!(window_count(Some("101")).unwrap(), 101);
    for invalid in ["", "1", "true", "20", "100", "102"] {
        assert!(window_count(Some(invalid)).is_err(), "accepted {invalid:?}");
    }
}

#[test]
fn sealed_cublas_topology_cohort_requires_sm120_cuda_13_2() {
    validate_sealed_cohort((12, 0), 13_020, (13, 2)).unwrap();
    for invalid in [
        ((9, 0), 13_020, (13, 2)),
        ((12, 1), 13_020, (13, 2)),
        ((12, 0), 13_010, (13, 2)),
        ((12, 0), 13_020, (13, 1)),
    ] {
        assert!(
            validate_sealed_cohort(invalid.0, invalid.1, invalid.2).is_err(),
            "accepted {invalid:?}"
        );
    }
}

#[test]
fn paired_schedule_runs_every_window_in_both_orders() {
    assert_eq!(
        paired_orders(3),
        ["candidate_then_cublas"; 3]
            .into_iter()
            .chain(["cublas_then_candidate"; 3])
            .collect::<Vec<_>>()
    );
}

#[test]
fn graph_timing_interpretation_is_one_standalone_replay_per_gemm() {
    assert_eq!(PathKind::Eager.name(), "eager");
    assert_eq!(PathKind::Graph.name(), "graph");
    assert_eq!(
        PathKind::Eager.timing_measurement_kind(),
        "eager_single_gemm_call"
    );
    assert_eq!(
        PathKind::Eager.graph_launch_submission_per_logical_gemm(),
        0
    );
    assert_eq!(PathKind::Eager.timing_clock(), "cuda_event_device_elapsed");
    assert!(!PathKind::Eager.host_api_enqueue_time_included());
    assert_eq!(
        PathKind::Graph.timing_measurement_kind(),
        "standalone_single_gemm_graph_replay"
    );
    assert_eq!(
        PathKind::Graph.graph_launch_submission_per_logical_gemm(),
        1
    );
    assert_eq!(PathKind::Graph.timing_clock(), "cuda_event_device_elapsed");
    assert!(!PathKind::Graph.host_api_enqueue_time_included());
}

#[test]
fn memory_safety_contract_names_each_measured_arm_without_aggregate_aliases() {
    assert_eq!(
        candidate_memory_safety_scope(),
        "test_owned_guarded_direct_launch_exact_candidate_symbol"
    );
    assert_eq!(
        comparator_memory_safety_scope(),
        "test_owned_guarded_cublas_gemm_ex_launch"
    );
}

#[test]
fn guarded_storage_validators_separate_red_zones_from_active_input_checks() {
    let guard = 0x7fc1_5a5a;
    let expected_active = [1.0_f32, -2.0, 3.5];
    let mut storage = vec![f32::from_bits(guard); 7];
    storage[2..5].copy_from_slice(&expected_active);
    assert_eq!(
        validate_storage_red_zones(&storage, expected_active.len(), 2, guard, "fixture").unwrap(),
        4
    );
    assert_eq!(
        validate_storage_active_unchanged(&storage, &expected_active, 2, "fixture").unwrap(),
        3
    );

    storage[0] = 0.0;
    assert!(
        validate_storage_red_zones(&storage, expected_active.len(), 2, guard, "fixture").is_err()
    );
    storage[0] = f32::from_bits(guard);
    storage[3] = 0.0;
    assert!(validate_storage_active_unchanged(&storage, &expected_active, 2, "fixture").is_err());
}

#[test]
fn graph_contract_failure_reports_node_types_functions_and_edges() {
    let nodes = [
        "0:CU_GRAPH_NODE_TYPE_KERNEL:function=main_gemm".to_owned(),
        "1:CU_GRAPH_NODE_TYPE_MEMSET".to_owned(),
    ];
    assert_eq!(
        graph_contract_failure(&nodes, &[(1, 0)]),
        concat!(
            "cuBLAS graph violates sealed SM120/CUDA13.2 contract: expected exactly ",
            "one CUTLASS TF32 GEMM kernel, one cublasLt split-K reducer kernel, and one ",
            "GEMM-to-reducer edge; ",
            "nodes=[0:CU_GRAPH_NODE_TYPE_KERNEL:function=main_gemm,",
            "1:CU_GRAPH_NODE_TYPE_MEMSET]; edges=[1->0]"
        )
    );
}

#[test]
fn cublas_graph_contract_accepts_only_gemm_then_split_k_reducer_roles() {
    let nodes = [
        GraphNodeFact::kernel(
            "_ZN7cutlass7Kernel2I50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE",
        ),
        GraphNodeFact::kernel("_ZN8cublasLt19splitKreduce_kernel_fixture"),
    ];
    assert_eq!(
        validate_cublas_graph_contract(&nodes, &[(0, 1)]).unwrap(),
        CublasGraphRoles {
            gemm: 0,
            split_k_reducer: 1,
        }
    );
}

#[test]
fn cublas_graph_contract_rejects_reversed_edge_with_full_diagnostics() {
    let nodes = [
        GraphNodeFact::kernel(CUBLAS_GEMM_SYMBOL),
        GraphNodeFact::kernel("_ZN8cublasLt19splitKreduce_kernel_fixture"),
    ];
    assert_eq!(
        validate_cublas_graph_contract(&nodes, &[(1, 0)]).unwrap_err(),
        concat!(
            "cuBLAS graph violates sealed SM120/CUDA13.2 contract: expected exactly ",
            "one CUTLASS TF32 GEMM kernel, one cublasLt split-K reducer kernel, and one ",
            "GEMM-to-reducer edge; nodes=[",
            "0:CU_GRAPH_NODE_TYPE_KERNEL:function=",
            "_ZN7cutlass7Kernel2I50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE,",
            "1:CU_GRAPH_NODE_TYPE_KERNEL:function=_ZN8cublasLt19splitKreduce_kernel_fixture",
            "]; edges=[1->0]"
        )
    );
}

#[test]
fn cublas_graph_contract_rejects_extra_or_non_kernel_nodes() {
    let gemm = GraphNodeFact::kernel(CUBLAS_GEMM_SYMBOL);
    let reducer = GraphNodeFact::kernel("_ZN8cublasLt19splitKreduce_kernel_fixture");
    let extra = [
        gemm.clone(),
        reducer.clone(),
        GraphNodeFact::kernel("unexpected_helper"),
    ];
    assert!(validate_cublas_graph_contract(&extra, &[(0, 1)]).is_err());

    let spoofed_reducer = GraphNodeFact {
        node_kind: "CU_GRAPH_NODE_TYPE_MEMSET".to_owned(),
        kernel_symbol: reducer.kernel_symbol,
    };
    assert!(validate_cublas_graph_contract(&[gemm, spoofed_reducer], &[(0, 1)]).is_err());
}

#[test]
fn completion_is_measurement_only_and_cannot_claim_admission() {
    let record = completion_record(4, 21, "abc123");
    assert_eq!(
        record.render(),
        concat!(
            "{\"schema\":\"MambaBiTf32NtCublasComparatorV2\",",
            "\"record_type\":\"completion\",\"records_before_completion\":4,",
            "\"records_total\":5,\"windows_per_order\":21,",
            "\"content_sha256_before_completion\":\"abc123\",",
            "\"dataset_complete\":true,\"measurement_only\":true,",
            "\"sealed_cohort\":\"sm120_cuda13_2\",",
            "\"cublas_graph_contract\":\"cutlass_tf32_gemm_to_cublaslt_split_k_reducer\",",
            "\"comparison_scope\":\"candidate_vs_cublas_within_same_path_only\",",
            "\"graph_measurement_kind\":\"standalone_single_gemm_graph_replay\",",
            "\"timing_clock\":\"cuda_event_device_elapsed\",",
            "\"host_api_enqueue_time_included\":false,",
            "\"graph_launch_submission_per_logical_gemm\":1,",
            "\"candidate_memory_safety_scope\":",
            "\"test_owned_guarded_direct_launch_exact_candidate_symbol\",",
            "\"comparator_memory_safety_scope\":",
            "\"test_owned_guarded_cublas_gemm_ex_launch\",",
            "\"cross_path_comparison_valid\":false,",
            "\"full_training_graph_projection_valid\":false,",
            "\"runtime_autotuning\":false,\"dispatcher_admitted\":false,",
            "\"decision\":\"measurement_only_no_dispatcher_admission\"}"
        )
    );
}

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::ffi::{CStr, c_void};
    use std::fmt::Write as _;
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write as _};
    use std::path::PathBuf;
    use std::sync::OnceLock;

    use super::*;
    use cudarc::driver::{CudaFunction, CudaGraph, DeviceRepr, LaunchConfig, PushKernelArg, sys};
    use gpu_quiet::QuietGpu;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedPhysicalLaunch,
        Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
        presize_physical_qualification_suite, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp,
    };
    use sha2::{Digest as _, Sha256};

    const OUTPUT_ENV: &str = "MAMBA_RS_TF32_NT_CUBLAS_JSONL";
    const FINAL_ENV: &str = "MAMBA_RS_TF32_NT_CUBLAS_FINAL";
    const SALT: u64 = 0x6e74_7038;
    const DETERMINISM_REPEATS: usize = 10;
    const GUARD_ELEMENTS: usize = 32;
    const GUARD_BITS: u32 = 0x7fc1_5a5a;
    const OUTPUT_CANARY_BITS: u32 = 0x7fc2_5a5a;
    const WARMUPS: usize = 64;
    const PILOT_ITERATIONS: usize = 16;
    const TARGET_WINDOW_US: f64 = 5_000.0;
    const MAX_WINDOW_ITERATIONS: usize = 4_096;
    const RECORDS_BEFORE_COMPLETION: usize = CELLS.len() * 2;
    const TEST_SOURCE: &str = include_str!("gemm_bi_tf32_nt_cublas.rs");
    const CUDA_SOURCE: &str = include_str!("../../kernels/gemm_bi_triad/sm80/mma.cu");

    const CANDIDATE_ROUTE: Tf32PhysicalRoute =
        Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S3,
        });

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Candidate,
        Cublas,
    }

    #[derive(Clone)]
    struct GraphKernelSnapshot {
        kernel_symbol: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        dynamic_shared_bytes: u32,
    }

    struct ObservedGraphNode {
        fact: GraphNodeFact,
        kernel: Option<GraphKernelSnapshot>,
    }

    #[derive(Clone)]
    struct GraphSnapshot {
        node_count: usize,
        edge_count: usize,
        gemm: GraphKernelSnapshot,
        split_k_reducer: GraphKernelSnapshot,
    }

    struct GuardedAllocation {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        active_len: usize,
    }

    impl GuardedAllocation {
        fn new(ctx: &GpuCtx, active: &[f32], label: &str) -> Result<Self, String> {
            let len = active
                .len()
                .checked_add(GUARD_ELEMENTS * 2)
                .ok_or_else(|| format!("{label} guarded extent overflows usize"))?;
            let mut expected = vec![f32::from_bits(GUARD_BITS); len];
            expected[GUARD_ELEMENTS..GUARD_ELEMENTS + active.len()].copy_from_slice(active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(&ctx.stream, &expected)?,
                expected,
                active_len: active.len(),
            })
        }

        fn active_ptr(&self, ctx: &GpuCtx) -> u64 {
            self.buffer.raw_ptr_at(&ctx.stream, GUARD_ELEMENTS)
        }

        fn upload_expected(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.buffer.upload(&ctx.stream, &self.expected)
        }

        fn download(&self, ctx: &GpuCtx) -> Result<Vec<f32>, String> {
            self.buffer.to_cpu(&ctx.stream)
        }

        fn validate_red_zones(&self, values: &[f32], label: &str) -> Result<usize, String> {
            validate_storage_red_zones(values, self.active_len, GUARD_ELEMENTS, GUARD_BITS, label)
        }

        fn expected_active(&self) -> &[f32] {
            &self.expected[GUARD_ELEMENTS..GUARD_ELEMENTS + self.active_len]
        }

        fn validate_input(&self, ctx: &GpuCtx, label: &str) -> Result<(usize, usize), String> {
            let values = self.download(ctx)?;
            let guards = self.validate_red_zones(&values, label)?;
            let active = validate_storage_active_unchanged(
                &values,
                self.expected_active(),
                GUARD_ELEMENTS,
                label,
            )?;
            Ok((guards, active))
        }
    }

    struct ComparatorBuffers {
        arm: &'static str,
        output: GuardedAllocation,
        a: GuardedAllocation,
        b: GuardedAllocation,
        output_elements: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct GuardCheck {
        guard_allocations: usize,
        guard_elements: usize,
        unchanged_input_elements: usize,
        output_prefill_elements: usize,
    }

    #[derive(Clone, Copy, Default)]
    struct MemorySafetyEvidence {
        validations: usize,
        guard_allocations_checked: usize,
        guard_elements_checked: usize,
        unchanged_input_elements_checked: usize,
        output_prefill_elements_checked: usize,
    }

    impl MemorySafetyEvidence {
        fn record(&mut self, check: GuardCheck) {
            self.validations += 1;
            self.guard_allocations_checked += check.guard_allocations;
            self.guard_elements_checked += check.guard_elements;
            self.unchanged_input_elements_checked += check.unchanged_input_elements;
            self.output_prefill_elements_checked += check.output_prefill_elements;
        }
    }

    impl ComparatorBuffers {
        fn new(
            ctx: &GpuCtx,
            cell: Cell,
            arm: &'static str,
            output: Vec<f32>,
        ) -> Result<Self, String> {
            let (m, k, n) = cell.dims;
            let a_elements = checked_extent(m, n, "comparator M*N")?;
            let b_elements = checked_extent(k, n, "comparator K*N")?;
            let output_elements = checked_extent(m, k, "comparator M*K")?;
            let a = seeded_values(a_elements, SALT ^ 0x2d);
            let b = seeded_values(b_elements, SALT ^ 0x67);
            if output.len() != output_elements {
                return Err(format!("{arm} output prefill length changed"));
            }
            Ok(Self {
                arm,
                output: GuardedAllocation::new(ctx, &output, &format!("{arm} output"))?,
                a: GuardedAllocation::new(ctx, &a, &format!("{arm} A"))?,
                b: GuardedAllocation::new(ctx, &b, &format!("{arm} B"))?,
                output_elements,
            })
        }

        fn new_cublas(ctx: &GpuCtx, cell: Cell) -> Result<Self, String> {
            let output_elements = checked_extent(cell.dims.0, cell.dims.1, "cuBLAS M*K")?;
            Self::new(
                ctx,
                cell,
                "cuBLAS",
                vec![f32::from_bits(OUTPUT_CANARY_BITS); output_elements],
            )
        }

        fn new_candidate(ctx: &GpuCtx, cell: Cell) -> Result<Self, String> {
            let output_elements = checked_extent(cell.dims.0, cell.dims.1, "candidate M*K")?;
            Self::new(
                ctx,
                cell,
                "candidate",
                seeded_values(output_elements, SALT ^ 0x91),
            )
        }

        fn reset_output(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.upload_expected(ctx)
        }

        fn reset_all(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            self.output.upload_expected(ctx)?;
            self.a.upload_expected(ctx)?;
            self.b.upload_expected(ctx)
        }

        fn output_bits(&self, ctx: &GpuCtx) -> Result<(Vec<u32>, GuardCheck), String> {
            let output = self.output.download(ctx)?;
            let output_label = format!("{} output", self.arm);
            let a_label = format!("{} A", self.arm);
            let b_label = format!("{} B", self.arm);
            let mut guard_elements = self.output.validate_red_zones(&output, &output_label)?;
            let start = GUARD_ELEMENTS;
            let end = start + self.output_elements;
            let active = &output[start..end];
            for (index, (value, prefill)) in
                active.iter().zip(self.output.expected_active()).enumerate()
            {
                if value.to_bits() == prefill.to_bits() {
                    return Err(format!(
                        "{} output prefill remained at element {index}",
                        self.arm
                    ));
                }
            }
            let (a_guards, a_active) = self.a.validate_input(ctx, &a_label)?;
            let (b_guards, b_active) = self.b.validate_input(ctx, &b_label)?;
            guard_elements += a_guards + b_guards;
            Ok((
                active.iter().map(|value| value.to_bits()).collect(),
                GuardCheck {
                    guard_allocations: 3,
                    guard_elements,
                    unchanged_input_elements: a_active + b_active,
                    output_prefill_elements: active.len(),
                },
            ))
        }
    }

    struct CorrectnessEvidence {
        candidate_bits: Vec<u32>,
        cublas_bits: Vec<u32>,
        candidate_digest: [u8; 32],
        cublas_digest: [u8; 32],
        oracle_digest: [u8; 32],
        candidate_memory: MemorySafetyEvidence,
        comparator_memory: MemorySafetyEvidence,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct CandidateKernelParams {
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for CandidateKernelParams {}

    struct CandidateGuardedLaunch {
        graph: CudaGraph,
        function: CudaFunction,
        buffers: ComparatorBuffers,
        partial: GpuBuffer,
        counters: GpuBuffer,
        config: LaunchConfig,
        params: CandidateKernelParams,
        ctx: GpuCtx,
    }

    fn candidate_sm80_source() -> String {
        [
            include_str!("../../kernels/_typed_prelude.cuh"),
            include_str!("../../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../../kernels/gemm_bi_triad/mma16.cuh"),
            include_str!("../../kernels/gemm_bi_triad/sm80/mma.cu"),
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
    }

    fn compile_guarded_candidate(ctx: &GpuCtx) -> Result<CudaFunction, String> {
        static CANDIDATE_PTX: OnceLock<Result<String, String>> = OnceLock::new();
        let ptx = CANDIDATE_PTX
            .get_or_init(|| {
                let options = cudarc::nvrtc::CompileOptions {
                    arch: Some("compute_120"),
                    options: vec![
                        "--fmad=true".to_owned(),
                        "--extra-device-vectorization".to_owned(),
                        "-DNDEBUG".to_owned(),
                        "-DGEMM_BI_GROUP_M=16".to_owned(),
                        "-DMAMBA_RS_STATE_CAP=256".to_owned(),
                        "--frandom-seed=1295072049".to_owned(),
                    ],
                    include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                    ..Default::default()
                };
                cudarc::nvrtc::compile_ptx_with_opts(candidate_sm80_source(), options)
                    .map(|image| image.to_src())
                    .map_err(|error| format!("compile guarded candidate source: {error:?}"))
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let module = ctx
            .stream
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx.clone()))
            .map_err(|error| format!("load guarded candidate module: {error:?}"))?;
        module
            .load_function(CANDIDATE_SYMBOL)
            .map_err(|error| format!("load guarded candidate symbol: {error:?}"))
    }

    fn launch_guarded_candidate(
        ctx: &GpuCtx,
        function: &CudaFunction,
        buffers: &ComparatorBuffers,
        partial: &GpuBuffer,
        counters: &GpuBuffer,
        config: LaunchConfig,
        params: CandidateKernelParams,
    ) -> Result<(), String> {
        let output = buffers.output.active_ptr(ctx);
        let partial = partial.raw_ptr(&ctx.stream);
        let counters = counters.raw_ptr(&ctx.stream);
        let a = buffers.a.active_ptr(ctx);
        let b = buffers.b.active_ptr(ctx);
        let bias = 0_u64;
        let mut builder = ctx.stream.launch_builder(function);
        builder.arg(&output);
        builder.arg(&partial);
        builder.arg(&counters);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&bias);
        builder.arg(&params);
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|error| format!("launch guarded candidate: {error:?}"))
    }

    impl CandidateGuardedLaunch {
        fn new(device: &GpuDevice, cell: Cell) -> Result<Self, String> {
            let ctx = GpuCtx::new(device)?;
            let function = compile_guarded_candidate(&ctx)?;
            let buffers = ComparatorBuffers::new_candidate(&ctx, cell)?;
            let output_elements = checked_extent(cell.dims.0, cell.dims.1, "candidate M*K")?;
            let partial_elements = output_elements
                .checked_mul(4)
                .ok_or_else(|| "candidate partial extent overflows usize".to_string())?;
            let manifest = candidate_manifest(cell);
            let counter_elements = (manifest.grid.0 as usize)
                .checked_mul(manifest.grid.1 as usize)
                .ok_or_else(|| "candidate counter extent overflows usize".to_string())?;
            let partial = GpuBuffer::zeros(&ctx.stream, partial_elements)?;
            let counters = GpuBuffer::zeros(&ctx.stream, counter_elements)?;
            let config = LaunchConfig {
                grid_dim: manifest.grid,
                block_dim: manifest.block,
                shared_mem_bytes: manifest.dynamic_shared_bytes,
            };
            let params = CandidateKernelParams {
                alpha: 1.0,
                beta: 0.0,
                m: i32::try_from(cell.dims.0).map_err(|_| "candidate M exceeds i32")?,
                k: i32::try_from(cell.dims.1).map_err(|_| "candidate K exceeds i32")?,
                n: i32::try_from(cell.dims.2).map_err(|_| "candidate N exceeds i32")?,
                lda: i32::try_from(cell.dims.2).map_err(|_| "candidate lda exceeds i32")?,
                ldb: i32::try_from(cell.dims.2).map_err(|_| "candidate ldb exceeds i32")?,
                ldc: i32::try_from(cell.dims.1).map_err(|_| "candidate ldc exceeds i32")?,
            };
            launch_guarded_candidate(
                &ctx, &function, &buffers, &partial, &counters, config, params,
            )?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize guarded candidate warmup: {error:?}"))?;
            let graph = unsafe {
                capture_into_graph(&ctx.stream, || {
                    launch_guarded_candidate(
                        &ctx, &function, &buffers, &partial, &counters, config, params,
                    )
                })
            }?;
            Ok(Self {
                graph,
                function,
                buffers,
                partial,
                counters,
                config,
                params,
                ctx,
            })
        }

        fn reset_run_validate(&mut self, path: PathKind) -> Result<(Vec<u32>, GuardCheck), String> {
            let ctx = &self.ctx;
            self.buffers.reset_all(ctx)?;
            match path {
                PathKind::Eager => launch_guarded_candidate(
                    ctx,
                    &self.function,
                    &self.buffers,
                    &self.partial,
                    &self.counters,
                    self.config,
                    self.params,
                )?,
                PathKind::Graph => self
                    .graph
                    .launch()
                    .map_err(|error| format!("launch guarded candidate graph: {error:?}"))?,
            }
            ctx.stream.synchronize().map_err(|error| {
                format!("synchronize guarded candidate {}: {error:?}", path.name())
            })?;
            self.buffers.output_bits(ctx)
        }
    }

    fn checked_extent(left: usize, right: usize, label: &str) -> Result<usize, String> {
        left.checked_mul(right)
            .ok_or_else(|| format!("{label} extent overflows usize"))
    }

    fn seeded_values(len: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    fn tf32_rne_bits(value: f32) -> u32 {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        if exponent == 0x7f80_0000 {
            return bits;
        }
        let least_kept_bit = (bits >> 13) & 1;
        bits.wrapping_add(0x0fff + least_kept_bit) & 0xffff_e000
    }

    fn tf32_rna_bits(value: f32) -> u32 {
        let bits = value.to_bits();
        let exponent = bits & 0x7f80_0000;
        if exponent == 0x7f80_0000 {
            return bits;
        }
        bits.wrapping_add(0x1000) & 0xffff_e000
    }

    fn require_exact_tf32_inputs(values: &[f32], label: &str) -> Result<(), String> {
        for (index, value) in values.iter().copied().enumerate() {
            let bits = value.to_bits();
            if tf32_rne_bits(value) != bits || tf32_rna_bits(value) != bits {
                return Err(format!(
                    "{label} element {index} is not exact under both TF32 conversions"
                ));
            }
        }
        Ok(())
    }

    fn cpu_tf32_nt_oracle(cell: Cell) -> Result<Vec<f64>, String> {
        let (m, k, n) = cell.dims;
        let a = seeded_values(checked_extent(m, n, "oracle M*N")?, SALT ^ 0x2d);
        let b = seeded_values(checked_extent(k, n, "oracle K*N")?, SALT ^ 0x67);
        require_exact_tf32_inputs(&a, "oracle A")?;
        require_exact_tf32_inputs(&b, "oracle B")?;
        let mut expected = Vec::with_capacity(checked_extent(m, k, "oracle M*K")?);
        for row in 0..m {
            for column in 0..k {
                let mut sum = 0.0_f64;
                for inner in 0..n {
                    sum += f64::from(a[row * n + inner]) * f64::from(b[column * n + inner]);
                }
                expected.push(sum);
            }
        }
        Ok(expected)
    }

    fn check_numeric(actual: &[u32], expected: &[f64], label: &str) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!(
                "{label} output length {} differs from oracle {}",
                actual.len(),
                expected.len()
            ));
        }
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(f32::from_bits(actual));
            let tolerance = 0.0025 * (1.0 + expected.abs());
            if !actual.is_finite() || (actual - expected).abs() > tolerance {
                return Err(format!(
                    "{label} output {index} was {actual}, oracle {expected}, tolerance {tolerance}"
                ));
            }
        }
        Ok(())
    }

    fn candidate_request(cell: Cell) -> PhysicalQualificationRequest {
        PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            cell.dims,
            PhysicalQualificationRoute::Tf32Forced(CANDIDATE_ROUTE),
        )
    }

    fn assert_candidate_manifest(
        cell: Cell,
        launch: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let evidence = launch.evidence();
        let [node] = evidence.nodes() else {
            return Err(format!(
                "{} candidate requires exactly one node, observed {:?}",
                cell.name,
                evidence.nodes()
            ));
        };
        let expected = candidate_manifest(cell);
        let expected_strides = (cell.dims.2, cell.dims.2, cell.dims.1);
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || evidence.single_launch_symbol() != Some(CANDIDATE_SYMBOL)
            || evidence.single_launch_tile() != Some((16, 32))
            || evidence.uniform_module_kind() != Some(ModuleKind::TriadSm80)
            || evidence.uniform_execution_dtype() != Some(PolicyDtype::F32)
            || node.kind != PhysicalLaunchKind::Gemm
            || node.symbol != CANDIDATE_SYMBOL
            || node.module_kind != ModuleKind::TriadSm80
            || node.logical_op != ResolvedGemmOp::Nt
            || node.logical_dtype != PolicyDtype::F32
            || node.execution_dtype != PolicyDtype::F32
            || node.shape != cell.dims
            || node.strides != expected_strides
            || node.tile != Some((16, 32))
            || node.launch.grid_dim != expected.grid
            || node.launch.block_dim != expected.block
            || node.launch.shared_mem_bytes != expected.dynamic_shared_bytes
        {
            return Err(format!(
                "{} candidate physical manifest changed: {:?}",
                cell.name,
                evidence.nodes()
            ));
        }
        Ok(())
    }

    fn launch_cublas(ctx: &GpuCtx, cell: Cell, buffers: &ComparatorBuffers) -> Result<(), String> {
        let geometry = nt_cublas_geometry(cell);
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *ctx.blas.handle(),
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                geometry.m,
                geometry.n,
                geometry.k,
                &alpha as *const f32 as *const c_void,
                buffers.b.active_ptr(ctx) as *const c_void,
                cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
                geometry.lda,
                buffers.a.active_ptr(ctx) as *const c_void,
                cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
                geometry.ldb,
                &beta as *const f32 as *const c_void,
                buffers.output.active_ptr(ctx) as *mut c_void,
                cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
                geometry.ldc,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|error| format!("{} cuBLAS FAST_TF32 NT failed: {error:?}", cell.name))?;
        }
        Ok(())
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed: {result:?}"))
        }
    }

    fn observe_graph_node(node: sys::CUgraphNode) -> Result<ObservedGraphNode, String> {
        let mut node_type = std::mem::MaybeUninit::uninit();
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) },
            "query cuBLAS graph node type",
        )?;
        let node_type = unsafe { node_type.assume_init() };
        let is_kernel = node_type == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        if !is_kernel {
            return Ok(ObservedGraphNode {
                fact: GraphNodeFact {
                    node_kind: format!("{node_type:?}"),
                    kernel_symbol: None,
                },
                kernel: None,
            });
        }
        let mut params = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "query cuBLAS diagnostic kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "query cuBLAS diagnostic kernel name",
        )?;
        if name.is_null() {
            return Err("cuBLAS diagnostic kernel name is null".into());
        }
        let function = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("cuBLAS diagnostic kernel name is not UTF-8: {error}"))?
            .to_owned();
        Ok(ObservedGraphNode {
            fact: GraphNodeFact {
                node_kind: format!("{node_type:?}"),
                kernel_symbol: Some(function.clone()),
            },
            kernel: Some(GraphKernelSnapshot {
                kernel_symbol: function,
                grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
                block: (params.blockDimX, params.blockDimY, params.blockDimZ),
                dynamic_shared_bytes: params.sharedMemBytes,
            }),
        })
    }

    fn validate_kernel_snapshot(label: &str, snapshot: &GraphKernelSnapshot) -> Result<(), String> {
        if snapshot.grid.0 == 0
            || snapshot.grid.1 == 0
            || snapshot.grid.2 == 0
            || snapshot.block.0 == 0
            || snapshot.block.1 == 0
            || snapshot.block.2 == 0
        {
            Err(format!("cuBLAS {label} kernel has a zero launch dimension"))
        } else {
            Ok(())
        }
    }

    fn graph_edge_indices(
        graph: &CudaGraph,
        nodes: &[sys::CUgraphNode],
    ) -> Result<Vec<(usize, usize)>, String> {
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
            "query cuBLAS graph edge count",
        )?;
        let mut from = vec![std::ptr::null_mut(); edge_count];
        let mut to = vec![std::ptr::null_mut(); edge_count];
        let mut edge_data = Vec::with_capacity(edge_count);
        edge_data.resize_with(edge_count, || unsafe { std::mem::zeroed() });
        if edge_count != 0 {
            cuda_ok(
                unsafe {
                    sys::cuGraphGetEdges_v2(
                        graph.cu_graph(),
                        from.as_mut_ptr(),
                        to.as_mut_ptr(),
                        edge_data.as_mut_ptr(),
                        &mut edge_count,
                    )
                },
                "query cuBLAS graph edges",
            )?;
        }
        from.into_iter()
            .zip(to)
            .map(|(from, to)| {
                let from = nodes
                    .iter()
                    .position(|node| *node == from)
                    .ok_or_else(|| "cuBLAS graph edge source is not in node list".to_string())?;
                let to = nodes
                    .iter()
                    .position(|node| *node == to)
                    .ok_or_else(|| "cuBLAS graph edge target is not in node list".to_string())?;
                Ok((from, to))
            })
            .collect()
    }

    fn cublas_graph_snapshot(graph: &CudaGraph) -> Result<GraphSnapshot, String> {
        let mut node_count = 0usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut node_count)
            },
            "query cuBLAS graph node count",
        )?;
        let mut nodes = vec![std::ptr::null_mut(); node_count];
        if node_count != 0 {
            cuda_ok(
                unsafe {
                    sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut node_count)
                },
                "query cuBLAS graph nodes",
            )?;
        }
        let observed = nodes
            .iter()
            .copied()
            .map(observe_graph_node)
            .collect::<Result<Vec<_>, _>>()?;
        let edges = graph_edge_indices(graph, &nodes)?;
        let facts = observed
            .iter()
            .map(|node| node.fact.clone())
            .collect::<Vec<_>>();
        let roles = validate_cublas_graph_contract(&facts, &edges)?;
        let gemm = observed[roles.gemm]
            .kernel
            .clone()
            .ok_or_else(|| cublas_graph_contract_failure(&facts, &edges))?;
        let split_k_reducer = observed[roles.split_k_reducer]
            .kernel
            .clone()
            .ok_or_else(|| cublas_graph_contract_failure(&facts, &edges))?;
        validate_kernel_snapshot("GEMM", &gemm)?;
        validate_kernel_snapshot("split-K reducer", &split_k_reducer)?;
        let snapshot = GraphSnapshot {
            node_count,
            edge_count: edges.len(),
            gemm,
            split_k_reducer,
        };
        Ok(snapshot)
    }

    fn capture_cublas_graph(
        ctx: &GpuCtx,
        cell: Cell,
        buffers: &mut ComparatorBuffers,
    ) -> Result<(CudaGraph, GraphSnapshot), String> {
        buffers.reset_output(ctx)?;
        launch_cublas(ctx, cell, buffers)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize cuBLAS warm launch: {error:?}"))?;
        buffers.reset_output(ctx)?;
        let graph =
            unsafe { capture_into_graph(&ctx.stream, || launch_cublas(ctx, cell, buffers)) }?;
        let snapshot = cublas_graph_snapshot(&graph)?;
        Ok((graph, snapshot))
    }

    fn run_candidate_once(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        path: PathKind,
    ) -> Result<(), String> {
        match path {
            PathKind::Eager => launch.measure_eager_window_ms(ctx, 1)?,
            PathKind::Graph => launch.measure_graph_window_ms(ctx, 1)?,
        };
        Ok(())
    }

    fn run_cublas_once(
        ctx: &GpuCtx,
        cell: Cell,
        buffers: &ComparatorBuffers,
        graph: &CudaGraph,
        path: PathKind,
    ) -> Result<(), String> {
        match path {
            PathKind::Eager => launch_cublas(ctx, cell, buffers)?,
            PathKind::Graph => graph
                .launch()
                .map_err(|error| format!("{} cuBLAS graph launch failed: {error:?}", cell.name))?,
        }
        ctx.stream.synchronize().map_err(|error| {
            format!(
                "synchronize {} cuBLAS {}: {error:?}",
                cell.name,
                path.name()
            )
        })
    }

    fn require_same_bits(
        reference: &mut Option<Vec<u32>>,
        actual: Vec<u32>,
        label: &str,
    ) -> Result<(), String> {
        if reference
            .as_ref()
            .is_some_and(|expected| expected != &actual)
        {
            return Err(format!("{label} changed output bits"));
        }
        reference.get_or_insert(actual);
        Ok(())
    }

    fn correctness_gate(
        cell: Cell,
        candidate_ctx: &GpuCtx,
        candidate: &mut QualifiedPhysicalLaunch<'_>,
        guarded_candidate: &mut CandidateGuardedLaunch,
        cublas_ctx: &GpuCtx,
        buffers: &mut ComparatorBuffers,
        graph: &CudaGraph,
    ) -> Result<CorrectnessEvidence, String> {
        let oracle = cpu_tf32_nt_oracle(cell)?;
        let oracle_bits = oracle
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>();
        let candidate_prefill = seeded_values(oracle.len(), SALT ^ 0x91)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let mut candidate_reference = None;
        let mut cublas_reference = None;
        let mut candidate_memory = MemorySafetyEvidence::default();
        let mut comparator_memory = MemorySafetyEvidence::default();

        for path in [PathKind::Eager, PathKind::Graph] {
            for repeat in 0..DETERMINISM_REPEATS {
                candidate.seed_f32_operands(candidate_ctx, SALT)?;
                run_candidate_once(candidate_ctx, candidate, path)?;
                let actual = candidate.f32_output_bits(candidate_ctx)?;
                check_numeric(
                    &actual,
                    &oracle,
                    &format!("{} candidate {} repeat {repeat}", cell.name, path.name()),
                )?;
                for (index, (actual, prefill)) in actual.iter().zip(&candidate_prefill).enumerate()
                {
                    if actual == prefill {
                        return Err(format!(
                            "{} candidate output prefill remained at element {index}",
                            cell.name
                        ));
                    }
                }
                let (guarded_bits, guarded_check) = guarded_candidate.reset_run_validate(path)?;
                check_numeric(
                    &guarded_bits,
                    &oracle,
                    &format!(
                        "{} guarded candidate {} repeat {repeat}",
                        cell.name,
                        path.name()
                    ),
                )?;
                if guarded_bits != actual {
                    return Err(format!(
                        "{} guarded direct candidate differs from qualified candidate on {} repeat {repeat}",
                        cell.name,
                        path.name()
                    ));
                }
                candidate_memory.record(guarded_check);
                require_same_bits(
                    &mut candidate_reference,
                    actual,
                    &format!("{} candidate {} repeat {repeat}", cell.name, path.name()),
                )?;

                buffers.reset_output(cublas_ctx)?;
                run_cublas_once(cublas_ctx, cell, buffers, graph, path)?;
                let (actual, guards) = buffers.output_bits(cublas_ctx)?;
                check_numeric(
                    &actual,
                    &oracle,
                    &format!("{} cuBLAS {} repeat {repeat}", cell.name, path.name()),
                )?;
                comparator_memory.record(guards);
                require_same_bits(
                    &mut cublas_reference,
                    actual,
                    &format!("{} cuBLAS {} repeat {repeat}", cell.name, path.name()),
                )?;
            }
        }

        let candidate_bits = candidate_reference
            .ok_or_else(|| "candidate correctness gate produced no output".to_string())?;
        let cublas_bits = cublas_reference
            .ok_or_else(|| "cuBLAS correctness gate produced no output".to_string())?;
        Ok(CorrectnessEvidence {
            candidate_digest: bits_digest(b"tf32-nt-output-content.v1", &candidate_bits),
            cublas_digest: bits_digest(b"tf32-nt-output-content.v1", &cublas_bits),
            oracle_digest: bits_digest(b"cpu-oracle.v1", &oracle_bits),
            candidate_bits,
            cublas_bits,
            candidate_memory,
            comparator_memory,
        })
    }

    #[derive(Clone, Copy)]
    struct Iterations {
        candidate: usize,
        cublas: usize,
    }

    struct OrderSamples {
        order: PairOrder,
        candidate_us: Vec<f64>,
        cublas_us: Vec<f64>,
        ratios: Vec<f64>,
    }

    impl OrderSamples {
        fn with_capacity(order: PairOrder, windows: usize) -> Self {
            Self {
                order,
                candidate_us: Vec::with_capacity(windows),
                cublas_us: Vec::with_capacity(windows),
                ratios: Vec::with_capacity(windows),
            }
        }

        fn push(&mut self, candidate_us: f64, cublas_us: f64) -> Result<(), String> {
            require_positive_finite("candidate timing sample", candidate_us)?;
            require_positive_finite("cuBLAS timing sample", cublas_us)?;
            let ratio = candidate_us / cublas_us;
            require_positive_finite("candidate over cuBLAS ratio", ratio)?;
            self.candidate_us.push(candidate_us);
            self.cublas_us.push(cublas_us);
            self.ratios.push(ratio);
            Ok(())
        }
    }

    struct PairedSamples {
        forward: OrderSamples,
        reverse: OrderSamples,
    }

    #[derive(Clone, Copy)]
    struct Percentiles {
        p05: f64,
        p50: f64,
        p95: f64,
    }

    struct OrderSummary {
        candidate: Percentiles,
        cublas: Percentiles,
        ratio: Percentiles,
    }

    struct PairedSummary {
        forward: OrderSummary,
        reverse: OrderSummary,
    }

    struct CohortEvidence {
        calibration_preflight: String,
        timed_preflight: String,
        postflight: String,
        iterations: Iterations,
        candidate_post_memory: GuardCheck,
        comparator_post_memory: GuardCheck,
    }

    struct PostTimingSafety {
        candidate: GuardCheck,
        comparator: GuardCheck,
    }

    struct MeasurementRuntime<'ctx, 'run> {
        cell: Cell,
        path: PathKind,
        candidate_ctx: &'ctx GpuCtx,
        candidate: &'run mut QualifiedPhysicalLaunch<'ctx>,
        guarded_candidate: &'run mut CandidateGuardedLaunch,
        cublas_ctx: &'ctx GpuCtx,
        buffers: &'run mut ComparatorBuffers,
        graph: &'run CudaGraph,
    }

    struct CohortRun<'ctx, 'run> {
        runtime: MeasurementRuntime<'ctx, 'run>,
        quiet: &'run QuietGpu,
        correctness: &'run CorrectnessEvidence,
        windows: usize,
    }

    fn require_positive_finite(label: &str, value: f64) -> Result<(), String> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(format!("{label} is not positive and finite: {value}"))
        }
    }

    fn measure_candidate(
        ctx: &GpuCtx,
        launch: &mut QualifiedPhysicalLaunch<'_>,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        let total_ms = match path {
            PathKind::Eager => launch.measure_eager_window_ms(ctx, iterations)?,
            PathKind::Graph => launch.measure_graph_window_ms(ctx, iterations)?,
        };
        let us = total_ms * 1_000.0 / iterations as f64;
        require_positive_finite("candidate timing window", us)?;
        Ok(us)
    }

    fn measure_cublas(
        ctx: &GpuCtx,
        cell: Cell,
        buffers: &ComparatorBuffers,
        graph: &CudaGraph,
        path: PathKind,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("cuBLAS timing window requires positive iterations".into());
        }
        let start = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record cuBLAS start event: {error:?}"))?;
        for _ in 0..iterations {
            match path {
                PathKind::Eager => launch_cublas(ctx, cell, buffers)?,
                PathKind::Graph => graph
                    .launch()
                    .map_err(|error| format!("launch cuBLAS timing graph: {error:?}"))?,
            }
        }
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record cuBLAS end event: {error:?}"))?;
        let total_ms = start
            .elapsed_ms(&end)
            .map(f64::from)
            .map_err(|error| format!("measure cuBLAS timing events: {error:?}"))?;
        let us = total_ms * 1_000.0 / iterations as f64;
        require_positive_finite("cuBLAS timing window", us)?;
        Ok(us)
    }

    fn measure_arm(
        runtime: &mut MeasurementRuntime<'_, '_>,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        match arm {
            Arm::Candidate => measure_candidate(
                runtime.candidate_ctx,
                runtime.candidate,
                runtime.path,
                iterations,
            ),
            Arm::Cublas => measure_cublas(
                runtime.cublas_ctx,
                runtime.cell,
                runtime.buffers,
                runtime.graph,
                runtime.path,
                iterations,
            ),
        }
    }

    fn calibrate_arm(runtime: &mut MeasurementRuntime<'_, '_>, arm: Arm) -> Result<usize, String> {
        let pilot_us = measure_arm(runtime, arm, PILOT_ITERATIONS)?;
        Ok(((TARGET_WINDOW_US / pilot_us).ceil() as usize).clamp(1, MAX_WINDOW_ITERATIONS))
    }

    fn measure_pair(
        runtime: &mut MeasurementRuntime<'_, '_>,
        order: PairOrder,
        iterations: Iterations,
    ) -> Result<(f64, f64), String> {
        match order {
            PairOrder::CandidateThenCublas => Ok((
                measure_arm(runtime, Arm::Candidate, iterations.candidate)?,
                measure_arm(runtime, Arm::Cublas, iterations.cublas)?,
            )),
            PairOrder::CublasThenCandidate => {
                let cublas = measure_arm(runtime, Arm::Cublas, iterations.cublas)?;
                let candidate = measure_arm(runtime, Arm::Candidate, iterations.candidate)?;
                Ok((candidate, cublas))
            }
        }
    }

    fn percentiles(values: &[f64], windows: usize) -> Result<Percentiles, String> {
        if values.len() != windows || windows < 3 {
            return Err(format!(
                "percentiles require {windows} samples, received {}",
                values.len()
            ));
        }
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        let rank = |percent: usize| (ordered.len() - 1) * percent / 100;
        Ok(Percentiles {
            p05: ordered[rank(5)],
            p50: ordered[rank(50)],
            p95: ordered[rank(95)],
        })
    }

    fn summarize_order(samples: &OrderSamples, windows: usize) -> Result<OrderSummary, String> {
        Ok(OrderSummary {
            candidate: percentiles(&samples.candidate_us, windows)?,
            cublas: percentiles(&samples.cublas_us, windows)?,
            ratio: percentiles(&samples.ratios, windows)?,
        })
    }

    fn summarize(samples: &PairedSamples, windows: usize) -> Result<PairedSummary, String> {
        Ok(PairedSummary {
            forward: summarize_order(&samples.forward, windows)?,
            reverse: summarize_order(&samples.reverse, windows)?,
        })
    }

    fn validate_post_timing(run: &mut CohortRun<'_, '_>) -> Result<PostTimingSafety, String> {
        let runtime = &mut run.runtime;
        runtime
            .candidate
            .seed_f32_operands(runtime.candidate_ctx, SALT)?;
        run_candidate_once(runtime.candidate_ctx, runtime.candidate, runtime.path)?;
        let qualified_bits = runtime.candidate.f32_output_bits(runtime.candidate_ctx)?;
        if qualified_bits != run.correctness.candidate_bits {
            return Err(format!(
                "{} candidate changed bits after {} timing",
                runtime.cell.name,
                runtime.path.name()
            ));
        }
        let (guarded_bits, candidate) =
            runtime.guarded_candidate.reset_run_validate(runtime.path)?;
        if guarded_bits != qualified_bits {
            return Err(format!(
                "{} guarded direct candidate differs after {} timing",
                runtime.cell.name,
                runtime.path.name()
            ));
        }
        runtime.buffers.reset_output(runtime.cublas_ctx)?;
        run_cublas_once(
            runtime.cublas_ctx,
            runtime.cell,
            runtime.buffers,
            runtime.graph,
            runtime.path,
        )?;
        let (actual, comparator) = runtime.buffers.output_bits(runtime.cublas_ctx)?;
        if actual != run.correctness.cublas_bits {
            return Err(format!(
                "{} cuBLAS changed bits after {} timing",
                runtime.cell.name,
                runtime.path.name()
            ));
        }
        Ok(PostTimingSafety {
            candidate,
            comparator,
        })
    }

    fn run_cohort(
        mut run: CohortRun<'_, '_>,
    ) -> Result<(PairedSamples, PairedSummary, CohortEvidence), String> {
        let label = format!(
            "tf32-nt-cublas/{}/{}",
            run.runtime.cell.name,
            run.runtime.path.name()
        );
        let calibration_preflight = run.quiet.require_cohort(&format!("{label}/calibration"))?;
        measure_arm(&mut run.runtime, Arm::Candidate, WARMUPS)?;
        measure_arm(&mut run.runtime, Arm::Cublas, WARMUPS)?;
        let iterations = Iterations {
            candidate: calibrate_arm(&mut run.runtime, Arm::Candidate)?,
            cublas: calibrate_arm(&mut run.runtime, Arm::Cublas)?,
        };
        let timed_preflight = run.quiet.require_cohort(&format!("{label}/timed"))?;
        let mut samples = PairedSamples {
            forward: OrderSamples::with_capacity(PairOrder::CandidateThenCublas, run.windows),
            reverse: OrderSamples::with_capacity(PairOrder::CublasThenCandidate, run.windows),
        };
        for _ in 0..run.windows {
            let (candidate, cublas) =
                measure_pair(&mut run.runtime, PairOrder::CandidateThenCublas, iterations)?;
            samples.forward.push(candidate, cublas)?;
            let (candidate, cublas) =
                measure_pair(&mut run.runtime, PairOrder::CublasThenCandidate, iterations)?;
            samples.reverse.push(candidate, cublas)?;
        }
        let postflight = run.quiet.verify_post_cohort(&format!("{label}/post"))?;
        let post_memory = validate_post_timing(&mut run)?;
        let summary = summarize(&samples, run.windows)?;
        Ok((
            samples,
            summary,
            CohortEvidence {
                calibration_preflight,
                timed_preflight,
                postflight,
                iterations,
                candidate_post_memory: post_memory.candidate,
                comparator_post_memory: post_memory.comparator,
            },
        ))
    }

    struct RunMetadata {
        pre_context: String,
        gpu_uuid: String,
        compute_capability: (u32, u32),
        multiprocessors: u32,
        driver_api_version: i32,
        driver_build_digest: String,
        cublas_version: i32,
        nvrtc_version: (i32, i32),
        nvrtc_target: String,
        compile_key: String,
        artifact_digest: String,
        invocation_digest: String,
        test_source_sha256: String,
        cuda_source_sha256: String,
        run_identity: String,
    }

    impl RunMetadata {
        fn new(
            launch: &QualifiedPhysicalLaunch<'_>,
            device: &GpuDevice,
            cublas_ctx: &GpuCtx,
            quiet: &QuietGpu,
            pre_context: String,
        ) -> Result<Self, String> {
            let route_identity = launch.evidence().route_identity();
            let compiler = route_identity.compiler;
            let artifact = route_identity.artifacts.triad_sm80;
            let identity = device.identity();
            let test_source_sha256 = sha256_bytes(TEST_SOURCE.as_bytes());
            let cuda_source_sha256 = sha256_bytes(CUDA_SOURCE.as_bytes());
            let cublas_version = cublas_version(cublas_ctx)?;
            let mut run = Sha256::new();
            run.update(b"gemm-bi-tf32-nt-cublas-comparator-run.v1");
            run.update(compiler.source_digest);
            run.update(compiler.invocation_digest);
            run.update(artifact.compile_key);
            run.update(artifact.artifact_digest);
            run.update(identity.compute_capability.0.to_le_bytes());
            run.update(identity.compute_capability.1.to_le_bytes());
            run.update(identity.multiprocessor_count.to_le_bytes());
            run.update(identity.driver.api_version.to_le_bytes());
            run.update(identity.driver.build_digest);
            run.update(cublas_version.to_le_bytes());
            run.update(test_source_sha256.as_bytes());
            Ok(Self {
                pre_context,
                gpu_uuid: quiet.uuid.clone(),
                compute_capability: identity.compute_capability,
                multiprocessors: identity.multiprocessor_count,
                driver_api_version: identity.driver.api_version,
                driver_build_digest: hex_digest(&identity.driver.build_digest),
                cublas_version,
                nvrtc_version: compiler.nvrtc_version,
                nvrtc_target: compiler.target.as_str().to_owned(),
                compile_key: hex_digest(&artifact.compile_key),
                artifact_digest: hex_digest(&artifact.artifact_digest),
                invocation_digest: hex_digest(&compiler.invocation_digest),
                test_source_sha256,
                cuda_source_sha256,
                run_identity: format!("{:x}", run.finalize()),
            })
        }
    }

    struct MeasurementRecord<'a, 'q> {
        metadata: &'a RunMetadata,
        cell: Cell,
        path: PathKind,
        windows: usize,
        candidate: &'a QualifiedPhysicalLaunch<'q>,
        cublas_graph: &'a GraphSnapshot,
        correctness: &'a CorrectnessEvidence,
        correctness_preflight: &'a str,
        correctness_postflight: &'a str,
        samples: &'a PairedSamples,
        summary: &'a PairedSummary,
        cohort: &'a CohortEvidence,
    }

    struct JsonlSink {
        path: PathBuf,
        writer: BufWriter<File>,
        digest: Sha256,
        records: usize,
    }

    impl JsonlSink {
        fn create_from_env() -> Result<Self, String> {
            let value = std::env::var_os(OUTPUT_ENV)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("{OUTPUT_ENV} must name a new JSONL file"))?;
            let path = PathBuf::from(value);
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create TF32 NT cuBLAS evidence {path:?}: {error}"))?;
            if !file
                .metadata()
                .map_err(|error| format!("inspect TF32 NT cuBLAS evidence {path:?}: {error}"))?
                .file_type()
                .is_file()
            {
                return Err(format!(
                    "TF32 NT cuBLAS evidence {path:?} is not a regular file"
                ));
            }
            Ok(Self {
                path,
                writer: BufWriter::new(file),
                digest: Sha256::new(),
                records: 0,
            })
        }

        fn write(&mut self, record: String) -> Result<(), String> {
            let mut bytes = record.into_bytes();
            bytes.push(b'\n');
            self.writer.write_all(&bytes).map_err(|error| {
                format!("write TF32 NT cuBLAS evidence {:?}: {error}", self.path)
            })?;
            self.digest.update(&bytes);
            self.records += 1;
            Ok(())
        }

        fn finish(mut self, windows: usize) -> Result<(), String> {
            if self.records != RECORDS_BEFORE_COMPLETION {
                return Err(format!(
                    "TF32 NT cuBLAS evidence requires {RECORDS_BEFORE_COMPLETION} measurement records, received {}",
                    self.records
                ));
            }
            let content_digest = format!("{:x}", self.digest.clone().finalize());
            self.write(completion_record(self.records, windows, &content_digest).render())?;
            self.writer.flush().map_err(|error| {
                format!("flush TF32 NT cuBLAS evidence {:?}: {error}", self.path)
            })?;
            self.writer.get_ref().sync_all().map_err(|error| {
                format!("sync TF32 NT cuBLAS evidence {:?}: {error}", self.path)
            })?;
            eprintln!(
                "TF32 NT cuBLAS comparator wrote {} records to {:?}",
                self.records, self.path
            );
            Ok(())
        }
    }

    fn cublas_version(ctx: &GpuCtx) -> Result<i32, String> {
        let mut version = 0;
        let status =
            unsafe { cudarc::cublas::sys::cublasGetVersion_v2(*ctx.blas.handle(), &mut version) };
        if status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(version)
        } else {
            Err(format!("query cuBLAS version failed: {status:?}"))
        }
    }

    fn render_samples(values: &[f64]) -> String {
        let mut rendered = String::new();
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            write!(rendered, "{value:.9}").expect("String writes cannot fail");
        }
        rendered
    }

    fn render_percentiles(values: Percentiles) -> String {
        format!(
            "{{\"p05\":{:.9},\"p50\":{:.9},\"p95\":{:.9}}}",
            values.p05, values.p50, values.p95
        )
    }

    fn render_order(samples: &OrderSamples, summary: &OrderSummary) -> String {
        format!(
            concat!(
                "{{\"order\":\"{}\",",
                "\"candidate_us\":{},\"cublas_us\":{},",
                "\"candidate_over_cublas_within_path\":{},",
                "\"candidate_samples_us\":[{}],\"cublas_samples_us\":[{}],",
                "\"candidate_over_cublas_within_path_samples\":[{}]}}"
            ),
            samples.order.name(),
            render_percentiles(summary.candidate),
            render_percentiles(summary.cublas),
            render_percentiles(summary.ratio),
            render_samples(&samples.candidate_us),
            render_samples(&samples.cublas_us),
            render_samples(&samples.ratios),
        )
    }

    fn render_measurement(record: MeasurementRecord<'_, '_>) -> String {
        let metadata = record.metadata;
        let evidence = record.candidate.evidence();
        let candidate = candidate_manifest(record.cell);
        let cublas = record.cublas_graph;
        let correctness = record.correctness;
        let candidate_memory = correctness.candidate_memory;
        let comparator_memory = correctness.comparator_memory;
        let (m, k, n) = record.cell.dims;
        let cross_arm_bit_exact = correctness.candidate_bits == correctness.cublas_bits;
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"paired_measurement\",",
                "\"scope\":\"measurement_only\",\"measurement_only\":true,",
                "\"sealed_cohort\":\"sm120_cuda13_2\",",
                "\"cublas_graph_contract\":\"cutlass_tf32_gemm_to_cublaslt_split_k_reducer\",",
                "\"comparison_scope\":\"{}\",",
                "\"cross_path_comparison_valid\":false,",
                "\"full_training_graph_projection_valid\":false,",
                "\"runtime_autotuning\":false,\"dispatcher_admitted\":false,",
                "\"decision\":\"{}\",\"run_identity\":\"{}\",",
                "\"cell\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                "\"logical_op\":\"NT\",\"nt_semantics\":\"{}\",",
                "\"a_layout\":\"row_major_M_by_N\",",
                "\"b_layout\":\"row_major_K_by_N\",",
                "\"output_layout\":\"row_major_M_by_K\",",
                "\"path\":\"{}\",\"timing_measurement_kind\":\"{}\",",
                "\"timing_clock\":\"{}\",",
                "\"host_api_enqueue_time_included\":{},",
                "\"graph_launch_submission_per_logical_gemm\":{},",
                "\"windows_per_order\":{},",
                "\"order_policy\":\"every_window_in_forward_and_reverse_order\",",
                "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
                "\"driver_api_version\":{},\"driver_build_digest\":\"{}\",",
                "\"cublas_version\":{},\"nvrtc_version\":\"{}.{}\",",
                "\"nvrtc_target\":\"{}\",\"sm80_compile_key\":\"{}\",",
                "\"sm80_artifact_digest\":\"{}\",",
                "\"sm80_invocation_digest\":\"{}\",",
                "\"test_source_sha256\":\"{}\",\"cuda_source_sha256\":\"{}\",",
                "\"pre_context\":\"{}\",\"correctness_preflight\":\"{}\",",
                "\"correctness_postflight\":\"{}\",",
                "\"calibration_preflight\":\"{}\",\"timed_preflight\":\"{}\",",
                "\"postflight\":\"{}\","
            ),
            SCHEMA,
            COMPARISON_SCOPE,
            DECISION,
            metadata.run_identity,
            record.cell.name,
            m,
            k,
            n,
            NT_SEMANTICS,
            record.path.name(),
            record.path.timing_measurement_kind(),
            record.path.timing_clock(),
            record.path.host_api_enqueue_time_included(),
            record.path.graph_launch_submission_per_logical_gemm(),
            record.windows,
            escape_json(&metadata.gpu_uuid),
            metadata.compute_capability.0,
            metadata.compute_capability.1,
            metadata.multiprocessors,
            metadata.driver_api_version,
            metadata.driver_build_digest,
            metadata.cublas_version,
            metadata.nvrtc_version.0,
            metadata.nvrtc_version.1,
            escape_json(&metadata.nvrtc_target),
            metadata.compile_key,
            metadata.artifact_digest,
            metadata.invocation_digest,
            metadata.test_source_sha256,
            metadata.cuda_source_sha256,
            escape_json(&metadata.pre_context),
            escape_json(record.correctness_preflight),
            escape_json(record.correctness_postflight),
            escape_json(&record.cohort.calibration_preflight),
            escape_json(&record.cohort.timed_preflight),
            escape_json(&record.cohort.postflight),
        )
        .expect("String writes cannot fail");
        write!(
            json,
            concat!(
                "\"candidate\":{{\"route\":\"forced_deterministic_tf32_nt_p4_s3\",",
                "\"symbol\":\"{}\",\"tile\":[16,32],\"bk\":32,",
                "\"partitions\":4,\"stages\":3,\"grid\":[{},{},{}],",
                "\"block\":[{},{},{}],\"dynamic_shared_bytes\":{},",
                "\"graph_node_count\":1,\"graph_node_kind\":\"kernel\",",
                "\"launch_digest\":\"{}\",\"request_digest\":\"{}\",",
                "\"iterations\":{},\"output_digest\":\"{}\"}},",
                "\"comparator\":{{\"route\":\"cublasGemmEx\",",
                "\"compute_type\":\"CUBLAS_COMPUTE_32F_FAST_TF32\",",
                "\"algorithm\":\"CUBLAS_GEMM_DEFAULT\",",
                "\"trans_a\":\"CUBLAS_OP_T\",\"trans_b\":\"CUBLAS_OP_N\",",
                "\"graph_node_count\":{},\"graph_node_kinds\":[\"kernel\",\"kernel\"],",
                "\"graph_edge_count\":{},\"graph_edge\":\"gemm_to_split_k_reducer\",",
                "\"gemm_kernel\":{{\"role\":\"cutlass_tf32_gemm\",",
                "\"symbol\":\"{}\",\"grid\":[{},{},{}],",
                "\"block\":[{},{},{}],\"dynamic_shared_bytes\":{}}},",
                "\"split_k_reducer_kernel\":{{\"role\":\"cublaslt_split_k_reducer\",",
                "\"symbol\":\"{}\",\"grid\":[{},{},{}],",
                "\"block\":[{},{},{}],\"dynamic_shared_bytes\":{}}},",
                "\"iterations\":{},\"output_digest\":\"{}\"}},"
            ),
            CANDIDATE_SYMBOL,
            candidate.grid.0,
            candidate.grid.1,
            candidate.grid.2,
            candidate.block.0,
            candidate.block.1,
            candidate.block.2,
            candidate.dynamic_shared_bytes,
            hex_digest(&evidence.launch_digest()),
            hex_digest(&evidence.request_identity_digest()),
            record.cohort.iterations.candidate,
            hex_digest(&correctness.candidate_digest),
            cublas.node_count,
            cublas.edge_count,
            escape_json(&cublas.gemm.kernel_symbol),
            cublas.gemm.grid.0,
            cublas.gemm.grid.1,
            cublas.gemm.grid.2,
            cublas.gemm.block.0,
            cublas.gemm.block.1,
            cublas.gemm.block.2,
            cublas.gemm.dynamic_shared_bytes,
            escape_json(&cublas.split_k_reducer.kernel_symbol),
            cublas.split_k_reducer.grid.0,
            cublas.split_k_reducer.grid.1,
            cublas.split_k_reducer.grid.2,
            cublas.split_k_reducer.block.0,
            cublas.split_k_reducer.block.1,
            cublas.split_k_reducer.block.2,
            cublas.split_k_reducer.dynamic_shared_bytes,
            record.cohort.iterations.cublas,
            hex_digest(&correctness.cublas_digest),
        )
        .expect("String writes cannot fail");
        write!(
            json,
            concat!(
                "\"correctness\":{{",
                "\"oracle\":\"independent_cpu_exact_tf32_input_nt_f64_accumulation\",",
                "\"operand_contract\":\"every_input_bit_exact_under_tf32_rna_and_rne\",",
                "\"oracle_tolerance\":\"0.0025*(1+abs(reference))\",",
                "\"oracle_digest\":\"{}\",",
                "\"determinism_repeats_per_path\":{},",
                "\"candidate_repeat_bit_exact\":true,",
                "\"cublas_repeat_bit_exact\":true,",
                "\"candidate_eager_graph_bit_exact\":true,",
                "\"cublas_eager_graph_bit_exact\":true,",
                "\"cross_arm_bit_exact\":{},",
                "\"candidate_memory_safety\":{{",
                "\"scope\":\"{}\",",
                "\"guarded_direct_launch_matches_qualified_candidate_bits\":true,",
                "\"guard_validations\":{},",
                "\"guard_allocations_per_validation\":3,",
                "\"guard_allocations_checked\":{},",
                "\"guard_elements_per_allocation_side\":{},",
                "\"guard_elements_checked\":{},",
                "\"unchanged_active_input_elements_checked\":{},",
                "\"output_prefill_elements_checked\":{},",
                "\"post_timing_guard_allocations_checked\":{},",
                "\"post_timing_guard_elements_checked\":{},",
                "\"post_timing_unchanged_active_input_elements_checked\":{},",
                "\"post_timing_output_prefill_elements_checked\":{}}},",
                "\"comparator_memory_safety\":{{",
                "\"scope\":\"{}\",",
                "\"guard_validations\":{},",
                "\"guard_allocations_per_validation\":3,",
                "\"guard_allocations_checked\":{},",
                "\"guard_elements_per_allocation_side\":{},",
                "\"guard_elements_checked\":{},",
                "\"unchanged_active_input_elements_checked\":{},",
                "\"output_prefill_elements_checked\":{},",
                "\"post_timing_guard_allocations_checked\":{},",
                "\"post_timing_guard_elements_checked\":{},",
                "\"post_timing_unchanged_active_input_elements_checked\":{},",
                "\"post_timing_output_prefill_elements_checked\":{}}}}},",
                "\"orders\":[{},{}]}}"
            ),
            hex_digest(&correctness.oracle_digest),
            DETERMINISM_REPEATS,
            cross_arm_bit_exact,
            CANDIDATE_MEMORY_SAFETY_SCOPE,
            candidate_memory.validations,
            candidate_memory.guard_allocations_checked,
            GUARD_ELEMENTS,
            candidate_memory.guard_elements_checked,
            candidate_memory.unchanged_input_elements_checked,
            candidate_memory.output_prefill_elements_checked,
            record.cohort.candidate_post_memory.guard_allocations,
            record.cohort.candidate_post_memory.guard_elements,
            record.cohort.candidate_post_memory.unchanged_input_elements,
            record.cohort.candidate_post_memory.output_prefill_elements,
            COMPARATOR_MEMORY_SAFETY_SCOPE,
            comparator_memory.validations,
            comparator_memory.guard_allocations_checked,
            GUARD_ELEMENTS,
            comparator_memory.guard_elements_checked,
            comparator_memory.unchanged_input_elements_checked,
            comparator_memory.output_prefill_elements_checked,
            record.cohort.comparator_post_memory.guard_allocations,
            record.cohort.comparator_post_memory.guard_elements,
            record
                .cohort
                .comparator_post_memory
                .unchanged_input_elements,
            record.cohort.comparator_post_memory.output_prefill_elements,
            render_order(&record.samples.forward, &record.summary.forward),
            render_order(&record.samples.reverse, &record.summary.reverse),
        )
        .expect("String writes cannot fail");
        json
    }

    fn bits_digest(domain: &[u8], bits: &[u32]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(domain);
        for value in bits {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn hex_digest(bytes: &[u8; 32]) -> String {
        let mut rendered = String::with_capacity(64);
        for byte in bytes {
            write!(rendered, "{byte:02x}").expect("String writes cannot fail");
        }
        rendered
    }

    fn escape_json(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                character if character.is_control() => {
                    write!(escaped, "\\u{:04x}", character as u32)
                        .expect("String writes cannot fail");
                }
                character => escaped.push(character),
            }
        }
        escaped
    }

    fn validate_release_build() -> Result<(), String> {
        if cfg!(debug_assertions) {
            Err("TF32 NT cuBLAS comparator requires --release".into())
        } else {
            Ok(())
        }
    }

    fn validate_device(device: &GpuDevice) -> Result<(), String> {
        if device.compute_capability == (12, 0) {
            Ok(())
        } else {
            Err(format!(
                "TF32 NT cuBLAS comparator requires sealed SM120, received SM{}.{}",
                device.compute_capability.0, device.compute_capability.1
            ))
        }
    }

    fn validate_runtime_cohort(
        device: &GpuDevice,
        candidate: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let compiler = candidate.evidence().route_identity().compiler;
        validate_sealed_cohort(
            device.compute_capability,
            device.identity().driver.api_version,
            compiler.nvrtc_version,
        )
    }

    #[test]
    fn cpu_tf32_oracle_uses_nt_operand_roles() {
        let cell = Cell {
            name: "oracle_fixture",
            dims: (2, 3, 4),
        };
        let expected = cpu_tf32_nt_oracle(cell).unwrap();
        let a = seeded_values(8, SALT ^ 0x2d);
        let b = seeded_values(12, SALT ^ 0x67);
        require_exact_tf32_inputs(&a, "fixture A").unwrap();
        require_exact_tf32_inputs(&b, "fixture B").unwrap();
        let manual = (0..4)
            .map(|inner| f64::from(a[4 + inner]) * f64::from(b[8 + inner]))
            .sum::<f64>();
        assert_eq!(expected[5], manual);
        let mut corrupted = expected
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>();
        check_numeric(&corrupted, &expected, "oracle fixture").unwrap();
        corrupted[5] = f32::INFINITY.to_bits();
        assert!(check_numeric(&corrupted, &expected, "oracle fixture").is_err());
    }

    #[test]
    fn qualification_seed_contract_is_exact_for_both_tf32_tie_rules() {
        for cell in CELLS {
            let (m, k, n) = cell.dims;
            let a = seeded_values(m * n, SALT ^ 0x2d);
            let b = seeded_values(k * n, SALT ^ 0x67);
            require_exact_tf32_inputs(&a, "qualified A").unwrap();
            require_exact_tf32_inputs(&b, "qualified B").unwrap();
        }
        assert_eq!(tf32_rne_bits(f32::from_bits(0x3f80_1000)), 0x3f80_0000);
        assert_eq!(tf32_rna_bits(f32::from_bits(0x3f80_1000)), 0x3f80_2000);
        assert_eq!(tf32_rne_bits(f32::from_bits(0xbf80_1000)), 0xbf80_0000);
        assert_eq!(tf32_rna_bits(f32::from_bits(0xbf80_1000)), 0xbf80_2000);
    }

    #[test]
    #[ignore = "requires a quiet SM120 CUDA 13.2 device, NVRTC, and a new JSONL path"]
    fn forced_tf32_nt_p4_s3_compares_with_cublas_fast_tf32() {
        validate_release_build().unwrap();
        let windows = window_count(std::env::var(FINAL_ENV).ok().as_deref()).unwrap();
        let quiet = QuietGpu::for_cuda_ordinal(0).unwrap();
        let pre_context = quiet
            .require_pre_context("tf32-nt-cublas/pre-context")
            .unwrap();
        let device = GpuDevice::new(0).unwrap();
        validate_device(&device).unwrap();
        let candidate_ctx = GpuCtx::new(&device).unwrap();
        let cublas_ctx = GpuCtx::new(&device).unwrap();
        let requests = CELLS.map(candidate_request);
        presize_physical_qualification_suite(&candidate_ctx, &requests).unwrap();
        let mut pre_context = Some(pre_context);
        let mut metadata = None;
        let mut sink = JsonlSink::create_from_env().unwrap();

        for cell in CELLS {
            let mut candidate =
                qualify_physical_launch(&candidate_ctx, candidate_request(cell)).unwrap();
            validate_runtime_cohort(&device, &candidate).unwrap();
            assert_candidate_manifest(cell, &candidate).unwrap();
            let mut guarded_candidate = CandidateGuardedLaunch::new(&device, cell).unwrap();
            let mut buffers = ComparatorBuffers::new_cublas(&cublas_ctx, cell).unwrap();
            let (cublas_graph, cublas_snapshot) =
                capture_cublas_graph(&cublas_ctx, cell, &mut buffers).unwrap();
            let metadata = metadata.get_or_insert_with(|| {
                RunMetadata::new(
                    &candidate,
                    &device,
                    &cublas_ctx,
                    &quiet,
                    pre_context.take().unwrap(),
                )
                .unwrap()
            });
            let correctness_preflight = quiet
                .require_cohort(&format!("tf32-nt-cublas/{}/correctness", cell.name))
                .unwrap();
            let correctness = correctness_gate(
                cell,
                &candidate_ctx,
                &mut candidate,
                &mut guarded_candidate,
                &cublas_ctx,
                &mut buffers,
                &cublas_graph,
            )
            .unwrap();
            let correctness_postflight = quiet
                .verify_post_cohort(&format!("tf32-nt-cublas/{}/correctness-post", cell.name))
                .unwrap();

            for path in [PathKind::Eager, PathKind::Graph] {
                let (samples, summary, cohort) = run_cohort(CohortRun {
                    runtime: MeasurementRuntime {
                        cell,
                        path,
                        candidate_ctx: &candidate_ctx,
                        candidate: &mut candidate,
                        guarded_candidate: &mut guarded_candidate,
                        cublas_ctx: &cublas_ctx,
                        buffers: &mut buffers,
                        graph: &cublas_graph,
                    },
                    quiet: &quiet,
                    correctness: &correctness,
                    windows,
                })
                .unwrap();
                sink.write(render_measurement(MeasurementRecord {
                    metadata,
                    cell,
                    path,
                    windows,
                    candidate: &candidate,
                    cublas_graph: &cublas_snapshot,
                    correctness: &correctness,
                    correctness_preflight: &correctness_preflight,
                    correctness_postflight: &correctness_postflight,
                    samples: &samples,
                    summary: &summary,
                    cohort: &cohort,
                }))
                .unwrap();
            }
        }
        sink.finish(windows).unwrap();
    }
}
