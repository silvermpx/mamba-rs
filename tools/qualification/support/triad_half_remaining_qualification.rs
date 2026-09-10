#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfDtype {
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainedFamily {
    Compact,
    RegpipeVec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentFamily {
    Compact,
    Regpipe,
    RegpipeVec2,
}

impl TournamentFamily {
    #[cfg(feature = "cuda")]
    const fn name(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Regpipe => "regpipe",
            Self::RegpipeVec2 => "regpipe_vec2",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Compact => 0,
            Self::Regpipe => 1,
            Self::RegpipeVec2 => 2,
        }
    }
}

pub const REMAINING_TN_REGISTER_CAP: i32 = 128;

pub const fn d768_in_tournament_pairs() -> [(TournamentFamily, TournamentFamily); 3] {
    [
        (TournamentFamily::Compact, TournamentFamily::Regpipe),
        (TournamentFamily::Compact, TournamentFamily::RegpipeVec2),
        (TournamentFamily::Regpipe, TournamentFamily::RegpipeVec2),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedTournamentOrder {
    pub order: [TournamentFamily; 3],
    pub pairwise_wins: [u8; 3],
    pub log_ratio_scores: [f64; 3],
}

pub fn observed_tournament_order(
    representative_ratios: [f64; 3],
) -> Result<ObservedTournamentOrder, String> {
    if representative_ratios
        .iter()
        .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
    {
        return Err("d768-in tournament ratios must be finite and positive".into());
    }
    let mut pairwise_wins = [0_u8; 3];
    let mut log_ratio_scores = [0.0_f64; 3];
    for ((candidate, reference), ratio) in d768_in_tournament_pairs()
        .into_iter()
        .zip(representative_ratios)
    {
        let log_ratio = ratio.ln();
        log_ratio_scores[candidate.index()] += log_ratio;
        log_ratio_scores[reference.index()] -= log_ratio;
        let winner = if ratio <= 1.0 { candidate } else { reference };
        pairwise_wins[winner.index()] += 1;
    }
    let mut order = [
        TournamentFamily::Compact,
        TournamentFamily::Regpipe,
        TournamentFamily::RegpipeVec2,
    ];
    order.sort_by(|left, right| {
        pairwise_wins[right.index()]
            .cmp(&pairwise_wins[left.index()])
            .then_with(|| {
                log_ratio_scores[left.index()].total_cmp(&log_ratio_scores[right.index()])
            })
    });
    Ok(ObservedTournamentOrder {
        order,
        pairwise_wins,
        log_ratio_scores,
    })
}

pub fn finish_performance_stops(stops: Vec<String>) -> Result<(), String> {
    if stops.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "remaining-half performance stops after full batch:\n{}",
            stops.join("\n")
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TnCase {
    pub cell: &'static str,
    pub dims: (usize, usize, usize),
    pub dtype: HalfDtype,
    pub retained: RetainedFamily,
    pub expected_grid: u32,
    pub direct_pair: bool,
}

impl TnCase {
    pub const fn new(
        cell: &'static str,
        dims: (usize, usize, usize),
        dtype: HalfDtype,
        retained: RetainedFamily,
        expected_grid: u32,
    ) -> Self {
        Self {
            cell,
            dims,
            dtype,
            retained,
            expected_grid,
            direct_pair: dims.1 == 768 && dims.2 == 3_072,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnCase {
    pub cell: &'static str,
    pub dims: (usize, usize, usize),
    pub dtype: HalfDtype,
}

pub fn tn_cases() -> &'static [TnCase] {
    const CASES: [TnCase; 6] = [
        TnCase::new(
            "d768_in_proj",
            (2_048, 768, 3_072),
            HalfDtype::F16,
            RetainedFamily::RegpipeVec2,
            576,
        ),
        TnCase::new(
            "d768_in_proj",
            (2_048, 768, 3_072),
            HalfDtype::Bf16,
            RetainedFamily::RegpipeVec2,
            576,
        ),
        TnCase::new(
            "d768_out_proj",
            (2_048, 1_536, 768),
            HalfDtype::F16,
            RetainedFamily::Compact,
            288,
        ),
        TnCase::new(
            "d768_out_proj",
            (2_048, 1_536, 768),
            HalfDtype::Bf16,
            RetainedFamily::RegpipeVec2,
            288,
        ),
        TnCase::new(
            "prism_in_proj",
            (4_621, 384, 1_928),
            HalfDtype::F16,
            RetainedFamily::Compact,
            186,
        ),
        TnCase::new(
            "prism_in_proj",
            (4_621, 384, 1_928),
            HalfDtype::Bf16,
            RetainedFamily::Compact,
            186,
        ),
    ];
    &CASES
}

pub const fn nn_case() -> NnCase {
    NnCase {
        cell: "d768_in_proj",
        dims: (2_048, 768, 3_072),
        dtype: HalfDtype::F16,
    }
}

#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
fn weight_dtype(dtype: HalfDtype) -> WeightDtype {
    match dtype {
        HalfDtype::F16 => WeightDtype::F16,
        HalfDtype::Bf16 => WeightDtype::Bf16,
    }
}

#[cfg(feature = "cuda")]
fn policy_dtype(dtype: WeightDtype) -> mamba_rs::mamba_ssm::gpu::kernel_identity::PolicyDtype {
    use mamba_rs::mamba_ssm::gpu::kernel_identity::PolicyDtype;
    match dtype {
        WeightDtype::F16 => PolicyDtype::F16,
        WeightDtype::Bf16 => PolicyDtype::Bf16,
        WeightDtype::F32 => panic!("remaining-half qualification requires a half dtype"),
    }
}

#[cfg(feature = "cuda")]
fn digest_hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(feature = "cuda")]
fn qualify_and_print_auto_evidence(
    t: &Ctx,
    cell: &str,
    op: mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
) -> Result<gemm_bi_triad::QualifiedPhysicalLaunchNode, String> {
    let request = gemm_bi_triad::PhysicalQualificationRequest::contiguous(
        op,
        dims,
        gemm_bi_triad::PhysicalQualificationRoute::HalfPolicy {
            dtype,
            tensor_cores: true,
            half_policy: t.ctx.half_triad_policy(),
        },
    );
    let qualified = gemm_bi_triad::qualify_physical_launch(&t.ctx, request)?;
    let evidence = qualified.evidence();
    let [node] = evidence.nodes() else {
        return Err(format!(
            "{cell} {dtype:?} actual AUTO qualification must record exactly one physical node, observed {}",
            evidence.nodes().len()
        ));
    };
    let expected_strides = (dims.1, dims.2, dims.2);
    if !evidence.eager_graph_equal()
        || node.logical_op != op
        || node.logical_dtype != policy_dtype(dtype)
        || node.execution_dtype != policy_dtype(dtype)
        || node.shape != dims
        || node.strides != expected_strides
        || node.numeric_contract.is_none()
        || node.ownership.is_none()
    {
        return Err(format!(
            "{cell} {dtype:?} actual AUTO physical evidence changed: {node:?}"
        ));
    }
    println!(
        "{{\"schema\":\"MambaBiHalfRemainingActualAutoEvidenceV2\",\"cell\":\"{cell}\",\"op\":\"{op:?}\",\"dtype\":\"{dtype:?}\",\"shape\":[{},{},{}],\"strides\":[{},{},{}],\"symbol\":\"{}\",\"module\":\"{:?}\",\"numeric_contract\":\"{:?}\",\"ownership\":\"{:?}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{},\"eager_graph_equal\":true,\"route_artifacts_sha256\":\"{}\"}}",
        dims.0,
        dims.1,
        dims.2,
        expected_strides.0,
        expected_strides.1,
        expected_strides.2,
        node.symbol,
        node.module_kind,
        node.numeric_contract,
        node.ownership,
        node.launch.grid_dim.0,
        node.launch.grid_dim.1,
        node.launch.grid_dim.2,
        node.launch.block_dim.0,
        node.launch.block_dim.1,
        node.launch.block_dim.2,
        node.launch.shared_mem_bytes,
        digest_hex(&evidence.route_identity().artifacts.ordered_digest),
    );
    let node = *node;
    drop(qualified);
    Ok(node)
}

#[cfg(feature = "cuda")]
fn validate_actual_graph_against_evidence(
    graph: &CudaGraph,
    expected: &gemm_bi_triad::QualifiedPhysicalLaunchNode,
    label: &str,
) -> Result<(), String> {
    let mut count = 0_usize;
    let count_result =
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
    if count_result != sys::CUresult::CUDA_SUCCESS || count != 1 {
        return Err(format!(
            "{label} actual AUTO graph must contain exactly one node: result={count_result:?} count={count}"
        ));
    }
    let mut node = std::ptr::null_mut();
    let nodes_result = unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) };
    if nodes_result != sys::CUresult::CUDA_SUCCESS || node.is_null() {
        return Err(format!(
            "{label} actual AUTO graph node query failed: result={nodes_result:?} count={count}"
        ));
    }
    let mut node_type = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
    let type_result = unsafe { sys::cuGraphNodeGetType(node, &mut node_type) };
    if type_result != sys::CUresult::CUDA_SUCCESS
        || node_type != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL
    {
        return Err(format!(
            "{label} actual AUTO graph node is not a kernel: result={type_result:?} type={node_type:?}"
        ));
    }
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    let params_result = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) };
    if params_result != sys::CUresult::CUDA_SUCCESS || params.func.is_null() {
        return Err(format!(
            "{label} actual AUTO graph omitted kernel parameters: result={params_result:?}"
        ));
    }
    let mut symbol = std::ptr::null();
    let name_result = unsafe { sys::cuFuncGetName(&mut symbol, params.func) };
    if name_result != sys::CUresult::CUDA_SUCCESS || symbol.is_null() {
        return Err(format!(
            "{label} actual AUTO graph function has no symbol: result={name_result:?}"
        ));
    }
    let symbol = unsafe { CStr::from_ptr(symbol) }.to_string_lossy();
    let observed_launch = (
        (params.gridDimX, params.gridDimY, params.gridDimZ),
        (params.blockDimX, params.blockDimY, params.blockDimZ),
        params.sharedMemBytes,
    );
    let expected_launch = (
        expected.launch.grid_dim,
        expected.launch.block_dim,
        expected.launch.shared_mem_bytes,
    );
    if symbol != expected.symbol || observed_launch != expected_launch {
        return Err(format!(
            "{label} actual public graph differs from qualified evidence: symbol={symbol:?}/{} launch={observed_launch:?}/{expected_launch:?}",
            expected.symbol
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn enqueue_actual_tn(t: &Ctx, fixture: &AdaHalfTnFixture) -> Result<(), String> {
    gemm_bi_backward_dw_typed(
        &t.ctx,
        fixture.current.ptr(),
        TypedPtr {
            ptr: fixture.b.ptr(),
            dtype: fixture.dtype,
        },
        TypedPtr {
            ptr: fixture.a.ptr(),
            dtype: fixture.dtype,
        },
        fixture.dims,
    )
}

#[cfg(feature = "cuda")]
fn observe_actual_tn(
    t: &Ctx,
    fixture: &mut AdaHalfTnFixture,
    graph: &CudaGraph,
    path: AdaHalfPath,
    gemms: usize,
    expected: &[u32],
) -> Result<f64, String> {
    fixture.reset(t, AdaHalfTnArm::CurrentTc64)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half TN actual AUTO timing start: {error:?}"))?;
    for _ in 0..gemms {
        match path {
            AdaHalfPath::Eager => enqueue_actual_tn(t, fixture)?,
            AdaHalfPath::Graph => graph
                .launch()
                .map_err(|error| format!("half TN actual AUTO graph: {error:?}"))?,
        }
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half TN actual AUTO timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("half TN actual AUTO timing: {error:?}"))?,
    ) * 1_000.0
        / gemms as f64;
    if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
        return Err(format!("invalid half TN actual AUTO timing {elapsed_us}"));
    }
    let bits = fixture.current.logical_bits(t);
    if bits != expected {
        let mismatch = bits
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(bits.len());
        return Err(format!(
            "half TN actual AUTO differs from exact TC64 oracle at word {mismatch}"
        ));
    }
    fixture.validate_inputs(t)?;
    Ok(elapsed_us)
}

#[cfg(feature = "cuda")]
fn screen_tn_candidate_vs_actual(
    t: &Ctx,
    fixture: &mut AdaHalfTnFixture,
    candidate: &AdaHalfTnCandidate,
    candidate_graph: &CudaGraph,
    actual_graph: &CudaGraph,
    expected: &[u32],
    cell: &str,
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half_tn_candidate_only(
            t,
            fixture,
            candidate,
            candidate_graph,
            path,
            ADA_HALF_TN_OBSERVATION_GEMMS,
            Some(expected),
        )?;
        observe_actual_tn(
            t,
            fixture,
            actual_graph,
            path,
            ADA_HALF_TN_OBSERVATION_GEMMS,
            expected,
        )?;
    }
    let candidate_first = matches!(order, BracketOrder::Abba);
    let sequence = if candidate_first {
        [true, false, false, true]
    } else {
        [false, true, true, false]
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, is_candidate) in sequence.into_iter().enumerate() {
            raw[index] = if is_candidate {
                observe_ada_half_tn_candidate_only(
                    t,
                    fixture,
                    candidate,
                    candidate_graph,
                    path,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    Some(expected),
                )?
                .0
            } else {
                observe_actual_tn(
                    t,
                    fixture,
                    actual_graph,
                    path,
                    ADA_HALF_TN_OBSERVATION_GEMMS,
                    expected,
                )?
            };
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid half TN actual-AUTO p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid half TN actual-AUTO p95")?;
    let order_name = if candidate_first { "ABBA" } else { "BAAB" };
    println!(
        "{{\"schema\":\"MambaBiHalfTnActualAutoScreenV1\",\"cell\":\"{cell}\",\"dtype\":\"{:?}\",\"shape\":[{},{},{}],\"candidate\":\"{}\",\"candidate_symbol\":\"{}\",\"comparator\":\"actual_public_auto\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{ADA_HALF_WINDOWS},\"warmups_per_arm\":{ADA_HALF_WARMUPS},\"logical_gemms_per_observation\":{ADA_HALF_TN_OBSERVATION_GEMMS},\"raw_observations_us\":{:?},\"ratio_direction\":\"candidate_over_actual_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        fixture.dtype,
        fixture.dims.0,
        fixture.dims.1,
        fixture.dims.2,
        candidate.name(),
        candidate.symbol(fixture.dtype),
        path.name(),
        raw_windows,
    );
    Ok([p50, p95])
}

#[cfg(feature = "cuda")]
fn retained_candidate<'a>(
    case: TnCase,
    compact: &'a AdaHalfTnCandidate,
    vec2: &'a AdaHalfTnCandidate,
) -> &'a AdaHalfTnCandidate {
    match case.retained {
        RetainedFamily::Compact => compact,
        RetainedFamily::RegpipeVec2 => vec2,
    }
}

#[cfg(feature = "cuda")]
fn gate_remaining_tn_candidate(
    candidate: &AdaHalfTnCandidate,
    dtype: WeightDtype,
) -> Result<(), String> {
    gate_ada_half_tn_resources(candidate, dtype)?;
    match candidate.kind {
        AdaHalfTnCandidateKind::Tc64Bk64S2Compact
        | AdaHalfTnCandidateKind::Tc64Bk64S2Regpipe
        | AdaHalfTnCandidateKind::Tc64Bk64S2RegpipeVec2 => {}
        _ => {
            return Err(format!(
                "remaining-half qualification received unplanned candidate {}",
                candidate.name()
            ));
        }
    }
    let registers = candidate
        .function(dtype)
        .num_regs()
        .map_err(|error| format!("remaining-half register cap: {error:?}"))?;
    if registers > REMAINING_TN_REGISTER_CAP {
        return Err(format!(
            "remaining-half {} {dtype:?} exceeds register cap: {registers}/{REMAINING_TN_REGISTER_CAP}",
            candidate.name()
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn qualify_tn_case(
    t: &Ctx,
    case: TnCase,
    candidate: &AdaHalfTnCandidate,
) -> Result<Option<String>, String> {
    let dtype = weight_dtype(case.dtype);
    gate_remaining_tn_candidate(candidate, dtype)?;
    let actual_evidence = qualify_and_print_auto_evidence(
        t,
        case.cell,
        mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn,
        dtype,
        case.dims,
    )?;
    let mut fixture = AdaHalfTnFixture::new_with_guards(t, dtype, case.dims, 128, 64);
    let pointer_mod_256 = [
        fixture.a.ptr() % 256,
        fixture.b.ptr() % 256,
        fixture.candidate.ptr() % 256,
        fixture.current.ptr() % 256,
        fixture.fast.ptr() % 256,
    ];
    if pointer_mod_256 != [0; 5] {
        return Err(format!(
            "{} {dtype:?} timing fixture is not 256-byte aligned: {pointer_mod_256:?}",
            case.cell
        ));
    }

    for arm in [
        AdaHalfTnArm::Candidate,
        AdaHalfTnArm::CurrentTc64,
        AdaHalfTnArm::Fast,
    ] {
        fixture.reset(t, arm)?;
        enqueue_ada_half_tn_arm(t, &fixture, candidate, arm)?;
        t.ctx
            .stream
            .synchronize()
            .map_err(|error| format!("{} {dtype:?} warmup: {error:?}", case.cell))?;
    }
    let graphs = [
        capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::Candidate)?,
        capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::CurrentTc64)?,
        capture_ada_half_tn_arm(t, &fixture, candidate, AdaHalfTnArm::Fast)?,
    ];
    validate_ada_half_tn_candidate_graph(&graphs[0], &fixture, candidate, case.expected_grid)?;
    validate_single_node_graph(&graphs[1], "exact TC64 oracle")?;
    validate_nonempty_graph(&graphs[2], "native-half Fast")?;

    let expected = observe_ada_half_tn(
        t,
        &mut fixture,
        candidate,
        &graphs,
        AdaHalfTnArm::CurrentTc64,
        AdaHalfPath::Eager,
        1,
        None,
    )?
    .1;

    fixture.reset(t, AdaHalfTnArm::CurrentTc64)?;
    enqueue_actual_tn(t, &fixture)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{} actual AUTO eager sync: {error:?}", case.cell))?;
    if fixture.current.logical_bits(t) != expected {
        return Err(format!(
            "{} {dtype:?} actual AUTO eager differs from exact TC64 oracle",
            case.cell
        ));
    }
    fixture.validate_inputs(t)?;

    fixture.reset(t, AdaHalfTnArm::CurrentTc64)?;
    let actual_graph =
        unsafe { capture_into_graph(&t.ctx.stream, || enqueue_actual_tn(t, &fixture)) }?;
    validate_actual_graph_against_evidence(&actual_graph, &actual_evidence, case.cell)?;
    observe_actual_tn(
        t,
        &mut fixture,
        &actual_graph,
        AdaHalfPath::Graph,
        1,
        &expected,
    )?;
    for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
        observe_ada_half_tn_candidate_only(
            t,
            &mut fixture,
            candidate,
            &graphs[0],
            path,
            1,
            Some(&expected),
        )?;
    }

    let fast_bits = observe_ada_half_tn(
        t,
        &mut fixture,
        candidate,
        &graphs,
        AdaHalfTnArm::Fast,
        AdaHalfPath::Eager,
        1,
        None,
    )?
    .1;
    if !fast_bits
        .iter()
        .all(|&word| f32::from_bits(word).is_finite())
        || !fast_bits.iter().any(|&word| f32::from_bits(word) != 0.0)
    {
        return Err(format!("{} {dtype:?} Fast output is invalid", case.cell));
    }
    observe_ada_half_tn(
        t,
        &mut fixture,
        candidate,
        &graphs,
        AdaHalfTnArm::Fast,
        AdaHalfPath::Graph,
        1,
        Some(&fast_bits),
    )?;

    let timing_expected = observe_ada_half_tn_candidate_only(
        t,
        &mut fixture,
        candidate,
        &graphs[0],
        AdaHalfPath::Eager,
        ADA_HALF_TN_OBSERVATION_GEMMS,
        None,
    )?
    .1;
    observe_actual_tn(
        t,
        &mut fixture,
        &actual_graph,
        AdaHalfPath::Eager,
        ADA_HALF_TN_OBSERVATION_GEMMS,
        &timing_expected,
    )?;
    let fast_timing_bits = observe_ada_half_tn(
        t,
        &mut fixture,
        candidate,
        &graphs,
        AdaHalfTnArm::Fast,
        AdaHalfPath::Eager,
        ADA_HALF_TN_OBSERVATION_GEMMS,
        None,
    )?
    .1;

    let mut actual_strata = Vec::with_capacity(4);
    let mut fast_strata = Vec::with_capacity(4);
    for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
        for order in [BracketOrder::Abba, BracketOrder::Baab] {
            actual_strata.push(screen_tn_candidate_vs_actual(
                t,
                &mut fixture,
                candidate,
                &graphs[0],
                &actual_graph,
                &timing_expected,
                case.cell,
                path,
                order,
            )?);
            fast_strata.push(screen_ada_half_tn_pair(
                t,
                &mut fixture,
                candidate,
                &graphs,
                AdaHalfTnArm::Fast,
                &timing_expected,
                &fast_timing_bits,
                path,
                order,
            )?);
        }
    }
    let actual_win = retain_decision(&actual_strata);
    let fast_win = retain_decision(&fast_strata);
    println!(
        "{{\"schema\":\"MambaBiHalfTnActualAutoDecisionV1\",\"cell\":\"{}\",\"dtype\":\"{dtype:?}\",\"shape\":[{},{},{}],\"candidate\":\"{}\",\"pointer_mod_256\":{:?},\"actual_auto_strata\":{:?},\"fast_strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"actual_auto_win\":{actual_win},\"fast_win\":{fast_win},\"promotion\":false}}",
        case.cell,
        case.dims.0,
        case.dims.1,
        case.dims.2,
        candidate.name(),
        pointer_mod_256,
        actual_strata,
        fast_strata,
    );
    Ok((!actual_win).then(|| {
        format!(
            "{} {dtype:?} retained candidate did not clear actual AUTO <0.99 p50/p95",
            case.cell
        )
    }))
}

#[cfg(feature = "cuda")]
fn qualify_d768_in_incremental_order(
    t: &Ctx,
    dtype: WeightDtype,
    compact: &AdaHalfTnCandidate,
    regpipe: &AdaHalfTnCandidate,
    vec2: &AdaHalfTnCandidate,
) -> Result<(), String> {
    for candidate in [compact, regpipe, vec2] {
        gate_remaining_tn_candidate(candidate, dtype)?;
    }
    let dims = (2_048, 768, 3_072);
    let mut fixture = AdaHalfTnFixture::new_with_guards(t, dtype, dims, 128, 64);
    let compact_graph = capture_ada_half_tn_arm(t, &fixture, compact, AdaHalfTnArm::Candidate)?;
    let regpipe_graph = capture_ada_half_tn_arm(t, &fixture, regpipe, AdaHalfTnArm::Candidate)?;
    let vec2_graph = capture_ada_half_tn_arm(t, &fixture, vec2, AdaHalfTnArm::Candidate)?;
    for (candidate, graph) in [
        (compact, &compact_graph),
        (regpipe, &regpipe_graph),
        (vec2, &vec2_graph),
    ] {
        validate_ada_half_tn_candidate_graph(graph, &fixture, candidate, 576)?;
    }
    fixture.reset(t, AdaHalfTnArm::CurrentTc64)?;
    enqueue_tc_tn(
        t,
        BackwardSchedule::Tile64,
        dtype,
        fixture.current.ptr(),
        fixture.a.ptr(),
        fixture.b.ptr(),
        dims,
    )?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("d768-in direct-pair oracle sync: {error:?}"))?;
    let expected = fixture.current.logical_bits(t);
    fixture.validate_inputs(t)?;
    for (candidate, graph) in [
        (compact, &compact_graph),
        (regpipe, &regpipe_graph),
        (vec2, &vec2_graph),
    ] {
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            observe_ada_half_tn_candidate_only(
                t,
                &mut fixture,
                candidate,
                graph,
                path,
                1,
                Some(&expected),
            )?;
        }
    }

    let timing_expected = observe_ada_half_tn_candidate_only(
        t,
        &mut fixture,
        compact,
        &compact_graph,
        AdaHalfPath::Eager,
        ADA_HALF_TN_OBSERVATION_GEMMS,
        None,
    )?
    .1;
    for (candidate, graph) in [(regpipe, &regpipe_graph), (vec2, &vec2_graph)] {
        observe_ada_half_tn_candidate_only(
            t,
            &mut fixture,
            candidate,
            graph,
            AdaHalfPath::Eager,
            ADA_HALF_TN_OBSERVATION_GEMMS,
            Some(&timing_expected),
        )?;
    }

    let candidates = [
        (TournamentFamily::Compact, compact, &compact_graph),
        (TournamentFamily::Regpipe, regpipe, &regpipe_graph),
        (TournamentFamily::RegpipeVec2, vec2, &vec2_graph),
    ];
    let mut representative_ratios = [0.0_f64; 3];
    for (pair_index, (candidate_family, reference_family)) in
        d768_in_tournament_pairs().into_iter().enumerate()
    {
        let (_, candidate, candidate_graph) = candidates[candidate_family.index()];
        let (_, reference, reference_graph) = candidates[reference_family.index()];
        let mut strata = Vec::with_capacity(4);
        for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
            for order in [BracketOrder::Abba, BracketOrder::Baab] {
                strata.push(screen_ada_half_tn_candidates(
                    t,
                    &mut fixture,
                    candidate,
                    candidate_graph,
                    reference,
                    reference_graph,
                    &timing_expected,
                    path,
                    order,
                )?);
            }
        }
        let representative_ratio =
            (strata.iter().map(|[p50, _]| p50.ln()).sum::<f64>() / strata.len() as f64).exp();
        representative_ratios[pair_index] = representative_ratio;
        let observed_faster = if representative_ratio <= 1.0 {
            candidate_family
        } else {
            reference_family
        };
        let strict_win = retain_decision(&strata);
        println!(
            "{{\"schema\":\"MambaBiHalfTnD768InPairDecisionV2\",\"dtype\":\"{dtype:?}\",\"shape\":[2048,768,3072],\"candidate_family\":\"{}\",\"candidate\":\"{}\",\"comparator_family\":\"{}\",\"comparator\":\"{}\",\"strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"candidate_clears_1pct\":{strict_win},\"observed_faster\":\"{}\",\"promotion\":false}}",
            candidate_family.name(),
            candidate.name(),
            reference_family.name(),
            reference.name(),
            strata,
            observed_faster.name(),
        );
    }
    let decision = observed_tournament_order(representative_ratios)?;
    println!(
        "{{\"schema\":\"MambaBiHalfTnD768InTournamentDecisionV2\",\"dtype\":\"{dtype:?}\",\"shape\":[2048,768,3072],\"observed_order\":[\"{}\",\"{}\",\"{}\"],\"pairwise_wins\":{:?},\"log_ratio_scores\":{:?},\"promotion\":false}}",
        decision.order[0].name(),
        decision.order[1].name(),
        decision.order[2].name(),
        decision.pairwise_wins,
        decision.log_ratio_scores,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn enqueue_actual_nn(t: &Ctx, fixture: &AdaHalfFixture) -> Result<(), String> {
    gemm_bi_forward_typed(
        &t.ctx,
        TypedPtr {
            ptr: fixture.reference.ptr(),
            dtype: fixture.dtype,
        },
        TypedPtr {
            ptr: fixture.a.ptr(),
            dtype: fixture.dtype,
        },
        TypedPtr {
            ptr: fixture.b.ptr(),
            dtype: fixture.dtype,
        },
        0,
        fixture.dims,
    )
}

#[cfg(feature = "cuda")]
fn observe_actual_nn(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    graph: &CudaGraph,
    path: AdaHalfPath,
    expected: &[u16],
) -> Result<f64, String> {
    fixture.reset_and_validate_inputs(t, AdaHalfArm::PortableTc128)?;
    let start = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half NN actual AUTO timing start: {error:?}"))?;
    match path {
        AdaHalfPath::Eager => enqueue_actual_nn(t, fixture)?,
        AdaHalfPath::Graph => graph
            .launch()
            .map_err(|error| format!("half NN actual AUTO graph: {error:?}"))?,
    }
    let end = t
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("half NN actual AUTO timing end: {error:?}"))?;
    let elapsed_us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("half NN actual AUTO timing: {error:?}"))?,
    ) * 1_000.0;
    let bits = fixture.reference.logical_bits(t);
    if bits != expected {
        return Err("half NN actual AUTO differs from exact forced-TC128 oracle".into());
    }
    fixture.validate_inputs(t)?;
    Ok(elapsed_us)
}

#[cfg(feature = "cuda")]
fn screen_nn_candidate_vs_actual(
    t: &Ctx,
    fixture: &AdaHalfFixture,
    candidate_graph: &CudaGraph,
    actual_graph: &CudaGraph,
    expected: &[u16],
    path: AdaHalfPath,
    order: BracketOrder,
) -> Result<[f64; 2], String> {
    for _ in 0..ADA_HALF_WARMUPS {
        observe_ada_half(
            t,
            fixture,
            AdaHalfArm::FixedSm89Tc128S3,
            candidate_graph,
            path,
            Some(expected),
        )?;
        observe_actual_nn(t, fixture, actual_graph, path, expected)?;
    }
    let candidate_first = matches!(order, BracketOrder::Abba);
    let sequence = if candidate_first {
        [true, false, false, true]
    } else {
        [false, true, true, false]
    };
    let mut raw_windows = Vec::with_capacity(ADA_HALF_WINDOWS);
    let mut ratios = Vec::with_capacity(ADA_HALF_WINDOWS);
    for _ in 0..ADA_HALF_WINDOWS {
        let mut raw = [0.0; 4];
        for (index, is_candidate) in sequence.into_iter().enumerate() {
            raw[index] = if is_candidate {
                observe_ada_half(
                    t,
                    fixture,
                    AdaHalfArm::FixedSm89Tc128S3,
                    candidate_graph,
                    path,
                    Some(expected),
                )?
                .0
            } else {
                observe_actual_nn(t, fixture, actual_graph, path, expected)?
            };
        }
        ratios.push(candidate_over_reference(raw, order));
        raw_windows.push(raw);
    }
    let p50 = ada_percentile(&ratios, 0.5).ok_or("invalid half NN actual-AUTO p50")?;
    let p95 = ada_percentile(&ratios, 0.95).ok_or("invalid half NN actual-AUTO p95")?;
    let order_name = if candidate_first { "ABBA" } else { "BAAB" };
    println!(
        "{{\"schema\":\"MambaBiHalfNnActualAutoScreenV1\",\"cell\":\"d768_in_proj\",\"dtype\":\"F16\",\"shape\":[2048,768,3072],\"candidate\":\"fixed_sm89_tc128_s3\",\"candidate_symbol\":\"{}\",\"comparator\":\"actual_public_auto\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{ADA_HALF_WINDOWS},\"raw_observations_us\":{:?},\"ratio_direction\":\"candidate_over_actual_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
        ada_half_symbol(AdaHalfArm::FixedSm89Tc128S3, WeightDtype::F16),
        path.name(),
        raw_windows,
    );
    Ok([p50, p95])
}

#[cfg(feature = "cuda")]
fn qualify_nn_case(t: &Ctx) -> Result<Option<String>, String> {
    let case = nn_case();
    let dtype = weight_dtype(case.dtype);
    ada_half_resource_gate(t, AdaHalfArm::FixedSm89Tc128S3, dtype)?;
    ada_half_resource_gate(t, AdaHalfArm::PortableTc128, dtype)?;
    let actual_evidence = qualify_and_print_auto_evidence(
        t,
        case.cell,
        mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn,
        dtype,
        case.dims,
    )?;
    let fixture = AdaHalfFixture::new_with_guard(t, dtype, case.dims, 128);
    let pointer_mod_256 = [
        fixture.a.ptr() % 256,
        fixture.b.ptr() % 256,
        fixture.candidate.ptr() % 256,
        fixture.reference.ptr() % 256,
        fixture.fast.ptr() % 256,
    ];
    if pointer_mod_256 != [0; 5] {
        return Err(format!(
            "half NN F16 d768-in fixture is not 256-byte aligned: {pointer_mod_256:?}"
        ));
    }
    let candidate_graph = capture_ada_half(t, &fixture, AdaHalfArm::FixedSm89Tc128S3)?;
    let reference_graph = capture_ada_half(t, &fixture, AdaHalfArm::PortableTc128)?;
    fixture.reset_fast(t)?;
    enqueue_ada_half_nn_fast(t, &fixture)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("half NN Fast warmup: {error:?}"))?;
    let fast_graph = capture_ada_half_nn_fast(t, &fixture)?;
    validate_single_node_graph(&candidate_graph, "half NN retained S3")?;
    validate_single_node_graph(&reference_graph, "half NN forced TC128 oracle")?;
    validate_nonempty_graph(&fast_graph, "half NN native Fast")?;
    let expected = observe_ada_half(
        t,
        &fixture,
        AdaHalfArm::PortableTc128,
        &reference_graph,
        AdaHalfPath::Eager,
        None,
    )?
    .1;

    fixture.reset_and_validate_inputs(t, AdaHalfArm::PortableTc128)?;
    enqueue_actual_nn(t, &fixture)?;
    t.ctx
        .stream
        .synchronize()
        .map_err(|error| format!("half NN actual AUTO eager sync: {error:?}"))?;
    if fixture.reference.logical_bits(t) != expected {
        return Err("half NN actual AUTO eager differs from forced-TC128 oracle".into());
    }
    fixture.validate_inputs(t)?;

    fixture.reset_and_validate_inputs(t, AdaHalfArm::PortableTc128)?;
    let actual_graph =
        unsafe { capture_into_graph(&t.ctx.stream, || enqueue_actual_nn(t, &fixture)) }?;
    validate_actual_graph_against_evidence(&actual_graph, &actual_evidence, case.cell)?;
    observe_actual_nn(t, &fixture, &actual_graph, AdaHalfPath::Graph, &expected)?;
    for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
        observe_ada_half(
            t,
            &fixture,
            AdaHalfArm::FixedSm89Tc128S3,
            &candidate_graph,
            path,
            Some(&expected),
        )?;
    }
    let fast_bits = observe_ada_half_nn_fast(t, &fixture, &fast_graph, AdaHalfPath::Eager, None)?.1;
    observe_ada_half_nn_fast(
        t,
        &fixture,
        &fast_graph,
        AdaHalfPath::Graph,
        Some(&fast_bits),
    )?;

    let mut actual_strata = Vec::with_capacity(4);
    let mut fast_strata = Vec::with_capacity(4);
    for path in [AdaHalfPath::Eager, AdaHalfPath::Graph] {
        for order in [BracketOrder::Abba, BracketOrder::Baab] {
            actual_strata.push(screen_nn_candidate_vs_actual(
                t,
                &fixture,
                &candidate_graph,
                &actual_graph,
                &expected,
                path,
                order,
            )?);
            fast_strata.push(ada_half_s3_fast_stratum(
                t,
                &fixture,
                &candidate_graph,
                &fast_graph,
                &expected,
                &fast_bits,
                case.cell,
                path,
                order,
            )?);
        }
    }
    let actual_win = retain_decision(&actual_strata);
    let fast_win = retain_decision(&fast_strata);
    println!(
        "{{\"schema\":\"MambaBiHalfNnActualAutoDecisionV1\",\"cell\":\"{}\",\"dtype\":\"F16\",\"shape\":[{},{},{}],\"candidate\":\"fixed_sm89_tc128_s3\",\"pointer_mod_256\":{:?},\"actual_auto_strata\":{:?},\"fast_strata\":{:?},\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"actual_auto_win\":{actual_win},\"fast_win\":{fast_win},\"promotion\":false}}",
        case.cell,
        case.dims.0,
        case.dims.1,
        case.dims.2,
        pointer_mod_256,
        actual_strata,
        fast_strata,
    );
    Ok((!actual_win)
        .then(|| "half NN F16 d768-in retained S3 did not clear actual AUTO <0.99 p50/p95".into()))
}

#[cfg(feature = "cuda")]
fn run_remaining_half_actual_auto_batch() -> Result<(), String> {
    assert!(
        !cfg!(debug_assertions),
        "remaining half actual-AUTO qualification requires --release"
    );
    let quiet = QuietGpu::for_cuda_ordinal(0)?;
    let _pre = quiet.require_pre_context("half-remaining-actual-auto/pre-context")?;
    let t = Ctx::new_ada()?;
    t.ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    let compiler = t.ctx.kernels.compiler_identity();
    if compiler.nvrtc_version != (13, 2) {
        return Err(format!(
            "remaining half actual-AUTO qualification requires CUDA13.2, found {:?}",
            compiler.nvrtc_version
        ));
    }
    let mut actual_auto_requests = tn_cases()
        .iter()
        .map(|case| {
            gemm_bi_triad::PhysicalQualificationRequest::contiguous(
                mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn,
                case.dims,
                gemm_bi_triad::PhysicalQualificationRoute::HalfPolicy {
                    dtype: weight_dtype(case.dtype),
                    tensor_cores: true,
                    half_policy: t.ctx.half_triad_policy(),
                },
            )
        })
        .collect::<Vec<_>>();
    let nn = nn_case();
    actual_auto_requests.push(gemm_bi_triad::PhysicalQualificationRequest::contiguous(
        mamba_rs::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn,
        nn.dims,
        gemm_bi_triad::PhysicalQualificationRoute::HalfPolicy {
            dtype: weight_dtype(nn.dtype),
            tensor_cores: true,
            half_policy: t.ctx.half_triad_policy(),
        },
    ));
    gemm_bi_triad::presize_physical_qualification_suite(&t.ctx, &actual_auto_requests)?;
    let compact = compile_ada_half_tn_compact_candidate(&t)?;
    let regpipe = compile_ada_half_tn_regpipe_candidate(&t)?;
    let vec2 = compile_ada_half_tn_regpipe_vec2_candidate(&t)?;
    let _cohort = quiet.require_cohort("half-remaining-actual-auto/cohort")?;

    let mut performance_stops = Vec::new();
    for case in tn_cases().iter().copied() {
        let candidate = retained_candidate(case, &compact, &vec2);
        if let Some(stop) = qualify_tn_case(&t, case, candidate)? {
            performance_stops.push(stop);
        }
    }
    for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
        qualify_d768_in_incremental_order(&t, dtype, &compact, &regpipe, &vec2)?;
    }
    if let Some(stop) = qualify_nn_case(&t)? {
        performance_stops.push(stop);
    }

    drop(t);
    quiet
        .verify_post_cohort("half-remaining-actual-auto/post")
        .map(|_| ())?;
    if !performance_stops.is_empty() {
        println!(
            "{{\"schema\":\"MambaBiHalfRemainingPerformanceStopsV1\",\"count\":{},\"stops\":{:?}}}",
            performance_stops.len(),
            performance_stops,
        );
    }
    finish_performance_stops(performance_stops)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive Ada CC8.9 CUDA13.2; seven missing half actual-AUTO comparators"]
fn ada_half_remaining_seven_cells_vs_actual_auto_and_fast_once7() -> Result<(), String> {
    run_remaining_half_actual_auto_batch()
}
