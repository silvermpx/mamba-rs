#[cfg(feature = "cuda")]
mod common;

const DISCOVERY_WINDOWS: usize = 21;
const FINAL_WINDOWS: usize = 101;
const SCRATCH_LIMIT_F32: usize = 1 << 23;
const PROMOTION_MEDIAN_RATIO: f64 = 1.01;
const PROMOTION_P05_RATIO: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    dims: (usize, usize, usize),
    shape_class: &'static str,
    a_offset: usize,
    b_offset: usize,
    output_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartitionPlan {
    name: &'static str,
    m_chunk: usize,
    chunks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartitionCell {
    cell: Cell,
    baseline: PartitionPlan,
    candidates: &'static [PartitionPlan],
}

impl PartitionCell {
    fn plans(&self) -> impl Iterator<Item = PartitionPlan> + '_ {
        std::iter::once(self.baseline).chain(self.candidates.iter().copied())
    }
}

const fn plan(name: &'static str, m_chunk: usize, chunks: usize) -> PartitionPlan {
    PartitionPlan {
        name,
        m_chunk,
        chunks,
    }
}

const PARTITION_CELLS: [PartitionCell; 7] = [
    PartitionCell {
        cell: Cell {
            name: "m256_k32_n2",
            dims: (256, 32, 2),
            shape_class: "launch_bound_aligned",
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        baseline: plan("wave2", 16, 16),
        candidates: &[plan("coarse", 32, 8)],
    },
    PartitionCell {
        cell: Cell {
            name: "m1024_k47_n17",
            dims: (1_024, 47, 17),
            shape_class: "intermediate_tail_aligned",
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        baseline: plan("wave2", 16, 64),
        candidates: &[plan("coarse", 32, 32)],
    },
    PartitionCell {
        cell: Cell {
            name: "m1024_k128_n25",
            dims: (1_024, 128, 25),
            shape_class: "live_tqc_width_aligned",
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        baseline: plan("wave2", 16, 64),
        candidates: &[plan("coarse", 32, 32)],
    },
    PartitionCell {
        cell: Cell {
            name: "m4096_k64_n64",
            dims: (4_096, 64, 64),
            shape_class: "large_aligned",
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        baseline: plan("wave2", 32, 128),
        candidates: &[plan("fine", 16, 256), plan("coarse", 48, 86)],
    },
    PartitionCell {
        cell: Cell {
            name: "m4096_k128_n96",
            dims: (4_096, 128, 96),
            shape_class: "large_multitile_aligned",
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        baseline: plan("wave2", 80, 52),
        candidates: &[plan("fine", 64, 64), plan("coarse", 96, 43)],
    },
    PartitionCell {
        cell: Cell {
            name: "m4096_k256_n101_a_offset1",
            dims: (4_096, 256, 101),
            shape_class: "large_row_tail_a_output_offset",
            a_offset: 1,
            b_offset: 0,
            output_offset: 1,
        },
        baseline: plan("wave2", 192, 22),
        candidates: &[plan("fine", 176, 24), plan("coarse", 208, 20)],
    },
    PartitionCell {
        cell: Cell {
            name: "m4111_k257_n127_b_offset1",
            dims: (4_111, 257, 127),
            shape_class: "full_dimensional_tail_b_output_offset",
            a_offset: 0,
            b_offset: 1,
            output_offset: 1,
        },
        baseline: plan("wave2", 256, 17),
        candidates: &[plan("fine", 240, 18), plan("coarse", 272, 16)],
    },
];

fn partition_cells() -> &'static [PartitionCell] {
    &PARTITION_CELLS
}

fn window_count(value: Option<&str>) -> Result<usize, String> {
    match value.filter(|value| !value.is_empty()) {
        None | Some("discovery") | Some("21") => Ok(DISCOVERY_WINDOWS),
        Some("final") | Some("101") => Ok(FINAL_WINDOWS),
        Some(value) => Err(format!(
            "MAMBA_RS_TN_NARROW_PARTITION_TOURNAMENT_FINAL must be discovery/21 or final/101, received {value:?}"
        )),
    }
}

fn scratch_elements(cell: Cell, plan: PartitionPlan) -> Result<usize, String> {
    cell.dims
        .1
        .checked_mul(cell.dims.2)
        .and_then(|output| output.checked_mul(plan.chunks))
        .ok_or_else(|| "TN narrow partition scratch size overflows usize".to_string())
}

fn partial_grid_blocks(cell: Cell, plan: PartitionPlan) -> Result<usize, String> {
    cell.dims
        .1
        .div_ceil(64)
        .checked_mul(cell.dims.2.div_ceil(32))
        .and_then(|base| base.checked_mul(plan.chunks))
        .ok_or_else(|| "TN narrow partition grid size overflows usize".to_string())
}

fn latin_order(arm_count: usize, window: usize, reverse: bool) -> Result<Vec<usize>, String> {
    if !(3..=4).contains(&arm_count) {
        return Err(format!(
            "TN narrow partition tournament requires three or four arms, received {arm_count}"
        ));
    }
    let mut order = (0..arm_count).collect::<Vec<_>>();
    order.rotate_left(window % arm_count);
    if reverse {
        order.reverse();
    }
    Ok(order)
}

fn promotion_eligible(median_ratio: f64, p05_ratio: f64) -> bool {
    median_ratio.is_finite()
        && p05_ratio.is_finite()
        && median_ratio >= PROMOTION_MEDIAN_RATIO
        && p05_ratio > PROMOTION_P05_RATIO
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WinnerAlignmentCell {
    cell: Cell,
    alignment: &'static str,
    baseline: PartitionPlan,
    winner: PartitionPlan,
}

const fn winner_cell(
    name: &'static str,
    dims: (usize, usize, usize),
    alignment: &'static str,
    offsets: (usize, usize, usize),
    baseline: PartitionPlan,
    winner: PartitionPlan,
) -> WinnerAlignmentCell {
    WinnerAlignmentCell {
        cell: Cell {
            name,
            dims,
            shape_class: "winner_alignment",
            a_offset: offsets.0,
            b_offset: offsets.1,
            output_offset: offsets.2,
        },
        alignment,
        baseline,
        winner,
    }
}

const WINNER_ALIGNMENT_CELLS: [WinnerAlignmentCell; 12] = [
    winner_cell(
        "m1024_k47_n17_aligned",
        (1_024, 47, 17),
        "aligned",
        (0, 0, 0),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k47_n17_a_plus4",
        (1_024, 47, 17),
        "a_plus4",
        (1, 0, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k47_n17_b_plus4",
        (1_024, 47, 17),
        "b_plus4",
        (0, 1, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k47_n17_a_b_plus4",
        (1_024, 47, 17),
        "a_b_plus4",
        (1, 1, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k128_n25_aligned",
        (1_024, 128, 25),
        "aligned",
        (0, 0, 0),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k128_n25_a_plus4",
        (1_024, 128, 25),
        "a_plus4",
        (1, 0, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k128_n25_b_plus4",
        (1_024, 128, 25),
        "b_plus4",
        (0, 1, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m1024_k128_n25_a_b_plus4",
        (1_024, 128, 25),
        "a_b_plus4",
        (1, 1, 1),
        plan("baseline", 16, 64),
        plan("winner", 32, 32),
    ),
    winner_cell(
        "m4096_k64_n64_aligned",
        (4_096, 64, 64),
        "aligned",
        (0, 0, 0),
        plan("baseline", 32, 128),
        plan("winner", 48, 86),
    ),
    winner_cell(
        "m4096_k64_n64_a_plus4",
        (4_096, 64, 64),
        "a_plus4",
        (1, 0, 1),
        plan("baseline", 32, 128),
        plan("winner", 48, 86),
    ),
    winner_cell(
        "m4096_k64_n64_b_plus4",
        (4_096, 64, 64),
        "b_plus4",
        (0, 1, 1),
        plan("baseline", 32, 128),
        plan("winner", 48, 86),
    ),
    winner_cell(
        "m4096_k64_n64_a_b_plus4",
        (4_096, 64, 64),
        "a_b_plus4",
        (1, 1, 1),
        plan("baseline", 32, 128),
        plan("winner", 48, 86),
    ),
];

fn winner_alignment_cells() -> &'static [WinnerAlignmentCell] {
    &WINNER_ALIGNMENT_CELLS
}

fn winner_window_count(value: Option<&str>) -> Result<usize, String> {
    match value.filter(|value| !value.is_empty()) {
        None | Some("discovery") | Some("21") => Ok(DISCOVERY_WINDOWS),
        Some("final") | Some("101") => Ok(FINAL_WINDOWS),
        Some(value) => Err(format!(
            "MAMBA_RS_TN_NARROW_WINNER_ALIGNMENT_FINAL must be discovery/21 or final/101, received {value:?}"
        )),
    }
}

const fn winner_order(reverse: bool) -> [usize; 2] {
    if reverse { [1, 0] } else { [0, 1] }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphReplayPath {
    Eager,
    Graph,
}

impl GraphReplayPath {
    const fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

const fn fixed_graph_replay_path_pair(reverse: bool) -> [GraphReplayPath; 4] {
    if reverse {
        [
            GraphReplayPath::Graph,
            GraphReplayPath::Eager,
            GraphReplayPath::Eager,
            GraphReplayPath::Graph,
        ]
    } else {
        [
            GraphReplayPath::Eager,
            GraphReplayPath::Graph,
            GraphReplayPath::Graph,
            GraphReplayPath::Eager,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphKernelDescriptor {
    symbol: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphEdgeDataDescriptor {
    from_port: u8,
    to_port: u8,
    dependency_type: u8,
    reserved: [u8; 5],
}

const ORDINARY_GRAPH_EDGE_DATA: GraphEdgeDataDescriptor = GraphEdgeDataDescriptor {
    from_port: 0,
    to_port: 0,
    dependency_type: 0,
    reserved: [0; 5],
};

fn require_ordinary_graph_edge_data(
    observed: GraphEdgeDataDescriptor,
    label: &str,
) -> Result<(), String> {
    if observed == ORDINARY_GRAPH_EDGE_DATA {
        Ok(())
    } else {
        Err(format!(
            "{label}: graph edge data {observed:?} is not an ordinary default dependency"
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OutputGuardLayout {
    output_offset: usize,
    output_elements: usize,
    prefix_bits: u32,
    suffix_bits: u32,
}

fn validate_output_red_zones(
    storage: &[f32],
    layout: OutputGuardLayout,
    label: &str,
) -> Result<(), String> {
    let end = layout
        .output_offset
        .checked_add(layout.output_elements)
        .ok_or_else(|| format!("{label} output guard boundary overflows usize"))?;
    if end >= storage.len() {
        return Err(format!(
            "{label} output storage has no complete suffix red zone"
        ));
    }
    for (index, value) in storage[..layout.output_offset].iter().enumerate() {
        if value.to_bits() != layout.prefix_bits {
            return Err(format!("{label} output prefix guard changed at {index}"));
        }
    }
    for (index, value) in storage[end..].iter().enumerate() {
        if value.to_bits() != layout.suffix_bits {
            return Err(format!("{label} output suffix guard changed at {index}"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphEdgeDescriptor {
    nodes: (usize, usize),
    data: GraphEdgeDataDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitGraphTopologyDescriptor {
    nodes: [GraphKernelDescriptor; 2],
    edge: GraphEdgeDescriptor,
}

fn split_graph_topology_descriptor(
    cell: Cell,
    plan: PartitionPlan,
    aligned_inputs: bool,
) -> Result<SplitGraphTopologyDescriptor, String> {
    let (_, k, n) = cell.dims;
    let grid_m = u32::try_from(k)
        .map_err(|_| "TN narrow topology K exceeds u32::MAX")?
        .div_ceil(64);
    let grid_n = u32::try_from(n)
        .map_err(|_| "TN narrow topology N exceeds u32::MAX")?
        .div_ceil(32);
    let chunks =
        u32::try_from(plan.chunks).map_err(|_| "TN narrow topology chunks exceed u32::MAX")?;
    let total = u32::try_from(
        k.checked_mul(n)
            .ok_or("TN narrow topology reducer size overflows usize")?,
    )
    .map_err(|_| "TN narrow topology reducer size exceeds u32::MAX")?;
    let partial = if aligned_inputs {
        "gemm_bi_tn_narrow_splitm_partial_aligned"
    } else {
        "gemm_bi_tn_narrow_splitm_partial"
    };
    Ok(SplitGraphTopologyDescriptor {
        nodes: [
            GraphKernelDescriptor {
                symbol: partial,
                grid: (grid_m, grid_n, chunks),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            GraphKernelDescriptor {
                symbol: "gemm_bi_splitm_reduce",
                grid: (total.div_ceil(256), 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        ],
        edge: GraphEdgeDescriptor {
            nodes: (0, 1),
            data: ORDINARY_GRAPH_EDGE_DATA,
        },
    })
}

const fn winner_symbols(cell: WinnerAlignmentCell) -> [&'static str; 2] {
    let partial = if cell.cell.a_offset == 0 && cell.cell.b_offset == 0 {
        "gemm_bi_tn_narrow_splitm_partial_aligned"
    } else {
        "gemm_bi_tn_narrow_splitm_partial"
    };
    [partial, "gemm_bi_splitm_reduce"]
}

#[test]
fn winner_alignment_matrix_is_independent_exact_and_balanced() {
    let cells = winner_alignment_cells();
    assert_eq!(cells.len(), 12);
    assert_eq!(
        cells
            .iter()
            .map(|cell| {
                (
                    cell.cell.dims,
                    cell.alignment,
                    (
                        cell.cell.a_offset,
                        cell.cell.b_offset,
                        cell.cell.output_offset,
                    ),
                    (cell.baseline.m_chunk, cell.baseline.chunks),
                    (cell.winner.m_chunk, cell.winner.chunks),
                    winner_symbols(*cell),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                (1_024, 47, 17),
                "aligned",
                (0, 0, 0),
                (16, 64),
                (32, 32),
                [
                    "gemm_bi_tn_narrow_splitm_partial_aligned",
                    "gemm_bi_splitm_reduce"
                ]
            ),
            (
                (1_024, 47, 17),
                "a_plus4",
                (1, 0, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (1_024, 47, 17),
                "b_plus4",
                (0, 1, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (1_024, 47, 17),
                "a_b_plus4",
                (1, 1, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (1_024, 128, 25),
                "aligned",
                (0, 0, 0),
                (16, 64),
                (32, 32),
                [
                    "gemm_bi_tn_narrow_splitm_partial_aligned",
                    "gemm_bi_splitm_reduce"
                ]
            ),
            (
                (1_024, 128, 25),
                "a_plus4",
                (1, 0, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (1_024, 128, 25),
                "b_plus4",
                (0, 1, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (1_024, 128, 25),
                "a_b_plus4",
                (1, 1, 1),
                (16, 64),
                (32, 32),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (4_096, 64, 64),
                "aligned",
                (0, 0, 0),
                (32, 128),
                (48, 86),
                [
                    "gemm_bi_tn_narrow_splitm_partial_aligned",
                    "gemm_bi_splitm_reduce"
                ]
            ),
            (
                (4_096, 64, 64),
                "a_plus4",
                (1, 0, 1),
                (32, 128),
                (48, 86),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (4_096, 64, 64),
                "b_plus4",
                (0, 1, 1),
                (32, 128),
                (48, 86),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
            (
                (4_096, 64, 64),
                "a_b_plus4",
                (1, 1, 1),
                (32, 128),
                (48, 86),
                ["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
            ),
        ]
    );
    assert_eq!(winner_order(false), [0, 1]);
    assert_eq!(winner_order(true), [1, 0]);
    assert_eq!(winner_window_count(None).unwrap(), DISCOVERY_WINDOWS);
    assert_eq!(winner_window_count(Some("final")).unwrap(), FINAL_WINDOWS);
    assert!(winner_window_count(Some("100")).is_err());
}

#[test]
fn partition_matrix_is_the_reviewed_seven_cell_sweep() {
    let actual = partition_cells()
        .iter()
        .map(|spec| {
            (
                spec.cell.dims,
                (spec.baseline.m_chunk, spec.baseline.chunks),
                spec.candidates
                    .iter()
                    .map(|plan| (plan.m_chunk, plan.chunks))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            ((256, 32, 2), (16, 16), vec![(32, 8)]),
            ((1_024, 47, 17), (16, 64), vec![(32, 32)]),
            ((1_024, 128, 25), (16, 64), vec![(32, 32)]),
            ((4_096, 64, 64), (32, 128), vec![(16, 256), (48, 86)]),
            ((4_096, 128, 96), (80, 52), vec![(64, 64), (96, 43)]),
            ((4_096, 256, 101), (192, 22), vec![(176, 24), (208, 20)]),
            ((4_111, 257, 127), (256, 17), vec![(240, 18), (272, 16)]),
        ]
    );
}

#[test]
fn window_modes_are_explicit_and_fail_closed() {
    assert_eq!(window_count(None).unwrap(), DISCOVERY_WINDOWS);
    assert_eq!(window_count(Some("discovery")).unwrap(), DISCOVERY_WINDOWS);
    assert_eq!(window_count(Some("21")).unwrap(), DISCOVERY_WINDOWS);
    assert_eq!(window_count(Some("final")).unwrap(), FINAL_WINDOWS);
    assert_eq!(window_count(Some("101")).unwrap(), FINAL_WINDOWS);
    assert!(window_count(Some("1")).is_err());
    assert!(window_count(Some("true")).is_err());
    assert!(window_count(Some("100")).is_err());
}

#[test]
fn every_partition_is_complete_aligned_and_within_the_scratch_cap() {
    for spec in partition_cells() {
        for plan in spec.plans() {
            assert_eq!(plan.m_chunk % 16, 0, "{} {}", spec.cell.name, plan.name);
            assert_eq!(plan.chunks, spec.cell.dims.0.div_ceil(plan.m_chunk));
            assert!(scratch_elements(spec.cell, plan).unwrap() <= SCRATCH_LIMIT_F32);
        }
    }
    let blocks = partition_cells()
        .iter()
        .map(|spec| {
            spec.plans()
                .map(|plan| partial_grid_blocks(spec.cell, plan).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        blocks,
        vec![
            vec![16, 8],
            vec![64, 32],
            vec![128, 64],
            vec![256, 512, 172],
            vec![312, 384, 258],
            vec![352, 384, 320],
            vec![340, 360, 320],
        ]
    );
}

#[test]
fn latin_reverse_orders_balance_every_arm_and_position() {
    for arm_count in [3, 4] {
        let mut positions = vec![vec![0usize; arm_count]; arm_count];
        for window in 0..arm_count {
            let forward = latin_order(arm_count, window, false).unwrap();
            let reverse = latin_order(arm_count, window, true).unwrap();
            assert_eq!(reverse, forward.iter().rev().copied().collect::<Vec<_>>());
            for (position, arm) in forward.into_iter().enumerate() {
                positions[arm][position] += 1;
            }
        }
        assert!(positions.into_iter().flatten().all(|count| count == 1));
    }
    assert!(latin_order(2, 0, false).is_err());
    assert!(latin_order(5, 0, false).is_err());
}

#[test]
fn promotion_policy_requires_a_robust_one_percent_win() {
    assert!(promotion_eligible(1.010_001, 1.000_001));
    assert!(promotion_eligible(1.01, 1.000_001));
    assert!(!promotion_eligible(1.009_999, 1.000_001));
    assert!(!promotion_eligible(1.010_001, 1.0));
    assert!(!promotion_eligible(f64::NAN, 1.1));
}

#[test]
fn graph_replay_path_pair_schedule_is_fixed_exact_and_balanced() {
    assert_eq!(
        fixed_graph_replay_path_pair(false),
        [
            GraphReplayPath::Eager,
            GraphReplayPath::Graph,
            GraphReplayPath::Graph,
            GraphReplayPath::Eager,
        ]
    );
    assert_eq!(
        fixed_graph_replay_path_pair(true),
        [
            GraphReplayPath::Graph,
            GraphReplayPath::Eager,
            GraphReplayPath::Eager,
            GraphReplayPath::Graph,
        ]
    );
    for reverse in [false, true] {
        let schedule = fixed_graph_replay_path_pair(reverse);
        assert_eq!(
            schedule
                .iter()
                .filter(|path| **path == GraphReplayPath::Eager)
                .count(),
            2
        );
        assert_eq!(
            schedule
                .iter()
                .filter(|path| **path == GraphReplayPath::Graph)
                .count(),
            2
        );
    }
}

#[test]
fn split_graph_topology_descriptor_is_exact_and_alignment_sensitive() {
    let cell = Cell {
        name: "qualification",
        dims: (1_024, 47, 17),
        shape_class: "qualification",
        a_offset: 0,
        b_offset: 0,
        output_offset: 0,
    };
    let winner = plan("winner", 32, 32);
    let aligned = split_graph_topology_descriptor(cell, winner, true).unwrap();
    assert_eq!(aligned.nodes.len(), 2);
    assert_eq!(
        aligned.edge,
        GraphEdgeDescriptor {
            nodes: (0, 1),
            data: ORDINARY_GRAPH_EDGE_DATA,
        }
    );
    assert_eq!(
        aligned.nodes[0],
        GraphKernelDescriptor {
            symbol: "gemm_bi_tn_narrow_splitm_partial_aligned",
            grid: (1, 1, 32),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    );
    assert_eq!(
        aligned.nodes[1],
        GraphKernelDescriptor {
            symbol: "gemm_bi_splitm_reduce",
            grid: (4, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    );
    assert_eq!(
        split_graph_topology_descriptor(cell, winner, false)
            .unwrap()
            .nodes[0]
            .symbol,
        "gemm_bi_tn_narrow_splitm_partial"
    );
}

#[test]
fn ordinary_graph_edge_data_rejects_every_non_default_field() {
    require_ordinary_graph_edge_data(ORDINARY_GRAPH_EDGE_DATA, "ordinary").unwrap();
    for observed in [
        GraphEdgeDataDescriptor {
            from_port: 1,
            ..ORDINARY_GRAPH_EDGE_DATA
        },
        GraphEdgeDataDescriptor {
            to_port: 1,
            ..ORDINARY_GRAPH_EDGE_DATA
        },
        GraphEdgeDataDescriptor {
            dependency_type: 1,
            ..ORDINARY_GRAPH_EDGE_DATA
        },
        GraphEdgeDataDescriptor {
            reserved: [0, 0, 1, 0, 0],
            ..ORDINARY_GRAPH_EDGE_DATA
        },
    ] {
        assert!(require_ordinary_graph_edge_data(observed, "ordinary").is_err());
    }
}

#[test]
fn output_red_zones_reject_prefix_and_suffix_corruption() {
    let layout = OutputGuardLayout {
        output_offset: 2,
        output_elements: 3,
        prefix_bits: 0x7fc0_51a7,
        suffix_bits: 0x7fc0_5aff,
    };
    let valid = [
        f32::from_bits(layout.prefix_bits),
        f32::from_bits(layout.prefix_bits),
        1.0,
        2.0,
        3.0,
        f32::from_bits(layout.suffix_bits),
        f32::from_bits(layout.suffix_bits),
    ];
    validate_output_red_zones(&valid, layout, "timed").unwrap();
    let mut corrupt_prefix = valid;
    corrupt_prefix[1] = 0.0;
    assert!(validate_output_red_zones(&corrupt_prefix, layout, "timed").is_err());
    let mut corrupt_suffix = valid;
    corrupt_suffix[5] = 0.0;
    assert!(validate_output_red_zones(&corrupt_suffix, layout, "timed").is_err());
}

#[cfg(feature = "cuda")]
mod cuda_suite {
    use std::ffi::CStr;
    use std::fmt::Write as _;
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write as _};
    use std::mem::size_of;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use super::common::gpu_quiet::QuietGpu;
    use super::{
        GraphEdgeDataDescriptor, GraphKernelDescriptor, GraphReplayPath, ORDINARY_GRAPH_EDGE_DATA,
        OutputGuardLayout, PARTITION_CELLS, PartitionCell, PartitionPlan, SCRATCH_LIMIT_F32,
        SplitGraphTopologyDescriptor, WINNER_ALIGNMENT_CELLS, WinnerAlignmentCell,
        fixed_graph_replay_path_pair, latin_order, partial_grid_blocks,
        require_ordinary_graph_edge_data, scratch_elements, split_graph_topology_descriptor,
        validate_output_red_zones, window_count, winner_order, winner_symbols, winner_window_count,
    };
    use cudarc::driver::{CudaGraph, PushKernelArg, sys};
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use sha2::{Digest as _, Sha256};

    const OUTPUT_ENV: &str = "MAMBA_RS_TN_NARROW_PARTITION_TOURNAMENT_JSONL";
    const FINAL_ENV: &str = "MAMBA_RS_TN_NARROW_PARTITION_TOURNAMENT_FINAL";
    const WINNER_OUTPUT_ENV: &str = "MAMBA_RS_TN_NARROW_WINNER_ALIGNMENT_JSONL";
    const WINNER_FINAL_ENV: &str = "MAMBA_RS_TN_NARROW_WINNER_ALIGNMENT_FINAL";
    const GRAPH_REPLAY_OUTPUT_ENV: &str = "MAMBA_RS_TN_NARROW_GRAPH_REPLAY_QUALIFICATION_JSONL";
    const SCHEMA: &str = "MambaBiExactF32TnNarrowPartitionTournamentV1";
    const WINNER_SCHEMA: &str = "MambaBiExactF32TnNarrowWinnerAlignmentTournamentV1";
    const WINNER_COMPLETION_SCHEMA: &str =
        "MambaBiExactF32TnNarrowWinnerAlignmentTournamentCompletionV1";
    const GRAPH_REPLAY_SCHEMA: &str =
        "MambaBiExactF32TnNarrowGraphReplayDeterministicQualificationV2";
    const DECISION: &str = "measurement_only_no_auto_admission";
    const SCHEDULE_REVISION: &str = "tn_narrow_partition_sweep_v1";
    const WINNER_SCHEDULE_REVISION: &str = "tn_narrow_winner_alignment_v1";
    const EAGER_REPEATS: usize = 10;
    const GRAPH_REPEATS: usize = 10;
    const WARMUPS: usize = 64;
    const PILOT_ITERATIONS: usize = 16;
    const TARGET_WINDOW_US: f64 = 5_000.0;
    const MAX_WINDOW_ITERATIONS: usize = 4_096;
    const GRAPH_REPLAY_GROUPS: usize = 101;
    const GRAPH_REPLAY_ITERATIONS: usize = 1_024;
    const GUARD_WORDS: usize = 8;
    const OUTPUT_PREFIX_GUARD_BITS: u32 = 0x7fc0_51a7;
    const OUTPUT_SUFFIX_GUARD_BITS: u32 = 0x7fc0_5aff;
    const A_PREFIX_GUARD_BITS: u32 = 0x7fc0_a11c;
    const B_PREFIX_GUARD_BITS: u32 = 0x7fc0_b17e;
    const INPUT_SUFFIX_GUARD_BITS: u32 = 0x7fc0_1aff;
    const WORKSPACE_SUFFIX_GUARD_BITS: u32 = 0x7fc0_caff;
    const RECORDS_BEFORE_COMPLETION: usize = PARTITION_CELLS.len() * 2;
    const TOURNAMENT_SOURCE: &str = include_str!("gemm_bi_tn_narrow_partition_tournament.rs");
    const CUDA_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar.cu");

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SplitRole {
        Baseline,
        Candidate,
    }

    impl SplitRole {
        const fn name(self) -> &'static str {
            match self {
                Self::Baseline => "baseline",
                Self::Candidate => "candidate",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Direct,
        Split {
            role: SplitRole,
            plan: PartitionPlan,
        },
    }

    impl Arm {
        const fn role(self) -> &'static str {
            match self {
                Self::Direct => "direct_control",
                Self::Split { role, .. } => role.name(),
            }
        }

        fn label(self) -> String {
            match self {
                Self::Direct => "direct".to_string(),
                Self::Split { role, plan } => format!(
                    "split_{}_{}_m{}_f{}",
                    role.name(),
                    plan.name,
                    plan.m_chunk,
                    plan.chunks
                ),
            }
        }

        const fn plan(self) -> Option<PartitionPlan> {
            match self {
                Self::Direct => None,
                Self::Split { plan, .. } => Some(plan),
            }
        }
    }

    fn arms(spec: PartitionCell) -> Vec<Arm> {
        let mut arms = Vec::with_capacity(spec.candidates.len() + 2);
        arms.push(Arm::Direct);
        arms.push(Arm::Split {
            role: SplitRole::Baseline,
            plan: spec.baseline,
        });
        arms.extend(spec.candidates.iter().copied().map(|plan| Arm::Split {
            role: SplitRole::Candidate,
            plan,
        }));
        arms
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
    }

    #[derive(Clone, Copy)]
    struct CohortSpec {
        path: PathKind,
        windows: usize,
    }

    struct HostData {
        x: Vec<f32>,
        dy: Vec<f32>,
        seed: Vec<f32>,
    }

    impl HostData {
        fn new(spec: PartitionCell) -> Self {
            Self::new_cell(spec.cell)
        }

        fn new_cell(cell: super::Cell) -> Self {
            let (m, k, n) = cell.dims;
            let salt = (m as u64).rotate_left(7) ^ (k as u64).rotate_left(23) ^ n as u64;
            Self {
                x: values(m * k, salt ^ 0x415f_93a1_19c7_2d0b, 0.25),
                dy: values(m * n, salt ^ 0x7c31_d5e9_a821_4b73, 0.125),
                seed: values(k * n, salt ^ 0xd00d_ba5e_81f3_6c29, 0.03125),
            }
        }
    }

    struct ArmBuffer {
        arm: Arm,
        output: GpuBuffer,
        scratch: Option<GpuBuffer>,
        scratch_elements: usize,
    }

    struct CellBuffers {
        x: GpuBuffer,
        dy: GpuBuffer,
        seed: GpuBuffer,
        arms: Vec<ArmBuffer>,
        a_offset: usize,
        b_offset: usize,
        output_offset: usize,
        output_elements: usize,
    }

    impl CellBuffers {
        fn new(ctx: &GpuCtx, spec: PartitionCell, host: &HostData) -> Result<Self, String> {
            let arm_specs = arms(spec);
            let buffers = Self::new_with_arms(ctx, spec, host, &arm_specs)?;
            if !(3..=4).contains(&buffers.arms.len()) {
                return Err(format!(
                    "{} requires three or four tournament arms",
                    spec.cell.name
                ));
            }
            Ok(buffers)
        }

        fn new_with_arms(
            ctx: &GpuCtx,
            spec: PartitionCell,
            host: &HostData,
            arm_specs: &[Arm],
        ) -> Result<Self, String> {
            let cell = spec.cell;
            let (m, k, n) = cell.dims;
            let output_elements = k
                .checked_mul(n)
                .ok_or_else(|| "TN narrow output size overflows usize".to_string())?;
            let mut output_storage =
                vec![f32::from_bits(OUTPUT_PREFIX_GUARD_BITS); cell.output_offset];
            output_storage.extend_from_slice(&host.seed);
            output_storage.extend(std::iter::repeat_n(
                f32::from_bits(OUTPUT_SUFFIX_GUARD_BITS),
                GUARD_WORDS,
            ));
            let mut x_storage = vec![f32::from_bits(A_PREFIX_GUARD_BITS); cell.a_offset];
            x_storage.extend_from_slice(&host.x);
            x_storage.extend(std::iter::repeat_n(
                f32::from_bits(INPUT_SUFFIX_GUARD_BITS),
                GUARD_WORDS,
            ));
            let mut dy_storage = vec![f32::from_bits(B_PREFIX_GUARD_BITS); cell.b_offset];
            dy_storage.extend_from_slice(&host.dy);
            dy_storage.extend(std::iter::repeat_n(
                f32::from_bits(INPUT_SUFFIX_GUARD_BITS),
                GUARD_WORDS,
            ));
            if x_storage.len() != cell.a_offset + m * k + GUARD_WORDS
                || dy_storage.len() != cell.b_offset + m * n + GUARD_WORDS
            {
                return Err("TN narrow guarded input size mismatch".into());
            }

            let mut arm_buffers = Vec::with_capacity(arm_specs.len());
            for &arm in arm_specs {
                let (scratch, scratch_elements) = if let Some(plan) = arm.plan() {
                    let elements = scratch_elements(cell, plan)?;
                    if elements > SCRATCH_LIMIT_F32 {
                        return Err(format!(
                            "{} {} exceeds the TN narrow scratch cap",
                            cell.name,
                            arm.label()
                        ));
                    }
                    let mut storage = vec![0.0f32; elements];
                    storage.extend(std::iter::repeat_n(
                        f32::from_bits(WORKSPACE_SUFFIX_GUARD_BITS),
                        GUARD_WORDS,
                    ));
                    (Some(GpuBuffer::from_cpu(&ctx.stream, &storage)?), elements)
                } else {
                    (None, 0)
                };
                arm_buffers.push(ArmBuffer {
                    arm,
                    output: GpuBuffer::from_cpu(&ctx.stream, &output_storage)?,
                    scratch,
                    scratch_elements,
                });
            }
            Ok(Self {
                x: GpuBuffer::from_cpu(&ctx.stream, &x_storage)?,
                dy: GpuBuffer::from_cpu(&ctx.stream, &dy_storage)?,
                seed: GpuBuffer::from_cpu(&ctx.stream, &output_storage)?,
                arms: arm_buffers,
                a_offset: cell.a_offset,
                b_offset: cell.b_offset,
                output_offset: cell.output_offset,
                output_elements,
            })
        }

        fn arm(&self, index: usize) -> Result<&ArmBuffer, String> {
            self.arms
                .get(index)
                .ok_or_else(|| format!("TN narrow arm index {index} is out of range"))
        }

        fn output_ptr(&self, index: usize) -> Result<u64, String> {
            Ok(self.arm(index)?.output.cached_ptr()
                + (self.output_offset * size_of::<f32>()) as u64)
        }

        fn scratch_ptr(&self, index: usize) -> Result<u64, String> {
            self.arm(index)?
                .scratch
                .as_ref()
                .map(GpuBuffer::cached_ptr)
                .ok_or_else(|| format!("TN narrow arm {index} has no split scratch"))
        }

        fn a_ptr(&self) -> u64 {
            self.x.cached_ptr() + (self.a_offset * size_of::<f32>()) as u64
        }

        fn b_ptr(&self) -> u64 {
            self.dy.cached_ptr() + (self.b_offset * size_of::<f32>()) as u64
        }

        fn reset(&mut self, ctx: &GpuCtx, index: usize) -> Result<(), String> {
            self.arms[index]
                .output
                .copy_from_raw(&self.seed, &ctx.stream)
        }

        fn reset_all(&mut self, ctx: &GpuCtx) -> Result<(), String> {
            for index in 0..self.arms.len() {
                self.reset(ctx, index)?;
            }
            Ok(())
        }

        fn output_guard_layout(&self) -> OutputGuardLayout {
            OutputGuardLayout {
                output_offset: self.output_offset,
                output_elements: self.output_elements,
                prefix_bits: OUTPUT_PREFIX_GUARD_BITS,
                suffix_bits: OUTPUT_SUFFIX_GUARD_BITS,
            }
        }

        fn validate_output_guards(&self, ctx: &GpuCtx, index: usize) -> Result<(), String> {
            let arm = self.arm(index)?;
            let storage = arm.output.to_cpu(&ctx.stream)?;
            validate_output_red_zones(&storage, self.output_guard_layout(), &arm.arm.label())
        }

        fn output_values(&self, ctx: &GpuCtx, index: usize) -> Result<Vec<f32>, String> {
            let arm = self.arm(index)?;
            let storage = arm.output.to_cpu(&ctx.stream)?;
            validate_output_red_zones(&storage, self.output_guard_layout(), &arm.arm.label())?;
            let end = self
                .output_offset
                .checked_add(self.output_elements)
                .ok_or_else(|| "TN narrow output slice overflows usize".to_string())?;
            Ok(storage[self.output_offset..end].to_vec())
        }

        fn validate_offsets(&self, spec: PartitionCell) -> Result<(), String> {
            if !self.x.cached_ptr().is_multiple_of(16)
                || !self.dy.cached_ptr().is_multiple_of(16)
                || self
                    .arms
                    .iter()
                    .any(|arm| !arm.output.cached_ptr().is_multiple_of(16))
            {
                return Err("CUDA allocation bases are not 16-byte aligned".into());
            }
            for (label, offset, pointer) in [
                ("A", spec.cell.a_offset, self.a_ptr()),
                ("B", spec.cell.b_offset, self.b_ptr()),
            ] {
                if (offset == 0) != pointer.is_multiple_of(16) {
                    return Err(format!(
                        "{label} offset {offset} produced unexpected pointer alignment"
                    ));
                }
            }
            for index in 0..self.arms.len() {
                let pointer = self.output_ptr(index)?;
                if (spec.cell.output_offset == 0) != pointer.is_multiple_of(16) {
                    return Err(format!(
                        "{} output offset produced unexpected pointer alignment",
                        self.arm(index)?.arm.label()
                    ));
                }
            }
            Ok(())
        }

        fn validate_guards(&self, ctx: &GpuCtx, spec: PartitionCell) -> Result<(), String> {
            let (m, k, n) = spec.cell.dims;
            let x = self.x.to_cpu(&ctx.stream)?;
            if x[..self.a_offset]
                .iter()
                .any(|value| value.to_bits() != A_PREFIX_GUARD_BITS)
            {
                return Err("A prefix guard changed".into());
            }
            let x_end = self
                .a_offset
                .checked_add(m.checked_mul(k).ok_or("A size overflow")?)
                .ok_or("A guard boundary overflow")?;
            if x[x_end..]
                .iter()
                .any(|value| value.to_bits() != INPUT_SUFFIX_GUARD_BITS)
            {
                return Err("A suffix guard changed".into());
            }
            let dy = self.dy.to_cpu(&ctx.stream)?;
            if dy[..self.b_offset]
                .iter()
                .any(|value| value.to_bits() != B_PREFIX_GUARD_BITS)
            {
                return Err("B prefix guard changed".into());
            }
            let dy_end = self
                .b_offset
                .checked_add(m.checked_mul(n).ok_or("B size overflow")?)
                .ok_or("B guard boundary overflow")?;
            if dy[dy_end..]
                .iter()
                .any(|value| value.to_bits() != INPUT_SUFFIX_GUARD_BITS)
            {
                return Err("B suffix guard changed".into());
            }
            for index in 0..self.arms.len() {
                self.output_values(ctx, index)?;
                let arm = self.arm(index)?;
                if let Some(scratch) = &arm.scratch {
                    let values = scratch.to_cpu(&ctx.stream)?;
                    if values[arm.scratch_elements..]
                        .iter()
                        .any(|value| value.to_bits() != WORKSPACE_SUFFIX_GUARD_BITS)
                    {
                        return Err(format!("{} scratch suffix guard changed", arm.arm.label()));
                    }
                }
            }
            Ok(())
        }
    }

    struct CapturedArms {
        graphs: Vec<CudaGraph>,
    }

    impl CapturedArms {
        fn graph(&self, index: usize) -> Result<&CudaGraph, String> {
            self.graphs
                .get(index)
                .ok_or_else(|| format!("TN narrow graph index {index} is out of range"))
        }
    }

    struct CorrectnessEvidence {
        reference_bits: Vec<Vec<u32>>,
        output_digests: Vec<[u8; 32]>,
        oracle_digests: Vec<[u8; 32]>,
        direct_matches_split: Vec<bool>,
    }

    fn values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let centered = ((state >> 40) as i32 - (1 << 23)) as f32;
                centered * (scale / (1 << 23) as f32)
            })
            .collect()
    }

    fn grid_dims(spec: PartitionCell) -> Result<(u32, u32), String> {
        let (_, k, n) = spec.cell.dims;
        let grid_m = u32::try_from(k)
            .map_err(|_| "TN narrow K exceeds u32::MAX")?
            .div_ceil(64);
        let grid_n = u32::try_from(n)
            .map_err(|_| "TN narrow N exceeds u32::MAX")?
            .div_ceil(32);
        Ok((grid_m, grid_n))
    }

    fn launch_arm(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
        index: usize,
    ) -> Result<(), String> {
        let (m, k, n) = spec.cell.dims;
        let m_i = i32::try_from(m).map_err(|_| "TN narrow M exceeds i32::MAX")?;
        let k_i = i32::try_from(k).map_err(|_| "TN narrow K exceeds i32::MAX")?;
        let n_i = i32::try_from(n).map_err(|_| "TN narrow N exceeds i32::MAX")?;
        let (grid_m, grid_n) = grid_dims(spec)?;
        let output = buffers.output_ptr(index)?;
        let a = buffers.a_ptr();
        let b = buffers.b_ptr();
        let alpha = 1.0f32;
        match buffers.arm(index)?.arm {
            Arm::Direct => {
                let config = cudarc::driver::LaunchConfig {
                    grid_dim: (
                        grid_m
                            .checked_mul(grid_n)
                            .ok_or_else(|| "TN narrow direct grid overflows u32".to_string())?,
                        1,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_tn_narrow);
                builder.arg(&output);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&alpha);
                builder.arg(&m_i);
                builder.arg(&k_i);
                builder.arg(&n_i);
                unsafe { builder.launch(config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch raw gemm_bi_tn_narrow: {error:?}"))
            }
            Arm::Split { plan, .. } => {
                let chunk_i = i32::try_from(plan.m_chunk)
                    .map_err(|_| "TN narrow M chunk exceeds i32::MAX")?;
                let chunks_i =
                    i32::try_from(plan.chunks).map_err(|_| "TN narrow chunks exceed i32::MAX")?;
                let chunks_u32 =
                    u32::try_from(plan.chunks).map_err(|_| "TN narrow chunks exceed u32::MAX")?;
                let scratch = buffers.scratch_ptr(index)?;
                let partial_config = cudarc::driver::LaunchConfig {
                    grid_dim: (grid_m, grid_n, chunks_u32),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let partial = if a.is_multiple_of(16) && b.is_multiple_of(16) {
                    &ctx.kernels.gemm_bi_tn_narrow_splitm_partial_aligned
                } else {
                    &ctx.kernels.gemm_bi_tn_narrow_splitm_partial
                };
                let mut partial_builder = ctx.stream.launch_builder(partial);
                partial_builder.arg(&scratch);
                partial_builder.arg(&a);
                partial_builder.arg(&b);
                partial_builder.arg(&m_i);
                partial_builder.arg(&k_i);
                partial_builder.arg(&n_i);
                partial_builder.arg(&chunk_i);
                unsafe { partial_builder.launch(partial_config) }
                    .map_err(|error| format!("launch raw TN narrow partial: {error:?}"))?;

                let total = u32::try_from(
                    k.checked_mul(n)
                        .ok_or_else(|| "TN narrow reducer size overflows usize".to_string())?,
                )
                .map_err(|_| "TN narrow reducer size exceeds u32::MAX")?;
                let reduce_config = cudarc::driver::LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut reducer = ctx
                    .stream
                    .launch_builder(&ctx.kernels.gemm_bi_splitm_reduce);
                reducer.arg(&output);
                reducer.arg(&scratch);
                reducer.arg(&alpha);
                reducer.arg(&k_i);
                reducer.arg(&n_i);
                reducer.arg(&chunks_i);
                unsafe { reducer.launch(reduce_config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch raw gemm_bi_splitm_reduce: {error:?}"))
            }
        }
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation} failed: {result:?}"))
        }
    }

    struct ObservedGraphKernel {
        node: sys::CUgraphNode,
        symbol: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
    }

    fn observe_graph_kernel(
        node: sys::CUgraphNode,
        label: &str,
    ) -> Result<ObservedGraphKernel, String> {
        let mut node_type = std::mem::MaybeUninit::uninit();
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) },
            &format!("{label}: query graph node type"),
        )?;
        let node_type = unsafe { node_type.assume_init() };
        if node_type != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!(
                "{label}: graph contains non-kernel node {node_type:?}"
            ));
        }
        let mut params = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            &format!("{label}: query graph kernel parameters"),
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            &format!("{label}: query graph kernel function name"),
        )?;
        if name.is_null() {
            return Err(format!("{label}: graph kernel function name is null"));
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("{label}: graph kernel function name is not UTF-8: {error}"))?
            .to_owned();
        Ok(ObservedGraphKernel {
            node,
            symbol,
            grid: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block: (params.blockDimX, params.blockDimY, params.blockDimZ),
            shared_mem_bytes: params.sharedMemBytes,
        })
    }

    fn require_graph_kernel(
        observed: &ObservedGraphKernel,
        expected: GraphKernelDescriptor,
        label: &str,
    ) -> Result<(), String> {
        if observed.grid != expected.grid
            || observed.block != expected.block
            || observed.shared_mem_bytes != expected.shared_mem_bytes
        {
            return Err(format!(
                "{label}: {} launch differs: grid={:?} block={:?} shared={} expected grid={:?} block={:?} shared={}",
                observed.symbol,
                observed.grid,
                observed.block,
                observed.shared_mem_bytes,
                expected.grid,
                expected.block,
                expected.shared_mem_bytes,
            ));
        }
        Ok(())
    }

    fn validate_split_graph_topology(
        graph: &CudaGraph,
        expected: SplitGraphTopologyDescriptor,
        label: &str,
    ) -> Result<(), String> {
        let mut node_count = 0usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut node_count)
            },
            &format!("{label}: query graph node count"),
        )?;
        if node_count != expected.nodes.len() {
            return Err(format!(
                "{label}: graph contains {node_count} nodes, expected {}",
                expected.nodes.len()
            ));
        }
        let mut nodes = vec![std::ptr::null_mut(); node_count];
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut node_count) },
            &format!("{label}: query graph nodes"),
        )?;
        let mut matched = [None; 2];
        for node in nodes {
            let observed = observe_graph_kernel(node, label)?;
            let expected_index = expected
                .nodes
                .iter()
                .position(|candidate| candidate.symbol == observed.symbol)
                .ok_or_else(|| {
                    format!(
                        "{label}: unexpected graph kernel function {}",
                        observed.symbol
                    )
                })?;
            if matched[expected_index].replace(observed.node).is_some() {
                return Err(format!(
                    "{label}: duplicate graph kernel function {}",
                    observed.symbol
                ));
            }
            require_graph_kernel(&observed, expected.nodes[expected_index], label)?;
        }
        let partial = matched[expected.edge.nodes.0]
            .ok_or_else(|| format!("{label}: partial graph kernel is missing"))?;
        let reducer = matched[expected.edge.nodes.1]
            .ok_or_else(|| format!("{label}: reducer graph kernel is missing"))?;
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
            &format!("{label}: query graph edge count"),
        )?;
        if edge_count != 1 {
            return Err(format!(
                "{label}: graph contains {edge_count} edges, expected one"
            ));
        }
        let mut from = [std::ptr::null_mut()];
        let mut to = [std::ptr::null_mut()];
        let mut edge_data = [unsafe { std::mem::zeroed() }];
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
            &format!("{label}: query graph edge"),
        )?;
        if edge_count != 1 {
            return Err(format!(
                "{label}: graph edge query returned {edge_count} edges, expected one"
            ));
        }
        if from[0] != partial || to[0] != reducer {
            return Err(format!(
                "{label}: graph edge is not the exact partial-to-reducer dependency"
            ));
        }
        let observed_edge_data = GraphEdgeDataDescriptor {
            from_port: edge_data[0].from_port,
            to_port: edge_data[0].to_port,
            dependency_type: edge_data[0].type_,
            reserved: edge_data[0].reserved,
        };
        let driver_default = sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8;
        if expected.edge.data != ORDINARY_GRAPH_EDGE_DATA
            || expected.edge.data.dependency_type != driver_default
        {
            return Err(format!(
                "{label}: expected graph edge descriptor is not the CUDA default dependency"
            ));
        }
        require_ordinary_graph_edge_data(observed_edge_data, label)?;
        Ok(())
    }

    fn capture_arms(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
    ) -> Result<CapturedArms, String> {
        let mut graphs = Vec::with_capacity(buffers.arms.len());
        for index in 0..buffers.arms.len() {
            let graph = unsafe {
                capture_into_graph(&ctx.stream, || launch_arm(ctx, buffers, spec, index))
            }?;
            if let Arm::Split { plan, .. } = buffers.arm(index)?.arm {
                ctx.stream.synchronize().map_err(|error| {
                    format!("synchronize {} topology upload: {error:?}", plan.name)
                })?;
                let aligned_inputs =
                    buffers.a_ptr().is_multiple_of(16) && buffers.b_ptr().is_multiple_of(16);
                let expected = split_graph_topology_descriptor(spec.cell, plan, aligned_inputs)?;
                validate_split_graph_topology(&graph, expected, &buffers.arm(index)?.arm.label())?;
            }
            graphs.push(graph);
        }
        Ok(CapturedArms { graphs })
    }

    fn launch_path(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
        index: usize,
        path: PathKind,
        graphs: &CapturedArms,
    ) -> Result<(), String> {
        match path {
            PathKind::Eager => launch_arm(ctx, buffers, spec, index),
            PathKind::Graph => graphs
                .graph(index)?
                .launch()
                .map_err(|error| format!("launch graph for arm {index}: {error:?}")),
        }
    }

    fn direct_oracle(host: &HostData, spec: PartitionCell) -> Vec<u32> {
        let (m, k, n) = spec.cell.dims;
        let mut bits = Vec::with_capacity(k * n);
        for row in 0..k {
            for column in 0..n {
                let mut sum = 0.0f32;
                for reduction in 0..m {
                    sum = host.x[reduction * k + row].mul_add(host.dy[reduction * n + column], sum);
                }
                bits.push((host.seed[row * n + column] + sum).to_bits());
            }
        }
        bits
    }

    fn split_oracle(host: &HostData, spec: PartitionCell, plan: PartitionPlan) -> Vec<u32> {
        let (m, k, n) = spec.cell.dims;
        let mut bits = Vec::with_capacity(k * n);
        for row in 0..k {
            for column in 0..n {
                let mut partition_sum = 0.0f64;
                for partition in 0..plan.chunks {
                    let begin = partition * plan.m_chunk;
                    let end = (begin + plan.m_chunk).min(m);
                    let mut partial = 0.0f32;
                    for reduction in begin..end {
                        partial = host.x[reduction * k + row]
                            .mul_add(host.dy[reduction * n + column], partial);
                    }
                    partition_sum += f64::from(partial);
                }
                bits.push((host.seed[row * n + column] + partition_sum as f32).to_bits());
            }
        }
        bits
    }

    fn scheduled_oracles(host: &HostData, spec: PartitionCell, arm_specs: &[Arm]) -> Vec<Vec<u32>> {
        arm_specs
            .iter()
            .map(|arm| match arm {
                Arm::Direct => direct_oracle(host, spec),
                Arm::Split { plan, .. } => split_oracle(host, spec, *plan),
            })
            .collect()
    }

    fn output_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn require_expected_bits(label: &str, actual: &[u32], expected: &[u32]) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!(
                "{label} length differs: actual={} expected={}",
                actual.len(),
                expected.len()
            ));
        }
        if let Some(index) = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(format!(
                "{label} differs at {index}: expected={:#010x}, actual={:#010x}",
                expected[index], actual[index]
            ));
        }
        Ok(())
    }

    fn require_route_bits(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        references: &[Vec<u32>],
    ) -> Result<(), String> {
        for (index, expected) in references.iter().enumerate() {
            let actual = output_bits(&buffers.output_values(ctx, index)?);
            require_expected_bits(&buffers.arm(index)?.arm.label(), &actual, expected)?;
        }
        Ok(())
    }

    fn correctness_gate(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        host: &HostData,
        spec: PartitionCell,
    ) -> Result<(CorrectnessEvidence, CapturedArms), String> {
        buffers.validate_offsets(spec)?;
        let arm_specs = buffers.arms.iter().map(|arm| arm.arm).collect::<Vec<_>>();
        let oracles = scheduled_oracles(host, spec, &arm_specs);
        let mut references = Vec::with_capacity(arm_specs.len());
        for (index, arm) in arm_specs.iter().enumerate() {
            buffers.reset(ctx, index)?;
            launch_arm(ctx, buffers, spec, index)?;
            ctx.stream.synchronize().map_err(|error| {
                format!("synchronize {} correctness launch: {error:?}", arm.label())
            })?;
            let actual = output_bits(&buffers.output_values(ctx, index)?);
            require_expected_bits(&arm.label(), &actual, &oracles[index])?;
            references.push(actual);
        }

        for repeat in 0..EAGER_REPEATS {
            buffers.reset_all(ctx)?;
            let order = latin_order(arm_specs.len(), repeat, repeat % 2 != 0)?;
            for index in order {
                launch_arm(ctx, buffers, spec, index)?;
            }
            require_route_bits(ctx, buffers, &references)?;
        }

        let graphs = capture_arms(ctx, buffers, spec)?;
        for repeat in 0..GRAPH_REPEATS {
            buffers.reset_all(ctx)?;
            let order = latin_order(arm_specs.len(), repeat, repeat % 2 != 0)?;
            for index in order {
                graphs.graph(index)?.launch().map_err(|error| {
                    format!("{} graph replay: {error:?}", arm_specs[index].label())
                })?;
            }
            require_route_bits(ctx, buffers, &references)?;
        }
        buffers.validate_guards(ctx, spec)?;

        Ok((
            CorrectnessEvidence {
                output_digests: references.iter().map(|bits| bits_digest(bits)).collect(),
                oracle_digests: oracles.iter().map(|bits| bits_digest(bits)).collect(),
                direct_matches_split: references[1..]
                    .iter()
                    .map(|bits| bits == &references[0])
                    .collect(),
                reference_bits: references,
            },
            graphs,
        ))
    }

    struct OrderedSamples {
        arm_us: Vec<Vec<f64>>,
        direct_over_split: Vec<Vec<f64>>,
        baseline_over_candidate: Vec<Vec<f64>>,
    }

    impl OrderedSamples {
        fn with_capacity(arm_count: usize, windows: usize) -> Self {
            Self {
                arm_us: (0..arm_count)
                    .map(|_| Vec::with_capacity(windows))
                    .collect(),
                direct_over_split: (1..arm_count)
                    .map(|_| Vec::with_capacity(windows))
                    .collect(),
                baseline_over_candidate: (2..arm_count)
                    .map(|_| Vec::with_capacity(windows))
                    .collect(),
            }
        }

        fn push(&mut self, values: &[f64]) -> Result<(), String> {
            if values.len() != self.arm_us.len() {
                return Err(format!(
                    "timing row has {} arms, expected {}",
                    values.len(),
                    self.arm_us.len()
                ));
            }
            for (samples, value) in self.arm_us.iter_mut().zip(values.iter().copied()) {
                validate_positive_finite("timing sample", value)?;
                samples.push(value);
            }
            for (samples, split) in self
                .direct_over_split
                .iter_mut()
                .zip(values[1..].iter().copied())
            {
                samples.push(positive_ratio(values[0], split)?);
            }
            for (samples, candidate) in self
                .baseline_over_candidate
                .iter_mut()
                .zip(values[2..].iter().copied())
            {
                samples.push(positive_ratio(values[1], candidate)?);
            }
            Ok(())
        }
    }

    struct TournamentSamples {
        forward: OrderedSamples,
        reverse: OrderedSamples,
    }

    #[derive(Clone, Copy)]
    struct Percentiles {
        p05: f64,
        p50: f64,
        p95: f64,
    }

    struct OrderedSummary {
        arm_us: Vec<Percentiles>,
        direct_over_split: Vec<Percentiles>,
        baseline_over_candidate: Vec<Percentiles>,
    }

    struct TournamentSummary {
        forward: OrderedSummary,
        reverse: OrderedSummary,
    }

    struct CohortEvidence {
        calibration_preflight: String,
        timed_preflight: String,
        postflight: String,
        iterations: Vec<usize>,
    }

    struct WinnerOrderedSamples {
        baseline_us: Vec<f64>,
        winner_us: Vec<f64>,
        baseline_over_winner: Vec<f64>,
    }

    impl WinnerOrderedSamples {
        fn with_capacity(windows: usize) -> Self {
            Self {
                baseline_us: Vec::with_capacity(windows),
                winner_us: Vec::with_capacity(windows),
                baseline_over_winner: Vec::with_capacity(windows),
            }
        }

        fn push(&mut self, baseline: f64, winner: f64) -> Result<(), String> {
            validate_positive_finite("winner baseline sample", baseline)?;
            validate_positive_finite("winner candidate sample", winner)?;
            self.baseline_us.push(baseline);
            self.winner_us.push(winner);
            self.baseline_over_winner
                .push(positive_ratio(baseline, winner)?);
            Ok(())
        }
    }

    const fn winner_measurement_record_count() -> usize {
        WINNER_ALIGNMENT_CELLS.len() * 2
    }

    fn validate_positive_finite(label: &str, value: f64) -> Result<(), String> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(format!("{label} is not positive and finite: {value}"))
        }
    }

    fn positive_ratio(left: f64, right: f64) -> Result<f64, String> {
        let ratio = left / right;
        validate_positive_finite("paired ratio", ratio)?;
        Ok(ratio)
    }

    fn percentiles(values: &[f64], windows: usize) -> Result<Percentiles, String> {
        if values.len() != windows || windows < 3 {
            return Err(format!(
                "partition percentile requires exactly {windows} samples, received {}",
                values.len()
            ));
        }
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        let rank = |percent: usize| (percent * (windows - 1) + 50) / 100;
        Ok(Percentiles {
            p05: ordered[rank(5)],
            p50: ordered[rank(50)],
            p95: ordered[rank(95)],
        })
    }

    fn summarize_order(samples: &OrderedSamples, windows: usize) -> Result<OrderedSummary, String> {
        let summarize = |series: &[Vec<f64>]| {
            series
                .iter()
                .map(|values| percentiles(values, windows))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(OrderedSummary {
            arm_us: summarize(&samples.arm_us)?,
            direct_over_split: summarize(&samples.direct_over_split)?,
            baseline_over_candidate: summarize(&samples.baseline_over_candidate)?,
        })
    }

    fn summarize(samples: &TournamentSamples, windows: usize) -> Result<TournamentSummary, String> {
        Ok(TournamentSummary {
            forward: summarize_order(&samples.forward, windows)?,
            reverse: summarize_order(&samples.reverse, windows)?,
        })
    }

    fn measure_window(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
        index: usize,
        path: PathKind,
        graphs: &CapturedArms,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing window iterations must be positive".into());
        }
        let start = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record arm {index} timing start: {error:?}"))?;
        for _ in 0..iterations {
            launch_path(ctx, buffers, spec, index, path, graphs)?;
        }
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record arm {index} timing end: {error:?}"))?;
        let elapsed_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure arm {index} timing window: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        validate_positive_finite("timing sample", elapsed_us)?;
        Ok(elapsed_us)
    }

    fn warm_arm(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
        index: usize,
        path: PathKind,
        graphs: &CapturedArms,
    ) -> Result<(), String> {
        for _ in 0..WARMUPS {
            launch_path(ctx, buffers, spec, index, path, graphs)?;
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize arm {index} warmup: {error:?}"))
    }

    fn calibrate_arm(
        ctx: &GpuCtx,
        buffers: &CellBuffers,
        spec: PartitionCell,
        index: usize,
        path: PathKind,
        graphs: &CapturedArms,
    ) -> Result<usize, String> {
        let pilot = measure_window(ctx, buffers, spec, index, path, graphs, PILOT_ITERATIONS)?;
        Ok(((TARGET_WINDOW_US / pilot).ceil() as usize).clamp(1, MAX_WINDOW_ITERATIONS))
    }

    struct CohortContext<'a> {
        ctx: &'a GpuCtx,
        quiet: &'a QuietGpu,
        buffers: &'a mut CellBuffers,
        spec: PartitionCell,
        graphs: &'a CapturedArms,
        correctness: &'a CorrectnessEvidence,
    }

    fn measure_order(
        run: &CohortContext<'_>,
        path: PathKind,
        order: &[usize],
        iterations: &[usize],
    ) -> Result<Vec<f64>, String> {
        let mut values = vec![0.0; run.buffers.arms.len()];
        for &index in order {
            values[index] = measure_window(
                run.ctx,
                run.buffers,
                run.spec,
                index,
                path,
                run.graphs,
                iterations[index],
            )?;
        }
        Ok(values)
    }

    fn run_cohort(
        run: &mut CohortContext<'_>,
        cohort: CohortSpec,
    ) -> Result<(TournamentSamples, TournamentSummary, CohortEvidence), String> {
        let arm_count = run.buffers.arms.len();
        let label = format!(
            "tn-narrow-partition/{}/{}",
            run.spec.cell.name,
            cohort.path.name()
        );
        let calibration_preflight = run.quiet.require_cohort(&format!("{label}/calibration"))?;
        run.buffers.reset_all(run.ctx)?;
        for index in 0..arm_count {
            warm_arm(
                run.ctx,
                run.buffers,
                run.spec,
                index,
                cohort.path,
                run.graphs,
            )?;
        }
        let mut iterations = vec![0usize; arm_count];
        for (index, count) in iterations.iter_mut().enumerate() {
            *count = calibrate_arm(
                run.ctx,
                run.buffers,
                run.spec,
                index,
                cohort.path,
                run.graphs,
            )?;
        }
        let timed_preflight = run.quiet.require_cohort(&format!("{label}/timed"))?;
        run.buffers.reset_all(run.ctx)?;
        let mut samples = TournamentSamples {
            forward: OrderedSamples::with_capacity(arm_count, cohort.windows),
            reverse: OrderedSamples::with_capacity(arm_count, cohort.windows),
        };
        for window in 0..cohort.windows {
            let forward_order = latin_order(arm_count, window, false)?;
            let reverse_order = latin_order(arm_count, window, true)?;
            let forward = measure_order(run, cohort.path, &forward_order, &iterations)?;
            samples.forward.push(&forward)?;
            let reverse = measure_order(run, cohort.path, &reverse_order, &iterations)?;
            samples.reverse.push(&reverse)?;
        }
        let postflight = run.quiet.verify_post_cohort(&format!("{label}/post"))?;
        run.buffers.reset_all(run.ctx)?;
        for index in 0..arm_count {
            launch_path(
                run.ctx,
                run.buffers,
                run.spec,
                index,
                cohort.path,
                run.graphs,
            )?;
        }
        require_route_bits(run.ctx, run.buffers, &run.correctness.reference_bits)?;
        run.buffers.validate_guards(run.ctx, run.spec)?;
        let summary = summarize(&samples, cohort.windows)?;
        Ok((
            samples,
            summary,
            CohortEvidence {
                calibration_preflight,
                timed_preflight,
                postflight,
                iterations,
            },
        ))
    }

    struct RunMetadata {
        run_identity: String,
        pre_context: String,
        gpu_uuid: String,
        compute_capability: (u32, u32),
        multiprocessors: u32,
        driver_api_version: i32,
        driver_build_digest: String,
        nvrtc_version: (i32, i32),
        nvrtc_target: String,
        scalar_compile_key: String,
        scalar_artifact: String,
        scalar_invocation: String,
        tournament_source_sha256: String,
        cuda_source_sha256: String,
        executable_sha256: String,
    }

    impl RunMetadata {
        fn new(
            ctx: &GpuCtx,
            device: &GpuDevice,
            quiet: &QuietGpu,
            pre_context: String,
        ) -> Result<Self, String> {
            let compiler = ctx.kernels.triad_scalar_compiler_identity();
            let artifact = ctx.kernels.artifact_set_identity().triad_scalar;
            let identity = device.identity();
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
            let tournament_source_sha256 = verify_compiled_source(
                &manifest.join("tests/gemm_bi_tn_narrow_partition_tournament.rs"),
                TOURNAMENT_SOURCE,
            )?;
            let cuda_source_sha256 = verify_compiled_source(
                &manifest.join("kernels/gemm_bi_triad/scalar.cu"),
                CUDA_SOURCE,
            )?;
            let executable_sha256 = sha256_file(
                &std::env::current_exe()
                    .map_err(|error| format!("resolve partition executable: {error}"))?,
            )?;
            let mut run = Sha256::new();
            run.update(b"tn-narrow-partition-tournament-run.v1");
            run.update(SCHEDULE_REVISION.as_bytes());
            run.update(compiler.source_digest);
            run.update(compiler.invocation_digest);
            run.update(artifact.compile_key);
            run.update(artifact.artifact_digest);
            run.update(identity.compute_capability.0.to_le_bytes());
            run.update(identity.compute_capability.1.to_le_bytes());
            run.update(identity.multiprocessor_count.to_le_bytes());
            run.update(identity.driver.api_version.to_le_bytes());
            run.update(identity.driver.build_digest);
            run.update(tournament_source_sha256.as_bytes());
            run.update(executable_sha256.as_bytes());
            Ok(Self {
                run_identity: format!("{:x}", run.finalize()),
                pre_context,
                gpu_uuid: quiet.uuid().to_owned(),
                compute_capability: identity.compute_capability,
                multiprocessors: identity.multiprocessor_count,
                driver_api_version: identity.driver.api_version,
                driver_build_digest: digest_hex(&identity.driver.build_digest),
                nvrtc_version: compiler.nvrtc_version,
                nvrtc_target: compiler.target.as_str().to_owned(),
                scalar_compile_key: digest_hex(&artifact.compile_key),
                scalar_artifact: digest_hex(&artifact.artifact_digest),
                scalar_invocation: digest_hex(&compiler.invocation_digest),
                tournament_source_sha256,
                cuda_source_sha256,
                executable_sha256,
            })
        }
    }

    struct PhysicalRoute {
        symbols: Vec<&'static str>,
        digest: [u8; 32],
    }

    impl PhysicalRoute {
        fn new(
            metadata: &RunMetadata,
            spec: PartitionCell,
            buffers: &CellBuffers,
            index: usize,
        ) -> Result<Self, String> {
            let arm = buffers.arm(index)?.arm;
            let aligned_inputs =
                buffers.a_ptr().is_multiple_of(16) && buffers.b_ptr().is_multiple_of(16);
            let symbols = match arm {
                Arm::Direct => vec!["gemm_bi_tn_narrow"],
                Arm::Split { .. } if aligned_inputs => vec![
                    "gemm_bi_tn_narrow_splitm_partial_aligned",
                    "gemm_bi_splitm_reduce",
                ],
                Arm::Split { .. } => {
                    vec!["gemm_bi_tn_narrow_splitm_partial", "gemm_bi_splitm_reduce"]
                }
            };
            let mut digest = Sha256::new();
            digest.update(b"tn-narrow-partition-route.v1");
            digest.update(SCHEDULE_REVISION.as_bytes());
            digest.update(metadata.scalar_artifact.as_bytes());
            digest.update(metadata.scalar_invocation.as_bytes());
            for symbol in &symbols {
                digest.update((symbol.len() as u64).to_le_bytes());
                digest.update(symbol.as_bytes());
            }
            for value in [spec.cell.dims.0, spec.cell.dims.1, spec.cell.dims.2] {
                digest.update(value.to_le_bytes());
            }
            for value in [
                spec.cell.a_offset,
                spec.cell.b_offset,
                spec.cell.output_offset,
            ] {
                digest.update(value.to_le_bytes());
            }
            if let Some(plan) = arm.plan() {
                digest.update(plan.m_chunk.to_le_bytes());
                digest.update(plan.chunks.to_le_bytes());
            }
            Ok(Self {
                symbols,
                digest: digest.finalize().into(),
            })
        }
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
            Self::create(PathBuf::from(value))
        }

        fn create(path: PathBuf) -> Result<Self, String> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| {
                    format!("create TN narrow partition evidence {path:?}: {error}")
                })?;
            if !file
                .metadata()
                .map_err(|error| format!("inspect TN narrow partition evidence {path:?}: {error}"))?
                .file_type()
                .is_file()
            {
                return Err(format!(
                    "TN narrow partition evidence {path:?} is not a regular file"
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
                format!(
                    "write TN narrow partition evidence {:?}: {error}",
                    self.path
                )
            })?;
            self.digest.update(&bytes);
            self.records += 1;
            Ok(())
        }

        fn finish(mut self, windows: usize) -> Result<(), String> {
            if self.records != RECORDS_BEFORE_COMPLETION {
                return Err(format!(
                    "TN narrow partition evidence requires {RECORDS_BEFORE_COMPLETION} measurement records, received {}",
                    self.records
                ));
            }
            let content_digest = format!("{:x}", self.digest.clone().finalize());
            self.write(format!(
                concat!(
                    "{{\"schema\":\"{}\",\"record_type\":\"completion\",",
                    "\"records_before_completion\":{},\"records_total\":{},",
                    "\"windows_per_order\":{},\"content_sha256_before_completion\":\"{}\",",
                    "\"dataset_complete\":true,\"measurement_only\":true,",
                    "\"runtime_autotuning\":false,\"promotion_evaluated\":false,",
                    "\"auto_admitted\":false,\"decision\":\"{}\"}}"
                ),
                SCHEMA,
                RECORDS_BEFORE_COMPLETION,
                RECORDS_BEFORE_COMPLETION + 1,
                windows,
                content_digest,
                DECISION,
            ))?;
            self.writer.flush().map_err(|error| {
                format!(
                    "flush TN narrow partition evidence {:?}: {error}",
                    self.path
                )
            })?;
            self.writer.get_ref().sync_all().map_err(|error| {
                format!("sync TN narrow partition evidence {:?}: {error}", self.path)
            })?;
            eprintln!(
                "TN narrow partition tournament wrote {} records to {:?}",
                self.records, self.path
            );
            Ok(())
        }
    }

    struct WinnerJsonlSink {
        path: PathBuf,
        writer: BufWriter<File>,
        digest: Sha256,
        records: usize,
    }

    impl WinnerJsonlSink {
        fn create_from_env() -> Result<Self, String> {
            let value = std::env::var_os(WINNER_OUTPUT_ENV)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("{WINNER_OUTPUT_ENV} must name a new JSONL file"))?;
            Self::create(PathBuf::from(value))
        }

        fn create(path: PathBuf) -> Result<Self, String> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| {
                    format!("create TN narrow winner-alignment evidence {path:?}: {error}")
                })?;
            if !file
                .metadata()
                .map_err(|error| {
                    format!("inspect TN narrow winner-alignment evidence {path:?}: {error}")
                })?
                .file_type()
                .is_file()
            {
                return Err(format!(
                    "TN narrow winner-alignment evidence {path:?} is not a regular file"
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
                format!(
                    "write TN narrow winner-alignment evidence {:?}: {error}",
                    self.path
                )
            })?;
            self.digest.update(&bytes);
            self.records += 1;
            Ok(())
        }

        fn finish(mut self, windows: usize) -> Result<(), String> {
            let expected = winner_measurement_record_count();
            if self.records != expected {
                return Err(format!(
                    "TN narrow winner-alignment evidence requires {expected} measurement records, received {}",
                    self.records
                ));
            }
            let content_digest = format!("{:x}", self.digest.clone().finalize());
            self.write(format!(
                concat!(
                    "{{\"schema\":\"{}\",\"record_type\":\"completion\",",
                    "\"schedule_revision\":\"{}\",\"inventory_sha256\":\"{}\",",
                    "\"records_before_completion\":{},\"records_total\":{},",
                    "\"cells\":{},\"paths_per_cell\":2,\"windows_per_order\":{},",
                    "\"content_sha256_before_completion\":\"{}\",",
                    "\"dataset_complete\":true,\"measurement_only\":true,",
                    "\"runtime_autotuning\":false,\"promotion_evaluated\":false,",
                    "\"auto_admitted\":false,\"existing_seven_cell_suite_unchanged\":true,",
                    "\"decision\":\"{}\"}}"
                ),
                WINNER_COMPLETION_SCHEMA,
                WINNER_SCHEDULE_REVISION,
                winner_inventory_digest(),
                expected,
                expected + 1,
                WINNER_ALIGNMENT_CELLS.len(),
                windows,
                content_digest,
                DECISION,
            ))?;
            self.writer.flush().map_err(|error| {
                format!(
                    "flush TN narrow winner-alignment evidence {:?}: {error}",
                    self.path
                )
            })?;
            self.writer.get_ref().sync_all().map_err(|error| {
                format!(
                    "sync TN narrow winner-alignment evidence {:?}: {error}",
                    self.path
                )
            })
        }
    }

    struct MeasurementRecord<'a> {
        metadata: &'a RunMetadata,
        spec: PartitionCell,
        buffers: &'a CellBuffers,
        path: PathKind,
        windows: usize,
        correctness: &'a CorrectnessEvidence,
        samples: &'a TournamentSamples,
        summary: &'a TournamentSummary,
        cohort: &'a CohortEvidence,
    }

    fn emit_measurement(sink: &mut JsonlSink, record: MeasurementRecord<'_>) -> Result<(), String> {
        let metadata = record.metadata;
        let cell = record.spec.cell;
        let (m, k, n) = cell.dims;
        let arm_specs = record
            .buffers
            .arms
            .iter()
            .map(|buffer| buffer.arm)
            .collect::<Vec<_>>();
        let routes = (0..arm_specs.len())
            .map(|index| PhysicalRoute::new(metadata, record.spec, record.buffers, index))
            .collect::<Result<Vec<_>, _>>()?;
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"partition_measurement\",",
                "\"run_identity\":\"{}\",\"schedule_revision\":\"{}\",",
                "\"cell\":\"{}\",\"shape_class\":\"{}\",",
                "\"m\":{},\"k\":{},\"n\":{},\"a_offset_f32\":{},",
                "\"b_offset_f32\":{},\"output_offset_f32\":{},",
                "\"path\":\"{}\",\"windows_per_order\":{},",
                "\"order_policy\":\"latin_rotation_with_per_window_reverse\",",
                "\"raw_physical_launches_only\":true,\"runtime_autotuning\":false,",
                "\"measurement_only\":true,\"auto_admitted\":false,",
                "\"seeded_nonzero_dw\":true,",
                "\"correctness_oracle\":\"per_arm_cpu_direct_f32_or_splitm_partition_f32_reduce_f64\",",
                "\"oracle_bit_exact\":true,\"route_self_repeat_bit_exact\":true,",
                "\"eager_graph_bit_exact\":true,\"eager_repeats\":{},",
                "\"graph_repeats\":{},\"scratch_cap_f32\":{},",
                "\"promotion_policy\":{{\"median_ratio_at_least\":1.01,",
                "\"paired_p05_strictly_above\":1.0,\"evaluated\":false}},"
            ),
            SCHEMA,
            metadata.run_identity,
            SCHEDULE_REVISION,
            cell.name,
            cell.shape_class,
            m,
            k,
            n,
            cell.a_offset,
            cell.b_offset,
            cell.output_offset,
            record.path.name(),
            record.windows,
            EAGER_REPEATS,
            GRAPH_REPEATS,
            SCRATCH_LIMIT_F32,
        )
        .expect("String writes cannot fail");
        json.push_str("\"arms\":[");
        for index in 0..arm_specs.len() {
            if index != 0 {
                json.push(',');
            }
            let arm = arm_specs[index];
            let plan = arm.plan();
            let forward_direct = index
                .checked_sub(1)
                .map(|ratio_index| record.summary.forward.direct_over_split[ratio_index]);
            let reverse_direct = index
                .checked_sub(1)
                .map(|ratio_index| record.summary.reverse.direct_over_split[ratio_index]);
            let forward_baseline = index
                .checked_sub(2)
                .map(|ratio_index| record.summary.forward.baseline_over_candidate[ratio_index]);
            let reverse_baseline = index
                .checked_sub(2)
                .map(|ratio_index| record.summary.reverse.baseline_over_candidate[ratio_index]);
            let forward_direct_samples = index
                .checked_sub(1)
                .map(|ratio_index| &record.samples.forward.direct_over_split[ratio_index][..]);
            let reverse_direct_samples = index
                .checked_sub(1)
                .map(|ratio_index| &record.samples.reverse.direct_over_split[ratio_index][..]);
            let forward_baseline_samples = index.checked_sub(2).map(|ratio_index| {
                &record.samples.forward.baseline_over_candidate[ratio_index][..]
            });
            let reverse_baseline_samples = index.checked_sub(2).map(|ratio_index| {
                &record.samples.reverse.baseline_over_candidate[ratio_index][..]
            });
            write!(
                json,
                concat!(
                    "{{\"index\":{},\"label\":\"{}\",\"role\":\"{}\",",
                    "\"m_chunk\":{},\"chunks\":{},\"partial_grid_blocks\":{},",
                    "\"scratch_elements_f32\":{},\"physical_symbols\":{},",
                    "\"physical_route_sha256\":\"{}\",",
                    "\"output_sha256\":\"{}\",\"oracle_sha256\":\"{}\",",
                    "\"matches_direct_bits\":{},\"iterations\":{},",
                    "\"forward_us\":{},\"reverse_us\":{},",
                    "\"forward_direct_over_arm\":{},\"reverse_direct_over_arm\":{},",
                    "\"forward_baseline_over_candidate\":{},",
                    "\"reverse_baseline_over_candidate\":{},",
                    "\"forward_samples_us\":[{}],\"reverse_samples_us\":[{}],",
                    "\"forward_direct_over_arm_samples\":{},",
                    "\"reverse_direct_over_arm_samples\":{},",
                    "\"forward_baseline_over_candidate_samples\":{},",
                    "\"reverse_baseline_over_candidate_samples\":{}}}"
                ),
                index,
                escape_json(&arm.label()),
                arm.role(),
                render_optional_usize(plan.map(|plan| plan.m_chunk)),
                render_optional_usize(plan.map(|plan| plan.chunks)),
                render_optional_usize(
                    plan.map(|plan| partial_grid_blocks(cell, plan))
                        .transpose()?
                ),
                record.buffers.arm(index)?.scratch_elements,
                render_strings(&routes[index].symbols),
                digest_hex(&routes[index].digest),
                digest_hex(&record.correctness.output_digests[index]),
                digest_hex(&record.correctness.oracle_digests[index]),
                if index == 0 {
                    true
                } else {
                    record.correctness.direct_matches_split[index - 1]
                },
                record.cohort.iterations[index],
                render_percentiles(record.summary.forward.arm_us[index]),
                render_percentiles(record.summary.reverse.arm_us[index]),
                render_optional_percentiles(forward_direct),
                render_optional_percentiles(reverse_direct),
                render_optional_percentiles(forward_baseline),
                render_optional_percentiles(reverse_baseline),
                render_samples(&record.samples.forward.arm_us[index]),
                render_samples(&record.samples.reverse.arm_us[index]),
                render_optional_samples(forward_direct_samples),
                render_optional_samples(reverse_direct_samples),
                render_optional_samples(forward_baseline_samples),
                render_optional_samples(reverse_baseline_samples),
            )
            .expect("String writes cannot fail");
        }
        json.push_str("],");
        write!(
            json,
            concat!(
                "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
                "\"cuda_driver_api_version\":{},\"cuda_driver_build_sha256\":\"{}\",",
                "\"nvrtc_version\":\"{}.{}\",\"nvrtc_target\":\"{}\",",
                "\"scalar_compile_key\":\"{}\",\"scalar_artifact\":\"{}\",",
                "\"scalar_invocation\":\"{}\",",
                "\"tournament_source_sha256\":\"{}\",\"cuda_source_sha256\":\"{}\",",
                "\"executable_sha256\":\"{}\",\"pre_context\":\"{}\",",
                "\"calibration_preflight\":\"{}\",\"timed_preflight\":\"{}\",",
                "\"postflight\":\"{}\"}}"
            ),
            escape_json(&metadata.gpu_uuid),
            metadata.compute_capability.0,
            metadata.compute_capability.1,
            metadata.multiprocessors,
            metadata.driver_api_version,
            metadata.driver_build_digest,
            metadata.nvrtc_version.0,
            metadata.nvrtc_version.1,
            escape_json(&metadata.nvrtc_target),
            metadata.scalar_compile_key,
            metadata.scalar_artifact,
            metadata.scalar_invocation,
            metadata.tournament_source_sha256,
            metadata.cuda_source_sha256,
            metadata.executable_sha256,
            escape_json(&metadata.pre_context),
            escape_json(&record.cohort.calibration_preflight),
            escape_json(&record.cohort.timed_preflight),
            escape_json(&record.cohort.postflight),
        )
        .expect("String writes cannot fail");
        sink.write(json)
    }

    fn render_optional_usize(value: Option<usize>) -> String {
        value.map_or_else(|| "null".to_string(), |value| value.to_string())
    }

    fn render_percentiles(values: Percentiles) -> String {
        format!(
            "{{\"p05\":{:.9},\"p50\":{:.9},\"p95\":{:.9}}}",
            values.p05, values.p50, values.p95
        )
    }

    fn render_optional_percentiles(values: Option<Percentiles>) -> String {
        values.map_or_else(|| "null".to_string(), render_percentiles)
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

    fn render_optional_samples(values: Option<&[f64]>) -> String {
        values.map_or_else(
            || "null".to_string(),
            |values| format!("[{}]", render_samples(values)),
        )
    }

    fn render_strings(values: &[&str]) -> String {
        let mut rendered = String::from("[");
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            write!(rendered, "\"{}\"", escape_json(value)).expect("String writes cannot fail");
        }
        rendered.push(']');
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

    fn bits_digest(bits: &[u32]) -> [u8; 32] {
        let mut digest = Sha256::new();
        for value in bits {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }

    fn digest_hex(digest: &[u8; 32]) -> String {
        let mut rendered = String::with_capacity(64);
        for byte in digest {
            write!(rendered, "{byte:02x}").expect("String writes cannot fail");
        }
        rendered
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("read {path:?}: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn verify_compiled_source(path: &Path, compiled: &str) -> Result<String, String> {
        let compiled_digest = format!("{:x}", Sha256::digest(compiled.as_bytes()));
        let disk_digest = sha256_file(path)?;
        if compiled_digest != disk_digest {
            return Err(format!(
                "compiled source for {path:?} differs from the on-disk source"
            ));
        }
        Ok(compiled_digest)
    }

    fn configure_context(ctx: &GpuCtx) {
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    }

    fn qualification_device() -> Result<(QuietGpu, String, GpuDevice, GpuCtx), String> {
        if cfg!(debug_assertions) {
            return Err("TN narrow partition tournament requires --release".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let pre_context = quiet.require_pre_context("tn-narrow-partition/pre-context")?;
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "TN narrow partition matrix is scoped to CC 12.0 with 170 SMs, received CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        configure_context(&ctx);
        Ok((quiet, pre_context, device, ctx))
    }

    fn winner_partition_spec(cell: WinnerAlignmentCell) -> PartitionCell {
        PartitionCell {
            cell: cell.cell,
            baseline: cell.baseline,
            candidates: &[],
        }
    }

    fn winner_arms(cell: WinnerAlignmentCell) -> [Arm; 2] {
        [
            Arm::Split {
                role: SplitRole::Baseline,
                plan: cell.baseline,
            },
            Arm::Split {
                role: SplitRole::Candidate,
                plan: cell.winner,
            },
        ]
    }

    struct WinnerCorrectness {
        reference_bits: Vec<Vec<u32>>,
        output_digests: Vec<[u8; 32]>,
        oracle_digests: Vec<[u8; 32]>,
    }

    fn winner_correctness_gate(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        host: &HostData,
        cell: WinnerAlignmentCell,
    ) -> Result<(WinnerCorrectness, CapturedArms), String> {
        let spec = winner_partition_spec(cell);
        let arm_specs = winner_arms(cell);
        buffers.validate_offsets(spec)?;
        let oracles = arm_specs
            .iter()
            .map(|arm| split_oracle(host, spec, arm.plan().expect("winner arm is split")))
            .collect::<Vec<_>>();
        let mut references = Vec::with_capacity(arm_specs.len());
        for index in 0..arm_specs.len() {
            buffers.reset(ctx, index)?;
            launch_arm(ctx, buffers, spec, index)?;
            ctx.stream.synchronize().map_err(|error| {
                format!(
                    "synchronize {} winner correctness launch: {error:?}",
                    arm_specs[index].label()
                )
            })?;
            let actual = output_bits(&buffers.output_values(ctx, index)?);
            require_expected_bits(&arm_specs[index].label(), &actual, &oracles[index])?;
            references.push(actual);
        }
        for repeat in 0..EAGER_REPEATS {
            buffers.reset_all(ctx)?;
            for index in winner_order(repeat % 2 != 0) {
                launch_arm(ctx, buffers, spec, index)?;
            }
            require_route_bits(ctx, buffers, &references)?;
        }
        let graphs = capture_arms(ctx, buffers, spec)?;
        for repeat in 0..GRAPH_REPEATS {
            buffers.reset_all(ctx)?;
            for index in winner_order(repeat % 2 != 0) {
                graphs.graph(index)?.launch().map_err(|error| {
                    format!(
                        "{} winner graph replay: {error:?}",
                        arm_specs[index].label()
                    )
                })?;
            }
            require_route_bits(ctx, buffers, &references)?;
        }
        buffers.validate_guards(ctx, spec)?;
        Ok((
            WinnerCorrectness {
                output_digests: references.iter().map(|bits| bits_digest(bits)).collect(),
                oracle_digests: oracles.iter().map(|bits| bits_digest(bits)).collect(),
                reference_bits: references,
            },
            graphs,
        ))
    }

    struct WinnerSamples {
        ab: WinnerOrderedSamples,
        ba: WinnerOrderedSamples,
    }

    struct WinnerCohortEvidence {
        calibration_preflight: String,
        timed_preflight: String,
        postflight: String,
        iterations: [usize; 2],
    }

    struct WinnerCohortContext<'a> {
        ctx: &'a GpuCtx,
        quiet: &'a QuietGpu,
        buffers: &'a mut CellBuffers,
        cell: WinnerAlignmentCell,
        graphs: &'a CapturedArms,
        correctness: &'a WinnerCorrectness,
    }

    fn winner_measure_arm(
        run: &mut WinnerCohortContext<'_>,
        path: PathKind,
        index: usize,
        iterations: usize,
    ) -> Result<f64, String> {
        run.buffers.reset(run.ctx, index)?;
        measure_window(
            run.ctx,
            run.buffers,
            winner_partition_spec(run.cell),
            index,
            path,
            run.graphs,
            iterations,
        )
    }

    fn winner_measure_order(
        run: &mut WinnerCohortContext<'_>,
        path: PathKind,
        order: [usize; 2],
        iterations: [usize; 2],
    ) -> Result<[f64; 2], String> {
        let mut values = [0.0; 2];
        for index in order {
            values[index] = winner_measure_arm(run, path, index, iterations[index])?;
        }
        Ok(values)
    }

    fn run_winner_cohort(
        run: &mut WinnerCohortContext<'_>,
        cohort: CohortSpec,
    ) -> Result<(WinnerSamples, WinnerCohortEvidence), String> {
        let spec = winner_partition_spec(run.cell);
        let label = format!(
            "tn-narrow-winner-alignment/{}/{}",
            run.cell.cell.name,
            cohort.path.name()
        );
        let calibration_preflight = run.quiet.require_cohort(&format!("{label}/calibration"))?;
        for index in 0..2 {
            run.buffers.reset(run.ctx, index)?;
            warm_arm(run.ctx, run.buffers, spec, index, cohort.path, run.graphs)?;
        }
        let mut iterations = [0usize; 2];
        for (index, count) in iterations.iter_mut().enumerate() {
            *count = {
                let pilot = winner_measure_arm(run, cohort.path, index, PILOT_ITERATIONS)?;
                ((TARGET_WINDOW_US / pilot).ceil() as usize).clamp(1, MAX_WINDOW_ITERATIONS)
            };
        }
        let timed_preflight = run.quiet.require_cohort(&format!("{label}/timed"))?;
        let mut samples = WinnerSamples {
            ab: WinnerOrderedSamples::with_capacity(cohort.windows),
            ba: WinnerOrderedSamples::with_capacity(cohort.windows),
        };
        for _ in 0..cohort.windows {
            let ab = winner_measure_order(run, cohort.path, winner_order(false), iterations)?;
            samples.ab.push(ab[0], ab[1])?;
            let ba = winner_measure_order(run, cohort.path, winner_order(true), iterations)?;
            samples.ba.push(ba[0], ba[1])?;
        }
        run.buffers.validate_guards(run.ctx, spec)?;
        let postflight = run.quiet.verify_post_cohort(&format!("{label}/post"))?;
        run.buffers.reset_all(run.ctx)?;
        for index in 0..2 {
            launch_path(run.ctx, run.buffers, spec, index, cohort.path, run.graphs)?;
        }
        require_route_bits(run.ctx, run.buffers, &run.correctness.reference_bits)?;
        run.buffers.validate_guards(run.ctx, spec)?;
        Ok((
            samples,
            WinnerCohortEvidence {
                calibration_preflight,
                timed_preflight,
                postflight,
                iterations,
            },
        ))
    }

    fn winner_inventory_digest() -> String {
        let mut digest = Sha256::new();
        digest.update(b"tn-narrow-winner-alignment-inventory.v1");
        for cell in WINNER_ALIGNMENT_CELLS {
            digest.update((cell.cell.name.len() as u64).to_le_bytes());
            digest.update(cell.cell.name.as_bytes());
            for value in [
                cell.cell.dims.0,
                cell.cell.dims.1,
                cell.cell.dims.2,
                cell.cell.a_offset,
                cell.cell.b_offset,
                cell.cell.output_offset,
                cell.baseline.m_chunk,
                cell.baseline.chunks,
                cell.winner.m_chunk,
                cell.winner.chunks,
            ] {
                digest.update(value.to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    }

    fn winner_run_identity(metadata: &RunMetadata) -> String {
        let mut digest = Sha256::new();
        digest.update(b"tn-narrow-winner-alignment-run.v1");
        digest.update(WINNER_SCHEDULE_REVISION.as_bytes());
        digest.update(metadata.scalar_compile_key.as_bytes());
        digest.update(metadata.scalar_artifact.as_bytes());
        digest.update(metadata.scalar_invocation.as_bytes());
        digest.update(metadata.tournament_source_sha256.as_bytes());
        digest.update(metadata.executable_sha256.as_bytes());
        digest.update(winner_inventory_digest().as_bytes());
        format!("{:x}", digest.finalize())
    }

    fn winner_route_digest(
        metadata: &RunMetadata,
        cell: WinnerAlignmentCell,
        plan: PartitionPlan,
    ) -> String {
        let mut digest = Sha256::new();
        digest.update(b"tn-narrow-winner-alignment-route.v1");
        digest.update(metadata.scalar_artifact.as_bytes());
        for symbol in winner_symbols(cell) {
            digest.update((symbol.len() as u64).to_le_bytes());
            digest.update(symbol.as_bytes());
        }
        for value in [
            cell.cell.dims.0,
            cell.cell.dims.1,
            cell.cell.dims.2,
            cell.cell.a_offset,
            cell.cell.b_offset,
            cell.cell.output_offset,
            plan.m_chunk,
            plan.chunks,
        ] {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    struct WinnerMeasurementRecord<'a> {
        metadata: &'a RunMetadata,
        cell: WinnerAlignmentCell,
        path: PathKind,
        windows: usize,
        correctness: &'a WinnerCorrectness,
        samples: &'a WinnerSamples,
        cohort: &'a WinnerCohortEvidence,
    }

    fn render_winner_order(samples: &WinnerOrderedSamples) -> String {
        format!(
            concat!(
                "{{\"baseline_samples_us\":[{}],\"winner_samples_us\":[{}],",
                "\"baseline_over_winner\":[{}]}}"
            ),
            render_samples(&samples.baseline_us),
            render_samples(&samples.winner_us),
            render_samples(&samples.baseline_over_winner),
        )
    }

    fn emit_winner_measurement(
        sink: &mut WinnerJsonlSink,
        record: &WinnerMeasurementRecord<'_>,
    ) -> Result<(), String> {
        let metadata = record.metadata;
        let cell = record.cell;
        let (m, k, n) = cell.cell.dims;
        let symbols = winner_symbols(cell);
        let baseline_ratio = percentiles(&record.samples.ab.baseline_over_winner, record.windows)?;
        let reverse_ratio = percentiles(&record.samples.ba.baseline_over_winner, record.windows)?;
        sink.write(format!(
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"winner_alignment_measurement\",",
                "\"run_identity\":\"{}\",\"schedule_revision\":\"{}\",",
                "\"inventory_sha256\":\"{}\",\"cell\":\"{}\",",
                "\"alignment_class\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
                "\"a_offset_bytes\":{},\"b_offset_bytes\":{},\"output_offset_bytes\":{},",
                "\"path\":\"{}\",\"windows_per_order\":{},",
                "\"order_policy\":\"balanced_ab_ba\",\"raw_two_launch_only\":true,",
                "\"physical_symbols\":{},\"alpha\":1.0,\"beta\":1.0,",
                "\"seeded_nonzero_dw\":true,\"independent_per_plan_cpu_oracle\":true,",
                "\"oracle_bit_exact\":true,\"eager_graph_bit_exact\":true,",
                "\"red_zones_validated\":true,\"measurement_only\":true,",
                "\"runtime_autotuning\":false,\"auto_admitted\":false,",
                "\"baseline\":{{\"m_chunk\":{},\"chunks\":{},\"iterations\":{},",
                "\"output_sha256\":\"{}\",\"oracle_sha256\":\"{}\",",
                "\"route_sha256\":\"{}\"}},",
                "\"winner\":{{\"m_chunk\":{},\"chunks\":{},\"iterations\":{},",
                "\"output_sha256\":\"{}\",\"oracle_sha256\":\"{}\",",
                "\"route_sha256\":\"{}\"}},",
                "\"ab\":{},\"ba\":{},\"ab_ratio\":{},\"ba_ratio\":{},",
                "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
                "\"nvrtc_version\":\"{}.{}\",\"nvrtc_target\":\"{}\",",
                "\"scalar_compile_key\":\"{}\",\"scalar_artifact\":\"{}\",",
                "\"pre_context\":\"{}\",\"calibration_preflight\":\"{}\",",
                "\"timed_preflight\":\"{}\",\"postflight\":\"{}\"}}"
            ),
            WINNER_SCHEMA,
            winner_run_identity(metadata),
            WINNER_SCHEDULE_REVISION,
            winner_inventory_digest(),
            cell.cell.name,
            cell.alignment,
            m,
            k,
            n,
            cell.cell.a_offset * size_of::<f32>(),
            cell.cell.b_offset * size_of::<f32>(),
            cell.cell.output_offset * size_of::<f32>(),
            record.path.name(),
            record.windows,
            render_strings(&symbols),
            cell.baseline.m_chunk,
            cell.baseline.chunks,
            record.cohort.iterations[0],
            digest_hex(&record.correctness.output_digests[0]),
            digest_hex(&record.correctness.oracle_digests[0]),
            winner_route_digest(metadata, cell, cell.baseline),
            cell.winner.m_chunk,
            cell.winner.chunks,
            record.cohort.iterations[1],
            digest_hex(&record.correctness.output_digests[1]),
            digest_hex(&record.correctness.oracle_digests[1]),
            winner_route_digest(metadata, cell, cell.winner),
            render_winner_order(&record.samples.ab),
            render_winner_order(&record.samples.ba),
            render_percentiles(baseline_ratio),
            render_percentiles(reverse_ratio),
            escape_json(&metadata.gpu_uuid),
            metadata.compute_capability.0,
            metadata.compute_capability.1,
            metadata.multiprocessors,
            metadata.nvrtc_version.0,
            metadata.nvrtc_version.1,
            escape_json(&metadata.nvrtc_target),
            metadata.scalar_compile_key,
            metadata.scalar_artifact,
            escape_json(&metadata.pre_context),
            escape_json(&record.cohort.calibration_preflight),
            escape_json(&record.cohort.timed_preflight),
            escape_json(&record.cohort.postflight),
        ))
    }

    fn winner_qualification_device() -> Result<(QuietGpu, String, GpuDevice, GpuCtx), String> {
        if cfg!(debug_assertions) {
            return Err("TN narrow winner-alignment tournament requires --release".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let pre_context = quiet.require_pre_context("tn-narrow-winner-alignment/pre-context")?;
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "TN narrow winner-alignment matrix is scoped to CC 12.0 with 170 SMs, received CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        configure_context(&ctx);
        Ok((quiet, pre_context, device, ctx))
    }

    fn run_winner_alignment_tournament() -> Result<(), String> {
        let windows = winner_window_count(std::env::var(WINNER_FINAL_ENV).ok().as_deref())?;
        let (quiet, pre_context, device, ctx) = winner_qualification_device()?;
        let metadata = RunMetadata::new(&ctx, &device, &quiet, pre_context)?;
        let mut sink = WinnerJsonlSink::create_from_env()?;
        for cell in WINNER_ALIGNMENT_CELLS {
            let spec = winner_partition_spec(cell);
            let host = HostData::new_cell(cell.cell);
            let arm_specs = winner_arms(cell);
            let mut buffers = CellBuffers::new_with_arms(&ctx, spec, &host, &arm_specs)?;
            let (correctness, graphs) = winner_correctness_gate(&ctx, &mut buffers, &host, cell)?;
            for path in [PathKind::Eager, PathKind::Graph] {
                let mut run = WinnerCohortContext {
                    ctx: &ctx,
                    quiet: &quiet,
                    buffers: &mut buffers,
                    cell,
                    graphs: &graphs,
                    correctness: &correctness,
                };
                let (samples, cohort) = run_winner_cohort(&mut run, CohortSpec { path, windows })?;
                emit_winner_measurement(
                    &mut sink,
                    &WinnerMeasurementRecord {
                        metadata: &metadata,
                        cell,
                        path,
                        windows,
                        correctness: &correctness,
                        samples: &samples,
                        cohort: &cohort,
                    },
                )?;
            }
            ctx.stream.synchronize().map_err(|error| {
                format!(
                    "synchronize {} winner-alignment graphs before drop: {error:?}",
                    cell.cell.name
                )
            })?;
        }
        sink.finish(windows)
    }

    #[derive(Clone, Copy)]
    struct GraphReplayTiming {
        path: GraphReplayPath,
        cuda_event_us: f64,
        host_enqueue_us: f64,
    }

    struct GraphReplayGroup {
        egge: [GraphReplayTiming; 4],
        geeg: [GraphReplayTiming; 4],
    }

    const fn graph_replay_path_kind(path: GraphReplayPath) -> PathKind {
        match path {
            GraphReplayPath::Eager => PathKind::Eager,
            GraphReplayPath::Graph => PathKind::Graph,
        }
    }

    fn graph_replay_qualification_spec() -> PartitionCell {
        PartitionCell {
            cell: super::Cell {
                name: "m1024_k47_n17_graph_replay_qualification",
                dims: (1_024, 47, 17),
                shape_class: "graph_replay_deterministic_qualification",
                a_offset: 0,
                b_offset: 0,
                output_offset: 0,
            },
            baseline: super::plan("winner", 32, 32),
            candidates: &[],
        }
    }

    fn measure_graph_replay_path(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        spec: PartitionCell,
        graphs: &CapturedArms,
        path: GraphReplayPath,
    ) -> Result<GraphReplayTiming, String> {
        buffers.reset(ctx, 0)?;
        let start = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {} timing start: {error:?}", path.name()))?;
        let host_start = Instant::now();
        for _ in 0..GRAPH_REPLAY_ITERATIONS {
            launch_path(ctx, buffers, spec, 0, graph_replay_path_kind(path), graphs)?;
        }
        let host_enqueue_us =
            host_start.elapsed().as_secs_f64() * 1_000_000.0 / GRAPH_REPLAY_ITERATIONS as f64;
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {} timing end: {error:?}", path.name()))?;
        let cuda_event_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure {} CUDA events: {error:?}", path.name()))?,
        ) * 1_000.0
            / GRAPH_REPLAY_ITERATIONS as f64;
        buffers.validate_output_guards(ctx, 0)?;
        validate_positive_finite("graph replay CUDA-event sample", cuda_event_us)?;
        validate_positive_finite("graph replay host-enqueue sample", host_enqueue_us)?;
        Ok(GraphReplayTiming {
            path,
            cuda_event_us,
            host_enqueue_us,
        })
    }

    fn measure_graph_replay_order(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        spec: PartitionCell,
        graphs: &CapturedArms,
        reverse: bool,
    ) -> Result<[GraphReplayTiming; 4], String> {
        let mut samples = Vec::with_capacity(4);
        for path in fixed_graph_replay_path_pair(reverse) {
            samples.push(measure_graph_replay_path(ctx, buffers, spec, graphs, path)?);
        }
        samples
            .try_into()
            .map_err(|_| "graph replay path-pair schedule did not contain four samples".into())
    }

    fn warm_graph_replay_paths(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        spec: PartitionCell,
        graphs: &CapturedArms,
    ) -> Result<(), String> {
        for path in fixed_graph_replay_path_pair(false) {
            buffers.reset(ctx, 0)?;
            warm_arm(ctx, buffers, spec, 0, graph_replay_path_kind(path), graphs)?;
        }
        Ok(())
    }

    fn require_graph_replay_bits(
        ctx: &GpuCtx,
        buffers: &mut CellBuffers,
        spec: PartitionCell,
        graphs: &CapturedArms,
        path: GraphReplayPath,
        expected: &[u32],
    ) -> Result<(), String> {
        buffers.reset(ctx, 0)?;
        launch_path(ctx, buffers, spec, 0, graph_replay_path_kind(path), graphs)?;
        ctx.stream.synchronize().map_err(|error| {
            format!("synchronize {} correctness replay: {error:?}", path.name())
        })?;
        let actual = output_bits(&buffers.output_values(ctx, 0)?);
        require_expected_bits(path.name(), &actual, expected)
    }

    fn render_graph_replay_timing(sample: GraphReplayTiming) -> String {
        format!(
            concat!(
                "{{\"path\":\"{}\",\"cuda_event_us_per_iteration\":{:.9},",
                "\"host_enqueue_us_per_iteration\":{:.9}}}"
            ),
            sample.path.name(),
            sample.cuda_event_us,
            sample.host_enqueue_us,
        )
    }

    fn render_graph_replay_order(samples: &[GraphReplayTiming; 4]) -> String {
        samples
            .iter()
            .copied()
            .map(render_graph_replay_timing)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn render_graph_replay_groups(groups: &[GraphReplayGroup]) -> String {
        let mut rendered = String::new();
        for (index, group) in groups.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            write!(
                rendered,
                "{{\"index\":{index},\"egge\":[{}],\"geeg\":[{}]}}",
                render_graph_replay_order(&group.egge),
                render_graph_replay_order(&group.geeg),
            )
            .expect("String writes cannot fail");
        }
        rendered
    }

    fn graph_replay_run_identity(
        metadata: &RunMetadata,
        topology: SplitGraphTopologyDescriptor,
    ) -> String {
        let mut digest = Sha256::new();
        digest.update(b"tn-narrow-graph-replay-deterministic-qualification.v2");
        digest.update(metadata.scalar_compile_key.as_bytes());
        digest.update(metadata.scalar_artifact.as_bytes());
        digest.update(metadata.scalar_invocation.as_bytes());
        digest.update(metadata.tournament_source_sha256.as_bytes());
        digest.update(metadata.executable_sha256.as_bytes());
        digest.update(GRAPH_REPLAY_GROUPS.to_le_bytes());
        digest.update(GRAPH_REPLAY_ITERATIONS.to_le_bytes());
        for node in topology.nodes {
            digest.update(node.symbol.as_bytes());
            for dimension in [
                node.grid.0,
                node.grid.1,
                node.grid.2,
                node.block.0,
                node.block.1,
                node.block.2,
                node.shared_mem_bytes,
            ] {
                digest.update(dimension.to_le_bytes());
            }
        }
        digest.update((topology.edge.nodes.0 as u64).to_le_bytes());
        digest.update((topology.edge.nodes.1 as u64).to_le_bytes());
        digest.update([
            topology.edge.data.from_port,
            topology.edge.data.to_port,
            topology.edge.data.dependency_type,
        ]);
        digest.update(topology.edge.data.reserved);
        format!("{:x}", digest.finalize())
    }

    fn render_graph_replay_record(
        metadata: &RunMetadata,
        topology: SplitGraphTopologyDescriptor,
        groups: &[GraphReplayGroup],
        expected: &[u32],
        timed_preflight: &str,
        postflight: &str,
    ) -> String {
        let spec = graph_replay_qualification_spec();
        let plan = spec.baseline;
        format!(
            concat!(
                "{{\"schema\":\"{}\",\"record_type\":\"deterministic_qualification\",",
                "\"run_identity\":\"{}\",\"cell\":\"{}\",",
                "\"m\":{},\"k\":{},\"n\":{},\"m_chunk\":{},\"chunks\":{},",
                "\"groups\":{},\"fixed_equal_iterations\":{},",
                "\"egge_schedule\":[\"eager\",\"graph\",\"graph\",\"eager\"],",
                "\"geeg_schedule\":[\"graph\",\"eager\",\"eager\",\"graph\"],",
                "\"cuda_event_timing\":true,\"host_instant_timing\":true,",
                "\"host_timer_scope\":\"enqueue_loop_only\",",
                "\"topology_validated\":true,\"topology_node_count\":2,",
                "\"topology_edge\":\"partial_to_reducer\",",
                "\"topology_edge_data\":{{\"dependency_type\":\"default\",",
                "\"dependency_type_value\":{},\"from_port\":{},\"to_port\":{},",
                "\"reserved\":[{},{},{},{},{}]}},",
                "\"physical_symbols\":[\"{}\",\"{}\"],",
                "\"partial_grid\":[{},{},{}],\"partial_block\":[{},{},{}],",
                "\"reducer_grid\":[{},{},{}],\"reducer_block\":[{},{},{}],",
                "\"runtime_pointer_alignment_used\":true,",
                "\"oracle_bit_exact\":true,\"eager_graph_bit_exact\":true,",
                "\"timed_output_guard_validated_after_each_block\":true,",
                "\"red_zones_validated\":true,\"output_sha256\":\"{}\",",
                "\"measurement_only\":true,\"deterministic_qualification\":true,",
                "\"append_only\":true,\"runtime_tuning\":false,",
                "\"dispatcher_unchanged\":true,\"samples\":[{}],",
                "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},",
                "\"cuda_driver_api_version\":{},\"cuda_driver_build_sha256\":\"{}\",",
                "\"nvrtc_version\":\"{}.{}\",\"nvrtc_target\":\"{}\",",
                "\"scalar_compile_key\":\"{}\",\"scalar_artifact\":\"{}\",",
                "\"scalar_invocation\":\"{}\",\"tournament_source_sha256\":\"{}\",",
                "\"cuda_source_sha256\":\"{}\",\"executable_sha256\":\"{}\",",
                "\"pre_context\":\"{}\",\"timed_preflight\":\"{}\",",
                "\"postflight\":\"{}\"}}"
            ),
            GRAPH_REPLAY_SCHEMA,
            graph_replay_run_identity(metadata, topology),
            spec.cell.name,
            spec.cell.dims.0,
            spec.cell.dims.1,
            spec.cell.dims.2,
            plan.m_chunk,
            plan.chunks,
            GRAPH_REPLAY_GROUPS,
            GRAPH_REPLAY_ITERATIONS,
            topology.edge.data.dependency_type,
            topology.edge.data.from_port,
            topology.edge.data.to_port,
            topology.edge.data.reserved[0],
            topology.edge.data.reserved[1],
            topology.edge.data.reserved[2],
            topology.edge.data.reserved[3],
            topology.edge.data.reserved[4],
            topology.nodes[0].symbol,
            topology.nodes[1].symbol,
            topology.nodes[0].grid.0,
            topology.nodes[0].grid.1,
            topology.nodes[0].grid.2,
            topology.nodes[0].block.0,
            topology.nodes[0].block.1,
            topology.nodes[0].block.2,
            topology.nodes[1].grid.0,
            topology.nodes[1].grid.1,
            topology.nodes[1].grid.2,
            topology.nodes[1].block.0,
            topology.nodes[1].block.1,
            topology.nodes[1].block.2,
            digest_hex(&bits_digest(expected)),
            render_graph_replay_groups(groups),
            escape_json(&metadata.gpu_uuid),
            metadata.compute_capability.0,
            metadata.compute_capability.1,
            metadata.multiprocessors,
            metadata.driver_api_version,
            metadata.driver_build_digest,
            metadata.nvrtc_version.0,
            metadata.nvrtc_version.1,
            escape_json(&metadata.nvrtc_target),
            metadata.scalar_compile_key,
            metadata.scalar_artifact,
            metadata.scalar_invocation,
            metadata.tournament_source_sha256,
            metadata.cuda_source_sha256,
            metadata.executable_sha256,
            escape_json(&metadata.pre_context),
            escape_json(timed_preflight),
            escape_json(postflight),
        )
    }

    fn append_graph_replay_record(record: String) -> Result<(), String> {
        let value = std::env::var_os(GRAPH_REPLAY_OUTPUT_ENV)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{GRAPH_REPLAY_OUTPUT_ENV} must name a JSONL file"))?;
        let path = PathBuf::from(value);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| format!("open graph replay qualification {path:?}: {error}"))?;
        if !file
            .metadata()
            .map_err(|error| format!("inspect graph replay qualification {path:?}: {error}"))?
            .file_type()
            .is_file()
        {
            return Err(format!(
                "graph replay qualification {path:?} is not a regular file"
            ));
        }
        let mut bytes = record.into_bytes();
        bytes.push(b'\n');
        file.write_all(&bytes)
            .map_err(|error| format!("append graph replay qualification {path:?}: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync graph replay qualification {path:?}: {error}"))
    }

    fn graph_replay_qualification_device() -> Result<(QuietGpu, String, GpuDevice, GpuCtx), String>
    {
        if cfg!(debug_assertions) {
            return Err("TN narrow graph replay qualification requires --release".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        let pre_context =
            quiet.require_pre_context("tn-narrow-graph-replay-qualification/pre-context")?;
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "TN narrow graph replay qualification requires CC 12.0 with 170 SMs, received CC {}.{} with {} SMs",
                identity.compute_capability.0,
                identity.compute_capability.1,
                identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        configure_context(&ctx);
        Ok((quiet, pre_context, device, ctx))
    }

    fn run_graph_replay_qualification() -> Result<(), String> {
        let (quiet, pre_context, device, ctx) = graph_replay_qualification_device()?;
        let metadata = RunMetadata::new(&ctx, &device, &quiet, pre_context)?;
        let spec = graph_replay_qualification_spec();
        let plan = spec.baseline;
        let host = HostData::new(spec);
        let arm_specs = [Arm::Split {
            role: SplitRole::Candidate,
            plan,
        }];
        let mut buffers = CellBuffers::new_with_arms(&ctx, spec, &host, &arm_specs)?;
        buffers.validate_offsets(spec)?;
        let expected = split_oracle(&host, spec, plan);
        buffers.reset(&ctx, 0)?;
        launch_arm(&ctx, &buffers, spec, 0)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize eager qualification gate: {error:?}"))?;
        require_expected_bits(
            "eager qualification gate",
            &output_bits(&buffers.output_values(&ctx, 0)?),
            &expected,
        )?;
        let graphs = capture_arms(&ctx, &buffers, spec)?;
        require_graph_replay_bits(
            &ctx,
            &mut buffers,
            spec,
            &graphs,
            GraphReplayPath::Graph,
            &expected,
        )?;
        buffers.validate_guards(&ctx, spec)?;
        let timed_preflight = quiet.require_cohort("tn-narrow-graph-replay-qualification/timed")?;
        warm_graph_replay_paths(&ctx, &mut buffers, spec, &graphs)?;
        let mut groups = Vec::with_capacity(GRAPH_REPLAY_GROUPS);
        for _ in 0..GRAPH_REPLAY_GROUPS {
            groups.push(GraphReplayGroup {
                egge: measure_graph_replay_order(&ctx, &mut buffers, spec, &graphs, false)?,
                geeg: measure_graph_replay_order(&ctx, &mut buffers, spec, &graphs, true)?,
            });
        }
        for path in [GraphReplayPath::Eager, GraphReplayPath::Graph] {
            require_graph_replay_bits(&ctx, &mut buffers, spec, &graphs, path, &expected)?;
        }
        buffers.validate_guards(&ctx, spec)?;
        let postflight = quiet.verify_post_cohort("tn-narrow-graph-replay-qualification/post")?;
        let aligned_inputs =
            buffers.a_ptr().is_multiple_of(16) && buffers.b_ptr().is_multiple_of(16);
        let topology = split_graph_topology_descriptor(spec.cell, plan, aligned_inputs)?;
        append_graph_replay_record(render_graph_replay_record(
            &metadata,
            topology,
            &groups,
            &expected,
            &timed_preflight,
            &postflight,
        ))?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize graph replay qualification drop: {error:?}"))
    }

    #[test]
    fn winner_alignment_samples_and_completion_count_are_exact() {
        assert_eq!(winner_measurement_record_count(), 24);
        let mut samples = WinnerOrderedSamples::with_capacity(super::DISCOVERY_WINDOWS);
        for index in 0..super::DISCOVERY_WINDOWS {
            let offset = index as f64 * 0.01;
            samples.push(2.0 + offset, 1.0 + offset).unwrap();
        }
        assert_eq!(samples.baseline_us.len(), super::DISCOVERY_WINDOWS);
        assert_eq!(samples.winner_us.len(), super::DISCOVERY_WINDOWS);
        assert_eq!(samples.baseline_over_winner.len(), super::DISCOVERY_WINDOWS);
        assert!(
            samples
                .baseline_over_winner
                .iter()
                .all(|ratio| ratio.is_finite() && *ratio > 1.0)
        );
    }

    #[test]
    fn winner_alignment_completion_is_separate_and_measurement_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("winner-alignment.jsonl");
        let mut sink = WinnerJsonlSink::create(path.clone()).unwrap();
        for index in 0..winner_measurement_record_count() {
            sink.write(format!("{{\"measurement\":{index}}}")).unwrap();
        }
        sink.finish(super::DISCOVERY_WINDOWS).unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text.lines().count(), winner_measurement_record_count() + 1);
        let completion = text.lines().last().unwrap();
        assert!(completion.contains(WINNER_COMPLETION_SCHEMA));
        assert!(completion.contains("\"measurement_only\":true"));
        assert!(completion.contains("\"auto_admitted\":false"));
        assert!(completion.contains("\"existing_seven_cell_suite_unchanged\":true"));
    }

    #[test]
    fn summaries_require_the_selected_window_count_for_every_arm() {
        let arm_count = 4;
        let windows = super::DISCOVERY_WINDOWS;
        let mut samples = TournamentSamples {
            forward: OrderedSamples::with_capacity(arm_count, windows),
            reverse: OrderedSamples::with_capacity(arm_count, windows),
        };
        for index in 0..windows {
            let offset = index as f64 * 0.001;
            samples
                .forward
                .push(&[10.0 + offset, 8.0 + offset, 7.0 + offset, 6.0 + offset])
                .unwrap();
            samples
                .reverse
                .push(&[10.1 + offset, 8.1 + offset, 7.1 + offset, 6.1 + offset])
                .unwrap();
        }
        let summary = summarize(&samples, windows).unwrap();
        assert_eq!(summary.forward.arm_us.len(), arm_count);
        assert_eq!(summary.forward.baseline_over_candidate.len(), 2);
        assert!(summary.forward.baseline_over_candidate[0].p50 > 1.0);
        samples.forward.baseline_over_candidate[0].pop();
        assert!(summarize(&samples, windows).is_err());
    }

    #[test]
    fn completion_is_measurement_only_and_never_admits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("complete.jsonl");
        let mut sink = JsonlSink::create(path.clone()).unwrap();
        for index in 0..RECORDS_BEFORE_COMPLETION {
            sink.write(format!("{{\"measurement\":{index}}}")).unwrap();
        }
        sink.finish(super::DISCOVERY_WINDOWS).unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text.lines().count(), RECORDS_BEFORE_COMPLETION + 1);
        let completion = text.lines().last().unwrap();
        assert!(completion.contains("\"dataset_complete\":true"));
        assert!(completion.contains("\"measurement_only\":true"));
        assert!(completion.contains("\"runtime_autotuning\":false"));
        assert!(completion.contains("\"promotion_evaluated\":false"));
        assert!(completion.contains("\"auto_admitted\":false"));
        assert!(completion.contains(&format!("\"decision\":\"{DECISION}\"")));
    }

    #[test]
    #[ignore = "requires an otherwise idle 170-SM CC 12.0 GPU; run release with one test thread"]
    fn raw_partition_factor_tournament() {
        let windows = window_count(std::env::var(FINAL_ENV).ok().as_deref()).unwrap();
        let (quiet, pre_context, device, ctx) = qualification_device().unwrap();
        let metadata = RunMetadata::new(&ctx, &device, &quiet, pre_context).unwrap();
        let mut sink =
            JsonlSink::create_from_env().expect("new TN narrow partition JSONL evidence file");
        for spec in PARTITION_CELLS {
            let host = HostData::new(spec);
            let mut buffers = CellBuffers::new(&ctx, spec, &host).unwrap();
            let (correctness, graphs) = correctness_gate(&ctx, &mut buffers, &host, spec).unwrap();
            for path in [PathKind::Eager, PathKind::Graph] {
                let mut run = CohortContext {
                    ctx: &ctx,
                    quiet: &quiet,
                    buffers: &mut buffers,
                    spec,
                    graphs: &graphs,
                    correctness: &correctness,
                };
                let (samples, summary, cohort) =
                    run_cohort(&mut run, CohortSpec { path, windows }).unwrap();
                emit_measurement(
                    &mut sink,
                    MeasurementRecord {
                        metadata: &metadata,
                        spec,
                        buffers: &buffers,
                        path,
                        windows,
                        correctness: &correctness,
                        samples: &samples,
                        summary: &summary,
                        cohort: &cohort,
                    },
                )
                .unwrap();
            }
        }
        sink.finish(windows).unwrap();
    }

    #[test]
    #[ignore = "requires an otherwise idle 170-SM CC 12.0 GPU; run release with one test thread"]
    fn raw_partition_winner_alignment_tournament() {
        run_winner_alignment_tournament().unwrap();
    }

    #[test]
    #[ignore = "requires an otherwise idle 170-SM CC 12.0 GPU; deterministic qualification only"]
    fn raw_partition_graph_replay_qualification() {
        run_graph_replay_qualification().unwrap();
    }
}
