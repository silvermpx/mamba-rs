//! The Ada-measured inference cells: one Fixed-overlay kernel per (shape,
//! operand dtypes, policy) that beat the ladder route on the Ada board. The
//! Ada board takes them as evidence; every other sm_80-tier board proves a
//! cell's output words against the ladder route at first use.

use super::super::context::F32TriadPolicy;
use super::super::dtype::WeightDtype;
use super::{FixedTileDevice, InferenceFwdOperands, InferenceShape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm89CellRoute {
    /// 4621 x 1928 x 384 exact f32: 112x128 tiles of scalar FMA chains.
    ExactM112N128Bk32S3,
    /// 2048 x 768 x 2304 half inputs with an f32 output: 128x144 tiles.
    HalfF32OutM128N144Bk32S2,
    /// 2048 x 2304 x 768 homogeneous half: 128x96 tiles with packed stores.
    HalfM128N96Bk64S2Vec,
    /// 2048 x 768 x 2304 bf16: 128x144 tiles with packed stores.
    HalfM128N144Bk32S2VecBf16,
    /// 2048 x 768 x 2304 TF32: 64x288 tiles.
    Tf32M64N288Bk16S2,
    /// 2048 x 2304 x 768 TF32 with a bias: 64x96 tiles.
    Tf32M64N96Bk32S2Bias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sm89CellFamily {
    ExactFma,
    HalfMma,
    Tf32Mma,
}

/// The output a cell writes: f32, or the input's own half type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sm89CellOutput {
    F32,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sm89CellSpec {
    pub route: Sm89CellRoute,
    pub family: Sm89CellFamily,
    pub symbol: &'static str,
    pub input: WeightDtype,
    pub output: Sm89CellOutput,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
}

/// A half cell from its geometry: (tile, bk, threads, dynamic shared bytes).
const fn half_spec(
    route: Sm89CellRoute,
    symbol: &'static str,
    input: WeightDtype,
    output: Sm89CellOutput,
    geometry: ((u32, u32), u32, u32, u32),
) -> Sm89CellSpec {
    let (tile, bk, threads, dynamic_shared_bytes) = geometry;
    Sm89CellSpec {
        route,
        family: Sm89CellFamily::HalfMma,
        symbol,
        input,
        output,
        tile,
        bk,
        stages: 2,
        threads,
        dynamic_shared_bytes,
        register_cap: 255,
        occupancy_gate: 1,
    }
}

pub(crate) const SM89_CELL_SPECS: [Sm89CellSpec; 8] = [
    Sm89CellSpec {
        route: Sm89CellRoute::ExactM112N128Bk32S3,
        family: Sm89CellFamily::ExactFma,
        symbol: "nn_sm89_m112n128_bk32_s3_f32",
        input: WeightDtype::F32,
        output: Sm89CellOutput::F32,
        tile: (112, 128),
        bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 97_536,
        register_cap: 255,
        occupancy_gate: 1,
    },
    half_spec(
        Sm89CellRoute::HalfF32OutM128N144Bk32S2,
        "nn_sm89_m128n144_bk32_s2_f32out_bf16",
        WeightDtype::Bf16,
        Sm89CellOutput::F32,
        ((128, 144), 32, 128, 39_936),
    ),
    half_spec(
        Sm89CellRoute::HalfF32OutM128N144Bk32S2,
        "nn_sm89_m128n144_bk32_s2_f32out_f16",
        WeightDtype::F16,
        Sm89CellOutput::F32,
        ((128, 144), 32, 128, 39_936),
    ),
    half_spec(
        Sm89CellRoute::HalfM128N96Bk64S2Vec,
        "nn_sm89_m128n96_bk64_s2_vec_bf16",
        WeightDtype::Bf16,
        Sm89CellOutput::Input,
        ((128, 96), 64, 256, 63_488),
    ),
    half_spec(
        Sm89CellRoute::HalfM128N96Bk64S2Vec,
        "nn_sm89_m128n96_bk64_s2_vec_f16",
        WeightDtype::F16,
        Sm89CellOutput::Input,
        ((128, 96), 64, 256, 63_488),
    ),
    // The packed-store epilogue stages the f32 tile (128 x 152 floats) on top
    // of the pipeline slabs, so the launch allocates the tile, not the slabs.
    half_spec(
        Sm89CellRoute::HalfM128N144Bk32S2VecBf16,
        "nn_sm89_m128n144_bk32_s2_vec_bf16",
        WeightDtype::Bf16,
        Sm89CellOutput::Input,
        ((128, 144), 32, 256, 77_824),
    ),
    Sm89CellSpec {
        route: Sm89CellRoute::Tf32M64N288Bk16S2,
        family: Sm89CellFamily::Tf32Mma,
        symbol: "nn_sm89_m64n288_bk16_s2_tf32",
        input: WeightDtype::F32,
        output: Sm89CellOutput::F32,
        tile: (64, 288),
        bk: 16,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 48_128,
        register_cap: 255,
        occupancy_gate: 2,
    },
    Sm89CellSpec {
        route: Sm89CellRoute::Tf32M64N96Bk32S2Bias,
        family: Sm89CellFamily::Tf32Mma,
        symbol: "nn_sm89_m64n96_bk32_s2_tf32",
        input: WeightDtype::F32,
        output: Sm89CellOutput::F32,
        tile: (64, 96),
        bk: 32,
        stages: 2,
        threads: 128,
        dynamic_shared_bytes: 45_056,
        register_cap: 255,
        occupancy_gate: 2,
    },
];

pub(crate) const SM89_CELL_SYMBOLS: [&str; 8] = [
    SM89_CELL_SPECS[0].symbol,
    SM89_CELL_SPECS[1].symbol,
    SM89_CELL_SPECS[2].symbol,
    SM89_CELL_SPECS[3].symbol,
    SM89_CELL_SPECS[4].symbol,
    SM89_CELL_SPECS[5].symbol,
    SM89_CELL_SPECS[6].symbol,
    SM89_CELL_SPECS[7].symbol,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sm89CellBias {
    Any,
    Required,
}

/// One measured cell: the forward (m, k, n) and the bias states measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sm89Cell {
    pub route: Sm89CellRoute,
    pub shape: (usize, usize, usize),
    pub bias: Sm89CellBias,
}

pub(crate) const SM89_CELLS: [Sm89Cell; 6] = [
    Sm89Cell {
        route: Sm89CellRoute::ExactM112N128Bk32S3,
        shape: (4621, 1928, 384),
        bias: Sm89CellBias::Any,
    },
    Sm89Cell {
        route: Sm89CellRoute::HalfF32OutM128N144Bk32S2,
        shape: (2048, 768, 2304),
        bias: Sm89CellBias::Any,
    },
    Sm89Cell {
        route: Sm89CellRoute::HalfM128N144Bk32S2VecBf16,
        shape: (2048, 768, 2304),
        bias: Sm89CellBias::Any,
    },
    Sm89Cell {
        route: Sm89CellRoute::Tf32M64N288Bk16S2,
        shape: (2048, 768, 2304),
        bias: Sm89CellBias::Any,
    },
    Sm89Cell {
        route: Sm89CellRoute::HalfM128N96Bk64S2Vec,
        shape: (2048, 2304, 768),
        bias: Sm89CellBias::Any,
    },
    Sm89Cell {
        route: Sm89CellRoute::Tf32M64N96Bk32S2Bias,
        shape: (2048, 2304, 768),
        bias: Sm89CellBias::Required,
    },
];

#[derive(Clone, Copy)]
pub(super) struct Sm89CellStack {
    pub(super) device: FixedTileDevice,
    pub(super) nvrtc: (i32, i32),
    pub(super) nvrtc_library_known: bool,
    pub(super) policy: F32TriadPolicy,
}

pub(crate) fn spec_for_route(
    route: Sm89CellRoute,
    input: WeightDtype,
) -> Option<&'static Sm89CellSpec> {
    SM89_CELL_SPECS
        .iter()
        .find(|spec| spec.route == route && spec.input == input)
}

/// Whether the operands carry exactly the dtypes the cell was measured with.
pub(crate) fn operands_match(spec: &Sm89CellSpec, operands: InferenceFwdOperands) -> bool {
    let output = match spec.output {
        Sm89CellOutput::F32 => WeightDtype::F32,
        Sm89CellOutput::Input => spec.input,
    };
    operands.x.dtype == spec.input && operands.w.dtype == spec.input && operands.c.dtype == output
}

fn policy_admits(spec: &Sm89CellSpec, policy: F32TriadPolicy) -> bool {
    match spec.family {
        Sm89CellFamily::ExactFma => policy == F32TriadPolicy::ExactScalarFma,
        Sm89CellFamily::Tf32Mma => policy == F32TriadPolicy::AllowDeterministicTf32,
        Sm89CellFamily::HalfMma => true,
    }
}

/// The cell a forward request lands on: the exact measured shape, the
/// measured operand dtypes and bias state, the policy the family serves,
/// an sm_80-tier board that composes the overlay, and the kernel bound.
pub(super) fn select_sm89_cell(
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    stack: Sm89CellStack,
    loaded: impl Fn(&str) -> bool,
) -> Option<&'static Sm89CellSpec> {
    if !stack.nvrtc_library_known
        || !super::fixed_portable_overlay_board(stack.device.compute_capability)
        || stack.device.multiprocessors == 0
        || !matches!(stack.nvrtc, (12, 8) | (13, 0) | (13, 2))
        || [operands.c.ptr, operands.x.ptr, operands.w.ptr]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
        || operands
            .bias_ptr
            .is_some_and(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
    {
        return None;
    }
    SM89_CELLS
        .iter()
        .filter(|cell| {
            cell.shape == (shape.m, shape.k, shape.n)
                && (cell.bias == Sm89CellBias::Any || operands.bias_ptr.is_some())
        })
        .find_map(|cell| {
            spec_for_route(cell.route, operands.x.dtype).filter(|spec| {
                operands_match(spec, operands)
                    && policy_admits(spec, stack.policy)
                    && loaded(spec.symbol)
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::gemm_bi_inference::TypedPtr;

    /// The half cells stage their pipeline slabs in dynamic shared memory,
    /// and the packed-store cells additionally stage the whole f32 tile
    /// there for the epilogue; the launch must allocate the larger of the
    /// two, or the epilogue writes past the allocation.
    #[test]
    fn half_cells_allocate_the_larger_of_pipeline_and_epilogue_tile() {
        for spec in &SM89_CELL_SPECS {
            if spec.family != Sm89CellFamily::HalfMma {
                continue;
            }
            let (rows, columns) = spec.tile;
            let pipeline =
                u32::from(spec.stages) * (rows * (spec.bk + 8) + spec.bk * (columns + 8)) * 2;
            let epilogue = match spec.route {
                Sm89CellRoute::HalfM128N96Bk64S2Vec | Sm89CellRoute::HalfM128N144Bk32S2VecBf16 => {
                    rows * (columns + 8) * 4
                }
                _ => 0,
            };
            assert_eq!(
                spec.dynamic_shared_bytes,
                pipeline.max(epilogue),
                "{} shared-memory allocation",
                spec.symbol
            );
        }
    }

    fn operands(x: WeightDtype, c: WeightDtype, bias: Option<u64>) -> InferenceFwdOperands {
        InferenceFwdOperands {
            c: TypedPtr {
                ptr: 0x1000,
                dtype: c,
            },
            x: TypedPtr {
                ptr: 0x2000,
                dtype: x,
            },
            w: TypedPtr {
                ptr: 0x3000,
                dtype: x,
            },
            bias_ptr: bias,
        }
    }

    fn shape(dims: (usize, usize, usize)) -> InferenceShape {
        InferenceShape {
            m: dims.0,
            k: dims.1,
            n: dims.2,
        }
    }

    fn stack(cc: (u32, u32), policy: F32TriadPolicy) -> Sm89CellStack {
        Sm89CellStack {
            device: FixedTileDevice {
                multiprocessors: 142,
                compute_capability: cc,
            },
            nvrtc: (13, 2),
            nvrtc_library_known: true,
            policy,
        }
    }

    const EXACT: F32TriadPolicy = F32TriadPolicy::ExactScalarFma;
    const TF32: F32TriadPolicy = F32TriadPolicy::AllowDeterministicTf32;

    fn select(
        ops: InferenceFwdOperands,
        dims: (usize, usize, usize),
        stack: Sm89CellStack,
    ) -> Option<&'static str> {
        select_sm89_cell(ops, shape(dims), stack, |_| true).map(|spec| spec.symbol)
    }

    #[test]
    fn every_measured_cell_selects_its_own_kernel() {
        use WeightDtype::{Bf16, F16, F32};
        let ada = stack((8, 9), EXACT);
        let ada_tf32 = stack((8, 9), TF32);
        for (ops, dims, stack, symbol) in [
            (
                operands(F32, F32, None),
                (4621, 1928, 384),
                ada,
                "nn_sm89_m112n128_bk32_s3_f32",
            ),
            (
                operands(F32, F32, Some(0x4004)),
                (4621, 1928, 384),
                ada,
                "nn_sm89_m112n128_bk32_s3_f32",
            ),
            (
                operands(Bf16, F32, None),
                (2048, 768, 2304),
                ada,
                "nn_sm89_m128n144_bk32_s2_f32out_bf16",
            ),
            (
                operands(F16, F32, Some(0x4004)),
                (2048, 768, 2304),
                ada_tf32,
                "nn_sm89_m128n144_bk32_s2_f32out_f16",
            ),
            (
                operands(Bf16, Bf16, None),
                (2048, 768, 2304),
                ada,
                "nn_sm89_m128n144_bk32_s2_vec_bf16",
            ),
            (
                operands(F32, F32, None),
                (2048, 768, 2304),
                ada_tf32,
                "nn_sm89_m64n288_bk16_s2_tf32",
            ),
            (
                operands(F32, F32, Some(0x4004)),
                (2048, 768, 2304),
                ada_tf32,
                "nn_sm89_m64n288_bk16_s2_tf32",
            ),
            (
                operands(Bf16, Bf16, None),
                (2048, 2304, 768),
                ada,
                "nn_sm89_m128n96_bk64_s2_vec_bf16",
            ),
            (
                operands(F16, F16, Some(0x4004)),
                (2048, 2304, 768),
                ada,
                "nn_sm89_m128n96_bk64_s2_vec_f16",
            ),
            (
                operands(F32, F32, Some(0x4004)),
                (2048, 2304, 768),
                ada_tf32,
                "nn_sm89_m64n96_bk32_s2_tf32",
            ),
        ] {
            assert_eq!(
                select(ops, dims, stack),
                Some(symbol),
                "{dims:?} {:?}",
                ops.x.dtype
            );
        }
    }

    #[test]
    fn unmeasured_states_fall_through_to_the_ladder() {
        use WeightDtype::{Bf16, F16, F32};
        let ada = stack((8, 9), EXACT);
        let ada_tf32 = stack((8, 9), TF32);
        for (ops, dims, stack, why) in [
            (
                operands(F32, F32, None),
                (2048, 2304, 768),
                ada_tf32,
                "the hot_e TF32 cell needs a bias",
            ),
            (
                operands(F32, F32, None),
                (2048, 2304, 768),
                ada,
                "hot_e exact f32 has no cell",
            ),
            (
                operands(F16, F16, None),
                (2048, 768, 2304),
                ada,
                "the hot_d packed cell is bf16 only",
            ),
            (
                operands(F32, F32, None),
                (4621, 1928, 384),
                ada_tf32,
                "hot_c under TF32 has no cell",
            ),
            (
                operands(F32, F32, None),
                (2048, 768, 2304),
                ada,
                "hot_d exact f32 has no cell",
            ),
            (
                operands(Bf16, Bf16, None),
                (1024, 256, 128),
                ada,
                "no cell at the classifier shape",
            ),
            (
                operands(Bf16, F16, None),
                (2048, 768, 2304),
                ada,
                "mismatched half operands",
            ),
        ] {
            assert_eq!(select(ops, dims, stack), None, "{why}");
        }
    }

    #[test]
    fn other_sm80_boards_reach_the_cells_and_the_cc12_family_does_not() {
        use WeightDtype::F32;
        let ops = operands(F32, F32, None);
        for cc in [(8, 0), (8, 6), (9, 0), (10, 0), (11, 0)] {
            assert!(
                select(ops, (4621, 1928, 384), stack(cc, EXACT)).is_some(),
                "{cc:?}"
            );
        }
        for cc in [(7, 5), (12, 0), (12, 1)] {
            assert_eq!(
                select(ops, (4621, 1928, 384), stack(cc, EXACT)),
                None,
                "{cc:?}"
            );
        }
        for nvrtc in [(12, 7), (12, 9), (13, 1), (13, 3)] {
            let mut stack = stack((8, 9), EXACT);
            stack.nvrtc = nvrtc;
            assert_eq!(select(ops, (4621, 1928, 384), stack), None, "{nvrtc:?}");
        }
        let mut unknown = stack((8, 9), EXACT);
        unknown.nvrtc_library_known = false;
        assert_eq!(select(ops, (4621, 1928, 384), unknown), None);
        let mut idle = stack((8, 9), EXACT);
        idle.device.multiprocessors = 0;
        assert_eq!(select(ops, (4621, 1928, 384), idle), None);
        assert_eq!(
            select_sm89_cell(ops, shape((4621, 1928, 384)), stack((8, 9), EXACT), |_| {
                false
            }),
            None,
            "an unbound kernel never selects"
        );
        for misaligned in [
            InferenceFwdOperands {
                c: TypedPtr {
                    ptr: 0x1008,
                    ..ops.c
                },
                ..ops
            },
            InferenceFwdOperands {
                x: TypedPtr { ptr: 0, ..ops.x },
                ..ops
            },
            InferenceFwdOperands {
                bias_ptr: Some(0x4002),
                ..ops
            },
        ] {
            assert_eq!(
                select(misaligned, (4621, 1928, 384), stack((8, 9), EXACT)),
                None
            );
        }
    }

    #[test]
    fn specs_and_symbols_are_one_closed_table() {
        assert_eq!(SM89_CELL_SPECS.len(), SM89_CELL_SYMBOLS.len());
        let unique: std::collections::BTreeSet<_> = SM89_CELL_SYMBOLS.into_iter().collect();
        assert_eq!(unique.len(), SM89_CELL_SYMBOLS.len());
        for cell in SM89_CELLS {
            assert!(
                SM89_CELL_SPECS.iter().any(|spec| spec.route == cell.route),
                "{:?} has no kernel",
                cell.route
            );
        }
        for spec in SM89_CELL_SPECS {
            assert_eq!(spec_for_route(spec.route, spec.input), Some(&spec));
            assert!(spec.threads.is_multiple_of(32));
            assert!(spec.dynamic_shared_bytes <= 101_376);
        }
    }
}
