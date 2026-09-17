#![cfg(feature = "cuda")]

use std::collections::BTreeSet;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM90A_AUTO_CELLS, SM90A_DYNAMIC_SHARED_BYTES, SM90A_STAGES, SM90A_TILE, Sm90aMapRequest,
    Sm90aOp, Sm90aShape, Sm90aTensorMap, Sm90aWarpgroupSchedule, resolve_sm90a_forced,
    validate_sm90a_map_request,
};

const SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm90a/wgmma.cu");

const SYMBOLS: &[&str] = &[
    "nn_sm90a_wgmma_wg1_bf16",
    "nn_sm90a_wgmma_wg1_f16",
    "tn_sm90a_wgmma_wg1_bf16",
    "tn_sm90a_wgmma_wg1_f16",
    "nt_sm90a_wgmma_wg1_bf16",
    "nt_sm90a_wgmma_wg1_f16",
    "nn_sm90a_wgmma_wg2_bf16",
    "nn_sm90a_wgmma_wg2_f16",
    "tn_sm90a_wgmma_wg2_bf16",
    "tn_sm90a_wgmma_wg2_f16",
    "nt_sm90a_wgmma_wg2_bf16",
    "nt_sm90a_wgmma_wg2_f16",
];

#[test]
fn hopper_source_owns_the_exact_forced_inventory() {
    let declared: BTreeSet<_> = SYMBOLS
        .iter()
        .copied()
        .filter(|symbol| SOURCE.contains(symbol))
        .collect();
    assert_eq!(declared.len(), SYMBOLS.len());
    assert_eq!(declared, SYMBOLS.iter().copied().collect());
}

#[test]
fn hopper_source_freezes_the_physical_contract() {
    for required in [
        "SM90A_TILE_M 64",
        "SM90A_TILE_N 128",
        "SM90A_TILE_K 64",
        "SM90A_STAGES 3",
        "SM90A_DYNAMIC_SHARED_BYTES 73984",
        "CUtensorMap",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx",
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.",
        "SM90A_DEFINE_WGMMA(sm90a_wgmma_nn_bf16, \"bf16\", 0, 1)",
        "SM90A_DEFINE_WGMMA(sm90a_wgmma_nn_f16, \"f16\", 0, 1)",
        "__launch_bounds__(128, 3)",
        "__maxnreg__(128)",
        "setmaxnreg.dec.sync.aligned.u32 40",
        "setmaxnreg.inc.sync.aligned.u32 128",
    ] {
        assert!(SOURCE.contains(required), "missing {required}");
    }
    assert!(!SOURCE.contains("atom."));
    assert!(!SOURCE.contains("red."));
    assert!(!SOURCE.contains("multicast"));
}

#[test]
fn forced_selector_fails_closed_without_exact_hopper() {
    let shape = Sm90aShape::contiguous(Sm90aOp::Nn, (64, 64, 128));
    assert_eq!(SM90A_AUTO_CELLS, &[]);
    for cc in [(8, 0), (8, 9), (9, 1), (10, 0), (12, 0)] {
        assert_eq!(
            resolve_sm90a_forced(
                cc,
                true,
                Sm90aOp::Nn,
                WeightDtype::Bf16,
                Sm90aWarpgroupSchedule::Wg1,
                shape,
            )
            .unwrap(),
            None
        );
    }
    assert_eq!(
        resolve_sm90a_forced(
            (9, 0),
            false,
            Sm90aOp::Nn,
            WeightDtype::Bf16,
            Sm90aWarpgroupSchedule::Wg1,
            shape,
        )
        .unwrap(),
        None
    );
}

#[test]
fn forced_selector_owns_all_twelve_symbols() {
    let mut symbols = BTreeSet::new();
    for op in [Sm90aOp::Nn, Sm90aOp::Tn, Sm90aOp::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for schedule in [Sm90aWarpgroupSchedule::Wg1, Sm90aWarpgroupSchedule::Wg2] {
                let route = resolve_sm90a_forced(
                    (9, 0),
                    true,
                    op,
                    dtype,
                    schedule,
                    Sm90aShape::contiguous(op, (65, 64, 129)),
                )
                .unwrap()
                .unwrap();
                symbols.insert(route.symbol());
            }
        }
    }
    assert_eq!(symbols, SYMBOLS.iter().copied().collect());
    assert!(
        resolve_sm90a_forced(
            (9, 0),
            true,
            Sm90aOp::Nn,
            WeightDtype::F32,
            Sm90aWarpgroupSchedule::Wg1,
            Sm90aShape::contiguous(Sm90aOp::Nn, (64, 64, 128)),
        )
        .is_err()
    );
}

#[test]
fn tensor_map_contract_uses_driver_abi_and_sixteen_byte_bases() {
    assert_eq!(SM90A_TILE, (64, 128, 64));
    assert_eq!(SM90A_STAGES, 3);
    assert_eq!(SM90A_DYNAMIC_SHARED_BYTES, 73_984);
    assert_eq!(
        std::mem::size_of::<Sm90aTensorMap>(),
        std::mem::size_of::<cudarc::driver::sys::CUtensorMap>()
    );
    assert_eq!(
        std::mem::align_of::<Sm90aTensorMap>(),
        std::mem::align_of::<cudarc::driver::sys::CUtensorMap>()
    );

    let shape = Sm90aShape::contiguous(Sm90aOp::Nn, (64, 64, 128));
    for base in [0x1010_u64, 0x1020, 0x1040] {
        validate_sm90a_map_request(Sm90aMapRequest {
            op: Sm90aOp::Nn,
            dtype: WeightDtype::Bf16,
            a_ptr: base,
            b_ptr: base + 0x1000,
            shape,
        })
        .unwrap();
    }
    let error = validate_sm90a_map_request(Sm90aMapRequest {
        op: Sm90aOp::Nn,
        dtype: WeightDtype::F16,
        a_ptr: 0x1008,
        b_ptr: 0x2000,
        shape,
    })
    .unwrap_err();
    assert!(error.contains("16-byte aligned"), "{error}");
}
