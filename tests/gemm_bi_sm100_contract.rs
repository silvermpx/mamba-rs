#![cfg(feature = "cuda")]

use std::collections::BTreeSet;

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM100_AUTO_CELLS_CC100, SM100_AUTO_CELLS_CC103, SM100_KERNEL_SPECS, Sm100ForcedRoute,
    Sm100MapRequest, Sm100Op, Sm100PhysicalRoute, Sm100Schedule, Sm100Shape, Sm100Stages,
    Sm100TargetCandidate, Sm100TargetKind, Sm100TensorMap, Sm100Tile, resolve_sm100_forced,
    sm100_target_candidates, validate_sm100_map_request,
};

const SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm100.cu");
const LAUNCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");

fn public_function_source(source: &str, name: &str) -> String {
    let marker = format!("pub fn {name}(");
    let start = source.find(&marker).expect("public function source");
    let tail = &source[start..];
    let end = tail[marker.len()..]
        .find("\npub fn ")
        .map(|offset| marker.len() + offset)
        .unwrap_or(tail.len());
    tail[..end].to_string()
}

fn expected_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["m128n64", "m128n128"] {
            for stages in ["s2", "s3", "s4"] {
                for schedule in ["c4", "p8"] {
                    for dtype in ["bf16", "f16"] {
                        symbols.insert(format!(
                            "gemm_bi_{op}_sm100_tcgen_{tile}_bk64_{stages}_{schedule}_{dtype}"
                        ));
                    }
                }
            }
        }
    }
    symbols
}

fn source_declared_symbols() -> BTreeSet<String> {
    SOURCE
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("SM100_DEFINE_KERNEL(")?
                .split(',')
                .next()
                .map(|symbol| symbol.trim().to_string())
        })
        .collect()
}

fn shape(op: Sm100Op) -> Sm100Shape {
    match op {
        Sm100Op::Nn | Sm100Op::Tn => Sm100Shape {
            m: 129,
            k: 65,
            n: 127,
            lda: 72,
            ldb: 128,
            ldc: 130,
        },
        Sm100Op::Nt => Sm100Shape {
            m: 129,
            k: 65,
            n: 127,
            lda: 128,
            ldb: 128,
            ldc: 72,
        },
    }
}

fn route(
    op: Sm100Op,
    dtype: WeightDtype,
    tile: Sm100Tile,
    stages: Sm100Stages,
    schedule: Sm100Schedule,
) -> Sm100ForcedRoute {
    Sm100ForcedRoute {
        op,
        dtype,
        physical: Sm100PhysicalRoute {
            tile,
            stages,
            schedule,
        },
        shape: shape(op),
    }
}

#[test]
fn sm100_source_owns_exact_72_symbol_cross_product() {
    let expected = expected_symbols();
    assert_eq!(expected.len(), 72);
    assert_eq!(source_declared_symbols(), expected);
}

#[test]
fn sm100_kernel_specs_cover_every_physical_route_once() {
    assert_eq!(SM100_KERNEL_SPECS.len(), 72);
    let symbols: BTreeSet<_> = SM100_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol.to_string())
        .collect();
    assert_eq!(symbols, expected_symbols());

    for op in [Sm100Op::Nn, Sm100Op::Tn, Sm100Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for tile in [Sm100Tile::M128N64, Sm100Tile::M128N128] {
                for stages in [Sm100Stages::S2, Sm100Stages::S3, Sm100Stages::S4] {
                    for schedule in [Sm100Schedule::C4, Sm100Schedule::P8] {
                        let route = route(op, dtype, tile, stages, schedule);
                        let spec = route.kernel_spec().unwrap();
                        assert_eq!(spec.op, op);
                        assert_eq!(spec.dtype, dtype);
                        assert_eq!(spec.physical, route.physical);
                    }
                }
            }
        }
    }
}

#[test]
fn sm100_physical_specs_freeze_threads_and_shared_bytes() {
    let shared = [
        (Sm100Tile::M128N64, Sm100Stages::S2, 49_408),
        (Sm100Tile::M128N64, Sm100Stages::S3, 73_984),
        (Sm100Tile::M128N64, Sm100Stages::S4, 98_560),
        (Sm100Tile::M128N128, Sm100Stages::S2, 65_792),
        (Sm100Tile::M128N128, Sm100Stages::S3, 98_560),
        (Sm100Tile::M128N128, Sm100Stages::S4, 131_328),
    ];
    for spec in SM100_KERNEL_SPECS {
        let expected_threads = match spec.physical.schedule {
            Sm100Schedule::C4 => 128,
            Sm100Schedule::P8 => 256,
        };
        assert_eq!(spec.threads, expected_threads, "{} threads", spec.symbol);
        let expected_shared = shared
            .iter()
            .find(|&&(tile, stages, _)| {
                tile == spec.physical.tile && stages == spec.physical.stages
            })
            .unwrap()
            .2;
        assert_eq!(
            spec.dynamic_shared_bytes, expected_shared,
            "{} smem",
            spec.symbol
        );
        assert_eq!(
            spec.physical.tile.tmem_columns(),
            spec.physical.tile.output_columns()
        );
        assert_eq!(spec.physical.tile.output_rows(), 128);
        assert_eq!(spec.bk, 64);
    }
}

#[test]
fn sm100_source_closes_every_private_macro() {
    let mut definitions = BTreeSet::new();
    let mut live = BTreeSet::new();
    for line in SOURCE.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("#define SM100_") {
            let suffix = rest
                .split(|ch: char| ch.is_ascii_whitespace() || ch == '(')
                .next()
                .unwrap();
            let name = format!("SM100_{suffix}");
            assert!(live.insert(name.clone()), "duplicate definition {name}");
            definitions.insert(name);
        }
        if let Some(rest) = line.strip_prefix("#undef SM100_") {
            let suffix = rest.split_ascii_whitespace().next().unwrap();
            let name = format!("SM100_{suffix}");
            assert!(live.remove(&name), "undef without a live definition {name}");
        }
    }
    assert!(!definitions.is_empty());
    assert!(
        live.is_empty(),
        "live macros at end of translation unit: {live:?}"
    );
}

#[test]
fn sm100_source_freezes_tcgen_contract_and_forbidden_families() {
    for required in [
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.",
        "shared::cluster.b64 [%0];",
        "tcgen05.fence::before_thread_sync",
        "tcgen05.fence::after_thread_sync",
        "tcgen05.ld.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::ld.sync.aligned",
        "tcgen05.st.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::st.sync.aligned",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.",
        "mbarrier::complete_tx::bytes ",
        "mbarrier.arrive.expect_tx",
        "0x08100010",
        "0x08100490",
        "0x08110010",
        "0x08110490",
        "0x08118010",
        "0x08118490",
        "0x08200010",
        "0x08200490",
        "0x08210010",
        "0x08210490",
        "0x08218010",
        "0x08218490",
    ] {
        assert!(SOURCE.contains(required), "missing {required}");
    }
    for forbidden in [
        "atom.",
        "red.",
        "tcgen05.ld.red",
        "wgmma.",
        "cta_group::2",
        "multicast",
        "cudaMalloc",
        "malloc(",
        "operator new",
    ] {
        assert!(!SOURCE.contains(forbidden), "forbidden {forbidden}");
    }
}

#[test]
fn sm100_auto_tables_are_separate_and_empty() {
    assert_eq!(SM100_AUTO_CELLS_CC100, &[]);
    assert_eq!(SM100_AUTO_CELLS_CC103, &[]);
}

#[test]
fn sm100_forced_selector_covers_72_routes_and_rejects_f32() {
    let mut symbols = BTreeSet::new();
    for op in [Sm100Op::Nn, Sm100Op::Tn, Sm100Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for tile in [Sm100Tile::M128N64, Sm100Tile::M128N128] {
                for stages in [Sm100Stages::S2, Sm100Stages::S3, Sm100Stages::S4] {
                    for schedule in [Sm100Schedule::C4, Sm100Schedule::P8] {
                        let requested = route(op, dtype, tile, stages, schedule);
                        let resolved = resolve_sm100_forced(
                            (10, 0),
                            Some(sm100_target_candidates((10, 0))[0]),
                            requested,
                        )
                        .unwrap()
                        .unwrap();
                        symbols.insert(resolved.kernel_spec().unwrap().symbol.to_string());
                    }
                }
            }
        }
    }
    assert_eq!(symbols, expected_symbols());
    assert!(
        resolve_sm100_forced(
            (10, 0),
            Some(sm100_target_candidates((10, 0))[0]),
            route(
                Sm100Op::Nn,
                WeightDtype::F32,
                Sm100Tile::M128N64,
                Sm100Stages::S2,
                Sm100Schedule::C4,
            ),
        )
        .is_err()
    );

    for op in [Sm100Op::Nn, Sm100Op::Tn, Sm100Op::Nt] {
        for axis in 0..3 {
            let mut requested = route(
                op,
                WeightDtype::Bf16,
                Sm100Tile::M128N64,
                Sm100Stages::S2,
                Sm100Schedule::C4,
            );
            match axis {
                0 => requested.shape.m = 0,
                1 => requested.shape.k = 0,
                _ => requested.shape.n = 0,
            }
            assert!(
                resolve_sm100_forced(
                    (10, 0),
                    Some(sm100_target_candidates((10, 0))[0]),
                    requested,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn sm100_target_candidates_are_minor_exact_and_ordered() {
    let cc100 = sm100_target_candidates((10, 0));
    assert_eq!(cc100.len(), 2);
    assert_eq!(cc100[0].nvrtc_arch, "compute_100f");
    assert_eq!(cc100[0].ptx_target, "sm_100f");
    assert_eq!(cc100[0].kind, Sm100TargetKind::Family);
    assert_eq!(cc100[1].nvrtc_arch, "compute_100a");
    assert_eq!(cc100[1].ptx_target, "sm_100a");
    assert_eq!(cc100[1].kind, Sm100TargetKind::Exact);

    let cc103 = sm100_target_candidates((10, 3));
    assert_eq!(cc103.len(), 2);
    assert_eq!(cc103[0].nvrtc_arch, "compute_103f");
    assert_eq!(cc103[0].ptx_target, "sm_103f");
    assert_eq!(cc103[0].kind, Sm100TargetKind::Family);
    assert_eq!(cc103[1].nvrtc_arch, "compute_103a");
    assert_eq!(cc103[1].ptx_target, "sm_103a");
    assert_eq!(cc103[1].kind, Sm100TargetKind::Exact);
    assert!(sm100_target_candidates((9, 0)).is_empty());
    assert!(sm100_target_candidates((12, 0)).is_empty());
}

#[test]
fn sm100_target_resolution_never_crosses_minor_or_feature_domains() {
    let requested = route(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
    );
    for (cc, nvrtc_arch, ptx_target) in [
        ((10, 0), "compute_103f", "sm_103f"),
        ((10, 0), "compute_103a", "sm_103a"),
        ((10, 3), "compute_100f", "sm_100f"),
        ((10, 3), "compute_100a", "sm_100a"),
        ((10, 0), "compute_100", "sm_100"),
        ((10, 3), "compute_103", "sm_103"),
    ] {
        let candidate = Sm100TargetCandidate {
            device_cc: cc,
            nvrtc_arch,
            ptx_target,
            kind: Sm100TargetKind::Exact,
        };
        assert_eq!(
            resolve_sm100_forced(cc, Some(candidate), requested).unwrap(),
            None
        );
    }
    assert_eq!(
        resolve_sm100_forced((8, 9), Some(sm100_target_candidates((10, 0))[0]), requested,)
            .unwrap(),
        None
    );
    assert_eq!(
        resolve_sm100_forced((10, 0), None, requested).unwrap(),
        None
    );
}

#[test]
fn sm100_maps_use_driver_abi_and_element_aligned_logical_pointers() {
    assert_eq!(
        std::mem::size_of::<Sm100TensorMap>(),
        std::mem::size_of::<cudarc::driver::sys::CUtensorMap>()
    );
    assert_eq!(
        std::mem::align_of::<Sm100TensorMap>(),
        std::mem::align_of::<cudarc::driver::sys::CUtensorMap>()
    );
    let shape = shape(Sm100Op::Nn);
    for base in [0x1002_u64, 0x1008, 0x1010, 0x1020] {
        validate_sm100_map_request(Sm100MapRequest {
            op: Sm100Op::Nn,
            dtype: WeightDtype::Bf16,
            tile: Sm100Tile::M128N64,
            a_ptr: base,
            b_ptr: base + 0x1000,
            shape,
        })
        .unwrap();
    }
    let error = validate_sm100_map_request(Sm100MapRequest {
        op: Sm100Op::Nn,
        dtype: WeightDtype::F16,
        tile: Sm100Tile::M128N128,
        a_ptr: 0x1001,
        b_ptr: 0x2000,
        shape,
    })
    .unwrap_err();
    assert!(error.contains("element aligned"), "{error}");
}

#[test]
fn sm100_maps_reject_bad_strides_and_zero_shapes() {
    let mut request = Sm100MapRequest {
        op: Sm100Op::Nn,
        dtype: WeightDtype::Bf16,
        tile: Sm100Tile::M128N64,
        a_ptr: 0x1000,
        b_ptr: 0x2000,
        shape: shape(Sm100Op::Nn),
    };
    request.shape.lda -= 1;
    assert!(validate_sm100_map_request(request).is_err());

    request.shape = shape(Sm100Op::Nn);
    request.shape.ldb = 1;
    assert!(validate_sm100_map_request(request).is_err());

    for field in [0, 1, 2] {
        request.shape = shape(Sm100Op::Nn);
        match field {
            0 => request.shape.m = 0,
            1 => request.shape.k = 0,
            _ => request.shape.n = 0,
        }
        assert!(validate_sm100_map_request(request).is_err());
    }
}

#[test]
fn sm100_capture_launch_stays_prepared_and_side_effect_free() {
    let launch = public_function_source(LAUNCH_SOURCE, "launch_sm100_tcgen_prepared");
    assert_eq!(launch.matches("builder.arg(").count(), 5);
    for forbidden in [
        "compute_capability",
        "capture_status",
        "sm100_map_binding",
        "allocation_identities",
        "validate_live_allocations",
        "tensor_map_keys",
        "encode_sm100_tensor_maps",
        "compile_sm100",
        "probe_sm100",
    ] {
        assert!(
            !launch.contains(forbidden),
            "capture launch contains {forbidden}"
        );
    }

    for name in ["prepare_sm100_tensor_maps", "prepare_sm100_tcgen_forced"] {
        let prepare = public_function_source(LAUNCH_SOURCE, name);
        let capture = prepare.find("capture_status").expect("capture guard");
        let capability = prepare
            .find("sm100_map_binding")
            .expect("capability and target binding");
        assert!(
            capture < capability,
            "{name} must reject capture before capability or allocation work"
        );
    }
}

#[test]
fn sm100_cuda_abi_carries_coordinate_origins_to_every_tma_load() {
    for required in [
        "struct Sm100KernelParams {",
        "static_assert(sizeof(Sm100KernelParams) == 40",
        "static_assert(alignof(Sm100KernelParams) == 4",
        "sizeof(((Sm100KernelParams*)0)->a_x) == 4",
        "sizeof(((Sm100KernelParams*)0)->a_y) == 4",
        "sizeof(((Sm100KernelParams*)0)->b_x) == 4",
        "sizeof(((Sm100KernelParams*)0)->b_y) == 4",
        "sizeof(((Sm100KernelParams*)0)->alpha) == 4",
        "sizeof(((Sm100KernelParams*)0)->beta) == 4",
        "sizeof(((Sm100KernelParams*)0)->m) == 4",
        "sizeof(((Sm100KernelParams*)0)->k) == 4",
        "sizeof(((Sm100KernelParams*)0)->n) == 4",
        "sizeof(((Sm100KernelParams*)0)->ldc) == 4",
        "const __grid_constant__ Sm100KernelParams params",
        "int map_x = x + origin_x;",
        "int map_y = y + origin_y;",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing CUDA ABI contract: {required}"
        );
    }

    assert_eq!(
        SOURCE.matches("params.a_x, params.a_y").count(),
        7,
        "every A tensor-map load must carry the A coordinate origin"
    );
    assert_eq!(
        SOURCE.matches("params.b_x, params.b_y").count(),
        8,
        "every B tensor-map load must carry the B coordinate origin"
    );
}

#[test]
fn sm100_cuda_tile_indices_are_i32_boundary_safe() {
    for required in [
        "unsigned column_tiles = 1U +",
        "static_cast<unsigned>(output_columns) - 1U",
        "unsigned output_row_value = (blockIdx.x / column_tiles) * 128U;",
        "unsigned output_col_value =",
        "(blockIdx.x % column_tiles) * static_cast<unsigned>(Columns);",
        "int output_row = static_cast<int>(output_row_value);",
        "int output_col = static_cast<int>(output_col_value);",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing safe tile math: {required}"
        );
    }
    assert!(!SOURCE.contains("(output_columns + Columns - 1) / Columns"));
}
