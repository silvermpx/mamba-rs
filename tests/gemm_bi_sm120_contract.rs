#![cfg(feature = "cuda")]

use std::collections::{BTreeSet, HashSet};

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, SM120_KERNEL_SPECS, SM120_STREAMK_CELLS_CC120,
    SM120_STREAMK_CELLS_CC121, SM120_STREAMK_KERNEL_SPECS, Sm120Bk, Sm120ForcedRoute,
    Sm120MapRequest, Sm120Op, Sm120PhysicalRoute, Sm120Schedule, Sm120Shape, Sm120Stages,
    Sm120TensorMap, Sm120Tile, resolve_sm120_forced, sm120_target_candidates,
    validate_sm120_map_request,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{CudaTarget, DeviceCaps};

const SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120/tma.cu");
const CONTRACT_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs");
const LAUNCH_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs");
const MODULE_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs");
const BLAS_SOURCE: &str = include_str!("../src/mamba_ssm/gpu/blas.rs");

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

fn function_source(source: &str, name: &str) -> String {
    let marker = format!("fn {name}");
    let start = source.find(&marker).expect("function source");
    let tail = &source[start..];
    let body = tail.find('{').expect("function body");
    let mut depth = 0_usize;
    for (offset, byte) in tail[body..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return tail[..=body + offset].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function source for {name}")
}

fn contains_opcode_prefix(source: &str, prefix: &str) -> bool {
    source.match_indices(prefix).any(|(offset, _)| {
        source[..offset].chars().next_back().is_none_or(|previous| {
            !previous.is_ascii_alphanumeric() && !matches!(previous, '_' | '.')
        })
    })
}

fn tiles() -> [Sm120Tile; 4] {
    [
        Sm120Tile::M64N64,
        Sm120Tile::M128N64,
        Sm120Tile::M64N128,
        Sm120Tile::M128N128,
    ]
}

fn expected_symbols() -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for op in ["nn", "tn", "nt"] {
        for tile in ["64x64", "128x64", "64x128", "128x128"] {
            for bk in [32, 64] {
                for stages in [2, 3] {
                    for dtype in ["bf16", "f16"] {
                        symbols.insert(format!("{op}_sm120_tma_{tile}_bk{bk}_s{stages}_{dtype}"));
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
                .strip_prefix("SM120_DEFINE_KERNEL(")?
                .split(',')
                .next()
                .map(|symbol| symbol.trim().to_string())
        })
        .collect()
}

fn shape(op: Sm120Op) -> Sm120Shape {
    match op {
        Sm120Op::Nn | Sm120Op::Tn => Sm120Shape {
            m: 129,
            k: 65,
            n: 127,
            lda: 72,
            ldb: 128,
            ldc: 130,
        },
        Sm120Op::Nt => Sm120Shape {
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
    op: Sm120Op,
    dtype: WeightDtype,
    tile: Sm120Tile,
    bk: Sm120Bk,
    stages: Sm120Stages,
) -> Sm120ForcedRoute {
    Sm120ForcedRoute {
        op,
        dtype,
        physical: Sm120PhysicalRoute {
            tile,
            bk,
            stages,
            schedule: Sm120Schedule::Tiled,
        },
        shape: shape(op),
    }
}

fn caps(
    compute_capability: (u32, u32),
    nvrtc_version: (i32, i32),
    accepted_target: Option<&str>,
) -> DeviceCaps {
    DeviceCaps {
        compute_capability,
        nvrtc_version,
        accepted_target: accepted_target.map(|target| CudaTarget::new(target).unwrap()),
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
    }
}

#[test]
fn sm120_source_owns_exact_96_symbol_inventory() {
    let expected = expected_symbols();
    assert_eq!(expected.len(), 96);
    assert_eq!(source_declared_symbols(), expected);
}

#[test]
fn sm120_tensor_maps_promote_full_l2_sectors() {
    let start = CONTRACT_SOURCE
        .find("impl Sm120TensorMap {")
        .expect("SM120 tensor-map encoder");
    let body = &CONTRACT_SOURCE[start..];
    let end = body
        .find("pub(super) struct Sm120TensorOrigins")
        .expect("SM120 tensor-map encoder end");
    let body = &body[..end];
    assert!(body.contains("CU_TENSOR_MAP_L2_PROMOTION_L2_256B"));
    assert!(!body.contains("CU_TENSOR_MAP_L2_PROMOTION_NONE"));
}

#[test]
fn sm120_kernel_specs_cover_every_forced_route_once() {
    assert_eq!(SM120_KERNEL_SPECS.len(), 96);
    assert!(
        SM120_KERNEL_SPECS
            .iter()
            .all(|spec| spec.physical.schedule == Sm120Schedule::Tiled)
    );
    // The stream-K bodies: TN only, one per dtype, over the batch tile.
    assert_eq!(SM120_STREAMK_KERNEL_SPECS.len(), 2);
    for spec in SM120_STREAMK_KERNEL_SPECS {
        assert_eq!(spec.op, Sm120Op::Tn, "{}", spec.symbol);
        assert_eq!(
            spec.physical.schedule,
            Sm120Schedule::StreamK,
            "{}",
            spec.symbol
        );
        assert_eq!(spec.physical.tile, Sm120Tile::M64N64, "{}", spec.symbol);
        assert_eq!(spec.physical.bk, Sm120Bk::Bk64, "{}", spec.symbol);
        assert_eq!(spec.physical.stages, Sm120Stages::S3, "{}", spec.symbol);
        assert!(spec.symbol.ends_with("_streamk_bf16") || spec.symbol.ends_with("_streamk_f16"));
        assert_eq!(spec.threads, 128, "{}", spec.symbol);
        assert_eq!(spec.dynamic_shared_bytes, 49_280, "{}", spec.symbol);
    }
    let symbols: BTreeSet<_> = SM120_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol.to_string())
        .collect();
    assert_eq!(symbols, expected_symbols());

    let mut routes = HashSet::new();
    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for tile in tiles() {
                for bk in [Sm120Bk::Bk32, Sm120Bk::Bk64] {
                    for stages in [Sm120Stages::S2, Sm120Stages::S3] {
                        let requested = route(op, dtype, tile, bk, stages);
                        let spec = requested.kernel_spec().unwrap();
                        assert_eq!(spec.op, op);
                        assert_eq!(spec.dtype, dtype);
                        assert_eq!(spec.physical, requested.physical);
                        let dtype_tag = match dtype {
                            WeightDtype::F32 => 0,
                            WeightDtype::F16 => 1,
                            WeightDtype::Bf16 => 2,
                        };
                        assert!(routes.insert((op, dtype_tag, tile, bk, stages)));
                    }
                }
            }
        }
    }
    assert_eq!(routes.len(), 96);
}

#[test]
fn sm120_specs_freeze_threads_barriers_and_exact_shared_bytes() {
    let expected = [
        (
            Sm120Tile::M64N64,
            Sm120Bk::Bk32,
            Sm120Stages::S2,
            128,
            16_512,
        ),
        (
            Sm120Tile::M64N64,
            Sm120Bk::Bk32,
            Sm120Stages::S3,
            128,
            24_704,
        ),
        (
            Sm120Tile::M64N64,
            Sm120Bk::Bk64,
            Sm120Stages::S2,
            128,
            32_896,
        ),
        (
            Sm120Tile::M64N64,
            Sm120Bk::Bk64,
            Sm120Stages::S3,
            128,
            49_280,
        ),
        (
            Sm120Tile::M128N64,
            Sm120Bk::Bk32,
            Sm120Stages::S2,
            256,
            24_704,
        ),
        (
            Sm120Tile::M128N64,
            Sm120Bk::Bk32,
            Sm120Stages::S3,
            256,
            36_992,
        ),
        (
            Sm120Tile::M128N64,
            Sm120Bk::Bk64,
            Sm120Stages::S2,
            256,
            49_280,
        ),
        (
            Sm120Tile::M128N64,
            Sm120Bk::Bk64,
            Sm120Stages::S3,
            256,
            73_856,
        ),
        (
            Sm120Tile::M64N128,
            Sm120Bk::Bk32,
            Sm120Stages::S2,
            256,
            24_704,
        ),
        (
            Sm120Tile::M64N128,
            Sm120Bk::Bk32,
            Sm120Stages::S3,
            256,
            36_992,
        ),
        (
            Sm120Tile::M64N128,
            Sm120Bk::Bk64,
            Sm120Stages::S2,
            256,
            49_280,
        ),
        (
            Sm120Tile::M64N128,
            Sm120Bk::Bk64,
            Sm120Stages::S3,
            256,
            73_856,
        ),
        (
            Sm120Tile::M128N128,
            Sm120Bk::Bk32,
            Sm120Stages::S2,
            256,
            32_896,
        ),
        (
            Sm120Tile::M128N128,
            Sm120Bk::Bk32,
            Sm120Stages::S3,
            256,
            49_280,
        ),
        (
            Sm120Tile::M128N128,
            Sm120Bk::Bk64,
            Sm120Stages::S2,
            512,
            65_664,
        ),
        (
            Sm120Tile::M128N128,
            Sm120Bk::Bk64,
            Sm120Stages::S3,
            512,
            98_432,
        ),
    ];
    for spec in SM120_KERNEL_SPECS {
        let &(.., threads, shared) = expected
            .iter()
            .find(|&&(tile, bk, stages, _, _)| {
                (tile, bk, stages) == (spec.physical.tile, spec.physical.bk, spec.physical.stages)
            })
            .unwrap_or_else(|| panic!("missing geometry for {}", spec.symbol));
        assert_eq!(spec.threads, threads, "{} threads", spec.symbol);
        assert_eq!(
            spec.empty_barrier_arrivals,
            spec.physical.compute_warps(),
            "{} empty arrivals",
            spec.symbol
        );
        assert_eq!(
            spec.full_barrier_arrivals, 1,
            "{} full arrivals",
            spec.symbol
        );
        assert_eq!(spec.dynamic_shared_bytes, shared, "{} smem", spec.symbol);
        assert_eq!(spec.cluster, (1, 1, 1), "{} cluster", spec.symbol);
        assert_eq!(
            spec.warp_tile,
            spec.physical.warp_tile(),
            "{} warp tile",
            spec.symbol
        );
        assert_eq!(spec.threads / 32, spec.physical.compute_warps());
        assert_eq!(
            spec.expected_transaction_bytes,
            match spec.physical.bk {
                Sm120Bk::Bk32 => match spec.physical.tile {
                    Sm120Tile::M64N64 => 8_192,
                    Sm120Tile::M128N64 | Sm120Tile::M64N128 => 12_288,
                    Sm120Tile::M128N128 => 16_384,
                },
                Sm120Bk::Bk64 => match spec.physical.tile {
                    Sm120Tile::M64N64 => 16_384,
                    Sm120Tile::M128N64 | Sm120Tile::M64N128 => 24_576,
                    Sm120Tile::M128N128 => 32_768,
                },
            }
        );
    }
}

#[test]
fn sm120_auto_tables_are_minor_specific() {
    assert_eq!(SM120_AUTO_CELLS_CC120.len(), 60);
    assert_eq!(SM120_AUTO_CELLS_CC121, &[]);
    assert_eq!(SM120_STREAMK_CELLS_CC120.len(), 12);
    assert_eq!(SM120_STREAMK_CELLS_CC121, &[]);
}

#[test]
fn sm120_targets_follow_real_toolkit_floors_and_generic_fallback_order() {
    assert!(sm120_target_candidates((12, 0), (12, 7)).is_empty());
    let cc120_128 = sm120_target_candidates((12, 0), (12, 8));
    assert_eq!(cc120_128.len(), 1);
    assert_eq!(cc120_128[0].nvrtc_arch, "compute_120");
    assert_eq!(cc120_128[0].ptx_target, "sm_120");

    let cc121_128 = sm120_target_candidates((12, 1), (12, 8));
    assert_eq!(cc121_128.len(), 1);
    assert_eq!(cc121_128[0].nvrtc_arch, "compute_120");
    assert_eq!(cc121_128[0].ptx_target, "sm_120");

    let cc121_129 = sm120_target_candidates((12, 1), (12, 9));
    assert_eq!(cc121_129.len(), 2);
    assert_eq!(cc121_129[0].nvrtc_arch, "compute_121");
    assert_eq!(cc121_129[0].ptx_target, "sm_121");
    assert_eq!(cc121_129[1].nvrtc_arch, "compute_120");
    assert_eq!(cc121_129[1].ptx_target, "sm_120");

    assert!(sm120_target_candidates((12, 2), (13, 2)).is_empty());
    assert!(sm120_target_candidates((10, 0), (13, 2)).is_empty());
    for candidate in cc120_128.iter().chain(cc121_129) {
        assert!(!candidate.nvrtc_arch.ends_with('a'));
        assert!(!candidate.nvrtc_arch.ends_with('f'));
        assert!(!candidate.ptx_target.ends_with('a'));
        assert!(!candidate.ptx_target.ends_with('f'));
    }
}

#[test]
fn sm120_forced_selector_covers_96_routes_and_rejects_f32() {
    let target = sm120_target_candidates((12, 0), (12, 8))[0];
    let mut symbols = BTreeSet::new();
    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for tile in tiles() {
                for bk in [Sm120Bk::Bk32, Sm120Bk::Bk64] {
                    for stages in [Sm120Stages::S2, Sm120Stages::S3] {
                        let resolved = resolve_sm120_forced(
                            caps((12, 0), (12, 8), Some("compute_120")),
                            Some(target),
                            route(op, dtype, tile, bk, stages),
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
        resolve_sm120_forced(
            caps((12, 0), (12, 8), Some("compute_120")),
            Some(target),
            route(
                Sm120Op::Nn,
                WeightDtype::F32,
                Sm120Tile::M64N64,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
            ),
        )
        .is_err()
    );
}

#[test]
fn sm120_forced_selector_fails_closed_on_wrong_target_or_shape() {
    let requested = route(
        Sm120Op::Nn,
        WeightDtype::Bf16,
        Sm120Tile::M128N128,
        Sm120Bk::Bk64,
        Sm120Stages::S3,
    );
    let cc120 = sm120_target_candidates((12, 0), (12, 8))[0];
    let cc121 = sm120_target_candidates((12, 1), (12, 9))[0];
    let cc121_fallback = sm120_target_candidates((12, 1), (12, 9))[1];
    assert_eq!(
        resolve_sm120_forced(caps((12, 0), (12, 8), Some("compute_120")), None, requested,)
            .unwrap(),
        None
    );
    assert_eq!(
        resolve_sm120_forced(
            caps((12, 0), (12, 8), Some("compute_120")),
            Some(cc121),
            requested,
        )
        .unwrap(),
        None
    );
    assert_eq!(
        resolve_sm120_forced(
            caps((12, 1), (12, 9), Some("compute_120")),
            Some(cc121_fallback),
            requested,
        )
        .unwrap(),
        Some(requested)
    );
    assert_eq!(
        resolve_sm120_forced(
            caps((8, 9), (13, 2), Some("compute_120")),
            Some(cc120),
            requested,
        )
        .unwrap(),
        None
    );
    let mut no_maps = caps((12, 0), (12, 8), Some("compute_120"));
    no_maps.tensor_map_access = false;
    assert_eq!(
        resolve_sm120_forced(no_maps, Some(cc120), requested).unwrap(),
        None
    );
    let mut too_little_shared = caps((12, 0), (12, 8), Some("compute_120"));
    too_little_shared.optin_shared_bytes = 98_431;
    assert_eq!(
        resolve_sm120_forced(too_little_shared, Some(cc120), requested).unwrap(),
        None
    );

    for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
        for axis in 0..3 {
            let mut invalid = route(
                op,
                WeightDtype::F16,
                Sm120Tile::M64N64,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
            );
            match axis {
                0 => invalid.shape.m = 0,
                1 => invalid.shape.k = 0,
                _ => invalid.shape.n = 0,
            }
            assert!(
                resolve_sm120_forced(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(cc120),
                    invalid,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn sm120_tensor_map_wrapper_tracks_the_active_cuda_header_abi() {
    assert_eq!(std::mem::size_of::<Sm120TensorMap>(), 128);
    assert_eq!(
        std::mem::size_of::<Sm120TensorMap>(),
        std::mem::size_of::<cudarc::driver::sys::CUtensorMap>()
    );
    assert_eq!(
        std::mem::align_of::<Sm120TensorMap>(),
        std::mem::align_of::<cudarc::driver::sys::CUtensorMap>()
    );
    assert!(matches!(std::mem::align_of::<Sm120TensorMap>(), 64 | 128));

    for required in [
        "#if __CUDACC_VER_MAJOR__ >= 13",
        "struct alignas(128) CUtensorMap",
        "struct alignas(64) CUtensorMap",
        "static_assert(sizeof(CUtensorMap) == 128",
        "static_assert(alignof(CUtensorMap) == 128",
        "static_assert(alignof(CUtensorMap) == 64",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing portable ABI guard: {required}"
        );
    }
}

#[test]
fn sm120_map_requests_cover_sw64_sw128_and_reject_unsupported_views() {
    let aligned_shape = Sm120Shape {
        m: 129,
        k: 65,
        n: 127,
        lda: 72,
        ldb: 128,
        ldc: 130,
    };
    for tile in tiles() {
        for bk in [Sm120Bk::Bk32, Sm120Bk::Bk64] {
            validate_sm120_map_request(Sm120MapRequest {
                op: Sm120Op::Nn,
                dtype: WeightDtype::Bf16,
                tile,
                bk,
                a_ptr: 0x1_0000,
                b_ptr: 0x2_0000,
                shape: aligned_shape,
            })
            .unwrap();
        }
    }
    for bad in [0, 0x1_0001, 0x1_0003] {
        let error = validate_sm120_map_request(Sm120MapRequest {
            op: Sm120Op::Nn,
            dtype: WeightDtype::F16,
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            a_ptr: bad,
            b_ptr: 0x2_0000,
            shape: aligned_shape,
        })
        .unwrap_err();
        assert!(
            error.contains("16-byte aligned") || error.contains("non-null"),
            "{error}"
        );
    }

    let mut bad_stride = aligned_shape;
    bad_stride.lda = 71;
    assert!(
        validate_sm120_map_request(Sm120MapRequest {
            op: Sm120Op::Nn,
            dtype: WeightDtype::Bf16,
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk32,
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            shape: bad_stride,
        })
        .is_err()
    );
}

#[test]
fn sm120_swizzled_maps_require_encoded_and_logical_tma_alignment() {
    let key_start = CONTRACT_SOURCE
        .find("impl Sm120TensorMapKey {")
        .expect("SM120 tensor-map key validator");
    let key_tail = &CONTRACT_SOURCE[key_start..];
    let key_end = key_tail
        .find("\n#[repr(transparent)]")
        .expect("end of SM120 tensor-map key validator");
    let key_validator = &key_tail[..key_end];

    for required in [
        "!self.base.is_multiple_of(128)",
        "!self.outer_byte_stride.is_multiple_of(16)",
    ] {
        assert!(
            key_validator.contains(required),
            "missing SM120 tensor-map alignment contract: {required}"
        );
    }
    assert!(
        !key_validator.contains("!self.base.is_multiple_of(16)"),
        "swizzled SM120 tensor-map base must not use the unswizzled 16-byte rule"
    );
    assert!(CONTRACT_SOURCE.contains("!layout.pointer.is_multiple_of(16)"));
}

#[test]
fn sm120_source_freezes_tma_swizzle_and_pipeline_contract() {
    for required in [
        "CU_TENSOR_MAP_DATA_TYPE_UINT16",
        "CU_TENSOR_MAP_SWIZZLE_64B",
        "CU_TENSOR_MAP_SWIZZLE_128B",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.init.shared::cta.b64",
        "fence.mbarrier_init.release.cluster",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "mbarrier.arrive.release.cta.shared::cta.b64",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "sm120_init_barrier<warps>",
        "if ((threadIdx.x & 31) == 0) {\n            sm120_arrive_empty",
        "__syncwarp()",
        "__align__(128)",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing SM120 contract: {required}"
        );
    }
    for forbidden in [
        "tcgen05",
        "tmem",
        "wgmma",
        "setmaxnreg",
        "multicast",
        "shared::cluster",
        "cta_group::2",
        "multimem",
        "mapa",
        "clusterlaunchcontrol",
        "griddepcontrol",
        "cudaMalloc",
        "cudaFree",
        "malloc(",
        "operator new",
    ] {
        assert!(
            !SOURCE.contains(forbidden),
            "forbidden SM120 feature {forbidden}"
        );
    }
    for forbidden_opcode in ["atom.", "red.", "redux."] {
        assert!(
            !contains_opcode_prefix(SOURCE, forbidden_opcode),
            "forbidden SM120 opcode {forbidden_opcode}"
        );
    }
    assert!(contains_opcode_prefix(
        "asm(\"red.global.add.u32\");",
        "red."
    ));
    assert!(!contains_opcode_prefix(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
        "red."
    ));
    // Eleven in the tiled kernels, three more in the stream-K mainloop.
    assert_eq!(SOURCE.matches("sm120_sync_warp();").count(), 14);
}

#[test]
fn sm120_device_tile_counts_use_overflow_safe_positive_ceil_division() {
    for extent in [
        1_u32,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        i32::MAX as u32,
    ] {
        for tile in [32_u32, 64, 128] {
            let expected = u64::from(extent).div_ceil(u64::from(tile)) as u32;
            let positive_ceil_div = 1 + (extent - 1) / tile;
            assert_eq!(positive_ceil_div, expected, "extent={extent}, tile={tile}");
        }
    }

    for unsafe_expression in ["(output_columns + N - 1) / N", "(reduction + BK - 1) / BK"] {
        assert!(
            !SOURCE.contains(unsafe_expression),
            "signed tile count can overflow at i32::MAX: {unsafe_expression}"
        );
    }
    for safe_expression in ["1 + (output_columns - 1) / N", "1 + (reduction - 1) / BK"] {
        assert!(
            SOURCE.contains(safe_expression),
            "missing overflow-safe positive ceil division: {safe_expression}"
        );
    }
}

#[test]
fn sm120_source_accounts_for_every_swizzle_offset_and_split_plane() {
    for required in [
        "shared_address + logical_row * groups * 16U",
        "(row_start / 128U) % groups",
        "logical_chunk ^ phase",
        "groups = 4",
        "groups = 8",
        "+ 64",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing swizzle proof hook: {required}"
        );
    }
    assert!(
        SOURCE.matches("SM120_SWIZZLE_64B").count() >= 2,
        "SW64 must be selected and consumed"
    );
    assert!(
        SOURCE.matches("SM120_SWIZZLE_128B").count() >= 2,
        "SW128 must be selected and consumed"
    );
}

#[test]
fn sm120_half_swizzle_matches_the_absolute_address_bit_permutation() {
    let decode = |base: u32, row: u32, element: u32, groups: u32| {
        let row_start = base + row * groups * 16;
        let phase = (row_start / 128) % groups;
        row_start + ((element / 8) ^ phase) * 16
    };
    for base_phase in 0_u32..8 {
        let base = base_phase * 128;
        for row in 0_u32..16 {
            for chunk in 0_u32..4 {
                let expected_chunk = chunk ^ ((base_phase + row / 2) % 4);
                assert_eq!(
                    decode(base, row, chunk * 8, 4),
                    base + row * 64 + expected_chunk * 16,
                    "SW64 base_phase={base_phase} row={row} chunk={chunk}"
                );
            }
            for chunk in 0_u32..8 {
                let expected_chunk = chunk ^ ((base_phase + row) % 8);
                assert_eq!(
                    decode(base, row, chunk * 8, 8),
                    base + row * 128 + expected_chunk * 16,
                    "SW128 base_phase={base_phase} row={row} chunk={chunk}"
                );
            }
        }
    }
}

#[test]
fn sm120_cuda_abi_is_five_parameters_and_carries_all_route_coordinates() {
    for required in [
        "struct Sm120KernelParams {",
        "static_assert(sizeof(Sm120KernelParams) == 40",
        "static_assert(alignof(Sm120KernelParams) == 4",
        "const __grid_constant__ CUtensorMap a_map",
        "const __grid_constant__ CUtensorMap b_map",
        "const __grid_constant__ Sm120KernelParams params",
        "params.a_x",
        "params.a_y",
        "params.b_x",
        "params.b_y",
        "params.alpha",
        "params.beta",
        "params.m",
        "params.k",
        "params.n",
        "params.ldc",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing CUDA ABI item: {required}"
        );
    }
    // Five arguments for the tiled ABI, plus the two stream-K workspace
    // pointers pushed under their schedule guard.
    let launch = public_function_source(LAUNCH_SOURCE, "launch_sm120_tma_prepared");
    assert_eq!(launch.matches("builder.arg(").count(), 7);
}

#[test]
fn sm120_capture_launch_is_prepared_and_has_no_replay_side_effects() {
    let launch = public_function_source(LAUNCH_SOURCE, "launch_sm120_tma_prepared");
    for forbidden in [
        "compute_capability",
        "capture_status",
        "pointer_get_attribute",
        "allocation_identities",
        "validate_live_allocations",
        "tensor_map_plan",
        "encode_sm120_tensor_maps",
        "compile_sm120",
        "probe_sm120",
        "fallback",
    ] {
        assert!(
            !launch.contains(forbidden),
            "capture launch contains {forbidden}"
        );
    }
    for name in ["prepare_sm120_tensor_maps", "prepare_sm120_tma_forced"] {
        let prepare = public_function_source(LAUNCH_SOURCE, name);
        let capture = prepare.find("capture_status").expect("capture guard");
        let capability = prepare
            .find("sm120_map_binding")
            .expect("capability and target binding");
        assert!(
            capture < capability,
            "{name} must reject capture before capability, query, encode, or allocation work"
        );
    }
}

#[test]
fn sm120_auto_bridge_is_request_based_and_capture_prepared_only() {
    let bridge = function_source(LAUNCH_SOURCE, "launch_sm120_auto_observed");
    for required in [
        "resolve_sm120_auto(",
        "with_sm120_prepared_launches",
        "Sm120PreparedKey::new",
    ] {
        assert!(bridge.contains(required), "auto bridge omits {required}");
    }
    for forbidden in [
        "resolve_sm120_forced(",
        "gemm_bi_forward_typed",
        "gemm_bi_backward_dw_typed",
        "gemm_bi_backward_dx_typed",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "auto bridge contains {forbidden}"
        );
    }

    let cache = function_source(LAUNCH_SOURCE, "ensure_sm120_prepared");
    let decision = cache.find("sm120_cache_action(").expect("capture decision");
    for preparation in ["prepare_sm120_tensor_maps(", "prepare_sm120_tma_forced("] {
        assert!(
            decision < cache.find(preparation).expect("eager preparation"),
            "{preparation} occurs before the capture decision"
        );
    }
    let errors = function_source(LAUNCH_SOURCE, "sm120_capture_cache_error");
    for error in [
        "prepared SM120 Triad cache entry is missing during graph capture; run eager warmup again",
        "prepared SM120 Triad allocation epoch changed during graph capture; run eager warmup again",
        "prepared SM120 Triad automatic capture requires managed allocations; run eager warmup again",
    ] {
        assert!(
            errors.contains(error),
            "cache omits fail-closed error {error}"
        );
    }

    let enqueue = function_source(LAUNCH_SOURCE, "enqueue_sm120_tma_prepared_observed");
    assert_eq!(enqueue.matches("builder.arg(").count(), 7);
    assert!(enqueue.contains("if let Some(workspace) = &workspace"));
    assert!(enqueue.contains("enqueue_with_physical_observation"));
    for forbidden in [
        "capture_status",
        "validate_sm120_graph_replay",
        "prepare_sm120_tensor_maps",
        "prepare_sm120_tma_forced",
        "pointer_get_attribute",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "prepared enqueue contains {forbidden}"
        );
    }
}

#[test]
fn sm120_half_graph_package_uses_the_cached_prepared_route() {
    let package = function_source(BLAS_SOURCE, "prepare_half_physical_graph_package");
    for required in [
        "ModuleKind::TriadSm120",
        "prepare_sm120_auto_graph_sequence",
        "Sm120AutoRequest",
        "manifest.nodes().len() != 1",
    ] {
        assert!(
            package.contains(required),
            "half graph package omits {required}"
        );
    }

    let adapter = function_source(LAUNCH_SOURCE, "prepare_sm120_auto_graph_sequence");
    assert_eq!(
        adapter.matches("validate_sm120_graph_replay(").count(),
        2,
        "prepared SM120 graph adapter must validate before and after binding"
    );
    for forbidden in [
        "prepare_sm120_tensor_maps(",
        "prepare_sm120_tma_forced(",
        "encode_sm120_tensor_maps",
        "resolve_sm120_forced(",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "graph adapter contains {forbidden}"
        );
    }
}

#[test]
fn sm120_module_selection_is_a_complete_same_target_artifact_transaction() {
    let selector = MODULE_SOURCE
        .find("fn compile_sm120_artifact_set")
        .map(|start| &MODULE_SOURCE[start..])
        .expect("whole-artifact SM120 selector");
    for required in [
        "ModuleKind::Fixed",
        "ModuleKind::TriadScalar",
        "ModuleKind::TriadSm80",
        "ModuleKind::TriadSm120",
        "qualify_specialized_module",
    ] {
        assert!(
            selector.contains(required),
            "artifact transaction omits {required}"
        );
    }
    assert!(
        selector.find("candidate.nvrtc_arch").unwrap()
            < selector.find("ModuleKind::Fixed").unwrap(),
        "one accepted target must bind every module in the artifact set"
    );
}

#[test]
fn sm120_source_closes_every_private_macro() {
    let mut definitions = BTreeSet::new();
    let mut live = BTreeSet::new();
    for line in SOURCE.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("#define SM120_") {
            let suffix = rest
                .split(|ch: char| ch.is_ascii_whitespace() || ch == '(')
                .next()
                .unwrap();
            let name = format!("SM120_{suffix}");
            assert!(live.insert(name.clone()), "duplicate definition {name}");
            definitions.insert(name);
        }
        if let Some(rest) = line.strip_prefix("#undef SM120_") {
            let suffix = rest.split_ascii_whitespace().next().unwrap();
            let name = format!("SM120_{suffix}");
            assert!(live.remove(&name), "undef without a live definition {name}");
        }
    }
    assert!(!definitions.is_empty());
    assert!(live.is_empty(), "live macros at end of fragment: {live:?}");
}

#[test]
fn streamk_cells_are_the_persistent_twins_of_measured_tn_cells() {
    let streamk = Sm120PhysicalRoute {
        tile: Sm120Tile::M64N64,
        bk: Sm120Bk::Bk64,
        stages: Sm120Stages::S3,
        schedule: Sm120Schedule::StreamK,
    };
    for cell in SM120_STREAMK_CELLS_CC120 {
        assert_eq!(cell.op, Sm120Op::Tn, "{cell:?}");
        assert_eq!(cell.physical, streamk, "{cell:?}");
        assert_eq!(
            cell.shape,
            Sm120Shape::contiguous(cell.op, (cell.shape.m, cell.shape.k, cell.shape.n))
        );
        let spec = cell.kernel_spec().expect("stream-K cell resolves a spec");
        assert!(
            SM120_STREAMK_KERNEL_SPECS
                .iter()
                .any(|known| known.symbol == spec.symbol),
            "{} is not a stream-K symbol",
            spec.symbol
        );
        // Every stream-K shape keeps a tiled cell, so the default half policy
        // never loses a measured route to the opt-in table.
        let tiled = SM120_AUTO_CELLS_CC120
            .iter()
            .filter(|tiled| {
                tiled.op == cell.op && tiled.dtype == cell.dtype && tiled.shape == cell.shape
            })
            .count();
        assert_eq!(tiled, 1, "{cell:?} needs exactly one tiled cell");
    }
    let shapes = |dtype: WeightDtype| {
        SM120_STREAMK_CELLS_CC120
            .iter()
            .filter(|cell| cell.dtype == dtype)
            .map(|cell| (cell.shape.m, cell.shape.k, cell.shape.n))
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(shapes(WeightDtype::Bf16), shapes(WeightDtype::F16));
    assert_eq!(shapes(WeightDtype::Bf16).len(), 6);
    assert_eq!(
        shapes(WeightDtype::Bf16).len() * 2,
        SM120_STREAMK_CELLS_CC120.len(),
        "no duplicate stream-K cell"
    );
}
