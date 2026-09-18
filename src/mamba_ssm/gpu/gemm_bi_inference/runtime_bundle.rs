use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::super::context::F32TriadPolicy;
use super::super::context::GpuCtx;
use super::super::dtype::WeightDtype;
use super::super::kernel_identity::{
    PhysicalLaunchObserver, PolicyDtype, enqueue_with_physical_observation,
};
use super::{
    FixedArgs, FixedSm89ExactF32Params, FixedSm89HalfParams, InferenceFwdOperands, InferenceShape,
    InferenceTile, identity, source_bundle,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InferenceBundleRoute {
    HalfF32S3,
    ExactF32M128N64Tail,
}

#[derive(Clone, Copy)]
struct InferenceBundleStack<'a> {
    compute_capability: (u32, u32),
    multiprocessors: u32,
    target: &'a str,
    state_cap: usize,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    half_bf16_available: bool,
    half_f16_available: bool,
    exact_available: bool,
}

enum InferenceBundleParams {
    Half(FixedSm89HalfParams),
    Exact(FixedSm89ExactF32Params),
}

trait BundleParameterWords {
    fn words(&self) -> [u32; 10];
}

macro_rules! bundle_parameter_words {
    ($params:ty) => {
        impl BundleParameterWords for $params {
            fn words(&self) -> [u32; 10] {
                [
                    self.alpha.to_bits(),
                    self.beta.to_bits(),
                    self.m as u32,
                    self.n as u32,
                    self.k as u32,
                    self.lda as u32,
                    self.ldb as u32,
                    self.ldc as u32,
                    0,
                    0,
                ]
            }
        }
    };
}

bundle_parameter_words!(FixedSm89HalfParams);
bundle_parameter_words!(FixedSm89ExactF32Params);

struct PreparedInferenceBundleLaunch {
    args: FixedArgs,
    params: InferenceBundleParams,
    config: LaunchConfig,
}

fn select_inference_bundle_route(
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    stack: InferenceBundleStack<'_>,
    policy: F32TriadPolicy,
) -> Option<InferenceBundleRoute> {
    let (major, minor) = stack.compute_capability;
    if !stack.nvrtc_library_known
        || stack.multiprocessors == 0
        || !source_bundle::compiler_supported(
            i32::try_from(major).ok().zip(i32::try_from(minor).ok()),
            stack.target,
            stack.state_cap,
            stack.nvrtc,
        )
        || [operands.c.ptr, operands.x.ptr, operands.w.ptr]
            .into_iter()
            .any(|pointer| pointer == 0 || !pointer.is_multiple_of(16))
        || operands
            .bias_ptr
            .is_some_and(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
    {
        return None;
    }

    let mixed_inputs = operands.x.dtype != WeightDtype::F32
        && operands.x.dtype == operands.w.dtype
        && operands.c.dtype == WeightDtype::F32;
    let mixed_available = match operands.x.dtype {
        WeightDtype::Bf16 => stack.half_bf16_available,
        WeightDtype::F16 => stack.half_f16_available,
        WeightDtype::F32 | WeightDtype::Tf32 => false,
    };
    if mixed_inputs
        && mixed_available
        && matches!(
            (shape.m, shape.k, shape.n),
            (4621, 768, 2304) | (4621, 1928, 384) | (2048, 2304, 768)
        )
    {
        return Some(InferenceBundleRoute::HalfF32S3);
    }

    let exact_f32 = operands.c.dtype == WeightDtype::F32
        && operands.x.dtype == WeightDtype::F32
        && operands.w.dtype == WeightDtype::F32;
    if exact_f32
        && stack.exact_available
        && policy == F32TriadPolicy::ExactScalarFma
        && (shape.m, shape.k, shape.n) == (4621, 1928, 384)
    {
        return Some(InferenceBundleRoute::ExactF32M128N64Tail);
    }
    None
}

pub(super) fn select_for_context(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) -> Option<InferenceBundleRoute> {
    let compiler = ctx.kernels.compiler_identity();
    select_inference_bundle_route(
        operands,
        shape,
        InferenceBundleStack {
            compute_capability: ctx.compute_capability(),
            multiprocessors: ctx.kernels.multiprocessor_count(),
            target: compiler.target.as_str(),
            state_cap: ctx.state_cap(),
            nvrtc: compiler.nvrtc_version,
            nvrtc_library_known: compiler.nvrtc_library_known,
            half_bf16_available: ctx
                .kernels
                .inference_sm89_bundle
                .half_f32out_s3_bf16
                .is_ok(),
            half_f16_available: ctx.kernels.inference_sm89_bundle.half_f32out_s3_f16.is_ok(),
            exact_available: ctx.kernels.inference_sm89_bundle.exact_m128n64_tail.is_ok(),
        },
        ctx.f32_triad_policy(),
    )
}

fn prepare_inference_bundle_launch(
    route: InferenceBundleRoute,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) -> Result<Option<PreparedInferenceBundleLaunch>, String> {
    // An empty output does no address arithmetic and preserves the established
    // inference zero-work contract even when unused input pointers are null.
    if shape.m == 0 || shape.n == 0 {
        return Ok(None);
    }
    let args = FixedArgs::try_new(operands, shape)?;
    let (params, tile_n, dynamic_shared) = match route {
        InferenceBundleRoute::HalfF32S3 => {
            if operands.x.dtype == WeightDtype::F32
                || operands.x.dtype != operands.w.dtype
                || operands.c.dtype != WeightDtype::F32
            {
                return Err(
                    "retained Inference mixed S3 requires matching bf16/f16 inputs and f32 output"
                        .into(),
                );
            }
            if args.c == 0 || !args.c.is_multiple_of(4) {
                return Err("retained Inference mixed S3 requires non-null f32-aligned C".into());
            }
            if [args.a, args.b]
                .into_iter()
                .any(|pointer| pointer == 0 || !pointer.is_multiple_of(2))
            {
                return Err(
                    "retained Inference mixed S3 requires non-null half-aligned A and B".into(),
                );
            }
            if operands
                .bias_ptr
                .is_some_and(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
            {
                return Err(
                    "retained Inference mixed S3 requires a non-null f32-aligned optional bias"
                        .into(),
                );
            }
            args.m
                .checked_add(127)
                .ok_or("retained Inference mixed S3 padded M exceeds i32")?;
            args.n
                .checked_add(127)
                .ok_or("retained Inference mixed S3 padded N exceeds i32")?;
            // The S3 schedule computes its next refill slab one stage ahead.
            args.k
                .checked_add(127)
                .ok_or("retained Inference mixed S3 padded K exceeds i32")?;
            validate_byte_end(args.c, args.m, args.n, 4, "mixed C")?;
            validate_byte_end(args.a, args.m, args.k, 2, "mixed A")?;
            validate_byte_end(args.b, args.k, args.n, 2, "mixed B")?;
            if args.bias != 0 {
                validate_byte_end(args.bias, 1, args.n, 4, "mixed bias")?;
            }
            (
                InferenceBundleParams::Half(FixedSm89HalfParams {
                    alpha: 1.0,
                    beta: 0.0,
                    m: args.m,
                    n: args.n,
                    k: args.k,
                    lda: args.k,
                    ldb: args.n,
                    ldc: args.n,
                }),
                128,
                98_304,
            )
        }
        InferenceBundleRoute::ExactF32M128N64Tail => {
            if [operands.c.dtype, operands.x.dtype, operands.w.dtype]
                .into_iter()
                .any(|dtype| dtype != WeightDtype::F32)
            {
                return Err(
                    "retained Inference exact M128N64 requires homogeneous f32 operands".into(),
                );
            }
            if operands
                .bias_ptr
                .is_some_and(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
            {
                return Err(
                    "retained Inference exact M128N64 requires a non-null f32-aligned optional bias"
                        .into(),
                );
            }
            if [args.c, args.a, args.b]
                .into_iter()
                .any(|pointer| pointer == 0 || !pointer.is_multiple_of(4))
            {
                return Err(
                    "retained Inference exact M128N64 requires non-null f32-aligned A, B, and C"
                        .into(),
                );
            }
            args.m
                .checked_add(127)
                .ok_or("retained Inference exact M128N64 padded M exceeds i32")?;
            args.n
                .checked_add(63)
                .ok_or("retained Inference exact M128N64 padded N exceeds i32")?;
            args.k
                .checked_add(31)
                .ok_or("retained Inference exact M128N64 padded K exceeds i32")?;
            validate_byte_end(args.c, args.m, args.n, 4, "exact C")?;
            validate_byte_end(args.a, args.m, args.k, 4, "exact A")?;
            validate_byte_end(args.b, args.k, args.n, 4, "exact B")?;
            if args.bias != 0 {
                validate_byte_end(args.bias, 1, args.n, 4, "exact bias")?;
            }
            (
                InferenceBundleParams::Exact(FixedSm89ExactF32Params::forward(&args)),
                64,
                0,
            )
        }
    };
    let rows = u32::try_from(args.m).map_err(|_| "retained Inference M is negative")?;
    let cols = u32::try_from(args.n).map_err(|_| "retained Inference N is negative")?;
    let grid = rows
        .div_ceil(128)
        .checked_mul(cols.div_ceil(tile_n))
        .filter(|grid| *grid <= i32::MAX as u32)
        .ok_or("retained Inference launch grid exceeds i32")?;
    Ok(Some(PreparedInferenceBundleLaunch {
        args,
        params,
        config: LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: dynamic_shared,
        },
    }))
}

fn validate_byte_end(
    pointer: u64,
    rows: i32,
    cols: i32,
    element_bytes: u64,
    label: &str,
) -> Result<(), String> {
    let bytes = u64::try_from(rows)
        .ok()
        .and_then(|rows| {
            u64::try_from(cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| format!("retained Inference {label} byte extent exceeds u64"))?;
    pointer
        .checked_add(bytes)
        .ok_or_else(|| format!("retained Inference {label} byte endpoint exceeds u64"))?;
    Ok(())
}

/// The retained member a route launches for `input`, the name the proof
/// ledger keys on.
pub(super) fn member_symbol(route: InferenceBundleRoute, input: WeightDtype) -> &'static str {
    match (route, input) {
        (InferenceBundleRoute::HalfF32S3, WeightDtype::F16) => "nn_sm89_tc128_f32out_s3_f16",
        (InferenceBundleRoute::HalfF32S3, _) => "nn_sm89_tc128_f32out_s3_bf16",
        (InferenceBundleRoute::ExactF32M128N64Tail, _) => "nn_sm89_f32_m128n64_tail_copyplan",
    }
}

pub(super) fn family_label(route: InferenceBundleRoute) -> InferenceTile {
    match route {
        InferenceBundleRoute::HalfF32S3 => InferenceTile::Tc128Sm89S3,
        InferenceBundleRoute::ExactF32M128N64Tail => InferenceTile::F32Sm89N64CopyPlan,
    }
}

pub(super) fn legacy_force_s3_dtype_supported(operands: InferenceFwdOperands) -> bool {
    operands.c.dtype != WeightDtype::F32
        && operands.c.dtype == operands.x.dtype
        && operands.x.dtype == operands.w.dtype
}

pub(super) fn launch_inference_bundle<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    route: InferenceBundleRoute,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    observer: &mut O,
) -> Result<(), String> {
    let Some(prepared) = prepare_inference_bundle_launch(route, operands, shape)? else {
        return Ok(());
    };
    // Loader admission owns attributes and resources, so capture only binds
    // the already-loaded function, fixed ABI arguments, and launch config.
    let function = match (route, operands.x.dtype) {
        (InferenceBundleRoute::HalfF32S3, WeightDtype::Bf16) => ctx
            .kernels
            .inference_sm89_bundle
            .half_f32out_s3_bf16
            .as_ref(),
        (InferenceBundleRoute::HalfF32S3, WeightDtype::F16) => ctx
            .kernels
            .inference_sm89_bundle
            .half_f32out_s3_f16
            .as_ref(),
        (InferenceBundleRoute::ExactF32M128N64Tail, WeightDtype::F32) => ctx
            .kernels
            .inference_sm89_bundle
            .exact_m128n64_tail
            .as_ref(),
        _ => return Err("retained Inference route and input dtype disagree".into()),
    }
    .map_err(|reason| reason.clone())?;
    let args = &prepared.args;
    let mut builder = ctx.stream.launch_builder(function);
    builder
        .arg(&args.c)
        .arg(&args.a)
        .arg(&args.b)
        .arg(&args.bias);
    match &prepared.params {
        InferenceBundleParams::Half(params) => {
            builder.arg(params);
        }
        InferenceBundleParams::Exact(params) => {
            builder.arg(params);
        }
    }
    let (storage, abi, words) = match &prepared.params {
        InferenceBundleParams::Half(params) => (
            [
                identity::policy_dtype(operands.x.dtype),
                identity::policy_dtype(operands.w.dtype),
                PolicyDtype::F32,
            ],
            identity::AbiKind::HalfSm89,
            params.words(),
        ),
        InferenceBundleParams::Exact(params) => (
            [PolicyDtype::F32; 3],
            identity::AbiKind::ExactF32,
            params.words(),
        ),
    };
    let config = prepared.config;
    let observation =
        identity::observation(ctx, observer, function, config, || identity::Arguments {
            pointers: [args.c, args.a, args.b, args.bias],
            storage,
            abi,
            words,
            maps: None,
            auxiliary: [0; 2],
        })?;
    unsafe { enqueue_with_physical_observation(observer, &mut builder, config, observation) }
        .map_err(|error| {
            error.with_driver_context(format_args!("Inference retained bundle launch"))
        })
}

#[cfg(test)]
mod inference_bundle_runtime_tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::super::super::blas::TypedPtr;
    use super::super::super::dtype::WeightDtype;
    use super::*;

    const HOT_B: InferenceShape = InferenceShape {
        m: 4621,
        k: 768,
        n: 2304,
    };
    const HOT_C: InferenceShape = InferenceShape {
        m: 4621,
        k: 1928,
        n: 384,
    };
    const HOT_E: InferenceShape = InferenceShape {
        m: 2048,
        k: 2304,
        n: 768,
    };

    fn mixed(dtype: WeightDtype, bias: Option<u64>) -> InferenceFwdOperands {
        InferenceFwdOperands {
            c: TypedPtr {
                ptr: 0x1000,
                dtype: WeightDtype::F32,
            },
            x: TypedPtr { ptr: 0x2000, dtype },
            w: TypedPtr { ptr: 0x3000, dtype },
            bias_ptr: bias,
        }
    }

    fn exact() -> InferenceFwdOperands {
        let f32_ptr = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        InferenceFwdOperands {
            c: f32_ptr(0x1000),
            x: f32_ptr(0x2000),
            w: f32_ptr(0x3000),
            bias_ptr: None,
        }
    }

    fn stack(nvrtc: (i32, i32), state_cap: usize) -> InferenceBundleStack<'static> {
        InferenceBundleStack {
            compute_capability: (8, 9),
            multiprocessors: 142,
            target: "sm_89",
            state_cap,
            nvrtc,
            nvrtc_library_known: true,
            half_bf16_available: true,
            half_f16_available: true,
            exact_available: true,
        }
    }

    fn select(
        operands: InferenceFwdOperands,
        shape: InferenceShape,
        stack: InferenceBundleStack<'_>,
        policy: F32TriadPolicy,
    ) -> Option<InferenceBundleRoute> {
        select_inference_bundle_route(operands, shape, stack, policy)
    }

    #[test]
    fn inference_bundle_runtime_selects_all_literal_six_compiler_cohorts() {
        let cohorts = [
            ((12, 8), 16),
            ((12, 8), 64),
            ((13, 0), 16),
            ((13, 0), 64),
            ((13, 2), 16),
            ((13, 2), 64),
        ];
        assert_eq!(cohorts.len(), 6);
        for (nvrtc, state_cap) in cohorts {
            let compiler = stack(nvrtc, state_cap);
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for shape in [HOT_B, HOT_C, HOT_E] {
                    for bias in [None, Some(0x4004)] {
                        assert_eq!(
                            select(
                                mixed(dtype, bias),
                                shape,
                                compiler,
                                F32TriadPolicy::ExactScalarFma,
                            ),
                            Some(InferenceBundleRoute::HalfF32S3),
                            "mixed winner missing for {nvrtc:?}/cap{state_cap} {dtype:?} {shape:?} bias={bias:?}"
                        );
                    }
                }
            }
            for bias in [None, Some(0x4004)] {
                assert_eq!(
                    select(
                        InferenceFwdOperands {
                            bias_ptr: bias,
                            ..exact()
                        },
                        HOT_C,
                        compiler,
                        F32TriadPolicy::ExactScalarFma,
                    ),
                    Some(InferenceBundleRoute::ExactF32M128N64Tail),
                    "exact winner missing for {nvrtc:?}/cap{state_cap} bias={bias:?}"
                );
            }
        }
    }

    #[test]
    fn inference_bundle_runtime_declines_each_closed_selector_gate() {
        let good = stack((13, 2), 64);
        let mixed_ops = mixed(WeightDtype::Bf16, Some(0x4004));
        let exact_ops = exact();
        for other_board in [
            InferenceBundleStack {
                compute_capability: (8, 6),
                ..good
            },
            InferenceBundleStack {
                multiprocessors: 141,
                ..good
            },
        ] {
            assert!(
                select(
                    mixed_ops,
                    HOT_B,
                    other_board,
                    F32TriadPolicy::ExactScalarFma
                )
                .is_some(),
                "an sm_80-tier board outside the frozen evidence takes the proof path"
            );
            assert!(
                select(
                    exact_ops,
                    HOT_C,
                    other_board,
                    F32TriadPolicy::ExactScalarFma
                )
                .is_some()
            );
        }
        let bad_stacks = [
            InferenceBundleStack {
                state_cap: 32,
                ..good
            },
            InferenceBundleStack {
                compute_capability: (7, 5),
                ..good
            },
            InferenceBundleStack {
                compute_capability: (12, 0),
                ..good
            },
            InferenceBundleStack {
                multiprocessors: 0,
                ..good
            },
            InferenceBundleStack {
                target: "compute_89",
                ..good
            },
            InferenceBundleStack {
                nvrtc: (13, 1),
                ..good
            },
            InferenceBundleStack {
                nvrtc_library_known: false,
                ..good
            },
        ];
        for bad in bad_stacks {
            assert_eq!(
                select(mixed_ops, HOT_B, bad, F32TriadPolicy::ExactScalarFma,),
                None
            );
            assert_eq!(
                select(exact_ops, HOT_C, bad, F32TriadPolicy::ExactScalarFma,),
                None
            );
        }

        for shape in [
            InferenceShape { m: 4620, ..HOT_B },
            InferenceShape { k: 767, ..HOT_B },
            InferenceShape { n: 2303, ..HOT_B },
            InferenceShape { m: 0, ..HOT_B },
        ] {
            assert_eq!(
                select(mixed_ops, shape, good, F32TriadPolicy::ExactScalarFma,),
                None
            );
        }
        assert_eq!(
            select(exact_ops, HOT_B, good, F32TriadPolicy::ExactScalarFma,),
            None
        );
        for bad_bias in [0, 0x4001, 0x4002, 0x4003] {
            assert_eq!(
                select(
                    InferenceFwdOperands {
                        bias_ptr: Some(bad_bias),
                        ..exact_ops
                    },
                    HOT_C,
                    good,
                    F32TriadPolicy::ExactScalarFma,
                ),
                None
            );
        }
        assert_eq!(
            select(
                exact_ops,
                HOT_C,
                good,
                F32TriadPolicy::AllowDeterministicTf32,
            ),
            None
        );

        for field in 0..3 {
            for ptr in [0, 0x1008] {
                let mut bad = mixed_ops;
                match field {
                    0 => bad.c.ptr = ptr,
                    1 => bad.x.ptr = ptr,
                    _ => bad.w.ptr = ptr,
                }
                assert_eq!(
                    select(bad, HOT_B, good, F32TriadPolicy::ExactScalarFma),
                    None
                );
            }
        }
        for bad_bias in [0, 0x4001, 0x4002, 0x4003] {
            assert_eq!(
                select(
                    mixed(WeightDtype::Bf16, Some(bad_bias)),
                    HOT_B,
                    good,
                    F32TriadPolicy::ExactScalarFma,
                ),
                None
            );
        }

        for bad in [
            InferenceFwdOperands {
                w: TypedPtr {
                    dtype: WeightDtype::F16,
                    ..mixed_ops.w
                },
                ..mixed_ops
            },
            InferenceFwdOperands {
                c: TypedPtr {
                    dtype: WeightDtype::Bf16,
                    ..mixed_ops.c
                },
                ..mixed_ops
            },
        ] {
            assert_eq!(
                select(bad, HOT_B, good, F32TriadPolicy::ExactScalarFma),
                None
            );
        }
    }

    #[test]
    fn inference_bundle_runtime_requires_only_the_selected_individual_holder() {
        let good = stack((13, 2), 64);
        let without_bf16 = InferenceBundleStack {
            half_bf16_available: false,
            ..good
        };
        assert_eq!(
            select(
                mixed(WeightDtype::Bf16, None),
                HOT_B,
                without_bf16,
                F32TriadPolicy::ExactScalarFma,
            ),
            None
        );
        assert_eq!(
            select(
                mixed(WeightDtype::F16, None),
                HOT_B,
                without_bf16,
                F32TriadPolicy::ExactScalarFma,
            ),
            Some(InferenceBundleRoute::HalfF32S3)
        );
        assert_eq!(
            select(exact(), HOT_C, without_bf16, F32TriadPolicy::ExactScalarFma,),
            Some(InferenceBundleRoute::ExactF32M128N64Tail)
        );
        assert_eq!(
            select(
                mixed(WeightDtype::F16, None),
                HOT_C,
                InferenceBundleStack {
                    half_f16_available: false,
                    ..good
                },
                F32TriadPolicy::ExactScalarFma,
            ),
            None
        );
        assert_eq!(
            select(
                exact(),
                HOT_C,
                InferenceBundleStack {
                    exact_available: false,
                    ..good
                },
                F32TriadPolicy::ExactScalarFma,
            ),
            None
        );
    }

    fn prepared(
        route: InferenceBundleRoute,
        operands: InferenceFwdOperands,
        shape: InferenceShape,
    ) -> PreparedInferenceBundleLaunch {
        let Some(prepared) = prepare_inference_bundle_launch(route, operands, shape).unwrap()
        else {
            panic!("nonempty qualified request must produce a launch");
        };
        prepared
    }

    #[test]
    fn inference_bundle_runtime_prepares_literal_grids_and_64_byte_abi() {
        let mixed_b = prepared(
            InferenceBundleRoute::HalfF32S3,
            mixed(WeightDtype::Bf16, Some(0x4004)),
            HOT_B,
        );
        let mixed_c = prepared(
            InferenceBundleRoute::HalfF32S3,
            mixed(WeightDtype::F16, None),
            HOT_C,
        );
        let exact_c = prepared(InferenceBundleRoute::ExactF32M128N64Tail, exact(), HOT_C);
        for (prepared, grid, dynamic) in [
            (&mixed_b, 666, 98_304),
            (&mixed_c, 111, 98_304),
            (&exact_c, 222, 0),
        ] {
            let words = match &prepared.params {
                InferenceBundleParams::Half(params) => params.words(),
                InferenceBundleParams::Exact(params) => params.words(),
            };
            assert_eq!(prepared.config.grid_dim, (grid, 1, 1));
            assert_eq!(prepared.config.block_dim, (256, 1, 1));
            assert_eq!(prepared.config.shared_mem_bytes, dynamic);
            assert_eq!(words[0], 1.0f32.to_bits());
            assert_eq!(words[1], 0.0f32.to_bits());
            assert_eq!(words[2], 4621);
            assert_eq!(words[5], words[4]);
            assert_eq!(words[6], words[3]);
            assert_eq!(words[7], words[3]);
        }
        for (size, align) in [
            (
                size_of::<FixedSm89HalfParams>(),
                align_of::<FixedSm89HalfParams>(),
            ),
            (
                size_of::<FixedSm89ExactF32Params>(),
                align_of::<FixedSm89ExactF32Params>(),
            ),
        ] {
            assert_eq!(size, 32);
            assert_eq!(align, 4);
            assert_eq!(4 * size_of::<u64>() + size, 64);
        }
        assert_eq!(offset_of!(FixedSm89HalfParams, alpha), 0);
        assert_eq!(offset_of!(FixedSm89HalfParams, beta), 4);
        assert_eq!(offset_of!(FixedSm89HalfParams, m), 8);
        assert_eq!(offset_of!(FixedSm89HalfParams, n), 12);
        assert_eq!(offset_of!(FixedSm89HalfParams, k), 16);
        assert_eq!(offset_of!(FixedSm89HalfParams, lda), 20);
        assert_eq!(offset_of!(FixedSm89HalfParams, ldb), 24);
        assert_eq!(offset_of!(FixedSm89HalfParams, ldc), 28);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, alpha), 0);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, beta), 4);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, m), 8);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, n), 12);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, k), 16);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, lda), 20);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, ldb), 24);
        assert_eq!(offset_of!(FixedSm89ExactF32Params, ldc), 28);
    }

    #[test]
    fn inference_bundle_runtime_preparer_preserves_zero_work_and_rejects_wrap() {
        for route in [
            InferenceBundleRoute::HalfF32S3,
            InferenceBundleRoute::ExactF32M128N64Tail,
        ] {
            for shape in [
                InferenceShape { m: 0, ..HOT_C },
                InferenceShape { n: 0, ..HOT_C },
            ] {
                let mut operands = exact();
                operands.c.ptr = 0;
                operands.x.ptr = 0;
                operands.w.ptr = 0;
                assert!(
                    prepare_inference_bundle_launch(route, operands, shape)
                        .unwrap()
                        .is_none()
                );
            }
        }

        let too_large = InferenceShape {
            m: i32::MAX as usize + 1,
            ..HOT_C
        };
        assert!(
            prepare_inference_bundle_launch(
                InferenceBundleRoute::ExactF32M128N64Tail,
                exact(),
                too_large,
            )
            .is_err()
        );
        let mut address_wrap = mixed(WeightDtype::Bf16, None);
        address_wrap.c.ptr = u64::MAX - 15;
        assert!(
            prepare_inference_bundle_launch(InferenceBundleRoute::HalfF32S3, address_wrap, HOT_B,)
                .is_err()
        );
        let mut wrong_input_domain = mixed(WeightDtype::Bf16, None);
        wrong_input_domain.x.ptr = 0x2001;
        assert!(
            prepare_inference_bundle_launch(
                InferenceBundleRoute::HalfF32S3,
                wrong_input_domain,
                HOT_B,
            )
            .is_err()
        );
        let mut wrong_output_domain = mixed(WeightDtype::F16, None);
        wrong_output_domain.c.ptr = 0x1002;
        assert!(
            prepare_inference_bundle_launch(
                InferenceBundleRoute::HalfF32S3,
                wrong_output_domain,
                HOT_C,
            )
            .is_err()
        );
    }

    #[test]
    fn inference_bundle_runtime_keeps_family_labels_distinct_from_force_semantics() {
        assert_eq!(
            family_label(InferenceBundleRoute::HalfF32S3),
            InferenceTile::Tc128Sm89S3
        );
        assert_eq!(
            family_label(InferenceBundleRoute::ExactF32M128N64Tail),
            InferenceTile::F32Sm89N64CopyPlan
        );
        assert!(!legacy_force_s3_dtype_supported(mixed(
            WeightDtype::Bf16,
            None
        )));
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let typed = |ptr| TypedPtr { ptr, dtype };
            assert!(legacy_force_s3_dtype_supported(InferenceFwdOperands {
                c: typed(0x1000),
                x: typed(0x2000),
                w: typed(0x3000),
                bias_ptr: None,
            }));
        }

        let Some((_, old_grid)) =
            super::super::prepare_sm89_exact_n64_launch(exact(), HOT_C, (8, 9)).unwrap()
        else {
            panic!("old exact force request must remain launchable");
        };
        let new_exact = prepared(InferenceBundleRoute::ExactF32M128N64Tail, exact(), HOT_C);
        assert_eq!(old_grid, 438);
        assert_eq!(new_exact.config.grid_dim, (222, 1, 1));
    }

    #[test]
    fn inference_bundle_runtime_retains_bias_seed_before_first_mma() {
        let source = include_str!("../../../../kernels/gemm_bi_inference/sm80/half_f32out_s3.cu");
        let bias_seed = source.find("float first = bias != nullptr").unwrap();
        let first_mma = source
            .find("sm89_fixed_half_swizzle::consume_fragments<T>")
            .unwrap();
        assert!(bias_seed < first_mma);
    }
}
