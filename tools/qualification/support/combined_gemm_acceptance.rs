//! Current-only physical census and released AUTO word acceptance.

use super::gpu_quiet;
use super::*;
use cudarc::driver::{CudaFunction, LaunchConfig, sys};
use mamba_rs::mamba_ssm::gpu::context::HalfTriadPolicy;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    PhysicalQualificationF32Epilogue, QualifiedPhysicalLaunchEvidence, SM89_HALF_AUTO_CELLS,
    SM89_HALF_KERNEL_SPECS, Sm89HalfRoute,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{ArtifactIdentity, CompilerIdentity};
use serde_json::{Value, json};
use std::collections::BTreeMap;
const HARNESS_SOURCE: &str = include_str!("../gemm_bi_inference_performance.rs");
const CURRENT_ACCEPTANCE_SOURCE: &[u8] = include_bytes!("combined_gemm_acceptance.rs");

#[path = "combined_gemm_words.rs"]
mod words;

pub(super) struct DriverNode {
    index: usize,
    pub params: sys::CUDA_KERNEL_NODE_PARAMS,
    pub symbol: String,
}

pub(super) struct DriverGraph {
    pub count: usize,
    pub non_kernel_nodes: usize,
    pub kernels: Vec<DriverNode>,
    edges: Vec<(usize, usize)>,
}

fn driver(result: sys::CUresult, label: &str) -> Result<(), String> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{label}: {result:?}"))
    }
}

pub(super) fn read_driver_graph(graph: &CudaGraph) -> Result<DriverGraph, String> {
    let mut count = 0;
    driver(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
        "graph node count",
    )?;
    if count == 0 {
        return Err("captured no work".into());
    }
    let mut handles = vec![std::ptr::null_mut(); count];
    driver(
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), handles.as_mut_ptr(), &mut count) },
        "graph nodes",
    )?;
    if handles.len() != count {
        return Err("graph node count changed".into());
    }
    let mut kernels = Vec::new();
    for (index, &handle) in handles.iter().enumerate() {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        driver(
            unsafe { sys::cuGraphNodeGetType(handle, &mut kind) },
            "graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            continue;
        }
        let mut params = unsafe { std::mem::zeroed() };
        driver(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(handle, &mut params) },
            "kernel node parameters",
        )?;
        let symbol = function_name(params.func)?;
        kernels.push(DriverNode {
            index,
            params,
            symbol,
        });
    }
    let mut edge_count = 0;
    driver(
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph.cu_graph(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut edge_count,
            )
        },
        "graph edge count",
    )?;
    let mut from = vec![std::ptr::null_mut(); edge_count];
    let mut to = vec![std::ptr::null_mut(); edge_count];
    let full_dependency = sys::CUgraphEdgeData {
        from_port: 0,
        to_port: 0,
        type_: 0,
        reserved: [0; 5],
    };
    let mut edge_data = vec![full_dependency; edge_count];
    if edge_count > 0 {
        driver(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph.cu_graph(),
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    edge_data.as_mut_ptr(),
                    &mut edge_count,
                )
            },
            "graph edges",
        )?;
    }
    if from.len() != edge_count {
        return Err("graph edge count changed".into());
    }
    // A plain topological edge proves ordering only for full serialization.
    // Read the v2 annotations instead of silently discarding partial dependencies.
    if edge_data.iter().any(|data| *data != full_dependency) {
        return Err(format!(
            "graph has non-default dependency data: {edge_data:?}"
        ));
    }
    let edges = from
        .into_iter()
        .zip(to)
        .map(|(a, b)| {
            Ok((
                handles
                    .iter()
                    .position(|&h| h == a)
                    .ok_or("unknown edge origin")?,
                handles
                    .iter()
                    .position(|&h| h == b)
                    .ok_or("unknown edge target")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(DriverGraph {
        count,
        non_kernel_nodes: count - kernels.len(),
        kernels,
        edges,
    })
}

fn function_name(function: sys::CUfunction) -> Result<String, String> {
    let mut name = std::ptr::null();
    driver(
        unsafe { sys::cuFuncGetName(&mut name, function) },
        "function name",
    )?;
    if name.is_null() {
        return Err("null function name".into());
    }
    unsafe { CStr::from_ptr(name) }
        .to_str()
        .map(str::to_owned)
        .map_err(|e| format!("function name UTF8: {e}"))
}

fn topological_order(count: usize, edges: &[(usize, usize)]) -> Result<Vec<usize>, String> {
    if count == 0 {
        return Err("graph has no nodes".into());
    }
    let mut unique = std::collections::BTreeSet::new();
    let mut indegree = vec![0; count];
    for &(from, to) in edges {
        if from >= count || to >= count || from == to || !unique.insert((from, to)) {
            return Err("invalid or duplicate dependency edge".into());
        }
        indegree[to] += 1;
    }
    let mut order = Vec::with_capacity(count);
    while order.len() < count {
        let ready = (0..count)
            .filter(|index| indegree[*index] == 0 && !order.contains(index))
            .collect::<Vec<_>>();
        if ready.len() != 1 {
            return Err("graph dependencies do not define one complete launch order".into());
        }
        let next = ready[0];
        order.push(next);
        for &(from, to) in edges {
            if from == next {
                indegree[to] -= 1;
            }
        }
    }
    Ok(order)
}

fn validate_abi(kind: &str, parameters: &[(usize, usize)], terminal: bool) -> Result<(), String> {
    let expected: &[(usize, usize)] = match kind {
        "bundle" => &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
        "tn16" => &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)],
        "transpose" => &[(0, 8), (8, 8), (16, 4), (20, 4)],
        "legacy" => &[
            (0, 8),
            (8, 8),
            (16, 8),
            (24, 8),
            (32, 4),
            (36, 4),
            (40, 4),
            (44, 4),
            (48, 4),
            (52, 4),
            (56, 4),
            (60, 4),
        ],
        "scalar-anchor" => &[
            (0, 8),
            (8, 8),
            (16, 8),
            (24, 4),
            (28, 4),
            (32, 4),
            (36, 4),
            (40, 4),
        ],
        _ => return Err(format!("unknown ABI {kind}")),
    };
    if parameters != expected || !terminal {
        return Err(format!(
            "wrong {kind} ABI {parameters:?}, terminal_invalid_value={terminal}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ExpectedNode {
    symbol: String,
    module: ModuleKind,
    abi: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    dynamic: u32,
    static_bytes: Option<u32>,
    minimum_ctas: i32,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
}

fn tiled_node(
    symbol: String,
    module: ModuleKind,
    shape: (usize, usize),
    tile: (u32, u32),
    resources: (u32, u32, Option<u32>),
    abi: &'static str,
) -> ExpectedNode {
    ExpectedNode {
        symbol,
        module,
        abi,
        grid: (
            (shape.0 as u32).div_ceil(tile.0) * (shape.1 as u32).div_ceil(tile.1),
            1,
            1,
        ),
        block: (resources.0, 1, 1),
        dynamic: resources.1,
        static_bytes: resources.2,
        minimum_ctas: 1,
        tile,
        bk: 32,
        stages: 2,
    }
}

fn expected_nodes(
    case: &words::AcceptanceCase,
    nvrtc: (i32, i32),
) -> Result<Vec<ExpectedNode>, String> {
    let (m, k, n) = case.dims;
    let mut node = if case.family == "inference" {
        let shape = InferenceShape { m, k, n };
        match case.row {
            "bf16" | "f16" => {
                let tile = expected_ada_half_auto_v45(
                    nvrtc,
                    words::dtype(case.storage[0])?,
                    shape,
                    case.bias,
                )
                .ok_or("missing literal homogeneous-half AUTO expectation")?;
                let symbol = fixed_force_spec(case.row, (8, 9), tile)?
                    .expected_symbol
                    .to_owned();
                let (extent, threads, dynamic) = match tile {
                    InferenceTile::Tc128Sm89Pipeline => ((128, 128), 256, 71_680),
                    InferenceTile::Tc128Sm89Swizzle => ((128, 128), 256, 69_632),
                    InferenceTile::Tc128Sm89S3 => ((128, 128), 256, 98_304),
                    InferenceTile::TcM64N64Sm89S3 => ((64, 64), 128, 49_152),
                    InferenceTile::TcM128N64Sm89S2 => ((128, 64), 128, 49_152),
                    _ => return Err("unexpected homogeneous-half family".into()),
                };
                let mut node = tiled_node(
                    symbol,
                    ModuleKind::Fixed,
                    (m, n),
                    extent,
                    (threads, dynamic, Some(0)),
                    "bundle",
                );
                node.bk = 64;
                node.stages = if matches!(
                    tile,
                    InferenceTile::Tc128Sm89S3 | InferenceTile::TcM64N64Sm89S3
                ) {
                    3
                } else {
                    2
                };
                node
            }
            "bf16_f32" | "f16_f32" => {
                if matches!((m, k, n), (4621, 768, 2304) | (4621, 1928, 384)) {
                    let mut node = tiled_node(
                        format!(
                            "gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_{}",
                            case.storage[0]
                        ),
                        ModuleKind::Fixed,
                        (m, n),
                        (128, 128),
                        (256, 98_304, Some(0)),
                        "bundle",
                    );
                    node.bk = 64;
                    node.stages = 3;
                    node
                } else {
                    let large = (m, k, n) == (2048, 2304, 768);
                    tiled_node(
                        format!(
                            "gemm_bi_nn_tc{}_f32out_{}",
                            if large { 128 } else { 64 },
                            case.storage[0]
                        ),
                        ModuleKind::Fixed,
                        (m, n),
                        if large { (128, 128) } else { (64, 64) },
                        if large {
                            (256, 71_680, Some(0))
                        } else {
                            (128, 0, None)
                        },
                        "legacy",
                    )
                }
            }
            "tf32" => {
                let narrow = (m, k, n) == (2048, 2304, 768) && !case.bias;
                let mut node = tiled_node(
                    if narrow {
                        "gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3"
                    } else {
                        "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3"
                    }
                    .into(),
                    ModuleKind::Fixed,
                    (m, n),
                    if narrow { (128, 96) } else { (128, 128) },
                    (256, if narrow { 86_016 } else { 98_304 }, Some(0)),
                    "bundle",
                );
                node.stages = 3;
                node
            }
            "f32_exact" => {
                if (m, k, n) == (4621, 1928, 384) {
                    if case.bias {
                        tiled_node(
                            "gemm_bi_f32_f32_s2".into(),
                            ModuleKind::Fixed,
                            (m, n),
                            (64, 64),
                            (128, 0, None),
                            "legacy",
                        )
                    } else {
                        tiled_node(
                            "gemm_bi_nn_inference_sm89_f32_m128n64_tail_copyplan_v1".into(),
                            ModuleKind::Fixed,
                            (m, n),
                            (128, 64),
                            (256, 0, Some(49_152)),
                            "bundle",
                        )
                    }
                } else {
                    tiled_node(
                        "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1".into(),
                        ModuleKind::Fixed,
                        (m, n),
                        (64, 64),
                        (128, 0, None),
                        "bundle",
                    )
                }
            }
            _ => return Err("invalid Inference row".into()),
        }
    } else if case.storage[0] != "f32" {
        let rows = if case.op == "tn" { k } else { m };
        let columns = if case.op == "nt" { k } else { n };
        if case.op == "tn" && (m, k, n) == (1024, 256, 128) {
            let mut node = tiled_node(
                format!(
                    "gemm_bi_tn_sm89_m16n16_bk64_s2_ldb72_v1_{}",
                    case.storage[0]
                ),
                ModuleKind::TriadSm89Half,
                (rows, columns),
                (16, 16),
                (32, 0, Some(36_864)),
                "tn16",
            );
            node.bk = 64;
            node.minimum_ctas = 2;
            node
        } else {
            let op = logical_op(case.op)?;
            let dtype = words::dtype(case.storage[0])?;
            let entry = SM89_HALF_AUTO_CELLS
                .iter()
                .find(|entry| entry.0 == op && entry.1 == dtype && entry.2 == case.dims)
                .ok_or("missing legacy half AUTO cell")?;
            let spec = SM89_HALF_KERNEL_SPECS
                .iter()
                .find(|spec| spec.route == entry.3 && spec.dtype == dtype)
                .ok_or("missing literal half kernel spec")?;
            let mut node = tiled_node(
                spec.symbol.into(),
                ModuleKind::TriadSm89Half,
                (rows, columns),
                spec.tile,
                (
                    spec.threads,
                    spec.dynamic_shared_bytes,
                    Some(spec.static_shared_bytes),
                ),
                if case.op == "nn" { "bundle" } else { "tn16" },
            );
            node.minimum_ctas = spec.occupancy_gate as i32;
            node.bk = spec.bk;
            node.stages = spec.stages;
            node
        }
    } else if case.row == "tf32" {
        let finalist = (m, k, n) == (4621, 384, 1928);
        let mut node = tiled_node(
            if finalist {
                "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2"
            } else {
                "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1"
            }
            .into(),
            if finalist {
                ModuleKind::TriadSm89Finalist
            } else {
                ModuleKind::TriadSm89Tf32Joint
            },
            (m, k),
            if finalist { (128, 64) } else { (128, 96) },
            (256, if finalist { 49_152 } else { 86_016 }, Some(0)),
            "bundle",
        );
        node.stages = if finalist { 2 } else { 3 };
        node
    } else if case.op == "tn" {
        let mut node = tiled_node(
            "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1".into(),
            ModuleKind::TriadScalar,
            (k, n),
            (16, 16),
            (64, 4_096, Some(0)),
            "tn16",
        );
        node.minimum_ctas = 8;
        node.bk = 16;
        node
    } else {
        tiled_node(
            "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1".into(),
            ModuleKind::Fixed,
            (m, if case.op == "nt" { k } else { n }),
            (64, 64),
            (128, 0, None),
            "bundle",
        )
    };
    if case.family == "inference" && case.row.ends_with("_f32") && !case.required {
        node.bk = 64;
    }
    if case.family == "triad" && case.row == "f32_exact" && case.op == "nt" {
        let transpose = ExpectedNode {
            symbol: "gemm_bi_transpose_f32_32x16_d768_v1".into(),
            module: ModuleKind::TriadScalar,
            abi: "transpose",
            grid: ((n as u32).div_ceil(32), (k as u32).div_ceil(32), 1),
            block: (32, 16, 1),
            dynamic: 0,
            static_bytes: Some(4_224),
            minimum_ctas: 2,
            tile: (32, 32),
            bk: 1,
            stages: 1,
        };
        Ok(vec![transpose, node])
    } else {
        Ok(vec![node])
    }
}

fn logical_op(op: &str) -> Result<ResolvedGemmOp, String> {
    match op {
        "nn" => Ok(ResolvedGemmOp::Nn),
        "tn" => Ok(ResolvedGemmOp::Tn),
        "nt" => Ok(ResolvedGemmOp::Nt),
        _ => Err("unknown operation".into()),
    }
}

#[derive(Clone)]
struct FunctionReceipt {
    function: sys::CUfunction,
    module: sys::CUmodule,
    abi: Vec<(usize, usize)>,
    json: Value,
}

fn function_receipt(
    function: sys::CUfunction,
    spec: &ExpectedNode,
) -> Result<FunctionReceipt, String> {
    if function_name(function)? != spec.symbol {
        return Err(format!(
            "function name differs from literal {}",
            spec.symbol
        ));
    }
    let mut module = std::ptr::null_mut();
    driver(
        unsafe { sys::cuFuncGetModule(&mut module, function) },
        "borrowed function module",
    )?;
    if module.is_null() {
        return Err("null borrowed module".into());
    }
    let mut abi = Vec::new();
    for index in 0..128 {
        let mut offset = 0;
        let mut size = 0;
        let result = unsafe { sys::cuFuncGetParamInfo(function, index, &mut offset, &mut size) };
        if result == sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            break;
        }
        driver(result, "Driver parameter ABI")?;
        if size == 0 || size > 4096 {
            return Err("invalid Driver parameter size".into());
        }
        abi.push((offset, size));
    }
    validate_abi(spec.abi, &abi, abi.len() < 128)?;
    let attributes = [
        (
            "registers",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
        ),
        (
            "local_bytes",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        ),
        (
            "static_shared_bytes",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        ),
        (
            "max_threads",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ),
        (
            "max_dynamic_shared_bytes",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        ),
        (
            "binary_version",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        ),
        (
            "ptx_version",
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION,
        ),
    ];
    let mut resources = BTreeMap::new();
    for (name, attribute) in attributes {
        let mut value = 0;
        driver(
            unsafe { sys::cuFuncGetAttribute(&mut value, attribute, function) },
            name,
        )?;
        if value < 0 {
            return Err(format!("negative {name}"));
        }
        resources.insert(name, value);
    }
    let block_threads = spec.block.0 * spec.block.1 * spec.block.2;
    let mut active = 0;
    driver(
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut active,
                function,
                block_threads as i32,
                spec.dynamic as usize,
            )
        },
        "actual-block occupancy",
    )?;
    if resources["registers"] == 0
        || resources["local_bytes"] != 0
        || resources["max_threads"] < block_threads as i32
        || resources["max_dynamic_shared_bytes"] < spec.dynamic as i32
        || spec
            .static_bytes
            .is_some_and(|bytes| resources["static_shared_bytes"] != bytes as i32)
        || active < spec.minimum_ctas
    {
        return Err(format!(
            "{} resource rejection {resources:?}, active_ctas={active}",
            spec.symbol
        ));
    }
    let json = json!({"symbol":spec.symbol,"module_kind":format!("{:?}",spec.module),"borrowed_module":module as usize,"function":function as usize,"abi":abi,"terminal_invalid_value":true,"parameter_bytes":abi.last().map(|(offset,size)|offset+size),"block":spec.block,"dynamic_shared_bytes":spec.dynamic,"resources":resources,"active_ctas":active});
    Ok(FunctionReceipt {
        function,
        module,
        abi,
        json,
    })
}

fn argument_bytes(node: &DriverNode, abi: &[(usize, usize)]) -> Result<Vec<Vec<u8>>, String> {
    if node.params.kernelParams.is_null() {
        return Err("graph has no kernelParams".into());
    }
    abi.iter()
        .enumerate()
        .map(|(index, &(_, size))| {
            let pointer = unsafe { *node.params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("null graph argument {index}"));
            }
            Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) }.to_vec())
        })
        .collect()
}

fn expected_arguments(fixture: &words::Fixture, spec: &ExpectedNode) -> Vec<Vec<u8>> {
    argument_contract(&fixture.case, fixture.pointers(), fixture.scratch, spec)
}

fn argument_contract(
    case: &words::AcceptanceCase,
    pointers: [u64; 4],
    scratch: u64,
    spec: &ExpectedNode,
) -> Vec<Vec<u8>> {
    let (m, k, n) = case.dims;
    let [c, a, b, bias] = pointers;
    if spec.abi == "transpose" {
        return vec![
            scratch.to_le_bytes().to_vec(),
            b.to_le_bytes().to_vec(),
            (k as u32).to_le_bytes().to_vec(),
            (n as u32).to_le_bytes().to_vec(),
        ];
    }
    let mut pointers = vec![c, a, b];
    let transpose_second = case.family == "triad" && case.row == "f32_exact" && case.op == "nt";
    if transpose_second {
        pointers[2] = scratch;
    }
    let scalar = if spec.abi == "tn16" {
        vec![
            0x3f800000,
            m as u32,
            if case.op == "nt" { n } else { k } as u32,
            if case.op == "nt" { k } else { n } as u32,
        ]
    } else {
        pointers.push(bias);
        let (middle, last, lda, ldb, ldc) = if transpose_second {
            (k, n, n, k, k)
        } else if case.row == "tf32" {
            (
                k,
                n,
                if case.op == "nt" { n } else { k },
                n,
                if case.op == "nt" { k } else { n },
            )
        } else {
            (n, k, k, n, n)
        };
        vec![
            0x3f800000,
            0,
            m as u32,
            middle as u32,
            last as u32,
            lda as u32,
            ldb as u32,
            ldc as u32,
        ]
    };
    let mut arguments = pointers
        .into_iter()
        .map(|p| p.to_le_bytes().to_vec())
        .collect::<Vec<_>>();
    if spec.abi == "bundle" {
        arguments.push(scalar.into_iter().flat_map(u32::to_le_bytes).collect());
    } else {
        arguments.extend(scalar.into_iter().map(|word| word.to_le_bytes().to_vec()));
    }
    arguments
}

fn module_identity(
    ctx: &GpuCtx,
    kind: ModuleKind,
) -> Result<(CompilerIdentity, ArtifactIdentity), String> {
    let artifacts = ctx.kernels.artifact_set_identity();
    match kind {
        ModuleKind::Fixed => Ok((ctx.kernels.compiler_identity(), artifacts.fixed)),
        ModuleKind::TriadScalar => Ok((
            ctx.kernels.triad_scalar_compiler_identity(),
            artifacts.triad_scalar,
        )),
        ModuleKind::TriadSm89Half => Ok((
            ctx.kernels
                .triad_sm89_half_compiler_identity()
                .ok_or("half compiler absent")?,
            artifacts.sm89_half.ok_or("half artifact absent")?,
        )),
        ModuleKind::TriadSm89Tf32Joint => Ok((
            ctx.kernels
                .triad_sm89_tf32_joint_compiler_identity()
                .ok_or("joint compiler absent")?,
            artifacts.sm89_tf32_joint.ok_or("joint artifact absent")?,
        )),
        ModuleKind::TriadSm89Finalist => {
            let module = ctx
                .kernels
                .f32_triad_availability()
                .finalist
                .ok_or("finalist compiler/artifact absent")?;
            Ok((module.compiler, module.artifact))
        }
        _ => Err(format!("module {kind:?} is outside combined census")),
    }
}

fn symbol_inventory() -> Result<BTreeMap<String, ExpectedNode>, String> {
    let mut symbols = BTreeMap::new();
    for nvrtc in [(12, 8), (13, 0), (13, 2)] {
        for case in words::inventory()? {
            for node in expected_nodes(&case, nvrtc)? {
                if let Some(previous) = symbols.insert(node.symbol.clone(), node.clone())
                    && (previous.module != node.module
                        || previous.block != node.block
                        || previous.dynamic != node.dynamic
                        || previous.abi != node.abi)
                {
                    return Err(format!("inconsistent census symbol {}", node.symbol));
                }
            }
        }
    }
    if symbols.len() != 35 {
        return Err(format!(
            "closed literal census requires35 symbols, got{}",
            symbols.len()
        ));
    }
    Ok(symbols)
}

struct Anchor {
    graph: CudaGraph,
    fixture: words::Fixture,
}

struct Census {
    anchors: Vec<Anchor>,
    modules: BTreeMap<u8, sys::CUmodule>,
    functions: BTreeMap<String, FunctionReceipt>,
    qualified: BTreeMap<String, QualifiedPhysicalLaunchEvidence>,
    routes: Vec<ResolvedGemmRoute>,
    auto_complete: bool,
}

fn direct_anchor(
    ctx: &GpuCtx,
    function: &CudaFunction,
    fixture: words::Fixture,
    spec: &ExpectedNode,
) -> Result<Anchor, String> {
    let pointers = fixture.pointers();
    let raw = expected_arguments(&fixture, spec);
    let bundle: [u32; 8] = if spec.abi == "bundle" {
        raw[4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "anchor bundle length")?
    } else {
        raw[4..]
            .iter()
            .map(|b| u32::from_le_bytes(b.as_slice().try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "legacy anchor word count")?
    };
    fixture.reset(ctx, 0)?;
    let config = LaunchConfig {
        grid_dim: spec.grid,
        block_dim: spec.block,
        shared_mem_bytes: spec.dynamic,
    };
    // Capturing establishes a real Driver handle without executing the anchor kernel.
    let graph = unsafe {
        capture_into_graph(&ctx.stream, || {
            let mut builder = ctx.stream.launch_builder(function);
            for pointer in &pointers {
                builder.arg(pointer);
            }
            if spec.abi == "bundle" {
                builder.arg(&bundle);
            } else {
                for word in &bundle {
                    builder.arg(word);
                }
            }
            builder
                .launch(config)
                .map(|_| ())
                .map_err(|e| format!("anchor capture: {e:?}"))
        })
    }?;
    Ok(Anchor { graph, fixture })
}

impl Census {
    fn new(runtime: &words::Runtime) -> Result<Self, String> {
        let ctx = &runtime.ctx;
        eprintln!(
            "{}",
            json!({"record":"actual-module-identities","identity":runtime.metadata,"fixed":format!("{:?}",module_identity(ctx,ModuleKind::Fixed)?),"scalar":format!("{:?}",module_identity(ctx,ModuleKind::TriadScalar)?),"half":format!("{:?}",module_identity(ctx,ModuleKind::TriadSm89Half)?),"joint":format!("{:?}",module_identity(ctx,ModuleKind::TriadSm89Tf32Joint)?),"finalist":format!("{:?}",ctx.kernels.f32_triad_availability().finalist),"finalist_loader_rejection":ctx.kernels.finalist_tf32_rejection(),"half_loader_rejection":ctx.kernels.triad_sm89_half_rejection(),"half_exclusions":ctx.kernels.triad_sm89_half_exclusions(),"finalist_live_function_complete":false,"promotion_eligible":false})
        );
        let mut census = Self {
            anchors: Vec::new(),
            modules: BTreeMap::new(),
            functions: BTreeMap::new(),
            qualified: BTreeMap::new(),
            routes: Vec::new(),
            auto_complete: false,
        };
        let mut case = words::inventory()?
            .into_iter()
            .find(|c| c.id == "inference.nn.f32_exact.hot_c.bias1")
            .ok_or("fixed anchor template")?;
        case.dims = (64, 32, 64);
        case.bias = false;
        words::configure(ctx, &case)?;
        let spec = tiled_node(
            "gemm_bi_f32_f32_s2".into(),
            ModuleKind::Fixed,
            (64, 64),
            (64, 64),
            (128, 0, None),
            "legacy",
        );
        let anchor = direct_anchor(
            ctx,
            &ctx.kernels.gemm_bi_f32_f32_s2,
            words::Fixture::new(ctx, &case, false, 0)?,
            &spec,
        )?;
        census.register_anchor(anchor, &spec)?;
        case.family = "triad";
        case.row = "f16";
        case.storage = ["f16"; 3];
        case.dims = (128, 64, 128);
        words::configure(ctx, &case)?;
        let spec = tiled_node(
            "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16".into(),
            ModuleKind::TriadSm89Half,
            (128, 128),
            (128, 128),
            (256, 98_304, Some(0)),
            "bundle",
        );
        let function = ctx
            .kernels
            .triad_sm89_half_function(Sm89HalfRoute::NnM128N128Bk64S3, WeightDtype::F16)
            .ok_or("loaded legacy half anchor missing")?;
        census.register_anchor(
            direct_anchor(
                ctx,
                function,
                words::Fixture::new(ctx, &case, false, 0)?,
                &spec,
            )?,
            &spec,
        )?;
        case.row = "f32_exact";
        case.storage = ["f32"; 3];
        case.op = "tn";
        case.dims = (32, 4, 1);
        words::configure(ctx, &case)?;
        let mut fixture = words::Fixture::new(ctx, &case, false, 0)?;
        fixture.reset(ctx, 0)?;
        let trace = ctx.record_eager_gemm_trace(|| fixture.launch(ctx))?;
        if trace.routes().len() != 1
            || trace.routes()[0].symbol != "gemm_bi_tn_gemv"
            || trace.routes()[0].module_kind != ModuleKind::TriadScalar
        {
            return Err("public scalar legacy anchor selected a different route".into());
        }
        fixture.reset(ctx, 0)?;
        let graph = unsafe { capture_into_graph(&ctx.stream, || fixture.launch(ctx)) }?;
        let spec = tiled_node(
            "gemm_bi_tn_gemv".into(),
            ModuleKind::TriadScalar,
            (4, 1),
            (4, 1),
            (128, 0, None),
            "scalar-anchor",
        );
        census.register_anchor(Anchor { graph, fixture }, &spec)?;
        case.row = "tf32";
        case.op = "nt";
        case.dims = (128, 96, 32);
        words::configure(ctx, &case)?;
        let spec = tiled_node(
            "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1".into(),
            ModuleKind::TriadSm89Tf32Joint,
            (128, 96),
            (128, 96),
            (256, 86_016, Some(0)),
            "bundle",
        );
        let function = ctx
            .kernels
            .triad_sm89_tf32_joint_function(&spec.symbol)
            .ok_or("loaded joint anchor missing")?;
        census.register_anchor(
            direct_anchor(
                ctx,
                function,
                words::Fixture::new(ctx, &case, false, 0)?,
                &spec,
            )?,
            &spec,
        )?;
        census.lookup_symbols(false)?;
        Ok(census)
    }

    fn register_anchor(&mut self, anchor: Anchor, spec: &ExpectedNode) -> Result<(), String> {
        let graph = read_driver_graph(&anchor.graph)?;
        if graph.count != 1 || graph.non_kernel_nodes != 0 {
            return Err("anchor must be exactly one kernel".into());
        }
        let node = &graph.kernels[0];
        let receipt = function_receipt(node.params.func, spec)?;
        if (
            node.params.gridDimX,
            node.params.gridDimY,
            node.params.gridDimZ,
        ) != spec.grid
            || (
                node.params.blockDimX,
                node.params.blockDimY,
                node.params.blockDimZ,
            ) != spec.block
            || node.params.sharedMemBytes != spec.dynamic
        {
            return Err(format!("{} anchor launch config mismatch", spec.symbol));
        }
        let actual = argument_bytes(node, &receipt.abi)?;
        let expected = if spec.abi == "scalar-anchor" {
            let [c, a, b, _] = anchor.fixture.pointers();
            vec![
                c.to_le_bytes().to_vec(),
                a.to_le_bytes().to_vec(),
                b.to_le_bytes().to_vec(),
                0x3f800000_u32.to_le_bytes().to_vec(),
                32_u32.to_le_bytes().to_vec(),
                4_u32.to_le_bytes().to_vec(),
                4_u32.to_le_bytes().to_vec(),
                1_u32.to_le_bytes().to_vec(),
            ]
        } else {
            expected_arguments(&anchor.fixture, spec)
        };
        if actual != expected {
            return Err(format!("{} anchor parameter mismatch", spec.symbol));
        }
        if let Some(previous) = self.modules.insert(spec.module as u8, receipt.module)
            && previous != receipt.module
        {
            return Err("one module kind resolved to multiple owners".into());
        }
        eprintln!(
            "{}",
            json!({"record":"borrowed-module-anchor","receipt":receipt.json,"replayed":false})
        );
        self.anchors.push(anchor);
        Ok(())
    }

    fn lookup_symbols(&mut self, include_finalist: bool) -> Result<(), String> {
        for (_, spec) in symbol_inventory()? {
            if !include_finalist && spec.module == ModuleKind::TriadSm89Finalist {
                continue;
            }
            let module = *self
                .modules
                .get(&(spec.module as u8))
                .ok_or_else(|| format!("missing borrowed {:?} module", spec.module))?;
            let symbol = std::ffi::CString::new(spec.symbol.as_str()).map_err(|e| e.to_string())?;
            let mut function = std::ptr::null_mut();
            driver(
                unsafe { sys::cuModuleGetFunction(&mut function, module, symbol.as_ptr()) },
                &format!("literal lookup {}", spec.symbol),
            )?;
            let receipt = function_receipt(function, &spec)?;
            if receipt.module != module {
                return Err(format!("{} belongs to another module", spec.symbol));
            }
            eprintln!(
                "{}",
                json!({"record":"literal-module-function","receipt":receipt.json})
            );
            self.functions.insert(spec.symbol, receipt);
        }
        Ok(())
    }

    fn complete_finalist(&mut self, runtime: &words::Runtime) -> Result<(), String> {
        let case = words::inventory()?
            .into_iter()
            .find(|case| case.id == "triad.nt.tf32.prism.bias0")
            .ok_or("missing finalist Prism cell")?;
        words::configure(&runtime.ctx, &case)?;
        let spec =
            expected_nodes(&case, runtime.ctx.kernels.compiler_identity().nvrtc_version)?.remove(0);
        let mut fixture = words::Fixture::new(&runtime.ctx, &case, false, 0)?;
        fixture.reset(&runtime.ctx, 0)?;
        let trace = runtime
            .ctx
            .record_eager_gemm_trace(|| fixture.launch(&runtime.ctx))?;
        if trace.routes().len() != 1
            || trace.routes()[0].symbol != spec.symbol
            || trace.routes()[0].module_kind != ModuleKind::TriadSm89Finalist
        {
            return Err("provisional measured identities must admit literal finalist Prism AUTO before completing census".into());
        }
        let expected_identity = module_identity(&runtime.ctx, ModuleKind::TriadSm89Finalist)?;
        if (trace.routes()[0].compiler, trace.routes()[0].artifact) != expected_identity {
            return Err("finalist AUTO disagrees with recorded loader identity".into());
        }
        fixture.reset(&runtime.ctx, 0)?;
        let graph =
            unsafe { capture_into_graph(&runtime.ctx.stream, || fixture.launch(&runtime.ctx)) }?;
        self.register_anchor(Anchor { graph, fixture }, &spec)?;
        self.lookup_symbols(true)?;
        if self.functions.len() != 35 {
            return Err("live literal census incomplete".into());
        }
        Ok(())
    }

    fn complete_auto_inventory(&mut self, runtime: &words::Runtime) -> Result<(), String> {
        self.complete_finalist(runtime)?;
        let mut count = 0;
        for case in words::inventory()? {
            words::configure(&runtime.ctx, &case)?;
            let mut fixture = words::Fixture::new(&runtime.ctx, &case, false, 0)?;
            words::Observer::prepare(self, runtime, &fixture)?;
            fixture.reset(&runtime.ctx, 0)?;
            words::Observer::eager(self, &runtime.ctx, &mut fixture)?;
            fixture.verify(&runtime.ctx)?;
            fixture.reset(&runtime.ctx, 1)?;
            let graph = unsafe {
                capture_into_graph(&runtime.ctx.stream, || fixture.launch(&runtime.ctx))
            }?;
            words::Observer::graph(self, runtime, &fixture, &graph)?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|e| format!("AUTO census sync: {e:?}"))?;
            count += 1;
        }
        if count != 99 {
            return Err("AUTO physical census incomplete".into());
        }
        negative_controls(runtime)?;
        self.auto_complete = true;
        eprintln!(
            "{}",
            json!({"record":"auto-census-complete","device_scope":runtime.device_scope(),"cases":count,"literal_functions":self.functions.len(),"live_finalist_function_complete":true,"raw_word_acceptance_complete":false,"promotion_eligible":false,"performance_timing_certified":false})
        );
        Ok(())
    }
}

fn verify_inference_holder(
    ctx: &GpuCtx,
    fixture: &words::Fixture,
    routes: &[ResolvedGemmRoute],
) -> Result<Option<FixedAutoPhysicalDescriptor>, String> {
    if fixture.case.family != "inference" {
        return Ok(None);
    }
    let [c, a, b, bias] = fixture.pointers();
    let (m, k, n) = fixture.case.dims;
    let operands = InferenceFwdOperands {
        c: TypedPtr {
            ptr: c,
            dtype: words::dtype(fixture.case.storage[2])?,
        },
        x: TypedPtr {
            ptr: a,
            dtype: words::dtype(fixture.case.storage[0])?,
        },
        w: TypedPtr {
            ptr: b,
            dtype: words::dtype(fixture.case.storage[1])?,
        },
        bias_ptr: if bias == 0 { None } else { Some(bias) },
    };
    let compiler = ctx.kernels.compiler_identity();
    let family = if fixture.case.row == "f32_exact" {
        InferenceTile::F32Sm89N64CopyPlan
    } else {
        InferenceTile::Tc128Sm89S3
    };
    fixed_auto_bundle_physical_descriptor(
        FixedAutoPhysicalRequest {
            row: fixture.case.row,
            operands,
            shape: InferenceShape { m, k, n },
            selected: family,
            compute_capability: ctx.gemm_route().device.compute_capability,
            multiprocessors: ctx.gemm_route().device.multiprocessor_count,
            compiler_target: compiler.target.as_str(),
            state_capacity: ctx.state_cap(),
            nvrtc: compiler.nvrtc_version,
            nvrtc_library_known: compiler.nvrtc_library_known,
            policy: ctx.f32_triad_policy(),
        },
        &routes
            .iter()
            .map(FixedAutoRecordedRoute::from)
            .collect::<Vec<_>>(),
    )
}

impl words::Observer for Census {
    fn prepare(
        &mut self,
        runtime: &words::Runtime,
        fixture: &words::Fixture,
    ) -> Result<(), String> {
        let case = &fixture.case;
        if case.family != "triad" || self.qualified.contains_key(&case.id) {
            return Ok(());
        }
        let route = if case.storage[0] == "f32" {
            PhysicalQualificationRoute::F32Policy(runtime.ctx.f32_triad_policy())
        } else {
            PhysicalQualificationRoute::HalfPolicy {
                dtype: words::dtype(case.storage[0])?,
                tensor_cores: true,
                half_policy: HalfTriadPolicy::TiledParityV1,
            }
        };
        let request = if case.storage[0] == "f32" {
            PhysicalQualificationRequest::contiguous_f32(
                logical_op(case.op)?,
                case.dims,
                route,
                PhysicalQualificationF32Epilogue::new(
                    1.0,
                    if case.op == "tn" { 1.0 } else { 0.0 },
                    case.bias,
                ),
            )
        } else {
            PhysicalQualificationRequest::contiguous(logical_op(case.op)?, case.dims, route)
        };
        let launch = qualify_physical_launch(&runtime.ctx, request)?;
        launch.validate_timed_request(&runtime.ctx, request)?;
        let evidence = launch.evidence().clone();
        if *evidence.route_identity() != runtime.ctx.gemm_route() {
            return Err("qualification prepared identity differs from live context".into());
        }
        let expected = expected_nodes(case, runtime.ctx.kernels.compiler_identity().nvrtc_version)?;
        if evidence.nodes().len() != expected.len()
            || evidence
                .nodes()
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| {
                    actual.symbol != expected.symbol
                        || actual.module_kind != expected.module
                        || actual.launch.grid_dim != expected.grid
                        || actual.launch.block_dim != expected.block
                        || actual.launch.shared_mem_bytes != expected.dynamic
                })
        {
            return Err(format!(
                "{} qualification differs from literal AUTO manifest",
                case.id
            ));
        }
        eprintln!(
            "{}",
            json!({"record":"production-prepared-agreement","case":case.id,"evidence":format!("{evidence:?}"),"half_bias_note":if case.storage[0]!="f32"&&case.bias {"facade identity uses unbiased sibling; public raw graph validates actual bias"}else{"exact epilogue"}})
        );
        drop(launch);
        if *evidence.route_identity() != runtime.ctx.gemm_route() {
            return Err(
                "qualification policy restoration changed shared public-call context".into(),
            );
        }
        self.qualified.insert(case.id.clone(), evidence);
        Ok(())
    }

    fn eager(&mut self, ctx: &GpuCtx, fixture: &mut words::Fixture) -> Result<(), String> {
        self.routes = ctx
            .record_eager_gemm_trace(|| fixture.launch(ctx))?
            .routes()
            .to_vec();
        let expected =
            expected_nodes(&fixture.case, ctx.kernels.compiler_identity().nvrtc_version)?;
        if self.routes.len() != expected.len() {
            return Err(format!(
                "{} eager AUTO launch count {} != {}",
                fixture.case.id,
                self.routes.len(),
                expected.len()
            ));
        }
        for (route, spec) in self.routes.iter().zip(&expected) {
            let identity = module_identity(ctx, spec.module)?;
            if route.symbol != spec.symbol
                || route.module_kind != spec.module
                || (route.compiler, route.artifact) != identity
                || route.target != identity.0.target
                || route.device != ctx.gemm_route().device
                || route.shape != fixture.case.dims
                || route.strides != fixture.case.strides()
                || route.tile != spec.tile
                || route.bk != spec.bk
                || route.stages != spec.stages
                || route.threads != spec.block.0 * spec.block.1 * spec.block.2
                || route.launch.grid_dim != spec.grid
                || route.launch.block_dim != spec.block
                || route.launch.shared_mem_bytes != spec.dynamic
            {
                return Err(format!(
                    "{} eager route differs from literal manifest {spec:?}: {route:?}",
                    fixture.case.id
                ));
            }
        }
        verify_inference_holder(ctx, fixture, &self.routes)?;
        if let Some(evidence) = self.qualified.get(&fixture.case.id) {
            if *evidence.route_identity() != ctx.gemm_route() {
                return Err("public eager and qualification prepared identities differ".into());
            }
            for (route, node) in self.routes.iter().zip(evidence.nodes()) {
                if route.symbol != node.symbol
                    || route.module_kind != node.module_kind
                    || route.op != node.logical_op
                    || route.dtype != node.logical_dtype
                    || route.shape != node.shape
                    || route.strides != node.strides
                    || node.tile != Some(route.tile)
                    || node
                        .numeric_contract
                        .is_some_and(|contract| contract != route.numeric_contract)
                {
                    return Err(
                        "public eager route disagrees with production-sealed qualification".into(),
                    );
                }
            }
        }
        eprintln!(
            "{}",
            json!({"record":"public-eager-trace","case":fixture.case.id,"routes":format!("{:?}",self.routes)})
        );
        Ok(())
    }

    fn graph(
        &mut self,
        runtime: &words::Runtime,
        fixture: &words::Fixture,
        graph: &CudaGraph,
    ) -> Result<(), String> {
        let observed = read_driver_graph(graph)?;
        let order = topological_order(observed.count, &observed.edges)?;
        let expected = expected_nodes(
            &fixture.case,
            runtime.ctx.kernels.compiler_identity().nvrtc_version,
        )?;
        if observed.non_kernel_nodes != 0
            || observed.count != expected.len()
            || self.routes.len() != expected.len()
        {
            return Err("physical graph node inventory differs from eager/literal manifest".into());
        }
        for ((index, spec), trace) in order.iter().zip(&expected).zip(&self.routes) {
            let node = observed
                .kernels
                .iter()
                .find(|node| node.index == *index)
                .ok_or("missing ordered graph node")?;
            let census = self.functions.get(&spec.symbol).ok_or_else(|| {
                format!("{} has not completed literal module census", spec.symbol)
            })?;
            let receipt = function_receipt(node.params.func, spec)?;
            if node.params.func != census.function
                || receipt.module != census.module
                || receipt.abi != census.abi
                || receipt.json["resources"] != census.json["resources"]
                || node.symbol != spec.symbol
                || (
                    node.params.gridDimX,
                    node.params.gridDimY,
                    node.params.gridDimZ,
                ) != spec.grid
                || (
                    node.params.blockDimX,
                    node.params.blockDimY,
                    node.params.blockDimZ,
                ) != spec.block
                || node.params.sharedMemBytes != spec.dynamic
            {
                return Err(format!(
                    "{} graph does not equal same-context censused function/config",
                    fixture.case.id
                ));
            }
            let actual = argument_bytes(node, &receipt.abi)?;
            let required = expected_arguments(fixture, spec);
            if actual != required {
                return Err(format!(
                    "{} graph argument mismatch for {}: actual={actual:02x?} expected={required:02x?}",
                    fixture.case.id, spec.symbol
                ));
            }
            if trace.symbol != node.symbol {
                return Err("eager/graph actual symbol mismatch".into());
            }
            if let Some(descriptor) = verify_inference_holder(&runtime.ctx, fixture, &self.routes)?
            {
                let pointers = fixture.pointers();
                let bundle: [u32; 8] = actual[4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| "retained holder bundle length")?;
                let (m, k, n) = fixture.case.dims;
                fixed_auto_bundle_graph_contract(
                    descriptor,
                    &FixedAutoGraphObservation {
                        node_count: observed.count,
                        symbol: &node.symbol,
                        grid: spec.grid,
                        block: spec.block,
                        dynamic_shared_bytes: spec.dynamic,
                        static_shared_bytes: receipt.json["resources"]["static_shared_bytes"]
                            .as_u64()
                            .ok_or("missing static shared receipt")?
                            as u32,
                        driver_abi: receipt.abi.clone(),
                        terminal_sixth_rejected: true,
                        pointers,
                        bundle,
                    },
                    pointers,
                    InferenceShape { m, k, n },
                )?;
            }
        }
        eprintln!(
            "{}",
            json!({"record":"public-graph-physical-pass","case":fixture.case.id,"dependency_edges":observed.edges,"topological_order":order,"function_identity":"same-context-literal-census","launch_parameters":"every-byte-exact"})
        );
        Ok(())
    }
}

fn identity_receipt(runtime: &words::Runtime, census: &Census) -> Result<Value, String> {
    let mut modules = BTreeMap::new();
    for kind in [
        ModuleKind::Fixed,
        ModuleKind::TriadScalar,
        ModuleKind::TriadSm89Half,
        ModuleKind::TriadSm89Tf32Joint,
        ModuleKind::TriadSm89Finalist,
    ] {
        modules.insert(
            format!("{kind:?}"),
            format!("{:?}", module_identity(&runtime.ctx, kind)?),
        );
    }
    Ok(
        json!({"schema":"combined-gemm-census.v1","device_scope":runtime.device_scope(),"identity":runtime.metadata,"modules":modules,"scalar_fixed_pair":[modules["TriadScalar"],modules["Fixed"]],"functions":census.functions.iter().map(|(symbol,function)|(symbol,&function.json)).collect::<BTreeMap<_,_>>(),"anchors":census.anchors.iter().map(|anchor|json!({"graph":anchor.graph.cu_graph() as usize,"managed_pointers":anchor.fixture.pointers(),"replayed":false})).collect::<Vec<_>>(),"auto_cases_complete":if census.auto_complete {99}else{0},"finalist_live_function_complete":census.auto_complete,"promotion_eligible":false,"performance_timing_certified":false}),
    )
}

fn require_staged_identity(runtime: &words::Runtime, census: &Census) -> Result<(), String> {
    let path = std::path::PathBuf::from(words::required_env("COMBINED_GEMM_IDENTITY_RECEIPT")?);
    let (metadata, payload) = words::read_reference(&path)?;
    if !payload.is_empty() {
        return Err("identity census receipt must have no numeric payload".into());
    }
    let previous: Value =
        serde_json::from_str(&metadata).map_err(|e| format!("identity receipt: {e}"))?;
    let current = identity_receipt(runtime, census)?;
    if previous["schema"] != "combined-gemm-census.v1"
        || previous["modules"] != current["modules"]
        || previous["scalar_fixed_pair"] != current["scalar_fixed_pair"]
        || previous["identity"]["cohort"] != current["identity"]["cohort"]
        || previous["device_scope"] != current["device_scope"]
        || previous["identity"]["device_scope"] != current["identity"]["device_scope"]
        || previous["finalist_live_function_complete"] != false
    {
        return Err("staged build changed measured compiler/artifact/cohort identity, or initial receipt is not pre-finalist".into());
    }
    Ok(())
}

fn save_census(runtime: &words::Runtime, census: &Census) -> Result<String, String> {
    let path = std::path::PathBuf::from(words::required_env("COMBINED_GEMM_CENSUS_RECEIPT")?);
    words::write_reference(&path, &identity_receipt(runtime, census)?.to_string(), &[])?;
    Ok(words::sha(&std::fs::read(path).map_err(|e| e.to_string())?))
}

#[test]
#[ignore = "requires explicit identities/auto phase, scope, cap, receipt paths and a 142-SM Ada GPU"]
fn combined_gemm_module_census() -> Result<(), String> {
    let phase = words::required_env("COMBINED_GEMM_PHASE")?;
    if phase == "verify-six" {
        return verify_six_cohorts();
    }
    if !matches!(phase.as_str(), "identities" | "auto") {
        return Err("COMBINED_GEMM_PHASE must be identities, auto or verify-six".into());
    }
    let mut runtime = words::Runtime::new(false)?;
    runtime.metadata["current_acceptance_source_sha256"] =
        json!(words::sha(CURRENT_ACCEPTANCE_SOURCE));
    let mut census = Census::new(&runtime)?;
    if phase == "auto" {
        require_staged_identity(&runtime, &census)?;
        census.complete_auto_inventory(&runtime)?;
    }
    let finish_observation = runtime.finish()?;
    let hash = save_census(&runtime, &census)?;
    eprintln!(
        "{}",
        json!({"record":"census-phase-receipt","device_scope":runtime.device_scope(),"finish_device_observation":finish_observation,"phase":phase,"receipt_sha256":hash,"finalist_live_function_complete":census.auto_complete,"promotion_eligible":false,"performance_timing_certified":false})
    );
    Ok(())
}

#[test]
#[ignore = "requires measured provisional identities, released reference files, explicit scope/cap and Ada GPU"]
fn combined_gemm_released_auto_bits() -> Result<(), String> {
    let mut runtime = words::Runtime::new(false)?;
    runtime.metadata["current_acceptance_source_sha256"] =
        json!(words::sha(CURRENT_ACCEPTANCE_SOURCE));
    let mut census = Census::new(&runtime)?;
    require_staged_identity(&runtime, &census)?;
    census.complete_auto_inventory(&runtime)?;
    runtime.metadata["physical_census"] = identity_receipt(&runtime, &census)?;
    runtime.metadata["physical_census_receipt_sha256"] =
        Value::String(save_census(&runtime, &census)?);
    words::run_words(&runtime, false, &mut census)
}

fn negative_controls(runtime: &words::Runtime) -> Result<(), String> {
    let ctx = &runtime.ctx;
    for positive in words::inventory()?
        .into_iter()
        .filter(|case| case.required && !case.bias)
    {
        let required = expected_nodes(&positive, ctx.kernels.compiler_identity().nvrtc_version)?;
        let retained = required
            .last()
            .ok_or("negative missing retained symbol")?
            .symbol
            .as_str();
        let mut neighbor = positive.clone();
        neighbor.dims.0 -= 1;
        words::configure(ctx, &neighbor)?;
        let mut fixture = words::Fixture::new(ctx, &neighbor, false, 0)?;
        fixture.reset(ctx, 0)?;
        let trace = ctx.record_eager_gemm_trace(|| fixture.launch(ctx))?;
        fixture.verify(ctx)?;
        if trace.routes().is_empty() || trace.routes().iter().any(|route| route.symbol == retained)
        {
            return Err(format!(
                "{} neighboring public shape did not leave retained holder {retained}",
                positive.id
            ));
        }
        eprintln!(
            "{}",
            json!({"record":"public-shape-negative","positive_case":positive.id,"negative_dimensions":neighbor.dims,"actual_fallback_routes":format!("{:?}",trace.routes()),"raw_reference_case":false})
        );
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let route = PhysicalQualificationRoute::HalfPolicy {
            dtype,
            tensor_cores: true,
            half_policy: HalfTriadPolicy::TiledParityV1,
        };
        for offset in [
            mamba_rs::mamba_ssm::gpu::gemm_bi_triad::PhysicalQualificationOffset::A,
            mamba_rs::mamba_ssm::gpu::gemm_bi_triad::PhysicalQualificationOffset::B,
            mamba_rs::mamba_ssm::gpu::gemm_bi_triad::PhysicalQualificationOffset::Output,
        ] {
            let request = PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Tn,
                (1024, 256, 128),
                route,
                offset,
            );
            let launch = qualify_physical_launch(ctx, request)?;
            if launch.evidence().nodes().iter().any(|node| {
                node.symbol
                    .starts_with("gemm_bi_tn_sm89_m16n16_bk64_s2_ldb72_v1_")
            }) {
                return Err("misaligned half view entered small16 AUTO".into());
            }
            eprintln!(
                "{}",
                json!({"record":"public-offset-negative","dtype":format!("{dtype:?}"),"offset":format!("{offset:?}"),"actual_fallback":format!("{:?}",launch.evidence()),"raw_reference_case":false})
            );
        }
    }
    let request = PhysicalQualificationRequest::contiguous_f32(
        ResolvedGemmOp::Nn,
        (4096, 3072, 1536),
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        PhysicalQualificationF32Epilogue::new(1.0, 0.0, true),
    );
    let launch = qualify_physical_launch(ctx, request)?;
    if launch
        .evidence()
        .nodes()
        .iter()
        .any(|node| node.symbol == "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1")
    {
        return Err("biased NN request entered no-bias Fixed copyplan AUTO".into());
    }
    eprintln!(
        "{}",
        json!({"record":"public-epilogue-negative","actual_fallback":format!("{:?}",launch.evidence()),"raw_reference_case":false})
    );
    Ok(())
}

fn completion_device_scope(receipt: &Value) -> Result<words::DeviceScope, String> {
    let scope = words::DeviceScope::parse(
        receipt["device_scope"]
            .as_str()
            .ok_or("completion receipt is missing a valid device scope")?,
    )?;
    let identity_scope = words::DeviceScope::parse(
        receipt["identity"]["device_scope"]
            .as_str()
            .ok_or("completion identity is missing a valid device scope")?,
    )?;
    if identity_scope != scope {
        return Err("completion and identity device scopes differ".into());
    }
    if !receipt["identity"]["physical_census"].is_null() {
        let census_scope = words::DeviceScope::parse(
            receipt["identity"]["physical_census"]["device_scope"]
                .as_str()
                .ok_or("physical census is missing a valid device scope")?,
        )?;
        if census_scope != scope {
            return Err("completion and physical-census device scopes differ".into());
        }
    }
    Ok(scope)
}

fn validate_completion_scope_pair(released: &Value, current: &Value) -> Result<(), String> {
    if completion_device_scope(released)? != completion_device_scope(current)? {
        return Err("released/current completion device scopes differ".into());
    }
    Ok(())
}

fn validate_six_current(receipts: &[Value]) -> Result<(), String> {
    if receipts.len() != 6 {
        return Err("six complete current cohorts are required".into());
    }
    let mut cohorts = std::collections::BTreeSet::new();
    let mut fixed = std::collections::BTreeSet::new();
    let mut invariant: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut common: BTreeMap<&str, String> = BTreeMap::new();
    let mut device_scope = None;
    let symbols = symbol_inventory()?;
    for receipt in receipts {
        let receipt_scope = completion_device_scope(receipt)?;
        if device_scope
            .replace(receipt_scope)
            .is_some_and(|scope| scope != receipt_scope)
        {
            return Err("current six-cohort completion device scopes differ".into());
        }
        let identity = &receipt["identity"];
        let toolkit = identity["toolkit"].as_str().ok_or("missing toolkit")?;
        let cap = identity["state_capacity"]
            .as_u64()
            .ok_or("missing state capacity")?
            .to_string();
        let cohort = words::explicit_cohort(toolkit, Some(&cap))?;
        if identity["cohort"] != cohort
            || !cohorts.insert(cohort)
            || identity["released"] != false
            || receipt["reference_records"] != 198
            || receipt["case_cohort_keys"] != 99
            || receipt["independent_runs_per_case_corpus"] != 6
        {
            return Err("invalid/duplicate/incomplete current cohort".into());
        }
        let census = &identity["physical_census"];
        if census["auto_cases_complete"] != 99
            || census["finalist_live_function_complete"] != true
            || census["functions"].as_object().map(|m| m.len()) != Some(35)
        {
            return Err("current cohort lacks complete literal AUTO physical census".into());
        }
        if symbols
            .keys()
            .any(|symbol| census["functions"].get(symbol).is_none())
        {
            return Err("current cohort is missing a literal censused symbol".into());
        }
        for field in [
            "case_schema_sha256",
            "shared_payload_sha256",
            "harness_source_sha256",
            "production_source_snapshot_sha256",
            "current_acceptance_source_sha256",
        ] {
            let value = identity[field]
                .as_str()
                .filter(|value| value.len() == 64 && value.bytes().all(|c| c.is_ascii_hexdigit()))
                .ok_or_else(|| format!("missing/invalid {field}"))?
                .to_owned();
            if let Some(previous) = common.insert(field, value.clone())
                && previous != value
            {
                return Err(format!("current six-cohort source/schema drift: {field}"));
            }
        }
        let modules = &census["modules"];
        let fixed_identity = modules["Fixed"]
            .as_str()
            .ok_or("missing Fixed identity")?
            .to_owned();
        if !fixed.insert(fixed_identity.clone()) {
            return Err("Fixed identities must distinguish all six toolkit/cap cohorts".into());
        }
        if census["scalar_fixed_pair"] != json!([modules["TriadScalar"], fixed_identity]) {
            return Err("missing explicit same-toolkit scalar+Fixed pair".into());
        }
        for module in [
            "TriadScalar",
            "TriadSm89Half",
            "TriadSm89Finalist",
            "TriadSm89Tf32Joint",
        ] {
            let value = modules[module]
                .as_str()
                .ok_or_else(|| format!("missing {module} identity"))?
                .to_owned();
            let key = (toolkit.to_owned(), module.to_owned());
            if let Some(previous) = invariant.insert(key, value.clone())
                && previous != value
            {
                return Err(format!(
                    "{toolkit} {module} compiler/artifact changed across capacities"
                ));
            }
        }
    }
    if cohorts.len() != 6 || fixed.len() != 6 || invariant.len() != 12 || device_scope.is_none() {
        return Err("cohort module closure incomplete".into());
    }
    for module in ["TriadScalar", "TriadSm89Half", "TriadSm89Finalist"] {
        if invariant
            .iter()
            .filter(|((_, kind), _)| kind == module)
            .map(|(_, value)| value)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != 3
        {
            return Err(format!(
                "requires exactly three toolkit identities for {module}"
            ));
        }
    }
    Ok(())
}

fn verify_six_cohorts() -> Result<(), String> {
    let directory = std::path::PathBuf::from(words::required_env("COMBINED_GEMM_REFERENCE_DIR")?);
    let mut current = Vec::new();
    for toolkit in ["12.8", "13.0", "13.2"] {
        for cap in ["16", "64"] {
            let cohort = words::explicit_cohort(toolkit, Some(cap))?;
            let mut pair = Vec::new();
            for version in ["released", "current"] {
                let (metadata, payload) = words::read_reference(
                    &directory.join(format!("{cohort}.{version}-complete.receipt")),
                )?;
                if !payload.is_empty() {
                    return Err("completion receipt has unexpected numeric payload".into());
                }
                let receipt: Value = serde_json::from_str(&metadata).map_err(|e| e.to_string())?;
                if receipt["identity"]["cohort"] != cohort
                    || receipt["reference_records"] != 198
                    || receipt["case_cohort_keys"] != 99
                    || receipt["independent_runs_per_case_corpus"] != 6
                {
                    return Err(format!("incomplete {cohort}/{version}"));
                }
                if version == "released"
                    && (receipt["identity"]["base_commit"] != words::RELEASED_SHA
                        || receipt["identity"]["released"] != true)
                {
                    return Err("reference completion is not exact released v0.7.0".into());
                }
                let files = receipt["reference_files"]
                    .as_object()
                    .ok_or("missing file manifest")?;
                if files.len() != 198 {
                    return Err("reference-file inventory count mismatch".into());
                }
                for case in words::inventory()? {
                    for corpus in ["finite", "exceptional"] {
                        let name = format!("{cohort}.{}.{corpus}.words", case.id);
                        let path = directory.join(&name);
                        let expected = files
                            .get(&name)
                            .and_then(Value::as_str)
                            .ok_or("missing reference key")?;
                        if words::sha(&std::fs::read(&path).map_err(|e| e.to_string())?) != expected
                        {
                            return Err(format!("reference hash changed {name}"));
                        }
                        words::read_reference(&path)?;
                    }
                }
                pair.push(receipt);
            }
            if pair[0]["reference_files"] != pair[1]["reference_files"]
                || pair[0]["identity"]["case_schema_sha256"]
                    != pair[1]["identity"]["case_schema_sha256"]
                || pair[0]["identity"]["shared_payload_sha256"]
                    != pair[1]["identity"]["shared_payload_sha256"]
            {
                return Err("released/current completion provenance differs".into());
            }
            validate_completion_scope_pair(&pair[0], &pair[1])?;
            current.push(pair.remove(1));
        }
    }
    validate_six_current(&current)?;
    let device_scope = completion_device_scope(&current[0])?.as_str();
    let receipt = json!({"schema":"combined-gemm-six-cohort.v1","device_scope":device_scope,"case_cohort_keys_per_version":594,"logical_cases":99,"toolkits":["12.8","13.0","13.2"],"state_capacities":[16,64],"literal_auto_and_raw_words_complete":true,"admission_arrays_modified":false,"performance_timing_certified":false,"receipts":current});
    let path = std::path::PathBuf::from(words::required_env("COMBINED_GEMM_CENSUS_RECEIPT")?);
    words::write_reference(&path, &receipt.to_string(), &[])?;
    eprintln!(
        "six-cohort acceptance complete:594 case/cohort keys per version; receipt {}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod combined_gemm_host {
    use super::*;

    #[test]
    fn literal_symbols_keep_joint_precedence_and_legacy_control() {
        let cases = words::inventory().unwrap();
        let symbols = symbol_inventory().unwrap();
        assert_eq!(symbols.len(), 35);
        assert_eq!(
            symbols
                .values()
                .filter(|node| node.module == ModuleKind::Fixed)
                .count(),
            19
        );
        assert_eq!(
            symbols
                .values()
                .filter(|node| node.module == ModuleKind::TriadSm89Half)
                .count(),
            12
        );
        for toolkit in [(12, 8), (13, 0), (13, 2)] {
            let exact = cases
                .iter()
                .find(|case| case.id == "inference.nn.f32_exact.hot_c.bias1")
                .unwrap();
            assert_eq!(
                expected_nodes(exact, toolkit).unwrap()[0].symbol,
                "gemm_bi_f32_f32_s2"
            );
            let finalist = cases
                .iter()
                .filter(|case| case.family == "triad" && case.row == "tf32")
                .map(|case| expected_nodes(case, toolkit).unwrap()[0].module)
                .collect::<Vec<_>>();
            assert_eq!(
                finalist
                    .iter()
                    .filter(|&&module| module == ModuleKind::TriadSm89Finalist)
                    .count(),
                1
            );
            assert_eq!(
                finalist
                    .iter()
                    .filter(|&&module| module == ModuleKind::TriadSm89Tf32Joint)
                    .count(),
                3
            );
            for id in [
                "inference.nn.bf16_f32.hot_b.bias0",
                "inference.nn.f16_f32.hot_c.bias1",
            ] {
                let case = cases.iter().find(|case| case.id == id).unwrap();
                assert!(
                    expected_nodes(case, toolkit).unwrap()[0]
                        .symbol
                        .starts_with("gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_")
                );
            }
        }
    }

    #[test]
    fn parameter_contract_preserves_nt_scratch_handoff_and_distinct_nn_orders() {
        let cases = words::inventory().unwrap();
        let case = cases
            .iter()
            .find(|case| case.id == "triad.nt.f32_exact.large_deep.bias0")
            .unwrap();
        let nodes = expected_nodes(case, (12, 8)).unwrap();
        let first = argument_contract(case, [0x1000, 0x2000, 0x3000, 0], 0x4000, &nodes[0]);
        assert_eq!(
            first,
            [
                0x4000_u64.to_le_bytes().to_vec(),
                0x3000_u64.to_le_bytes().to_vec(),
                3072_u32.to_le_bytes().to_vec(),
                1536_u32.to_le_bytes().to_vec()
            ]
        );
        let second = argument_contract(case, [0x1000, 0x2000, 0x3000, 0], 0x4000, &nodes[1]);
        assert_eq!(second[2], 0x4000_u64.to_le_bytes());
        assert_eq!(
            second[4],
            [0x3f800000_u32, 0, 4096, 3072, 1536, 1536, 3072, 3072]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        for (id, expected) in [
            (
                "inference.nn.tf32.hot_b.bias0",
                [0x3f800000_u32, 0, 4621, 768, 2304, 768, 2304, 2304],
            ),
            (
                "inference.nn.bf16_f32.hot_b.bias0",
                [0x3f800000_u32, 0, 4621, 2304, 768, 768, 2304, 2304],
            ),
        ] {
            let case = cases.iter().find(|case| case.id == id).unwrap();
            let spec = expected_nodes(case, (12, 8)).unwrap();
            assert_eq!(
                argument_contract(case, [0x1000, 0x2000, 0x3000, 0], 0x4000, &spec[0])[4],
                expected
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
        }
    }

    fn six_current_receipt_fixture() -> Vec<Value> {
        let literal_symbols = [
            "gemm_bi_f32_f32_s2",
            "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_bf16",
            "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_f16",
            "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16",
            "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_f16",
            "gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16",
            "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16",
            "gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16",
            "gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16",
            "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1",
            "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3",
            "gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3",
            "gemm_bi_nn_tc64_f32out_bf16",
            "gemm_bi_nn_tc64_f32out_f16",
            "gemm_bi_nn_tc128_f32out_bf16",
            "gemm_bi_nn_tc128_f32out_f16",
            "gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_bf16",
            "gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_f16",
            "gemm_bi_nn_inference_sm89_f32_m128n64_tail_copyplan_v1",
            "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_bf16",
            "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16",
            "gemm_bi_tn_sm89_m64n64_bk64_s2_compact_bxor_v1_bf16",
            "gemm_bi_tn_sm89_m64n64_bk64_s2_compact_bxor_v1_f16",
            "gemm_bi_tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_bf16",
            "gemm_bi_tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_f16",
            "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_bf16",
            "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_f16",
            "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_bf16",
            "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_f16",
            "gemm_bi_tn_sm89_m16n16_bk64_s2_ldb72_v1_bf16",
            "gemm_bi_tn_sm89_m16n16_bk64_s2_ldb72_v1_f16",
            "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1",
            "gemm_bi_transpose_f32_32x16_d768_v1",
            "gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2",
            "gemm_bi_nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1",
        ];
        let mut receipts = Vec::new();
        for (toolkit, cap) in [
            ("12.8", 16),
            ("12.8", 64),
            ("13.0", 16),
            ("13.0", 64),
            ("13.2", 16),
            ("13.2", 64),
        ] {
            let fixed = format!("fixed-{toolkit}-{cap}");
            let scalar = format!("scalar-{toolkit}");
            let functions = literal_symbols
                .iter()
                .map(|symbol| (*symbol, json!({})))
                .collect::<BTreeMap<_, _>>();
            receipts.push(json!({"device_scope":"shared-functional","reference_records":198,"case_cohort_keys":99,"independent_runs_per_case_corpus":6,"identity":{"device_scope":"shared-functional","toolkit":toolkit,"state_capacity":cap,"cohort":format!("cuda{toolkit}-cap{cap}"),"released":false,"case_schema_sha256":"a".repeat(64),"shared_payload_sha256":"b".repeat(64),"harness_source_sha256":"c".repeat(64),"production_source_snapshot_sha256":"d".repeat(64),"physical_census":{"device_scope":"shared-functional","auto_cases_complete":99,"finalist_live_function_complete":true,"functions":functions,"modules":{"Fixed":fixed,"TriadScalar":scalar,"TriadSm89Half":format!("half-{toolkit}"),"TriadSm89Finalist":format!("finalist-{toolkit}"),"TriadSm89Tf32Joint":format!("joint-{toolkit}")},"scalar_fixed_pair":[scalar,fixed]}}}));
        }
        for receipt in &mut receipts {
            receipt["identity"]["current_acceptance_source_sha256"] = json!("e".repeat(64));
        }
        receipts
    }

    #[test]
    fn six_cohort_closure_rejects_current_only_source_drift() {
        let mut receipts = six_current_receipt_fixture();
        validate_six_current(&receipts).unwrap();
        receipts[5]["identity"]["current_acceptance_source_sha256"] = json!("f".repeat(64));
        assert!(
            validate_six_current(&receipts).is_err(),
            "different current-only validator revisions must not share a six-cohort receipt"
        );
    }

    #[test]
    fn six_cohort_closure_rejects_current_only_source_missing_or_invalid() {
        let receipts = six_current_receipt_fixture();
        validate_six_current(&receipts).unwrap();
        for invalid in [
            None,
            Some(json!("e".repeat(63))),
            Some(json!("z".repeat(64))),
        ] {
            let mut changed = receipts.clone();
            let identity = changed[0]["identity"].as_object_mut().unwrap();
            identity.remove("current_acceptance_source_sha256");
            if let Some(value) = invalid {
                identity.insert("current_acceptance_source_sha256".into(), value);
            }
            assert!(
                validate_six_current(&changed).is_err(),
                "missing or malformed current-only validator provenance must fail closed"
            );
        }
    }

    #[test]
    fn combined_gemm_shared_scope_six_cohort_closure_rejects_missing_invalid_or_mixed_scope() {
        let receipts = six_current_receipt_fixture();
        validate_six_current(&receipts).unwrap();
        for invalid in [None, Some(json!("")), Some(json!("shared"))] {
            let mut changed = receipts.clone();
            let completion = changed[0].as_object_mut().unwrap();
            completion.remove("device_scope");
            if let Some(value) = invalid {
                completion.insert("device_scope".into(), value);
            }
            assert!(validate_six_current(&changed).is_err());
        }
        let mut changed = receipts.clone();
        changed[5]["device_scope"] = json!("exclusive");
        changed[5]["identity"]["device_scope"] = json!("exclusive");
        changed[5]["identity"]["physical_census"]["device_scope"] = json!("exclusive");
        assert!(validate_six_current(&changed).is_err());
    }

    #[test]
    fn combined_gemm_shared_scope_released_current_pair_rejects_missing_invalid_or_mixed_scope() {
        let current = six_current_receipt_fixture().remove(0);
        let mut released = current.clone();
        released["identity"]["released"] = json!(true);
        validate_completion_scope_pair(&released, &current).unwrap();

        for invalid in [None, Some(json!("")), Some(json!("shared"))] {
            let mut changed = released.clone();
            let completion = changed.as_object_mut().unwrap();
            completion.remove("device_scope");
            if let Some(value) = invalid {
                completion.insert("device_scope".into(), value);
            }
            assert!(validate_completion_scope_pair(&changed, &current).is_err());
        }
        released["device_scope"] = json!("exclusive");
        released["identity"]["device_scope"] = json!("exclusive");
        released["identity"]["physical_census"]["device_scope"] = json!("exclusive");
        assert!(validate_completion_scope_pair(&released, &current).is_err());
    }

    #[test]
    fn six_cohort_closure_rejects_duplicate_keys_and_cross_capacity_drift() {
        let mut receipts = six_current_receipt_fixture();
        validate_six_current(&receipts).unwrap();
        assert!(validate_six_current(&receipts[..5]).is_err());
        let mut duplicate = receipts.clone();
        duplicate[5] = duplicate[0].clone();
        assert!(validate_six_current(&duplicate).is_err());
        let mut drift = receipts.clone();
        drift[1]["identity"]["physical_census"]["modules"]["TriadSm89Half"] = json!("changed-half");
        assert!(validate_six_current(&drift).is_err());
        receipts[0]["identity"]["physical_census"]["finalist_live_function_complete"] =
            json!(false);
        assert!(validate_six_current(&receipts).is_err());
    }

    #[test]
    fn topology_uses_edges_instead_of_driver_enumeration_order() {
        assert_eq!(topological_order(2, &[(1, 0)]).unwrap(), [1, 0]);
        assert_eq!(topological_order(1, &[]).unwrap(), [0]);
        for (count, edges) in [
            (0, vec![]),
            (2, vec![]),
            (2, vec![(0, 1), (1, 0)]),
            (2, vec![(0, 2)]),
            (2, vec![(1, 0), (1, 0)]),
            (2, vec![(1, 1)]),
        ] {
            assert!(
                topological_order(count, &edges).is_err(),
                "{count} {edges:?}"
            );
        }
    }

    #[test]
    fn abi_distinguishes_bundle_total_size_and_terminal_arguments() {
        validate_abi(
            "bundle",
            &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
            true,
        )
        .unwrap();
        validate_abi(
            "tn16",
            &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)],
            true,
        )
        .unwrap();
        validate_abi("transpose", &[(0, 8), (8, 8), (16, 4), (20, 4)], true).unwrap();
        assert!(
            validate_abi(
                "bundle",
                &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)],
                false
            )
            .is_err()
        );
        assert!(
            validate_abi(
                "bundle",
                &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 64)],
                true
            )
            .is_err()
        );
        assert!(validate_abi("unknown", &[], true).is_err());
        assert!(validate_abi("transpose", &[(0, 8), (8, 8), (16, 4), (24, 4)], true).is_err());
    }
}
