//! Installed cuBLAS FAST/default versus the three SM120 Triad NN AUTO projection cells.
//! This comparator is independent of the frozen selector/performance snapshots.

#[cfg(feature = "cuda")]
#[path = "../../tests/common/gpu_quiet.rs"]
mod gpu_quiet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    dims: (usize, usize, usize),
    bn: u32,
    grid: u32,
    threads: u32,
    shared: u32,
    symbol: &'static str,
}

fn cells() -> Vec<Cell> {
    vec![
        Cell {
            name: "d768_in_proj",
            dims: (2048, 768, 3072),
            bn: 128,
            grid: 768,
            threads: 256,
            shared: 49_280,
            symbol: "nn_sm120_tma_mma_tf32_m64n128_bk32_s2",
        },
        Cell {
            name: "d768_out_proj",
            dims: (2048, 1536, 768),
            bn: 64,
            grid: 384,
            threads: 128,
            shared: 32_896,
            symbol: "nn_sm120_tma_mma_tf32_m64n64_bk32_s2",
        },
        Cell {
            name: "prism_in_proj",
            dims: (4621, 384, 1928),
            bn: 128,
            grid: 1168,
            threads: 256,
            shared: 49_280,
            symbol: "nn_sm120_tma_mma_tf32_m64n128_bk32_s2",
        },
    ]
}

fn windows(value: Option<&str>) -> Result<usize, String> {
    match value {
        None | Some("21") => Ok(21),
        Some("101") => Ok(101),
        Some(value) => Err(format!(
            "vendor windows must be exactly 21 or 101, got {value:?}"
        )),
    }
}

fn pair_schedule(reverse: bool, index: usize) -> [bool; 4] {
    let _ = index;
    [!reverse, reverse, reverse, !reverse]
}

fn require_bits(label: &str, actual: &[u32], expected: &[u32]) -> Result<(), String> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err(format!(
            "{label}: empty or unequal bit extents {} / {}",
            actual.len(),
            expected.len()
        ));
    }
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        return Err(format!(
            "{label}: bits differ at {index}: actual={:#010x} expected={:#010x}",
            actual[index], expected[index]
        ));
    }
    Ok(())
}

fn numeric_gate(label: &str, actual: &[u32], reference: &[u32]) -> Result<f64, String> {
    if actual.is_empty() || actual.len() != reference.len() {
        return Err(format!("{label}: empty or unequal numeric extents"));
    }
    let mut maximum = 0.0f64;
    for (index, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        let a = f64::from(f32::from_bits(a));
        let r = f64::from(f32::from_bits(r));
        let error = (a - r).abs();
        if !a.is_finite() || !r.is_finite() || error > 0.0025 * (1.0 + r.abs()) {
            return Err(format!(
                "{label}: numeric gate failed at {index}: actual={a} PEDANTIC={r}"
            ));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

fn guard_gate(values: &[u32], active: usize, guard: usize, sentinel: u32) -> Result<(), String> {
    if guard == 0
        || active.checked_add(guard).and_then(|n| n.checked_add(guard)) != Some(values.len())
    {
        return Err("invalid guarded extent".into());
    }
    if values[..guard]
        .iter()
        .chain(&values[guard + active..])
        .any(|&word| word != sentinel)
    {
        return Err("prefix or suffix guard changed".into());
    }
    Ok(())
}

fn validate_manifest(
    cell: Cell,
    symbol: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
) -> Result<(), String> {
    if symbol != cell.symbol
        || grid != (cell.grid, 1, 1)
        || block != (cell.threads, 1, 1)
        || shared != cell.shared
    {
        return Err(format!(
            "{}: wrong physical launch: {symbol} {grid:?} {block:?} shared={shared}",
            cell.name
        ));
    }
    Ok(())
}

#[test]
fn vendor_cpu_inventory_admits_only_the_three_exact_winners() {
    assert_eq!(
        cells()
            .iter()
            .map(|c| (c.name, c.dims, c.bn))
            .collect::<Vec<_>>(),
        [
            ("d768_in_proj", (2048, 768, 3072), 128),
            ("d768_out_proj", (2048, 1536, 768), 64),
            ("prism_in_proj", (4621, 384, 1928), 128),
        ]
    );
}

#[test]
fn vendor_cpu_window_parser_rejects_unqualified_sample_counts() {
    assert_eq!(windows(None).unwrap(), 21);
    assert_eq!(windows(Some("101")).unwrap(), 101);
    for bad in ["0", "20", "100", "102", " 21", "", "final"] {
        assert!(windows(Some(bad)).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn vendor_cpu_schedule_balances_both_arms_in_each_window_and_reverses_order() {
    for count in [21, 101] {
        for index in 0..count {
            assert_eq!(pair_schedule(false, index), [true, false, false, true]);
            assert_eq!(pair_schedule(true, index), [false, true, true, false]);
        }
    }
}

#[test]
fn vendor_cpu_raw_input_proof_rejects_mutation_and_extent_drift() {
    let expected = [1.0f32.to_bits(), (-0.0f32).to_bits()];
    require_bits("input", &expected, &expected).unwrap();
    for changed in [vec![expected[0], 0], vec![expected[0]], vec![]] {
        assert!(require_bits("input", &changed, &expected).is_err());
    }
    assert!(require_bits("input", &[], &[]).is_err());
}

#[test]
fn vendor_cpu_numeric_gate_rejects_poison_nonfinite_and_wrong_results() {
    let expected = [6.0f32.to_bits(), (-2.0f32).to_bits()];
    numeric_gate("candidate", &expected, &expected).unwrap();
    for bad in [0.0f32.to_bits(), 0x7fc12345, f32::INFINITY.to_bits()] {
        assert!(numeric_gate("candidate", &[bad, expected[1]], &expected).is_err());
    }
    assert!(numeric_gate("candidate", &expected[..1], &expected).is_err());
    assert!(numeric_gate("candidate", &expected, &[0x7fc12345, expected[1]]).is_err());
}

#[test]
fn vendor_cpu_guard_gate_rejects_each_red_zone_and_wrong_extent() {
    let sentinel = 0x4f123456;
    let good = [sentinel, 1, 2, sentinel];
    guard_gate(&good, 2, 1, sentinel).unwrap();
    for bad in [[0, 1, 2, sentinel], [sentinel, 1, 2, 0]] {
        assert!(guard_gate(&bad, 2, 1, sentinel).is_err());
    }
    assert!(guard_gate(&good[..3], 2, 1, sentinel).is_err());
    assert!(guard_gate(&good, 2, 0, sentinel).is_err());
}

#[test]
fn vendor_cpu_physical_gate_rejects_wrong_symbol_launch_or_shared() {
    let cell = Cell {
        name: "fixture",
        dims: (2048, 1536, 768),
        bn: 64,
        grid: 384,
        threads: 128,
        shared: 32896,
        symbol: "nn_sm120_tma_mma_tf32_m64n64_bk32_s2",
    };
    validate_manifest(cell, cell.symbol, (384, 1, 1), (128, 1, 1), 32896).unwrap();
    assert!(validate_manifest(cell, "wrong", (384, 1, 1), (128, 1, 1), 32896).is_err());
    assert!(validate_manifest(cell, cell.symbol, (383, 1, 1), (128, 1, 1), 32896).is_err());
    assert!(validate_manifest(cell, cell.symbol, (384, 1, 1), (256, 1, 1), 32896).is_err());
    assert!(validate_manifest(cell, cell.symbol, (384, 1, 1), (128, 1, 1), 32800).is_err());
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::{CudaGraph, sys};
    use gpu_quiet::QuietGpu;
    use mamba_rs::mamba_ssm::gpu::GemmMode;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationF32Epilogue, PhysicalQualificationRequest, PhysicalQualificationRoute,
        QualifiedPhysicalLaunch, Tf32PhysicalRoute, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile,
        qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, ResolvedGemmOp, ResolvedNumericContract, digest_hex,
    };
    use sha2::{Digest, Sha256};
    use std::ffi::{CStr, c_void};
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};

    const SALT: u64 = 0x1205_32a1;
    const GUARD: usize = 32;
    const CANARY: u32 = 0x4f12_3456;
    const SCHEMA: &str = "MambaBiSm120Tf32AutoVendorV1";
    const OUTPUT_ENV: &str = "MAMBA_RS_SM120_TF32_VENDOR_JSONL";
    const WINDOWS_ENV: &str = "MAMBA_RS_SM120_TF32_VENDOR_WINDOWS";

    fn quoted(s: &str) -> String {
        let mut out = String::from("\"");
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                ch if ch.is_control() => {
                    use std::fmt::Write;
                    write!(out, "\\u{:04x}", ch as u32).unwrap();
                }
                ch => out.push(ch),
            }
        }
        out.push('"');
        out
    }

    fn sha(bytes: &[u8]) -> String {
        digest_hex(&Sha256::digest(bytes).into())
    }
    fn bits_digest(bits: &[u32]) -> String {
        let mut hash = Sha256::new();
        for word in bits {
            hash.update(word.to_le_bytes());
        }
        digest_hex(&hash.finalize().into())
    }
    fn as_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }
    fn raw(values: &[f64]) -> String {
        format!(
            "[{}]",
            values
                .iter()
                .map(|x| format!("{x:.9}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
    fn sync(ctx: &GpuCtx) -> Result<(), String> {
        ctx.stream
            .synchronize()
            .map_err(|e| format!("synchronize: {e:?}"))
    }

    // Independent reproduction of the public holder's finite corpus. Runtime
    // readback of every active A/B word proves this is not merely a same-salt claim.
    fn seeded(len: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ salt;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = (state.wrapping_add(index as u64) % 4093) as i32 - 2046;
                signed as f32 / 1024.0
            })
            .collect()
    }

    struct Corpus {
        a: Vec<f32>,
        b: Vec<f32>,
        output: Vec<f32>,
    }
    impl Corpus {
        fn dense(cell: Cell) -> Self {
            let (m, k, n) = cell.dims;
            Self {
                a: seeded(m * k, SALT ^ 0x2d),
                b: seeded(k * n, SALT ^ 0x67),
                output: seeded(m * n, SALT ^ 0x91),
            }
        }
        fn single(cell: Cell) -> Self {
            let (m, k, n) = cell.dims;
            let mut a = vec![0.0; m * k];
            let mut b = vec![0.0; k * n];
            for row in 0..m {
                a[row * k] = ((row % 7) as i32 - 3) as f32 * 0.125;
            }
            for (col, value) in b.iter_mut().take(n).enumerate() {
                *value = ((col % 11) as i32 - 5) as f32 * 0.125;
            }
            Self {
                a,
                b,
                output: vec![0.0; m * n],
            }
        }
        fn verify_representable(&self) -> Result<(), String> {
            for value in self.a.iter().chain(&self.b) {
                if !value.is_finite() || value.to_bits() & 0x1fff != 0 {
                    return Err("timed input is not finite exactly representable TF32".into());
                }
            }
            Ok(())
        }
    }

    struct Guarded {
        buffer: GpuBuffer,
        active: usize,
    }
    impl Guarded {
        fn new(ctx: &GpuCtx, active: usize) -> Result<Self, String> {
            let mut result = Self {
                buffer: GpuBuffer::zeros(&ctx.stream, active + 2 * GUARD)?,
                active,
            };
            result.upload(ctx, &vec![0.0; active])?;
            Ok(result)
        }
        fn ptr(&self) -> u64 {
            self.buffer.cached_ptr() + (GUARD * 4) as u64
        }
        fn upload(&mut self, ctx: &GpuCtx, active: &[f32]) -> Result<(), String> {
            if active.len() != self.active {
                return Err("vendor upload extent mismatch".into());
            }
            let mut host = vec![f32::from_bits(CANARY); self.active + 2 * GUARD];
            host[GUARD..GUARD + self.active].copy_from_slice(active);
            self.buffer.upload(&ctx.stream, &host)?;
            sync(ctx)
        }
        fn read(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
            let host = self.buffer.to_cpu(&ctx.stream)?;
            sync(ctx)?;
            let bits = as_bits(&host);
            guard_gate(&bits, self.active, GUARD, CANARY)?;
            Ok(bits[GUARD..GUARD + self.active].to_vec())
        }
    }

    struct Vendor {
        a: Guarded,
        b: Guarded,
        fast: Guarded,
        pedantic: Guarded,
    }
    impl Vendor {
        fn new(ctx: &GpuCtx, cell: Cell) -> Result<Self, String> {
            let (m, k, n) = cell.dims;
            Ok(Self {
                a: Guarded::new(ctx, m * k)?,
                b: Guarded::new(ctx, k * n)?,
                fast: Guarded::new(ctx, m * n)?,
                pedantic: Guarded::new(ctx, m * n)?,
            })
        }
        fn upload(&mut self, ctx: &GpuCtx, corpus: &Corpus) -> Result<(), String> {
            self.a.upload(ctx, &corpus.a)?;
            self.b.upload(ctx, &corpus.b)?;
            self.fast.upload(ctx, &corpus.output)?;
            self.pedantic.upload(ctx, &corpus.output)
        }
        fn verify_inputs(&self, ctx: &GpuCtx, corpus: &Corpus) -> Result<(), String> {
            require_bits("vendor A readback", &self.a.read(ctx)?, &as_bits(&corpus.a))?;
            require_bits("vendor B readback", &self.b.read(ctx)?, &as_bits(&corpus.b))
        }
        fn launch(&self, ctx: &GpuCtx, cell: Cell, pedantic: bool) -> Result<(), String> {
            use cudarc::cublas::sys::{
                cublasComputeType_t::*, cublasGemmAlgo_t, cublasOperation_t::CUBLAS_OP_N,
            };
            let (m, k, n) = cell.dims;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let output = if pedantic {
                self.pedantic.ptr()
            } else {
                self.fast.ptr()
            };
            // Row-major NN is transposed as column-major C^T = B^T * A^T.
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *ctx.blas.handle(),
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha as *const _ as *const c_void,
                    self.b.ptr() as *const c_void,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
                    n as i32,
                    self.a.ptr() as *const c_void,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
                    k as i32,
                    &beta as *const _ as *const c_void,
                    output as *mut c_void,
                    cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
                    n as i32,
                    if pedantic {
                        CUBLAS_COMPUTE_32F_PEDANTIC
                    } else {
                        CUBLAS_COMPUTE_32F_FAST_TF32
                    },
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| format!("cuBLAS NN pedantic={pedantic}: {e:?}"))
        }
    }

    fn request(cell: Cell, forced: bool) -> PhysicalQualificationRequest {
        let route = if forced {
            PhysicalQualificationRoute::Tf32Forced(Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(
                Tf32Sm120Route {
                    tile: if cell.bn == 64 {
                        Tf32Sm120Tile::M64N64
                    } else {
                        Tf32Sm120Tile::M64N128
                    },
                    stages: Tf32Sm120Stages::S2,
                },
            ))
        } else {
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32)
        };
        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            cell.dims,
            route,
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
        )
    }

    fn own_manifest(
        ctx: &GpuCtx,
        cell: Cell,
        launch: &QualifiedPhysicalLaunch<'_>,
        forced: bool,
    ) -> Result<String, String> {
        launch.validate_timed_request(ctx, request(cell, forced))?;
        let evidence = launch.evidence();
        let [node] = evidence.nodes() else {
            return Err("expected exactly one actual own GEMM node".into());
        };
        validate_manifest(
            cell,
            node.symbol,
            node.launch.grid_dim,
            node.launch.block_dim,
            node.launch.shared_mem_bytes,
        )?;
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || evidence.route_identity().tuning_table_revision != 39
            || node.module_kind != ModuleKind::TriadSm120
            || node.logical_op != ResolvedGemmOp::Nn
            || node.shape != cell.dims
            || node.strides != (cell.dims.1, cell.dims.2, cell.dims.2)
            || node.tile != Some((64, cell.bn))
            || node.numeric_contract != Some(ResolvedNumericContract::Sm120TmaMmaTf32Rna)
            || node.launch.arguments_digest == [0; 32]
        {
            return Err(format!(
                "{}: actual physical evidence changed: {evidence:?}",
                cell.name
            ));
        }
        Ok(format!(
            "{{\"symbol\":{},\"grid\":[{},1,1],\"block\":[{},1,1],\"shared\":{},\"arguments_digest\":{},\"launch_digest\":{},\"request_digest\":{},\"eager_graph_equal\":true}}",
            quoted(node.symbol),
            cell.grid,
            cell.threads,
            cell.shared,
            quoted(&digest_hex(&node.launch.arguments_digest)),
            quoted(&digest_hex(&evidence.launch_digest())),
            quoted(&digest_hex(&evidence.request_identity_digest()))
        ))
    }

    fn own_inputs(
        ctx: &GpuCtx,
        launch: &QualifiedPhysicalLaunch<'_>,
        corpus: &Corpus,
    ) -> Result<(), String> {
        let (a, b) = launch.f32_operand_bits(ctx)?;
        require_bits("own A public-seed readback", &a, &as_bits(&corpus.a))?;
        require_bits("own B public-seed readback", &b, &as_bits(&corpus.b))?;
        let guards = launch.validate_red_zones(ctx)?;
        if guards.allocation_count() != 3 || guards.element_count() != 96 {
            return Err(format!("own trailing guard inventory changed: {guards:?}"));
        }
        Ok(())
    }

    fn cuda_ok(status: sys::CUresult, label: &str) -> Result<(), String> {
        if status == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {status:?}"))
        }
    }
    // Read the installed implementation as it is, including multiple kernels,
    // memcpy/memset nodes and child graphs. No historical vendor-symbol seal.
    fn graph_description(graph: sys::CUgraph, depth: usize) -> Result<(String, usize), String> {
        if depth > 8 {
            return Err("vendor graph nesting exceeds diagnostic bound".into());
        }
        let mut count = 0;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) },
            "graph node count",
        )?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        if count == 0 {
            return Err("empty vendor graph".into());
        }
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) },
            "graph nodes",
        )?;
        let mut rendered = Vec::new();
        let mut kernels = 0;
        for node in &nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            cuda_ok(
                unsafe { sys::cuGraphNodeGetType(*node, &mut kind) },
                "graph node kind",
            )?;
            if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                let mut params = unsafe { std::mem::zeroed() };
                cuda_ok(
                    unsafe { sys::cuGraphKernelNodeGetParams_v2(*node, &mut params) },
                    "graph kernel params",
                )?;
                let mut name = std::ptr::null();
                cuda_ok(
                    unsafe { sys::cuFuncGetName(&mut name, params.func) },
                    "graph kernel symbol",
                )?;
                if name.is_null()
                    || [
                        params.gridDimX,
                        params.gridDimY,
                        params.gridDimZ,
                        params.blockDimX,
                        params.blockDimY,
                        params.blockDimZ,
                    ]
                    .contains(&0)
                {
                    return Err("invalid vendor kernel descriptor".into());
                }
                let name = unsafe { CStr::from_ptr(name) }
                    .to_str()
                    .map_err(|e| e.to_string())?;
                rendered.push(format!("{{\"kind\":\"kernel\",\"symbol\":{},\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared\":{}}}",quoted(name),params.gridDimX,params.gridDimY,params.gridDimZ,params.blockDimX,params.blockDimY,params.blockDimZ,params.sharedMemBytes));
                kernels += 1;
            } else if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH {
                let mut child = std::ptr::null_mut();
                cuda_ok(
                    unsafe { sys::cuGraphChildGraphNodeGetGraph(*node, &mut child) },
                    "child graph",
                )?;
                let (description, children) = graph_description(child, depth + 1)?;
                kernels += children;
                rendered.push(format!(
                    "{{\"kind\":\"child_graph\",\"graph\":{description}}}"
                ));
            } else {
                rendered.push(format!("{{\"kind\":{}}}", quoted(&format!("{kind:?}"))));
            }
        }
        let mut edge_count = 0;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                )
            },
            "graph edge count",
        )?;
        let mut from = vec![std::ptr::null_mut(); edge_count];
        let mut to = from.clone();
        let mut data = Vec::with_capacity(edge_count);
        data.resize_with(edge_count, || unsafe { std::mem::zeroed() });
        if edge_count > 0 {
            cuda_ok(
                unsafe {
                    sys::cuGraphGetEdges_v2(
                        graph,
                        from.as_mut_ptr(),
                        to.as_mut_ptr(),
                        data.as_mut_ptr(),
                        &mut edge_count,
                    )
                },
                "graph edges",
            )?;
        }
        let edges = from
            .iter()
            .zip(&to)
            .map(|(a, b)| {
                let a = nodes
                    .iter()
                    .position(|node| node == a)
                    .ok_or("missing edge source")?;
                let b = nodes
                    .iter()
                    .position(|node| node == b)
                    .ok_or("missing edge target")?;
                Ok::<_, String>(format!("[{a},{b}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            format!(
                "{{\"nodes\":[{}],\"edges\":[{}]}}",
                rendered.join(","),
                edges.join(",")
            ),
            kernels,
        ))
    }

    fn measure_vendor(
        ctx: &GpuCtx,
        vendor: &Vendor,
        cell: Cell,
        graph: Option<&CudaGraph>,
        iterations: usize,
    ) -> Result<f64, String> {
        let start = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("start event: {e:?}"))?;
        for _ in 0..iterations {
            if let Some(graph) = graph {
                graph.launch().map_err(|e| format!("graph replay: {e:?}"))?;
            } else {
                vendor.launch(ctx, cell, false)?;
            }
        }
        let end = ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| format!("end event: {e:?}"))?;
        let ms = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|e| format!("elapsed event: {e:?}"))?,
        );
        sync(ctx)?;
        checked_us(ms, iterations)
    }
    fn checked_us(ms: f64, iterations: usize) -> Result<f64, String> {
        let us = ms * 1000.0 / iterations as f64;
        if iterations == 0 || !us.is_finite() || us <= 0.0 {
            Err(format!("invalid timing {ms}ms / {iterations}"))
        } else {
            Ok(us)
        }
    }
    fn measure_own(
        ctx: &GpuCtx,
        own: &mut QualifiedPhysicalLaunch<'_>,
        graph: bool,
        forced: bool,
        iterations: usize,
    ) -> Result<f64, String> {
        let ms = if graph {
            own.measure_graph_window_ms(ctx, iterations)?
        } else if forced {
            own.measure_prevalidated_forced_eager_window_ms(ctx, iterations)?
        } else {
            own.measure_eager_window_ms(ctx, iterations)?
        };
        checked_us(ms, iterations)
    }
    fn calibrate(us: f64) -> usize {
        (5000.0 / us).ceil().clamp(1.0, 1_000_000.0) as usize
    }
    fn percentile(values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    fn verify_outputs(
        ctx: &GpuCtx,
        own: &QualifiedPhysicalLaunch<'_>,
        vendor_ctx: &GpuCtx,
        vendor: &Vendor,
        corpus: &Corpus,
        reference: &[u32],
        goldens: (Option<&[u32]>, Option<&[u32]>),
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        own_inputs(ctx, own, corpus)?;
        vendor.verify_inputs(vendor_ctx, corpus)?;
        let own_bits = own.f32_output_bits(ctx)?;
        let vendor_bits = vendor.fast.read(vendor_ctx)?;
        numeric_gate("own versus independent PEDANTIC", &own_bits, reference)?;
        numeric_gate("FAST versus independent PEDANTIC", &vendor_bits, reference)?;
        require_bits(
            "PEDANTIC output immutable",
            &vendor.pedantic.read(vendor_ctx)?,
            reference,
        )?;
        if let Some(golden) = goldens.0 {
            require_bits("own eager/graph/raw repeat", &own_bits, golden)?;
        }
        if let Some(golden) = goldens.1 {
            require_bits("FAST eager/graph/raw repeat", &vendor_bits, golden)?;
        }
        Ok((own_bits, vendor_bits))
    }

    struct Sink {
        writer: BufWriter<std::fs::File>,
        hash: Sha256,
        records: usize,
    }
    impl Sink {
        fn new() -> Result<Self, String> {
            let path = std::env::var(OUTPUT_ENV)
                .map_err(|_| format!("{OUTPUT_ENV} must name a NEW JSONL file"))?;
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| format!("create {path}: {e}"))?;
            Ok(Self {
                writer: BufWriter::new(file),
                hash: Sha256::new(),
                records: 0,
            })
        }
        fn record(&mut self, record: &str) -> Result<(), String> {
            writeln!(self.writer, "{record}").map_err(|e| e.to_string())?;
            self.hash.update(record.as_bytes());
            self.hash.update(b"\n");
            self.records += 1;
            self.writer.flush().map_err(|e| e.to_string())
        }
        fn finish(mut self, windows: usize) -> Result<(), String> {
            if self.records != 24 {
                return Err(format!(
                    "incomplete vendor dataset: {}/24 records",
                    self.records
                ));
            }
            let digest = digest_hex(&self.hash.finalize().into());
            writeln!(self.writer,"{{\"schema\":\"{SCHEMA}Complete\",\"complete\":true,\"cells\":3,\"measurement_records\":24,\"windows_per_order\":{windows},\"measurement_records_sha256\":\"{digest}\",\"dispatch_admission\":false}}").map_err(|e|e.to_string())?;
            self.writer.flush().map_err(|e| e.to_string())?;
            self.writer.get_ref().sync_all().map_err(|e| e.to_string())
        }
    }

    fn context(device: &GpuDevice) -> Result<GpuCtx, String> {
        let ctx = GpuCtx::new(device)?;
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.route_controls().set_family(BiGemmFamily::Triad);
        ctx.route_controls()
            .set_f32_policy(F32TriadPolicy::AllowDeterministicTf32);
        Ok(ctx)
    }
    fn binding(ctx: &GpuCtx) -> Result<String, String> {
        let bound = ctx
            .kernels
            .f32_triad_availability()
            .specialized
            .ok_or("no bound SM120 module")?;
        if bound.module_kind != ModuleKind::TriadSm120
            || bound.compiler.nvrtc_version != (13, 2)
            || !bound.compiler.nvrtc_library_known
            || digest_hex(&bound.artifact.artifact_digest)
                != "6a5feb9e46d25b635ec84e53b39aad644f97d83fa06ccbc8d94c95fa32ad3235"
        {
            return Err(format!("unqualified SM120 module identity: {bound:?}"));
        }
        Ok(quoted(&format!("{bound:?}")))
    }

    fn qualify_warmed(
        ctx: &GpuCtx,
        cell: Cell,
        forced: bool,
    ) -> Result<QualifiedPhysicalLaunch<'_>, String> {
        // The public constructor captures at construction. Warm a temporary
        // holder first, then construct the graph that will actually be timed.
        {
            let mut warm = qualify_physical_launch(ctx, request(cell, forced))?;
            warm.seed_f32_operands(ctx, SALT)?;
            measure_own(ctx, &mut warm, false, forced, 128)?;
        }
        qualify_physical_launch(ctx, request(cell, forced))
    }

    fn run_cell(
        device: &GpuDevice,
        quiet: &QuietGpu,
        cell: Cell,
        windows: usize,
        sink: &mut Sink,
        poison_red: bool,
    ) -> Result<(), String> {
        let auto_ctx = context(device)?;
        let forced_ctx = context(device)?;
        let vendor_ctx = GpuCtx::new(device)?;
        let identity = binding(&auto_ctx)?;
        if binding(&forced_ctx)? != identity {
            return Err("AUTO/forced module identities differ".into());
        }
        let mut auto = qualify_warmed(&auto_ctx, cell, false)?;
        let mut forced = qualify_warmed(&forced_ctx, cell, true)?;
        let auto_manifest = own_manifest(&auto_ctx, cell, &auto, false)?;
        let forced_manifest = own_manifest(&forced_ctx, cell, &forced, true)?;
        let mut vendor = Vendor::new(&vendor_ctx, cell)?;
        let mut version = 0;
        let status = unsafe {
            cudarc::cublas::sys::cublasGetVersion_v2(*vendor_ctx.blas.handle(), &mut version)
        };
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || version <= 0 {
            return Err(format!("cuBLAS version: {status:?}/{version}"));
        }

        // Independent one-term exact orientation probe runs before any timing.
        let single = Corpus::single(cell);
        vendor.upload(&vendor_ctx, &single)?;
        vendor.launch(&vendor_ctx, cell, true)?;
        vendor.launch(&vendor_ctx, cell, false)?;
        sync(&vendor_ctx)?;
        let expected = (0..cell.dims.0 * cell.dims.2)
            .map(|i| {
                let a = ((i / cell.dims.2 % 7) as i32 - 3) as f32 * 0.125;
                let b = ((i % cell.dims.2 % 11) as i32 - 5) as f32 * 0.125;
                a.mul_add(b, 0.0).to_bits()
            })
            .collect::<Vec<_>>();
        require_bits(
            "PEDANTIC orientation",
            &vendor.pedantic.read(&vendor_ctx)?,
            &expected,
        )?;
        require_bits(
            "FAST orientation",
            &vendor.fast.read(&vendor_ctx)?,
            &expected,
        )?;
        for (ctx, launch) in [(&auto_ctx, &mut auto), (&forced_ctx, &mut forced)] {
            for graph in [false, true] {
                launch.seed_f32_nn_single_term_probe(ctx)?;
                own_inputs(ctx, launch, &single)?;
                if poison_red {
                    return require_bits(
                        "VENDOR_GPU_MISSING_LAUNCH_RED",
                        &launch.f32_output_bits(ctx)?,
                        &expected,
                    );
                }
                if graph {
                    launch.measure_graph_window_ms(ctx, 1)?;
                } else {
                    launch.measure_eager_window_ms(ctx, 1)?;
                }
                require_bits("own orientation", &launch.f32_output_bits(ctx)?, &expected)?;
                own_inputs(ctx, launch, &single)?;
            }
        }

        let corpus = Corpus::dense(cell);
        corpus.verify_representable()?;
        vendor.upload(&vendor_ctx, &corpus)?;
        auto.seed_f32_operands(&auto_ctx, SALT)?;
        forced.seed_f32_operands(&forced_ctx, SALT)?;
        own_inputs(&auto_ctx, &auto, &corpus)?;
        own_inputs(&forced_ctx, &forced, &corpus)?;
        vendor.verify_inputs(&vendor_ctx, &corpus)?;
        vendor.launch(&vendor_ctx, cell, true)?;
        sync(&vendor_ctx)?;
        let reference = vendor.pedantic.read(&vendor_ctx)?;
        numeric_gate("PEDANTIC finite reference", &reference, &reference)?;

        // Warm both production and vendor eager paths BEFORE graph capture.
        measure_own(&auto_ctx, &mut auto, false, false, 128)?;
        measure_own(&forced_ctx, &mut forced, false, true, 128)?;
        measure_vendor(&vendor_ctx, &vendor, cell, None, 128)?;
        let graph = unsafe {
            capture_into_graph(&vendor_ctx.stream, || {
                vendor.launch(&vendor_ctx, cell, false)
            })
        }?;
        let (vendor_graph, kernel_count) = graph_description(graph.cu_graph(), 0)?;
        if kernel_count == 0 {
            return Err("vendor graph contains no kernels".into());
        }

        let mut auto_golden = None;
        let mut forced_golden = None;
        let mut vendor_golden = None;
        for use_graph in [false, true, true, false] {
            auto.seed_f32_operands(&auto_ctx, SALT)?;
            forced.seed_f32_operands(&forced_ctx, SALT)?;
            vendor.fast.upload(&vendor_ctx, &corpus.output)?;
            measure_own(&auto_ctx, &mut auto, use_graph, false, 1)?;
            measure_own(&forced_ctx, &mut forced, use_graph, true, 1)?;
            measure_vendor(&vendor_ctx, &vendor, cell, use_graph.then_some(&graph), 1)?;
            let (a, v) = verify_outputs(
                &auto_ctx,
                &auto,
                &vendor_ctx,
                &vendor,
                &corpus,
                &reference,
                (auto_golden.as_deref(), vendor_golden.as_deref()),
            )?;
            let (f, _) = verify_outputs(
                &forced_ctx,
                &forced,
                &vendor_ctx,
                &vendor,
                &corpus,
                &reference,
                (forced_golden.as_deref(), vendor_golden.as_deref()),
            )?;
            require_bits("AUTO versus forced exact bits", &a, &f)?;
            auto_golden = Some(a);
            forced_golden = Some(f);
            vendor_golden = Some(v);
        }
        let auto_golden = auto_golden.unwrap();
        let forced_golden = forced_golden.unwrap();
        let vendor_golden = vendor_golden.unwrap();
        for (mode, ctx, own, own_golden, manifest, is_forced) in [
            (
                "auto",
                &auto_ctx,
                &mut auto,
                &auto_golden,
                &auto_manifest,
                false,
            ),
            (
                "forced",
                &forced_ctx,
                &mut forced,
                &forced_golden,
                &forced_manifest,
                true,
            ),
        ] {
            for use_graph in [false, true] {
                let path = if use_graph { "graph" } else { "eager" };
                let vendor_replay = use_graph.then_some(&graph);
                measure_own(ctx, own, use_graph, is_forced, 128)?;
                measure_vendor(&vendor_ctx, &vendor, cell, vendor_replay, 128)?;
                let own_iterations = calibrate(measure_own(ctx, own, use_graph, is_forced, 16)?);
                let vendor_iterations = calibrate(measure_vendor(
                    &vendor_ctx,
                    &vendor,
                    cell,
                    vendor_replay,
                    16,
                )?);
                for reverse in [false, true] {
                    let order = if reverse { "BAAB" } else { "ABBA" };
                    let label = format!("sm120-vendor/{}/{mode}/{path}/{order}", cell.name);
                    let pre = quiet.require_cohort(&label)?;
                    verify_outputs(
                        ctx,
                        own,
                        &vendor_ctx,
                        &vendor,
                        &corpus,
                        &reference,
                        (Some(own_golden), Some(&vendor_golden)),
                    )?;
                    let mut own_samples = Vec::with_capacity(windows);
                    let mut vendor_samples = Vec::with_capacity(windows);
                    let mut ratios = Vec::with_capacity(windows);
                    let mut subwindows = Vec::with_capacity(windows);
                    for index in 0..windows {
                        let mut a = 0.0;
                        let mut b = 0.0;
                        let mut sequence = Vec::with_capacity(4);
                        for candidate in pair_schedule(reverse, index) {
                            let us = if candidate {
                                measure_own(ctx, own, use_graph, is_forced, own_iterations)?
                            } else {
                                measure_vendor(
                                    &vendor_ctx,
                                    &vendor,
                                    cell,
                                    vendor_replay,
                                    vendor_iterations,
                                )?
                            };
                            if candidate {
                                a += us;
                            } else {
                                b += us;
                            }
                            sequence.push(us);
                        }
                        a /= 2.0;
                        b /= 2.0;
                        own_samples.push(a);
                        vendor_samples.push(b);
                        ratios.push(a / b);
                        subwindows.push(raw(&sequence));
                    }
                    verify_outputs(
                        ctx,
                        own,
                        &vendor_ctx,
                        &vendor,
                        &corpus,
                        &reference,
                        (Some(own_golden), Some(&vendor_golden)),
                    )?;
                    let post = quiet.verify_post_cohort(&label)?;
                    let (m, k, n) = cell.dims;
                    sink.record(&format!(concat!("{{\"schema\":\"{}\",\"cell\":{},\"dims_mkn\":[{},{},{}],\"mode\":{},\"path\":{},\"order\":{},",
                        "\"windows\":{},\"own_iterations\":{},\"vendor_iterations\":{},\"alpha\":1,\"beta_bits\":0,\"bias\":false,",
                        "\"vendor_compute\":\"CUBLAS_COMPUTE_32F_FAST_TF32\",\"vendor_algorithm\":\"CUBLAS_GEMM_DEFAULT\",\"cublas_version\":{},\"reference_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",",
                        "\"timing\":\"CUDA_event_device_elapsed\",\"target_window_ms\":5,\"warmup_launches\":128,\"reference_timed\":false,\"dispatch_admission\":false,",
                        "\"own_physical\":{},\"vendor_graph\":{},\"module_identity\":{},\"test_source_sha256\":{},\"qualification_source_sha256\":{},\"gpu_uuid\":{},",
                        "\"input_a_sha256\":{},\"input_b_sha256\":{},\"own_output_sha256\":{},\"vendor_output_sha256\":{},\"reference_sha256\":{},",
                        "\"public_seed_inputs_verified\":true,\"own_forced_auto_bits_equal\":true,\"eager_graph_raw_repeat_equal\":true,\"numeric_and_guards_pre_post\":true,",
                        "\"own_guard_scope\":\"three_trailing_red_zones_32_words_each\",\"vendor_guard_scope\":\"four_prefix_and_suffix_red_zones_32_words_each\",",
                        "\"numeric_tolerance\":\"0.0025*(1+abs(PEDANTIC))\",\"timed_corpus\":\"finite_TF32_representable\",\"preflight\":{},\"postflight\":{},",
                        "\"own_us\":{},\"vendor_us\":{},\"own_over_vendor\":{},\"subwindows_execution_order_us\":[{}],\"ratio_p50\":{:.9},\"ratio_p95\":{:.9}}}"),
                        SCHEMA,quoted(cell.name),m,k,n,quoted(mode),quoted(path),quoted(order),windows,own_iterations,vendor_iterations,version,manifest,vendor_graph,identity,
                        quoted(&sha(include_bytes!("gemm_bi_sm120_tf32_vendor.rs"))),quoted(&sha(include_bytes!("../../src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs"))),quoted(&quiet.uuid),
                        quoted(&bits_digest(&as_bits(&corpus.a))),quoted(&bits_digest(&as_bits(&corpus.b))),quoted(&bits_digest(own_golden)),quoted(&bits_digest(&vendor_golden)),quoted(&bits_digest(&reference)),quoted(&pre),quoted(&post),
                        raw(&own_samples),raw(&vendor_samples),raw(&ratios),subwindows.join(","),percentile(&ratios,0.5),percentile(&ratios,0.95)))?;
                    println!(
                        "{} {mode} {path} {order}: own/FAST p50={:.6} p95={:.6}",
                        cell.name,
                        percentile(&ratios, 0.5),
                        percentile(&ratios, 0.95)
                    );
                }
            }
        }
        verify_outputs(
            &auto_ctx,
            &auto,
            &vendor_ctx,
            &vendor,
            &corpus,
            &reference,
            (Some(&auto_golden), Some(&vendor_golden)),
        )?;
        verify_outputs(
            &forced_ctx,
            &forced,
            &vendor_ctx,
            &vendor,
            &corpus,
            &reference,
            (Some(&forced_golden), Some(&vendor_golden)),
        )?;
        sync(&vendor_ctx)?;
        sync(&auto_ctx)?;
        sync(&forced_ctx)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090 on CUDA 13.2; explicit enabled release-only vendor measurement"]
    fn sm120_tf32_auto_forced_vs_installed_fast_cublas() -> Result<(), String> {
        if cfg!(debug_assertions) {
            return Err("vendor comparator requires --release".into());
        }
        if std::env::var("MAMBA_RS_SM120_TF32_VENDOR").as_deref() != Ok("1") {
            return Err("set MAMBA_RS_SM120_TF32_VENDOR=1".into());
        }
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0") {
            return Err("NVIDIA_TF32_OVERRIDE=0 disables FAST_TF32".into());
        }
        let count = windows(std::env::var(WINDOWS_ENV).ok().as_deref())?;
        let poison_red =
            std::env::var("MAMBA_RS_SM120_TF32_VENDOR_POISON_RED").as_deref() == Ok("1");
        let mut sink = Sink::new()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("sm120-vendor/pre-context")?;
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (12, 0) || device.multiprocessor_count() != 170 {
            return Err("requires exact CC12.0/170SM".into());
        }
        for cell in cells() {
            run_cell(&device, &quiet, cell, count, &mut sink, poison_red)?;
        }
        sink.finish(count)
    }
}
