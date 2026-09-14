//! Public-call payload shared verbatim with the isolated v0.7.0 overlay.

use super::gpu_quiet::QuietGpu;
use cudarc::driver::{CudaGraph, DevicePtr};
use mamba_rs::mamba_ssm::gpu::{
    GemmMode,
    blas::{
        TypedPtr, gpu_gemm_bi_backward_dw_grad, gpu_gemm_bi_backward_dw_grad_typed,
        gpu_gemm_bi_backward_dx_raw, gpu_gemm_ex_backward_dx_typed, gpu_gemm_typed_forward_raw,
    },
    buffers::{DtypedBuf, GpuBuffer, GradSlice},
    context::{BiGemmFamily, F32TriadPolicy, GpuCtx, HalfTriadPolicy},
    device::GpuDevice,
    dtype::WeightDtype,
    graph_capture::capture_into_graph,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::Path;

pub(super) const RELEASED_SHA: &str = "e2917a47494b4a1d652f5c818c3974ec3fcdd1ca";
const GUARD_BYTES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AcceptanceCase {
    pub id: String,
    pub family: &'static str,
    pub op: &'static str,
    pub row: &'static str,
    pub storage: [&'static str; 3],
    pub dims: (usize, usize, usize),
    pub bias: bool,
    pub required: bool,
}

pub(super) fn inventory() -> Result<Vec<AcceptanceCase>, String> {
    let cases = expand_inventory();
    validate_inventory(&cases)?;
    Ok(cases)
}

fn expand_inventory() -> Vec<AcceptanceCase> {
    let mut cases = Vec::new();
    let inference = [
        ("hot_a", (4621, 384, 1928)),
        ("hot_b", (4621, 768, 2304)),
        ("hot_c", (4621, 1928, 384)),
        ("hot_d", (2048, 768, 2304)),
        ("hot_e", (2048, 2304, 768)),
    ];
    for row in ["bf16", "f16", "bf16_f32", "f16_f32", "tf32", "f32_exact"] {
        for (label, dims) in inference {
            for bias in [false, true] {
                let required = (matches!(row, "bf16_f32" | "f16_f32")
                    && matches!(label, "hot_b" | "hot_c"))
                    || (row == "f32_exact" && label == "hot_c" && !bias);
                cases.push(case("inference", "nn", row, (label, dims), bias, required));
            }
        }
    }
    let triad = [
        ("d768_in", (2048, 768, 3072)),
        ("d768_out", (2048, 1536, 768)),
        ("prism", (4621, 384, 1928)),
    ];
    for row in ["bf16", "f16"] {
        for op in ["nn", "tn", "nt"] {
            for shape in triad {
                cases.push(case("triad", op, row, shape, false, false));
                if op == "nn" {
                    cases.push(case("triad", op, row, shape, true, false));
                }
            }
        }
        cases.push(case(
            "triad",
            "tn",
            row,
            ("small16", (1024, 256, 128)),
            false,
            true,
        ));
    }
    for shape in triad
        .into_iter()
        .chain([("large_deep", (4096, 3072, 1536))])
    {
        cases.push(case(
            "triad",
            "nt",
            "tf32",
            shape,
            false,
            shape.0 == "prism",
        ));
        cases.push(case("triad", "nn", "f32_exact", shape, false, false));
        cases.push(case(
            "triad",
            "nt",
            "f32_exact",
            shape,
            false,
            shape.0 == "large_deep",
        ));
    }
    cases.push(case(
        "triad",
        "tn",
        "f32_exact",
        ("small16", (256, 512, 384)),
        false,
        true,
    ));
    cases
}

fn case(
    family: &'static str,
    op: &'static str,
    row: &'static str,
    shape: (&str, (usize, usize, usize)),
    bias: bool,
    required: bool,
) -> AcceptanceCase {
    let input = match row {
        "bf16" | "bf16_f32" => "bf16",
        "f16" | "f16_f32" => "f16",
        _ => "f32",
    };
    let output = if op == "tn" || row.ends_with("_f32") {
        "f32"
    } else {
        input
    };
    AcceptanceCase {
        id: format!("{family}.{op}.{row}.{}.bias{}", shape.0, u8::from(bias)),
        family,
        op,
        row,
        storage: [input, input, output],
        dims: shape.1,
        bias,
        required,
    }
}

impl AcceptanceCase {
    pub(super) fn strides(&self) -> (usize, usize, usize) {
        let (_, k, n) = self.dims;
        if self.op == "nt" {
            (n, n, k)
        } else {
            (k, n, n)
        }
    }

    pub(super) fn key(&self) -> String {
        format!(
            "{}|{}|{:?}|{:?}|{}|3f800000|{}|{}|{:?}",
            self.family,
            self.op,
            self.storage,
            self.dims,
            self.row,
            if self.op == "tn" {
                "3f800000"
            } else {
                "00000000"
            },
            self.bias,
            self.strides()
        )
    }
}

pub(super) fn validate_inventory(cases: &[AcceptanceCase]) -> Result<(), String> {
    let mut keys = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for case in cases {
        if !keys.insert(case.key()) || !ids.insert(&case.id) {
            return Err(format!("duplicate acceptance key {}", case.id));
        }
    }
    let expected = expand_inventory();
    if cases.len() != 99
        || cases.iter().filter(|case| case.required).count() != 14
        || cases.iter().any(|case| !expected.contains(case))
    {
        return Err("acceptance inventory must contain exactly the closed 99 cases, including all 14 required cells and 85 controls".into());
    }
    Ok(())
}

pub(super) fn explicit_cohort(toolkit: &str, cap: Option<&str>) -> Result<String, String> {
    match (toolkit, cap) {
        ("12.8" | "13.0" | "13.2", Some("16" | "64")) => {
            Ok(format!("cuda{toolkit}-cap{}", cap.unwrap()))
        }
        _ => Err(format!(
            "requires CUDA 12.8/13.0/13.2 and explicitly present GEMM_BI_QUAL_STATE_CAP=16/64; got {toolkit:?}/{cap:?}"
        )),
    }
}

pub(super) fn raw_corpus(
    storage: &str,
    shape: (usize, usize),
    exceptional: bool,
    seed: u32,
) -> Result<Vec<u8>, String> {
    let width = storage_width(storage)?;
    let count = shape
        .0
        .checked_mul(shape.1)
        .filter(|&n| n > 0)
        .ok_or("invalid corpus shape")?;
    let mut bytes = Vec::with_capacity(count.checked_mul(width).ok_or("corpus byte overflow")?);
    let mut state = seed | 1;
    for index in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Full mantissas and alternating signs expose rounding and cancellation.
        let word = match storage {
            "f32" => {
                ((index as u32 & 1) << 31) | ((119 + (state >> 24) % 16) << 23) | (state & 0x7fffff)
            }
            "bf16" => {
                ((index as u32 & 1) << 15) | ((119 + (state >> 24) % 16) << 7) | (state & 0x7f)
            }
            "f16" => ((index as u32 & 1) << 15) | ((9 + (state >> 24) % 8) << 10) | (state & 0x3ff),
            _ => unreachable!(),
        };
        bytes.extend_from_slice(&word.to_le_bytes()[..width]);
    }
    let boundary = match storage {
        "f32" => [0x3f7fffff, 0x3f800001, 0x33800001, 0xb3800001],
        "bf16" => [0x3f7f, 0x3f81, 0x3381, 0xb381],
        _ => [0x3bff, 0x3c01, 0x1401, 0x9401],
    };
    for (i, word) in boundary.into_iter().enumerate().take(count) {
        bytes[i * width..(i + 1) * width].copy_from_slice(&u32::to_le_bytes(word)[..width]);
    }
    if exceptional {
        let special: [u32; 10] = match storage {
            "f32" => [
                0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
                0x7f812345, 0xff854321,
            ],
            "bf16" => [
                0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc5, 0xffe3, 0x7f85, 0xffa3,
            ],
            _ => [
                0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7e45, 0xfe23, 0x7c45, 0xfc23,
            ],
        };
        // Only boundary rows and columns carry exceptional words, preserving finite interiors.
        let rows = [0, shape.0 - 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256];
        let columns = [0, shape.1 - 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256];
        let mut positions = BTreeSet::new();
        for row in rows.into_iter().filter(|&row| row < shape.0) {
            for col in columns.into_iter().filter(|&col| col < shape.1) {
                positions.insert(row * shape.1 + col);
            }
        }
        for (i, index) in positions.into_iter().enumerate() {
            bytes[index * width..(index + 1) * width]
                .copy_from_slice(&special[i % special.len()].to_le_bytes()[..width]);
        }
    }
    Ok(bytes)
}

pub(super) fn storage_width(storage: &str) -> Result<usize, String> {
    match storage {
        "f32" => Ok(4),
        "f16" | "bf16" => Ok(2),
        _ => Err(format!("unknown storage {storage}")),
    }
}

pub(super) fn compare_words(expected: &[u8], actual: &[u8], width: usize) -> Result<(), String> {
    if !matches!(width, 2 | 4)
        || expected.len() != actual.len()
        || !expected.len().is_multiple_of(width)
    {
        return Err(format!(
            "invalid word comparison lengths {}/{} width={width}",
            expected.len(),
            actual.len()
        ));
    }
    let mut first = None;
    let mut mismatches = 0;
    for (index, (a, b)) in expected
        .chunks_exact(width)
        .zip(actual.chunks_exact(width))
        .enumerate()
    {
        if a != b {
            first.get_or_insert(index);
            mismatches += 1;
        }
    }
    match first {
        None => Ok(()),
        Some(index) => Err(format!(
            "first_word={index} first_byte={} mismatches={mismatches} expected={:02x?} actual={:02x?}",
            index * width,
            &expected[index * width..(index + 1) * width],
            &actual[index * width..(index + 1) * width]
        )),
    }
}

pub(super) fn write_reference(path: &Path, metadata: &str, words: &[u8]) -> Result<(), String> {
    let mut frame = b"WWGEMM01".to_vec();
    frame.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    frame.extend_from_slice(&(words.len() as u64).to_le_bytes());
    frame.extend_from_slice(metadata.as_bytes());
    frame.extend_from_slice(words);
    let digest = Sha256::digest(&frame);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create-new {}: {e}", path.display()))?;
    file.write_all(&frame)
        .and_then(|()| file.write_all(&digest))
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

pub(super) fn read_reference(path: &Path) -> Result<(String, Vec<u8>), String> {
    let mut frame = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut file| file.read_to_end(&mut frame))
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if frame.len() < 56 || &frame[..8] != b"WWGEMM01" {
        return Err("invalid reference header".into());
    }
    let body_end = frame.len() - 32;
    if Sha256::digest(&frame[..body_end])[..] != frame[body_end..] {
        return Err("reference SHA256 mismatch".into());
    }
    let metadata_len = usize::try_from(u64::from_le_bytes(frame[8..16].try_into().unwrap()))
        .map_err(|_| "metadata length overflow")?;
    let words_len = usize::try_from(u64::from_le_bytes(frame[16..24].try_into().unwrap()))
        .map_err(|_| "word length overflow")?;
    let metadata_end = 24usize
        .checked_add(metadata_len)
        .ok_or("metadata length overflow")?;
    if metadata_end.checked_add(words_len) != Some(body_end) {
        return Err("reference frame length mismatch".into());
    }
    let metadata = std::str::from_utf8(&frame[24..metadata_end])
        .map_err(|e| format!("reference metadata UTF8: {e}"))?
        .to_owned();
    Ok((metadata, frame[metadata_end..body_end].to_vec()))
}

pub(super) fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn snapshot_files(path: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.file_type().map_err(|e| e.to_string())?.is_symlink() {
                return Err("source snapshot rejects symlinks".into());
            }
            snapshot_files(&entry.path(), files)?;
        }
    } else {
        files.push(path.to_owned());
    }
    Ok(())
}

fn source_snapshot() -> Result<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for path in ["src", "kernels", "Cargo.toml", "Cargo.lock"] {
        snapshot_files(&root.join(path), &mut files)?;
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy();
        let bytes =
            std::fs::read(&path).map_err(|e| format!("snapshot {}: {e}", path.display()))?;
        hash.update((relative.len() as u64).to_le_bytes());
        hash.update(relative.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn required_env(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|e| format!("{key} is required: {e}"))
}

pub(super) struct Runtime {
    pub ctx: GpuCtx,
    pub device: GpuDevice,
    pub quiet: QuietGpu,
    pub metadata: Value,
    source_digest: String,
}

impl Runtime {
    pub(super) fn new(released: bool) -> Result<Self, String> {
        for (key, _) in std::env::vars_os() {
            let key = key.to_string_lossy();
            if key == "NVIDIA_TF32_OVERRIDE"
                || key == "MAMBA_RS_ARCH_RUNG"
                || (key.starts_with("MAMBA")
                    && (key.contains("FORCE") || key.contains("CANDIDATE")))
            {
                return Err(format!(
                    "stale force/candidate control {key} is forbidden in released/current AUTO acceptance"
                ));
            }
        }
        let cap = required_env("GEMM_BI_QUAL_STATE_CAP")?;
        if !matches!(cap.as_str(), "16" | "64") {
            return Err("state capacity must be explicitly 16 or 64".into());
        }
        let base = required_env("COMBINED_GEMM_BASE_SHA")?;
        if base.len() != 40
            || !base.bytes().all(|c| c.is_ascii_hexdigit())
            || (released && base != RELEASED_SHA)
        {
            return Err(
                "invalid source base commit; released export requires exact peeled v0.7.0".into(),
            );
        }
        let source_digest = source_snapshot()?;
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        // Final acceptance cannot coexist with another owner, even when memory is available.
        let preflight = quiet.require_pre_context("combined-gemm/pre-context")?;
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err("combined acceptance requires the 142-SM CC8.9 device".into());
        }
        let ctx = GpuCtx::new_with_state_cap(
            &device,
            cap.parse().map_err(|e| format!("state capacity: {e}"))?,
        )?;
        let compiler = ctx.kernels.compiler_identity();
        let toolkit = format!("{}.{}", compiler.nvrtc_version.0, compiler.nvrtc_version.1);
        let cohort = explicit_cohort(&toolkit, Some(&cap))?;
        if ctx.state_cap().to_string() != cap || !compiler.nvrtc_library_known {
            return Err("context capacity or NVRTC library identity is not exact".into());
        }
        let rustc = std::process::Command::new("rustc")
            .arg("--version")
            .arg("--verbose")
            .output()
            .map_err(|e| format!("rustc identity: {e}"))?;
        if !rustc.status.success() {
            return Err("rustc identity command failed".into());
        }
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let cases = inventory()?;
        let schema = cases
            .iter()
            .map(|case| format!("{}:{}:{}", case.id, case.key(), case.required))
            .collect::<Vec<_>>()
            .join("\n");
        let metadata = json!({"schema":"combined-gemm-words.v1", "base_commit":base, "source_state":"snapshot", "production_source_snapshot_sha256":source_digest,
            "released":released, "package_version":env!("CARGO_PKG_VERSION"), "cohort":cohort, "state_capacity":ctx.state_cap(), "toolkit":toolkit,
            "shared_payload_sha256":sha(include_bytes!("combined_gemm_words.rs")), "harness_source_sha256":sha(super::HARNESS_SOURCE.as_bytes()),
            "released_overlay_sha256":if released {Some(sha(super::HARNESS_SOURCE.as_bytes()))}else{None},
            "case_schema_sha256":sha(schema.as_bytes()), "lock_sha256":sha(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock")).map_err(|e| e.to_string())?),
            "executable_sha256":sha(&std::fs::read(&executable).map_err(|e| e.to_string())?), "executable":executable,
            "rustc":String::from_utf8(rustc.stdout).map_err(|e| e.to_string())?, "device":format!("{:?}",device.identity()), "gpu_uuid":quiet.uuid, "preflight":preflight,
            "fixed_compiler":format!("{compiler:?}"), "artifact_set":format!("{:?}",ctx.kernels.artifact_set_identity()), "logical_cases":99, "required_cases":14, "additional_cases":85});
        Ok(Self {
            ctx,
            device,
            quiet,
            metadata,
            source_digest,
        })
    }

    pub(super) fn finish(&self) -> Result<(), String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("final sync: {e:?}"))?;
        self.quiet.verify_post_cohort("combined-gemm/post-cohort")?;
        if source_snapshot()? != self.source_digest {
            return Err("production source drifted during acceptance".into());
        }
        if self.metadata["device"] != format!("{:?}", self.device.identity()) {
            return Err("device identity drifted during acceptance".into());
        }
        Ok(())
    }
}

pub(super) fn dtype(storage: &str) -> Result<WeightDtype, String> {
    match storage {
        "f32" => Ok(WeightDtype::F32),
        "bf16" => Ok(WeightDtype::Bf16),
        "f16" => Ok(WeightDtype::F16),
        _ => Err(format!("invalid storage {storage}")),
    }
}

pub(super) fn configure(ctx: &GpuCtx, case: &AcceptanceCase) -> Result<(), String> {
    ctx.set_gemm_mode(GemmMode::Deterministic)?;
    ctx.set_bi_gemm_family(if case.family == "inference" {
        BiGemmFamily::Inference
    } else {
        BiGemmFamily::Triad
    });
    ctx.set_bi_tensor_cores(case.storage[0] != "f32");
    ctx.set_half_triad_policy(HalfTriadPolicy::TiledParityV1);
    ctx.set_f32_triad_policy(if case.row == "tf32" {
        F32TriadPolicy::AllowDeterministicTf32V1
    } else {
        F32TriadPolicy::ExactScalarFmaV1
    });
    Ok(())
}

enum Managed {
    F32(GpuBuffer),
    Half(DtypedBuf),
}

pub(super) struct Guarded {
    owner: Managed,
    pub pointer: u64,
    pub bytes: Vec<u8>,
    start: usize,
    active: usize,
    width: usize,
}

impl Guarded {
    fn new(ctx: &GpuCtx, storage: &str, payload: Vec<u8>, prefix: usize) -> Result<Self, String> {
        let width = storage_width(storage)?;
        if !prefix.is_multiple_of(16) || !payload.len().is_multiple_of(width) {
            return Err("guarded storage lost legal alignment".into());
        }
        let mut bytes = vec![0xa5; prefix + payload.len() + GUARD_BYTES];
        bytes[prefix..prefix + payload.len()].copy_from_slice(&payload);
        let owner = if storage == "f32" {
            Managed::F32(GpuBuffer::zeros(&ctx.stream, bytes.len() / width)?)
        } else {
            Managed::Half(DtypedBuf::zeros(
                &ctx.stream,
                bytes.len() / width,
                dtype(storage)?,
            )?)
        };
        let base = match &owner {
            Managed::F32(buf) => buf.cached_ptr(),
            Managed::Half(buf) => buf.cached_ptr(),
        };
        Ok(Self {
            owner,
            pointer: base + prefix as u64,
            bytes,
            start: prefix,
            active: payload.len(),
            width,
        })
    }

    fn base(&self) -> u64 {
        self.pointer - self.start as u64
    }
    pub(super) fn payload(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.active]
    }
    fn f32(&self) -> Result<&GpuBuffer, String> {
        match &self.owner {
            Managed::F32(buf) if self.start == 0 => Ok(buf),
            _ => Err("buffer-taking API requires owned F32 origin0".into()),
        }
    }
    fn f32_mut(&mut self) -> Result<&mut GpuBuffer, String> {
        match &mut self.owner {
            Managed::F32(buf) if self.start == 0 => Ok(buf),
            _ => Err("buffer-taking API requires owned F32 origin0".into()),
        }
    }
    fn reset(&self, ctx: &GpuCtx, complement: bool) -> Result<(), String> {
        let mut bytes = self.bytes.clone();
        if complement {
            for byte in &mut bytes[self.start..self.start + self.active] {
                *byte = !*byte;
            }
        }
        upload(ctx, self.base(), &bytes)?;
        compare_words(
            &bytes,
            &download(ctx, self.base(), bytes.len())?,
            self.width,
        )
        .map_err(|e| format!("reset upload verification: {e}"))
    }
    fn verify(&self, ctx: &GpuCtx, output: bool) -> Result<Vec<u8>, String> {
        let actual = download(ctx, self.base(), self.bytes.len())?;
        if output {
            compare_words(&self.bytes[..self.start], &actual[..self.start], self.width)?;
            compare_words(
                &self.bytes[self.start + self.active..],
                &actual[self.start + self.active..],
                self.width,
            )?;
        } else {
            compare_words(&self.bytes, &actual, self.width)?;
        }
        Ok(actual[self.start..self.start + self.active].to_vec())
    }
}

pub(super) fn upload(ctx: &GpuCtx, pointer: u64, bytes: &[u8]) -> Result<(), String> {
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            pointer,
            bytes.as_ptr().cast(),
            bytes.len(),
            ctx.stream.cu_stream(),
        )
    };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("raw upload: {result:?}"));
    }
    // The host allocation must outlive its asynchronous copy.
    ctx.stream
        .synchronize()
        .map_err(|e| format!("upload sync: {e:?}"))
}

pub(super) fn download(ctx: &GpuCtx, pointer: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; len];
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            bytes.as_mut_ptr().cast(),
            pointer,
            len,
            ctx.stream.cu_stream(),
        )
    };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("raw download: {result:?}"));
    }
    ctx.stream
        .synchronize()
        .map_err(|e| format!("download sync: {e:?}"))?;
    Ok(bytes)
}

pub(super) struct Fixture {
    pub case: AcceptanceCase,
    pub a: Guarded,
    pub b: Guarded,
    pub c: Guarded,
    pub bias: Option<Guarded>,
    pub scratch: u64,
    pub scratch_len: usize,
    splitk: u64,
    splitk_len: usize,
    exceptional: bool,
}

impl Fixture {
    pub(super) fn new(
        ctx: &GpuCtx,
        case: &AcceptanceCase,
        exceptional: bool,
        prefix: usize,
    ) -> Result<Self, String> {
        let (m, k, n) = case.dims;
        let a_shape = if case.op == "nt" { (m, n) } else { (m, k) };
        let b_shape = if case.op == "tn" { (m, n) } else { (k, n) };
        let c_shape = match case.op {
            "nt" => (m, k),
            "tn" => (k, n),
            _ => (m, n),
        };
        let a_prefix = if case.storage[0] == "f32" && case.op != "nn" {
            0
        } else {
            prefix
        };
        let b_prefix = if case.storage[1] == "f32" && case.op == "tn" {
            0
        } else {
            prefix
        };
        let c_prefix = if case.storage[2] == "f32" && case.op == "nt" {
            0
        } else {
            prefix
        };
        let a = Guarded::new(
            ctx,
            case.storage[0],
            raw_corpus(case.storage[0], a_shape, exceptional, 0x18374625)?,
            a_prefix,
        )?;
        let b = Guarded::new(
            ctx,
            case.storage[1],
            raw_corpus(case.storage[1], b_shape, exceptional, 0x53627189)?,
            b_prefix,
        )?;
        let initial = if case.op == "tn" {
            raw_corpus(case.storage[2], c_shape, false, 0x71425369)?
        } else {
            let word: u32 = match case.storage[2] {
                "f32" => 0x7fc0dead,
                "bf16" => 0x7fc5,
                _ => 0x7e55,
            };
            word.to_le_bytes()[..storage_width(case.storage[2])?].repeat(c_shape.0 * c_shape.1)
        };
        let c = Guarded::new(ctx, case.storage[2], initial, c_prefix)?;
        let bias = if case.bias {
            Some(Guarded::new(
                ctx,
                "f32",
                raw_corpus("f32", (1, n), exceptional, 0x97863521)?,
                prefix,
            )?)
        } else {
            None
        };
        let scratch_buf = ctx.kernels.transpose_scratch_buf(&ctx.stream)?;
        let scratch = scratch_buf.device_ptr(&ctx.stream).0;
        let scratch_len = scratch_buf.len();
        let splitk_buf = ctx.kernels.splitk_scratch_buf(&ctx.stream)?;
        let splitk = splitk_buf.device_ptr(&ctx.stream).0;
        let splitk_len = splitk_buf.len();
        Ok(Self {
            case: case.clone(),
            a,
            b,
            c,
            bias,
            scratch,
            scratch_len,
            splitk,
            splitk_len,
            exceptional,
        })
    }

    pub(super) fn pointers(&self) -> [u64; 4] {
        [
            self.c.pointer,
            self.a.pointer,
            self.b.pointer,
            self.bias.as_ref().map_or(0, |b| b.pointer),
        ]
    }

    pub(super) fn reset(&self, ctx: &GpuCtx, run: usize) -> Result<(), String> {
        self.a.reset(ctx, false)?;
        self.b.reset(ctx, false)?;
        self.c.reset(ctx, self.case.op != "tn" && run % 2 == 1)?;
        if let Some(bias) = &self.bias {
            bias.reset(ctx, false)?;
        }
        let scratch_buf = ctx.kernels.transpose_scratch_buf(&ctx.stream)?;
        if scratch_buf.len() != self.scratch_len
            || scratch_buf.device_ptr(&ctx.stream).0 != self.scratch
        {
            return Err("transpose scratch changed address or capacity".into());
        }
        upload(ctx, self.scratch, &vec![0xa6; self.scratch_len * 4])?;
        upload(ctx, self.splitk, &vec![0xb9; self.splitk_len * 4])?;
        Ok(())
    }

    pub(super) fn launch(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        let (m, k, n) = self.case.dims;
        let [c, a, b, bias] = self.pointers();
        let typed = |ptr, storage| {
            Ok::<_, String>(TypedPtr {
                ptr,
                dtype: dtype(storage)?,
            })
        };
        match self.case.op {
            "nn" => gpu_gemm_typed_forward_raw(
                ctx,
                typed(c, self.case.storage[2])?,
                typed(a, self.case.storage[0])?,
                typed(b, self.case.storage[1])?,
                if bias == 0 { None } else { Some(bias) },
                (m, k, n),
            ),
            "tn" => {
                let grad = GradSlice::from_offset(self.c.base(), self.c.start / 4, k * n);
                if self.case.storage[0] == "f32" {
                    gpu_gemm_bi_backward_dw_grad(ctx, &grad, self.b.f32()?, self.a.f32()?, m, k, n)
                } else {
                    gpu_gemm_bi_backward_dw_grad_typed(
                        ctx,
                        &grad,
                        typed(b, self.case.storage[1])?,
                        typed(a, self.case.storage[0])?,
                        m,
                        k,
                        n,
                    )
                }
            }
            "nt" => {
                if self.case.storage[0] == "f32" {
                    gpu_gemm_bi_backward_dx_raw(ctx, self.c.f32_mut()?, self.a.f32()?, b, m, k, n)
                } else {
                    gpu_gemm_ex_backward_dx_typed(
                        ctx,
                        typed(c, self.case.storage[2])?,
                        typed(a, self.case.storage[0])?,
                        typed(b, self.case.storage[1])?,
                        m,
                        k,
                        n,
                    )
                }
            }
            _ => Err("invalid public launch operation".into()),
        }
    }

    pub(super) fn verify(&self, ctx: &GpuCtx) -> Result<Vec<u8>, String> {
        self.a
            .verify(ctx, false)
            .map_err(|e| format!("A preservation: {e}"))?;
        self.b
            .verify(ctx, false)
            .map_err(|e| format!("B preservation: {e}"))?;
        if let Some(bias) = &self.bias {
            bias.verify(ctx, false)
                .map_err(|e| format!("bias preservation: {e}"))?;
        }
        let output = self
            .c
            .verify(ctx, true)
            .map_err(|e| format!("C red zones: {e}"))?;
        let finite = output
            .chunks_exact(self.c.width)
            .filter(|bytes| {
                let word = if self.c.width == 4 {
                    u32::from_le_bytes((*bytes).try_into().unwrap())
                } else {
                    u16::from_le_bytes((*bytes).try_into().unwrap()) as u32
                };
                let mask = match self.case.storage[2] {
                    "f32" => 0x7f800000,
                    "bf16" => 0x7f80,
                    _ => 0x7c00,
                };
                word & mask != mask
            })
            .count();
        let words = output.len() / self.c.width;
        if (!self.exceptional && finite != words) || (self.exceptional && finite < words / 2) {
            return Err(format!(
                "finite output coverage {finite}/{words} is insufficient; poison or all-NaN output cannot be accepted"
            ));
        }
        Ok(output)
    }

    pub(super) fn verify_transpose(&self, ctx: &GpuCtx) -> Result<(), String> {
        let (_, k, n) = self.case.dims;
        if self.scratch_len != 12_582_912 {
            return Err("unexpected exact NT scratch capacity".into());
        }
        let expected = transpose_expectation(self.b.payload(), (k, n), self.scratch_len)?;
        compare_words(
            &expected,
            &download(ctx, self.scratch, self.scratch_len * 4)?,
            4,
        )
        .map_err(|e| format!("full transpose scratch {} words: {e}", self.scratch_len))
    }

    fn logical_metadata(&self, exceptional: bool) -> Value {
        json!({"case_id":self.case.id,"case_key":self.case.key(),"family":self.case.family,"operation":self.case.op,"storage":self.case.storage,"dimensions_mkn":self.case.dims,"strides":self.case.strides(),"row":self.case.row,"mode":"Deterministic","f32_policy":if self.case.row=="tf32" {"AllowDeterministicTf32V1"}else{"ExactScalarFmaV1"},"half_policy":"TiledParityV1","tensor_cores":self.case.storage[0]!="f32","alpha_bits":0x3f800000_u32,"beta_bits":if self.case.op=="tn" {0x3f800000_u32}else{0},"bias":self.case.bias,"corpus":if exceptional {"sparse-exceptional.v1"}else{"full-mantissa-finite.v1"},"input_sha256":[sha(self.a.payload()),sha(self.b.payload())],"initial_c_sha256":sha(self.c.payload()),"bias_sha256":self.bias.as_ref().map(|b|sha(b.payload())),"output_words":self.c.active/self.c.width,"word_width":self.c.width,"encoding":if self.c.width==2 {"original-u16-le"}else{"raw-u32-le"}})
    }
}

fn transpose_expectation(
    input: &[u8],
    shape: (usize, usize),
    capacity: usize,
) -> Result<Vec<u8>, String> {
    let (k, n) = shape;
    let active = k.checked_mul(n).ok_or("transpose shape overflow")?;
    if active > capacity || active.checked_mul(4) != Some(input.len()) {
        return Err("transpose input/capacity mismatch".into());
    }
    let mut expected = vec![
        0xa6;
        capacity
            .checked_mul(4)
            .ok_or("transpose capacity overflow")?
    ];
    for ki in 0..k {
        for ni in 0..n {
            expected[(ni * k + ki) * 4..(ni * k + ki + 1) * 4]
                .copy_from_slice(&input[(ki * n + ni) * 4..(ki * n + ni + 1) * 4]);
        }
    }
    Ok(expected)
}

pub(super) trait Observer {
    fn prepare(&mut self, _runtime: &Runtime, _fixture: &Fixture) -> Result<(), String> {
        Ok(())
    }
    fn eager(&mut self, ctx: &GpuCtx, fixture: &mut Fixture) -> Result<(), String> {
        fixture.launch(ctx)
    }
    fn graph(
        &mut self,
        _runtime: &Runtime,
        _fixture: &Fixture,
        _graph: &CudaGraph,
    ) -> Result<(), String> {
        Ok(())
    }
    fn after(&mut self, _ctx: &GpuCtx, _fixture: &Fixture) -> Result<(), String> {
        Ok(())
    }
}

pub(super) fn run_words(
    runtime: &Runtime,
    released: bool,
    observer: &mut impl Observer,
) -> Result<(), String> {
    let directory = std::path::PathBuf::from(required_env("COMBINED_GEMM_REFERENCE_DIR")?);
    if !directory.is_dir() {
        return Err("COMBINED_GEMM_REFERENCE_DIR must already exist".into());
    }
    let cohort = runtime.metadata["cohort"]
        .as_str()
        .ok_or("missing cohort")?;
    let released_completion = directory.join(format!("{cohort}.released-complete.receipt"));
    let mut expected_files = serde_json::Map::new();
    if !released {
        let (metadata, payload) = read_reference(&released_completion)?;
        let completion: Value = serde_json::from_str(&metadata)
            .map_err(|e| format!("released completion metadata: {e}"))?;
        if !payload.is_empty()
            || completion["reference_records"] != 198
            || completion["case_cohort_keys"] != 99
            || completion["identity"]["released"] != true
            || completion["identity"]["base_commit"] != RELEASED_SHA
            || completion["identity"]["cohort"] != cohort
        {
            return Err("released cohort has no complete exact AUTO replay/word receipt".into());
        }
        expected_files = completion["reference_files"]
            .as_object()
            .ok_or("released completion has no reference-file hashes")?
            .clone();
        if expected_files.len() != 198 {
            return Err("released completion reference-file count mismatch".into());
        }
    }
    let mut reference_files = serde_json::Map::new();
    let mut records = 0;
    for case in inventory()? {
        configure(&runtime.ctx, &case)?;
        runtime.quiet.require_cohort(&case.id)?;
        for exceptional in [false, true] {
            let path = directory.join(format!(
                "{cohort}.{}.{}.words",
                case.id,
                if exceptional { "exceptional" } else { "finite" }
            ));
            if released && path.exists() {
                return Err(format!("reference already exists: {}", path.display()));
            }
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("reference filename UTF8")?
                .to_owned();
            if !released
                && expected_files.get(&filename)
                    != Some(&Value::String(sha(
                        &std::fs::read(&path).map_err(|e| format!("reference file: {e}"))?
                    )))
            {
                return Err(format!("released completed reference changed: {filename}"));
            }
            let mut reference = None;
            for prefix in [0, 64] {
                let mut fixture = Fixture::new(&runtime.ctx, &case, exceptional, prefix)?;
                let logical = fixture.logical_metadata(exceptional);
                observer.prepare(runtime, &fixture)?;
                fixture.reset(&runtime.ctx, 0)?;
                observer.eager(&runtime.ctx, &mut fixture)?;
                let actual = fixture.verify(&runtime.ctx)?;
                if !released && case.family == "triad" && case.op == "nt" && case.row == "f32_exact"
                {
                    fixture.verify_transpose(&runtime.ctx)?;
                }
                observer.after(&runtime.ctx, &fixture)?;
                if reference.is_none() {
                    if released {
                        let metadata = json!({"identity":runtime.metadata,"logical":logical,"reference_run":"independently-reset-eager","required_runs":["eager","replay1","replay2"],"required_prefix_bytes":[0,64],"words_sha256":sha(&actual)});
                        write_reference(&path, &metadata.to_string(), &actual)?;
                    }
                    let (text, expected) = read_reference(&path)?;
                    let metadata: Value = serde_json::from_str(&text)
                        .map_err(|e| format!("reference metadata: {e}"))?;
                    if metadata["logical"] != logical
                        || metadata["identity"]["released"] != true
                        || metadata["identity"]["base_commit"] != RELEASED_SHA
                        || metadata["words_sha256"] != sha(&expected)
                    {
                        return Err(format!(
                            "reference logical or released identity mismatch {}",
                            path.display()
                        ));
                    }
                    for field in ["cohort", "case_schema_sha256", "shared_payload_sha256"] {
                        if metadata["identity"][field] != runtime.metadata[field] {
                            return Err(format!("reference {field} mismatch"));
                        }
                    }
                    reference = Some(expected);
                }
                let expected = reference.as_ref().ok_or("reference not initialized")?;
                compare_words(expected, &actual, fixture.c.width)
                    .map_err(|e| format!("{} eager prefix={prefix}: {e}", case.id))?;
                fixture.reset(&runtime.ctx, 1)?;
                let graph = unsafe {
                    capture_into_graph(&runtime.ctx.stream, || fixture.launch(&runtime.ctx))
                }?;
                observer.graph(runtime, &fixture, &graph)?;
                for replay in [1, 2] {
                    fixture.reset(&runtime.ctx, replay)?;
                    graph.launch().map_err(|e| format!("replay: {e:?}"))?;
                    let actual = fixture.verify(&runtime.ctx)?;
                    if !released
                        && case.family == "triad"
                        && case.op == "nt"
                        && case.row == "f32_exact"
                    {
                        fixture.verify_transpose(&runtime.ctx)?;
                    }
                    observer.after(&runtime.ctx, &fixture)?;
                    compare_words(expected, &actual, fixture.c.width)
                        .map_err(|e| format!("{} replay={replay} prefix={prefix}: {e}", case.id))?;
                }
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|e| format!("fixture lifetime sync: {e:?}"))?;
            }
            records += 1;
            reference_files.insert(
                filename,
                Value::String(sha(&std::fs::read(&path).map_err(|e| e.to_string())?)),
            );
            eprintln!(
                "{}",
                json!({"record":"raw-word-pass","cohort":cohort,"case":case.id,"exceptional":exceptional,"independent_runs":6,"reference":path,"reference_file_sha256":sha(&std::fs::read(&path).map_err(|e|e.to_string())?)})
            );
        }
    }
    if records != 198 {
        return Err(format!("incomplete raw word inventory {records}/198"));
    }
    runtime.finish()?;
    let completion = json!({"record":"cohort-complete","identity":runtime.metadata,"reference_records":records,"case_cohort_keys":99,"independent_runs_per_case_corpus":6,"reference_files":reference_files,"campaign_case_cohort_keys_required":594,"six_cohort_campaign_complete":false});
    let completion_path = directory.join(format!(
        "{cohort}.{}-complete.receipt",
        if released { "released" } else { "current" }
    ));
    write_reference(&completion_path, &completion.to_string(), &[])?;
    eprintln!("{completion}");
    Ok(())
}

#[cfg(test)]
mod combined_gemm_host {
    use super::*;

    #[test]
    fn transpose_expectation_keeps_every_active_raw_word_and_inactive_suffix() {
        let input = [0_u32, 0x80000000, 0x7fc12345, 0x7f812345, 1, 0xffc54321]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let actual = transpose_expectation(&input, (2, 3), 8).unwrap();
        let expected = [
            0_u32, 0x7f812345, 0x80000000, 1, 0x7fc12345, 0xffc54321, 0xa6a6a6a6, 0xa6a6a6a6,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(transpose_expectation(&input, (2, 3), 5).is_err());
        assert!(transpose_expectation(&input[..20], (2, 3), 8).is_err());
    }

    #[test]
    fn closed_inventory_retains_all_controls_and_required_fourteen() {
        let cases = inventory().unwrap();
        assert_eq!(cases.len(), 99);
        assert_eq!(cases.iter().filter(|case| case.required).count(), 14);
        assert_eq!(cases.iter().filter(|case| !case.required).count(), 85);
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.family == "inference")
                .count(),
            60
        );
        assert_eq!(cases.iter().filter(|case| case.op == "tn").count(), 9);
        for id in [
            "inference.nn.bf16_f32.hot_a.bias0",
            "inference.nn.f16_f32.hot_e.bias1",
            "inference.nn.f32_exact.hot_c.bias1",
            "inference.nn.tf32.hot_e.bias0",
            "triad.nt.tf32.large_deep.bias0",
            "triad.nt.f32_exact.large_deep.bias0",
            "triad.tn.bf16.small16.bias0",
            "triad.tn.f16.small16.bias0",
            "triad.tn.f32_exact.small16.bias0",
        ] {
            assert_eq!(cases.iter().filter(|case| case.id == id).count(), 1, "{id}");
        }
        validate_inventory(&cases).unwrap();
        let mut duplicate = cases.clone();
        duplicate[98] = duplicate[0].clone();
        assert!(validate_inventory(&duplicate).is_err());
        assert!(validate_inventory(&cases[..98]).is_err());
        let mut invalid = cases;
        invalid[0].dims.0 += 1;
        assert!(validate_inventory(&invalid).is_err());
    }

    #[test]
    fn cohort_requires_explicit_capacity_and_six_literal_keys() {
        let mut keys = Vec::new();
        for toolkit in ["12.8", "13.0", "13.2"] {
            for cap in ["16", "64"] {
                keys.push(explicit_cohort(toolkit, Some(cap)).unwrap());
            }
        }
        assert_eq!(
            keys,
            [
                "cuda12.8-cap16",
                "cuda12.8-cap64",
                "cuda13.0-cap16",
                "cuda13.0-cap64",
                "cuda13.2-cap16",
                "cuda13.2-cap64"
            ]
        );
        assert_eq!(keys.len() * 99, 594);
        for invalid in [None, Some(""), Some("016"), Some(" 64"), Some("32")] {
            assert!(explicit_cohort("12.8", invalid).is_err());
        }
        assert!(explicit_cohort("13.1", Some("64")).is_err());
    }

    #[test]
    fn finite_corpus_retains_low_mantissas_and_sparse_exceptional_words() {
        for (storage, width, exponent_mask, literals) in [
            (
                "f32",
                4,
                0x7f800000_u32,
                [
                    0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
                    0x7f812345, 0xff854321,
                ],
            ),
            (
                "bf16",
                2,
                0x7f80,
                [
                    0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc5, 0xffe3, 0x7f85, 0xffa3,
                ],
            ),
            (
                "f16",
                2,
                0x7c00,
                [
                    0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7e45, 0xfe23, 0x7c45, 0xfc23,
                ],
            ),
        ] {
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(width)
                    .map(|word| {
                        if width == 4 {
                            u32::from_le_bytes(word.try_into().unwrap())
                        } else {
                            u16::from_le_bytes(word.try_into().unwrap()) as u32
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let finite = decode(&raw_corpus(storage, (257, 257), false, 0x12345678).unwrap());
            assert_eq!(finite.len(), 66049);
            assert!(
                finite
                    .iter()
                    .all(|word| word & exponent_mask != exponent_mask)
            );
            assert!(finite.iter().any(|word| word & 1 != 0));
            assert!(
                finite
                    .iter()
                    .any(|word| word & (if width == 4 { 0x80000000 } else { 0x8000 }) != 0)
            );
            let exceptional = decode(&raw_corpus(storage, (257, 257), true, 0x12345678).unwrap());
            for literal in literals {
                assert!(exceptional.contains(&literal), "{storage} {literal:08x}");
            }
            assert!(
                exceptional
                    .iter()
                    .zip(&finite)
                    .filter(|(a, b)| a != b)
                    .count()
                    < 512
            );
            assert_eq!(exceptional[128 * 257 + 17], finite[128 * 257 + 17]);
        }
    }

    #[test]
    fn comparator_reports_every_raw_word_without_nan_or_zero_normalization() {
        let expected = [0_u32, 0x7fc12345, 0x7f812345, 0xffc54321]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let actual = [0x80000000_u32, 0x7fc12346, 0x7f812345, 0xffc54322]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        compare_words(&expected, &expected, 4).unwrap();
        let error = compare_words(&expected, &actual, 4).unwrap_err();
        assert!(error.contains("first_word=0"), "{error}");
        assert!(error.contains("mismatches=3"), "{error}");
        assert!(
            compare_words(&[0, 0, 0x45, 0x7e], &[0, 0x80, 0x46, 0x7e], 2)
                .unwrap_err()
                .contains("mismatches=2")
        );
        assert!(compare_words(&[0], &[0], 2).is_err());
        assert!(compare_words(&expected, &actual[..12], 4).is_err());
    }

    #[test]
    fn reference_is_hashed_create_new_and_rejects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.words");
        let metadata = "{\"schema\":\"combined-gemm-words.v1\",\"case\":\"fixture\"}";
        let words = [0x00, 0x80, 0x45, 0x7e, 0x01, 0xfc];
        write_reference(&path, metadata, &words).unwrap();
        let original = std::fs::read(&path).unwrap();
        assert_eq!(
            read_reference(&path).unwrap(),
            (metadata.to_owned(), words.to_vec())
        );
        assert!(write_reference(&path, "overwrite", &[1, 2]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        let damaged = dir.path().join("damaged.words");
        let mut bytes = original;
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&damaged, bytes).unwrap();
        assert!(read_reference(&damaged).is_err());
    }
}
