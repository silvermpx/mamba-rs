//! Stable compiler, artifact, policy, and device identities for CUDA graphs.

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::context::BiGemmFamily;

pub type Sha256Digest = [u8; 32];

pub fn digest_hex(value: &Sha256Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in value {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

const HASH_DOMAIN: &[u8] = b"mamba-rs.length-framed-sha256.v1";
const CACHE_MAGIC: [u8; 16] = *b"MAMBA-PTX-CACHE\0";
const CACHE_FORMAT_VERSION: u16 = 1;

pub const COMPOSER_REVISION: u16 = 1;
pub const COMPILER_REVISION: u16 = 1;
pub const NUMERIC_ABI_REVISION: u16 = 1;
pub const SCHEDULE_REVISION: u16 = 1;
pub const POLICY_REVISION: u16 = 1;

/// SHA-256 framing with explicit tags, presence, and byte lengths.
pub struct FramedSha256(Sha256);

impl FramedSha256 {
    pub fn new(domain: &[u8]) -> Self {
        let mut value = Self(Sha256::new());
        value = value.required(b"hash-domain", HASH_DOMAIN);
        value.required(b"application-domain", domain)
    }

    pub fn required(mut self, tag: &[u8], value: &[u8]) -> Self {
        self.write(tag, 1, value);
        self
    }

    pub fn optional(mut self, tag: &[u8], value: Option<&[u8]>) -> Self {
        match value {
            Some(value) => self.write(tag, 1, value),
            None => self.write(tag, 0, &[]),
        }
        self
    }

    pub fn finish(self) -> Sha256Digest {
        self.0.finalize().into()
    }

    pub fn bytes(value: &[u8]) -> Sha256Digest {
        Sha256::digest(value).into()
    }

    fn write(&mut self, tag: &[u8], presence: u8, value: &[u8]) {
        self.0.update((tag.len() as u64).to_le_bytes());
        self.0.update(tag);
        self.0.update([presence]);
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArtifactKind {
    Ptx = 1,
    Cubin = 2,
}

impl ArtifactKind {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Ptx),
            2 => Some(Self::Cubin),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileKeyMaterial {
    pub source: Vec<u8>,
    pub target: Vec<u8>,
    pub argv: Vec<Vec<u8>>,
    pub include_roots: Vec<Vec<u8>>,
    pub header_manifest: Option<Vec<u8>>,
    pub nvrtc_version: (i32, i32),
    pub nvrtc_library_domain: Option<Vec<u8>>,
    pub output_kind: ArtifactKind,
    pub composer_revision: u16,
    pub compiler_revision: u16,
    pub numeric_abi_revision: u16,
    pub schedule_revision: u16,
}

impl CompileKeyMaterial {
    /// Returns no key when the filesystem header closure is not known.
    pub fn digest(&self) -> Option<Sha256Digest> {
        self.header_manifest.as_ref()?;
        self.nvrtc_library_domain.as_ref()?;
        Some(self.invocation_digest())
    }

    pub fn invocation_digest(&self) -> Sha256Digest {
        let mut hash = FramedSha256::new(b"cuda-compile-key.v1")
            .required(b"source", &self.source)
            .required(b"target", &self.target)
            .required(b"output-kind", &[self.output_kind as u8])
            .optional(b"header-manifest", self.header_manifest.as_deref())
            .required(b"nvrtc-major", &self.nvrtc_version.0.to_le_bytes())
            .required(b"nvrtc-minor", &self.nvrtc_version.1.to_le_bytes())
            .optional(
                b"nvrtc-library-domain",
                self.nvrtc_library_domain.as_deref(),
            )
            .required(b"composer-revision", &self.composer_revision.to_le_bytes())
            .required(b"compiler-revision", &self.compiler_revision.to_le_bytes())
            .required(
                b"numeric-abi-revision",
                &self.numeric_abi_revision.to_le_bytes(),
            )
            .required(b"schedule-revision", &self.schedule_revision.to_le_bytes())
            .required(b"argv-count", &(self.argv.len() as u64).to_le_bytes());
        for value in &self.argv {
            hash = hash.required(b"argv", value);
        }
        hash = hash.required(
            b"include-root-count",
            &(self.include_roots.len() as u64).to_le_bytes(),
        );
        for value in &self.include_roots {
            hash = hash.required(b"include-root", value);
        }
        hash.finish()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CacheHit {
    pub artifact_digest: Sha256Digest,
    pub payload: Vec<u8>,
}

pub struct CacheEnvelope;

impl CacheEnvelope {
    pub fn encode(
        compile_key: Sha256Digest,
        artifact_kind: ArtifactKind,
        payload: &[u8],
    ) -> Vec<u8> {
        let artifact_digest = FramedSha256::bytes(payload);
        let mut bytes = Vec::with_capacity(91 + payload.len());
        bytes.extend_from_slice(&CACHE_MAGIC);
        bytes.extend_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&compile_key);
        bytes.push(artifact_kind as u8);
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&artifact_digest);
        bytes.extend_from_slice(payload);
        bytes
    }

    pub fn decode(
        expected_key: Sha256Digest,
        expected_kind: ArtifactKind,
        bytes: &[u8],
    ) -> Result<CacheHit, String> {
        const HEADER: usize = 16 + 2 + 32 + 1 + 8 + 32;
        if bytes.len() < HEADER {
            return Err("kernel cache entry is truncated".into());
        }
        if bytes[..16] != CACHE_MAGIC {
            return Err("kernel cache magic does not match".into());
        }
        let version = u16::from_le_bytes(bytes[16..18].try_into().unwrap());
        if version != CACHE_FORMAT_VERSION {
            return Err(format!("unsupported kernel cache format {version}"));
        }
        let key: Sha256Digest = bytes[18..50].try_into().unwrap();
        if key != expected_key {
            return Err("kernel cache compile key does not match".into());
        }
        let kind = ArtifactKind::from_byte(bytes[50])
            .ok_or_else(|| "kernel cache artifact kind is invalid".to_string())?;
        if kind != expected_kind {
            return Err("kernel cache artifact kind does not match".into());
        }
        let payload_len = u64::from_le_bytes(bytes[51..59].try_into().unwrap());
        let payload_len = usize::try_from(payload_len)
            .map_err(|_| "kernel cache payload length exceeds usize".to_string())?;
        if bytes.len() != HEADER.saturating_add(payload_len) {
            return Err("kernel cache payload length does not match".into());
        }
        let stored_digest: Sha256Digest = bytes[59..91].try_into().unwrap();
        let payload = &bytes[HEADER..];
        let artifact_digest = FramedSha256::bytes(payload);
        if artifact_digest != stored_digest {
            return Err("kernel cache artifact digest does not match".into());
        }
        Ok(CacheHit {
            artifact_digest,
            payload: payload.to_vec(),
        })
    }
}

pub fn canonical_ptx_image(image: &[u8]) -> Result<String, String> {
    if image.last() != Some(&0) {
        return Err("PTX image has no terminal NUL".into());
    }
    let payload = &image[..image.len() - 1];
    if payload.contains(&0) {
        return Err("PTX image contains an interior NUL".into());
    }
    String::from_utf8(payload.to_vec()).map_err(|_| "PTX image is not UTF-8".into())
}

pub(crate) fn canonical_ptx_from_string(source: String) -> Result<String, String> {
    let mut image = source.into_bytes();
    image.push(0);
    canonical_ptx_image(&image)
}

pub(crate) fn canonical_ptx_from_cache(payload: Vec<u8>) -> Result<String, String> {
    let mut image = payload;
    image.push(0);
    canonical_ptx_image(&image)
}

static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn read_cache(
    path: &Path,
    compile_key: Sha256Digest,
    artifact_kind: ArtifactKind,
) -> Option<CacheHit> {
    let bytes = std::fs::read(path).ok()?;
    match CacheEnvelope::decode(compile_key, artifact_kind, &bytes) {
        Ok(hit) => Some(hit),
        Err(_) => {
            let _ = std::fs::remove_file(path);
            None
        }
    }
}

pub(crate) fn publish_cache(
    path: &Path,
    compile_key: Sha256Digest,
    artifact_kind: ArtifactKind,
    payload: &[u8],
) {
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let encoded = CacheEnvelope::encode(compile_key, artifact_kind, payload);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    let counter = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    let tmp = parent.join(format!(
        ".{file_name}.tmp-{}-{stamp:x}-{counter:x}",
        std::process::id()
    ));
    let wrote = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|mut file| std::io::Write::write_all(&mut file, &encoded))
        .is_ok();
    if wrote {
        let _ = std::fs::rename(&tmp, path);
    }
    let _ = std::fs::remove_file(tmp);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CudaTarget {
    bytes: [u8; 16],
    len: u8,
}

impl CudaTarget {
    pub fn new(value: &str) -> Result<Self, String> {
        let raw = value.as_bytes();
        let len =
            u8::try_from(raw.len()).map_err(|_| format!("CUDA target {value:?} is too long"))?;
        if raw.len() > 16 || !raw.is_ascii() {
            return Err(format!("CUDA target {value:?} is not a short ASCII target"));
        }
        let mut bytes = [0; 16];
        bytes[..raw.len()].copy_from_slice(raw);
        Ok(Self { bytes, len })
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("CudaTarget contains ASCII bytes")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ModuleKind {
    LegacyCombined = 1,
    TriadScalar = 2,
    TriadSm80 = 3,
    TriadSm90a = 4,
    TriadSm100 = 5,
    TriadSm120 = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactIdentity {
    pub module_kind: ModuleKind,
    pub artifact_kind: ArtifactKind,
    pub compile_key: Sha256Digest,
    pub artifact_digest: Sha256Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactSetIdentity {
    pub module_count: u8,
    pub ordered_digest: Sha256Digest,
    pub legacy_combined: ArtifactIdentity,
}

pub fn build_artifact_set(artifacts: &[ArtifactIdentity]) -> Result<ArtifactSetIdentity, String> {
    let module_count = u8::try_from(artifacts.len())
        .map_err(|_| "artifact set contains more than 255 modules".to_string())?;
    let mut kinds = BTreeSet::new();
    let mut legacy = None;
    let mut hash =
        FramedSha256::new(b"cuda-artifact-set.v1").required(b"module-count", &[module_count]);
    for artifact in artifacts {
        if !kinds.insert(artifact.module_kind as u8) {
            return Err(format!(
                "artifact set contains duplicate module kind {:?}",
                artifact.module_kind
            ));
        }
        if artifact.module_kind == ModuleKind::LegacyCombined {
            legacy = Some(*artifact);
        }
        hash = hash
            .required(b"module-kind", &[artifact.module_kind as u8])
            .required(b"artifact-kind", &[artifact.artifact_kind as u8])
            .required(b"compile-key", &artifact.compile_key)
            .required(b"artifact-digest", &artifact.artifact_digest);
    }
    Ok(ArtifactSetIdentity {
        module_count,
        ordered_digest: hash.finish(),
        legacy_combined: legacy
            .ok_or_else(|| "artifact set has no legacy combined module".to_string())?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompilerIdentity {
    pub source_digest: Sha256Digest,
    pub invocation_digest: Sha256Digest,
    pub header_manifest_digest: Sha256Digest,
    pub target: CudaTarget,
    pub nvrtc_version: (i32, i32),
    pub nvrtc_library_domain: Sha256Digest,
    pub nvrtc_library_known: bool,
    pub output_kind: ArtifactKind,
    pub composer_revision: u16,
    pub compiler_revision: u16,
    pub numeric_abi_revision: u16,
    pub schedule_revision: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DriverIdentity {
    pub api_version: i32,
    pub build_sources: u8,
    pub build_digest: Sha256Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    pub compute_capability: (u32, u32),
    pub target: CudaTarget,
    pub driver: DriverIdentity,
}

pub(crate) fn query_driver_identity() -> Result<DriverIdentity, String> {
    static IDENTITY: OnceLock<Result<DriverIdentity, String>> = OnceLock::new();
    IDENTITY.get_or_init(query_driver_identity_uncached).clone()
}

fn query_driver_identity_uncached() -> Result<DriverIdentity, String> {
    let mut api_version = 0;
    let result = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut api_version) };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuDriverGetVersion failed: {result:?}"));
    }

    let mut sources = 0u8;
    let mut hash = FramedSha256::new(b"cuda-driver-build.v1");
    for (bit, path) in [
        (1u8, Path::new("/proc/driver/nvidia/version")),
        (2u8, Path::new("/sys/module/nvidia/version")),
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            sources |= bit;
            hash = hash
                .required(b"build-source", path.as_os_str().as_encoded_bytes())
                .required(b"build-bytes", &bytes);
        }
    }
    if let Some(domain) = loaded_library_domain("libcuda.so") {
        sources |= 4;
        hash = hash.required(b"loaded-libcuda", &domain);
    }
    Ok(DriverIdentity {
        api_version,
        build_sources: sources,
        build_digest: hash.finish(),
    })
}

pub(crate) fn nvrtc_library_domain() -> Option<Vec<u8>> {
    static DOMAIN: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    DOMAIN
        .get_or_init(|| loaded_library_domain("libnvrtc.so").map(|digest| digest.to_vec()))
        .clone()
}

fn loaded_library_domain(needle: &str) -> Option<Sha256Digest> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let path = maps.lines().find_map(|line| {
        let path = line.split_whitespace().last()?;
        (path.starts_with('/') && path.contains(needle)).then_some(path)
    })?;
    let canonical = std::fs::canonicalize(path).ok()?;
    let content_digest = sha256_file(&canonical).ok()?;
    Some(
        FramedSha256::new(b"loaded-library.v1")
            .required(b"path", canonical.as_os_str().as_encoded_bytes())
            .required(b"content-sha256", &content_digest)
            .finish(),
    )
}

fn sha256_file(path: &Path) -> std::io::Result<Sha256Digest> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hash.update(&buf[..read]);
    }
    Ok(hash.finalize().into())
}

/// Builds an over-approximation of all literal filesystem includes.
/// Macro or unresolved includes disable persistent caching.
pub(crate) fn header_manifest(source: &[u8], include_roots: &[String]) -> Option<Vec<u8>> {
    let roots: Vec<PathBuf> = include_roots.iter().map(PathBuf::from).collect();
    let mut pending = Vec::new();
    let mut records = BTreeMap::<String, Sha256Digest>::new();
    collect_includes(&String::from_utf8_lossy(source), None, &roots, &mut pending)?;

    while let Some(path) = pending.pop() {
        let canonical = std::fs::canonicalize(&path).ok()?;
        let key = manifest_path_key(&canonical, &roots);
        if records.contains_key(&key) {
            continue;
        }
        let bytes = std::fs::read(&canonical).ok()?;
        records.insert(key, FramedSha256::bytes(&bytes));
        collect_includes(
            &String::from_utf8_lossy(&bytes),
            canonical.parent(),
            &roots,
            &mut pending,
        )?;
    }

    let mut output = Vec::new();
    output.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (path, digest) in records {
        append_manifest_field(&mut output, path.as_bytes());
        append_manifest_field(&mut output, &digest);
    }
    Some(output)
}

fn append_manifest_field(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn manifest_path_key(path: &Path, roots: &[PathBuf]) -> String {
    for (index, root) in roots.iter().enumerate() {
        if let Ok(relative) = path.strip_prefix(root) {
            return format!("{index}:{}", relative.to_string_lossy());
        }
    }
    format!("external:{}", path.to_string_lossy())
}

fn collect_includes(
    text: &str,
    current_dir: Option<&Path>,
    roots: &[PathBuf],
    pending: &mut Vec<PathBuf>,
) -> Option<()> {
    for line in text.lines() {
        let Some(rest) = include_directive(line)? else {
            continue;
        };
        let (quoted, name) = if let Some(rest) = rest.strip_prefix('"') {
            (true, rest.split_once('"')?.0)
        } else if let Some(rest) = rest.strip_prefix('<') {
            (false, rest.split_once('>')?.0)
        } else {
            return None;
        };
        let mut candidates = Vec::new();
        if quoted && let Some(dir) = current_dir {
            candidates.push(dir.join(name));
        }
        candidates.extend(roots.iter().map(|root| root.join(name)));
        if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
            pending.push(path);
        } else if quoted {
            return None;
        } else {
            // NVRTC 13.2 cannot enumerate bundled headers, so this include
            // has no content identity suitable for persistent caching.
            return None;
        }
    }
    Some(())
}

fn include_directive(line: &str) -> Option<Option<&str>> {
    let line = line.trim_start();
    let Some(line) = line.strip_prefix('#') else {
        return Some(None);
    };
    let line = line.trim_start();
    let Some(rest) = line.strip_prefix("include") else {
        return Some(None);
    };
    if rest.starts_with(|value: char| value.is_ascii_alphanumeric() || value == '_') {
        return Some(None);
    }
    Some(Some(rest.trim_start()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PolicyOp {
    Dw = 1,
    Dx = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PolicyDtype {
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackwardPolicyCell {
    pub op: PolicyOp,
    pub dtype: PolicyDtype,
    pub dims: (usize, usize, usize),
}

const fn policy_cell(
    op: PolicyOp,
    dtype: PolicyDtype,
    dims: (usize, usize, usize),
) -> BackwardPolicyCell {
    BackwardPolicyCell { op, dtype, dims }
}

const LEGACY_BACKWARD_CELLS: [BackwardPolicyCell; 18] = [
    policy_cell(PolicyOp::Dw, PolicyDtype::Bf16, (1024, 8, 256)),
    policy_cell(PolicyOp::Dw, PolicyDtype::Bf16, (2048, 16, 512)),
    policy_cell(PolicyOp::Dw, PolicyDtype::Bf16, (2048, 48, 1536)),
    policy_cell(PolicyOp::Dw, PolicyDtype::Bf16, (1024, 256, 40)),
    policy_cell(PolicyOp::Dw, PolicyDtype::Bf16, (2048, 512, 48)),
    policy_cell(PolicyOp::Dx, PolicyDtype::Bf16, (1024, 8, 256)),
    policy_cell(PolicyOp::Dx, PolicyDtype::Bf16, (2048, 16, 512)),
    policy_cell(PolicyOp::Dx, PolicyDtype::Bf16, (2048, 48, 1536)),
    policy_cell(PolicyOp::Dx, PolicyDtype::Bf16, (32, 256, 1024)),
    policy_cell(PolicyOp::Dw, PolicyDtype::F16, (1024, 8, 256)),
    policy_cell(PolicyOp::Dw, PolicyDtype::F16, (2048, 16, 512)),
    policy_cell(PolicyOp::Dw, PolicyDtype::F16, (2048, 48, 1536)),
    policy_cell(PolicyOp::Dw, PolicyDtype::F16, (1024, 256, 40)),
    policy_cell(PolicyOp::Dw, PolicyDtype::F16, (2048, 512, 48)),
    policy_cell(PolicyOp::Dx, PolicyDtype::F16, (1024, 8, 256)),
    policy_cell(PolicyOp::Dx, PolicyDtype::F16, (2048, 16, 512)),
    policy_cell(PolicyOp::Dx, PolicyDtype::F16, (2048, 48, 1536)),
    policy_cell(PolicyOp::Dx, PolicyDtype::F16, (32, 256, 1024)),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LegacySm80Policy {
    pub reject_zero_axes: bool,
    pub square_tile_min: usize,
    pub large_tile_min: usize,
    pub tile128_prefer_min_tiles: u32,
    pub forward_min_columns: usize,
    pub forward_thin_max_rows: usize,
    pub forward_thin_below_square_columns: bool,
    pub backward_one_axis_tile64: bool,
    pub backward_two_small_fallback: bool,
    pub backward_cells: [BackwardPolicyCell; 18],
}

impl LegacySm80Policy {
    pub const fn current() -> Self {
        Self {
            reject_zero_axes: true,
            square_tile_min: 64,
            large_tile_min: 128,
            tile128_prefer_min_tiles: 72,
            forward_min_columns: 32,
            forward_thin_max_rows: 64,
            forward_thin_below_square_columns: true,
            backward_one_axis_tile64: true,
            backward_two_small_fallback: true,
            backward_cells: LEGACY_BACKWARD_CELLS,
        }
    }

    pub fn admits(&self, op: PolicyOp, dtype: PolicyDtype, dims: (usize, usize, usize)) -> bool {
        self.backward_cells
            .iter()
            .any(|cell| cell.op == op && cell.dtype == dtype && cell.dims == dims)
    }

    pub fn digest(&self) -> Sha256Digest {
        let mut hash = FramedSha256::new(b"legacy-sm80-policy.v1")
            .required(b"reject-zero-axes", &[self.reject_zero_axes as u8])
            .required(
                b"square-tile-min",
                &(self.square_tile_min as u64).to_le_bytes(),
            )
            .required(
                b"large-tile-min",
                &(self.large_tile_min as u64).to_le_bytes(),
            )
            .required(
                b"tile128-prefer-min-tiles",
                &self.tile128_prefer_min_tiles.to_le_bytes(),
            )
            .required(
                b"forward-min-columns",
                &(self.forward_min_columns as u64).to_le_bytes(),
            )
            .required(
                b"forward-thin-max-rows",
                &(self.forward_thin_max_rows as u64).to_le_bytes(),
            )
            .required(
                b"forward-thin-below-square-columns",
                &[self.forward_thin_below_square_columns as u8],
            )
            .required(
                b"backward-one-axis-tile64",
                &[self.backward_one_axis_tile64 as u8],
            )
            .required(
                b"backward-two-small-fallback",
                &[self.backward_two_small_fallback as u8],
            )
            .required(
                b"backward-cell-count",
                &(self.backward_cells.len() as u64).to_le_bytes(),
            );
        for cell in self.backward_cells {
            hash = hash
                .required(b"backward-op", &[cell.op as u8])
                .required(b"backward-dtype", &[cell.dtype as u8])
                .required(b"backward-m", &(cell.dims.0 as u64).to_le_bytes())
                .required(b"backward-k", &(cell.dims.1 as u64).to_le_bytes())
                .required(b"backward-n", &(cell.dims.2 as u64).to_le_bytes());
        }
        hash.finish()
    }
}

pub fn legacy_sm80_policy_digest() -> Sha256Digest {
    static DIGEST: OnceLock<Sha256Digest> = OnceLock::new();
    *DIGEST.get_or_init(|| LegacySm80Policy::current().digest())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmPolicy {
    pub batch_invariant: bool,
    pub bi_tensor_cores: bool,
    pub fast_gemm: bool,
    pub tf32: bool,
    pub bi_gemm_family: BiGemmFamily,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackendSet(u8);

impl BackendSet {
    pub const CUBLAS: Self = Self(1 << 0);
    pub const TRIAD: Self = Self(1 << 1);
    pub const FIXED: Self = Self(1 << 2);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericContractSet(u8);

impl NumericContractSet {
    pub const CUBLAS_POLICY_V1: Self = Self(1 << 0);
    pub const TRIAD_SCALAR_FMA_V1: Self = Self(1 << 1);
    pub const TRIAD_MMA_SYNC_V1: Self = Self(1 << 2);
    pub const FIXED_SCALAR_FMA_V1: Self = Self(1 << 3);
    pub const FIXED_MMA_SYNC_V1: Self = Self(1 << 4);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmRouteIdentity {
    pub policy: GemmPolicy,
    pub backend_set: BackendSet,
    pub numeric_contracts: NumericContractSet,
    pub compiler: CompilerIdentity,
    pub artifacts: ArtifactSetIdentity,
    pub policy_revision: u16,
    pub policy_hash: Sha256Digest,
    pub device: DeviceIdentity,
    pub state_capacity: u32,
}

impl GemmRouteIdentity {
    pub fn ensure_current(self, live: Self, prefix: &str) -> Result<(), String> {
        if self == live {
            Ok(())
        } else {
            Err(format!(
                "{prefix}: GEMM route changed since capture; re-capture before replay"
            ))
        }
    }
}
