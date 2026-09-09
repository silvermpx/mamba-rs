//! Stable compiler, artifact, policy, and device identities for CUDA graphs.

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::buffers::ManagedAllocationEpochStamp;
use super::context::{BiGemmFamily, F32TriadPolicy, GpuCtx, GpuCtxResources, HalfTriadPolicy};

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
pub const COMPILER_REVISION: u16 = 3;
pub const NUMERIC_ABI_REVISION: u16 = 5;
// Host dispatch epoch: qualified Ada Fixed finalists enter literal toolkit/cell
// AUTO rows; unchanged modules and retained cohorts keep their original evidence.
pub const TUNING_TABLE_REVISION: u16 = 45;
pub const SCHEDULE_REVISION: u16 = 8;

const NUMERIC_CONTRACT_DOMAIN: &[u8] = b"mamba-rs.resolved-numeric-contract.v2";
const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mamba-rs.artifact-digest.v1";
const COMPILER_TARGET_DOMAIN: &[u8] = b"mamba-rs.compiler-target.v1";
const DRIVER_BUILD_DIGEST_DOMAIN: &[u8] = b"mamba-rs.driver-build-digest.v1";
const LAUNCH_COUNT_DOMAIN: &[u8] = b"mamba-rs.resolved-launch-count.v1";
const ROUTE_INDEX_DOMAIN: &[u8] = b"mamba-rs.resolved-route-index.v1";
const PHYSICAL_LAUNCH_INDEX_DOMAIN: &[u8] = b"mamba-rs.resolved-physical-launch-index.v1";

pub(crate) fn deterministic_nvrtc_options(
    nvrtc_version: (i32, i32),
    random_seed: &str,
) -> Vec<String> {
    let mut options = Vec::new();
    if nvrtc_version >= (12, 9) {
        options.push(format!("--frandom-seed={random_seed}"));
    }
    options
}

pub const POLICY_REVISION: u16 = 5;

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
    pub module_kind: ModuleKind,
    pub source: Vec<u8>,
    pub target: Vec<u8>,
    pub argv: Vec<Vec<u8>>,
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
        let header_manifest = self.header_manifest.as_ref()?;
        self.nvrtc_library_domain.as_ref()?;
        let source_text = normalized_preprocessor_text(&self.source)?;
        let mut paste_analysis = header_manifest_analysis(header_manifest)?;
        paste_analysis.observe(&source_text)?;
        let mut volatile_analysis = PasteAnalysis::default();
        volatile_analysis.observe(&source_text)?;
        if contains_volatile_predefined_macro(&self.source)? {
            return None;
        }
        let mut argv_text = Vec::with_capacity(self.argv.len());
        for argument in &self.argv {
            let argument_text = normalized_preprocessor_text(argument)?;
            if contains_volatile_predefined_macro(argument)? {
                return None;
            }
            paste_analysis.has_date_time_fragment |=
                contains_date_time_fragment_in_text(&argument_text);
            volatile_analysis.has_date_time_fragment |=
                contains_date_time_fragment_in_text(&argument_text);
            argv_text.push(argument_text);
        }
        paste_analysis.observe_argv_definitions(&argv_text)?;
        volatile_analysis.observe_argv_definitions(&argv_text)?;
        let conditional_paste = paste_analysis.conditional_uses_paste();
        let conditional_has_include = paste_analysis.conditional_uses_has_include_fragment();
        let volatile_paste = volatile_analysis.uses_paste_to_form_volatile();
        let fragments_with_paste =
            volatile_analysis.has_paste && volatile_analysis.has_date_time_fragment;
        let manifest_fragments_with_paste =
            paste_analysis.has_paste && paste_analysis.has_date_time_fragment;
        if conditional_paste
            || conditional_has_include
            || volatile_paste
            || fragments_with_paste
            || manifest_fragments_with_paste
        {
            return None;
        }
        Some(self.invocation_digest())
    }

    pub fn invocation_digest(&self) -> Sha256Digest {
        let mut hash = FramedSha256::new(b"cuda-compile-key.v1")
            .required(b"module-kind", &[self.module_kind as u8])
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

pub(crate) fn canonical_ptx_from_cache(payload: Vec<u8>) -> Result<String, String> {
    let mut image = payload;
    image.push(0);
    canonical_ptx_image(&image)
}

#[cfg(target_os = "linux")]
static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
#[cfg(target_os = "linux")]
const MAX_CACHE_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(target_os = "linux")]
const CACHE_ENVELOPE_HEADER_BYTES: usize = 16 + 2 + 32 + 1 + 8 + 32;

#[cfg(target_os = "linux")]
fn effective_uid() -> u32 {
    unsafe { libc::geteuid() }
}

#[cfg(target_os = "linux")]
struct CacheDirectory {
    fd: std::os::fd::OwnedFd,
}

#[cfg(target_os = "linux")]
fn cache_path_components(path: &Path) -> Option<Vec<std::ffi::CString>> {
    use std::os::unix::ffi::OsStrExt;

    if !path.is_absolute() {
        return None;
    }
    let bytes = path.as_os_str().as_bytes();
    let mut components = Vec::new();
    for component in bytes.strip_prefix(b"/")?.split(|byte| *byte == b'/') {
        if component.is_empty() || matches!(component, b"." | b"..") {
            return None;
        }
        components.push(std::ffi::CString::new(component).ok()?);
    }
    (!components.is_empty()).then_some(components)
}

#[cfg(target_os = "linux")]
fn fstat(fd: std::os::fd::RawFd) -> Option<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return None;
    }
    Some(unsafe { stat.assume_init() })
}

#[cfg(target_os = "linux")]
fn trusted_directory_stat(stat: &libc::stat, final_component: bool) -> bool {
    if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
        return false;
    }
    let uid = effective_uid();
    if final_component {
        stat.st_uid == uid && stat.st_mode & 0o777 == 0o700
    } else {
        (stat.st_uid == 0 || stat.st_uid == uid) && stat.st_mode & 0o022 == 0
    }
}

#[cfg(target_os = "linux")]
fn open_cache_directory(path: &Path, create: bool) -> Option<CacheDirectory> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let components = cache_path_components(path)?;
    let root = std::ffi::CString::new("/").unwrap();
    let root_fd = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if root_fd < 0 {
        return None;
    }
    let mut current = CacheDirectory {
        fd: unsafe { std::os::fd::OwnedFd::from_raw_fd(root_fd) },
    };
    let root_stat = fstat(current.fd.as_raw_fd())?;
    if !trusted_directory_stat(&root_stat, false) {
        return None;
    }

    for (index, component) in components.iter().enumerate() {
        let final_component = index + 1 == components.len();
        let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
        let mut fd = unsafe { libc::openat(current.fd.as_raw_fd(), component.as_ptr(), flags) };
        if fd < 0 && create && std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT)
        {
            let made = unsafe { libc::mkdirat(current.fd.as_raw_fd(), component.as_ptr(), 0o700) };
            if made != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) {
                return None;
            }
            fd = unsafe { libc::openat(current.fd.as_raw_fd(), component.as_ptr(), flags) };
        }
        if fd < 0 {
            return None;
        }
        let next = CacheDirectory {
            fd: unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) },
        };
        let stat = fstat(next.fd.as_raw_fd())?;
        if !trusted_directory_stat(&stat, final_component) {
            return None;
        }
        current = next;
    }
    Some(current)
}

#[cfg(target_os = "linux")]
pub(crate) fn prepare_private_cache_dir(path: &Path) -> Option<PathBuf> {
    open_cache_directory(path, true).map(|_| path.to_path_buf())
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn prepare_private_cache_dir(_path: &Path) -> Option<PathBuf> {
    None
}

#[cfg(target_os = "linux")]
fn cache_location(
    path: &Path,
    create_directory: bool,
) -> Option<(CacheDirectory, std::ffi::CString)> {
    use std::os::unix::ffi::OsStrExt;

    let parent = path.parent()?;
    let name = path.file_name()?.as_bytes();
    if name.is_empty() || matches!(name, b"." | b"..") {
        return None;
    }
    Some((
        open_cache_directory(parent, create_directory)?,
        std::ffi::CString::new(name).ok()?,
    ))
}

#[cfg(target_os = "linux")]
fn open_cache_entry_at(directory: &CacheDirectory, name: &std::ffi::CStr) -> Option<std::fs::File> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let fd = unsafe {
        libc::openat(
            directory.fd.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return None;
    }
    let file = unsafe { std::fs::File::from_raw_fd(fd) };
    let stat = fstat(file.as_raw_fd())?;
    (stat.st_mode & libc::S_IFMT == libc::S_IFREG
        && stat.st_uid == effective_uid()
        && stat.st_mode & 0o777 == 0o600
        && stat.st_nlink == 1
        && stat.st_size >= 0
        && stat.st_size as u64 <= MAX_CACHE_ENTRY_BYTES)
        .then_some(file)
}

#[cfg(target_os = "linux")]
fn cache_entry_len(payload_len: usize) -> Option<usize> {
    let encoded_len = CACHE_ENVELOPE_HEADER_BYTES.checked_add(payload_len)?;
    let encoded_len_u64 = u64::try_from(encoded_len).ok()?;
    (encoded_len_u64 <= MAX_CACHE_ENTRY_BYTES).then_some(encoded_len)
}

#[cfg(target_os = "linux")]
pub(crate) fn read_cache(
    path: &Path,
    compile_key: Sha256Digest,
    artifact_kind: ArtifactKind,
) -> Option<CacheHit> {
    let (directory, name) = cache_location(path, false)?;
    let mut file = open_cache_entry_at(&directory, &name)?;
    let expected_len = file.metadata().ok()?.len();
    let capacity = usize::try_from(expected_len).ok()?;
    let mut bytes = Vec::with_capacity(capacity);
    file.by_ref()
        .take(MAX_CACHE_ENTRY_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 != expected_len {
        return None;
    }
    CacheEnvelope::decode(compile_key, artifact_kind, &bytes).ok()
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn read_cache(
    _path: &Path,
    _compile_key: Sha256Digest,
    _artifact_kind: ArtifactKind,
) -> Option<CacheHit> {
    None
}

#[cfg(target_os = "linux")]
pub(crate) fn publish_cache(
    path: &Path,
    compile_key: Sha256Digest,
    artifact_kind: ArtifactKind,
    payload: &[u8],
) {
    let Some(encoded_len) = cache_entry_len(payload.len()) else {
        return;
    };
    let encoded = CacheEnvelope::encode(compile_key, artifact_kind, payload);
    if encoded.len() != encoded_len {
        return;
    }
    let Some((directory, name)) = cache_location(path, true) else {
        return;
    };
    publish_cache_at(&directory, &name, &encoded);
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn publish_cache(
    _path: &Path,
    _compile_key: Sha256Digest,
    _artifact_kind: ArtifactKind,
    _payload: &[u8],
) {
}

#[cfg(target_os = "linux")]
fn publish_cache_at(directory: &CacheDirectory, name: &std::ffi::CStr, encoded: &[u8]) {
    use std::io::Write as _;
    use std::os::fd::{AsRawFd, FromRawFd};

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    let counter = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let Ok(file_name) = name.to_str() else {
        return;
    };
    let Ok(tmp) = std::ffi::CString::new(format!(
        ".{file_name}.tmp-{}-{stamp:x}-{counter:x}",
        std::process::id()
    )) else {
        return;
    };
    let fd = unsafe {
        libc::openat(
            directory.fd.as_raw_fd(),
            tmp.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if fd < 0 {
        return;
    }
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    let wrote = unsafe { libc::fchmod(file.as_raw_fd(), 0o600) } == 0
        && file.write_all(encoded).is_ok()
        && file.sync_all().is_ok();
    drop(file);
    if wrote
        && unsafe {
            libc::renameat(
                directory.fd.as_raw_fd(),
                tmp.as_ptr(),
                directory.fd.as_raw_fd(),
                name.as_ptr(),
            )
        } == 0
    {
        unsafe {
            libc::fsync(directory.fd.as_raw_fd());
        }
    }
    unsafe {
        libc::unlinkat(directory.fd.as_raw_fd(), tmp.as_ptr(), 0);
    }
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
    Fixed = 1,
    TriadScalar = 2,
    TriadSm80 = 3,
    TriadSm90a = 4,
    TriadSm100 = 5,
    TriadSm120 = 6,
    Mamba3Combined = 7,
    TriadSm89Finalist = 8,
    /// Exact-SM89 half-precision Triad finalists. Kept separate from both
    /// Fixed and the TF32 finalist so every loaded module has an independent
    /// artifact identity.
    TriadSm89Half = 9,
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
    pub fixed: ArtifactIdentity,
    pub triad_scalar: ArtifactIdentity,
    pub triad_sm80: ArtifactIdentity,
    pub specialized: Option<ArtifactIdentity>,
    pub sm89_half: Option<ArtifactIdentity>,
}

pub fn build_artifact_set(artifacts: &[ArtifactIdentity]) -> Result<ArtifactSetIdentity, String> {
    if !(3..=5).contains(&artifacts.len()) {
        return Err("artifact set must contain fixed, scalar triad, SM80 triad, one optional architecture-specialized module, and one optional SM89 half module".into());
    }
    if artifacts[0].module_kind != ModuleKind::Fixed
        || artifacts[1].module_kind != ModuleKind::TriadScalar
        || artifacts[2].module_kind != ModuleKind::TriadSm80
    {
        return Err(
            "artifact set must start with Fixed, TriadScalar, and TriadSm80 in order".into(),
        );
    }
    let mut specialized = None;
    let mut sm89_half = None;
    if let Some(artifact) = artifacts.get(3).copied() {
        match artifact.module_kind {
            ModuleKind::TriadSm89Half => sm89_half = Some(artifact),
            ModuleKind::TriadSm89Finalist
            | ModuleKind::TriadSm90a
            | ModuleKind::TriadSm100
            | ModuleKind::TriadSm120 => specialized = Some(artifact),
            _ => return Err("fourth artifact must be a supported specialized triad module".into()),
        }
    }
    if let Some(artifact) = artifacts.get(4).copied() {
        if !matches!(specialized, Some(value) if value.module_kind == ModuleKind::TriadSm89Finalist)
            || artifact.module_kind != ModuleKind::TriadSm89Half
        {
            return Err(
                "fifth artifact is only valid for TriadSm89Finalist followed by TriadSm89Half"
                    .into(),
            );
        }
        sm89_half = Some(artifact);
    }
    let module_count = u8::try_from(artifacts.len())
        .map_err(|_| "artifact set contains more than 255 modules".to_string())?;
    let mut kinds = BTreeSet::new();
    let mut hash =
        FramedSha256::new(b"cuda-artifact-set.v1").required(b"module-count", &[module_count]);
    for artifact in artifacts {
        if !kinds.insert(artifact.module_kind as u8) {
            return Err(format!(
                "artifact set contains duplicate module kind {:?}",
                artifact.module_kind
            ));
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
        fixed: artifacts[0],
        triad_scalar: artifacts[1],
        triad_sm80: artifacts[2],
        specialized,
        sm89_half,
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
    pub multiprocessor_count: u32,
    pub target: CudaTarget,
    pub driver: DriverIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceCaps {
    pub compute_capability: (u32, u32),
    pub nvrtc_version: (i32, i32),
    pub accepted_target: Option<CudaTarget>,
    pub optin_shared_bytes: u32,
    pub tensor_map_access: bool,
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
    static DOMAIN: OnceLock<Vec<u8>> = OnceLock::new();
    if let Some(domain) = DOMAIN.get() {
        return Some(domain.clone());
    }
    let domain = query_nvrtc_library_domain()?;
    let _ = DOMAIN.set(domain.clone());
    Some(domain)
}

pub(crate) fn nvrtc_library_domain_is_current(expected: &[u8]) -> bool {
    query_nvrtc_library_domain().as_deref() == Some(expected)
}

fn query_nvrtc_library_domain() -> Option<Vec<u8>> {
    let runtime = unique_loaded_library_path("libnvrtc.so")?;
    let builtins = match unique_loaded_library_path("libnvrtc-builtins.so") {
        Some(path) => path,
        None => nvrtc_builtins_sibling(&runtime)?,
    };
    nvrtc_library_domain_from_paths(&runtime, &builtins)
}

fn loaded_library_domain(needle: &str) -> Option<Sha256Digest> {
    let canonical = unique_loaded_library_path(needle)?;
    let content_digest = sha256_file(&canonical).ok()?;
    Some(
        FramedSha256::new(b"loaded-library.v1")
            .required(b"path", canonical.as_os_str().as_encoded_bytes())
            .required(b"content-sha256", &content_digest)
            .finish(),
    )
}

fn unique_loaded_library_path(needle: &str) -> Option<PathBuf> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        let Some(start) = line.find('/') else {
            continue;
        };
        let path = &line[start..];
        if !path.contains(needle) {
            continue;
        }
        if path.ends_with(" (deleted)") {
            return None;
        }
        let canonical = std::fs::canonicalize(path).ok()?;
        let name = canonical.file_name()?.to_str()?;
        if name.starts_with(needle) {
            paths.insert(canonical);
        }
    }
    (paths.len() == 1).then(|| paths.into_iter().next().unwrap())
}

fn nvrtc_builtins_sibling(runtime: &Path) -> Option<PathBuf> {
    let directory = runtime.parent()?;
    let runtime_name = runtime.file_name()?.to_str()?;
    let suffix = runtime_name.strip_prefix("libnvrtc.so")?;
    if !suffix.is_empty() {
        let exact = directory.join(format!("libnvrtc-builtins.so{suffix}"));
        if exact.is_file() {
            return std::fs::canonicalize(exact).ok();
        }
    }

    let mut candidates = BTreeSet::new();
    for entry in std::fs::read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("libnvrtc-builtins.so") && path.is_file() {
            candidates.insert(std::fs::canonicalize(path).ok()?);
        }
    }
    (candidates.len() == 1).then(|| candidates.into_iter().next().unwrap())
}

fn nvrtc_library_domain_from_paths(runtime: &Path, builtins: &Path) -> Option<Vec<u8>> {
    let runtime = std::fs::canonicalize(runtime).ok()?;
    let builtins = std::fs::canonicalize(builtins).ok()?;
    if runtime == builtins {
        return None;
    }
    let runtime_digest = sha256_file(&runtime).ok()?;
    let builtins_digest = sha256_file(&builtins).ok()?;
    Some(
        FramedSha256::new(b"nvrtc-library-set.v2")
            .required(b"runtime-path", runtime.as_os_str().as_encoded_bytes())
            .required(b"runtime-sha256", &runtime_digest)
            .required(b"builtins-path", builtins.as_os_str().as_encoded_bytes())
            .required(b"builtins-sha256", &builtins_digest)
            .finish()
            .to_vec(),
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

/// Builds an over-approximation of literal includes. Filesystem headers carry
/// content hashes; unresolved angle names are bound by the separately required
/// NVRTC builtins-library hash. Ambiguous forms disable persistent caching.
pub(crate) fn header_manifest(source: &[u8], include_roots: &[String]) -> Option<Vec<u8>> {
    const MAX_HEADER_CONTEXTS: usize = 16 * 1024;

    let roots: Vec<PathBuf> = include_roots
        .iter()
        .map(|root| {
            let path = Path::new(root);
            if path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                return None;
            }
            lexical_normalize_absolute(path)
        })
        .collect::<Option<_>>()?;
    let mut pending = Vec::new();
    let mut records = BTreeMap::<Vec<u8>, (Vec<u8>, Sha256Digest)>::new();
    let mut builtin_headers = BTreeSet::<Vec<u8>>::new();
    let mut paste_analysis = PasteAnalysis::default();
    let mut volatile_analysis = PasteAnalysis::default();
    volatile_analysis.observe(&normalized_preprocessor_text(source)?)?;
    collect_includes(
        source,
        None,
        &roots,
        &[],
        &mut pending,
        &mut builtin_headers,
        &mut paste_analysis,
    )?;
    if pending.len() > MAX_HEADER_CONTEXTS {
        return None;
    }

    let mut visited_contexts = 0usize;
    while let Some(header) = pending.pop() {
        visited_contexts = visited_contexts.checked_add(1)?;
        if visited_contexts > MAX_HEADER_CONTEXTS {
            return None;
        }
        let logical = lexical_normalize_absolute(&header.logical_path)?;
        let canonical = std::fs::canonicalize(&logical).ok()?;
        let logical_key = manifest_path_key(&logical, &roots)?;
        if records.contains_key(&logical_key) {
            continue;
        }
        if header.canonical_ancestry.contains(&canonical) {
            return None;
        }
        let bytes = std::fs::read(&canonical).ok()?;
        let canonical_key = manifest_path_key(&canonical, &roots)?;
        records.insert(logical_key, (canonical_key, FramedSha256::bytes(&bytes)));
        let mut ancestry = header.canonical_ancestry;
        ancestry.push(canonical);
        collect_includes(
            &bytes,
            logical.parent(),
            &roots,
            &ancestry,
            &mut pending,
            &mut builtin_headers,
            &mut paste_analysis,
        )?;
        if records.len().checked_add(pending.len())? > MAX_HEADER_CONTEXTS {
            return None;
        }
    }
    let conditional_paste = paste_analysis.conditional_uses_paste();
    let conditional_has_include = paste_analysis.conditional_uses_has_include_fragment();
    let volatile_paste = volatile_analysis.uses_paste_to_form_volatile();
    if conditional_paste
        || conditional_has_include
        || volatile_paste
        || volatile_analysis.has_paste && volatile_analysis.has_date_time_fragment
        || paste_analysis.has_paste && paste_analysis.has_date_time_fragment
    {
        return None;
    }

    let mut output = Vec::new();
    output.extend_from_slice(b"MAMBA-HDR-MANIFEST-V3\0");
    output.push(u8::from(paste_analysis.has_paste));
    output.push(u8::from(paste_analysis.has_date_time_fragment));
    let analysis_bytes = paste_analysis.encode();
    append_manifest_field(&mut output, &analysis_bytes);
    output.extend_from_slice(&(builtin_headers.len() as u64).to_le_bytes());
    for name in builtin_headers {
        append_manifest_field(&mut output, &name);
    }
    output.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (path, (canonical, digest)) in records {
        append_manifest_field(&mut output, &path);
        append_manifest_field(&mut output, &canonical);
        append_manifest_field(&mut output, &digest);
    }
    Some(output)
}

pub(crate) fn header_manifest_is_current(
    source: &[u8],
    include_roots: &[String],
    expected: &Option<Vec<u8>>,
) -> bool {
    header_manifest(source, include_roots) == *expected
}

/// Final cache-hit closure check. Call only after parsing/loading the cached
/// artifact so this is the last filesystem observation before acceptance.
pub(crate) fn cache_hit_header_closure_is_current(
    source: &[u8],
    include_roots: &[String],
    expected: &Option<Vec<u8>>,
) -> bool {
    header_manifest_is_current(source, include_roots, expected)
}

fn append_manifest_field(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn manifest_path_key(path: &Path, roots: &[PathBuf]) -> Option<Vec<u8>> {
    for (index, root) in roots.iter().enumerate() {
        if let Ok(relative) = path.strip_prefix(root) {
            let mut key = vec![0];
            key.extend_from_slice(&(index as u64).to_le_bytes());
            key.extend_from_slice(relative.as_os_str().as_encoded_bytes());
            return Some(key);
        }
    }
    let mut key = vec![1];
    key.extend_from_slice(path.as_os_str().as_encoded_bytes());
    Some(key)
}

struct PendingHeader {
    logical_path: PathBuf,
    canonical_ancestry: Vec<PathBuf>,
}

fn lexical_normalize_absolute(path: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if !path.is_absolute() {
        return None;
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    Some(normalized)
}

fn collect_includes(
    source: &[u8],
    current_dir: Option<&Path>,
    roots: &[PathBuf],
    canonical_ancestry: &[PathBuf],
    pending: &mut Vec<PendingHeader>,
    builtin_headers: &mut BTreeSet<Vec<u8>>,
    paste_analysis: &mut PasteAnalysis,
) -> Option<()> {
    let text = normalized_preprocessor_text(source)?;
    if contains_volatile_predefined_macro_in_text(&text) {
        return None;
    }
    paste_analysis.observe(&text)?;
    for line in text.lines() {
        let Some(rest) = include_directive(line)? else {
            continue;
        };
        let (quoted, name, trailing) = if let Some(rest) = rest.strip_prefix('"') {
            let (name, trailing) = rest.split_once('"')?;
            (true, name, trailing)
        } else {
            let rest = rest.strip_prefix('<')?;
            let (name, trailing) = rest.split_once('>')?;
            (false, name, trailing)
        };
        if name.is_empty() || !trailing.trim().is_empty() {
            return None;
        }
        if Path::new(name)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return None;
        }
        let mut candidates = Vec::new();
        if quoted && let Some(dir) = current_dir {
            candidates.push(dir.join(name));
        }
        candidates.extend(roots.iter().map(|root| root.join(name)));
        if let Some(path) = candidates
            .into_iter()
            .filter_map(|path| lexical_normalize_absolute(&path))
            .find(|path| path.is_file())
        {
            pending.push(PendingHeader {
                logical_path: path,
                canonical_ancestry: canonical_ancestry.to_vec(),
            });
        } else if quoted {
            return None;
        } else if known_nvrtc_builtin_header(name) {
            builtin_headers.insert(name.as_bytes().to_vec());
        } else {
            return None;
        }
    }
    Some(())
}

#[derive(Clone, Default)]
struct MacroAnalysis {
    pasted: bool,
    references: BTreeSet<String>,
    has_include_fragment: bool,
    object_alias: Option<String>,
    alias_ambiguous: bool,
    object_aliases: BTreeSet<String>,
    object_barrier: bool,
    object_complex: bool,
    zero_arg_aliases: BTreeSet<String>,
    paste_tokens: Vec<String>,
    parameter_count: Option<usize>,
    parameter_ambiguous: bool,
    direct_sensitive_parameters: BTreeSet<usize>,
    parameter_flows: Vec<ParameterFlow>,
}

#[derive(Clone, Default)]
struct ParameterFlow {
    callee: String,
    arguments: Vec<ParameterSource>,
}

#[derive(Clone, Default)]
struct ParameterSource {
    parameters: BTreeSet<usize>,
    tokens: BTreeSet<String>,
    unknown: bool,
}

#[derive(Clone, Default)]
struct ConditionalAnalysis {
    identifiers: BTreeSet<String>,
    has_include_fragment: bool,
}

#[derive(Clone, Default)]
struct PasteAnalysis {
    macros: BTreeMap<String, MacroAnalysis>,
    conditionals: Vec<ConditionalAnalysis>,
    direct_conditional_paste: bool,
    has_paste: bool,
    has_date_time_fragment: bool,
    macro_calls: Vec<(String, Vec<String>)>,
}

impl PasteAnalysis {
    fn observe(&mut self, text: &str) -> Option<()> {
        self.has_paste |= contains_token_paste(text);
        self.has_date_time_fragment |= contains_date_time_fragment_in_text(text);
        for line in text.lines() {
            let Some((directive, rest)) = preprocessor_directive(line) else {
                self.macro_calls.extend(simple_macro_calls(line));
                continue;
            };
            match directive {
                "define" => {
                    let (name, replacement, parameters) = macro_definition(rest)?;
                    self.observe_macro(name, replacement, parameters.as_deref());
                }
                "undef" => {
                    let name = rest.trim();
                    if is_identifier(name) {
                        self.macros
                            .entry(name.to_owned())
                            .or_default()
                            .alias_ambiguous = true;
                    }
                }
                "if" | "elif" => {
                    self.macro_calls.extend(simple_macro_calls(rest));
                    self.direct_conditional_paste |= rest.contains("##") || rest.contains("%:%:");
                    self.conditionals.push(ConditionalAnalysis {
                        identifiers: expanded_conditional_identifiers(rest),
                        has_include_fragment: contains_has_include_fragment_in_text(rest),
                    });
                }
                _ => {}
            }
        }
        Some(())
    }

    fn observe_macro(&mut self, name: &str, replacement: &str, parameters: Option<&[String]>) {
        let function_like = parameters.is_some();
        let zero_arg = parameters.is_some_and(<[String]>::is_empty);
        let entry = self.macros.entry(name.to_owned()).or_default();
        entry.pasted |= contains_token_paste(replacement);
        entry
            .references
            .extend(identifiers(replacement).map(str::to_owned));
        entry.has_include_fragment |= contains_has_include_fragment_in_text(replacement);
        let new_alias =
            (!function_like && is_identifier(replacement)).then(|| replacement.to_owned());
        if entry.object_alias.is_some() && entry.object_alias != new_alias {
            entry.alias_ambiguous = true;
        }
        if new_alias.is_some() {
            entry.object_alias = new_alias.clone();
        }
        if let Some(alias) = &new_alias {
            entry.object_aliases.insert(alias.clone());
        }
        if !function_like && new_alias.is_none() {
            if is_safe_paste_barrier(replacement) {
                entry.object_barrier = true;
            } else {
                entry.object_complex = true;
            }
        }
        if zero_arg && is_identifier(replacement) {
            entry.zero_arg_aliases.insert(replacement.to_owned());
        }
        if let Some(parameters) = parameters {
            if entry.parameter_count.is_some() && entry.parameter_count != Some(parameters.len()) {
                entry.parameter_ambiguous = true;
            }
            entry.parameter_count = Some(
                entry
                    .parameter_count
                    .map_or(parameters.len(), |count| count.max(parameters.len())),
            );
            for token in paste_chain_identifiers(replacement) {
                if let Some(index) = parameters.iter().position(|parameter| parameter == &token) {
                    entry.direct_sensitive_parameters.insert(index);
                }
            }
            for (callee, arguments) in simple_macro_calls(replacement) {
                let arguments = arguments
                    .into_iter()
                    .map(|argument| {
                        let mut source = ParameterSource::default();
                        let (complex, identifiers) = argument.strip_prefix('?').map_or(
                            (false, vec![argument.as_str()]),
                            |value| {
                                (
                                    true,
                                    value.split(':').filter(|value| !value.is_empty()).collect(),
                                )
                            },
                        );
                        source.unknown = complex;
                        for identifier in identifiers {
                            if let Some(index) = parameters
                                .iter()
                                .position(|parameter| parameter == identifier)
                            {
                                source.parameters.insert(index);
                            } else if identifier != "$" {
                                source.tokens.insert(identifier.to_owned());
                            }
                        }
                        source
                    })
                    .collect();
                entry
                    .parameter_flows
                    .push(ParameterFlow { callee, arguments });
            }
        }
        entry
            .paste_tokens
            .extend(paste_chain_identifiers(replacement));
    }

    fn observe_argv_definitions(&mut self, argv: &[String]) -> Option<()> {
        let mut index = 0;
        while index < argv.len() {
            let argument = argv[index].as_str();
            let definition = if argument == "-D" || argument == "--define-macro" {
                index = index.checked_add(1)?;
                Some(argv.get(index)?.as_str())
            } else {
                argument
                    .strip_prefix("-D")
                    .filter(|definition| !definition.is_empty())
                    .or_else(|| argument.strip_prefix("--define-macro="))
            };
            if let Some(definition) = definition {
                let (declarator, replacement) = definition
                    .split_once('=')
                    .map_or((definition, "1"), |(left, right)| (left, right));
                let parsed = format!("{declarator} {replacement}");
                let (name, replacement, parameters) = macro_definition(&parsed)?;
                self.observe_macro(name, replacement, parameters.as_deref());
                self.has_paste |= contains_token_paste(replacement);
            }
            index += 1;
        }
        Some(())
    }

    fn conditional_uses_paste(&self) -> bool {
        let mut paste_capable: BTreeSet<String> = self
            .macros
            .iter()
            .filter_map(|(name, analysis)| analysis.pasted.then_some(name.clone()))
            .collect();
        paste_capable.extend(
            self.macros
                .values()
                .flat_map(|analysis| analysis.references.iter())
                .filter(|name| is_builtin_paste_macro(name))
                .cloned(),
        );
        loop {
            let mut changed = false;
            for (name, analysis) in &self.macros {
                if !paste_capable.contains(name)
                    && analysis
                        .references
                        .iter()
                        .any(|reference| paste_capable.contains(reference))
                {
                    changed |= paste_capable.insert(name.clone());
                }
            }
            if !changed {
                break;
            }
        }

        self.direct_conditional_paste
            || self.conditionals.iter().any(|conditional| {
                conditional
                    .identifiers
                    .iter()
                    .any(|name| paste_capable.contains(name))
            })
    }

    fn conditional_uses_has_include_fragment(&self) -> bool {
        let mut fragment_capable: BTreeSet<String> = self
            .macros
            .iter()
            .filter_map(|(name, analysis)| analysis.has_include_fragment.then_some(name.clone()))
            .collect();
        loop {
            let mut changed = false;
            for (name, analysis) in &self.macros {
                if !fragment_capable.contains(name)
                    && analysis
                        .references
                        .iter()
                        .any(|reference| fragment_capable.contains(reference))
                {
                    changed |= fragment_capable.insert(name.clone());
                }
            }
            if !changed {
                break;
            }
        }
        self.conditionals.iter().any(|conditional| {
            conditional.has_include_fragment
                || conditional
                    .identifiers
                    .iter()
                    .any(|name| fragment_capable.contains(name))
        })
    }

    fn uses_paste_to_form_volatile(&self) -> bool {
        let mut paste_capable: BTreeSet<String> = self
            .macros
            .iter()
            .filter_map(|(name, analysis)| analysis.pasted.then_some(name.clone()))
            .collect();
        paste_capable.extend(
            self.macros
                .values()
                .flat_map(|analysis| analysis.references.iter())
                .filter(|name| is_builtin_paste_macro(name))
                .cloned(),
        );
        loop {
            let mut changed = false;
            for (name, analysis) in &self.macros {
                if !paste_capable.contains(name)
                    && analysis
                        .references
                        .iter()
                        .any(|reference| paste_capable.contains(reference))
                {
                    changed |= paste_capable.insert(name.clone());
                }
            }
            if !changed {
                break;
            }
        }
        let mut sensitive: BTreeMap<String, BTreeSet<usize>> = self
            .macros
            .iter()
            .map(|(name, analysis)| {
                let mut parameters = analysis.direct_sensitive_parameters.clone();
                if analysis.parameter_ambiguous
                    && let Some(count) = analysis.parameter_count
                {
                    parameters.extend(0..count);
                }
                (name.clone(), parameters)
            })
            .collect();
        loop {
            let snapshot = sensitive.clone();
            let mut changed = false;
            for (name, analysis) in &self.macros {
                let Some(_) = analysis.parameter_count else {
                    continue;
                };
                let entry = sensitive.entry(name.clone()).or_default();
                for flow in &analysis.parameter_flows {
                    let callee_parameters: BTreeSet<usize> = if is_builtin_paste_macro(&flow.callee)
                    {
                        (0..flow.arguments.len()).collect()
                    } else {
                        snapshot.get(&flow.callee).cloned().unwrap_or_default()
                    };
                    for index in callee_parameters {
                        let Some(source) = flow.arguments.get(index) else {
                            continue;
                        };
                        for parameter in &source.parameters {
                            changed |= entry.insert(*parameter);
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for analysis in self.macros.values() {
            for flow in &analysis.parameter_flows {
                let callee_parameters: BTreeSet<usize> = if is_builtin_paste_macro(&flow.callee) {
                    (0..flow.arguments.len()).collect()
                } else {
                    sensitive.get(&flow.callee).cloned().unwrap_or_default()
                };
                let mut slots = Vec::new();
                for index in callee_parameters {
                    let Some(source) = flow.arguments.get(index) else {
                        continue;
                    };
                    if !source.parameters.is_empty() {
                        continue;
                    }
                    for token in &source.tokens {
                        let Some(candidates) = resolve_alias_candidates(&self.macros, token) else {
                            return true;
                        };
                        slots.push(candidates.into_iter().collect());
                    }
                }
                if let Some(callee) = self.macros.get(&flow.callee) {
                    slots.extend(callee.paste_tokens.iter().cloned().map(|token| vec![token]));
                }
                if paste_slots_form_volatile(&slots) {
                    return true;
                }
            }
        }
        let mut fixed_slots = BTreeMap::<String, Vec<Vec<String>>>::new();
        for (name, analysis) in &self.macros {
            let slots = fixed_slots.entry(name.clone()).or_default();
            for flow in &analysis.parameter_flows {
                let callee_parameters: BTreeSet<usize> = if is_builtin_paste_macro(&flow.callee) {
                    (0..flow.arguments.len()).collect()
                } else {
                    sensitive.get(&flow.callee).cloned().unwrap_or_default()
                };
                for index in callee_parameters {
                    let Some(source) = flow.arguments.get(index) else {
                        continue;
                    };
                    for token in &source.tokens {
                        let Some(candidates) = resolve_alias_candidates(&self.macros, token) else {
                            return true;
                        };
                        slots.push(candidates.into_iter().collect());
                    }
                }
            }
        }
        self.macro_calls.iter().any(|(name, arguments)| {
            if !paste_capable.contains(name) && !is_builtin_paste_macro(name) {
                return false;
            }
            let sensitive_arguments: BTreeSet<usize> = if is_builtin_paste_macro(name) {
                (0..arguments.len()).collect()
            } else if let Some(analysis) = self.macros.get(name) {
                if analysis.parameter_count.is_none() {
                    (0..arguments.len()).collect()
                } else {
                    sensitive.get(name).cloned().unwrap_or_default()
                }
            } else {
                (0..arguments.len()).collect()
            };
            let mut slots = Vec::new();
            for (index, argument) in arguments.iter().enumerate() {
                if !sensitive_arguments.contains(&index) {
                    continue;
                }
                if argument.is_empty() || argument == "$" {
                    continue;
                }
                if argument.starts_with('?') {
                    return true;
                }
                let Some(candidates) = resolve_alias_candidates(&self.macros, argument) else {
                    return true;
                };
                slots.push(candidates.into_iter().collect());
            }
            if let Some(analysis) = self.macros.get(name) {
                slots.extend(
                    analysis
                        .paste_tokens
                        .iter()
                        .cloned()
                        .map(|token| vec![token]),
                );
            }
            if let Some(fixed) = fixed_slots.get(name) {
                slots.extend(fixed.iter().cloned());
            }
            paste_slots_form_volatile(&slots)
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut output = Vec::new();
        output.push(u8::from(self.direct_conditional_paste));
        output.extend_from_slice(&(self.macros.len() as u64).to_le_bytes());
        for (name, analysis) in &self.macros {
            append_manifest_field(&mut output, name.as_bytes());
            output.push(u8::from(analysis.pasted));
            output.push(u8::from(analysis.has_include_fragment));
            output.push(u8::from(analysis.object_alias.is_some()));
            if let Some(alias) = &analysis.object_alias {
                append_manifest_field(&mut output, alias.as_bytes());
            }
            output.push(u8::from(analysis.alias_ambiguous));
            output.extend_from_slice(&(analysis.object_aliases.len() as u64).to_le_bytes());
            for alias in &analysis.object_aliases {
                append_manifest_field(&mut output, alias.as_bytes());
            }
            output.push(u8::from(analysis.object_barrier));
            output.push(u8::from(analysis.object_complex));
            output.extend_from_slice(&(analysis.zero_arg_aliases.len() as u64).to_le_bytes());
            for alias in &analysis.zero_arg_aliases {
                append_manifest_field(&mut output, alias.as_bytes());
            }
            output.extend_from_slice(&(analysis.paste_tokens.len() as u64).to_le_bytes());
            for token in &analysis.paste_tokens {
                append_manifest_field(&mut output, token.as_bytes());
            }
            output.push(u8::from(analysis.parameter_count.is_some()));
            if let Some(count) = analysis.parameter_count {
                output.extend_from_slice(&(count as u64).to_le_bytes());
            }
            output.push(u8::from(analysis.parameter_ambiguous));
            output.extend_from_slice(
                &(analysis.direct_sensitive_parameters.len() as u64).to_le_bytes(),
            );
            for parameter in &analysis.direct_sensitive_parameters {
                output.extend_from_slice(&(*parameter as u64).to_le_bytes());
            }
            output.extend_from_slice(&(analysis.parameter_flows.len() as u64).to_le_bytes());
            for flow in &analysis.parameter_flows {
                append_manifest_field(&mut output, flow.callee.as_bytes());
                output.extend_from_slice(&(flow.arguments.len() as u64).to_le_bytes());
                for source in &flow.arguments {
                    output.push(u8::from(source.unknown));
                    output.extend_from_slice(&(source.parameters.len() as u64).to_le_bytes());
                    for parameter in &source.parameters {
                        output.extend_from_slice(&(*parameter as u64).to_le_bytes());
                    }
                    output.extend_from_slice(&(source.tokens.len() as u64).to_le_bytes());
                    for token in &source.tokens {
                        append_manifest_field(&mut output, token.as_bytes());
                    }
                }
            }
            output.extend_from_slice(&(analysis.references.len() as u64).to_le_bytes());
            for reference in &analysis.references {
                append_manifest_field(&mut output, reference.as_bytes());
            }
        }
        output.extend_from_slice(&(self.conditionals.len() as u64).to_le_bytes());
        for conditional in &self.conditionals {
            output.push(u8::from(conditional.has_include_fragment));
            output.extend_from_slice(&(conditional.identifiers.len() as u64).to_le_bytes());
            for identifier in &conditional.identifiers {
                append_manifest_field(&mut output, identifier.as_bytes());
            }
        }
        output.extend_from_slice(&(self.macro_calls.len() as u64).to_le_bytes());
        for (name, arguments) in &self.macro_calls {
            append_manifest_field(&mut output, name.as_bytes());
            output.extend_from_slice(&(arguments.len() as u64).to_le_bytes());
            for argument in arguments {
                append_manifest_field(&mut output, argument.as_bytes());
            }
        }
        output
    }

    fn decode(bytes: &[u8], has_paste: bool, has_date_time_fragment: bool) -> Option<Self> {
        let mut cursor = ManifestCursor::new(bytes);
        let direct_conditional_paste = cursor.byte()? != 0;
        let macro_count = cursor.usize()?;
        let mut macros = BTreeMap::new();
        for _ in 0..macro_count {
            let name = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
            if !is_identifier(&name) {
                return None;
            }
            let pasted = cursor.byte()? != 0;
            let has_include_fragment = cursor.byte()? != 0;
            let object_alias = if cursor.byte()? != 0 {
                let alias = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&alias) {
                    return None;
                }
                Some(alias)
            } else {
                None
            };
            let alias_ambiguous = cursor.byte()? != 0;
            let alias_count = cursor.usize()?;
            let mut object_aliases = BTreeSet::new();
            for _ in 0..alias_count {
                let alias = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&alias) {
                    return None;
                }
                object_aliases.insert(alias);
            }
            let object_barrier = cursor.byte()? != 0;
            let object_complex = cursor.byte()? != 0;
            let zero_arg_alias_count = cursor.usize()?;
            let mut zero_arg_aliases = BTreeSet::new();
            for _ in 0..zero_arg_alias_count {
                let alias = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&alias) {
                    return None;
                }
                zero_arg_aliases.insert(alias);
            }
            let paste_token_count = cursor.usize()?;
            let mut paste_tokens = Vec::with_capacity(paste_token_count);
            for _ in 0..paste_token_count {
                let token = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&token) {
                    return None;
                }
                paste_tokens.push(token);
            }
            let parameter_count = if cursor.byte()? != 0 {
                Some(cursor.usize()?)
            } else {
                None
            };
            let parameter_ambiguous = cursor.byte()? != 0;
            let direct_sensitive_count = cursor.usize()?;
            let mut direct_sensitive_parameters = BTreeSet::new();
            for _ in 0..direct_sensitive_count {
                let parameter = cursor.usize()?;
                if parameter_count.is_none_or(|count| parameter >= count) {
                    return None;
                }
                direct_sensitive_parameters.insert(parameter);
            }
            let flow_count = cursor.usize()?;
            let mut parameter_flows = Vec::with_capacity(flow_count);
            for _ in 0..flow_count {
                let callee = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&callee) {
                    return None;
                }
                let argument_count = cursor.usize()?;
                let mut arguments = Vec::with_capacity(argument_count);
                for _ in 0..argument_count {
                    let unknown = cursor.byte()? != 0;
                    let source_count = cursor.usize()?;
                    let mut parameters = BTreeSet::new();
                    for _ in 0..source_count {
                        let parameter = cursor.usize()?;
                        if parameter_count.is_none_or(|count| parameter >= count) {
                            return None;
                        }
                        parameters.insert(parameter);
                    }
                    let token_count = cursor.usize()?;
                    let mut tokens = BTreeSet::new();
                    for _ in 0..token_count {
                        let token = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                        if !is_identifier(&token)
                            && !token.strip_prefix('@').is_some_and(is_identifier)
                        {
                            return None;
                        }
                        tokens.insert(token);
                    }
                    arguments.push(ParameterSource {
                        parameters,
                        tokens,
                        unknown,
                    });
                }
                parameter_flows.push(ParameterFlow { callee, arguments });
            }
            let reference_count = cursor.usize()?;
            let mut references = BTreeSet::new();
            for _ in 0..reference_count {
                let reference = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&reference) {
                    return None;
                }
                references.insert(reference);
            }
            macros.insert(
                name,
                MacroAnalysis {
                    pasted,
                    references,
                    has_include_fragment,
                    object_alias,
                    alias_ambiguous,
                    object_aliases,
                    object_barrier,
                    object_complex,
                    zero_arg_aliases,
                    paste_tokens,
                    parameter_count,
                    parameter_ambiguous,
                    direct_sensitive_parameters,
                    parameter_flows,
                },
            );
        }
        let conditional_count = cursor.usize()?;
        let mut conditionals = Vec::with_capacity(conditional_count);
        for _ in 0..conditional_count {
            let has_include_fragment = cursor.byte()? != 0;
            let identifier_count = cursor.usize()?;
            let mut identifiers = BTreeSet::new();
            for _ in 0..identifier_count {
                let identifier = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                if !is_identifier(&identifier) {
                    return None;
                }
                identifiers.insert(identifier);
            }
            conditionals.push(ConditionalAnalysis {
                identifiers,
                has_include_fragment,
            });
        }
        let call_count = cursor.usize()?;
        let mut macro_calls = Vec::with_capacity(call_count);
        for _ in 0..call_count {
            let name = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
            if !is_identifier(&name) {
                return None;
            }
            let argument_count = cursor.usize()?;
            let mut arguments = Vec::with_capacity(argument_count);
            for _ in 0..argument_count {
                let argument = std::str::from_utf8(cursor.field()?).ok()?.to_owned();
                let nested_zero_arg = argument.strip_prefix('@').is_some_and(is_identifier);
                let complex_argument = argument.strip_prefix('?').is_some_and(|identifiers| {
                    identifiers
                        .split(':')
                        .filter(|identifier| !identifier.is_empty())
                        .all(is_identifier)
                });
                if argument != "$"
                    && !argument.is_empty()
                    && !is_identifier(&argument)
                    && !nested_zero_arg
                    && !complex_argument
                {
                    return None;
                }
                arguments.push(argument);
            }
            macro_calls.push((name, arguments));
        }
        cursor.is_empty().then_some(Self {
            macros,
            conditionals,
            direct_conditional_paste,
            has_paste,
            has_date_time_fragment,
            macro_calls,
        })
    }
}

fn resolve_alias_candidates(
    macros: &BTreeMap<String, MacroAnalysis>,
    token: &str,
) -> Option<BTreeSet<String>> {
    fn visit(
        macros: &BTreeMap<String, MacroAnalysis>,
        token: &str,
        depth: usize,
        active: &mut BTreeSet<String>,
        output: &mut BTreeSet<String>,
    ) -> Option<()> {
        if depth == 0 || !active.insert(token.to_owned()) {
            return None;
        }
        let (name, zero_arg) = token
            .strip_prefix('@')
            .map_or((token, false), |name| (name, true));
        let analysis = macros.get(name);
        if zero_arg || analysis.is_some_and(|analysis| analysis.object_complex) {
            return None;
        }
        let aliases = analysis.map(|analysis| {
            if zero_arg {
                &analysis.zero_arg_aliases
            } else {
                &analysis.object_aliases
            }
        });
        if let Some(aliases) = aliases
            && !aliases.is_empty()
        {
            for alias in aliases {
                visit(macros, alias, depth - 1, active, output)?;
            }
            if analysis.is_some_and(|analysis| analysis.alias_ambiguous) {
                output.insert(name.to_owned());
            }
        } else if analysis.is_some_and(|analysis| analysis.object_barrier) {
            // An empty, numeric, string, or character replacement cannot
            // contribute identifier bytes to a pasted volatile name.
        } else {
            output.insert(token.to_owned());
        }
        active.remove(token);
        Some(())
    }

    let mut output = BTreeSet::new();
    visit(macros, token, 32, &mut BTreeSet::new(), &mut output)?;
    Some(output)
}

fn paste_slots_form_volatile(slots: &[Vec<String>]) -> bool {
    const TARGETS: [&str; 4] = [
        "__DATE__",
        "__TIME__",
        "__has_include",
        "__has_include_next",
    ];
    let slots: Vec<Vec<&str>> = slots
        .iter()
        .map(|slot| {
            slot.iter()
                .map(String::as_str)
                .filter(|candidate| {
                    !candidate.is_empty() && TARGETS.iter().any(|target| target.contains(candidate))
                })
                .collect()
        })
        .filter(|slot: &Vec<_>| !slot.is_empty())
        .collect();

    fn search(target: &str, offset: usize, slots: &[Vec<&str>], used: &mut [bool]) -> bool {
        if offset == target.len() {
            return true;
        }
        for (index, candidates) in slots.iter().enumerate() {
            if used[index] {
                continue;
            }
            for candidate in candidates {
                if target[offset..].starts_with(candidate) {
                    used[index] = true;
                    if search(target, offset + candidate.len(), slots, used) {
                        return true;
                    }
                    used[index] = false;
                }
            }
        }
        false
    }

    TARGETS
        .iter()
        .any(|target| search(target, 0, &slots, &mut vec![false; slots.len()]))
}

fn expanded_conditional_identifiers(expression: &str) -> BTreeSet<String> {
    let tokens: Vec<_> = identifiers(expression).collect();
    let mut expanded = BTreeSet::new();
    let mut index = 0;
    while index < tokens.len() {
        if tokens[index] == "defined" {
            index = index.saturating_add(2);
        } else {
            expanded.insert(tokens[index].to_owned());
            index += 1;
        }
    }
    expanded
}

fn macro_definition(rest: &str) -> Option<(&str, &str, Option<Vec<String>>)> {
    let rest = rest.trim_start();
    let name_len = rest
        .find(|value: char| !(value.is_ascii_alphanumeric() || value == '_'))
        .unwrap_or(rest.len());
    let name = &rest[..name_len];
    if name.is_empty() || !is_identifier(name) {
        return None;
    }
    let after_name = &rest[name_len..];
    if let Some(parameters) = after_name.strip_prefix('(') {
        let close = parameters.find(')')?;
        let parameters = parameters[..close].trim();
        let parameters = if parameters.is_empty() {
            Vec::new()
        } else {
            parameters
                .split(',')
                .map(|parameter| {
                    let parameter = parameter.trim();
                    if parameter == "..." {
                        Some("__VA_ARGS__".to_owned())
                    } else if is_identifier(parameter) {
                        Some(parameter.to_owned())
                    } else {
                        parameter
                            .strip_suffix("...")
                            .filter(|name| is_identifier(name))
                            .map(str::to_owned)
                    }
                })
                .collect::<Option<Vec<_>>>()?
        };
        Some((name, after_name[close + 2..].trim_start(), Some(parameters)))
    } else {
        Some((name, after_name.trim_start(), None))
    }
}

fn is_safe_paste_barrier(replacement: &str) -> bool {
    let value = replacement.trim();
    if value.is_empty() {
        return true;
    }
    let bytes = value.as_bytes();
    if matches!(bytes[0], b'"' | b'\'') {
        let quote = bytes[0];
        let mut index = 1;
        while index < bytes.len() {
            if bytes[index] == b'\\' {
                index = index.saturating_add(2);
            } else if bytes[index] == quote {
                return index + 1 == bytes.len();
            } else {
                index += 1;
            }
        }
        return false;
    }
    (bytes[0].is_ascii_digit() || bytes[0] == b'.' && bytes.get(1).is_some_and(u8::is_ascii_digit))
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'\''))
}

fn is_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn identifiers(mut value: &str) -> impl Iterator<Item = &str> {
    std::iter::from_fn(move || {
        let start = value.find(|byte: char| byte.is_ascii_alphabetic() || byte == '_')?;
        value = &value[start..];
        let end = value
            .find(|byte: char| !(byte.is_ascii_alphanumeric() || byte == '_'))
            .unwrap_or(value.len());
        let identifier = &value[..end];
        value = &value[end..];
        Some(identifier)
    })
}

fn contains_volatile_predefined_macro(source: &[u8]) -> Option<bool> {
    let text = normalized_preprocessor_text(source)?;
    Some(contains_volatile_predefined_macro_in_text(&text))
}

struct ManifestCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ManifestCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn byte(&mut self) -> Option<u8> {
        let byte = *self.bytes.get(self.offset)?;
        self.offset += 1;
        Some(byte)
    }

    fn usize(&mut self) -> Option<usize> {
        let end = self.offset.checked_add(8)?;
        let value = u64::from_le_bytes(self.bytes.get(self.offset..end)?.try_into().ok()?);
        self.offset = end;
        usize::try_from(value).ok()
    }

    fn field(&mut self) -> Option<&'a [u8]> {
        let len = self.usize()?;
        let end = self.offset.checked_add(len)?;
        let field = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(field)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn header_manifest_analysis(manifest: &[u8]) -> Option<PasteAnalysis> {
    const PREFIX: &[u8] = b"MAMBA-HDR-MANIFEST-V3\0";
    let Some(bytes) = manifest.strip_prefix(PREFIX) else {
        return Some(PasteAnalysis::default());
    };
    let mut cursor = ManifestCursor::new(bytes);
    let has_paste = cursor.byte()? != 0;
    let has_date_time_fragment = cursor.byte()? != 0;
    PasteAnalysis::decode(cursor.field()?, has_paste, has_date_time_fragment)
}

fn contains_token_paste(text: &str) -> bool {
    text.contains("##") || text.contains("%:%:")
}

fn paste_chain_identifiers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let hash = text[offset..].find("##").map(|index| (offset + index, 2));
        let digraph = text[offset..].find("%:%:").map(|index| (offset + index, 4));
        let Some((paste, width)) = (match (hash, digraph) {
            (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
            (Some(found), None) | (None, Some(found)) => Some(found),
            (None, None) => None,
        }) else {
            break;
        };
        let mut left = paste;
        while left > 0 && bytes[left - 1].is_ascii_whitespace() {
            left -= 1;
        }
        let left_end = left;
        while left > 0 && (bytes[left - 1].is_ascii_alphanumeric() || bytes[left - 1] == b'_') {
            left -= 1;
        }
        if left < left_end && is_identifier(&text[left..left_end]) {
            tokens.push(text[left..left_end].to_owned());
        }
        let mut right = paste + width;
        while right < bytes.len() && bytes[right].is_ascii_whitespace() {
            right += 1;
        }
        let right_start = right;
        while right < bytes.len() && (bytes[right].is_ascii_alphanumeric() || bytes[right] == b'_')
        {
            right += 1;
        }
        if right_start < right && is_identifier(&text[right_start..right]) {
            tokens.push(text[right_start..right].to_owned());
        }
        offset = paste + width;
    }
    tokens
}

fn is_date_time_fragment(identifier: &[u8]) -> bool {
    if identifier.len() < 3 {
        return false;
    }
    [b"__DATE__".as_slice(), b"__TIME__".as_slice()]
        .iter()
        .any(|name| {
            identifier != *name && (name.starts_with(identifier) || name.ends_with(identifier))
        })
}

fn contains_date_time_fragment_in_text(text: &str) -> bool {
    contains_identifier_in_text(text, is_date_time_fragment)
}

fn is_has_include_fragment(identifier: &[u8]) -> bool {
    [
        b"__has_include".as_slice(),
        b"__has_include_next".as_slice(),
    ]
    .iter()
    .any(|name| identifier != *name && (name.starts_with(identifier) || name.ends_with(identifier)))
}

fn contains_has_include_fragment_in_text(text: &str) -> bool {
    contains_identifier_in_text(text, is_has_include_fragment)
}

fn contains_volatile_predefined_macro_in_text(text: &str) -> bool {
    if contains_nv_builtin_volatile_paste(text) {
        return true;
    }
    contains_identifier_in_text(text, |identifier| {
        matches!(
            identifier,
            b"__DATE__" | b"__TIME__" | b"__has_include" | b"__has_include_next"
        )
    })
}

fn contains_nv_builtin_volatile_paste(text: &str) -> bool {
    simple_macro_calls(text)
        .iter()
        .filter(|(name, _)| is_builtin_paste_macro(name))
        .any(|(_, arguments)| {
            let pasted = arguments.concat();
            matches!(
                pasted.as_str(),
                "__DATE__" | "__TIME__" | "__has_include" | "__has_include_next"
            )
        })
}

fn is_builtin_paste_macro(name: &str) -> bool {
    name.starts_with("_NV_PASTE") || name == "_NV_CONCAT_EVAL"
}

fn simple_macro_calls(text: &str) -> Vec<(String, Vec<String>)> {
    let bytes = text.as_bytes();
    let mut offset = 0;
    let mut calls = Vec::new();
    while offset < bytes.len() {
        if bytes[offset] == b'"' || bytes[offset] == b'\'' {
            let quote = bytes[offset];
            offset += 1;
            while offset < bytes.len() {
                if bytes[offset] == b'\\' {
                    offset = (offset + 2).min(bytes.len());
                } else if bytes[offset] == quote {
                    offset += 1;
                    break;
                } else {
                    offset += 1;
                }
            }
            continue;
        }
        if !(bytes[offset].is_ascii_alphabetic() || bytes[offset] == b'_') {
            offset += 1;
            continue;
        }
        let start = offset;
        offset += 1;
        while offset < bytes.len()
            && (bytes[offset].is_ascii_alphanumeric() || bytes[offset] == b'_')
        {
            offset += 1;
        }
        let name_end = offset;
        let mut open = name_end;
        while bytes.get(open).is_some_and(u8::is_ascii_whitespace) {
            open += 1;
        }
        if bytes.get(open) != Some(&b'(') {
            continue;
        }
        let mut close = open + 1;
        let mut depth = 1usize;
        while close < bytes.len() && depth != 0 {
            match bytes[close] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            close += 1;
        }
        if depth != 0 {
            break;
        }
        let close = close - 1;
        let mut arguments = Vec::new();
        let mut argument_start = open + 1;
        let mut argument_depth = 0usize;
        let mut parts = Vec::new();
        for index in open + 1..=close {
            match bytes.get(index).copied() {
                Some(b'(') => argument_depth += 1,
                Some(b')') if argument_depth > 0 => argument_depth -= 1,
                Some(b',') if argument_depth == 0 => {
                    parts.push(&text[argument_start..index]);
                    argument_start = index + 1;
                }
                None | Some(b')') if argument_depth == 0 => {
                    parts.push(&text[argument_start..index]);
                }
                _ => {}
            }
        }
        for argument in parts {
            let argument = argument.trim();
            if argument.is_empty() {
                continue;
            }
            if is_identifier(argument) {
                arguments.push(argument.to_owned());
            } else if let Some(name) = zero_arg_macro_call(argument) {
                arguments.push(format!("@{name}"));
            } else if is_safe_paste_barrier(argument) {
                arguments.push("$".to_owned());
            } else {
                let identifiers = identifiers(argument).collect::<Vec<_>>().join(":");
                arguments.push(format!("?{identifiers}"));
            }
        }
        calls.push((text[start..name_end].to_owned(), arguments));
        // Continue inside the call as well, so a non-paste wrapper cannot
        // hide a nested paste-capable invocation.
        offset = open + 1;
    }
    calls
}

fn zero_arg_macro_call(argument: &str) -> Option<&str> {
    let open = argument.find('(')?;
    let name = argument[..open].trim_end();
    (is_identifier(name) && argument[open + 1..].trim() == ")").then_some(name)
}

fn contains_identifier_in_text(
    text: &str,
    mut matches_identifier: impl FnMut(&[u8]) -> bool,
) -> bool {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LexState {
        Normal,
        String,
        Character,
    }

    let bytes = text.as_bytes();
    let mut state = LexState::Normal;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match state {
            LexState::Normal if byte == b'"' => {
                state = LexState::String;
                index += 1;
            }
            LexState::Normal if byte == b'\'' => {
                state = LexState::Character;
                index += 1;
            }
            LexState::Normal if byte.is_ascii_alphabetic() || byte == b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
                if matches_identifier(&bytes[start..index]) {
                    return true;
                }
            }
            LexState::Normal => index += 1,
            LexState::String | LexState::Character if byte == b'\\' => {
                index = index.saturating_add(2);
            }
            LexState::String if byte == b'"' => {
                state = LexState::Normal;
                index += 1;
            }
            LexState::Character if byte == b'\'' => {
                state = LexState::Normal;
                index += 1;
            }
            LexState::String | LexState::Character => index += 1,
        }
    }
    false
}

fn known_nvrtc_builtin_header(name: &str) -> bool {
    matches!(
        name,
        "__config"
            | "assert.h"
            | "cassert"
            | "cmath"
            | "corecrt.h"
            | "cstdarg"
            | "cstddef"
            | "cstdlib"
            | "cstring"
            | "ctype.h"
            | "cuda_bf16.h"
            | "cuda_fp16.h"
            | "features.h"
            | "functional"
            | "limits.h"
            | "math.h"
            | "new"
            | "stdint.h"
            | "stddef.h"
            | "stdio.h"
            | "stdlib.h"
            | "string.h"
            | "time.h"
            | "type_traits"
            | "utility"
    )
}

fn normalized_preprocessor_text(source: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(source).ok()?;
    if text.contains("??") {
        return None;
    }

    let bytes = text.as_bytes();
    let mut spliced = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' && bytes.get(index + 1) == Some(&b'\n') {
            index += 2;
        } else if bytes[index] == b'\\'
            && bytes.get(index + 1) == Some(&b'\r')
            && bytes.get(index + 2) == Some(&b'\n')
        {
            index += 3;
        } else {
            spliced.push(bytes[index]);
            index += 1;
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LexState {
        Normal,
        String,
        Character,
    }

    let mut output = Vec::with_capacity(spliced.len());
    let mut state = LexState::Normal;
    let mut index = 0;
    while index < spliced.len() {
        let byte = spliced[index];
        match state {
            LexState::Normal if spliced[index..].starts_with(b"//") => {
                output.extend_from_slice(b"  ");
                index += 2;
                while index < spliced.len() && spliced[index] != b'\n' {
                    output.push(b' ');
                    index += 1;
                }
            }
            LexState::Normal if spliced[index..].starts_with(b"/*") => {
                output.extend_from_slice(b"  ");
                index += 2;
                let mut closed = false;
                while index < spliced.len() {
                    if spliced[index..].starts_with(b"*/") {
                        output.extend_from_slice(b"  ");
                        index += 2;
                        closed = true;
                        break;
                    }
                    output.push(if spliced[index] == b'\n' { b'\n' } else { b' ' });
                    index += 1;
                }
                if !closed {
                    return None;
                }
            }
            LexState::Normal => {
                if spliced[index..].starts_with(b"R\"") {
                    return None;
                }
                output.push(byte);
                index += 1;
                state = match byte {
                    b'"' => LexState::String,
                    b'\'' => LexState::Character,
                    _ => LexState::Normal,
                };
            }
            LexState::String | LexState::Character => {
                output.push(byte);
                index += 1;
                if byte == b'\\' {
                    let escaped = *spliced.get(index)?;
                    output.push(escaped);
                    index += 1;
                } else if (state == LexState::String && byte == b'"')
                    || (state == LexState::Character && byte == b'\'')
                {
                    state = LexState::Normal;
                }
            }
        }
    }
    (state == LexState::Normal)
        .then(|| String::from_utf8(output).expect("normalization preserves UTF-8"))
}

fn include_directive(line: &str) -> Option<Option<&str>> {
    let Some((directive, rest)) = preprocessor_directive(line) else {
        return Some(None);
    };
    if directive == "import" || directive.starts_with("include") && directive != "include" {
        return None;
    }
    if directive != "include" {
        return Some(None);
    }
    Some(Some(rest))
}

fn preprocessor_directive(line: &str) -> Option<(&str, &str)> {
    let line = line.trim_start();
    let line = if let Some(line) = line.strip_prefix('#') {
        line
    } else {
        line.strip_prefix("%:")?
    };
    let line = line.trim_start();
    let directive_end = line
        .find(|value: char| !(value.is_ascii_alphanumeric() || value == '_'))
        .unwrap_or(line.len());
    let directive = &line[..directive_end];
    Some((directive, line[directive_end..].trim_start()))
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
pub struct Sm80TcPolicyV3 {
    pub reject_zero_axes: bool,
    pub square_tile_min: usize,
    pub large_tile_min: usize,
    pub forward_min_columns: usize,
    pub forward_thin_max_rows: usize,
    pub forward_thin_below_square_columns: bool,
    pub forward_underfill_max_reduction: usize,
    pub forward_underfill_min_columns: usize,
    pub forward_underfill_wave_numerator: u64,
    pub forward_underfill_wave_denominator: u64,
    pub forward_short_reduction_max: usize,
    pub forward_short_reduction_wave_numerator: u64,
    pub forward_short_reduction_wave_denominator: u64,
    pub tile128_base_wave_numerator: u64,
    pub tile128_base_wave_denominator: u64,
    pub tn_rectangular_min_aspect: u64,
    pub tn_rectangular_wave_numerator: u64,
    pub tn_rectangular_wave_denominator: u64,
    pub backward_tail_min_reduction: usize,
    pub backward_tail_min_tile64_ctas: u64,
    pub deep_split_k_compute_capability: (u32, u32),
    pub deep_split_k_output_columns: usize,
    pub deep_split_k_tail_min_reduction: usize,
    pub deep_split_k_aligned_min_reduction: usize,
    /// The dW stream-K schedule (SM89, under the half policy that permits
    /// the fixed-order fold) takes a 64x64 tile grid of at most this many
    /// waves, as a fraction: the census shows the persistent grid winning up
    /// to one wave and an eighth (144 tiles on 142 multiprocessors, 0.86)
    /// and losing from 1.3 waves on (186 tiles, 1.12).
    pub stream_k_max_wave_numerator: u64,
    pub stream_k_max_wave_denominator: u64,
    /// ... and only when the (tile, slab) units give every CTA of the
    /// persistent grid at least this many 64-row slabs: the fold of one slab
    /// per contributor is paid per tile, and the census shows the schedule
    /// losing wherever a CTA holds fewer than four slabs (the d128 and thin
    /// cells) and winning from thirty-seven on.
    pub stream_k_min_slabs_per_cta: u64,
}

impl Sm80TcPolicyV3 {
    pub const fn current() -> Self {
        Self {
            reject_zero_axes: true,
            square_tile_min: 64,
            large_tile_min: 128,
            forward_min_columns: 32,
            forward_thin_max_rows: 64,
            forward_thin_below_square_columns: true,
            forward_underfill_max_reduction: 512,
            forward_underfill_min_columns: 256,
            forward_underfill_wave_numerator: 3,
            forward_underfill_wave_denominator: 16,
            forward_short_reduction_max: 16,
            forward_short_reduction_wave_numerator: 7,
            forward_short_reduction_wave_denominator: 16,
            tile128_base_wave_numerator: 1,
            tile128_base_wave_denominator: 2,
            tn_rectangular_min_aspect: 4,
            tn_rectangular_wave_numerator: 9,
            tn_rectangular_wave_denominator: 8,
            backward_tail_min_reduction: 256,
            backward_tail_min_tile64_ctas: 4,
            deep_split_k_compute_capability: (8, 9),
            deep_split_k_output_columns: 128,
            deep_split_k_tail_min_reduction: 511,
            deep_split_k_aligned_min_reduction: 1024,
            stream_k_max_wave_numerator: 9,
            stream_k_max_wave_denominator: 8,
            stream_k_min_slabs_per_cta: 32,
        }
    }

    pub fn digest(&self, multiprocessor_count: u32) -> Sha256Digest {
        FramedSha256::new(b"sm80-tc-policy.v3")
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
                b"forward-underfill-max-reduction",
                &(self.forward_underfill_max_reduction as u64).to_le_bytes(),
            )
            .required(
                b"forward-underfill-min-columns",
                &(self.forward_underfill_min_columns as u64).to_le_bytes(),
            )
            .required(
                b"forward-underfill-wave-numerator",
                &self.forward_underfill_wave_numerator.to_le_bytes(),
            )
            .required(
                b"forward-underfill-wave-denominator",
                &self.forward_underfill_wave_denominator.to_le_bytes(),
            )
            .required(
                b"forward-short-reduction-max",
                &(self.forward_short_reduction_max as u64).to_le_bytes(),
            )
            .required(
                b"forward-short-reduction-wave-numerator",
                &self.forward_short_reduction_wave_numerator.to_le_bytes(),
            )
            .required(
                b"forward-short-reduction-wave-denominator",
                &self.forward_short_reduction_wave_denominator.to_le_bytes(),
            )
            .required(
                b"tile128-base-wave-numerator",
                &self.tile128_base_wave_numerator.to_le_bytes(),
            )
            .required(
                b"tile128-base-wave-denominator",
                &self.tile128_base_wave_denominator.to_le_bytes(),
            )
            .required(
                b"tn-rectangular-min-aspect",
                &self.tn_rectangular_min_aspect.to_le_bytes(),
            )
            .required(
                b"tn-rectangular-wave-numerator",
                &self.tn_rectangular_wave_numerator.to_le_bytes(),
            )
            .required(
                b"tn-rectangular-wave-denominator",
                &self.tn_rectangular_wave_denominator.to_le_bytes(),
            )
            .required(
                b"backward-tail-min-reduction",
                &(self.backward_tail_min_reduction as u64).to_le_bytes(),
            )
            .required(
                b"backward-tail-min-tile64-ctas",
                &self.backward_tail_min_tile64_ctas.to_le_bytes(),
            )
            .required(
                b"deep-split-k-cc-major",
                &self.deep_split_k_compute_capability.0.to_le_bytes(),
            )
            .required(
                b"deep-split-k-cc-minor",
                &self.deep_split_k_compute_capability.1.to_le_bytes(),
            )
            .required(
                b"deep-split-k-output-columns",
                &(self.deep_split_k_output_columns as u64).to_le_bytes(),
            )
            .required(
                b"deep-split-k-tail-min-reduction",
                &(self.deep_split_k_tail_min_reduction as u64).to_le_bytes(),
            )
            .required(
                b"deep-split-k-aligned-min-reduction",
                &(self.deep_split_k_aligned_min_reduction as u64).to_le_bytes(),
            )
            .required(
                b"stream-k-max-wave-numerator",
                &self.stream_k_max_wave_numerator.to_le_bytes(),
            )
            .required(
                b"stream-k-max-wave-denominator",
                &self.stream_k_max_wave_denominator.to_le_bytes(),
            )
            .required(
                b"stream-k-min-slabs-per-cta",
                &self.stream_k_min_slabs_per_cta.to_le_bytes(),
            )
            .required(b"multiprocessor-count", &multiprocessor_count.to_le_bytes())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarWavePolicyV1 {
    pub thin_split_wave_numerator: u64,
    pub thin_split_wave_denominator: u64,
    pub slim_split_wave_numerator: u64,
    pub slim_split_wave_denominator: u64,
    pub tn_split_m_wave_numerator: u64,
    pub tn_split_m_wave_denominator: u64,
}

impl ScalarWavePolicyV1 {
    pub const fn current() -> Self {
        Self {
            thin_split_wave_numerator: 1,
            thin_split_wave_denominator: 1,
            slim_split_wave_numerator: 3,
            slim_split_wave_denominator: 1,
            tn_split_m_wave_numerator: 2,
            tn_split_m_wave_denominator: 1,
        }
    }

    pub fn digest(&self, multiprocessor_count: u32) -> Sha256Digest {
        FramedSha256::new(b"scalar-wave-policy.v1")
            .required(
                b"thin-split-wave-numerator",
                &self.thin_split_wave_numerator.to_le_bytes(),
            )
            .required(
                b"thin-split-wave-denominator",
                &self.thin_split_wave_denominator.to_le_bytes(),
            )
            .required(
                b"slim-split-wave-numerator",
                &self.slim_split_wave_numerator.to_le_bytes(),
            )
            .required(
                b"slim-split-wave-denominator",
                &self.slim_split_wave_denominator.to_le_bytes(),
            )
            .required(
                b"tn-split-m-wave-numerator",
                &self.tn_split_m_wave_numerator.to_le_bytes(),
            )
            .required(
                b"tn-split-m-wave-denominator",
                &self.tn_split_m_wave_denominator.to_le_bytes(),
            )
            .required(b"multiprocessor-count", &multiprocessor_count.to_le_bytes())
            .finish()
    }
}

pub fn gemm_dispatch_policy_digest(multiprocessor_count: u32) -> Sha256Digest {
    let tensor_core = Sm80TcPolicyV3::current().digest(multiprocessor_count);
    let scalar = ScalarWavePolicyV1::current().digest(multiprocessor_count);
    FramedSha256::new(b"gemm-dispatch-policy.v4")
        .required(b"sm80-tensor-core-policy", &tensor_core)
        .required(b"scalar-wave-policy", &scalar)
        .required(b"multiprocessor-count", &multiprocessor_count.to_le_bytes())
        .finish()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmPolicy {
    pub batch_invariant: bool,
    pub bi_tensor_cores: bool,
    pub fast_gemm: bool,
    pub cublas_tf32: bool,
    pub f32_triad_policy: F32TriadPolicy,
    pub half_triad_policy: HalfTriadPolicy,
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
pub struct NumericContractSet(u16);

impl NumericContractSet {
    pub const CUBLAS_POLICY_V1: Self = Self(1 << 0);
    pub const TRIAD_SCALAR_FMA_V1: Self = Self(1 << 1);
    pub const TRIAD_MMA_SYNC_V1: Self = Self(1 << 2);
    pub const FIXED_SCALAR_FMA_V1: Self = Self(1 << 3);
    pub const FIXED_MMA_SYNC_V1: Self = Self(1 << 4);
    pub const FIXED_MATVEC_TREE_V1: Self = Self(1 << 5);
    pub const TRIAD_DETERMINISTIC_TF32_V1: Self = Self(1 << 6);
    pub const TRIAD_DETERMINISTIC_TF32_SPLIT_K_V1: Self = Self(1 << 7);
    /// The stream-K half routes: a persistent grid whose per-CTA partials
    /// fold in a fixed order, distinct from the tiled `mma.sync` reduction.
    pub const TRIAD_MMA_SYNC_STREAM_K_V1: Self = Self(1 << 8);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

pub fn route_backend_contract_sets(policy: GemmPolicy) -> (BackendSet, NumericContractSet) {
    if !policy.batch_invariant {
        return (BackendSet::CUBLAS, NumericContractSet::CUBLAS_POLICY_V1);
    }
    let (backends, contracts) = match (policy.bi_gemm_family, policy.bi_tensor_cores) {
        (BiGemmFamily::Triad, false) => (
            BackendSet::TRIAD.union(BackendSet::FIXED),
            NumericContractSet::TRIAD_SCALAR_FMA_V1.union(NumericContractSet::FIXED_MATVEC_TREE_V1),
        ),
        (BiGemmFamily::Triad, true) => (
            BackendSet::TRIAD.union(BackendSet::FIXED),
            NumericContractSet::TRIAD_SCALAR_FMA_V1
                .union(NumericContractSet::TRIAD_MMA_SYNC_V1)
                .union(NumericContractSet::FIXED_MATVEC_TREE_V1),
        ),
        (BiGemmFamily::Fixed, false) => (
            BackendSet::FIXED.union(BackendSet::TRIAD),
            NumericContractSet::FIXED_SCALAR_FMA_V1
                .union(NumericContractSet::FIXED_MMA_SYNC_V1)
                .union(NumericContractSet::TRIAD_SCALAR_FMA_V1),
        ),
        (BiGemmFamily::Fixed, true) => (
            BackendSet::FIXED.union(BackendSet::TRIAD),
            NumericContractSet::FIXED_SCALAR_FMA_V1
                .union(NumericContractSet::FIXED_MMA_SYNC_V1)
                .union(NumericContractSet::TRIAD_SCALAR_FMA_V1)
                .union(NumericContractSet::TRIAD_MMA_SYNC_V1),
        ),
    };
    let contracts = if policy.bi_gemm_family == BiGemmFamily::Triad
        && policy.f32_triad_policy == F32TriadPolicy::AllowDeterministicTf32V1
    {
        contracts
            .union(NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1)
            .union(NumericContractSet::TRIAD_DETERMINISTIC_TF32_SPLIT_K_V1)
    } else {
        contracts
    };
    // The stream-K half routes are reachable only through the tensor-core
    // tier, under either family, and only with the half policy's permission.
    let contracts = if policy.bi_tensor_cores
        && policy.half_triad_policy == HalfTriadPolicy::AllowStreamKFixedOrderV1
    {
        contracts.union(NumericContractSet::TRIAD_MMA_SYNC_STREAM_K_V1)
    } else {
        contracts
    };
    (backends, contracts)
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
    pub device_caps: DeviceCaps,
    pub tuning_table_revision: u16,
    pub schedule_set_revision: u16,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResolvedGemmOp {
    Nn = 1,
    Tn = 2,
    Nt = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PhysicalGemmBackend {
    Sm80Mma16V1 = 1,
    Sm90aWgmmaV1 = 2,
    Sm100Tcgen05V1 = 3,
    Sm120TmaMma16V1 = 4,
    MmaTf32RnaV1 = 5,
    Sm90aWgmmaTf32TmaV1 = 6,
    Sm100Tcgen05Tf32TmaV1 = 7,
    Sm120TmaMmaTf32RnaV1 = 8,
    ScalarFmaV1 = 9,
    MmaTf32RnaSplitK4V1 = 10,
    MmaTf32RnaSplitK2V1 = 11,
    ScalarFmaTnNarrowSplitMPartialV1 = 13,
    ScalarFmaTnSplitMF64ReduceV1 = 15,
    MmaTf32RnaSplitK8V1 = 16,
    ScalarFmaSplitKPartialV1 = 17,
    ScalarFmaSplitKF32ReduceV1 = 18,
    Sm120TmaMmaTf32RnaStreamKV1 = 19,
    Sm120TmaFmaExactV1 = 20,
    Sm89MmaTf32Compact8V1 = 21,
    ScalarFmaSm89FixedCopyPlanV1 = 22,
    Sm89Mma16HalfS3V1 = 23,
}

/// Scoped route epoch for the Ada scalar NN reuse of the already-qualified
/// Fixed CopyPlan kernel. This must not invalidate unrelated tuning-45 routes.
pub(crate) const SM89_FIXED_COPYPLAN_ROUTE_REVISION: u16 = 1;
/// Scoped route epoch for the isolated Ada half-Triad S3 AUTO cohort.
pub(crate) const SM89_HALF_ROUTE_REVISION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResolvedNumericContract {
    ScalarFmaV1 = 1,
    MmaSyncF32V1 = 2,
    WgmmaF32V1 = 3,
    Tcgen05F32V1 = 4,
    MmaTf32RnaV1 = 5,
    Sm90aWgmmaTf32TmaV1 = 6,
    Sm100Tcgen05Tf32TmaV1 = 7,
    Sm120TmaMmaTf32RnaV1 = 8,
    ZeroReductionEpilogueF32V1 = 9,
    MmaTf32RnaSplitK4V1 = 10,
    MmaTf32RnaSplitK2V1 = 11,
    ScalarFmaTnSplitMF64ReduceV1 = 12,
    ScalarFmaTnNarrowSplitMPartialV1 = 13,
    ScalarFmaTnNarrowSplitMF64ReduceV1 = 14,
    MmaTf32RnaSplitK8V1 = 15,
    ScalarFmaSplitKPartialV1 = 16,
    ScalarFmaSplitKF32ReduceV1 = 17,
    Sm120TmaMmaTf32RnaStreamKV1 = 18,
    ScalarFmaFixedSplitFoldV1 = 19,
    /// `mma.sync` FP32 accumulation over a persistent stream-K grid whose
    /// partial slabs fold in a fixed order: bit-stable for a shape on a
    /// device, not bit-equal to the one-CTA-per-tile ladder.
    MmaSyncF32StreamKFixedOrderV1 = 20,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResolvedInstructionFamily {
    ScalarFma = 1,
    MmaSync = 2,
    Wgmma = 3,
    Tcgen05 = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedInstructionShape {
    pub m: u16,
    pub n: u16,
    pub k: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResolvedOperandConversion {
    None = 0,
    RegisterCvtRnaTf32F32V1 = 1,
    TensorMapTfloat32V1 = 2,
    TensorMapUint32ThenCvtRnaTf32F32V1 = 3,
    /// TF32 rounding as one integer add of half an ulp of the ten-bit
    /// mantissa (0x1000) in registers, with no finiteness guard: the tensor
    /// core reads only the upper 19 bits of an operand, so every finite
    /// value and both infinities multiply exactly as after cvt.rna.tf32.f32,
    /// and a NaN whose payload sits in the low bits stays a NaN instead of
    /// turning into an infinity.
    RegisterAddHalfUlpTf32V1 = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResolvedOutputOwnership {
    OneCtaPerOutputTileV1 = 1,
    LastCtaPerOutputTileFixedSplitK4ReduceV1 = 4,
    LastCtaPerOutputTileFixedSplitK2ReduceV1 = 5,
    OneThreadPerOutputElementFixedSplitMReduceV1 = 7,
    LastCtaPerOutputTileFixedSplitK8ReduceV1 = 8,
    OneCtaPerOutputTilePerSplitKPartitionV1 = 9,
    OneThreadPerOutputElementFixedSplitKReduceV1 = 10,
    OwnerCtaPerOutputTileStreamKFixedOrderV1 = 11,
    OwnerCtaPerOutputTileFixedSplitFoldV1 = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedKernelLaunch {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub arguments_digest: Sha256Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedGemmRoute {
    pub op: ResolvedGemmOp,
    pub dtype: PolicyDtype,
    pub backend: PhysicalGemmBackend,
    pub numeric_contract: ResolvedNumericContract,
    pub instruction_family: ResolvedInstructionFamily,
    pub instruction_shape: ResolvedInstructionShape,
    pub operand_conversion: ResolvedOperandConversion,
    pub ownership: ResolvedOutputOwnership,
    pub symbol: &'static str,
    pub module_kind: ModuleKind,
    pub target: CudaTarget,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: DeviceCaps,
    pub shape: (usize, usize, usize),
    pub strides: (usize, usize, usize),
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub launch: ResolvedKernelLaunch,
    pub tensor_map_revision: u16,
    pub tensor_maps_digest: Sha256Digest,
    pub resources_digest: Sha256Digest,
    pub tuning_table_revision: u16,
    pub schedule_revision: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedGemmLaunchSet {
    pub launch_count: u32,
    pub ordered_digest: Sha256Digest,
}

impl ResolvedGemmLaunchSet {
    pub fn ensure_current(self, live: Self, prefix: &str) -> Result<(), String> {
        if self == live {
            Ok(())
        } else {
            Err(format!(
                "{prefix}: resolved GEMM launch set changed since capture; re-capture before replay"
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[doc(hidden)]
/// Physical work category exposed by qualification evidence.
pub enum PhysicalLaunchKind {
    /// A GEMM kernel launch.
    Gemm = 1,
    /// A conversion from the logical input dtype to the execution dtype.
    InputUpcast = 2,
    /// A conversion from the execution dtype to the logical output dtype.
    OutputDowncast = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ResolvedPhysicalKernelLaunch {
    pub(crate) kind: PhysicalLaunchKind,
    pub(crate) symbol: &'static str,
    pub(crate) module_kind: ModuleKind,
    pub(crate) logical_op: ResolvedGemmOp,
    pub(crate) logical_dtype: PolicyDtype,
    pub(crate) execution_dtype: PolicyDtype,
    pub(crate) shape: (usize, usize, usize),
    pub(crate) strides: (usize, usize, usize),
    pub(crate) tile: Option<(u32, u32)>,
    pub(crate) launch: ResolvedKernelLaunch,
    pub(crate) gemm_route: Option<ResolvedGemmRoute>,
}

impl ResolvedPhysicalKernelLaunch {
    pub(crate) fn kind(&self) -> PhysicalLaunchKind {
        self.kind
    }

    pub(crate) fn symbol(&self) -> &'static str {
        self.symbol
    }

    pub(crate) fn module_kind(&self) -> ModuleKind {
        self.module_kind
    }

    pub(crate) fn logical_op(&self) -> ResolvedGemmOp {
        self.logical_op
    }

    pub(crate) fn logical_dtype(&self) -> PolicyDtype {
        self.logical_dtype
    }

    pub(crate) fn execution_dtype(&self) -> PolicyDtype {
        self.execution_dtype
    }

    pub(crate) fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    pub(crate) fn strides(&self) -> (usize, usize, usize) {
        self.strides
    }

    pub(crate) fn tile(&self) -> Option<(u32, u32)> {
        self.tile
    }

    pub(crate) fn launch(&self) -> ResolvedKernelLaunch {
        self.launch
    }

    pub(crate) fn gemm_route(&self) -> Option<ResolvedGemmRoute> {
        self.gemm_route
    }
}

mod physical_observer_private {
    use super::ResolvedPhysicalKernelLaunch;

    pub struct Authority(());

    impl Authority {
        pub(super) fn new() -> Self {
            Self(())
        }
    }

    pub trait Sealed {
        fn record_before_enqueue(
            &mut self,
            authority: &mut Authority,
            launch: ResolvedPhysicalKernelLaunch,
        ) -> Result<(), String>;

        fn invalidate_enqueue(&mut self, authority: &mut Authority);
    }
}

pub(super) trait PhysicalLaunchObserver: physical_observer_private::Sealed {
    const ENABLED: bool;

    fn route_context(&self) -> Option<GemmRouteIdentity>;

    fn argument_identity_digest(
        &self,
        pointer: cudarc::driver::sys::CUdeviceptr,
        required_bytes: u64,
    ) -> Result<Sha256Digest, String>;
}

#[derive(Clone, Copy, Default)]
pub(super) struct NoPhysicalObserver;

const _: [(); 0] = [(); std::mem::size_of::<NoPhysicalObserver>()];

impl PhysicalLaunchObserver for NoPhysicalObserver {
    const ENABLED: bool = false;

    #[inline(always)]
    fn route_context(&self) -> Option<GemmRouteIdentity> {
        None
    }

    #[inline(always)]
    fn argument_identity_digest(
        &self,
        _: cudarc::driver::sys::CUdeviceptr,
        _: u64,
    ) -> Result<Sha256Digest, String> {
        Ok([0; 32])
    }
}

impl physical_observer_private::Sealed for NoPhysicalObserver {
    #[inline(always)]
    fn record_before_enqueue(
        &mut self,
        _: &mut physical_observer_private::Authority,
        _: ResolvedPhysicalKernelLaunch,
    ) -> Result<(), String> {
        Ok(())
    }

    #[inline(always)]
    fn invalidate_enqueue(&mut self, _: &mut physical_observer_private::Authority) {}
}

impl<T: PhysicalLaunchObserver + ?Sized> PhysicalLaunchObserver for &mut T {
    const ENABLED: bool = T::ENABLED;

    #[inline(always)]
    fn route_context(&self) -> Option<GemmRouteIdentity> {
        T::route_context(self)
    }

    #[inline(always)]
    fn argument_identity_digest(
        &self,
        pointer: cudarc::driver::sys::CUdeviceptr,
        required_bytes: u64,
    ) -> Result<Sha256Digest, String> {
        T::argument_identity_digest(self, pointer, required_bytes)
    }
}

impl<T: PhysicalLaunchObserver + ?Sized> physical_observer_private::Sealed for &mut T {
    #[inline(always)]
    fn record_before_enqueue(
        &mut self,
        authority: &mut physical_observer_private::Authority,
        launch: ResolvedPhysicalKernelLaunch,
    ) -> Result<(), String> {
        physical_observer_private::Sealed::record_before_enqueue(*self, authority, launch)
    }

    #[inline(always)]
    fn invalidate_enqueue(&mut self, authority: &mut physical_observer_private::Authority) {
        physical_observer_private::Sealed::invalidate_enqueue(*self, authority);
    }
}

#[derive(Clone, Copy)]
pub(super) struct PhysicalConversionArguments {
    source: cudarc::driver::sys::CUdeviceptr,
    source_bytes: u64,
    destination: cudarc::driver::sys::CUdeviceptr,
    destination_bytes: u64,
}

impl PhysicalConversionArguments {
    pub(super) fn new(
        source: cudarc::driver::sys::CUdeviceptr,
        source_bytes: u64,
        destination: cudarc::driver::sys::CUdeviceptr,
        destination_bytes: u64,
    ) -> Self {
        Self {
            source,
            source_bytes,
            destination,
            destination_bytes,
        }
    }
}

#[derive(Clone, Copy)]
struct PhysicalGemmObservation {
    logical_dtype: PolicyDtype,
    resources_digest: Option<Sha256Digest>,
    route: ResolvedGemmRoute,
}

#[derive(Clone, Copy)]
struct PhysicalConversionObservation {
    kind: PhysicalLaunchKind,
    logical_op: ResolvedGemmOp,
    logical_dtype: PolicyDtype,
    shape: (usize, usize, usize),
    strides: (usize, usize, usize),
    element_count: u64,
    arguments: PhysicalConversionArguments,
}

/// Opaque, allocation-free semantic input for one physical CUDA enqueue.
/// Exactly one private payload is populated by the owning constructors.
#[derive(Clone, Copy)]
pub(super) struct PhysicalLaunchObservation {
    gemm: Option<PhysicalGemmObservation>,
    conversion: Option<PhysicalConversionObservation>,
}

impl PhysicalLaunchObservation {
    pub(super) fn gemm(
        logical_dtype: PolicyDtype,
        resources_digest: Option<Sha256Digest>,
        route: ResolvedGemmRoute,
    ) -> Self {
        Self {
            gemm: Some(PhysicalGemmObservation {
                logical_dtype,
                resources_digest,
                route,
            }),
            conversion: None,
        }
    }

    pub(super) fn conversion(
        kind: PhysicalLaunchKind,
        logical_op: ResolvedGemmOp,
        logical_dtype: PolicyDtype,
        shape: (usize, usize, usize),
        strides: (usize, usize, usize),
        element_count: u64,
        arguments: PhysicalConversionArguments,
    ) -> Self {
        Self {
            gemm: None,
            conversion: Some(PhysicalConversionObservation {
                kind,
                logical_op,
                logical_dtype,
                shape,
                strides,
                element_count,
                arguments,
            }),
        }
    }

    fn resolve<O: PhysicalLaunchObserver>(
        self,
        observer: &O,
        config: cudarc::driver::LaunchConfig,
    ) -> Result<ResolvedPhysicalKernelLaunch, String> {
        match (self.gemm, self.conversion) {
            (
                Some(PhysicalGemmObservation {
                    logical_dtype,
                    resources_digest,
                    route,
                }),
                None,
            ) => {
                let launch = ResolvedKernelLaunch {
                    grid_dim: config.grid_dim,
                    block_dim: config.block_dim,
                    shared_mem_bytes: config.shared_mem_bytes,
                    arguments_digest: route.launch.arguments_digest,
                };
                if launch != route.launch {
                    return Err("physical GEMM launch config changed after route binding".into());
                }
                let mut physical_route = route;
                if let Some(resources_digest) = resources_digest {
                    physical_route.resources_digest = resources_digest;
                }
                Ok(ResolvedPhysicalKernelLaunch {
                    kind: PhysicalLaunchKind::Gemm,
                    symbol: route.symbol,
                    module_kind: route.module_kind,
                    logical_op: route.op,
                    logical_dtype,
                    execution_dtype: route.dtype,
                    shape: route.shape,
                    strides: route.strides,
                    tile: Some(route.tile),
                    launch,
                    gemm_route: Some(physical_route),
                })
            }
            (
                None,
                Some(PhysicalConversionObservation {
                    kind,
                    logical_op,
                    logical_dtype,
                    shape,
                    strides,
                    element_count,
                    arguments,
                }),
            ) => {
                let symbol = match (kind, logical_dtype) {
                    (PhysicalLaunchKind::InputUpcast, PolicyDtype::Bf16) => "cast_bf16_to_f32",
                    (PhysicalLaunchKind::InputUpcast, PolicyDtype::F16) => "cast_f16_to_f32",
                    (PhysicalLaunchKind::OutputDowncast, PolicyDtype::Bf16) => "cast_f32_to_bf16",
                    (PhysicalLaunchKind::OutputDowncast, PolicyDtype::F16) => "cast_f32_to_f16",
                    (PhysicalLaunchKind::Gemm, _) => {
                        return Err("conversion launch cannot use GEMM kind".into());
                    }
                    (_, PolicyDtype::F32) => {
                        return Err("conversion launch does not accept f32 logical dtype".into());
                    }
                };
                let source_identity =
                    observer.argument_identity_digest(arguments.source, arguments.source_bytes)?;
                let destination_identity = observer
                    .argument_identity_digest(arguments.destination, arguments.destination_bytes)?;
                let arguments_digest = FramedSha256::new(b"triad-half-conversion-arguments.v2")
                    .required(b"symbol", symbol.as_bytes())
                    .required(b"kind", &[kind as u8])
                    .required(b"logical-op", &[logical_op as u8])
                    .required(b"logical-dtype", &[logical_dtype as u8])
                    .required(b"element-count", &element_count.to_le_bytes())
                    .required(b"null-pointer-mask", &0_u64.to_le_bytes())
                    .required(b"source-allocation", &source_identity)
                    .required(b"destination-allocation", &destination_identity)
                    .finish();
                Ok(ResolvedPhysicalKernelLaunch {
                    kind,
                    symbol,
                    module_kind: ModuleKind::Fixed,
                    logical_op,
                    logical_dtype,
                    execution_dtype: PolicyDtype::F32,
                    shape,
                    strides,
                    tile: None,
                    launch: ResolvedKernelLaunch {
                        grid_dim: config.grid_dim,
                        block_dim: config.block_dim,
                        shared_mem_bytes: config.shared_mem_bytes,
                        arguments_digest,
                    },
                    gemm_route: None,
                })
            }
            _ => Err("physical launch observation must contain exactly one payload".into()),
        }
    }
}

pub(super) fn resolve_physical_launch_observation<O: PhysicalLaunchObserver>(
    observer: &O,
    observation: PhysicalLaunchObservation,
    config: cudarc::driver::LaunchConfig,
) -> Result<ResolvedPhysicalKernelLaunch, String> {
    observation.resolve(observer, config)
}

pub(super) enum PhysicalCudaLaunchError {
    #[cfg(test)]
    Prepared(&'static str),
    Identity(String),
    Driver(cudarc::driver::result::DriverError),
}

impl PhysicalCudaLaunchError {
    pub(super) fn with_driver_context(self, context: std::fmt::Arguments<'_>) -> String {
        match self {
            #[cfg(test)]
            Self::Prepared(error) => error.to_string(),
            Self::Identity(error) => error,
            Self::Driver(error) => format!("{context}: {error:?}"),
        }
    }
}

impl From<String> for PhysicalCudaLaunchError {
    fn from(error: String) -> Self {
        Self::Identity(error)
    }
}

/// Records one physical node and submits the exact pre-bound cudarc launch.
///
/// The builder already owns the selected `CudaFunction` and ordered ABI
/// arguments. This function alone couples recorder mutation to its real Driver
/// enqueue, using the same copied launch configuration for both identities.
///
/// # Safety
///
/// The caller must uphold [`cudarc::driver::LaunchArgs::launch`]'s function,
/// argument, pointer-lifetime, and asynchronous-mutation requirements.
#[inline(always)]
pub(super) unsafe fn enqueue_with_physical_observation<O: PhysicalLaunchObserver>(
    observer: &mut O,
    builder: &mut cudarc::driver::LaunchArgs<'_>,
    config: cudarc::driver::LaunchConfig,
    observation: Option<PhysicalLaunchObservation>,
) -> Result<(), PhysicalCudaLaunchError> {
    let mut authority = physical_observer_private::Authority::new();
    if O::ENABLED {
        let record = observation
            .ok_or_else(|| "recording CUDA launch has no physical observation".to_string())
            .and_then(|observation| observation.resolve(observer, config))
            .and_then(|launch| {
                physical_observer_private::Sealed::record_before_enqueue(
                    observer,
                    &mut authority,
                    launch,
                )
            });
        if let Err(error) = record {
            physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
            return Err(PhysicalCudaLaunchError::Identity(error));
        }
    }
    match unsafe { builder.launch(config) } {
        Ok(_) => Ok(()),
        Err(error) => {
            if O::ENABLED {
                physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
            }
            Err(PhysicalCudaLaunchError::Driver(error))
        }
    }
}

/// Records one already-resolved physical node and submits its pre-bound launch.
///
/// Resolution, hashing, function selection, argument binding, and backing
/// storage growth must all finish before this boundary is entered. The exact
/// graph package is the only production caller.
///
/// # Safety
///
/// The prepared node, function, argument ABI, and launch configuration must
/// describe the same CUDA enqueue and all referenced allocations must remain
/// live until the stream completes.
#[inline(always)]
pub(super) unsafe fn enqueue_prepared_physical_launch(
    observer: &mut RecordingPhysicalObserver,
    builder: &mut cudarc::driver::LaunchArgs<'_>,
    config: cudarc::driver::LaunchConfig,
    launch: ResolvedPhysicalKernelLaunch,
) -> Result<(), PhysicalCudaLaunchError> {
    let mut authority = physical_observer_private::Authority::new();
    if let Err(error) =
        physical_observer_private::Sealed::record_before_enqueue(observer, &mut authority, launch)
    {
        physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
        return Err(PhysicalCudaLaunchError::Identity(error));
    }
    match unsafe { builder.launch(config) } {
        Ok(_) => Ok(()),
        Err(error) => {
            physical_observer_private::Sealed::invalidate_enqueue(observer, &mut authority);
            Err(PhysicalCudaLaunchError::Driver(error))
        }
    }
}

struct PhysicalTraceRecorder {
    capacity: usize,
    nodes: Vec<ResolvedPhysicalKernelLaunch>,
    enqueue_events: Vec<PhysicalEnqueueEvent>,
    overflowed: bool,
    enqueue_failed: bool,
}

#[derive(Clone, Copy)]
struct PhysicalEnqueueEvent {
    launch: ResolvedPhysicalKernelLaunch,
}

impl PhysicalTraceRecorder {
    fn with_capacity(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("physical launch observer capacity must be nonzero".into());
        }
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(capacity)
            .map_err(|error| format!("reserve physical launch observer capacity: {error}"))?;
        let mut enqueue_events = Vec::new();
        enqueue_events
            .try_reserve_exact(capacity)
            .map_err(|error| format!("reserve physical enqueue provenance capacity: {error}"))?;
        Ok(Self {
            capacity,
            nodes,
            enqueue_events,
            overflowed: false,
            enqueue_failed: false,
        })
    }

    fn validate_start(&self, expected_capacity: usize) -> Result<(), String> {
        if self.capacity != expected_capacity {
            return Err(format!(
                "physical launch observer capacity {} does not match expected capacity {expected_capacity}",
                self.capacity
            ));
        }
        if !self.nodes.is_empty()
            || !self.enqueue_events.is_empty()
            || self.overflowed
            || self.enqueue_failed
        {
            return Err("physical launch observer must be empty before recording starts".into());
        }
        if self.nodes.capacity() != self.capacity || self.enqueue_events.capacity() != self.capacity
        {
            return Err("physical launch observer backing capacity is not exact".into());
        }
        Ok(())
    }

    fn validate_complete(&self) -> Result<(), String> {
        if self.enqueue_failed {
            return Err("physical launch recording includes a failed CUDA enqueue".into());
        }
        if self.overflowed {
            return Err(format!(
                "physical launch recording exceeded its fixed capacity {}",
                self.capacity
            ));
        }
        if self.nodes.is_empty() {
            return Err("physical launch recording produced no nodes".into());
        }
        if self.enqueue_events.len() != self.nodes.len() {
            return Err("physical enqueue provenance count does not match recorded nodes".into());
        }
        Ok(())
    }

    fn record(&mut self, launch: ResolvedPhysicalKernelLaunch) -> Result<(), String> {
        if self.nodes.len() == self.capacity {
            self.overflowed = true;
            return Ok(());
        }
        self.enqueue_events.push(PhysicalEnqueueEvent { launch });
        self.nodes.push(launch);
        Ok(())
    }

    fn invalidate_enqueue(&mut self) {
        self.enqueue_failed = true;
    }

    fn finish(
        self,
        context: GemmRouteIdentity,
        binding: PhysicalGraphBinding,
    ) -> Result<RecordedPhysicalTrace, String> {
        self.validate_complete()?;
        binding
            .context
            .ensure_current(context, "recorded physical trace")?;
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&self.nodes)?;
        let enqueue_provenance = physical_enqueue_provenance_digest(
            self.enqueue_events.iter().map(|event| event.launch),
        )?;
        Ok(RecordedPhysicalTrace {
            context,
            binding,
            launches,
            enqueue_provenance,
            nodes: self.nodes.into_boxed_slice(),
        })
    }
}

type PhysicalArgumentIdentityResolver =
    dyn Fn(cudarc::driver::sys::CUdeviceptr, u64) -> Result<Sha256Digest, String>;

pub(super) struct RecordingPhysicalObserver {
    recorder: PhysicalTraceRecorder,
    context: Option<GemmRouteIdentity>,
    argument_identity: Box<PhysicalArgumentIdentityResolver>,
    replay_provenance: Option<PhysicalReplayProvenance>,
}

impl RecordingPhysicalObserver {
    fn with_argument_identity<Resolve>(
        capacity: usize,
        context: Option<GemmRouteIdentity>,
        resolve: Resolve,
    ) -> Result<Self, String>
    where
        Resolve:
            Fn(cudarc::driver::sys::CUdeviceptr, u64) -> Result<Sha256Digest, String> + 'static,
    {
        Ok(Self {
            recorder: PhysicalTraceRecorder::with_capacity(capacity)?,
            context,
            argument_identity: Box::new(resolve),
            replay_provenance: None,
        })
    }

    fn with_replay_provenance(
        mut self,
        ctx: &GpuCtx,
        managed_epoch: Option<ManagedAllocationEpochStamp>,
    ) -> Result<Self, String> {
        self.replay_provenance = Some(PhysicalReplayProvenance::new(ctx, managed_epoch)?);
        Ok(self)
    }

    pub(super) fn validate_start(&self, expected_capacity: usize) -> Result<(), String> {
        self.recorder.validate_start(expected_capacity)
    }

    pub(super) fn validate_capture_start(
        &self,
        ctx: &GpuCtx,
        manifest: &PreparedPhysicalCaptureManifest,
    ) -> Result<(), String> {
        let provenance = self.replay_provenance.as_ref().ok_or_else(|| {
            "physical graph observer has no prepared replay provenance".to_string()
        })?;
        manifest.validate_capture_binding(provenance.binding, self.recorder.capacity)?;
        self.validate_start(manifest.launch_capacity)?;
        provenance.validate(ctx, "physical graph capture")
    }

    pub(super) fn validate_capture_binding(&self, ctx: &GpuCtx) -> Result<(), String> {
        self.replay_provenance
            .as_ref()
            .ok_or_else(|| "physical graph observer has no prepared replay provenance".to_string())?
            .validate(ctx, "physical graph post-capture")
    }

    fn finish(self, context: GemmRouteIdentity) -> Result<RecordedPhysicalTrace, String> {
        if let Some(bound) = self.context {
            bound.ensure_current(context, "physical launch observer")?;
        }
        let binding = self
            .replay_provenance
            .as_ref()
            .ok_or_else(|| {
                "physical launch observer has no prepared replay provenance".to_string()
            })?
            .binding;
        self.recorder.finish(context, binding)
    }

    fn finish_capture(
        self,
        ctx: &GpuCtx,
        manifest: &PreparedPhysicalCaptureManifest,
    ) -> Result<CapturedPhysicalGraphPlan, String> {
        let Self {
            recorder,
            context,
            argument_identity: _,
            replay_provenance,
        } = self;
        let provenance = replay_provenance.ok_or_else(|| {
            "physical graph observer has no prepared replay provenance".to_string()
        })?;
        provenance.validate(ctx, "physical graph post-capture")?;
        if let Some(bound) = context {
            bound.ensure_current(ctx.gemm_route(), "physical graph observer")?;
        }
        let trace = recorder.finish(ctx.gemm_route(), provenance.binding)?;
        manifest.validate_trace(&trace)?;
        let identity =
            CapturedPhysicalGraphIdentity::from_nodes(provenance.binding, trace.nodes.into_vec())?;
        Ok(CapturedPhysicalGraphPlan {
            identity,
            provenance,
        })
    }
}

pub(super) fn prepare_recording_physical_observer<Resolve>(
    ctx: &GpuCtx,
    capacity: usize,
    managed_epoch: Option<ManagedAllocationEpochStamp>,
    resolve: Resolve,
) -> Result<RecordingPhysicalObserver, String>
where
    Resolve: Fn(cudarc::driver::sys::CUdeviceptr, u64) -> Result<Sha256Digest, String> + 'static,
{
    RecordingPhysicalObserver::with_argument_identity(capacity, Some(ctx.gemm_route()), resolve)?
        .with_replay_provenance(ctx, managed_epoch)
}

pub(super) fn finish_recording_physical_observer(
    observer: RecordingPhysicalObserver,
    context: GemmRouteIdentity,
) -> Result<RecordedPhysicalTrace, String> {
    observer.finish(context)
}

pub(super) fn finish_recording_physical_capture(
    observer: RecordingPhysicalObserver,
    ctx: &GpuCtx,
    manifest: &PreparedPhysicalCaptureManifest,
) -> Result<CapturedPhysicalGraphPlan, String> {
    observer.finish_capture(ctx, manifest)
}

impl PhysicalLaunchObserver for RecordingPhysicalObserver {
    const ENABLED: bool = true;

    fn route_context(&self) -> Option<GemmRouteIdentity> {
        self.context
    }

    fn argument_identity_digest(
        &self,
        pointer: cudarc::driver::sys::CUdeviceptr,
        required_bytes: u64,
    ) -> Result<Sha256Digest, String> {
        (self.argument_identity)(pointer, required_bytes)
    }
}

impl physical_observer_private::Sealed for RecordingPhysicalObserver {
    fn record_before_enqueue(
        &mut self,
        _: &mut physical_observer_private::Authority,
        launch: ResolvedPhysicalKernelLaunch,
    ) -> Result<(), String> {
        self.recorder.record(launch)
    }

    fn invalidate_enqueue(&mut self, _: &mut physical_observer_private::Authority) {
        self.recorder.invalidate_enqueue();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ResolvedPhysicalLaunchSet {
    launch_count: u32,
    ordered_digest: Sha256Digest,
    physical_symbol: Option<&'static str>,
    tile: Option<(u32, u32)>,
}

impl ResolvedPhysicalLaunchSet {
    pub(crate) fn from_nodes(nodes: &[ResolvedPhysicalKernelLaunch]) -> Result<Self, String> {
        if nodes.is_empty() {
            return Err("resolved physical launch set must not be empty".into());
        }
        let launch_count = u32::try_from(nodes.len())
            .map_err(|_| "resolved physical launch count exceeds u32::MAX".to_string())?;
        let mut hash = FramedSha256::new(b"resolved-physical-launch-set.v1")
            .required(b"launch-count-domain", LAUNCH_COUNT_DOMAIN)
            .required(b"launch-count", &launch_count.to_le_bytes());
        for (index, node) in nodes.iter().enumerate() {
            let gemm_route_digest = validate_physical_launch(node)?;
            hash = append_resolved_physical_launch(hash, index, node, gemm_route_digest.as_ref());
        }
        let (physical_symbol, tile) = if let [node] = nodes {
            (Some(node.symbol), node.tile)
        } else {
            (None, None)
        };
        Ok(Self {
            launch_count,
            ordered_digest: hash.finish(),
            physical_symbol,
            tile,
        })
    }

    pub(crate) fn launch_count(&self) -> u32 {
        self.launch_count
    }

    pub(crate) fn ordered_digest(&self) -> Sha256Digest {
        self.ordered_digest
    }

    pub(crate) fn physical_symbol(&self) -> Option<&'static str> {
        self.physical_symbol
    }

    pub(crate) fn tile(&self) -> Option<(u32, u32)> {
        self.tile
    }

    pub(crate) fn validate_nodes(
        self,
        nodes: &[ResolvedPhysicalKernelLaunch],
    ) -> Result<(), String> {
        let actual = Self::from_nodes(nodes)?;
        if self == actual {
            Ok(())
        } else {
            Err("resolved physical launch set does not match its exact ordered nodes".into())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecordedPhysicalTrace {
    context: GemmRouteIdentity,
    binding: PhysicalGraphBinding,
    launches: ResolvedPhysicalLaunchSet,
    enqueue_provenance: Sha256Digest,
    nodes: Box<[ResolvedPhysicalKernelLaunch]>,
}

impl RecordedPhysicalTrace {
    pub(crate) fn context(&self) -> GemmRouteIdentity {
        self.context
    }

    pub(crate) fn nodes(&self) -> &[ResolvedPhysicalKernelLaunch] {
        &self.nodes
    }

    pub(crate) fn launches(&self) -> ResolvedPhysicalLaunchSet {
        self.launches
    }

    pub(crate) fn validate_integrity(&self) -> Result<(), String> {
        self.binding
            .context
            .ensure_current(self.context, "recorded physical trace")?;
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&self.nodes)?;
        if launches != self.launches {
            return Err("recorded physical trace does not match its immutable identity".into());
        }
        if physical_enqueue_provenance_digest(self.nodes.iter().copied())?
            != self.enqueue_provenance
        {
            return Err("recorded physical trace lacks its enqueue-bound provenance".into());
        }
        let manifest = self.manifest();
        manifest.validate_capture_request(self.context, self.nodes.len())?;
        manifest.validate_capture_binding(self.binding, self.nodes.len())?;
        manifest.validate_capture_result(self.context, &self.nodes)
    }

    pub(crate) fn manifest(&self) -> PreparedPhysicalCaptureManifest {
        PreparedPhysicalCaptureManifest::from_nodes(
            self.binding,
            self.enqueue_provenance,
            self.nodes.to_vec(),
        )
        .expect("a recorded physical trace remains a valid non-empty manifest")
    }
}

fn physical_enqueue_event_digest(
    node: &ResolvedPhysicalKernelLaunch,
) -> Result<Sha256Digest, String> {
    let launch = ResolvedPhysicalLaunchSet::from_nodes(std::slice::from_ref(node))?;
    Ok(FramedSha256::new(b"physical-enqueue-event.v1")
        .required(b"launch", &launch.ordered_digest)
        .finish())
}

fn physical_enqueue_provenance_digest(
    events: impl ExactSizeIterator<Item = ResolvedPhysicalKernelLaunch>,
) -> Result<Sha256Digest, String> {
    let mut digest = FramedSha256::new(b"physical-enqueue-provenance.v1")
        .required(b"event-count", &(events.len() as u64).to_le_bytes());
    for (index, launch) in events.enumerate() {
        let event = physical_enqueue_event_digest(&launch)?;
        digest = digest
            .required(b"event-index", &(index as u64).to_le_bytes())
            .required(b"event", &event);
    }
    Ok(digest.finish())
}

#[cfg(test)]
fn recorded_physical_trace_for_test(
    context: GemmRouteIdentity,
    nodes: Vec<ResolvedPhysicalKernelLaunch>,
) -> Result<RecordedPhysicalTrace, String> {
    let mut recorder = PhysicalTraceRecorder::with_capacity(nodes.len().max(1))?;
    for node in nodes {
        recorder.record(node)?;
    }
    recorder.finish(context, PhysicalGraphBinding::new(context, 0x1200, 0x3400))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedPhysicalCaptureManifest {
    context: GemmRouteIdentity,
    binding: PhysicalGraphBinding,
    launches: ResolvedPhysicalLaunchSet,
    enqueue_provenance: Sha256Digest,
    launch_capacity: usize,
    nodes: Box<[ResolvedPhysicalKernelLaunch]>,
}

impl PreparedPhysicalCaptureManifest {
    fn from_nodes(
        binding: PhysicalGraphBinding,
        enqueue_provenance: Sha256Digest,
        nodes: Vec<ResolvedPhysicalKernelLaunch>,
    ) -> Result<Self, String> {
        let context = binding.context;
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&nodes)?;
        let launch_capacity = nodes.len();
        Ok(Self {
            context,
            binding,
            launches,
            enqueue_provenance,
            launch_capacity,
            nodes: nodes.into_boxed_slice(),
        })
    }

    pub(crate) fn launch_capacity(&self) -> usize {
        self.launch_capacity
    }

    pub(crate) fn nodes(&self) -> &[ResolvedPhysicalKernelLaunch] {
        &self.nodes
    }

    pub(crate) fn validate_capture_request(
        &self,
        live_context: GemmRouteIdentity,
        launch_capacity: usize,
    ) -> Result<(), String> {
        self.validate_invariants()?;
        self.context
            .ensure_current(live_context, "prepared physical capture manifest")?;
        if launch_capacity != self.launch_capacity {
            return Err(format!(
                "physical capture launch capacity {launch_capacity} does not match prepared manifest capacity {}",
                self.launch_capacity
            ));
        }
        Ok(())
    }

    fn validate_capture_binding(
        &self,
        live_binding: PhysicalGraphBinding,
        launch_capacity: usize,
    ) -> Result<(), String> {
        self.validate_capture_request(live_binding.context, launch_capacity)?;
        self.binding
            .ensure_current(live_binding, "prepared physical capture manifest")
    }

    pub(crate) fn validate_nodes(
        &self,
        nodes: &[ResolvedPhysicalKernelLaunch],
    ) -> Result<(), String> {
        self.validate_invariants()?;
        self.launches.validate_nodes(nodes)?;
        if self.nodes.as_ref() == nodes {
            Ok(())
        } else {
            Err("physical capture nodes do not exactly match the prepared eager manifest".into())
        }
    }

    pub(crate) fn validate_capture_result(
        &self,
        recorded_context: GemmRouteIdentity,
        nodes: &[ResolvedPhysicalKernelLaunch],
    ) -> Result<(), String> {
        self.context
            .ensure_current(recorded_context, "prepared physical capture manifest")?;
        self.validate_nodes(nodes)
    }

    fn validate_trace(&self, trace: &RecordedPhysicalTrace) -> Result<(), String> {
        self.validate_capture_binding(trace.binding, trace.nodes.len())?;
        self.validate_capture_result(trace.context, &trace.nodes)?;
        if self.enqueue_provenance != trace.enqueue_provenance {
            return Err(
                "physical capture enqueue provenance does not match the eager manifest".into(),
            );
        }
        Ok(())
    }

    fn validate_invariants(&self) -> Result<(), String> {
        self.binding
            .context
            .ensure_current(self.context, "prepared physical capture manifest")?;
        if self.nodes.is_empty() {
            return Err("prepared physical capture manifest must not be empty".into());
        }
        if self.launch_capacity != self.nodes.len() {
            return Err(format!(
                "prepared physical capture manifest has launch capacity {} but stores {} nodes",
                self.launch_capacity,
                self.nodes.len()
            ));
        }
        self.launches.validate_nodes(&self.nodes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalConversionModuleBinding {
    artifact: ArtifactIdentity,
    compiler: CompilerIdentity,
}

impl PhysicalConversionModuleBinding {
    #[cfg(test)]
    fn from_context(context: GemmRouteIdentity) -> Self {
        Self {
            artifact: context.artifacts.fixed,
            compiler: context.compiler,
        }
    }

    fn from_gpu_ctx(ctx: &GpuCtx) -> Self {
        Self {
            artifact: ctx.kernels.artifact_set_identity().fixed,
            compiler: ctx.kernels.compiler_identity(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalGraphBinding {
    context: GemmRouteIdentity,
    context_token: u64,
    stream_token: usize,
    conversion: PhysicalConversionModuleBinding,
}

impl PhysicalGraphBinding {
    #[cfg(test)]
    fn new(context: GemmRouteIdentity, context_token: u64, stream_token: usize) -> Self {
        Self {
            context,
            context_token,
            stream_token,
            conversion: PhysicalConversionModuleBinding::from_context(context),
        }
    }

    fn from_context(ctx: &GpuCtx) -> Self {
        Self {
            context: ctx.gemm_route(),
            context_token: ctx.instance_token(),
            stream_token: ctx.stream_token(),
            conversion: PhysicalConversionModuleBinding::from_gpu_ctx(ctx),
        }
    }

    fn ensure_current(self, live: Self, label: &str) -> Result<(), String> {
        self.context.ensure_current(live.context, label)?;
        if self.context_token != live.context_token {
            return Err(format!(
                "{label}: CUDA context instance changed since capture; re-capture before replay"
            ));
        }
        if self.stream_token != live.stream_token {
            return Err(format!(
                "{label}: CUDA stream changed since capture; re-capture before replay"
            ));
        }
        if self.conversion != live.conversion {
            return Err(format!(
                "{label}: physical conversion module binding changed since capture"
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct CapturedPhysicalGraphIdentity {
    binding: PhysicalGraphBinding,
    launches: ResolvedPhysicalLaunchSet,
    launch_capacity: usize,
    nodes: Box<[ResolvedPhysicalKernelLaunch]>,
}

struct PhysicalAllocationLiveness {
    epoch: ManagedAllocationEpochStamp,
}

struct PhysicalReplayProvenance {
    binding: PhysicalGraphBinding,
    allocation_liveness: PhysicalAllocationLiveness,
    resources: Rc<GpuCtxResources>,
    half_staging: cudarc::driver::sys::CUdeviceptr,
    bi_upcast: [cudarc::driver::sys::CUdeviceptr; 3],
}

impl PhysicalReplayProvenance {
    fn new(
        ctx: &GpuCtx,
        managed_epoch: Option<ManagedAllocationEpochStamp>,
    ) -> Result<Self, String> {
        let provenance = Self {
            binding: PhysicalGraphBinding::from_context(ctx),
            allocation_liveness: PhysicalAllocationLiveness::new(managed_epoch)?,
            resources: ctx.resource_anchor(),
            half_staging: ctx.half_staging_ptr(),
            bi_upcast: ctx.bi_upcast_scratch_ptrs(),
        };
        provenance.validate(ctx, "physical graph preparation")?;
        Ok(provenance)
    }

    fn validate(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
        self.binding
            .ensure_current(PhysicalGraphBinding::from_context(ctx), label)?;
        if !Rc::ptr_eq(&self.resources, &ctx.resource_anchor()) {
            return Err(format!(
                "{label}: CUDA context resources changed since physical graph preparation"
            ));
        }
        self.allocation_liveness.validate()?;
        ctx.ensure_graph_scratch_ptrs(self.half_staging, self.bi_upcast, label)
    }
}

impl PhysicalAllocationLiveness {
    fn new(epoch: Option<ManagedAllocationEpochStamp>) -> Result<Self, String> {
        let epoch = epoch.ok_or_else(|| {
            "physical graph capture requires managed allocation generation provenance".to_string()
        })?;
        let liveness = Self { epoch };
        liveness.validate()?;
        Ok(liveness)
    }

    fn validate(&self) -> Result<(), String> {
        if self.epoch.is_current() {
            Ok(())
        } else {
            Err("physical graph allocation generation is no longer live".into())
        }
    }
}

impl CapturedPhysicalGraphIdentity {
    fn from_nodes(
        binding: PhysicalGraphBinding,
        nodes: Vec<ResolvedPhysicalKernelLaunch>,
    ) -> Result<Self, String> {
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&nodes)?;
        let launch_capacity = nodes.len();
        Ok(Self {
            binding,
            launches,
            launch_capacity,
            nodes: nodes.into_boxed_slice(),
        })
    }

    fn validate(
        &self,
        live_binding: PhysicalGraphBinding,
        allocations_current: bool,
        mut validate_route: impl FnMut(&ResolvedGemmRoute) -> Result<(), String>,
    ) -> Result<(), String> {
        self.binding
            .ensure_current(live_binding, "captured physical graph")?;
        if !allocations_current {
            return Err("captured physical graph allocation generation is no longer live".into());
        }
        if self.launch_capacity == 0 || self.launch_capacity != self.nodes.len() {
            return Err(format!(
                "captured physical graph capacity {} does not match its {} exact nodes",
                self.launch_capacity,
                self.nodes.len()
            ));
        }
        self.launches.validate_nodes(&self.nodes)?;
        for node in &self.nodes {
            match (node.kind, node.gemm_route.as_ref()) {
                (PhysicalLaunchKind::Gemm, Some(route)) => validate_route(route)?,
                (PhysicalLaunchKind::InputUpcast | PhysicalLaunchKind::OutputDowncast, None)
                    if node.module_kind == ModuleKind::Fixed => {}
                _ => {
                    return Err(
                        "captured physical graph node lost its exact route or conversion binding"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct CapturedPhysicalGraphPlan {
    identity: CapturedPhysicalGraphIdentity,
    provenance: PhysicalReplayProvenance,
}

impl CapturedPhysicalGraphPlan {
    pub(crate) fn nodes(&self) -> &[ResolvedPhysicalKernelLaunch] {
        &self.identity.nodes
    }

    pub(crate) fn launches(&self) -> ResolvedPhysicalLaunchSet {
        self.identity.launches
    }

    pub(crate) fn validate_replay(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
        self.provenance.validate(ctx, label)?;
        self.identity
            .validate(PhysicalGraphBinding::from_context(ctx), true, |route| {
                ctx.validate_resolved_gemm_route(route, label)
            })
    }
}

/// Exact physical GEMM inventory prepared by one successful eager execution.
///
/// CUDA Graph capture must reproduce both the presence/absence of GEMM work
/// and the ordered launch-set digest. `route_capacity` is deliberately stored
/// alongside the digest so capture can reserve all recorder storage before
/// entering CUDA's allocation-free capture region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedGemmCaptureManifest {
    pub context: GemmRouteIdentity,
    pub launches: Option<ResolvedGemmLaunchSet>,
    pub route_capacity: usize,
}

impl PreparedGemmCaptureManifest {
    pub(crate) fn new(context: GemmRouteIdentity, launches: Option<ResolvedGemmLaunchSet>) -> Self {
        Self {
            context,
            launches,
            route_capacity: Self::route_capacity_for_launches(launches),
        }
    }

    pub(crate) fn route_capacity_for_launches(launches: Option<ResolvedGemmLaunchSet>) -> usize {
        launches
            .map(|launches| {
                usize::try_from(launches.launch_count)
                    .expect("u32 GEMM launch count always fits in usize")
            })
            .unwrap_or(0)
    }

    pub(crate) fn validate_capture_request(
        &self,
        live_context: GemmRouteIdentity,
        route_capacity: usize,
    ) -> Result<(), String> {
        self.context
            .ensure_current(live_context, "prepared GEMM capture manifest")?;
        let exact_capacity = Self::route_capacity_for_launches(self.launches);
        if self.route_capacity != exact_capacity {
            return Err(format!(
                "prepared GEMM capture manifest has route capacity {} but its ordered launch set requires {exact_capacity}",
                self.route_capacity
            ));
        }
        if route_capacity != self.route_capacity {
            return Err(format!(
                "GEMM capture route capacity {route_capacity} does not match prepared manifest capacity {}",
                self.route_capacity
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_exact_launches(
        required: Option<ResolvedGemmLaunchSet>,
        actual: Option<ResolvedGemmLaunchSet>,
    ) -> Result<(), String> {
        if required == actual {
            Ok(())
        } else {
            Err(
                "captured GEMM ordered launch set does not match its prepared eager manifest"
                    .into(),
            )
        }
    }

    pub(crate) fn validate_capture_result(
        &self,
        recorded_context: GemmRouteIdentity,
        actual: Option<ResolvedGemmLaunchSet>,
    ) -> Result<(), String> {
        self.context
            .ensure_current(recorded_context, "prepared GEMM capture manifest")?;
        Self::ensure_exact_launches(self.launches, actual)
    }
}

/// Exact ordered physical routes observed during one eager GEMM body.
///
/// The trace owns its route storage so qualification code can inspect the
/// actual launches after recording has ended without keeping the recorder
/// active. Its read-only launch set is derived from the same routes; trace
/// construction and mutation remain internal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedGemmTrace {
    context: GemmRouteIdentity,
    launches: Option<ResolvedGemmLaunchSet>,
    routes: Box<[ResolvedGemmRoute]>,
}

impl RecordedGemmTrace {
    pub(crate) fn from_routes(
        context: GemmRouteIdentity,
        routes: Vec<ResolvedGemmRoute>,
    ) -> Result<Self, String> {
        let launches = if routes.is_empty() {
            None
        } else {
            Some(build_resolved_gemm_launch_set(&routes)?)
        };
        Ok(Self {
            context,
            launches,
            routes: routes.into_boxed_slice(),
        })
    }

    /// Returns the physical routes in their actual enqueue order.
    pub fn routes(&self) -> &[ResolvedGemmRoute] {
        &self.routes
    }

    /// Returns the ordered launch count and digest, or `None` when the body
    /// launched no recorded GEMM route.
    pub fn launches(&self) -> Option<ResolvedGemmLaunchSet> {
        self.launches
    }

    pub(crate) fn manifest(&self) -> PreparedGemmCaptureManifest {
        let manifest = PreparedGemmCaptureManifest::new(self.context, self.launches);
        debug_assert_eq!(manifest.route_capacity, self.routes().len());
        manifest
    }
}

pub struct ResolvedGemmLaunchSetBuilder {
    expected_launch_count: u32,
    pushed_launch_count: u32,
    digest: Option<FramedSha256>,
}

impl ResolvedGemmLaunchSetBuilder {
    pub fn new(expected_launch_count: usize) -> Result<Self, String> {
        if expected_launch_count == 0 {
            return Err("resolved GEMM launch set must not be empty".into());
        }
        let expected_launch_count = u32::try_from(expected_launch_count)
            .map_err(|_| "resolved GEMM launch count exceeds u32::MAX".to_string())?;
        let digest = FramedSha256::new(b"resolved-gemm-launch-set.v1")
            .required(b"launch-count-domain", LAUNCH_COUNT_DOMAIN)
            .required(b"launch-count", &expected_launch_count.to_le_bytes());
        Ok(Self {
            expected_launch_count,
            pushed_launch_count: 0,
            digest: Some(digest),
        })
    }

    pub fn push(&mut self, route: &ResolvedGemmRoute) -> Result<(), String> {
        if self.pushed_launch_count == self.expected_launch_count {
            return Err(format!(
                "resolved GEMM launch set exceeds its declared count {}",
                self.expected_launch_count
            ));
        }
        if [
            route.launch.grid_dim.0,
            route.launch.grid_dim.1,
            route.launch.grid_dim.2,
        ]
        .contains(&0)
        {
            return Err(format!(
                "resolved GEMM launch {} has a zero grid dimension",
                route.symbol
            ));
        }
        let block_threads = [
            route.launch.block_dim.0,
            route.launch.block_dim.1,
            route.launch.block_dim.2,
        ]
        .into_iter()
        .try_fold(1_u32, |threads, dimension| threads.checked_mul(dimension))
        .ok_or_else(|| {
            format!(
                "resolved GEMM launch {} block dimensions overflow u32",
                route.symbol
            )
        })?;
        if block_threads == 0 {
            return Err(format!(
                "resolved GEMM launch {} has a zero block dimension",
                route.symbol
            ));
        }
        if block_threads != route.threads {
            return Err(format!(
                "resolved GEMM launch {} block has {block_threads} threads but route records {}",
                route.symbol, route.threads
            ));
        }
        if route.launch.arguments_digest == [0; 32] {
            return Err(format!(
                "resolved GEMM launch {} has a placeholder arguments digest",
                route.symbol
            ));
        }
        let digest = self
            .digest
            .take()
            .ok_or_else(|| "resolved GEMM launch-set builder is already finished".to_string())?;
        self.digest = Some(append_resolved_gemm_route(
            digest,
            usize::try_from(self.pushed_launch_count)
                .expect("u32 launch index always fits in usize"),
            route,
        ));
        self.pushed_launch_count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<ResolvedGemmLaunchSet, String> {
        if self.pushed_launch_count != self.expected_launch_count {
            return Err(format!(
                "resolved GEMM launch set expected {} launches but recorded {}",
                self.expected_launch_count, self.pushed_launch_count
            ));
        }
        let digest = self
            .digest
            .take()
            .ok_or_else(|| "resolved GEMM launch-set builder is already finished".to_string())?;
        Ok(ResolvedGemmLaunchSet {
            launch_count: self.pushed_launch_count,
            ordered_digest: digest.finish(),
        })
    }
}

pub fn build_resolved_gemm_launch_set(
    routes: &[ResolvedGemmRoute],
) -> Result<ResolvedGemmLaunchSet, String> {
    let mut builder = ResolvedGemmLaunchSetBuilder::new(routes.len())?;
    for route in routes {
        builder.push(route)?;
    }
    builder.finish()
}

pub(crate) fn build_zero_reduction_route_identity(
    route: ResolvedGemmRoute,
) -> Result<ResolvedGemmLaunchSet, String> {
    if route.numeric_contract != ResolvedNumericContract::ZeroReductionEpilogueF32V1 {
        return Err("zero-reduction route identity requires the zero epilogue contract".into());
    }
    if route.tensor_map_revision == 0 || route.tensor_maps_digest == [0; 32] {
        return Err("zero-reduction route identity requires revisioned mapless identity".into());
    }
    build_resolved_gemm_launch_set(&[route])
}

pub(crate) struct CapturedGemmGraphPlan {
    pub(crate) context: GemmRouteIdentity,
    pub(crate) launches: ResolvedGemmLaunchSet,
    routes: Box<[ResolvedGemmRoute]>,
}

impl CapturedGemmGraphPlan {
    pub(crate) fn new(
        context: GemmRouteIdentity,
        launches: ResolvedGemmLaunchSet,
        routes: Box<[ResolvedGemmRoute]>,
    ) -> Self {
        Self {
            context,
            launches,
            routes,
        }
    }

    pub(crate) fn routes(&self) -> &[ResolvedGemmRoute] {
        &self.routes
    }

    pub(crate) fn with_validated_launch(
        &self,
        ctx: &GpuCtx,
        label: &str,
        launch: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        self.context.ensure_current(ctx.gemm_route(), label)?;
        let mut live_launch_set = ResolvedGemmLaunchSetBuilder::new(self.routes().len())?;
        for route in self.routes() {
            ctx.validate_resolved_gemm_route(route, label)?;
            live_launch_set.push(route)?;
        }
        self.launches
            .ensure_current(live_launch_set.finish()?, label)?;
        launch()
    }
}

fn append_resolved_gemm_route(
    digest: FramedSha256,
    index: usize,
    route: &ResolvedGemmRoute,
) -> FramedSha256 {
    let accepted_target = route
        .device_caps
        .accepted_target
        .as_ref()
        .map(|target| target.as_str().as_bytes());
    digest
        .required(b"route-index-domain", ROUTE_INDEX_DOMAIN)
        .required(b"route-index", &(index as u64).to_le_bytes())
        .required(b"op", &[route.op as u8])
        .required(b"dtype", &[route.dtype as u8])
        .required(b"backend", &[route.backend as u8])
        .required(b"numeric-contract-domain", NUMERIC_CONTRACT_DOMAIN)
        .required(b"numeric-contract", &[route.numeric_contract as u8])
        .required(b"instruction-family", &[route.instruction_family as u8])
        .required(
            b"instruction-shape-m",
            &route.instruction_shape.m.to_le_bytes(),
        )
        .required(
            b"instruction-shape-n",
            &route.instruction_shape.n.to_le_bytes(),
        )
        .required(
            b"instruction-shape-k",
            &route.instruction_shape.k.to_le_bytes(),
        )
        .required(b"operand-conversion", &[route.operand_conversion as u8])
        .required(b"ownership", &[route.ownership as u8])
        .required(b"symbol", route.symbol.as_bytes())
        .required(b"module-kind", &[route.module_kind as u8])
        .required(b"target", route.target.as_str().as_bytes())
        .required(b"artifact-module", &[route.artifact.module_kind as u8])
        .required(b"artifact-kind", &[route.artifact.artifact_kind as u8])
        .required(b"compile-key", &route.artifact.compile_key)
        .required(b"artifact-digest-domain", ARTIFACT_DIGEST_DOMAIN)
        .required(b"artifact-digest", &route.artifact.artifact_digest)
        .required(b"source-digest", &route.compiler.source_digest)
        .required(b"invocation-digest", &route.compiler.invocation_digest)
        .required(
            b"header-manifest-digest",
            &route.compiler.header_manifest_digest,
        )
        .required(b"compiler-target-domain", COMPILER_TARGET_DOMAIN)
        .required(
            b"compiler-target",
            route.compiler.target.as_str().as_bytes(),
        )
        .required(
            b"nvrtc-major",
            &route.compiler.nvrtc_version.0.to_le_bytes(),
        )
        .required(
            b"nvrtc-minor",
            &route.compiler.nvrtc_version.1.to_le_bytes(),
        )
        .required(b"nvrtc-domain", &route.compiler.nvrtc_library_domain)
        .required(
            b"nvrtc-domain-known",
            &[u8::from(route.compiler.nvrtc_library_known)],
        )
        .required(b"compiler-output-kind", &[route.compiler.output_kind as u8])
        .required(
            b"composer-revision",
            &route.compiler.composer_revision.to_le_bytes(),
        )
        .required(
            b"compiler-revision",
            &route.compiler.compiler_revision.to_le_bytes(),
        )
        .required(
            b"numeric-abi-revision",
            &route.compiler.numeric_abi_revision.to_le_bytes(),
        )
        .required(
            b"compiler-schedule-revision",
            &route.compiler.schedule_revision.to_le_bytes(),
        )
        .required(
            b"device-cc-major",
            &route.device.compute_capability.0.to_le_bytes(),
        )
        .required(
            b"device-cc-minor",
            &route.device.compute_capability.1.to_le_bytes(),
        )
        .required(
            b"device-multiprocessor-count",
            &route.device.multiprocessor_count.to_le_bytes(),
        )
        .required(b"device-target", route.device.target.as_str().as_bytes())
        .required(
            b"driver-api-version",
            &route.device.driver.api_version.to_le_bytes(),
        )
        .required(
            b"driver-build-sources",
            &[route.device.driver.build_sources],
        )
        .required(b"driver-build-digest-domain", DRIVER_BUILD_DIGEST_DOMAIN)
        .required(b"driver-build-digest", &route.device.driver.build_digest)
        .required(
            b"caps-cc-major",
            &route.device_caps.compute_capability.0.to_le_bytes(),
        )
        .required(
            b"caps-cc-minor",
            &route.device_caps.compute_capability.1.to_le_bytes(),
        )
        .required(
            b"caps-nvrtc-major",
            &route.device_caps.nvrtc_version.0.to_le_bytes(),
        )
        .required(
            b"caps-nvrtc-minor",
            &route.device_caps.nvrtc_version.1.to_le_bytes(),
        )
        .optional(b"caps-accepted-target", accepted_target)
        .required(
            b"caps-optin-shared-bytes",
            &route.device_caps.optin_shared_bytes.to_le_bytes(),
        )
        .required(
            b"caps-tensor-map-access",
            &[u8::from(route.device_caps.tensor_map_access)],
        )
        .required(b"shape-m", &(route.shape.0 as u64).to_le_bytes())
        .required(b"shape-k", &(route.shape.1 as u64).to_le_bytes())
        .required(b"shape-n", &(route.shape.2 as u64).to_le_bytes())
        .required(b"stride-a", &(route.strides.0 as u64).to_le_bytes())
        .required(b"stride-b", &(route.strides.1 as u64).to_le_bytes())
        .required(b"stride-c", &(route.strides.2 as u64).to_le_bytes())
        .required(b"tile-m", &route.tile.0.to_le_bytes())
        .required(b"tile-n", &route.tile.1.to_le_bytes())
        .required(b"bk", &route.bk.to_le_bytes())
        .required(b"stages", &[route.stages])
        .required(b"threads", &route.threads.to_le_bytes())
        .required(b"grid-x", &route.launch.grid_dim.0.to_le_bytes())
        .required(b"grid-y", &route.launch.grid_dim.1.to_le_bytes())
        .required(b"grid-z", &route.launch.grid_dim.2.to_le_bytes())
        .required(b"block-x", &route.launch.block_dim.0.to_le_bytes())
        .required(b"block-y", &route.launch.block_dim.1.to_le_bytes())
        .required(b"block-z", &route.launch.block_dim.2.to_le_bytes())
        .required(
            b"dynamic-shared-bytes",
            &route.launch.shared_mem_bytes.to_le_bytes(),
        )
        .required(b"kernel-arguments-digest", &route.launch.arguments_digest)
        .required(
            b"tensor-map-revision",
            &route.tensor_map_revision.to_le_bytes(),
        )
        .required(b"tensor-maps-digest", &route.tensor_maps_digest)
        .required(b"resources-digest", &route.resources_digest)
        .required(
            b"tuning-table-revision",
            &route.tuning_table_revision.to_le_bytes(),
        )
        .required(b"schedule-revision", &route.schedule_revision.to_le_bytes())
}

fn validate_physical_launch(
    node: &ResolvedPhysicalKernelLaunch,
) -> Result<Option<Sha256Digest>, String> {
    if node.symbol.is_empty() || !node.symbol.is_ascii() {
        return Err("resolved physical launch requires a non-empty ASCII CUDA symbol".into());
    }
    if [
        node.launch.grid_dim.0,
        node.launch.grid_dim.1,
        node.launch.grid_dim.2,
    ]
    .contains(&0)
    {
        return Err(format!(
            "resolved physical launch {} has a zero grid dimension",
            node.symbol
        ));
    }
    let block_threads = [
        node.launch.block_dim.0,
        node.launch.block_dim.1,
        node.launch.block_dim.2,
    ]
    .into_iter()
    .try_fold(1_u32, |threads, dimension| threads.checked_mul(dimension))
    .ok_or_else(|| {
        format!(
            "resolved physical launch {} block dimensions overflow u32",
            node.symbol
        )
    })?;
    if block_threads == 0 {
        return Err(format!(
            "resolved physical launch {} has a zero block dimension",
            node.symbol
        ));
    }
    if node.launch.arguments_digest == [0; 32] {
        return Err(format!(
            "resolved physical launch {} has a placeholder argument-layout digest",
            node.symbol
        ));
    }
    let native_half_suffix = match (node.logical_dtype, node.execution_dtype) {
        (PolicyDtype::Bf16, PolicyDtype::Bf16) => Some("_bf16"),
        (PolicyDtype::F16, PolicyDtype::F16) => Some("_f16"),
        _ => None,
    };
    if node.kind == PhysicalLaunchKind::Gemm
        && native_half_suffix.is_some_and(|suffix| !node.symbol.ends_with(suffix))
    {
        return Err(format!(
            "resolved native half GEMM launch {} is missing its exact dtype suffix",
            node.symbol
        ));
    }

    match (node.kind, node.gemm_route) {
        (PhysicalLaunchKind::Gemm, Some(route)) => {
            if route.module_kind != route.artifact.module_kind {
                return Err(format!(
                    "resolved physical GEMM launch {} has an unresolved module owner",
                    node.symbol
                ));
            }
            if node.symbol != route.symbol
                || node.module_kind != route.module_kind
                || node.logical_op != route.op
                || node.execution_dtype != route.dtype
                || node.shape != route.shape
                || node.strides != route.strides
                || node.tile != Some(route.tile)
                || node.launch != route.launch
            {
                return Err(format!(
                    "resolved physical GEMM launch {} contradicts its GEMM route",
                    node.symbol
                ));
            }
            Ok(Some(
                build_resolved_gemm_launch_set(&[route])?.ordered_digest,
            ))
        }
        (PhysicalLaunchKind::Gemm, None) => Err(format!(
            "resolved physical GEMM launch {} has no GEMM route",
            node.symbol
        )),
        (PhysicalLaunchKind::InputUpcast | PhysicalLaunchKind::OutputDowncast, Some(_)) => {
            Err(format!(
                "resolved physical conversion launch {} must not claim a GEMM route",
                node.symbol
            ))
        }
        (PhysicalLaunchKind::InputUpcast | PhysicalLaunchKind::OutputDowncast, None)
            if node.module_kind != ModuleKind::Fixed =>
        {
            Err(format!(
                "resolved physical conversion launch {} has an unresolved module owner",
                node.symbol
            ))
        }
        (PhysicalLaunchKind::InputUpcast | PhysicalLaunchKind::OutputDowncast, None) => Ok(None),
    }
}

fn append_resolved_physical_launch(
    hash: FramedSha256,
    index: usize,
    node: &ResolvedPhysicalKernelLaunch,
    gemm_route_digest: Option<&Sha256Digest>,
) -> FramedSha256 {
    let hash = hash
        .required(
            b"physical-launch-index-domain",
            PHYSICAL_LAUNCH_INDEX_DOMAIN,
        )
        .required(b"physical-launch-index", &(index as u64).to_le_bytes())
        .required(b"kind", &[node.kind as u8])
        .required(b"symbol", node.symbol.as_bytes())
        .required(b"module-kind", &[node.module_kind as u8])
        .required(b"logical-op", &[node.logical_op as u8])
        .required(b"logical-dtype", &[node.logical_dtype as u8])
        .required(b"execution-dtype", &[node.execution_dtype as u8])
        .required(b"shape-m", &(node.shape.0 as u64).to_le_bytes())
        .required(b"shape-k", &(node.shape.1 as u64).to_le_bytes())
        .required(b"shape-n", &(node.shape.2 as u64).to_le_bytes())
        .required(b"stride-a", &(node.strides.0 as u64).to_le_bytes())
        .required(b"stride-b", &(node.strides.1 as u64).to_le_bytes())
        .required(b"stride-output", &(node.strides.2 as u64).to_le_bytes());
    let hash = match node.tile {
        Some((tile_m, tile_n)) => {
            let mut bytes = [0_u8; 8];
            bytes[..4].copy_from_slice(&tile_m.to_le_bytes());
            bytes[4..].copy_from_slice(&tile_n.to_le_bytes());
            hash.optional(b"tile", Some(&bytes))
        }
        None => hash.optional(b"tile", None),
    };
    hash.required(b"grid-x", &node.launch.grid_dim.0.to_le_bytes())
        .required(b"grid-y", &node.launch.grid_dim.1.to_le_bytes())
        .required(b"grid-z", &node.launch.grid_dim.2.to_le_bytes())
        .required(b"block-x", &node.launch.block_dim.0.to_le_bytes())
        .required(b"block-y", &node.launch.block_dim.1.to_le_bytes())
        .required(b"block-z", &node.launch.block_dim.2.to_le_bytes())
        .required(
            b"dynamic-shared-memory-bytes",
            &node.launch.shared_mem_bytes.to_le_bytes(),
        )
        .required(b"argument-layout-digest", &node.launch.arguments_digest)
        .optional(
            b"gemm-route-digest",
            gemm_route_digest.map(Sha256Digest::as_slice),
        )
}

#[cfg(test)]
mod physical_launch_tests {
    use super::*;

    fn physical_launch(
        kind: PhysicalLaunchKind,
        symbol: &'static str,
    ) -> ResolvedPhysicalKernelLaunch {
        ResolvedPhysicalKernelLaunch {
            kind,
            symbol,
            module_kind: ModuleKind::Fixed,
            logical_op: ResolvedGemmOp::Nn,
            logical_dtype: PolicyDtype::Bf16,
            execution_dtype: PolicyDtype::F32,
            shape: (17, 65, 33),
            strides: (80, 48, 48),
            tile: None,
            launch: ResolvedKernelLaunch {
                grid_dim: (5, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
                arguments_digest: [37; 32],
            },
            gemm_route: None,
        }
    }

    fn physical_context() -> GemmRouteIdentity {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new("sm_89").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = |module_kind, seed| ArtifactIdentity {
            module_kind,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [seed; 32],
            artifact_digest: [seed + 1; 32],
        };
        GemmRouteIdentity {
            policy: GemmPolicy {
                batch_invariant: true,
                bi_tensor_cores: true,
                fast_gemm: false,
                cublas_tf32: false,
                f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
                half_triad_policy: HalfTriadPolicy::TiledParityV1,
                bi_gemm_family: BiGemmFamily::Triad,
            },
            backend_set: BackendSet::TRIAD,
            numeric_contracts: NumericContractSet::TRIAD_SCALAR_FMA_V1,
            compiler,
            artifacts: build_artifact_set(&[
                artifact(ModuleKind::Fixed, 5),
                artifact(ModuleKind::TriadScalar, 7),
                artifact(ModuleKind::TriadSm80, 9),
            ])
            .unwrap(),
            policy_revision: POLICY_REVISION,
            policy_hash: [11; 32],
            device: DeviceIdentity {
                compute_capability: (8, 9),
                multiprocessor_count: 142,
                target: CudaTarget::new("sm_89").unwrap(),
                driver: DriverIdentity {
                    api_version: 13_200,
                    build_sources: 1,
                    build_digest: [12; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability: (8, 9),
                nvrtc_version: (13, 2),
                accepted_target: None,
                optin_shared_bytes: 101_376,
                tensor_map_access: false,
            },
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_set_revision: SCHEDULE_REVISION,
            state_capacity: 64,
        }
    }

    #[test]
    fn ada_rna_auto_epoch_rejects_revision39_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 39;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision39 Fixed AUTO graph replay")
            .expect_err("RNA AUTO promotion reused the revision39 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current Fixed AUTO graph replay")
            .unwrap();
    }

    #[test]
    fn ada_finalist_is_an_optional_fourth_artifact_without_changing_three_module_identity() {
        let base = physical_context().artifacts;
        let unchanged = build_artifact_set(&[base.fixed, base.triad_scalar, base.triad_sm80])
            .expect("three-module artifact set");
        assert_eq!(unchanged, base);
        let finalist = ArtifactIdentity {
            module_kind: ModuleKind::TriadSm89Finalist,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [23; 32],
            artifact_digest: [24; 32],
        };
        let with_finalist =
            build_artifact_set(&[base.fixed, base.triad_scalar, base.triad_sm80, finalist])
                .expect("Ada finalist fourth artifact");
        assert_eq!(with_finalist.module_count, 4);
        assert_eq!(with_finalist.specialized, Some(finalist));
        assert_ne!(with_finalist.ordered_digest, base.ordered_digest);
        assert_eq!(with_finalist.fixed, base.fixed);
        assert_eq!(with_finalist.triad_scalar, base.triad_scalar);
        assert_eq!(with_finalist.triad_sm80, base.triad_sm80);
        let foreign = ArtifactIdentity {
            module_kind: ModuleKind::Mamba3Combined,
            ..finalist
        };
        assert!(
            build_artifact_set(&[base.fixed, base.triad_scalar, base.triad_sm80, foreign]).is_err(),
            "an unrelated module occupied the optional triad artifact slot"
        );
        assert!(
            build_artifact_set(&[
                base.fixed,
                base.triad_scalar,
                base.triad_sm80,
                finalist,
                foreign,
            ])
            .is_err(),
            "the artifact set accepted two optional modules"
        );
        assert_eq!(ModuleKind::TriadSm89Finalist as u8, 8);
        assert_eq!(PhysicalGemmBackend::Sm89MmaTf32Compact8V1 as u8, 21);
        assert_eq!(PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1 as u8, 22);
        assert_eq!(PhysicalGemmBackend::Sm89Mma16HalfS3V1 as u8, 23);
    }

    #[test]
    fn ada_rna_toolkit_auto_epoch_rejects_revision40_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 40;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision40 Fixed replay")
            .expect_err("toolkit AUTO reused an old captured route");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current Fixed replay")
            .unwrap();
    }

    #[test]
    fn fresh_sm120_dispatch_epoch_rejects_revision38_graph_identity() {
        let current = physical_context();
        let mut captured = current;
        captured.tuning_table_revision = 38;
        // A host-only table change must invalidate captured dispatch identity
        // even though its compiled modules and numeric/schedule contracts stay.
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        let error = captured
            .ensure_current(current, "revision38 graph replay")
            .expect_err("host-only cohort promotion reused the revision38 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current graph replay")
            .unwrap();
    }

    #[test]
    fn ada_half_auto_epoch_rejects_revision41_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 41;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision41 Fixed replay")
            .expect_err("Ada half AUTO promotion reused a revision41 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current revision45 Fixed replay")
            .unwrap();
    }

    #[test]
    fn ada_half_s3_auto_epoch_rejects_revision42_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 42;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision42 Fixed replay")
            .expect_err("Ada half S3 AUTO promotion reused a revision42 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current revision45 Fixed replay")
            .unwrap();
    }

    #[test]
    fn ada_exact_toolkit_auto_epoch_rejects_revision43_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 43;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision43 Fixed replay")
            .expect_err("Ada exact toolkit AUTO promotion reused a revision43 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current revision45 Fixed replay")
            .unwrap();
    }

    #[test]
    fn ada_finalist_auto_epoch_rejects_revision44_graph_identity() {
        let current = physical_context();
        assert_eq!(current.tuning_table_revision, 45);
        let mut captured = current;
        captured.tuning_table_revision = 44;
        assert_eq!(captured.compiler, current.compiler);
        assert_eq!(captured.artifacts, current.artifacts);
        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
        assert_eq!(
            captured.schedule_set_revision,
            current.schedule_set_revision
        );
        let error = captured
            .ensure_current(current, "revision44 Fixed finalist replay")
            .expect_err("Ada finalist AUTO promotion reused a revision44 graph identity");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "current revision45 Fixed replay")
            .unwrap();
    }

    #[test]
    fn fixed_only_artifact_replacement_invalidates_graph_at_unchanged_epoch45() {
        let captured = physical_context();
        let mut current = captured;
        let mut replaced_fixed = captured.artifacts.fixed;
        replaced_fixed.artifact_digest[0] ^= 1;
        current.artifacts = build_artifact_set(&[
            replaced_fixed,
            captured.artifacts.triad_scalar,
            captured.artifacts.triad_sm80,
        ])
        .unwrap();
        assert_eq!(captured.tuning_table_revision, 45);
        assert_eq!(
            current.tuning_table_revision,
            captured.tuning_table_revision
        );
        assert_eq!(current.compiler, captured.compiler);
        assert_eq!(
            current.artifacts.triad_scalar,
            captured.artifacts.triad_scalar
        );
        assert_eq!(current.artifacts.triad_sm80, captured.artifacts.triad_sm80);
        assert_eq!(
            current.schedule_set_revision,
            captured.schedule_set_revision
        );
        assert_eq!(current.numeric_contracts, captured.numeric_contracts);
        captured
            .ensure_current(captured, "unchanged Fixed graph")
            .unwrap();
        let error = captured
            .ensure_current(current, "replaced Fixed artifact graph")
            .expect_err("Fixed-only source replacement reused a stale captured graph");
        assert!(error.contains("re-capture before replay"));
        current
            .ensure_current(current, "recaptured Fixed graph")
            .unwrap();
    }

    pub(super) fn synthetic_scalar_route() -> ResolvedGemmRoute {
        let context = physical_context();
        ResolvedGemmRoute {
            op: ResolvedGemmOp::Nn,
            dtype: PolicyDtype::F32,
            backend: PhysicalGemmBackend::ScalarFmaV1,
            numeric_contract: ResolvedNumericContract::ScalarFmaV1,
            instruction_family: ResolvedInstructionFamily::ScalarFma,
            instruction_shape: ResolvedInstructionShape { m: 1, n: 1, k: 1 },
            operand_conversion: ResolvedOperandConversion::None,
            ownership: ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            symbol: "gemm_bi_nn",
            module_kind: ModuleKind::TriadScalar,
            target: context.compiler.target,
            artifact: context.artifacts.triad_scalar,
            compiler: context.compiler,
            device: context.device,
            device_caps: context.device_caps,
            shape: (17, 65, 33),
            strides: (80, 48, 48),
            tile: (1, 1),
            bk: 1,
            stages: 1,
            threads: 256,
            launch: ResolvedKernelLaunch {
                grid_dim: (5, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
                arguments_digest: [41; 32],
            },
            tensor_map_revision: 0,
            tensor_maps_digest: [0; 32],
            resources_digest: [42; 32],
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        }
    }

    fn physical_gemm_launch() -> ResolvedPhysicalKernelLaunch {
        let route = synthetic_scalar_route();
        ResolvedPhysicalKernelLaunch {
            kind: PhysicalLaunchKind::Gemm,
            symbol: route.symbol,
            module_kind: route.module_kind,
            logical_op: route.op,
            logical_dtype: PolicyDtype::Bf16,
            execution_dtype: route.dtype,
            shape: route.shape,
            strides: route.strides,
            tile: Some(route.tile),
            launch: route.launch,
            gemm_route: Some(route),
        }
    }

    fn physical_graph_binding() -> PhysicalGraphBinding {
        PhysicalGraphBinding::new(physical_context(), 0x1200, 0x3400)
    }

    fn physical_graph_nodes() -> Vec<ResolvedPhysicalKernelLaunch> {
        vec![
            physical_launch(PhysicalLaunchKind::InputUpcast, "cast_bf16_to_f32"),
            physical_gemm_launch(),
            physical_launch(PhysicalLaunchKind::OutputDowncast, "cast_f32_to_bf16"),
        ]
    }

    #[test]
    fn captured_physical_graph_identity_rejects_exact_contract_mutations() {
        let binding = physical_graph_binding();
        let nodes = physical_graph_nodes();
        assert!(CapturedPhysicalGraphIdentity::from_nodes(binding, Vec::new()).is_err());
        let identity = CapturedPhysicalGraphIdentity::from_nodes(binding, nodes.clone()).unwrap();
        let validated_routes = std::cell::Cell::new(0_usize);
        identity
            .validate(binding, true, |route| {
                assert_eq!(*route, physical_gemm_launch().gemm_route.unwrap());
                validated_routes.set(validated_routes.get() + 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(validated_routes.get(), 1);

        let mut mutations = Vec::new();
        let mut missing = identity.clone();
        missing.nodes = nodes[..2].to_vec().into_boxed_slice();
        mutations.push(missing);
        let mut extra = identity.clone();
        extra.nodes = [nodes.as_slice(), &nodes[2..]].concat().into_boxed_slice();
        mutations.push(extra);
        let mut reordered = identity.clone();
        reordered.nodes.swap(0, 1);
        mutations.push(reordered);
        let mut renamed = identity.clone();
        renamed.nodes[0].symbol = "renamed_cast_bf16_to_f32";
        mutations.push(renamed);
        let mut conversion = identity.clone();
        conversion.nodes[0].kind = PhysicalLaunchKind::OutputDowncast;
        mutations.push(conversion);
        let mut tile = identity.clone();
        tile.nodes[1].tile = Some((64, 64));
        mutations.push(tile);
        let mut arguments = identity.clone();
        arguments.nodes[2].launch.arguments_digest[0] ^= 1;
        mutations.push(arguments);
        let mut capacity = identity.clone();
        capacity.launch_capacity += 1;
        mutations.push(capacity);
        let mut launch_count = identity.clone();
        launch_count.launches.launch_count += 1;
        mutations.push(launch_count);

        for mutation in mutations {
            assert!(
                mutation.validate(binding, true, |_| Ok(())).is_err(),
                "captured physical graph identity mutation was accepted"
            );
        }

        let mut changed_context = binding;
        changed_context.context.state_capacity += 1;
        assert!(
            identity
                .validate(changed_context, true, |_| Ok(()))
                .is_err()
        );
        let mut changed_instance = binding;
        changed_instance.context_token += 1;
        assert!(
            identity
                .validate(changed_instance, true, |_| Ok(()))
                .is_err()
        );
        let mut changed_stream = binding;
        changed_stream.stream_token += 1;
        assert!(identity.validate(changed_stream, true, |_| Ok(())).is_err());
        let mut changed_conversion_binding = binding;
        changed_conversion_binding
            .conversion
            .artifact
            .artifact_digest[0] ^= 1;
        assert!(
            identity
                .validate(changed_conversion_binding, true, |_| Ok(()))
                .is_err()
        );
        assert!(identity.validate(binding, false, |_| Ok(())).is_err());
        assert!(
            identity
                .validate(binding, true, |_| Err("embedded route drift".into()))
                .is_err()
        );
    }

    #[test]
    fn physical_graph_allocation_liveness_rejects_generation_drift() {
        use crate::mamba_ssm::gpu::buffers::{
            managed_allocation_epoch_for_ranges, register_managed_allocation_range,
        };

        assert!(PhysicalAllocationLiveness::new(None).is_err());
        let registration = register_managed_allocation_range(0x7010, 0x8000, 4096).unwrap();
        let epoch = managed_allocation_epoch_for_ranges(0x7010, &[(0x8100, 1024)]).unwrap();
        let liveness = PhysicalAllocationLiveness::new(Some(epoch)).unwrap();
        liveness.validate().unwrap();
        drop(registration);
        assert!(liveness.validate().is_err());
    }

    #[test]
    fn private_recorder_preserves_order_and_sticky_invalidates_failures() {
        let first = physical_launch(PhysicalLaunchKind::InputUpcast, "cast_bf16_to_f32");
        let second = physical_gemm_launch();
        let third = physical_launch(PhysicalLaunchKind::OutputDowncast, "cast_f32_to_bf16");
        let mut recorder = PhysicalTraceRecorder::with_capacity(3).unwrap();
        recorder.validate_start(3).unwrap();
        for launch in [first, second, third] {
            recorder.record(launch).unwrap();
        }
        assert_eq!(recorder.nodes, vec![first, second, third]);
        assert_eq!(recorder.enqueue_events.len(), 3);
        recorder
            .record(first)
            .expect("capture-time overflow handling must not allocate an error");
        let error = recorder
            .validate_complete()
            .expect_err("post-capture validation must reject capacity overflow");
        assert!(error.contains("capacity"), "{error}");

        let mut failed = PhysicalTraceRecorder::with_capacity(1).unwrap();
        failed.record(first).unwrap();
        failed.invalidate_enqueue();
        let error = failed
            .validate_complete()
            .expect_err("a Driver failure must permanently invalidate its recorder");
        assert!(error.contains("failed CUDA enqueue"), "{error}");

        let mut deferred = PhysicalTraceRecorder::with_capacity(1).unwrap();
        let mut malformed = first;
        malformed.kind = PhysicalLaunchKind::Gemm;
        deferred
            .record(malformed)
            .expect("capture-time recording must only copy the enqueue event");
        assert!(
            deferred
                .finish(physical_context(), physical_graph_binding())
                .is_err(),
            "physical hashing and validation must remain deferred until finish"
        );

        let mut oversized = PhysicalTraceRecorder::with_capacity(1).unwrap();
        oversized.nodes.reserve_exact(2);
        let error = oversized
            .validate_start(1)
            .expect_err("capture must reject oversized recorder backing storage");
        assert!(error.contains("backing capacity"), "{error}");
    }

    #[test]
    fn physical_launch_digest_covers_every_semantic_field_and_ordered_position() {
        let first = physical_launch(PhysicalLaunchKind::InputUpcast, "cast_in");
        let second = physical_gemm_launch();
        let ordered = ResolvedPhysicalLaunchSet::from_nodes(&[first, second]).unwrap();
        assert_eq!(ordered.launch_count(), 2);
        assert_ne!(
            ordered,
            ResolvedPhysicalLaunchSet::from_nodes(&[second, first]).unwrap()
        );
        assert_ne!(
            ordered,
            ResolvedPhysicalLaunchSet::from_nodes(&[first, first]).unwrap()
        );

        let baseline = ResolvedPhysicalLaunchSet::from_nodes(&[first]).unwrap();
        let mut mutations = Vec::new();
        let mut value = first;
        value.kind = PhysicalLaunchKind::OutputDowncast;
        mutations.push(value);
        let mut value = first;
        value.symbol = "renamed_cast_in";
        mutations.push(value);
        let mut value = first;
        value.logical_op = ResolvedGemmOp::Tn;
        mutations.push(value);
        let mut value = first;
        value.logical_dtype = PolicyDtype::F16;
        mutations.push(value);
        let mut value = first;
        value.execution_dtype = PolicyDtype::F16;
        mutations.push(value);
        for dimension in 0..3 {
            let mut value = first;
            let mut shape = [value.shape.0, value.shape.1, value.shape.2];
            shape[dimension] += 1;
            value.shape = (shape[0], shape[1], shape[2]);
            mutations.push(value);

            let mut value = first;
            let mut strides = [value.strides.0, value.strides.1, value.strides.2];
            strides[dimension] += 1;
            value.strides = (strides[0], strides[1], strides[2]);
            mutations.push(value);

            let mut value = first;
            let mut grid = [
                value.launch.grid_dim.0,
                value.launch.grid_dim.1,
                value.launch.grid_dim.2,
            ];
            grid[dimension] += 1;
            value.launch.grid_dim = (grid[0], grid[1], grid[2]);
            mutations.push(value);

            let mut value = first;
            let mut block = [
                value.launch.block_dim.0,
                value.launch.block_dim.1,
                value.launch.block_dim.2,
            ];
            block[dimension] += 1;
            value.launch.block_dim = (block[0], block[1], block[2]);
            mutations.push(value);
        }
        let mut value = first;
        value.tile = Some((64, 128));
        mutations.push(value);
        let mut value = first;
        value.launch.shared_mem_bytes += 4;
        mutations.push(value);
        let mut value = first;
        value.launch.arguments_digest[0] ^= 1;
        mutations.push(value);

        for mutation in mutations {
            assert_ne!(
                ResolvedPhysicalLaunchSet::from_nodes(&[mutation]).unwrap(),
                baseline,
                "physical launch mutation was omitted from the ordered digest: {mutation:?}"
            );
        }

        let mut tiled = first;
        tiled.tile = Some((64, 64));
        let tiled_baseline = ResolvedPhysicalLaunchSet::from_nodes(&[tiled]).unwrap();
        for tile in [(128, 64), (64, 128)] {
            let mut changed = tiled;
            changed.tile = Some(tile);
            assert_ne!(
                ResolvedPhysicalLaunchSet::from_nodes(&[changed]).unwrap(),
                tiled_baseline
            );
        }

        let with_gemm_route = physical_gemm_launch();
        let mut changed_route = with_gemm_route;
        changed_route.gemm_route.as_mut().unwrap().resources_digest[0] ^= 1;
        assert_ne!(
            ResolvedPhysicalLaunchSet::from_nodes(&[with_gemm_route]).unwrap(),
            ResolvedPhysicalLaunchSet::from_nodes(&[changed_route]).unwrap()
        );
    }

    #[test]
    fn physical_launch_set_rejects_empty_placeholder_and_incoherent_nodes() {
        assert!(ResolvedPhysicalLaunchSet::from_nodes(&[]).is_err());

        let baseline = physical_launch(PhysicalLaunchKind::InputUpcast, "cast_in");
        let mut invalid = Vec::new();
        let mut empty_symbol = baseline;
        empty_symbol.symbol = "";
        invalid.push(empty_symbol);
        let mut non_ascii_symbol = baseline;
        non_ascii_symbol.symbol = "upcast_\u{00df}";
        invalid.push(non_ascii_symbol);
        let mut wrong_conversion_owner = baseline;
        wrong_conversion_owner.module_kind = ModuleKind::TriadScalar;
        invalid.push(wrong_conversion_owner);
        let mut placeholder_arguments = baseline;
        placeholder_arguments.launch.arguments_digest = [0; 32];
        invalid.push(placeholder_arguments);
        let mut zero_grid = baseline;
        zero_grid.launch.grid_dim.0 = 0;
        invalid.push(zero_grid);
        let mut zero_block = baseline;
        zero_block.launch.block_dim.1 = 0;
        invalid.push(zero_block);
        let mut overflowing_block = baseline;
        overflowing_block.launch.block_dim = (u32::MAX, 2, 1);
        invalid.push(overflowing_block);
        let mut cast_claims_route = baseline;
        cast_claims_route.gemm_route = physical_gemm_launch().gemm_route;
        invalid.push(cast_claims_route);
        let mut gemm_without_route = physical_gemm_launch();
        gemm_without_route.gemm_route = None;
        invalid.push(gemm_without_route);
        let mut unsuffixed_half = physical_gemm_launch();
        unsuffixed_half.logical_dtype = PolicyDtype::Bf16;
        unsuffixed_half.execution_dtype = PolicyDtype::Bf16;
        unsuffixed_half.gemm_route.as_mut().unwrap().dtype = PolicyDtype::Bf16;
        invalid.push(unsuffixed_half);
        let mut post_hoc_tile = physical_gemm_launch();
        post_hoc_tile.tile = Some((64, 64));
        invalid.push(post_hoc_tile);
        let mut unresolved_module_owner = physical_gemm_launch();
        unresolved_module_owner
            .gemm_route
            .as_mut()
            .unwrap()
            .artifact
            .module_kind = ModuleKind::TriadSm80;
        invalid.push(unresolved_module_owner);
        for node in invalid {
            assert!(
                ResolvedPhysicalLaunchSet::from_nodes(&[node]).is_err(),
                "invalid physical launch must fail closed: {node:?}"
            );
        }
    }

    #[test]
    fn native_half_manifest_rejects_unsuffixed_and_post_hoc_selector_metadata() {
        let mut native = physical_gemm_launch();
        native.symbol = "gemm_bi_nn_big_bf16";
        native.logical_dtype = PolicyDtype::Bf16;
        native.execution_dtype = PolicyDtype::Bf16;
        let route = native.gemm_route.as_mut().unwrap();
        route.symbol = native.symbol;
        route.dtype = PolicyDtype::Bf16;
        ResolvedPhysicalLaunchSet::from_nodes(&[native]).unwrap();

        let mut unsuffixed = native;
        unsuffixed.symbol = "gemm_bi_nn_big";
        unsuffixed.gemm_route.as_mut().unwrap().symbol = unsuffixed.symbol;
        assert!(ResolvedPhysicalLaunchSet::from_nodes(&[unsuffixed]).is_err());

        let mut post_hoc_tile = native;
        post_hoc_tile.tile = Some((64, 64));
        assert!(ResolvedPhysicalLaunchSet::from_nodes(&[post_hoc_tile]).is_err());

        let mut coherent_post_hoc_tile = native;
        coherent_post_hoc_tile.tile = Some((64, 64));
        coherent_post_hoc_tile.gemm_route.as_mut().unwrap().tile = (64, 64);
        let mut trace = recorded_physical_trace_for_test(physical_context(), vec![native]).unwrap();
        trace.nodes = vec![coherent_post_hoc_tile].into_boxed_slice();
        trace.launches = ResolvedPhysicalLaunchSet::from_nodes(&trace.nodes).unwrap();
        assert!(trace.validate_integrity().is_err());
    }

    #[test]
    fn physical_launch_set_validation_rejects_count_digest_and_singular_drift() {
        let nodes = [
            physical_launch(PhysicalLaunchKind::InputUpcast, "cast_in"),
            physical_gemm_launch(),
        ];
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&nodes).unwrap();
        assert_eq!(launches.physical_symbol(), None);
        assert_eq!(launches.tile(), None);

        let mut wrong_count = launches;
        wrong_count.launch_count += 1;
        let mut wrong_digest = launches;
        wrong_digest.ordered_digest[0] ^= 1;
        let mut singular_claim = launches;
        singular_claim.physical_symbol = Some(nodes[1].symbol);
        singular_claim.tile = nodes[1].tile;
        for changed in [wrong_count, wrong_digest, singular_claim] {
            assert!(changed.validate_nodes(&nodes).is_err());
        }
    }

    #[test]
    fn physical_launch_trace_and_manifest_reject_empty_nodes() {
        assert!(recorded_physical_trace_for_test(physical_context(), Vec::new()).is_err());
        assert!(
            PreparedPhysicalCaptureManifest::from_nodes(
                physical_graph_binding(),
                [0; 32],
                Vec::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn physical_launch_manifest_rejects_context_capacity_and_node_mutations() {
        let nodes = vec![
            physical_launch(PhysicalLaunchKind::InputUpcast, "cast_in"),
            physical_launch(PhysicalLaunchKind::OutputDowncast, "cast_out"),
        ];
        let context = physical_context();
        let trace = recorded_physical_trace_for_test(context, nodes.clone()).unwrap();
        let manifest = trace.manifest();

        assert_eq!(trace.nodes(), nodes);
        assert_eq!(trace.context(), context);
        assert_eq!(manifest.context, context);
        assert_eq!(manifest.launch_capacity(), nodes.len());
        assert!(
            manifest
                .validate_capture_request(context, nodes.len())
                .is_ok()
        );
        assert!(
            manifest
                .validate_capture_binding(manifest.binding, nodes.len())
                .is_ok()
        );
        assert!(manifest.validate_nodes(&nodes).is_ok());
        assert!(manifest.validate_capture_result(context, &nodes).is_ok());

        let mut changed_context = context;
        changed_context.state_capacity += 1;
        assert_ne!(trace.context(), changed_context);
        assert!(
            manifest
                .validate_capture_request(changed_context, nodes.len())
                .is_err()
        );
        assert!(
            manifest
                .validate_capture_request(context, nodes.len() + 1)
                .is_err()
        );
        assert!(
            manifest
                .validate_capture_result(changed_context, &nodes)
                .is_err()
        );
        let mut changed_instance = manifest.binding;
        changed_instance.context_token += 1;
        assert!(
            manifest
                .validate_capture_binding(changed_instance, nodes.len())
                .is_err()
        );
        let mut changed_stream = manifest.binding;
        changed_stream.stream_token += 1;
        assert!(
            manifest
                .validate_capture_binding(changed_stream, nodes.len())
                .is_err()
        );

        let mut wrong_capacity = manifest.clone();
        wrong_capacity.launch_capacity += 1;
        assert!(
            wrong_capacity
                .validate_capture_request(context, nodes.len() + 1)
                .is_err()
        );
        let mut wrong_count = manifest.clone();
        wrong_count.launches.launch_count += 1;
        assert!(wrong_count.validate_nodes(&nodes).is_err());

        let mut renamed = nodes.clone();
        renamed[0].symbol = "renamed_cast_in";
        for changed in [
            vec![nodes[1], nodes[0]],
            renamed,
            vec![nodes[0]],
            vec![nodes[0], nodes[1], nodes[1]],
        ] {
            assert!(manifest.validate_nodes(&changed).is_err());
            assert!(manifest.validate_capture_result(context, &changed).is_err());
        }
    }

    #[test]
    fn physical_launch_ordered_digest_is_pinned_to_v1_framing() {
        let nodes = [
            physical_launch(PhysicalLaunchKind::InputUpcast, "cast_in"),
            physical_launch(PhysicalLaunchKind::OutputDowncast, "cast_out"),
        ];
        let launches = ResolvedPhysicalLaunchSet::from_nodes(&nodes).unwrap();

        assert_eq!(launches.launch_count(), 2);
        assert_eq!(
            digest_hex(&launches.ordered_digest()),
            "40da3eeb418858775b3377cfa8b01f7716c93d5e6271cf41ca374a7a964d6a46"
        );
    }
}

#[cfg(test)]
mod cache_and_header_tests {
    use super::*;
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use std::cell::Cell;
    #[cfg(target_os = "linux")]
    use std::sync::{Arc, Barrier};

    #[test]
    fn only_the_tuning_table_revision_moved() {
        assert_eq!(NUMERIC_ABI_REVISION, 5);
        assert_eq!(TUNING_TABLE_REVISION, 45);
    }

    #[test]
    fn prepared_capture_manifest_matches_an_exact_optional_ordered_launch_set() {
        let launch = ResolvedGemmLaunchSet {
            launch_count: 2,
            ordered_digest: [7; 32],
        };
        let reordered = ResolvedGemmLaunchSet {
            launch_count: 2,
            ordered_digest: [8; 32],
        };

        assert_eq!(
            PreparedGemmCaptureManifest::route_capacity_for_launches(None),
            0
        );
        assert_eq!(
            PreparedGemmCaptureManifest::route_capacity_for_launches(Some(launch)),
            2
        );
        assert!(
            PreparedGemmCaptureManifest::ensure_exact_launches(Some(launch), Some(launch)).is_ok()
        );
        assert!(
            PreparedGemmCaptureManifest::ensure_exact_launches(Some(launch), Some(reordered))
                .is_err()
        );
        assert!(PreparedGemmCaptureManifest::ensure_exact_launches(None, Some(launch)).is_err());
        assert!(PreparedGemmCaptureManifest::ensure_exact_launches(Some(launch), None).is_err());
    }

    fn scalar_test_route(ctx: &GpuCtx, op: ResolvedGemmOp) -> ResolvedGemmRoute {
        let context = ctx.gemm_route();
        let compiler = ctx.kernels.triad_scalar_compiler_identity();
        let symbol = match op {
            ResolvedGemmOp::Nn => "gemm_bi_nn",
            ResolvedGemmOp::Tn => "gemm_bi_tn",
            ResolvedGemmOp::Nt => "gemm_bi_nt",
        };
        let arguments_digest = FramedSha256::new(b"scalar-test-kernel-arguments.v1")
            .required(b"symbol", symbol.as_bytes())
            .required(b"op", &[op as u8])
            .finish();
        ResolvedGemmRoute {
            op,
            dtype: PolicyDtype::F32,
            backend: PhysicalGemmBackend::ScalarFmaV1,
            numeric_contract: ResolvedNumericContract::ScalarFmaV1,
            instruction_family: ResolvedInstructionFamily::ScalarFma,
            instruction_shape: ResolvedInstructionShape { m: 1, n: 1, k: 1 },
            operand_conversion: ResolvedOperandConversion::None,
            ownership: ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            symbol,
            module_kind: ModuleKind::TriadScalar,
            target: compiler.target,
            artifact: context.artifacts.triad_scalar,
            compiler,
            device: context.device,
            device_caps: context.device_caps,
            shape: (8, 16, 24),
            strides: (16, 24, 24),
            tile: (1, 1),
            bk: 1,
            stages: 1,
            threads: 32,
            launch: ResolvedKernelLaunch {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
                arguments_digest,
            },
            tensor_map_revision: 0,
            tensor_maps_digest: [0; 32],
            resources_digest: [0; 32],
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        }
    }

    #[test]
    fn zero_reduction_identity_is_domain_separated_and_pointer_free() {
        use crate::mamba_ssm::gpu::gemm_bi_triad::{
            TF32_TENSOR_MAP_REVISION, ZERO_REDUCTION_DIGEST_DOMAIN, ZERO_REDUCTION_MAP_REVISION,
        };

        enum PreparedMapMode {
            ZeroReductionV1,
            EncodedV1,
        }
        let modes = [PreparedMapMode::ZeroReductionV1, PreparedMapMode::EncodedV1];
        assert_ne!(
            std::mem::discriminant(&modes[0]),
            std::mem::discriminant(&modes[1])
        );
        let mapless_digest = FramedSha256::new(ZERO_REDUCTION_DIGEST_DOMAIN)
            .required(b"revision", &ZERO_REDUCTION_MAP_REVISION.to_le_bytes())
            .finish();
        let encoded_digest = FramedSha256::new(b"tf32-encoded-maps.v1")
            .required(b"revision", &TF32_TENSOR_MAP_REVISION.to_le_bytes())
            .finish();
        let mut zero = super::physical_launch_tests::synthetic_scalar_route();
        zero.numeric_contract = ResolvedNumericContract::ZeroReductionEpilogueF32V1;
        zero.tensor_map_revision = ZERO_REDUCTION_MAP_REVISION;
        zero.tensor_maps_digest = mapless_digest;
        zero.launch.arguments_digest = FramedSha256::bytes(b"zero-reduction-args");
        let zero_identity = build_zero_reduction_route_identity(zero).unwrap();
        let mut encoded = zero;
        encoded.numeric_contract = ResolvedNumericContract::ScalarFmaV1;
        encoded.tensor_map_revision = TF32_TENSOR_MAP_REVISION;
        encoded.tensor_maps_digest = encoded_digest;
        let encoded_identity = build_resolved_gemm_launch_set(&[encoded]).unwrap();
        assert_eq!(zero.tensor_map_revision, ZERO_REDUCTION_MAP_REVISION);
        assert_ne!(mapless_digest, encoded_digest);
        assert_ne!(zero_identity, encoded_identity);
    }

    fn graph_test_context() -> GpuCtx {
        let device = GpuDevice::new(0).expect("CUDA device for graph-plan test");
        let ctx = GpuCtx::new(&device).expect("GPU context for graph-plan test");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        ctx
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn fixed_policy_accepts_only_declared_triad_fallback_contracts() {
        let ctx = graph_test_context();
        ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            ctx.validate_resolved_gemm_route(&scalar_test_route(&ctx, op), "fixed scalar")
                .unwrap();
        }

        let mut tf32 = scalar_test_route(&ctx, ResolvedGemmOp::Nn);
        tf32.numeric_contract = ResolvedNumericContract::MmaTf32RnaV1;
        let error = ctx
            .validate_resolved_gemm_route(&tf32, "fixed TF32")
            .expect_err("Fixed policy must not admit a Triad TF32 contract");
        assert!(error.contains("numeric contract"), "{error}");
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn validated_launch_accepts_an_unchanged_plan_once() {
        let ctx = graph_test_context();
        let guard = ctx.begin_gemm_route_recording(1).unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))
            .unwrap();
        let plan = guard.finish().unwrap();
        assert_eq!(plan.routes().len(), 1);
        assert_eq!(plan.routes()[0].op, ResolvedGemmOp::Nn);
        let attempts = Cell::new(0_u64);

        plan.with_validated_launch(&ctx, "unchanged graph plan", || {
            attempts.set(attempts.get() + 1);
            Ok(())
        })
        .unwrap();

        assert_eq!(attempts.get(), 1);
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn validated_launch_rejects_policy_mutation_before_closure() {
        let ctx = graph_test_context();
        let guard = ctx.begin_gemm_route_recording(2).unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))
            .unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))
            .unwrap();
        let mut plan = guard.finish().unwrap();
        let attempts = Cell::new(0_u64);

        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        let result = plan.with_validated_launch(&ctx, "mutated graph policy", || {
            attempts.set(attempts.get() + 1);
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(attempts.get(), 0);

        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        plan.routes.swap(0, 1);
        let result = plan.with_validated_launch(&ctx, "mutated route order", || {
            attempts.set(attempts.get() + 1);
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(attempts.get(), 0);
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn eager_manifest_records_exact_routes_and_error_paths_clear_the_recorder() {
        let ctx = graph_test_context();
        let trace = ctx
            .record_eager_gemm_trace(|| {
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))?;
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))?;
                Ok(())
            })
            .unwrap();
        assert_eq!(trace.routes().len(), 2);
        assert_eq!(trace.routes()[0].op, ResolvedGemmOp::Nn);
        assert_eq!(trace.routes()[1].op, ResolvedGemmOp::Tn);
        let manifest = trace.manifest();
        assert_eq!(manifest.route_capacity, 2);
        assert!(manifest.launches.is_some());
        let wrapped_manifest = ctx
            .record_eager_gemm_manifest(|| {
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))?;
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))?;
                Ok(())
            })
            .unwrap();
        assert_eq!(wrapped_manifest, manifest);

        let error = ctx
            .record_eager_gemm_manifest(|| Err("expected eager failure".to_string()))
            .unwrap_err();
        assert_eq!(error, "expected eager failure");
        drop(ctx.begin_gemm_route_recording(0).unwrap());

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ctx.record_eager_gemm_manifest(|| -> Result<(), String> {
                panic!("expected eager panic")
            });
        }));
        assert!(panic.is_err());
        drop(ctx.begin_gemm_route_recording(0).unwrap());
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn fixed_capture_recording_rejects_route_order_drift_and_remains_reusable() {
        let ctx = graph_test_context();
        let manifest = ctx
            .record_eager_gemm_manifest(|| {
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))?;
                ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))?;
                Ok(())
            })
            .unwrap();
        assert!(
            manifest
                .validate_capture_request(ctx.gemm_route(), manifest.route_capacity - 1)
                .is_err()
        );

        let recording = ctx
            .begin_gemm_route_recording(manifest.route_capacity)
            .unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))
            .unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))
            .unwrap();
        assert!(
            recording
                .finish_against_manifest(&manifest)
                .unwrap()
                .is_some()
        );

        let recording = ctx
            .begin_gemm_route_recording(manifest.route_capacity)
            .unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Tn))
            .unwrap();
        ctx.record_resolved_gemm_route(scalar_test_route(&ctx, ResolvedGemmOp::Nn))
            .unwrap();
        assert!(recording.finish_against_manifest(&manifest).is_err());
        drop(ctx.begin_gemm_route_recording(0).unwrap());
    }

    #[cfg(target_os = "linux")]
    fn trusted_tempdir() -> tempfile::TempDir {
        let home = std::env::var_os("HOME").expect("HOME for cache security tests");
        let home = std::fs::canonicalize(home).expect("canonical HOME for cache security tests");
        tempfile::Builder::new()
            .prefix("mamba-cache-test-")
            .tempdir_in(home)
            .expect("trusted temporary directory")
    }

    #[cfg(target_os = "linux")]
    fn key() -> Sha256Digest {
        [7; 32]
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn private_cache_publish_and_read_round_trip() {
        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).expect("private cache directory");
        let path = directory.join("entry.bin");
        publish_cache(&path, key(), ArtifactKind::Ptx, b"payload");
        let hit = read_cache(&path, key(), ArtifactKind::Ptx).expect("cache hit");
        assert_eq!(hit.payload, b"payload");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_truncation_trailing_bytes_and_oversize() {
        use std::os::unix::fs::PermissionsExt;

        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let path = directory.join("entry.bin");
        let encoded = CacheEnvelope::encode(key(), ArtifactKind::Ptx, b"payload");

        std::fs::write(&path, &encoded[..encoded.len() - 1]).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());

        let mut trailing = encoded;
        trailing.push(0);
        std::fs::write(&path, trailing).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());

        let file = std::fs::File::create(&path).unwrap();
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .unwrap();
        file.set_len(MAX_CACHE_ENTRY_BYTES + 1).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_symlinks_and_untrusted_directories() {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let real = directory.join("real.bin");
        let link = directory.join("link.bin");
        std::fs::write(
            &real,
            CacheEnvelope::encode(key(), ArtifactKind::Ptx, b"payload"),
        )
        .unwrap();
        symlink(&real, &link).unwrap();
        assert!(read_cache(&link, key(), ArtifactKind::Ptx).is_none());

        let handle = open_cache_directory(&directory, false).unwrap();
        let mut stat = fstat(handle.fd.as_raw_fd()).unwrap();
        stat.st_uid = effective_uid().wrapping_add(1);
        assert!(!trusted_directory_stat(&stat, true));
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(open_cache_directory(&directory, false).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_an_intermediate_symlink() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = trusted_tempdir();
        let real = root.path().join("real");
        let cache = real.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
        let alias = root.path().join("alias");
        symlink(&real, &alias).unwrap();

        assert!(prepare_private_cache_dir(&alias.join("cache")).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_a_writable_ancestor() {
        use std::os::unix::fs::PermissionsExt;

        let root = trusted_tempdir();
        let writable = root.path().join("writable");
        let cache = writable.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::set_permissions(&writable, std::fs::Permissions::from_mode(0o777)).unwrap();
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();

        assert!(prepare_private_cache_dir(&cache).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_relative_paths() {
        assert!(open_cache_directory(Path::new("relative/cache"), false).is_none());
        assert!(prepare_private_cache_dir(Path::new("relative/cache")).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn retained_directory_handle_survives_path_replacement() {
        let root = trusted_tempdir();
        let cache = root.path().join("cache");
        prepare_private_cache_dir(&cache).unwrap();
        let directory = open_cache_directory(&cache, false).unwrap();
        let displaced = root.path().join("displaced");
        std::fs::rename(&cache, &displaced).unwrap();
        prepare_private_cache_dir(&cache).unwrap();

        let name = std::ffi::CString::new("entry.bin").unwrap();
        let encoded = CacheEnvelope::encode(key(), ArtifactKind::Ptx, b"anchored");
        publish_cache_at(&directory, &name, &encoded);

        assert!(displaced.join("entry.bin").is_file());
        assert!(!cache.join("entry.bin").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_non_private_or_multiply_linked_entries() {
        use std::os::unix::fs::PermissionsExt;

        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let path = directory.join("entry.bin");
        publish_cache(&path, key(), ArtifactKind::Ptx, b"payload");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::hard_link(&path, directory.join("second-link.bin")).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_rejects_a_private_fifo_without_blocking() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;
        use std::time::Duration;

        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let path = directory.join("entry.bin");
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let worker_path = path.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            tx.send(read_cache(&worker_path, key(), ArtifactKind::Ptx).is_none())
                .unwrap();
        });
        let result = rx.recv_timeout(Duration::from_millis(250));
        if result.is_err() {
            let fd = unsafe {
                libc::open(
                    c_path.as_ptr(),
                    libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC,
                )
            };
            if fd >= 0 {
                unsafe { libc::close(fd) };
            }
        }
        worker.join().unwrap();
        assert_eq!(result, Ok(true), "cache read blocked while opening a FIFO");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_size_limit_includes_the_envelope_without_overflow() {
        let largest_payload = MAX_CACHE_ENTRY_BYTES as usize - CACHE_ENVELOPE_HEADER_BYTES;
        assert_eq!(
            cache_entry_len(largest_payload),
            Some(MAX_CACHE_ENTRY_BYTES as usize)
        );
        assert!(cache_entry_len(largest_payload + 1).is_none());
        assert!(cache_entry_len(usize::MAX).is_none());
    }

    #[test]
    fn nvrtc_toolchain_domain_hashes_runtime_and_builtins_in_order() {
        let root = tempfile::tempdir().unwrap();
        let runtime = root.path().join("libnvrtc.so");
        let builtins = root.path().join("libnvrtc-builtins.so");
        std::fs::write(&runtime, b"runtime-a").unwrap();
        std::fs::write(&builtins, b"builtins-a").unwrap();
        let baseline = nvrtc_library_domain_from_paths(&runtime, &builtins).unwrap();
        assert_ne!(
            baseline,
            nvrtc_library_domain_from_paths(&builtins, &runtime).unwrap()
        );

        std::fs::write(&runtime, b"runtime-b").unwrap();
        let runtime_changed = nvrtc_library_domain_from_paths(&runtime, &builtins).unwrap();
        assert_ne!(baseline, runtime_changed);

        std::fs::write(&runtime, b"runtime-a").unwrap();
        std::fs::write(&builtins, b"builtins-b").unwrap();
        let builtins_changed = nvrtc_library_domain_from_paths(&runtime, &builtins).unwrap();
        assert_ne!(baseline, builtins_changed);
        assert_ne!(runtime_changed, builtins_changed);

        let key_for = |domain: Vec<u8>| {
            CompileKeyMaterial {
                module_kind: ModuleKind::Fixed,
                source: b"source".to_vec(),
                target: b"sm_89".to_vec(),
                argv: vec![b"--gpu-architecture=sm_89".to_vec()],
                header_manifest: Some(vec![]),
                nvrtc_version: (13, 2),
                nvrtc_library_domain: Some(domain),
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            }
            .digest()
            .unwrap()
        };
        assert_ne!(key_for(baseline), key_for(runtime_changed.clone()));
        assert_ne!(key_for(runtime_changed), key_for(builtins_changed));
    }

    #[test]
    fn nvrtc_toolchain_domain_requires_both_libraries() {
        let root = tempfile::tempdir().unwrap();
        let runtime = root.path().join("libnvrtc.so");
        let builtins = root.path().join("libnvrtc-builtins.so");
        std::fs::write(&runtime, b"runtime").unwrap();
        assert!(nvrtc_library_domain_from_paths(&runtime, &builtins).is_none());
        std::fs::remove_file(&runtime).unwrap();
        std::fs::write(&builtins, b"builtins").unwrap();
        assert!(nvrtc_library_domain_from_paths(&runtime, &builtins).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn concurrent_publication_never_exposes_partial_entries() {
        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let path = Arc::new(directory.join("entry.bin"));
        let barrier = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for index in 0..4u8 {
            let path = path.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                let payload = vec![index; 4096];
                publish_cache(&path, key(), ArtifactKind::Ptx, &payload);
            }));
        }
        barrier.wait();
        for _ in 0..100 {
            if let Some(hit) = read_cache(&path, key(), ArtifactKind::Ptx) {
                assert_eq!(hit.payload.len(), 4096);
                assert!(hit.payload.iter().all(|value| *value == hit.payload[0]));
            }
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn invalid_read_does_not_remove_a_replacement() {
        use std::os::unix::fs::PermissionsExt;

        let root = trusted_tempdir();
        let directory = root.path().join("cache");
        prepare_private_cache_dir(&directory).unwrap();
        let path = directory.join("entry.bin");
        std::fs::write(&path, b"bad").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_cache(&path, key(), ArtifactKind::Ptx).is_none());
        publish_cache(&path, key(), ArtifactKind::Ptx, b"good");
        assert_eq!(
            read_cache(&path, key(), ArtifactKind::Ptx).unwrap().payload,
            b"good"
        );
    }

    #[test]
    fn include_next_and_header_mutation_fail_the_closure_guard() {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        assert!(
            header_manifest(
                b"#include_next <value.cuh>",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );
        assert!(
            header_manifest(
                b"#include_magic <value.cuh>",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );

        let header = root.path().join("value.cuh");
        std::fs::write(&header, b"first").unwrap();
        let source = b"#include \"value.cuh\"";
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(manifest.is_some());
        std::fs::write(header, b"second").unwrap();
        assert!(!header_manifest_is_current(
            source,
            &[include_root],
            &manifest
        ));
    }

    fn assert_literal_header_is_tracked(source: &[u8]) {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let header = root.path().join("value.cuh");
        std::fs::write(&header, b"first").unwrap();
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(manifest.is_some(), "literal include was not resolved");
        std::fs::write(header, b"second").unwrap();
        assert!(!header_manifest_is_current(
            source,
            &[include_root],
            &manifest
        ));
    }

    #[test]
    fn comments_cannot_hide_a_literal_include() {
        assert_literal_header_is_tracked(b"#/**/include/**/\"value.cuh\"");
    }

    #[test]
    fn digraph_cannot_hide_a_literal_include() {
        assert_literal_header_is_tracked(b"%:include \"value.cuh\"");
    }

    #[test]
    fn line_splicing_cannot_hide_a_literal_include() {
        assert_literal_header_is_tracked(b"#inc\\\nlude \\\n\"value.cuh\"");
    }

    #[test]
    fn filesystem_dependent_preprocessor_forms_disable_the_manifest() {
        for source in [
            b"#import \"value.cuh\"".as_slice(),
            b"#include_next \"value.cuh\"",
            b"#include HEADER",
            b"#if __has_include(\"value.cuh\")\n#endif",
            b"#if __has_include_next(\"value.cuh\")\n#endif",
            b"#include \"value.cuh\" trailing",
            b"#include_magic \"value.cuh\"",
            b"\xff#include \"value.cuh\"",
        ] {
            assert!(
                header_manifest(source, &[]).is_none(),
                "ambiguous source was accepted: {source:?}"
            );
        }
    }

    #[test]
    fn token_pasted_has_include_forms_disable_the_manifest() {
        for source in [
            b"#define JOIN(a, b) a ## b\n#if JOIN(__has_, include)(\"probe.h\")\n#endif".as_slice(),
            b"#define JOIN(a, b) a ## b\n#if JOIN(__has_include_, next)(\"probe.h\")\n#endif",
            b"#define JOIN(a, b) a %:%: b\n#if JOIN(__has_, include)(\"probe.h\")\n#endif",
            b"#define JOIN(a, b) a ## b\n#if JOIN(__has_, include)(\"probe.h\")\n#endif\n#undef JOIN\n#define JOIN(a, b) 0",
            b"#if _NV_PASTE5(_, _, has, _include, )(\"probe.h\")\n#endif",
        ] {
            assert!(
                header_manifest(source, &[]).is_none(),
                "token-pasted filesystem probe was accepted: {source:?}"
            );
        }
    }

    #[test]
    fn token_paste_state_crosses_literal_include_boundaries() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("probe_logic.h"),
            b"#if JOIN(__has_, include)(\"probe.h\")\n#endif",
        )
        .unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let source = b"#define JOIN(a, b) a ## b\n#include \"probe_logic.h\"";
        assert!(header_manifest(source, &[include_root]).is_none());
    }

    #[test]
    fn token_paste_outside_a_conditional_remains_cacheable() {
        let source =
            b"#define NAME(a, b) a ## b\nextern \"C\" __global__ void NAME(kernel_, main)() {}";
        assert!(header_manifest(source, &[]).is_some());

        let defined_only = b"#define NAME(a, b) a ## b\n#if defined(NAME)\n#endif";
        assert!(header_manifest(defined_only, &[]).is_some());
    }

    #[test]
    fn volatile_predefined_macros_disable_the_manifest() {
        for source in [
            b"const char *built = __DATE__;".as_slice(),
            b"#define BUILD_CLOCK __TIME__\nconst char *built = BUILD_CLOCK;",
        ] {
            assert!(header_manifest(source, &[]).is_none());
        }

        assert!(header_manifest(b"// __DATE__\nconst char *name = \"__TIME__\";", &[]).is_some());
    }

    #[test]
    fn volatile_predefined_macros_in_headers_disable_the_manifest() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("build_stamp.h"), b"#define STAMP __TIME__").unwrap();
        let include_root = root.path().to_string_lossy().into_owned();

        assert!(
            header_manifest(
                b"#include \"build_stamp.h\"",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );
    }

    #[test]
    fn token_pasted_volatile_macros_disable_the_manifest() {
        let source = b"#define CAT_(a, b) a ## b\n\
#define CAT(a, b) CAT_(a, b)\n\
const char *stamp = CAT(__TI, ME__);";
        assert!(header_manifest(source, &[]).is_none());
        assert!(
            header_manifest(b"const char *stamp = _NV_PASTE5(_, _, TI, ME, __);", &[],).is_none(),
            "NVRTC builtin paste macros make every DATE/TIME fragment volatile"
        );
        for source in [
            b"const char *stamp = _NV_PASTE5(_, _, TIME, _, _);".as_slice(),
            b"#define U _\n#define T TIME\nconst char *stamp = _NV_PASTE5(U,U,T,U,U);",
            b"#define CAT8(a,b,c,d,e,f,g,h) a##b##c##d##e##f##g##h\nconst char *stamp = CAT8(_,_,T,I,M,E,_,_);",
            b"#define WRAP(a,b,c,d,e) _NV_PASTE5(a,b,c,d,e)\nconst char *stamp = WRAP(_,_,TIME,_,_);",
            b"#define A _\n#define T TIME\n#define P5(a,b,c,d,e) _NV_PASTE5(a,b,c,d,e)\nconst char *stamp = P5(A,A,T,A,A);",
            b"#define A __TI\n#define B ME__\nconst char *stamp = _NV_CONCAT_EVAL(A,B);",
            b"#define A() _\n#define T() TIME\n#define P5(a,b,c,d,e) _NV_PASTE5(a,b,c,d,e)\nconst char *stamp = P5(A(),A(),T(),A(),A());",
            b"#define P(a,b) a##b\n#define A P(_,_)\n#define B P(T,I)\n#define C P(M,E)\n#define D P(_,_)\n#define P4_I(a,b,c,d) a##b##c##d\n#define P4(a,b,c,d) P4_I(a,b,c,d)\nconst char *stamp=P4(A,B,C,D);",
            b"#define P(a,b) a##b\n#define P4_I(a,b,c,d) a##b##c##d\n#define P4(a,b,c,d) P4_I(a,b,c,d)\nconst char *stamp=P4(P(_,_),P(T,I),P(M,E),P(_,_));",
            b"#define CAT8(a,b,c,d,e,f,g,h) a##b##c##d##e##f##g##h\n#define F() CAT8(_,_,T,I,M,E,_,_)\nconst char *stamp=F();",
            b"#define P2_I(a,b) a##b\n#define P2(a,b) P2_I(a,b)\n#define P4_I(a,b,c,d) a##b##c##d\n#define P4(a,b,c,d) P4_I(a,b,c,d)\n#define F(x,y) P4(P2(_,_),P2(x,I),P2(y,E),P2(_,_))\nconst char *stamp=F(T,M);",
            b"#define U _\n#define T TIME\nconst char *stamp = _NV_PASTE5(U,U,T,U,U);\n#undef U\n#define U X",
            b"#define CAT8(a,b,c,d,e,f,g,h) a##b##c##d##e##f##g##h\n#define T X\n#undef T\nconst char *stamp=CAT8(_,_,T,I,M,E,_,_);",
            b"#define CAT8(a,b,c,d,e,f,g,h) a##b##c##d##e##f##g##h\n#define OUTER(x) x\nconst char *stamp = OUTER(CAT8(_,_,T,I,M,E,_,_));",
        ] {
            assert!(
                header_manifest(source, &[]).is_none(),
                "paste-call argument composition was accepted: {source:?}"
            );
        }
        let mut deep = String::new();
        for index in 0..33 {
            deep.push_str(&format!("#define A{index} A{}\n", index + 1));
        }
        deep.push_str("#define A33 _\n#define T TIME\n");
        deep.push_str("const char *stamp = _NV_PASTE5(A0,A0,T,A0,A0);");
        assert!(
            header_manifest(deep.as_bytes(), &[]).is_none(),
            "alias depth exhaustion must fail closed"
        );
    }

    #[test]
    fn numeric_paste_operands_round_trip_through_the_manifest() {
        let source = b"#define INDEX(N) N ## 0\nextern \"C\" __global__ void kernel() {}";
        let manifest = header_manifest(source, &[]).unwrap();
        assert!(header_manifest_analysis(&manifest).is_some());
        assert!(
            CompileKeyMaterial {
                module_kind: ModuleKind::Fixed,
                source: source.to_vec(),
                target: b"sm_80".to_vec(),
                argv: vec![],
                header_manifest: Some(manifest),
                nvrtc_version: (13, 2),
                nvrtc_library_domain: Some(b"runtime-and-builtins".to_vec()),
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            }
            .digest()
            .is_some()
        );
    }

    #[test]
    fn deterministic_nvrtc_seed_is_version_gated() {
        assert!(deterministic_nvrtc_options((12, 8), "17").is_empty());
        assert_eq!(
            deterministic_nvrtc_options((12, 9), "17"),
            vec!["--frandom-seed=17"]
        );
        assert_eq!(
            deterministic_nvrtc_options((13, 2), "29"),
            vec!["--frandom-seed=29"]
        );
    }

    #[test]
    fn token_pasted_volatile_macros_in_headers_disable_the_manifest() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("build_stamp.h"),
            b"#define CAT_(a, b) a ## b\n\
#define CAT(a, b) CAT_(a, b)\n\
const char *stamp = CAT(__DA, TE__);",
        )
        .unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        assert!(
            header_manifest(
                b"#include \"build_stamp.h\"",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );
    }

    #[test]
    fn unrelated_token_paste_remains_cacheable() {
        let source = b"#define CAT_(a, b) a ## b\n\
#define CAT(a, b) CAT_(a, b)\n\
extern \"C\" __global__ void CAT(kernel_, main)() {}";
        assert!(header_manifest(source, &[]).is_some());
    }

    #[test]
    fn literal_include_and_unrelated_token_paste_remain_cacheable() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("stable.h"), b"#define STABLE_VALUE 7").unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let source = b"#include \"stable.h\"\n\
#define SYMBOL(name) kernel_ ## name\n\
extern \"C\" __global__ void SYMBOL(main)() {}";
        assert!(
            header_manifest(source, std::slice::from_ref(&include_root)).is_some(),
            "the include directive keyword is not a __has_include fragment"
        );
    }

    #[test]
    fn cache_hit_revalidates_the_header_closure_after_lookup() {
        let root = tempfile::tempdir().unwrap();
        let header = root.path().join("mutable.h");
        std::fs::write(&header, b"#define VALUE 1").unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let source = b"#include \"mutable.h\"\nint value = VALUE;";
        let lookup_manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(lookup_manifest.is_some());

        std::fs::write(&header, b"#define VALUE 2").unwrap();
        assert!(
            !cache_hit_header_closure_is_current(
                source,
                std::slice::from_ref(&include_root),
                &lookup_manifest,
            ),
            "a hit keyed before header mutation must fall through to compilation"
        );
    }

    #[test]
    fn volatile_predefined_macros_in_compile_material_disable_persistent_keys() {
        let material = |source: &[u8], argv: Vec<Vec<u8>>| CompileKeyMaterial {
            module_kind: ModuleKind::Fixed,
            source: source.to_vec(),
            target: b"sm_89".to_vec(),
            argv,
            header_manifest: Some(vec![]),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: Some(b"runtime-and-builtins".to_vec()),
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };

        assert!(
            material(b"int stamp = __DATE__[0];", vec![b"--fmad=true".to_vec()])
                .digest()
                .is_none()
        );
        assert!(
            material(
                b"extern \"C\" __global__ void stable() {}",
                vec![b"-DNOW=__TIME__".to_vec()]
            )
            .digest()
            .is_none()
        );
        assert!(
            material(
                b"extern \"C\" __global__ void stable() {}",
                vec![b"-DNOW=steady".to_vec()]
            )
            .digest()
            .is_some()
        );
        assert!(
            material(
                b"#define CAT(a, b) a ## b\nint stable;",
                vec![b"-DPART=__TI".to_vec()]
            )
            .digest()
            .is_none()
        );
    }

    #[test]
    fn volatile_fragments_and_token_paste_combine_across_manifest_and_argv() {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let material = |manifest: Vec<u8>, argv: Vec<Vec<u8>>| CompileKeyMaterial {
            module_kind: ModuleKind::Fixed,
            source: b"extern \"C\" __global__ void stable() {}".to_vec(),
            target: b"sm_89".to_vec(),
            argv,
            header_manifest: Some(manifest),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: Some(b"runtime-and-builtins".to_vec()),
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };

        std::fs::write(
            root.path().join("fragments.h"),
            b"#define LEFT __TI\n#define RIGHT ME__",
        )
        .unwrap();
        let fragments = header_manifest(
            b"#include \"fragments.h\"",
            std::slice::from_ref(&include_root),
        )
        .unwrap();
        assert!(
            material(fragments, vec![b"-DCAT(a,b)=a##b".to_vec()])
                .digest()
                .is_none()
        );

        std::fs::write(root.path().join("paste.h"), b"#define CAT(a,b) a ## b").unwrap();
        let paste =
            header_manifest(b"#include \"paste.h\"", std::slice::from_ref(&include_root)).unwrap();
        assert!(
            material(paste, vec![b"-DLEFT=__DA".to_vec()])
                .digest()
                .is_none()
        );
    }

    #[test]
    fn argv_has_include_alias_disables_persistent_keys() {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let digest = || {
            CompileKeyMaterial {
                module_kind: ModuleKind::Fixed,
                source: b"#if CHECK\nint selected;\n#endif".to_vec(),
                target: b"sm_89".to_vec(),
                argv: vec![b"-DCHECK=__has_include(\"probe_optional.h\")".to_vec()],
                header_manifest: header_manifest(
                    b"#if CHECK\nint selected;\n#endif",
                    std::slice::from_ref(&include_root),
                ),
                nvrtc_version: (13, 2),
                nvrtc_library_domain: Some(b"runtime-and-builtins".to_vec()),
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            }
            .digest()
        };

        assert!(digest().is_none());
        std::fs::write(root.path().join("probe_optional.h"), b"#define PRESENT 1").unwrap();
        assert!(digest().is_none());
    }

    #[test]
    fn has_include_fragments_and_paste_combine_across_source_and_argv() {
        let material = |source: &[u8], argv: Vec<Vec<u8>>| CompileKeyMaterial {
            module_kind: ModuleKind::Fixed,
            source: source.to_vec(),
            target: b"sm_89".to_vec(),
            argv,
            header_manifest: header_manifest(source, &[]),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: Some(b"runtime-and-builtins".to_vec()),
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };

        assert!(
            material(
                b"#define LEFT __has_\n#if CAT(LEFT, include)(\"probe.h\")\n#endif",
                vec![b"-DCAT(a,b)=a##b".to_vec()]
            )
            .digest()
            .is_none()
        );
        assert!(
            material(
                b"#define CAT(a,b) a ## b\nint stable;",
                vec![b"-DLEFT=__has_".to_vec()]
            )
            .digest()
            .is_some(),
            "an unused fragment is not a volatile operator"
        );

        assert!(
            material(
                b"#define CAT_(a,b) a ## b\n\
#define CAT(a,b) CAT_(a,b)\n\
#define A __has_\n\
#define B include\n\
#if H\nint selected;\n#endif",
                vec![b"-DH=CAT(A,B)(\"probe.h\")".to_vec()]
            )
            .digest()
            .is_none(),
            "argv aliases must participate in the source/header macro graph"
        );
    }

    #[cfg(unix)]
    #[test]
    fn quoted_include_recurses_from_its_logical_symlink_parent() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let include = root.path().join("include");
        let alias = include.join("alias");
        let real = root.path().join("real");
        std::fs::create_dir_all(&alias).unwrap();
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("header.h"), b"#include \"sibling.h\"").unwrap();
        std::fs::write(real.join("sibling.h"), b"canonical sibling").unwrap();
        let logical_sibling = alias.join("sibling.h");
        std::fs::write(&logical_sibling, b"logical sibling a").unwrap();
        symlink(real.join("header.h"), alias.join("header.h")).unwrap();

        let include_root = include.to_string_lossy().into_owned();
        let source = b"#include \"alias/header.h\"";
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(manifest.is_some());
        std::fs::write(logical_sibling, b"logical sibling b").unwrap();
        assert!(!header_manifest_is_current(
            source,
            &[include_root],
            &manifest
        ));
    }

    #[cfg(unix)]
    #[test]
    fn parent_dir_in_a_literal_include_disables_the_manifest() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let include = root.path().join("include");
        let real = root.path().join("real");
        let sub = real.join("sub");
        std::fs::create_dir_all(&include).unwrap();
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("header.h"), b"#include \"../dep.h\"").unwrap();
        std::fs::write(real.join("dep.h"), b"compiler target").unwrap();
        std::fs::write(include.join("dep.h"), b"lexical target").unwrap();
        symlink(&sub, include.join("alias")).unwrap();

        let include_root = include.to_string_lossy().into_owned();
        assert!(
            header_manifest(
                b"#include \"alias/header.h\"",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn parent_dir_in_an_include_root_disables_the_manifest() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let include = root.path().join("include");
        let real = root.path().join("real");
        std::fs::create_dir_all(&include).unwrap();
        std::fs::create_dir_all(real.join("sub")).unwrap();
        symlink(real.join("sub"), include.join("alias")).unwrap();
        let include_root = include.join("alias/..").to_string_lossy().into_owned();

        assert!(header_manifest(b"", &[include_root]).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn one_canonical_header_is_traversed_in_each_logical_context() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let include = root.path().join("include");
        let left = include.join("left");
        let right = include.join("right");
        let real = root.path().join("real");
        std::fs::create_dir_all(&left).unwrap();
        std::fs::create_dir_all(&right).unwrap();
        std::fs::create_dir_all(&real).unwrap();
        let common = real.join("common.h");
        std::fs::write(&common, b"#include \"sibling.h\"").unwrap();
        std::fs::write(real.join("sibling.h"), b"canonical sibling").unwrap();
        symlink(&common, left.join("header.h")).unwrap();
        symlink(&common, right.join("header.h")).unwrap();
        let left_sibling = left.join("sibling.h");
        let right_sibling = right.join("sibling.h");
        std::fs::write(&left_sibling, b"left a").unwrap();
        std::fs::write(&right_sibling, b"right a").unwrap();

        let include_root = include.to_string_lossy().into_owned();
        let source = b"#include \"left/header.h\"\n#include \"right/header.h\"";
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(manifest.is_some());
        std::fs::write(&left_sibling, b"left b").unwrap();
        assert!(!header_manifest_is_current(
            source,
            std::slice::from_ref(&include_root),
            &manifest
        ));

        std::fs::write(&left_sibling, b"left a").unwrap();
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        std::fs::write(&right_sibling, b"right b").unwrap();
        assert!(!header_manifest_is_current(
            source,
            &[include_root],
            &manifest
        ));
    }

    #[cfg(unix)]
    #[test]
    fn a_directory_symlink_include_cycle_disables_the_manifest() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let include = root.path().join("include");
        std::fs::create_dir_all(&include).unwrap();
        std::fs::write(include.join("header.h"), b"#include \"loop/header.h\"").unwrap();
        symlink(&include, include.join("loop")).unwrap();

        let include_root = include.to_string_lossy().into_owned();
        assert!(
            header_manifest(
                b"#include \"header.h\"",
                std::slice::from_ref(&include_root)
            )
            .is_none()
        );
    }

    #[test]
    fn an_include_guarded_self_include_remains_cacheable() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("header.h"),
            b"#ifndef HEADER_H\n#define HEADER_H\n#include \"header.h\"\n#endif",
        )
        .unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        assert!(
            header_manifest(
                b"#include \"header.h\"",
                std::slice::from_ref(&include_root)
            )
            .is_some()
        );
    }

    #[test]
    fn header_appearance_and_disappearance_change_manifest_availability() {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let source = b"#include \"value.cuh\"";
        assert!(header_manifest(source, std::slice::from_ref(&include_root)).is_none());

        let header = root.path().join("value.cuh");
        std::fs::write(&header, b"present").unwrap();
        let manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(manifest.is_some());
        std::fs::remove_file(header).unwrap();
        assert!(!header_manifest_is_current(
            source,
            &[include_root],
            &manifest
        ));
    }

    #[test]
    fn builtin_header_names_are_bound_until_a_filesystem_header_appears() {
        let root = tempfile::tempdir().unwrap();
        let include_root = root.path().to_string_lossy().into_owned();
        let source = b"#include <cuda_fp16.h>\n\
#include <cassert>\n\
#include <stdint.h>\n\
#include <utility>\n\
#include <cstring>\n\
#include <stdlib.h>\n\
#include <ctype.h>\n\
#include <limits.h>\n\
#include <stddef.h>\n\
#include <functional>";
        let builtin_manifest = header_manifest(source, std::slice::from_ref(&include_root));
        assert!(
            builtin_manifest.is_some(),
            "the NVRTC builtins library supplies unresolved angle headers"
        );

        assert!(
            header_manifest(
                b"#include <unknown-environment-header.h>",
                std::slice::from_ref(&include_root)
            )
            .is_none(),
            "unknown angle headers may come from an implicit filesystem root"
        );

        std::fs::write(root.path().join("cuda_fp16.h"), b"filesystem").unwrap();
        let filesystem_manifest = header_manifest(source, &[include_root]);
        assert!(filesystem_manifest.is_some());
        assert_ne!(builtin_manifest, filesystem_manifest);
    }
}
