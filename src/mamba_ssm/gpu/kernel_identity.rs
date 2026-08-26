//! Stable compiler, artifact, policy, and device identities for CUDA graphs.

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
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
pub const COMPILER_REVISION: u16 = 2;
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

pub(crate) fn canonical_ptx_from_cache(payload: Vec<u8>) -> Result<String, String> {
    let mut image = payload;
    image.push(0);
    canonical_ptx_image(&image)
}

#[cfg(target_os = "linux")]
static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const MAX_CACHE_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
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
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
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
    LegacyCombined = 1,
    TriadScalar = 2,
    TriadSm80 = 3,
    TriadSm90a = 4,
    TriadSm100 = 5,
    TriadSm120 = 6,
    // Value 7 is reserved for the baseline triad module identity.
    Mamba3Combined = 8,
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
    let roots: Vec<PathBuf> = include_roots.iter().map(PathBuf::from).collect();
    if roots.iter().any(|root| !root.is_absolute()) {
        return None;
    }
    let mut pending = Vec::new();
    let mut records = BTreeMap::<Vec<u8>, Sha256Digest>::new();
    let mut builtin_headers = BTreeSet::<Vec<u8>>::new();
    collect_includes(source, None, &roots, &mut pending, &mut builtin_headers)?;

    while let Some(path) = pending.pop() {
        let canonical = std::fs::canonicalize(&path).ok()?;
        let key = manifest_path_key(&canonical, &roots)?;
        if records.contains_key(&key) {
            continue;
        }
        let bytes = std::fs::read(&canonical).ok()?;
        records.insert(key, FramedSha256::bytes(&bytes));
        collect_includes(
            &bytes,
            canonical.parent(),
            &roots,
            &mut pending,
            &mut builtin_headers,
        )?;
    }

    let mut output = Vec::new();
    output.extend_from_slice(&(builtin_headers.len() as u64).to_le_bytes());
    for name in builtin_headers {
        append_manifest_field(&mut output, &name);
    }
    output.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (path, digest) in records {
        append_manifest_field(&mut output, &path);
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

fn collect_includes(
    source: &[u8],
    current_dir: Option<&Path>,
    roots: &[PathBuf],
    pending: &mut Vec<PathBuf>,
    builtin_headers: &mut BTreeSet<Vec<u8>>,
) -> Option<()> {
    let text = normalized_preprocessor_text(source)?;
    if text.contains("__has_include") {
        return None;
    }
    for line in text.lines() {
        let Some(rest) = include_directive(line)? else {
            continue;
        };
        let (quoted, name, trailing) = if let Some(rest) = rest.strip_prefix('"') {
            let (name, trailing) = rest.split_once('"')?;
            (true, name, trailing)
        } else if let Some(rest) = rest.strip_prefix('<') {
            let (name, trailing) = rest.split_once('>')?;
            (false, name, trailing)
        } else {
            return None;
        };
        if name.is_empty() || !trailing.trim().is_empty() {
            return None;
        }
        let mut candidates = Vec::new();
        if quoted && let Some(dir) = current_dir {
            candidates.push(dir.join(name));
        }
        candidates.extend(roots.iter().map(|root| root.join(name)));
        if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
            pending.push(path);
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
    let line = line.trim_start();
    let line = if let Some(line) = line.strip_prefix('#') {
        line
    } else if let Some(line) = line.strip_prefix("%:") {
        line
    } else {
        return Some(None);
    };
    let line = line.trim_start();
    let directive_end = line
        .find(|value: char| !(value.is_ascii_alphanumeric() || value == '_'))
        .unwrap_or(line.len());
    let directive = &line[..directive_end];
    if directive == "import" || directive.starts_with("include") && directive != "include" {
        return None;
    }
    if directive != "include" {
        return Some(None);
    }
    Some(Some(line[directive_end..].trim_start()))
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
    pub const FIXED_MATVEC_TREE_V1: Self = Self(1 << 5);

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
    match (policy.bi_gemm_family, policy.bi_tensor_cores) {
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

#[cfg(test)]
mod cache_and_header_tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[cfg(target_os = "linux")]
    fn trusted_tempdir() -> tempfile::TempDir {
        let home = std::env::var_os("HOME").expect("HOME for cache security tests");
        let home = std::fs::canonicalize(home).expect("canonical HOME for cache security tests");
        tempfile::Builder::new()
            .prefix("mamba-cache-test-")
            .tempdir_in(home)
            .expect("trusted temporary directory")
    }

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
                source: b"source".to_vec(),
                target: b"sm_89".to_vec(),
                argv: vec![b"--gpu-architecture=sm_89".to_vec()],
                include_roots: vec![],
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
        assert!(header_manifest(b"#include_next <value.cuh>", &[include_root.clone()]).is_none());
        assert!(header_manifest(b"#include_magic <value.cuh>", &[include_root.clone()]).is_none());

        let header = root.path().join("value.cuh");
        std::fs::write(&header, b"first").unwrap();
        let source = b"#include \"value.cuh\"";
        let manifest = header_manifest(source, &[include_root.clone()]);
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
