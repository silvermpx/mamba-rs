//! Stable compiler, artifact, policy, and device identities for CUDA graphs.

use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::context::{BiGemmFamily, F32TriadPolicy};

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
pub const NUMERIC_ABI_REVISION: u16 = 2;
pub const TUNING_TABLE_REVISION: u16 = 1;
pub const SCHEDULE_REVISION: u16 = 2;

const NUMERIC_CONTRACT_DOMAIN: &[u8] = b"mamba-rs.resolved-numeric-contract.v2";
const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mamba-rs.artifact-digest.v1";
const COMPILER_TARGET_DOMAIN: &[u8] = b"mamba-rs.compiler-target.v1";
const DRIVER_BUILD_DIGEST_DOMAIN: &[u8] = b"mamba-rs.driver-build-digest.v1";
const LAUNCH_COUNT_DOMAIN: &[u8] = b"mamba-rs.resolved-launch-count.v1";

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
    pub module_kind: ModuleKind,
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
}

pub fn build_artifact_set(artifacts: &[ArtifactIdentity]) -> Result<ArtifactSetIdentity, String> {
    if !(3..=4).contains(&artifacts.len()) {
        return Err("artifact set must contain fixed, scalar triad, SM80 triad, and at most one specialized triad module".into());
    }
    if artifacts[0].module_kind != ModuleKind::Fixed
        || artifacts[1].module_kind != ModuleKind::TriadScalar
        || artifacts[2].module_kind != ModuleKind::TriadSm80
    {
        return Err(
            "artifact set must start with Fixed, TriadScalar, and TriadSm80 in order".into(),
        );
    }
    let specialized = artifacts.get(3).copied();
    if let Some(artifact) = specialized
        && !matches!(
            artifact.module_kind,
            ModuleKind::TriadSm90a | ModuleKind::TriadSm100 | ModuleKind::TriadSm120
        )
    {
        return Err("fourth artifact must be a supported specialized triad module".into());
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
    pub cublas_tf32: bool,
    pub f32_triad_policy: F32TriadPolicy,
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
    pub const TRIAD_DETERMINISTIC_TF32_V1: Self = Self(1 << 6);

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
        contracts.union(NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1)
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
}

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

fn append_resolved_gemm_route(
    digest: FramedSha256,
    index: usize,
    route: &ResolvedGemmRoute,
) -> FramedSha256 {
    let accepted_target = route
        .device_caps
        .accepted_target
        .map(|target| target.as_str().as_bytes().to_vec());
    digest
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
        .optional(b"caps-accepted-target", accepted_target.as_deref())
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

#[cfg(test)]
mod cache_and_header_tests {
    use super::*;
    #[cfg(target_os = "linux")]
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
                include_roots: vec![],
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
            include_roots: vec![],
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
            include_roots: vec![include_root.as_bytes().to_vec()],
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
                include_roots: vec![include_root.as_bytes().to_vec()],
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
            include_roots: vec![],
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
