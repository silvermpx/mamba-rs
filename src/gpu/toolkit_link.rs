//! Machine code from the toolkit's assembler, not the driver's.
//!
//! Every module is compiled to PTX, and a card runs machine code assembled
//! from that PTX. Left to the driver, that last step runs the assembler the
//! driver build carries, so two boxes on different drivers may assemble the
//! same PTX differently. Where the PTX leaves a multiply and an add without
//! a rounding mode, one assembler may fuse them into an FMA and another may
//! not, and the same kernel then rounds its last bit differently from box to
//! box.
//!
//! nvJitLink ships with the CUDA toolkit beside NVRTC and runs the
//! assembler of the toolkit's own release. The machine code then follows
//! the toolkit alone, whatever driver the box runs. It is the code the
//! driver of the same release assembles, so no kernel gets slower. The
//! assembled code is cached on disk under a key of the PTX, the target and
//! the linker library, the way the PTX itself is cached.
//!
//! A box without nvJitLink, or with one older than its NVRTC, keeps the
//! driver's assembler and says so once on stderr.

use std::ffi::{CString, c_char, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once, OnceLock, PoisonError};

use cudarc::driver::{CudaContext, CudaModule};

use super::kernel_identity::{self as ki, ArtifactKind, FramedSha256, Sha256Digest};

type Handle = *mut c_void;
type CreateFn = unsafe extern "C" fn(*mut Handle, u32, *const *const c_char) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut Handle) -> i32;
type AddDataFn = unsafe extern "C" fn(Handle, i32, *const c_void, usize, *const c_char) -> i32;
type CompleteFn = unsafe extern "C" fn(Handle) -> i32;
type SizeFn = unsafe extern "C" fn(Handle, *mut usize) -> i32;
type CopyFn = unsafe extern "C" fn(Handle, *mut c_void) -> i32;
type VersionFn = unsafe extern "C" fn(*mut c_uint, *mut c_uint) -> i32;

/// `NVJITLINK_SUCCESS` and `NVJITLINK_INPUT_PTX` from nvJitLink.h.
const SUCCESS: i32 = 0;
const INPUT_PTX: i32 = 2;

/// The nvJitLink entry points this module calls, resolved once.
struct JitLink {
    create: CreateFn,
    destroy: DestroyFn,
    add_data: AddDataFn,
    complete: CompleteFn,
    cubin_size: SizeFn,
    cubin: CopyFn,
    error_log_size: SizeFn,
    error_log: CopyFn,
    version: (u32, u32),
    /// The library file and its size and modification time: part of the
    /// cache key, so a replaced library never serves code it did not make.
    stamp: Vec<u8>,
    // Kept last so the entry points above never outlive the mapping.
    _library: libloading::Library,
}

/// Destroys a link handle however the assembly ends.
struct LinkHandle<'a> {
    link: &'a JitLink,
    handle: Handle,
}

impl Drop for LinkHandle<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful nvJitLinkCreate and is
        // destroyed exactly once, here.
        unsafe {
            (self.link.destroy)(&mut self.handle);
        }
    }
}

impl JitLink {
    fn open() -> Result<Self, String> {
        let mut tried = Vec::new();
        for candidate in library_candidates() {
            // SAFETY: loading the toolkit's nvJitLink runs its initialisers,
            // which only set up the library's own state.
            match unsafe { libloading::Library::new(&candidate) } {
                Ok(library) => return Self::bind(library, &candidate),
                Err(error) => tried.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(format!("nvJitLink not found ({})", tried.join("; ")))
    }

    fn bind(library: libloading::Library, path: &Path) -> Result<Self, String> {
        // SAFETY: nvJitLinkVersion has this signature in every release that
        // exports it, and it only writes the two integers.
        let version = unsafe {
            let version_fn = *library
                .get::<VersionFn>(b"nvJitLinkVersion\0")
                .map_err(|error| format!("nvJitLinkVersion: {error}"))?;
            let (mut major, mut minor) = (0, 0);
            let status = version_fn(&mut major, &mut minor);
            if status != SUCCESS {
                return Err(format!("nvJitLinkVersion failed with {status}"));
            }
            (major, minor)
        };
        // SAFETY: each symbol is resolved by the name nvJitLink.h declares
        // for it, under the C signature declared there.
        let link = unsafe {
            Self {
                create: symbol(&library, "Create", version)?,
                destroy: symbol(&library, "Destroy", version)?,
                add_data: symbol(&library, "AddData", version)?,
                complete: symbol(&library, "Complete", version)?,
                cubin_size: symbol(&library, "GetLinkedCubinSize", version)?,
                cubin: symbol(&library, "GetLinkedCubin", version)?,
                error_log_size: symbol(&library, "GetErrorLogSize", version)?,
                error_log: symbol(&library, "GetErrorLog", version)?,
                version,
                stamp: library_stamp(path),
                _library: library,
            }
        };
        Ok(link)
    }

    /// Assembles one module's PTX for `target` (`sm_120`, `sm_90a`).
    fn assemble(&self, ptx: &str, target: &str) -> Result<Vec<u8>, String> {
        let arch = CString::new(format!("-arch={target}")).map_err(|error| error.to_string())?;
        let ptx = CString::new(ptx).map_err(|_| "PTX holds a NUL byte".to_string())?;
        let options = [arch.as_ptr()];
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: the option array holds one valid C string that outlives
        // the call; nvJitLink writes the new handle.
        let status = unsafe { (self.create)(&mut handle, 1, options.as_ptr()) };
        if status != SUCCESS {
            return Err(format!("nvJitLinkCreate({target}) failed with {status}"));
        }
        let guard = LinkHandle { link: self, handle };
        let bytes = ptx.as_bytes_with_nul();
        // SAFETY: the data pointer and length describe the NUL-terminated
        // PTX, which outlives the call; a null name is allowed.
        let status = unsafe {
            (self.add_data)(
                guard.handle,
                INPUT_PTX,
                bytes.as_ptr().cast(),
                bytes.len(),
                std::ptr::null(),
            )
        };
        if status != SUCCESS {
            return Err(self.failure("nvJitLinkAddData", status, &guard));
        }
        // SAFETY: the handle is live and holds the PTX added above.
        let status = unsafe { (self.complete)(guard.handle) };
        if status != SUCCESS {
            return Err(self.failure("nvJitLinkComplete", status, &guard));
        }
        let mut size = 0usize;
        // SAFETY: the handle completed; nvJitLink writes the size.
        let status = unsafe { (self.cubin_size)(guard.handle, &mut size) };
        if status != SUCCESS || size == 0 {
            return Err(self.failure("nvJitLinkGetLinkedCubinSize", status, &guard));
        }
        let mut cubin = vec![0u8; size];
        // SAFETY: the buffer holds exactly the size nvJitLink reported.
        let status = unsafe { (self.cubin)(guard.handle, cubin.as_mut_ptr().cast()) };
        if status != SUCCESS {
            return Err(self.failure("nvJitLinkGetLinkedCubin", status, &guard));
        }
        Ok(cubin)
    }

    fn failure(&self, call: &str, status: i32, guard: &LinkHandle<'_>) -> String {
        let mut size = 0usize;
        // SAFETY: the handle is live; nvJitLink writes the log size and then
        // at most that many bytes into a buffer of that size.
        let log = unsafe {
            if (self.error_log_size)(guard.handle, &mut size) == SUCCESS && size > 0 {
                let mut text = vec![0u8; size];
                if (self.error_log)(guard.handle, text.as_mut_ptr().cast()) == SUCCESS {
                    String::from_utf8_lossy(&text)
                        .trim_end_matches('\0')
                        .trim()
                        .to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        };
        format!("{call} failed with {status}: {log}")
    }
}

/// Resolves `nvJitLink<name>`, or the versioned export of the release's
/// first ABI when a library exports only the versioned names.
///
/// # Safety
///
/// `T` must be the C signature nvJitLink.h declares for the entry point.
unsafe fn symbol<T: Copy>(
    library: &libloading::Library,
    name: &str,
    version: (u32, u32),
) -> Result<T, String> {
    let plain = format!("nvJitLink{name}\0");
    // SAFETY: forwarded from the caller.
    if let Ok(found) = unsafe { library.get::<T>(plain.as_bytes()) } {
        return Ok(*found);
    }
    let versioned = format!("__nvJitLink{name}_{}_0\0", version.0);
    // SAFETY: forwarded from the caller.
    unsafe { library.get::<T>(versioned.as_bytes()) }
        .map(|found| *found)
        .map_err(|error| format!("nvJitLink{name}: {error}"))
}

/// Where to look for nvJitLink: first beside the NVRTC this process
/// loaded, which is the same toolkit, then by the names the loader knows.
fn library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let (major, _) = crate::gpu::kernels::nvrtc_version();
    if let Some(nvrtc) = ki::unique_loaded_library_path("libnvrtc.so")
        && let Some(directory) = nvrtc.parent()
    {
        candidates.push(directory.join(format!("libnvJitLink.so.{major}")));
        candidates.push(directory.join("libnvJitLink.so"));
    }
    candidates.push(PathBuf::from(format!("libnvJitLink.so.{major}")));
    candidates.push(PathBuf::from("libnvJitLink.so"));
    candidates.push(PathBuf::from(format!("nvJitLink_{major}0_0.dll")));
    candidates
}

fn library_stamp(path: &Path) -> Vec<u8> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let meta = std::fs::metadata(&canonical).ok();
    let mut stamp = canonical.as_os_str().as_encoded_bytes().to_vec();
    if let Some(meta) = meta {
        stamp.extend_from_slice(&meta.len().to_le_bytes());
        if let Ok(modified) = meta.modified()
            && let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            stamp.extend_from_slice(&since.as_nanos().to_le_bytes());
        }
    }
    stamp
}

/// The linker this process assembles with, or why it falls back to the
/// driver. Opened once.
fn jit_link() -> Result<&'static JitLink, &'static str> {
    static LINK: OnceLock<Result<JitLink, String>> = OnceLock::new();
    LINK.get_or_init(|| {
        let link = JitLink::open()?;
        let nvrtc = crate::gpu::kernels::nvrtc_version();
        let linker = (
            i32::try_from(link.version.0).unwrap_or(i32::MAX),
            i32::try_from(link.version.1).unwrap_or(i32::MAX),
        );
        // PTX from a newer NVRTC may use an ISA an older assembler rejects.
        if linker < nvrtc {
            return Err(format!(
                "nvJitLink {}.{} is older than NVRTC {}.{}",
                linker.0, linker.1, nvrtc.0, nvrtc.1
            ));
        }
        Ok(link)
    })
    .as_ref()
    .map_err(String::as_str)
}

/// The machine target for a module's PTX on this card: the card's own SM,
/// with the arch-specific suffix when the PTX was written for one (PTX for
/// `sm_90a` assembles only for `sm_90a`).
fn machine_target(compute_capability: (i32, i32), ptx: &str) -> String {
    let suffix = ptx
        .lines()
        .map(str::trim_start)
        .find_map(|line| line.strip_prefix(".target"))
        .and_then(|rest| rest.split([' ', '\t', ',']).find(|token| !token.is_empty()))
        .and_then(|target| target.chars().last())
        .filter(char::is_ascii_alphabetic)
        .map(String::from)
        .unwrap_or_default();
    format!(
        "sm_{}{}{suffix}",
        compute_capability.0, compute_capability.1
    )
}

fn cubin_cache_key(link: &JitLink, ptx: &str, target: &str) -> Sha256Digest {
    FramedSha256::new(b"toolkit-cubin.v1")
        .required(b"ptx", ptx.as_bytes())
        .required(b"target", target.as_bytes())
        .required(
            b"linker-version",
            &[link.version.0.to_le_bytes(), link.version.1.to_le_bytes()].concat(),
        )
        .required(b"linker-library", &link.stamp)
        .finish()
}

/// How many images this process keeps at hand. A compile loads its module
/// and then the Driver ABI census loads the same module once per family it
/// reads; every one of those loads takes the image from here.
const RECENT_IMAGES: usize = 4;

static RECENT: Mutex<Vec<(Sha256Digest, Arc<[u8]>)>> = Mutex::new(Vec::new());

fn recent_image(key: &Sha256Digest) -> Option<Arc<[u8]>> {
    let mut recent = RECENT.lock().unwrap_or_else(PoisonError::into_inner);
    let index = recent.iter().position(|(known, _)| known == key)?;
    let entry = recent.remove(index);
    let image = Arc::clone(&entry.1);
    recent.push(entry);
    Some(image)
}

fn remember_image(key: Sha256Digest, image: &Arc<[u8]>) {
    let mut recent = RECENT.lock().unwrap_or_else(PoisonError::into_inner);
    recent.retain(|(known, _)| *known != key);
    if recent.len() == RECENT_IMAGES {
        recent.remove(0);
    }
    recent.push((key, Arc::clone(image)));
}

fn forget_image(key: &Sha256Digest) {
    RECENT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|(known, _)| known != key);
}

/// The machine code a module's PTX loads as on this card, from this
/// process or the disk cache when either holds it; `None` when this box
/// has no usable nvJitLink and the driver has to assemble the PTX.
/// `reuse` is false when a kept image just failed to load and has to be
/// assembled again.
pub(crate) fn machine_code(
    ctx: &CudaContext,
    ptx: &str,
    reuse: bool,
) -> Result<Option<Arc<[u8]>>, String> {
    let link = match jit_link() {
        Ok(link) => link,
        Err(reason) => {
            static FALLBACK: Once = Once::new();
            super::diagnostics::warn_once(&FALLBACK, || {
                format!(
                    "{reason}; the driver assembles the kernels, so their bits can follow the \
                     driver build"
                )
            });
            return Ok(None);
        }
    };
    let compute_capability = ctx
        .compute_capability()
        .map_err(|error| format!("compute capability query failed: {error:?}"))?;
    let target = machine_target(compute_capability, ptx);
    let key = cubin_cache_key(link, ptx, &target);
    if !reuse {
        forget_image(&key);
    } else if let Some(image) = recent_image(&key) {
        return Ok(Some(image));
    }
    let path = crate::gpu::kernels::kernel_cache_dir()
        .map(|directory| directory.join(format!("mamba-cubin-v1-{}.bin", ki::digest_hex(&key))));
    if reuse
        && let Some(path) = &path
        && let Some(hit) = ki::read_cache(path, key, ArtifactKind::Cubin)
    {
        let image: Arc<[u8]> = hit.payload.into();
        remember_image(key, &image);
        return Ok(Some(image));
    }
    match link.assemble(ptx, &target) {
        Ok(cubin) => {
            if let Some(path) = &path {
                ki::publish_cache(path, key, ArtifactKind::Cubin, &cubin);
            }
            let image: Arc<[u8]> = cubin.into();
            remember_image(key, &image);
            Ok(Some(image))
        }
        Err(reason) => {
            static ASSEMBLY: Once = Once::new();
            super::diagnostics::warn_once(&ASSEMBLY, || {
                format!(
                    "{reason}; the driver assembles this module, so its bits can follow the \
                     driver build"
                )
            });
            Ok(None)
        }
    }
}

/// The bytes `cuModuleLoadData` takes for a module: the toolkit's machine
/// code, or the NUL-terminated PTX when the driver has to assemble it.
pub(crate) fn load_image(ctx: &CudaContext, ptx: &str) -> Result<Arc<[u8]>, String> {
    match machine_code(ctx, ptx, true)? {
        Some(cubin) => Ok(cubin),
        None => CString::new(ptx)
            .map(|text| text.into_bytes_with_nul().into())
            .map_err(|_| "PTX holds a NUL byte".to_string()),
    }
}

/// Loads a module's PTX into `ctx` as machine code from the toolkit's
/// assembler; without nvJitLink, as PTX the driver assembles.
pub(crate) fn load_ptx_module(
    ctx: &Arc<CudaContext>,
    ptx: String,
) -> Result<Arc<CudaModule>, String> {
    let Some(cubin) = machine_code(ctx, &ptx, true)? else {
        return ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("PTX module load failed: {error:?}"));
    };
    if let Ok(module) = ctx.load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec())) {
        return Ok(module);
    }
    // A kept image that no longer loads is assembled again.
    match machine_code(ctx, &ptx, false)? {
        Some(cubin) => ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(cubin.to_vec()))
            .map_err(|error| format!("machine code load failed: {error:?}")),
        None => ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|error| format!("PTX module load failed: {error:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::machine_target;

    #[test]
    fn the_machine_target_is_the_card_with_the_ptx_suffix() {
        let generic = "//\n.version 9.2\n.target sm_120\n.address_size 64\n";
        assert_eq!(machine_target((12, 0), generic), "sm_120");
        assert_eq!(machine_target((12, 1), generic), "sm_121");
        let hopper = ".version 9.2\n.target sm_90a\n.address_size 64\n";
        assert_eq!(machine_target((9, 0), hopper), "sm_90a");
        let blackwell = ".version 9.2\n.target sm_100a, debug\n";
        assert_eq!(machine_target((10, 0), blackwell), "sm_100a");
        assert_eq!(machine_target((8, 9), ".target sm_89\n"), "sm_89");
        assert_eq!(machine_target((8, 9), "no target line"), "sm_89");
    }
}
