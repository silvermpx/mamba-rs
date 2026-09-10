//! Acceptance evidence: machine-readable capture of every bit-gate cell.
//!
//! "Re-run and diff stdout" turns a sixteen-cell hash table into a
//! human eyeball test; a merge acceptance needs the cells as data. When
//! `MAMBA_RS_ACCEPTANCE_TSV` names a file, every participating printer
//! appends its cells there as tab-separated rows
//! (`suite<TAB>arm<TAB>key<TAB>value`), and `examples/acceptance_diff`
//! compares two such files. With the variable unset this module is
//! inert, so ordinary runs pay nothing. The TSV shape follows the
//! cublas-probe artifact precedent: append-only, zero dependencies,
//! diffable by eye when the tooling is not at hand. Writers serialize
//! within the process and issue each complete row with one append write;
//! a short write is an error rather than a retry that another writer
//! could split.
static EVIDENCE_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn write_record_once(
    writer: &mut impl std::io::Write,
    record: &[u8],
    path: &std::path::Path,
) -> Result<(), String> {
    let written = writer
        .write(record)
        .map_err(|error| format!("write acceptance evidence {path:?}: {error}"))?;
    if written != record.len() {
        return Err(format!(
            "short acceptance evidence write {path:?}: wrote {written} of {} bytes",
            record.len()
        ));
    }
    Ok(())
}

pub(crate) fn append_open_options() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK);
    }
    options
}

pub(crate) fn require_opened_regular(
    file: &std::fs::File,
    path: &std::path::Path,
) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect opened acceptance evidence {path:?}: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "opened acceptance evidence {path:?} is not a regular file"
        ));
    }
    Ok(())
}

pub(crate) fn record_to_path(
    path: Option<&std::ffi::OsStr>,
    suite: &str,
    arm: &str,
    key: &str,
    value: &str,
) -> Result<(), String> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return Ok(());
    };
    let path = std::path::Path::new(path);
    let line = format!("{suite}\t{arm}\t{key}\t{value}\n");
    let _write_guard = EVIDENCE_WRITE_LOCK
        .lock()
        .map_err(|_| "acceptance evidence write lock is poisoned".to_string())?;
    match std::fs::metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(format!(
                "acceptance evidence target {path:?} is not a regular file"
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "inspect acceptance evidence target {path:?}: {error}"
            ));
        }
    }
    let mut file = append_open_options()
        .open(path)
        .map_err(|error| format!("open acceptance evidence {path:?}: {error}"))?;
    require_opened_regular(&file, path)?;
    write_record_once(&mut file, line.as_bytes(), path)
}

pub fn record(suite: &str, arm: &str, key: &str, value: &str) -> Result<(), String> {
    let path = std::env::var_os("MAMBA_RS_ACCEPTANCE_TSV");
    record_to_path(path.as_deref(), suite, arm, key, value)
}
