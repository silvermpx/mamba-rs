//! The acceptance evidence recorder: inert without a target, exact
//! tab-separated rows with one append write each, regular files only.

#[path = "common/evidence.rs"]
mod evidence;

#[cfg(unix)]
use evidence::{append_open_options, require_opened_regular};
use evidence::{record, record_to_path, write_record_once};
use std::ffi::OsStr;

#[test]
fn missing_and_empty_paths_are_inert() {
    record_to_path(None, "suite", "arm", "key", "value").unwrap();
    record_to_path(Some(OsStr::new("")), "suite", "arm", "key", "value").unwrap();
}

#[test]
fn bad_parent_directory_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing").join("evidence.tsv");
    let err = record_to_path(Some(path.as_os_str()), "suite", "arm", "key", "value").unwrap_err();
    assert!(err.contains("open acceptance evidence"), "{err}");
}

#[test]
fn records_append_as_exact_tsv_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.tsv");
    record_to_path(
        Some(path.as_os_str()),
        "suite-a",
        "arm-a",
        "key-a",
        "value-a",
    )
    .unwrap();
    record_to_path(
        Some(path.as_os_str()),
        "suite-b",
        "arm-b",
        "key-b",
        "value-b",
    )
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "suite-a\tarm-a\tkey-a\tvalue-a\nsuite-b\tarm-b\tkey-b\tvalue-b\n"
    );
}

#[test]
fn directory_targets_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let err =
        record_to_path(Some(dir.path().as_os_str()), "suite", "arm", "key", "value").unwrap_err();
    assert!(err.contains("regular file"), "{err}");
}

#[cfg(unix)]
#[test]
fn opened_handle_must_be_regular() {
    let dir = tempfile::tempdir().unwrap();
    let handle = std::fs::File::open(dir.path()).unwrap();
    let err = require_opened_regular(&handle, dir.path()).unwrap_err();
    assert!(err.contains("opened acceptance evidence"), "{err}");
}

struct ShortWriter {
    calls: usize,
}

impl std::io::Write for ShortWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.calls += 1;
        Ok(bytes.len().saturating_sub(1))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn short_writes_are_errors_and_never_retried() {
    let mut writer = ShortWriter { calls: 0 };
    let err = write_record_once(
        &mut writer,
        b"suite\tarm\tkey\tvalue\n",
        std::path::Path::new("evidence.tsv"),
    )
    .unwrap_err();
    assert!(err.contains("short acceptance evidence write"), "{err}");
    assert_eq!(writer.calls, 1);
}

#[test]
fn concurrent_threads_append_complete_rows() {
    const THREADS: usize = 8;
    const ROWS_PER_THREAD: usize = 32;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.tsv");
    let barrier = std::sync::Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let path = &path;
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for row in 0..ROWS_PER_THREAD {
                    record_to_path(
                        Some(path.as_os_str()),
                        &format!("suite-{thread}"),
                        "threaded",
                        &format!("row-{row}"),
                        &"x".repeat(4_096),
                    )
                    .unwrap();
                }
            });
        }
    });

    let contents = std::fs::read_to_string(path).unwrap();
    assert!(contents.ends_with('\n'));
    let rows = contents.lines().collect::<Vec<_>>();
    assert_eq!(rows.len(), THREADS * ROWS_PER_THREAD);
    let actual = rows.into_iter().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual.len(), THREADS * ROWS_PER_THREAD);
    for thread in 0..THREADS {
        for row in 0..ROWS_PER_THREAD {
            let expected = format!("suite-{thread}\tthreaded\trow-{row}\t{}", "x".repeat(4_096));
            assert!(actual.contains(expected.as_str()), "missing {thread}/{row}");
        }
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_paths_are_preserved() {
    use std::os::unix::ffi::OsStringExt as _;

    let dir = tempfile::tempdir().unwrap();
    let name = std::ffi::OsString::from_vec(b"evidence-\xff.tsv".to_vec());
    let path = dir.path().join(name);
    let result = record_to_path(Some(path.as_os_str()), "suite", "arm", "key", "value");
    match result {
        Ok(()) => assert_eq!(std::fs::read(path).unwrap(), b"suite\tarm\tkey\tvalue\n"),
        Err(error) => assert!(
            error.contains("open acceptance evidence") && error.contains("\\xFF"),
            "{error}"
        ),
    }
}

#[cfg(unix)]
#[test]
fn device_targets_are_rejected() {
    let err = record_to_path(
        Some(OsStr::new("/dev/null")),
        "suite",
        "arm",
        "key",
        "value",
    )
    .unwrap_err();
    assert!(err.contains("regular file"), "{err}");
}

#[cfg(unix)]
#[test]
fn socket_targets_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.sock");
    let _listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    let err = record_to_path(Some(path.as_os_str()), "suite", "arm", "key", "value").unwrap_err();
    assert!(err.contains("regular file"), "{err}");
}

#[cfg(unix)]
#[test]
fn append_open_does_not_block_or_accept_a_fifo() {
    use std::os::unix::fs::OpenOptionsExt as _;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.fifo");
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());

    let writer_path = path.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        sender
            .send(append_open_options().open(writer_path))
            .unwrap();
    });
    let result = match receiver.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            let reader = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&path)
                .unwrap();
            let result = receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("blocked FIFO writer did not resume after a reader opened");
            drop(reader);
            writer.join().unwrap();
            drop(result);
            panic!("append open blocked on a FIFO");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            writer.join().unwrap();
            panic!("append-open worker disconnected");
        }
    };
    writer.join().unwrap();
    let error = result.unwrap_err();
    assert_eq!(error.raw_os_error(), Some(libc::ENXIO));
}

#[cfg(unix)]
#[test]
fn fifo_targets_are_rejected_without_opening() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.fifo");
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let err = record_to_path(Some(path.as_os_str()), "suite", "arm", "key", "value").unwrap_err();
    assert!(err.contains("regular file"), "{err}");
}

#[cfg(target_os = "linux")]
#[test]
fn full_device_is_rejected() {
    let err = record_to_path(
        Some(OsStr::new("/dev/full")),
        "suite",
        "arm",
        "key",
        "value",
    )
    .unwrap_err();
    assert!(err.contains("regular file"), "{err}");
}

#[test]
fn record_is_inert_without_a_target_in_the_environment() {
    // The recorder reads MAMBA_RS_ACCEPTANCE_TSV; the test process does not
    // set it, so the call must succeed without touching the filesystem.
    assert!(std::env::var_os("MAMBA_RS_ACCEPTANCE_TSV").is_none_or(|value| value.is_empty()));
    record("suite", "arm", "key", "value").unwrap();
}
