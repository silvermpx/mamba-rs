//! Deadline tools for the distributed lanes: a run-with-deadline for
//! calls that may block forever inside a foreign library (NCCL init),
//! and an armed abort timer for collectives already in flight.
//!
//! Threading note: this module deliberately uses `std` threads and the
//! `std::sync::mpsc` timeout receive — mamba-rs has no async runtime,
//! and a deadline on a BLOCKING foreign call needs a real OS thread the
//! call can occupy while the caller keeps its own timeline.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::error::DistError;

/// Run `f` on a helper thread; give up after `deadline`. On timeout the
/// helper is deliberately ORPHANED, still blocked inside `f` — the only
/// safe reclamation for a call hung inside a foreign library is process
/// exit, which the fail-fast path forces: the caller turns the timeout
/// into a rank error, the rank exits non-zero, and the supervisor kills
/// the rest of the world.
pub(super) fn run_with_deadline<T: Send + 'static>(
    what: &str,
    deadline: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, DistError> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(format!("mamba-dist-{what}"))
        .spawn(move || {
            // The receiver may be gone after a timeout — a failed send
            // just drops the late result.
            let _ = tx.send(f());
        })
        .map_err(|e| DistError::Transport(format!("{what}: helper thread spawn: {e}")))?;
    rx.recv_timeout(deadline).map_err(|_| {
        DistError::Transport(format!(
            "{what} did not complete within {deadline:?} — failing fast \
             (the supervisor reaps the world; the blocked helper thread \
             is reclaimed by process exit)"
        ))
    })
}

/// An armed one-shot timer: unless [`Watchdog::disarm`] runs first,
/// `on_deadline` fires once after `deadline`. The distributed lanes arm
/// it around collective windows with a communicator-abort action, so a
/// transport hang converts into a loud error instead of an eternal wait.
pub(super) struct Watchdog {
    disarm: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    pub(super) fn arm(
        name: &str,
        deadline: Duration,
        on_deadline: impl FnOnce() + Send + 'static,
    ) -> Result<Self, DistError> {
        let disarm = Arc::new(AtomicBool::new(false));
        let seen = disarm.clone();
        let handle = std::thread::Builder::new()
            .name(format!("mamba-watchdog-{name}"))
            .spawn(move || {
                let end = Instant::now() + deadline;
                while Instant::now() < end {
                    if seen.load(Ordering::Acquire) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                if !seen.load(Ordering::Acquire) {
                    on_deadline();
                }
            })
            .map_err(|e| DistError::Transport(format!("watchdog {name}: spawn: {e}")))?;
        Ok(Self {
            disarm,
            handle: Some(handle),
        })
    }

    /// Stand down and reap the timer thread.
    pub(super) fn disarm(mut self) {
        self.disarm.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Disarm without joining: drop can run during unwinding, and a
        // join there would stall the panic path for up to one sleep
        // step. The timer thread sees the flag and exits on its own.
        self.disarm.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn deadline_returns_result_when_fast() {
        let out = run_with_deadline("fast", Duration::from_secs(5), || 41 + 1).unwrap();
        assert_eq!(out, 42);
    }

    #[test]
    fn deadline_errors_when_blocked() {
        let err = run_with_deadline("stuck", Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_secs(600));
        })
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("did not complete"), "{msg}");
    }

    #[test]
    fn watchdog_fires_on_deadline_and_only_once() {
        let fired = Arc::new(AtomicUsize::new(0));
        let f2 = fired.clone();
        let wd = Watchdog::arm("fire", Duration::from_millis(30), move || {
            f2.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(fired.load(Ordering::SeqCst), 1, "must fire exactly once");
        wd.disarm();
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn watchdog_stays_quiet_when_disarmed_in_time() {
        let fired = Arc::new(AtomicUsize::new(0));
        let f2 = fired.clone();
        let wd = Watchdog::arm("quiet", Duration::from_millis(200), move || {
            f2.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        wd.disarm();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(fired.load(Ordering::SeqCst), 0, "disarmed watchdog fired");
    }
}
