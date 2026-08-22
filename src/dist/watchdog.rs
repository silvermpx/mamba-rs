//! Deadline tools for the distributed lanes: a run-with-deadline for
//! calls that may block forever inside a foreign library (NCCL init),
//! and an armed abort timer for collective windows.
//!
//! Threading note: this module deliberately uses `std` threads and the
//! `std::sync::mpsc` timeout receive — mamba-rs has no async runtime,
//! and a deadline on a BLOCKING foreign call needs a real OS thread the
//! call can occupy while the caller keeps its own timeline.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use super::error::DistError;

/// Run `f` on a helper thread; give up after `deadline`. On timeout the
/// helper is deliberately ORPHANED, still blocked inside `f` — the only
/// safe reclamation for a call hung inside a foreign library is process
/// exit, which the fail-fast path forces: the caller turns the timeout
/// into a rank error, the rank exits non-zero, and the supervisor kills
/// the rest of the world. (In a supervisor-less `attach()` world the
/// orphan persists until the launcher reaps the process; its late
/// result is dropped, and a communicator it may eventually produce is
/// dropped-then-aborted — safe teardown, but the job must not be
/// retried in the same process.)
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

const ARMED: u8 = 0;
const DISARMED: u8 = 1;
const FIRED: u8 = 2;

/// An armed one-shot timer: unless [`Watchdog::disarm`] wins the race
/// first, `on_deadline` fires once after `deadline`. The hand-off is a
/// compare-exchange on a three-state cell, so exactly one of
/// {disarm, fire} ever wins — the deadline action can never run after a
/// successful disarm, and a disarm that lost reports the fire to the
/// caller. The distributed lanes arm this around collective windows
/// with a communicator-abort action, so a transport hang converts into
/// a loud error instead of an eternal wait.
pub(super) struct Watchdog {
    state: Arc<AtomicU8>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    pub(super) fn arm(
        name: &str,
        deadline: Duration,
        on_deadline: impl FnOnce() + Send + 'static,
    ) -> Result<Self, DistError> {
        let state = Arc::new(AtomicU8::new(ARMED));
        let seen = state.clone();
        let handle = std::thread::Builder::new()
            .name(format!("mamba-watchdog-{name}"))
            .spawn(move || {
                let end = Instant::now() + deadline;
                loop {
                    if seen.load(Ordering::Acquire) != ARMED {
                        return;
                    }
                    let now = Instant::now();
                    if now >= end {
                        break;
                    }
                    // A disarm unparks immediately; a spurious wakeup
                    // just re-checks the state and the clock.
                    std::thread::park_timeout(end - now);
                }
                // Claim the fire atomically: if disarm got here first,
                // stand down without acting.
                if seen
                    .compare_exchange(ARMED, FIRED, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    on_deadline();
                }
            })
            .map_err(|e| DistError::Transport(format!("watchdog {name}: spawn: {e}")))?;
        Ok(Self {
            state,
            handle: Some(handle),
        })
    }

    /// Stand down. Returns `true` when the deadline action already ran
    /// (the disarm lost the race) — the caller must treat the guarded
    /// window as failed even if its own work appeared to succeed.
    pub(super) fn disarm(mut self) -> bool {
        let won = self
            .state
            .compare_exchange(ARMED, DISARMED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        if let Some(h) = self.handle.take() {
            // Wake a parked timer so the join is immediate.
            h.thread().unpark();
            let _ = h.join();
        }
        !won
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Disarm without joining: drop can run during unwinding, and
        // the CAS guarantees the deadline action cannot start after a
        // successful disarm — no join is needed for correctness.
        let _ = self
            .state
            .compare_exchange(ARMED, DISARMED, Ordering::AcqRel, Ordering::Acquire);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
        }
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
    fn watchdog_fires_on_deadline_and_reports_through_disarm() {
        let fired = Arc::new(AtomicUsize::new(0));
        let f2 = fired.clone();
        let wd = Watchdog::arm("fire", Duration::from_millis(30), move || {
            f2.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(fired.load(Ordering::SeqCst), 1, "must fire exactly once");
        assert!(wd.disarm(), "disarm must report the fire");
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
        assert!(!wd.disarm(), "in-time disarm must report no fire");
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(fired.load(Ordering::SeqCst), 0, "disarmed watchdog fired");
    }

    #[test]
    fn disarm_is_immediate_not_poll_paced() {
        // The old implementation slept in 10 ms steps and joined in
        // disarm — a per-window tax. The park-based timer must disarm
        // in well under one step.
        let wd = Watchdog::arm("swift", Duration::from_secs(30), || {}).unwrap();
        let t0 = Instant::now();
        assert!(!wd.disarm());
        assert!(
            t0.elapsed() < Duration::from_millis(8),
            "disarm took {:?}",
            t0.elapsed()
        );
    }
}
