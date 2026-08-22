//! Configuration for a data-parallel world: which devices, which
//! reduction contract, how ranks find each other, and the seed law.

use std::path::PathBuf;
use std::time::Duration;

use super::error::DistError;

/// Which GPUs a run should use.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Devices {
    /// One specific device ordinal — no processes are spawned, no
    /// communicator is created, numerics are byte-identical to a
    /// non-distributed run.
    Single(usize),
    /// Every visible device.
    All,
    /// The first `n` visible devices.
    Count(usize),
    /// An explicit ordinal list, e.g. skipping a device that serves
    /// inference. Order is meaningful: position i hosts logical rank i.
    List(Vec<usize>),
}

impl Devices {
    /// Parse the CLI grammar: `"1"` (count), `"all"`, `"0,2,3"` (list).
    /// A single integer means COUNT, not ordinal — `Devices::Single` is
    /// chosen by the caller's non-distributed flag instead, so "1" and
    /// "4" read uniformly as "use this many GPUs".
    pub fn parse(s: &str) -> Result<Self, DistError> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("all") {
            return Ok(Devices::All);
        }
        if t.contains(',') {
            let mut list = Vec::new();
            for part in t.split(',') {
                let ord: usize = part.trim().parse().map_err(|_| {
                    DistError::Config(format!("bad device ordinal {part:?} in {s:?}"))
                })?;
                if list.contains(&ord) {
                    return Err(DistError::Config(format!(
                        "device ordinal {ord} repeats in {s:?}"
                    )));
                }
                list.push(ord);
            }
            return Ok(Devices::List(list));
        }
        let n: usize = t
            .parse()
            .map_err(|_| DistError::Config(format!("bad device spec {s:?}")))?;
        if n == 0 {
            return Err(DistError::Config("device count must be positive".into()));
        }
        Ok(if n == 1 {
            Devices::Single(0)
        } else {
            Devices::Count(n)
        })
    }
}

/// How the cross-rank gradient sum is performed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReduceContract {
    /// The house fold: shard-owner ranks sum the W addends per element
    /// in strictly ascending logical-rank order — bit-identical across
    /// runs regardless of transport, topology, library version, or
    /// physical GPU permutation, for a fixed logical world size. Live
    /// worlds run the transport-backed house reducer: addends move as
    /// pure bytes (send/recv/broadcast, no library arithmetic) and
    /// every floating-point add happens in the `det_sum_ranks` kernel
    /// in program-text order. The emulated world remains the oracle the
    /// transport path is asserted against.
    #[default]
    FixedOrder,
    /// Library all-reduce. Run-to-run stable on a frozen box, but a
    /// CONFIG contract, not a portability guarantee: the library picks
    /// the association.
    NcclSum,
}

/// How ranks find each other at startup.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Rendezvous {
    /// A shared directory (single node, or a shared filesystem): ranks
    /// coordinate through atomically renamed files under
    /// `dir/job_id/`. The default for self-spawned single-node runs.
    File { dir: PathBuf, job_id: String },
}

impl Default for Rendezvous {
    fn default() -> Self {
        // The empty job id is a deliberate sentinel: the self-spawn
        // supervisor fills in a fresh per-launch id (and tells the
        // children through the environment), while attach() under an
        // external launcher REFUSES it loudly — every rank derives its
        // own default independently, so a process-local id like a PID
        // would put each rank in a different rendezvous and hang the
        // world.
        Rendezvous::File {
            dir: std::env::temp_dir().join("mamba-rs-rendezvous"),
            job_id: String::new(),
        }
    }
}

/// Full configuration of a data-parallel run.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DistConfig {
    pub devices: Devices,
    /// The logical world size W — the numeric identity of the run. When
    /// `None`, W equals the resolved device count. The fold is keyed to
    /// logical ranks, never devices, so replaying a larger-world run on
    /// fewer GPUs is bit-exact BY CONTRACT — but that replay EXECUTION
    /// mode is not wired yet and `bootstrap` refuses W > device count
    /// loudly. Today W must equal the device count (or 1).
    pub logical_world: Option<usize>,
    pub rendezvous: Rendezvous,
    pub reduce: ReduceContract,
    pub seed: u64,
    /// Deadline for all ranks to arrive at the rendezvous.
    pub init_timeout: Duration,
    /// Deadline for one collective window, ENQUEUE THROUGH COMPLETION:
    /// the watchdog holds the window open across the stream sync and
    /// aborts the communicator on expiry, so a peer dying mid-run
    /// becomes a loud rank error instead of an eternal wait. Also the
    /// file-barrier deadline.
    pub collective_timeout: Duration,
}

impl Default for DistConfig {
    fn default() -> Self {
        Self {
            devices: Devices::Single(0),
            logical_world: None,
            rendezvous: Rendezvous::default(),
            reduce: ReduceContract::default(),
            seed: 0,
            init_timeout: Duration::from_secs(120),
            collective_timeout: Duration::from_secs(300),
        }
    }
}

impl DistConfig {
    #[must_use]
    pub fn with_devices(mut self, d: Devices) -> Self {
        self.devices = d;
        self
    }

    #[must_use]
    pub fn with_logical_world(mut self, w: usize) -> Self {
        self.logical_world = Some(w);
        self
    }

    #[must_use]
    pub fn with_rendezvous(mut self, r: Rendezvous) -> Self {
        self.rendezvous = r;
        self
    }

    #[must_use]
    pub fn with_reduce(mut self, r: ReduceContract) -> Self {
        self.reduce = r;
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn validate(&self) -> Result<(), DistError> {
        if let Devices::List(l) = &self.devices {
            if l.is_empty() {
                return Err(DistError::Config("empty device list".into()));
            }
            for (i, ord) in l.iter().enumerate() {
                if l[..i].contains(ord) {
                    return Err(DistError::Config(format!(
                        "device ordinal {ord} repeats in the device list"
                    )));
                }
            }
        }
        if let Devices::Count(0) = self.devices {
            return Err(DistError::Config("device count must be positive".into()));
        }
        if let Some(0) = self.logical_world {
            return Err(DistError::Config("logical world must be positive".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_grammar() {
        assert_eq!(Devices::parse("1").unwrap(), Devices::Single(0));
        assert_eq!(Devices::parse("4").unwrap(), Devices::Count(4));
        assert_eq!(Devices::parse("all").unwrap(), Devices::All);
        assert_eq!(
            Devices::parse("0,2,3").unwrap(),
            Devices::List(vec![0, 2, 3])
        );
        assert!(Devices::parse("0,0").is_err());
        assert!(Devices::parse("").is_err());
        assert!(Devices::parse("0x2").is_err());
    }
}
