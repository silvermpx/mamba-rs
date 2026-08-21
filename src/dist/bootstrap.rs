//! Process bootstrap: how a run becomes a world of ranks.
//!
//! Two entry styles cover every launcher:
//!
//! - `bootstrap(cfg)`: a process started WITHOUT a rank in its
//!   environment becomes the supervisor — it resolves the device set,
//!   re-executes itself once per rank with the rank contract set in the
//!   child environment, waits for all children, and aggregates their
//!   exit codes. Children re-enter `bootstrap`, see their rank, and get
//!   a rank context back. Single-device configs short-circuit: no
//!   spawn, no rendezvous, numerics byte-identical to a plain run.
//! - `attach(cfg)`: for ranks launched by an external launcher
//!   (torchrun/srun/mpirun) — reads the rank contract from the
//!   environment (own variables first, then the common conventions) and
//!   joins the world without spawning anything.

use std::path::PathBuf;

use super::config::{Devices, DistConfig, Rendezvous};
use super::context::DistContext;
use super::error::DistError;

/// Environment contract set by the supervisor for each child rank.
pub const ENV_RANK: &str = "MAMBA_RS_RANK";
pub const ENV_WORLD: &str = "MAMBA_RS_WORLD";
pub const ENV_DEVICE: &str = "MAMBA_RS_DEVICE";
pub const ENV_RENDEZVOUS_DIR: &str = "MAMBA_RS_RENDEZVOUS_DIR";
pub const ENV_JOB_ID: &str = "MAMBA_RS_JOB_ID";
pub const ENV_SEED: &str = "MAMBA_RS_SEED";

/// What `bootstrap` returned: this process is either the supervisor
/// (children already ran to completion) or one rank of the world.
pub enum Bootstrap {
    Supervisor(SupervisorStatus),
    Rank(DistContext),
}

/// Exit summary of a supervised world.
pub struct SupervisorStatus {
    /// Per-rank exit codes, index = logical rank.
    pub exit_codes: Vec<i32>,
}

impl SupervisorStatus {
    pub fn all_ok(&self) -> bool {
        self.exit_codes.iter().all(|&c| c == 0)
    }
}

/// Rank contract read from the environment, if any.
struct EnvRank {
    rank: usize,
    world: usize,
    device: usize,
}

fn read_env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn parse_usize(k: &str, v: &str) -> Result<usize, DistError> {
    v.parse()
        .map_err(|_| DistError::EnvContract(format!("{k}={v:?} is not a number")))
}

/// Read the rank contract with the fallback chain: our own variables
/// first, then the torchrun convention, then SLURM, then OpenMPI.
/// Returns `None` when no launcher set a rank — the caller is either a
/// plain single-process run or the supervisor about to spawn one.
fn env_rank() -> Result<Option<EnvRank>, DistError> {
    if let Some(r) = read_env(ENV_RANK) {
        let rank = parse_usize(ENV_RANK, &r)?;
        let world = match read_env(ENV_WORLD) {
            Some(w) => parse_usize(ENV_WORLD, &w)?,
            None => {
                return Err(DistError::EnvContract(format!(
                    "{ENV_RANK} is set but {ENV_WORLD} is missing"
                )));
            }
        };
        let device = match read_env(ENV_DEVICE) {
            Some(d) => parse_usize(ENV_DEVICE, &d)?,
            None => rank,
        };
        return Ok(Some(EnvRank {
            rank,
            world,
            device,
        }));
    }
    // torchrun convention.
    if let (Some(r), Some(w)) = (read_env("RANK"), read_env("WORLD_SIZE")) {
        let rank = parse_usize("RANK", &r)?;
        let world = parse_usize("WORLD_SIZE", &w)?;
        let device = match read_env("LOCAL_RANK") {
            Some(l) => parse_usize("LOCAL_RANK", &l)?,
            None => rank,
        };
        return Ok(Some(EnvRank {
            rank,
            world,
            device,
        }));
    }
    // SLURM.
    if let (Some(r), Some(w)) = (read_env("SLURM_PROCID"), read_env("SLURM_NTASKS")) {
        let rank = parse_usize("SLURM_PROCID", &r)?;
        let world = parse_usize("SLURM_NTASKS", &w)?;
        let device = match read_env("SLURM_LOCALID") {
            Some(l) => parse_usize("SLURM_LOCALID", &l)?,
            None => rank,
        };
        return Ok(Some(EnvRank {
            rank,
            world,
            device,
        }));
    }
    // OpenMPI.
    if let (Some(r), Some(w)) = (
        read_env("OMPI_COMM_WORLD_RANK"),
        read_env("OMPI_COMM_WORLD_SIZE"),
    ) {
        let rank = parse_usize("OMPI_COMM_WORLD_RANK", &r)?;
        let world = parse_usize("OMPI_COMM_WORLD_SIZE", &w)?;
        let device = match read_env("OMPI_COMM_WORLD_LOCAL_RANK") {
            Some(l) => parse_usize("OMPI_COMM_WORLD_LOCAL_RANK", &l)?,
            None => rank,
        };
        return Ok(Some(EnvRank {
            rank,
            world,
            device,
        }));
    }
    Ok(None)
}

fn rendezvous_paths(r: &Rendezvous) -> (PathBuf, String) {
    match r {
        Rendezvous::File { dir, job_id } => (dir.clone(), job_id.clone()),
        // Non-exhaustive enum: future variants must be handled here
        // when they are added.
        #[allow(
            unreachable_patterns,
            reason = "Rendezvous is non_exhaustive for future Tcp/Preset variants; \
                      today File is the only one"
        )]
        _ => unreachable!("unhandled rendezvous variant"),
    }
}

/// Resolve the device ordinal list a config asks for.
fn resolve_devices(d: &Devices) -> Result<Vec<usize>, DistError> {
    match d {
        Devices::Single(ord) => Ok(vec![*ord]),
        Devices::Count(n) => Ok((0..*n).collect()),
        Devices::List(l) => Ok(l.clone()),
        Devices::All => {
            #[cfg(feature = "cuda")]
            {
                let n = cudarc::driver::CudaContext::device_count()
                    .map_err(|e| DistError::Config(format!("device count query: {e:?}")))?;
                if n <= 0 {
                    return Err(DistError::Config("no CUDA devices visible".into()));
                }
                Ok((0..n as usize).collect())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(DistError::Config(
                    "Devices::All needs the cuda feature to enumerate devices".into(),
                ))
            }
        }
    }
}

/// The environment a child rank receives. Split out so the spawn
/// contract is unit-testable without launching processes.
fn child_env(cfg: &DistConfig, rank: usize, world: usize, device: usize) -> Vec<(String, String)> {
    let (dir, job) = rendezvous_paths(&cfg.rendezvous);
    vec![
        (ENV_RANK.into(), rank.to_string()),
        (ENV_WORLD.into(), world.to_string()),
        (ENV_DEVICE.into(), device.to_string()),
        (ENV_RENDEZVOUS_DIR.into(), dir.display().to_string()),
        (ENV_JOB_ID.into(), job),
        (ENV_SEED.into(), cfg.seed.to_string()),
    ]
}

/// Build the rank context for this process from a config plus the
/// resolved rank contract.
fn rank_context(cfg: &DistConfig, er: EnvRank) -> Result<DistContext, DistError> {
    if er.world == 0 || er.rank >= er.world {
        return Err(DistError::EnvContract(format!(
            "rank {} outside world {}",
            er.rank, er.world
        )));
    }
    if er.world == 1 {
        return Ok(DistContext::single(er.device, cfg.seed));
    }
    // Children may override the rendezvous location through the
    // environment the supervisor set.
    let (dir, job) = match (read_env(ENV_RENDEZVOUS_DIR), read_env(ENV_JOB_ID)) {
        (Some(d), Some(j)) => (PathBuf::from(d), j),
        _ => rendezvous_paths(&cfg.rendezvous),
    };
    let seed = match read_env(ENV_SEED) {
        Some(s) => s
            .parse()
            .map_err(|_| DistError::EnvContract(format!("{ENV_SEED}={s:?} is not a number")))?,
        None => cfg.seed,
    };
    let barrier_dir = dir.join(&job);
    std::fs::create_dir_all(&barrier_dir)
        .map_err(|e| DistError::Rendezvous(format!("create {}: {e}", barrier_dir.display())))?;
    #[cfg_attr(not(feature = "nccl"), allow(unused_mut))]
    let mut ctx = DistContext::process(
        er.rank,
        er.world,
        er.device,
        seed,
        barrier_dir.clone(),
        cfg.collective_timeout,
    );
    // With the transport feature on, join the NCCL world here: bind the
    // rank's CUDA device (the primary context the trainer will reuse),
    // exchange the unique id through the job's rendezvous directory, and
    // run the blocking init.
    #[cfg(feature = "nccl")]
    {
        use super::comm::MambaComm;
        MambaComm::preflight_version()?;
        cudarc::driver::CudaContext::new(er.device)
            .map_err(|e| DistError::Transport(format!("bind device {}: {e:?}", er.device)))?;
        let id_path = barrier_dir.join("nccl-id");
        let id = MambaComm::exchange_unique_id(&id_path, er.rank, cfg.init_timeout)?;
        let comm = MambaComm::init(id, er.rank, er.world)?;
        ctx.set_comm(comm);
    }
    Ok(ctx)
}

/// Join a world whose ranks an EXTERNAL launcher started. Errs when the
/// environment carries no rank contract.
pub fn attach(cfg: DistConfig) -> Result<DistContext, DistError> {
    cfg.validate()?;
    match env_rank()? {
        Some(er) => rank_context(&cfg, er),
        None => Err(DistError::EnvContract(
            "no rank in the environment — use bootstrap() for self-spawned runs".into(),
        )),
    }
}

/// Become a world. See the module docs for the supervisor/rank split.
pub fn bootstrap(cfg: DistConfig) -> Result<Bootstrap, DistError> {
    cfg.validate()?;
    // A rank contract in the environment means a supervisor (or an
    // external launcher) already placed this process.
    if let Some(er) = env_rank()? {
        return Ok(Bootstrap::Rank(rank_context(&cfg, er)?));
    }
    let devices = resolve_devices(&cfg.devices)?;
    let world = cfg.logical_world.unwrap_or(devices.len());
    if world < devices.len() {
        return Err(DistError::Config(format!(
            "logical world {world} smaller than the device list ({})",
            devices.len()
        )));
    }
    if world == 1 {
        return Ok(Bootstrap::Rank(DistContext::single(devices[0], cfg.seed)));
    }
    if world > devices.len() {
        return Err(DistError::Config(format!(
            "replaying logical world {world} on {} devices is not wired yet — \
             it arrives with the communicator layer",
            devices.len()
        )));
    }

    // Supervisor: re-execute this binary once per rank with the rank
    // contract in the child environment, inheriting argv so the child
    // re-enters the same code path and lands in the Rank branch above.
    let exe =
        std::env::current_exe().map_err(|e| DistError::Config(format!("current_exe: {e}")))?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut children = Vec::with_capacity(world);
    for (rank, device) in devices.iter().enumerate() {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(&args);
        for (k, v) in child_env(&cfg, rank, world, *device) {
            cmd.env(k, v);
        }
        let child = cmd
            .spawn()
            .map_err(|e| DistError::Config(format!("spawn rank {rank}: {e}")))?;
        children.push(child);
    }
    let mut exit_codes = Vec::with_capacity(world);
    for (rank, mut child) in children.into_iter().enumerate() {
        let status = child.wait().map_err(|e| DistError::RankFailed {
            rank,
            detail: format!("wait: {e}"),
        })?;
        exit_codes.push(status.code().unwrap_or(-1));
    }
    Ok(Bootstrap::Supervisor(SupervisorStatus { exit_codes }))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Deadline used by tests to keep barrier failures fast.
    const TEST_TIMEOUT: Duration = Duration::from_secs(10);

    #[test]
    fn child_env_carries_the_full_contract() {
        let cfg = DistConfig::default()
            .with_devices(Devices::Count(2))
            .with_seed(7)
            .with_rendezvous(Rendezvous::File {
                dir: PathBuf::from("/tmp/rdzv"),
                job_id: "j1".into(),
            });
        let env = child_env(&cfg, 1, 2, 3);
        let get = |k: &str| {
            env.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        assert_eq!(get(ENV_RANK), "1");
        assert_eq!(get(ENV_WORLD), "2");
        assert_eq!(get(ENV_DEVICE), "3");
        assert_eq!(get(ENV_RENDEZVOUS_DIR), "/tmp/rdzv");
        assert_eq!(get(ENV_JOB_ID), "j1");
        assert_eq!(get(ENV_SEED), "7");
    }

    #[test]
    fn file_barrier_two_threads_meet() {
        let dir =
            std::env::temp_dir().join(format!("mamba-rs-barrier-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mk = |rank: usize| DistContext::process(rank, 2, rank, 0, dir.clone(), TEST_TIMEOUT);
        let a = std::thread::spawn({
            let ctx = mk(0);
            move || {
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
            }
        });
        let b = std::thread::spawn({
            let ctx = mk(1);
            move || {
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
            }
        });
        a.join().unwrap();
        b.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn barrier_times_out_without_peers() {
        let dir =
            std::env::temp_dir().join(format!("mamba-rs-barrier-solo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ctx = DistContext::process(0, 2, 0, 0, dir.clone(), Duration::from_millis(50));
        let err = ctx.barrier().unwrap_err();
        assert!(matches!(err, DistError::Rendezvous(_)), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
