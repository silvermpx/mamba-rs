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

/// A job id becomes a directory segment that is later joined and (on
/// the supervisor) recursively DELETED — reject anything that could
/// escape the rendezvous directory.
fn validate_job_id(job: &str) -> Result<(), DistError> {
    if job.contains('/') || job.contains('\\') || job.contains("..") {
        return Err(DistError::Config(format!(
            "job_id {job:?} must be a plain directory segment (no separators, no ..)"
        )));
    }
    Ok(())
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
        // The seed override must survive the single-rank short-circuit —
        // an external launcher that pinned MAMBA_RS_SEED expects it
        // honored at any world size.
        let seed = match read_env(ENV_SEED) {
            Some(s) => s
                .parse()
                .map_err(|_| DistError::EnvContract(format!("{ENV_SEED}={s:?} is not a number")))?,
            None => cfg.seed,
        };
        return Ok(DistContext::single(er.device, seed));
    }
    // An explicit logical_world that disagrees with the launcher's world
    // is a config/launcher mismatch — refuse instead of silently
    // training a different numeric identity.
    if let Some(w) = cfg.logical_world
        && w != er.world
    {
        return Err(DistError::EnvContract(format!(
            "launcher world {} != configured logical_world {w} — a wrapper \
             (srun/torchrun) probably split or collapsed the world",
            er.world
        )));
    }
    // Children may override the rendezvous location through the
    // environment the supervisor set.
    let (dir, job) = match (read_env(ENV_RENDEZVOUS_DIR), read_env(ENV_JOB_ID)) {
        (Some(d), Some(j)) => (PathBuf::from(d), j),
        _ => rendezvous_paths(&cfg.rendezvous),
    };
    if job.is_empty() {
        return Err(DistError::Rendezvous(
            "no job id: multi-process ranks must share ONE rendezvous. Under an \
             external launcher set MAMBA_RS_RENDEZVOUS_DIR + MAMBA_RS_JOB_ID (or \
             pass Rendezvous::File with an explicit, per-launch-unique job_id) — \
             a rank-local default would put every rank in its own directory"
                .into(),
        ));
    }
    let seed = match read_env(ENV_SEED) {
        Some(s) => s
            .parse()
            .map_err(|_| DistError::EnvContract(format!("{ENV_SEED}={s:?} is not a number")))?,
        None => cfg.seed,
    };
    validate_job_id(&job)?;
    let barrier_dir = dir.join(&job);
    std::fs::create_dir_all(&barrier_dir)
        .map_err(|e| DistError::Rendezvous(format!("create {}: {e}", barrier_dir.display())))?;
    #[cfg_attr(not(feature = "nccl"), allow(unused_mut))]
    let mut ctx = DistContext::process(
        er.rank,
        er.world,
        er.device,
        seed,
        cfg.reduce,
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
        // The context handle must stay alive for the communicator's whole
        // lifetime — NCCL binds to the context current at init.
        let cuda_ctx = cudarc::driver::CudaContext::new(er.device)
            .map_err(|e| DistError::Transport(format!("bind device {}: {e:?}", er.device)))?;
        let id_path = barrier_dir.join("nccl-id");
        // One budget for the whole join: the id exchange spends part of
        // init_timeout and the NCCL init gets the remainder, so the
        // combined join can never exceed the configured deadline.
        let join_start = std::time::Instant::now();
        let id = MambaComm::exchange_unique_id(&id_path, er.rank, cfg.init_timeout)?;
        let remaining = cfg
            .init_timeout
            .saturating_sub(join_start.elapsed())
            .max(std::time::Duration::from_secs(1));
        let comm = MambaComm::init_with_deadline(id, er.rank, er.world, cuda_ctx, remaining)?;
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
            "replaying logical world {world} on {} devices is planned (the \
             logical-W replay mode) but not wired yet",
            devices.len()
        )));
    }

    // Supervisor: re-execute this binary once per rank with the rank
    // contract in the child environment, inheriting argv (as OS strings —
    // argv is not required to be UTF-8) so the child re-enters the same
    // code path and lands in the Rank branch above.
    let mut cfg = cfg;
    if let Rendezvous::File { job_id, .. } = &mut cfg.rendezvous
        && job_id.is_empty()
    {
        *job_id = format!("job-{}", std::process::id());
    }
    // A crashed prior run with the same job id would leave barrier
    // markers and a unique-id file behind; a stale marker makes a
    // barrier pass with no peer present and a stale id cross-connects
    // ranks to a dead world. No rank exists yet, so purging is safe.
    {
        let (dir, job) = rendezvous_paths(&cfg.rendezvous);
        validate_job_id(&job)?;
        let _ = std::fs::remove_dir_all(dir.join(job));
    }
    let exe =
        std::env::current_exe().map_err(|e| DistError::Config(format!("current_exe: {e}")))?;
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let mut children: Vec<std::process::Child> = Vec::with_capacity(world);
    for (rank, device) in devices.iter().enumerate() {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(&args);
        for (k, v) in child_env(&cfg, rank, world, *device) {
            cmd.env(k, v);
        }
        // A dead supervisor must not orphan a half-world that keeps
        // training: on Linux the kernel delivers SIGKILL to the child
        // when the parent exits. (Other platforms ride the runbook rule:
        // kill the process group.)
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(child) => children.push(child),
            Err(e) => {
                // Do not leak already-spawned ranks: they would bind
                // their GPUs and block in the communicator init forever
                // waiting for a world that can no longer form.
                for c in &mut children {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                return Err(DistError::Config(format!("spawn rank {rank}: {e}")));
            }
        }
    }
    // Fail-fast wait: poll every child; the FIRST non-zero exit kills
    // the remaining ranks (a mid-run world-size change would silently
    // change the numbers, so a partial world must never keep training).
    let mut exit_codes: Vec<Option<i32>> = vec![None; children.len()];
    loop {
        let mut all_done = true;
        let mut fail_fast = false;
        for (rank, child) in children.iter_mut().enumerate() {
            if exit_codes[rank].is_some() {
                continue;
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code().unwrap_or(-1);
                    exit_codes[rank] = Some(code);
                    if code != 0 {
                        fail_fast = true;
                    }
                }
                Ok(None) => all_done = false,
                Err(e) => {
                    exit_codes[rank] = Some(-1);
                    let _ = e;
                    fail_fast = true;
                }
            }
        }
        if fail_fast {
            for (rank, child) in children.iter_mut().enumerate() {
                if exit_codes[rank].is_none() {
                    let _ = child.kill();
                }
            }
        }
        if all_done {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let exit_codes: Vec<i32> = exit_codes.into_iter().map(|c| c.unwrap_or(-1)).collect();
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
        let mk = |rank: usize| {
            DistContext::process(
                rank,
                2,
                rank,
                0,
                crate::dist::ReduceContract::default(),
                dir.clone(),
                TEST_TIMEOUT,
            )
        };
        let a = std::thread::spawn({
            let ctx = mk(0);
            move || {
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
            }
        });
        let b = std::thread::spawn({
            let ctx = mk(1);
            move || {
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
                ctx.barrier().unwrap();
            }
        });
        a.join().unwrap();
        b.join().unwrap();
        // Generation cleanup: after barrier 2 completes, rank 0 removes
        // gen-0; the two live generations stay.
        assert!(
            !dir.join("gen-0").exists(),
            "stale barrier generation must be cleaned up"
        );
        assert!(dir.join("gen-1").exists());
        assert!(dir.join("gen-2").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn barrier_times_out_without_peers() {
        let dir =
            std::env::temp_dir().join(format!("mamba-rs-barrier-solo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ctx = DistContext::process(
            0,
            2,
            0,
            0,
            crate::dist::ReduceContract::default(),
            dir.clone(),
            Duration::from_millis(50),
        );
        let err = ctx.barrier().unwrap_err();
        assert!(matches!(err, DistError::Rendezvous(_)), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
