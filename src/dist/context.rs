//! Per-process handle on a data-parallel world.
//!
//! Three backings share one API. `Single` is the always-on no-op: world
//! size 1, sharding is identity, reductions return immediately —
//! downstream code compiles and runs unchanged with distribution off.
//! `Process` is a rank in a multi-process world (file-based rendezvous
//! today; the byte transport arrives with the communicator layer).
//! [`EmulatedWorld`] lives beside them for single-process oracle tests
//! of the reduction contract.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::config::ReduceContract;
use super::error::DistError;
use super::fold::{reduce_mean_reference, shard_plan};
use super::seed::SeedLaw;

pub struct DistContext {
    inner: ContextInner,
}

enum ContextInner {
    Single {
        device: usize,
        seed: SeedLaw,
    },
    Process {
        rank: usize,
        world: usize,
        device: usize,
        seed: SeedLaw,
        barrier_dir: PathBuf,
        barrier_generation: std::cell::Cell<u64>,
        barrier_timeout: Duration,
        #[cfg(feature = "nccl")]
        comm: Option<super::comm::MambaComm>,
    },
}

impl DistContext {
    /// The no-op single-process context.
    pub fn single(device: usize, seed: u64) -> Self {
        Self {
            inner: ContextInner::Single {
                device,
                seed: SeedLaw::new(seed),
            },
        }
    }

    pub(super) fn process(
        rank: usize,
        world: usize,
        device: usize,
        seed: u64,
        barrier_dir: PathBuf,
        barrier_timeout: Duration,
    ) -> Self {
        Self {
            inner: ContextInner::Process {
                rank,
                world,
                device,
                seed: SeedLaw::new(seed),
                barrier_dir,
                barrier_generation: std::cell::Cell::new(0),
                barrier_timeout,
                #[cfg(feature = "nccl")]
                comm: None,
            },
        }
    }

    /// Attach an initialized communicator (bootstrap does this for
    /// multi-process worlds when the transport feature is on).
    #[cfg(feature = "nccl")]
    pub(super) fn set_comm(&mut self, c: super::comm::MambaComm) {
        if let ContextInner::Process { comm, .. } = &mut self.inner {
            *comm = Some(c);
        }
    }

    /// In-place SUM of the flat f32 gradient arena across ranks — the
    /// transport half of the gradient exchange. The mean scale and the
    /// optimizer tail stay with the trainer (sum then multiply by 1/W,
    /// exact for power-of-two worlds). Single-process worlds return
    /// immediately.
    #[cfg(feature = "cuda")]
    pub fn all_reduce_grad_sum(
        &self,
        arena: &mut crate::mamba_ssm::gpu::buffers::GpuBuffer,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(()),
            #[cfg(feature = "nccl")]
            ContextInner::Process { comm: Some(c), .. } => {
                c.all_reduce_sum_f32(arena.cached_ptr(), arena.len(), stream)
            }
            ContextInner::Process { .. } => Err(DistError::Transport(
                "no communicator attached to this rank (built without the nccl \
                 feature, or bootstrap did not initialize one)"
                    .into(),
            )),
        }
    }

    /// Logical rank of this process.
    pub fn rank(&self) -> usize {
        match &self.inner {
            ContextInner::Single { .. } => 0,
            ContextInner::Process { rank, .. } => *rank,
        }
    }

    /// Logical world size W — the numeric identity of the run.
    pub fn world_size(&self) -> usize {
        match &self.inner {
            ContextInner::Single { .. } => 1,
            ContextInner::Process { world, .. } => *world,
        }
    }

    /// CUDA device ordinal this rank should construct its trainer on.
    pub fn device_ordinal(&self) -> usize {
        match &self.inner {
            ContextInner::Single { device, .. } => *device,
            ContextInner::Process { device, .. } => *device,
        }
    }

    /// True on exactly one rank — the one that writes checkpoints and logs.
    pub fn is_leader(&self) -> bool {
        self.rank() == 0
    }

    /// The seed law all ranks share.
    pub fn seed_law(&self) -> SeedLaw {
        match &self.inner {
            ContextInner::Single { seed, .. } => *seed,
            ContextInner::Process { seed, .. } => *seed,
        }
    }

    /// This rank's strided slice of a rank-identical global order: item
    /// k goes to rank `k % W`. Taken AFTER the global permutation is
    /// fixed, so the sample set at any optimizer step is
    /// world-size-invariant.
    pub fn shard<'a, T>(&self, global: &'a [T]) -> impl Iterator<Item = &'a T> + 'a {
        let world = self.world_size();
        let rank = self.rank();
        global
            .iter()
            .enumerate()
            .filter(move |(k, _)| k % world == rank)
            .map(|(_, v)| v)
    }

    /// Wait until every rank reaches the same barrier call. Single-world
    /// contexts return immediately.
    pub fn barrier(&self) -> Result<(), DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(()),
            ContextInner::Process {
                rank,
                world,
                barrier_dir,
                barrier_generation,
                barrier_timeout,
                ..
            } => {
                let generation = barrier_generation.get();
                barrier_generation.set(generation + 1);
                file_barrier(barrier_dir, generation, *rank, *world, *barrier_timeout)
            }
        }
    }

    /// Sum-then-mean over a host f32 buffer across ranks (the seam for
    /// small CPU-side heads riding a GPU backbone). Single-world: no-op.
    pub fn all_reduce_host_f32(&self, xs: &mut [f32]) -> Result<(), DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(()),
            ContextInner::Process { .. } => {
                let _ = xs;
                Err(DistError::Transport(
                    "cross-process reduction arrives with the communicator layer".into(),
                ))
            }
        }
    }

    /// Logical OR across ranks (encoded as an integer maximum — exact,
    /// dtype-independent). Single-world: returns the local flag.
    pub fn any(&self, flag: bool) -> Result<bool, DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(flag),
            ContextInner::Process { .. } => Err(DistError::Transport(
                "cross-process reduction arrives with the communicator layer".into(),
            )),
        }
    }
}

/// One generation of a file-based barrier: each rank atomically creates
/// its marker under `dir/gen-<g>/`, then waits until all `world` markers
/// exist. Atomic rename is the only filesystem primitive relied on.
fn file_barrier(
    dir: &std::path::Path,
    generation: u64,
    rank: usize,
    world: usize,
    timeout: Duration,
) -> Result<(), DistError> {
    let gen_dir = dir.join(format!("gen-{generation}"));
    std::fs::create_dir_all(&gen_dir)
        .map_err(|e| DistError::Rendezvous(format!("create {}: {e}", gen_dir.display())))?;
    let tmp = gen_dir.join(format!(".rank-{rank}.tmp"));
    let dst = gen_dir.join(format!("rank-{rank}"));
    std::fs::write(&tmp, b"ok")
        .map_err(|e| DistError::Rendezvous(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &dst)
        .map_err(|e| DistError::Rendezvous(format!("rename {}: {e}", dst.display())))?;
    let deadline = Instant::now() + timeout;
    loop {
        let mut present = 0usize;
        for r in 0..world {
            if gen_dir.join(format!("rank-{r}")).exists() {
                present += 1;
            }
        }
        if present == world {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(DistError::Rendezvous(format!(
                "barrier generation {generation}: {present}/{world} ranks after {timeout:?}"
            )));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Single-process emulation of a W-rank world for oracle tests: all rank
/// arenas live in this process, and the reduction runs the SAME sharded
/// dataflow a real transport uses — shard owners collect the other
/// ranks' copies of their shard (in any delivery order), fold the
/// addends per element in strictly ascending logical-rank order, scale
/// by `1/W`, and broadcast the reduced shard back. Its bits are the
/// contract every real transport must reproduce.
pub struct EmulatedWorld {
    world: usize,
    reduce: ReduceContract,
}

impl EmulatedWorld {
    pub fn new(world: usize) -> Result<Self, DistError> {
        if world == 0 {
            return Err(DistError::Config("world size must be positive".into()));
        }
        Ok(Self {
            world,
            reduce: ReduceContract::FixedOrder,
        })
    }

    pub fn world_size(&self) -> usize {
        self.world
    }

    pub fn reduce_contract(&self) -> ReduceContract {
        self.reduce
    }

    /// Reduce all rank arenas to their mean, in place, via the sharded
    /// owner fold. `delivery_order` optionally scrambles the order in
    /// which each owner receives peer contributions — the result must
    /// not depend on it, and the equivalence test proves exactly that.
    pub fn all_reduce_mean(
        &self,
        arenas: &mut [Vec<f32>],
        delivery_order: Option<&[usize]>,
    ) -> Result<(), DistError> {
        if arenas.len() != self.world {
            return Err(DistError::Config(format!(
                "expected {} rank arenas, got {}",
                self.world,
                arenas.len()
            )));
        }
        let n = arenas[0].len();
        if arenas.iter().any(|a| a.len() != n) {
            return Err(DistError::Config("rank arena lengths differ".into()));
        }
        if let Some(order) = delivery_order {
            let mut seen: Vec<bool> = vec![false; self.world];
            for &r in order {
                if r >= self.world || seen[r] {
                    return Err(DistError::Config(format!("bad delivery order {order:?}")));
                }
                seen[r] = true;
            }
            if seen.iter().any(|s| !s) {
                return Err(DistError::Config(format!("bad delivery order {order:?}")));
            }
        }
        let inv_w = 1.0f32 / self.world as f32;
        let plan = shard_plan(n, self.world);

        // Owner phase: each rank reduces its own shard. Contributions
        // arrive in `delivery_order` (a transport artifact), but land in
        // a rank-indexed staging table, so the fold below reads them in
        // ascending logical-rank order regardless of arrival.
        let mut reduced: Vec<Vec<f32>> = Vec::with_capacity(self.world);
        for (owner, shard) in plan.iter().enumerate() {
            let mut staging: Vec<&[f32]> = vec![&[]; self.world];
            let arrival: Vec<usize> = match delivery_order {
                Some(o) => o.to_vec(),
                None => (0..self.world).collect(),
            };
            for r in arrival {
                staging[r] = &arenas[r][shard.start..shard.start + shard.len];
            }
            let mut out = vec![0.0f32; shard.len];
            for (i, o) in out.iter_mut().enumerate() {
                let mut acc = staging[0][i];
                for s in &staging[1..] {
                    acc += s[i];
                }
                *o = acc * inv_w;
            }
            let _ = owner;
            reduced.push(out);
        }

        // All-gather phase: pure copies of the reduced shards back into
        // every rank arena.
        for arena in arenas.iter_mut() {
            for (shard, red) in plan.iter().zip(&reduced) {
                arena[shard.start..shard.start + shard.len].copy_from_slice(red);
            }
        }
        Ok(())
    }

    /// The straight-line reference: full-arena ascending fold, no
    /// sharding. The sharded dataflow above must match it bit-for-bit.
    pub fn reference_mean(&self, arenas: &[Vec<f32>]) -> Vec<f32> {
        let views: Vec<&[f32]> = arenas.iter().map(|a| a.as_slice()).collect();
        let mut out = vec![0.0f32; arenas[0].len()];
        reduce_mean_reference(&views, &mut out);
        out
    }
}
