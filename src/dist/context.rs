//! Per-process handle on a data-parallel world.
//!
//! Two backings share one API. `Single` is the always-on no-op: world
//! size 1, sharding is identity, reductions return immediately —
//! downstream code compiles and runs unchanged with distribution off.
//! `Process` is a rank in a multi-process world: file-based rendezvous,
//! and — with the `nccl` feature — a live communicator for the
//! transport-backed collectives that exist today.
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
    /// Lazily-built device state for the transport-backed fixed-order
    /// reducer (the fold kernel + stacked receive scratch). Lives on
    /// the context so every reduction reuses one compile.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fixed_order: std::cell::OnceCell<super::reducer::FixedOrderState>,
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
        reduce: ReduceContract,
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
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            fixed_order: std::cell::OnceCell::new(),
        }
    }

    pub(super) fn process(
        rank: usize,
        world: usize,
        device: usize,
        seed: u64,
        reduce: ReduceContract,
        barrier_dir: PathBuf,
        barrier_timeout: Duration,
    ) -> Self {
        Self {
            inner: ContextInner::Process {
                rank,
                world,
                device,
                seed: SeedLaw::new(seed),
                reduce,
                barrier_dir,
                barrier_generation: std::cell::Cell::new(0),
                barrier_timeout,
                #[cfg(feature = "nccl")]
                comm: None,
            },
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            fixed_order: std::cell::OnceCell::new(),
        }
    }

    /// Attach an initialized communicator (bootstrap does this for
    /// multi-process worlds when the transport feature is on).
    #[cfg(feature = "nccl")]
    pub(super) fn set_comm(&mut self, c: super::comm::MambaComm) {
        if let ContextInner::Process { comm, world, .. } = &mut self.inner {
            debug_assert_eq!(
                c.world(),
                *world,
                "communicator world diverges from the context world"
            );
            *comm = Some(c);
        }
    }

    /// In-place SUM of the flat f32 gradient arena across ranks — the
    /// transport half of the gradient exchange. The mean scale and the
    /// optimizer tail stay with the trainer (sum then multiply by 1/W,
    /// exact for power-of-two worlds). Single-process worlds return
    /// immediately.
    ///
    /// Both contracts are transport-backed. `FixedOrder` (the default)
    /// runs the house reducer: peer addends move as pure bytes (NCCL
    /// send/recv/broadcast — no library arithmetic) and every
    /// floating-point add happens in the `det_sum_ranks` kernel in
    /// strictly ascending source-rank order, so the reduced bits are
    /// independent of transport, delivery order, topology, and library
    /// version. `NcclSum` runs the library collective — run-to-run
    /// stable on a frozen box configuration, without the fixed-order
    /// portability guarantee.
    #[cfg(feature = "cuda")]
    pub fn all_reduce_grad_sum(
        &self,
        arena: &mut crate::mamba_ssm::gpu::buffers::GpuBuffer,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(()),
            #[cfg(feature = "nccl")]
            ContextInner::Process {
                reduce: ReduceContract::FixedOrder,
                comm: Some(c),
                device,
                barrier_timeout,
                ..
            } => {
                let st = self.fixed_order_state(*device)?;
                c.with_watchdog("fixed-order-reduce", *barrier_timeout, || {
                    super::reducer::reduce_sum_fixed_order_nccl(c, st, arena, stream)?;
                    // Completion INSIDE the guarded window: a peer dying
                    // mid-run hangs this sync, the watchdog aborts the
                    // communicator, the sync unblocks, and the window
                    // reports the deadline — real fail-fast, not an
                    // enqueue-only illusion. The sync also serializes
                    // every collective on this communicator in host
                    // program order (the NCCL single-comm requirement)
                    // and quiesces the reducer scratch between uses.
                    stream
                        .synchronize()
                        .map_err(|e| DistError::Transport(format!("reduce sync: {e:?}")))
                })
            }
            #[cfg(feature = "nccl")]
            ContextInner::Process {
                reduce: ReduceContract::NcclSum,
                comm: Some(c),
                barrier_timeout,
                ..
            } => c.with_watchdog("nccl-sum", *barrier_timeout, || {
                c.all_reduce_sum_f32(arena.cached_ptr(), arena.len(), stream)?;
                stream
                    .synchronize()
                    .map_err(|e| DistError::Transport(format!("reduce sync: {e:?}")))
            }),
            // Deliberately NOT a catch-all over the contract: the two
            // comm-bearing arms above enumerate every ReduceContract
            // variant, so adding a third variant fails compilation here
            // instead of silently landing in the no-comm error below.
            #[cfg(feature = "nccl")]
            ContextInner::Process { comm: None, .. } => Err(DistError::Transport(
                "no communicator attached to this rank (bootstrap did not \
                 initialize one)"
                    .into(),
            )),
            #[cfg(not(feature = "nccl"))]
            ContextInner::Process { .. } => {
                // Without the nccl feature this is the only Process path
                // and the operands go unused — bind them so the
                // cuda-without-nccl build stays warning-free.
                let _ = (&arena, &stream);
                Err(DistError::Transport(
                    "no communicator attached to this rank (built without the nccl \
                     feature)"
                        .into(),
                ))
            }
        }
    }

    /// The lazily-compiled device state for the fixed-order reducer.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn fixed_order_state(
        &self,
        device: usize,
    ) -> Result<&super::reducer::FixedOrderState, DistError> {
        if self.fixed_order.get().is_none() {
            let st = super::reducer::FixedOrderState::compile(device)?;
            // A concurrent set is impossible (the context is used from
            // its owning thread); a lost race would only drop a spare.
            let _ = self.fixed_order.set(st);
        }
        let Some(st) = self.fixed_order.get() else {
            return Err(DistError::Transport(
                "fixed-order reducer state missing after initialization".into(),
            ));
        };
        Ok(st)
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

    /// The reduction contract this world was configured with. Part of
    /// the run's numeric identity — training harnesses should stamp it
    /// into their checkpoint sidecars alongside the GEMM tier and scan
    /// mode (the classify trainer's refuse-on-drift pattern).
    pub fn reduce_contract(&self) -> ReduceContract {
        match &self.inner {
            ContextInner::Single { .. } => ReduceContract::default(),
            ContextInner::Process { reduce, .. } => *reduce,
        }
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
    ///
    /// Collective-count warning: when the global length is not a
    /// multiple of W, ranks receive UNEQUAL item counts — a training
    /// loop that reduces once per item would desynchronize the world on
    /// the tail (some ranks enter a collective the others never post).
    /// Reduce once per optimizer STEP over a schedule derived from the
    /// global length (every rank computes the same step count), or drop
    /// the tail.
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
                file_barrier(barrier_dir, generation, *rank, *world, *barrier_timeout)?;
                // Keep at most two generations on disk. Once generation g
                // completes, every rank has already returned from g-1 (a
                // rank writes its g marker only after exiting the g-1
                // wait loop), so g-2 is provably dead; rank 0 removes it
                // best-effort — a failure leaks a directory, never blocks.
                if *rank == 0 && generation >= 2 {
                    let dead = barrier_dir.join(format!("gen-{}", generation - 2));
                    let _ = std::fs::remove_dir_all(dead);
                }
                Ok(())
            }
        }
    }

    /// Sum-then-mean over a host f32 buffer across ranks (the seam for
    /// small CPU-side heads riding a GPU backbone). Single-world: no-op.
    /// The reduction rides the configured contract — the fixed-order
    /// house reducer or the library sum — after a device round-trip
    /// (NCCL only moves device memory).
    ///
    /// Cross-rank contract: `xs.len()` must agree on every rank,
    /// INCLUDING emptiness — a world where some ranks pass an empty
    /// buffer and others do not desynchronizes the communicator (the
    /// empty ranks skip the collective the rest are blocked in).
    pub fn all_reduce_host_f32(&self, xs: &mut [f32]) -> Result<(), DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(()),
            #[cfg(feature = "nccl")]
            ContextInner::Process {
                comm: Some(c),
                reduce,
                device,
                world,
                barrier_timeout,
                ..
            } => {
                if xs.is_empty() {
                    return Ok(());
                }
                let gpu = crate::mamba_ssm::gpu::device::GpuDevice::new(*device)
                    .map_err(|e| DistError::Transport(format!("host reduce device: {e}")))?;
                let stream = gpu.context().default_stream();
                let mut buf = crate::mamba_ssm::gpu::buffers::GpuBuffer::from_cpu(&stream, xs)
                    .map_err(|e| DistError::Transport(format!("host reduce stage: {e}")))?;
                let sync = |tag: &str| {
                    stream
                        .synchronize()
                        .map_err(move |e| DistError::Transport(format!("{tag} sync: {e:?}")))
                };
                match reduce {
                    ReduceContract::FixedOrder => {
                        let st = self.fixed_order_state(*device)?;
                        c.with_watchdog("host-fixed-order", *barrier_timeout, || {
                            super::reducer::reduce_sum_fixed_order_nccl(c, st, &mut buf, &stream)?;
                            sync("host fixed-order reduce")
                        })?;
                    }
                    ReduceContract::NcclSum => {
                        c.with_watchdog("host-nccl-sum", *barrier_timeout, || {
                            c.all_reduce_sum_f32(buf.cached_ptr(), buf.len(), &stream)?;
                            sync("host nccl-sum reduce")
                        })?;
                    }
                }
                let summed = buf
                    .to_cpu(&stream)
                    .map_err(|e| DistError::Transport(format!("host reduce readback: {e}")))?;
                let inv_w = 1.0f32 / *world as f32;
                for (x, s) in xs.iter_mut().zip(&summed) {
                    *x = s * inv_w;
                }
                Ok(())
            }
            ContextInner::Process { .. } => {
                let _ = &xs;
                Err(DistError::Transport(
                    "no communicator attached to this rank (built without the nccl \
                     feature, or bootstrap did not initialize one)"
                        .into(),
                ))
            }
        }
    }

    /// Logical OR across ranks (encoded as an integer maximum — exact
    /// and order-independent by construction, so it is contract-neutral).
    /// Single-world: returns the local flag.
    pub fn any(&self, flag: bool) -> Result<bool, DistError> {
        match &self.inner {
            ContextInner::Single { .. } => Ok(flag),
            #[cfg(feature = "nccl")]
            ContextInner::Process {
                comm: Some(c),
                device,
                barrier_timeout,
                ..
            } => {
                let gpu = crate::mamba_ssm::gpu::device::GpuDevice::new(*device)
                    .map_err(|e| DistError::Transport(format!("flag reduce device: {e}")))?;
                let stream = gpu.context().default_stream();
                let staged = stream
                    .clone_htod(&[i32::from(flag)])
                    .map_err(|e| DistError::Transport(format!("flag stage: {e:?}")))?;
                {
                    use cudarc::driver::DevicePtr;
                    let (ptr, _guard) = staged.device_ptr(&stream);
                    c.with_watchdog("flag-reduce", *barrier_timeout, || {
                        c.all_reduce_max_i32(ptr, 1, &stream)?;
                        stream
                            .synchronize()
                            .map_err(|e| DistError::Transport(format!("flag sync: {e:?}")))
                    })?;
                }
                let back: Vec<i32> = stream
                    .clone_dtoh(&staged)
                    .map_err(|e| DistError::Transport(format!("flag readback: {e:?}")))?;
                Ok(back.first().copied().unwrap_or(0) != 0)
            }
            ContextInner::Process { .. } => Err(DistError::Transport(
                "no communicator attached to this rank (built without the nccl \
                 feature, or bootstrap did not initialize one)"
                    .into(),
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
    /// Validates its inputs like [`Self::all_reduce_mean`] — an oracle
    /// that silently mis-indexes on malformed input proves nothing.
    pub fn reference_mean(&self, arenas: &[Vec<f32>]) -> Result<Vec<f32>, DistError> {
        if arenas.len() != self.world {
            return Err(DistError::Config(format!(
                "expected {} rank arenas, got {}",
                self.world,
                arenas.len()
            )));
        }
        let n = arenas[0].len();
        for (r, a) in arenas.iter().enumerate() {
            if a.len() != n {
                return Err(DistError::Config(format!(
                    "arena length mismatch: rank 0 has {n}, rank {r} has {}",
                    a.len()
                )));
            }
        }
        let views: Vec<&[f32]> = arenas.iter().map(|a| a.as_slice()).collect();
        let mut out = vec![0.0f32; n];
        reduce_mean_reference(&views, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_is_the_strided_slice_law() {
        let ctx0 = DistContext::process(
            0,
            3,
            0,
            7,
            ReduceContract::default(),
            std::env::temp_dir(),
            Duration::from_secs(1),
        );
        let ctx2 = DistContext::process(
            2,
            3,
            2,
            7,
            ReduceContract::default(),
            std::env::temp_dir(),
            Duration::from_secs(1),
        );
        let global: Vec<u32> = (0..10).collect();
        let r0: Vec<u32> = ctx0.shard(&global).copied().collect();
        let r2: Vec<u32> = ctx2.shard(&global).copied().collect();
        assert_eq!(r0, vec![0, 3, 6, 9], "rank 0 takes k % 3 == 0");
        assert_eq!(r2, vec![2, 5, 8], "rank 2 takes k % 3 == 2");
        // Single-world context is the identity slice.
        let s = DistContext::single(0, 7);
        let all: Vec<u32> = s.shard(&global).copied().collect();
        assert_eq!(all, global);
    }

    #[test]
    fn reference_mean_validates_like_its_sibling() {
        let ew = EmulatedWorld::new(2).unwrap();
        assert!(
            ew.reference_mean(&[vec![1.0]]).is_err(),
            "wrong arena count"
        );
        assert!(
            ew.reference_mean(&[vec![1.0], vec![1.0, 2.0]]).is_err(),
            "length mismatch"
        );
        let out = ew.reference_mean(&[vec![2.0], vec![4.0]]).unwrap();
        assert_eq!(out, vec![3.0]);
    }
}
