//! Printed output hashes for kernels without digest coverage: record once,
//! compare forever. Declare `common/digest.rs`, `common/evidence.rs` and
//! `common/evidence_digest.rs` beside this module; it reaches them through
//! `super`.

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;

use super::digest::fnv1a_f32;

/// Print one `HASH <name> <hex>` line per buffer (f32 elements, `n`
/// leading elements each). The bit gate for kernels without digest
/// coverage: record once, compare forever.
pub fn hash_outputs(ctx: &GpuCtx, bufs: &[(&str, &GpuBuffer, usize)]) {
    for (name, buf, n) in bufs {
        let mut v = vec![0f32; *n];
        buf.download(&ctx.stream, &mut v).unwrap();
        ctx.stream.synchronize().unwrap();
        let h = fnv1a_f32(&v);
        eprintln!("HASH {name} {h:016x}");
        super::evidence_digest::record_digest("hash_outputs", "", name, h)
            .expect("acceptance evidence");
    }
}
