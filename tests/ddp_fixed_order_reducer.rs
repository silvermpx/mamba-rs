//! Single-GPU oracles for the transport-backed fixed-order reducer: the
//! `det_sum_ranks` fold kernel must reproduce the host reference fold
//! bit for bit, and the full sharded dataflow (byte-only exchange +
//! device fold + distribute) must land the identical ascending-rank sum
//! in every rank arena — immune to delivery order, consistent with the
//! emulated world's mean after the caller's `1/W` scale.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use mamba_rs::dist::EmulatedWorld;
use mamba_rs::dist::reducer::{DetReduceKernel, LoopbackWorld, reduce_sum_reference};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

fn stream() -> Arc<cudarc::driver::CudaStream> {
    GpuDevice::new(0)
        .expect("cuda device")
        .context()
        .default_stream()
}

/// Adversarial magnitudes: mixes huge and tiny values so any change in
/// association order changes bits (the same trick the fold-plan test
/// uses to prove order sensitivity).
fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let base = (s & 0xFFFF) as f32 / 65536.0 - 0.5;
            match i % 3 {
                0 => base * 1.0e8,
                1 => base,
                _ => base * 1.0e-4,
            }
        })
        .collect()
}

#[test]
fn det_sum_ranks_matches_host_fold_bitwise() {
    let stream = stream();
    let kernel = DetReduceKernel::compile(0).expect("kernel");
    for world in [2usize, 3, 5] {
        let len = 4099usize;
        let addends: Vec<Vec<f32>> = (0..world).map(|r| det(len, 0x40 + r as u32)).collect();
        let mut stacked_host = Vec::with_capacity(world * len);
        for a in &addends {
            stacked_host.extend_from_slice(a);
        }
        let stacked = GpuBuffer::from_cpu(&stream, &stacked_host).expect("stacked");
        let mut out = GpuBuffer::zeros(&stream, len).expect("out");
        unsafe { kernel.launch(out.cached_ptr(), stacked.cached_ptr(), world, len, &stream) }
            .expect("launch");
        stream.synchronize().expect("sync");
        let got = out.to_cpu(&stream).expect("dtoh");
        let refs: Vec<&[f32]> = addends.iter().map(|a| a.as_slice()).collect();
        let mut want = vec![0.0f32; len];
        reduce_sum_reference(&refs, &mut want);
        let got_bits: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
        let want_bits: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got_bits, want_bits, "W={world}: device fold diverged");
        let _ = &mut out;
    }
}

#[test]
fn loopback_dataflow_lands_reference_bits_in_every_arena() {
    let stream = stream();
    let kernel = DetReduceKernel::compile(0).expect("kernel");
    // Prime length: uneven shards, remainder spread over low ranks.
    let n = 10_007usize;
    for world in [2usize, 3, 4] {
        let addends: Vec<Vec<f32>> = (0..world).map(|r| det(n, 0x900 + r as u32)).collect();
        let refs: Vec<&[f32]> = addends.iter().map(|a| a.as_slice()).collect();
        let mut want = vec![0.0f32; n];
        reduce_sum_reference(&refs, &mut want);
        let want_bits: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();

        let arenas: Vec<GpuBuffer> = addends
            .iter()
            .map(|a| GpuBuffer::from_cpu(&stream, a).expect("arena"))
            .collect();
        let mut world_h = LoopbackWorld::new(arenas);
        world_h.run_round(&kernel, &stream).expect("round");
        for (r, a) in world_h.arenas().iter().enumerate() {
            let got = a.to_cpu(&stream).expect("dtoh");
            let got_bits: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got_bits, want_bits, "W={world}: rank {r} arena diverged");
        }
    }
}

#[test]
fn loopback_dataflow_is_delivery_order_immune() {
    let stream = stream();
    let kernel = DetReduceKernel::compile(0).expect("kernel");
    let n = 4097usize;
    let world = 3usize;
    let addends: Vec<Vec<f32>> = (0..world).map(|r| det(n, 0xA00 + r as u32)).collect();

    let run = |reverse: bool| -> Vec<u32> {
        let arenas: Vec<GpuBuffer> = addends
            .iter()
            .map(|a| GpuBuffer::from_cpu(&stream, a).expect("arena"))
            .collect();
        let mut w = LoopbackWorld::new(arenas);
        w.reverse_delivery = reverse;
        w.run_round(&kernel, &stream).expect("round");
        w.arenas()[0]
            .to_cpu(&stream)
            .expect("dtoh")
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    assert_eq!(
        run(false),
        run(true),
        "delivery order leaked into the reduced bits"
    );
}

#[test]
fn loopback_sum_scaled_matches_emulated_mean() {
    // The trainer applies mean as sum-then-multiply; the emulated world
    // folds ascending then multiplies by the same 1/W. Identical float
    // op sequences must give identical bits.
    let stream = stream();
    let kernel = DetReduceKernel::compile(0).expect("kernel");
    let n = 2053usize;
    let world = 4usize;
    let addends: Vec<Vec<f32>> = (0..world).map(|r| det(n, 0xB00 + r as u32)).collect();

    let arenas: Vec<GpuBuffer> = addends
        .iter()
        .map(|a| GpuBuffer::from_cpu(&stream, a).expect("arena"))
        .collect();
    let mut w = LoopbackWorld::new(arenas);
    w.run_round(&kernel, &stream).expect("round");
    let inv_w = 1.0f32 / world as f32;
    let scaled: Vec<u32> = w.arenas()[0]
        .to_cpu(&stream)
        .expect("dtoh")
        .iter()
        .map(|x| (x * inv_w).to_bits())
        .collect();

    let ew = EmulatedWorld::new(world).expect("emulated");
    let mut host = addends.clone();
    ew.all_reduce_mean(&mut host, None).expect("emulated mean");
    let want: Vec<u32> = host[0].iter().map(|x| x.to_bits()).collect();
    assert_eq!(scaled, want, "device sum x 1/W diverged from emulated mean");
}

#[test]
fn loopback_handles_empty_shards_and_zero_length() {
    let stream = stream();
    let kernel = DetReduceKernel::compile(0).expect("kernel");
    // n < world: the high ranks own EMPTY shards — the dataflow must
    // still land the reference bits everywhere without touching the
    // empty owners.
    let n = 3usize;
    let world = 5usize;
    let addends: Vec<Vec<f32>> = (0..world).map(|r| det(n, 0xC00 + r as u32)).collect();
    let refs: Vec<&[f32]> = addends.iter().map(|a| a.as_slice()).collect();
    let mut want = vec![0.0f32; n];
    reduce_sum_reference(&refs, &mut want);
    let want_bits: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
    let arenas: Vec<GpuBuffer> = addends
        .iter()
        .map(|a| GpuBuffer::from_cpu(&stream, a).expect("arena"))
        .collect();
    let mut w = LoopbackWorld::new(arenas);
    w.run_round(&kernel, &stream).expect("round");
    for (r, a) in w.arenas().iter().enumerate() {
        let got: Vec<u32> = a
            .to_cpu(&stream)
            .expect("dtoh")
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(got, want_bits, "rank {r} with empty shards diverged");
    }

    // len == 0 short-circuits before any pointer is touched — null
    // pointers are fine BECAUSE the guard returns first (that is the
    // property under test).
    unsafe { kernel.launch(0, 0, 3, 0, &stream) }.expect("zero-length launch");
}
