//! A lone row through the CPU linear layer, `y = x W + b`, at the shapes a
//! decode step runs: the row-major product on one thread against the
//! packed one across the rayon pool, with the bits of both compared.
//!
//! Run: `cargo bench --bench cpu_gemv_bench`

use std::time::Instant;

use mamba_rs::ops::blas::sgemm_forward;
use mamba_rs::ops::packed::PackedLinear;

/// (K, N, what it is).
const SHAPES: [(usize, usize, &str); 6] = [
    (512, 2048, "d512 in_proj"),
    (768, 3072, "mamba-130m in_proj"),
    (1536, 768, "mamba-130m out_proj"),
    (2048, 8192, "mamba-1.4b in_proj"),
    (4096, 2048, "mamba-1.4b out_proj"),
    (2560, 10240, "mamba-2.8b in_proj"),
];

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

/// The median of `iters` timed calls after a few untimed ones, in
/// microseconds.
fn median_us(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut t: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn main() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!(
        "lone-row linear, row-major on one thread against packed on {} pool threads (us; GB/s of W)",
        rayon::current_num_threads()
    );
    for (k, n, what) in SHAPES {
        let (x, w, b) = (det(k, 1), det(k * n, 2), det(n, 3));
        let (mut serial, mut pooled) = (vec![0f32; n], vec![0f32; n]);
        let iters = if k * n > 1 << 24 { 20 } else { 100 };
        let t1 = one.install(|| {
            median_us(iters, || {
                sgemm_forward(&mut serial, &x, &w, Some(&b), 1, k, n);
            })
        });
        let packed = PackedLinear::pack(&w, k, n);
        let tp = median_us(iters, || {
            packed.forward(&mut pooled, &x, Some(&b), 1);
        });
        assert_eq!(
            serial, pooled,
            "{what}: the packed product changed the bits"
        );
        let gb = (k * n * 4) as f64 / 1e3;
        println!(
            "  {what:<22} {k:>5}x{n:<6} row-major {t1:>8.1} ({:>5.0})  packed {tp:>8.1} ({:>5.0})  x{:.2}",
            gb / t1,
            gb / tp,
            t1 / tp
        );
    }
}
