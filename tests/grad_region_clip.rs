//! Region clip unit: the two-block per-layer region fold must (1) match
//! a CPU L2 norm over exactly the described elements, (2) scale ONLY
//! those elements by the PyTorch clip coefficient, and (3) leave every
//! other arena element bit-untouched. Geometry mirrors the Mamba-3
//! control region shape in miniature (a column stripe of a row-major
//! matrix plus a contiguous bias vector, repeating per layer).
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::grad_clip::{GradRegionGeom, alloc_partials, clip_region_device};

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

#[test]
fn region_clip_scales_only_the_region() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();

    // Miniature m3-like layout: 3 layers, each layer holding a 7x11
    // row-major matrix (stripe = columns 6..11) and a 5-long bias,
    // padded fore and aft so complement coverage is meaningful.
    let (n_layers, rows, row_stride, ctrl0, ctrl_len, b_len) =
        (3usize, 7usize, 11usize, 6usize, 5usize, 5usize);
    let head_pad = 13usize;
    let mat = rows * row_stride;
    let per_layer = mat + b_len + 3; // 3 trailing pad elems per layer
    let total = head_pad + n_layers * per_layer;

    let host = det(total, 0xC1AB_5EED, 4.0);
    let mut flat = GpuBuffer::zeros(&ctx.stream, total).unwrap();
    flat.upload(&ctx.stream, &host).unwrap();
    let mut partials = alloc_partials(&ctx.stream).unwrap();
    let mut scratch = GpuBuffer::zeros(&ctx.stream, 2).unwrap();

    let geom = GradRegionGeom {
        n_layers: n_layers as i32,
        layer_stride: per_layer as i32,
        a_off: (head_pad + ctrl0) as i32,
        a_rows: rows as i32,
        a_cols: ctrl_len as i32,
        a_row_stride: row_stride as i32,
        b_off: (head_pad + mat) as i32,
        b_len: b_len as i32,
    };

    // CPU reference: the exact region element set.
    let mut in_region = vec![false; total];
    for l in 0..n_layers {
        for r in 0..rows {
            for c in 0..ctrl_len {
                in_region[head_pad + l * per_layer + r * row_stride + ctrl0 + c] = true;
            }
        }
        for b in 0..b_len {
            in_region[head_pad + l * per_layer + mat + b] = true;
        }
    }
    let cpu_norm = in_region
        .iter()
        .zip(&host)
        .filter(|(m, _)| **m)
        .map(|(_, v)| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();

    let max_norm = (cpu_norm / 3.0) as f32; // force the gate to fire
    let norm =
        clip_region_device(&ctx, &mut flat, &mut partials, &mut scratch, geom, max_norm).unwrap();
    assert!(
        (f64::from(norm) - cpu_norm).abs() / cpu_norm < 1e-6,
        "region norm {norm} vs cpu {cpu_norm}"
    );

    let coef = f64::from(max_norm) / (cpu_norm + 1e-6);
    let mut out = vec![0.0f32; total];
    flat.download(&ctx.stream, &mut out).unwrap();
    for i in 0..total {
        if in_region[i] {
            let want = (f64::from(host[i]) * coef) as f32;
            assert!(
                (out[i] - want).abs() <= want.abs() * 1e-5 + 1e-7,
                "region elem {i}: {} vs {}",
                out[i],
                want
            );
        } else {
            assert!(
                out[i].to_bits() == host[i].to_bits(),
                "complement elem {i} was touched: {} vs {}",
                out[i],
                host[i]
            );
        }
    }

    // No-fire contract: a bound above the norm leaves the region
    // bit-identical (coefficient exactly 1.0).
    let mut flat2 = GpuBuffer::zeros(&ctx.stream, total).unwrap();
    flat2.upload(&ctx.stream, &host).unwrap();
    let n2 = clip_region_device(
        &ctx,
        &mut flat2,
        &mut partials,
        &mut scratch,
        geom,
        (cpu_norm * 2.0) as f32,
    )
    .unwrap();
    assert!((f64::from(n2) - cpu_norm).abs() / cpu_norm < 1e-6);
    let mut out2 = vec![0.0f32; total];
    flat2.download(&ctx.stream, &mut out2).unwrap();
    for i in 0..total {
        assert!(
            out2[i].to_bits() == host[i].to_bits(),
            "no-fire touched {i}"
        );
    }
}
