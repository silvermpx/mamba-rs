//! Exercise admitted geometry and the neighboring legacy routes.

use super::{Case, check_case, encode, values};
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::gpu::Mamba3Kernels;
use std::sync::Arc;

fn suffix(dtype: WeightDtype) -> &'static str {
    match dtype {
        WeightDtype::F32 | WeightDtype::Tf32 => "",
        WeightDtype::Bf16 => "_bf16",
        WeightDtype::F16 => "_f16",
    }
}

pub(super) fn dqkv(
    stream: &Arc<CudaStream>,
    kernels: &Mamba3Kernels,
    baseline: &Arc<CudaModule>,
    dtype: WeightDtype,
) {
    let original = baseline
        .load_function(&format!("m3_dqkv{}", suffix(dtype)))
        .unwrap();
    original.set_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 99 * 1024).unwrap();
    // Include partial chunks, non-paired fallback, and both sides of the time cutoff.
    for (b, t, nh, hd, ds, cs, pairs) in [
        (8usize, 1300usize, 48usize, 16usize, 16usize, 64usize, true),
        (1, 128, 2, 16, 16, 64, true),
        (2, 129, 3, 16, 16, 64, true),
        (1, 127, 2, 16, 16, 64, true),
        (1, 129, 2, 16, 16, 64, false),
        (1, 129, 2, 8, 16, 64, true),
        (1, 129, 2, 16, 8, 64, true),
        (1, 129, 2, 16, 16, 32, true),
    ] {
        let nc = t.div_ceil(cs);
        let th = b * t * nh;
        let ch = b * nc * nh;
        let mut da = vec![0.0; ch * cs];
        let mut sums = vec![0.0; ch];
        for batch in 0..b {
            for c in 0..nc {
                for h in 0..nh {
                    let row = (batch * nc + c) * nh + h;
                    let len = cs.min(t - c * cs);
                    let step = 0.003 + ((batch + 3 * c + h) % 11) as f32 * 0.001;
                    for time in 0..len {
                        da[row * cs + time] = -((time + 1) as f32) * step;
                    }
                    sums[row] = da[row * cs + len - 1];
                }
            }
        }
        let fields = [
            values(th * ds, 0xa001),
            values(th * ds, 0xa002),
            values(th * hd, 0xa003),
            da,
            sums,
            values(th, 0xa007),
            values(ch * hd * ds, 0xa008),
            values(th * hd, 0xa004),
            values(nh, 0xa009),
            values(ch * hd * ds, 0xa00a),
        ];
        let inputs = fields
            .iter()
            .enumerate()
            .map(|(i, field)| {
                encode(
                    field,
                    if [0, 1, 2, 7].contains(&i) {
                        dtype
                    } else {
                        WeightDtype::F32
                    },
                )
            })
            .collect();
        let tiles = 2 * cs * (ds + 1) + 2 * cs * (hd + 1) + 4 * cs + 2 * hd * ds;
        let smem = (tiles
            + if pairs {
                3 * cs * (cs - 1) / 2 + 2 * cs
            } else {
                0
            })
            * 4;
        assert!(smem <= 99 * 1024);
        let cfg = LaunchConfig {
            grid_dim: (nh as u32, b as u32, nc as u32),
            block_dim: (hd as u32, 16, 1),
            shared_mem_bytes: smem as u32,
        };
        check_case(
            stream,
            Case {
                label: format!(
                    "dqkv cap={} {dtype:?} B={b} T={t} NH={nh} HD={hd} DS={ds} CS={cs} pairs={pairs}",
                    kernels.state_cap
                ),
                outputs: &["dQ", "dK", "dV", "dADT", "dQK", "dD_partials"],
                lengths: vec![th * ds, th * ds, th * hd, th, th, ch],
                inputs,
                dims: [b, t, nh, hd, ds, cs, usize::from(pairs)]
                    .map(|v| v as i32)
                    .to_vec(),
                functions: [
                    &original,
                    kernels.dqkv_for_shape(dtype, ds, hd, cs, t, pairs),
                ],
                configs: [cfg, cfg],
                initial_output: None,
            },
        );
    }
}

pub(super) fn dqktheta(
    stream: &Arc<CudaStream>,
    kernels: &Mamba3Kernels,
    baseline: &Arc<CudaModule>,
    dtype: WeightDtype,
) {
    let original = baseline
        .load_function(&format!("m3_dqktheta{}", suffix(dtype)))
        .unwrap();
    for (b, t, nh, ds, na, cs) in [
        (8usize, 1300usize, 48usize, 16usize, 4usize, 64usize),
        (1, 1, 2, 16, 4, 64),
        (2, 65, 3, 16, 4, 64),
        (1, 129, 2, 8, 4, 64),
        (1, 129, 2, 16, 4, 32),
        (1, 129, 2, 16, 0, 64),
    ] {
        let th = b * t * nh;
        let fields = [
            values(th * ds, 0xb001),
            values(th * ds, 0xb002),
            values(th, 0xb003).iter().map(|v| v + 0.85).collect(),
            values(th, 0xb004).iter().map(|v| v + 0.65).collect(),
            values(th * na, 0xb005).iter().map(|v| v * 8.0).collect(),
            values(th * ds, 0xb006),
            values(th * ds, 0xb007),
            values(th, 0xb008),
        ];
        let inputs: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, field)| encode(field, if i < 2 { dtype } else { WeightDtype::F32 }))
            .collect();
        let (selected, tiles) = kernels.dqktheta_for_shape(dtype, ds, cs, na);
        for staging in [false, true] {
            let config = |tile_count| LaunchConfig {
                grid_dim: ((b * t.div_ceil(cs)) as u32, nh as u32, 1),
                block_dim: (cs as u32, 1, 1),
                shared_mem_bytes: if staging {
                    (tile_count * cs * ds * 4) as u32
                } else {
                    0
                },
            };
            check_case(
                stream,
                Case {
                    label: format!(
                        "dqktheta cap={} {dtype:?} B={b} T={t} NH={nh} DS={ds} NA={na} CS={cs} staging={staging} tiles={tiles}",
                        kernels.state_cap
                    ),
                    outputs: &["dQ_pre", "dK_pre", "dAngles", "dScale", "dGamma"],
                    lengths: vec![th * ds, th * ds, th * na, th, th],
                    inputs: inputs.clone(),
                    dims: [b, t, nh, ds, na, cs, usize::from(staging)]
                        .map(|v| v as i32)
                        .to_vec(),
                    functions: [&original, selected],
                    configs: [config(6), config(tiles)],
                    initial_output: None,
                },
            );
        }
    }
}

pub(super) fn axis0(stream: &Arc<CudaStream>, kernels: &Mamba3Kernels, baseline: &Arc<CudaModule>) {
    let original = baseline.load_function("reduce_sum_axis0").unwrap();
    for (rows, columns) in [
        (4usize, 499200usize),
        (48, 41600),
        (1, 4096),
        (3, 4097),
        (32, 4096),
        (33, 4097),
        (64, 4096),
        (65, 4096),
        (4, 4095),
        (0, 7),
    ] {
        let lanes = rows.next_power_of_two().clamp(32, 256) as u32;
        let legacy = LaunchConfig {
            grid_dim: (columns as u32, 1, 1),
            block_dim: (lanes, 1, 1),
            shared_mem_bytes: lanes * 4,
        };
        let (selected, cfg) = kernels.axis0_reduction(rows, columns);
        for pattern in 0..5 {
            let partials: Vec<_> = (0..rows * columns)
                .map(|i| {
                    let row = i / columns;
                    let col = i % columns;
                    let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                    match pattern {
                        0 => sign * (1 + (row * 37 + col * 17) % 127) as f32 / 64.0,
                        1 => sign * (1 + (row / 2 * 19 + col * 7) % 61) as f32 / 8.0,
                        2 => {
                            if row % 2 == 0 {
                                0.0
                            } else {
                                f32::from_bits(0x8000_0000)
                            }
                        }
                        3 => sign * f32::from_bits(1 + (col as u32 & 0x3ff)),
                        _ => sign * 2.0f32.powi([-40, -20, -3, 8, 30, 60][(row + col) % 6]),
                    }
                })
                .collect();
            for accumulate in [false, true] {
                check_case(
                    stream,
                    Case {
                        label: format!(
                            "axis0 cap={} rows={rows} columns={columns} pattern={pattern} accumulate={accumulate}",
                            kernels.state_cap
                        ),
                        outputs: &["sum"],
                        lengths: vec![columns],
                        inputs: vec![encode(&partials, WeightDtype::F32)],
                        dims: vec![rows as i32, columns as i32, i32::from(accumulate)],
                        functions: [&original, selected],
                        configs: [legacy, cfg],
                        initial_output: accumulate
                            .then(|| encode(&values(columns, 0xc003), WeightDtype::F32)),
                    },
                );
            }
        }
    }
}

pub(super) fn angle(stream: &Arc<CudaStream>, kernels: &Mamba3Kernels, baseline: &Arc<CudaModule>) {
    let original = baseline.load_function("m3_angle_dt_bwd_seq").unwrap();
    for (b, t, nh, na) in [
        (8usize, 1300usize, 48usize, 4usize),
        (1, 128, 48, 4),
        (32, 128, 48, 4),
        (33, 128, 48, 4),
        (8, 127, 48, 4),
        (1, 129, 3, 2),
    ] {
        let th = b * t * nh;
        let total = nh * na;
        let inputs = vec![
            encode(&values(th * na, 0xd001), WeightDtype::F32),
            encode(&values(b * t * na, 0xd002), WeightDtype::F32),
            encode(
                &values(th, 0xd003)
                    .iter()
                    .map(|v| v + 0.55)
                    .collect::<Vec<_>>(),
                WeightDtype::F32,
            ),
        ];
        check_case(
            stream,
            Case {
                label: format!(
                    "angle cap={} B={b} T={t} NH={nh} NA={na}",
                    kernels.state_cap
                ),
                outputs: &["dAngles_partials", "dDt_partials"],
                lengths: vec![th * na, th * na],
                inputs,
                dims: [b, t, nh, na].map(|v| v as i32).to_vec(),
                functions: [&original, &kernels.m3_angle_dt_bwd_seq],
                configs: [
                    LaunchConfig {
                        grid_dim: (b as u32, total.div_ceil(256) as u32, 1),
                        block_dim: (total.min(256) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    kernels.angle_backward_cfg(b, t, nh, na),
                ],
                initial_output: None,
            },
        );
    }
}
