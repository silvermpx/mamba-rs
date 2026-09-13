//! Qualified Mamba-3 backward transport routes.

use super::kernels::Mamba3Kernels;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernels::{HalfKernel, TypedKernel};
use cudarc::driver::{CudaFunction, LaunchConfig};

#[derive(Clone, Copy)]
pub(super) struct TransportAdmission {
    environment: bool,
    state_cap: usize,
}

impl TransportAdmission {
    pub(super) const fn new(environment: bool, state_cap: usize) -> Self {
        Self {
            environment,
            state_cap,
        }
    }

    const fn dqkv(self) -> bool {
        self.environment && matches!(self.state_cap, 16 | 32 | 64)
    }

    const fn cap16(self) -> bool {
        self.environment && self.state_cap == 16
    }
}

pub(super) struct TransportKernels {
    admission: TransportAdmission,
    pub(super) dqkv: Option<TypedKernel>,
    dqktheta: Option<HalfKernel>,
    axis0: Option<CudaFunction>,
}

impl TransportKernels {
    pub(super) fn load(
        admission: TransportAdmission,
        get: impl Fn(&str) -> Result<CudaFunction, String>,
    ) -> Result<Self, String> {
        let dqkv = if admission.dqkv() {
            Some(TypedKernel {
                f32: get("m3_dqkv_transport_f32")?,
                bf16: get("m3_dqkv_transport_bf16")?,
                f16: get("m3_dqkv_transport_f16")?,
            })
        } else {
            None
        };
        let dqktheta = if admission.cap16() {
            Some(HalfKernel {
                bf16: get("m3_dqktheta_transport_bf16")?,
                f16: get("m3_dqktheta_transport_f16")?,
            })
        } else {
            None
        };
        let axis0 = if admission.cap16() {
            Some(get("m3_reduce_sum_axis0_packed")?)
        } else {
            None
        };
        Ok(Self {
            admission,
            dqkv,
            dqktheta,
            axis0,
        })
    }
}

pub(super) fn transport_environment_admitted(
    actual_device: (i32, i32),
    arch: &str,
    nvrtc: (i32, i32),
) -> bool {
    actual_device == (8, 9) && arch == "sm_89" && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
}

const fn dqkv_transport_admitted(
    admission: TransportAdmission,
    state: usize,
    head: usize,
    chunk: usize,
    time: usize,
    pairs: bool,
) -> bool {
    admission.dqkv() && state == 16 && head == 16 && chunk == 64 && time >= 128 && pairs
}

const fn dqktheta_transport_admitted(
    admission: TransportAdmission,
    dtype: WeightDtype,
    state: usize,
    chunk: usize,
    angles: usize,
) -> bool {
    admission.cap16()
        && !matches!(dtype, WeightDtype::F32)
        && state == 16
        && chunk == 64
        && angles == 4
}

#[derive(Clone, Copy)]
struct Axis0LaunchPolicy {
    packed: bool,
    config: LaunchConfig,
}

fn axis0_launch_policy(
    admission: TransportAdmission,
    rows: usize,
    columns: usize,
) -> Axis0LaunchPolicy {
    assert!(columns > 0, "axis-0 reduction columns must be positive");
    let lanes = rows.next_power_of_two().clamp(32, 256) as u32;
    if admission.cap16() && (1..=64).contains(&rows) && columns >= 4096 {
        let packed_columns = 256 / lanes;
        Axis0LaunchPolicy {
            packed: true,
            config: LaunchConfig {
                grid_dim: ((columns as u32).div_ceil(packed_columns), 1, 1),
                block_dim: (packed_columns, lanes, 1),
                shared_mem_bytes: 1024,
            },
        }
    } else {
        Axis0LaunchPolicy {
            packed: false,
            config: LaunchConfig {
                grid_dim: (columns as u32, 1, 1),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: lanes * 4,
            },
        }
    }
}

fn angle_backward_launch_policy(
    admission: TransportAdmission,
    batch: usize,
    time: usize,
    heads: usize,
    angles: usize,
) -> LaunchConfig {
    assert!(
        batch > 0 && time > 0 && heads > 0 && angles > 0,
        "angle backward dimensions must be positive"
    );
    let total = heads * angles;
    if admission.cap16() && batch <= 32 && time >= 128 && heads == 48 && angles == 4 {
        LaunchConfig {
            grid_dim: (batch as u32, (total as u32).div_ceil(32), 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (batch as u32, (total as u32).div_ceil(256), 1),
            block_dim: (total.min(256) as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl Mamba3Kernels {
    /// Select the DQKV function for `state`, `head`, `chunk`, `time`, and
    /// `pairs`. The returned handle must use the same shape, pair-matrix
    /// decision, and launch geometry; ineligible shapes use the legacy typed
    /// function for `dtype`.
    pub fn dqkv_for_shape(
        &self,
        dtype: WeightDtype,
        state: usize,
        head: usize,
        chunk: usize,
        time: usize,
        pairs: bool,
    ) -> &CudaFunction {
        if dqkv_transport_admitted(self.transport.admission, state, head, chunk, time, pairs) {
            match &self.transport.dqkv {
                Some(kernels) => kernels.get(dtype),
                None => unreachable!("admitted DQKV transport handles were not loaded"),
            }
        } else {
            self.m3_dqkv_typed.get(dtype)
        }
    }

    /// Select the DQK-theta function for `state`, `chunk`, and `angles`, and
    /// return its staging-tile count. The function and tile count are one
    /// launch contract: callers derive shared memory and staging from the
    /// count. Ineligible shapes and F32 use the legacy function with six tiles.
    pub fn dqktheta_for_shape(
        &self,
        dtype: WeightDtype,
        state: usize,
        chunk: usize,
        angles: usize,
    ) -> (&CudaFunction, usize) {
        if dqktheta_transport_admitted(self.transport.admission, dtype, state, chunk, angles) {
            match &self.transport.dqktheta {
                Some(kernels) => (kernels.get(dtype), 4),
                None => unreachable!("admitted DQK-theta transport handles were not loaded"),
            }
        } else {
            (self.m3_dqktheta_typed.get(dtype), 6)
        }
    }

    /// Select the axis-0 reduction for a `rows` by `columns` input.
    /// The function and configuration must be launched together. Ineligible
    /// dimensions use the legacy reduction configuration; `columns` must be
    /// positive.
    pub fn axis0_reduction(&self, rows: usize, columns: usize) -> (&CudaFunction, LaunchConfig) {
        let policy = axis0_launch_policy(self.transport.admission, rows, columns);
        if policy.packed {
            match &self.transport.axis0 {
                Some(function) => (function, policy.config),
                None => unreachable!("admitted packed axis-0 handle was not loaded"),
            }
        } else {
            (&self.reduce_sum_axis0, policy.config)
        }
    }

    /// Return the angle-backward launch configuration for `batch`, `time`,
    /// `heads`, and `angles`. Ineligible dimensions retain the legacy launch
    /// geometry, and all dimensions must be positive.
    pub fn angle_backward_cfg(
        &self,
        batch: usize,
        time: usize,
        heads: usize,
        angles: usize,
    ) -> LaunchConfig {
        angle_backward_launch_policy(self.transport.admission, batch, time, heads, angles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::dtype::WeightDtype;

    #[test]
    fn compiler_and_device_admission_is_exact() {
        assert!(transport_environment_admitted((8, 9), "sm_89", (12, 8)));
        assert!(transport_environment_admitted((8, 9), "sm_89", (13, 0)));
        assert!(transport_environment_admitted((8, 9), "sm_89", (13, 2)));

        assert!(!transport_environment_admitted((9, 0), "sm_89", (12, 8)));
        assert!(!transport_environment_admitted((12, 0), "sm_89", (12, 8)));
        assert!(!transport_environment_admitted(
            (8, 9),
            "compute_89",
            (12, 8)
        ));
        assert!(!transport_environment_admitted((8, 9), "sm_80", (12, 8)));
        assert!(!transport_environment_admitted((8, 9), "sm_89", (13, 1)));
        assert!(!transport_environment_admitted((8, 9), "sm_89", (14, 0)));
    }

    #[test]
    fn dqkv_admission_requires_supported_cap_and_production_shape() {
        for cap in [16, 32, 64] {
            assert!(dqkv_transport_admitted(
                TransportAdmission::new(true, cap),
                16,
                16,
                64,
                128,
                true,
            ));
        }
        assert!(!dqkv_transport_admitted(
            TransportAdmission::new(true, 8),
            16,
            16,
            64,
            128,
            true,
        ));
        assert!(!dqkv_transport_admitted(
            TransportAdmission::new(true, 128),
            16,
            16,
            64,
            128,
            true,
        ));
        assert!(!dqkv_transport_admitted(
            TransportAdmission::new(false, 16),
            16,
            16,
            64,
            128,
            true,
        ));

        for (state, head, chunk, time, pairs) in [
            (8, 16, 64, 128, true),
            (16, 8, 64, 128, true),
            (16, 16, 32, 128, true),
            (16, 16, 64, 127, true),
            (16, 16, 64, 128, false),
        ] {
            assert!(!dqkv_transport_admitted(
                TransportAdmission::new(true, 16),
                state,
                head,
                chunk,
                time,
                pairs,
            ));
        }
    }

    #[test]
    fn dqktheta_admission_requires_cap16_half_dtype_and_production_shape() {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            assert!(dqktheta_transport_admitted(
                TransportAdmission::new(true, 16),
                dtype,
                16,
                64,
                4,
            ));
        }
        assert!(!dqktheta_transport_admitted(
            TransportAdmission::new(true, 16),
            WeightDtype::F32,
            16,
            64,
            4,
        ));
        assert!(!dqktheta_transport_admitted(
            TransportAdmission::new(true, 32),
            WeightDtype::Bf16,
            16,
            64,
            4,
        ));
        for (state, chunk, angles) in [(8, 64, 4), (16, 32, 4), (16, 64, 0)] {
            assert!(!dqktheta_transport_admitted(
                TransportAdmission::new(true, 16),
                WeightDtype::Bf16,
                state,
                chunk,
                angles,
            ));
        }
    }

    #[test]
    fn axis0_admission_and_geometry_preserve_boundaries() {
        let legacy = axis0_launch_policy(TransportAdmission::new(true, 16), 4, 4095);
        assert!(!legacy.packed);
        assert_eq!(legacy.config.grid_dim, (4095, 1, 1));
        assert_eq!(legacy.config.block_dim, (32, 1, 1));
        assert_eq!(legacy.config.shared_mem_bytes, 128);

        let packed = axis0_launch_policy(TransportAdmission::new(true, 16), 4, 4096);
        assert!(packed.packed);
        assert_eq!(packed.config.grid_dim, (512, 1, 1));
        assert_eq!(packed.config.block_dim, (8, 32, 1));
        assert_eq!(packed.config.shared_mem_bytes, 1024);

        let rows64 = axis0_launch_policy(TransportAdmission::new(true, 16), 64, 4096);
        assert!(rows64.packed);
        assert_eq!(rows64.config.grid_dim, (1024, 1, 1));
        assert_eq!(rows64.config.block_dim, (4, 64, 1));
        assert_eq!(rows64.config.shared_mem_bytes, 1024);

        let rows65 = axis0_launch_policy(TransportAdmission::new(true, 16), 65, 4096);
        assert!(!rows65.packed);
        assert_eq!(rows65.config.grid_dim, (4096, 1, 1));
        assert_eq!(rows65.config.block_dim, (128, 1, 1));
        assert_eq!(rows65.config.shared_mem_bytes, 512);

        let cap32 = axis0_launch_policy(TransportAdmission::new(true, 32), 4, 4096);
        assert!(!cap32.packed);
        let zero_rows = axis0_launch_policy(TransportAdmission::new(true, 16), 0, 4096);
        assert!(!zero_rows.packed);
        assert_eq!(zero_rows.config.block_dim, (32, 1, 1));
    }

    #[test]
    #[should_panic(expected = "axis-0 reduction columns must be positive")]
    fn axis0_rejects_zero_columns_before_making_a_cuda_grid() {
        let _ = axis0_launch_policy(TransportAdmission::new(true, 16), 4, 0);
    }

    #[test]
    fn angle_backward_admission_and_geometry_preserve_boundaries() {
        let production =
            angle_backward_launch_policy(TransportAdmission::new(true, 16), 32, 128, 48, 4);
        assert_eq!(production.grid_dim, (32, 6, 1));
        assert_eq!(production.block_dim, (32, 1, 1));
        assert_eq!(production.shared_mem_bytes, 0);

        for (cap, batch, time, heads, angles, block, grid_y) in [
            (16, 33, 128, 48, 4, 192, 1),
            (16, 32, 127, 48, 4, 192, 1),
            (16, 32, 128, 47, 4, 188, 1),
            (16, 32, 128, 48, 3, 144, 1),
            (32, 32, 128, 48, 4, 192, 1),
        ] {
            let config = angle_backward_launch_policy(
                TransportAdmission::new(true, cap),
                batch,
                time,
                heads,
                angles,
            );
            assert_eq!(config.grid_dim, (batch as u32, grid_y, 1));
            assert_eq!(config.block_dim, (block, 1, 1));
            assert_eq!(config.shared_mem_bytes, 0);
        }
    }

    #[test]
    #[should_panic(expected = "angle backward dimensions must be positive")]
    fn angle_backward_rejects_zero_dimensions() {
        let _ = angle_backward_launch_policy(TransportAdmission::new(true, 16), 0, 128, 48, 4);
    }
}
