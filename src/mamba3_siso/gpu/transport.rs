//! Qualified Mamba-3 backward transport routes.

use super::kernels::Mamba3Kernels;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::kernels::TypedKernel;
use crate::gpu::launch::IndexWidth;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

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
        let axis0 = if admission.cap16() {
            Some(get("m3_reduce_sum_axis0_packed")?)
        } else {
            None
        };
        Ok(Self {
            admission,
            dqkv,
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
    let lanes = axis0_lanes(rows);
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

/// The shape a DQKV launch selects its function by: the state and head
/// widths, the chunk, the window and whether the pair matrices are staged.
#[derive(Clone, Copy, Debug)]
pub struct DqkvRoute {
    pub state: usize,
    pub head: usize,
    pub chunk: usize,
    pub time: usize,
    pub pairs: bool,
}

/// The block width of the plain axis-0 reduction for `rows` partial rows:
/// a lane per row up to 256, never under a warp.
fn axis0_lanes(rows: usize) -> u32 {
    rows.next_power_of_two().clamp(32, 256) as u32
}

/// One axis-0 reduction: `rows` partial rows of `columns` floats at
/// `partials`, summed per column into `out`, or added to it.
pub struct Axis0Sum {
    pub out: CUdeviceptr,
    pub partials: CUdeviceptr,
    pub rows: usize,
    pub columns: usize,
    pub accumulate: bool,
}

impl Mamba3Kernels {
    /// Select the DQKV function for `route` at the launch's index `width`.
    /// The returned handle must use the same shape, pair-matrix decision, and
    /// launch geometry; ineligible shapes use the legacy typed function for
    /// `dtype`. The transport is built at 32 bits only, so a launch past that
    /// range takes the legacy function's 64-bit build.
    pub fn dqkv_for_shape(
        &self,
        dtype: WeightDtype,
        route: DqkvRoute,
        width: IndexWidth,
    ) -> &CudaFunction {
        let DqkvRoute {
            state,
            head,
            chunk,
            time,
            pairs,
        } = route;
        if width == IndexWidth::I32
            && dqkv_transport_admitted(self.transport.admission, state, head, chunk, time, pairs)
        {
            match &self.transport.dqkv {
                Some(kernels) => kernels.get(dtype),
                None => unreachable!("admitted DQKV transport handles were not loaded"),
            }
        } else {
            self.m3_dqkv_typed.get(width).get(dtype)
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

    /// Launch the axis-0 reduction `sum`: on the 32-bit route the selection
    /// of [`Self::axis0_reduction`] in one launch, on the 64-bit route the
    /// wide reduction at the plain reduction's block width in launches of at
    /// most the grid's x extent of columns. Every route sums a column in the
    /// same order, so all of them write the same bits.
    pub fn launch_axis0_sum(
        &self,
        stream: &CudaStream,
        sum: Axis0Sum,
        width: IndexWidth,
    ) -> Result<(), String> {
        let accumulate = i32::from(sum.accumulate);
        let rows = i32::try_from(sum.rows)
            .map_err(|_| format!("axis-0 reduction of {} rows", sum.rows))?;
        if width == IndexWidth::I32 {
            let columns = i32::try_from(sum.columns).map_err(|_| {
                format!(
                    "{} columns reached the 32-bit axis-0 reduction",
                    sum.columns
                )
            })?;
            let (reduction, cfg) = self.axis0_reduction(sum.rows, sum.columns);
            let mut builder = stream.launch_builder(reduction);
            builder.arg(&sum.out);
            builder.arg(&sum.partials);
            builder.arg(&rows);
            builder.arg(&columns);
            builder.arg(&accumulate);
            return unsafe { builder.launch(cfg) }
                .map(|_| ())
                .map_err(|e| format!("axis-0 reduction: {e:?}"));
        }
        assert!(sum.columns > 0, "axis-0 reduction columns must be positive");
        let lanes = axis0_lanes(sum.rows);
        let dim = i64::try_from(sum.columns)
            .map_err(|_| format!("axis-0 reduction of {} columns", sum.columns))?;
        let mut col0 = 0usize;
        while col0 < sum.columns {
            let piece = (sum.columns - col0).min(i32::MAX as usize);
            let first = col0 as i64;
            let mut builder = stream.launch_builder(&self.reduce_sum_axis0_wide);
            builder.arg(&sum.out);
            builder.arg(&sum.partials);
            builder.arg(&rows);
            builder.arg(&dim);
            builder.arg(&first);
            builder.arg(&accumulate);
            let cfg = LaunchConfig {
                grid_dim: (piece as u32, 1, 1),
                block_dim: (lanes, 1, 1),
                shared_mem_bytes: lanes * 4,
            };
            unsafe { builder.launch(cfg) }.map_err(|e| format!("wide axis-0 reduction: {e:?}"))?;
            col0 += piece;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
