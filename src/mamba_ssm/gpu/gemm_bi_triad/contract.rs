use super::super::blas::TypedPtr;

pub(super) type CUptr = cudarc::driver::sys::CUdeviceptr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct GemmDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub m_i32: i32,
    pub k_i32: i32,
    pub n_i32: i32,
    pub mk: usize,
    pub mn: usize,
    pub kn: usize,
    pub(super) m_u32: u32,
    pub(super) k_u32: u32,
    pub(super) n_u32: u32,
    pub(super) mk_u32: u32,
    pub(super) mn_u32: u32,
    pub(super) kn_u32: u32,
}

impl GemmDims {
    pub(super) fn checked(
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<Self, String> {
        Self::checked_storage(m, k, n, [lda, ldb, ldc], [k, n, n], [m, k, m])
    }

    pub(in crate::mamba_ssm::gpu) fn nn(
        dims: (usize, usize, usize),
        lda: usize,
    ) -> Result<Self, String> {
        Self::checked(dims.0, dims.1, dims.2, lda, dims.2, dims.2)
    }

    pub(in crate::mamba_ssm::gpu) fn tn(dims: (usize, usize, usize)) -> Result<Self, String> {
        Self::checked_storage(
            dims.0,
            dims.1,
            dims.2,
            [dims.1, dims.2, dims.2],
            [dims.1, dims.2, dims.2],
            [dims.0, dims.0, dims.1],
        )
    }

    pub(in crate::mamba_ssm::gpu) fn nt(dims: (usize, usize, usize)) -> Result<Self, String> {
        Self::checked_storage(
            dims.0,
            dims.1,
            dims.2,
            [dims.2, dims.2, dims.1],
            [dims.2, dims.2, dims.1],
            [dims.0, dims.1, dims.0],
        )
    }

    fn checked_storage(
        m: usize,
        k: usize,
        n: usize,
        strides: [usize; 3],
        widths: [usize; 3],
        row_counts: [usize; 3],
    ) -> Result<Self, String> {
        let product = |lhs: usize, rhs: usize, name: &str| {
            lhs.checked_mul(rhs).ok_or_else(|| {
                invalid_gemm_dimensions(format!("{name} overflows usize ({lhs} * {rhs})"))
            })
        };
        let mk = product(m, k, "M*K")?;
        let mn = product(m, n, "M*N")?;
        let kn = product(k, n, "K*N")?;

        if m == 0 || k == 0 || n == 0 {
            return Err(invalid_gemm_dimensions(format!(
                "axes must be positive, got M={m} K={k} N={n}"
            )));
        }

        let axis_i32 = |value: usize, name: &str| {
            i32::try_from(value)
                .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds i32::MAX")))
        };
        let m_i32 = axis_i32(m, "M")?;
        let k_i32 = axis_i32(k, "K")?;
        let n_i32 = axis_i32(n, "N")?;
        for (value, name) in [(mk, "M*K"), (mn, "M*N"), (kn, "K*N")] {
            axis_i32(value, name)?;
        }

        let [lda, ldb, ldc] = strides;
        let [lda_min, ldb_min, ldc_min] = widths;
        let [a_rows, b_rows, c_rows] = row_counts;
        for (value, minimum, name) in [
            (lda, lda_min, "lda"),
            (ldb, ldb_min, "ldb"),
            (ldc, ldc_min, "ldc"),
        ] {
            if value == 0 || value < minimum {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={value} is smaller than the physical width {minimum}"
                )));
            }
        }
        let lda = axis_i32(lda, "lda")?;
        let ldb = axis_i32(ldb, "ldb")?;
        let ldc = axis_i32(ldc, "ldc")?;

        for (rows, stride, width, name) in [
            (a_rows, strides[0], widths[0], "A storage"),
            (b_rows, strides[1], widths[1], "B storage"),
            (c_rows, strides[2], widths[2], "C storage"),
        ] {
            let span = rows
                .checked_sub(1)
                .and_then(|last_row| last_row.checked_mul(stride))
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| invalid_gemm_dimensions(format!("{name} span overflows usize")))?;
            axis_i32(span, name)?;
        }

        let to_u32 = |value: usize, name: &str| {
            u32::try_from(value)
                .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds u32::MAX")))
        };
        Ok(Self {
            m,
            k,
            n,
            lda,
            ldb,
            ldc,
            m_i32,
            k_i32,
            n_i32,
            mk,
            mn,
            kn,
            m_u32: to_u32(m, "M")?,
            k_u32: to_u32(k, "K")?,
            n_u32: to_u32(n, "N")?,
            mk_u32: to_u32(mk, "M*K")?,
            mn_u32: to_u32(mn, "M*N")?,
            kn_u32: to_u32(kn, "K*N")?,
        })
    }

    pub(super) fn tuple(self) -> (usize, usize, usize) {
        (self.m, self.k, self.n)
    }
}

pub(super) fn invalid_gemm_dimensions(reason: impl std::fmt::Display) -> String {
    format!("invalid GEMM dimensions: {reason}")
}

pub(super) fn checked_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds u32::MAX")))
}

pub(super) fn checked_i32(value: usize, name: &str) -> Result<i32, String> {
    i32::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds i32::MAX")))
}

pub(super) fn checked_usize(value: u32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds usize::MAX")))
}

pub(super) fn checked_tile_grid(
    rows: u32,
    row_tile: u32,
    cols: u32,
    col_tile: u32,
) -> Result<u32, String> {
    rows.div_ceil(row_tile)
        .checked_mul(cols.div_ceil(col_tile))
        .ok_or_else(|| invalid_gemm_dimensions("tile grid overflows u32"))
}

pub(super) fn checked_grid_product(lhs: u32, rhs: u32, depth: u32) -> Result<u32, String> {
    lhs.checked_mul(rhs)
        .and_then(|value| value.checked_mul(depth))
        .ok_or_else(|| invalid_gemm_dimensions("launch grid overflows u32"))
}

pub(super) fn checked_u32_product(lhs: u32, rhs: u32, name: &str) -> Result<u32, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} overflows u32")))
}

pub(super) fn checked_mul3(
    lhs: usize,
    middle: usize,
    rhs: usize,
    name: &str,
) -> Result<usize, String> {
    lhs.checked_mul(middle)
        .and_then(|value| value.checked_mul(rhs))
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} overflows usize")))
}

pub(super) fn checked_byte_offset(
    elements: usize,
    element_bytes: usize,
    name: &str,
) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} byte offset overflows usize")))?;
    u64::try_from(bytes)
        .map_err(|_| invalid_gemm_dimensions(format!("{name} byte offset exceeds u64::MAX")))
}

pub(super) fn checked_ptr_add(base: u64, offset: u64, name: &str) -> Result<u64, String> {
    base.checked_add(offset)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} pointer offset overflows u64")))
}

pub(super) fn validate_bias_preseed(
    alpha: f32,
    bias_ptr: CUptr,
    route: &str,
) -> Result<(), String> {
    if bias_ptr != 0 && alpha != 1.0 {
        return Err(format!(
            "{route}: bias pre-seeding requires alpha == 1.0, got {alpha}"
        ));
    }
    Ok(())
}

/// Operand bundle for the TC NN forward (`Y = X @ W + bias`).
pub struct TcFwdOperands {
    pub y: TypedPtr,
    pub x: TypedPtr,
    pub w: TypedPtr,
    /// f32 bias pointer, 0 = none.
    pub bias_ptr: CUptr,
}

/// Strided scalar-forward operands shared by every deterministic NN bucket.
#[derive(Clone, Copy)]
pub struct SgemmFwdSubOperands {
    pub x_ptr: CUptr,
    pub lda: usize,
    pub w_ptr: CUptr,
    /// f32 bias pointer, 0 = none.
    pub bias_ptr: CUptr,
}

#[cfg(test)]
mod tests {
    use super::{GemmDims, checked_grid_product, checked_u32, validate_bias_preseed};

    fn assert_invalid_error(error: String) {
        assert!(error.starts_with("invalid GEMM dimensions"), "{error}");
        assert!(!error.starts_with("UNCOVERED"), "{error}");
    }

    fn assert_invalid(result: Result<GemmDims, String>) {
        assert_invalid_error(result.expect_err("dimensions must be rejected"));
    }

    #[test]
    fn gemm_dims_reject_zero_axes() {
        for dims in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert_invalid(GemmDims::nn(dims, dims.1.max(1)));
        }
    }

    #[test]
    fn gemm_dims_accept_i32_max_boundary() {
        let limit = i32::MAX as usize;
        let dims = GemmDims::checked(limit, 1, 1, 1, 1, 1).unwrap();

        assert_eq!(dims.m_i32, i32::MAX);
        assert_eq!(dims.mk, limit);
        assert_eq!(dims.mn, limit);
        assert_eq!(dims.kn, 1);
    }

    #[test]
    fn gemm_dims_reject_axis_above_i32_max() {
        let too_large = i32::MAX as usize + 1;
        assert_invalid(GemmDims::checked(too_large, 1, 1, 1, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_product_overflow() {
        assert_invalid(GemmDims::checked(usize::MAX, 2, 1, 2, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_device_total_overflow() {
        let limit = i32::MAX as usize;
        assert_invalid(GemmDims::checked(limit, 2, 1, 2, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_grid_conversion_overflow() {
        assert_invalid_error(
            checked_u32(u32::MAX as usize + 1, "grid axis")
                .expect_err("an oversized grid axis must be rejected"),
        );
        assert_invalid_error(
            checked_grid_product(u32::MAX, 2, 1)
                .expect_err("an overflowing grid product must be rejected"),
        );
    }

    #[test]
    fn gemm_dims_reject_bad_nn_strides() {
        for strides in [(2, 5, 5), (3, 4, 5), (3, 5, 4), (0, 5, 5)] {
            assert_invalid(GemmDims::checked(
                2,
                strides.0.max(3),
                5,
                strides.0,
                strides.1,
                strides.2,
            ));
        }
        assert_invalid(GemmDims::checked(2, 1, 1, i32::MAX as usize, 1, 1));
        assert_invalid(GemmDims::checked(1, 1, 1, i32::MAX as usize + 1, 1, 1));
    }

    #[test]
    fn gemm_dims_preserve_tn_storage_strides() {
        let dims = GemmDims::tn((2, 3, 5)).unwrap();
        assert_eq!((dims.lda, dims.ldb, dims.ldc), (3, 5, 5));
    }

    #[test]
    fn gemm_dims_reject_bad_tn_strides() {
        assert_invalid(GemmDims::checked_storage(
            2,
            3,
            5,
            [2, 5, 5],
            [3, 5, 5],
            [2, 2, 3],
        ));
    }

    #[test]
    fn gemm_dims_preserve_nt_storage_strides() {
        let dims = GemmDims::nt((2, 3, 5)).unwrap();
        assert_eq!((dims.lda, dims.ldb, dims.ldc), (5, 5, 3));
    }

    #[test]
    fn gemm_dims_reject_bad_nt_strides() {
        assert_invalid(GemmDims::checked_storage(
            2,
            3,
            5,
            [4, 5, 3],
            [5, 5, 3],
            [2, 3, 2],
        ));
    }

    #[test]
    fn bias_preseed_rejects_non_identity_alpha() {
        let error = validate_bias_preseed(0.5, 1, "triad-test")
            .expect_err("bias pre-seeding must reject alpha != 1");
        assert!(error.contains("alpha == 1.0"), "{error}");
        validate_bias_preseed(0.5, 0, "triad-test").unwrap();
        validate_bias_preseed(1.0, 1, "triad-test").unwrap();
    }
}
