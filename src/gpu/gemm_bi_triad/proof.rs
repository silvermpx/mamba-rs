//! First-use admission of a route on a board that holds no frozen evidence
//! for it.
//!
//! A route measured on one board is a candidate on every other board that
//! runs its instruction tier, but its bits are not: the tensor-core
//! accumulator rounds differently from one generation to the next, so a
//! kernel that reproduces the incumbent bit for bit on Ada may not on
//! Blackwell. The board therefore decides for itself. At the first eager use
//! of a candidate for an (op, dtype, shape) the launcher runs the candidate
//! and the reference route of the same numeric contract on the same
//! operands, each into its own scratch output, and compares every output
//! word. Equal words admit the candidate for the rest of the process; any
//! difference declines it, once, with the reason said aloud. A proof never
//! runs inside a graph capture: the reference serves there, and since the
//! two routes are bit-equal whenever the candidate is admitted, the bits a
//! process produces do not depend on which call came first.
//!
//! What the proof cannot say is speed. A candidate admitted here carries the
//! speed evidence of the board it was measured on, not of this one; the
//! route identity records the admission kind so a report can say so.

use std::collections::HashMap;
use std::sync::Arc;

use super::super::buffers::{DtypedBuf, GpuBuffer};
use super::super::dtype::WeightDtype;
use super::super::kernel_identity::ResolvedGemmOp;

/// One tile's footprint on a board: the output tile each CTA computes and
/// how many of its CTAs stay resident on one multiprocessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileFootprint {
    pub(crate) tile: (u32, u32),
    pub(crate) resident: u32,
}

/// The waves a tile's grid takes over an output on a board: its CTA count
/// over the CTAs the board keeps resident, rounded up. An empty output
/// takes none.
pub(crate) fn tile_waves(
    rows: usize,
    columns: usize,
    footprint: TileFootprint,
    multiprocessors: u32,
) -> Option<u64> {
    let (tile_rows, tile_columns) = footprint.tile;
    if tile_rows == 0 || tile_columns == 0 || multiprocessors == 0 {
        return None;
    }
    let ctas = u64::try_from(rows)
        .ok()?
        .div_ceil(u64::from(tile_rows))
        .checked_mul(
            u64::try_from(columns)
                .ok()?
                .div_ceil(u64::from(tile_columns)),
        )?;
    let resident = u64::from(multiprocessors).checked_mul(u64::from(footprint.resident.max(1)))?;
    Some(ctas.div_ceil(resident))
}

/// Whether a candidate may serve on a board that holds no evidence for it,
/// by that board's own arithmetic. A candidate measured on one board wins
/// there partly by how its grid falls onto that board's multiprocessors;
/// on a board with a different count the same grid can spill into an
/// extra wave. The guard declines exactly the case no throughput advantage
/// can recover: the candidate takes strictly more waves than the reference
/// while each of its waves computes a tile no smaller. Everything else is
/// left to the bit proof and to the speed evidence of the measured board.
pub(crate) fn wave_guard_admits(
    rows: usize,
    columns: usize,
    candidate: TileFootprint,
    reference: TileFootprint,
    multiprocessors: u32,
) -> bool {
    let (Some(candidate_waves), Some(reference_waves)) = (
        tile_waves(rows, columns, candidate, multiprocessors),
        tile_waves(rows, columns, reference, multiprocessors),
    ) else {
        return false;
    };
    let candidate_area = u64::from(candidate.tile.0) * u64::from(candidate.tile.1);
    let reference_area = u64::from(reference.tile.0) * u64::from(reference.tile.1);
    !(candidate_waves > reference_waves && candidate_area >= reference_area)
}

/// One candidate on one cell: the candidate's symbol, the operation, the
/// output dtype and the exact dimensions the launch resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RouteProofKey {
    pub(crate) candidate: &'static str,
    pub(crate) op: ResolvedGemmOp,
    pub(crate) dtype: WeightDtype,
    pub(crate) dims: (usize, usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouteProofVerdict {
    /// Every output word matched the reference on this board.
    Admitted,
    /// At least one output word differed, or a launch failed; the reference
    /// serves this cell for the rest of the process.
    Declined,
}

/// The verdicts a context has reached, one per candidate and cell.
#[derive(Default)]
pub(crate) struct RouteProofLedger {
    verdicts: HashMap<RouteProofKey, RouteProofVerdict>,
}

impl RouteProofLedger {
    pub(crate) fn verdict(&self, key: RouteProofKey) -> Option<RouteProofVerdict> {
        self.verdicts.get(&key).copied()
    }

    pub(crate) fn record(&mut self, key: RouteProofKey, verdict: RouteProofVerdict) {
        self.verdicts.insert(key, verdict);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.verdicts.len()
    }
}

/// A scratch output the proof launches into: the same element count and
/// dtype as the real output, seeded from it so a route that accumulates
/// into its output (beta = 1) starts from the same words.
pub(crate) enum ProofOutput {
    F32(GpuBuffer),
    Typed(DtypedBuf),
}

impl ProofOutput {
    pub(crate) fn seeded_from(
        stream: &Arc<cudarc::driver::CudaStream>,
        source: cudarc::driver::sys::CUdeviceptr,
        elements: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let output = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => Self::F32(GpuBuffer::zeros(stream, elements)?),
            WeightDtype::Bf16 | WeightDtype::F16 => {
                Self::Typed(DtypedBuf::zeros(stream, elements, dtype)?)
            }
        };
        let bytes = elements
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| "proof output byte count overflows usize".to_string())?;
        if bytes > 0 {
            let result = unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    output.ptr(),
                    source,
                    bytes,
                    stream.cu_stream(),
                )
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!("seed the proof output: {result:?}"));
            }
        }
        Ok(output)
    }

    pub(crate) fn ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        match self {
            Self::F32(buffer) => buffer.cached_ptr(),
            Self::Typed(buffer) => buffer.cached_ptr(),
        }
    }

    /// The output words as f32 bit patterns. Half words widen exactly, so
    /// two half outputs are equal exactly when their widened words are.
    pub(crate) fn words(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<Vec<u32>, String> {
        let mut host = match self {
            Self::F32(buffer) => {
                let mut host = vec![0.0f32; buffer.len()];
                buffer.download(stream, &mut host)?;
                host
            }
            Self::Typed(buffer) => {
                let mut host = vec![0.0f32; buffer.len_elems()];
                buffer.download_f32(stream, &mut host)?;
                host
            }
        };
        Ok(host.drain(..).map(f32::to_bits).collect())
    }
}

/// Runs the two arms and compares their words. `launch` receives the
/// scratch output pointer for the arm it is asked to run; the caller owns
/// the stream synchronisation that makes the words readable.
pub(crate) fn prove_bits<Candidate, Reference>(
    stream: &Arc<cudarc::driver::CudaStream>,
    output: cudarc::driver::sys::CUdeviceptr,
    elements: usize,
    dtype: WeightDtype,
    candidate: Candidate,
    reference: Reference,
) -> Result<RouteProofVerdict, String>
where
    Candidate: FnOnce(cudarc::driver::sys::CUdeviceptr) -> Result<(), String>,
    Reference: FnOnce(cudarc::driver::sys::CUdeviceptr) -> Result<(), String>,
{
    let candidate_output = ProofOutput::seeded_from(stream, output, elements, dtype)?;
    let reference_output = ProofOutput::seeded_from(stream, output, elements, dtype)?;
    candidate(candidate_output.ptr())?;
    reference(reference_output.ptr())?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronise the route proof: {error:?}"))?;
    let candidate_words = candidate_output.words(stream)?;
    let reference_words = reference_output.words(stream)?;
    Ok(if candidate_words == reference_words {
        RouteProofVerdict::Admitted
    } else {
        RouteProofVerdict::Declined
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADA_MULTIPROCESSORS: u32 = 142;

    fn footprint(tile: (u32, u32), resident: u32) -> TileFootprint {
        TileFootprint { tile, resident }
    }

    #[test]
    fn tile_waves_count_grids_against_the_resident_ctas_of_the_board() {
        // 288 tiles of 64x64 over 284 resident CTAs: the tail wave holds four.
        assert_eq!(
            tile_waves(1536, 768, footprint((64, 64), 2), ADA_MULTIPROCESSORS),
            Some(2)
        );
        // 132 tiles of 144x96 at one CTA per multiprocessor: one wave.
        assert_eq!(
            tile_waves(4621, 384, footprint((144, 96), 1), ADA_MULTIPROCESSORS),
            Some(1)
        );
        assert_eq!(tile_waves(0, 384, footprint((64, 64), 2), 142), Some(0));
        assert_eq!(tile_waves(64, 64, footprint((0, 64), 2), 142), None);
        assert_eq!(tile_waves(64, 64, footprint((64, 64), 2), 0), None);
    }

    #[test]
    fn the_wave_guard_admits_every_measured_winner_on_its_own_board() {
        // The wide TF32 tiles against the portable tiles they replaced.
        for (rows, columns, candidate, reference) in [
            (4621, 384, footprint((144, 96), 1), footprint((128, 64), 2)),
            (
                4096,
                3072,
                footprint((128, 192), 1),
                footprint((128, 64), 2),
            ),
            (3072, 1536, footprint((192, 192), 1), footprint((64, 64), 4)),
            (768, 3072, footprint((96, 192), 1), footprint((128, 96), 1)),
            (1536, 768, footprint((96, 96), 1), footprint((128, 96), 1)),
            // The half tiles against the tiled 64x64 reference.
            (2048, 3072, footprint((128, 128), 1), footprint((64, 64), 2)),
            (128, 512, footprint((32, 16), 3), footprint((64, 64), 2)),
            (1024, 128, footprint((16, 64), 2), footprint((64, 64), 2)),
        ] {
            assert!(
                wave_guard_admits(rows, columns, candidate, reference, ADA_MULTIPROCESSORS),
                "{rows}x{columns} {candidate:?} against {reference:?}"
            );
        }
    }

    #[test]
    fn the_wave_guard_declines_a_grid_that_spills_into_a_wave_the_reference_avoids() {
        // 132 tiles of 144x96 land in one wave on 142 multiprocessors and in
        // two on 130, where the portable 128x64 grid of 222 CTAs still fits
        // one wave at two resident CTAs.
        let candidate = footprint((144, 96), 1);
        let reference = footprint((128, 64), 2);
        assert!(wave_guard_admits(4621, 384, candidate, reference, 142));
        assert!(!wave_guard_admits(4621, 384, candidate, reference, 130));
        // A smaller tile is never declined: more waves of less work is a
        // throughput question the guard does not pretend to answer.
        assert!(wave_guard_admits(
            4621,
            384,
            footprint((64, 64), 1),
            footprint((128, 64), 2),
            130
        ));
    }

    #[test]
    fn ledger_records_one_verdict_per_candidate_and_cell() {
        let mut ledger = RouteProofLedger::default();
        let key = RouteProofKey {
            candidate: "tn_sm89_relay_m64n64_bk64_s3_bf16",
            op: ResolvedGemmOp::Tn,
            dtype: WeightDtype::Bf16,
            dims: (2048, 1536, 768),
        };
        assert_eq!(ledger.verdict(key), None);
        ledger.record(key, RouteProofVerdict::Admitted);
        assert_eq!(ledger.verdict(key), Some(RouteProofVerdict::Admitted));
        let other_dims = RouteProofKey {
            dims: (2048, 768, 3072),
            ..key
        };
        assert_eq!(ledger.verdict(other_dims), None);
        ledger.record(other_dims, RouteProofVerdict::Declined);
        assert_eq!(ledger.len(), 2);
        ledger.record(key, RouteProofVerdict::Declined);
        assert_eq!(ledger.len(), 2);
        assert_eq!(ledger.verdict(key), Some(RouteProofVerdict::Declined));
    }
}
