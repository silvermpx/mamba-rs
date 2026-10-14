//! A trainer's packing on the device (see [`crate::varlen`]): the sequence
//! ids `[B, T]` the kernels read as i32.

use std::sync::Arc;

use super::buffers::GpuByteBuffer;
use crate::varlen::Varlen;

/// The packing of a trainer's next windows: the ids, uploaded on the
/// stream ahead of the windows that read them into one buffer the trainer
/// keeps, so a graph captured under a packing reads every later one.
#[derive(Default)]
pub(crate) struct GpuVarlen {
    ids: Option<GpuByteBuffer>,
    active: bool,
}

impl GpuVarlen {
    /// Pack the next windows of `batch` rows of `seq_len` positions as
    /// `packing`, or run each row as one sequence (`None`).
    pub(crate) fn set(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        packing: Option<&Varlen>,
        (batch, seq_len): (usize, usize),
    ) -> Result<(), String> {
        let Some(v) = packing else {
            self.active = false;
            return Ok(());
        };
        v.check_shape(batch, seq_len)?;
        let bytes: &[u8] = bytemuck::cast_slice(v.seq_idx());
        if bytes.is_empty() {
            self.active = false;
            return Ok(());
        }
        let buf = match self.ids.take() {
            Some(buf) => buf,
            None => GpuByteBuffer::zeros(stream, bytes.len())?,
        };
        self.ids.insert(buf).upload_bytes(stream, bytes)?;
        self.active = true;
        Ok(())
    }

    /// The address of the ids the kernels take: 0 for one sequence a row.
    pub(crate) fn ptr(&self) -> u64 {
        match (&self.ids, self.active) {
            (Some(buf), true) => buf.cached_ptr(),
            _ => 0,
        }
    }

    /// Refuse to turn a packing on or off under a captured graph, which
    /// baked in the one it was captured with; new ids under a packing
    /// replay as they are.
    pub(crate) fn check_graph(
        &self,
        captured: bool,
        packing: Option<&Varlen>,
    ) -> Result<(), String> {
        if captured && (self.ptr() != 0) != packing.is_some() {
            return Err(
                "turning a packing on or off needs drop_graph first: the captured \
                        graph runs with the packing it was captured with"
                    .into(),
            );
        }
        Ok(())
    }
}
