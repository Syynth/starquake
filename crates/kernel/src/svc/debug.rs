//! `OutputDebugString` (0x27): a guest's debug log line, read from guest memory and handed to a
//! [`DebugSink`].
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_debug_string.cpp` @ cf654052
//! (GPL-3.0-or-later), which succeeds at once when `len == 0`, otherwise reads `len` bytes at the
//! address and logs them.
//!
//! Intentional divergences from Eden, each pinned by a test below:
//! - **An extent that wraps past the top of the address space** (`address + len` overflows) returns
//!   [`KernelError::InvalidCurrentMemory`]. Eden doesn't check.
//! - **Unreadable guest memory** returns [`KernelError::InvalidCurrentMemory`] and outputs nothing.
//!   Eden's `ReadBlock` logs the fault, zero-fills the unreadable bytes and the call succeeds
//!   (`eden/src/core/memory.cpp`, `ReadBlockImpl`). The error code is a guess: no pinned reference
//!   checks the range.
//! - **At most [`MAX_DEBUG_STRING_LEN`] bytes are read and output.** Eden allocates and reads `len`
//!   bytes, so a guest could make the host allocate any amount. Longer strings succeed, truncated
//!   to their first [`MAX_DEBUG_STRING_LEN`] bytes, and only those bytes must be readable. The
//!   limit is starquake's own, not Horizon's.

use starquake_memory::GuestMemory;

use crate::result::{KernelError, SvcResult};

/// The most bytes of one debug string that are read and output (64 KiB). Starquake's own bound,
/// not a Horizon limit.
pub const MAX_DEBUG_STRING_LEN: u64 = 0x1_0000;

/// Where guest debug strings go.
pub trait DebugSink {
    /// Receives one debug string's bytes, at most [`MAX_DEBUG_STRING_LEN`] of them. They are the
    /// guest's bytes, not necessarily UTF-8.
    fn output(&mut self, message: &[u8]);
}

/// Reads `len` bytes at `address` from the caller's memory and outputs them.
pub fn output_debug_string(
    memory: &dyn GuestMemory,
    sink: &mut dyn DebugSink,
    address: u64,
    len: u64,
) -> SvcResult<()> {
    if len == 0 {
        return Ok(());
    }
    address
        .checked_add(len)
        .ok_or(KernelError::InvalidCurrentMemory)?;
    let read = usize::try_from(len.min(MAX_DEBUG_STRING_LEN))
        .map_err(|_| KernelError::InvalidCurrentMemory)?;
    let mut message = vec![0; read];
    memory
        .read(address, &mut message)
        .map_err(|_| KernelError::InvalidCurrentMemory)?;
    if len > MAX_DEBUG_STRING_LEN {
        tracing::warn!(
            len,
            max = MAX_DEBUG_STRING_LEN,
            "guest debug string truncated"
        );
    }
    sink.output(&message);
    Ok(())
}

#[cfg(test)]
mod tests {
    use starquake_memory::FlatMemory;

    use super::*;

    const BASE: u64 = 0x8000_0000;

    #[derive(Default)]
    struct Collect(Vec<Vec<u8>>);

    impl DebugSink for Collect {
        fn output(&mut self, message: &[u8]) {
            self.0.push(message.to_vec());
        }
    }

    fn memory_with(bytes: &[u8], size: usize) -> FlatMemory {
        let memory = FlatMemory::new(BASE, size);
        memory.write(BASE, bytes).unwrap();
        memory
    }

    #[test]
    fn outputs_the_bytes_at_the_address() {
        let memory = memory_with(b"hello, world", 0x1000);
        let mut sink = Collect::default();
        output_debug_string(&memory, &mut sink, BASE + 7, 5).unwrap();
        assert_eq!(sink.0, [b"world".to_vec()]);
    }

    #[test]
    fn empty_string_succeeds_without_reading() {
        // Eden: R_SUCCEED_IF(len == 0), before any memory access.
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        assert_eq!(output_debug_string(&memory, &mut sink, 0, 0), Ok(()));
        assert_eq!(output_debug_string(&memory, &mut sink, u64::MAX, 0), Ok(()));
        assert_eq!(sink.0, Vec::<Vec<u8>>::new());
    }

    #[test]
    fn unreadable_memory_is_invalid_current_memory_and_outputs_nothing() {
        // Divergence: Eden zero-fills and succeeds.
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        assert_eq!(
            output_debug_string(&memory, &mut sink, BASE - 1, 2),
            Err(KernelError::InvalidCurrentMemory)
        );
        // Trailing bytes past the end of mapped memory: the whole extent must be readable.
        assert_eq!(
            output_debug_string(&memory, &mut sink, BASE + 0xf, 2),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(sink.0, Vec::<Vec<u8>>::new());
        // The last readable byte exactly at the end is fine.
        output_debug_string(&memory, &mut sink, BASE + 0xf, 1).unwrap();
        assert_eq!(sink.0.len(), 1);
    }

    #[test]
    fn an_extent_that_wraps_the_address_space_is_rejected() {
        // Divergence: Eden doesn't check. address + len == 2^64 overflows by one.
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        assert_eq!(
            output_debug_string(&memory, &mut sink, u64::MAX, 1),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(
            output_debug_string(&memory, &mut sink, 1, u64::MAX),
            Err(KernelError::InvalidCurrentMemory)
        );
        // Ending exactly at the top of the address space doesn't overflow; it is then an
        // unreadable address in this memory.
        assert_eq!(
            output_debug_string(&memory, &mut sink, u64::MAX - 1, 1),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(sink.0, Vec::<Vec<u8>>::new());
    }

    #[test]
    fn a_string_at_the_limit_is_output_whole_and_one_past_it_is_truncated() {
        // Divergence: Eden reads all `len` bytes.
        let limit = usize::try_from(MAX_DEBUG_STRING_LEN).unwrap();
        let bytes: Vec<u8> = (0..=u8::MAX).cycle().take(limit + 1).collect();
        let memory = memory_with(&bytes, limit + 1);
        let mut sink = Collect::default();
        output_debug_string(&memory, &mut sink, BASE, MAX_DEBUG_STRING_LEN).unwrap();
        output_debug_string(&memory, &mut sink, BASE, MAX_DEBUG_STRING_LEN + 1).unwrap();
        assert_eq!(sink.0.len(), 2);
        assert_eq!(sink.0[0], bytes[..limit]);
        assert_eq!(sink.0[1], bytes[..limit]);
    }

    #[test]
    fn a_huge_length_reads_only_the_limit() {
        // The length is guest-controlled: it must not size the host allocation.
        let limit = usize::try_from(MAX_DEBUG_STRING_LEN).unwrap();
        let memory = FlatMemory::new(BASE, limit);
        let mut sink = Collect::default();
        output_debug_string(&memory, &mut sink, BASE, u64::MAX - BASE).unwrap();
        assert_eq!(sink.0[0].len(), limit);
    }
}
