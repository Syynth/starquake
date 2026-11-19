//! Byte-addressed memory accesses: the alignment and bounds checks, in that order, and
//! little-endian reads and writes (see the parent module's "Memory").

use std::ops::Range;

use crate::ir::Value;

use super::{InterpError, Space};

/// The largest access, in bytes (`LoadGlobal128` and its siblings).
pub(super) const MAX_ACCESS: usize = 16;

/// Where an access goes: the space, the address of its first byte (an offset into the space for
/// every space but global memory) and the address of the space's first byte.
#[derive(Clone, Copy, Debug)]
pub(super) struct Place {
    /// The instruction.
    pub(super) at: Value,
    /// The space.
    pub(super) space: Space,
    /// The address accessed.
    pub(super) address: u64,
    /// The address of the space's first byte (0 but for global memory).
    pub(super) base: u64,
}

impl Place {
    /// The bytes an access of `size` bytes covers in a space of `len` bytes: the address must
    /// be a multiple of `size` ([`InterpError::Misaligned`]), then the whole access must lie
    /// inside the space ([`InterpError::OutOfBounds`]).
    pub(super) fn range(self, size: u64, len: usize) -> Result<Range<usize>, InterpError> {
        let Self {
            at,
            space,
            address,
            base,
        } = self;
        if size == 0 || address % size != 0 {
            return Err(InterpError::Misaligned {
                at,
                space,
                offset: address,
                size,
            });
        }
        let len64 = u64::try_from(len).unwrap_or(u64::MAX);
        let out_of_bounds = || InterpError::OutOfBounds {
            at,
            space,
            offset: address,
            size,
            len: len64,
        };
        let start = address.checked_sub(base).ok_or_else(out_of_bounds)?;
        let end = start.checked_add(size).ok_or_else(out_of_bounds)?;
        if end > len64 {
            return Err(out_of_bounds());
        }
        let start = usize::try_from(start).map_err(|_| out_of_bounds())?;
        let end = usize::try_from(end).map_err(|_| out_of_bounds())?;
        Ok(start..end)
    }

    /// `size` bytes (at most [`MAX_ACCESS`]) of `bytes`, little-endian, as a 128-bit integer.
    pub(super) fn read(self, bytes: &[u8], size: u64) -> Result<u128, InterpError> {
        let range = self.range(size, bytes.len())?;
        let mut buffer = [0_u8; MAX_ACCESS];
        if let (Some(source), Some(target)) =
            (bytes.get(range.clone()), buffer.get_mut(..range.len()))
        {
            target.copy_from_slice(source);
        }
        Ok(u128::from_le_bytes(buffer))
    }

    /// Writes the low `size` bytes (at most [`MAX_ACCESS`]) of `value` into `bytes`,
    /// little-endian.
    pub(super) fn write(self, bytes: &mut [u8], size: u64, value: u128) -> Result<(), InterpError> {
        let range = self.range(size, bytes.len())?;
        let source = value.to_le_bytes();
        if let (Some(target), Some(source)) =
            (bytes.get_mut(range.clone()), source.get(..range.len()))
        {
            target.copy_from_slice(source);
        }
        Ok(())
    }
}
