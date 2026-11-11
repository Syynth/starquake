//! Guest address spaces and physical memory.
//!
//! Backends differ in who translates guest virtual addresses: HVF and KVM let the hardware walk
//! the guest's stage-1 tables (we manage stage-2, which must be 16 KiB aligned on Apple hosts),
//! while a JIT translates in software. Everything above the CPU layer (kernel, IPC, GPU) only sees
//! [`GuestMemory`].
//!
//! [`translation`] holds the [`Translation`] trait that maps guest virtual ranges; [`stage1`]
//! implements it with AArch64 stage-1 tables for the native backends.

pub mod stage1;
pub mod translation;

pub use translation::{Access, BackingRange, MapError, PAGE_SIZE, Perms, Translation};

/// An access to guest memory that isn't mapped, or isn't allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("guest memory fault at {address:#x}")]
pub struct Fault {
    /// The first guest address that could not be accessed.
    pub address: u64,
}

/// Byte-level access to one guest address space.
pub trait GuestMemory: Send + Sync {
    /// Copies `buf.len()` bytes starting at guest virtual address `va` into `buf`.
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault>;
    /// Copies `data` into guest memory starting at guest virtual address `va`.
    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault>;
}

/// A single contiguous region of guest memory, for tests and tools.
#[derive(Debug)]
pub struct FlatMemory {
    base: u64,
    bytes: std::sync::RwLock<Vec<u8>>,
}

impl FlatMemory {
    /// Creates `size` zeroed bytes mapped at guest address `base`.
    pub fn new(base: u64, size: usize) -> Self {
        Self {
            base,
            bytes: std::sync::RwLock::new(vec![0; size]),
        }
    }

    fn range(&self, va: u64, len: usize, size: usize) -> Result<std::ops::Range<usize>, Fault> {
        let fault = Fault { address: va };
        let start = usize::try_from(va.checked_sub(self.base).ok_or(fault)?).map_err(|_| fault)?;
        let end = start.checked_add(len).ok_or(fault)?;
        if end > size {
            return Err(fault);
        }
        Ok(start..end)
    }
}

impl GuestMemory for FlatMemory {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let bytes = self
            .bytes
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let range = self.range(va, buf.len(), bytes.len())?;
        buf.copy_from_slice(&bytes[range]);
        Ok(())
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let mut bytes = self
            .bytes
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let len = bytes.len();
        let range = self.range(va, data.len(), len)?;
        bytes[range].copy_from_slice(data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_memory_round_trips_and_faults_outside() {
        let mem = FlatMemory::new(0x8000_0000, 0x1000);
        mem.write(0x8000_0ff8, &42u64.to_le_bytes()).unwrap();
        let mut buf = [0; 8];
        mem.read(0x8000_0ff8, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), 42);
        assert_eq!(
            mem.read(0x8000_0ffc, &mut buf),
            Err(Fault {
                address: 0x8000_0ffc
            })
        );
        assert_eq!(
            mem.read(0x7fff_ffff, &mut buf),
            Err(Fault {
                address: 0x7fff_ffff
            })
        );
    }
}
