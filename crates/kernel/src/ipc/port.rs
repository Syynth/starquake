//! [`PortName`]: the name a named port is registered under and `ConnectToNamedPort` asks for.
//!
//! Horizon keeps a named port's name in a fixed array of [`NAME_LENGTH_MAX`] bytes whose last
//! byte must be zero, so a name has at most `NAME_LENGTH_MAX - 1` characters
//! (`eden/src/core/hle/kernel/k_object_name.h` @ cf654052, `NameLengthMax = 12`).
//!
//! Reading a name out of guest memory follows Horizon, as Atmosphère's kernel shows it (fact
//! only: `atmosphere/libraries/libmesosphere/source/svc/kern_svc_port.cpp`, `GetPortName` and
//! `ConnectToNamedPort`, and the byte loop of `CopyStringFromUser` in
//! `atmosphere/libraries/libmesosphere/source/arch/arm64/kern_userspace_memory_access_asm.s`):
//! bytes are copied one at a time with an unprivileged load until a zero byte has been copied
//! or [`NAME_LENGTH_MAX`] bytes have; a byte the guest may not read is
//! [`KernelError::InvalidPointer`]; and when the last of the twelve is not zero the name is
//! [`KernelError::OutOfRange`].
//!
//! Where Eden differs, starquake follows the facts above, pinned by the test named:
//! - Eden reads the name with `ReadCString`, which does not check that the guest may read it,
//!   and copies it with `strncpy(name, s, NameLengthMax - 1)`, which keeps the last byte zero,
//!   so its `ResultOutOfRange` check can never fail and a twelve-byte name is cut to eleven
//!   (`eden/src/core/hle/kernel/svc/svc_port.cpp` @ cf654052, `ConnectToNamedPort`). Here a name
//!   with no zero byte in its first twelve is [`KernelError::OutOfRange`]
//!   (`a_name_with_no_terminator_in_twelve_bytes_is_out_of_range`), and an unreadable byte is
//!   [`KernelError::InvalidPointer`] (`a_name_the_guest_may_not_read_is_an_invalid_pointer`).

use std::fmt;

use starquake_memory::GuestMemory;

use crate::memory::AddressSpace;
use crate::result::KernelError;

/// The size of a port name's array, terminator included (`KObjectName::NameLengthMax`).
pub const NAME_LENGTH_MAX: usize = 12;

/// Why a string cannot name a port.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PortNameError {
    /// The name is longer than `NAME_LENGTH_MAX - 1` bytes.
    #[error("port name {0:?} is longer than {max} bytes", max = NAME_LENGTH_MAX - 1)]
    TooLong(String),
    /// The name holds a zero byte, which would end it early.
    #[error("port name {0:?} holds a zero byte")]
    Nul(String),
}

/// A port's name: its bytes, then zeros, in [`NAME_LENGTH_MAX`] bytes whose last is zero.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PortName([u8; NAME_LENGTH_MAX]);

impl PortName {
    /// The name `name`: at most `NAME_LENGTH_MAX - 1` bytes, none of them zero. The empty name
    /// is a name: a guest can ask for it, and nothing is registered under it.
    pub fn new(name: &str) -> Result<Self, PortNameError> {
        if name.len() >= NAME_LENGTH_MAX {
            return Err(PortNameError::TooLong(name.to_owned()));
        }
        if name.bytes().any(|byte| byte == 0) {
            return Err(PortNameError::Nul(name.to_owned()));
        }
        let mut bytes = [0; NAME_LENGTH_MAX];
        for (slot, byte) in bytes.iter_mut().zip(name.bytes()) {
            *slot = byte;
        }
        Ok(Self(bytes))
    }

    /// Reads the name a guest passed at `address`, as the module docs say: byte by byte, each
    /// one checked against what the guest may read in `space`, until a zero byte or
    /// [`NAME_LENGTH_MAX`] bytes. Nothing past the terminator is read.
    pub fn read(
        space: &AddressSpace,
        memory: &dyn GuestMemory,
        address: u64,
    ) -> Result<Self, KernelError> {
        let mut bytes = [0; NAME_LENGTH_MAX];
        for (offset, slot) in (0_u64..).zip(bytes.iter_mut()) {
            let at = address
                .checked_add(offset)
                .ok_or(KernelError::InvalidPointer)?;
            if !space.user_can_read(at, 1) {
                return Err(KernelError::InvalidPointer);
            }
            let mut byte = [0; 1];
            memory
                .read(at, &mut byte)
                .map_err(|_| KernelError::InvalidPointer)?;
            *slot = byte[0];
            if byte[0] == 0 {
                return Ok(Self(bytes));
            }
        }
        // Twelve bytes and no terminator: the last is not zero.
        Err(KernelError::OutOfRange)
    }

    /// The name's bytes, without the zeros after it.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.0.iter().position(|byte| *byte == 0).unwrap_or(0);
        self.0.get(..len).unwrap_or_default()
    }
}

impl fmt::Debug for PortName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PortName({self})")
    }
}

impl fmt::Display for PortName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&String::from_utf8_lossy(self.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use starquake_memory::FlatMemory;
    use starquake_memory::PAGE_SIZE;

    use super::*;
    use crate::memory::fixture::{self, DATA, DATA_PAGES};

    /// The size of the fixture's data segment.
    const DATA_SIZE: u64 = DATA_PAGES * PAGE_SIZE;

    /// The fixture's address space and a flat memory over its data segment and the page after.
    fn guest() -> (AddressSpace, FlatMemory) {
        let (space, _, _) = fixture::loaded();
        let size = usize::try_from(DATA_SIZE + PAGE_SIZE).unwrap();
        (space, FlatMemory::new(DATA, size))
    }

    #[test]
    fn a_name_keeps_its_bytes_and_zeros_after_them() {
        let name = PortName::new("sm:").unwrap();
        assert_eq!(name.as_bytes(), b"sm:");
        assert_eq!(name.0, *b"sm:\0\0\0\0\0\0\0\0\0");
        assert_eq!(name.to_string(), "sm:");
        assert_eq!(format!("{name:?}"), "PortName(sm:)");
        assert_eq!(PortName::new("").unwrap().as_bytes(), b"");
    }

    #[test]
    fn a_name_has_at_most_eleven_bytes_and_no_zero_byte() {
        let longest = "a".repeat(NAME_LENGTH_MAX - 1);
        assert_eq!(
            PortName::new(&longest).unwrap().as_bytes(),
            longest.as_bytes()
        );
        let too_long = "a".repeat(NAME_LENGTH_MAX);
        assert_eq!(
            PortName::new(&too_long),
            Err(PortNameError::TooLong(too_long.clone()))
        );
        assert_eq!(
            PortNameError::TooLong(too_long.clone()).to_string(),
            format!("port name {too_long:?} is longer than 11 bytes")
        );
        assert_eq!(
            PortName::new("s\0m"),
            Err(PortNameError::Nul("s\0m".to_owned()))
        );
    }

    #[test]
    fn a_name_is_read_up_to_its_terminator() {
        let (space, memory) = guest();
        memory.write(DATA, b"sm:\0garbage!").unwrap();
        assert_eq!(
            PortName::read(&space, &memory, DATA),
            Ok(PortName::new("sm:").unwrap())
        );
        // An eleven-byte name ends in the twelfth byte, exactly at the limit.
        let longest = "b".repeat(NAME_LENGTH_MAX - 1);
        memory.write(DATA, longest.as_bytes()).unwrap();
        memory.write(DATA + 11, &[0]).unwrap();
        assert_eq!(
            PortName::read(&space, &memory, DATA),
            Ok(PortName::new(&longest).unwrap())
        );
    }

    #[test]
    fn a_name_with_no_terminator_in_twelve_bytes_is_out_of_range() {
        let (space, memory) = guest();
        memory.write(DATA, &[b'c'; NAME_LENGTH_MAX + 4]).unwrap();
        assert_eq!(
            PortName::read(&space, &memory, DATA),
            Err(KernelError::OutOfRange)
        );
    }

    #[test]
    fn a_name_the_guest_may_not_read_is_an_invalid_pointer() {
        let (space, memory) = guest();
        // The data segment ends where nothing is mapped: a name that runs up to its last byte
        // without a terminator reads one byte past it.
        let end = DATA + DATA_SIZE;
        memory.write(end - 2, b"ab").unwrap();
        assert_eq!(
            PortName::read(&space, &memory, end - 2),
            Err(KernelError::InvalidPointer)
        );
        // The same name with its terminator inside the segment reads.
        memory.write(end - 3, b"a\0").unwrap();
        assert_eq!(
            PortName::read(&space, &memory, end - 3),
            Ok(PortName::new("a").unwrap())
        );
        // An address at the top of the address space does not wrap.
        assert_eq!(
            PortName::read(&space, &memory, u64::MAX),
            Err(KernelError::InvalidPointer)
        );
    }
}
