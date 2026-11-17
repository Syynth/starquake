//! nvdrv ioctl numbers and their fixed-size arguments: how an ioctl's 32-bit number is decoded,
//! and how a device reads its argument struct from the request's in buffer and gives it back
//! (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv"; port slice 9).
//!
//! Ported from Eden (GPL-3.0-or-later) @ cf654052:
//!
//! - `eden/src/core/hle/service/nvdrv/nvdata.h`: `union Ioctl`, whose bit fields are
//!   `cmd` (bits 0..8), `group` (8..16), `length` (16..30), `is_in` (30) and `is_out` (31).
//!   libnx lays the number out the same way (`libnx/nx/include/switch/nvidia/ioctl.h`:
//!   `_NV_IOC_NRBITS` 8, `_NV_IOC_TYPEBITS` 8, `_NV_IOC_SIZEBITS` 14, `_NV_IOC_DIRBITS` 2, and
//!   `_NV_IOC_WRITE` 1, `_NV_IOC_READ` 2 in the direction field: "write" is the caller sending, so
//!   Eden's `is_in`).
//! - `eden/src/core/hle/service/nvdrv/devices/ioctl_serialization.h`: `WrapFixed`, through
//!   `WrapGeneric` with only a fixed argument, which [`Fixed`] ports. The variable-sized and
//!   inline forms (`WrapVariable`, `WrapFixedVariable`, `WrapFixedInlIn`, `WrapFixedInlOut`) come
//!   with the devices that use them (port slices 10 and 11).
//!
//! Row 23 of NX2's blocker log (`eden-nx2/NX2_NOTES.md` @ 50fa86bb, observed) is why this
//! applies to Switch 2 at all: nvsrv is the nvdrv protocol, served by both references with
//! Eden's nvdrv code.
//!
//! # What a device sees of an ioctl
//!
//! Eden's devices dispatch on [`Ioctl::group`] and [`Ioctl::command`] only
//! (`nvmap::Ioctl1`: `switch (command.group)`, then `switch (command.cmd)`): the length and the
//! two direction bits are not checked against anything. The length is what the caller declares
//! its argument struct to be; the buffers' own sizes are what is read and written. The
//! [`Ioctl::is_out`] bit decides whether the out buffer is written at all
//! (`NVDRV::Ioctl1`: `if (command.is_out != 0) ctx.WriteBuffer(output_buffer)`); `is_in` is read
//! by nothing.

#[cfg(test)]
mod tests;

/// Where [`Ioctl::command`] starts: bit 0 (`nvdata.h`: `BitField<0, 8, u32> cmd`).
pub const COMMAND_SHIFT: u32 = 0;
/// How wide [`Ioctl::command`] is: 8 bits (`BitField<0, 8, u32> cmd`; `_NV_IOC_NRBITS`).
pub const COMMAND_BITS: u32 = 8;
/// Where [`Ioctl::group`] starts: bit 8 (`BitField<8, 8, u32> group`).
pub const GROUP_SHIFT: u32 = 8;
/// How wide [`Ioctl::group`] is: 8 bits (`BitField<8, 8, u32> group`; `_NV_IOC_TYPEBITS`).
pub const GROUP_BITS: u32 = 8;
/// Where [`Ioctl::size`] starts: bit 16 (`BitField<16, 14, u32> length`).
pub const SIZE_SHIFT: u32 = 16;
/// How wide [`Ioctl::size`] is: 14 bits (`BitField<16, 14, u32> length`; `_NV_IOC_SIZEBITS`).
pub const SIZE_BITS: u32 = 14;
/// The bit that says the caller sends the argument: bit 30 (`BitField<30, 1, u32> is_in`;
/// libnx `_NV_IOC_WRITE` in the direction field at `_NV_IOC_DIRSHIFT`, 30).
pub const IN_BIT: u32 = 30;
/// The bit that says the caller receives the argument back: bit 31 (`BitField<31, 1, u32>
/// is_out`; libnx `_NV_IOC_READ`, the direction field's second bit).
pub const OUT_BIT: u32 = 31;

/// The largest argument an ioctl number can declare: [`Ioctl::size`]'s widest value, 0x3FFF
/// bytes.
pub const MAX_ARGUMENT_BYTES: u32 = mask(SIZE_BITS);

/// The low `bits` bits set.
const fn mask(bits: u32) -> u32 {
    (1 << bits) - 1
}

/// An ioctl number as the caller sends it, decoded field by field (`nvdata.h`: `union Ioctl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ioctl(u32);

impl Ioctl {
    /// The ioctl `raw` names.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// The number as sent.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// The command within the group (`cmd`).
    #[must_use]
    pub const fn command(self) -> u32 {
        (self.0 >> COMMAND_SHIFT) & mask(COMMAND_BITS)
    }

    /// The group, which names the device's ioctl family (`group`; nvmap's is 1).
    #[must_use]
    pub const fn group(self) -> u32 {
        (self.0 >> GROUP_SHIFT) & mask(GROUP_BITS)
    }

    /// The size the caller declares its argument struct to be (`length`).
    #[must_use]
    pub const fn size(self) -> u32 {
        (self.0 >> SIZE_SHIFT) & mask(SIZE_BITS)
    }

    /// Whether the caller sends the argument (`is_in`).
    #[must_use]
    pub const fn is_in(self) -> bool {
        (self.0 >> IN_BIT) & 1 != 0
    }

    /// Whether the caller wants the argument back in its out buffer (`is_out`).
    #[must_use]
    pub const fn is_out(self) -> bool {
        (self.0 >> OUT_BIT) & 1 != 0
    }
}

/// A device's fixed-size argument: `N` bytes, read from the start of the in buffer and given back
/// whole (`ioctl_serialization.h`, `WrapGeneric` with a `FixedArg` and nothing else).
///
/// `WrapGeneric` value-initialises the struct (`FixedArg fixed{}`), copies
/// `min(sizeof(FixedArg), input.size())` bytes of the input over it, lets the device change it,
/// and copies `min(output.size(), sizeof(FixedArg))` bytes of it to the output. So a short input
/// leaves the fields past its end zero, bytes past the struct are ignored, and every field the
/// device does not change goes back to the caller as it came. [`Fixed::from_input`] and
/// [`Fixed::into_output`] are the two copies; the framing clips the output to the caller's out
/// buffer.
///
/// Fields are read and written at constant byte offsets, checked against `N` when the program is
/// compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fixed<const N: usize>([u8; N]);

impl<const N: usize> Fixed<N> {
    /// The argument as `input` holds it: its first `N` bytes, zero past its end.
    pub(crate) fn from_input(input: &[u8]) -> Self {
        let mut bytes = [0; N];
        let len = input.len().min(N);
        if let (Some(to), Some(from)) = (bytes.get_mut(..len), input.get(..len)) {
            to.copy_from_slice(from);
        }
        Self(bytes)
    }

    /// The argument as the device leaves it, all `N` bytes.
    pub(crate) fn into_output(self) -> Vec<u8> {
        self.0.to_vec()
    }

    /// The four bytes at `AT`.
    const fn word_bytes<const AT: usize>(&self) -> [u8; 4] {
        const { assert!(AT + 4 <= N, "a field past the end of its argument") };
        [self.0[AT], self.0[AT + 1], self.0[AT + 2], self.0[AT + 3]]
    }

    /// The byte at `AT`.
    pub(crate) const fn u8_at<const AT: usize>(&self) -> u8 {
        const { assert!(AT < N, "a field past the end of its argument") };
        self.0[AT]
    }

    /// The little-endian `u32` at `AT`.
    pub(crate) const fn u32_at<const AT: usize>(&self) -> u32 {
        u32::from_le_bytes(self.word_bytes::<AT>())
    }

    /// The little-endian `u64` at `AT`.
    pub(crate) const fn u64_at<const AT: usize>(&self) -> u64 {
        const { assert!(AT + 8 <= N, "a field past the end of its argument") };
        u64::from_le_bytes([
            self.0[AT],
            self.0[AT + 1],
            self.0[AT + 2],
            self.0[AT + 3],
            self.0[AT + 4],
            self.0[AT + 5],
            self.0[AT + 6],
            self.0[AT + 7],
        ])
    }

    /// Writes `value` as a little-endian `u16` at `AT`.
    pub(crate) fn set_u16<const AT: usize>(&mut self, value: u16) {
        const { assert!(AT + 2 <= N, "a field past the end of its argument") };
        let [a, b] = value.to_le_bytes();
        self.0[AT] = a;
        self.0[AT + 1] = b;
    }

    /// The whole argument, as bytes.
    pub(crate) const fn bytes(&self) -> &[u8; N] {
        &self.0
    }

    /// Writes `value` as a little-endian `u32` at `AT`.
    pub(crate) fn set_u32<const AT: usize>(&mut self, value: u32) {
        const { assert!(AT + 4 <= N, "a field past the end of its argument") };
        let [a, b, c, d] = value.to_le_bytes();
        self.0[AT] = a;
        self.0[AT + 1] = b;
        self.0[AT + 2] = c;
        self.0[AT + 3] = d;
    }

    /// Writes `value` as a little-endian `u64` at `AT`.
    pub(crate) fn set_u64<const AT: usize>(&mut self, value: u64) {
        const { assert!(AT + 8 <= N, "a field past the end of its argument") };
        for (offset, byte) in value.to_le_bytes().into_iter().enumerate() {
            self.0[AT + offset] = byte;
        }
    }
}
