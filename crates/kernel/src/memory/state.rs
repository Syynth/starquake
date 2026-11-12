//! What a memory block says about its pages: Horizon's memory states, permissions, attributes
//! and merge barriers, as bit sets with the reference's values.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_memory_block.h` @ cf654052 (GPL-3.0-or-later):
//! `KMemoryState`, `KMemoryPermission`, `ConvertToKMemoryPermission`, `KMemoryAttribute` and
//! `KMemoryBlockDisableMergeAttribute`. The values a system call sees are those of
//! `eden/src/core/hle/kernel/svc_types.h` @ cf654052 (`MemoryState`, `MemoryPermission`,
//! `MemoryAttribute`).
//!
//! Not ported: the Switch 2 attribute bit `GpuAccessible` (`1 << 5`,
//! `eden-pr4588/src/core/hle/kernel/svc_types.h` @ e5fb3b97), which only `svc 0x80` sets. It
//! lands with that call.

use core::ops::{BitAnd, BitOr, Not};

use starquake_memory::Perms;

/// The flag bits of a memory state word (`KMemoryState::Flag*`): what may be done with memory in
/// that state. Bits 0-7 of the word are the state a system call reports ([`STATE_MASK`]).
pub mod flag {
    /// `FlagCanReprotect`: `SetMemoryPermission` may change it.
    pub const CAN_REPROTECT: u32 = 1 << 8;
    /// `FlagCanDebug`.
    pub const CAN_DEBUG: u32 = 1 << 9;
    /// `FlagCanUseIpc`.
    pub const CAN_USE_IPC: u32 = 1 << 10;
    /// `FlagCanUseNonDeviceIpc`.
    pub const CAN_USE_NON_DEVICE_IPC: u32 = 1 << 11;
    /// `FlagCanUseNonSecureIpc`.
    pub const CAN_USE_NON_SECURE_IPC: u32 = 1 << 12;
    /// `FlagMapped`.
    pub const MAPPED: u32 = 1 << 13;
    /// `FlagCode`: `SetProcessMemoryPermission` may change it.
    pub const CODE: u32 = 1 << 14;
    /// `FlagCanAlias`.
    pub const CAN_ALIAS: u32 = 1 << 15;
    /// `FlagCanCodeAlias`.
    pub const CAN_CODE_ALIAS: u32 = 1 << 16;
    /// `FlagCanTransfer`.
    pub const CAN_TRANSFER: u32 = 1 << 17;
    /// `FlagCanQueryPhysical`.
    pub const CAN_QUERY_PHYSICAL: u32 = 1 << 18;
    /// `FlagCanDeviceMap`.
    pub const CAN_DEVICE_MAP: u32 = 1 << 19;
    /// `FlagCanAlignedDeviceMap`.
    pub const CAN_ALIGNED_DEVICE_MAP: u32 = 1 << 20;
    /// `FlagCanIpcUserBuffer`.
    pub const CAN_IPC_USER_BUFFER: u32 = 1 << 21;
    /// `FlagReferenceCounted`.
    pub const REFERENCE_COUNTED: u32 = 1 << 22;
    /// `FlagCanMapProcess`.
    pub const CAN_MAP_PROCESS: u32 = 1 << 23;
    /// `FlagCanChangeAttribute`: `SetMemoryAttribute` may change its `Uncached` bit.
    pub const CAN_CHANGE_ATTRIBUTE: u32 = 1 << 24;
    /// `FlagCanCodeMemory`.
    pub const CAN_CODE_MEMORY: u32 = 1 << 25;
    /// `FlagLinearMapped`.
    pub const LINEAR_MAPPED: u32 = 1 << 26;
    /// `FlagCanPermissionLock`: `SetMemoryAttribute` may set its `PermissionLocked` bit.
    pub const CAN_PERMISSION_LOCK: u32 = 1 << 27;

    /// `FlagsData`.
    pub const DATA: u32 = CAN_REPROTECT
        | CAN_USE_IPC
        | CAN_USE_NON_DEVICE_IPC
        | CAN_USE_NON_SECURE_IPC
        | MAPPED
        | CAN_ALIAS
        | CAN_TRANSFER
        | CAN_QUERY_PHYSICAL
        | CAN_DEVICE_MAP
        | CAN_ALIGNED_DEVICE_MAP
        | CAN_IPC_USER_BUFFER
        | REFERENCE_COUNTED
        | CAN_CHANGE_ATTRIBUTE
        | LINEAR_MAPPED;
    /// `FlagsCode`.
    pub const CODE_FLAGS: u32 = CAN_DEBUG
        | CAN_USE_IPC
        | CAN_USE_NON_DEVICE_IPC
        | CAN_USE_NON_SECURE_IPC
        | MAPPED
        | CODE
        | CAN_QUERY_PHYSICAL
        | CAN_DEVICE_MAP
        | CAN_ALIGNED_DEVICE_MAP
        | REFERENCE_COUNTED
        | LINEAR_MAPPED;
    /// `FlagsMisc`.
    pub const MISC: u32 =
        MAPPED | REFERENCE_COUNTED | CAN_QUERY_PHYSICAL | CAN_DEVICE_MAP | LINEAR_MAPPED;
}

/// The bits of a state word that are the state a system call reports (`KMemoryState::Mask`).
pub const STATE_MASK: u32 = 0xff;

/// Every bit of a state word (`KMemoryState::All`): a test with this mask compares whole words.
pub const STATE_ALL: u32 = u32::MAX;

/// Defines [`MemoryState`] from `Name = svc value, flags;` rows, so a state's number and its
/// flags are written once.
macro_rules! memory_states {
    ($($(#[$doc:meta])* $name:ident = $svc:literal, $flags:expr;)*) => {
        /// The state of a run of pages (`KMemoryState`): what the memory is, which is the number
        /// a system call reports, and what may be done with it ([`flag`]).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum MemoryState {
            $($(#[$doc])* $name,)*
        }

        impl MemoryState {
            /// Every state, in the order of Eden's `KMemoryState`.
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            /// The whole state word: the reported state in bits 0-7 and the flags above.
            #[must_use]
            pub const fn word(self) -> u32 {
                match self {
                    $(Self::$name => $svc | $flags,)*
                }
            }

            /// Eden's name for the state.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$name => stringify!($name),)*
                }
            }
        }
    };
}

memory_states! {
    /// Nothing is mapped.
    Free = 0x00, 0;
    /// Memory-mapped I/O that is mapped.
    IoMemory = 0x01, flag::MAPPED | flag::CAN_DEVICE_MAP | flag::CAN_ALIGNED_DEVICE_MAP;
    /// Memory-mapped I/O registers.
    IoRegister = 0x01, flag::CAN_DEVICE_MAP | flag::CAN_ALIGNED_DEVICE_MAP;
    /// Static kernel mappings.
    Static = 0x02, flag::MAPPED | flag::CAN_QUERY_PHYSICAL;
    /// A module's code, as loaded.
    Code = 0x03, flag::CODE_FLAGS | flag::CAN_MAP_PROCESS;
    /// A module's writable data.
    CodeData = 0x04,
        flag::DATA | flag::CAN_MAP_PROCESS | flag::CAN_CODE_MEMORY | flag::CAN_PERMISSION_LOCK;
    /// Heap memory.
    Normal = 0x05, flag::DATA | flag::CAN_CODE_MEMORY;
    /// Shared memory.
    Shared = 0x06, flag::MAPPED | flag::REFERENCE_COUNTED | flag::LINEAR_MAPPED;
    /// Code mapped with `MapProcessCodeMemory`.
    AliasCode = 0x08, flag::CODE_FLAGS | flag::CAN_MAP_PROCESS | flag::CAN_CODE_ALIAS;
    /// Data mapped with `MapProcessCodeMemory`.
    AliasCodeData = 0x09,
        flag::DATA
            | flag::CAN_MAP_PROCESS
            | flag::CAN_CODE_ALIAS
            | flag::CAN_CODE_MEMORY
            | flag::CAN_PERMISSION_LOCK;
    /// An IPC buffer mapped into a server.
    Ipc = 0x0a,
        flag::MISC
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_USE_IPC
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// A thread's stack.
    Stack = 0x0b,
        flag::MISC
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_USE_IPC
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// A page of thread-local regions.
    ThreadLocal = 0x0c, flag::LINEAR_MAPPED;
    /// Transfer memory mapped by its receiver.
    Transferred = 0x0d,
        flag::MISC
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_CHANGE_ATTRIBUTE
            | flag::CAN_USE_IPC
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// Shared transfer memory.
    SharedTransferred = 0x0e,
        flag::MISC
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// Shared code.
    SharedCode = 0x0f,
        flag::MAPPED
            | flag::REFERENCE_COUNTED
            | flag::LINEAR_MAPPED
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// Not accessible: what `QueryMemory` reports above the address space.
    Inaccessible = 0x10, 0;
    /// An IPC buffer mapped without the secure flag.
    NonSecureIpc = 0x11,
        flag::MISC
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
    /// An IPC buffer mapped without the device flag.
    NonDeviceIpc = 0x12, flag::MISC | flag::CAN_USE_NON_DEVICE_IPC;
    /// Kernel memory.
    Kernel = 0x13, 0;
    /// Code written by a JIT.
    GeneratedCode = 0x14,
        flag::MAPPED | flag::REFERENCE_COUNTED | flag::CAN_DEBUG | flag::LINEAR_MAPPED;
    /// The writable alias of JIT code.
    CodeOut = 0x15, flag::MAPPED | flag::REFERENCE_COUNTED | flag::LINEAR_MAPPED;
    /// Coverage data.
    Coverage = 0x16, flag::MAPPED;
    /// Insecure memory.
    Insecure = 0x17,
        flag::MAPPED
            | flag::REFERENCE_COUNTED
            | flag::LINEAR_MAPPED
            | flag::CAN_CHANGE_ATTRIBUTE
            | flag::CAN_DEVICE_MAP
            | flag::CAN_ALIGNED_DEVICE_MAP
            | flag::CAN_QUERY_PHYSICAL
            | flag::CAN_USE_NON_SECURE_IPC
            | flag::CAN_USE_NON_DEVICE_IPC;
}

impl MemoryState {
    /// The state a system call reports (`Svc::MemoryState`): the word's low eight bits.
    #[must_use]
    pub const fn svc(self) -> u32 {
        self.word() & STATE_MASK
    }
}

/// Defines a set of bits in a `u8`, with the set operations the port needs.
macro_rules! bit_set {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(u8);

        impl $name {
            /// The set with exactly these bits.
            #[must_use]
            pub const fn from_bits(bits: u8) -> Self {
                Self(bits)
            }

            /// The bits.
            #[must_use]
            pub const fn bits(self) -> u8 {
                self.0
            }

            /// Whether every bit of `other` is in this set.
            #[must_use]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Whether no bit of `other` is in this set.
            #[must_use]
            pub const fn excludes(self, other: Self) -> bool {
                self.0 & other.0 == 0
            }
        }

        impl BitOr for $name {
            type Output = Self;

            fn bitor(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
        }

        impl BitAnd for $name {
            type Output = Self;

            fn bitand(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }
        }

        impl Not for $name {
            type Output = Self;

            fn not(self) -> Self {
                Self(!self.0)
            }
        }
    };
}

bit_set! {
    /// Who may do what with a run of pages (`KMemoryPermission`): the user's read, write and
    /// execute bits in bits 0-2, which is what a system call reports, the kernel's in bits 3-5,
    /// and `NotMapped` in bit 6.
    MemoryPermission
}

impl MemoryPermission {
    /// How far the kernel's bits sit above the user's (`KernelShift`).
    const KERNEL_SHIFT: u32 = 3;
    /// `Svc::MemoryPermission::Read`.
    const READ: u8 = 1 << 0;
    /// `Svc::MemoryPermission::Write`.
    const WRITE: u8 = 1 << 1;
    /// `Svc::MemoryPermission::Execute`.
    const EXECUTE: u8 = 1 << 2;

    /// No bit.
    pub const NONE: Self = Self(0);
    /// Every bit: a test with this mask compares whole permissions.
    pub const ALL: Self = Self(u8::MAX);
    /// The kernel may read.
    pub const KERNEL_READ: Self = Self(Self::READ << Self::KERNEL_SHIFT);
    /// The kernel may write.
    pub const KERNEL_WRITE: Self = Self(Self::WRITE << Self::KERNEL_SHIFT);
    /// Nothing is mapped for the user.
    pub const NOT_MAPPED: Self = Self(1 << (2 * Self::KERNEL_SHIFT));
    /// The user (and so the kernel) may read.
    pub const USER_READ: Self = Self(Self::READ | Self::KERNEL_READ.0);
    /// The user (and so the kernel) may write.
    pub const USER_WRITE: Self = Self(Self::WRITE | Self::KERNEL_WRITE.0);
    /// The user may execute.
    pub const USER_EXECUTE: Self = Self(Self::EXECUTE);
    /// The user may read and write.
    pub const USER_READ_WRITE: Self = Self(Self::USER_READ.0 | Self::USER_WRITE.0);
    /// The user may read and execute.
    pub const USER_READ_EXECUTE: Self = Self(Self::USER_READ.0 | Self::USER_EXECUTE.0);
    /// The user's three bits: what a system call reports.
    pub const USER_MASK: Self = Self(Self::READ | Self::WRITE | Self::EXECUTE);
    /// The bits an IPC lock changes (`IpcLockChangeMask`).
    pub const IPC_LOCK_CHANGE_MASK: Self = Self(Self::NOT_MAPPED.0 | Self::USER_READ_WRITE.0);

    /// The kernel's permission for what a system call asks (`ConvertToKMemoryPermission`): the
    /// user's bits, the kernel's read bit always, the kernel's write bit with the user's, and
    /// `NotMapped` when the user gets nothing.
    #[must_use]
    pub const fn from_svc(permission: SvcPermission) -> Self {
        let user = permission.bits();
        let write = (user & Self::WRITE) << Self::KERNEL_SHIFT;
        let not_mapped = if user == 0 { Self::NOT_MAPPED.0 } else { 0 };
        Self(user | Self::KERNEL_READ.0 | write | not_mapped)
    }

    /// The permission a system call reports (`Svc::MemoryPermission`): the user's bits.
    #[must_use]
    pub fn svc(self) -> u32 {
        u32::from(self.0 & Self::USER_MASK.0)
    }

    /// What guest code may do with the pages, for the translation: the user's bits as
    /// [`Perms`]. `None` for write without read, which no stage-1 descriptor can express and
    /// no Horizon permission asks for.
    #[must_use]
    pub const fn guest(self) -> Option<Perms> {
        let user = self.0 & Self::USER_MASK.0;
        let (read, write, execute) = (
            user & Self::READ != 0,
            user & Self::WRITE != 0,
            user & Self::EXECUTE != 0,
        );
        match (read, write, execute) {
            (false, false, false) => Some(Perms::None),
            (true, false, false) => Some(Perms::Read),
            (true, true, false) => Some(Perms::ReadWrite),
            (false, false, true) => Some(Perms::Execute),
            (true, false, true) => Some(Perms::ReadExecute),
            (true, true, true) => Some(Perms::ReadWriteExecute),
            (false, true, _) => None,
        }
    }
}

/// A permission a system call may ask for (`Svc::MemoryPermission`, without `DontCare`): the
/// combinations of read, write and execute that Horizon's calls accept somewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SvcPermission {
    /// No access.
    None,
    /// Read only.
    Read,
    /// Read and write.
    ReadWrite,
    /// Read and execute.
    ReadExecute,
}

impl SvcPermission {
    /// Every permission, in the order of their values.
    pub const ALL: [Self; 4] = [Self::None, Self::Read, Self::ReadWrite, Self::ReadExecute];

    /// The value a guest passes: read is bit 0, write bit 1, execute bit 2.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Read => MemoryPermission::READ,
            Self::ReadWrite => MemoryPermission::READ | MemoryPermission::WRITE,
            Self::ReadExecute => MemoryPermission::READ | MemoryPermission::EXECUTE,
        }
    }

    /// The permission with this value, if it is one of the four.
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|permission| u32::from(permission.bits()) == raw)
    }
}

bit_set! {
    /// The attributes of a run of pages (`KMemoryAttribute`), which a system call reports as
    /// they are (`UserMask` is every bit).
    MemoryAttribute
}

impl MemoryAttribute {
    /// No attribute.
    pub const NONE: Self = Self(0);
    /// Every bit: a test with this mask compares whole attribute sets.
    pub const ALL: Self = Self(u8::MAX);
    /// Locked by the kernel (`Locked`).
    pub const LOCKED: Self = Self(1 << 0);
    /// Locked for IPC: the block's IPC lock count is not zero.
    pub const IPC_LOCKED: Self = Self(1 << 1);
    /// Shared with a device: the block's device use count is not zero.
    pub const DEVICE_SHARED: Self = Self(1 << 2);
    /// Mapped uncached.
    pub const UNCACHED: Self = Self(1 << 3);
    /// Its permission can no longer be changed.
    pub const PERMISSION_LOCKED: Self = Self(1 << 4);
    /// What `SetMemoryAttribute` may change (`SetMask`).
    pub const SET_MASK: Self = Self(Self::UNCACHED.0 | Self::PERMISSION_LOCKED.0);
    /// The two attributes that only count locks and shares, which most checks ignore
    /// (`DefaultMemoryIgnoreAttr` in `eden/src/core/hle/kernel/k_page_table_base.h`, and
    /// `AttributeIgnoreMask` in `KMemoryBlock::HasProperties`).
    pub const COUNTED: Self = Self(Self::IPC_LOCKED.0 | Self::DEVICE_SHARED.0);

    /// The attributes a system call reports (`Svc::MemoryAttribute`).
    #[must_use]
    pub fn svc(self) -> u32 {
        u32::from(self.0)
    }
}

bit_set! {
    /// Why a block must not be merged with a neighbour (`KMemoryBlockDisableMergeAttribute`).
    DisableMerge
}

impl DisableMerge {
    /// No barrier.
    pub const NONE: Self = Self(0);
    /// The block starts a mapping (`Normal`).
    pub const NORMAL: Self = Self(1 << 0);
    /// A device mapping starts here (`DeviceLeft`).
    pub const DEVICE_LEFT: Self = Self(1 << 1);
    /// An IPC lock starts here (`IpcLeft`).
    pub const IPC_LEFT: Self = Self(1 << 2);
    /// Locked (`Locked`).
    pub const LOCKED: Self = Self(1 << 3);
    /// A device mapping ends here (`DeviceRight`).
    pub const DEVICE_RIGHT: Self = Self(1 << 4);
    /// The barriers against the block below (`AllLeft`).
    pub const ALL_LEFT: Self =
        Self(Self::NORMAL.0 | Self::DEVICE_LEFT.0 | Self::IPC_LEFT.0 | Self::LOCKED.0);
    /// The barriers against the block above (`AllRight`).
    pub const ALL_RIGHT: Self = Self(Self::DEVICE_RIGHT.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eden's `static_assert`s, copied from `k_memory_block.h` (see the file's header).
    const REFERENCE: &str = include_str!("reference/k_memory_state.txt");

    fn reference() -> Vec<(&'static str, u32)> {
        REFERENCE
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .map(|line| {
                let (name, value) = line.split_once(" = ").unwrap();
                let value = value.trim().strip_prefix("0x").unwrap();
                (name.trim(), u32::from_str_radix(value, 16).unwrap())
            })
            .collect()
    }

    #[test]
    fn every_state_word_matches_edens_static_asserts_in_order() {
        let ours: Vec<_> = MemoryState::ALL
            .iter()
            .map(|state| (state.name(), state.word()))
            .collect();
        assert_eq!(ours, reference());
    }

    #[test]
    fn the_reported_state_is_the_low_byte_of_the_word() {
        for &state in MemoryState::ALL {
            assert_eq!(state.svc(), state.word() & STATE_MASK, "{state:?}");
            assert_eq!(state.word() & !STATE_MASK & 0xff, 0);
        }
        // The two I/O states report the same number and differ in their flags only.
        assert_eq!(MemoryState::IoMemory.svc(), MemoryState::IoRegister.svc());
        assert_ne!(MemoryState::IoMemory.word(), MemoryState::IoRegister.word());
    }

    #[test]
    fn the_flags_the_memory_calls_test_are_on_the_states_horizon_allows() {
        use MemoryState::{AliasCode, AliasCodeData, Code, CodeData, Insecure, Normal};
        let with = |flag: u32| -> Vec<_> {
            MemoryState::ALL
                .iter()
                .copied()
                .filter(|state| state.word() & flag != 0)
                .collect()
        };
        assert_eq!(with(flag::CAN_REPROTECT), [CodeData, Normal, AliasCodeData]);
        assert_eq!(with(flag::CODE), [Code, AliasCode]);
        assert_eq!(with(flag::CAN_PERMISSION_LOCK), [CodeData, AliasCodeData]);
        assert_eq!(
            with(flag::CAN_CHANGE_ATTRIBUTE),
            [
                CodeData,
                Normal,
                AliasCodeData,
                MemoryState::Transferred,
                Insecure
            ]
        );
    }

    #[test]
    fn a_requested_permission_becomes_the_kernels_as_eden_converts_it() {
        // `ConvertToKMemoryPermission`, worked by hand for each value a call may pass: the
        // user's bits, KernelRead (0x08), KernelWrite (0x10) with write, NotMapped (0x40) for
        // none.
        let expected = [
            (SvcPermission::None, 0x48),
            (SvcPermission::Read, 0x09),
            (SvcPermission::ReadWrite, 0x1b),
            (SvcPermission::ReadExecute, 0x0d),
        ];
        for (requested, bits) in expected {
            let converted = MemoryPermission::from_svc(requested);
            assert_eq!(converted.bits(), bits, "{requested:?}");
            assert_eq!(converted.svc(), u32::from(requested.bits()));
        }
        assert_eq!(
            MemoryPermission::from_svc(SvcPermission::None),
            MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED
        );
        assert_eq!(
            MemoryPermission::from_svc(SvcPermission::Read),
            MemoryPermission::USER_READ
        );
        assert_eq!(
            MemoryPermission::from_svc(SvcPermission::ReadWrite),
            MemoryPermission::USER_READ_WRITE
        );
        assert_eq!(
            MemoryPermission::from_svc(SvcPermission::ReadExecute),
            MemoryPermission::USER_READ_EXECUTE
        );
    }

    #[test]
    fn only_the_four_permissions_have_a_raw_value() {
        for permission in SvcPermission::ALL {
            assert_eq!(
                SvcPermission::from_raw(u32::from(permission.bits())),
                Some(permission)
            );
        }
        // Write alone, execute alone, write and execute, all three, and `DontCare` (1 << 28).
        for raw in [2, 4, 6, 7, 8, 1 << 28, u32::MAX] {
            assert_eq!(SvcPermission::from_raw(raw), None, "{raw:#x}");
        }
    }

    #[test]
    fn guest_permissions_are_the_user_bits_and_write_needs_read() {
        let cases = [
            (MemoryPermission::NONE, Some(Perms::None)),
            (
                MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED,
                Some(Perms::None),
            ),
            (MemoryPermission::USER_READ, Some(Perms::Read)),
            (MemoryPermission::USER_READ_WRITE, Some(Perms::ReadWrite)),
            (MemoryPermission::USER_EXECUTE, Some(Perms::Execute)),
            (
                MemoryPermission::USER_READ_EXECUTE,
                Some(Perms::ReadExecute),
            ),
            (
                MemoryPermission::USER_READ_WRITE | MemoryPermission::USER_EXECUTE,
                Some(Perms::ReadWriteExecute),
            ),
            (MemoryPermission::USER_WRITE, None),
            (
                MemoryPermission::USER_WRITE | MemoryPermission::USER_EXECUTE,
                None,
            ),
        ];
        for (permission, perms) in cases {
            assert_eq!(permission.guest(), perms, "{permission:?}");
        }
    }

    #[test]
    fn the_bit_sets_have_the_references_values() {
        // `Svc::MemoryAttribute` in svc_types.h: Locked, IpcLocked, DeviceShared, Uncached,
        // PermissionLocked are bits 0 to 4.
        let attributes = [
            MemoryAttribute::LOCKED,
            MemoryAttribute::IPC_LOCKED,
            MemoryAttribute::DEVICE_SHARED,
            MemoryAttribute::UNCACHED,
            MemoryAttribute::PERMISSION_LOCKED,
        ];
        for (bit, attribute) in (0..).zip(attributes) {
            assert_eq!(attribute.bits(), 1 << bit);
            assert_eq!(attribute.svc(), 1 << bit);
        }
        assert_eq!(MemoryAttribute::SET_MASK.bits(), 0x18);
        assert_eq!(MemoryAttribute::COUNTED.bits(), 0x06);
        // `KMemoryBlockDisableMergeAttribute`: Normal, DeviceLeft, IpcLeft, Locked, DeviceRight
        // are bits 0 to 4; the first four face left.
        assert_eq!(DisableMerge::ALL_LEFT.bits(), 0x0f);
        assert_eq!(DisableMerge::ALL_RIGHT.bits(), 0x10);
        assert_eq!(MemoryPermission::IPC_LOCK_CHANGE_MASK.bits(), 0x5b);
        assert_eq!(MemoryPermission::USER_MASK.bits(), 0x07);
    }

    #[test]
    fn set_operations_behave_as_sets() {
        let both = MemoryAttribute::UNCACHED | MemoryAttribute::LOCKED;
        assert!(both.contains(MemoryAttribute::UNCACHED));
        assert!(!both.contains(MemoryAttribute::SET_MASK));
        assert!(both.excludes(MemoryAttribute::IPC_LOCKED));
        assert!(!both.excludes(MemoryAttribute::SET_MASK));
        assert_eq!(both & MemoryAttribute::SET_MASK, MemoryAttribute::UNCACHED);
        assert_eq!((!both).bits(), !both.bits());
        assert_eq!(MemoryAttribute::from_bits(0x09), both);
    }
}
