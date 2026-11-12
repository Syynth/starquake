//! Exception syndromes: the detail an [`Exit`](crate::Exit) carries about the exception the guest
//! took, so the kernel can tell a missing page from a write to a read-only one, and a crash
//! report can say what happened.
//!
//! A [`Syndrome`] is the value `ESR_EL1` holds for the exception, in the architecture's own
//! layout. The native backends read it from the vCPU; a JIT backend builds one with the
//! constructors here, so every backend reports the same values.
//!
//! Source: Arm Architecture Reference Manual for A-profile architecture (Arm DDI 0487), the
//! description of `ESR_ELx`: the exception class field `EC` (bits 31:26), the instruction length
//! bit `IL` (bit 25), and the instruction-specific syndrome `ISS` (bits 24:0) with its encodings
//! for Data Aborts and Instruction Aborts (the fault status code in bits 5:0, `WnR` in bit 6).
//! Sections are cited by name because the numbering moves between issues of the manual.
//!
//! What a real vCPU has reported (the hardware tests in `tests/hvf.rs` and
//! `src/hvf/hardware.rs`, and `tools/hvprobe/README.md`): classes 0x00 (an undefined
//! instruction), 0x15 (`svc`), 0x18 (a trapped system register access), 0x1c (a failed pointer
//! authentication), 0x20, 0x24 and 0x3c (`brk`); translation faults at levels 0 to 3 and
//! permission faults at level 3. For those aborts the whole syndrome was exactly what
//! [`Syndrome::data_abort`] and [`Syndrome::instruction_abort`] build: no other `ISS` bit was
//! set. The other encodings below are taken from the manual and have not been seen on
//! hardware: classes 0x22 and 0x26, and address size, access flag and alignment faults.

/// `ESR_ELx.EC` starts at this bit.
const EC_SHIFT: u32 = 26;
/// `ESR_ELx.EC` is six bits wide.
const EC_MASK: u64 = 0x3f;
/// `ESR_ELx.IL` (bit 25): set for a 32-bit instruction, and for every class whose syndrome does
/// not describe an instruction.
const IL: u64 = 1 << 25;
/// `ISS.WnR` (bit 6) of a Data Abort: the access was a write.
const ISS_WNR: u64 = 1 << 6;
/// The fault status code of an abort: `ISS` bits 5:0 (`DFSC` for data, `IFSC` for instructions).
const FSC_MASK: u64 = 0x3f;
/// The two low bits of a fault status code that has a level hold the level.
const FSC_LEVEL_MASK: u8 = 0b11;
/// Fault status code 0b100001: an alignment fault. It has no level.
const FSC_ALIGNMENT: u8 = 0b10_0001;

/// Why the guest took an exception: `ESR_ELx.EC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExceptionClass {
    /// 0x00, "unknown reason": an instruction that is undefined at EL0.
    Unknown,
    /// 0x15: `svc` in AArch64 state.
    Svc,
    /// 0x18: a trapped `msr`, `mrs` or system instruction.
    TrappedSystemAccess,
    /// 0x1c: a pointer authentication failure (an `aut*` of a pointer with a wrong signature).
    PointerAuthentication,
    /// 0x20: an Instruction Abort from a lower exception level: a fetch that did not translate
    /// or was not permitted.
    InstructionAbort,
    /// 0x22: a misaligned program counter.
    PcAlignment,
    /// 0x24: a Data Abort from a lower exception level: a load or store that did not translate
    /// or was not permitted.
    DataAbort,
    /// 0x26: a misaligned stack pointer.
    SpAlignment,
    /// 0x3c: `brk` in AArch64 state.
    Breakpoint,
    /// Any other class, by its six-bit value.
    Other(u8),
}

impl ExceptionClass {
    /// The class with the six-bit value `ec`; higher bits are ignored.
    #[must_use]
    pub const fn from_ec(ec: u8) -> Self {
        match ec & 0x3f {
            0x00 => Self::Unknown,
            0x15 => Self::Svc,
            0x18 => Self::TrappedSystemAccess,
            0x1c => Self::PointerAuthentication,
            0x20 => Self::InstructionAbort,
            0x22 => Self::PcAlignment,
            0x24 => Self::DataAbort,
            0x26 => Self::SpAlignment,
            0x3c => Self::Breakpoint,
            other => Self::Other(other),
        }
    }

    /// The six-bit value of the class.
    #[must_use]
    pub const fn ec(self) -> u8 {
        match self {
            Self::Unknown => 0x00,
            Self::Svc => 0x15,
            Self::TrappedSystemAccess => 0x18,
            Self::PointerAuthentication => 0x1c,
            Self::InstructionAbort => 0x20,
            Self::PcAlignment => 0x22,
            Self::DataAbort => 0x24,
            Self::SpAlignment => 0x26,
            Self::Breakpoint => 0x3c,
            Self::Other(ec) => ec & 0x3f,
        }
    }
}

/// Why an access aborted: the fault status code of a Data Abort or an Instruction Abort.
///
/// `level` is the translation table level the fault was found at, 0 to 3. With the tables
/// [`starquake_memory::stage1::Stage1Tables`] builds the walk starts at level 1 and pages are at
/// level 3; a translation fault at level 0 is an address in neither half of the address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultStatus {
    /// Codes 0b0000LL: an address outside what the translation regime can address.
    AddressSize {
        /// The level.
        level: u8,
    },
    /// Codes 0b0001LL: no valid descriptor (the address is not mapped).
    Translation {
        /// The level of the invalid descriptor.
        level: u8,
    },
    /// Codes 0b0010LL: a descriptor with the access flag clear.
    AccessFlag {
        /// The level of the descriptor.
        level: u8,
    },
    /// Codes 0b0011LL: the address is mapped, but not for this access.
    Permission {
        /// The level of the descriptor.
        level: u8,
    },
    /// Code 0b100001: a misaligned access.
    Alignment,
    /// Any other code, by its six-bit value.
    Other(u8),
}

impl FaultStatus {
    /// The status with the six-bit code `code`; higher bits are ignored.
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        let code = code & 0x3f;
        let level = code & FSC_LEVEL_MASK;
        match code >> 2 {
            0b0000 => Self::AddressSize { level },
            0b0001 => Self::Translation { level },
            0b0010 => Self::AccessFlag { level },
            0b0011 => Self::Permission { level },
            _ if code == FSC_ALIGNMENT => Self::Alignment,
            _ => Self::Other(code),
        }
    }

    /// The six-bit code. A level above 3 keeps its two low bits.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::AddressSize { level } => level & FSC_LEVEL_MASK,
            Self::Translation { level } => 0b0001 << 2 | (level & FSC_LEVEL_MASK),
            Self::AccessFlag { level } => 0b0010 << 2 | (level & FSC_LEVEL_MASK),
            Self::Permission { level } => 0b0011 << 2 | (level & FSC_LEVEL_MASK),
            Self::Alignment => FSC_ALIGNMENT,
            Self::Other(code) => code & 0x3f,
        }
    }
}

/// The syndrome of an exception the guest took at EL0: what `ESR_EL1` holds for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Syndrome(u64);

impl Syndrome {
    /// The syndrome `ESR_EL1` holds as `esr`.
    #[must_use]
    pub const fn from_esr(esr: u64) -> Self {
        Self(esr)
    }

    /// The syndrome of a Data Abort from EL0 with `status`, for a store if `write` and a load
    /// otherwise. The other `ISS` fields are zero.
    #[must_use]
    pub const fn data_abort(status: FaultStatus, write: bool) -> Self {
        let wnr = if write { ISS_WNR } else { 0 };
        Self::with_class(ExceptionClass::DataAbort, wnr | status.code() as u64)
    }

    /// The syndrome of an Instruction Abort from EL0 with `status`. The other `ISS` fields are
    /// zero.
    #[must_use]
    pub const fn instruction_abort(status: FaultStatus) -> Self {
        Self::with_class(ExceptionClass::InstructionAbort, status.code() as u64)
    }

    /// The syndrome of an undefined instruction: class 0x00 with an empty `ISS`.
    #[must_use]
    pub const fn undefined() -> Self {
        Self::with_class(ExceptionClass::Unknown, 0)
    }

    const fn with_class(class: ExceptionClass, iss: u64) -> Self {
        Self(((class.ec() as u64) << EC_SHIFT) | IL | iss)
    }

    /// The value as `ESR_EL1` holds it.
    #[must_use]
    pub const fn esr(self) -> u64 {
        self.0
    }

    /// The exception class.
    #[must_use]
    pub const fn class(self) -> ExceptionClass {
        // Going through bytes avoids a truncating cast; the mask leaves six bits.
        let [ec, ..] = ((self.0 >> EC_SHIFT) & EC_MASK).to_le_bytes();
        ExceptionClass::from_ec(ec)
    }

    /// Why the access aborted, for a Data Abort or an Instruction Abort; `None` for every other
    /// class.
    #[must_use]
    pub const fn fault(self) -> Option<FaultStatus> {
        match self.class() {
            ExceptionClass::DataAbort | ExceptionClass::InstructionAbort => {
                let [code, ..] = (self.0 & FSC_MASK).to_le_bytes();
                Some(FaultStatus::from_code(code))
            }
            _ => None,
        }
    }

    /// Whether a Data Abort was for a store (`ISS.WnR`). `false` for every other class.
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(self.class(), ExceptionClass::DataAbort) && self.0 & ISS_WNR != 0
    }

    /// The immediate of the `svc` or `brk` that raised the exception (`ISS` bits 15:0); `None`
    /// for every other class.
    #[must_use]
    pub const fn immediate(self) -> Option<u16> {
        match self.class() {
            ExceptionClass::Svc | ExceptionClass::Breakpoint => {
                let [lo, hi, ..] = self.0.to_le_bytes();
                Some(u16::from_le_bytes([lo, hi]))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values are written from the Arm DDI 0487 bit positions, not from the constants
    // above, so a wrong constant fails here.

    #[test]
    fn every_class_value_round_trips() {
        for ec in 0..=0x3fu8 {
            let class = ExceptionClass::from_ec(ec);
            assert_eq!(class.ec(), ec, "{class:?}");
            assert_eq!(ExceptionClass::from_ec(class.ec()), class);
        }
        // Bits above the field are ignored on the way in and never produced on the way out.
        assert_eq!(ExceptionClass::from_ec(0x40 | 0x15), ExceptionClass::Svc);
        assert_eq!(ExceptionClass::Other(0xff).ec(), 0x3f);
    }

    #[test]
    fn named_classes_have_the_architecture_values() {
        let named = [
            (ExceptionClass::Unknown, 0x00),
            (ExceptionClass::Svc, 0x15),
            (ExceptionClass::TrappedSystemAccess, 0x18),
            (ExceptionClass::PointerAuthentication, 0x1c),
            (ExceptionClass::InstructionAbort, 0x20),
            (ExceptionClass::PcAlignment, 0x22),
            (ExceptionClass::DataAbort, 0x24),
            (ExceptionClass::SpAlignment, 0x26),
            (ExceptionClass::Breakpoint, 0x3c),
        ];
        for (class, ec) in named {
            assert_eq!(class.ec(), ec, "{class:?}");
            assert_eq!(ExceptionClass::from_ec(ec), class);
        }
        // A value with no name of its own keeps its value.
        assert_eq!(ExceptionClass::from_ec(0x16), ExceptionClass::Other(0x16));
    }

    #[test]
    fn every_fault_status_code_round_trips() {
        for code in 0..=0x3fu8 {
            let status = FaultStatus::from_code(code);
            assert_eq!(status.code(), code, "{status:?}");
            assert_eq!(FaultStatus::from_code(status.code()), status);
        }
    }

    #[test]
    fn fault_status_codes_have_the_architecture_values() {
        for level in 0..=3u8 {
            assert_eq!(
                FaultStatus::from_code(level),
                FaultStatus::AddressSize { level }
            );
            assert_eq!(
                FaultStatus::from_code(0b00_0100 | level),
                FaultStatus::Translation { level }
            );
            assert_eq!(
                FaultStatus::from_code(0b00_1000 | level),
                FaultStatus::AccessFlag { level }
            );
            assert_eq!(
                FaultStatus::from_code(0b00_1100 | level),
                FaultStatus::Permission { level }
            );
        }
        assert_eq!(FaultStatus::from_code(0x21), FaultStatus::Alignment);
        // The codes on either side of the alignment fault, and the first code above the
        // levelled groups, have no name here.
        for other in [0x10, 0x20, 0x22] {
            assert_eq!(FaultStatus::from_code(other), FaultStatus::Other(other));
        }
        // A level that does not fit keeps its two low bits.
        assert_eq!(FaultStatus::Permission { level: 7 }.code(), 0b00_1111);
    }

    #[test]
    fn built_syndromes_have_the_architecture_layout() {
        // EC 0x24 at bits 31:26, IL at bit 25, WnR at bit 6, DFSC 0b001111 (permission, level 3).
        let store = Syndrome::data_abort(FaultStatus::Permission { level: 3 }, true);
        assert_eq!(store.esr(), 0x9200_004f);
        assert_eq!(store.class(), ExceptionClass::DataAbort);
        assert_eq!(store.fault(), Some(FaultStatus::Permission { level: 3 }));
        assert!(store.is_write());
        let load = Syndrome::data_abort(FaultStatus::Translation { level: 1 }, false);
        assert_eq!(load.esr(), 0x9200_0005);
        assert!(!load.is_write());
        // EC 0x20, IL, IFSC 0b000111 (translation, level 3).
        let fetch = Syndrome::instruction_abort(FaultStatus::Translation { level: 3 });
        assert_eq!(fetch.esr(), 0x8200_0007);
        assert_eq!(fetch.class(), ExceptionClass::InstructionAbort);
        assert_eq!(fetch.fault(), Some(FaultStatus::Translation { level: 3 }));
        // Bit 6 of an Instruction Abort is not WnR.
        assert!(!Syndrome::from_esr(fetch.esr() | 0x40).is_write());
        // What a real vCPU reported for an undefined instruction (`tools/hvprobe/README.md`).
        assert_eq!(Syndrome::undefined().esr(), 0x0200_0000);
        assert_eq!(Syndrome::undefined().class(), ExceptionClass::Unknown);
        assert_eq!(Syndrome::undefined().fault(), None);
    }

    #[test]
    fn immediates_come_from_svc_and_brk_only() {
        // `svc #0x27` and `brk #0x123`: EC, IL, and the immediate in the low sixteen bits.
        let svc = Syndrome::from_esr((0x15 << 26) | (1 << 25) | 0x27);
        let brk = Syndrome::from_esr((0x3c << 26) | (1 << 25) | 0x123);
        assert_eq!(svc.immediate(), Some(0x27));
        assert_eq!(brk.immediate(), Some(0x123));
        assert_eq!(brk.class(), ExceptionClass::Breakpoint);
        let abort = Syndrome::data_abort(FaultStatus::Translation { level: 3 }, false);
        assert_eq!(abort.immediate(), None);
        assert_eq!(svc.fault(), None);
        assert!(!svc.is_write());
    }
}
