//! A64 instruction encoders for guest test programs.
//!
//! Each encoder returns the 32-bit instruction word, or an [`EncodeError`] when an operand does
//! not fit its field: a register number above [`MAX_REG`], an immediate above its field's maximum,
//! a branch distance outside the instruction's range or not a multiple of [`INSN_SIZE`]. An
//! out-of-range operand is never truncated into a different instruction.
//!
//! Only 64-bit (`X`) forms are encoded, which is all the guest test programs need. Register
//! number 31 means `XZR` or `SP` according to the instruction, as in the Arm encoding: `SP` for
//! the base register of [`ldr`] and [`str`] and for both registers of [`add_imm`] and
//! [`sub_imm`], `XZR` everywhere else. [`Assembler`] adds labels for branches.
//!
//! The encodings follow the Arm Architecture Reference Manual (A64 instruction set) and are
//! pinned by test constants, each with its assembly in a comment, that were checked once against
//! a real assembler (`clang -target arm64 -c`, disassembled with `llvm-objdump`). No source under
//! `docs/references.md` encodes instructions; the one word it quotes, `svc #0x7`
//! (`eden/src/core/loader/nro.cpp`, `kSvcExitProcessInstruction`), is also among the tests.

mod assembler;

pub use assembler::{AsmError, Assembler, Label};

/// Size of one instruction in bytes; branch offsets are multiples of it.
pub const INSN_SIZE: u32 = 4;
/// Highest register number (31 is `XZR` or `SP`, see the module docs).
pub const MAX_REG: u8 = 31;
/// Largest immediate of `movz`, `movk`, `svc` and `brk` (16 bits).
pub const MAX_IMM16: u32 = 0xFFFF;
/// Largest immediate of `add` and `sub` (12 bits, unshifted).
pub const MAX_IMM12: u32 = 0xFFF;
/// Alignment of `ldr` and `str` byte offsets for 64-bit accesses.
pub const MEM_OFFSET_ALIGN: u32 = 8;
/// Largest byte offset of `ldr` and `str` (unsigned 12-bit immediate scaled by 8).
pub const MAX_MEM_OFFSET: u32 = MAX_IMM12 * MEM_OFFSET_ALIGN;
/// Number of bits in the word offset of `b` and `bl`.
pub const BRANCH_IMM_BITS: u32 = 26;
/// Number of bits in the word offset of `cbz` and `cbnz`.
pub const COMPARE_BRANCH_IMM_BITS: u32 = 19;

/// Why an operand could not be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    /// A register number above [`MAX_REG`].
    #[error("register number {reg} is not an A64 general register")]
    Register {
        /// The register number given.
        reg: u8,
    },
    /// An immediate larger than its field holds.
    #[error("immediate {value:#x} is larger than the field's maximum {max:#x}")]
    Immediate {
        /// The immediate given.
        value: u32,
        /// The largest value the field holds.
        max: u32,
    },
    /// A `movz`/`movk` shift other than 0, 16, 32 or 48.
    #[error("shift {shift} is not 0, 16, 32 or 48")]
    Shift {
        /// The shift given, in bits.
        shift: u32,
    },
    /// A byte offset that is not a multiple of the instruction's alignment.
    #[error("offset {offset} is not a multiple of {align}")]
    Misaligned {
        /// The offset given, in bytes.
        offset: i64,
        /// The required alignment, in bytes.
        align: u32,
    },
    /// A byte offset outside the instruction's range.
    #[error("offset {offset} is outside {min}..={max}")]
    Offset {
        /// The offset given, in bytes.
        offset: i64,
        /// The lowest encodable offset.
        min: i64,
        /// The highest encodable offset.
        max: i64,
    },
}

/// The system registers [`mrs`] can read: the counter and the thread pointers guest programs use.
///
/// Each variant is the `(op0, op1, CRn, CRm, op2)` of the Arm system register; `op0` is 3 for all
/// of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysReg {
    /// `CNTFRQ_EL0`, the counter frequency.
    CntFrqEl0,
    /// `CNTPCT_EL0`, the physical counter.
    CntPctEl0,
    /// `CNTVCT_EL0`, the virtual counter.
    CntVctEl0,
    /// `TPIDR_EL0`, the read/write thread pointer.
    TpidrEl0,
    /// `TPIDRRO_EL0`, the read-only thread pointer.
    TpidrroEl0,
}

impl SysReg {
    /// The `o0:op1:CRn:CRm:op2` field of `mrs` (15 bits, instruction bits 19 down to 5), shifted
    /// down to bit 0. `o0` is `op0 - 2`, which is 1 for every variant.
    const fn encoding(self) -> u32 {
        let (op1, crn, crm, op2) = match self {
            Self::CntFrqEl0 => (3, 14, 0, 0),
            Self::CntPctEl0 => (3, 14, 0, 1),
            Self::CntVctEl0 => (3, 14, 0, 2),
            Self::TpidrEl0 => (3, 13, 0, 2),
            Self::TpidrroEl0 => (3, 13, 0, 3),
        };
        (1 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
    }
}

/// Checks a register number.
fn reg(reg: u8) -> Result<u32, EncodeError> {
    if reg > MAX_REG {
        return Err(EncodeError::Register { reg });
    }
    Ok(u32::from(reg))
}

/// Checks that `value` fits a field whose largest value is `max`.
fn imm(value: u32, max: u32) -> Result<u32, EncodeError> {
    if value > max {
        return Err(EncodeError::Immediate { value, max });
    }
    Ok(value)
}

/// The two's-complement word offset of a branch of `offset` bytes, in a field of `bits` bits.
fn branch_field(offset: i64, bits: u32) -> Result<u32, EncodeError> {
    // `bits` signed bits of whole words span [-2^(bits+1), 2^(bits+1) - 4] bytes.
    let reach = 1i64 << (bits + 1);
    let insn = i64::from(INSN_SIZE);
    let (min, max) = (-reach, reach - insn);
    if offset < min || offset > max {
        return Err(EncodeError::Offset { offset, min, max });
    }
    if offset % insn != 0 {
        return Err(EncodeError::Misaligned {
            offset,
            align: INSN_SIZE,
        });
    }
    let mask = (1i64 << bits) - 1;
    u32::try_from((offset >> 2) & mask).map_err(|_| EncodeError::Offset { offset, min, max })
}

/// `movz xd, #imm16, lsl #shift`.
pub fn movz(rd: u8, imm16: u32, shift: u32) -> Result<u32, EncodeError> {
    move_wide(0xD280_0000, rd, imm16, shift)
}

/// `movk xd, #imm16, lsl #shift`.
pub fn movk(rd: u8, imm16: u32, shift: u32) -> Result<u32, EncodeError> {
    move_wide(0xF280_0000, rd, imm16, shift)
}

/// Shared by [`movz`] and [`movk`]: `hw` (the shift in 16-bit halfwords) sits at bit 21.
fn move_wide(base: u32, rd: u8, imm16: u32, shift: u32) -> Result<u32, EncodeError> {
    let rd = reg(rd)?;
    let imm16 = imm(imm16, MAX_IMM16)?;
    let hw = match shift {
        0 => 0,
        16 => 1,
        32 => 2,
        48 => 3,
        _ => return Err(EncodeError::Shift { shift }),
    };
    Ok(base | (hw << 21) | (imm16 << 5) | rd)
}

/// `mov xd, xm`, which is `orr xd, xzr, xm`.
pub fn mov(rd: u8, rm: u8) -> Result<u32, EncodeError> {
    Ok(0xAA00_03E0 | (reg(rm)? << 16) | reg(rd)?)
}

/// `add xd, xn, #imm12`. Register 31 is `SP` for both operands.
pub fn add_imm(rd: u8, rn: u8, imm12: u32) -> Result<u32, EncodeError> {
    arith_imm(0x9100_0000, rd, rn, imm12)
}

/// `sub xd, xn, #imm12`. Register 31 is `SP` for both operands.
pub fn sub_imm(rd: u8, rn: u8, imm12: u32) -> Result<u32, EncodeError> {
    arith_imm(0xD100_0000, rd, rn, imm12)
}

/// Shared by [`add_imm`] and [`sub_imm`].
fn arith_imm(base: u32, rd: u8, rn: u8, imm12: u32) -> Result<u32, EncodeError> {
    Ok(base | (imm(imm12, MAX_IMM12)? << 10) | (reg(rn)? << 5) | reg(rd)?)
}

/// `ldr xt, [xn, #offset]` (unsigned offset). `offset` is in bytes, a multiple of
/// [`MEM_OFFSET_ALIGN`] up to [`MAX_MEM_OFFSET`]. Register 31 as `xn` is `SP`.
pub fn ldr(rt: u8, rn: u8, offset: u32) -> Result<u32, EncodeError> {
    mem_imm(0xF940_0000, rt, rn, offset)
}

/// `str xt, [xn, #offset]` (unsigned offset); operands as for [`ldr`].
pub fn str(rt: u8, rn: u8, offset: u32) -> Result<u32, EncodeError> {
    mem_imm(0xF900_0000, rt, rn, offset)
}

/// Shared by [`ldr`] and [`str`]: the byte offset is stored divided by 8, at bit 10.
fn mem_imm(base: u32, rt: u8, rn: u8, offset: u32) -> Result<u32, EncodeError> {
    let (rt, rn) = (reg(rt)?, reg(rn)?);
    let offset = imm(offset, MAX_MEM_OFFSET)?;
    if offset % MEM_OFFSET_ALIGN != 0 {
        return Err(EncodeError::Misaligned {
            offset: i64::from(offset),
            align: MEM_OFFSET_ALIGN,
        });
    }
    Ok(base | ((offset / MEM_OFFSET_ALIGN) << 10) | (rn << 5) | rt)
}

/// `b .+offset`, with `offset` in bytes from this instruction.
pub fn b(offset: i64) -> Result<u32, EncodeError> {
    Ok(0x1400_0000 | branch_field(offset, BRANCH_IMM_BITS)?)
}

/// `bl .+offset`, with `offset` in bytes from this instruction.
pub fn bl(offset: i64) -> Result<u32, EncodeError> {
    Ok(0x9400_0000 | branch_field(offset, BRANCH_IMM_BITS)?)
}

/// `cbz xt, .+offset`, with `offset` in bytes from this instruction.
pub fn cbz(rt: u8, offset: i64) -> Result<u32, EncodeError> {
    Ok(0xB400_0000 | (branch_field(offset, COMPARE_BRANCH_IMM_BITS)? << 5) | reg(rt)?)
}

/// `cbnz xt, .+offset`, with `offset` in bytes from this instruction.
pub fn cbnz(rt: u8, offset: i64) -> Result<u32, EncodeError> {
    Ok(0xB500_0000 | (branch_field(offset, COMPARE_BRANCH_IMM_BITS)? << 5) | reg(rt)?)
}

/// `ret` (to `x30`).
pub const fn ret() -> u32 {
    0xD65F_03C0
}

/// `svc #imm16`.
pub fn svc(imm16: u32) -> Result<u32, EncodeError> {
    Ok(0xD400_0001 | (imm(imm16, MAX_IMM16)? << 5))
}

/// `brk #imm16`.
pub fn brk(imm16: u32) -> Result<u32, EncodeError> {
    Ok(0xD420_0000 | (imm(imm16, MAX_IMM16)? << 5))
}

/// `nop`.
pub const fn nop() -> u32 {
    0xD503_201F
}

/// `mrs xt, <sysreg>`.
pub fn mrs(rt: u8, sysreg: SysReg) -> Result<u32, EncodeError> {
    Ok(0xD530_0000 | (sysreg.encoding() << 5) | reg(rt)?)
}

/// The little-endian bytes of `words`, as they sit in guest memory.
pub fn to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_A640;

    /// Furthest branch distances in bytes (lowest, highest) for a field of `bits` word-offset
    /// bits.
    const fn reach(bits: u32) -> (i64, i64) {
        (-(1 << (bits + 1)), (1 << (bits + 1)) - 4)
    }

    /// Every expected word below was produced by `clang -target arm64-apple-macos -c` and read
    /// back with `llvm-objdump -d` (Apple clang from Xcode), once, when the encoders were
    /// written. The assembly is the first element of each case.
    type Case = (&'static str, Result<u32, EncodeError>, u32);

    fn check(cases: &[Case]) {
        for (asm, got, want) in cases {
            assert_eq!(got.as_ref(), Ok(want), "{asm}");
        }
    }

    #[test]
    fn move_words_match_a_real_assembler() {
        check(&[
            ("movz x0, #1", movz(0, 1, 0), 0xD280_0020),
            ("movz x1, #0xffff", movz(1, MAX_IMM16, 0), 0xD29F_FFE1),
            (
                "movz x2, #0xabcd, lsl #16",
                movz(2, 0xABCD, 16),
                0xD2B5_79A2,
            ),
            (
                "movz x3, #0x1234, lsl #32",
                movz(3, 0x1234, 32),
                0xD2C2_4683,
            ),
            (
                "movz x30, #0x5678, lsl #48",
                movz(30, 0x5678, 48),
                0xD2EA_CF1E,
            ),
            ("movz xzr, #7", movz(31, 7, 0), 0xD280_00FF),
            ("movk x0, #1", movk(0, 1, 0), 0xF280_0020),
            (
                "movk x4, #0xffff, lsl #16",
                movk(4, MAX_IMM16, 16),
                0xF2BF_FFE4,
            ),
            (
                "movk x5, #0x9abc, lsl #32",
                movk(5, 0x9ABC, 32),
                0xF2D3_5785,
            ),
            (
                "movk x6, #0xdef0, lsl #48",
                movk(6, 0xDEF0, 48),
                0xF2FB_DE06,
            ),
            ("mov x0, x1", mov(0, 1), 0xAA01_03E0),
            ("mov x29, x30", mov(29, 30), 0xAA1E_03FD),
            ("mov x0, xzr", mov(0, 31), 0xAA1F_03E0),
            ("mov x30, x0", mov(30, 0), 0xAA00_03FE),
        ]);
    }

    #[test]
    fn arithmetic_and_memory_words_match_a_real_assembler() {
        check(&[
            ("add x0, x1, #1", add_imm(0, 1, 1), 0x9100_0420),
            ("add x2, sp, #0xfff", add_imm(2, 31, MAX_IMM12), 0x913F_FFE2),
            ("add sp, sp, #0x10", add_imm(31, 31, 0x10), 0x9100_43FF),
            ("add x30, x29, #0", add_imm(30, 29, 0), 0x9100_03BE),
            ("sub x0, x1, #1", sub_imm(0, 1, 1), 0xD100_0420),
            ("sub x2, sp, #0xfff", sub_imm(2, 31, MAX_IMM12), 0xD13F_FFE2),
            ("sub sp, sp, #0x10", sub_imm(31, 31, 0x10), 0xD100_43FF),
            ("sub x30, x29, #0", sub_imm(30, 29, 0), 0xD100_03BE),
            ("ldr x0, [x1]", ldr(0, 1, 0), 0xF940_0020),
            ("ldr x2, [x3, #8]", ldr(2, 3, 8), 0xF940_0462),
            (
                "ldr x30, [sp, #0x7ff8]",
                ldr(30, 31, MAX_MEM_OFFSET),
                0xF97F_FFFE,
            ),
            ("ldr x4, [x5, #0x1230]", ldr(4, 5, 0x1230), 0xF949_18A4),
            ("str x0, [x1]", str(0, 1, 0), 0xF900_0020),
            ("str x2, [x3, #8]", str(2, 3, 8), 0xF900_0462),
            (
                "str x30, [sp, #0x7ff8]",
                str(30, 31, MAX_MEM_OFFSET),
                0xF93F_FFFE,
            ),
            ("str xzr, [x5, #0x1230]", str(31, 5, 0x1230), 0xF909_18BF),
        ]);
    }

    #[test]
    fn branch_words_match_a_real_assembler() {
        check(&[
            ("b .", b(0), 0x1400_0000),
            ("b .+4", b(4), 0x1400_0001),
            ("b .-4", b(-4), 0x17FF_FFFF),
            ("b .+0x7fffffc", b(reach(BRANCH_IMM_BITS).1), 0x15FF_FFFF),
            ("b .-0x8000000", b(reach(BRANCH_IMM_BITS).0), 0x1600_0000),
            ("bl .", bl(0), 0x9400_0000),
            ("bl .+8", bl(8), 0x9400_0002),
            ("bl .-8", bl(-8), 0x97FF_FFFE),
            ("bl .+0x7fffffc", bl(reach(BRANCH_IMM_BITS).1), 0x95FF_FFFF),
            ("bl .-0x8000000", bl(reach(BRANCH_IMM_BITS).0), 0x9600_0000),
            ("cbz x0, .", cbz(0, 0), 0xB400_0000),
            ("cbz x1, .+8", cbz(1, 8), 0xB400_0041),
            ("cbz x30, .-4", cbz(30, -4), 0xB4FF_FFFE),
            (
                "cbz x2, .+0xffffc",
                cbz(2, reach(COMPARE_BRANCH_IMM_BITS).1),
                0xB47F_FFE2,
            ),
            (
                "cbz x3, .-0x100000",
                cbz(3, reach(COMPARE_BRANCH_IMM_BITS).0),
                0xB480_0003,
            ),
            ("cbnz x0, .", cbnz(0, 0), 0xB500_0000),
            ("cbnz x1, .+8", cbnz(1, 8), 0xB500_0041),
            ("cbnz x30, .-4", cbnz(30, -4), 0xB5FF_FFFE),
            (
                "cbnz x2, .+0xffffc",
                cbnz(2, reach(COMPARE_BRANCH_IMM_BITS).1),
                0xB57F_FFE2,
            ),
            (
                "cbnz x3, .-0x100000",
                cbnz(3, reach(COMPARE_BRANCH_IMM_BITS).0),
                0xB580_0003,
            ),
        ]);
    }

    #[test]
    fn system_words_match_a_real_assembler() {
        check(&[
            ("ret", Ok(ret()), 0xD65F_03C0),
            ("svc #0", svc(0), 0xD400_0001),
            ("svc #0xffff", svc(MAX_IMM16), 0xD41F_FFE1),
            // The word Eden's loader scans for (`kSvcExitProcessInstruction`).
            ("svc #7", svc(7), 0xD400_00E1),
            ("brk #0", brk(0), 0xD420_0000),
            ("brk #0xffff", brk(MAX_IMM16), 0xD43F_FFE0),
            ("brk #0x1234", brk(0x1234), 0xD422_4680),
            ("nop", Ok(nop()), 0xD503_201F),
            ("mrs x0, cntfrq_el0", mrs(0, SysReg::CntFrqEl0), 0xD53B_E000),
            ("mrs x1, cntpct_el0", mrs(1, SysReg::CntPctEl0), 0xD53B_E021),
            ("mrs x2, cntvct_el0", mrs(2, SysReg::CntVctEl0), 0xD53B_E042),
            ("mrs x3, tpidr_el0", mrs(3, SysReg::TpidrEl0), 0xD53B_D043),
            (
                "mrs x30, tpidrro_el0",
                mrs(30, SysReg::TpidrroEl0),
                0xD53B_D07E,
            ),
        ]);
    }

    #[test]
    fn register_numbers_above_the_maximum_are_errors() {
        for bad in [MAX_REG + 1, u8::MAX] {
            let err = Err(EncodeError::Register { reg: bad });
            assert_eq!(movz(bad, 0, 0), err);
            assert_eq!(movk(bad, 0, 0), err);
            assert_eq!(mov(bad, 0), err);
            assert_eq!(mov(0, bad), err);
            assert_eq!(add_imm(bad, 0, 0), err);
            assert_eq!(add_imm(0, bad, 0), err);
            assert_eq!(sub_imm(bad, 0, 0), err);
            assert_eq!(sub_imm(0, bad, 0), err);
            assert_eq!(ldr(bad, 0, 0), err);
            assert_eq!(ldr(0, bad, 0), err);
            assert_eq!(str(bad, 0, 0), err);
            assert_eq!(str(0, bad, 0), err);
            assert_eq!(cbz(bad, 0), err);
            assert_eq!(cbnz(bad, 0), err);
            assert_eq!(mrs(bad, SysReg::CntVctEl0), err);
        }
    }

    #[test]
    fn immediates_above_the_field_are_errors() {
        let over16 = |value| {
            Err(EncodeError::Immediate {
                value,
                max: MAX_IMM16,
            })
        };
        let over12 = |value| {
            Err(EncodeError::Immediate {
                value,
                max: MAX_IMM12,
            })
        };
        assert_eq!(movz(0, MAX_IMM16 + 1, 0), over16(MAX_IMM16 + 1));
        assert_eq!(movk(0, u32::MAX, 0), over16(u32::MAX));
        assert_eq!(svc(MAX_IMM16 + 1), over16(MAX_IMM16 + 1));
        assert_eq!(brk(MAX_IMM16 + 1), over16(MAX_IMM16 + 1));
        assert_eq!(add_imm(0, 0, MAX_IMM12 + 1), over12(MAX_IMM12 + 1));
        assert_eq!(sub_imm(0, 0, MAX_IMM12 + 1), over12(MAX_IMM12 + 1));
        // The next aligned offset past the largest one.
        let past = MAX_MEM_OFFSET + MEM_OFFSET_ALIGN;
        let mem = Err(EncodeError::Immediate {
            value: past,
            max: MAX_MEM_OFFSET,
        });
        assert_eq!(ldr(0, 0, past), mem);
        assert_eq!(str(0, 0, past), mem);
    }

    #[test]
    fn shifts_other_than_halfword_multiples_are_errors() {
        for shift in [1, 8, 15, 17, 24, 31, 33, 47, 49, 64] {
            let err = Err(EncodeError::Shift { shift });
            assert_eq!(movz(0, 0, shift), err);
            assert_eq!(movk(0, 0, shift), err);
        }
    }

    #[test]
    fn memory_offsets_must_be_multiples_of_eight() {
        for offset in [1, 4, 7, 9, MAX_MEM_OFFSET - 1] {
            let err = Err(EncodeError::Misaligned {
                offset: i64::from(offset),
                align: MEM_OFFSET_ALIGN,
            });
            assert_eq!(ldr(0, 0, offset), err);
            assert_eq!(str(0, 0, offset), err);
        }
    }

    #[test]
    fn branch_distances_are_checked_against_the_field_and_the_alignment() {
        type Encoder = fn(i64) -> Result<u32, EncodeError>;
        let (b_min, b_max) = reach(BRANCH_IMM_BITS);
        let (cb_min, cb_max) = reach(COMPARE_BRANCH_IMM_BITS);
        let encoders: [(Encoder, i64, i64); 4] = [
            (b, b_min, b_max),
            (bl, b_min, b_max),
            (|offset| cbz(0, offset), cb_min, cb_max),
            (|offset| cbnz(0, offset), cb_min, cb_max),
        ];
        let insn = i64::from(INSN_SIZE);
        for (encode, min, max) in encoders {
            assert!(encode(min).is_ok());
            assert!(encode(max).is_ok());
            for offset in [min - insn, max + insn, i64::MIN, i64::MAX] {
                assert_eq!(
                    encode(offset),
                    Err(EncodeError::Offset { offset, min, max })
                );
            }
            for offset in [1, 2, 3, insn + 1, -1, max - 1, min + 1] {
                assert_eq!(
                    encode(offset),
                    Err(EncodeError::Misaligned {
                        offset,
                        align: INSN_SIZE
                    })
                );
            }
        }
    }

    #[test]
    fn words_become_little_endian_bytes_in_order() {
        assert_eq!(
            to_bytes(&[0x0403_0201, 0x0807_0605]),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(to_bytes(&[]), Vec::<u8>::new());
    }

    /// Sign-extends the low `bits` bits of `value`.
    fn sign_extend(value: u32, bits: u32) -> i64 {
        let shift = 64 - bits;
        (i64::from(value) << shift) >> shift
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]

        /// Each operand lands in its own field, so swapping two of them changes the result.
        #[test]
        fn operands_land_in_their_fields(
            rd in 0..=MAX_REG,
            rn in 0..=MAX_REG,
            rm in 0..=MAX_REG,
            imm12 in 0..=MAX_IMM12,
            imm16 in 0..=MAX_IMM16,
            slot in 0..=MAX_IMM12,
            hw in 0u32..4,
        ) {
            let (rd32, rn32, rm32) = (u32::from(rd), u32::from(rn), u32::from(rm));
            for word in [add_imm(rd, rn, imm12).unwrap(), sub_imm(rd, rn, imm12).unwrap()] {
                prop_assert_eq!(word & 0x1F, rd32);
                prop_assert_eq!((word >> 5) & 0x1F, rn32);
                prop_assert_eq!((word >> 10) & 0xFFF, imm12);
            }
            let offset = slot * MEM_OFFSET_ALIGN;
            for word in [ldr(rd, rn, offset).unwrap(), str(rd, rn, offset).unwrap()] {
                prop_assert_eq!(word & 0x1F, rd32);
                prop_assert_eq!((word >> 5) & 0x1F, rn32);
                prop_assert_eq!((word >> 10) & 0xFFF, slot);
            }
            for word in [movz(rd, imm16, hw * 16).unwrap(), movk(rd, imm16, hw * 16).unwrap()] {
                prop_assert_eq!(word & 0x1F, rd32);
                prop_assert_eq!((word >> 5) & 0xFFFF, imm16);
                prop_assert_eq!((word >> 21) & 0x3, hw);
            }
            let word = mov(rd, rm).unwrap();
            prop_assert_eq!(word & 0x1F, rd32);
            prop_assert_eq!((word >> 16) & 0x1F, rm32);
            prop_assert_eq!((svc(imm16).unwrap() >> 5) & 0xFFFF, imm16);
            prop_assert_eq!((brk(imm16).unwrap() >> 5) & 0xFFFF, imm16);
            let word = mrs(rd, SysReg::TpidrEl0).unwrap();
            prop_assert_eq!(word & 0x1F, rd32);
        }

        /// Every encodable branch distance decodes back to itself with an independent decoder.
        #[test]
        fn branch_offsets_decode_back(
            words26 in -(1i64 << (BRANCH_IMM_BITS - 1))..(1i64 << (BRANCH_IMM_BITS - 1)),
            words19 in -(1i64 << (COMPARE_BRANCH_IMM_BITS - 1))..(1i64 << (COMPARE_BRANCH_IMM_BITS - 1)),
            rt in 0..=MAX_REG,
        ) {
            let insn = i64::from(INSN_SIZE);
            for word in [b(words26 * insn).unwrap(), bl(words26 * insn).unwrap()] {
                prop_assert_eq!(sign_extend(word & 0x03FF_FFFF, BRANCH_IMM_BITS), words26);
            }
            for word in [cbz(rt, words19 * insn).unwrap(), cbnz(rt, words19 * insn).unwrap()] {
                prop_assert_eq!(
                    sign_extend((word >> 5) & 0x7_FFFF, COMPARE_BRANCH_IMM_BITS),
                    words19
                );
                prop_assert_eq!(word & 0x1F, u32::from(rt));
            }
        }

        /// Out-of-range operands are always errors, never another instruction.
        #[test]
        fn out_of_range_operands_never_encode(
            reg in (MAX_REG + 1)..=u8::MAX,
            imm12 in (MAX_IMM12 + 1)..=u32::MAX,
            imm16 in (MAX_IMM16 + 1)..=u32::MAX,
            offset in any::<i64>(),
        ) {
            prop_assert!(mov(reg, 0).is_err());
            prop_assert!(add_imm(0, 0, imm12).is_err());
            prop_assert!(movz(0, imm16, 0).is_err());
            prop_assert!(svc(imm16).is_err());
            let (min, max) = reach(BRANCH_IMM_BITS);
            prop_assert_eq!(
                b(offset).is_ok(),
                (min..=max).contains(&offset) && offset % i64::from(INSN_SIZE) == 0
            );
        }
    }
}
