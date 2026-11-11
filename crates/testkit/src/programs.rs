//! Whole guest programs, as NRO files, for end-to-end runs.
//!
//! Each program is a handful of instructions that does one observable thing and then exits,
//! crashes or loops, so a test (or a person reading a trace) can tell whether the emulator ran it
//! correctly. They are written against Horizon's system-call ABI as a real program would be, and
//! know nothing about the emulator: the two system-call numbers are restated here from the
//! reference ([`SVC_EXIT_PROCESS`], [`SVC_OUTPUT_DEBUG_STRING`]), because this crate depends on
//! none of the workspace's crates. Tests that run these programs cross-check the numbers against
//! the kernel's table.
//!
//! Every program is position-independent: it finds its own data with `adr`, so it runs wherever
//! the image is mapped. Offsets the programs report (`*_offset` fields) are from the first byte of
//! the image.

use crate::a64::{self, AsmError, Assembler, EncodeError, SysReg};
use crate::nro::{self, NroBuildError, NroBuilder};

/// `svc 0x07`, `ExitProcess` (`SvcId::ExitProcess` in `eden/src/core/hle/kernel/svc.h`; the word
/// `svc #0x7` is `kSvcExitProcessInstruction` in `eden/src/core/loader/nro.cpp`). No arguments.
pub const SVC_EXIT_PROCESS: u32 = 0x07;
/// `svc 0x27`, `OutputDebugString` (`SvcId::OutputDebugString` in
/// `eden/src/core/hle/kernel/svc.h`): `x0` is the string's address and `x1` its length
/// (`SvcWrap_OutputDebugString64` in `eden/src/core/hle/kernel/svc.cpp`).
pub const SVC_OUTPUT_DEBUG_STRING: u32 = 0x27;

/// The two lines [`hello`] prints, in order.
pub const HELLO_LINES: [&[u8]; 2] = [b"hello from the guest", b"starquake walking skeleton"];

/// The address [`crash`] stores to: in the first page of the address space, which no loader maps.
pub const CRASH_ADDRESS: u64 = 0xDEA8;

/// How many times [`counter`] spins between its two counter reads: enough that even a counter
/// much slower than the console's (19.2 MHz or 31.25 MHz) has advanced.
pub const COUNTER_SPIN_ITERATIONS: u32 = 1 << 16;

/// Why a program could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ProgramError {
    /// An operand did not fit its instruction.
    #[error(transparent)]
    Encode(#[from] EncodeError),
    /// The program's branches could not be resolved.
    #[error(transparent)]
    Assemble(#[from] AsmError),
    /// The image could not be laid out.
    #[error(transparent)]
    Nro(#[from] NroBuildError),
    /// A line is longer than a `movz` immediate can say.
    #[error("line {index} is {len} bytes, longer than the {max} a program can print")]
    LineTooLong {
        /// Which line, counting from zero.
        index: usize,
        /// Its length in bytes.
        len: usize,
        /// The longest printable line.
        max: u32,
    },
    /// An offset inside the image does not fit the arithmetic used to compute it.
    #[error("the program is too large to address")]
    TooLarge,
}

/// The image offset of the instruction at index `word` of a program's code.
fn code_offset(word: usize) -> Result<u64, ProgramError> {
    let insn = usize::try_from(a64::INSN_SIZE).map_err(|_| ProgramError::TooLarge)?;
    word.checked_mul(insn)
        .and_then(|bytes| bytes.checked_add(nro::CODE_OFFSET))
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or(ProgramError::TooLarge)
}

/// A program that prints [`HELLO_LINES`] with `OutputDebugString`, one call per line, then calls
/// `ExitProcess`.
pub fn hello() -> Result<Vec<u8>, ProgramError> {
    print_lines(&HELLO_LINES)
}

/// A program that prints each of `lines` with `OutputDebugString`, in order, then calls
/// `ExitProcess`. The lines sit back to back in the read-only data segment.
///
/// ```text
/// adr  x0, <line>      ; once per line
/// movz x1, #<length>
/// svc  #0x27
/// ...
/// svc  #0x7
/// ```
pub fn print_lines(lines: &[&[u8]]) -> Result<Vec<u8>, ProgramError> {
    /// Instructions emitted per line.
    const PER_LINE: usize = 3;
    let rodata: Vec<u8> = lines.concat();
    let word_count = lines
        .len()
        .checked_mul(PER_LINE)
        .and_then(|words| words.checked_add(1))
        .ok_or(ProgramError::TooLarge)?;
    // The layout depends on the number of instructions, not on what they are.
    let placeholder = vec![a64::nop(); word_count];
    let [_, rodata_offset, _] = NroBuilder::new(&placeholder)
        .rodata(&rodata)
        .segment_offsets()?;

    let mut code = Vec::with_capacity(word_count);
    let mut line_offset = u64::from(rodata_offset);
    for (index, line) in lines.iter().enumerate() {
        let too_long = ProgramError::LineTooLong {
            index,
            len: line.len(),
            max: a64::MAX_IMM16,
        };
        let len = u32::try_from(line.len()).map_err(|_| too_long)?;
        if len > a64::MAX_IMM16 {
            return Err(too_long);
        }
        let here = code_offset(code.len())?;
        let distance = i64::try_from(line_offset)
            .ok()
            .zip(i64::try_from(here).ok())
            .and_then(|(to, from)| to.checked_sub(from))
            .ok_or(ProgramError::TooLarge)?;
        code.push(a64::adr(0, distance)?);
        code.push(a64::movz(1, len, 0)?);
        code.push(a64::svc(SVC_OUTPUT_DEBUG_STRING)?);
        line_offset = line_offset
            .checked_add(u64::from(len))
            .ok_or(ProgramError::TooLarge)?;
    }
    code.push(a64::svc(SVC_EXIT_PROCESS)?);
    Ok(NroBuilder::new(&code).rodata(&rodata).build()?)
}

/// The crash program and where it faults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crash {
    /// The NRO file.
    pub nro: Vec<u8>,
    /// The address the faulting store writes to ([`CRASH_ADDRESS`]).
    pub fault_address: u64,
    /// Image offset of the faulting `str`.
    pub store_offset: u64,
}

/// A program that stores to [`CRASH_ADDRESS`], which is not mapped.
///
/// ```text
/// movz x0, #0xdea0
/// movz x1, #0x1234
/// str  x1, [x0, #8]    ; faults: a write to 0xdea8
/// svc  #0x7            ; never reached
/// ```
pub fn crash() -> Result<Crash, ProgramError> {
    /// Offset of the store from the base register, so the fault address is not the register's
    /// value.
    const DISPLACEMENT: u32 = 8;
    let base = CRASH_ADDRESS
        .checked_sub(u64::from(DISPLACEMENT))
        .and_then(|base| u32::try_from(base).ok())
        .ok_or(ProgramError::TooLarge)?;
    let code = [
        a64::movz(0, base, 0)?,
        a64::movz(1, 0x1234, 0)?,
        a64::str(1, 0, DISPLACEMENT)?,
        a64::svc(SVC_EXIT_PROCESS)?,
    ];
    Ok(Crash {
        nro: NroBuilder::new(&code).build()?,
        fault_address: CRASH_ADDRESS,
        store_offset: code_offset(2)?,
    })
}

/// A program that never stops: `b .`.
pub fn endless_loop() -> Result<Vec<u8>, ProgramError> {
    Ok(NroBuilder::new(&[a64::b(0)?]).build()?)
}

/// The counter program and the places in it a test wants to name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counter {
    /// The NRO file.
    pub nro: Vec<u8>,
    /// Image offsets of the three counter reads, in program order: `mrs x19, cntpct_el0`,
    /// `mrs x20, cntpct_el0` and `mrs x23, cntfrq_el0`.
    pub read_offsets: [u64; 3],
    /// Image offset of the `brk #1` reached when the second read equals the first.
    pub not_advanced_offset: u64,
    /// Image offset of the `brk #2` reached when the second read is below the first.
    pub went_backwards_offset: u64,
    /// Image offset of the `brk #3` reached when `cntfrq_el0` does not read `guest_hz`.
    pub wrong_rate_offset: u64,
}

/// A program that reads the physical counter twice with a spin loop in between, reads the
/// counter's frequency, and calls `ExitProcess` only if the counter advanced and the frequency is
/// `guest_hz`. Each failed check stops at its own `brk`, so a crash report's instruction address
/// says which one it was.
///
/// ```text
///       mrs  x19, cntpct_el0
///       movz x9, #1, lsl #16
/// spin: sub  x9, x9, #1
///       cbnz x9, spin
///       mrs  x20, cntpct_el0
///       sub  x21, x20, x19
///       cbz  x21, not_advanced
///       lsr  x22, x21, #63
///       cbnz x22, went_backwards
///       mrs  x23, cntfrq_el0
///       movz x24, #<guest_hz bits 0-15>
///       movk x24, #<bits 16-31>, lsl #16
///       movk x24, #<bits 32-47>, lsl #32
///       movk x24, #<bits 48-63>, lsl #48
///       sub  x25, x23, x24
///       cbnz x25, wrong_rate
///       svc  #0x7
/// not_advanced:   brk #1
/// went_backwards: brk #2
/// wrong_rate:     brk #3
/// ```
pub fn counter(guest_hz: u64) -> Result<Counter, ProgramError> {
    let halfword = |shift: u32| {
        u32::try_from((guest_hz >> shift) & u64::from(a64::MAX_IMM16))
            .map_err(|_| ProgramError::TooLarge)
    };
    let spin_count = COUNTER_SPIN_ITERATIONS >> 16;

    let mut asm = Assembler::new();
    let (spin, not_advanced, went_backwards, wrong_rate) =
        (asm.label(), asm.label(), asm.label(), asm.label());
    let first_read = asm.len();
    asm.emit(a64::mrs(19, SysReg::CntPctEl0)?);
    asm.emit(a64::movz(9, spin_count, 16)?);
    asm.bind(spin)?;
    asm.emit(a64::sub_imm(9, 9, 1)?);
    asm.cbnz(9, spin)?;
    let second_read = asm.len();
    asm.emit(a64::mrs(20, SysReg::CntPctEl0)?);
    asm.emit(a64::sub(21, 20, 19)?);
    asm.cbz(21, not_advanced)?;
    asm.emit(a64::lsr_imm(22, 21, a64::MAX_SHIFT)?);
    asm.cbnz(22, went_backwards)?;
    let rate_read = asm.len();
    asm.emit(a64::mrs(23, SysReg::CntFrqEl0)?);
    asm.emit(a64::movz(24, halfword(0)?, 0)?);
    for shift in [16, 32, 48] {
        asm.emit(a64::movk(24, halfword(shift)?, shift)?);
    }
    asm.emit(a64::sub(25, 23, 24)?);
    asm.cbnz(25, wrong_rate)?;
    asm.emit(a64::svc(SVC_EXIT_PROCESS)?);
    let not_advanced_at = asm.len();
    asm.bind(not_advanced)?;
    asm.emit(a64::brk(1)?);
    let went_backwards_at = asm.len();
    asm.bind(went_backwards)?;
    asm.emit(a64::brk(2)?);
    let wrong_rate_at = asm.len();
    asm.bind(wrong_rate)?;
    asm.emit(a64::brk(3)?);

    Ok(Counter {
        nro: NroBuilder::new(&asm.finish()?).build()?,
        read_offsets: [
            code_offset(first_read)?,
            code_offset(second_read)?,
            code_offset(rate_read)?,
        ],
        not_advanced_offset: code_offset(not_advanced_at)?,
        went_backwards_offset: code_offset(went_backwards_at)?,
        wrong_rate_offset: code_offset(wrong_rate_at)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instruction word at image offset `offset`.
    fn word_at(file: &[u8], offset: u64) -> u32 {
        let at = usize::try_from(offset).unwrap();
        u32::from_le_bytes(file[at..at + 4].try_into().unwrap())
    }

    /// The code words of a program, from the first instruction up to (not including) `count`.
    fn code(file: &[u8], count: usize) -> Vec<u32> {
        (0..count)
            .map(|word| word_at(file, code_offset(word).unwrap()))
            .collect()
    }

    /// Where an `adr` at image offset `at` points, decoded independently of the encoder: `immhi`
    /// in bits 23:5, `immlo` in bits 30:29, sign-extended from 21 bits.
    fn adr_target(word: u32, at: u64) -> u64 {
        assert_eq!(word & 0x9F00_0000, 0x1000_0000, "not an adr: {word:#x}");
        let raw = i64::from((((word >> 5) & 0x7_FFFF) << 2) | ((word >> 29) & 0x3));
        let offset = (raw << 43) >> 43;
        at.checked_add_signed(offset).unwrap()
    }

    #[test]
    fn hello_points_each_call_at_its_own_line() {
        let file = hello().unwrap();
        let words = code(&file, HELLO_LINES.len() * 3 + 1);
        for (index, line) in HELLO_LINES.iter().enumerate() {
            let (adr, len, svc) = (words[index * 3], words[index * 3 + 1], words[index * 3 + 2]);
            let target = adr_target(adr, code_offset(index * 3).unwrap());
            let target = usize::try_from(target).unwrap();
            assert_eq!(&file[target..target + line.len()], *line, "line {index}");
            assert_eq!(adr & 0x1F, 0, "the address goes in x0");
            assert_eq!(
                len,
                a64::movz(1, u32::try_from(line.len()).unwrap(), 0).unwrap()
            );
            assert_eq!(svc, 0xD400_04E1, "svc #0x27");
        }
        assert_eq!(words.last(), Some(&0xD400_00E1), "svc #0x7");
        // The two lines differ, so a program that printed one of them twice would be caught.
        assert_ne!(HELLO_LINES[0], HELLO_LINES[1]);
    }

    #[test]
    fn print_lines_handles_no_lines_and_rejects_a_line_too_long_to_say() {
        let file = print_lines(&[]).unwrap();
        assert_eq!(code(&file, 1), [0xD400_00E1]);
        let longest = vec![b'x'; usize::try_from(a64::MAX_IMM16).unwrap()];
        assert!(print_lines(&[b"ok", &longest]).is_ok());
        let too_long = vec![b'x'; longest.len() + 1];
        assert_eq!(
            print_lines(&[b"ok", &too_long]),
            Err(ProgramError::LineTooLong {
                index: 1,
                len: too_long.len(),
                max: a64::MAX_IMM16
            })
        );
    }

    #[test]
    fn crash_stores_to_the_address_it_reports() {
        let crash = crash().unwrap();
        // movz x0, #0xdea0 ; movz x1, #0x1234 ; str x1, [x0, #8] ; svc #0x7
        assert_eq!(
            code(&crash.nro, 4),
            [0xD29B_D400, 0xD282_4681, 0xF900_0401, 0xD400_00E1]
        );
        assert_eq!(crash.store_offset, code_offset(2).unwrap());
        assert_eq!(crash.fault_address, 0xDEA0 + 8);
        assert_eq!(crash.fault_address, CRASH_ADDRESS);
    }

    #[test]
    fn the_endless_loop_branches_to_itself() {
        assert_eq!(code(&endless_loop().unwrap(), 1), [0x1400_0000]);
    }

    #[test]
    fn counter_reports_where_its_reads_and_traps_are() {
        // A rate with four distinct halfwords, so a swapped `movk` would show.
        let guest_hz = 0x1122_3344_5566_7788;
        let counter = counter(guest_hz).unwrap();
        let at = |offset| word_at(&counter.nro, offset);
        assert_eq!(
            at(counter.read_offsets[0]),
            0xD53B_E033,
            "mrs x19, cntpct_el0"
        );
        assert_eq!(
            at(counter.read_offsets[1]),
            0xD53B_E034,
            "mrs x20, cntpct_el0"
        );
        assert_eq!(
            at(counter.read_offsets[2]),
            0xD53B_E017,
            "mrs x23, cntfrq_el0"
        );
        assert_eq!(at(counter.not_advanced_offset), 0xD420_0020, "brk #1");
        assert_eq!(at(counter.went_backwards_offset), 0xD420_0040, "brk #2");
        assert_eq!(at(counter.wrong_rate_offset), 0xD420_0060, "brk #3");
        // The four words after the rate read build `guest_hz` in x24, low halfword first.
        let mut built = 0u64;
        for (index, shift) in (1u64..).zip([0u32, 16, 32, 48]) {
            let word = at(counter.read_offsets[2] + index * 4);
            assert_eq!(word & 0x1F, 24);
            assert_eq!((word >> 21) & 0x3, shift / 16);
            built |= u64::from((word >> 5) & 0xFFFF) << shift;
        }
        assert_eq!(built, guest_hz);
        // The spin count is the one documented.
        assert_eq!(
            at(counter.read_offsets[0] + 4),
            a64::movz(9, 1, 16).unwrap()
        );
        assert_eq!(COUNTER_SPIN_ITERATIONS, 1 << 16);
    }

    #[test]
    fn counter_branches_reach_their_own_traps() {
        let counter = counter(19_200_000).unwrap();
        // cbz/cbnz: imm19 at bits 23:5, in words, sign-extended.
        let target = |offset: u64| {
            let word = word_at(&counter.nro, offset);
            let words = (i64::from((word >> 5) & 0x7_FFFF) << 45) >> 45;
            offset.checked_add_signed(words * 4).unwrap()
        };
        let second_read = counter.read_offsets[1];
        assert_eq!(target(second_read + 8), counter.not_advanced_offset);
        assert_eq!(target(second_read + 16), counter.went_backwards_offset);
        assert_eq!(
            target(counter.read_offsets[2] + 24),
            counter.wrong_rate_offset
        );
        // The spin loop's cbnz goes back one instruction, to the sub.
        assert_eq!(
            target(counter.read_offsets[0] + 12),
            counter.read_offsets[0] + 8
        );
    }
}
