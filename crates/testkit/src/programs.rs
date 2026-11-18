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

/// The rate, in hertz, a Switch 1 process's counter runs at, and so what its `cntfrq_el0`
/// reads: 19.2 MHz (`CNTFREQ` in `eden/src/core/hardware_properties.h`, "`CNTPCT_EL0` Frequency =
/// 19.2 MHz"). The rate to build [`counter`] with for a run under the kernel as it is today.
pub const COUNTER_HZ: u64 = 19_200_000;

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
    /// A message is longer than a thread's message buffer.
    #[error("message {index} is {len} bytes, longer than the {max} of a message buffer")]
    MessageTooLong {
        /// Which message, counting from zero.
        index: usize,
        /// Its length in bytes.
        len: usize,
        /// The longest message.
        max: usize,
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
    let [base, value, store] = store_to_crash_address()?;
    let code = [base, value, store, a64::svc(SVC_EXIT_PROCESS)?];
    Ok(Crash {
        nro: NroBuilder::new(&code).build()?,
        fault_address: CRASH_ADDRESS,
        store_offset: code_offset(2)?,
    })
}

/// The three instructions of [`crash`] that fault: a store of `0x1234` to [`CRASH_ADDRESS`],
/// through a base register that holds the address less the store's displacement.
fn store_to_crash_address() -> Result<[u32; 3], ProgramError> {
    /// Offset of the store from the base register, so the fault address is not the register's
    /// value.
    const DISPLACEMENT: u32 = 8;
    let base = CRASH_ADDRESS
        .checked_sub(u64::from(DISPLACEMENT))
        .and_then(|base| u32::try_from(base).ok())
        .ok_or(ProgramError::TooLarge)?;
    Ok([
        a64::movz(0, base, 0)?,
        a64::movz(1, 0x1234, 0)?,
        a64::str(1, 0, DISPLACEMENT)?,
    ])
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

/// `svc 0x08`, `CreateThread` (`SvcId::CreateThread` in `eden/src/core/hle/kernel/svc.h`):
/// `x1` entry, `x2` argument, `x3` stack, `w4` priority, `w5` core; `w0` result and `w1` the
/// handle out (`SvcWrap_CreateThread64` in `eden/src/core/hle/kernel/svc.cpp`).
pub const SVC_CREATE_THREAD: u32 = 0x08;
/// `svc 0x09`, `StartThread`: `w0` the handle; `w0` the result.
pub const SVC_START_THREAD: u32 = 0x09;
/// `svc 0x0a`, `ExitThread`. No arguments.
pub const SVC_EXIT_THREAD: u32 = 0x0a;
/// `svc 0x10`, `GetCurrentProcessorNumber`: `w0` the core out.
pub const SVC_GET_CURRENT_PROCESSOR_NUMBER: u32 = 0x10;
/// `svc 0x18`, `WaitSynchronization`: `x1` the handles, `w2` their count, `x3` the timeout;
/// `w0` result and `w1` the index out (`SvcWrap_WaitSynchronization64`).
pub const SVC_WAIT_SYNCHRONIZATION: u32 = 0x18;

/// The priority [`threads`] gives its second thread: Eden's default for a homebrew main thread
/// (`0x2c`, the main thread priority of a program with no metadata), so the two are equals.
pub const THREAD_PRIORITY: u32 = 0x2c;
/// The core [`threads`] starts its second thread on: not the main thread's (core 0).
pub const SECOND_THREAD_CORE: u32 = 1;
/// What [`threads`]'s second thread stores for the main thread to find: proof it ran.
pub const THREAD_MARKER: u32 = 0x7EAD;

/// The threads program and the places in it a test wants to name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Threads {
    /// The NRO file.
    pub nro: Vec<u8>,
    /// Image offsets of the `brk`s each failed check stops at, in program order: `CreateThread`
    /// failed, `StartThread` failed, the join failed, the second thread ran on another core
    /// than [`SECOND_THREAD_CORE`], and it did not leave [`THREAD_MARKER`].
    pub failure_offsets: [u64; 5],
}

/// A program that starts a second thread on core [`SECOND_THREAD_CORE`] and joins it: the
/// second thread reports the core it runs on and [`THREAD_MARKER`] in a slot on the main
/// thread's stack, and ends; the main thread waits on the thread's handle with no timeout,
/// checks both, and calls `ExitProcess`. Each failed check stops at its own `brk`.
///
/// ```text
///         b    main
/// thread: mov  x19, x0              ; where to report
///         svc  #0x10                ; GetCurrentProcessorNumber
///         str  x0, [x19]
///         movz x9, #MARKER
///         str  x9, [x19, #8]
///         svc  #0xa                 ; ExitThread
/// main:   add  x20, sp, #0
///         sub  x20, x20, #32        ; a slot: the handle, then the core, then the marker
///         adr  x1, thread
///         add  x2, x20, #8
///         sub  x3, x20, #0x800      ; its stack, below the slot
///         movz x4, #0x2c
///         movz x5, #1
///         svc  #0x8                 ; CreateThread
///         cbnz x0, create_failed
///         str  x1, [x20]
///         mov  x0, x1
///         svc  #0x9                 ; StartThread
///         cbnz x0, start_failed
///         add  x1, x20, #0          ; the handle list
///         movz x2, #1
///         movz x3, #0
///         sub  x3, x3, #1           ; no timeout
///         svc  #0x18                ; WaitSynchronization: the join
///         cbnz x0, wait_failed
///         ldr  x9, [x20, #8]
///         sub  x9, x9, #1
///         cbnz x9, wrong_core
///         ldr  x9, [x20, #16]
///         movz x10, #MARKER
///         sub  x9, x9, x10
///         cbnz x9, no_marker
///         svc  #0x7
/// create_failed: brk #1 ... no_marker: brk #5
/// ```
pub fn threads() -> Result<Threads, ProgramError> {
    /// The slot's size below the stack pointer, and where the second thread's stack starts
    /// below the slot.
    const SLOT: u32 = 32;
    const STACK_BELOW: u32 = 0x800;
    /// Offsets in the slot.
    const CORE_AT: u32 = 8;
    const MARKER_AT: u32 = 16;
    const SP: u8 = 31;

    let mut asm = Assembler::new();
    let main = asm.label();
    let failures = [
        asm.label(),
        asm.label(),
        asm.label(),
        asm.label(),
        asm.label(),
    ];
    asm.b(main)?;
    let thread = asm.len();
    asm.emit(a64::mov(19, 0)?);
    asm.emit(a64::svc(SVC_GET_CURRENT_PROCESSOR_NUMBER)?);
    asm.emit(a64::str(0, 19, 0)?);
    asm.emit(a64::movz(9, THREAD_MARKER, 0)?);
    asm.emit(a64::str(9, 19, MARKER_AT - CORE_AT)?);
    asm.emit(a64::svc(SVC_EXIT_THREAD)?);

    asm.bind(main)?;
    asm.emit(a64::add_imm(20, SP, 0)?);
    asm.emit(a64::sub_imm(20, 20, SLOT)?);
    let back = |here: usize| -> Result<i64, ProgramError> {
        let words = i64::try_from(thread)
            .ok()
            .zip(i64::try_from(here).ok())
            .and_then(|(to, from)| to.checked_sub(from))
            .ok_or(ProgramError::TooLarge)?;
        words
            .checked_mul(i64::from(a64::INSN_SIZE))
            .ok_or(ProgramError::TooLarge)
    };
    asm.emit(a64::adr(1, back(asm.len())?)?);
    asm.emit(a64::add_imm(2, 20, CORE_AT)?);
    asm.emit(a64::sub_imm(3, 20, STACK_BELOW)?);
    asm.emit(a64::movz(4, THREAD_PRIORITY, 0)?);
    asm.emit(a64::movz(5, SECOND_THREAD_CORE, 0)?);
    asm.emit(a64::svc(SVC_CREATE_THREAD)?);
    asm.cbnz(0, failures[0])?;
    asm.emit(a64::str(1, 20, 0)?);
    asm.emit(a64::mov(0, 1)?);
    asm.emit(a64::svc(SVC_START_THREAD)?);
    asm.cbnz(0, failures[1])?;
    asm.emit(a64::add_imm(1, 20, 0)?);
    asm.emit(a64::movz(2, 1, 0)?);
    asm.emit(a64::movz(3, 0, 0)?);
    asm.emit(a64::sub_imm(3, 3, 1)?);
    asm.emit(a64::svc(SVC_WAIT_SYNCHRONIZATION)?);
    asm.cbnz(0, failures[2])?;
    asm.emit(a64::ldr(9, 20, CORE_AT)?);
    asm.emit(a64::sub_imm(9, 9, SECOND_THREAD_CORE)?);
    asm.cbnz(9, failures[3])?;
    asm.emit(a64::ldr(9, 20, MARKER_AT)?);
    asm.emit(a64::movz(10, THREAD_MARKER, 0)?);
    asm.emit(a64::sub(9, 9, 10)?);
    asm.cbnz(9, failures[4])?;
    asm.emit(a64::svc(SVC_EXIT_PROCESS)?);
    let mut failure_offsets = [0; 5];
    for ((label, offset), code) in failures.into_iter().zip(&mut failure_offsets).zip(1..) {
        *offset = code_offset(asm.len())?;
        asm.bind(label)?;
        asm.emit(a64::brk(code)?);
    }
    Ok(Threads {
        nro: NroBuilder::new(&asm.finish()?).build()?,
        failure_offsets,
    })
}

/// `svc 0x1f`, `ConnectToNamedPort` (`SvcId::ConnectToNamedPort` in
/// `eden/src/core/hle/kernel/svc.h`): `x1` is the port name's address, and the session handle
/// comes back in `x1` (`SvcWrap_ConnectToNamedPort64` in `eden/src/core/hle/kernel/svc.cpp`).
pub const SVC_CONNECT_TO_NAMED_PORT: u32 = 0x1F;
/// `svc 0x21`, `SendSyncRequest` (`SvcId::SendSyncRequest` in `eden/src/core/hle/kernel/svc.h`):
/// `w0` is the session handle; the message is the calling thread's, at the start of the block
/// `TPIDRRO_EL0` points to (`SvcWrap_SendSyncRequest64` in `eden/src/core/hle/kernel/svc.cpp`).
pub const SVC_SEND_SYNC_REQUEST: u32 = 0x21;

/// The size of a message a [`requests`] step may copy: the thread's whole message buffer, 0x100
/// bytes (`COMMAND_BUFFER_LENGTH = 0x100 / sizeof(u32)` in `eden/src/core/hle/ipc.h`; restated
/// here, as this crate depends on none of the workspace's).
pub const MESSAGE_LIMIT: usize = 0x100;

/// The session a [`requests`] step sends its message on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendOn {
    /// The session `ConnectToNamedPort` gave.
    Port,
    /// The handle in 32-bit word `word` of the reply the previous step got (for a reply that
    /// moves one handle and carries no process id, word 3: after the two header words and the
    /// special header).
    Reply {
        /// The word's index from the start of the message.
        word: u32,
    },
}

/// One request of [`requests`]: its message, built by the test, and the session it goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step<'a> {
    /// The message's bytes, at most [`MESSAGE_LIMIT`].
    pub message: &'a [u8],
    /// Where it goes.
    pub on: SendOn,
}

/// A program that connects to the named port `port` (its bytes, with the terminating zero),
/// sends each of `steps` with `SendSyncRequest`, in order and whatever each reply holds, and
/// then stores to [`CRASH_ADDRESS`] as [`crash`] does, so the run ends in a crash report. The
/// fault address is [`CRASH_ADDRESS`].
///
/// ```text
/// mrs  x19, tpidrro_el0
/// adr  x1, <port>
/// svc  #0x1f                ; ConnectToNamedPort
/// mov  x20, x1
/// ; each step:
/// mov  x0, x20              ; Port, or for Reply { word }:
/// ldr  x0, [x19, #<word / 2 * 8>]
/// lsr  x0, x0, #32          ;   when the word is odd
/// adr  x2, <message>
/// ldr  x3, [x2, #8k]        ; for each 8 bytes of the message
/// str  x3, [x19, #8k]
/// svc  #0x21                ; SendSyncRequest
/// ; then:
/// movz x0, #0xdea0
/// movz x1, #0x1234
/// str  x1, [x0, #8]
/// ```
///
/// A handle is loaded before the next message is copied over the reply that holds it. An even
/// word leaves the next word in the register's upper half; `SendSyncRequest` reads the handle
/// from `w0` alone.
pub fn requests(port: &[u8], steps: &[Step<'_>]) -> Result<Vec<u8>, ProgramError> {
    /// The thread pointer and the port's session, kept across the steps.
    const TLS: u8 = 19;
    const PORT: u8 = 20;
    let eight = usize::try_from(a64::MEM_OFFSET_ALIGN).map_err(|_| ProgramError::TooLarge)?;
    let padded = |bytes: &[u8]| {
        let mut bytes = bytes.to_vec();
        bytes.resize(bytes.len().next_multiple_of(eight), 0);
        bytes
    };
    if let Some(index) = steps
        .iter()
        .position(|step| step.message.len() > MESSAGE_LIMIT)
    {
        return Err(ProgramError::MessageTooLong {
            index,
            len: steps.get(index).map_or(0, |step| step.message.len()),
            max: MESSAGE_LIMIT,
        });
    }
    let mut rodata = padded(port);
    let mut message_offsets = Vec::with_capacity(steps.len());
    for step in steps {
        message_offsets.push(rodata.len());
        rodata.extend(padded(step.message));
    }

    let emit_steps = |code: &mut Vec<u32>, rodata_offset: u64| -> Result<(), ProgramError> {
        let adr_to = |code: &Vec<u32>, rodata_at: usize| -> Result<i64, ProgramError> {
            let here = code_offset(code.len())?;
            let to = u64::try_from(rodata_at)
                .ok()
                .and_then(|at| rodata_offset.checked_add(at))
                .ok_or(ProgramError::TooLarge)?;
            i64::try_from(to)
                .ok()
                .zip(i64::try_from(here).ok())
                .and_then(|(to, from)| to.checked_sub(from))
                .ok_or(ProgramError::TooLarge)
        };
        code.push(a64::mrs(TLS, SysReg::TpidrroEl0)?);
        code.push(a64::adr(1, adr_to(code, 0)?)?);
        code.push(a64::svc(SVC_CONNECT_TO_NAMED_PORT)?);
        code.push(a64::mov(PORT, 1)?);
        for (step, at) in steps.iter().zip(&message_offsets) {
            match step.on {
                SendOn::Port => code.push(a64::mov(0, PORT)?),
                SendOn::Reply { word } => {
                    let offset = (word / 2)
                        .checked_mul(a64::MEM_OFFSET_ALIGN)
                        .ok_or(ProgramError::TooLarge)?;
                    code.push(a64::ldr(0, TLS, offset)?);
                    if word % 2 == 1 {
                        code.push(a64::lsr_imm(0, 0, u32::BITS)?);
                    }
                }
            }
            code.push(a64::adr(2, adr_to(code, *at)?)?);
            let words = step.message.len().div_ceil(eight);
            for k in 0..words {
                let offset = k
                    .checked_mul(eight)
                    .and_then(|offset| u32::try_from(offset).ok())
                    .ok_or(ProgramError::TooLarge)?;
                code.push(a64::ldr(3, 2, offset)?);
                code.push(a64::str(3, TLS, offset)?);
            }
            code.push(a64::svc(SVC_SEND_SYNC_REQUEST)?);
        }
        code.extend(store_to_crash_address()?);
        Ok(())
    };
    // The layout depends on the number of instructions, not on what they are.
    let mut sizing = Vec::new();
    emit_steps(&mut sizing, 0)?;
    let placeholder = vec![a64::nop(); sizing.len()];
    let [_, rodata_offset, _] = NroBuilder::new(&placeholder)
        .rodata(&rodata)
        .segment_offsets()?;
    let mut code = Vec::with_capacity(sizing.len());
    emit_steps(&mut code, u64::from(rodata_offset))?;
    Ok(NroBuilder::new(&code).rodata(&rodata).build()?)
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

    #[test]
    fn threads_starts_its_second_thread_at_its_own_code_and_stops_at_numbered_traps() {
        let program = threads().unwrap();
        let file = &program.nro;
        let entry = code_offset(0).unwrap();
        // The second thread's code follows the first instruction, a branch over it.
        let thread = entry + u64::from(a64::INSN_SIZE);
        assert_eq!(word_at(file, thread), a64::mov(19, 0).unwrap());
        // Find the `adr x1` and check it points at the thread.
        let adr_at = (0..64)
            .map(|word| code_offset(word).unwrap())
            .find(|offset| {
                let word = word_at(file, *offset);
                word & 0x9F00_001F == 0x1000_0001
            })
            .unwrap();
        assert_eq!(adr_target(word_at(file, adr_at), adr_at), thread);
        // The calls, in program order.
        let svcs: Vec<u32> = (0..64)
            .map(|word| word_at(file, code_offset(word).unwrap()))
            .filter(|word| word & 0xFFE0_001F == 0xD400_0001)
            .map(|word| (word >> 5) & 0xFFFF)
            .collect();
        assert_eq!(
            svcs,
            [
                SVC_GET_CURRENT_PROCESSOR_NUMBER,
                SVC_EXIT_THREAD,
                SVC_CREATE_THREAD,
                SVC_START_THREAD,
                SVC_WAIT_SYNCHRONIZATION,
                SVC_EXIT_PROCESS,
            ]
        );
        for (offset, code) in program.failure_offsets.iter().zip(1..) {
            assert_eq!(word_at(file, *offset), a64::brk(code).unwrap());
        }
    }

    #[test]
    fn requests_sends_each_message_on_its_session_then_crashes() {
        let first = [0x11_u8; 12];
        let second: Vec<u8> = (1..=16).collect();
        let third = [0x33_u8; 8];
        let steps = [
            Step {
                message: &first,
                on: SendOn::Port,
            },
            Step {
                message: &second,
                on: SendOn::Reply { word: 3 },
            },
            Step {
                message: &third,
                on: SendOn::Reply { word: 4 },
            },
        ];
        let file = requests(b"sm:\0", &steps).unwrap();
        // 4 to connect; per step a handle (1 or 2), an adr, a load and a store per 8 bytes, and
        // the call; then the crash's 3.
        let count = 4 + (1 + 1 + 2 * 2 + 1) + (2 + 1 + 2 * 2 + 1) + (1 + 1 + 2 + 1) + 3;
        let words = code(&file, count);
        let at = |word: usize| code_offset(word).unwrap();
        let bytes_at = |word: usize, len: usize| {
            let target = usize::try_from(adr_target(words[word], at(word))).unwrap();
            file[target..target + len].to_vec()
        };
        assert_eq!(words[0], a64::mrs(19, SysReg::TpidrroEl0).unwrap());
        assert_eq!(words[1] & 0x1F, 1, "the port's name goes in x1");
        assert_eq!(bytes_at(1, 4), b"sm:\0");
        assert_eq!(words[2], a64::svc(SVC_CONNECT_TO_NAMED_PORT).unwrap());
        assert_eq!(words[3], a64::mov(20, 1).unwrap());
        // The first step: on the port's session.
        assert_eq!(words[4], a64::mov(0, 20).unwrap());
        assert_eq!(bytes_at(5, first.len()), first);
        assert_eq!(
            words[6..10],
            [
                a64::ldr(3, 2, 0).unwrap(),
                a64::str(3, 19, 0).unwrap(),
                a64::ldr(3, 2, 8).unwrap(),
                a64::str(3, 19, 8).unwrap(),
            ]
        );
        assert_eq!(words[10], a64::svc(SVC_SEND_SYNC_REQUEST).unwrap());
        // The second: the handle in word 3 of the reply, the upper half of its second
        // doubleword, loaded before the message is copied over it.
        assert_eq!(words[11], a64::ldr(0, 19, 8).unwrap());
        assert_eq!(words[12], a64::lsr_imm(0, 0, 32).unwrap());
        assert_eq!(bytes_at(13, second.len()), second);
        assert_eq!(words[18], a64::svc(SVC_SEND_SYNC_REQUEST).unwrap());
        // The third: word 4, the lower half of the third doubleword, needs no shift.
        assert_eq!(words[19], a64::ldr(0, 19, 16).unwrap());
        assert_eq!(bytes_at(20, third.len()), third);
        assert_eq!(
            words[21..23],
            [a64::ldr(3, 2, 0).unwrap(), a64::str(3, 19, 0).unwrap()]
        );
        assert_eq!(words[23], a64::svc(SVC_SEND_SYNC_REQUEST).unwrap());
        // Then the crash program's store to `CRASH_ADDRESS`.
        let crash = crash().unwrap();
        assert_eq!(words[24..27], code(&crash.nro, 3)[..]);
    }

    #[test]
    fn requests_takes_a_message_as_long_as_a_message_buffer_and_no_longer() {
        let longest = vec![1; MESSAGE_LIMIT];
        let step = |message| Step {
            message,
            on: SendOn::Port,
        };
        assert!(requests(b"sm:\0", &[step(&longest)]).is_ok());
        let too_long = vec![1; MESSAGE_LIMIT + 1];
        assert_eq!(
            requests(b"sm:\0", &[step(&longest), step(&too_long)]),
            Err(ProgramError::MessageTooLong {
                index: 1,
                len: MESSAGE_LIMIT + 1,
                max: MESSAGE_LIMIT
            })
        );
    }
}
