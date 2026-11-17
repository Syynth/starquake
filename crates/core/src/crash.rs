//! The crash report: what a guest crash leaves behind, as data first (`crash.json`) and text
//! second (`crash.txt`).
//!
//! Source: `docs/specs/observability.md`, "Crash reports". This is the part of that list the
//! walking skeleton can fill in: the exception, the registers, the faulting instruction as module
//! plus offset, a backtrace by frame pointer with each frame as module plus offset
//! ([`Backtrace`]), the memory map and the block the faulting address is in, the modules with
//! their build ids, the last events from the flight recorder, the emulator's version and the
//! run's command line.
//!
//! PROVISIONAL (walking skeleton, #74). Not here yet, and owned by the crash-report step of the
//! observability spec's order of work: symbol names, and the events split per thread (the report
//! quotes the newest events of every thread, merged in order). The exception is one of four
//! kinds: a data abort, an instruction abort, a `Break` system call that is not a notification
//! (the program stopped itself), and everything else as an undefined instruction. The CPU
//! backend's syndrome for it ([`CrashReport::with_syndrome`]) says the rest: whether an abort was
//! a missing page or a refused access, and what an "undefined instruction" really was. A `Break`
//! has no syndrome: the CPU took an ordinary system call, and the report's `syndrome` is `null`.
//!
//! # The backtrace
//!
//! [`Backtrace::walk`] follows Eden's `GetAArch64Backtrace` (`eden/src/core/arm/debug.cpp` @
//! cf654052): frame 0 is the program counter, frame 1 the link register (`x30`) as the
//! registers hold it, and each later frame the return address read from a frame record, the two
//! words `x29` points to (the caller's frame pointer at +0, the return address at +8). Frame 1
//! repeats frame 2 when the crashing function had already saved its link register in its own
//! frame record, as Eden's does; a leaf function that saved none needs it. Addresses are the
//! return addresses as read, not the calls before them.
//!
//! The walk stops at the first of these, checked in this order for each frame pointer:
//!
//! 1. it is zero ([`BacktraceEnd::NullFramePointer`]: the chain is complete);
//! 2. it is not a multiple of [`FRAME_POINTER_ALIGN`] ([`BacktraceEnd::Misaligned`]);
//! 3. the walk already read the record it points to ([`BacktraceEnd::Loop`]);
//! 4. the [`FRAME_RECORD_SIZE`] bytes it points to are not all inside the stack
//!    ([`BacktraceEnd::OutsideStack`]);
//! 5. [`MAX_FRAME_RECORDS`] records have been read already ([`BacktraceEnd::DepthLimit`]);
//! 6. guest memory refuses the read of the record ([`BacktraceEnd::Unreadable`]).
//!
//! Where it differs from Eden's:
//!
//! - **The stack** is starquake's own idea (a guess at what a frame record may be in, not a
//!   reference fact): the block of the kernel's memory map that holds the crashing thread's stack
//!   pointer, when the guest may both read and write it. Eden accepts a record anywhere its
//!   `IsValidVirtualAddressRange` does (any mapped memory). A crash whose stack pointer is in no
//!   such block has no stack, and its walk stops at the first record. Test:
//!   `a_record_outside_the_stack_ends_the_walk`.
//! - **Loops** end the walk at the first record read twice; Eden reads on until its loop bound.
//!   Test: `a_chain_that_loops_ends_at_the_first_record_read_twice`.
//! - **The depth limit** is Eden's loop bound, 256 records, but Eden adds the return address of
//!   only the first 255 of them (the last one it reads is never added); here every record read
//!   adds its frame. Test: `the_walk_reads_at_most_its_limit_of_records`.
//! - **A record the guest's memory refuses** ends the walk with a note
//!   ([`BacktraceEnd::Unreadable`]), never a panic. Eden checks the range is mapped first and then
//!   reads it.
//!
//! The JSON layout is this crate's own (the spec gives none), versioned by [`CRASH_VERSION`] and
//! pinned by a test. Addresses and register values are hex strings, so no reader loses bits to a
//! floating-point number. Version 2 added the `syndrome` field, version 3 the `svc_break`
//! exception kind, and version 4 the `backtrace` field. (The `instruction_abort` kind came
//! earlier, inside version 1; before it, a fetch from an address nothing maps was reported as a
//! `data_abort` read.)

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Serialize, Serializer};
use starquake_cpu::{ExceptionClass, FaultStatus, GuestRegs, Syndrome};
use starquake_kernel::svc::exception::Break;
use starquake_memory::GuestMemory;

use crate::report::Mapping;

/// Version of the `crash.json` layout. Any change to the layout bumps it: a field or exception
/// kind added, renamed or removed, or a field's meaning changed.
pub const CRASH_VERSION: u32 = 4;

/// Serialises a `u64` as a hex string (`"0x80000000"`).
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `serialize_with` calls this with a reference to the field"
)]
pub(crate) fn hex<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{value:#x}"))
}

/// Serialises a `u32` as a hex string (`"0x1"`).
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `serialize_with` calls this with a reference to the field"
)]
fn hex_word<S: Serializer>(value: &u32, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{value:#x}"))
}

/// Serialises an optional `u32` as a hex string, or `null`.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    clippy::ref_option,
    reason = "serde's `serialize_with` calls this with a reference to the field"
)]
fn hex_word_or_null<S: Serializer>(value: &Option<u32>, serializer: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => hex_word(value, serializer),
        None => serializer.serialize_none(),
    }
}

/// Serialises an optional `u64` as a hex string, or `null`.
#[expect(
    clippy::ref_option,
    reason = "serde's `serialize_with` calls this with a reference to the field"
)]
fn hex_or_null<S: Serializer>(value: &Option<u64>, serializer: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => hex(value, serializer),
        None => serializer.serialize_none(),
    }
}

/// Serialises `u64`s as a list of hex strings.
fn hex_list<S: Serializer>(values: &[u64], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter().map(|value| format!("{value:#x}")))
}

/// Serialises bytes as one lowercase hex string with no prefix (a build id).
fn hex_bytes<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&HexBytes(bytes))
}

struct HexBytes<'a>(&'a [u8]);

impl std::fmt::Display for HexBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// The exception that ended the guest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Exception {
    /// A load or store that nothing maps, or that the page's permissions refuse.
    DataAbort {
        /// The address the guest tried to access.
        #[serde(serialize_with = "hex")]
        address: u64,
        /// Whether the access was a write.
        write: bool,
    },
    /// An instruction fetch from an address that nothing maps, or that is not executable.
    InstructionAbort {
        /// The address the guest tried to execute.
        #[serde(serialize_with = "hex")]
        address: u64,
    },
    /// An instruction the guest may not execute: undefined, a breakpoint, a failed pointer
    /// authentication, a misaligned PC or SP.
    UndefinedInstruction,
    /// The program called `Break` without the notification flag: it stopped itself
    /// ([`starquake_kernel::svc::exception`]).
    SvcBreak {
        /// The reason word as passed (`Svc::BreakReason`); the notification flag is clear.
        #[serde(serialize_with = "hex_word")]
        reason: u32,
        /// The reason in words (`panic`, `assert`, ...), or `null` for a reason Eden does not
        /// name.
        reason_name: Option<&'static str>,
        /// The buffer's address (`x1`).
        #[serde(serialize_with = "hex")]
        address: u64,
        /// The buffer's size (`x2`).
        #[serde(serialize_with = "hex")]
        size: u64,
        /// The word a four-byte buffer held, usually a result code; `null` when the buffer was
        /// not four bytes or could not be read.
        #[serde(serialize_with = "hex_word_or_null")]
        code: Option<u32>,
    },
}

impl Exception {
    /// The exception a `Break` call that is not a notification ends the run with.
    #[must_use]
    pub const fn svc_break(call: &Break) -> Self {
        Self::SvcBreak {
            reason: call.raw_reason(),
            reason_name: call.reason.name(),
            address: call.address,
            size: call.size,
            code: call.code,
        }
    }
}

/// The exception's syndrome, as the CPU backend reported it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SyndromeReport {
    /// The syndrome in the architecture's layout (`ESR_EL1`).
    #[serde(serialize_with = "hex")]
    pub esr: u64,
    /// What it says, in words: `data abort, translation fault at level 3`.
    pub text: String,
}

impl SyndromeReport {
    /// The report of `syndrome`.
    #[must_use]
    pub fn new(syndrome: Syndrome) -> Self {
        let class = class_text(syndrome.class());
        let text = match (syndrome.fault(), syndrome.immediate()) {
            (Some(fault), _) => format!("{class}, {}", fault_text(fault)),
            (None, Some(immediate)) => format!("{class} #{immediate:#x}"),
            (None, None) => class,
        };
        Self {
            esr: syndrome.esr(),
            text,
        }
    }
}

/// An exception class in words.
fn class_text(class: ExceptionClass) -> String {
    match class {
        ExceptionClass::Unknown => "undefined instruction".to_owned(),
        ExceptionClass::TrappedWait => "trapped wfi or wfe".to_owned(),
        ExceptionClass::Svc => "svc".to_owned(),
        ExceptionClass::TrappedSystemAccess => {
            "system register or instruction not allowed at EL0".to_owned()
        }
        ExceptionClass::PointerAuthentication => "pointer authentication failure".to_owned(),
        ExceptionClass::InstructionAbort => "instruction abort".to_owned(),
        ExceptionClass::PcAlignment => "misaligned pc".to_owned(),
        ExceptionClass::DataAbort => "data abort".to_owned(),
        ExceptionClass::SpAlignment => "misaligned stack pointer".to_owned(),
        ExceptionClass::Breakpoint => "brk".to_owned(),
        ExceptionClass::Other(ec) => format!("exception class {ec:#04x}"),
    }
}

/// Why an access aborted, in words.
fn fault_text(fault: FaultStatus) -> String {
    match fault {
        FaultStatus::AddressSize { level } => format!("address size fault at level {level}"),
        FaultStatus::Translation { level } => format!("translation fault at level {level}"),
        FaultStatus::AccessFlag { level } => format!("access flag fault at level {level}"),
        FaultStatus::Permission { level } => format!("permission fault at level {level}"),
        FaultStatus::Alignment => "alignment fault".to_owned(),
        FaultStatus::Other(code) => format!("fault status {code:#04x}"),
    }
}

/// The guest's registers at the exception.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Registers {
    /// `x0`-`x30`.
    #[serde(serialize_with = "hex_list")]
    pub x: Vec<u64>,
    /// `SP_EL0`.
    #[serde(serialize_with = "hex")]
    pub sp: u64,
    /// The program counter: the address of the instruction that took the exception.
    #[serde(serialize_with = "hex")]
    pub pc: u64,
    /// The [`starquake_cpu::GUEST_PSTATE_MASK`] bits of `PSTATE` (the condition flags `NZCV`,
    /// `DIT` and `SSBS`) in the `SPSR_EL1` layout; every other bit is zero.
    #[serde(serialize_with = "hex")]
    pub pstate: u64,
    /// `TPIDR_EL0`.
    #[serde(serialize_with = "hex")]
    pub tpidr: u64,
    /// `TPIDRRO_EL0`.
    #[serde(serialize_with = "hex")]
    pub tpidrro: u64,
}

/// A loaded module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Module {
    /// The module's name.
    pub name: String,
    /// Guest address of its first byte.
    #[serde(serialize_with = "hex")]
    pub base: u64,
    /// Its size in memory, `.bss` included.
    #[serde(serialize_with = "hex")]
    pub size: u64,
    /// Its build id.
    #[serde(serialize_with = "hex_bytes")]
    pub build_id: Vec<u8>,
}

/// An address as a module plus an offset into it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Location {
    /// The module's name.
    pub module: String,
    /// Offset from the module's base.
    #[serde(serialize_with = "hex")]
    pub offset: u64,
}

impl Location {
    /// `address` as the module of `modules` that holds it plus the offset into it; `None` when
    /// no module holds it. A module holds its base and not its end (`base + size`).
    #[must_use]
    pub fn of(address: u64, modules: &[Module]) -> Option<Self> {
        modules.iter().find_map(|module| {
            let offset = address.checked_sub(module.base)?;
            (offset < module.size).then(|| Self {
                module: module.name.clone(),
                offset,
            })
        })
    }
}

/// `main+0xb8`.
impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}+{:#x}", self.module, self.offset)
    }
}

/// An address and where it is, in words: `0x800000b8 (main+0xb8)`, or `0x10 (in no module)`.
#[must_use]
pub fn address_text(address: u64, location: Option<&Location>) -> String {
    match location {
        Some(location) => format!("{address:#x} ({location})"),
        None => format!("{address:#x} (in no module)"),
    }
}

/// The register that holds the frame pointer, `x29` (AAPCS64; Eden's `ThreadContext::fp`, after
/// `r[0..29]`, `eden/src/core/hle/kernel/svc_types.h` @ cf654052).
pub const FRAME_POINTER_REGISTER: usize = 29;
/// The link register, `x30` (Eden's `ThreadContext::lr`, the same file).
pub const LINK_REGISTER: usize = 30;
/// The most frame records a backtrace reads: the bound of Eden's loop in `GetAArch64Backtrace`
/// (`eden/src/core/arm/debug.cpp` @ cf654052, `for (size_t i = 0; i < 256; i++)`).
pub const MAX_FRAME_RECORDS: usize = 256;
/// A frame pointer the walk follows is a multiple of this: Eden's check, `fp % 4 != 0` ends its
/// walk (the same function).
pub const FRAME_POINTER_ALIGN: u64 = 4;
/// The size of a frame record in bytes: the caller's frame pointer, then the return address, a
/// 64-bit word each (Eden's comment in the same function: "Frame records are two words long").
pub const FRAME_RECORD_SIZE: usize = 16;
/// Where the return address is in a frame record (`fp+8 : value of lr for frame`, the same
/// comment); the caller's frame pointer is at its start.
pub const FRAME_RECORD_RETURN: usize = 8;

/// One frame of a backtrace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Frame {
    /// The program counter (frame 0), the link register (frame 1), or the return address a
    /// frame record holds (every later frame).
    #[serde(serialize_with = "hex")]
    pub address: u64,
    /// That address as module plus offset, when it is inside a module.
    pub location: Option<Location>,
    /// The address of the frame record the return address was read from; `null` for frames 0
    /// and 1, which come from the registers.
    #[serde(serialize_with = "hex_or_null")]
    pub record: Option<u64>,
}

/// Why a backtrace's walk ended (see the module docs for the order of the checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum BacktraceEnd {
    /// The frame pointer is zero: the chain is complete.
    NullFramePointer,
    /// The frame pointer is not a multiple of [`FRAME_POINTER_ALIGN`].
    Misaligned {
        /// The frame pointer.
        #[serde(serialize_with = "hex")]
        frame_pointer: u64,
    },
    /// The frame pointer names a record the walk already read: the chain loops.
    Loop {
        /// The frame pointer.
        #[serde(serialize_with = "hex")]
        frame_pointer: u64,
    },
    /// The record the frame pointer names is not wholly inside the stack, or there is no stack.
    OutsideStack {
        /// The frame pointer.
        #[serde(serialize_with = "hex")]
        frame_pointer: u64,
    },
    /// [`MAX_FRAME_RECORDS`] records were read, and the chain goes on.
    DepthLimit,
    /// Guest memory refused the read of the record the frame pointer names.
    Unreadable {
        /// The frame pointer.
        #[serde(serialize_with = "hex")]
        frame_pointer: u64,
    },
}

impl BacktraceEnd {
    /// Why the walk ended, in words.
    #[must_use]
    pub fn text(&self) -> String {
        match *self {
            Self::NullFramePointer => "the chain ends at a null frame pointer".to_owned(),
            Self::Misaligned { frame_pointer } => format!(
                "cut short: the frame pointer {frame_pointer:#x} is not a multiple of \
                 {FRAME_POINTER_ALIGN}"
            ),
            Self::Loop { frame_pointer } => format!(
                "cut short: the frame record at {frame_pointer:#x} was read before (the chain \
                 loops)"
            ),
            Self::OutsideStack { frame_pointer } => {
                format!("cut short: the frame record at {frame_pointer:#x} is outside the stack")
            }
            Self::DepthLimit => format!(
                "cut short: {MAX_FRAME_RECORDS} frame records were read, the most a backtrace \
                 reads"
            ),
            Self::Unreadable { frame_pointer } => {
                format!("cut short: the frame record at {frame_pointer:#x} could not be read")
            }
        }
    }
}

/// The crashing thread's frame-pointer chain (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Backtrace {
    /// The memory block every frame record must be in: the one holding the stack pointer, named
    /// by its state in the kernel's memory map (`Stack`, `Normal`, ...); `null` when the stack
    /// pointer is in no block the guest can read and write.
    pub stack: Option<Mapping>,
    /// Innermost first: the program counter, the link register, then one frame per record read.
    pub frames: Vec<Frame>,
    /// Why the walk ended.
    pub end: BacktraceEnd,
}

impl Backtrace {
    /// Walks the chain of frame records from the registers `regs`, reading `memory`, with every
    /// record inside `stack` (see the module docs), and places each frame in `modules`.
    #[must_use]
    pub fn walk(
        regs: &GuestRegs,
        stack: Option<Mapping>,
        memory: &dyn GuestMemory,
        modules: &[Module],
    ) -> Self {
        let frame = |address: u64, record: Option<u64>| Frame {
            address,
            location: Location::of(address, modules),
            record,
        };
        let mut frames = vec![frame(regs.pc, None), frame(regs.x[LINK_REGISTER], None)];
        // The records read, by address: at most `MAX_FRAME_RECORDS` of them.
        let mut read = BTreeSet::new();
        let mut frame_pointer = regs.x[FRAME_POINTER_REGISTER];
        let end = loop {
            if frame_pointer == 0 {
                break BacktraceEnd::NullFramePointer;
            }
            if !frame_pointer.is_multiple_of(FRAME_POINTER_ALIGN) {
                break BacktraceEnd::Misaligned { frame_pointer };
            }
            if read.contains(&frame_pointer) {
                break BacktraceEnd::Loop { frame_pointer };
            }
            if !stack
                .as_ref()
                .is_some_and(|stack| holds_record(stack, frame_pointer))
            {
                break BacktraceEnd::OutsideStack { frame_pointer };
            }
            if read.len() >= MAX_FRAME_RECORDS {
                break BacktraceEnd::DepthLimit;
            }
            let mut record = [0; FRAME_RECORD_SIZE];
            if memory.read(frame_pointer, &mut record).is_err() {
                break BacktraceEnd::Unreadable { frame_pointer };
            }
            read.insert(frame_pointer);
            let (caller, rest) = record.split_at(FRAME_RECORD_RETURN);
            frames.push(frame(word(rest), Some(frame_pointer)));
            frame_pointer = word(caller);
        };
        Self { stack, frames, end }
    }

    fn write_text(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "\nbacktrace, innermost first:")?;
        for (index, frame) in self.frames.iter().enumerate() {
            let source = match (index, frame.record) {
                (_, Some(record)) => format!("frame record at {record:#x}"),
                (0, None) => "pc".to_owned(),
                (_, None) => "lr".to_owned(),
            };
            writeln!(
                out,
                "  #{index} {}, {source}",
                address_text(frame.address, frame.location.as_ref())
            )?;
        }
        match &self.stack {
            Some(Mapping { name, start, end }) => {
                writeln!(out, "  stack: {start:#x}-{end:#x} {name}")?;
            }
            None => writeln!(
                out,
                "  stack: none (the stack pointer is in no memory the guest can read and write)"
            )?,
        }
        writeln!(out, "  {}", self.end.text())
    }
}

/// Whether the frame record at `address` is wholly inside `stack`: its first byte at or above
/// the start, and its end at or below the stack's end, without wrapping.
fn holds_record(stack: &Mapping, address: u64) -> bool {
    let size = FRAME_RECORD_SIZE as u64;
    address >= stack.start
        && address
            .checked_add(size)
            .is_some_and(|end| end <= stack.end)
}

/// The little-endian 64-bit word `bytes` holds (eight bytes; fewer read as zero-padded).
fn word(bytes: &[u8]) -> u64 {
    let mut word = [0; 8];
    for (to, from) in word.iter_mut().zip(bytes) {
        *to = *from;
    }
    u64::from_le_bytes(word)
}

/// Everything recorded about a guest crash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CrashReport {
    /// [`CRASH_VERSION`].
    pub version: u32,
    /// The emulator and its version.
    pub emulator: String,
    /// The command line that started the run; empty when there was none.
    pub command_line: Vec<String>,
    /// What happened.
    pub exception: Exception,
    /// The exception's syndrome; `null` when the report was built without one.
    pub syndrome: Option<SyndromeReport>,
    /// The address of the instruction that took the exception.
    #[serde(serialize_with = "hex")]
    pub pc: u64,
    /// That address as module plus offset, when it is inside a module.
    pub pc_location: Option<Location>,
    /// The crashing thread's frame-pointer chain; `null` when the report was built without one.
    pub backtrace: Option<Backtrace>,
    /// For a data abort or an instruction abort, the name of the mapping the faulting address
    /// is in; `null` when the address is not mapped, and for other exceptions.
    pub fault_mapping: Option<String>,
    /// The registers at the exception.
    pub registers: Registers,
    /// The guest memory the run set up, in address order.
    pub memory_map: Vec<Mapping>,
    /// The loaded modules.
    pub modules: Vec<Module>,
    /// The newest events from the flight recorder, oldest first, one line each.
    pub last_events: Vec<String>,
}

impl CrashReport {
    /// Builds the report for `exception`, taken with the guest's registers as `regs`.
    #[must_use]
    pub fn new(
        exception: Exception,
        regs: &GuestRegs,
        memory_map: &[Mapping],
        modules: &[Module],
        last_events: Vec<String>,
        command_line: &[String],
    ) -> Self {
        let pc_location = Location::of(regs.pc, modules);
        let fault_mapping = match exception {
            Exception::DataAbort { address, .. } | Exception::InstructionAbort { address } => {
                memory_map
                    .iter()
                    .find(|mapping| mapping.contains(address))
                    .map(|mapping| mapping.name.clone())
            }
            Exception::UndefinedInstruction | Exception::SvcBreak { .. } => None,
        };
        Self {
            version: CRASH_VERSION,
            emulator: format!("starquake {}", env!("CARGO_PKG_VERSION")),
            command_line: command_line.to_vec(),
            exception,
            syndrome: None,
            pc: regs.pc,
            pc_location,
            backtrace: None,
            fault_mapping,
            registers: Registers {
                x: regs.x.to_vec(),
                sp: regs.sp,
                pc: regs.pc,
                pstate: regs.pstate,
                tpidr: regs.tpidr,
                tpidrro: regs.tpidrro,
            },
            memory_map: memory_map.to_vec(),
            modules: modules.to_vec(),
            last_events,
        }
    }

    /// The report with `syndrome`, the one the CPU backend reported for the exception.
    #[must_use]
    pub fn with_syndrome(mut self, syndrome: Syndrome) -> Self {
        self.syndrome = Some(SyndromeReport::new(syndrome));
        self
    }

    /// The report with `backtrace`, the crashing thread's ([`Backtrace::walk`]).
    #[must_use]
    pub fn with_backtrace(mut self, backtrace: Backtrace) -> Self {
        self.backtrace = Some(backtrace);
        self
    }

    /// The report as text, for `crash.txt`.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        // Writing to a `String` cannot fail, so the results are dropped.
        let _ = self.write_text(&mut text);
        text
    }

    /// What happened, in a few words: `data abort, a write at address 0xdea8 (not mapped)`.
    #[must_use]
    pub fn exception_text(&self) -> String {
        let block = || {
            self.fault_mapping
                .as_deref()
                .map_or_else(|| "not mapped".to_owned(), |name| format!("in {name}"))
        };
        match self.exception {
            Exception::DataAbort { address, write } => {
                let access = if write { "write" } else { "read" };
                format!(
                    "data abort, a {access} at address {address:#x} ({})",
                    block()
                )
            }
            Exception::InstructionAbort { address } => {
                format!(
                    "instruction abort, a fetch at address {address:#x} ({})",
                    block()
                )
            }
            Exception::UndefinedInstruction => "undefined instruction".to_owned(),
            Exception::SvcBreak {
                reason,
                reason_name,
                address,
                size,
                code,
            } => {
                let reason = reason_name.map_or_else(
                    || format!("reason {reason:#x}"),
                    |name| format!("reason {name} ({reason:#x})"),
                );
                let code = code.map_or_else(String::new, |code| format!(", code {code:#x}"));
                format!("svc Break, {reason}, buffer {address:#x} size {size:#x}{code}")
            }
        }
    }

    /// Where it happened: `pc 0x800000b8 (main+0xb8)`.
    #[must_use]
    pub fn instruction_text(&self) -> String {
        format!("pc {}", address_text(self.pc, self.pc_location.as_ref()))
    }

    fn write_text(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "{} crash report", self.emulator)?;
        if !self.command_line.is_empty() {
            writeln!(out, "command line: {}", self.command_line.join(" "))?;
        }
        writeln!(out)?;
        writeln!(out, "exception: {}", self.exception_text())?;
        if let Some(SyndromeReport { esr, text }) = &self.syndrome {
            writeln!(out, "syndrome: {esr:#010x} ({text})")?;
        }
        writeln!(out, "instruction: {}", self.instruction_text())?;
        if let Some(backtrace) = &self.backtrace {
            backtrace.write_text(out)?;
        }
        writeln!(out, "\nregisters:")?;
        for (row, values) in self.registers.x.chunks(4).enumerate() {
            write!(out, " ")?;
            for (column, value) in values.iter().enumerate() {
                write!(out, " x{:<2} {value:#018x}", row * 4 + column)?;
            }
            writeln!(out)?;
        }
        let Registers {
            sp,
            pc,
            pstate,
            tpidr,
            tpidrro,
            x: _,
        } = &self.registers;
        writeln!(out, "  sp  {sp:#018x} pc  {pc:#018x} pstate {pstate:#x}")?;
        writeln!(out, "  tpidr {tpidr:#018x} tpidrro {tpidrro:#018x}")?;
        writeln!(out, "\nmemory map:")?;
        for Mapping { name, start, end } in &self.memory_map {
            writeln!(out, "  {start:#x}-{end:#x} {name}")?;
        }
        writeln!(out, "\nmodules:")?;
        for module in &self.modules {
            writeln!(
                out,
                "  {} base {:#x} size {:#x} build id {}",
                module.name,
                module.base,
                module.size,
                HexBytes(&module.build_id)
            )?;
        }
        writeln!(out, "\nlast events, oldest first:")?;
        for event in &self.last_events {
            writeln!(out, "  {event}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registers that all differ, so a swapped pair shows.
    fn regs() -> GuestRegs {
        let mut regs = GuestRegs {
            sp: 0x80ff_fff0,
            pc: 0x8000_00b8,
            pstate: 0x6000_0000,
            tpidr: 0x1111,
            tpidrro: 0x8000_3000,
            ..GuestRegs::default()
        };
        for (index, x) in (0u64..).zip(regs.x.iter_mut()) {
            *x = 0x100 + index;
        }
        regs
    }

    fn memory_map() -> Vec<Mapping> {
        vec![
            Mapping {
                name: "main.text".into(),
                start: 0x8000_0000,
                end: 0x8000_1000,
            },
            Mapping {
                name: "stack".into(),
                start: 0x80f0_0000,
                end: 0x8100_0000,
            },
        ]
    }

    fn modules() -> Vec<Module> {
        vec![Module {
            name: "main".into(),
            base: 0x8000_0000,
            size: 0x1000,
            build_id: vec![0xab, 0x01, 0xff],
        }]
    }

    fn report(exception: Exception) -> CrashReport {
        CrashReport::new(
            exception,
            &regs(),
            &memory_map(),
            &modules(),
            vec!["#0 core 0 thread 1: lifecycle ProcessStart 1".into()],
            &["starquake".into(), "run".into(), "crash.nro".into()],
        )
    }

    #[test]
    fn the_report_names_the_faulting_address_and_the_instruction() {
        let report = report(Exception::DataAbort {
            address: 0xdea8,
            write: true,
        });
        assert_eq!(report.pc, 0x8000_00b8);
        assert_eq!(
            report.pc_location,
            Some(Location {
                module: "main".into(),
                offset: 0xb8
            })
        );
        assert_eq!(report.fault_mapping, None);
        let text = report.to_text();
        assert!(
            text.contains("exception: data abort, a write at address 0xdea8 (not mapped)"),
            "{text}"
        );
        assert!(
            text.contains("instruction: pc 0x800000b8 (main+0xb8)"),
            "{text}"
        );
        assert!(text.contains("command line: starquake run crash.nro"));
        assert!(text.contains("  0x80f00000-0x81000000 stack"));
        assert!(text.contains("  main base 0x80000000 size 0x1000 build id ab01ff"));
        assert!(text.contains("  #0 core 0 thread 1: lifecycle ProcessStart 1"));
        // Every register appears, each with its own value.
        for index in 0..31u64 {
            let value = format!("x{index:<2} {:#018x}", 0x100 + index);
            assert!(text.contains(&value), "{value} missing from\n{text}");
        }
        assert!(text.contains("sp  0x0000000080fffff0 pc  0x00000000800000b8 pstate 0x60000000"));
        assert!(text.contains("tpidr 0x0000000000001111 tpidrro 0x0000000080003000"));
    }

    #[test]
    fn a_fault_inside_a_mapping_names_it_and_a_read_is_called_a_read() {
        let report = report(Exception::DataAbort {
            address: 0x80f0_0010,
            write: false,
        });
        assert_eq!(report.fault_mapping.as_deref(), Some("stack"));
        assert!(
            report
                .to_text()
                .contains("exception: data abort, a read at address 0x80f00010 (in stack)")
        );
    }

    #[test]
    fn an_instruction_abort_names_the_address_fetched_from_and_its_mapping() {
        // A jump into the stack: mapped, not executable.
        let report = report(Exception::InstructionAbort {
            address: 0x80f0_0010,
        });
        assert_eq!(report.fault_mapping.as_deref(), Some("stack"));
        assert!(
            report
                .to_text()
                .contains("exception: instruction abort, a fetch at address 0x80f00010 (in stack)")
        );
        // A jump to an address nothing maps.
        let report = super::tests::report(Exception::InstructionAbort { address: 0xdea8 });
        assert_eq!(report.fault_mapping, None);
        assert!(
            report
                .to_text()
                .contains("exception: instruction abort, a fetch at address 0xdea8 (not mapped)")
        );
    }

    #[test]
    fn a_pc_outside_every_module_has_no_location() {
        let mut regs = regs();
        // One past the module's last byte, and one below its first.
        for pc in [0x8000_1000, 0x7fff_fffc] {
            regs.pc = pc;
            let report = CrashReport::new(
                Exception::UndefinedInstruction,
                &regs,
                &memory_map(),
                &modules(),
                Vec::new(),
                &[],
            );
            assert_eq!(report.pc_location, None, "{pc:#x}");
            let text = report.to_text();
            assert!(text.contains("exception: undefined instruction"));
            assert!(text.contains(&format!("instruction: pc {pc:#x} (in no module)")));
            assert!(!text.contains("command line"));
        }
        // The module's last byte is inside it.
        regs.pc = 0x8000_0fff;
        let last = CrashReport::new(
            Exception::UndefinedInstruction,
            &regs,
            &[],
            &modules(),
            Vec::new(),
            &[],
        );
        assert_eq!(
            last.pc_location.map(|location| location.offset),
            Some(0xfff)
        );
    }

    /// `ESR_EL1` with exception class `ec` and instruction-specific syndrome `iss`, `IL` set (Arm
    /// ARM, `ESR_EL1`: `EC` bits 31:26, `IL` bit 25).
    fn esr(ec: u64, iss: u64) -> Syndrome {
        Syndrome::from_esr((ec << 26) | (1 << 25) | iss)
    }

    #[test]
    fn every_class_and_fault_status_is_said_in_words() {
        let cases = [
            (Syndrome::undefined(), "undefined instruction"),
            (esr(0x01, 0x1e0_0000), "trapped wfi or wfe"),
            (esr(0x15, 0x27), "svc #0x27"),
            (
                esr(0x18, 0x32_0400),
                "system register or instruction not allowed at EL0",
            ),
            (esr(0x1c, 0), "pointer authentication failure"),
            (esr(0x22, 0), "misaligned pc"),
            (esr(0x26, 0), "misaligned stack pointer"),
            (esr(0x3c, 0x123), "brk #0x123"),
            (esr(0x16, 2), "exception class 0x16"),
            (
                Syndrome::data_abort(FaultStatus::AddressSize { level: 1 }, false),
                "data abort, address size fault at level 1",
            ),
            (
                Syndrome::data_abort(FaultStatus::Translation { level: 2 }, true),
                "data abort, translation fault at level 2",
            ),
            (
                Syndrome::data_abort(FaultStatus::AccessFlag { level: 3 }, false),
                "data abort, access flag fault at level 3",
            ),
            (
                Syndrome::instruction_abort(FaultStatus::Permission { level: 3 }),
                "instruction abort, permission fault at level 3",
            ),
            (
                Syndrome::data_abort(FaultStatus::Alignment, true),
                "data abort, alignment fault",
            ),
            (
                Syndrome::instruction_abort(FaultStatus::Other(0x30)),
                "instruction abort, fault status 0x30",
            ),
        ];
        for (syndrome, text) in cases {
            let report = SyndromeReport::new(syndrome);
            assert_eq!(report.text, text, "{:#x}", syndrome.esr());
            assert_eq!(report.esr, syndrome.esr());
        }
    }

    #[test]
    fn the_text_says_the_syndrome_after_the_exception_when_there_is_one() {
        let plain = report(Exception::UndefinedInstruction);
        assert_eq!(plain.syndrome, None);
        assert!(!plain.to_text().contains("syndrome"));
        let brk = plain.with_syndrome(esr(0x3c, 0x123));
        assert!(
            brk.to_text().contains(
                "exception: undefined instruction\nsyndrome: 0xf2000123 (brk #0x123)\n\
                 instruction: pc 0x800000b8 (main+0xb8)\n"
            ),
            "{}",
            brk.to_text()
        );
        // The ESR is always eight digits, so a class 0x00 syndrome keeps its leading zero.
        let undefined =
            report(Exception::UndefinedInstruction).with_syndrome(Syndrome::undefined());
        assert!(
            undefined
                .to_text()
                .contains("syndrome: 0x02000000 (undefined instruction)\n"),
            "{}",
            undefined.to_text()
        );
    }

    #[test]
    fn the_json_layout_is_pinned() {
        let mut report = report(Exception::DataAbort {
            address: 0x80f0_0010,
            write: true,
        })
        .with_syndrome(Syndrome::data_abort(
            FaultStatus::Permission { level: 3 },
            true,
        ))
        .with_backtrace(Backtrace {
            stack: None,
            frames: vec![Frame {
                address: 0x7000_0010,
                location: None,
                record: Some(0x80ff_ffd0),
            }],
            end: BacktraceEnd::DepthLimit,
        });
        report.emulator = "starquake 9.9.9".into();
        report.registers.x.truncate(2);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            json,
            r##"{
  "version": 4,
  "emulator": "starquake 9.9.9",
  "command_line": [
    "starquake",
    "run",
    "crash.nro"
  ],
  "exception": {
    "kind": "data_abort",
    "address": "0x80f00010",
    "write": true
  },
  "syndrome": {
    "esr": "0x9200004f",
    "text": "data abort, permission fault at level 3"
  },
  "pc": "0x800000b8",
  "pc_location": {
    "module": "main",
    "offset": "0xb8"
  },
  "backtrace": {
    "stack": null,
    "frames": [
      {
        "address": "0x70000010",
        "location": null,
        "record": "0x80ffffd0"
      }
    ],
    "end": {
      "reason": "depth_limit"
    }
  },
  "fault_mapping": "stack",
  "registers": {
    "x": [
      "0x100",
      "0x101"
    ],
    "sp": "0x80fffff0",
    "pc": "0x800000b8",
    "pstate": "0x60000000",
    "tpidr": "0x1111",
    "tpidrro": "0x80003000"
  },
  "memory_map": [
    {
      "name": "main.text",
      "start": "0x80000000",
      "end": "0x80001000"
    },
    {
      "name": "stack",
      "start": "0x80f00000",
      "end": "0x81000000"
    }
  ],
  "modules": [
    {
      "name": "main",
      "base": "0x80000000",
      "size": "0x1000",
      "build_id": "ab01ff"
    }
  ],
  "last_events": [
    "#0 core 0 thread 1: lifecycle ProcessStart 1"
  ]
}"##
        );
        // The golden text pins the version as a literal, so a layout change that forgets to bump
        // `CRASH_VERSION` fails here.
        assert_eq!(CRASH_VERSION, 4);
        assert_eq!(report.version, CRASH_VERSION);
    }

    #[test]
    fn the_plain_kinds_and_the_parts_a_report_was_built_without_are_pinned() {
        let undefined = serde_json::to_value(Exception::UndefinedInstruction).unwrap();
        assert_eq!(
            undefined,
            serde_json::json!({ "kind": "undefined_instruction" })
        );
        let fetch = serde_json::to_value(Exception::InstructionAbort {
            address: 0x80f0_0010,
        })
        .unwrap();
        assert_eq!(
            fetch,
            serde_json::json!({ "kind": "instruction_abort", "address": "0x80f00010" })
        );
        // A report built without a syndrome says so.
        let without =
            serde_json::to_value(super::tests::report(Exception::UndefinedInstruction)).unwrap();
        // The key is written as `null`, not left out (indexing a missing key would also give
        // `Null`, so check the object itself).
        let without = without.as_object().unwrap();
        assert_eq!(without.get("syndrome"), Some(&serde_json::Value::Null));
        // And one built without a backtrace.
        assert_eq!(without.get("backtrace"), Some(&serde_json::Value::Null));
    }

    /// A break with every field set, each to a distinct value.
    fn assert_break() -> Exception {
        Exception::SvcBreak {
            reason: 1,
            reason_name: Some("assert"),
            address: 0x80f0_0010,
            size: 4,
            code: Some(0x2a_0c0e),
        }
    }

    #[test]
    fn a_break_names_its_reason_its_buffer_and_its_code() {
        let report = report(assert_break());
        // The buffer is not a faulting address: no mapping is named, though it is in one.
        assert_eq!(report.fault_mapping, None);
        assert_eq!(report.syndrome, None);
        assert_eq!(
            report.exception_text(),
            "svc Break, reason assert (0x1), buffer 0x80f00010 size 0x4, code 0x2a0c0e"
        );
        assert!(
            report.to_text().contains(
                "exception: svc Break, reason assert (0x1), buffer 0x80f00010 size 0x4, \
                 code 0x2a0c0e\ninstruction: pc 0x800000b8 (main+0xb8)\n"
            ),
            "{}",
            report.to_text()
        );
        // A reason Eden does not name is given by number, and no code is no code.
        let unnamed = super::tests::report(Exception::SvcBreak {
            reason: 0x99,
            reason_name: None,
            address: 0,
            size: 0,
            code: None,
        });
        assert_eq!(
            unnamed.exception_text(),
            "svc Break, reason 0x99, buffer 0x0 size 0x0"
        );
    }

    #[test]
    fn a_break_is_reported_as_the_kernel_decoded_it() {
        use starquake_kernel::svc::exception::{BreakReason, NOTIFICATION_ONLY_FLAG};
        let mut call = Break::decode(BreakReason::Assert.raw(), 0x80f0_0010, 4);
        call.code = Some(0x2a_0c0e);
        assert_eq!(Exception::svc_break(&call), assert_break());
        // A reason with no name keeps its number, and has no name.
        let other = Break::decode(0x99, 0x10, 0x20);
        assert_eq!(
            Exception::svc_break(&other),
            Exception::SvcBreak {
                reason: 0x99,
                reason_name: None,
                address: 0x10,
                size: 0x20,
                code: None,
            }
        );
        // Were a notification ever reported, its word would keep the flag.
        let notice = Break::decode(NOTIFICATION_ONLY_FLAG | 3, 0, 0);
        assert!(matches!(
            Exception::svc_break(&notice),
            Exception::SvcBreak {
                reason: 0x8000_0003,
                reason_name: Some("pre_load_dll"),
                ..
            }
        ));
    }

    #[test]
    fn the_break_kind_s_json_layout_is_pinned() {
        // Every field distinct and non-zero.
        assert_eq!(
            serde_json::to_string_pretty(&assert_break()).unwrap(),
            r#"{
  "kind": "svc_break",
  "reason": "0x1",
  "reason_name": "assert",
  "address": "0x80f00010",
  "size": "0x4",
  "code": "0x2a0c0e"
}"#
        );
        // The two optional fields are written as `null`, not left out.
        let unnamed = Exception::SvcBreak {
            reason: 0x99,
            reason_name: None,
            address: 0x10,
            size: 0x20,
            code: None,
        };
        assert_eq!(
            serde_json::to_value(unnamed).unwrap(),
            serde_json::json!({
                "kind": "svc_break",
                "reason": "0x99",
                "reason_name": null,
                "address": "0x10",
                "size": "0x20",
                "code": null,
            })
        );
        // In a whole report (version 3 added the kind).
        let json = serde_json::to_value(report(assert_break())).unwrap();
        assert_eq!(json["version"], CRASH_VERSION);
        assert_eq!(json["exception"]["kind"], "svc_break");
        assert_eq!(json["syndrome"], serde_json::Value::Null);
    }

    #[test]
    fn pstate_carries_dit_and_ssbs_as_well_as_the_flags() {
        use starquake_cpu::{PSTATE_DIT, PSTATE_NZCV, PSTATE_SSBS};
        let mut report = report(Exception::UndefinedInstruction);
        report.registers.pstate = PSTATE_NZCV | PSTATE_DIT | PSTATE_SSBS;
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["registers"]["pstate"], "0xf1001000");
        assert!(report.to_text().contains("pstate 0xf1001000\n"));
    }
}

#[cfg(test)]
mod backtrace_tests;
