//! The front thread's macro host: GPU protocol slice G13 (`docs/specs/gpu-protocol.md`,
//! "Macros", "Engine methods: the register file" and "Malformed streams").
//!
//! The front runs every macro call (owner's decision 1) against its own register files, with
//! the existing interpreter (`crate::mme`) behind the memo ([`MacroMemo`], slice G6). This module
//! owns what sits around them:
//!
//! - [`ProcessMacros`]: one process's macro code and start tables, shared by all its channels
//!   (owner's decision 11 (a)), with a record of which channel uploaded what, the memo of each
//!   code store, and the scratch of the run in progress.
//! - [`MacroChannel`]: one channel's macro state: its data RAM, shared by its 3D and compute
//!   classes (row 84), the call whose parameters are arriving, and a call waiting for a
//!   catch-up.
//! - [`ProcessMacros::write`]: the front's entry for an engine data word ([`EngineWord`]) of any
//!   class. A word to a macro method collects a parameter and runs the call once the last one
//!   arrives; any other word goes to the register file ([`RegFile`], slice G9), and a word to
//!   one of the `LOAD_MME_*` methods also uploads code or binds a start.
//! - [`ProcessMacros::resume`]: runs a call again once the back has caught up.
//! - [`BackWork`]: the one question the host asks the back thread, whether it has work it has
//!   not performed yet.
//!
//! Ported from `eden-nx2/src/video_core/engines/maxwell_3d.cpp` (`FrontWrite`,
//! `FrontCallMethod`, `FrontCallMultiMethod`, `ProcessMacro`, `CallMacroMethod`),
//! `eden-nx2/src/video_core/engines/kepler_compute.cpp` (`ProcessNativeMethod` from line 123,
//! `FrontApply`, `FrontCallMethod`, `FrontRunMacro`), `eden-nx2/src/video_core/macro.h`
//! (`MacroEngine::AddCode`, `ClearCode`, `NativeMacroContext`), `eden-nx2/src/video_core/macro.cpp`
//! (`MacroEngine::Execute`, the native path, its MME DMA reads) and
//! `eden-nx2/src/video_core/control/channel_state.cpp` (`SetNativeDataRam`), Eden-NX2 @ 50fa86bb,
//! GPL-3.0-or-later; PR #4588's `eden-pr4588/src/video_core/macro.h` @ e5fb3b97 agrees on the code
//! store. Method numbers and fields come from the generated `clc797` and `clc7c0` tables
//! (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, MIT). Paths are relative to
//! the reference folder (`tools/refs-path.sh`).
//!
//! # Uploads and start tables
//!
//! `LOAD_MME_INSTRUCTION_RAM_POINTER` (method 0x45) sets a code store's upload offset to three
//! words per instruction (an offset of [`CODE_INSTRUCTIONS`] or more points past the end, so
//! later words are dropped); each `LOAD_MME_INSTRUCTION_RAM` word (0x46) is stored at the offset
//! if it is inside the store, the offset advances, and the store's code generation moves (also
//! for a dropped word, as `AddCode` bumps it). `LOAD_MME_START_ADDRESS_RAM` (0x48) stores its
//! word in the start table entry that `LOAD_MME_START_ADDRESS_RAM_POINTER` (0x47) held before
//! the word, if that is below [`START_ENTRIES`]; the register file advances the pointer
//! (slice G9). Every one of these words also goes to the register file and on to the ring as
//! any other write. The values used are the register's after the write, so the 3D class's
//! shadow RAM applies to them (Eden-NX2's `FrontWrite` reads them from its registers). The
//! methods are the same numbers in both classes ([`takes`] lists them).
//!
//! Each process has two code stores with their own start tables and memo, one for each engine
//! ([`Engine`]), as both references' `NativeMacroContext` does. A channel's compute class uses
//! the 3D store and start table until it writes `LOAD_MME_INSTRUCTION_RAM_POINTER` or
//! `LOAD_MME_START_ADDRESS_RAM_POINTER` itself, and its own from that word on (`FrontApply`,
//! `compute_uploaded`); the switch is per channel.
//!
//! Uploads are recorded per channel ([`ProcessMacros::uploads`],
//! [`ProcessMacros::code_writer`]), and a code word or start entry that one channel rewrites with
//! a value different from the one another channel wrote is counted
//! ([`ProcessMacros::cross_channel_rewrites`]) and logged the first time in the process, so
//! verification item 4 (per-channel or per-process code) can be read from a run.
//!
//! # Calls
//!
//! Words to `CALL_MME_MACRO(j)` (method 0xE00 + 2j) and `CALL_MME_DATA(j)` (0xE01 + 2j), and
//! every method above them, are a macro call's parameters, in both classes (the generated
//! `clc7c0` table has `CALL_MME_MACRO` at byte 0x3800, the same as `clc797`). The first word of
//! a call fixes its macro: entry `(method - 0xE00) / 2` of the start table, modulo
//! [`START_ENTRIES`] (`CallMacroMethod`); every later word, whatever its method, is the next
//! parameter. The call runs when the word's header owes no more ([`EngineWord::remaining`] is
//! 0), even if its words came in several GP entries or submissions (the channel keeps the
//! header's state across them, slice G8). The 3D and compute classes collect calls separately.
//!
//! # A run
//!
//! A call runs the interpreter from instruction `start` of its code store, through the store's
//! memo, synchronised with the store's code generation, with a [`StepLimit::DEFAULT`]
//! (`crate::mme::StepLimit`) step limit. The host the interpreter sees:
//!
//! - `load` gives the call's parameters (the memo serves them), then the words the run's MME DMA
//!   reads appended, in order;
//! - `state` reads the register file of the call's class, including the run's own writes so
//!   far (0 past the file, as Eden-NX2's reads of methods past its registers);
//! - `method` writes into the same register file as a stream word would (shadow RAM, side
//!   effects, same-value dropping), and through the same upload and bind rules;
//! - `barrier` performs the run's pending MME DMA reads (3D only, below).
//!
//! **A run's writes are committed only when it finishes.** Its register writes change the file
//! at once, so its own `state` reads see them, but their ring entries are held, and the file
//! words they changed are journaled. Its uploads and binds are held too, and its data RAM
//! writes are journaled by the memo. When the run finishes, its entries go to the sink in
//! order and its uploads and binds are applied in order; when it fails or is abandoned for a
//! catch-up, the file and the data RAM are put back and nothing is sent or applied
//! ([`CallResult`]). Neither the running macro's code nor its start table entry changes while it
//! runs, even if it uploads code.
//!
//! A run fails ([`CallFailure`]) on the first of: an interpreter error, a write to a method
//! outside its class's register file (Eden-NX2's "unsupported emitted method"), a DMA read that
//! cannot be read, DMA reads past [`DMA_RUN_MAX_WORDS`], or more than [`MAX_RUN_WRITES`] writes.
//! After the host's own failure the run is starved (no more loads; writes ignored) and ends on
//! the interpreter's next error or its end; the host's failure is reported, as it came first.
//! Starving only shortens the run: what it does after the failure is discarded either way.
//! A failed call is logged once per code store and start (with its parameters), and
//! [`CallResult::Failed`] says whether this was the first; all starts at or past
//! [`CODE_INSTRUCTIONS`] count as one, so the record is bounded.
//!
//! # MME DMA reads (3D, row 58)
//!
//! A 3D macro's write to `MME_DMA_READ_FIFOED` (method 0x158) queues a read of `LENGTH` words,
//! at most [`DMA_READ_MAX_WORDS`], from the GPU address in `SET_MME_MEM_ADDRESS_A` (`UPPER`,
//! 16:0, as bits 48:32) and `SET_MME_MEM_ADDRESS_B` (0x154 and 0x155) as the register file
//! holds them at that write. A read of 0 words queues nothing. At the next barrier:
//!
//! - if reads are queued and the back has work it has not performed ([`BackWork`]), the run is
//!   abandoned: nothing is committed, the call waits on its channel
//!   ([`MacroChannel::is_waiting`]; the channel's sink holds the pusher at the word after the
//!   call, [`crate::channel::ChannelSink::holds`]), and the front (slice G14) calls
//!   [`ProcessMacros::resume`] once the back reports that it has performed everything sent
//!   before it. The rerun reads at once, without asking the back again: the channel has sent
//!   nothing since, and the abandoned run's writes were never sent;
//! - otherwise the reads happen at once, in queue order, through [`GpuMemory`].
//!
//! The query is made only at a barrier with reads queued.
//!
//! # Divergences
//!
//! Each is pinned by the named test in `macros/tests.rs`.
//!
//! - **Writes are buffered, and a failed run commits nothing** (the spec). Eden-NX2 applies
//!   each write as the macro makes it and keeps them when the run fails
//!   (`a_failing_run_commits_nothing`, `a_failed_replay_commits_nothing`).
//! - **DMA reads happen at the barrier**, from the address queued at the write. Eden-NX2 reads
//!   at the `MME_DMA_READ_FIFOED` write itself, so a macro that loads the words without a
//!   barrier gets them there and underflows here (`dma_words_arrive_only_at_a_barrier`).
//! - **A DMA read that cannot be read fails the call**; Eden-NX2's `MemoryManager::Read` gives
//!   words without a failure result (`an_unreadable_dma_read_fails_the_call`).
//! - **`SET_MME_MEM_ADDRESS_A` gives 17 bits** (`UPPER`, 16:0, the header's field; decision 10).
//!   Eden-NX2 masks it with `0x1ffffff` (`the_dma_address_uses_the_headers_upper_field`).
//! - **A run's DMA reads fetch at most [`DMA_RUN_MAX_WORDS`] in all**, and a run makes at most
//!   [`MAX_RUN_WRITES`] writes; Eden-NX2 bounds neither (`dma_reads_are_bounded_per_run`,
//!   `writes_are_bounded_per_run`). Both are **guesses** at bounds no real macro reaches, to
//!   bound what a hostile one can make the front hold; a trace of the largest real call would
//!   confirm them.
//! - **`MME_DMA_READ` (0x157, into data RAM at `SET_MME_DATA_RAM_ADDRESS`) is not performed**:
//!   it goes to the register file as any write. Row 58 names only the load-FIFO form, and the
//!   slice covers that; Eden-NX2 performs both (`dma_read_into_data_ram_is_not_performed`).
//! - **A call that starts on a `CALL_MME_DATA` method runs the macro below it** in both
//!   classes, as Eden-NX2's 3D front does; its compute class drops such a word
//!   (`a_call_may_start_on_a_data_method`).
//! - **A macro branching to before its start fails** (`BranchOutOfRange`): the interpreter
//!   runs the store from the start as its own program, as Mesa's simulator runs one macro.
//!   Eden-NX2 runs within the whole code RAM (`a_branch_before_the_start_fails`).
//! - **Compute calls run through the interpreter and the memo on every host.** Eden-NX2's
//!   split front runs them only with its x86-64 JIT and skips them elsewhere (`FrontRunMacro`'s
//!   `#else`) (`compute_uses_the_3d_store_until_it_uploads` runs compute calls).
//!
//! # Departures from the approved spec (owner questions)
//!
//! - **The macro upload offset is not a register.** The spec's "Engine methods: the register
//!   file" lists, among the side effects computed on the front, "the macro upload pointer".
//!   Both references keep it inside the code store as a word offset (`native_write_offset =
//!   pointer * 3`, `MacroEngine::ClearCode` in `eden-nx2/src/video_core/macro.h`, and the same
//!   in `eden-pr4588/src/video_core/macro.h`), and leave method 0x45 as written. A Turing
//!   instruction is three words, so no register value can hold the offset; this module keeps
//!   it in the store, and the register file (slice G9) leaves 0x45 alone
//!   (`the_upload_offset_lives_in_the_code_store`).
//! - **Two code stores per process.** The spec's "Macros" says: "The front holds the code and
//!   the 3D and compute engines' start tables", and that Eden-NX2 "keeps one code store for all
//!   of a process's channels (row 53); this design does the same". Eden-NX2's (and PR #4588's)
//!   `NativeMacroContext` holds one store for 3D and one for compute, each shared by every
//!   channel, with the compute fallback above. This module does what Eden-NX2 does, reading
//!   "the same" as that; a single store for both classes would let a compute upload overwrite
//!   3D macros (`compute_uses_the_3d_store_until_it_uploads`).
//!
//! # Not here
//!
//! The front loop (slice G14) owns the register files, the channels and the ring: it routes a
//! channel's words for the methods [`takes`] names through [`ProcessMacros::write`] (and may
//! batch the others straight into [`RegFile::write_run`]), implements
//! [`crate::channel::ChannelSink::holds`] with [`MacroChannel::is_waiting`], answers
//! [`BackWork`] from the ring, and calls [`ProcessMacros::resume`] after the back's report (a
//! `Marker`). It also calls [`ProcessMacros::forget_channel`] when a channel is destroyed. A
//! macro JIT, the older Fermi macro processor, and counters for the run summary come later.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use thiserror::Error;

use crate::channel::EngineWord;
use crate::classes::{Method, clc7c0, clc797};
use crate::field::Field;
use crate::interface::ChannelId;
use crate::memory::{GpuFault, GpuMemory};
use crate::mme::{
    CallOutcome, DataRam, INSTRUCTION_WORDS, Instruction, MacroError, MacroHost, MacroMemo,
};
use crate::regs::{EntrySink, Journal, RegFile, RegsError, WORD_BYTES};
use crate::ring::{ClassId, RingEntry};

/// Words in one code store: Eden-NX2's `MacroEngine::native_code` (`std::array<u32, 0x3000>`,
/// `eden-nx2/src/video_core/macro.h`; the same in `eden-pr4588/src/video_core/macro.h`).
pub const CODE_WORDS: usize = 0x3000;

/// Instructions in one code store: [`CODE_WORDS`] over three words each. Also the bound of an
/// upload pointer (`ClearCode`: `method < 0x1000 ? method * 3 : native_code.size()`).
pub const CODE_INSTRUCTIONS: usize = CODE_WORDS / INSTRUCTION_WORDS;

/// Entries in one start table: Eden-NX2's `NativeMacroContext::positions` and
/// `compute_positions` (`std::array<u32, 0x80>`, `eden-nx2/src/video_core/macro.h`). Mesa's
/// `class_parser.py` gives `CALL_MME_MACRO` 256 elements, so calls `j` and `j + 0x80` share an
/// entry, as Eden-NX2's `% positions.size()` makes them.
pub const START_ENTRIES: usize = 0x80;

/// The most words one `MME_DMA_READ_FIFOED` reads: Eden-NX2's `std::min<u32>(value, 0x1000)`
/// (`MacroEngine::Execute`, `eden-nx2/src/video_core/macro.cpp`).
pub const DMA_READ_MAX_WORDS: u32 = 0x1000;

/// The most words one run's DMA reads may fetch in all: one maximal read (**guess**; module
/// docs, "Divergences").
pub const DMA_RUN_MAX_WORDS: u32 = DMA_READ_MAX_WORDS;

/// The most method writes one run may make (**guess**, 2^18; module docs, "Divergences").
pub const MAX_RUN_WRITES: usize = 1 << 18;

/// `SET_MME_MEM_ADDRESS_A` (3D method 0x154): the upper address bits, `UPPER` (16:0).
const MEM_ADDRESS_A: u32 = clc797::SET_MME_MEM_ADDRESS_A.method();
/// `SET_MME_MEM_ADDRESS_B` (3D method 0x155): the lower address bits, `LOWER` (31:0).
const MEM_ADDRESS_B: u32 = clc797::SET_MME_MEM_ADDRESS_B.method();
/// `MME_DMA_READ_FIFOED` (3D method 0x158): `LENGTH` (31:0) words into the load FIFO.
const DMA_READ_FIFOED: u32 = clc797::MME_DMA_READ_FIFOED.method();
/// Where `SET_MME_MEM_ADDRESS_A`'s bits go in the address.
const MEM_ADDRESS_UPPER_SHIFT: u32 = u32::BITS;

/// The engines that run macros.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Engine {
    /// The 3D class (`AMPERE_B`, `0xC797`).
    ThreeD,
    /// The compute class (`AMPERE_COMPUTE_B`, `0xC7C0`).
    Compute,
}

/// The methods one engine's macros use, from its class table.
struct EngineMethods {
    /// `CALL_MME_MACRO`: the first macro method.
    call: &'static Method,
    /// `LOAD_MME_INSTRUCTION_RAM_POINTER`.
    instruction_pointer: u32,
    /// `LOAD_MME_INSTRUCTION_RAM`.
    instruction: u32,
    /// `LOAD_MME_START_ADDRESS_RAM_POINTER`.
    start_pointer: u32,
    /// `LOAD_MME_START_ADDRESS_RAM`.
    start: u32,
}

const THREE_D_METHODS: EngineMethods = EngineMethods {
    call: &clc797::CALL_MME_MACRO,
    instruction_pointer: clc797::LOAD_MME_INSTRUCTION_RAM_POINTER.method(),
    instruction: clc797::LOAD_MME_INSTRUCTION_RAM.method(),
    start_pointer: clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
    start: clc797::LOAD_MME_START_ADDRESS_RAM.method(),
};

const COMPUTE_METHODS: EngineMethods = EngineMethods {
    call: &clc7c0::CALL_MME_MACRO,
    instruction_pointer: clc7c0::LOAD_MME_INSTRUCTION_RAM_POINTER.method(),
    instruction: clc7c0::LOAD_MME_INSTRUCTION_RAM.method(),
    start_pointer: clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
    start: clc7c0::LOAD_MME_START_ADDRESS_RAM.method(),
};

impl Engine {
    /// Both engines, in declaration order.
    pub const ALL: [Self; 2] = [Self::ThreeD, Self::Compute];

    /// The engine of class `class`, if it runs macros.
    pub const fn of(class: ClassId) -> Option<Self> {
        match class.0 {
            clc797::CLASS => Some(Self::ThreeD),
            clc7c0::CLASS => Some(Self::Compute),
            _ => None,
        }
    }

    /// The engine's class.
    pub const fn class(self) -> ClassId {
        match self {
            Self::ThreeD => ClassId(clc797::CLASS),
            Self::Compute => ClassId(clc7c0::CLASS),
        }
    }

    /// The engine's name, for logs.
    pub const fn name(self) -> &'static str {
        match self {
            Self::ThreeD => "3D",
            Self::Compute => "compute",
        }
    }

    /// The first macro method, `CALL_MME_MACRO(0)` (0xE00 in both classes).
    pub const fn first_macro_method(self) -> u32 {
        self.methods().call.method()
    }

    /// The methods one call spans: `CALL_MME_MACRO`'s stride (8 bytes, a macro and a data
    /// method).
    const fn call_stride(self) -> u32 {
        match self.methods().call.strides {
            [stride] if *stride >= WORD_BYTES => *stride / WORD_BYTES,
            _ => 1,
        }
    }

    const fn methods(self) -> &'static EngineMethods {
        match self {
            Self::ThreeD => &THREE_D_METHODS,
            Self::Compute => &COMPUTE_METHODS,
        }
    }

    /// The start table entry a call that starts on `method` uses (`CallMacroMethod`); `None`
    /// below the macro methods.
    pub fn entry(self, method: u32) -> Option<usize> {
        let offset = method.checked_sub(self.first_macro_method())?;
        let call = usize::try_from(offset / self.call_stride()).ok()?;
        Some(call % START_ENTRIES)
    }
}

/// Whether the front must send a word for `method` of class `class` through
/// [`ProcessMacros::write`]: a macro method, or one of the four `LOAD_MME_*` methods, of the 3D
/// or compute class. Words for every other method may go to the register file directly.
pub fn takes(class: ClassId, method: u32) -> bool {
    let Some(engine) = Engine::of(class) else {
        return false;
    };
    let methods = engine.methods();
    method >= engine.first_macro_method()
        || [
            methods.instruction_pointer,
            methods.instruction,
            methods.start_pointer,
            methods.start,
        ]
        .contains(&method)
}

/// What the macro host asks of the back thread.
pub trait BackWork {
    /// Whether the back has been sent work it has not performed yet: a DMA read now could miss
    /// a write that work makes (spec, "Macros").
    fn has_unperformed_work(&self) -> bool;
}

/// What a run reads through: guest memory for MME DMA reads, and the back's [`BackWork`].
#[derive(Clone, Copy, Debug)]
pub struct MacroEnv<'a, M: ?Sized, B: ?Sized> {
    /// The channel's GPU address space.
    pub memory: &'a M,
    /// The back thread.
    pub back: &'a B,
}

/// What a channel uploaded into one code store of its process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChannelUploads {
    /// Code words stored (words past the store's end are not counted).
    pub code_words: u64,
    /// Start table entries stored (binds past the table are not counted).
    pub start_entries: u64,
}

/// Why a run committed nothing.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CallFailure {
    /// The interpreter failed.
    #[error("the macro failed: {0}")]
    Macro(MacroError),
    /// The macro wrote a method outside its class's register file.
    #[error("the macro wrote method {method:#x}, which is outside its class's register file")]
    MethodOutOfRange {
        /// The method.
        method: u16,
    },
    /// An MME DMA read could not be read.
    #[error("the macro's DMA read of {words} words at GPU address {address:#x} failed: {fault}")]
    DmaRead {
        /// The GPU address.
        address: u64,
        /// The words asked for.
        words: u32,
        /// Why.
        fault: GpuFault,
    },
    /// The run's DMA reads asked for more than [`DMA_RUN_MAX_WORDS`] words in all.
    #[error(
        "the macro's DMA reads asked for {words} words in all, more than the {limit} one call may read",
        limit = DMA_RUN_MAX_WORDS
    )]
    DmaTooLarge {
        /// The words asked for, counting the read refused.
        words: u32,
    },
    /// The run made more than [`MAX_RUN_WRITES`] writes.
    #[error("the macro made more than {limit} method writes", limit = MAX_RUN_WRITES)]
    TooManyWrites,
}

/// What became of a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallResult {
    /// The run finished and its effects were committed; what the memo did.
    Committed(CallOutcome),
    /// The run failed and nothing was committed.
    Failed {
        /// Why.
        error: CallFailure,
        /// Whether this is the first failed call of this macro (code store and start) in the
        /// process: only that one is logged.
        first_for_macro: bool,
    },
    /// The run needed a DMA read while the back had work to perform: nothing was committed,
    /// and the call waits on its channel for [`ProcessMacros::resume`].
    CatchUp,
}

/// A call that ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallReport {
    /// The class that called it.
    pub engine: Engine,
    /// The code store it ran from (the 3D store for a compute call before the channel's compute
    /// class uploads).
    pub store: Engine,
    /// Its start table entry.
    pub entry: usize,
    /// Its first instruction in the store.
    pub start: u32,
    /// What became of it.
    pub result: CallResult,
}

/// What a word did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordOutcome {
    /// It went to the register file.
    Register,
    /// It was a parameter of a call still collecting.
    Parameter,
    /// It was a call's last parameter, and the call ran.
    Call(CallReport),
}

/// A call the host refuses; nothing changes.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum HostError {
    /// The channel waits for a catch-up: it takes no word until [`ProcessMacros::resume`].
    #[error("the channel's macro call waits for the back to catch up")]
    Waiting,
    /// [`ProcessMacros::resume`] on a channel with no call waiting.
    #[error("the channel has no macro call waiting")]
    NotWaiting,
    /// The register file is not the one of the word's (or the waiting call's) channel and class.
    #[error(
        "the register file of class {file_class:#x} on channel {file_channel} does not belong to class {class:#x} on channel {channel}"
    )]
    WrongFile {
        /// The file's channel.
        file_channel: u32,
        /// The file's class.
        file_class: u32,
        /// The channel.
        channel: u32,
        /// The class.
        class: u32,
    },
    /// The register file refused the word.
    #[error("{0}")]
    Regs(RegsError),
}

/// One code store: its words, decoded, the upload offset, its start table and its memo.
#[derive(Debug)]
struct CodeStore {
    words: Box<[u32]>,
    decoded: Box<[Instruction]>,
    /// The next word an upload stores (`native_write_offset`).
    write_offset: usize,
    /// Moved by every instruction word uploaded (`native_code_generation`).
    generation: u64,
    starts: Box<[u32]>,
    memo: MacroMemo,
    /// The channel that last stored each word.
    word_writers: Box<[Option<ChannelId>]>,
    /// The channel that last stored each start table entry.
    start_writers: Box<[Option<ChannelId>]>,
}

impl CodeStore {
    fn new() -> Self {
        Self {
            words: vec![0; CODE_WORDS].into_boxed_slice(),
            decoded: vec![Instruction::decode([0; INSTRUCTION_WORDS]); CODE_INSTRUCTIONS]
                .into_boxed_slice(),
            write_offset: 0,
            generation: 0,
            starts: vec![0; START_ENTRIES].into_boxed_slice(),
            memo: MacroMemo::new(),
            word_writers: vec![None; CODE_WORDS].into_boxed_slice(),
            start_writers: vec![None; START_ENTRIES].into_boxed_slice(),
        }
    }

    /// Decodes the instruction that holds word `offset` again.
    fn redecode(&mut self, offset: usize) {
        let first = offset - offset % INSTRUCTION_WORDS;
        let Some(words) = first
            .checked_add(INSTRUCTION_WORDS)
            .and_then(|end| self.words.get(first..end))
        else {
            return;
        };
        let Ok(words) = <[u32; INSTRUCTION_WORDS]>::try_from(words) else {
            return;
        };
        if let Some(slot) = self.decoded.get_mut(first / INSTRUCTION_WORDS) {
            *slot = Instruction::decode(words);
        }
    }
}

/// An upload or bind a word makes, applied at once for a stream word and at commit for a
/// macro's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UploadOp {
    engine: Engine,
    kind: UploadKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadKind {
    /// `LOAD_MME_INSTRUCTION_RAM_POINTER`, with the register's value.
    Pointer(u32),
    /// `LOAD_MME_INSTRUCTION_RAM`, with the register's value.
    Instruction(u32),
    /// `LOAD_MME_START_ADDRESS_RAM_POINTER` (the register file keeps the pointer).
    StartPointer,
    /// `LOAD_MME_START_ADDRESS_RAM`: the pointer before the word and the register's value.
    Start { entry: u32, start: u32 },
}

impl UploadOp {
    /// The upload a write to `method` of `engine` made, given the start pointer before it and
    /// the file after it.
    fn of(engine: Engine, method: u32, start_pointer: u32, file: &RegFile) -> Option<Self> {
        let methods = engine.methods();
        let value = || file.get(method).unwrap_or(0);
        let kind = if method == methods.instruction_pointer {
            UploadKind::Pointer(value())
        } else if method == methods.instruction {
            UploadKind::Instruction(value())
        } else if method == methods.start_pointer {
            UploadKind::StartPointer
        } else if method == methods.start {
            UploadKind::Start {
                entry: start_pointer,
                start: value(),
            }
        } else {
            return None;
        };
        Some(Self { engine, kind })
    }

    /// Whether the op switches a channel's compute class to its own store (`FrontApply`:
    /// `method == 0x45 || method == 0x47`).
    fn switches_compute(self) -> bool {
        self.engine == Engine::Compute
            && matches!(self.kind, UploadKind::Pointer(_) | UploadKind::StartPointer)
    }
}

/// A queued MME DMA read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DmaRead {
    address: u64,
    words: u32,
}

/// The scratch of the run in progress, kept for its capacity.
#[derive(Debug, Default)]
struct RunScratch {
    /// The run's ring entries, held until it commits.
    entries: Vec<RingEntry<'static>>,
    /// The register file words the run changed.
    journal: Journal,
    /// The run's uploads and binds, held until it commits.
    ops: Vec<UploadOp>,
    /// The DMA reads queued and not yet performed.
    reads: Vec<DmaRead>,
    /// The words DMA reads appended to the load FIFO.
    fifo: Vec<u32>,
    /// A DMA read's bytes.
    bytes: Vec<u8>,
}

impl RunScratch {
    fn clear(&mut self) {
        self.entries.clear();
        self.journal.clear();
        self.ops.clear();
        self.reads.clear();
        self.fifo.clear();
    }
}

/// One process's macro code, start tables, upload record and memo (module docs).
#[derive(Debug)]
pub struct ProcessMacros {
    stores: [CodeStore; 2],
    uploads: BTreeMap<(ChannelId, Engine), ChannelUploads>,
    rewrites: u64,
    /// Code stores and starts whose failure was logged (starts at or past
    /// [`CODE_INSTRUCTIONS`] as one).
    failures_logged: BTreeSet<(Engine, u32)>,
    run: RunScratch,
}

impl Default for ProcessMacros {
    fn default() -> Self {
        Self::new()
    }
}

/// A call whose parameters are arriving, or waiting for a catch-up.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Call {
    /// The method of its first word; `None` while no call is collecting.
    method: Option<u16>,
    parameters: Vec<u32>,
}

/// One channel's macro state (module docs).
#[derive(Clone, Debug)]
pub struct MacroChannel {
    id: ChannelId,
    data_ram: DataRam,
    compute_uploaded: bool,
    /// Per engine, the call collecting parameters (and its buffer, kept for its capacity).
    collecting: [Call; 2],
    /// The call waiting for a catch-up, and its engine.
    waiting: Option<(Engine, Call)>,
}

impl MacroChannel {
    /// A channel with zeroed data RAM, nothing collecting or waiting, and its compute class on
    /// the 3D code store.
    pub fn new(id: ChannelId) -> Self {
        Self {
            id,
            data_ram: DataRam::new(),
            compute_uploaded: false,
            collecting: [Call::default(), Call::default()],
            waiting: None,
        }
    }

    /// The channel.
    pub const fn id(&self) -> ChannelId {
        self.id
    }

    /// The channel's data RAM, shared by its 3D and compute classes.
    pub const fn data_ram(&self) -> &DataRam {
        &self.data_ram
    }

    /// Whether a call waits for a catch-up: the channel's pusher must hold until
    /// [`ProcessMacros::resume`] has run it.
    pub const fn is_waiting(&self) -> bool {
        self.waiting.is_some()
    }

    /// Whether the channel's compute class uses its own code store and start table.
    pub const fn compute_uploaded(&self) -> bool {
        self.compute_uploaded
    }

    /// The parameters collected so far for `engine`'s call.
    pub fn collected(&self, engine: Engine) -> &[u32] {
        let [three_d, compute] = &self.collecting;
        match engine {
            Engine::ThreeD => &three_d.parameters,
            Engine::Compute => &compute.parameters,
        }
    }

    /// The call collecting for `engine`.
    fn collecting_mut(&mut self, engine: Engine) -> &mut Call {
        let [three_d, compute] = &mut self.collecting;
        match engine {
            Engine::ThreeD => three_d,
            Engine::Compute => compute,
        }
    }

    /// The store `engine`'s calls and uploads use on this channel.
    const fn store_for(&self, engine: Engine) -> Engine {
        match engine {
            Engine::Compute if self.compute_uploaded => Engine::Compute,
            _ => Engine::ThreeD,
        }
    }
}

/// Where a run stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Running,
    Abandoned,
    Failed(CallFailure),
}

/// The host the interpreter runs a call against.
struct RunHost<'a, M: ?Sized, B: ?Sized> {
    engine: Engine,
    file: &'a mut RegFile,
    scratch: &'a mut RunScratch,
    memory: &'a M,
    back: &'a B,
    /// The call is a rerun after a catch-up: DMA reads happen without asking the back.
    caught_up: bool,
    status: RunStatus,
    writes: usize,
    /// DMA words queued so far, performed or not.
    dma_words: u32,
    /// The next FIFO word to load.
    next: usize,
}

impl<M: GpuMemory + ?Sized, B: BackWork + ?Sized> RunHost<'_, M, B> {
    fn fail(&mut self, failure: CallFailure) {
        if self.status == RunStatus::Running {
            self.status = RunStatus::Failed(failure);
        }
    }

    fn queue_dma(&mut self, length: u32) {
        let words = DMA_READ_FIFOED_LENGTH.get(length).min(DMA_READ_MAX_WORDS);
        if words == 0 {
            return;
        }
        let total = self.dma_words.saturating_add(words);
        if total > DMA_RUN_MAX_WORDS {
            self.fail(CallFailure::DmaTooLarge { words: total });
            return;
        }
        self.dma_words = total;
        let upper = MEM_ADDRESS_A_UPPER.get(self.file.get(MEM_ADDRESS_A).unwrap_or(0));
        let lower = MEM_ADDRESS_B_LOWER.get(self.file.get(MEM_ADDRESS_B).unwrap_or(0));
        let address = (u64::from(upper) << MEM_ADDRESS_UPPER_SHIFT) | u64::from(lower);
        self.scratch.reads.push(DmaRead { address, words });
    }

    fn perform_reads(&mut self) {
        for index in 0..self.scratch.reads.len() {
            let Some(read) = self.scratch.reads.get(index).copied() else {
                break;
            };
            // At most `DMA_READ_MAX_WORDS` words: the product fits.
            let bytes = usize::try_from(u64::from(read.words) * u64::from(WORD_BYTES)).unwrap_or(0);
            self.scratch.bytes.clear();
            self.scratch.bytes.resize(bytes, 0);
            if let Err(fault) = self.memory.read(read.address, &mut self.scratch.bytes) {
                self.fail(CallFailure::DmaRead {
                    address: read.address,
                    words: read.words,
                    fault,
                });
                return;
            }
            let (words, _) = self.scratch.bytes.as_chunks::<4>();
            self.scratch
                .fifo
                .extend(words.iter().map(|&bytes| u32::from_le_bytes(bytes)));
        }
        self.scratch.reads.clear();
    }
}

/// `SET_MME_MEM_ADDRESS_A_UPPER` (16:0).
const MEM_ADDRESS_A_UPPER: Field = clc797::SET_MME_MEM_ADDRESS_A_UPPER;
/// `SET_MME_MEM_ADDRESS_B_LOWER` (31:0).
const MEM_ADDRESS_B_LOWER: Field = clc797::SET_MME_MEM_ADDRESS_B_LOWER;
/// `MME_DMA_READ_FIFOED_LENGTH` (31:0).
const DMA_READ_FIFOED_LENGTH: Field = clc797::MME_DMA_READ_FIFOED_LENGTH;

impl<M: GpuMemory + ?Sized, B: BackWork + ?Sized> MacroHost for RunHost<'_, M, B> {
    fn load(&mut self) -> Option<u32> {
        if self.status != RunStatus::Running {
            return None;
        }
        let word = self.scratch.fifo.get(self.next).copied()?;
        self.next = self.next.saturating_add(1);
        Some(word)
    }

    fn state(&mut self, method: u16) -> u32 {
        self.file.get(u32::from(method)).unwrap_or(0)
    }

    fn method(&mut self, method: u16, value: u32) {
        if self.status != RunStatus::Running {
            return;
        }
        if self.writes >= MAX_RUN_WRITES {
            self.fail(CallFailure::TooManyWrites);
            return;
        }
        self.writes = self.writes.saturating_add(1);
        let number = u32::from(method);
        let start_pointer = self
            .file
            .get(self.engine.methods().start_pointer)
            .unwrap_or(0);
        self.file.save(number, &mut self.scratch.journal);
        if self
            .file
            .write(number, value, &mut self.scratch.entries)
            .is_err()
        {
            self.fail(CallFailure::MethodOutOfRange { method });
            return;
        }
        if let Some(op) = UploadOp::of(self.engine, number, start_pointer, self.file) {
            self.scratch.ops.push(op);
        }
        if self.engine == Engine::ThreeD && number == DMA_READ_FIFOED {
            self.queue_dma(value);
        }
    }

    fn barrier(&mut self) {
        if self.status != RunStatus::Running || self.scratch.reads.is_empty() {
            return;
        }
        if !self.caught_up && self.back.has_unperformed_work() {
            self.status = RunStatus::Abandoned;
            return;
        }
        self.perform_reads();
    }
}

/// Parameters as plain hex words, for the failure log.
struct Words<'a>(&'a [u32]);

impl fmt::Display for Words<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, word) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{word:#x}")?;
        }
        Ok(())
    }
}

impl ProcessMacros {
    /// Empty code stores and start tables (every word and entry 0, upload offsets 0), empty
    /// memos, and no uploads recorded.
    pub fn new() -> Self {
        Self {
            stores: [CodeStore::new(), CodeStore::new()],
            uploads: BTreeMap::new(),
            rewrites: 0,
            failures_logged: BTreeSet::new(),
            run: RunScratch::default(),
        }
    }

    /// Takes one engine data word of channel `channel` (module docs): a parameter of a macro
    /// call, which runs once its last word arrives, or a register write through `file`, which
    /// also uploads code or binds a start for the `LOAD_MME_*` methods. What the register file
    /// sends, and a committed call's writes, go to `out` in order.
    ///
    /// `file` must be the register file of the word's class on `channel`.
    ///
    /// # Errors
    ///
    /// Checked in this order, each changing nothing: [`HostError::Waiting`] while a call waits
    /// for a catch-up; [`HostError::WrongFile`]; [`HostError::Regs`] if the register file
    /// refuses the word.
    pub fn write<M, B, S>(
        &mut self,
        channel: &mut MacroChannel,
        file: &mut RegFile,
        word: EngineWord,
        env: &MacroEnv<'_, M, B>,
        out: &mut S,
    ) -> Result<WordOutcome, HostError>
    where
        M: GpuMemory + ?Sized,
        B: BackWork + ?Sized,
        S: EntrySink + ?Sized,
    {
        if channel.waiting.is_some() {
            return Err(HostError::Waiting);
        }
        check_file(file, channel.id, word.class)?;
        let method = u32::from(word.method);
        let engine = Engine::of(word.class);
        if let Some(engine) = engine
            && method >= engine.first_macro_method()
        {
            let call = channel.collecting_mut(engine);
            call.method.get_or_insert(word.method);
            call.parameters.push(word.value);
            if word.remaining > 0 {
                return Ok(WordOutcome::Parameter);
            }
            let call = std::mem::take(call);
            return Ok(WordOutcome::Call(
                self.run_call(channel, file, engine, call, env, out, false),
            ));
        }
        let start_pointer = engine.map_or(0, |engine| {
            file.get(engine.methods().start_pointer).unwrap_or(0)
        });
        file.write(method, word.value, out)
            .map_err(HostError::Regs)?;
        if let Some(engine) = engine
            && let Some(op) = UploadOp::of(engine, method, start_pointer, file)
        {
            self.apply(channel, op);
        }
        Ok(WordOutcome::Register)
    }

    /// Runs the call waiting on `channel` again, once the back has performed everything sent
    /// before it (module docs, "MME DMA reads"). Its DMA reads happen at once.
    ///
    /// # Errors
    ///
    /// Checked in this order, each changing nothing: [`HostError::NotWaiting`];
    /// [`HostError::WrongFile`] unless `file` is the register file of the waiting call's class
    /// on `channel`.
    pub fn resume<M, B, S>(
        &mut self,
        channel: &mut MacroChannel,
        file: &mut RegFile,
        env: &MacroEnv<'_, M, B>,
        out: &mut S,
    ) -> Result<CallReport, HostError>
    where
        M: GpuMemory + ?Sized,
        B: BackWork + ?Sized,
        S: EntrySink + ?Sized,
    {
        let Some((engine, _)) = channel.waiting else {
            return Err(HostError::NotWaiting);
        };
        check_file(file, channel.id, engine.class())?;
        let Some((engine, call)) = channel.waiting.take() else {
            return Err(HostError::NotWaiting);
        };
        Ok(self.run_call(channel, file, engine, call, env, out, true))
    }

    /// Words of `store`'s code; `None` past [`CODE_WORDS`].
    pub fn code_word(&self, store: Engine, offset: usize) -> Option<u32> {
        self.store(store).words.get(offset).copied()
    }

    /// The channel that last stored word `offset` of `store`.
    pub fn code_writer(&self, store: Engine, offset: usize) -> Option<ChannelId> {
        self.store(store)
            .word_writers
            .get(offset)
            .copied()
            .flatten()
    }

    /// `store`'s start table.
    pub fn starts(&self, store: Engine) -> &[u32] {
        &self.store(store).starts
    }

    /// `store`'s upload offset, in words.
    pub fn upload_offset(&self, store: Engine) -> usize {
        self.store(store).write_offset
    }

    /// `store`'s code generation: moved by every instruction word uploaded.
    pub fn code_generation(&self, store: Engine) -> u64 {
        self.store(store).generation
    }

    /// What `channel` uploaded into `store` (zero for a channel that uploaded nothing).
    pub fn uploads(&self, channel: ChannelId, store: Engine) -> ChannelUploads {
        self.uploads
            .get(&(channel, store))
            .copied()
            .unwrap_or_default()
    }

    /// How many times a channel stored a code word or start entry, in either store, that
    /// another channel had stored with a different value.
    pub const fn cross_channel_rewrites(&self) -> u64 {
        self.rewrites
    }

    /// Forgets `channel`'s upload record ([`ProcessMacros::uploads`]), for a destroyed
    /// channel. The writers of code words and start entries stay as they are.
    pub fn forget_channel(&mut self, channel: ChannelId) {
        for store in Engine::ALL {
            self.uploads.remove(&(channel, store));
        }
    }

    fn store(&self, store: Engine) -> &CodeStore {
        let [three_d, compute] = &self.stores;
        match store {
            Engine::ThreeD => three_d,
            Engine::Compute => compute,
        }
    }

    /// Applies one upload or bind of `channel`'s.
    fn apply(&mut self, channel: &mut MacroChannel, op: UploadOp) {
        if op.switches_compute() {
            channel.compute_uploaded = true;
        }
        let store_engine = channel.store_for(op.engine);
        let store = store_mut(&mut self.stores, store_engine);
        let id = channel.id;
        let mut stored = ChannelUploads::default();
        let mut rewrote = false;
        match op.kind {
            UploadKind::Pointer(pointer) => {
                store.write_offset = usize::try_from(pointer)
                    .ok()
                    .filter(|&pointer| pointer < CODE_INSTRUCTIONS)
                    .map_or(CODE_WORDS, |pointer| pointer * INSTRUCTION_WORDS);
            }
            UploadKind::Instruction(word) => {
                store.generation = store.generation.wrapping_add(1);
                let offset = store.write_offset;
                if let Some(slot) = store.words.get_mut(offset) {
                    let old = std::mem::replace(slot, word);
                    if let Some(writer) = store.word_writers.get_mut(offset) {
                        rewrote = writer.is_some_and(|writer| writer != id) && old != word;
                        *writer = Some(id);
                    }
                    store.redecode(offset);
                    store.write_offset = offset + 1;
                    stored.code_words = 1;
                }
            }
            UploadKind::StartPointer => {}
            UploadKind::Start { entry, start } => {
                if let Ok(entry) = usize::try_from(entry)
                    && let Some(slot) = store.starts.get_mut(entry)
                {
                    let old = std::mem::replace(slot, start);
                    if let Some(writer) = store.start_writers.get_mut(entry) {
                        rewrote = writer.is_some_and(|writer| writer != id) && old != start;
                        *writer = Some(id);
                    }
                    stored.start_entries = 1;
                }
            }
        }
        if stored != ChannelUploads::default() {
            let record = self.uploads.entry((id, store_engine)).or_default();
            record.code_words = record.code_words.saturating_add(stored.code_words);
            record.start_entries = record.start_entries.saturating_add(stored.start_entries);
        }
        if rewrote {
            if self.rewrites == 0 {
                tracing::warn!(
                    channel = id.0,
                    store = store_engine.name(),
                    "a channel rewrote macro code or a start another channel uploaded with a different value; later rewrites are only counted"
                );
            }
            self.rewrites = self.rewrites.saturating_add(1);
        }
    }

    /// Runs `call` of `engine` on `channel` and commits or discards its effects.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call's full context; a struct would only rename the same fields"
    )]
    fn run_call<M, B, S>(
        &mut self,
        channel: &mut MacroChannel,
        file: &mut RegFile,
        engine: Engine,
        call: Call,
        env: &MacroEnv<'_, M, B>,
        out: &mut S,
        caught_up: bool,
    ) -> CallReport
    where
        M: GpuMemory + ?Sized,
        B: BackWork + ?Sized,
        S: EntrySink + ?Sized,
    {
        let store_engine = channel.store_for(engine);
        let entry = call
            .method
            .and_then(|method| engine.entry(u32::from(method)))
            .unwrap_or(0);
        let Self { stores, run, .. } = self;
        let store = store_mut(stores, store_engine);
        let start = store.starts.get(entry).copied().unwrap_or(0);
        store.memo.synchronize(store.generation);
        let program = usize::try_from(start)
            .ok()
            .and_then(|start| store.decoded.get(start..))
            .unwrap_or(&[]);
        run.clear();
        let mut host = RunHost {
            engine,
            file,
            scratch: run,
            memory: env.memory,
            back: env.back,
            caught_up,
            status: RunStatus::Running,
            writes: 0,
            dma_words: 0,
            next: 0,
        };
        let outcome = store.memo.call(
            program,
            start,
            &call.parameters,
            &mut channel.data_ram,
            &mut host,
        );
        let status = host.status;
        let failure = match (status, outcome) {
            (RunStatus::Running, Ok(outcome)) => {
                // Commit: the file keeps the run's writes, the entries and uploads go out.
                run.journal.clear();
                for entry in run.entries.drain(..) {
                    out.send(entry);
                }
                let ops = std::mem::take(&mut run.ops);
                for op in &ops {
                    self.apply(channel, *op);
                }
                self.run.ops = ops;
                self.run.ops.clear();
                Self::recycle(channel, engine, call);
                return CallReport {
                    engine,
                    store: store_engine,
                    entry,
                    start,
                    result: CallResult::Committed(outcome),
                };
            }
            (RunStatus::Abandoned, _) => None,
            (RunStatus::Failed(failure), _) => Some(failure),
            (RunStatus::Running, Err(error)) => Some(CallFailure::Macro(error)),
        };
        // Discard: put the file and the data RAM back, send and apply nothing.
        file.undo(&mut run.journal);
        store.memo.undo_data_ram(&mut channel.data_ram);
        run.clear();
        let result = match failure {
            None => {
                channel.waiting = Some((engine, call));
                CallResult::CatchUp
            }
            Some(error) => {
                let past_code = u32::try_from(CODE_INSTRUCTIONS).unwrap_or(u32::MAX);
                let key = (store_engine, start.min(past_code));
                let first_for_macro = self.failures_logged.insert(key);
                if first_for_macro {
                    tracing::warn!(
                        engine = engine.name(),
                        store = store_engine.name(),
                        start,
                        %error,
                        parameters = %Words(&call.parameters),
                        "macro call failed and committed nothing; later failures of this macro are not logged"
                    );
                }
                Self::recycle(channel, engine, call);
                CallResult::Failed {
                    error,
                    first_for_macro,
                }
            }
        };
        CallReport {
            engine,
            store: store_engine,
            entry,
            start,
            result,
        }
    }

    /// Gives a finished call's parameter buffer back to its channel, for its capacity.
    fn recycle(channel: &mut MacroChannel, engine: Engine, mut call: Call) {
        let slot = channel.collecting_mut(engine);
        if slot.method.is_none() && slot.parameters.capacity() == 0 {
            call.method = None;
            call.parameters.clear();
            *slot = call;
        }
    }
}

/// One store of a process.
fn store_mut(stores: &mut [CodeStore; 2], store: Engine) -> &mut CodeStore {
    let [three_d, compute] = stores;
    match store {
        Engine::ThreeD => three_d,
        Engine::Compute => compute,
    }
}

/// Whether `file` is the register file of `class` on `channel`.
fn check_file(file: &RegFile, channel: ChannelId, class: ClassId) -> Result<(), HostError> {
    if file.channel() == channel && file.class() == class {
        return Ok(());
    }
    Err(HostError::WrongFile {
        file_channel: file.channel().0,
        file_class: file.class().0,
        channel: channel.0,
        class: class.0,
    })
}
