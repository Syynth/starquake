//! One GPU channel's pusher: the state machine that turns the words of GP entries into method
//! calls, runs the host methods, parks the channel at an unmet semaphore acquire and resumes it
//! at the exact word after it (gpu-protocol spec, slice G8: "The method stream", "Host methods",
//! "Semaphore acquires", "Split methods", "Malformed streams").
//!
//! Ported from `eden-nx2/src/video_core/dma_pusher.cpp` (`Step`, `ProcessCommands`, the
//! `command_resume_offset` resume) and `eden-nx2/src/video_core/engines/puller.cpp`
//! (`CallPullerMethod`, `ProcessBindMethod`, `ProcessSemaphoreTriggerMethod`,
//! `ProcessSemaphoreExecute`, `PendingAcquire::Matches`), both @ 50fa86bb, GPL-3.0-or-later.
//! Host method numbers and fields are those of `classes/clc76f.h`
//! (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, MIT), read through the
//! generated [`crate::classes::clc76f`].
//!
//! # What a channel does
//!
//! [`Channel`] holds one channel's pusher state: the GP entry it is reading and how far it has
//! read, the method still owed data words (method, subchannel, remaining count and mode), the
//! subchannel bindings, the host semaphore registers, and the acquire it is parked on. The front
//! loop (slice G14) owns the submission queues, the waiter table and the scheduling; it drives a
//! channel with these calls:
//!
//! - [`Channel::start_entry`] gives an idle channel its next GP entry.
//! - [`Channel::run`] reads and executes words until the entry ends, the channel parks on an
//!   acquire, the sink holds it ([`RunOutcome::Held`]: a macro call waits for a catch-up), or
//!   the quantum of words is used. Engine method words go to the
//!   [`ChannelSink`] as [`EngineWord`]s (the register files, slice G9, sit behind it); binds,
//!   releases, reductions, wait for idle and memory operations go to it as
//!   [`RingEntry`] values; dropped words and carried methods go to it as [`Note`]s.
//! - When `run` returns [`RunOutcome::Parked`], the front registers a waiter for the
//!   [`Acquire`] and *then* calls [`Channel::recheck`], which reads the semaphore again (spec,
//!   "Semaphore acquires", steps 1 and 2: registering first means a release in between is
//!   either seen by the second read or latches the waiter). Later wake-ups call `recheck`
//!   again, or [`Channel::resume_latched`] for a latched waiter, which resumes even if the word
//!   has moved on since (row 70).
//! - [`Channel::end_submission`] marks the end of a submission: a method still owed words is
//!   kept for the next submission and noted once ([`Note::MethodCarried`]).
//!
//! # Reading words
//!
//! Command words are read during the [`Channel::run`] call that executes them, never at
//! submission (row 50), through [`GpuMemory`] as copies, in chunks that end at the entry's end
//! or at the end of a page ([`starquake_memory::PAGE_SIZE`]), whichever is first. The exact
//! read-ahead window: when the next word to execute is not in the chunk, the channel reads
//! from it to the end of its page or of the entry, and executes from that copy until the
//! chunk runs out, an acquire is checked, the entry ends, or `run` returns
//! [`RunOutcome::QuantumUsed`] or [`RunOutcome::Held`]; each of these drops the rest of the
//! chunk. So no word read in
//! one `run` call is executed by a later one, whatever the front does in between (serving
//! other channels, changing the address table)
//! (`words_read_ahead_are_not_kept_across_run_calls`). Within one call, a word read ahead can
//! be up to a page older than its execution.
//!
//! A chunk never runs past the page holding the next word to execute, so a read fault is
//! raised by the word that needs the unreadable page, not by a word read ahead
//! (`an_unmapped_second_page_is_not_read_before_it_is_needed`).
//!
//! **After every acquire, met or not, the rest of the chunk is dropped and the words after the
//! acquire are read again.** The CPU writes a command buffer and then releases the acquire
//! that gates it (row 50), so words read before the acquire was checked may be stale even when
//! the check passes at once (`words_written_while_a_met_acquire_is_read_are_read_fresh`).
//! A parked channel's position already points past the acquire's word; resuming reads from
//! there (`a_parked_channel_resumes_at_the_word_after_the_acquire`).
//!
//! # Split methods
//!
//! Method state belongs to the channel, not to the GP entry (row 73): a method whose data words
//! run past the end of an entry continues in the channel's next entry, whatever its address or
//! level (`a_method_split_across_entries_continues_in_the_next_entry`), and in the next
//! submission's first entry. [`Channel::end_submission`] notes such a method once per method
//! header, however many submissions it spans
//! (`a_method_split_across_submissions_is_noted_once`).
//!
//! # Malformed streams
//!
//! Owner's decision 13 (c), 2026-11-12: skip unknown or unbound methods, stop the channel on
//! structural errors. The exact sets:
//!
//! - **The channel stops** ([`ChannelError`], returned by [`Channel::run`] and by every later
//!   call, sticky): a command word that cannot be read ([`ChannelError::ReadFault`]), a
//!   semaphore an acquire cannot read ([`ChannelError::SemaphoreFault`]), an entry whose
//!   address is not 4-byte aligned ([`ChannelError::UnalignedEntry`], from
//!   [`Channel::start_entry`]; only a hand-built [`GpEntry`] can hold one), an entry that runs
//!   past the GP entry's 40-bit address space ([`ChannelError::EntryPastAddressSpace`], from
//!   [`Channel::start_entry`]), and a method header whose data would run past method 0xFFF
//!   ([`ChannelError::MethodPastSpace`]).
//! - **The word is dropped** ([`DroppedWord`], reported as [`Note::Dropped`] and counted per
//!   [`DroppedKind`], and the channel goes on): a word that is not a method header
//!   ([`DroppedWord::BadHeader`]: opcode 6 and the undefined subdevice-mask and group-2 forms),
//!   a host method `clc76f.h` does not define ([`DroppedWord::UnknownHostMethod`]), an engine
//!   method on a subchannel with no class bound ([`DroppedWord::UnboundSubchannel`], row 49), a
//!   `SEMAPHORED` or `SEM_EXECUTE` word with an undefined operation, reduction or format
//!   ([`DroppedWord::BadSemaphore`]), and a `MEM_OP_D` with an undefined operation
//!   ([`DroppedWord::UndefinedMemOp`]).
//!
//! What to do about either (log once per kind, end the run under the strict policy,
//! `observability.md`) is the caller's: the channel reports each one, with whether it is the
//! first of its kind on this channel, and does not log.
//!
//! # Host methods
//!
//! A word to a method below [`HOST_METHODS_END`] goes to the host methods whatever its
//! subchannel and whatever is bound there (Eden's `ExecuteMethodOnEngine`). The set is
//! `clc76f.h`'s:
//!
//! | Method | Name | What the channel does |
//! |---|---|---|
//! | 0x0 | `SET_OBJECT` | binds `NVCLASS` (15:0) on the subchannel, or unbinds it for class 0; [`RingEntry::Bind`] |
//! | 0x4–0x6 | `SEMAPHOREA`–`C` | stores the address (A 7:0 high, B 31:2 low) and the payload |
//! | 0x7 | `SEMAPHORED` | acquire (on the front), release or reduction ([`RingEntry::Release`]) |
//! | 0xA–0xC | `MEM_OP_A`–`C` | accepted; their operands are not used yet |
//! | 0xD | `MEM_OP_D` | [`RingEntry::Invalidate`] with its `OPERATION` (31:27) |
//! | 0x14 | `SET_REFERENCE` | stores the count; [`ChannelSink::set_reference`] |
//! | 0x17–0x1A | `SEM_ADDR_LO`, `_HI`, `SEM_PAYLOAD_LO`, `_HI` | stores the address (HI 7:0, LO 31:2) and the 64-bit payload |
//! | 0x1B | `SEM_EXECUTE` | acquire (on the front), release or reduction ([`RingEntry::Release`]) |
//! | 0x1E | `WFI` | [`RingEntry::WaitForIdle`] |
//! | 0x1, 0x2, 0x8, 0x9, 0x20, 0x21 | `ILLEGAL`, `NOP`, `NON_STALL_INTERRUPT`, `FB_FLUSH`, `YIELD`, `CLEAR_FAULTED` | nothing |
//!
//! Every other method below 0x40 is [`DroppedWord::UnknownHostMethod`]. Pinned against the
//! header's byte addresses written out as literals in `host_method_numbers_match_clc76f`.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the named test in `channel/tests.rs`:
//!
//! - **`SEM_EXECUTE` acquires park the channel** (spec, "Semaphore acquires"). Eden-NX2's
//!   `ProcessSemaphoreExecute` checks once and continues (`sem_execute_acquire_parks_item_5`).
//! - **Words after a met acquire are read again.** Eden-NX2 reads an entry's words once per
//!   `Step` and goes on with them after an acquire that is met at once
//!   (`words_written_while_a_met_acquire_is_read_are_read_fresh`).
//! - **Class 0 unbinds the subchannel** (spec, "Host methods"). Eden-NX2's `ProcessBindMethod`
//!   returns early for class 0 and keeps the old binding (`class_zero_unbinds_the_subchannel`).
//! - **Only subchannel 2 starts bound**, to inline-to-memory (`0xA140`, row 49; spec, "Host
//!   methods"). Eden-NX2's `BindNvkDefaultSubchannels` (`control/channel_state.cpp`) also binds
//!   3D, compute, 2D and copy on subchannels 0, 1, 3 and 4 (`only_subchannel_2_starts_bound`).
//! - **Old-format method headers** (opcode 0 or 2 with tertiary opcode 0, [`MethodHeader::Legacy`])
//!   run as methods with their data words, as `clc76f.h` defines them; Eden-NX2's
//!   `ProcessCommands` skips modes 0 and 2 as single words (`old_format_headers_run_as_methods`).
//!   The all-zero `NOP` word is one with a count of 0 and does nothing either way. This is the
//!   spec's rule (next section).
//! - **Opcode 7 ends the entry** (a guess, below). Eden-NX2 skips the word
//!   (`end_segment_ends_the_entry_guess_item_9`).
//! - **Semaphore address widths follow the header** (decision 10): `SEMAPHOREA` and
//!   `SEM_ADDR_HI` give address bits 39:32 from their bits 7:0, and `SEMAPHOREB` gives bits
//!   31:2 (`OFFSET_LOWER`), so its bits 1:0 are dropped. Eden-NX2 shifts all 32 bits of
//!   `SEMAPHOREA` and 25 bits of `SEM_ADDR_HI` (`0x1FFFFFF`), and its `SemaphoreAddress()`
//!   (`engines/puller.h`) ORs in the whole of `SEMAPHOREB`; only its `SEM_ADDR_LO` path masks
//!   with `~3`, as this module does (`semaphore_address_upper_bits_follow_the_header`).
//! - **`MEM_OP_D` carries the memory operation**; Eden-NX2 invalidates its caches on
//!   `MEM_OP_B` and ignores `MEM_OP_D` (`mem_op_d_triggers_the_operation`).
//! - **Host methods 0x1C and 0x1D** (Maxwell's syncpoint payload and operation, which Eden-NX2
//!   still runs) are not in `clc76f.h` and are dropped here (`unknown_host_methods_are_dropped`).
//! - **Binding reads `NVCLASS` (15:0) only.** Eden-NX2 masks to 16 bits only when nothing above
//!   bit 20 is set (`Gf100BindValidMask`) (`bind_reads_nvclass_only`).
//! - **An unreadable semaphore stops the channel** (decision 13 (c), unreadable memory).
//!   Eden-NX2 reads it through `MemoryManager::Read`, which has no failure result, and registers
//!   no waiter
//!   (`an_unreadable_semaphore_stops_the_channel`).
//! - **Errors are reported, not only logged**, and the structural ones stop the channel
//!   (decision 13 (c)); Eden-NX2 logs and goes on
//!   (`malformed_words_are_dropped_and_the_stream_goes_on`,
//!   `an_unmapped_entry_stops_the_channel_for_good`).
//!
//! # Opcodes 0 and 2
//!
//! gpu-protocol, "The method stream", and its facts table read opcodes 0 and 2 through the
//! tertiary opcode, as `clc76f.h` defines them (decided by the owner, 2026-11-18, under
//! decision 10: follow the headers for layouts and field meanings). This module skips only the
//! words `clc76f.h` defines as subdevice-mask operations (opcode 0 with tertiary opcode 1–3,
//! [`MethodHeader::SubdeviceMask`], `subdevice_mask_operations_are_skipped`); opcode 2 with a
//! non-zero tertiary opcode, and opcode 0 with an undefined subdevice-mask form, are dropped as
//! [`DroppedWord::BadHeader`] (`malformed_words_are_dropped_and_the_stream_goes_on`). A word with
//! opcode 0 or 2 and tertiary opcode 0 is an old-format method header in `clc76f.h`
//! (`NVC76F_DMA_METHOD_ADDRESS_OLD`, `_COUNT_OLD`), and runs as one with its data words
//! ([`MethodHeader::Legacy`], `old_format_headers_run_as_methods`). Eden-NX2 skips every opcode
//! 0 and 2 word, which runs an old-format header's data words as headers (a divergence listed
//! above).
//!
//! # Guesses
//!
//! Each is marked **guess** where it is made and pinned by the named test:
//!
//! - **Opcode 7 (`END_PB_SEGMENT`) ends the current entry**, from the opcode's name in
//!   `clc76f.h`; Eden ignores it. A trace with an opcode 7 would confirm it (spec verification
//!   item 9) (`end_segment_ends_the_entry_guess_item_9`).
//! - **NVN2's `SEM_EXECUTE` acquires may block.** The spec suspends on both acquire kinds;
//!   whether NVN2 uses `SEM_EXECUTE` acquires at all, and whether any must not block, is spec
//!   verification item 5 (`sem_execute_acquire_parks_item_5`).
//! - **A 64-bit `SEM_EXECUTE` acquire** (`PAYLOAD_SIZE_64BIT`) compares the 32-bit word at the
//!   address with `SEM_PAYLOAD_LO`, as the spec's acquire table and Eden-NX2 do; no reference
//!   says how the hardware compares it
//!   (`a_64_bit_sem_execute_acquire_compares_the_low_word`).
//! - **An entry past the 40-bit address space stops the channel.** A GP entry's address is 40
//!   bits (`GP_ENTRY0_GET`, `GP_ENTRY1_GET_HI`), so words past 2^40 cannot be fetched; an end
//!   exactly at 2^40 is accepted (`an_entry_past_the_address_space_stops_the_channel`).
//! - **A method header whose data would run past method 0xFFF stops the channel.** The method
//!   field is 12 bits (`NVC76F_DMA_METHOD_ADDRESS` 11:0), so such a header names methods that do
//!   not exist; Eden-NX2 counts on in 13 bits (`a_method_run_past_0xfff_stops_the_channel`).
//! - **`ILLEGAL`, `NON_STALL_INTERRUPT`, `YIELD` and `CLEAR_FAULTED` do nothing** on one GPU
//!   with no interrupt routing; `NOP` and `FB_FLUSH` do nothing in Eden-NX2 too
//!   (`no_op_host_methods_emit_nothing`).
//! - **`RELEASE_WFI` changes nothing**: the back performs a release only after everything
//!   before it in the ring (spec, "Ordering guarantees" 3), which already includes a wait for
//!   idle. Eden-NX2 ignores the bit too (`release_wfi_does_not_change_the_release`).
//!
//! Also not used: `ACQUIRE_SWITCH` and `ACQUIRE_SWITCH_TSG` (a scheduling hint), `WFI_SCOPE`,
//! the `SET_OBJECT` `ENGINE` field, and a GP entry's level (row 73), fetch and sync flags; the
//! sync flag is honoured by the front loop (slice G14, `crate::front`) before it starts the
//! entry.
//!
//! # Not here yet
//!
//! - `SET_REFERENCE` reaches the sink through [`ChannelSink::set_reference`], and the front
//!   loop's sink (slice G14, `crate::front`) turns it into [`RingEntry::SetReference`] (decided
//!   by the owner on 2026-11-18: it stays a ring entry).
//! - A release or acquire on the syncpoint shim page (spec, "Semaphore acquires") is resolved
//!   by the production [`GpuMemory`] ([`crate::memory::VmMemory`], slice G22), not here.

mod host;

#[cfg(test)]
mod tests;

use starquake_memory::PAGE_SIZE;
use thiserror::Error;

use crate::classes::cla140;
use crate::fifo::{
    self, GP_ENTRY_GET_HI_SHIFT, GP_ENTRY1_GET_HI, GpEntry, METHOD_ADDRESS, MethodHeader,
    MethodMode,
};
use crate::interface::ChannelId;
use crate::memory::{GpuFault, GpuMemory};
use crate::ring::{ClassId, RingEntry, SUBCHANNEL_COUNT, Subchannel};
use crate::sema::AcquireOp;

pub use host::HOST_METHODS_END;

/// Bytes per command word.
const WORD_BYTES: u64 = 4;

/// The subchannel NVN2 uses as inline-to-memory without binding a class (row 49; Eden-NX2's
/// `NvkM2mfSubchannel`, `control/channel_state.cpp`).
pub const INLINE_TO_MEMORY_SUBCHANNEL: u8 = 2;

/// The class a fresh channel has bound on `subchannel`: inline-to-memory (`0xA140`,
/// `classes/cla140.h`) on [`INLINE_TO_MEMORY_SUBCHANNEL`], nothing elsewhere.
///
/// The intended contract, not yet in force: the back's channel state (slices G10 and G19)
/// must start from this function, because subchannel 2's starting binding never reaches the
/// back as a [`RingEntry::Bind`]. Nothing outside this module calls it yet.
pub fn initial_binding(subchannel: Subchannel) -> Option<ClassId> {
    (subchannel.get() == INLINE_TO_MEMORY_SUBCHANNEL).then_some(ClassId(cla140::CLASS))
}

/// The end (exclusive) of the address space a GP entry can name: 40 bits, `GP_ENTRY0_GET`
/// (31:2) and `GP_ENTRY1_GET_HI` (7:0) (**guess** that fetching past it is an error, module
/// docs).
pub const GP_ADDRESS_END: u64 = 1 << (GP_ENTRY_GET_HI_SHIFT + GP_ENTRY1_GET_HI.width());

/// The highest method number a header can name (`NVC76F_DMA_METHOD_ADDRESS`, 11:0).
pub const METHOD_MAX: u32 = METHOD_ADDRESS.max();

/// Which host method requested an acquire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AcquireMethod {
    /// `SEMAPHORED`, with the address from `SEMAPHOREA`/`B` and the payload from `SEMAPHOREC`.
    Semaphored,
    /// `SEM_EXECUTE`, with the address from `SEM_ADDR_LO`/`HI` and the payload from
    /// `SEM_PAYLOAD_LO`.
    SemExecute,
}

/// An acquire a channel is parked on: what the front's waiter table records (spec,
/// "Semaphore acquires").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Acquire {
    /// The GPU address of the 32-bit semaphore word.
    pub address: u64,
    /// The condition.
    pub op: AcquireOp,
    /// The value compared with the word.
    pub payload: u32,
    /// The method that requested it.
    pub method: AcquireMethod,
}

impl Acquire {
    /// Whether the semaphore word `value` meets the acquire.
    pub const fn is_met(&self, value: u32) -> bool {
        self.op.is_met(value, self.payload)
    }
}

/// One data word for an engine method: what the register file (slice G9) receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EngineWord {
    /// The subchannel the method was sent on.
    pub subchannel: Subchannel,
    /// The class bound there.
    pub class: ClassId,
    /// The method (dword index), at least [`HOST_METHODS_END`].
    pub method: u16,
    /// The data word.
    pub value: u32,
    /// The data words its header still owes after this one: 0 for the last word and for an
    /// immediate (Eden's `MethodCall::IsLastCall`). A macro call runs when this reaches 0.
    pub remaining: u16,
}

/// What a channel reports besides its output: a dropped word or a carried method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Note {
    /// A word was dropped and the channel went on.
    Dropped {
        /// The GPU address of the word.
        va: u64,
        /// Why.
        word: DroppedWord,
        /// Whether it is the first word of its [`DroppedKind`] this channel dropped.
        first_of_kind: bool,
    },
    /// A submission ended with a method still owed data words; they continue in the channel's
    /// next submission. Noted once per method header.
    MethodCarried {
        /// The subchannel.
        subchannel: Subchannel,
        /// The method the next word goes to.
        method: u16,
        /// The data words still owed.
        remaining: u16,
    },
}

/// Where a channel sends what it executes, in stream order.
pub trait ChannelSink {
    /// A host method's ring entry: a bind, a release or reduction, a wait for idle or a memory
    /// operation.
    fn entry(&mut self, entry: RingEntry<'static>);
    /// A data word for an engine method.
    fn method(&mut self, word: EngineWord);
    /// `SET_REFERENCE` with its count; the front loop sends it on as
    /// [`RingEntry::SetReference`] (module docs).
    fn set_reference(&mut self, value: u32);
    /// A dropped word or a carried method.
    fn note(&mut self, note: Note);
    /// Whether the channel must stop before its next word: a macro call is waiting for a
    /// catch-up (slice G13, `crate::macros`). [`Channel::run`] asks after every word it executes,
    /// and before it reads any; while this is `true` it returns [`RunOutcome::Held`] and reads
    /// nothing more. The default never holds.
    fn holds(&self) -> bool {
        false
    }
}

/// A structural error: the channel stops and every later call returns the same error.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ChannelError {
    /// A command word could not be read.
    #[error("the command word at GPU address {va:#x} could not be read: {fault}")]
    ReadFault {
        /// The word's GPU address.
        va: u64,
        /// Why.
        fault: GpuFault,
    },
    /// An acquire's semaphore word could not be read.
    #[error("the semaphore at GPU address {address:#x} could not be read: {fault}")]
    SemaphoreFault {
        /// The semaphore's GPU address.
        address: u64,
        /// Why.
        fault: GpuFault,
    },
    /// A GP entry whose words run past [`GP_ADDRESS_END`].
    #[error(
        "the GP entry of {length} words at GPU address {address:#x} runs past the 40-bit address space"
    )]
    EntryPastAddressSpace {
        /// The entry's address.
        address: u64,
        /// Its length in words.
        length: u32,
    },
    /// A GP entry whose address is not 4-byte aligned. [`GpEntry::from_words`] cannot make
    /// one (`GP_ENTRY0_GET` is bits 31:2); only a hand-built [`GpEntry`] can.
    #[error("the GP entry at GPU address {address:#x} is not 4-byte aligned")]
    UnalignedEntry {
        /// The entry's address.
        address: u64,
    },
    /// A method header whose data words would run past method [`METHOD_MAX`].
    #[error(
        "a method header for {count} words from method {method:#x} runs past method {max:#x}",
        max = METHOD_MAX
    )]
    MethodPastSpace {
        /// The header's first method.
        method: u16,
        /// Its data word count.
        count: u16,
    },
}

/// Why a word was dropped (the channel goes on).
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DroppedWord {
    /// The word is not a method header `clc76f.h` defines.
    #[error("{0}")]
    BadHeader(fifo::DecodeError),
    /// A method below [`HOST_METHODS_END`] that `clc76f.h` does not define.
    #[error("host method {method:#x} is not defined (data {value:#010x})")]
    UnknownHostMethod {
        /// The method.
        method: u16,
        /// The data word.
        value: u32,
    },
    /// An engine method on a subchannel with no class bound.
    #[error("method {method:#x} was sent on subchannel {subchannel}, which has no class bound")]
    UnboundSubchannel {
        /// The subchannel.
        subchannel: u8,
        /// The method.
        method: u16,
        /// The data word.
        value: u32,
    },
    /// A `SEMAPHORED` or `SEM_EXECUTE` word with an undefined operation, reduction or format.
    #[error("{0}")]
    BadSemaphore(crate::sema::DecodeError),
    /// A `MEM_OP_D` whose `OPERATION` (31:27) `clc76f.h` does not define.
    #[error("MEM_OP_D operation {operation:#x} is not defined")]
    UndefinedMemOp {
        /// The operation field.
        operation: u32,
    },
}

/// The kinds of [`DroppedWord`], for counting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DroppedKind {
    /// [`DroppedWord::BadHeader`].
    BadHeader,
    /// [`DroppedWord::UnknownHostMethod`].
    UnknownHostMethod,
    /// [`DroppedWord::UnboundSubchannel`].
    UnboundSubchannel,
    /// [`DroppedWord::BadSemaphore`].
    BadSemaphore,
    /// [`DroppedWord::UndefinedMemOp`].
    UndefinedMemOp,
}

impl DroppedKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::BadHeader,
        Self::UnknownHostMethod,
        Self::UnboundSubchannel,
        Self::BadSemaphore,
        Self::UndefinedMemOp,
    ];

    /// The kind's index in [`DroppedKind::ALL`].
    const fn index(self) -> usize {
        match self {
            Self::BadHeader => 0,
            Self::UnknownHostMethod => 1,
            Self::UnboundSubchannel => 2,
            Self::BadSemaphore => 3,
            Self::UndefinedMemOp => 4,
        }
    }
}

impl DroppedWord {
    /// The word's kind.
    pub const fn kind(&self) -> DroppedKind {
        match self {
            Self::BadHeader(_) => DroppedKind::BadHeader,
            Self::UnknownHostMethod { .. } => DroppedKind::UnknownHostMethod,
            Self::UnboundSubchannel { .. } => DroppedKind::UnboundSubchannel,
            Self::BadSemaphore(_) => DroppedKind::BadSemaphore,
            Self::UndefinedMemOp { .. } => DroppedKind::UndefinedMemOp,
        }
    }
}

/// A call the channel's state does not allow; nothing changes.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum StateError {
    /// The channel is in the middle of a GP entry.
    #[error("the channel is still running a GP entry")]
    Running,
    /// The channel is parked on an acquire.
    #[error("the channel is parked on an acquire")]
    Parked,
    /// The channel has stopped.
    #[error("the channel has stopped: {0}")]
    Stopped(ChannelError),
}

/// Where a channel stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelState {
    /// No GP entry: ready for [`Channel::start_entry`].
    Idle,
    /// In the middle of a GP entry; [`Channel::run`] goes on with it.
    Running,
    /// Parked on an acquire, inside its GP entry.
    Parked(Acquire),
    /// Stopped by a structural error.
    Stopped(ChannelError),
}

/// Why [`Channel::run`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// The channel had no GP entry.
    Idle,
    /// The GP entry ended; the channel is idle.
    EntryDone,
    /// The channel is parked on an acquire (now, or still).
    Parked(Acquire),
    /// The quantum of words was used; the entry goes on at the next call.
    QuantumUsed,
    /// The sink holds the channel ([`ChannelSink::holds`]): a macro call waits for a catch-up.
    /// The channel goes on from the word after the last one it executed once the sink lets it.
    Held,
}

/// A method header's data words still owed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending {
    mode: MethodMode,
    subchannel: Subchannel,
    /// The method the next word goes to.
    method: u16,
    /// Words still owed, at least 1.
    remaining: u16,
    /// Whether [`Note::MethodCarried`] was reported for this header.
    carried: bool,
}

/// The GP entry a channel is reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Current {
    entry: GpEntry,
    /// Words of the entry consumed so far.
    offset: u32,
}

/// Words read ahead from the current entry, from entry offset `start`. Emptied when the entry
/// ends, after every acquire and when [`Channel::run`] returns [`RunOutcome::QuantumUsed`] or
/// [`RunOutcome::Held`], so it never outlives the `run` call that read it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Chunk {
    start: u32,
    words: Vec<u32>,
}

impl Chunk {
    /// The word at entry offset `offset`, if this chunk holds it.
    fn word(&self, offset: u32) -> Option<u32> {
        let index = usize::try_from(offset.checked_sub(self.start)?).ok()?;
        self.words.get(index).copied()
    }

    fn clear(&mut self) {
        self.words.clear();
    }
}

/// What executing one word asks of the run loop.
enum Step {
    Continue,
    Park(Acquire),
    EndEntry,
}

/// The host semaphore registers (Eden's `Puller::Regs`, zero-initialised).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HostRegs {
    semaphore_a: u32,
    semaphore_b: u32,
    semaphore_c: u32,
    sem_addr_lo: u32,
    sem_addr_hi: u32,
    sem_payload_lo: u32,
    sem_payload_hi: u32,
    reference: u32,
}

/// One GPU channel's pusher state (module docs).
#[derive(Clone, Debug)]
pub struct Channel {
    id: ChannelId,
    bindings: [Option<ClassId>; SUBCHANNEL_COUNT as usize],
    regs: HostRegs,
    pending: Option<Pending>,
    current: Option<Current>,
    chunk: Chunk,
    parked: Option<Acquire>,
    stopped: Option<ChannelError>,
    dropped: [u64; DroppedKind::ALL.len()],
    /// Command words read and executed by [`Channel::run`] so far.
    words: u64,
    scratch: Vec<u8>,
}

impl Channel {
    /// A fresh channel: idle, with [`initial_binding`]s, every host register zero (Eden's
    /// `Puller::Regs regs{}`) and no method pending.
    pub fn new(id: ChannelId) -> Self {
        let mut bindings = [None; SUBCHANNEL_COUNT as usize];
        for (raw, slot) in (0u8..).zip(bindings.iter_mut()) {
            *slot = Subchannel::new(raw).and_then(initial_binding);
        }
        Self {
            id,
            bindings,
            regs: HostRegs::default(),
            pending: None,
            current: None,
            chunk: Chunk::default(),
            parked: None,
            stopped: None,
            dropped: [0; DroppedKind::ALL.len()],
            words: 0,
            scratch: Vec::new(),
        }
    }

    /// The channel's id, as its ring entries name it.
    pub const fn id(&self) -> ChannelId {
        self.id
    }

    /// Where the channel stands.
    pub fn state(&self) -> ChannelState {
        if let Some(error) = self.stopped {
            ChannelState::Stopped(error)
        } else if let Some(acquire) = self.parked {
            ChannelState::Parked(acquire)
        } else if self.current.is_some() {
            ChannelState::Running
        } else {
            ChannelState::Idle
        }
    }

    /// The class bound on `subchannel`, if any.
    pub fn binding(&self, subchannel: Subchannel) -> Option<ClassId> {
        self.bindings
            .get(usize::from(subchannel.get()))
            .copied()
            .flatten()
    }

    /// The last `SET_REFERENCE` count (0 on a fresh channel).
    pub const fn reference(&self) -> u32 {
        self.regs.reference
    }

    /// How many command words [`Channel::run`] has read and executed on this channel, over its
    /// life (0 on a fresh channel): what the front loop (slice G14) counts against a channel's
    /// quantum. A word read ahead but not executed is not counted.
    pub const fn words_read(&self) -> u64 {
        self.words
    }

    /// How many words of `kind` this channel has dropped.
    pub fn dropped_count(&self, kind: DroppedKind) -> u64 {
        self.dropped.get(kind.index()).copied().unwrap_or(0)
    }

    /// Gives an idle channel its next GP entry. Nothing is read yet.
    ///
    /// The checks, in order: a channel that is running an entry, parked or stopped refuses with
    /// [`StateError`] and does not change, so a busy channel given an impossible entry is not
    /// stopped (`a_busy_channel_refuses_before_the_entry_is_checked`); an address that is not
    /// 4-byte aligned stops the channel ([`ChannelError::UnalignedEntry`],
    /// `an_unaligned_entry_stops_the_channel`); an entry whose words run past
    /// [`GP_ADDRESS_END`] stops the channel ([`ChannelError::EntryPastAddressSpace`], a
    /// **guess**, module docs).
    pub fn start_entry(&mut self, entry: GpEntry) -> Result<(), StateError> {
        match self.state() {
            ChannelState::Idle => {}
            ChannelState::Running => return Err(StateError::Running),
            ChannelState::Parked(_) => return Err(StateError::Parked),
            ChannelState::Stopped(error) => return Err(StateError::Stopped(error)),
        }
        // Alignment first, so an entry that is both unaligned and past the address space
        // reports the alignment (`an_unaligned_entry_stops_the_channel`).
        if !entry.address.is_multiple_of(WORD_BYTES) {
            return Err(StateError::Stopped(self.stop(
                ChannelError::UnalignedEntry {
                    address: entry.address,
                },
            )));
        }
        // The address is at most 40 bits and the length 21, so the sum cannot overflow; the
        // checked form keeps that true whatever `GpEntry` holds.
        let end = entry.address.checked_add(entry.byte_len());
        if end.is_none_or(|end| end > GP_ADDRESS_END) {
            let error = ChannelError::EntryPastAddressSpace {
                address: entry.address,
                length: entry.length,
            };
            self.stopped = Some(error);
            return Err(StateError::Stopped(error));
        }
        // The chunk is already empty: every way to `Idle` (a fresh channel, `finish_entry`)
        // leaves it so.
        self.current = Some(Current { entry, offset: 0 });
        Ok(())
    }

    /// Reads and executes words of the current entry until it ends, the channel parks on an
    /// acquire, the sink holds it ([`ChannelSink::holds`]), or `quantum` words have been read in
    /// this call.
    ///
    /// An idle channel returns [`RunOutcome::Idle`] and a parked one
    /// [`RunOutcome::Parked`], reading nothing; a running channel whose sink holds it returns
    /// [`RunOutcome::Held`], reading nothing. A stopped channel returns its error, also one
    /// stopped while parked (`an_unreadable_semaphore_stops_the_channel`). The end
    /// of the entry is checked before the quantum, so an entry whose last word is the
    /// quantum's last returns [`RunOutcome::EntryDone`].
    pub fn run<M, S>(
        &mut self,
        memory: &M,
        sink: &mut S,
        quantum: u32,
    ) -> Result<RunOutcome, ChannelError>
    where
        M: GpuMemory + ?Sized,
        S: ChannelSink + ?Sized,
    {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        if let Some(acquire) = self.parked {
            return Ok(RunOutcome::Parked(acquire));
        }
        if self.current.is_some() && sink.holds() {
            return Ok(RunOutcome::Held);
        }
        let mut used = 0u32;
        loop {
            let Some(current) = self.current else {
                return Ok(RunOutcome::Idle);
            };
            if current.offset >= current.entry.length {
                self.finish_entry();
                return Ok(RunOutcome::EntryDone);
            }
            if used >= quantum {
                // Words read ahead are not kept for a later call: the front may serve other
                // channels and the address table may change in between (module docs,
                // "Reading words"; `words_read_ahead_are_not_kept_across_run_calls`).
                self.chunk.clear();
                return Ok(RunOutcome::QuantumUsed);
            }
            let (va, word) = match self.next_word(memory, current) {
                Ok(read) => read,
                Err(error) => return Err(self.stop(error)),
            };
            used = used.saturating_add(1);
            self.words = self.words.saturating_add(1);
            if let Some(current) = self.current.as_mut() {
                current.offset = current.offset.saturating_add(1);
            }
            match self.execute(va, word, memory, sink) {
                Ok(Step::Continue) => {
                    if sink.holds() {
                        // As for a used quantum, words read ahead are not kept: the channel
                        // reads the next word again when the sink lets it go on.
                        self.chunk.clear();
                        return Ok(RunOutcome::Held);
                    }
                }
                Ok(Step::Park(acquire)) => {
                    self.parked = Some(acquire);
                    return Ok(RunOutcome::Parked(acquire));
                }
                Ok(Step::EndEntry) => {
                    self.finish_entry();
                    return Ok(RunOutcome::EntryDone);
                }
                Err(error) => return Err(self.stop(error)),
            }
        }
    }

    /// Reads a parked channel's semaphore again and resumes the channel if the acquire is met
    /// (the front calls this after registering the waiter, and on each wake-up).
    ///
    /// Returns whether the channel is not parked after the call: `true` also for a channel
    /// that was not parked. A semaphore that cannot be read stops the channel
    /// ([`ChannelError::SemaphoreFault`]).
    pub fn recheck<M>(&mut self, memory: &M) -> Result<bool, ChannelError>
    where
        M: GpuMemory + ?Sized,
    {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let Some(acquire) = self.parked else {
            return Ok(true);
        };
        match read_semaphore(memory, acquire.address) {
            Ok(value) if acquire.is_met(value) => {
                self.parked = None;
                Ok(true)
            }
            Ok(_) => Ok(false),
            Err(error) => Err(self.stop(error)),
        }
    }

    /// Resumes a parked channel without reading its semaphore: its waiter was latched by a
    /// release that met it, even if the word has moved on since (row 70). Returns whether the
    /// channel was parked.
    pub fn resume_latched(&mut self) -> bool {
        if self.stopped.is_some() {
            return false;
        }
        self.parked.take().is_some()
    }

    /// Marks the end of a submission, once its last entry is done. A method still owed words
    /// is kept for the next submission, and noted ([`Note::MethodCarried`]) the first time
    /// only, however many submissions it spans.
    pub fn end_submission<S>(&mut self, sink: &mut S) -> Result<(), StateError>
    where
        S: ChannelSink + ?Sized,
    {
        match self.state() {
            ChannelState::Idle => {}
            ChannelState::Running => return Err(StateError::Running),
            ChannelState::Parked(_) => return Err(StateError::Parked),
            ChannelState::Stopped(error) => return Err(StateError::Stopped(error)),
        }
        if let Some(pending) = self.pending.as_mut()
            && !pending.carried
        {
            pending.carried = true;
            sink.note(Note::MethodCarried {
                subchannel: pending.subchannel,
                method: pending.method,
                remaining: pending.remaining,
            });
        }
        Ok(())
    }

    /// Stops the channel with `error` and returns it.
    fn stop(&mut self, error: ChannelError) -> ChannelError {
        self.stopped = Some(error);
        error
    }

    fn finish_entry(&mut self) {
        self.current = None;
        self.chunk.clear();
    }

    /// The word at the current offset and its GPU address, reading a new chunk if needed.
    fn next_word<M>(&mut self, memory: &M, current: Current) -> Result<(u64, u32), ChannelError>
    where
        M: GpuMemory + ?Sized,
    {
        // In range: `start_entry` checked the whole entry's end against `GP_ADDRESS_END`.
        let va = current
            .entry
            .address
            .wrapping_add(u64::from(current.offset) * WORD_BYTES);
        if let Some(word) = self.chunk.word(current.offset) {
            return Ok((va, word));
        }
        let left_in_entry = u64::from(current.entry.length - current.offset);
        let left_in_page = (PAGE_SIZE - va % PAGE_SIZE) / WORD_BYTES;
        let words = left_in_entry.min(left_in_page);
        let bytes = usize::try_from(words * WORD_BYTES).map_err(|_| ChannelError::ReadFault {
            va,
            fault: GpuFault::OutOfRange {
                va,
                len: words * WORD_BYTES,
            },
        })?;
        self.scratch.clear();
        self.scratch.resize(bytes, 0);
        memory
            .read(va, &mut self.scratch)
            .map_err(|fault| ChannelError::ReadFault { va, fault })?;
        self.chunk.start = current.offset;
        self.chunk.words.clear();
        let (words, _) = self.scratch.as_chunks::<4>();
        self.chunk
            .words
            .extend(words.iter().map(|&bytes| u32::from_le_bytes(bytes)));
        self.chunk
            .word(current.offset)
            .map(|word| (va, word))
            .ok_or(ChannelError::ReadFault {
                va,
                fault: GpuFault::OutOfRange { va, len: 0 },
            })
    }

    /// Executes one word: a data word of the pending method, or a header.
    fn execute<M, S>(
        &mut self,
        va: u64,
        word: u32,
        memory: &M,
        sink: &mut S,
    ) -> Result<Step, ChannelError>
    where
        M: GpuMemory + ?Sized,
        S: ChannelSink + ?Sized,
    {
        if let Some(pending) = self.pending {
            self.pending = advance(pending);
            let remaining = pending.remaining - 1;
            return self.dispatch(
                va,
                pending.subchannel,
                pending.method,
                word,
                remaining,
                memory,
                sink,
            );
        }
        match fifo::decode_header(word) {
            Ok(
                MethodHeader::Methods {
                    mode,
                    subchannel,
                    method,
                    count,
                }
                | MethodHeader::Legacy {
                    mode,
                    subchannel,
                    method,
                    count,
                },
            ) => {
                if runs_past_method_space(mode, method, count) {
                    return Err(ChannelError::MethodPastSpace { method, count });
                }
                let Some(subchannel) = Subchannel::new(subchannel) else {
                    // Unreachable: the header's subchannel field is 3 bits.
                    self.drop_word(
                        va,
                        DroppedWord::UnboundSubchannel {
                            subchannel,
                            method,
                            value: word,
                        },
                        sink,
                    );
                    return Ok(Step::Continue);
                };
                if count > 0 {
                    self.pending = Some(Pending {
                        mode,
                        subchannel,
                        method,
                        remaining: count,
                        carried: false,
                    });
                }
                Ok(Step::Continue)
            }
            Ok(MethodHeader::Immediate {
                subchannel,
                method,
                data,
            }) => {
                let Some(subchannel) = Subchannel::new(subchannel) else {
                    // Unreachable: the header's subchannel field is 3 bits.
                    self.drop_word(
                        va,
                        DroppedWord::UnboundSubchannel {
                            subchannel,
                            method,
                            value: word,
                        },
                        sink,
                    );
                    return Ok(Step::Continue);
                };
                self.dispatch(va, subchannel, method, u32::from(data), 0, memory, sink)
            }
            // Single-GPU no-ops (spec, "The method stream").
            Ok(MethodHeader::SubdeviceMask(_)) => Ok(Step::Continue),
            // **Guess**: opcode 7 ends the entry (verification item 9; module docs).
            Ok(MethodHeader::EndSegment) => Ok(Step::EndEntry),
            Err(error) => {
                self.drop_word(va, DroppedWord::BadHeader(error), sink);
                Ok(Step::Continue)
            }
        }
    }

    /// Sends one data word to its method: a host method below [`HOST_METHODS_END`], otherwise
    /// the class bound on the subchannel.
    #[expect(
        clippy::too_many_arguments,
        reason = "one word's full context; a struct would only rename the same fields"
    )]
    fn dispatch<M, S>(
        &mut self,
        va: u64,
        subchannel: Subchannel,
        method: u16,
        value: u32,
        remaining: u16,
        memory: &M,
        sink: &mut S,
    ) -> Result<Step, ChannelError>
    where
        M: GpuMemory + ?Sized,
        S: ChannelSink + ?Sized,
    {
        if u32::from(method) < HOST_METHODS_END {
            return self.host_method(va, subchannel, method, value, memory, sink);
        }
        match self.binding(subchannel) {
            Some(class) => sink.method(EngineWord {
                subchannel,
                class,
                method,
                value,
                remaining,
            }),
            None => self.drop_word(
                va,
                DroppedWord::UnboundSubchannel {
                    subchannel: subchannel.get(),
                    method,
                    value,
                },
                sink,
            ),
        }
        Ok(Step::Continue)
    }

    /// Reports a dropped word and counts it.
    fn drop_word<S>(&mut self, va: u64, word: DroppedWord, sink: &mut S)
    where
        S: ChannelSink + ?Sized,
    {
        let mut first_of_kind = false;
        if let Some(count) = self.dropped.get_mut(word.kind().index()) {
            first_of_kind = *count == 0;
            *count = count.saturating_add(1);
        }
        sink.note(Note::Dropped {
            va,
            word,
            first_of_kind,
        });
    }

    /// Checks an acquire: continue if it is met, park otherwise. Either way the words after it
    /// are read again (module docs, "Reading words").
    fn acquire<M>(&mut self, acquire: Acquire, memory: &M) -> Result<Step, ChannelError>
    where
        M: GpuMemory + ?Sized,
    {
        self.chunk.clear();
        let value = read_semaphore(memory, acquire.address)?;
        if acquire.is_met(value) {
            Ok(Step::Continue)
        } else {
            Ok(Step::Park(acquire))
        }
    }
}

/// The pending method after one more data word: `None` once its count is used up.
fn advance(pending: Pending) -> Option<Pending> {
    let remaining = pending.remaining.checked_sub(1).filter(|&r| r > 0)?;
    // The header check keeps `method + count - 1` within the method space, so the next
    // method of a pending header fits.
    let next = pending.method.saturating_add(1);
    let (mode, method) = match pending.mode {
        MethodMode::Incrementing => (MethodMode::Incrementing, next),
        MethodMode::NonIncrementing => (MethodMode::NonIncrementing, pending.method),
        // The first word went to the method, every later one goes to the next.
        MethodMode::IncrementOnce => (MethodMode::NonIncrementing, next),
    };
    Some(Pending {
        mode,
        method,
        remaining,
        ..pending
    })
}

/// Whether a header's data words would go past method [`METHOD_MAX`] (a **guess** that this
/// is an error; module docs).
fn runs_past_method_space(mode: MethodMode, method: u16, count: u16) -> bool {
    let last = match (mode, count) {
        (_, 0) | (MethodMode::NonIncrementing, _) | (MethodMode::IncrementOnce, 1) => {
            u32::from(method)
        }
        (MethodMode::Incrementing, count) => u32::from(method) + u32::from(count) - 1,
        (MethodMode::IncrementOnce, _) => u32::from(method) + 1,
    };
    last > METHOD_MAX
}

/// Reads the 32-bit semaphore word at `address`.
fn read_semaphore<M>(memory: &M, address: u64) -> Result<u32, ChannelError>
where
    M: GpuMemory + ?Sized,
{
    let mut bytes = [0u8; 4];
    memory
        .read(address, &mut bytes)
        .map_err(|fault| ChannelError::SemaphoreFault { address, fault })?;
    Ok(u32::from_le_bytes(bytes))
}
