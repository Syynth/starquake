//! The `SQGC` file: its records ([`Capture`] and the types under it), the writer
//! ([`Capture::to_bytes`]) and the reader ([`Capture::parse`]).
//!
//! The layout is this crate's own (gpu-protocol, "What a capture holds"); no reference defines
//! it. Every number is little-endian. The parent module's docs give the byte layout of each
//! record; [`TAGS`] lists every record tag.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use starquake_memory::{Fault, PageRange};
use starquake_shader::lookup::{GuestStage, ProgramRequest};
use thiserror::Error;
use twox_hash::XxHash3_128;

use crate::back::draw::WaitWhileCompiling;
use crate::back::fence::Mode;
use crate::back::fence::Waiter;
use crate::interface::{
    AsId, ChannelConfig, ChannelId, DisplayId, Fence, GpfifoEntry, ImageDesc, Kind, LayerId,
    PresentRequest, Rect, RegionId, Submission, SyncpointId, Ticket, VmUpdate,
};
use crate::memory::GpuFault;
use crate::ring::MarkerToken;
use crate::sema::AcquireOp;

/// The first four bytes of every capture.
pub const MAGIC: [u8; 4] = *b"SQGC";

/// The format version this build writes and reads. Bump it whenever a record's layout changes,
/// or the layout of a blob it stores in someone else's format (the render `Caps` as `postcard`
/// writes them, the address-table and syncpoint snapshots) changes.
pub const VERSION: u32 = 1;

/// The longest build id a header holds, in bytes.
pub const MAX_BUILD_ID: usize = 256;

/// The largest capture [`Capture::parse`] reads: the bound on what a file can make the reader
/// hold, since every record and blob it keeps is at most as large as its bytes (module docs,
/// "Bounds"). A **guess** at a size no synthetic capture reaches.
pub const MAX_CAPTURE_BYTES: usize = 1 << 30;

/// A 128-bit content hash: XXH3's 128-bit variant with seed 0 (`twox-hash`'s `XxHash3_128`, as
/// the shader crate's program hash uses), defined byte for byte by the xxHash specification, so
/// stable across hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(u128);

impl ContentHash {
    /// The hash with this value.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// The hash's value.
    pub const fn value(self) -> u128 {
        self.0
    }

    /// The hash of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self(XxHash3_128::oneshot(bytes))
    }
}

impl Display for ContentHash {
    /// Thirty-two hex digits.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// The capture's blob table: every byte string a record names, once, by its [`ContentHash`]
/// (spec: "A blob table of read contents, deduplicated by hash").
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Blobs {
    by_hash: BTreeMap<ContentHash, Vec<u8>>,
}

impl Blobs {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `bytes` (once, whatever how many times it is stored) and gives its hash.
    pub fn insert(&mut self, bytes: &[u8]) -> ContentHash {
        let hash = ContentHash::of(bytes);
        self.by_hash.entry(hash).or_insert_with(|| bytes.to_vec());
        hash
    }

    /// The bytes named `hash`.
    pub fn get(&self, hash: ContentHash) -> Option<&[u8]> {
        self.by_hash.get(&hash).map(Vec::as_slice)
    }

    /// How many distinct blobs the table holds.
    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    /// The bytes the table holds, summed over its blobs.
    pub fn bytes(&self) -> usize {
        self.by_hash.values().map(Vec::len).sum()
    }
}

/// One channel of the header's channel list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedChannel {
    /// The channel's id, as the front gave it.
    pub id: ChannelId,
    /// What it was made with.
    pub config: ChannelConfig,
}

/// The header (spec: "magic `SQGC`, a format version, the build id, and the channel list"),
/// with what replay needs to rebuild the GPU the capture ran on: the render backend's `Caps` and
/// memory budget, the executor's wait-while-compiling setting and the fence path's mode, and
/// the address spaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// The build of the emulator that wrote the capture: free text, at most [`MAX_BUILD_ID`]
    /// bytes of UTF-8.
    pub build_id: String,
    /// The render backend's `Caps`, as `postcard` writes them, in the blob table.
    pub caps: ContentHash,
    /// The render backend's memory budget in bytes.
    pub memory_allowed: u64,
    /// The executor's wait-while-compiling setting.
    pub setting: WaitWhileCompiling,
    /// The fence path's mode.
    pub mode: Mode,
    /// The address spaces, in the order they were made; pairwise distinct.
    pub spaces: Vec<AsId>,
    /// The channels, in the order they were made; ids pairwise distinct, each in a listed space.
    pub channels: Vec<CapturedChannel>,
}

/// What kind of front or back state an [`StateRecord::Engine`] holds. The layouts of these
/// blobs are not defined yet (parent module's docs, "Starting state").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStateKind {
    /// A channel's register file of one class.
    RegisterFile,
    /// A channel's shadow register file of one class.
    ShadowFile,
    /// A channel's pusher state.
    Pusher,
    /// A channel's subchannel bindings.
    Bindings,
    /// A process's macro code.
    MacroCode,
    /// A channel's macro data RAM.
    DataRam,
    /// The back's QMD table.
    QmdTable,
}

impl EngineStateKind {
    /// Every kind, in tag order.
    pub const ALL: [Self; 7] = [
        Self::RegisterFile,
        Self::ShadowFile,
        Self::Pusher,
        Self::Bindings,
        Self::MacroCode,
        Self::DataRam,
        Self::QmdTable,
    ];

    /// The kind's byte in the file.
    pub const fn value(self) -> u8 {
        match self {
            Self::RegisterFile => 1,
            Self::ShadowFile => 2,
            Self::Pusher => 3,
            Self::Bindings => 4,
            Self::MacroCode => 5,
            Self::DataRam => 6,
            Self::QmdTable => 7,
        }
    }

    /// The kind whose byte is `value`.
    pub fn from_value(value: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.value() == value)
    }

    /// The kind in plain words.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RegisterFile => "a register file",
            Self::ShadowFile => "a shadow register file",
            Self::Pusher => "a pusher state",
            Self::Bindings => "subchannel bindings",
            Self::MacroCode => "macro code",
            Self::DataRam => "macro data RAM",
            Self::QmdTable => "the QMD table",
        }
    }
}

/// One record of the starting state (spec, "What a capture holds", item 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateRecord {
    /// An address space's table, as [`crate::vm::GpuVm::save`] writes it, in the blob table.
    AddressTable {
        /// The address space.
        space: AsId,
        /// The snapshot's blob.
        blob: ContentHash,
    },
    /// The syncpoints, as [`crate::syncpoint::Syncpoints::save`] writes them.
    Syncpoints {
        /// The snapshot's blob.
        blob: ContentHash,
    },
    /// Front or back state of a kind replay cannot restore yet.
    Engine {
        /// What it is.
        kind: EngineStateKind,
        /// The channel, process or (for the QMD table) 0 it belongs to.
        owner: u32,
        /// Its blob.
        blob: ContentHash,
    },
}

/// One access to a GPU address space, in either thread's log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// [`crate::memory::GpuMemory::read`]: the bytes read (by blob), or the fault.
    Read {
        /// The address space.
        space: AsId,
        /// The GPU address.
        va: u64,
        /// The length in bytes.
        len: u64,
        /// The blob of the bytes read, or the fault.
        result: Result<ContentHash, GpuFault>,
    },
    /// [`crate::memory::GpuMemory::write`]: the bytes written (by blob) and the outcome.
    Write {
        /// The address space.
        space: AsId,
        /// The GPU address.
        va: u64,
        /// The blob of the bytes written.
        data: ContentHash,
        /// The outcome.
        result: Result<(), GpuFault>,
    },
    /// [`crate::memory::GpuMemory::is_gpu_only`] and its answer.
    GpuOnly {
        /// The address space.
        space: AsId,
        /// The GPU address.
        va: u64,
        /// The length in bytes.
        len: u64,
        /// The answer.
        answer: bool,
    },
    /// [`crate::memory::GpuMemory::generation`] and its answer.
    Generation {
        /// The address space.
        space: AsId,
        /// The answer.
        value: u64,
    },
}

/// One record of the front thread's input log (spec, "What a capture holds", item 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontRecord {
    /// The front ran one turn ([`crate::front::Front::turn`]), taking every request and report
    /// recorded since the previous turn.
    Turn,
    /// A submission the front took.
    Submit {
        /// The channel.
        channel: ChannelId,
        /// The submission.
        submission: Submission,
    },
    /// An ordered update the front took.
    Update {
        /// The address space.
        space: AsId,
        /// The update.
        update: VmUpdate,
        /// The ticket to report it with.
        ticket: Option<Ticket>,
    },
    /// A barrier the front took.
    Barrier {
        /// The ticket to report it with.
        ticket: Ticket,
    },
    /// A present the front took.
    Present {
        /// The present.
        request: PresentRequest,
    },
    /// A channel destroy the front took.
    Destroy {
        /// The channel.
        channel: ChannelId,
    },
    /// A report from the back the front received: what [`crate::front::FrontShared`]'s wake
    /// keeps of it.
    Report {
        /// The markers performed, in order.
        markers: Vec<MarkerToken>,
        /// Whether a release was written, a waiter latched, or a release failed.
        released: bool,
        /// Whether a syncpoint was incremented.
        incremented: bool,
    },
    /// A syncpoint's value changed (by the back) since the front last saw it: what fence waits
    /// read.
    Syncpoint {
        /// The syncpoint.
        id: SyncpointId,
        /// Its new value.
        value: u32,
    },
    /// The back latched this channel's acquire waiter since the front's last turn.
    Latch {
        /// The channel.
        channel: ChannelId,
    },
    /// The front's clock was read (the acquire poll's ticks).
    Clock {
        /// The reading.
        now: Duration,
    },
    /// An access to an address space.
    Access(Access),
}

/// What the shader cache answered a lookup or a wait with. Only the kind of answer is kept: the
/// shaders of a ready answer and the reason of a rejection need the shader capture format
/// (parent module's docs, "Shader lookups").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderAnswer {
    /// The shaders were ready.
    Ready,
    /// The shaders were being translated, under this ticket.
    Pending {
        /// The ticket's number.
        ticket: u64,
    },
    /// A program was rejected.
    Rejected,
}

/// One record of the back thread's input log (spec, "What a capture holds", item 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackRecord {
    /// The back took the next ring entry and applied it ([`crate::back::exec::Executor::apply`]).
    Apply,
    /// The back closed its open list for an empty ring ([`crate::back::exec::Executor::close`]).
    Close,
    /// The back ran completion processing ([`crate::back::exec::Executor::complete`]), with the
    /// backend's completed fence value at that moment.
    Complete {
        /// `Backend::completed` as the back read it.
        completed: u64,
    },
    /// The back's clock was read for the time a list has been open.
    Elapsed {
        /// The reading.
        elapsed: Duration,
    },
    /// The back's clock was read for a release's timestamp.
    Timestamp {
        /// The reading.
        value: u64,
    },
    /// The acquire waiter table, as the front left it, before the back's next step: every
    /// waiter and whether it is latched, in table order.
    Waiters {
        /// The waiters.
        waiters: Vec<(Waiter, bool)>,
    },
    /// A write-tracking drain of an address space's watch, and the ranges it gave.
    Drain {
        /// The address space.
        space: AsId,
        /// The dirty ranges.
        ranges: Vec<PageRange>,
    },
    /// A shader lookup: the requests, and the answer's kind.
    ShaderLookup {
        /// The programs asked for.
        requests: Vec<ProgramRequest>,
        /// The answer.
        answer: ShaderAnswer,
    },
    /// A wait for a pending translation, and the answer's kind.
    ShaderWait {
        /// The ticket waited for.
        ticket: u64,
        /// The answer.
        answer: ShaderAnswer,
    },
    /// An access to an address space.
    Access(Access),
}

/// One output, for comparison (spec, "What a capture holds", item 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// The next ring entry the front pushed: the hash of its ring encoding
    /// ([`crate::ring::encode`]'s words, little-endian).
    Ring {
        /// The hash.
        hash: ContentHash,
    },
    /// The render command stream: the recording backend's snapshot text at the capture's end,
    /// in the blob table.
    Render {
        /// The text's blob.
        text: ContentHash,
    },
}

/// A whole capture (spec, "What a capture holds").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    /// The header.
    pub header: Header,
    /// The starting state.
    pub state: Vec<StateRecord>,
    /// The front thread's input log, in the order the front consumed it.
    pub front: Vec<FrontRecord>,
    /// The back thread's input log, in the order the back consumed it.
    pub back: Vec<BackRecord>,
    /// The blob table.
    pub blobs: Blobs,
    /// The outputs.
    pub outputs: Vec<Output>,
}

/// The four bytes where the magic belongs, shown as hex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Found(pub [u8; 4]);

impl Display for Found {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let hex: Vec<String> = self.0.iter().map(|byte| format!("{byte:02x}")).collect();
        f.write_str(&hex.join(" "))
    }
}

/// Why a capture could not be written or read. The reader checks the header (magic, then
/// version) before anything after it.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CaptureError {
    /// The first four bytes are not [`MAGIC`].
    #[error("the file is not a GPU capture: it starts with {found}, not SQGC")]
    BadMagic {
        /// The four bytes.
        found: Found,
    },
    /// The version is not [`VERSION`].
    #[error("the capture is format version {found}; this build reads version {VERSION} only")]
    UnsupportedVersion {
        /// The version in the file.
        found: u32,
    },
    /// The file is larger than [`MAX_CAPTURE_BYTES`].
    #[error("the capture is {len} bytes; at most {MAX_CAPTURE_BYTES} bytes are read")]
    TooLarge {
        /// The file's length.
        len: usize,
    },
    /// The file ends inside something.
    #[error("the capture ends at byte {offset}, inside {what}")]
    Truncated {
        /// Where the file ends.
        offset: usize,
        /// What it ends inside.
        what: &'static str,
    },
    /// A record's tag is not one of its section's.
    #[error("byte {offset}: {tag:#04x} is not a record of the {section}")]
    UnknownTag {
        /// The tag's offset.
        offset: usize,
        /// The tag.
        tag: u8,
        /// The section.
        section: &'static str,
    },
    /// A field holds a value it cannot.
    #[error("byte {offset}: {what}")]
    BadValue {
        /// The field's offset.
        offset: usize,
        /// What is wrong, in plain words.
        what: &'static str,
    },
    /// Bytes follow the last section.
    #[error("the capture has {count} bytes after its last section, from byte {offset}")]
    TrailingBytes {
        /// Where they start.
        offset: usize,
        /// How many.
        count: usize,
    },
    /// The blob table is not in strictly ascending hash order (a hash given twice included).
    #[error("byte {offset}: the blob table is out of order or names a blob twice")]
    BlobOrder {
        /// The offset of the blob out of order.
        offset: usize,
    },
    /// A blob's bytes do not hash to its name.
    #[error("blob {hash} does not hash to its name")]
    CorruptBlob {
        /// Its name.
        hash: ContentHash,
    },
    /// A record names a blob the table does not hold.
    #[error("a record names blob {hash}, which the blob table does not hold")]
    MissingBlob {
        /// The name.
        hash: ContentHash,
    },
    /// A read's length is not its blob's.
    #[error("a read of {len} bytes names blob {hash}, which holds {found} bytes")]
    ReadLength {
        /// The read's length.
        len: u64,
        /// The blob.
        hash: ContentHash,
        /// The blob's length.
        found: usize,
    },
    /// The build id is longer than [`MAX_BUILD_ID`] bytes.
    #[error("the build id is {len} bytes; at most {MAX_BUILD_ID} are allowed")]
    BuildIdTooLong {
        /// Its length.
        len: usize,
    },
    /// An address space is listed twice in the header.
    #[error("address space {} is listed twice", .space.0)]
    DuplicateSpace {
        /// The space.
        space: AsId,
    },
    /// A channel is listed twice in the header.
    #[error("channel {} is listed twice", .channel.0)]
    DuplicateChannel {
        /// The channel.
        channel: ChannelId,
    },
    /// A channel's address space is not listed in the header.
    #[error("channel {} is in address space {}, which the header does not list", .channel.0, .space.0)]
    UnknownSpace {
        /// The channel.
        channel: ChannelId,
        /// Its space.
        space: AsId,
    },
    /// A list has more items than its 32-bit count can say (writing only).
    #[error("{what} has more than 4294967295 items")]
    TooMany {
        /// The list.
        what: &'static str,
    },
}

/// Every record tag, by section: the state, front, back, access (in both logs) and output
/// records. Pairwise distinct, so a record read in the wrong section is an unknown tag.
pub const TAGS: [(&str, u8); 28] = [
    ("state: address table", tag::ADDRESS_TABLE),
    ("state: syncpoints", tag::SYNCPOINTS),
    ("state: engine", tag::ENGINE),
    ("front: turn", tag::TURN),
    ("front: submit", tag::SUBMIT),
    ("front: update", tag::UPDATE),
    ("front: barrier", tag::BARRIER),
    ("front: present", tag::PRESENT),
    ("front: destroy", tag::DESTROY),
    ("front: report", tag::REPORT),
    ("front: syncpoint", tag::SYNCPOINT),
    ("front: latch", tag::LATCH),
    ("front: clock", tag::CLOCK),
    ("back: apply", tag::APPLY),
    ("back: close", tag::CLOSE),
    ("back: complete", tag::COMPLETE),
    ("back: elapsed", tag::ELAPSED),
    ("back: timestamp", tag::TIMESTAMP),
    ("back: waiters", tag::WAITERS),
    ("back: drain", tag::DRAIN),
    ("back: shader lookup", tag::SHADER_LOOKUP),
    ("back: shader wait", tag::SHADER_WAIT),
    ("access: read", tag::READ),
    ("access: write", tag::WRITE),
    ("access: GPU-only query", tag::GPU_ONLY),
    ("access: generation", tag::GENERATION),
    ("output: ring", tag::RING),
    ("output: render", tag::RENDER),
];

/// The record tags.
pub(crate) mod tag {
    pub(crate) const ADDRESS_TABLE: u8 = 0x01;
    pub(crate) const SYNCPOINTS: u8 = 0x02;
    pub(crate) const ENGINE: u8 = 0x03;
    pub(crate) const TURN: u8 = 0x10;
    pub(crate) const SUBMIT: u8 = 0x11;
    pub(crate) const UPDATE: u8 = 0x12;
    pub(crate) const BARRIER: u8 = 0x13;
    pub(crate) const PRESENT: u8 = 0x14;
    pub(crate) const DESTROY: u8 = 0x15;
    pub(crate) const REPORT: u8 = 0x16;
    pub(crate) const SYNCPOINT: u8 = 0x17;
    pub(crate) const LATCH: u8 = 0x18;
    pub(crate) const CLOCK: u8 = 0x19;
    pub(crate) const APPLY: u8 = 0x20;
    pub(crate) const CLOSE: u8 = 0x21;
    pub(crate) const COMPLETE: u8 = 0x22;
    pub(crate) const ELAPSED: u8 = 0x23;
    pub(crate) const TIMESTAMP: u8 = 0x24;
    pub(crate) const WAITERS: u8 = 0x25;
    pub(crate) const DRAIN: u8 = 0x26;
    pub(crate) const SHADER_LOOKUP: u8 = 0x27;
    pub(crate) const SHADER_WAIT: u8 = 0x28;
    pub(crate) const READ: u8 = 0x30;
    pub(crate) const WRITE: u8 = 0x31;
    pub(crate) const GPU_ONLY: u8 = 0x32;
    pub(crate) const GENERATION: u8 = 0x33;
    pub(crate) const RING: u8 = 0x40;
    pub(crate) const RENDER: u8 = 0x41;
}

/// The byte of each sub-kind field (an enum inside a record).
pub(crate) mod sub {
    /// `WaitWhileCompiling`.
    pub(crate) const WAIT: u8 = 0;
    pub(crate) const SKIP: u8 = 1;
    /// `Mode`.
    pub(crate) const FENCED: u8 = 0;
    pub(crate) const AT_RECORD: u8 = 1;
    /// `VmUpdate`.
    pub(crate) const UNMAP: u8 = 0;
    pub(crate) const MODIFY_KIND: u8 = 1;
    pub(crate) const DESTROY_MEMORY: u8 = 2;
    pub(crate) const DESTROY_SPARSE: u8 = 3;
    /// `GpuFault`.
    pub(crate) const UNMAPPED: u8 = 0;
    pub(crate) const OUT_OF_RANGE: u8 = 1;
    pub(crate) const GUEST: u8 = 2;
    pub(crate) const SHIM_WRITE: u8 = 3;
    /// A `Result`'s outcome byte.
    pub(crate) const OK: u8 = 0;
    pub(crate) const FAULT: u8 = 1;
    /// `ShaderAnswer`.
    pub(crate) const READY: u8 = 0;
    pub(crate) const PENDING: u8 = 1;
    pub(crate) const REJECTED: u8 = 2;
    /// An `Option`'s presence byte.
    pub(crate) const NONE: u8 = 0;
    pub(crate) const SOME: u8 = 1;
}

/// A `GuestStage`'s byte. The order is this format's own.
pub const fn stage_value(stage: GuestStage) -> u8 {
    match stage {
        GuestStage::VertexCullBeforeFetch => 0,
        GuestStage::Vertex => 1,
        GuestStage::TessellationInit => 2,
        GuestStage::Tessellation => 3,
        GuestStage::Geometry => 4,
        GuestStage::Pixel => 5,
        GuestStage::Compute => 6,
    }
}

/// The stage whose byte is `value`.
pub const fn stage_from(value: u8) -> Option<GuestStage> {
    Some(match value {
        0 => GuestStage::VertexCullBeforeFetch,
        1 => GuestStage::Vertex,
        2 => GuestStage::TessellationInit,
        3 => GuestStage::Tessellation,
        4 => GuestStage::Geometry,
        5 => GuestStage::Pixel,
        6 => GuestStage::Compute,
        _ => return None,
    })
}

/// An `AcquireOp`'s byte. The order is this format's own.
pub const fn acquire_value(op: AcquireOp) -> u8 {
    match op {
        AcquireOp::Equal => 0,
        AcquireOp::GreaterOrEqual => 1,
        AcquireOp::CircularGreaterOrEqual => 2,
        AcquireOp::And => 3,
        AcquireOp::Nor => 4,
    }
}

/// The operation whose byte is `value`.
pub const fn acquire_from(value: u8) -> Option<AcquireOp> {
    Some(match value {
        0 => AcquireOp::Equal,
        1 => AcquireOp::GreaterOrEqual,
        2 => AcquireOp::CircularGreaterOrEqual,
        3 => AcquireOp::And,
        4 => AcquireOp::Nor,
        _ => return None,
    })
}

/// A duration as whole nanoseconds, saturating at `u64::MAX` (584 years).
pub(crate) fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

// --- Writing ---------------------------------------------------------------------------------

/// The bytes being written.
struct Out(Vec<u8>);

impl Out {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn hash(&mut self, hash: ContentHash) {
        self.0.extend_from_slice(&hash.0.to_le_bytes());
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn count(&mut self, len: usize, what: &'static str) -> Result<(), CaptureError> {
        self.u32(u32::try_from(len).map_err(|_| CaptureError::TooMany { what })?);
        Ok(())
    }
    fn range(&mut self, range: PageRange) {
        self.u64(range.start());
        self.u64(range.size());
    }
    fn fault(&mut self, fault: GpuFault) {
        match fault {
            GpuFault::Unmapped { va } => {
                self.u8(sub::UNMAPPED);
                self.u64(va);
            }
            GpuFault::OutOfRange { va, len } => {
                self.u8(sub::OUT_OF_RANGE);
                self.u64(va);
                self.u64(len);
            }
            GpuFault::Guest { va, fault } => {
                self.u8(sub::GUEST);
                self.u64(va);
                self.u64(fault.address);
            }
            GpuFault::ShimWrite { va } => {
                self.u8(sub::SHIM_WRITE);
                self.u64(va);
            }
        }
    }
    fn answer(&mut self, answer: ShaderAnswer) {
        match answer {
            ShaderAnswer::Ready => self.u8(sub::READY),
            ShaderAnswer::Pending { ticket } => {
                self.u8(sub::PENDING);
                self.u64(ticket);
            }
            ShaderAnswer::Rejected => self.u8(sub::REJECTED),
        }
    }

    fn header(&mut self, header: &Header) -> Result<(), CaptureError> {
        self.0.extend_from_slice(&MAGIC);
        self.u32(VERSION);
        let id = header.build_id.as_bytes();
        if id.len() > MAX_BUILD_ID {
            return Err(CaptureError::BuildIdTooLong { len: id.len() });
        }
        self.count(id.len(), "the build id")?;
        self.0.extend_from_slice(id);
        self.hash(header.caps);
        self.u64(header.memory_allowed);
        self.u8(match header.setting {
            WaitWhileCompiling::Wait => sub::WAIT,
            WaitWhileCompiling::Skip => sub::SKIP,
        });
        self.u8(match header.mode {
            Mode::Fenced => sub::FENCED,
            Mode::AtRecord => sub::AT_RECORD,
        });
        self.count(header.spaces.len(), "the address space list")?;
        for space in &header.spaces {
            self.u32(space.0);
        }
        self.count(header.channels.len(), "the channel list")?;
        for channel in &header.channels {
            self.u32(channel.id.0);
            self.u32(channel.config.space.0);
            self.u8(channel.config.syncpoint.get());
        }
        Ok(())
    }

    fn state(&mut self, record: &StateRecord) {
        match *record {
            StateRecord::AddressTable { space, blob } => {
                self.u8(tag::ADDRESS_TABLE);
                self.u32(space.0);
                self.hash(blob);
            }
            StateRecord::Syncpoints { blob } => {
                self.u8(tag::SYNCPOINTS);
                self.hash(blob);
            }
            StateRecord::Engine { kind, owner, blob } => {
                self.u8(tag::ENGINE);
                self.u8(kind.value());
                self.u32(owner);
                self.hash(blob);
            }
        }
    }

    fn access(&mut self, access: &Access) {
        match *access {
            Access::Read {
                space,
                va,
                len,
                result,
            } => {
                self.u8(tag::READ);
                self.u32(space.0);
                self.u64(va);
                self.u64(len);
                match result {
                    Ok(blob) => {
                        self.u8(sub::OK);
                        self.hash(blob);
                    }
                    Err(fault) => {
                        self.u8(sub::FAULT);
                        self.fault(fault);
                    }
                }
            }
            Access::Write {
                space,
                va,
                data,
                result,
            } => {
                self.u8(tag::WRITE);
                self.u32(space.0);
                self.u64(va);
                self.hash(data);
                match result {
                    Ok(()) => self.u8(sub::OK),
                    Err(fault) => {
                        self.u8(sub::FAULT);
                        self.fault(fault);
                    }
                }
            }
            Access::GpuOnly {
                space,
                va,
                len,
                answer,
            } => {
                self.u8(tag::GPU_ONLY);
                self.u32(space.0);
                self.u64(va);
                self.u64(len);
                self.bool(answer);
            }
            Access::Generation { space, value } => {
                self.u8(tag::GENERATION);
                self.u32(space.0);
                self.u64(value);
            }
        }
    }

    fn submit(&mut self, channel: ChannelId, submission: &Submission) -> Result<(), CaptureError> {
        self.u8(tag::SUBMIT);
        self.u32(channel.0);
        match submission.wait {
            None => self.u8(sub::NONE),
            Some(fence) => {
                self.u8(sub::SOME);
                self.u8(fence.id.get());
                self.u32(fence.value);
            }
        }
        self.u32(submission.increments);
        self.bool(submission.wfi);
        self.count(submission.entries.len(), "a submission")?;
        for entry in &submission.entries {
            self.u64(entry.0);
        }
        Ok(())
    }

    fn update(&mut self, space: AsId, update: &VmUpdate, ticket: Option<Ticket>) {
        self.u8(tag::UPDATE);
        self.u32(space.0);
        match *update {
            VmUpdate::Unmap { range } => {
                self.u8(sub::UNMAP);
                self.range(range);
            }
            VmUpdate::ModifyKind { range, kind } => {
                self.u8(sub::MODIFY_KIND);
                self.range(range);
                self.u8(kind.0);
            }
            VmUpdate::DestroyMemory { range } => {
                self.u8(sub::DESTROY_MEMORY);
                self.range(range);
            }
            VmUpdate::DestroySparseRegion { range, region } => {
                self.u8(sub::DESTROY_SPARSE);
                self.range(range);
                self.u32(region.0);
            }
        }
        match ticket {
            None => self.u8(sub::NONE),
            Some(ticket) => {
                self.u8(sub::SOME);
                self.u64(ticket.0);
            }
        }
    }

    fn present(&mut self, request: &PresentRequest) {
        self.u8(tag::PRESENT);
        let image = request.image;
        self.u64(image.address);
        self.u32(image.width);
        self.u32(image.height);
        self.u32(image.pitch);
        self.u32(image.block_height_log2);
        self.u32(image.format);
        self.u32(request.shown.x);
        self.u32(request.shown.y);
        self.u32(request.shown.width);
        self.u32(request.shown.height);
        self.u64(request.layer.0);
        self.u64(request.display.0);
    }

    fn front(&mut self, record: &FrontRecord) -> Result<(), CaptureError> {
        match record {
            FrontRecord::Turn => self.u8(tag::TURN),
            FrontRecord::Submit {
                channel,
                submission,
            } => self.submit(*channel, submission)?,
            FrontRecord::Update {
                space,
                update,
                ticket,
            } => self.update(*space, update, *ticket),
            FrontRecord::Barrier { ticket } => {
                self.u8(tag::BARRIER);
                self.u64(ticket.0);
            }
            FrontRecord::Present { request } => self.present(request),
            FrontRecord::Destroy { channel } => {
                self.u8(tag::DESTROY);
                self.u32(channel.0);
            }
            FrontRecord::Report {
                markers,
                released,
                incremented,
            } => {
                self.u8(tag::REPORT);
                self.bool(*released);
                self.bool(*incremented);
                self.count(markers.len(), "a report's markers")?;
                for marker in markers {
                    self.u64(marker.0);
                }
            }
            FrontRecord::Syncpoint { id, value } => {
                self.u8(tag::SYNCPOINT);
                self.u8(id.get());
                self.u32(*value);
            }
            FrontRecord::Latch { channel } => {
                self.u8(tag::LATCH);
                self.u32(channel.0);
            }
            FrontRecord::Clock { now } => {
                self.u8(tag::CLOCK);
                self.u64(nanos(*now));
            }
            FrontRecord::Access(access) => self.access(access),
        }
        Ok(())
    }

    fn back(&mut self, record: &BackRecord) -> Result<(), CaptureError> {
        match record {
            BackRecord::Apply => self.u8(tag::APPLY),
            BackRecord::Close => self.u8(tag::CLOSE),
            BackRecord::Complete { completed } => {
                self.u8(tag::COMPLETE);
                self.u64(*completed);
            }
            BackRecord::Elapsed { elapsed } => {
                self.u8(tag::ELAPSED);
                self.u64(nanos(*elapsed));
            }
            BackRecord::Timestamp { value } => {
                self.u8(tag::TIMESTAMP);
                self.u64(*value);
            }
            BackRecord::Waiters { waiters } => {
                self.u8(tag::WAITERS);
                self.count(waiters.len(), "the waiter table")?;
                for (waiter, latched) in waiters {
                    self.u32(waiter.channel.0);
                    self.u32(waiter.space.0);
                    self.u64(waiter.address);
                    self.u8(acquire_value(waiter.op));
                    self.u32(waiter.payload);
                    self.bool(*latched);
                }
            }
            BackRecord::Drain { space, ranges } => {
                self.u8(tag::DRAIN);
                self.u32(space.0);
                self.count(ranges.len(), "a drain")?;
                for range in ranges {
                    self.range(*range);
                }
            }
            BackRecord::ShaderLookup { requests, answer } => {
                self.u8(tag::SHADER_LOOKUP);
                self.count(requests.len(), "a shader lookup")?;
                for request in requests {
                    self.u8(stage_value(request.stage));
                    self.u64(request.address);
                }
                self.answer(*answer);
            }
            BackRecord::ShaderWait { ticket, answer } => {
                self.u8(tag::SHADER_WAIT);
                self.u64(*ticket);
                self.answer(*answer);
            }
            BackRecord::Access(access) => self.access(access),
        }
        Ok(())
    }

    fn output(&mut self, output: &Output) {
        match *output {
            Output::Ring { hash } => {
                self.u8(tag::RING);
                self.hash(hash);
            }
            Output::Render { text } => {
                self.u8(tag::RENDER);
                self.hash(text);
            }
        }
    }
}

// --- Reading ---------------------------------------------------------------------------------

/// A field at `at` holds a value it cannot.
fn bad(at: usize, what: &'static str) -> CaptureError {
    CaptureError::BadValue { offset: at, what }
}

/// The bytes being read, and where.
struct In<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> In<'a> {
    fn take(&mut self, len: usize, what: &'static str) -> Result<&'a [u8], CaptureError> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len());
        let Some(end) = end else {
            return Err(CaptureError::Truncated {
                offset: self.bytes.len(),
                what,
            });
        };
        let taken = self.bytes.get(self.at..end).ok_or(CaptureError::Truncated {
            offset: self.bytes.len(),
            what,
        })?;
        self.at = end;
        Ok(taken)
    }
    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], CaptureError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N, what)?);
        Ok(out)
    }
    fn u8(&mut self, what: &'static str) -> Result<u8, CaptureError> {
        Ok(self.array::<1>(what)?[0])
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, CaptureError> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64, CaptureError> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }
    fn hash(&mut self, what: &'static str) -> Result<ContentHash, CaptureError> {
        Ok(ContentHash(u128::from_le_bytes(self.array(what)?)))
    }
    fn bool(&mut self, what: &'static str) -> Result<bool, CaptureError> {
        let at = self.at;
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad(at, "a flag is neither 0 nor 1")),
        }
    }
    fn count(&mut self, what: &'static str) -> Result<u32, CaptureError> {
        self.u32(what)
    }
    fn space(&mut self) -> Result<AsId, CaptureError> {
        Ok(AsId(self.u32("an address space")?))
    }
    fn syncpoint(&mut self) -> Result<SyncpointId, CaptureError> {
        let at = self.at;
        SyncpointId::new(self.u8("a syncpoint")?)
            .ok_or_else(|| bad(at, "a syncpoint number is past the last syncpoint"))
    }
    fn range(&mut self) -> Result<PageRange, CaptureError> {
        let at = self.at;
        let start = self.u64("a page range")?;
        let size = self.u64("a page range")?;
        PageRange::new(start, size).map_err(|_| bad(at, "a page range is not whole pages"))
    }
    fn fault(&mut self) -> Result<GpuFault, CaptureError> {
        let at = self.at;
        match self.u8("a fault")? {
            sub::UNMAPPED => Ok(GpuFault::Unmapped {
                va: self.u64("a fault")?,
            }),
            sub::OUT_OF_RANGE => Ok(GpuFault::OutOfRange {
                va: self.u64("a fault")?,
                len: self.u64("a fault")?,
            }),
            sub::GUEST => Ok(GpuFault::Guest {
                va: self.u64("a fault")?,
                fault: Fault {
                    address: self.u64("a fault")?,
                },
            }),
            sub::SHIM_WRITE => Ok(GpuFault::ShimWrite {
                va: self.u64("a fault")?,
            }),
            _ => Err(bad(at, "a fault's kind is not one this build knows")),
        }
    }
    fn answer(&mut self) -> Result<ShaderAnswer, CaptureError> {
        let at = self.at;
        match self.u8("a shader answer")? {
            sub::READY => Ok(ShaderAnswer::Ready),
            sub::PENDING => Ok(ShaderAnswer::Pending {
                ticket: self.u64("a shader answer")?,
            }),
            sub::REJECTED => Ok(ShaderAnswer::Rejected),
            _ => Err(bad(at, "a shader answer's kind is not one this build knows")),
        }
    }

    fn header(&mut self) -> Result<Header, CaptureError> {
        let magic = self.array::<4>("the magic")?;
        if magic != MAGIC {
            return Err(CaptureError::BadMagic { found: Found(magic) });
        }
        let version = self.u32("the version")?;
        if version != VERSION {
            return Err(CaptureError::UnsupportedVersion { found: version });
        }
        let at = self.at;
        let len = usize::try_from(self.count("the build id")?).unwrap_or(usize::MAX);
        if len > MAX_BUILD_ID {
            return Err(CaptureError::BuildIdTooLong { len });
        }
        let id = self.take(len, "the build id")?;
        let build_id = std::str::from_utf8(id)
            .map_err(|_| bad(at, "the build id is not UTF-8"))?
            .to_owned();
        let caps = self.hash("the header")?;
        let memory_allowed = self.u64("the header")?;
        let at = self.at;
        let setting = match self.u8("the header")? {
            sub::WAIT => WaitWhileCompiling::Wait,
            sub::SKIP => WaitWhileCompiling::Skip,
            _ => return Err(bad(at, "the wait-while-compiling setting is unknown")),
        };
        let at = self.at;
        let mode = match self.u8("the header")? {
            sub::FENCED => Mode::Fenced,
            sub::AT_RECORD => Mode::AtRecord,
            _ => return Err(bad(at, "the fence mode is unknown")),
        };
        let mut spaces = Vec::new();
        for _ in 0..self.count("the address space list")? {
            let space = self.space()?;
            if spaces.contains(&space) {
                return Err(CaptureError::DuplicateSpace { space });
            }
            spaces.push(space);
        }
        let mut channels: Vec<CapturedChannel> = Vec::new();
        for _ in 0..self.count("the channel list")? {
            let id = ChannelId(self.u32("a channel")?);
            let space = self.space()?;
            let syncpoint = self.syncpoint()?;
            if channels.iter().any(|known| known.id == id) {
                return Err(CaptureError::DuplicateChannel { channel: id });
            }
            if !spaces.contains(&space) {
                return Err(CaptureError::UnknownSpace { channel: id, space });
            }
            channels.push(CapturedChannel {
                id,
                config: ChannelConfig { space, syncpoint },
            });
        }
        Ok(Header {
            build_id,
            caps,
            memory_allowed,
            setting,
            mode,
            spaces,
            channels,
        })
    }

    fn tag(&mut self, section: &'static str) -> Result<(usize, u8), CaptureError> {
        let at = self.at;
        Ok((at, self.u8(section)?))
    }

    fn state(&mut self) -> Result<StateRecord, CaptureError> {
        const SECTION: &str = "starting state";
        let (at, tag) = self.tag(SECTION)?;
        Ok(match tag {
            tag::ADDRESS_TABLE => StateRecord::AddressTable {
                space: self.space()?,
                blob: self.hash(SECTION)?,
            },
            tag::SYNCPOINTS => StateRecord::Syncpoints {
                blob: self.hash(SECTION)?,
            },
            tag::ENGINE => {
                let kind_at = self.at;
                let kind = EngineStateKind::from_value(self.u8(SECTION)?)
                    .ok_or_else(|| bad(kind_at, "an engine state's kind is unknown"))?;
                StateRecord::Engine {
                    kind,
                    owner: self.u32(SECTION)?,
                    blob: self.hash(SECTION)?,
                }
            }
            _ => {
                return Err(CaptureError::UnknownTag {
                    offset: at,
                    tag,
                    section: SECTION,
                });
            }
        })
    }

    /// The access record whose tag `tag` was read at `at`, or `None` if it is not one.
    fn access(&mut self, tag: u8) -> Result<Option<Access>, CaptureError> {
        const WHAT: &str = "an access";
        Ok(Some(match tag {
            tag::READ => {
                let space = self.space()?;
                let va = self.u64(WHAT)?;
                let len = self.u64(WHAT)?;
                let at = self.at;
                let result = match self.u8(WHAT)? {
                    sub::OK => Ok(self.hash(WHAT)?),
                    sub::FAULT => Err(self.fault()?),
                    _ => return Err(bad(at, "an access's outcome is neither 0 nor 1")),
                };
                Access::Read {
                    space,
                    va,
                    len,
                    result,
                }
            }
            tag::WRITE => {
                let space = self.space()?;
                let va = self.u64(WHAT)?;
                let data = self.hash(WHAT)?;
                let at = self.at;
                let result = match self.u8(WHAT)? {
                    sub::OK => Ok(()),
                    sub::FAULT => Err(self.fault()?),
                    _ => return Err(bad(at, "an access's outcome is neither 0 nor 1")),
                };
                Access::Write {
                    space,
                    va,
                    data,
                    result,
                }
            }
            tag::GPU_ONLY => Access::GpuOnly {
                space: self.space()?,
                va: self.u64(WHAT)?,
                len: self.u64(WHAT)?,
                answer: self.bool(WHAT)?,
            },
            tag::GENERATION => Access::Generation {
                space: self.space()?,
                value: self.u64(WHAT)?,
            },
            _ => return Ok(None),
        }))
    }

    fn submit(&mut self) -> Result<FrontRecord, CaptureError> {
        const SECTION: &str = "a submission";
        let channel = ChannelId(self.u32(SECTION)?);
        let wait_at = self.at;
        let wait = match self.u8(SECTION)? {
            sub::NONE => None,
            sub::SOME => Some(Fence {
                id: self.syncpoint()?,
                value: self.u32(SECTION)?,
            }),
            _ => return Err(bad(wait_at, "a presence flag is neither 0 nor 1")),
        };
        let increments = self.u32(SECTION)?;
        let wfi = self.bool(SECTION)?;
        let mut entries = Vec::new();
        for _ in 0..self.count(SECTION)? {
            entries.push(GpfifoEntry(self.u64(SECTION)?));
        }
        Ok(FrontRecord::Submit {
            channel,
            submission: Submission {
                entries,
                wait,
                increments,
                wfi,
            },
        })
    }

    fn update(&mut self) -> Result<FrontRecord, CaptureError> {
        const SECTION: &str = "an update";
        let space = self.space()?;
        let kind_at = self.at;
        let update = match self.u8(SECTION)? {
            sub::UNMAP => VmUpdate::Unmap {
                range: self.range()?,
            },
            sub::MODIFY_KIND => VmUpdate::ModifyKind {
                range: self.range()?,
                kind: Kind(self.u8(SECTION)?),
            },
            sub::DESTROY_MEMORY => VmUpdate::DestroyMemory {
                range: self.range()?,
            },
            sub::DESTROY_SPARSE => VmUpdate::DestroySparseRegion {
                range: self.range()?,
                region: RegionId(self.u32(SECTION)?),
            },
            _ => return Err(bad(kind_at, "an update's kind is unknown")),
        };
        let ticket_at = self.at;
        let ticket = match self.u8(SECTION)? {
            sub::NONE => None,
            sub::SOME => Some(Ticket(self.u64(SECTION)?)),
            _ => return Err(bad(ticket_at, "a presence flag is neither 0 nor 1")),
        };
        Ok(FrontRecord::Update {
            space,
            update,
            ticket,
        })
    }

    fn present(&mut self) -> Result<FrontRecord, CaptureError> {
        const SECTION: &str = "a present";
        Ok(FrontRecord::Present {
            request: PresentRequest {
                image: ImageDesc {
                    address: self.u64(SECTION)?,
                    width: self.u32(SECTION)?,
                    height: self.u32(SECTION)?,
                    pitch: self.u32(SECTION)?,
                    block_height_log2: self.u32(SECTION)?,
                    format: self.u32(SECTION)?,
                },
                shown: Rect {
                    x: self.u32(SECTION)?,
                    y: self.u32(SECTION)?,
                    width: self.u32(SECTION)?,
                    height: self.u32(SECTION)?,
                },
                layer: LayerId(self.u64(SECTION)?),
                display: DisplayId(self.u64(SECTION)?),
            },
        })
    }

    fn front(&mut self) -> Result<FrontRecord, CaptureError> {
        const SECTION: &str = "front log";
        let (at, tag) = self.tag(SECTION)?;
        Ok(match tag {
            tag::TURN => FrontRecord::Turn,
            tag::SUBMIT => self.submit()?,
            tag::UPDATE => self.update()?,
            tag::BARRIER => FrontRecord::Barrier {
                ticket: Ticket(self.u64(SECTION)?),
            },
            tag::PRESENT => self.present()?,
            tag::DESTROY => FrontRecord::Destroy {
                channel: ChannelId(self.u32(SECTION)?),
            },
            tag::REPORT => {
                let released = self.bool(SECTION)?;
                let incremented = self.bool(SECTION)?;
                let mut markers = Vec::new();
                for _ in 0..self.count(SECTION)? {
                    markers.push(MarkerToken(self.u64("a report")?));
                }
                FrontRecord::Report {
                    markers,
                    released,
                    incremented,
                }
            }
            tag::SYNCPOINT => FrontRecord::Syncpoint {
                id: self.syncpoint()?,
                value: self.u32(SECTION)?,
            },
            tag::LATCH => FrontRecord::Latch {
                channel: ChannelId(self.u32(SECTION)?),
            },
            tag::CLOCK => FrontRecord::Clock {
                now: Duration::from_nanos(self.u64(SECTION)?),
            },
            _ => match self.access(tag)? {
                Some(access) => FrontRecord::Access(access),
                None => {
                    return Err(CaptureError::UnknownTag {
                        offset: at,
                        tag,
                        section: SECTION,
                    });
                }
            },
        })
    }

    fn back(&mut self) -> Result<BackRecord, CaptureError> {
        const SECTION: &str = "back log";
        let (at, tag) = self.tag(SECTION)?;
        Ok(match tag {
            tag::APPLY => BackRecord::Apply,
            tag::CLOSE => BackRecord::Close,
            tag::COMPLETE => BackRecord::Complete {
                completed: self.u64(SECTION)?,
            },
            tag::ELAPSED => BackRecord::Elapsed {
                elapsed: Duration::from_nanos(self.u64(SECTION)?),
            },
            tag::TIMESTAMP => BackRecord::Timestamp {
                value: self.u64(SECTION)?,
            },
            tag::WAITERS => {
                let mut waiters = Vec::new();
                for _ in 0..self.count(SECTION)? {
                    let channel = ChannelId(self.u32("a waiter")?);
                    let space = self.space()?;
                    let address = self.u64("a waiter")?;
                    let op_at = self.at;
                    let op = acquire_from(self.u8("a waiter")?)
                        .ok_or_else(|| bad(op_at, "an acquire operation is unknown"))?;
                    let payload = self.u32("a waiter")?;
                    let latched = self.bool("a waiter")?;
                    waiters.push((
                        Waiter {
                            channel,
                            space,
                            address,
                            op,
                            payload,
                        },
                        latched,
                    ));
                }
                BackRecord::Waiters { waiters }
            }
            tag::DRAIN => {
                let space = self.space()?;
                let mut ranges = Vec::new();
                for _ in 0..self.count(SECTION)? {
                    ranges.push(self.range()?);
                }
                BackRecord::Drain { space, ranges }
            }
            tag::SHADER_LOOKUP => {
                let mut requests = Vec::new();
                for _ in 0..self.count(SECTION)? {
                    let stage_at = self.at;
                    let stage = stage_from(self.u8("a shader request")?)
                        .ok_or_else(|| bad(stage_at, "a shader stage is unknown"))?;
                    requests.push(ProgramRequest {
                        stage,
                        address: self.u64("a shader request")?,
                    });
                }
                BackRecord::ShaderLookup {
                    requests,
                    answer: self.answer()?,
                }
            }
            tag::SHADER_WAIT => BackRecord::ShaderWait {
                ticket: self.u64(SECTION)?,
                answer: self.answer()?,
            },
            _ => match self.access(tag)? {
                Some(access) => BackRecord::Access(access),
                None => {
                    return Err(CaptureError::UnknownTag {
                        offset: at,
                        tag,
                        section: SECTION,
                    });
                }
            },
        })
    }

    fn output(&mut self) -> Result<Output, CaptureError> {
        const SECTION: &str = "outputs";
        let (at, tag) = self.tag(SECTION)?;
        Ok(match tag {
            tag::RING => Output::Ring {
                hash: self.hash(SECTION)?,
            },
            tag::RENDER => Output::Render {
                text: self.hash(SECTION)?,
            },
            _ => {
                return Err(CaptureError::UnknownTag {
                    offset: at,
                    tag,
                    section: SECTION,
                });
            }
        })
    }

    fn blobs(&mut self) -> Result<Blobs, CaptureError> {
        let mut blobs = Blobs::new();
        let mut last: Option<ContentHash> = None;
        for _ in 0..self.count("the blob table")? {
            let at = self.at;
            let hash = self.hash("a blob")?;
            if last.is_some_and(|last| hash <= last) {
                return Err(CaptureError::BlobOrder { offset: at });
            }
            last = Some(hash);
            let len = usize::try_from(self.u64("a blob")?).unwrap_or(usize::MAX);
            let bytes = self.take(len, "a blob")?;
            if ContentHash::of(bytes) != hash {
                return Err(CaptureError::CorruptBlob { hash });
            }
            blobs.by_hash.insert(hash, bytes.to_vec());
        }
        Ok(blobs)
    }
}

impl Capture {
    /// The capture as a file.
    ///
    /// # Errors
    ///
    /// [`CaptureError::BuildIdTooLong`], [`CaptureError::TooMany`] for a list past a 32-bit
    /// count, and [`CaptureError::MissingBlob`] (then [`CaptureError::ReadLength`]) when a
    /// record names a blob the table does not hold (or a blob the wrong length), so that every
    /// file written reads back.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CaptureError> {
        self.check_blobs()?;
        let mut out = Out(Vec::new());
        out.header(&self.header)?;
        out.count(self.state.len(), "the starting state")?;
        for record in &self.state {
            out.state(record);
        }
        out.count(self.front.len(), "the front log")?;
        for record in &self.front {
            out.front(record)?;
        }
        out.count(self.back.len(), "the back log")?;
        for record in &self.back {
            out.back(record)?;
        }
        out.count(self.blobs.len(), "the blob table")?;
        for (hash, bytes) in &self.blobs.by_hash {
            out.hash(*hash);
            out.u64(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
            out.0.extend_from_slice(bytes);
        }
        out.count(self.outputs.len(), "the outputs")?;
        for output in &self.outputs {
            out.output(output);
        }
        Ok(out.0)
    }

    /// Reads a capture: the header first (magic, then version), then the starting state, the
    /// front log, the back log, the blob table and the outputs, each a 32-bit count of records,
    /// with no bytes after them. Then every blob a record names must be in the table, and every
    /// read's length must be its blob's.
    ///
    /// # Errors
    ///
    /// [`CaptureError`]: the first problem in file order (the header's in the order above).
    pub fn parse(bytes: &[u8]) -> Result<Self, CaptureError> {
        if bytes.len() > MAX_CAPTURE_BYTES {
            return Err(CaptureError::TooLarge { len: bytes.len() });
        }
        let mut input = In { bytes, at: 0 };
        let header = input.header()?;
        let mut state = Vec::new();
        for _ in 0..input.count("the starting state")? {
            state.push(input.state()?);
        }
        let mut front = Vec::new();
        for _ in 0..input.count("the front log")? {
            front.push(input.front()?);
        }
        let mut back = Vec::new();
        for _ in 0..input.count("the back log")? {
            back.push(input.back()?);
        }
        let blobs = input.blobs()?;
        let mut outputs = Vec::new();
        for _ in 0..input.count("the outputs")? {
            outputs.push(input.output()?);
        }
        if input.at < bytes.len() {
            return Err(CaptureError::TrailingBytes {
                offset: input.at,
                count: bytes.len() - input.at,
            });
        }
        let capture = Self {
            header,
            state,
            front,
            back,
            blobs,
            outputs,
        };
        capture.check_blobs()?;
        Ok(capture)
    }

    /// Every blob a record names, with the length a read gives it, in file order.
    fn named(&self) -> Vec<(ContentHash, Option<u64>)> {
        let mut named = vec![(self.header.caps, None)];
        for record in &self.state {
            let (StateRecord::AddressTable { blob, .. }
            | StateRecord::Syncpoints { blob }
            | StateRecord::Engine { blob, .. }) = *record;
            named.push((blob, None));
        }
        let access = |access: &Access| match *access {
            Access::Read {
                len,
                result: Ok(blob),
                ..
            } => Some((blob, Some(len))),
            Access::Write { data, .. } => Some((data, None)),
            Access::Read { .. } | Access::GpuOnly { .. } | Access::Generation { .. } => None,
        };
        for record in &self.front {
            if let FrontRecord::Access(a) = record {
                named.extend(access(a));
            }
        }
        for record in &self.back {
            if let BackRecord::Access(a) = record {
                named.extend(access(a));
            }
        }
        for output in &self.outputs {
            if let Output::Render { text } = *output {
                named.push((text, None));
            }
        }
        named
    }

    fn check_blobs(&self) -> Result<(), CaptureError> {
        for (hash, len) in self.named() {
            let Some(bytes) = self.blobs.get(hash) else {
                return Err(CaptureError::MissingBlob { hash });
            };
            if let Some(len) = len
                && u64::try_from(bytes.len()).ok() != Some(len)
            {
                return Err(CaptureError::ReadLength {
                    len,
                    hash,
                    found: bytes.len(),
                });
            }
        }
        Ok(())
    }
}
