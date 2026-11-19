//! The shader cache: three layers, invalidation, the negative cache, and translations on the
//! shared compile pool behind tickets (`docs/specs/shader-pipeline.md`, "Identity, caching and
//! invalidation", "Compilation, threads and stutter" and "What the GPU front/back protocol and
//! engine state (#12) gets"; port slice 19; Eden-NX2 rows 90 and 112).
//!
//! [`ShaderCache`] is what the GPU crate's back thread asks for shaders on every draw that
//! changes a program. Translation itself (decoding, flow, structuring, lowering) is not all built
//! yet, so the cache drives it through the [`Translator`] seam; slice 17's translator implements
//! it, and until then a fake does in the tests (the slice table allows it: "a fake translator
//! before 17").
//!
//! # The three layers
//!
//! 1. **Address map**: `(stage, program address)` to the program's content hash, or to why it
//!    could not be captured. On a miss the cache reads the program through [`ProgramMemory`]
//!    and captures it ([`crate::capture`], slice 3). Each entry remembers every guest byte the
//!    capture read, which [`ShaderCache::invalidate`] and [`ShaderCache::tracked_ranges`] use.
//! 2. **Program entries**: content hash to what [`Translator::program`] made of the captured
//!    bytes (the decoded, structured IR and the questions it asks), or to its rejection.
//! 3. **Shader entries**: keyed by a [`ShaderStamp`], the digest of the translator version
//!    ([`TRANSLATOR_VERSION`]), the [`HostTarget`], the stage, the content hash, the answers and
//!    the variant key; each is pending, ready or rejected.
//!
//! # A lookup
//!
//! [`ShaderCache::lookup`] never blocks and never translates. For each request, in order:
//!
//! 1. the address map gives the content hash, capturing on a miss;
//! 2. the program entries give the program, calling [`Translator::program`] on a miss;
//! 3. slice 9's question loop ([`crate::exchange::exchange`]) answers the program's questions
//!    from [`EngineState`], on every lookup, so a shader that depends on a constant-buffer word
//!    is found again only while the word is unchanged (spec, "Translation inputs");
//! 4. the shader entry is found, or created pending with its translation queued on the
//!    [`Executor`].
//!
//! The first request that is rejected at any step ends the lookup with
//! [`CacheLookup::Rejected`], naming that request's program; the requests after it are not
//! looked at. Otherwise the set is [`CacheLookup::Ready`] when every shader is, and
//! [`CacheLookup::Pending`] with one ticket when any is still being translated. Every lookup of
//! the same requests with the same shader entries while they are pending gets the same ticket,
//! and a shader already being translated is never queued again (spec: "Duplicate requests
//! coalesce").
//!
//! [`ShaderCache::wait`] blocks until a ticket's set is settled and gives the result, as a
//! lookup would; [`ShaderCache::wait_timeout`] gives up after a time. A ticket keeps the shader
//! of every member a lookup of its set found ready, so a lookup whose own miss flushes the
//! settled entries (module docs, "Limits"), its set's ready members included, still waits to
//! [`CacheLookup::Ready`] with the shaders it found; pinned by
//! `tests::races::a_lookup_that_flushes_its_own_ready_member_still_waits_to_ready`.
//!
//! The stamp's key bytes are written into one scratch buffer the cache keeps, so a lookup does
//! not allocate them. A lookup does allocate its member list and its ready list, each one entry
//! per request (at most the seven stages of a pipeline from G18): the ready list is the
//! [`CacheLookup::Ready`] it returns, and a pending set's member list becomes its ticket's.
//! These two allocations per lookup are accepted.
//!
//! # Invalidation
//!
//! [`ShaderCache::invalidate`] drops the address-map entries (positive and negative) any of
//! whose bytes lie in the range, and nothing else: program and shader entries are
//! content-addressed, so the same bytes written back cost a capture and a hash, not a
//! translation. An entry's bytes are `[address, end)`, `end` being the first address past every
//! byte the capture read or tried to read, so a range that ends exactly at an entry's address,
//! or starts exactly at its end, leaves it.
//!
//! # The negative cache
//!
//! A rejection is remembered where its cause lies (spec, "Negative cache"; rows 90 and 112):
//!
//! - a program that cannot be captured, in the address map, until a range holding a byte it
//!   read or tried to read is invalidated (below, "A refusal kept after a map");
//! - a program [`Translator::program`] rejects, in the program entries, by content hash: bytes
//!   written back unchanged are rejected again without asking the translator;
//! - a shader the translation rejects, or that needs a device feature the target lacks
//!   ([`CacheRefusal::Unsupported`]; spec: "The cache never hands out a module that needs a
//!   feature the target lacks"), in the shader entries.
//!
//! Each is counted once in [`CacheCounters::rejections`], when it is made. A program whose
//! questions take too many rounds ([`Rejection::Questions`]) depends on engine state, so it is
//! asked again on every lookup, and counted once per program entry.
//!
//! The rejection names the failing program ([`ProgramRejection`]: the request and its content
//! hash), so the GPU crate can log and count a rejection once per program (owner's ruling 10 of
//! 2026-11-19). The reason includes the first unsupported instruction and its index where the
//! translator's rejection has them ([`Rejection::Decode`]).
//!
//! ## A refusal kept after a map
//!
//! Only [`ShaderCache::invalidate`] drops an address-map entry, and G18 calls it for the CPU
//! writes it drains, for GPU writes, and for the virtual-memory updates it applies (unmap,
//! modify-kind, destroy memory, destroy sparse region): never for a map. A capture refused
//! because a byte it tried could not be read ([`crate::capture::CaptureError`]'s
//! `HeaderUnreadable` and `InstructionUnreadable`) is therefore kept when that memory is mapped
//! later with no write after it, and a program used before its memory was mapped stays rejected
//! for the rest of the run. G18's `GuestPrograms` reports every read fault as unmapped (an
//! unmapped page, an out-of-range read, a guest fault, a shim write), so each of those causes is
//! kept the same way. Pinned by
//! `tests::rejections::a_refused_capture_stays_refused_when_its_memory_is_mapped_without_a_write`.
//!
//! This departs from the spec, whose address map is "Valid until any byte of the program's
//! captured range is written or unmapped" and whose negative cache keeps a rejection "until its
//! memory is written": neither says a refusal outlives a map of the bytes it could not read.
//! An **owner question**, with two ways out: the GPU crate invalidates the mapped range on a
//! map as it does on an unmap, or the cache does not keep refusals caused by unreadable bytes
//! (`HeaderUnreadable`, `InstructionUnreadable`) and captures them again on every lookup.
//!
//! # Translations and the executor
//!
//! Translations run on `core`'s shared compile pool through [`Executor`]; the crate owns no
//! threads (review guide, M16). A worker posts each result, and the back thread takes the
//! results when it next looks up or waits. A job the executor drops unrun, or whose translator
//! panics, settles as [`CacheRefusal::Abandoned`] (`jobs`' `Settle`), so no wait is left
//! hanging. Lookups take the results' lock only when a worker has posted since the last one
//! took them.
//!
//! # Limits
//!
//! A hostile guest can write new programs for ever, so every layer is capped (all **guesses**,
//! to be set from title captures):
//!
//! | Layer | Cap | At the cap |
//! |---|---|---|
//! | Address map | [`MAX_ADDRESSES`] entries | every entry is dropped |
//! | Programs | [`MAX_PROGRAMS`] entries and [`MAX_PROGRAM_BYTES`] of captured bytes | every entry is dropped |
//! | Shaders | [`MAX_SHADERS`] entries and [`MAX_SHADER_BYTES`], each charged its program's captured bytes | every ready and rejected entry is dropped; if the pending ones still fill it, the request is refused, not remembered: [`CacheRefusal::Full`] when the count binds (whether or not the bytes do too), else [`CacheRefusal::OverBudget`] |
//! | Tickets | [`MAX_TICKETS`] | tickets whose sets have nothing pending are dropped, then the oldest |
//!
//! Captured bytes stand for the memory a program entry and a shader hold, which grows with the
//! program (spec: "every stage is linear or close to it in the program size"). A flush costs
//! captures and translations, never a wrong shader. A ticket whose set lost to a flush a shader
//! that no lookup of the set found ready (it finished after the last lookup), or that was
//! dropped, answers [`CacheLookup::Pending`], which the GPU crate takes as a translation that
//! never finished.
//!
//! The shaders a ticket keeps are outside the shader entries' caps once a flush drops their
//! entries: at most [`MAX_TICKETS`] tickets, each keeping at most one shader per request of its
//! set, outlive a flush until their tickets are spent or dropped.
//!
//! # G18's seam
//!
//! The cache is adapted to the GPU crate's `ShaderLookup` seam (gpu-protocol G18; owner's
//! ruling 11 of 2026-11-19): [`ShaderCache::lookup`] and [`ShaderCache::wait`] take `&mut
//! self` as the seam's methods do, each ready shader carries its [`ShaderStamp`] (ruling 4), and
//! the translator's [`Translator::Shader`] carries whatever the GPU crate needs beside the
//! module, its resource table included, which slice 17 adds to the host shader. `wait(ticket)`
//! is the seam's.
//!
//! # Where this differs from the spec
//!
//! - **`&mut self`, and no generation counter.** The spec's sketch has `lookup(&self, ..)` and
//!   `invalidate(&self, ..)`, and says the address map is read "without locks on the hot path (a
//!   generation counter per entry, bumped by invalidation)". G18's seam takes `&mut self`, and
//!   both calls come from the back thread only (review guide, M7), so no read of the map can run
//!   while an invalidation does: the map is a plain ordered map with no lock and no counter.
//!   The one race left, between workers finishing and the back thread, goes through the results
//!   inbox. An **owner question**: whether the counter is still wanted with the seam's `&mut
//!   self`; the tests of a fixed schedule of finishes, writes and lookups are
//!   `tests::races::*`.
//! - **The stage is part of the shader key.** The spec's key has no stage, and the content hash
//!   is seeded only with the header's length, so a vertex program and a pixel program with the
//!   same bytes share a hash; their shaders differ.
//! - **Program entries are made on the back thread**, at a program-entry miss, because the
//!   question loop needs the program's questions there and `wait` has no engine state to answer
//!   them with. [`Translator::program`] does the stage-independent part only (decoding and
//!   structuring); the rest runs on the executor.
//! - **A refused capture outlives a map of the bytes it could not read**: an owner question
//!   (above, "A refusal kept after a map").
//! - **Jump tables do not extend the capture yet** (row 92): [`Translator::program`] gets the
//!   capture as slice 3 makes it. Driving [`crate::capture::Capture::resolve_jump_tables`] with
//!   the answers needs slice 10's table finding through the seam.

mod address;
mod jobs;
mod stamp;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use thiserror::Error;

pub use address::MAX_EXTENT;
pub use jobs::{Executor, TranslateJob};
pub use stamp::ShaderStamp;

use address::{AddressEntry, AddressMap, Captured, GuestReader};
use jobs::{Inbox, Settle};
use stamp::ShaderKey;

use crate::capture::{Capture, ContentHash, capture};
use crate::exchange::{Answers, EngineState, Questioner, exchange};
use crate::host::{HostShader, Requirements};
use crate::lookup::{ProgramMemory, ProgramRequest, Ticket};
use crate::program::Rejection;
use crate::translate::GuestStage;
use crate::variant::VariantKey;

/// The translator's version, part of every shader key: bumped on any change to translation
/// output, as PR #4588 bumps `NATIVE_TRANSLATOR_SALT`
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/native.h`).
pub const TRANSLATOR_VERSION: u32 = 1;

/// The most address-map entries (a **guess**; module docs, "Limits").
pub const MAX_ADDRESSES: usize = 65_536;

/// The most program entries (a **guess**; module docs, "Limits").
pub const MAX_PROGRAMS: usize = 65_536;

/// The most captured bytes the program entries may hold together (a **guess**, 256 MiB; module
/// docs, "Limits").
pub const MAX_PROGRAM_BYTES: u64 = 256 << 20;

/// The most shader entries (a **guess**, the GPU crate's `MAX_PIPELINES`; module docs,
/// "Limits").
pub const MAX_SHADERS: usize = 65_536;

/// The most bytes the shader entries may be charged together (a **guess**, 256 MiB; module docs,
/// "Limits").
pub const MAX_SHADER_BYTES: u64 = 256 << 20;

/// The most tickets held at once (a **guess**; module docs, "Limits").
pub const MAX_TICKETS: usize = 4_096;

/// The form of code the host wants (spec, "What the render backend (#14) gets").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeForm {
    /// A validated naga module: what the wgpu backend takes. SPIR-V and MSL, for a later direct
    /// backend, come with slice 22.
    NagaModule,
}

/// What the host wants and can run, given by `core` from the render backend's caps when the
/// cache is created (spec, `HostTarget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostTarget {
    /// The form of code.
    pub form: CodeForm,
    /// The device features the host supports, in the terms of a shader's [`Requirements`].
    pub caps: Requirements,
}

/// What the cache needs of a translated shader.
pub trait Translated: Send + Sync + 'static {
    /// The device features the shader needs.
    fn requirements(&self) -> Requirements;
}

impl Translated for HostShader {
    fn requirements(&self) -> Requirements {
        self.requirements
    }
}

/// Everything one shader's translation takes (layer 3's key, with the program itself).
#[derive(Debug)]
pub struct ShaderInput<'a, P> {
    /// The stage the program runs as.
    pub stage: GuestStage,
    /// The program entry.
    pub program: &'a P,
    /// The answers to the program's questions.
    pub answers: &'a Answers,
    /// The variant key.
    pub variant: &'a VariantKey,
    /// The host target.
    pub target: &'a HostTarget,
}

/// The translation the cache drives (module docs): slice 17's translator, or a fake before it.
pub trait Translator: Send + Sync + 'static {
    /// A program entry (layer 2): the decoded, structured, stage-independent IR, which says
    /// which questions it asks.
    type Program: Questioner + Send + Sync + 'static;
    /// A translated shader, with whatever the GPU crate takes beside it.
    type Shader: Translated;

    /// The program entry for `capture`; runs on the back thread at a program-entry miss. It
    /// must not depend on the capture's stage: an entry is shared by every stage whose
    /// programs have the same bytes.
    fn program(&self, capture: &Capture) -> Result<Self::Program, Rejection>;

    /// One shader's translation; runs on a worker.
    fn shader(&self, input: &ShaderInput<'_, Self::Program>) -> Result<Self::Shader, Rejection>;
}

/// Why the cache refused a shader itself (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CacheRefusal {
    /// The translation job was dropped before it finished: the executor shut down, or the
    /// translator panicked.
    #[error("the translation stopped before it finished")]
    Abandoned,
    /// The shader needs device features the host target lacks.
    #[error("the shader needs {}, which the host lacks", feature_names(*.missing))]
    Unsupported {
        /// The features needed and not supported.
        missing: Requirements,
    },
    /// Every shader entry the cache may hold is a translation in progress. Reported when the
    /// count binds, whether or not the byte budget binds too.
    #[error(
        "the shader cache is full: all {limit} of its shader entries are translations in progress"
    )]
    Full {
        /// The cap on shader entries.
        limit: usize,
    },
    /// The translations in progress are charged so many bytes that the shader's charge does
    /// not fit the byte budget, while the count has room.
    #[error(
        "the shader cache is full: its translations in progress are charged {charged} of its \
         {limit} bytes, and this shader needs {needed} more"
    )]
    OverBudget {
        /// The bytes the translations in progress are charged together.
        charged: u64,
        /// The shader's own charge: its program's captured bytes.
        needed: u64,
        /// The byte budget of the shader entries.
        limit: u64,
    },
}

/// The features `needs` asks for that `caps` lacks, or `None` when it lacks none.
fn missing_features(needs: Requirements, caps: Requirements) -> Option<Requirements> {
    let Requirements {
        subgroup,
        f16,
        f64,
        i64,
        i64_atomics,
    } = needs;
    let missing = Requirements {
        subgroup: subgroup && !caps.subgroup,
        f16: f16 && !caps.f16,
        f64: f64 && !caps.f64,
        i64: i64 && !caps.i64,
        i64_atomics: i64_atomics && !caps.i64_atomics,
    };
    (missing != Requirements::default()).then_some(missing)
}

/// The names of the features in `features`, in plain words.
fn feature_names(features: Requirements) -> String {
    let Requirements {
        subgroup,
        f16,
        f64,
        i64,
        i64_atomics,
    } = features;
    let names: Vec<&str> = [
        (subgroup, "subgroup operations"),
        (f16, "16-bit floats"),
        (f64, "64-bit floats"),
        (i64, "64-bit integers"),
        (i64_atomics, "64-bit integer atomics"),
    ]
    .into_iter()
    .filter_map(|(set, name)| set.then_some(name))
    .collect();
    if names.is_empty() {
        "no device features".into()
    } else {
        names.join(", ")
    }
}

/// A rejection, with the program it names (owner's ruling 10 of 2026-11-19).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("the {} program at {:#x}{}: {}", .request.stage, .request.address, hash_words(.hash.as_ref()), .reason)]
pub struct ProgramRejection {
    /// The request whose program was rejected.
    pub request: ProgramRequest,
    /// The program's content hash, when it was captured.
    pub hash: Option<ContentHash>,
    /// Why.
    pub reason: Arc<Rejection>,
}

/// `hash` as the words of [`ProgramRejection`]'s message.
fn hash_words(hash: Option<&ContentHash>) -> String {
    hash.map_or_else(String::new, |hash| format!(" (content hash {hash})"))
}

/// One shader of a ready set.
#[derive(Debug)]
pub struct ReadyShader<S> {
    /// Its stamp.
    pub stamp: ShaderStamp,
    /// The shader.
    pub shader: Arc<S>,
}

impl<S> Clone for ReadyShader<S> {
    fn clone(&self) -> Self {
        Self {
            stamp: self.stamp,
            shader: Arc::clone(&self.shader),
        }
    }
}

/// What a lookup or a wait gives (G18's `SetLookup`, with the cache's own ready set).
#[derive(Debug)]
pub enum CacheLookup<S> {
    /// Every shader is ready: one per request, in request order.
    Ready(Vec<ReadyShader<S>>),
    /// Some shader is being translated.
    Pending(Ticket),
    /// A program was rejected.
    Rejected(Arc<ProgramRejection>),
}

impl<S> Clone for CacheLookup<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Ready(shaders) => Self::Ready(shaders.clone()),
            Self::Pending(ticket) => Self::Pending(*ticket),
            Self::Rejected(rejection) => Self::Rejected(Arc::clone(rejection)),
        }
    }
}

/// What the cache has done, for the run's summary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheCounters {
    /// Programs read from guest memory and captured, or refused by the capture.
    pub captures: u64,
    /// Program entries made ([`Translator::program`] calls).
    pub programs: u64,
    /// Translations queued on the executor.
    pub translations: u64,
    /// Rejections, each counted once (module docs, "The negative cache").
    pub rejections: u64,
    /// Times the address map was emptied at its cap.
    pub address_flushes: u64,
    /// Times the program entries were emptied at their cap.
    pub program_flushes: u64,
    /// Times the settled shader entries were dropped at their cap.
    pub shader_flushes: u64,
    /// Tickets dropped while their sets were still pending.
    pub tickets_dropped: u64,
}

/// The caps of every layer (module docs, "Limits").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Limits {
    addresses: usize,
    programs: usize,
    program_bytes: u64,
    shaders: usize,
    shader_bytes: u64,
    tickets: usize,
}

impl Limits {
    /// The shipping caps.
    const SHIPPING: Self = Self {
        addresses: MAX_ADDRESSES,
        programs: MAX_PROGRAMS,
        program_bytes: MAX_PROGRAM_BYTES,
        shaders: MAX_SHADERS,
        shader_bytes: MAX_SHADER_BYTES,
        tickets: MAX_TICKETS,
    };
}

/// A program entry (layer 2).
enum ProgramSlot<P> {
    /// The program, and whether a question-round rejection of it has been counted.
    Ready {
        program: Arc<P>,
        questions_counted: bool,
    },
    /// Its rejection.
    Rejected(Arc<Rejection>),
}

/// A program entry with its charge.
struct ProgramEntry<P> {
    slot: ProgramSlot<P>,
    /// Its captured bytes.
    bytes: u64,
}

/// A program as a lookup found it.
struct FoundProgram<P> {
    hash: ContentHash,
    program: Arc<P>,
    /// Its captured bytes, the charge of its shaders.
    bytes: u64,
}

/// What looking for a program gives: the program, or the content hash, if known, and the
/// rejection.
type Found<P> = Result<FoundProgram<P>, (Option<ContentHash>, Arc<Rejection>)>;

/// A shader entry's state (layer 3).
enum ShaderState<S> {
    Pending,
    Ready(Arc<S>),
    Rejected(Arc<Rejection>),
}

/// A shader entry with its charge.
struct ShaderEntry<S> {
    state: ShaderState<S>,
    charge: u64,
}

/// One request of a set, as its shader entry names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Member {
    request: ProgramRequest,
    hash: ContentHash,
    stamp: ShaderStamp,
}

/// One request resolved: its member and, when ready, its shader; or the rejection.
type Resolved<S> = Result<(Member, Option<Arc<S>>), Arc<ProgramRejection>>;

/// A pending set's ticket entry: its members, and for each member the shader a lookup of the
/// set found ready, if any, so a flush of its entry before the wait does not lose it.
struct TicketSet<S> {
    members: Vec<Member>,
    /// One per member, in order.
    held: Vec<Option<Arc<S>>>,
}

/// Where a set stands.
enum SetState<S> {
    /// Some shader is still being translated.
    Pending,
    /// Settled: what to answer.
    Done(CacheLookup<S>),
}

/// The shader cache (module docs).
pub struct ShaderCache<T: Translator> {
    translator: Arc<T>,
    executor: Arc<dyn Executor>,
    target: HostTarget,
    limits: Limits,
    addresses: AddressMap,
    programs: HashMap<ContentHash, ProgramEntry<T::Program>>,
    program_bytes: u64,
    shaders: HashMap<ShaderStamp, ShaderEntry<T::Shader>>,
    shader_bytes: u64,
    inbox: Arc<Inbox<T::Shader>>,
    tickets: BTreeMap<Ticket, TicketSet<T::Shader>>,
    set_tickets: HashMap<Vec<Member>, Ticket>,
    next_ticket: u64,
    counters: CacheCounters,
    /// The bytes of the last stamp's key, kept so a lookup does not allocate them anew.
    key_scratch: Vec<u8>,
}

impl<T: Translator> core::fmt::Debug for ShaderCache<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShaderCache")
            .field("target", &self.target)
            .field("addresses", &self.addresses.len())
            .field("programs", &self.programs.len())
            .field("shaders", &self.shaders.len())
            .field("tickets", &self.tickets.len())
            .field("counters", &self.counters)
            .finish_non_exhaustive()
    }
}

impl<T: Translator> ShaderCache<T> {
    /// An empty cache translating with `translator` on `executor` for `target`.
    pub fn new(translator: Arc<T>, executor: Arc<dyn Executor>, target: HostTarget) -> Self {
        Self::with_limits(translator, executor, target, Limits::SHIPPING)
    }

    fn with_limits(
        translator: Arc<T>,
        executor: Arc<dyn Executor>,
        target: HostTarget,
        limits: Limits,
    ) -> Self {
        Self {
            translator,
            executor,
            target,
            limits,
            addresses: AddressMap::default(),
            programs: HashMap::new(),
            program_bytes: 0,
            shaders: HashMap::new(),
            shader_bytes: 0,
            inbox: Arc::new(Inbox::new()),
            tickets: BTreeMap::new(),
            set_tickets: HashMap::new(),
            next_ticket: 1,
            counters: CacheCounters::default(),
            key_scratch: Vec::new(),
        }
    }

    /// The host target the cache translates for.
    pub const fn target(&self) -> HostTarget {
        self.target
    }

    /// What the cache has done so far.
    pub const fn counters(&self) -> CacheCounters {
        self.counters
    }

    /// The shaders of `requests` specialised on `variant`; never blocks and never translates
    /// (module docs, "A lookup").
    pub fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> CacheLookup<T::Shader> {
        self.collect();
        let mut members = Vec::with_capacity(requests.len());
        let mut ready = Vec::with_capacity(requests.len());
        for &request in requests {
            match self.resolve(request, variant, memory, state) {
                Err(rejection) => return CacheLookup::Rejected(rejection),
                Ok((member, shader)) => {
                    if let Some(shader) = shader {
                        ready.push(ReadyShader {
                            stamp: member.stamp,
                            shader,
                        });
                    }
                    members.push(member);
                }
            }
        }
        if ready.len() == members.len() {
            if let Some(ticket) = self.set_tickets.remove(members.as_slice()) {
                self.tickets.remove(&ticket);
            }
            return CacheLookup::Ready(ready);
        }
        // `ready` holds the ready members' shaders in member order; the members with equal
        // stamps share one entry, so they are all ready or all not.
        let mut ready = ready.into_iter().peekable();
        let held = members
            .iter()
            .map(|member| {
                ready
                    .next_if(|shader| shader.stamp == member.stamp)
                    .map(|shader| shader.shader)
            })
            .collect();
        CacheLookup::Pending(self.ticket_for(members, held))
    }

    /// Blocks until `ticket`'s set is settled and gives the result as a lookup would; an
    /// unknown ticket, or one whose set can no longer be assembled, answers
    /// [`CacheLookup::Pending`] at once (module docs, "Limits").
    pub fn wait(&mut self, ticket: Ticket) -> CacheLookup<T::Shader> {
        self.wait_until(ticket, None)
    }

    /// [`ShaderCache::wait`], giving up with [`CacheLookup::Pending`] after `timeout`; the
    /// ticket stays valid.
    pub fn wait_timeout(&mut self, ticket: Ticket, timeout: Duration) -> CacheLookup<T::Shader> {
        self.wait_until(ticket, Instant::now().checked_add(timeout))
    }

    /// Drops the address-map entries that read any byte of `range` (module docs,
    /// "Invalidation"). Called by the GPU back thread only.
    pub fn invalidate(&mut self, range: Range<u64>) {
        self.addresses.invalidate(&range);
    }

    /// The guest ranges the cache depends on, for the caller to write-track: sorted, with no
    /// two sharing or touching a byte.
    pub fn tracked_ranges(&self) -> Vec<Range<u64>> {
        self.addresses.tracked()
    }

    fn wait_until(&mut self, ticket: Ticket, deadline: Option<Instant>) -> CacheLookup<T::Shader> {
        loop {
            self.collect();
            let Some(set) = self.tickets.get(&ticket) else {
                return CacheLookup::Pending(ticket);
            };
            match self.set_state(ticket, set) {
                SetState::Done(result) => {
                    self.drop_ticket(ticket);
                    return result;
                }
                SetState::Pending => {
                    if !self.inbox.wait(deadline) {
                        return CacheLookup::Pending(ticket);
                    }
                }
            }
        }
    }

    /// Where `set`, ticket `ticket`, stands: the first rejected member in order, else pending
    /// while any member is, else ready; [`CacheLookup::Pending`] when a member's entry is gone
    /// and no lookup of the set found its shader ready.
    fn set_state(&self, ticket: Ticket, set: &TicketSet<T::Shader>) -> SetState<T::Shader> {
        let mut ready = Vec::with_capacity(set.members.len());
        let mut pending = false;
        let mut gone = false;
        for (member, held) in set.members.iter().zip(&set.held) {
            match self.shaders.get(&member.stamp).map(|entry| &entry.state) {
                Some(ShaderState::Rejected(reason)) => {
                    return SetState::Done(CacheLookup::Rejected(Arc::new(ProgramRejection {
                        request: member.request,
                        hash: Some(member.hash),
                        reason: Arc::clone(reason),
                    })));
                }
                Some(ShaderState::Pending) => pending = true,
                Some(ShaderState::Ready(shader)) => ready.push(ReadyShader {
                    stamp: member.stamp,
                    shader: Arc::clone(shader),
                }),
                None => match held {
                    Some(shader) => ready.push(ReadyShader {
                        stamp: member.stamp,
                        shader: Arc::clone(shader),
                    }),
                    None => gone = true,
                },
            }
        }
        if pending {
            SetState::Pending
        } else if gone {
            SetState::Done(CacheLookup::Pending(ticket))
        } else {
            SetState::Done(CacheLookup::Ready(ready))
        }
    }

    /// Settles the shader entries of every result posted since the last call.
    fn collect(&mut self) {
        for finished in self.inbox.take() {
            let Some(entry) = self.shaders.get_mut(&finished.stamp) else {
                // Cannot happen: pending entries are never dropped.
                continue;
            };
            if !matches!(entry.state, ShaderState::Pending) {
                continue;
            }
            entry.state = match finished.result {
                Ok(shader) => match missing_features(shader.requirements(), self.target.caps) {
                    None => ShaderState::Ready(Arc::new(shader)),
                    Some(missing) => {
                        self.counters.rejections = self.counters.rejections.saturating_add(1);
                        ShaderState::Rejected(Arc::new(Rejection::Cache(
                            CacheRefusal::Unsupported { missing },
                        )))
                    }
                },
                Err(reason) => {
                    self.counters.rejections = self.counters.rejections.saturating_add(1);
                    ShaderState::Rejected(Arc::new(reason))
                }
            };
        }
    }

    /// One request's member and, when ready, its shader (module docs, "A lookup").
    fn resolve(
        &mut self,
        request: ProgramRequest,
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> Resolved<T::Shader> {
        let reject = |hash, reason| {
            Arc::new(ProgramRejection {
                request,
                hash,
                reason,
            })
        };
        let found = self
            .program_of(request, memory)
            .map_err(|(hash, reason)| reject(hash, reason))?;
        let answers = match exchange(request.stage, &*found.program, state) {
            Ok(exchanged) => exchanged.answers,
            Err(error) => {
                self.count_question_rejection(found.hash);
                return Err(reject(
                    Some(found.hash),
                    Arc::new(Rejection::Questions(error)),
                ));
            }
        };
        let stamp = ShaderKey {
            version: TRANSLATOR_VERSION,
            target: &self.target,
            stage: request.stage,
            hash: found.hash,
            answers: &answers,
            variant,
        }
        .stamp_with(&mut self.key_scratch);
        let member = Member {
            request,
            hash: found.hash,
            stamp,
        };
        match self.shaders.get(&stamp).map(|entry| &entry.state) {
            Some(ShaderState::Ready(shader)) => Ok((member, Some(Arc::clone(shader)))),
            Some(ShaderState::Pending) => Ok((member, None)),
            Some(ShaderState::Rejected(reason)) => {
                Err(reject(Some(found.hash), Arc::clone(reason)))
            }
            None => {
                self.queue(request.stage, stamp, found, answers, variant)
                    .map_err(|reason| reject(Some(member.hash), reason))?;
                Ok((member, None))
            }
        }
    }

    /// The program of `request`: through the address map, capturing on a miss, and the program
    /// entries, made on a miss; or the content hash, if known, and the rejection.
    fn program_of(
        &mut self,
        request: ProgramRequest,
        memory: &dyn ProgramMemory,
    ) -> Found<T::Program> {
        if let Some(entry) = self.addresses.get(request) {
            match &entry.captured {
                Captured::Refused(reason) => return Err((None, Arc::clone(reason))),
                Captured::Program(hash) => {
                    if let Some(found) = self.program_entry(*hash) {
                        return found;
                    }
                    // The program entries were flushed: the bytes are needed again, and the
                    // capture replaces this entry.
                }
            }
        }
        self.capture(request, memory)
    }

    /// The program entry of `hash`, if there is one.
    fn program_entry(&self, hash: ContentHash) -> Option<Found<T::Program>> {
        let entry = self.programs.get(&hash)?;
        Some(match &entry.slot {
            ProgramSlot::Ready { program, .. } => Ok(FoundProgram {
                hash,
                program: Arc::clone(program),
                bytes: entry.bytes,
            }),
            ProgramSlot::Rejected(reason) => Err((Some(hash), Arc::clone(reason))),
        })
    }

    /// Captures `request`'s program, records it in the address map, and finds or makes its
    /// program entry.
    fn capture(
        &mut self,
        request: ProgramRequest,
        memory: &dyn ProgramMemory,
    ) -> Found<T::Program> {
        self.counters.captures = self.counters.captures.saturating_add(1);
        if self.addresses.len() >= self.limits.addresses {
            self.addresses.clear();
            self.counters.address_flushes = self.counters.address_flushes.saturating_add(1);
        }
        let reader = GuestReader::new(memory, request.address);
        let captured = capture(&reader, request.stage);
        let end = reader.end();
        let capture = match captured {
            Ok(capture) => capture,
            Err(error) => {
                let reason = Arc::new(Rejection::Capture(error));
                self.addresses.insert(
                    request,
                    AddressEntry {
                        end,
                        captured: Captured::Refused(Arc::clone(&reason)),
                    },
                );
                self.counters.rejections = self.counters.rejections.saturating_add(1);
                return Err((None, reason));
            }
        };
        let hash = capture.hash();
        self.addresses.insert(
            request,
            AddressEntry {
                end,
                captured: Captured::Program(hash),
            },
        );
        if let Some(found) = self.program_entry(hash) {
            return found;
        }
        self.make_program(hash, &capture)
    }

    /// Makes the program entry of `capture`, whose hash is `hash`.
    fn make_program(&mut self, hash: ContentHash, capture: &Capture) -> Found<T::Program> {
        let bytes = u64::try_from(capture.bytes().len()).unwrap_or(u64::MAX);
        let over_bytes = self
            .program_bytes
            .checked_add(bytes)
            .is_none_or(|total| total > self.limits.program_bytes);
        if self.programs.len() >= self.limits.programs || over_bytes {
            self.programs.clear();
            self.program_bytes = 0;
            self.counters.program_flushes = self.counters.program_flushes.saturating_add(1);
        }
        self.counters.programs = self.counters.programs.saturating_add(1);
        let (slot, result) = match self.translator.program(capture) {
            Ok(program) => {
                let program = Arc::new(program);
                (
                    ProgramSlot::Ready {
                        program: Arc::clone(&program),
                        questions_counted: false,
                    },
                    Ok(FoundProgram {
                        hash,
                        program,
                        bytes,
                    }),
                )
            }
            Err(reason) => {
                self.counters.rejections = self.counters.rejections.saturating_add(1);
                let reason = Arc::new(reason);
                (
                    ProgramSlot::Rejected(Arc::clone(&reason)),
                    Err((Some(hash), reason)),
                )
            }
        };
        self.program_bytes = self.program_bytes.saturating_add(bytes);
        self.programs.insert(hash, ProgramEntry { slot, bytes });
        result
    }

    /// Counts a question-round rejection of the program `hash` once per program entry.
    fn count_question_rejection(&mut self, hash: ContentHash) {
        if let Some(ProgramEntry {
            slot: ProgramSlot::Ready {
                questions_counted, ..
            },
            ..
        }) = self.programs.get_mut(&hash)
            && !*questions_counted
        {
            *questions_counted = true;
            self.counters.rejections = self.counters.rejections.saturating_add(1);
        }
    }

    /// Makes a pending shader entry `stamp` and queues its translation.
    fn queue(
        &mut self,
        stage: GuestStage,
        stamp: ShaderStamp,
        found: FoundProgram<T::Program>,
        answers: Answers,
        variant: &VariantKey,
    ) -> Result<(), Arc<Rejection>> {
        self.make_room_for_shader(found.bytes)?;
        self.shaders.insert(
            stamp,
            ShaderEntry {
                state: ShaderState::Pending,
                charge: found.bytes,
            },
        );
        self.shader_bytes = self.shader_bytes.saturating_add(found.bytes);
        self.counters.translations = self.counters.translations.saturating_add(1);
        let translator = Arc::clone(&self.translator);
        let target = self.target;
        let variant = variant.clone();
        let program = found.program;
        let settle = Settle::new(Arc::clone(&self.inbox), stamp);
        self.executor.spawn(Box::new(move || {
            let result = translator.shader(&ShaderInput {
                stage,
                program: &*program,
                answers: &answers,
                variant: &variant,
                target: &target,
            });
            settle.post(result);
        }));
        Ok(())
    }

    /// Makes room for one more shader entry charged `charge`: drops the settled entries when
    /// the layer is at a cap, and refuses when the pending ones still fill it.
    fn make_room_for_shader(&mut self, charge: u64) -> Result<(), Arc<Rejection>> {
        let fits = |cache: &Self| {
            cache.shaders.len() < cache.limits.shaders
                && cache
                    .shader_bytes
                    .checked_add(charge)
                    .is_some_and(|total| total <= cache.limits.shader_bytes)
        };
        if fits(self) {
            return Ok(());
        }
        self.shaders
            .retain(|_, entry| matches!(entry.state, ShaderState::Pending));
        self.shader_bytes = self
            .shaders
            .values()
            .fold(0, |total: u64, entry| total.saturating_add(entry.charge));
        self.counters.shader_flushes = self.counters.shader_flushes.saturating_add(1);
        let refusal = if self.shaders.len() >= self.limits.shaders {
            CacheRefusal::Full {
                limit: self.limits.shaders,
            }
        } else if fits(self) {
            return Ok(());
        } else {
            CacheRefusal::OverBudget {
                charged: self.shader_bytes,
                needed: charge,
                limit: self.limits.shader_bytes,
            }
        };
        Err(Arc::new(Rejection::Cache(refusal)))
    }

    /// The ticket of the pending set `members`, whose ready shaders this lookup found are
    /// `held`: the one it already has, keeping every shader a lookup of it found, or a new one.
    fn ticket_for(&mut self, members: Vec<Member>, held: Vec<Option<Arc<T::Shader>>>) -> Ticket {
        if let Some(&ticket) = self.set_tickets.get(members.as_slice()) {
            if let Some(set) = self.tickets.get_mut(&ticket) {
                for (kept, found) in set.held.iter_mut().zip(held) {
                    if found.is_some() {
                        *kept = found;
                    }
                }
            }
            return ticket;
        }
        if self.tickets.len() >= self.limits.tickets {
            self.trim_tickets();
        }
        let ticket = Ticket::new(self.next_ticket);
        self.next_ticket = self.next_ticket.saturating_add(1);
        self.tickets.insert(
            ticket,
            TicketSet {
                members: members.clone(),
                held,
            },
        );
        self.set_tickets.insert(members, ticket);
        ticket
    }

    /// Drops the tickets whose sets have nothing pending, then, while still at the cap, the
    /// oldest.
    fn trim_tickets(&mut self) {
        let shaders = &self.shaders;
        let settled: Vec<Ticket> = self
            .tickets
            .iter()
            .filter(|(_, set)| {
                !set.members.iter().any(|member| {
                    matches!(
                        shaders.get(&member.stamp).map(|entry| &entry.state),
                        Some(ShaderState::Pending)
                    )
                })
            })
            .map(|(ticket, _)| *ticket)
            .collect();
        for ticket in settled {
            self.drop_ticket(ticket);
        }
        while self.tickets.len() >= self.limits.tickets {
            let Some(&oldest) = self.tickets.keys().next() else {
                break;
            };
            self.drop_ticket(oldest);
            self.counters.tickets_dropped = self.counters.tickets_dropped.saturating_add(1);
        }
    }

    fn drop_ticket(&mut self, ticket: Ticket) {
        if let Some(set) = self.tickets.remove(&ticket) {
            self.set_tickets.remove(&set.members);
        }
    }
}
