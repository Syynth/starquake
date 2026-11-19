//! The back thread's executor (gpu-protocol spec, slice G19: "The back thread: engine state",
//! "What the back hands the render backend", "Ordering guarantees" and "Threads and data flow"):
//! it takes ring entries in order and hands each to the engine it is for, and the commands the
//! engines make to the fence path's open list.
//!
//! No reference is ported here: the routing is new code over the slices it joins (G10's mirrors,
//! G11's copy engines, G12's QMD scheduler, G15's fence path, G16's and G17's caches, G18's draw
//! path and G23's pipeline-key cache).
//!
//! # What the executor owns
//!
//! - **Address spaces** ([`Executor::add_space`]): each with its [`GpuMemory`], its buffer and
//!   texture caches (each cache serves one address space, `back::buffer` and `back::texture`) and
//!   its pending inline writes ([`PendingWrites`]).
//! - **Channels** ([`Executor::add_channel`]): each in one address space, with what each of its
//!   subchannels has bound, and one **engine per class** it has ever bound: the class's
//!   [`Mirror`] and, by class, the 3D draw path ([`Draws`]) and inline upload, the compute
//!   class's inline upload, or nothing more (DMA and 2D read the mirror). An engine is made on
//!   the first bind of its class on the channel and kept across unbinds and rebinds, so its
//!   registers are not seeded again: Eden-NX2 builds each engine once per channel and keeps its
//!   registers across `SET_OBJECT` rebinds (the wiring note on issue #330;
//!   `tests::a_rebind_keeps_the_classs_registers`). Subchannel 2 starts bound to
//!   inline-to-memory ([`crate::channel::initial_binding`], row 49), with no `Bind` entry.
//!   A class other than the five below can be bound but gets no engine and no mirror, so a
//!   guest binding many classes costs nothing; its words are counted
//!   ([`ExecCounters::unhandled_words`]) and dropped.
//! - **One QMD table for every channel** ([`QmdTable`], row 84), read through the signalling
//!   channel's memory.
//! - The pipeline-key cache, the compute path's host shaders, the readback ids, and the fence
//!   path ([`Fences`]) with its open list.
//!
//! # Routing
//!
//! [`Executor::apply`] takes one entry:
//!
//! | Entry | What happens |
//! |---|---|
//! | `Bind` | the subchannel's class changes (class 0 unbinds); the class's engine is made if new |
//! | `Write`, `Run` | each word is stored in the class's mirror, then handed to the class's engine (below) |
//! | `Release` | a fenced release ([`FencedAction::Release`]) through the issuing channel's memory and address space |
//! | `Syncpoint` | a fenced increment |
//! | `Marker` | a fenced marker, with the ordered update the front filed under its token ([`OrderedTable::take`]) |
//! | `WaitForIdle`, `SetReference` | the open list is closed ([`CloseReason::WaitForIdle`], [`CloseReason::SetReference`]) |
//! | `Invalidate` | the CPU-write drain of the channel's address space (below) |
//! | `Present` | a `Command::Present` of the texture cache's view, then the list is closed ([`CloseReason::Present`]) |
//! | `CaptureControl` | counted ([`ExecCounters::captures`]); captures are slice G20's |
//!
//! A `Write` or `Run` for a class bound on none of the channel's subchannels is
//! [`ExecError::Unbound`], and one for a channel never added is [`ExecError::UnknownChannel`];
//! either changes nothing.
//!
//! Each word reaches its engine one at a time, after the mirror stored it, since the engines
//! read their registers from the mirror (an inline upload reads `LOAD_INLINE_DATA` there):
//!
//! - **3D (`C797`)**: [`Draws::method`] (draw triggers, conditional render), the inline upload
//!   ([`INLINE_C797`]), and each run of `LOAD_ROOT_TABLE` words to [`Draws::load_root_table`]
//!   as a whole.
//! - **Compute (`C7C0`)**: the inline upload ([`INLINE_C7C0`]) and [`QmdTable::compute_method`];
//!   each dispatch it decides on becomes a `Command::Dispatch` (`compute`).
//! - **DMA (`C7B5`)**: [`dma_method`]. **2D (`902D`)**: [`blit_method`]. **Inline (`A140`)**:
//!   [`INLINE_A140`].
//!
//! # Before each draw, dispatch and copy
//!
//! The CPU-write drain runs first (gpu-protocol, "Learning of CPU writes"): the buffer cache
//! drains the write watch ([`BufferCache::sync_cpu_writes`]), and every range it reports also
//! goes to the texture cache ([`TextureCache::cpu_writes`]) and the shader cache
//! ([`ShaderCache`]'s `invalidate`). Then, before a draw or dispatch, the merged inline writes
//! ([`PendingWrites::flush`]) reach the caches (a DMA or 2D launch flushes them itself). A
//! word "starts a draw" here when its method is one of the draw path's
//! ([`crate::back::draw::call::DRAW_METHODS`], [`crate::back::draw::call::UNHANDLED_METHODS`]);
//! a dispatch, when it is the last inline QMD word or `SEND_SIGNALING_PCAS2_B`; a copy, when it
//! is a `LAUNCH_DMA` or `PIXELS_FROM_MEMORY_SRC_Y0_INT`.
//!
//! Writes the back makes to guest memory go to every cache of the address space and the shader
//! cache: a copy's destination and the inline writes (through [`CopySink::invalidate`]), the
//! inline QMD's 256 bytes (the wiring note on issue #330: "invalidate the inline-QMD range in the
//! back's caches"), and a write-back's ranges. An applied ordered update drops its range from
//! that space's caches ([`BufferCache::apply_update`], [`TextureCache::apply_update`]) and goes
//! to the shader cache (gpu-protocol, "Ordered address-table updates and barriers", step 4).
//!
//! # Copies of images the texture cache holds
//!
//! When a copy reads or partly overwrites memory the texture cache holds a GPU-modified image
//! of ([`CopySink::make_current`], [`TextureCache::needs_write_back`]), the image is written
//! back first: its readbacks are recorded, the open list is closed ([`CloseReason::CatchUp`]),
//! the executor waits for it ([`Backend::wait`], at most [`ExecConfig::catch_up`]) and writes
//! the readbacks to guest memory ([`TextureCache::finish_write_back`]); then the copy runs on the
//! CPU. A write-back that fails is logged and counted ([`ExecCounters::write_back_failures`]),
//! and the copy runs on what guest memory holds.
//!
//! **This departs from the issue's scope, for the owner.** Issue #330 asks for "copies, inline
//! writes and 2D to G11, using the host copy or blit command when G17 holds either surface",
//! and the spec says such a copy "is a host copy command". Here every copy is a CPU copy, made
//! correct by writing the images back first; turning a DMA copy or 2D blit into a
//! `Command::Copy` or `Command::Blit` between the cache's textures is not built.
//!
//! # Deferred instanced draws
//!
//! Eden-NX2 draws a 3D engine's deferred instances at the end of each macro
//! (`Maxwell3D::FinishFrontMacro` and `CallMacroMethod` call `DrawDeferred`,
//! `eden-nx2/src/video_core/engines/maxwell_3d.cpp` @ 50fa86bb), which its ring carries. The
//! spec's ring has no entry for the end of a macro. Here a channel's deferred instances are
//! drawn ([`Draws::flush`]) before any entry of that channel that is not a `Write` or `Run` of
//! its 3D class, and every channel's before a `Syncpoint`, `Marker` or `Present`, so the work
//! after them in the stream comes after the draw. A **divergence**, and an owner question: a
//! ring entry for the end of a macro would let the back draw them where Eden-NX2 does.
//! Pinned by `tests::deferred_instances_are_drawn_before_the_channels_next_release`.
//!
//! # Conditional rendering
//!
//! Before the draw path reads a conditional-render report (`SET_RENDER_ENABLE_C` written with a
//! mode that reads memory), every fenced release of the channel's address space that writes the
//! report and is not yet performed is made visible ([`Fences::settle`], waiting at most
//! [`ExecConfig::catch_up`]): the spec's "after a catch-up when that value is still pending".
//! A catch-up that fails is logged and counted ([`ExecCounters::catch_up_failures`]), and the
//! report is read as guest memory holds it.
//!
//! # Lists and the caches
//!
//! The fence path closes the open list for reasons of its own (a waiter, the elapsed time, an
//! empty ring), so the executor learns of each submission afterwards
//! ([`Fences::newest_submitted`]). Before an address space's caches or an engine's draw path are
//! next used, they are told the newest fence submitted, once ([`BufferCache::submitted`],
//! [`TextureCache::submitted`], [`Draws::submitted`]): everything they used before a close went
//! into a list closed by then, and nothing has used them since, so a fence at or after theirs
//! keeps what they used alive long enough. Every use goes through this catch-up first: a draw,
//! a dispatch (before each of a signal's dispatches, since a release between two can close the
//! list for a waiter), a copy, a present and the drain
//! (`tests::a_list_closed_between_two_draws_of_one_run_keeps_their_buffers_apart`). Two 3D
//! engines drawing into one list each set their render targets: a draw path that did not make
//! the last draw is told its targets are gone the same way.
//!
//! # Presents
//!
//! A present names no address space ([`PresentRequest`] carries only a GPU address), so presents
//! read through the space [`Executor::set_present_space`] names; the wiring (slice G21) sets the
//! title's. A **guess**: one GPU address space per title. The image is a block-linear 2D image
//! of the request's size and block height, in the texture cache as any sampled texture
//! ([`present`]), so a frame the GPU rendered is found as the render target it was drawn into.
//! The frame number counts presents from 0. The command carries no fence wait (owner's
//! decision of 2026-11-12, "Presents pace the game through a buffer queue"); it is ordered after
//! every command before it by its place in the list, and the list is closed after it.
//!
//! # Not handled here
//!
//! - Writing a `LOAD_CONSTANT_BUFFER` run to guest memory (Eden's `ProcessCBMultiData`): no
//!   slice of the spec's table owns it.
//! - The compute class's texture and sampler pools: a dispatch whose shader binds a texture, a
//!   sampler or the driver constants is refused (`compute`).
//! - Indirect, inline-index, zero-index and auto draws (the draw path skips and counts them).

mod compute;
mod present;
mod sinks;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use starquake_memory::{PageRange, WriteWatch};
use starquake_render::command::{PresentDesc, PresentTransform};
use starquake_render::{Backend, CommandList, FenceValue};
use thiserror::Error;

use crate::back::buffer::{BufferCache, InvalidationSink};
use crate::back::copy::{
    CopyError, INLINE_A140, INLINE_C7C0, INLINE_C797, InlineUpload, PendingWrites, blit_method,
    dma_method,
};
use crate::back::draw::call::{DRAW_METHODS, UNHANDLED_METHODS};
use crate::back::draw::conditional;
use crate::back::draw::shaders::{HostShaders, ShaderLookup};
use crate::back::draw::{DrawContext, DrawError, Draws, Outcome, WaitWhileCompiling};
use crate::back::fence::{CloseReason, FenceError, FencedAction, Fences, Settled};
use crate::back::pipeline::PipelineCache;
use crate::back::qmd::{LAST_INLINE_QMD_METHOD, QMD_ADDRESS_SHIFT, QMD_BYTES, QmdError, QmdTable};
use crate::back::readback::ReadbackIds;
use crate::back::state::{Mirror, StateError};
use crate::back::texture::{TextureCache, TextureError};
use crate::channel::initial_binding;
use crate::classes::{cl902d, cla140, clc7b5, clc7c0, clc797};
use crate::front::OrderedTable;
use crate::interface::{AsId, ChannelId, PresentRequest};
use crate::memory::GpuMemory;
use crate::ring::{ClassId, Kind, RingEntry, SUBCHANNEL_COUNT, Subchannel};

use compute::ComputeSink;
use sinks::{CacheSink, Drained};

pub use compute::ComputeError;

/// The shader cache as the executor sees it: the draw path's lookups ([`ShaderLookup`]) and the
/// invalidations the back thread sends it for CPU writes, GPU writes and unmaps
/// ([`InvalidationSink`]; gpu-protocol, "Shader pipeline (#13)": "`invalidate` calls from the
/// back thread for CPU writes (dirty log), GPU writes and unmaps").
pub trait ShaderCache: ShaderLookup + InvalidationSink {}

impl<T: ShaderLookup + InvalidationSink + ?Sized> ShaderCache for T {}

/// How long a catch-up or a write-back waits for the host GPU by default: a **guess**, long
/// enough for any list the back records, short enough that a backend that never completes is
/// reported rather than hanging the back thread.
pub const CATCH_UP_TIMEOUT: Duration = Duration::from_secs(5);

/// The executor's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecConfig {
    /// What a draw or dispatch does while its shaders are being translated.
    pub setting: WaitWhileCompiling,
    /// How long a catch-up or write-back waits for the host GPU.
    pub catch_up: Duration,
}

impl Default for ExecConfig {
    /// [`WaitWhileCompiling::Wait`] and [`CATCH_UP_TIMEOUT`].
    fn default() -> Self {
        Self {
            setting: WaitWhileCompiling::Wait,
            catch_up: CATCH_UP_TIMEOUT,
        }
    }
}

/// Why the executor refused an entry, or a part of one.
#[derive(Debug, Error, PartialEq)]
pub enum ExecError {
    /// The entry names a channel the executor was never given.
    #[error("channel {} is not known to the back", .channel.0)]
    UnknownChannel {
        /// The channel.
        channel: ChannelId,
    },
    /// An address space the executor was never given.
    #[error("address space {} is not known to the back", .space.0)]
    UnknownSpace {
        /// The address space.
        space: AsId,
    },
    /// [`Executor::add_space`] for a space already added.
    #[error("address space {} was already added", .space.0)]
    DuplicateSpace {
        /// The address space.
        space: AsId,
    },
    /// [`Executor::add_channel`] for a channel already added.
    #[error("channel {} was already added", .channel.0)]
    DuplicateChannel {
        /// The channel.
        channel: ChannelId,
    },
    /// A `Write` or `Run` for a class bound on none of the channel's subchannels.
    #[error("channel {} wrote to class {:#x}, which none of its subchannels has bound", .channel.0, .class.0)]
    Unbound {
        /// The channel.
        channel: ChannelId,
        /// The class.
        class: ClassId,
    },
    /// The mirror refused a word.
    #[error("engine state: {0}")]
    State(#[from] StateError),
    /// A draw could not be derived.
    #[error("draw: {0}")]
    Draw(#[from] DrawError),
    /// A copy, inline upload or blit failed.
    #[error("copy: {0}")]
    Copy(#[from] CopyError),
    /// The QMD scheduler refused.
    #[error("compute: {0}")]
    Qmd(#[from] QmdError),
    /// The fence path failed.
    #[error("fence path: {0}")]
    Fence(#[from] FenceError),
    /// The texture cache refused a present's image.
    #[error("present: {0}")]
    Present(TextureError),
    /// A present's format word is not one this crate knows.
    #[error("present format {format} is not supported (1 is RGBA8, 4 is RGB10A2)")]
    PresentFormat {
        /// The format word.
        format: u32,
    },
    /// A present arrived before [`Executor::set_present_space`] named the space it reads.
    #[error("a present arrived, but no address space is set for presents")]
    NoPresentSpace,
}

/// Counts of what the executor did, for counters and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecCounters {
    /// Entries applied without an error.
    pub applied: u64,
    /// Entries (or parts of entries) refused with an [`ExecError`], each logged.
    pub failed: u64,
    /// Words written to a class with no engine here.
    pub unhandled_words: u64,
    /// `CaptureControl` entries, which slice G20 handles.
    pub captures: u64,
    /// Presents recorded.
    pub presents: u64,
    /// Dispatches recorded.
    pub dispatches: u64,
    /// Dispatches not recorded: a rejected or pending program, a failed pipeline, or a dispatch
    /// that could not be built (each QMD still completes and releases).
    pub dispatches_skipped: u64,
    /// Write-backs run before a copy.
    pub write_backs: u64,
    /// Write-backs that failed (the copy then ran on guest memory as it was).
    pub write_back_failures: u64,
    /// Conditional-render catch-ups that waited for a release.
    pub catch_ups: u64,
    /// Catch-ups that failed (the report was then read as it was).
    pub catch_up_failures: u64,
}

/// What one [`Executor::step`] did.
#[derive(Debug, PartialEq)]
pub enum Step {
    /// An entry was applied.
    Applied,
    /// An entry was refused; it was logged and counted.
    Refused(ExecError),
    /// Completion processing ran.
    Completed,
    /// The wait's timeout passed with nothing to do.
    TimedOut,
}

/// One address space's memory and caches.
struct Space {
    memory: Arc<dyn GpuMemory>,
    buffers: BufferCache,
    textures: TextureCache,
    pending: PendingWrites,
    /// The newest fence its caches were told of ([`catch_up`]).
    notified: FenceValue,
}

/// What an engine does with its class's words beside its mirror.
enum Role {
    /// The 3D class: draws and inline uploads.
    ThreeD {
        draws: Box<Draws>,
        inline: InlineUpload,
    },
    /// The compute class: inline uploads and the QMD scheduler.
    Compute { inline: InlineUpload },
    /// The DMA copy class.
    Dma,
    /// The 2D class.
    TwoD,
    /// The inline-to-memory class.
    Inline(InlineUpload),
}

/// One class's engine on one channel.
struct Engine {
    mirror: Mirror,
    role: Role,
    /// The newest fence its draw path was told of.
    notified: FenceValue,
}

impl Engine {
    /// The engine of `class`, or `None` for a class this executor has no engine for.
    fn new(
        channel: ChannelId,
        class: ClassId,
        backend: &Arc<dyn Backend>,
        setting: WaitWhileCompiling,
        notified: FenceValue,
    ) -> Option<Self> {
        let role = match class.0 {
            clc797::CLASS => Role::ThreeD {
                draws: Box::new(Draws::new(Arc::clone(backend), setting)),
                inline: InlineUpload::new(),
            },
            clc7c0::CLASS => Role::Compute {
                inline: InlineUpload::new(),
            },
            clc7b5::CLASS => Role::Dma,
            cl902d::CLASS => Role::TwoD,
            cla140::CLASS => Role::Inline(InlineUpload::new()),
            _ => return None,
        };
        Some(Self {
            mirror: Mirror::new(channel, class),
            role,
            notified,
        })
    }
}

/// The subchannels' bindings as an array index.
const SUBCHANNELS: usize = SUBCHANNEL_COUNT as usize;

/// One channel's bindings and engines.
struct Channel {
    space: AsId,
    bound: [Option<ClassId>; SUBCHANNELS],
    engines: BTreeMap<ClassId, Engine>,
}

impl Channel {
    fn is_bound(&self, class: ClassId) -> bool {
        self.bound.contains(&Some(class))
    }
}

/// The back thread's executor (module docs). Owned by the back thread.
pub struct Executor {
    backend: Arc<dyn Backend>,
    fences: Fences,
    shaders: Box<dyn ShaderCache>,
    ordered: Arc<OrderedTable>,
    pipelines: PipelineCache,
    qmds: QmdTable,
    readbacks: ReadbackIds,
    compute_shaders: HostShaders,
    spaces: BTreeMap<AsId, Space>,
    channels: BTreeMap<ChannelId, Channel>,
    config: ExecConfig,
    present_space: Option<AsId>,
    frame: u64,
    /// The 3D engine that made the open list's last draw.
    last_drawer: Option<(ChannelId, ClassId)>,
    counters: ExecCounters,
    /// The commands of the draw, dispatch, write-back or present being built.
    scratch: CommandList,
    /// The ranges of one CPU-write drain.
    drained: Vec<PageRange>,
}

impl Executor {
    /// An executor over `backend`, recording through `fences`, asking `shaders` for shaders, and
    /// taking ordered updates from `ordered` (the front's table, [`crate::front::Front`]).
    pub fn new(
        backend: Arc<dyn Backend>,
        fences: Fences,
        shaders: Box<dyn ShaderCache>,
        ordered: Arc<OrderedTable>,
        config: ExecConfig,
    ) -> Self {
        Self {
            pipelines: PipelineCache::new(Arc::clone(&backend)),
            compute_shaders: HostShaders::new(Arc::clone(&backend)),
            backend,
            fences,
            shaders,
            ordered,
            qmds: QmdTable::new(),
            readbacks: ReadbackIds::new(),
            spaces: BTreeMap::new(),
            channels: BTreeMap::new(),
            config,
            present_space: None,
            frame: 0,
            last_drawer: None,
            counters: ExecCounters::default(),
            scratch: CommandList::new(),
            drained: Vec::new(),
        }
    }

    /// Adds address space `space`, whose GPU memory is `memory` and whose process's write watch
    /// is `watch`, with empty caches.
    ///
    /// # Errors
    ///
    /// [`ExecError::DuplicateSpace`] for a space already added; nothing changes.
    pub fn add_space(
        &mut self,
        space: AsId,
        memory: Arc<dyn GpuMemory>,
        watch: Arc<WriteWatch>,
    ) -> Result<(), ExecError> {
        if self.spaces.contains_key(&space) {
            return Err(ExecError::DuplicateSpace { space });
        }
        self.spaces.insert(
            space,
            Space {
                memory,
                buffers: BufferCache::new(Arc::clone(&self.backend), Arc::clone(&watch)),
                textures: TextureCache::new(Arc::clone(&self.backend), watch),
                pending: PendingWrites::new(),
                notified: self.fences.newest_submitted(),
            },
        );
        Ok(())
    }

    /// Adds channel `channel`, in address space `space`, with subchannel 2 bound to
    /// inline-to-memory ([`initial_binding`]) and nothing else.
    ///
    /// # Errors
    ///
    /// [`ExecError::DuplicateChannel`], then [`ExecError::UnknownSpace`]; nothing changes.
    pub fn add_channel(&mut self, channel: ChannelId, space: AsId) -> Result<(), ExecError> {
        if self.channels.contains_key(&channel) {
            return Err(ExecError::DuplicateChannel { channel });
        }
        if !self.spaces.contains_key(&space) {
            return Err(ExecError::UnknownSpace { space });
        }
        let mut state = Channel {
            space,
            bound: [None; SUBCHANNELS],
            engines: BTreeMap::new(),
        };
        for (index, slot) in (0u8..).zip(state.bound.iter_mut()) {
            *slot = Subchannel::new(index).and_then(initial_binding);
        }
        for class in state.bound.iter().flatten() {
            if let Some(engine) = Engine::new(
                channel,
                *class,
                &self.backend,
                self.config.setting,
                self.fences.newest_submitted(),
            ) {
                state.engines.insert(*class, engine);
            }
        }
        self.channels.insert(channel, state);
        Ok(())
    }

    /// Forgets channel `channel` and its engines; `false` if it was not known.
    pub fn remove_channel(&mut self, channel: ChannelId) -> bool {
        if self.last_drawer.is_some_and(|(c, _)| c == channel) {
            self.last_drawer = None;
        }
        self.channels.remove(&channel).is_some()
    }

    /// Names the address space presents read their images through (module docs, "Presents").
    ///
    /// # Errors
    ///
    /// [`ExecError::UnknownSpace`] for a space not added; nothing changes.
    pub fn set_present_space(&mut self, space: AsId) -> Result<(), ExecError> {
        if !self.spaces.contains_key(&space) {
            return Err(ExecError::UnknownSpace { space });
        }
        self.present_space = Some(space);
        Ok(())
    }

    /// What the executor has done so far.
    pub fn counters(&self) -> ExecCounters {
        self.counters
    }

    /// The fence path, for its counters and lists in flight.
    pub fn fences(&self) -> &Fences {
        &self.fences
    }

    /// The class bound on `subchannel` of `channel`, if the channel is known and the subchannel
    /// bound.
    pub fn bound(&self, channel: ChannelId, subchannel: Subchannel) -> Option<ClassId> {
        let state = self.channels.get(&channel)?;
        state
            .bound
            .get(usize::from(subchannel.get()))
            .copied()
            .flatten()
    }

    /// The mirror of `class` on `channel`, if the channel has an engine for the class.
    pub fn mirror(&self, channel: ChannelId, class: ClassId) -> Option<&Mirror> {
        Some(&self.channels.get(&channel)?.engines.get(&class)?.mirror)
    }

    /// The draw path of the 3D engine on `channel`, if it has one.
    pub fn draws(&self, channel: ChannelId) -> Option<&Draws> {
        let engine = self
            .channels
            .get(&channel)?
            .engines
            .get(&ClassId(clc797::CLASS))?;
        match &engine.role {
            Role::ThreeD { draws, .. } => Some(draws),
            _ => None,
        }
    }

    /// The back thread's next piece of work, waiting at most `timeout` ([`Fences::wait`]): an
    /// entry is applied, and one refused is logged at debug level and counted
    /// ([`ExecCounters::failed`]) without stopping the thread.
    ///
    /// # Errors
    ///
    /// [`FenceError`] from the wait itself: the ring failed, or closing the open list for an
    /// empty ring was refused.
    pub fn step(
        &mut self,
        ring: &mut crate::ring::Consumer,
        timeout: Duration,
    ) -> Result<Step, FenceError> {
        let woke = self.fences.wait(ring, timeout);
        self.after_fences();
        match woke? {
            crate::back::fence::Woke::Entry(entry) => Ok(match self.apply(&entry) {
                Ok(()) => Step::Applied,
                Err(error) => Step::Refused(error),
            }),
            crate::back::fence::Woke::Completed(_) => Ok(Step::Completed),
            crate::back::fence::Woke::TimedOut => Ok(Step::TimedOut),
        }
    }

    /// Applies one ring entry (module docs, "Routing"). An error is logged at debug level and
    /// counted ([`ExecCounters::failed`]) before it is returned.
    ///
    /// # Errors
    ///
    /// [`ExecError`]; what an entry did before the error stands (the variant's source says
    /// what).
    pub fn apply(&mut self, entry: &RingEntry<'_>) -> Result<(), ExecError> {
        let result = self.route(entry);
        self.after_fences();
        match &result {
            Ok(()) => self.counters.applied = self.counters.applied.saturating_add(1),
            Err(error) => {
                self.counters.failed = self.counters.failed.saturating_add(1);
                // Debug, not warning: a guest can repeat a refused entry at will, and the count
                // is the signal.
                tracing::debug!(%error, kind = Kind::of(entry).name(), "the back refused a ring entry");
            }
        }
        result
    }

    /// Closes the open list for an empty ring, as [`Fences::wait`] does before it sleeps: for
    /// callers that drive [`Executor::apply`] themselves.
    ///
    /// # Errors
    ///
    /// [`FenceError::Submit`] when the backend refused the list.
    pub fn close(&mut self, reason: CloseReason) -> Result<Option<FenceValue>, FenceError> {
        let closed = self.fences.close(reason);
        self.after_fences();
        closed
    }

    /// Runs completion processing ([`Fences::complete`]) and hands what it applied on.
    pub fn complete(&mut self) -> crate::back::fence::Report {
        let report = self.fences.complete();
        self.after_fences();
        report
    }

    fn route(&mut self, entry: &RingEntry<'_>) -> Result<(), ExecError> {
        match entry {
            RingEntry::Bind {
                channel,
                subchannel,
                class,
            } => self.bind(*channel, *subchannel, *class),
            RingEntry::Write {
                channel,
                class,
                method,
                value,
            } => self.words(*channel, *class, *method, &[*value]),
            RingEntry::Run {
                channel,
                class,
                method,
                words,
            } => self.words(*channel, *class, *method, words),
            RingEntry::Release { channel, release } => {
                let space = self.space_of(*channel)?;
                self.flush_deferred(Some(*channel))?;
                let memory = self.memory(space)?;
                self.fences.attach(FencedAction::Release {
                    memory,
                    space,
                    release: *release,
                })?;
                Ok(())
            }
            RingEntry::Syncpoint { id } => {
                self.flush_deferred(None)?;
                self.fences.attach(FencedAction::Increment(*id))?;
                Ok(())
            }
            RingEntry::Marker { token } => {
                self.flush_deferred(None)?;
                let ordered = self.ordered.take(*token);
                self.fences.attach(FencedAction::Marker {
                    token: *token,
                    ordered,
                })?;
                Ok(())
            }
            RingEntry::WaitForIdle { channel } => {
                self.space_of(*channel)?;
                self.flush_deferred(Some(*channel))?;
                self.fences.close(CloseReason::WaitForIdle)?;
                Ok(())
            }
            RingEntry::SetReference { channel, .. } => {
                self.space_of(*channel)?;
                self.flush_deferred(Some(*channel))?;
                self.fences.close(CloseReason::SetReference)?;
                Ok(())
            }
            RingEntry::Invalidate { channel, .. } => {
                let space = self.space_of(*channel)?;
                self.flush_deferred(Some(*channel))?;
                self.drain(space)
            }
            RingEntry::Present(request) => {
                self.flush_deferred(None)?;
                self.present(request)
            }
            RingEntry::CaptureControl(_) => {
                self.counters.captures = self.counters.captures.saturating_add(1);
                Ok(())
            }
        }
    }

    /// The address space of `channel`.
    fn space_of(&self, channel: ChannelId) -> Result<AsId, ExecError> {
        self.channels
            .get(&channel)
            .map(|c| c.space)
            .ok_or(ExecError::UnknownChannel { channel })
    }

    /// The memory of `space`.
    fn memory(&self, space: AsId) -> Result<Arc<dyn GpuMemory>, ExecError> {
        self.spaces
            .get(&space)
            .map(|s| Arc::clone(&s.memory))
            .ok_or(ExecError::UnknownSpace { space })
    }

    /// A `Bind` (module docs, "What the executor owns").
    fn bind(
        &mut self,
        channel: ChannelId,
        subchannel: Subchannel,
        class: ClassId,
    ) -> Result<(), ExecError> {
        self.flush_deferred(Some(channel))?;
        let state = self
            .channels
            .get_mut(&channel)
            .ok_or(ExecError::UnknownChannel { channel })?;
        let bound = (class.0 != 0).then_some(class);
        if let Some(slot) = state.bound.get_mut(usize::from(subchannel.get())) {
            *slot = bound;
        }
        if let Some(class) = bound
            && !state.engines.contains_key(&class)
            && let Some(engine) = Engine::new(
                channel,
                class,
                &self.backend,
                self.config.setting,
                self.fences.newest_submitted(),
            )
        {
            state.engines.insert(class, engine);
        }
        Ok(())
    }

    /// Hands the updates the fence path applied to the caches: each drops its range.
    fn after_fences(&mut self) {
        for (space, update) in self.fences.take_applied() {
            if let Some(state) = self.spaces.get_mut(&space) {
                state.buffers.apply_update(&update);
                state.textures.apply_update(&update);
            }
            self.shaders.invalidate(update.range());
        }
    }

    /// The CPU-write drain of `space` (module docs, "Before each draw, dispatch and copy").
    fn drain(&mut self, space: AsId) -> Result<(), ExecError> {
        let state = self
            .spaces
            .get_mut(&space)
            .ok_or(ExecError::UnknownSpace { space })?;
        catch_up_space(state, self.fences.newest_submitted());
        let mut drained = Drained(std::mem::take(&mut self.drained));
        state.buffers.sync_cpu_writes(&mut drained);
        if !drained.0.is_empty() {
            state.textures.cpu_writes(&drained.0);
            for range in &drained.0 {
                self.shaders.invalidate(*range);
            }
        }
        drained.0.clear();
        self.drained = drained.0;
        Ok(())
    }

    /// Draws the deferred instances of `channel`'s 3D engine, or of every channel's for `None`
    /// (module docs, "Deferred instanced draws").
    ///
    /// Every channel's deferred instances are drawn even when one fails; the first failure is
    /// returned. Allocates only when some channel has deferred instances.
    fn flush_deferred(&mut self, channel: Option<ChannelId>) -> Result<(), ExecError> {
        let Some(channel) = channel else {
            let deferred: Vec<ChannelId> = self
                .channels
                .keys()
                .copied()
                .filter(|&c| self.deferred_space(c).is_some())
                .collect();
            let mut first_error = None;
            for channel in deferred {
                if let Err(error) = self.flush_deferred(Some(channel))
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
            return first_error.map_or(Ok(()), Err);
        };
        let Some(space) = self.deferred_space(channel) else {
            return Ok(());
        };
        self.before_work(space, true)?;
        self.draw_step(channel, DrawStep::Flush)
    }

    /// The address space of `channel` when its 3D engine has deferred instances.
    fn deferred_space(&self, channel: ChannelId) -> Option<AsId> {
        let state = self.channels.get(&channel)?;
        let engine = state.engines.get(&ClassId(clc797::CLASS))?;
        match &engine.role {
            Role::ThreeD { draws, .. } if draws.has_deferred() => Some(state.space),
            _ => None,
        }
    }

    /// The words of a `Write` or `Run` (module docs, "Routing").
    fn words(
        &mut self,
        channel: ChannelId,
        class: ClassId,
        method: u16,
        words: &[u32],
    ) -> Result<(), ExecError> {
        let state = self
            .channels
            .get(&channel)
            .ok_or(ExecError::UnknownChannel { channel })?;
        if !state.is_bound(class) {
            return Err(ExecError::Unbound { channel, class });
        }
        if !state.engines.contains_key(&class) {
            let count = u64::try_from(words.len()).unwrap_or(u64::MAX);
            self.counters.unhandled_words = self.counters.unhandled_words.saturating_add(count);
            return Ok(());
        }
        if class.0 != clc797::CLASS {
            self.flush_deferred(Some(channel))?;
        }
        let method = u32::from(method);
        if class.0 == clc797::CLASS && method == clc797::LOAD_ROOT_TABLE.method() {
            return self.root_table(channel, words);
        }
        for &word in words {
            self.word(channel, class, method, word)?;
        }
        Ok(())
    }

    /// A run of `LOAD_ROOT_TABLE` words: stored in the mirror, then handed to the draw path whole.
    fn root_table(&mut self, channel: ChannelId, words: &[u32]) -> Result<(), ExecError> {
        let engine = self.engine_mut(channel, ClassId(clc797::CLASS))?;
        engine.mirror.run(clc797::LOAD_ROOT_TABLE.method(), words)?;
        if let Role::ThreeD { draws, .. } = &mut engine.role {
            draws.load_root_table(&engine.mirror, words)?;
        }
        Ok(())
    }

    fn engine_mut(&mut self, channel: ChannelId, class: ClassId) -> Result<&mut Engine, ExecError> {
        self.channels
            .get_mut(&channel)
            .and_then(|c| c.engines.get_mut(&class))
            .ok_or(ExecError::Unbound { channel, class })
    }

    /// One word to one method of one engine.
    fn word(
        &mut self,
        channel: ChannelId,
        class: ClassId,
        method: u32,
        value: u32,
    ) -> Result<(), ExecError> {
        let space = self.space_of(channel)?;
        self.engine_mut(channel, class)?
            .mirror
            .write(method, value)?;
        match class.0 {
            clc797::CLASS => {
                if method == clc797::SET_RENDER_ENABLE_C.method() {
                    self.conditional_catch_up(channel, space);
                }
                if starts_draw(method) {
                    self.before_work(space, true)?;
                }
                self.inline_word(channel, class, space, method)?;
                self.draw_step(channel, DrawStep::Method { method, value })
            }
            clc7c0::CLASS => {
                self.inline_word(channel, class, space, method)?;
                if method == LAST_INLINE_QMD_METHOD
                    || method == clc7c0::SEND_SIGNALING_PCAS2_B.method()
                {
                    self.before_work(space, true)?;
                    self.compute_word(channel, space, method)?;
                }
                Ok(())
            }
            clc7b5::CLASS => {
                if method == clc7b5::LAUNCH_DMA.method() {
                    self.before_work(space, false)?;
                }
                self.copy_word(channel, class, space, method)
            }
            cl902d::CLASS => {
                if method == cl902d::PIXELS_FROM_MEMORY_SRC_Y0_INT.method() {
                    self.before_work(space, false)?;
                }
                self.copy_word(channel, class, space, method)
            }
            cla140::CLASS => self.inline_word(channel, class, space, method),
            _ => Ok(()),
        }
    }

    /// The CPU-write drain, and, when `flush_pending`, the merged inline writes, before a draw,
    /// dispatch or copy.
    fn before_work(&mut self, space: AsId, flush_pending: bool) -> Result<(), ExecError> {
        self.drain(space)?;
        if flush_pending {
            self.with_sink(space, None, |_, _, pending, sink| {
                pending.flush(sink);
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Runs `work` with a [`CacheSink`] over `space`'s caches, its memory, its pending inline
    /// writes and, when `engine` names one, that engine; then attaches the releases the sink
    /// collected, in order, as fenced releases of `space`. The releases are attached whether or
    /// not `work` failed (an engine releases only once its work is done, so those it gave are
    /// owed). `work`'s error comes first, then an attach's.
    fn with_sink(
        &mut self,
        space: AsId,
        engine: Option<(ChannelId, ClassId)>,
        work: impl FnOnce(
            Option<&mut Engine>,
            &dyn GpuMemory,
            &mut PendingWrites,
            &mut CacheSink<'_>,
        ) -> Result<(), ExecError>,
    ) -> Result<(), ExecError> {
        let Self {
            channels,
            spaces,
            fences,
            shaders,
            readbacks,
            backend,
            config,
            counters,
            scratch,
            ..
        } = self;
        let engine = match engine {
            Some((channel, class)) => Some(
                channels
                    .get_mut(&channel)
                    .and_then(|c| c.engines.get_mut(&class))
                    .ok_or(ExecError::Unbound { channel, class })?,
            ),
            None => None,
        };
        let state = spaces
            .get_mut(&space)
            .ok_or(ExecError::UnknownSpace { space })?;
        catch_up_space(state, fences.newest_submitted());
        let memory = Arc::clone(&state.memory);
        let mut pending = std::mem::take(&mut state.pending);
        let mut sink = CacheSink {
            buffers: &mut state.buffers,
            textures: &mut state.textures,
            memory: &*memory,
            shaders: &mut **shaders,
            fences,
            readbacks,
            backend: &**backend,
            catch_up: config.catch_up,
            counters,
            scratch,
            releases: Vec::new(),
        };
        let result = work(engine, &*memory, &mut pending, &mut sink);
        let releases = std::mem::take(&mut sink.releases);
        state.pending = pending;
        let mut attached = Ok(());
        for release in releases {
            let action = FencedAction::Release {
                memory: Arc::clone(&memory),
                space,
                release,
            };
            if let Err(error) = fences.attach(action)
                && attached.is_ok()
            {
                attached = Err(ExecError::Fence(error));
            }
        }
        result.and(attached)
    }

    /// One word to an engine's inline upload, if it has one.
    fn inline_word(
        &mut self,
        channel: ChannelId,
        class: ClassId,
        space: AsId,
        method: u32,
    ) -> Result<(), ExecError> {
        let inline_class = match class.0 {
            clc797::CLASS => &INLINE_C797,
            clc7c0::CLASS => &INLINE_C7C0,
            cla140::CLASS => &INLINE_A140,
            _ => return Ok(()),
        };
        let launch = method == inline_class.launch.method();
        if !launch && method != inline_class.data.method() {
            return Ok(());
        }
        if launch {
            self.drain(space)?;
        }
        self.with_sink(
            space,
            Some((channel, class)),
            |engine, memory, pending, sink| {
                let Some(engine) = engine else {
                    return Ok(());
                };
                let inline = match &mut engine.role {
                    Role::ThreeD { inline, .. }
                    | Role::Compute { inline }
                    | Role::Inline(inline) => inline,
                    Role::Dma | Role::TwoD => return Ok(()),
                };
                inline.method(
                    inline_class,
                    method,
                    engine.mirror.registers(),
                    memory,
                    pending,
                    sink,
                )?;
                Ok(())
            },
        )
    }

    /// One word to a DMA or 2D engine.
    fn copy_word(
        &mut self,
        channel: ChannelId,
        class: ClassId,
        space: AsId,
        method: u32,
    ) -> Result<(), ExecError> {
        self.with_sink(
            space,
            Some((channel, class)),
            |engine, memory, pending, sink| {
                let Some(engine) = engine else {
                    return Ok(());
                };
                let registers = engine.mirror.registers();
                match engine.role {
                    Role::Dma => dma_method(method, registers, memory, pending, sink)?,
                    Role::TwoD => blit_method(method, registers, memory, pending, sink)?,
                    _ => {}
                }
                Ok(())
            },
        )
    }

    /// One word of the 3D class to its draw path, or the deferred instances (module docs).
    fn draw_step(&mut self, channel: ChannelId, step: DrawStep) -> Result<(), ExecError> {
        let space = self.space_of(channel)?;
        let class = ClassId(clc797::CLASS);
        let drawer_changed = self.last_drawer != Some((channel, class));
        let Self {
            channels,
            spaces,
            fences,
            shaders,
            pipelines,
            scratch,
            last_drawer,
            ..
        } = self;
        let engine = channels
            .get_mut(&channel)
            .and_then(|c| c.engines.get_mut(&class))
            .ok_or(ExecError::Unbound { channel, class })?;
        let Role::ThreeD { draws, .. } = &mut engine.role else {
            return Ok(());
        };
        let newest = fences.newest_submitted();
        if engine.notified < newest || drawer_changed {
            // A new list, or another engine drew last: its render targets are gone.
            draws.submitted(newest);
            engine.notified = newest;
        }
        let state = spaces
            .get_mut(&space)
            .ok_or(ExecError::UnknownSpace { space })?;
        catch_up_space(state, fences.newest_submitted());
        let memory = Arc::clone(&state.memory);
        let mut ctx = DrawContext {
            memory: &*memory,
            buffers: &mut state.buffers,
            textures: &mut state.textures,
            pipelines,
            shaders: &mut **shaders,
            list: scratch,
        };
        let outcome = match step {
            DrawStep::Method { method, value } => {
                draws.method(&mut ctx, &mut engine.mirror, method, value)
            }
            DrawStep::Flush => draws.flush(&mut ctx, &mut engine.mirror),
        };
        if matches!(outcome, Ok(Some(Outcome::Drawn))) {
            *last_drawer = Some((channel, class));
        }
        // Cache uploads made before a failure stay in the list (`back::draw`, "Errors and what
        // they leave").
        let recorded = fences.record_list(scratch);
        outcome?;
        recorded?;
        Ok(())
    }

    /// A word the QMD scheduler acts on (`compute`).
    fn compute_word(
        &mut self,
        channel: ChannelId,
        space: AsId,
        method: u32,
    ) -> Result<(), ExecError> {
        let class = ClassId(clc7c0::CLASS);
        let Self {
            channels,
            spaces,
            fences,
            shaders,
            pipelines,
            qmds,
            compute_shaders,
            backend,
            config,
            counters,
            scratch,
            last_drawer,
            ..
        } = self;
        let engine = channels
            .get_mut(&channel)
            .and_then(|c| c.engines.get_mut(&class))
            .ok_or(ExecError::Unbound { channel, class })?;
        let state = spaces
            .get_mut(&space)
            .ok_or(ExecError::UnknownSpace { space })?;
        catch_up_space(state, fences.newest_submitted());
        let memory = Arc::clone(&state.memory);
        let registers = engine.mirror.registers();
        let inline = (method == LAST_INLINE_QMD_METHOD)
            .then(|| inline_qmd_range(registers))
            .flatten();
        let dispatched = counters.dispatches;
        let mut sink = ComputeSink {
            space,
            memory: &memory,
            buffers: &mut state.buffers,
            textures: &mut state.textures,
            notified: &mut state.notified,
            shaders: &mut **shaders,
            host: compute_shaders,
            pipelines,
            fences,
            backend: &**backend,
            setting: config.setting,
            counters,
            scratch,
            inline,
            error: None,
        };
        let result = qmds.compute_method(method, registers, &*memory, &mut sink);
        // The inline QMD was written even when nothing ran.
        if result.is_ok() {
            sink.written();
        }
        let fence_error = sink.error.take();
        if sink.counters.dispatches != dispatched {
            // A dispatch between draws: the next draw sets its targets again.
            *last_drawer = None;
        }
        result?;
        fence_error.map_or(Ok(()), |e| Err(ExecError::Fence(e)))
    }

    /// A present (module docs, "Presents").
    fn present(&mut self, request: &PresentRequest) -> Result<(), ExecError> {
        let space = self.present_space.ok_or(ExecError::NoPresentSpace)?;
        let info = present::image(&request.image)?;
        let Self {
            spaces,
            fences,
            scratch,
            last_drawer,
            frame,
            counters,
            ..
        } = self;
        let state = spaces
            .get_mut(&space)
            .ok_or(ExecError::UnknownSpace { space })?;
        catch_up_space(state, fences.newest_submitted());
        let view = state
            .textures
            .texture(&*state.memory, &present::header(info), scratch);
        let view = match view {
            Ok(view) => view,
            Err(error) => {
                // Uploads made before the failure are the cache's own consistent work.
                fences.record_list(scratch)?;
                return Err(ExecError::Present(error));
            }
        };
        scratch.push(starquake_render::Command::Present(PresentDesc {
            view: view.view,
            src_rect: request.shown,
            transform: PresentTransform::Identity,
            frame: *frame,
            dump: false,
        }));
        *frame = frame.wrapping_add(1);
        *last_drawer = None;
        counters.presents = counters.presents.saturating_add(1);
        fences.record_list(scratch)?;
        fences.close(CloseReason::Present)?;
        Ok(())
    }

    /// The conditional-render catch-up (module docs, "Conditional rendering").
    fn conditional_catch_up(&mut self, channel: ChannelId, space: AsId) {
        let Some(range) = self
            .channels
            .get(&channel)
            .and_then(|c| c.engines.get(&ClassId(clc797::CLASS)))
            .and_then(|engine| conditional::report_range(&engine.mirror))
        else {
            return;
        };
        match self.fences.settle(space, range, self.config.catch_up) {
            Ok(Settled::Nothing) => {}
            Ok(Settled::Performed(_)) => {
                self.counters.catch_ups = self.counters.catch_ups.saturating_add(1);
            }
            Err(error) => {
                self.counters.catch_up_failures = self.counters.catch_up_failures.saturating_add(1);
                tracing::warn!(%error, "a conditional-render catch-up failed; the report is read as it is");
            }
        }
    }
}

/// What [`Executor::draw_step`] does.
#[derive(Clone, Copy, Debug)]
enum DrawStep {
    /// One word to a 3D method.
    Method { method: u32, value: u32 },
    /// The deferred instances.
    Flush,
}

/// Whether a word to 3D method `method` may start a draw (module docs).
fn starts_draw(method: u32) -> bool {
    DRAW_METHODS
        .iter()
        .chain(UNHANDLED_METHODS.iter())
        .any(|m| m.method() == method)
}

/// Tells `space`'s caches of `newest`, the newest fence submitted, if they have not been told
/// (module docs, "Lists and the caches").
fn catch_up_space(space: &mut Space, newest: FenceValue) {
    catch_up(
        &mut space.buffers,
        &mut space.textures,
        &mut space.notified,
        newest,
    );
}

/// Tells `buffers` and `textures` of `newest` if `notified` is older, and remembers it: a cache
/// is told each fence once, before its first use after the list closed.
fn catch_up(
    buffers: &mut BufferCache,
    textures: &mut TextureCache,
    notified: &mut FenceValue,
    newest: FenceValue,
) {
    if *notified < newest {
        buffers.submitted(newest);
        textures.submitted(newest);
        *notified = newest;
    }
}

/// The bytes the inline QMD is written to: `SET_INLINE_QMD_ADDRESS_A/B` shifted by
/// [`QMD_ADDRESS_SHIFT`], [`QMD_BYTES`] long, as [`QmdTable::compute_method`] writes them;
/// `None` when the address does not fit in 64 bits (the scheduler refuses it) or the range would
/// pass the end of the address space.
fn inline_qmd_range(registers: &[u32]) -> Option<Range<u64>> {
    let register = |method: u32| registers.get(usize::try_from(method).ok()?).copied();
    let upper = register(clc7c0::SET_INLINE_QMD_ADDRESS_A.method())?;
    let lower = register(clc7c0::SET_INLINE_QMD_ADDRESS_B.method())?;
    let shifted = (u64::from(upper) << u32::BITS) | u64::from(lower);
    if shifted.leading_zeros() < QMD_ADDRESS_SHIFT {
        return None;
    }
    let start = shifted << QMD_ADDRESS_SHIFT;
    Some(start..start.checked_add(u64::try_from(QMD_BYTES).ok()?)?)
}
