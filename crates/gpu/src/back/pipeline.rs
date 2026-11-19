//! The back thread's pipeline-key cache (gpu-protocol spec, "Caches" and "3D: from state to a
//! draw" step 3; render-backend, "Resources and who owns the caches" and "Compiling without
//! stalling a frame"; slice G23).
//!
//! [`PipelineCache`] maps the pipeline state of a draw or dispatch to a [`PipelineId`] made
//! through [`Backend::create_graphics_pipeline`] and [`Backend::create_compute_pipeline`]. The
//! key is this crate's and the host pipeline the backend's (decision log 2026-11-12, "Draw
//! commands cross to the backend as plain data; caches are split by what they are keyed on").
//! The caller, the draw path (slice G18), derives a [`GraphicsPipelineDesc`] from engine state
//! and hands it over with the [`StageStamps`] of its shaders; this module does not read
//! registers.
//!
//! # Keys
//!
//! A graphics pipeline is keyed on its [`Stages`] and on everything else in its description
//! that the device does not set per draw ([`starquake_render::Caps::dynamic_state`]), the
//! [`EmulationFlags`](starquake_render::EmulationFlags) included (review guide, M13); `key.rs`
//! has the details. A change to a dynamic field therefore makes no new pipeline, and the
//! pipeline is created with canonical values in those fields ([`CANONICAL_DEPTH_BIAS`],
//! [`CANONICAL_LINE_WIDTH`], [`CANONICAL_STENCIL_MASK`]), whichever draw made it. A compute
//! pipeline is keyed on its shader. The dynamic states are read from the backend's `Caps` once,
//! when the cache is made: `Caps` are fixed for a backend's lifetime.
//!
//! # Status and failures
//!
//! A lookup gives a [`Lookup`]:
//!
//! - a new key's pipeline is created and reported [`Lookup::Compiling`]: creation only queues
//!   the build (`create_*_pipeline` "queues compilation ... and returns at once"), and
//!   [`Backend::pipeline_status`] is not asked until the next lookup;
//! - a known key that was compiling is asked about again: it stays [`Lookup::Compiling`], or
//!   turns [`Lookup::Ready`] (and is never asked about again), or turns [`Lookup::Failed`];
//! - a pipeline the backend refused to create (a [`RenderError`]) or reported failed is cached
//!   as failed: it is not created or asked about again, and every later lookup reports
//!   [`Lookup::Failed`] with the same [`FailCause`] (rows 90 and 112: cache rejections
//!   negatively). [`Failure::newly`] is true only on the lookup that learned of the failure, so
//!   the caller counts each failed pipeline once (render-backend: "skipped and counted once per
//!   pipeline").
//!
//! What a draw does with a compiling pipeline is the one wait-while-compiling setting, which
//! this crate owns and the caller applies (decision log 2026-11-12, the first of the twenty-one
//! graphics decisions; review guide, M10); the cache does not wait.
//!
//! Nothing here destroys a pipeline: an entry lives as long as the cache. A pipeline that failed
//! to compile keeps its id reserved in the backend.
//!
//! # The key list
//!
//! The cache records the key of each pipeline the backend accepted for creation, once, in the
//! order it first did: shaders by stamp, so that the next boot can map them to its own shaders.
//! [`PipelineCache::key_list`] returns them as a [`KeyList`], which [`KeyList::to_bytes`] turns
//! into a versioned file's bytes ([`list`]); `core` writes the file. On the next boot `core`
//! reads it back with [`KeyList::from_bytes`] and hands it to [`PipelineCache::precompile`]
//! with a map from stamps to the shaders it has made for them, before the first draw, as Eden's
//! `LoadDiskResources` builds the pipelines of its file at start-up
//! (`eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` @ cf654052; policy only).
//! Every key of the loaded list is recorded again, built or not, while the record has room, so
//! the next file keeps it.
//!
//! # Limits
//!
//! The cache holds at most [`MAX_PIPELINES`] entries (graphics and compute together, failed ones
//! included) and records at most [`MAX_PIPELINES`] keys. A lookup of a new key in a full cache is
//! [`PipelineCacheError::Full`] and creates nothing; a key that finds the record full is not
//! recorded and counted ([`PipelineCache::unrecorded`]). Neither reference caps its pipeline
//! cache: the cap bounds what a hostile title can make the host hold, and its value is a
//! **guess** (Eden-NX2's notes give no pipeline counts), stricter than both references.
//!
//! # Divergences from Eden
//!
//! - **Recorded on acceptance, not on a successful build.** Eden serialises a key once its
//!   pipeline is built (`CreateGraphicsPipeline()` and `CreateComputePipeline` in
//!   `vk_pipeline_cache.cpp` queue `SerializePipeline` only when the pipeline is non-null). Here
//!   the build finishes later, on the compile pool, so a key is recorded when the backend
//!   accepts it, and one whose build then fails stays in the list and is tried again on the next
//!   boot. Pinned by `tests::a_pipeline_that_fails_to_compile_stays_in_the_key_list`; a pipeline
//!   the backend refuses outright is not recorded
//!   (`tests::a_refused_pipeline_is_not_rebuilt_or_recorded`).
//! - **Shader identity.** Eden's keys carry a 64-bit hash per stage (`unique_hashes` in
//!   `GraphicsPipelineCacheKey`, `vk_graphics_pipeline.h`) and its file carries each shader's
//!   environment too. Here a key carries a [`ShaderStamp`] per stage, and the shaders themselves
//!   are the shader cache's to persist (shader-pipeline slice 21).
//! - **A key from another dynamic-state set is skipped, as Eden skips it** (`load_graphics` in
//!   `LoadDiskResources` returns when a key's dynamic-feature bits differ from the device's),
//!   and also kept in the list. Pinned by
//!   `tests::precompile_reports_every_outcome`.

mod key;
pub mod list;

#[cfg(test)]
mod tests;

pub use key::{
    CANONICAL_DEPTH_BIAS, CANONICAL_LINE_WIDTH, CANONICAL_STENCIL_MASK, KEYED_DYNAMIC_STATE,
    KeyFault, KeyField, ShaderStamp, StageStamps,
};
pub use list::{KeyList, KeyListError};

use key::{FixedState, GraphicsKey, PipelineKey, mismatched_stage, resolve};
use starquake_render::caps::DynamicState;
use starquake_render::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc, Stages};
use starquake_render::{Backend, PipelineId, PipelineStatus, RenderError, ShaderId, ShaderStage};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Most entries the cache holds, and most keys it records and a key list may have (module docs,
/// "Limits"; the value is a guess).
pub const MAX_PIPELINES: usize = 1 << 16;

/// Why a pipeline could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailCause {
    /// The backend refused to create it.
    Refused(RenderError),
    /// The backend accepted it and then reported that its build failed.
    Compile,
}

/// A pipeline that cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Whether this lookup is the one that learned of the failure (module docs, "Status and
    /// failures").
    pub newly: bool,
    /// Why it failed.
    pub cause: FailCause,
}

/// What a lookup found (module docs, "Status and failures").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The pipeline is being built; what the draw does is the caller's wait-while-compiling
    /// setting.
    Compiling(PipelineId),
    /// The pipeline is ready.
    Ready(PipelineId),
    /// The pipeline cannot be built; its draws are skipped.
    Failed(Failure),
}

/// Why a lookup was refused. Nothing changed: no pipeline was created and nothing recorded.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PipelineCacheError {
    /// A stage has a shader but no stamp, or a stamp but no shader.
    #[error("the {} stage has a shader without a stamp or a stamp without a shader", .stage.name())]
    StampMismatch {
        /// The first such stage, in the order tessellation control, tessellation evaluation,
        /// geometry, fragment.
        stage: ShaderStage,
    },
    /// The cache holds [`MAX_PIPELINES`] entries and the key is new.
    #[error("the pipeline cache is full ({limit} pipelines)")]
    Full {
        /// How many entries it holds.
        limit: usize,
    },
}

/// What [`PipelineCache::precompile`] did with each key of a list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Precompiled {
    /// Keys whose pipeline was created now.
    pub queued: usize,
    /// Keys whose pipeline the cache already held.
    pub known: usize,
    /// Keys whose pipeline the backend refused to create.
    pub refused: usize,
    /// Graphics keys built under another set of dynamic states than this backend's, skipped.
    pub other_dynamic_state: usize,
    /// Keys with a stamp the caller had no shader for, skipped.
    pub missing_shader: usize,
    /// Keys skipped because the cache was full.
    pub full: usize,
}

/// A new key met a full cache.
struct Full;

/// What [`PipelineCache::find_or_create`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Made {
    /// The entry was there already.
    Known,
    /// The backend accepted the new pipeline.
    Created,
    /// The backend refused the new pipeline; the entry is stored as failed.
    Refused,
}

/// One entry's state.
#[derive(Clone, Debug)]
enum Entry {
    Compiling(PipelineId),
    Ready(PipelineId),
    Failed(FailCause),
}

/// What the cache tells pipelines apart by: shaders by id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum EntryKey {
    Graphics(Stages, Box<FixedState>),
    Compute(ShaderId),
}

/// The back thread's pipeline-key cache (module docs).
pub struct PipelineCache {
    backend: Arc<dyn Backend>,
    /// The backend's dynamic states, within [`KEYED_DYNAMIC_STATE`].
    dynamic: DynamicState,
    /// Most entries, and most recorded keys.
    limit: usize,
    entries: HashMap<EntryKey, Entry>,
    /// Each recorded key and its place in the record.
    recorded: HashMap<PipelineKey, usize>,
    unrecorded: u64,
}

impl PipelineCache {
    /// An empty cache making pipelines through `backend`, keyed on what its `Caps` do not set
    /// per draw.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self::with_limit(backend, MAX_PIPELINES)
    }

    fn with_limit(backend: Arc<dyn Backend>, limit: usize) -> Self {
        let dynamic = backend.caps().dynamic_state;
        Self {
            dynamic: FixedState::keyed_dynamic(dynamic),
            backend,
            limit,
            entries: HashMap::new(),
            recorded: HashMap::new(),
            unrecorded: 0,
        }
    }

    /// How many entries the cache holds, failed ones included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many keys were not recorded because the record was full.
    pub fn unrecorded(&self) -> u64 {
        self.unrecorded
    }

    /// The pipeline for `desc`, whose shaders have the stamps `stamps` (module docs).
    ///
    /// Errors: [`PipelineCacheError::StampMismatch`] when a stage's shader and stamp are not
    /// both present or both absent, then [`PipelineCacheError::Full`] for a new key in a full
    /// cache; either leaves the cache and the backend as they were.
    pub fn lookup_graphics(
        &mut self,
        desc: &GraphicsPipelineDesc,
        stamps: &StageStamps,
    ) -> Result<Lookup, PipelineCacheError> {
        if let Some(stage) = mismatched_stage(&desc.stages, stamps) {
            return Err(PipelineCacheError::StampMismatch { stage });
        }
        let fixed = FixedState::of(desc, self.dynamic);
        let create =
            |backend: &dyn Backend| backend.create_graphics_pipeline(fixed.desc(desc.stages));
        let (lookup, made) = self
            .find_or_create(
                EntryKey::Graphics(desc.stages, Box::new(fixed.clone())),
                create,
            )
            .map_err(|Full| self.full())?;
        if made == Made::Created {
            self.record(&PipelineKey::Graphics(Box::new(GraphicsKey {
                stamps: *stamps,
                fixed,
            })));
        }
        Ok(lookup)
    }

    /// The pipeline for `desc`, whose shader has the stamp `stamp` (module docs).
    ///
    /// Errors: [`PipelineCacheError::Full`] for a new key in a full cache, which leaves the
    /// cache and the backend as they were.
    pub fn lookup_compute(
        &mut self,
        desc: ComputePipelineDesc,
        stamp: ShaderStamp,
    ) -> Result<Lookup, PipelineCacheError> {
        let (lookup, made) = self
            .find_or_create(EntryKey::Compute(desc.shader), |backend| {
                backend.create_compute_pipeline(desc)
            })
            .map_err(|Full| self.full())?;
        if made == Made::Created {
            self.record(&PipelineKey::Compute(stamp));
        }
        Ok(lookup)
    }

    fn full(&self) -> PipelineCacheError {
        PipelineCacheError::Full { limit: self.limit }
    }

    /// The keys recorded so far, in the order they were first recorded (module docs, "The key
    /// list").
    pub fn key_list(&self) -> KeyList {
        let mut keys: Vec<(&PipelineKey, usize)> =
            self.recorded.iter().map(|(key, &at)| (key, at)).collect();
        keys.sort_unstable_by_key(|&(_, at)| at);
        KeyList {
            keys: keys.into_iter().map(|(key, _)| key.clone()).collect(),
        }
    }

    /// Creates the pipeline of each key of `list`, in order, with the shader `shader` maps each
    /// stamp to, and records every key (module docs, "The key list"). A graphics key built
    /// under other dynamic states than this backend's is skipped, as is a key with a stamp
    /// `shader` maps to nothing; a key whose pipeline the cache holds already is left as it is.
    /// Returns what happened to each key, counted.
    pub fn precompile(
        &mut self,
        list: &KeyList,
        shader: &mut dyn FnMut(ShaderStamp) -> Option<ShaderId>,
    ) -> Precompiled {
        let mut done = Precompiled::default();
        for key in &list.keys {
            self.record(key);
            let outcome = match key {
                PipelineKey::Graphics(graphics) => {
                    let fixed = &graphics.fixed;
                    if fixed.dynamic != self.dynamic {
                        done.other_dynamic_state += 1;
                        continue;
                    }
                    let Some(stages) = resolve(&graphics.stamps, shader) else {
                        done.missing_shader += 1;
                        continue;
                    };
                    self.find_or_create(
                        EntryKey::Graphics(stages, Box::new(fixed.clone())),
                        |backend| backend.create_graphics_pipeline(fixed.desc(stages)),
                    )
                }
                PipelineKey::Compute(stamp) => {
                    let Some(id) = shader(*stamp) else {
                        done.missing_shader += 1;
                        continue;
                    };
                    let desc = ComputePipelineDesc { shader: id };
                    self.find_or_create(EntryKey::Compute(id), |backend| {
                        backend.create_compute_pipeline(desc)
                    })
                }
            };
            let counter = match outcome {
                Err(Full) => &mut done.full,
                Ok((_, Made::Known)) => &mut done.known,
                Ok((_, Made::Refused)) => &mut done.refused,
                Ok((_, Made::Created)) => &mut done.queued,
            };
            *counter += 1;
        }
        done
    }

    /// The entry for `entry_key`, created through `create` when it is new. Returns the lookup
    /// and what this call did. The one check that can refuse (a full cache) comes before the
    /// backend is called; once it has been, the entry is stored and the call cannot fail.
    fn find_or_create(
        &mut self,
        entry_key: EntryKey,
        create: impl FnOnce(&dyn Backend) -> Result<PipelineId, RenderError>,
    ) -> Result<(Lookup, Made), Full> {
        if let Some(entry) = self.entries.get_mut(&entry_key) {
            return Ok((poll(&*self.backend, entry), Made::Known));
        }
        if self.entries.len() >= self.limit {
            return Err(Full);
        }
        match create(&*self.backend) {
            Ok(id) => {
                self.entries.insert(entry_key, Entry::Compiling(id));
                Ok((Lookup::Compiling(id), Made::Created))
            }
            Err(error) => {
                let cause = FailCause::Refused(error);
                self.entries.insert(entry_key, Entry::Failed(cause.clone()));
                Ok((
                    Lookup::Failed(Failure { newly: true, cause }),
                    Made::Refused,
                ))
            }
        }
    }

    /// Records `key` unless it is recorded already; counts it when the record is full.
    fn record(&mut self, key: &PipelineKey) {
        if self.recorded.contains_key(key) {
            return;
        }
        if self.recorded.len() >= self.limit {
            self.unrecorded = self.unrecorded.saturating_add(1);
            return;
        }
        let at = self.recorded.len();
        self.recorded.insert(key.clone(), at);
    }
}

/// A known entry's lookup: asks the backend only while the pipeline is compiling.
fn poll(backend: &dyn Backend, entry: &mut Entry) -> Lookup {
    match entry {
        Entry::Ready(id) => Lookup::Ready(*id),
        Entry::Failed(cause) => Lookup::Failed(Failure {
            newly: false,
            cause: cause.clone(),
        }),
        Entry::Compiling(id) => {
            let id = *id;
            match backend.pipeline_status(id) {
                PipelineStatus::Compiling => Lookup::Compiling(id),
                PipelineStatus::Ready => {
                    *entry = Entry::Ready(id);
                    Lookup::Ready(id)
                }
                PipelineStatus::Failed => {
                    *entry = Entry::Failed(FailCause::Compile);
                    Lookup::Failed(Failure {
                        newly: true,
                        cause: FailCause::Compile,
                    })
                }
            }
        }
    }
}
