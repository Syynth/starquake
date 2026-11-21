//! The shader cache behind G18's seam (shader-pipeline slice 19, "G18's seam" in
//! `starquake_shader::cache`; owner's ruling 11 of 2026-11-19: "Slice 19 adapts to G18's
//! `ShaderLookup`/`SetLookup` seam").
//!
//! [`CachedShaders`] puts a [`starquake_shader::cache::ShaderCache`] behind [`ShaderLookup`]
//! and [`InvalidationSink`], the two halves of the executor's shader cache
//! ([`crate::back::exec::ShaderCache`]). It is the one place the cache's answer becomes the
//! seam's:
//!
//! - **A ready set** becomes a [`ReadySet`]: the host shaders cloned into one [`ShaderSet`],
//!   each with a [`StageInfo`] whose stamp is the cache's stamp value as the pipeline-key
//!   cache's [`ShaderStamp`] (the one conversion between the two types) and whose resources
//!   the translated shader gives ([`SeamShader::resources`]). The conversion is kept per set,
//!   keyed by its stamps in order, so a draw that finds a set again clones no module; past
//!   [`MAX_READY_SETS`] kept sets, all are dropped and converted again on their next use.
//! - **A rejection** becomes [`SetLookup::Rejected`] with the cache's reason. The seam's arm
//!   carries the reason only, not the program the cache names
//!   ([`starquake_shader::cache::ProgramRejection`]), so the draw path still logs and counts a
//!   rejection once per set; carrying the named rejection through the seam, so it can count
//!   once per program (ruling 10), is a change to the draw path left to G18.
//! - **A ticket** and [`ShaderLookup::wait`] pass through: the cache's `wait` blocks the back
//!   thread until the set is settled.
//! - **An invalidation** of a page range becomes the cache's byte range.
//!
//! **The production translator's shaders.** [`SeamShader`] is implemented for the shader
//! crate's [`HostShader`], what its production translator
//! (`starquake_shader::pipeline::PipelineTranslator`, issue #484) gives, so `core` can build
//! the cache with it on the run's compile pool. Its resources are the host shader's resource
//! table ([`HostShader::resources`], slice 17): one seam row per table row, in the table's
//! order, at the group and binding the lowering wrote into the module:
//!
//! | Host table row | Seam row ([`ResourceUse`]) |
//! |---|---|
//! | `ResourceSource::ConstantBuffer { bank }`, `ResourceKind::UniformBuffer { size }` | [`ResourceSource::ConstantBuffer`]` { bank, size }`, the render crate's `ResourceKind::UniformBuffer`, `count: None` |
//! | `ResourceSource::DriverConstants`, `ResourceKind::UniformBuffer { size }` | [`ResourceSource::DriverConstants`], `ResourceKind::UniformBuffer`, `count: None` (the draw binds the driver constants' own size) |
//!
//! The table holds only those two kinds so far; the textures, samplers and storage buffers join
//! it with the rest of slice 17, and the match in [`resource_use`] is exhaustive, so a new kind
//! does not compile until it is converted here. Pinned by
//! `tests::a_host_shader_s_resource_table_reaches_the_seam` (through the production translator:
//! a constant-buffer load's bank row, and the driver constants' row a flipped viewport adds)
//! and `tests::each_row_field_reaches_the_seam` (each field, with distinct values).

use std::collections::HashMap;
use std::sync::Arc;

use starquake_memory::PageRange;
use starquake_render::shader::ResourceKind as RenderKind;
use starquake_shader::cache::{CacheLookup, ReadyShader, ShaderCache, Translated, Translator};
use starquake_shader::host::{
    HostShader, ResourceKind as HostKind, ResourceSlot, ResourceSource as HostSource,
};
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, ShaderSet, Ticket};
use starquake_shader::variant::VariantKey;

use super::shaders::{
    ReadySet, ResourceSource, ResourceUse, SetLookup, ShaderLookup, Slot, StageInfo,
};
use crate::back::buffer::InvalidationSink;
use crate::back::pipeline::ShaderStamp;

/// What the seam needs of a translated shader beside what the cache needs.
pub trait SeamShader: Translated {
    /// The host shader.
    fn host(&self) -> &HostShader;
    /// The resources it binds, in the seam's terms.
    fn resources(&self) -> Vec<ResourceUse>;
}

/// The production translator's shader (module docs, "The production translator's shaders").
impl SeamShader for HostShader {
    fn host(&self) -> &HostShader {
        self
    }

    /// The resource table, one seam row per table row (module docs).
    fn resources(&self) -> Vec<ResourceUse> {
        self.resources.iter().map(resource_use).collect()
    }
}

/// The seam's row for one row of a host shader's resource table (module docs, "The production
/// translator's shaders").
fn resource_use(row: &ResourceSlot) -> ResourceUse {
    let HostKind::UniformBuffer { size } = row.kind;
    ResourceUse {
        slot: Slot {
            group: row.group,
            binding: row.binding,
        },
        kind: RenderKind::UniformBuffer,
        count: None,
        source: match row.source {
            HostSource::DriverConstants => ResourceSource::DriverConstants,
            HostSource::ConstantBuffer { bank } => ResourceSource::ConstantBuffer { bank, size },
        },
    }
}

/// The most converted sets [`CachedShaders`] keeps (a **guess**: a frame rarely draws with more
/// distinct sets).
pub const MAX_READY_SETS: usize = 4_096;

/// The shader cache as the executor takes it (module docs).
pub struct CachedShaders<T: Translator>
where
    T::Shader: SeamShader,
{
    cache: ShaderCache<T>,
    sets: HashMap<Vec<ShaderStamp>, ReadySet>,
    limit: usize,
}

impl<T: Translator> core::fmt::Debug for CachedShaders<T>
where
    T::Shader: SeamShader,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CachedShaders")
            .field("cache", &self.cache)
            .field("sets", &self.sets.len())
            .finish_non_exhaustive()
    }
}

impl<T: Translator> CachedShaders<T>
where
    T::Shader: SeamShader,
{
    /// `cache`, behind the seam.
    pub fn new(cache: ShaderCache<T>) -> Self {
        Self::with_limit(cache, MAX_READY_SETS)
    }

    fn with_limit(cache: ShaderCache<T>, limit: usize) -> Self {
        Self {
            cache,
            sets: HashMap::new(),
            limit,
        }
    }

    /// The cache.
    pub const fn cache(&self) -> &ShaderCache<T> {
        &self.cache
    }

    /// The seam's answer for the cache's.
    fn answer(&mut self, lookup: CacheLookup<T::Shader>) -> SetLookup {
        match lookup {
            CacheLookup::Ready(shaders) => SetLookup::Ready(self.ready_set(&shaders)),
            CacheLookup::Pending(ticket) => SetLookup::Pending(ticket),
            CacheLookup::Rejected(rejection) => SetLookup::Rejected(Arc::clone(&rejection.reason)),
        }
    }

    /// The seam's set for `shaders`, converted once and kept (module docs).
    fn ready_set(&mut self, shaders: &[ReadyShader<T::Shader>]) -> ReadySet {
        let stamps: Vec<ShaderStamp> = shaders
            .iter()
            .map(|shader| ShaderStamp::new(shader.stamp.value()))
            .collect();
        if let Some(set) = self.sets.get(&stamps) {
            return set.clone();
        }
        let set = ReadySet {
            shaders: Arc::new(ShaderSet {
                stages: shaders
                    .iter()
                    .map(|shader| shader.shader.host().clone())
                    .collect(),
            }),
            stages: shaders
                .iter()
                .zip(&stamps)
                .map(|(shader, &stamp)| StageInfo {
                    stamp,
                    resources: shader.shader.resources(),
                })
                .collect(),
        };
        if self.sets.len() >= self.limit {
            self.sets.clear();
        }
        self.sets.insert(stamps, set.clone());
        set
    }
}

impl<T: Translator> ShaderLookup for CachedShaders<T>
where
    T::Shader: SeamShader,
{
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> SetLookup {
        let lookup = self.cache.lookup(requests, variant, memory, state);
        self.answer(lookup)
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        let lookup = self.cache.wait(ticket);
        self.answer(lookup)
    }
}

impl<T: Translator> InvalidationSink for CachedShaders<T>
where
    T::Shader: SeamShader,
{
    fn invalidate(&mut self, range: PageRange) {
        self.cache.invalidate(range.start()..range.end());
    }
}

#[cfg(test)]
mod tests;
