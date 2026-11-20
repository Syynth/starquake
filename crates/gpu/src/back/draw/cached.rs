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
//! Nothing builds it in a run yet: `core` keeps its `NoShaders` until slice 17's translator
//! implements [`SeamShader`] for its output and `core` hands the cache its compile pool.

use std::collections::HashMap;
use std::sync::Arc;

use starquake_memory::PageRange;
use starquake_shader::cache::{CacheLookup, ReadyShader, ShaderCache, Translated, Translator};
use starquake_shader::host::HostShader;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, ShaderSet, Ticket};
use starquake_shader::variant::VariantKey;

use super::shaders::{ReadySet, ResourceUse, SetLookup, ShaderLookup, StageInfo};
use crate::back::buffer::InvalidationSink;
use crate::back::pipeline::ShaderStamp;

/// What the seam needs of a translated shader beside what the cache needs.
pub trait SeamShader: Translated {
    /// The host shader.
    fn host(&self) -> &HostShader;
    /// The resources it binds, in the seam's terms.
    fn resources(&self) -> Vec<ResourceUse>;
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
