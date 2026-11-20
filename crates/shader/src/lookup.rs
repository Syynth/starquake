//! What the GPU crate's back thread uses to ask for shaders (`docs/specs/shader-pipeline.md`,
//! "What the GPU front/back protocol and engine state (#12) gets"; port slice 9).
//!
//! The shader cache that takes these, [`crate::cache::ShaderCache`] with its `lookup`, `wait`,
//! `invalidate` and `tracked_ranges`, is port slice 19; its lookups answer
//! [`crate::cache::CacheLookup`], shaped after G18's seam (owner's ruling 11 of 2026-11-19).
//! This module defines the types GPU slice G18 compiles against, with the names of the spec's
//! sketch: [`GuestStage`],
//! [`ProgramRequest`], [`ProgramMemory`], the answering side [`EngineState`] (defined with the
//! exchange in [`crate::exchange`] and re-exported here), the variant key [`VariantKey`]
//! ([`crate::variant`]), and the result of a lookup, [`Lookup`]: [`Lookup::Ready`] with a
//! [`ShaderSet`], [`Lookup::Pending`] with a [`Ticket`], or [`Lookup::Rejected`] with a
//! [`Rejection`]. G18's fake cache returns these.

use std::sync::Arc;

use thiserror::Error;

pub use crate::exchange::EngineState;
use crate::host::HostShader;
pub use crate::program::Rejection;
pub use crate::translate::GuestStage;
pub use crate::variant::VariantKey;

/// What the back thread asks for, per stage slot or per dispatch: the stage and the program's
/// GPU address (equal to its CPU address on Switch 2, row 30).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgramRequest {
    /// The stage the program runs as.
    pub stage: GuestStage,
    /// The program's address: the header's first byte for a graphics program, the first
    /// instruction's for a compute program (spec, "Where programs come from").
    pub address: u64,
}

/// A read of guest memory that reached an address with nothing mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("guest address {address:#x} is not mapped")]
pub struct Unmapped {
    /// The first address of the read that is not mapped.
    pub address: u64,
}

/// Read access to guest memory at GPU addresses (equal to CPU addresses on Switch 2).
/// Implemented by the GPU crate over its address table (slice G18).
pub trait ProgramMemory {
    /// Fills `out` with the bytes at `address` onwards, or reports the first address that is
    /// not mapped.
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped>;
}

/// The host shaders of one request, one per host stage, linked (spec, "What the render backend
/// (#14) gets, through the GPU crate").
#[derive(Clone, Debug)]
pub struct ShaderSet {
    /// The shaders, in pipeline order.
    pub stages: Vec<HostShader>,
}

/// A translation in progress, which a later lookup finds ready or rejected (spec,
/// "Compilation, threads and stutter").
///
/// Slice 19's cache is the only producer in production, but [`Ticket::new`] is public, so any
/// caller can mint one; G18's fake cache does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ticket(u64);

impl Ticket {
    /// The ticket numbered `id`.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The ticket's number.
    pub const fn id(self) -> u64 {
        self.0
    }
}

/// What a lookup gives: never blocks and never translates (spec, `ShaderCache::lookup`).
#[derive(Clone, Debug)]
pub enum Lookup {
    /// The shaders are ready.
    Ready(Arc<ShaderSet>),
    /// The shaders are being translated; the GPU crate's wait-while-compiling setting decides
    /// what the draw does (owner's decision 6).
    Pending(Ticket),
    /// A program was rejected, with the reason.
    Rejected(Arc<Rejection>),
}

#[cfg(test)]
mod tests;
