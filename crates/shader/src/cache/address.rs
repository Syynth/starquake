//! Layer 1 of the shader cache, the address map: `(stage, program address)` to what capturing
//! the program there gave, and the guest bytes that answer depends on (spec, "Identity, caching
//! and invalidation", layer 1).
//!
//! Entries are ordered by address, so [`AddressMap::invalidate`] visits only the entries that
//! start within [`MAX_EXTENT`] bytes below the written range, or inside it: no entry reaches
//! further than that past its start. Reading guest memory for a capture goes through
//! [`GuestReader`], which records how far the capture read.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crate::capture::{ContentHash, MAX_INSTRUCTIONS, ProgramReader, Unreadable};
use crate::decode::Instruction;
use crate::lookup::{ProgramMemory, ProgramRequest, Rejection, Unmapped};
use crate::sph::HEADER_BYTES;
use crate::translate::GuestStage;

/// The most bytes a capture reads from its program's address: the graphics header and
/// [`MAX_INSTRUCTIONS`] instructions (spec, "Where programs come from"). An entry's range
/// never reaches further than this past its start.
pub const MAX_EXTENT: u64 = HEADER_BYTES as u64 + MAX_INSTRUCTIONS as u64 * Instruction::SIZE as u64;

/// What capturing the program at an address gave.
#[derive(Clone, Debug)]
pub(super) enum Captured {
    /// The program was captured; its content hash names its program entry (layer 2).
    Program(ContentHash),
    /// The program could not be captured: the negative entry, kept until the bytes it read are
    /// written.
    Refused(Arc<Rejection>),
}

/// One entry of the address map.
#[derive(Clone, Debug)]
pub(super) struct AddressEntry {
    /// The first address past every byte the capture read (or tried to read).
    pub(super) end: u64,
    /// What the capture gave.
    pub(super) captured: Captured,
}

/// The order of a stage in the map's key: the six graphics slots, then compute.
const fn stage_order(stage: GuestStage) -> u8 {
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

/// The map's key for `request`: address first, so a range of addresses is a range of keys.
const fn key(request: ProgramRequest) -> (u64, u8) {
    (request.address, stage_order(request.stage))
}

/// Layer 1 (module docs).
#[derive(Debug, Default)]
pub(super) struct AddressMap {
    entries: BTreeMap<(u64, u8), AddressEntry>,
}

impl AddressMap {
    /// The entry for `request`, if any.
    pub(super) fn get(&self, request: ProgramRequest) -> Option<&AddressEntry> {
        self.entries.get(&key(request))
    }

    /// Records what capturing `request` gave.
    pub(super) fn insert(&mut self, request: ProgramRequest, entry: AddressEntry) {
        self.entries.insert(key(request), entry);
    }

    /// Forgets `request`'s entry.
    pub(super) fn remove(&mut self, request: ProgramRequest) {
        self.entries.remove(&key(request));
    }

    /// How many entries the map holds.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Forgets every entry.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drops every entry whose range `[address, end)` shares a byte with `range`, and returns
    /// how many it dropped. An empty `range` drops nothing.
    pub(super) fn invalidate(&mut self, range: &Range<u64>) -> usize {
        if range.start >= range.end {
            return 0;
        }
        let lowest = range.start.saturating_sub(MAX_EXTENT);
        let hit: Vec<(u64, u8)> = self
            .entries
            .range((lowest, 0)..(range.end, 0))
            .filter(|(_, entry)| entry.end > range.start)
            .map(|(key, _)| *key)
            .collect();
        for key in &hit {
            self.entries.remove(key);
        }
        hit.len()
    }

    /// The guest ranges the entries depend on: sorted, with overlapping and adjacent ranges
    /// merged, so no two share or touch a byte.
    pub(super) fn tracked(&self) -> Vec<Range<u64>> {
        let mut ranges: Vec<Range<u64>> = Vec::new();
        for (&(start, _), entry) in &self.entries {
            match ranges.last_mut() {
                Some(last) if start <= last.end => last.end = last.end.max(entry.end),
                _ => ranges.push(start..entry.end),
            }
        }
        ranges
    }
}

/// Reads a program through [`ProgramMemory`] at offsets from its address (the capture's
/// [`ProgramReader`] seam), and records the first address past every byte it read or tried to
/// read.
pub(super) struct GuestReader<'a> {
    memory: &'a dyn ProgramMemory,
    address: u64,
    /// Offset of the first byte past every byte read or tried, from `address`.
    furthest: Cell<u64>,
}

impl<'a> GuestReader<'a> {
    /// A reader of the program at `address`.
    pub(super) fn new(memory: &'a dyn ProgramMemory, address: u64) -> Self {
        Self {
            memory,
            address,
            furthest: Cell::new(0),
        }
    }

    /// The first address past every byte read or tried, at most [`MAX_EXTENT`] past the
    /// program's address and at most `u64::MAX`.
    pub(super) fn end(&self) -> u64 {
        self.address
            .saturating_add(self.furthest.get().min(MAX_EXTENT))
    }

    fn reach(&self, offset: u64) {
        self.furthest.set(self.furthest.get().max(offset));
    }
}

impl ProgramReader for GuestReader<'_> {
    /// Reads `out.len()` bytes at `offset`. A read whose last byte would lie past `u64::MAX`
    /// is refused whole at `offset`, before memory is asked. An unmapped address the memory
    /// reports outside the read is taken as the read's first byte.
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), Unreadable> {
        let len = u64::try_from(out.len()).unwrap_or(u64::MAX);
        let refused = |at: u64| {
            self.reach(at.saturating_add(1));
            Unreadable { offset: at }
        };
        let Some(start) = self.address.checked_add(offset) else {
            return Err(refused(offset));
        };
        let Some(end_offset) = offset.checked_add(len) else {
            return Err(refused(offset));
        };
        if len > 0 && start.checked_add(len - 1).is_none() {
            return Err(refused(offset));
        }
        match self.memory.read(start, out) {
            Ok(()) => {
                self.reach(end_offset);
                Ok(())
            }
            Err(Unmapped { address }) => {
                let at = address
                    .checked_sub(self.address)
                    .filter(|at| (offset..end_offset).contains(at))
                    .unwrap_or(offset);
                Err(refused(at))
            }
        }
    }
}
