//! [`TranslationCache`]: one GPU thread's recent page translations, keyed by the address table's
//! version.
//!
//! Ported from Eden-NX2's per-thread `TranslationCache`
//! (`eden-nx2/src/core/device_memory_manager.h` and `.inc` @ 50fa86bb, `Translations()` and
//! `ReadBlock`; row 125 of `eden-nx2/NX2_NOTES.md`). GPL-3.0-or-later, Copyright 2026 Eden
//! Emulator Project, on code Copyright 2023 yuzu Emulator Project (GPL-2.0-or-later).

use std::sync::Arc;

use super::{Translate, page_of};

/// How many page translations a cache keeps: Eden-NX2's `std::array<TranslationEntry, 4>`, whose
/// slots are replaced in turn (`cache.cursor = (cache.cursor + 1) & 3U`).
pub const TRANSLATION_CACHE_SLOTS: usize = 4;

/// One GPU thread's last [`TRANSLATION_CACHE_SLOTS`] page translations from one table.
///
/// [`Self::translate`] takes `&mut self`: a cache belongs to one thread, and each GPU thread
/// makes its own (see the module docs of [`crate::memory`] for why a shared one crashed). It is
/// emptied whenever the table's version differs from the one its entries were made under.
#[derive(Debug)]
pub struct TranslationCache<S: Translate> {
    source: Arc<S>,
    /// The table version the entries were made under.
    generation: u64,
    /// `(page, target)` pairs.
    slots: [Option<(u64, S::Target)>; TRANSLATION_CACHE_SLOTS],
    /// The slot the next miss fills.
    cursor: usize,
}

impl<S: Translate> TranslationCache<S> {
    /// An empty cache over `source`.
    pub fn new(source: Arc<S>) -> Self {
        let generation = source.generation();
        Self {
            source,
            generation,
            slots: [None; TRANSLATION_CACHE_SLOTS],
            cursor: 0,
        }
    }

    /// The table this cache translates through.
    pub fn source(&self) -> &Arc<S> {
        &self.source
    }

    /// Where the page holding `va` leads, or `None` if it is not mapped.
    ///
    /// Reads the table's version first and empties the cache if it moved; then answers from the
    /// cache, or asks the table and keeps the answer in the next slot in turn. A page that does
    /// not translate is not kept, so a later map of it is seen at once.
    pub fn translate(&mut self, va: u64) -> Option<S::Target> {
        // The version is read before any translation, so an entry is never tagged with a version
        // newer than the table it was read from.
        let current = self.source.generation();
        if current != self.generation {
            self.slots = [None; TRANSLATION_CACHE_SLOTS];
            self.cursor = 0;
            self.generation = current;
        }
        let page = page_of(va);
        if let Some(target) = self
            .slots
            .iter()
            .flatten()
            .find_map(|&(cached, target)| (cached == page).then_some(target))
        {
            return Some(target);
        }
        let target = self.source.translate_page(page)?;
        self.slots[self.cursor] = Some((page, target));
        self.cursor = (self.cursor + 1) % TRANSLATION_CACHE_SLOTS;
        Some(target)
    }
}
