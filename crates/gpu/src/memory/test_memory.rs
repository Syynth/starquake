//! [`TestGpuMemory`]: a [`GpuMemory`] whose address table maps GPU page ranges onto guest memory,
//! for tests (gpu-protocol slice G7, "test implementation").
//!
//! Compiled for this crate's own tests and for crates that enable the `test-support` feature
//! (`docs/specs/testing.md`, "Shared test support").

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock, RwLockReadGuard};

use starquake_memory::{GuestMemory, MapError, PAGE_SIZE, PageRange};

use super::{GpuFault, GpuMemory, Translate, page_pieces};

/// A mapping's end (exclusive) and the CPU address its first byte leads to.
#[derive(Debug, Clone, Copy)]
struct Mapping {
    end: u64,
    cpu_va: u64,
}

/// A GPU address space for tests: page ranges mapped onto a [`GuestMemory`] at CPU addresses,
/// with a version that every [`Self::map`] and [`Self::unmap`] moves.
///
/// It implements [`GpuMemory`] by translating each page through its table (no cache), and
/// [`Translate`] (a page's target is the CPU address of its first byte), so a
/// [`super::TranslationCache`] can run over it.
///
/// A [`GpuMemory::write`] whose pages do not all translate changes nothing. One whose guest
/// memory faults part-way keeps the pieces written before the fault.
#[derive(Debug)]
pub struct TestGpuMemory<M> {
    guest: M,
    /// Mappings by GPU start address; they never overlap.
    table: RwLock<BTreeMap<u64, Mapping>>,
    generation: AtomicU64,
}

impl<M: GuestMemory> TestGpuMemory<M> {
    /// An empty address space over `guest`, at version 0.
    pub fn new(guest: M) -> Self {
        Self {
            guest,
            table: RwLock::new(BTreeMap::new()),
            generation: AtomicU64::new(0),
        }
    }

    /// The guest memory behind the mappings.
    pub fn guest(&self) -> &M {
        &self.guest
    }

    fn table(&self) -> RwLockReadGuard<'_, BTreeMap<u64, Mapping>> {
        self.table.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Maps `gpu` onto the guest memory starting at CPU address `cpu_va`, and moves the version.
    ///
    /// `cpu_va` must be page aligned ([`MapError::MisalignedAddress`]) and `cpu_va + gpu.size()`
    /// representable ([`MapError::VirtualRange`]); no page of `gpu` may be mapped already
    /// ([`MapError::AlreadyMapped`], naming the lowest such page).
    pub fn map(&self, gpu: PageRange, cpu_va: u64) -> Result<(), MapError> {
        PageRange::new(cpu_va, gpu.size())?;
        let mut table = self.table.write().unwrap_or_else(PoisonError::into_inner);
        let below = table
            .range(..gpu.start())
            .next_back()
            .filter(|(_, m)| m.end > gpu.start())
            .map(|_| gpu.start());
        let inside = table.range(gpu.start()..gpu.end()).next().map(|(&s, _)| s);
        if let Some(va) = below.or(inside) {
            return Err(MapError::AlreadyMapped { va });
        }
        table.insert(
            gpu.start(),
            Mapping {
                end: gpu.end(),
                cpu_va,
            },
        );
        drop(table);
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Removes the mapping that is exactly `gpu`, and moves the version.
    ///
    /// Anything else is [`MapError::NotMapped`], naming `gpu`'s start.
    pub fn unmap(&self, gpu: PageRange) -> Result<(), MapError> {
        let mut table = self.table.write().unwrap_or_else(PoisonError::into_inner);
        match table.get(&gpu.start()) {
            Some(m) if m.end == gpu.end() => {
                table.remove(&gpu.start());
            }
            _ => return Err(MapError::NotMapped { va: gpu.start() }),
        }
        drop(table);
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

/// The CPU address the GPU page starting at `page` leads to, under `table`.
fn translate(table: &BTreeMap<u64, Mapping>, page: u64) -> Option<u64> {
    let (&start, mapping) = table.range(..=page).next_back()?;
    (page < mapping.end).then(|| mapping.cpu_va + (page - start))
}

/// The CPU address GPU address `va` leads to, under `table`.
fn translate_va(table: &BTreeMap<u64, Mapping>, va: u64) -> Result<u64, GpuFault> {
    let offset = va % PAGE_SIZE;
    translate(table, va - offset)
        .map(|page| page + offset)
        .ok_or(GpuFault::Unmapped { va })
}

impl<M: GuestMemory> GpuMemory for TestGpuMemory<M> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        let table = self.table();
        for (piece_va, range) in page_pieces(va, buf.len())? {
            let cpu = translate_va(&table, piece_va)?;
            self.guest
                .read(cpu, &mut buf[range])
                .map_err(|fault| GpuFault::Guest {
                    va: piece_va,
                    fault,
                })?;
        }
        Ok(())
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let table = self.table();
        // Every page translates before any byte is written.
        for (piece_va, _) in page_pieces(va, data.len())? {
            translate_va(&table, piece_va)?;
        }
        for (piece_va, range) in page_pieces(va, data.len())? {
            let cpu = translate_va(&table, piece_va)?;
            self.guest
                .write(cpu, &data[range])
                .map_err(|fault| GpuFault::Guest {
                    va: piece_va,
                    fault,
                })?;
        }
        Ok(())
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

impl<M: GuestMemory> Translate for TestGpuMemory<M> {
    type Target = u64;

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn translate_page(&self, page: u64) -> Option<u64> {
        translate(&self.table(), page)
    }
}
