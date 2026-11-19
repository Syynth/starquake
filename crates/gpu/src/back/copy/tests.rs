//! Tests of the copy engines: a fake GPU memory over flat guest memory, a recording sink, and
//! the layout crate's tile and untile as the oracle for block-linear sides.

mod dma;
mod inline;
mod surface;
mod twod;

use core::ops::Range;

use starquake_layout::{Extent4D, GobType, LinearLayout, Offset4D, Tiling};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange};

use super::{CopySink, PendingWrites};
use crate::classes::Method;
use crate::field::Field;
use crate::memory::{GpuMemory, TestGpuMemory};
use crate::ring::Release;

/// The CPU side of the test memory.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages of guest memory behind the GPU mapping.
const PAGES: u64 = 512;
/// The GPU address the guest memory is mapped at: above 32 bits, so every upper address
/// register is exercised, and with bits above bit 39 set, so an 8-bit upper field would lose
/// them.
const GPU_BASE: u64 = 0x1_2345_0000_0000;
/// A second GPU address of the same guest memory, low enough for the classes whose upper
/// address fields hold 8 bits (`902D`, and `C797`'s inline uploads).
const LOW_BASE: u64 = 0x45_0000_0000;

type Memory = TestGpuMemory<FlatMemory>;

/// A memory with `PAGES` pages mapped at `GPU_BASE`, and again at `LOW_BASE`, filled with a
/// pattern whose every byte differs from its neighbours.
fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    mem.map(PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(), CPU_BASE)
        .unwrap();
    mem.map(PageRange::new(LOW_BASE, PAGES * PAGE_SIZE).unwrap(), CPU_BASE)
        .unwrap();
    let pattern: Vec<u8> = (0..PAGES * PAGE_SIZE)
        .map(|o| (o.wrapping_mul(7).wrapping_add(o / 251) & 0xff) as u8)
        .collect();
    mem.guest().write(CPU_BASE, &pattern).unwrap();
    mem
}

/// A memory whose GPU mapping is `PAGES` pages at `GPU_BASE` except page `hole`, which is
/// unmapped.
fn memory_with_hole(hole: u64) -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    if hole > 0 {
        mem.map(PageRange::new(GPU_BASE, hole * PAGE_SIZE).unwrap(), CPU_BASE)
            .unwrap();
    }
    let after = (hole + 1) * PAGE_SIZE;
    mem.map(
        PageRange::new(GPU_BASE + after, PAGES * PAGE_SIZE - after).unwrap(),
        CPU_BASE + after,
    )
    .unwrap();
    mem
}

/// The bytes at `va..va + len`.
fn read(mem: &Memory, va: u64, len: u64) -> Vec<u8> {
    let mut out = vec![0; usize::try_from(len).unwrap()];
    mem.read(va, &mut out).unwrap();
    out
}

/// Every byte of the mapping, for comparing whole states.
fn snapshot(mem: &Memory) -> Vec<u8> {
    read(mem, GPU_BASE, PAGES * PAGE_SIZE)
}

/// What a sink saw.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    MakeCurrent(Range<u64>),
    Invalidate(Range<u64>),
    Release(Release),
}

/// A sink that records every call in order.
#[derive(Default)]
struct Recorder {
    events: Vec<Event>,
}

impl CopySink for Recorder {
    fn make_current(&mut self, range: Range<u64>) {
        self.events.push(Event::MakeCurrent(range));
    }
    fn invalidate(&mut self, range: Range<u64>) {
        self.events.push(Event::Invalidate(range));
    }
    fn release(&mut self, release: Release) {
        self.events.push(Event::Release(release));
    }
}

impl Recorder {
    fn releases(&self) -> Vec<Release> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Release(r) => Some(*r),
                _ => None,
            })
            .collect()
    }
}

/// A register file being built.
#[derive(Clone, Debug)]
struct Regs(Vec<u32>);

impl Regs {
    fn new(len: usize) -> Self {
        Self(vec![0; len])
    }

    fn set(&mut self, method: Method, value: u32) -> &mut Self {
        self.0[method.method() as usize] = value;
        self
    }

    /// Inserts `value` into `field` of `method`, keeping its other fields.
    fn field(&mut self, method: Method, field: Field, value: u32) -> &mut Self {
        let index = method.method() as usize;
        self.0[index] = field.insert(self.0[index], value).unwrap();
        self
    }
}

fn lo(x: u64) -> u32 {
    x as u32
}

fn hi(x: u64) -> u32 {
    (x >> 32) as u32
}

/// A block-linear surface as the oracle sees it: the colour GOB, one GOB wide, `2^bh` GOBs tall
/// and `2^bd` deep, `width` bytes by `height` rows by `depth` slices.
#[derive(Clone, Copy, Debug)]
struct Bl {
    width: u32,
    height: u32,
    depth: u32,
    bh: u8,
    bd: u8,
}

impl Bl {
    fn tiling(self) -> Tiling {
        Tiling::new(GobType::TuringColor2d, 0, self.bh, self.bd).unwrap()
    }

    fn extent(self) -> Extent4D<starquake_layout::Bytes> {
        Extent4D::new(self.width, self.height, self.depth, 1)
    }

    /// Bytes of the surface padded to whole blocks.
    fn size(self) -> u64 {
        let tile = self.tiling().extent_b();
        u64::from(self.width.next_multiple_of(tile.width))
            * u64::from(self.height.next_multiple_of(tile.height))
            * u64::from(self.depth.next_multiple_of(tile.depth))
    }

    /// The layout crate's tiling of `linear` (rows of `row` bytes) into `tiled` at `origin`.
    fn tile(self, tiled: &mut [u8], linear: &[u8], row: usize, origin: (u32, u32), w: u32, h: u32) {
        starquake_layout::linear_to_tiled(
            tiled,
            self.extent(),
            linear,
            LinearLayout {
                row_stride_b: row,
                plane_stride_b: row * h as usize,
            },
            Offset4D::new(origin.0, origin.1, 0, 0),
            Extent4D::new(w, h, 1, 1),
            &self.tiling(),
        )
        .unwrap();
    }

    /// The layout crate's untiling of a `w` by `h` region at `origin` of `tiled`.
    fn untile(self, tiled: &[u8], origin: (u32, u32), w: u32, h: u32) -> Vec<u8> {
        let mut linear = vec![0; (w * h) as usize];
        starquake_layout::tiled_to_linear(
            &mut linear,
            LinearLayout {
                row_stride_b: w as usize,
                plane_stride_b: (w * h) as usize,
            },
            tiled,
            self.extent(),
            Offset4D::new(origin.0, origin.1, 0, 0),
            Extent4D::new(w, h, 1, 1),
            &self.tiling(),
        )
        .unwrap();
        linear
    }
}

/// The pending list's ranges after `adds`.
fn pending_after(adds: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut pending = PendingWrites::new();
    for range in adds {
        pending.add(range.clone());
    }
    pending.ranges()
}
