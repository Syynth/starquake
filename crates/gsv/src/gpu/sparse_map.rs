//! The entries of worker 216 `ExecuteSparseMapRequests`: their 40-byte layout
//! ([`SparseMapRequest`]) and their decoding ([`decode`]) into a GPU range, a memory handle and
//! a source offset ([`SparseMap`]) (`docs/specs/graphics-server.md`, the worker's table: "a batch
//! of up to 256 40-byte entries in a buffer (address, size, flags, kinds, memory handle, source
//! offset) → u64 processed", read, PR only; port slice 7).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/gpu_memory_registration.h` @ e5fb3b97
//! (`GpuSparseMapRequest`, `DecodeGpuSparseMap`) and
//! `eden-pr4588/src/core/hle/service/nvdrv/devices/native_sparse_remap.h` @ e5fb3b97
//! (`NativeSparseRemapEntry`, `DecodeNativeSparseRemap`), GPL-3.0-or-later, Copyright 2026 Eden
//! Emulator Project. NX2 does not serve 216.
//!
//! # The layout
//!
//! PR's `GpuSparseMapRequest` (`static_assert(sizeof(GpuSparseMapRequest) == 40)`), little
//! endian, at the offsets of [`offset`]:
//!
//! | Offset | Width | Field |
//! |---|---|---|
//! | 0 | 8 | GPU address |
//! | 8 | 8 | size in bytes |
//! | 16 | 1 | read-only |
//! | 17 | 1 | cache |
//! | 18 | 1 | reserved |
//! | 19 | 1 | wanted kind |
//! | 20 | 1 | fallback kind |
//! | 21 | 1 | compression tag mode |
//! | 22 | 2 | reserved |
//! | 24 | 4 | memory handle (101's), 0 for none |
//! | 28 | 4 | reserved |
//! | 32 | 8 | offset into the memory |
//!
//! # Decoding
//!
//! PR decodes an entry in two steps: `DecodeGpuSparseMap` checks the request's fields and builds
//! a `NativeSparseRemapEntry` in 4 KiB pages (flags `0x1000`, plus 4 when cached), which
//! `DecodeNativeSparseRemap` checks again and turns back into bytes. [`decode`] makes the checks
//! both steps make that an entry built by the first can fail, and refuses (`None`) an entry for
//! which any holds:
//!
//! 1. the size is zero (`!request.size`);
//! 2. the address, the size or the source offset is not 4 KiB aligned (`(address | size |
//!    source_offset) & 0xfff`);
//! 3. the cache byte is over 1 (`request.cache > 1`);
//! 4. the read-only byte, the compression tag mode or a reserved field is not zero;
//! 5. the range's end does not fit in 64 bits (`address > u64::MAX - size`);
//! 6. the source's end does not fit (`source > u64::MAX - size`);
//! 7. there is no memory handle but a source offset (`!entry.handle && source`).
//!
//! The rest of `DecodeNativeSparseRemap`'s checks cannot fail on what `DecodeGpuSparseMap`
//! builds: its flags are always one page-size bit (`0x1000`) and the cacheable bit, its reserved
//! word is 0, and its page counts are byte counts shifted down by 12, which never pass
//! `u64::MAX >> 12`. The kinds are not checked here: PR selects them only for an entry with a handle,
//! in `nvhost_as_gpu::NativeRemap` ([`super::memory`]'s `check_sparse_map`). Every refusal here
//! is one answer, `NotImplemented` (`R_UNLESS(entry, Kernel::ResultNotImplemented)`), so the
//! order of the checks above cannot be observed. Checks 1 and 2's address and size alignment are
//! made again by `PageRange::new`, which [`decode`] builds the range with, so dropping one of
//! them alone changes no answer.
//!
//! The cache byte is checked and then dropped: PR passes it on as the cacheable flag, which
//! nothing after the decoding reads but `DecodeNativeSparseRemap`'s flag check; the table keeps
//! no cache attribute (as for 211).

#[cfg(test)]
mod tests;

use starquake_memory::{PAGE_SIZE, PageRange};

/// The size of one entry: PR's `static_assert(sizeof(GpuSparseMapRequest) == 40)`. 216's request
/// names it (words 11 and 12), and PR refuses any other with `InvalidSize`.
pub const ENTRY_SIZE: usize = 40;

/// The most entries one 216 carries: PR's `count && count <= 256` (`InvalidSize` otherwise), the
/// same bound `nvhost_as_gpu::NativeRemap` checks (`entries.size() > 256`).
pub const MAX_ENTRIES: u64 = 256;

/// Where each field of an entry starts, in bytes (the module docs' table). The decoder and the
/// tests' encoder both read these.
pub mod offset {
    /// The GPU address (`u64`).
    pub const ADDRESS: usize = 0;
    /// The size in bytes (`u64`).
    pub const SIZE: usize = 8;
    /// The read-only byte.
    pub const READ_ONLY: usize = 16;
    /// The cache byte.
    pub const CACHE: usize = 17;
    /// The first reserved field (`u8`).
    pub const RESERVED0: usize = 18;
    /// The wanted kind byte.
    pub const KIND: usize = 19;
    /// The fallback kind byte.
    pub const FALLBACK: usize = 20;
    /// The compression tag mode byte.
    pub const COMPTAG_MODE: usize = 21;
    /// The second reserved field (`u16`).
    pub const RESERVED1: usize = 22;
    /// The memory handle (`u32`).
    pub const MEMORY: usize = 24;
    /// The third reserved field (`u32`).
    pub const RESERVED2: usize = 28;
    /// The offset into the memory (`u64`).
    pub const SOURCE_OFFSET: usize = 32;
}

/// The most a cache byte may be (`request.cache > 1` refuses).
pub const MAX_CACHE: u8 = 1;

/// One entry as laid out (PR `GpuSparseMapRequest`), before any check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMapRequest {
    /// The first GPU address.
    pub address: u64,
    /// The size in bytes.
    pub size: u64,
    /// The read-only byte.
    pub read_only: u8,
    /// The cache byte.
    pub cache: u8,
    /// The first reserved field.
    pub reserved0: u8,
    /// The wanted kind.
    pub kind: u8,
    /// The fallback kind.
    pub fallback: u8,
    /// The compression tag mode.
    pub comptag_mode: u8,
    /// The second reserved field.
    pub reserved1: u16,
    /// The memory handle 101 returned, or 0 for none.
    pub memory: u32,
    /// The third reserved field.
    pub reserved2: u32,
    /// Where the range starts in the memory.
    pub source_offset: u64,
}

/// The `N` bytes of `bytes` at `at`; zeros if they are not all there, which no [`offset`] does
/// (`every_field_is_read_from_prs_offsets` pins each).
fn field<const N: usize>(bytes: &[u8; ENTRY_SIZE], at: usize) -> [u8; N] {
    bytes
        .get(at..)
        .and_then(<[u8]>::first_chunk::<N>)
        .copied()
        .unwrap_or([0; N])
}

impl SparseMapRequest {
    /// Reads an entry from its 40 bytes (the module docs' layout).
    #[must_use]
    pub fn parse(bytes: &[u8; ENTRY_SIZE]) -> Self {
        let byte = |at: usize| field::<1>(bytes, at)[0];
        Self {
            address: u64::from_le_bytes(field(bytes, offset::ADDRESS)),
            size: u64::from_le_bytes(field(bytes, offset::SIZE)),
            read_only: byte(offset::READ_ONLY),
            cache: byte(offset::CACHE),
            reserved0: byte(offset::RESERVED0),
            kind: byte(offset::KIND),
            fallback: byte(offset::FALLBACK),
            comptag_mode: byte(offset::COMPTAG_MODE),
            reserved1: u16::from_le_bytes(field(bytes, offset::RESERVED1)),
            memory: u32::from_le_bytes(field(bytes, offset::MEMORY)),
            reserved2: u32::from_le_bytes(field(bytes, offset::RESERVED2)),
            source_offset: u64::from_le_bytes(field(bytes, offset::SOURCE_OFFSET)),
        }
    }
}

/// A decoded entry: what PR's `NativeSparseRange` holds, with the two kind bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMap {
    /// The GPU range.
    pub range: PageRange,
    /// The memory handle, or 0: the range goes back to its sparse region's GPU-only pages.
    pub handle: u32,
    /// Where the range starts in the memory (0 when there is no handle).
    pub source_offset: u64,
    /// The wanted kind.
    pub kind: u8,
    /// The fallback kind.
    pub fallback: u8,
}

/// PR's `DecodeGpuSparseMap` with the `DecodeNativeSparseRemap` it calls: the entry's range,
/// handle and source, or `None` for an entry either refuses (the module docs list each check).
#[must_use]
pub fn decode(request: &SparseMapRequest) -> Option<SparseMap> {
    let aligned = |value: u64| value.is_multiple_of(PAGE_SIZE);
    if request.size == 0
        || !aligned(request.address)
        || !aligned(request.size)
        || !aligned(request.source_offset)
        || request.cache > MAX_CACHE
        || request.read_only != 0
        || request.comptag_mode != 0
        || request.reserved0 != 0
        || request.reserved1 != 0
        || request.reserved2 != 0
    {
        return None;
    }
    // `address > max - size` and `source > max - size`: each end must fit in 64 bits.
    request.source_offset.checked_add(request.size)?;
    if request.memory == 0 && request.source_offset != 0 {
        return None;
    }
    let range = PageRange::new(request.address, request.size).ok()?;
    Some(SparseMap {
        range,
        handle: request.memory,
        source_offset: request.source_offset,
        kind: request.kind,
        fallback: request.fallback,
    })
}
