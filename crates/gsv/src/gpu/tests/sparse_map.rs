//! Worker 216 `ExecuteSparseMapRequests`, driven through raw TIPC messages and on the bare
//! bookkeeping.
//!
//! Requests are laid out by hand from PR's handler
//! (`eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp` @ e5fb3b97,
//! `ExecuteSparseMapRequests`: `packet[0] == 0x011000e8 && packet[1] == 7`, then `ReadPair(packet,
//! 8)`, `packet[10]`, `ReadPair(packet, 11)`, `packet[13]`, `packet[14]`), and entries from its
//! `GpuSparseMapRequest` (`gpu_memory_registration.h`), at the literal offsets that struct gives.

use super::super::memory::SparseCompletion;
use super::super::sparse_map::{ENTRY_SIZE, MAX_ENTRIES, SparseMapRequest};
use super::*;
use crate::completion::testing::{Locks, process};

/// Where the entries are: below 2^32, so a descriptor's third word holds no address bits.
const IN_AT: u32 = 0x8000_0000;
/// Where the out buffer is.
const OUT_AT: u32 = 0x9000_0000;
/// An address nothing is mapped at.
const UNMAPPED_AT: u32 = 0xA000_0000;
/// The bytes mapped at [`OUT_AT`].
const OUT_ROOM: usize = 0x1000;
/// The tests' sparse region: eight pages at [`REGION`], region 1.
const REGION_SIZE: u64 = 8 * PAGE_SIZE;
/// The tests' fallback byte for an entry with no handle, which PR sends back as it came.
const UNTOUCHED: u8 = 0x77;

/// One entry's fields, laid out by [`Entry::bytes`].
#[derive(Clone, Copy)]
struct Entry {
    address: u64,
    size: u64,
    read_only: u8,
    cache: u8,
    kind: u8,
    fallback: u8,
    memory: u32,
    source: u64,
}

impl Entry {
    /// `pages` pages of memory `memory` from `source` at `address`, cached, kinds 6 and 6.
    fn mapped(address: u64, pages: u64, memory: u32, source: u64) -> Self {
        Self {
            address,
            size: pages * PAGE_SIZE,
            read_only: 0,
            cache: 1,
            kind: 6,
            fallback: 6,
            memory,
            source,
        }
    }

    /// `pages` pages at `address` back to the region's own pages, fallback [`UNTOUCHED`].
    fn back(address: u64, pages: u64) -> Self {
        Self {
            memory: 0,
            source: 0,
            kind: 0,
            fallback: UNTOUCHED,
            ..Self::mapped(address, pages, 0, 0)
        }
    }

    /// PR's `GpuSparseMapRequest`: address 0–7, size 8–15, read-only 16, cache 17, reserved 18,
    /// kind 19, fallback 20, tag mode 21, reserved 22–23, memory 24–27, reserved 28–31, source
    /// 32–39; the reserved fields and the tag mode zero.
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; 40];
        bytes[0..8].copy_from_slice(&self.address.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.size.to_le_bytes());
        bytes[16] = self.read_only;
        bytes[17] = self.cache;
        bytes[19] = self.kind;
        bytes[20] = self.fallback;
        bytes[24..28].copy_from_slice(&self.memory.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.source.to_le_bytes());
        bytes
    }
}

/// 216's words: PR's exact header words (`0x011000e8`: type `0xe8`, one A and one B
/// descriptor; 7 data words), the A descriptor (`sent` bytes at [`IN_AT`]) and the B
/// descriptor (`room` bytes at `out`), then count 8–9, space 10, entry size 11–12, sync 13, slot
/// 14.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per request word, in the order the layout gives them"
)]
fn batch_words(
    sent: u32,
    room: u32,
    out: u32,
    count: u64,
    space: u32,
    entry_size: u64,
    sync: u32,
    slot: u32,
) -> Vec<u32> {
    assert_eq!(tipc_type(EXECUTE_SPARSE_MAP_REQUESTS), 0xE8);
    let mut words = vec![0x0110_00E8, 7, sent, IN_AT, 0, room, out, 0];
    words.extend(halves(count));
    words.push(space);
    words.extend(halves(entry_size));
    words.extend([sync, slot]);
    assert_eq!(words.len(), 15);
    words
}

/// The usual set-up ([`ready`]) with the entries' and the out buffer's memory mapped, and a
/// sparse region of [`REGION_SIZE`] at [`REGION`].
fn sparse_ready() -> Ready {
    let mut s = ready();
    let room = usize::try_from(MAX_ENTRIES).unwrap() * ENTRY_SIZE + 64;
    s.rig.kernel.map(u64::from(IN_AT), vec![0; room]);
    s.rig.kernel.map(u64::from(OUT_AT), vec![STALE; OUT_ROOM]);
    let words = sparse_words(
        s.space,
        REGION,
        REGION + REGION_SIZE - 1,
        12,
        0,
        s.device,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    s
}

impl Ready {
    /// Writes `entries` at [`IN_AT`] and fills the out buffer with [`STALE`]; their bytes.
    fn place(&mut self, entries: &[Entry]) -> Vec<u8> {
        let bytes: Vec<u8> = entries.iter().flat_map(Entry::bytes).collect();
        Kernel::write_memory(&mut self.rig.kernel, u64::from(IN_AT), &bytes).unwrap();
        Kernel::write_memory(&mut self.rig.kernel, u64::from(OUT_AT), &[STALE; OUT_ROOM]).unwrap();
        bytes
    }

    /// 216 of `entries` in the tests' space, slot 0, with an out buffer of one byte per entry.
    fn batch(&mut self, entries: &[Entry]) -> Vec<u8> {
        let count = entries.len();
        self.batch_with(entries, count)
    }

    /// As [`batch`](Self::batch), with `room` bytes of out buffer.
    fn batch_with(&mut self, entries: &[Entry], room: usize) -> Vec<u8> {
        let bytes = self.place(entries);
        let words = batch_words(
            u32::try_from(bytes.len()).unwrap(),
            u32::try_from(room).unwrap(),
            OUT_AT,
            u64::try_from(entries.len()).unwrap(),
            self.space,
            u64::try_from(ENTRY_SIZE).unwrap(),
            1,
            0,
        );
        self.rig.send_on(self.worker, &words)
    }

    /// The out buffer's first `len` bytes.
    fn out(&self, len: usize) -> Vec<u8> {
        self.rig
            .kernel
            .memory(u64::from(OUT_AT), len)
            .unwrap()
            .to_vec()
    }

    fn signals(&self) -> u32 {
        self.rig.kernel.signals(self.event).unwrap()
    }
}

/// What the table shows at a page of the tests' region `offset` bytes in.
fn region_page(offset: u64) -> Option<(Target, Kind)> {
    let target = Target::GpuOnly {
        region: RegionId(1),
        offset,
    };
    shown_at(
        &[Mapping {
            range: page_range(REGION + offset, PAGE_SIZE),
            target,
            kind: SPARSE_KIND,
        }],
        REGION + offset,
    )
}

// ----- What a batch does -----------------------------------------------------------------------

/// A batch of entries with handles maps at once: the table shows each, the reply is the count
/// processed, the event is signalled during the request, no update or barrier goes to the GPU,
/// and the out buffer gets one selected kind per entry (an entry with no handle's fallback byte
/// as sent), the room past them untouched.
#[test]
fn a_batch_maps_at_once_and_writes_the_selected_kinds() {
    let mut s = sparse_ready();
    let before = (s.signals(), s.rig.front.calls().len());
    let compressed = Entry {
        kind: 0xC,
        fallback: 3,
        ..Entry::mapped(REGION, 1, s.handle, PAGE_SIZE)
    };
    let generic = Entry {
        kind: 8,
        fallback: 6,
        ..Entry::mapped(REGION + 2 * PAGE_SIZE, 2, s.handle, 0)
    };
    let back = Entry::back(REGION + 5 * PAGE_SIZE, 1);
    let reply = s.batch_with(&[compressed, generic, back], 5);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), halves(3));
    assert_eq!(s.signals(), before.0 + 1);
    assert_eq!(s.rig.front.calls().len(), before.1);
    // PR: 0xc/3 selects 3 (backend S8Z24), 8/6 selects 6 (generic).
    assert_eq!(s.out(5), [3, 6, UNTOUCHED, STALE, STALE]);
    let space = s.space;
    assert_eq!(s.rig.at(space, REGION), page_at(MEMORY + PAGE_SIZE, S8Z24));
    assert_eq!(s.rig.at(space, REGION + PAGE_SIZE), region_page(PAGE_SIZE));
    assert_eq!(
        s.rig.at(space, REGION + 2 * PAGE_SIZE),
        page_at(MEMORY, GENERIC_16BX2)
    );
    assert_eq!(
        s.rig.at(space, REGION + 3 * PAGE_SIZE),
        page_at(MEMORY + PAGE_SIZE, GENERIC_16BX2)
    );
    assert_eq!(
        s.rig.at(space, REGION + 4 * PAGE_SIZE),
        region_page(4 * PAGE_SIZE)
    );
    assert_eq!(
        s.rig.at(space, REGION + 5 * PAGE_SIZE),
        region_page(5 * PAGE_SIZE)
    );
    assert_eq!(s.rig.gpu().gpu_memory().pending_jobs(), 0);
}

/// An entry with no handle over a mapping takes the part it covers out in order: an unmap and a
/// barrier go to the GPU, the table keeps the mapping until the GPU applies it, the pages are a
/// hole until the drain, and then the region's; the event fires at the drain. The parts of the
/// mapping outside the entry keep their place in the registration.
#[test]
fn an_entry_with_no_handle_completes_in_order() {
    let mut s = sparse_ready();
    let space = s.space;
    s.batch(&[Entry::mapped(REGION + PAGE_SIZE, 4, s.handle, 0)]);
    let before = s.signals();
    let middle = REGION + 2 * PAGE_SIZE;
    let reply = s.batch(&[Entry::back(middle, 1)]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), halves(1));
    assert_eq!(s.out(2), [UNTOUCHED, STALE]);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 2..],
        [
            FrontCall::Update(
                AsId(space),
                VmUpdate::Unmap {
                    range: page_range(middle, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    assert_eq!(s.signals(), before);
    assert_eq!(
        s.rig.at(space, middle),
        page_at(MEMORY + PAGE_SIZE, GENERIC_16BX2)
    );
    s.rig.apply();
    assert_eq!(s.rig.at(space, middle), None);
    let calls = s.rig.drain();
    assert_eq!(calls.last(), Some(&TimerCall::Signal(s.event)));
    assert_eq!(s.signals(), before + 1);
    assert_eq!(s.rig.at(space, middle), region_page(2 * PAGE_SIZE));
    assert_eq!(
        s.rig.at(space, REGION + PAGE_SIZE),
        page_at(MEMORY, GENERIC_16BX2)
    );
    assert_eq!(
        s.rig.at(space, REGION + 3 * PAGE_SIZE),
        page_at(MEMORY + 2 * PAGE_SIZE, GENERIC_16BX2)
    );
}

/// An entry with no handle over pages already the region's takes nothing out: the batch
/// completes at once.
#[test]
fn an_entry_over_bare_region_pages_completes_at_once() {
    let mut s = sparse_ready();
    let before = (s.signals(), s.rig.front.calls().len());
    let reply = s.batch(&[Entry::back(REGION, 8)]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(s.signals(), before.0 + 1);
    assert_eq!(s.rig.front.calls().len(), before.1);
    assert_eq!(s.rig.at(s.space, REGION), region_page(0));
}

/// Entries apply in order, each over what the ones before it left: a later map over an earlier
/// one wins; a removal of two pages then a map of the first replaces the first at once and
/// unmaps only the second, its net removal; a map then a removal of it leaves the region.
#[test]
fn later_entries_apply_over_earlier_ones() {
    let mut s = sparse_ready();
    let space = s.space;
    let over = Entry {
        kind: 0xB,
        fallback: 1,
        ..Entry::mapped(REGION + PAGE_SIZE, 1, s.handle, 3 * PAGE_SIZE)
    };
    let reply = s.batch(&[Entry::mapped(REGION, 2, s.handle, 0), over]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(s.out(2), [6, 1]);
    assert_eq!(s.rig.at(space, REGION), page_at(MEMORY, GENERIC_16BX2));
    assert_eq!(
        s.rig.at(space, REGION + PAGE_SIZE),
        page_at(MEMORY + 3 * PAGE_SIZE, Z16)
    );

    let reply = s.batch(&[
        Entry::back(REGION, 2),
        Entry::mapped(REGION, 1, s.handle, 2 * PAGE_SIZE),
    ]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 2..],
        [
            FrontCall::Update(
                AsId(space),
                VmUpdate::Unmap {
                    range: page_range(REGION + PAGE_SIZE, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    let replaced = page_at(MEMORY + 2 * PAGE_SIZE, GENERIC_16BX2);
    assert_eq!(s.rig.at(space, REGION), replaced);
    assert_eq!(
        s.rig.at(space, REGION + PAGE_SIZE),
        page_at(MEMORY + 3 * PAGE_SIZE, Z16)
    );
    s.rig.apply();
    assert_eq!(s.rig.at(space, REGION), replaced);
    assert_eq!(s.rig.at(space, REGION + PAGE_SIZE), None);
    s.rig.drain();
    assert_eq!(s.rig.at(space, REGION), replaced);
    assert_eq!(s.rig.at(space, REGION + PAGE_SIZE), region_page(PAGE_SIZE));

    let last = REGION + 7 * PAGE_SIZE;
    let reply = s.batch(&[Entry::mapped(last, 1, s.handle, 0), Entry::back(last, 1)]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    s.rig.finish();
    assert_eq!(s.rig.at(space, last), region_page(7 * PAGE_SIZE));
    assert!(
        !s.rig.gpu().gpu_memory().spaces[&AsId(space)]
            .records
            .contains_key(&last)
    );
}

/// A removal is a page a mapping covered before the batch and none covers after it, not a cut
/// within the batch: a map then a removal of the same bare region page changes nothing, so the
/// batch completes at once, sends the GPU nothing, signals during the request, and the region's
/// page shows throughout.
#[test]
fn a_map_then_back_over_a_bare_page_completes_at_once() {
    let mut s = sparse_ready();
    let space = s.space;
    let page = REGION + 3 * PAGE_SIZE;
    let before = (s.signals(), s.rig.front.calls().len());
    let reply = s.batch(&[Entry::mapped(page, 1, s.handle, 0), Entry::back(page, 1)]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(s.out(2), [6, UNTOUCHED]);
    assert_eq!(s.signals(), before.0 + 1);
    assert_eq!(s.rig.front.calls().len(), before.1);
    assert_eq!(s.rig.gpu().gpu_memory().pending_jobs(), 0);
    assert_eq!(s.rig.at(space, page), region_page(3 * PAGE_SIZE));
    s.rig.apply();
    assert_eq!(s.rig.at(space, page), region_page(3 * PAGE_SIZE));
    assert_eq!(s.rig.drain(), []);
    assert_eq!(s.signals(), before.0 + 1);
    assert_eq!(s.rig.at(space, page), region_page(3 * PAGE_SIZE));
}

/// A removal then a map of the same pages of a live mapping is a replacement: published at
/// once, as a 211 over a live mapping is, with no unmap and no job.
#[test]
fn a_back_then_map_over_a_mapping_replaces_it_at_once() {
    let mut s = sparse_ready();
    let space = s.space;
    s.batch(&[Entry::mapped(REGION, 1, s.handle, 0)]);
    assert_eq!(s.rig.at(space, REGION), page_at(MEMORY, GENERIC_16BX2));
    let before = (s.signals(), s.rig.front.calls().len());
    let reply = s.batch(&[
        Entry::back(REGION, 1),
        Entry::mapped(REGION, 1, s.handle, 2 * PAGE_SIZE),
    ]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(s.signals(), before.0 + 1);
    assert_eq!(s.rig.front.calls().len(), before.1);
    assert_eq!(s.rig.gpu().gpu_memory().pending_jobs(), 0);
    assert_eq!(
        s.rig.at(space, REGION),
        page_at(MEMORY + 2 * PAGE_SIZE, GENERIC_16BX2)
    );
}

/// Two entries with no handle whose ranges are apart, each over its own mapping, take both out:
/// an unmap for each, one barrier, and no mapping left after the drain.
#[test]
fn two_disjoint_removals_unmap_both() {
    let mut s = sparse_ready();
    let space = s.space;
    let far = REGION + 4 * PAGE_SIZE;
    s.batch(&[
        Entry::mapped(REGION, 1, s.handle, 0),
        Entry::mapped(far, 1, s.handle, 0),
    ]);
    let reply = s.batch(&[Entry::back(REGION, 1), Entry::back(far, 1)]);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 3..],
        [
            FrontCall::Update(
                AsId(space),
                VmUpdate::Unmap {
                    range: page_range(REGION, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Update(
                AsId(space),
                VmUpdate::Unmap {
                    range: page_range(far, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    s.rig.finish();
    assert!(
        s.rig.gpu().gpu_memory().spaces[&AsId(space)]
            .records
            .is_empty()
    );
    assert_eq!(s.rig.at(space, REGION), region_page(0));
    assert_eq!(s.rig.at(space, far), region_page(4 * PAGE_SIZE));
}

/// An entry with no handle sends its unmap only where the mapping shows: the shim inside it stays
/// in the table throughout.
#[test]
fn a_batch_keeps_the_shim() {
    let mut s = sparse_ready();
    let space = AsId(s.space);
    s.batch(&[Entry::mapped(REGION, 4, s.handle, 0)]);
    let shim = REGION + PAGE_SIZE;
    s.rig.gpu().gpu().map_syncpoint_shim(space, shim).unwrap();
    s.batch(&[Entry::back(REGION, 4)]);
    let calls = s.rig.front.calls();
    assert_eq!(
        calls[calls.len() - 3..],
        [
            FrontCall::Update(
                space,
                VmUpdate::Unmap {
                    range: page_range(REGION, PAGE_SIZE)
                },
                None
            ),
            FrontCall::Update(
                space,
                VmUpdate::Unmap {
                    range: page_range(REGION + 2 * PAGE_SIZE, 2 * PAGE_SIZE)
                },
                None
            ),
            FrontCall::Barrier(Ticket(1)),
        ]
    );
    s.rig.apply();
    assert_eq!(
        s.rig.at(s.space, shim),
        Some((Target::SyncpointShim, SELF_MAP_KIND))
    );
    s.rig.drain();
    assert_eq!(
        s.rig.at(s.space, shim),
        Some((Target::SyncpointShim, SELF_MAP_KIND))
    );
    assert_eq!(s.rig.at(s.space, REGION), region_page(0));
}

/// A batch's mapping is a 211 mapping: a 213 of exactly its range unmaps it, and a 102 of its
/// registration takes it out; the region shows again each time.
#[test]
fn a_batch_s_mapping_is_unmapped_and_destroyed_as_a_211_one() {
    let mut s = sparse_ready();
    let space = s.space;
    s.batch(&[Entry::mapped(REGION, 2, s.handle, 0)]);
    let words = range_words(
        NORMAL_UNMAP_GPU_MEMORY_4K,
        space,
        REGION,
        REGION + 2 * PAGE_SIZE - 1,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    s.rig.finish();
    assert_eq!(s.rig.at(space, REGION), region_page(0));
    s.batch(&[Entry::mapped(REGION + PAGE_SIZE, 1, s.handle, 0)]);
    s.rig.send_on(s.worker, &[0x76, 2, s.handle, 0]);
    s.rig.finish();
    assert_eq!(s.rig.at(space, REGION + PAGE_SIZE), region_page(PAGE_SIZE));
}

// ----- The checks ------------------------------------------------------------------------------

/// The request's own checks in PR's order, each refusal signalled and answered with the result
/// alone; a pair of faults answers the first; the count at its edges (256 accepted).
#[test]
fn header_checks_answer_in_prs_order() {
    let mut s = sparse_ready();
    let entry = Entry::mapped(REGION, 1, s.handle, 0);
    let size = u64::try_from(ENTRY_SIZE).unwrap();
    let one = u32::try_from(ENTRY_SIZE).unwrap();
    let space = s.space;
    let ask = |s: &mut Ready,
               sent: u32,
               room: u32,
               count: u64,
               space: u32,
               entry_size: u64,
               sync: u32| {
        s.place(&[entry]);
        let before = s.signals();
        let reply = s.rig.send_on(
            s.worker,
            &batch_words(sent, room, OUT_AT, count, space, entry_size, sync, 0),
        );
        assert_eq!(s.signals(), before + 1, "every answer is signalled");
        let (result, words) = reply_of(&reply);
        if result != ResultCode::SUCCESS {
            assert!(words.is_empty(), "a refusal answers the result alone");
        }
        result
    };
    let invalid_size = code(KernelError::InvalidSize);
    let not_implemented = code(KernelError::NotImplemented);
    let not_found = code(KernelError::NotFound);
    // The count, with a sync byte over 1 too.
    assert_eq!(ask(&mut s, one, 1, 0, space, size, 2), invalid_size);
    assert_eq!(
        ask(&mut s, one, 1, MAX_ENTRIES + 1, space, size, 2),
        invalid_size
    );
    assert_eq!(ask(&mut s, one, 1, 1 << 32, space, size, 2), invalid_size);
    // The entry size, at both sides, with a sync byte over 1 too.
    assert_eq!(ask(&mut s, one, 1, 1, space, size - 1, 2), invalid_size);
    assert_eq!(ask(&mut s, one, 1, 1, space, size + 1, 2), invalid_size);
    assert_eq!(
        ask(&mut s, one, 1, 1, space, size + (1 << 32), 2),
        invalid_size
    );
    // The sync byte, with no such address space too; only its low byte counts.
    assert_eq!(
        ask(&mut s, one, 1, 1, space + 9, size, u32::from(MAX_SYNC) + 1),
        not_implemented
    );
    assert_eq!(
        ask(&mut s, one, 1, 1, space, size, 0x100),
        ResultCode::SUCCESS
    );
    // The address space, with a short in buffer too.
    assert_eq!(ask(&mut s, one - 1, 1, 1, space + 9, size, 0), not_found);
    // The buffers: the in buffer exactly the entries, the out buffer at least a byte each.
    assert_eq!(ask(&mut s, one - 1, 1, 1, space, size, 0), invalid_size);
    assert_eq!(ask(&mut s, one + 1, 1, 1, space, size, 0), invalid_size);
    assert_eq!(ask(&mut s, one, 0, 1, space, size, 0), invalid_size);
    assert_eq!(ask(&mut s, one, 1, 1, space, size, 0), ResultCode::SUCCESS);
    // A refused entry is checked after them all: with a short out buffer, the buffer answers.
    let bad = Entry {
        read_only: 1,
        ..entry
    };
    s.place(&[bad]);
    let words = batch_words(one, 0, OUT_AT, 1, space, size, 0, 0);
    assert_eq!(result_of(&s.rig.send_on(s.worker, &words)), invalid_size);
}

/// The out buffer holds at least a byte per entry: with two entries, one byte is `InvalidSize`,
/// signalled, the out buffer untouched and nothing mapped; two bytes succeed.
#[test]
fn an_out_buffer_short_of_the_count_is_refused() {
    let mut s = sparse_ready();
    let entries = [
        Entry::mapped(REGION, 1, s.handle, 0),
        Entry::mapped(REGION + PAGE_SIZE, 1, s.handle, 0),
    ];
    let before = s.signals();
    assert_eq!(
        result_of(&s.batch_with(&entries, 1)),
        code(KernelError::InvalidSize)
    );
    assert_eq!(s.signals(), before + 1);
    assert_eq!(s.out(2), [STALE, STALE]);
    assert_eq!(s.rig.at(s.space, REGION), region_page(0));
    assert_eq!(result_of(&s.batch_with(&entries, 2)), ResultCode::SUCCESS);
    assert_eq!(s.out(3), [6, 6, STALE]);
}

/// [`MAX_ENTRIES`] entries are accepted, one more refused.
#[test]
fn a_batch_holds_up_to_256_entries() {
    let mut s = sparse_ready();
    let full = vec![Entry::back(REGION, 1); usize::try_from(MAX_ENTRIES).unwrap()];
    let reply = s.batch(&full);
    assert_eq!(result_of(&reply), ResultCode::SUCCESS);
    assert_eq!(data(&reply), halves(MAX_ENTRIES));
    let over = vec![Entry::back(REGION, 1); usize::try_from(MAX_ENTRIES).unwrap() + 1];
    let reply = s.batch(&over);
    assert_eq!(result_of(&reply), code(KernelError::InvalidSize));
}

/// A slot with no event is `NotFound`, after the count, entry size and sync checks, and signals
/// nothing.
#[test]
fn a_slot_with_no_event_is_not_found_and_signals_nothing() {
    let mut s = sparse_ready();
    s.place(&[Entry::mapped(REGION, 1, s.handle, 0)]);
    let size = u64::try_from(ENTRY_SIZE).unwrap();
    let one = u32::try_from(ENTRY_SIZE).unwrap();
    let before = s.signals();
    let words = batch_words(one, 1, OUT_AT, 1, s.space, size, 0, 5);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::NotFound)
    );
    let words = batch_words(one, 1, OUT_AT, 1, s.space, size, 2, 5);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::NotImplemented)
    );
    assert_eq!(s.signals(), before);
    assert!(
        s.rig.gpu().gpu_memory().spaces[&AsId(s.space)]
            .records
            .is_empty()
    );
}

/// The entries are checked in PR's two passes: first each entry's decoding (`NotImplemented`)
/// and handle (`NotFound`), entry by entry; then each entry's place (`InvalidCurrentMemory`).
/// Each pair of faults answers the first; a refused batch changes nothing, writes nothing, and
/// the next succeeds.
#[test]
fn entry_checks_walk_in_prs_order() {
    let mut s = sparse_ready();
    let good = Entry::mapped(REGION, 1, s.handle, 0);
    let undecodable = Entry {
        read_only: 1,
        ..good
    };
    let unknown = Entry {
        memory: s.handle + 1,
        ..good
    };
    let outside = Entry::mapped(REGION - PAGE_SIZE, 1, s.handle, 0);
    let refused_kinds = Entry {
        kind: 7,
        fallback: 7,
        ..good
    };
    let not_implemented = code(KernelError::NotImplemented);
    let not_found = code(KernelError::NotFound);
    let misplaced = code(KernelError::InvalidCurrentMemory);
    let front_calls = s.rig.front.calls().len();
    // Both faults in one entry: its decoding answers first.
    let both = Entry {
        memory: s.handle + 1,
        ..undecodable
    };
    for (entries, expected) in [
        (vec![both], not_implemented),
        (vec![undecodable, unknown], not_implemented),
        (vec![unknown, undecodable], not_found),
        (vec![good, outside, undecodable], not_implemented),
        (vec![outside, unknown], not_found),
        (vec![good, outside], misplaced),
        (vec![good, refused_kinds], misplaced),
    ] {
        let before = s.signals();
        assert_eq!(result_of(&s.batch(&entries)), expected);
        assert_eq!(s.signals(), before + 1);
        assert_eq!(s.out(entries.len()), vec![STALE; entries.len()]);
        assert_eq!(s.rig.at(s.space, REGION), region_page(0));
    }
    assert_eq!(s.rig.front.calls().len(), front_calls);
    // Kinds are selected only for an entry with a handle.
    let back = Entry {
        kind: 7,
        fallback: 7,
        ..Entry::back(REGION + PAGE_SIZE, 1)
    };
    assert_eq!(result_of(&s.batch(&[good, back])), ResultCode::SUCCESS);
    assert_eq!(s.out(2), [6, 7]);
    assert_eq!(s.rig.at(s.space, REGION), page_at(MEMORY, GENERIC_16BX2));
}

/// An entry lies inside one sparse region and inside its registration from its source: each
/// bound at both sides, with a second region right after the first.
#[test]
fn entry_bounds_are_checked_at_their_edges() {
    let mut s = sparse_ready();
    let next = REGION + REGION_SIZE;
    let words = sparse_words(
        s.space,
        next,
        next + 4 * PAGE_SIZE - 1,
        12,
        0,
        s.device,
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        ResultCode::SUCCESS
    );
    let refused = code(KernelError::InvalidCurrentMemory);
    let mut ask = |entry: Entry| result_of(&s.batch(&[entry]));
    assert_eq!(ask(Entry::back(REGION, 8)), ResultCode::SUCCESS);
    assert_eq!(ask(Entry::back(REGION, 9)), refused);
    assert_eq!(ask(Entry::back(REGION - PAGE_SIZE, 1)), refused);
    assert_eq!(ask(Entry::back(REGION - PAGE_SIZE, 2)), refused);
    assert_eq!(ask(Entry::back(next - PAGE_SIZE, 2)), refused);
    assert_eq!(ask(Entry::back(next - PAGE_SIZE, 1)), ResultCode::SUCCESS);
    assert_eq!(ask(Entry::back(next, 4)), ResultCode::SUCCESS);
    assert_eq!(ask(Entry::back(next + 4 * PAGE_SIZE, 1)), refused);
    assert_eq!(ask(Entry::back(next + 3 * PAGE_SIZE, 2)), refused);
    let handle = s.handle;
    let mut ask = |pages: u64, source: u64| {
        result_of(&s.batch(&[Entry::mapped(REGION, pages, handle, source)]))
    };
    assert_eq!(ask(1, MEMORY_SIZE - PAGE_SIZE), ResultCode::SUCCESS);
    assert_eq!(ask(1, MEMORY_SIZE), refused);
    assert_eq!(ask(2, MEMORY_SIZE - PAGE_SIZE), refused);
    assert_eq!(ask(4, 0), ResultCode::SUCCESS);
    assert_eq!(ask(5, 0), refused);
    assert_eq!(ask(1, u64::MAX - 2 * PAGE_SIZE + 1), refused);
}

/// The out buffer is written before anything changes: one the kernel cannot write refuses the
/// batch with the kernel's error, after every other check, and nothing is mapped or sent; the
/// same batch then succeeds.
#[test]
fn an_unwritable_out_buffer_changes_nothing() {
    let mut s = sparse_ready();
    let space = s.space;
    s.batch(&[Entry::mapped(REGION, 2, s.handle, 0)]);
    let calls = s.rig.front.calls().len();
    let entries = [
        Entry::back(REGION, 1),
        Entry::mapped(REGION + 4 * PAGE_SIZE, 1, s.handle, 0),
    ];
    let bytes = s.place(&entries);
    let words = batch_words(
        u32::try_from(bytes.len()).unwrap(),
        2,
        UNMAPPED_AT,
        2,
        space,
        u64::try_from(ENTRY_SIZE).unwrap(),
        0,
        0,
    );
    let before = s.signals();
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(s.signals(), before + 1);
    assert_eq!(s.rig.front.calls().len(), calls);
    assert_eq!(s.rig.gpu().gpu_memory().pending_jobs(), 0);
    assert_eq!(
        s.rig.at(space, REGION + 4 * PAGE_SIZE),
        region_page(4 * PAGE_SIZE)
    );
    assert_eq!(s.rig.at(space, REGION), page_at(MEMORY, GENERIC_16BX2));
    // A refusal before the out buffer is reached answers that refusal instead.
    let undecodable = Entry {
        read_only: 1,
        ..entries[0]
    };
    s.place(&[undecodable]);
    let words = batch_words(
        u32::try_from(ENTRY_SIZE).unwrap(),
        1,
        UNMAPPED_AT,
        1,
        space,
        u64::try_from(ENTRY_SIZE).unwrap(),
        0,
        0,
    );
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &words)),
        code(KernelError::NotImplemented)
    );
    assert_eq!(result_of(&s.batch(&entries)), ResultCode::SUCCESS);
    assert_eq!(s.rig.front.calls().len(), calls + 2);
}

/// Descriptors other than one A and one B are refused by the framework before the method runs
/// (`INVALID_CMIF_REQUEST`, no event signalled; PR answers `InvalidCombination` and signals), and
/// so is a request cut short.
#[test]
fn a_malformed_request_is_refused_unsignalled() {
    let mut s = sparse_ready();
    s.place(&[Entry::mapped(REGION, 1, s.handle, 0)]);
    let before = s.signals();
    // No B descriptor: `0x001000e8`, its three words dropped.
    let mut no_out = batch_words(40, 1, OUT_AT, 1, s.space, 40, 0, 0);
    no_out[0] = 0x0010_00E8;
    no_out.drain(5..8);
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &no_out)),
        INVALID_CMIF_REQUEST
    );
    let mut short = batch_words(40, 1, OUT_AT, 1, s.space, 40, 0, 0);
    short[1] = 6;
    short.pop();
    assert_eq!(
        result_of(&s.rig.send_on(s.worker, &short)),
        INVALID_CMIF_REQUEST
    );
    assert_eq!(s.signals(), before);
}

// ----- On the bare bookkeeping -----------------------------------------------------------------

/// An entry as [`SparseMapRequest`]: `pages` pages at `address` of `memory` from 0, cached,
/// generic.
fn request_at(address: u64, pages: u64, memory: u32) -> SparseMapRequest {
    SparseMapRequest {
        address,
        size: pages * PAGE_SIZE,
        read_only: 0,
        cache: 1,
        reserved0: 0,
        kind: 6,
        fallback: 6,
        comptag_mode: 0,
        reserved1: 0,
        memory,
        reserved2: 0,
        source_offset: 0,
    }
}

/// A batch that takes something out is `Busy` at [`MAX_PENDING_JOBS`] waiting jobs and changes
/// nothing; one that only maps needs no job and completes at once.
#[test]
fn a_batch_that_takes_out_is_busy_at_the_job_limit() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, PAGE_SIZE));
    fill_with_regions(&mut gpu, space, 2, 1);
    let event = ObjectId::from_parts(4, 5);
    let map = gpu
        .check_sparse_map(space, &[request_at(REGION, 1, handle)])
        .unwrap();
    assert!(!map.is_ordered());
    assert_eq!(map.apply(&front, event), SparseCompletion::AtOnce);
    let modify = ModifyRequest {
        space,
        first: REGION,
        last: REGION + PAGE_SIZE - 1,
        kind: 6,
        fallback: 6,
    };
    for _ in 1..MAX_PENDING_JOBS {
        gpu.modify_kind(&front, &modify, event).unwrap();
    }
    let back = [request_at(REGION, 1, 0)];
    assert!(gpu.check_sparse_map(space, &back).unwrap().is_ordered());
    gpu.modify_kind(&front, &modify, event).unwrap();
    assert_eq!(gpu.pending_jobs(), MAX_PENDING_JOBS);
    assert_eq!(
        gpu.check_sparse_map(space, &back).err(),
        Some(KernelError::Busy)
    );
    assert!(
        gpu.spaces[&space].records.contains_key(&REGION),
        "the refused batch took nothing out"
    );
    let only_maps = gpu
        .check_sparse_map(space, &[request_at(REGION + PAGE_SIZE, 1, handle)])
        .unwrap();
    assert_eq!(only_maps.apply(&front, event), SparseCompletion::AtOnce);
}

/// A bookkeeping one piece below [`MAX_PIECES`]: an address space with a registration of
/// [`MEMORY_SIZE`], a three-page region at [`REGION`], then one-page regions after it.
fn one_below_the_piece_limit() -> (GpuMemory, FakeFront, AsId, u32) {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let wide = SparseRequest {
        last: REGION + 3 * PAGE_SIZE - 1,
        ..sparse_at(space, 0, 1)
    };
    gpu.create_sparse_region(&wide).unwrap();
    for n in 3..u64::try_from(MAX_PIECES).unwrap() {
        gpu.create_sparse_region(&sparse_at(space, n, 1)).unwrap();
    }
    assert_eq!(gpu.pieces(space), MAX_PIECES - 1);
    (gpu, front, space, handle)
}

/// A batch that would leave more than [`MAX_PIECES`] pieces is `InvalidCurrentMemory` and
/// changes nothing; one that reaches the limit exactly fits; one that replaces a mapping adds
/// nothing and fits at the limit; one that splits a mapping adds two.
#[test]
fn a_batch_is_bounded_at_the_piece_limit() {
    let (mut gpu, front, space, handle) = one_below_the_piece_limit();
    let event = ObjectId::from_parts(4, 5);
    let three = request_at(REGION, 3, handle);
    gpu.check_sparse_map(space, &[three])
        .unwrap()
        .apply(&front, event);
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    let other = request_at(REGION + 3 * PAGE_SIZE, 1, handle);
    assert_eq!(
        gpu.check_sparse_map(space, &[other]).err(),
        Some(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    gpu.check_sparse_map(space, &[three])
        .unwrap()
        .apply(&front, event);
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    // Taking the middle page back splits the mapping in two: one more piece, refused at the
    // limit; with a region gone, it fits.
    let middle = request_at(REGION + PAGE_SIZE, 1, 0);
    assert_eq!(
        gpu.check_sparse_map(space, &[middle]).err(),
        Some(KernelError::InvalidCurrentMemory)
    );
    let last_region = RangeRequest {
        space,
        first: REGION + 3 * PAGE_SIZE,
        last: REGION + 4 * PAGE_SIZE - 1,
    };
    gpu.destroy_sparse_region(&front, &last_region, event)
        .unwrap();
    assert_eq!(gpu.pieces(space), MAX_PIECES - 1);
    let completion = gpu
        .check_sparse_map(space, &[middle])
        .unwrap()
        .apply(&front, event);
    assert!(matches!(completion, SparseCompletion::Ordered(_)));
    assert_eq!(gpu.pieces(space), MAX_PIECES);
}

/// The piece bound (check 4) answers before room for a job (check 5): at [`MAX_PIECES`] pieces
/// with [`MAX_PENDING_JOBS`] jobs waiting, a removal that splits a mapping is
/// `InvalidCurrentMemory`; one that differs only in adding no piece is `Busy`. Neither changes
/// anything.
#[test]
fn the_piece_bound_answers_before_busy() {
    let (mut gpu, front, space, handle) = one_below_the_piece_limit();
    let event = ObjectId::from_parts(4, 5);
    gpu.check_sparse_map(space, &[request_at(REGION, 3, handle)])
        .unwrap()
        .apply(&front, event);
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    let modify = ModifyRequest {
        space,
        first: REGION,
        last: REGION + 3 * PAGE_SIZE - 1,
        kind: 6,
        fallback: 6,
    };
    for _ in 0..MAX_PENDING_JOBS {
        gpu.modify_kind(&front, &modify, event).unwrap();
    }
    assert_eq!(gpu.pending_jobs(), MAX_PENDING_JOBS);
    assert_eq!(gpu.pieces(space), MAX_PIECES);
    let records = gpu.spaces[&space].records.clone();
    // Both faults: the middle page back splits the mapping, and there is no room for its job.
    let middle = request_at(REGION + PAGE_SIZE, 1, 0);
    assert_eq!(
        gpu.check_sparse_map(space, &[middle]).err(),
        Some(KernelError::InvalidCurrentMemory)
    );
    // The job fault alone: the first page back only shortens the mapping.
    let first = request_at(REGION, 1, 0);
    assert_eq!(
        gpu.check_sparse_map(space, &[first]).err(),
        Some(KernelError::Busy)
    );
    assert_eq!(gpu.spaces[&space].records, records);
    assert_eq!(gpu.pending_jobs(), MAX_PENDING_JOBS);
}

/// A bookkeeping with a batch's mapping and its waiting removal saves, restores, and completes
/// the removal against the restored table.
#[test]
fn a_batch_saves_and_restores() {
    let (mut gpu, front) = bare();
    let space = gpu.create_address_space(&front).unwrap();
    let handle = register_bare(&mut gpu, page_range(MEMORY, MEMORY_SIZE));
    let wide = SparseRequest {
        last: REGION + 4 * PAGE_SIZE - 1,
        ..sparse_at(space, 0, 1)
    };
    gpu.create_sparse_region(&wide).unwrap();
    // A job's event is a notification event the bookkeeping holds (the snapshot checks it).
    let event = ObjectId::from_parts(4, 5);
    gpu.set_notification_event(0, event);
    gpu.check_sparse_map(space, &[request_at(REGION, 2, handle)])
        .unwrap()
        .apply(&front, event);
    let completion = gpu
        .check_sparse_map(space, &[request_at(REGION + PAGE_SIZE, 1, 0)])
        .unwrap()
        .apply(&front, event);
    let SparseCompletion::Ordered(ticket) = completion else {
        unreachable!("a removal is ordered");
    };
    let bytes = gpu.save().unwrap();
    let kernel = Locks {
        process: process(),
        live: [((MEMORY, MEMORY_SIZE), 1)].into_iter().collect(),
    };
    let mut restored = GpuMemory::restore(&bytes, &kernel, &front).unwrap();
    assert_eq!(restored.spaces[&space].records, gpu.spaces[&space].records);
    let finished = restored.ordered_done(ticket).unwrap();
    assert_eq!(finished.event, event);
    let shown = restored.mappings(space);
    assert_eq!(shown_at(&shown, REGION), page_at(MEMORY, GENERIC_16BX2));
    assert_eq!(shown_at(&shown, REGION + PAGE_SIZE), region_page(PAGE_SIZE));
}
