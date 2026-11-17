//! The shared syncpoint page driven as a title drives it: 204 sent to the root as raw TIPC words
//! over the services framework's fake kernel, the title's request bits written into its memory
//! and its signal played on the request event's writable end (whose hook is gsv's completion
//! queue), and the timer thread's turn (the drain, then the page turn) played by the test.
//!
//! 204 is laid out by hand, word by word, from the layout PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97,
//! `BindSyncpointMemory`: `packet[0] != 0xdc || packet[1] != 0x80000006 || packet[2] != 4 ||
//! packet[5] != 1 || packet[6] != 0x1ff || packet[7] != 0x1000 || packet[8] != 0 || packet[9] !=
//! 0x2000 || packet[10] != 0`), which the tests write as they are. The page's words are read and
//! written at the offsets PR's `graphics_syncpoint_cache.h` @ e5fb3b97 gives, which the header
//! names.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};

use starquake_gpu::interface::{Completions, GpuCompletion, SYNCPOINT_COUNT};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::handle::Handle;
use starquake_kernel::memory::{DeviceShareToken, SvcPermission};
use starquake_kernel::slab::SlabId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel, Server, UnknownPolicy};
use starquake_trace::EventKind;

use super::*;
use crate::completion::Completion;
use crate::protocol::command::BIND_SYNCPOINT_MEMORY;
use crate::root::{RootState, root_server_with_page};
use crate::shared::WatchError;
use crate::testing::{SPECIAL, STALE, copied, halves, le, result_only, success, tipc_type};

/// The number the tests' root server is known by.
const ROOT: ServiceId = ServiceId(0x0657);
/// The number gsv records its notes under: not the root's, so mixing them up fails.
const GSV: ServiceId = ServiceId(0x065A);
/// Where the title's writable memory is, and its read-only memory.
const WRITABLE: u64 = 0x1_0000_0000;
const READ_ONLY: u64 = 0x1_0010_0000;
/// What the title's memory holds before 204, so that what 204 clears is seen to be cleared.
const DIRT: u8 = 0xEE;
/// A syncpoint in group 0 no other test value collides with, and two more.
const SP: u8 = 40;
const SP2: u8 = 41;
const SP3: u8 = 42;
/// A time the tests' first turn runs at.
const T0: HostNanos = HostNanos(0x10_0000);

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

fn object(index: u32, generation: u32) -> ObjectId {
    ObjectId::from_parts(index, generation)
}

fn len(size: u32) -> usize {
    usize::try_from(size).unwrap()
}

/// `BindSyncpointMemory` (204) as PR's check reads it: the header `0xdc, 0x80000006, 4`, the two
/// copied handles at words 3 and 4, then `1, 0x1ff`, the writable size and the read-only size.
fn bind_words(writable: u32, read_only: u32) -> Vec<u32> {
    vec![
        0xDC,
        0x8000_0006,
        4,
        writable,
        read_only,
        1,
        0x1FF,
        0x1000,
        0,
        0x2000,
        0,
    ]
}

#[test]
fn the_reference_request_words_are_the_shipping_constants() {
    let words = bind_words(7, 9);
    assert_eq!(words[0], tipc_type(BIND_SYNCPOINT_MEMORY));
    // A special header and six data words; two copied handles.
    assert_eq!(words[1], SPECIAL | 6);
    assert_eq!(words.len() - 5, 6);
    assert_eq!(words[2], 2 << 1);
    assert_eq!((words[3], words[4]), (7, 9));
    assert_eq!((words[5], words[6]), (REQUESTED_VERSION, REQUESTED_COUNT));
    assert_eq!(halves(u64::from(WRITABLE_SIZE)), [words[7], words[8]]);
    assert_eq!(halves(u64::from(READ_ONLY_SIZE)), [words[9], words[10]]);
}

// ----- The rig ------------------------------------------------------------------------------------

/// What the timer host was asked, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Act {
    /// An event recorded.
    Record(EventKind),
    /// `len` bytes written into `memory` at `offset`.
    Write {
        memory: ObjectId,
        offset: u64,
        len: usize,
    },
    /// An event signalled.
    Signal(ObjectId),
}

/// A [`TimerHost`] over the fake kernel, as the wiring's is over the real one: it reaches a
/// transfer memory through the fake's bounds check, which is the kernel's own (`span_within`),
/// so its refusals are the kernel's `IpcContext::read_transfer_memory`'s (`InvalidHandle` for an
/// object that is no live transfer memory, `InvalidCurrentMemory` past its end). A test can make
/// it refuse reads, or writes after a number of them.
struct PageHost<'a> {
    kernel: &'a mut FakeKernel,
    acts: Vec<Act>,
    refuse_reads: bool,
    writes_left: Option<usize>,
}

impl TimerHost for PageHost<'_> {
    fn record(&mut self, kind: EventKind) {
        self.acts.push(Act::Record(kind));
    }

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.acts.push(Act::Signal(event));
        Kernel::signal_event(self.kernel, event)
    }

    fn read_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        out: &mut [u8],
    ) -> SvcResult<()> {
        if self.refuse_reads {
            return Err(KernelError::InvalidHandle);
        }
        let address = self.kernel.lent_range(memory, offset, out.len())?;
        self.kernel.read_memory(address, out)
    }

    fn write_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        if let Some(left) = self.writes_left.as_mut() {
            if *left == 0 {
                return Err(KernelError::InvalidCurrentMemory);
            }
            *left -= 1;
        }
        let address = self.kernel.lent_range(memory, offset, data.len())?;
        self.kernel.write_memory(address, data)?;
        self.acts.push(Act::Write {
            memory,
            offset,
            len: data.len(),
        });
        Ok(())
    }
}

/// gsv's shared state over fresh syncpoints and a fresh queue.
fn fresh_shared() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Arc::new(Shared::new(syncpoints, queue, GSV))
}

/// A root server with the shared page, a kernel with the title's two memories lent (each
/// followed by a neighbour byte that starts exactly at its end), and one root session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    shared: Arc<Shared>,
    writable: ObjectId,
    read_only: ObjectId,
    /// The handles to the two memories in the title's table.
    handles: [u32; 2],
    /// What the next turn's host refuses.
    refuse_reads: bool,
    writes_left: Option<usize>,
}

impl Rig {
    fn with_sizes(writable_size: u32, read_only_size: u32) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(ROOT);
        kernel.map(WRITABLE, vec![DIRT; len(writable_size) + 1]);
        kernel.map(READ_ONLY, vec![DIRT; len(read_only_size) + 1]);
        let writable = kernel.lend(WRITABLE, u64::from(writable_size), SvcPermission::None);
        let read_only = kernel.lend(READ_ONLY, u64::from(read_only_size), SvcPermission::Read);
        let handles = [
            kernel.insert_handle(writable).unwrap().raw(),
            kernel.insert_handle(read_only).unwrap().raw(),
        ];
        let session = kernel.new_object();
        let shared = fresh_shared();
        let mut server =
            root_server_with_page(RootState::new(), UnknownPolicy::Strict, Arc::clone(&shared))
                .with_service_id(ROOT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            shared,
            writable,
            read_only,
            handles,
            refuse_reads: false,
            writes_left: None,
        }
    }

    fn new() -> Self {
        Self::with_sizes(WRITABLE_SIZE, READ_ONLY_SIZE)
    }

    /// A rig whose page is bound.
    fn bound() -> Self {
        let mut rig = Self::new();
        let reply = rig.bind();
        assert_eq!(
            reply,
            success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]),
            "bind"
        );
        rig
    }

    /// Sends the message `words` lays out; the message buffer afterwards.
    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// 204 with the rig's two memories.
    fn bind(&mut self) -> Vec<u8> {
        let [writable, read_only] = self.handles;
        self.send(&bind_words(writable, read_only))
    }

    fn page(&self) -> Page {
        self.shared.lock().page().clone()
    }

    fn request_event(&self) -> ObjectId {
        self.page().request_event().unwrap().writable
    }

    fn answer_event(&self) -> ObjectId {
        self.page().answer_event().unwrap()
    }

    /// The `u64` at `offset` of the read-only memory.
    fn read_only_word(&self, offset: u64) -> u64 {
        let bytes = self.kernel.memory(READ_ONLY + offset, 8).unwrap();
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    /// The `u64` at `offset` of the writable memory.
    fn writable_word(&self, offset: u64) -> u64 {
        let bytes = self.kernel.memory(WRITABLE + offset, 8).unwrap();
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn write_writable(&mut self, offset: u64, value: u64) {
        self.kernel
            .write_memory(WRITABLE + offset, &value.to_le_bytes())
            .unwrap();
    }

    /// The value the page shows for `sp`.
    fn value(&self, sp: u8) -> u64 {
        self.read_only_word(value_offset(id(sp)))
    }

    /// Plays the title asking about `sps` as the client does: each slot's request bit flipped
    /// against its acknowledge bit, and each touched group's summary bit flipped against the
    /// acknowledge summary. Nothing is signalled.
    fn ask(&mut self, sps: &[u8]) {
        let mut groups = 0;
        for sp in sps {
            let group = usize::from(*sp) / GROUP_BITS;
            let bit = usize::from(*sp) % GROUP_BITS;
            let ack = self.read_only_word(group_offset(SLOT_ACK_OFFSET, group));
            let current = self.writable_word(group_offset(SLOT_REQUEST_OFFSET, group));
            // The bit is set to differ from the acknowledge bit; other bits are kept.
            let wanted = (current & !(1 << bit)) | ((!ack) & (1 << bit));
            self.write_writable(group_offset(SLOT_REQUEST_OFFSET, group), wanted);
            groups |= 1 << group;
        }
        let ack = self.read_only_word(u64::from(GROUP_ACK_OFFSET));
        self.write_writable(u64::from(GROUP_REQUEST_OFFSET), ack ^ groups);
    }

    /// Plays the title's `SignalEvent` on the request event's writable end.
    fn signal(&mut self) {
        let event = self.request_event();
        self.kernel.guest_signal(event).unwrap();
    }

    /// Plays the timer thread's turn at `now`: the drain, then the page turn. What the host was
    /// asked.
    fn turn(&mut self, now: HostNanos) -> Vec<Act> {
        let mut host = PageHost {
            kernel: &mut self.kernel,
            acts: Vec::new(),
            refuse_reads: self.refuse_reads,
            writes_left: self.writes_left,
        };
        self.shared.drain(&mut host);
        self.shared.tick_page(&mut host, now);
        host.acts
    }

    fn increment(&self, sp: u8) -> u32 {
        self.shared.syncpoints().increment(id(sp))
    }

    fn watchers(&self, sp: u8) -> u32 {
        self.shared.lock().watchers(id(sp))
    }

    fn answers(&self) -> u32 {
        self.kernel.signals(self.answer_event()).unwrap()
    }
}

/// The note an answer records for `sp` read as `value`.
fn answered(sp: u8, value: u32) -> Act {
    Act::Record(Note::SyncptPageAnswered { id: id(sp), value }.event(GSV))
}

/// The value written for `sp`, a group's acknowledge word and the summary.
fn value_write(rig: &Rig, sp: u8) -> Act {
    Act::Write {
        memory: rig.read_only,
        offset: value_offset(id(sp)),
        len: 8,
    }
}

fn group_ack_write(rig: &Rig, group: usize) -> Act {
    Act::Write {
        memory: rig.read_only,
        offset: group_offset(SLOT_ACK_OFFSET, group),
        len: 8,
    }
}

/// The note an answer records once its acknowledge summary is written: the groups it answered.
fn acknowledged(groups: u64) -> Act {
    Act::Record(Note::SyncptPageAcknowledged { groups }.event(GSV))
}

fn summary_write(rig: &Rig) -> Act {
    Act::Write {
        memory: rig.read_only,
        offset: u64::from(GROUP_ACK_OFFSET),
        len: 8,
    }
}

/// Whether the read-only memory acknowledges everything the writable memory asks.
fn all_acknowledged(rig: &Rig) -> bool {
    let summary = rig.writable_word(u64::from(GROUP_REQUEST_OFFSET))
        == rig.read_only_word(u64::from(GROUP_ACK_OFFSET));
    summary
        && (0..GROUP_COUNT).all(|group| {
            rig.writable_word(group_offset(SLOT_REQUEST_OFFSET, group))
                == rig.read_only_word(group_offset(SLOT_ACK_OFFSET, group))
        })
}

/// A bound rig whose first request for `SP` was answered, so the page shows its counter: the
/// next request for it waits for a change.
fn shown() -> Rig {
    let mut rig = Rig::bound();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.answers(), 1);
    rig
}

// ----- The layout ---------------------------------------------------------------------------------

#[test]
fn the_header_is_pinned() {
    // Eight little-endian words, written out from PR's `Header`: 1, 0, 191, 0x40, 0, 0x640, 8,
    // 0x648.
    let golden: [u8; 32] = [
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xBF, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x06, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x48, 0x06,
        0x00, 0x00,
    ];
    assert_eq!(header(), golden);
    assert_eq!(HEADER_SIZE, golden.len());
}

#[test]
fn the_header_count_names_exactly_the_page_s_groups() {
    // NX2's comment: the client requires `((count + 0x40) >> 6) - 1 < 0x40`.
    let groups = |count: u32| usize::try_from((count + 0x40) >> 6).unwrap();
    assert_eq!(groups(HEADER_COUNT), GROUP_COUNT);
    assert!(groups(HEADER_COUNT) - 1 < 0x40);
    // NX2's reading, the count itself, would name one group more than the page has.
    assert_eq!(groups(SYNCPOINT_COUNT), GROUP_COUNT + 1);
}

#[test]
fn the_areas_fit_their_memories_and_do_not_overlap() {
    let double = 8_u64;
    let groups = u64::try_from(GROUP_COUNT).unwrap();
    let values = u64::from(SYNCPOINT_COUNT) * double;
    let read_only = [
        (0, u64::try_from(HEADER_SIZE).unwrap()),
        (u64::from(VALUES_OFFSET), values),
        (u64::from(GROUP_ACK_OFFSET), double),
        (u64::from(SLOT_ACK_OFFSET), groups * double),
    ];
    let writable = [
        (u64::from(GROUP_REQUEST_OFFSET), double),
        (u64::from(SLOT_REQUEST_OFFSET), groups * double),
    ];
    for (areas, size) in [
        (&read_only[..], READ_ONLY_SIZE),
        (&writable[..], WRITABLE_SIZE),
    ] {
        for (index, (start, length)) in areas.iter().enumerate() {
            assert!(start + length <= u64::from(size), "{start:#x}");
            for (other_start, other_length) in &areas[index + 1..] {
                assert!(
                    start + length <= *other_start || other_start + other_length <= *start,
                    "{start:#x} and {other_start:#x}"
                );
            }
        }
    }
    // The values end exactly where the acknowledge summary starts.
    assert_eq!(
        u64::from(VALUES_OFFSET) + values,
        u64::from(GROUP_ACK_OFFSET)
    );
    // Every bit of every group names a syncpoint.
    assert_eq!(
        GROUP_COUNT * GROUP_BITS,
        usize::try_from(SYNCPOINT_COUNT).unwrap()
    );
}

#[test]
fn ids_place_each_syncpoint_in_its_group_and_bit() {
    // The first and last of each group, in a set and out of it.
    let edges = [0, 63, 64, 127, 128, 191];
    let set = Ids::of(edges.iter().map(|sp| id(*sp)));
    assert_eq!(set.0, [1 | 1 << 63, 1 | 1 << 63, 1 | 1 << 63]);
    assert_eq!(
        set.iter().map(SyncpointId::get).collect::<Vec<_>>(),
        edges.to_vec()
    );
    assert!(set.contains(id(64)) && !set.contains(id(65)));
    assert_eq!(
        set.in_group(1).map(SyncpointId::get).collect::<Vec<_>>(),
        [64, 127]
    );
    assert!(Ids::default().is_empty() && !set.is_empty());
    assert_eq!(value_offset(id(191)), 0x40 + 191 * 8);
}

// ----- Binding (204) ------------------------------------------------------------------------------

#[test]
fn bind_clears_the_memories_and_writes_the_header_and_unallocated_values() {
    let mut rig = Rig::new();
    let reply = rig.bind();
    let handles = copied(&reply);
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &handles, &[]));
    // The two handles: the request event's writable end, then the answer event.
    let page = rig.page();
    let request = page.request_event().unwrap();
    let answer = page.answer_event().unwrap();
    let handed: Vec<ObjectId> = handles
        .iter()
        .map(|handle| {
            rig.kernel
                .resolve_handle(Handle::from_raw(*handle))
                .unwrap()
        })
        .collect();
    assert_eq!(handed, [request.writable, answer]);
    assert_eq!(
        page.binding(),
        Some(Binding {
            writable: rig.writable,
            read_only: rig.read_only,
        })
    );
    assert_eq!(rig.kernel.is_signalled(answer), Some(false));
    // The writable memory is cleared, its neighbour is not.
    let writable = rig.kernel.memory(WRITABLE, len(WRITABLE_SIZE) + 1).unwrap();
    assert!(writable[..len(WRITABLE_SIZE)].iter().all(|byte| *byte == 0));
    assert_eq!(writable[len(WRITABLE_SIZE)], DIRT);
    // The read-only memory: the header, zeros to the values, every value all ones, zeros after;
    // its neighbour untouched.
    let read_only = rig
        .kernel
        .memory(READ_ONLY, len(READ_ONLY_SIZE) + 1)
        .unwrap();
    let values = len(VALUES_OFFSET);
    let values_end = values + usize::try_from(SYNCPOINT_COUNT).unwrap() * 8;
    assert_eq!(read_only[..HEADER_SIZE], header());
    assert!(read_only[HEADER_SIZE..values].iter().all(|byte| *byte == 0));
    assert!(
        read_only[values..values_end]
            .iter()
            .all(|byte| *byte == 0xFF)
    );
    assert!(
        read_only[values_end..len(READ_ONLY_SIZE)]
            .iter()
            .all(|byte| *byte == 0)
    );
    assert_eq!(read_only[len(READ_ONLY_SIZE)], DIRT);
    // The request event's hook is gsv's queue: the title's signal reaches it.
    rig.signal();
    assert_eq!(
        rig.shared.queue().take_all(),
        [Completion::GuestSignalled {
            event: request.writable
        }]
    );
    assert_eq!(rig.page().shown(id(SP)), UNALLOCATED);
}

#[test]
fn bind_reads_any_data_words() {
    // Every data word other than PR's, each distinct.
    let mut rig = Rig::new();
    let [writable, read_only] = rig.handles;
    let mut words = bind_words(writable, read_only);
    words[5..].copy_from_slice(&[2, 0x40, 0x10, 0x3, 0x20, 0x4]);
    let reply = rig.send(&words);
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
    assert!(rig.page().binding().is_some());
}

#[test]
fn a_short_bind_request_is_refused_and_changes_nothing() {
    let mut rig = Rig::new();
    let [writable, read_only] = rig.handles;
    // One copied handle, and the six data words with the last one missing.
    let mut one_handle = bind_words(writable, read_only);
    one_handle[2] = 1 << 1;
    one_handle.remove(4);
    let mut short = bind_words(writable, read_only);
    short[1] = SPECIAL | 5;
    short.pop();
    for words in [one_handle, short] {
        assert_eq!(
            rig.send(&words),
            result_only(BIND_SYNCPOINT_MEMORY, INVALID_CMIF_REQUEST)
        );
        assert_eq!(rig.page(), Page::default());
        assert_eq!(rig.kernel.memory(WRITABLE, 1), Some(&[DIRT][..]));
    }
}

/// Whether nothing of 204 happened: no events, nothing bound, the memories as they were.
fn untouched(rig: &Rig) -> bool {
    rig.page() == Page::default()
        && rig.kernel.memory(WRITABLE, 1) == Some(&[DIRT][..])
        && rig.kernel.memory(READ_ONLY, 1) == Some(&[DIRT][..])
}

#[test]
fn each_handle_must_name_a_live_transfer_memory_and_not_the_same_one() {
    let refused = result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidHandle.code());
    for case in ["first", "second", "both the same", "ended"] {
        let mut rig = Rig::new();
        let [writable, read_only] = rig.handles;
        let other = rig.kernel.new_object();
        let other = rig.kernel.insert_handle(other).unwrap().raw();
        let words = match case {
            "first" => bind_words(other, read_only),
            "second" => bind_words(writable, other),
            "both the same" => bind_words(read_only, read_only),
            _ => {
                assert!(rig.kernel.end_transfer_memory(rig.writable));
                bind_words(writable, read_only)
            }
        };
        assert_eq!(rig.send(&words), refused, "{case}");
        assert!(untouched(&rig), "{case}");
    }
}

#[test]
fn each_size_is_checked_at_its_edge() {
    let refused = result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidSize.code());
    for (writable, read_only) in [
        (WRITABLE_SIZE - 1, READ_ONLY_SIZE),
        (WRITABLE_SIZE + 1, READ_ONLY_SIZE),
        (WRITABLE_SIZE, READ_ONLY_SIZE - 1),
        (WRITABLE_SIZE, READ_ONLY_SIZE + 1),
    ] {
        let mut rig = Rig::with_sizes(writable, read_only);
        assert_eq!(rig.bind(), refused, "{writable:#x} {read_only:#x}");
        assert!(untouched(&rig));
    }
    // Exactly the sizes binds.
    let mut rig = Rig::new();
    let reply = rig.bind();
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
}

#[test]
fn the_bind_checks_come_in_their_documented_order() {
    // Two faults at once each time: the earlier check's error is the answer.
    let state = result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidState.code());
    let handle = result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidHandle.code());
    // Bound already, and a handle to something else: the state first.
    let mut rig = Rig::bound();
    let other = rig.kernel.new_object();
    let other = rig.kernel.insert_handle(other).unwrap().raw();
    assert_eq!(rig.send(&bind_words(other, rig.handles[1])), state);
    // One memory for both, and of the wrong sizes: the handles first.
    let mut rig = Rig::with_sizes(WRITABLE_SIZE + 1, READ_ONLY_SIZE + 1);
    assert_eq!(
        rig.send(&bind_words(rig.handles[0], rig.handles[0])),
        handle
    );
    assert!(untouched(&rig));
    // A missing data word, and bound already: the request's shape first.
    let mut rig = Rig::bound();
    let mut short = bind_words(rig.handles[0], rig.handles[1]);
    short[1] = SPECIAL | 5;
    short.pop();
    assert_eq!(
        rig.send(&short),
        result_only(BIND_SYNCPOINT_MEMORY, INVALID_CMIF_REQUEST)
    );
}

#[test]
fn a_second_bind_is_refused_and_changes_nothing() {
    let mut rig = Rig::bound();
    let before = rig.page();
    // A second pair of memories, lent and handed over.
    rig.kernel
        .map(0x2_0000_0000, vec![DIRT; len(WRITABLE_SIZE)]);
    rig.kernel
        .map(0x2_0010_0000, vec![DIRT; len(READ_ONLY_SIZE)]);
    let writable = rig
        .kernel
        .lend(0x2_0000_0000, u64::from(WRITABLE_SIZE), SvcPermission::None);
    let read_only = rig.kernel.lend(
        0x2_0010_0000,
        u64::from(READ_ONLY_SIZE),
        SvcPermission::Read,
    );
    let writable = rig.kernel.insert_handle(writable).unwrap().raw();
    let read_only = rig.kernel.insert_handle(read_only).unwrap().raw();
    assert_eq!(
        rig.send(&bind_words(writable, read_only)),
        result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidState.code())
    );
    assert_eq!(rig.page(), before);
    assert_eq!(rig.kernel.memory(0x2_0010_0000, 1), Some(&[DIRT][..]));
    // A second root session reaches the same page.
    let second = rig.kernel.new_object();
    rig.server.accept(second).unwrap();
    rig.session = second;
    assert_eq!(
        rig.bind(),
        result_only(BIND_SYNCPOINT_MEMORY, KernelError::InvalidState.code())
    );
}

#[test]
fn a_refused_event_binds_nothing_and_the_next_bind_succeeds() {
    let mut rig = Rig::new();
    rig.kernel.refuse_objects(Some(KernelError::OutOfResource));
    assert_eq!(
        rig.bind(),
        result_only(BIND_SYNCPOINT_MEMORY, KernelError::OutOfResource.code())
    );
    // The events are made before the memories are touched, as PR makes its event first: a
    // refused event leaves the memories as they were.
    assert!(untouched(&rig));
    rig.kernel.refuse_objects(None);
    let reply = rig.bind();
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &copied(&reply), &[]));
}

#[test]
fn events_made_before_a_failed_bind_are_handed_out_by_the_next() {
    // A page whose events were made but which is not bound: what a 204 that failed after making
    // them leaves (restored here, since the fake kernel refuses every object at once).
    let mut rig = Rig::new();
    let request = RequestEvent {
        writable: object(0x51, 2),
        readable: object(0x52, 2),
    };
    let page = Page {
        request: Some(request),
        answer: Some(object(0x53, 2)),
        ..Page::default()
    };
    rig.shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    let reply = rig.bind();
    let handles = copied(&reply);
    assert_eq!(reply, success(BIND_SYNCPOINT_MEMORY, &handles, &[]));
    let handed: Vec<ObjectId> = handles
        .iter()
        .map(|handle| {
            rig.kernel
                .resolve_handle(Handle::from_raw(*handle))
                .unwrap()
        })
        .collect();
    assert_eq!(handed, [request.writable, object(0x53, 2)]);
}

#[test]
fn a_root_without_the_page_answers_204_with_the_stub_s_events() {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = crate::root::root_server(UnknownPolicy::Strict).with_service_id(ROOT);
    server.accept(session).unwrap();
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    let bytes = le(&bind_words(1, 2));
    buffer[..bytes.len()].copy_from_slice(&bytes);
    server.handle(&mut kernel, session, &mut buffer).unwrap();
    // The stub reads nothing and hands out two fresh events.
    assert_eq!(
        buffer,
        success(BIND_SYNCPOINT_MEMORY, &copied(&buffer), &[])
    );
    assert_eq!(copied(&buffer).len(), 2);
    let family = server.family::<crate::root::RootFamily>().unwrap();
    assert!(!family.has_page());
}

// ----- Answering --------------------------------------------------------------------------------

#[test]
fn a_request_for_a_counter_that_moved_is_answered_on_signal() {
    let mut rig = Rig::bound();
    rig.increment(SP);
    rig.increment(SP);
    rig.ask(&[SP]);
    // Nothing happens before the title signals, however much time passes.
    assert_eq!(rig.turn(HostNanos(T0.0 + 10 * ANSWER_BOUND)), []);
    rig.signal();
    let acts = rig.turn(T0);
    assert_eq!(
        acts,
        [
            value_write(&rig, SP),
            answered(SP, 2),
            group_ack_write(&rig, 0),
            summary_write(&rig),
            acknowledged(1 << 0),
            Act::Signal(rig.answer_event()),
        ]
    );
    assert_eq!(rig.value(SP), 2);
    assert!(all_acknowledged(&rig));
    assert_eq!(rig.answers(), 1);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.watchers(SP), 0);
    assert!(!rig.shared.syncpoints().is_watched(id(SP)));
}

#[test]
fn a_request_for_a_counter_that_did_not_move_is_held_and_watched() {
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    assert_eq!(rig.turn(T0), []);
    assert_eq!(rig.answers(), 1);
    assert_eq!(
        rig.page().held(),
        Some(Hold {
            ids: Ids::of([id(SP)]),
            since: T0,
        })
    );
    assert_eq!(rig.watchers(SP), 1);
    assert!(rig.shared.syncpoints().is_watched(id(SP)));
    assert_eq!(
        rig.shared.page_deadline(),
        Some(HostNanos(T0.0 + ANSWER_BOUND))
    );
    assert!(!all_acknowledged(&rig));
}

#[test]
fn a_held_request_is_answered_when_its_counter_changes() {
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    // The hold watches the counter, so its increment is pushed and drained, and the turn after
    // answers with the new value, well before the bound.
    assert_eq!(rig.increment(SP), 1);
    let acts = rig.turn(HostNanos(T0.0 + 1));
    let observed = Act::Record(
        Note::SyncpointObserved {
            id: id(SP),
            value: 1,
        }
        .event(GSV),
    );
    assert_eq!(
        acts,
        [
            observed,
            value_write(&rig, SP),
            answered(SP, 1),
            group_ack_write(&rig, 0),
            summary_write(&rig),
            acknowledged(1 << 0),
            Act::Signal(rig.answer_event()),
        ]
    );
    assert_eq!(rig.value(SP), 1);
    assert_eq!(rig.answers(), 2);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.shared.page_deadline(), None);
}

#[test]
fn a_held_request_is_answered_at_exactly_the_bound_and_not_before() {
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    let due = HostNanos(T0.0 + ANSWER_BOUND);
    assert_eq!(rig.shared.page_deadline(), Some(due));
    // A nanosecond before: still held, and the hold keeps its first time.
    assert_eq!(rig.turn(HostNanos(due.0 - 1)), []);
    assert_eq!(rig.page().held().map(|held| held.since), Some(T0));
    // At the bound: answered with the value it shows already.
    let acts = rig.turn(due);
    assert_eq!(
        acts,
        [
            value_write(&rig, SP),
            answered(SP, 0),
            group_ack_write(&rig, 0),
            summary_write(&rig),
            acknowledged(1 << 0),
            Act::Signal(rig.answer_event()),
        ]
    );
    assert_eq!(rig.answers(), 2);
    assert!(all_acknowledged(&rig));
    assert_eq!(rig.watchers(SP), 0);
}

#[test]
fn a_hold_s_due_time_stops_at_the_end_of_the_clock() {
    let hold = |since: u64| Hold {
        ids: Ids::of([id(SP)]),
        since: HostNanos(since),
    };
    let last = u64::MAX - ANSWER_BOUND;
    assert_eq!(hold(last).due(), HostNanos(u64::MAX));
    assert_eq!(hold(last + 1).due(), HostNanos(u64::MAX));
    assert_eq!(hold(u64::MAX).due(), HostNanos(u64::MAX));
}

#[test]
fn an_answer_writes_values_then_each_group_s_acknowledge_then_the_summary_then_signals() {
    // Syncpoints in all three groups, two in the middle one, each counter at a distinct value.
    let sps = [3, 69, 70, 130];
    let mut rig = Rig::bound();
    for (times, sp) in sps.iter().enumerate() {
        for _ in 0..=times {
            rig.increment(*sp);
        }
    }
    rig.ask(&sps);
    rig.signal();
    let acts = rig.turn(T0);
    assert_eq!(
        acts,
        [
            value_write(&rig, 3),
            answered(3, 1),
            group_ack_write(&rig, 0),
            value_write(&rig, 69),
            answered(69, 2),
            value_write(&rig, 70),
            answered(70, 3),
            group_ack_write(&rig, 1),
            value_write(&rig, 130),
            answered(130, 4),
            group_ack_write(&rig, 2),
            summary_write(&rig),
            acknowledged(GROUP_MASK),
            Act::Signal(rig.answer_event()),
        ]
    );
    for (value, sp) in (1..).zip(sps) {
        assert_eq!(rig.value(sp), value, "{sp}");
    }
    assert!(all_acknowledged(&rig));
    // Syncpoints not asked about still show nothing.
    assert_eq!(rig.value(4), UNALLOCATED);
}

#[test]
fn slot_bits_of_a_group_whose_summary_bit_matches_are_not_answered() {
    let mut rig = Rig::bound();
    rig.ask(&[SP]);
    // Group 1's slot bit for 70 flipped, its summary bit not.
    let group1 = group_offset(SLOT_REQUEST_OFFSET, 1);
    rig.write_writable(group1, 1 << (70 - 64));
    rig.signal();
    let acts = rig.turn(T0);
    assert!(!acts.contains(&value_write(&rig, 70)));
    assert!(!acts.contains(&group_ack_write(&rig, 1)));
    assert_eq!(rig.value(70), UNALLOCATED);
    assert_eq!(rig.read_only_word(group_offset(SLOT_ACK_OFFSET, 1)), 0);
    assert_eq!(rig.value(SP), 0);
    assert_eq!(rig.answers(), 1);
}

#[test]
fn a_summary_bit_flipped_alone_is_answered_at_once() {
    // Group 2's summary bit flips with no slot bit: no counter could change, so it is not held.
    // It writes no value, and is recorded all the same: the acknowledge words it wrote.
    let mut rig = shown();
    let summary = rig.read_only_word(u64::from(GROUP_ACK_OFFSET));
    rig.write_writable(u64::from(GROUP_REQUEST_OFFSET), summary ^ (1 << 2));
    rig.signal();
    let acts = rig.turn(T0);
    assert_eq!(
        acts,
        [
            group_ack_write(&rig, 2),
            summary_write(&rig),
            acknowledged(1 << 2),
            Act::Signal(rig.answer_event()),
        ]
    );
    assert_eq!(rig.page().held(), None);
    assert!(all_acknowledged(&rig));
}

#[test]
fn the_page_shows_the_counter_not_a_stale_drained_value() {
    // The page shows 5; the counter moves to 7, and a stale change of 3 is queued (an
    // incrementer that lost the race). Extending against 3 would read it as a wrap.
    let mut rig = Rig::bound();
    for _ in 0..5 {
        rig.increment(SP);
    }
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.value(SP), 5);
    rig.increment(SP);
    rig.increment(SP);
    rig.shared
        .queue()
        .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
            id: id(SP),
            value: 3,
        }));
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.value(SP), 7);
    assert_eq!(rig.page().shown(id(SP)), 7);
}

#[test]
fn an_unallocated_syncpoint_s_counter_is_shown() {
    let mut rig = Rig::bound();
    assert!(!rig.shared.syncpoints().is_allocated(id(SP)));
    rig.increment(SP);
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.value(SP), 1);
}

#[test]
fn a_counter_that_wraps_is_shown_past_the_wrap() {
    // The page shows u32::MAX; one more increment wraps the counter to 0, which the page shows
    // as one past u32::MAX.
    let mut rig = Rig::bound();
    // Taking a counter to u32::MAX takes 2^32 increments, so the page is put where an answer at
    // u32::MAX leaves it, through a restore, and the fresh counter (0) is one increment past it.
    let mut page = rig.page();
    page.shown.observe(id(SP), Some(u32::MAX));
    rig.shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    assert_eq!(rig.shared.syncpoints().read(id(SP)), 0);
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.value(SP), u64::from(u32::MAX) + 1);
}

#[test]
fn answering_gives_back_the_hold_s_watches() {
    let mut rig = Rig::bound();
    rig.ask(&[SP, SP2, SP3]);
    rig.signal();
    rig.turn(T0);
    // All three shown; ask two again: held, each watched once.
    rig.ask(&[SP, SP2]);
    rig.signal();
    rig.turn(T0);
    assert_eq!((rig.watchers(SP), rig.watchers(SP2)), (1, 1));
    // The title adds a third and withdraws the second: the hold follows, keeping its first time.
    rig.ask(&[SP3]);
    let group0 = group_offset(SLOT_REQUEST_OFFSET, 0);
    let request = rig.writable_word(group0);
    rig.write_writable(group0, request ^ (1 << SP2));
    rig.signal();
    rig.turn(HostNanos(T0.0 + 5));
    assert_eq!(
        rig.page().held(),
        Some(Hold {
            ids: Ids::of([id(SP), id(SP3)]),
            since: T0,
        })
    );
    assert_eq!(
        (rig.watchers(SP), rig.watchers(SP2), rig.watchers(SP3)),
        (1, 0, 1)
    );
    // A change answers, and every watch goes back.
    rig.increment(SP3);
    rig.turn(HostNanos(T0.0 + 6));
    assert_eq!(rig.answers(), 2);
    assert_eq!(
        (rig.watchers(SP), rig.watchers(SP2), rig.watchers(SP3)),
        (0, 0, 0)
    );
    for sp in [SP, SP2, SP3] {
        assert!(!rig.shared.syncpoints().is_watched(id(sp)));
    }
}

#[test]
fn a_withdrawn_request_drops_the_hold_without_an_answer() {
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    assert!(rig.page().held().is_some());
    // The title puts its bits back as the acknowledges have them.
    let ack = rig.read_only_word(u64::from(GROUP_ACK_OFFSET));
    let slots = rig.read_only_word(group_offset(SLOT_ACK_OFFSET, 0));
    rig.write_writable(u64::from(GROUP_REQUEST_OFFSET), ack);
    rig.write_writable(group_offset(SLOT_REQUEST_OFFSET, 0), slots);
    assert_eq!(rig.turn(HostNanos(T0.0 + 1)), []);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.watchers(SP), 0);
    assert_eq!(rig.answers(), 1);
}

#[test]
fn a_request_that_cannot_be_watched_is_answered_at_once() {
    let mut rig = Rig::bound();
    rig.ask(&[SP, SP2]);
    rig.signal();
    rig.turn(T0);
    // SP2's waiter count is at its maximum, so a hold cannot watch it: the request is answered
    // at once, and the watch taken of SP is given back.
    rig.shared.set_watchers(id(SP2), u32::MAX);
    assert_eq!(
        rig.shared.lock().watch(id(SP2)),
        Err(WatchError::TooManyWaiters(id(SP2)))
    );
    rig.ask(&[SP, SP2]);
    rig.signal();
    rig.turn(T0);
    assert_eq!(rig.answers(), 2);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.watchers(SP), 0);
    assert_eq!(rig.watchers(SP2), u32::MAX);
}

#[test]
fn a_refused_access_drops_the_hold_and_the_next_signal_tries_again() {
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    // Reads refused: the hold is dropped, nothing answered.
    rig.refuse_reads = true;
    assert_eq!(rig.turn(HostNanos(T0.0 + 1)), []);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.watchers(SP), 0);
    // Without a signal the next turn does nothing; with one, it reads again.
    rig.refuse_reads = false;
    rig.increment(SP);
    assert_eq!(rig.turn(HostNanos(T0.0 + 2)), []);
    rig.signal();
    rig.turn(HostNanos(T0.0 + 3));
    assert_eq!(rig.answers(), 2);
    assert_eq!(rig.value(SP), 1);
    // Writes refused after the first value: the rest is not written, nothing is signalled, and
    // the next signal answers in full.
    rig.increment(SP);
    rig.increment(SP2);
    rig.ask(&[SP, SP2]);
    rig.signal();
    rig.writes_left = Some(1);
    rig.turn(HostNanos(T0.0 + 4));
    assert_eq!(rig.answers(), 2);
    assert_eq!((rig.value(SP), rig.value(SP2)), (2, UNALLOCATED));
    assert!(!all_acknowledged(&rig));
    assert_eq!(rig.page().held(), None);
    rig.writes_left = None;
    rig.signal();
    rig.turn(HostNanos(T0.0 + 5));
    assert_eq!(rig.answers(), 3);
    assert_eq!((rig.value(SP), rig.value(SP2)), (2, 1));
    assert!(all_acknowledged(&rig));
}

#[test]
fn a_stopped_answer_s_values_count_as_shown_and_the_next_signal_holds_to_the_bound() {
    // The value is written and the group's acknowledge write refused: the page shows the
    // counter, so the next signal finds nothing newer and holds the request, to the bound.
    let mut rig = Rig::bound();
    rig.increment(SP);
    rig.ask(&[SP]);
    rig.signal();
    rig.writes_left = Some(1);
    assert_eq!(rig.turn(T0), [value_write(&rig, SP), answered(SP, 1)]);
    assert_eq!(rig.answers(), 0);
    assert_eq!(rig.page().held(), None);
    rig.writes_left = None;
    rig.signal();
    let since = HostNanos(T0.0 + 1);
    assert_eq!(rig.turn(since), []);
    assert_eq!(
        rig.page().held(),
        Some(Hold {
            ids: Ids::of([id(SP)]),
            since,
        })
    );
    let due = HostNanos(since.0 + ANSWER_BOUND);
    assert_eq!(rig.turn(HostNanos(due.0 - 1)), []);
    assert_eq!(
        rig.turn(due),
        [
            value_write(&rig, SP),
            answered(SP, 1),
            group_ack_write(&rig, 0),
            summary_write(&rig),
            acknowledged(1 << 0),
            Act::Signal(rig.answer_event()),
        ]
    );
    assert_eq!(rig.answers(), 1);
    assert!(all_acknowledged(&rig));
}

#[test]
fn the_acknowledge_summary_is_the_request_summary_as_read() {
    // The title sets a summary bit that names no group with a real request: the acknowledge
    // summary is the whole word it wrote (PR writes `requests` back as read), so the title sees
    // its request acknowledged; only the real group is answered and recorded.
    const OUTSIDE: u64 = 1 << (u64::BITS - 1);
    assert_eq!(OUTSIDE & GROUP_MASK, 0);
    let mut rig = Rig::bound();
    rig.increment(SP);
    rig.ask(&[SP]);
    let request = rig.writable_word(u64::from(GROUP_REQUEST_OFFSET)) | OUTSIDE;
    rig.write_writable(u64::from(GROUP_REQUEST_OFFSET), request);
    rig.signal();
    let acts = rig.turn(T0);
    assert_eq!(rig.read_only_word(u64::from(GROUP_ACK_OFFSET)), request);
    assert!(acts.contains(&acknowledged(1 << 0)));
    assert!(all_acknowledged(&rig));
    assert_eq!(rig.answers(), 1);
}

#[test]
fn each_written_value_is_recorded_after_its_write() {
    let mut rig = Rig::bound();
    rig.increment(SP);
    rig.increment(SP2);
    rig.ask(&[SP, SP2]);
    rig.signal();
    // The second value's write is refused: only the first is recorded, after its write, and the
    // stopped answer records no acknowledge note (it wrote no acknowledge word).
    rig.writes_left = Some(1);
    let acts = rig.turn(T0);
    assert_eq!(acts, [value_write(&rig, SP), answered(SP, 1)]);
    // What the page shows follows what was written.
    assert_eq!(rig.page().shown(id(SP)), 1);
    assert_eq!(rig.page().shown(id(SP2)), UNALLOCATED);
}

#[test]
fn a_closed_transfer_memory_is_refused_by_the_kernel() {
    let mut rig = shown();
    assert!(rig.kernel.end_transfer_memory(rig.read_only));
    rig.ask(&[SP]);
    rig.signal();
    assert_eq!(rig.turn(T0), []);
    assert_eq!(rig.page().held(), None);
}

/// A [`TimerHost`] offering only what every host must: the transfer-memory calls take their
/// defaults.
struct Bare<'a>(PageHost<'a>);

impl TimerHost for Bare<'_> {
    fn record(&mut self, kind: EventKind) {
        self.0.record(kind);
    }

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.0.unlock_device_share(token)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.0.signal_event(event)
    }
}

#[test]
fn a_host_without_transfer_memory_answers_nothing() {
    let mut rig = Rig::bound();
    rig.increment(SP);
    rig.ask(&[SP]);
    rig.signal();
    let mut host = Bare(PageHost {
        kernel: &mut rig.kernel,
        acts: Vec::new(),
        refuse_reads: false,
        writes_left: None,
    });
    rig.shared.drain(&mut host);
    rig.shared.tick_page(&mut host, T0);
    assert_eq!(host.0.acts, []);
    let mut out = [0; 8];
    assert_eq!(
        host.read_transfer_memory(rig.read_only, 0, &mut out),
        Err(KernelError::NotImplemented)
    );
    assert_eq!(
        host.write_transfer_memory(rig.read_only, 0, &out),
        Err(KernelError::NotImplemented)
    );
    drop(host);
    assert_eq!(rig.answers(), 0);
    assert_eq!(rig.page().held(), None);
    assert_eq!(rig.value(SP), UNALLOCATED);
}

#[test]
fn only_the_bound_page_s_request_event_marks_it_signalled() {
    // Before 204 nothing is marked.
    let rig = Rig::new();
    rig.shared.queue().push(Completion::GuestSignalled {
        event: object(0x77, 1),
    });
    let mut rig = rig;
    rig.turn(T0);
    assert!(!rig.page().is_signalled());
    // Bound: another event's signal is not the page's.
    let mut rig = Rig::bound();
    rig.ask(&[SP]);
    rig.shared.queue().push(Completion::GuestSignalled {
        event: rig.page().request_event().unwrap().readable,
    });
    assert_eq!(rig.turn(T0), []);
    assert_eq!(rig.answers(), 0);
    // The drain alone marks it; the page turn reads and clears the mark.
    rig.signal();
    let mut host = PageHost {
        kernel: &mut rig.kernel,
        acts: Vec::new(),
        refuse_reads: false,
        writes_left: None,
    };
    rig.shared.drain(&mut host);
    drop(host);
    assert!(rig.page().is_signalled());
    rig.turn(T0);
    assert!(!rig.page().is_signalled());
    assert_eq!(rig.answers(), 1);
}

#[test]
fn an_unbound_page_with_its_events_is_not_marked_signalled() {
    // A page whose events were made but which is not bound (a 204 that failed after making
    // them): the title's signal of the request event is not a request yet.
    let mut rig = Rig::new();
    let request = RequestEvent {
        writable: object(0x51, 2),
        readable: object(0x52, 2),
    };
    let page = Page {
        request: Some(request),
        answer: Some(object(0x53, 2)),
        ..Page::default()
    };
    rig.shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    rig.shared.queue().push(Completion::GuestSignalled {
        event: request.writable,
    });
    assert_eq!(rig.turn(T0), []);
    assert!(!rig.page().is_signalled());
    assert_eq!(rig.shared.save_page().map(|_| ()), Ok(()));
}

// ----- Snapshots ------------------------------------------------------------------------------------

/// What [`Accepting`] attached: the service, the event and the hook.
type Attached = (ServiceId, ObjectId, Arc<dyn EventHook>);

/// A [`Rehook`] standing in for the kernel's `Dispatcher::reattach_event_hook`: it holds the
/// writable ends of hooked events each service made, takes a re-attach only for one of those
/// (the kernel's [`KernelError::InvalidHandle`] otherwise, as it answers), and keeps what it
/// attached. It does not tell a readable end from an unknown id, which the kernel does by the
/// object's kind; both are refused alike. Parity with the kernel:
/// `the_fake_kernel_refuses_a_rehook_as_the_real_one`.
#[derive(Default)]
struct Accepting {
    /// The events it holds, by service; empty means it takes any.
    made: BTreeSet<(ServiceId, ObjectId)>,
    attached: Mutex<Vec<Attached>>,
}

impl Rehook for Accepting {
    fn reattach(
        &self,
        service: ServiceId,
        event: ObjectId,
        hook: Arc<dyn EventHook>,
    ) -> SvcResult<()> {
        if !self.made.is_empty() && !self.made.contains(&(service, event)) {
            return Err(KernelError::InvalidHandle);
        }
        self.attached
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((service, event, hook));
        Ok(())
    }
}

/// A snapshot holding `page`, as `save_page` lays one out.
fn snapshot_of(page: &Page) -> Vec<u8> {
    let mut bytes = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    bytes.extend(postcard::to_allocvec(page).unwrap());
    bytes
}

/// A page in every field: distinct non-zero objects, a value shown, a hold, a signal waiting.
fn full_page() -> Page {
    let mut shown = CounterExtension::default();
    shown.observe(id(1), Some(0x81));
    Page {
        request: Some(RequestEvent {
            writable: object(0x11, 1),
            readable: object(0x12, 2),
        }),
        answer: Some(object(0x13, 3)),
        binding: Some(Binding {
            writable: object(0x14, 4),
            read_only: object(0x15, 5),
        }),
        shown,
        held: Some(Hold {
            ids: Ids::of([id(1), id(64), id(191)]),
            since: HostNanos(0x16),
        }),
        signalled: true,
    }
}

/// Watches each syncpoint `page`'s hold asks about once, as the hold that made it would have.
fn watch_hold(shared: &Shared, page: &Page) {
    for sp in page
        .held()
        .map_or_else(Ids::default, |held| held.ids)
        .iter()
    {
        shared.lock().watch(sp).unwrap();
    }
}

#[test]
fn the_page_snapshot_layout_is_pinned() {
    let shared = fresh_shared();
    let page = full_page();
    watch_hold(&shared, &page);
    shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    // The magic and version 1.
    let mut expected = vec![b'S', b'Q', b'P', b'G', 0x01];
    // The request event (some: its two ends, each index and generation), the answer event, the
    // binding.
    expected.extend([0x01, 0x11, 0x01, 0x12, 0x02]);
    expected.extend([0x01, 0x13, 0x03]);
    expected.extend([0x01, 0x14, 0x04, 0x15, 0x05]);
    // What is shown: 192 values, syncpoint 1 at 0x81 (a two-byte varint), the rest all ones.
    expected.extend([0xC0, 0x01]);
    let all_ones = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
    for sp in 0..SYNCPOINT_COUNT {
        if sp == 1 {
            expected.extend([0x81, 0x01]);
        } else {
            expected.extend(all_ones);
        }
    }
    // The hold: its three words (1 << 1; 1 << 0; 1 << 63), its time; then the signal.
    expected.push(0x01);
    expected.push(0x02);
    expected.push(0x01);
    expected.extend([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    expected.push(0x16);
    expected.push(0x01);
    assert_eq!(shared.save_page().unwrap(), expected);
}

#[test]
fn a_restored_page_is_used_as_saved() {
    // Save a page holding a request, restore it into fresh state, and drive the restored one:
    // a change of the held counter answers on it.
    let mut rig = shown();
    rig.ask(&[SP]);
    rig.signal();
    rig.turn(T0);
    let saved_shared = rig.shared.save().unwrap();
    let saved_page = rig.shared.save_page().unwrap();
    let syncpoints = rig.shared.syncpoints().save().unwrap();
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::restore(&syncpoints, completions).unwrap());
    let restored = Arc::new(Shared::restore(&saved_shared, syncpoints, queue, GSV).unwrap());
    let request = rig.request_event();
    let kernel = Accepting {
        made: [(ROOT, request)].into(),
        ..Accepting::default()
    };
    restored.restore_page(&saved_page, &kernel, ROOT).unwrap();
    assert_eq!(restored.lock().page(), &rig.page());
    // The hook re-attached is the restored queue: the title's signal reaches it.
    let attached = kernel.attached.lock().unwrap();
    assert_eq!(attached.len(), 1);
    assert_eq!((attached[0].0, attached[0].1), (ROOT, request));
    attached[0].2.signalled(request);
    drop(attached);
    assert_eq!(
        restored.queue().take_all(),
        [Completion::GuestSignalled { event: request }]
    );
    // Swap the rig onto the restored state and carry on.
    rig.shared = restored;
    rig.increment(SP);
    rig.turn(HostNanos(T0.0 + 1));
    assert_eq!(rig.answers(), 2);
    assert_eq!(rig.value(SP), 1);
}

/// Restores `bytes` into fresh state; the error.
fn restore_error(bytes: &[u8]) -> PageSnapshotError {
    fresh_shared()
        .restore_page(bytes, &Accepting::default(), ROOT)
        .unwrap_err()
}

#[test]
fn a_snapshot_with_a_bad_header_or_trailing_bytes_is_refused() {
    let good = snapshot_of(&Page::default());
    let mut magic = good.clone();
    magic[0] = b'X';
    assert_eq!(restore_error(&magic), PageSnapshotError::BadMagic);
    let mut version = good.clone();
    version[4] = 2;
    assert_eq!(
        restore_error(&version),
        PageSnapshotError::UnsupportedVersion { found: 2 }
    );
    // A bad magic and a bad version at once: the magic first.
    let mut both = version.clone();
    both[0] = b'X';
    assert_eq!(restore_error(&both), PageSnapshotError::BadMagic);
    assert!(matches!(
        restore_error(&good[..good.len() - 1]),
        PageSnapshotError::Decode(_)
    ));
    let mut trailing = good.clone();
    trailing.push(0);
    assert_eq!(
        restore_error(&trailing),
        PageSnapshotError::TrailingBytes { count: 1 }
    );
    assert_eq!(
        PageSnapshotError::UnsupportedVersion { found: 2 }.to_string(),
        "syncpoint page snapshot layout version 2 is not supported (this build reads 1)"
    );
}

/// Each relation a page snapshot breaks, the page breaking it, and the error.
fn broken_pages() -> Vec<(Page, PageStateError)> {
    let bound = full_page();
    let unbound = Page {
        binding: None,
        held: None,
        signalled: false,
        shown: CounterExtension::default(),
        ..full_page()
    };
    let mut shown = CounterExtension::default();
    shown.observe(id(7), Some(1));
    vec![
        (
            Page {
                request: Some(RequestEvent {
                    writable: object(0x11, 1),
                    readable: object(0x11, 1),
                }),
                ..bound.clone()
            },
            PageStateError::RequestEnds,
        ),
        (
            Page {
                answer: Some(object(0x12, 2)),
                ..bound.clone()
            },
            PageStateError::SharedEvent,
        ),
        (
            Page {
                answer: None,
                ..bound.clone()
            },
            PageStateError::BoundWithoutEvents,
        ),
        (
            Page {
                request: None,
                ..bound.clone()
            },
            PageStateError::BoundWithoutEvents,
        ),
        (
            Page {
                binding: Some(Binding {
                    writable: object(0x14, 4),
                    read_only: object(0x14, 4),
                }),
                ..bound.clone()
            },
            PageStateError::SameMemory,
        ),
        (
            Page {
                shown,
                ..unbound.clone()
            },
            PageStateError::ShownWhileUnbound { id: 7 },
        ),
        (
            Page {
                held: bound.held,
                ..unbound.clone()
            },
            PageStateError::HeldWhileUnbound,
        ),
        (
            Page {
                signalled: true,
                ..unbound
            },
            PageStateError::SignalledWhileUnbound,
        ),
        (
            Page {
                held: Some(Hold {
                    ids: Ids::default(),
                    since: HostNanos(1),
                }),
                ..bound
            },
            PageStateError::EmptyHold,
        ),
    ]
}

#[test]
fn a_page_whose_fields_do_not_fit_together_is_refused() {
    for (page, error) in broken_pages() {
        let shared = fresh_shared();
        watch_hold(&shared, &page);
        assert_eq!(
            shared
                .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
                .unwrap_err(),
            PageSnapshotError::State(error),
            "{error}"
        );
        assert_eq!(shared.lock().page(), &Page::default());
    }
    // Each edge accepted: an unbound page with its events, a bound page with no hold.
    for page in [
        Page {
            binding: None,
            held: None,
            signalled: false,
            shown: CounterExtension::default(),
            ..full_page()
        },
        Page {
            held: None,
            ..full_page()
        },
        Page::default(),
    ] {
        fresh_shared()
            .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
            .unwrap();
    }
    assert_eq!(
        PageSnapshotError::State(PageStateError::ShownWhileUnbound { id: 7 }).to_string(),
        "the syncpoint page snapshot: an unbound page shows a value for syncpoint 7"
    );
}

#[test]
fn the_relations_are_checked_in_the_order_of_their_errors() {
    // Every pair of faults at once: the earlier variant is the answer. The same object for both
    // request ends and for both memories: the request ends first.
    let page = Page {
        request: Some(RequestEvent {
            writable: object(0x11, 1),
            readable: object(0x11, 1),
        }),
        binding: Some(Binding {
            writable: object(0x14, 4),
            read_only: object(0x14, 4),
        }),
        ..full_page()
    };
    let shared = fresh_shared();
    watch_hold(&shared, &page);
    assert_eq!(
        shared
            .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
            .unwrap_err(),
        PageSnapshotError::State(PageStateError::RequestEnds)
    );
    // A wrong watcher count and a broken relation: the relation first.
    let page = Page {
        answer: Some(object(0x12, 2)),
        ..full_page()
    };
    assert_eq!(
        restore_error(&snapshot_of(&page)),
        PageSnapshotError::State(PageStateError::SharedEvent)
    );
    // A wrong watcher count and a kernel that refuses the hook: the count first.
    let refusing = Accepting {
        made: [(ROOT, object(0x99, 9))].into(),
        ..Accepting::default()
    };
    assert!(matches!(
        fresh_shared()
            .restore_page(&snapshot_of(&full_page()), &refusing, ROOT)
            .unwrap_err(),
        PageSnapshotError::Watchers { .. }
    ));
}

#[test]
fn each_waiter_count_must_be_the_waits_plus_the_hold_s_watch() {
    let page = full_page();
    // One watch too few on the first held syncpoint, and one too many on a syncpoint not held.
    let shared = fresh_shared();
    shared.lock().watch(id(64)).unwrap();
    shared.lock().watch(id(191)).unwrap();
    assert_eq!(
        shared
            .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
            .unwrap_err(),
        PageSnapshotError::Watchers {
            id: 1,
            watchers: 0,
            needed: 1,
        }
    );
    let shared = fresh_shared();
    watch_hold(&shared, &page);
    shared.lock().watch(id(2)).unwrap();
    assert_eq!(
        shared
            .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
            .unwrap_err(),
        PageSnapshotError::Watchers {
            id: 2,
            watchers: 1,
            needed: 0,
        }
    );
    // A watch held by an armed wait counts as the wait's, not the hold's.
    let shared = fresh_shared();
    watch_hold(&shared, &page);
    let wait = crate::syncpt::waiters::Wait::new(
        id(1),
        5,
        starquake_gpu::syncpoint::CompareMode::Wrapping32,
    )
    .unwrap();
    shared.lock().arm(object(0x40, 1), wait).unwrap();
    shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    assert_eq!(
        PageSnapshotError::Watchers {
            id: 2,
            watchers: 1,
            needed: 0,
        }
        .to_string(),
        "syncpoint 2 has 1 waiters, but its armed waits and the page's hold need 0"
    );
}

#[test]
fn a_hook_the_kernel_refuses_fails_the_restore_and_leaves_the_page() {
    let shared = fresh_shared();
    let page = Page {
        held: None,
        ..full_page()
    };
    let refusing = Accepting {
        made: [(ROOT, object(0x99, 9))].into(),
        ..Accepting::default()
    };
    // The right event, named for another service; then another event for the right service.
    for (service, kernel) in [
        (
            ServiceId(ROOT.0 + 1),
            Accepting {
                made: [(ROOT, object(0x11, 1))].into(),
                ..Accepting::default()
            },
        ),
        (ROOT, refusing),
    ] {
        assert_eq!(
            shared
                .restore_page(&snapshot_of(&page), &kernel, service)
                .unwrap_err(),
            PageSnapshotError::Rehook(KernelError::InvalidHandle)
        );
        assert_eq!(shared.lock().page(), &Page::default());
    }
    // A page with no request event asks the kernel nothing.
    let kernel = Accepting {
        made: [(ROOT, object(0x99, 9))].into(),
        ..Accepting::default()
    };
    shared
        .restore_page(&snapshot_of(&Page::default()), &kernel, ROOT)
        .unwrap();
    assert!(kernel.attached.lock().unwrap().is_empty());
}

#[test]
fn saving_runs_the_restore_s_checks() {
    let shared = fresh_shared();
    let page = full_page();
    watch_hold(&shared, &page);
    shared
        .restore_page(&snapshot_of(&page), &Accepting::default(), ROOT)
        .unwrap();
    // A watch taken behind the page's back breaks the count the snapshot would carry.
    shared.lock().watch(id(5)).unwrap();
    assert_eq!(
        shared.save_page().unwrap_err(),
        PageSnapshotError::Watchers {
            id: 5,
            watchers: 1,
            needed: 0,
        }
    );
}

#[test]
fn the_fake_kernel_refuses_a_rehook_as_the_real_one() {
    // The kernel's dispatcher has no server for the root and so made no event: it refuses, as
    // the fake holding events for another service does.
    let (dispatcher, _) = crate::completion::testing::real_kernel();
    let fake = Accepting {
        made: [(ServiceId(ROOT.0 + 1), object(0x11, 1))].into(),
        ..Accepting::default()
    };
    let page = Page {
        held: None,
        ..full_page()
    };
    let real = fresh_shared()
        .restore_page(&snapshot_of(&page), &dispatcher, ROOT)
        .unwrap_err();
    let faked = fresh_shared()
        .restore_page(&snapshot_of(&page), &fake, ROOT)
        .unwrap_err();
    assert_eq!(real, faked);
    assert_eq!(real, PageSnapshotError::Rehook(KernelError::InvalidHandle));
}
