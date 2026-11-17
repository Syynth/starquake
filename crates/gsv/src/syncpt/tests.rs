//! The syncpt session driven through raw TIPC messages, over the services framework's fake
//! kernel, with gsv's shared state behind it and the timer thread's drain played by the test.
//!
//! Requests are laid out by hand, word by word, from the layouts PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` and
//! `graphics_supervisor_protocol.h` @ e5fb3b97): each handler compares the two header words
//! with literals (`ArmWait` through `DecodeSyncpointWait`: `packet[0] != 0xd9 || packet[1] !=
//! 5`; `GetEvent`: `packet[0] == 0x26 && packet[1] == 1`), which the tests write as they are.
//! 0 is NX2's (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `Handshake`: the token in
//! words 2 and 3). A `SyncCounterId` is written as the bitwise NOT of the syncpoint id, as both
//! references read it.

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_gpu::interface::{Completions, SYNCPOINT_COUNT};
use starquake_gpu::syncpoint::{CompareMode, Syncpoints};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::handle::Handle;
use starquake_kernel::ipc::EventHook;
use starquake_kernel::memory::{DeviceShareToken, SharedPages, SvcPermission};
use starquake_kernel::result::{ResultCode, SvcResult};
use starquake_kernel::slab::SlabId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel, UNKNOWN_COMMAND_ID};
use starquake_trace::{EventKind, ServiceId};

use super::waiters::{MODE_WIDE_64, MODE_WRAPPING_32};
use super::*;
use crate::completion::CompletionQueue;
use crate::note::Note;
use crate::shared::TimerHost;
use crate::testing::{STALE, buffer_of, copied, halves, le, result_only, success, tipc_type};

/// The number the tests' syncpt server is known by.
const SYNCPT: ServiceId = ServiceId(0x0658);
/// The number gsv records its notes under: not the server's, so mixing them up fails.
const GSV: ServiceId = ServiceId(0x0657);

/// A syncpoint no other test value collides with, below the reserved ids' range and not
/// allocated by anything.
const SP: u8 = 40;

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

/// The `SyncCounterId` of syncpoint `sp`: its bitwise NOT (NX2 `QueryCounter`, PR
/// `DecodeSyncpointWait`: `~packet[3]`).
fn counter(sp: u8) -> u32 {
    !u32::from(sp)
}

// ----- Requests, word by word -------------------------------------------------------------------

/// `Handshake` (0): NX2 reads the token from words 2 and 3.
fn handshake(token: u64) -> Vec<u32> {
    let [low, high] = halves(token);
    vec![0x10, 2, low, high]
}

/// `GetMaximumEventCount` (1): PR `packet[0] == 0x11 && packet[1] == 0`.
fn get_maximum_event_count() -> Vec<u32> {
    vec![0x11, 0]
}

/// `SetEventCount` (21): PR `packet[0] == 0x25 && packet[1] == 1`, the count at word 2.
fn set_event_count(count: u32) -> Vec<u32> {
    vec![0x25, 1, count]
}

/// `GetEvent` (22): PR `packet[0] == 0x26 && packet[1] == 1`, the slot at word 2.
fn get_event(slot: u32) -> Vec<u32> {
    vec![0x26, 1, slot]
}

/// `QueryCounter` (103): PR `packet[0] == 0x77 && packet[1] == 1`, the `SyncCounterId` at
/// word 2.
fn query_counter(counter: u32) -> Vec<u32> {
    vec![0x77, 1, counter]
}

/// `ArmWait` (201): PR `DecodeSyncpointWait`: `packet[0] == 0xd9`, `packet[1] == 5`, the slot
/// at word 2, the `SyncCounterId` at 3, the threshold at 4 (low) and 5 (high), the mode at 6.
fn arm_wait(slot: u32, counter: u32, threshold: u64, mode: u32) -> Vec<u32> {
    let [low, high] = halves(threshold);
    vec![0xD9, 5, slot, counter, low, high, mode]
}

/// `CancelWait` (202): PR `packet[0] == 0xda && packet[1] == 1`, the slot at word 2.
fn cancel_wait(slot: u32) -> Vec<u32> {
    vec![0xDA, 1, slot]
}

#[test]
fn the_reference_header_words_are_the_command_ids() {
    // Each literal type word above is TIPC's offset plus the command id it stands for.
    for (words, command) in [
        (handshake(0), HANDSHAKE),
        (get_maximum_event_count(), GET_MAXIMUM_EVENT_COUNT),
        (set_event_count(0), SET_EVENT_COUNT),
        (get_event(0), GET_EVENT),
        (query_counter(0), QUERY_COUNTER),
        (arm_wait(0, 0, 0, 0), ARM_WAIT),
        (cancel_wait(0), CANCEL_WAIT),
    ] {
        assert_eq!(words[0], tipc_type(command), "{command}");
        // The data word count is the words after the header.
        assert_eq!(usize::try_from(words[1]).unwrap(), words.len() - 2);
    }
}

// ----- The rig ------------------------------------------------------------------------------------

/// A [`TimerHost`] that signals through the fake kernel, as the wiring's does through the real
/// one, and keeps what was recorded.
struct KernelHost<'a> {
    kernel: &'a mut FakeKernel,
    recorded: Vec<EventKind>,
}

impl TimerHost for KernelHost<'_> {
    fn record(&mut self, kind: EventKind) {
        self.recorded.push(kind);
    }

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        Kernel::signal_event(self.kernel, event)
    }
}

/// gsv's shared state over fresh syncpoints and a fresh queue.
fn fresh_shared() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Arc::new(Shared::new(syncpoints, queue, GSV))
}

/// A syncpt server over gsv's shared state, a kernel, and one session on it.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    shared: Arc<Shared>,
}

impl Rig {
    fn with(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(SYNCPT);
        let session = kernel.new_object();
        let shared = fresh_shared();
        let mut server = syncpt_server(policy, Arc::clone(&shared)).with_service_id(SYNCPT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            shared,
        }
    }

    fn new() -> Self {
        Self::with(UnknownPolicy::Strict)
    }

    /// A rig whose session has every slot (21 with the maximum).
    fn counted() -> Self {
        let mut rig = Self::new();
        assert_eq!(
            rig.send(&set_event_count(MAXIMUM_EVENT_COUNT)),
            success(SET_EVENT_COUNT, &[], &[])
        );
        rig
    }

    /// Sends the message `words` lays out on `session`; the message buffer afterwards.
    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        self.send_on(self.session, words)
    }

    /// The event 22 hands out for `slot`.
    fn event(&mut self, slot: u32) -> ObjectId {
        let reply = self.send(&get_event(slot));
        let handles = copied(&reply);
        assert_eq!(reply, success(GET_EVENT, &handles, &[]), "slot {slot}");
        self.kernel
            .resolve_handle(Handle::from_raw(handles[0]))
            .unwrap()
    }

    /// Plays the timer thread's turn; what it recorded.
    fn drain(&mut self) -> Vec<EventKind> {
        let mut host = KernelHost {
            kernel: &mut self.kernel,
            recorded: Vec::new(),
        };
        self.shared.drain(&mut host);
        host.recorded
    }

    fn signalled(&self, event: ObjectId) -> bool {
        self.kernel.is_signalled(event).unwrap()
    }

    fn signals(&self, event: ObjectId) -> u32 {
        self.kernel.signals(event).unwrap()
    }

    fn wait(&self, event: ObjectId) -> Option<Wait> {
        self.shared.lock().waiters().get(event)
    }

    fn watchers(&self, sp: u8) -> u32 {
        self.shared.lock().watchers(id(sp))
    }

    fn increment(&self, sp: u8, times: u32) {
        for _ in 0..times {
            self.shared.syncpoints().increment(id(sp));
        }
    }
}

fn ok(command: u32) -> Vec<u8> {
    success(command, &[], &[])
}

fn refused(command: u32, error: KernelError) -> Vec<u8> {
    result_only(command, error.code())
}

fn note(sp: u8, value: u32) -> EventKind {
    Note::SyncpointObserved { id: id(sp), value }.event(GSV)
}

// ----- 0, 1 and 21 ---------------------------------------------------------------------------------

#[test]
fn the_handshake_echoes_the_token() {
    let mut rig = Rig::new();
    let token = 0x0123_4567_89AB_CDEF;
    let [low, high] = halves(token);
    assert_eq!(
        rig.send(&handshake(token)),
        success(HANDSHAKE, &[], &[low, high])
    );
}

#[test]
fn get_maximum_event_count_answers_63() {
    // PR `MaximumEventCount = 63`; NX2 answers 32 (a conflict, decision 8's larger answer).
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&get_maximum_event_count()),
        success(GET_MAXIMUM_EVENT_COUNT, &[], &[63])
    );
}

#[test]
fn set_event_count_refuses_more_than_the_maximum() {
    let mut rig = Rig::new();
    assert_eq!(rig.send(&set_event_count(4)), ok(SET_EVENT_COUNT));
    assert_eq!(
        rig.send(&set_event_count(MAXIMUM_EVENT_COUNT + 1)),
        refused(SET_EVENT_COUNT, KernelError::InvalidCombination)
    );
    // Nothing changed: slot 4 is still out, slot 3 still in.
    assert_eq!(
        rig.send(&get_event(4)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
    rig.event(3);
    // The maximum itself is accepted, and opens the last slot.
    assert_eq!(
        rig.send(&set_event_count(MAXIMUM_EVENT_COUNT)),
        ok(SET_EVENT_COUNT)
    );
    rig.event(MAXIMUM_EVENT_COUNT - 1);
}

#[test]
fn set_event_count_clears_the_slots_from_the_count_up() {
    let mut rig = Rig::counted();
    let count = 10;
    let slots = [0, count - 1, count, MAXIMUM_EVENT_COUNT - 1];
    let events = slots.map(|slot| rig.event(slot));
    for (slot, event) in slots.into_iter().zip(events) {
        // An unreached wait on each, its event signalled behind it.
        assert_eq!(rig.send(&arm_wait(slot, counter(SP), 5, 0)), ok(ARM_WAIT));
        Kernel::signal_event(&mut rig.kernel, event).unwrap();
    }
    assert_eq!(rig.watchers(SP), 4);
    assert_eq!(rig.send(&set_event_count(count)), ok(SET_EVENT_COUNT));
    // Below the count: unchanged.
    for event in &events[..2] {
        assert!(rig.wait(*event).is_some());
        assert!(rig.signalled(*event));
    }
    // From the count up: no wait, not signalled.
    for event in &events[2..] {
        assert_eq!(rig.wait(*event), None);
        assert!(!rig.signalled(*event));
    }
    assert_eq!(rig.watchers(SP), 2);
    // And the slot at the count is now out of range.
    assert_eq!(
        rig.send(&get_event(count)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
    // Raising the count again gives the slot back with the same event, still clear.
    assert_eq!(
        rig.send(&set_event_count(MAXIMUM_EVENT_COUNT)),
        ok(SET_EVENT_COUNT)
    );
    assert_eq!(rig.event(count), events[2]);
    assert!(!rig.signalled(events[2]));
}

// ----- 22 -----------------------------------------------------------------------------------------

#[test]
fn get_event_hands_out_the_slot_s_event_copied() {
    let mut rig = Rig::counted();
    let reply = rig.send(&get_event(5));
    // One copied handle, no data (the copy list: `HANDLE_KIND`).
    let handles = copied(&reply);
    assert_eq!(handles.len(), 1);
    assert_eq!(reply, success(GET_EVENT, &handles, &[]));
    let first = rig
        .kernel
        .resolve_handle(Handle::from_raw(handles[0]))
        .unwrap();
    // The same slot hands out the same event; another slot another one.
    assert_eq!(rig.event(5), first);
    assert_ne!(rig.event(6), first);
    assert!(!rig.signalled(first));
}

#[test]
fn a_slot_at_the_count_is_refused_and_one_below_is_not() {
    // A session starts with a count of 0: even slot 0 is out.
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&get_event(0)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
    let count = 5;
    rig.send(&set_event_count(count));
    for slot in [count, u32::MAX] {
        assert_eq!(
            rig.send(&get_event(slot)),
            refused(GET_EVENT, KernelError::InvalidArgument)
        );
        assert_eq!(
            rig.send(&arm_wait(slot, counter(SP), 1, 0)),
            refused(ARM_WAIT, KernelError::InvalidArgument)
        );
        assert_eq!(
            rig.send(&cancel_wait(slot)),
            refused(CANCEL_WAIT, KernelError::InvalidArgument)
        );
    }
    // Nothing was armed or made by the refusals.
    assert!(rig.shared.lock().waiters().is_empty());
    let below = count - 1;
    assert_eq!(rig.send(&arm_wait(below, counter(SP), 1, 0)), ok(ARM_WAIT));
    assert_eq!(rig.send(&cancel_wait(below)), ok(CANCEL_WAIT));
    rig.event(below);
}

// ----- 103 ----------------------------------------------------------------------------------------

#[test]
fn query_counter_reads_any_id() {
    let mut rig = Rig::new();
    // Neither id is allocated; each reads its own counter, and the first and last ids too.
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    rig.increment(SP, 3);
    rig.increment(SP + 1, 5);
    rig.increment(0, 7);
    rig.increment(last, 9);
    for (sp, value) in [(SP, 3), (SP + 1, 5), (0, 7), (last, 9)] {
        assert_eq!(
            rig.send(&query_counter(counter(sp))),
            success(QUERY_COUNTER, &[], &[value, 0, 0]),
            "{sp}"
        );
    }
    // An allocated id reads the same way.
    let allocated = rig.shared.syncpoints().allocate().unwrap();
    rig.increment(allocated.get(), 2);
    assert_eq!(
        rig.send(&query_counter(counter(allocated.get()))),
        success(QUERY_COUNTER, &[], &[2, 0, 0])
    );
}

#[test]
fn a_counter_id_naming_no_syncpoint_reads_zero() {
    let mut rig = Rig::new();
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    rig.increment(last, 4);
    // The last id reads; one past it, and the id a zero word names, read 0.
    assert_eq!(
        rig.send(&query_counter(counter(last))),
        success(QUERY_COUNTER, &[], &[4, 0, 0])
    );
    for word in [!SYNCPOINT_COUNT, 0, u32::from(u8::MAX)] {
        assert_eq!(
            rig.send(&query_counter(word)),
            success(QUERY_COUNTER, &[], &[0, 0, 0]),
            "{word:#x}"
        );
    }
}

// ----- 201 ----------------------------------------------------------------------------------------

#[test]
fn a_wait_on_an_unallocated_id_fires_on_its_counter() {
    // The spec's offline test: the wait fires exactly at the threshold, not one before.
    let mut rig = Rig::counted();
    let event = rig.event(2);
    let threshold = 2;
    assert_eq!(
        rig.send(&arm_wait(2, counter(SP), threshold, 0)),
        ok(ARM_WAIT)
    );
    assert_eq!(rig.watchers(SP), 1);
    rig.increment(SP, 1);
    assert_eq!(rig.drain(), [note(SP, 1)]);
    assert!(!rig.signalled(event));
    rig.increment(SP, 1);
    assert_eq!(rig.drain(), [note(SP, 2)]);
    assert!(rig.signalled(event));
    assert_eq!(rig.signals(event), 1);
    // The wait is gone, and so is its watch: a later increment is not pushed.
    assert_eq!(rig.wait(event), None);
    assert_eq!(rig.watchers(SP), 0);
    rig.increment(SP, 1);
    assert_eq!(rig.drain(), []);
    assert_eq!(rig.signals(event), 1);
}

#[test]
fn a_wide_wait_fires_at_its_threshold_too() {
    let mut rig = Rig::counted();
    let event = rig.event(0);
    rig.increment(SP, 5);
    assert_eq!(rig.send(&arm_wait(0, counter(SP), 7, 1)), ok(ARM_WAIT));
    rig.increment(SP, 1);
    rig.drain();
    assert!(!rig.signalled(event));
    rig.increment(SP, 1);
    rig.drain();
    assert!(rig.signalled(event));
}

#[test]
fn a_wait_reached_when_armed_fires_on_the_next_drain() {
    let mut rig = Rig::counted();
    let event = rig.event(1);
    rig.increment(SP, 5);
    // At the threshold already: not signalled by the handler...
    assert_eq!(rig.send(&arm_wait(1, counter(SP), 5, 0)), ok(ARM_WAIT));
    assert!(!rig.signalled(event));
    assert!(rig.wait(event).is_some());
    // ...but queued, and fired and recorded by the drain.
    assert_eq!(rig.drain(), [note(SP, 5)]);
    assert!(rig.signalled(event));
    assert_eq!(rig.wait(event), None);
    assert_eq!(rig.watchers(SP), 0);
    // One below the threshold: nothing queued, nothing fired.
    let other = rig.event(2);
    assert_eq!(rig.send(&arm_wait(2, counter(SP), 6, 0)), ok(ARM_WAIT));
    assert_eq!(rig.drain(), []);
    assert!(!rig.signalled(other));
}

#[test]
fn arming_clears_a_signalled_slot() {
    // Row 147: a slot left signalled (here by a cancel) is cleared by the next 201.
    let mut rig = Rig::counted();
    let event = rig.event(0);
    assert_eq!(rig.send(&cancel_wait(0)), ok(CANCEL_WAIT));
    assert!(rig.signalled(event));
    assert_eq!(rig.send(&arm_wait(0, counter(SP), 9, 0)), ok(ARM_WAIT));
    assert!(!rig.signalled(event));
    // Another slot's signalled event is left alone.
    let neighbour = rig.event(1);
    Kernel::signal_event(&mut rig.kernel, neighbour).unwrap();
    assert_eq!(rig.send(&arm_wait(0, counter(SP), 9, 0)), ok(ARM_WAIT));
    assert!(rig.signalled(neighbour));
}

#[test]
fn arming_a_slot_again_replaces_its_wait() {
    let mut rig = Rig::counted();
    let event = rig.event(4);
    assert_eq!(rig.send(&arm_wait(4, counter(SP), 10, 0)), ok(ARM_WAIT));
    assert_eq!(rig.send(&arm_wait(4, counter(SP + 1), 3, 1)), ok(ARM_WAIT));
    assert_eq!(rig.watchers(SP), 0);
    assert_eq!(rig.watchers(SP + 1), 1);
    assert_eq!(
        rig.wait(event),
        Some(Wait::new(id(SP + 1), 3, CompareMode::Wide64).unwrap())
    );
    // The old wait's syncpoint reaching its threshold fires nothing.
    rig.increment(SP, 10);
    rig.drain();
    assert!(!rig.signalled(event));
    rig.increment(SP + 1, 3);
    rig.drain();
    assert!(rig.signalled(event));
}

#[test]
fn a_wait_on_no_syncpoint_signals_at_once() {
    let mut rig = Rig::counted();
    let event = rig.event(3);
    // An earlier wait on the slot is taken out.
    assert_eq!(rig.send(&arm_wait(3, counter(SP), 9, 0)), ok(ARM_WAIT));
    let nowhere = !SYNCPOINT_COUNT;
    assert_eq!(rig.send(&arm_wait(3, nowhere, 9, 0)), ok(ARM_WAIT));
    assert!(rig.signalled(event));
    assert_eq!(rig.wait(event), None);
    assert_eq!(rig.watchers(SP), 0);
    // Even with a threshold 201 would refuse for a real syncpoint, and on a slot with no event
    // yet.
    assert_eq!(rig.send(&arm_wait(7, nowhere, u64::MAX, 1)), ok(ARM_WAIT));
    let made = rig.event(7);
    assert!(rig.signalled(made));
    // The last id is a syncpoint: armed, not signalled.
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    let other = rig.event(8);
    assert_eq!(rig.send(&arm_wait(8, counter(last), 9, 0)), ok(ARM_WAIT));
    assert!(!rig.signalled(other));
    assert!(rig.wait(other).is_some());
}

#[test]
fn a_wait_with_an_unknown_mode_is_refused() {
    let mut rig = Rig::counted();
    let event = rig.event(0);
    assert_eq!(
        rig.send(&arm_wait(0, counter(SP), 1, u32::from(MODE_WIDE_64) + 1)),
        refused(ARM_WAIT, KernelError::InvalidCombination)
    );
    assert_eq!(rig.wait(event), None);
    // Only the low byte is the mode (PR `packet[6] & 0xff`).
    for (word, mode) in [
        (0x100, CompareMode::Wrapping32),
        (0xFFFF_FF01, CompareMode::Wide64),
    ] {
        assert_eq!(rig.send(&arm_wait(0, counter(SP), 1, word)), ok(ARM_WAIT));
        assert_eq!(
            rig.wait(event),
            Some(Wait::new(id(SP), 1, mode).unwrap()),
            "{word:#x}"
        );
    }
    assert_eq!(
        rig.send(&arm_wait(0, counter(SP), 1, 0x1FF)),
        refused(ARM_WAIT, KernelError::InvalidCombination)
    );
}

#[test]
fn a_wide_wait_above_any_counter_is_refused() {
    let mut rig = Rig::counted();
    let event = rig.event(0);
    let edge = u64::from(u32::MAX);
    assert_eq!(
        rig.send(&arm_wait(0, counter(SP), edge + 1, u32::from(MODE_WIDE_64))),
        refused(ARM_WAIT, KernelError::NotImplemented)
    );
    assert_eq!(rig.wait(event), None);
    assert_eq!(rig.watchers(SP), 0);
    assert_eq!(
        rig.send(&arm_wait(0, counter(SP), edge, u32::from(MODE_WIDE_64))),
        ok(ARM_WAIT)
    );
    // A wrapping wait takes the same threshold and compares its low half.
    assert_eq!(
        rig.send(&arm_wait(
            1,
            counter(SP),
            edge + 1,
            u32::from(MODE_WRAPPING_32)
        )),
        ok(ARM_WAIT)
    );
}

#[test]
fn arm_checks_run_in_the_documented_order() {
    let mut rig = Rig::new();
    rig.send(&set_event_count(2));
    let bad_mode = u32::from(MODE_WIDE_64) + 1;
    let too_wide = u64::from(u32::MAX) + 1;
    // Mode and slot: the mode.
    assert_eq!(
        rig.send(&arm_wait(2, counter(SP), too_wide, bad_mode)),
        refused(ARM_WAIT, KernelError::InvalidCombination)
    );
    // Slot and no syncpoint: the slot.
    assert_eq!(
        rig.send(&arm_wait(2, !SYNCPOINT_COUNT, 1, 0)),
        refused(ARM_WAIT, KernelError::InvalidArgument)
    );
    // Slot and threshold: the slot.
    assert_eq!(
        rig.send(&arm_wait(2, counter(SP), too_wide, 1)),
        refused(ARM_WAIT, KernelError::InvalidArgument)
    );
    // No syncpoint and threshold: no syncpoint, which signals.
    assert_eq!(
        rig.send(&arm_wait(1, !SYNCPOINT_COUNT, too_wide, 1)),
        ok(ARM_WAIT)
    );
}

#[test]
fn a_short_request_is_refused_and_a_longer_one_is_read() {
    let mut rig = Rig::counted();
    let event = rig.event(0);
    // Without the mode word: refused before anything changes.
    let mut short = arm_wait(0, counter(SP), 1, 0);
    short.pop();
    short[1] = 4;
    Kernel::signal_event(&mut rig.kernel, event).unwrap();
    assert_eq!(
        rig.send(&short),
        result_only(ARM_WAIT, INVALID_CMIF_REQUEST)
    );
    assert_eq!(rig.wait(event), None);
    assert!(rig.signalled(event));
    // With a word more: read as if it were not there.
    let mut long = arm_wait(0, counter(SP), 1, 0);
    long.push(0xDEAD_BEEF);
    long[1] = 6;
    assert_eq!(rig.send(&long), ok(ARM_WAIT));
    assert!(rig.wait(event).is_some());
    // The other commands, one word short.
    for command in [SET_EVENT_COUNT, GET_EVENT, QUERY_COUNTER, CANCEL_WAIT] {
        assert_eq!(
            rig.send(&[tipc_type(command), 0]),
            result_only(command, INVALID_CMIF_REQUEST),
            "{command}"
        );
    }
    assert_eq!(
        rig.send(&[tipc_type(HANDSHAKE), 1, 7]),
        result_only(HANDSHAKE, INVALID_CMIF_REQUEST)
    );
}

proptest! {
    #![proptest_config(Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5_1182),
        ..Config::default()
    })]

    /// 201's fields, laid out by hand with distinct values, come back as the wait armed on the
    /// slot they name: reading any two fields in swapped order arms another wait or another
    /// slot.
    #[test]
    fn arm_wait_reads_its_fields_in_order(
        slot in 0..MAXIMUM_EVENT_COUNT,
        sp in 0..u8::try_from(SYNCPOINT_COUNT).unwrap(),
        threshold in 0..=u64::from(u32::MAX),
        high in any::<u32>(),
        wide in any::<bool>(),
    ) {
        // A wrapping wait takes any high half; a wide one none (it would be refused).
        let (threshold, mode) = if wide {
            (threshold, MODE_WIDE_64)
        } else {
            (threshold | u64::from(high) << u32::BITS, MODE_WRAPPING_32)
        };
        let mut rig = Rig::counted();
        prop_assert_eq!(
            rig.send(&arm_wait(slot, counter(sp), threshold, u32::from(mode))),
            ok(ARM_WAIT)
        );
        let event = rig.event(slot);
        let compare = waiters::compare_mode(mode).unwrap();
        prop_assert_eq!(rig.wait(event), Some(Wait::new(id(sp), threshold, compare).unwrap()));
        prop_assert_eq!(rig.shared.lock().waiters().len(), 1);
    }
}

// ----- 202 ----------------------------------------------------------------------------------------

#[test]
fn cancel_always_signals() {
    let mut rig = Rig::counted();
    let event = rig.event(0);
    // With a wait armed: taken out and signalled.
    assert_eq!(rig.send(&arm_wait(0, counter(SP), 3, 0)), ok(ARM_WAIT));
    assert_eq!(rig.send(&cancel_wait(0)), ok(CANCEL_WAIT));
    assert!(rig.signalled(event));
    assert_eq!(rig.signals(event), 1);
    assert_eq!(rig.wait(event), None);
    assert_eq!(rig.watchers(SP), 0);
    // The cancelled wait's threshold reached later fires nothing more.
    rig.increment(SP, 3);
    rig.drain();
    assert_eq!(rig.signals(event), 1);
    // With nothing armed: signalled all the same.
    assert_eq!(rig.send(&cancel_wait(0)), ok(CANCEL_WAIT));
    assert_eq!(rig.signals(event), 2);
    // A neighbour's wait stays.
    let neighbour = rig.event(1);
    assert_eq!(rig.send(&arm_wait(1, counter(SP), 9, 0)), ok(ARM_WAIT));
    assert_eq!(rig.send(&cancel_wait(0)), ok(CANCEL_WAIT));
    assert!(rig.wait(neighbour).is_some());
    assert!(!rig.signalled(neighbour));
}

#[test]
fn cancel_takes_out_a_wait_already_queued_to_fire() {
    // Reached when armed, so queued; cancelled before the drain: the drain records the value
    // and signals nothing more.
    let mut rig = Rig::counted();
    let event = rig.event(0);
    rig.increment(SP, 2);
    rig.send(&arm_wait(0, counter(SP), 2, 0));
    rig.send(&cancel_wait(0));
    assert_eq!(rig.drain(), [note(SP, 2)]);
    assert_eq!(rig.signals(event), 1);
    // Re-armed for a value not yet reached before the drain: the queued value does not fire it.
    rig.send(&arm_wait(0, counter(SP), 2, 0));
    rig.send(&arm_wait(0, counter(SP), 3, 0));
    rig.drain();
    assert!(!rig.signalled(event));
    assert!(rig.wait(event).is_some());
}

#[test]
fn cancel_makes_and_signals_an_event_no_command_made() {
    let mut rig = Rig::counted();
    assert_eq!(rig.send(&cancel_wait(3)), ok(CANCEL_WAIT));
    let event = rig.event(3);
    assert!(rig.signalled(event));
    assert_eq!(rig.signals(event), 1);
}

// ----- The lock order -------------------------------------------------------------------------------

/// A kernel that answers as the fake does, but refuses to clear the events in `refused` (as the
/// kernel refuses an id it does not have), and notes, for each event clear and signal, whether
/// gsv's shared lock was held.
struct Checking {
    inner: FakeKernel,
    shared: Arc<Shared>,
    calls: Vec<(&'static str, bool)>,
    refused: Vec<ObjectId>,
}

impl Checking {
    fn over(rig: Rig) -> (Self, Server, ObjectId) {
        let kernel = Self {
            inner: rig.kernel,
            shared: rig.shared,
            calls: Vec::new(),
            refused: Vec::new(),
        };
        (kernel, rig.server, rig.session)
    }
}

impl Kernel for Checking {
    fn process_id(&self) -> u64 {
        self.inner.process_id()
    }

    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        self.inner.resolve_handle(handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        self.inner.insert_handle(object)
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        self.inner.create_event()
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.calls.push(("signal", self.shared.lock_is_held()));
        Kernel::signal_event(&mut self.inner, event)
    }

    fn clear_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.calls.push(("clear", self.shared.lock_is_held()));
        if self.refused.contains(&event) {
            return Err(KernelError::InvalidHandle);
        }
        self.inner.clear_event(event)
    }

    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        self.inner.create_shared_memory(size, permission)
    }

    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        self.inner.create_session()
    }

    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        self.inner.connect(service)
    }

    fn create_hooked_event(&mut self, hook: Arc<dyn EventHook>) -> SvcResult<(ObjectId, ObjectId)> {
        self.inner.create_hooked_event(hook)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        self.inner.read_memory(address, out)
    }

    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        self.inner.write_memory(address, data)
    }

    fn record(&mut self, kind: EventKind) {
        self.inner.record(kind);
    }
}

/// Sends `words` on `session` of `server` through `kernel`; the message buffer afterwards.
fn send_through(
    server: &mut Server,
    kernel: &mut Checking,
    session: ObjectId,
    words: &[u32],
) -> Vec<u8> {
    let mut buffer = buffer_of(words);
    server.handle(kernel, session, &mut buffer).unwrap();
    buffer
}

#[test]
fn arm_and_cancel_call_the_kernel_with_the_shared_lock_held() {
    let (mut kernel, mut server, session) = Checking::over(Rig::counted());
    let mut send = |words: &[u32]| send_through(&mut server, &mut kernel, session, words);
    assert_eq!(send(&arm_wait(0, counter(SP), 1, 0)), ok(ARM_WAIT));
    assert_eq!(send(&cancel_wait(0)), ok(CANCEL_WAIT));
    assert_eq!(send(&arm_wait(0, !SYNCPOINT_COUNT, 1, 0)), ok(ARM_WAIT));
    assert_eq!(send(&set_event_count(0)), ok(SET_EVENT_COUNT));
    assert_eq!(
        kernel.calls,
        [
            ("clear", true),
            ("signal", true),
            ("signal", true),
            ("clear", true),
        ]
    );
    // And released after each.
    assert!(!kernel.shared.lock_is_held());
}

#[test]
fn a_refused_clear_still_leaves_every_slot_from_the_count_up_without_a_wait() {
    let (mut kernel, mut server, session) = Checking::over(Rig::counted());
    let mut events = Vec::new();
    for slot in [3, 4, 5] {
        let reply = send_through(&mut server, &mut kernel, session, &get_event(slot));
        let handle = Handle::from_raw(copied(&reply)[0]);
        events.push(kernel.resolve_handle(handle).unwrap());
        let arm = arm_wait(slot, counter(SP), 9, 0);
        assert_eq!(
            send_through(&mut server, &mut kernel, session, &arm),
            ok(ARM_WAIT)
        );
        Kernel::signal_event(&mut kernel.inner, *events.last().unwrap()).unwrap();
    }
    // The kernel refuses the first of the two clears: the reply is its error, and both slots
    // lose their waits, and the second is still cleared.
    kernel.refused.push(events[1]);
    assert_eq!(
        send_through(&mut server, &mut kernel, session, &set_event_count(4)),
        refused(SET_EVENT_COUNT, KernelError::InvalidHandle)
    );
    let shared = Arc::clone(&kernel.shared);
    assert!(shared.lock().waiters().get(events[0]).is_some());
    assert_eq!(shared.lock().waiters().get(events[1]), None);
    assert_eq!(shared.lock().waiters().get(events[2]), None);
    assert_eq!(kernel.inner.is_signalled(events[2]), Some(false));
    assert_eq!(kernel.inner.is_signalled(events[0]), Some(true));
    // The count was set.
    assert_eq!(
        send_through(&mut server, &mut kernel, session, &get_event(4)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
}

// ----- Sessions, recording and snapshots ------------------------------------------------------------

#[test]
fn two_sessions_keep_their_own_counts_and_events() {
    let mut rig = Rig::new();
    let second = rig.kernel.new_object();
    rig.server.accept(second).unwrap();
    rig.send(&set_event_count(2));
    let first = rig.event(1);
    // The second session's count is its own: still 0.
    assert_eq!(
        rig.send_on(second, &get_event(1)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
    rig.send_on(second, &set_event_count(2));
    let theirs = copied(&rig.send_on(second, &get_event(1)))[0];
    let theirs = rig.kernel.resolve_handle(Handle::from_raw(theirs)).unwrap();
    assert_ne!(theirs, first);
    // A cancel on one session's slot signals only its own event.
    rig.send_on(second, &cancel_wait(1));
    assert!(rig.signalled(theirs));
    assert!(!rig.signalled(first));
}

#[test]
fn a_closed_session_s_waits_stay_armed() {
    // The framework tells a session nothing when it closes (module docs, "Where this differs
    // from the references").
    let mut rig = Rig::counted();
    let event = rig.event(0);
    rig.send(&arm_wait(0, counter(SP), 1, 0));
    rig.server.close(rig.session).unwrap();
    assert!(rig.wait(event).is_some());
    rig.increment(SP, 1);
    rig.drain();
    assert!(rig.signalled(event));
}

#[test]
fn a_command_the_table_does_not_list_goes_to_the_policy() {
    // 2 and 104 fall between the table's ids; neither reference has them.
    for command in [2, 104] {
        let mut rig = Rig::new();
        assert_eq!(
            rig.send(&[tipc_type(command), 0]),
            result_only(command, UNKNOWN_COMMAND_ID),
            "{command}"
        );
        let mut rig = Rig::with(UnknownPolicy::AutoStub);
        assert_eq!(
            rig.send(&[tipc_type(command), 0]),
            result_only(command, ResultCode::SUCCESS),
            "{command}"
        );
    }
}

#[test]
fn each_request_is_recorded() {
    let mut rig = Rig::counted();
    rig.send(&arm_wait(0, counter(SP), 1, 9));
    rig.send(&query_counter(counter(SP)));
    let recorded: Vec<_> = rig
        .kernel
        .recorded()
        .into_iter()
        .filter_map(|event| match event.kind {
            EventKind::Ipc {
                service,
                command,
                result,
                ..
            } => Some((service, command, result.0)),
            _ => None,
        })
        .collect();
    assert_eq!(
        recorded,
        [
            (SYNCPT, SET_EVENT_COUNT, ResultCode::SUCCESS.raw()),
            (
                SYNCPT,
                ARM_WAIT,
                KernelError::InvalidCombination.code().raw()
            ),
            (SYNCPT, QUERY_COUNTER, ResultCode::SUCCESS.raw()),
        ]
    );
}

/// postcard's encoding of a session with `count` and `events`, field for field: what
/// `SavedSyncptSession` reads.
fn encoded(count: u32, events: &[Option<ObjectId>]) -> Vec<u8> {
    postcard::to_allocvec(&(count, events.to_vec())).unwrap()
}

#[test]
fn the_session_layout_is_pinned() {
    let mut rig = Rig::counted();
    rig.send(&set_event_count(0x21));
    rig.send(&cancel_wait(0x20));
    rig.send(&cancel_wait(1));
    let saved = rig.server.save().unwrap();
    // The session's bytes: the count, 63 slots (a varint each way), slot 1's event and slot
    // 0x20's (each a `Some` tag, then the index and generation varints of the fake's ids).
    let one = rig.event(1);
    let other = rig.event(0x20);
    assert_eq!((one.index(), one.generation()), (2, 1));
    assert_eq!((other.index(), other.generation()), (1, 1));
    let mut expected = vec![0x21, 0x3F, 0x00, 0x01, 0x02, 0x01];
    expected.extend([0x00; 0x1E]);
    expected.extend([0x01, 0x01, 0x01]);
    expected.extend([0x00; 0x1E]);
    assert_eq!(expected.len(), 6 + 0x1E + 3 + 0x1E);
    assert!(
        saved.windows(expected.len()).any(|w| w == expected),
        "the session's bytes are not in the snapshot"
    );
}

#[test]
fn a_saved_session_is_restored_and_used() {
    let mut rig = Rig::counted();
    rig.send(&set_event_count(5));
    let event = rig.event(4);
    rig.send(&arm_wait(4, counter(SP), 2, 0));
    let saved = rig.server.save().unwrap();
    let mut restored =
        syncpt_server(UnknownPolicy::Strict, Arc::clone(&rig.shared)).with_service_id(SYNCPT);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    // The count came back (5 is out, 4 in), with the slot's event, and the family's handle on
    // gsv's shared state still reaches the armed wait.
    assert_eq!(
        rig.send(&get_event(5)),
        refused(GET_EVENT, KernelError::InvalidArgument)
    );
    assert_eq!(rig.event(4), event);
    rig.increment(SP, 2);
    assert_eq!(
        rig.send(&query_counter(counter(SP))),
        success(QUERY_COUNTER, &[], &[2, 0, 0])
    );
    rig.drain();
    assert!(rig.signalled(event));
}

#[test]
fn a_saved_session_no_command_reaches_is_refused() {
    let none = vec![None; SLOTS];
    let event = ObjectId::from_parts(7, 3);
    let mut twice = none.clone();
    twice[2] = Some(event);
    twice[9] = Some(event);
    let cases = [
        (
            encoded(MAXIMUM_EVENT_COUNT + 1, &none),
            SyncptSessionError::EventCount {
                found: MAXIMUM_EVENT_COUNT + 1,
            },
            "a syncpt session's event count is 64, above the maximum of 63",
        ),
        (
            encoded(0, &none[1..]),
            SyncptSessionError::Slots { found: SLOTS - 1 },
            "a syncpt session holds 62 slots, not 63",
        ),
        (
            encoded(0, &[none.clone(), vec![None]].concat()),
            SyncptSessionError::Slots { found: SLOTS + 1 },
            "a syncpt session holds 64 slots, not 63",
        ),
        (
            encoded(0, &twice),
            SyncptSessionError::SharedEvent {
                first: 2,
                second: 9,
            },
            "syncpt slots 2 and 9 hold the same event",
        ),
    ];
    for (bytes, error, message) in cases {
        let saved =
            SavedSyncptSession::deserialize(&mut postcard::Deserializer::from_bytes(&bytes))
                .unwrap();
        assert_eq!(SyncptSession::try_from(saved), Err(error));
        assert_eq!(error.to_string(), message);
        assert!(postcard::from_bytes::<SyncptSession>(&bytes).is_err());
    }
    // At each edge: the maximum count, and two slots with events that differ only in their
    // generation.
    assert!(postcard::from_bytes::<SyncptSession>(&encoded(MAXIMUM_EVENT_COUNT, &none)).is_ok());
    let mut two = none;
    two[2] = Some(event);
    two[9] = Some(ObjectId::from_parts(7, 4));
    assert!(postcard::from_bytes::<SyncptSession>(&encoded(0, &two)).is_ok());
}

#[test]
fn the_family_saves_nothing_and_keeps_its_handle() {
    let shared = fresh_shared();
    let mut family = SyncptFamily::new(Arc::clone(&shared));
    assert_eq!(family.save_state(), Ok(Vec::new()));
    assert_eq!(family.load_state(&[]), Ok(()));
    assert!(Arc::ptr_eq(family.shared(), &shared));
    assert_eq!(
        family.load_state(&[0]),
        Err(SnapshotError::Decode(
            postcard::Error::DeserializeBadEncoding
        ))
    );
    assert!(Arc::ptr_eq(family.shared(), &shared));
}

#[test]
fn a_fresh_session_has_no_count_and_no_events() {
    let session = SyncptSession::default();
    assert_eq!(session, SyncptSession::new());
    assert_eq!(session.event_count(), 0);
    assert_eq!(session.event(0), None);
    assert_eq!(session.event(u32::MAX), None);
}

#[test]
fn syncpoint_of_is_the_bitwise_not_below_the_count() {
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    assert_eq!(syncpoint_of(!0), Some(id(0)));
    assert_eq!(syncpoint_of(!u32::from(last)), Some(id(last)));
    assert_eq!(syncpoint_of(!SYNCPOINT_COUNT), None);
    assert_eq!(syncpoint_of(u32::from(SP)), None);
    assert_eq!(mode_byte(0xABCD_EF01), 0x01);
    assert_eq!(mode_byte(0xFF), u8::MAX);
}
