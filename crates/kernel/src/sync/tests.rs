//! The synchronisation rows driven through the scheduler directly, with several threads, as the
//! scheduler's own tests drive it: events, wake order, several objects, cancellation, timeouts
//! and late timers, and a waiter being woken onto a core.

use starquake_cpu::GuestRegs;
use starquake_trace::{GuestTicks, HostNanos, SVC_ARG_COUNT};

use super::*;
use crate::handle::{Closed, ObjectRefs};
use crate::id::{ObjectId, ProcessId, SlabId, ThreadId};
use crate::object::{Object, ObjectError, Objects};
use crate::result::{CallResult, KernelError, ResultCode, SvcError};
use crate::sched::{CoreId, CoreSignals, Priority, SchedError, Scheduler, ThreadState};

/// What the fake core loop was asked to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Interrupt(u8),
    Wake(u8),
}

/// A [`CoreSignals`] that records every call.
#[derive(Default)]
struct Fake {
    log: Vec<Signal>,
}

impl CoreSignals for Fake {
    fn interrupt(&mut self, core: CoreId) {
        self.log.push(Signal::Interrupt(core.index()));
    }

    fn wake(&mut self, core: CoreId) {
        self.log.push(Signal::Wake(core.index()));
    }
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// A started thread at `priority` on core `on`.
fn spawn(s: &mut Scheduler, priority: u8, on: u8) -> ThreadId {
    let thread = s
        .create_thread(Priority::new(priority).unwrap(), core(on))
        .unwrap();
    s.start(thread).unwrap();
    thread
}

/// Recomputes every core's choice with `caller` as the calling core, and returns the signals.
fn update(s: &mut Scheduler, caller: Option<u8>) -> Fake {
    let mut fake = Fake::default();
    s.update(GuestTicks(0), caller.map(core), &mut fake)
        .unwrap();
    fake
}

/// Makes core `on` switch to the thread the scheduler chose for it, and returns it.
fn switch(s: &mut Scheduler, on: u8) -> Option<ThreadId> {
    update(s, None);
    s.schedule(core(on), &mut Fake::default()).unwrap()
}

/// Registers that all differ, so a stray write shows.
fn regs(seed: u64) -> GuestRegs {
    let mut regs = GuestRegs::default();
    for (i, x) in (1_u64..).zip(regs.x.iter_mut()) {
        *x = seed.wrapping_mul(0x0101_0101_0101_0101).wrapping_add(i);
    }
    regs
}

/// Who a test wait is for: distinct per `n`, so a swapped origin shows.
fn origin(n: u8) -> WaitOrigin {
    let mut args = [0; SVC_ARG_COUNT];
    args[1] = u64::from(n) * 0x100;
    WaitOrigin {
        thread: starquake_trace::ThreadId(0x7000 + u64::from(n)),
        core: n,
        args,
    }
}

/// An event's two ends, each holding its creator's reference (standing in for a handle).
fn event(s: &mut Scheduler) -> (Event, ObjectId) {
    let (writable, readable) = s.create_event().unwrap();
    let object = s.objects().get(writable);
    let event = match object {
        Ok(Object::Event(event)) => event,
        other => unreachable!("not an event: {other:?}"),
    };
    assert_eq!(event.readable(), readable);
    (event, readable)
}

/// What the svc layer does before it waits: opens a reference to each object.
fn wait(
    s: &mut Scheduler,
    on: u8,
    objects: &[ObjectId],
    timeout: Timeout,
    regs: &GuestRegs,
    who: WaitOrigin,
) -> CallResult<WaitBegun> {
    for object in objects {
        s.objects_mut().open(*object).unwrap();
    }
    s.wait_synchronization(core(on), objects, timeout, regs, who)
}

fn refs(s: &Scheduler, objects: &[ObjectId]) -> Vec<u32> {
    objects
        .iter()
        .map(|object| s.objects().refs(*object))
        .collect()
}

fn word(result: ResultCode) -> u64 {
    u64::from(result.raw())
}

/// The index -1 as `x1` holds it: four bytes, zero-extended.
const MINUS_ONE: u64 = 0xFFFF_FFFF;

/// A scheduler with one thread running on core 0, and that thread.
fn one_running() -> (Scheduler, ThreadId) {
    let mut s = Scheduler::default();
    let thread = spawn(&mut s, 30, 0);
    assert_eq!(switch(&mut s, 0), Some(thread));
    (s, thread)
}

// ----- Events ------------------------------------------------------------------------------

#[test]
fn a_new_event_is_not_signalled_and_has_two_rows() {
    let mut s = Scheduler::default();
    let (writable, readable) = s.create_event().unwrap();
    assert_eq!(s.objects().len(), 2);
    assert_eq!(
        s.objects().get(readable),
        Ok(Object::ReadableEvent(ReadableEvent::default()))
    );
    assert!(!ReadableEvent::default().is_signalled());
    assert_eq!(s.is_signalled(readable), Ok(false));
    assert_eq!(refs(&s, &[writable, readable]), [1, 1]);
}

#[test]
fn signal_reset_and_clear_follow_the_readable_event() {
    let mut s = Scheduler::default();
    let (event, readable) = event(&mut s);
    // Reset before a signal: the readable end's InvalidState.
    assert_eq!(
        s.reset_readable(readable),
        Err(SvcError::Kernel(KernelError::InvalidState))
    );
    // Clear ignores that.
    assert_eq!(s.clear_readable(readable), Ok(()));
    assert_eq!(s.clear_event(event), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(false));

    assert_eq!(s.signal_event(event), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(true));
    // Signalling a signalled event changes nothing.
    assert_eq!(s.signal_readable(readable), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(true));
    assert_eq!(s.reset_readable(readable), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(false));

    s.signal_readable(readable).unwrap();
    assert_eq!(s.clear_event(event), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(false));
    s.signal_readable(readable).unwrap();
    assert_eq!(s.clear_readable(readable), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(false));
}

#[test]
fn signalling_after_the_readable_end_is_closed_succeeds_and_does_nothing() {
    // Eden: `R_SUCCEED_IF(m_readable_event_destroyed)` in `KEvent::Signal` and `Clear`.
    let mut s = Scheduler::default();
    let (event, readable) = event(&mut s);
    assert_eq!(s.objects_mut().close(readable), Ok(Closed::Last));
    assert_eq!(s.signal_event(event), Ok(()));
    assert_eq!(s.clear_event(event), Ok(()));
    assert_eq!(s.objects().len(), 1);
    // The slot is reused by a new readable end: the old event still reaches nothing.
    let other = s.create_readable_event().unwrap();
    assert_eq!(other.index(), readable.index());
    assert_eq!(s.signal_event(event), Ok(()));
    assert_eq!(s.is_signalled(other), Ok(false));
}

#[test]
fn creating_an_event_is_all_or_nothing_when_the_object_table_fills() {
    let mut s = Scheduler::default();
    // Room for the readable end only: it is taken back.
    *s.objects_mut() = Objects::with_max_slots(1);
    assert_eq!(
        s.create_event(),
        Err(SvcError::Kernel(KernelError::OutOfResource))
    );
    assert!(s.objects().is_empty());
    // No room at all.
    *s.objects_mut() = Objects::with_max_slots(0);
    assert_eq!(
        s.create_event(),
        Err(SvcError::Kernel(KernelError::OutOfResource))
    );
    // Room for exactly both.
    *s.objects_mut() = Objects::with_max_slots(2);
    assert!(s.create_event().is_ok());
    assert_eq!(s.objects().len(), 2);
}

#[test]
fn a_service_event_is_a_readable_end_the_service_signals() {
    let mut s = Scheduler::default();
    let readable = s.create_readable_event().unwrap();
    assert_eq!(s.objects().len(), 1);
    assert_eq!(s.signal_object(readable), Ok(()));
    assert_eq!(s.is_signalled(readable), Ok(true));
    // The writable end of a guest's event works too.
    let (writable, guest_readable) = s.create_event().unwrap();
    assert_eq!(s.signal_object(writable), Ok(()));
    assert_eq!(s.is_signalled(guest_readable), Ok(true));
    // Anything else is an invalid handle: another kind of object, or a stale id.
    let thread = spawn(&mut s, 30, 0);
    let thread_object = s.objects_mut().create(Object::Thread(thread)).unwrap();
    let session = s.objects_mut().create(Object::Session).unwrap();
    let stale = ObjectId::from_parts(readable.index(), readable.generation() + 1);
    for refused in [thread_object, session, stale] {
        assert_eq!(
            s.signal_object(refused),
            Err(SvcError::Kernel(KernelError::InvalidHandle)),
            "{refused:?}"
        );
    }
}

#[test]
fn only_readable_events_threads_and_processes_are_synchronisation_objects() {
    let mut s = Scheduler::default();
    let (event, readable) = event(&mut s);
    assert!(is_synchronization_object(
        s.objects().get(readable).unwrap()
    ));
    assert!(is_synchronization_object(Object::Thread(
        ThreadId::from_parts(1, 1)
    )));
    assert!(is_synchronization_object(Object::Process(
        ProcessId::from_parts(0, 1)
    )));
    assert!(!is_synchronization_object(Object::Event(event)));
    assert!(!is_synchronization_object(Object::Session));
}

#[test]
fn a_thread_object_is_signalled_once_terminated_and_a_process_never() {
    let mut s = Scheduler::default();
    let thread = spawn(&mut s, 30, 0);
    let thread_object = s.objects_mut().create(Object::Thread(thread)).unwrap();
    let process = s
        .objects_mut()
        .create(Object::Process(ProcessId::from_parts(0, 1)))
        .unwrap();
    assert_eq!(s.is_signalled(thread_object), Ok(false));
    assert_eq!(s.is_signalled(process), Ok(false));
    s.set_state(thread, ThreadState::Terminated).unwrap();
    assert_eq!(s.is_signalled(thread_object), Ok(true));
    assert_eq!(s.is_signalled(process), Ok(false));
    // Asking about something that is not a synchronisation object is the kernel's own bug.
    let (event, _) = event(&mut s);
    let writable = s.objects_mut().create(Object::Event(event)).unwrap();
    assert!(matches!(
        s.is_signalled(writable),
        Err(SvcError::Internal(_))
    ));
}

// ----- Waiting before and without a wait -----------------------------------------------------

#[test]
fn a_signalled_object_ends_the_call_at_once_with_the_first_signalled_index() {
    let (mut s, thread) = one_running();
    let objects: Vec<ObjectId> = (0..3).map(|_| event(&mut s).1).collect();
    s.signal_readable(objects[1]).unwrap();
    s.signal_readable(objects[2]).unwrap();
    let before = refs(&s, &objects);
    let begun = wait(&mut s, 0, &objects, Timeout::Never, &regs(1), origin(1));
    assert_eq!(begun, Ok(WaitBegun::Signalled(1)));
    // The thread never waited, and the references the call took are released.
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    assert_eq!(s.sync().wait(thread), None);
    assert_eq!(refs(&s, &objects), before);
    assert_eq!(s.take_wakes(), []);
}

#[test]
fn a_zero_timeout_times_out_at_once_unless_an_object_is_signalled() {
    let (mut s, thread) = one_running();
    let (_, readable) = event(&mut s);
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Poll, &regs(1), origin(1)),
        Ok(WaitBegun::Ended(KernelError::TimedOut))
    );
    assert_eq!(refs(&s, &[readable]), [1]);
    // No objects at all: still a poll.
    assert_eq!(
        wait(&mut s, 0, &[], Timeout::Poll, &regs(1), origin(1)),
        Ok(WaitBegun::Ended(KernelError::TimedOut))
    );
    // Both at once: the signalled object is checked first.
    s.signal_readable(readable).unwrap();
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Poll, &regs(1), origin(1)),
        Ok(WaitBegun::Signalled(0))
    );
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
}

#[test]
fn a_core_running_nothing_fails_only_once_the_thread_is_needed() {
    let mut s = Scheduler::default();
    let (_, readable) = event(&mut s);
    // A poll never needs the thread.
    assert_eq!(
        wait(&mut s, 2, &[readable], Timeout::Poll, &regs(1), origin(1)),
        Ok(WaitBegun::Ended(KernelError::TimedOut))
    );
    // A wait does, and the references are released on the way out.
    let expected = SchedError::NoCurrentThread { core: 2 };
    assert_eq!(
        wait(&mut s, 2, &[readable], Timeout::Never, &regs(1), origin(1)),
        Err(SvcError::Internal(expected.to_string()))
    );
    assert_eq!(refs(&s, &[readable]), [1]);
    assert!(s.sync().waiters(readable).next().is_none());
}

// ----- Waiting and waking --------------------------------------------------------------------

#[test]
fn waiting_links_the_thread_saves_its_registers_and_holds_its_objects() {
    let (mut s, thread) = one_running();
    let (_, first) = event(&mut s);
    let (_, second) = event(&mut s);
    let saved = regs(7);
    let begun = wait(
        &mut s,
        0,
        &[first, second],
        Timeout::Never,
        &saved,
        origin(3),
    );
    assert_eq!(begun, Ok(WaitBegun::Waiting(thread)));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Waiting);
    assert_eq!(s.thread(thread).unwrap().context(), &saved);
    assert_eq!(refs(&s, &[first, second]), [2, 2]);
    assert_eq!(s.sync().waiters(first).collect::<Vec<_>>(), [thread]);
    assert_eq!(s.sync().waiters(second).collect::<Vec<_>>(), [thread]);
    let Some(Wait::Sync {
        objects,
        deadline,
        origin: recorded,
        ..
    }) = s.sync().wait(thread).cloned()
    else {
        unreachable!("no wait record");
    };
    assert_eq!(objects, [first, second]);
    assert_eq!((deadline, recorded), (None, origin(3)));
    // The core has nothing left to run.
    assert_eq!(switch(&mut s, 0), None);
}

#[test]
fn a_wake_writes_the_result_and_index_into_the_saved_registers() {
    let (mut s, thread) = one_running();
    let (_, first) = event(&mut s);
    let (event, second) = event(&mut s);
    let saved = regs(9);
    // The second object twice: the index is its first place in the list.
    let objects = [first, second, second];
    wait(&mut s, 0, &objects, Timeout::Never, &saved, origin(4)).unwrap();
    assert_eq!(refs(&s, &[first, second]), [2, 3]);
    s.signal_event(event).unwrap();

    let context = s.thread(thread).unwrap().context().clone();
    assert_eq!(context.x[0], word(ResultCode::SUCCESS));
    assert_eq!(context.x[1], 1);
    let mut expected = saved.clone();
    expected.x[0] = context.x[0];
    expected.x[1] = context.x[1];
    assert_eq!(context, expected);
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    assert_eq!(s.sync().wait(thread), None);
    assert_eq!(refs(&s, &[first, second]), [1, 1]);
    assert!(s.sync().waiters(first).next().is_none());
    assert!(s.sync().waiters(second).next().is_none());
    assert_eq!(
        s.take_wakes(),
        [Wake {
            thread,
            origin: origin(4),
            result: ResultCode::SUCCESS,
            index: 1,
        }]
    );
    assert_eq!(s.take_wakes(), []);
    // The other object being signalled later finds no waiter.
    s.signal_readable(first).unwrap();
    assert_eq!(s.take_wakes(), []);
}

#[test]
fn wake_order_follows_the_order_waits_began() {
    for first_waiter in [0, 1] {
        let mut s = Scheduler::default();
        let threads = [spawn(&mut s, 30, 0), spawn(&mut s, 30, 0)];
        let (_, readable) = event(&mut s);
        // Run each in turn and let it wait, the one named first first.
        for n in [first_waiter, 1 - first_waiter] {
            let thread = threads[n];
            // Put this thread at the front, so the core runs it.
            // Two threads at one priority: at most one yield brings this one to the front.
            if switch(&mut s, 0) != Some(thread) {
                s.yield_without_core_migration(core(0)).unwrap();
            }
            assert_eq!(switch(&mut s, 0), Some(thread));
            let who = origin(u8::try_from(n).unwrap());
            let begun = wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), who);
            assert_eq!(begun, Ok(WaitBegun::Waiting(thread)));
        }
        let order = [threads[first_waiter], threads[1 - first_waiter]];
        assert_eq!(s.sync().waiters(readable).collect::<Vec<_>>(), order);
        s.signal_readable(readable).unwrap();
        let woken: Vec<ThreadId> = s.take_wakes().iter().map(|wake| wake.thread).collect();
        assert_eq!(woken, order, "first waiter {first_waiter}");
        // Woken in that order, they queue in that order: the first runs first.
        assert_eq!(switch(&mut s, 0), Some(order[0]));
    }
}

#[test]
fn an_unsignalled_object_wakes_nobody() {
    let (mut s, thread) = one_running();
    let (_, readable) = event(&mut s);
    wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), origin(1)).unwrap();
    assert_eq!(s.notify_available(readable), Ok(()));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Waiting);
    assert_eq!(s.take_wakes(), []);
}

#[test]
fn closing_the_last_handle_while_waiting_keeps_the_object_until_the_wait_ends() {
    let (mut s, thread) = one_running();
    let (event, readable) = event(&mut s);
    wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), origin(1)).unwrap();
    // The "handle" (the creator's reference) goes; the wait's keeps the row.
    assert_eq!(
        s.objects_mut().close(readable),
        Ok(Closed::Remaining(core::num::NonZeroU32::MIN))
    );
    s.signal_event(event).unwrap();
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    // The wake released the last reference.
    assert_eq!(
        s.objects().get(readable),
        Err(ObjectError::Stale { id: readable })
    );
}

// ----- Cancellation --------------------------------------------------------------------------

#[test]
fn cancelling_a_wait_ends_it_with_cancelled_and_index_minus_one() {
    let (mut s, thread) = one_running();
    let (_, readable) = event(&mut s);
    let deadline = HostNanos(5_000);
    wait(
        &mut s,
        0,
        &[readable],
        Timeout::At(deadline),
        &regs(2),
        origin(2),
    )
    .unwrap();
    assert_eq!(s.sync().next_deadline(), Some(deadline));
    assert_eq!(s.cancel_synchronization(thread), Ok(()));
    let context = s.thread(thread).unwrap().context();
    assert_eq!(context.x[0], word(KernelError::Cancelled.code()));
    assert_eq!(context.x[1], MINUS_ONE);
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    // The timer went with the wait, and no cancel is left pending.
    assert_eq!(s.sync().next_deadline(), None);
    assert!(!s.sync().is_cancel_pending(thread));
    assert_eq!(refs(&s, &[readable]), [1]);
    let wakes = s.take_wakes();
    assert_eq!(
        wakes
            .iter()
            .map(|w| (w.result, w.index))
            .collect::<Vec<_>>(),
        [(KernelError::Cancelled.code(), -1)]
    );
}

#[test]
fn a_cancel_before_the_wait_stays_pending_until_a_wait_would_block() {
    let (mut s, thread) = one_running();
    let (_, readable) = event(&mut s);
    assert_eq!(s.cancel_synchronization(thread), Ok(()));
    assert!(s.sync().is_cancel_pending(thread));
    // A signalled object comes first, and the cancel stays pending.
    s.signal_readable(readable).unwrap();
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), origin(1)),
        Ok(WaitBegun::Signalled(0))
    );
    assert!(s.sync().is_cancel_pending(thread));
    // So does a zero timeout.
    s.reset_readable(readable).unwrap();
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Poll, &regs(1), origin(1)),
        Ok(WaitBegun::Ended(KernelError::TimedOut))
    );
    assert!(s.sync().is_cancel_pending(thread));
    // A wait that would block is cancelled instead, and the cancel is used up.
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), origin(1)),
        Ok(WaitBegun::Ended(KernelError::Cancelled))
    );
    assert!(!s.sync().is_cancel_pending(thread));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    assert_eq!(refs(&s, &[readable]), [1]);
    // The next one waits.
    assert_eq!(
        wait(&mut s, 0, &[readable], Timeout::Never, &regs(1), origin(1)),
        Ok(WaitBegun::Waiting(thread))
    );
}

#[test]
fn cancelling_a_thread_that_does_not_exist_is_the_kernels_bug() {
    let mut s = Scheduler::default();
    let nowhere = ThreadId::from_parts(4, 4);
    assert!(matches!(
        s.cancel_synchronization(nowhere),
        Err(SvcError::Internal(_))
    ));
    assert!(!s.sync().is_cancel_pending(nowhere));
}

// ----- Timeouts ------------------------------------------------------------------------------

#[test]
fn a_timeout_converts_as_edens_does() {
    let now = HostNanos(1_000);
    assert_eq!(Timeout::from_nanos(now, 0), Timeout::Poll);
    assert_eq!(Timeout::from_nanos(now, -1), Timeout::Never);
    assert_eq!(Timeout::from_nanos(now, i64::MIN), Timeout::Never);
    assert_eq!(
        Timeout::from_nanos(now, 1),
        Timeout::At(HostNanos(now.0 + 1 + TIMEOUT_SLACK_NS))
    );
    // The last deadline Eden's `s64` holds, and one past it.
    let latest = i64::MAX.cast_unsigned();
    let fits = i64::try_from(latest - now.0 - TIMEOUT_SLACK_NS).unwrap();
    assert_eq!(
        Timeout::from_nanos(now, fits),
        Timeout::At(HostNanos(latest))
    );
    assert_eq!(Timeout::from_nanos(now, fits + 1), Timeout::Never);
    assert_eq!(Timeout::from_nanos(now, i64::MAX), Timeout::Never);
    // A sum that overflows u64.
    assert_eq!(Timeout::from_nanos(HostNanos(u64::MAX), 1), Timeout::Never);
}

#[test]
fn a_timeout_fires_at_its_deadline_and_not_before() {
    let (mut s, thread) = one_running();
    let (_, readable) = event(&mut s);
    let deadline = HostNanos(10_000);
    wait(
        &mut s,
        0,
        &[readable],
        Timeout::At(deadline),
        &regs(3),
        origin(3),
    )
    .unwrap();
    assert_eq!(s.fire_timers(HostNanos(deadline.0 - 1)), Ok(0));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Waiting);
    assert_eq!(s.fire_timers(deadline), Ok(1));
    let context = s.thread(thread).unwrap().context();
    assert_eq!(context.x[0], word(KernelError::TimedOut.code()));
    assert_eq!(context.x[1], MINUS_ONE);
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    assert!(s.sync().waiters(readable).next().is_none());
    assert_eq!(refs(&s, &[readable]), [1]);
    assert_eq!(
        s.take_wakes()
            .iter()
            .map(|w| (w.result, w.index))
            .collect::<Vec<_>>(),
        [(KernelError::TimedOut.code(), -1)]
    );
}

#[test]
fn a_timeout_after_the_wake_is_ignored() {
    let (mut s, thread) = one_running();
    let (event, readable) = event(&mut s);
    let deadline = HostNanos(10_000);
    wait(
        &mut s,
        0,
        &[readable],
        Timeout::At(deadline),
        &regs(3),
        origin(3),
    )
    .unwrap();
    s.signal_event(event).unwrap();
    let after_wake = s.thread(thread).unwrap().context().clone();
    s.take_wakes();
    // The wake took the entry out of the queue: nothing fires.
    assert_eq!(s.sync().next_deadline(), None);
    assert_eq!(s.fire_timers(deadline), Ok(0));
    assert_eq!(s.thread(thread).unwrap().context(), &after_wake);
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
    assert_eq!(s.take_wakes(), []);
}

#[test]
fn a_late_timer_entry_is_ignored_by_its_token() {
    let (mut s, thread) = one_running();
    let (event, readable) = event(&mut s);
    let deadline = HostNanos(10_000);
    wait(
        &mut s,
        0,
        &[readable],
        Timeout::At(deadline),
        &regs(3),
        origin(3),
    )
    .unwrap();
    // The timer thread takes the entry; before it expires it, the event wakes the thread...
    let entries = s.take_due_timers(deadline);
    assert_eq!(entries.len(), 1);
    s.signal_event(event).unwrap();
    s.clear_event(event).unwrap();
    // ...which runs again and waits again, with no timeout.
    assert_eq!(switch(&mut s, 0), Some(thread));
    wait(&mut s, 0, &[readable], Timeout::Never, &regs(4), origin(4)).unwrap();
    s.take_wakes();
    assert_eq!(s.expire(entries[0]), Ok(false));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Waiting);
    assert_eq!(s.take_wakes(), []);
    // The same entry while its own wait is current does end it.
    let mut s2 = one_running().0;
    let (_, readable) = s2.create_event().unwrap();
    wait(
        &mut s2,
        0,
        &[readable],
        Timeout::At(deadline),
        &regs(3),
        origin(3),
    )
    .unwrap();
    let entries = s2.take_due_timers(deadline);
    assert_eq!(s2.expire(entries[0]), Ok(true));
}

#[test]
fn timers_with_the_same_deadline_expire_in_the_order_their_waits_began() {
    let mut s = Scheduler::default();
    let first = spawn(&mut s, 30, 0);
    let second = spawn(&mut s, 30, 1);
    let deadline = HostNanos(500);
    assert_eq!(switch(&mut s, 1), Some(second));
    assert_eq!(switch(&mut s, 0), Some(first));
    wait(&mut s, 1, &[], Timeout::At(deadline), &regs(1), origin(1)).unwrap();
    wait(&mut s, 0, &[], Timeout::At(deadline), &regs(1), origin(0)).unwrap();
    let order: Vec<ThreadId> = s.sync().timers().map(|entry| entry.thread).collect();
    assert_eq!(order, [second, first]);
    assert_eq!(s.fire_timers(deadline), Ok(2));
    let woken: Vec<ThreadId> = s.take_wakes().iter().map(|wake| wake.thread).collect();
    assert_eq!(woken, [second, first]);
}

#[test]
fn an_empty_wait_without_a_timeout_ends_only_by_cancel() {
    let (mut s, thread) = one_running();
    wait(&mut s, 0, &[], Timeout::Never, &regs(1), origin(1)).unwrap();
    assert_eq!(s.sync().next_deadline(), None);
    assert_eq!(s.fire_timers(HostNanos(u64::MAX)), Ok(0));
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Waiting);
    s.cancel_synchronization(thread).unwrap();
    assert_eq!(s.thread(thread).unwrap().state(), ThreadState::Runnable);
}

// ----- Onto a core ---------------------------------------------------------------------------

#[test]
fn a_woken_waiter_preempts_its_core_and_that_core_is_interrupted() {
    let mut s = Scheduler::default();
    let urgent = spawn(&mut s, 20, 1);
    let background = spawn(&mut s, 40, 1);
    let signaller = spawn(&mut s, 30, 0);
    assert_eq!(switch(&mut s, 1), Some(urgent));
    assert_eq!(switch(&mut s, 0), Some(signaller));
    let (event, readable) = event(&mut s);

    let saved = regs(5);
    wait(&mut s, 1, &[readable], Timeout::Never, &saved, origin(1)).unwrap();
    // Core 1 runs the background thread while the urgent one waits.
    assert_eq!(switch(&mut s, 1), Some(background));

    // Core 0's thread signals: the urgent thread is runnable, and core 1, running guest code,
    // is interrupted; the calling core is not signalled.
    s.signal_event(event).unwrap();
    let fake = update(&mut s, Some(0));
    assert_eq!(fake.log, [Signal::Interrupt(1)]);
    assert_eq!(s.highest(core(1)), Ok(Some(urgent)));
    assert_eq!(s.schedule(core(1), &mut Fake::default()), Ok(Some(urgent)));
    // It resumes with the wait's result in its registers.
    let context = s.thread(urgent).unwrap().context();
    assert_eq!(context.x[0], word(ResultCode::SUCCESS));
    assert_eq!(context.x[1], 0);
    assert_eq!(context.x[2..], saved.x[2..]);
}

#[test]
fn a_waiter_woken_onto_an_idle_core_wakes_that_core() {
    let mut s = Scheduler::default();
    let waiter = spawn(&mut s, 20, 2);
    let signaller = spawn(&mut s, 30, 0);
    assert_eq!(switch(&mut s, 2), Some(waiter));
    assert_eq!(switch(&mut s, 0), Some(signaller));
    let (event, readable) = event(&mut s);
    wait(&mut s, 2, &[readable], Timeout::Never, &regs(1), origin(1)).unwrap();
    // Core 2 parks.
    assert_eq!(switch(&mut s, 2), None);
    s.signal_event(event).unwrap();
    let fake = update(&mut s, Some(0));
    assert_eq!(fake.log, [Signal::Wake(2)]);
    assert_eq!(s.schedule(core(2), &mut Fake::default()), Ok(Some(waiter)));
}
