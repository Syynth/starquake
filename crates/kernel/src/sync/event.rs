//! Events: a writable end that signals and clears, and a readable end that threads wait on.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_event.{h,cpp}` and
//! `eden/src/core/hle/kernel/k_readable_event.{h,cpp}` @ cf654052 (GPL-3.0-or-later):
//! - `KReadableEvent::Signal`: if the event is not signalled, it becomes signalled and its
//!   waiters are notified (`NotifyAvailable`); signalling a signalled event does nothing.
//!   [`Scheduler::signal_readable`].
//! - `KReadableEvent::Reset`: [`KernelError::InvalidState`] unless the event is signalled, which
//!   it then stops being. [`Scheduler::reset_readable`].
//! - `KReadableEvent::Clear`: `Reset`, its result ignored, so it always succeeds.
//!   [`Scheduler::clear_readable`].
//! - `KEvent::Signal` and `KEvent::Clear`: the readable end's, or nothing when the readable end
//!   was destroyed (`R_SUCCEED_IF(m_readable_event_destroyed)`). [`Scheduler::signal_event`],
//!   [`Scheduler::clear_event`].
//!
//! An event's state is its rows in the object table ([`Object::Event`],
//! [`Object::ReadableEvent`]), so it goes when the last reference to the row is closed and
//! nothing else has to tear it down. How the two ends differ from Eden's is listed in the
//! module docs of [`crate::sync`].

use crate::id::ObjectId;
use crate::object::Object;
use crate::result::{CallResult, KernelError, SvcError};
use crate::sched::Scheduler;

/// The writable end of an event (Horizon's `KEvent`): what `SignalEvent` and `ClearEvent` take.
/// It is not a synchronisation object; threads wait on its readable end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Event {
    readable: ObjectId,
}

impl Event {
    /// The readable end this event signals. Its row is gone once every reference to it was
    /// closed; signalling and clearing then do nothing.
    #[must_use]
    pub const fn readable(self) -> ObjectId {
        self.readable
    }
}

/// The readable end of an event (Horizon's `KReadableEvent`): a synchronisation object, which
/// is signalled from the moment it is signalled until it is reset or cleared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReadableEvent {
    signalled: bool,
}

impl ReadableEvent {
    /// Whether the event is signalled (`KReadableEvent::IsSignaled`). A new event is not
    /// (`KReadableEvent::Initialize`).
    #[must_use]
    pub const fn is_signalled(self) -> bool {
        self.signalled
    }
}

impl Scheduler {
    /// Creates the readable end of an event, not signalled, with one reference: its creator's.
    /// [`KernelError::OutOfResource`] when the object table is full.
    pub fn create_readable_event(&mut self) -> CallResult<ObjectId> {
        let readable = Object::ReadableEvent(ReadableEvent::default());
        Ok(self.objects_mut().create(readable)?)
    }

    /// Creates an event (`KEvent::Create` and `Initialize`): its writable end and its readable
    /// end, in that order, each with one reference, its creator's. All or nothing: when the
    /// object table has room for the readable end only, it is released again and the result is
    /// [`KernelError::OutOfResource`].
    pub fn create_event(&mut self) -> CallResult<(ObjectId, ObjectId)> {
        let readable = self.create_readable_event()?;
        match self.objects_mut().create(Object::Event(Event { readable })) {
            Ok(writable) => Ok((writable, readable)),
            Err(error) => {
                self.release_object(readable)?;
                Err(error.into())
            }
        }
    }

    /// The state of the readable end `readable`. An id that names anything else is the
    /// caller's bookkeeping, not a guest's: the system calls resolve their handles first.
    fn readable_state(&self, readable: ObjectId) -> CallResult<ReadableEvent> {
        match self.objects().get(readable)? {
            Object::ReadableEvent(state) => Ok(state),
            other => Err(SvcError::Internal(format!(
                "{readable:?} is {other:?}, not an event's readable end"
            ))),
        }
    }

    fn set_readable_state(&mut self, readable: ObjectId, state: ReadableEvent) -> CallResult<()> {
        self.objects_mut()
            .replace(readable, Object::ReadableEvent(state))?;
        Ok(())
    }

    /// The readable end of `event`, or `None` when its row is gone (Eden's
    /// `m_readable_event_destroyed`).
    fn live_readable(&self, event: Event) -> Option<ObjectId> {
        self.objects()
            .get(event.readable)
            .ok()
            .map(|_| event.readable)
    }

    /// Signals the readable end `readable` (`KReadableEvent::Signal`): if it was not
    /// signalled, it becomes signalled and every thread waiting on it is woken, in the order
    /// they began to wait.
    pub fn signal_readable(&mut self, readable: ObjectId) -> CallResult<()> {
        if self.readable_state(readable)?.is_signalled() {
            return Ok(());
        }
        self.set_readable_state(readable, ReadableEvent { signalled: true })?;
        self.notify_available(readable)
    }

    /// Resets the readable end `readable` (`KReadableEvent::Reset`):
    /// [`KernelError::InvalidState`] unless it is signalled, which it then stops being.
    pub fn reset_readable(&mut self, readable: ObjectId) -> CallResult<()> {
        if !self.readable_state(readable)?.is_signalled() {
            return Err(KernelError::InvalidState.into());
        }
        self.set_readable_state(readable, ReadableEvent { signalled: false })
    }

    /// Clears the readable end `readable` (`KReadableEvent::Clear`): a reset whose
    /// [`KernelError::InvalidState`] is ignored, so an event that was not signalled stays as it
    /// was and the call still succeeds.
    pub fn clear_readable(&mut self, readable: ObjectId) -> CallResult<()> {
        match self.reset_readable(readable) {
            Ok(()) | Err(SvcError::Kernel(KernelError::InvalidState)) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Signals through the writable end `event` (`KEvent::Signal`): its readable end's
    /// [`Scheduler::signal_readable`], or nothing when the readable end is gone.
    pub fn signal_event(&mut self, event: Event) -> CallResult<()> {
        match self.live_readable(event) {
            Some(readable) => self.signal_readable(readable),
            None => Ok(()),
        }
    }

    /// Clears through the writable end `event` (`KEvent::Clear`): its readable end's
    /// [`Scheduler::clear_readable`], or nothing when the readable end is gone.
    pub fn clear_event(&mut self, event: Event) -> CallResult<()> {
        match self.live_readable(event) {
            Some(readable) => self.clear_readable(readable),
            None => Ok(()),
        }
    }

    /// Signals the event `object` is an end of, whichever end: what a service does with an
    /// event it made ([`Scheduler::create_readable_event`]) or was given a handle to. An id that
    /// names no live object, or an object that is not an event, is
    /// [`KernelError::InvalidHandle`], as a guest's handle to one would be.
    pub fn signal_object(&mut self, object: ObjectId) -> CallResult<()> {
        match self.objects().get(object) {
            Ok(Object::Event(event)) => self.signal_event(event),
            Ok(Object::ReadableEvent(_)) => self.signal_readable(object),
            Ok(Object::Thread(_) | Object::Process(_) | Object::Session) | Err(_) => {
                Err(KernelError::InvalidHandle.into())
            }
        }
    }
}
