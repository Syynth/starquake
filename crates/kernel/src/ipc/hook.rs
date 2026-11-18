//! [`EventHook`]: what a service attaches to an event it created, so that it hears when the guest
//! signals it (kernel spec, "Waiting and waking": "A hook on service-created events").
//!
//! The graphics server is the first user: the title signals a request event to ask for fresh
//! syncpoint values, and the hook queues a refresh of the shared page
//! (`docs/specs/graphics-server.md`, "The shared syncpoint page").
//!
//! # When a hook runs
//!
//! A hooked event is made by [`IpcContext::create_hooked_event`]: an event with both ends, the
//! writable one meant for the guest. Its hook runs once for every `SignalEvent` (0x11) system
//! call that names that writable end and returns success, whether or not the event was already
//! signalled, and after the dispatch has released the kernel lock. It does not run for:
//!
//! - a `SignalEvent` that fails (a handle that names nothing, or something else);
//! - `ClearEvent`, `ResetSignal`, or a wait on the readable end;
//! - the service's own signals ([`IpcContext::signal_event`]) or a host thread's
//!   ([`Dispatcher::signal_event`](crate::dispatch::Dispatcher::signal_event)): the hook is for
//!   what the guest does.
//!
//! Tests (in `dispatch/hooks/tests.rs`): `a_hook_runs_after_the_guest_signals_its_event`,
//! `a_hook_runs_without_the_kernel_lock`, `a_hook_runs_only_for_a_successful_guest_signal`.
//!
//! # After a restore
//!
//! A hook is code, so a snapshot cannot hold it: the owning service re-attaches it in its
//! restore step, before the guest runs again
//! ([`Dispatcher::reattach_event_hook`](crate::dispatch::Dispatcher::reattach_event_hook), which
//! takes only the writable end of an event that service's server made; owner's ruling of
//! 2026-11-17). Tests: `a_hook_reattached_after_a_restore_runs_when_the_guest_signals`,
//! `a_hook_is_reattached_only_to_a_writable_end_its_service_made`. The snapshot saves which
//! events are hooked and by which service ([`crate::object::HookedEvent`], saved by
//! [`Dispatcher::save_objects`](crate::dispatch::Dispatcher::save_objects)), and
//! [`Dispatcher::unattached_event_hooks`](crate::dispatch::Dispatcher::unattached_event_hooks)
//! names the ones no service has re-attached yet. Test:
//! `a_saved_hook_is_named_by_the_restore_until_its_service_reattaches_it`.
//!
//! Eden has no such hook: its services signal and wait on their `KEvent`s from their own host
//! threads. This is starquake's, from the kernel spec.
//!
//! [`IpcContext::create_hooked_event`]: crate::dispatch::IpcContext::create_hooked_event
//! [`IpcContext::signal_event`]: crate::dispatch::IpcContext::signal_event

use crate::id::ObjectId;

/// Code a service attaches to an event it created ([`IpcContext::create_hooked_event`]), run when
/// the guest signals the event. See the [module docs](self) for exactly when.
///
/// It runs on the thread of the core that made the `SignalEvent` call, without the kernel lock,
/// and must only queue work for the service (kernel spec): it never calls back into the kernel
/// synchronously, and never blocks.
///
/// [`IpcContext::create_hooked_event`]: crate::dispatch::IpcContext::create_hooked_event
pub trait EventHook: Send + Sync {
    /// The guest signalled `event`, the writable end the hook was attached to.
    fn signalled(&self, event: ObjectId);
}
