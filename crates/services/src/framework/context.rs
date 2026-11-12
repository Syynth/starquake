//! [`Context`]: what a service method may touch outside its own state.

use core::any::Any;
use std::sync::Arc;

use starquake_kernel::id::ObjectId;
use starquake_trace::Env;

use super::defer::{Deferral, ReplyToken, TokenId};
use super::kernel::Kernel;
use super::result::{Fault, NOT_SUPPORTED};
use super::server::UnknownPolicy;
use super::snapshot::Family;
use super::targets::Targets;
use super::work::Executor;

/// What a service method may touch while it handles a request: the caller's memory, by copying in
/// and out, a narrow part of the kernel, its family's state, a reply token if it cannot answer
/// yet, and the host, through the injectable environment and a worker
/// ([`work`](super::work)).
///
/// Buffers a request describes are reached through the request
/// ([`Request::in_buffer`](super::Request::in_buffer),
/// [`Request::out_buffer`](super::Request::out_buffer)); the memory calls here are for addresses a
/// command carries in its payload. The clock of the service spec arrives with the first service
/// that reads it.
pub struct Context<'a> {
    pub(super) kernel: &'a mut dyn Kernel,
    pub(super) targets: &'a mut Targets,
    pub(super) family: Option<&'a mut dyn Family>,
    pub(super) policy: UnknownPolicy,
    pub(super) deferral: Deferral<'a>,
    pub(super) env: Option<&'a Arc<dyn Env>>,
    pub(super) executor: Option<&'a Arc<dyn Executor>>,
}

impl Context<'_> {
    /// The server's policy for what it does not know. The framework applies it to unknown
    /// commands itself; a service reads it where it has an unknown of its own to answer, as
    /// `sm:` has for an unknown service name.
    #[must_use]
    pub fn policy(&self) -> UnknownPolicy {
        self.policy
    }

    /// Creates an event. Keep the id to signal it later; give the caller a handle with
    /// [`Reply::copy_handle`](super::Reply::copy_handle).
    pub fn create_event(&mut self) -> Result<ObjectId, Fault> {
        Ok(self.kernel.create_event()?)
    }

    /// Signals an event.
    pub fn signal_event(&mut self, event: ObjectId) -> Result<(), Fault> {
        Ok(self.kernel.signal_event(event)?)
    }

    /// Copies `out.len()` bytes of the caller's memory at `address` into `out`.
    pub fn read_memory(&self, address: u64, out: &mut [u8]) -> Result<(), Fault> {
        Ok(self.kernel.read_memory(address, out)?)
    }

    /// Copies `data` into the caller's memory at `address`.
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Result<(), Fault> {
        Ok(self.kernel.write_memory(address, data)?)
    }

    /// The state the server's sessions share, as the family's own type
    /// ([`Server::with_family`](super::Server::with_family)).
    ///
    /// A server with no family state, or with state of another type, is a mistake in how the
    /// server was put together, not something a guest did: it is logged as an error and the
    /// request fails with [`NOT_SUPPORTED`].
    pub fn family<T: Family>(&mut self) -> Result<&mut T, Fault> {
        let family = self.family.as_deref_mut();
        let family = family.and_then(|family| {
            let family: &mut dyn Any = family;
            family.downcast_mut()
        });
        if family.is_none() {
            tracing::error!(
                family = core::any::type_name::<T>(),
                "the server has no family state of the type a method asked for"
            );
        }
        family.ok_or(Fault::Result(NOT_SUPPORTED))
    }

    /// The injectable environment the server was given
    /// ([`Server::with_env`](super::Server::with_env)): the only way a method reaches host files,
    /// the clock or random numbers (`docs/specs/observability.md`, rule 1). A copy of the handle,
    /// so a [`Job`](super::work::Job) can take it along.
    ///
    /// A server with no environment is a mistake in how it was put together: it is logged as an
    /// error and the request fails with [`NOT_SUPPORTED`], as for [`Context::family`].
    pub fn env(&self) -> Result<Arc<dyn Env>, Fault> {
        let env = self.env.map(Arc::clone);
        if env.is_none() {
            tracing::error!("the server has no environment, and a method asked for it");
        }
        env.ok_or(Fault::Result(NOT_SUPPORTED))
    }

    /// The executor the server was given ([`Server::with_executor`](super::Server::with_executor)),
    /// for work that takes time ([`work`](super::work)). A server with none fails the request as
    /// [`Context::env`] does.
    pub fn executor(&self) -> Result<Arc<dyn Executor>, Fault> {
        let executor = self.executor.map(Arc::clone);
        if executor.is_none() {
            tracing::error!("the server has no executor, and a method asked for one");
        }
        executor.ok_or(Fault::Result(NOT_SUPPORTED))
    }

    /// Takes the reply token for the request being handled: the method cannot answer yet. It
    /// hands the token to whoever will (a worker, a timer) and returns
    /// [`Reply::deferred`](super::Reply::deferred). See [`ReplyToken`].
    ///
    /// A request has one token. Asking for a second is a mistake in the service: it is logged
    /// as an error and fails with [`NOT_SUPPORTED`]. So is running out of token ids, which
    /// takes 2^64 deferred requests on one server.
    pub fn defer(&mut self) -> Result<ReplyToken, Fault> {
        let next = self.deferral.next_token.checked_add(1);
        let (None, Some(next)) = (self.deferral.taken, next) else {
            tracing::error!("a method asked for a second reply token, or token ids ran out");
            return Err(Fault::Result(NOT_SUPPORTED));
        };
        let id = TokenId(*self.deferral.next_token);
        *self.deferral.next_token = next;
        self.deferral.taken = Some(id);
        Ok(ReplyToken::new(id, Arc::clone(self.deferral.mailbox)))
    }
}
