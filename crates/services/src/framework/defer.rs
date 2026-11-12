//! Deferred replies: [`ReplyToken`], and the mailbox its completion goes through.
//!
//! A method that cannot answer yet takes a token with [`Context::defer`](super::Context::defer)
//! and returns [`Reply::deferred`](super::Reply::deferred). The handler has then finished, the
//! server takes its next request, and the caller waits as a guest thread
//! (`docs/specs/execution-model.md`, "IPC and services"). Whoever finishes the work, on any host
//! thread, completes the token:
//!
//! ```
//! use starquake_kernel::result::ResultCode;
//! use starquake_services::framework::{Reply, ReplyToken};
//!
//! fn frame_ready(token: ReplyToken, frame: u64) {
//!     let mut reply = Reply::default();
//!     reply.payload().write_u64(frame);
//!     token.complete(reply);
//! }
//!
//! fn cancelled(token: ReplyToken, why: ResultCode) {
//!     token.fail(why);
//! }
//! ```
//!
//! Completing a token does not touch the caller: it posts the reply to the server's mailbox and
//! tells the server's [`Notify`]. The kernel then delivers it on a core thread
//! ([`Server::take_completed`](super::Server::take_completed),
//! [`Server::deliver`](super::Server::deliver)), which writes the reply into the caller's message
//! buffer; waking the caller is the kernel's step after that. The order in which the kernel
//! takes completions is the order they were posted in, and each delivery records the request's
//! one trace event, so the event stream holds the order deferred replies completed in
//! (`docs/specs/observability.md`, "Replaying the kernel and services", rule 3).
//!
//! A token is an id ([`TokenId`]) and a way to reach the mailbox. The id is what a server's
//! snapshot holds; after a restore the work is resumed with a new token for the same id
//! ([`Server::reissue`](super::Server::reissue)).
//!
//! **Dropping a token without completing it is a bug** in the service: the drop is logged as an
//! error and the caller gets [`DEFERRED_REPLY_DROPPED`](super::DEFERRED_REPLY_DROPPED).
//!
//! No reference source is ported: Eden's services answer on their own host threads, and its
//! "deferred" requests (`eden/src/core/hle/service/server_manager.cpp`, `OnDeferralEvent`) are
//! re-run from the start, which is a different mechanism. This design is the execution model
//! spec's.
//!
//! A completion carries a [`Reply`]: a payload, handles, new interfaces, and bytes for the
//! caller's out buffers ([`Reply::write_buffer`]). The thread that completes a token has no access
//! to guest memory; the bytes are written when the reply is delivered, by the kernel that speaks
//! for the caller then.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::reply::Reply;

/// Names one deferred request of a server. Ids count up from zero in the order tokens are
/// taken, and are never reused by the same server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenId(pub(super) u64);

impl TokenId {
    /// The id as a number, for logs.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Told when a reply token of a server is completed or dropped, on the thread that did it.
///
/// The kernel's session code implements it to schedule a delivery
/// ([`Server::take_completed`](super::Server::take_completed)). It must not block, and it is
/// called with no lock of the framework held.
pub trait Notify: Send + Sync {
    /// A completion was posted to the server's mailbox.
    fn completion_posted(&self);
}

/// How a token ended.
pub(super) enum Posted {
    /// Completed with a reply.
    Reply(Reply),
    /// Completed with a failure.
    Failed(ResultCode),
    /// Dropped.
    Dropped,
}

/// What a mailbox holds.
#[derive(Default)]
struct Held {
    posted: VecDeque<(TokenId, Posted)>,
    /// Set when the server stopped reading the mailbox ([`Mailbox::close`]).
    closed: bool,
}

/// Where tokens post their completions, in the order they arrive.
///
/// The lock is held only to push or pop one entry, never across a call out of this module, so
/// no thread waits on it for longer than that.
pub(super) struct Mailbox {
    held: Mutex<Held>,
    notify: Option<Arc<dyn Notify>>,
}

impl Mailbox {
    pub(super) fn new(notify: Option<Arc<dyn Notify>>) -> Self {
        Self {
            held: Mutex::default(),
            notify,
        }
    }

    /// What the mailbox holds. A poisoned lock means a panic elsewhere while an entry was pushed
    /// or popped; the queue is valid at every step, so carry on.
    fn held(&self) -> MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn post(&self, token: TokenId, posted: Posted) {
        {
            let mut held = self.held();
            if held.closed {
                return;
            }
            held.posted.push_back((token, posted));
        }
        if let Some(notify) = &self.notify {
            notify.completion_posted();
        }
    }

    /// The oldest completion not taken yet.
    pub(super) fn take(&self) -> Option<(TokenId, Posted)> {
        self.held().posted.pop_front()
    }

    /// Stops the mailbox: what it holds is thrown away, and what is posted from now on is
    /// dropped without telling anybody. A server does this to the mailbox of a run it has been
    /// restored away from, whose tokens may still be out.
    pub(super) fn close(&self) {
        let mut held = self.held();
        held.closed = true;
        held.posted.clear();
    }
}

/// The right to answer one deferred request, from any thread. See the [module docs](self).
#[must_use = "a dropped token replies with an error"]
pub struct ReplyToken {
    id: TokenId,
    mailbox: Arc<Mailbox>,
    spent: bool,
}

impl ReplyToken {
    pub(super) fn new(id: TokenId, mailbox: Arc<Mailbox>) -> Self {
        Self {
            id,
            mailbox,
            spent: false,
        }
    }

    /// The token's id: what a session or a worker stores to name the request.
    #[must_use]
    pub fn id(&self) -> TokenId {
        self.id
    }

    /// Answers the request with `reply`, as if the method had filled it in and returned
    /// [`Reply::ok`].
    pub fn complete(mut self, reply: Reply) {
        self.spent = true;
        self.mailbox.post(self.id, Posted::Reply(reply));
    }

    /// Answers the request with a result code and nothing else, as if the method had failed
    /// with it.
    pub fn fail(mut self, result: ResultCode) {
        self.spent = true;
        self.mailbox.post(self.id, Posted::Failed(result));
    }
}

impl Drop for ReplyToken {
    fn drop(&mut self) {
        if self.spent {
            return;
        }
        tracing::error!(
            token = self.id.raw(),
            "a reply token was dropped without being completed: the caller gets an error"
        );
        self.mailbox.post(self.id, Posted::Dropped);
    }
}

/// How a deferred request's reply is framed. The session's domain, if it is one, is looked up
/// when the reply is delivered: a session with a request in flight cannot be converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Framing {
    /// CMIF without a domain header.
    Cmif,
    /// CMIF with a domain header.
    Domain,
    /// TIPC, echoing this command id.
    Tipc(u16),
}

/// What a server keeps of a deferred request until its token completes: where the reply goes,
/// and what the request's trace event says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Pending {
    pub(super) session: ObjectId,
    pub(super) framing: Framing,
    pub(super) command: u32,
    pub(super) request_size: u32,
}

/// A deferred request whose token has completed, taken from the server to be delivered
/// ([`Server::deliver`](super::Server::deliver)).
#[must_use = "an undelivered completion leaves its caller waiting"]
pub struct Completed {
    pub(super) token: TokenId,
    pub(super) pending: Pending,
    pub(super) posted: Posted,
}

impl Completed {
    /// The token that completed.
    #[must_use]
    pub fn token(&self) -> TokenId {
        self.token
    }

    /// The session the request was sent on: the kernel finds the waiting thread and its message
    /// buffer by it.
    #[must_use]
    pub fn session(&self) -> ObjectId {
        self.pending.session
    }
}

/// What one dispatch may defer with: the server's mailbox and its token counter.
pub(super) struct Deferral<'a> {
    pub(super) mailbox: &'a Arc<Mailbox>,
    pub(super) next_token: &'a mut u64,
    /// The token the method took, if it took one.
    pub(super) taken: Option<TokenId>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::result::NOT_SUPPORTED;
    use super::*;

    #[derive(Default)]
    struct Count(AtomicUsize);

    impl Notify for Count {
        fn completion_posted(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn token(mailbox: &Arc<Mailbox>, id: u64) -> ReplyToken {
        ReplyToken::new(TokenId(id), Arc::clone(mailbox))
    }

    #[test]
    fn completions_are_taken_in_the_order_they_were_posted() {
        let count = Arc::new(Count::default());
        let notify: Arc<dyn Notify> = Arc::<Count>::clone(&count);
        let mailbox = Arc::new(Mailbox::new(Some(notify)));
        let (first, second, third) = (token(&mailbox, 7), token(&mailbox, 3), token(&mailbox, 5));
        assert_eq!(first.id().raw(), 7);
        assert!(mailbox.take().is_none());

        let mut reply = Reply::default();
        reply.payload().write_u8(9);
        third.complete(reply);
        first.fail(NOT_SUPPORTED);
        drop(second);
        // Each one told the server, once.
        assert_eq!(count.0.load(Ordering::SeqCst), 3);

        let (id, posted) = mailbox.take().unwrap();
        assert_eq!(id, TokenId(5));
        let Posted::Reply(reply) = posted else {
            unreachable!("token 5 was completed with a reply")
        };
        let (payload, copies, moves) = reply.into_parts();
        assert_eq!(payload.as_bytes(), [9]);
        assert!(copies.is_empty() && moves.is_empty());
        assert!(matches!(
            mailbox.take(),
            Some((TokenId(7), Posted::Failed(code))) if code == NOT_SUPPORTED
        ));
        assert!(matches!(
            mailbox.take(),
            Some((TokenId(3), Posted::Dropped))
        ));
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn a_closed_mailbox_drops_what_it_held_and_what_comes_later() {
        let count = Arc::new(Count::default());
        let notify: Arc<dyn Notify> = Arc::<Count>::clone(&count);
        let mailbox = Arc::new(Mailbox::new(Some(notify)));
        let late = token(&mailbox, 2);
        token(&mailbox, 1).fail(NOT_SUPPORTED);
        mailbox.close();
        assert!(mailbox.take().is_none());
        late.complete(Reply::default());
        drop(token(&mailbox, 3));
        // Nothing arrived and nobody was told after the first.
        assert!(mailbox.take().is_none());
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_completed_token_posts_once() {
        let mailbox = Arc::new(Mailbox::new(None));
        token(&mailbox, 1).complete(Reply::default());
        token(&mailbox, 2).fail(NOT_SUPPORTED);
        // Completing consumed each token; its drop posted nothing more.
        assert!(mailbox.take().is_some());
        assert!(mailbox.take().is_some());
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn a_token_is_completed_from_another_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<ReplyToken>();
        let mailbox = Arc::new(Mailbox::new(None));
        let token = token(&mailbox, 4);
        std::thread::spawn(move || token.fail(NOT_SUPPORTED))
            .join()
            .unwrap();
        assert!(matches!(
            mailbox.take(),
            Some((TokenId(4), Posted::Failed(_)))
        ));
    }
}
