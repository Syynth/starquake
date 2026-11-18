//! `ectx:aw`, the error-context service as an application reaches it: a factory of context
//! registrars.
//!
//! Ported from `eden/src/core/hle/service/glue/ectx.cpp` @ cf654052 (GPL-3.0-or-later):
//! `ECTX_AW` (`CreateContextRegistrar`) and `IContextRegistrar`, the registration in
//! `eden/src/core/hle/service/glue/glue.cpp` (`RegisterNamedService("ectx:aw", ..)`). Eden-NX2
//! @ 50fa86bb and Eden PR #4588 @ e5fb3b97 have the same `ectx.cpp` and `ectx.h`, and the
//! Switch 2 scoping document (`docs/switch2-eden-diff-scoping.md`) has no `ectx` row, so Switch
//! 2 changes nothing here.
//!
//! # Commands
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `ectx:aw` ([`EctxAw`]) | 0 | `CreateContextRegistrar` | nothing | an [`IContextRegistrar`] |
//!
//! Eden's `CreateContextRegistrar` reads nothing and replies with the new interface
//! (`PushIpcInterface<IContextRegistrar>`). `ectx:aw`'s `CommitContext` (1), which Eden lists
//! without a handler, goes to the server's unknown-command policy, and so does every command of
//! [`IContextRegistrar`]. `ectx:w` and `ectx:r`, which Eden registers with no handlers at all,
//! are not registered here.
//!
//! # Why `Complete` is left to the unknown-command policy
//!
//! [`IContextRegistrar`]'s one command in Eden, `Complete` (0), reads a `u32` and a buffer and
//! answers success with a `u32` 0 (`rb.Push(ResultSuccess); rb.Push(0);`). Issue #278 ports only
//! the commands the title reaches, and no title has been seen to send this one. Under the
//! auto-stub policy, which a run uses, its answer is already Eden's: success, and a payload of
//! zeroes, so the `u32` the caller reads is 0. The run's list of auto-stubbed commands names it
//! if a title sends it, which is the sign to port it. Under the strict policy, the tests'
//! default, it is answered [`UNKNOWN_COMMAND_ID`](crate::framework::result::UNKNOWN_COMMAND_ID)
//! (`ectx/tests.rs` pins both answers).

use serde::{Deserialize, Serialize};

use crate::commands;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// The name `ectx:aw` is registered under (`eden/src/core/hle/service/glue/glue.cpp`,
/// `LoopProcess`: `RegisterNamedService("ectx:aw", ..)`).
pub const AW_NAME: &str = "ectx:aw";

/// The root interface of an `ectx:aw` session (`ECTX_AW`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EctxAw;

/// A context registrar (`nn::err::context::IContextRegistrar`): what `CreateContextRegistrar`
/// hands out. None of its commands is ported yet (module docs), so it keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IContextRegistrar;

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `ectx:aw`'s commands (`ECTX_AW::ECTX_AW`).
    pub mod aw {
        /// `CreateContextRegistrar` (`{0, &ECTX_AW::CreateContextRegistrar,
        /// "CreateContextRegistrar"}`).
        pub const CREATE_CONTEXT_REGISTRAR: u32 = 0;
    }
}

commands! {
    impl EctxAw {
        command::aw::CREATE_CONTEXT_REGISTRAR => create_context_registrar,
    }
}

// No command of `IContextRegistrar` is ported yet; every request to one goes to the
// unknown-command policy.
commands! {
    impl IContextRegistrar {}
}

impl EctxAw {
    /// `CreateContextRegistrar`. out: an [`IContextRegistrar`].
    fn create_context_registrar(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IContextRegistrar));
        reply.ok()
    }
}

/// The interfaces an `ectx:aw` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<EctxAw>()
        .with::<IContextRegistrar>()
}

/// The `ectx:aw` server.
#[must_use]
pub fn aw_server(policy: UnknownPolicy) -> Server {
    Server::new(AW_NAME, policy)
        .with_root(|| Box::new(EctxAw))
        .with_interfaces(interfaces())
}
