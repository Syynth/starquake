//! `apm`, the performance-management service: as far as a libnx start-up asks, which is one
//! command.
//!
//! Ported from `eden/src/core/hle/service/apm/apm_interface.cpp` (`APM`, `ISession`),
//! `apm_controller.h` (`PerformanceMode`) and `apm_controller.cpp`
//! (`Controller::GetCurrentPerformanceMode`) @ cf654052 (GPL-3.0-or-later). The request layout is
//! checked against the client, libnx (`libnx/nx/source/services/apm.c`, ISC).
//!
//! # Commands
//!
//! Ids and names are Eden's tables (`APM::APM`, `ISession::ISession`).
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `apm` ([`Apm`]) | 0 | `OpenSession` | nothing | an [`ISession`] |
//!
//! libnx's `_apmInitialize` opens `apm` and sends `OpenSession` (`_apmCmdGetSession(&g_apmSrv,
//! &g_apmISession, 0)`: no payload, one out object), and nothing else at start-up: the startup
//! fixture's trace (`tests/homebrew/startup/reference/startup.trace`) has no other `apm` request.
//! Eden's `OpenSession` replies with the new interface (`PushIpcInterface<ISession>`).
//!
//! Every other command is left out and answered by the server's unknown-command policy: on
//! `apm`, `GetPerformanceMode` (1) and `IsCpuOverclockEnabled` (6); on [`ISession`], all three
//! of Eden's (`SetPerformanceConfiguration`, `GetPerformanceConfiguration`,
//! `SetCpuOverclockEnabled`), which libnx sends only when a program asks for them. The other
//! names Eden registers the same interface under (`apm:am`, `apm:p`) and `apm:sys` are not
//! registered here (`eden/src/core/hle/service/apm/apm.cpp`, `LoopProcess`).
//!
//! # The performance mode
//!
//! [`PerformanceMode`] is Eden's, and [`PerformanceMode::for_docked`] is Eden's rule for the
//! current one (`Controller::GetCurrentPerformanceMode`: boost when the console is docked, normal
//! when it is not). `appletOE`'s `ICommonStateGetter::GetPerformanceMode` reports it
//! ([`crate::am`]), as Eden's does (`system.GetAPMController().GetCurrentPerformanceMode()`).

use serde::{Deserialize, Serialize};

use crate::commands;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// The name `apm` is registered under (`eden/src/core/hle/service/apm/apm.cpp`, `LoopProcess`:
/// `RegisterNamedService("apm", ..)`).
pub const NAME: &str = "apm";

/// The console's performance mode (`eden/src/core/hle/service/apm/apm_controller.h`,
/// `enum class PerformanceMode : s32`). Eden's third value, `Invalid = -1`, is not a mode the
/// console is ever in, and nothing here reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMode {
    /// `Normal = 0`: the console is not docked.
    Normal,
    /// `Boost = 1`: the console is docked.
    Boost,
}

impl PerformanceMode {
    /// The mode of a console that is docked or not (`Controller::GetCurrentPerformanceMode`:
    /// `Settings::IsDockedMode() ? PerformanceMode::Boost : PerformanceMode::Normal`).
    #[must_use]
    pub const fn for_docked(docked: bool) -> Self {
        if docked { Self::Boost } else { Self::Normal }
    }

    /// The mode as the `s32` a reply carries.
    #[must_use]
    pub const fn raw(self) -> i32 {
        match self {
            Self::Normal => 0,
            Self::Boost => 1,
        }
    }
}

/// The root interface of an `apm` session (`APM`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Apm;

/// A performance session (`ISession`): what `OpenSession` hands out. None of its commands is
/// ported yet (see the module docs), so it keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ISession;

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `apm`'s commands (`APM::APM`).
    pub mod apm {
        /// `OpenSession` (`{0, &APM::OpenSession, "OpenSession"}`).
        pub const OPEN_SESSION: u32 = 0;
    }
}

commands! {
    impl Apm {
        command::apm::OPEN_SESSION => open_session,
    }
}

// No command of `ISession` is ported yet; every request to one goes to the unknown-command
// policy.
commands! {
    impl ISession {}
}

impl Apm {
    /// `OpenSession`. out: an [`ISession`].
    fn open_session(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ISession));
        reply.ok()
    }
}

/// The interfaces an `apm` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new().with::<Apm>().with::<ISession>()
}

/// The `apm` server.
#[must_use]
pub fn server(policy: UnknownPolicy) -> Server {
    Server::new(NAME, policy)
        .with_root(|| Box::new(Apm))
        .with_interfaces(interfaces())
}
