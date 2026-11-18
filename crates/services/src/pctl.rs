//! Parental control: `pctl`, `pctl:a`, `pctl:s` and `pctl:r`, as far as a title's start-up asks.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/service/pctl/pctl.cpp` (`LoopProcess`): the four names and the
//!   capability each one's factory has ([`Name`], [`Name::capability`]);
//! - `eden/src/core/hle/service/pctl/parental_control_service_factory.cpp`
//!   (`IParentalControlServiceFactory`): `CreateService` and `CreateServiceWithoutInitialize`;
//! - `eden/src/core/hle/service/pctl/parental_control_service.cpp` (`IParentalControlService`):
//!   its command table and `Initialize`;
//! - `eden/src/core/hle/service/pctl/pctl_results.h`: [`NO_CAPABILITY`];
//! - `eden/src/core/hle/service/pctl/pctl_types.h`: [`Capability`].
//!
//! The same eight files are byte for byte the same in Eden-NX2 (`eden-nx2/src/core/hle/service/
//! pctl/` @ 50fa86bb) and Eden PR #4588 (`eden-pr4588/src/core/hle/service/pctl/` @ e5fb3b97),
//! so every id and constant below is the same in all three, and the scoping doc
//! (`docs/switch2-eden-diff-scoping.md`) has no `pctl` row. How a client lays out its requests
//! is checked against libnx (`libnx/nx/source/services/pctl.c`, ISC): `_pctlCreateService` sends
//! the caller's process id and a reserved `u64`, and takes one object back, on a session it has
//! made a domain.
//!
//! # Commands
//!
//! Ids and names are Eden's tables (`IParentalControlServiceFactory::
//! IParentalControlServiceFactory`, `IParentalControlService::IParentalControlService`).
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | [`IParentalControlServiceFactory`] | 0 | `CreateService` | the process id (not read) | an [`IParentalControlService`] |
//! | [`IParentalControlServiceFactory`] | 1 | `CreateServiceWithoutInitialize` | the process id (not read) | an [`IParentalControlService`] |
//! | [`IParentalControlService`] | 1 | `Initialize` | nothing | nothing, or [`NO_CAPABILITY`] |
//!
//! These are the commands a title's start-up was seen to send in the local title check: one
//! of the factory commands on each name, then `Initialize` on the object `pctl` handed out.
//! Every other command of `IParentalControlService` (the restriction queries, the play timer,
//! stereo vision and the rest of Eden's table) is left out and answered by the server's
//! unknown-command policy, until a trace shows a title sending one.
//!
//! # Capabilities
//!
//! Each name's factory has a fixed [`Capability`] (`pctl.cpp`): `pctl` has `Application`,
//! `SnsPost`, `Status` and `StereoVision`; `pctl:a`, `pctl:s` and `pctl:r` have none. Eden marks
//! the last three `TODO(ogniK): Implement remaining capabilities`, so on a console they
//! presumably carry system capabilities; that is a guess Eden makes too, and a trace of a
//! system module's `Initialize` would settle it. An [`IParentalControlService`] has its
//! factory's capability, as Eden's does (`CreateService` passes `capability` on): here it reads
//! it from its server's family state ([`Context::family`]), which the factory's server holds.
//!
//! `Initialize` refuses a capability with neither `Application` nor `System` with
//! [`NO_CAPABILITY`], Eden's exact check: `if (False(capability & (Capability::Application |
//! Capability::System))) R_THROW(PCTL::ResultNoCapability)`. So it succeeds on `pctl` and fails
//! on the other three names.
//!
//! # Where this differs from Eden
//!
//! - **The process id is not read.** Eden resolves it to the caller's program id
//!   (`system.ResolveCallerProgramId`) and hands that to the new service, whose `Initialize`
//!   reads the title's control data with it (its age rating and parental-control flag, into
//!   `states.application_info`) when it is not zero. starquake's services have no way to read a
//!   title's control data yet, and no ported command reads what it would fill in, so the id is
//!   not read and `Initialize` records only that it ran ([`IParentalControlService`]). Like
//!   Eden (`cmif_serialization.h`: `ctx.GetPID()`, with `TODO: abort parsing if PID is not
//!   provided?`), a request that did not send the id is not refused, and the reserved `u64` is
//!   not read either (`a_factory_request_without_the_process_id_or_payload_still_creates`).
//! - **`CreateService` does not run `Initialize`**, the same as Eden, whose two factory
//!   commands are identical. The name suggests that a console's runs it; a guess, which a trace
//!   of a title sending `CreateService` (0) would confirm. Pinned by
//!   `create_service_does_not_initialize`.
//!
//! # Snapshots
//!
//! - **Family state**: the server's [`Capability`]. It is the run's wiring, fixed by the name,
//!   so a restore refuses a snapshot whose capability is not the server's with
//!   [`SnapshotError::FamilyWiring`] (`a_snapshot_with_another_capability_is_refused`), and one
//!   with a bit [`Capability::KNOWN`] does not name with [`SnapshotError::Decode`]
//!   (`a_snapshot_with_an_unknown_capability_bit_is_refused`). A snapshot of another name's
//!   server is the framework's [`SnapshotError::WrongServer`]
//!   (`a_snapshot_is_restored_only_into_its_own_name_s_server`).
//! - **[`IParentalControlService`]**: whether `Initialize` succeeded on it. The factory keeps
//!   nothing.
//!
//! Not checked here: a service marked initialized on a server whose capability refuses
//! `Initialize` (`pctl:a`, `pctl:s`, `pctl:r`). No live run makes one, but the framework thaws
//! a session's state without its family's, so the relation cannot be checked where the state is
//! decoded. No ported command reads the flag, so a restored one changes no reply. Pinned by
//! `a_service_initialized_where_initialize_fails_is_restored_as_saved`.
//!
//! [`SnapshotError::FamilyWiring`]: crate::framework::SnapshotError::FamilyWiring
//! [`SnapshotError::Decode`]: crate::framework::SnapshotError::Decode
//! [`SnapshotError::WrongServer`]: crate::framework::SnapshotError::WrongServer

use serde::{Deserialize, Serialize};
use starquake_kernel::result::ResultCode;

use crate::commands;
use crate::framework::result::code;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// `ErrorModule::PCTL` (`eden/src/core/hle/result.h` @ cf654052: `PCTL = 142`).
pub const PCTL_MODULE: u32 = 142;

/// `ResultNoCapability` (`eden/src/core/hle/service/pctl/pctl_results.h` @ cf654052:
/// `constexpr Result ResultNoCapability{ErrorModule::PCTL, 131}`), which `Initialize` answers
/// when the capability has neither [`Capability::APPLICATION`] nor [`Capability::SYSTEM`].
/// The header's other three results belong to commands not ported here.
pub const NO_CAPABILITY: ResultCode = code(PCTL_MODULE, 131);

/// The four names parental control is registered under (`pctl.cpp`, `LoopProcess`:
/// `RegisterNamedService("pctl", ..)`, `("pctl:a", ..)`, `("pctl:r", ..)`, `("pctl:s", ..)`).
/// Each is a server of its own, since each one's factory has its own capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Name {
    /// `pctl`, the name an application uses.
    Pctl,
    /// `pctl:a`.
    A,
    /// `pctl:s`.
    S,
    /// `pctl:r`.
    R,
}

impl Name {
    /// Every name, in the order Eden registers them.
    pub const ALL: [Self; 4] = [Self::Pctl, Self::A, Self::R, Self::S];

    /// The name, as a client asks the service manager for it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pctl => "pctl",
            Self::A => "pctl:a",
            Self::S => "pctl:s",
            Self::R => "pctl:r",
        }
    }

    /// The capability the name's factory has (`pctl.cpp`, `LoopProcess`; module docs,
    /// "Capabilities").
    #[must_use]
    pub const fn capability(self) -> Capability {
        match self {
            Self::Pctl => Capability::APPLICATION
                .union(Capability::SNS_POST)
                .union(Capability::STATUS)
                .union(Capability::STEREO_VISION),
            Self::A | Self::S | Self::R => Capability::NONE,
        }
    }
}

/// A capability a parental-control factory and its services have
/// (`eden/src/core/hle/service/pctl/pctl_types.h` @ cf654052: `enum class Capability : u32`,
/// a set of flags). It holds only the bits Eden names ([`Capability::KNOWN`]): it is made with
/// [`Capability::from_bits`], and a snapshot's is decoded through the same check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct Capability(u32);

/// A capability word with a bit [`Capability::KNOWN`] does not name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("capability {0:#x} has a bit parental control does not define")]
pub struct UnknownCapability(pub u32);

impl Capability {
    /// `None = 0`.
    pub const NONE: Self = Self(0);
    /// `Application = 1 << 0`.
    pub const APPLICATION: Self = Self(1 << 0);
    /// `SnsPost = 1 << 1`.
    pub const SNS_POST: Self = Self(1 << 1);
    /// `Recovery = 1 << 6`.
    pub const RECOVERY: Self = Self(1 << 6);
    /// `Status = 1 << 8`.
    pub const STATUS: Self = Self(1 << 8);
    /// `StereoVision = 1 << 9`.
    pub const STEREO_VISION: Self = Self(1 << 9);
    /// `System = 1 << 15`.
    pub const SYSTEM: Self = Self(1 << 15);

    /// Every bit Eden's enum names.
    pub const KNOWN: u32 = Self::APPLICATION.0
        | Self::SNS_POST.0
        | Self::RECOVERY.0
        | Self::STATUS.0
        | Self::STEREO_VISION.0
        | Self::SYSTEM.0;

    /// The capability `bits` name, or [`UnknownCapability`] if one of them is not in
    /// [`Capability::KNOWN`].
    pub const fn from_bits(bits: u32) -> Result<Self, UnknownCapability> {
        if bits & !Self::KNOWN == 0 {
            Ok(Self(bits))
        } else {
            Err(UnknownCapability(bits))
        }
    }

    /// The capability's word.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Both capabilities' bits.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the two share a bit (Eden's `True(a & b)`).
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl TryFrom<u32> for Capability {
    type Error = UnknownCapability;

    fn try_from(bits: u32) -> Result<Self, Self::Error> {
        Self::from_bits(bits)
    }
}

impl From<Capability> for u32 {
    fn from(capability: Capability) -> Self {
        capability.0
    }
}

/// The capability `Initialize` needs one bit of (module docs, "Capabilities").
const INITIALIZE_NEEDS: Capability = Capability::APPLICATION.union(Capability::SYSTEM);

/// The root interface of a parental-control session (`IParentalControlServiceFactory`). Its
/// capability is its server's family state, so it keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IParentalControlServiceFactory;

/// A parental-control service (`IParentalControlService`): what the factory hands out.
///
/// It keeps whether `Initialize` succeeded on it, the part of Eden's `states` this port can
/// fill in (module docs, "Where this differs from Eden"). A new one has not been initialized:
/// Eden's `States states{}` starts value-initialized, and only `Initialize` fills it.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IParentalControlService {
    initialized: bool,
}

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `IParentalControlServiceFactory`'s commands.
    pub mod factory {
        /// `CreateService` (`{0, D<&IParentalControlServiceFactory::CreateService>,
        /// "CreateService"}`).
        pub const CREATE_SERVICE: u32 = 0;
        /// `CreateServiceWithoutInitialize` (`{1,
        /// D<&IParentalControlServiceFactory::CreateServiceWithoutInitialize>,
        /// "CreateServiceWithoutInitialize"}`).
        pub const CREATE_SERVICE_WITHOUT_INITIALIZE: u32 = 1;
    }

    /// `IParentalControlService`'s commands.
    pub mod service {
        /// `Initialize` (`{1, D<&IParentalControlService::Initialize>, "Initialize"}`).
        pub const INITIALIZE: u32 = 1;
    }
}

commands! {
    impl IParentalControlServiceFactory {
        command::factory::CREATE_SERVICE => create_service,
        command::factory::CREATE_SERVICE_WITHOUT_INITIALIZE => create_service_without_initialize,
    }
}

commands! {
    impl IParentalControlService {
        command::service::INITIALIZE => initialize,
    }
}

impl IParentalControlServiceFactory {
    /// `CreateService`. in: the caller's process id (not read). out: an
    /// [`IParentalControlService`], not initialized (module docs, "Where this differs from
    /// Eden").
    fn create_service(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::create(reply)
    }

    /// `CreateServiceWithoutInitialize`. in: the caller's process id (not read). out: an
    /// [`IParentalControlService`].
    fn create_service_without_initialize(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        Self::create(reply)
    }

    /// What both factory commands do: hand out a new service.
    fn create(reply: &mut Reply) -> Outcome {
        reply.push_session(Box::new(IParentalControlService::default()));
        reply.ok()
    }
}

impl IParentalControlService {
    /// `Initialize`. in: nothing. out: nothing; [`NO_CAPABILITY`] if the server's capability
    /// has neither [`Capability::APPLICATION`] nor [`Capability::SYSTEM`], which leaves the
    /// service as it was.
    fn initialize(
        this: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let capability = *cx.family::<Capability>()?;
        if !capability.intersects(INITIALIZE_NEEDS) {
            tracing::debug!(
                capability = capability.bits(),
                "pctl: Initialize without the Application or System capability"
            );
            return Err(NO_CAPABILITY.into());
        }
        this.initialized = true;
        reply.ok()
    }
}

/// The interfaces a parental-control server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IParentalControlServiceFactory>()
        .with::<IParentalControlService>()
}

/// The server for the parental-control name `name`, whose factory has
/// [`Name::capability`].
#[must_use]
pub fn server(name: Name, policy: UnknownPolicy) -> Server {
    server_with(name, name.capability(), policy)
}

/// A server named `name` whose capability is `capability`: what [`server`] builds, with the
/// capability given, so that tests can try one a name does not have.
fn server_with(name: Name, capability: Capability, policy: UnknownPolicy) -> Server {
    Server::new(name.as_str(), policy)
        .with_root(|| Box::new(IParentalControlServiceFactory))
        .with_interfaces(interfaces())
        // The restore check of the server's capability: a snapshot's is the server's (module
        // docs, "Snapshots").
        .with_checked_family(capability, |run, saved| {
            if run == saved {
                Ok(())
            } else {
                Err("its parental-control capability is not this server's")
            }
        })
}
