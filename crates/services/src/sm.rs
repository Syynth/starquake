//! `sm:`, the service manager: the named port a program asks for every other service
//! (`docs/specs/services.md`, "The service manager"; port plan slice 4).
//!
//! Ported from `eden/src/core/hle/service/sm/sm.cpp` and `sm.h` (`SM`, `ServiceManager`)
//! @ cf654052 (GPL-3.0-or-later).
//!
//! # What is here
//!
//! - [`Registry`]: the map from service names to the servers that provide them. It is the `sm:`
//!   family's state, so all `sm:` sessions share it and it is saved once in a snapshot. A
//!   provider today is always an HLE server, named by its number ([`Provider::Hle`]); a service a
//!   guest process registers (deferred, `docs/specs/kernel.md`) is one more kind of provider, and
//!   a caller of `GetService` cannot tell which kind it got.
//! - [`Sm`]: the interface behind an `sm:` session, with the commands below.
//! - [`Stub`]: the generic session an unknown name gets under the auto-stub policy.
//! - [`server`]: the `sm:` server, put together.
//!
//! # Commands
//!
//! Ids and names are Eden's table (`eden/src/core/hle/service/sm/sm.cpp`, `SM::SM`), the same in
//! CMIF and TIPC. The request layouts are what libnx sends (`libnx/nx/source/services/sm.c`).
//!
//! | Id | Command | In | Out |
//! |---|---|---|---|
//! | 0 | `Initialize` | the caller's process id (CMIF: `_smCmifCmdInPid`, a `u64` placeholder and the id in the header; TIPC: `_smTipcCmdInPid`, the id in the header only). Not read, as in Eden's `SM::Initialize` | nothing |
//! | 1 | `GetService` | the name, 8 bytes (`smGetServiceOriginal`: `SmServiceName`) | a moved session handle (`SfOutHandleAttr_HipcMove`; Eden: `PushMoveObjects`) |
//!
//! Left out, and so answered by the server's unknown-command policy: 2 `RegisterService`,
//! 3 `UnregisterService` (both need guest-hosted services, which are deferred), 4 `DetachClient`,
//! and the Atmosphère extensions 65000 to 65101, of which Eden implements only 65100.
//!
//! - **`Initialize`** marks the session initialised (Eden: `SetIsInitializedForSm` on the
//!   session's manager, which its clones share; here on the [`Sm`] object, which a cloned
//!   session shares too).
//! - **`GetService`** on a session that is not initialised fails with [`INVALID_CLIENT`]
//!   (Eden: `GetServiceImpl`, `!GetIsInitializedForSm()`), before the name is read.
//! - **The name** is read as Eden's `PopServiceName` reads it: of the 8 bytes, those from `' '`
//!   to `'~'` are kept, in order, and every other byte is dropped wherever it is. A name with
//!   nothing left is [`INVALID_SERVICE_NAME`] (Eden: `ValidateServiceName`, `name.empty()`; its
//!   other check, longer than 8, cannot happen). Guess: a console compares all 8 bytes, so a
//!   name with a byte after its first zero byte names something else there; libnx never sends
//!   one (`smEncodeName` pads with zeros).
//!
//! # Where this differs from Eden
//!
//! Each row is pinned by the test named in it, in `sm/tests.rs`.
//!
//! | Case | Eden | Here |
//! |---|---|---|
//! | `GetService` for a name nothing is registered under, strict policy | `GetServiceImpl` marks the request deferred and it is re-run whenever a service is registered (`ServerManager::OnDeferralEvent`); the caller waits until one is | reply [`NOT_REGISTERED`], the result Eden's `GetServicePort` returns for it (`an_unknown_name_is_an_error_under_strict`) |
//! | The same, auto-stub policy | as above | the name is logged and recorded ([`Registry::unknown_names`]) and the caller gets a session on a [`Stub`], which has no commands, so every command sent to it is auto-stubbed (`an_unknown_name_gets_a_stub_under_auto_stub`). This is the service spec's rule, after Eden-NX2's auto-stub |
//! | TIPC `GetService` that fails | `GetServiceTipc` replies the result and one moved handle slot holding zero | the reply holds the result only, like every failed reply of the framework. libnx reads no handle from a failed reply (`libnx/nx/include/switch/sf/tipc.h`, `tipcParseResponse`: `if (R_FAILED(rc)) return rc`) (`a_failed_tipc_get_service_carries_no_handle`) |
//! | The caller's handle table is full | the session is created, then `PushMoveObjects` fails to add its handle | no session is created ([`Kernel::connect`](crate::framework::Kernel::connect) is all or nothing); the reply is the kernel's "out of handles" (`a_full_handle_table_gets_no_session`) |
//!
//! Access control from the NPDM (which names a title may open) is not checked or logged yet:
//! the kernel does not yet tell a service which process is calling beyond its id. The service
//! spec wants it logged at first.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use starquake_kernel::result::ResultCode;
use starquake_trace::ServiceId;

use crate::commands;
use crate::framework::result::code;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// The name of the port the service manager listens on (`eden/src/core/hle/service/sm/sm.cpp`,
/// `LoopProcess`: `ManageNamedPort("sm:", ..)`).
pub const PORT_NAME: &str = "sm:";

/// `ErrorModule::SM` (`eden/src/core/hle/result.h`).
pub const SM_MODULE: u32 = 21;

/// `GetService` on a session that has not sent `Initialize`. Eden's `ResultInvalidClient`
/// (`ErrorModule::SM`, 2; `eden/src/core/hle/service/sm/sm.cpp`).
pub const INVALID_CLIENT: ResultCode = code(SM_MODULE, 2);
/// The name has no printable character. Eden's `ResultInvalidServiceName` (`ErrorModule::SM`, 6).
pub const INVALID_SERVICE_NAME: ResultCode = code(SM_MODULE, 6);
/// Nothing is registered under the name. Eden's `ResultNotRegistered` (`ErrorModule::SM`, 7).
pub const NOT_REGISTERED: ResultCode = code(SM_MODULE, 7);

/// The most bytes a service name has: the size of the name field in a request
/// (`eden/src/core/hle/service/sm/sm.cpp`, `PopServiceName`: `std::array<char, 8>`;
/// `ValidateServiceName`: `name.size() > 8`).
pub const MAX_NAME_LEN: usize = 8;

/// The most distinct unknown names a registry records.
///
/// starquake's own bound: names are the guest's to choose, so the record must not grow with
/// them. A name past the bound is still answered by the policy; it is just not recorded.
pub const MAX_UNKNOWN_NAMES: usize = 256;

/// Why a string is not a service name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NameError {
    /// The name is empty.
    #[error("a service name is not empty")]
    Empty,
    /// The name is longer than [`MAX_NAME_LEN`] bytes.
    #[error("service name {0:?} is longer than {MAX_NAME_LEN} bytes")]
    TooLong(String),
    /// The name has a byte outside `' '` to `'~'`.
    #[error("service name {0:?} has a character that is not printable ASCII")]
    NotPrintable(String),
}

/// A service's name: 1 to [`MAX_NAME_LEN`] bytes of printable ASCII (`' '` to `'~'`), which is
/// every name Eden's `PopServiceName` and `ValidateServiceName` let through.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ServiceName(String);

/// Whether a byte is one `PopServiceName` keeps: `c >= ' ' && c <= '~'`.
const fn is_name_byte(byte: u8) -> bool {
    matches!(byte, b' '..=b'~')
}

impl ServiceName {
    /// The name `name`, if it is one.
    pub fn new(name: &str) -> Result<Self, NameError> {
        if name.is_empty() {
            return Err(NameError::Empty);
        }
        if name.len() > MAX_NAME_LEN {
            return Err(NameError::TooLong(name.to_owned()));
        }
        if !name.bytes().all(is_name_byte) {
            return Err(NameError::NotPrintable(name.to_owned()));
        }
        Ok(Self(name.to_owned()))
    }

    /// The name in a request's name field, read as Eden's `PopServiceName` reads it: the
    /// printable bytes of the field, in order. `None` if there is none (Eden's
    /// `ValidateServiceName` then answers `ResultInvalidServiceName`).
    #[must_use]
    pub fn from_field(field: [u8; MAX_NAME_LEN]) -> Option<Self> {
        let name: String = field
            .into_iter()
            .filter(|b| is_name_byte(*b))
            .map(char::from)
            .collect();
        (!name.is_empty()).then_some(Self(name))
    }

    /// The name as a request carries it: its bytes, then zeros (`libnx/nx/include/switch/services/sm.h`,
    /// `smEncodeName`).
    #[must_use]
    pub fn to_field(&self) -> [u8; MAX_NAME_LEN] {
        let mut field = [0; MAX_NAME_LEN];
        for (slot, byte) in field.iter_mut().zip(self.0.bytes()) {
            *slot = byte;
        }
        field
    }

    /// The name as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ServiceName {
    type Error = NameError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::new(&name)
    }
}

impl From<ServiceName> for String {
    fn from(name: ServiceName) -> Self {
        name.0
    }
}

/// What provides a registered service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    /// An HLE server, by the number it is known to the kernel by
    /// ([`Server::with_service_id`]): `GetService` hands out a session on it.
    Hle(ServiceId),
}

/// Why a service could not be registered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    /// A service is already registered under the name. (To a guest this would be Eden's
    /// `ResultAlreadyRegistered`; registering is not a guest's to do yet.)
    #[error("a service is already registered as {0}")]
    AlreadyRegistered(ServiceName),
}

/// Why serialised registry data was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RegistryLayoutError {
    /// More unknown names are on record than a registry keeps.
    #[error(
        "{count} unknown names are on record, more than the {MAX_UNKNOWN_NAMES} a registry keeps"
    )]
    TooManyUnknownNames {
        /// The names in the data.
        count: usize,
    },
}

/// The service manager's map from names to providers. See the [module docs](self).
///
/// It is saved in a snapshot, so it derives `serde`. Deserialising goes through
/// [`RegistryData`], which checks the one bound [`Registry`]'s own operations keep,
/// [`MAX_UNKNOWN_NAMES`]; each name is checked as a [`ServiceName`] is.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RegistryData")]
pub struct Registry {
    services: BTreeMap<ServiceName, Provider>,
    /// Names that were asked for and that nothing provides. At most [`MAX_UNKNOWN_NAMES`].
    unknown: BTreeSet<ServiceName>,
}

/// The serialised form of a [`Registry`], and what deserialisation validates.
#[derive(Deserialize)]
struct RegistryData {
    services: BTreeMap<ServiceName, Provider>,
    unknown: BTreeSet<ServiceName>,
}

impl TryFrom<RegistryData> for Registry {
    type Error = RegistryLayoutError;

    fn try_from(data: RegistryData) -> Result<Self, Self::Error> {
        let count = data.unknown.len();
        if count > MAX_UNKNOWN_NAMES {
            return Err(RegistryLayoutError::TooManyUnknownNames { count });
        }
        Ok(Self {
            services: data.services,
            unknown: data.unknown,
        })
    }
}

impl Registry {
    /// A registry with no services.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the HLE server known as `service` under `name`.
    pub fn register(&mut self, name: ServiceName, service: ServiceId) -> Result<(), RegistryError> {
        if self.services.contains_key(&name) {
            return Err(RegistryError::AlreadyRegistered(name));
        }
        self.services.insert(name, Provider::Hle(service));
        Ok(())
    }

    /// What provides `name`, if anything is registered under it.
    #[must_use]
    pub fn provider(&self, name: &ServiceName) -> Option<Provider> {
        self.services.get(name).copied()
    }

    /// The registered names, in order.
    pub fn names(&self) -> impl Iterator<Item = &ServiceName> {
        self.services.keys()
    }

    /// The names `GetService` was asked for that nothing provides, in order: the services a
    /// title wants that nobody has written. At most [`MAX_UNKNOWN_NAMES`] are kept.
    pub fn unknown_names(&self) -> impl Iterator<Item = &ServiceName> {
        self.unknown.iter()
    }

    /// Records a name nothing provides. Returns whether it is the first time the name was
    /// recorded.
    fn note_unknown(&mut self, name: &ServiceName) -> bool {
        if self.unknown.contains(name) || self.unknown.len() >= MAX_UNKNOWN_NAMES {
            return false;
        }
        self.unknown.insert(name.clone())
    }
}

/// The interface behind an `sm:` session.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Sm {
    /// Whether `Initialize` was sent on the session.
    initialized: bool,
}

commands! {
    impl Sm {
        0 => initialize,
        1 => get_service,
        tipc {
            0 => initialize,
            1 => get_service,
        }
    }
}

impl Sm {
    /// `Initialize`. in: the caller's process id, not read. out: nothing.
    fn initialize(
        &mut self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        self.initialized = true;
        reply.ok()
    }

    /// `GetService`. in: `char name[8]`. out: a moved session handle.
    fn get_service(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if !self.initialized {
            return Err(INVALID_CLIENT.into());
        }
        let name = ServiceName::from_field(req.payload().read_array()?);
        let name = name.ok_or(INVALID_SERVICE_NAME)?;

        let policy = cx.policy();
        let registry = cx.family::<Registry>()?;
        if let Some(Provider::Hle(service)) = registry.provider(&name) {
            reply.push_service(service);
            return reply.ok();
        }
        if registry.note_unknown(&name) {
            tracing::warn!(%name, ?policy, "a service was asked for that nothing provides");
        }
        match policy {
            UnknownPolicy::Strict => Err(NOT_REGISTERED.into()),
            UnknownPolicy::AutoStub => {
                reply.push_session(Box::new(Stub { name }));
                reply.ok()
            }
        }
    }
}

/// The session `GetService` hands out for a name nothing provides, under the auto-stub policy:
/// an interface with no commands, so that every command a title sends it is answered by the
/// server's auto-stub and shows up in the trace and in [`Server::stubbed`]. It is a session of
/// the `sm:` server.
#[derive(Debug, Serialize, Deserialize)]
pub struct Stub {
    /// The name that was asked for: kept so that a snapshot says what each stub stands in for.
    name: ServiceName,
}

commands! {
    impl Stub {}
}

/// The interfaces an `sm:` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new().with::<Sm>().with::<Stub>()
}

/// The `sm:` server: every session it accepts gets an [`Sm`] of its own, and they share
/// `registry`. `policy` decides what an unknown name and an unknown command get.
#[must_use]
pub fn server(registry: Registry, policy: UnknownPolicy) -> Server {
    Server::new(PORT_NAME, policy)
        .with_root(|| Box::new(Sm::default()))
        .with_interfaces(interfaces())
        .with_family(registry)
}
