//! The HLE services a run has, and how they are put in front of the kernel: `sm:` on its named
//! port, and `set:sys` behind it (`docs/specs/services.md`, "Milestone 1 services").
//!
//! Each service's server is a [`starquake_services::framework::Server`], given to the kernel's
//! dispatcher as the handler of a server of its own ([`Dispatcher::add_server`]). `sm:`'s
//! registry names `set:sys` by the number the kernel knows its server by ([`SET_SYS`]), so a
//! `GetService("set:sys")` hands out a session on it.
//!
//! The numbers ([`SM`], [`SET_SYS`]) are starquake's own: Horizon names a service by its name
//! only. They appear in the trace, as the `service` of an IPC event; [`service_name`] says
//! which is which.
//!
//! Unknown commands and names are auto-stubbed ([`UnknownPolicy::AutoStub`]), the policy for
//! bringing up a title (`docs/specs/services.md`, decision 3). The firmware version `set:sys`
//! reports is Eden's ([`FirmwareVersion::eden_default`]).

use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::ipc::{IpcError, PortName, PortNameError};
use starquake_services::framework::UnknownPolicy;
use starquake_services::set::{self, FirmwareVersion, Settings};
use starquake_services::sm::{self, NameError, Registry, RegistryError, ServiceName};
use starquake_trace::ServiceId;

/// The number the kernel knows the service manager's server by.
pub const SM: ServiceId = ServiceId(1);
/// The number the kernel knows `set:sys`'s server by.
pub const SET_SYS: ServiceId = ServiceId(2);

/// The services a run has, with the names they are known by.
const SERVICES: [(ServiceId, &str); 2] = [(SM, sm::PORT_NAME), (SET_SYS, set::SYS_NAME)];

/// The name of the service a number stands for, if a run has it.
#[must_use]
pub fn service_name(service: ServiceId) -> Option<&'static str> {
    SERVICES
        .iter()
        .find(|(number, _)| *number == service)
        .map(|(_, name)| *name)
}

/// Why the services could not be set up.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ServicesError {
    /// The kernel refused a server or a port.
    #[error("the kernel refused a server: {0}")]
    Kernel(#[from] IpcError),
    /// A port name does not fit the kernel's.
    #[error(transparent)]
    PortName(#[from] PortNameError),
    /// A service name does not fit the service manager's.
    #[error(transparent)]
    ServiceName(#[from] NameError),
    /// The service manager refused a registration.
    #[error(transparent)]
    Registry(#[from] RegistryError),
}

/// Puts the run's services in front of `dispatcher`.
pub fn install(dispatcher: &Dispatcher) -> Result<(), ServicesError> {
    let policy = UnknownPolicy::AutoStub;
    let mut registry = Registry::new();
    registry.register(ServiceName::new(set::SYS_NAME)?, SET_SYS)?;

    let manager = sm::server(registry, policy).with_service_id(SM);
    let port = PortName::new(sm::PORT_NAME)?;
    dispatcher.add_server(sm::PORT_NAME, SM, Some(port), Box::new(manager))?;

    let settings = Settings {
        firmware: FirmwareVersion::eden_default(),
    };
    let settings = set::sys_server(settings, policy).with_service_id(SET_SYS);
    dispatcher.add_server(set::SYS_NAME, SET_SYS, None, Box::new(settings))?;
    Ok(())
}

#[cfg(test)]
mod tests;
