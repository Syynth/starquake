//! The HLE services a run has, and how they are put in front of the kernel: `sm:` on its named
//! port, and `set:sys`, `fsp-srv`, `apm`, `appletOE`, `hid` and `time:u` behind it
//! (`docs/specs/services.md`, "Milestone 1 services", and port plan slice 6; `apm`, `appletOE`,
//! `hid` and `time:u` as far as a default libnx start-up asks).
//!
//! Each service's server is a [`starquake_services::framework::Server`], given to the kernel's
//! dispatcher as the handler of a server of its own ([`Dispatcher::add_server`]). `sm:`'s
//! registry names the others by the numbers the kernel knows their servers by ([`SET_SYS`],
//! [`FSP_SRV`], [`APM`], [`APPLET_OE`], [`HID`], [`TIME_U`]), so a `GetService("set:sys")` hands
//! out a session on it.
//!
//! The numbers ([`SM`], [`SET_SYS`], [`FSP_SRV`], [`APM`], [`APPLET_OE`], [`HID`], [`TIME_U`]) are
//! starquake's own:
//! Horizon names a service by its name only. They appear in the trace, as the `service` of an IPC
//! event; [`service_name`] says which is which.
//!
//! Unknown commands and names are auto-stubbed ([`UnknownPolicy::AutoStub`]), the policy for
//! bringing up a title (`docs/specs/services.md`, decision 3). The firmware version `set:sys`
//! reports is Eden's ([`FirmwareVersion::eden_default`]), and so is what `appletOE` reports
//! ([`AppletSettings::eden_default`]). `time:u`'s clocks start from Eden's defaults
//! ([`TimeSettings::eden_default`]) and read the run's environment (`install`'s `env`).
//!
//! `fsp-srv` reads the host through the environment it is given ([`install`]'s `env`): the SD
//! card is the folder the run's configuration names, which [`HostFolder`] serves on the
//! environment's host I/O channel; a run that names none has no SD card. PROVISIONAL: the server
//! has no executor, so `Read` answers the framework's `NOT_SUPPORTED`. A read defers its reply;
//! the core loop delivers deferred replies a service says are ready (`crate::machine`,
//! "Deferred replies"), and an executor that says so when its job is done is not built yet.

use std::sync::Arc;

use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::ipc::{IpcError, PortName, PortNameError};
use starquake_services::am::{self, AppletSettings};
use starquake_services::apm;
use starquake_services::framework::UnknownPolicy;
use starquake_services::fsp;
use starquake_services::hid;
use starquake_services::set::{self, FirmwareVersion, Settings};
use starquake_services::sm::{self, NameError, Registry, RegistryError, ServiceName};
use starquake_services::time::{self, TimeSettings};
use starquake_trace::{Env, ServiceId};

mod host_folder;

pub use host_folder::HostFolder;

/// The number the kernel knows the service manager's server by.
pub const SM: ServiceId = ServiceId(1);
/// The number the kernel knows `set:sys`'s server by.
pub const SET_SYS: ServiceId = ServiceId(2);
/// The number the kernel knows `fsp-srv`'s server by.
pub const FSP_SRV: ServiceId = ServiceId(3);
/// The number the kernel knows `apm`'s server by.
pub const APM: ServiceId = ServiceId(4);
/// The number the kernel knows `appletOE`'s server by.
pub const APPLET_OE: ServiceId = ServiceId(5);
/// The number the kernel knows `hid`'s server by.
pub const HID: ServiceId = ServiceId(6);
/// The number the kernel knows `time:u`'s server by.
pub const TIME_U: ServiceId = ServiceId(7);

/// The services a run has, with the names they are known by.
const SERVICES: [(ServiceId, &str); 7] = [
    (SM, sm::PORT_NAME),
    (SET_SYS, set::SYS_NAME),
    (FSP_SRV, fsp::NAME),
    (APM, apm::NAME),
    (APPLET_OE, am::OE_NAME),
    (HID, hid::NAME),
    (TIME_U, time::NAME),
];

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

/// Puts the run's services in front of `dispatcher`. `env` is the environment the services read
/// the host through: the run's own, so that what they read is recorded with everything else.
pub fn install(dispatcher: &Dispatcher, env: Arc<dyn Env>) -> Result<(), ServicesError> {
    let policy = UnknownPolicy::AutoStub;
    let mut registry = Registry::new();
    registry.register(ServiceName::new(set::SYS_NAME)?, SET_SYS)?;
    registry.register(ServiceName::new(fsp::NAME)?, FSP_SRV)?;
    registry.register(ServiceName::new(apm::NAME)?, APM)?;
    registry.register(ServiceName::new(am::OE_NAME)?, APPLET_OE)?;
    registry.register(ServiceName::new(hid::NAME)?, HID)?;
    registry.register(ServiceName::new(time::NAME)?, TIME_U)?;

    let manager = sm::server(registry, policy).with_service_id(SM);
    let port = PortName::new(sm::PORT_NAME)?;
    dispatcher.add_server(sm::PORT_NAME, SM, Some(port), Box::new(manager))?;

    let settings = Settings {
        firmware: FirmwareVersion::eden_default(),
    };
    let settings = set::sys_server(settings, policy).with_service_id(SET_SYS);
    dispatcher.add_server(set::SYS_NAME, SET_SYS, None, Box::new(settings))?;

    let files = fsp::server(policy, Arc::clone(&env)).with_service_id(FSP_SRV);
    dispatcher.add_server(fsp::NAME, FSP_SRV, None, Box::new(files))?;

    let performance = apm::server(policy).with_service_id(APM);
    dispatcher.add_server(apm::NAME, APM, None, Box::new(performance))?;

    let applets = am::oe_server(AppletSettings::eden_default(), policy).with_service_id(APPLET_OE);
    dispatcher.add_server(am::OE_NAME, APPLET_OE, None, Box::new(applets))?;

    let input = hid::server(policy).with_service_id(HID);
    dispatcher.add_server(hid::NAME, HID, None, Box::new(input))?;

    let clocks = time::server(TimeSettings::eden_default(), policy, env).with_service_id(TIME_U);
    dispatcher.add_server(time::NAME, TIME_U, None, Box::new(clocks))?;
    Ok(())
}

#[cfg(test)]
mod tests;
