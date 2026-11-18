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
//! The graphics server's root ([`GSV_ROOT`], `docs/specs/graphics-server.md`, "The root") is a
//! private entry of the registry ([`Registry::register_private`]): `GetService` never hands it
//! out, and `appletOE`'s `ISelfController` 2000 is given the same number, the constant
//! [`GSV_ROOT`] that the private entry is registered under, to hand a session on it to the
//! title. The root hands out sessions on the graphics server's other servers, [`GSV_SYNCPT`],
//! [`GSV_GPU`], [`GSV_NVSRV`], [`GSV_VI2`] and [`GSV_VIDEO`], which are in no registry;
//! [`graphics`] builds the graphics server and says what it wires.
//!
//! Parental control's four names (`pctl`, `pctl:a`, `pctl:s`, `pctl:r`) are four servers, one
//! per name, known by [`PCTL`], [`PCTL_A`], [`PCTL_S`] and [`PCTL_R`]: each name's factory has a
//! capability of its own (`starquake_services::pctl`, "Capabilities").
//!
//! `lm`, `ectx:aw` and `aoc:u` are one server each, known by [`LM`], [`ECTX_AW`] and [`AOC_U`].
//! `lm`'s loggers write each finished message's text to the guest log [`install_with_guest_log`]
//! is given (the run's, in a run); [`install`] gives them one that keeps nothing.
//!
//! The numbers ([`SM`], [`SET_SYS`], [`FSP_SRV`], [`APM`], [`APPLET_OE`], [`HID`], [`TIME_U`],
//! [`GSV_ROOT`] and the others above) are starquake's own:
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
//! environment's host I/O channel, and the data storage is the run's title's RomFS; [`FspHost`]
//! serves both. A run that names no folder has no SD card, and one with no title RomFS no data
//! storage.
//!
//! # Reads
//!
//! `IFile::Read` and `IStorage::Read` defer their reply and hand the read to `fsp-srv`'s
//! executor ([`Reads::executor`]): in a run, `fsp-srv`'s workers
//! (`starquake_services::fsp::worker::Workers`), whose jobs read the title's RomFS through the
//! loader's positional source behind [`FspHost`], each worker on an environment of its own.
//! When a job is finished, its completion reaches the `fsp-srv` server's mailbox, and the
//! server's notification tells [`Reads::ready`] the server's number in the kernel; the core
//! loop then delivers the reply (`crate::machine`, "Deferred replies"). What is recorded, and
//! how a replay delivers the reads in the recorded order, is the worker module's
//! (`starquake_services::fsp::worker`, "What is recorded").

use std::sync::{Arc, OnceLock};

use starquake_gsv::{gpu, nvsrv, root, syncpt, vi2, video};
use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::id::ServerId;
use starquake_kernel::ipc::{IpcError, PortName, PortNameError};
use starquake_services::am::{self, Applet, AppletSettings};
use starquake_services::apm;
use starquake_services::framework::{Executor, Halt, Notify, UnknownPolicy};
use starquake_services::fsp;
use starquake_services::hid;
use starquake_services::pctl;
use starquake_services::set::{self, FirmwareVersion, Settings};
use starquake_services::sm::{self, NameError, Registry, RegistryError, ServiceName};
use starquake_services::time::{self, TimeSettings};
use starquake_services::{aoc, ectx, lm};
use starquake_trace::{Env, ServiceId};

mod fsp_host;
pub mod graphics;
mod host_folder;

pub use fsp_host::FspHost;
pub use graphics::{FrontMode, Graphics};
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
/// The number the kernel knows the graphics server's root's server by.
pub const GSV_ROOT: ServiceId = ServiceId(8);
/// The number the kernel knows `pctl`'s server by.
pub const PCTL: ServiceId = ServiceId(12);
/// The number the kernel knows `pctl:a`'s server by.
pub const PCTL_A: ServiceId = ServiceId(13);
/// The number the kernel knows `pctl:s`'s server by.
pub const PCTL_S: ServiceId = ServiceId(14);
/// The number the kernel knows `pctl:r`'s server by.
pub const PCTL_R: ServiceId = ServiceId(15);
/// The number the kernel knows `lm`'s server by.
pub const LM: ServiceId = ServiceId(16);
/// The number the kernel knows `ectx:aw`'s server by.
pub const ECTX_AW: ServiceId = ServiceId(17);
/// The number the kernel knows `aoc:u`'s server by.
pub const AOC_U: ServiceId = ServiceId(18);
/// The number the kernel knows the graphics server's syncpt server by.
pub const GSV_SYNCPT: ServiceId = ServiceId(19);
/// The number the kernel knows the graphics server's gpu server by.
pub const GSV_GPU: ServiceId = ServiceId(20);
/// The number the kernel knows the graphics server's nvsrv server by.
pub const GSV_NVSRV: ServiceId = ServiceId(21);
/// The number the kernel knows the graphics server's vi2 server by.
pub const GSV_VI2: ServiceId = ServiceId(22);
/// The number the kernel knows the graphics server's video server by.
pub const GSV_VIDEO: ServiceId = ServiceId(23);

/// The parental-control names, each with the number its server is known by: one server per
/// name, since each name's factory has its own capability ([`pctl::Name::capability`]).
const PCTL_SERVICES: [(ServiceId, pctl::Name); 4] = [
    (PCTL, pctl::Name::Pctl),
    (PCTL_A, pctl::Name::A),
    (PCTL_S, pctl::Name::S),
    (PCTL_R, pctl::Name::R),
];

/// The services a run has, with the names they are known by.
const SERVICES: &[(ServiceId, &str)] = &[
    (SM, sm::PORT_NAME),
    (SET_SYS, set::SYS_NAME),
    (FSP_SRV, fsp::NAME),
    (APM, apm::NAME),
    (APPLET_OE, am::OE_NAME),
    (HID, hid::NAME),
    (TIME_U, time::NAME),
    (GSV_ROOT, root::ROOT_NAME),
    (PCTL, pctl::Name::Pctl.as_str()),
    (PCTL_A, pctl::Name::A.as_str()),
    (PCTL_S, pctl::Name::S.as_str()),
    (PCTL_R, pctl::Name::R.as_str()),
    (LM, lm::NAME),
    (ECTX_AW, ectx::AW_NAME),
    (AOC_U, aoc::U_NAME),
    (GSV_SYNCPT, syncpt::SYNCPT_NAME),
    (GSV_GPU, gpu::GPU_NAME),
    (GSV_NVSRV, nvsrv::NVSRV_NAME),
    (GSV_VI2, vi2::VI2_NAME),
    (GSV_VIDEO, video::VIDEO_NAME),
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

/// Told when a server has deferred replies ready to deliver: the core loop's, in a run
/// (`crate::machine`, "Deferred replies"). It is called on the thread that completed or dropped
/// a reply token, from the server's notification, and must not block. For `fsp-srv`'s reads that
/// is the executor's finish step, which runs with a lock of the executor held: on one of the
/// workers (`starquake_services::fsp::worker::Workers`), the workers' record lock; in a replay
/// (`starquake_services::fsp::worker::Replay`), the replay's state lock, on the thread that
/// submitted the job. So it must not call `Workers::take_record` or submit a job to the
/// server's executor, which wait for those locks.
pub trait Deliveries: Send + Sync {
    /// The deferred replies of the server the kernel numbers `server` are ready.
    fn ready(&self, server: ServerId);
}

/// What `fsp-srv`'s reads need from the run (module docs, "Reads").
#[derive(Clone)]
pub struct Reads {
    /// Where the reads run: `fsp-srv`'s workers in a run.
    pub executor: Arc<dyn Executor>,
    /// Told when a read's reply is ready.
    pub ready: Arc<dyn Deliveries>,
    /// Told when the environment fails a service, so that the run stops
    /// (`starquake_services::framework::halt`): while one of `fsp-srv`'s handlers runs or one of
    /// its reads does, and while `time:u`'s handler does ([`install`] gives it to both servers).
    /// The run's own, in a run. The executor is given its own when it is made.
    pub halt: Arc<dyn Halt>,
}

/// The notification of a server whose number the kernel gives it only once it has been added:
/// it passes each completion on to [`Deliveries`] under that number.
struct Delivering {
    server: OnceLock<ServerId>,
    ready: Arc<dyn Deliveries>,
}

impl Notify for Delivering {
    fn completion_posted(&self) {
        if let Some(server) = self.server.get() {
            self.ready.ready(*server);
        } else {
            // `install` names the server before it returns, and no request reaches a server
            // before then.
            tracing::error!("a reply is ready on a server the kernel has not numbered");
        }
    }
}

/// A guest log that keeps nothing: what [`install`] gives `lm`'s loggers.
struct NoGuestLog;

impl lm::GuestLog for NoGuestLog {
    fn line(&self, _text: &[u8]) {}
}

/// Puts the run's services in front of `dispatcher`, as [`install_with_guest_log`] does, with a
/// guest log that keeps nothing: the text `lm`'s loggers are sent is dropped.
pub fn install(
    dispatcher: &Dispatcher,
    env: Arc<dyn Env>,
    reads: Reads,
) -> Result<(), ServicesError> {
    install_with_guest_log(dispatcher, env, reads, Arc::new(NoGuestLog))
}

/// Puts the run's services in front of `dispatcher`. `env` is the environment the services'
/// methods read the host through while they handle a request: the run's own, so that what they
/// read there is recorded with everything else. `fsp-srv`'s reads do not use it: their jobs read
/// through the environments of `reads`' executor (module docs, "Reads"). `reads` is where
/// `fsp-srv`'s reads run, and who is told when one is done. `guest_log` is where `lm`'s loggers
/// write each finished message's text: the run's guest log, in a run
/// (`starquake_services::lm`, "Where the messages go").
pub fn install_with_guest_log(
    dispatcher: &Dispatcher,
    env: Arc<dyn Env>,
    reads: Reads,
    guest_log: Arc<dyn lm::GuestLog>,
) -> Result<(), ServicesError> {
    let graphics = Graphics::new(FrontMode::Live, None);
    install_with_graphics(dispatcher, env, reads, guest_log, &graphics)
}

/// Puts the run's services in front of `dispatcher`, as [`install_with_guest_log`] does, with
/// `graphics` as the graphics server: its servers are added, and the caller keeps it to run its
/// timer thread's turn ([`Graphics::turn`]) and, in a replay, feed its notes back
/// ([`Graphics::replay`]). [`install_with_guest_log`] gives one whose queue rings nothing and
/// whose turns nothing runs: its requests are answered, and what waits for the GPU or for a
/// turn never completes.
pub fn install_with_graphics(
    dispatcher: &Dispatcher,
    env: Arc<dyn Env>,
    reads: Reads,
    guest_log: Arc<dyn lm::GuestLog>,
    graphics: &Graphics,
) -> Result<(), ServicesError> {
    let policy = UnknownPolicy::AutoStub;
    let mut registry = Registry::new();
    registry.register(ServiceName::new(set::SYS_NAME)?, SET_SYS)?;
    registry.register(ServiceName::new(fsp::NAME)?, FSP_SRV)?;
    registry.register(ServiceName::new(apm::NAME)?, APM)?;
    registry.register(ServiceName::new(am::OE_NAME)?, APPLET_OE)?;
    registry.register(ServiceName::new(hid::NAME)?, HID)?;
    registry.register(ServiceName::new(time::NAME)?, TIME_U)?;
    registry.register_private(ServiceName::new(root::ROOT_NAME)?, GSV_ROOT)?;
    for (number, name) in PCTL_SERVICES {
        registry.register(ServiceName::new(name.as_str())?, number)?;
    }
    registry.register(ServiceName::new(lm::NAME)?, LM)?;
    registry.register(ServiceName::new(ectx::AW_NAME)?, ECTX_AW)?;
    registry.register(ServiceName::new(aoc::U_NAME)?, AOC_U)?;

    let manager = sm::server(registry, policy).with_service_id(SM);
    let port = PortName::new(sm::PORT_NAME)?;
    dispatcher.add_server(sm::PORT_NAME, SM, Some(port), Box::new(manager))?;

    let settings = Settings {
        firmware: FirmwareVersion::eden_default(),
    };
    let settings = set::sys_server(settings, policy).with_service_id(SET_SYS);
    dispatcher.add_server(set::SYS_NAME, SET_SYS, None, Box::new(settings))?;

    let delivering = Arc::new(Delivering {
        server: OnceLock::new(),
        ready: Arc::clone(&reads.ready),
    });
    let notify: Arc<dyn Notify> = Arc::<Delivering>::clone(&delivering);
    let files = fsp::server(policy, Arc::clone(&env), Arc::clone(&reads.halt))
        .with_service_id(FSP_SRV)
        .with_executor(reads.executor)
        .with_notify(notify);
    let files = dispatcher.add_server(fsp::NAME, FSP_SRV, None, Box::new(files))?;
    if delivering.server.set(files).is_err() {
        tracing::error!("fsp-srv's server was numbered twice");
    }

    let performance = apm::server(policy).with_service_id(APM);
    dispatcher.add_server(apm::NAME, APM, None, Box::new(performance))?;

    // `appletOE` hands out sessions on `GSV_ROOT`, the number the registry's private entry holds.
    let applet = Applet::new(AppletSettings::eden_default()).with_graphics_server(GSV_ROOT);
    let applets = am::oe_server_for(applet, policy).with_service_id(APPLET_OE);
    dispatcher.add_server(am::OE_NAME, APPLET_OE, None, Box::new(applets))?;

    graphics.add_servers(dispatcher, policy, reads.ready)?;
    graphics.add_vi2_server(
        dispatcher,
        policy,
        Arc::clone(&env),
        Arc::clone(&reads.halt),
    )?;

    let input = hid::server(policy).with_service_id(HID);
    dispatcher.add_server(hid::NAME, HID, None, Box::new(input))?;

    let clocks =
        time::server(TimeSettings::eden_default(), policy, env, reads.halt).with_service_id(TIME_U);
    dispatcher.add_server(time::NAME, TIME_U, None, Box::new(clocks))?;

    for (number, name) in PCTL_SERVICES {
        let parental = pctl::server(name, policy).with_service_id(number);
        dispatcher.add_server(name.as_str(), number, None, Box::new(parental))?;
    }

    let loggers = lm::server(policy, guest_log).with_service_id(LM);
    dispatcher.add_server(lm::NAME, LM, None, Box::new(loggers))?;

    let contexts = ectx::aw_server(policy).with_service_id(ECTX_AW);
    dispatcher.add_server(ectx::AW_NAME, ECTX_AW, None, Box::new(contexts))?;

    let add_ons = aoc::u_server(policy).with_service_id(AOC_U);
    dispatcher.add_server(aoc::U_NAME, AOC_U, None, Box::new(add_ons))?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
