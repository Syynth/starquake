//! What a run's services leave behind for its report: the requests each was sent, the commands
//! each auto-stubbed, the service names nothing provides, and the last auto-stubbed requests
//! before the run ended (`docs/specs/observability.md`, the artifacts table: "counts per system
//! call and service command, every stubbed command hit"; the owner's ruling of 2026-11-17 on
//! issue #207: the unknown service names, every auto-stubbed command with its interface, and
//! the last ones in the crash report).
//!
//! - [`ServicesReport::read`] reads the servers once the run is over, through the kernel's
//!   [`Dispatcher::inspect_handlers`]: each handler that is a service framework [`Server`]
//!   gives its name, its [`Server::requests`], its [`Server::stubbed`], and, for `sm:`, its
//!   registry's [`Registry::unknown_names`]. A handler of another kind (a test's double) gives
//!   nothing.
//! - [`LastStubbed`] is a sink on the run's event stream that keeps the newest
//!   [`LAST_STUBBED`] IPC events marked [`IpcOutcome::Stubbed`], for the crash report. It keeps
//!   them itself, not from the flight recorder, so they are there however many other events
//!   came after them.
//!
//! Every name here is the services' own: a server's [`Server::name`], an interface's
//! [`InterfaceName`] as its trace event names it (for a session `sm:` handed out for a name
//! nothing provides, that name), a command's name in its table.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;

use serde::Serialize;
use starquake_kernel::dispatch::Dispatcher;
use starquake_services::framework::{Protocol, RequestCount, Server};
use starquake_services::sm::Registry;
use starquake_trace::{Event, EventKind, InterfaceName, IpcOutcome, IpcProtocol, ServiceId, Sink};

/// How many of the newest auto-stubbed requests [`LastStubbed`] keeps for the crash report.
///
/// starquake's own number: enough to show the run of stubs a title went through just before it
/// stopped, few enough to read at a glance.
pub const LAST_STUBBED: usize = 16;

/// The text a request's protocol has in a count's key: none for CMIF, and for TIPC
/// ` (tipc)`, since TIPC numbers its commands apart from CMIF. A control request's interface,
/// `IpcController`, already says it is one.
fn key_suffix(protocol: IpcProtocol) -> &'static str {
    match protocol {
        IpcProtocol::Tipc => " (tipc)",
        IpcProtocol::Cmif | IpcProtocol::Control => "",
    }
}

/// The key a count of requests has in `summary.json`'s `service_commands`:
/// `<service>:<interface>::<command>`, with the command's name when its table has one and its
/// id in decimal otherwise, and ` (tipc)` after a TIPC command (`set:sys:SetSys::GetFirmwareVersion`,
/// `sm::lm::5`, `sm::Sm::GetService (tipc)`).
#[must_use]
pub fn command_key(service: &str, count: &RequestCount) -> String {
    let command = count
        .name
        .map_or_else(|| count.command.id.to_string(), str::to_owned);
    format!(
        "{service}:{}::{command}{}",
        count.command.interface,
        key_suffix(count.command.protocol)
    )
}

/// The protocol of a stubbed command, as `summary.json` writes it.
const fn protocol_word(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Cmif => "cmif",
        Protocol::Tipc => "tipc",
    }
}

/// One command a server answered by auto-stub, in `summary.json`'s `stubbed_commands`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct StubbedCommand {
    /// The server's name (`sm:`, `appletOE`).
    pub service: String,
    /// The interface the command was sent to: for a session `sm:` handed out for a name nothing
    /// provides, that name; for a control command, `IpcController`.
    pub interface: String,
    /// `cmif` or `tipc` (a control command is `cmif`).
    pub protocol: &'static str,
    /// The command id.
    pub command: u32,
}

/// What the run's services left behind ([module docs](self)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServicesReport {
    /// Requests by `<service>:<interface>::<command>` ([`command_key`]), in key order.
    pub commands: BTreeMap<String, u64>,
    /// Requests the servers did not count by command because they had reached their bound of
    /// distinct commands ([`Server::uncounted_requests`]), summed over the servers.
    pub uncounted: u64,
    /// The commands the servers answered by auto-stub, in order of service, interface, protocol
    /// and id. Each server keeps at most
    /// [`MAX_STUBBED_COMMANDS`](starquake_services::framework::MAX_STUBBED_COMMANDS).
    pub stubbed: Vec<StubbedCommand>,
    /// The service names a title asked `sm:` for that nothing provides, in order: at most
    /// [`MAX_UNKNOWN_NAMES`](starquake_services::sm::MAX_UNKNOWN_NAMES).
    pub unknown_services: Vec<String>,
    /// Each server's name, by the number it is known by: how [`LastStubbed`]'s events are
    /// named.
    pub names: BTreeMap<ServiceId, String>,
}

impl ServicesReport {
    /// Reads what the servers in front of `dispatcher` left behind. A handler out of its slot
    /// cannot be read; that is logged, and it adds nothing. (Once every core and the timer
    /// thread have stopped, as when a run reads this, none is out.)
    #[must_use]
    pub fn read(dispatcher: &Dispatcher) -> Self {
        let mut report = Self::default();
        let mut unknown = BTreeSet::new();
        let out = dispatcher.inspect_handlers(|service, handler| {
            let handler: &dyn core::any::Any = handler;
            if let Some(server) = handler.downcast_ref::<Server>() {
                report.add(service, server, &mut unknown);
            }
        });
        if out > 0 {
            tracing::warn!(
                handlers = out,
                "servers whose handlers were busy are missing from the run's report"
            );
        }
        report.unknown_services = unknown.into_iter().collect();
        report.stubbed.sort();
        report
    }

    /// Adds one server's part.
    fn add(&mut self, service: ServiceId, server: &Server, unknown: &mut BTreeSet<String>) {
        let name = server.name();
        self.names.insert(service, name.to_owned());
        for count in server.requests() {
            let total = self.commands.entry(command_key(name, &count)).or_default();
            *total = total.saturating_add(count.count);
        }
        self.uncounted = self.uncounted.saturating_add(server.uncounted_requests());
        self.stubbed
            .extend(server.stubbed().map(|stubbed| StubbedCommand {
                service: name.to_owned(),
                interface: stubbed.interface.as_str().to_owned(),
                protocol: protocol_word(stubbed.protocol),
                command: stubbed.id,
            }));
        if let Some(registry) = server.family::<Registry>() {
            unknown.extend(
                registry
                    .unknown_names()
                    .map(|name| name.as_str().to_owned()),
            );
        }
    }

    /// The name of the server known as `service`, or its number in hex when the run has no
    /// server by that number.
    #[must_use]
    pub fn service_name(&self, service: ServiceId) -> String {
        self.names
            .get(&service)
            .cloned()
            .unwrap_or_else(|| format!("{:#x}", service.0))
    }

    /// `requests` ([`LastStubbed::requests`]) with their servers named, for the crash report.
    #[must_use]
    pub fn name_requests(&self, requests: &[StubbedRequest]) -> Vec<StubbedRequestReport> {
        requests
            .iter()
            .map(|request| StubbedRequestReport {
                seq: request.seq,
                thread: request.thread,
                service: self.service_name(request.service),
                interface: request.interface.as_str().to_owned(),
                protocol: request.protocol.word(),
                command: request.command,
            })
            .collect()
    }
}

/// One of the last auto-stubbed requests before a crash, as `crash.json` holds it
/// ([`CrashReport::last_stubbed`](crate::crash::CrashReport::last_stubbed)).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StubbedRequestReport {
    /// The event's place in the run's order.
    pub seq: u64,
    /// The guest thread that sent it.
    pub thread: u64,
    /// The server's name, or its number in hex ([`ServicesReport::service_name`]).
    pub service: String,
    /// The interface it went to: for a session `sm:` handed out for a name nothing provides,
    /// that name.
    pub interface: String,
    /// `cmif`, `tipc` or `control` ([`IpcProtocol::word`]).
    pub protocol: &'static str,
    /// The command id.
    pub command: u32,
}

impl StubbedRequestReport {
    /// The request in one line, as `crash.txt` writes it:
    /// `#40 thread 74: lm (on sm:) cmif command 5`.
    #[must_use]
    pub fn text(&self) -> String {
        format!(
            "#{} thread {}: {} (on {}) {} command {}",
            self.seq, self.thread, self.interface, self.service, self.protocol, self.command
        )
    }
}

/// One auto-stubbed request, as [`LastStubbed`] kept it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StubbedRequest {
    /// The event's place in the run's order.
    pub seq: u64,
    /// The guest thread that sent it.
    pub thread: u64,
    /// The server it went to.
    pub service: ServiceId,
    /// The interface it went to.
    pub interface: InterfaceName,
    /// The protocol it came in.
    pub protocol: IpcProtocol,
    /// The command id.
    pub command: u32,
}

/// The newest [`LAST_STUBBED`] auto-stubbed requests of the run ([module docs](self)).
#[derive(Debug, Default)]
pub struct LastStubbed {
    kept: Mutex<VecDeque<StubbedRequest>>,
}

impl LastStubbed {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// What it keeps, oldest first.
    #[must_use]
    pub fn requests(&self) -> Vec<StubbedRequest> {
        let kept = self
            .kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        kept.iter().copied().collect()
    }
}

impl Sink for LastStubbed {
    fn record(&self, event: &Event) {
        let EventKind::Ipc {
            service,
            interface,
            protocol,
            command,
            outcome: IpcOutcome::Stubbed,
            ..
        } = event.kind
        else {
            return;
        };
        let mut kept = self
            .kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if kept.len() >= LAST_STUBBED {
            kept.pop_front();
        }
        kept.push_back(StubbedRequest {
            seq: event.seq,
            thread: event.thread.0,
            service,
            interface,
            protocol,
            command,
        });
    }
}

#[cfg(test)]
mod tests;
