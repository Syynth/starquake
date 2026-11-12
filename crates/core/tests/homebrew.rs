//! The homebrew fixture (`tests/homebrew/hello/hello.nro`, a real libnx program) against the
//! reference emulator: its run in starquake is compared, line for line, with the normalised
//! trace of the same file under Eden-NX2 (`tests/homebrew/hello/reference/startup.trace`; the
//! README beside it has the provenance and the format), from its first system call to
//! `ExitProcess`.
//!
//! Each line of the reference is one of three kinds, and matches when:
//!
//! - **`svc`**: its number, its decoded arguments and its result are the reference's.
//!   Addresses are compared as `anchor+offset` (the module, or a region of the address space),
//!   so the two emulators may place the module and the regions differently but must agree on
//!   everything inside them. Handles are compared as a mapping: the first time a reference
//!   handle value appears, the run's handle in the same place is paired with it, and from then
//!   on each must appear only with the other (pseudo-handles and zero are compared as values).
//!   `ExitProcess` has no result in the reference, which never saw it return; starquake ends the
//!   run there (`docs/reference-emulator.md`). Outputs other than the result are not in
//!   starquake's event stream; what `GetInfo` answers is checked against the same reference in
//!   `starquake-kernel` (`svc/info/tests.rs`, `memory/layout.rs`).
//! - **`ipc`**: the service, by the name starquake registered its number under
//!   (`starquake_core::hle::service_name`), and the command id are the reference's, and the
//!   reference's name for the command is the one starquake's command table lists, written in
//!   the reference's case. A reference control request (`IpcController::3`) is a request with
//!   that id on the session of the service the reference's next non-control request goes to:
//!   starquake's IPC event does not say whether a request was a control request
//!   (`starquake_services::framework::trace`), so this is the most the comparison can check.
//! - **`log`**: the run logged a line of the same length and hash (an event keeps no text).
//!   The text itself is checked against the fixture's source by the run's guest log.
//!
//! The comparison runs on every host, against made-up runs that must fail it: a wrong
//! argument, result, service or command, a log line that differs, a handle used for two
//! sessions, a missing line, a run that stops early. The run on a real vCPU needs a Mac with
//! Apple Silicon and skips itself only when `CI` is set.

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod common;

/// Everything is in a `cfg(test)` module so that the helpers may unwrap like the tests do.
#[cfg(test)]
mod trace {
    use std::collections::BTreeMap;

    use starquake_core::hle::{SET_SYS, SM, service_name};
    use starquake_kernel::handle::Handle;
    use starquake_kernel::memory::MemoryArrangement;
    use starquake_kernel::memory::layout::{LayoutRequest, Region, RegionOffsets, Regions};
    use starquake_kernel::svc::{Kind, SvcId};
    use starquake_services::framework::control::{CONTROL_COMMANDS, CONTROL_INTERFACE};
    use starquake_services::framework::{Protocol, Session};
    use starquake_services::set::SetSys;
    use starquake_services::sm::Sm;
    use starquake_trace::{EventKind, SVC_ARG_COUNT, ServiceId};

    /// The reference emulator's run, normalised.
    const REFERENCE: &str = include_str!("../../../tests/homebrew/hello/reference/startup.trace");

    /// The fixture, and the source it was built from.
    const HELLO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nro");
    const HELLO_SOURCE: &str = include_str!("../../../tests/homebrew/hello/hello.c");

    /// Where an NRO's header keeps the size of its image (`starquake_loader::nro`; the same offset
    /// `normalise.py` reads).
    const NRO_IMAGE_SIZE_OFFSET: usize = 0x18;

    /// The arguments that are addresses, for each call the fixture makes: the same list as
    /// `ADDRESSES` in `tests/homebrew/hello/reference/normalise.py`. A call that is not listed is
    /// refused, so no address is compared as a number by oversight.
    const ADDRESS_ARGUMENTS: &[(SvcId, &[&str])] = &[
        (SvcId::SetHeapSize, &[]),
        (SvcId::SetMemoryPermission, &["address"]),
        (SvcId::SetMemoryAttribute, &["address"]),
        (SvcId::QueryMemory, &["out_memory_info", "address"]),
        (SvcId::GetInfo, &[]),
        (SvcId::ConnectToNamedPort, &["name"]),
        (SvcId::SendSyncRequest, &[]),
        (SvcId::CloseHandle, &[]),
        (SvcId::OutputDebugString, &["debug_str"]),
        (SvcId::ExitProcess, &[]),
    ];

    /// One line of a run: what the event stream keeps of a system call, a request, or a line
    /// the guest logged.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Line {
        Svc {
            number: u32,
            /// The decoded arguments, each in the slot of its register.
            args: [u64; SVC_ARG_COUNT],
            result: u32,
        },
        Ipc {
            service: ServiceId,
            command: u32,
        },
        Log {
            len: u32,
            hash: u64,
        },
    }

    impl Line {
        /// The line an event is, if it is one of the three kinds.
        fn of(kind: &EventKind) -> Option<Self> {
            match *kind {
                EventKind::Svc { id, args, result } => Some(Self::Svc {
                    number: id,
                    args,
                    result: result.0,
                }),
                EventKind::Ipc {
                    service, command, ..
                } => Some(Self::Ipc { service, command }),
                EventKind::GuestLog { len, hash } => Some(Self::Log { len, hash }),
                _ => None,
            }
        }
    }

    /// Where a run put things, for writing its addresses as `anchor+offset`.
    #[derive(Debug, Clone, Copy)]
    struct Anchors {
        /// The module's image: its first byte to the first byte after it, inclusive (libnx asks
        /// about that address).
        main: Region,
        regions: Regions,
    }

    impl Anchors {
        /// The anchors of a run whose module is `module_size` bytes at `base`, of which the
        /// first `image_size` are its image, on a 4 GiB console: the regions are where the kernel
        /// places them around that code.
        fn new(base: u64, image_size: u64, module_size: u64) -> Self {
            let regions = Regions::new(&LayoutRequest {
                code_address: base,
                code_size: module_size,
                physical_memory_size: MemoryArrangement::FourGb.physical_memory_size(),
                alias_region_extra: false,
                offsets: RegionOffsets::default(),
            })
            .expect("the module lies in the code region");
            Self {
                main: Region {
                    start: base,
                    end: base + image_size,
                },
                regions,
            }
        }

        /// `address` as `anchor+offset`: the module first, then the regions, the ASLR region last
        /// because it contains the others (the order `normalise.py` uses).
        fn symbol(&self, address: u64) -> Result<String, String> {
            if (self.main.start..=self.main.end).contains(&address) {
                return Ok(format!("main+{:#x}", address - self.main.start));
            }
            let regions = [
                ("stack_region", self.regions.stack),
                ("alias_region", self.regions.alias),
                ("heap_region", self.regions.heap),
                ("aslr_region", self.regions.code),
            ];
            regions
                .into_iter()
                .find(|(_, region)| region.holds(address, 1))
                .map(|(name, region)| format!("{name}+{:#x}", address - region.start))
                .ok_or_else(|| format!("address {address:#x} lies in no known region"))
        }
    }

    /// The pairing of the reference's handle values with the run's (see the module docs).
    #[derive(Debug, Default)]
    struct Handles {
        ours: BTreeMap<u64, String>,
        theirs: BTreeMap<String, u64>,
    }

    impl Handles {
        /// The run's handle `ours`, written as the reference handle it stands for. `expected` is
        /// the value the reference has in its place, which pairs the two the first time either
        /// is seen.
        fn text(&mut self, ours: u64, expected: Option<&str>) -> String {
            let literal = format!("{ours:#x}");
            if ours == 0 || Handle::from_register(ours).is_pseudo() {
                return literal;
            }
            if let Some(paired) = self.ours.get(&ours) {
                return paired.clone();
            }
            let Some(expected) = expected else {
                return literal;
            };
            if self.theirs.contains_key(expected) {
                // The reference handle already stands for another of ours.
                return literal;
            }
            self.ours.insert(ours, expected.to_owned());
            self.theirs.insert(expected.to_owned(), ours);
            expected.to_owned()
        }
    }

    /// The value a reference line gives argument `name`: `name=<value>` inside the parentheses.
    fn argument<'a>(line: &'a str, name: &str) -> Option<&'a str> {
        let (_, args) = line.split_once('(')?;
        let (args, _) = args.split_once(')')?;
        args.split(", ")
            .filter_map(|arg| arg.split_once('='))
            .find(|(arg, _)| *arg == name)
            .map(|(_, value)| value)
    }

    /// The part of a system call's line that names the call and its arguments:
    /// `svc 0x02 SetMemoryPermission(address=main+0x4000, size=0x1000, perm=0x1)`.
    fn call_text(
        number: u32,
        args: &[u64; SVC_ARG_COUNT],
        anchors: &Anchors,
        handles: &mut Handles,
        expected: &str,
    ) -> Result<String, String> {
        let id = u16::try_from(number)
            .ok()
            .and_then(SvcId::from_number)
            .ok_or_else(|| format!("svc {number:#x} is not a listed call"))?;
        let convention = id
            .convention()
            .ok_or_else(|| format!("{} has no register convention", id.name()))?;
        let (_, addresses) = ADDRESS_ARGUMENTS
            .iter()
            .find(|(listed, _)| *listed == id)
            .ok_or_else(|| format!("no address rule for {}", id.name()))?;
        let mut arguments = Vec::new();
        for input in convention.inputs {
            let value = args[usize::from(input.reg)];
            let text = if addresses.contains(&input.name) {
                anchors.symbol(value)?
            } else if input.kind == Kind::Handle {
                handles.text(value, argument(expected, input.name))
            } else {
                format!("{value:#x}")
            };
            arguments.push(format!("{}={text}", input.name));
        }
        Ok(format!(
            "svc {number:#04x} {}({})",
            id.name(),
            arguments.join(", ")
        ))
    }

    /// The interface whose command table names a service's commands.
    fn interface(service: ServiceId) -> Option<Box<dyn Session>> {
        match service {
            SM => Some(Box::new(Sm::default())),
            SET_SYS => Some(Box::new(SetSys)),
            _ => None,
        }
    }

    /// A command table's name (a method's, `get_service`) in the reference's case
    /// (`GetService`).
    fn camel(name: &str) -> String {
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_uppercase().chain(chars).collect()
                })
            })
            .collect()
    }

    /// The reference's lines, without its comments and without each call's outputs, with every
    /// control request (`IpcController::3`) written as a request to the service the next
    /// non-control request goes to, and its command's name kept apart:
    /// `(line, Some(name))` for a request.
    fn reference() -> Vec<(String, Option<String>)> {
        let lines: Vec<&str> = REFERENCE
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        let mut out = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if let Some(call) = line.strip_prefix("svc ") {
                // Outputs follow the result after a space; a call with no result has no `->`.
                let text = match call.split_once(") -> ") {
                    Some((call, rest)) => {
                        let result = rest.split(' ').next().expect("a result");
                        format!("svc {call}) -> {result}")
                    }
                    None => (*line).to_owned(),
                };
                out.push((text, None));
            } else if let Some(request) = line.strip_prefix("ipc ") {
                let (target, name) = request.split_once(" (").expect("a named request");
                let name = name.strip_suffix(')').expect("a closing parenthesis");
                let (service, command) = target.rsplit_once("::").expect("service::command");
                let service = if service == CONTROL_INTERFACE {
                    lines[index + 1..]
                        .iter()
                        .filter_map(|line| line.strip_prefix("ipc "))
                        .map(|next| next.rsplit_once("::").expect("service::command").0)
                        .find(|next| *next != CONTROL_INTERFACE)
                        .expect("a request after the control request")
                } else {
                    service
                };
                out.push((format!("ipc {service}::{command}"), Some(name.to_owned())));
            } else {
                out.push(((*line).to_owned(), None));
            }
        }
        out
    }

    /// One line of the run, written as the reference writes it.
    fn text(
        line: &Line,
        anchors: &Anchors,
        handles: &mut Handles,
        expected: &str,
    ) -> Result<String, String> {
        match *line {
            Line::Svc {
                number,
                args,
                result,
            } => {
                let call = call_text(number, &args, anchors, handles, expected)?;
                if number == u32::from(SvcId::ExitProcess.number()) {
                    // The reference never saw it return.
                    return Ok(call);
                }
                Ok(format!("{call} -> {result:#x}"))
            }
            Line::Ipc { service, command } => {
                let name = service_name(service)
                    .ok_or_else(|| format!("service {service:?} has no name"))?;
                Ok(format!("ipc {name}::{command}"))
            }
            Line::Log { len, hash } => {
                // The reference's text, if it is this line's: an event keeps only the
                // length and the hash.
                let text = expected.strip_prefix("log ").unwrap_or_default();
                if EventKind::guest_log(text.as_bytes()) == (EventKind::GuestLog { len, hash }) {
                    return Ok(expected.to_owned());
                }
                Ok(format!("log len={len} hash={hash:#x}"))
            }
        }
    }

    /// The reference's name for a request, checked against starquake's command tables: the
    /// control commands' for a control request, the service's otherwise.
    fn check_name(line: &Line, name: &str, control: bool) -> Result<(), String> {
        let Line::Ipc { service, command } = *line else {
            return Ok(());
        };
        let ours = if control {
            CONTROL_COMMANDS
                .name(Protocol::Cmif, command)
                .map(str::to_owned)
        } else {
            interface(service)
                .and_then(|interface| interface.commands().name(Protocol::Cmif, command))
                .map(camel)
        };
        match ours {
            Some(ours) if ours == name => Ok(()),
            ours => Err(format!(
                "the reference calls command {command} {name}, starquake {ours:?}"
            )),
        }
    }

    /// Compares a run's lines with the reference's, in order, through `ExitProcess`. The error
    /// names the first line that differs.
    fn compare(ours: &[Line], anchors: &Anchors) -> Result<(), String> {
        let expected = reference();
        let raw: Vec<&str> = REFERENCE
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        let mut handles = Handles::default();
        for (index, (expected, name)) in expected.iter().enumerate() {
            let Some(line) = ours.get(index) else {
                return Err(format!(
                    "line {index}: expected `{expected}`, and the run has no more lines"
                ));
            };
            let found = text(line, anchors, &mut handles, expected)
                .map_err(|error| format!("line {index}: {error}"))?;
            if found != *expected {
                return Err(format!(
                    "line {index}: expected `{expected}`, found `{found}`"
                ));
            }
            if let Some(name) = name {
                let control = raw[index].starts_with(&format!("ipc {CONTROL_INTERFACE}::"));
                check_name(line, name, control)
                    .map_err(|error| format!("line {index}: {error}"))?;
            }
        }
        if ours.len() > expected.len() {
            return Err(format!(
                "line {}: the run goes on after the reference's last line",
                expected.len()
            ));
        }
        Ok(())
    }

    /// The size of the fixture's image, from its header.
    fn image_size() -> u64 {
        let word: [u8; 4] = HELLO[NRO_IMAGE_SIZE_OFFSET..NRO_IMAGE_SIZE_OFFSET + 4]
            .try_into()
            .expect("four bytes");
        u64::from(u32::from_le_bytes(word))
    }

    /// The text of `static const char <name>[] = "...";` in the fixture's source.
    fn source_string(name: &str) -> &'static str {
        let declaration = format!("static const char {name}[] = \"");
        let (_, rest) = HELLO_SOURCE
            .split_once(&declaration)
            .unwrap_or_else(|| panic!("hello.c declares {name}"));
        rest.split_once('"').expect("a closing quote").0
    }

    /// The comparison against made-up runs, on every host.
    #[cfg(test)]
    mod comparison {
        use starquake_kernel::memory::GUARD_PAGES;
        use starquake_kernel::result::KernelError;
        use starquake_memory::PAGE_SIZE;

        use super::*;

        /// Where the made-up runs put the module: starquake's address for it, and the fixture's
        /// size in memory (its image and a page of `.bss`).
        const BASE: u64 = 0x8000_0000;
        /// The two session handles of the made-up runs: not the reference's values, so that the
        /// pairing is what makes them match.
        const SM_HANDLE: u64 = 0x2_0011;
        const SET_HANDLE: u64 = 0x4_0012;

        fn anchors() -> Anchors {
            Anchors::new(BASE, image_size(), image_size() + PAGE_SIZE)
        }

        fn call(id: SvcId, registers: &[(usize, u64)], result: u32) -> Line {
            let mut args = [0; SVC_ARG_COUNT];
            for (register, value) in registers {
                args[*register] = *value;
            }
            Line::Svc {
                number: u32::from(id.number()),
                args,
                result,
            }
        }

        fn ipc(service: ServiceId, command: u32) -> Line {
            Line::Ipc { service, command }
        }

        fn log(name: &str) -> Line {
            let EventKind::GuestLog { len, hash } =
                EventKind::guest_log(source_string(name).as_bytes())
            else {
                unreachable!("a guest log event")
            };
            Line::Log { len, hash }
        }

        /// A run that does what the reference did, with every address where starquake puts it:
        /// the module at [`BASE`], the stack four pages into the stack region.
        fn matching() -> Vec<Line> {
            let anchors = anchors();
            let regions = anchors.regions;
            let relro = BASE + 0x4000;
            let process = 0xffff_8001;
            // libnx's two buffers on its stack, 0x38 bytes below the stack's top.
            let stack_top = regions.stack.start + GUARD_PAGES * PAGE_SIZE + 0x10_0000;
            let buffer = stack_top - 0x38;
            let closed = KernelError::SessionClosed.code().raw();
            let send = |handle, result| call(SvcId::SendSyncRequest, &[(0, handle)], result);
            let close = |handle| call(SvcId::CloseHandle, &[(0, handle)], 0);
            let say = |at: u64, len| call(SvcId::OutputDebugString, &[(0, BASE + at), (1, len)], 0);
            let mut lines = vec![
                call(
                    SvcId::SetMemoryPermission,
                    &[(0, relro), (1, 0x1000), (2, 1)],
                    0,
                ),
                call(
                    SvcId::SetMemoryAttribute,
                    &[(0, relro), (1, 0x1000), (2, 0x10), (3, 0x10)],
                    0,
                ),
            ];
            for info_type in [2, 3, 0x1c, 4, 5, 0xc, 0xd, 0xe, 0xf, 6, 7] {
                lines.push(call(SvcId::GetInfo, &[(1, info_type), (2, process)], 0));
            }
            lines.extend([
                call(SvcId::SetHeapSize, &[(1, 0xcd00_0000)], 0),
                call(SvcId::QueryMemory, &[(0, buffer), (2, buffer)], 0),
                call(
                    SvcId::QueryMemory,
                    &[(0, buffer), (2, BASE + image_size())],
                    0,
                ),
                call(SvcId::ConnectToNamedPort, &[(1, BASE + 0x3148)], 0),
                ipc(SM, 3),
                send(SM_HANDLE, 0),
                ipc(SM, 0),
                send(SM_HANDLE, 0),
                log("LINE_ONE"),
                say(0x30a0, 0x19),
                log("LINE_TWO"),
                say(0x30c0, 0x19),
                ipc(SM, 1),
                send(SM_HANDLE, 0),
                ipc(SET_SYS, 3),
                send(SET_HANDLE, 0),
                ipc(SET_SYS, 3),
                send(SET_HANDLE, 0),
                send(SET_HANDLE, closed),
                close(SET_HANDLE),
                log("FIRMWARE_OK"),
                say(0x30e0, 0x2c),
                send(SM_HANDLE, closed),
                close(SM_HANDLE),
                call(SvcId::ExitProcess, &[], 0),
            ]);
            lines
        }

        fn failure(lines: &[Line]) -> String {
            compare(lines, &anchors()).unwrap_err()
        }

        #[test]
        fn a_run_that_does_what_the_reference_did_passes() {
            assert_eq!(compare(&matching(), &anchors()), Ok(()));
            // The reference's 38 lines: 17 calls up to the port; five requests, each with its
            // `SendSyncRequest`; two close requests and their `CloseHandle`s; three log lines,
            // each with its `OutputDebugString`; and the exit.
            assert_eq!(matching().len(), reference().len());
            assert_eq!(reference().len(), 17 + 5 * 2 + 2 * 2 + 3 * 2 + 1);
        }

        #[test]
        fn the_reference_is_read_as_the_module_docs_say() {
            let lines = reference();
            // A control request goes to the service of the next request after it.
            assert_eq!(
                lines[17],
                (
                    "ipc sm:::3".to_owned(),
                    Some("QueryPointerBufferSize".to_owned())
                )
            );
            assert_eq!(
                lines[27],
                (
                    "ipc set:sys::3".to_owned(),
                    Some("QueryPointerBufferSize".to_owned())
                )
            );
            assert_eq!(
                lines[29],
                (
                    "ipc set:sys::3".to_owned(),
                    Some("GetFirmwareVersion".to_owned())
                )
            );
            // Outputs are dropped; a call with none keeps its line.
            assert_eq!(
                lines[16].0,
                "svc 0x1f ConnectToNamedPort(name=main+0x3148) -> 0x0"
            );
            assert_eq!(lines[37].0, "svc 0x07 ExitProcess()");
            assert_eq!(lines[21].0, "log starquake hello: line one");
        }

        #[test]
        fn where_the_module_and_the_regions_are_does_not_matter() {
            // The same run with the module where the reference emulator put it in one of its
            // runs: every address moves with its anchor.
            let moved = Anchors::new(0x80c0_1000, image_size(), image_size() + PAGE_SIZE);
            let here = anchors();
            let shift = |address: u64| {
                if (here.main.start..=here.main.end).contains(&address) {
                    address - here.main.start + moved.main.start
                } else if here.regions.stack.holds(address, 1) {
                    address - here.regions.stack.start + moved.regions.stack.start
                } else {
                    address
                }
            };
            let lines: Vec<Line> = matching()
                .into_iter()
                .map(|line| match line {
                    // Registers 0 to 2 hold every address these calls take; sizes, ids and
                    // handles are too small to be taken for one.
                    Line::Svc {
                        number,
                        mut args,
                        result,
                    } => {
                        for register in 0..3 {
                            args[register] = shift(args[register]);
                        }
                        Line::Svc {
                            number,
                            args,
                            result,
                        }
                    }
                    other => other,
                })
                .collect();
            assert_ne!(lines, matching());
            assert_eq!(compare(&lines, &moved), Ok(()));
            // Against the wrong anchors the same run fails at its first address.
            assert!(compare(&lines, &here).is_err());
        }

        #[test]
        fn a_wrong_argument_or_result_fails_at_that_line() {
            // A different page of the module.
            let mut lines = matching();
            let Line::Svc { args, .. } = &mut lines[1] else {
                unreachable!()
            };
            args[0] += 0x1000;
            assert_eq!(
                failure(&lines),
                "line 1: expected `svc 0x03 SetMemoryAttribute(address=main+0x4000, size=0x1000, \
                 mask=0x10, attr=0x10) -> 0x0`, found `svc 0x03 \
                 SetMemoryAttribute(address=main+0x5000, size=0x1000, mask=0x10, attr=0x10) -> 0x0`"
            );
            // A close request that succeeded, where the reference's session was closed.
            let mut lines = matching();
            let Line::Svc { result, .. } = &mut lines[31] else {
                unreachable!()
            };
            *result = 0;
            assert_eq!(
                failure(&lines),
                "line 31: expected `svc 0x21 SendSyncRequest(session_handle=0x183fd) -> 0xf601`, \
                 found `svc 0x21 SendSyncRequest(session_handle=0x183fd) -> 0x0`"
            );
            // The port's name at another address.
            let mut lines = matching();
            let Line::Svc { args, .. } = &mut lines[16] else {
                unreachable!()
            };
            args[1] += 8;
            assert!(failure(&lines).starts_with(
                "line 16: expected `svc 0x1f ConnectToNamedPort(name=main+0x3148) -> 0x0`, found \
                 `svc 0x1f ConnectToNamedPort(name=main+0x3150)"
            ));
        }

        #[test]
        fn a_request_to_another_service_or_command_fails_at_that_line() {
            let mut lines = matching();
            lines[29] = ipc(SM, 3);
            assert_eq!(
                failure(&lines),
                "line 29: expected `ipc set:sys::3`, found `ipc sm:::3`"
            );
            let mut lines = matching();
            lines[19] = ipc(SM, 4);
            assert_eq!(
                failure(&lines),
                "line 19: expected `ipc sm:::0`, found `ipc sm:::4`"
            );
            let mut lines = matching();
            lines[19] = ipc(ServiceId(0x99), 0);
            assert_eq!(
                failure(&lines),
                "line 19: service ServiceId(153) has no name"
            );
        }

        #[test]
        fn a_command_name_the_tables_do_not_give_fails() {
            // A control request with the id of `Initialize`: the reference calls command 0
            // `Initialize`, and no control command has that name.
            assert_eq!(
                check_name(&ipc(SM, 0), "Initialize", true),
                Err("the reference calls command 0 Initialize, starquake \
                     Some(\"ConvertCurrentObjectToDomain\")"
                    .to_owned())
            );
            assert_eq!(check_name(&ipc(SM, 0), "Initialize", false), Ok(()));
            assert_eq!(
                check_name(&ipc(SET_SYS, 3), "GetFirmwareVersion", false),
                Ok(())
            );
            assert!(check_name(&ipc(SET_SYS, 4), "GetFirmwareVersion", false).is_err());
            assert!(check_name(&ipc(ServiceId(0x99), 0), "Initialize", false).is_err());
            assert_eq!(camel("get_service"), "GetService");
        }

        #[test]
        fn a_log_line_that_differs_fails_at_that_line() {
            let mut lines = matching();
            lines[21] = log("FIRMWARE_FAILED");
            let found = failure(&lines);
            assert!(
                found.starts_with(
                    "line 21: expected `log starquake hello: line one`, found `log len=48 hash="
                ),
                "{found}"
            );
        }

        #[test]
        fn a_handle_must_stand_for_one_reference_handle_throughout() {
            // The set:sys request sent on the sm: session.
            let mut lines = matching();
            lines[30] = call(SvcId::SendSyncRequest, &[(0, SM_HANDLE)], 0);
            assert_eq!(
                failure(&lines),
                "line 30: expected `svc 0x21 SendSyncRequest(session_handle=0x183fd) -> 0x0`, \
                 found `svc 0x21 SendSyncRequest(session_handle=0x103fe) -> 0x0`"
            );
            // A second handle where the reference used the first one again.
            let mut lines = matching();
            lines[20] = call(SvcId::SendSyncRequest, &[(0, 0x6_0013)], 0);
            assert!(failure(&lines).starts_with("line 20: expected"));
        }

        #[test]
        fn a_missing_line_a_short_run_and_a_long_one_fail() {
            let mut lines = matching();
            lines.remove(17);
            assert!(failure(&lines).starts_with("line 17: expected `ipc sm:::3`"));
            let lines = &matching()[..37];
            assert_eq!(
                failure(lines),
                "line 37: expected `svc 0x07 ExitProcess()`, and the run has no more lines"
            );
            let mut lines = matching();
            lines.push(ipc(SM, 0));
            assert_eq!(
                failure(&lines),
                "line 38: the run goes on after the reference's last line"
            );
        }

        #[test]
        fn a_call_or_an_address_the_comparison_does_not_know_is_refused() {
            let mut lines = matching();
            lines[0] = call(SvcId::MapMemory, &[], 0);
            assert_eq!(failure(&lines), "line 0: no address rule for MapMemory");
            let mut lines = matching();
            lines[0] = Line::Svc {
                number: 0x7777,
                args: [0; SVC_ARG_COUNT],
                result: 0,
            };
            assert_eq!(failure(&lines), "line 0: svc 0x7777 is not a listed call");
            let mut lines = matching();
            let Line::Svc { args, .. } = &mut lines[0] else {
                unreachable!()
            };
            args[0] = 0x1000;
            assert_eq!(
                failure(&lines),
                "line 0: address 0x1000 lies in no known region"
            );
        }

        #[test]
        fn the_fixtures_strings_are_read_from_its_source() {
            assert_eq!(source_string("LINE_ONE"), "starquake hello: line one");
            assert_eq!(
                source_string("FIRMWARE_OK"),
                "starquake hello: set:sys firmware version ok"
            );
        }
    }

    /// The fixture on a real vCPU.
    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    mod on_hardware {
        use starquake_core::{ExitReason, RunReport, exit_code};
        use starquake_memory::PAGE_SIZE;
        use starquake_trace::Dump;

        use super::*;
        use crate::common::{GENEROUS, run};

        /// The lines of a run, in order, from its event dump.
        fn lines(report: &RunReport) -> Vec<Line> {
            Dump::from_bytes(&report.events)
                .expect("a readable dump")
                .merged()
                .iter()
                .filter_map(|event| Line::of(&event.kind))
                .collect()
        }

        /// Where the run put the module, from its memory map.
        fn anchors(report: &RunReport) -> Anchors {
            let module: Vec<_> = report
                .memory_map
                .iter()
                .filter(|mapping| mapping.name.starts_with("main."))
                .collect();
            let base = module.iter().map(|mapping| mapping.start).min().unwrap();
            let end = module.iter().map(|mapping| mapping.end).max().unwrap();
            Anchors::new(base, image_size(), end.next_multiple_of(PAGE_SIZE) - base)
        }

        #[test]
        fn the_fixture_matches_the_reference_line_for_line_through_exit_process() {
            let Some(report) = run(HELLO, GENEROUS) else {
                return;
            };
            assert_eq!(
                compare(&lines(&report), &anchors(&report)),
                Ok(()),
                "{}",
                report.trace
            );
            // The run ends where the program does, with the program's exit code.
            assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
            assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
            // The three lines are the fixture's own strings.
            let logged: Vec<_> = report
                .guest_log
                .iter()
                .map(|line| String::from_utf8_lossy(line).into_owned())
                .collect();
            assert_eq!(
                logged,
                [
                    source_string("LINE_ONE"),
                    source_string("LINE_TWO"),
                    source_string("FIRMWARE_OK")
                ]
            );
            // Every call it makes is implemented.
            assert_eq!(report.stubbed, Default::default());
            let made: Vec<_> = report
                .system_calls
                .iter()
                .map(|(name, count)| (name.as_str(), *count))
                .collect();
            assert_eq!(
                made,
                [
                    ("CloseHandle", 2),
                    ("ConnectToNamedPort", 1),
                    ("ExitProcess", 1),
                    ("GetInfo", 11),
                    ("OutputDebugString", 3),
                    ("QueryMemory", 2),
                    ("SendSyncRequest", 7),
                    ("SetHeapSize", 1),
                    ("SetMemoryAttribute", 1),
                    ("SetMemoryPermission", 1),
                ]
            );
            // The fixture has no counter reads to patch, so nothing but the kernel's mappings is
            // in its address space.
            assert_eq!(report.patched_counter_reads, Vec::<u64>::new());
            insta::assert_snapshot!("hello_fixture_trace", report.trace);
        }
    }
}
