//! The homebrew fixtures (real libnx programs under `tests/homebrew/`) against the reference
//! emulator: a fixture's run in starquake is compared, line for line, with the normalised trace
//! of the same file under Eden-NX2 (`reference/startup.trace` in the fixture's folder; the
//! README beside the hello fixture's has the provenance and the format).
//!
//! - **`hello`** (`tests/homebrew/hello/hello.nro`) runs from its first system call to
//!   `ExitProcess`, and must match the reference all the way.
//! - **`startup`** (`tests/homebrew/startup/startup.nro`) keeps libnx's own start-up, which
//!   needs services starquake does not have yet. Its run is compared as far as it matches, and
//!   the test fails if that is fewer lines than [`STARTUP_MATCHED`] records: a ratchet that
//!   later work raises. The services and calls it is missing are listed in
//!   `tests/homebrew/startup/reference/README.md`, a list this file derives from the trace and
//!   checks (`the_startup_gap_list_in_its_readme_is_the_one_the_trace_gives`).
//!
//! Each line of the reference is one of three kinds, and matches when:
//!
//! - **`svc`**: its number, its decoded arguments and its result are the reference's.
//!   Addresses are compared as `anchor+offset` (the module, or a region of the address space),
//!   so the two emulators may place the module and the regions differently but must agree on
//!   everything inside them. Handles are compared as a mapping: the first time a reference
//!   handle value appears, the run's handle in the same place is paired with it, and from then
//!   on each must appear only with the other (pseudo-handles and zero are compared as values).
//!   A run's handle that is not paired is written `unpaired <value>`, so it cannot read as a
//!   reference handle whose value it happens to have.
//!   Placements (`placement1+0x1000`: a range libnx put at a random page of the ASLR region,
//!   `tools/oracle/normalise.py` says how) are paired the same way: the first time the
//!   reference names one, the run's address in the same place, less the same offset, is the
//!   placement's anchor (the guard page below its range, which libnx probes) and is paired
//!   with it. libnx's `_memregionFindRandom` (`libnx/nx/source/kernel/virtmem.c`) draws the
//!   range `[map, map + size)` inside the ASLR region, refuses it if it overlaps the alias or
//!   the heap region (`_memregionOverlaps`), and refuses it if anything is mapped in it or in a
//!   guard page on either side (`_memregionIsMapped`, guard 0x1000 from `shmemMap` in
//!   `libnx/nx/source/kernel/shmem.c`): the guard pages themselves may lie in the alias or the
//!   heap region, where nothing is mapped there. The kernel lets a shared memory lie anywhere
//!   else in the ASLR region (`eden/src/core/hle/kernel/k_page_table_base.cpp`,
//!   `KPageTableBase::CanContain`: `is_in_region && !is_in_heap && !is_in_alias` for
//!   `MemoryState::Shared`, all three on the mapped range), the stack region included. So a
//!   placement pairs when:
//!   - first named, before its size is known: its anchor lies in the ASLR region and outside
//!     the module, and the page above the anchor (the range's first) outside the alias and the
//!     heap region;
//!   - on the line that maps it (`MapSharedMemory`), with its size: the range lies outside the
//!     alias and the heap region, and the range with its two guard pages lies in the ASLR
//!     region and overlaps neither the module nor the range of another placement the run has
//!     mapped.
//!
//!   These are libnx's conditions, with what is mapped narrowed to what the comparison knows
//!   (the module, through `__argdata__`, and the placements). A guard page above the ASLR
//!   region is refused as libnx refuses it: for a 39-bit address space the ASLR region ends
//!   where the address space does (`eden/src/core/hle/kernel/k_address_space_info.cpp`,
//!   `Map39Bit`; pinned by `the_aslr_region_ends_where_the_address_space_does`), and
//!   `KPageTableBase::QueryInfo` reports an address outside it as `Inaccessible`, which
//!   `_memregionIsMapped` counts as mapped. Two divergences are the comparison's own, both
//!   stricter than libnx, both shared with `normalise.py`, and neither reached by the
//!   fixtures' traces:
//!   - A guard page below the ASLR region is refused, so a range that starts exactly at the
//!     region's start does not pair (pinned in
//!     `a_placement_is_not_paired_where_libnx_does_not_place_one`). The address space starts
//!     below the region (at 0, `KPageTableBase::InitializeForProcess`), so libnx only needs
//!     that page unmapped; that it is unmapped there in the reference is a guess the
//!     comparison does not rely on. It costs one page of the region per placement.
//!   - A placement's range stays taken after it is unmapped, so a later placement over it does
//!     not pair (pinned by `a_placement_over_an_unmapped_one_does_not_pair`); the traces map
//!     every placement before they unmap one.
//!
//!   `ExitProcess` has no result in the reference, which never saw it return;
//!   starquake ends the run there (`docs/reference-emulator.md`). Outputs other than the result
//!   are not in starquake's event stream; what `GetInfo` answers is checked against the same
//!   reference in `starquake-kernel` (`svc/info/tests.rs`, `memory/layout.rs`).
//! - **`ipc`**: the service, by the name starquake registered its number under
//!   (`starquake_core::hle::service_name`), and the command id are the reference's, and the
//!   reference's name for the command is the one starquake's command table lists, written in
//!   the reference's case. A reference control request (`IpcController::<id>`) is a request
//!   with that id on the session of the service the reference's next non-control request goes
//!   to: starquake's IPC event does not say whether a request was a control request
//!   (`starquake_services::framework::trace`), so this is the most the comparison can check.
//! - **`log`**: the run logged a line of the same length and hash (an event keeps no text).
//!   The text itself is checked against the fixture's source by the run's guest log.
//!
//! The comparison runs on every host, against made-up runs that must fail it: a wrong
//! argument, result, service or command, a log line that differs, a handle used for two
//! sessions, a placement paired twice, a missing line, a run that stops early. The runs on a
//! real vCPU need a Mac with Apple Silicon and skip themselves only when `CI` is set.

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod common;

/// Everything is in a `cfg(test)` module so that the helpers may unwrap like the tests do.
#[cfg(test)]
mod trace {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;

    use starquake_core::hle::{FSP_SRV, SET_SYS, SM, service_name};
    use starquake_kernel::handle::Handle;
    use starquake_kernel::memory::MemoryArrangement;
    use starquake_kernel::memory::layout::{LayoutRequest, Region, RegionOffsets, Regions};
    use starquake_kernel::svc::{Kind, Status, SvcId};
    use starquake_loader::nro::Nro;
    use starquake_memory::PAGE_SIZE;
    use starquake_services::framework::control::{CONTROL_COMMANDS, CONTROL_INTERFACE};
    use starquake_services::framework::{Protocol, Session};
    use starquake_services::fsp::FspSrv;
    use starquake_services::set::SetSys;
    use starquake_services::sm::Sm;
    use starquake_trace::{EventKind, SVC_ARG_COUNT, ServiceId};

    /// A homebrew fixture and the reference emulator's run of it, normalised.
    #[derive(Debug, Clone, Copy)]
    struct Fixture {
        /// The NRO that is run.
        nro: &'static [u8],
        /// Its `reference/startup.trace`.
        reference: &'static str,
    }

    /// The milestone 1 fixture: start-up opens only `sm:`.
    const HELLO: Fixture = Fixture {
        nro: include_bytes!("../../../tests/homebrew/hello/hello.nro"),
        reference: include_str!("../../../tests/homebrew/hello/reference/startup.trace"),
    };
    /// The hello fixture's source.
    const HELLO_SOURCE: &str = include_str!("../../../tests/homebrew/hello/hello.c");

    /// The fixture built with libnx's own start-up.
    const STARTUP: Fixture = Fixture {
        nro: include_bytes!("../../../tests/homebrew/startup/startup.nro"),
        reference: include_str!("../../../tests/homebrew/startup/reference/startup.trace"),
    };
    /// The README beside the startup fixture's trace, which holds its gap list.
    const STARTUP_README: &str =
        include_str!("../../../tests/homebrew/startup/reference/README.md");

    /// How many lines of the startup fixture's reference a run in starquake matches, at least:
    /// the ratchet (see the module docs). Raise it when a change makes the run match more; the
    /// on-hardware test prints the new count when it does.
    const STARTUP_MATCHED: usize = 31;

    /// The arguments that are addresses, for each call the fixtures make: the same list as
    /// `ADDRESSES` in `tools/oracle/normalise.py`. A call that is not listed is refused, so no
    /// address is compared as a number by oversight.
    const ADDRESS_ARGUMENTS: &[(SvcId, &[&str])] = &[
        (SvcId::SetHeapSize, &[]),
        (SvcId::SetMemoryPermission, &["address"]),
        (SvcId::SetMemoryAttribute, &["address"]),
        (SvcId::QueryMemory, &["out_memory_info", "address"]),
        (SvcId::MapSharedMemory, &["address"]),
        (SvcId::UnmapSharedMemory, &["address"]),
        (SvcId::CreateTransferMemory, &["address"]),
        (SvcId::WaitSynchronization, &["handles"]),
        (SvcId::GetInfo, &[]),
        (SvcId::ConnectToNamedPort, &["name"]),
        (SvcId::SendSyncRequest, &[]),
        (SvcId::CloseHandle, &[]),
        (SvcId::OutputDebugString, &["debug_str"]),
        (SvcId::ExitProcess, &[]),
    ];

    /// The calls that map a placement, with the arguments holding its range's start and size:
    /// the same list as `PLACEMENTS` in `tools/oracle/normalise.py`.
    const PLACEMENTS: &[(SvcId, &str, &str)] = &[(SvcId::MapSharedMemory, "address", "size")];
    /// The guard libnx keeps free on each side of a placement: `virtmemFindAslr(s->size,
    /// 0x1000)` in `shmemMap` (`libnx/nx/source/kernel/shmem.c`), `PLACEMENT_GUARD` in
    /// `normalise.py`. A placement's anchor is its range's start less this.
    const PLACEMENT_GUARD: u64 = 0x1000;

    /// The name `normalise.py` gives the `n`th placement is this followed by `n`.
    const PLACEMENT: &str = "placement";
    /// The anchor `normalise.py` writes for the ASLR region.
    const ASLR_REGION: &str = "aslr_region";

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
        /// The module's image and `.bss`: its first byte to the first byte after them, inclusive
        /// (libnx asks about that address, `__argdata__`).
        main: Region,
        regions: Regions,
    }

    impl Anchors {
        /// The anchors of a run whose module is `module_size` bytes at `base`, of which the
        /// first `extent` are its image and `.bss` ([`module_extent`]), on a 4 GiB console: the
        /// regions are where the kernel places them around that code.
        fn new(base: u64, extent: u64, module_size: u64) -> Self {
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
                    end: base + extent,
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
                (ASLR_REGION, self.regions.code),
            ];
            regions
                .into_iter()
                .find(|(_, region)| region.holds(address, 1))
                .map(|(name, region)| format!("{name}+{:#x}", address - region.start))
                .ok_or_else(|| format!("address {address:#x} lies in no known region"))
        }

        /// Whether `address` is in the module, its last address included, as
        /// [`Anchors::symbol`] counts it.
        fn in_module(&self, address: u64) -> bool {
            (self.main.start..=self.main.end).contains(&address)
        }

        /// Whether a placement may be anchored at `anchor`, before its size is known: the
        /// anchor in the ASLR region and outside the module, the page above it (its range's
        /// first) outside the alias and the heap region (see the module docs).
        fn anchorable(&self, anchor: u64) -> bool {
            let Some(start) = anchor.checked_add(PLACEMENT_GUARD) else {
                return false;
            };
            self.regions.code.holds(anchor, 1)
                && !self.in_module(anchor)
                && !self.regions.alias.holds(start, 1)
                && !self.regions.heap.holds(start, 1)
        }

        /// Whether a placement may map `range`, with `others` the ranges the run's other
        /// placements map: the range outside the alias and the heap region, and the range with
        /// its guard pages in the ASLR region, outside the module and outside `others` (see the
        /// module docs).
        fn mappable(&self, range: Region, mut others: impl Iterator<Item = Region>) -> bool {
            let (Some(start), Some(end)) = (
                range.start.checked_sub(PLACEMENT_GUARD),
                range.end.checked_add(PLACEMENT_GUARD),
            ) else {
                return false;
            };
            let guarded = Region { start, end };
            range.start < range.end
                && !overlaps(range, self.regions.alias)
                && !overlaps(range, self.regions.heap)
                && self
                    .regions
                    .code
                    .holds(guarded.start, guarded.end - guarded.start)
                && !(guarded.start <= self.main.end && self.main.start < guarded.end)
                && others.all(|other| !overlaps(guarded, other))
        }
    }

    /// Whether two ranges share an address: libnx's `_memregionOverlaps`
    /// (`libnx/nx/source/kernel/virtmem.c`), `start < r->end && r->start < end`.
    fn overlaps(a: Region, b: Region) -> bool {
        a.start < b.end && b.start < a.end
    }

    /// A one-to-one pairing of the reference's names for things (handle values, placements)
    /// with the run's values (see the module docs).
    #[derive(Debug, Default)]
    struct Pairing {
        ours: BTreeMap<u64, String>,
        theirs: BTreeMap<String, u64>,
    }

    impl Pairing {
        /// The reference name the run's `ours` is paired with, if it is.
        fn get(&self, ours: u64) -> Option<&str> {
            self.ours.get(&ours).map(String::as_str)
        }

        /// The run's value the reference name `theirs` is paired with, if it is.
        fn ours_for(&self, theirs: &str) -> Option<u64> {
            self.theirs.get(theirs).copied()
        }

        /// Pairs `ours` with `theirs` if neither is paired yet; whether it did.
        fn pair(&mut self, ours: u64, theirs: &str) -> bool {
            if self.ours.contains_key(&ours) || self.theirs.contains_key(theirs) {
                return false;
            }
            self.ours.insert(ours, theirs.to_owned());
            self.theirs.insert(theirs.to_owned(), ours);
            true
        }
    }

    /// What a comparison has paired so far.
    #[derive(Debug, Default)]
    struct Pairings {
        /// The reference's handle values with the run's handles.
        handles: Pairing,
        /// The reference's placements with the run's anchors for them.
        placements: Pairing,
        /// The range each placement maps in the run, by the reference's name for it, from the
        /// line that maps it.
        mapped: BTreeMap<String, Region>,
    }

    impl Pairings {
        /// The run's handle `ours`, written as the reference handle it stands for. `expected` is
        /// the value the reference has in its place, which pairs the two the first time either
        /// is seen. A handle that cannot be paired is written `unpaired <value>`, which no
        /// reference line has.
        fn handle(&mut self, ours: u64, expected: Option<&str>) -> String {
            if ours == 0 || Handle::from_register(ours).is_pseudo() {
                return format!("{ours:#x}");
            }
            if let Some(paired) = self.handles.get(ours) {
                return paired.to_owned();
            }
            match expected {
                Some(expected) if self.handles.pair(ours, expected) => expected.to_owned(),
                // Nothing to pair with, or the reference handle stands for another of ours.
                _ => format!("unpaired {ours:#x}"),
            }
        }

        /// The run's address `ours`, written as the reference would: in a placement it is
        /// paired with (or is paired with now, if `expected` names a placement and its anchor
        /// lies where one may), or else in the module or a region. `size` is set when `ours`
        /// starts a range that a placement maps ([`PLACEMENTS`]): the placement is then
        /// written only if that range lies where libnx may map it (see the module docs).
        fn address(
            &mut self,
            ours: u64,
            expected: Option<&str>,
            size: Option<u64>,
            anchors: &Anchors,
        ) -> Result<String, String> {
            let symbol = anchors.symbol(ours)?;
            let Some((name, offset)) = expected.and_then(placement) else {
                return Ok(symbol);
            };
            // The anchor the run's placement has if this address is where the reference's is.
            let Some(base) = ours.checked_sub(offset) else {
                return Ok(symbol);
            };
            let anchor = match self.placements.ours_for(name) {
                Some(paired) => paired,
                None if anchors.anchorable(base) && self.placements.pair(base, name) => base,
                None => return Ok(symbol),
            };
            let Some(offset) = ours.checked_sub(anchor) else {
                return Ok(symbol);
            };
            if let Some(size) = size {
                let Some(end) = ours.checked_add(size) else {
                    return Ok(symbol);
                };
                let range = Region { start: ours, end };
                let others = self
                    .mapped
                    .iter()
                    .filter(|(other, _)| other.as_str() != name)
                    .map(|(_, other)| *other);
                if !anchors.mappable(range, others) {
                    return Ok(symbol);
                }
                self.mapped.insert(name.to_owned(), range);
            }
            Ok(format!("{name}+{offset:#x}"))
        }
    }

    /// A reference address that is in a placement, as (placement, offset):
    /// `placement2+0x1000` is `("placement2", 0x1000)`.
    fn placement(text: &str) -> Option<(&str, u64)> {
        let (name, offset) = text.split_once('+')?;
        let index = name.strip_prefix(PLACEMENT)?;
        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let offset = u64::from_str_radix(offset.strip_prefix("0x")?, 16).ok()?;
        Some((name, offset))
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
        pairings: &mut Pairings,
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
        // The value of the argument called `name`.
        let value_of = |name: &str| {
            convention
                .inputs
                .iter()
                .find(|input| input.name == name)
                .map(|input| args[usize::from(input.reg)])
        };
        let mut arguments = Vec::new();
        for input in convention.inputs {
            let value = args[usize::from(input.reg)];
            let text = if addresses.contains(&input.name) {
                // The size of the range a placement maps from here, if this call maps one.
                let size = PLACEMENTS
                    .iter()
                    .find(|(listed, start, _)| *listed == id && *start == input.name)
                    .and_then(|(_, _, size)| value_of(size));
                pairings.address(value, argument(expected, input.name), size, anchors)?
            } else if input.kind == Kind::Handle {
                pairings.handle(value, argument(expected, input.name))
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

    /// The services whose command tables starquake has, in the order of their numbers.
    const SERVICES: [ServiceId; 3] = [SM, SET_SYS, FSP_SRV];

    /// The interface whose command table names a service's commands.
    fn interface(service: ServiceId) -> Option<Box<dyn Session>> {
        match service {
            SM => Some(Box::new(Sm::default())),
            SET_SYS => Some(Box::new(SetSys)),
            FSP_SRV => Some(Box::new(FspSrv)),
            _ => None,
        }
    }

    /// The number of the service starquake registered under `name`, if it has it.
    fn service_id(name: &str) -> Option<ServiceId> {
        SERVICES
            .into_iter()
            .find(|service| service_name(*service) == Some(name))
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

    /// The reference's lines, without its comments.
    fn raw_lines(fixture: &Fixture) -> Vec<&'static str> {
        fixture
            .reference
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect()
    }

    /// A reference request line's parts, without its `ipc `: `sm:::1 (GetService)` is
    /// `("sm:", "1", "GetService")`. The error says the line is not written that way.
    fn request_parts(request: &str) -> Result<(&str, &str, &str), String> {
        let parts = request.split_once(" (").and_then(|(target, name)| {
            let name = name.strip_suffix(')')?;
            let (service, command) = target.rsplit_once("::")?;
            Some((service, command, name))
        });
        parts.ok_or_else(|| {
            format!("the request `{request}` is not written `<service>::<command> (<name>)`")
        })
    }

    /// The reference's lines, without its comments and without each call's outputs, with every
    /// control request (`IpcController::<id>`) written as a request to the service the next
    /// non-control request goes to, and its command's name kept apart:
    /// `(line, Some(name))` for a request. A control request with no other request after it is
    /// an error naming its line (counted from 0, without the comments).
    fn reference(fixture: &Fixture) -> Result<Vec<(String, Option<String>)>, String> {
        let lines = raw_lines(fixture);
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
                let (service, command, name) =
                    request_parts(request).map_err(|error| format!("line {index}: {error}"))?;
                let service = if service == CONTROL_INTERFACE {
                    // A request after it that cannot be read is skipped here; it is an error
                    // when this loop reaches its own line.
                    lines[index + 1..]
                        .iter()
                        .filter_map(|line| line.strip_prefix("ipc "))
                        .filter_map(|next| request_parts(next).ok())
                        .map(|(next, _, _)| next)
                        .find(|next| *next != CONTROL_INTERFACE)
                        .ok_or_else(|| {
                            format!(
                                "line {index}: the control request `{line}` has no request \
                                 after it"
                            )
                        })?
                } else {
                    service
                };
                out.push((format!("ipc {service}::{command}"), Some(name.to_owned())));
            } else {
                out.push(((*line).to_owned(), None));
            }
        }
        Ok(out)
    }

    /// One line of the run, written as the reference writes it.
    fn text(
        line: &Line,
        anchors: &Anchors,
        pairings: &mut Pairings,
        expected: &str,
    ) -> Result<String, String> {
        match *line {
            Line::Svc {
                number,
                args,
                result,
            } => {
                let call = call_text(number, &args, anchors, pairings, expected)?;
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
                // The reference's text, if the reference has a log line here and it is this
                // line's: an event keeps only the length and the hash.
                if let Some(text) = expected.strip_prefix("log ")
                    && EventKind::guest_log(text.as_bytes()) == (EventKind::GuestLog { len, hash })
                {
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

    /// How far a run matches a reference.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Progress {
        /// How many of the reference's lines the run matches, from the first.
        matched: usize,
        /// The first difference, naming its line (counted from 0, without the comments), or
        /// `None` if the run matches the whole reference and stops where it does.
        difference: Option<String>,
    }

    /// Compares a run's lines with a fixture's reference, in order, until the first that
    /// differs.
    fn progress(fixture: &Fixture, ours: &[Line], anchors: &Anchors) -> Progress {
        let stop = |matched: usize, difference: String| Progress {
            matched,
            difference: Some(difference),
        };
        // A reference that cannot be read matches nothing.
        let expected = match reference(fixture) {
            Ok(expected) => expected,
            Err(error) => return stop(0, error),
        };
        let raw = raw_lines(fixture);
        let mut pairings = Pairings::default();
        for (index, (expected, name)) in expected.iter().enumerate() {
            let Some(line) = ours.get(index) else {
                return stop(
                    index,
                    format!("line {index}: expected `{expected}`, and the run has no more lines"),
                );
            };
            let found = match text(line, anchors, &mut pairings, expected) {
                Ok(found) => found,
                Err(error) => return stop(index, format!("line {index}: {error}")),
            };
            if found != *expected {
                return stop(
                    index,
                    format!("line {index}: expected `{expected}`, found `{found}`"),
                );
            }
            if let Some(name) = name {
                let control = raw[index].starts_with(&format!("ipc {CONTROL_INTERFACE}::"));
                if let Err(error) = check_name(line, name, control) {
                    return stop(index, format!("line {index}: {error}"));
                }
            }
        }
        if ours.len() > expected.len() {
            return stop(
                expected.len(),
                format!(
                    "line {}: the run goes on after the reference's last line",
                    expected.len()
                ),
            );
        }
        Progress {
            matched: expected.len(),
            difference: None,
        }
    }

    /// Compares a run's lines with a fixture's reference, in order, through its last line. The
    /// error names the first line that differs.
    fn compare(fixture: &Fixture, ours: &[Line], anchors: &Anchors) -> Result<(), String> {
        match progress(fixture, ours, anchors).difference {
            Some(difference) => Err(difference),
            None => Ok(()),
        }
    }

    /// How far a fixture's module reaches past its base, to libnx's `__argdata__`, the address
    /// start-up asks `QueryMemory` about (`libnx/nx/source/runtime/argv.c`), which libnx's linker
    /// script puts at the first page boundary after `.bss` (`libnx/nx/switch.ld`): the image and
    /// `.bss` sizes of its NRO header, each rounded up to a page, the same rule as
    /// `tools/oracle/normalise.py`. For both fixtures this ends exactly at `__argdata__`
    /// (checked against each ELF's symbol table); for other NROs that is a guess. It is not the
    /// extent Eden maps: Eden's `LoadNroImpl` takes `.bss` from the `MOD0` header every libnx
    /// NRO has (`eden/src/core/loader/nro.cpp`), which maps one page of `.bss` more for both
    /// fixtures (`normalise.py`'s module docs give the sizes).
    fn module_extent(fixture: &Fixture) -> u64 {
        let nro = Nro::parse(fixture.nro).expect("the fixture parses");
        let page_up = |size: u32| u64::from(size).next_multiple_of(PAGE_SIZE);
        page_up(nro.size) + page_up(nro.bss_size)
    }

    /// The text of `static const char <name>[] = "...";` in the hello fixture's source.
    fn source_string(name: &str) -> &'static str {
        let declaration = format!("static const char {name}[] = \"");
        let (_, rest) = HELLO_SOURCE
            .split_once(&declaration)
            .expect("hello.c declares the string");
        rest.split_once('"').expect("a closing quote").0
    }

    // ----- The gap list ---------------------------------------------------------------------

    /// One system call or service command a reference shows that starquake does not implement.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Gap {
        /// The line of the trace file (counted from 1, its comment lines included) where it is
        /// first used.
        line: usize,
        /// The call (`svc 0x18 WaitSynchronization`) or the request
        /// (`ipc apm::0 (OpenSession)`).
        what: String,
        /// What the reference returned there: the call's result (`none` for a call without
        /// one), or, for a request, the result of the `SendSyncRequest` that carried it.
        result: String,
    }

    /// Whether starquake implements a system call: its status is
    /// [`Status::Implemented`].
    fn implements_call(number: u16) -> bool {
        SvcId::from_number(number).is_some_and(|id| id.status() == Status::Implemented)
    }

    /// Whether starquake implements a request: a control command the framework's table names
    /// the same, or a command of a service starquake has, whose table names it the same.
    fn implements_request(service: &str, command: u32, name: &str) -> bool {
        if service == CONTROL_INTERFACE {
            // Any service will do: the control table is the framework's.
            return check_name(
                &Line::Ipc {
                    service: SM,
                    command,
                },
                name,
                true,
            )
            .is_ok();
        }
        service_id(service).is_some_and(|id| {
            check_name(
                &Line::Ipc {
                    service: id,
                    command,
                },
                name,
                false,
            )
            .is_ok()
        })
    }

    /// The result on a reference `svc` line: `0x0` in `... -> 0x0 out=...`, or `none`.
    fn result_of(line: &str) -> String {
        line.split_once(" -> ")
            .and_then(|(_, rest)| rest.split(' ').next())
            .unwrap_or("none")
            .to_owned()
    }

    /// Every system call and request in `fixture`'s reference that starquake does not
    /// implement, once each, in the order of first use.
    fn gaps(fixture: &Fixture) -> Vec<Gap> {
        let send = format!("svc {:#04x} ", SvcId::SendSyncRequest.number());
        let lines: Vec<(usize, &str)> = fixture
            .reference
            .lines()
            .enumerate()
            .map(|(index, line)| (index + 1, line))
            .filter(|(_, line)| !line.starts_with('#'))
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        let mut gaps = Vec::new();
        for (position, (number, line)) in lines.iter().enumerate() {
            let gap = if let Some(call) = line.strip_prefix("svc ") {
                let (id, rest) = call.split_once(' ').expect("svc <number> <call>");
                let id = u16::from_str_radix(id.trim_start_matches("0x"), 16).expect("a number");
                let name = rest.split_once('(').expect("an argument list").0;
                (!implements_call(id)).then(|| (format!("svc {id:#04x} {name}"), result_of(line)))
            } else if let Some(request) = line.strip_prefix("ipc ") {
                let (service, command, name) = request_parts(request).expect("a request line");
                let id: u32 = command.parse().expect("a command id");
                let result = lines[position + 1..]
                    .iter()
                    .find(|(_, next)| next.starts_with(&send))
                    .map_or_else(|| "none".to_owned(), |(_, next)| result_of(next));
                (!implements_request(service, id, name)).then(|| ((*line).to_owned(), result))
            } else {
                None
            };
            if let Some((what, result)) = gap
                && seen.insert(what.clone())
            {
                gaps.push(Gap {
                    line: *number,
                    what,
                    result,
                });
            }
        }
        gaps
    }

    /// The first line of the gap list's block in a README.
    const GAPS_BEGIN: &str = "<!-- gap list: written by crates/core/tests/homebrew.rs -->";
    /// The last line of the gap list's block in a README.
    const GAPS_END: &str = "<!-- end of the gap list -->";

    /// The gap list as a README carries it, between [`GAPS_BEGIN`] and [`GAPS_END`].
    fn gap_table(gaps: &[Gap]) -> String {
        let mut table =
            String::from("| Line | Call or request | Reference's result |\n|---:|---|---|\n");
        for gap in gaps {
            writeln!(
                table,
                "| {} | `{}` | `{}` |",
                gap.line, gap.what, gap.result
            )
            .expect("writing to a string");
        }
        table
    }

    /// The text between [`GAPS_BEGIN`] and [`GAPS_END`] in `readme`.
    fn gap_block(readme: &str) -> Option<&str> {
        let (_, rest) = readme.split_once(&format!("{GAPS_BEGIN}\n"))?;
        let (block, _) = rest.split_once(GAPS_END)?;
        Some(block)
    }

    /// The comparison and the gap list against made-up runs and traces, on every host.
    #[cfg(test)]
    mod comparison {
        use starquake_kernel::memory::GUARD_PAGES;
        use starquake_kernel::result::KernelError;

        use super::*;

        /// Where the made-up runs put the module: starquake's address for it, and the fixture's
        /// size in memory (its image and a page of `.bss`).
        const BASE: u64 = 0x8000_0000;
        /// The two session handles of the made-up runs: not the reference's values, so that the
        /// pairing is what makes them match.
        const SM_HANDLE: u64 = 0x2_0011;
        const SET_HANDLE: u64 = 0x4_0012;

        fn anchors() -> Anchors {
            Anchors::new(
                BASE,
                module_extent(&HELLO),
                module_extent(&HELLO) + PAGE_SIZE,
            )
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
                    &[(0, buffer), (2, BASE + module_extent(&HELLO))],
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
            compare(&HELLO, lines, &anchors()).unwrap_err()
        }

        #[test]
        fn a_run_that_does_what_the_reference_did_passes() {
            assert_eq!(compare(&HELLO, &matching(), &anchors()), Ok(()));
            // The reference's 38 lines: 17 calls up to the port; five requests, each with its
            // `SendSyncRequest`; two close requests and their `CloseHandle`s; three log lines,
            // each with its `OutputDebugString`; and the exit.
            assert_eq!(matching().len(), reference(&HELLO).unwrap().len());
            assert_eq!(
                reference(&HELLO).unwrap().len(),
                17 + 5 * 2 + 2 * 2 + 3 * 2 + 1
            );
        }

        #[test]
        fn the_reference_is_read_as_the_module_docs_say() {
            let lines = reference(&HELLO).unwrap();
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
        fn the_startup_reference_is_read_the_same_way() {
            let lines = reference(&STARTUP).unwrap();
            // A control request other than `QueryPointerBufferSize` goes to the next service
            // too: `ConvertCurrentObjectToDomain` before `appletOE`'s `OpenApplicationProxy`.
            let index = lines
                .iter()
                .position(|(_, name)| name.as_deref() == Some("ConvertCurrentObjectToDomain"))
                .unwrap();
            assert_eq!(lines[index].0, "ipc appletOE::0");
            assert_eq!(
                lines[index + 2],
                (
                    "ipc appletOE::0".to_owned(),
                    Some("OpenApplicationProxy".to_owned())
                )
            );
            // A control request followed by another control request goes past it, to the
            // next request that is not one: `QueryPointerBufferSize` before the conversion, and
            // the two `CloneCurrentObject`s before `fsp-srv`'s `OpenSdCardFileSystem`.
            assert_eq!(
                lines[index - 2],
                (
                    "ipc appletOE::3".to_owned(),
                    Some("QueryPointerBufferSize".to_owned())
                )
            );
            let clones: Vec<&str> = lines
                .iter()
                .filter(|(_, name)| name.as_deref() == Some("CloneCurrentObject"))
                .map(|(line, _)| line.as_str())
                .collect();
            assert_eq!(clones, ["ipc fsp-srv::2", "ipc fsp-srv::2"]);
            // A random answer is an output, so it is dropped like the others.
            assert!(lines.iter().any(|(line, _)| line
                == "svc 0x29 GetInfo(info_type=0xb, handle=0x0, info_subtype=0x0) -> 0x0"));
            // The run ends with the request that ends the process.
            assert_eq!(
                lines.last().unwrap().0,
                "svc 0x21 SendSyncRequest(session_handle=0x303fb) -> 0x7601"
            );
        }

        #[test]
        fn where_the_module_and_the_regions_are_does_not_matter() {
            // The same run with the module where the reference emulator put it in one of its
            // runs: every address moves with its anchor.
            let moved = Anchors::new(
                0x80c0_1000,
                module_extent(&HELLO),
                module_extent(&HELLO) + PAGE_SIZE,
            );
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
                        for arg in args.iter_mut().take(3) {
                            *arg = shift(*arg);
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
            assert_eq!(compare(&HELLO, &lines, &moved), Ok(()));
            // Against the wrong anchors the same run fails at its first address.
            assert!(compare(&HELLO, &lines, &here).is_err());
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
            // A line of the same length: the hash tells them apart.
            assert_eq!(
                source_string("LINE_TWO").len(),
                source_string("LINE_ONE").len()
            );
            let mut lines = matching();
            lines[21] = log("LINE_TWO");
            let Line::Log { len, hash } = lines[21] else {
                unreachable!()
            };
            assert_eq!(
                failure(&lines),
                format!(
                    "line 21: expected `log starquake hello: line one`, found `log len={len} \
                     hash={hash:#x}`"
                )
            );
        }

        #[test]
        fn an_empty_log_line_does_not_match_a_line_of_another_kind() {
            let EventKind::GuestLog { len, hash } = EventKind::guest_log(b"") else {
                unreachable!("a guest log event")
            };
            let mut lines = matching();
            lines[0] = Line::Log { len, hash };
            assert_eq!(
                failure(&lines),
                format!(
                    "line 0: expected `svc 0x02 SetMemoryPermission(address=main+0x4000, \
                     size=0x1000, perm=0x1) -> 0x0`, found `log len={len} hash={hash:#x}`"
                )
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
            // A second handle whose value is the reference's value for the first one: it is
            // not paired, so it must not read as that value.
            let expected = &reference(&HELLO).unwrap()[20].0;
            let theirs = argument(expected, "session_handle").unwrap();
            let theirs = u64::from_str_radix(theirs.strip_prefix("0x").unwrap(), 16).unwrap();
            let mut lines = matching();
            lines[20] = call(SvcId::SendSyncRequest, &[(0, theirs)], 0);
            assert_eq!(
                failure(&lines),
                format!(
                    "line 20: expected `{expected}`, found `svc 0x21 \
                     SendSyncRequest(session_handle=unpaired {theirs:#x}) -> 0x0`"
                )
            );
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

        /// A call with the arguments `named` in their registers, by the call's convention.
        fn call_named(id: SvcId, named: &[(&str, u64)], result: u32) -> Line {
            let inputs = id.convention().unwrap().inputs;
            let registers: Vec<(usize, u64)> = named
                .iter()
                .map(|(name, value)| {
                    let input = inputs.iter().find(|input| input.name == *name).unwrap();
                    (usize::from(input.reg), *value)
                })
                .collect();
            call(id, &registers, result)
        }

        #[test]
        fn the_address_of_a_transfer_memory_and_of_a_handle_list_is_compared_by_anchor() {
            // Made-up lines in the startup trace's form (its lines 171 and 193), which the
            // startup run does not reach yet.
            let fixture = Fixture {
                nro: HELLO.nro,
                reference: "svc 0x15 CreateTransferMemory(address=heap_region+0x1000, \
                    size=0x800000, map_perm=0x0) -> 0x0 out_handle=0xf03e3\n\
                    svc 0x18 WaitSynchronization(handles=stack_region+0x103da0, num_handles=0x1, \
                    timeout_ns=0xffffffffffffffff) -> 0x0 out_index=0x0\n",
            };
            let regions = anchors().regions;
            let lines = |transfer: u64, handles: u64| {
                [
                    call_named(
                        SvcId::CreateTransferMemory,
                        &[("address", transfer), ("size", 0x80_0000), ("map_perm", 0)],
                        0,
                    ),
                    call_named(
                        SvcId::WaitSynchronization,
                        &[
                            ("handles", handles),
                            ("num_handles", 1),
                            ("timeout_ns", u64::MAX),
                        ],
                        0,
                    ),
                ]
            };
            let transfer = regions.heap.start + 0x1000;
            let handles = regions.stack.start + 0x10_3da0;
            assert_eq!(
                compare(&fixture, &lines(transfer, handles), &anchors()),
                Ok(())
            );
            // Each at another address of its region.
            assert_eq!(
                compare(&fixture, &lines(transfer + 0x1000, handles), &anchors()),
                Err(
                    "line 0: expected `svc 0x15 CreateTransferMemory(address=heap_region+0x1000, \
                     size=0x800000, map_perm=0x0) -> 0x0`, found `svc 0x15 \
                     CreateTransferMemory(address=heap_region+0x2000, size=0x800000, \
                     map_perm=0x0) -> 0x0`"
                        .to_owned()
                )
            );
            assert_eq!(
                compare(&fixture, &lines(transfer, handles + 8), &anchors()),
                Err("line 1: expected `svc 0x18 \
                     WaitSynchronization(handles=stack_region+0x103da0, num_handles=0x1, \
                     timeout_ns=0xffffffffffffffff) -> 0x0`, found `svc 0x18 \
                     WaitSynchronization(handles=stack_region+0x103da8, num_handles=0x1, \
                     timeout_ns=0xffffffffffffffff) -> 0x0`"
                    .to_owned())
            );
        }

        #[test]
        fn a_reference_request_that_cannot_be_read_differs_at_that_line() {
            // No parentheses, no closing one, and no `::` between the service and the command.
            for request in [
                "sm:::1 GetService",
                "sm:::1 (GetService",
                "sm: (GetService)",
            ] {
                let fixture = Fixture {
                    nro: HELLO.nro,
                    reference: format!("ipc sm:::0 (Initialize)\nipc {request}\n").leak(),
                };
                assert_eq!(
                    progress(&fixture, &[ipc(SM, 0), ipc(SM, 1)], &anchors()),
                    Progress {
                        matched: 0,
                        difference: Some(format!(
                            "line 1: the request `{request}` is not written \
                             `<service>::<command> (<name>)`"
                        ))
                    }
                );
            }
        }

        #[test]
        fn three_kinds_of_event_are_lines_and_no_other_is() {
            let svc = EventKind::Svc {
                id: 0x21,
                args: [7; SVC_ARG_COUNT],
                result: starquake_trace::ResultCode(0xf601),
            };
            assert_eq!(
                Line::of(&svc),
                Some(Line::Svc {
                    number: 0x21,
                    args: [7; SVC_ARG_COUNT],
                    result: 0xf601
                })
            );
            let request = EventKind::Ipc {
                session: starquake_trace::SessionId(5),
                service: SET_SYS,
                command: 3,
                request_size: 8,
                reply_size: 2,
                result: starquake_trace::ResultCode(0),
                outcome: starquake_trace::IpcOutcome::Done,
            };
            assert_eq!(
                Line::of(&request),
                Some(Line::Ipc {
                    service: SET_SYS,
                    command: 3
                })
            );
            assert_eq!(
                Line::of(&EventKind::GuestLog { len: 4, hash: 9 }),
                Some(Line::Log { len: 4, hash: 9 })
            );
            let start = EventKind::Lifecycle {
                phase: starquake_trace::LifecyclePhase::ProcessStart,
                subject: 1,
            };
            assert_eq!(Line::of(&start), None);
        }

        #[test]
        fn the_fixtures_strings_are_read_from_its_source() {
            assert_eq!(source_string("LINE_ONE"), "starquake hello: line one");
            assert_eq!(
                source_string("FIRMWARE_OK"),
                "starquake hello: set:sys firmware version ok"
            );
        }

        // ----- How far a run gets ---------------------------------------------------------

        #[test]
        fn progress_counts_the_lines_before_the_first_difference() {
            let all = reference(&HELLO).unwrap().len();
            assert_eq!(
                progress(&HELLO, &matching(), &anchors()),
                Progress {
                    matched: all,
                    difference: None
                }
            );
            // A run that stops after a line, at the last line, and one that differs at the first.
            for stop in [1, all - 1] {
                let progress = progress(&HELLO, &matching()[..stop], &anchors());
                assert_eq!(progress.matched, stop);
                assert!(
                    progress
                        .difference
                        .unwrap()
                        .ends_with("and the run has no more lines")
                );
            }
            let mut lines = matching();
            lines[0] = ipc(SM, 0);
            assert_eq!(
                progress(&HELLO, &lines, &anchors()),
                Progress {
                    matched: 0,
                    difference: Some(
                        "line 0: expected `svc 0x02 SetMemoryPermission(address=main+0x4000, \
                         size=0x1000, perm=0x1) -> 0x0`, found `ipc sm:::0`"
                            .to_owned()
                    )
                }
            );
            // A command the reference names differently from starquake's table stops it at that
            // line too, though the line's text matches.
            let renamed = Fixture {
                nro: HELLO.nro,
                reference: "ipc sm:::0 (Initialize)\nipc sm:::1 (Initialize)\n",
            };
            assert_eq!(
                progress(&renamed, &[ipc(SM, 0), ipc(SM, 1)], &anchors()),
                Progress {
                    matched: 1,
                    difference: Some(
                        "line 1: the reference calls command 1 Initialize, starquake \
                         Some(\"GetService\")"
                            .to_owned()
                    )
                }
            );
            // A line the comparison cannot write stops it at that line, with the lines before
            // it matched.
            for at in [0, 5] {
                let mut lines = matching();
                lines[at] = Line::Svc {
                    number: 0x7777,
                    args: [0; SVC_ARG_COUNT],
                    result: 0,
                };
                assert_eq!(
                    progress(&HELLO, &lines, &anchors()),
                    Progress {
                        matched: at,
                        difference: Some(format!("line {at}: svc 0x7777 is not a listed call"))
                    }
                );
            }
            // A run that goes on matches every line and still differs.
            let mut lines = matching();
            lines.push(ipc(SM, 0));
            let progress = progress(&HELLO, &lines, &anchors());
            assert_eq!(progress.matched, all);
            assert!(progress.difference.is_some());
        }

        #[test]
        fn a_reference_that_ends_with_a_control_request_differs_at_that_line() {
            let truncated = Fixture {
                nro: HELLO.nro,
                reference: "ipc sm:::0 (Initialize)\n\
                    ipc IpcController::3 (QueryPointerBufferSize)\n",
            };
            assert_eq!(
                progress(&truncated, &[ipc(SM, 0), ipc(SM, 3)], &anchors()),
                Progress {
                    matched: 0,
                    difference: Some(
                        "line 1: the control request `ipc IpcController::3 \
                         (QueryPointerBufferSize)` has no request after it"
                            .to_owned()
                    )
                }
            );
        }

        #[test]
        fn the_ratchet_is_inside_the_startup_reference() {
            assert!(STARTUP_MATCHED <= reference(&STARTUP).unwrap().len());
        }

        // ----- Placements -----------------------------------------------------------------

        /// A made-up reference that probes and maps two placements, with the hello fixture's
        /// module.
        const PLACED: Fixture = Fixture {
            nro: HELLO.nro,
            reference: "# comment\n\
                svc 0x06 QueryMemory(out_memory_info=stack_region+0x0, address=placement1+0x0) -> 0x0\n\
                svc 0x13 MapSharedMemory(shmem_handle=0x583f6, address=placement1+0x1000, size=0x40000, map_perm=0x1) -> 0x0\n\
                svc 0x13 MapSharedMemory(shmem_handle=0x903ef, address=placement2+0x1000, size=0x1000, map_perm=0x1) -> 0x0\n\
                svc 0x14 UnmapSharedMemory(shmem_handle=0x583f6, address=placement1+0x1000, size=0x40000) -> 0x0\n",
        };

        /// Where the made-up runs put the two placements: in the ASLR region, far above the
        /// module and below the other regions.
        fn placed(first: u64, second: u64) -> Vec<Line> {
            let stack = anchors().regions.stack.start;
            vec![
                call(SvcId::QueryMemory, &[(0, stack), (2, first)], 0),
                call(
                    SvcId::MapSharedMemory,
                    &[(0, 0x1_0001), (1, first + 0x1000), (2, 0x4_0000), (3, 1)],
                    0,
                ),
                call(
                    SvcId::MapSharedMemory,
                    &[(0, 0x1_0002), (1, second + 0x1000), (2, 0x1000), (3, 1)],
                    0,
                ),
                call(
                    SvcId::UnmapSharedMemory,
                    &[(0, 0x1_0001), (1, first + 0x1000), (2, 0x4_0000)],
                    0,
                ),
            ]
        }

        /// Two pages of the ASLR region that no other region and not the module holds.
        fn free_pages() -> (u64, u64) {
            let anchors = anchors();
            let first = anchors.main.end.next_multiple_of(PAGE_SIZE) + 0x10_0000;
            let second = first + 0x10_0000;
            for page in [first, second] {
                assert!(
                    anchors
                        .symbol(page)
                        .unwrap()
                        .starts_with(&format!("{ASLR_REGION}+"))
                );
            }
            (first, second)
        }

        #[test]
        fn a_placement_is_paired_where_the_reference_first_names_it() {
            let (first, second) = free_pages();
            assert_eq!(compare(&PLACED, &placed(first, second), &anchors()), Ok(()));
            // The same two placements, the other way round: each is paired on first use.
            assert_eq!(compare(&PLACED, &placed(second, first), &anchors()), Ok(()));
        }

        #[test]
        fn a_placement_must_stay_where_it_was_paired() {
            let (first, second) = free_pages();
            // The unmap at another page than the probe and the map paired.
            let mut lines = placed(first, second);
            let Line::Svc { args, .. } = &mut lines[3] else {
                unreachable!()
            };
            args[1] += PAGE_SIZE;
            assert_eq!(
                compare(&PLACED, &lines, &anchors()),
                Err(
                    "line 3: expected `svc 0x14 UnmapSharedMemory(shmem_handle=0x583f6, \
                     address=placement1+0x1000, size=0x40000) -> 0x0`, found `svc 0x14 \
                     UnmapSharedMemory(shmem_handle=0x583f6, address=placement1+0x2000, \
                     size=0x40000) -> 0x0`"
                        .to_owned()
                )
            );
            // The map below the probe that paired it.
            let mut lines = placed(first, second);
            let Line::Svc { args, .. } = &mut lines[1] else {
                unreachable!()
            };
            args[1] = first - PAGE_SIZE;
            let below = first - PAGE_SIZE - anchors().regions.code.start;
            assert_eq!(
                compare(&PLACED, &lines, &anchors()),
                Err(format!(
                    "line 1: expected `svc 0x13 MapSharedMemory(shmem_handle=0x583f6, \
                     address=placement1+0x1000, size=0x40000, map_perm=0x1) -> 0x0`, found `svc \
                     0x13 MapSharedMemory(shmem_handle=0x583f6, address=aslr_region+{below:#x}, \
                     size=0x40000, map_perm=0x1) -> 0x0`"
                ))
            );
        }

        #[test]
        fn two_placements_cannot_share_a_page_of_the_run() {
            let (first, _) = free_pages();
            let offset = first - anchors().regions.code.start + 0x1000;
            assert_eq!(
                compare(&PLACED, &placed(first, first), &anchors()),
                Err(format!(
                    "line 2: expected `svc 0x13 MapSharedMemory(shmem_handle=0x903ef, \
                     address=placement2+0x1000, size=0x1000, map_perm=0x1) -> 0x0`, found `svc \
                     0x13 MapSharedMemory(shmem_handle=0x903ef, address=aslr_region+{offset:#x}, \
                     size=0x1000, map_perm=0x1) -> 0x0`"
                ))
            );
        }

        /// The made-up runs against [`PLACED`] with the second placement's map at `address`.
        fn second_map_at(address: u64) -> Result<(), String> {
            let (first, second) = free_pages();
            let mut lines = placed(first, second);
            let Line::Svc { args, .. } = &mut lines[2] else {
                unreachable!()
            };
            args[1] = address;
            compare(&PLACED, &lines, &anchors())
        }

        /// The difference [`second_map_at`] reports when its map does not pair: the address
        /// is written as `found`.
        fn second_refused(found: &str) -> Result<(), String> {
            Err(format!(
                "line 2: expected `svc 0x13 MapSharedMemory(shmem_handle=0x903ef, \
                 address=placement2+0x1000, size=0x1000, map_perm=0x1) -> 0x0`, found `svc \
                 0x13 MapSharedMemory(shmem_handle=0x903ef, address={found}, size=0x1000, \
                 map_perm=0x1) -> 0x0`"
            ))
        }

        /// The difference the made-up runs report when the first placement's map, at `found`,
        /// does not pair though its probe did.
        fn first_map_refused(found: &str) -> Result<(), String> {
            Err(format!(
                "line 1: expected `svc 0x13 MapSharedMemory(shmem_handle=0x583f6, \
                 address=placement1+0x1000, size=0x40000, map_perm=0x1) -> 0x0`, found `svc \
                 0x13 MapSharedMemory(shmem_handle=0x583f6, address={found}, size=0x40000, \
                 map_perm=0x1) -> 0x0`"
            ))
        }

        /// The difference the made-up runs report when the first placement's probe, at
        /// `found`, does not pair.
        fn probe_refused(found: &str) -> Result<(), String> {
            Err(format!(
                "line 0: expected `svc 0x06 QueryMemory(out_memory_info=stack_region+0x0, \
                 address=placement1+0x0) -> 0x0`, found `svc 0x06 \
                 QueryMemory(out_memory_info=stack_region+0x0, address={found}) -> 0x0`"
            ))
        }

        /// The size of the first placement's range in [`PLACED`] and [`placed`].
        const FIRST_SIZE: u64 = 0x4_0000;

        /// [`anchors`] with the heap region moved a gigabyte above its place. In the layout
        /// the made-up runs use, the alias region ends where the heap region starts, so no
        /// placement can lie just below the heap region; here free pages of the ASLR region do.
        fn moved_heap() -> Anchors {
            let mut moved = anchors();
            let heap = moved.regions.heap;
            let start = heap.end + 0x4000_0000;
            moved.regions.heap = Region {
                start,
                end: start + heap.size(),
            };
            assert!(moved.regions.code.holds(start, heap.size()));
            moved
        }

        #[test]
        fn a_placement_is_not_paired_where_libnx_does_not_place_one() {
            let anchors = anchors();
            let regions = anchors.regions;
            let (_, second) = free_pages();
            // The probe, and so the range above it, in the heap region.
            assert_eq!(
                compare(&PLACED, &placed(regions.heap.start, second), &anchors),
                probe_refused("heap_region+0x0")
            );
            // The probe in the module, at its first page and at its last address, which libnx
            // asks about (`__argdata__`); a page past that it pairs
            // (`a_placement_may_lie_next_to_the_module_and_the_other_regions`).
            assert_eq!(
                compare(&PLACED, &placed(BASE, second), &anchors),
                probe_refused("main+0x0")
            );
            let extent = anchors.main.end - anchors.main.start;
            assert_eq!(
                compare(&PLACED, &placed(anchors.main.end, second), &anchors),
                probe_refused(&format!("main+{extent:#x}"))
            );
            // The probe on the stack region's last page, so that the range above it starts at
            // the alias region's first byte: the probe alone would be placeable, the range is
            // not.
            let probe = regions.alias.start - PLACEMENT_GUARD;
            assert!(regions.stack.holds(probe, PLACEMENT_GUARD));
            assert_eq!(
                compare(&PLACED, &placed(probe, second), &anchors),
                probe_refused(&format!("stack_region+{:#x}", probe - regions.stack.start))
            );
            // The second placement's map (no probe names it first) at the alias region's
            // first byte, whose anchor is in the stack region.
            assert_eq!(
                second_map_at(regions.alias.start),
                second_refused("alias_region+0x0")
            );
            // At the ASLR region's first byte: its anchor is the page below the region, which
            // the comparison refuses (a divergence the module docs list).
            assert_eq!(
                second_map_at(regions.code.start),
                second_refused(&format!("{ASLR_REGION}+0x0"))
            );
            // A line that names a placement past its anchor first, without mapping it, at the
            // ASLR region's first byte: its anchor would be the page below the region.
            let unmapped = Fixture {
                nro: HELLO.nro,
                reference: "svc 0x06 QueryMemory(out_memory_info=stack_region+0x0, \
                    address=placement1+0x1000) -> 0x0\n",
            };
            let query = call(
                SvcId::QueryMemory,
                &[(0, regions.stack.start), (2, regions.code.start)],
                0,
            );
            assert_eq!(
                compare(&unmapped, &[query], &anchors),
                Err(format!(
                    "line 0: expected `svc 0x06 QueryMemory(out_memory_info=stack_region+0x0, \
                     address=placement1+0x1000) -> 0x0`, found `svc 0x06 \
                     QueryMemory(out_memory_info=stack_region+0x0, address={ASLR_REGION}+0x0) \
                     -> 0x0`"
                ))
            );
            // Just past the module, so that its anchor (a page lower) is inside the module.
            let past = anchors.main.end + 1;
            assert_eq!(
                second_map_at(past),
                second_refused(&format!("{ASLR_REGION}+{:#x}", past - regions.code.start))
            );
        }

        #[test]
        fn a_placement_may_lie_next_to_the_module_and_the_other_regions() {
            // libnx keeps a placement's range, not its guard pages, out of the alias and the
            // heap region (`_memregionFindRandom`), and the kernel lets a shared memory be
            // mapped in the stack region.
            let anchors = anchors();
            let regions = anchors.regions;
            let (first, second) = free_pages();
            let pairs = |first: u64, second: u64| {
                assert_eq!(
                    compare(&PLACED, &placed(first, second), &anchors),
                    Ok(()),
                    "{first:#x} {second:#x}"
                );
            };
            // In the stack region: the probe at its first page, and one further in.
            for probe in [regions.stack.start, regions.stack.start + 0x10_0000] {
                assert!(regions.stack.holds(probe + PLACEMENT_GUARD, 1));
                pairs(probe, second);
            }
            // The range ending exactly where the alias region starts: its upper guard page is
            // the alias region's first.
            pairs(regions.alias.start - FIRST_SIZE - PLACEMENT_GUARD, second);
            assert_eq!(second_map_at(regions.alias.start - PAGE_SIZE), Ok(()));
            // The range starting exactly at the heap region's end: its anchor is the heap
            // region's last page. The alias region ends where the heap region starts here, so
            // a range at the alias region's end lies in the heap region.
            assert_eq!(regions.alias.end, regions.heap.start);
            pairs(regions.heap.end - PLACEMENT_GUARD, second);
            assert_eq!(second_map_at(regions.heap.end), Ok(()));
            // The range ending exactly where the heap region starts, with the heap region moved
            // (`moved_heap`) so that free pages of the ASLR region lie below it.
            let moved = moved_heap();
            let probe = moved.regions.heap.start - FIRST_SIZE - PLACEMENT_GUARD;
            assert_eq!(compare(&PLACED, &placed(probe, second), &moved), Ok(()));
            // The probe one page past the module's last address.
            pairs(anchors.main.end + PAGE_SIZE, second);
            // The second range's upper guard page just below the module, and its anchor just
            // above the first placement's range (that range's upper guard page).
            assert_eq!(
                second_map_at(anchors.main.start - PAGE_SIZE - PLACEMENT_GUARD),
                Ok(())
            );
            pairs(first, first + PLACEMENT_GUARD + FIRST_SIZE);
            // The second range's upper guard page on the first range's anchor, and the range
            // ending just below the ASLR region's end, with its upper guard page the region's
            // last.
            pairs(first, first - 2 * PLACEMENT_GUARD);
            assert_eq!(
                second_map_at(regions.code.end - PAGE_SIZE - PLACEMENT_GUARD),
                Ok(())
            );
        }

        #[test]
        fn a_placement_range_must_lie_where_libnx_maps_one() {
            let anchors = anchors();
            let regions = anchors.regions;
            let (_, second) = free_pages();
            let symbol = |address: u64| anchors.symbol(address).unwrap();
            // A probe that pairs, whose 0x40000-byte range reaches a page into the alias
            // region, and one into the heap region, moved so that the pages below it are free.
            let moved = moved_heap();
            for (anchors, region) in [(anchors, regions.alias), (moved, moved.regions.heap)] {
                let probe = region.start - FIRST_SIZE;
                let map = probe + PLACEMENT_GUARD;
                assert!(!overlaps(
                    Region {
                        start: probe,
                        end: map
                    },
                    region
                ));
                assert_eq!(
                    compare(&PLACED, &placed(probe, second), &anchors),
                    first_map_refused(&anchors.symbol(map).unwrap())
                );
            }
            // The second range with its upper guard page on the module's first page.
            let map = anchors.main.start - PAGE_SIZE;
            assert_eq!(second_map_at(map), second_refused(&symbol(map)));
            // The second range ending at the ASLR region's end, so that its upper guard page
            // lies above the region, outside the address space.
            assert_eq!(regions.code.end, regions.address_space.end);
            let map = regions.code.end - PAGE_SIZE;
            assert_eq!(second_map_at(map), second_refused(&symbol(map)));
        }

        #[test]
        fn two_placements_cannot_overlap() {
            let (first, _) = free_pages();
            let anchors = anchors();
            let offset = |address: u64| address - anchors.regions.code.start;
            // The second anchored inside the first's range, the second's map two pages in.
            for second in [
                first + 2 * PLACEMENT_GUARD,
                first + FIRST_SIZE,
                // Below the first, with the second's upper guard page the first range's first.
                first - PLACEMENT_GUARD,
            ] {
                let map = second + PLACEMENT_GUARD;
                assert_eq!(
                    compare(&PLACED, &placed(first, second), &anchors),
                    second_refused(&format!("{ASLR_REGION}+{:#x}", offset(map))),
                    "{second:#x}"
                );
            }
        }

        /// A made-up reference that maps a placement, unmaps it, and maps another.
        const REPLACED: Fixture = Fixture {
            nro: HELLO.nro,
            reference: "svc 0x13 MapSharedMemory(shmem_handle=0x583f6, address=placement1+0x1000, size=0x1000, map_perm=0x1) -> 0x0\n\
                svc 0x14 UnmapSharedMemory(shmem_handle=0x583f6, address=placement1+0x1000, size=0x1000) -> 0x0\n\
                svc 0x13 MapSharedMemory(shmem_handle=0x903ef, address=placement2+0x1000, size=0x1000, map_perm=0x1) -> 0x0\n",
        };

        #[test]
        fn a_placement_may_be_mapped_again_where_it_was() {
            // Its own earlier range does not count as taken.
            let fixture = Fixture {
                nro: HELLO.nro,
                reference: "svc 0x13 MapSharedMemory(shmem_handle=0x583f6, \
                    address=placement1+0x1000, size=0x1000, map_perm=0x1) -> 0x0\n\
                    svc 0x14 UnmapSharedMemory(shmem_handle=0x583f6, \
                    address=placement1+0x1000, size=0x1000) -> 0x0\n\
                    svc 0x13 MapSharedMemory(shmem_handle=0x583f6, \
                    address=placement1+0x1000, size=0x1000, map_perm=0x1) -> 0x0\n",
            };
            let (first, _) = free_pages();
            let map = call(
                SvcId::MapSharedMemory,
                &[
                    (0, 0x1_0001),
                    (1, first + PLACEMENT_GUARD),
                    (2, PAGE_SIZE),
                    (3, 1),
                ],
                0,
            );
            let unmap = call(
                SvcId::UnmapSharedMemory,
                &[(0, 0x1_0001), (1, first + PLACEMENT_GUARD), (2, PAGE_SIZE)],
                0,
            );
            assert_eq!(compare(&fixture, &[map, unmap, map], &anchors()), Ok(()));
        }

        #[test]
        fn a_placement_over_an_unmapped_one_does_not_pair() {
            // A divergence the module docs list: libnx could map the second placement where
            // the first was, but the comparison keeps the first's range taken.
            let (first, second) = free_pages();
            let replaced = |second: u64| {
                let map = |handle, at: u64| {
                    call(
                        SvcId::MapSharedMemory,
                        &[
                            (0, handle),
                            (1, at + PLACEMENT_GUARD),
                            (2, PAGE_SIZE),
                            (3, 1),
                        ],
                        0,
                    )
                };
                let unmap = call(
                    SvcId::UnmapSharedMemory,
                    &[(0, 0x1_0001), (1, first + PLACEMENT_GUARD), (2, PAGE_SIZE)],
                    0,
                );
                compare(
                    &REPLACED,
                    &[map(0x1_0001, first), unmap, map(0x1_0002, second)],
                    &anchors(),
                )
            };
            assert_eq!(replaced(second), Ok(()));
            let offset = first + PLACEMENT_GUARD - anchors().regions.code.start;
            assert_eq!(
                replaced(first + PLACEMENT_GUARD),
                Err(format!(
                    "line 2: expected `svc 0x13 MapSharedMemory(shmem_handle=0x903ef, \
                     address=placement2+0x1000, size=0x1000, map_perm=0x1) -> 0x0`, found `svc \
                     0x13 MapSharedMemory(shmem_handle=0x903ef, \
                     address={ASLR_REGION}+{:#x}, size=0x1000, map_perm=0x1) -> 0x0`",
                    offset + PLACEMENT_GUARD
                ))
            );
        }

        #[test]
        fn the_aslr_region_ends_where_the_address_space_does() {
            // What the module docs rely on to refuse a guard page above the ASLR region.
            let regions = anchors().regions;
            assert_eq!(regions.code.end, regions.address_space.end);
            assert!(regions.address_space.start < regions.code.start);
        }

        #[test]
        fn a_placement_name_is_read_only_from_a_placement_address() {
            assert_eq!(placement("placement1+0x1000"), Some(("placement1", 0x1000)));
            assert_eq!(placement("placement12+0x0"), Some(("placement12", 0)));
            assert_eq!(placement("placement+0x0"), None);
            assert_eq!(placement("placementx+0x0"), None);
            assert_eq!(placement("aslr_region+0x1000"), None);
            assert_eq!(placement("placement1+1000"), None);
            assert_eq!(placement("placement1"), None);
        }
    }

    /// The gap list against made-up traces, and the startup fixture's README against its trace.
    #[cfg(test)]
    mod gap_list {
        use super::*;

        /// A made-up trace with one line of each case.
        const TRACE: &str = "# comment\n\
            svc 0x29 GetInfo(info_type=0xb, handle=0x0, info_subtype=0x0) -> 0x0 out=random\n\
            ipc IpcController::3 (QueryPointerBufferSize)\n\
            svc 0x21 SendSyncRequest(session_handle=0x203fd) -> 0x0\n\
            ipc apm::0 (OpenSession)\n\
            svc 0x21 SendSyncRequest(session_handle=0x203fd) -> 0x0\n\
            ipc sm:::1 (GetService)\n\
            svc 0x21 SendSyncRequest(session_handle=0x103fe) -> 0x0\n\
            ipc set:sys::3 (GetSettingsItemValue)\n\
            svc 0x15 CreateTransferMemory(address=heap_region+0x1000, size=0x800000, map_perm=0x0) -> 0xd401\n\
            svc 0x21 SendSyncRequest(session_handle=0x183fd) -> 0xf601\n\
            svc 0x15 CreateTransferMemory(address=heap_region+0x1000, size=0x800000, map_perm=0x0) -> 0x0 out_handle=0xf03e3\n\
            ipc apm::0 (OpenSession)\n\
            svc 0x21 SendSyncRequest(session_handle=0x203fd) -> 0x0\n\
            ipc IpcController::7 (Unheard)\n\
            ipc nvdrv::3 (Initialize)\n";

        fn made_up() -> Vec<Gap> {
            gaps(&Fixture {
                nro: HELLO.nro,
                reference: TRACE,
            })
        }

        fn gap(line: usize, what: &str, result: &str) -> Gap {
            Gap {
                line,
                what: what.to_owned(),
                result: result.to_owned(),
            }
        }

        #[test]
        fn what_starquake_lacks_is_listed_once_in_order_of_first_use() {
            assert_eq!(
                made_up(),
                [
                    // An implemented call (GetInfo), a control command the framework answers
                    // and a command `sm:` has are not gaps.
                    gap(5, "ipc apm::0 (OpenSession)", "0x0"),
                    // A command id a service has under another name is one; a request's result
                    // is its `SendSyncRequest`'s, past the calls in between.
                    gap(9, "ipc set:sys::3 (GetSettingsItemValue)", "0xf601"),
                    // A planned call, with the result of its first use.
                    gap(10, "svc 0x15 CreateTransferMemory", "0xd401"),
                    // A control command the framework does not have, and a request with no
                    // `SendSyncRequest` after it.
                    gap(15, "ipc IpcController::7 (Unheard)", "none"),
                    gap(16, "ipc nvdrv::3 (Initialize)", "none"),
                ]
            );
        }

        #[test]
        fn what_starquake_implements_is_decided_by_its_tables() {
            assert!(implements_call(SvcId::GetInfo.number()));
            assert!(!implements_call(SvcId::CreateTransferMemory.number()));
            assert!(!implements_call(0x7777));
            assert!(implements_request(
                CONTROL_INTERFACE,
                3,
                "QueryPointerBufferSize"
            ));
            assert!(!implements_request(CONTROL_INTERFACE, 3, "GetService"));
            assert!(implements_request("sm:", 1, "GetService"));
            assert!(!implements_request("sm:", 1, "Initialize"));
            assert!(implements_request("set:sys", 3, "GetFirmwareVersion"));
            assert!(!implements_request("apm", 0, "OpenSession"));
            assert_eq!(service_id("set:sys"), Some(SET_SYS));
            assert_eq!(service_id("apm"), None);
        }

        #[test]
        fn the_table_has_one_row_per_gap() {
            assert_eq!(
                gap_table(&made_up()[..2]),
                "| Line | Call or request | Reference's result |\n\
                 |---:|---|---|\n\
                 | 5 | `ipc apm::0 (OpenSession)` | `0x0` |\n\
                 | 9 | `ipc set:sys::3 (GetSettingsItemValue)` | `0xf601` |\n"
            );
            let readme = format!("before\n{GAPS_BEGIN}\nthe block\n{GAPS_END}\nafter\n");
            assert_eq!(gap_block(&readme), Some("the block\n"));
            assert_eq!(gap_block("no block\n"), None);
            assert_eq!(gap_block(&format!("{GAPS_BEGIN}\nunended\n")), None);
        }

        #[test]
        fn the_startup_gap_list_in_its_readme_is_the_one_the_trace_gives() {
            let expected = gap_table(&gaps(&STARTUP));
            assert_eq!(
                gap_block(STARTUP_README),
                Some(expected.as_str()),
                "tests/homebrew/startup/reference/README.md's gap list is not the one its trace \
                 gives; replace the lines between its two markers with:\n{expected}"
            );
        }
    }

    /// The fixtures on a real vCPU.
    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    mod on_hardware {
        use starquake_core::{ExitReason, RunReport, exit_code};
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

        /// Where the run of `fixture` put the module, from its memory map.
        fn anchors(fixture: &Fixture, report: &RunReport) -> Anchors {
            let module: Vec<_> = report
                .memory_map
                .iter()
                .filter(|mapping| mapping.name.starts_with("main."))
                .collect();
            let base = module.iter().map(|mapping| mapping.start).min().unwrap();
            let end = module.iter().map(|mapping| mapping.end).max().unwrap();
            Anchors::new(
                base,
                module_extent(fixture),
                end.next_multiple_of(PAGE_SIZE) - base,
            )
        }

        #[test]
        fn the_fixture_matches_the_reference_line_for_line_through_exit_process() {
            let Some(report) = run(HELLO.nro, GENEROUS) else {
                return;
            };
            assert_eq!(
                compare(&HELLO, &lines(&report), &anchors(&HELLO, &report)),
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
            assert_eq!(report.stubbed, std::collections::BTreeSet::new());
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

        #[test]
        fn the_startup_fixture_matches_the_reference_at_least_as_far_as_recorded() {
            let Some(report) = run(STARTUP.nro, GENEROUS) else {
                return;
            };
            let progress = progress(&STARTUP, &lines(&report), &anchors(&STARTUP, &report));
            assert!(
                progress.matched >= STARTUP_MATCHED,
                "the startup fixture's run matches its reference for {} lines, fewer than the \
                 {STARTUP_MATCHED} recorded; first difference: {:?}\n{}",
                progress.matched,
                progress.difference,
                report.trace
            );
            if progress.matched > STARTUP_MATCHED {
                eprintln!(
                    "the startup fixture's run now matches {} lines: raise STARTUP_MATCHED",
                    progress.matched
                );
            }
        }
    }
}
