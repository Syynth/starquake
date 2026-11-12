//! The homebrew fixture (`tests/homebrew/hello/hello.nro`, a real libnx program) against the
//! reference emulator: its system calls in starquake are compared, call by call, with the
//! normalised trace of the same file under Eden-NX2
//! (`tests/homebrew/hello/reference/startup.trace`; the README beside it has the provenance
//! and the format), up to `ConnectToNamedPort("sm:")`, which starquake does not have yet.
//!
//! A call matches when its number, its decoded arguments and its result are the reference's.
//! Addresses are compared as `anchor+offset` (the module, or a region of the address space),
//! so the two emulators may place the module and the regions differently but must agree on
//! everything inside them: which page the program reprotects, where its stack pointer is in
//! the stack region. Outputs other than the result are not in starquake's event stream; what
//! `GetInfo` answers is checked against the same reference in `starquake-kernel`
//! (`svc/info/tests.rs`, `memory/layout.rs`).
//!
//! The comparison itself runs on every host, against made-up runs that must fail it: the run
//! starquake made before it had these calls, a wrong argument, a missing call. The run on a
//! real vCPU needs a Mac with Apple Silicon and skips itself only when `CI` is set.

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod common;

/// Everything is in a `cfg(test)` module so that the helpers may unwrap like the tests do.
#[cfg(test)]
mod trace {
    use starquake_kernel::memory::MemoryArrangement;
    use starquake_kernel::memory::layout::{LayoutRequest, Region, RegionOffsets, Regions};
    use starquake_kernel::svc::SvcId;
    use starquake_trace::SVC_ARG_COUNT;

    /// The reference emulator's run, normalised.
    const REFERENCE: &str = include_str!("../../../tests/homebrew/hello/reference/startup.trace");

    /// The fixture.
    const HELLO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nro");

    /// Where an NRO's header keeps the size of its image (`starquake_loader::nro`; the same offset
    /// `normalise.py` reads).
    const NRO_IMAGE_SIZE_OFFSET: usize = 0x18;

    /// The call the comparison stops at: the first one starquake does not have.
    const LAST: SvcId = SvcId::ConnectToNamedPort;

    /// The arguments that are addresses, for each call the fixture makes up to [`LAST`]: the same
    /// list as `ADDRESSES` in `tests/homebrew/hello/reference/normalise.py`. A call that is not
    /// listed is refused, so no address is compared as a number by oversight.
    const ADDRESS_ARGUMENTS: &[(SvcId, &[&str])] = &[
        (SvcId::SetHeapSize, &[]),
        (SvcId::SetMemoryPermission, &["address"]),
        (SvcId::SetMemoryAttribute, &["address"]),
        (SvcId::QueryMemory, &["out_memory_info", "address"]),
        (SvcId::GetInfo, &[]),
        (SvcId::ConnectToNamedPort, &["name"]),
    ];

    /// One system call of a run: what the event stream keeps of it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Call {
        number: u32,
        /// The decoded arguments, each in the slot of its register.
        args: [u64; SVC_ARG_COUNT],
        result: u32,
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

    /// The part of a trace line that names the call and its arguments:
    /// `svc 0x02 SetMemoryPermission(address=main+0x4000, size=0x1000, perm=0x1)`.
    fn call_text(call: &Call, anchors: &Anchors) -> Result<String, String> {
        let id = u16::try_from(call.number)
            .ok()
            .and_then(SvcId::from_number)
            .ok_or_else(|| format!("svc {:#x} is not a listed call", call.number))?;
        let convention = id
            .convention()
            .ok_or_else(|| format!("{} has no register convention", id.name()))?;
        let (_, addresses) = ADDRESS_ARGUMENTS
            .iter()
            .find(|(listed, _)| *listed == id)
            .ok_or_else(|| format!("no address rule for {}", id.name()))?;
        let mut arguments = Vec::new();
        for input in convention.inputs {
            let value = call.args[usize::from(input.reg)];
            let text = if addresses.contains(&input.name) {
                anchors.symbol(value)?
            } else {
                format!("{value:#x}")
            };
            arguments.push(format!("{}={text}", input.name));
        }
        Ok(format!(
            "svc {:#04x} {}({})",
            call.number,
            id.name(),
            arguments.join(", ")
        ))
    }

    /// A whole trace line for `call`, as the reference has it without its outputs.
    fn line(call: &Call, anchors: &Anchors) -> Result<String, String> {
        Ok(format!(
            "{} -> {:#x}",
            call_text(call, anchors)?,
            call.result
        ))
    }

    /// The reference's system calls up to and including [`LAST`], each without its outputs:
    /// `svc 0x01 SetHeapSize(size=0xcd000000) -> 0x0`.
    fn reference_calls() -> Vec<String> {
        let mut calls = Vec::new();
        for text in REFERENCE.lines().filter(|text| text.starts_with("svc ")) {
            // Everything up to the result: outputs follow it after a space.
            let (call, rest) = text.split_once(") -> ").expect("a call with a result");
            let result = rest.split(' ').next().expect("a result");
            calls.push(format!("{call}) -> {result}"));
            if call.contains(&format!(" {}(", LAST.name())) {
                return calls;
            }
        }
        unreachable!("the reference trace has no {} call", LAST.name());
    }

    /// Compares a run's calls with the reference's, in order, up to [`LAST`]: every call before
    /// it must be the reference's line exactly (call, arguments, result), and [`LAST`] itself the
    /// reference's call and arguments. What the run did after it is not looked at. The error
    /// names the first call that differs.
    fn compare(ours: &[Call], anchors: &Anchors) -> Result<(), String> {
        let expected = reference_calls();
        let last = expected.len() - 1;
        for (index, expected) in expected.iter().enumerate() {
            let Some(call) = ours.get(index) else {
                return Err(format!(
                    "call {index}: expected `{expected}`, and the run made no more calls"
                ));
            };
            let (found, expected) = if index == last {
                // The reference's result for the last call is not starquake's yet.
                let (expected, _) = expected.split_once(" -> ").expect("a result");
                (call_text(call, anchors), expected)
            } else {
                (line(call, anchors), expected.as_str())
            };
            let found = found.map_err(|error| format!("call {index}: {error}"))?;
            if found != expected {
                return Err(format!(
                    "call {index}: expected `{expected}`, found `{found}`"
                ));
            }
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

        fn anchors() -> Anchors {
            Anchors::new(BASE, image_size(), image_size() + PAGE_SIZE)
        }

        fn call(id: SvcId, registers: &[(usize, u64)], result: u32) -> Call {
            let mut args = [0; SVC_ARG_COUNT];
            for (register, value) in registers {
                args[*register] = *value;
            }
            Call {
                number: u32::from(id.number()),
                args,
                result,
            }
        }

        /// A run that does what the reference did, with every address where starquake puts it:
        /// the module at [`BASE`], the stack four pages into the stack region.
        fn matching() -> Vec<Call> {
            let anchors = anchors();
            let regions = anchors.regions;
            let relro = BASE + 0x4000;
            let process = 0xffff_8001;
            // libnx's two buffers on its stack, 0x38 bytes below the stack's top.
            let stack_top = regions.stack.start + GUARD_PAGES * PAGE_SIZE + 0x10_0000;
            let buffer = stack_top - 0x38;
            let mut calls = vec![
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
                calls.push(call(SvcId::GetInfo, &[(1, info_type), (2, process)], 0));
            }
            calls.extend([
                call(SvcId::SetHeapSize, &[(1, 0xcd00_0000)], 0),
                call(SvcId::QueryMemory, &[(0, buffer), (2, buffer)], 0),
                call(
                    SvcId::QueryMemory,
                    &[(0, buffer), (2, BASE + image_size())],
                    0,
                ),
                call(SvcId::ConnectToNamedPort, &[(1, BASE + 0x3148)], 0),
            ]);
            calls
        }

        #[test]
        fn a_run_that_does_what_the_reference_did_passes() {
            assert_eq!(compare(&matching(), &anchors()), Ok(()));
            // The reference's 17 calls: two on the RELRO page, eleven `GetInfo`s, the heap, two
            // queries and the port.
            assert_eq!(matching().len(), reference_calls().len());
            assert_eq!(reference_calls().len(), 17);
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
            let calls: Vec<Call> = matching()
                .into_iter()
                .map(|mut call| {
                    // Registers 0 to 2 hold every address these calls take; sizes and ids are
                    // too small to be taken for one.
                    for register in 0..3 {
                        call.args[register] = shift(call.args[register]);
                    }
                    call
                })
                .collect();
            assert_ne!(calls, matching());
            assert_ne!(moved.regions.stack, here.regions.stack);
            assert_eq!(compare(&calls, &moved), Ok(()));
            // Against the wrong anchors the same run fails at its first address.
            assert!(compare(&calls, &here).is_err());
        }

        /// The run starquake made of the fixture before it had these calls, as
        /// `tests/homebrew/README.md` recorded it then: `SetMemoryPermission` returned "not
        /// implemented" and libnx called `Break`.
        #[test]
        fn the_run_before_these_calls_existed_fails_at_its_first_call() {
            let not_implemented = KernelError::NotImplemented.code().raw();
            let old = [
                call(
                    SvcId::SetMemoryPermission,
                    &[(0, BASE + 0x4000), (1, 0x1000), (2, 1)],
                    not_implemented,
                ),
                call(SvcId::Break, &[(1, 0x80ff_ffdc), (2, 4)], not_implemented),
            ];
            assert_eq!(
                compare(&old, &anchors()),
                Err(
                    "call 0: expected `svc 0x02 SetMemoryPermission(address=main+0x4000, \
                 size=0x1000, perm=0x1) -> 0x0`, found `svc 0x02 \
                 SetMemoryPermission(address=main+0x4000, size=0x1000, perm=0x1) -> 0x4201`"
                        .to_owned()
                )
            );
        }

        #[test]
        fn a_wrong_argument_result_order_or_count_fails_at_that_call() {
            let anchors = anchors();
            let failure = |calls: &[Call]| compare(calls, &anchors).unwrap_err();

            // A different page of the module.
            let mut calls = matching();
            calls[1].args[0] += 0x1000;
            assert_eq!(
                failure(&calls),
                "call 1: expected `svc 0x03 SetMemoryAttribute(address=main+0x4000, size=0x1000, \
             mask=0x10, attr=0x10) -> 0x0`, found `svc 0x03 \
             SetMemoryAttribute(address=main+0x5000, size=0x1000, mask=0x10, attr=0x10) -> 0x0`"
            );
            // A different heap size.
            let mut calls = matching();
            calls[13].args[1] = 0xcd20_0000;
            assert_eq!(
                failure(&calls),
                "call 13: expected `svc 0x01 SetHeapSize(size=0xcd000000) -> 0x0`, found `svc 0x01 \
             SetHeapSize(size=0xcd200000) -> 0x0`"
            );
            // A result that is not the reference's.
            let mut calls = matching();
            calls[14].result = KernelError::InvalidPointer.code().raw();
            assert!(failure(&calls).starts_with("call 14: expected `svc 0x06 QueryMemory("));
            // A stack pointer eight bytes off.
            let mut calls = matching();
            calls[15].args[0] -= 8;
            assert!(
                failure(&calls)
                    .contains("found `svc 0x06 QueryMemory(out_memory_info=stack_region+0x103fc0")
            );
            // Two calls the wrong way round, a call left out, and a run that stops early.
            let mut calls = matching();
            calls.swap(3, 4);
            assert!(
                failure(&calls).starts_with("call 3: expected `svc 0x29 GetInfo(info_type=0x3,")
            );
            let mut calls = matching();
            calls.remove(0);
            assert!(failure(&calls).starts_with("call 0: expected `svc 0x02 SetMemoryPermission("));
            let calls = &matching()[..16];
            assert_eq!(
                failure(calls),
                "call 16: expected `svc 0x1f ConnectToNamedPort(name=main+0x3148) -> 0x0`, and the \
             run made no more calls"
            );
        }

        #[test]
        fn the_last_call_is_compared_by_its_arguments_only_and_nothing_after_it_at_all() {
            let anchors = anchors();
            // Its result is starquake's own until it has the call.
            let mut calls = matching();
            calls[16].result = KernelError::NotImplemented.code().raw();
            calls.push(call(SvcId::Break, &[(1, 0x1234)], 0));
            assert_eq!(compare(&calls, &anchors), Ok(()));
            // Another port name is a different call.
            calls[16].args[1] += 8;
            assert_eq!(
                compare(&calls, &anchors),
                Err(
                    "call 16: expected `svc 0x1f ConnectToNamedPort(name=main+0x3148)`, found \
                 `svc 0x1f ConnectToNamedPort(name=main+0x3150)`"
                        .to_owned()
                )
            );
        }

        #[test]
        fn a_call_or_an_address_the_comparison_does_not_know_is_refused() {
            let anchors = anchors();
            // A call with no address rule, in place of the first.
            let mut calls = matching();
            calls[0] = call(SvcId::MapMemory, &[], 0);
            assert_eq!(
                compare(&calls, &anchors),
                Err("call 0: no address rule for MapMemory".to_owned())
            );
            // A number that is no call at all.
            calls[0].number = 0x7777;
            assert_eq!(
                compare(&calls, &anchors),
                Err("call 0: svc 0x7777 is not a listed call".to_owned())
            );
            // An address below every region.
            let mut calls = matching();
            calls[0].args[0] = 0x1000;
            assert_eq!(
                compare(&calls, &anchors),
                Err("call 0: address 0x1000 lies in no known region".to_owned())
            );
        }
    }

    /// The fixture on a real vCPU.
    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    mod on_hardware {
        use starquake_core::{ExitReason, RunReport};
        use starquake_kernel::result::KernelError;
        use starquake_memory::PAGE_SIZE;
        use starquake_trace::{Dump, EventKind};

        use super::*;
        use crate::common::{GENEROUS, run};

        /// The system calls of a run, in order, from its event dump.
        fn calls(report: &RunReport) -> Vec<Call> {
            Dump::from_bytes(&report.events)
                .expect("a readable dump")
                .merged()
                .iter()
                .filter_map(|event| match event.kind {
                    EventKind::Svc { id, args, result } => Some(Call {
                        number: id,
                        args,
                        result: result.0,
                    }),
                    _ => None,
                })
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
        fn the_fixture_matches_the_reference_call_for_call_up_to_connect_to_named_port() {
            let Some(report) = run(HELLO, GENEROUS) else {
                return;
            };
            let ours = calls(&report);
            assert_eq!(
                compare(&ours, &anchors(&report)),
                Ok(()),
                "{}",
                report.trace
            );

            // Every call before `sm:` is implemented; the run then ends where the reference goes
            // on. libnx's `smInitialize` fails, the fixture aborts with `Break`, which is not
            // built either, and the `udf` the compiler put after it is a guest crash.
            let stubbed: Vec<_> = report.stubbed.iter().map(String::as_str).collect();
            assert_eq!(stubbed, ["Break (0x26)", "ConnectToNamedPort (0x1f)"]);
            let made: Vec<_> = report
                .system_calls
                .iter()
                .map(|(name, count)| (name.as_str(), *count))
                .collect();
            assert_eq!(
                made,
                [
                    ("Break", 1),
                    ("ConnectToNamedPort", 1),
                    ("GetInfo", 11),
                    ("QueryMemory", 2),
                    ("SetHeapSize", 1),
                    ("SetMemoryAttribute", 1),
                    ("SetMemoryPermission", 1),
                ]
            );
            assert_eq!(
                ours[16].result,
                KernelError::NotImplemented.code().raw(),
                "ConnectToNamedPort is built now: compare its result with the reference's, and \
             move the comparison's last call on"
            );
            assert!(
                matches!(report.exit, ExitReason::GuestCrashed(_)),
                "{:?}",
                report.exit
            );
            // The fixture has no counter reads to patch, so nothing but the kernel's mappings is
            // in its address space.
            assert_eq!(report.patched_counter_reads, Vec::<u64>::new());
            insta::assert_snapshot!("hello_fixture_trace", report.trace);
        }
    }
}
