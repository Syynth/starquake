//! The crash report: what a guest crash leaves behind, as data first (`crash.json`) and text
//! second (`crash.txt`).
//!
//! Source: `docs/specs/observability.md`, "Crash reports". This is the part of that list the
//! walking skeleton can fill in: the exception, the registers, the faulting instruction as module
//! plus offset, the memory map and the block the faulting address is in, the modules with their
//! build ids, the last events from the flight recorder, the emulator's version and the run's
//! command line.
//!
//! PROVISIONAL (walking skeleton, #74). Not here yet, and owned by the crash-report step of the
//! observability spec's order of work: a backtrace by frame pointer, symbol names, and the events
//! split per thread (there is one guest thread). The exception is one of three kinds: a data
//! abort, an instruction abort, and everything else as an undefined instruction. The CPU
//! backend's syndrome for it ([`CrashReport::with_syndrome`]) says the rest: whether an abort was
//! a missing page or a refused access, and what an "undefined instruction" really was.
//!
//! The JSON layout is this crate's own (the spec gives none), versioned by [`CRASH_VERSION`] and
//! pinned by a test. Addresses and register values are hex strings, so no reader loses bits to a
//! floating-point number. The `instruction_abort` kind and the `syndrome` field were each added
//! without a new version: nothing was renamed or removed, and every field that was there before
//! keeps its meaning. (Before the first, a fetch from an address nothing maps was reported as a
//! `data_abort` read.)

use std::fmt::Write as _;

use serde::{Serialize, Serializer};
use starquake_cpu::{ExceptionClass, FaultStatus, GuestRegs, Syndrome};

use crate::report::Mapping;

/// Version of the `crash.json` layout. Bump it when a field is renamed, removed or changes
/// meaning.
pub const CRASH_VERSION: u32 = 1;

/// Serialises a `u64` as a hex string (`"0x80000000"`).
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `serialize_with` calls this with a reference to the field"
)]
pub(crate) fn hex<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{value:#x}"))
}

/// Serialises `u64`s as a list of hex strings.
fn hex_list<S: Serializer>(values: &[u64], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter().map(|value| format!("{value:#x}")))
}

/// Serialises bytes as one lowercase hex string with no prefix (a build id).
fn hex_bytes<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&HexBytes(bytes))
}

struct HexBytes<'a>(&'a [u8]);

impl std::fmt::Display for HexBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// The exception that ended the guest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Exception {
    /// A load or store that nothing maps, or that the page's permissions refuse.
    DataAbort {
        /// The address the guest tried to access.
        #[serde(serialize_with = "hex")]
        address: u64,
        /// Whether the access was a write.
        write: bool,
    },
    /// An instruction fetch from an address that nothing maps, or that is not executable.
    InstructionAbort {
        /// The address the guest tried to execute.
        #[serde(serialize_with = "hex")]
        address: u64,
    },
    /// An instruction the guest may not execute: undefined, a breakpoint, a failed pointer
    /// authentication, a misaligned PC or SP.
    UndefinedInstruction,
}

/// The exception's syndrome, as the CPU backend reported it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SyndromeReport {
    /// The syndrome in the architecture's layout (`ESR_EL1`).
    #[serde(serialize_with = "hex")]
    pub esr: u64,
    /// What it says, in words: `data abort, translation fault at level 3`.
    pub text: String,
}

impl SyndromeReport {
    /// The report of `syndrome`.
    #[must_use]
    pub fn new(syndrome: Syndrome) -> Self {
        let class = class_text(syndrome.class());
        let text = match (syndrome.fault(), syndrome.immediate()) {
            (Some(fault), _) => format!("{class}, {}", fault_text(fault)),
            (None, Some(immediate)) => format!("{class} #{immediate:#x}"),
            (None, None) => class,
        };
        Self {
            esr: syndrome.esr(),
            text,
        }
    }
}

/// An exception class in words.
fn class_text(class: ExceptionClass) -> String {
    match class {
        ExceptionClass::Unknown => "undefined instruction".to_owned(),
        ExceptionClass::TrappedWait => "trapped wfi or wfe".to_owned(),
        ExceptionClass::Svc => "svc".to_owned(),
        ExceptionClass::TrappedSystemAccess => {
            "system register or instruction not allowed at EL0".to_owned()
        }
        ExceptionClass::PointerAuthentication => "pointer authentication failure".to_owned(),
        ExceptionClass::InstructionAbort => "instruction abort".to_owned(),
        ExceptionClass::PcAlignment => "misaligned pc".to_owned(),
        ExceptionClass::DataAbort => "data abort".to_owned(),
        ExceptionClass::SpAlignment => "misaligned stack pointer".to_owned(),
        ExceptionClass::Breakpoint => "brk".to_owned(),
        ExceptionClass::Other(ec) => format!("exception class {ec:#04x}"),
    }
}

/// Why an access aborted, in words.
fn fault_text(fault: FaultStatus) -> String {
    match fault {
        FaultStatus::AddressSize { level } => format!("address size fault at level {level}"),
        FaultStatus::Translation { level } => format!("translation fault at level {level}"),
        FaultStatus::AccessFlag { level } => format!("access flag fault at level {level}"),
        FaultStatus::Permission { level } => format!("permission fault at level {level}"),
        FaultStatus::Alignment => "alignment fault".to_owned(),
        FaultStatus::Other(code) => format!("fault status {code:#04x}"),
    }
}

/// The guest's registers at the exception.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Registers {
    /// `x0`-`x30`.
    #[serde(serialize_with = "hex_list")]
    pub x: Vec<u64>,
    /// `SP_EL0`.
    #[serde(serialize_with = "hex")]
    pub sp: u64,
    /// The program counter: the address of the instruction that took the exception.
    #[serde(serialize_with = "hex")]
    pub pc: u64,
    /// The flags (NZCV).
    #[serde(serialize_with = "hex")]
    pub pstate: u64,
    /// `TPIDR_EL0`.
    #[serde(serialize_with = "hex")]
    pub tpidr: u64,
    /// `TPIDRRO_EL0`.
    #[serde(serialize_with = "hex")]
    pub tpidrro: u64,
}

/// A loaded module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Module {
    /// The module's name.
    pub name: String,
    /// Guest address of its first byte.
    #[serde(serialize_with = "hex")]
    pub base: u64,
    /// Its size in memory, `.bss` included.
    #[serde(serialize_with = "hex")]
    pub size: u64,
    /// Its build id.
    #[serde(serialize_with = "hex_bytes")]
    pub build_id: Vec<u8>,
}

/// An address as a module plus an offset into it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Location {
    /// The module's name.
    pub module: String,
    /// Offset from the module's base.
    #[serde(serialize_with = "hex")]
    pub offset: u64,
}

/// Everything recorded about a guest crash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CrashReport {
    /// [`CRASH_VERSION`].
    pub version: u32,
    /// The emulator and its version.
    pub emulator: String,
    /// The command line that started the run; empty when there was none.
    pub command_line: Vec<String>,
    /// What happened.
    pub exception: Exception,
    /// The exception's syndrome; `null` when the report was built without one.
    pub syndrome: Option<SyndromeReport>,
    /// The address of the instruction that took the exception.
    #[serde(serialize_with = "hex")]
    pub pc: u64,
    /// That address as module plus offset, when it is inside a module.
    pub pc_location: Option<Location>,
    /// For a data abort or an instruction abort, the name of the mapping the faulting address
    /// is in; `null` when the address is not mapped, and for other exceptions.
    pub fault_mapping: Option<String>,
    /// The registers at the exception.
    pub registers: Registers,
    /// The guest memory the run set up, in address order.
    pub memory_map: Vec<Mapping>,
    /// The loaded modules.
    pub modules: Vec<Module>,
    /// The newest events from the flight recorder, oldest first, one line each.
    pub last_events: Vec<String>,
}

impl CrashReport {
    /// Builds the report for `exception`, taken with the guest's registers as `regs`.
    #[must_use]
    pub fn new(
        exception: Exception,
        regs: &GuestRegs,
        memory_map: &[Mapping],
        modules: &[Module],
        last_events: Vec<String>,
        command_line: &[String],
    ) -> Self {
        let pc_location = modules.iter().find_map(|module| {
            let offset = regs.pc.checked_sub(module.base)?;
            (offset < module.size).then(|| Location {
                module: module.name.clone(),
                offset,
            })
        });
        let fault_mapping = match exception {
            Exception::DataAbort { address, .. } | Exception::InstructionAbort { address } => {
                memory_map
                    .iter()
                    .find(|mapping| mapping.contains(address))
                    .map(|mapping| mapping.name.clone())
            }
            Exception::UndefinedInstruction => None,
        };
        Self {
            version: CRASH_VERSION,
            emulator: format!("starquake {}", env!("CARGO_PKG_VERSION")),
            command_line: command_line.to_vec(),
            exception,
            syndrome: None,
            pc: regs.pc,
            pc_location,
            fault_mapping,
            registers: Registers {
                x: regs.x.to_vec(),
                sp: regs.sp,
                pc: regs.pc,
                pstate: regs.pstate,
                tpidr: regs.tpidr,
                tpidrro: regs.tpidrro,
            },
            memory_map: memory_map.to_vec(),
            modules: modules.to_vec(),
            last_events,
        }
    }

    /// The report with `syndrome`, the one the CPU backend reported for the exception.
    #[must_use]
    pub fn with_syndrome(mut self, syndrome: Syndrome) -> Self {
        self.syndrome = Some(SyndromeReport::new(syndrome));
        self
    }

    /// The report as text, for `crash.txt`.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        // Writing to a `String` cannot fail, so the results are dropped.
        let _ = self.write_text(&mut text);
        text
    }

    /// What happened, in a few words: `data abort, a write at address 0xdea8 (not mapped)`.
    #[must_use]
    pub fn exception_text(&self) -> String {
        let block = || {
            self.fault_mapping
                .as_deref()
                .map_or_else(|| "not mapped".to_owned(), |name| format!("in {name}"))
        };
        match self.exception {
            Exception::DataAbort { address, write } => {
                let access = if write { "write" } else { "read" };
                format!(
                    "data abort, a {access} at address {address:#x} ({})",
                    block()
                )
            }
            Exception::InstructionAbort { address } => {
                format!(
                    "instruction abort, a fetch at address {address:#x} ({})",
                    block()
                )
            }
            Exception::UndefinedInstruction => "undefined instruction".to_owned(),
        }
    }

    /// Where it happened: `pc 0x800000b8 (main+0xb8)`.
    #[must_use]
    pub fn instruction_text(&self) -> String {
        match &self.pc_location {
            Some(Location { module, offset }) => {
                format!("pc {:#x} ({module}+{offset:#x})", self.pc)
            }
            None => format!("pc {:#x} (in no module)", self.pc),
        }
    }

    fn write_text(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "{} crash report", self.emulator)?;
        if !self.command_line.is_empty() {
            writeln!(out, "command line: {}", self.command_line.join(" "))?;
        }
        writeln!(out)?;
        writeln!(out, "exception: {}", self.exception_text())?;
        if let Some(SyndromeReport { esr, text }) = &self.syndrome {
            writeln!(out, "syndrome: {esr:#010x} ({text})")?;
        }
        writeln!(out, "instruction: {}", self.instruction_text())?;
        writeln!(out, "\nregisters:")?;
        for (row, values) in self.registers.x.chunks(4).enumerate() {
            write!(out, " ")?;
            for (column, value) in values.iter().enumerate() {
                write!(out, " x{:<2} {value:#018x}", row * 4 + column)?;
            }
            writeln!(out)?;
        }
        let Registers {
            sp,
            pc,
            pstate,
            tpidr,
            tpidrro,
            x: _,
        } = &self.registers;
        writeln!(out, "  sp  {sp:#018x} pc  {pc:#018x} pstate {pstate:#x}")?;
        writeln!(out, "  tpidr {tpidr:#018x} tpidrro {tpidrro:#018x}")?;
        writeln!(out, "\nmemory map:")?;
        for Mapping { name, start, end } in &self.memory_map {
            writeln!(out, "  {start:#x}-{end:#x} {name}")?;
        }
        writeln!(out, "\nmodules:")?;
        for module in &self.modules {
            writeln!(
                out,
                "  {} base {:#x} size {:#x} build id {}",
                module.name,
                module.base,
                module.size,
                HexBytes(&module.build_id)
            )?;
        }
        writeln!(out, "\nlast events, oldest first:")?;
        for event in &self.last_events {
            writeln!(out, "  {event}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registers that all differ, so a swapped pair shows.
    fn regs() -> GuestRegs {
        let mut regs = GuestRegs {
            sp: 0x80ff_fff0,
            pc: 0x8000_00b8,
            pstate: 0x6000_0000,
            tpidr: 0x1111,
            tpidrro: 0x8000_3000,
            ..GuestRegs::default()
        };
        for (index, x) in (0u64..).zip(regs.x.iter_mut()) {
            *x = 0x100 + index;
        }
        regs
    }

    fn memory_map() -> Vec<Mapping> {
        vec![
            Mapping {
                name: "main.text".into(),
                start: 0x8000_0000,
                end: 0x8000_1000,
            },
            Mapping {
                name: "stack".into(),
                start: 0x80f0_0000,
                end: 0x8100_0000,
            },
        ]
    }

    fn modules() -> Vec<Module> {
        vec![Module {
            name: "main".into(),
            base: 0x8000_0000,
            size: 0x1000,
            build_id: vec![0xab, 0x01, 0xff],
        }]
    }

    fn report(exception: Exception) -> CrashReport {
        CrashReport::new(
            exception,
            &regs(),
            &memory_map(),
            &modules(),
            vec!["#0 core 0 thread 1: lifecycle ProcessStart 1".into()],
            &["starquake".into(), "run".into(), "crash.nro".into()],
        )
    }

    #[test]
    fn the_report_names_the_faulting_address_and_the_instruction() {
        let report = report(Exception::DataAbort {
            address: 0xdea8,
            write: true,
        });
        assert_eq!(report.pc, 0x8000_00b8);
        assert_eq!(
            report.pc_location,
            Some(Location {
                module: "main".into(),
                offset: 0xb8
            })
        );
        assert_eq!(report.fault_mapping, None);
        let text = report.to_text();
        assert!(
            text.contains("exception: data abort, a write at address 0xdea8 (not mapped)"),
            "{text}"
        );
        assert!(
            text.contains("instruction: pc 0x800000b8 (main+0xb8)"),
            "{text}"
        );
        assert!(text.contains("command line: starquake run crash.nro"));
        assert!(text.contains("  0x80f00000-0x81000000 stack"));
        assert!(text.contains("  main base 0x80000000 size 0x1000 build id ab01ff"));
        assert!(text.contains("  #0 core 0 thread 1: lifecycle ProcessStart 1"));
        // Every register appears, each with its own value.
        for index in 0..31u64 {
            let value = format!("x{index:<2} {:#018x}", 0x100 + index);
            assert!(text.contains(&value), "{value} missing from\n{text}");
        }
        assert!(text.contains("sp  0x0000000080fffff0 pc  0x00000000800000b8 pstate 0x60000000"));
        assert!(text.contains("tpidr 0x0000000000001111 tpidrro 0x0000000080003000"));
    }

    #[test]
    fn a_fault_inside_a_mapping_names_it_and_a_read_is_called_a_read() {
        let report = report(Exception::DataAbort {
            address: 0x80f0_0010,
            write: false,
        });
        assert_eq!(report.fault_mapping.as_deref(), Some("stack"));
        assert!(
            report
                .to_text()
                .contains("exception: data abort, a read at address 0x80f00010 (in stack)")
        );
    }

    #[test]
    fn an_instruction_abort_names_the_address_fetched_from_and_its_mapping() {
        // A jump into the stack: mapped, not executable.
        let report = report(Exception::InstructionAbort {
            address: 0x80f0_0010,
        });
        assert_eq!(report.fault_mapping.as_deref(), Some("stack"));
        assert!(
            report
                .to_text()
                .contains("exception: instruction abort, a fetch at address 0x80f00010 (in stack)")
        );
        // A jump to an address nothing maps.
        let report = super::tests::report(Exception::InstructionAbort { address: 0xdea8 });
        assert_eq!(report.fault_mapping, None);
        assert!(
            report
                .to_text()
                .contains("exception: instruction abort, a fetch at address 0xdea8 (not mapped)")
        );
    }

    #[test]
    fn a_pc_outside_every_module_has_no_location() {
        let mut regs = regs();
        // One past the module's last byte, and one below its first.
        for pc in [0x8000_1000, 0x7fff_fffc] {
            regs.pc = pc;
            let report = CrashReport::new(
                Exception::UndefinedInstruction,
                &regs,
                &memory_map(),
                &modules(),
                Vec::new(),
                &[],
            );
            assert_eq!(report.pc_location, None, "{pc:#x}");
            let text = report.to_text();
            assert!(text.contains("exception: undefined instruction"));
            assert!(text.contains(&format!("instruction: pc {pc:#x} (in no module)")));
            assert!(!text.contains("command line"));
        }
        // The module's last byte is inside it.
        regs.pc = 0x8000_0fff;
        let last = CrashReport::new(
            Exception::UndefinedInstruction,
            &regs,
            &[],
            &modules(),
            Vec::new(),
            &[],
        );
        assert_eq!(
            last.pc_location.map(|location| location.offset),
            Some(0xfff)
        );
    }

    /// `ESR_EL1` with exception class `ec` and instruction-specific syndrome `iss`, `IL` set (Arm
    /// ARM, `ESR_EL1`: `EC` bits 31:26, `IL` bit 25).
    fn esr(ec: u64, iss: u64) -> Syndrome {
        Syndrome::from_esr((ec << 26) | (1 << 25) | iss)
    }

    #[test]
    fn every_class_and_fault_status_is_said_in_words() {
        let cases = [
            (Syndrome::undefined(), "undefined instruction"),
            (esr(0x01, 0x1e0_0000), "trapped wfi or wfe"),
            (esr(0x15, 0x27), "svc #0x27"),
            (
                esr(0x18, 0x32_0400),
                "system register or instruction not allowed at EL0",
            ),
            (esr(0x1c, 0), "pointer authentication failure"),
            (esr(0x22, 0), "misaligned pc"),
            (esr(0x26, 0), "misaligned stack pointer"),
            (esr(0x3c, 0x123), "brk #0x123"),
            (esr(0x16, 2), "exception class 0x16"),
            (
                Syndrome::data_abort(FaultStatus::AddressSize { level: 1 }, false),
                "data abort, address size fault at level 1",
            ),
            (
                Syndrome::data_abort(FaultStatus::Translation { level: 2 }, true),
                "data abort, translation fault at level 2",
            ),
            (
                Syndrome::data_abort(FaultStatus::AccessFlag { level: 3 }, false),
                "data abort, access flag fault at level 3",
            ),
            (
                Syndrome::instruction_abort(FaultStatus::Permission { level: 3 }),
                "instruction abort, permission fault at level 3",
            ),
            (
                Syndrome::data_abort(FaultStatus::Alignment, true),
                "data abort, alignment fault",
            ),
            (
                Syndrome::instruction_abort(FaultStatus::Other(0x30)),
                "instruction abort, fault status 0x30",
            ),
        ];
        for (syndrome, text) in cases {
            let report = SyndromeReport::new(syndrome);
            assert_eq!(report.text, text, "{:#x}", syndrome.esr());
            assert_eq!(report.esr, syndrome.esr());
        }
    }

    #[test]
    fn the_text_says_the_syndrome_after_the_exception_when_there_is_one() {
        let plain = report(Exception::UndefinedInstruction);
        assert_eq!(plain.syndrome, None);
        assert!(!plain.to_text().contains("syndrome"));
        let brk = plain.with_syndrome(esr(0x3c, 0x123));
        assert!(
            brk.to_text().contains(
                "exception: undefined instruction\nsyndrome: 0xf2000123 (brk #0x123)\n\
                 instruction: pc 0x800000b8 (main+0xb8)\n"
            ),
            "{}",
            brk.to_text()
        );
    }

    #[test]
    fn the_json_layout_is_pinned() {
        let mut report = report(Exception::DataAbort {
            address: 0x80f0_0010,
            write: true,
        })
        .with_syndrome(Syndrome::data_abort(
            FaultStatus::Permission { level: 3 },
            true,
        ));
        report.emulator = "starquake 9.9.9".into();
        report.registers.x.truncate(2);
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            json,
            r##"{
  "version": 1,
  "emulator": "starquake 9.9.9",
  "command_line": [
    "starquake",
    "run",
    "crash.nro"
  ],
  "exception": {
    "kind": "data_abort",
    "address": "0x80f00010",
    "write": true
  },
  "syndrome": {
    "esr": "0x9200004f",
    "text": "data abort, permission fault at level 3"
  },
  "pc": "0x800000b8",
  "pc_location": {
    "module": "main",
    "offset": "0xb8"
  },
  "fault_mapping": "stack",
  "registers": {
    "x": [
      "0x100",
      "0x101"
    ],
    "sp": "0x80fffff0",
    "pc": "0x800000b8",
    "pstate": "0x60000000",
    "tpidr": "0x1111",
    "tpidrro": "0x80003000"
  },
  "memory_map": [
    {
      "name": "main.text",
      "start": "0x80000000",
      "end": "0x80001000"
    },
    {
      "name": "stack",
      "start": "0x80f00000",
      "end": "0x81000000"
    }
  ],
  "modules": [
    {
      "name": "main",
      "base": "0x80000000",
      "size": "0x1000",
      "build_id": "ab01ff"
    }
  ],
  "last_events": [
    "#0 core 0 thread 1: lifecycle ProcessStart 1"
  ]
}"##
        );
        let undefined = serde_json::to_value(Exception::UndefinedInstruction).unwrap();
        assert_eq!(
            undefined,
            serde_json::json!({ "kind": "undefined_instruction" })
        );
        let fetch = serde_json::to_value(Exception::InstructionAbort {
            address: 0x80f0_0010,
        })
        .unwrap();
        assert_eq!(
            fetch,
            serde_json::json!({ "kind": "instruction_abort", "address": "0x80f00010" })
        );
        assert_eq!(report.version, CRASH_VERSION);
        // A report built without a syndrome says so.
        let without =
            serde_json::to_value(super::tests::report(Exception::UndefinedInstruction)).unwrap();
        assert_eq!(without["syndrome"], serde_json::Value::Null);
    }
}
