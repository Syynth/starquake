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
//! split per thread (there is one guest thread). The exception is only as detailed as the CPU
//! backend's exits: an instruction abort arrives as a data-abort read, and every other exception
//! as an undefined instruction, until the richer exit information of issue #36.
//!
//! The JSON layout is this crate's own (the spec gives none), versioned by [`CRASH_VERSION`] and
//! pinned by a test. Addresses and register values are hex strings, so no reader loses bits to a
//! floating-point number.

use std::fmt::Write as _;

use serde::{Serialize, Serializer};
use starquake_cpu::GuestRegs;

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
    /// A memory access that nothing maps (or, until issue #36, an instruction fetch that
    /// nothing maps, reported as a read).
    DataAbort {
        /// The address the guest tried to access.
        #[serde(serialize_with = "hex")]
        address: u64,
        /// Whether the access was a write.
        write: bool,
    },
    /// An instruction the guest may not execute: undefined, a breakpoint, a failed pointer
    /// authentication, a misaligned PC or SP.
    UndefinedInstruction,
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
    /// The address of the instruction that took the exception.
    #[serde(serialize_with = "hex")]
    pub pc: u64,
    /// That address as module plus offset, when it is inside a module.
    pub pc_location: Option<Location>,
    /// For a data abort, the name of the mapping the faulting address is in; `null` when the
    /// address is not mapped, and for other exceptions.
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
            Exception::DataAbort { address, .. } => memory_map
                .iter()
                .find(|mapping| mapping.contains(address))
                .map(|mapping| mapping.name.clone()),
            Exception::UndefinedInstruction => None,
        };
        Self {
            version: CRASH_VERSION,
            emulator: format!("starquake {}", env!("CARGO_PKG_VERSION")),
            command_line: command_line.to_vec(),
            exception,
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

    /// The report as text, for `crash.txt`.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        // Writing to a `String` cannot fail, so the results are dropped.
        let _ = self.write_text(&mut text);
        text
    }

    fn write_text(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "{} crash report", self.emulator)?;
        if !self.command_line.is_empty() {
            writeln!(out, "command line: {}", self.command_line.join(" "))?;
        }
        writeln!(out)?;
        match self.exception {
            Exception::DataAbort { address, write } => {
                let access = if write { "write" } else { "read" };
                let block = self
                    .fault_mapping
                    .as_deref()
                    .map_or_else(|| "not mapped".to_owned(), |name| format!("in {name}"));
                writeln!(
                    out,
                    "exception: data abort, a {access} at address {address:#x} ({block})"
                )?;
            }
            Exception::UndefinedInstruction => {
                writeln!(out, "exception: undefined instruction")?;
            }
        }
        match &self.pc_location {
            Some(Location { module, offset }) => {
                writeln!(out, "instruction: pc {:#x} ({module}+{offset:#x})", self.pc)?;
            }
            None => writeln!(out, "instruction: pc {:#x} (in no module)", self.pc)?,
        }
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

    #[test]
    fn the_json_layout_is_pinned() {
        let mut report = report(Exception::DataAbort {
            address: 0x80f0_0010,
            write: true,
        });
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
        assert_eq!(report.version, CRASH_VERSION);
    }
}
