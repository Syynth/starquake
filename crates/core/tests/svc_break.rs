//! `svc Break` from synthetic programs run natively under the hypervisor backend: a break that
//! is not a notification ends the run as a guest crash whose report names the reason, and a
//! notification-only break returns to the program with its registers as they were.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. The same paths are tested on every host against a scripted vCPU in
//! `src/machine/tests.rs` (`a_break_ends_the_run_as_a_crash_naming_its_reason_and_a_notification_returns`).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use starquake_core::crash::{Exception, Location};
use starquake_core::exit_code;
use starquake_kernel::svc::SvcId;
use starquake_kernel::svc::exception::{BreakReason, NOTIFICATION_ONLY_FLAG, RESULT_ARGUMENT_SIZE};
use starquake_testkit::a64;
use starquake_testkit::nro::{CODE_OFFSET, NroBuilder};

#[cfg(test)]
mod common;

use common::{GENEROUS, run};

/// Shared by the tests. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// Register 31 as a base or source register of the instructions used here: `SP`.
    pub(crate) const SP: u8 = 31;

    /// The word the crashing program leaves in its buffer: every byte different.
    pub(crate) const CODE: u32 = 0x1234_5678;

    /// The `svc` instruction for `id`.
    pub(crate) fn svc(id: SvcId) -> u32 {
        a64::svc(u32::from(id.number())).unwrap()
    }

    /// The 16-bit halves of a 32-bit value, low first.
    pub(crate) fn halves(value: u32) -> [u32; 2] {
        [value & a64::MAX_IMM16, value >> 16]
    }

    /// The offset in the module of instruction `index`.
    pub(crate) fn offset_of(index: usize) -> u64 {
        u64::try_from(CODE_OFFSET).unwrap()
            + u64::try_from(index).unwrap() * u64::from(a64::INSN_SIZE)
    }
}

use support::{CODE, SP, halves, offset_of, svc};

#[test]
fn an_assertion_break_ends_the_run_as_a_crash_naming_its_reason_and_its_code() {
    // The program writes a result-code word below its stack pointer and breaks with an
    // assertion, the buffer, and its size:
    //
    //   sub  x1, sp, #16
    //   movz x9, #0x5678
    //   movk x9, #0x1234, lsl #16
    //   str  x9, [x1]
    //   movz x0, #1          Assert
    //   movz x2, #4
    //   svc  #0x26           Break
    //   brk  #0              never reached
    let [low, high] = halves(CODE);
    let size = u32::try_from(RESULT_ARGUMENT_SIZE).unwrap();
    let words = [
        a64::sub_imm(1, SP, 16).unwrap(),
        a64::movz(9, low, 0).unwrap(),
        a64::movk(9, high, 16).unwrap(),
        a64::str(9, 1, 0).unwrap(),
        a64::movz(0, BreakReason::Assert.raw(), 0).unwrap(),
        a64::movz(2, size, 0).unwrap(),
        svc(SvcId::Break),
        a64::brk(0).unwrap(),
    ];
    let svc_index = 6;
    let file = NroBuilder::new(&words).build().unwrap();
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("a crash report");
    let buffer = crash.registers.x[1];
    assert_eq!(
        crash.exception,
        Exception::SvcBreak {
            reason: BreakReason::Assert.raw(),
            reason_name: Some("assert"),
            address: buffer,
            size: RESULT_ARGUMENT_SIZE,
            code: Some(CODE),
        },
        "{}",
        crash.to_text()
    );
    assert_eq!(buffer, crash.registers.sp - 16);
    assert_eq!(crash.syndrome, None);
    // The program counter is the one the backend reports for a system call: the instruction
    // after the `svc`.
    assert_eq!(
        crash.pc_location,
        Some(Location {
            module: "main".into(),
            offset: offset_of(svc_index + 1)
        }),
        "{}",
        crash.to_text()
    );
    let text = crash.to_text();
    assert!(
        text.contains(&format!(
            "exception: svc Break, reason assert (0x1), buffer {buffer:#x} size 0x4, code {CODE:#x}\n"
        )),
        "{text}"
    );
    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [(&SvcId::Break.name().to_owned(), &1)]
    );
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);
}

#[test]
fn a_notification_only_break_returns_and_leaves_the_registers_as_they_were() {
    // A module notice with the notification flag, then a `brk` that stops the run where the
    // crash report shows the registers the break left:
    //
    //   movz x0, #0x8000, lsl #16
    //   movk x0, #3          PreLoadDll | NotificationOnlyFlag
    //   movz x1, #0x1000
    //   movz x2, #0x2000
    //   svc  #0x26           Break: returns
    //   brk  #0
    let raw = BreakReason::PreLoadDll.raw() | NOTIFICATION_ONLY_FLAG;
    let [low, high] = halves(raw);
    let words = [
        a64::movz(0, high, 16).unwrap(),
        a64::movk(0, low, 0).unwrap(),
        a64::movz(1, 0x1000, 0).unwrap(),
        a64::movz(2, 0x2000, 0).unwrap(),
        svc(SvcId::Break),
        a64::brk(0).unwrap(),
    ];
    let brk_index = 5;
    let file = NroBuilder::new(&words).build().unwrap();
    let Some(report) = run(&file, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit_code(), exit_code::GUEST_CRASHED);
    let crash = report.crash().expect("the program stops at its brk");
    assert_eq!(
        crash.exception,
        Exception::UndefinedInstruction,
        "{}",
        crash.to_text()
    );
    assert_eq!(
        crash.pc_location.as_ref().map(|location| location.offset),
        Some(offset_of(brk_index))
    );
    // `Break` writes no register: the reason word and the buffer are where the program put
    // them.
    assert_eq!(
        crash.registers.x[..3],
        [u64::from(raw), 0x1000, 0x2000],
        "{}",
        crash.to_text()
    );
    assert_eq!(
        report.system_calls.iter().collect::<Vec<_>>(),
        [(&SvcId::Break.name().to_owned(), &1)]
    );
    assert!(report.stubbed.is_empty(), "{:?}", report.stubbed);
}
