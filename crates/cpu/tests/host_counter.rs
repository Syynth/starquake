//! The counter a guest reads under the hypervisor backend is the one `host_counter::read` reads
//! on the host (`mach_absolute_time`): same zero, same rate. `starquake-core` relies on this to
//! give guest time one zero point (see `starquake_cpu::host_counter`).
//!
//! Needs a Mac with Apple Silicon. Skips itself only when the `CI` environment variable is set
//! (hosted runners can't create VMs); anywhere else, failing to create a VM fails the test.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use starquake_cpu::hvf::{HvfBackend, RamConfig};
use starquake_cpu::{Backend, Exit, GuestRegs, Vcpu, host_counter};
use starquake_memory::GuestMemory;

/// `mrs x0, cntvct_el0`.
const MRS_X0_CNTVCT: u32 = 0xd53b_e040;
/// `svc #0`.
const SVC_0: u32 = 0xd400_0001;

#[test]
fn host_readings_do_not_go_backwards() {
    let first = host_counter::read();
    let second = host_counter::read();
    assert!(first > 0, "the host counter read as 0");
    assert!(first <= second, "{first:#x} then {second:#x}");
}

#[test]
fn a_guest_read_lies_between_the_host_reads_around_it() {
    if std::env::var_os("CI").is_some() {
        return;
    }
    let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
    let ram = backend.memory();
    let code: Vec<u8> = [MRS_X0_CNTVCT, SVC_0]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    ram.write(ram.base(), &code).expect("load code");
    let mut vcpu = backend.create_vcpu().expect("create a vCPU");
    vcpu.set_regs(&GuestRegs {
        pc: ram.base(),
        ..GuestRegs::default()
    })
    .expect("set registers");

    let before = host_counter::read();
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    let after = host_counter::read();
    let guest = vcpu.regs().unwrap().x[0];

    // One counter, so the guest's reading is bracketed by the host's. With any offset between
    // the two views, or a different rate, it would fall outside: the bracket is only as wide as
    // one entry to the guest and back.
    assert!(
        before <= guest && guest <= after,
        "host {before:#x}, guest {guest:#x}, host {after:#x}"
    );
    // The bracket is narrow next to the counter's age (the host has been up far longer than the
    // run took), so passing it is not an accident of a wide window: under a second of ticks.
    assert!(
        after - before < backend.counter_frequency(),
        "the run took {} ticks",
        after - before
    );
}
