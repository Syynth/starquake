//! Hypervisor.framework backend: guest programs run at EL0 in a real VM.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. Guest programs are raw instruction words with their assembly alongside.
//!
//! Hypervisor.framework allows one VM per process, so the tests take a lock before creating one
//! (`cargo nextest` runs each test in its own process anyway).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_cpu::hvf::{HvfBackend, HvfVcpu, PacKeys, RamConfig};
use starquake_cpu::{Backend, Exit, GuestRegs, Interrupter, Vcpu};
use starquake_memory::GuestMemory;

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// One VM per process: tests that create a backend hold this.
    pub(crate) static VM_LOCK: Mutex<()> = Mutex::new(());

    /// Where test data lives, relative to the start of guest RAM (code goes at the start).
    pub(crate) const DATA_OFFSET: u64 = 0x10_0000;

    /// `svc #0`, the end of most test programs.
    pub(crate) const SVC_0: u32 = 0xd400_0001;

    pub(crate) struct Guest {
        pub(crate) backend: Arc<HvfBackend>,
        _lock: MutexGuard<'static, ()>,
    }

    impl Guest {
        /// A fresh VM with the default bring-up RAM, or `None` on hosted CI.
        pub(crate) fn new() -> Option<Self> {
            if std::env::var_os("CI").is_some() {
                return None;
            }
            let lock = VM_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
            let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
            Some(Self {
                backend: Arc::new(backend),
                _lock: lock,
            })
        }

        pub(crate) fn ram_base(&self) -> u64 {
            self.backend.memory().base()
        }

        pub(crate) fn data(&self) -> u64 {
            self.ram_base() + DATA_OFFSET
        }

        /// Copies `words` to the start of guest RAM and returns their address.
        pub(crate) fn load(&self, words: &[u32]) -> u64 {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let base = self.ram_base();
            self.backend
                .memory()
                .write(base, &bytes)
                .expect("load code");
            base
        }

        /// A vCPU on this thread, about to run `words` with `setup` applied to its registers.
        pub(crate) fn vcpu(&self, words: &[u32], setup: impl FnOnce(&mut GuestRegs)) -> HvfVcpu {
            let pc = self.load(words);
            let mut vcpu = self.backend.create_vcpu().expect("create a vCPU");
            let ram = self.backend.memory();
            let mut regs = GuestRegs {
                pc,
                sp: ram.base() + u64::try_from(ram.size()).expect("RAM size fits u64"),
                ..GuestRegs::default()
            };
            setup(&mut regs);
            vcpu.set_regs(&regs).expect("set registers");
            vcpu
        }
    }

    pub(crate) fn read_u64(memory: &impl GuestMemory, va: u64) -> u64 {
        let mut buf = [0; 8];
        memory.read(va, &mut buf).expect("read guest memory");
        u64::from_le_bytes(buf)
    }

    /// Runs the PAuth program on a fresh vCPU and returns x0-x7.
    pub(crate) fn pac_results(
        backend: &HvfBackend,
        keys: &PacKeys,
        pointer: u64,
        modifier: u64,
    ) -> [u64; 8] {
        let mut vcpu = backend.create_vcpu().unwrap();
        vcpu.set_pac_keys(keys).unwrap();
        let mut regs = GuestRegs {
            pc: backend.memory().base(),
            ..GuestRegs::default()
        };
        for i in [0, 3, 4, 5, 6] {
            regs.x[i] = pointer;
        }
        regs.x[1] = modifier;
        vcpu.set_regs(&regs).unwrap();
        assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
        let x = vcpu.regs().unwrap().x;
        [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]
    }
}

use support::{Guest, SVC_0, pac_results, read_u64};

#[test]
fn svc_exits_with_its_immediate_and_pc_after_it() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd280_0540, // mov x0, #42
            0xd400_04e1, // svc #0x27
        ],
        |_| {},
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0x27));
    let regs = vcpu.regs().unwrap();
    assert_eq!(regs.x[0], 42);
    assert_eq!(regs.pc, guest.ram_base() + 8);
}

#[test]
fn set_regs_before_resuming_is_seen_by_the_guest() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd280_0540, // mov x0, #42
            0xd400_04e1, // svc #0x27
            0x9100_0400, // add x0, x0, #1
            0xd400_0501, // svc #0x28
        ],
        |_| {},
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0x27));
    let mut regs = vcpu.regs().unwrap();
    regs.x[0] = 100;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0x28));
    assert_eq!(vcpu.regs().unwrap().x[0], 101);
}

#[test]
fn registers_round_trip_and_survive_an_exit() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    let mut regs = vcpu.regs().unwrap();
    for (i, x) in regs.x.iter_mut().enumerate() {
        *x = 0x1111_0000_0000_0000 | u64::try_from(i).unwrap();
    }
    regs.sp = guest.data();
    regs.pstate = 0x6000_0000; // Z and C
    regs.tpidr = 0x1234;
    regs.tpidrro = 0x5678;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(vcpu.regs().unwrap(), regs);
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    // The shim borrows x9-x11 while it handles the exception and must give them back.
    let after = vcpu.regs().unwrap();
    assert_eq!(after.x, regs.x);
    assert_eq!(after.sp, regs.sp);
    assert_eq!(after.pstate, regs.pstate);
    assert_eq!(after.pc, regs.pc + 4);
}

#[test]
fn pstate_keeps_only_the_condition_flags() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |regs| regs.pstate = 0xf000_03c5);
    assert_eq!(vcpu.regs().unwrap().pstate, 0xf000_0000);
    // Even with EL1h requested, the guest still runs at EL0: its svc reaches the host.
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
}

#[test]
fn loads_and_stores_to_unmapped_addresses_are_data_aborts() {
    let Some(guest) = Guest::new() else { return };
    let ram = guest.backend.memory();
    let unmapped = ram.base() + u64::try_from(ram.size()).unwrap() + 0x10;
    let program = [
        0xf940_0001, // ldr x1, [x0]
        0xf900_0001, // str x1, [x0]
    ];
    let mut vcpu = guest.vcpu(&program, |regs| regs.x[0] = unmapped);
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::DataAbort {
            address: unmapped,
            write: false
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, guest.ram_base());

    // Skip the load, as an HLE kernel would after handling the fault.
    let mut regs = vcpu.regs().unwrap();
    regs.pc += 4;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::DataAbort {
            address: unmapped,
            write: true
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, guest.ram_base() + 4);
}

#[test]
fn udf_is_undefined_at_its_own_pc() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd503_201f, // nop
            0x0000_0000, // udf #0
        ],
        |_| {},
    );
    let pc = guest.ram_base() + 4;
    assert_eq!(vcpu.run().unwrap(), Exit::Undefined { pc });
    assert_eq!(vcpu.regs().unwrap().pc, pc);
}

#[test]
fn another_thread_interrupts_a_spinning_vcpu() {
    let Some(guest) = Guest::new() else { return };
    let flag = guest.data();
    let mut vcpu = guest.vcpu(
        &[
            0x5280_0021, // mov w1, #1
            0xb900_0001, // str w1, [x0]
            0x1400_0000, // b .
        ],
        |regs| regs.x[0] = flag,
    );
    let interrupter = vcpu.interrupter();
    let backend = Arc::clone(&guest.backend);
    // Wait until the guest is provably spinning, then interrupt it once.
    let waker = std::thread::spawn(move || {
        while read_u64(backend.memory(), flag) == 0 {
            std::thread::yield_now();
        }
        interrupter.interrupt().unwrap();
    });
    assert_eq!(vcpu.run().unwrap(), Exit::Interrupted);
    waker.join().unwrap();
    assert_eq!(vcpu.regs().unwrap().pc, guest.ram_base() + 8);
}

#[test]
fn an_interrupt_requested_while_stopped_ends_the_next_run() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    vcpu.interrupter().interrupt().unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Interrupted);
    // Nothing ran: the guest is still at its first instruction and resumes normally.
    assert_eq!(vcpu.regs().unwrap().pc, guest.ram_base());
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
}

#[test]
fn interrupting_a_destroyed_vcpu_is_harmless() {
    let Some(guest) = Guest::new() else { return };
    let vcpu = guest.vcpu(&[SVC_0], |_| {});
    let interrupter = vcpu.interrupter();
    drop(vcpu);
    interrupter.interrupt().unwrap();
}

#[test]
fn pointer_authentication_round_trips_and_matches_across_vcpus() {
    let Some(guest) = Guest::new() else { return };
    guest.load(&[
        0xdac1_0020, // pacia x0, x1
        0xaa00_03e2, // mov x2, x0
        0xdac1_1022, // autia x2, x1
        0xdac1_0423, // pacib x3, x1
        0xdac1_0825, // pacda x5, x1
        0xdac1_0c26, // pacdb x6, x1
        0x9ac1_3087, // pacga x7, x4, x1
        SVC_0,
    ]);
    let keys = PacKeys {
        ia: 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        ib: 0x1111_2222_3333_4444_5555_6666_7777_8888,
        da: 0x9999_aaaa_bbbb_cccc_dddd_eeee_ffff_0000,
        db: 0x0f0f_0f0f_f0f0_f0f0_0f0f_0f0f_f0f0_f0f0,
        ga: 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff,
    };
    let pointer = guest.data();
    let modifier = 0x1234;

    let here = pac_results(&guest.backend, &keys, pointer, modifier);
    // Every key is in use: each instruction changed the pointer...
    for signed in [here[0], here[3], here[5], here[6]] {
        assert_ne!(signed, pointer);
    }
    assert_ne!(here[7], 0, "pacga produced no PAC");
    // ...the keys differ, so do the signatures...
    assert_ne!(here[0], here[3]);
    assert_ne!(here[5], here[6]);
    // ...and authentication strips the signature again.
    assert_eq!(here[2], pointer);

    let backend = Arc::clone(&guest.backend);
    let there = std::thread::spawn(move || pac_results(&backend, &keys, pointer, modifier))
        .join()
        .unwrap();
    assert_eq!(here, there);
}

#[test]
fn thread_pointers_set_by_the_host_are_visible_at_el0() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd53b_d040, // mrs x0, tpidr_el0
            0xd53b_d061, // mrs x1, tpidrro_el0
            0x9100_03e2, // mov x2, sp
            SVC_0,
        ],
        |regs| {
            regs.tpidr = 0xaaaa_0000_1111;
            regs.tpidrro = 0xbbbb_0000_2222;
        },
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    let regs = vcpu.regs().unwrap();
    assert_eq!(regs.x[0], regs.tpidr);
    assert_eq!(regs.x[1], regs.tpidrro);
    assert_eq!(regs.x[2], regs.sp);
}

#[test]
fn floating_point_runs_at_el0() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0x1e6e_1000, // fmov d0, #1.0
            0x1e60_2801, // fadd d1, d0, d0
            0x9e66_0020, // fmov x0, d1
            SVC_0,
        ],
        |_| {},
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], 2.0f64.to_bits());
}

#[test]
fn counters_read_at_el0_without_a_host_exit() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd53b_e040, // mrs x0, cntvct_el0
            0xd53b_e021, // mrs x1, cntpct_el0   (traps to the shim)
            0xd53b_e029, // mrs x9, cntpct_el0   (a register the shim borrows)
            0xd53b_e042, // mrs x2, cntvct_el0
            SVC_0,
        ],
        |regs| {
            regs.x[10] = 0x10;
            regs.x[11] = 0x11;
        },
    );
    // Any exit to the host other than the final svc would surface here.
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    let x = vcpu.regs().unwrap().x;
    assert!(x[0] > 0, "cntvct_el0 read as 0");
    assert!(x[0] <= x[1] && x[1] <= x[9] && x[9] <= x[2], "{x:x?}");
    assert_eq!((x[10], x[11]), (0x10, 0x11));
}

#[test]
fn cache_maintenance_and_cache_info_work_at_el0() {
    let Some(guest) = Guest::new() else { return };
    let block = guest.data();
    guest
        .backend
        .memory()
        .write(block, &u64::MAX.to_le_bytes())
        .unwrap();
    let mut vcpu = guest.vcpu(
        &[
            0xd53b_0021, // mrs x1, ctr_el0
            0xd53b_00e2, // mrs x2, dczid_el0
            0xd50b_7420, // dc zva, x0
            SVC_0,
        ],
        |regs| regs.x[0] = block,
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    assert_ne!(vcpu.regs().unwrap().x[1], 0, "ctr_el0 read as 0");
    assert_eq!(read_u64(guest.backend.memory(), block), 0);
}

#[test]
fn interrupts_never_lose_or_repeat_guest_work() {
    let Some(guest) = Guest::new() else { return };
    // Most of this loop's time is spent in the EL1 shim (counter emulation and svc forwarding),
    // so interrupts land there as well as in guest code.
    let mut vcpu = guest.vcpu(
        &[
            0xd53b_e021, // 0x00  mrs x1, cntpct_el0
            0xd53b_e023, // 0x04  mrs x3, cntpct_el0
            0x9100_0442, // 0x08  add x2, x2, #1
            0xd400_00a1, // 0x0c  svc #5
            0x17ff_fffc, // 0x10  b 0x00
        ],
        |regs| {
            regs.x[9] = 0x99;
            regs.x[10] = 0x1010;
            regs.x[11] = 0x1111;
        },
    );
    let base = guest.ram_base();
    let interrupter = vcpu.interrupter();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let pest = std::thread::spawn(move || {
        while !stopped.load(std::sync::atomic::Ordering::Relaxed) {
            interrupter.interrupt().unwrap();
            std::thread::yield_now();
        }
    });
    let (mut svcs, mut interrupts) = (0u64, 0u64);
    while svcs < 2_000 {
        let exit = vcpu.run().unwrap();
        let regs = vcpu.regs().unwrap();
        assert_eq!((regs.x[9], regs.x[10], regs.x[11]), (0x99, 0x1010, 0x1111));
        match exit {
            Exit::Svc(5) => {
                svcs += 1;
                // Every svc reaches the host exactly once per loop iteration.
                assert_eq!(regs.x[2], svcs);
                assert_eq!(regs.pc, base + 0x10);
            }
            Exit::Interrupted => {
                interrupts += 1;
                assert!((base..=base + 0x10).contains(&regs.pc), "{:#x}", regs.pc);
            }
            other => assert_eq!(other, Exit::Svc(5), "unexpected exit"),
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    pest.join().unwrap();
    assert!(interrupts > 0, "no interrupt landed");
}
