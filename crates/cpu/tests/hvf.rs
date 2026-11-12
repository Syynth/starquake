//! Hypervisor.framework backend: guest programs run at EL0 in a real VM, in address spaces built
//! with `Stage1Tables`.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. Guest programs are raw instruction words with their assembly alongside.
//!
//! Hypervisor.framework allows one VM per process, so the tests take a lock before creating one
//! (`cargo nextest` runs each test in its own process anyway).
//!
//! Nothing here is identity mapped: the fixture maps guest RAM at [`support::WINDOW`], which is
//! not its guest-physical address, so a test that confused the two would read the wrong memory.
//! The fixture's address space is tagged with a non-zero address-space id.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::{Duration, Instant};

use starquake_cpu::hvf::{
    HvfBackend, HvfVcpu, MAX_PAGE_FLUSHES, PacKeys, PhysicalMemory, RamConfig,
};
use starquake_cpu::{
    Backend, Error, ExceptionClass, Exit, FaultStatus, GuestRegs, Interrupter, Syndrome, Vcpu,
};
use starquake_memory::stage1::{
    FlushRange, Stage1Tables, TableRoot, VA_SPACE_END, WalkError, translate,
};
use starquake_memory::{Access, BackingRange, Fault, GuestMemory, PAGE_SIZE, Perms, Translation};

/// Shared setup. In a `cfg(test)` module so it may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// One VM per process: tests that create a backend hold this.
    pub(crate) static VM_LOCK: Mutex<()> = Mutex::new(());

    /// Guest virtual address the fixture maps all of RAM at, readable, writable and executable:
    /// code goes at its start, the stack at its end.
    pub(crate) const WINDOW: u64 = 0x8000_0000;

    /// Where test data lives, relative to the start of the window.
    pub(crate) const DATA_OFFSET: u64 = 0x10_0000;

    /// Where the RAM pages that tests map a second time start, relative to the start of RAM:
    /// clear of the code, the data and the stack.
    pub(crate) const SPARE_OFFSET: u64 = 0x80_0000;

    /// The address-space id of the fixture's address space: not zero, and inside the eight
    /// bits every host has.
    pub(crate) const ASID: u16 = 0x5a;

    /// `svc #0`, the end of most test programs.
    pub(crate) const SVC_0: u32 = 0xd400_0001;

    /// All six permissions.
    pub(crate) const ALL_PERMS: [Perms; 6] = [
        Perms::None,
        Perms::Read,
        Perms::ReadWrite,
        Perms::Execute,
        Perms::ReadExecute,
        Perms::ReadWriteExecute,
    ];

    /// `svc #imm`.
    pub(crate) fn svc(imm: u16) -> u32 {
        SVC_0 | (u32::from(imm) << 5)
    }

    /// `movz x0, #imm`.
    pub(crate) fn movz_x0(imm: u16) -> u32 {
        0xd280_0000 | (u32::from(imm) << 5)
    }

    pub(crate) fn pages(n: u64) -> u64 {
        n * PAGE_SIZE
    }

    /// Guest RAM as guest code sees it through the fixture's window: virtual addresses.
    #[derive(Clone)]
    pub(crate) struct Window {
        physical: PhysicalMemory,
        ram: BackingRange,
    }

    impl Window {
        fn pa(&self, va: u64, len: usize) -> Result<u64, Fault> {
            let fault = Fault { address: va };
            let offset = va.checked_sub(WINDOW).ok_or(fault)?;
            let end = offset.checked_add(len as u64).ok_or(fault)?;
            if end > self.ram.size() {
                return Err(fault);
            }
            Ok(self.ram.base() + offset)
        }
    }

    impl GuestMemory for Window {
        fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
            self.physical.read(self.pa(va, buf.len())?, buf)
        }

        fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
            self.physical.write(self.pa(va, data.len())?, data)
        }
    }

    pub(crate) type Tables = Stage1Tables<PhysicalMemory>;

    pub(crate) struct Guest {
        pub(crate) backend: Arc<HvfBackend>,
        /// The fixture's address space: the window, plus whatever a test maps.
        pub(crate) tables: Arc<Tables>,
        /// Guest virtual address of the window.
        window: u64,
        _lock: MutexGuard<'static, ()>,
    }

    impl Guest {
        /// A fresh VM with the default RAM mapped at [`WINDOW`], or `None` on hosted CI.
        pub(crate) fn new() -> Option<Self> {
            Self::with_ram(RamConfig::default())
        }

        pub(crate) fn with_ram(ram: RamConfig) -> Option<Self> {
            if std::env::var_os("CI").is_some() {
                return None;
            }
            let lock = VM_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
            let backend = HvfBackend::new(ram).expect("create the VM");
            // The fixture's tables take the first half of the pool; `second_pool` is the rest.
            let pool = backend.table_pool();
            let half = BackingRange::new(pool.base(), pool.size() / 2).unwrap();
            let tables = Stage1Tables::new(backend.physical(), half).expect("create the tables");
            tables
                .map(WINDOW, backend.ram(), Perms::ReadWriteExecute)
                .expect("map the window");
            Some(Self {
                backend: Arc::new(backend),
                tables: Arc::new(tables),
                window: WINDOW,
                _lock: lock,
            })
        }

        /// The half of the table pool the fixture's own tables don't use.
        pub(crate) fn second_pool(&self) -> BackingRange {
            let pool = self.backend.table_pool();
            let half = pool.size() / 2;
            BackingRange::new(pool.base() + half, half).unwrap()
        }

        /// Guest virtual address of the start of the window: where code is loaded.
        pub(crate) fn base(&self) -> u64 {
            self.window
        }

        /// Guest virtual address one past the window: the first unmapped address above it.
        pub(crate) fn end(&self) -> u64 {
            self.window + self.backend.ram().size()
        }

        pub(crate) fn data(&self) -> u64 {
            self.window + DATA_OFFSET
        }

        /// RAM by guest virtual address, through the window.
        pub(crate) fn memory(&self) -> Window {
            Window {
                physical: self.backend.physical(),
                ram: self.backend.ram(),
            }
        }

        /// The guest-physical address of spare RAM page `n`.
        pub(crate) fn spare(&self, n: u64) -> u64 {
            self.backend.ram().base() + SPARE_OFFSET + pages(n)
        }

        /// One spare RAM page as backing for a mapping.
        pub(crate) fn spare_page(&self, n: u64) -> BackingRange {
            BackingRange::new(self.spare(n), PAGE_SIZE).unwrap()
        }

        /// The fixture's address space.
        pub(crate) fn root(&self) -> TableRoot {
            self.tables.root()
        }

        /// Copies `words` to the start of the window and returns their address.
        pub(crate) fn load(&self, words: &[u32]) -> u64 {
            self.load_at(0, words)
        }

        /// Copies `words` to `offset` bytes into the window and returns their address.
        pub(crate) fn load_at(&self, offset: u64, words: &[u32]) -> u64 {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let at = WINDOW + offset;
            self.memory().write(at, &bytes).expect("load code");
            at
        }

        /// A vCPU on this thread, in the fixture's address space, about to run `words` with
        /// `setup` applied to its registers.
        pub(crate) fn vcpu(&self, words: &[u32], setup: impl FnOnce(&mut GuestRegs)) -> HvfVcpu {
            let pc = self.load(words);
            let mut vcpu = new_vcpu(&self.backend, self.root());
            let mut regs = GuestRegs {
                pc,
                sp: self.end(),
                ..GuestRegs::default()
            };
            setup(&mut regs);
            vcpu.set_regs(&regs).expect("set registers");
            vcpu
        }

        /// Executes the flushes the fixture's tables have recorded.
        pub(crate) fn flush(&self, vcpu: &mut HvfVcpu) {
            vcpu.flush_pending(&self.tables, ASID).expect("flush");
        }
    }

    /// A vCPU on this thread in the address space at `root`, tagged [`ASID`].
    pub(crate) fn new_vcpu(backend: &HvfBackend, root: TableRoot) -> HvfVcpu {
        let mut vcpu = backend.create_vcpu().expect("create a vCPU");
        vcpu.set_address_space(root, ASID)
            .expect("set the address space");
        vcpu
    }

    /// Points the vCPU at `pc` with `x0` set, and runs it to its next exit.
    pub(crate) fn run_at(vcpu: &mut HvfVcpu, pc: u64, x0: u64) -> Exit {
        let mut regs = vcpu.regs().unwrap();
        regs.pc = pc;
        regs.x[0] = x0;
        vcpu.set_regs(&regs).unwrap();
        vcpu.run().unwrap()
    }

    pub(crate) fn read_u64(memory: &impl GuestMemory, address: u64) -> u64 {
        let mut buf = [0; 8];
        memory.read(address, &mut buf).expect("read guest memory");
        u64::from_le_bytes(buf)
    }

    pub(crate) fn write_u64(memory: &impl GuestMemory, address: u64, value: u64) {
        memory
            .write(address, &value.to_le_bytes())
            .expect("write guest memory");
    }

    /// Why a real vCPU must refuse an access the software walker refuses with `error`: the
    /// fault status its exit carries.
    pub(crate) fn status_of(tables: &Tables, error: WalkError) -> FaultStatus {
        match error {
            WalkError::Unmapped { level, .. } => FaultStatus::Translation { level },
            WalkError::PermissionDenied { va, .. } => FaultStatus::Permission {
                level: tables.walk(va).expect("a denied page is mapped").level,
            },
            other => unreachable!("the walker reported {other:?}"),
        }
    }

    /// The three probes of the cross-check, loaded at the start of the window. Each takes the
    /// address to try in x0.
    pub(crate) const PROBES: [u32; 5] = [
        0xf940_0001, // 0x00  ldr x1, [x0]      a load
        0xd400_0021, // 0x04  svc #1
        0xf900_0001, // 0x08  str x1, [x0]      a store
        0xd400_0041, // 0x0c  svc #2
        0xd61f_0000, // 0x10  br  x0            a fetch
    ];
    pub(crate) const LOAD_PROBE: u64 = 0x00;
    pub(crate) const STORE_PROBE: u64 = 0x08;
    pub(crate) const FETCH_PROBE: u64 = 0x10;
    /// What a probed page holds at its start: `svc #3`, so a fetch that succeeds comes straight back.
    pub(crate) const FETCHED: u16 = 3;
    /// Offset in a probed page of the word the load and store probes use.
    pub(crate) const WORD_OFFSET: u64 = 8;

    /// What one probe of `va` must do on a real vCPU, given what the software walker says about it,
    /// checked against what it did.
    pub(crate) fn check_probe(
        guest: &Guest,
        vcpu: &mut HvfVcpu,
        va: u64,
        access: Access,
        context: &str,
    ) {
        let physical = guest.backend.physical();
        let base = guest.base();
        let (probe, address) = match access {
            Access::Read => (LOAD_PROBE, va + WORD_OFFSET),
            Access::Write => (STORE_PROBE, va + WORD_OFFSET),
            Access::Execute => (FETCH_PROBE, va),
        };
        let stored = 0x5707_0000_0000_0000 | va;
        let mut regs = vcpu.regs().unwrap();
        regs.x[1] = stored;
        vcpu.set_regs(&regs).unwrap();
        let walked = translate(&physical, guest.root(), address, access);
        let before = walked.as_ref().ok().map(|&pa| read_u64(&physical, pa));
        let exit = run_at(vcpu, base + probe, address);
        let after = vcpu.regs().unwrap();
        match (access, walked) {
            (Access::Read, Ok(pa)) => {
                assert_eq!(exit, Exit::Svc(1), "{context}");
                // The vCPU read the very word the walker's physical address names.
                assert_eq!(after.x[1], read_u64(&physical, pa), "{context}");
                assert_eq!(Some(after.x[1]), before, "{context}");
            }
            (Access::Write, Ok(pa)) => {
                assert_eq!(exit, Exit::Svc(2), "{context}");
                assert_eq!(read_u64(&physical, pa), stored, "{context}");
                assert_ne!(before, Some(stored), "{context}: the store changed nothing");
            }
            (Access::Execute, Ok(_)) => {
                assert_eq!(exit, Exit::Svc(FETCHED), "{context}");
                assert_eq!(after.pc, va + 4, "{context}");
            }
            (Access::Read | Access::Write, Err(error)) => {
                let write = access == Access::Write;
                let status = status_of(&guest.tables, error);
                assert_eq!(
                    exit,
                    Exit::DataAbort {
                        address,
                        write,
                        syndrome: Syndrome::data_abort(status, write),
                    },
                    "{context}"
                );
                assert_eq!(after.pc, base + probe, "{context}");
            }
            (Access::Execute, Err(error)) => {
                let status = status_of(&guest.tables, error);
                assert_eq!(
                    exit,
                    Exit::InstructionAbort {
                        address,
                        syndrome: Syndrome::instruction_abort(status),
                    },
                    "{context}"
                );
                assert_eq!(after.pc, va, "{context}");
            }
        }
    }

    /// A guest loop that never leaves the guest by itself: it loads from the page in x0, compares
    /// what it read with x5, counts the iteration in x2 and publishes the count at the address in
    /// x3. A value other than x5 ends it with `svc #9`.
    pub(crate) const SPIN_ON_A_PAGE: [u32; 7] = [
        0xf940_0004, // 0x00  ldr  x4, [x0]
        0xeb05_009f, // 0x04  cmp  x4, x5
        0x5400_0081, // 0x08  b.ne 0x18
        0x9100_0442, // 0x0c  add  x2, x2, #1
        0xf900_0062, // 0x10  str  x2, [x3]
        0x17ff_fffb, // 0x14  b    0x00
        0xd400_0121, // 0x18  svc  #9
    ];

    /// What the spinning vCPU's thread tells the test.
    #[derive(Debug, PartialEq)]
    pub(crate) enum Spin {
        /// It faulted on the page with this many iterations done.
        Faulted {
            /// Completed iterations.
            iterations: u64,
            /// The data abort.
            exit: Exit,
        },
        /// Anything else, which fails the test.
        Unexpected(String),
    }

    /// Runs [`SPIN_ON_A_PAGE`] on a vCPU of its own thread. Every time the guest stops, the thread
    /// reports why and waits for a go-ahead before running it again (`false` ends it).
    pub(crate) fn spin_on_a_page(
        guest: &Guest,
        page: u64,
        expected: u64,
        counter: u64,
    ) -> (
        mpsc::Receiver<Spin>,
        mpsc::Sender<bool>,
        std::thread::JoinHandle<()>,
    ) {
        let code = guest.load_at(0x1000, &SPIN_ON_A_PAGE);
        let (report, reports) = mpsc::channel();
        let (go, resume) = mpsc::channel();
        let backend = Arc::clone(&guest.backend);
        let root = guest.root();
        let thread = std::thread::spawn(move || {
            let mut vcpu = new_vcpu(&backend, root);
            let mut regs = GuestRegs {
                pc: code,
                ..GuestRegs::default()
            };
            regs.x[0] = page;
            regs.x[3] = counter;
            regs.x[5] = expected;
            vcpu.set_regs(&regs).unwrap();
            loop {
                let message = match vcpu.run() {
                    Ok(exit @ Exit::DataAbort { .. }) => Spin::Faulted {
                        iterations: vcpu.regs().unwrap().x[2],
                        exit,
                    },
                    Ok(other) => {
                        Spin::Unexpected(format!("{other:?}, {:x?}", vcpu.regs().unwrap()))
                    }
                    Err(error) => Spin::Unexpected(error.to_string()),
                };
                if report.send(message).is_err() || resume.recv() != Ok(true) {
                    return;
                }
            }
        });
        (reports, go, thread)
    }

    /// Waits until the published count has passed `at_least`.
    pub(crate) fn wait_for_count(memory: &impl GuestMemory, counter: u64, at_least: u64) -> u64 {
        let started = Instant::now();
        loop {
            let count = read_u64(memory, counter);
            if count > at_least {
                return count;
            }
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "the spinning vCPU made no progress"
            );
            std::thread::yield_now();
        }
    }

    /// Runs the PAuth program on a fresh vCPU and returns x0-x7.
    pub(crate) fn pac_results(
        backend: &HvfBackend,
        root: TableRoot,
        keys: &PacKeys,
        pointer: u64,
        modifier: u64,
    ) -> [u64; 8] {
        let mut vcpu = new_vcpu(backend, root);
        vcpu.set_pac_keys(keys).unwrap();
        let mut regs = GuestRegs {
            pc: WINDOW,
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

use support::{
    ALL_PERMS, ASID, FETCH_PROBE, FETCHED, Guest, LOAD_PROBE, PROBES, STORE_PROBE, SVC_0, Spin,
    WORD_OFFSET, check_probe, movz_x0, pac_results, pages, read_u64, run_at, spin_on_a_page,
    status_of, svc, wait_for_count, write_u64,
};

// ---------------------------------------------------------------------------------------------
// Address spaces: the software walker against a real vCPU.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_software_walker_and_a_real_vcpu_agree_for_every_permission() {
    let Some(guest) = Guest::new() else { return };
    let physical = guest.backend.physical();
    // One page per permission, spread over the address space: its first and last pages, both
    // sides of a level-1 boundary, and the page below the window. Their backing is handed out
    // in the opposite order, so no two neighbours in one space are neighbours in the other.
    let vas = [
        0,
        (1 << 30) - PAGE_SIZE,
        1 << 30,
        guest.base() - PAGE_SIZE,
        0x12_3456_7000,
        VA_SPACE_END - PAGE_SIZE,
    ];
    for ((va, perms), n) in vas.into_iter().zip(ALL_PERMS).zip((0..6).rev()) {
        guest.tables.map(va, guest.spare_page(n), perms).unwrap();
        // Each page starts with `svc #3` and holds a word naming its own physical page.
        let pa = guest.spare(n);
        physical.write(pa, &svc(FETCHED).to_le_bytes()).unwrap();
        write_u64(&physical, pa + WORD_OFFSET, 0x9a9e_0000_0000_0000 | pa);
    }
    // Addresses nothing maps, missing at each level of the walk: an untouched level-1 slot, an
    // untouched level-2 slot under a used level-1 slot, and the page after a mapped one.
    let unmapped = [100 << 30, (1 << 30) + (5 << 21), PAGE_SIZE];
    for (va, level) in unmapped.into_iter().zip(1..) {
        assert_eq!(
            guest.tables.walk(va),
            Err(WalkError::Unmapped { va, level })
        );
    }

    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    assert_ne!(ASID, 0);
    let mut allowed = Vec::new();
    for (va, perms) in vas.into_iter().zip(ALL_PERMS) {
        for access in [Access::Read, Access::Write, Access::Execute] {
            check_probe(
                &guest,
                &mut vcpu,
                va,
                access,
                &format!("{perms:?} {access:?}"),
            );
            allowed.push(perms.allows(access));
        }
    }
    for va in unmapped {
        for access in [Access::Read, Access::Write, Access::Execute] {
            check_probe(
                &guest,
                &mut vcpu,
                va,
                access,
                &format!("{va:#x} {access:?}"),
            );
        }
    }
    // The cross-check saw every combination go both ways: it did not pass by refusing (or
    // allowing) everything. Rows are permissions, columns read, write, execute.
    #[rustfmt::skip]
    assert_eq!(allowed, [
        false, false, false, // None
        true,  false, false, // Read
        true,  true,  false, // ReadWrite
        false, false, true,  // Execute (execute-only)
        true,  false, true,  // ReadExecute
        true,  true,  true,  // ReadWriteExecute
    ]);
}

#[test]
fn a_store_to_a_read_only_page_and_a_jump_to_a_data_page_are_the_right_exits() {
    let Some(guest) = Guest::new() else { return };
    let physical = guest.backend.physical();
    let read_only = 0x40_0000_0000;
    let data = read_only + pages(3);
    guest
        .tables
        .map(read_only, guest.spare_page(0), Perms::Read)
        .unwrap();
    guest
        .tables
        .map(data, guest.spare_page(1), Perms::ReadWrite)
        .unwrap();
    write_u64(&physical, guest.spare(0) + 0x18, 0x1111);
    let mut vcpu = guest.vcpu(&PROBES, |regs| regs.x[1] = 0x2222);
    let base = guest.base();

    // The page can be read...
    assert_eq!(
        run_at(&mut vcpu, base + LOAD_PROBE, read_only + 0x18),
        Exit::Svc(1)
    );
    assert_eq!(vcpu.regs().unwrap().x[1], 0x1111);
    // ...and a store to it is a write permission fault at the address stored to, with the
    // program counter still on the store and the page unchanged.
    assert_eq!(
        run_at(&mut vcpu, base + STORE_PROBE, read_only + 0x18),
        Exit::DataAbort {
            address: read_only + 0x18,
            write: true,
            syndrome: Syndrome::data_abort(FaultStatus::Permission { level: 3 }, true),
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, base + STORE_PROBE);
    assert_eq!(read_u64(&physical, guest.spare(0) + 0x18), 0x1111);

    // A jump into a page that is readable and writable but not executable is an instruction
    // abort at the address jumped to, which is also where the program counter is.
    assert_eq!(
        run_at(&mut vcpu, base + FETCH_PROBE, data + 0x40),
        Exit::InstructionAbort {
            address: data + 0x40,
            syndrome: Syndrome::instruction_abort(FaultStatus::Permission { level: 3 }),
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, data + 0x40);
    // The same page takes a store.
    assert_eq!(
        run_at(&mut vcpu, base + STORE_PROBE, data + 0x40),
        Exit::Svc(2)
    );
    assert_eq!(read_u64(&physical, guest.spare(1) + 0x40), 0x1111);
}

#[test]
fn a_vcpu_that_was_given_no_address_space_refuses_to_run() {
    let Some(guest) = Guest::new() else { return };
    let pc = guest.load(&[SVC_0]);
    let mut vcpu = guest.backend.create_vcpu().unwrap();
    let regs = GuestRegs {
        pc,
        ..GuestRegs::default()
    };
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(vcpu.run(), Err(Error::NoAddressSpace));
    assert_eq!(
        Error::NoAddressSpace.to_string(),
        "the vCPU has no address space: give it one before running it"
    );
    // A refused address space does not count as one.
    let outside = TableRoot::new(guest.backend.ram().base()).unwrap();
    assert!(vcpu.set_address_space(outside, ASID).is_err());
    assert_eq!(vcpu.run(), Err(Error::NoAddressSpace));
    // Given the address space, the same vCPU runs the program from where it was pointed.
    assert_eq!(vcpu.regs().unwrap(), regs);
    vcpu.set_address_space(guest.root(), ASID).unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
}

#[test]
fn an_address_space_id_must_fit_the_width_the_host_gives_a_guest() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    let bits = vcpu.asid_bits();
    assert!(bits == 8 || bits == 16, "{bits}");
    let page = FlushRange {
        va: guest.base(),
        size: PAGE_SIZE,
        leaf_only: true,
    };
    // The widest id that fits is accepted, for an address space and for a flush.
    let widest = u16::MAX >> (16 - bits);
    vcpu.set_address_space(guest.root(), widest).unwrap();
    vcpu.flush_tlb(widest, &[page]).unwrap();
    assert_eq!(
        run_at(&mut vcpu, guest.base() + LOAD_PROBE, guest.data()),
        Exit::Svc(1)
    );
    // One more is refused, where sixteen bits can hold one more, and so is every bit above:
    // the vCPU keeps its address space and nothing is flushed.
    let before = vcpu.regs().unwrap();
    for shift in bits..16 {
        let asid = 1u16 << shift;
        let too_wide = Error::AsidTooWide { asid, bits };
        assert_eq!(
            vcpu.set_address_space(guest.root(), asid),
            Err(too_wide.clone())
        );
        assert_eq!(vcpu.flush_tlb(asid, &[page]), Err(too_wide));
    }
    assert_eq!(
        Error::AsidTooWide {
            asid: 0x100,
            bits: 8
        }
        .to_string(),
        "address-space id 0x100 does not fit the 8 bits this host gives a guest"
    );
    assert_eq!(vcpu.regs().unwrap(), before);
    assert_eq!(
        run_at(&mut vcpu, guest.base() + LOAD_PROBE, guest.data()),
        Exit::Svc(1)
    );
}

#[test]
fn a_table_root_outside_the_table_pool_is_refused_and_changes_nothing() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    let pool = guest.backend.table_pool();
    let ram = guest.backend.ram();
    // In RAM (which the guest can write), on either side of the pool, and far away.
    for outside in [
        ram.base(),
        ram.end() - PAGE_SIZE,
        pool.base() - PAGE_SIZE,
        pool.end(),
        0,
    ] {
        let root = TableRoot::new(outside).unwrap();
        assert_eq!(
            vcpu.set_address_space(root, ASID),
            Err(Error::BadTableRoot { root: outside }),
            "{outside:#x}"
        );
    }
    assert_eq!(
        Error::BadTableRoot { root: ram.base() }.to_string(),
        format!(
            "table root {:#x} is not inside the backend's table pool",
            ram.base()
        )
    );
    // The vCPU still has the address space it had.
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));

    // The first and the last page of the pool are inside. The last one holds no table, so the
    // program's first fetch finds nothing mapped. (It gets an id of its own: under the
    // fixture's, the window's translations are still cached.)
    let last = TableRoot::new(pool.end() - PAGE_SIZE).unwrap();
    vcpu.set_address_space(last, ASID + 1).unwrap();
    assert_eq!(
        run_at(&mut vcpu, guest.base(), 0),
        Exit::InstructionAbort {
            address: guest.base(),
            syndrome: Syndrome::instruction_abort(FaultStatus::Translation { level: 1 }),
        }
    );
    assert_eq!(guest.root().pa(), pool.base());
    vcpu.set_address_space(guest.root(), ASID).unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
}

#[test]
fn address_spaces_keep_their_own_translations_without_a_flush_between_them() {
    let Some(guest) = Guest::new() else { return };
    let physical = guest.backend.physical();
    // A second address space: the same window, and the same extra page mapped onto a different
    // physical page. Neither id is the fixture's.
    let (first_id, second_id) = (1, 2);
    let other = Stage1Tables::new(guest.backend.physical(), guest.second_pool()).unwrap();
    other
        .map(guest.base(), guest.backend.ram(), Perms::ReadWriteExecute)
        .unwrap();
    let shared = 0x7_0000_0000;
    guest
        .tables
        .map(shared, guest.spare_page(0), Perms::Read)
        .unwrap();
    other.map(shared, guest.spare_page(1), Perms::Read).unwrap();
    write_u64(&physical, guest.spare(0), 0xaaaa);
    write_u64(&physical, guest.spare(1), 0xbbbb);

    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    let probe = guest.base() + LOAD_PROBE;
    for round in 0..200 {
        for (root, id, expected) in [
            (guest.root(), first_id, 0xaaaa),
            (other.root(), second_id, 0xbbbb),
        ] {
            vcpu.set_address_space(root, id).unwrap();
            assert_eq!(run_at(&mut vcpu, probe, shared), Exit::Svc(1));
            assert_eq!(vcpu.regs().unwrap().x[1], expected, "round {round}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// TLB invalidation.
// ---------------------------------------------------------------------------------------------

#[test]
fn an_unmapped_page_faults_after_the_flush_and_a_protected_one_refuses_stores() {
    let Some(guest) = Guest::new() else { return };
    let physical = guest.backend.physical();
    // Two pages under one level-3 table, so unmapping the first leaves the table in place.
    let page = 0x20_0000_0000;
    let neighbour = page + PAGE_SIZE;
    guest
        .tables
        .map(page, guest.spare_page(0), Perms::ReadWrite)
        .unwrap();
    guest
        .tables
        .map(neighbour, guest.spare_page(1), Perms::ReadWrite)
        .unwrap();
    write_u64(&physical, guest.spare(0), 0x1111);
    let mut vcpu = guest.vcpu(&PROBES, |regs| regs.x[1] = 0x2222);
    let (load, store) = (guest.base() + LOAD_PROBE, guest.base() + STORE_PROBE);

    // Touch both pages, so their translations are cached.
    assert_eq!(run_at(&mut vcpu, load, page), Exit::Svc(1));
    assert_eq!(vcpu.regs().unwrap().x[1], 0x1111);
    assert_eq!(run_at(&mut vcpu, store, neighbour), Exit::Svc(2));

    // Unmap, flush, touch: a translation fault at the page.
    guest.tables.unmap(page, PAGE_SIZE).unwrap();
    let before = vcpu.regs().unwrap();
    guest.flush(&mut vcpu);
    assert_eq!(vcpu.regs().unwrap(), before, "a flush changes no register");
    assert_eq!(guest.tables.take_pending_flushes(), []);
    assert_eq!(
        run_at(&mut vcpu, load, page),
        Exit::DataAbort {
            address: page,
            write: false,
            syndrome: Syndrome::data_abort(FaultStatus::Translation { level: 3 }, false),
        }
    );
    // The neighbour is untouched by the flush of the other page.
    assert_eq!(run_at(&mut vcpu, load, neighbour), Exit::Svc(1));
    assert_eq!(vcpu.regs().unwrap().x[1], 0x1111);

    // Mapping the page again, onto other memory, needs no flush: nothing was cached for it.
    write_u64(&physical, guest.spare(2), 0x3333);
    guest
        .tables
        .map(page, guest.spare_page(2), Perms::ReadWrite)
        .unwrap();
    assert_eq!(run_at(&mut vcpu, load, page), Exit::Svc(1));
    assert_eq!(vcpu.regs().unwrap().x[1], 0x3333);

    // Write-protect it (what write tracking does): after the flush a store is a permission
    // fault and a load still works; restoring the permission and flushing lets the store in.
    assert_eq!(run_at(&mut vcpu, store, page), Exit::Svc(2));
    guest.tables.protect(page, PAGE_SIZE, Perms::Read).unwrap();
    guest.flush(&mut vcpu);
    assert_eq!(
        run_at(&mut vcpu, store, page),
        Exit::DataAbort {
            address: page,
            write: true,
            syndrome: Syndrome::data_abort(FaultStatus::Permission { level: 3 }, true),
        }
    );
    assert_eq!(run_at(&mut vcpu, load, page), Exit::Svc(1));
    guest
        .tables
        .protect(page, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    guest.flush(&mut vcpu);
    let mut regs = vcpu.regs().unwrap();
    regs.x[1] = 0x4444;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(run_at(&mut vcpu, store, page), Exit::Svc(2));
    assert_eq!(read_u64(&physical, guest.spare(2)), 0x4444);
}

#[test]
fn unmapping_a_whole_table_and_runs_at_the_page_by_page_limit_fault_after_the_flush() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    let load = guest.base() + LOAD_PROBE;
    let in_use = guest.tables.table_pages_in_use();

    // A single page: unmapping it frees its level-3 and level-2 tables, so the flush is the
    // every-level kind, and the fault afterwards is at level 1.
    let single = 0x30_0000_0000;
    guest
        .tables
        .map(single, guest.spare_page(0), Perms::Read)
        .unwrap();
    assert_eq!(guest.tables.table_pages_in_use(), in_use + 2);
    assert_eq!(run_at(&mut vcpu, load, single), Exit::Svc(1));
    guest.tables.unmap(single, PAGE_SIZE).unwrap();
    guest.flush(&mut vcpu);
    assert_eq!(guest.tables.table_pages_in_use(), in_use);
    assert_eq!(
        run_at(&mut vcpu, load, single),
        Exit::DataAbort {
            address: single,
            write: false,
            syndrome: Syndrome::data_abort(FaultStatus::Translation { level: 1 }, false),
        }
    );

    // The freed table pages are reused for a mapping elsewhere, which works at once.
    let elsewhere = 0x31_0000_0000;
    guest
        .tables
        .map(elsewhere, guest.spare_page(1), Perms::Read)
        .unwrap();
    assert_eq!(run_at(&mut vcpu, load, elsewhere), Exit::Svc(1));

    // The longest run of pages the backend flushes one by one, and one page more (for which
    // it drops everything cached for the address space instead): touch the first, middle and
    // last pages, unmap, flush.
    let long = 0x32_0000_0000;
    for count in [MAX_PAGE_FLUSHES, MAX_PAGE_FLUSHES + 1] {
        let backing = BackingRange::new(guest.spare(0), pages(count)).unwrap();
        guest.tables.map(long, backing, Perms::Read).unwrap();
        let touched = [long, long + pages(count / 2), long + pages(count - 1)];
        for va in touched {
            assert_eq!(run_at(&mut vcpu, load, va), Exit::Svc(1), "{count} pages");
        }
        guest.tables.unmap(long, pages(count)).unwrap();
        guest.flush(&mut vcpu);
        for va in touched {
            assert_eq!(
                run_at(&mut vcpu, load, va),
                Exit::DataAbort {
                    address: va,
                    write: false,
                    syndrome: Syndrome::data_abort(FaultStatus::Translation { level: 1 }, false),
                },
                "{count} pages, {va:#x}"
            );
        }
    }
    // The window survived all of it: the probe itself still runs from there.
    assert_eq!(run_at(&mut vcpu, load, elsewhere), Exit::Svc(1));
}

#[test]
fn a_bad_flush_range_is_refused_before_anything_runs() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |regs| regs.x[1] = 0x77);
    let before = vcpu.regs().unwrap();
    let good = FlushRange {
        va: guest.base(),
        size: PAGE_SIZE,
        leaf_only: true,
    };
    for (va, size) in [
        (guest.base(), 0),
        (guest.base() + 1, PAGE_SIZE),
        (guest.base(), PAGE_SIZE - 1),
        (VA_SPACE_END - PAGE_SIZE, pages(2)),
        (u64::MAX - (PAGE_SIZE - 1), PAGE_SIZE),
    ] {
        let bad = FlushRange {
            va,
            size,
            leaf_only: false,
        };
        assert_eq!(
            vcpu.flush_tlb(ASID, &[good, bad]),
            Err(Error::BadFlushRange { va, size }),
            "{va:#x} {size:#x}"
        );
    }
    // Nothing ran and nothing changed: the vCPU still runs its program from the start.
    assert_eq!(vcpu.regs().unwrap(), before);
    vcpu.flush_tlb(ASID, &[]).unwrap();
    vcpu.flush_tlb(ASID, &[good]).unwrap();
    assert_eq!(vcpu.regs().unwrap(), before);
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[1], 0x77);
}

#[test]
fn a_flush_that_fails_leaves_the_tables_owing_it() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    let page = 0x33_0000_0000;
    guest
        .tables
        .map(page, guest.spare_page(0), Perms::Read)
        .unwrap();
    guest.tables.unmap(page, PAGE_SIZE).unwrap();
    // A flush the tables' ranges can't fail, so make one fail the only way a caller can: by
    // running it through the tables with a closure that reports an error.
    let failed: Result<(), Error> = guest
        .tables
        .flush_pending_with(|_| Err(Error::Host("stopped".into())));
    assert_eq!(failed, Err(Error::Host("stopped".into())));
    // The range is still owed, and the real flush then runs it.
    let owed = FlushRange {
        va: page,
        size: PAGE_SIZE,
        leaf_only: false,
    };
    let mut offered = Vec::new();
    guest
        .tables
        .flush_pending_with(|ranges| {
            offered.extend_from_slice(ranges);
            vcpu.flush_tlb(ASID, ranges)
        })
        .unwrap();
    assert_eq!(offered, [owed]);
    assert_eq!(guest.tables.take_pending_flushes(), []);
}

#[test]
fn an_interrupt_during_a_flush_is_reported_by_the_next_run() {
    let Some(guest) = Guest::new() else { return };
    let page = 0x34_0000_0000;
    guest
        .tables
        .map(page, guest.spare_page(0), Perms::Read)
        .unwrap();
    let mut vcpu = guest.vcpu(&PROBES, |_| {});
    let load = guest.base() + LOAD_PROBE;
    assert_eq!(run_at(&mut vcpu, load, page), Exit::Svc(1));

    // The interrupt is pending when the flush enters the guest, so it stops the flush's first
    // entry. The flush must still happen in full, and the interrupt must not be lost.
    guest.tables.unmap(page, PAGE_SIZE).unwrap();
    vcpu.interrupter().interrupt().unwrap();
    let before = vcpu.regs().unwrap();
    guest.flush(&mut vcpu);
    assert_eq!(vcpu.regs().unwrap(), before);
    let mut regs = before.clone();
    regs.pc = load;
    regs.x[0] = page;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Interrupted);
    // No guest code ran for that exit, and the interrupt is used up: now the load runs, and
    // finds the page gone.
    assert_eq!(vcpu.regs().unwrap(), regs);
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::DataAbort {
            address: page,
            write: false,
            syndrome: Syndrome::data_abort(FaultStatus::Translation { level: 1 }, false),
        }
    );
}

/// How many more iterations a spinning vCPU may complete after a flush on another vCPU has
/// returned. A flush that does not reach it leaves it reading the stale page until the host
/// happens to interrupt it: 130 ms and more in `tools/hvprobe/README.md`, hundreds of millions
/// of iterations of a loop this short. One that does reach it stops it within microseconds.
const ITERATIONS_AFTER_A_FLUSH: u64 = 2_000_000;

#[test]
fn a_flush_on_one_vcpu_stops_another_that_never_leaves_the_guest() {
    let Some(guest) = Guest::new() else { return };
    let memory = guest.memory();
    let physical = guest.backend.physical();
    let page = 0x35_0000_0000;
    let counter = guest.data();
    let marker = 0x600d_600d;
    write_u64(&physical, guest.spare(0), marker);
    guest
        .tables
        .map(page, guest.spare_page(0), Perms::Read)
        .unwrap();
    let mut flusher = guest.vcpu(&[SVC_0], |_| {});
    let (reports, go, thread) = spin_on_a_page(&guest, page, marker, counter);

    let mut seen = 0;
    for round in 0..40 {
        // The other vCPU is spinning on the page, with its translation cached.
        wait_for_count(&memory, counter, seen + 1000);
        guest.tables.unmap(page, PAGE_SIZE).unwrap();
        guest.flush(&mut flusher);
        let at_flush = read_u64(&memory, counter);
        let report = reports
            .recv_timeout(Duration::from_secs(60))
            .expect("the spinning vCPU never faulted");
        assert!(
            matches!(report, Spin::Faulted { .. }),
            "round {round}: {report:?}"
        );
        let Spin::Faulted { iterations, exit } = report else {
            return;
        };
        // A load from the page that found it unmapped. Once the flush has run that is at
        // level 1 (the page was alone under its tables, which the unmap unlinked). But the
        // spinner does not always keep its cached translation until the flush: on a busy
        // host it often walks again earlier, and then faults at level 3 or 2. Measured with
        // level 1 as the only accepted answer: 32 of 40 runs of this test failed while other
        // builds kept all eight cores busy (17 at level 2, 15 at level 3); the same assertion
        // had passed while the machine was quieter. Two ways to get there, which this test does
        // not tell apart (a guess): the walk falls between the unmap's three descriptor
        // writes (the page, then each table), or it starts from a cached walk entry that
        // still points at a table the unmap has unlinked. Either way the page is gone for
        // the spinner at that moment, which is all the caller of `unmap` may rely on.
        let unmapped = [1, 2, 3].map(|level| Exit::DataAbort {
            address: page,
            write: false,
            syndrome: Syndrome::data_abort(FaultStatus::Translation { level }, false),
        });
        assert!(unmapped.contains(&exit), "round {round}: {exit:?}");
        let after = iterations.saturating_sub(at_flush);
        assert!(
            after <= ITERATIONS_AFTER_A_FLUSH,
            "round {round}: {after} iterations after the flush returned"
        );
        seen = iterations;
        guest
            .tables
            .map(page, guest.spare_page(0), Perms::Read)
            .unwrap();
        go.send(true).unwrap();
    }
    // End the spinner: unmap once more, and tell it to stop when it faults.
    guest.tables.unmap(page, PAGE_SIZE).unwrap();
    guest.flush(&mut flusher);
    assert!(matches!(
        reports.recv_timeout(Duration::from_secs(60)),
        Ok(Spin::Faulted { .. })
    ));
    go.send(false).unwrap();
    thread.join().unwrap();
}

#[test]
fn a_running_vcpu_never_sees_half_a_descriptor() {
    // Guest RAM above 4 GiB puts output-address bits in the upper half of every page
    // descriptor, so a descriptor with only its lower half written would send the walker to
    // memory that does not exist, and the vCPU would stop with a host error.
    let Some(guest) = Guest::new() else { return };
    assert!(guest.backend.ram().base() > u64::from(u32::MAX));
    let memory = guest.memory();
    let physical = guest.backend.physical();
    let page = 0x36_0000_0000;
    let counter = guest.data();
    let marker = 0x0dd_ba11;
    write_u64(&physical, guest.spare(0), marker);
    let backing = guest.spare_page(0);
    guest.tables.map(page, backing, Perms::Read).unwrap();
    let mut flusher = guest.vcpu(&[SVC_0], |_| {});
    let (reports, go, thread) = spin_on_a_page(&guest, page, marker, counter);
    wait_for_count(&memory, counter, 1000);

    // Unmap and map the page as fast as the flushes allow while the other vCPU keeps walking
    // to it. The page is alone under its level-3 and level-2 tables, so every round also
    // unlinks both tables and links them again. The spinner either reads the marker or takes a
    // translation fault at the page; it is resumed after each fault.
    let done = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&done);
    let resumer = std::thread::spawn(move || {
        let mut faults = 0u64;
        loop {
            let report = reports.recv_timeout(Duration::from_secs(60));
            // The only thing the spinner may report: a load from the page that found it
            // unmapped, at whichever level the unmap had reached.
            let fault = match &report {
                Ok(Spin::Faulted {
                    exit:
                        Exit::DataAbort {
                            address,
                            write: false,
                            syndrome,
                        },
                    ..
                }) if *address == page => syndrome.fault(),
                _ => None,
            };
            assert!(
                matches!(fault, Some(FaultStatus::Translation { .. })),
                "after {faults} faults: {report:?}"
            );
            faults += 1;
            let more = !stop.load(Ordering::SeqCst);
            go.send(more).unwrap();
            if !more {
                return faults;
            }
        }
    });
    for _ in 0..3000 {
        guest.tables.unmap(page, PAGE_SIZE).unwrap();
        guest.flush(&mut flusher);
        guest.tables.map(page, backing, Perms::Read).unwrap();
    }
    // Leave the page unmapped, so the spinner faults once more and is told to stop.
    done.store(true, Ordering::SeqCst);
    guest.tables.unmap(page, PAGE_SIZE).unwrap();
    guest.flush(&mut flusher);
    let faults = resumer.join().unwrap();
    thread.join().unwrap();
    assert!(faults > 0);
    // It really was running throughout.
    assert!(read_u64(&memory, counter) > 1000);
}

// ---------------------------------------------------------------------------------------------
// Exits, registers and what runs at EL0.
// ---------------------------------------------------------------------------------------------

#[test]
fn undefined_exits_say_what_the_exception_was() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0x0000_0000, // 0x00  udf    #0
            0xd420_2460, // 0x04  brk    #0x123
            0xdac1_1020, // 0x08  autia  x0, x1      (x0 was never signed)
            0xd53b_e220, // 0x0c  mrs    x0, cntp_ctl_el0   (EL0 may not use the timer)
        ],
        |_| {},
    );
    let base = guest.base();
    // Each is an undefined exit at the instruction's own address; what differs is the syndrome.
    let mut at = |offset: u64| {
        let exit = run_at(&mut vcpu, base + offset, 0x1234);
        let syndrome = match exit {
            Exit::Undefined { pc, syndrome } if pc == base + offset => Some(syndrome),
            _ => None,
        };
        assert!(syndrome.is_some(), "{exit:?}");
        assert_eq!(vcpu.regs().unwrap().pc, base + offset);
        syndrome
    };
    assert_eq!(at(0x00), Some(Syndrome::undefined()));
    assert_eq!(
        at(0x04).map(|brk| (brk.class(), brk.immediate())),
        Some((ExceptionClass::Breakpoint, Some(0x123)))
    );
    assert_eq!(
        at(0x08).map(Syndrome::class),
        Some(ExceptionClass::PointerAuthentication)
    );
    assert_eq!(
        at(0x0c).map(Syndrome::class),
        Some(ExceptionClass::TrappedSystemAccess)
    );
}

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
    assert_eq!(regs.pc, guest.base() + 8);
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
    // Just past the window: nothing maps it.
    let unmapped = guest.end() + 0x10;
    let status = status_of(&guest.tables, guest.tables.walk(unmapped).unwrap_err());
    assert_eq!(status, FaultStatus::Translation { level: 2 });
    let program = [
        0xf940_0001, // ldr x1, [x0]
        0xf900_0001, // str x1, [x0]
    ];
    let mut vcpu = guest.vcpu(&program, |regs| regs.x[0] = unmapped);
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::DataAbort {
            address: unmapped,
            write: false,
            syndrome: Syndrome::data_abort(status, false),
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, guest.base());

    // Skip the load, as an HLE kernel would after handling the fault.
    let mut regs = vcpu.regs().unwrap();
    regs.pc += 4;
    vcpu.set_regs(&regs).unwrap();
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::DataAbort {
            address: unmapped,
            write: true,
            syndrome: Syndrome::data_abort(status, true),
        }
    );
    assert_eq!(vcpu.regs().unwrap().pc, guest.base() + 4);
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
    let pc = guest.base() + 4;
    assert_eq!(
        vcpu.run().unwrap(),
        Exit::Undefined {
            pc,
            syndrome: Syndrome::undefined()
        }
    );
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
    let memory = guest.memory();
    // Wait until the guest is provably spinning, then interrupt it once.
    let waker = std::thread::spawn(move || {
        while read_u64(&memory, flag) == 0 {
            std::thread::yield_now();
        }
        interrupter.interrupt().unwrap();
    });
    assert_eq!(vcpu.run().unwrap(), Exit::Interrupted);
    waker.join().unwrap();
    assert_eq!(vcpu.regs().unwrap().pc, guest.base() + 8);
}

#[test]
fn an_interrupt_requested_while_stopped_ends_the_next_run() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    vcpu.interrupter().interrupt().unwrap();
    assert_eq!(vcpu.run().unwrap(), Exit::Interrupted);
    // Nothing ran: the guest is still at its first instruction and resumes normally.
    assert_eq!(vcpu.regs().unwrap().pc, guest.base());
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

    let root = guest.root();
    let here = pac_results(&guest.backend, root, &keys, pointer, modifier);
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
    let there = std::thread::spawn(move || pac_results(&backend, root, &keys, pointer, modifier))
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
fn the_guest_reads_the_counter_frequency_the_backend_reports() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(
        &[
            0xd53b_e000, // mrs x0, cntfrq_el0
            SVC_0,
        ],
        |regs| regs.x[0] = 0,
    );
    assert_eq!(vcpu.run().unwrap(), Exit::Svc(0));
    let guest_reads = vcpu.regs().unwrap().x[0];
    assert_ne!(guest_reads, 0);
    assert_eq!(guest_reads, guest.backend.counter_frequency());
}

#[test]
fn cache_maintenance_and_cache_info_work_at_el0() {
    let Some(guest) = Guest::new() else { return };
    let block = guest.data();
    write_u64(&guest.memory(), block, u64::MAX);
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
    assert_eq!(read_u64(&guest.memory(), block), 0);
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
    let base = guest.base();
    let interrupter = vcpu.interrupter();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let pest = std::thread::spawn(move || {
        while !stopped.load(Ordering::Relaxed) {
            interrupter.interrupt().unwrap();
            std::thread::yield_now();
        }
    });
    let (mut svcs, mut interrupts) = (0u64, 0u64);
    // Keep going until interrupts have landed too, however late the interrupting thread gets
    // scheduled on a loaded machine.
    let started = Instant::now();
    while svcs < 2_000 || interrupts < 10 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "only {interrupts} interrupts landed in 60 s"
        );
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
    stop.store(true, Ordering::Relaxed);
    pest.join().unwrap();
    assert!(interrupts > 0, "no interrupt landed");
}

#[test]
fn code_rewritten_by_the_host_runs_on_the_same_vcpu() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[movz_x0(0), svc(0)], |_| {});
    let base = guest.base();
    // Stale instruction-cache lines show up within a few rewrites; a thousand leave no doubt.
    for i in 1..=1000 {
        guest.load(&[movz_x0(i), svc(i)]);
        let mut regs = vcpu.regs().unwrap();
        regs.pc = base;
        vcpu.set_regs(&regs).unwrap();
        assert_eq!(vcpu.run().unwrap(), Exit::Svc(i), "rewrite {i}");
        assert_eq!(vcpu.regs().unwrap().x[0], u64::from(i), "rewrite {i}");
    }
}

#[test]
fn code_written_in_pieces_is_seen_whole() {
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    let base = guest.base();
    let memory = guest.memory();
    // Rewrites that straddle cache lines and end mid-line, one word at a time.
    for i in 1..=200u16 {
        for (word, offset) in [movz_x0(i), svc(i)].into_iter().zip([60, 64]) {
            memory.write(base + offset, &word.to_le_bytes()).unwrap();
        }
        let mut regs = vcpu.regs().unwrap();
        regs.pc = base + 60;
        vcpu.set_regs(&regs).unwrap();
        assert_eq!(vcpu.run().unwrap(), Exit::Svc(i), "rewrite {i}");
        assert_eq!(vcpu.regs().unwrap().x[0], u64::from(i), "rewrite {i}");
    }
}

#[test]
fn a_new_vm_never_runs_the_previous_vms_code() {
    // Each VM's RAM is fresh host memory, which the allocator and the kernel hand out again from
    // pages an earlier VM's guest executed. Alternate two programs at the same address, with the
    // run on a different thread each time (and so often on a different core), and check that
    // every VM runs the program it was given.
    for round in 0..100u16 {
        let imm = 1 + round % 2;
        std::thread::spawn(move || {
            let Some(guest) = Guest::new() else { return };
            let mut vcpu = guest.vcpu(&[movz_x0(imm), svc(imm), SVC_0], |_| {});
            assert_eq!(vcpu.run().unwrap(), Exit::Svc(imm), "round {round}");
            assert_eq!(vcpu.regs().unwrap().x[0], u64::from(imm), "round {round}");
        })
        .join()
        .unwrap();
    }
}

#[test]
fn svc_reaches_the_host_whatever_x0_holds() {
    // Hypervisor.framework answers an `hvc #0` itself, without exiting, when w0 looks like an
    // SMCCC function id it knows (observed: top byte 0xc1). The guest's x0 is an arbitrary
    // argument at every svc, so the shim must not forward with `hvc #0`
    // (see "SMCCC" in `tools/hvprobe/README.md`).
    let Some(guest) = Guest::new() else { return };
    let mut vcpu = guest.vcpu(&[SVC_0], |_| {});
    let base = guest.base();
    let mut values: Vec<u64> = (0..=0xff).map(|top| (top << 24) | 0x12_3456).collect();
    values.extend([
        0xc100_0000,
        0xc100_0001,
        0xc1ff_ffff,
        0xffff_ffff_c100_0000,
        0x29bb_c193_b0af, // a counter value that tripped it
        0xc400_0003,
        0x8400_0000,
    ]);
    for x0 in values {
        let mut regs = GuestRegs {
            pc: base,
            sp: guest.data(),
            ..GuestRegs::default()
        };
        regs.x[0] = x0;
        regs.x[1] = 0x1111;
        vcpu.set_regs(&regs).unwrap();
        assert_eq!(vcpu.run().unwrap(), Exit::Svc(0), "x0 = {x0:#x}");
        let after = vcpu.regs().unwrap();
        assert_eq!((after.x[0], after.x[1]), (x0, 0x1111), "x0 = {x0:#x}");
        assert_eq!(after.pc, base + 4, "x0 = {x0:#x}");
    }
}
