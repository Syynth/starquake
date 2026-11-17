//! A guest program that walks its address space with `QueryMemory` the way a module walker
//! does, run natively under the hypervisor backend: it must find its own module and no other,
//! even though its counter read was patched and its counter trampolines are mapped below it.
//!
//! The walker's rule is Atmosphère's (`stratosphere/dmnt.gen2/source/dmnt2_debug_process.cpp`,
//! `DebugProcess::CollectModules` @ 4e3f3703, fact only): start at address 0, take every block
//! whose state is `Code` and whose permission is read-execute for a module, move to the end of
//! each block, and stop at the inaccessible block above the address space. A module walker
//! then reads the word at the module's offset 4 as the offset of its `MOD0` header; this one
//! does too, and checks that the header is the program's own. That a title's `rtld` walks with
//! the same rule is a guess (`crates/core/src/image.rs`, "Permissions").
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM
//! fails the test.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

#[cfg(test)]
mod common;

/// Everything is in a `cfg(test)` module so that the helpers may unwrap like the tests do.
#[cfg(test)]
mod walk {
    use starquake_core::crash::Exception;
    use starquake_core::{ExitReason, RunReport};
    use starquake_kernel::memory::SvcMemoryInfo;
    use starquake_kernel::memory::state::{MemoryState, SvcPermission};
    use starquake_kernel::svc::SvcId;
    use starquake_testkit::a64::{self, Assembler, SysReg};
    use starquake_testkit::nro::{CODE_OFFSET, MOD0_OFFSET, MOD0_POINTER_OFFSET, NroBuilder};
    use starquake_testkit::programs::SVC_EXIT_PROCESS;

    use super::common::{GENEROUS, run};

    /// `XZR` as an operand of `sub`, and `SP` as the base of `add` (`a64` module docs).
    const ZR_OR_SP: u8 = a64::MAX_REG;
    /// Where in the program the walker stops when something is wrong, each at its own `brk`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Failure {
        /// `QueryMemory` returned an error.
        QueryFailed = 1,
        /// A block the walker takes for a module is not the program's own module.
        ExtraModule = 2,
        /// The program's own module's `MOD0` is not where its offset 4 says.
        NoMod0 = 3,
        /// The walk found other than exactly one module.
        WrongCount = 4,
    }

    /// The walker program, and the image offset of each failure's `brk`.
    struct Walker {
        nro: Vec<u8>,
        failures: Vec<(Failure, u64)>,
    }

    /// The byte offset in `SvcMemoryInfo`'s guest layout of the field `pick` sets: found by
    /// giving it a value no other field has.
    fn field_offset(pick: impl Fn(&mut SvcMemoryInfo)) -> u32 {
        let mut info = SvcMemoryInfo {
            base_address: 0,
            size: 0,
            state: 0,
            attribute: 0,
            permission: 0,
            ipc_count: 0,
            device_count: 0,
        };
        pick(&mut info);
        let bytes = info.to_bytes();
        let at = bytes.iter().position(|&byte| byte != 0).unwrap();
        u32::try_from(at).unwrap()
    }

    /// The image offset of the instruction at index `word`.
    fn code_offset(word: usize) -> u64 {
        u64::try_from(CODE_OFFSET + word * a64::INSN_SIZE as usize).unwrap()
    }

    /// Loads the 64-bit `value` into `rd` with `movz` and three `movk`s.
    fn load_constant(asm: &mut Assembler, rd: u8, value: u64) {
        let half = |shift: u32| u32::try_from((value >> shift) & 0xFFFF).unwrap();
        asm.emit(a64::movz(rd, half(0), 0).unwrap());
        for shift in [16, 32, 48] {
            asm.emit(a64::movk(rd, half(shift), shift).unwrap());
        }
    }

    /// A program that reads the counter once (so a run patches it and maps trampolines), then
    /// walks the address space from 0 as the module docs say and exits only if the one module
    /// it finds is its own, with its own `MOD0` where the module's offset 4 says.
    ///
    /// ```text
    ///       adr  x20, <image start>        ; the program's own module
    ///       mrs  x9, cntpct_el0            ; the read a run patches
    ///       add  x19, sp, #0
    ///       sub  x19, x19, #0x30           ; the MemoryInfo, on the stack
    ///       sub  x21, x19, #4              ; so a 64-bit load puts a u32 field in the top half
    ///       movz x22, #0                   ; the address to ask about
    ///       movz x23, #0                   ; modules found
    /// loop: add  x0, x19, #0
    ///       mov  x2, x22
    ///       svc  #0x6                      ; QueryMemory
    ///       cbnz x0, query_failed
    ///       ldr  x12, [x19, #base]
    ///       ldr  x13, [x19, #size]
    ///       sub  x14, xzr, x13
    ///       sub  x22, x12, x14             ; the next address: base + size
    ///       ldr  x10, [x21, #state]
    ///       lsr  x10, x10, #32             ; the state
    ///       sub  x11, x10, #Inaccessible
    ///       cbz  x11, done
    ///       sub  x11, x10, #Code
    ///       cbnz x11, loop
    ///       ldr  x10, [x21, #permission]
    ///       lsr  x10, x10, #32             ; the permission
    ///       sub  x11, x10, #ReadExecute
    ///       cbnz x11, loop
    ///       sub  x11, x12, x20             ; a module: is it ours?
    ///       cbnz x11, extra_module
    ///       ldr  x10, [x12, #0]
    ///       lsr  x10, x10, #32             ; the word at offset 4: where MOD0 is
    ///       sub  x14, xzr, x10
    ///       sub  x10, x12, x14             ; MOD0's address
    ///       ldr  x10, [x10, #0]
    ///       <x15 = the file's first 8 bytes of MOD0>
    ///       sub  x11, x10, x15
    ///       cbnz x11, no_mod0
    ///       add  x23, x23, #1
    ///       b    loop
    /// done: sub  x11, x23, #1
    ///       cbnz x11, wrong_count
    ///       svc  #0x7
    /// query_failed: brk #1 ... wrong_count: brk #4
    /// ```
    fn walker() -> Walker {
        let base_at = field_offset(|info| info.base_address = 1);
        let size_at = field_offset(|info| info.size = 1);
        let state_at = field_offset(|info| info.state = 1);
        let permission_at = field_offset(|info| info.permission = 1);
        let info_size = u32::try_from(SvcMemoryInfo::SIZE).unwrap();
        // The 32-bit fields are read as the top half of a 64-bit load from 4 bytes before.
        const HALF: u32 = 4;
        let svc = |id: SvcId| u32::from(id.number());

        // The MOD0 the program's file holds, whose first 8 bytes the walker compares.
        let file_mod0 =
            |nro: &[u8]| u64::from_le_bytes(nro[MOD0_OFFSET..MOD0_OFFSET + 8].try_into().unwrap());
        // The file does not depend on the constant the program compares with, only on how
        // many instructions there are, so the constant is read from a first build.
        let build = |mod0: u64| {
            let mut asm = Assembler::new();
            let (top, done) = (asm.label(), asm.label());
            let failures = [
                Failure::QueryFailed,
                Failure::ExtraModule,
                Failure::NoMod0,
                Failure::WrongCount,
            ]
            .map(|failure| (failure, asm.label()));
            let label = |which: Failure| failures.iter().find(|(f, _)| *f == which).unwrap().1;

            let here = i64::try_from(code_offset(asm.len())).unwrap();
            asm.emit(a64::adr(20, -here).unwrap());
            asm.emit(a64::mrs(9, SysReg::CntPctEl0).unwrap());
            asm.emit(a64::add_imm(19, ZR_OR_SP, 0).unwrap());
            asm.emit(a64::sub_imm(19, 19, info_size.next_multiple_of(16)).unwrap());
            asm.emit(a64::sub_imm(21, 19, HALF).unwrap());
            asm.emit(a64::movz(22, 0, 0).unwrap());
            asm.emit(a64::movz(23, 0, 0).unwrap());
            asm.bind(top).unwrap();
            asm.emit(a64::add_imm(0, 19, 0).unwrap());
            asm.emit(a64::mov(2, 22).unwrap());
            asm.emit(a64::svc(svc(SvcId::QueryMemory)).unwrap());
            asm.cbnz(0, label(Failure::QueryFailed)).unwrap();
            asm.emit(a64::ldr(12, 19, base_at).unwrap());
            asm.emit(a64::ldr(13, 19, size_at).unwrap());
            asm.emit(a64::sub(14, ZR_OR_SP, 13).unwrap());
            asm.emit(a64::sub(22, 12, 14).unwrap());
            asm.emit(a64::ldr(10, 21, state_at).unwrap());
            asm.emit(a64::lsr_imm(10, 10, 32).unwrap());
            asm.emit(a64::sub_imm(11, 10, MemoryState::Inaccessible.svc()).unwrap());
            asm.cbz(11, done).unwrap();
            asm.emit(a64::sub_imm(11, 10, MemoryState::Code.svc()).unwrap());
            asm.cbnz(11, top).unwrap();
            asm.emit(a64::ldr(10, 21, permission_at).unwrap());
            asm.emit(a64::lsr_imm(10, 10, 32).unwrap());
            let read_execute = u32::from(SvcPermission::ReadExecute.bits());
            asm.emit(a64::sub_imm(11, 10, read_execute).unwrap());
            asm.cbnz(11, top).unwrap();
            asm.emit(a64::sub(11, 12, 20).unwrap());
            asm.cbnz(11, label(Failure::ExtraModule)).unwrap();
            let pointer_at = u32::try_from(MOD0_POINTER_OFFSET).unwrap();
            // The pointer is the top half of the module's first 8 bytes.
            assert_eq!(pointer_at, HALF);
            asm.emit(a64::ldr(10, 12, 0).unwrap());
            asm.emit(a64::lsr_imm(10, 10, 32).unwrap());
            asm.emit(a64::sub(14, ZR_OR_SP, 10).unwrap());
            asm.emit(a64::sub(10, 12, 14).unwrap());
            asm.emit(a64::ldr(10, 10, 0).unwrap());
            load_constant(&mut asm, 15, mod0);
            asm.emit(a64::sub(11, 10, 15).unwrap());
            asm.cbnz(11, label(Failure::NoMod0)).unwrap();
            asm.emit(a64::add_imm(23, 23, 1).unwrap());
            asm.b(top).unwrap();
            asm.bind(done).unwrap();
            asm.emit(a64::sub_imm(11, 23, 1).unwrap());
            asm.cbnz(11, label(Failure::WrongCount)).unwrap();
            asm.emit(a64::svc(SVC_EXIT_PROCESS).unwrap());
            let mut offsets = Vec::new();
            for (code, (failure, at)) in (1..).zip(failures) {
                offsets.push((failure, code_offset(asm.len())));
                asm.bind(at).unwrap();
                asm.emit(a64::brk(code).unwrap());
            }
            Walker {
                nro: NroBuilder::new(&asm.finish().unwrap()).build().unwrap(),
                failures: offsets,
            }
        };
        let first = build(0);
        let walker = build(file_mod0(&first.nro));
        assert_eq!(file_mod0(&walker.nro), file_mod0(&first.nro));
        walker
    }

    /// Where the program was placed: its text's first byte.
    fn image_base(report: &RunReport) -> u64 {
        report
            .memory_map
            .iter()
            .find(|mapping| mapping.name == "main.text")
            .unwrap()
            .start
    }

    /// Which of the walker's checks a run stopped at, if it stopped at one of its `brk`s.
    fn failure(walker: &Walker, report: &RunReport) -> Option<Failure> {
        let crash = report.crash()?;
        if crash.exception != Exception::UndefinedInstruction {
            return None;
        }
        let offset = crash.pc.checked_sub(image_base(report))?;
        walker
            .failures
            .iter()
            .find(|(_, at)| *at == offset)
            .map(|(failure, _)| *failure)
    }

    #[test]
    fn a_module_walk_finds_the_programs_own_module_and_not_its_trampolines() {
        let walker = walker();
        let Some(report) = run(&walker.nro, GENEROUS) else {
            return;
        };
        assert_eq!(failure(&walker, &report), None, "{:?}", report.crash());
        assert_eq!(
            report.exit,
            ExitReason::ProcessExited,
            "{:?}",
            report.crash()
        );
        // The walk ran with trampolines mapped below the module: the counter read was patched.
        let base = image_base(&report);
        let trampolines = report
            .memory_map
            .iter()
            .find(|mapping| mapping.name == "trampolines")
            .expect("the counter read was patched");
        assert_eq!(trampolines.end, base);
        assert_eq!(report.patched_counter_reads.len(), 1);
    }
}
