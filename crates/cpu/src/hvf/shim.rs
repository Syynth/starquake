//! The guest-side EL1 shim: an exception vector table that turns EL0 exceptions into host exits.
//!
//! Guest code runs at EL0. Every exception it takes lands in this table at EL1, which either
//! handles it in the guest (a trapped `mrs xN, cntpct_el0`, answered from the virtual counter) or
//! hands it to the host with `hvc #2`. The host then reads `ESR_EL1`, `FAR_EL1` and `ELR_EL1`
//! directly, so the shim passes nothing in general-purpose registers. Between runs a vCPU is
//! *parked* on the shim's `eret`: its EL0 state lives in `ELR_EL1`/`SPSR_EL1`, and every run starts
//! by returning to EL0.
//!
//! The shim borrows x9-x11 while it works and saves them in a per-vCPU scratch area just below
//! `SP_EL1`. If the host interrupts a vCPU part-way through the shim, [`recovery`] says how to
//! undo the partial work so the EL0 state is exact.
//!
//! The shim is mapped in the upper half of the address space, through `TTBR1_EL1` and static
//! tables the backend owns (`docs/specs/execution-model.md`, "Native backends"), so the tables
//! a guest's address space is built from never describe it.
//!
//! # TLB flush
//!
//! The host cannot invalidate a guest's stage-1 TLB entries, so the shim has three entry points
//! that do, each ending in `hvc #3` ([`HVC_FLUSH_DONE`]) back to the host. The host enters one
//! on a parked vCPU with the operand in x0 (and a page count in x1), having saved the EL0
//! values of both, and restores them when the `hvc` arrives; nothing else of the EL0 state is
//! touched (the flags `subs` sets are EL1's own). Each is the sequence the Arm Architecture
//! Reference Manual (Arm DDI 0487) gives for a change to live translation tables, "TLB
//! maintenance requirements and the TLB maintenance instructions": `dsb ishst` so the changed
//! descriptors are visible to every walker, the invalidation, `dsb ish` to wait for it on every
//! core, and `isb`. All three use the broadcast (`IS`) forms.
//!
//! - [`FLUSH_PAGES`]: `tlbi vae1is` for x1 pages from the one x0 names, every level of the walk.
//! - [`FLUSH_LEAVES`]: the same with `tlbi vale1is`, last-level entries only.
//! - [`FLUSH_ASID`]: `tlbi aside1is`, everything cached under the address-space id in x0.
//!
//! An interrupt that stops a vCPU inside a flush needs no recovery: the host runs it again from
//! where it stopped.
//!
//! Ported from `tools/hvprobe/src/guest.rs` (vector table, `hvc` forwarding, in-guest counter
//! emulation); this version preserves every EL0 register and emulates `cntpct_el0` for any `Rt`.
//! The words were assembled with LLVM from the assembly beside them. Exception classes and the
//! ISS layout are from the Arm Architecture Reference Manual (DDI 0487), `ESR_EL1`.

use crate::{ExceptionClass, Exit, Syndrome};

/// Size of the shim in bytes.
pub(super) const LEN: usize = FLUSH_END;

/// Offset of the "lower EL, AArch64, synchronous" vector (Arm ARM, `VBAR_EL1` vector offsets).
pub(super) const EL0_SYNC: usize = 0x400;
/// First instruction after the shim has saved x9-x11.
pub(super) const AFTER_SAVE: usize = 0x408;
/// First instruction after the counter path has advanced `ELR_EL1`.
pub(super) const ELR_BUMPED: usize = 0x44c;
/// Start of the path that restores x9-x11 and exits to the host.
pub(super) const TO_HOST: usize = 0x450;
/// The `eret` a vCPU is parked on between runs (right after the `hvc`).
pub(super) const PARKED: usize = 0x45c;
/// Start of the 32 counter stubs, one per `Rt` (31 is `xzr`).
pub(super) const STUBS: usize = 0x800;
/// Size of one counter stub.
pub(super) const STUB_LEN: usize = 16;
/// Offset of the `eret` inside a counter stub (after it has written `Rt`).
const STUB_ERET: usize = 12;
/// End of the counter stubs.
const STUBS_END: usize = STUBS + 32 * STUB_LEN;
/// Entry point: invalidate x1 pages starting with the one x0 names, at every level.
pub(super) const FLUSH_PAGES: usize = STUBS_END;
/// Entry point: invalidate the last-level entries of x1 pages starting with the one x0 names.
pub(super) const FLUSH_LEAVES: usize = FLUSH_PAGES + 4 * FLUSH_BY_PAGE.len();
/// Entry point: invalidate everything cached under the address-space id in x0.
pub(super) const FLUSH_ASID: usize = FLUSH_LEAVES + 4 * FLUSH_BY_PAGE.len();
/// End of the flush entry points, and of the shim.
const FLUSH_END: usize = FLUSH_ASID + 4 * FLUSH_WHOLE_ASID.len();

/// How far below `SP_EL1` the shim saves x9 (x10 and x11 follow at +8 and +16).
pub(super) const SCRATCH_BELOW_SP: u64 = 32;

/// `hvc` immediate for "EL0 took an exception the host must handle". It must not be 0: the
/// guest's x0 is arbitrary at an exception (an `svc` argument, a counter value), and
/// Hypervisor.framework treats `hvc #0` as an SMCCC call and answers it itself, without exiting,
/// when w0 looks like a function id it knows (observed: top byte 0xc1). The guest then resumes
/// past the `svc` with x0-x3 overwritten (`tools/hvprobe/README.md`, "SMCCC"). Other immediates
/// always reach the host.
pub(super) const HVC_EL0_EXCEPTION: u16 = 2;
/// `hvc` immediate for "a flush entry point has finished" (see "TLB flush" in the module docs).
/// Not 0, for the reason above; x0 holds the last `TLBI` operand there.
pub(super) const HVC_FLUSH_DONE: u16 = 3;

/// `CPSR` of a parked vCPU: `EL1h` with D, A, I and F masked (`tools/hvprobe/src/probe.rs`).
pub(super) const CPSR_EL1H_MASKED: u64 = 0x3c5;
/// `SPSR_EL1` bits for `EL0t` with D, A, I and F masked (`tools/hvprobe/src/probe.rs`).
pub(super) const SPSR_EL0T_MASKED: u64 = 0x3c0;
/// `PSTATE.M[3:0]` values (Arm ARM, `SPSR_EL1`).
pub(super) const MODE_MASK: u64 = 0xf;
pub(super) const MODE_EL0T: u64 = 0x0;
pub(super) const MODE_EL1H: u64 = 0x5;

/// `ESR_EL2.EC` of an `hvc` from AArch64 state (Arm ARM): the only exception the shim sends to
/// the host. The classes of exceptions EL0 takes are [`ExceptionClass`]'s.
pub(super) const EC_HVC64: u8 = 0x16;

/// `hvc #imm`.
const fn hvc(imm: u16) -> u32 {
    0xd400_0002 | ((imm as u32) << 5)
}

const HVC_TO_HOST: u32 = hvc(HVC_EL0_EXCEPTION);
const HVC_AFTER_FLUSH: u32 = hvc(HVC_FLUSH_DONE);
const HVC_1: u32 = hvc(1); // hvc #1: an exception the shim doesn't expect
const ERET: u32 = 0xd69f_03e0; // eret
const STUB_RESTORE: [u32; 2] = [
    0xa97e_2be9, // ldp  x9, x10, [sp, #-32]
    0xf85f_03eb, // ldur x11, [sp, #-16]
];
const MRS_CNTVCT_X0: u32 = 0xd53b_e040; // mrs x0, cntvct_el0 (Rt in bits 4:0)

/// The handler at [`EL0_SYNC`].
const EL0_SYNC_HANDLER: [u32; 24] = [
    0xa93e_2be9, // 0x400  stp  x9, x10, [sp, #-32]
    0xf81f_03eb, // 0x404  stur x11, [sp, #-16]
    0xd538_5209, // 0x408  mrs  x9, esr_el1                 AFTER_SAVE
    0xd35a_fd2a, // 0x40c  lsr  x10, x9, #26                EC
    0xf100_615f, // 0x410  cmp  x10, #0x18                  trapped MSR/MRS?
    0x5400_01e1, // 0x414  b.ne TO_HOST
    0x529f_83eb, // 0x418  mov  w11, #0xfc1f
    0x72a0_3feb, // 0x41c  movk w11, #0x1ff, lsl #16        ISS mask without Rt (bits 9:5)
    0x8a0b_012a, // 0x420  and  x10, x9, x11
    0x529f_002b, // 0x424  mov  w11, #0xf801
    0x72a0_064b, // 0x428  movk w11, #0x32, lsl #16         ISS of `mrs xN, cntpct_el0`
    0xeb0b_015f, // 0x42c  cmp  x10, x11
    0x5400_0101, // 0x430  b.ne TO_HOST
    0xd345_252a, // 0x434  ubfx x10, x9, #5, #5             Rt
    0x1000_1e4b, // 0x438  adr  x11, STUBS
    0x8b0a_116b, // 0x43c  add  x11, x11, x10, lsl #4       16 bytes per stub
    0xd538_402a, // 0x440  mrs  x10, elr_el1
    0x9100_114a, // 0x444  add  x10, x10, #4                skip the mrs
    0xd518_402a, // 0x448  msr  elr_el1, x10
    0xd61f_0160, // 0x44c  br   x11                         ELR_BUMPED
    0xa97e_2be9, // 0x450  ldp  x9, x10, [sp, #-32]         TO_HOST
    0xf85f_03eb, // 0x454  ldur x11, [sp, #-16]
    HVC_TO_HOST, // 0x458  hvc  #HVC_EL0_EXCEPTION
    ERET,        // 0x45c  eret                             PARKED
];

const DSB_ISHST: u32 = 0xd503_3a9f; // dsb ishst
const DSB_ISH: u32 = 0xd503_3b9f; // dsb ish
const ISB: u32 = 0xd503_3fdf; // isb
const TLBI_VAE1IS_X0: u32 = 0xd508_8320; // tlbi vae1is, x0
const TLBI_VALE1IS_X0: u32 = 0xd508_83a0; // tlbi vale1is, x0
const TLBI_ASIDE1IS_X0: u32 = 0xd508_8340; // tlbi aside1is, x0

/// The code at [`FLUSH_PAGES`] and, with [`TLBI_VALE1IS_X0`] as its second word, at
/// [`FLUSH_LEAVES`]. x1 must not be zero: the loop counts down to it.
const FLUSH_BY_PAGE: [u32; 8] = [
    DSB_ISHST,       // +0x00  dsb  ishst
    TLBI_VAE1IS_X0,  // +0x04  tlbi vae1is, x0
    0x9100_0400,     // +0x08  add  x0, x0, #1           the next page's operand
    0xf100_0421,     // +0x0c  subs x1, x1, #1
    0x54ff_ffa1,     // +0x10  b.ne +0x04
    DSB_ISH,         // +0x14  dsb  ish
    ISB,             // +0x18  isb
    HVC_AFTER_FLUSH, // +0x1c  hvc  #HVC_FLUSH_DONE
];
/// Index of the `tlbi` in [`FLUSH_BY_PAGE`].
const FLUSH_BY_PAGE_TLBI: usize = 1;

/// The code at [`FLUSH_ASID`].
const FLUSH_WHOLE_ASID: [u32; 5] = [
    DSB_ISHST,        // +0x00  dsb  ishst
    TLBI_ASIDE1IS_X0, // +0x04  tlbi aside1is, x0
    DSB_ISH,          // +0x08  dsb  ish
    ISB,              // +0x0c  isb
    HVC_AFTER_FLUSH,  // +0x10  hvc  #HVC_FLUSH_DONE
];

/// The shim's instruction words, in order from offset 0.
pub(super) const WORDS: [u32; LEN / 4] = build();

const fn build() -> [u32; LEN / 4] {
    // Every vector the shim doesn't use reports itself to the host with `hvc #1`.
    let mut words = [HVC_1; LEN / 4];
    let mut i = 0;
    while i < EL0_SYNC_HANDLER.len() {
        words[EL0_SYNC / 4 + i] = EL0_SYNC_HANDLER[i];
        i += 1;
    }
    // Stub `Rt`: restore x9-x11, then `mrs xRt, cntvct_el0` and `eret`.
    let (mut rt, mut stub) = (0, STUBS / 4);
    while stub < STUBS_END / 4 {
        words[stub] = STUB_RESTORE[0];
        words[stub + 1] = STUB_RESTORE[1];
        words[stub + 2] = MRS_CNTVCT_X0 | rt;
        words[stub + 3] = ERET;
        rt += 1;
        stub += STUB_LEN / 4;
    }
    let mut i = 0;
    while i < FLUSH_BY_PAGE.len() {
        words[FLUSH_PAGES / 4 + i] = FLUSH_BY_PAGE[i];
        words[FLUSH_LEAVES / 4 + i] = if i == FLUSH_BY_PAGE_TLBI {
            TLBI_VALE1IS_X0
        } else {
            FLUSH_BY_PAGE[i]
        };
        i += 1;
    }
    let mut i = 0;
    while i < FLUSH_WHOLE_ASID.len() {
        words[FLUSH_ASID / 4 + i] = FLUSH_WHOLE_ASID[i];
        i += 1;
    }
    words
}

/// Turns an exception EL0 took (as the shim reported it) into an [`Exit`].
///
/// SVCs carry their immediate, and `ELR_EL1` already points past the `svc`. Data aborts and
/// instruction aborts report `FAR_EL1`. Every other class (among them UNDEFINED, `brk`, a failed
/// `AUT*`, a trapped system register access the shim doesn't emulate, PC and SP alignment
/// faults) is [`Exit::Undefined`] at `ELR_EL1`. Data-access alignment faults are data aborts
/// (EC 0x24) and report `FAR_EL1`. Each exit but the SVC carries `esr` as its syndrome.
pub(super) fn decode_el0_exception(esr: u64, far: u64, elr: u64) -> Exit {
    let syndrome = Syndrome::from_esr(esr);
    match syndrome.class() {
        ExceptionClass::Svc => Exit::Svc(syndrome.immediate().unwrap_or_default()),
        ExceptionClass::DataAbort => Exit::DataAbort {
            address: far,
            write: syndrome.is_write(),
            syndrome,
        },
        ExceptionClass::InstructionAbort => Exit::InstructionAbort {
            address: far,
            syndrome,
        },
        _ => Exit::Undefined { pc: elr, syndrome },
    }
}

/// What the host must undo before parking a vCPU that was interrupted inside the shim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Recovery {
    /// x9-x11 hold shim values: reload the EL0 values from the scratch area.
    pub restore_scratch: bool,
    /// `ELR_EL1` is one instruction ahead of where EL0 must resume: subtract 4.
    pub rewind_elr: bool,
}

/// How to recover a vCPU interrupted at `offset` into the shim, with `esr` in `ESR_EL1`; `None`
/// if the shim never stops there while handling an EL0 exception.
///
/// The aim is the state EL0 was in when it took the exception, so the exception happens again on
/// resume. Saved registers can be reloaded at any point between the save and the `eret`, because
/// the scratch area keeps its contents. An SVC's `ELR_EL1` already points past the `svc`, so it
/// is rewound to run the `svc` again.
pub(super) fn recovery(offset: usize, esr: u64) -> Option<Recovery> {
    let svc = Syndrome::from_esr(esr).class() == ExceptionClass::Svc;
    let recovery = |restore_scratch, rewind_elr| {
        Some(Recovery {
            restore_scratch,
            rewind_elr,
        })
    };
    match offset {
        PARKED => recovery(false, false),
        o if (EL0_SYNC..AFTER_SAVE).contains(&o) => recovery(false, svc),
        o if (AFTER_SAVE..ELR_BUMPED).contains(&o) => recovery(true, svc),
        // Counter path: ELR_EL1 already skips the `mrs`.
        o if (ELR_BUMPED..TO_HOST).contains(&o) => recovery(true, true),
        o if (TO_HOST..PARKED).contains(&o) => recovery(true, svc),
        // At a stub's `eret` the `mrs` is done: the emulation is complete.
        o if (STUBS..STUBS_END).contains(&o) && (o - STUBS) % STUB_LEN == STUB_ERET => {
            recovery(false, false)
        }
        o if (STUBS..STUBS_END).contains(&o) => recovery(true, true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(offset: usize) -> u32 {
        WORDS[offset / 4]
    }

    const SVC_ESR: u64 = (0x15 << 26) | (1 << 25) | 0x27;
    const DATA_ABORT_ESR: u64 = (0x24 << 26) | (1 << 25) | 0x07;
    const INSTRUCTION_ABORT_ESR: u64 = (0x20 << 26) | (1 << 25) | 0x0f;
    /// `ISS.WnR` of a data abort.
    const WNR: u64 = 1 << 6;
    const CNTPCT_ESR: u64 = (0x18 << 26) | (1 << 25) | 0x32_f821; // mrs x1, cntpct_el0

    #[test]
    fn layout_constants_point_at_the_right_instructions() {
        assert_eq!(word(EL0_SYNC), 0xa93e_2be9); // stp x9, x10, [sp, #-32]
        assert_eq!(word(AFTER_SAVE), 0xd538_5209); // mrs x9, esr_el1
        assert_eq!(word(ELR_BUMPED - 4), 0xd518_402a); // msr elr_el1, x10
        assert_eq!(word(ELR_BUMPED), 0xd61f_0160); // br x11
        assert_eq!(word(TO_HOST), STUB_RESTORE[0]);
        assert_eq!(word(PARKED - 4), HVC_TO_HOST);
        assert_ne!(
            HVC_EL0_EXCEPTION, 0,
            "hvc #0 is an SMCCC call to Hypervisor.framework"
        );
        assert_ne!(HVC_EL0_EXCEPTION, 1, "hvc #1 means an unexpected vector");
        assert_eq!(hvc(1), 0xd400_0022);
        assert_eq!(hvc(0), 0xd400_0002);
        assert_eq!(word(PARKED), ERET);
        assert_eq!(STUBS_END, FLUSH_PAGES);
        assert_eq!(FLUSH_END, LEN);
        assert_eq!(WORDS.len() * 4, LEN);
    }

    #[test]
    fn flush_entry_points_hold_the_flush_sequences() {
        // The words LLVM assembles for each sequence, written out in full.
        let by_page = |tlbi: u32| {
            [
                0xd503_3a9f, // dsb  ishst
                tlbi,
                0x9100_0400, // add  x0, x0, #1
                0xf100_0421, // subs x1, x1, #1
                0x54ff_ffa1, // b.ne -12
                0xd503_3b9f, // dsb  ish
                0xd503_3fdf, // isb
                0xd400_0062, // hvc  #3
            ]
        };
        let at = |entry: usize, len: usize| WORDS[entry / 4..entry / 4 + len].to_vec();
        assert_eq!(at(FLUSH_PAGES, 8), by_page(0xd508_8320)); // tlbi vae1is, x0
        assert_eq!(at(FLUSH_LEAVES, 8), by_page(0xd508_83a0)); // tlbi vale1is, x0
        assert_eq!(
            at(FLUSH_ASID, 5),
            [
                0xd503_3a9f, // dsb  ishst
                0xd508_8340, // tlbi aside1is, x0
                0xd503_3b9f, // dsb  ish
                0xd503_3fdf, // isb
                0xd400_0062, // hvc  #3
            ]
        );
        // The loop branches back to its own `tlbi`: imm19 at bits 23:5, in words, negative.
        for entry in [FLUSH_PAGES, FLUSH_LEAVES] {
            let b_ne = entry + 0x10;
            let imm19 = (word(b_ne) >> 5) & 0x7_ffff;
            let back = usize::try_from((1 << 19) - imm19).unwrap() * 4;
            assert_eq!(b_ne - back, entry + 4 * FLUSH_BY_PAGE_TLBI);
        }
        assert_eq!(hvc(HVC_FLUSH_DONE), 0xd400_0062);
        assert_ne!(HVC_FLUSH_DONE, 0, "hvc #0 is an SMCCC call");
        assert_ne!(HVC_FLUSH_DONE, HVC_EL0_EXCEPTION);
        assert_ne!(HVC_FLUSH_DONE, 1, "hvc #1 means an unexpected vector");
    }

    #[test]
    fn branches_land_on_their_labels() {
        // b.ne: imm19 at bits 23:5, in words.
        for at in [0x414, 0x430] {
            let imm19 = usize::try_from((word(at) >> 5) & 0x7_ffff).unwrap();
            assert_eq!(at + imm19 * 4, TO_HOST);
        }
        // adr x11, STUBS: immhi at bits 23:5, immlo at bits 30:29.
        let adr = word(0x438);
        let imm = usize::try_from(((adr >> 5) & 0x7_ffff) << 2 | ((adr >> 29) & 3)).unwrap();
        assert_eq!(0x438 + imm, STUBS);
    }

    #[test]
    fn every_unused_vector_reports_itself() {
        for vector in (0..0x800).step_by(0x80) {
            if vector != EL0_SYNC {
                assert_eq!(word(vector), HVC_1, "vector {vector:#x}");
            }
        }
    }

    #[test]
    fn each_stub_writes_its_own_register() {
        for rt in 0..32 {
            let stub = STUBS + rt * STUB_LEN;
            assert_eq!(word(stub), STUB_RESTORE[0]);
            assert_eq!(word(stub + 4), STUB_RESTORE[1]);
            assert_eq!(word(stub + 8), MRS_CNTVCT_X0 | u32::try_from(rt).unwrap());
            assert_eq!(word(stub + STUB_ERET), ERET);
        }
    }

    #[test]
    fn decodes_el0_exceptions() {
        let elr = 0x8000_0010;
        let far = 0x1234_5678;
        assert_eq!(decode_el0_exception(SVC_ESR, far, elr), Exit::Svc(0x27));
        assert_eq!(
            decode_el0_exception(DATA_ABORT_ESR, far, elr),
            Exit::DataAbort {
                address: far,
                write: false,
                syndrome: Syndrome::from_esr(DATA_ABORT_ESR),
            }
        );
        assert_eq!(
            decode_el0_exception(DATA_ABORT_ESR | WNR, far, elr),
            Exit::DataAbort {
                address: far,
                write: true,
                syndrome: Syndrome::from_esr(DATA_ABORT_ESR | WNR),
            }
        );
        // An instruction abort is its own exit, at the address that could not be fetched, even
        // with the bit that means "write" in a data abort set.
        for esr in [INSTRUCTION_ABORT_ESR, INSTRUCTION_ABORT_ESR | WNR] {
            assert_eq!(
                decode_el0_exception(esr, far, elr),
                Exit::InstructionAbort {
                    address: far,
                    syndrome: Syndrome::from_esr(esr),
                }
            );
        }
        // PC (EC 0x22) and SP (EC 0x26) alignment faults are undefined; a misaligned data access
        // is a data abort (EC 0x24, above).
        let alignment = |ec: u64| (ec << 26) | (1 << 25);
        for undefined in [
            1 << 25,
            (0x1c << 26) | (1 << 25),
            CNTPCT_ESR,
            alignment(0x22),
            alignment(0x26),
        ] {
            assert_eq!(
                decode_el0_exception(undefined, far, elr),
                Exit::Undefined {
                    pc: elr,
                    syndrome: Syndrome::from_esr(undefined),
                }
            );
        }
    }

    #[test]
    fn recovery_covers_every_instruction_on_the_el0_path() {
        let none = Recovery {
            restore_scratch: false,
            rewind_elr: false,
        };
        // Before the save: nothing to restore; an SVC must run again.
        for offset in [EL0_SYNC, EL0_SYNC + 4] {
            assert_eq!(recovery(offset, CNTPCT_ESR), Some(none));
            assert_eq!(
                recovery(offset, SVC_ESR),
                Some(Recovery {
                    restore_scratch: false,
                    rewind_elr: true
                })
            );
        }
        // Dispatch and the host path: x9-x11 are borrowed.
        for offset in (AFTER_SAVE..ELR_BUMPED).chain(TO_HOST..PARKED).step_by(4) {
            assert_eq!(
                recovery(offset, CNTPCT_ESR),
                Some(Recovery {
                    restore_scratch: true,
                    rewind_elr: false
                }),
                "{offset:#x}"
            );
            assert_eq!(
                recovery(offset, SVC_ESR),
                Some(Recovery {
                    restore_scratch: true,
                    rewind_elr: true
                }),
                "{offset:#x}"
            );
        }
        // Counter path after ELR_EL1 moved, and inside a stub before it wrote Rt.
        let undo = Some(Recovery {
            restore_scratch: true,
            rewind_elr: true,
        });
        assert_eq!(recovery(ELR_BUMPED, CNTPCT_ESR), undo);
        for rt in 0..32 {
            let stub = STUBS + rt * STUB_LEN;
            for offset in [stub, stub + 4, stub + 8] {
                assert_eq!(recovery(offset, CNTPCT_ESR), undo, "{offset:#x}");
            }
            assert_eq!(recovery(stub + STUB_ERET, CNTPCT_ESR), Some(none));
        }
        assert_eq!(recovery(PARKED, SVC_ESR), Some(none));
        // Unused vectors and anything outside the shim.
        for offset in [
            0,
            0x200,
            0x480,
            PARKED + 4,
            STUBS - 4,
            // The flush entry points are never interrupted on the way to or from EL0.
            FLUSH_PAGES,
            FLUSH_LEAVES,
            FLUSH_ASID,
            FLUSH_END,
            usize::MAX,
        ] {
            assert_eq!(recovery(offset, SVC_ESR), None, "{offset:#x}");
        }
    }
}
