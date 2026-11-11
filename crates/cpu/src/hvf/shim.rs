//! The guest-side EL1 shim: an exception vector table that turns EL0 exceptions into host exits.
//!
//! Guest code runs at EL0. Every exception it takes lands in this table at EL1, which either
//! handles it in the guest (a trapped `mrs xN, cntpct_el0`, answered from the virtual counter) or
//! hands it to the host with `hvc #0`. The host then reads `ESR_EL1`, `FAR_EL1` and `ELR_EL1`
//! directly, so the shim passes nothing in general-purpose registers. Between runs a vCPU is
//! *parked* on the shim's `eret`: its EL0 state lives in `ELR_EL1`/`SPSR_EL1`, and every run starts
//! by returning to EL0.
//!
//! The shim borrows x9-x11 while it works and saves them in a per-vCPU scratch area just below
//! `SP_EL1`. If the host interrupts a vCPU part-way through the shim, [`recovery`] says how to
//! undo the partial work so the EL0 state is exact.
//!
//! Ported from `tools/hvprobe/src/guest.rs` (vector table, `hvc` forwarding, in-guest counter
//! emulation); this version preserves every EL0 register and emulates `cntpct_el0` for any `Rt`.
//! The words were assembled with LLVM from the assembly beside them. Exception classes and the
//! ISS layout are from the Arm Architecture Reference Manual (DDI 0487), `ESR_EL1`.

use crate::Exit;

/// Size of the shim in bytes.
pub(super) const LEN: usize = 0xa00;

/// Offset of the "lower EL, AArch64, synchronous" vector (Arm ARM, `VBAR_EL1` vector offsets).
pub(super) const EL0_SYNC: usize = 0x400;
/// First instruction after the shim has saved x9-x11.
pub(super) const AFTER_SAVE: usize = 0x408;
/// First instruction after the counter path has advanced `ELR_EL1`.
pub(super) const ELR_BUMPED: usize = 0x44c;
/// Start of the path that restores x9-x11 and exits to the host.
pub(super) const TO_HOST: usize = 0x450;
/// The `eret` a vCPU is parked on between runs (right after `hvc #0`).
pub(super) const PARKED: usize = 0x45c;
/// Start of the 32 counter stubs, one per `Rt` (31 is `xzr`).
pub(super) const STUBS: usize = 0x800;
/// Size of one counter stub.
pub(super) const STUB_LEN: usize = 16;
/// Offset of the `eret` inside a counter stub (after it has written `Rt`).
const STUB_ERET: usize = 12;
/// End of the counter stubs.
const STUBS_END: usize = STUBS + 32 * STUB_LEN;

/// How far below `SP_EL1` the shim saves x9 (x10 and x11 follow at +8 and +16).
pub(super) const SCRATCH_BELOW_SP: u64 = 32;

/// `hvc` immediate for "EL0 took an exception the host must handle".
pub(super) const HVC_EL0_EXCEPTION: u64 = 0;

/// `CPSR` of a parked vCPU: `EL1h` with D, A, I and F masked (`tools/hvprobe/src/probe.rs`).
pub(super) const CPSR_EL1H_MASKED: u64 = 0x3c5;
/// `SPSR_EL1` bits for `EL0t` with D, A, I and F masked (`tools/hvprobe/src/probe.rs`).
pub(super) const SPSR_EL0T_MASKED: u64 = 0x3c0;
/// `PSTATE.M[3:0]` values (Arm ARM, `SPSR_EL1`).
pub(super) const MODE_MASK: u64 = 0xf;
pub(super) const MODE_EL0T: u64 = 0x0;
pub(super) const MODE_EL1H: u64 = 0x5;

// Exception classes, `ESR_ELx.EC` (Arm ARM).
pub(super) const EC_SVC64: u64 = 0x15;
pub(super) const EC_HVC64: u64 = 0x16;
const EC_INSTRUCTION_ABORT_LOWER: u64 = 0x20;
const EC_DATA_ABORT_LOWER: u64 = 0x24;
/// `ESR_ELx.ISS.WnR` for data aborts.
const ISS_WNR: u64 = 1 << 6;

const HVC_1: u32 = 0xd400_0022; // hvc #1: an exception the shim doesn't expect
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
    0xd400_0002, // 0x458  hvc  #0                          HVC_EL0_EXCEPTION
    ERET,        // 0x45c  eret                             PARKED
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
    words
}

/// `ESR_ELx.EC`.
pub(super) const fn exception_class(esr: u64) -> u64 {
    (esr >> 26) & 0x3f
}

/// Turns an exception EL0 took (as the shim reported it) into an [`Exit`].
///
/// SVCs carry their immediate, and `ELR_EL1` already points past the `svc`. Instruction and data
/// aborts report `FAR_EL1`. Anything else (UNDEFINED, a failed `AUT*`, a trapped system register
/// access the shim doesn't emulate, alignment faults) is [`Exit::Undefined`] at `ELR_EL1`.
pub(super) fn decode_el0_exception(esr: u64, far: u64, elr: u64) -> Exit {
    match exception_class(esr) {
        EC_SVC64 => {
            let [lo, hi, ..] = esr.to_le_bytes();
            Exit::Svc(u16::from_le_bytes([lo, hi]))
        }
        EC_DATA_ABORT_LOWER => Exit::DataAbort {
            address: far,
            write: esr & ISS_WNR != 0,
        },
        EC_INSTRUCTION_ABORT_LOWER => Exit::DataAbort {
            address: far,
            write: false,
        },
        _ => Exit::Undefined { pc: elr },
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
    let svc = exception_class(esr) == EC_SVC64;
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

    const SVC_ESR: u64 = (EC_SVC64 << 26) | (1 << 25) | 0x27;
    const CNTPCT_ESR: u64 = (0x18 << 26) | (1 << 25) | 0x32_f821; // mrs x1, cntpct_el0

    #[test]
    fn layout_constants_point_at_the_right_instructions() {
        assert_eq!(word(EL0_SYNC), 0xa93e_2be9); // stp x9, x10, [sp, #-32]
        assert_eq!(word(AFTER_SAVE), 0xd538_5209); // mrs x9, esr_el1
        assert_eq!(word(ELR_BUMPED - 4), 0xd518_402a); // msr elr_el1, x10
        assert_eq!(word(ELR_BUMPED), 0xd61f_0160); // br x11
        assert_eq!(word(TO_HOST), STUB_RESTORE[0]);
        assert_eq!(
            word(PARKED - 4),
            0xd400_0002 | (u32::try_from(HVC_EL0_EXCEPTION).unwrap() << 5)
        );
        assert_eq!(word(PARKED), ERET);
        assert_eq!(STUBS_END, LEN);
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
        let data_abort = (EC_DATA_ABORT_LOWER << 26) | (1 << 25) | 0x07;
        assert_eq!(
            decode_el0_exception(data_abort, far, elr),
            Exit::DataAbort {
                address: far,
                write: false
            }
        );
        assert_eq!(
            decode_el0_exception(data_abort | ISS_WNR, far, elr),
            Exit::DataAbort {
                address: far,
                write: true
            }
        );
        let instruction_abort = (EC_INSTRUCTION_ABORT_LOWER << 26) | (1 << 25) | ISS_WNR;
        assert_eq!(
            decode_el0_exception(instruction_abort, far, elr),
            Exit::DataAbort {
                address: far,
                write: false
            }
        );
        for undefined in [1 << 25, (0x1c << 26) | (1 << 25), CNTPCT_ESR] {
            assert_eq!(
                decode_el0_exception(undefined, far, elr),
                Exit::Undefined { pc: elr }
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
            STUBS_END,
            usize::MAX,
        ] {
            assert_eq!(recovery(offset, SVC_ESR), None, "{offset:#x}");
        }
    }
}
