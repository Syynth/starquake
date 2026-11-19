//! Tests of slice 6g, the attribute and system-value group (`op::attribute`): NAK's expected
//! text for `ISBERD` and `ISBEWR`, one hand-built word per form and per modifier bit, every
//! `IPA` mode and special register name, attribute offsets at the edges of their fields, the
//! values and bits the group refuses, the divergences from PR #4588's table, and generated round
//! trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::attribute::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A hand-built word's opcode and fields, and its expected text.
type Case = (u128, &'static [(u32, u128)], &'static str);
/// [`Case`] with fields built at run time.
type BuiltCase = (u128, Vec<(u32, u128)>, &'static str);

/// NAK's `test_isberd`, ported as a fixture table (`nvdisasm_tests.rs` @ 52238762): `r1` from
/// `[r2]` and `[r2+0x42]`, with every output, access type, skew and memory type, and the text
/// NAK checks against `nvdisasm` (SM86 has every case: the loops skip only SM below 75 and 86).
#[test]
fn isberd_text_matches_nak_nvdisasm_cases() {
    let mem_types = [
        (IsbeMemType::U8, ""),
        (IsbeMemType::U16, ".u16"),
        (IsbeMemType::B32, ".32"),
    ];
    let outputs = [(false, ""), (true, ".o")];
    let skews = [(false, ""), (true, ".skew")];
    let accesses = [
        (IsbeAccess::Map, ""),
        (IsbeAccess::Patch, ".patch"),
        (IsbeAccess::Primitive, ".prim"),
        (IsbeAccess::Attribute, ".attr"),
    ];
    let mut checked = 0;
    for (output, output_str) in outputs {
        for (access, access_str) in accesses {
            for (skew, skew_str) in skews {
                for (mem_type, mem_str) in mem_types {
                    for imm_offset in [0, 0x42] {
                        let instr = Instr::new(Op::Isberd(OpIsberd {
                            dst: r(1),
                            offset: r(2),
                            imm_offset,
                            mem_type,
                            access,
                            output,
                            skew,
                        }));
                        let addr = if imm_offset == 0 {
                            "[r2]".to_string()
                        } else {
                            format!("[r2+0x{imm_offset:x}]")
                        };
                        let expected = format!(
                            "isberd{output_str}{access_str}{skew_str}{mem_str} r1, {addr};"
                        );
                        assert_eq!(round_text(instr), expected);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 96);
}

/// NAK's `test_isbewr`, ported: `[r2]` and `[r2+0x42]` from `r1`, always `.o`, with the two
/// access types `ISBEWR` writes, skew and every memory type.
#[test]
fn isbewr_text_matches_nak_nvdisasm_cases() {
    let mem_types = [
        (IsbeMemType::U8, ""),
        (IsbeMemType::U16, ".u16"),
        (IsbeMemType::B32, ".32"),
    ];
    let skews = [(false, ""), (true, ".skew")];
    let accesses = [(IsbeAccess::Map, ""), (IsbeAccess::Attribute, ".attr")];
    let mut checked = 0;
    for (access, access_str) in accesses {
        for (skew, skew_str) in skews {
            for (mem_type, mem_str) in mem_types {
                for imm_offset in [0, 0x42] {
                    let instr = Instr::new(Op::Isbewr(OpIsbewr {
                        offset: r(2),
                        data: r(1),
                        imm_offset,
                        mem_type,
                        access,
                        output: true,
                        skew,
                    }));
                    let addr = if imm_offset == 0 {
                        "[r2]".to_string()
                    } else {
                        format!("[r2+0x{imm_offset:x}]")
                    };
                    let expected = format!("isbewr.o{access_str}{skew_str}{mem_str} {addr}, r1;");
                    assert_eq!(round_text(instr), expected);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 24);
}

/// One hand-built word per form of the group, every field at a distinct value, and its text
/// (unconfirmed but for `ISBERD` and `ISBEWR`: `print`'s module docs).
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = [
        // ALD.o.p.phys.128 r4, a[r7][0x84+r9]: dst 4, offset 9 << 24, vtx 7 << 32,
        // addr 0x84 << 40, comps - 1 = 3 << 74, patch 76, phys 77, output 79.
        (
            word(
                0x321,
                &[
                    (16, 4),
                    (24, 9),
                    (32, 7),
                    (40, 0x84),
                    (74, 3),
                    (76, 1),
                    (77, 1),
                    (79, 1),
                ],
            ),
            "ald.o.p.phys.128 r4, a[r7][0x84+r9];",
        ),
        // AST.96 a[r5][0x90+r3], r2: offset 3, data 2 << 32, addr 0x90, vtx 5 << 64, comps 2.
        (
            word(0x322, &[(24, 3), (32, 2), (40, 0x90), (64, 5), (74, 2)]),
            "ast.96 a[r5][0x90+r3], r2;",
        ),
        // S2R r2, SR_TID.X: dst 2, index 33 << 72.
        (word(0x919, &[(16, 2), (72, 33)]), "s2r r2, sr_tid.x;"),
        // S2UR ur6, SR_CTAID.Z: uniform dst 6 at bit 16, index 39.
        (word(0x9c3, &[(16, 6), (72, 39)]), "s2ur ur6, sr_ctaid.z;"),
        // CS2R r8, SR_CLOCKLO (the pair): dst 8, index 80, .64 at bit 80.
        (
            word(0x805, &[(16, 8), (72, 80), (80, 1)]),
            "cs2r r8, sr_clocklo;",
        ),
        // AL2P.o.64 r1, a[0x84+r3]: dst 1, offset 3, addr 0x84, comps - 1 = 1, output 79.
        (
            word(0x920, &[(16, 1), (24, 3), (40, 0x84), (74, 1), (79, 1)]),
            "al2p.o.64 r1, a[0x84+r3];",
        ),
        // IPA.state.offset p2, r1, a[0x88], r6: dst 1, offset 6 << 32, word 0x22 << 64 (byte
        // 0x88), loc 2 << 76, freq 2 << 78, predicate 2 << 81.
        (
            word(
                0x326,
                &[(16, 1), (32, 6), (64, 0x22), (76, 2), (78, 2), (81, 2)],
            ),
            "ipa.state.offset p2, r1, a[0x88], r6;",
        ),
        // LDTRAM.C r3, a[0x8c+ur5]: dst 3, ureg 5 << 24, word 0x23 << 64, .C at 72, bit 91.
        (
            word(0x3ad, &[(16, 3), (24, 5), (64, 0x23), (72, 1), (91, 1)]),
            "ldtram.c r3, a[0x8c+ur5];",
        ),
        // PIXLD.MY_INDEX p4, r9: dst 9, value 3 << 78, predicate 4 << 81.
        (
            word(0x925, &[(16, 9), (78, 3), (81, 4)]),
            "pixld.my_index p4, r9;",
        ),
        // OUT.EMIT r1, r2, r3 (form 1): dst 1, handle 2 << 24, stream 3 << 32, type 1 << 78.
        (
            word(0x324, &[(16, 1), (24, 2), (32, 3), (78, 1)]),
            "out.emit r1, r2, r3;",
        ),
        // OUT.CUT r1, r2, 0x12345678 (form 4): the stream at bits 32..64, type 2.
        (
            word(0x924, &[(16, 1), (24, 2), (32, 0x1234_5678), (78, 2)]),
            "out.cut r1, r2, 0x12345678;",
        ),
        // OUT.EMIT_THEN_CUT r1, r2, c[0x3][0x10] (form 5): offset 0x10 << 38, index 3 << 54.
        (
            word(0xb24, &[(16, 1), (24, 2), (38, 0x10), (54, 3), (78, 3)]),
            "out.emit_then_cut r1, r2, c[0x3][0x10];",
        ),
        // OUT.FINAL rz, r2, ur9 (form 6): dst RZ, ur 9 at bit 32, bit 91, type 0.
        (
            word(0xd24, &[(16, 0xff), (24, 2), (32, 9), (91, 1)]),
            "out.final rz, r2, ur9;",
        ),
        // ISBERD.o.prim.skew.u16 r1, [r2+0x42]: dst 1, offset 2, imm 0x42 << 40, mem 1 << 74,
        // access 2 << 76, skew 78, output 79.
        (
            word(
                0x923,
                &[
                    (16, 1),
                    (24, 2),
                    (40, 0x42),
                    (74, 1),
                    (76, 2),
                    (78, 1),
                    (79, 1),
                ],
            ),
            "isberd.o.prim.skew.u16 r1, [r2+0x42];",
        ),
        // ISBEWR.o.attr.32 [r2+0x42], r1: offset 2, data 1 << 32, imm 0x42, mem 2, access 3.
        (
            word(
                0x927,
                &[(24, 2), (32, 1), (40, 0x42), (74, 2), (76, 3), (79, 1)],
            ),
            "isbewr.o.attr.32 [r2+0x42], r1;",
        ),
    ];
    // One case per form of the group, in form order.
    let opcodes: Vec<u64> = cases.iter().map(|(word, _)| word.field(OPCODE)).collect();
    let forms: Vec<u64> = attribute::FORMS
        .iter()
        .map(|form| u64::from(form.opcode))
        .collect();
    assert_eq!(opcodes, forms);
    for (word, expected) in cases {
        assert_eq!(text(word), expected, "{word:x?}");
    }
}

/// Each modifier bit and each value of each mode field, set alone in a hand-built word, changes
/// the text the way its field says: the plain word of each instruction, then one bit or value
/// at a time.
#[test]
fn hand_built_words_pin_each_modifier_bit() {
    // ALD r1, a[0x80]: offset and vertex RZ.
    let ald = [(16, 1), (24, 0xff), (32, 0xff), (40, 0x80)];
    // AST a[0x70], r0.
    let ast = [(24, 0xff), (32, 0), (40, 0x70), (64, 0xff)];
    // AL2P r1, a[0x80].
    let al2p = [(16, 1), (24, 0xff), (40, 0x80)];
    // IPA.pass r1, a[0x80]: offset RZ, predicate PT.
    let ipa = [(16, 1), (32, 0xff), (64, 0x20), (81, 7)];
    // LDTRAM.AB r1, a[0x80]: URZ, bit 91.
    let ldtram = [(16, 1), (24, 63), (64, 0x20), (91, 1)];
    // CS2R.32 r2, SR_LANEID.
    let cs2r = [(16, 2)];
    // PIXLD.MSCOUNT r1: predicate PT.
    let pixld = [(16, 1), (81, 7)];
    // OUT.FINAL r1, r2, r3.
    let out = [(16, 1), (24, 2), (32, 3)];
    // ISBERD r1, [r2].
    let isberd = [(16, 1), (24, 2)];
    // ISBEWR [r2], r1.
    let isbewr = [(24, 2), (32, 1)];
    let with = |base: &[(u32, u128)], extra: &[(u32, u128)]| -> Vec<(u32, u128)> {
        base.iter().chain(extra).copied().collect()
    };
    let cases: Vec<BuiltCase> = vec![
        (0x321, with(&ald, &[]), "ald r1, a[0x80];"),
        (0x321, with(&ald, &[(79, 1)]), "ald.o r1, a[0x80];"),
        (0x321, with(&ald, &[(76, 1)]), "ald.p r1, a[0x80];"),
        (0x321, with(&ald, &[(77, 1)]), "ald.phys r1, a[0x80];"),
        (0x321, with(&ald, &[(74, 1)]), "ald.64 r1, a[0x80];"),
        (0x321, with(&ald, &[(74, 2)]), "ald.96 r1, a[0x80];"),
        (0x321, with(&ald, &[(74, 3)]), "ald.128 r1, a[0x80];"),
        (0x322, with(&ast, &[]), "ast a[0x70], r0;"),
        (0x322, with(&ast, &[(76, 1)]), "ast.p a[0x70], r0;"),
        (0x322, with(&ast, &[(77, 1)]), "ast.phys a[0x70], r0;"),
        (0x322, with(&ast, &[(74, 1)]), "ast.64 a[0x70], r0;"),
        (0x920, with(&al2p, &[]), "al2p r1, a[0x80];"),
        (0x920, with(&al2p, &[(79, 1)]), "al2p.o r1, a[0x80];"),
        (0x920, with(&al2p, &[(74, 3)]), "al2p.128 r1, a[0x80];"),
        (0x326, with(&ipa, &[]), "ipa.pass r1, a[0x80];"),
        (
            0x326,
            with(&ipa, &[(76, 1)]),
            "ipa.pass.centroid r1, a[0x80];",
        ),
        (0x326, with(&ipa, &[(78, 1)]), "ipa.constant r1, a[0x80];"),
        (0x3ad, with(&ldtram, &[]), "ldtram.ab r1, a[0x80];"),
        (0x3ad, with(&ldtram, &[(72, 1)]), "ldtram.c r1, a[0x80];"),
        (0x805, with(&cs2r, &[]), "cs2r.32 r2, sr_laneid;"),
        (0x805, with(&cs2r, &[(80, 1)]), "cs2r r2, sr_laneid;"),
        (0x925, with(&pixld, &[]), "pixld.mscount r1;"),
        (0x925, with(&pixld, &[(78, 1)]), "pixld.covmask r1;"),
        (0x925, with(&pixld, &[(79, 1)]), "pixld.centroid_offset r1;"),
        (0x925, with(&pixld, &[(80, 1)]), "pixld.inner_coverage r1;"),
        (0x324, with(&out, &[]), "out.final r1, r2, r3;"),
        (0x324, with(&out, &[(78, 1)]), "out.emit r1, r2, r3;"),
        (0x324, with(&out, &[(79, 1)]), "out.cut r1, r2, r3;"),
        (0x923, with(&isberd, &[]), "isberd r1, [r2];"),
        (0x923, with(&isberd, &[(79, 1)]), "isberd.o r1, [r2];"),
        (0x923, with(&isberd, &[(78, 1)]), "isberd.skew r1, [r2];"),
        (0x923, with(&isberd, &[(76, 1)]), "isberd.patch r1, [r2];"),
        (0x923, with(&isberd, &[(77, 1)]), "isberd.prim r1, [r2];"),
        (0x923, with(&isberd, &[(74, 1)]), "isberd.u16 r1, [r2];"),
        (0x923, with(&isberd, &[(75, 1)]), "isberd.32 r1, [r2];"),
        (0x923, with(&isberd, &[(40, 1)]), "isberd r1, [r2+0x1];"),
        (0x923, with(&isberd, &[(55, 1)]), "isberd r1, [r2+0x8000];"),
        (0x927, with(&isbewr, &[]), "isbewr [r2], r1;"),
        (0x927, with(&isbewr, &[(79, 1)]), "isbewr.o [r2], r1;"),
        (0x927, with(&isbewr, &[(78, 1)]), "isbewr.skew [r2], r1;"),
        (0x927, with(&isbewr, &[(76, 3)]), "isbewr.attr [r2], r1;"),
        (0x927, with(&isbewr, &[(74, 1)]), "isbewr.u16 [r2], r1;"),
        (0x927, with(&isbewr, &[(75, 1)]), "isbewr.32 [r2], r1;"),
    ];
    for (opcode, fields, expected) in cases {
        assert_eq!(
            text(word(opcode, &fields)),
            expected,
            "{opcode:#x} {fields:?}"
        );
    }
}

/// The registers and predicates of the group, each moved off its default alone: a register in
/// place of `RZ`, a predicate in place of `PT`, a uniform register in place of `URZ`, and each
/// register field at a value no other field has.
#[test]
fn hand_built_words_pin_each_register_field() {
    let cases: [Case; 12] = [
        // ALD: vertex 7, offset 9, destination 4.
        (
            0x321,
            &[(16, 4), (24, 0xff), (32, 7), (40, 0x80)],
            "ald r4, a[r7][0x80];",
        ),
        (
            0x321,
            &[(16, 4), (24, 9), (32, 0xff), (40, 0x80)],
            "ald r4, a[0x80+r9];",
        ),
        // AST: vertex 5 at bit 64, data 3.
        (
            0x322,
            &[(24, 0xff), (32, 3), (40, 0x70), (64, 5)],
            "ast a[r5][0x70], r3;",
        ),
        // AL2P: offset 6.
        (
            0x920,
            &[(16, 1), (24, 6), (40, 0x80)],
            "al2p r1, a[0x80+r6];",
        ),
        // IPA: predicate p0, then an offset register r3 with the default location.
        (
            0x326,
            &[(16, 1), (32, 0xff), (64, 0x20), (81, 0)],
            "ipa.pass p0, r1, a[0x80];",
        ),
        (
            0x326,
            &[(16, 1), (32, 3), (64, 0x20), (81, 7)],
            "ipa.pass r1, a[0x80], r3;",
        ),
        // LDTRAM: uniform register 0.
        (
            0x3ad,
            &[(16, 1), (24, 0), (64, 0x20), (91, 1)],
            "ldtram.ab r1, a[0x80+ur0];",
        ),
        // PIXLD: predicate p6.
        (0x925, &[(16, 1), (81, 6)], "pixld.mscount p6, r1;"),
        // S2UR: URZ.
        (0x9c3, &[(16, 63), (72, 0)], "s2ur urz, sr_laneid;"),
        // OUT.FINAL with the destination and stream NAK writes: RZ.
        (
            0x324,
            &[(16, 0xff), (24, 5), (32, 0xff)],
            "out.final rz, r5, rz;",
        ),
        // ISBERD: offset RZ.
        (0x923, &[(16, 1), (24, 0xff)], "isberd r1, [rz];"),
        // ISBEWR: data RZ, offset 7.
        (0x927, &[(24, 7), (32, 0xff)], "isbewr [r7], rz;"),
    ];
    for (opcode, fields, expected) in cases {
        assert_eq!(
            text(word(opcode, fields)),
            expected,
            "{opcode:#x} {fields:?}"
        );
    }
}

/// Every `IPA` mode, location by frequency, in a hand-built word: the offset location prints
/// the offset register even when it is `RZ`.
#[test]
fn every_ipa_mode_decodes() {
    let locs = [
        (0_u64, InterpLoc::Default, ""),
        (1, InterpLoc::Centroid, ".centroid"),
        (2, InterpLoc::Offset, ".offset"),
    ];
    let freqs = [
        (0_u64, InterpFreq::Pass, ".pass"),
        (1, InterpFreq::Constant, ".constant"),
        (2, InterpFreq::State, ".state"),
    ];
    let mut checked = 0;
    for (loc_bits, loc, loc_str) in locs {
        for (freq_bits, freq, freq_str) in freqs {
            let word = word(
                0x326,
                &[
                    (16, 1),
                    (32, 0xff),
                    (64, 0x20),
                    (76, u128::from(loc_bits)),
                    (78, u128::from(freq_bits)),
                    (81, 7),
                ],
            );
            let offset = if loc == InterpLoc::Offset { ", rz" } else { "" };
            assert_eq!(
                text(word),
                format!("ipa{freq_str}{loc_str} r1, a[0x80]{offset};")
            );
            assert_eq!(
                decode(word).map(|instr| instr.op),
                Ok(Op::Ipa(OpIpa {
                    dst: r(1),
                    addr: 0x80,
                    freq,
                    loc,
                    offset: Gpr::RZ,
                    pred: Pred::PT,
                }))
            );
            assert_eq!((loc.bits(), freq.bits()), (loc_bits, freq_bits));
            checked += 1;
        }
    }
    assert_eq!(checked, InterpLoc::ALL.len() * InterpFreq::ALL.len());
}

/// The special register names, written out from Eden's `SpecialRegister` (`eden/src/
/// shader_recompiler/frontend/maxwell/translate/impl/move_special_register.cpp` @ cf654052),
/// independently of the decoder's table: every index without a name prints as `sr` and its
/// number, 255 as `srz`.
const EDEN_NAMES: &[(u8, &str)] = &[
    (0, "sr_laneid"),
    (1, "sr_clock"),
    (2, "sr_virtcfg"),
    (3, "sr_virtid"),
    (4, "sr_pm0"),
    (5, "sr_pm1"),
    (6, "sr_pm2"),
    (7, "sr_pm3"),
    (8, "sr_pm4"),
    (9, "sr_pm5"),
    (10, "sr_pm6"),
    (11, "sr_pm7"),
    (15, "sr_ordering_ticket"),
    (16, "sr_prim_type"),
    (17, "sr_invocation_id"),
    (18, "sr_y_direction"),
    (19, "sr_thread_kill"),
    (20, "sm_shader_type"),
    (21, "sr_directcbewriteaddresslow"),
    (22, "sr_directcbewriteaddresshigh"),
    (23, "sr_directcbewriteenable"),
    (24, "sr_machine_id_0"),
    (25, "sr_machine_id_1"),
    (26, "sr_machine_id_2"),
    (27, "sr_machine_id_3"),
    (28, "sr_affinity"),
    (29, "sr_invocation_info"),
    (30, "sr_wscalefactor_xy"),
    (31, "sr_wscalefactor_z"),
    (32, "sr_tid"),
    (33, "sr_tid.x"),
    (34, "sr_tid.y"),
    (35, "sr_tid.z"),
    (36, "sr_cta_param"),
    (37, "sr_ctaid.x"),
    (38, "sr_ctaid.y"),
    (39, "sr_ctaid.z"),
    (40, "sr_ntid"),
    (41, "sr_cirqueueincrminusone"),
    (42, "sr_nlatc"),
    (44, "sr_sm_spa_version"),
    (45, "sr_multipassshaderinfo"),
    (46, "sr_lwinhi"),
    (47, "sr_swinhi"),
    (48, "sr_swinlo"),
    (49, "sr_swinsz"),
    (50, "sr_smemsz"),
    (51, "sr_smembanks"),
    (52, "sr_lwinlo"),
    (53, "sr_lwinsz"),
    (54, "sr_lmemlosz"),
    (55, "sr_lmemhioff"),
    (56, "sr_eqmask"),
    (57, "sr_ltmask"),
    (58, "sr_lemask"),
    (59, "sr_gtmask"),
    (60, "sr_gemask"),
    (61, "sr_regalloc"),
    (62, "sr_barrieralloc"),
    (64, "sr_globalerrorstatus"),
    (66, "sr_warperrorstatus"),
    (67, "sr_warperrorstatusclear"),
    (72, "sr_pm_hi0"),
    (73, "sr_pm_hi1"),
    (74, "sr_pm_hi2"),
    (75, "sr_pm_hi3"),
    (76, "sr_pm_hi4"),
    (77, "sr_pm_hi5"),
    (78, "sr_pm_hi6"),
    (79, "sr_pm_hi7"),
    (80, "sr_clocklo"),
    (81, "sr_clockhi"),
    (82, "sr_globaltimerlo"),
    (83, "sr_globaltimerhi"),
    (96, "sr_hwtaskid"),
    (97, "sr_circularqueueentryindex"),
    (98, "sr_circularqueueentryaddresslow"),
    (99, "sr_circularqueueentryaddresshigh"),
    (255, "srz"),
];

/// The expected name of special register `index`.
fn expected_name(index: u8) -> String {
    EDEN_NAMES
        .iter()
        .find(|&&(named, _)| named == index)
        .map_or_else(|| format!("sr{index}"), |&(_, name)| name.to_string())
}

/// Every special register index, 0 to 255, in a hand-built word of `S2R`, `S2UR` and `CS2R`
/// (both widths), prints its name.
#[test]
fn every_special_register_prints_its_name() {
    for index in 0..=u8::MAX {
        let name = expected_name(index);
        let sr = u128::from(index);
        assert_eq!(
            text(word(0x919, &[(16, 3), (72, sr)])),
            format!("s2r r3, {name};")
        );
        assert_eq!(
            text(word(0x9c3, &[(16, 3), (72, sr)])),
            format!("s2ur ur3, {name};")
        );
        assert_eq!(
            text(word(0x805, &[(16, 3), (72, sr)])),
            format!("cs2r.32 r3, {name};")
        );
        assert_eq!(
            text(word(0x805, &[(16, 3), (72, sr), (80, 1)])),
            format!("cs2r r3, {name};")
        );
    }
    let constants = [
        (SpecialReg::LANE_ID, 0),
        (SpecialReg::TID_X, 33),
        (SpecialReg::TID_Y, 34),
        (SpecialReg::TID_Z, 35),
        (SpecialReg::CTAID_X, 37),
        (SpecialReg::CTAID_Y, 38),
        (SpecialReg::CTAID_Z, 39),
        (SpecialReg::CLOCK_LO, 80),
        (SpecialReg::SRZ, 255),
    ];
    for (sr, index) in constants {
        assert_eq!(sr, SpecialReg(index));
        assert_eq!(sr.to_string(), expected_name(index));
    }
    // Unnamed indices have no name; the first past Eden's enum is unnamed too.
    assert_eq!(SpecialReg(12).name(), None);
    assert_eq!(SpecialReg(100).name(), None);
    assert_eq!(SpecialReg(254).name(), None);
}

/// The attribute offsets at both edges of their fields, in hand-built words: the ten-bit byte
/// address of `ALD`, `AST` and `AL2P` (0 and `0x3ff`), the eight-bit word address of `IPA` and
/// `LDTRAM` (bytes 0 and `0x3fc`) and the sixteen-bit immediate of `ISBERD` and `ISBEWR` (0 and
/// `0xffff`).
#[test]
fn attribute_offsets_at_both_edges() {
    let cases: [Case; 12] = [
        (
            0x321,
            &[(16, 1), (24, 0xff), (32, 0xff), (40, 0)],
            "ald r1, a[0x0];",
        ),
        (
            0x321,
            &[(16, 1), (24, 0xff), (32, 0xff), (40, 0x3ff)],
            "ald r1, a[0x3ff];",
        ),
        (
            0x322,
            &[(24, 0xff), (32, 1), (40, 0), (64, 0xff)],
            "ast a[0x0], r1;",
        ),
        (
            0x322,
            &[(24, 0xff), (32, 1), (40, 0x3ff), (64, 0xff)],
            "ast a[0x3ff], r1;",
        ),
        (0x920, &[(16, 1), (24, 0xff), (40, 0)], "al2p r1, a[0x0];"),
        (
            0x920,
            &[(16, 1), (24, 0xff), (40, 0x3ff)],
            "al2p r1, a[0x3ff];",
        ),
        (
            0x326,
            &[(16, 1), (32, 0xff), (64, 0), (81, 7)],
            "ipa.pass r1, a[0x0];",
        ),
        (
            0x326,
            &[(16, 1), (32, 0xff), (64, 0xff), (81, 7)],
            "ipa.pass r1, a[0x3fc];",
        ),
        (
            0x3ad,
            &[(16, 1), (24, 63), (64, 0), (91, 1)],
            "ldtram.ab r1, a[0x0];",
        ),
        (
            0x3ad,
            &[(16, 1), (24, 63), (64, 0xff), (91, 1)],
            "ldtram.ab r1, a[0x3fc];",
        ),
        (
            0x923,
            &[(16, 1), (24, 2), (40, 0xffff)],
            "isberd r1, [r2+0xffff];",
        ),
        (
            0x927,
            &[(24, 2), (32, 1), (40, 0xffff)],
            "isbewr [r2+0xffff], r1;",
        ),
    ];
    for (opcode, fields, expected) in cases {
        assert_eq!(text(word(opcode, fields)), expected, "{opcode:#x}");
    }
}

/// The encoder at and one past the edge of each attribute offset field.
#[test]
fn encoder_refuses_offsets_past_their_fields() {
    // The fields' last values, from the shipping constants.
    let last_addr = (1_u16 << ATTR_ADDR.width) - 1;
    let last_word = ((1_u16 << ATTR_WORD.width) - 1) * ATTR_WORD_BYTES;
    assert_eq!((last_addr, last_word), (0x3ff, 0x3fc));

    let al2p = |addr| {
        assemble(&Instr::new(Op::AL2P(OpAL2P {
            dst: r(1),
            offset: Gpr::RZ,
            addr,
            comps: 1,
            output: false,
        })))
    };
    assert!(al2p(last_addr).is_ok());
    assert_eq!(
        al2p(last_addr + 1),
        Err(EncodeError::TooWide {
            what: "attribute address",
            value: u64::from(last_addr) + 1,
            bits: ATTR_ADDR.width
        })
    );
    let ipa = |addr| {
        assemble(&Instr::new(Op::Ipa(OpIpa {
            dst: r(1),
            addr,
            freq: InterpFreq::Pass,
            loc: InterpLoc::Default,
            offset: Gpr::RZ,
            pred: Pred::PT,
        })))
    };
    assert!(ipa(last_word).is_ok());
    assert_eq!(
        ipa(last_word + ATTR_WORD_BYTES),
        Err(EncodeError::TooWide {
            what: "attribute word",
            value: 1 << ATTR_WORD.width,
            bits: ATTR_WORD.width
        })
    );
    // One byte either side of a word boundary is misaligned.
    for addr in [last_word - 1, last_word + 1, 1] {
        assert_eq!(
            ipa(addr),
            Err(EncodeError::AttrAlign {
                addr,
                align: ATTR_WORD_BYTES
            })
        );
    }
    let ldtram = |addr| {
        assemble(&Instr::new(Op::LdTram(OpLdTram {
            dst: r(1),
            addr,
            use_c: false,
            ureg: UniformReg::URZ,
        })))
    };
    assert!(ldtram(last_word).is_ok());
    assert_eq!(
        ldtram(last_word + ATTR_WORD_BYTES),
        Err(EncodeError::TooWide {
            what: "attribute word",
            value: 1 << ATTR_WORD.width,
            bits: ATTR_WORD.width
        })
    );
    assert_eq!(
        ldtram(2).map_err(|error| error.to_string()),
        Err("attribute address 0x2 is not a multiple of 4".into())
    );
}

/// The values the group refuses, each alone and with a second refused value in a field read
/// later: the error names the field read first (`IPA`'s location before its frequency,
/// `ISBEWR`'s memory type before its access type).
#[test]
fn values_the_group_refuses() {
    let unknown = |opcode, bit, width, value| DecodeError::UnknownValue {
        opcode,
        bit,
        width,
        value,
    };
    let ipa = |loc, freq| {
        word(
            0x326,
            &[(16, 1), (32, 0xff), (76, loc), (78, freq), (81, 7)],
        )
    };
    assert_eq!(decode(ipa(3, 0)), Err(unknown(0x326, 76, 2, 3)));
    assert_eq!(decode(ipa(2, 3)), Err(unknown(0x326, 78, 2, 3)));
    assert_eq!(decode(ipa(3, 3)), Err(unknown(0x326, 76, 2, 3)));
    // One below each refused value decodes.
    assert!(decode(ipa(2, 2)).is_ok());
    for value in 5..=7 {
        assert_eq!(
            decode(word(0x925, &[(16, 1), (78, value), (81, 7)])),
            Err(unknown(0x925, 78, 3, u64::try_from(value).expect("3 bits")))
        );
    }
    assert!(decode(word(0x925, &[(16, 1), (78, 4), (81, 7)])).is_ok());
    let isberd = |mem, access| word(0x923, &[(16, 1), (24, 2), (74, mem), (76, access)]);
    assert_eq!(decode(isberd(3, 0)), Err(unknown(0x923, 74, 2, 3)));
    for access in 0..=3 {
        assert!(decode(isberd(2, access)).is_ok(), "{access}");
    }
    let isbewr = |mem, access| word(0x927, &[(24, 2), (32, 1), (74, mem), (76, access)]);
    assert_eq!(decode(isbewr(3, 0)), Err(unknown(0x927, 74, 2, 3)));
    assert_eq!(decode(isbewr(0, 1)), Err(unknown(0x927, 76, 2, 1)));
    assert_eq!(decode(isbewr(0, 2)), Err(unknown(0x927, 76, 2, 2)));
    assert_eq!(decode(isbewr(3, 1)), Err(unknown(0x927, 74, 2, 3)));
    assert!(decode(isbewr(2, 0)).is_ok());
    assert!(decode(isbewr(2, 3)).is_ok());
    assert_eq!(
        decode(ipa(3, 0)).map_err(|error| error.to_string()),
        Err(
            "opcode 0x326 has the value 3 in its 2-bit field at bit 76, which its form does \
             not define"
                .into()
        )
    );
}

/// `LDTRAM` requires bit 91 (NAK: "Unknown but required"), naming it when clear.
#[test]
fn ldtram_requires_bit_91() {
    let without = word(0x3ad, &[(16, 1), (24, 63), (64, 0x20)]);
    assert_eq!(
        decode(without),
        Err(DecodeError::MissingBit {
            opcode: 0x3ad,
            bit: 91
        })
    );
    assert_eq!(
        decode(Instruction::from_bits(without.bits() | 1 << 91)).map(|i| i.to_string()),
        Ok("ldtram.ab r1, a[0x80];".into())
    );
}

/// `OUT`'s forms 2, 3 and 7, which PR #4588's table has rows for, are unknown opcodes (module
/// docs); so are the neighbours of the group's opcodes that no reference names.
#[test]
fn out_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        AluForm::ImmSrc2.opcode(OUT_OPCODE),
        AluForm::CBufSrc2.opcode(OUT_OPCODE),
        AluForm::UniformSrc2.opcode(OUT_OPCODE),
        0x124,
        0x325,
        0x926,
    ];
    assert_eq!(unknown[..3], [0x524, 0x724, 0xf24], "the rows' opcodes");
    for opcode in unknown {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(decode(word), Err(DecodeError::UnknownOpcode { opcode }));
    }
}

/// `OUT` rejects its sources' modifier bits, which PR #4588's rows mark known: bits 72 and 73
/// in every form, bits 62 and 63 in the forms where they are not the immediate's. The rows'
/// `dropped` masks are exactly these, and no other row of the group drops a bit.
#[test]
fn out_rejects_source_modifier_bits() {
    let dropping: Vec<(u16, u128)> = attribute::REFERENCE_ROWS
        .iter()
        .filter(|row| row.dropped != 0)
        .map(|row| (row.opcode, row.dropped))
        .collect();
    let all_four = (0b11 << 72) | (0b11 << 62);
    assert_eq!(
        dropping,
        [
            (0x324, all_four),
            (0x924, 0b11 << 72),
            (0xb24, all_four),
            (0xd24, all_four),
        ]
    );
    for (opcode, dropped) in dropping {
        let fixed = lookup(opcode).expect("a form").fixed;
        let plain = Instruction::from_bits(word(u128::from(opcode), &[(78, 1)]).bits() | fixed);
        assert!(decode(plain).is_ok(), "{opcode:#x}");
        for bit in [62, 63, 72, 73] {
            let refused = decode(plain.with_flag(Field::bit(bit), true));
            if dropped & (1 << bit) != 0 {
                assert_eq!(refused, Err(DecodeError::UnknownBit { opcode, bit }));
            } else {
                // The immediate's top bits.
                assert!(refused.is_ok(), "{opcode:#x} bit {bit}");
            }
        }
    }
    let modified = AluSrc::Reg {
        reg: r(3),
        abs: true,
        neg: false,
    };
    assert_eq!(
        assemble(&Instr::new(Op::Out(OpOut {
            dst: r(1),
            handle: r(2),
            stream: modified,
            out_type: OutType::Emit,
        }))),
        Err(EncodeError::Modifier {
            mnemonic: "out",
            index: 1
        })
    );
    assert_eq!(
        assemble(&Instr::new(Op::OutFinal(OpOutFinal {
            dst: r(1),
            handle: r(2),
            stream: modified,
        }))),
        Err(EncodeError::Modifier {
            mnemonic: "out.final",
            index: 1
        })
    );
}

/// `OUT`'s type field picks the operation: 0 is `OUT.FINAL`, 1 to 3 are `OUT` with its type;
/// the other fields read the same either way.
#[test]
fn out_type_zero_is_out_final() {
    let fields = |out_type| word(0x324, &[(16, 1), (24, 2), (32, 3), (78, out_type)]);
    let stream = AluSrc::reg(r(3));
    assert_eq!(
        decode(fields(0)).map(|i| i.op),
        Ok(Op::OutFinal(OpOutFinal {
            dst: r(1),
            handle: r(2),
            stream,
        }))
    );
    for out_type in OutType::ALL {
        assert_eq!(
            decode(fields(u128::from(out_type.bits()))).map(|i| i.op),
            Ok(Op::Out(OpOut {
                dst: r(1),
                handle: r(2),
                stream,
                out_type,
            }))
        );
    }
    assert_eq!(
        OutType::ALL.map(OutType::bits),
        [1, 2, 3],
        "NAK's encoding: Emit 1, Cut 2, EmitThenCut 3"
    );
}

/// `S2UR` is `S2R` with a uniform destination: one operation, two forms, two printed names.
#[test]
fn s2ur_is_s2r_with_a_uniform_destination() {
    let uniform = decode(word(0x9c3, &[(16, 5), (72, 33)])).expect("decodes");
    assert_eq!(
        uniform.op,
        Op::S2R(OpS2R {
            dst: S2RDst::Uniform(ur(5)),
            sr: SpecialReg::TID_X,
        })
    );
    assert_eq!(uniform.op.mnemonic(), "s2r");
    let per_thread = decode(word(0x919, &[(16, 5), (72, 33)])).expect("decodes");
    assert_eq!(
        per_thread.op,
        Op::S2R(OpS2R {
            dst: S2RDst::Reg(r(5)),
            sr: SpecialReg::TID_X,
        })
    );
    // S2UR's destination is six bits: bits 22 and 23 are not its.
    for bit in [22, 23] {
        assert_eq!(
            decode(Instruction::from_bits(
                word(0x9c3, &[(16, 5)]).bits() | 1 << bit
            )),
            Err(DecodeError::UnknownBit { opcode: 0x9c3, bit })
        );
    }
}

/// The group's opcodes and field positions against the references' values, written out here:
/// NAK's `set_opcode`, `encode_alu` and field ranges (`sm70_encode.rs` @ 52238762).
#[test]
fn layout_constants_match_the_references() {
    let opcodes = [
        (ALD_OPCODE, 0x321),
        (AST_OPCODE, 0x322),
        (AL2P_OPCODE, 0x920),
        (IPA_OPCODE, 0x326),
        (LDTRAM_OPCODE, 0x3ad),
        (S2R_OPCODE, 0x919),
        (S2UR_OPCODE, 0x9c3),
        (CS2R_OPCODE, 0x805),
        (PIXLD_OPCODE, 0x925),
        (OUT_OPCODE, 0x124),
        (ISBERD_OPCODE, 0x923),
        (ISBEWR_OPCODE, 0x927),
    ];
    for (constant, literal) in opcodes {
        assert_eq!(constant, literal);
    }
    let fields = [
        (ATTR_OFFSET, (24, 8)),
        (ATTR_DATA, (32, 8)),
        (ATTR_ADDR, (40, 10)),
        (ATTR_COMPS, (74, 2)),
        (ATTR_OUTPUT, (79, 1)),
        (ATTR_WORD, (64, 8)),
        (SR_INDEX, (72, 8)),
        (ALD_VTX, (32, 8)),
        (AST_VTX, (64, 8)),
        (ATTR_PATCH, (76, 1)),
        (ATTR_PHYS, (77, 1)),
        (IPA_OFFSET, (32, 8)),
        (IPA_LOC, (76, 2)),
        (IPA_FREQ, (78, 2)),
        (LDTRAM_UREG, (24, 6)),
        (LDTRAM_C, (72, 1)),
        (LDTRAM_FIXED, (91, 1)),
        (CS2R_64, (80, 1)),
        (PIXLD_VAL, (78, 3)),
        (OUT_TYPE, (78, 2)),
        (ISBE_IMM, (40, 16)),
        (ISBE_MEM, (74, 2)),
        (ISBE_ACCESS, (76, 2)),
        (ISBE_SKEW, (78, 1)),
    ];
    for (field, (start, width)) in fields {
        assert_eq!((field.start, field.width), (start, width));
    }
    assert_eq!(ATTR_WORD_BYTES, 4);
    // NAK's enumerations, in encoding order.
    assert_eq!(InterpLoc::ALL.map(InterpLoc::bits), [0, 1, 2]);
    assert_eq!(InterpFreq::ALL.map(InterpFreq::bits), [0, 1, 2]);
    assert_eq!(PixVal::ALL.map(PixVal::bits), [0, 1, 2, 3, 4]);
    assert_eq!(IsbeMemType::ALL.map(IsbeMemType::bits), [0, 1, 2]);
    assert_eq!(IsbeAccess::ALL.map(IsbeAccess::bits), [0, 1, 2, 3]);
    assert_eq!(
        IsbeAccess::ALL.map(IsbeAccess::writable),
        [true, false, false, true]
    );
}

/// The encoder refuses what the fields cannot hold, with plain-word messages.
#[test]
fn unencodable_attribute_instructions_are_errors() {
    let al2p = |comps| {
        assemble(&Instr::new(Op::AL2P(OpAL2P {
            dst: r(1),
            offset: Gpr::RZ,
            addr: 0,
            comps,
            output: false,
        })))
    };
    assert!(al2p(1).is_ok());
    assert!(al2p(4).is_ok());
    assert_eq!(al2p(0), Err(EncodeError::Comps(0)));
    assert_eq!(al2p(5), Err(EncodeError::Comps(5)));
    let isbewr = |access| {
        assemble(&Instr::new(Op::Isbewr(OpIsbewr {
            offset: r(2),
            data: r(1),
            imm_offset: 0,
            mem_type: IsbeMemType::B32,
            access,
            output: true,
            skew: false,
        })))
    };
    assert!(isbewr(IsbeAccess::Map).is_ok());
    assert!(isbewr(IsbeAccess::Attribute).is_ok());
    assert_eq!(isbewr(IsbeAccess::Patch), Err(EncodeError::IsbewrAccess(1)));
    assert_eq!(
        isbewr(IsbeAccess::Primitive).map_err(|error| error.to_string()),
        Err("isbewr cannot write access type 2".into())
    );
    assert_eq!(
        EncodeError::AttrAlign { addr: 0, align: 4 }.to_string(),
        "attribute address 0x0 is not a multiple of 4"
    );
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn attribute_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(attribute::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit its layout does not read (the other constant-buffer
    /// layout's bits in `OUT`'s form 5) or for a value that names nothing.
    #[test]
    fn attribute_decode_then_assemble_round_trips(word in any_word_of(attribute::FORMS)) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { opcode, bit }) => {
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                prop_assert_eq!(opcode, AluForm::CBufSrc1.opcode(OUT_OPCODE), "bit {}", bit);
            }
            Err(DecodeError::UnknownValue { opcode, bit, .. }) => {
                prop_assert!(
                    [
                        (IPA_OPCODE, IPA_LOC.start),
                        (IPA_OPCODE, IPA_FREQ.start),
                        (PIXLD_OPCODE, PIXLD_VAL.start),
                        (ISBERD_OPCODE, ISBE_MEM.start),
                        (ISBEWR_OPCODE, ISBE_MEM.start),
                        (ISBEWR_OPCODE, ISBE_ACCESS.start),
                    ]
                    .contains(&(opcode, bit)),
                    "{opcode:#x} bit {bit}"
                );
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
