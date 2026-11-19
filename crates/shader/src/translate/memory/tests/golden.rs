//! Golden IR: the exact text of representative forms, so a change to what a form translates
//! to shows in review.

use super::*;

fn text(op: Op) -> String {
    let function = build_op(op).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    function.to_string()
}

/// `LDG.64 R8, [R2.64+UR4.64+-0x10], P3`: the uniform pair added to the register pair with a
/// carry, the signed offset as a 64-bit addition, the load predicate selecting each word.
#[test]
fn ldg_with_a_uniform_pointer_and_a_load_predicate() {
    let op = Op::Ld(OpLd {
        pred: ps(3, false),
        ..ldg(8, MemType::B64, with_uniform(2, true, 4, true, true, -0x10))
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetUniformRegister ur4\n\
         \x20   %1: u32 = GetUniformRegister ur5\n\
         \x20   %2: u32 = GetRegister r2\n\
         \x20   %3: u32 = IAdd32 %0, %2\n\
         \x20   %4: u1 = ULessThan %3, %0\n\
         \x20   %5: u32 = GetRegister r3\n\
         \x20   %6: u32 = IAdd32 %1, %5\n\
         \x20   %7: u32 = SelectU32 %4, u32 0x1, u32 0x0\n\
         \x20   %8: u32 = IAdd32 %6, %7\n\
         \x20   %9: u32x2 = CompositeConstructU32x2 %3, %8\n\
         \x20   %10: u64 = PackUint2x32 %9\n\
         \x20   %11: u64 = IAdd64 %10, u64 0xfffffffffffffff0\n\
         \x20   %12: u32x2 = LoadGlobal64 %11\n\
         \x20   %13: u1 = GetPred p3\n\
         \x20   %14: u32 = CompositeExtractU32x2 %12, u32 0x0\n\
         \x20   %15: u32 = CompositeExtractU32x2 %12, u32 0x1\n\
         \x20   %16: u32 = SelectU32 %13, %14, u32 0x0\n\
         \x20   %17 = SetRegister r8, %16\n\
         \x20   %18: u32 = SelectU32 %13, %15, u32 0x0\n\
         \x20   %19 = SetRegister r9, %18\n\
         \x20   return\n"
    );
}

/// `STG.E.128 [R2.64], R4`: the pair as the address, the quad built from the data registers.
#[test]
fn stg_128() {
    let op = Op::St(st(4, MemType::B128, GLOBAL, plain(2, true, 0)));
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = GetRegister r3\n\
         \x20   %2: u32x2 = CompositeConstructU32x2 %0, %1\n\
         \x20   %3: u64 = PackUint2x32 %2\n\
         \x20   %4: u32 = GetRegister r4\n\
         \x20   %5: u32 = GetRegister r5\n\
         \x20   %6: u32 = GetRegister r6\n\
         \x20   %7: u32 = GetRegister r7\n\
         \x20   %8: u32x4 = CompositeConstructU32x4 %4, %5, %6, %7\n\
         \x20   %9 = WriteGlobal128 %3, %8\n\
         \x20   return\n"
    );
}

/// `LDS.S16 R1, [R2.X8+0x6]`: the register scaled by a shift of 3, the offset added.
#[test]
fn lds_scaled() {
    let op = Op::Ld(lds(1, MemType::I16, 2, OffsetStride::X8, 6));
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = ShiftLeftLogical32 %0, u32 0x3\n\
         \x20   %2: u32 = IAdd32 %1, u32 0x6\n\
         \x20   %3: u32 = LoadSharedS16 %2\n\
         \x20   %4 = SetRegister r1, %3\n\
         \x20   return\n"
    );
}

/// `STL.U8 [R2+0x3], R5`: the word read, the byte inserted at the address's bit position, the
/// word written back.
#[test]
fn stl_byte() {
    let op = Op::St(st(5, MemType::U8, LOCAL, plain(2, false, 3)));
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = IAdd32 %0, u32 0x3\n\
         \x20   %2: u32 = ShiftRightArithmetic32 %1, u32 0x2\n\
         \x20   %3: u32 = ShiftLeftLogical32 %1, u32 0x3\n\
         \x20   %4: u32 = BitwiseAnd32 %3, u32 0x18\n\
         \x20   %5: u32 = LoadLocal %2\n\
         \x20   %6: u32 = GetRegister r5\n\
         \x20   %7: u32 = BitFieldInsert %5, %6, %4, u32 0x8\n\
         \x20   %8 = WriteLocal %2, %7\n\
         \x20   return\n"
    );
}

/// `LDC.128 R8, c[0x1f][R6+0x10]`: four 32-bit reads four bytes apart.
#[test]
fn ldc_128() {
    let op = Op::Ldc(OpLdc {
        dst: r(8),
        cbuf: cbuf(31, 0x10),
        offset: r(6),
        mode: LdcMode::Indexed,
        mem_type: MemType::B128,
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r6\n\
         \x20   %1: u32 = IAdd32 %0, u32 0x10\n\
         \x20   %2: u32 = GetCbufU32 u32 0x1f, %1\n\
         \x20   %3 = SetRegister r8, %2\n\
         \x20   %4: u32 = IAdd32 %1, u32 0x4\n\
         \x20   %5: u32 = GetCbufU32 u32 0x1f, %4\n\
         \x20   %6 = SetRegister r9, %5\n\
         \x20   %7: u32 = IAdd32 %1, u32 0x8\n\
         \x20   %8: u32 = GetCbufU32 u32 0x1f, %7\n\
         \x20   %9 = SetRegister r10, %8\n\
         \x20   %10: u32 = IAdd32 %1, u32 0xc\n\
         \x20   %11: u32 = GetCbufU32 u32 0x1f, %10\n\
         \x20   %12 = SetRegister r11, %11\n\
         \x20   return\n"
    );
}

/// `ATOMS.MIN.S32 R5, [R2+0x8], R3` and `RED.E.ADD [R2.64], R4`.
#[test]
fn atomics() {
    let shared = Op::Atom(atom(
        5,
        AtomOp::Min,
        AtomType::I32,
        MemSpace::Shared(OffsetStride::X1),
        plain(2, false, 8),
        3,
    ));
    assert_eq!(
        text(shared),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = IAdd32 %0, u32 0x8\n\
         \x20   %2: u32 = GetRegister r3\n\
         \x20   %3: u32 = SharedAtomicSMin32 %1, %2\n\
         \x20   %4 = SetRegister r5, %3\n\
         \x20   return\n"
    );
    let reduction = Op::Atom(red(AtomOp::Add, AtomType::U32, 2, true, 0, 4));
    assert_eq!(
        text(reduction),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = GetRegister r3\n\
         \x20   %2: u32x2 = CompositeConstructU32x2 %0, %1\n\
         \x20   %3: u64 = PackUint2x32 %2\n\
         \x20   %4: u32 = GetRegister r4\n\
         \x20   %5: u32 = GlobalAtomicIAdd32 %3, %4\n\
         \x20   return\n"
    );
}

/// `LDGSTS` from `R4:R5 + 0x10` to `R6 + 0x20`: per word, the global load selected by the
/// enable predicate and the shared write; then the workgroup barrier.
#[test]
fn ldgsts() {
    let op = Op::LdGsts(crate::decode::OpLdGsts {
        shared: r(6),
        shared_offset: 2,
        global: r(4),
        global_offset: 0x10,
        descriptor: 0,
        enable: ps(0, true),
    });
    let mut expected = String::from(
        "block0:\n\
         \x20   %0: u1 = GetPred p0\n\
         \x20   %1: u1 = LogicalNot %0\n",
    );
    let mut next = 2;
    for k in 0..4_u32 {
        let base = next;
        let lines = [
            format!("%{base}: u32 = GetRegister r4"),
            format!("%{}: u32 = GetRegister r5", base + 1),
            format!(
                "%{}: u32x2 = CompositeConstructU32x2 %{base}, %{}",
                base + 2,
                base + 1
            ),
            format!("%{}: u64 = PackUint2x32 %{}", base + 3, base + 2),
            format!(
                "%{}: u64 = IAdd64 %{}, u64 {:#x}",
                base + 4,
                base + 3,
                0x10 + 4 * k
            ),
            format!("%{}: u32 = LoadGlobal32 %{}", base + 5, base + 4),
            format!("%{}: u32 = SelectU32 %1, %{}, u32 0x0", base + 6, base + 5),
            format!("%{}: u32 = GetRegister r6", base + 7),
            format!(
                "%{}: u32 = IAdd32 %{}, u32 {:#x}",
                base + 8,
                base + 7,
                0x20 + 4 * k
            ),
            format!(
                "%{} = WriteSharedU32 %{}, %{}",
                base + 9,
                base + 8,
                base + 6
            ),
        ];
        for line in lines {
            expected.push_str("    ");
            expected.push_str(&line);
            expected.push('\n');
        }
        next += 10;
    }
    expected.push_str(&format!(
        "    %{next} = WorkgroupMemoryBarrier\n    return\n"
    ));
    assert_eq!(text(op), expected);
}
