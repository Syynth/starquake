//! Texture and surface translator tests: every instruction and modifier run through the IR
//! interpreter (slice 7), whose scripted image calls check each operation's opcode, texture
//! information and argument values, with expected values worked out from the instruction's
//! documented behaviour and written as literals; golden IR; PR #4588's cases for this group,
//! rebuilt from their words; every refusal with its exact error and order; and a fixed-seed
//! property test that no decoded instruction panics.
//!
//! Each instruction a test runs goes through the test assembler and the decoder first and must
//! come back unchanged ([`encodable`]), so each case is a word the decoder reads; the few
//! hand-built operations no word encodes are translated directly ([`translate_raw`]).

mod fetch;
mod gather;
mod golden;
mod guards;
mod pr4588;
mod query;
mod refusals;
mod sample;
mod surface;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{
    GROUP_HOOKS, GROUPS, Instruction, MemEvictionPriority, MemOrder, OPCODE, PredSrc, TexCBufRef,
    TexChannel, decode,
};
use crate::interp::{ImageCall, Machine, ScriptedImage, Val};
use crate::ir::{Function, Terminator};
use crate::testing::{any_pred_src, any_word_of, config};

pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

pub(super) fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

pub(super) fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate")
}

pub(super) fn mask(bits: u8) -> ChannelMask {
    ChannelMask::new(bits).expect("four bits")
}

/// A constant-buffer handle.
pub(super) const fn cb(index: u8, offset: u16) -> TexRef {
    TexRef::CBuf(TexCBufRef { index, offset })
}

/// The bits of `value`.
pub(super) const fn f(value: f32) -> u32 {
    value.to_bits()
}

/// `instr`, after checking that the test assembler encodes it and the decoder reads the word
/// back as the same instruction.
pub(super) fn encodable(instr: Instr) -> Instr {
    let word = assemble(&instr).expect("assembles");
    assert_eq!(decode(word), Ok(instr), "{instr}");
    instr
}

/// `instr` translated into a fresh function that returns after it, starting from `buffer`,
/// without the encoding check; the function and the descriptor buffer it leaves.
pub(super) fn translate_raw(
    instr: Instr,
    buffer: DescriptorBuffer,
) -> Result<(Function, DescriptorBuffer), TextureError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let mut buffer = buffer;
    translate_instr(&mut builder, instr, false, &mut buffer)?;
    builder.terminate(Terminator::Return)?;
    Ok((function, buffer))
}

/// `op`, checked by [`encodable`], translated with no guard; the result verifies.
pub(super) fn translated(op: Op) -> Function {
    let instr = encodable(Instr::new(op));
    let (function, _) = translate_raw(instr, DescriptorBuffer::new()).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    function
}

/// A machine with registers set from `regs`.
pub(super) fn machine_with(regs: &[(u8, u32)]) -> Machine {
    let mut machine = Machine::new();
    for &(index, bits) in regs {
        machine.regs[usize::from(index)] = bits;
    }
    machine
}

/// Runs `function` on `machine` with `script` as its image calls, and checks that every
/// scripted call was made.
pub(super) fn run_script(function: &Function, machine: &mut Machine, script: Vec<ScriptedImage>) {
    machine.images = script.into();
    machine.run(function).expect("runs");
    assert!(machine.images.is_empty(), "a scripted call was not made");
}

/// Translates `op` (checked by [`encodable`]) and runs it on a machine with `regs`, against
/// `script`.
pub(super) fn run_op(op: Op, regs: &[(u8, u32)], script: Vec<ScriptedImage>) -> Machine {
    let function = translated(op);
    let mut machine = machine_with(regs);
    run_script(&function, &mut machine, script);
    machine
}

/// A scripted call.
pub(super) fn call(
    opcode: Opcode,
    info: TextureInfo,
    args: Vec<Val>,
    reply: Option<Val>,
) -> ScriptedImage {
    ScriptedImage {
        call: ImageCall { opcode, info, args },
        reply,
    }
}

/// Texture information of type `ty`, otherwise the default.
pub(super) fn info(ty: TextureType) -> TextureInfo {
    TextureInfo {
        ty,
        ..TextureInfo::default()
    }
}

/// A texture dimension, its IR type, the registers a case sets and the coordinates read.
pub(super) type DimCase<'a> = (TexDim, TextureType, &'a [(u8, u32)], Val);

/// Where [`TEXEL`]'s channels selected by `bits` go: the first two to `first` and the
/// register after it, the rest to `second` and the one after it; each as (register, value).
pub(super) fn placed(bits: u32, first: u32, second: u32) -> Vec<(usize, u32)> {
    let mut stored = 0;
    let mut placed = Vec::new();
    for (channel, &value) in TEXEL.iter().enumerate() {
        if bits & (1 << channel) != 0 {
            let target = if stored < 2 {
                first + stored
            } else {
                second + stored - 2
            };
            placed.push((usize::try_from(target).expect("small"), value));
            stored += 1;
        }
    }
    placed
}

/// What [`ABSENT`] reads as.
pub(super) const NONE: Val = Val::U32(0);

/// A result of four distinct floats, and their bits.
pub(super) const TEXEL: [u32; 4] = [f(0.125), f(0.25), f(0.5), f(0.75)];

/// The rejection translating `instr` gives, and that nothing was appended and the buffer is
/// unchanged.
pub(super) fn assert_refused_raw(instr: Instr, buffer: DescriptorBuffer, what: TextureUnsupported) {
    let mut function = Function::new();
    let entry = function.entry();
    let before = function.clone();
    let mut builder = Builder::new(&mut function, entry);
    let mut after = buffer;
    assert_eq!(
        translate_instr(&mut builder, instr, false, &mut after),
        Err(TextureError::Unsupported(what)),
        "{instr}"
    );
    assert_eq!(builder.function(), &before, "{instr}");
    assert_eq!(after, buffer, "{instr}");
}

/// [`assert_refused_raw`] with no guard and a fresh buffer, after checking `op` encodes.
pub(super) fn assert_refused(op: Op, what: TextureUnsupported) {
    assert_refused_raw(encodable(Instr::new(op)), DescriptorBuffer::new(), what);
}

/// [`assert_refused_raw`] with no guard and a fresh buffer, for an op no word encodes.
pub(super) fn assert_refused_unencodable(op: Op, what: TextureUnsupported) {
    let instr = Instr::new(op);
    assert!(assemble(&instr).is_err(), "{instr} encodes");
    assert_refused_raw(instr, DescriptorBuffer::new(), what);
}

/// A plain `TEX`: a constant-buffer handle at `c[2][0x40]`, coordinates from R10 and R20,
/// every channel to R0 to R3, 2D, implicit level, scalar.
pub(super) fn tex() -> OpTex {
    OpTex {
        dsts: [r(0), r(2)],
        fault: Pred::PT,
        tex: cb(2, 0x40),
        srcs: [r(10), r(20)],
        dim: TexDim::D2,
        lod_mode: TexLodMode::Auto,
        ndv: false,
        z_cmpr: false,
        offset: false,
        f16: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
        scalar: true,
    }
}

/// A plain `TLD`, as [`tex`], at level zero.
pub(super) fn tld() -> OpTld {
    OpTld {
        dsts: [r(0), r(2)],
        fault: Pred::PT,
        tex: cb(2, 0x40),
        srcs: [r(10), r(20)],
        dim: TexDim::D2,
        is_ms: false,
        clamp: false,
        lod_mode: TexLodMode::Zero,
        offset: false,
        f16: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
        scalar: true,
    }
}

/// A plain `TLD4`, as [`tex`], gathering green.
pub(super) fn tld4() -> OpTld4 {
    OpTld4 {
        dsts: [r(0), r(2)],
        fault: Pred::PT,
        tex: cb(2, 0x40),
        srcs: [r(10), r(20)],
        dim: TexDim::D2,
        comp: TexChannel::G,
        offset_mode: TexOffsetMode::None,
        z_cmpr: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
        scalar: true,
    }
}

/// A plain `TXD`: coordinates from R10, derivatives from R20.
pub(super) fn txd() -> OpTxd {
    OpTxd {
        dsts: [r(0), r(2)],
        fault: Pred::PT,
        tex: cb(2, 0x40),
        srcs: [r(10), r(20)],
        dim: TexDim::D2,
        offset: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
    }
}

/// A plain `TMML`: coordinates from R10, the relative level to R4.
pub(super) fn tmml() -> OpTmml {
    OpTmml {
        dsts: [r(4), Gpr::RZ],
        tex: cb(2, 0x40),
        srcs: [r(10), Gpr::RZ],
        dim: TexDim::D2,
        ndv: false,
        nodep: false,
        channel_mask: mask(0x1),
    }
}

/// A plain `TXQ`: the dimensions of the level in R10, every word to R4 to R7.
pub(super) fn txq() -> OpTxq {
    OpTxq {
        dsts: [r(4), Gpr::RZ],
        tex: cb(2, 0x40),
        src: r(10),
        query: TexQuery::Dimension,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
    }
}

/// A plain formatted `SULD`: 2D, coordinates from R10, handle in R30, every channel to R0.
pub(super) fn suld() -> OpSuLd {
    OpSuLd {
        dst: r(0),
        fault: Pred::PT,
        access: ImageAccess::Formatted(ChannelMask::RGBA),
        dim: ImageDim::D2,
        order: MemOrder::Weak,
        eviction: MemEvictionPriority::Normal,
        handle: r(30),
        coord: r(10),
    }
}

/// A plain formatted `SUST`, as [`suld`], data from R20.
pub(super) fn sust() -> OpSuSt {
    OpSuSt {
        access: ImageAccess::Formatted(ChannelMask::RGBA),
        dim: ImageDim::D2,
        order: MemOrder::Weak,
        eviction: MemEvictionPriority::Normal,
        handle: r(30),
        coord: r(10),
        data: r(20),
    }
}

/// A plain `SUATOM.ADD.U32`, as [`sust`], the old value to R0.
pub(super) fn suatom() -> OpSuAtom {
    OpSuAtom {
        dst: r(0),
        fault: Pred::PT,
        dim: ImageDim::D2,
        atom_op: AtomOp::Add,
        atom_type: AtomType::U32,
        order: MemOrder::Weak,
        eviction: MemEvictionPriority::Normal,
        handle: r(30),
        coord: r(10),
        data: r(20),
        ba: false,
        reduce: false,
    }
}

/// The texture group's generators (`decode`'s per-group hooks) and forms, found by the group
/// whose forms hold `TEX`'s opcode.
fn texture_group() -> usize {
    let word = assemble(&Instr::new(Op::Tex(tex()))).expect("assembles");
    let opcode = word.field(OPCODE);
    GROUPS
        .iter()
        .position(|forms| forms.iter().any(|form| u64::from(form.opcode) == opcode))
        .expect("a group decodes tex")
}

fn texture_ops() -> Vec<BoxedStrategy<Op>> {
    (GROUP_HOOKS[texture_group()].any_ops)()
}

fn texture_words() -> impl Strategy<Value = Instruction> {
    any_word_of(GROUPS[texture_group()])
}

/// Checks one generated instruction: it translates to IR that verifies (and sets the buffer
/// only to a bound buffer), or is refused with nothing appended and the buffer unchanged.
fn check_generated(instr: Instr, tested: bool, bound: Option<u8>) -> Result<(), TestCaseError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let before = builder.function().clone();
    let start = bound.map_or(DescriptorBuffer::new(), |index| {
        DescriptorBuffer::new().bind(index).unwrap_or_default()
    });
    let mut buffer = start;
    match translate_instr(&mut builder, instr, tested, &mut buffer) {
        Ok(()) => {
            builder.terminate(Terminator::Return).expect("terminates");
            prop_assert_eq!(function.verify(), Ok(()));
            prop_assert!(
                buffer
                    .index()
                    .is_none_or(|index| index < DESCRIPTOR_BUFFERS)
            );
        }
        Err(TextureError::Unsupported(_)) => {
            prop_assert_eq!(builder.function(), &before);
            prop_assert_eq!(buffer, start);
        }
        Err(TextureError::Build(error)) => prop_assert!(false, "{}", error),
    }
    Ok(())
}

proptest! {
    #![proptest_config(config())]

    /// Every generated texture or surface instruction under every guard translates to IR that
    /// verifies, or is refused with nothing appended; none panics.
    #[test]
    fn generated_instructions_never_panic(
        op in proptest::strategy::Union::new(texture_ops()),
        guard in any_pred_src(),
        tested in any::<bool>(),
        bound in proptest::option::of(0_u8..DESCRIPTOR_BUFFERS),
    ) {
        check_generated(Instr { guard, op }, tested, bound)?;
    }

    /// Every word of the group's forms that decodes translates or is refused; none panics.
    #[test]
    fn decoded_words_never_panic(word in texture_words(), tested in any::<bool>()) {
        if let Ok(instr) = decode(word) {
            check_generated(instr, tested, None)?;
        }
    }
}
