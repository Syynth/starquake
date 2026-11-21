//! Tracing a bindless texture handle back to the constant-buffer words it is built from (Eden's
//! `Track` and `TryGetConstBuffer`, `eden/src/shader_recompiler/ir_opt/texture_pass.cpp` @
//! cf654052), exactly rather than by Eden's breadth-first search (the parent module's "Tracing
//! is exact").

use std::fmt::{self, Display, Formatter};

use super::table::{HandleSource, HandleWord, IndexWord};
use super::{
    DESCRIPTOR_MAX_COUNT, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE_SHIFT, DYNAMIC_ARRAY_BYTES,
    MAX_TRACE_DEPTH, MIN_DYNAMIC_SIZE_SHIFT,
};
use crate::ir::{Function, Imm, Inst, Opcode, Operand, Operation, Value};
use crate::opt::Work;
use crate::translate::float::{CBUF_BANKS, CBUF_WORD_BYTES};

/// The bytes a `GetCbufU32x2` reads, which it aligns its offset down to (Eden's `GetCbuf`:
/// "Hardware been proved to read the aligned offset").
const PAIR_BYTES: u32 = 2 * CBUF_WORD_BYTES;

/// Why a bindless texture handle could not be traced. Each is a rejection (decision 8: "texture
/// handles are traced first"); the parent module's divergence table says which Eden accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Untraced {
    /// The handle is an immediate: no constant buffer holds it (Eden's `BreadthFirstSearch`:
    /// "Nothing to do with immediates").
    Immediate,
    /// The handle passes through a phi.
    Phi(Value),
    /// The handle comes from an operation the tracer does not follow.
    Opcode {
        /// The operation.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// The handle is loaded from global or storage memory (PR #4588:
    /// `NotImplementedException("Texture handles loaded from storage or global memory")`).
    GlobalMemory {
        /// The load.
        at: Value,
        /// Its opcode.
        opcode: Opcode,
    },
    /// The handle comes from a vertex attribute (Eden's `IsAttributeSourcedHandle`, taken
    /// only where the host sets `support_attribute_texture_handle`).
    Attribute,
    /// The handle is read from a constant buffer whose bank is a value (Eden: "Reading a
    /// bindless texture from variable indices is valid but not supported here at the moment").
    VariableBank(Value),
    /// The handle is read from a bank at or past
    /// [`CBUF_BANKS`](crate::translate::float::CBUF_BANKS).
    BankPastLast(u32),
    /// A shift whose amount is a value.
    ShiftNotImmediate(Value),
    /// A shift by 32 or more.
    ShiftTooWide {
        /// The shift.
        at: Value,
        /// Its amount.
        shift: u32,
    },
    /// A shift of a word that is shifted already.
    ShiftOfShifted(Value),
    /// A shift of a handle built from two words.
    ShiftOfPair(Value),
    /// A mask whose operands are both values (Eden takes a mask from a word of constant
    /// buffer 1, `TryGetConstant`; not followed here).
    MaskNotImmediate(Value),
    /// A mask of a handle built from two words.
    MaskOfPair(Value),
    /// A handle OR-ed from more than two words.
    MoreThanTwoWords(Value),
    /// Two handle arrays indexed at run time OR-ed together.
    TwoArrays(Value),
    /// Two handle arrays whose index each comes from a constant buffer, OR-ed together.
    TwoIndexWords(Value),
    /// A register offset that is a sum with no immediate side.
    DynamicOffset(Value),
    /// An element of something other than a `GetCbufU32x2` at an immediate bank and offset,
    /// or an element index other than an immediate 0 or 1.
    PairElement(Value),
    /// More than [`MAX_TRACE_DEPTH`] operations deep.
    TooDeep,
    /// An operation with an argument missing or of the wrong kind, or a value naming no
    /// instruction.
    Malformed(Value),
}

impl Display for Untraced {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Immediate => f.write_str("it is an immediate, which no constant buffer holds"),
            Self::Phi(at) => write!(f, "it passes through the phi {at}"),
            Self::Opcode { at, opcode } => write!(f, "it comes from {opcode} at {at}"),
            Self::GlobalMemory { at, opcode } => write!(
                f,
                "it is loaded from storage or global memory by {opcode} at {at}"
            ),
            Self::Attribute => f.write_str("it comes from a vertex attribute"),
            Self::VariableBank(at) => {
                write!(
                    f,
                    "it is read at {at} from a constant buffer named by a value"
                )
            }
            Self::BankPastLast(bank) => write!(
                f,
                "it is read from constant buffer {bank}; banks stop at {}",
                CBUF_BANKS - 1
            ),
            Self::ShiftNotImmediate(at) => write!(f, "{at} shifts it by a value"),
            Self::ShiftTooWide { at, shift } => write!(f, "{at} shifts it by {shift} bits"),
            Self::ShiftOfShifted(at) => write!(f, "{at} shifts a word that is shifted already"),
            Self::ShiftOfPair(at) => write!(f, "{at} shifts a handle built from two words"),
            Self::MaskNotImmediate(at) => write!(f, "{at} masks it with a value"),
            Self::MaskOfPair(at) => write!(f, "{at} masks a handle built from two words"),
            Self::MoreThanTwoWords(at) => write!(f, "{at} builds it from more than two words"),
            Self::TwoArrays(at) => {
                write!(f, "{at} combines two handle arrays indexed at run time")
            }
            Self::TwoIndexWords(at) => write!(
                f,
                "{at} combines two handle arrays indexed by constant-buffer words"
            ),
            Self::DynamicOffset(at) => {
                write!(f, "{at} reads it at a sum of two values")
            }
            Self::PairElement(at) => write!(
                f,
                "{at} takes it from a pair other than one constant-buffer read at an immediate offset"
            ),
            Self::TooDeep => write!(f, "it is more than {MAX_TRACE_DEPTH} operations deep"),
            Self::Malformed(at) => {
                write!(f, "{at} has an argument missing or of the wrong kind")
            }
        }
    }
}

/// A traced handle: where it is found, and the register offset that indexes an array of
/// handles at run time, which stays in the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Traced {
    /// Where the handle is found.
    pub(super) source: HandleSource,
    /// The array's byte offset, for an array read at a register offset.
    pub(super) dynamic: Option<Operand>,
}

/// Traces `handle`, the first argument of a bindless texture operation.
pub(super) fn trace(
    function: &Function,
    handle: Operand,
    work: &mut Work,
) -> Result<Traced, Untraced> {
    if is_attribute_sourced(function, handle) {
        return Err(Untraced::Attribute);
    }
    walk(function, handle, 0, false, work)
}

/// The immediate `u32` `operand` holds, if it is one.
pub(super) const fn imm_u32(operand: Operand) -> Option<u32> {
    match operand {
        Operand::Imm(Imm::U32(value)) => Some(value),
        _ => None,
    }
}

/// `operand` with `Identity` chains followed, at most [`MAX_TRACE_DEPTH`] links (Eden's
/// `InstRecursive`).
fn resolve(function: &Function, operand: Operand) -> Operand {
    let mut current = operand;
    for _ in 0..MAX_TRACE_DEPTH {
        let Some(Inst::Op(op)) = current.value().and_then(|at| function.inst(at)) else {
            break;
        };
        match (op.opcode, op.args.first()) {
            (Opcode::Identity, Some(&next)) => current = next,
            _ => break,
        }
    }
    current
}

/// The operation `operand` names, after `Identity` chains.
fn op_of(function: &Function, operand: Operand) -> Option<(Value, &Operation)> {
    let at = resolve(function, operand).value()?;
    match function.inst(at)? {
        Inst::Op(op) => Some((at, op)),
        Inst::Phi(_) => None,
    }
}

/// Whether `handle` is an attribute load, possibly through a `BitCastU32F32` (Eden's
/// `IsAttributeSourcedHandle`).
fn is_attribute_sourced(function: &Function, handle: Operand) -> bool {
    let Some((_, mut op)) = op_of(function, handle) else {
        return false;
    };
    if op.opcode == Opcode::BitCastU32F32
        && let Some(inner) = op
            .args
            .first()
            .and_then(|&arg| op_of(function, arg).map(|(_, inner)| inner))
    {
        op = inner;
    }
    matches!(op.opcode, Opcode::GetAttribute | Opcode::GetAttributeU32)
}

/// Whether `opcode` loads from global or storage memory (PR #4588's list in
/// `TryGetConstBuffer`).
const fn is_memory_load(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::LoadGlobalU8
            | Opcode::LoadGlobalS8
            | Opcode::LoadGlobalU16
            | Opcode::LoadGlobalS16
            | Opcode::LoadGlobal32
            | Opcode::LoadGlobal64
            | Opcode::LoadGlobal128
            | Opcode::LoadStorageU8
            | Opcode::LoadStorageS8
            | Opcode::LoadStorageU16
            | Opcode::LoadStorageS16
            | Opcode::LoadStorage32
            | Opcode::LoadStorage64
            | Opcode::LoadStorage128
    )
}

/// One step of the trace, `depth` operations below the handle; `in_pair` says an OR above
/// has already split the handle in two.
fn walk(
    function: &Function,
    operand: Operand,
    depth: u32,
    in_pair: bool,
    work: &mut Work,
) -> Result<Traced, Untraced> {
    work.step();
    if depth > MAX_TRACE_DEPTH {
        return Err(Untraced::TooDeep);
    }
    let Operand::Value(at) = operand else {
        return Err(Untraced::Immediate);
    };
    let op = match function.inst(at) {
        Some(Inst::Op(op)) => op,
        Some(Inst::Phi(_)) => return Err(Untraced::Phi(at)),
        None => return Err(Untraced::Malformed(at)),
    };
    let arg = |index: usize| op.args.get(index).copied().ok_or(Untraced::Malformed(at));
    let deeper = depth + 1;
    match op.opcode {
        Opcode::Identity => walk(function, arg(0)?, deeper, in_pair, work),
        Opcode::BitwiseOr32 => {
            if in_pair {
                return Err(Untraced::MoreThanTwoWords(at));
            }
            let lhs = walk(function, arg(0)?, deeper, true, work)?;
            let rhs = walk(function, arg(1)?, deeper, true, work)?;
            combine(at, lhs, rhs)
        }
        Opcode::ShiftLeftLogical32 => {
            let shift = arg(1)?;
            let Some(shift) = imm_u32(shift) else {
                return Err(if shift.value().is_some() {
                    Untraced::ShiftNotImmediate(at)
                } else {
                    Untraced::Malformed(at)
                });
            };
            if shift >= u32::BITS {
                return Err(Untraced::ShiftTooWide { at, shift });
            }
            let inner = walk(function, arg(0)?, deeper, in_pair, work)?;
            if inner.source.secondary().is_some() {
                return Err(Untraced::ShiftOfPair(at));
            }
            let primary = inner.source.primary();
            if primary.shift_left() != 0 {
                return Err(Untraced::ShiftOfShifted(at));
            }
            Ok(Traced {
                source: inner.source.with_primary(primary.shifted(shift)),
                ..inner
            })
        }
        Opcode::BitwiseAnd32 => {
            let (mut value, mut mask) = (arg(0)?, arg(1)?);
            if value.imm().is_some() {
                std::mem::swap(&mut value, &mut mask);
            }
            if value.imm().is_some() {
                return Err(Untraced::Immediate);
            }
            let Some(mask) = imm_u32(mask) else {
                return Err(if mask.value().is_some() {
                    Untraced::MaskNotImmediate(at)
                } else {
                    Untraced::Malformed(at)
                });
            };
            let inner = walk(function, value, deeper, in_pair, work)?;
            if inner.source.secondary().is_some() {
                return Err(Untraced::MaskOfPair(at));
            }
            let primary = inner.source.primary().masked(mask);
            Ok(Traced {
                source: inner.source.with_primary(primary),
                ..inner
            })
        }
        Opcode::GetCbufU32 => cbuf_read(function, at, arg(0)?, arg(1)?),
        Opcode::CompositeExtractU32x2 => pair_element(function, at, arg(0)?, arg(1)?),
        opcode if is_memory_load(opcode) => Err(Untraced::GlobalMemory { at, opcode }),
        opcode => Err(Untraced::Opcode { at, opcode }),
    }
}

/// The bank of a constant-buffer read at `at`: an immediate below [`CBUF_BANKS`].
fn bank_of(at: Value, bank: Operand) -> Result<u8, Untraced> {
    match bank {
        Operand::Value(_) => Err(Untraced::VariableBank(at)),
        Operand::Imm(Imm::U32(bank)) => u8::try_from(bank)
            .ok()
            .filter(|&b| b < CBUF_BANKS)
            .ok_or(Untraced::BankPastLast(bank)),
        Operand::Imm(_) => Err(Untraced::Malformed(at)),
    }
}

/// `offset` aligned down to a word, as a `GetCbufU32` reads it.
const fn word_aligned(offset: u32) -> u32 {
    offset - offset % CBUF_WORD_BYTES
}

/// A `GetCbufU32` (Eden's `TryGetConstBuffer`, its `GetCbufU32` arm): a word at an immediate
/// offset, or an array of handles at a register offset, optionally plus an immediate base.
fn cbuf_read(
    function: &Function,
    at: Value,
    bank: Operand,
    offset: Operand,
) -> Result<Traced, Untraced> {
    let bank = bank_of(at, bank)?;
    let dynamic = match offset {
        Operand::Imm(Imm::U32(offset)) => {
            return Ok(Traced {
                source: HandleSource::single(HandleWord::new(bank, word_aligned(offset))),
                dynamic: None,
            });
        }
        Operand::Imm(_) => return Err(Untraced::Malformed(at)),
        Operand::Value(_) => offset,
    };
    let (base, dynamic) = match op_of(function, dynamic) {
        Some((_, add)) if add.opcode == Opcode::IAdd32 => {
            let (Some(&a), Some(&b)) = (add.args.first(), add.args.get(1)) else {
                return Err(Untraced::Malformed(at));
            };
            if let Some(base) = imm_u32(a) {
                (base, b)
            } else if let Some(base) = imm_u32(b) {
                (base, a)
            } else {
                return Err(Untraced::DynamicOffset(at));
            }
        }
        // The handle array starts at the beginning of the constant buffer (Eden).
        _ => (0, dynamic),
    };
    let size_shift = dynamic_size_shift(function, dynamic);
    let index = uniform_index(function, dynamic);
    let count = dynamic_count(base, size_shift);
    Ok(Traced {
        source: HandleSource::array(
            HandleWord::new(bank, word_aligned(base)),
            count,
            size_shift,
            index,
        ),
        dynamic: Some(dynamic),
    })
}

/// A `CompositeExtractU32x2` of a `GetCbufU32x2` at an immediate bank and offset: the word
/// the element names, in the eight bytes the read covers.
fn pair_element(
    function: &Function,
    at: Value,
    pair: Operand,
    element: Operand,
) -> Result<Traced, Untraced> {
    let element = imm_u32(element)
        .filter(|&e| e < 2)
        .ok_or(Untraced::PairElement(at))?;
    let Some((read, op)) = op_of(function, pair) else {
        return Err(Untraced::PairElement(at));
    };
    if op.opcode != Opcode::GetCbufU32x2 {
        return Err(Untraced::PairElement(at));
    }
    let (Some(&bank), Some(&offset)) = (op.args.first(), op.args.get(1)) else {
        return Err(Untraced::Malformed(read));
    };
    let bank = bank_of(read, bank)?;
    let offset = imm_u32(offset).ok_or(Untraced::PairElement(at))?;
    let word = offset - offset % PAIR_BYTES + element * CBUF_WORD_BYTES;
    Ok(Traced {
        source: HandleSource::single(HandleWord::new(bank, word)),
        dynamic: None,
    })
}

/// Combines two traced halves of `BitwiseOr32` at `at` (Eden's `BitwiseOr32` arm): which one
/// is the primary decides which is indexed and how descriptors sort.
fn combine(at: Value, lhs: Traced, rhs: Traced) -> Result<Traced, Untraced> {
    let (mut lhs, mut rhs) = (lhs, rhs);
    if lhs.source.count() > 1 && rhs.source.count() > 1 {
        return Err(Untraced::TwoArrays(at));
    }
    let (l, r) = (lhs.source, rhs.source);
    if l.index().is_some() || r.index().is_some() {
        if l.index().is_some() && r.index().is_some() {
            return Err(Untraced::TwoIndexWords(at));
        }
        if r.index().is_some() {
            std::mem::swap(&mut lhs, &mut rhs);
        }
    } else if r.count() > 1 {
        // An indexed handle array combined with a single handle: only the primary is indexed.
        std::mem::swap(&mut lhs, &mut rhs);
    } else if l.count() == 1 {
        let (lp, rp) = (l.primary(), r.primary());
        if lp.shift_left() > 0 || lp.bank() > rp.bank() || lp.offset() > rp.offset() {
            std::mem::swap(&mut lhs, &mut rhs);
        }
    }
    Ok(Traced {
        source: lhs.source.with_secondary(rhs.source.primary()),
        dynamic: lhs.dynamic,
    })
}

/// How far apart the handles of an array are, from the shift that scales its index (Eden's
/// `DynamicDescriptorSizeShift`: a shift by an immediate from [`MIN_DYNAMIC_SIZE_SHIFT`] to
/// 30, else [`DESCRIPTOR_SIZE_SHIFT`]).
pub(super) fn dynamic_size_shift(function: &Function, dynamic: Operand) -> u32 {
    let Some((_, op)) = op_of(function, dynamic) else {
        return DESCRIPTOR_SIZE_SHIFT;
    };
    if op.opcode != Opcode::ShiftLeftLogical32 {
        return DESCRIPTOR_SIZE_SHIFT;
    }
    match op.args.get(1).copied().and_then(imm_u32) {
        Some(shift) if (MIN_DYNAMIC_SIZE_SHIFT..u32::BITS - 1).contains(&shift) => shift,
        _ => DESCRIPTOR_SIZE_SHIFT,
    }
}

/// How many handles an array starting at `base` holds (Eden's `DynamicDescriptorCount` with
/// the limit [`DESCRIPTOR_MAX_COUNT`]).
pub(super) const fn dynamic_count(base: u32, size_shift: u32) -> u32 {
    if size_shift >= u32::BITS - 1 || base >= DYNAMIC_ARRAY_BYTES {
        return 1;
    }
    let stride = 1_u32 << size_shift;
    let available = DYNAMIC_ARRAY_BYTES - base;
    if available < DESCRIPTOR_SIZE {
        return 1;
    }
    let count = 1 + (available - DESCRIPTOR_SIZE) / stride;
    if count < DESCRIPTOR_MAX_COUNT {
        count
    } else {
        DESCRIPTOR_MAX_COUNT
    }
}

/// The constant-buffer word an array's offset is, possibly shifted left by an immediate
/// (Eden's `TrackUniformIndex`); `None` when the offset is anything else, or the word's bank
/// is past [`CBUF_BANKS`].
fn uniform_index(function: &Function, dynamic: Operand) -> Option<IndexWord> {
    if dynamic.imm().is_some() {
        return None;
    }
    let (_, mut op) = op_of(function, dynamic)?;
    let mut shift = 0;
    if op.opcode == Opcode::ShiftLeftLogical32 {
        let inner = *op.args.first()?;
        shift = op.args.get(1).copied().and_then(imm_u32)?;
        if inner.imm().is_some() {
            return None;
        }
        op = op_of(function, inner)?.1;
    }
    if op.opcode != Opcode::GetCbufU32 || shift >= u32::BITS {
        return None;
    }
    let bank = op.args.first().copied().and_then(imm_u32)?;
    let offset = op.args.get(1).copied().and_then(imm_u32)?;
    let bank = u8::try_from(bank).ok().filter(|&b| b < CBUF_BANKS)?;
    Some(IndexWord::new(bank, word_aligned(offset), shift))
}
