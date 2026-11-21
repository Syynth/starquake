//! Local and shared memory: pass 6 of the spec's "Passes needed for correctness", which slice
//! 16 owns (decided by the owner, 2026-11-18).
//!
//! Local memory is a private array of `u32` words and shared memory a workgroup array of `u32`
//! words, each `ceil(bytes / 4)` words long, as Eden sizes them
//! (`eden/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp` @ cf654052,
//! `DefineLocalMemory`, and `DefineSharedMemory`'s path without an explicit workgroup layout).
//! An array is declared only when the program uses it and its size is not zero.
//!
//! - `LoadLocal` and `WriteLocal` take a word index, as Eden's `EmitLoadLocal` and
//!   `EmitWriteLocal` do (`emit_spirv_context_get_set.cpp`) and the interpreter does.
//! - The shared accesses take a byte offset; the word is `offset >> 2`, and the 64- and 128-bit
//!   accesses touch that word and the next one or three, as Eden's non-explicit-layout path
//!   does (`emit_spirv_shared_memory.cpp`, `EmitLoadSharedU64`, `EmitLoadSharedU128`,
//!   `EmitWriteSharedU64`, `EmitWriteSharedU128`). The 8- and 16-bit loads extract the field at
//!   bit `(offset << 3) & 24` (or `& 16`) of the word, sign- or zero-extended (`ExtractArgs`
//!   with Eden's masks 24 and 16).
//!
//! **Bounds.** Every word is checked on its own against the array's length: a load of a word
//! past the end gives 0 and a store to one is dropped; an access that straddles the end keeps
//! its words that are inside. For loads and stores alike the index is clamped to the last word
//! for the access itself, so neither the host nor naga's validation sees an out-of-bounds
//! index: naga refuses a constant index past the end, which the production pipeline's passes
//! make of a `[RZ + imm]` offset (pinned by `local_accesses_at_and_past_the_end`'s constant run
//! and, through the pipeline, `crate::pipeline`'s
//! `a_local_store_at_a_constant_offset_past_the_end_lowers`). A program that uses an array of zero
//! words reads 0 and writes nothing. The word index is at most `u32::MAX >> 2` and the last word
//! of an access three more, so the index arithmetic cannot wrap.
//!
//! The local array's size is the one each reference's environment gives `LocalMemorySize()`
//! ([`crate::pipeline::local_memory_bytes`]: a graphics program's header size plus its CRS
//! size, a compute program's QMD size through the variant key).
//!
//! Where this differs, each pinned by a test in `tests::memory`:
//!
//! - Eden's backend checks nothing (SPIR-V leaves an out-of-bounds access undefined;
//!   `EmitLoadLocal` and `EmitWriteLocal` in `emit_spirv_context_get_set.cpp` index the array
//!   directly), and the interpreter refuses one ([`crate::interp::InterpError::OutOfBounds`]):
//!   here it reads 0 or writes nothing (`local_accesses_at_and_past_the_end`,
//!   `shared_offsets_at_and_past_the_end`, and, translated and run on the interpreter too,
//!   `translated_local_accesses_match_the_interpreter_at_the_edge`).
//! - Eden bounds local memory in its translator instead
//!   (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/load_store_local_shared.cpp`
//!   @ cf654052): `LoadLocal` selects 0 unless the **byte** offset is below
//!   `env.LocalMemorySize()` (`ILessThan(offset, local_memory_size, false)`, each word of a
//!   64- or 128-bit `LDL` checked at its own offset), and `STL` drops a store whose offset is
//!   an immediate not below the size, while a store at a register offset is not bounded (its
//!   `TODO`). Here the bound is per **word** against `ceil(size / 4)` words, for loads and
//!   stores alike. They give the same result for every load, and for every 8-, 16- or 32-bit
//!   store at an immediate offset, except, when the size is not a multiple of 4, an access whose
//!   byte offset lies in the last word at or past the size: Eden's load reads 0 and its
//!   immediate-offset store is dropped, while here both reach the word (the load:
//!   `divergence_a_narrow_load_past_the_byte_size_in_the_last_word_reads_it`; the store:
//!   `divergence_a_narrow_store_past_the_byte_size_in_the_last_word_lands`).
//! - A 64- or 128-bit `STL` at an immediate offset: Eden checks only its first byte offset
//!   against the size and then writes `word_offset + i` for the remaining words with no bound
//!   (`ir.WriteLocal(ir.IAdd(word_offset, ir.Imm32(i)), ...)`), and the translator here emits
//!   the same `WriteLocal`s (`stl` in `translate/memory.rs`). So one whose first word is inside and
//!   whose later words are past the end writes those words out of bounds in Eden, while here
//!   they are dropped and the words inside kept
//!   (`divergence_a_wide_store_at_the_last_word_drops_the_words_past_the_end`, an `STL.128` at
//!   the last of three words).
//! - A store at a register offset past the end writes out of bounds in Eden and is dropped
//!   here (`translated_local_accesses_match_the_interpreter_at_the_edge`, which also pins the
//!   word edge for loads and one-word immediate-offset stores).
//! - Eden's word index is `OpShiftRightArithmetic(offset, 2)`; here it is logical. They differ
//!   only for offsets of 2^31 and up, which are out of bounds either way, except that Eden's
//!   next word after index `0xffff_ffff` wraps to word 0
//!   (`shared_offsets_at_and_past_the_end`).
//! - The interpreter refuses a misaligned shared access; here, as in Eden, the low bits of the
//!   offset choose the field of an 8- or 16-bit load and are ignored by the wider ones.
//! - `WriteSharedU8` and `WriteSharedU16` are not lowered: Eden writes them with a
//!   compare-and-swap loop on the word, which needs the array to be atomic, and the shared
//!   atomics are not lowered either ([`super::LowerError::Operation`]).

use std::num::NonZeroU32;

use naga::{
    AddressSpace, ArraySize, BinaryOperator, Block, Expression, GlobalVariable, Handle, Literal,
    MathFunction, MemoryDecorations, ScalarKind, Span, Statement, TypeInner,
};

use super::builder::{push, ty};
use super::{Interface, LowerError, LowerInputs, Lowering, MEMORY_WORD_BYTES};
use crate::ir::{Opcode, Operation, Type, Value};

/// Whether `opcode` accesses local memory.
pub(super) const fn is_local(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::LoadLocal | Opcode::WriteLocal)
}

/// Whether `opcode` accesses shared memory (the atomics included, which only compute programs
/// may use, though they are not lowered yet).
pub(super) const fn is_shared(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::LoadSharedU8
            | Opcode::LoadSharedS8
            | Opcode::LoadSharedU16
            | Opcode::LoadSharedS16
            | Opcode::LoadSharedU32
            | Opcode::LoadSharedU64
            | Opcode::LoadSharedU128
            | Opcode::WriteSharedU8
            | Opcode::WriteSharedU16
            | Opcode::WriteSharedU32
            | Opcode::WriteSharedU64
            | Opcode::WriteSharedU128
            | Opcode::SharedAtomicIAdd32
            | Opcode::SharedAtomicSMin32
            | Opcode::SharedAtomicUMin32
            | Opcode::SharedAtomicSMax32
            | Opcode::SharedAtomicUMax32
            | Opcode::SharedAtomicInc32
            | Opcode::SharedAtomicDec32
            | Opcode::SharedAtomicAnd32
            | Opcode::SharedAtomicOr32
            | Opcode::SharedAtomicXor32
            | Opcode::SharedAtomicExchange32
            | Opcode::SharedAtomicExchange64
            | Opcode::SharedAtomicExchange32x2
    )
}

/// The words of a memory of `bytes` bytes: `ceil(bytes / 4)` (Eden's `DivCeil(size, 4U)`).
pub(crate) const fn words(bytes: u64) -> u64 {
    bytes.div_ceil(MEMORY_WORD_BYTES as u64)
}

/// One declared array: the global and its length in words.
#[derive(Clone, Copy)]
struct Array {
    global: Handle<GlobalVariable>,
    words: u32,
}

/// The arrays a function uses, if declared.
#[derive(Default)]
pub(super) struct Memory {
    local: Option<Array>,
    shared: Option<Array>,
    /// Whether the function uses each space, so an access to an undeclared (empty) array is
    /// told apart from a bookkeeping fault.
    uses_local: bool,
    uses_shared: bool,
}

/// What a shared or local access reads or writes.
#[derive(Clone, Copy)]
enum Access {
    /// A 32-bit word, or `count` consecutive words as a vector.
    Words { count: u32 },
    /// An 8- or 16-bit field of one word, sign- or zero-extended.
    Field { bits: u32, signed: bool },
}

/// The access an opcode makes and whether it is a store.
const fn access(opcode: Opcode) -> Option<(Access, bool)> {
    let one = Access::Words { count: 1 };
    Some(match opcode {
        Opcode::LoadLocal | Opcode::LoadSharedU32 => (one, false),
        Opcode::WriteLocal | Opcode::WriteSharedU32 => (one, true),
        Opcode::LoadSharedU64 => (Access::Words { count: 2 }, false),
        Opcode::LoadSharedU128 => (Access::Words { count: 4 }, false),
        Opcode::WriteSharedU64 => (Access::Words { count: 2 }, true),
        Opcode::WriteSharedU128 => (Access::Words { count: 4 }, true),
        Opcode::LoadSharedU8 => (
            Access::Field {
                bits: 8,
                signed: false,
            },
            false,
        ),
        Opcode::LoadSharedS8 => (
            Access::Field {
                bits: 8,
                signed: true,
            },
            false,
        ),
        Opcode::LoadSharedU16 => (
            Access::Field {
                bits: 16,
                signed: false,
            },
            false,
        ),
        Opcode::LoadSharedS16 => (
            Access::Field {
                bits: 16,
                signed: true,
            },
            false,
        ),
        _ => return None,
    })
}

/// Eden's field-offset masks (`ExtractArgs(ctx, offset, 24, 8)` and `(16, 16)`): the bit of
/// an 8-bit field is `(offset << 3) & 24`, of a 16-bit field `(offset << 3) & 16`.
const fn field_mask(bits: u32) -> u32 {
    if bits == 8 { 24 } else { 16 }
}

/// The shift from a byte offset to its bit (`offset << 3`).
const BYTE_TO_BIT_SHIFT: u32 = 3;
/// The shift from a byte offset to its word (`offset >> 2`).
const BYTE_TO_WORD_SHIFT: u32 = 2;

impl Lowering {
    /// Declares the arrays the function uses, sized from `inputs` (checked against their caps
    /// before this).
    pub(super) fn declare_memory(
        &mut self,
        interface: &Interface,
        inputs: &LowerInputs,
    ) -> Result<(), LowerError> {
        self.memory.uses_local = interface.memory.local;
        self.memory.uses_shared = interface.memory.shared;
        if interface.memory.local {
            self.memory.local = self.array(
                "local_memory",
                AddressSpace::Private,
                words(inputs.local_memory_bytes),
            )?;
        }
        if interface.memory.shared {
            self.memory.shared = self.array(
                "shared_memory",
                AddressSpace::WorkGroup,
                words(u64::from(inputs.shared_memory_bytes)),
            )?;
        }
        Ok(())
    }

    /// A global array of `words` `u32`s in `space`, or `None` for zero words.
    fn array(
        &mut self,
        name: &str,
        space: AddressSpace,
        words: u64,
    ) -> Result<Option<Array>, LowerError> {
        let Some(length) = u32::try_from(words).ok().and_then(NonZeroU32::new) else {
            return if words == 0 {
                Ok(None)
            } else {
                Err(LowerError::Internal(format!("{words} words of {name}")))
            };
        };
        let array_ty = ty(
            &mut self.module,
            TypeInner::Array {
                base: self.types.u32,
                size: ArraySize::Constant(length),
                stride: MEMORY_WORD_BYTES,
            },
        );
        let global = self.module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space,
                binding: None,
                ty: array_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        Ok(Some(Array {
            global,
            words: length.get(),
        }))
    }

    /// Lowers a local or shared memory access (see the module docs).
    pub(super) fn memory_op(
        &mut self,
        body: &mut Block,
        at: Value,
        op: &Operation,
    ) -> Result<Option<Handle<Expression>>, LowerError> {
        let opcode = op.opcode;
        let Some((access, store)) = access(opcode) else {
            return Err(LowerError::Operation { at, opcode });
        };
        let local = is_local(opcode);
        let (array, used) = if local {
            (self.memory.local, self.memory.uses_local)
        } else {
            (self.memory.shared, self.memory.uses_shared)
        };
        if !used {
            return Err(LowerError::Internal(format!("{at}: memory not collected")));
        }
        let address = self.arg(body, at, op, 0)?;
        let word = if local {
            address
        } else {
            let shift = self.lit(body, Literal::U32(BYTE_TO_WORD_SHIFT));
            self.binary_expr(body, BinaryOperator::ShiftRight, address, shift)
        };
        match access {
            Access::Words { count } if store => {
                let value = self.arg(body, at, op, 1)?;
                for index in 0..count {
                    let part = if count == 1 {
                        value
                    } else {
                        self.component(body, value, index)
                    };
                    self.store_word(body, array, word, index, part);
                }
                Ok(None)
            }
            Access::Words { count } => {
                let mut parts = Vec::new();
                for index in 0..count {
                    parts.push(self.load_word(body, array, word, index));
                }
                if count == 1 {
                    return Ok(parts.first().copied());
                }
                let vector = if count == 2 { Type::U32x2 } else { Type::U32x4 };
                let ty = self.ir_type(vector)?;
                Ok(Some(self.expr(
                    body,
                    Expression::Compose {
                        ty,
                        components: parts,
                    },
                )))
            }
            Access::Field { .. } if store => Err(LowerError::Operation { at, opcode }),
            Access::Field { bits, signed } => {
                let loaded = self.load_word(body, array, word, 0);
                let shift = self.lit(body, Literal::U32(BYTE_TO_BIT_SHIFT));
                let bit = self.binary_expr(body, BinaryOperator::ShiftLeft, address, shift);
                let mask = self.lit(body, Literal::U32(field_mask(bits)));
                let bit = self.binary_expr(body, BinaryOperator::And, bit, mask);
                let count = self.lit(body, Literal::U32(bits));
                let source = if signed {
                    self.cast(body, loaded, ScalarKind::Sint, None)
                } else {
                    loaded
                };
                let field = self.math(body, MathFunction::ExtractBits, source, &[bit, count]);
                Ok(Some(if signed {
                    self.cast(body, field, ScalarKind::Uint, None)
                } else {
                    field
                }))
            }
        }
    }

    fn binary_expr(
        &mut self,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(body, Expression::Binary { op, left, right })
    }

    /// `word + offset` and whether it is inside `array` (`offset` is at most 3, and `word` at
    /// most `u32::MAX >> 2` for shared memory; a local index can be any `u32`, so the sum is
    /// only formed for `offset` 0 there).
    fn index(
        &mut self,
        body: &mut Block,
        array: Array,
        word: Handle<Expression>,
        offset: u32,
    ) -> (Handle<Expression>, Handle<Expression>) {
        let index = if offset == 0 {
            word
        } else {
            let offset = self.lit(body, Literal::U32(offset));
            self.binary_expr(body, BinaryOperator::Add, word, offset)
        };
        let length = self.lit(body, Literal::U32(array.words));
        let inside = self.binary_expr(body, BinaryOperator::Less, index, length);
        (index, inside)
    }

    /// `index` clamped to the last word of `array`, the index the access itself takes, so
    /// neither the host nor naga's validation (which refuses a constant index past the end)
    /// sees one out of bounds; the access is still made only when `index` is inside.
    fn clamped(
        &mut self,
        body: &mut Block,
        array: Array,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let last = self.lit(body, Literal::U32(array.words - 1));
        self.math(body, MathFunction::Min, index, &[last])
    }

    /// Word `word + offset` of `array`, or 0 past its end or when there is no array.
    fn load_word(
        &mut self,
        body: &mut Block,
        array: Option<Array>,
        word: Handle<Expression>,
        offset: u32,
    ) -> Handle<Expression> {
        let zero = self.lit(body, Literal::U32(0));
        let Some(array) = array else {
            return zero;
        };
        let (index, inside) = self.index(body, array, word, offset);
        let clamped = self.clamped(body, array, index);
        let base = self.expr(body, Expression::GlobalVariable(array.global));
        let pointer = self.expr(
            body,
            Expression::Access {
                base,
                index: clamped,
            },
        );
        let loaded = self.expr(body, Expression::Load { pointer });
        self.select(body, inside, loaded, zero)
    }

    /// Stores `value` to word `word + offset` of `array` when it is inside; nothing otherwise.
    fn store_word(
        &mut self,
        body: &mut Block,
        array: Option<Array>,
        word: Handle<Expression>,
        offset: u32,
        value: Handle<Expression>,
    ) {
        let Some(array) = array else {
            return;
        };
        let (index, inside) = self.index(body, array, word, offset);
        let clamped = self.clamped(body, array, index);
        let mut accept = Block::new();
        let base = self.expr(&mut accept, Expression::GlobalVariable(array.global));
        let pointer = self.expr(
            &mut accept,
            Expression::Access {
                base,
                index: clamped,
            },
        );
        push(&mut accept, Statement::Store { pointer, value });
        push(
            body,
            Statement::If {
                condition: inside,
                accept,
                reject: Block::new(),
            },
        );
    }
}
