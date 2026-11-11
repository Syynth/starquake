//! A tiny label-and-fixup assembler, so a test can write a loop.
//!
//! Instructions are appended one word at a time; branches to a [`Label`] are emitted with a zero
//! offset and patched by [`Assembler::finish`], once every label has been bound. A label may be
//! bound before or after the branches that use it.

use super::{EncodeError, INSN_SIZE, b, bl, cbnz, cbz};

/// A position in the program, created by [`Assembler::label`] and placed by [`Assembler::bind`].
///
/// A label belongs to the assembler that created it; using it with another one is a bug the
/// assembler can only detect when the label's index does not exist there ([`AsmError::UnknownLabel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Label(usize);

/// Why a program could not be assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AsmError {
    /// An operand does not fit its field (a register number, say).
    #[error(transparent)]
    Encode(#[from] EncodeError),
    /// A branch to a label that was never bound.
    #[error("label {label} is used by the branch at word {at} but never bound")]
    UnboundLabel {
        /// The label's index in creation order.
        label: usize,
        /// The word index of the branch.
        at: usize,
    },
    /// A label bound twice.
    #[error("label {label} is already bound")]
    AlreadyBound {
        /// The label's index in creation order.
        label: usize,
    },
    /// A label that this assembler did not create.
    #[error("label {label} does not belong to this assembler")]
    UnknownLabel {
        /// The label's index.
        label: usize,
    },
    /// A branch whose distance to its label does not fit the instruction.
    #[error("the branch at word {at} cannot reach label {label}: {source}")]
    Unreachable {
        /// The word index of the branch.
        at: usize,
        /// The label's index in creation order.
        label: usize,
        /// Why the distance does not encode.
        source: EncodeError,
    },
    /// The program has more words than a byte offset can express.
    #[error("the program is too long to address")]
    TooLong,
}

/// Which branch instruction a fixup patches.
#[derive(Debug, Clone, Copy)]
enum Kind {
    B,
    Bl,
    Cbz(u8),
    Cbnz(u8),
}

impl Kind {
    /// Encodes the instruction with `offset` bytes to its target.
    fn encode(self, offset: i64) -> Result<u32, EncodeError> {
        match self {
            Self::B => b(offset),
            Self::Bl => bl(offset),
            Self::Cbz(rt) => cbz(rt, offset),
            Self::Cbnz(rt) => cbnz(rt, offset),
        }
    }
}

/// A branch waiting for its label's position.
#[derive(Debug, Clone, Copy)]
struct Fixup {
    /// Word index of the branch.
    at: usize,
    label: Label,
    kind: Kind,
}

/// Builds a sequence of A64 words with forward and backward branches to labels.
///
/// ```
/// use starquake_testkit::a64::{self, Assembler};
///
/// // x0 = 3; do { x0 -= 1 } while x0 != 0; ret
/// let mut asm = Assembler::new();
/// let top = asm.label();
/// asm.emit(a64::movz(0, 3, 0)?);
/// asm.bind(top)?;
/// asm.emit(a64::sub_imm(0, 0, 1)?);
/// asm.cbnz(0, top)?;
/// asm.emit(a64::ret());
/// assert_eq!(asm.finish()?.len(), 4);
/// # Ok::<(), starquake_testkit::a64::AsmError>(())
/// ```
#[derive(Debug, Default)]
pub struct Assembler {
    words: Vec<u32>,
    /// Word index each label is bound to, in creation order.
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
}

impl Assembler {
    /// An empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of words emitted so far; the next word's index.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// True when nothing has been emitted.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Appends one encoded instruction word.
    pub fn emit(&mut self, word: u32) {
        self.words.push(word);
    }

    /// Creates a label that is not yet placed.
    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    /// Places `label` at the next word to be emitted.
    pub fn bind(&mut self, label: Label) -> Result<(), AsmError> {
        let here = self.words.len();
        let slot = self
            .labels
            .get_mut(label.0)
            .ok_or(AsmError::UnknownLabel { label: label.0 })?;
        if slot.is_some() {
            return Err(AsmError::AlreadyBound { label: label.0 });
        }
        *slot = Some(here);
        Ok(())
    }

    /// Appends `b label`.
    pub fn b(&mut self, label: Label) -> Result<(), AsmError> {
        self.branch(Kind::B, label)
    }

    /// Appends `bl label`.
    pub fn bl(&mut self, label: Label) -> Result<(), AsmError> {
        self.branch(Kind::Bl, label)
    }

    /// Appends `cbz xt, label`.
    pub fn cbz(&mut self, rt: u8, label: Label) -> Result<(), AsmError> {
        self.branch(Kind::Cbz(rt), label)
    }

    /// Appends `cbnz xt, label`.
    pub fn cbnz(&mut self, rt: u8, label: Label) -> Result<(), AsmError> {
        self.branch(Kind::Cbnz(rt), label)
    }

    /// Appends a branch with a zero offset (which also checks its register) and records the fixup.
    fn branch(&mut self, kind: Kind, label: Label) -> Result<(), AsmError> {
        if label.0 >= self.labels.len() {
            return Err(AsmError::UnknownLabel { label: label.0 });
        }
        let word = kind.encode(0)?;
        self.fixups.push(Fixup {
            at: self.words.len(),
            label,
            kind,
        });
        self.words.push(word);
        Ok(())
    }

    /// Patches every branch and returns the program's words.
    ///
    /// Fails if a branch uses a label that was never bound, or one too far away for the
    /// instruction.
    pub fn finish(mut self) -> Result<Vec<u32>, AsmError> {
        for fixup in &self.fixups {
            let target = self.labels.get(fixup.label.0).copied().flatten().ok_or(
                AsmError::UnboundLabel {
                    label: fixup.label.0,
                    at: fixup.at,
                },
            )?;
            let offset = word_distance(fixup.at, target)?;
            let word = fixup
                .kind
                .encode(offset)
                .map_err(|source| AsmError::Unreachable {
                    at: fixup.at,
                    label: fixup.label.0,
                    source,
                })?;
            if let Some(slot) = self.words.get_mut(fixup.at) {
                *slot = word;
            }
        }
        Ok(self.words)
    }
}

/// The byte distance from the word at index `from` to the word at index `to`.
fn word_distance(from: usize, to: usize) -> Result<i64, AsmError> {
    let words = |index: usize| i64::try_from(index).map_err(|_| AsmError::TooLong);
    words(to)?
        .checked_sub(words(from)?)
        .and_then(|distance| distance.checked_mul(i64::from(INSN_SIZE)))
        .ok_or(AsmError::TooLong)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a64::{COMPARE_BRANCH_IMM_BITS, MAX_REG, add_imm, movz, nop, ret, sub_imm};

    /// Expected words come from assembling the same program (the assembly is in the comments)
    /// with `clang -target arm64-apple-macos -c` and reading it back with `llvm-objdump -d`, once.
    #[test]
    fn a_countdown_loop_and_forward_and_backward_branches_match_a_real_assembler() {
        let mut asm = Assembler::new();
        let (looping, skip, callee, join, first, last, middle) = (
            asm.label(),
            asm.label(),
            asm.label(),
            asm.label(),
            asm.label(),
            asm.label(),
            asm.label(),
        );
        asm.emit(movz(0, 3, 0).unwrap()); //     movz x0, #3
        asm.bind(looping).unwrap(); //           L:
        asm.emit(sub_imm(0, 0, 1).unwrap()); //  sub x0, x0, #1
        asm.cbnz(0, looping).unwrap(); //        cbnz x0, L
        asm.emit(ret()); //                      ret
        asm.b(skip).unwrap(); //                 b skip
        asm.emit(nop()); //                      nop
        asm.bind(skip).unwrap(); //              skip:
        asm.emit(nop()); //                      nop
        asm.bl(callee).unwrap(); //                   bl f
        asm.cbz(5, join).unwrap(); //               cbz x5, g
        asm.emit(nop()); //                      nop
        asm.bind(callee).unwrap(); //                 f:
        asm.emit(nop()); //                      nop
        asm.bind(join).unwrap(); //                 g:
        asm.emit(ret()); //                      ret
        asm.bind(first).unwrap(); //                 h:
        asm.b(last).unwrap(); //                    b i
        asm.bind(middle).unwrap(); //                 j:
        asm.b(first).unwrap(); //                    b h
        asm.bind(last).unwrap(); //                 i:
        asm.bl(middle).unwrap(); //                   bl j
        let expected = [
            0xD280_0060, // movz x0, #3
            0xD100_0400, // sub x0, x0, #1
            0xB5FF_FFE0, // cbnz x0, L
            0xD65F_03C0, // ret
            0x1400_0002, // b skip
            0xD503_201F, // nop
            0xD503_201F, // nop
            0x9400_0003, // bl f
            0xB400_0065, // cbz x5, g
            0xD503_201F, // nop
            0xD503_201F, // nop
            0xD65F_03C0, // ret
            0x1400_0002, // b i
            0x17FF_FFFF, // b h
            0x97FF_FFFF, // bl j
        ];
        assert_eq!(asm.finish().unwrap(), expected);
    }

    #[test]
    fn len_counts_emitted_words_and_labels_do_not_occupy_one() {
        let mut asm = Assembler::new();
        assert!(asm.is_empty());
        let label = asm.label();
        asm.bind(label).unwrap();
        assert_eq!(asm.len(), 0);
        asm.emit(nop());
        asm.b(label).unwrap();
        assert_eq!((asm.len(), asm.is_empty()), (2, false));
        assert_eq!(asm.finish().unwrap(), [nop(), 0x17FF_FFFF]);
    }

    #[test]
    fn a_branch_to_an_unbound_label_is_an_error() {
        let mut asm = Assembler::new();
        asm.emit(nop());
        let first = asm.label();
        let second = asm.label();
        asm.bind(first).unwrap();
        asm.b(second).unwrap();
        assert_eq!(
            asm.finish(),
            Err(AsmError::UnboundLabel { label: 1, at: 1 })
        );
    }

    #[test]
    fn an_unused_unbound_label_is_fine() {
        let mut asm = Assembler::new();
        let _unused = asm.label();
        asm.emit(ret());
        assert_eq!(asm.finish().unwrap(), [ret()]);
    }

    #[test]
    fn binding_twice_is_an_error() {
        let mut asm = Assembler::new();
        let label = asm.label();
        asm.bind(label).unwrap();
        assert_eq!(asm.bind(label), Err(AsmError::AlreadyBound { label: 0 }));
    }

    #[test]
    fn a_label_from_another_assembler_is_an_error() {
        let mut other = Assembler::new();
        let _ = other.label();
        let foreign = other.label();
        let mut asm = Assembler::new();
        assert_eq!(asm.bind(foreign), Err(AsmError::UnknownLabel { label: 1 }));
        assert_eq!(asm.b(foreign), Err(AsmError::UnknownLabel { label: 1 }));
        assert!(asm.is_empty());
    }

    #[test]
    fn a_bad_register_fails_when_the_branch_is_emitted() {
        let mut asm = Assembler::new();
        let label = asm.label();
        assert_eq!(
            asm.cbz(MAX_REG + 1, label),
            Err(AsmError::Encode(EncodeError::Register { reg: MAX_REG + 1 }))
        );
        assert!(asm.is_empty());
    }

    #[test]
    fn a_branch_beyond_the_compare_branch_reach_is_an_error() {
        // `cbz` has a signed word offset of COMPARE_BRANCH_IMM_BITS bits: 2^(bits - 1) - 1 words
        // forward at most.
        let max_words = (1usize << (COMPARE_BRANCH_IMM_BITS - 1)) - 1;
        for (padding, reachable) in [(max_words - 1, true), (max_words, false)] {
            let mut asm = Assembler::new();
            let end = asm.label();
            asm.cbz(0, end).unwrap();
            for _ in 0..padding {
                asm.emit(add_imm(0, 0, 0).unwrap());
            }
            asm.bind(end).unwrap();
            asm.emit(ret());
            let result = asm.finish();
            if reachable {
                assert_eq!(result.unwrap().len(), padding + 2);
            } else {
                let distance = i64::try_from(padding + 1).unwrap() * i64::from(INSN_SIZE);
                let limit = i64::from(INSN_SIZE) << (COMPARE_BRANCH_IMM_BITS - 1);
                assert_eq!(
                    result,
                    Err(AsmError::Unreachable {
                        at: 0,
                        label: 0,
                        source: EncodeError::Offset {
                            offset: distance,
                            min: -limit,
                            max: limit - i64::from(INSN_SIZE),
                        },
                    })
                );
            }
        }
    }

    #[test]
    fn a_branch_beyond_the_backward_compare_branch_reach_is_an_error() {
        // The mirror of the forward test: a `cbz` at word `p` back to word 0 is `p` words
        // backward, and the lowest encodable distance is -2^(bits - 1) words.
        let max_back_words = 1usize << (COMPARE_BRANCH_IMM_BITS - 1);
        for (padding, reachable) in [(max_back_words, true), (max_back_words + 1, false)] {
            let mut asm = Assembler::new();
            let start = asm.label();
            asm.bind(start).unwrap();
            for _ in 0..padding {
                asm.emit(add_imm(0, 0, 0).unwrap());
            }
            asm.cbz(0, start).unwrap();
            let result = asm.finish();
            if reachable {
                assert_eq!(result.unwrap().len(), padding + 1);
            } else {
                let distance = -(i64::try_from(padding).unwrap() * i64::from(INSN_SIZE));
                let limit = i64::from(INSN_SIZE) << (COMPARE_BRANCH_IMM_BITS - 1);
                assert_eq!(
                    result,
                    Err(AsmError::Unreachable {
                        at: padding,
                        label: 0,
                        source: EncodeError::Offset {
                            offset: distance,
                            min: -limit,
                            max: limit - i64::from(INSN_SIZE),
                        },
                    })
                );
            }
        }
    }

    #[test]
    fn an_encode_error_is_reported_once_in_the_error_chain() {
        use std::error::Error as _;
        let inner = EncodeError::Register { reg: MAX_REG + 1 };
        let outer = AsmError::Encode(inner);
        assert_eq!(outer.to_string(), inner.to_string());
        assert!(outer.source().is_none());
    }
}
