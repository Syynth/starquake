//! The intermediate representation (slice 5 of the spec's port slices, the IR core).
//!
//! The owner's decision 1 (2026-11-12): a new Rust IR with Eden's operation set and pass
//! behaviour, stored in arenas and referred to by typed ids, with its own register class for
//! uniform registers, a text form that prints and parses, and a verifier (spec, "The
//! intermediate representation"). Eden's IR is ported by behaviour from
//! `eden/src/shader_recompiler/frontend/ir/` @ cf654052 (GPL-3.0-or-later; the verifier's checks
//! from `ir_opt/verification_pass.cpp`, GPL-2.0-or-later):
//!
//! - **Storage.** A [`Function`] owns an arena of instructions and an arena of blocks. An
//!   instruction's id is a [`Value`], its index in the arena; a block's id is a [`BlockId`].
//!   Blocks list the ids of their instructions in execution order. Ids are handed out in creation
//!   order and every iteration is in id or list order, so the same input always prints the same
//!   text.
//! - **Operations** ([`Opcode`]) are Eden's table (`opcodes.inc`), with the differences listed
//!   in the `opcode` module: `Phi` is its own kind of instruction ([`Inst::Phi`]), three
//!   rescaling-only opcodes are left out, and the uniform register class is added
//!   ([`Opcode::GetUniformRegister`] and its siblings, [`UniformReg`], [`UniformPred`]).
//! - **Operands** ([`Operand`]) are values or immediates ([`Imm`]), as Eden's `IR::Value` is an
//!   instruction or an immediate. Types ([`Type`]) are Eden's, plus the uniform name types.
//! - **Modifiers** ([`Flags`]) are typed: floating-point controls or texture information, as the
//!   opcode table says ([`Opcode::flags_kind`]).
//! - **Control flow** has two forms. Before structuring, a function is a control-flow graph:
//!   each block ends in a [`Terminator`] and block 0 is the entry. After structuring, the
//!   function carries a tree of [`Node`]s over its blocks (`if`, `loop` with a repeat
//!   condition, conditional `break`, `return`, `unreachable`), which is Eden's flat
//!   `AbstractSyntaxList` (`abstract_syntax_list.h`) nested into a tree, and its blocks have no
//!   terminators. Before SSA, guest registers are read and written with `GetRegister` and
//!   `SetRegister` (and the predicate and uniform equivalents), as in Eden.
//! - **Building** goes through [`Builder`], which checks each instruction's arguments and flags
//!   as it appends it; [`Function::verify`] checks a whole function (types, SSA dominance and
//!   structure). The text form is [`Function`]'s `Display` and [`Function::parse`].
//!
//! Where this IR is stricter than Eden, each pinned by a test in `tests`:
//!
//! - Reading or writing `RZ`, `URZ`, `PT` or `UPT` through the get and set opcodes is an
//!   error (`ReservedName`); Eden's emitter never emits a predicate access to `PT`
//!   (`ir_emitter.cpp`, `GetPred`, `SetPred`) and folds `RZ` in its SSA pass. Pinned by
//!   `tests::reserved_names_are_refused`.
//! - Values must be defined in a block that dominates the use, not only earlier in the block
//!   list (Eden's `ValidateForwardDeclarations`); a phi's incoming values are checked against
//!   its type, which Eden skips, and in a control-flow graph against the block's predecessors.
//!   Pinned by `verify::tests::dominance_is_checked_not_block_order` and
//!   `verify::tests::phi_checks`.
//! - Eden's use-count check (`ValidateUses`) has no counterpart: this IR keeps no use counts.
//! - A structured tree may nest at most [`MAX_NESTING`] levels; Eden has no bound. The bound
//!   is a starquake choice (a guess at what real shaders need, to be checked against the
//!   structurizer's output, slice 11) that keeps the verifier's recursion bounded. Pinned by
//!   `verify::tests::nesting_is_bounded` and `parse::tests::nesting_is_bounded`.
//!
//! Slice 1's minimal IR is replaced: its instructions are the Eden opcodes of the same names
//! (`Imm32` is now an immediate operand), and the skeleton lowers to the same naga modules
//! (`lower::tests::skeleton_modules_are_unchanged`).

mod builder;
mod modifiers;
mod opcode;
mod operand;
mod parse;
mod print;
#[cfg(test)]
mod tests;
mod types;
mod verify;

use std::fmt::{self, Display, Formatter};

pub use builder::{BuildError, Builder};
pub use modifiers::{
    Flags, FmzMode, FpControl, FpRounding, ImageFormat, NO_CONTRACTION, TEXTURE_TWO_BIT_MAX,
    TextureInfo, TextureType,
};
pub use opcode::{FlagsKind, Opcode};
pub use operand::{ATTRIBUTE_WORD_BYTES, Attribute, Imm, Operand, Patch, UniformPred, UniformReg};
pub use parse::{ParseError, ParseErrorKind};
pub use types::Type;
pub use verify::{Site, VerifyError};

/// How deeply structured nodes may nest (see the module docs).
pub const MAX_NESTING: usize = 256;

/// The id of an instruction and of the value it produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);

impl Value {
    /// The instruction's index in its function's arena.
    pub const fn index(self) -> u32 {
        self.0
    }

    fn slot(self) -> Option<usize> {
        usize::try_from(self.0).ok()
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// The id of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl BlockId {
    /// The block's index in its function's arena.
    pub const fn index(self) -> u32 {
        self.0
    }

    fn slot(self) -> Option<usize> {
        usize::try_from(self.0).ok()
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "block{}", self.0)
    }
}

/// An operation with its arguments and modifiers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operation {
    /// The opcode.
    pub opcode: Opcode,
    /// The arguments, as many as [`Opcode::args`] names.
    pub args: Vec<Operand>,
    /// The modifiers, of the kind [`Opcode::flags_kind`] names.
    pub flags: Flags,
}

impl Operation {
    /// The operation with the default flags of its opcode's kind.
    pub fn new(opcode: Opcode, args: Vec<Operand>) -> Self {
        Self {
            opcode,
            args,
            flags: Flags::default_of(opcode.flags_kind()),
        }
    }
}

/// A phi: the value that arrives from each predecessor block (Eden's `Phi`, whose type Eden
/// keeps in the flags word).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Phi {
    /// The type of the phi and of every incoming value.
    pub ty: Type,
    /// The value that arrives from each block.
    pub incoming: Vec<(BlockId, Operand)>,
}

/// One instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    /// An operation.
    Op(Operation),
    /// A phi.
    Phi(Phi),
}

impl Inst {
    /// The type of the instruction's result as its opcode states it: [`Type::Void`] for none,
    /// [`Type::Opaque`] for `Identity`, whose value has its argument's type
    /// ([`Function::type_of`] resolves it).
    pub const fn result(&self) -> Type {
        match self {
            Self::Op(op) => op.opcode.result(),
            Self::Phi(phi) => phi.ty,
        }
    }

    /// Whether this is a phi.
    pub const fn is_phi(&self) -> bool {
        matches!(self, Self::Phi(_))
    }

    /// An arena slot no block lists: what a parsed function holds at an id the text skips
    /// (Eden's invalidated instruction, `Inst::Invalidate`, is a `Void` with no arguments).
    fn detached() -> Self {
        Self::Op(Operation::new(Opcode::Void, Vec::new()))
    }
}

/// How control leaves a block of a control-flow graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Terminator {
    /// None: not set yet, or the function is structured and its tree says where control goes.
    #[default]
    None,
    /// Continue at a block.
    Jump(BlockId),
    /// Continue at `then` when `cond` (a `u1`) is true, else at `otherwise`.
    Branch {
        /// The condition.
        cond: Operand,
        /// Where control goes when it is true.
        then: BlockId,
        /// Where control goes when it is false.
        otherwise: BlockId,
    },
    /// Return from the program.
    Return,
    /// Control never reaches the end of this block.
    Unreachable,
}

impl Terminator {
    /// The blocks control can go to, in order (a branch to one block twice lists it twice).
    pub fn successors(&self) -> impl Iterator<Item = BlockId> {
        let (first, second) = match *self {
            Self::Jump(target) => (Some(target), None),
            Self::Branch {
                then, otherwise, ..
            } => (Some(then), Some(otherwise)),
            Self::None | Self::Return | Self::Unreachable => (None, None),
        };
        first.into_iter().chain(second)
    }
}

/// A structured statement (Eden's `AbstractSyntaxNode`, nested).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// Run a block's instructions.
    Block(BlockId),
    /// Run `body` when `cond` (a `u1`) is true (Eden's `If` … `EndIf`).
    If {
        /// The condition.
        cond: Operand,
        /// The statements.
        body: Vec<Node>,
    },
    /// Run `body`, then again while `repeat` (a `u1`, evaluated at the end of the body) is true
    /// (Eden's `Loop` … `Repeat`).
    Loop {
        /// The statements.
        body: Vec<Node>,
        /// The condition to run the body again.
        repeat: Operand,
    },
    /// Leave the innermost loop when `cond` (a `u1`) is true (Eden's `Break`).
    Break {
        /// The condition.
        cond: Operand,
    },
    /// Return from the program.
    Return,
    /// Control never reaches this point.
    Unreachable,
}

/// A basic block: instruction ids in execution order and, in a control-flow graph, how control
/// leaves it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Block {
    /// The instructions, phis first.
    pub insts: Vec<Value>,
    /// How control leaves the block ([`Terminator::None`] in a structured function).
    pub terminator: Terminator,
}

/// A function: arenas of instructions and blocks, and, once structured, the tree over its
/// blocks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Function {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    structure: Option<Vec<Node>>,
}

impl Default for Function {
    fn default() -> Self {
        Self::new()
    }
}

impl Function {
    /// A control-flow graph with one empty, unterminated block: the entry.
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            blocks: vec![Block::default()],
            structure: None,
        }
    }

    /// The entry block, block 0.
    pub const fn entry(&self) -> BlockId {
        BlockId(0)
    }

    /// Adds an empty, unterminated block.
    pub fn add_block(&mut self) -> Result<BlockId, BuildError> {
        let id = BlockId(u32::try_from(self.blocks.len()).map_err(|_| BuildError::Full)?);
        self.blocks.push(Block::default());
        Ok(id)
    }

    /// Appends `inst` to the end of `block` without checking it ([`Builder`] checks), and
    /// returns its id.
    pub fn push(&mut self, block: BlockId, inst: Inst) -> Result<Value, BuildError> {
        let id = Value(u32::try_from(self.insts.len()).map_err(|_| BuildError::Full)?);
        let list = block
            .slot()
            .and_then(|slot| self.blocks.get_mut(slot))
            .ok_or(BuildError::NoBlock(block))?;
        list.insts.push(id);
        self.insts.push(inst);
        Ok(id)
    }

    /// The instruction `value` names.
    pub fn inst(&self, value: Value) -> Option<&Inst> {
        value.slot().and_then(|slot| self.insts.get(slot))
    }

    /// The instruction `value` names, to change it.
    pub fn inst_mut(&mut self, value: Value) -> Option<&mut Inst> {
        value.slot().and_then(|slot| self.insts.get_mut(slot))
    }

    /// The block `block` names.
    pub fn block(&self, block: BlockId) -> Option<&Block> {
        block.slot().and_then(|slot| self.blocks.get(slot))
    }

    /// The block `block` names, to change it.
    pub fn block_mut(&mut self, block: BlockId) -> Option<&mut Block> {
        block.slot().and_then(|slot| self.blocks.get_mut(slot))
    }

    /// The blocks with their ids, entry first.
    pub fn blocks(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        (0_u32..).map(BlockId).zip(self.blocks.iter())
    }

    /// How many blocks there are.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Every instruction in the arena with its id, in creation order, whether a block lists it
    /// or not.
    pub fn insts(&self) -> impl Iterator<Item = (Value, &Inst)> {
        (0_u32..).map(Value).zip(self.insts.iter())
    }

    /// The instructions the blocks list, block by block in arena order, each block's in its
    /// order. Ids with no instruction are skipped (the verifier reports them).
    pub fn listed(&self) -> impl Iterator<Item = (Value, &Inst)> {
        self.blocks.iter().flat_map(move |block| {
            block
                .insts
                .iter()
                .filter_map(move |&value| self.inst(value).map(|inst| (value, inst)))
        })
    }

    /// The structured tree, if the function has been structured.
    pub fn structure(&self) -> Option<&[Node]> {
        self.structure.as_deref()
    }

    /// Makes the function structured with `nodes` as its tree, or, with `None`, a control-flow
    /// graph again. Terminators are left as they are; the verifier checks they fit the form.
    pub fn set_structure(&mut self, nodes: Option<Vec<Node>>) {
        self.structure = nodes;
    }

    /// The type of `operand`: an immediate's own type, or the result type of the instruction a
    /// value names, with `Identity` resolved to its argument's type (Eden's `Value::Type`).
    /// `None` if the value names no instruction, or an `Identity` chain does not end at another
    /// instruction or an immediate (a cycle, which only a malformed function has); the chain is
    /// followed at most once per instruction in the arena.
    pub fn type_of(&self, operand: Operand) -> Option<Type> {
        let mut current = operand;
        for _ in 0..=self.insts.len() {
            let value = match current {
                Operand::Imm(imm) => return Some(imm.ty()),
                Operand::Value(value) => value,
            };
            match self.inst(value)? {
                Inst::Op(op) if op.opcode == Opcode::Identity => {
                    current = *op.args.first()?;
                }
                inst => return Some(inst.result()),
            }
        }
        None
    }

    /// Replaces the arena slot `value` with `inst`, growing the arena with detached `Void`
    /// slots up to it (the parser places instructions at the ids the text gives).
    fn place(&mut self, value: Value, inst: Inst) -> Option<()> {
        let slot = value.slot()?;
        if self.insts.len() <= slot {
            self.insts.resize_with(slot.checked_add(1)?, Inst::detached);
        }
        *self.insts.get_mut(slot)? = inst;
        Some(())
    }
}
