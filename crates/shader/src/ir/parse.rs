//! The text form's parser, for tests and hand-written IR (spec: "a parser for test input, so
//! golden files are readable and a pass can be tested from hand-written IR").
//!
//! It reads the format the printer writes (see the `print` module docs), line by line, and also
//! accepts blank lines, `//` comments, a missing result type on an operation, and integers and
//! floats in decimal as well as hexadecimal (a float in hexadecimal is its bits). It checks
//! syntax and the shape of the arenas, and leaves meaning to [`Function::verify`], so a test can
//! write malformed IR on purpose. Specifically, it refuses:
//!
//! - a line it cannot read, an unknown opcode, type, name or flag, a number out of range;
//! - a block header out of sequence (headers must be `block0`, `block1`, ... in order);
//! - an instruction id defined twice, or larger than the text's length in bytes (ids index the
//!   arena, so this bounds the memory a text can make the parser allocate);
//! - a result type that differs from the opcode's ([`Inst::result`](super::Inst::result)), and a
//!   phi without one;
//! - a line after a block's terminator, an instruction before the first block;
//! - an unbalanced tree, or one nested deeper than [`MAX_NESTING`].
//!
//! Ids the text does not define become detached `Void` slots, as the printer skips detached
//! instructions, so printing a parsed function gives back the text in canonical form.

use thiserror::Error;

use super::print::texture_keys as k;
use super::{
    Attribute, BlockId, Flags, FlagsKind, FmzMode, FpControl, FpRounding, Function, ImageFormat,
    Imm, Inst, MAX_NESTING, NO_CONTRACTION, Node, Opcode, Operand, Operation, Patch, Phi,
    Terminator, TextureInfo, TextureType, Type, UniformPred, UniformReg, Value,
};
use crate::decode::{Gpr, Pred};

/// What went wrong on a line.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseErrorKind {
    /// The line does not have the expected shape.
    #[error("expected {0}")]
    Expected(&'static str),
    /// An unknown opcode.
    #[error("unknown opcode {0}")]
    UnknownOpcode(String),
    /// An unknown type.
    #[error("unknown type {0}")]
    UnknownType(String),
    /// An operand that is not a value, a name or a typed immediate.
    #[error("unknown operand {0}")]
    UnknownOperand(String),
    /// A number that does not parse or does not fit.
    #[error("bad number {0}")]
    Number(String),
    /// A flag the opcode does not carry, or a repeated one.
    #[error("unknown or repeated flag {0}")]
    Flag(String),
    /// A block header out of sequence.
    #[error("block{found} is out of order: block{expected} comes next")]
    BlockOrder {
        /// The header's number.
        found: u32,
        /// The number expected.
        expected: usize,
    },
    /// An id defined twice.
    #[error("{0} is defined twice")]
    Redefined(Value),
    /// An id larger than the text.
    #[error("{0} is larger than the text's length")]
    IdTooLarge(Value),
    /// A result type that is not the opcode's.
    #[error("{opcode} has type {wanted}, not {found}")]
    Annotation {
        /// The opcode.
        opcode: Opcode,
        /// Its result type.
        wanted: Type,
        /// The type written.
        found: Type,
    },
    /// A phi without a type.
    #[error("a phi needs a result type")]
    PhiType,
    /// A line after a block's terminator.
    #[error("a line after the block's terminator")]
    AfterTerminator,
    /// An instruction before the first block header.
    #[error("an instruction outside a block")]
    OutsideBlock,
    /// A closing brace with nothing open, or the wrong closing form.
    #[error("a closing brace that does not match")]
    Unbalanced,
    /// The text ends with an `if` or `loop` open.
    #[error("the structure ends inside an if or a loop")]
    Unclosed,
    /// The tree nests too deeply.
    #[error("the structure nests deeper than {max} levels", max = MAX_NESTING)]
    TooDeep,
}

/// Why [`Function::parse`] failed: the line (counting from 1) and what was wrong.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("line {line}: {kind}")]
pub struct ParseError {
    /// The line, counting from 1.
    pub line: usize,
    /// What was wrong.
    pub kind: ParseErrorKind,
}

type Result<T> = std::result::Result<T, ParseErrorKind>;

/// An open `if` or `loop` in the tree.
enum Frame {
    Root,
    If(Operand),
    Loop,
}

/// The parser's state.
struct Parser {
    function: Function,
    /// Which ids are defined.
    defined: Vec<bool>,
    /// The largest id a text of this length may define.
    max_id: usize,
    /// Whether the current block has its terminator.
    terminated: bool,
    /// `Some` once `structure:` is read: the open frames, innermost last.
    frames: Option<Vec<(Frame, Vec<Node>)>>,
}

impl Function {
    /// Parses the text form (see the `parse` module docs).
    pub fn parse(text: &str) -> std::result::Result<Self, ParseError> {
        let mut parser = Parser {
            function: Self {
                insts: Vec::new(),
                blocks: Vec::new(),
                structure: None,
            },
            defined: Vec::new(),
            max_id: text.len(),
            terminated: false,
            frames: None,
        };
        let mut last = 0;
        for (index, raw) in text.lines().enumerate() {
            let line = index.saturating_add(1);
            last = line;
            let content = raw.split_once("//").map_or(raw, |(code, _)| code).trim();
            if content.is_empty() {
                continue;
            }
            parser
                .line(content)
                .map_err(|kind| ParseError { line, kind })?;
        }
        parser
            .finish()
            .map_err(|kind| ParseError { line: last, kind })
    }
}

impl std::str::FromStr for Function {
    type Err = ParseError;

    fn from_str(text: &str) -> std::result::Result<Self, ParseError> {
        Self::parse(text)
    }
}

impl Parser {
    fn line(&mut self, line: &str) -> Result<()> {
        if self.frames.is_some() {
            return self.structure_line(line);
        }
        if line == "structure:" {
            self.frames = Some(vec![(Frame::Root, Vec::new())]);
            return Ok(());
        }
        if let Some(label) = line.strip_suffix(':') {
            let block = parse_block(label)?;
            let expected = self.function.blocks.len();
            if usize::try_from(block.0).ok() != Some(expected) {
                return Err(ParseErrorKind::BlockOrder {
                    found: block.0,
                    expected,
                });
            }
            self.function
                .add_block()
                .map_err(|_| ParseErrorKind::Expected("fewer blocks"))?;
            self.terminated = false;
            return Ok(());
        }
        let block = self
            .function
            .blocks
            .len()
            .checked_sub(1)
            .ok_or(ParseErrorKind::OutsideBlock)?;
        if self.terminated {
            return Err(ParseErrorKind::AfterTerminator);
        }
        if line.starts_with('%') {
            return self.instruction(line, block);
        }
        let terminator = parse_terminator(line)?;
        if let Some(b) = self.function.blocks.get_mut(block) {
            b.terminator = terminator;
        }
        self.terminated = true;
        Ok(())
    }

    fn instruction(&mut self, line: &str, block: usize) -> Result<()> {
        let (head, body) = line
            .split_once('=')
            .ok_or(ParseErrorKind::Expected("= after the id"))?;
        let (id, annotation) = match head.split_once(':') {
            Some((id, ty)) => (id.trim(), Some(parse_type(ty.trim())?)),
            None => (head.trim(), None),
        };
        let value = parse_value(id)?;
        let slot = value.slot().filter(|&slot| slot <= self.max_id);
        let Some(slot) = slot else {
            return Err(ParseErrorKind::IdTooLarge(value));
        };
        if self.defined.get(slot).copied().unwrap_or(false) {
            return Err(ParseErrorKind::Redefined(value));
        }
        let inst = parse_inst(body.trim(), annotation)?;
        if self.defined.len() <= slot {
            self.defined.resize(slot.saturating_add(1), false);
        }
        if let Some(seen) = self.defined.get_mut(slot) {
            *seen = true;
        }
        self.function
            .place(value, inst)
            .ok_or(ParseErrorKind::IdTooLarge(value))?;
        if let Some(b) = self.function.blocks.get_mut(block) {
            b.insts.push(value);
        }
        Ok(())
    }

    fn structure_line(&mut self, line: &str) -> Result<()> {
        let frames = self.frames.get_or_insert_with(Vec::new);
        let node = if let Some(rest) = line.strip_prefix('}') {
            let rest = rest.trim();
            let (frame, body) = frames.pop().ok_or(ParseErrorKind::Unbalanced)?;
            match (frame, rest.strip_prefix("repeat ")) {
                (Frame::If(cond), None) if rest.is_empty() => Node::If { cond, body },
                (Frame::Loop, Some(repeat)) => Node::Loop {
                    body,
                    repeat: parse_operand(repeat.trim())?,
                },
                _ => return Err(ParseErrorKind::Unbalanced),
            }
        } else if let Some(rest) = line.strip_suffix('{') {
            let rest = rest.trim();
            let frame = if rest == "loop" {
                Frame::Loop
            } else if let Some(cond) = rest.strip_prefix("if ") {
                Frame::If(parse_operand(cond.trim())?)
            } else {
                return Err(ParseErrorKind::Expected("if or loop"));
            };
            // The root is level 0, so the new body is at level `frames.len()`.
            if frames.len() > MAX_NESTING {
                return Err(ParseErrorKind::TooDeep);
            }
            frames.push((frame, Vec::new()));
            return Ok(());
        } else if let Some(cond) = line.strip_prefix("break ") {
            Node::Break {
                cond: parse_operand(cond.trim())?,
            }
        } else if line == "return" {
            Node::Return
        } else if line == "unreachable" {
            Node::Unreachable
        } else {
            Node::Block(parse_block(line)?)
        };
        frames
            .last_mut()
            .ok_or(ParseErrorKind::Unbalanced)?
            .1
            .push(node);
        Ok(())
    }

    fn finish(mut self) -> Result<Function> {
        if let Some(mut frames) = self.frames.take() {
            let (frame, nodes) = frames.pop().ok_or(ParseErrorKind::Unbalanced)?;
            if !frames.is_empty() || !matches!(frame, Frame::Root) {
                return Err(ParseErrorKind::Unclosed);
            }
            self.function.structure = Some(nodes);
        }
        Ok(self.function)
    }
}

fn parse_number<T: TryFrom<u64>>(text: &str) -> Result<T> {
    let bad = || ParseErrorKind::Number(text.to_owned());
    let wide = match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).map_err(|_| bad())?,
        None => text.parse::<u64>().map_err(|_| bad())?,
    };
    T::try_from(wide).map_err(|_| bad())
}

fn parse_value(text: &str) -> Result<Value> {
    let digits = text
        .strip_prefix('%')
        .ok_or(ParseErrorKind::Expected("an id such as %3"))?;
    if digits.starts_with("0x") {
        return Err(ParseErrorKind::Number(text.to_owned()));
    }
    Ok(Value(parse_number(digits)?))
}

fn parse_block(text: &str) -> Result<BlockId> {
    let digits = text
        .strip_prefix("block")
        .ok_or(ParseErrorKind::Expected("a block such as block1"))?;
    if digits.starts_with("0x") {
        return Err(ParseErrorKind::Number(text.to_owned()));
    }
    Ok(BlockId(parse_number(digits)?))
}

fn parse_type(text: &str) -> Result<Type> {
    Type::from_name(text).ok_or_else(|| ParseErrorKind::UnknownType(text.to_owned()))
}

/// A register-like name: `prefix` then a number up to `max`, or the reserved spelling.
fn parse_index(text: &str, prefix: &str, max: u8) -> Option<u8> {
    let digits = text.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u8>().ok().filter(|&index| index <= max)
}

fn parse_name(text: &str) -> Option<Imm> {
    Some(match text {
        "rz" => Imm::Reg(Gpr::RZ),
        "pt" => Imm::Pred(Pred::PT),
        "urz" => Imm::UniformReg(UniformReg::URZ),
        "upt" => Imm::UniformPred(UniformPred::UPT),
        _ => {
            if let Some(index) = parse_index(text, "ur", UniformReg::URZ.index() - 1) {
                Imm::UniformReg(UniformReg::new(index)?)
            } else if let Some(index) = parse_index(text, "up", UniformPred::UPT.index() - 1) {
                Imm::UniformPred(UniformPred::new(index)?)
            } else if let Some(index) = parse_index(text, "patch", Patch::COUNT - 1) {
                Imm::Patch(Patch::new(index)?)
            } else if let Some(index) = parse_index(text, "r", Gpr::RZ.index() - 1) {
                Imm::Reg(Gpr::new(index))
            } else if let Some(index) = parse_index(text, "p", Pred::PT.index() - 1) {
                Imm::Pred(Pred::new(index)?)
            } else {
                let hex = text.strip_prefix("a0x")?;
                let address = u32::from_str_radix(hex, 16).ok()?;
                if !address.is_multiple_of(super::ATTRIBUTE_WORD_BYTES) {
                    return None;
                }
                Imm::Attribute(Attribute(
                    u16::try_from(address / super::ATTRIBUTE_WORD_BYTES).ok()?,
                ))
            }
        }
    })
}

fn parse_float_bits(text: &str, double: bool) -> Result<u64> {
    if text.starts_with("0x") {
        return parse_number(text);
    }
    let bad = || ParseErrorKind::Number(text.to_owned());
    if double {
        text.parse::<f64>().map(f64::to_bits).map_err(|_| bad())
    } else {
        text.parse::<f32>()
            .map(|value| u64::from(value.to_bits()))
            .map_err(|_| bad())
    }
}

fn parse_operand(text: &str) -> Result<Operand> {
    if text.starts_with('%') {
        return parse_value(text).map(Operand::Value);
    }
    let Some((ty, literal)) = text.split_once(' ') else {
        return parse_name(text)
            .map(Operand::Imm)
            .ok_or_else(|| ParseErrorKind::UnknownOperand(text.to_owned()));
    };
    let literal = literal.trim();
    let imm = match parse_type(ty)? {
        Type::U1 => match literal {
            "true" => Imm::U1(true),
            "false" => Imm::U1(false),
            _ => return Err(ParseErrorKind::Number(literal.to_owned())),
        },
        Type::U8 => Imm::U8(parse_number(literal)?),
        Type::U16 => Imm::U16(parse_number(literal)?),
        Type::U32 => Imm::U32(parse_number(literal)?),
        Type::U64 => Imm::U64(parse_number(literal)?),
        Type::F32 => Imm::F32(
            u32::try_from(parse_float_bits(literal, false)?)
                .map_err(|_| ParseErrorKind::Number(literal.to_owned()))?,
        ),
        Type::F64 => Imm::F64(parse_float_bits(literal, true)?),
        _ => return Err(ParseErrorKind::UnknownOperand(text.to_owned())),
    };
    Ok(Operand::Imm(imm))
}

fn parse_terminator(line: &str) -> Result<Terminator> {
    if line == "return" {
        return Ok(Terminator::Return);
    }
    if line == "unreachable" {
        return Ok(Terminator::Unreachable);
    }
    if let Some(target) = line.strip_prefix("jump ") {
        return Ok(Terminator::Jump(parse_block(target.trim())?));
    }
    if let Some(rest) = line.strip_prefix("branch ") {
        let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
        let [cond, then, otherwise] = parts.as_slice() else {
            return Err(ParseErrorKind::Expected(
                "branch <condition>, <block>, <block>",
            ));
        };
        return Ok(Terminator::Branch {
            cond: parse_operand(cond)?,
            then: parse_block(then)?,
            otherwise: parse_block(otherwise)?,
        });
    }
    Err(ParseErrorKind::Expected("an instruction or a terminator"))
}

/// The list items of `text`, split at commas; none for an empty text.
fn items(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        text.split(',').map(str::trim).collect()
    }
}

fn parse_inst(body: &str, annotation: Option<Type>) -> Result<Inst> {
    let (name, rest) = body.split_once(' ').unwrap_or((body, ""));
    if name == "Phi" {
        let ty = annotation.ok_or(ParseErrorKind::PhiType)?;
        let mut incoming = Vec::new();
        for item in items(rest) {
            let inner = item
                .strip_prefix('[')
                .and_then(|item| item.strip_suffix(']'))
                .ok_or(ParseErrorKind::Expected("[block: operand]"))?;
            let (block, operand) = inner
                .split_once(':')
                .ok_or(ParseErrorKind::Expected("[block: operand]"))?;
            incoming.push((parse_block(block.trim())?, parse_operand(operand.trim())?));
        }
        return Ok(Inst::Phi(Phi { ty, incoming }));
    }
    let opcode =
        Opcode::from_name(name).ok_or_else(|| ParseErrorKind::UnknownOpcode(name.to_owned()))?;
    if let Some(found) = annotation
        && found != opcode.result()
    {
        return Err(ParseErrorKind::Annotation {
            opcode,
            wanted: opcode.result(),
            found,
        });
    }
    let (args, flags) = match rest.split_once('{') {
        Some((args, flags)) => {
            let flags = flags
                .trim()
                .strip_suffix('}')
                .ok_or(ParseErrorKind::Expected("} after the flags"))?;
            (args, Some(flags))
        }
        None => (rest, None),
    };
    let args = items(args)
        .into_iter()
        .map(parse_operand)
        .collect::<Result<Vec<_>>>()?;
    let flags = parse_flags(opcode.flags_kind(), &flags.map(items).unwrap_or_default())?;
    Ok(Inst::Op(Operation {
        opcode,
        args,
        flags,
    }))
}

fn parse_flags(kind: FlagsKind, words: &[&str]) -> Result<Flags> {
    match kind {
        FlagsKind::None => match words.first() {
            Some(word) => Err(ParseErrorKind::Flag((*word).to_owned())),
            None => Ok(Flags::None),
        },
        FlagsKind::Fp => parse_fp(words).map(Flags::Fp),
        FlagsKind::Texture => parse_texture(words).map(Flags::Texture),
    }
}

fn parse_fp(words: &[&str]) -> Result<FpControl> {
    let mut control = FpControl::default();
    for &word in words {
        let repeated = || ParseErrorKind::Flag(word.to_owned());
        if word == NO_CONTRACTION {
            if control.no_contraction {
                return Err(repeated());
            }
            control.no_contraction = true;
        } else if let Some(mode) = FpRounding::NAMED
            .into_iter()
            .find(|m| m.name() == Some(word))
        {
            if control.rounding != FpRounding::DontCare {
                return Err(repeated());
            }
            control.rounding = mode;
        } else if let Some(mode) = FmzMode::NAMED.into_iter().find(|m| m.name() == Some(word)) {
            if control.fmz_mode != FmzMode::DontCare {
                return Err(repeated());
            }
            control.fmz_mode = mode;
        } else {
            return Err(repeated());
        }
    }
    Ok(control)
}

fn parse_texture(words: &[&str]) -> Result<TextureInfo> {
    let mut info = TextureInfo::default();
    let mut seen: Vec<&str> = Vec::new();
    for &word in words {
        let bad = || ParseErrorKind::Flag(word.to_owned());
        let (key, value) = match word.split_once('=') {
            Some((key, value)) => (key, Some(value)),
            None => (word, None),
        };
        if seen.contains(&key) {
            return Err(bad());
        }
        seen.push(key);
        match (key, value) {
            (k::DESCRIPTOR, Some(value)) => info.descriptor_index = parse_number(value)?,
            (k::TYPE, Some(value)) => {
                info.ty = TextureType::ALL
                    .into_iter()
                    .find(|ty| ty.name() == value)
                    .ok_or_else(bad)?;
            }
            (k::DEPTH, None) => info.is_depth = true,
            (k::BIAS, None) => info.has_bias = true,
            (k::LOD_CLAMP, None) => info.has_lod_clamp = true,
            (k::RELAXED, None) => info.relaxed_precision = true,
            (k::GATHER, Some(value)) => info.gather_component = parse_two_bit(value)?,
            (k::DERIVATIVES, Some(value)) => info.num_derivatives = parse_two_bit(value)?,
            (k::FORMAT, Some(value)) => {
                info.image_format = ImageFormat::ALL
                    .into_iter()
                    .find(|format| format.name() == value)
                    .ok_or_else(bad)?;
            }
            (k::NDV, None) => info.ndv_is_active = true,
            (k::INTEGER, None) => info.is_integer = true,
            _ => return Err(bad()),
        }
    }
    Ok(info)
}

fn parse_two_bit(text: &str) -> Result<u8> {
    let value: u8 = parse_number(text)?;
    if value > super::TEXTURE_TWO_BIT_MAX {
        return Err(ParseErrorKind::Number(text.to_owned()));
    }
    Ok(value)
}
