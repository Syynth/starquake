//! The text form's printer.
//!
//! The format (the parser reads exactly this, plus blank lines and `//` comments):
//!
//! ```text
//! block0:
//!     %0: f32 = GetAttribute a0x80, u32 0x0
//!     %1: f32 = FPFma32 %0, %0, %0 {no_contraction, rz, ftz}
//!     %2 = SetAttribute a0x70, %1, u32 0x0
//!     branch %3, block1, block2
//! block1:
//!     %4: u32 = Phi [block0: %5], [block2: u32 0x0]
//! structure:
//!     block0
//!     if %3 {
//!         block1
//!     }
//!     loop {
//!         block2
//!         break u1 true
//!     } repeat %6
//!     return
//! ```
//!
//! Blocks print in arena order, each with its instructions in list order. An instruction prints
//! as its id, its result type when it has one (`opaque` for `Identity`, whose value has its
//! argument's type), Eden's opcode name, its arguments and, in braces, its flags where they
//! differ from the default of their kind. A control-flow graph's block ends with its terminator
//! (`jump`, `branch`, `return`, `unreachable`; nothing when unset); a structured function has
//! no terminators and ends with its tree after `structure:`. Arena slots that no block lists
//! (detached instructions) are not printed. The format follows Eden's `DumpBlock`
//! (`eden/src/shader_recompiler/frontend/ir/basic_block.cpp` @ cf654052) in spirit, with
//! every immediate typed so the parser can read it back.

use std::fmt::{self, Display, Formatter, Write};

use super::{
    Flags, FpControl, Function, Inst, NO_CONTRACTION, Node, Operation, Terminator, TextureInfo,
    Type,
};

/// One indentation level.
const INDENT: &str = "    ";

/// The flags of an instruction, as the words inside its braces; empty for the defaults.
pub(super) fn flag_words(flags: &Flags) -> Vec<String> {
    match flags {
        Flags::None => Vec::new(),
        Flags::Fp(control) => fp_words(*control),
        Flags::Texture(info) => texture_words(info),
    }
}

fn fp_words(control: FpControl) -> Vec<String> {
    let mut words = Vec::new();
    if control.no_contraction {
        words.push(NO_CONTRACTION.to_owned());
    }
    words.extend(control.rounding.name().map(str::to_owned));
    words.extend(control.fmz_mode.name().map(str::to_owned));
    words
}

/// The texture flag words, in the order the parser documents.
pub(super) mod texture_keys {
    /// `descriptor=<n>`.
    pub(in super::super) const DESCRIPTOR: &str = "descriptor";
    /// `type=<name>`.
    pub(in super::super) const TYPE: &str = "type";
    /// A depth comparison.
    pub(in super::super) const DEPTH: &str = "depth";
    /// A bias.
    pub(in super::super) const BIAS: &str = "bias";
    /// A level-of-detail clamp.
    pub(in super::super) const LOD_CLAMP: &str = "lod_clamp";
    /// Relaxed precision.
    pub(in super::super) const RELAXED: &str = "relaxed_precision";
    /// `gather=<n>`.
    pub(in super::super) const GATHER: &str = "gather";
    /// `derivatives=<n>`.
    pub(in super::super) const DERIVATIVES: &str = "derivatives";
    /// `format=<name>`.
    pub(in super::super) const FORMAT: &str = "format";
    /// `.NDV`.
    pub(in super::super) const NDV: &str = "ndv";
    /// An integer texture.
    pub(in super::super) const INTEGER: &str = "integer";
}

fn texture_words(info: &TextureInfo) -> Vec<String> {
    use texture_keys as k;
    let default = TextureInfo::default();
    let mut words = Vec::new();
    if info.descriptor_index != default.descriptor_index {
        words.push(format!("{}={}", k::DESCRIPTOR, info.descriptor_index));
    }
    if info.ty != default.ty {
        words.push(format!("{}={}", k::TYPE, info.ty.name()));
    }
    let bools = [
        (info.is_depth, k::DEPTH),
        (info.has_bias, k::BIAS),
        (info.has_lod_clamp, k::LOD_CLAMP),
        (info.relaxed_precision, k::RELAXED),
    ];
    words.extend(
        bools
            .iter()
            .filter(|(on, _)| *on)
            .map(|(_, w)| (*w).to_owned()),
    );
    if info.gather_component != default.gather_component {
        words.push(format!("{}={}", k::GATHER, info.gather_component));
    }
    if info.num_derivatives != default.num_derivatives {
        words.push(format!("{}={}", k::DERIVATIVES, info.num_derivatives));
    }
    if info.image_format != default.image_format {
        words.push(format!("{}={}", k::FORMAT, info.image_format.name()));
    }
    let bools = [(info.ndv_is_active, k::NDV), (info.is_integer, k::INTEGER)];
    words.extend(
        bools
            .iter()
            .filter(|(on, _)| *on)
            .map(|(_, w)| (*w).to_owned()),
    );
    words
}

fn write_operation(f: &mut Formatter<'_>, op: &Operation) -> fmt::Result {
    f.write_str(op.opcode.name())?;
    for (index, arg) in op.args.iter().enumerate() {
        f.write_str(if index == 0 { " " } else { ", " })?;
        write!(f, "{arg}")?;
    }
    let words = flag_words(&op.flags);
    if !words.is_empty() {
        write!(f, " {{{}}}", words.join(", "))?;
    }
    Ok(())
}

impl Display for Inst {
    /// The instruction without its id: `FPAdd32 %1, %2 {rz}`, `Phi [block1: %3]`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Op(op) => write_operation(f, op),
            Self::Phi(phi) => {
                f.write_str("Phi")?;
                for (index, (block, operand)) in phi.incoming.iter().enumerate() {
                    f.write_str(if index == 0 { " " } else { ", " })?;
                    write!(f, "[{block}: {operand}]")?;
                }
                Ok(())
            }
        }
    }
}

impl Display for Terminator {
    /// The terminator's line; empty for [`Terminator::None`].
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Jump(target) => write!(f, "jump {target}"),
            Self::Branch {
                cond,
                then,
                otherwise,
            } => write!(f, "branch {cond}, {then}, {otherwise}"),
            Self::Return => f.write_str("return"),
            Self::Unreachable => f.write_str("unreachable"),
        }
    }
}

fn write_nodes(out: &mut String, nodes: &[Node], depth: usize) -> fmt::Result {
    let pad = INDENT.repeat(depth.saturating_add(1));
    for node in nodes {
        match node {
            Node::Block(block) => writeln!(out, "{pad}{block}")?,
            Node::If { cond, body } => {
                writeln!(out, "{pad}if {cond} {{")?;
                write_nodes(out, body, depth.saturating_add(1))?;
                writeln!(out, "{pad}}}")?;
            }
            Node::Loop { body, repeat } => {
                writeln!(out, "{pad}loop {{")?;
                write_nodes(out, body, depth.saturating_add(1))?;
                writeln!(out, "{pad}}} repeat {repeat}")?;
            }
            Node::Break { cond } => writeln!(out, "{pad}break {cond}")?,
            Node::Return => writeln!(out, "{pad}return")?,
            Node::Unreachable => writeln!(out, "{pad}unreachable")?,
        }
    }
    Ok(())
}

impl Display for Function {
    /// The text form (see the `print` module docs). The tree is printed recursively, one level
    /// per nesting level.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (id, block) in self.blocks() {
            writeln!(f, "{id}:")?;
            for &value in &block.insts {
                let Some(inst) = self.inst(value) else {
                    writeln!(f, "{INDENT}{value} = <missing>")?;
                    continue;
                };
                match inst.result() {
                    Type::Void => writeln!(f, "{INDENT}{value} = {inst}")?,
                    ty => writeln!(f, "{INDENT}{value}: {ty} = {inst}")?,
                }
            }
            if block.terminator != Terminator::None {
                writeln!(f, "{INDENT}{}", block.terminator)?;
            }
        }
        if let Some(nodes) = self.structure() {
            writeln!(f, "structure:")?;
            let mut out = String::new();
            write_nodes(&mut out, nodes, 0)?;
            f.write_str(&out)?;
        }
        Ok(())
    }
}
