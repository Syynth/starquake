//! The constant-buffer and texture pass's tests: golden resource tables and IR, every refusal
//! with its exact reason, the edges of every bound, and interpreter equivalence where the pass
//! rewrites reads. Answers come through the production exchange loop
//! ([`crate::exchange::exchange`]) from a scripted engine state.

mod answers;
mod cbuf;
mod forms;
mod refusals;
mod rewrite;
mod table;
mod trace;

use std::collections::BTreeMap;

use super::{ResourceError, ResourceTable, TextureScan, Untraced, run, scan};
use crate::exchange::{
    Answers, EngineState, SampledType, TextureFormat, TextureKind, exchange,
};
use crate::ir::{Function, Value};
use crate::translate::GuestStage;
use crate::translate::texture::DescriptorBuffer;

/// The stage every test asks for.
const STAGE: GuestStage = GuestStage::Pixel;

/// The descriptor buffer bank most tests use, one that a bindless handle's bank differs from.
const BOUND_BANK: u8 = 2;

/// Engine state from a script: constant-buffer words and what handles point to.
#[derive(Clone, Debug, Default)]
struct Engine {
    words: BTreeMap<(u8, u32), u32>,
    kinds: BTreeMap<u32, TextureKind>,
}

impl Engine {
    fn new() -> Self {
        Self::default()
    }

    /// The word at `offset` of `bank` holds `value`.
    fn word(mut self, bank: u8, offset: u32, value: u32) -> Self {
        self.words.insert((bank, offset), value);
        self
    }

    /// `handle` points to `kind`.
    fn kind(mut self, handle: u32, kind: TextureKind) -> Self {
        self.kinds.insert(handle, kind);
        self
    }
}

impl EngineState for Engine {
    fn cbuf_word(&self, _stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        self.words.get(&(bank, offset)).copied()
    }

    fn texture_kind(&self, _stage: GuestStage, handle: u32) -> Option<TextureKind> {
        self.kinds.get(&handle).copied()
    }
}

/// A texture kind with these fields and the format bits `format`.
const fn kind_of(sampled: SampledType, buffer: bool, format: u32) -> TextureKind {
    TextureKind {
        sampled,
        depth: false,
        multisampled: false,
        buffer,
        format: TextureFormat::from_texhead_word0(format),
    }
}

/// A float, non-buffer texture.
const FLOAT: TextureKind = kind_of(SampledType::Float, false, 0);

/// Parses `text`, which must be well-formed and verify.
fn parse(text: &str) -> Function {
    let function = Function::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    function
        .verify()
        .unwrap_or_else(|e| panic!("{e}\n{function}"));
    function
}

/// The descriptor buffer of a program whose bound handles are in `bank`.
fn buffer(bank: u8) -> DescriptorBuffer {
    DescriptorBuffer::bound(bank).expect("a bank below the limit")
}

/// The scan of `function` and the answers the exchange collects for it from `engine`.
fn answered(
    function: &Function,
    bound: DescriptorBuffer,
    engine: &Engine,
) -> (TextureScan, Answers) {
    let scan = scan(function, bound).expect("scans");
    let answers = exchange(STAGE, &scan, engine).expect("exchanges").answers;
    (scan, answers)
}

/// `text` after the pass, with the table, given `engine` and the bound bank [`BOUND_BANK`].
fn pass(text: &str, engine: &Engine) -> (Function, ResourceTable) {
    let mut function = parse(text);
    let (scan, answers) = answered(&function, buffer(BOUND_BANK), engine);
    let (table, _) = run(&mut function, &scan, &answers).expect("runs");
    (function, table)
}

/// The refusal scanning `text` gives.
fn scan_error(text: &str) -> ResourceError {
    scan(&parse(text), buffer(BOUND_BANK)).expect_err("refused")
}

/// Why the handle of the texture operation in `text` cannot be traced.
fn untraced(text: &str) -> Untraced {
    match scan_error(text) {
        ResourceError::Untraced { why, .. } => why,
        other => panic!("not an untraced handle: {other}"),
    }
}

/// The value id `index`, taken from a parsed function (ids have no public constructor).
fn value(index: u32) -> Value {
    // The parser takes ids up to the text's length, so the line is padded past it.
    let pad = " ".repeat(usize::try_from(index).unwrap_or(0));
    let text = format!("block0:\n    %{index} = Void{pad}\n    return\n");
    Function::parse(&text)
        .expect("parses")
        .blocks()
        .next()
        .and_then(|(_, block)| block.insts.first().copied())
        .expect("one instruction")
}

/// A program whose `lines` compute a handle in `%h` (any id below 50) and sample a 2D texture
/// with it at `%60`.
fn sample_of(lines: &str, handle: &str) -> String {
    format!(
        "block0:\n{lines}    %50: f32x2 = CompositeConstructF32x2 f32 0x0, f32 0x0\n    \
         %60: f32x4 = ImageSampleImplicitLod {handle}, %50, u32 0x0, u32 0x0 {{type=2d}}\n    \
         return\n"
    )
}

/// The handle sources [`scan`] finds in `text`, as text, one per line.
fn sources(text: &str) -> String {
    scan(&parse(text), buffer(BOUND_BANK))
        .expect("scans")
        .to_string()
}
