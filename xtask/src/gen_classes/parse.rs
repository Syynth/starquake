//! Reading one of NVIDIA's class headers (`mesa-nouveau/src/nouveau/headers/nvidia/classes/*.h`)
//! into a [`Header`]: its methods, the bit fields of each, the named values of each field, the
//! fields that belong to no method (structure layouts such as the QMD, the texture header or the
//! GPFIFO entry) and the plain constants.
//!
//! Mesa's `class_parser.py` (MIT, in `mesa-nouveau/src/nouveau/headers/`) is the model for how
//! the headers are laid out; this is a new reader, not a translation of it, and it differs on
//! purpose in the ways listed below.
//!
//! # The format
//!
//! Comments (`/* */`, which may span lines, and `//`) are removed first. Every remaining line
//! that starts with `#` is a directive: `#ifndef`, `#ifdef`, `#endif` and `#include` are
//! skipped, `#define` is read, and any other directive is an error. Lines that are not
//! directives (C declarations such as the channel's control structure) are skipped.
//!
//! A `#define` whose name does not start with the class prefix (`NV` and the four hex digits of
//! the file name: `NVC797_` for `clc797.h` and `clc797tex.h`) is either an include guard (no
//! value), skipped, or the class id (`AMPERE_B 0xC797`), of which there is at most one. Under the
//! prefix, the value is one of:
//!
//! - a number, hex or decimal, in any number of parentheses (`0x0100`, `(0x00000100)`, `(8)`);
//! - a single-word bit range `high:low`;
//! - a multi-word bit range `MW(high:low)`, or an array of them,
//!   `MW((h+(i)*s):(l+(i)*s))`;
//! - an array method address, `(base+(i)*stride)` or `(base+(i)*s1+(j)*s2)`, whose index names
//!   match the parameters of the define's name (`NAME(i)`, `NAME(j)`, `NAME(i,j)`).
//!
//! A name ending in `TYPEDEF` with an identifier value is skipped. Anything else is a
//! [`ParseError`] naming the file and line.
//!
//! # How defines are grouped
//!
//! In file order, with a current method and a current field:
//!
//! - An array address starts a method.
//! - A number is a value of the current field if its name extends the field's name with `_`;
//!   otherwise a value of the field of the current method whose name it extends (the longest such
//!   name); otherwise it is held back: if the next define is a single-word bit range extending its
//!   name, it was a method (its number is the byte address), and if not, it is a constant.
//! - A single-word bit range extending the current method's name is a field of that method;
//!   any other bit range is a field of no method, and ends the current method.
//!
//! A name (after the class prefix) is letters, digits and underscores starting with an
//! upper-case letter; a few value names have lower-case letters (`MONOCHROME_8x8` in
//! `cl902d.h`), which the generated constant names upper-case. A define repeated with the same
//! name and value is read once (`clc76f.h` defines `DMA_IMMD_DATA` twice); a name that comes
//! back with another value, or spelled in another case, is an error.
//!
//! # Divergences from Mesa's `class_parser.py`
//!
//! - A number that no field follows is kept as a constant; Mesa drops it.
//! - A define that belongs to no field in the middle of a method (in `clc76f.h`,
//!   `MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES` among `MEM_OP_D`'s values) becomes a
//!   constant, and the method's later values and fields are still read. Mesa's state machine
//!   leaves the method there and loses them.
//! - Every value is kept, including two names for one number (`clc76f.h`, `WFI_SCOPE`:
//!   `CURRENT_SCG_TYPE` and `CURRENT_VEID`); Mesa skips `CURRENT_SCG_TYPE` (`SKIP_FIELD`).
//! - Two-index arrays (`SET_STREAM_OUT_LAYOUT_SELECT(i,j)` in `clc797.h`) are methods with two
//!   strides; Mesa does not read them as arrays.
//! - Fields of no method are kept; Mesa's class parser skips them, and its `struct_parser.py`
//!   rewrites dword-numbered names (`TEXSAMP0_ADDRESS_U 2:0`) into absolute bit positions, where
//!   this reader keeps the header's own words.
//! - Mesa's hand-kept lists (array lengths, float methods) are not part of the output: they are
//!   not in the headers.

use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

/// One class header, read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// The class prefix of every define, from the file name (`NVC797_`).
    pub prefix: String,
    /// The licence notice at the top of the file, one entry per line, with the comment
    /// decoration removed.
    pub notice: Vec<String>,
    /// The class id define, if the header has one.
    pub class: Option<ClassId>,
    /// The methods, in file order.
    pub methods: Vec<Method>,
    /// The fields that belong to no method, in file order.
    pub fields: Vec<Field>,
    /// The numbers that are neither methods nor values, in file order.
    pub constants: Vec<Value>,
}

/// The define that names the class (`AMPERE_B 0xC797`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassId {
    /// The define's name.
    pub name: String,
    /// The class number.
    pub id: u32,
}

/// A method: a register in the class's method space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Method {
    /// The name without the class prefix or index parameters (`SET_VIEWPORT_SCALE_X`).
    pub name: String,
    /// The byte address of the method (of element 0 for an array).
    pub byte_address: u32,
    /// For an array, the byte stride of each index, in the order of the name's parameters.
    pub strides: Vec<u32>,
    /// The method's fields, in file order.
    pub fields: Vec<Field>,
}

/// A bit field, and its named values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// For a method's field, the part after the method's name (`CLASS_ID`); for a field of no
    /// method, the whole name without the class prefix or index parameters.
    pub name: String,
    /// Where the field lives.
    pub bits: Bits,
    /// The named values, in file order: the part of each name after the field's name.
    pub values: Vec<Value>,
}

/// Where a field lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bits {
    /// `high:low` within one 32-bit word.
    Word {
        /// The highest bit.
        high: u32,
        /// The lowest bit.
        low: u32,
    },
    /// `MW(high:low)`: bit positions counted across the words of a structure.
    Multi {
        /// The highest bit.
        high: u32,
        /// The lowest bit.
        low: u32,
    },
    /// `MW((high+(i)*stride):(low+(i)*stride))`: element `i` of an array of fields.
    MultiArray {
        /// The highest bit of element 0.
        high: u32,
        /// The lowest bit of element 0.
        low: u32,
        /// The distance in bits between elements.
        stride: u32,
    },
}

/// A named number: a value of a field, or a constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    /// For a field's value, the part after the field's name; for a constant, the whole name
    /// without the class prefix.
    pub name: String,
    /// The number.
    pub value: u32,
}

/// A header that could not be read: which file, which line, and why.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{file}:{line}: {kind}")]
pub struct ParseError {
    /// The header's file name.
    pub file: String,
    /// The line, counted from 1.
    pub line: usize,
    /// What is wrong.
    pub kind: ErrorKind,
}

/// What is wrong with a header line.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ErrorKind {
    /// The file name is not `cl` and four hex digits, so it gives no class prefix.
    #[error("the file name does not start with cl and four hex digits")]
    FileName,
    /// The file does not start with a comment holding the MIT permission notice.
    #[error("the file does not start with a licence notice granting the MIT permissions")]
    NoNotice,
    /// A `/*` comment that is never closed.
    #[error("the comment opened here is never closed")]
    UnclosedComment,
    /// A directive other than the ones a class header uses.
    #[error("unsupported directive #{0}")]
    Directive(String),
    /// A `#define` with no name.
    #[error("#define without a name")]
    NoName,
    /// A name under the class prefix that is not letters, digits and underscores starting with
    /// an upper-case letter, with optional `(i)`, `(j)` or `(i,j)` parameters.
    #[error("{0} is not a name this reader understands")]
    Name(String),
    /// A define under the class prefix with no value.
    #[error("{0} has no value")]
    NoValue(String),
    /// More than one token after the name.
    #[error("{name} has more than one value: {rest}")]
    ExtraText {
        /// The define's name.
        name: String,
        /// Everything after the name.
        rest: String,
    },
    /// A value that is none of the forms the reader knows.
    #[error("{0} is not a number, a bit range or an array address")]
    UnknownValue(String),
    /// A define outside the class prefix whose value is not a number.
    #[error("{0} is outside the class prefix and is not a class id")]
    Unprefixed(String),
    /// A second class id define.
    #[error("{name} is a second class id (the first is {first})")]
    SecondClass {
        /// This define's name.
        name: String,
        /// The first class id's name.
        first: String,
    },
    /// A number that does not fit in 32 bits.
    #[error("{0} does not fit in 32 bits")]
    TooBig(String),
    /// A bit range whose high bit is below its low bit.
    #[error("bit range {high}:{low} is inverted")]
    Inverted {
        /// The high bit.
        high: u32,
        /// The low bit.
        low: u32,
    },
    /// A single-word bit range above bit 31.
    #[error("bit range {high}:{low} goes past bit 31 of a word")]
    PastWord {
        /// The high bit.
        high: u32,
        /// The low bit.
        low: u32,
    },
    /// A multi-word bit range wider than 32 bits.
    #[error("bit range {high}:{low} is wider than 32 bits")]
    TooWide {
        /// The high bit.
        high: u32,
        /// The low bit.
        low: u32,
    },
    /// A method byte address or stride that is not a whole number of words.
    #[error("{name}: byte address or stride {value:#x} is not a multiple of 4")]
    Unaligned {
        /// The method's name.
        name: String,
        /// The offending number.
        value: u32,
    },
    /// The index names in an array value do not match the name's parameters, or the value
    /// has an index the name does not, or the other way round.
    #[error("{name}: the indices of {value} do not match the name's parameters")]
    Index {
        /// The define's name.
        name: String,
        /// The define's value.
        value: String,
    },
    /// The two ends of a multi-word array range step by different amounts.
    #[error("{0}: the high and low bits of the array step by different amounts")]
    Stride(String),
    /// A name defined again with another value, or in another case.
    #[error(
        "{name} is defined again with another value or in another case (first at line {first})"
    )]
    Redefined {
        /// The name.
        name: String,
        /// The line of the first definition.
        first: usize,
    },
    /// An array method that no field follows.
    #[error("array method {0} has no fields")]
    EmptyArray(String),
}

/// Words of the notice that mark the MIT permission grant.
const MIT_GRANT: &str = "Permission is hereby granted, free of charge";

/// The class prefix for a header file name: `NV` and the first four hex digits after `cl`,
/// upper-cased, then `_` (`clc797tex.h` gives `NVC797_`). `None` if the name has no such digits.
pub fn class_prefix(file: &str) -> Option<String> {
    let digits = file.strip_prefix("cl")?.get(..4)?;
    digits
        .chars()
        .all(|c| c.is_ascii_hexdigit())
        .then(|| format!("NV{}_", digits.to_ascii_uppercase()))
}

/// Reads the header `text` of the file called `file`, keeping only the defines whose name,
/// after the class prefix, starts with `only` (when given).
pub fn parse(file: &str, text: &str, only: Option<&str>) -> Result<Header, ParseError> {
    let error = |line: usize, kind: ErrorKind| ParseError {
        file: file.to_owned(),
        line,
        kind,
    };
    let prefix = class_prefix(file).ok_or_else(|| error(1, ErrorKind::FileName))?;
    let stripped = strip_comments(text).map_err(|line| error(line, ErrorKind::UnclosedComment))?;
    if !stripped.notice.iter().any(|line| line.contains(MIT_GRANT)) {
        return Err(error(1, ErrorKind::NoNotice));
    }
    let mut reader = Reader {
        header: Header {
            prefix: prefix.clone(),
            notice: stripped.notice,
            class: None,
            methods: Vec::new(),
            fields: Vec::new(),
            constants: Vec::new(),
        },
        method_lines: Vec::new(),
        seen: BTreeMap::new(),
        method: None,
        field: None,
        pending: None,
    };
    for (index, line) in stripped.lines.iter().enumerate() {
        let number = index + 1;
        let Some(define) = directive(line).map_err(|kind| error(number, kind))? else {
            continue;
        };
        reader
            .define(&prefix, only, number, &define)
            .map_err(|(line, kind)| error(line, kind))?;
    }
    reader.finish().map_err(|(line, kind)| error(line, kind))
}

/// The file with its comments blanked out, and the text of its first comment.
struct Stripped {
    lines: Vec<String>,
    notice: Vec<String>,
}

/// Removes `/* */` and `//` comments, keeping the line structure, and returns the first block
/// comment's lines as the notice. On an unclosed comment, returns the line it opens on.
fn strip_comments(text: &str) -> Result<Stripped, usize> {
    let mut lines = Vec::new();
    let mut first_comment: Option<Vec<String>> = None;
    let mut in_comment: Option<(usize, String)> = None;
    for (index, raw) in text.lines().enumerate() {
        let mut kept = String::new();
        let mut rest = raw;
        loop {
            if let Some((_, comment)) = in_comment.as_mut() {
                if let Some(end) = rest.find("*/") {
                    comment.push_str(&rest[..end]);
                    rest = &rest[end + 2..];
                    if let Some((_, finished)) = in_comment.take() {
                        if first_comment.is_none() {
                            first_comment = Some(finished.lines().map(str::to_owned).collect());
                        }
                    }
                    kept.push(' ');
                    continue;
                }
                comment.push_str(rest);
                comment.push('\n');
                break;
            }
            let block = rest.find("/*");
            let line = rest.find("//");
            match (block, line) {
                (Some(start), Some(slash)) if slash < start => {
                    kept.push_str(&rest[..slash]);
                    break;
                }
                (Some(start), _) => {
                    kept.push_str(&rest[..start]);
                    in_comment = Some((index + 1, String::new()));
                    rest = &rest[start + 2..];
                }
                (None, Some(slash)) => {
                    kept.push_str(&rest[..slash]);
                    break;
                }
                (None, None) => {
                    kept.push_str(rest);
                    break;
                }
            }
        }
        lines.push(kept);
    }
    if let Some((line, _)) = in_comment {
        return Err(line);
    }
    let notice = first_comment.map(clean_notice).unwrap_or_default();
    Ok(Stripped { lines, notice })
}

/// Removes the decoration of a block comment: leading space and `*`s on each line, trailing
/// space, and empty lines at either end.
fn clean_notice(lines: Vec<String>) -> Vec<String> {
    let cleaned: Vec<String> = lines
        .iter()
        .map(|line| {
            let line = line.trim_start().trim_start_matches('*');
            line.strip_prefix(' ').unwrap_or(line).trim_end().to_owned()
        })
        .collect();
    let start = cleaned.iter().position(|line| !line.is_empty());
    let end = cleaned.iter().rposition(|line| !line.is_empty());
    match (start, end) {
        (Some(start), Some(end)) => cleaned[start..=end].to_vec(),
        _ => Vec::new(),
    }
}

/// A `#define` line, split.
struct Define<'a> {
    name: &'a str,
    value: Option<&'a str>,
    rest: Option<String>,
}

/// Reads a comment-free line: `None` for a line that is not a `#define` (or a skipped
/// directive), the define otherwise.
fn directive(line: &str) -> Result<Option<Define<'_>>, ErrorKind> {
    let Some(after_hash) = line.trim_start().strip_prefix('#') else {
        return Ok(None);
    };
    let mut words = after_hash.split_whitespace();
    let Some(keyword) = words.next() else {
        return Ok(None);
    };
    match keyword {
        "ifndef" | "ifdef" | "endif" | "include" => return Ok(None),
        "define" => {}
        other => return Err(ErrorKind::Directive(other.to_owned())),
    }
    let name = words.next().ok_or(ErrorKind::NoName)?;
    let value = words.next();
    let rest: Vec<&str> = words.collect();
    let rest = (!rest.is_empty()).then(|| {
        let mut all = vec![value.unwrap_or_default()];
        all.extend(rest);
        all.join(" ")
    });
    Ok(Some(Define { name, value, rest }))
}

/// The value of a define under the class prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Number(u32),
    Bits(Bits),
    Array { base: u32, strides: [Option<u32>; 2] },
}

/// A number held back until the next define says whether it is a method or a constant.
struct Pending {
    name: String,
    value: u32,
    line: usize,
}

/// Where the current field lives.
#[derive(Debug, Clone, Copy)]
enum FieldAt {
    /// Field `.1` of method `.0`.
    Method(usize, usize),
    /// Field `.0` of the header's fields of no method.
    Free(usize),
}

/// The grouping state while a header is read.
struct Reader {
    header: Header,
    method_lines: Vec<usize>,
    /// Every name read so far (without index parameters, upper-cased), with its line and its
    /// define as written (name and value).
    seen: BTreeMap<String, (usize, String)>,
    method: Option<usize>,
    field: Option<FieldAt>,
    pending: Option<Pending>,
}

/// An error and the line it belongs to.
type Located = (usize, ErrorKind);

impl Reader {
    /// Reads the define on line `line` and files it (see the module docs).
    fn define(
        &mut self,
        prefix: &str,
        only: Option<&str>,
        line: usize,
        define: &Define<'_>,
    ) -> Result<(), Located> {
        match self.check(prefix, only, line, define) {
            Ok(Some((base, kind))) => self.place(base, kind, line),
            Ok(None) => Ok(()),
            Err(kind) => Err((line, kind)),
        }
    }

    /// Checks a define and reads its value: `None` for one that is skipped (an include guard,
    /// the class id, a typedef, a repeat, or one the filter leaves out), its name without the
    /// class prefix or parameters and its value otherwise.
    fn check<'a>(
        &mut self,
        prefix: &str,
        only: Option<&str>,
        line: usize,
        define: &Define<'a>,
    ) -> Result<Option<(&'a str, Kind)>, ErrorKind> {
        let Some(name) = define.name.strip_prefix(prefix) else {
            return self.unprefixed(define).map(|()| None);
        };
        if only.is_some_and(|only| !name.starts_with(only)) {
            return Ok(None);
        }
        let Some(value) = define.value else {
            return Err(ErrorKind::NoValue(define.name.to_owned()));
        };
        if let Some(rest) = &define.rest {
            return Err(ErrorKind::ExtraText {
                name: define.name.to_owned(),
                rest: rest.clone(),
            });
        }
        if name.ends_with("TYPEDEF") && is_identifier(value) {
            return Ok(None);
        }
        let (base, params) = split_name(name).ok_or_else(|| ErrorKind::Name(name.to_owned()))?;
        let key = base.to_ascii_uppercase();
        let written = format!("{} {value}", define.name);
        if let Some((first, raw)) = self.seen.get(&key) {
            return if *raw == written {
                Ok(None)
            } else {
                Err(ErrorKind::Redefined {
                    name: base.to_owned(),
                    first: *first,
                })
            };
        }
        self.seen.insert(key, (line, written));
        Ok(Some((base, read_value(name, params, value)?)))
    }

    /// A define outside the class prefix: an include guard or the class id.
    fn unprefixed(&mut self, define: &Define<'_>) -> Result<(), ErrorKind> {
        let Some(value) = define.value else {
            return Ok(());
        };
        let id = parse_number(strip_parens(value))
            .ok_or_else(|| ErrorKind::Unprefixed(define.name.to_owned()))??;
        if let Some(first) = &self.header.class {
            return Err(ErrorKind::SecondClass {
                name: define.name.to_owned(),
                first: first.name.clone(),
            });
        }
        self.header.class = Some(ClassId {
            name: define.name.to_owned(),
            id,
        });
        Ok(())
    }

    /// Files a define under the current method and field (see the module docs). A misaligned
    /// method address is reported on the method's own line, which for a plain method is the
    /// line before its first field.
    fn place(&mut self, name: &str, kind: Kind, line: usize) -> Result<(), Located> {
        match kind {
            Kind::Array { base, strides } => {
                self.flush_pending();
                let strides: Vec<u32> = strides.into_iter().flatten().collect();
                for value in std::iter::once(base).chain(strides.iter().copied()) {
                    check_aligned(name, value).map_err(|kind| (line, kind))?;
                }
                self.start_method(name, base, strides, line);
                Ok(())
            }
            Kind::Number(value) => {
                self.flush_pending();
                if let Some(field) = self.field
                    && let Some(part) = name.strip_prefix(&self.full_name(field)).and_then(after_underscore)
                {
                    self.field_mut(field).values.push(Value {
                        name: part.to_owned(),
                        value,
                    });
                    return Ok(());
                }
                if let Some(method) = self.method
                    && let Some((index, part)) = self.method_field_for(method, name)
                {
                    self.header.methods[method].fields[index]
                        .values
                        .push(Value { name: part, value });
                    return Ok(());
                }
                self.pending = Some(Pending {
                    name: name.to_owned(),
                    value,
                    line,
                });
                Ok(())
            }
            Kind::Bits(bits) => {
                if let Some(pending) = self.pending.take() {
                    if matches!(bits, Bits::Word { .. })
                        && name.strip_prefix(&pending.name).and_then(after_underscore).is_some()
                    {
                        check_aligned(&pending.name, pending.value)
                            .map_err(|kind| (pending.line, kind))?;
                        self.start_method(&pending.name, pending.value, Vec::new(), pending.line);
                    } else {
                        self.header.constants.push(Value {
                            name: pending.name,
                            value: pending.value,
                        });
                    }
                }
                if let Bits::Word { .. } = bits
                    && let Some(method) = self.method
                    && let Some(part) = name
                        .strip_prefix(self.header.methods[method].name.as_str())
                        .and_then(after_underscore)
                {
                    let fields = &mut self.header.methods[method].fields;
                    fields.push(Field {
                        name: part.to_owned(),
                        bits,
                        values: Vec::new(),
                    });
                    self.field = Some(FieldAt::Method(method, fields.len() - 1));
                    return Ok(());
                }
                self.header.fields.push(Field {
                    name: name.to_owned(),
                    bits,
                    values: Vec::new(),
                });
                self.method = None;
                self.field = Some(FieldAt::Free(self.header.fields.len() - 1));
                Ok(())
            }
        }
    }

    fn start_method(&mut self, name: &str, byte_address: u32, strides: Vec<u32>, line: usize) {
        self.header.methods.push(Method {
            name: name.to_owned(),
            byte_address,
            strides,
            fields: Vec::new(),
        });
        self.method_lines.push(line);
        self.method = Some(self.header.methods.len() - 1);
        self.field = None;
    }

    /// A held-back number that no field followed is a constant.
    fn flush_pending(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.header.constants.push(Value {
                name: pending.name,
                value: pending.value,
            });
        }
    }

    /// The full name (without the class prefix) of a field.
    fn full_name(&self, field: FieldAt) -> String {
        match field {
            FieldAt::Method(method, index) => {
                let method = &self.header.methods[method];
                format!("{}_{}", method.name, method.fields[index].name)
            }
            FieldAt::Free(index) => self.header.fields[index].name.clone(),
        }
    }

    fn field_mut(&mut self, field: FieldAt) -> &mut Field {
        match field {
            FieldAt::Method(method, index) => &mut self.header.methods[method].fields[index],
            FieldAt::Free(index) => &mut self.header.fields[index],
        }
    }

    /// The field of `method` whose full name `name` extends with `_` (the longest such field),
    /// and the rest of the name.
    fn method_field_for(&self, method: usize, name: &str) -> Option<(usize, String)> {
        let method = &self.header.methods[method];
        method
            .fields
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                let full = format!("{}_{}", method.name, field.name);
                let part = name.strip_prefix(full.as_str()).and_then(after_underscore)?;
                Some((full.len(), index, part.to_owned()))
            })
            .max_by_key(|(length, _, _)| *length)
            .map(|(_, index, part)| (index, part))
    }

    fn finish(mut self) -> Result<Header, (usize, ErrorKind)> {
        self.flush_pending();
        for (method, line) in self.header.methods.iter().zip(&self.method_lines) {
            if !method.strides.is_empty() && method.fields.is_empty() {
                return Err((*line, ErrorKind::EmptyArray(method.name.clone())));
            }
        }
        Ok(self.header)
    }
}

/// `rest` without a leading `_`, if it has one and something follows it.
fn after_underscore(rest: &str) -> Option<&str> {
    rest.strip_prefix('_').filter(|rest| !rest.is_empty())
}

fn check_aligned(name: &str, value: u32) -> Result<(), ErrorKind> {
    if value % 4 == 0 {
        Ok(())
    } else {
        Err(ErrorKind::Unaligned {
            name: name.to_owned(),
            value,
        })
    }
}

/// Whether `text` is a C identifier.
fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Splits `NAME(i,j)` into `NAME` and its parameters. The name must be letters, digits and
/// underscores, starting with an upper-case letter; the parameters, if any, `i`, `j` or `i,j`.
fn split_name(name: &str) -> Option<(&str, Vec<char>)> {
    let (base, params) = match name.split_once('(') {
        Some((base, rest)) => {
            let inner = rest.strip_suffix(')')?;
            let params: Vec<char> = match inner {
                "i" => vec!['i'],
                "j" => vec!['j'],
                "i,j" => vec!['i', 'j'],
                _ => return None,
            };
            (base, params)
        }
        None => (name, Vec::new()),
    };
    let mut chars = base.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_uppercase())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    valid.then_some((base, params))
}

/// Removes any number of enclosing parentheses.
fn strip_parens(mut text: &str) -> &str {
    while let Some(inner) = text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        text = inner;
    }
    text
}

/// Reads a hex (`0x`) or decimal number: `None` if `text` is not one, `Some(Err)` if it is one
/// that does not fit in 32 bits.
fn parse_number(text: &str) -> Option<Result<u32, ErrorKind>> {
    let (digits, radix) = match text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
    {
        Some(hex) => (hex, 16),
        None => (text, 10),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    Some(u32::from_str_radix(digits, radix).map_err(|_| ErrorKind::TooBig(text.to_owned())))
}

/// Reads a decimal bit number.
fn parse_bit(text: &str) -> Option<Result<u32, ErrorKind>> {
    if text.starts_with("0x") || text.starts_with("0X") {
        return None;
    }
    parse_number(text)
}

/// Reads the value of the define `name` (with its parameters `params`).
fn read_value(name: &str, params: Vec<char>, value: &str) -> Result<Kind, ErrorKind> {
    let unknown = || ErrorKind::UnknownValue(value.to_owned());
    let index_error = || ErrorKind::Index {
        name: name.to_owned(),
        value: value.to_owned(),
    };
    if let Some(inner) = value.strip_prefix("MW(").and_then(|v| v.strip_suffix(')')) {
        let (high, low) = inner.split_once(':').ok_or_else(unknown)?;
        if let (Some(high), Some(low)) = (parse_bit(high), parse_bit(low)) {
            if !params.is_empty() {
                return Err(index_error());
            }
            let (high, low) = (high?, low?);
            check_range(high, low, 32)?;
            return Ok(Kind::Bits(Bits::Multi { high, low }));
        }
        let (high, high_terms) = read_sum(strip_parens(high)).ok_or_else(unknown)??;
        let (low, low_terms) = read_sum(strip_parens(low)).ok_or_else(unknown)??;
        let ([(high_var, high_stride)], [(low_var, low_stride)]) =
            (high_terms.as_slice(), low_terms.as_slice())
        else {
            return Err(index_error());
        };
        if params.as_slice() != [*high_var] || high_var != low_var {
            return Err(index_error());
        }
        if high_stride != low_stride {
            return Err(ErrorKind::Stride(name.to_owned()));
        }
        check_range(high, low, 32)?;
        return Ok(Kind::Bits(Bits::MultiArray {
            high,
            low,
            stride: *high_stride,
        }));
    }
    if let Some((high, low)) = value.split_once(':')
        && let (Some(high), Some(low)) = (parse_bit(high), parse_bit(low))
    {
        if !params.is_empty() {
            return Err(index_error());
        }
        let (high, low) = (high?, low?);
        check_range(high, low, 0)?;
        return Ok(Kind::Bits(Bits::Word { high, low }));
    }
    let bare = strip_parens(value);
    if let Some(number) = parse_number(bare) {
        if !params.is_empty() {
            return Err(index_error());
        }
        return Ok(Kind::Number(number?));
    }
    let (base, terms) = read_sum(bare).ok_or_else(unknown)??;
    let vars: Vec<char> = terms.iter().map(|(var, _)| *var).collect();
    if terms.is_empty() || vars != params {
        return Err(index_error());
    }
    let mut strides = [None; 2];
    for (slot, (_, stride)) in strides.iter_mut().zip(&terms) {
        *slot = Some(*stride);
    }
    Ok(Kind::Array { base, strides })
}

/// Checks a bit range: not inverted, and either within one word (`multi_width == 0`) or at
/// most `multi_width` bits wide.
fn check_range(high: u32, low: u32, multi_width: u32) -> Result<(), ErrorKind> {
    if high < low {
        return Err(ErrorKind::Inverted { high, low });
    }
    if multi_width == 0 {
        if high > 31 {
            return Err(ErrorKind::PastWord { high, low });
        }
    } else if high - low >= multi_width {
        return Err(ErrorKind::TooWide { high, low });
    }
    Ok(())
}

/// A sum `base+(v)*n+(w)*m`: the base and each index term, as an index name and its stride.
type Sum = (u32, Vec<(char, u32)>);

/// Reads `base+(v)*n+(w)*m`: the base and each index term. `None` if the text has another
/// shape; `Some(Err)` if a number in it does not fit in 32 bits.
fn read_sum(text: &str) -> Option<Result<Sum, ErrorKind>> {
    let mut parts = text.split('+');
    let base = match parse_number(parts.next()?)? {
        Ok(base) => base,
        Err(error) => return Some(Err(error)),
    };
    let mut terms = Vec::new();
    for part in parts {
        let (var, stride) = part.split_once('*')?;
        let var = match var {
            "(i)" => 'i',
            "(j)" => 'j',
            _ => return None,
        };
        let stride = match parse_number(stride)? {
            Ok(stride) => stride,
            Err(error) => return Some(Err(error)),
        };
        terms.push((var, stride));
    }
    (terms.len() <= 2).then_some(Ok((base, terms)))
}
