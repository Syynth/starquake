//! The `host-access` gate step: kernel and service code does not reach the host clock, host
//! randomness, the host file system, the network or thread sleeping directly.
//!
//! Replay (`docs/specs/observability.md`, "Replaying the kernel and services") only works if
//! everything nondeterministic that `crates/kernel`, `crates/services` and `crates/gsv` consume
//! comes through the injectable `Env`: a live run records what it returned and a replay feeds the
//! recording back. A direct `Instant::now()` makes a replay diverge silently, so this step bans
//! the calls in [`RULES`] from the non-test code of those crates ([`CRATES`]).
//!
//! # Why not clippy
//!
//! `disallowed-methods` and `disallowed-types` are configured once per `clippy.toml`, and clippy
//! reads one file per crate (the nearest one), which replaces the workspace file instead of
//! extending it. A per-crate file would copy the workspace settings and drift from them, a
//! workspace-wide one would also ban the calls in `core`, the frontends and `xtask`, which own
//! the live environment, and neither can name a whole module (`std::fs`) or a crate (`rand`).
//! This step reads the source text instead.
//!
//! # What is checked
//!
//! Every `.rs` file below `src/` of the crates, except test code. Test code is exactly:
//! an item, field, variant, parameter or statement under `#[cfg(test)]` or `#[cfg(all(test, ..))]`
//! (a predicate that cannot hold without `test`; `any(test, ..)` and `not(test)` are not test
//! code, because they compile into builds that are not tests), a `#[test]` function or one under
//! a path ending in `test` (`#[tokio::test]`), a file whose first item is preceded only by inner
//! attributes and that has `#![cfg(test)]`, the body of a block that holds `#![cfg(test)]`, and
//! the files of a `#[cfg(test)] mod name;` declaration. Folders other than `src/` (`tests/`,
//! `benches/`, `examples/`) and `build.rs` are not read. Comments and string, character and
//! byte-string literals are ignored.
//!
//! A path is resolved through the file's `use` declarations and `extern crate` items
//! (`use std::time::Instant as Clock;` makes `Clock::now` mean `std::time::Instant::now`;
//! `extern crate rand as r;` makes `r::random` mean `rand::random`; groups, `self` and `as` are
//! understood) and then compared with each rule: it is banned when it starts with the rule's
//! path. A `use` declaration is checked the same way, so `use std::fs::File;` is a finding by
//! itself. A name that no `use` introduces is also tried below each glob import of the file
//! (`use std::time::*;` then `Instant::now()` reads as `std::time::Instant::now`). Qualified
//! paths are read like plain ones: `<std::time::Instant>::now()` is `std::time::Instant::now`.
//! A name after a `.` is a method, never a path.
//!
//! Modules are not told apart: every import in the file that introduces the first segment of a
//! path is tried, and any banned result is a finding, so a harmless import of a name in one module
//! does not hide a banned import of it in another (and the reverse can report a call that is
//! fine).
//!
//! # What is not checked
//!
//! This is a tripwire for the listed direct calls, not a sandbox. It does not see:
//!
//! - Method calls on a value, which a text search cannot tie to a type: `at.elapsed()` and
//!   `UNIX_EPOCH.elapsed()` read the host clock (only the associated-function forms
//!   `Instant::elapsed(&at)` and `SystemTime::elapsed(&at)` are rules), and `Path::exists`,
//!   `metadata`, `read_dir`, `canonicalize` and `is_file` on a `std::path::Path` or `PathBuf`
//!   touch the host file system. Test: `what_a_text_search_cannot_see_stays_unreported_and_is_listed_in_the_module_docs`.
//! - A banned item named through a type alias (`type Clock = std::time::Instant;`), a macro that
//!   expands to a banned call, or a trait-qualified path (`<Instant as Clock>::now()`).
//! - Code in another crate that these crates call.
//! - Other crates that read the host clock or randomness (`rand_core::OsRng`, `tokio::time`, the
//!   `time` crate), and any dependency renamed in a manifest (`rng = { package = "rand" }`):
//!   only `rand`, `getrandom` and `fastrand` are rules, under their own names.
//! - `std::os` below `unix::fs` and `unix::net` (`std::os::linux`, `std::os::fd`).
//! - A hasher seeded by `HashMap::new()` (the standard map seeds itself from the host;
//!   CLAUDE.md's rule against iterating one where order matters covers that).
//! - Host state other than the listed items (`std::env`, `std::process`).
//!
//! There is no escape hatch: the way to satisfy the step is to read the value from `Env`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::walk;

/// The crates the step covers, relative to the workspace root. `crates/gsv` is a scaffold today;
/// it is listed so that its first line of code is already under the rule.
pub const CRATES: [&str; 3] = ["crates/kernel", "crates/services", "crates/gsv"];

/// One banned path: any path that starts with these segments.
#[derive(Debug, PartialEq, Eq)]
pub struct Rule {
    /// The banned path, as segments.
    pub path: &'static [&'static str],
    /// What reaching it means, for the message.
    pub what: &'static str,
}

/// The banned paths. A path is banned when it starts with one of these.
pub const RULES: [Rule; 19] = [
    Rule {
        path: &["std", "time", "Instant", "now"],
        what: "the host monotonic clock",
    },
    Rule {
        path: &["std", "time", "SystemTime", "now"],
        what: "the host wall clock",
    },
    Rule {
        path: &["std", "time", "Instant", "elapsed"],
        what: "the host monotonic clock",
    },
    Rule {
        path: &["std", "time", "SystemTime", "elapsed"],
        what: "the host wall clock",
    },
    Rule {
        path: &["std", "fs"],
        what: "the host file system",
    },
    Rule {
        path: &["std", "os", "unix", "fs"],
        what: "the host file system",
    },
    Rule {
        path: &["std", "os", "unix", "net"],
        what: "the host network",
    },
    Rule {
        path: &["std", "io", "stdin"],
        what: "host standard input",
    },
    Rule {
        path: &["std", "io", "stdout"],
        what: "host standard output",
    },
    Rule {
        path: &["std", "io", "stderr"],
        what: "host standard error",
    },
    Rule {
        path: &["std", "net"],
        what: "the host network",
    },
    Rule {
        path: &["std", "thread", "sleep"],
        what: "host thread sleeping",
    },
    Rule {
        path: &["std", "thread", "sleep_ms"],
        what: "host thread sleeping",
    },
    Rule {
        path: &["std", "thread", "park_timeout"],
        what: "host thread sleeping",
    },
    Rule {
        path: &["std", "hash", "RandomState"],
        what: "a hasher seeded from host randomness",
    },
    Rule {
        path: &["std", "collections", "hash_map", "RandomState"],
        what: "a hasher seeded from host randomness",
    },
    Rule {
        path: &["rand"],
        what: "a random-number source",
    },
    Rule {
        path: &["getrandom"],
        what: "a host random-number source",
    },
    Rule {
        path: &["fastrand"],
        what: "a random-number source",
    },
];

/// One use of a banned path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The 1-based line the path starts on.
    pub line: usize,
    /// The path as it resolves, with `::` between segments (a glob import ends in `::*`).
    pub path: String,
    /// What the path reaches.
    pub what: &'static str,
}

/// What reading one file found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Scan {
    /// Uses of banned paths outside test code.
    pub findings: Vec<Finding>,
    /// Names of modules declared as `#[cfg(test)] mod name;`: their files are test code.
    pub test_modules: Vec<String>,
    /// The whole file is test code (`#![cfg(test)]`).
    pub all_test: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Ident(String),
    PathSep,
    Punct(char),
    Literal,
}

#[derive(Debug, Clone)]
struct Token {
    kind: Kind,
    line: usize,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Splits Rust source into tokens, leaving out comments and whitespace and reducing every
/// literal to [`Kind::Literal`]. It is not a full lexer: it knows enough to tell a lifetime from
/// a character literal, to skip raw strings and nested block comments, and to count lines.
fn tokenize(source: &str) -> Vec<Token> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut line = 1;
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let next = chars.get(i + 1).copied();
        let start_line = line;
        let kind = match c {
            '\n' => {
                line += 1;
                i += 1;
                continue;
            }
            c if c.is_whitespace() => {
                i += 1;
                continue;
            }
            '/' if next == Some('/') => {
                while chars.get(i).is_some_and(|&c| c != '\n') {
                    i += 1;
                }
                continue;
            }
            '/' if next == Some('*') => {
                i = skip_block_comment(&chars, i, &mut line);
                continue;
            }
            '"' => {
                i = skip_string(&chars, i + 1, &mut line);
                Kind::Literal
            }
            '\'' => {
                let (end, kind) = lex_quote(&chars, i);
                i = end;
                kind
            }
            c if c.is_ascii_digit() => {
                while chars.get(i).copied().is_some_and(is_ident_continue) {
                    i += 1;
                }
                Kind::Literal
            }
            c if is_ident_start(c) => {
                let (end, kind) = lex_word(&chars, i, &mut line);
                i = end;
                kind
            }
            ':' if next == Some(':') => {
                i += 2;
                Kind::PathSep
            }
            c => {
                i += 1;
                Kind::Punct(c)
            }
        };
        tokens.push(Token {
            kind,
            line: start_line,
        });
    }
    tokens
}

/// The index after the (possibly nested) block comment that starts at `i`.
fn skip_block_comment(chars: &[char], mut i: usize, line: &mut usize) -> usize {
    let mut depth = 0usize;
    while let Some(&c) = chars.get(i) {
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('*') {
            depth += 1;
            i += 2;
        } else if c == '*' && next == Some('/') {
            depth -= 1;
            i += 2;
            if depth == 0 {
                break;
            }
        } else {
            if c == '\n' {
                *line += 1;
            }
            i += 1;
        }
    }
    i
}

/// A `'` at `i` starts a character literal (`'x'`, `'\n'`, `'\u{..}'`) or is the quote of a
/// lifetime. Returns the index after what it read and the token.
fn lex_quote(chars: &[char], i: usize) -> (usize, Kind) {
    if chars.get(i + 1) == Some(&'\\') {
        // Skip the backslash and the escaped character, then to the closing quote.
        let mut end = i + 3;
        while chars.get(end).is_some_and(|&c| c != '\'') {
            end += 1;
        }
        (end + 1, Kind::Literal)
    } else if chars.get(i + 2) == Some(&'\'') && chars.get(i + 1) != Some(&'\'') {
        (i + 3, Kind::Literal)
    } else {
        (i + 1, Kind::Punct('\''))
    }
}

/// An identifier, a keyword, or a prefixed literal (`b".."`, `c".."`, `r#".."#`, `br".."`) that
/// starts at `i`. Returns the index after it and the token.
fn lex_word(chars: &[char], i: usize, line: &mut usize) -> (usize, Kind) {
    let mut end = i;
    while chars.get(end).copied().is_some_and(is_ident_continue) {
        end += 1;
    }
    let word: String = chars[i..end].iter().collect();
    let next = chars.get(end).copied();
    if matches!(word.as_str(), "b" | "c") && next == Some('"') {
        return (skip_string(chars, end + 1, line), Kind::Literal);
    }
    if matches!(word.as_str(), "r" | "br" | "cr") && matches!(next, Some('"' | '#')) {
        let mut quote = end;
        while chars.get(quote) == Some(&'#') {
            quote += 1;
        }
        let hashes = quote - end;
        if chars.get(quote) == Some(&'"') {
            return (
                skip_raw_string(chars, quote + 1, hashes, line),
                Kind::Literal,
            );
        }
        if word == "r" && hashes == 1 && chars.get(quote).copied().is_some_and(is_ident_start) {
            // A raw identifier, `r#name`.
            let mut name_end = quote;
            while chars.get(name_end).copied().is_some_and(is_ident_continue) {
                name_end += 1;
            }
            return (
                name_end,
                Kind::Ident(chars[quote..name_end].iter().collect()),
            );
        }
    }
    (end, Kind::Ident(word))
}

/// The index after the closing quote of a string whose body starts at `i`, counting its lines.
fn skip_string(chars: &[char], mut i: usize, line: &mut usize) -> usize {
    while let Some(&c) = chars.get(i) {
        match c {
            '\\' => {
                if chars.get(i + 1) == Some(&'\n') {
                    *line += 1;
                }
                i += 2;
            }
            '"' => return i + 1,
            _ => {
                if c == '\n' {
                    *line += 1;
                }
                i += 1;
            }
        }
    }
    i
}

/// The index after the end of a raw string with `hashes` hashes whose body starts at `i`.
fn skip_raw_string(chars: &[char], mut i: usize, hashes: usize, line: &mut usize) -> usize {
    while let Some(&c) = chars.get(i) {
        if c == '"' && (1..=hashes).all(|n| chars.get(i + n) == Some(&'#')) {
            return i + 1 + hashes;
        }
        if c == '\n' {
            *line += 1;
        }
        i += 1;
    }
    i
}

fn ident(token: &Token) -> Option<&str> {
    match &token.kind {
        Kind::Ident(name) => Some(name),
        _ => None,
    }
}

fn is_punct(token: Option<&Token>, c: char) -> bool {
    token.is_some_and(|token| token.kind == Kind::Punct(c))
}

/// The index of the token that closes the bracket opened at `open` (`(`, `[` or `{`), or the
/// token count when it never closes.
fn matching_close(tokens: &[Token], open: usize) -> usize {
    let mut depth = 0usize;
    for (offset, token) in tokens[open..].iter().enumerate() {
        match token.kind {
            Kind::Punct('(' | '[' | '{') => depth += 1,
            Kind::Punct(')' | ']' | '}') => {
                depth -= 1;
                if depth == 0 {
                    return open + offset;
                }
            }
            _ => {}
        }
    }
    tokens.len()
}

/// Whether the predicate that starts at `tokens[*i]` can hold only in a test build, and moves
/// `*i` past it. `test` does; `all(..)` does when any of its parts does; `any(..)` when all of
/// its parts do (`any(test, feature = "x")` also holds in a build that is not a test); `not(..)`,
/// an option (`feature = "x"`) and every other name do not.
fn predicate_implies_test(tokens: &[Token], i: &mut usize) -> bool {
    let Some(name) = tokens.get(*i).and_then(ident) else {
        *i += 1;
        return false;
    };
    *i += 1;
    if is_punct(tokens.get(*i), '=') {
        *i += 2;
        return false;
    }
    if !is_punct(tokens.get(*i), '(') {
        return name == "test";
    }
    *i += 1;
    let mut parts = Vec::new();
    while *i < tokens.len() && !is_punct(tokens.get(*i), ')') {
        parts.push(predicate_implies_test(tokens, i));
        if is_punct(tokens.get(*i), ',') {
            *i += 1;
        }
    }
    *i += 1;
    match name {
        "all" => parts.contains(&true),
        "any" => !parts.is_empty() && !parts.contains(&false),
        _ => false,
    }
}

/// Whether the tokens inside an attribute's brackets mark test code: a path ending in `test`
/// (`test`, `tokio::test`), or a `cfg(..)` whose predicate holds only in a test build
/// ([`predicate_implies_test`]). `cfg_attr(.., test)` is not: the function stays when the
/// condition is off.
fn is_test_attribute(inner: &[Token]) -> bool {
    let mut path = Vec::new();
    let mut i = 0;
    while let Some(name) = inner.get(i).and_then(ident) {
        path.push(name);
        i += 1;
        if inner.get(i).is_some_and(|t| t.kind == Kind::PathSep) {
            i += 1;
        } else {
            break;
        }
    }
    match path.as_slice() {
        ["cfg"] => {
            let mut at = i + 1;
            is_punct(inner.get(i), '(') && predicate_implies_test(inner, &mut at)
        }
        [.., "test"] => true,
        _ => false,
    }
}

/// Removes the test code from `tokens`: items under a test attribute; a `#![cfg(test)]` that
/// leads the file (which reports the whole file); and the body of a block that holds one, header
/// included (`mod t { #![cfg(test)] .. }`). Returns the remaining tokens, the names of
/// `#[cfg(test)] mod name;` declarations, and whether the whole file is test code.
fn drop_test_code(tokens: &[Token]) -> (Vec<Token>, Vec<String>, bool) {
    let mut kept = Vec::with_capacity(tokens.len());
    let mut modules = Vec::new();
    // For each open `{` that was kept: how many tokens were kept before it, and its index.
    let mut open_blocks: Vec<(usize, usize)> = Vec::new();
    // Whether anything but inner attributes has been kept: an inner attribute only leads the file
    // (applies to all of it) before that.
    let mut started = false;
    let mut i = 0;
    while i < tokens.len() {
        let is_attribute = is_punct(tokens.get(i), '#')
            && (is_punct(tokens.get(i + 1), '[')
                || (is_punct(tokens.get(i + 1), '!') && is_punct(tokens.get(i + 2), '[')));
        if !is_attribute {
            match tokens[i].kind {
                Kind::Punct('{') => open_blocks.push((kept.len(), i)),
                Kind::Punct('}') => {
                    open_blocks.pop();
                }
                _ => {}
            }
            started = true;
            kept.push(tokens[i].clone());
            i += 1;
            continue;
        }
        let inner_attribute = is_punct(tokens.get(i + 1), '!');
        let open = i + if inner_attribute { 2 } else { 1 };
        let close = matching_close(tokens, open);
        let test = is_test_attribute(&tokens[open + 1..close.min(tokens.len())]);
        if test && inner_attribute {
            if let Some((kept_len, brace)) = open_blocks.pop() {
                // Drop the enclosing block and its header (`pub mod t`, `fn f()`).
                kept.truncate(kept_len);
                while kept
                    .last()
                    .is_some_and(|t| !matches!(t.kind, Kind::Punct(';' | '{' | '}')))
                {
                    kept.pop();
                }
                i = matching_close(tokens, brace) + 1;
                continue;
            }
            if !started {
                return (Vec::new(), Vec::new(), true);
            }
        }
        if !test || inner_attribute {
            started |= !inner_attribute;
            kept.extend(tokens[i..=close.min(tokens.len() - 1)].iter().cloned());
            i = close + 1;
            continue;
        }
        i = skip_item(tokens, close + 1, &mut modules);
    }
    (kept, modules, false)
}

/// What follows a test attribute, which decides where the attributed thing ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// An item: it ends at its `;` or at the `}` that matches its first `{`.
    Item,
    /// A `let` statement: it ends at its `;`.
    Statement,
    /// A struct or tuple field, an enum variant, a function parameter or a match arm: it ends at
    /// a `,`, at the end of an arm's block, or at the bracket that closes the list it is in.
    Loose,
}

/// The words that start an item (after attributes and visibility).
const ITEM_KEYWORDS: [&str; 16] = [
    "fn",
    "struct",
    "enum",
    "union",
    "trait",
    "impl",
    "type",
    "mod",
    "use",
    "const",
    "static",
    "unsafe",
    "async",
    "extern",
    "macro_rules",
    "macro",
];

/// The [`Shape`] of what starts at `tokens[i]`, after further attributes and visibility.
fn shape_at(tokens: &[Token], mut i: usize) -> Shape {
    while is_punct(tokens.get(i), '#') && is_punct(tokens.get(i + 1), '[') {
        i = matching_close(tokens, i + 1) + 1;
    }
    if tokens.get(i).and_then(ident) == Some("pub") {
        i += 1;
        if is_punct(tokens.get(i), '(') {
            i = matching_close(tokens, i) + 1;
        }
    }
    match tokens.get(i).map(|token| &token.kind) {
        Some(Kind::Ident(name)) if ITEM_KEYWORDS.contains(&name.as_str()) => Shape::Item,
        Some(Kind::Ident(name)) if name == "let" => Shape::Statement,
        // A macro invocation item: `thread_local! { .. }`.
        Some(Kind::Ident(_)) if is_punct(tokens.get(i + 1), '!') => Shape::Item,
        Some(Kind::Punct('{')) | None => Shape::Item,
        _ => Shape::Loose,
    }
}

/// The index after the thing that starts at `i` (after its test attribute): any further
/// attributes are skipped with it, and where it ends depends on its [`Shape`]. It never consumes
/// a bracket that closes something it did not open. A `mod name;` is recorded in `modules`.
fn skip_item(tokens: &[Token], mut i: usize, modules: &mut Vec<String>) -> usize {
    let shape = shape_at(tokens, i);
    let mut depth = 0usize;
    let mut arrow = false;
    let mut previous: Vec<&str> = Vec::new();
    while let Some(token) = tokens.get(i) {
        match &token.kind {
            Kind::Punct('#') if depth == 0 && is_punct(tokens.get(i + 1), '[') => {
                i = matching_close(tokens, i + 1) + 1;
                continue;
            }
            Kind::Punct('{') if depth == 0 && shape == Shape::Item => {
                return matching_close(tokens, i) + 1;
            }
            Kind::Punct('(' | '[' | '{') => depth += 1,
            Kind::Punct(')' | ']' | '}') => {
                if depth == 0 {
                    return i;
                }
                depth -= 1;
                if depth == 0 && arrow && token.kind == Kind::Punct('}') {
                    // The block that is a match arm's body.
                    return i + 1;
                }
            }
            Kind::Punct(';') if depth == 0 => {
                if let [.., "mod", name] = previous.as_slice() {
                    modules.push((*name).to_owned());
                }
                return i + 1;
            }
            Kind::Punct(',') if depth == 0 && shape == Shape::Loose => return i + 1,
            Kind::Punct('=')
                if depth == 0 && shape == Shape::Loose && is_punct(tokens.get(i + 1), '>') =>
            {
                arrow = true;
            }
            Kind::Ident(name) => previous.push(name),
            _ => {}
        }
        i += 1;
    }
    i
}

/// One name a `use` declaration brings in, or one glob.
#[derive(Debug)]
struct UseEntry {
    path: Vec<String>,
    /// The name the path is known by in the file; `None` for a glob or `as _`.
    name: Option<String>,
    glob: bool,
    line: usize,
}

/// Reads a use tree from `tokens[*i..]` below `prefix` into `out`.
fn parse_use_tree(tokens: &[Token], i: &mut usize, prefix: &[String], out: &mut Vec<UseEntry>) {
    let mut path = prefix.to_vec();
    let line = tokens.get(*i).map_or(0, |token| token.line);
    if tokens.get(*i).is_some_and(|t| t.kind == Kind::PathSep) {
        *i += 1;
    }
    loop {
        match tokens.get(*i) {
            Some(token) if ident(token).is_some() => {
                path.extend(ident(token).map(str::to_owned));
                *i += 1;
            }
            Some(token) if token.kind == Kind::Punct('{') => {
                *i += 1;
                loop {
                    if is_punct(tokens.get(*i), '}') || *i >= tokens.len() {
                        *i += 1;
                        return;
                    }
                    let before = *i;
                    parse_use_tree(tokens, i, &path, out);
                    if is_punct(tokens.get(*i), ',') {
                        *i += 1;
                    } else if *i == before {
                        // Something that is not a use tree: step over it rather than loop.
                        *i += 1;
                    }
                }
            }
            Some(token) if token.kind == Kind::Punct('*') => {
                *i += 1;
                out.push(UseEntry {
                    path,
                    name: None,
                    glob: true,
                    line,
                });
                return;
            }
            _ => return,
        }
        if tokens.get(*i).is_some_and(|t| t.kind == Kind::PathSep) {
            *i += 1;
        } else {
            break;
        }
    }
    // `use a::b::self` (inside a group) names the prefix itself.
    if path.last().map(String::as_str) == Some("self") && path.len() > prefix.len() {
        path.pop();
    }
    let mut name = path.last().cloned();
    if tokens.get(*i).and_then(ident) == Some("as") {
        name = tokens.get(*i + 1).and_then(ident).map(str::to_owned);
        *i += 2;
        if name.as_deref() == Some("_") {
            name = None;
        }
    }
    out.push(UseEntry {
        path,
        name,
        glob: false,
        line,
    });
}

/// Reads every `use` declaration and `extern crate` item. Returns the entries and, for each
/// token index, whether it is part of one of those (those tokens are not read again as path
/// expressions).
///
/// `use` is a declaration only when a path, a group or a glob follows: `+ use<'a, T>` in a return
/// type (Rust 2024 precise capturing) is not.
fn parse_uses(tokens: &[Token]) -> (Vec<UseEntry>, Vec<bool>) {
    let mut entries = Vec::new();
    let mut in_use = vec![false; tokens.len()];
    let mut i = 0;
    while i < tokens.len() {
        let after_path = i > 0 && tokens[i - 1].kind == Kind::PathSep;
        let after_dot = i > 0 && tokens[i - 1].kind == Kind::Punct('.');
        let word = ident(&tokens[i]).filter(|_| !after_path && !after_dot);
        let starts_a_tree = tokens.get(i + 1).is_some_and(|next| {
            ident(next).is_some() || matches!(next.kind, Kind::PathSep | Kind::Punct('{' | '*'))
        });
        if word == Some("use") && starts_a_tree {
            let start = i;
            i += 1;
            parse_use_tree(tokens, &mut i, &[], &mut entries);
            while i < tokens.len() && tokens[i].kind != Kind::Punct(';') {
                i += 1;
            }
            for flag in &mut in_use[start..i.min(tokens.len())] {
                *flag = true;
            }
        } else if word == Some("extern") && tokens.get(i + 1).and_then(ident) == Some("crate") {
            // `extern crate rand as r;` names the crate `rand` as `r`.
            let start = i;
            let line = tokens[i].line;
            let krate = tokens.get(i + 2).and_then(ident);
            i += 3;
            let mut name = krate;
            if tokens.get(i).and_then(ident) == Some("as") {
                name = tokens.get(i + 1).and_then(ident);
                i += 2;
            }
            if let Some(krate) = krate {
                entries.push(UseEntry {
                    path: vec![krate.to_owned()],
                    name: name.filter(|name| *name != "_").map(str::to_owned),
                    glob: false,
                    line,
                });
            }
            for flag in &mut in_use[start..i.min(tokens.len())] {
                *flag = true;
            }
            continue;
        }
        i += 1;
    }
    (entries, in_use)
}

/// The words that can come before a path without being part of one. A path that follows one of
/// them with a leading `::` (`return ::std::fs::read(..)`) starts a new path. The path keywords
/// (`crate`, `self`, `super`, `Self`) are not in the list.
const NON_PATH_KEYWORDS: [&str; 34] = [
    "as", "async", "await", "box", "break", "const", "continue", "dyn", "else", "enum", "extern",
    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "static", "struct", "trait", "type", "unsafe", "use", "where", "while", "yield",
];

/// A qualified path, `<std::time::Instant>::now`, whose `<` is at `tokens[i]`: its segments (the
/// type's, then the rest) and the index after it. `<T as Trait>::f` is not one (its meaning
/// depends on `Trait`), and neither is a comparison or a generic argument list.
fn qualified_path(tokens: &[Token], in_use: &[bool], i: usize) -> Option<(Vec<String>, usize)> {
    let follows_path_sep = i > 0 && tokens[i - 1].kind == Kind::PathSep;
    if in_use[i] || tokens[i].kind != Kind::Punct('<') || follows_path_sep {
        return None;
    }
    let mut j = i + 1;
    if tokens.get(j)?.kind == Kind::PathSep {
        j += 1;
    }
    let mut segments = vec![ident(tokens.get(j)?)?.to_owned()];
    j += 1;
    while tokens.get(j)?.kind == Kind::PathSep {
        segments.push(ident(tokens.get(j + 1)?)?.to_owned());
        j += 2;
    }
    if !is_punct(tokens.get(j), '>') || tokens.get(j + 1)?.kind != Kind::PathSep {
        return None;
    }
    j += 2;
    segments.push(ident(tokens.get(j)?)?.to_owned());
    j += 1;
    while tokens.get(j).is_some_and(|t| t.kind == Kind::PathSep)
        && let Some(next) = tokens.get(j + 1).and_then(ident)
    {
        segments.push(next.to_owned());
        j += 2;
    }
    Some((segments, j))
}

/// The paths written in expressions, types and patterns: maximal `a::b::c` chains outside `use`
/// declarations, and qualified paths (`<a::B>::c`). A leading `::` is dropped; `::<` ends a
/// chain; a name after a `.` is a method or field, not a path.
fn read_paths(tokens: &[Token], in_use: &[bool]) -> Vec<(usize, Vec<String>)> {
    let mut paths = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if let Some((segments, end)) = qualified_path(tokens, in_use, i) {
            paths.push((tokens[i].line, segments));
            i = end;
            continue;
        }
        let Some(first) = ident(&tokens[i]).filter(|_| !in_use[i]) else {
            i += 1;
            continue;
        };
        let continues_a_path = i >= 2
            && tokens[i - 1].kind == Kind::PathSep
            && ident(&tokens[i - 2]).is_some_and(|word| !NON_PATH_KEYWORDS.contains(&word));
        let after_dot = i > 0 && tokens[i - 1].kind == Kind::Punct('.');
        // A keyword is not the start of a path: `::std::fs::read` after `return` is its own.
        if continues_a_path || after_dot || NON_PATH_KEYWORDS.contains(&first) {
            i += 1;
            continue;
        }
        let line = tokens[i].line;
        let mut segments = vec![first.to_owned()];
        i += 1;
        while tokens.get(i).is_some_and(|t| t.kind == Kind::PathSep)
            && let Some(next) = tokens.get(i + 1).and_then(ident)
        {
            segments.push(next.to_owned());
            i += 2;
        }
        paths.push((line, segments));
    }
    paths
}

/// Whether `path` starts with all of `prefix`.
fn starts_with(path: &[String], prefix: &[&str]) -> bool {
    path.len() >= prefix.len() && path.iter().zip(prefix).all(|(a, b)| a == b)
}

/// The rule a resolved path breaks, if any.
fn broken_rule(path: &[String]) -> Option<&'static str> {
    RULES
        .iter()
        .find(|rule| starts_with(path, rule.path))
        .map(|rule| rule.what)
}

/// Reads one source file.
pub fn scan(source: &str) -> Scan {
    let (tokens, test_modules, all_test) = drop_test_code(&tokenize(source));
    if all_test {
        return Scan {
            all_test,
            ..Scan::default()
        };
    }
    let (uses, in_use) = parse_uses(&tokens);
    let mut findings = Vec::new();
    for entry in &uses {
        // `use std::fs::*;` is the module itself; the globs of other modules are resolved
        // below, where a name they bring in is used.
        if let Some(what) = broken_rule(&entry.path) {
            let mut path = entry.path.join("::");
            if entry.glob {
                path.push_str("::*");
            }
            findings.push(Finding {
                line: entry.line,
                path,
                what,
            });
        }
    }
    for (line, written) in read_paths(&tokens, &in_use) {
        // Every import in the file that introduces the first segment is tried: the file's modules
        // are not told apart, so a harmless import of the name in one module does not hide a
        // banned one in another.
        let named: Vec<&UseEntry> = uses
            .iter()
            .filter(|entry| {
                !entry.glob && entry.name.as_deref() == written.first().map(String::as_str)
            })
            .collect();
        let candidates: Vec<Vec<String>> = if named.is_empty() {
            // It may come from a glob import, or be written in full; a lone name that no
            // glob explains is a local (`let rand = ..`), not a crate path.
            let mut candidates: Vec<Vec<String>> = uses
                .iter()
                .filter(|entry| entry.glob)
                .map(|entry| entry.path.iter().chain(&written).cloned().collect())
                .collect();
            if written.len() >= 2 {
                candidates.push(written);
            }
            candidates
        } else {
            // The first segment is a name a `use` introduced.
            named
                .iter()
                .map(|entry| {
                    let mut full = entry.path.clone();
                    full.extend(written.iter().skip(1).cloned());
                    full
                })
                .collect()
        };
        let broken = candidates
            .into_iter()
            .filter(|path| path.len() >= 2)
            .find_map(|path| broken_rule(&path).map(|what| (path, what)));
        if let Some((path, what)) = broken {
            findings.push(Finding {
                line,
                path: path.join("::"),
                what,
            });
        }
    }
    findings.sort_by(|a, b| a.line.cmp(&b.line).then_with(|| a.path.cmp(&b.path)));
    Scan {
        findings,
        test_modules,
        all_test,
    }
}

/// Why the step could not reach a verdict.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A listed crate has no `src/` folder, so it would silently go unchecked.
    #[error("{0}: no src/ folder; update CRATES in xtask/src/host_access.rs if the crate moved")]
    MissingCrate(String),
    /// A folder or file could not be read.
    #[error(transparent)]
    Unreadable(#[from] walk::Unreadable),
}

/// One finding in one file.
#[derive(Debug, PartialEq, Eq)]
pub struct Violation {
    /// The file, relative to the workspace root, with `/` separators.
    pub file: String,
    /// What it uses.
    pub finding: Finding,
}

/// The folder that holds the submodules of `file`: next to a `lib.rs`, `main.rs` or `mod.rs`,
/// and in the folder named like the file otherwise.
fn module_folder(file: &Path) -> PathBuf {
    let parent = file.parent().map(Path::to_path_buf).unwrap_or_default();
    match file.file_stem().and_then(|stem| stem.to_str()) {
        Some("lib" | "main" | "mod") | None => parent,
        Some(stem) => parent.join(stem),
    }
}

/// Reads the non-test code of the crate at `folder` (its `src/`), returning violations with
/// paths relative to `root_dir`.
fn check_crate(root_dir: &Path, krate: &str) -> Result<Vec<Violation>, Error> {
    let src = root_dir.join(krate).join("src");
    if !src.is_dir() {
        return Err(Error::MissingCrate(krate.to_owned()));
    }
    let files = walk::rust_files(&src, &|_| false)?;
    let mut scans = Vec::new();
    for file in &files {
        scans.push((file, scan(&walk::read_text(file)?)));
    }
    // Files that are test code because a `#[cfg(test)] mod name;` names them, and everything
    // below such a module.
    let mut test_files: BTreeSet<PathBuf> = BTreeSet::new();
    let mut test_folders: BTreeSet<PathBuf> = BTreeSet::new();
    for (file, scan) in &scans {
        for name in &scan.test_modules {
            let folder = module_folder(file);
            test_files.insert(folder.join(format!("{name}.rs")));
            test_files.insert(folder.join(name).join("mod.rs"));
            test_folders.insert(folder.join(name));
        }
    }
    let mut violations = Vec::new();
    for (file, scan) in scans {
        let is_test_file = scan.all_test
            || test_files.contains(file)
            || test_folders.iter().any(|folder| file.starts_with(folder));
        if is_test_file {
            continue;
        }
        let relative = file
            .strip_prefix(root_dir)
            .unwrap_or(file)
            .components()
            .map(|part| part.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        for finding in scan.findings {
            violations.push(Violation {
                file: relative.clone(),
                finding,
            });
        }
    }
    Ok(violations)
}

/// Runs the step on the checkout at `root_dir`: every crate in [`CRATES`], in that order.
pub fn check(root_dir: &Path) -> Result<Vec<Violation>, Error> {
    let mut violations = Vec::new();
    for krate in CRATES {
        violations.extend(check_crate(root_dir, krate)?);
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// `(line, path)` of every finding in `source`.
    fn found(source: &str) -> Vec<(usize, String)> {
        scan(source)
            .findings
            .into_iter()
            .map(|finding| (finding.line, finding.path))
            .collect()
    }

    fn at(line: usize, path: &str) -> (usize, String) {
        (line, path.to_owned())
    }

    #[test]
    fn every_rule_fires_on_the_call_written_out_in_full() {
        // The deliberate violations: if the step never fired, none of these would be found.
        for (source, path) in [
            (
                "fn f() { let _ = std::time::Instant::now(); }",
                "std::time::Instant::now",
            ),
            (
                "fn f() { let _ = std::time::SystemTime::now(); }",
                "std::time::SystemTime::now",
            ),
            ("fn f() { let _ = std::fs::read(\"x\"); }", "std::fs::read"),
            (
                "fn f() { let _ = std::fs::File::open(\"x\"); }",
                "std::fs::File::open",
            ),
            (
                "fn f() { let _ = std::net::TcpStream::connect(\"x\"); }",
                "std::net::TcpStream::connect",
            ),
            ("fn f() { std::thread::sleep(d); }", "std::thread::sleep"),
            (
                "fn f() { std::thread::sleep_ms(1); }",
                "std::thread::sleep_ms",
            ),
            (
                "fn f() { std::thread::park_timeout(d); }",
                "std::thread::park_timeout",
            ),
            (
                "fn f() { let _ = std::hash::RandomState::new(); }",
                "std::hash::RandomState::new",
            ),
            (
                "fn f() { let _ = std::collections::hash_map::RandomState::new(); }",
                "std::collections::hash_map::RandomState::new",
            ),
            ("fn f() -> u8 { rand::random() }", "rand::random"),
            (
                "fn f() { getrandom::getrandom(&mut b); }",
                "getrandom::getrandom",
            ),
            ("fn f() -> u8 { fastrand::u8(..) }", "fastrand::u8"),
            (
                "fn f() { Instant::elapsed(&at); }\nuse std::time::Instant;",
                "std::time::Instant::elapsed",
            ),
            (
                "fn f() { std::time::SystemTime::elapsed(&at); }",
                "std::time::SystemTime::elapsed",
            ),
            (
                "fn f() { std::os::unix::fs::symlink(a, b); }",
                "std::os::unix::fs::symlink",
            ),
            (
                "fn f() { std::os::unix::net::UnixStream::connect(a); }",
                "std::os::unix::net::UnixStream::connect",
            ),
            ("fn f() { std::io::stdin(); }", "std::io::stdin"),
            ("fn f() { std::io::stdout(); }", "std::io::stdout"),
            ("fn f() { std::io::stderr(); }", "std::io::stderr"),
        ] {
            assert_eq!(found(source), [at(1, path)], "{source}");
        }
    }

    #[test]
    fn the_rules_cover_what_the_issue_asks_for() {
        let covered = |path: &[&str]| RULES.iter().any(|rule| rule.path == path);
        for path in [
            &["std", "time", "Instant", "now"][..],
            &["std", "time", "SystemTime", "now"],
            &["std", "fs"],
            &["std", "net"],
            &["std", "thread", "sleep"],
            &["rand"],
            &["getrandom"],
        ] {
            assert!(covered(path), "{path:?}");
        }
    }

    #[test]
    fn imports_do_not_hide_a_call() {
        for (source, line, path) in [
            (
                "use std::time::Instant;\nfn f() { Instant::now(); }",
                2,
                "std::time::Instant::now",
            ),
            (
                "use std::time::{Duration, Instant};\nfn f() { Instant::now(); }",
                2,
                "std::time::Instant::now",
            ),
            (
                "use std::{time::Instant, io};\nfn f() { Instant::now(); }",
                2,
                "std::time::Instant::now",
            ),
            (
                "use std::time::Instant as Clock;\nfn f() { Clock::now(); }",
                2,
                "std::time::Instant::now",
            ),
            (
                "use std::time;\nfn f() { time::Instant::now(); }",
                2,
                "std::time::Instant::now",
            ),
            (
                "use std::time::{self};\nfn f() { time::SystemTime::now(); }",
                2,
                "std::time::SystemTime::now",
            ),
            (
                "use std::thread;\nfn f() { thread::sleep(d); }",
                2,
                "std::thread::sleep",
            ),
            (
                "use std::thread::sleep;\nfn f() { sleep(d); }",
                2,
                "std::thread::sleep",
            ),
            (
                "use std::fs;\nfn f() { fs::write(\"a\", b); }",
                2,
                "std::fs::write",
            ),
            ("use rand::Rng;\nfn f() { rand::rng(); }", 2, "rand::rng"),
            ("fn f() { ::std::fs::read(\"x\"); }", 1, "std::fs::read"),
        ] {
            let all = found(source);
            assert!(all.contains(&at(line, path)), "{source}: {all:?}");
        }
    }

    #[test]
    fn a_use_declaration_is_a_finding_by_itself() {
        assert_eq!(found("use std::fs::File;"), [at(1, "std::fs::File")]);
        assert_eq!(found("use std::fs;"), [at(1, "std::fs")]);
        assert_eq!(found("use std::{fs, io::Read};"), [at(1, "std::fs")]);
        assert_eq!(
            found("use std::net::{TcpStream, UdpSocket};"),
            [at(1, "std::net::TcpStream"), at(1, "std::net::UdpSocket"),]
        );
        assert_eq!(found("use rand::Rng;"), [at(1, "rand::Rng")]);
        assert_eq!(
            found("use std::thread::sleep;"),
            [at(1, "std::thread::sleep")]
        );
    }

    #[test]
    fn a_glob_import_does_not_hide_a_call() {
        // The glob itself names the banned module: a finding.
        assert_eq!(found("use std::fs::*;"), [at(1, "std::fs::*")]);
        assert_eq!(found("use rand::*;"), [at(1, "rand::*")]);
        // For other modules the name used after the glob is what is found.
        assert_eq!(
            found("use std::time::*;\nfn f() { Instant::now(); }"),
            [at(2, "std::time::Instant::now")]
        );
        assert_eq!(
            found("use std::time::*;\nfn f() { SystemTime::now(); }"),
            [at(2, "std::time::SystemTime::now")]
        );
        assert_eq!(
            found("use std::thread::*;\nfn f() { sleep(d); }"),
            [at(2, "std::thread::sleep")]
        );
        assert_eq!(
            found("use std::collections::*;\nfn f() { hash_map::RandomState::new(); }"),
            [at(2, "std::collections::hash_map::RandomState::new")]
        );
        assert_eq!(
            found("use std::time::Instant::*;\nfn f() { now(); }"),
            [at(2, "std::time::Instant::now")]
        );
    }

    #[test]
    fn globs_that_bring_in_nothing_banned_are_not_findings() {
        let source = "use std::collections::*;\nuse super::*;\nuse crate::env::*;\nuse std::io::*;\n\
                      use std::time::*;\n\
                      fn f() { let d = Duration::from_secs(1); let epoch = SystemTime::UNIX_EPOCH; \
                      Instant::checked_add(&at, d); sleep_for(d); }";
        assert_eq!(found(source), []);
    }

    #[test]
    fn harmless_uses_of_the_same_modules_are_not_findings() {
        let source = "use std::time::{Duration, Instant, SystemTime};\n\
                      use std::thread::{self, JoinHandle};\n\
                      use std::collections::HashMap;\n\
                      fn f(at: Instant) -> Duration {\n\
                          let epoch = SystemTime::UNIX_EPOCH;\n\
                          let h = thread::spawn(|| ());\n\
                          at.checked_duration_since(at).unwrap_or_default() + Duration::from_secs(1)\n\
                      }";
        assert_eq!(found(source), []);
    }

    #[test]
    fn names_that_only_resemble_a_banned_path_are_not_findings() {
        let source = "fn brand() {}\n\
                      fn operand() { rand_helper(); }\n\
                      fn f(s: &S) { s.fs::<u8>(); s.net.read(); let sleep = 3; sleep_for(sleep); }\n\
                      fn g(clock: &dyn Clock) { clock.now(); self.env.rand(); crate::fs::read(); }";
        assert_eq!(found(source), []);
    }

    #[test]
    fn comments_and_literals_are_not_code() {
        let source = "// Instant::now() is banned\n\
                      /* std::fs::read, and /* nested std::net */ */\n\
                      /// ```\n\
                      /// let _ = std::time::SystemTime::now();\n\
                      /// ```\n\
                      fn f() -> &'static str {\n\
                          let _ = \"std::fs::read\";\n\
                          let _ = r#\"Instant::now() \" still inside\"#;\n\
                          let _ = b\"rand::random\";\n\
                          let _ = 'x';\n\
                          \"thread::sleep\"\n\
                      }";
        assert_eq!(found(source), []);
    }

    #[test]
    fn line_numbers_survive_comments_strings_and_literals() {
        let source = "/* one\n\
                      two */\n\
                      const A: &str = \"line\\\n\
                      continued\";\n\
                      const B: &str = r#\"\n\
                      raw\n\
                      \"#;\n\
                      fn f() { let c = '\"'; std::fs::read(\"x\"); }\n\
                      fn g<'a>(x: &'a str) { std::thread::sleep(d); }";
        assert_eq!(
            found(source),
            [at(8, "std::fs::read"), at(9, "std::thread::sleep")]
        );
    }

    #[test]
    fn test_code_is_exempt() {
        let source = "fn live() {}\n\
                      #[cfg(test)]\n\
                      mod tests {\n\
                          use std::fs;\n\
                          #[test]\n\
                          fn t() { std::time::Instant::now(); fs::read(\"x\"); }\n\
                      }\n\
                      #[test]\n\
                      fn stray() { std::thread::sleep(d); }\n\
                      #[cfg(all(test, feature = \"x\"))]\n\
                      fn helper() { rand::random::<u8>(); }\n\
                      #[tokio::test]\n\
                      async fn t2() { std::net::TcpStream::connect(\"x\"); }\n\
                      #[cfg(test)]\n\
                      #[derive(Debug)]\n\
                      struct S(std::fs::File);\n";
        assert_eq!(found(source), []);
    }

    #[test]
    fn code_around_a_test_item_is_still_read() {
        let source = "fn before() { std::fs::read(\"a\"); }\n\
                      #[cfg(test)]\n\
                      mod tests { fn t() { let _ = [1, 2]; } }\n\
                      fn after() { std::fs::read(\"b\"); }\n\
                      #[cfg(test)]\n\
                      const TABLE: [u8; 2] = [1, 2];\n\
                      fn last() { std::fs::read(\"c\"); }";
        assert_eq!(
            found(source),
            [
                at(1, "std::fs::read"),
                at(4, "std::fs::read"),
                at(7, "std::fs::read")
            ]
        );
    }

    #[test]
    fn only_cfg_test_without_not_is_exempt() {
        let source = "#[cfg(not(test))]\n\
                      fn live() { std::fs::read(\"a\"); }\n\
                      #[cfg(feature = \"test\")]\n\
                      fn gated() { std::fs::read(\"b\"); }\n\
                      #[derive(Debug)]\n\
                      #[allow(dead_code)]\n\
                      fn attributed() { std::fs::read(\"c\"); }";
        assert_eq!(
            found(source),
            [
                at(2, "std::fs::read"),
                at(4, "std::fs::read"),
                at(7, "std::fs::read")
            ]
        );
    }

    #[test]
    fn an_inner_cfg_test_marks_the_whole_file() {
        let scanned = scan("#![cfg(test)]\nfn f() { std::fs::read(\"x\"); }");
        assert!(scanned.all_test);
        assert_eq!(scanned.findings, []);
        // Other inner attributes do not.
        let scanned =
            scan("#![expect(unsafe_code, reason = \"r\")]\nfn f() { std::fs::read(\"x\"); }");
        assert!(!scanned.all_test);
        assert_eq!(scanned.findings.len(), 1);
    }

    #[test]
    fn a_test_module_declared_in_another_file_is_recorded_by_name() {
        let scanned = scan("#[cfg(test)]\nmod tests;\n#[cfg(test)]\nmod more;\nmod real;\n");
        assert_eq!(scanned.test_modules, ["tests", "more"]);
        let scanned = scan("#[cfg(test)]\nmod inline { fn f() {} }\n");
        assert_eq!(scanned.test_modules, Vec::<String>::new());
    }

    const LIVE: &str = "std::fs::read(\"x\");";

    #[test]
    fn precise_capturing_use_is_not_a_use_declaration() {
        // Rust 2024 `impl Trait + use<..>` must not swallow the tokens up to the next `;`.
        for (source, line) in [
            (
                "fn f() -> impl Sized + use<> {\n std::fs::read(\"x\"); }",
                2,
            ),
            ("fn f() -> impl Sized + use<> { std::fs::read(\"x\"); }", 1),
            (
                "fn f<'a, T>() -> impl Sized + use<'a, T> { std::fs::read(\"x\"); }",
                1,
            ),
            (
                "fn all(&self) -> impl Iterator<Item = u8> + use<> { std::thread::sleep(d); }",
                1,
            ),
        ] {
            let all = found(source);
            assert_eq!(all.len(), 1, "{source}: {all:?}");
            assert_eq!(all[0].0, line, "{source}");
        }
        // A real `use` declaration after it is still read.
        assert_eq!(
            found("fn f() -> impl Sized + use<> {}\nuse std::fs;"),
            [at(2, "std::fs")]
        );
    }

    #[test]
    fn a_test_attribute_on_a_non_item_does_not_swallow_the_live_code_after_it() {
        for (what, source) in [
            (
                "struct field",
                "struct S { a: u8, #[cfg(test)] probe: u32 }\nfn live() { std::fs::read(\"x\"); }",
            ),
            (
                "last enum variant",
                "enum E { A, #[cfg(test)] B }\nfn live() { std::fs::read(\"x\"); }",
            ),
            (
                "enum variant with data",
                "enum E { A, #[cfg(test)] B(u8), C }\nfn live() { std::fs::read(\"x\"); }",
            ),
            (
                "match arm with a comma",
                "fn live(x: u8) { match x { #[cfg(test)] 1 => {}, _ => { std::fs::read(\"x\"); } } }",
            ),
            (
                "match arm without a comma",
                "fn live(x: u8) { match x { #[cfg(test)] 1 => {} _ => { std::fs::read(\"x\"); } } }",
            ),
            (
                "struct literal field",
                "fn f() -> S { S { a: 1, #[cfg(test)] probe: 2 } }\nfn live() { std::fs::read(\"x\"); }",
            ),
            (
                "function parameter",
                "fn f(a: u8, #[cfg(test)] b: u8) { std::fs::read(\"x\"); }",
            ),
            (
                "last function parameter",
                "fn f(a: u8, #[cfg(test)] b: u8)\n{ std::fs::read(\"x\"); }",
            ),
            (
                "tuple field",
                "struct S(u8, #[cfg(test)] u8);\nfn live() { std::fs::read(\"x\"); }",
            ),
            (
                "statement",
                "fn live() { #[cfg(test)] let x = if a { 1 } else { 2 }; std::fs::read(\"x\"); }",
            ),
        ] {
            let all = found(source);
            assert_eq!(all.len(), 1, "{what}: {all:?}");
            assert_eq!(all[0].1, "std::fs::read", "{what}");
        }
    }

    #[test]
    fn a_test_attribute_on_an_item_still_skips_the_whole_item() {
        // Commas inside generics and parameter lists, visibility, and macro items.
        let source = "#[cfg(test)]\n\
                      fn helper<T, U>(a: T, b: U) { std::fs::read(\"a\"); }\n\
                      #[cfg(test)]\n\
                      pub struct S<A, B>(A, B, std::fs::File);\n\
                      #[cfg(test)]\n\
                      pub(crate) fn g() { std::fs::read(\"b\"); }\n\
                      #[cfg(test)]\n\
                      thread_local! { static X: std::fs::File = open(); }\n\
                      #[cfg(test)]\n\
                      #[derive(Debug)]\n\
                      pub enum E { A(std::fs::File), B }\n\
                      #[cfg(test)]\n\
                      unsafe impl Send for S<u8, u8> {}\n\
                      fn live() { std::fs::read(\"c\"); }";
        assert_eq!(found(source), [at(14, "std::fs::read")]);
    }

    #[test]
    fn an_inner_cfg_test_marks_the_file_only_when_it_leads_the_file() {
        // After other inner attributes it is still the whole file.
        let scanned = scan(
            "#![expect(unsafe_code, reason = \"r\")]\n#![cfg(test)]\nfn f() { std::fs::read(\"x\"); }",
        );
        assert!(scanned.all_test);
        // Inside a module it drops that module and nothing else.
        let scanned = scan("mod t {\n #![cfg(test)]\n}\npub fn live() { std::fs::read(\"x\"); }");
        assert!(!scanned.all_test);
        assert_eq!(
            scanned.findings,
            [Finding {
                line: 4,
                path: "std::fs::read".to_owned(),
                what: "the host file system"
            }]
        );
        let source = "fn live() { std::fs::read(\"a\"); }\n\
                      mod t {\n\
                          #![cfg(test)]\n\
                          fn f() { std::fs::read(\"b\"); }\n\
                          mod deeper { fn g() { std::fs::read(\"c\"); } }\n\
                      }\n\
                      #[derive(Debug)]\n\
                      pub mod u {\n\
                          #![cfg(test)]\n\
                          use std::fs;\n\
                      }\n\
                      fn after() { std::fs::read(\"d\"); }";
        let scanned = scan(source);
        assert!(!scanned.all_test);
        assert_eq!(
            found(source),
            [at(1, "std::fs::read"), at(12, "std::fs::read")]
        );
        // In a function body it drops that body.
        assert_eq!(
            found(
                "fn f() { #![cfg(test)] std::fs::read(\"x\"); }\nfn g() { std::fs::read(\"y\"); }"
            ),
            [at(2, "std::fs::read")]
        );
        // After an item at the top level it cannot be the whole file (and is not valid Rust).
        let scanned = scan("fn live() { std::fs::read(\"x\"); }\n#![cfg(test)]\n");
        assert!(!scanned.all_test);
        assert_eq!(scanned.findings.len(), 1);
    }

    #[test]
    fn only_cfg_test_and_cfg_all_with_test_are_exempt() {
        // Test-only: the predicate cannot hold without `test`.
        for attribute in [
            "#[cfg(test)]",
            "#[cfg(all(test, feature = \"x\"))]",
            "#[cfg(all(feature = \"x\", test))]",
            "#[cfg(all(test, any(feature = \"a\", feature = \"b\")))]",
            "#[cfg(any(test, test))]",
            "#[test]",
            "#[tokio::test]",
            "#[tokio::test(flavor = \"multi_thread\")]",
        ] {
            let source = format!("{attribute}\nfn f() {{ {LIVE} }}");
            assert_eq!(found(&source), [], "{attribute}");
        }
        // Compiled into builds that are not tests: findings.
        for attribute in [
            "#[cfg(any(test, feature = \"x\"))]",
            "#[cfg(any(test, debug_assertions))]",
            "#[cfg(not(test))]",
            "#[cfg(not(any(test)))]",
            "#[cfg(all(feature = \"x\", any(test, unix)))]",
            "#[cfg(feature = \"test\")]",
            "#[cfg_attr(feature = \"bench\", test)]",
            "#[cfg_attr(test, derive(Debug))]",
        ] {
            let source = format!("{attribute}\nfn f() {{ {LIVE} }}");
            assert_eq!(found(&source), [at(2, "std::fs::read")], "{attribute}");
        }
        // Such a module's file is read too.
        let scanned = scan("#[cfg(any(test, feature = \"fakes\"))]\npub mod fake;\n");
        assert_eq!(scanned.test_modules, Vec::<String>::new());
        let scanned = scan("#[cfg(all(test, feature = \"fakes\"))]\npub mod fake;\n");
        assert_eq!(scanned.test_modules, ["fake"]);
    }

    #[test]
    fn every_import_that_introduces_a_name_is_tried() {
        for source in [
            "use crate::env::Instant;\nmod inner { use std::time::Instant; fn f() { Instant::now(); } }",
            "mod a { use crate::env::Instant; }\nuse std::time::Instant;\nfn f() { Instant::now(); }",
            "use std::time::Instant;\nmod a { use crate::env::Instant; }\nfn f() { Instant::now(); }",
        ] {
            let all = found(source);
            assert!(
                all.iter()
                    .any(|(_, path)| path == "std::time::Instant::now"),
                "{source}: {all:?}"
            );
        }
        // A name that only a harmless import introduces is still harmless.
        assert_eq!(
            found("use crate::env::Instant;\nfn f() { Instant::now(); }"),
            []
        );
    }

    #[test]
    fn a_qualified_path_is_read_like_a_plain_one() {
        for (source, path) in [
            (
                "fn f() { <std::time::Instant>::now(); }",
                "std::time::Instant::now",
            ),
            (
                "fn f() { <::std::time::SystemTime>::now(); }",
                "std::time::SystemTime::now",
            ),
            (
                "use std::time::Instant;\nfn f() { <Instant>::now(); }",
                "std::time::Instant::now",
            ),
            (
                "use std::time as t;\nfn f() { <t::Instant>::now(); }",
                "std::time::Instant::now",
            ),
        ] {
            let all = found(source);
            assert!(
                all.iter().any(|(_, found)| found == path),
                "{source}: {all:?}"
            );
        }
        // Comparisons, generics and turbofish are not qualified paths.
        let source = "fn f(a: u8, b: u8, c: u8) -> bool { a < b && b > c }\n\
                      fn g<T>() { let v: Vec<Instant> = Vec::new(); let w = h::<Instant>(); }";
        assert_eq!(found(source), []);
    }

    #[test]
    fn a_leading_path_separator_after_a_keyword_still_starts_a_path() {
        for source in [
            "fn f() { return ::std::fs::read(\"x\"); }",
            "fn f() { match ::std::fs::read(\"x\") { _ => {} } }",
            "fn f() { for _ in ::std::fs::read_dir(\"x\") {} }",
            "fn f() { if ::std::fs::read(\"x\").is_ok() {} }",
            "fn f() { let _ = x as ::std::fs::File; }",
        ] {
            let all = found(source);
            assert_eq!(all.len(), 1, "{source}: {all:?}");
            assert!(all[0].1.starts_with("std::fs::"), "{source}: {all:?}");
        }
        // The path keywords still continue a path and are not read twice.
        assert_eq!(found("fn f() { crate::fs::read(); self::fs::read(); }"), []);
    }

    #[test]
    fn extern_crate_names_are_resolved_like_imports() {
        assert_eq!(
            found("extern crate rand as r;\nfn f() { r::random::<u8>(); }"),
            [at(1, "rand"), at(2, "rand::random")]
        );
        assert_eq!(
            found("extern crate std as s;\nfn f() { s::fs::read(\"x\"); }"),
            [at(2, "std::fs::read")]
        );
        assert_eq!(
            found("extern crate std;\nfn f() { std::thread::sleep(d); }"),
            [at(2, "std::thread::sleep")]
        );
        // `extern` ABI declarations and harmless crates are not findings.
        assert_eq!(
            found("extern crate alloc;\nextern \"C\" fn f() {}\nextern \"C\" { fn g(); }"),
            []
        );
    }

    #[test]
    fn a_method_call_on_a_glob_imported_module_is_not_a_path() {
        // `x.sleep(d)` is a method, not `std::thread::sleep`, whatever is glob-imported.
        assert_eq!(found("use std::thread::*;\nfn f(x: X) { x.sleep(d); }"), []);
        assert_eq!(
            found("use std::time::Instant::*;\nfn f(x: X) { x.now(); }"),
            []
        );
        // (The glob of a banned module is a finding itself, and reads every other name below it.)
        let all = found("use std::fs::*;\nfn f(x: X) { x.read(); }");
        assert!(
            all.iter().all(|(_, path)| path != "std::fs::read"),
            "{all:?}"
        );
        // The same name called as a function is a path.
        assert_eq!(
            found("use std::thread::*;\nfn f(x: X) { x.sleep(d); sleep(d); }"),
            [at(2, "std::thread::sleep")]
        );
    }

    #[test]
    fn what_a_text_search_cannot_see_stays_unreported_and_is_listed_in_the_module_docs() {
        // Each source is a host access this step does not report. The module docs list every one
        // under "What is not checked"; closing a gap means moving its source to a finding test.
        for (what, source) in [
            (
                "a method call that reads the host clock",
                "fn f(at: std::time::Instant) { at.elapsed(); }",
            ),
            (
                "UNIX_EPOCH.elapsed() is the same method",
                "fn f() { std::time::UNIX_EPOCH.elapsed(); }",
            ),
            (
                "file-system methods of Path",
                "use std::path::Path;\nfn f(p: &Path) { p.exists(); p.metadata(); p.read_dir(); p.canonicalize(); }",
            ),
            (
                "file-system methods on a freshly built Path",
                "fn f() { std::path::Path::new(\"x\").exists(); }",
            ),
            (
                "a type alias",
                "type Clock = std::time::Instant;\nfn f() { Clock::now(); }",
            ),
            (
                "a trait-qualified path",
                "fn f() { <std::time::Instant as Clock>::now(); }",
            ),
            (
                "another clock crate",
                "fn f() { tokio::time::sleep(d); time::OffsetDateTime::now_utc(); }",
            ),
            ("another randomness crate", "fn f() { rand_core::OsRng; }"),
            (
                "a dependency renamed in the manifest",
                "fn f() { rng::random::<u8>(); }",
            ),
            (
                "other std::os modules",
                "use std::os::linux::fs::MetadataExt;\nuse std::os::fd::AsRawFd;",
            ),
            (
                "a macro that expands to a banned call",
                "fn f() { my_clock!(); include_bytes!(\"x\"); }",
            ),
        ] {
            assert_eq!(found(source), [], "{what}");
        }
    }

    /// A scratch checkout laid out as `files` says, for the test called `name`, with a `src/`
    /// folder for each covered crate.
    fn checkout(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let root = walk::scratch_checkout("host-access", name, files);
        for krate in CRATES {
            fs::create_dir_all(root.join(krate).join("src")).unwrap();
        }
        root
    }

    fn violations(root: &Path) -> Vec<(String, usize, String)> {
        check(root)
            .unwrap()
            .into_iter()
            .map(|v| (v.file, v.finding.line, v.finding.path))
            .collect()
    }

    #[test]
    fn a_violation_in_each_covered_crate_is_found_with_its_file_and_line() {
        let root = checkout(
            "each",
            &[
                (
                    "crates/kernel/src/lib.rs",
                    "pub fn f() {\n    let _ = std::time::Instant::now();\n}\n",
                ),
                ("crates/services/src/clock/mod.rs", "use std::fs;\n"),
                (
                    "crates/gsv/src/lib.rs",
                    "fn f() { rand::random::<u8>(); }\n",
                ),
            ],
        );
        assert_eq!(
            violations(&root),
            [
                (
                    "crates/kernel/src/lib.rs".to_owned(),
                    2,
                    "std::time::Instant::now".to_owned()
                ),
                (
                    "crates/services/src/clock/mod.rs".to_owned(),
                    1,
                    "std::fs".to_owned()
                ),
                (
                    "crates/gsv/src/lib.rs".to_owned(),
                    1,
                    "rand::random".to_owned()
                ),
            ]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_clean_checkout_has_no_violations() {
        let root = checkout(
            "clean",
            &[(
                "crates/kernel/src/lib.rs",
                "pub fn f(env: &dyn Env) -> u64 { env.now() }\n",
            )],
        );
        assert_eq!(violations(&root), []);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_files_and_folders_other_than_src_are_not_read() {
        let violating = "fn f() { std::fs::read(\"x\"); }\n";
        let all_test = format!("#![cfg(test)]\n{violating}");
        let root = checkout(
            "exempt",
            &[
                // Declared by `#[cfg(test)] mod tests;`, so a test file, with its own submodule.
                (
                    "crates/kernel/src/lib.rs",
                    "#[cfg(test)]\nmod tests;\nmod sched;\n",
                ),
                ("crates/kernel/src/tests.rs", violating),
                ("crates/kernel/src/tests/deep.rs", violating),
                // A module declared in a file that is not lib.rs lives in a folder named after it.
                ("crates/kernel/src/sched.rs", "#[cfg(test)]\nmod tests;\n"),
                ("crates/kernel/src/sched/tests.rs", violating),
                ("crates/kernel/src/sched/live.rs", violating),
                // Not under src/.
                ("crates/kernel/tests/it.rs", violating),
                ("crates/kernel/benches/b.rs", violating),
                ("crates/kernel/build.rs", violating),
                // A whole-file cfg(test).
                ("crates/services/src/only_tests.rs", all_test.as_str()),
            ],
        );
        assert_eq!(
            violations(&root),
            [(
                "crates/kernel/src/sched/live.rs".to_owned(),
                1,
                "std::fs::read".to_owned()
            )]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_test_module_in_mod_rs_exempts_its_sibling_file_only() {
        let violating = "fn f() { std::fs::read(\"x\"); }\n";
        let root = checkout(
            "mod-rs",
            &[
                ("crates/kernel/src/lib.rs", "mod clock;\n"),
                (
                    "crates/kernel/src/clock/mod.rs",
                    "#[cfg(test)]\nmod tests;\n",
                ),
                // `foo/mod.rs` declares `tests` as `foo/tests.rs`...
                ("crates/kernel/src/clock/tests.rs", violating),
                // ...and not as `foo/mod/tests.rs`.
                ("crates/kernel/src/clock/mod/tests.rs", violating),
            ],
        );
        assert_eq!(
            violations(&root),
            [(
                "crates/kernel/src/clock/mod/tests.rs".to_owned(),
                1,
                "std::fs::read".to_owned()
            )]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_module_that_is_test_code_only_with_a_feature_is_read() {
        let violating = "fn f() { std::fs::read(\"x\"); }\n";
        let root = checkout(
            "any-test",
            &[
                (
                    "crates/services/src/lib.rs",
                    "#[cfg(any(test, feature = \"fakes\"))]\npub mod fake;\n\
                     #[cfg(all(test, feature = \"fakes\"))]\nmod only_tests;\n",
                ),
                ("crates/services/src/fake.rs", violating),
                ("crates/services/src/only_tests.rs", violating),
            ],
        );
        assert_eq!(
            violations(&root),
            [(
                "crates/services/src/fake.rs".to_owned(),
                1,
                "std::fs::read".to_owned()
            )]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_covered_crate_without_src_is_an_error_not_a_pass() {
        let root = checkout("missing", &[]);
        fs::remove_dir_all(root.join("crates/services/src")).unwrap();
        let error = check(&root).unwrap_err();
        assert!(matches!(&error, Error::MissingCrate(krate) if krate == "crates/services"));
        assert_eq!(
            error.to_string(),
            "crates/services: no src/ folder; update CRATES in xtask/src/host_access.rs if the crate moved"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_crates_are_the_ones_the_architecture_names() {
        assert_eq!(CRATES, ["crates/kernel", "crates/services", "crates/gsv"]);
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for krate in CRATES {
            assert!(root.join(krate).join("src").is_dir(), "{krate}");
        }
    }

    /// This repository: no covered crate reaches the host directly.
    #[test]
    fn this_repository_is_clean() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        assert_eq!(check(&root).unwrap(), []);
    }
}
