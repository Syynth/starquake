//! Tallies what a NUP title's `RomFS` holds, for poking around a dump of your own:
//!
//! ```sh
//! cargo run --release -p starquake-loader --example nup_tally -- <title.nup>
//! ```
//!
//! For each package it prints the top-level folders and the file extensions, largest first, with
//! the first bytes of one file per extension (its magic, usually). When the title has two
//! packages it also counts the files the second adds, removes and changes against the first, by
//! folder. It only reads: the table of contents, and eight bytes of one file per extension.
//!
//! The output describes a title's files, so it stays on your machine like everything else read
//! from a dump (`docs/specs/observability.md`, "Storage and what may be committed").

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool: it prints its tally on stdout and errors on stderr"
)]

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;

use starquake_loader::nup::layout::SHA256_SIZE;
use starquake_loader::nup::{FileId, Nup, Package};

/// How many rows of each table are printed.
const ROWS: usize = 40;
/// How many bytes of a sample file are shown.
const MAGIC_BYTES: usize = 8;
/// Bytes in a mebibyte.
const MIB: u64 = 1 << 20;

/// A count of files and their total uncompressed size.
#[derive(Default, Clone, Copy)]
struct Total {
    files: u64,
    bytes: u64,
}

impl Total {
    fn add(&mut self, size: u64) {
        self.files = self.files.saturating_add(1);
        self.bytes = self.bytes.saturating_add(size);
    }

    /// The size in mebibytes with one decimal place, without floating point.
    fn mib(self) -> String {
        let tenths = self.bytes.saturating_mul(10) / MIB;
        format!("{}.{}", tenths / 10, tenths % 10)
    }
}

/// The first folder of `path`, or a marker for a file at the root.
fn top_folder(path: &str) -> &str {
    match path.split_once('/') {
        Some((folder, _)) => folder,
        None => "(root files)",
    }
}

/// Everything from the first dot of the file name, lower-cased: `a/b.bntx.zs` gives `.bntx.zs`.
fn extension(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.find('.') {
        Some(dot) => name.get(dot..).unwrap_or_default().to_lowercase(),
        None => "(none)".to_owned(),
    }
}

/// The first two folders of `path`, the grouping the comparison uses.
fn two_folders(path: &str) -> String {
    let mut parts = path.split('/');
    let first = parts.next().unwrap_or_default();
    match parts.next() {
        Some(second) => format!("{first}/{second}"),
        None => first.to_owned(),
    }
}

/// The first bytes of `file`, with anything unprintable shown as a dot.
fn magic(nup: &mut Nup<File>, file: FileId) -> String {
    let mut bytes = [0_u8; MAGIC_BYTES];
    let read = nup
        .reader(file)
        .ok()
        .and_then(|mut reader| reader.read(&mut bytes).ok())
        .unwrap_or(0);
    bytes
        .iter()
        .take(read)
        .map(|byte| {
            if byte.is_ascii_graphic() {
                char::from(*byte)
            } else {
                '.'
            }
        })
        .collect()
}

/// Rows sorted by size, largest first.
fn by_size<T>(table: BTreeMap<String, T>, total: impl Fn(&T) -> Total) -> Vec<(String, T)> {
    let mut rows: Vec<_> = table.into_iter().collect();
    rows.sort_by_key(|(_, value)| Reverse(total(value).bytes));
    rows
}

/// A package's files by path, each with its content hash.
type Hashes = BTreeMap<String, [u8; SHA256_SIZE]>;

/// Prints one package's tables and returns its files' hashes for the comparison.
fn tally(nup: &mut Nup<File>, package: &Package) -> Option<Hashes> {
    let files = nup.walk_files(package.romfs?);
    let mut folders: BTreeMap<String, Total> = BTreeMap::new();
    let mut extensions: BTreeMap<String, (Total, FileId)> = BTreeMap::new();
    let mut hashes = Hashes::new();
    for (path, id) in &files {
        let Some(entry) = nup.file(*id) else { continue };
        let path = String::from_utf8_lossy(path)
            .trim_start_matches('/')
            .to_owned();
        folders
            .entry(top_folder(&path).to_owned())
            .or_default()
            .add(entry.size);
        let row = extensions
            .entry(extension(&path))
            .or_insert((Total::default(), *id));
        row.0.add(entry.size);
        hashes.insert(path, entry.sha256);
    }

    println!(
        "== package {} ({}): {} RomFS files",
        package.index,
        package.kind.name(),
        files.len()
    );
    println!("-- top-level folders: files, MiB");
    for (name, total) in by_size(folders, |total| *total).iter().take(ROWS) {
        println!("  {name:<28} {:>8} {:>10}", total.files, total.mib());
    }
    let rows = by_size(extensions, |(total, _)| *total);
    println!(
        "-- extensions ({} distinct): files, MiB, first bytes of one file",
        rows.len()
    );
    for (name, (total, sample)) in rows.iter().take(ROWS) {
        let shown = magic(nup, *sample);
        println!(
            "  {name:<28} {:>8} {:>10}  {shown}",
            total.files,
            total.mib()
        );
    }
    Some(hashes)
}

/// Prints what `newer` adds, removes and changes against `older`, grouped by folder.
fn compare(older: &Hashes, newer: &Hashes) {
    let added = newer.keys().filter(|path| !older.contains_key(*path));
    let changed = newer
        .iter()
        .filter(|(path, hash)| older.get(*path).is_some_and(|old| old != *hash))
        .map(|(path, _)| path);
    let removed = older
        .keys()
        .filter(|path| !newer.contains_key(*path))
        .count();
    let mut folders: BTreeMap<String, u64> = BTreeMap::new();
    let (mut added_count, mut changed_count) = (0_u64, 0_u64);
    for path in added {
        added_count = added_count.saturating_add(1);
        *folders.entry(two_folders(path)).or_default() += 1;
    }
    for path in changed {
        changed_count = changed_count.saturating_add(1);
        *folders.entry(two_folders(path)).or_default() += 1;
    }
    println!(
        "== second package against the first: {added_count} added, {removed} removed, {changed_count} changed"
    );
    let mut rows: Vec<_> = folders.into_iter().collect();
    rows.sort_by_key(|(_, count)| Reverse(*count));
    for (name, count) in rows.iter().take(ROWS) {
        println!("  {name:<44} {count}");
    }
}

fn run(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut nup = Nup::open(File::open(path)?)?;
    let packages = nup.packages().to_vec();
    let tallies: Vec<Hashes> = packages
        .iter()
        .filter_map(|package| tally(&mut nup, package))
        .collect();
    if let [older, newer] = tallies.as_slice() {
        compare(older, newer);
    }
    Ok(())
}

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: nup_tally <title.nup>");
        return ExitCode::from(2);
    };
    match run(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nup_tally: {error}");
            ExitCode::FAILURE
        }
    }
}
