//! `starquake inspect <file>` and `starquake verify <file>`: what a NUP title holds, and whether
//! it is intact.
//!
//! Both open the file through the loader's NUP reader (`starquake_core::loader::nup`), which
//! reads only the superblock and the table of contents up front, so a title of tens of
//! gigabytes is never loaded whole.
//!
//! - `inspect` prints the container, its packages, the title (base and newest update), the
//!   `ExeFS` that title runs with each file's size and codec, a summary of its `main.npdm` (the
//!   loader's NPDM parser), and the header of each `ExeFS` module (the loader's NSO parser).
//! - `verify` checks every file's SHA-256 (the table of contents' own hash is checked when the
//!   file is opened) and reports the first file that does not match.
//!
//! Exit codes ([`exit`]): 0 when the command did what it was asked and the title is intact, 1
//! when the title is damaged or something in it could not be parsed, 2 when the file could not
//! be read. [`exit_code_for`] makes that choice for a loader error.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

use starquake_core::loader::npdm::service_access::ServiceAccess;
use starquake_core::loader::npdm::{AcidLayout, AddressSpaceType, Npdm};
use starquake_core::loader::nso::{Nso, NsoError};
use starquake_core::loader::nup::{Codec, FileId, Node, Nup, NupError, Package, Title};

/// The exit codes of `inspect` and `verify`.
pub mod exit {
    /// The command did what it was asked, and the title is intact.
    pub const OK: u8 = 0;
    /// The title is damaged, or a part of it could not be parsed.
    pub const DAMAGED: u8 = 1;
    /// The file could not be read.
    pub const UNREADABLE: u8 = 2;
}

/// The largest `main.npdm` `inspect` reads: 1 MiB. A starquake limit; an NPDM is a few
/// kilobytes.
pub const NPDM_READ_LIMIT: u64 = 1 << 20;
/// The largest `ExeFS` module `inspect` reads whole to parse its header and segments: 1 GiB.
/// A starquake limit, so a corrupt size cannot make it allocate more.
pub const NSO_READ_LIMIT: u64 = 1 << 30;

/// The exit code for a loader error: [`exit::UNREADABLE`] when the source could not be read,
/// [`exit::DAMAGED`] for anything the container itself got wrong.
pub fn exit_code_for(error: &NupError) -> u8 {
    if error.is_io() {
        exit::UNREADABLE
    } else {
        exit::DAMAGED
    }
}

/// An open title file.
type Opened = Nup<BufReader<File>>;

/// Opens `path` as a NUP container, or says why not on `stderr` and returns the exit code.
fn open(path: &Path, stderr: &mut dyn Write) -> Result<Opened, u8> {
    let file = File::open(path).map_err(|error| {
        let _ = writeln!(
            stderr,
            "starquake: could not open {}: {error}",
            path.display()
        );
        exit::UNREADABLE
    })?;
    Nup::open(BufReader::new(file)).map_err(|error| {
        let _ = writeln!(stderr, "starquake: {}: {error}", path.display());
        exit_code_for(&error)
    })
}

/// Runs `starquake verify <path>` and returns the exit code.
pub fn verify(path: &Path, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let mut nup = match open(path, stderr) {
        Ok(nup) => nup,
        Err(code) => return code,
    };
    match nup.verify() {
        Ok(summary) => {
            let _ = writeln!(
                stdout,
                "starquake: {} is intact: the table of contents and all {} files ({} distinct \
                 extents, {} bytes) match their SHA-256",
                path.display(),
                summary.files,
                summary.extents,
                summary.bytes
            );
            exit::OK
        }
        Err(error) => {
            let _ = writeln!(stderr, "starquake: {}: {error}", path.display());
            let _ = writeln!(
                stdout,
                "starquake: {} is damaged: the first file that failed is {} in package {}",
                path.display(),
                error.path,
                error.package
            );
            exit_code_for(&error.source)
        }
    }
}

/// Runs `starquake inspect <path>` and returns the exit code.
pub fn inspect(path: &Path, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let mut nup = match open(path, stderr) {
        Ok(nup) => nup,
        Err(code) => return code,
    };
    let mut out = String::new();
    let mut problems = Vec::new();
    describe_container(&nup, &mut out);
    let code = match nup.title() {
        Err(error) => {
            problems.push(format!("no title: {error}"));
            exit_code_for(&error)
        }
        Ok(title) => {
            describe_title(&mut nup, &title, &mut out, &mut problems);
            if problems.is_empty() {
                exit::OK
            } else {
                exit::DAMAGED
            }
        }
    };
    let _ = stdout.write_all(out.as_bytes());
    for problem in problems {
        let _ = writeln!(stderr, "starquake: {}: {problem}", path.display());
    }
    code
}

/// `version` as the spec packs it, for display: `131072 (2.0.0)`.
fn version_text(package: &Package) -> String {
    let (major, minor, micro) = package.version_parts();
    format!("{} ({major}.{minor}.{micro})", package.version)
}

/// The container line and one line per package.
fn describe_container(nup: &Opened, out: &mut String) {
    let packages = nup.packages();
    let _ = writeln!(
        out,
        "container: NUP version 1, {} packages, {} directories, {} files",
        packages.len(),
        nup.dir_count(),
        nup.file_count()
    );
    for package in packages {
        let mut line = format!(
            "package {}: {}, title {:016x}, version {}",
            package.index,
            package.kind.name(),
            package.title_id,
            version_text(package)
        );
        if package.standalone_update {
            line.push_str(", standalone");
        }
        for (label, dir) in [("exefs", package.exefs), ("romfs", package.romfs)] {
            let Some(dir) = dir else {
                let _ = write!(line, ", no {label}");
                continue;
            };
            let files = nup.walk_files(dir);
            let bytes: u64 = files
                .iter()
                .filter_map(|(_, file)| nup.file(*file))
                .map(|entry| entry.size)
                .fold(0, u64::saturating_add);
            let _ = write!(line, ", {label} {} files ({bytes} bytes)", files.len());
        }
        let _ = writeln!(out, "{line}");
    }
}

/// The title, its `ExeFS`, its `main.npdm` and its modules.
fn describe_title(nup: &mut Opened, title: &Title, out: &mut String, problems: &mut Vec<String>) {
    let program = nup.packages().get(title.program).cloned();
    let Some(program) = program else {
        return;
    };
    let mut line = format!("title: {:016x}", program.title_id);
    if let Some(base) = title.base {
        let _ = write!(line, ", base in package {base}");
    }
    match title.update {
        Some(update) => {
            let _ = write!(line, ", newest update in package {update}");
        }
        None => line.push_str(", no update"),
    }
    let _ = writeln!(out, "{line}");

    let Some(exefs) = title.exefs else {
        problems.push("the title has no ExeFS".to_owned());
        return;
    };
    let _ = writeln!(out, "exefs (from package {}):", exefs.package);
    for file in title.exefs_files(nup) {
        let Some(entry) = nup.file(file) else {
            continue;
        };
        let name = String::from_utf8_lossy(nup.name(Node::File(file)));
        let stored = match entry.codec {
            Codec::Raw => "raw".to_owned(),
            Codec::Zstd => format!("zstd, {} bytes stored", entry.extent_size),
        };
        let _ = writeln!(out, "  {name:<16} {:>12} bytes  {stored}", entry.size);
    }

    match title.exefs_file(nup, "main.npdm") {
        None => problems.push("the ExeFS has no main.npdm".to_owned()),
        Some(file) => match read(nup, file, NPDM_READ_LIMIT) {
            Err(error) => problems.push(format!("main.npdm: {error}")),
            Ok(bytes) => match Npdm::parse(&bytes) {
                Err(error) => problems.push(format!("main.npdm: {error}")),
                Ok(npdm) => describe_npdm(&npdm, out),
            },
        },
    }

    let modules: Vec<FileId> = title
        .exefs_files(nup)
        .into_iter()
        .filter(|&file| nup.name(Node::File(file)) != b"main.npdm")
        .collect();
    if !modules.is_empty() {
        let _ = writeln!(out, "modules:");
    }
    for file in modules {
        let name = String::from_utf8_lossy(nup.name(Node::File(file))).into_owned();
        match read(nup, file, NSO_READ_LIMIT) {
            Err(error) => problems.push(format!("{name}: {error}")),
            Ok(bytes) => match Nso::parse(&bytes) {
                Ok(nso) => describe_nso(&name, &nso, out),
                // A known gap of the loader, not damage: the module is an NSO whose segments
                // it cannot decompress yet.
                Err(NsoError::UnsupportedCompression) => {
                    let _ = writeln!(
                        out,
                        "  {name:<16} NSO with ZBIC-compressed segments, which the loader does \
                         not decompress yet"
                    );
                }
                Err(error) => {
                    problems.push(format!("{name}: not an NSO this loader reads: {error}"));
                }
            },
        }
    }
}

fn read(nup: &mut Opened, file: FileId, limit: u64) -> Result<Vec<u8>, NupError> {
    nup.read_to_vec(file, limit)
}

/// The `main.npdm` summary.
fn describe_npdm(npdm: &Npdm, out: &mut String) {
    let address_space = match npdm.address_space {
        AddressSpaceType::Is32Bit => "32-bit",
        AddressSpaceType::Is36Bit => "36-bit",
        AddressSpaceType::Is32BitNoMap => "32-bit, no map region",
        AddressSpaceType::Is39Bit => "39-bit",
    };
    let layout = match npdm.acid.layout {
        AcidLayout::Switch1 => "Switch 1",
        AcidLayout::Switch2 => "Switch 2",
    };
    let caps = &npdm.aci.kernel_capabilities;
    let _ = writeln!(out, "main.npdm:");
    let _ = writeln!(out, "  name: {}", npdm.name);
    let _ = writeln!(out, "  program id (ACI0): {:016x}", npdm.aci.title_id);
    let _ = writeln!(
        out,
        "  {}, address space {address_space}",
        if npdm.is_64_bit { "64-bit" } else { "32-bit" }
    );
    let _ = writeln!(
        out,
        "  main thread: priority {}, core {}, stack {:#x} bytes",
        npdm.main_thread_priority, npdm.main_thread_core, npdm.main_thread_stack_size
    );
    let _ = writeln!(
        out,
        "  system resource size: {:#x} bytes",
        npdm.system_resource_size
    );
    let _ = writeln!(
        out,
        "  ACID layout: {layout} (magic at {:#x})",
        npdm.acid.layout.magic_offset()
    );
    match &caps.thread_info {
        Some(info) => {
            let mask = core_mask(*info.cores.start(), *info.cores.end());
            let _ = writeln!(
                out,
                "  cores {}-{} (core mask {mask:#x}), priorities {}-{}",
                info.cores.start(),
                info.cores.end(),
                info.priorities.start(),
                info.priorities.end()
            );
        }
        None => {
            let _ = writeln!(out, "  no thread info capability");
        }
    }
    let svcs: Vec<String> = caps
        .svc_mask
        .iter()
        .map(|id| format!("{id:#04x}"))
        .collect();
    let _ = writeln!(
        out,
        "  allowed system calls ({}): {}",
        svcs.len(),
        svcs.join(" ")
    );
    let (hosted, opened): (Vec<_>, Vec<_>) = npdm.aci.services.iter().partition(|s| s.server);
    let names = |list: &[&ServiceAccess]| {
        list.iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let _ = writeln!(
        out,
        "  services it may open ({}): {}",
        opened.len(),
        names(&opened)
    );
    if !hosted.is_empty() {
        let _ = writeln!(
            out,
            "  services it may host ({}): {}",
            hosted.len(),
            names(&hosted)
        );
    }
}

/// The mask of cores `first..=last`: bit `n` set for core `n` (cores past bit 63 are left out).
pub fn core_mask(first: u8, last: u8) -> u64 {
    (first..=last)
        .filter_map(|core| 1u64.checked_shl(u32::from(core)))
        .fold(0, |mask, bit| mask | bit)
}

/// One module's line.
fn describe_nso(name: &str, nso: &Nso, out: &mut String) {
    let build_id = nso.build_id.iter().fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    });
    let _ = writeln!(
        out,
        "  {name:<16} build id {build_id}, text {:#x}, rodata {:#x}, data {:#x}, bss {:#x}",
        nso.text.data.len(),
        nso.rodata.data.len(),
        nso.data.data.len(),
        nso.bss_size
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_io_error_is_unreadable_and_any_other_error_is_damage() {
        assert_eq!(
            exit_code_for(&NupError::Io(std::io::Error::other("gone"))),
            exit::UNREADABLE
        );
        assert_eq!(exit_code_for(&NupError::TocHashMismatch), exit::DAMAGED);
        assert_eq!(exit_code_for(&NupError::NoProgram), exit::DAMAGED);
        // The contract `USAGE` documents.
        assert_eq!([exit::OK, exit::DAMAGED, exit::UNREADABLE], [0, 1, 2]);
    }

    #[test]
    fn the_core_mask_has_one_bit_per_core_in_the_range() {
        assert_eq!(core_mask(0, 0), 0b1);
        assert_eq!(core_mask(0, 2), 0b111);
        assert_eq!(core_mask(1, 3), 0b1110);
        assert_eq!(core_mask(63, 63), 1 << 63);
        assert_eq!(core_mask(62, 255), (1 << 63) | (1 << 62));
        assert_eq!(core_mask(3, 2), 0);
    }
}
