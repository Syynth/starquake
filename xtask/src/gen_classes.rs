//! `cargo xtask gen-classes`: the GPU register tables of `crates/gpu/src/classes/`, generated
//! from NVIDIA's class headers in Mesa (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` in
//! the reference folder, `docs/references.md`). GPU protocol slice G3
//! (`docs/specs/gpu-protocol.md`, "Port slices"; decision 6 of that spec).
//!
//! [`SOURCES`] lists the headers and the module each becomes. [`parse`] reads a header and
//! [`emit`] writes it as Rust; the output carries the header's own MIT notice. The generated
//! files are committed, and the gate's `classes` step regenerates them in memory and fails if a
//! committed file differs, is missing, or is not one this list generates. CI has no reference
//! folder, so there the step is skipped with a notice ([`gate_check`]); `cargo xtask
//! gen-classes` and `cargo xtask gen-classes --check` need the headers and fail without them.

pub mod emit;
pub mod parse;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::walk::{self, Unreadable};

/// One class header and the module generated from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Source {
    /// The header's file name in the class header folder; the module is named after it.
    pub header: &'static str,
    /// When set, only the defines whose name after the class prefix starts with this are read
    /// (the QMD header holds several QMD versions).
    pub only: Option<&'static str>,
    /// The first line of the generated module's documentation.
    pub about: &'static str,
}

/// The headers the GPU crate uses: the classes listed in `docs/specs/gpu-protocol.md` (summary,
/// "Ampere classes are modelled natively", and row G3), QMD 3.0 and the texture header.
pub const SOURCES: &[Source] = &[
    Source {
        header: "cl902d.h",
        only: None,
        about: "The 2D engine class, `FERMI_TWOD_A` (`0x902D`).",
    },
    Source {
        header: "cla140.h",
        only: None,
        about: "The inline-to-memory class, `KEPLER_INLINE_TO_MEMORY_B` (`0xA140`).",
    },
    Source {
        header: "clc76f.h",
        only: None,
        about: "The host methods of `AMPERE_CHANNEL_GPFIFO_B` (`0xC76F`), and the GPFIFO entry \
                and push-buffer method header layouts.",
    },
    Source {
        header: "clc797.h",
        only: None,
        about: "The 3D class, `AMPERE_B` (`0xC797`).",
    },
    Source {
        header: "clc797tex.h",
        only: None,
        about: "The texture header (block-linear, pitch and 1D forms) and the texture sampler \
                of the 3D class.",
    },
    Source {
        header: "clc7b5.h",
        only: None,
        about: "The copy engine class, `AMPERE_DMA_COPY_B` (`0xC7B5`).",
    },
    Source {
        header: "clc7c0.h",
        only: None,
        about: "The compute class, `AMPERE_COMPUTE_B` (`0xC7C0`).",
    },
    Source {
        header: "clc7c0qmd.h",
        only: Some("QMDV03_00_"),
        about: "The compute class's queue metadata (QMD), version `03_00`.",
    },
];

/// Where the generated modules go, relative to the workspace root.
pub const OUTPUT_FOLDER: &str = "crates/gpu/src/classes";

/// The file name of the module generated from `source` (`clc797.h` gives `clc797.rs`).
pub fn module_file(source: &Source) -> String {
    let stem = source.header.strip_suffix(".h").unwrap_or(source.header);
    format!("{stem}.rs")
}

/// Why the tables could not be generated.
#[derive(Debug, thiserror::Error)]
pub enum GenError {
    /// A header could not be read from disk.
    #[error(transparent)]
    Read(#[from] Unreadable),
    /// A header is not in a form the reader understands.
    #[error(transparent)]
    Parse(#[from] parse::ParseError),
}

/// The module generated from the header text `text` of `source`.
pub fn generate(source: &Source, text: &str) -> Result<String, parse::ParseError> {
    let header = parse::parse(source.header, text, source.only)?;
    Ok(emit::emit(source, &header))
}

/// Every module of [`SOURCES`], as (file name, text), reading each header with `read`.
pub fn generate_all(
    read: &dyn Fn(&str) -> Result<String, Unreadable>,
) -> Result<Vec<(String, String)>, GenError> {
    SOURCES
        .iter()
        .map(|source| {
            let text = read(source.header)?;
            Ok((module_file(source), generate(source, &text)?))
        })
        .collect()
}

/// How a committed module differs from a fresh generation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Difference {
    /// The file's text is not what the generator writes.
    #[error(
        "{OUTPUT_FOLDER}/{0} differs from what its header generates: run `cargo xtask \
         gen-classes` (and do not edit generated files by hand)"
    )]
    Changed(String),
    /// A module the generator writes is not committed.
    #[error("{OUTPUT_FOLDER}/{0} is missing: run `cargo xtask gen-classes`")]
    Missing(String),
    /// A file in the folder that no header generates.
    #[error(
        "{OUTPUT_FOLDER}/{0} is generated from no header in xtask's list (SOURCES in \
         xtask/src/gen_classes.rs): delete it, or add its header to the list"
    )]
    Extra(String),
}

/// Compares the generated modules with the files committed in the output folder (file name to
/// text), in file-name order: generated files first, then files no header generates.
pub fn compare(
    generated: &[(String, String)],
    committed: &BTreeMap<String, String>,
) -> Vec<Difference> {
    let mut differences = Vec::new();
    for (file, text) in generated {
        match committed.get(file) {
            None => differences.push(Difference::Missing(file.clone())),
            Some(committed) if committed != text => {
                differences.push(Difference::Changed(file.clone()));
            }
            Some(_) => {}
        }
    }
    for file in committed.keys() {
        if !generated.iter().any(|(generated, _)| generated == file) {
            differences.push(Difference::Extra(file.clone()));
        }
    }
    differences
}

/// What the gate's `classes` step does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateCheck {
    /// Regenerate in memory and compare with the committed files.
    Compare,
    /// The reference folder has no class headers (as in CI): skip, with a notice.
    SkipNoHeaders,
}

/// Decides the gate's `classes` step from whether the class header folder exists. Under CI as
/// locally: CI has no reference folder, so it always skips.
pub fn gate_check(headers_present: bool) -> GateCheck {
    if headers_present {
        GateCheck::Compare
    } else {
        GateCheck::SkipNoHeaders
    }
}

/// The class header folder of the reference folder that `tools/refs-path.sh` prints, run from
/// the workspace root `root`; `None` if the script cannot be run.
pub fn header_folder(root: &Path) -> Option<PathBuf> {
    let output = Command::new("sh")
        .arg("tools/refs-path.sh")
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let refs = String::from_utf8(output.stdout).ok()?;
    Some(Path::new(refs.trim()).join(emit::HEADER_FOLDER))
}

/// Generates every module from the headers in `headers`.
pub fn generate_from(headers: &Path) -> Result<Vec<(String, String)>, GenError> {
    generate_all(&|file| walk::read_text(&headers.join(file)))
}

/// The `.rs` files committed in the output folder below `root`, by file name.
pub fn committed(root: &Path) -> Result<BTreeMap<String, String>, Unreadable> {
    let folder = root.join(OUTPUT_FOLDER);
    let mut files = BTreeMap::new();
    let entries = fs::read_dir(&folder).map_err(walk::unreadable(&folder))?;
    for entry in entries {
        let path = entry.map_err(walk::unreadable(&folder))?.path();
        if path.extension().is_some_and(|extension| extension == "rs")
            && let Some(name) = path.file_name().and_then(|name| name.to_str())
        {
            files.insert(name.to_owned(), walk::read_text(&path)?);
        }
    }
    Ok(files)
}

/// Writes the generated modules into the output folder below `root`.
pub fn write(root: &Path, generated: &[(String, String)]) -> Result<(), Unreadable> {
    let folder = root.join(OUTPUT_FOLDER);
    fs::create_dir_all(&folder).map_err(walk::unreadable(&folder))?;
    for (file, text) in generated {
        let path = folder.join(file);
        fs::write(&path, text).map_err(walk::unreadable(&path))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
