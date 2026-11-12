//! Reading a checkout's Rust files: the one directory walk and file read that the source-scanning
//! steps ([`crate::host_access`], [`crate::table_deps`]) share, and the scratch checkout their
//! tests build.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A folder or file that could not be read.
#[derive(Debug, thiserror::Error)]
#[error("{path}: {source}")]
pub struct Unreadable {
    /// What was being read.
    pub path: String,
    /// The operating-system error.
    pub source: io::Error,
}

/// Turns an I/O error on `path` into an [`Unreadable`], for `map_err`.
pub fn unreadable(path: &Path) -> impl FnOnce(io::Error) -> Unreadable {
    let path = path.display().to_string();
    move |source| Unreadable { path, source }
}

/// The text of the file at `path`.
pub fn read_text(path: &Path) -> Result<String, Unreadable> {
    fs::read_to_string(path).map_err(unreadable(path))
}

/// Every `.rs` file below `folder`, in path order, leaving out each file or folder for which
/// `skip` is true (a skipped folder is not entered).
pub fn rust_files(folder: &Path, skip: &dyn Fn(&Path) -> bool) -> Result<Vec<PathBuf>, Unreadable> {
    let mut files = Vec::new();
    collect(folder, skip, &mut files)?;
    Ok(files)
}

fn collect(
    folder: &Path,
    skip: &dyn Fn(&Path) -> bool,
    files: &mut Vec<PathBuf>,
) -> Result<(), Unreadable> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(folder).map_err(unreadable(folder))? {
        entries.push(entry.map_err(unreadable(folder))?.path());
    }
    entries.sort();
    for path in entries {
        if skip(&path) {
            continue;
        }
        if path.is_dir() {
            collect(&path, skip, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

/// A scratch checkout laid out as `files` says (path and text), in a folder unique to `prefix`,
/// `name` and the test process. The caller removes it.
#[cfg(test)]
pub fn scratch_checkout(prefix: &str, name: &str, files: &[(&str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("{prefix}-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    for (path, text) in files {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_files_are_listed_in_path_order_and_skips_apply_to_files_and_folders() {
        let root = scratch_checkout(
            "walk",
            "order",
            &[
                ("b.rs", ""),
                ("a/z.rs", ""),
                ("a/y.txt", ""),
                ("skip_me/in.rs", ""),
                ("c/skip.rs", ""),
                ("c/keep.rs", ""),
            ],
        );
        let skip = |path: &Path| {
            path.file_name()
                .is_some_and(|name| name == "skip_me" || name == "skip.rs")
        };
        let files = rust_files(&root, &skip).unwrap();
        let relative: Vec<_> = files
            .iter()
            .map(|file| file.strip_prefix(&root).unwrap().to_owned())
            .collect();
        assert_eq!(
            relative,
            [
                PathBuf::from("a/z.rs"),
                PathBuf::from("b.rs"),
                PathBuf::from("c/keep.rs")
            ]
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_unreadable_folder_names_itself_in_the_error() {
        let root = scratch_checkout("walk", "missing", &[]);
        let gone = root.join("gone");
        let error = rust_files(&gone, &|_| false).unwrap_err();
        assert_eq!(error.path, gone.display().to_string());
        assert!(error.to_string().starts_with(&error.path));
        assert!(read_text(&gone.join("x.rs")).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
