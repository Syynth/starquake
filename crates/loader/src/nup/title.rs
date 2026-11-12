//! The title a container holds: its base, its newest update, and the `ExeFS` they make.
//!
//! [`Nup::title`] picks the program packages and the `ExeFS` to run:
//!
//! - **The base** is the container's one `base` package, if it has one (`Nup::open` refuses a
//!   second, SPEC §3.1.2).
//! - **The newest update** is the `update` package with the highest `version`, compared as an
//!   unsigned integer. Two updates sharing the highest version are
//!   [`NupError::AmbiguousUpdate`] (SPEC §3.1.2 says a container's updates have distinct
//!   versions). The spec tells readers to treat `version` as opaque (§3.1.1, §6.1 rule 3), so
//!   **this ordering is a guess**: it assumes content versions grow with each release, as the
//!   spec's own packing `(major << 16) | (minor << 8) | micro` does. A title whose versions do
//!   not would confirm or refute it. Every update is taken to belong to the base: the title id
//!   relation (Eden's `GetUpdateTitleID` ORs in 0x800, `eden/src/core/file_sys/registered_cache.h`
//!   @ cf654052) is not checked, because no source here says it holds for Switch 2 ids.
//! - **The `ExeFS`** is the newest update's `ExeFS` directory as a whole, when that update has
//!   one, and the base's otherwise. This follows Eden, whose `PatchManager::PatchExeFS`
//!   (`eden/src/core/file_sys/patch_manager.cpp` @ cf654052) replaces the base's `ExeFS` with the
//!   update's (`exefs = update->GetExeFS();`) instead of merging them, and considers only the one
//!   update it selected. So "the title's `main`" is the newest `main`, `main.npdm`, `sdk`, `rtld`
//!   and `subsdk*`, all from the same package.
//!
//! # Divergence: no path shadowing in `ExeFS`
//!
//! The spec's informational appendix (`nup/SPEC.md`, Appendix A, "Cross-cutting") describes
//! update layering as the consumer overlaying the update's tree onto the base's by path
//! shadowing, under which a base `ExeFS` file that the update lacks would survive. For the
//! `ExeFS` this reader follows Eden instead, so an `ExeFS` file that only the base has is not
//! part of the title once an update with an `ExeFS` is present. Pinned by
//! `an_update_replaces_the_whole_exefs`. `RomFS` layering is not done here (out of scope for
//! issue #95).

use super::{DirId, FileId, Node, Nup, NupError, PackageKind};

/// Where the title's `ExeFS` comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExeFs {
    /// The package whose `ExeFS` it is.
    pub package: usize,
    /// Its directory.
    pub dir: DirId,
}

/// The program a container holds. See the module docs for how it is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Title {
    /// The base package, if the container has one.
    pub base: Option<usize>,
    /// The newest update, if the container has any.
    pub update: Option<usize>,
    /// The package that names the program: the base, or the newest update without one.
    pub program: usize,
    /// The title's `ExeFS`, if the chosen packages have one.
    pub exefs: Option<ExeFs>,
}

impl Title {
    /// The `ExeFS` file called `name` (`main`, `main.npdm`, `rtld`, `sdk`, `subsdk0`, ...).
    pub fn exefs_file<S>(&self, nup: &Nup<S>, name: &str) -> Option<FileId> {
        match nup.lookup(self.exefs?.dir, name.as_bytes())? {
            Node::File(file) => Some(file),
            Node::Dir(_) => None,
        }
    }

    /// The `ExeFS` files in name order (subdirectories, which an `ExeFS` does not have, are
    /// left out).
    pub fn exefs_files<S>(&self, nup: &Nup<S>) -> Vec<FileId> {
        let Some(exefs) = self.exefs else {
            return Vec::new();
        };
        nup.children(exefs.dir)
            .iter()
            .filter_map(|&node| match node {
                Node::File(file) => Some(file),
                Node::Dir(_) => None,
            })
            .collect()
    }
}

impl<S> Nup<S> {
    /// The title this container holds: see the module docs.
    pub fn title(&self) -> Result<Title, NupError> {
        let packages = self.packages();
        let base = packages.iter().find(|p| p.kind == PackageKind::Base);
        let updates = || packages.iter().filter(|p| p.kind == PackageKind::Update);
        let newest_version = updates().map(|p| p.version).max();
        let mut newest = updates().filter(|p| Some(p.version) == newest_version);
        let update = newest.next();
        if let (Some(first), Some(second)) = (update, newest.next()) {
            return Err(NupError::AmbiguousUpdate {
                first: first.index,
                second: second.index,
                version: first.version,
            });
        }
        let program = update
            .filter(|_| base.is_none())
            .or(base)
            .ok_or(NupError::NoProgram)?;
        let exefs = update
            .and_then(|u| u.exefs.map(|dir| (u.index, dir)))
            .or_else(|| base.and_then(|b| b.exefs.map(|dir| (b.index, dir))))
            .map(|(package, dir)| ExeFs { package, dir });
        Ok(Title {
            base: base.map(|b| b.index),
            update: update.map(|u| u.index),
            program: program.index,
            exefs,
        })
    }
}
