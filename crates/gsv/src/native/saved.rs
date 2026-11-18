//! The native tables' snapshot ([`NativeObjects::save`], [`NativeObjects::restore`];
//! graphics-server, "What is snapshotted": "native memories and images").
//!
//! # Format
//!
//! `postcard`, after a header of [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is decoded and
//! checked before the body. The body ([`State`]) is, in order: the memories (id, mode byte,
//! size, whether it is retired, and its binding if it has one: address, the process of its lock,
//! then the source: 0 for process memory with the GPU-accessible flag byte, 1 for a gpu worker
//! registration with its handle and offset), the images (id, memory id, the sixteen preference
//! words, count, size, whether it is in a window pool), then the next memory id and the next
//! image id. The golden-bytes test `the_snapshot_layout_is_pinned` pins it; changing it means
//! bumping [`SNAPSHOT_VERSION`].
//!
//! # Checks
//!
//! A snapshot is a file and is not trusted. [`NativeObjects::restore`] checks, in this order,
//! and returns the first failure: the magic, the version, that the body decodes, that no bytes
//! follow it; then each memory in list order: its id (from [`FIRST_ID`], below the next memory
//! id unless the ids have run out (0), increasing: [`SnapshotError::OutOfRange`],
//! [`SnapshotError::Order`]), its mode (at most [`MAX_MODE`]), its size (1 to
//! [`MAX_MEMORY_SIZE`]), and its binding's range (an address that is not 0, page aligned, and
//! does not wrap with the size), each [`SnapshotError::OutOfRange`]; the live memories (at most
//! [`MAX_MEMORIES`], [`SnapshotError::TooMany`]); then each image in list order: its id (as a
//! memory's, against the next image id), its memory (a memory of the snapshot that is bound,
//! [`SnapshotError::Dangling`]), its size ([`super::layout::requirements`] of its preference
//! and count, with the preference's kind [`super::layout::KIND_NATIVE`],
//! [`SnapshotError::Size`]; at most its memory's size, [`SnapshotError::OutOfRange`]); the
//! images (at most [`MAX_IMAGES`], [`SnapshotError::TooMany`]); that every retired memory has
//! an image ([`SnapshotError::RetiredUnused`]). Last, every binding's device-share lock is taken
//! back from the kernel in memory order ([`Reclaim`]), counting earlier ones of the same exact
//! range, so that N locks on a range take back N tokens and the next is refused at its own index
//! ([`SnapshotError::Release`]). [`NativeObjects::save`] runs the same checks, but the kernel's,
//! on what it is about to write.
//!
//! These are every relation the commands keep: 111 hands out increasing ids below the next one,
//! with a mode and size it checked; 114 and 423 bind a page-aligned range of the memory's size;
//! 429 needs a bound memory at least the image's size and a preference
//! [`super::layout::requirements`] accepts with kind 1; 112 retires only a memory an image uses,
//! and 430 removes a retired memory with its last image.
//!
//! **Not checked here:**
//!
//! - **That a 423 binding's parent is a registration of the GPU memory.** A 102 of the parent
//!   goes ahead while the binding stays ([`super`], "Where this differs from the references"),
//!   so a live table can hold a parent that no longer exists. Its range is checked as 114's is,
//!   and its lock is taken back like any other.
//! - **That the GPU memory's registrations and these bindings together take back no more locks
//!   than the kernel holds.** Each snapshot counts its own; a 423 binding whose range is the
//!   whole of its parent's has the same exact range as the parent's lock. Composing the two
//!   restores so that they count together is the wiring slice's (16), as for the completion
//!   queue's releases (`crate::gpu::saved`, "Checks").

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::ProcessId;
use starquake_memory::PAGE_SIZE;
use thiserror::Error;

use super::layout::{KIND_NATIVE, PREFERENCE_WORDS, WORD_KIND, requirements};
use super::{
    Binding, FIRST_ID, MAX_IMAGES, MAX_MEMORIES, MAX_MEMORY_SIZE, MAX_MODE, NativeImage,
    NativeMemory, NativeObjects, Source,
};
use crate::completion::Reclaim;

#[cfg(test)]
mod tests;

/// First bytes of every native tables snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQNI";

/// Layout version written into every native tables snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The part of a snapshot a check names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// A native memory.
    Memory,
    /// A native image.
    Image,
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Memory => "native memory",
            Self::Image => "native image",
        })
    }
}

/// Why a native tables snapshot could not be written or read back (module docs, "Checks").
#[derive(Debug, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the native tables snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the native tables snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a native tables snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "native tables snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the native tables snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// An id, mode, size or range is outside what the commands make.
    #[error("{part} {index} of the native tables snapshot is out of range")]
    OutOfRange {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// An entry's id is not above the one before it in its list.
    #[error("{part} {index} of the native tables snapshot is out of order or repeats one before it")]
    Order {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// An image names a memory the snapshot does not hold, or one that is not bound.
    #[error("native image {index} of the snapshot names a memory that is missing or not bound")]
    Dangling {
        /// The image's position in its list.
        index: usize,
    },
    /// An image's size is not what its preference and count make, or its preference is not one
    /// 429 accepts.
    #[error("native image {index} of the snapshot has a size its preference does not make")]
    Size {
        /// The image's position in its list.
        index: usize,
    },
    /// A list is longer than its bound.
    #[error("the native tables snapshot holds more {part} entries than the limit")]
    TooMany {
        /// What it holds too many of.
        part: Part,
    },
    /// A retired memory has no image: 430 removes it with its last one.
    #[error("native memory {index} of the snapshot is retired but no image uses it")]
    RetiredUnused {
        /// The memory's position in its list.
        index: usize,
    },
    /// The kernel refused to take back a binding's lock.
    #[error("the kernel refused native memory {index}'s device-share lock: {error}")]
    Release {
        /// The memory's position in its list.
        index: usize,
        /// The kernel's refusal.
        error: DeviceShareError,
    },
}

/// The header before the body.
#[derive(Serialize, Deserialize)]
struct Header {
    magic: [u8; 4],
    version: u32,
}

/// A binding's source, as saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum SavedSource {
    /// 114, with whether it checked the GPU-accessible attribute.
    Process(bool),
    /// 423: the registration's handle and the offset.
    Gpu(u32, u64),
}

/// A binding, as saved: address, the process of its lock, source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedBinding {
    address: u64,
    process: ProcessId,
    source: SavedSource,
}

/// A memory, as saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedMemory {
    id: u32,
    mode: u8,
    size: u64,
    retired: bool,
    binding: Option<SavedBinding>,
}

/// An image, as saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedImage {
    id: u32,
    memory: u32,
    preference: [u32; PREFERENCE_WORDS],
    count: u32,
    size: u64,
    window_pool: bool,
}

/// The body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    memories: Vec<SavedMemory>,
    images: Vec<SavedImage>,
    next_memory: u32,
    next_image: u32,
}

/// Whether `id` is one a table whose next id is `next` can have handed out: from
/// [`FIRST_ID`], and below `next` unless the ids have run out (0).
fn issued(id: u32, next: u32) -> bool {
    id >= FIRST_ID && (next == 0 || id < next)
}

/// Whether a binding's range is one 114 and 423 accept: an address that is not 0, page aligned,
/// that does not wrap with `size`.
fn bindable(address: u64, size: u64) -> bool {
    address != 0 && address.is_multiple_of(PAGE_SIZE) && address.checked_add(size).is_some()
}

impl State {
    /// The tables as saved.
    fn of(objects: &NativeObjects) -> Self {
        let memories = objects
            .memories
            .iter()
            .map(|(&id, memory)| SavedMemory {
                id,
                mode: memory.mode,
                size: memory.size,
                retired: memory.retired,
                binding: memory.binding.map(|binding| SavedBinding {
                    address: binding.address,
                    process: binding.token.process(),
                    source: match binding.source {
                        Source::Process { gpu_accessible } => SavedSource::Process(gpu_accessible),
                        Source::Gpu { parent, offset } => SavedSource::Gpu(parent, offset),
                    },
                }),
            })
            .collect();
        let images = objects
            .images
            .iter()
            .map(|(&id, image)| SavedImage {
                id,
                memory: image.memory,
                preference: image.preference,
                count: image.count,
                size: image.size,
                window_pool: image.window_pool,
            })
            .collect();
        Self {
            memories,
            images,
            next_memory: objects.next_memory,
            next_image: objects.next_image,
        }
    }

    /// Every check but the kernel's, in the module docs' order ("Checks").
    fn check(&self) -> Result<(), SnapshotError> {
        let memory_part = Part::Memory;
        let mut previous = None;
        for (index, memory) in self.memories.iter().enumerate() {
            let out_of_range = SnapshotError::OutOfRange {
                part: memory_part,
                index,
            };
            if !issued(memory.id, self.next_memory) {
                return Err(out_of_range);
            }
            if previous.is_some_and(|before| memory.id <= before) {
                return Err(SnapshotError::Order {
                    part: memory_part,
                    index,
                });
            }
            previous = Some(memory.id);
            if memory.mode > MAX_MODE || memory.size == 0 || memory.size > MAX_MEMORY_SIZE {
                return Err(out_of_range);
            }
            if let Some(binding) = memory.binding
                && !bindable(binding.address, memory.size)
            {
                return Err(out_of_range);
            }
        }
        let live = self.memories.iter().filter(|memory| !memory.retired).count();
        if live > MAX_MEMORIES {
            return Err(SnapshotError::TooMany { part: memory_part });
        }
        let memories: BTreeMap<u32, &SavedMemory> = self
            .memories
            .iter()
            .map(|memory| (memory.id, memory))
            .collect();
        let image_part = Part::Image;
        let mut previous = None;
        for (index, image) in self.images.iter().enumerate() {
            if !issued(image.id, self.next_image) {
                return Err(SnapshotError::OutOfRange {
                    part: image_part,
                    index,
                });
            }
            if previous.is_some_and(|before| image.id <= before) {
                return Err(SnapshotError::Order {
                    part: image_part,
                    index,
                });
            }
            previous = Some(image.id);
            let backing = memories
                .get(&image.memory)
                .filter(|memory| memory.binding.is_some())
                .ok_or(SnapshotError::Dangling { index })?;
            let made = requirements(&image.preference, image.count)
                .ok()
                .filter(|_| image.preference[WORD_KIND] == KIND_NATIVE)
                .map(|made| made.size);
            if made != Some(image.size) {
                return Err(SnapshotError::Size { index });
            }
            if image.size > backing.size {
                return Err(SnapshotError::OutOfRange {
                    part: image_part,
                    index,
                });
            }
        }
        if self.images.len() > MAX_IMAGES {
            return Err(SnapshotError::TooMany { part: image_part });
        }
        let used: BTreeSet<u32> = self.images.iter().map(|image| image.memory).collect();
        for (index, memory) in self.memories.iter().enumerate() {
            if memory.retired && !used.contains(&memory.id) {
                return Err(SnapshotError::RetiredUnused { index });
            }
        }
        Ok(())
    }

    /// The tables, every binding's lock taken back from `kernel` in memory order, counting
    /// earlier ones of the same exact range (module docs, "Checks"). Run after
    /// [`State::check`].
    fn reclaim(&self, kernel: &dyn Reclaim) -> Result<NativeObjects, SnapshotError> {
        let mut counts: BTreeMap<(ProcessId, u64, u64), u32> = BTreeMap::new();
        let mut memories = BTreeMap::new();
        for (index, saved) in self.memories.iter().enumerate() {
            let binding = match saved.binding {
                None => None,
                Some(binding) => {
                    let range = (binding.process, binding.address, saved.size);
                    let earlier = counts.entry(range).or_insert(0);
                    let token = kernel
                        .reclaim(range.0, range.1, range.2, *earlier)
                        .map_err(|error| SnapshotError::Release { index, error })?;
                    *earlier = earlier.saturating_add(1);
                    let source = match binding.source {
                        SavedSource::Process(gpu_accessible) => Source::Process { gpu_accessible },
                        SavedSource::Gpu(parent, offset) => Source::Gpu { parent, offset },
                    };
                    Some(Binding {
                        address: binding.address,
                        source,
                        token,
                    })
                }
            };
            memories.insert(
                saved.id,
                NativeMemory {
                    mode: saved.mode,
                    size: saved.size,
                    binding,
                    retired: saved.retired,
                },
            );
        }
        let images = self
            .images
            .iter()
            .map(|saved| {
                (
                    saved.id,
                    NativeImage {
                        memory: saved.memory,
                        preference: saved.preference,
                        count: saved.count,
                        size: saved.size,
                        window_pool: saved.window_pool,
                    },
                )
            })
            .collect();
        Ok(NativeObjects {
            memories,
            images,
            next_memory: self.next_memory,
            next_image: self.next_image,
        })
    }
}

impl NativeObjects {
    /// The tables as bytes (module docs, "Format"), after the restore's checks but the
    /// kernel's.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = State::of(self);
        state.check()?;
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&state).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// The tables [`save`](Self::save) wrote, checked first, their locks taken back from
    /// `kernel` (module docs, "Checks").
    pub fn restore(bytes: &[u8], kernel: &dyn Reclaim) -> Result<Self, SnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(SnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (state, rest) =
            postcard::take_from_bytes::<State>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        state.check()?;
        state.reclaim(kernel)
    }
}
