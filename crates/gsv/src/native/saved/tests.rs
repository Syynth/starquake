//! The native tables' snapshot: its bytes, every check's refusal, and a restored table in use.

use std::collections::BTreeMap;

use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::{ProcessId, SlabId};
use starquake_kernel::memory::testing::device_share_token;

use super::*;
use crate::completion::testing::{Locks, process};
use crate::native::layout::Preference;
use crate::native::tests::laid_out;

/// A one-page image: 16 pixels, 64 rows in 64-row blocks (64 * 64 = 4096 bytes).
fn image_a() -> Preference {
    laid_out(1, 16, 64, 6, 1, 1)
}

/// A 512-byte RGB10A2 image: 16 pixels, 8 rows in 8-row blocks.
fn image_b() -> Preference {
    laid_out(1, 16, 8, 3, 1, 4)
}

/// One of each kind of entry, every field distinct where its range allows: a memory bound to
/// process memory, one to a registration (retired, its image in a window pool), one bound with
/// the GPU-accessible check, one unbound; two images.
fn golden() -> State {
    let binding = |address, source| {
        Some(SavedBinding {
            address,
            process: process(),
            source,
        })
    };
    State {
        memories: vec![
            SavedMemory {
                id: 3,
                mode: 1,
                size: 0x2000,
                retired: false,
                binding: binding(0x4000, SavedSource::Process(false)),
            },
            SavedMemory {
                id: 5,
                mode: 2,
                size: 0x3000,
                retired: true,
                binding: binding(0x6000, SavedSource::Gpu(7, 0x1000)),
            },
            SavedMemory {
                id: 6,
                mode: 3,
                size: 0x1000,
                retired: false,
                binding: binding(0x5000, SavedSource::Process(true)),
            },
            SavedMemory {
                id: 9,
                mode: 1,
                size: 0x2000,
                retired: false,
                binding: None,
            },
        ],
        images: vec![
            SavedImage {
                id: 4,
                memory: 5,
                preference: image_a(),
                count: 1,
                size: 0x1000,
                window_pool: true,
            },
            SavedImage {
                id: 8,
                memory: 3,
                preference: image_b(),
                count: 3,
                size: 1536,
                window_pool: false,
            },
        ],
        next_memory: 10,
        next_image: 11,
        pools: vec![SavedPool {
            first: 9,
            memory: 6,
            preference: video_pool(),
            count: 1,
        }],
        mappings: vec![SavedMapping {
            space: 2,
            image: 9,
            references: 3,
        }],
    }
}

/// A one-image NV12 pool preference: code 1, 2x2 pixels, 4096 bytes (PR's smallest image).
fn video_pool() -> Preference {
    let mut words = [0; PREFERENCE_WORDS];
    words[1] = 1;
    words[2] = 1 | (1 << 16);
    words
}

/// The bytes of `state` with the header.
fn encode(state: &State) -> Vec<u8> {
    let mut bytes = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    bytes.extend(postcard::to_allocvec(state).unwrap());
    bytes
}

/// A kernel holding one lock on each of the golden bindings' ranges.
fn kernel() -> Locks {
    Locks {
        process: process(),
        live: BTreeMap::from([
            ((0x4000, 0x2000), 1),
            ((0x6000, 0x3000), 1),
            ((0x5000, 0x1000), 1),
        ]),
    }
}

/// What `state` restores to, or the error.
fn restore(state: &State) -> Result<NativeObjects, SnapshotError> {
    NativeObjects::restore(&encode(state), &kernel())
}

/// The error a restore of `state` returns, as its message (the errors hold a `postcard` error,
/// which has no equality).
fn refusal(state: &State) -> String {
    restore(state).unwrap_err().to_string()
}

#[test]
fn the_snapshot_layout_is_pinned() {
    // Varints, by hand: 0x1000 is 80 20, 0x2000 80 40, 0x3000 80 60, 0x4000 80 80 01, 0x5000
    // 80 A0 01, 0x6000 80 C0 01, 1536 80 0C. The process is `from_parts(2, 3)`.
    let (pi, pg) = (0x02, 0x03);
    let mut expected: Vec<u8> = b"SQNI".to_vec();
    expected.push(2); // version
    expected.push(4); // memories: four
    // 3: mode 1, 0x2000, live, bound at 0x4000 to process memory, not GPU-checked.
    expected.extend([3, 1, 0x80, 0x40, 0, 1, 0x80, 0x80, 0x01, pi, pg, 0, 0]);
    // 5: mode 2, 0x3000, retired, bound at 0x6000 to registration 7 at offset 0x1000.
    expected.extend([
        5, 2, 0x80, 0x60, 1, 1, 0x80, 0xC0, 0x01, pi, pg, 1, 7, 0x80, 0x20,
    ]);
    // 6: mode 3, 0x1000, live, bound at 0x5000 to process memory, GPU-checked.
    expected.extend([6, 3, 0x80, 0x20, 0, 1, 0x80, 0xA0, 0x01, pi, pg, 0, 1]);
    // 9: mode 1, 0x2000, live, unbound.
    expected.extend([9, 1, 0x80, 0x40, 0, 0]);
    expected.push(2); // images: two
    // 4: memory 5, the preference (word 2: 15 | 63 << 16 = 0x3F000F, varint 8F 80 FC 01),
    // count 1, 0x1000, in a window pool.
    expected.extend([4, 5, 0, 1, 0x8F, 0x80, 0xFC, 0x01, 0, 6, 1, 1, 1]);
    expected.extend([0; 8]);
    expected.extend([1, 0x80, 0x20, 1]);
    // 8: memory 3, the preference (word 2: 15 | 7 << 16 = 0x7000F, varint 8F 80 1C), count 3,
    // 1536, not in a pool.
    expected.extend([8, 3, 0, 1, 0x8F, 0x80, 0x1C, 0, 3, 1, 4, 1]);
    expected.extend([0; 8]);
    expected.extend([3, 0x80, 0x0C, 0]);
    expected.extend([10, 11]); // next ids
    expected.push(1); // pools: one
    // first image 9, memory 6, the preference (word 1: code 1; word 2: 1 | 1 << 16 = 0x10001,
    // varint 81 80 04), count 1.
    expected.extend([9, 6, 0, 1, 0x81, 0x80, 0x04]);
    expected.extend([0; 13]);
    expected.push(1);
    expected.push(1); // mappings: one
    expected.extend([2, 9, 3]); // address space 2, image 9, three references
    assert_eq!(encode(&golden()), expected);
    let restored = NativeObjects::restore(&expected, &kernel()).unwrap();
    assert_eq!(restored.save().unwrap(), expected);
}

/// The restored tables hold what was saved, field for field, with the tokens the kernel gave.
#[test]
fn a_restore_reads_every_field() {
    let restored = restore(&golden()).unwrap();
    let five = restored.memories().find(|(id, _)| *id == 5).unwrap().1;
    assert_eq!(
        (five.mode(), five.size(), five.retired()),
        (2, 0x3000, true)
    );
    assert_eq!(
        five.binding(),
        Some(&Binding {
            address: 0x6000,
            source: Source::Gpu {
                parent: 7,
                offset: 0x1000
            },
            token: device_share_token(process(), 0x6000, 0x3000),
        })
    );
    assert_eq!(
        restored.memory(6).unwrap().binding().unwrap().source,
        Source::Process {
            gpu_accessible: true
        }
    );
    assert_eq!(restored.memory(9).unwrap().binding(), None);
    let image = restored.image(8).unwrap();
    assert_eq!(
        (
            image.memory(),
            image.count(),
            image.size(),
            image.window_pool()
        ),
        (3, 3, 1536, false)
    );
    assert_eq!(image.preference(), &image_b());
    assert!(restored.image(4).unwrap().window_pool());
    assert_eq!((restored.next_memory, restored.next_image), (10, 11));
    let pool = restored.pool_image(9).unwrap();
    assert_eq!(
        (
            pool.first,
            pool.index,
            pool.pool.memory(),
            pool.pool.count()
        ),
        (9, 0, 6, 1)
    );
    assert_eq!(pool.pool.preference(), &video_pool());
    assert_eq!(pool.pool.layout().image_bytes, 4096);
    let mappings: Vec<_> = restored.gpu_mappings().collect();
    assert_eq!(mappings.len(), 1);
    let ((space, image), mapping) = mappings[0];
    assert_eq!((space, image), (crate::gpu::AsId(2), 9));
    // The image's address: memory 6's binding, 0x5000, plus offset 0.
    assert_eq!(
        (mapping.address(), mapping.size(), mapping.references()),
        (0x5000, 4096, 3)
    );
}

/// The header and the framing, each refused with its own error.
#[test]
fn the_header_and_framing_are_checked() {
    let good = encode(&golden());
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(matches!(
        NativeObjects::restore(&magic, &kernel()),
        Err(SnapshotError::BadMagic)
    ));
    // Version 1, the layout before the video pools, is no longer read.
    let mut version = good.clone();
    version[4] = 1;
    assert!(matches!(
        NativeObjects::restore(&version, &kernel()),
        Err(SnapshotError::UnsupportedVersion { found: 1 })
    ));
    assert!(matches!(
        NativeObjects::restore(&good[..good.len() - 1], &kernel()),
        Err(SnapshotError::Decode(_))
    ));
    let mut trailing = good;
    trailing.push(0);
    assert!(matches!(
        NativeObjects::restore(&trailing, &kernel()),
        Err(SnapshotError::TrailingBytes { count: 1 })
    ));
}

/// Each memory check, alone, with its exact error, and its accepted side at the edge.
#[test]
fn each_memory_check_refuses_on_its_own() {
    let out = |index| {
        SnapshotError::OutOfRange {
            part: Part::Memory,
            index,
        }
        .to_string()
    };
    let edit = |change: &dyn Fn(&mut State)| {
        let mut state = golden();
        change(&mut state);
        state
    };
    // Ids: from 1, below the next.
    assert_eq!(refusal(&edit(&|s| s.memories[0].id = 0)), out(0));
    assert_eq!(refusal(&edit(&|s| s.memories[3].id = 10)), out(3));
    // Ids increase: a repeat and a swap.
    let order = |index| {
        SnapshotError::Order {
            part: Part::Memory,
            index,
        }
        .to_string()
    };
    assert_eq!(refusal(&edit(&|s| s.memories[3].id = 6)), order(3));
    assert_eq!(refusal(&edit(&|s| s.memories.swap(2, 3))), order(3));
    // Mode, size.
    assert_eq!(refusal(&edit(&|s| s.memories[3].mode = 4)), out(3));
    assert_eq!(refusal(&edit(&|s| s.memories[3].size = 0)), out(3));
    assert_eq!(
        refusal(&edit(&|s| s.memories[3].size = 0x1_0000_0000_0001)),
        out(3)
    );
    assert!(restore(&edit(&|s| s.memories[3].size = 0x1_0000_0000_0000)).is_ok());
    // A binding's range: not 0, page aligned, not wrapping.
    let bind = |address| {
        move |s: &mut State| {
            s.memories[3].binding = Some(SavedBinding {
                address,
                process: process(),
                source: SavedSource::Process(false),
            });
        }
    };
    assert_eq!(refusal(&edit(&bind(0))), out(3));
    assert_eq!(refusal(&edit(&bind(0x7800))), out(3));
    assert_eq!(refusal(&edit(&bind(u64::MAX - 0xFFF))), out(3));
    // The last that does not wrap passes every check here, and the kernel holds no lock on it.
    assert_eq!(
        refusal(&edit(&bind(u64::MAX - 0x2FFF))),
        SnapshotError::Release {
            index: 3,
            error: DeviceShareError::NotLocked {
                address: u64::MAX - 0x2FFF,
                size: 0x2000
            }
        }
        .to_string()
    );
}

/// Ids up to `u32::MAX` once the ids have run out (next 0); up to the next one otherwise.
#[test]
fn ids_run_to_the_end_when_the_next_is_zero() {
    let mut state = golden();
    state.memories[3].id = u32::MAX;
    state.images[1].id = u32::MAX;
    assert!(restore(&state).is_err());
    state.next_memory = 0;
    state.next_image = 0;
    let restored = restore(&state).unwrap();
    assert!(restored.memory(u32::MAX).is_some());
    assert!(restored.image(u32::MAX).is_some());
    // An id exactly one below the next is accepted.
    let mut state = golden();
    state.memories[3].id = 9;
    state.next_memory = 10;
    assert!(restore(&state).is_ok());
}

/// Each image check, alone, with its exact error.
#[test]
fn each_image_check_refuses_on_its_own() {
    let edit = |change: &dyn Fn(&mut State)| {
        let mut state = golden();
        change(&mut state);
        refusal(&state)
    };
    let out = |index| {
        SnapshotError::OutOfRange {
            part: Part::Image,
            index,
        }
        .to_string()
    };
    assert_eq!(edit(&|s| s.images[0].id = 0), out(0));
    assert_eq!(edit(&|s| s.images[1].id = 11), out(1));
    assert_eq!(
        edit(&|s| s.images[1].id = 4),
        SnapshotError::Order {
            part: Part::Image,
            index: 1
        }
        .to_string()
    );
    let dangling = |index| SnapshotError::Dangling { index }.to_string();
    assert_eq!(edit(&|s| s.images[1].memory = 7), dangling(1));
    // Memory 9 is unbound.
    assert_eq!(edit(&|s| s.images[1].memory = 9), dangling(1));
    let size = |index| SnapshotError::Size { index }.to_string();
    assert_eq!(edit(&|s| s.images[1].size = 1537), size(1));
    assert_eq!(edit(&|s| s.images[1].count = 2), size(1));
    assert_eq!(edit(&|s| s.images[1].count = 0), size(1));
    assert_eq!(edit(&|s| s.images[1].preference[6] = 2), size(1));
    // Kind 2 sizes the same but 429 never makes it.
    assert_eq!(edit(&|s| s.images[1].preference[1] = 2), size(1));
    // An image larger than its memory: four pages in memory 3's two.
    assert_eq!(
        edit(&|s| {
            s.images[1].preference = laid_out(1, 16, 256, 8, 1, 1);
            s.images[1].count = 1;
            s.images[1].size = 0x4000;
        }),
        out(1)
    );
    // Exactly the memory's size is accepted.
    let mut fits = golden();
    fits.images[1].preference = laid_out(1, 16, 128, 7, 1, 1);
    fits.images[1].count = 1;
    fits.images[1].size = 0x2000;
    assert!(restore(&fits).is_ok());
}

/// A retired memory no image uses is refused.
#[test]
fn a_retired_memory_needs_an_image() {
    let mut state = golden();
    state.memories[3].retired = true;
    assert_eq!(
        refusal(&state),
        SnapshotError::RetiredUnused { index: 3 }.to_string()
    );
    // Its image gone, the retired golden memory is refused too.
    let mut state = golden();
    state.images.remove(0);
    assert_eq!(
        refusal(&state),
        SnapshotError::RetiredUnused { index: 1 }.to_string()
    );
}

/// The checks run memories first, then images, then the retired relation, then the kernel's:
/// with faults in two parts at once, the earlier part's is reported.
#[test]
fn the_checks_run_in_order() {
    let mut state = golden();
    state.memories[3].mode = 9;
    state.images[1].memory = 7;
    assert_eq!(
        refusal(&state),
        SnapshotError::OutOfRange {
            part: Part::Memory,
            index: 3
        }
        .to_string()
    );
    let mut state = golden();
    state.images[1].memory = 7;
    state.memories[3].retired = true;
    assert_eq!(
        refusal(&state),
        SnapshotError::Dangling { index: 1 }.to_string()
    );
    let mut state = golden();
    state.memories[3].retired = true;
    let none = Locks {
        process: process(),
        live: BTreeMap::new(),
    };
    assert_eq!(
        NativeObjects::restore(&encode(&state), &none)
            .unwrap_err()
            .to_string(),
        SnapshotError::RetiredUnused { index: 3 }.to_string()
    );
}

/// At most 1024 live memories (retired ones not counted) and 1024 images, each exactly at the
/// limit and one past.
#[test]
fn the_tables_are_bounded_at_the_limits() {
    let memory = |id| SavedMemory {
        id,
        mode: 0,
        size: 0x1000,
        retired: false,
        binding: None,
    };
    let limit = u32::try_from(MAX_MEMORIES).unwrap();
    let mut state = State {
        memories: (1..=limit).map(memory).collect(),
        images: Vec::new(),
        next_memory: limit + 2,
        next_image: 1,
        pools: Vec::new(),
        mappings: Vec::new(),
    };
    assert!(restore(&state).is_ok());
    state.memories.push(memory(limit + 1));
    assert_eq!(
        refusal(&state),
        SnapshotError::TooMany { part: Part::Memory }.to_string()
    );
    // One of them retired, with an image: 1024 live again.
    state.memories[0].retired = true;
    state.memories[0].binding = Some(SavedBinding {
        address: 0x4000,
        process: process(),
        source: SavedSource::Process(false),
    });
    let image = |id| SavedImage {
        id,
        memory: 1,
        preference: laid_out(1, 16, 64, 6, 1, 1),
        count: 1,
        size: 0x1000,
        window_pool: false,
    };
    state.images = (1..=limit).map(image).collect();
    state.next_image = limit + 2;
    let locks = Locks {
        process: process(),
        live: BTreeMap::from([((0x4000, 0x1000), 1)]),
    };
    assert!(NativeObjects::restore(&encode(&state), &locks).is_ok());
    state.images.push(image(limit + 1));
    assert_eq!(
        NativeObjects::restore(&encode(&state), &locks)
            .unwrap_err()
            .to_string(),
        SnapshotError::TooMany { part: Part::Image }.to_string()
    );
}

/// Locks are taken back in memory order, counting earlier ones of the same range: with two
/// locks on a range, two bindings of it are taken back and a third is refused at its own index.
#[test]
fn locks_beyond_the_kernel_s_are_refused_at_the_first_extra_one() {
    let bound = |id, address| SavedMemory {
        id,
        mode: 0,
        size: 0x1000,
        retired: false,
        binding: Some(SavedBinding {
            address,
            process: process(),
            source: SavedSource::Process(false),
        }),
    };
    let mut state = State {
        memories: vec![bound(1, 0x4000), bound(2, 0x8000), bound(3, 0x4000)],
        images: Vec::new(),
        next_memory: 5,
        next_image: 1,
        pools: Vec::new(),
        mappings: Vec::new(),
    };
    let locks = Locks {
        process: process(),
        live: BTreeMap::from([((0x4000, 0x1000), 2), ((0x8000, 0x1000), 1)]),
    };
    assert!(NativeObjects::restore(&encode(&state), &locks).is_ok());
    state.memories.push(bound(4, 0x4000));
    let refused = NativeObjects::restore(&encode(&state), &locks).unwrap_err();
    assert_eq!(
        refused.to_string(),
        SnapshotError::Release {
            index: 3,
            error: DeviceShareError::NotLocked {
                address: 0x4000,
                size: 0x1000
            }
        }
        .to_string()
    );
    // Another process's lock is refused as the kernel refuses it.
    let other = Locks {
        process: ProcessId::from_parts(5, 6),
        live: locks.live,
    };
    assert!(matches!(
        NativeObjects::restore(&encode(&state), &other),
        Err(SnapshotError::Release {
            index: 0,
            error: DeviceShareError::WrongProcess
        })
    ));
}

/// A kernel that holds locks in several processes: how many live locks it holds, by `(process,
/// address, size)`. Its contract is [`Locks`]'s for each process, the refusal being
/// [`DeviceShareError::NotLocked`] when it holds no more locks on the range in that process
/// than `earlier`; a process it holds nothing in is refused the same way, where [`Locks`] (one
/// process) answers [`DeviceShareError::WrongProcess`].
struct ProcessLocks(BTreeMap<(ProcessId, u64, u64), u32>);

impl Reclaim for ProcessLocks {
    fn reclaim(
        &self,
        process: ProcessId,
        address: u64,
        size: u64,
        earlier: u32,
    ) -> Result<starquake_kernel::memory::DeviceShareToken, DeviceShareError> {
        if self.0.get(&(process, address, size)).copied().unwrap_or(0) <= earlier {
            return Err(DeviceShareError::NotLocked { address, size });
        }
        Ok(device_share_token(process, address, size))
    }
}

/// Earlier bindings are counted per process: two bindings of the same exact range in two
/// processes, against a kernel that holds one lock on it in each, both take their lock back,
/// each with its own process; a third in the first process is refused at its own index.
#[test]
fn locks_are_counted_per_process() {
    let other = ProcessId::from_parts(5, 6);
    let bound = |id, process| SavedMemory {
        id,
        mode: 0,
        size: 0x1000,
        retired: false,
        binding: Some(SavedBinding {
            address: 0x4000,
            process,
            source: SavedSource::Process(false),
        }),
    };
    let mut state = State {
        memories: vec![bound(1, process()), bound(2, other)],
        images: Vec::new(),
        next_memory: 4,
        next_image: 1,
        pools: Vec::new(),
        mappings: Vec::new(),
    };
    let locks = ProcessLocks(BTreeMap::from([
        ((process(), 0x4000, 0x1000), 1),
        ((other, 0x4000, 0x1000), 1),
    ]));
    let restored = NativeObjects::restore(&encode(&state), &locks).unwrap();
    let token = |id| restored.memory(id).unwrap().binding().unwrap().token;
    assert_eq!(token(1), device_share_token(process(), 0x4000, 0x1000));
    assert_eq!(token(2), device_share_token(other, 0x4000, 0x1000));
    state.memories.push(bound(3, process()));
    assert_eq!(
        NativeObjects::restore(&encode(&state), &locks)
            .unwrap_err()
            .to_string(),
        SnapshotError::Release {
            index: 2,
            error: DeviceShareError::NotLocked {
                address: 0x4000,
                size: 0x1000
            }
        }
        .to_string()
    );
}

/// Saving runs the restore's checks: a table no command can make is not written.
#[test]
fn saving_checks_what_it_writes() {
    let mut objects = restore(&golden()).unwrap();
    assert!(objects.save().is_ok());
    if let Some(memory) = objects.memories.get_mut(&9) {
        memory.retired = true;
    }
    assert!(matches!(
        objects.save(),
        Err(SnapshotError::RetiredUnused { index: 3 })
    ));
}

/// Error messages are plain words, the zero case included.
#[test]
fn error_messages_are_plain_words() {
    assert_eq!(
        SnapshotError::TrailingBytes { count: 0 }.to_string(),
        "0 unexpected bytes after the native tables snapshot"
    );
    assert_eq!(
        SnapshotError::OutOfRange {
            part: Part::Image,
            index: 0
        }
        .to_string(),
        "native image 0 of the native tables snapshot is out of range"
    );
    assert_eq!(
        SnapshotError::TooMany { part: Part::Memory }.to_string(),
        "the native tables snapshot holds more native memory entries than the limit"
    );
}

/// The golden state with `pools` and no mappings.
fn with_pools(pools: Vec<SavedPool>) -> State {
    State {
        pools,
        mappings: Vec::new(),
        ..golden()
    }
}

/// A one-image pool of [`video_pool`] at `first` in memory `memory`.
fn pool_at(first: u32, memory: u32, count: u32) -> SavedPool {
    SavedPool {
        first,
        memory,
        preference: video_pool(),
        count,
    }
}

#[test]
fn each_pool_check_refuses_on_its_own() {
    let pool = Part::Pool;
    let out_of_range = |index| SnapshotError::OutOfRange { part: pool, index }.to_string();
    // The first id 0.
    assert_eq!(
        refusal(&with_pools(vec![pool_at(0, 6, 1)])),
        out_of_range(0)
    );
    // The end exactly at the next image id is taken; one past is refused.
    assert!(restore(&with_pools(vec![pool_at(10, 6, 1)])).is_ok());
    assert_eq!(
        refusal(&with_pools(vec![pool_at(10, 6, 2)])),
        out_of_range(0)
    );
    // With the ids run out (0), the end may be 2^32 exactly.
    let mut run_out = with_pools(vec![pool_at(u32::MAX, 6, 1)]);
    run_out.next_image = 0;
    assert!(restore(&run_out).is_ok());
    // A pool that starts before the end of the one before it; one that starts exactly at it is
    // taken (memory 5, 0x3000 bytes, holds three images).
    let overlapping = vec![pool_at(1, 5, 2), pool_at(2, 5, 1)];
    assert_eq!(
        refusal(&with_pools(overlapping)),
        SnapshotError::Order {
            part: pool,
            index: 1
        }
        .to_string()
    );
    assert!(restore(&with_pools(vec![pool_at(1, 5, 2), pool_at(3, 5, 1)])).is_ok());
    // A pool holding a native image's id (8), at its first id or inside it; one that ends at 8
    // is taken.
    for overlapping in [pool_at(8, 5, 1), pool_at(7, 5, 2)] {
        assert_eq!(
            refusal(&with_pools(vec![overlapping])),
            SnapshotError::PoolOverlap { index: 0 }.to_string()
        );
    }
    assert!(restore(&with_pools(vec![pool_at(5, 5, 3)])).is_ok());
    // A memory missing, or not bound (9).
    for memory in [42, 9] {
        assert_eq!(
            refusal(&with_pools(vec![pool_at(9, memory, 1)])),
            SnapshotError::PoolDangling { index: 0 }.to_string()
        );
    }
    // A preference 911 refuses, and a count over the limit.
    let mut bad = pool_at(9, 6, 1);
    bad.preference[1] = 7;
    assert_eq!(
        refusal(&with_pools(vec![bad])),
        SnapshotError::PoolLayout { index: 0 }.to_string()
    );
    let mut many = with_pools(vec![pool_at(0x100, 6, MAX_POOL_IMAGES + 1)]);
    many.next_image = 0;
    assert_eq!(
        refusal(&many),
        SnapshotError::PoolLayout { index: 0 }.to_string()
    );
    // A pool larger than its memory: two 4096-byte images in memory 6's 0x1000.
    assert_eq!(
        refusal(&with_pools(vec![pool_at(9, 6, 2)])),
        out_of_range(0)
    );
}

#[test]
fn pools_are_bounded_at_the_limit() {
    let limit = u32::try_from(MAX_POOLS).unwrap();
    let mut state = with_pools((0..limit).map(|n| pool_at(0x100 + n, 6, 1)).collect());
    state.next_image = 0;
    assert!(restore(&state).is_ok());
    state.pools.push(pool_at(0x100 + limit, 6, 1));
    assert_eq!(
        refusal(&state),
        SnapshotError::TooMany { part: Part::Pool }.to_string()
    );
}

#[test]
fn each_mapping_check_refuses_on_its_own() {
    let part = Part::Mapping;
    let mapping = |space, image, references| SavedMapping {
        space,
        image,
        references,
    };
    let with = |mappings: Vec<SavedMapping>| State {
        mappings,
        ..golden()
    };
    // Two of one key, and two out of order.
    for mappings in [
        vec![mapping(2, 9, 1), mapping(2, 9, 1)],
        vec![mapping(3, 9, 1), mapping(2, 9, 1)],
    ] {
        assert_eq!(
            refusal(&with(mappings)),
            SnapshotError::Order { part, index: 1 }.to_string()
        );
    }
    assert!(restore(&with(vec![mapping(2, 9, 1), mapping(3, 9, 1)])).is_ok());
    // An image no pool holds: the one past the pool, and a native image.
    for image in [10, 8] {
        assert_eq!(
            refusal(&with(vec![mapping(2, image, 1)])),
            SnapshotError::MappingDangling { index: 0 }.to_string()
        );
    }
    // No references.
    assert_eq!(
        refusal(&with(vec![mapping(2, 9, 0)])),
        SnapshotError::OutOfRange { part, index: 0 }.to_string()
    );
    // An image whose offset passes 32 bits: code 2 at 65536 x 16256 is 4_261_412_864 bytes an
    // image, so image 2 of a pool starts past 2^32; image 1 starts under it.
    let mut wide = golden();
    let image = 4_261_412_864_u64;
    wide.memories[3].size = 3 * image;
    wide.memories[3].binding = Some(SavedBinding {
        address: 0x1_0000_0000,
        process: process(),
        source: SavedSource::Process(false),
    });
    let mut preference = [0; PREFERENCE_WORDS];
    preference[1] = 2;
    preference[2] = 0xFFFF | (16_255 << 16);
    wide.next_image = 0;
    wide.pools = vec![SavedPool {
        first: 0x20,
        memory: 9,
        preference,
        count: 3,
    }];
    wide.mappings = vec![mapping(1, 0x21, 1)];
    assert!(State::check(&wide).is_ok());
    wide.mappings = vec![mapping(1, 0x22, 1)];
    assert_eq!(
        State::check(&wide).unwrap_err().to_string(),
        SnapshotError::OutOfRange { part, index: 0 }.to_string()
    );
}

#[test]
fn mappings_are_bounded_at_the_limit() {
    let limit = u32::try_from(MAX_MAPPINGS).unwrap();
    let mut state = golden();
    state.mappings = (0..limit)
        .map(|space| SavedMapping {
            space,
            image: 9,
            references: 1,
        })
        .collect();
    assert!(restore(&state).is_ok());
    state.mappings.push(SavedMapping {
        space: limit,
        image: 9,
        references: 1,
    });
    assert_eq!(
        refusal(&state),
        SnapshotError::TooMany {
            part: Part::Mapping
        }
        .to_string()
    );
}

#[test]
fn a_retired_memory_a_pool_uses_is_kept() {
    // Memory 5 (retired) loses its image to memory 3 and holds the pool alone.
    let mut state = golden();
    state.images[0].memory = 3;
    state.pools = vec![pool_at(9, 5, 1)];
    assert!(restore(&state).is_ok());
    // With the pool (and its mapping) gone too, it is refused.
    state.pools.clear();
    state.mappings.clear();
    assert_eq!(
        refusal(&state),
        SnapshotError::RetiredUnused { index: 1 }.to_string()
    );
}

/// The pools are checked after the images and before the mappings, and the mappings before the
/// retired relation: with faults in two parts at once, the earlier part's is reported.
#[test]
fn pools_and_mappings_are_checked_in_order() {
    let mut state = with_pools(vec![pool_at(0, 6, 1)]);
    state.images[0].size = 1;
    assert_eq!(
        refusal(&state),
        SnapshotError::Size { index: 0 }.to_string()
    );
    let mut state = with_pools(vec![pool_at(0, 6, 1)]);
    state.mappings = vec![SavedMapping {
        space: 1,
        image: 77,
        references: 1,
    }];
    assert_eq!(
        refusal(&state),
        SnapshotError::OutOfRange {
            part: Part::Pool,
            index: 0
        }
        .to_string()
    );
    let mut state = golden();
    state.mappings[0].references = 0;
    state.images.remove(0);
    assert_eq!(
        refusal(&state),
        SnapshotError::OutOfRange {
            part: Part::Mapping,
            index: 0
        }
        .to_string()
    );
}

/// The save runs the same checks: a pool whose memory the live tables do not hold.
#[test]
fn saving_checks_the_pools() {
    let mut objects = restore(&golden()).unwrap();
    let pool = *objects.pools.get(&9).unwrap();
    objects.pools.insert(9, VideoPool { memory: 42, ..pool });
    assert_eq!(
        objects.save().unwrap_err().to_string(),
        SnapshotError::PoolDangling { index: 0 }.to_string()
    );
}
