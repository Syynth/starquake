//! The recording and replay seams, one call at a time, and the replay's refusals of what it
//! cannot feed back.

use std::sync::Arc;
use std::time::Duration;

use starquake_memory::{Fault, FlatMemory};
use starquake_shader::exchange::TextureKind;
use starquake_shader::lookup::{
    EngineState, GuestStage, ProgramMemory, ProgramRequest, Ticket, Unmapped, VariantKey,
};

use super::*;
use crate::back::draw::shaders::{SetLookup, ShaderLookup};
use crate::back::fence::{FenceClock, Waiter, WaiterTable};
use crate::capture::replay::seams;
use crate::capture::{
    Access, BackRecord, Blobs, Capture, ContentHash, Divergence, EngineStateKind, FrontRecord,
    RecordError, Recorder, RecordingMemory, RecordingShaders, ReplayError, ReplayOptions,
    ShaderAnswer, StateRecord, Thread, replay,
};
use crate::front::FrontClock;
use crate::memory::{GpuFault, GpuMemory, TestGpuMemory};

/// Guest memory of four pages at `CPU_BASE`, mapped at `GPU_BASE`, holding 1, 2, 3, ... from its
/// first byte.
fn memory() -> Arc<dyn GpuMemory> {
    let guest = FlatMemory::new(CPU_BASE, 0x4000);
    let ramp: Vec<u8> = (1..=255).cycle().take(0x4000).collect();
    starquake_memory::GuestMemory::write(&guest, CPU_BASE, &ramp).unwrap();
    let memory = TestGpuMemory::new(guest);
    memory
        .map(PageRange::new(GPU_BASE, 0x4000).unwrap(), CPU_BASE)
        .unwrap();
    Arc::new(memory)
}

#[test]
fn recording_memory_records_each_call_with_its_answer_in_its_threads_log() {
    let recorder = Arc::new(Recorder::new(1 << 20));
    let front = RecordingMemory::new(memory(), Arc::clone(&recorder), Thread::Front, SPACE);
    let back = RecordingMemory::new(memory(), Arc::clone(&recorder), Thread::Back, AsId(0x22));
    let mut buf = [0; 3];
    front.read(GPU_BASE + 1, &mut buf).unwrap();
    assert_eq!(buf, [2, 3, 4]);
    assert_eq!(
        front.read(GPU_BASE + 0x3fff, &mut buf),
        Err(GpuFault::Unmapped {
            va: GPU_BASE + 0x4000
        })
    );
    back.write(GPU_BASE + 8, &[9, 8]).unwrap();
    assert_eq!(back.generation(), 1);
    assert!(!back.is_gpu_only(GPU_BASE, 16));
    let taken = recorder.take().unwrap();
    assert_eq!(
        taken.front,
        [
            FrontRecord::Access(Access::Read {
                space: SPACE,
                va: GPU_BASE + 1,
                len: 3,
                result: Ok(ContentHash::of(&[2, 3, 4])),
            }),
            FrontRecord::Access(Access::Read {
                space: SPACE,
                va: GPU_BASE + 0x3fff,
                len: 3,
                result: Err(GpuFault::Unmapped {
                    va: GPU_BASE + 0x4000
                }),
            }),
        ]
    );
    assert_eq!(
        taken.back,
        [
            BackRecord::Access(Access::Write {
                space: AsId(0x22),
                va: GPU_BASE + 8,
                data: ContentHash::of(&[9, 8]),
                result: Ok(()),
            }),
            BackRecord::Access(Access::Generation {
                space: AsId(0x22),
                value: 1,
            }),
            BackRecord::Access(Access::GpuOnly {
                space: AsId(0x22),
                va: GPU_BASE,
                len: 16,
                answer: false,
            }),
        ]
    );
    assert_eq!(
        taken.blobs.get(ContentHash::of(&[2, 3, 4])),
        Some(&[2, 3, 4][..])
    );
    assert_eq!(taken.blobs.get(ContentHash::of(&[9, 8])), Some(&[9, 8][..]));
    assert_eq!(taken.blobs.len(), 2);
}

#[test]
fn the_recorder_keeps_blobs_up_to_its_limit_and_counts_a_repeated_blob_once() {
    let recorder = Recorder::new(8);
    recorder.blob(&[1; 5]);
    recorder.blob(&[1; 5]);
    recorder.blob(&[2; 3]);
    recorder.front(FrontRecord::Turn);
    let taken = recorder.take().unwrap();
    assert_eq!((taken.blobs.len(), taken.blobs.bytes()), (2, 8));
    assert_eq!(taken.front, [FrontRecord::Turn]);

    let recorder = Recorder::new(8);
    recorder.blob(&[1; 5]);
    recorder.blob(&[2; 4]);
    // Nothing is kept after the blob that passed the limit.
    recorder.front(FrontRecord::Turn);
    let error = recorder.take().unwrap_err();
    assert_eq!(error, RecordError::Limit { limit: 8 });
    assert_eq!(
        error.to_string(),
        "the capture passed its limit of 8 bytes of blobs"
    );
}

/// A blob table holding `blobs`.
fn blobs(blobs: &[&[u8]]) -> Blobs {
    let mut table = Blobs::new();
    for blob in blobs {
        table.insert(blob);
    }
    table
}

fn read(space: AsId, va: u64, data: &[u8]) -> FrontRecord {
    FrontRecord::Access(Access::Read {
        space,
        va,
        len: u64::try_from(data.len()).unwrap(),
        result: Ok(ContentHash::of(data)),
    })
}

#[test]
fn replay_memory_answers_reads_writes_and_queries_in_order() {
    let fault = GpuFault::Guest {
        va: 0x2000,
        fault: Fault { address: 0x3000 },
    };
    let log = seams::Front::new(
        vec![
            read(SPACE, 0x1000, &[5, 6]),
            FrontRecord::Access(Access::Read {
                space: SPACE,
                va: 0x2000,
                len: 2,
                result: Err(fault),
            }),
            FrontRecord::Access(Access::Write {
                space: SPACE,
                va: 0x4000,
                data: ContentHash::of(&[7]),
                result: Err(GpuFault::ShimWrite { va: 0x4000 }),
            }),
            FrontRecord::Access(Access::Generation {
                space: SPACE,
                value: 0x55,
            }),
            FrontRecord::Access(Access::GpuOnly {
                space: SPACE,
                va: 0x5000,
                len: 0x10,
                answer: true,
            }),
        ],
        blobs(&[&[5, 6], &[7]]),
    );
    let memory = log.memory(SPACE);
    let mut buf = [0; 2];
    memory.read(0x1000, &mut buf).unwrap();
    assert_eq!(buf, [5, 6]);
    assert_eq!(memory.read(0x2000, &mut buf), Err(fault));
    assert_eq!(buf, [0, 0], "a faulted read gives zeros");
    assert_eq!(
        memory.write(0x4000, &[7]),
        Err(GpuFault::ShimWrite { va: 0x4000 })
    );
    assert_eq!(memory.generation(), 0x55);
    assert!(memory.is_gpu_only(0x5000, 0x10));
    log.check().unwrap();
    assert_eq!(log.next(), 5);
    assert_eq!(log.writes().len(), 1);
}

/// What diverges when the GPU's call differs from the log's one record in one way.
#[test]
fn replay_memory_diverges_on_each_difference() {
    let record = read(SPACE, 0x1000, &[5, 6]);
    let found = format!("a read of 2 bytes at 0x1000 in address space {}", SPACE.0);
    let cases: [(AsId, u64, usize, String); 3] = [
        (
            AsId(0x22),
            0x1000,
            2,
            "the GPU read 2 bytes at 0x1000 in address space 34".to_owned(),
        ),
        (
            SPACE,
            0x1004,
            2,
            format!(
                "the GPU read 2 bytes at 0x1004 in address space {}",
                SPACE.0
            ),
        ),
        (
            SPACE,
            0x1000,
            3,
            format!(
                "the GPU read 3 bytes at 0x1000 in address space {}",
                SPACE.0
            ),
        ),
    ];
    for (space, va, len, made) in cases {
        let log = seams::Front::new(vec![record.clone()], blobs(&[&[5, 6]]));
        let mut buf = vec![0xee; len];
        assert_eq!(
            log.memory(space).read(va, &mut buf),
            Err(GpuFault::Unmapped { va })
        );
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(
            log.check().unwrap_err(),
            ReplayError::Diverged(Divergence {
                thread: Thread::Front,
                record: 0,
                found: found.clone(),
                made,
            })
        );
        assert_eq!(log.next(), 0);
    }
    // A write where the log has a read.
    let log = seams::Front::new(vec![record.clone()], blobs(&[&[5, 6]]));
    assert_eq!(
        log.memory(SPACE).write(0x1000, &[5, 6]),
        Err(GpuFault::Unmapped { va: 0x1000 })
    );
    assert!(matches!(log.check(), Err(ReplayError::Diverged(_))));
    // A write of other bytes.
    let log = seams::Front::new(
        vec![FrontRecord::Access(Access::Write {
            space: SPACE,
            va: 0x1000,
            data: ContentHash::of(&[1]),
            result: Ok(()),
        })],
        blobs(&[&[1]]),
    );
    assert_eq!(
        log.memory(SPACE).write(0x1000, &[2]),
        Err(GpuFault::Unmapped { va: 0x1000 })
    );
    let hash = ContentHash::of(&[2]);
    assert_eq!(
        log.check().unwrap_err().to_string(),
        format!(
            "the replay diverged at front log record 0: the capture has a write of blob {} at \
             0x1000 in address space {}, but the GPU wrote blob {hash} at 0x1000 in address \
             space {}",
            ContentHash::of(&[1]),
            SPACE.0,
            SPACE.0
        )
    );
    assert_eq!(log.writes(), []);
    // A query past the log's end.
    let log = seams::Front::new(vec![], Blobs::new());
    assert_eq!(log.memory(SPACE).generation(), 0);
    assert_eq!(
        log.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            record: 0,
            found: "the end of the log".to_owned(),
            made: format!("the GPU asked address space {} for its generation", SPACE.0),
        })
    );
}

/// A GPU-only query, a generation query and a write each diverge on every field they match:
/// the address space, the address and (a query's) length, each the only difference.
#[test]
fn queries_and_writes_diverge_on_each_difference() {
    let gpu_only = FrontRecord::Access(Access::GpuOnly {
        space: SPACE,
        va: 0x5000,
        len: 0x10,
        answer: true,
    });
    let found = format!(
        "a GPU-only query of 16 bytes at 0x5000 in address space {}",
        SPACE.0
    );
    for (space, va, len) in [
        (AsId(0x22), 0x5000, 0x10),
        (SPACE, 0x5004, 0x10),
        (SPACE, 0x5000, 0x11),
    ] {
        let log = seams::Front::new(vec![gpu_only.clone()], Blobs::new());
        assert!(!log.memory(space).is_gpu_only(va, len));
        assert_eq!(
            log.check().unwrap_err(),
            ReplayError::Diverged(Divergence {
                thread: Thread::Front,
                record: 0,
                found: found.clone(),
                made: format!(
                    "the GPU asked whether {len} bytes at {va:#x} in address space {} are GPU-only",
                    space.0
                ),
            })
        );
    }
    // A generation query of another space.
    let log = seams::Front::new(
        vec![FrontRecord::Access(Access::Generation {
            space: SPACE,
            value: 0x55,
        })],
        Blobs::new(),
    );
    assert_eq!(log.memory(AsId(0x22)).generation(), 0);
    assert_eq!(
        log.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            record: 0,
            found: format!("a generation query of address space {}", SPACE.0),
            made: "the GPU asked address space 34 for its generation".to_owned(),
        })
    );
    // A write of the logged bytes at another address, and in another space.
    let hash = ContentHash::of(&[1]);
    let write = FrontRecord::Access(Access::Write {
        space: SPACE,
        va: 0x1000,
        data: hash,
        result: Ok(()),
    });
    let found = format!(
        "a write of blob {hash} at 0x1000 in address space {}",
        SPACE.0
    );
    for (space, va) in [(SPACE, 0x1004), (AsId(0x22), 0x1000)] {
        let log = seams::Front::new(vec![write.clone()], blobs(&[&[1]]));
        assert_eq!(
            log.memory(space).write(va, &[1]),
            Err(GpuFault::Unmapped { va })
        );
        assert_eq!(
            log.check().unwrap_err(),
            ReplayError::Diverged(Divergence {
                thread: Thread::Front,
                record: 0,
                found: found.clone(),
                made: format!(
                    "the GPU wrote blob {hash} at {va:#x} in address space {}",
                    space.0
                ),
            })
        );
        assert_eq!(log.writes(), []);
    }
    // The same write matches, and is kept by its blob and length.
    let log = seams::Front::new(vec![write], blobs(&[&[1]]));
    log.memory(SPACE).write(0x1000, &[1]).unwrap();
    log.check().unwrap();
    assert_eq!(
        log.writes(),
        [crate::capture::ReplayedWrite {
            thread: Thread::Front,
            space: SPACE,
            va: 0x1000,
            data: hash,
            len: 1,
            result: Ok(()),
        }]
    );
}

#[test]
fn the_first_divergence_is_kept_and_every_later_call_fails() {
    let log = seams::Front::new(
        vec![read(SPACE, 0x1000, &[5]), read(SPACE, 0x2000, &[6])],
        blobs(&[&[5], &[6]]),
    );
    let memory = log.memory(SPACE);
    let mut buf = [0; 1];
    // The first read is at the wrong address; the second matches the log's second record, but
    // the replay has stopped.
    assert!(memory.read(0x1001, &mut buf).is_err());
    assert_eq!(
        memory.read(0x2000, &mut buf),
        Err(GpuFault::Unmapped { va: 0x2000 })
    );
    assert!(!memory.is_gpu_only(0, 1));
    let ReplayError::Diverged(divergence) = log.check().unwrap_err() else {
        unreachable!("not a divergence")
    };
    assert_eq!(divergence.record, 0);
    assert!(divergence.made.contains("0x1001"));
}

#[test]
fn the_replay_clocks_answer_from_their_logs() {
    let front = seams::Front::new(
        vec![FrontRecord::Clock {
            now: Duration::from_nanos(77),
        }],
        Blobs::new(),
    );
    let clock = front.clock();
    assert_eq!(clock.now(), Duration::from_nanos(77));
    assert_eq!(clock.now(), Duration::ZERO);
    assert_eq!(
        front.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            record: 1,
            found: "the end of the log".to_owned(),
            made: "the front read its clock".to_owned(),
        })
    );
    let back = seams::Back::new(vec![
        BackRecord::Elapsed {
            elapsed: Duration::from_nanos(5),
        },
        BackRecord::Timestamp { value: 0x99 },
    ]);
    let clock = back.clock();
    assert_eq!(clock.elapsed(), Duration::from_nanos(5));
    assert_eq!(clock.gpu_timestamp(), 0x99);
    back.check().unwrap();
    // A timestamp where the log has an elapsed time.
    let back = seams::Back::new(vec![BackRecord::Elapsed {
        elapsed: Duration::from_nanos(5),
    }]);
    assert_eq!(back.clock().gpu_timestamp(), 0);
    assert_eq!(
        back.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: 0,
            found: "an elapsed-time reading".to_owned(),
            made: "the back read a timestamp".to_owned(),
        })
    );
}

/// No program and no state: the replay's shader cache reads neither.
struct Nothing;

impl ProgramMemory for Nothing {
    fn read(&self, address: u64, _out: &mut [u8]) -> Result<(), Unmapped> {
        Err(Unmapped { address })
    }
}

impl EngineState for Nothing {
    fn cbuf_word(&self, _stage: GuestStage, _bank: u8, _offset: u32) -> Option<u32> {
        None
    }
    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

fn request() -> ProgramRequest {
    ProgramRequest {
        stage: GuestStage::Pixel,
        address: 0x4000,
    }
}

fn lookup(shaders: &mut impl ShaderLookup) -> SetLookup {
    shaders.lookup(&[request()], &VariantKey::default(), &Nothing, &Nothing)
}

#[test]
fn a_pending_answer_is_fed_back_past_the_reads_the_cache_made() {
    let back = seams::Back::new(vec![
        BackRecord::Access(Access::Read {
            space: SPACE,
            va: 0x4000,
            len: 16,
            result: Err(GpuFault::Unmapped { va: 0x4000 }),
        }),
        BackRecord::Access(Access::Generation {
            space: SPACE,
            value: 1,
        }),
        BackRecord::ShaderLookup {
            cache_accesses: 2,
            requests: vec![request()],
            answer: ShaderAnswer::Pending { ticket: 0x31 },
        },
        BackRecord::ShaderWait {
            ticket: 0x31,
            answer: ShaderAnswer::Pending { ticket: 0x31 },
        },
    ]);
    let mut shaders = back.shaders();
    assert!(matches!(lookup(&mut shaders), SetLookup::Pending(t) if t == Ticket::new(0x31)));
    assert!(
        matches!(shaders.wait(Ticket::new(0x31)), SetLookup::Pending(t) if t == Ticket::new(0x31))
    );
    back.check().unwrap();
    assert_eq!(back.next(), 4);
}

#[test]
fn a_lookup_of_other_programs_diverges_and_a_write_is_not_skipped() {
    let back = seams::Back::new(vec![BackRecord::ShaderLookup {
        cache_accesses: 0,
        requests: vec![ProgramRequest {
            stage: GuestStage::Vertex,
            address: 0x4000,
        }],
        answer: ShaderAnswer::Pending { ticket: 1 },
    }]);
    lookup(&mut back.shaders());
    assert_eq!(
        back.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: 0,
            found: "a shader lookup of 1 programs".to_owned(),
            made: "the back looked up 1 programs".to_owned(),
        })
    );
    let back = seams::Back::new(vec![
        BackRecord::Access(Access::Write {
            space: SPACE,
            va: 0,
            data: ContentHash::of(&[]),
            result: Ok(()),
        }),
        BackRecord::ShaderLookup {
            cache_accesses: 0,
            requests: vec![request()],
            answer: ShaderAnswer::Pending { ticket: 1 },
        },
    ]);
    lookup(&mut back.shaders());
    assert!(matches!(
        back.check().unwrap_err(),
        ReplayError::Diverged(Divergence { record: 0, .. })
    ));
}

/// A read of the back's own, then the cache's one read, then the lookup that counts one.
fn back_read_then_lookup() -> Vec<BackRecord> {
    let read = |va: u64| {
        BackRecord::Access(Access::Read {
            space: SPACE,
            va,
            len: 4,
            result: Err(GpuFault::Unmapped { va }),
        })
    };
    vec![
        read(0x1000),
        read(0x4000),
        BackRecord::ShaderLookup {
            cache_accesses: 1,
            requests: vec![request()],
            answer: ShaderAnswer::Pending { ticket: 7 },
        },
    ]
}

/// The lookup steps past exactly the accesses its record counts: a read of the back's own that
/// the replayed back did not make, just before a lookup, is a divergence at that read's index.
#[test]
fn a_back_read_the_replay_skipped_before_a_lookup_diverges_at_its_index() {
    // The back makes its read, then looks up: the cache's one read is stepped past.
    let back = seams::Back::new(back_read_then_lookup());
    let mut buf = [0; 4];
    let reads = back.memory(SPACE);
    assert_eq!(
        reads.read(0x1000, &mut buf),
        Err(GpuFault::Unmapped { va: 0x1000 })
    );
    assert!(matches!(lookup(&mut back.shaders()), SetLookup::Pending(t) if t == Ticket::new(7)));
    back.check().unwrap();
    assert_eq!(back.next(), 3);

    // The back skips its read and looks up at once.
    let back = seams::Back::new(back_read_then_lookup());
    assert!(matches!(
        lookup(&mut back.shaders()),
        SetLookup::Pending(t) if t == Ticket::new(0)
    ));
    assert_eq!(
        back.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: 0,
            found: format!("a read of 4 bytes at 0x1000 in address space {}", SPACE.0),
            made: "the back looked up 1 programs".to_owned(),
        })
    );
    assert_eq!(back.next(), 0);

    // A lookup that counts more accesses than precede it diverges at the cursor too.
    let mut records = back_read_then_lookup();
    records.drain(..2);
    let back = seams::Back::new(records);
    lookup(&mut back.shaders());
    assert_eq!(
        back.check().unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: 0,
            found: "a shader lookup of 1 programs".to_owned(),
            made: "the back looked up 1 programs".to_owned(),
        })
    );
}

/// A shader cache that reads through a recording memory while it answers, as a live cache reads
/// its programs through the back's memory.
struct ReadingShaders {
    memory: RecordingMemory,
    reads: usize,
}

impl ShaderLookup for ReadingShaders {
    fn lookup(
        &mut self,
        _requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        let mut buf = [0; 4];
        for n in 0..self.reads {
            let va = GPU_BASE + 4 * u64::try_from(n).unwrap();
            self.memory.read(va, &mut buf).unwrap();
        }
        SetLookup::Pending(Ticket::new(3))
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        SetLookup::Pending(ticket)
    }
}

impl crate::back::buffer::InvalidationSink for ReadingShaders {
    fn invalidate(&mut self, _range: PageRange) {}
}

#[test]
fn recording_shaders_count_the_reads_the_cache_made_while_answering() {
    let recorder = Arc::new(Recorder::new(1 << 10));
    let back_memory = RecordingMemory::new(memory(), Arc::clone(&recorder), Thread::Back, SPACE);
    // A read of the back's own before the lookup is not the cache's.
    let mut buf = [0; 4];
    back_memory.read(GPU_BASE + 0x100, &mut buf).unwrap();
    let inner = ReadingShaders {
        memory: RecordingMemory::new(memory(), Arc::clone(&recorder), Thread::Back, SPACE),
        reads: 3,
    };
    let mut shaders = RecordingShaders::new(Box::new(inner), Arc::clone(&recorder));
    lookup(&mut shaders);
    let back = recorder.take().unwrap().back;
    assert_eq!(back.len(), 5);
    assert_eq!(
        back[4],
        BackRecord::ShaderLookup {
            cache_accesses: 3,
            requests: vec![request()],
            answer: ShaderAnswer::Pending { ticket: 3 },
        }
    );
    // The recorded log replays: the back's read, then the lookup past the cache's three.
    let blobs = {
        let mut table = Blobs::new();
        let guest = memory();
        for va in [GPU_BASE + 0x100, GPU_BASE, GPU_BASE + 4, GPU_BASE + 8] {
            guest.read(va, &mut buf).unwrap();
            table.insert(&buf);
        }
        table
    };
    let replayed = seams::Back::with_blobs(back, blobs);
    replayed
        .memory(SPACE)
        .read(GPU_BASE + 0x100, &mut buf)
        .unwrap();
    assert!(
        matches!(lookup(&mut replayed.shaders()), SetLookup::Pending(t) if t == Ticket::new(3))
    );
    replayed.check().unwrap();
    assert_eq!(replayed.next(), 5);
}

#[test]
fn a_ready_or_rejected_answer_stops_the_replay() {
    for (answer, what) in [
        (ShaderAnswer::Ready, "a ready shader answer"),
        (ShaderAnswer::Rejected, "a rejected shader answer"),
    ] {
        let back = seams::Back::new(vec![BackRecord::ShaderLookup {
            cache_accesses: 0,
            requests: vec![request()],
            answer,
        }]);
        assert!(matches!(lookup(&mut back.shaders()), SetLookup::Pending(_)));
        let error = back.check().unwrap_err();
        assert_eq!(
            error,
            ReplayError::Unsupported {
                thread: Thread::Back,
                record: 0,
                what
            }
        );
        assert_eq!(
            error.to_string(),
            format!("back log record 0 is {what}, which replay cannot feed back yet")
        );
    }
}

#[test]
fn recording_shaders_keep_each_lookup_and_wait_with_its_answer() {
    let recorder = Arc::new(Recorder::new(1 << 10));
    let mut shaders = RecordingShaders::new(
        Box::new(crate::capture::builder::PendingShaders),
        Arc::clone(&recorder),
    );
    lookup(&mut shaders);
    shaders.wait(Ticket::new(9));
    assert_eq!(
        recorder.take().unwrap().back,
        [
            BackRecord::ShaderLookup {
                cache_accesses: 0,
                requests: vec![request()],
                answer: ShaderAnswer::Pending { ticket: 0 },
            },
            BackRecord::ShaderWait {
                ticket: 9,
                answer: ShaderAnswer::Pending { ticket: 9 },
            },
        ]
    );
}

#[test]
fn the_waiter_table_snapshots_latches_by_channel_and_is_replaced_whole() {
    let table = WaiterTable::new();
    let waiter = |channel: u32, address: u64| Waiter {
        channel: ChannelId(channel),
        space: SPACE,
        address,
        op: AcquireOp::Equal,
        payload: 1,
    };
    table.register(waiter(1, 0x100));
    table.register(waiter(2, 0x200));
    assert!(table.latch(ChannelId(2)));
    assert!(!table.latch(ChannelId(3)), "no waiter, nothing latched");
    assert_eq!(
        table.snapshot(),
        [(waiter(1, 0x100), false), (waiter(2, 0x200), true)]
    );
    assert!(!table.is_latched(ChannelId(1)));
    // Replaced: the count follows, so the fast paths see the new table.
    table.replace(&[(waiter(3, 0x300), false)]);
    assert_eq!(table.len(), 1);
    assert!(table.has_waiter(SPACE, 0x300));
    assert!(!table.has_waiter(SPACE, 0x100));
    table.replace(&[]);
    assert!(table.is_empty());
    assert!(!table.has_waiter(SPACE, 0x300));
}

/// A built capture with one release.
fn built() -> Capture {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let submission = submission_at(&builder, CMD, &releases(SEM, 1));
    builder.submit(channel, submission).unwrap();
    builder.run().unwrap();
    builder.finish().unwrap()
}

#[test]
fn starting_state_the_replay_cannot_restore_is_refused_by_kind() {
    for kind in EngineStateKind::ALL {
        let mut capture = built();
        let blob = capture.blobs.insert(b"state");
        capture.state.push(StateRecord::Engine {
            kind,
            owner: 1,
            blob,
        });
        let error = replay(capture.clone(), ReplayOptions::default()).unwrap_err();
        assert_eq!(error, ReplayError::UnsupportedState { kind });
        assert_eq!(
            error.to_string(),
            format!(
                "the capture's starting state holds {}, which replay cannot restore yet",
                kind.name()
            )
        );
    }
}

#[test]
fn the_starting_state_is_checked() {
    // A table for a space the header does not list.
    let mut capture = built();
    let StateRecord::AddressTable { blob, .. } = capture.state[0] else {
        unreachable!("the builder saves the table first")
    };
    capture.state.push(StateRecord::AddressTable {
        space: AsId(0x99),
        blob,
    });
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::StateSpace { space: AsId(0x99) }
    );
    // The same space's table twice.
    let mut capture = built();
    let table = capture.state[0];
    capture.state.push(table);
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::StateSpace { space: SPACE }
    );
    // The syncpoints twice.
    let mut capture = built();
    let syncpoints = capture.state[1];
    capture.state.push(syncpoints);
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::StateSyncpoints
    );
    // A table that does not restore.
    let mut capture = built();
    let junk = capture.blobs.insert(b"junk");
    capture.state[0] = StateRecord::AddressTable {
        space: SPACE,
        blob: junk,
    };
    assert!(matches!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::AddressTable { space, .. } if space == SPACE
    ));
    // Syncpoints that do not restore.
    let mut capture = built();
    let junk = capture.blobs.insert(b"junk");
    capture.state[1] = StateRecord::Syncpoints { blob: junk };
    assert!(matches!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Syncpoints(_)
    ));
    // Capabilities that do not decode.
    let mut capture = built();
    let junk = capture.blobs.insert(b"junk");
    capture.header.caps = junk;
    assert!(matches!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Caps(_)
    ));
}

#[test]
fn a_channel_a_fresh_gpu_would_number_differently_is_refused() {
    let mut capture = built();
    capture.header.channels[0].id = ChannelId(5);
    let error = replay(capture.clone(), ReplayOptions::default()).unwrap_err();
    assert_eq!(
        error,
        ReplayError::ChannelId {
            expected: ChannelId(5),
            made: ChannelId(0)
        }
    );
    assert_eq!(
        error.to_string(),
        "the capture's channel 5 would be channel 0 in a fresh GPU"
    );
}

#[test]
fn a_drain_record_is_refused_where_the_back_meets_it() {
    let mut capture = built();
    capture.back.insert(
        0,
        BackRecord::Drain {
            space: SPACE,
            ranges: vec![],
        },
    );
    let error = replay(capture.clone(), ReplayOptions::default()).unwrap_err();
    assert_eq!(
        error,
        ReplayError::Unsupported {
            thread: Thread::Back,
            record: 0,
            what: "a write-tracking drain"
        }
    );
}

#[test]
fn a_record_only_a_turn_reads_diverges_between_turns() {
    let mut capture = built();
    capture.front.insert(
        0,
        FrontRecord::Clock {
            now: Duration::from_nanos(1),
        },
    );
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Front,
            record: 0,
            found: "a clock reading".to_owned(),
            made: "the front had finished its turn".to_owned(),
        })
    );
    let mut capture = built();
    capture.back.insert(0, BackRecord::Timestamp { value: 1 });
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: 0,
            found: "a timestamp reading".to_owned(),
            made: "the back had finished its step".to_owned(),
        })
    );
}

#[test]
fn a_completion_whose_fence_the_backend_has_not_reached_diverges() {
    // A present records commands, so its list is closed and completed.
    let mut builder = builder();
    builder.present(present_request()).unwrap();
    builder.run().unwrap();
    let mut capture = builder.finish().unwrap();
    let index = capture
        .back
        .iter()
        .position(|record| matches!(record, BackRecord::Complete { .. }))
        .unwrap();
    let BackRecord::Complete { completed } = &mut capture.back[index] else {
        unreachable!()
    };
    let reached = *completed;
    *completed += 1;
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: index,
            found: format!("completion processing at fence {}", reached + 1),
            made: format!("the backend had completed fence {reached}"),
        })
    );
    // A completion below the fence the backend reached diverges too: the value must be equal.
    let BackRecord::Complete { completed } = &mut capture.back[index] else {
        unreachable!()
    };
    assert!(reached > 0, "the present's list completed a fence");
    *completed = reached - 1;
    assert_eq!(
        replay(capture.clone(), ReplayOptions::default()).unwrap_err(),
        ReplayError::Diverged(Divergence {
            thread: Thread::Back,
            record: index,
            found: format!("completion processing at fence {}", reached - 1),
            made: format!("the backend had completed fence {reached}"),
        })
    );
    // At the fence reached, it replays.
    let BackRecord::Complete { completed } = &mut capture.back[index] else {
        unreachable!()
    };
    *completed = reached;
    replay(capture, ReplayOptions::default()).unwrap();
}
