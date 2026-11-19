//! `gpu_replay` over capture files the GPU crate's builder makes, in a folder unique to each
//! test.

use std::fs;
use std::path::PathBuf;

use starquake_gpu::capture::builder::{BuilderConfig, CaptureBuilder, PendingShaders};
use starquake_gpu::capture::{Capture, FrontRecord, ReplayError};
use starquake_gpu::classes::clc76f;
use starquake_gpu::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
use starquake_gpu::interface::{
    AsId, DisplayId, GpfifoEntry, ImageDesc, LayerId, PresentRequest, Rect, Submission,
    SyncpointId,
};
use starquake_gpu::sema::{self, HostSemaphore, PayloadSize, ReleaseForm};
use starquake_memory::{PAGE_SIZE, PageRange};

use super::*;

const CPU_BASE: u64 = 0x8000_0000;
const PAGES: u64 = 16;
const GPU_BASE: u64 = 0x12_3400_0000;
const SEM: u64 = GPU_BASE + 8 * PAGE_SIZE + 0x10;
const FRAME: u64 = GPU_BASE + 4 * PAGE_SIZE;
const SPACE: AsId = AsId(0x21);

/// A folder for one test, unique to it and to the test process, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("starquake-gpu-replay-{}-{test}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A capture of one release of `payload` at `SEM`, then a present.
fn capture(payload: u32) -> Capture {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mut builder = CaptureBuilder::new(
        BuilderConfig::synthetic("core test", CPU_BASE, size),
        Box::new(PendingShaders),
    )
    .unwrap();
    builder
        .address_space(SPACE, PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(), CPU_BASE)
        .unwrap();
    let channel = builder.channel(SPACE, SyncpointId::new(1).unwrap()).unwrap();
    let op = sema::encode_semaphored(&HostSemaphore::Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        wfi: false,
    })
    .unwrap();
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        0,
        u16::try_from(clc76f::SEMAPHOREA.method()).unwrap(),
        &[
            u32::try_from(SEM >> 32).unwrap(),
            u32::try_from(SEM & 0xffff_ffff).unwrap(),
            payload,
            op,
        ],
    )
    .unwrap();
    let words = pb.into_words();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    builder.cpu_write(SPACE, GPU_BASE, &bytes).unwrap();
    let entry = GpEntry {
        address: GPU_BASE,
        length: u32::try_from(words.len()).unwrap(),
        sync: Sync::Proceed,
        ..GpEntry::default()
    };
    builder
        .submit(
            channel,
            Submission {
                entries: vec![GpfifoEntry(entry.to_raw().unwrap())],
                wait: None,
                increments: 0,
                wfi: false,
            },
        )
        .unwrap();
    builder
        .present(PresentRequest {
            image: ImageDesc {
                address: FRAME,
                width: 8,
                height: 8,
                pitch: 1,
                block_height_log2: 0,
                format: 1,
            },
            shown: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            layer: LayerId(1),
            display: DisplayId(2),
        })
        .unwrap();
    builder.run().unwrap();
    builder.finish().unwrap()
}

fn args(capture: PathBuf, until_present: Option<u64>, dump: Option<Dump>) -> GpuReplayArgs {
    GpuReplayArgs {
        capture,
        until_present,
        dump,
    }
}

#[test]
fn a_capture_replays_and_prints_the_dump_asked_for() {
    let scratch = Scratch::new("dumps");
    let path = scratch.file("one.sqgc", &capture(0x2a).to_bytes().unwrap());
    let summary = "replayed the whole capture: 2 ring entries, 1 presents; the ring and the \
                   render stream match the capture\n";
    assert_eq!(gpu_replay(&args(path.clone(), None, None)).unwrap(), summary);
    assert_eq!(
        gpu_replay(&args(path.clone(), None, Some(Dump::Writes))).unwrap(),
        format!(
            "{summary}back space 33 wrote 4 bytes at {SEM:#x}: 2a 00 00 00\n"
        )
    );
    let ring = gpu_replay(&args(path.clone(), None, Some(Dump::Ring))).unwrap();
    assert_eq!(
        ring,
        format!(
            "{summary}release channel=0 address={SEM:#x} payload=0x2a size=4 timestamp=no \
             op=write\npresent address={FRAME:#x} width=8 height=8 pitch=1 block-height=0 \
             format=1 shown=0,0,8,8 layer=1 display=2\n"
        )
    );
    let render = gpu_replay(&args(path, None, Some(Dump::Render))).unwrap();
    assert!(render.starts_with(summary));
    assert!(render.contains("present v0 (0,0 8x8) Identity frame 0"));
}

#[test]
fn until_present_says_where_it_stopped_or_that_the_capture_ended_first() {
    let scratch = Scratch::new("until");
    let path = scratch.file("one.sqgc", &capture(1).to_bytes().unwrap());
    assert_eq!(
        gpu_replay(&args(path.clone(), Some(1), None)).unwrap(),
        "stopped after present 1: 2 ring entries replayed\n"
    );
    assert_eq!(
        gpu_replay(&args(path, Some(2), None)).unwrap(),
        "replayed the whole capture: 2 ring entries, 1 presents; the ring and the render stream \
         match the capture\nthe capture ended before present 2\n"
    );
}

#[test]
fn each_failure_is_an_emulator_error_in_plain_words() {
    let scratch = Scratch::new("failures");
    let missing = scratch.0.join("missing.sqgc");
    let error = gpu_replay(&args(missing.clone(), None, None)).unwrap_err();
    assert!(matches!(error, GpuReplayError::Read { .. }));
    assert!(
        error
            .to_string()
            .starts_with(&format!("the capture {} could not be read: ", missing.display()))
    );
    assert_eq!(error.exit_code(), exit_code::EMULATOR_ERROR);

    let junk = scratch.file("junk.sqgc", b"JUNKJUNK");
    let error = gpu_replay(&args(junk.clone(), None, None)).unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "the capture {} is malformed: the file is not a GPU capture: it starts with 4a 55 4e \
             4b, not SQGC",
            junk.display()
        )
    );
    assert_eq!(error.exit_code(), exit_code::EMULATOR_ERROR);

    // A capture whose first read was moved: the replay diverges.
    let mut moved = capture(1);
    let index = moved
        .front
        .iter()
        .position(|record| {
            matches!(
                record,
                FrontRecord::Access(starquake_gpu::capture::Access::Read { .. })
            )
        })
        .unwrap();
    if let FrontRecord::Access(starquake_gpu::capture::Access::Read { va, .. }) =
        &mut moved.front[index]
    {
        *va += 4;
    }
    let path = scratch.file("moved.sqgc", &moved.to_bytes().unwrap());
    let error = gpu_replay(&args(path, None, None)).unwrap_err();
    assert!(matches!(
        error,
        GpuReplayError::Replay(ReplayError::Diverged(_))
    ));
    assert!(
        error
            .to_string()
            .starts_with(&format!("the replay diverged at front log record {index}: "))
    );
    assert_eq!(error.exit_code(), exit_code::EMULATOR_ERROR);
}

#[test]
fn a_file_past_the_limit_is_refused_without_being_read_whole() {
    let scratch = Scratch::new("large");
    let path = scratch.0.join("large.sqgc");
    let file = fs::File::create(&path).unwrap();
    // Sparse: no disk space is used for the zeros.
    file.set_len(u64::try_from(MAX_CAPTURE_BYTES).unwrap() + 2)
        .unwrap();
    drop(file);
    let error = gpu_replay(&args(path.clone(), None, None)).unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "the capture {} is malformed: the capture is {} bytes; at most {MAX_CAPTURE_BYTES} \
             bytes are read",
            path.display(),
            MAX_CAPTURE_BYTES + 2
        )
    );
}

#[test]
fn dump_names_are_the_three_the_contract_lists() {
    assert_eq!(Dump::from_name("ring"), Some(Dump::Ring));
    assert_eq!(Dump::from_name("render"), Some(Dump::Render));
    assert_eq!(Dump::from_name("writes"), Some(Dump::Writes));
    assert_eq!(Dump::from_name("Ring"), None);
    assert_eq!(Dump::from_name(""), None);
}
