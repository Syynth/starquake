//! Tests of the capture format, its recording seams and the replay.
//!
//! `golden` pins every record's bytes; `parse` the reader's checks, in order, and its bounds;
//! `seams` the recording and replay seams one call at a time; `replay` whole captures made by
//! the builder from hand-written streams, replayed, and tampered with.

mod drive;
mod golden;
mod mirror;
mod parse;
mod replay;
mod seams;
mod timestamps;

use starquake_memory::{PAGE_SIZE, PageRange};
use starquake_render::command::Rect;

use super::builder::{BuilderConfig, CaptureBuilder, PendingShaders};
use crate::back::draw::WaitWhileCompiling;
use crate::back::exec::PRESENT_FORMAT_RGBA8;
use crate::back::fence::Mode;
use crate::classes::clc76f;
use crate::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
use crate::interface::{
    AsId, ChannelId, DisplayId, GpfifoEntry, ImageDesc, LayerId, PresentRequest, Submission,
    SyncpointId,
};
use crate::sema::{self, AcquireOp, HostSemaphore, PayloadSize, ReleaseForm};

/// The CPU address of the rig's guest memory.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages of guest memory, all mapped.
const PAGES: u64 = 64;
/// The GPU address of `CPU_BASE`: not equal to it, so a missed translation reads elsewhere.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Command buffers.
const CMD: u64 = GPU_BASE;
const CMD2: u64 = GPU_BASE + 2 * PAGE_SIZE;
/// Semaphores: not page aligned.
const SEM: u64 = GPU_BASE + 48 * PAGE_SIZE + 0x10;
const SEM2: u64 = SEM + 0x40;
/// An 8x8 RGBA8 frame.
const FRAME: u64 = GPU_BASE + 32 * PAGE_SIZE;
const FRAME_SIZE: u32 = 8;
/// The rig's address space: not 0, so a hard-coded space fails.
const SPACE: AsId = AsId(0x21);
/// The syncpoint the rig's channels increment.
const SYNCPOINT: u8 = 5;

fn syncpoint() -> SyncpointId {
    SyncpointId::new(SYNCPOINT).unwrap()
}

/// The rig's builder configuration.
fn config() -> BuilderConfig {
    BuilderConfig::synthetic(
        "test build",
        CPU_BASE,
        usize::try_from(PAGES * PAGE_SIZE).unwrap(),
    )
}

/// A builder with [`SPACE`] mapped and no channel.
fn builder() -> CaptureBuilder {
    let mut builder = CaptureBuilder::new(config(), Box::new(PendingShaders)).unwrap();
    builder
        .address_space(
            SPACE,
            PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
            CPU_BASE,
        )
        .unwrap();
    builder
}

/// Words as bytes.
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A GP entry of `length` words at `va`.
fn raw_entry(va: u64, length: usize) -> GpfifoEntry {
    let entry = GpEntry {
        address: va,
        length: u32::try_from(length).unwrap(),
        sync: Sync::Proceed,
        ..GpEntry::default()
    };
    GpfifoEntry(entry.to_raw().unwrap())
}

/// Writes `words` at `va` as the CPU and gives a submission of one entry over them.
fn submission_at(builder: &CaptureBuilder, va: u64, words: &[u32]) -> Submission {
    builder.cpu_write(SPACE, va, &bytes(words)).unwrap();
    plain(vec![raw_entry(va, words.len())])
}

fn plain(entries: Vec<GpfifoEntry>) -> Submission {
    Submission {
        entries,
        wait: None,
        increments: 0,
        wfi: false,
    }
}

/// A method number as a header takes it.
fn m(method: u32) -> u16 {
    u16::try_from(method).unwrap()
}

/// The four `SEMAPHOREA`–`D` words for `address`, `payload` and the operation word `op`.
fn semaphored(pb: &mut PushBuffer, address: u64, payload: u32, op: u32) {
    let upper = u32::try_from(address >> 32).unwrap();
    let lower = u32::try_from(address & 0xffff_ffff).unwrap();
    pb.methods(
        MethodMode::Incrementing,
        0,
        m(clc76f::SEMAPHOREA.method()),
        &[upper, lower, payload, op],
    )
    .unwrap();
}

fn acquire(pb: &mut PushBuffer, address: u64, payload: u32) {
    let word = sema::encode_semaphored(&HostSemaphore::Acquire {
        op: AcquireOp::Equal,
        size: PayloadSize::Four,
        switch: false,
    })
    .unwrap();
    semaphored(pb, address, payload, word);
}

fn release(pb: &mut PushBuffer, address: u64, payload: u32) {
    let word = sema::encode_semaphored(&HostSemaphore::Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        wfi: false,
    })
    .unwrap();
    semaphored(pb, address, payload, word);
}

/// A stream that releases `payload` at `address`.
fn releases(address: u64, payload: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    release(&mut pb, address, payload);
    pb.into_words()
}

/// A stream that waits for `address` to hold `payload`, then releases `then` at `release_at`.
fn acquires_then_releases(address: u64, payload: u32, release_at: u64, then: u32) -> Vec<u32> {
    let mut pb = PushBuffer::new();
    acquire(&mut pb, address, payload);
    release(&mut pb, release_at, then);
    pb.into_words()
}

/// A present of the 8x8 RGBA8 frame at [`FRAME`], all of it shown.
fn present_request() -> PresentRequest {
    PresentRequest {
        image: ImageDesc {
            address: FRAME,
            width: FRAME_SIZE,
            height: FRAME_SIZE,
            pitch: 1,
            block_height_log2: 0,
            format: PRESENT_FORMAT_RGBA8,
        },
        shown: Rect {
            x: 0,
            y: 0,
            width: FRAME_SIZE,
            height: FRAME_SIZE,
        },
        layer: LayerId(1),
        display: DisplayId(2),
    }
}

/// A channel in [`SPACE`].
fn channel(builder: &mut CaptureBuilder) -> ChannelId {
    builder.channel(SPACE, syncpoint()).unwrap()
}
