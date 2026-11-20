//! vi2's presents as a run installs the graphics server ([`install_with_graphics`]), behind the
//! real dispatcher: a native image made and bound through the root, a client registered,
//! bound and given its layer through the vi2 session 305 hands out, and a present that the
//! timer thread's turn latches on the vsync tick, handing it to the run's GPU front and
//! releasing the frame it replaces to the guest's notification event.
//!
//! Requests are laid out by hand from PR's checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97), as gsv's
//! own tests lay them out: `CreateNativeMemory` (`0x7f, 3`: mode, size), `BindNativeMemory`
//! (`0x82, 9`: id, address, size, flags, offset), `CreateNativeImage` (`0x1bd, 18`: the
//! preference at 2–17, the count at 18, the memory at 19),
//! `CreateNativeWindowAliasImagePool` (`0x14b, 2`: image, count), and the vi2 session's
//! `OpenDisplay` (`0x75, 3`), `RegisterClientState` (`0xd9, 4`), `BindClientDisplay` (`0x10b,
//! 4`), `GetClientAvailability` (`0xe4, 2`), `GetNotificationEvent` (`0x1b, 1`) and
//! `PresentClientBuffer` (`0xf5, 24`: the token at 2–3, the slot at 4, the image at 5, the
//! offset at 6, the shown extent at 7, the layout at 8, the fence at 22–25).

use starquake_gsv::native::layout::{FORMAT_RGBA8, KIND_NATIVE, PREFERENCE_WORDS};
use starquake_gsv::root::DISPLAY_SESSION_FLAGS;
use starquake_gsv::vi2::command as vi2_command;
use starquake_gsv::vi2::{
    ALL_SLOTS, DEFAULT_DISPLAY, NO_FENCE, OPEN_DISPLAY_FLAGS, PRESENT_BLOCK_HEIGHT, PRESENT_LAYOUT,
    PRESENT_WORD_FENCE, PRESENT_WORD_LAYOUT,
};

use std::num::NonZeroU64;

use starquake_trace::SystemClock;

use super::*;
use crate::hle::graphics::{ReplayError, frames_to_hand};
use crate::hle::tests::Scratch;

/// The client's token and its two notification bits.
const TOKEN: u64 = 0x0000_0055_0000_0066;
const RETIRE: u32 = 1;
const AVAILABLE: u32 = 2;

/// The image's memory: the rig's last two pages, the size of one 16 by 16 image of one GOB a
/// row in 128-row blocks (`1 * 64 * 128` bytes). `crate::machine`'s frame tests present an image
/// of this size too.
const IMAGE_AT: u64 = BASE + 2 * PAGE_SIZE;
pub(crate) const IMAGE_SIZE: u64 = 2 * PAGE_SIZE;

/// The image's extent word: 16 by 16.
pub(crate) const EXTENT: u32 = 15 | (15 << 16);

/// The data words of a request for a `u64`.
fn words_of(value: u64) -> [u32; 2] {
    halves(value)
}

/// A bound native image in a window pool, made through the root: its id.
fn native_image(rig: &Rig, root: Handle) -> u32 {
    let [size_low, size_high] = words_of(IMAGE_SIZE);
    let created = rig.tipc(
        root,
        &[
            tipc_type(root_command::CREATE_NATIVE_MEMORY),
            3,
            0,
            size_low,
            size_high,
        ],
    );
    assert_eq!(created.result, 0);
    let memory = created.data[0];
    let [at_low, at_high] = words_of(IMAGE_AT);
    let bound = rig.tipc(
        root,
        &[
            tipc_type(root_command::BIND_NATIVE_MEMORY),
            9,
            memory,
            at_low,
            at_high,
            size_low,
            size_high,
            0,
            0,
            0,
            0,
        ],
    );
    assert_eq!(bound.result, 0);
    let mut preference = [0; PREFERENCE_WORDS];
    preference[1] = KIND_NATIVE;
    preference[2] = EXTENT;
    preference[4] = PRESENT_BLOCK_HEIGHT;
    preference[5] = 1;
    preference[6] = FORMAT_RGBA8;
    preference[7] = 1;
    let mut words = vec![tipc_type(root_command::CREATE_NATIVE_IMAGE), 18];
    words.extend(preference);
    words.extend([1, memory]);
    let image = rig.tipc(root, &words);
    assert_eq!(image.result, 0);
    let image = image.data[0];
    let pooled = rig.tipc(
        root,
        &[
            tipc_type(root_command::CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL),
            2,
            image,
            1,
        ],
    );
    assert_eq!(pooled.result, 0);
    image
}

/// 229 of `image` in `slot`, the whole image shown, with no fence.
fn present(image: u32, slot: u32) -> Vec<u32> {
    fenced_present(image, slot, NO_FENCE, 0)
}

/// 229 of `image` in `slot`, the whole image shown, with the fence word `fence` (`!id`, or
/// [`NO_FENCE`]) and the fence value `value` (PR `fence_id = ~packet[22]`, the value at 24).
fn fenced_present(image: u32, slot: u32, fence: u32, value: u32) -> Vec<u32> {
    let mut words = vec![tipc_type(vi2_command::PRESENT_CLIENT_BUFFER), 24];
    words.extend(words_of(TOKEN));
    words.extend([slot, image, 0, EXTENT, PRESENT_LAYOUT]);
    words.extend(vec![0; PRESENT_WORD_FENCE - PRESENT_WORD_LAYOUT - 1]);
    words.extend([fence, 0, value, 0]);
    words
}

/// The number gsv's present table gives the run's first client.
const CLIENT: u8 = 0;

/// A vi2 session from 305 with its notification event, the default display open, and the
/// client [`TOKEN`] registered, bound and given its layer (212), and a native image to present:
/// the session, the event, the 212 request and the image.
fn presenting(rig: &Rig) -> (Handle, Handle, [u32; 4], u32) {
    let root = rig.graphics_root();
    let image = native_image(rig, root);
    let [low, high] = halves(DISPLAY_SESSION_FLAGS[0]);
    let opened = rig.tipc(
        root,
        &[tipc_type(root_command::OPEN_DISPLAY_SESSION), 2, low, high],
    );
    assert_eq!((opened.result, opened.copied.len()), (0, 1));
    let vi2 = opened.copied[0];

    let notify = rig.tipc(vi2, &[tipc_type(vi2_command::GET_NOTIFICATION_EVENT), 1, 0]);
    assert_eq!((notify.result, notify.copied.len()), (0, 1));
    let notification = notify.copied[0];
    let [name_low, name_high] = words_of(DEFAULT_DISPLAY);
    let display = rig.tipc(
        vi2,
        &[
            tipc_type(vi2_command::OPEN_DISPLAY),
            3,
            name_low,
            name_high,
            OPEN_DISPLAY_FLAGS,
        ],
    );
    assert_eq!(display.result, 0);
    let [token_low, token_high] = words_of(TOKEN);
    let registered = rig.tipc(
        vi2,
        &[
            tipc_type(vi2_command::REGISTER_CLIENT_STATE),
            4,
            token_low,
            token_high,
            RETIRE,
            AVAILABLE,
        ],
    );
    assert_eq!(registered.result, 0);
    let bound = rig.tipc(
        vi2,
        &[
            tipc_type(vi2_command::BIND_CLIENT_DISPLAY),
            4,
            token_low,
            token_high,
            name_low,
            name_high,
        ],
    );
    assert_eq!(bound.result, 0);
    let availability = [
        tipc_type(vi2_command::GET_CLIENT_AVAILABILITY),
        2,
        token_low,
        token_high,
    ];
    let layer = rig.tipc(vi2, &availability);
    assert_eq!((layer.result, mask(&layer.data)), (0, ALL_SLOTS));
    (vi2, notification, availability, image)
}

/// The free-slot mask a 228, 229 or 212 reply's data starts with.
fn mask(data: &[u32]) -> u64 {
    u64::from(data[0]) | (u64::from(data[1]) << 32)
}

/// The production constructor's wiring carries a present from the vi2 session to the latch: the
/// turn at the vsync deadline latches it and hands it to the run's front, and the next present's
/// latch releases the first one's slot to the guest's notification event, which 212 then shows.
#[test]
fn a_present_reaches_the_latch_on_a_vsync_tick() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (vi2, notification, availability, image) = presenting(&rig);
    assert!(graphics.is_idle());

    // The present takes slot 0 and starts the clock; nothing is latched before the deadline.
    let first = rig.tipc(vi2, &present(image, 0));
    assert_eq!((first.result, mask(&first.data)), (0, ALL_SLOTS & !1));
    assert_eq!(first.data[2..], [0, 0, 0, 0]);
    let deadline = graphics.deadline().unwrap();
    assert!(!graphics.is_idle());
    let before = rig.all_events().len();
    rig.turn_at(&graphics, HostNanos(deadline.0 - 1));
    assert_eq!(notes(&rig.all_events()[before..]), []);
    assert_eq!(graphics.null_front().unwrap().presented(), 0);
    rig.turn_at(&graphics, deadline);
    let latched = Note::FrameLatched {
        client: CLIENT,
        slot: 0,
        image,
    };
    assert_eq!(
        notes(&rig.all_events()[before..]),
        [Note::VsyncTick { tick: 1 }, latched]
    );
    assert!(rig.all_events()[before..].contains(&(TIMER.0, TIMER.1, latched.event(GSV_ROOT))));
    assert_eq!(graphics.null_front().unwrap().presented(), 1);
    assert_eq!(poll(&rig, notification), KernelError::TimedOut.code());

    // The second present's tick releases slot 0: the guest's event is signalled and 212 shows
    // the slot free again.
    let second = rig.tipc(vi2, &present(image, 1));
    assert_eq!((second.result, mask(&second.data)), (0, ALL_SLOTS & !0b11));
    let deadline = graphics.deadline().unwrap();
    let before = rig.all_events().len();
    rig.turn_at(&graphics, deadline);
    assert_eq!(
        notes(&rig.all_events()[before..]),
        [
            Note::VsyncTick { tick: 2 },
            Note::FrameLatched {
                client: CLIENT,
                slot: 1,
                image
            },
            Note::BufferReleased {
                client: CLIENT,
                slot: 0,
                image
            },
        ]
    );
    assert_eq!(graphics.null_front().unwrap().presented(), 2);
    assert_eq!(poll(&rig, notification), ResultCode::SUCCESS);
    let freed = rig.tipc(vi2, &availability);
    assert_eq!((freed.result, mask(&freed.data)), (0, ALL_SLOTS & !0b10));
    // Nothing was left for the clock once the queue emptied: that tick stopped it, and the run
    // is idle.
    assert_eq!(graphics.deadline(), None);
    assert!(graphics.is_idle());
}

/// A replay refuses the latch's notes: they follow from the ticks and the syncpoint values fed
/// back, not from outside.
#[test]
fn a_replay_does_not_feed_back_the_latch_s_notes() {
    let graphics = Graphics::new(FrontMode::Replay, None);
    for note in [
        Note::FrameLatched {
            client: 5,
            slot: 1,
            image: 2,
        },
        Note::BufferReleased {
            client: 6,
            slot: 3,
            image: 4,
        },
    ] {
        assert_eq!(
            graphics.replay(note),
            Err(ReplayError::NotAnInput { code: note.code() })
        );
    }
    assert!(graphics.is_idle());
}

/// A present fenced on a syncpoint whose counter a submission already moved past the fence (the
/// null front performs a submission's increments at once, so this is the common case) replays:
/// the live run's drain records the value that reached the frame, and a replay, whose counter
/// moves only on the notes fed back, latches it and then releases it on the same ticks, with the
/// same events and the same state (`starquake_gsv::shared::present`, "Reached means drained").
#[test]
fn an_already_reached_fence_replays_to_the_same_latch() {
    // `turn` is the turn of tick `n`; `live` says whether the submission's increment is
    // performed (the run's front) or not (a replay's).
    let session = |rig: &Rig, graphics: &Graphics, live: bool, turn: &mut dyn FnMut(u64)| {
        let (vi2, notification, availability, image) = presenting(rig);
        let id = graphics.shared().syncpoints().allocate().unwrap();
        if live {
            graphics.shared().syncpoints().increment_by(id, 2);
        }
        let fence = !u32::from(id.get());
        let first = rig.tipc(vi2, &fenced_present(image, 0, fence, 1));
        assert_eq!((first.result, mask(&first.data)), (0, ALL_SLOTS & !1));
        turn(1);
        assert_eq!(poll(rig, notification), KernelError::TimedOut.code());
        let second = rig.tipc(vi2, &present(image, 1));
        assert_eq!((second.result, mask(&second.data)), (0, ALL_SLOTS & !0b11));
        turn(2);
        assert_eq!(poll(rig, notification), ResultCode::SUCCESS);
        let freed = rig.tipc(vi2, &availability);
        assert_eq!((freed.result, mask(&freed.data)), (0, ALL_SLOTS & !0b10));
        (id, image)
    };
    let run = Graphics::new(FrontMode::Live, None);
    let recorded = rig_with(&run);
    let mut drained: Vec<Vec<Note>> = Vec::new();
    let (id, image) = session(&recorded, &run, true, &mut |_| {
        let before = recorded.all_events().len();
        let deadline = run.deadline().unwrap();
        recorded.turn_at(&run, deadline);
        drained.push(notes(&recorded.all_events()[before..]));
    });
    let latched = |slot| Note::FrameLatched {
        client: CLIENT,
        slot,
        image,
    };
    assert_eq!(
        drained,
        [
            vec![
                Note::SyncpointObserved { id, value: 2 },
                Note::VsyncTick { tick: 1 },
                latched(0),
            ],
            vec![
                Note::VsyncTick { tick: 2 },
                latched(1),
                Note::BufferReleased {
                    client: CLIENT,
                    slot: 0,
                    image
                },
            ],
        ]
    );
    assert_eq!(run.null_front().unwrap().presented(), 2);

    let replay = Graphics::new(FrontMode::Replay, None);
    let replayed = rig_with(&replay);
    let mut fed = drained.iter();
    session(&replayed, &replay, false, &mut |_| {
        for &note in fed.next().unwrap() {
            let fed_back = replay.replay(note);
            if matches!(
                note,
                Note::FrameLatched { .. } | Note::BufferReleased { .. }
            ) {
                assert_eq!(fed_back, Err(ReplayError::NotAnInput { code: note.code() }));
            } else {
                assert_eq!(fed_back, Ok(()));
            }
        }
        replayed.turn_at(&replay, HostNanos(0));
    });
    assert_eq!(replayed.all_events(), recorded.all_events());
    assert_eq!(state_of(&replayed, &replay), state_of(&recorded, &run));
    assert_eq!(replay.null_front().unwrap().presented(), 2);
}

// ----- Frames: the count, the limit and the dumps (issue #386) ---------------------------------

/// `count` presents of `image`, one per slot from slot 0, then turns at each vsync deadline until
/// the clock stops, at most `count + 1` turns; the presents each turn handed the front.
fn present_and_latch(rig: &Rig, graphics: &Graphics, count: u32) -> Vec<u64> {
    let (vi2, _, _, image) = presenting(rig);
    for slot in 0..count {
        let reply = rig.tipc(vi2, &present(image, slot));
        assert_eq!(reply.result, 0, "slot {slot}");
    }
    let mut handed = Vec::new();
    for _ in 0..=count {
        let Some(deadline) = graphics.deadline() else {
            break;
        };
        let before = graphics.frames();
        rig.turn_at(graphics, deadline);
        handed.push(graphics.frames() - before);
    }
    handed
}

#[test]
fn frames_to_hand_leaves_room_for_exactly_the_limit() {
    let limit = |n: u64| NonZeroU64::new(n);
    // No limit: every latched frame, however many were handed.
    for (handed, latched) in [(0, 0), (0, 3), (u64::MAX, 2)] {
        assert_eq!(frames_to_hand(handed, None, latched), latched);
    }
    // Room left at, one below and one above each edge.
    let cases = [
        // (handed, limit, latched, expected)
        (0, 3, 1, 1),
        (2, 3, 1, 1),
        (2, 3, 2, 1),
        (3, 3, 1, 0),
        (4, 3, 1, 0),
        (0, 3, 3, 3),
        (0, 3, 4, 3),
        (1, 3, 0, 0),
        // A limit far past what a `usize` counts hands every latched frame.
        (0, u64::MAX, 5, 5),
        (u64::MAX, u64::MAX, 5, 0),
    ];
    for (handed, max, latched, expected) in cases {
        assert_eq!(
            frames_to_hand(handed, limit(max), latched),
            expected,
            "{handed} handed, limit {max}, {latched} latched"
        );
    }
}

/// With a limit of `N`, presents of `N - 1`, `N` and `N + 1` frames hand the front `N - 1`, `N`
/// and `N` (module docs, "Frames"): the count and the null front's agree, and the limit is
/// reached exactly when `N` were handed.
#[test]
fn a_frame_limit_hands_the_front_exactly_that_many_presents() {
    const N: u32 = 3;
    let limit = NonZeroU64::new(u64::from(N));
    for (count, handed, reached) in [(N - 1, N - 1, false), (N, N, true), (N + 1, N, true)] {
        let graphics = Graphics::new(FrontMode::Live, None).with_frame_limit(limit);
        let rig = rig_with(&graphics);
        let turns = present_and_latch(&rig, &graphics, count);
        // One frame a tick, while the limit allows.
        assert!(turns.iter().all(|&turn| turn <= 1), "{turns:?}");
        assert_eq!(graphics.frames(), u64::from(handed), "{count} presented");
        assert_eq!(
            graphics.null_front().unwrap().presented(),
            u64::from(handed),
            "{count} presented"
        );
        assert_eq!(graphics.frame_limit_reached(), reached, "{count} presented");
    }
    // With no limit every frame is handed, and the limit is never reached.
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    present_and_latch(&rig, &graphics, N + 1);
    assert_eq!(graphics.frames(), u64::from(N + 1));
    assert!(!graphics.frame_limit_reached());
}

/// A wake that does nothing: the rig turns the timer thread's turn by hand.
struct NoWake;

impl starquake_gsv::completion::Wake for NoWake {
    fn completion_queued(&self) {}
}

/// The image's pixels as the guest means them, row after row, each pixel distinct.
fn linear_image() -> Vec<u8> {
    let (width, height) = (16_usize, 16_usize);
    (0..width * height)
        .flat_map(|pixel| {
            let low = u8::try_from(pixel % 256).unwrap();
            [low, low ^ 0x5a, 0x80 | low >> 1, 0xff - low]
        })
        .collect()
}

/// `linear` laid out as the GPU reads a presented image ([`IMAGE_AT`]: 16 by 16 RGBA8, the
/// colour GOB, `PRESENT_BLOCK_HEIGHT` as the block height, clamped as the hardware does).
fn tiled_image(linear: &[u8]) -> Vec<u8> {
    use starquake_layout::{Bytes, Extent4D, LinearLayout, Offset4D, Tiling};
    let row = 16 * starquake_render::frame::RGBA8_BYTES;
    let extent: Extent4D<Bytes> = Extent4D::new(u32::try_from(row).unwrap(), 16, 1, 1);
    let gob = starquake_gpu::back::texture::gob_for(starquake_render::PixelFormat::A8B8G8R8Unorm);
    let tiling = Tiling::new(gob, 0, u8::try_from(PRESENT_BLOCK_HEIGHT).unwrap(), 0)
        .unwrap()
        .clamp(extent)
        .unwrap();
    let mut tiled = vec![0; usize::try_from(IMAGE_SIZE).unwrap()];
    starquake_layout::linear_to_tiled(
        &mut tiled,
        extent,
        linear,
        LinearLayout {
            row_stride_b: row,
            plane_stride_b: row * 16,
        },
        Offset4D::new(0, 0, 0, 0),
        extent,
        &tiling,
    )
    .unwrap();
    tiled
}

/// The guest memory the GPU reads in these tests: the rig's pages, with the image the guest
/// wrote at [`IMAGE_AT`], `linear` laid out as the GPU reads it.
fn image_memory(linear: &[u8]) -> Arc<FlatMemory> {
    let memory = Arc::new(FlatMemory::new(
        BASE,
        usize::try_from(PAGES * PAGE_SIZE).unwrap(),
    ));
    memory.write(IMAGE_AT, &tiled_image(linear)).unwrap();
    memory
}

/// Makes the GPU address space presents read through on `graphics`'s front, mapping the image's
/// pages at their own address, as a title's `nvhost-as-gpu` and gpu 101 registration would
/// (`starquake_gsv::gpu::memory`, "What the table shows": registrations identity-map in every
/// space).
fn map_the_image(graphics: &Graphics) {
    let vm = Arc::new(starquake_gpu::vm::GpuVm::new());
    vm.map(
        starquake_memory::PageRange::new(IMAGE_AT, IMAGE_SIZE).unwrap(),
        starquake_gpu::interface::Target::Guest { cpu_va: IMAGE_AT },
        starquake_gpu::interface::Kind(0),
    )
    .unwrap();
    graphics
        .front()
        .create_address_space(starquake_gpu::interface::AsId(0x31), vm);
}

/// The longest a test here waits for the GPU.
const GPU_WAIT: Duration = Duration::from_secs(20);

/// Waits, at most [`GPU_WAIT`], until `gpu` has presented `count` frames and is idle.
fn wait_for_presents(gpu: &starquake_gpu::wiring::Gpu, count: u64) {
    let deadline = std::time::Instant::now() + GPU_WAIT;
    while gpu.counters().presents < count || !gpu.is_idle() {
        assert!(
            std::time::Instant::now() < deadline,
            "the GPU presented {} of {count}",
            gpu.counters().presents
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The end-to-end frame dump (issue #386, "Tests"): a present of an image the guest wrote,
/// through gsv's vi2 session and the timer thread's turn to the GPU a run builds
/// (`crate::machine`'s `gpu` module) over the recording backend, and through the run's own end
/// (`gpu::finish`, which drains and stops the GPU and reads the frame keeper), comes out of the
/// artifacts folder as `frames/000000.png`, whose decoded pixels are the image's.
///
/// Two stand-ins, each said where it is made: the GPU reads its own copy of guest memory (the
/// rig's kernel writes the rig's), into which the test writes the image ([`image_memory`]); and
/// the GPU address space presents read through is made on the GPU front directly
/// ([`map_the_image`]).
#[test]
fn a_presented_image_is_dumped_as_a_png_of_its_pixels() {
    use crate::frames::{FRAMES_FOLDER, frame_file_name};
    use crate::machine::{Ending, gpu};
    use crate::report::{EnvSource, RunReport};

    let linear = linear_image();
    let shared: Arc<dyn GuestMemory> = image_memory(&linear);
    let (graphics, keeper) = gpu::graphics(
        &EnvSource::Live,
        Some(&shared),
        None,
        Arc::new(NoWake),
        SystemClock::new(),
    )
    .unwrap_or_else(|error| unreachable!("{error}"));
    let keeper = keeper.unwrap();
    map_the_image(&graphics);

    let rig = rig_with(&graphics);
    let turns = present_and_latch(&rig, &graphics, 1);
    assert_eq!(turns, [1]);
    let gpu = Arc::clone(graphics.gpu().unwrap());
    wait_for_presents(&gpu, 1);
    let mut ending = Ending::Exited;
    let report = RunReport {
        graphics: gpu::finish(&graphics, Some(&keeper), &mut ending),
        ..RunReport::emulator_error(String::new(), Duration::ZERO)
    };
    assert!(matches!(ending, Ending::Exited));
    assert!(!gpu.is_running());
    assert_eq!(report.graphics.gpu, Some(gpu.counters()));
    assert_eq!(report.graphics.dumps.files.len(), 1);
    assert_eq!(
        (report.graphics.dumps.unknown, report.graphics.dumps.dropped),
        (0, 0)
    );
    let scratch = Scratch::new("frame-dump");
    let folder = scratch.path().join("artifacts");
    crate::artifacts::write(&folder, &report).unwrap();

    let png = fs::read(folder.join(FRAMES_FOLDER).join(frame_file_name(0))).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    pixels.truncate(info.buffer_size());
    assert_eq!((info.width, info.height), (16, 16));
    assert_eq!(pixels, linear);
    // The present went through the GPU, and `gpu.txt` says so.
    let recorder = fs::read_to_string(folder.join(crate::artifacts::GPU_FILE)).unwrap();
    assert!(recorder.contains("present"), "{recorder}");
    assert_eq!(report.graphics.frames, 1);
}

/// What the run's render backend holds after `count` presents of the image, each latched and
/// handed to the GPU before the next, alternating between two slots: the snapshot's length, the
/// lists it kept, the present feedback waiting, and the bytes of its live resources.
fn backend_held_after(count: u32) -> (usize, usize, usize, u64) {
    use crate::machine::gpu;
    use crate::report::EnvSource;
    use starquake_render::Backend;

    let shared: Arc<dyn GuestMemory> = image_memory(&linear_image());
    let (graphics, built) = gpu::graphics_and_backend(
        &EnvSource::Live,
        Some(&shared),
        None,
        Arc::new(NoWake),
        SystemClock::new(),
    )
    .unwrap_or_else(|error| unreachable!("{error}"));
    let (_keeper, backend) = built.unwrap();
    map_the_image(&graphics);
    let rig = rig_with(&graphics);
    let (vi2, _, _, image) = presenting(&rig);
    for n in 0..count {
        let reply = rig.tipc(vi2, &present(image, n % 2));
        assert_eq!(reply.result, 0, "present {n}");
        let deadline = graphics.deadline().unwrap();
        rig.turn_at(&graphics, deadline);
    }
    assert_eq!(graphics.frames(), u64::from(count));
    let gpu = Arc::clone(graphics.gpu().unwrap());
    wait_for_presents(&gpu, u64::from(count));
    let held = (
        backend.snapshot().len(),
        backend.lists().len(),
        backend.take_present_feedback().len(),
        backend.budget().used,
    );
    gpu.stop().unwrap();
    held
}

/// The run's render backend keeps no call log and no present feedback (`crate::machine`'s `gpu`
/// module, "The GPU's parts"): after fifty presents it holds what it held after one. A backend
/// that kept its log would hold every list, and every upload's staging bytes with it.
#[test]
fn the_run_s_backend_holds_no_more_after_fifty_presents_than_after_one() {
    let one = backend_held_after(1);
    let fifty = backend_held_after(50);
    assert_eq!(fifty, one);
    assert_eq!((one.0, one.1, one.2), (0, 0, 0));
    // The image's texture is live: the backend still models its bytes.
    assert!(one.3 > 0);
}

/// Guest memory whose every read panics: the GPU thread that reads it ends with a panic.
struct PanickingMemory(Arc<FlatMemory>);

impl GuestMemory for PanickingMemory {
    fn read(&self, va: u64, _buf: &mut [u8]) -> Result<(), starquake_memory::Fault> {
        unreachable!("the test's guest memory panics on every read, here at {va:#x}");
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), starquake_memory::Fault> {
        self.0.write(va, data)
    }
}

/// A GPU thread that failed during a run that then reached its stop condition makes the run an
/// emulator error once `gpu::finish` stops the GPU (`crate::machine`'s `gpu` module, "At the
/// end"), and the report keeps the GPU's counters. Its threads have already ended, so `finish`
/// does not wait for it to be idle.
#[test]
fn a_gpu_thread_that_failed_turns_a_clean_ending_into_an_emulator_error() {
    use crate::machine::{Ending, gpu};
    use crate::report::EnvSource;
    use starquake_gpu::wiring::{BackError, WiringError};

    for clean in [Ending::Exited, Ending::FramesReached] {
        let shared: Arc<dyn GuestMemory> = Arc::new(PanickingMemory(image_memory(&linear_image())));
        let (graphics, keeper) = gpu::graphics(
            &EnvSource::Live,
            Some(&shared),
            None,
            Arc::new(NoWake),
            SystemClock::new(),
        )
        .unwrap_or_else(|error| unreachable!("{error}"));
        map_the_image(&graphics);
        let rig = rig_with(&graphics);
        assert_eq!(present_and_latch(&rig, &graphics, 1), [1]);
        let gpu = Arc::clone(graphics.gpu().unwrap());
        let deadline = std::time::Instant::now() + GPU_WAIT;
        while gpu.is_running() {
            assert!(std::time::Instant::now() < deadline, "the GPU never failed");
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut ending = clean;
        let started = std::time::Instant::now();
        let report = gpu::finish(&graphics, keeper.as_deref(), &mut ending);
        assert!(started.elapsed() < gpu::GPU_DRAIN_WAIT / 2);
        match ending {
            Ending::Failed(message) => assert_eq!(
                message,
                format!(
                    "{}: {}",
                    gpu::GPU_FAILED,
                    WiringError::Back(BackError::Panicked)
                )
            ),
            _ => unreachable!("a failed GPU thread must make the run an emulator error"),
        }
        assert!(report.gpu.is_some());
        assert_eq!(report.frames, 1);
    }
}

/// The wgpu backend's tests here run on a real GPU (render-backend slice R12, "Tests"): they
/// skip themselves, with a notice, in CI and where the host has no adapter, and stay local-only
/// until verification 3 shows wgpu runs cleanly on lavapipe (decision 9; `CLAUDE.md`, "Tests").
/// `true` when `test` should run; `result` is what opening the backend gave.
#[cfg(feature = "wgpu")]
fn gpu_test_runs<T>(test: &str, result: &Result<T, String>) -> bool {
    use crate::machine::gpu::WGPU_UNAVAILABLE;
    if std::env::var_os("CI").is_some() {
        eprintln!("skipping {test}: GPU tests are local-only until verification 3 (decision 9)");
        return false;
    }
    let no_adapter = format!("{WGPU_UNAVAILABLE}: no GPU adapter on this host");
    match result {
        Err(message) if message.starts_with(&no_adapter) => {
            eprintln!("skipping {test}: {message}");
            false
        }
        _ => true,
    }
}

/// What one present of the image the guest wrote dumps through the backend `setup` chose, by
/// the run's own path (`gpu::graphics_opening` with the host's opener, the vsync tick's latch,
/// `gpu::finish`): the frame's decoded pixels, and the backend. `Err`: the words the run would
/// end with when the backend cannot be opened.
#[cfg(feature = "wgpu")]
fn frame_dumped_with(
    setup: &crate::renderer::RenderSetup,
) -> Result<(Vec<u8>, Arc<dyn starquake_render::Backend>), String> {
    use crate::machine::{Ending, gpu};
    use crate::report::EnvSource;

    let shared: Arc<dyn GuestMemory> = image_memory(&linear_image());
    let (graphics, built) = gpu::graphics_opening(
        &EnvSource::Live,
        Some(&shared),
        (None, Arc::new(NoWake), SystemClock::new()),
        setup,
        &gpu::WgpuOpener::host(),
    )?;
    let (keeper, backend) = built.unwrap();
    map_the_image(&graphics);
    let rig = rig_with(&graphics);
    assert_eq!(present_and_latch(&rig, &graphics, 1), [1]);
    let gpu = Arc::clone(graphics.gpu().unwrap());
    wait_for_presents(&gpu, 1);
    let mut ending = Ending::Exited;
    let report = gpu::finish(&graphics, Some(&keeper), &mut ending);
    assert!(matches!(ending, Ending::Exited));
    assert_eq!(report.dumps.files.len(), 1);
    assert_eq!((report.dumps.unknown, report.dumps.dropped), (0, 0));
    let decoder = png::Decoder::new(std::io::Cursor::new(&report.dumps.files[0].png));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    pixels.truncate(info.buffer_size());
    assert_eq!((info.width, info.height), (16, 16));
    Ok((pixels, backend))
}

/// Slice R12's local GPU test: a run with the wgpu backend writes the same frame as a run with
/// the recording backend for an uploaded image (the image's own pixels), and keeps no present
/// feedback (`machine/gpu/wgpu_backend.rs`, "Present feedback").
#[cfg(feature = "wgpu")]
#[test]
fn the_wgpu_backend_dumps_the_same_frame_as_the_recording_backend() {
    use crate::renderer::{Choice, IfUnavailable, RenderSetup};

    let wgpu = frame_dumped_with(&RenderSetup::headless(Choice::Wgpu(IfUnavailable::EndRun)));
    if !gpu_test_runs(
        "the_wgpu_backend_dumps_the_same_frame_as_the_recording_backend",
        &wgpu,
    ) {
        return;
    }
    let (wgpu_pixels, wgpu_backend) = wgpu.unwrap();
    let (recording_pixels, recording_backend) =
        frame_dumped_with(&RenderSetup::recording()).unwrap();
    assert_eq!(wgpu_pixels, linear_image());
    assert_eq!(wgpu_pixels, recording_pixels);
    // Each is the backend it was asked for: wgpu has no view swizzle, the recording backend
    // claims one (`crate::machine`'s `gpu` module, `recording_caps`).
    assert!(!wgpu_backend.caps().view_swizzle);
    assert!(recording_backend.caps().view_swizzle);
    // The run's sink took the wgpu backend's present feedback as the frame came.
    assert_eq!(wgpu_backend.take_present_feedback().len(), 0);
}

/// Hands each job to a thread of its own: a main thread that is not the process's.
#[cfg(feature = "wgpu")]
struct OtherThread;

#[cfg(feature = "wgpu")]
impl crate::renderer::MainThread for OtherThread {
    fn post(&self, job: crate::renderer::MainJob) -> Result<(), crate::renderer::MainThreadGone> {
        std::thread::spawn(job);
        Ok(())
    }
}

/// A window's present target goes through the wgpu backend's own checks, on the thread the
/// frontend's main thread runs it on, and a refusal ends the run with the window's words
/// (`machine/gpu.rs`, "The wgpu backend"). The test's "main thread" is not the process's, and
/// its window has no handles: on macOS the backend refuses the thread, elsewhere the surface.
#[cfg(feature = "wgpu")]
#[test]
fn a_window_the_backend_refuses_ends_the_run_with_the_window_s_words() {
    use crate::machine::{gpu, gpu::wgpu_backend::WINDOW_FAILED};
    use crate::renderer::RenderSetup;
    use crate::report::EnvSource;

    let render = RenderSetup::windowed(crate::renderer::tests::no_window(
        320,
        240,
        Arc::new(OtherThread),
    ));
    let shared: Arc<dyn GuestMemory> = image_memory(&linear_image());
    let ended = gpu::graphics_opening(
        &EnvSource::Live,
        Some(&shared),
        (None, Arc::new(NoWake), SystemClock::new()),
        &render,
        &gpu::WgpuOpener::host(),
    )
    .map(|_| ());
    if !gpu_test_runs(
        "a_window_the_backend_refuses_ends_the_run_with_the_window_s_words",
        &ended,
    ) {
        return;
    }
    let refusal = if cfg!(target_os = "macos") {
        starquake_render_wgpu::NOT_MAIN_THREAD
    } else {
        starquake_render_wgpu::NO_SURFACE
    };
    assert_eq!(ended, Err(format!("{WINDOW_FAILED}: {refusal}")));
}
