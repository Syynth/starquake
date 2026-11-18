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

use super::*;
use crate::hle::graphics::ReplayError;

/// The client's token and its two notification bits.
const TOKEN: u64 = 0x0000_0055_0000_0066;
const RETIRE: u32 = 1;
const AVAILABLE: u32 = 2;

/// The image's memory: the rig's last two pages, the size of one 16 by 16 image of one GOB a
/// row in 128-row blocks (`1 * 64 * 128` bytes).
const IMAGE_AT: u64 = BASE + 2 * PAGE_SIZE;
const IMAGE_SIZE: u64 = 2 * PAGE_SIZE;

/// The image's extent word: 16 by 16.
const EXTENT: u32 = 15 | (15 << 16);

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
    let mut words = vec![tipc_type(vi2_command::PRESENT_CLIENT_BUFFER), 24];
    words.extend(words_of(TOKEN));
    words.extend([slot, image, 0, EXTENT, PRESENT_LAYOUT]);
    words.extend(vec![0; PRESENT_WORD_FENCE - PRESENT_WORD_LAYOUT - 1]);
    words.extend([NO_FENCE, 0, 0, 0]);
    words
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
    let root = rig.graphics_root();
    let image = native_image(&rig, root);
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
    assert_eq!(graphics.front().presented(), 0);
    rig.turn_at(&graphics, deadline);
    let latched = Note::FrameLatched { slot: 0, image };
    assert_eq!(
        notes(&rig.all_events()[before..]),
        [Note::VsyncTick { tick: 1 }, latched]
    );
    assert!(rig.all_events()[before..].contains(&(TIMER.0, TIMER.1, latched.event(GSV_ROOT))));
    assert_eq!(graphics.front().presented(), 1);
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
            Note::FrameLatched { slot: 1, image },
            Note::BufferReleased { slot: 0, image },
        ]
    );
    assert_eq!(graphics.front().presented(), 2);
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
        Note::FrameLatched { slot: 1, image: 2 },
        Note::BufferReleased { slot: 3, image: 4 },
    ] {
        assert_eq!(
            graphics.replay(note),
            Err(ReplayError::NotAnInput { code: note.code() })
        );
    }
    assert!(graphics.is_idle());
}
