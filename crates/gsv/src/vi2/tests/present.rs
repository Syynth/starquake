//! 228 and 229 through the vi2 server, with the vsync tick played by the test: the reply bytes,
//! every check in its order, a slot coming back on the latching tick through 212, 21 and the
//! notification event, and the fence wait fallback's held reply.
//!
//! Requests are laid out by hand from PR's checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97):
//! `PresentClientBuffer` compares `packet[0] == 0xf5 && packet[1] == 24` and `QueueNativeImage`
//! reads the token at words 2–3, `DecodeBufferSlot(packet[4])`, the image at 5, the offset at 6,
//! the shown extent at 7, the layout at 8 and the fence at 22–25; `CancelClientBuffer` compares
//! `packet[0] == 0xf4 && packet[1] == 19`, the token at 2–3, the slot at 4.

use starquake_gpu::interface::GpuCompletion;
use starquake_kernel::result::KernelError;

use super::*;
use crate::completion::Completion;
use crate::native::layout::{KIND_NATIVE, PREFERENCE_WORDS, Preference, WORD_ONE};
use crate::native::{BIND_PROCESS_MEMORY, BindRequest, WINDOW_POOL_COUNT};
use crate::note::Note;
use crate::shared::present::slot_bit;

/// The client the tests present for, and its bits.
const TOKEN: u64 = 0x0000_00C3_0000_00D4;
const RETIRE: u32 = 4;
const AVAILABLE: u32 = 6;

/// Where the image's memory is, in the fake caller's memory: two pages, the size of one 16 by
/// 16 image of one GOB a row in 128-row blocks (`1 * 64 * 128` bytes).
const IMAGE_AT: u64 = 0x2_0000;
const IMAGE_SIZE: u64 = 0x2000;

/// The image's extent word: 16 by 16 (`(16 - 1) | (16 - 1) << 16`).
const EXTENT: u32 = 15 | (15 << 16);

/// The words 9 to 21 a 229 request carries and nothing reads.
const UNREAD: usize = PRESENT_WORD_FENCE - PRESENT_WORD_LAYOUT - 1;

/// The fields of one 229 request.
#[derive(Clone, Copy)]
struct Present {
    token: u64,
    slot: u32,
    image: u32,
    offset: u32,
    shown: u32,
    layout: u32,
    fence: u32,
    high: u32,
    value: u32,
    tail: u32,
}

impl Present {
    /// A present of `image` in `slot`, the whole image shown, with no fence.
    fn of(image: u32, slot: u32) -> Self {
        Self {
            token: TOKEN,
            slot,
            image,
            offset: 0,
            shown: EXTENT,
            layout: PRESENT_LAYOUT,
            fence: NO_FENCE,
            high: 0,
            value: 0,
            tail: 0,
        }
    }

    /// The same present, fenced on syncpoint `id` at `value`.
    fn fenced(self, id: SyncpointId, value: u32) -> Self {
        Self {
            fence: !u32::from(id.get()),
            value,
            ..self
        }
    }

    /// 229 laid out word by word (module docs).
    fn words(self) -> Vec<u32> {
        let mut words = vec![0xF5, 24];
        words.extend(w(self.token));
        words.extend([self.slot, self.image, self.offset, self.shown, self.layout]);
        words.extend([0; UNREAD]);
        words.extend([self.fence, self.high, self.value, self.tail]);
        words
    }
}

/// 228 laid out word by word (module docs): the token, the slot, then sixteen words nothing
/// reads.
fn cancel(token: u64, slot: u32) -> Vec<u32> {
    let mut words = vec![0xF4, 19];
    words.extend(w(token));
    words.push(slot);
    words.extend([0; 16]);
    words
}

/// The success reply of 228 or 229 with free slots `available`.
fn slots_reply(command: u32, available: u64) -> Vec<u8> {
    let [low, high] = w(available);
    success(command, &[], &[low, high, 0, 0, 0, 0])
}

/// A native image preference: 16 by 16, one GOB a row, block height `block`, `format`.
fn preference(block: u32, format: u32) -> Preference {
    let mut words = [0; PREFERENCE_WORDS];
    words[1] = KIND_NATIVE;
    words[2] = EXTENT;
    words[4] = block;
    words[5] = 1;
    words[6] = format;
    words[WORD_ONE] = 1;
    words
}

impl Rig {
    /// A rig whose server paces with `pacing`.
    fn paced(pacing: Pacing) -> Self {
        let mut rig = Self::new();
        let env: Arc<dyn Env> = Arc::new(LiveEnv::new(FixedClock::new(START, 7), 0));
        rig.server = vi2_server_paced(UnknownPolicy::Strict, Arc::clone(&rig.shared), pacing)
            .with_service_id(VI2)
            .with_env(env, Arc::new(NoHalt));
        rig.session = rig.kernel.new_object();
        rig.server.accept(rig.session).unwrap();
        rig
    }

    /// A native image in a window pool, its memory bound at [`IMAGE_AT`], with `preference`: its
    /// id. `in_pool` puts it in the window pool.
    fn image_with(&mut self, preference: &Preference, in_pool: bool) -> u32 {
        if self.kernel.memory(IMAGE_AT, 1).is_none() {
            self.kernel
                .map(IMAGE_AT, vec![0; usize::try_from(IMAGE_SIZE).unwrap()]);
        }
        let lock = Kernel::lock_device_share(&mut self.kernel, IMAGE_AT, IMAGE_SIZE).unwrap();
        let mut state = self.shared.lock();
        let native = state.native();
        let memory = native.create_memory(0, IMAGE_SIZE).unwrap();
        let request = BindRequest {
            id: memory,
            address: IMAGE_AT,
            size: IMAGE_SIZE,
            flags: BIND_PROCESS_MEMORY,
            offset: 0,
        };
        native.check_bind(&request).unwrap().bind(lock);
        let image = native.create_image(preference, 1, memory).unwrap();
        if in_pool {
            native
                .set_window_pool(image, true, WINDOW_POOL_COUNT)
                .unwrap();
        }
        image
    }

    /// An RGBA8 image in a window pool, with the block height 229 takes.
    fn image(&mut self) -> u32 {
        self.image_with(&preference(PRESENT_BLOCK_HEIGHT, PRESENT_FORMATS[0]), true)
    }

    /// The client [`TOKEN`], bound to the open display, with its layer (212).
    fn presenting_client(&mut self) {
        self.bound_client(TOKEN, RETIRE, AVAILABLE);
        assert_eq!(
            self.send(&client_only(0xE4, TOKEN)),
            slots_reply_212(ALL_SLOTS)
        );
    }

    /// 229 of `present`, expected to succeed: the free slots it answered.
    fn present(&mut self, present: Present) -> u64 {
        let reply = self.send(&present.words());
        let data = &reply[12..20];
        let available = u64::from_le_bytes(data.try_into().unwrap());
        assert_eq!(reply, slots_reply(PRESENT_CLIENT_BUFFER, available));
        available
    }

    /// The vsync tick at the clock's deadline (none while it is stopped): the notes it recorded.
    fn next_tick(&mut self) -> Vec<Note> {
        let Some(deadline) = self.shared.vsync_deadline() else {
            return Vec::new();
        };
        self.tick(deadline.0)
            .into_iter()
            .filter_map(|kind| match kind {
                EventKind::ServiceNote {
                    service,
                    code,
                    value,
                } if service == GSV => Some(Note::decode(code, value).unwrap()),
                _ => None,
            })
            .collect()
    }

    /// Drains a change of syncpoint `id` to `value`.
    fn drain(&mut self, id: SyncpointId, value: u32) {
        self.shared
            .queue()
            .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
                id,
                value,
            }));
        let mut host = KernelHost {
            kernel: &mut self.kernel,
            recorded: Vec::new(),
        };
        self.shared.drain(&mut host);
    }
}

/// 212's success reply with free slots `available`.
fn slots_reply_212(available: u64) -> Vec<u8> {
    let [low, high] = w(available);
    success(GET_CLIENT_AVAILABILITY, &[], &[low, high])
}

/// The header words are PR's, and the command ids its table's, written out.
#[test]
fn the_present_header_words_are_the_command_ids() {
    let present = Present::of(0, 0).words();
    assert_eq!(present[0], tipc_type(PRESENT_CLIENT_BUFFER));
    assert_eq!(usize::try_from(present[1]).unwrap(), present.len() - 2);
    assert_eq!(present.len(), PRESENT_WORD_FENCE + 4);
    let cancelled = cancel(0, 0);
    assert_eq!(cancelled[0], tipc_type(CANCEL_CLIENT_BUFFER));
    assert_eq!(usize::try_from(cancelled[1]).unwrap(), cancelled.len() - 2);
    assert_eq!((CANCEL_CLIENT_BUFFER, PRESENT_CLIENT_BUFFER), (228, 229));
    assert_eq!((PRESENT_LAYOUT, PRESENT_BLOCK_HEIGHT), (3, 7));
    assert_eq!(PRESENT_FORMATS, [1, 4]);
    assert_eq!(
        (PRESENT_WORD_LAYOUT, PRESENT_WORD_FENCE, NO_FENCE),
        (8, 22, 0)
    );
}

/// A present takes its slot and answers the rest free; the next tick latches it, handing its
/// request to the caller; a second present's tick releases the first one's slot, raising the
/// client's notification and signalling the event, which 212 and 21 then show.
#[test]
fn a_present_s_slot_comes_back_on_the_tick_that_latches_the_next() {
    let mut rig = Rig::new();
    let notification = rig.notification_event();
    rig.presenting_client();
    let image = rig.image();
    assert_eq!(rig.present(Present::of(image, 2)), ALL_SLOTS & !slot_bit(2));
    assert_eq!(
        rig.next_tick(),
        [
            Note::VsyncTick { tick: 1 },
            Note::FrameLatched { slot: 2, image }
        ]
    );
    assert_eq!(rig.signals(notification), 0);
    assert_eq!(
        rig.present(Present::of(image, 5)),
        ALL_SLOTS & !slot_bit(2) & !slot_bit(5)
    );
    assert_eq!(
        rig.next_tick(),
        [
            Note::VsyncTick { tick: 2 },
            Note::FrameLatched { slot: 5, image },
            Note::BufferReleased { slot: 2, image },
        ]
    );
    assert_eq!(rig.signals(notification), 1);
    assert_eq!(
        rig.send(&client_only(0xE4, TOKEN)),
        slots_reply_212(ALL_SLOTS & !slot_bit(5))
    );
    rig.ok(&get_notifications(128));
    let bitmap = rig.kernel.memory(BUFFER, 4).unwrap().to_vec();
    assert_eq!(bitmap, (1_u32 << AVAILABLE).to_le_bytes());
}

/// The latched frame's request names the image's address, size, pitch, block height and
/// format, the shown area of word 7, the client's layer and the default display.
#[test]
fn the_latched_request_describes_the_image() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let shown = 7 | (3 << 16);
    rig.present(Present {
        shown,
        ..Present::of(image, 0)
    });
    let mut host = KernelHost {
        kernel: &mut rig.kernel,
        recorded: Vec::new(),
    };
    let deadline = rig.shared.vsync_deadline().unwrap();
    let presents = rig.shared.tick_vsync(&mut host, deadline);
    assert_eq!(
        presents,
        [PresentRequest {
            image: ImageDesc {
                address: IMAGE_AT,
                width: 16,
                height: 16,
                pitch: 1,
                block_height_log2: PRESENT_BLOCK_HEIGHT,
                format: PRESENT_FORMATS[0],
            },
            shown: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 4,
            },
            layer: LayerId(FIRST_LAYER),
            display: DisplayId(DEFAULT_DISPLAY),
        }]
    );
}

/// An unreached fence holds the frame back tick after tick; the tick after a drained value
/// reaches it latches it.
#[test]
fn a_fenced_present_waits_for_its_fence() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let id = rig.shared.syncpoints().allocate().unwrap();
    rig.present(Present::of(image, 0).fenced(id, 2));
    assert_eq!(rig.next_tick(), [Note::VsyncTick { tick: 1 }]);
    rig.drain(id, 1);
    assert_eq!(rig.next_tick(), [Note::VsyncTick { tick: 2 }]);
    rig.drain(id, 2);
    assert_eq!(
        rig.next_tick(),
        [
            Note::VsyncTick { tick: 3 },
            Note::FrameLatched { slot: 0, image }
        ]
    );
}

/// A present with no listener on the clock starts it, one interval after the `Env` clock's
/// reading; one on a running clock does not move it.
#[test]
fn a_present_starts_a_stopped_clock() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    assert_eq!(rig.shared.vsync_deadline(), None);
    rig.present(Present::of(image, 0));
    let deadline = rig.shared.vsync_deadline().unwrap();
    assert_eq!(deadline.0, START + INTERVAL_NS);
    rig.present(Present::of(image, 1));
    assert_eq!(rig.shared.vsync_deadline(), Some(deadline));
}

/// The refusal of each check of 229, one fault at a time, each with its exact result and the
/// slot still free; and the accepted side of each edge.
#[test]
fn each_present_check_refuses_its_fault() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let loose = rig.image_with(&preference(PRESENT_BLOCK_HEIGHT, 1), false);
    let low = rig.image_with(&preference(PRESENT_BLOCK_HEIGHT - 1, 1), true);
    let free = rig.shared.syncpoints().allocate().unwrap();
    let unallocated = SyncpointId::new(free.get() + 1).unwrap();
    let base = Present::of(image, 0);
    let cases = [
        (Present { token: 0, ..base }, KernelError::NotFound),
        (
            Present {
                token: TOKEN + 1,
                ..base
            },
            KernelError::NotFound,
        ),
        (
            Present {
                slot: u32::from(SLOTS),
                ..base
            },
            KernelError::InvalidState,
        ),
        (
            Present {
                image: image + 100,
                ..base
            },
            KernelError::NotFound,
        ),
        (
            Present {
                image: loose,
                ..base
            },
            KernelError::NotFound,
        ),
        (Present { offset: 1, ..base }, KernelError::NotImplemented),
        (
            Present {
                shown: EXTENT + 1,
                ..base
            },
            KernelError::NotImplemented,
        ),
        (
            Present {
                shown: EXTENT + (1 << 16),
                ..base
            },
            KernelError::NotImplemented,
        ),
        (
            Present {
                layout: PRESENT_LAYOUT + 1,
                ..base
            },
            KernelError::NotImplemented,
        ),
        (Present { image: low, ..base }, KernelError::NotImplemented),
        (Present { high: 1, ..base }, KernelError::InvalidCombination),
        (Present { tail: 1, ..base }, KernelError::InvalidCombination),
        (base.fenced(unallocated, 1), KernelError::InvalidCombination),
        (
            Present {
                fence: !192,
                ..base
            },
            KernelError::InvalidCombination,
        ),
        (
            Present { fence: 1, ..base },
            KernelError::InvalidCombination,
        ),
    ];
    for (present, error) in cases {
        assert_eq!(
            rig.send(&present.words()),
            refused(PRESENT_CLIENT_BUFFER, error),
            "{:x?}",
            present.words()
        );
    }
    assert!(!rig.shared.lock().presents().has_work());
}

/// The accepted side of 229's edges: the last slot, an RGB10A2 image, a fenced present on an
/// allocated syncpoint, and the slot in the word's low byte.
#[test]
fn the_accepted_side_of_each_present_edge() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let free = rig.shared.syncpoints().allocate().unwrap();
    let base = Present::of(image, 0);
    let ten = rig.image_with(&preference(PRESENT_BLOCK_HEIGHT, PRESENT_FORMATS[1]), true);
    assert_eq!(
        rig.present(Present {
            slot: u32::from(SLOTS) - 1,
            ..base
        }),
        ALL_SLOTS & !slot_bit(SLOTS - 1)
    );
    assert_eq!(
        rig.present(
            Present {
                image: ten,
                slot: 0x100 + 3,
                ..base
            }
            .fenced(free, 1)
        ),
        ALL_SLOTS & !slot_bit(SLOTS - 1) & !slot_bit(3)
    );
}

/// A client with no layer (no 212 yet) or not bound to the display, or a slot a frame holds,
/// is `InvalidState`.
#[test]
fn a_present_needs_a_layer_a_display_and_a_free_slot() {
    let mut rig = Rig::new();
    rig.bound_client(TOKEN, RETIRE, AVAILABLE);
    let image = rig.image();
    let base = Present::of(image, 0);
    assert_eq!(
        rig.send(&base.words()),
        refused(PRESENT_CLIENT_BUFFER, KernelError::InvalidState)
    );
    rig.send(&client_only(0xE4, TOKEN));
    rig.ok(&client_only(0x10C, TOKEN));
    assert_eq!(
        rig.send(&base.words()),
        refused(PRESENT_CLIENT_BUFFER, KernelError::InvalidState)
    );
    rig.ok(&bind(TOKEN, DEFAULT_DISPLAY));
    rig.present(base);
    // Queued, then shown: the slot is held either way until it is released.
    assert_eq!(
        rig.send(&base.words()),
        refused(PRESENT_CLIENT_BUFFER, KernelError::InvalidState)
    );
    rig.next_tick();
    assert_eq!(
        rig.send(&base.words()),
        refused(PRESENT_CLIENT_BUFFER, KernelError::InvalidState)
    );
    rig.present(Present::of(image, 1));
    rig.next_tick();
    rig.present(base);
}

/// The checks come in PR's order: with two faults at once, the earlier check answers.
#[test]
fn the_present_checks_come_in_order() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let base = Present::of(image, 0);
    let cases = [
        // An unknown client and a slot past the last: the client.
        (
            Present {
                token: TOKEN + 1,
                slot: 99,
                ..base
            },
            KernelError::NotFound,
        ),
        // A slot past the last and an unknown image: the slot.
        (
            Present {
                slot: 99,
                image: 0,
                ..base
            },
            KernelError::InvalidState,
        ),
        // An unknown image and an offset: the image.
        (
            Present {
                image: 0,
                offset: 1,
                ..base
            },
            KernelError::NotFound,
        ),
        // An offset and a non-zero word 23: the offset.
        (
            Present {
                offset: 1,
                high: 1,
                ..base
            },
            KernelError::NotImplemented,
        ),
    ];
    for (present, error) in cases {
        assert_eq!(
            rig.send(&present.words()),
            refused(PRESENT_CLIENT_BUFFER, error)
        );
    }
}

/// A 229 one word short is the framework's refusal and changes nothing; a 228 one word short
/// too.
#[test]
fn a_short_present_or_cancel_is_refused() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let mut words = Present::of(image, 0).words();
    words.pop();
    words[1] -= 1;
    assert_eq!(
        rig.send(&words),
        result_only(PRESENT_CLIENT_BUFFER, INVALID_CMIF_REQUEST)
    );
    assert!(!rig.shared.lock().presents().has_work());
    assert_eq!(
        rig.send(&[tipc_type(CANCEL_CLIENT_BUFFER), 2, 0, 0]),
        result_only(CANCEL_CLIENT_BUFFER, INVALID_CMIF_REQUEST)
    );
}

/// 228 gives a slot back with no image; a slot past the last is refused before the client, and
/// an unknown client is `NotFound`; the slot is the word's low byte.
#[test]
fn a_cancel_gives_a_slot_back() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    rig.present(Present::of(image, 3));
    assert_eq!(
        rig.send(&cancel(TOKEN + 1, u32::from(SLOTS))),
        refused(CANCEL_CLIENT_BUFFER, KernelError::InvalidCombination)
    );
    assert_eq!(
        rig.send(&cancel(TOKEN + 1, 3)),
        refused(CANCEL_CLIENT_BUFFER, KernelError::NotFound)
    );
    assert_eq!(
        rig.send(&cancel(0, 3)),
        refused(CANCEL_CLIENT_BUFFER, KernelError::NotFound)
    );
    assert_eq!(
        rig.send(&cancel(TOKEN, 0x200 + 3)),
        slots_reply(CANCEL_CLIENT_BUFFER, ALL_SLOTS)
    );
    // Its frame still holds the slot (PR: cancel leaves `queued[slot]`): a present of it is
    // refused until the frame is released.
    assert_eq!(
        rig.send(&Present::of(image, 3).words()),
        refused(PRESENT_CLIENT_BUFFER, KernelError::InvalidState)
    );
    assert_eq!(
        rig.send(&cancel(TOKEN, u32::from(SLOTS) - 1)),
        slots_reply(CANCEL_CLIENT_BUFFER, ALL_SLOTS)
    );
}

/// 201 refuses a token another session's client holds, as the table's key (module docs,
/// "Where this differs"); once that client is unregistered, the token is free again.
#[test]
fn a_token_is_one_client_across_sessions() {
    let mut rig = Rig::new();
    rig.ok(&register_client(TOKEN, RETIRE, AVAILABLE));
    let other = rig.kernel.new_object();
    rig.server.accept(other).unwrap();
    assert_eq!(
        rig.send_on(other, &register_client(TOKEN, RETIRE + 1, AVAILABLE + 1)),
        refused(REGISTER_CLIENT_STATE, KernelError::InvalidState)
    );
    rig.ok(&client_only(0xDA, TOKEN));
    assert_eq!(
        rig.send_on(other, &register_client(TOKEN, RETIRE + 1, AVAILABLE + 1)),
        success(REGISTER_CLIENT_STATE, &[], &[])
    );
}

/// 202 drops the client's presents: its frames are gone, and the clock stops once nothing else
/// keeps it running.
#[test]
fn unregistering_a_client_drops_its_presents() {
    let mut rig = Rig::new();
    rig.presenting_client();
    let image = rig.image();
    let id = rig.shared.syncpoints().allocate().unwrap();
    rig.present(Present::of(image, 0).fenced(id, 5));
    assert_eq!(rig.shared.lock().watchers(id), 1);
    rig.ok(&client_only(0xDA, TOKEN));
    assert_eq!(rig.shared.lock().watchers(id), 0);
    assert!(rig.shared.lock().presents().client(TOKEN).is_none());
    assert_eq!(rig.next_tick(), [Note::VsyncTick { tick: 1 }]);
    assert_eq!(rig.shared.vsync_deadline(), None);
}

/// Takes the held reply the server completed, delivered into a fresh buffer.
fn delivered(rig: &mut Rig) -> Option<Vec<u8>> {
    let completed = rig.server.take_completed()?;
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    rig.server
        .deliver(&mut rig.kernel, completed, &mut buffer)
        .unwrap();
    Some(buffer)
}

/// The fence wait fallback: a present's reply is held, and completes with every slot free on
/// the first tick after the previous frame's fence (the tick that latches the previous frame),
/// not before; with no previous frame waiting, on the next tick.
#[test]
fn the_fallback_s_reply_completes_on_the_first_tick_after_the_previous_frame_s_fence() {
    let mut rig = Rig::paced(Pacing::FenceWait);
    rig.presenting_client();
    let image = rig.image();
    let id = rig.shared.syncpoints().allocate().unwrap();
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    let first = Present::of(image, 0).fenced(id, 1);
    let bytes = le(&first.words());
    buffer[..bytes.len()].copy_from_slice(&bytes);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    assert!(matches!(handled, Ok(Handled::Deferred(_))));
    assert!(delivered(&mut rig).is_none());
    // No frame before it: the next tick completes it, its own frame still waiting.
    rig.next_tick();
    assert_eq!(
        delivered(&mut rig),
        Some(slots_reply(PRESENT_CLIENT_BUFFER, ALL_SLOTS))
    );
    // The second waits behind the first, whose fence is not reached.
    let second = Present::of(image, 0).fenced(id, 2);
    let bytes = le(&second.words());
    buffer[..bytes.len()].copy_from_slice(&bytes);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    assert!(matches!(handled, Ok(Handled::Deferred(_))));
    for n in 2..=3 {
        assert_eq!(rig.next_tick(), [Note::VsyncTick { tick: n }]);
        assert!(delivered(&mut rig).is_none());
    }
    rig.drain(id, 1);
    assert_eq!(
        rig.next_tick(),
        [
            Note::VsyncTick { tick: 4 },
            Note::FrameLatched { slot: 0, image }
        ]
    );
    assert_eq!(
        delivered(&mut rig),
        Some(slots_reply(PRESENT_CLIENT_BUFFER, ALL_SLOTS))
    );
    // Its own frame, fenced at 2, latches once 2 is drained; no slot was taken, so none comes
    // back and nothing is raised.
    rig.drain(id, 2);
    assert_eq!(
        rig.next_tick(),
        [
            Note::VsyncTick { tick: 5 },
            Note::FrameLatched { slot: 0, image },
            Note::BufferReleased { slot: 0, image },
        ]
    );
    assert_eq!(
        rig.send(&client_only(0xE4, TOKEN)),
        slots_reply_212(ALL_SLOTS)
    );
}

/// The held reply saves with its token id while the server holds the request, and restores,
/// with the server saved beside it, to a reply the restored server completes on the restored
/// table's next tick.
#[test]
fn a_held_reply_is_saved_and_restored() {
    let mut rig = Rig::paced(Pacing::FenceWait);
    rig.presenting_client();
    let image = rig.image();
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    let bytes = le(&Present::of(image, 0).words());
    buffer[..bytes.len()].copy_from_slice(&bytes);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    let Ok(Handled::Deferred(token)) = handled else {
        unreachable!("the fallback holds its reply");
    };
    let held = rig
        .shared
        .lock()
        .presents()
        .client(TOKEN)
        .map(|client| client.waiting().map(crate::shared::present::Waiting::token));
    assert_eq!(held, Some(Some(Some(token))));
    let table = rig.shared.save_presents(&rig.server).unwrap();
    let server = rig.server.save().unwrap();
    let env: Arc<dyn Env> = Arc::new(LiveEnv::new(FixedClock::new(START, 7), 0));
    let mut restored = vi2_server_paced(
        UnknownPolicy::Strict,
        Arc::clone(&rig.shared),
        Pacing::FenceWait,
    )
    .with_service_id(VI2)
    .with_env(env, Arc::new(NoHalt));
    restored.restore(&server).unwrap();
    rig.shared.restore_presents(&table, &restored).unwrap();
    assert_eq!(rig.shared.save_presents(&restored).unwrap(), table);
    rig.server = restored;
    assert!(delivered(&mut rig).is_none());
    rig.next_tick();
    assert_eq!(
        delivered(&mut rig),
        Some(slots_reply(PRESENT_CLIENT_BUFFER, ALL_SLOTS))
    );
}

/// A rig whose client [`TOKEN`] has had slot 0 released by a tick, with `notify` saying
/// whether 11 made the notification event before the client registered, after it, or not at
/// all: the rig and the image.
fn released_slot_zero(notify: Notify) -> (Rig, u32) {
    let mut rig = Rig::new();
    if notify == Notify::Before {
        rig.notification_event();
    }
    rig.presenting_client();
    if notify == Notify::After {
        rig.notification_event();
    }
    let image = rig.image();
    rig.present(Present::of(image, 0));
    rig.next_tick();
    rig.present(Present::of(image, 1));
    rig.next_tick();
    (rig, image)
}

/// When 11 makes the notification event, relative to the client's 201.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Notify {
    Before,
    After,
    Never,
}

/// Each command that answers from a client's free slots or the notification bitmap takes the
/// tick's release first, on its own: 212, 228 and 229 answer slot 0 free, 21 hands out the
/// "buffer available" bit, and an 11 that makes the event after the release signals it.
#[test]
fn each_refreshing_command_takes_the_tick_s_release() {
    let freed = ALL_SLOTS & !slot_bit(1);
    let (mut rig, _) = released_slot_zero(Notify::Never);
    assert_eq!(rig.send(&client_only(0xE4, TOKEN)), slots_reply_212(freed));
    let (mut rig, _) = released_slot_zero(Notify::Never);
    assert_eq!(
        rig.send(&cancel(TOKEN, 14)),
        slots_reply(CANCEL_CLIENT_BUFFER, freed)
    );
    let (mut rig, image) = released_slot_zero(Notify::Never);
    assert_eq!(rig.present(Present::of(image, 2)), freed & !slot_bit(2));
    let (mut rig, _) = released_slot_zero(Notify::Never);
    rig.ok(&get_notifications(128));
    assert_eq!(
        rig.kernel.memory(BUFFER, 4).unwrap(),
        (1_u32 << AVAILABLE).to_le_bytes()
    );
    let (mut rig, _) = released_slot_zero(Notify::Never);
    let event = rig.notification_event();
    assert_eq!(rig.signals(event), 1);
}

/// The tick signals the session's notification event whether 11 made it before the client's
/// 201 or after it.
#[test]
fn the_release_signals_the_event_made_before_or_after_the_client() {
    for notify in [Notify::Before, Notify::After] {
        let (mut rig, _) = released_slot_zero(notify);
        let event = rig.notification_event();
        // The tick's signal, then 11's own while the bit is raised.
        assert_eq!(rig.signals(event), 2);
    }
}

/// A restore refuses two clients holding one pending reply's token, at the second, though the
/// token is pending: each held request is reissued once. One client holding it is taken.
#[test]
fn a_restore_refuses_one_held_reply_named_twice() {
    use crate::shared::present::{
        PresentSnapshotError, SNAPSHOT_MAGIC, SNAPSHOT_VERSION, SavedClient, SavedFrame,
        SavedWaiting,
    };
    let mut rig = Rig::paced(Pacing::FenceWait);
    rig.presenting_client();
    let image = rig.image();
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    let bytes = le(&Present::of(image, 0).words());
    buffer[..bytes.len()].copy_from_slice(&bytes);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    let Ok(Handled::Deferred(token)) = handled else {
        unreachable!("the fallback holds its reply");
    };
    let frame = SavedFrame {
        slot: 0,
        image,
        address: IMAGE_AT,
        width: 16,
        height: 16,
        pitch: 1,
        block_height_log2: PRESENT_BLOCK_HEIGHT,
        format: PRESENT_FORMATS[0],
        shown: [0, 0, 16, 16],
        layer: FIRST_LAYER,
        display: DEFAULT_DISPLAY,
        fence: None,
        reached: true,
        holds_slot: false,
    };
    let client = |key: u64| SavedClient {
        token: key,
        available_bit: AVAILABLE,
        event: None,
        queued: vec![frame.clone()],
        shown: None,
        released: 0,
        raised: false,
        waiting: Some(SavedWaiting {
            token: Some(token),
            ahead: 0,
        }),
    };
    let snapshot = |clients: Vec<SavedClient>| {
        let mut bytes = SNAPSHOT_MAGIC.to_vec();
        bytes.extend(postcard::to_allocvec(&SNAPSHOT_VERSION).unwrap());
        bytes.extend(postcard::to_allocvec(&clients).unwrap());
        bytes
    };
    let twice = snapshot(vec![client(TOKEN), client(TOKEN + 1)]);
    assert_eq!(
        rig.shared.restore_presents(&twice, &rig.server),
        Err(PresentSnapshotError::DuplicateToken {
            index: 1,
            token: token.raw()
        })
    );
    let once = snapshot(vec![client(TOKEN + 1)]);
    assert_eq!(rig.shared.restore_presents(&once, &rig.server), Ok(()));
}
