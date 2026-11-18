//! The video context and the decoder service driven through raw TIPC messages laid out by hand,
//! word by word, from the headers PR checks (`native_video_decoder.cpp` and
//! `graphics_supervisor_protocol.h` @ e5fb3b97: `CreateDecoder` `0x11, 0`, `Submit` `0x10007d,
//! 3`, `RegisterImage` `0x147, 4`, and so on, written here as they are), over the services
//! framework's fake kernel.

use std::sync::Arc;

use starquake_gpu::interface::{Completions, Fence, SyncpointId};
use starquake_gpu::syncpoint::{FIRST_ALLOCATABLE, Syncpoints};
use starquake_ipc::BUFFER_SIZE;
use starquake_kernel::handle::Handle;
use starquake_kernel::slab::SlabId;
use starquake_kernel::memory::testing::device_share_token;
use starquake_kernel::result::KernelError;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel, Server, UnknownPolicy};
use starquake_trace::ServiceId;

use super::*;
use crate::completion::CompletionQueue;
use crate::completion::testing::process;
use crate::native::{BIND_PROCESS_MEMORY, BindRequest};
use crate::testing::{STALE, copied, halves, le, result_only, success, tipc_type};

/// The number the tests' video server is known by.
const VIDEO: ServiceId = ServiceId(0x0858);
/// The number gsv's notes are recorded under.
const GSV: ServiceId = ServiceId(0x0859);
/// Where the pools' memory is.
const MEMORY: u64 = 0x3000_0000;
/// One page.
const PAGE: u64 = 0x1000;
/// How many images the rig's pool holds: more than a decoder registers.
const POOL: u32 = 12;
/// The A descriptor's count bit in a TIPC header word 0 (`hipc.rs`: `num_a: (w0 >> 20) & 0xF`).
const ONE_A: u32 = 1 << 20;
/// Where the rig's A buffers say they are.
const BUFFER: u32 = 0x4000_0000;

/// A video server over gsv's shared state with a pool of [`POOL`] images, one context session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    context: ObjectId,
    shared: Arc<Shared>,
    /// The pool's first image id.
    first: u32,
    /// The decoder services made, in order.
    decoders: Vec<ObjectId>,
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(VIDEO);
        let context = kernel.new_object();
        let queue = Arc::new(CompletionQueue::new());
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV));
        let first = {
            let mut locked = shared.lock();
            let native = locked.native();
            let size = u64::from(POOL) * PAGE;
            let memory = native.create_memory(0, size).unwrap();
            let request = BindRequest {
                id: memory,
                address: MEMORY,
                size,
                flags: BIND_PROCESS_MEMORY,
                offset: 0,
            };
            let admission = native.check_bind(&request).unwrap();
            admission.bind(device_share_token(process(), MEMORY, size));
            let mut preference = [0; layout::PREFERENCE_WORDS];
            preference[layout::WORD_CODE] = layout::CODE_NV12;
            preference[layout::WORD_EXTENT] = 1 | (1 << 16);
            native.create_pool(&preference, POOL, memory).unwrap()
        };
        let mut server = video_server(Arc::clone(&shared), UnknownPolicy::Strict)
            .with_service_id(VIDEO);
        server.accept(context).unwrap();
        Self {
            kernel,
            server,
            context,
            shared,
            first,
            decoders: Vec::new(),
        }
    }

    fn send_on(&mut self, session: ObjectId, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// A new decoder service from the context's 1 (`0x11, 0`).
    fn decoder(&mut self) -> ObjectId {
        let reply = self.send_on(self.context, &[0x11, 0]);
        let copies = copied(&reply);
        assert_eq!(reply, success(CREATE_DECODER, &copies, &[]));
        assert_eq!(copies.len(), 1);
        let object = self
            .kernel
            .resolve_handle(Handle::from_raw(copies[0]))
            .unwrap();
        assert_eq!(self.kernel.owner(object), Some(VIDEO));
        self.decoders.push(object);
        object
    }

    /// The decoder service's state, read from the server's snapshot: each session is saved as
    /// its interface's name and its state, each with its length (`SavedTarget::Interface` in
    /// the framework's `targets.rs`), in the order the sessions were made.
    fn state(&self, decoder: ObjectId) -> DecoderService {
        let saved = self.server.save().unwrap();
        let name = b"DecoderService";
        let mut states = Vec::new();
        let mut at = 0;
        while at + name.len() < saved.len() {
            if saved[at] == 14 && &saved[at + 1..at + 1 + name.len()] == name {
                let mut cursor = at + 1 + name.len();
                let (mut length, mut shift) = (0_usize, 0);
                loop {
                    let byte = saved[cursor];
                    cursor += 1;
                    length |= usize::from(byte & 0x7F) << shift;
                    shift += 7;
                    if byte & 0x80 == 0 {
                        break;
                    }
                }
                let state: DecoderService =
                    postcard::from_bytes(&saved[cursor..cursor + length]).unwrap();
                states.push(state);
                at = cursor + length;
            } else {
                at += 1;
            }
        }
        let index = self.decoders.iter().position(|&d| d == decoder).unwrap();
        states.swap_remove(index)
    }

    fn syncpoints(&self) -> Arc<Syncpoints> {
        Arc::clone(self.shared.syncpoints())
    }
}

/// 311: `0x147, 4`, image, three zeros.
fn register(image: u32) -> Vec<u32> {
    vec![0x147, 4, image, 0, 0, 0]
}

/// 314: `0x14a, 8`, image, then seven words NX2 reads as a release fence, not read here.
fn release(image: u32) -> Vec<u32> {
    vec![0x14A, 8, image, 0, !5, 0, 9, 0, 0, 0]
}

/// 109: one A descriptor of `bytes` bytes, timestamp, end of stream.
fn submit(bytes: u32, timestamp: u64, end: u32) -> Vec<u32> {
    let mut words = vec![0x7D | ONE_A, 3, bytes, BUFFER, 0];
    words.extend(halves(timestamp));
    words.push(end);
    words
}

/// 101: one A descriptor of the codec's name (10 bytes, `"video/vp9\0"`), width, height,
/// frame rate, input capacity, four zeros.
fn configure(width: u32, height: u32) -> Vec<u32> {
    vec![0x75 | ONE_A, 8, 10, BUFFER, 0, width, height, 30, 0x10_0000, 0, 0, 0, 0]
}

/// The eight data words of a native sync handle reply.
fn handle_words(id: SyncpointId, value: u32) -> Vec<u32> {
    vec![counter_id(id), 0, value, 0, 0, 0, 0, 0]
}

/// The syncpoint a fresh rig's first decoder allocates.
fn first_syncpoint() -> SyncpointId {
    SyncpointId::new(FIRST_ALLOCATABLE).unwrap()
}

/// The value word of a handle reply.
fn value_of(reply: &[u8]) -> u32 {
    u32::from_le_bytes(reply[20..24].try_into().unwrap())
}

/// 213's reply for an output.
fn output_reply(image: u32, width: u32, height: u32, ns: u64, end: bool) -> Vec<u8> {
    let mut words = vec![image, width, height, width, height, 0];
    words.extend(halves(ns));
    words.push(u32::from(end));
    words.extend([0; 7]);
    success(GET_OUTPUT_PROPERTIES, &[], &words)
}

/// The references' constants, written out as the cited sources write them.
#[test]
fn the_constants_are_the_references_values() {
    assert_eq!(
        [
            CREATE_DECODER,
            GET_SUBMISSION_EVENT,
            START,
            FLUSH,
            CONFIGURE,
            SUBMIT,
            FINALIZE,
            INITIALIZE_200,
            INITIALIZE,
            STOP,
            UNREGISTER_IMAGE,
            GET_OUTPUT_PROPERTIES,
            REGISTER_IMAGE,
            RELEASE_IMAGE,
        ],
        [1, 3, 6, 8, 101, 109, 115, 200, 201, 207, 212, 213, 311, 314]
    );
    // PR's header words are the ids' TIPC types, with one A descriptor for 101 and 109.
    for (id, header) in [
        (CREATE_DECODER, 0x11),
        (GET_SUBMISSION_EVENT, 0x13),
        (START, 0x16),
        (FLUSH, 0x18),
        (FINALIZE, 0x83),
        (INITIALIZE_200, 0xD8),
        (INITIALIZE, 0xD9),
        (STOP, 0xDF),
        (UNREGISTER_IMAGE, 0xE4),
        (GET_OUTPUT_PROPERTIES, 0xE5),
        (REGISTER_IMAGE, 0x147),
        (RELEASE_IMAGE, 0x14A),
    ] {
        assert_eq!(tipc_type(id), header);
    }
    assert_eq!(tipc_type(CONFIGURE) | ONE_A, 0x0010_0075);
    assert_eq!(tipc_type(SUBMIT) | ONE_A, 0x0010_007D);
    assert_eq!(MAX_REGISTERED, 8);
    assert_eq!(MAX_PENDING, 3);
    assert_eq!(MAX_HANDOUTS, 16);
    assert_eq!(MAX_OUTPUTS, 8);
    assert_eq!(VIDEO_QUEUE_FULL.raw(), 0x001C_266F);
    assert_eq!(VIDEO_NO_OUTPUT.raw(), 0x001C_286F);
    assert_eq!(OUTPUT_WORDS, 16);
    assert_eq!(NS_PER_US, 1000);
    assert_eq!(SYNC_HANDLE_MODE, 0);
    assert_eq!(counter_id(first_syncpoint()), !1);
    assert_eq!(
        crate::syncpt::syncpoint_of(counter_id(first_syncpoint())),
        Some(first_syncpoint())
    );
}

#[test]
fn a_fresh_decoder_holds_nothing() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    assert_eq!(rig.state(decoder), DecoderService::new());
    let fresh = DecoderService::new();
    assert_eq!(fresh.configuration(), None);
    assert_eq!(fresh.event(), None);
    assert_eq!(fresh.syncpoint(), None);
    let none: &[u32] = &[];
    assert_eq!(fresh.images(), none);
    assert_eq!(fresh.handed(), 0);
    assert_eq!(fresh.handouts().count(), 0);
    assert_eq!(fresh.pending().count(), 0);
    assert_eq!(fresh.outputs().count(), 0);
}

#[test]
fn register_answers_a_sync_handle() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let syncpoints = rig.syncpoints();
    let id = first_syncpoint();
    // The counter before the allocation is the base: move it first.
    syncpoints.increment_by(id, 0x40);
    let reply = rig.send_on(decoder, &register(rig.first));
    assert_eq!(reply, success(REGISTER_IMAGE, &[], &handle_words(id, 0x41)));
    assert!(syncpoints.is_allocated(id));
    let reply = rig.send_on(decoder, &register(rig.first + 1));
    assert_eq!(reply, success(REGISTER_IMAGE, &[], &handle_words(id, 0x42)));
    // 314 answers the same shape, with the next value.
    let reply = rig.send_on(decoder, &release(rig.first));
    assert_eq!(reply, success(RELEASE_IMAGE, &[], &handle_words(id, 0x43)));
}

/// The spec's test: sync handles count frames in order. Each handout's value is reached by
/// its own frame, exactly, and not before.
#[test]
fn sync_handles_count_frames_in_order() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let syncpoints = rig.syncpoints();
    let images = [rig.first + 2, rig.first, rig.first + 1];
    let mut values = Vec::new();
    for image in images {
        values.push(value_of(&rig.send_on(decoder, &register(image))));
    }
    let id = rig.state(decoder).syncpoint().unwrap();
    let base = syncpoints.read(id);
    assert_eq!(values, [base + 1, base + 2, base + 3]);
    for (frame, (&image, &value)) in images.iter().zip(&values).enumerate() {
        let fence = Fence { id, value };
        assert!(!syncpoints.reached(fence), "frame {frame} reached early");
        let timestamp = 1000 + u64::try_from(frame).unwrap();
        let reply = rig.send_on(decoder, &submit(0x80, timestamp, 0));
        assert_eq!(reply, success(SUBMIT, &[], &halves(0x80)));
        assert!(syncpoints.reached(fence), "frame {frame}");
        assert_eq!(syncpoints.read(id), value, "frame {frame}");
        // Its output names its image.
        let output = rig.send_on(decoder, &[0xE5, 0]);
        assert_eq!(output, output_reply(image, 0, 0, timestamp * 1000, false));
    }
    // A release hands an image out again with the next value; a frame submitted before it
    // waits, then takes it.
    rig.send_on(decoder, &submit(0x80, 7, 0));
    assert_eq!(syncpoints.read(id), base + 3);
    let again = value_of(&rig.send_on(decoder, &release(images[1])));
    assert_eq!(again, base + 4);
    assert_eq!(syncpoints.read(id), base + 4);
}

#[test]
fn frames_wait_for_images_and_images_for_frames() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // Frames first: they wait.
    rig.send_on(decoder, &submit(1, 10, 0));
    rig.send_on(decoder, &submit(1, 11, 0));
    assert_eq!(rig.state(decoder).pending().count(), 2);
    // An image: the oldest frame takes it at once.
    rig.send_on(decoder, &register(rig.first));
    let state = rig.state(decoder);
    assert_eq!(state.pending().copied().collect::<Vec<_>>(), [Input {
        timestamp: 11,
        end: false
    }]);
    assert_eq!(state.outputs().next().unwrap().timestamp, 10);
    assert_eq!(state.handouts().count(), 0);
}

#[test]
fn flush_keeps_images_already_handed_out() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let syncpoints = rig.syncpoints();
    // One frame finished into the first image, then three frames waiting: frames wait only
    // while no handout does.
    let first = value_of(&rig.send_on(decoder, &register(rig.first)));
    rig.send_on(decoder, &submit(1, 1, 0));
    let id = rig.state(decoder).syncpoint().unwrap();
    assert_eq!(syncpoints.read(id), first);
    for timestamp in 2..5 {
        rig.send_on(decoder, &submit(1, timestamp, 0));
    }
    assert_eq!(rig.state(decoder).pending().count(), 3);
    // The flush drops the waiting frames, and keeps the finished output.
    assert_eq!(rig.send_on(decoder, &[0x18, 0]), success(FLUSH, &[], &[]));
    let state = rig.state(decoder);
    assert_eq!(state.pending().count(), 0);
    assert_eq!(state.outputs().count(), 1);
    // Images handed out before a flush keep their place and value: hand out two, flush, and
    // the next frames fill them in order with the values given.
    let second = value_of(&rig.send_on(decoder, &register(rig.first + 1)));
    let third = value_of(&rig.send_on(decoder, &register(rig.first + 2)));
    assert_eq!((second, third), (first + 1, first + 2));
    let before = rig.state(decoder).handouts().copied().collect::<Vec<_>>();
    assert_eq!(rig.send_on(decoder, &[0x18, 0]), success(FLUSH, &[], &[]));
    assert_eq!(rig.state(decoder).handouts().copied().collect::<Vec<_>>(), before);
    rig.send_on(decoder, &submit(1, 5, 0));
    assert_eq!(syncpoints.read(id), second);
    rig.send_on(decoder, &submit(1, 6, 0));
    assert_eq!(syncpoints.read(id), third);
    // The finished outputs, in order: the one from before the flushes, then these two.
    let images: Vec<u32> = rig.state(decoder).outputs().map(|output| output.image).collect();
    assert_eq!(images, [rig.first, rig.first + 1, rig.first + 2]);
}

#[test]
fn stop_is_flush() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    rig.send_on(decoder, &register(rig.first));
    rig.send_on(decoder, &submit(1, 1, 0));
    rig.send_on(decoder, &submit(1, 2, 0));
    let handed = rig.send_on(decoder, &register(rig.first + 1));
    // The second frame took the second image; a third waits.
    rig.send_on(decoder, &submit(1, 3, 0));
    let before = rig.state(decoder);
    assert_eq!(before.pending().count(), 1);
    assert_eq!(rig.send_on(decoder, &[0xDF, 0]), success(STOP, &[], &[]));
    let after = rig.state(decoder);
    assert_eq!(after.pending().count(), 0);
    assert_eq!(
        after.outputs().copied().collect::<Vec<_>>(),
        before.outputs().copied().collect::<Vec<_>>()
    );
    assert_eq!(value_of(&handed), after.value(2));
}

#[test]
fn an_unregistered_image_s_handouts_are_skipped() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let syncpoints = rig.syncpoints();
    let a = value_of(&rig.send_on(decoder, &register(rig.first)));
    let b = value_of(&rig.send_on(decoder, &register(rig.first + 1)));
    let c = value_of(&rig.send_on(decoder, &register(rig.first + 2)));
    assert_eq!(
        rig.send_on(decoder, &[0xE4, 8, rig.first + 1, 0, 0, 0, 0, 0, 0, 0]),
        success(UNREGISTER_IMAGE, &[], &[])
    );
    let id = rig.state(decoder).syncpoint().unwrap();
    rig.send_on(decoder, &submit(1, 1, 0));
    assert_eq!(syncpoints.read(id), a);
    // The second frame finishes the third handout: the counter passes the gap to its value.
    rig.send_on(decoder, &submit(1, 2, 0));
    assert_eq!(syncpoints.read(id), c);
    assert!(syncpoints.reached(Fence { id, value: b }));
    let images: Vec<u32> = rig.state(decoder).outputs().map(|output| output.image).collect();
    assert_eq!(images, [rig.first, rig.first + 2]);
}

#[test]
fn unregister_refuses_an_unknown_image() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    rig.send_on(decoder, &register(rig.first));
    assert_eq!(
        rig.send_on(decoder, &[0xE4, 8, rig.first + 1, 0, 0, 0, 0, 0, 0, 0]),
        result_only(UNREGISTER_IMAGE, KernelError::NotFound.code())
    );
    assert_eq!(rig.state(decoder).images(), [rig.first]);
    // 314 of an image not registered is NotFound too, and hands nothing out.
    assert_eq!(
        rig.send_on(decoder, &release(rig.first + 1)),
        result_only(RELEASE_IMAGE, KernelError::NotFound.code())
    );
    assert_eq!(rig.state(decoder).handouts().count(), 1);
}

#[test]
fn releasing_a_waiting_image_hands_it_out_again() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let syncpoints = rig.syncpoints();
    let first = value_of(&rig.send_on(decoder, &register(rig.first)));
    let second = value_of(&rig.send_on(decoder, &release(rig.first)));
    assert_eq!(second, first + 1);
    assert_eq!(rig.state(decoder).handouts().count(), 2);
    rig.send_on(decoder, &submit(1, 1, 0));
    rig.send_on(decoder, &submit(1, 2, 0));
    let id = rig.state(decoder).syncpoint().unwrap();
    assert_eq!(syncpoints.read(id), second);
    let images: Vec<u32> = rig.state(decoder).outputs().map(|output| output.image).collect();
    assert_eq!(images, [rig.first, rig.first]);
}

#[test]
fn register_checks_in_pr_s_order() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // An image no pool holds: NotFound; and image 0.
    for image in [rig.first + POOL, 0] {
        assert_eq!(
            rig.send_on(decoder, &register(image)),
            result_only(REGISTER_IMAGE, KernelError::NotFound.code())
        );
    }
    // Nothing allocated by a refusal.
    assert!(!rig.syncpoints().is_allocated(first_syncpoint()));
    for index in 0..u32::try_from(MAX_REGISTERED).unwrap() {
        rig.send_on(decoder, &register(rig.first + index));
    }
    // Registered already: InvalidState (with room left, a ninth image of the eight).
    let mut fresh = Rig::new();
    let other = fresh.decoder();
    fresh.send_on(other, &register(fresh.first));
    assert_eq!(
        fresh.send_on(other, &register(fresh.first)),
        result_only(REGISTER_IMAGE, KernelError::InvalidState.code())
    );
    // At the limit: OutOfResource, before the duplicate and before the pool lookup.
    for image in [rig.first, rig.first + POOL, rig.first + 8] {
        assert_eq!(
            rig.send_on(decoder, &register(image)),
            result_only(REGISTER_IMAGE, KernelError::OutOfResource.code()),
            "{image}"
        );
    }
    // A duplicate that is no pool image cannot be: a duplicate is registered, so in a pool.
    assert_eq!(rig.state(decoder).images().len(), MAX_REGISTERED);
}

#[test]
fn handouts_are_bounded_at_the_limit() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    for index in 0..u32::try_from(MAX_REGISTERED).unwrap() {
        rig.send_on(decoder, &register(rig.first + index));
    }
    for index in 0..u32::try_from(MAX_HANDOUTS - MAX_REGISTERED).unwrap() {
        let reply = rig.send_on(decoder, &release(rig.first + index));
        assert_eq!(&reply[8..12], &[0; 4], "release {index}");
    }
    assert_eq!(rig.state(decoder).handouts().count(), MAX_HANDOUTS);
    let before = rig.state(decoder);
    assert_eq!(
        rig.send_on(decoder, &release(rig.first)),
        result_only(RELEASE_IMAGE, KernelError::OutOfResource.code())
    );
    assert_eq!(rig.state(decoder), before);
    // One frame takes one handout: room for one more.
    rig.send_on(decoder, &submit(1, 1, 0));
    let reply = rig.send_on(decoder, &release(rig.first));
    assert_eq!(&reply[8..12], &[0; 4]);
}

#[test]
fn a_register_at_the_handout_limit_is_refused() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // Seven images, each handed out again: 14 handouts and room for one image (8 registered).
    for index in 0..7 {
        rig.send_on(decoder, &register(rig.first + index));
        rig.send_on(decoder, &release(rig.first + index));
    }
    // A 15th and a 16th: one release; the register then finds 16 handouts.
    rig.send_on(decoder, &release(rig.first));
    rig.send_on(decoder, &release(rig.first));
    assert_eq!(rig.state(decoder).handouts().count(), MAX_HANDOUTS);
    let before = rig.state(decoder);
    assert_eq!(
        rig.send_on(decoder, &register(rig.first + 7)),
        result_only(REGISTER_IMAGE, KernelError::OutOfResource.code())
    );
    assert_eq!(rig.state(decoder), before);
}

#[test]
fn a_full_input_queue_refuses_the_next_frame() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    for timestamp in 0..u64::try_from(MAX_PENDING).unwrap() {
        assert_eq!(
            rig.send_on(decoder, &submit(4, timestamp, 0)),
            success(SUBMIT, &[], &halves(4))
        );
    }
    assert_eq!(
        rig.send_on(decoder, &submit(4, 9, 0)),
        result_only(SUBMIT, VIDEO_QUEUE_FULL)
    );
    assert_eq!(rig.state(decoder).pending().count(), MAX_PENDING);
}

#[test]
fn outputs_are_bounded_and_a_frame_waits() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    for index in 0..u32::try_from(MAX_REGISTERED).unwrap() {
        rig.send_on(decoder, &register(rig.first + index));
        rig.send_on(decoder, &release(rig.first + index));
    }
    for timestamp in 0..u64::try_from(MAX_OUTPUTS).unwrap() {
        rig.send_on(decoder, &submit(1, timestamp, 0));
    }
    assert_eq!(rig.state(decoder).outputs().count(), MAX_OUTPUTS);
    // One more frame waits, though handouts wait too.
    rig.send_on(decoder, &submit(1, 99, 0));
    let state = rig.state(decoder);
    assert_eq!((state.pending().count(), state.outputs().count()), (1, MAX_OUTPUTS));
    // Reading one output makes room: the frame finishes.
    rig.send_on(decoder, &[0xE5, 0]);
    let state = rig.state(decoder);
    assert_eq!((state.pending().count(), state.outputs().count()), (0, MAX_OUTPUTS));
    assert_eq!(state.outputs().last().unwrap().timestamp, 99);
}

#[test]
fn no_output_answers_the_empty_ring_code() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    assert_eq!(
        rig.send_on(decoder, &[0xE5, 0]),
        result_only(GET_OUTPUT_PROPERTIES, VIDEO_NO_OUTPUT)
    );
}

#[test]
fn configure_takes_any_codec_and_size() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    assert_eq!(
        rig.send_on(decoder, &configure(640, 360)),
        success(CONFIGURE, &[], &[0, 0])
    );
    // Configured twice, sizes PR refuses (0, over 65535): taken.
    assert_eq!(
        rig.send_on(decoder, &configure(0x1_0000, 0)),
        success(CONFIGURE, &[], &[0, 0])
    );
    assert_eq!(
        rig.state(decoder).configuration(),
        Some(&Configuration {
            width: 0x1_0000,
            height: 0,
            frame_rate: 30,
            input_capacity: 0x10_0000,
        })
    );
    // The output carries the configured size as both sizes.
    rig.send_on(decoder, &configure(640, 360));
    rig.send_on(decoder, &register(rig.first));
    rig.send_on(decoder, &submit(1, 3, 1));
    assert_eq!(
        rig.send_on(decoder, &[0xE5, 0]),
        output_reply(rig.first, 640, 360, 3000, true)
    );
    // A configure without its last read word is refused, changing nothing.
    let mut short = configure(1, 1);
    short.truncate(8);
    short[1] = 3;
    assert_eq!(rig.send_on(decoder, &short), result_only(CONFIGURE, INVALID_CMIF_REQUEST));
    assert_eq!(rig.state(decoder).configuration().unwrap().width, 640);
}

#[test]
fn a_submit_answers_the_frame_s_bytes() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    assert_eq!(
        rig.send_on(decoder, &submit(0x1234, 0, 0)),
        success(SUBMIT, &[], &[0x1234, 0])
    );
    assert_eq!(
        rig.send_on(decoder, &submit(0, 0, 1)),
        success(SUBMIT, &[], &[0, 0])
    );
}

#[test]
fn any_end_byte_but_0_ends_the_stream() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // The byte is the low one: 0x100 is 0, 2 and 0xFF are not.
    for (word, end) in [(0x100, false), (2, true), (0xFF, true), (0, false)] {
        rig.send_on(decoder, &submit(1, 0, word));
        let state = rig.state(decoder);
        assert_eq!(state.pending().last().unwrap().end, end, "{word:#x}");
        rig.send_on(decoder, &[0x18, 0]);
    }
}

#[test]
fn the_timestamp_is_in_ns_wrapping() {
    let output = Output {
        image: 7,
        width: 1,
        height: 2,
        timestamp: u64::MAX / NS_PER_US + 1,
        end: false,
    };
    let ns = (u64::MAX / NS_PER_US + 1).wrapping_mul(NS_PER_US);
    let words = output.words();
    assert_eq!([words[6], words[7]], halves(ns));
    // Each field in its own word, the rest zero.
    let output = Output {
        image: 0x11,
        width: 0x22,
        height: 0x33,
        timestamp: 0x0000_0044_0000_0055,
        end: true,
    };
    let ns = 0x0000_0044_0000_0055_u64 * 1000;
    let mut expected = [0; OUTPUT_WORDS];
    expected[..9].copy_from_slice(&[0x11, 0x22, 0x33, 0x22, 0x33, 0, halves(ns)[0], halves(ns)[1], 1]);
    assert_eq!(output.words(), expected);
}

#[test]
fn the_ready_event_follows_the_input_queue() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // No event before 3 or 101: a submit signals nothing.
    rig.send_on(decoder, &submit(1, 0, 0));
    let reply = rig.send_on(decoder, &[0x13, 0]);
    let copies = copied(&reply);
    assert_eq!(reply, success(GET_SUBMISSION_EVENT, &copies, &[]));
    let event = rig.state(decoder).event().unwrap();
    assert_eq!(
        rig.kernel.resolve_handle(Handle::from_raw(copies[0])).unwrap(),
        event
    );
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    // Two more frames fill the queue: cleared.
    rig.send_on(decoder, &submit(1, 1, 0));
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    rig.send_on(decoder, &submit(1, 2, 0));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    // An image takes a frame: room again, signalled.
    rig.send_on(decoder, &register(rig.first));
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    // Full again, then a flush: signalled.
    rig.send_on(decoder, &submit(1, 3, 0));
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    rig.send_on(decoder, &[0x18, 0]);
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
    // 3 again hands out the same event.
    let again = copied(&rig.send_on(decoder, &[0x13, 0]));
    assert_eq!(
        rig.kernel.resolve_handle(Handle::from_raw(again[0])).unwrap(),
        event
    );
}

#[test]
fn configure_makes_the_ready_event() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    rig.send_on(decoder, &configure(2, 2));
    let event = rig.state(decoder).event().unwrap();
    assert_eq!(rig.kernel.is_signalled(event), Some(true));
}

#[test]
fn commands_are_taken_in_any_order() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    // Start with nothing configured or registered, a submit before 200 and 101, a register
    // after start, a start twice: each taken.
    for words in [
        vec![0x16, 0],
        submit(1, 0, 0),
        register(rig.first),
        vec![0x16, 0],
        vec![0xDF, 0],
        vec![0x18, 0],
    ] {
        let reply = rig.send_on(decoder, &words);
        assert_eq!(&reply[8..12], &[0; 4], "{words:x?}");
    }
}

#[test]
fn initialize_reads_its_words_and_checks_nothing() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let memory = rig.kernel.new_object();
    let handle = rig.kernel.insert_handle(memory).unwrap().raw();
    // 200: `0xd8, 0x80000005, 2`, the handle, size, codec 1, formats 3 and 3.
    let two_hundred = [0xD8, 0x8000_0005, 2, handle, 0x2000, 0, 1, 3, 3];
    assert_eq!(rig.send_on(decoder, &two_hundred), success(INITIALIZE_200, &[], &[]));
    // 201: `0xd9, 0x80000006, 2`, then the output mode; a codec PR refuses is taken.
    let two_oh_one = [0xD9, 0x8000_0006, 2, handle, 0, 0, 99, 99, 99, 99];
    assert_eq!(rig.send_on(decoder, &two_oh_one), success(INITIALIZE, &[], &[]));
    // 201 without its output mode: refused; 200 without a handle: refused.
    let short = [0xD9, 0x8000_0005, 2, handle, 0, 0, 1, 1, 1];
    assert_eq!(rig.send_on(decoder, &short), result_only(INITIALIZE, INVALID_CMIF_REQUEST));
    let unhandled = rig.send_on(decoder, &[0xD8, 5, 0, 0, 1, 3, 3]);
    assert_ne!(&unhandled[8..12], &[0; 4]);
    assert_eq!(rig.state(decoder), DecoderService::new());
}

#[test]
fn handle_values_wrap_at_32_bits() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    let id = first_syncpoint();
    rig.syncpoints().increment_by(id, u32::MAX - 1);
    let values: Vec<u32> = (0..3)
        .map(|index| value_of(&rig.send_on(decoder, &register(rig.first + index))))
        .collect();
    assert_eq!(values, [u32::MAX, 0, 1]);
    // The high word of each value is 0.
    let reply = rig.send_on(decoder, &release(rig.first));
    assert_eq!(&reply[24..28], &[0; 4]);
    // The frames bring the counter to each in turn.
    for value in values {
        rig.send_on(decoder, &submit(1, 0, 0));
        assert_eq!(rig.syncpoints().read(id), value);
    }
}

#[test]
fn finalize_frees_the_syncpoint_and_forgets() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    rig.send_on(decoder, &configure(2, 2));
    rig.send_on(decoder, &register(rig.first));
    rig.send_on(decoder, &submit(1, 0, 0));
    rig.send_on(decoder, &submit(1, 1, 0));
    let event = rig.state(decoder).event().unwrap();
    let id = first_syncpoint();
    assert_eq!(rig.syncpoints().read(id), 1);
    assert_eq!(rig.send_on(decoder, &[0x83, 0]), success(FINALIZE, &[], &[]));
    assert!(!rig.syncpoints().is_allocated(id));
    let mut expected = DecoderService::new();
    expected.event = Some(event);
    assert_eq!(rig.state(decoder), expected);
    assert_eq!(rig.kernel.is_signalled(event), Some(false));
    // Registering again allocates afresh, from the counter as it is.
    let value = value_of(&rig.send_on(decoder, &register(rig.first)));
    assert_eq!(value, 2);
    assert!(rig.syncpoints().is_allocated(id));
}

#[test]
fn two_decoders_count_on_their_own_syncpoints() {
    let mut rig = Rig::new();
    let one = rig.decoder();
    let two = rig.decoder();
    rig.send_on(one, &register(rig.first));
    rig.send_on(two, &register(rig.first + 1));
    let (a, b) = (
        rig.state(one).syncpoint().unwrap(),
        rig.state(two).syncpoint().unwrap(),
    );
    assert_ne!(a, b);
    rig.send_on(two, &submit(1, 0, 0));
    assert_eq!((rig.syncpoints().read(a), rig.syncpoints().read(b)), (0, 1));
}

// ---- What is saved ----

/// A decoder service with every field set, each value distinct.
fn golden() -> DecoderService {
    DecoderService {
        configuration: Some(Configuration {
            width: 0x11,
            height: 0x12,
            frame_rate: 0x13,
            input_capacity: 0x14,
        }),
        event: Some(ObjectId::from_parts(0x15, 0x16)),
        syncpoint: Some(0x17),
        base: 0x18,
        handed: 0x1C,
        reached: 0x19,
        images: vec![0x1A, 0x1B],
        handouts: VecDeque::from([
            Handout {
                image: 0x1B,
                number: 0x1B,
            },
            Handout {
                image: 0x1A,
                number: 0x1C,
            },
        ]),
        pending: VecDeque::new(),
        outputs: VecDeque::from([Output {
            image: 0x1D,
            width: 0x1E,
            height: 0x1F,
            timestamp: 0x20,
            end: true,
        }]),
    }
}

#[test]
fn the_session_layout_is_pinned() {
    let mut state = golden();
    let bytes = postcard::to_allocvec(&state).unwrap();
    let mut expected = vec![1, 0x11, 0x12, 0x13, 0x14]; // Some(configuration)
    expected.extend([1, 0x15, 0x16]); // Some(event): index, generation
    expected.extend([1, 0x17, 0x18, 0x1C, 0x19]); // syncpoint, base, handed, reached
    expected.extend([2, 0x1A, 0x1B]); // images
    expected.extend([2, 0x1B, 0x1B, 0x1A, 0x1C]); // handouts
    expected.push(0); // pending
    expected.extend([1, 0x1D, 0x1E, 0x1F, 0x20, 1]); // outputs
    assert_eq!(bytes, expected);
    assert_eq!(postcard::from_bytes::<DecoderService>(&bytes).unwrap(), state);
    // The pending frames: a waiting frame needs no handout waiting, or a full output queue.
    state.handouts.clear();
    state.pending.push_back(Input {
        timestamp: 0x21,
        end: true,
    });
    let bytes = postcard::to_allocvec(&state).unwrap();
    let tail = [1, 0x21, 1, 1, 0x1D, 0x1E, 0x1F, 0x20, 1];
    assert_eq!(&bytes[bytes.len() - tail.len()..], tail);
    assert_eq!(postcard::from_bytes::<DecoderService>(&bytes).unwrap(), state);
    // The context keeps nothing.
    assert_eq!(postcard::to_allocvec(&VideoContext).unwrap(), Vec::<u8>::new());
}

/// Reads `state` back through its bytes.
fn read(state: &DecoderService) -> Result<DecoderService, postcard::Error> {
    postcard::from_bytes::<DecoderService>(&postcard::to_allocvec(state).unwrap())
}

/// One state per relation the reading checks, each breaking that relation alone, with its
/// error.
fn unreachable_sessions() -> Vec<(DecoderService, DecoderServiceError)> {
    vec![
        (
            DecoderService {
                images: (1..=9).collect(),
                handouts: VecDeque::new(),
                ..golden()
            },
            DecoderServiceError::Images { found: 9 },
        ),
        (
            DecoderService {
                images: vec![0x1A, 0x1B, 0x1A],
                ..golden()
            },
            DecoderServiceError::Image { image: 0x1A },
        ),
        (
            DecoderService {
                images: vec![0, 0x1A, 0x1B],
                ..golden()
            },
            DecoderServiceError::Image { image: 0 },
        ),
        (
            DecoderService {
                syncpoint: Some(192),
                ..golden()
            },
            DecoderServiceError::Syncpoint,
        ),
        (
            DecoderService {
                syncpoint: None,
                ..golden()
            },
            DecoderServiceError::Syncpoint,
        ),
        (
            DecoderService {
                handed: 0,
                reached: 0,
                handouts: VecDeque::new(),
                ..golden()
            },
            DecoderServiceError::Syncpoint,
        ),
        (
            DecoderService {
                reached: 0x1D,
                handouts: VecDeque::new(),
                ..golden()
            },
            DecoderServiceError::Reached {
                reached: 0x1D,
                handed: 0x1C,
            },
        ),
        (
            DecoderService {
                pending: VecDeque::from([Input { timestamp: 0, end: false }; MAX_PENDING + 1]),
                handouts: VecDeque::new(),
                ..golden()
            },
            DecoderServiceError::Queues,
        ),
        (
            DecoderService {
                outputs: VecDeque::from(
                    [Output {
                        image: 1,
                        width: 0,
                        height: 0,
                        timestamp: 0,
                        end: false,
                    }; MAX_OUTPUTS + 1],
                ),
                ..golden()
            },
            DecoderServiceError::Queues,
        ),
        (
            DecoderService {
                handouts: VecDeque::from([Handout { image: 0x1A, number: 0x1C }; MAX_HANDOUTS + 1]),
                ..golden()
            },
            DecoderServiceError::Queues,
        ),
    ]
}

/// One state per relation of the handouts' queue, each breaking it alone, with its error.
fn unreachable_handouts() -> Vec<(DecoderService, DecoderServiceError)> {
    vec![
        // A handout of an image not registered.
        (
            DecoderService {
                handouts: VecDeque::from([Handout { image: 0x30, number: 0x1C }]),
                ..golden()
            },
            DecoderServiceError::Handout { index: 0 },
        ),
        // A handout at the last finished number, not above it.
        (
            DecoderService {
                handouts: VecDeque::from([Handout { image: 0x1A, number: 0x19 }]),
                ..golden()
            },
            DecoderServiceError::Handout { index: 0 },
        ),
        // Two handouts out of order: the second refused.
        (
            DecoderService {
                handouts: VecDeque::from([
                    Handout { image: 0x1A, number: 0x1C },
                    Handout { image: 0x1B, number: 0x1B },
                ]),
                ..golden()
            },
            DecoderServiceError::Handout { index: 1 },
        ),
        // A handout past the last made.
        (
            DecoderService {
                handouts: VecDeque::from([Handout { image: 0x1A, number: 0x1D }]),
                ..golden()
            },
            DecoderServiceError::Handout { index: 0 },
        ),
        (
            DecoderService {
                pending: VecDeque::from([Input { timestamp: 0, end: false }]),
                ..golden()
            },
            DecoderServiceError::Unfinished,
        ),
    ]
}

/// Each relation the reading checks, refused alone with its own error.
#[test]
fn a_saved_session_no_command_reaches_is_refused() {
    assert!(read(&golden()).is_ok());
    for (state, error) in unreachable_sessions()
        .into_iter()
        .chain(unreachable_handouts())
    {
        let saved = SavedDecoderService {
            configuration: state.configuration,
            event: state.event,
            syncpoint: state.syncpoint,
            base: state.base,
            handed: state.handed,
            reached: state.reached,
            images: state.images.clone(),
            handouts: state.handouts.clone(),
            pending: state.pending.clone(),
            outputs: state.outputs.clone(),
        };
        assert_eq!(DecoderService::try_from(saved), Err(error), "{error}");
        assert!(
            matches!(read(&state), Err(postcard::Error::SerdeDeCustom)),
            "{error}"
        );
    }
    // At the edges: a reached equal to handed, full queues, and a waiting frame with a full
    // output queue are all taken.
    let edge = DecoderService {
        reached: 0x1C,
        handouts: VecDeque::new(),
        pending: VecDeque::from([Input { timestamp: 0, end: false }; MAX_PENDING]),
        ..golden()
    };
    assert!(read(&edge).is_ok());
    let full = DecoderService {
        pending: VecDeque::from([Input { timestamp: 0, end: false }]),
        outputs: VecDeque::from(
            [Output {
                image: 1,
                width: 0,
                height: 0,
                timestamp: 0,
                end: false,
            }; MAX_OUTPUTS],
        ),
        ..golden()
    };
    assert!(read(&full).is_ok());
}

#[test]
fn error_text_is_plain_words() {
    assert_eq!(
        DecoderServiceError::Images { found: 9 }.to_string(),
        "a decoder service holds 9 images, more than 8"
    );
    assert_eq!(
        DecoderServiceError::Reached {
            reached: 0,
            handed: 0
        }
        .to_string(),
        "a decoder service finished handout 0, but made only 0"
    );
    assert_eq!(
        DecoderServiceError::Unfinished.to_string(),
        "a decoder service has a frame, a handout and room, unfinished"
    );
}

#[test]
fn a_saved_server_is_restored_and_used() {
    let mut rig = Rig::new();
    let decoder = rig.decoder();
    rig.send_on(decoder, &configure(4, 4));
    let first = value_of(&rig.send_on(decoder, &register(rig.first)));
    let second = value_of(&rig.send_on(decoder, &register(rig.first + 1)));
    rig.send_on(decoder, &submit(1, 0, 0));
    let saved = rig.server.save().unwrap();
    let mut restored =
        video_server(Arc::clone(&rig.shared), UnknownPolicy::Strict).with_service_id(VIDEO);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    rig.server = restored;
    // The output from before, then the next frame into the second handout.
    assert_eq!(
        rig.send_on(decoder, &[0xE5, 0]),
        output_reply(rig.first, 4, 4, 0, false)
    );
    rig.send_on(decoder, &submit(1, 5, 0));
    let id = rig.state(decoder).syncpoint().unwrap();
    assert_eq!(rig.syncpoints().read(id), second);
    assert_eq!(second, first + 1);
    // The context still makes decoders.
    rig.decoder();
}
