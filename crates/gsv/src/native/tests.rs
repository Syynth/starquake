//! The native commands driven through raw TIPC messages to a root built with gsv's shared
//! state, over the services framework's fake kernel.
//!
//! Requests are laid out by hand, word by word, from the layouts PR checks
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97): each
//! handler compares the header words with literals (`CreateNativeMemory`: `packet[0] == 0x7f &&
//! packet[1] == 3`), which the tests write as they are. Replies are compared as whole message
//! buffers, laid out by hand the same way.

use std::collections::BTreeMap;
use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_gpu::interface::{Completions, Ticket, VmUpdate};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::GpuVm;
use starquake_ipc::BUFFER_SIZE;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Kernel, Server, UnknownPolicy};
use starquake_trace::ServiceId;

use super::*;
use crate::completion::CompletionQueue;
use crate::completion::testing::Locks;
use crate::gpu::memory::{Device, RegisterRequest};
use crate::gpu::{AsId, GpuVmFront};
use crate::root::{RootState, root_server_with_page};
use crate::testing::{STALE, halves, le, result_only, success, tipc_type};

/// The number the tests' root server is known by.
const ROOT: ServiceId = ServiceId(0x0658);
/// The number gsv's notes are recorded under.
const GSV: ServiceId = ServiceId(0x0659);

/// The title's memory: four pages, then a neighbour page that starts exactly at its end.
const MEMORY: u64 = 0x1000_0000;
const PAGE: u64 = 0x1000;
const MEMORY_SIZE: u64 = 4 * PAGE;
const NEIGHBOUR: u64 = MEMORY + MEMORY_SIZE;

/// A preference as PR reads it (`native_image_layout.h`): kind, extent word, block height,
/// pitch and format at words 1, 2, 4, 5 and 6, 1 at word 7, zeros elsewhere. Shared with the
/// sizer's and the snapshot's tests.
pub(super) fn laid_out(
    kind: u32,
    width: u32,
    height: u32,
    block: u32,
    pitch: u32,
    format: u32,
) -> Preference {
    let mut words = [0; 16];
    words[1] = kind;
    words[2] = (width - 1) | ((height - 1) << 16);
    words[4] = block;
    words[5] = pitch;
    words[6] = format;
    words[7] = 1;
    words
}

/// A one-page native image: 16 pixels (one GOB) wide, 64 rows in 64-row blocks: 64 * 64 =
/// 4096 bytes.
fn page_image() -> Preference {
    laid_out(1, 16, 64, 6, 1, 1)
}

/// A two-page native image: 128 rows.
fn two_page_image() -> Preference {
    laid_out(1, 16, 128, 6, 1, 1)
}

/// A GPU front that does nothing: the parent registration's destroy needs one.
struct NoFront;

impl GpuVmFront for NoFront {
    fn create_address_space(&self, _: AsId, _: Arc<GpuVm>) {}
    fn update_vm(&self, _: AsId, _: VmUpdate, _: Option<Ticket>) {}
    fn barrier(&self, _: Ticket) {}
}

fn fresh_shared() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Arc::new(Shared::new(syncpoints, queue, GSV))
}

/// A root server with gsv's shared state, a kernel with the title's memory mapped and marked
/// GPU-accessible (the neighbour mapped apart and not marked), and one root session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    shared: Arc<Shared>,
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(ROOT);
        kernel.map(MEMORY, vec![0; usize::try_from(MEMORY_SIZE).unwrap()]);
        kernel.map(NEIGHBOUR, vec![0; usize::try_from(PAGE).unwrap()]);
        kernel.mark_gpu_accessible(MEMORY, MEMORY_SIZE);
        let session = kernel.new_object();
        let shared = fresh_shared();
        let mut server =
            root_server_with_page(RootState::new(), UnknownPolicy::Strict, Arc::clone(&shared))
                .with_service_id(ROOT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            shared,
        }
    }

    /// Sends the message `words` lays out; the message buffer afterwards.
    fn send(&mut self, words: &[u32]) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        let bytes = le(words);
        buffer[..bytes.len()].copy_from_slice(&bytes);
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// The native tables now.
    fn native(&self) -> NativeObjects {
        self.shared.lock().native_objects().clone()
    }

    /// 111: `0x7f, 3`, mode, size.
    fn create(&mut self, mode: u32, size: u64) -> Vec<u8> {
        let [low, high] = halves(size);
        self.send(&[0x7F, 3, mode, low, high])
    }

    /// 111 that must succeed: its id.
    fn created(&mut self, size: u64) -> u32 {
        let reply = self.create(0, size);
        let id = word(&reply, 3);
        assert_eq!(reply, success(CREATE_NATIVE_MEMORY, &[], &[id]));
        id
    }

    /// 112: `0x80, 1`, id.
    fn destroy(&mut self, id: u32) -> Vec<u8> {
        self.send(&[0x80, 1, id])
    }

    /// 114: `0x82, 9`, id, address, size, flags, offset.
    fn bind(&mut self, id: u32, address: u64, size: u64, flags: u64, offset: u64) -> Vec<u8> {
        let mut words = vec![0x82, 9, id];
        for value in [address, size, flags, offset] {
            words.extend(halves(value));
        }
        self.send(&words)
    }

    /// A memory of `size` made and bound at `address` with flags 0: its id.
    fn bound(&mut self, address: u64, size: u64) -> u32 {
        let id = self.created(size);
        let reply = self.bind(id, address, size, BIND_PROCESS_MEMORY, 0);
        assert_eq!(reply, success(BIND_NATIVE_MEMORY, &[], &[]));
        id
    }

    /// 423: `0x1b7, 12`, id, address, size, parent, offset, then `tail`'s four words.
    fn bind_gpu(
        &mut self,
        id: u32,
        address: u64,
        size: u64,
        parent: u32,
        offset: u64,
        tail: [u32; 4],
    ) -> Vec<u8> {
        let mut words = vec![0x1B7, 12, id];
        words.extend(halves(address));
        words.extend(halves(size));
        words.push(parent);
        words.extend(halves(offset));
        words.extend(tail);
        self.send(&words)
    }

    /// 429: `0x1bd, 18`, the preference, count, memory.
    fn create_image(&mut self, preference: &Preference, count: u32, memory: u32) -> Vec<u8> {
        let mut words = vec![0x1BD, 18];
        words.extend(preference);
        words.extend([count, memory]);
        self.send(&words)
    }

    /// 429 that must succeed: its id.
    fn image(&mut self, preference: &Preference, memory: u32) -> u32 {
        let reply = self.create_image(preference, 1, memory);
        let id = word(&reply, 3);
        assert_eq!(reply, success(CREATE_NATIVE_IMAGE, &[], &[id]));
        id
    }

    /// 430: `0x1be, 1`, id.
    fn destroy_image(&mut self, id: u32) -> Vec<u8> {
        self.send(&[0x1BE, 1, id])
    }

    /// 315: `0x14b, 2`, image, count.
    fn pool(&mut self, image: u32, count: u32) -> Vec<u8> {
        self.send(&[0x14B, 2, image, count])
    }

    /// 314: `0x14a, 1`, image.
    fn unpool(&mut self, image: u32) -> Vec<u8> {
        self.send(&[0x14A, 1, image])
    }

    /// 435 (`0x1c3, 17`) or 1510 (`0x5f6, 17`): the preference and count.
    fn calculate(&mut self, header: u32, preference: &Preference, count: u32) -> Vec<u8> {
        let mut words = vec![header, 17];
        words.extend(preference);
        words.push(count);
        self.send(&words)
    }

    /// A gpu worker registration of `size` bytes at `address`, its lock taken as 101 takes it:
    /// its handle.
    fn register(&mut self, address: u64, size: u64) -> u32 {
        let mut locked = self.shared.lock();
        let gpu = locked.gpu();
        let device = gpu.create_device(Device { name: 1, label: 2 }).unwrap();
        let request = RegisterRequest {
            first: address,
            last: address + size - 1,
            flags: 0,
            device,
        };
        let admitted = gpu.check_registration(&request).unwrap();
        let token = self.kernel.lock_device_share(address, size).unwrap();
        admitted.register(token)
    }
}

/// Word `index` of a message buffer.
fn word(buffer: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(buffer[index * 4..index * 4 + 4].try_into().unwrap())
}

fn code(error: KernelError) -> ResultCode {
    error.code()
}

// ----- The protocol facts ----------------------------------------------------------------------

/// Each command id and constant, written out as PR and NX2 write them, and the header word PR
/// compares against the id it encodes (`16 + id`).
#[test]
fn the_ids_and_constants_are_the_references() {
    let headers = [
        (CREATE_NATIVE_MEMORY, 111, 0x7F),
        (DESTROY_NATIVE_MEMORY, 112, 0x80),
        (BIND_NATIVE_MEMORY, 114, 0x82),
        (BIND_NATIVE_GPU_MEMORY, 423, 0x1B7),
        (CREATE_NATIVE_IMAGE, 429, 0x1BD),
        (DESTROY_NATIVE_IMAGE, 430, 0x1BE),
        (DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL, 314, 0x14A),
        (CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, 315, 0x14B),
        (CALCULATE_NATIVE_IMAGE_SIZE, 435, 0x1C3),
        (CALCULATE_IMAGE_REQUIREMENTS, 1510, 0x5F6),
    ];
    for (id, literal, header) in headers {
        assert_eq!(id, literal);
        assert_eq!(tipc_type(id), header, "{id}");
    }
    assert_eq!(GSV_INVALID_SIZE.raw(), 0x100F3);
    assert_eq!((MAX_MEMORIES, MAX_IMAGES), (1024, 1024));
    assert_eq!(MAX_MODE, 3);
    assert_eq!(MAX_MEMORY_SIZE, 0x1_0000_0000_0000);
    assert_eq!((BIND_PROCESS_MEMORY, BIND_GPU_ACCESSIBLE), (0, 2));
    assert_eq!((FIRST_ID, WINDOW_POOL_COUNT), (1, 1));
}

// ----- 111 and 112 -----------------------------------------------------------------------------

/// Ids count from 1 for both tables (NX2's numbering: the **conflict** with PR's nvmap ids,
/// which step by 4 from 4).
#[test]
fn ids_count_from_one() {
    let mut rig = Rig::new();
    let first = rig.bound(MEMORY, PAGE);
    let second = rig.created(PAGE);
    assert_eq!((first, second), (1, 2));
    let image = rig.image(&page_image(), first);
    let next = rig.image(&page_image(), first);
    assert_eq!((image, next), (1, 2));
}

/// 111's checks, each alone, at its edges: the mode is its word's low byte; sizes 1 and 2^48
/// are made, 0 and 2^48 + 1 are not; a refusal makes nothing.
#[test]
fn each_create_check_refuses_on_its_own() {
    let mut rig = Rig::new();
    let invalid = result_only(CREATE_NATIVE_MEMORY, code(KernelError::InvalidCombination));
    assert_eq!(rig.create(4, PAGE), invalid);
    assert_eq!(rig.create(0x104, PAGE), invalid);
    let size = result_only(CREATE_NATIVE_MEMORY, GSV_INVALID_SIZE);
    assert_eq!(rig.create(0, 0), size);
    assert_eq!(rig.create(0, 0x1_0000_0000_0001), size);
    // A size differing from an accepted one only in its high word is refused.
    assert_eq!(rig.create(0, 0x1_0001_0000_0000), size);
    // Both faults: the mode first.
    assert_eq!(rig.create(4, 0), invalid);
    assert_eq!(rig.native(), NativeObjects::new());

    // The accepted side of each edge.
    assert_eq!(
        rig.create(0x103, 1),
        success(CREATE_NATIVE_MEMORY, &[], &[1])
    );
    assert_eq!(
        rig.create(0, 0x1_0000_0000_0000),
        success(CREATE_NATIVE_MEMORY, &[], &[2])
    );
    let native = rig.native();
    let first = native.memory(1).unwrap();
    assert_eq!((first.mode(), first.size(), first.binding()), (3, 1, None));
    assert_eq!(native.memory(2).unwrap().size(), 0x1_0000_0000_0000);
}

/// At most 1024 live memories: the 1025th is refused, and a retired memory does not count.
#[test]
fn memories_are_bounded_at_the_limit() {
    let mut rig = Rig::new();
    let first = rig.bound(MEMORY, PAGE);
    for _ in 1..MAX_MEMORIES {
        rig.created(PAGE);
    }
    let full = result_only(CREATE_NATIVE_MEMORY, code(KernelError::OutOfResource));
    assert_eq!(rig.create(0, PAGE), full);
    // Retiring one (an image uses it) makes room for one more, and no more.
    rig.image(&page_image(), first);
    assert_eq!(rig.destroy(first), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    let id = u32::try_from(MAX_MEMORIES).unwrap() + 1;
    assert_eq!(
        rig.create(0, PAGE),
        success(CREATE_NATIVE_MEMORY, &[], &[id])
    );
    assert_eq!(rig.create(0, PAGE), full);
}

/// The last id is `u32::MAX`; then there are none (both tables).
#[test]
fn ids_run_out_after_the_last() {
    let mut rig = Rig::new();
    let memory = rig.bound(MEMORY, PAGE);
    rig.shared.lock().native().set_next_ids(u32::MAX, u32::MAX);
    assert_eq!(
        rig.create(0, PAGE),
        success(CREATE_NATIVE_MEMORY, &[], &[u32::MAX])
    );
    assert_eq!(
        rig.create(0, PAGE),
        result_only(CREATE_NATIVE_MEMORY, code(KernelError::OutOfResource))
    );
    assert_eq!(
        rig.create_image(&page_image(), 1, memory),
        success(CREATE_NATIVE_IMAGE, &[], &[u32::MAX])
    );
    assert_eq!(
        rig.create_image(&page_image(), 1, memory),
        result_only(CREATE_NATIVE_IMAGE, code(KernelError::OutOfResource))
    );
}

/// A memory's life: made, bound (one lock on exactly its range, the neighbour untouched), an
/// image made in it and destroyed, then the memory destroyed, which releases the lock.
#[test]
fn a_memory_is_made_bound_imaged_and_destroyed() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, 2 * PAGE);
    assert_eq!(rig.kernel.device_locks(MEMORY, 2 * PAGE), 1);
    assert_eq!(rig.kernel.device_locks(MEMORY + 2 * PAGE, PAGE), 0);
    assert_eq!(rig.kernel.device_locks(NEIGHBOUR, PAGE), 0);
    let binding = *rig.native().memory(id).unwrap().binding().unwrap();
    assert_eq!(binding.address, MEMORY);
    assert_eq!(
        binding.source,
        Source::Process {
            gpu_accessible: false
        }
    );
    assert_eq!(
        (binding.token.address(), binding.token.size()),
        (MEMORY, 2 * PAGE)
    );

    let image = rig.image(&two_page_image(), id);
    let made = *rig.native().image(image).unwrap();
    assert_eq!(
        (made.memory(), made.count(), made.size(), made.window_pool()),
        (id, 1, 2 * PAGE, false)
    );
    assert_eq!(made.preference(), &two_page_image());
    assert_eq!(
        rig.destroy_image(image),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
    assert_eq!(rig.kernel.device_locks(MEMORY, 2 * PAGE), 1);
    assert_eq!(rig.destroy(id), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    assert_eq!(rig.kernel.device_locks(MEMORY, 2 * PAGE), 0);
    assert_eq!(rig.native().memories().count(), 0);
    assert_eq!(
        rig.destroy(id),
        result_only(DESTROY_NATIVE_MEMORY, code(KernelError::NotFound))
    );
}

/// A memory never bound is destroyed with no release.
#[test]
fn an_unbound_memory_is_destroyed_with_no_release() {
    let mut rig = Rig::new();
    let kept = rig.bound(MEMORY, PAGE);
    let id = rig.created(PAGE);
    assert_eq!(rig.destroy(id), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    assert!(rig.native().memory(id).is_none());
    // The other memory's lock is untouched.
    assert!(rig.native().memory(kept).is_some());
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);
}

/// 112 of a memory an image uses retires it: success, gone for every command, its lock kept
/// until its last image is destroyed (PR `retired_memories`).
#[test]
fn a_memory_in_use_is_retired_until_its_last_image_goes() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let first = rig.image(&page_image(), id);
    let second = rig.image(&page_image(), id);
    assert_eq!(rig.destroy(id), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    let native = rig.native();
    assert!(native.memory(id).is_none());
    assert!(
        native
            .memories()
            .any(|(at, memory)| at == id && memory.retired())
    );
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);

    let not_found = |command| result_only(command, code(KernelError::NotFound));
    assert_eq!(rig.destroy(id), not_found(DESTROY_NATIVE_MEMORY));
    assert_eq!(
        rig.create_image(&page_image(), 1, id),
        not_found(CREATE_NATIVE_IMAGE)
    );
    assert_eq!(
        rig.bind(id, MEMORY, PAGE, BIND_PROCESS_MEMORY, 0),
        not_found(BIND_NATIVE_MEMORY)
    );

    assert_eq!(
        rig.destroy_image(first),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);
    assert!(rig.native().memories().any(|(at, _)| at == id));
    assert_eq!(
        rig.destroy_image(second),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 0);
    assert_eq!(rig.native().memories().count(), 0);
}

/// Destroying the last image of a memory that is not retired keeps the memory and its lock.
#[test]
fn the_last_image_of_a_live_memory_keeps_it() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let image = rig.image(&page_image(), id);
    assert_eq!(
        rig.destroy_image(image),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
    assert!(rig.native().memory(id).is_some());
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);
}

/// A release the kernel refuses is 112's (and 430's) answer, the memory gone all the same (PR
/// erases it before `R_TRY(UnlockForDeviceAddressSpace)`).
#[test]
fn a_refused_release_is_the_answer_and_the_memory_goes() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let token = rig.native().memory(id).unwrap().binding().unwrap().token;
    // Something else released the lock: the kernel no longer holds it.
    rig.kernel.unlock_device_share(token).unwrap();
    assert_eq!(
        rig.destroy(id),
        result_only(
            DESTROY_NATIVE_MEMORY,
            code(KernelError::InvalidCurrentMemory)
        )
    );
    assert_eq!(rig.native().memories().count(), 0);

    // 430's release of a retired memory answers the same way.
    let id = rig.bound(MEMORY, PAGE);
    let image = rig.image(&page_image(), id);
    rig.destroy(id);
    let token = rig
        .native()
        .memories()
        .next()
        .unwrap()
        .1
        .binding()
        .unwrap()
        .token;
    rig.kernel.unlock_device_share(token).unwrap();
    assert_eq!(
        rig.destroy_image(image),
        result_only(
            DESTROY_NATIVE_IMAGE,
            code(KernelError::InvalidCurrentMemory)
        )
    );
    assert_eq!(rig.native().memories().count(), 0);
    assert_eq!(rig.native().images().count(), 0);
}

// ----- 114 -------------------------------------------------------------------------------------

/// 114's checks, each alone, in PR's order: the refusal binds nothing and takes no lock, and a
/// good bind of the same memory succeeds afterwards.
#[test]
fn each_bind_check_refuses_on_its_own() {
    let mut rig = Rig::new();
    let id = rig.created(PAGE);
    let other = rig.bound(MEMORY + PAGE, PAGE);
    let answer = |error| result_only(BIND_NATIVE_MEMORY, code(error));
    let cases = [
        (id + 7, MEMORY, PAGE, 0, 0, KernelError::NotFound),
        (id, MEMORY, PAGE, 1, 0, KernelError::NotImplemented),
        (id, MEMORY, PAGE, 3, 0, KernelError::NotImplemented),
        // Differs from flags 2 only in the high word.
        (
            id,
            MEMORY,
            PAGE,
            0x1_0000_0002,
            0,
            KernelError::NotImplemented,
        ),
        (id, MEMORY, PAGE, 0, 1, KernelError::NotImplemented),
        (
            id,
            MEMORY,
            PAGE,
            0,
            0x1_0000_0000,
            KernelError::NotImplemented,
        ),
        (other, MEMORY, PAGE, 0, 0, KernelError::InvalidState),
        (id, MEMORY, PAGE - 1, 0, 0, KernelError::InvalidSize),
        (id, MEMORY, PAGE + 1, 0, 0, KernelError::InvalidSize),
        (
            id,
            MEMORY,
            PAGE + 0x1_0000_0000,
            0,
            0,
            KernelError::InvalidSize,
        ),
        (id, 0, PAGE, 0, 0, KernelError::InvalidSize),
        (
            id,
            u64::MAX - PAGE + 1,
            PAGE,
            0,
            0,
            KernelError::InvalidSize,
        ),
        (id, MEMORY + 1, PAGE, 0, 0, KernelError::InvalidAddress),
        (id, MEMORY + 0x800, PAGE, 0, 0, KernelError::InvalidAddress),
        // Not GPU accessible (the neighbour), with flags 2.
        (id, NEIGHBOUR, PAGE, 2, 0, KernelError::InvalidCurrentMemory),
        // Not mapped: the kernel's lock refuses it.
        (
            id,
            NEIGHBOUR + PAGE,
            PAGE,
            0,
            0,
            KernelError::InvalidCurrentMemory,
        ),
        (
            id,
            NEIGHBOUR + PAGE,
            PAGE,
            2,
            0,
            KernelError::InvalidCurrentMemory,
        ),
    ];
    for (memory, address, size, flags, offset, error) in cases {
        let reply = rig.bind(memory, address, size, flags, offset);
        assert_eq!(
            reply,
            answer(error),
            "{memory} {address:#x} {size:#x} {flags} {offset}"
        );
        assert_eq!(rig.native().memory(id).unwrap().binding(), None);
        assert_eq!(rig.kernel.device_locks(address, size), 0);
    }
    // The last page-aligned range that does not wrap passes the size and alignment checks and
    // reaches the kernel, which refuses it as unmapped; one page later it wraps.
    assert_eq!(
        rig.bind(id, u64::MAX - 2 * PAGE + 1, PAGE, 0, 0),
        answer(KernelError::InvalidCurrentMemory)
    );
    // The neighbour binds with flags 0: it is mapped, just not GPU accessible.
    assert_eq!(
        rig.bind(id, NEIGHBOUR, PAGE, BIND_PROCESS_MEMORY, 0),
        success(BIND_NATIVE_MEMORY, &[], &[])
    );
    assert_eq!(rig.kernel.device_locks(NEIGHBOUR, PAGE), 1);
}

/// With two faults, the one PR checks first answers, for each adjacent pair.
#[test]
fn bind_checks_run_in_pr_s_order() {
    let mut rig = Rig::new();
    let id = rig.created(PAGE);
    let bound = rig.bound(MEMORY + PAGE, PAGE);
    let answer = |error| result_only(BIND_NATIVE_MEMORY, code(error));
    // Not found and bad flags.
    assert_eq!(
        rig.bind(id + 9, MEMORY, PAGE, 1, 0),
        answer(KernelError::NotFound)
    );
    // Bad flags and bound.
    assert_eq!(
        rig.bind(bound, MEMORY, PAGE, 1, 0),
        answer(KernelError::NotImplemented)
    );
    // Bound and a bad size.
    assert_eq!(
        rig.bind(bound, MEMORY, 0, 0, 0),
        answer(KernelError::InvalidState)
    );
    // A bad size and a bad address.
    assert_eq!(
        rig.bind(id, MEMORY + 1, 0, 0, 0),
        answer(KernelError::InvalidSize)
    );
    // A bad address and not GPU accessible.
    assert_eq!(
        rig.bind(id, NEIGHBOUR + 1, PAGE, 2, 0),
        answer(KernelError::InvalidAddress)
    );
}

/// With flags 2 the range must be GPU accessible; one byte of the neighbour past the marked
/// range is enough to refuse, and the marked range up to its end binds.
#[test]
fn a_gpu_accessible_bind_checks_the_whole_range() {
    let mut rig = Rig::new();
    let id = rig.created(2 * PAGE);
    assert_eq!(
        rig.bind(id, MEMORY + 3 * PAGE, 2 * PAGE, BIND_GPU_ACCESSIBLE, 0),
        result_only(BIND_NATIVE_MEMORY, code(KernelError::InvalidCurrentMemory))
    );
    assert_eq!(
        rig.bind(id, MEMORY + 2 * PAGE, 2 * PAGE, BIND_GPU_ACCESSIBLE, 0),
        success(BIND_NATIVE_MEMORY, &[], &[])
    );
    assert_eq!(
        rig.native().memory(id).unwrap().binding().unwrap().source,
        Source::Process {
            gpu_accessible: true
        }
    );
}

/// A request one word short of a command's fields changes nothing.
#[test]
fn a_short_request_changes_nothing() {
    let mut rig = Rig::new();
    let short = |command| result_only(command, INVALID_CMIF_REQUEST);
    assert_eq!(rig.send(&[0x7F, 2, 0, 0x1000]), short(CREATE_NATIVE_MEMORY));
    assert_eq!(rig.native(), NativeObjects::new());
    let id = rig.created(PAGE);
    let mut bind = vec![0x82, 8, id];
    for value in [MEMORY, PAGE, 0] {
        bind.extend(halves(value));
    }
    bind.push(0);
    assert_eq!(rig.send(&bind), short(BIND_NATIVE_MEMORY));
    assert_eq!(rig.native().memory(id).unwrap().binding(), None);
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 0);
    rig.bind(id, MEMORY, PAGE, 0, 0);
    let mut image = vec![0x1BD, 17];
    image.extend(page_image());
    image.push(1);
    assert_eq!(rig.send(&image), short(CREATE_NATIVE_IMAGE));
    assert_eq!(rig.native().images().count(), 0);
    assert_eq!(
        rig.send(&[0x14B, 1, 1]),
        short(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL)
    );
    let mut calculate = vec![0x5F6, 16];
    calculate.extend(page_image());
    assert_eq!(rig.send(&calculate), short(CALCULATE_IMAGE_REQUIREMENTS));
}

/// Every command is answered before the bootstrap is complete (decision 7; PR answers
/// `InvalidState`): the rig never bootstraps.
#[test]
fn commands_before_the_bootstrap_are_accepted() {
    let mut rig = Rig::new();
    let state = rig
        .server
        .family::<crate::root::RootFamily>()
        .unwrap()
        .state()
        .phase();
    assert_eq!(state, crate::root::Phase::Created);
    let id = rig.bound(MEMORY, PAGE);
    let image = rig.image(&page_image(), id);
    assert_eq!(
        rig.pool(image, 1),
        success(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, &[], &[])
    );
}

// ----- 423 -------------------------------------------------------------------------------------

/// 423 binds part of a gpu worker registration: the range at the offset, its own lock taken.
/// The memory id (3), the parent's handle (2) and the other registration's and memories'
/// numbers all differ, so a decoder that swapped the id (word 2) and the parent (word 7) binds
/// nothing here.
#[test]
fn a_gpu_bind_binds_part_of_a_registration() {
    let mut rig = Rig::new();
    let other_registration = rig.register(NEIGHBOUR, PAGE);
    let parent = rig.register(MEMORY, 3 * PAGE);
    let others = [rig.created(PAGE), rig.created(PAGE)];
    let id = rig.created(PAGE);
    assert_eq!(
        (other_registration, parent, others, id),
        (
            FIRST_ID,
            FIRST_ID + 1,
            [FIRST_ID, FIRST_ID + 1],
            FIRST_ID + 2
        )
    );
    let reply = rig.bind_gpu(id, MEMORY + PAGE, PAGE, parent, PAGE, [0; 4]);
    assert_eq!(reply, success(BIND_NATIVE_GPU_MEMORY, &[], &[]));
    for other in others {
        assert_eq!(rig.native().memory(other).unwrap().binding(), None);
    }
    let binding = *rig.native().memory(id).unwrap().binding().unwrap();
    assert_eq!(binding.address, MEMORY + PAGE);
    assert_eq!(
        binding.source,
        Source::Gpu {
            parent,
            offset: PAGE
        }
    );
    assert_eq!(rig.kernel.device_locks(MEMORY + PAGE, PAGE), 1);
    assert_eq!(rig.kernel.device_locks(MEMORY, 3 * PAGE), 1);
}

/// 423's own checks at their edges, each paired so the accepted side reaches the next check:
/// the offset at the parent's end is in range (then the size 0 is `InvalidSize`), one page
/// past is `NotFound`; a size up to the parent's end passes, one byte more does not; the
/// address must be the parent's start plus the offset.
#[test]
fn gpu_bind_parent_checks() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, 2 * PAGE);
    let id = rig.created(PAGE);
    let answer = |error| result_only(BIND_NATIVE_GPU_MEMORY, code(error));
    let cases = [
        (
            MEMORY + 2 * PAGE,
            0,
            parent,
            2 * PAGE,
            KernelError::InvalidSize,
        ),
        (
            MEMORY + 3 * PAGE,
            0,
            parent,
            3 * PAGE,
            KernelError::NotFound,
        ),
        // A size reaching the parent's end exactly passes (then it is not the memory's size).
        (MEMORY, 2 * PAGE, parent, 0, KernelError::InvalidSize),
        (MEMORY, 2 * PAGE + 1, parent, 0, KernelError::NotFound),
        (MEMORY + PAGE, PAGE + 1, parent, PAGE, KernelError::NotFound),
        (MEMORY + PAGE + 1, PAGE, parent, PAGE, KernelError::NotFound),
        (MEMORY, PAGE, parent, PAGE, KernelError::NotFound),
        (MEMORY, PAGE, parent + 1, 0, KernelError::NotFound),
        // An offset that wraps the parent's start.
        (MEMORY, PAGE, parent, u64::MAX, KernelError::NotFound),
        // An offset that differs from the good one only in its high word.
        (
            MEMORY + PAGE,
            PAGE,
            parent,
            PAGE + 0x1_0000_0000,
            KernelError::NotFound,
        ),
    ];
    for (address, size, from, offset, error) in cases {
        let reply = rig.bind_gpu(id, address, size, from, offset, [0; 4]);
        assert_eq!(
            reply,
            answer(error),
            "{address:#x} {size:#x} {from} {offset:#x}"
        );
        assert_eq!(rig.native().memory(id).unwrap().binding(), None);
    }
    assert_eq!(rig.kernel.device_locks(MEMORY + PAGE, PAGE), 0);
}

/// Words 10 to 13 must be zero, each of them, high halves too.
#[test]
fn gpu_bind_trailing_words_must_be_zero() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, PAGE);
    let id = rig.created(PAGE);
    let invalid = result_only(
        BIND_NATIVE_GPU_MEMORY,
        code(KernelError::InvalidCombination),
    );
    for at in 0..4 {
        let mut tail = [0; 4];
        tail[at] = 1;
        assert_eq!(
            rig.bind_gpu(id, MEMORY, PAGE, parent, 0, tail),
            invalid,
            "{at}"
        );
    }
    // Before the parent: both faults at once.
    assert_eq!(
        rig.bind_gpu(id, MEMORY, PAGE, parent + 5, 0, [0, 0, 0, 1]),
        invalid
    );
    assert_eq!(rig.native().memory(id).unwrap().binding(), None);
}

/// The parent is checked before the memory, and then 114's checks run with flags 2.
#[test]
fn gpu_bind_checks_run_in_pr_s_order() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, 2 * PAGE);
    let bound = rig.bound(MEMORY + 2 * PAGE, PAGE);
    let answer = |error| result_only(BIND_NATIVE_GPU_MEMORY, code(error));
    // A bad parent range and a bound memory: the parent first.
    assert_eq!(
        rig.bind_gpu(bound, MEMORY + 5 * PAGE, PAGE, parent, 0, [0; 4]),
        answer(KernelError::NotFound)
    );
    // A good parent and a bound memory.
    assert_eq!(
        rig.bind_gpu(bound, MEMORY, PAGE, parent, 0, [0; 4]),
        answer(KernelError::InvalidState)
    );
    // A good parent and an unknown memory.
    assert_eq!(
        rig.bind_gpu(bound + 9, MEMORY, PAGE, parent, 0, [0; 4]),
        answer(KernelError::NotFound)
    );
}

/// A gpu bind checks the GPU-accessible attribute (flags 2): a registration over memory that
/// is no longer marked is refused with `InvalidCurrentMemory`, as PR's walk refuses it.
#[test]
fn a_gpu_bind_checks_the_attribute() {
    let mut rig = Rig::new();
    // A registration over the neighbour page, which is mapped but not GPU accessible (as if
    // the guest cleared the attribute after 101).
    let parent = rig.register(NEIGHBOUR, PAGE);
    let id = rig.created(PAGE);
    assert_eq!(
        rig.bind_gpu(id, NEIGHBOUR, PAGE, parent, 0, [0; 4]),
        result_only(
            BIND_NATIVE_GPU_MEMORY,
            code(KernelError::InvalidCurrentMemory)
        )
    );
    assert_eq!(rig.native().memory(id).unwrap().binding(), None);
}

/// 423's parent is looked up among gpu worker registration handles only (PR takes any bound
/// nvmap handle of the process; a listed divergence). Native memory ids and registration
/// handles are separate counters that both start at [`FIRST_ID`], so a number names whatever
/// registration has it: a bound native memory's id that no registration has is `NotFound`,
/// and a native memory's id that equals a live registration's handle names that registration.
#[test]
fn a_gpu_bind_s_parent_is_a_registration() {
    let mut rig = Rig::new();
    let registration = rig.register(MEMORY, 2 * PAGE);
    let same_number = rig.created(PAGE);
    let native_parent = rig.bound(MEMORY + 2 * PAGE, 2 * PAGE);
    let id = rig.created(PAGE);
    assert_eq!(
        (registration, same_number, native_parent),
        (FIRST_ID, FIRST_ID, FIRST_ID + 1)
    );
    // A bound native memory's id, which no registration has, over the native memory's range.
    assert_eq!(
        rig.bind_gpu(id, MEMORY + 2 * PAGE, PAGE, native_parent, 0, [0; 4]),
        result_only(BIND_NATIVE_GPU_MEMORY, code(KernelError::NotFound))
    );
    assert_eq!(rig.native().memory(id).unwrap().binding(), None);
    // An unbound native memory's id that is also the registration's handle: the registration.
    assert_eq!(
        rig.bind_gpu(id, MEMORY, PAGE, same_number, 0, [0; 4]),
        success(BIND_NATIVE_GPU_MEMORY, &[], &[])
    );
    assert_eq!(
        rig.native().memory(id).unwrap().binding().unwrap().source,
        Source::Gpu {
            parent: registration,
            offset: 0
        }
    );
}

/// A 102 of the parent goes ahead while the native memory stays bound, holding its own lock on
/// its range, which its 112 releases (PR holds the parent's destroy; a listed divergence).
#[test]
fn a_parent_destroy_leaves_the_bind_s_own_lock() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, 2 * PAGE);
    let id = rig.created(PAGE);
    rig.bind_gpu(id, MEMORY + PAGE, PAGE, parent, PAGE, [0; 4]);
    let event = rig.kernel.new_object();
    let token = rig
        .shared
        .lock()
        .gpu_memory()
        .registration(parent)
        .unwrap()
        .token;
    rig.shared
        .lock()
        .gpu()
        .destroy_memory(&NoFront, parent, event)
        .unwrap();
    // The drain releases the parent's lock once the GPU is done; here, at once.
    rig.kernel.unlock_device_share(token).unwrap();
    assert!(
        rig.shared
            .lock()
            .gpu_memory()
            .registration(parent)
            .is_none()
    );
    assert_eq!(rig.kernel.device_locks(MEMORY, 2 * PAGE), 0);
    assert!(rig.native().memory(id).unwrap().binding().is_some());
    assert_eq!(rig.kernel.device_locks(MEMORY + PAGE, PAGE), 1);
    assert_eq!(rig.destroy(id), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    assert_eq!(rig.kernel.device_locks(MEMORY + PAGE, PAGE), 0);
}

/// Closing the root session releases nothing: the tables live in gsv's shared state for the
/// whole run, and every lock stays held (PR's `~IGraphicsSupervisor` releases every native
/// memory and `~NativeImageRegistry` drains the retired ones; a listed divergence).
#[test]
fn closing_the_root_session_releases_nothing() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, 2 * PAGE);
    let gpu = rig.created(PAGE);
    rig.bind_gpu(gpu, MEMORY, PAGE, parent, 0, [0; 4]);
    let live = rig.bound(MEMORY + 2 * PAGE, PAGE);
    rig.image(&page_image(), live);
    let retired = rig.bound(MEMORY + 3 * PAGE, PAGE);
    rig.image(&page_image(), retired);
    assert_eq!(
        rig.destroy(retired),
        success(DESTROY_NATIVE_MEMORY, &[], &[])
    );
    let ranges = [
        (MEMORY, PAGE),
        (MEMORY, 2 * PAGE),
        (MEMORY + 2 * PAGE, PAGE),
        (MEMORY + 3 * PAGE, PAGE),
    ];
    let locks = |rig: &Rig| ranges.map(|(address, size)| rig.kernel.device_locks(address, size));
    let (tables, held) = (rig.native(), locks(&rig));
    assert_eq!(held, [1; 4]);
    assert_eq!(tables.memories().count(), 3);
    assert_eq!(tables.images().count(), 2);
    rig.server.close(rig.session).unwrap();
    assert_eq!(rig.native(), tables);
    assert_eq!(locks(&rig), held);
}

// ----- 429 and 430 -----------------------------------------------------------------------------

/// 429's checks, each alone; a refusal makes nothing.
#[test]
fn each_image_check_refuses_on_its_own() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let unbound = rig.created(PAGE);
    let answer = |error| result_only(CREATE_NATIVE_IMAGE, code(error));
    let mut block = page_image();
    block[1] = 2;
    let mut bad = page_image();
    bad[6] = 3;
    let cases = [
        (block, 1, id, KernelError::InvalidCombination),
        (bad, 1, id, KernelError::NotImplemented),
        // 429 has no special answer for a count of 0 (435 and 1510 do).
        (page_image(), 0, id, KernelError::NotImplemented),
        (page_image(), 1, id + 9, KernelError::NotFound),
        (page_image(), 1, unbound, KernelError::NotFound),
        // Two pages in a one-page memory; and one page twice over.
        (two_page_image(), 1, id, KernelError::InvalidSize),
        (page_image(), 2, id, KernelError::InvalidSize),
    ];
    for (preference, count, memory, error) in cases {
        let reply = rig.create_image(&preference, count, memory);
        assert_eq!(reply, answer(error), "{count} {memory}");
        assert_eq!(rig.native().images().count(), 0);
    }
    // Exactly the memory's size fits.
    assert_eq!(
        rig.create_image(&page_image(), 1, id),
        success(CREATE_NATIVE_IMAGE, &[], &[1])
    );
}

/// With two faults, the one PR checks first answers.
#[test]
fn image_checks_run_in_pr_s_order() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let answer = |error| result_only(CREATE_NATIVE_IMAGE, code(error));
    let mut both = page_image();
    both[1] = 2;
    both[6] = 3;
    assert_eq!(
        rig.create_image(&both, 1, id),
        answer(KernelError::InvalidCombination)
    );
    let mut bad = page_image();
    bad[6] = 3;
    assert_eq!(
        rig.create_image(&bad, 1, id + 9),
        answer(KernelError::NotImplemented)
    );
    assert_eq!(
        rig.create_image(&two_page_image(), 1, id + 9),
        answer(KernelError::NotFound)
    );
}

/// At most 1024 images: the 1025th is refused, and destroying one makes room for one.
#[test]
fn images_are_bounded_at_the_limit() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    for _ in 0..MAX_IMAGES {
        rig.image(&page_image(), id);
    }
    let full = result_only(CREATE_NATIVE_IMAGE, code(KernelError::OutOfResource));
    assert_eq!(rig.create_image(&page_image(), 1, id), full);
    // Both faults: too small first.
    assert_eq!(
        rig.create_image(&two_page_image(), 1, id),
        result_only(CREATE_NATIVE_IMAGE, code(KernelError::InvalidSize))
    );
    rig.destroy_image(1);
    let next = u32::try_from(MAX_IMAGES).unwrap() + 1;
    assert_eq!(
        rig.create_image(&page_image(), 1, id),
        success(CREATE_NATIVE_IMAGE, &[], &[next])
    );
    assert_eq!(rig.create_image(&page_image(), 1, id), full);
}

/// 430 of an unknown image is `NotFound`; of one in a window pool, `Busy`, and it stays.
#[test]
fn an_image_in_a_window_pool_is_busy() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let image = rig.image(&page_image(), id);
    assert_eq!(
        rig.destroy_image(image + 1),
        result_only(DESTROY_NATIVE_IMAGE, code(KernelError::NotFound))
    );
    assert_eq!(
        rig.pool(image, 1),
        success(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, &[], &[])
    );
    assert!(rig.native().image(image).unwrap().window_pool());
    assert_eq!(
        rig.destroy_image(image),
        result_only(DESTROY_NATIVE_IMAGE, code(KernelError::Busy))
    );
    assert!(rig.native().image(image).is_some());
    assert_eq!(
        rig.unpool(image),
        success(DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL, &[], &[])
    );
    assert!(!rig.native().image(image).unwrap().window_pool());
    assert_eq!(
        rig.destroy_image(image),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
}

/// 314's and 315's checks, each alone, and both faults at once.
#[test]
fn window_pool_checks() {
    let mut rig = Rig::new();
    let id = rig.bound(MEMORY, PAGE);
    let image = rig.image(&page_image(), id);
    let pool = |error| result_only(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, code(error));
    let unpool = |error| result_only(DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL, code(error));
    assert_eq!(rig.pool(image + 1, 1), pool(KernelError::NotFound));
    assert_eq!(rig.pool(image + 1, 2), pool(KernelError::NotFound));
    assert_eq!(rig.pool(image, 2), pool(KernelError::NotImplemented));
    assert_eq!(rig.pool(image, 0x101), pool(KernelError::NotImplemented));
    assert_eq!(rig.pool(image, 0), pool(KernelError::NotImplemented));
    assert_eq!(rig.unpool(image), unpool(KernelError::InvalidState));
    assert_eq!(rig.unpool(image + 1), unpool(KernelError::NotFound));
    assert!(!rig.native().image(image).unwrap().window_pool());
    assert_eq!(
        rig.pool(image, 1),
        success(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, &[], &[])
    );
    assert_eq!(rig.pool(image, 1), pool(KernelError::InvalidState));
    // In a pool, a bad count is still `NotImplemented`: it comes before the state.
    assert_eq!(rig.pool(image, 2), pool(KernelError::NotImplemented));
    // 314's count word, if sent, is not read.
    assert_eq!(
        rig.send(&[0x14A, 2, image, 9]),
        success(DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL, &[], &[])
    );
}

// ----- 435 and 1510 ----------------------------------------------------------------------------

/// 435 answers the size, 1510 the size and the alignment, for the kind each requires.
#[test]
fn calculations_reply_the_size_and_alignment() {
    let mut rig = Rig::new();
    // 1280 by 720, 80 GOBs, 128-row blocks, three images: 3 * 3,932,160 bytes.
    let size = 3 * 3_932_160;
    let native = laid_out(1, 1280, 720, 7, 80, 1);
    assert_eq!(
        rig.calculate(0x1C3, &native, 3),
        success(CALCULATE_NATIVE_IMAGE_SIZE, &[], &halves(size))
    );
    let block = laid_out(2, 1280, 720, 7, 80, 4);
    let mut words = halves(size).to_vec();
    words.extend(halves(4096));
    assert_eq!(
        rig.calculate(0x5F6, &block, 3),
        success(CALCULATE_IMAGE_REQUIREMENTS, &[], &words)
    );
    // Each requires its own kind.
    let invalid = |command| result_only(command, code(KernelError::InvalidCombination));
    assert_eq!(
        rig.calculate(0x1C3, &block, 1),
        invalid(CALCULATE_NATIVE_IMAGE_SIZE)
    );
    assert_eq!(
        rig.calculate(0x5F6, &native, 1),
        invalid(CALCULATE_IMAGE_REQUIREMENTS)
    );
    // A kind word that differs from 2 only above its low byte.
    let mut high = block;
    high[1] = 0x102;
    assert_eq!(
        rig.calculate(0x5F6, &high, 1),
        invalid(CALCULATE_IMAGE_REQUIREMENTS)
    );
}

/// A count of 0 with a good preference is gsv's result `0x100f3`; with a bad one, and any other
/// refusal, `NotImplemented`; the kind is checked before both.
#[test]
fn a_zero_count_is_gsv_s_invalid_size() {
    let mut rig = Rig::new();
    let native = laid_out(1, 64, 64, 6, 4, 1);
    for (header, command) in [
        (0x1C3, CALCULATE_NATIVE_IMAGE_SIZE),
        (0x5F6, CALCULATE_IMAGE_REQUIREMENTS),
    ] {
        let mut preference = native;
        preference[1] = if header == 0x1C3 { 1 } else { 2 };
        assert_eq!(
            rig.calculate(header, &preference, 0),
            result_only(command, GSV_INVALID_SIZE)
        );
        let mut bad = preference;
        bad[4] = 2;
        let not_implemented = result_only(command, code(KernelError::NotImplemented));
        assert_eq!(rig.calculate(header, &bad, 0), not_implemented);
        assert_eq!(rig.calculate(header, &bad, 1), not_implemented);
        let mut wrong = bad;
        wrong[1] = 3;
        assert_eq!(
            rig.calculate(header, &wrong, 0),
            result_only(command, code(KernelError::InvalidCombination))
        );
    }
}

// ----- The snapshot, in use --------------------------------------------------------------------

/// The state live commands leave (a retired memory held by its image, a memory bound by 423)
/// saves, restores against the kernel's locks, and the restored tables answer the next
/// commands: the image's destroy releases the retired memory's lock, and ids go on.
#[test]
fn a_restored_table_answers_commands() {
    let mut rig = Rig::new();
    let parent = rig.register(MEMORY, 2 * PAGE);
    let retired = rig.bound(MEMORY + 2 * PAGE, PAGE);
    let image = rig.image(&page_image(), retired);
    rig.destroy(retired);
    let gpu = rig.created(PAGE);
    rig.bind_gpu(gpu, MEMORY, PAGE, parent, 0, [0; 4]);
    let bytes = rig.native().save().unwrap();
    let locks = Locks {
        process: rig.kernel.process(),
        live: BTreeMap::from([((MEMORY + 2 * PAGE, PAGE), 1), ((MEMORY, PAGE), 1)]),
    };
    let restored = NativeObjects::restore(&bytes, &locks).unwrap();
    assert_eq!(restored, rig.native());
    // Without it, the image is unknown; with it, its destroy releases the retired memory.
    *rig.shared.lock().native() = NativeObjects::new();
    assert_eq!(
        rig.destroy_image(image),
        result_only(DESTROY_NATIVE_IMAGE, code(KernelError::NotFound))
    );
    *rig.shared.lock().native() = restored;
    assert_eq!(
        rig.destroy_image(image),
        success(DESTROY_NATIVE_IMAGE, &[], &[])
    );
    assert_eq!(rig.kernel.device_locks(MEMORY + 2 * PAGE, PAGE), 0);
    assert_eq!(rig.destroy(gpu), success(DESTROY_NATIVE_MEMORY, &[], &[]));
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 0);
    assert_eq!(rig.created(PAGE), gpu + 1);
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 64,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x4E41_5449_5645_4D45),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// 114 and 423 laid out from generated fields record each where it belongs: the binding's
    /// address, its lock's range (the memory's size) and its source, on the memory the request
    /// names and no other. Other registrations (over the neighbour page) and other unbound
    /// memories are made first, so the memory id and the parent's handle always differ, either
    /// one the larger: a decoder that swapped any two reads, the id (word 2) and the parent
    /// (word 7) included, fails here.
    #[test]
    fn binds_read_each_field_from_its_own_words(
        first_page in 0..4_u64,
        pages in 1..=4_u64,
        offset_pages in 0..4_u64,
        gpu in any::<bool>(),
        gpu_accessible in any::<bool>(),
        base in 0..2_u32,
        gap in 1..3_u32,
        parent_larger in any::<bool>(),
    ) {
        prop_assume!(offset_pages < pages);
        let mut rig = Rig::new();
        let parent_start = MEMORY + first_page.min(3) * PAGE;
        let parent_pages = pages.min((MEMORY + MEMORY_SIZE - parent_start) / PAGE);
        prop_assume!(offset_pages < parent_pages);
        let size = (parent_pages - offset_pages) * PAGE;
        let address = parent_start + offset_pages * PAGE;
        let (other_registrations, other_memories) =
            if parent_larger { (base + gap, base) } else { (base, base + gap) };
        for _ in 0..other_registrations {
            rig.register(NEIGHBOUR, PAGE);
        }
        let others: Vec<u32> = (0..other_memories).map(|_| rig.created(PAGE)).collect();
        let id = rig.created(size);
        let (reply, source) = if gpu {
            let parent = rig.register(parent_start, parent_pages * PAGE);
            prop_assert_ne!(parent, id);
            prop_assert_eq!(parent > id, parent_larger);
            let offset = offset_pages * PAGE;
            (
                rig.bind_gpu(id, address, size, parent, offset, [0; 4]),
                Source::Gpu { parent, offset },
            )
        } else {
            let flags = if gpu_accessible { BIND_GPU_ACCESSIBLE } else { BIND_PROCESS_MEMORY };
            (rig.bind(id, address, size, flags, 0), Source::Process { gpu_accessible })
        };
        let command = if gpu { BIND_NATIVE_GPU_MEMORY } else { BIND_NATIVE_MEMORY };
        prop_assert_eq!(reply, success(command, &[], &[]));
        let tables = rig.native();
        for other in others {
            prop_assert_eq!(tables.memory(other).unwrap().binding(), None);
        }
        let binding = *rig.native().memory(id).unwrap().binding().unwrap();
        prop_assert_eq!(binding.address, address);
        prop_assert_eq!(binding.source, source);
        prop_assert_eq!((binding.token.address(), binding.token.size()), (address, size));
    }
}
