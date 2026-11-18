//! The pool commands: the tables' checks driven directly, and the root's replies through raw
//! TIPC messages laid out by hand, word by word, from the headers PR checks
//! (`graphics_supervisor.cpp` @ e5fb3b97: `CreateVideoImagePool`: `packet[0] == 0x39f &&
//! packet[1] == 18`, and so on, written here as they are).

use std::sync::Arc;

use starquake_gpu::interface::{Completions, Ticket, VmUpdate};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::GpuVm;
use starquake_ipc::BUFFER_SIZE;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::testing::device_share_token;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::result::INVALID_CMIF_REQUEST;
use starquake_services::framework::{Handled, Server, UnknownPolicy};
use starquake_trace::ServiceId;

use super::*;
use crate::completion::CompletionQueue;
use crate::completion::testing::process;
use crate::gpu::GpuVmFront;
use crate::native::layout::PREFERENCE_WORDS;
use crate::native::{Binding, FIRST_ID, MAX_IMAGES, Source};
use crate::root::{RootState, root_server_with_page};
use crate::testing::{STALE, halves, le, result_only, success, tipc_type};
use crate::video::layout::{
    CODE_BGRA_NX2, CODE_NV12, CODE_NV12_NX2, CODE_RGBA_BLOCK, CODES_RGBA_NX2,
};

/// The number the tests' root server is known by.
const ROOT: ServiceId = ServiceId(0x0758);
/// The number gsv's notes are recorded under.
const GSV: ServiceId = ServiceId(0x0759);
/// Where the title's memory is.
const MEMORY: u64 = 0x2000_0000;
/// One page.
const PAGE: u64 = 0x1000;

/// A pool preference with `code` and an extent of `width` by `height`.
pub(crate) fn pool_preference(code: u32, width: u32, height: u32) -> Preference {
    let mut words = [0; PREFERENCE_WORDS];
    words[layout::WORD_CODE] = code;
    words[layout::WORD_EXTENT] = (width - 1) | ((height - 1) << 16);
    words
}

/// The smallest NV12 pool image: 2x2, 4096 bytes (PR's floor).
pub(crate) fn small() -> Preference {
    pool_preference(CODE_NV12, 2, 2)
}

/// Tables holding one memory of `size` bytes bound at `address`: the tables and its id.
fn tables(address: u64, size: u64) -> (NativeObjects, u32) {
    let mut objects = NativeObjects::new();
    let id = objects.create_memory(0, size).unwrap();
    objects.memories.get_mut(&id).unwrap().binding = Some(Binding {
        address,
        source: Source::Process {
            gpu_accessible: false,
        },
        token: device_share_token(process(), address, size),
    });
    (objects, id)
}

fn code(error: KernelError) -> ResultCode {
    error.code()
}

/// The references' constants, written out as the cited sources write them.
#[test]
fn the_constants_are_the_references_values() {
    assert_eq!(MAX_POOLS, 1024);
    assert_eq!(MAX_POOL_IMAGES, 1024);
    assert_eq!(MAX_MAPPINGS, 1024);
    assert_eq!(DESCRIPTOR_NV12, 0xA);
    assert_eq!(DESCRIPTOR_RGBA, 8);
    assert_eq!(DESCRIPTOR_BGRA, 9);
    assert_eq!(GPU_DESCRIPTOR_KIND, 1);
    assert_eq!(GPU_DESCRIPTOR_WORD_4, 0x1_0100_0001);
    assert_eq!(MAP_WORD_3, 1);
    assert_eq!(MAP_MODES, [1, 3]);
    assert_eq!(DESCRIPTOR_WORDS, 16);
    // The headers PR checks are the ones the ids make.
    for (id, header) in [
        (991, 0x3EF),
        (911, 0x39F),
        (912, 0x3A0),
        (961, 0x3D1),
        (426, 0x1BA),
        (427, 0x1BB),
        (428, 0x1BC),
    ] {
        assert_eq!(tipc_type(id), header);
    }
}

// ---- The tables ----

#[test]
fn a_pool_takes_its_ids_from_the_native_image_counter() {
    let (mut objects, memory) = tables(MEMORY, 16 * PAGE);
    let native = crate::native::tests::laid_out(1, 16, 64, 6, 1, 1);
    let image = objects.create_image(&native, 1, memory).unwrap();
    assert_eq!(image, FIRST_ID);
    let first = objects.create_pool(&small(), 3, memory).unwrap();
    assert_eq!(first, FIRST_ID + 1);
    // Image i of the pool is first + i; the next native image comes after the pool.
    for index in 0..3 {
        let found = objects.pool_image(first + index).unwrap();
        assert_eq!((found.first, found.index), (first, index));
    }
    assert!(objects.pool_image(first + 3).is_none());
    assert!(objects.pool_image(image).is_none());
    assert_eq!(objects.create_image(&native, 1, memory).unwrap(), first + 3);
    // A second pool right after: its first image is not the first pool's.
    let second = objects.create_pool(&small(), 1, memory).unwrap();
    assert_eq!(second, first + 4);
    assert_eq!(objects.pool_image(second).unwrap().first, second);
    assert_eq!(objects.pool_image(first + 2).unwrap().first, first);
}

/// One case per check of 911, each alone, in PR's order (module docs, "Checks, in order").
#[test]
fn each_create_check_refuses_on_its_own() {
    let (fresh, memory) = tables(MEMORY, 4 * PAGE);
    // No such memory.
    let mut objects = fresh.clone();
    assert_eq!(
        objects.create_pool(&small(), 1, memory + 1),
        Err(code(KernelError::NotFound))
    );
    // A count of 0.
    assert_eq!(
        objects.create_pool(&small(), 0, memory),
        Err(code(KernelError::OutOfResource))
    );
    // A preference the layout refuses.
    assert_eq!(
        objects.create_pool(&pool_preference(7, 2, 2), 1, memory),
        Err(code(KernelError::NotImplemented))
    );
    // A memory that is not bound.
    let unbound = objects.create_memory(0, 4 * PAGE).unwrap();
    assert_eq!(
        objects.create_pool(&small(), 1, unbound),
        Err(code(KernelError::NotFound))
    );
    // A memory smaller than the pool: four 4096-byte images fit exactly; five do not.
    assert_eq!(objects.create_pool(&small(), 4, memory), Ok(FIRST_ID));
    assert_eq!(
        objects.create_pool(&small(), 5, memory),
        Err(code(KernelError::InvalidSize))
    );
    // A refusal changes nothing: the next pool takes the next ids.
    assert_eq!(objects.create_pool(&small(), 1, memory), Ok(FIRST_ID + 4));
}

/// Two faults at once: the earlier check answers, for each neighbouring pair.
#[test]
fn the_create_checks_come_in_pr_s_order() {
    let (mut objects, memory) = tables(MEMORY, PAGE);
    let unbound = objects.create_memory(0, PAGE).unwrap();
    // No memory and a count of 0: NotFound.
    assert_eq!(
        objects.create_pool(&small(), 0, 99),
        Err(code(KernelError::NotFound))
    );
    // A count of 0 and a bad preference: OutOfResource.
    assert_eq!(
        objects.create_pool(&pool_preference(7, 2, 2), 0, memory),
        Err(code(KernelError::OutOfResource))
    );
    // A bad preference and an unbound memory: NotImplemented.
    assert_eq!(
        objects.create_pool(&pool_preference(7, 2, 2), 1, unbound),
        Err(code(KernelError::NotImplemented))
    );
    // An unbound memory too small: NotFound.
    assert_eq!(
        objects.create_pool(&small(), 2, unbound),
        Err(code(KernelError::NotFound))
    );
}

#[test]
fn ids_run_out_exactly_at_the_end() {
    let (fresh, memory) = tables(MEMORY, 4 * PAGE);
    // Two ids left: a pool of two takes them, and the next id is then 0 (run out).
    let mut objects = fresh.clone();
    objects.set_next_ids(2, u32::MAX - 1);
    assert_eq!(objects.create_pool(&small(), 2, memory), Ok(u32::MAX - 1));
    assert_eq!(objects.next_image, 0);
    assert_eq!(
        objects.create_pool(&small(), 1, memory),
        Err(code(KernelError::OutOfResource))
    );
    let native = crate::native::tests::laid_out(1, 16, 64, 6, 1, 1);
    assert_eq!(
        objects.create_image(&native, 1, memory),
        Err(code(KernelError::OutOfResource))
    );
    // One past: three images from u32::MAX - 1 pass the end.
    let mut objects = fresh;
    objects.set_next_ids(2, u32::MAX - 1);
    assert_eq!(
        objects.create_pool(&small(), 3, memory),
        Err(code(KernelError::OutOfResource))
    );
    assert_eq!(objects.next_image, u32::MAX - 1);
}

#[test]
fn pools_are_bounded_at_the_limit() {
    let (mut objects, memory) = tables(MEMORY, u64::from(MAX_POOL_IMAGES) * PAGE);
    // A pool of the most images is taken; one more image is NotImplemented, as PR.
    assert_eq!(
        objects.create_pool(&small(), MAX_POOL_IMAGES + 1, memory),
        Err(code(KernelError::NotImplemented))
    );
    let first = objects
        .create_pool(&small(), MAX_POOL_IMAGES, memory)
        .unwrap();
    objects.destroy_pool(first).unwrap();
    for _ in 0..MAX_POOLS {
        objects.create_pool(&small(), 1, memory).unwrap();
    }
    assert_eq!(objects.pools().count(), MAX_POOLS);
    assert_eq!(
        objects.create_pool(&small(), 1, memory),
        Err(code(KernelError::OutOfResource))
    );
}

#[test]
fn destroying_a_pool_needs_its_first_id() {
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let first = objects.create_pool(&small(), 2, memory).unwrap();
    // Its second image is not a pool's first id.
    assert_eq!(
        objects.destroy_pool(first + 1),
        Err(code(KernelError::NotFound))
    );
    assert_eq!(objects.destroy_pool(first), Ok(None));
    assert!(objects.pool_image(first).is_none());
    assert_eq!(
        objects.destroy_pool(first),
        Err(code(KernelError::NotFound))
    );
    // Its ids are not handed out again.
    assert_eq!(objects.create_pool(&small(), 1, memory), Ok(first + 2));
}

#[test]
fn a_memory_in_use_by_a_pool_is_retired_until_the_pool_goes() {
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let first = objects.create_pool(&small(), 1, memory).unwrap();
    let second = objects.create_pool(&small(), 1, memory).unwrap();
    // 112: retired, answered success, gone for every command.
    assert_eq!(objects.destroy_memory(memory), Ok(None));
    assert!(objects.memory(memory).is_none());
    assert_eq!(
        objects.create_pool(&small(), 1, memory),
        Err(code(KernelError::NotFound))
    );
    // The first pool goes: the second still uses the memory, which stays.
    assert_eq!(objects.destroy_pool(first), Ok(None));
    assert!(objects.memories().any(|(id, _)| id == memory));
    // The last pool goes: the memory goes with it, its lock to release.
    assert_eq!(
        objects.destroy_pool(second),
        Ok(Some(device_share_token(process(), MEMORY, 4 * PAGE)))
    );
    assert!(objects.memories().all(|(id, _)| id != memory));
}

#[test]
fn a_pool_and_an_image_together_keep_a_retired_memory() {
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let native = crate::native::tests::laid_out(1, 16, 64, 6, 1, 1);
    let image = objects.create_image(&native, 1, memory).unwrap();
    let pool = objects.create_pool(&small(), 1, memory).unwrap();
    assert_eq!(objects.destroy_memory(memory), Ok(None));
    // The image goes first: the pool keeps the memory.
    assert_eq!(objects.destroy_image(image), Ok(None));
    assert_eq!(
        objects.destroy_pool(pool),
        Ok(Some(device_share_token(process(), MEMORY, 4 * PAGE)))
    );
    // The other way round: the pool first, then the image releases it.
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let image = objects.create_image(&native, 1, memory).unwrap();
    let pool = objects.create_pool(&small(), 1, memory).unwrap();
    assert_eq!(objects.destroy_memory(memory), Ok(None));
    assert_eq!(objects.destroy_pool(pool), Ok(None));
    assert_eq!(
        objects.destroy_image(image),
        Ok(Some(device_share_token(process(), MEMORY, 4 * PAGE)))
    );
}

#[test]
fn the_nv12_descriptor_is_pr_s_record() {
    // Code 1, 640x360, two images: image 1 is 0x65400 bytes in.
    let (mut objects, memory) = tables(MEMORY, 2 * 0x65400);
    let first = objects
        .create_pool(&pool_preference(CODE_NV12, 640, 360), 2, memory)
        .unwrap();
    let descriptor = objects.video_descriptor(first + 1).unwrap();
    let extent = u64::from(0x27F_u32 | (0x167 << 16));
    let mut expected = [0; DESCRIPTOR_WORDS];
    expected[0] = (extent << 32) | 0xA;
    expected[1] = (276_480 << 32) | 0x65400;
    expected[2] = 768;
    expected[3] = (138_240 << 32) | (0x65400 + 276_480);
    expected[4] = 768;
    assert_eq!(descriptor, expected);
    // Code 4 (NX2's layout) in PR's record: pitch 640, chroma at 245760, plane sizes 640 * 360
    // and 640 * 180.
    let (mut objects, memory) = tables(MEMORY, 0x64000);
    let first = objects
        .create_pool(&pool_preference(CODE_NV12_NX2, 640, 360), 1, memory)
        .unwrap();
    let descriptor = objects.video_descriptor(first).unwrap();
    assert_eq!(descriptor[0], (extent << 32) | 0xA);
    assert_eq!(descriptor[1], 230_400 << 32);
    assert_eq!(descriptor[2], 640);
    assert_eq!(descriptor[3], (115_200 << 32) | 0x3_C000);
    assert_eq!(descriptor[4], 640);
    assert_eq!(descriptor[5..], [0; 11]);
}

#[test]
fn a_pitch_linear_rgba_descriptor_is_nx2_s() {
    // 100x50: pitch 448, image 57344; image 1 at 57344.
    for (code, kind) in [
        (CODE_BGRA_NX2, 9),
        (CODES_RGBA_NX2[0], 8),
        (CODES_RGBA_NX2[5], 8),
    ] {
        let (mut objects, memory) = tables(MEMORY, 2 * 57_344);
        let first = objects
            .create_pool(&pool_preference(code, 100, 50), 2, memory)
            .unwrap();
        let mut expected = [0; DESCRIPTOR_WORDS];
        expected[0] = kind;
        expected[1] = 57_344;
        expected[2] = 448;
        assert_eq!(
            objects.video_descriptor(first + 1),
            Ok(expected),
            "code {code}"
        );
    }
}

#[test]
fn a_block_linear_pool_has_no_cpu_descriptor() {
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let first = objects
        .create_pool(&pool_preference(CODE_RGBA_BLOCK, 1, 1), 1, memory)
        .unwrap();
    assert_eq!(
        objects.video_descriptor(first),
        Err(code(KernelError::NotImplemented))
    );
    // Its GPU descriptor is answered.
    assert!(objects.gpu_descriptor(first).is_ok());
    // An id no pool holds.
    assert_eq!(
        objects.video_descriptor(first + 1),
        Err(code(KernelError::NotFound))
    );
    assert_eq!(
        objects.gpu_descriptor(first + 1),
        Err(code(KernelError::NotFound))
    );
}

#[test]
fn an_offset_past_32_bits_is_refused() {
    // Code 2, 65536 x 16256: 4_261_412_864 bytes an image. Image 0 ends under 2^32; image 1
    // starts under it (its offset fits 428) but ends past it (961 refuses); image 2 starts past
    // it (both refuse).
    let image = 4_261_412_864_u64;
    let (mut objects, memory) = tables(MEMORY, 3 * image);
    let first = objects
        .create_pool(&pool_preference(CODE_BGRA_NX2, 65_536, 16_256), 3, memory)
        .unwrap();
    assert!(objects.video_descriptor(first).is_ok());
    assert_eq!(
        objects.video_descriptor(first + 1),
        Err(code(KernelError::InvalidSize))
    );
    assert!(objects.gpu_descriptor(first + 1).is_ok());
    assert_eq!(
        objects.gpu_descriptor(first + 2),
        Err(code(KernelError::InvalidSize))
    );
}

#[test]
fn the_gpu_descriptor_is_pr_s() {
    let (mut objects, memory) = tables(MEMORY, 2 * 0x65400);
    let first = objects
        .create_pool(&pool_preference(CODE_NV12, 640, 360), 2, memory)
        .unwrap();
    let mut expected = [0; DESCRIPTOR_WORDS];
    expected[0] = 1;
    expected[1] = u64::from(0x27F_u32 | (0x167 << 16));
    expected[4] = 0x1_0100_0001;
    expected[5] = 0x65400 | (0x65400 << 32);
    assert_eq!(objects.gpu_descriptor(first + 1), Ok(expected));
}

#[test]
fn a_mapping_answers_the_image_s_cpu_address() {
    let (mut objects, memory) = tables(MEMORY, 2 * 0x65400);
    let first = objects
        .create_pool(&pool_preference(CODE_NV12, 640, 360), 2, memory)
        .unwrap();
    let space = AsId(3);
    let (mapping, descriptor) = objects.map_gpu_image(space, first + 1).unwrap();
    assert_eq!(
        (mapping.address(), mapping.size(), mapping.references()),
        (MEMORY + 0x65400, 0x65400, 1)
    );
    // Word 5's low half cleared, as PR.
    assert_eq!(descriptor[5], 0x65400 << 32);
    // Again: counted, the same address.
    let (again, _) = objects.map_gpu_image(space, first + 1).unwrap();
    assert_eq!((again.address(), again.references()), (MEMORY + 0x65400, 2));
    // In another space: a mapping of its own.
    let (other, _) = objects.map_gpu_image(AsId(4), first + 1).unwrap();
    assert_eq!(other.references(), 1);
    assert_eq!(objects.gpu_mappings().count(), 2);
}

#[test]
fn an_unmap_needs_the_exact_range() {
    let (mut objects, memory) = tables(MEMORY, 2 * PAGE);
    let first = objects.create_pool(&small(), 2, memory).unwrap();
    let space = AsId(3);
    objects.map_gpu_image(space, first).unwrap();
    objects.map_gpu_image(space, first).unwrap();
    let last = MEMORY + PAGE - 1;
    for (space, image, address, end) in [
        (AsId(4), first, MEMORY, last),
        (space, first + 1, MEMORY, last),
        (space, first, MEMORY + 1, last),
        (space, first, MEMORY, last - 1),
        (space, first, MEMORY, last + 1),
        // A last address below the address: no wrap.
        (space, first, MEMORY, MEMORY - 1),
    ] {
        assert_eq!(
            objects.unmap_gpu_image(space, image, address, end),
            Err(code(KernelError::NotFound)),
            "{space:?} {image} {address:#x} {end:#x}"
        );
    }
    assert_eq!(objects.unmap_gpu_image(space, first, MEMORY, last), Ok(()));
    assert_eq!(objects.gpu_mappings().next().unwrap().1.references(), 1);
    assert_eq!(objects.unmap_gpu_image(space, first, MEMORY, last), Ok(()));
    assert_eq!(objects.gpu_mappings().count(), 0);
    assert_eq!(
        objects.unmap_gpu_image(space, first, MEMORY, last),
        Err(code(KernelError::NotFound))
    );
}

#[test]
fn a_mapping_count_stops_at_its_maximum() {
    let (mut objects, memory) = tables(MEMORY, PAGE);
    let first = objects.create_pool(&small(), 1, memory).unwrap();
    let space = AsId(3);
    objects.map_gpu_image(space, first).unwrap();
    objects.set_mapping_references(space, first, u32::MAX - 1);
    assert_eq!(
        objects.map_gpu_image(space, first).unwrap().0.references(),
        u32::MAX
    );
    assert_eq!(
        objects.map_gpu_image(space, first),
        Err(code(KernelError::OutOfResource))
    );
    assert_eq!(
        objects.gpu_mappings().next().unwrap().1.references(),
        u32::MAX
    );
}

#[test]
fn mappings_are_bounded_at_the_limit() {
    let (mut objects, memory) = tables(MEMORY, u64::from(MAX_POOL_IMAGES) * PAGE);
    let first = objects
        .create_pool(&small(), MAX_POOL_IMAGES, memory)
        .unwrap();
    let limit = u32::try_from(MAX_MAPPINGS).unwrap();
    for index in 0..limit {
        objects.map_gpu_image(AsId(1), first + index).unwrap();
    }
    assert_eq!(
        objects.map_gpu_image(AsId(2), first),
        Err(code(KernelError::OutOfResource))
    );
    // A mapping already made is still counted at the limit.
    assert!(objects.map_gpu_image(AsId(1), first).is_ok());
}

#[test]
fn destroying_a_pool_drops_its_mappings() {
    let (mut objects, memory) = tables(MEMORY, 4 * PAGE);
    let first = objects.create_pool(&small(), 2, memory).unwrap();
    let neighbour = objects.create_pool(&small(), 1, memory).unwrap();
    // The neighbour's first id is exactly the pool's end.
    assert_eq!(neighbour, first + 2);
    objects.map_gpu_image(AsId(1), first).unwrap();
    objects.map_gpu_image(AsId(1), first + 1).unwrap();
    objects.map_gpu_image(AsId(1), neighbour).unwrap();
    objects.destroy_pool(first).unwrap();
    let left: Vec<_> = objects.gpu_mappings().map(|(key, _)| key).collect();
    assert_eq!(left, [(AsId(1), neighbour)]);
}

// ---- Through the root ----

/// A GPU front that does nothing.
struct NoFront;

impl GpuVmFront for NoFront {
    fn create_address_space(&self, _: AsId, _: Arc<GpuVm>) {}
    fn update_vm(&self, _: AsId, _: VmUpdate, _: Option<Ticket>) {}
    fn barrier(&self, _: Ticket) {}
}

/// A root server with gsv's shared state, the title's memory mapped, and one root session.
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
        kernel.map(MEMORY, vec![0; usize::try_from(16 * PAGE).unwrap()]);
        let session = kernel.new_object();
        let queue = Arc::new(CompletionQueue::new());
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV));
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

    /// A memory of `size` made (111) and bound at [`MEMORY`] (114): its id.
    fn memory(&mut self, size: u64) -> u32 {
        let [low, high] = halves(size);
        let reply = self.send(&[0x7F, 3, 0, low, high]);
        let id = u32::from_le_bytes(reply[12..16].try_into().unwrap());
        let mut words = vec![0x82, 9, id];
        for value in [MEMORY, size, 0, 0] {
            words.extend(halves(value));
        }
        assert_eq!(self.send(&words), success(114, &[], &[]));
        id
    }

    /// 911: `0x39f, 18`, the preference, count, memory.
    fn create(&mut self, preference: &Preference, count: u32, memory: u32) -> Vec<u8> {
        let mut words = vec![0x39F, 18];
        words.extend(preference);
        words.extend([count, memory]);
        self.send(&words)
    }
}

/// `u64` words as reply data words.
fn data(values: &[u64]) -> Vec<u32> {
    values.iter().flat_map(|&value| halves(value)).collect()
}

#[test]
fn calculate_answers_the_size_and_alignment() {
    let mut rig = Rig::new();
    // 991: `0x3ef, 17`, the preference, count.
    let mut words = vec![0x3EF, 17];
    words.extend(pool_preference(CODE_NV12, 640, 360));
    words.push(2);
    assert_eq!(
        rig.send(&words),
        success(991, &[], &data(&[2 * 0x65400, 4096]))
    );
    let mut words = vec![0x3EF, 17];
    words.extend(pool_preference(CODE_NV12_NX2, 640, 360));
    words.push(1);
    // NX2's row 96 answer, with NX2's alignment.
    assert_eq!(
        rig.send(&words),
        success(991, &[], &data(&[0x64000, 0x2000]))
    );
    let mut words = vec![0x3EF, 17];
    words.extend(pool_preference(7, 2, 2));
    words.push(1);
    assert_eq!(
        rig.send(&words),
        result_only(991, code(KernelError::NotImplemented))
    );
    // A count missing: the request is refused before anything is read into a layout.
    let mut words = vec![0x3EF, 16];
    words.extend(small());
    assert_eq!(rig.send(&words), result_only(991, INVALID_CMIF_REQUEST));
}

#[test]
fn create_query_and_destroy_through_the_root() {
    let mut rig = Rig::new();
    let memory = rig.memory(2 * PAGE);
    let reply = rig.create(&small(), 2, memory);
    assert_eq!(reply, success(911, &[], &[FIRST_ID]));
    // 961: `0x3d1, 1`, image.
    let mut expected = [0; DESCRIPTOR_WORDS];
    expected[0] = (u64::from(1_u32 | (1 << 16)) << 32) | 0xA;
    expected[1] = (512 << 32) | 4096;
    expected[2] = 256;
    expected[3] = (256 << 32) | (4096 + 1024);
    expected[4] = 256;
    assert_eq!(
        rig.send(&[0x3D1, 1, FIRST_ID + 1]),
        success(961, &[], &data(&expected))
    );
    // 428: `0x1bc, 1`, image: two u64 zeros, the descriptor, a u32 zero.
    let mut gpu = vec![0, 0];
    gpu.extend([
        1,
        u64::from(1_u32 | (1 << 16)),
        0,
        0,
        0x1_0100_0001,
        4096 | (4096 << 32),
    ]);
    gpu.extend([0; 10]);
    let mut gpu = data(&gpu);
    gpu.push(0);
    assert_eq!(rig.send(&[0x1BC, 1, FIRST_ID + 1]), success(428, &[], &gpu));
    // 912: `0x3a0, 1`, first id.
    assert_eq!(rig.send(&[0x3A0, 1, FIRST_ID]), success(912, &[], &[]));
    assert_eq!(
        rig.send(&[0x3D1, 1, FIRST_ID]),
        result_only(961, code(KernelError::NotFound))
    );
    assert_eq!(
        rig.send(&[0x3A0, 1, FIRST_ID]),
        result_only(912, code(KernelError::NotFound))
    );
}

#[test]
fn a_destroy_releases_a_retired_memory_s_lock() {
    let mut rig = Rig::new();
    let memory = rig.memory(PAGE);
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);
    assert_eq!(
        rig.create(&small(), 1, memory),
        success(911, &[], &[FIRST_ID])
    );
    // 112: retired; the lock stays.
    assert_eq!(rig.send(&[0x80, 1, memory]), success(112, &[], &[]));
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 1);
    // 912: the last user goes; the lock is released.
    assert_eq!(rig.send(&[0x3A0, 1, FIRST_ID]), success(912, &[], &[]));
    assert_eq!(rig.kernel.device_locks(MEMORY, PAGE), 0);
}

#[test]
fn a_mapping_needs_an_address_space() {
    let mut rig = Rig::new();
    let memory = rig.memory(PAGE);
    rig.create(&small(), 1, memory);
    // 426: `0x1ba, 4`, image, 1, address space, mode.
    assert_eq!(
        rig.send(&[0x1BA, 4, FIRST_ID, 1, 0, 1]),
        result_only(426, code(KernelError::NotFound))
    );
    let space = rig
        .shared
        .lock()
        .gpu()
        .create_address_space(&NoFront)
        .unwrap();
    let map = |mode: u32, word_3: u32| [0x1BA, 4, FIRST_ID, word_3, space.0, mode];
    // Word 3 not 1, or a mode byte neither 1 nor 3: InvalidCombination, before the space.
    for (mode, word_3) in [(1, 0), (1, 2), (0, 1), (2, 1), (4, 1), (0x101, 1)] {
        let refused = rig.send(&map(mode, word_3));
        if mode == 0x101 {
            // The mode is the low byte: 0x101 is mode 1.
            assert!(refused[8..12] == [0, 0, 0, 0], "{mode:#x}");
            continue;
        }
        assert_eq!(
            refused,
            result_only(426, code(KernelError::InvalidCombination)),
            "mode {mode}, word 3 {word_3}"
        );
    }
    // The mapping: the address, the last address, the descriptor with word 5's low half
    // cleared, a u32 zero.
    let mut expected = vec![MEMORY, MEMORY + PAGE - 1];
    expected.extend([
        1,
        u64::from(1_u32 | (1 << 16)),
        0,
        0,
        0x1_0100_0001,
        4096 << 32,
    ]);
    expected.extend([0; 10]);
    let mut expected = data(&expected);
    expected.push(0);
    assert_eq!(rig.send(&map(3, 1)), success(426, &[], &expected));
    // 427: `0x1bb, 6`, image, space, address, last address: once for each of the two maps.
    let mut unmap = vec![0x1BB, 6, FIRST_ID, space.0];
    unmap.extend(halves(MEMORY));
    unmap.extend(halves(MEMORY + PAGE - 1));
    assert_eq!(rig.send(&unmap), success(427, &[], &[]));
    assert_eq!(rig.send(&unmap), success(427, &[], &[]));
    assert_eq!(
        rig.send(&unmap),
        result_only(427, code(KernelError::NotFound))
    );
}

#[test]
fn pool_commands_need_no_901() {
    // No 4, 11, 101 or 901 sent: each answers from the tables (decision 7).
    let mut rig = Rig::new();
    let memory = rig.memory(PAGE);
    assert_eq!(
        rig.create(&small(), 1, memory),
        success(911, &[], &[FIRST_ID])
    );
    assert_eq!(rig.shared.lock().native_objects().pools().count(), 1);
}

#[test]
fn every_word_is_read_before_anything_changes() {
    let mut rig = Rig::new();
    let memory = rig.memory(PAGE);
    // 911 without its memory word.
    let mut words = vec![0x39F, 17];
    words.extend(small());
    words.push(1);
    assert_eq!(rig.send(&words), result_only(911, INVALID_CMIF_REQUEST));
    assert_eq!(rig.shared.lock().native_objects().pools().count(), 0);
    // A wider one is read.
    let mut words = vec![0x39F, 19];
    words.extend(small());
    words.extend([1, memory, 0xDEAD]);
    assert_eq!(rig.send(&words), success(911, &[], &[FIRST_ID]));
    // 426 without its mode word.
    assert_eq!(
        rig.send(&[0x1BA, 3, FIRST_ID, 1, 0]),
        result_only(426, INVALID_CMIF_REQUEST)
    );
    assert_eq!(MAX_IMAGES, MAX_MAPPINGS);
}
