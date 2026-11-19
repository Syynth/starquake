//! The fence path over the production memory (slice G22): a release to the syncpoint shim is an
//! increment, and a sparse-region destroy frees the GPU-only store, each only once its list has
//! completed.

use super::*;
use crate::interface::RegionId;
use crate::memory::{SHIM_STRIDE, VmMemory};
use crate::vm::SHIM_PAGES;

/// Where the production memory's shim and sparse region sit, above 32 bits.
const SHIM: u64 = 0x2_0000_0000_0000;
const SPARSE: u64 = SHIM + 0x10 * PAGE_SIZE;
const SPARSE_PAGES: u64 = 2;
const REGION: RegionId = RegionId(0x2a);

/// A table with the shim and two pages of [`REGION`], and the production memory over it, its
/// shim reading the rig's syncpoints.
fn production(r: &Rig) -> (Arc<GpuVm>, Arc<VmMemory<FlatMemory>>) {
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(SHIM, SHIM_PAGES * PAGE_SIZE).unwrap(),
        Target::SyncpointShim,
        Kind(0),
    )
    .unwrap();
    vm.map(
        PageRange::new(SPARSE, SPARSE_PAGES * PAGE_SIZE).unwrap(),
        Target::GpuOnly {
            region: REGION,
            offset: 0,
        },
        Kind(0),
    )
    .unwrap();
    let guest = Arc::new(FlatMemory::new(CPU_BASE, 0));
    let mem = Arc::new(VmMemory::new(
        Arc::clone(&vm),
        guest,
        Arc::clone(&r.syncpoints),
    ));
    (vm, mem)
}

#[test]
fn a_release_to_the_shim_increments_its_syncpoint_after_its_list_completes() {
    let mut r = rig();
    let (_vm, mem) = production(&r);
    let target = syncpoint(9);
    r.syncpoints.watch(target);
    r.fences.record(command(1)).unwrap();
    let release = FencedAction::Release {
        memory: Arc::<VmMemory<FlatMemory>>::clone(&mem),
        space: SPACE,
        release: four(SHIM + 9 * SHIM_STRIDE, 0xdead_beef),
    };
    assert_eq!(r.fences.attach(release), Ok(Attached::Open));
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();

    // In flight: not incremented, nothing pushed.
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(r.syncpoints.read(target), 0);
    assert_eq!(r.pushed.all(), vec![]);

    r.backend.finish(fence);
    assert_eq!(r.fences.complete(), released(1));
    assert_eq!(r.syncpoints.read(target), 1);
    assert_eq!(
        r.pushed.all(),
        vec![GpuCompletion::SyncpointChanged {
            id: target,
            value: 1
        }]
    );
    // The shim reads the new value back through the same memory.
    let mut word = [0; 4];
    mem.read(SHIM + 9 * SHIM_STRIDE, &mut word).unwrap();
    assert_eq!(u32::from_le_bytes(word), 1);
}

#[test]
fn a_release_the_shim_cannot_take_is_reported_and_increments_nothing() {
    let mut r = rig();
    let (_vm, mem) = production(&r);
    let address = SHIM + 2;
    let release = FencedAction::Release {
        memory: Arc::<VmMemory<FlatMemory>>::clone(&mem),
        space: SPACE,
        release: four(address, 1),
    };
    // Nothing in flight: performed at once, and listed as failed with the shim's fault.
    assert_eq!(
        r.fences.attach(release),
        Ok(Attached::Performed(Report {
            failed: vec![ReleaseFailure {
                address,
                fault: GpuFault::ShimWrite { va: address },
            }],
            ..Report::default()
        }))
    );
    assert_eq!(r.fences.counters().failed_releases, 1);
    assert_eq!(r.syncpoints.read(syncpoint(0)), 0);
}

#[test]
fn a_sparse_region_destroy_frees_its_store_only_once_its_marker_is_performed() {
    let mut r = rig();
    let (vm, mem) = production(&r);
    mem.write(SPARSE + PAGE_SIZE + 4, &[1, 2, 3]).unwrap();
    assert_eq!(vm.gpu_only().resident_pages_of(REGION), 1);

    r.fences.record(command(1)).unwrap();
    let marker = FencedAction::Marker {
        token: MarkerToken(3),
        ordered: Some(OrderedUpdate {
            change: Some(VmChange {
                vm: Arc::clone(&vm),
                update: VmUpdate::DestroySparseRegion {
                    range: PageRange::new(SPARSE, SPARSE_PAGES * PAGE_SIZE).unwrap(),
                    region: REGION,
                },
            }),
            ticket: Some(Ticket(4)),
        }),
    };
    assert_eq!(r.fences.attach(marker), Ok(Attached::Open));
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();

    // In flight: the store keeps the page, and the memory still reads it.
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(vm.gpu_only().resident_pages_of(REGION), 1);
    let mut bytes = [0; 3];
    mem.read(SPARSE + PAGE_SIZE + 4, &mut bytes).unwrap();
    assert_eq!(bytes, [1, 2, 3]);

    r.backend.finish(fence);
    let report = r.fences.complete();
    assert_eq!(report.markers, vec![MarkerToken(3)]);
    assert_eq!(vm.gpu_only().resident_pages_of(REGION), 0);
    assert_eq!(
        mem.read(SPARSE + PAGE_SIZE + 4, &mut bytes),
        Err(GpuFault::Unmapped {
            va: SPARSE + PAGE_SIZE + 4
        })
    );
    assert_eq!(
        r.pushed.all(),
        vec![GpuCompletion::GpuOrdered { ticket: Ticket(4) }]
    );
}
