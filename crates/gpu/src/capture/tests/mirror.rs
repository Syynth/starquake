//! The semaphore mirror query through the recording and replay seams: passed on to the wrapped
//! memory and not recorded, and answered `None` on replay (the capture module's docs, "Not
//! built, and departures").

use std::sync::Arc;

use starquake_memory::FlatMemory;

use super::*;
use crate::capture::drive::Discard;
use crate::capture::replay::seams;
use crate::capture::{Recorder, RecordingMemory, Thread};
use crate::memory::{GpuMemory, VmMemory};
use crate::syncpoint::Syncpoints;
use crate::vm::{GpuVm, SEMAPHORE_REGISTRATION_BYTES, SemaphoreRegistration};

#[test]
fn the_recording_memory_passes_the_mirror_on_and_the_replay_answers_none() {
    let vm = Arc::new(GpuVm::new());
    let registration = SemaphoreRegistration::new(
        PageRange::new(GPU_BASE, SEMAPHORE_REGISTRATION_BYTES).unwrap(),
    )
    .unwrap();
    vm.add_semaphore_registration(registration);
    let inner = Arc::new(VmMemory::new(
        vm,
        Arc::new(FlatMemory::new(CPU_BASE, 0)),
        Arc::new(Syncpoints::new(Arc::new(Discard))),
    ));
    let recorder = Arc::new(Recorder::new(1 << 10));
    let back = RecordingMemory::new(inner, Arc::clone(&recorder), Thread::Back, SPACE);
    let release = GPU_BASE + PAGE_SIZE + 0x10;
    assert_eq!(back.semaphore_mirror(release), Some(GPU_BASE + 0x10));
    let taken = recorder.take().unwrap();
    assert_eq!(taken.front, []);
    assert_eq!(taken.back, []);

    let log = seams::Back::new(vec![]);
    assert_eq!(log.memory(SPACE).semaphore_mirror(release), None);
    // The query took nothing from the log and did not diverge.
    assert_eq!(log.next(), 0);
    assert_eq!(log.check(), Ok(()));
}
