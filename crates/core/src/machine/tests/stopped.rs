//! What a run reports about where the guest was when it stopped: a crash's backtrace, walked
//! in the block the kernel's memory map has the crashing thread's stack in, and a timed-out
//! run's threads, through the whole run (`run`).

use starquake_kernel::sched::FIRST_THREAD_ID;

use super::*;
use crate::crash::{Backtrace, BacktraceEnd, FRAME_POINTER_REGISTER, Frame, LINK_REGISTER};
use crate::report::ThreadPc;

/// The guest's flat memory, and a second flat memory over its main thread's stack, which the
/// first does not reach: what a crash's backtrace reads.
struct WithStack<'a> {
    image: &'a FlatMemory,
    stack: FlatMemory,
}

impl<'a> WithStack<'a> {
    /// `guest`'s memory with its stack mapping backed.
    fn new(guest: &'a Guest) -> Self {
        let start = guest.start_of("stack");
        let size = usize::try_from(guest.placed.stack_top - start).unwrap();
        Self {
            image: &guest.memory,
            stack: FlatMemory::new(start, size),
        }
    }
}

impl GuestMemory for WithStack<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.image
            .read(va, buf)
            .or_else(|_| self.stack.read(va, buf))
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.image
            .write(va, data)
            .or_else(|_| self.stack.write(va, data))
    }
}

impl<V> GuestSpace<V> for WithStack<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
    }
}

/// Writes a frame record at `at` in `memory`: the caller's frame pointer, then the return
/// address at [`crate::crash::FRAME_RECORD_RETURN`].
fn put_record(memory: &dyn GuestMemory, at: u64, caller: u64, ret: u64) {
    memory.write(at, &caller.to_le_bytes()).unwrap();
    let ret_at = at + u64::try_from(crate::crash::FRAME_RECORD_RETURN).unwrap();
    memory.write(ret_at, &ret.to_le_bytes()).unwrap();
}

/// A run that crashes on an undefined instruction at `entry + 0x40`, with `sp` and a frame
/// pointer `fp`, its link register at `entry + 0x80`.
fn crash_with(guest: &Guest, space: &WithStack<'_>, sp: u64, fp: u64) -> RunReport {
    let entry = guest.placed.entry;
    let pc = entry + 0x40;
    let vcpu = ScriptedVcpu::new().then(
        move |regs| {
            regs.pc = pc;
            regs.sp = sp;
            regs.x[LINK_REGISTER] = entry + 0x80;
            regs.x[FRAME_POINTER_REGISTER] = fp;
        },
        Exit::Undefined {
            pc,
            syndrome: Syndrome::undefined(),
        },
    );
    let mut vcpu = vcpu;
    run(
        vcpus(&mut vcpu),
        guest.program(space),
        &RunOptions::default(),
        Instant::now(),
        timebase(),
    )
}

/// The location of `address` in the guest's modules.
fn location(guest: &Guest, address: u64) -> Option<crate::crash::Location> {
    crate::crash::Location::of(address, &guest.placed.modules)
}

#[test]
fn a_crash_walks_the_frame_records_in_the_kernel_block_that_holds_its_stack() {
    let guest = Guest::new(&programs::hello().unwrap());
    let space = WithStack::new(&guest);
    let (bottom, top) = (guest.start_of("stack"), guest.placed.stack_top);
    let entry = guest.placed.entry;
    // Two records near the top of the stack: the inner one returns into the program, the
    // outer one into no module and ends the chain.
    let (inner, outer) = (top - 0x40, top - 0x20);
    put_record(&space, inner, outer, entry + 0xc4);
    put_record(&space, outer, 0, 0x10);
    let report = crash_with(&guest, &space, top - 0x60, inner);

    let crash = report.crash().expect("a crash report");
    let frame = |address: u64, record: Option<u64>| Frame {
        address,
        location: location(&guest, address),
        record,
    };
    assert_eq!(
        crash.backtrace,
        Some(Backtrace {
            // The kernel's block of the main thread's stack: the loader's stack mapping.
            stack: Some(Mapping {
                name: "Stack".to_owned(),
                start: bottom,
                end: top,
            }),
            frames: vec![
                frame(entry + 0x40, None),
                frame(entry + 0x80, None),
                frame(entry + 0xc4, Some(inner)),
                frame(0x10, Some(outer)),
            ],
            end: BacktraceEnd::NullFramePointer,
        })
    );
    assert!(
        crash.backtrace.as_ref().unwrap().frames[2]
            .location
            .is_some()
    );
    assert_eq!(crash.backtrace.as_ref().unwrap().frames[3].location, None);
    // A crash leaves no threads list: that is a timeout's.
    assert_eq!(report.threads, []);
}

#[test]
fn a_crash_whose_stack_pointer_left_its_stack_walks_the_block_of_its_frame_pointer() {
    let program = programs::hello().unwrap();
    let probe = Guest::new(&program);
    let (bottom, top) = (probe.start_of("stack"), probe.placed.stack_top);
    // One page below the stack's start, in the free guard pages the loader keeps there (a stack
    // overflow), and exactly at the stack's top, the end of its block (an empty stack).
    for sp in [bottom - crate::image::PAGE, top] {
        // A guest runs once.
        let guest = Guest::new(&program);
        let space = WithStack::new(&guest);
        let entry = guest.placed.entry;
        let (inner, outer) = (top - 0x40, top - 0x20);
        put_record(&space, inner, outer, entry + 0xc4);
        put_record(&space, outer, 0, 0x10);
        let report = crash_with(&guest, &space, sp, inner);
        let backtrace = report.crash().unwrap().backtrace.clone().unwrap();
        assert_eq!(
            backtrace.stack,
            Some(Mapping {
                name: "Stack".to_owned(),
                start: bottom,
                end: top,
            }),
            "{sp:#x}"
        );
        let records: Vec<_> = backtrace
            .frames
            .iter()
            .map(|frame| (frame.address, frame.record))
            .collect();
        assert_eq!(
            records,
            [
                (entry + 0x40, None),
                (entry + 0x80, None),
                (entry + 0xc4, Some(inner)),
                (0x10, Some(outer)),
            ],
            "{sp:#x}"
        );
        assert_eq!(backtrace.end, BacktraceEnd::NullFramePointer, "{sp:#x}");
    }
}

#[test]
fn a_crash_whose_stack_and_frame_pointers_are_in_no_writable_block_has_no_stack() {
    let program = programs::hello().unwrap();
    let entry = Guest::new(&program).placed.entry;
    // In the program's text, which the guest may read and execute but not write; in nothing
    // the kernel maps; and above the address space.
    let outside = [entry, 0x1000, u64::MAX - 0xf];
    for (sp, fp) in outside.iter().flat_map(|&sp| outside.map(|fp| (sp, fp))) {
        // A guest runs once.
        let guest = Guest::new(&program);
        let space = WithStack::new(&guest);
        let report = crash_with(&guest, &space, sp, fp);
        let backtrace = report.crash().unwrap().backtrace.clone().unwrap();
        assert_eq!(backtrace.stack, None, "{sp:#x} {fp:#x}");
        assert_eq!(backtrace.frames.len(), 2, "{sp:#x} {fp:#x}");
        assert_eq!(
            backtrace.end,
            BacktraceEnd::OutsideStack { frame_pointer: fp },
            "{sp:#x} {fp:#x}"
        );
    }
}

#[test]
fn a_timed_out_run_reports_where_its_thread_spins() {
    // The main thread spins at its entry. Whether or not its core has loaded it when the
    // timeout passes, its program counter is the entry; its core has loaded it long before.
    let guest = Guest::new(&programs::endless_loop().unwrap());
    let timeout = Duration::from_millis(100);
    let options = RunOptions {
        timeout: Some(timeout),
        command_line: Vec::new(),
        sd_card: None,
        romfs: None,
    };
    let vcpu = ScriptedVcpu::new().then_spin();
    let interrupters = Interrupters::default();
    lock(&interrupters).push(vcpu.interrupter());
    let report = bounded(&interrupters, || guest.run(vcpu, &options));
    assert_eq!(report.exit, ExitReason::Timeout);
    let entry = guest.placed.entry;
    assert_eq!(
        report.threads,
        [ThreadPc::new(
            FIRST_THREAD_ID,
            ThreadState::Runnable,
            Some(CORE),
            entry,
            &guest.placed.modules,
        )]
    );
    assert_eq!(report.threads[0].pc_location, location(&guest, entry));
    assert!(report.threads[0].pc_location.is_some());
    assert_eq!(report.crash(), None);
}
