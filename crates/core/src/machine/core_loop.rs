//! One core: a host thread that owns a vCPU and runs the guest threads the scheduler gives it
//! (see the module docs of `machine`).
//!
//! The vCPU's registers are the running thread's while it runs. They go back into the thread's
//! saved context only when the core switches away from it
//! ([`Dispatcher::switch`](starquake_kernel::dispatch::Dispatcher::switch), under the kernel
//! lock, so no other core can load a stale copy), and not after a call that began a wait,
//! which saved them itself: then the saved context is the newer, and a core that keeps the
//! thread (it was woken before its core rescheduled) loads it again, with the result its wake
//! wrote. A core reschedules after every system call and every interrupt; an abort that only
//! lacked backing resumes the thread without asking.

use starquake_cpu::{Exit, GuestRegs, Syndrome, Vcpu};
use starquake_kernel::dispatch::{Caller, Held, Outcome, Switch};
use starquake_kernel::sched::CoreId;
use starquake_memory::Access;
use starquake_trace::{EventKind, FaultAction, HostClock, LifecyclePhase, ThreadId};

use super::cores::Parked;
use super::{Ending, LogSink, Machine, PROCESS_ID, UNDEFINED, WAITING, lock};
use crate::crash::Exception;
use crate::space::Touch;

/// The thread a core runs.
#[derive(Clone, Copy)]
struct Running {
    /// As the event stream names it.
    guest_id: ThreadId,
}

/// What a step left the core doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stepped {
    /// It runs a thread, or will ask the scheduler again at its next step.
    Running,
    /// It has nothing to run: it parks until the scheduler wakes it.
    Parked,
}

/// Runs `core` on `vcpu` until the run stops, parking when it has nothing to run, and stops
/// the run if this core ends it.
pub(super) fn run<V: Vcpu, C: HostClock + Clone>(
    machine: &Machine<'_, V, C>,
    core: CoreId,
    vcpu: &mut V,
) {
    machine.cores.attach(core, vcpu.interrupter());
    let mut this = Core::new(machine, core);
    while !machine.is_stopped() {
        match this.step(vcpu) {
            Ok(Stepped::Running) => {}
            Ok(Stepped::Parked) => {
                let parked = machine
                    .cores
                    .park(core, || machine.dispatcher.is_stalled());
                match parked {
                    Parked::Woken | Parked::Stopped => {}
                    Parked::Stalled => machine.stop(Ending::Failed(WAITING.to_owned())),
                }
            }
            Err(ending) => machine.stop(ending),
        }
    }
}

/// What a core's vCPU holds of the thread it runs, against the thread's saved context: the
/// owned form of [`Held`].
enum Holds {
    Same,
    Newer(GuestRegs),
    Older,
}

impl Holds {
    fn held(&self) -> Held<'_> {
        match self {
            Self::Same => Held::Same,
            Self::Newer(regs) => Held::Newer(regs),
            Self::Older => Held::Older,
        }
    }
}

/// A core's state between steps.
pub(super) struct Core<'m, 'a, V: Vcpu, C> {
    machine: &'m Machine<'a, V, C>,
    core: CoreId,
    running: Option<Running>,
    /// What the vCPU holds of the running thread, against its saved context.
    holds: Holds,
    /// Whether to ask the scheduler before the guest runs again.
    must_switch: bool,
}

impl<'m, 'a, V: Vcpu, C: HostClock + Clone> Core<'m, 'a, V, C> {
    /// `core` of `machine`, which has not asked the scheduler anything yet.
    pub(super) fn new(machine: &'m Machine<'a, V, C>, core: CoreId) -> Self {
        Self {
            machine,
            core,
            running: None,
            holds: Holds::Same,
            must_switch: true,
        }
    }

    fn failed(error: &starquake_cpu::Error) -> Ending {
        Ending::Failed(format!("the vCPU failed: {error}"))
    }

    /// One turn: reschedule if needed, then run the guest to its next exit and handle it. A
    /// core the scheduler gives nothing says so, and asks again at its next step.
    pub(super) fn step(&mut self, vcpu: &mut V) -> Result<Stepped, Ending> {
        if self.must_switch && !self.switch(vcpu)? {
            return Ok(Stepped::Parked);
        }
        self.must_switch = false;
        let Some(running) = self.running else {
            self.must_switch = true;
            return Ok(Stepped::Running);
        };
        let exit = vcpu.run().map_err(|error| Self::failed(&error))?;
        let handled = match exit {
            Exit::Svc(number) => self.svc(vcpu, running, number),
            Exit::DataAbort {
                address,
                write,
                syndrome,
            } => {
                let access = if write { Access::Write } else { Access::Read };
                self.aborted(vcpu, running, address, access, syndrome)
            }
            Exit::InstructionAbort { address, syndrome } => {
                self.aborted(vcpu, running, address, Access::Execute, syndrome)
            }
            Exit::Undefined { syndrome, .. } => Err(crashed(vcpu, UNDEFINED, syndrome)),
            Exit::Interrupted => {
                if self.machine.is_stopped() {
                    return Ok(Stepped::Running);
                }
                // Something changed elsewhere (a wake, a rotation): the scheduler says what.
                self.holds = Holds::Newer(vcpu.regs().map_err(|error| Self::failed(&error))?);
                self.must_switch = true;
                Ok(())
            }
        };
        handled.map(|()| Stepped::Running)
    }

    /// Asks the scheduler what this core runs now and loads it: `false` when it has nothing.
    fn switch(&mut self, vcpu: &mut V) -> Result<bool, Ending> {
        let machine = self.machine;
        let mut signals = machine.signals();
        let switch = machine
            .dispatcher
            .switch(self.core, self.holds.held(), &mut signals)
            .map_err(|error| Ending::Failed(format!("the scheduler failed: {error}")))?;
        match switch {
            Switch::Stay if self.running.is_none() => Err(Ending::Failed(
                "the scheduler kept a core on a thread it never loaded".to_owned(),
            )),
            Switch::Stay => Ok(true),
            Switch::Load(loaded) => {
                vcpu.set_regs(&loaded.context)
                    .map_err(|error| Self::failed(&error))?;
                // Whatever the address space owes its vCPUs is run before this thread runs here.
                machine
                    .space
                    .settle(vcpu)
                    .map_err(|error| Self::failed(&error))?;
                self.running = Some(Running {
                    guest_id: loaded.guest_id,
                });
                self.holds = Holds::Same;
                Ok(true)
            }
            Switch::Park => {
                self.running = None;
                self.holds = Holds::Same;
                Ok(false)
            }
        }
    }

    /// A system call from the running thread.
    fn svc(&mut self, vcpu: &mut V, running: Running, number: u16) -> Result<(), Ending> {
        let machine = self.machine;
        let mut regs = vcpu.regs().map_err(|error| Self::failed(&error))?;
        let mut debug = LogSink {
            session: machine.session,
            log: &machine.log,
            core: self.core.index(),
            thread: running.guest_id,
        };
        let record = machine.dispatcher.dispatch(
            number,
            &mut regs,
            &mut Caller {
                core: self.core.index(),
                thread: running.guest_id,
                env: &**machine.env,
                memory: machine.session.memory,
                debug: &mut debug,
                recorder: &machine.session.recorder,
            },
        );
        // PROVISIONAL: nothing keeps the environment's recording (see the module docs), and
        // every dispatch adds a clock reading to it.
        drop(machine.env.take_recording());
        lock(&machine.tally).count(&record);
        // The call may have unmapped or reprotected memory. What the vCPUs have cached about it
        // must go before the guest runs again, also when the call failed part-way.
        machine
            .space
            .settle(vcpu)
            .map_err(|error| Self::failed(&error))?;
        self.must_switch = true;
        match record.outcome {
            Outcome::Completed | Outcome::NotImplemented => {
                vcpu.set_regs(&regs).map_err(|error| Self::failed(&error))?;
                self.holds = Holds::Newer(regs);
                Ok(())
            }
            Outcome::Waiting => {
                // The call saved the thread's registers, where its wake writes its result; its
                // wait may have a timeout the timer thread has not seen.
                self.holds = Holds::Older;
                machine.bell.ring();
                Ok(())
            }
            Outcome::ThreadExited => {
                self.holds = Holds::Same;
                Ok(())
            }
            Outcome::ProcessExited => {
                machine.session.record_on(
                    self.core.index(),
                    running.guest_id,
                    EventKind::Lifecycle {
                        phase: LifecyclePhase::ProcessExit,
                        subject: PROCESS_ID,
                    },
                );
                Err(Ending::Exited)
            }
            Outcome::EnvFailed(error) => Err(Ending::Failed(format!(
                "the environment failed during svc {number:#x}: {error}"
            ))),
        }
    }

    /// What an abort on an `access` to `address` comes to: the guest resumes when the page only
    /// lacked backing and now has it; otherwise the run ends. `syndrome` is the abort's, for
    /// the crash report.
    fn aborted(
        &mut self,
        vcpu: &V,
        running: Running,
        address: u64,
        access: Access,
        syndrome: Syndrome,
    ) -> Result<(), Ending> {
        match self.machine.space.touch(address, access) {
            Touch::Backed => Ok(()),
            Touch::Refused => {
                // The event stream has one kind of fault; a fetch is recorded as a read, as it
                // was when the backend reported fetch faults as data aborts.
                let (exception, write) = match access {
                    Access::Execute => (Exception::InstructionAbort { address }, false),
                    Access::Write => (
                        Exception::DataAbort {
                            address,
                            write: true,
                        },
                        true,
                    ),
                    Access::Read => (
                        Exception::DataAbort {
                            address,
                            write: false,
                        },
                        false,
                    ),
                };
                self.machine.session.record_on(
                    self.core.index(),
                    running.guest_id,
                    EventKind::Fault {
                        address,
                        write,
                        action: FaultAction::Fatal,
                    },
                );
                Err(crashed(vcpu, exception, syndrome))
            }
            Touch::Failed(error) => Err(Ending::Failed(format!(
                "guest memory at {address:#x} could not be backed: {error}"
            ))),
        }
    }
}

/// The run ends as a crash with `exception` and its `syndrome`, with the guest's registers as
/// they are now.
fn crashed<V: Vcpu>(vcpu: &V, exception: Exception, syndrome: Syndrome) -> Ending {
    match vcpu.regs() {
        Ok(regs) => Ending::Crashed(exception, syndrome, Box::new(regs)),
        Err(error) => Ending::Failed(format!("the vCPU failed: {error}")),
    }
}
