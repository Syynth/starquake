//! A [`Vcpu`] that plays back scripted exits, so code that drives a vCPU (the core loop, the
//! kernel's exit handling) can be tested on any host, with no hypervisor.
//!
//! Compiled for this crate's own tests and for crates that enable the `test-support` feature
//! (`docs/specs/testing.md`, "Shared test support": each crate exports its own fakes).
//!
//! A script is a list of steps. Each [`Vcpu::run`] plays the next one: the scripted "guest"
//! changes its registers with a closure (which also sees what the host wrote since the last
//! exit) and stops with an [`Exit`]. A [spin](ScriptedVcpu::then_spin) step is a guest that never
//! stops by itself: `run` blocks until the vCPU's [`Interrupter`] fires. Running past the end of
//! the script is an error, not a panic.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use crate::{Error, Exit, GuestRegs, Interrupter, Vcpu};

/// What the scripted guest does to its registers before an exit.
type Guest = Box<dyn FnOnce(&mut GuestRegs)>;

enum Step {
    /// The guest runs `guest` and stops with `exit`.
    Exit { guest: Guest, exit: Exit },
    /// The guest runs until it is interrupted.
    Spin,
}

/// Whether an interrupt is pending, and the condition a spinning vCPU waits on.
#[derive(Debug, Default)]
struct Signal {
    pending: Mutex<bool>,
    raised: Condvar,
}

impl Signal {
    fn lock(&self) -> MutexGuard<'_, bool> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Stops a [`ScriptedVcpu`] from another thread.
#[derive(Debug, Clone)]
pub struct ScriptedInterrupter {
    signal: Arc<Signal>,
}

impl Interrupter for ScriptedInterrupter {
    fn interrupt(&self) -> Result<(), Error> {
        *self.signal.lock() = true;
        self.signal.raised.notify_all();
        Ok(())
    }
}

/// A vCPU whose guest is a script (see the module docs).
#[derive(Default)]
pub struct ScriptedVcpu {
    regs: GuestRegs,
    steps: VecDeque<Step>,
    signal: Arc<Signal>,
}

impl ScriptedVcpu {
    /// A vCPU with zeroed registers and an empty script.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a step: the guest applies `guest` to its registers, then stops with `exit`.
    #[must_use]
    pub fn then(mut self, guest: impl FnOnce(&mut GuestRegs) + 'static, exit: Exit) -> Self {
        self.steps.push_back(Step::Exit {
            guest: Box::new(guest),
            exit,
        });
        self
    }

    /// Appends a step in which the guest never stops by itself. Every `run` from here on blocks
    /// until the vCPU is interrupted and then returns [`Exit::Interrupted`]; steps after a spin
    /// are never reached.
    #[must_use]
    pub fn then_spin(mut self) -> Self {
        self.steps.push_back(Step::Spin);
        self
    }

    /// How many steps have not been played yet (a spin step is never used up).
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.steps.len()
    }
}

impl Vcpu for ScriptedVcpu {
    type Interrupter = ScriptedInterrupter;

    fn run(&mut self) -> Result<Exit, Error> {
        {
            // An interrupt raised while the vCPU wasn't running stops the next run before any
            // guest code, as the trait requires.
            let mut pending = self.signal.lock();
            if *pending {
                *pending = false;
                return Ok(Exit::Interrupted);
            }
        }
        match self.steps.pop_front() {
            None => Err(Error::Host("the scripted vCPU has no more exits".into())),
            Some(Step::Exit { guest, exit }) => {
                guest(&mut self.regs);
                Ok(exit)
            }
            Some(Step::Spin) => {
                self.steps.push_front(Step::Spin);
                let mut pending = self.signal.lock();
                while !*pending {
                    pending = self
                        .signal
                        .raised
                        .wait(pending)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                *pending = false;
                Ok(Exit::Interrupted)
            }
        }
    }

    fn interrupter(&self) -> ScriptedInterrupter {
        ScriptedInterrupter {
            signal: Arc::clone(&self.signal),
        }
    }

    fn regs(&self) -> Result<GuestRegs, Error> {
        Ok(self.regs.clone())
    }

    fn set_regs(&mut self, regs: &GuestRegs) -> Result<(), Error> {
        self.regs = regs.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_play_in_order_and_see_what_the_host_wrote() {
        let mut vcpu = ScriptedVcpu::new()
            .then(|regs| regs.x[0] = 1, Exit::Svc(0x27))
            .then(
                |regs| {
                    // The host's write between the two exits is visible to the guest.
                    assert_eq!(regs.x[0], 0xAA);
                    regs.pc = 0x1000;
                },
                Exit::Undefined { pc: 0x1000 },
            );
        assert_eq!(vcpu.remaining(), 2);
        assert_eq!(vcpu.run(), Ok(Exit::Svc(0x27)));
        let mut regs = vcpu.regs().unwrap();
        assert_eq!(regs.x[0], 1);
        regs.x[0] = 0xAA;
        vcpu.set_regs(&regs).unwrap();
        assert_eq!(vcpu.run(), Ok(Exit::Undefined { pc: 0x1000 }));
        assert_eq!(vcpu.regs().unwrap().pc, 0x1000);
        assert_eq!(vcpu.remaining(), 0);
    }

    #[test]
    fn running_past_the_script_is_an_error() {
        let mut vcpu = ScriptedVcpu::new().then(|_| {}, Exit::Svc(1));
        assert_eq!(vcpu.run(), Ok(Exit::Svc(1)));
        assert_eq!(
            vcpu.run(),
            Err(Error::Host("the scripted vCPU has no more exits".into()))
        );
    }

    #[test]
    fn an_interrupt_before_a_run_stops_it_without_playing_a_step() {
        let mut vcpu = ScriptedVcpu::new().then(|regs| regs.x[1] = 7, Exit::Svc(2));
        vcpu.interrupter().interrupt().unwrap();
        assert_eq!(vcpu.run(), Ok(Exit::Interrupted));
        assert_eq!(vcpu.regs().unwrap().x[1], 0);
        // The interrupt is used up; the step plays next.
        assert_eq!(vcpu.run(), Ok(Exit::Svc(2)));
        assert_eq!(vcpu.regs().unwrap().x[1], 7);
    }

    #[test]
    fn a_spinning_guest_stops_only_when_interrupted_and_keeps_spinning_after() {
        let mut vcpu = ScriptedVcpu::new().then_spin().then(|_| {}, Exit::Svc(3));
        for _ in 0..2 {
            let interrupter = vcpu.interrupter();
            let waker = std::thread::spawn(move || interrupter.interrupt());
            assert_eq!(vcpu.run(), Ok(Exit::Interrupted));
            waker.join().unwrap().unwrap();
        }
        // The step after the spin is never reached.
        assert_eq!(vcpu.remaining(), 2);
    }

    #[test]
    fn the_interrupter_can_cross_threads() {
        fn shared<T: Send + Sync + Clone + 'static>() {}
        shared::<ScriptedInterrupter>();
    }
}
