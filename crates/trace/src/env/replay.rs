//! The replay [`Env`]: hands a recording back and reports the first mismatch.

use std::sync::Mutex;

use crate::env::{Env, EnvCall, EnvError, EnvRecord, HostIoReply, HostIoRequest};
use crate::event::GuestTicks;
use crate::recorder::lock;

struct State {
    records: Vec<EnvRecord>,
    cursor: usize,
    /// The first failure. Once set it is returned by every later call, so a replay reports the
    /// first divergence rather than whatever cascades from it.
    failure: Option<EnvError>,
}

impl State {
    /// Take the next record if `extract` accepts it (that is, it answers a call of kind
    /// `requested`).
    fn take<T>(
        &mut self,
        requested: EnvCall,
        extract: impl FnOnce(&EnvRecord) -> Option<T>,
    ) -> Result<T, EnvError> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone());
        }
        let index = self.cursor;
        let error = match self.records.get(index) {
            None => EnvError::Exhausted { index, requested },
            Some(record) => match extract(record) {
                Some(value) => {
                    self.cursor = index + 1;
                    return Ok(value);
                }
                None => EnvError::Mismatch {
                    index,
                    requested,
                    recorded: record.call(),
                },
            },
        };
        self.failure = Some(error.clone());
        Err(error)
    }
}

/// An [`Env`] that replays a recording made by [`LiveEnv`](crate::LiveEnv).
///
/// It returns the recorded answers in order. A call of the wrong kind, a host I/O request that
/// differs from the recorded one, or a call after the recording ends is an [`EnvError`], never a
/// panic; the first such error is remembered and returned by every later call.
pub struct ReplayEnv {
    state: Mutex<State>,
}

impl ReplayEnv {
    /// A replay of `records`, starting at the first.
    #[must_use]
    pub fn new(records: Vec<EnvRecord>) -> Self {
        Self {
            state: Mutex::new(State {
                records,
                cursor: 0,
                failure: None,
            }),
        }
    }

    /// How many records have been consumed.
    #[must_use]
    pub fn consumed(&self) -> usize {
        lock(&self.state).cursor
    }

    /// The first error any call returned, if one has.
    #[must_use]
    pub fn first_error(&self) -> Option<EnvError> {
        lock(&self.state).failure.clone()
    }

    /// Check that the replay used the whole recording without error: returns the first error if
    /// there was one, and [`EnvError::Unconsumed`] if records remain.
    pub fn finish(&self) -> Result<(), EnvError> {
        let state = lock(&self.state);
        if let Some(failure) = &state.failure {
            return Err(failure.clone());
        }
        let remaining = state.records.len().saturating_sub(state.cursor);
        if remaining > 0 {
            return Err(EnvError::Unconsumed {
                index: state.cursor,
                remaining,
            });
        }
        Ok(())
    }
}

impl Env for ReplayEnv {
    fn now(&self) -> Result<GuestTicks, EnvError> {
        lock(&self.state).take(EnvCall::Clock, |record| match record {
            EnvRecord::Clock(ticks) => Some(*ticks),
            _ => None,
        })
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        lock(&self.state).take(EnvCall::Random, |record| match record {
            EnvRecord::Random(value) => Some(*value),
            _ => None,
        })
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        let mut state = lock(&self.state);
        let (recorded, reply) = state.take(EnvCall::HostIo, |record| match record {
            EnvRecord::HostIo { request, reply } => Some((request.clone(), reply.clone())),
            _ => None,
        })?;
        if recorded == *request {
            return Ok(reply);
        }
        let error = EnvError::HostIoRequestMismatch {
            index: state.cursor.saturating_sub(1),
        };
        state.failure = Some(error.clone());
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{FixedClock, LiveEnv};

    fn io(channel: u32, payload: &[u8]) -> HostIoRequest {
        HostIoRequest {
            channel,
            payload: payload.to_vec(),
        }
    }

    /// A workload that mixes every kind of call, so replay is checked against order as well as
    /// values.
    fn workload(env: &dyn Env) -> Result<Vec<String>, EnvError> {
        let mut out = Vec::new();
        out.push(format!("{:?}", env.now()?));
        out.push(format!("{:?}", env.random_u64()?));
        out.push(format!("{:?}", env.host_io(&io(2, b"abc"))?));
        out.push(format!("{:?}", env.random_u64()?));
        out.push(format!("{:?}", env.now()?));
        Ok(out)
    }

    #[test]
    fn replay_returns_the_recorded_values_in_order() {
        let live = LiveEnv::new(FixedClock::new(500, 25), 99);
        let live_out = workload(&live).unwrap();
        let replay = ReplayEnv::new(live.recording());
        assert_eq!(workload(&replay).unwrap(), live_out);
        assert_eq!(replay.consumed(), live.recording().len());
        assert_eq!(replay.finish(), Ok(()));
        assert_eq!(replay.first_error(), None);
    }

    #[test]
    fn a_call_of_the_wrong_kind_is_a_mismatch_and_does_not_advance() {
        let replay = ReplayEnv::new(vec![EnvRecord::Random(5), EnvRecord::Clock(GuestTicks(1))]);
        let expected = EnvError::Mismatch {
            index: 0,
            requested: EnvCall::Clock,
            recorded: EnvCall::Random,
        };
        assert_eq!(replay.now(), Err(expected.clone()));
        assert_eq!(replay.consumed(), 0);
        assert_eq!(replay.first_error(), Some(expected));
    }

    #[test]
    fn the_first_error_sticks_even_if_a_later_call_would_have_matched() {
        let replay = ReplayEnv::new(vec![EnvRecord::Random(5), EnvRecord::Clock(GuestTicks(1))]);
        let first = replay.now().unwrap_err();
        assert_eq!(replay.random_u64(), Err(first.clone()));
        assert_eq!(replay.finish(), Err(first));
    }

    #[test]
    fn running_past_the_end_is_exhausted() {
        let replay = ReplayEnv::new(vec![EnvRecord::Clock(GuestTicks(1))]);
        assert_eq!(replay.now(), Ok(GuestTicks(1)));
        assert_eq!(
            replay.random_u64(),
            Err(EnvError::Exhausted {
                index: 1,
                requested: EnvCall::Random
            })
        );
        let empty = ReplayEnv::new(Vec::new());
        assert_eq!(
            empty.host_io(&io(0, b"")),
            Err(EnvError::Exhausted {
                index: 0,
                requested: EnvCall::HostIo
            })
        );
    }

    #[test]
    fn a_different_host_io_request_is_reported_at_that_record() {
        let live = LiveEnv::new(FixedClock::new(0, 1), 0);
        live.now().unwrap();
        live.host_io(&io(2, b"abc")).unwrap();
        let replay = ReplayEnv::new(live.recording());
        replay.now().unwrap();
        for different in [io(3, b"abc"), io(2, b"abd"), io(2, b"ab")] {
            let replay = ReplayEnv::new(live.recording());
            replay.now().unwrap();
            assert_eq!(
                replay.host_io(&different),
                Err(EnvError::HostIoRequestMismatch { index: 1 })
            );
        }
    }

    #[test]
    fn finish_reports_unconsumed_records() {
        let replay = ReplayEnv::new(vec![
            EnvRecord::Clock(GuestTicks(1)),
            EnvRecord::Random(2),
            EnvRecord::Random(3),
        ]);
        replay.now().unwrap();
        assert_eq!(
            replay.finish(),
            Err(EnvError::Unconsumed {
                index: 1,
                remaining: 2
            })
        );
    }

    #[test]
    fn a_recording_survives_postcard() {
        let live = LiveEnv::new(FixedClock::new(500, 25), 99);
        workload(&live).unwrap();
        let records = live.recording();
        let bytes = postcard::to_allocvec(&records).unwrap();
        let back: Vec<EnvRecord> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, records);
    }
}
