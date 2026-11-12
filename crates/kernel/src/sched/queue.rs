//! The scheduler's priority queues: for each core and each of the 64 priorities, a list of the
//! threads *scheduled* on that core (their active core) and a list of the threads *suggested*
//! to it (allowed there, but active elsewhere or nowhere).
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_priority_queue.h` @ cf654052
//! (GPL-3.0-or-later), operation for operation. Eden links threads intrusively: each `KThread`
//! has one `QueueEntry` (previous and next pointers) per core, and each list's root is an entry
//! holding its head and tail. Here the entries are [`QueueLink`]s in the thread's row, holding
//! [`ThreadId`]s instead of pointers, and the roots live in this struct. Every operation takes
//! the thread table as a separate `&mut` borrow, so the queue and the rows can be edited in one
//! call without a borrow of the whole scheduler.
//!
//! **Editing several rows at once.** Linking or unlinking touches up to three rows (the member
//! and its two neighbours) plus a root. No operation needs two rows borrowed at the same time:
//! each one copies the neighbour ids out of the member's link (they are `Copy`), then writes
//! the rows one after another. That is the pattern the rest of the kernel can follow; the slab
//! did not need a "borrow two rows mutably" method.
//!
//! **Walking by id.** A walk reads the next id out of a row and looks it up again with
//! [`Slab::stored`], the lookup for ids the kernel stored itself. Each lookup is fallible (a
//! stale id is a [`SchedError::Corrupt`], through `?`), and loops in the scheduler bound
//! themselves with a [`Walk`] by the number of live threads, the most entries any list can hold.
//!
//! Differences from Eden, none observable:
//! - Eden's queue takes priorities `0..=64` and ignores 64 (the idle and dummy thread
//!   priorities, `k_thread.h`). [`Priority`] cannot hold 64, and there are no idle or dummy
//!   threads, so those branches and the `IsDummyThread` checks are gone.
//! - Removing a thread clears its link; Eden leaves the stale pointers in place.

use super::SchedError;
use super::cores::{AffinityMask, CoreId};
use super::thread::{PRIORITY_COUNT, Priority, QueueLink, Thread};
use crate::id::ThreadId;
use crate::slab::Slab;

/// The thread table, as the queue sees it.
pub(super) type Threads = Slab<ThreadId, Thread>;

/// A bound on a walk over a queue: at most one step per live thread. A longer walk means the
/// links form a cycle, which only damaged bookkeeping could cause.
pub(super) struct Walk {
    left: usize,
}

impl Walk {
    /// A walk over the threads in `threads`.
    pub(super) fn new(threads: &Threads) -> Self {
        Self {
            left: threads.len(),
        }
    }

    /// Takes one step, or fails if the walk is longer than the thread table.
    pub(super) fn step(&mut self) -> Result<(), SchedError> {
        self.left = self.left.checked_sub(1).ok_or(SchedError::Corrupt)?;
        Ok(())
    }
}

fn link(threads: &Threads, id: ThreadId, core: CoreId) -> Result<QueueLink, SchedError> {
    threads
        .stored(id)?
        .links
        .get(usize::from(core.index()))
        .copied()
        .ok_or(SchedError::Corrupt)
}

fn link_mut(
    threads: &mut Threads,
    id: ThreadId,
    core: CoreId,
) -> Result<&mut QueueLink, SchedError> {
    threads
        .stored_mut(id)?
        .links
        .get_mut(usize::from(core.index()))
        .ok_or(SchedError::Corrupt)
}

/// One list's root: Eden's root `QueueEntry`, whose next is the head and previous the tail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Root {
    head: Option<ThreadId>,
    tail: Option<ThreadId>,
}

/// One set of lists over every priority and core, with a bitmap per core of the non-empty
/// priorities: Eden's `KPriorityQueueImpl` (its `KPerCoreQueue` array and
/// `m_available_priorities`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Levels {
    /// Indexed by `level * core_count + core`.
    roots: Vec<Root>,
    /// Per core: bit `level` set when that level's list is non-empty.
    available: Vec<u64>,
    core_count: usize,
}

impl Levels {
    fn new(core_count: u8) -> Self {
        let core_count = usize::from(core_count);
        Self {
            roots: vec![Root::default(); PRIORITY_COUNT * core_count],
            available: vec![0; core_count],
            core_count,
        }
    }

    fn slot(&self, level: usize, core: CoreId) -> Result<usize, SchedError> {
        let core = usize::from(core.index());
        if core >= self.core_count {
            return Err(SchedError::Corrupt);
        }
        level
            .checked_mul(self.core_count)
            .and_then(|base| base.checked_add(core))
            .ok_or(SchedError::Corrupt)
    }

    fn root(&self, level: usize, core: CoreId) -> Result<Root, SchedError> {
        let slot = self.slot(level, core)?;
        self.roots.get(slot).copied().ok_or(SchedError::Corrupt)
    }

    fn root_mut(&mut self, level: usize, core: CoreId) -> Result<&mut Root, SchedError> {
        let slot = self.slot(level, core)?;
        self.roots.get_mut(slot).ok_or(SchedError::Corrupt)
    }

    fn available_mut(&mut self, core: CoreId) -> Result<&mut u64, SchedError> {
        self.available
            .get_mut(usize::from(core.index()))
            .ok_or(SchedError::Corrupt)
    }

    fn set_available(
        &mut self,
        level: usize,
        core: CoreId,
        non_empty: bool,
    ) -> Result<(), SchedError> {
        let bit = u32::try_from(level)
            .ok()
            .and_then(|level| 1_u64.checked_shl(level))
            .ok_or(SchedError::Corrupt)?;
        let available = self.available_mut(core)?;
        if non_empty {
            *available |= bit;
        } else {
            *available &= !bit;
        }
        Ok(())
    }

    /// `KPerCoreQueue::PushBack` plus the bitmap update in `KPriorityQueueImpl::PushBack`.
    fn push_back(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        core: CoreId,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let level = priority.level();
        let tail = self.root(level, core)?.tail;
        *link_mut(threads, member, core)? = QueueLink {
            prev: tail,
            next: None,
        };
        match tail {
            Some(tail) => link_mut(threads, tail, core)?.next = Some(member),
            None => self.root_mut(level, core)?.head = Some(member),
        }
        self.root_mut(level, core)?.tail = Some(member);
        if tail.is_none() {
            self.set_available(level, core, true)?;
        }
        Ok(())
    }

    /// `KPerCoreQueue::PushFront` plus the bitmap update in `KPriorityQueueImpl::PushFront`.
    fn push_front(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        core: CoreId,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let level = priority.level();
        let head = self.root(level, core)?.head;
        *link_mut(threads, member, core)? = QueueLink {
            prev: None,
            next: head,
        };
        match head {
            Some(head) => link_mut(threads, head, core)?.prev = Some(member),
            None => self.root_mut(level, core)?.tail = Some(member),
        }
        self.root_mut(level, core)?.head = Some(member);
        if head.is_none() {
            self.set_available(level, core, true)?;
        }
        Ok(())
    }

    /// `KPerCoreQueue::Remove` plus the bitmap update in `KPriorityQueueImpl::Remove`.
    fn remove(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        core: CoreId,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let level = priority.level();
        let QueueLink { prev, next } = link(threads, member, core)?;
        match prev {
            Some(prev) => link_mut(threads, prev, core)?.next = next,
            None => self.root_mut(level, core)?.head = next,
        }
        match next {
            Some(next) => link_mut(threads, next, core)?.prev = prev,
            None => self.root_mut(level, core)?.tail = prev,
        }
        *link_mut(threads, member, core)? = QueueLink::default();
        if self.root(level, core)?.head.is_none() {
            self.set_available(level, core, false)?;
        }
        Ok(())
    }

    /// `KPriorityQueueImpl::GetFront(core)`: the head of the most urgent non-empty level.
    fn front(&self, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        let available = *self
            .available
            .get(usize::from(core.index()))
            .ok_or(SchedError::Corrupt)?;
        if available == 0 {
            return Ok(None);
        }
        let level = usize::try_from(available.trailing_zeros()).map_err(|_| SchedError::Corrupt)?;
        Ok(self.root(level, core)?.head)
    }

    /// `KPriorityQueueImpl::GetFront(priority, core)`.
    fn front_at(&self, priority: Priority, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        Ok(self.root(priority.level(), core)?.head)
    }

    /// `KPriorityQueueImpl::GetNext`: the next thread at the same priority, or else the head of
    /// the next non-empty level below the member's priority.
    fn next(
        &self,
        threads: &Threads,
        core: CoreId,
        member: ThreadId,
    ) -> Result<Option<ThreadId>, SchedError> {
        if let Some(next) = link(threads, member, core)?.next {
            return Ok(Some(next));
        }
        let level = threads.stored(member)?.priority.level();
        let available = *self
            .available
            .get(usize::from(core.index()))
            .ok_or(SchedError::Corrupt)?;
        // Levels strictly after `level`; `level` is at most 63, so the shift is in range.
        let after = u32::try_from(level)
            .ok()
            .and_then(|level| level.checked_add(1))
            .and_then(|from| u64::MAX.checked_shl(from))
            .unwrap_or(0);
        let rest = available & after;
        if rest == 0 {
            return Ok(None);
        }
        let next_level = usize::try_from(rest.trailing_zeros()).map_err(|_| SchedError::Corrupt)?;
        Ok(self.root(next_level, core)?.head)
    }

    /// `KPriorityQueueImpl::MoveToBack`: requeue the member at the back of its level and return
    /// that level's new head.
    fn move_to_back(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        core: CoreId,
        member: ThreadId,
    ) -> Result<Option<ThreadId>, SchedError> {
        self.remove(threads, priority, core, member)?;
        self.push_back(threads, priority, core, member)?;
        self.front_at(priority, core)
    }

    /// The members of one list, head first. Test support for checking the links.
    #[cfg(test)]
    fn list(
        &self,
        threads: &Threads,
        priority: Priority,
        core: CoreId,
    ) -> Result<Vec<ThreadId>, SchedError> {
        let mut out = Vec::new();
        let mut walk = Walk::new(threads);
        let mut at = self.front_at(priority, core)?;
        while let Some(id) = at {
            walk.step()?;
            out.push(id);
            at = link(threads, id, core)?.next;
        }
        Ok(out)
    }
}

/// Eden's `KPriorityQueue`: the scheduled and suggested sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PriorityQueue {
    scheduled: Levels,
    suggested: Levels,
}

impl PriorityQueue {
    /// Empty queues for `core_count` cores.
    pub(super) fn new(core_count: u8) -> Self {
        Self {
            scheduled: Levels::new(core_count),
            suggested: Levels::new(core_count),
        }
    }

    /// Eden's private `PushBack(priority, member)`: onto the back of its active core's scheduled
    /// list, and the back of the suggested list of every other core in its affinity.
    fn push_back_at(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let (mut affinity, active) = placement(threads, member)?;
        if let Some(core) = active {
            self.scheduled.push_back(threads, priority, core, member)?;
            affinity = affinity.without(core);
        }
        for core in affinity.iter() {
            self.suggested.push_back(threads, priority, core, member)?;
        }
        Ok(())
    }

    /// Eden's private `PushFront(priority, member)`. Only the scheduled list is pushed at the
    /// front; Eden notes that Nintendo pushes suggestions at the back.
    fn push_front_at(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let (mut affinity, active) = placement(threads, member)?;
        if let Some(core) = active {
            self.scheduled.push_front(threads, priority, core, member)?;
            affinity = affinity.without(core);
        }
        for core in affinity.iter() {
            self.suggested.push_back(threads, priority, core, member)?;
        }
        Ok(())
    }

    /// Eden's private `Remove(priority, member)`.
    fn remove_at(
        &mut self,
        threads: &mut Threads,
        priority: Priority,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let (mut affinity, active) = placement(threads, member)?;
        if let Some(core) = active {
            self.scheduled.remove(threads, priority, core, member)?;
            affinity = affinity.without(core);
        }
        for core in affinity.iter() {
            self.suggested.remove(threads, priority, core, member)?;
        }
        Ok(())
    }

    /// `PushBack(member)`, at the member's current priority.
    pub(super) fn push_back(
        &mut self,
        threads: &mut Threads,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let priority = threads.stored(member)?.priority;
        self.push_back_at(threads, priority, member)
    }

    /// `Remove(member)`, at the member's current priority.
    pub(super) fn remove(
        &mut self,
        threads: &mut Threads,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let priority = threads.stored(member)?.priority;
        self.remove_at(threads, priority, member)
    }

    /// `GetScheduledFront(core)`.
    pub(super) fn scheduled_front(&self, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        self.scheduled.front(core)
    }

    /// `GetSuggestedFront(core)`.
    pub(super) fn suggested_front(&self, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        self.suggested.front(core)
    }

    /// `GetScheduledFront(core, priority)`: the head of one level of a core's scheduled lists.
    pub(super) fn scheduled_front_at(
        &self,
        core: CoreId,
        priority: Priority,
    ) -> Result<Option<ThreadId>, SchedError> {
        self.scheduled.front_at(priority, core)
    }

    /// `GetSuggestedFront(core, priority)`: the head of one level of a core's suggested lists.
    pub(super) fn suggested_front_at(
        &self,
        core: CoreId,
        priority: Priority,
    ) -> Result<Option<ThreadId>, SchedError> {
        self.suggested.front_at(priority, core)
    }

    /// `GetSamePriorityNext(core, member)`: the member's successor in the one list it is on for
    /// `core` (scheduled or suggested), without moving to another level.
    pub(super) fn same_priority_next(
        threads: &Threads,
        core: CoreId,
        member: ThreadId,
    ) -> Result<Option<ThreadId>, SchedError> {
        Ok(link(threads, member, core)?.next)
    }

    /// `GetScheduledNext(core, member)`.
    pub(super) fn scheduled_next(
        &self,
        threads: &Threads,
        core: CoreId,
        member: ThreadId,
    ) -> Result<Option<ThreadId>, SchedError> {
        self.scheduled.next(threads, core, member)
    }

    /// `GetSuggestedNext(core, member)`.
    pub(super) fn suggested_next(
        &self,
        threads: &Threads,
        core: CoreId,
        member: ThreadId,
    ) -> Result<Option<ThreadId>, SchedError> {
        self.suggested.next(threads, core, member)
    }

    /// `MoveToScheduledBack(member)`: requeue the member at the back of its level on its active
    /// core, and return that level's new head. The caller checks the member has an active core.
    pub(super) fn move_to_scheduled_back(
        &mut self,
        threads: &mut Threads,
        member: ThreadId,
        core: CoreId,
    ) -> Result<Option<ThreadId>, SchedError> {
        let priority = threads.stored(member)?.priority;
        self.scheduled.move_to_back(threads, priority, core, member)
    }

    /// `ChangePriority(prev_priority, is_running, member)`: requeue at the member's new
    /// priority, at the front of its scheduled list if it is running so it keeps running.
    pub(super) fn change_priority(
        &mut self,
        threads: &mut Threads,
        prev_priority: Priority,
        is_running: bool,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let new_priority = threads.stored(member)?.priority;
        self.remove_at(threads, prev_priority, member)?;
        if is_running {
            self.push_front_at(threads, new_priority, member)
        } else {
            self.push_back_at(threads, new_priority, member)
        }
    }

    /// `ChangeAffinityMask(prev_core, prev_affinity, member)`: leave every list the old
    /// placement put the member in, and join (at the back) every list the new one does.
    pub(super) fn change_affinity_mask(
        &mut self,
        threads: &mut Threads,
        prev_core: Option<CoreId>,
        prev_affinity: AffinityMask,
        member: ThreadId,
    ) -> Result<(), SchedError> {
        let priority = threads.stored(member)?.priority;
        for core in prev_affinity.iter() {
            if Some(core) == prev_core {
                self.scheduled.remove(threads, priority, core, member)?;
            } else {
                self.suggested.remove(threads, priority, core, member)?;
            }
        }
        let (new_affinity, new_core) = placement(threads, member)?;
        for core in new_affinity.iter() {
            if Some(core) == new_core {
                self.scheduled.push_back(threads, priority, core, member)?;
            } else {
                self.suggested.push_back(threads, priority, core, member)?;
            }
        }
        Ok(())
    }

    /// `ChangeCore(prev_core, member, to_front)`: the member's active core changed from
    /// `prev_core`; move it from the old core's scheduled list to its suggested list, and from
    /// the new core's suggested list to its scheduled list.
    pub(super) fn change_core(
        &mut self,
        threads: &mut Threads,
        prev_core: Option<CoreId>,
        member: ThreadId,
        to_front: bool,
    ) -> Result<(), SchedError> {
        let (_, new_core) = placement(threads, member)?;
        let priority = threads.stored(member)?.priority;
        if prev_core == new_core {
            return Ok(());
        }
        if let Some(prev) = prev_core {
            self.scheduled.remove(threads, priority, prev, member)?;
        }
        if let Some(new) = new_core {
            self.suggested.remove(threads, priority, new, member)?;
            if to_front {
                self.scheduled.push_front(threads, priority, new, member)?;
            } else {
                self.scheduled.push_back(threads, priority, new, member)?;
            }
        }
        if let Some(prev) = prev_core {
            self.suggested.push_back(threads, priority, prev, member)?;
        }
        Ok(())
    }

    /// One scheduled list, head first. Test support.
    #[cfg(test)]
    pub(super) fn scheduled_list(
        &self,
        threads: &Threads,
        priority: Priority,
        core: CoreId,
    ) -> Result<Vec<ThreadId>, SchedError> {
        self.scheduled.list(threads, priority, core)
    }

    /// One suggested list, head first. Test support.
    #[cfg(test)]
    pub(super) fn suggested_list(
        &self,
        threads: &Threads,
        priority: Priority,
        core: CoreId,
    ) -> Result<Vec<ThreadId>, SchedError> {
        self.suggested.list(threads, priority, core)
    }

    /// Whether a core's bitmap says the list at `priority` is non-empty. Test support.
    #[cfg(test)]
    pub(super) fn is_available(&self, scheduled: bool, priority: Priority, core: CoreId) -> bool {
        let levels = if scheduled {
            &self.scheduled
        } else {
            &self.suggested
        };
        levels
            .available
            .get(usize::from(core.index()))
            .is_some_and(|bits| bits & (1_u64 << priority.level()) != 0)
    }
}

/// A member's affinity and active core, copied out of its row.
fn placement(
    threads: &Threads,
    member: ThreadId,
) -> Result<(AffinityMask, Option<CoreId>), SchedError> {
    let thread = threads.stored(member)?;
    Ok((thread.affinity, thread.active_core))
}

#[cfg(test)]
mod tests;
