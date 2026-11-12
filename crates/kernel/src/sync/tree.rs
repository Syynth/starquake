//! Threads waiting on a key, by key and then by priority: the trees the address arbiter and the
//! condition variables keep (Eden's `KThread::ConditionVariableThreadTree`).
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_thread.h` @ cf654052 (GPL-3.0-or-later):
//! `ConditionVariableComparator` sorts by key, then by priority, and returns 1 (not 0) for an
//! equal key and priority, so `RB_INSERT` (`eden/src/common/tree.h`) puts a thread after every
//! thread already in the tree with the same key and priority: one key's threads are in priority
//! order, and in the order they were inserted within one priority. `nfind_key({key, -1})` finds
//! the first thread with the key, since every priority is above -1. Each process keeps its own
//! trees (`KProcess::m_address_arbiter` and `m_cond_var`), so a key here is a process and an
//! address ([`ProcessKey`]).
//!
//! The tree is plain data: the priorities that order it are the scheduler's, so the
//! [`Scheduler`](crate::sched::Scheduler) computes where a thread goes
//! ([`WaitTree::insert_at`]) and keeps a thread's place right when its priority changes.

use std::collections::BTreeMap;

use crate::id::{ProcessId, ThreadId};

/// What a tree is keyed by: the process whose tree it is and the address (or condition-variable
/// key) in it. A thread that belongs to no process (the scheduler's own tests make them) keys
/// with `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessKey {
    /// The process, if any.
    pub process: Option<ProcessId>,
    /// The address or condition-variable key.
    pub address: u64,
}

/// Threads waiting on keys: each key's threads in the order a signal takes them (priority, then
/// insertion). A key with no threads has no entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaitTree {
    lists: BTreeMap<ProcessKey, Vec<ThreadId>>,
}

impl WaitTree {
    /// The threads waiting on `key`, in the order a signal takes them.
    #[must_use]
    pub fn waiters(&self, key: ProcessKey) -> &[ThreadId] {
        self.lists.get(&key).map_or(&[], Vec::as_slice)
    }

    /// How many keys have waiters.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.lists.len()
    }

    /// Whether `thread` waits on `key` here.
    #[must_use]
    pub fn contains(&self, key: ProcessKey, thread: ThreadId) -> bool {
        self.waiters(key).contains(&thread)
    }

    /// Puts `thread` at `index` of `key`'s threads (an index past the end appends).
    pub(crate) fn insert_at(&mut self, key: ProcessKey, index: usize, thread: ThreadId) {
        let list = self.lists.entry(key).or_default();
        let index = index.min(list.len());
        list.insert(index, thread);
    }

    /// Takes `thread` off `key`'s threads, and says whether it was there.
    pub(crate) fn remove(&mut self, key: ProcessKey, thread: ThreadId) -> bool {
        let Some(list) = self.lists.get_mut(&key) else {
            return false;
        };
        let before = list.len();
        list.retain(|waiter| *waiter != thread);
        let removed = list.len() != before;
        if list.is_empty() {
            self.lists.remove(&key);
        }
        removed
    }

    /// Takes the first of `key`'s threads, the one a signal ends first.
    pub(crate) fn take_front(&mut self, key: ProcessKey) -> Option<ThreadId> {
        let list = self.lists.get_mut(&key)?;
        let front = (!list.is_empty()).then(|| list.remove(0));
        if list.is_empty() {
            self.lists.remove(&key);
        }
        front
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::SlabId;

    fn key(address: u64) -> ProcessKey {
        ProcessKey {
            process: None,
            address,
        }
    }

    fn thread(index: u32) -> ThreadId {
        SlabId::from_parts(index, 1)
    }

    #[test]
    fn insert_remove_and_take_keep_each_keys_order_and_drop_empty_keys() {
        let mut tree = WaitTree::default();
        tree.insert_at(key(4), 0, thread(1));
        tree.insert_at(key(4), 9, thread(2));
        tree.insert_at(key(4), 0, thread(3));
        tree.insert_at(key(8), 0, thread(4));
        assert_eq!(tree.waiters(key(4)), &[thread(3), thread(1), thread(2)]);
        assert_eq!(tree.key_count(), 2);
        assert!(tree.contains(key(4), thread(1)));
        assert!(!tree.contains(key(8), thread(1)));

        assert!(tree.remove(key(4), thread(1)));
        assert!(!tree.remove(key(4), thread(1)));
        assert!(!tree.remove(key(12), thread(1)));
        assert_eq!(tree.take_front(key(4)), Some(thread(3)));
        assert_eq!(tree.take_front(key(4)), Some(thread(2)));
        assert_eq!(tree.take_front(key(4)), None);
        assert_eq!(tree.key_count(), 1, "an emptied key leaves no entry");
        assert!(tree.remove(key(8), thread(4)));
        assert_eq!(tree.key_count(), 0);
    }

    #[test]
    fn processes_do_not_share_keys() {
        let mut tree = WaitTree::default();
        let other = ProcessKey {
            process: Some(SlabId::from_parts(0, 1)),
            address: 4,
        };
        tree.insert_at(key(4), 0, thread(1));
        tree.insert_at(other, 0, thread(2));
        assert_eq!(tree.waiters(key(4)), &[thread(1)]);
        assert_eq!(tree.waiters(other), &[thread(2)]);
    }
}
