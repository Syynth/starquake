//! What a server's sessions lead to: interfaces and domains, shared by reference count.
//!
//! A session handle leads to one [`Target`]: a single interface, or a [`Domain`] of them. Targets
//! are shared. A cloned session leads to the same target as the original, and an interface can
//! be both an entry of a domain and the target of a session of its own (after "copy from
//! domain"). So targets live in one table and are referred to by [`TargetId`]; each session and
//! each domain entry holds one reference, and a target is dropped when its last reference goes.
//!
//! This is starquake's own structure (Eden shares `std::shared_ptr`s to its handlers,
//! `eden/src/core/hle/service/hle_ipc.h`); nothing here is visible to a guest except domain
//! object ids, see [`Domain`].
//!
//! The table is saved in a server's snapshot as it is, slot for slot, so that the ids handed out
//! after a restore are the ones the saved run would have handed out. An interface is written as
//! its name and its saved state; reading one back gives a frozen row, which [`Targets::thaw`]
//! turns into a live session once it knows the server's interfaces.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use starquake_kernel::slab::{Slab, SlabError, SlabId};

use super::commands::Session;
use super::snapshot::{Interfaces, SnapshotError};

/// The most objects one domain holds at a time.
///
/// starquake's own bound, not a protocol fact: it stops a guest that never closes its objects
/// from growing the table without limit. A console's server has such a limit too, but no pinned
/// reference gives a number; this one is a guess chosen to be far above what a title keeps open.
pub const MAX_DOMAIN_OBJECTS: usize = 0x1_0000;

/// Names a row of the [`Targets`] table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(super) struct TargetId {
    index: u32,
    generation: u32,
}

impl SlabId for TargetId {
    fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    fn index(self) -> u32 {
        self.index
    }

    fn generation(self) -> u32 {
        self.generation
    }
}

/// A domain's table of objects, by the object ids the guest uses.
///
/// Ids start at 1 (`eden/src/core/hle/service/hle_ipc.cpp`, `HandleDomainSyncRequest`:
/// `DomainHandler(object_id - 1)`), and the object a session had when it was converted is id 1
/// (`eden/src/core/hle/service/sm/sm_controller.cpp`, `ConvertCurrentObjectToDomain`).
///
/// Divergence from Eden: Eden appends every new object to the end of a vector and never reuses
/// the id of a closed one (`hle_ipc.h`, `AppendDomainHandler`, `CloseDomainHandler`), so its table
/// only grows. Here the id closed most recently is the next one handed out, and the table holds
/// at most [`MAX_DOMAIN_OBJECTS`]. Guess: a client cannot tell, because it only uses ids it was
/// given; a trace of ids from a console would confirm the order a real server reuses them in.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Domain {
    /// Entry `i` is object id `i + 1`.
    entries: Vec<Option<TargetId>>,
    /// Indices of empty entries, most recently closed last.
    free: Vec<u32>,
}

impl Domain {
    /// The id of the object a session had before it became a domain.
    pub(super) const ROOT_OBJECT_ID: u32 = 1;

    /// A domain holding `root` as object [`Domain::ROOT_OBJECT_ID`].
    pub(super) fn with_root(root: TargetId) -> Self {
        Self {
            entries: vec![Some(root)],
            free: Vec::new(),
        }
    }

    /// The object an id names.
    pub(super) fn get(&self, object_id: u32) -> Option<TargetId> {
        let index = usize::try_from(object_id.checked_sub(1)?).ok()?;
        *self.entries.get(index)?
    }

    /// How many more objects fit.
    pub(super) fn room(&self) -> usize {
        MAX_DOMAIN_OBJECTS.saturating_sub(self.entries.len() - self.free.len())
    }

    /// Adds an object and returns its id, or `None` if the domain is full.
    pub(super) fn add(&mut self, target: TargetId) -> Option<u32> {
        if self.room() == 0 {
            return None;
        }
        let index = match self.free.pop() {
            Some(index) => usize::try_from(index).ok()?,
            None => self.entries.len(),
        };
        let object_id = u32::try_from(index.checked_add(1)?).ok()?;
        match self.entries.get_mut(index) {
            Some(entry) => *entry = Some(target),
            None => self.entries.push(Some(target)),
        }
        Some(object_id)
    }

    /// Empties an id's entry and returns what it held.
    pub(super) fn remove(&mut self, object_id: u32) -> Option<TargetId> {
        let index = object_id.checked_sub(1)?;
        let target = self.entries.get_mut(usize::try_from(index).ok()?)?.take()?;
        self.free.push(index);
        Some(target)
    }

    /// The objects still in the domain.
    fn into_targets(self) -> impl Iterator<Item = TargetId> {
        self.entries.into_iter().flatten()
    }

    /// The objects in the domain.
    fn targets(&self) -> impl Iterator<Item = TargetId> {
        self.entries.iter().copied().flatten()
    }

    /// Whether the table is one [`Domain::add`] and [`Domain::remove`] can have built: no more
    /// entries than a domain holds, and the free list names every empty entry exactly once.
    fn is_well_formed(&self) -> bool {
        let mut empty: Vec<bool> = self.entries.iter().map(Option::is_none).collect();
        let listed_once = self.free.iter().all(|index| {
            let slot = usize::try_from(*index)
                .ok()
                .and_then(|at| empty.get_mut(at));
            // An entry is ticked off when the list names it; a second naming finds it ticked.
            slot.is_some_and(|slot| core::mem::replace(slot, false))
        });
        self.entries.len() <= MAX_DOMAIN_OBJECTS && listed_once && !empty.contains(&true)
    }
}

/// An interface's place in its row.
pub(super) enum Slot {
    /// The interface.
    Live(Box<dyn Session>),
    /// Empty while a request is being dispatched to the interface.
    Out,
    /// As read from a snapshot, until [`Targets::thaw`] makes it live. Never in a table a
    /// server uses.
    Frozen {
        /// [`Session::name`].
        interface: String,
        /// [`Session::save`].
        state: Vec<u8>,
    },
}

/// What a session leads to.
pub(super) enum Target {
    /// One interface.
    Interface(Slot),
    /// A domain of interfaces.
    Domain(Domain),
}

/// One row of the table.
pub(super) struct Row {
    /// Sessions and domain entries that refer to this row. Never zero.
    refs: u32,
    target: Target,
}

/// A [`Row`] as a snapshot holds it. The variant order is the layout.
#[derive(Serialize, Deserialize)]
enum SavedTarget {
    Interface { interface: String, state: Vec<u8> },
    Domain(Domain),
}

/// A [`Row`] as a snapshot holds it. The field order is the layout.
#[derive(Serialize, Deserialize)]
struct SavedRow {
    refs: u32,
    target: SavedTarget,
}

impl Serialize for Row {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let target = match &self.target {
            Target::Interface(Slot::Live(session)) => SavedTarget::Interface {
                interface: session.name().to_owned(),
                state: session.save().map_err(serde::ser::Error::custom)?,
            },
            Target::Interface(Slot::Frozen { interface, state }) => SavedTarget::Interface {
                interface: interface.clone(),
                state: state.clone(),
            },
            Target::Interface(Slot::Out) => {
                let busy = "an interface is handling a request";
                return Err(serde::ser::Error::custom(busy));
            }
            Target::Domain(domain) => SavedTarget::Domain(domain.clone()),
        };
        let refs = self.refs;
        SavedRow { refs, target }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Row {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let SavedRow { refs, target } = SavedRow::deserialize(deserializer)?;
        let target = match target {
            SavedTarget::Interface { interface, state } => {
                Target::Interface(Slot::Frozen { interface, state })
            }
            SavedTarget::Domain(domain) => Target::Domain(domain),
        };
        Ok(Self { refs, target })
    }
}

/// Why an operation on a server's table of interfaces and domains failed. None of these is
/// reachable from a guest request; they mean the server's own bookkeeping is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TargetError {
    /// The table rejected an id or is full.
    #[error("target table: {0}")]
    Table(#[from] SlabError),
    /// A reference count would pass `u32::MAX`.
    #[error("too many references to one target")]
    TooManyReferences,
    /// An interface was expected and a domain found, or the reverse, or the interface is already
    /// handling a request.
    #[error("target is not what the operation needs")]
    WrongKind,
}

/// The table of a server's targets.
#[derive(Default)]
pub(super) struct Targets {
    rows: Slab<TargetId, Row>,
    interfaces: usize,
}

impl Targets {
    /// Number of interface objects alive, in or out of domains.
    pub(super) fn interface_count(&self) -> usize {
        self.interfaces
    }

    fn insert(&mut self, target: Target) -> Result<TargetId, TargetError> {
        Ok(self.rows.insert(Row { refs: 1, target })?)
    }

    /// Stores an interface with one reference, which the caller now holds.
    pub(super) fn insert_interface(
        &mut self,
        session: Box<dyn Session>,
    ) -> Result<TargetId, TargetError> {
        let id = self.insert(Target::Interface(Slot::Live(session)))?;
        self.interfaces = self.interfaces.saturating_add(1);
        Ok(id)
    }

    /// Stores a domain with one reference, which the caller now holds. The domain's entries take
    /// over references the caller already holds.
    pub(super) fn insert_domain(&mut self, domain: Domain) -> Result<TargetId, TargetError> {
        self.insert(Target::Domain(domain))
    }

    /// Adds a reference.
    pub(super) fn retain(&mut self, id: TargetId) -> Result<(), TargetError> {
        let row = self.rows.get_mut(id)?;
        row.refs = row
            .refs
            .checked_add(1)
            .ok_or(TargetError::TooManyReferences)?;
        Ok(())
    }

    /// Drops a reference. The last one drops the target; a domain then drops its reference to
    /// each of its objects.
    pub(super) fn release(&mut self, id: TargetId) -> Result<(), TargetError> {
        let row = self.rows.get_mut(id)?;
        row.refs = row.refs.saturating_sub(1);
        if row.refs > 0 {
            return Ok(());
        }
        match self.rows.remove(id)?.target {
            Target::Interface(_) => self.interfaces = self.interfaces.saturating_sub(1),
            Target::Domain(domain) => {
                // A domain holds interfaces only, so this recursion is one level deep.
                for object in domain.into_targets() {
                    self.release(object)?;
                }
            }
        }
        Ok(())
    }

    /// Whether the target is a domain.
    pub(super) fn is_domain(&self, id: TargetId) -> Result<bool, TargetError> {
        Ok(matches!(self.rows.get(id)?.target, Target::Domain(_)))
    }

    /// The domain a target is.
    pub(super) fn domain(&self, id: TargetId) -> Result<&Domain, TargetError> {
        match &self.rows.get(id)?.target {
            Target::Domain(domain) => Ok(domain),
            Target::Interface(_) => Err(TargetError::WrongKind),
        }
    }

    /// The domain a target is, mutably.
    pub(super) fn domain_mut(&mut self, id: TargetId) -> Result<&mut Domain, TargetError> {
        match &mut self.rows.get_mut(id)?.target {
            Target::Domain(domain) => Ok(domain),
            Target::Interface(_) => Err(TargetError::WrongKind),
        }
    }

    fn slot(&mut self, id: TargetId) -> Result<&mut Slot, TargetError> {
        match &mut self.rows.get_mut(id)?.target {
            Target::Interface(slot) => Ok(slot),
            Target::Domain(_) => Err(TargetError::WrongKind),
        }
    }

    /// Takes an interface out of its row for the length of a dispatch.
    pub(super) fn check_out(&mut self, id: TargetId) -> Result<Box<dyn Session>, TargetError> {
        let slot = self.slot(id)?;
        match core::mem::replace(slot, Slot::Out) {
            Slot::Live(session) => Ok(session),
            other => {
                *slot = other;
                Err(TargetError::WrongKind)
            }
        }
    }

    /// Puts an interface back after a dispatch.
    pub(super) fn check_in(
        &mut self,
        id: TargetId,
        session: Box<dyn Session>,
    ) -> Result<(), TargetError> {
        *self.slot(id)? = Slot::Live(session);
        Ok(())
    }

    /// The interface in a row, unless it is a domain or checked out.
    pub(super) fn interface_mut(&mut self, id: TargetId) -> Option<&mut dyn Session> {
        match self.slot(id).ok()? {
            Slot::Live(session) => Some(&mut **session),
            Slot::Out | Slot::Frozen { .. } => None,
        }
    }

    /// Overwrites a row's reference count, to build a damaged table.
    #[cfg(test)]
    pub(super) fn set_refs(&mut self, id: TargetId, refs: u32) {
        self.rows.get_mut(id).unwrap().refs = refs;
    }

    /// The rows, for a snapshot.
    pub(super) fn rows(&self) -> &Slab<TargetId, Row> {
        &self.rows
    }

    /// Checks that a snapshot of the table could be restored with `interfaces`: every interface
    /// is in its row and is of a listed type.
    pub(super) fn check_restorable(&self, interfaces: &Interfaces) -> Result<(), SnapshotError> {
        for (_, row) in self.rows.iter() {
            match &row.target {
                Target::Interface(Slot::Live(session)) => interfaces.check(session.name())?,
                Target::Interface(Slot::Out | Slot::Frozen { .. }) => {
                    let busy = "an interface is not in its row";
                    return Err(SnapshotError::Inconsistent(busy));
                }
                Target::Domain(_) => {}
            }
        }
        Ok(())
    }

    /// Makes a table read from a snapshot usable: turns every frozen interface into a session,
    /// and checks every id and count in it against what the table really holds. `sessions` are
    /// the targets the snapshot's sessions lead to, one per session.
    ///
    /// The rows come from a file, so nothing in them is believed: a target a session or a
    /// domain names has to exist, a domain's objects have to be interfaces, and every row's
    /// reference count has to be at least one and equal the number of sessions and domain
    /// entries that name it.
    pub(super) fn thaw<'a>(
        mut rows: Slab<TargetId, Row>,
        interfaces: &Interfaces,
        sessions: impl Iterator<Item = &'a TargetId>,
    ) -> Result<Self, SnapshotError> {
        let ids: Vec<TargetId> = rows.iter().map(|(id, _)| id).collect();
        let mut references: BTreeMap<TargetId, u32> = BTreeMap::new();
        let mut count = |target: TargetId| {
            let references = references.entry(target).or_default();
            *references = references.saturating_add(1);
        };
        for target in sessions {
            if !rows.contains(*target) {
                let dangling = "a session leads to a target that does not exist";
                return Err(SnapshotError::Inconsistent(dangling));
            }
            count(*target);
        }
        let mut live = 0_usize;
        for id in &ids {
            let row = rows.get_mut(*id).map_err(|_| CORRUPT)?;
            match &mut row.target {
                Target::Interface(slot) => {
                    let Slot::Frozen { interface, state } = &*slot else {
                        return Err(CORRUPT);
                    };
                    *slot = Slot::Live(interfaces.thaw(interface, state)?);
                    live = live.saturating_add(1);
                }
                Target::Domain(domain) if !domain.is_well_formed() => {
                    let damaged = "a domain's table of object ids is damaged";
                    return Err(SnapshotError::Inconsistent(damaged));
                }
                Target::Domain(domain) => domain.targets().for_each(&mut count),
            }
        }
        let is_interface = |object| {
            let row = rows.get(object).ok();
            row.is_some_and(|row| matches!(row.target, Target::Interface(_)))
        };
        for (_, row) in rows.iter() {
            if let Target::Domain(domain) = &row.target
                && !domain.targets().all(is_interface)
            {
                let dangling = "a domain holds an object that is not an interface";
                return Err(SnapshotError::Inconsistent(dangling));
            }
        }
        for (id, row) in rows.iter() {
            // A row goes with its last reference, so a count of zero is no row's, even one
            // that nothing refers to.
            if row.refs == 0 {
                let unreferenced = "a target has a reference count of zero";
                return Err(SnapshotError::Inconsistent(unreferenced));
            }
            if references.get(&id).copied().unwrap_or(0) != row.refs {
                let miscounted = "a reference count is not the number of references";
                return Err(SnapshotError::Inconsistent(miscounted));
            }
        }
        Ok(Self {
            rows,
            interfaces: live,
        })
    }
}

/// The table changed under the walk that checks it. Unreachable: nothing else holds it.
const CORRUPT: SnapshotError = SnapshotError::Inconsistent("the table of targets is damaged");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands;

    /// An interface with no commands, to put in the table.
    #[derive(Serialize, Deserialize)]
    struct Probe;

    commands! {
        impl Probe {}
    }

    fn probe() -> Box<dyn Session> {
        Box::new(Probe)
    }

    fn target(index: u32) -> TargetId {
        TargetId::from_parts(index, 1)
    }

    fn stale(id: TargetId) -> TargetError {
        TargetError::Table(SlabError::Stale {
            index: id.index(),
            generation: id.generation(),
        })
    }

    #[test]
    fn an_interface_lives_until_its_last_reference_is_released() {
        let mut targets = Targets::default();
        let id = targets.insert_interface(probe()).unwrap();
        assert_eq!(targets.interface_count(), 1);
        targets.retain(id).unwrap();
        targets.release(id).unwrap();
        assert_eq!(targets.interface_count(), 1);
        assert_eq!(targets.is_domain(id), Ok(false));
        targets.release(id).unwrap();
        assert_eq!(targets.interface_count(), 0);
        // The id is stale from then on, for every operation.
        assert_eq!(targets.release(id), Err(stale(id)));
        assert_eq!(targets.retain(id), Err(stale(id)));
        assert_eq!(targets.is_domain(id), Err(stale(id)));
        assert!(matches!(targets.check_out(id), Err(e) if e == stale(id)));
        assert!(targets.interface_mut(id).is_none());
    }

    #[test]
    fn a_domain_releases_its_objects_when_it_goes() {
        let mut targets = Targets::default();
        let root = targets.insert_interface(probe()).unwrap();
        let shared = targets.insert_interface(probe()).unwrap();
        let mut domain = Domain::with_root(root);
        assert_eq!(domain.add(shared), Some(Domain::ROOT_OBJECT_ID + 1));
        let domain = targets.insert_domain(domain).unwrap();
        assert_eq!(targets.is_domain(domain), Ok(true));
        assert_eq!(targets.interface_count(), 2);
        // `shared` is also held from outside the domain.
        targets.retain(shared).unwrap();

        targets.retain(domain).unwrap();
        targets.release(domain).unwrap();
        assert_eq!(targets.interface_count(), 2);
        targets.release(domain).unwrap();
        // The root went with the domain; the shared one has its outside reference left.
        assert_eq!(targets.interface_count(), 1);
        assert_eq!(targets.is_domain(root), Err(stale(root)));
        targets.release(shared).unwrap();
        assert_eq!(targets.interface_count(), 0);
    }

    #[test]
    fn interfaces_and_domains_are_not_mistaken_for_each_other() {
        let mut targets = Targets::default();
        let interface = targets.insert_interface(probe()).unwrap();
        let domain = targets.insert_domain(Domain::default()).unwrap();
        assert!(matches!(
            targets.domain(interface),
            Err(TargetError::WrongKind)
        ));
        assert!(matches!(
            targets.domain_mut(interface),
            Err(TargetError::WrongKind)
        ));
        assert!(matches!(
            targets.check_out(domain),
            Err(TargetError::WrongKind)
        ));
        assert_eq!(
            targets.check_in(domain, probe()),
            Err(TargetError::WrongKind)
        );
        assert!(targets.interface_mut(domain).is_none());
        assert!(targets.domain(domain).is_ok());
        assert_eq!(targets.interface_count(), 1);
    }

    #[test]
    fn an_interface_is_checked_out_once_at_a_time() {
        let mut targets = Targets::default();
        let id = targets.insert_interface(probe()).unwrap();
        assert!(targets.interface_mut(id).is_some());
        let session = targets.check_out(id).unwrap();
        // Out: it cannot be taken again or reached as an input object, but it still counts.
        assert!(matches!(targets.check_out(id), Err(TargetError::WrongKind)));
        assert!(targets.interface_mut(id).is_none());
        assert_eq!(targets.interface_count(), 1);
        targets.check_in(id, session).unwrap();
        assert!(targets.interface_mut(id).is_some());
    }

    #[test]
    fn a_reference_count_stops_at_its_limit() {
        let mut targets = Targets::default();
        let id = targets.insert_interface(probe()).unwrap();
        // One below the limit: one more reference fits, the next does not.
        targets.rows.get_mut(id).unwrap().refs = u32::MAX - 1;
        targets.retain(id).unwrap();
        assert_eq!(targets.retain(id), Err(TargetError::TooManyReferences));
        assert_eq!(targets.rows.get(id).unwrap().refs, u32::MAX);
        assert_eq!(
            TargetError::TooManyReferences.to_string(),
            "too many references to one target"
        );
    }

    /// The interfaces the tables of these tests hold.
    fn interfaces() -> Interfaces {
        Interfaces::new().with::<Probe>()
    }

    /// A table as a snapshot holds it: written out and read back, every interface frozen.
    fn frozen(targets: &Targets) -> Slab<TargetId, Row> {
        let bytes = postcard::to_allocvec(targets.rows()).unwrap();
        postcard::from_bytes(&bytes).unwrap()
    }

    /// A table with a session's interface, and a domain of a root and one more object. Returns
    /// the table and the targets the two sessions lead to.
    fn table() -> (Targets, [TargetId; 2]) {
        let mut targets = Targets::default();
        let lone = targets.insert_interface(probe()).unwrap();
        let root = targets.insert_interface(probe()).unwrap();
        let object = targets.insert_interface(probe()).unwrap();
        let mut domain = Domain::with_root(root);
        assert_eq!(domain.add(object), Some(Domain::ROOT_OBJECT_ID + 1));
        let domain = targets.insert_domain(domain).unwrap();
        (targets, [lone, domain])
    }

    /// What thawing the table says after `damage` was done to it.
    fn thawed(
        damage: impl FnOnce(&mut Targets, &mut Vec<TargetId>),
    ) -> Result<Targets, SnapshotError> {
        let (mut targets, sessions) = table();
        let mut sessions = sessions.to_vec();
        damage(&mut targets, &mut sessions);
        Targets::thaw(frozen(&targets), &interfaces(), sessions.iter())
    }

    fn inconsistent(message: &'static str) -> SnapshotError {
        SnapshotError::Inconsistent(message)
    }

    #[test]
    fn a_table_read_from_a_snapshot_is_the_table_that_was_saved() {
        let (mut targets, sessions) = table();
        let mut thawed = Targets::thaw(frozen(&targets), &interfaces(), sessions.iter()).unwrap();
        assert_eq!(thawed.interface_count(), targets.interface_count());
        assert_eq!(
            postcard::to_allocvec(thawed.rows()),
            postcard::to_allocvec(targets.rows())
        );
        // The next row either table hands out is the same one.
        assert_eq!(
            thawed.insert_interface(probe()).ok(),
            targets.insert_interface(probe()).ok()
        );
        // And every interface is live again.
        assert!(thawed.interface_mut(sessions[0]).is_some());
        // An interface that is out handling a request cannot be written.
        let out = targets.check_out(sessions[0]).unwrap();
        assert_eq!(
            postcard::to_allocvec(targets.rows()),
            Err(postcard::Error::SerdeSerCustom)
        );
        assert_eq!(
            targets.check_restorable(&interfaces()),
            Err(SnapshotError::Inconsistent(
                "an interface is not in its row"
            ))
        );
        targets.check_in(sessions[0], out).unwrap();
        assert_eq!(targets.check_restorable(&interfaces()), Ok(()));
        assert_eq!(
            targets.check_restorable(&Interfaces::new()),
            Err(SnapshotError::UnknownInterface("Probe".to_owned()))
        );
    }

    #[test]
    fn a_table_whose_ids_or_counts_disagree_is_refused() {
        let dangling_session = thawed(|_, sessions| sessions.push(target(40)));
        assert_eq!(
            dangling_session.err(),
            Some(inconsistent(
                "a session leads to a target that does not exist"
            ))
        );
        // One reference too many, one too few, and a row nothing refers to.
        let miscounted = inconsistent("a reference count is not the number of references");
        let extra = thawed(|targets, sessions| targets.retain(sessions[0]).unwrap());
        assert_eq!(extra.err(), Some(miscounted.clone()));
        let missing = thawed(|_, sessions| sessions.push(sessions[0]));
        assert_eq!(missing.err(), Some(miscounted.clone()));
        let orphan = thawed(|_, sessions| {
            sessions.remove(0);
        });
        assert_eq!(orphan.err(), Some(miscounted.clone()));
        // The same table with its two sessions and nothing else is consistent.
        assert!(thawed(|_, _| {}).is_ok());
    }

    #[test]
    fn a_row_with_no_references_is_refused() {
        // A row nothing refers to, whose count says so: the count is right, and the row is
        // still one no table holds, since a row goes with its last reference.
        let unreferenced = thawed(|targets, sessions| {
            let lone = sessions.remove(0);
            targets.set_refs(lone, 0);
        });
        assert_eq!(
            unreferenced.err(),
            Some(inconsistent("a target has a reference count of zero"))
        );
        // One reference, the fewest a row has, is accepted when one session leads to it.
        let (targets, sessions) = table();
        assert_eq!(targets.rows.get(sessions[0]).unwrap().refs, 1);
        assert!(thawed(|_, _| {}).is_ok());
    }

    #[test]
    fn a_domain_whose_objects_are_not_interfaces_is_refused() {
        let not_interface = inconsistent("a domain holds an object that is not an interface");
        // An object id that leads to a row that does not exist, and one that leads to a domain.
        let dangling = thawed(|targets, sessions| {
            let entries = &mut targets.domain_mut(sessions[1]).unwrap().entries;
            entries[1] = Some(target(40));
        });
        assert_eq!(dangling.err(), Some(not_interface.clone()));
        let nested = thawed(|targets, sessions| {
            let domain = sessions[1];
            targets.domain_mut(domain).unwrap().entries[1] = Some(domain);
        });
        assert_eq!(nested.err(), Some(not_interface.clone()));
    }

    #[test]
    fn a_domain_whose_free_list_is_damaged_is_refused() {
        let damaged = inconsistent("a domain's table of object ids is damaged");
        let cases: [fn(&mut Domain); 4] = [
            // Names a full entry.
            |domain| domain.free.push(0),
            // Names an entry past the table.
            |domain| domain.free.push(2),
            // An empty entry it does not name.
            |domain| domain.entries.push(None),
            // Names an empty entry twice.
            |domain| {
                domain.entries.push(None);
                domain.free.extend([2, 2]);
            },
        ];
        for damage in cases {
            let refused = thawed(|targets, sessions| {
                damage(targets.domain_mut(sessions[1]).unwrap());
            });
            assert_eq!(refused.err(), Some(damaged.clone()));
        }
        // An empty entry named once is how a closed id looks.
        let closed = thawed(|targets, sessions| {
            let domain = targets.domain_mut(sessions[1]).unwrap();
            domain.entries.push(None);
            domain.free.push(2);
        });
        assert!(closed.is_ok());
    }

    #[test]
    fn a_domain_larger_than_a_domain_can_be_is_refused() {
        let filled = |entries: usize| {
            thawed(|targets, sessions| {
                let domain = targets.domain_mut(sessions[1]).unwrap();
                while domain.entries.len() < entries {
                    let index = u32::try_from(domain.entries.len()).unwrap();
                    domain.entries.push(None);
                    domain.free.push(index);
                }
            })
        };
        assert!(filled(MAX_DOMAIN_OBJECTS).is_ok());
        assert_eq!(
            filled(MAX_DOMAIN_OBJECTS + 1).err(),
            Some(inconsistent("a domain's table of object ids is damaged"))
        );
    }

    #[test]
    fn a_domain_starts_with_its_root_as_object_one() {
        let domain = Domain::with_root(target(7));
        assert_eq!(domain.get(Domain::ROOT_OBJECT_ID), Some(target(7)));
        assert_eq!(domain.get(0), None);
        assert_eq!(domain.get(Domain::ROOT_OBJECT_ID + 1), None);
        assert_eq!(domain.get(u32::MAX), None);
        assert_eq!(domain.room(), MAX_DOMAIN_OBJECTS - 1);
    }

    #[test]
    fn ids_count_up_and_the_last_closed_is_reused_first() {
        let mut domain = Domain::with_root(target(0));
        assert_eq!(domain.add(target(1)), Some(2));
        assert_eq!(domain.add(target(2)), Some(3));
        assert_eq!(domain.add(target(3)), Some(4));
        assert_eq!(domain.remove(2), Some(target(1)));
        assert_eq!(domain.remove(4), Some(target(3)));
        // Closed twice, never opened, and zero are all nothing.
        assert_eq!(domain.remove(4), None);
        assert_eq!(domain.remove(9), None);
        assert_eq!(domain.remove(0), None);
        assert_eq!(domain.get(2), None);
        assert_eq!(domain.add(target(4)), Some(4));
        assert_eq!(domain.add(target(5)), Some(2));
        assert_eq!(domain.add(target(6)), Some(5));
        assert_eq!(domain.get(2), Some(target(5)));
    }

    #[test]
    fn a_domain_is_full_exactly_at_its_limit() {
        let mut domain = Domain::with_root(target(0));
        for _ in 1..MAX_DOMAIN_OBJECTS {
            assert!(domain.add(target(1)).is_some());
        }
        assert_eq!(domain.room(), 0);
        assert_eq!(domain.add(target(1)), None);
        // Closing one makes room for exactly one.
        let last = u32::try_from(MAX_DOMAIN_OBJECTS).unwrap();
        assert_eq!(domain.remove(last), Some(target(1)));
        assert_eq!(domain.room(), 1);
        assert_eq!(domain.add(target(2)), Some(last));
        assert_eq!(domain.add(target(2)), None);
    }
}
