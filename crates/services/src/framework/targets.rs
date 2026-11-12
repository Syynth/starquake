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

use starquake_kernel::slab::{Slab, SlabError, SlabId};

use super::commands::Session;

/// The most objects one domain holds at a time.
///
/// starquake's own bound, not a protocol fact: it stops a guest that never closes its objects
/// from growing the table without limit. A console's server has such a limit too, but no pinned
/// reference gives a number; this one is a guess chosen to be far above what a title keeps open.
pub const MAX_DOMAIN_OBJECTS: usize = 0x1_0000;

/// Names a row of the [`Targets`] table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Default)]
pub(super) struct Domain {
    /// Entry `i` is object id `i + 1`.
    entries: Vec<Option<TargetId>>,
    /// Indices of empty entries, most recently closed last.
    free: Vec<usize>,
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
        let index = self.free.pop().unwrap_or(self.entries.len());
        let object_id = u32::try_from(index.checked_add(1)?).ok()?;
        match self.entries.get_mut(index) {
            Some(entry) => *entry = Some(target),
            None => self.entries.push(Some(target)),
        }
        Some(object_id)
    }

    /// Empties an id's entry and returns what it held.
    pub(super) fn remove(&mut self, object_id: u32) -> Option<TargetId> {
        let index = usize::try_from(object_id.checked_sub(1)?).ok()?;
        let target = self.entries.get_mut(index)?.take()?;
        self.free.push(index);
        Some(target)
    }

    /// The objects still in the domain.
    fn into_targets(self) -> impl Iterator<Item = TargetId> {
        self.entries.into_iter().flatten()
    }
}

/// What a session leads to.
pub(super) enum Target {
    /// One interface. `None` while a request is being dispatched to it.
    Interface(Option<Box<dyn Session>>),
    /// A domain of interfaces.
    Domain(Domain),
}

struct Row {
    /// Sessions and domain entries that refer to this row. Never zero.
    refs: u32,
    target: Target,
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
        let id = self.insert(Target::Interface(Some(session)))?;
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

    fn slot(&mut self, id: TargetId) -> Result<&mut Option<Box<dyn Session>>, TargetError> {
        match &mut self.rows.get_mut(id)?.target {
            Target::Interface(slot) => Ok(slot),
            Target::Domain(_) => Err(TargetError::WrongKind),
        }
    }

    /// Takes an interface out of its row for the length of a dispatch.
    pub(super) fn check_out(&mut self, id: TargetId) -> Result<Box<dyn Session>, TargetError> {
        self.slot(id)?.take().ok_or(TargetError::WrongKind)
    }

    /// Puts an interface back after a dispatch.
    pub(super) fn check_in(
        &mut self,
        id: TargetId,
        session: Box<dyn Session>,
    ) -> Result<(), TargetError> {
        *self.slot(id)? = Some(session);
        Ok(())
    }

    /// The interface in a row, unless it is a domain or checked out.
    pub(super) fn interface_mut(&mut self, id: TargetId) -> Option<&mut dyn Session> {
        self.slot(id).ok()?.as_deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands;

    /// An interface with no commands, to put in the table.
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
