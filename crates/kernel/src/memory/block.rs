//! One memory block: a run of pages that share a state, a permission and attributes, with
//! Horizon's lock and share counts and merge barriers, and what `QueryMemory` reports of it.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_memory_block.h` @ cf654052 (GPL-3.0-or-later):
//! `KMemoryBlock` (`HasProperties`, `HasSameProperties`, `CanMergeWith`, `Add`, `Update`,
//! `UpdateAttribute`, `Split`) and `KMemoryInfo::GetSvcMemoryInfo`. The layout a guest reads is
//! `Svc::MemoryInfo` in `eden/src/core/hle/kernel/svc_types.h` @ cf654052, which matches libnx's
//! `MemoryInfo` (`libnx/nx/include/switch/kernel/svc.h` @ b156b477).
//!
//! A block carries every count Horizon's does (kernel spec, rule 4), though this slice has no
//! operation that changes one: the IPC lock count, the device use count, and the three counts
//! behind the merge barriers. The operations that do (`LockForIpc`, `ShareToDevice` and their
//! inverses) land with IPC buffer mapping and device address spaces.
//!
//! Where Eden asserts, this module returns `None` or an error and changes nothing; the map
//! turns that into an internal failure.

use starquake_memory::PAGE_SIZE;

use super::state::{DisableMerge, MemoryAttribute, MemoryPermission, MemoryState};

/// A run of pages with one state, permission and set of attributes (`KMemoryBlock`). What
/// `QueryInfo` returns is a copy of one (`KMemoryInfo` has the same fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBlock {
    address: u64,
    pages: u64,
    state: MemoryState,
    permission: MemoryPermission,
    original_permission: MemoryPermission,
    attribute: MemoryAttribute,
    ipc_lock_count: u16,
    device_use_count: u16,
    ipc_disable_merge_count: u16,
    device_disable_merge_left_count: u16,
    device_disable_merge_right_count: u16,
    disable_merge: DisableMerge,
}

impl MemoryBlock {
    /// A block of `pages` pages at `address` with no locks, shares or merge barriers
    /// (`KMemoryBlock::Initialize`). `None` unless `address` is page aligned, `pages` is not
    /// zero and the block ends at or below 2^64.
    #[must_use]
    pub fn new(
        address: u64,
        pages: u64,
        state: MemoryState,
        permission: MemoryPermission,
        attribute: MemoryAttribute,
    ) -> Option<Self> {
        if !address.is_multiple_of(PAGE_SIZE) || pages == 0 {
            return None;
        }
        // The end may be exactly 2^64 (the block `QueryMemory` reports above the address space
        // ends there), so the last byte is what must be addressable.
        let last = pages.checked_mul(PAGE_SIZE)?.checked_sub(1)?;
        address.checked_add(last)?;
        Some(Self {
            address,
            pages,
            state,
            permission,
            original_permission: MemoryPermission::NONE,
            attribute,
            ipc_lock_count: 0,
            device_use_count: 0,
            ipc_disable_merge_count: 0,
            device_disable_merge_left_count: 0,
            device_disable_merge_right_count: 0,
            disable_merge: DisableMerge::NONE,
        })
    }

    /// The address of the first page.
    #[must_use]
    pub const fn address(&self) -> u64 {
        self.address
    }

    /// How many pages the block has.
    #[must_use]
    pub const fn pages(&self) -> u64 {
        self.pages
    }

    /// The size in bytes. Never overflows: [`Self::new`] and [`Self::split`] keep the last byte
    /// addressable.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.pages.wrapping_mul(PAGE_SIZE)
    }

    /// The address of the last byte.
    #[must_use]
    pub const fn last(&self) -> u64 {
        self.address.wrapping_add(self.size().wrapping_sub(1))
    }

    /// One past the last byte, or `None` for a block that ends at 2^64.
    #[must_use]
    pub const fn end(&self) -> Option<u64> {
        self.address.checked_add(self.size())
    }

    /// The state.
    #[must_use]
    pub const fn state(&self) -> MemoryState {
        self.state
    }

    /// The permission.
    #[must_use]
    pub const fn permission(&self) -> MemoryPermission {
        self.permission
    }

    /// The permission the block had before it was locked for IPC, or none.
    #[must_use]
    pub const fn original_permission(&self) -> MemoryPermission {
        self.original_permission
    }

    /// The attributes.
    #[must_use]
    pub const fn attribute(&self) -> MemoryAttribute {
        self.attribute
    }

    /// How many IPC locks hold the block.
    #[must_use]
    pub const fn ipc_lock_count(&self) -> u16 {
        self.ipc_lock_count
    }

    /// How many device mappings use the block.
    #[must_use]
    pub const fn device_use_count(&self) -> u16 {
        self.device_use_count
    }

    /// Why the block must not be merged with a neighbour.
    #[must_use]
    pub const fn disable_merge(&self) -> DisableMerge {
        self.disable_merge
    }

    /// Whether `address` is inside the block.
    #[must_use]
    pub const fn contains(&self, address: u64) -> bool {
        self.address <= address && address <= self.last()
    }

    /// Whether the block already has this state, permission and attributes, the lock and share
    /// attributes aside (`HasProperties`).
    #[must_use]
    pub fn has_properties(
        &self,
        state: MemoryState,
        permission: MemoryPermission,
        attribute: MemoryAttribute,
    ) -> bool {
        self.state == state
            && self.permission == permission
            && (self.attribute | MemoryAttribute::COUNTED) == (attribute | MemoryAttribute::COUNTED)
    }

    /// Whether the two blocks agree in everything a merge would have to keep
    /// (`HasSameProperties`).
    #[must_use]
    pub fn has_same_properties(&self, other: &Self) -> bool {
        self.state == other.state
            && self.permission == other.permission
            && self.original_permission == other.original_permission
            && self.attribute == other.attribute
            && self.ipc_lock_count == other.ipc_lock_count
            && self.device_use_count == other.device_use_count
    }

    /// Whether `above`, the block right after this one, can be merged into it (`CanMergeWith`):
    /// the same properties, no barrier on this block's upper side and none on `above`'s lower
    /// side.
    #[must_use]
    pub fn can_merge_with(&self, above: &Self) -> bool {
        self.has_same_properties(above)
            && self.disable_merge.excludes(DisableMerge::ALL_RIGHT)
            && above.disable_merge.excludes(DisableMerge::ALL_LEFT)
    }

    /// Whether [`Self::update`] may be applied: the block is not locked for IPC (Eden asserts
    /// `m_original_permission == None` and that `IpcLocked` is clear).
    #[must_use]
    pub fn can_update(&self) -> bool {
        self.original_permission == MemoryPermission::NONE
            && self.attribute.excludes(MemoryAttribute::IPC_LOCKED)
    }

    /// Appends `above`, the block right after this one, to this block (`Add`): the pages, its
    /// barriers, and its count of device mappings that end there.
    pub(super) fn add(&mut self, above: &Self) {
        self.pages = self.pages.saturating_add(above.pages);
        self.disable_merge = self.disable_merge | above.disable_merge;
        self.device_disable_merge_right_count = above.device_disable_merge_right_count;
    }

    /// Gives the block a new state, permission and attributes (`Update`). The lock and share
    /// attributes stay. `set` is added to the barriers only when `head` is set (the block is
    /// the first of the range being updated), and `clear` is then removed from them, so a bit
    /// in both ends up clear.
    pub(super) fn update(
        &mut self,
        state: MemoryState,
        permission: MemoryPermission,
        attribute: MemoryAttribute,
        head: bool,
        set: DisableMerge,
        clear: DisableMerge,
    ) {
        self.state = state;
        self.permission = permission;
        self.attribute = attribute | (self.attribute & MemoryAttribute::COUNTED);
        if head {
            self.disable_merge = self.disable_merge | set;
        }
        self.disable_merge = self.disable_merge & !clear;
    }

    /// Replaces the attributes in `mask` with `attribute` (`UpdateAttribute`).
    pub(super) fn update_attribute(&mut self, mask: MemoryAttribute, attribute: MemoryAttribute) {
        self.attribute = (self.attribute & !mask) | attribute;
    }

    /// Splits the block at `address` (`Split`): this block becomes the part from `address` up,
    /// and the part below is returned. The lower part keeps the barriers that face down and the
    /// counts behind them; the upper part keeps the ones that face up.
    ///
    /// `None`, with nothing changed, unless `address` is page aligned and strictly inside the
    /// block.
    pub(super) fn split(&mut self, address: u64) -> Option<Self> {
        if !address.is_multiple_of(PAGE_SIZE) || address <= self.address || !self.contains(address)
        {
            return None;
        }
        let lower_pages = (address - self.address) / PAGE_SIZE;
        let lower = Self {
            pages: lower_pages,
            disable_merge: self.disable_merge & DisableMerge::ALL_LEFT,
            device_disable_merge_right_count: 0,
            ..*self
        };
        self.address = address;
        self.pages -= lower_pages;
        self.ipc_disable_merge_count = 0;
        self.device_disable_merge_left_count = 0;
        self.disable_merge = self.disable_merge & DisableMerge::ALL_RIGHT;
        Some(lower)
    }

    /// What `QueryMemory` writes for this block (`KMemoryInfo::GetSvcMemoryInfo`).
    #[must_use]
    pub fn svc_info(&self) -> SvcMemoryInfo {
        SvcMemoryInfo {
            base_address: self.address,
            size: self.size(),
            state: self.state.svc(),
            attribute: self.attribute.svc(),
            permission: self.permission.svc(),
            ipc_count: u32::from(self.ipc_lock_count),
            device_count: u32::from(self.device_use_count),
        }
    }
}

/// What `QueryMemory` reports about a block (`Svc::MemoryInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvcMemoryInfo {
    /// The address of the block's first page.
    pub base_address: u64,
    /// The block's size in bytes.
    pub size: u64,
    /// The state (`Svc::MemoryState`).
    pub state: u32,
    /// The attributes (`Svc::MemoryAttribute`).
    pub attribute: u32,
    /// The user's permission (`Svc::MemoryPermission`).
    pub permission: u32,
    /// The IPC lock count.
    pub ipc_count: u32,
    /// The device use count.
    pub device_count: u32,
}

impl SvcMemoryInfo {
    /// The size of the structure in guest memory: two 64-bit and six 32-bit fields, the last of
    /// them padding.
    pub const SIZE: usize = 0x28;

    /// The structure as the guest reads it: the fields in declaration order, little-endian,
    /// then four zero bytes of padding.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0; Self::SIZE];
        let (addresses, words) = bytes.split_at_mut(16);
        let (addresses, _) = addresses.as_chunks_mut::<8>();
        for (slot, value) in addresses.iter_mut().zip([self.base_address, self.size]) {
            *slot = value.to_le_bytes();
        }
        let fields = [
            self.state,
            self.attribute,
            self.permission,
            self.ipc_count,
            self.device_count,
        ];
        let (words, _) = words.as_chunks_mut::<4>();
        for (slot, value) in words.iter_mut().zip(fields) {
            *slot = value.to_le_bytes();
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x1_0000;

    fn block(pages: u64) -> MemoryBlock {
        MemoryBlock::new(
            BASE,
            pages,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE,
        )
        .unwrap()
    }

    #[test]
    fn a_block_needs_an_aligned_address_pages_and_an_addressable_last_byte() {
        let new = |address, pages| {
            MemoryBlock::new(
                address,
                pages,
                MemoryState::Free,
                MemoryPermission::NONE,
                MemoryAttribute::NONE,
            )
        };
        assert!(new(BASE + 1, 1).is_none());
        assert!(new(BASE, 0).is_none());
        // The last page of the 64-bit space ends exactly at 2^64: allowed, with no `end`.
        let top = u64::MAX - (PAGE_SIZE - 1);
        let last = new(top, 1).unwrap();
        assert_eq!(
            (last.last(), last.end(), last.size()),
            (u64::MAX, None, PAGE_SIZE)
        );
        // One page more wraps.
        assert!(new(top, 2).is_none());
        // A page count whose byte size overflows.
        assert!(new(0, u64::MAX / PAGE_SIZE + 1).is_none());
        // The most pages a block at zero can have: one short of the whole 64-bit space.
        assert!(new(0, u64::MAX / PAGE_SIZE).is_some());
        let block = block(3);
        assert_eq!(
            (block.address(), block.pages(), block.size(), block.end()),
            (BASE, 3, 3 * PAGE_SIZE, Some(BASE + 3 * PAGE_SIZE))
        );
        assert!(block.contains(BASE) && block.contains(block.last()));
        assert!(!block.contains(BASE - 1) && !block.contains(block.last() + 1));
    }

    #[test]
    fn a_split_leaves_the_upper_part_in_place_and_returns_the_lower() {
        let mut upper = block(5);
        upper.disable_merge = DisableMerge::NORMAL | DisableMerge::DEVICE_RIGHT;
        upper.ipc_disable_merge_count = 3;
        upper.device_disable_merge_left_count = 4;
        upper.device_disable_merge_right_count = 5;
        upper.ipc_lock_count = 6;
        upper.device_use_count = 7;
        let whole = upper;
        let at = BASE + 2 * PAGE_SIZE;
        let lower = upper.split(at).unwrap();

        assert_eq!((lower.address(), lower.pages()), (BASE, 2));
        assert_eq!((upper.address(), upper.pages()), (at, 3));
        // Barriers and their counts go with the side they face.
        assert_eq!(lower.disable_merge(), DisableMerge::NORMAL);
        assert_eq!(upper.disable_merge(), DisableMerge::DEVICE_RIGHT);
        assert_eq!(
            (
                lower.ipc_disable_merge_count,
                lower.device_disable_merge_left_count,
                lower.device_disable_merge_right_count
            ),
            (3, 4, 0)
        );
        assert_eq!(
            (
                upper.ipc_disable_merge_count,
                upper.device_disable_merge_left_count,
                upper.device_disable_merge_right_count
            ),
            (0, 0, 5)
        );
        // Everything else is the same on both sides.
        for part in [lower, upper] {
            assert!(part.has_same_properties(&whole));
            assert_eq!((part.ipc_lock_count(), part.device_use_count()), (6, 7));
        }
    }

    #[test]
    fn a_split_outside_the_block_or_off_a_page_is_refused_and_changes_nothing() {
        let mut subject = block(2);
        let before = subject;
        for at in [
            BASE,
            BASE - PAGE_SIZE,
            BASE + 2 * PAGE_SIZE,
            BASE + PAGE_SIZE + 1,
        ] {
            assert_eq!(subject.split(at), None, "{at:#x}");
            assert_eq!(subject, before);
        }
        // The one address inside a two-page block that is a page boundary.
        assert!(subject.split(BASE + PAGE_SIZE).is_some());
    }

    #[test]
    fn blocks_merge_only_with_the_same_properties_and_no_barrier_between_them() {
        let below = block(1);
        let mut above = block(1);
        above.address = BASE + PAGE_SIZE;
        assert!(below.can_merge_with(&above));

        // Each property on its own stops the merge.
        let differing: [fn(&mut MemoryBlock); 6] = [
            |b| b.state = MemoryState::Stack,
            |b| b.permission = MemoryPermission::USER_READ,
            |b| b.original_permission = MemoryPermission::USER_READ,
            |b| b.attribute = MemoryAttribute::UNCACHED,
            |b| b.ipc_lock_count = 1,
            |b| b.device_use_count = 1,
        ];
        for change in differing {
            let mut changed = above;
            change(&mut changed);
            assert!(!below.can_merge_with(&changed), "{changed:?}");
        }
        // A barrier counts only on the side it faces.
        for barrier in [
            DisableMerge::NORMAL,
            DisableMerge::DEVICE_LEFT,
            DisableMerge::IPC_LEFT,
            DisableMerge::LOCKED,
        ] {
            let mut barred = above;
            barred.disable_merge = barrier;
            assert!(!below.can_merge_with(&barred), "{barrier:?}");
            let mut harmless = below;
            harmless.disable_merge = barrier;
            assert!(harmless.can_merge_with(&above), "{barrier:?}");
        }
        let mut barred = below;
        barred.disable_merge = DisableMerge::DEVICE_RIGHT;
        assert!(!barred.can_merge_with(&above));
        let mut harmless = above;
        harmless.disable_merge = DisableMerge::DEVICE_RIGHT;
        assert!(below.can_merge_with(&harmless));
    }

    #[test]
    fn adding_a_block_takes_its_pages_barriers_and_right_count() {
        let mut below = block(2);
        below.disable_merge = DisableMerge::NORMAL;
        below.device_disable_merge_right_count = 9;
        let mut above = block(3);
        above.address = BASE + 2 * PAGE_SIZE;
        above.disable_merge = DisableMerge::DEVICE_RIGHT;
        above.device_disable_merge_right_count = 2;
        below.add(&above);
        assert_eq!((below.address(), below.pages()), (BASE, 5));
        assert_eq!(
            below.disable_merge(),
            DisableMerge::NORMAL | DisableMerge::DEVICE_RIGHT
        );
        assert_eq!(below.device_disable_merge_right_count, 2);
    }

    #[test]
    fn an_update_keeps_the_counted_attributes_and_sets_barriers_only_at_the_head() {
        let mut subject = block(1);
        subject.attribute = MemoryAttribute::DEVICE_SHARED | MemoryAttribute::UNCACHED;
        subject.disable_merge = DisableMerge::LOCKED;
        subject.update(
            MemoryState::Stack,
            MemoryPermission::USER_READ,
            MemoryAttribute::LOCKED,
            false,
            DisableMerge::NORMAL,
            DisableMerge::NONE,
        );
        assert_eq!(subject.state(), MemoryState::Stack);
        assert_eq!(subject.permission(), MemoryPermission::USER_READ);
        // Uncached was replaced; DeviceShared, which counts shares, stayed.
        assert_eq!(
            subject.attribute(),
            MemoryAttribute::LOCKED | MemoryAttribute::DEVICE_SHARED
        );
        // Not the head: nothing was set.
        assert_eq!(subject.disable_merge(), DisableMerge::LOCKED);

        subject.update(
            MemoryState::Stack,
            MemoryPermission::USER_READ,
            MemoryAttribute::LOCKED,
            true,
            DisableMerge::NORMAL | DisableMerge::IPC_LEFT,
            DisableMerge::LOCKED | DisableMerge::IPC_LEFT,
        );
        // The head: Normal was set; Locked was cleared; IpcLeft, in both, ends up clear.
        assert_eq!(subject.disable_merge(), DisableMerge::NORMAL);
    }

    #[test]
    fn a_block_locked_for_ipc_cannot_be_updated() {
        let mut subject = block(1);
        assert!(subject.can_update());
        subject.attribute = MemoryAttribute::IPC_LOCKED;
        assert!(!subject.can_update());
        subject.attribute = MemoryAttribute::DEVICE_SHARED;
        assert!(subject.can_update());
        subject.original_permission = MemoryPermission::USER_READ_WRITE;
        assert!(!subject.can_update());
    }

    #[test]
    fn has_properties_ignores_the_counted_attributes_only() {
        let mut subject = block(1);
        subject.attribute = MemoryAttribute::UNCACHED | MemoryAttribute::IPC_LOCKED;
        let (state, permission) = (subject.state(), subject.permission());
        assert!(subject.has_properties(state, permission, MemoryAttribute::UNCACHED));
        assert!(subject.has_properties(
            state,
            permission,
            MemoryAttribute::UNCACHED | MemoryAttribute::DEVICE_SHARED
        ));
        assert!(!subject.has_properties(state, permission, MemoryAttribute::NONE));
        assert!(!subject.has_properties(MemoryState::Stack, permission, MemoryAttribute::UNCACHED));
        assert!(!subject.has_properties(
            state,
            MemoryPermission::USER_READ,
            MemoryAttribute::UNCACHED
        ));
    }

    #[test]
    fn an_attribute_update_replaces_exactly_the_masked_bits() {
        let mut subject = block(1);
        subject.attribute = MemoryAttribute::UNCACHED | MemoryAttribute::DEVICE_SHARED;
        subject.update_attribute(
            MemoryAttribute::SET_MASK,
            MemoryAttribute::PERMISSION_LOCKED,
        );
        assert_eq!(
            subject.attribute(),
            MemoryAttribute::PERMISSION_LOCKED | MemoryAttribute::DEVICE_SHARED
        );
    }

    /// The layout of `Svc::MemoryInfo`, written out by hand from `svc_types.h`: `base_address`
    /// at 0x00, `size` at 0x08, then `state`, `attribute`, `permission`, `ipc_count`,
    /// `device_count` and `padding`, four bytes each from 0x10.
    #[test]
    fn the_guests_memory_info_has_the_documented_layout() {
        let info = SvcMemoryInfo {
            base_address: 0x0102_0304_0506_0708,
            size: 0x1112_1314_1516_1718,
            state: 0x2122_2324,
            attribute: 0x3132_3334,
            permission: 0x4142_4344,
            ipc_count: 0x5152_5354,
            device_count: 0x6162_6364,
        };
        #[rustfmt::skip]
        let golden: [u8; SvcMemoryInfo::SIZE] = [
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
            0x24, 0x23, 0x22, 0x21,
            0x34, 0x33, 0x32, 0x31,
            0x44, 0x43, 0x42, 0x41,
            0x54, 0x53, 0x52, 0x51,
            0x64, 0x63, 0x62, 0x61,
            0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(info.to_bytes(), golden);
    }

    #[test]
    fn a_blocks_report_is_its_user_visible_part() {
        let mut subject = block(2);
        subject.attribute = MemoryAttribute::PERMISSION_LOCKED | MemoryAttribute::IPC_LOCKED;
        subject.ipc_lock_count = 2;
        subject.device_use_count = 3;
        assert_eq!(
            subject.svc_info(),
            SvcMemoryInfo {
                base_address: BASE,
                size: 2 * PAGE_SIZE,
                state: MemoryState::Normal.svc(),
                attribute: (MemoryAttribute::PERMISSION_LOCKED | MemoryAttribute::IPC_LOCKED).svc(),
                // The kernel's bits are not reported.
                permission: u32::from(SvcPermissionBits::READ_WRITE),
                ipc_count: 2,
                device_count: 3,
            }
        );
    }

    /// `Svc::MemoryPermission::ReadWrite`, by its definition in `svc_types.h`: read (bit 0) and
    /// write (bit 1).
    struct SvcPermissionBits;

    impl SvcPermissionBits {
        const READ_WRITE: u8 = 0b11;
    }
}
