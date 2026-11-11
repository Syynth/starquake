//! Checks that relate the parts of an NPDM to each other: `ACI0` against the `ACID` that bounds
//! it ([`Acid::permits`], [`Npdm::validate_against_acid`]), and the main thread against the
//! capabilities and core count ([`Npdm::check_main_thread`]).
//!
//! # Every rule here is a guess
//!
//! None of the pinned references checks `ACI0` against `ACID` (Eden and Eden-NX2 read both blocks
//! and trust them), and the SwitchBrew NPDM page, which names the fields, does not state the
//! subset rules. The rules below are the least permissive reading of "`ACID` is the signed bound
//! on what `ACI0` asks for": a request must be covered field by field, and where a capability has
//! no ordering (program type, kernel version) it must be equal. Each is pinned by a test and none
//! is checked against a console or a retail file. A title that fails a rule here may still be
//! accepted by the console; use [`Npdm::validate_against_acid`] as a diagnostic or a strict mode,
//! not as the reason a title is refused, until the rules are confirmed.
//!
//! The main-thread rules are the `CreateThread` checks in
//! `eden/src/core/hle/kernel/svc/svc_thread.cpp` @ cf654052 applied to the main thread's core and
//! priority from `META`. Eden does not run them for the main thread, so applying them there is
//! also a guess (by analogy).

use std::ops::RangeInclusive;

use super::capabilities::{
    KernelCapabilities, KernelVersion, LOWEST_THREAD_PRIORITY, RegionType, ThreadInfo,
};
use super::{Aci0, Acid, Npdm};

/// The first rule an `ACI0` breaks against its `ACID`. See the module docs: all rules are guesses.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AcidViolation {
    /// The `ACI0` title id is outside the `ACID` title id range.
    #[error("ACI0 title id {title_id:#x} is outside the ACID range {min:#x}..={max:#x}")]
    TitleIdOutOfRange {
        /// The `ACI0` title id.
        title_id: u64,
        /// `ACID` minimum.
        min: u64,
        /// `ACID` maximum.
        max: u64,
    },
    /// The `ACI0` file access header requests permission bits the `ACID` does not grant.
    #[error("ACI0 requests file permissions {requested:#x}, ACID allows {allowed:#x}")]
    FilePermissionsNotAllowed {
        /// Bits the `ACI0` requests.
        requested: u64,
        /// Bits the `ACID` grants.
        allowed: u64,
    },
    /// A service the `ACI0` lists is not covered by any `ACID` entry.
    #[error(
        "ACI0 service {name:?} (server: {server}, wildcard: {wildcard}) is not allowed by ACID"
    )]
    ServiceNotAllowed {
        /// The name, without any trailing `*`.
        name: String,
        /// Whether the entry is a server entry.
        server: bool,
        /// Whether the entry is a wildcard.
        wildcard: bool,
    },
    /// The `ACI0` core range is not inside the `ACID` core range (or the `ACID` has none).
    #[error("ACI0 cores {requested:?} are not inside the ACID cores {allowed:?}")]
    CoresNotAllowed {
        /// The `ACI0` cores.
        requested: RangeInclusive<u8>,
        /// The `ACID` cores, if it declares thread info.
        allowed: Option<RangeInclusive<u8>>,
    },
    /// The `ACI0` priority range is not inside the `ACID` priority range (or the `ACID` has none).
    #[error("ACI0 priorities {requested:?} are not inside the ACID priorities {allowed:?}")]
    PrioritiesNotAllowed {
        /// The `ACI0` priorities.
        requested: RangeInclusive<u8>,
        /// The `ACID` priorities, if it declares thread info.
        allowed: Option<RangeInclusive<u8>>,
    },
    /// The `ACI0` enables an SVC the `ACID` does not.
    #[error("ACI0 enables SVC {id:#x}, which ACID does not")]
    SvcNotAllowed {
        /// The lowest SVC id that is not allowed.
        id: u32,
    },
    /// An `ACI0` memory map is not inside an `ACID` map of the same kind and no less restricted.
    #[error("ACI0 memory map of {size:#x} bytes at {address:#x} is not covered by ACID")]
    MemoryMapNotAllowed {
        /// Start of the map.
        address: u64,
        /// Size of the map.
        size: u64,
    },
    /// An `ACI0` region map has no matching `ACID` entry.
    #[error("ACI0 region map {region:?} is not covered by ACID")]
    RegionNotAllowed {
        /// The region.
        region: RegionType,
    },
    /// The `ACI0` binds an interrupt the `ACID` does not list.
    #[error("ACI0 interrupt {id:#x} is not listed by ACID")]
    InterruptNotAllowed {
        /// The interrupt id.
        id: u16,
    },
    /// The program type differs from the `ACID`'s.
    #[error("ACI0 program type {requested} differs from ACID's {allowed:?}")]
    ProgramTypeMismatch {
        /// The `ACI0` value.
        requested: u8,
        /// The `ACID` value, if it declares one.
        allowed: Option<u8>,
    },
    /// The kernel version differs from the `ACID`'s.
    #[error("ACI0 kernel version {requested:?} differs from ACID's {allowed:?}")]
    KernelVersionMismatch {
        /// The `ACI0` value.
        requested: KernelVersion,
        /// The `ACID` value, if it declares one.
        allowed: Option<KernelVersion>,
    },
    /// The handle table is larger than the `ACID` allows.
    #[error("ACI0 handle table size {requested:#x} exceeds ACID's {allowed:?}")]
    HandleTableTooLarge {
        /// The `ACI0` value.
        requested: u16,
        /// The `ACID` value, if it declares one.
        allowed: Option<u16>,
    },
    /// The `ACI0` sets a debug flag the `ACID` does not.
    #[error("ACI0 debug flags exceed ACID's")]
    DebugFlagsNotAllowed,
    /// The `ACI0` has a capability of a type this crate does not decode that the `ACID` lacks.
    #[error("ACI0 capability {raw:#010x} of an unknown type is not in ACID")]
    UnknownCapabilityNotAllowed {
        /// The descriptor word.
        raw: u32,
    },
}

/// Why the main thread's core or priority from `META` is unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MainThreadViolation {
    /// The core id is not below the core count (`IsValidVirtualCoreId`).
    #[error("main thread core {core} does not exist with {core_count} cores")]
    CoreOutOfRange {
        /// The core.
        core: u8,
        /// Cores the caller says exist.
        core_count: u32,
    },
    /// The `ACI0` declares no core and priority ranges, so no thread may run
    /// (`m_core_mask` stays 0 in Eden).
    #[error("ACI0 declares no thread info capability")]
    NoThreadInfo,
    /// The core is not in the capability's core range.
    #[error("main thread core {core} is not in the capability's core range")]
    CoreNotAllowed {
        /// The core.
        core: u8,
    },
    /// The priority is above the lowest thread priority.
    #[error("main thread priority {priority} is not a thread priority")]
    PriorityOutOfRange {
        /// The priority.
        priority: u8,
    },
    /// The priority is not in the capability's priority range.
    #[error("main thread priority {priority} is not in the capability's priority range")]
    PriorityNotAllowed {
        /// The priority.
        priority: u8,
    },
}

impl Acid {
    /// Checks that `aci` asks for nothing this `ACID` does not grant, and returns the first rule
    /// it breaks. The rules are guesses; see the module docs.
    pub fn permits(&self, aci: &Aci0) -> Result<(), AcidViolation> {
        if !(self.title_id_min..=self.title_id_max).contains(&aci.title_id) {
            return Err(AcidViolation::TitleIdOutOfRange {
                title_id: aci.title_id,
                min: self.title_id_min,
                max: self.title_id_max,
            });
        }
        let requested = aci.file_access.permissions;
        let allowed = self.file_access.permissions;
        if requested & !allowed != 0 {
            return Err(AcidViolation::FilePermissionsNotAllowed { requested, allowed });
        }
        if let Some(service) = aci
            .services
            .iter()
            .find(|request| !self.services.iter().any(|grant| grant.covers(request)))
        {
            return Err(AcidViolation::ServiceNotAllowed {
                name: service.name.clone(),
                server: service.server,
                wildcard: service.wildcard,
            });
        }
        permits_capabilities(&self.kernel_capabilities, &aci.kernel_capabilities)
    }
}

/// True if `inner` lies within `outer`.
fn range_within(inner: &RangeInclusive<u8>, outer: &RangeInclusive<u8>) -> bool {
    outer.start() <= inner.start() && inner.end() <= outer.end()
}

/// True if a request with `request_read_only` is no less restricted than a grant with
/// `grant_read_only`: a read-only grant cannot back a writable request.
fn permission_fits(grant_read_only: bool, request_read_only: bool) -> bool {
    request_read_only || !grant_read_only
}

fn permits_capabilities(
    bound: &KernelCapabilities,
    request: &KernelCapabilities,
) -> Result<(), AcidViolation> {
    if let Some(wanted) = &request.thread_info {
        let allowed: Option<&ThreadInfo> = bound.thread_info.as_ref();
        if !allowed.is_some_and(|a| range_within(&wanted.cores, &a.cores)) {
            return Err(AcidViolation::CoresNotAllowed {
                requested: wanted.cores.clone(),
                allowed: allowed.map(|a| a.cores.clone()),
            });
        }
        if !allowed.is_some_and(|a| range_within(&wanted.priorities, &a.priorities)) {
            return Err(AcidViolation::PrioritiesNotAllowed {
                requested: wanted.priorities.clone(),
                allowed: allowed.map(|a| a.priorities.clone()),
            });
        }
    }
    if let Some(id) = request
        .svc_mask
        .iter()
        .find(|&id| !bound.svc_mask.contains(id))
    {
        return Err(AcidViolation::SvcNotAllowed { id });
    }
    if let Some(map) = request.memory_maps.iter().find(|map| {
        !bound.memory_maps.iter().any(|grant| {
            let inside = match (map.end(), grant.end()) {
                (Some(map_end), Some(grant_end)) => {
                    grant.address <= map.address && map_end <= grant_end
                }
                _ => false,
            };
            grant.kind == map.kind && inside && permission_fits(grant.read_only, map.read_only)
        })
    }) {
        return Err(AcidViolation::MemoryMapNotAllowed {
            address: map.address,
            size: map.size,
        });
    }
    if let Some(map) = request.region_maps.iter().find(|map| {
        !bound.region_maps.iter().any(|grant| {
            grant.region == map.region && permission_fits(grant.read_only, map.read_only)
        })
    }) {
        return Err(AcidViolation::RegionNotAllowed { region: map.region });
    }
    if let Some(&id) = request
        .interrupts
        .iter()
        .find(|id| !bound.interrupts.contains(id))
    {
        return Err(AcidViolation::InterruptNotAllowed { id });
    }
    if let Some(requested) = request.program_type
        && bound.program_type != Some(requested)
    {
        return Err(AcidViolation::ProgramTypeMismatch {
            requested,
            allowed: bound.program_type,
        });
    }
    if let Some(requested) = request.kernel_version
        && bound.kernel_version != Some(requested)
    {
        return Err(AcidViolation::KernelVersionMismatch {
            requested,
            allowed: bound.kernel_version,
        });
    }
    if let Some(requested) = request.handle_table_size
        && bound
            .handle_table_size
            .is_none_or(|allowed| requested > allowed)
    {
        return Err(AcidViolation::HandleTableTooLarge {
            requested,
            allowed: bound.handle_table_size,
        });
    }
    if let Some(wanted) = request.debug_flags {
        let allowed = bound.debug_flags;
        let allow = allowed.is_some_and(|a| a.allow_debug);
        let force = allowed.is_some_and(|a| a.force_debug);
        if (wanted.allow_debug && !allow) || (wanted.force_debug && !force) {
            return Err(AcidViolation::DebugFlagsNotAllowed);
        }
    }
    if let Some(&raw) = request
        .unknown
        .iter()
        .find(|raw| !bound.unknown.contains(raw))
    {
        return Err(AcidViolation::UnknownCapabilityNotAllowed { raw });
    }
    Ok(())
}

impl Npdm {
    /// Checks the `ACI0` against the `ACID` (see [`Acid::permits`]). Parsing does not call this:
    /// whether a violation refuses the title is the caller's policy, and the rules are guesses.
    pub fn validate_against_acid(&self) -> Result<(), AcidViolation> {
        self.acid.permits(&self.aci)
    }

    /// Checks the main thread's core and priority from `META` against `core_count` and the
    /// `ACI0` thread info capability, in the order `CreateThread` does (valid core, core in the
    /// mask, valid priority, priority in the mask). `core_count` is the number of cores the guest
    /// sees; see [`KernelCapabilities::check_policy`].
    pub fn check_main_thread(&self, core_count: u32) -> Result<(), MainThreadViolation> {
        let core = self.main_thread_core;
        let priority = self.main_thread_priority;
        if u32::from(core) >= core_count {
            return Err(MainThreadViolation::CoreOutOfRange { core, core_count });
        }
        let info = self
            .aci
            .kernel_capabilities
            .thread_info
            .as_ref()
            .ok_or(MainThreadViolation::NoThreadInfo)?;
        if !info.allows_core(core) {
            return Err(MainThreadViolation::CoreNotAllowed { core });
        }
        if priority > LOWEST_THREAD_PRIORITY {
            return Err(MainThreadViolation::PriorityOutOfRange { priority });
        }
        if !info.allows_priority(priority) {
            return Err(MainThreadViolation::PriorityNotAllowed { priority });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npdm::capabilities::{
        DebugFlags, KernelCapabilities, KernelVersion, MemoryMap, MemoryMapKind, RegionMap,
        RegionType, ThreadInfo,
    };
    use crate::npdm::service_access::ServiceAccess;
    use crate::npdm::{
        Aci0, Acid, AcidLayout, AddressSpaceType, BlockLocation, FileAccessControl,
        FileAccessHeader, Npdm, PoolPartition,
    };

    const TITLE_ID: u64 = 0x0100_0000_0000_1000;

    fn service(name: &str, server: bool, wildcard: bool) -> ServiceAccess {
        ServiceAccess {
            name: name.into(),
            server,
            wildcard,
        }
    }

    fn map(address: u64, size: u64, read_only: bool, kind: MemoryMapKind) -> MemoryMap {
        MemoryMap {
            address,
            size,
            read_only,
            kind,
        }
    }

    fn svc_mask(ids: &[u32]) -> crate::npdm::capabilities::SvcMask {
        // Built through the public decoder: one descriptor per SVC index.
        let mut words = Vec::new();
        for index in 0..8u32 {
            let mut mask = 0u32;
            for &id in ids {
                if id / 24 == index {
                    mask |= 1 << (id % 24);
                }
            }
            if mask != 0 {
                words.push(0b1111 | (mask << 5) | (index << 29));
            }
        }
        KernelCapabilities::parse(&words).unwrap().svc_mask
    }

    /// The signed bound: everything a request in [`request`] asks for, and more.
    fn bound() -> Acid {
        Acid {
            layout: AcidLayout::Switch1,
            production: true,
            unqualified_approval: false,
            pool_partition: PoolPartition::Application,
            title_id_min: TITLE_ID - 0x10,
            title_id_max: TITLE_ID + 0x10,
            file_access: FileAccessControl {
                version: 1,
                permissions: 0b1111,
                unknown: [0; 0x20],
            },
            services: vec![
                service("fs:ro", false, false),
                service("sm:m", true, false),
                service("ldr:", false, true),
                service("pl:u", false, false),
            ],
            kernel_capabilities: KernelCapabilities {
                thread_info: Some(ThreadInfo {
                    priorities: 20..=63,
                    cores: 0..=5,
                }),
                svc_mask: svc_mask(&[1, 2, 3, 0x30, 0x80]),
                memory_maps: vec![
                    map(0x10_0000, 0x10_0000, false, MemoryMapKind::Static),
                    map(0x40_0000, 0x1000, true, MemoryMapKind::Io),
                ],
                region_maps: vec![RegionMap {
                    region: RegionType::Dtb,
                    read_only: false,
                }],
                interrupts: vec![0x41, 0x42],
                program_type: Some(1),
                kernel_version: Some(KernelVersion { major: 9, minor: 1 }),
                handle_table_size: Some(0x200),
                debug_flags: Some(DebugFlags {
                    allow_debug: true,
                    force_debug: false,
                }),
                unknown: vec![0x1F | (0x55 << 6)],
            },
        }
    }

    /// A request that [`bound`] permits, touching every rule.
    fn request() -> Aci0 {
        Aci0 {
            title_id: TITLE_ID,
            file_access: FileAccessHeader {
                version: 1,
                permissions: 0b0101,
                unk_offset: 0,
                unk_size: 0,
                unk_offset_2: 0,
                unk_size_2: 0,
            },
            services: vec![
                service("fs:ro", false, false),
                service("sm:m", true, false),
                service("ldr:ro", false, false),
            ],
            kernel_capabilities: KernelCapabilities {
                thread_info: Some(ThreadInfo {
                    priorities: 28..=59,
                    cores: 0..=3,
                }),
                svc_mask: svc_mask(&[1, 3, 0x80]),
                memory_maps: vec![
                    map(0x10_1000, 0x2000, true, MemoryMapKind::Static),
                    map(0x40_0000, 0x1000, true, MemoryMapKind::Io),
                ],
                region_maps: vec![RegionMap {
                    region: RegionType::Dtb,
                    read_only: true,
                }],
                interrupts: vec![0x42],
                program_type: Some(1),
                kernel_version: Some(KernelVersion { major: 9, minor: 1 }),
                handle_table_size: Some(0x100),
                debug_flags: Some(DebugFlags {
                    allow_debug: true,
                    force_debug: false,
                }),
                unknown: vec![0x1F | (0x55 << 6)],
            },
        }
    }

    /// Applies `edit` to a copy of [`request`] and returns what the bound says about it.
    fn verdict(edit: impl FnOnce(&mut Aci0)) -> Result<(), AcidViolation> {
        let mut aci = request();
        edit(&mut aci);
        bound().permits(&aci)
    }

    #[test]
    fn a_request_inside_the_bound_is_permitted() {
        assert_eq!(bound().permits(&request()), Ok(()));
        // An empty request is inside any bound.
        let empty = Aci0 {
            title_id: TITLE_ID,
            file_access: request().file_access,
            services: vec![],
            kernel_capabilities: KernelCapabilities::default(),
        };
        assert_eq!(
            bound().permits(&Aci0 {
                file_access: FileAccessHeader {
                    permissions: 0,
                    ..empty.file_access
                },
                ..empty
            }),
            Ok(())
        );
    }

    #[test]
    fn title_id_must_be_in_the_acid_range() {
        let acid = bound();
        let at = |title_id| verdict(|aci| aci.title_id = title_id);
        assert_eq!(at(acid.title_id_min), Ok(()));
        assert_eq!(at(acid.title_id_max), Ok(()));
        for title_id in [acid.title_id_min - 1, acid.title_id_max + 1] {
            assert_eq!(
                at(title_id),
                Err(AcidViolation::TitleIdOutOfRange {
                    title_id,
                    min: acid.title_id_min,
                    max: acid.title_id_max
                })
            );
        }
    }

    #[test]
    fn file_permissions_must_be_a_subset() {
        assert_eq!(verdict(|aci| aci.file_access.permissions = 0b1111), Ok(()));
        assert_eq!(
            verdict(|aci| aci.file_access.permissions = 0b1_0101),
            Err(AcidViolation::FilePermissionsNotAllowed {
                requested: 0b1_0101,
                allowed: 0b1111
            })
        );
    }

    #[test]
    fn services_must_be_covered() {
        assert_eq!(
            verdict(|aci| aci.services.push(service("pl:u", false, false))),
            Ok(())
        );
        // Not listed, listed with the other role, and a wildcard that widens an exact grant.
        for rejected in [
            service("set:sys", false, false),
            service("fs:ro", true, false),
            service("fs:", false, true),
        ] {
            assert_eq!(
                verdict(|aci| aci.services.push(rejected.clone())),
                Err(AcidViolation::ServiceNotAllowed {
                    name: rejected.name,
                    server: rejected.server,
                    wildcard: rejected.wildcard
                })
            );
        }
        // The first uncovered entry is the one reported.
        assert_eq!(
            verdict(|aci| {
                aci.services.push(service("aaa", false, false));
                aci.services.push(service("bbb", false, false));
            }),
            Err(AcidViolation::ServiceNotAllowed {
                name: "aaa".into(),
                server: false,
                wildcard: false
            })
        );
    }

    #[test]
    fn core_and_priority_ranges_must_be_inside() {
        let inside = |priorities, cores| {
            verdict(|aci| {
                aci.kernel_capabilities.thread_info = Some(ThreadInfo { priorities, cores });
            })
        };
        assert_eq!(inside(20..=63, 0..=5), Ok(()));
        assert_eq!(
            inside(20..=63, 0..=6),
            Err(AcidViolation::CoresNotAllowed {
                requested: 0..=6,
                allowed: Some(0..=5)
            })
        );
        assert_eq!(
            inside(19..=63, 0..=5),
            Err(AcidViolation::PrioritiesNotAllowed {
                requested: 19..=63,
                allowed: Some(20..=63)
            })
        );
        // The bound declares no ranges at all.
        let mut acid = bound();
        acid.kernel_capabilities.thread_info = None;
        assert_eq!(
            acid.permits(&request()),
            Err(AcidViolation::CoresNotAllowed {
                requested: 0..=3,
                allowed: None
            })
        );
        // Declaring none is always fine.
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.thread_info = None),
            Ok(())
        );
    }

    #[test]
    fn svcs_must_be_a_subset() {
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.svc_mask = svc_mask(&[1, 2, 3, 0x30, 0x80])),
            Ok(())
        );
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.svc_mask = svc_mask(&[1, 4, 5])),
            Err(AcidViolation::SvcNotAllowed { id: 4 })
        );
    }

    #[test]
    fn memory_maps_must_sit_inside_a_grant_of_the_same_kind() {
        let one =
            |request: MemoryMap| verdict(|aci| aci.kernel_capabilities.memory_maps = vec![request]);
        let denied = |request: MemoryMap| {
            Err(AcidViolation::MemoryMapNotAllowed {
                address: request.address,
                size: request.size,
            })
        };
        // Exactly the grant, and the grant's last page.
        assert_eq!(
            one(map(0x10_0000, 0x10_0000, false, MemoryMapKind::Static)),
            Ok(())
        );
        assert_eq!(
            one(map(0x1F_F000, 0x1000, false, MemoryMapKind::Static)),
            Ok(())
        );
        // One page past the end, one page before the start, the wrong kind, and writable where
        // the grant is read-only.
        let past = map(0x1F_F000, 0x2000, false, MemoryMapKind::Static);
        assert_eq!(one(past), denied(past));
        let before = map(0xF_F000, 0x1000, false, MemoryMapKind::Static);
        assert_eq!(one(before), denied(before));
        let wrong_kind = map(0x10_0000, 0x1000, false, MemoryMapKind::Io);
        assert_eq!(one(wrong_kind), denied(wrong_kind));
        let writable = map(0x40_0000, 0x1000, false, MemoryMapKind::Io);
        assert_eq!(one(writable), denied(writable));
        // Sums that overflow 64 bits are never inside a grant.
        let wrap = map(u64::MAX - 0xFFF, 0x2000, false, MemoryMapKind::Static);
        assert_eq!(one(wrap), denied(wrap));
        let mut acid = bound();
        acid.kernel_capabilities.memory_maps =
            vec![map(u64::MAX - 0xFFF, 0x2000, false, MemoryMapKind::Static)];
        let inside_wrapped = map(u64::MAX - 0xFFF, 0x1000, false, MemoryMapKind::Static);
        let mut aci = request();
        aci.kernel_capabilities.memory_maps = vec![inside_wrapped];
        assert_eq!(
            acid.permits(&aci),
            Err(AcidViolation::MemoryMapNotAllowed {
                address: inside_wrapped.address,
                size: inside_wrapped.size
            })
        );
    }

    #[test]
    fn regions_and_interrupts_must_be_granted() {
        let region = |region, read_only| {
            verdict(|aci| {
                aci.kernel_capabilities.region_maps = vec![RegionMap { region, read_only }];
            })
        };
        assert_eq!(region(RegionType::Dtb, false), Ok(()));
        assert_eq!(
            region(RegionType::KernelTraceBuffer, true),
            Err(AcidViolation::RegionNotAllowed {
                region: RegionType::KernelTraceBuffer
            })
        );
        let mut acid = bound();
        acid.kernel_capabilities.region_maps[0].read_only = true;
        let mut aci = request();
        aci.kernel_capabilities.region_maps[0].read_only = false;
        assert_eq!(
            acid.permits(&aci),
            Err(AcidViolation::RegionNotAllowed {
                region: RegionType::Dtb
            })
        );

        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.interrupts = vec![0x41, 0x42]),
            Ok(())
        );
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.interrupts = vec![0x42, 0x43]),
            Err(AcidViolation::InterruptNotAllowed { id: 0x43 })
        );
    }

    #[test]
    fn scalar_capabilities_must_match_or_stay_below() {
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.program_type = Some(2)),
            Err(AcidViolation::ProgramTypeMismatch {
                requested: 2,
                allowed: Some(1)
            })
        );
        let version = KernelVersion { major: 9, minor: 2 };
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.kernel_version = Some(version)),
            Err(AcidViolation::KernelVersionMismatch {
                requested: version,
                allowed: Some(KernelVersion { major: 9, minor: 1 })
            })
        );
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.handle_table_size = Some(0x200)),
            Ok(())
        );
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.handle_table_size = Some(0x201)),
            Err(AcidViolation::HandleTableTooLarge {
                requested: 0x201,
                allowed: Some(0x200)
            })
        );
        // The bound declares nothing: any request for the capability is refused.
        let mut acid = bound();
        acid.kernel_capabilities.program_type = None;
        acid.kernel_capabilities.handle_table_size = None;
        assert_eq!(
            acid.permits(&request()),
            Err(AcidViolation::ProgramTypeMismatch {
                requested: 1,
                allowed: None
            })
        );
        acid.kernel_capabilities.program_type = Some(1);
        assert_eq!(
            acid.permits(&request()),
            Err(AcidViolation::HandleTableTooLarge {
                requested: 0x100,
                allowed: None
            })
        );
    }

    #[test]
    fn debug_flags_cannot_exceed_the_bound() {
        let flags = |allow_debug, force_debug| {
            verdict(|aci| {
                aci.kernel_capabilities.debug_flags = Some(DebugFlags {
                    allow_debug,
                    force_debug,
                });
            })
        };
        assert_eq!(flags(false, false), Ok(()));
        assert_eq!(flags(true, false), Ok(()));
        assert_eq!(flags(true, true), Err(AcidViolation::DebugFlagsNotAllowed));
        assert_eq!(flags(false, true), Err(AcidViolation::DebugFlagsNotAllowed));
        // A bound without debug flags permits only an all-clear request.
        let mut acid = bound();
        acid.kernel_capabilities.debug_flags = None;
        assert_eq!(
            acid.permits(&request()),
            Err(AcidViolation::DebugFlagsNotAllowed)
        );
        let mut aci = request();
        aci.kernel_capabilities.debug_flags = Some(DebugFlags {
            allow_debug: false,
            force_debug: false,
        });
        assert_eq!(acid.permits(&aci), Ok(()));
    }

    #[test]
    fn unknown_descriptors_must_appear_verbatim() {
        let other = 0x1F | (0x56 << 6);
        assert_eq!(
            verdict(|aci| aci.kernel_capabilities.unknown = vec![other]),
            Err(AcidViolation::UnknownCapabilityNotAllowed { raw: other })
        );
    }

    /// An `Npdm` shell around the given thread info and main thread, for the main-thread rules.
    fn npdm(core: u8, priority: u8, thread_info: Option<ThreadInfo>) -> Npdm {
        let mut aci = request();
        aci.kernel_capabilities.thread_info = thread_info;
        Npdm {
            is_64_bit: true,
            address_space: AddressSpaceType::Is39Bit,
            main_thread_priority: priority,
            main_thread_core: core,
            system_resource_size: 0,
            process_category: 0,
            main_thread_stack_size: 0,
            name: String::new(),
            aci0_location: BlockLocation { offset: 0, size: 0 },
            acid_location: BlockLocation { offset: 0, size: 0 },
            acid: bound(),
            aci,
        }
    }

    #[test]
    fn main_thread_must_fit_the_core_count_and_the_capability_ranges() {
        let info = || {
            Some(ThreadInfo {
                priorities: 28..=59,
                cores: 1..=3,
            })
        };
        assert_eq!(npdm(1, 28, info()).check_main_thread(4), Ok(()));
        assert_eq!(npdm(3, 59, info()).check_main_thread(4), Ok(()));
        assert_eq!(
            npdm(3, 28, info()).check_main_thread(3),
            Err(MainThreadViolation::CoreOutOfRange {
                core: 3,
                core_count: 3
            })
        );
        assert_eq!(
            npdm(0, 28, info()).check_main_thread(4),
            Err(MainThreadViolation::CoreNotAllowed { core: 0 })
        );
        // The lowest priority (numerically largest) is still a priority.
        let to_lowest = Some(ThreadInfo {
            priorities: 28..=LOWEST_THREAD_PRIORITY,
            cores: 1..=3,
        });
        assert_eq!(
            npdm(1, LOWEST_THREAD_PRIORITY, to_lowest).check_main_thread(4),
            Ok(())
        );
        assert_eq!(
            npdm(1, 64, info()).check_main_thread(4),
            Err(MainThreadViolation::PriorityOutOfRange { priority: 64 })
        );
        assert_eq!(
            npdm(1, 27, info()).check_main_thread(4),
            Err(MainThreadViolation::PriorityNotAllowed { priority: 27 })
        );
        assert_eq!(
            npdm(1, 28, None).check_main_thread(4),
            Err(MainThreadViolation::NoThreadInfo)
        );
        // A core count of zero or a huge one never panics.
        let _ = npdm(255, 255, info()).check_main_thread(0);
        let _ = npdm(255, 255, info()).check_main_thread(u32::MAX);
    }

    #[test]
    fn npdm_validation_checks_the_parsed_blocks() {
        let mut file = npdm(1, 28, None);
        assert_eq!(file.validate_against_acid(), Ok(()));
        file.aci.title_id = file.acid.title_id_max + 1;
        assert!(matches!(
            file.validate_against_acid(),
            Err(AcidViolation::TitleIdOutOfRange { .. })
        ));
    }
}
