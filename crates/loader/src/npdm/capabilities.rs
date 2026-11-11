//! Kernel capability descriptors (KAC): the 32-bit words in an NPDM that tell the kernel which
//! cores, priorities, SVCs and memory a title may use.
//!
//! Decoded into [`KernelCapabilities`]. The encoding is ported from
//! `eden/src/core/hle/kernel/k_capabilities.{h,cpp}` @ cf654052 (GPL-3.0-or-later): a
//! descriptor's type is the number of trailing one bits, `0xFFFF_FFFF` is padding, and each type
//! has its own bit fields (see the `*Shift`-style constants below, each cited to the Eden union it
//! comes from). `eden-nx2/tools/nx2/npdm_info.py` is the independent reader the layouts were
//! cross-checked against.
//!
//! This decodes and validates structure; it does not enforce policy (the kernel decides whether a
//! core id or physical address is acceptable), except where Eden rejects a descriptor outright:
//! reserved bits set, a map-range without its pair, a zero kernel major version, duplicate
//! once-only descriptors and reversed core or priority ranges.
//!
//! Unknown descriptor types are collected in [`KernelCapabilities::unknown`] rather than rejected,
//! because Switch 2 titles may add types that the Eden reference predates. A word with bit 0
//! clear has no type at all and is an error, as in Eden (`CapabilityType::Invalid`).

use std::ops::RangeInclusive;

/// Padding word, skipped (`CapabilityType::Padding`).
const PADDING: u32 = u32::MAX;
/// Interrupt id meaning "no interrupt" (`KCapabilities::PaddingInterruptId`).
const PADDING_INTERRUPT_ID: u32 = 0x3FF;
/// Size of a physical page for address and size fields (`PageSize` in `k_capabilities.cpp`).
const PAGE_SIZE: u64 = 0x1000;

// Descriptor types, as trailing-one counts: `CapabilityType` is `(1 << n) - 1`.
const TYPE_THREAD_INFO: u32 = 3;
const TYPE_SYSCALL_MASK: u32 = 4;
const TYPE_MAP_RANGE: u32 = 6;
const TYPE_MAP_IO_PAGE: u32 = 7;
const TYPE_MAP_REGION: u32 = 10;
const TYPE_INTERRUPT_PAIR: u32 = 11;
const TYPE_PROGRAM_TYPE: u32 = 13;
const TYPE_KERNEL_VERSION: u32 = 14;
const TYPE_HANDLE_TABLE: u32 = 15;
const TYPE_DEBUG_FLAGS: u32 = 16;

/// Number of SVC ids the mask can express: 8 indices of 24 bits (`SyscallMask::index` is 3 bits,
/// `SyscallMask::mask` 24).
pub const SVC_ID_LIMIT: u32 = 8 * 24;
/// Bits per `SyscallMask` descriptor.
const SVC_BITS_PER_INDEX: u32 = 24;

/// A set of SVC numbers, decoded from the syscall-mask descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SvcMask {
    words: [u64; 3],
}

impl SvcMask {
    /// True if SVC `id` is allowed.
    pub fn contains(&self, id: u32) -> bool {
        if id >= SVC_ID_LIMIT {
            return false;
        }
        let word = usize::try_from(id / 64).unwrap_or(usize::MAX);
        self.words
            .get(word)
            .is_some_and(|bits| bits & (1u64 << (id % 64)) != 0)
    }

    /// The allowed SVC numbers in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..SVC_ID_LIMIT).filter(|&id| self.contains(id))
    }

    fn insert(&mut self, id: u32) {
        if id >= SVC_ID_LIMIT {
            return;
        }
        let word = usize::try_from(id / 64).unwrap_or(usize::MAX);
        if let Some(bits) = self.words.get_mut(word) {
            *bits |= 1u64 << (id % 64);
        }
    }
}

/// Core and priority ranges the title may use (`KCapabilities::CorePriority`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadInfo {
    /// Thread priorities allowed, lowest number (highest priority) to highest number.
    pub priorities: RangeInclusive<u8>,
    /// Cores allowed, lowest to highest.
    pub cores: RangeInclusive<u8>,
}

/// What kind of mapping a [`MemoryMap`] asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMapKind {
    /// A map-range pair with the "normal" bit set: static physical memory
    /// (`MapRangeSize::normal`, `MapStatic` in Eden).
    Static,
    /// A map-range pair with the "normal" bit clear: device (IO) memory (`MapIo` in Eden).
    Io,
    /// A single IO page (`KCapabilities::MapIoPage`); always one page, read-write.
    IoPage,
}

/// A physical memory range the title asks to have mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryMap {
    /// Physical address in bytes (the page number times 0x1000).
    pub address: u64,
    /// Size in bytes.
    pub size: u64,
    /// True if the mapping is read-only.
    pub read_only: bool,
    /// Which descriptor form this came from.
    pub kind: MemoryMapKind,
}

/// A kernel memory region a title may map (`KCapabilities::RegionType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    /// The kernel trace buffer.
    KernelTraceBuffer,
    /// The on-memory boot image.
    OnMemoryBootImage,
    /// The device tree blob.
    Dtb,
    /// A region value that Eden rejects and this crate does not know.
    Other(u8),
}

/// One region mapping from a map-region descriptor, which holds up to three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionMap {
    /// The region.
    pub region: RegionType,
    /// True if the mapping is read-only.
    pub read_only: bool,
}

/// The intended kernel version (`KCapabilities::KernelVersion`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelVersion {
    /// Major version; never zero in a valid descriptor.
    pub major: u16,
    /// Minor version.
    pub minor: u8,
}

/// Debug permissions (`KCapabilities::DebugFlags`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugFlags {
    /// The process may be debugged.
    pub allow_debug: bool,
    /// The process may force other processes to be debugged.
    pub force_debug: bool,
}

/// Decoded kernel capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelCapabilities {
    /// Core and priority ranges, if the title declared them.
    pub thread_info: Option<ThreadInfo>,
    /// Allowed SVC numbers.
    pub svc_mask: SvcMask,
    /// Physical memory maps (map-range pairs and IO pages), in descriptor order.
    pub memory_maps: Vec<MemoryMap>,
    /// Region maps, in descriptor order.
    pub region_maps: Vec<RegionMap>,
    /// Interrupt ids the title may bind (padding ids removed).
    pub interrupts: Vec<u16>,
    /// Program type, a 3-bit value.
    pub program_type: Option<u8>,
    /// Intended kernel version.
    pub kernel_version: Option<KernelVersion>,
    /// Handle table size.
    pub handle_table_size: Option<u16>,
    /// Debug flags.
    pub debug_flags: Option<DebugFlags>,
    /// Descriptors whose type this crate does not know, verbatim.
    pub unknown: Vec<u32>,
}

/// Why a capability list could not be decoded. `index` is the word index in the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CapabilityError {
    /// A word with bit 0 clear, which has no descriptor type.
    #[error("capability {index} ({raw:#010x}) has no descriptor type")]
    InvalidDescriptor {
        /// Word index.
        index: usize,
        /// The word.
        raw: u32,
    },
    /// A map-range descriptor is last in the list, so it has no size word.
    #[error("map-range capability {index} has no size word")]
    MissingMapRangePair {
        /// Word index of the address word.
        index: usize,
    },
    /// The word after a map-range is not itself a map-range descriptor.
    #[error("map-range capability {index} is followed by a different descriptor type")]
    BadMapRangePair {
        /// Word index of the address word.
        index: usize,
    },
    /// A map-range maps zero pages, or its size word sets reserved bits.
    #[error("map-range capability {index} has zero pages or reserved bits set")]
    InvalidMapRange {
        /// Word index of the address word.
        index: usize,
    },
    /// Reserved bits are set in a program-type or handle-table descriptor.
    #[error("capability {index} sets reserved bits")]
    ReservedBitsSet {
        /// Word index.
        index: usize,
    },
    /// A descriptor that may appear once appears again (or an SVC index repeats).
    #[error("capability {index} repeats a descriptor that may appear only once")]
    Duplicate {
        /// Word index of the repeat.
        index: usize,
    },
    /// The kernel version descriptor has a major version of zero.
    #[error("kernel version capability {index} has major version 0")]
    ZeroKernelVersion {
        /// Word index.
        index: usize,
    },
    /// The core or priority range has its minimum above its maximum.
    #[error("thread info capability {index} has a reversed core or priority range")]
    ReversedRange {
        /// Word index.
        index: usize,
    },
}

impl KernelCapabilities {
    /// Decodes a list of capability words.
    pub fn parse(words: &[u32]) -> Result<Self, CapabilityError> {
        let mut caps = Self::default();
        let mut svc_indices_seen = 0u32;
        let mut index = 0usize;
        while let Some(&raw) = words.get(index) {
            let at = index;
            index += 1;
            if raw == PADDING {
                continue;
            }
            match (!raw).trailing_zeros() {
                0 => return Err(CapabilityError::InvalidDescriptor { index: at, raw }),
                TYPE_THREAD_INFO => caps.add_thread_info(raw, at)?,
                TYPE_SYSCALL_MASK => caps.add_syscall_mask(raw, at, &mut svc_indices_seen)?,
                TYPE_MAP_RANGE => {
                    let size_word = *words
                        .get(index)
                        .ok_or(CapabilityError::MissingMapRangePair { index: at })?;
                    index += 1;
                    caps.add_map_range(raw, size_word, at)?;
                }
                TYPE_MAP_IO_PAGE => {
                    // MapIoPage: address bits 8..32.
                    caps.memory_maps.push(MemoryMap {
                        address: u64::from(field(raw, 8, 24)) * PAGE_SIZE,
                        size: PAGE_SIZE,
                        read_only: false,
                        kind: MemoryMapKind::IoPage,
                    });
                }
                TYPE_MAP_REGION => caps.add_regions(raw),
                TYPE_INTERRUPT_PAIR => {
                    // InterruptPair: ids at bits 12..22 and 22..32.
                    for shift in [12, 22] {
                        let id = field(raw, shift, 10);
                        if id != PADDING_INTERRUPT_ID {
                            caps.interrupts.push(narrow(id));
                        }
                    }
                }
                TYPE_PROGRAM_TYPE => {
                    // ProgramType: type bits 14..17, reserved bits 17..32.
                    if caps.program_type.is_some() {
                        return Err(CapabilityError::Duplicate { index: at });
                    }
                    if field(raw, 17, 15) != 0 {
                        return Err(CapabilityError::ReservedBitsSet { index: at });
                    }
                    caps.program_type = Some(narrow(field(raw, 14, 3)));
                }
                TYPE_KERNEL_VERSION => {
                    // KernelVersion: minor bits 15..19, major bits 19..32.
                    if caps.kernel_version.is_some() {
                        return Err(CapabilityError::Duplicate { index: at });
                    }
                    let major = field(raw, 19, 13);
                    if major == 0 {
                        return Err(CapabilityError::ZeroKernelVersion { index: at });
                    }
                    caps.kernel_version = Some(KernelVersion {
                        major: narrow(major),
                        minor: narrow(field(raw, 15, 4)),
                    });
                }
                TYPE_HANDLE_TABLE => {
                    // HandleTable: size bits 16..26, reserved bits 26..32.
                    if caps.handle_table_size.is_some() {
                        return Err(CapabilityError::Duplicate { index: at });
                    }
                    if field(raw, 26, 6) != 0 {
                        return Err(CapabilityError::ReservedBitsSet { index: at });
                    }
                    caps.handle_table_size = Some(narrow(field(raw, 16, 10)));
                }
                TYPE_DEBUG_FLAGS => {
                    // DebugFlags: allow bit 17, force bit 18. Eden ignores the reserved bits
                    // (19..32) because enforcing them breaks homebrew loaders; so does this.
                    if caps.debug_flags.is_some() {
                        return Err(CapabilityError::Duplicate { index: at });
                    }
                    caps.debug_flags = Some(DebugFlags {
                        allow_debug: field(raw, 17, 1) != 0,
                        force_debug: field(raw, 18, 1) != 0,
                    });
                }
                _ => caps.unknown.push(raw),
            }
        }
        Ok(caps)
    }

    /// `CorePriority`: `lowest_thread_priority` (bits 4..10) is the numerically largest priority and
    /// `highest_thread_priority` (bits 10..16) the smallest; cores are bits 16..24 and 24..32.
    /// `GetDefault` in `program_metadata.cpp` documents 0x30043F7 as cores 0..3 and priorities
    /// 16..63, which pins the orientation.
    fn add_thread_info(&mut self, raw: u32, index: usize) -> Result<(), CapabilityError> {
        if self.thread_info.is_some() {
            return Err(CapabilityError::Duplicate { index });
        }
        let max_priority = field(raw, 4, 6);
        let min_priority = field(raw, 10, 6);
        let min_core = field(raw, 16, 8);
        let max_core = field(raw, 24, 8);
        if min_priority > max_priority || min_core > max_core {
            return Err(CapabilityError::ReversedRange { index });
        }
        self.thread_info = Some(ThreadInfo {
            priorities: narrow(min_priority)..=narrow(max_priority),
            cores: narrow(min_core)..=narrow(max_core),
        });
        Ok(())
    }

    /// `SyscallMask`: mask bits 5..29, index bits 29..32. Each index may appear once.
    fn add_syscall_mask(
        &mut self,
        raw: u32,
        index: usize,
        seen: &mut u32,
    ) -> Result<(), CapabilityError> {
        let mask = field(raw, 5, 24);
        let svc_index = field(raw, 29, 3);
        let flag = 1u32 << svc_index;
        if *seen & flag != 0 {
            return Err(CapabilityError::Duplicate { index });
        }
        *seen |= flag;
        for bit in 0..SVC_BITS_PER_INDEX {
            if mask & (1 << bit) != 0 {
                self.svc_mask.insert(SVC_BITS_PER_INDEX * svc_index + bit);
            }
        }
        Ok(())
    }

    /// `MapRange`: address bits 7..31, `read_only` bit 31. Its pair, `MapRangeSize`: pages bits 7..27,
    /// reserved bits 27..31, normal bit 31.
    fn add_map_range(
        &mut self,
        raw: u32,
        size_word: u32,
        index: usize,
    ) -> Result<(), CapabilityError> {
        if (!size_word).trailing_zeros() != TYPE_MAP_RANGE {
            return Err(CapabilityError::BadMapRangePair { index });
        }
        let pages = field(size_word, 7, 20);
        if pages == 0 || field(size_word, 27, 4) != 0 {
            return Err(CapabilityError::InvalidMapRange { index });
        }
        self.memory_maps.push(MemoryMap {
            address: u64::from(field(raw, 7, 24)) * PAGE_SIZE,
            size: u64::from(pages) * PAGE_SIZE,
            read_only: field(raw, 31, 1) != 0,
            kind: if field(size_word, 31, 1) != 0 {
                MemoryMapKind::Static
            } else {
                MemoryMapKind::Io
            },
        });
        Ok(())
    }

    /// `MapRegion`: (region, `read_only`) pairs at bits 11/17, 18/24 and 25/31.
    fn add_regions(&mut self, raw: u32) {
        for (region_shift, ro_shift) in [(11, 17), (18, 24), (25, 31)] {
            let region = match field(raw, region_shift, 6) {
                0 => continue,
                1 => RegionType::KernelTraceBuffer,
                2 => RegionType::OnMemoryBootImage,
                3 => RegionType::Dtb,
                other => RegionType::Other(narrow(other)),
            };
            self.region_maps.push(RegionMap {
                region,
                read_only: field(raw, ro_shift, 1) != 0,
            });
        }
    }
}

/// `width` bits of `value` starting at bit `shift`.
fn field(value: u32, shift: u32, width: u32) -> u32 {
    (value >> shift) & ((1u32 << width) - 1)
}

/// Narrows a field value that is known to fit (the callers extract at most 13 bits).
fn narrow<T: TryFrom<u32> + Default>(value: u32) -> T {
    T::try_from(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_5143;

    // Test-side encoders: build descriptors from fields, independently of the decoder.
    fn thread_info(min_prio: u32, max_prio: u32, min_core: u32, max_core: u32) -> u32 {
        0b111 | (max_prio << 4) | (min_prio << 10) | (min_core << 16) | (max_core << 24)
    }
    fn syscall_mask(index: u32, mask: u32) -> u32 {
        0b1111 | (mask << 5) | (index << 29)
    }
    fn kernel_version(major: u32, minor: u32) -> u32 {
        0x3FFF | (minor << 15) | (major << 19)
    }

    #[test]
    fn core_range_0_to_5_and_priorities_28_to_59() {
        // Raw word asserted directly: 59 << 4 | 28 << 10 | 0 << 16 | 5 << 24 | 0b111.
        let raw = 0x0500_73B7;
        assert_eq!(raw, thread_info(28, 59, 0, 5));
        let caps = KernelCapabilities::parse(&[raw]).unwrap();
        let info = caps.thread_info.unwrap();
        assert_eq!(info.cores, 0..=5);
        assert_eq!(info.priorities, 28..=59);
    }

    #[test]
    fn eden_default_is_cores_0_to_3_and_priorities_16_to_63() {
        // `ProgramMetadata::GetDefault`: "cores 0~3 and thread priorities 16~63".
        let caps = KernelCapabilities::parse(&[0x0300_43F7]).unwrap();
        let info = caps.thread_info.unwrap();
        assert_eq!(info.cores, 0..=3);
        assert_eq!(info.priorities, 16..=63);
    }

    #[test]
    fn svc_mask_bits_land_on_the_right_svc_numbers() {
        // SVCs 0x01..=0x36 and 0x80, as in the Switch 2 titles in the NX2 notes.
        let mut words = Vec::new();
        for index in 0..3u32 {
            let mut mask = 0u32;
            for bit in 0..24 {
                let id = index * 24 + bit;
                if (0x01..=0x36).contains(&id) {
                    mask |= 1 << bit;
                }
            }
            words.push(syscall_mask(index, mask));
        }
        // Raw word for index 0, SVCs 1..=23: mask 0xFFFFFE << 5, type bits 0xF.
        assert_eq!(words[0], 0x1FFF_FFCF);
        words.push(syscall_mask(5, 1 << 8));
        let caps = KernelCapabilities::parse(&words).unwrap();
        let allowed: Vec<u32> = caps.svc_mask.iter().collect();
        let expected: Vec<u32> = (0x01..=0x36).chain([0x80]).collect();
        assert_eq!(allowed, expected);
        assert!(caps.svc_mask.contains(0x36));
        assert!(!caps.svc_mask.contains(0x37));
        assert!(!caps.svc_mask.contains(0));
        assert!(caps.svc_mask.contains(0x80));
        assert!(!caps.svc_mask.contains(SVC_ID_LIMIT));
    }

    #[test]
    fn highest_svc_index_decodes() {
        let caps = KernelCapabilities::parse(&[syscall_mask(7, 1 << 23)]).unwrap();
        assert_eq!(caps.svc_mask.iter().collect::<Vec<_>>(), [SVC_ID_LIMIT - 1]);
    }

    #[test]
    fn decodes_the_remaining_descriptor_types() {
        let words = [
            // MapRange: page 0x1234, read-only; size 0x10 pages, normal.
            0x3F | (0x1234 << 7) | (1 << 31),
            0x3F | (0x10 << 7) | (1 << 31),
            // MapRange (io): page 7; size 2 pages, not normal.
            0x3F | (7 << 7),
            0x3F | (2 << 7),
            // MapIoPage at page 0xABCDE.
            0x7F | (0xABCDE << 8),
            // MapRegion: DTB read-only in slot 0, OnMemoryBootImage read-write in slot 1.
            0x3FF | (3 << 11) | (1 << 17) | (2 << 18),
            // Interrupt pair: 0x41 and padding.
            0x7FF | (0x41 << 12) | (PADDING_INTERRUPT_ID << 22),
            // ProgramType 1.
            0x1FFF | (1 << 14),
            kernel_version(9, 1),
            // HandleTable 0x200.
            0x7FFF | (0x200 << 16),
            // DebugFlags: allow + force.
            0xFFFF | (1 << 17) | (1 << 18),
            PADDING,
        ];
        let caps = KernelCapabilities::parse(&words).unwrap();
        assert_eq!(
            caps.memory_maps,
            [
                MemoryMap {
                    address: 0x1234 * PAGE_SIZE,
                    size: 0x10 * PAGE_SIZE,
                    read_only: true,
                    kind: MemoryMapKind::Static
                },
                MemoryMap {
                    address: 7 * PAGE_SIZE,
                    size: 2 * PAGE_SIZE,
                    read_only: false,
                    kind: MemoryMapKind::Io
                },
                MemoryMap {
                    address: 0xABCDE * PAGE_SIZE,
                    size: PAGE_SIZE,
                    read_only: false,
                    kind: MemoryMapKind::IoPage
                },
            ]
        );
        assert_eq!(
            caps.region_maps,
            [
                RegionMap {
                    region: RegionType::Dtb,
                    read_only: true
                },
                RegionMap {
                    region: RegionType::OnMemoryBootImage,
                    read_only: false
                },
            ]
        );
        assert_eq!(caps.interrupts, [0x41]);
        assert_eq!(caps.program_type, Some(1));
        assert_eq!(
            caps.kernel_version,
            Some(KernelVersion { major: 9, minor: 1 })
        );
        assert_eq!(caps.handle_table_size, Some(0x200));
        assert_eq!(
            caps.debug_flags,
            Some(DebugFlags {
                allow_debug: true,
                force_debug: true
            })
        );
        assert_eq!(caps.unknown, Vec::<u32>::new());
    }

    #[test]
    fn unknown_descriptor_types_are_kept_verbatim() {
        // Five trailing ones: a type Eden does not define.
        let word = 0x1F | (0x55 << 6);
        let caps = KernelCapabilities::parse(&[word]).unwrap();
        assert_eq!(caps.unknown, [word]);
    }

    #[test]
    fn malformed_lists_are_errors() {
        use CapabilityError as E;
        let parse = KernelCapabilities::parse;
        assert_eq!(parse(&[0]), Err(E::InvalidDescriptor { index: 0, raw: 0 }));
        assert_eq!(
            parse(&[thread_info(1, 2, 0, 1), 0x2]),
            Err(E::InvalidDescriptor { index: 1, raw: 2 })
        );
        assert_eq!(
            parse(&[thread_info(1, 2, 0, 1), thread_info(1, 2, 0, 1)]),
            Err(E::Duplicate { index: 1 })
        );
        assert_eq!(
            parse(&[thread_info(40, 30, 0, 1)]),
            Err(E::ReversedRange { index: 0 })
        );
        assert_eq!(
            parse(&[thread_info(30, 40, 3, 1)]),
            Err(E::ReversedRange { index: 0 })
        );
        assert_eq!(
            parse(&[syscall_mask(1, 1), syscall_mask(1, 2)]),
            Err(E::Duplicate { index: 1 })
        );
        assert_eq!(parse(&[0x3F]), Err(E::MissingMapRangePair { index: 0 }));
        assert_eq!(parse(&[0x3F, 0x7F]), Err(E::BadMapRangePair { index: 0 }));
        assert_eq!(parse(&[0x3F, 0x3F]), Err(E::InvalidMapRange { index: 0 }));
        assert_eq!(
            parse(&[0x3F, 0x3F | (1 << 7) | (1 << 27)]),
            Err(E::InvalidMapRange { index: 0 })
        );
        assert_eq!(
            parse(&[0x1FFF | (1 << 17)]),
            Err(E::ReservedBitsSet { index: 0 })
        );
        assert_eq!(
            parse(&[0x7FFF | (1 << 26)]),
            Err(E::ReservedBitsSet { index: 0 })
        );
        assert_eq!(
            parse(&[kernel_version(0, 3)]),
            Err(E::ZeroKernelVersion { index: 0 })
        );
        assert_eq!(
            parse(&[kernel_version(9, 1), kernel_version(9, 2)]),
            Err(E::Duplicate { index: 1 })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]
        #[test]
        fn arbitrary_words_never_panic(words in proptest::collection::vec(any::<u32>(), 0..64)) {
            let _ = KernelCapabilities::parse(&words);
        }
    }
}
