//! Hypervisor.framework backend (macOS on Apple Silicon).
//!
//! Guest code runs natively at EL0 inside an HVF VM, in an address space built with
//! [`Stage1Tables`]. A small EL1 shim (`shim`) catches every EL0 exception: `svc`, aborts and
//! undefined instructions come back to the host as an [`Exit`], and EL0 reads of `cntpct_el0`
//! are answered in the guest from the virtual counter, without a host exit. `cntvct_el0` reads,
//! FP/SIMD and pointer authentication run natively.
//!
//! Design: `docs/specs/execution-model.md`, "Guest memory" ("Native backends: stage 1 is ours,
//! stage 2 is coarse").
//!
//! # Guest-physical memory (stage 2)
//!
//! Set up once, when the VM is created, and never changed:
//!
//! | Guest-physical range | What | Guest access through stage 2 |
//! |---|---|---|
//! | [`RamConfig`] (default 16 MiB at `0x1_0000_0000`) | guest RAM: the backing of every page a guest address space maps | read, write, execute |
//! | `0xf_0000_0000`, 16 KiB | the EL1 shim | read, execute |
//! | `0xf_0000_4000`, 16 KiB | the shim's own translation tables | read |
//! | `0xf_0000_8000`, 16 KiB | per-vCPU scratch slots (`SP_EL1`) | read, write |
//! | `0xf_1000_0000`, 4 MiB | the table pool: where guest translation tables live | read |
//!
//! [`HvfBackend::physical`] reads and writes RAM and the table pool by guest-physical address.
//! Stage 2 is the last line of defence, not the mechanism: the tables and the shim are read-only
//! there, so even a guest mapping that wrongly pointed at them could not change them. A guest
//! access stage 2 refuses stops the vCPU with an [`Error::Host`].
//!
//! PROVISIONAL: one RAM region of a size the caller picks, not Horizon's memory pools at fixed
//! bases (the spec's "Backing memory"), and the bases above are this backend's own choices
//! inside the 36-bit guest-physical space Hypervisor.framework provides
//! (`tools/hvprobe/README.md`, "Guest IPA space").
//!
//! # Address spaces (stage 1)
//!
//! - **The guest's** is whatever [`Vcpu::set_address_space`] was last given: the root of a set
//!   of [`Stage1Tables`] built over [`HvfBackend::physical`] with its pool inside
//!   [`HvfBackend::table_pool`], and an address-space id. It is the lower half of the virtual
//!   address space (`TTBR0_EL1`, 39 bits, 4 KiB granule). A vCPU that was never given one
//!   refuses to run ([`Error::NoAddressSpace`]).
//! - **The shim's** is the upper half (`TTBR1_EL1`): the shim's page and the scratch page,
//!   mapped by three static tables the backend builds once (`tables`). No guest table describes
//!   them, and EL0 can't touch them.
//!
//! Every page [`Stage1Tables`] writes is tagged with the current address-space id (`nG`), so
//! switching a vCPU between address spaces needs no flush as long as each id stays with its
//! tables. **Ids are as wide as the host makes them** ([`HvfVcpu::asid_bits`]): the spec's
//! interface takes 16 bits, but Hypervisor.framework on the Mac this was written on (Apple M2)
//! gives a guest 8 (`ID_AA64MMFR0_EL1.ASIDBits` = 0, `tools/hvprobe/README.md`), and ids that
//! differ only above that share cached translations. A wider id is refused
//! ([`Error::AsidTooWide`]), not truncated. Translations cached under an id outlive both exits
//! to the host and a change of tables, so reusing an id for other tables needs a flush first.
//!
//! # TLB invalidation
//!
//! After `unmap` or `protect` on live tables, run [`HvfVcpu::flush_pending`] on any vCPU of the
//! VM before the next change to those tables: it executes the invalidations the tables recorded
//! at EL1, in their broadcast form, which reaches every vCPU. A freed table page stays out of
//! use until that has happened ([`Stage1Tables::flush_pending_with`]). A change must never be
//! left unflushed: a vCPU that stays in the guest keeps a stale translation for hundreds of
//! milliseconds (`tools/hvprobe/README.md`, "TLB invalidation across vCPUs").
//!
//! # Threads
//!
//! Hypervisor.framework allows one VM per process, so only one [`HvfBackend`] can exist at a
//! time. A vCPU belongs to the thread that created it ([`HvfVcpu`] is not `Send`); other threads
//! stop it with an [`HvfInterrupter`].
//!
//! Grown from `tools/hvprobe` (`src/hv.rs`, `src/guest.rs`, `src/probe.rs`); measured behaviour
//! is in `tools/hvprobe/README.md`. System register fields are from the Arm Architecture
//! Reference Manual (Arm DDI 0487).

mod ffi;
mod flush;
#[cfg(test)]
mod hardware;
mod memory;
mod shim;
mod tables;

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_memory::stage1::{FlushRange, Stage1Tables, TableRoot};
use starquake_memory::{BackingRange, GuestMemory};

use crate::{Backend, Error, ExceptionClass, Exit, GuestRegs, Interrupter, Syndrome, Vcpu};
use ffi::VcpuId;
use flush::FlushOp;
use memory::{Extent, HOST_PAGE, HostRegion, Kind};
use tables::{AP_EL1_RO, AP_EL1_RW, PAGE, PXN, Region, UXN, VA_LIMIT};

pub use flush::MAX_PAGE_FLUSHES;
pub use memory::PhysicalMemory;

/// Default guest-physical base of RAM. Starquake's own choice, provisional: above 4 GiB, so the
/// output address of a RAM page has bits in both halves of its descriptor (a half-written
/// descriptor then names memory that does not exist), and away from every guest virtual address
/// the walking skeleton uses, so code that confuses the two kinds of address fails at once.
const DEFAULT_RAM_BASE: u64 = 0x1_0000_0000;
/// Default guest RAM size (`GUEST_SIZE` in `tools/hvprobe/src/probe.rs`).
const DEFAULT_RAM_SIZE: usize = 16 << 20;

/// Guest-physical base of the backend's own region. It sits at the top of the 36-bit IPA space
/// HVF provides by default (`tools/hvprobe/README.md`, "Guest IPA space"). Provisional.
const SYSTEM_BASE: u64 = 0xf_0000_0000;
/// Offset of the shim in the system region. Each part of the region starts on a host page, so
/// each has its own stage-2 permissions.
const CODE_OFFSET: u64 = 0;
/// Offset of the shim's translation tables, followed by the empty table.
const SYSTEM_TABLES_OFFSET: u64 = HOST_PAGE;
/// Tables that map the shim and the scratch page: one per level (see `tables`).
const SYSTEM_TABLES: usize = 3;
/// Offset of the per-vCPU scratch slots: each slot's top is that vCPU's `SP_EL1`.
const SCRATCH_OFFSET: u64 = 2 * HOST_PAGE;
/// Size of the system region.
const SYSTEM_SIZE: u64 = 3 * HOST_PAGE;
/// Bytes per scratch slot (16-byte aligned, as `SP_EL1` must be).
const SLOT_SIZE: u64 = 64;
/// vCPUs one VM supports: one scratch page's worth of slots.
const MAX_VCPUS: usize = 64;

/// Guest-physical base of the table pool. Provisional, like [`SYSTEM_BASE`].
const TABLE_POOL_BASE: u64 = 0xf_1000_0000;
/// Size of the table pool: 1024 table pages, enough to map about 2 GiB of scattered 2 MiB
/// windows. Starquake's own number, provisional until the kernel sizes address spaces.
const TABLE_POOL_SIZE: u64 = 4 << 20;

/// First address of the upper half of the address space: with `TCR_EL1.T1SZ` = 25 its
/// addresses have their top 25 bits set.
const UPPER_HALF: u64 = u64::MAX << 39;
/// The shim's offset into the upper half: the start of its last 2 MiB. This backend's choice.
const SHIM_HALF_OFFSET: u64 = VA_LIMIT - (2 << 20);
/// The scratch page's offset into the upper half, with unmapped pages between it and the shim.
const SCRATCH_HALF_OFFSET: u64 = SHIM_HALF_OFFSET + 0x1_0000;
/// Virtual address of the shim (`VBAR_EL1`).
const SHIM_VA: u64 = UPPER_HALF | SHIM_HALF_OFFSET;
/// Virtual address of the scratch page.
const SCRATCH_VA: u64 = UPPER_HALF | SCRATCH_HALF_OFFSET;

const _: () = assert!(shim::LEN as u64 <= PAGE);
const _: () = assert!(MAX_VCPUS as u64 * SLOT_SIZE <= PAGE);
const _: () = assert!(SLOT_SIZE >= shim::SCRATCH_BELOW_SP && SLOT_SIZE.is_multiple_of(16));
const _: () = assert!(SYSTEM_TABLES_OFFSET + SYSTEM_TABLES as u64 * PAGE <= SCRATCH_OFFSET);
const _: () = assert!(SCRATCH_OFFSET + PAGE <= SYSTEM_SIZE);
const _: () = assert!(SYSTEM_BASE + SYSTEM_SIZE <= TABLE_POOL_BASE);
const _: () = assert!(TABLE_POOL_SIZE.is_multiple_of(HOST_PAGE));

// EL1 configuration, from `Vcpu::new` in `tools/hvprobe/src/probe.rs` unless noted. Bit positions
// are from the Arm Architecture Reference Manual (DDI 0487).
/// `MAIR_EL1`: attribute 0 is normal memory, write-back cacheable.
const MAIR: u64 = 0xff;
/// `TCR_EL1`. For each half: a 39-bit address space (`T0SZ` = `T1SZ` = 25), a 4 KiB granule
/// (`TG0` = 0b00, `TG1` = 0b10), table walks in inner-shareable write-back memory. 40-bit `IPS`.
/// The address-space id comes from `TTBR0_EL1` (`A1` = 0). The probe ran the lower half only,
/// with upper-half walks disabled (`EPD1`); the upper-half fields and the absence of `EPD1` are
/// this backend's, checked by its hardware tests. [`tcr`] adds [`TCR_AS`] where it applies.
const TCR: u64 = 0x19 // T0SZ = 25
    | (1 << 8) | (1 << 10) | (3 << 12) // IRGN0, ORGN0, SH0
    | (0x19 << 16) // T1SZ = 25
    | (1 << 24) | (1 << 26) | (3 << 28) // IRGN1, ORGN1, SH1
    | (2 << 30) // TG1
    | (2 << 32); // IPS
/// `TCR_EL1.AS`: address-space ids are 16 bits wide. Reserved, and left clear, on a host that
/// has 8-bit ids.
const TCR_AS: u64 = 1 << 36;
/// `ID_AA64MMFR0_EL1.ASIDBits` (bits 7:4) is 0b0010 where ids are 16 bits wide, and 0b0000
/// where they are 8.
const ASID_BITS_16: u64 = 0b0010;

/// How many bits of address-space id a guest has, from its `ID_AA64MMFR0_EL1`. Any encoding
/// other than the one for 16 bits is taken as 8, the architecture's minimum.
fn asid_bits(mmfr0: u64) -> u8 {
    if (mmfr0 >> 4) & 0xf == ASID_BITS_16 {
        16
    } else {
        8
    }
}

/// `TCR_EL1` for a guest whose address-space ids have `asid_bits` bits.
fn tcr(asid_bits: u8) -> u64 {
    if asid_bits == 16 { TCR | TCR_AS } else { TCR }
}
/// `CPACR_EL1.FPEN` = 0b11: FP/SIMD doesn't trap at EL0 or EL1.
const CPACR_FP_ON: u64 = 3 << 20;
/// `CNTKCTL_EL1`: EL0VCTEN only, so EL0 reads of `cntpct_el0` trap to the shim
/// (`CNTKCTL_EL0_VCT_ONLY` in the probe).
const CNTKCTL_VCT_ONLY: u64 = 0b10;
/// `SCTLR_EL1`: RES1 bits, MMU and caches on, WFI/WFE not trapped, the four PAuth enables
/// (`EnIA`, `EnIB` from the probe; `EnDA`, `EnDB` added), and EL0 access to `CTR_EL0` (`UCT`),
/// `DC ZVA` (`DZE`) and cache maintenance by address (`UCI`), which ordinary user-space code
/// relies on. `WXN` is clear: a writable page may be executable, as the software walker assumes.
const SCTLR: u64 = 0x30d0_0800 // RES1
    | 1 | (1 << 2) | (1 << 12) // M, C, I
    | (1 << 16) | (1 << 18) // nTWI, nTWE
    | (1 << 31) | (1 << 30) // EnIA, EnIB
    | (1 << 27) | (1 << 13) // EnDA, EnDB
    | (1 << 15) | (1 << 14) | (1 << 26); // UCT, DZE, UCI

/// `PSTATE` bits the host can see and set: NZCV. The mode and interrupt masks are the backend's.
const PSTATE_EL0_MASK: u64 = 0xf000_0000;

/// Why a [`RamConfig`] is invalid, or the backend's own address space can't be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// Guest RAM has no bytes.
    #[error("guest RAM is empty")]
    Empty,
    /// An address or size isn't aligned as the hardware requires.
    #[error("{value:#x} is not aligned to {alignment:#x}")]
    Misaligned {
        /// The offending address or size.
        value: u64,
        /// The alignment it needs.
        alignment: u64,
    },
    /// The region doesn't fit below the backend's own region, or overflows.
    #[error("guest RAM must end at or below {limit:#x}", limit = SYSTEM_BASE)]
    OutOfRange,
    /// Two mappings cover the same virtual page.
    #[error("virtual page {va:#x} is mapped twice")]
    Overlap {
        /// The page mapped twice.
        va: u64,
    },
}

/// Where guest RAM goes: `size` bytes at guest-physical address `base`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamConfig {
    base: u64,
    size: u64,
}

impl RamConfig {
    /// Checks a RAM region: non-empty, `base` and `size` multiples of the 16 KiB host page, and
    /// ending at or below the backend's own region (`0xf_0000_0000`).
    pub fn new(base: u64, size: usize) -> Result<Self, LayoutError> {
        let size = u64::try_from(size).map_err(|_| LayoutError::OutOfRange)?;
        if size == 0 {
            return Err(LayoutError::Empty);
        }
        for value in [base, size] {
            if value % HOST_PAGE != 0 {
                return Err(LayoutError::Misaligned {
                    value,
                    alignment: HOST_PAGE,
                });
            }
        }
        match base.checked_add(size) {
            Some(end) if end <= SYSTEM_BASE => Ok(Self { base, size }),
            _ => Err(LayoutError::OutOfRange),
        }
    }
}

impl Default for RamConfig {
    /// 16 MiB at guest-physical `0x1_0000_0000`.
    fn default() -> Self {
        Self {
            base: DEFAULT_RAM_BASE,
            size: DEFAULT_RAM_SIZE as u64,
        }
    }
}

/// The VM: guest RAM, the table pool, the backend's own region, and which scratch slots are
/// taken.
struct Vm {
    ram: Extent,
    table_pool: Extent,
    system: HostRegion,
    slots: Mutex<Vec<bool>>,
    /// The guest-physical ranges mapped into the VM so far, unmapped again on drop.
    mapped: Vec<(u64, usize)>,
}

impl Drop for Vm {
    fn drop(&mut self) {
        // Every vCPU holds an `Arc<Vm>`, so all of them are destroyed by now, as
        // `hv_vm_destroy` requires. Failures can't be reported from here; the memory is freed
        // after this either way, when the regions drop.
        for &(ipa, len) in &self.mapped {
            // SAFETY: unmaps exactly a range `Vm::map` mapped.
            unsafe { ffi::hv_vm_unmap(ipa, len) };
        }
        // SAFETY: this process created the VM in `HvfBackend::new`, and nothing uses it after.
        unsafe { ffi::hv_vm_destroy() };
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Vm {
    /// Maps `len` bytes of `region` from `offset` at guest-physical `ipa` with `flags`.
    fn map(
        mapped: &mut Vec<(u64, usize)>,
        what: &str,
        region: &HostRegion,
        (offset, len): (u64, u64),
        ipa: u64,
        flags: u64,
    ) -> Result<(), Error> {
        let outside = || Error::Host(format!("hv_vm_map ({what}): range outside its region"));
        let len = usize::try_from(len).map_err(|_| outside())?;
        let host = region.host_address(offset, len).ok_or_else(outside)?;
        // SAFETY: the host range is inside a live allocation aligned to the 16 KiB host page
        // (`offset` and `len` are multiples of it), owned by the `Vm` that also owns `mapped`
        // and unmaps the range before the allocation is freed.
        ffi::check(&format!("hv_vm_map ({what})"), unsafe {
            ffi::hv_vm_map(host.cast(), ipa, len, flags)
        })?;
        mapped.push((ipa, len));
        Ok(())
    }

    /// The EL0 values of x9-x11 the shim saved for the vCPU in `slot`.
    fn scratch(&self, slot: usize) -> Result<[u64; 3], Error> {
        let mut bytes = [0; 24];
        slot_top(slot)
            .and_then(|top| top.checked_sub(shim::SCRATCH_BELOW_SP))
            .and_then(|offset| offset.checked_add(SCRATCH_OFFSET))
            .and_then(|offset| self.system.read(offset, &mut bytes))
            .ok_or_else(|| Error::Host(format!("no scratch slot {slot}")))?;
        let mut saved = [0; 3];
        let (chunks, _) = bytes.as_chunks::<8>();
        for (value, chunk) in saved.iter_mut().zip(chunks) {
            *value = u64::from_le_bytes(*chunk);
        }
        Ok(saved)
    }
}

/// Offset in the scratch page of the top of scratch slot `slot` (its `SP_EL1`, less
/// [`SCRATCH_VA`]).
fn slot_top(slot: usize) -> Option<u64> {
    let slot = u64::try_from(slot).ok()?;
    SLOT_SIZE.checked_mul(slot.checked_add(1)?)
}

/// A claimed scratch slot, released when dropped.
struct Slot {
    vm: Arc<Vm>,
    index: usize,
}

impl Slot {
    fn claim(vm: &Arc<Vm>) -> Result<Self, Error> {
        let mut slots = lock(&vm.slots);
        let index = slots
            .iter()
            .position(|taken| !taken)
            .ok_or_else(|| Error::Host(format!("at most {MAX_VCPUS} vCPUs per VM")))?;
        slots[index] = true;
        Ok(Self {
            vm: Arc::clone(vm),
            index,
        })
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(taken) = lock(&self.vm.slots).get_mut(self.index) {
            *taken = false;
        }
    }
}

/// The contents of the system region: the shim, and the static tables that map it and the
/// scratch page in the upper half. Returns `(offset, bytes)` pairs.
fn system_image() -> Result<[(u64, Vec<u8>); 2], Error> {
    let regions = [
        Region {
            va: SHIM_HALF_OFFSET,
            pa: SYSTEM_BASE + CODE_OFFSET,
            len: PAGE,
            attrs: AP_EL1_RO | UXN,
        },
        Region {
            va: SCRATCH_HALF_OFFSET,
            pa: SYSTEM_BASE + SCRATCH_OFFSET,
            len: PAGE,
            attrs: AP_EL1_RW | PXN | UXN,
        },
    ];
    let tables = tables::build(&regions, SYSTEM_BASE + SYSTEM_TABLES_OFFSET)
        .map_err(|e| Error::Host(format!("the shim's address space: {e}")))?;
    if tables.len() != SYSTEM_TABLES {
        return Err(Error::Host(format!(
            "the shim's address space needs {} tables, not {SYSTEM_TABLES}",
            tables.len()
        )));
    }
    let shim = shim::WORDS.iter().flat_map(|w| w.to_le_bytes()).collect();
    let table_bytes = tables
        .iter()
        .flatten()
        .flat_map(|e| e.to_le_bytes())
        .collect();
    Ok([(CODE_OFFSET, shim), (SYSTEM_TABLES_OFFSET, table_bytes)])
}

/// The Hypervisor.framework backend: one VM with guest RAM and a table pool.
pub struct HvfBackend {
    vm: Arc<Vm>,
    ram: BackingRange,
    table_pool: BackingRange,
}

impl HvfBackend {
    /// Creates the process's VM with `ram` as its guest RAM (see the module docs for the
    /// layout).
    ///
    /// Fails if the binary lacks the hypervisor entitlement (the workspace's cargo runner adds
    /// it), if another VM exists in this process, or on hosts without HVF.
    pub fn new(ram: RamConfig) -> Result<Self, Error> {
        let range = |base, size| {
            BackingRange::new(base, size)
                .map_err(|e| Error::Host(format!("guest-physical layout: {e}")))
        };
        let ram_range = range(ram.base, ram.size)?;
        let pool_range = range(TABLE_POOL_BASE, TABLE_POOL_SIZE)?;

        let ram_region = HostRegion::new(ram.size, Kind::Ram)?;
        let pool_region = HostRegion::new(TABLE_POOL_SIZE, Kind::Tables)?;
        let system = HostRegion::new(SYSTEM_SIZE, Kind::System)?;
        for (offset, bytes) in system_image()? {
            system
                .write(offset, &bytes)
                .ok_or_else(|| Error::Host("system region too small".into()))?;
        }

        // SAFETY: a null config asks for the default VM configuration.
        ffi::check("hv_vm_create", unsafe {
            ffi::hv_vm_create(std::ptr::null_mut())
        })?;
        // From here on, dropping `vm` destroys the VM and unmaps what was mapped.
        let mut vm = Vm {
            ram: Extent {
                base: ram.base,
                region: ram_region,
            },
            table_pool: Extent {
                base: TABLE_POOL_BASE,
                region: pool_region,
            },
            system,
            slots: Mutex::new(vec![false; MAX_VCPUS]),
            mapped: Vec::new(),
        };
        let (read, write, exec) = (ffi::MEMORY_READ, ffi::MEMORY_WRITE, ffi::MEMORY_EXEC);
        let part = |offset| (offset, HOST_PAGE);
        let mapped = &mut vm.mapped;
        Vm::map(
            mapped,
            "RAM",
            &vm.ram.region,
            (0, ram.size),
            ram.base,
            read | write | exec,
        )?;
        // The guest's MMU reads the tables; only the host writes them.
        Vm::map(
            mapped,
            "table pool",
            &vm.table_pool.region,
            (0, TABLE_POOL_SIZE),
            TABLE_POOL_BASE,
            read,
        )?;
        for (what, offset, flags) in [
            ("shim", CODE_OFFSET, read | exec),
            ("shim tables", SYSTEM_TABLES_OFFSET, read),
            ("scratch", SCRATCH_OFFSET, read | write),
        ] {
            Vm::map(
                mapped,
                what,
                &vm.system,
                part(offset),
                SYSTEM_BASE + offset,
                flags,
            )?;
        }
        Ok(Self {
            vm: Arc::new(vm),
            ram: ram_range,
            table_pool: pool_range,
        })
    }

    /// The VM's guest-physical memory: RAM and the table pool, readable and writable from any
    /// thread while vCPUs run.
    #[must_use]
    pub fn physical(&self) -> PhysicalMemory {
        PhysicalMemory {
            vm: Arc::clone(&self.vm),
        }
    }

    /// The guest-physical range of RAM: where the backing of mapped pages must lie.
    #[must_use]
    pub fn ram(&self) -> BackingRange {
        self.ram
    }

    /// The guest-physical range reserved for translation tables. A [`Stage1Tables`] whose pool
    /// is inside it (all of it, or a part per address space) can be given to
    /// [`Vcpu::set_address_space`].
    #[must_use]
    pub fn table_pool(&self) -> BackingRange {
        self.table_pool
    }

    /// The rate, in hertz, of the counter guest code reads natively (`CNTVCT_EL0`), which is also
    /// what an unpatched `mrs xN, cntfrq_el0` reads in the guest: the host's `CNTFRQ_EL0`
    /// (24 MHz on Apple Silicon, `tools/hvprobe/README.md`). Hypervisor.framework can offset the
    /// guest's counter but not scale it, so the loader's counter patcher needs this as its host
    /// rate.
    #[must_use]
    pub fn counter_frequency(&self) -> u64 {
        ffi::cntfrq_el0()
    }
}

impl Backend for HvfBackend {
    type Vcpu = HvfVcpu;

    fn create_vcpu(&self) -> Result<HvfVcpu, Error> {
        let slot = Slot::claim(&self.vm)?;
        let sp_el1 = slot_top(slot.index)
            .and_then(|top| top.checked_add(SCRATCH_VA))
            .ok_or_else(|| Error::Host("scratch slot out of range".into()))?;
        let mut id: VcpuId = 0;
        let mut exit = std::ptr::null();
        // SAFETY: both out-pointers are valid for writes; a null config means the default.
        ffi::check("hv_vcpu_create", unsafe {
            ffi::hv_vcpu_create(&raw mut id, &raw mut exit, std::ptr::null_mut())
        })?;
        // From here on, dropping `vcpu` destroys it.
        let mut vcpu = HvfVcpu {
            id,
            exit,
            interrupter: HvfInterrupter {
                target: Arc::new(Mutex::new(Some(id))),
            },
            slot,
            interrupt_pending: false,
            asid_bits: 8,
            has_address_space: false,
        };
        vcpu.asid_bits = asid_bits(vcpu.sys(ffi::ID_AA64MMFR0_EL1)?);
        for (reg, value) in [
            (ffi::MAIR_EL1, MAIR),
            (ffi::TCR_EL1, tcr(vcpu.asid_bits)),
            (ffi::TTBR1_EL1, SYSTEM_BASE + SYSTEM_TABLES_OFFSET),
            (ffi::VBAR_EL1, SHIM_VA),
            (ffi::CPACR_EL1, CPACR_FP_ON),
            (ffi::SP_EL1, sp_el1),
            (ffi::CNTKCTL_EL1, CNTKCTL_VCT_ONLY),
            (ffi::SCTLR_EL1, SCTLR),
            (ffi::SPSR_EL1, shim::SPSR_EL0T_MASKED),
        ] {
            vcpu.set_sys(reg, value)?;
        }
        // SAFETY: called on the owning thread with a live vCPU id.
        ffi::check("hv_vcpu_set_vtimer_mask", unsafe {
            ffi::hv_vcpu_set_vtimer_mask(vcpu.id, true)
        })?;
        vcpu.park()?;
        Ok(vcpu)
    }
}

/// Pointer-authentication keys, each as one 128-bit value (`KeyHi:KeyLo`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PacKeys {
    /// Instruction key A (`APIAKey`).
    pub ia: u128,
    /// Instruction key B (`APIBKey`).
    pub ib: u128,
    /// Data key A (`APDAKey`).
    pub da: u128,
    /// Data key B (`APDBKey`).
    pub db: u128,
    /// Generic key (`APGAKey`), used by `pacga`.
    pub ga: u128,
}

/// Stops a running [`HvfVcpu`] from another thread (`hv_vcpus_exit`).
#[derive(Debug, Clone)]
pub struct HvfInterrupter {
    /// The vCPU, or `None` once it's destroyed. Held locked while destroying, so an interrupt
    /// never reaches a stale id.
    target: Arc<Mutex<Option<VcpuId>>>,
}

impl Interrupter for HvfInterrupter {
    fn interrupt(&self) -> Result<(), Error> {
        let target = lock(&self.target);
        match *target {
            // SAFETY: the id is live while the lock is held (see `HvfVcpu::drop`), and
            // `hv_vcpus_exit` may be called from any thread.
            Some(id) => ffi::check("hv_vcpus_exit", unsafe {
                ffi::hv_vcpus_exit(&raw const id, 1)
            }),
            None => Ok(()),
        }
    }
}

/// One HVF vCPU, bound to the thread that created it.
pub struct HvfVcpu {
    id: VcpuId,
    /// Written by HVF during `hv_vcpu_run`; valid until the vCPU is destroyed.
    exit: *const ffi::ExitInfo,
    interrupter: HvfInterrupter,
    /// Keeps the VM alive and this vCPU's scratch slot reserved; dropped after the vCPU.
    slot: Slot,
    /// An interrupt arrived while the vCPU was running a TLB flush, which is always finished
    /// first: the next [`Vcpu::run`] reports it.
    interrupt_pending: bool,
    /// How many bits of address-space id the host gives this guest: 8 or 16.
    asid_bits: u8,
    /// Whether [`Vcpu::set_address_space`] has succeeded on this vCPU.
    has_address_space: bool,
}

impl HvfVcpu {
    /// Sets the pointer-authentication keys this vCPU signs and authenticates with. All five
    /// key types are enabled at EL0.
    pub fn set_pac_keys(&mut self, keys: &PacKeys) -> Result<(), Error> {
        for (lo, hi, key) in [
            (ffi::APIAKEYLO_EL1, ffi::APIAKEYHI_EL1, keys.ia),
            (ffi::APIBKEYLO_EL1, ffi::APIBKEYHI_EL1, keys.ib),
            (ffi::APDAKEYLO_EL1, ffi::APDAKEYHI_EL1, keys.da),
            (ffi::APDBKEYLO_EL1, ffi::APDBKEYHI_EL1, keys.db),
            (ffi::APGAKEYLO_EL1, ffi::APGAKEYHI_EL1, keys.ga),
        ] {
            let [low, high] = split_u128(key);
            self.set_sys(lo, low)?;
            self.set_sys(hi, high)?;
        }
        Ok(())
    }

    /// How many bits an address-space id may have on this host: 8 or 16
    /// (`ID_AA64MMFR0_EL1.ASIDBits` as the guest reads it). The same for every vCPU of a VM.
    #[must_use]
    pub fn asid_bits(&self) -> u8 {
        self.asid_bits
    }

    /// [`Error::AsidTooWide`] unless `asid` fits the host's id width.
    fn check_asid(&self, asid: u16) -> Result<(), Error> {
        if u32::from(asid) >> self.asid_bits == 0 {
            Ok(())
        } else {
            Err(Error::AsidTooWide {
                asid,
                bits: self.asid_bits,
            })
        }
    }

    /// Invalidates what every vCPU of the VM has cached about `ranges` of the address space
    /// with id `asid`, by running the EL1 shim's flush entry points on this vCPU. The broadcast
    /// forms are used, so the other vCPUs need not be stopped. The guest's registers are as
    /// they were when this returns, and guest code has not run.
    ///
    /// `asid` must fit the host's id width ([`Error::AsidTooWide`] otherwise), and each range
    /// must be a non-empty run of whole pages inside the 39-bit address space; otherwise this
    /// is [`Error::BadFlushRange`]. Either way nothing is invalidated. Ranges that cover
    /// more than [`MAX_PAGE_FLUSHES`] pages in total are invalidated by dropping everything
    /// cached under `asid`.
    ///
    /// An interrupt that arrives meanwhile does not cut the flush short; the next
    /// [`Vcpu::run`] returns [`Exit::Interrupted`] instead.
    ///
    /// Prefer [`Self::flush_pending`], which also keeps the tables' freed pages out of use
    /// until the flush has run.
    pub fn flush_tlb(&mut self, asid: u16, ranges: &[FlushRange]) -> Result<(), Error> {
        self.check_asid(asid)?;
        let ops = flush::plan(asid, ranges)?;
        if ops.is_empty() {
            return Ok(());
        }
        // The entry points take their operands in x0 and x1, which hold the guest's values.
        let x1 = ffi::REG_X0 + 1;
        let saved = [self.reg(ffi::REG_X0)?, self.reg(x1)?];
        let flushed = ops.iter().try_for_each(|op| self.run_flush(*op));
        // Give the guest its registers back and park again, also after a flush that failed.
        let restored = self
            .set_reg(ffi::REG_X0, saved[0])
            .and_then(|()| self.set_reg(x1, saved[1]))
            .and_then(|()| self.park());
        flushed.and(restored)
    }

    /// Executes the invalidations `tables` has recorded since its last flush
    /// ([`Self::flush_tlb`] with its pending ranges), for the address space with id `asid`
    /// that the tables belong to. Does nothing when none is pending.
    ///
    /// The tables are locked meanwhile, and only a flush that ran lets them reuse the table
    /// pages they freed ([`Stage1Tables::flush_pending_with`]). Call it after every `unmap` and
    /// `protect`, before the next change to the same tables, on any vCPU of the VM.
    pub fn flush_pending<M: GuestMemory>(
        &mut self,
        tables: &Stage1Tables<M>,
        asid: u16,
    ) -> Result<(), Error> {
        tables.flush_pending_with(|ranges| self.flush_tlb(asid, ranges))
    }

    /// Runs one flush entry point to its `hvc`.
    fn run_flush(&mut self, op: FlushOp) -> Result<(), Error> {
        let (entry, operand, count) = match op {
            FlushOp::Pages {
                operand,
                count,
                leaf_only: false,
            } => (shim::FLUSH_PAGES, operand, count),
            FlushOp::Pages {
                operand,
                count,
                leaf_only: true,
            } => (shim::FLUSH_LEAVES, operand, count),
            FlushOp::Asid { operand } => (shim::FLUSH_ASID, operand, 0),
        };
        self.set_reg(ffi::REG_X0, operand)?;
        self.set_reg(ffi::REG_X0 + 1, count)?;
        self.set_reg(ffi::REG_PC, SHIM_VA + entry as u64)?;
        self.set_reg(ffi::REG_CPSR, shim::CPSR_EL1H_MASKED)?;
        loop {
            let exit = self.enter()?;
            match exit.reason {
                // Stopped part-way (or before starting): nothing is lost, the registers hold
                // the loop's progress. Finish the flush and report the interrupt afterwards.
                ffi::EXIT_REASON_CANCELED => self.interrupt_pending = true,
                ffi::EXIT_REASON_EXCEPTION
                    if hvc_immediate(exit.exception.syndrome) == Some(shim::HVC_FLUSH_DONE) =>
                {
                    return Ok(());
                }
                ffi::EXIT_REASON_EXCEPTION => {
                    return Err(self.unexpected_exception(&exit.exception));
                }
                reason => return Err(Error::Host(format!("unexpected exit reason {reason}"))),
            }
        }
    }

    /// Runs the vCPU until it exits to the host, and returns why.
    fn enter(&self) -> Result<ffi::ExitInfo, Error> {
        // SAFETY: called on the owning thread with a live id.
        ffi::check("hv_vcpu_run", unsafe { ffi::hv_vcpu_run(self.id) })?;
        // SAFETY: HVF keeps `exit` valid until the vCPU is destroyed and fills it in before
        // `hv_vcpu_run` returns on this thread; it is copied out, not borrowed.
        Ok(unsafe { self.exit.read() })
    }

    fn reg(&self, reg: u32) -> Result<u64, Error> {
        let mut value = 0;
        // SAFETY: called on the owning thread (`HvfVcpu` is not `Send`) with a live id and a
        // valid out-pointer.
        ffi::check("hv_vcpu_get_reg", unsafe {
            ffi::hv_vcpu_get_reg(self.id, reg, &raw mut value)
        })?;
        Ok(value)
    }

    fn set_reg(&self, reg: u32, value: u64) -> Result<(), Error> {
        // SAFETY: called on the owning thread with a live id.
        ffi::check("hv_vcpu_set_reg", unsafe {
            ffi::hv_vcpu_set_reg(self.id, reg, value)
        })
    }

    fn sys(&self, reg: u16) -> Result<u64, Error> {
        let mut value = 0;
        // SAFETY: called on the owning thread with a live id and a valid out-pointer.
        ffi::check("hv_vcpu_get_sys_reg", unsafe {
            ffi::hv_vcpu_get_sys_reg(self.id, reg, &raw mut value)
        })?;
        Ok(value)
    }

    fn set_sys(&self, reg: u16, value: u64) -> Result<(), Error> {
        // SAFETY: called on the owning thread with a live id.
        ffi::check("hv_vcpu_set_sys_reg", unsafe {
            ffi::hv_vcpu_set_sys_reg(self.id, reg, value)
        })
    }

    /// Parks the vCPU on the shim's `eret`, so the next run returns to EL0 at `ELR_EL1`.
    fn park(&self) -> Result<(), Error> {
        self.set_reg(ffi::REG_PC, SHIM_VA + shim::PARKED as u64)?;
        self.set_reg(ffi::REG_CPSR, shim::CPSR_EL1H_MASKED)
    }

    /// The error for an exception exit that is not one of the shim's `hvc`s.
    fn unexpected_exception(&self, exception: &ffi::ExitException) -> Error {
        let syndrome = exception.syndrome;
        let pc = self.reg(ffi::REG_PC).unwrap_or_default();
        match Syndrome::from_esr(syndrome).class() {
            // An abort that reaches the host is a stage-2 fault: stage-1 faults go to the shim.
            ExceptionClass::DataAbort | ExceptionClass::InstructionAbort => Error::Host(format!(
                "the guest reached guest-physical address {:#x} in a way stage 2 does not \
                     allow (ESR_EL2 {syndrome:#x}, pc {pc:#x})",
                exception.physical_address
            )),
            _ => Error::Host(format!(
                "unexpected exception exit: ESR_EL2 {syndrome:#x} at pc {pc:#x}"
            )),
        }
    }

    /// Handles an exception exit to the host. The only one expected is the shim's `hvc` for an
    /// EL0 exception.
    fn exception(&self, exception: &ffi::ExitException) -> Result<Exit, Error> {
        let esr = self.sys(ffi::ESR_EL1)?;
        let elr = self.sys(ffi::ELR_EL1)?;
        match hvc_immediate(exception.syndrome) {
            Some(shim::HVC_EL0_EXCEPTION) => {}
            Some(_) => {
                return Err(Error::Host(format!(
                    "the EL1 shim took an unexpected exception: ESR_EL1 {esr:#x}, ELR_EL1 {elr:#x}"
                )));
            }
            None => return Err(self.unexpected_exception(exception)),
        }
        let far = self.sys(ffi::FAR_EL1)?;
        self.park()?;
        Ok(shim::decode_el0_exception(esr, far, elr))
    }

    /// Brings an interrupted vCPU back to a parked state with exact EL0 registers.
    fn interrupted(&self) -> Result<Exit, Error> {
        let cpsr = self.reg(ffi::REG_CPSR)?;
        let pc = self.reg(ffi::REG_PC)?;
        match cpsr & shim::MODE_MASK {
            shim::MODE_EL0T => {
                // Stopped in guest code: move its state to where a parked vCPU keeps it.
                self.set_sys(ffi::ELR_EL1, pc)?;
                self.set_sys(ffi::SPSR_EL1, cpsr)?;
            }
            shim::MODE_EL1H => {
                let unexpected =
                    || Error::Host(format!("interrupted at unexpected EL1 pc {pc:#x}"));
                let offset = pc
                    .checked_sub(SHIM_VA)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .ok_or_else(unexpected)?;
                let recovery =
                    shim::recovery(offset, self.sys(ffi::ESR_EL1)?).ok_or_else(unexpected)?;
                if recovery.restore_scratch {
                    let saved = self.slot.vm.scratch(self.slot.index)?;
                    for (reg, value) in (ffi::REG_X0 + 9..).zip(saved) {
                        self.set_reg(reg, value)?;
                    }
                }
                if recovery.rewind_elr {
                    let elr = self.sys(ffi::ELR_EL1)?;
                    self.set_sys(ffi::ELR_EL1, elr.wrapping_sub(4))?;
                }
            }
            mode => {
                return Err(Error::Host(format!(
                    "interrupted in unexpected mode {mode:#x} at pc {pc:#x}"
                )));
            }
        }
        self.park()?;
        Ok(Exit::Interrupted)
    }
}

/// The immediate of the `hvc` that `ESR_EL2` value `syndrome` describes; `None` if it describes
/// anything else.
fn hvc_immediate(syndrome: u64) -> Option<u16> {
    (Syndrome::from_esr(syndrome).class().ec() == shim::EC_HVC64).then(|| {
        let [lo, hi, ..] = syndrome.to_le_bytes();
        u16::from_le_bytes([lo, hi])
    })
}

/// `TTBR0_EL1` for the tables at `root` and the address-space id `asid`: the table's address in
/// `BADDR` and the id from bit 48 up (Arm ARM, `TTBR0_EL1`).
fn ttbr0(root: TableRoot, asid: u16) -> u64 {
    root.pa() | (u64::from(asid) << flush::ASID_SHIFT)
}

/// `[low 64 bits, high 64 bits]`.
fn split_u128(value: u128) -> [u64; 2] {
    let bytes = value.to_le_bytes();
    let mut lo = [0; 8];
    let mut hi = [0; 8];
    lo.copy_from_slice(&bytes[..8]);
    hi.copy_from_slice(&bytes[8..]);
    [u64::from_le_bytes(lo), u64::from_le_bytes(hi)]
}

impl Vcpu for HvfVcpu {
    type Interrupter = HvfInterrupter;

    fn run(&mut self) -> Result<Exit, Error> {
        if !self.has_address_space {
            return Err(Error::NoAddressSpace);
        }
        if std::mem::take(&mut self.interrupt_pending) {
            // The interrupt was used up by a flush: report it now, before any guest code.
            return Ok(Exit::Interrupted);
        }
        let exit = self.enter()?;
        match exit.reason {
            ffi::EXIT_REASON_EXCEPTION => self.exception(&exit.exception),
            ffi::EXIT_REASON_CANCELED => self.interrupted(),
            // The vtimer is masked at creation, so this and unknown reasons are both unexpected.
            reason => Err(Error::Host(format!("unexpected exit reason {reason}"))),
        }
    }

    fn interrupter(&self) -> HvfInterrupter {
        self.interrupter.clone()
    }

    fn regs(&self) -> Result<GuestRegs, Error> {
        let mut regs = GuestRegs::default();
        for (reg, x) in (ffi::REG_X0..).zip(regs.x.iter_mut()) {
            *x = self.reg(reg)?;
        }
        regs.sp = self.sys(ffi::SP_EL0)?;
        regs.pc = self.sys(ffi::ELR_EL1)?;
        regs.pstate = self.sys(ffi::SPSR_EL1)? & PSTATE_EL0_MASK;
        regs.tpidr = self.sys(ffi::TPIDR_EL0)?;
        regs.tpidrro = self.sys(ffi::TPIDRRO_EL0)?;
        Ok(regs)
    }

    fn set_regs(&mut self, regs: &GuestRegs) -> Result<(), Error> {
        for (reg, &x) in (ffi::REG_X0..).zip(regs.x.iter()) {
            self.set_reg(reg, x)?;
        }
        self.set_sys(ffi::SP_EL0, regs.sp)?;
        self.set_sys(ffi::ELR_EL1, regs.pc)?;
        // Only the flags come from the caller: the guest always resumes at EL0t, interrupts
        // masked.
        self.set_sys(
            ffi::SPSR_EL1,
            (regs.pstate & PSTATE_EL0_MASK) | shim::SPSR_EL0T_MASKED,
        )?;
        self.set_sys(ffi::TPIDR_EL0, regs.tpidr)?;
        self.set_sys(ffi::TPIDRRO_EL0, regs.tpidrro)
    }

    /// The root's table page must lie inside [`HvfBackend::table_pool`]
    /// ([`Error::BadTableRoot`] otherwise) and `asid` must fit [`HvfVcpu::asid_bits`]
    /// ([`Error::AsidTooWide`]). After an error the vCPU keeps the address space it had.
    fn set_address_space(&mut self, root: TableRoot, asid: u16) -> Result<(), Error> {
        self.check_asid(asid)?;
        let pool = &self.slot.vm.table_pool;
        let inside = root.pa() >= pool.base
            && root
                .pa()
                .checked_add(PAGE)
                .is_some_and(|end| end <= pool.end());
        if !inside {
            return Err(Error::BadTableRoot { root: root.pa() });
        }
        self.set_sys(ffi::TTBR0_EL1, ttbr0(root, asid))?;
        self.has_address_space = true;
        Ok(())
    }
}

impl Drop for HvfVcpu {
    fn drop(&mut self) {
        // Hold the interrupter's lock across the destroy, so no `hv_vcpus_exit` sees a stale id.
        let mut target = lock(&self.interrupter.target);
        *target = None;
        // SAFETY: `HvfVcpu` is not `Send`, so this is the owning thread, and the id is live.
        // A failure can't be reported from `drop`.
        unsafe { ffi::hv_vcpu_destroy(self.id) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_config_checks_alignment_size_and_range() {
        let page = usize::try_from(HOST_PAGE).unwrap();
        assert_eq!(
            RamConfig::new(DEFAULT_RAM_BASE, DEFAULT_RAM_SIZE),
            Ok(RamConfig::default())
        );
        assert_eq!(RamConfig::new(DEFAULT_RAM_BASE, 0), Err(LayoutError::Empty));
        assert_eq!(
            RamConfig::new(DEFAULT_RAM_BASE + 0x1000, page),
            Err(LayoutError::Misaligned {
                value: DEFAULT_RAM_BASE + 0x1000,
                alignment: HOST_PAGE
            })
        );
        assert_eq!(
            RamConfig::new(DEFAULT_RAM_BASE, page + 0x1000),
            Err(LayoutError::Misaligned {
                value: HOST_PAGE + 0x1000,
                alignment: HOST_PAGE
            })
        );
        assert_eq!(
            RamConfig::new(SYSTEM_BASE - HOST_PAGE, page),
            Ok(RamConfig {
                base: SYSTEM_BASE - HOST_PAGE,
                size: HOST_PAGE
            })
        );
        assert_eq!(
            RamConfig::new(SYSTEM_BASE, page),
            Err(LayoutError::OutOfRange)
        );
        assert_eq!(
            RamConfig::new(u64::MAX - (HOST_PAGE - 1), page),
            Err(LayoutError::OutOfRange)
        );
    }

    #[test]
    fn scratch_slots_stay_inside_the_scratch_page() {
        assert_eq!(slot_top(0), Some(SLOT_SIZE));
        assert_eq!(slot_top(MAX_VCPUS - 1), Some(PAGE));
        assert_eq!(slot_top(usize::MAX), None);
    }

    #[test]
    fn splits_keys_into_lo_and_hi() {
        assert_eq!(
            split_u128(0x0011_2233_4455_6677_8899_aabb_ccdd_eeff),
            [0x8899_aabb_ccdd_eeff, 0x0011_2233_4455_6677]
        );
    }

    #[test]
    fn the_translation_control_register_has_the_documented_fields() {
        // Each field read back from its Arm ARM bit position.
        let field = |low: u32, bits: u32| (TCR >> low) & ((1 << bits) - 1);
        assert_eq!(field(0, 6), 25, "T0SZ: a 39-bit lower half");
        assert_eq!(field(7, 1), 0, "EPD0: lower-half walks enabled");
        assert_eq!(field(14, 2), 0b00, "TG0: 4 KiB");
        assert_eq!(field(16, 6), 25, "T1SZ: a 39-bit upper half");
        assert_eq!(field(22, 1), 0, "A1: the id comes from TTBR0_EL1");
        assert_eq!(field(23, 1), 0, "EPD1: upper-half walks enabled");
        assert_eq!(field(30, 2), 0b10, "TG1: 4 KiB");
        assert_eq!(field(32, 3), 0b010, "IPS: 40 bits");
        assert_eq!(field(36, 1), 0, "AS: set only where ids are 16 bits wide");
        // The two halves walk the same kind of memory.
        assert_eq!(field(8, 6), field(24, 6), "IRGN, ORGN, SH");
        assert_eq!(TCR, 0x0000_0002_b519_3519);
        assert_eq!(tcr(8), TCR);
        assert_eq!(tcr(16), 0x0000_0012_b519_3519);
        // WXN (bit 19) is clear, as the software walker assumes.
        assert_eq!(SCTLR & (1 << 19), 0);
    }

    #[test]
    fn the_shim_is_in_the_upper_half_and_guest_tables_cannot_name_it() {
        assert_eq!(UPPER_HALF, 0xffff_ff80_0000_0000);
        assert_eq!(SHIM_VA, 0xffff_ffff_ffe0_0000);
        assert_eq!(SCRATCH_VA, 0xffff_ffff_ffe1_0000);
        // Guest tables translate addresses below `VA_SPACE_END` only.
        const { assert!(SHIM_VA >= starquake_memory::stage1::VA_SPACE_END) };
        // The last scratch slot's top is still inside the scratch page's mapping.
        assert_eq!(
            slot_top(MAX_VCPUS - 1).map(|top| SCRATCH_VA + top),
            Some(SCRATCH_VA + PAGE)
        );
    }

    #[test]
    fn the_system_region_holds_the_shim_and_three_tables_that_map_it() {
        let [(shim_at, shim), (tables_at, tables)] = system_image().unwrap();
        assert_eq!((shim_at, shim.len()), (CODE_OFFSET, shim::LEN));
        assert_eq!(tables_at, SYSTEM_TABLES_OFFSET);
        assert_eq!(tables.len() as u64, SYSTEM_TABLES as u64 * PAGE);
        // Each fits its part of the region.
        assert!(shim.len() as u64 <= HOST_PAGE);
        assert!(tables.len() as u64 <= HOST_PAGE);

        // Walk the bytes as the MMU would, for an upper-half address.
        let entry = |table: u64, index: u64| {
            let at =
                usize::try_from(table - SYSTEM_BASE - SYSTEM_TABLES_OFFSET + index * 8).unwrap();
            u64::from_le_bytes(tables[at..at + 8].try_into().unwrap())
        };
        let walk = |va: u64| {
            assert_eq!(va >> 39, u64::MAX >> 39, "{va:#x} is in the upper half");
            let mut table = SYSTEM_BASE + SYSTEM_TABLES_OFFSET;
            for shift in [30, 21] {
                let descriptor = entry(table, (va >> shift) & 0x1ff);
                if descriptor & 0b11 != 0b11 {
                    return None;
                }
                table = descriptor & 0x0000_ffff_ffff_f000;
            }
            let leaf = entry(table, (va >> 12) & 0x1ff);
            (leaf & 0b11 == 0b11).then_some(leaf)
        };
        // The shim: read-only at EL1 (AP 0b10), no EL0 access, executable at EL1 only (UXN),
        // global (nG clear), at the shim's physical page.
        let shim_leaf = walk(SHIM_VA).unwrap();
        assert_eq!(shim_leaf, 0x0040_000f_0000_0783);
        // The scratch page: read-write at EL1 (AP 0b00), never executable (PXN and UXN).
        let scratch_leaf = walk(SCRATCH_VA).unwrap();
        assert_eq!(scratch_leaf, 0x0060_000f_0000_8703);
        // Nothing else is mapped, in particular not the pages next to either.
        for unmapped in [
            SHIM_VA - PAGE,
            SHIM_VA + PAGE,
            SCRATCH_VA - PAGE,
            SCRATCH_VA + PAGE,
            UPPER_HALF,
            !(PAGE - 1),
        ] {
            assert_eq!(walk(unmapped), None, "{unmapped:#x}");
        }
    }

    #[test]
    fn the_id_register_decides_the_asid_width() {
        // What this backend read on an Apple M2 under Hypervisor.framework: ASIDBits (bits 7:4)
        // is 0, so ids are 8 bits wide.
        assert_eq!(asid_bits(0x1000_0f10_0001), 8);
        // The same value with ASIDBits = 0b0010, and with the two encodings next to it.
        assert_eq!(asid_bits(0x1000_0f10_0021), 16);
        assert_eq!(asid_bits(0x1000_0f10_0011), 8);
        assert_eq!(asid_bits(0x1000_0f10_0031), 8);
        // The other fields don't matter.
        assert_eq!(asid_bits(!0xf0), 8);
        assert_eq!(asid_bits(0x20), 16);
    }

    #[test]
    fn ttbr0_holds_the_root_and_the_asid() {
        let root = TableRoot::new(TABLE_POOL_BASE + 0x3000).unwrap();
        assert_eq!(ttbr0(root, 0), 0x0000_000f_1000_3000);
        assert_eq!(ttbr0(root, 0xbeef), 0xbeef_000f_1000_3000);
    }

    #[test]
    fn only_an_hvc_has_an_hvc_immediate() {
        // ESR_EL2 of `hvc #2` and `hvc #3`: EC 0x16, IL, the immediate in the low bits.
        assert_eq!(hvc_immediate(0x5a00_0002), Some(shim::HVC_EL0_EXCEPTION));
        assert_eq!(hvc_immediate(0x5a00_0003), Some(shim::HVC_FLUSH_DONE));
        // A stage-2 data abort and an `svc` are not.
        assert_eq!(hvc_immediate(0x9200_004f), None);
        assert_eq!(hvc_immediate(0x5600_0002), None);
    }
}
