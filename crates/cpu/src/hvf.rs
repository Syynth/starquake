//! Hypervisor.framework backend (macOS on Apple Silicon).
//!
//! Guest code runs natively at EL0 inside an HVF VM. A small EL1 shim (`shim`) catches every EL0
//! exception: `svc`, aborts and undefined instructions come back to the host as an [`Exit`], and
//! EL0 reads of `cntpct_el0` are answered in the guest from the virtual counter, without a host
//! exit. `cntvct_el0` reads, FP/SIMD and pointer authentication run natively.
//!
//! # Bring-up memory (provisional)
//!
//! The address space here exists so a vCPU can be tested; the real address-space design is a
//! separate spec. [`HvfBackend`] maps one contiguous region of guest RAM ([`RamConfig`]),
//! identity-mapped (virtual address == IPA) with 4 KiB stage-1 pages that EL0 can read, write and
//! execute, and exposes it through [`GuestMemory`]. The backend's own pages (shim, per-vCPU
//! scratch, translation tables) sit in a second region at a fixed IPA above guest RAM, which EL0
//! cannot touch. Every other address is unmapped and faults.
//!
//! # Threads
//!
//! Hypervisor.framework allows one VM per process, so only one [`HvfBackend`] can exist at a
//! time. A vCPU belongs to the thread that created it ([`HvfVcpu`] is not `Send`); other threads
//! stop it with an [`HvfInterrupter`].
//!
//! Ported from `tools/hvprobe` (`src/hv.rs`, `src/guest.rs`, `src/probe.rs`); measured behaviour
//! is in `tools/hvprobe/README.md`.

mod ffi;
mod shim;
mod tables;

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_memory::{Fault, GuestMemory};

use crate::{Backend, Error, Exit, GuestRegs, Interrupter, Vcpu};
use ffi::VcpuId;
use tables::{AP_ALL_RW, AP_EL1_RO, AP_EL1_RW, PAGE, PXN, Region, UXN};

/// Default guest RAM base (`GUEST_BASE` in `tools/hvprobe/src/probe.rs`).
const DEFAULT_RAM_BASE: u64 = 0x8000_0000;
/// Default guest RAM size (`GUEST_SIZE` in `tools/hvprobe/src/probe.rs`).
const DEFAULT_RAM_SIZE: usize = 16 << 20;

/// Stage-2 (IPA → host) mappings must be aligned to the 16 KiB host page
/// (`tools/hvprobe/README.md`, "4 KiB stage-1 pages on a 16 KiB host").
const HOST_PAGE: u64 = 0x4000;

/// IPA (and virtual address) of the backend's own region. It sits at the top of the 36-bit IPA
/// space HVF provides by default (`tools/hvprobe/README.md`, "Guest IPA space"). Provisional.
const SYSTEM_BASE: u64 = 0xf_0000_0000;
/// Offset of the shim in the system region (`VBAR_EL1` needs 2 KiB alignment).
const SHIM_OFFSET: u64 = 0;
/// Offset of the per-vCPU scratch slots: each slot's top is that vCPU's `SP_EL1`.
const SCRATCH_OFFSET: u64 = 0x1000;
/// Bytes per scratch slot (16-byte aligned, as `SP_EL1` must be).
const SLOT_SIZE: u64 = 64;
/// vCPUs one VM supports: one scratch page's worth of slots.
const MAX_VCPUS: usize = 64;
/// Offset of the translation tables, after the scratch page.
const TABLES_OFFSET: u64 = 0x2000;

const _: () = assert!(shim::LEN as u64 <= SCRATCH_OFFSET);
const _: () = assert!(SCRATCH_OFFSET + MAX_VCPUS as u64 * SLOT_SIZE <= TABLES_OFFSET);
const _: () = assert!(SLOT_SIZE >= shim::SCRATCH_BELOW_SP && SLOT_SIZE.is_multiple_of(16));

// EL1 configuration, from `Vcpu::new` in `tools/hvprobe/src/probe.rs` unless noted. Bit positions
// are from the Arm Architecture Reference Manual (DDI 0487).
/// `MAIR_EL1`: attribute 0 is normal memory, write-back cacheable.
const MAIR: u64 = 0xff;
/// `TCR_EL1`: T0SZ = 25 (39-bit VA), inner/outer write-back, inner shareable, 4 KiB granule,
/// TTBR1 walks disabled (EPD1), 40-bit IPS.
const TCR: u64 = 0x19 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2 << 32);
/// `CPACR_EL1.FPEN` = 0b11: FP/SIMD doesn't trap at EL0 or EL1.
const CPACR_FP_ON: u64 = 3 << 20;
/// `CNTKCTL_EL1`: EL0VCTEN only, so EL0 reads of `cntpct_el0` trap to the shim
/// (`CNTKCTL_EL0_VCT_ONLY` in the probe).
const CNTKCTL_VCT_ONLY: u64 = 0b10;
/// `SCTLR_EL1`: RES1 bits, MMU and caches on, WFI/WFE not trapped, the four PAuth enables
/// (`EnIA`, `EnIB` from the probe; `EnDA`, `EnDB` added), and EL0 access to `CTR_EL0` (`UCT`),
/// `DC ZVA` (`DZE`) and cache maintenance by address (`UCI`), which ordinary user-space code
/// relies on.
const SCTLR: u64 = 0x30d0_0800 // RES1
    | 1 | (1 << 2) | (1 << 12) // M, C, I
    | (1 << 16) | (1 << 18) // nTWI, nTWE
    | (1 << 31) | (1 << 30) // EnIA, EnIB
    | (1 << 27) | (1 << 13) // EnDA, EnDB
    | (1 << 15) | (1 << 14) | (1 << 26); // UCT, DZE, UCI

/// `PSTATE` bits the host can see and set: NZCV. The mode and interrupt masks are the backend's.
const PSTATE_EL0_MASK: u64 = 0xf000_0000;

/// Why a [`RamConfig`] is invalid, or the bring-up address space can't be built.
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

/// Where guest RAM goes: `size` bytes at guest address `base` (identity-mapped).
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
    /// 16 MiB at `0x8000_0000`, as in `tools/hvprobe`.
    fn default() -> Self {
        Self {
            base: DEFAULT_RAM_BASE,
            size: DEFAULT_RAM_SIZE as u64,
        }
    }
}

/// Host memory with the alignment HVF needs, zeroed, freed on drop.
struct HostRegion {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: HostRegion owns its allocation, and every access goes through raw-pointer copies with
// bounds checks; no references into it are ever handed out, so sharing it between threads is
// sound as far as Rust is concerned. (Guest vCPUs write it concurrently, as real RAM would be.)
unsafe impl Send for HostRegion {}
// SAFETY: as above.
unsafe impl Sync for HostRegion {}

impl HostRegion {
    fn new(len: u64) -> Result<Self, Error> {
        let too_big = |_| Error::Host(format!("cannot allocate {len:#x} bytes of guest memory"));
        let size = usize::try_from(len).map_err(too_big)?;
        let align = usize::try_from(HOST_PAGE).map_err(too_big)?;
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| Error::Host(format!("bad guest memory layout: {len:#x} bytes")))?;
        if size == 0 {
            return Err(Error::Host("guest memory region is empty".into()));
        }
        // SAFETY: the layout has a non-zero size (checked above).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::Host(format!("out of memory allocating {len:#x} bytes")))?;
        Ok(Self { ptr, layout })
    }

    fn len(&self) -> usize {
        self.layout.size()
    }

    /// The byte range `offset..offset + len`, if it is inside the region.
    fn range(&self, offset: u64, len: usize) -> Option<usize> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= self.len()).then_some(start)
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Option<()> {
        let start = self.range(offset, buf.len())?;
        // SAFETY: `start..start + buf.len()` is inside the allocation (checked by `range`), and
        // `buf` is a separate Rust allocation, so the two don't overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(start),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        Some(())
    }

    fn write(&self, offset: u64, data: &[u8]) -> Option<()> {
        let start = self.range(offset, data.len())?;
        // SAFETY: as in `read`, with the copy going the other way. The region is only ever
        // accessed through raw pointers, so writing through `&self` aliases no Rust reference.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(start), data.len());
        }
        // The guest may execute what was just written, and Hypervisor.framework does not keep its
        // instruction fetches coherent with host stores (see "Instruction cache" in
        // `tools/hvprobe/README.md`).
        self.sync_icache(start, data.len());
        Some(())
    }

    /// Makes guest instruction fetches see everything written to `start..start + len` so far.
    /// `start..start + len` must be inside the region (callers check it with `range`).
    fn sync_icache(&self, start: usize, len: usize) {
        debug_assert!(start.checked_add(len).is_some_and(|end| end <= self.len()));
        // SAFETY: the range is inside the allocation, which stays live for the call, and the
        // function only does cache maintenance by address: it neither reads nor writes through
        // the pointer in a way Rust can observe.
        unsafe { ffi::sys_icache_invalidate(self.ptr.as_ptr().add(start).cast(), len) };
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with this same layout and is freed only here.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// The bring-up guest RAM: one contiguous, identity-mapped region (see the module docs).
pub struct BringupRam {
    base: u64,
    region: HostRegion,
}

impl BringupRam {
    /// Guest address of the first byte.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.region.len()
    }

    /// The fault for an access at `va` that didn't fit: its first address outside RAM.
    fn fault(&self, va: u64) -> Fault {
        let end = self.base.saturating_add(self.region.len() as u64);
        Fault {
            address: if va < self.base { va } else { va.max(end) },
        }
    }
}

impl GuestMemory for BringupRam {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        va.checked_sub(self.base)
            .and_then(|offset| self.region.read(offset, buf))
            .ok_or_else(|| self.fault(va))
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        va.checked_sub(self.base)
            .and_then(|offset| self.region.write(offset, data))
            .ok_or_else(|| self.fault(va))
    }
}

/// The VM: guest RAM, the backend's own region, and which scratch slots are taken.
struct Vm {
    ram: BringupRam,
    system: HostRegion,
    slots: Mutex<Vec<bool>>,
    ram_mapped: bool,
    system_mapped: bool,
}

impl Drop for Vm {
    fn drop(&mut self) {
        // Every vCPU holds an `Arc<Vm>`, so all of them are destroyed by now, as
        // `hv_vm_destroy` requires. Failures can't be reported from here; the memory is freed
        // after this either way, when the regions drop.
        if self.ram_mapped {
            // SAFETY: unmaps exactly the IPA range mapped in `HvfBackend::new`.
            unsafe { ffi::hv_vm_unmap(self.ram.base, self.ram.region.len()) };
        }
        if self.system_mapped {
            // SAFETY: as above, for the system region.
            unsafe { ffi::hv_vm_unmap(SYSTEM_BASE, self.system.len()) };
        }
        // SAFETY: this process created the VM in `HvfBackend::new`, and nothing uses it after.
        unsafe { ffi::hv_vm_destroy() };
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Vm {
    /// The EL0 values of x9-x11 the shim saved for the vCPU in `slot`.
    fn scratch(&self, slot: usize) -> Result<[u64; 3], Error> {
        let mut bytes = [0; 24];
        slot_top(slot)
            .and_then(|top| top.checked_sub(shim::SCRATCH_BELOW_SP))
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

/// Offset in the system region of the top of scratch slot `slot` (its `SP_EL1`).
fn slot_top(slot: usize) -> Option<u64> {
    let slot = u64::try_from(slot).ok()?;
    SLOT_SIZE
        .checked_mul(slot.checked_add(1)?)?
        .checked_add(SCRATCH_OFFSET)
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

/// The Hypervisor.framework backend: one VM with bring-up guest RAM.
pub struct HvfBackend {
    vm: Arc<Vm>,
}

impl HvfBackend {
    /// Creates the process's VM with `ram` mapped (see the module docs for the layout).
    ///
    /// Fails if the binary lacks the hypervisor entitlement (the workspace's cargo runner adds
    /// it), if another VM exists in this process, or on hosts without HVF.
    pub fn new(ram: RamConfig) -> Result<Self, Error> {
        let shim_va = SYSTEM_BASE + SHIM_OFFSET;
        let tables_pa = SYSTEM_BASE + TABLES_OFFSET;
        let regions = [
            Region {
                va: ram.base,
                pa: ram.base,
                len: ram.size,
                attrs: AP_ALL_RW | PXN,
            },
            Region {
                va: shim_va,
                pa: shim_va,
                len: PAGE,
                attrs: AP_EL1_RO | UXN,
            },
            Region {
                va: SYSTEM_BASE + SCRATCH_OFFSET,
                pa: SYSTEM_BASE + SCRATCH_OFFSET,
                len: PAGE,
                attrs: AP_EL1_RW | PXN | UXN,
            },
        ];
        let layout_error = |e: LayoutError| Error::Host(format!("bring-up address space: {e}"));
        let tables = tables::build(&regions, tables_pa).map_err(layout_error)?;
        let tables_len = u64::try_from(tables.len())
            .ok()
            .and_then(|n| n.checked_mul(PAGE))
            .ok_or(LayoutError::OutOfRange)
            .map_err(layout_error)?;
        let system_len = (TABLES_OFFSET + tables_len).next_multiple_of(HOST_PAGE);

        let ram_region = HostRegion::new(ram.size)?;
        let system = HostRegion::new(system_len)?;
        let shim: Vec<u8> = shim::WORDS.iter().flat_map(|w| w.to_le_bytes()).collect();
        let table_bytes: Vec<u8> = tables
            .iter()
            .flatten()
            .flat_map(|e| e.to_le_bytes())
            .collect();
        system
            .write(SHIM_OFFSET, &shim)
            .and_then(|()| system.write(TABLES_OFFSET, &table_bytes))
            .ok_or_else(|| Error::Host("system region too small".into()))?;

        // SAFETY: a null config asks for the default VM configuration.
        ffi::check("hv_vm_create", unsafe {
            ffi::hv_vm_create(std::ptr::null_mut())
        })?;
        // From here on, dropping `vm` destroys the VM and unmaps what was mapped.
        let mut vm = Vm {
            ram: BringupRam {
                base: ram.base,
                region: ram_region,
            },
            system,
            slots: Mutex::new(vec![false; MAX_VCPUS]),
            ram_mapped: false,
            system_mapped: false,
        };
        let rwx = ffi::MEMORY_READ | ffi::MEMORY_WRITE | ffi::MEMORY_EXEC;
        // SAFETY: the host range is a live, 16 KiB-aligned allocation of exactly this size,
        // owned by `vm`, which unmaps it before freeing it.
        ffi::check("hv_vm_map (RAM)", unsafe {
            ffi::hv_vm_map(
                vm.ram.region.ptr.as_ptr().cast(),
                ram.base,
                vm.ram.region.len(),
                rwx,
            )
        })?;
        vm.ram_mapped = true;
        // SAFETY: as above, for the system region.
        ffi::check("hv_vm_map (system)", unsafe {
            ffi::hv_vm_map(
                vm.system.ptr.as_ptr().cast(),
                SYSTEM_BASE,
                vm.system.len(),
                rwx,
            )
        })?;
        vm.system_mapped = true;
        Ok(Self { vm: Arc::new(vm) })
    }

    /// The guest RAM, readable and writable from any thread while vCPUs run.
    pub fn memory(&self) -> &BringupRam {
        &self.vm.ram
    }
}

impl Backend for HvfBackend {
    type Vcpu = HvfVcpu;

    fn create_vcpu(&self) -> Result<HvfVcpu, Error> {
        let slot = Slot::claim(&self.vm)?;
        let sp_el1 = slot_top(slot.index)
            .and_then(|top| top.checked_add(SYSTEM_BASE))
            .ok_or_else(|| Error::Host("scratch slot out of range".into()))?;
        let mut id: VcpuId = 0;
        let mut exit = std::ptr::null();
        // SAFETY: both out-pointers are valid for writes; a null config means the default.
        ffi::check("hv_vcpu_create", unsafe {
            ffi::hv_vcpu_create(&raw mut id, &raw mut exit, std::ptr::null_mut())
        })?;
        // From here on, dropping `vcpu` destroys it.
        let vcpu = HvfVcpu {
            id,
            exit,
            interrupter: HvfInterrupter {
                target: Arc::new(Mutex::new(Some(id))),
            },
            slot,
        };
        for (reg, value) in [
            (ffi::MAIR_EL1, MAIR),
            (ffi::TCR_EL1, TCR),
            (ffi::TTBR0_EL1, SYSTEM_BASE + TABLES_OFFSET),
            (ffi::VBAR_EL1, SYSTEM_BASE + SHIM_OFFSET),
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
        self.set_reg(ffi::REG_PC, SYSTEM_BASE + SHIM_OFFSET + shim::PARKED as u64)?;
        self.set_reg(ffi::REG_CPSR, shim::CPSR_EL1H_MASKED)
    }

    /// Handles an exception exit to the host. The only one expected is the shim's `hvc`.
    fn exception(&self, syndrome: u64) -> Result<Exit, Error> {
        let esr = self.sys(ffi::ESR_EL1)?;
        let elr = self.sys(ffi::ELR_EL1)?;
        if shim::exception_class(syndrome) != shim::EC_HVC64 {
            let pc = self.reg(ffi::REG_PC)?;
            return Err(Error::Host(format!(
                "unexpected exception exit: ESR_EL2 {syndrome:#x} at pc {pc:#x}"
            )));
        }
        if syndrome & 0xffff != u64::from(shim::HVC_EL0_EXCEPTION) {
            return Err(Error::Host(format!(
                "the EL1 shim took an unexpected exception: ESR_EL1 {esr:#x}, ELR_EL1 {elr:#x}"
            )));
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
                    .checked_sub(SYSTEM_BASE + SHIM_OFFSET)
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
        // SAFETY: called on the owning thread with a live id.
        ffi::check("hv_vcpu_run", unsafe { ffi::hv_vcpu_run(self.id) })?;
        // SAFETY: HVF keeps `exit` valid until the vCPU is destroyed and fills it in before
        // `hv_vcpu_run` returns on this thread; it is copied out, not borrowed.
        let exit = unsafe { self.exit.read() };
        match exit.reason {
            ffi::EXIT_REASON_EXCEPTION => self.exception(exit.exception.syndrome),
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
        assert_eq!(slot_top(0), Some(SCRATCH_OFFSET + SLOT_SIZE));
        assert_eq!(slot_top(MAX_VCPUS - 1), Some(SCRATCH_OFFSET + PAGE));
        assert_eq!(slot_top(usize::MAX), None);
    }

    #[test]
    fn splits_keys_into_lo_and_hi() {
        assert_eq!(
            split_u128(0x0011_2233_4455_6677_8899_aabb_ccdd_eeff),
            [0x8899_aabb_ccdd_eeff, 0x0011_2233_4455_6677]
        );
    }
}
