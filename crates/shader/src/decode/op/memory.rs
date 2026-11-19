//! Slice 6e, memory: loads and stores in every space (`LDG`, `LDL`, `LDS`, `STG`, `STL`,
//! `STS`), constant-buffer loads (`LDC`), atomics (`ATOMG`, `ATOMS`, `RED`, with their
//! compare-and-swap forms, and `ATOMS.POPC`), cache control (`CCTL`), barriers (`MEMBAR`,
//! `ERRBAR`, `LDGDEPBAR`) and the global-to-shared copy `LDGSTS`. Every instruction of the group
//! is decoded; none is left a shell.
//!
//! # Sources
//!
//! Shapes follow NAK's `OpLd`, `OpSt`, `OpLdc`, `OpAtom`, `OpCCtl` and `OpMemBar`, and their
//! field types (`MemType`, `MemOrder`, `MemScope`, `MemEvictionPriority`, `OffsetStride`,
//! `AtomOp`, `AtomType`, `CCtlOp`, `LdcMode`); encodings are the inverse of NAK's `impl SM70Op`
//! blocks and `SM70Encoder` helpers (`set_mem_type`, `set_mem_order`, `set_eviction_priority`,
//! `set_atom_op`, `set_atom_type`, `set_reg_addr`, `set_ureg_addr`, `set_rev_pred_src`)
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT, Copyright
//! 2022 Collabora, Ltd.), for SM 8.6: `has_ugpr` (SM 7.3 and later) and the SM 8.0 memory-order
//! encoding. Each form's known-bit mask is checked against PR #4588's row for it
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97,
//! [`REFERENCE_ROWS`]), and PR #4588's translator (`ops_memory.cpp`, `ops_misc.cpp`, `LDC`, and
//! `visitor.cpp`, `Visitor::Run`) says how it reads them.
//!
//! The field value types [`MemType`], [`MemOrder`], [`MemScope`], [`MemEvictionPriority`],
//! [`AtomType`], [`AtomOp`] and [`AtomCmpSrc`], their printed text and the fields
//! [`MEM_TYPE`], [`MEM_ORDER`], [`MEM_EVICTION`], [`ATOM_TYPE`], [`ATOM_OP`] and [`RED_OP`] are
//! defined here once and shared with slice 6f's surface instructions (`op::texture`), as NAK
//! shares them. [`AtomCmpSrc::Packed`] is the surface atomics' compare source; the memory
//! atomics' forms do not encode it, and the test assembler refuses it.
//!
//! **From PR #4588 alone**, each needing an `nvdisasm` check (spec, verification E2): `LDGSTS`
//! (`0xfae`, `Visitor::LDGSTS`), `LDGDEPBAR` (`0x9af`), `ERRBAR` (`0x9ab`) and `ATOMS.POPC`
//! (`0xf8c`, `Visitor::ATOMS_POPC`). NAK has no op for them. So does the meaning of bit 76 of
//! `LDG` and `STG` (below), and the scope value 5 of `MEMBAR`.
//!
//! # Forms
//!
//! NAK encodes two forms of most memory instructions: the uniform one (`0x9..`), with a uniform
//! register added to the address, which SM 7.3 and later get (`has_ugpr`), and the older one
//! (`0x3..`) without it. Both are decoded: PR #4588's translator handles `STL` (`0x387`),
//! `STS` (`0x388`) and `ATOMS` (`0x38c`) in the older form, and NAK still encodes the
//! compare-and-swap atomics (`0x3a9`, `0x38d`) only so. `LDL` and `LDS` have only the uniform
//! form (`0x983`, `0x984`), and so does `RED` (`0x98e`), which NAK writes in the uniform layout
//! whenever it has uniform registers. Generic (`LD`, `ST`) loads and stores have no row in PR
//! #4588's table and no NAK encoding, so there is none.
//!
//! The memory spaces share a variant, as in NAK: [`OpLd`], [`OpSt`] and [`OpAtom`] carry a
//! [`MemSpace`]; `RED` is an [`OpAtom`] whose [`AtomDst`] is [`AtomDst::Reduction`], and the
//! compare-and-swap forms are [`AtomOp::CmpExch`]. The uniform datapath's `ULDC` (`0xab9`) and
//! `ULDC.64` (`0xabb`) are slice 6d's (spec, port slices), which will give [`OpLdc`] its uniform
//! destination.
//!
//! # Decisions taken here
//!
//! - **The uniform address bit (91) is read as written.** NAK always sets it in the uniform
//!   forms ("We always enable UGPR mode, because the .E bit changes which source it applies to
//!   depending on it"), and PR #4588's translator reads it as "use the uniform register"
//!   (`LDG`, `STG`, `ATOMG`: `use_uniform = Take(91, 1)`) but requires it clear in `LDL`, `LDS`,
//!   `STL` and `STS`. It is [`UniformAddr::enabled`]; the other address bits keep the meaning NAK
//!   gives them with it set (bit 72, `.E`, the uniform register's 64-bit flag; bit 90, or 70 in
//!   `ATOMG`, the register's). With it clear the references conflict: NAK's comment says `.E`
//!   then applies to the register, PR #4588 still takes the register's flag from bit 90. The
//!   decoder reads the bits; which reading holds is slice 12e's, and a guess until an `nvdisasm`
//!   check.
//! - **Bit 76 of `LDG` (`0x381`, `0x981`) and of the uniform `STG` (`0x986`)** is marked known by
//!   PR #4588's rows, and its translator ignores it with the memory order (`Ignore(76, 4)`, "memory
//!   ordering"), while NAK never writes it (its SM 8.0 order is bits 77..81). Under decision 8
//!   ("where one answer contains the other, the larger") the bit is accepted; its meaning is
//!   unknown, so it is kept as `bit76` and not printed: two words that differ only there print
//!   the same text.
//! - **Fields NAK writes as a constant** are carried as fields, so the decoder reads every value:
//!   the predicate destination of `LDG` and `ATOMG` (NAK: `set_pred_dst(81..84, &Dst::None)`),
//!   `LDS`'s bit 87 (NAK: `set_bit(87, false); // !.ZD - Returns a predicate?`), the eviction
//!   priority of `LDL` and `STL` (NAK: `set_field(84..87, 1_u8)`, `MemEvictionPriority::Normal`'s
//!   code; PR #4588: `Ignore(84, 3); // cache eviction priority`), `MEMBAR`'s `.MMIO` (bit 72) and
//!   `.SC` (bit 80) bits, and `RED`'s destination field (NAK writes `RZ` there through
//!   `set_dst(&Dst::None)`; PR #4588's translator does not read it), kept as
//!   [`AtomDst::Reduction`]'s `ignored` and not printed.
//! - **Constant-buffer handles.** A bindless `LDC`'s handle is any uniform register, odd ones
//!   and `URZ` included, as the encoding allows; NAK always passes a 64-bit pair. Whether an
//!   unaligned handle is valid is for slices 12e and 13 to decide, with a source.
//! - **`MEMBAR`'s scope** decodes NAK's values (`CTA` 0, `GPU` 2, `SYS` 3, and `SM` 1 from NAK's
//!   comment `// SM => 1_u8`) and PR #4588's 5, which its translator accepts as a device-wide
//!   barrier (`Require(scope == 0 || scope == 2 || scope == 3 || scope == 5)`). Neither set
//!   contains the other, and decision 8 would then take the one a title ran on (PR #4588's);
//!   the decoder takes both. **Owner question**: the union departs from decision 8's
//!   wording, so that `SM`, which NAK names, is not refused.
//! - **`LDGSTS` and `ATOMS.POPC`** are decoded as PR #4588's translator reads them: the bits it
//!   requires set are fixed bits of the form, the bits it requires clear or does not read are
//!   rejected (the `dropped` bits of their rows), and every field it reads is decoded. The
//!   `LDGSTS` enable predicate is the one exception: PR #4588 requires bits 85..90 clear
//!   (`Take(85, 5) == 0`) and then reads bits 87..90 as the predicate (`Take(87, 3)`, "the enable
//!   predicate (87..89 plus the negate bit 70)"), so its translator accepts only `P0`; the
//!   decoder reads any predicate there, the larger answer under decision 8. **Owner question**:
//!   the alternative is to accept the whole row and keep the unexplained bits as raw fields, as
//!   `BAR` does (`op::control`); the stricter reading was taken here because no reference names
//!   those fields.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | Bit 91 of `LDG` `0x381`, `STG` `0x386`, `STL` `0x387`, `STS` `0x388` and `LDC` `0xb82` | Rejected: NAK writes it clear in these forms (`set_bit(91, has_ugpr)` without uniform registers; `set_bit(91, false); // Bound`), which have no uniform register for it to enable, and PR #4588's translator requires it clear (`"native global load flags"`, `"native bindless constant load"`, and the others) | `decode::tests::memory::memory_rows_drop_exactly_the_listed_bits` |
//! | Bits 64..71 and 91 of `ATOMG` `0x3a8` | Rejected: the row is the uniform form's (`0x9a8`); NAK's `0x3a8` has no uniform register (`set_reg_addr(24..32, .., 72)` only), and PR #4588's translator reads neither there | `memory_rows_drop_exactly_the_listed_bits` |
//! | `LDGSTS` bits 44..48, 71, 72..74, 77..84 and 85..87 | Rejected: PR #4588's translator requires them clear or does not read them | `memory_rows_drop_exactly_the_listed_bits`, `ldgsts_requires_the_bits_pr_4588_requires` |
//! | `LDGSTS` bits 74..77, 84, 90 and 91 | Fixed bits: PR #4588 requires them set (`Take(74, 3) == 7`, `Take(84, 1) == 1`, `Take(90, 2) == 3`) | `ldgsts_requires_the_bits_pr_4588_requires` |
//! | `ATOMS.POPC` bits 32..64, 73..76, 87 and 89 | Rejected: PR #4588 requires them clear (`Take(32, 32) == 0`, `Take(73, 3) == 0`, the operation 10 in bits 87..91) | `memory_rows_drop_exactly_the_listed_bits`, `atoms_popc_requires_the_bits_pr_4588_requires` |
//! | `ATOMS.POPC` bits 16..24, 64..70, 88, 90 and 91 | Fixed bits: PR #4588 requires `RZ`, `URZ`, the operation 10 and bit 91 (`dest == 255`, `Take(64, 6) == 63`, `Take(87, 4) == 10 && Take(91, 1) == 1`) | `atoms_popc_requires_the_bits_pr_4588_requires` |
//! | `ULDC` and `ULDC.64` (`0xab9`, `0xabb`) | Unknown opcodes here: slice 6d's | `memory_rows_left_undecoded_are_unknown_opcodes` |
//! | `LDCG` (`0x8b8`, NAK's `OpLdcg`) | Unknown opcode: no slice owns it yet (no slice lists it) | `memory_rows_left_undecoded_are_unknown_opcodes` |
//!
//! Every other form's mask is its row's exactly (`decode::tests::masks_match_the_reference_table`).
//!
//! # Values the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: a memory type of 7 (NAK's
//! `set_mem_type` encodes 0 to 6; PR #4588: `Require(type <= 6)`); a memory order other than
//! the five NAK's SM 8.0 encoding names (0, 4, 5, 7 and 10); an eviction priority of 6 or 7 (NAK
//! encodes 0 to 5); an atomic type of 7 (NAK encodes 0 to 6); an atomic operation of 9 to 15 in
//! the four-bit field (NAK's `set_atom_op` encodes 0 to 8; PR #4588: `Require(atom_op <= 8)`);
//! a cache operation of 9 to 15 (NAK encodes 0 to 8); a `MEMBAR` scope of 4, 6 or 7. A word with
//! more than one is refused for the lowest field: each instruction reads its fields in bit
//! order (type at 73, order at 77, eviction at 84, operation at 87). Translator refusals PR
//! #4588 makes (an odd register pair, a type its atomics do not handle, a predicate destination
//! other than `PT`) are slice 12e's, not the decoder's.
//!
//! # Not decided here
//!
//! The text of `LDC` (NAK's `test_ldc`, every type but `.128`) and of `LDG`, `LDS`, `LDL`,
//! `STG`, `STS`, `STL`, `ATOMG`, `ATOMS` and `RED` in the cases NAK's `test_ld_st_atom` covers
//! (32-bit accesses, `.ef` and `.strong.sm`, register addresses with `.64`, `.x8` and a uniform
//! register, positive and negative offsets) is `nvdisasm`'s. Everything else is a guess in its
//! style (`print`'s module docs): the other types, orders and eviction priorities, `.cas`, a
//! zero offset left out, the uniform register left out when bit 91 is clear or it is `URZ`, the
//! load predicate and predicate destinations, `.zd`, the text of `CCTL`, `MEMBAR`, `ERRBAR`,
//! `LDGDEPBAR`, `LDGSTS` and `ATOMS.POPC`, and `LDC`'s modes and `.128`.

#[cfg(test)]
mod generate;

use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::fields::{CBUF_INDEX, CBUF_OFFSET, DST, mask_of};
use super::super::form::{DecodeFn, Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{CBuf, CBufRef, Gpr, Pred, PredSrc, UniformReg};
use super::super::print::flag;
use super::control::{COND, PRED_DST, UNIFORM_BIT};
use super::{Named, Op, shells};
#[cfg(test)]
use {super::super::form::ReferenceRow, proptest::strategy::BoxedStrategy};

/// Defines a field value type: an enum whose variants are encoded by the listed values, with
/// its every variant in encoding order (`ALL`), its encoding (`bits`) and the variant a value
/// names (`from_bits`, `None` for a value that names nothing).
macro_rules! coded {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $($(#[$vmeta:meta])* $variant:ident = $bits:literal,)*
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)*
        }

        impl $name {
            /// Every value, in encoding order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// The encoding.
            pub const fn bits(self) -> u64 {
                match self {
                    $(Self::$variant => $bits,)*
                }
            }

            /// The value an encoding names, if any.
            pub const fn from_bits(bits: u64) -> Option<Self> {
                match bits {
                    $($bits => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Fields several instructions share.

/// The address register of every load, store, atomic and `CCTL` (NAK: `set_reg_addr(24..32, ..)`
/// or `set_reg_src(24..32, &self.addr)`).
pub(crate) const MEM_ADDR: Field = Field::new(24, Gpr::BITS);
/// The signed 24-bit address offset of loads, stores and atomics (NAK: `set_field(40..64,
/// self.offset)`; PR #4588: `Take(40, 24)`, sign-extended from bit 23).
pub(crate) const MEM_OFFSET: Field = Field::new(40, 24);
/// The data register of stores and atomics but compare-and-swap (NAK: `set_reg_src(32..40,
/// &self.data)`).
pub(crate) const MEM_DATA: Field = Field::new(32, Gpr::BITS);
/// The uniform address register of the uniform loads (NAK `OpLd`: `set_ureg_addr(32, ..)` and
/// `set_ureg_src(32, ..)`, six bits below SM 10.0).
pub(crate) const LD_UNIFORM: Field = Field::new(32, UniformReg::BITS);
/// The uniform address register of the uniform stores and atomics (NAK `OpSt` and `OpAtom`:
/// `set_ureg_addr(64, ..)` and `set_ureg_src(64, ..)`).
pub(crate) const ST_UNIFORM: Field = Field::new(64, UniformReg::BITS);
/// `.E`, the 64-bit flag at bit 72 of the global forms: the uniform register's in the uniform
/// forms (NAK: `set_ureg_addr(.., 72)`), the register's in the others (`set_reg_addr(24..32, ..,
/// 72)`).
pub(crate) const MEM_E: Field = Field::bit(72);
/// The register's 64-bit flag in the uniform `LDG`, `STG` and `RED` (NAK: `set_reg_addr(24..32,
/// .., 90)`).
pub(crate) const MEM_REG_64: Field = Field::bit(90);
/// The register's 64-bit flag in the uniform `ATOMG` (NAK: `set_reg_addr(24..32, .., 70)`; PR
/// #4588: `reg_pair = Take(70, 1)`).
pub(crate) const ATOMG_REG_64: Field = Field::bit(70);
/// The memory type of loads, stores and `LDC` (NAK: `set_mem_type(73..76, ..)`).
pub(crate) const MEM_TYPE: Field = Field::new(73, 3);
/// Bit 76 of `LDG` and the uniform `STG` (module docs).
pub(crate) const MEM_BIT_76: Field = Field::bit(76);
/// The memory order of the global forms (NAK's SM 8.0 `set_mem_order`: `set_field(77..81, ..)`).
pub(crate) const MEM_ORDER: Field = Field::new(77, 4);
/// The address stride of the shared forms (NAK: `set_field(78..80,
/// self.stride.encode_sm75())`).
pub(crate) const MEM_STRIDE: Field = Field::new(78, 2);
/// The eviction priority of the global and local forms (NAK: `set_eviction_priority`,
/// `set_field(84..87, ..)`).
pub(crate) const MEM_EVICTION: Field = Field::new(84, 3);

// ---------------------------------------------------------------------------------------------
// Field value types.

coded! {
    /// How many bytes a load or store moves, and how a narrow load extends (NAK's `MemType` and
    /// `set_mem_type`).
    pub enum MemType {
        /// 0: a byte, zero-extended.
        U8 = 0,
        /// 1: a byte, sign-extended.
        I8 = 1,
        /// 2: 16 bits, zero-extended.
        U16 = 2,
        /// 3: 16 bits, sign-extended.
        I16 = 3,
        /// 4: 32 bits.
        B32 = 4,
        /// 5: 64 bits, a register pair.
        B64 = 5,
        /// 6: 128 bits, four registers.
        B128 = 6,
    }
}

/// The scope of a strong memory order (NAK's `MemScope` without `SM`, which its SM 8.0 order
/// encoding does not name).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemScope {
    /// The workgroup (`nvdisasm`: `.sm` on SM 8.0 and later, NAK's `test_ld_st_atom`).
    Cta,
    /// The GPU.
    Gpu,
    /// The system.
    System,
}

/// A memory order (NAK's `MemOrder` and its SM 8.0 encoding in `set_mem_order`: weak 0,
/// constant 4, strong 5, 7 and 10 for the workgroup, the GPU and the system).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemOrder {
    /// 0: no ordering.
    Weak,
    /// 4: the memory does not change while the program runs.
    Constant,
    /// Ordered at a scope.
    Strong(MemScope),
}

impl MemOrder {
    /// Every order, in encoding order.
    pub const ALL: &'static [Self] = &[
        Self::Weak,
        Self::Constant,
        Self::Strong(MemScope::Cta),
        Self::Strong(MemScope::Gpu),
        Self::Strong(MemScope::System),
    ];

    /// The four-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Weak => 0x0,
            Self::Constant => 0x4,
            Self::Strong(MemScope::Cta) => 0x5,
            Self::Strong(MemScope::Gpu) => 0x7,
            Self::Strong(MemScope::System) => 0xa,
        }
    }

    /// The order an encoding names, if any.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0x0 => Some(Self::Weak),
            0x4 => Some(Self::Constant),
            0x5 => Some(Self::Strong(MemScope::Cta)),
            0x7 => Some(Self::Strong(MemScope::Gpu)),
            0xa => Some(Self::Strong(MemScope::System)),
            _ => None,
        }
    }
}

coded! {
    /// How the cache keeps the line (NAK's `MemEvictionPriority` and `set_eviction_priority`).
    pub enum MemEvictionPriority {
        /// 0: evict first.
        First = 0,
        /// 1: normal.
        Normal = 1,
        /// 2: evict last.
        Last = 2,
        /// 3: last use.
        LastUse = 3,
        /// 4: leave the priority unchanged.
        Unchanged = 4,
        /// 5: do not allocate.
        NoAllocate = 5,
    }
}

coded! {
    /// The scale of a shared address register (NAK's `OffsetStride` and `encode_sm75`).
    pub enum OffsetStride {
        /// 0: bytes.
        X1 = 0,
        /// 1: times 4.
        X4 = 1,
        /// 2: times 8.
        X8 = 2,
        /// 3: times 16.
        X16 = 3,
    }
}

/// The fields of a global access that local and shared accesses do not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlobalAccess {
    /// The memory order ([`MEM_ORDER`]).
    pub order: MemOrder,
    /// The eviction priority ([`MEM_EVICTION`]).
    pub eviction: MemEvictionPriority,
}

/// Which memory a load, store or atomic accesses, and the fields only that space has (NAK's
/// `MemSpace`, with `MemAccess`'s order and eviction priority where the space encodes them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemSpace {
    /// Global memory.
    Global(GlobalAccess),
    /// Per-thread local memory, with its eviction priority ([`MEM_EVICTION`]).
    Local(MemEvictionPriority),
    /// Workgroup shared memory, with the scale of its address register ([`MEM_STRIDE`]).
    Shared(OffsetStride),
}

/// The uniform register added to an address, in the uniform forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UniformAddr {
    /// The register; `URZ` adds nothing.
    pub reg: UniformReg,
    /// `.E`, bit 72: the register is a 64-bit pair (global forms only).
    pub wide: bool,
    /// Bit 91, which NAK always sets in these forms (module docs).
    pub enabled: bool,
}

/// An address: a register, in the uniform forms a uniform register, and a signed immediate
/// (NAK's `addr`, `uniform_addr` and `offset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemAddr {
    /// The register; `RZ` adds nothing.
    pub reg: Gpr,
    /// The register is a 64-bit pair (global forms only: bit 90, or 70 in `ATOMG`, in the
    /// uniform forms, `.E` in the others).
    pub reg_64: bool,
    /// The uniform register, in the uniform forms.
    pub uniform: Option<UniformAddr>,
    /// The byte offset, a signed 24-bit number ([`MEM_OFFSET`]).
    pub offset: i32,
}

/// Where a form keeps the parts of its address that not every form has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AddrLayout {
    /// The uniform register's field, in the uniform forms.
    pub(crate) uniform: Option<Field>,
    /// Whether the form has `.E` ([`MEM_E`]): the global forms.
    pub(crate) e: bool,
    /// The register's 64-bit flag in the uniform global forms.
    pub(crate) reg_64: Option<Field>,
}

impl AddrLayout {
    /// A form with no uniform register and no 64-bit flags (`LDS`, `STL`, `ATOMS`, ...).
    pub(crate) const PLAIN: Self = Self {
        uniform: None,
        e: false,
        reg_64: None,
    };

    /// The bits this layout's address occupies.
    pub(crate) const fn known(self) -> u128 {
        let mut mask = MEM_ADDR.mask() | MEM_OFFSET.mask();
        if let Some(field) = self.uniform {
            mask |= field.mask() | UNIFORM_BIT.mask();
        }
        if self.e {
            mask |= MEM_E.mask();
        }
        if let Some(field) = self.reg_64 {
            mask |= field.mask();
        }
        mask
    }
}

/// The address of the global forms without a uniform register: `.E` is the register's flag.
pub(crate) const GLOBAL: AddrLayout = AddrLayout {
    uniform: None,
    e: true,
    reg_64: None,
};
/// `LDG`'s uniform form.
pub(crate) const LDG_UNIFORM: AddrLayout = AddrLayout {
    uniform: Some(LD_UNIFORM),
    e: true,
    reg_64: Some(MEM_REG_64),
};
/// `LDL`'s and `LDS`'s only form.
pub(crate) const LD_LOCAL_SHARED: AddrLayout = AddrLayout {
    uniform: Some(LD_UNIFORM),
    e: false,
    reg_64: None,
};
/// The uniform `STG` and `RED`.
pub(crate) const ST_GLOBAL_UNIFORM: AddrLayout = AddrLayout {
    uniform: Some(ST_UNIFORM),
    e: true,
    reg_64: Some(MEM_REG_64),
};
/// The uniform `STL`, `STS` and `ATOMS`.
pub(crate) const ST_UNIFORM_PLAIN: AddrLayout = AddrLayout {
    uniform: Some(ST_UNIFORM),
    e: false,
    reg_64: None,
};
/// The uniform `ATOMG`.
pub(crate) const ATOMG_UNIFORM: AddrLayout = AddrLayout {
    uniform: Some(ST_UNIFORM),
    e: true,
    reg_64: Some(ATOMG_REG_64),
};

/// Reads an address in `layout`.
fn read_addr(reader: &mut Reader, layout: AddrLayout) -> MemAddr {
    let reg = reader.gpr(MEM_ADDR);
    // Cannot fail: the field is 24 bits wide.
    let offset = i32::try_from(reader.signed(MEM_OFFSET)).unwrap_or(0);
    match layout.uniform {
        Some(field) => MemAddr {
            reg,
            reg_64: layout.reg_64.is_some_and(|field| reader.flag(field)),
            uniform: Some(UniformAddr {
                reg: reader.uniform_reg(field),
                wide: layout.e && reader.flag(MEM_E),
                enabled: reader.flag(UNIFORM_BIT),
            }),
            offset,
        },
        None => MemAddr {
            reg,
            reg_64: layout.e && reader.flag(MEM_E),
            uniform: None,
            offset,
        },
    }
}

/// Reads a field holding a [`coded!`] value, refusing a value that names nothing.
fn read_coded<T>(
    reader: &mut Reader,
    field: Field,
    from_bits: fn(u64) -> Option<T>,
) -> Result<T, DecodeError> {
    let bits = reader.field(field);
    from_bits(bits).ok_or_else(|| reader.unknown_value(field, bits))
}

/// Reads a global access: order, then eviction priority.
fn global_access(reader: &mut Reader) -> Result<GlobalAccess, DecodeError> {
    Ok(GlobalAccess {
        order: read_coded(reader, MEM_ORDER, MemOrder::from_bits)?,
        eviction: read_coded(reader, MEM_EVICTION, MemEvictionPriority::from_bits)?,
    })
}

/// Reads a predicate destination at bits 81..84 ([`PRED_DST`]); `PT` writes nothing.
fn pred_dst(reader: &mut Reader) -> Pred {
    // Cannot fail: the field is `Pred::BITS` wide.
    Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT)
}

/// Which space a load or store form accesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Space {
    /// Global memory.
    Global,
    /// Local memory.
    Local,
    /// Shared memory.
    Shared,
}

/// Reads the space's fields, in bit order, after the memory type.
fn read_space(reader: &mut Reader, space: Space) -> Result<MemSpace, DecodeError> {
    Ok(match space {
        Space::Global => MemSpace::Global(global_access(reader)?),
        Space::Local => MemSpace::Local(read_coded(
            reader,
            MEM_EVICTION,
            MemEvictionPriority::from_bits,
        )?),
        Space::Shared => MemSpace::Shared(stride(reader)),
    })
}

/// Reads a shared address's stride ([`MEM_STRIDE`]).
fn stride(reader: &mut Reader) -> OffsetStride {
    // Cannot fail: every two-bit value names a stride.
    OffsetStride::from_bits(reader.field(MEM_STRIDE)).unwrap_or(OffsetStride::X1)
}

/// Writes `value` as a signed hexadecimal offset after a `+`: `+0x12`, `+-0x1` (NAK's
/// `test_ld_st_atom`), nothing for 0.
fn offset_text(f: &mut Formatter<'_>, value: i64) -> fmt::Result {
    match value.cmp(&0) {
        Ordering::Equal => Ok(()),
        Ordering::Less => write!(f, "+-{:#x}", value.unsigned_abs()),
        Ordering::Greater => write!(f, "+{value:#x}"),
    }
}

/// An address as `nvdisasm` writes it: `[r4.64+ur2+0x12]`, `[r4.x8+ur2+-0x1]`.
struct AddrText {
    addr: MemAddr,
    /// Whether the register's 64-bit flag prints as `.64` (the uniform global forms; in the
    /// others it is `.E`).
    reg_64_suffix: bool,
    stride: OffsetStride,
}

impl Display for AddrText {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[{}", self.addr.reg)?;
        flag(f, self.reg_64_suffix && self.addr.reg_64, ".64")?;
        write!(f, "{}", self.stride)?;
        if let Some(uniform) = self.addr.uniform
            && uniform.enabled
            && !uniform.reg.is_zero()
        {
            write!(f, "+{}", uniform.reg)?;
        }
        offset_text(f, i64::from(self.addr.offset))?;
        f.write_str("]")
    }
}

impl MemAddr {
    /// `.E` as `nvdisasm` prints it: the uniform register's flag in the uniform forms, the
    /// register's in the others.
    fn e(self) -> bool {
        match self.uniform {
            Some(uniform) => uniform.wide,
            None => self.reg_64,
        }
    }

    /// The address's text in `space`.
    fn text(self, space: MemSpace) -> AddrText {
        AddrText {
            addr: self,
            reg_64_suffix: matches!(space, MemSpace::Global(_)) && self.uniform.is_some(),
            stride: match space {
                MemSpace::Shared(stride) => stride,
                MemSpace::Global(_) | MemSpace::Local(_) => OffsetStride::X1,
            },
        }
    }
}

impl Display for MemType {
    /// NAK's `test_ldc`: `.u8`, `.s8`, `.u16`, `.s16`, nothing for 32 bits, `.64`; `.128` is a
    /// guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::U8 => ".u8",
            Self::I8 => ".s8",
            Self::U16 => ".u16",
            Self::I16 => ".s16",
            Self::B32 => "",
            Self::B64 => ".64",
            Self::B128 => ".128",
        })
    }
}

impl Display for MemOrder {
    /// NAK's `test_ld_st_atom`: `.strong.sm` for the workgroup on SM 8.0 and later; the others
    /// are guesses (nothing for weak).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Weak => "",
            Self::Constant => ".constant",
            Self::Strong(MemScope::Cta) => ".strong.sm",
            Self::Strong(MemScope::Gpu) => ".strong.gpu",
            Self::Strong(MemScope::System) => ".strong.sys",
        })
    }
}

impl Display for MemEvictionPriority {
    /// NAK's `MemEvictionPriority` printer; `test_ld_st_atom` confirms `.ef` and nothing for
    /// normal, the others are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::First => ".ef",
            Self::Normal => "",
            Self::Last => ".el",
            Self::LastUse => ".lu",
            Self::Unchanged => ".eu",
            Self::NoAllocate => ".na",
        })
    }
}

impl Display for OffsetStride {
    /// NAK's `OffsetStride` printer; `test_ld_st_atom` confirms `.x8`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::X1 => "",
            Self::X4 => ".x4",
            Self::X8 => ".x8",
            Self::X16 => ".x16",
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores.

/// `LDG` without a uniform register (NAK `OpLd`, `has_ugpr` false: `set_opcode(0x381)`).
pub(crate) const LDG_OPCODE: u16 = 0x381;
/// `LDG` (`set_opcode(0x981)`).
pub(crate) const LDG_UNIFORM_OPCODE: u16 = 0x981;
/// `LDL` (`set_opcode(0x983)`).
pub(crate) const LDL_OPCODE: u16 = 0x983;
/// `LDS` (`set_opcode(0x984)`).
pub(crate) const LDS_OPCODE: u16 = 0x984;
/// `STG` without a uniform register (NAK `OpSt`: `set_opcode(0x386)`).
pub(crate) const STG_OPCODE: u16 = 0x386;
/// `STG` (`set_opcode(0x986)`).
pub(crate) const STG_UNIFORM_OPCODE: u16 = 0x986;
/// `STL` without a uniform register (`set_opcode(0x387)`).
pub(crate) const STL_OPCODE: u16 = 0x387;
/// `STL` (`set_opcode(0x987)`).
pub(crate) const STL_UNIFORM_OPCODE: u16 = 0x987;
/// `STS` without a uniform register (`set_opcode(0x388)`).
pub(crate) const STS_OPCODE: u16 = 0x388;
/// `STS` (`set_opcode(0x988)`).
pub(crate) const STS_UNIFORM_OPCODE: u16 = 0x988;
/// The load predicate of the uniform `LDG`, encoded reversed: the field holds 7 minus the
/// register (NAK: `set_rev_pred_src(64..67, 67, &self.pred)`, "These sources are funky.
/// They're encoded backwards."; PR #4588: `Pred(7 - load_pred_field, ..)`).
pub(crate) const LDG_PRED: Field = Field::new(64, Pred::BITS);
/// The load predicate's negation.
pub(crate) const LDG_PRED_NOT: Field = Field::bit(67);
/// `LDS`'s bit 87 (NAK: `set_bit(87, false); // !.ZD - Returns a predicate?`).
pub(crate) const LDS_ZD: Field = Field::bit(87);

/// `LD dst, [addr]`: a load (NAK's `OpLd`): `LDG`, `LDL` or `LDS` by its space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLd {
    /// The first destination register; a wide type fills the ones after it.
    pub dst: Gpr,
    /// The address.
    pub addr: MemAddr,
    /// The type.
    pub mem_type: MemType,
    /// The space.
    pub space: MemSpace,
    /// The uniform `LDG`'s load predicate: where it is false the load returns 0 (NAK: "On false
    /// the load returns 0"); `PT` in the other forms.
    pub pred: PredSrc,
    /// `LDG`'s predicate destination ([`PRED_DST`]; NAK writes `PT`); `PT` in the other forms.
    pub pred_dst: Pred,
    /// `LDS`'s bit 87 ([`LDS_ZD`]); false in the other forms.
    pub zd: bool,
    /// `LDG`'s bit 76 ([`MEM_BIT_76`], module docs); false in the other forms.
    pub bit76: bool,
}

/// `ST [addr], data`: a store (NAK's `OpSt`): `STG`, `STL` or `STS` by its space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSt {
    /// The first data register; a wide type reads the ones after it.
    pub data: Gpr,
    /// The address.
    pub addr: MemAddr,
    /// The type.
    pub mem_type: MemType,
    /// The space.
    pub space: MemSpace,
    /// The uniform `STG`'s bit 76 ([`MEM_BIT_76`], module docs); false in the other forms.
    pub bit76: bool,
}

impl Named for OpLd {
    const MNEMONIC: &'static str = "ld";
}

impl Named for OpSt {
    const MNEMONIC: &'static str = "st";
}

/// The `.E`, eviction, type and order modifiers of a global access, in `test_ld_st_atom`'s
/// order; a local access prints its eviction priority and type, a shared one its type.
fn access_modifiers(
    f: &mut Formatter<'_>,
    addr: MemAddr,
    space: MemSpace,
    mem_type: &dyn Display,
) -> fmt::Result {
    match space {
        MemSpace::Global(access) => {
            flag(f, addr.e(), ".e")?;
            write!(f, "{}{mem_type}{}", access.eviction, access.order)
        }
        MemSpace::Local(eviction) => write!(f, "{eviction}{mem_type}"),
        MemSpace::Shared(_) => write!(f, "{mem_type}"),
    }
}

/// Writes the mnemonic of a load or store in `space`: `prefix` and the space's letter.
fn space_mnemonic(f: &mut Formatter<'_>, prefix: &str, space: MemSpace) -> fmt::Result {
    let suffix = match space {
        MemSpace::Global(_) => "g",
        MemSpace::Local(_) => "l",
        MemSpace::Shared(_) => "s",
    };
    write!(f, "{prefix}{suffix}")
}

impl Display for OpLd {
    /// NAK's `test_ld_st_atom`: `ldg.e.ef.strong.sm r0, [r4.64+ur2+0x12], p4;`,
    /// `lds r0, [r4.x8+ur2+0x12];`, `ldl r0, [r4+ur2+0x12];`. The predicate destination before
    /// the destination and the load predicate left out when `PT` are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        space_mnemonic(f, "ld", self.space)?;
        access_modifiers(f, self.addr, self.space, &self.mem_type)?;
        flag(f, self.zd, ".zd")?;
        f.write_str(" ")?;
        if !self.pred_dst.is_true() {
            write!(f, "{}, ", self.pred_dst)?;
        }
        write!(f, "{}, {}", self.dst, self.addr.text(self.space))?;
        if !self.pred.is_true() {
            write!(f, ", {}", self.pred)?;
        }
        Ok(())
    }
}

impl Display for OpSt {
    /// NAK's `test_ld_st_atom`: `stg.e.ef.strong.sm [r4.64+ur2+0x12], r2;`,
    /// `sts [r4.x8+ur2+0x12], r2;`, `stl [r4+ur2+0x12], r2;`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        space_mnemonic(f, "st", self.space)?;
        access_modifiers(f, self.addr, self.space, &self.mem_type)?;
        write!(f, " {}, {}", self.addr.text(self.space), self.data)
    }
}

/// Reads a load in `space` with its address in `layout`.
fn decode_ld(reader: &mut Reader, space_of: Space, layout: AddrLayout) -> Result<Op, DecodeError> {
    let dst = reader.gpr(DST);
    let addr = read_addr(reader, layout);
    let mem_type = read_coded(reader, MEM_TYPE, MemType::from_bits)?;
    let global = space_of == Space::Global;
    let bit76 = global && reader.flag(MEM_BIT_76);
    let space = read_space(reader, space_of)?;
    let pred = if global && layout.uniform.is_some() {
        let field = reader.byte(LDG_PRED);
        PredSrc {
            // Cannot fail: the field is three bits wide.
            reg: Pred::new(Pred::PT.index().saturating_sub(field)).unwrap_or(Pred::PT),
            negated: reader.flag(LDG_PRED_NOT),
        }
    } else {
        PredSrc::TRUE
    };
    Ok(Op::Ld(OpLd {
        dst,
        addr,
        mem_type,
        space,
        pred,
        pred_dst: if global { pred_dst(reader) } else { Pred::PT },
        zd: space_of == Space::Shared && reader.flag(LDS_ZD),
        bit76,
    }))
}

/// Reads a store in `space` with its address in `layout`.
fn decode_st(reader: &mut Reader, space_of: Space, layout: AddrLayout) -> Result<Op, DecodeError> {
    let data = reader.gpr(MEM_DATA);
    let addr = read_addr(reader, layout);
    let mem_type = read_coded(reader, MEM_TYPE, MemType::from_bits)?;
    let bit76 = space_of == Space::Global && layout.uniform.is_some() && reader.flag(MEM_BIT_76);
    Ok(Op::St(OpSt {
        data,
        addr,
        mem_type,
        space: read_space(reader, space_of)?,
        bit76,
    }))
}

/// The bits of a space's own fields.
pub(crate) const fn space_known(space: Space) -> u128 {
    match space {
        Space::Global => MEM_ORDER.mask() | MEM_EVICTION.mask(),
        Space::Local => MEM_EVICTION.mask(),
        Space::Shared => MEM_STRIDE.mask(),
    }
}

/// The form of a load.
const fn ld_form(opcode: u16, space: Space, layout: AddrLayout, decode: DecodeFn) -> Form {
    let mut fields = DST.mask() | MEM_TYPE.mask() | layout.known() | space_known(space);
    if matches!(space, Space::Global) {
        fields |= MEM_BIT_76.mask() | PRED_DST.mask();
        if layout.uniform.is_some() {
            fields |= LDG_PRED.mask() | LDG_PRED_NOT.mask();
        }
    }
    if matches!(space, Space::Shared) {
        fields |= LDS_ZD.mask();
    }
    Form::new(opcode, fields, decode)
}

/// The form of a store.
const fn st_form(opcode: u16, space: Space, layout: AddrLayout, decode: DecodeFn) -> Form {
    let mut fields = MEM_DATA.mask() | MEM_TYPE.mask() | layout.known() | space_known(space);
    if matches!(space, Space::Global) && layout.uniform.is_some() {
        fields |= MEM_BIT_76.mask();
    }
    Form::new(opcode, fields, decode)
}

// ---------------------------------------------------------------------------------------------
// LDC.

/// `LDC` with a bound buffer (NAK `OpLdc`: `set_opcode(0xb82)`).
pub(crate) const LDC_OPCODE: u16 = 0xb82;
/// `LDC` with a bindless buffer (`set_opcode(0x582)`).
pub(crate) const LDC_BINDLESS_OPCODE: u16 = 0x582;
/// The bound `LDC`'s offset register (`set_reg_src(24..32, &self.offset)`).
pub(crate) const LDC_OFFSET: Field = Field::new(24, Gpr::BITS);
/// The bindless `LDC`'s offset register (`set_reg_src(64..72, &self.offset)`).
pub(crate) const LDC_BINDLESS_OFFSET: Field = Field::new(64, Gpr::BITS);
/// The bindless `LDC`'s handle (`set_ureg(24, handle)`, six bits below SM 10.0).
pub(crate) const LDC_HANDLE: Field = Field::new(24, UniformReg::BITS);
/// The bound `LDC`'s indexing mode (`set_field(78..80, ..)`).
pub(crate) const LDC_MODE: Field = Field::new(78, 2);

coded! {
    /// How `LDC` indexes the buffer (NAK's `LdcMode` and its encoding in `OpLdc`).
    pub enum LdcMode {
        /// 0: the register is added to the offset.
        Indexed = 0,
        /// 1: `.IL`.
        IndexedLinear = 1,
        /// 2: `.IS`.
        IndexedSegmented = 2,
        /// 3: `.ISL`.
        IndexedSegmentedLinear = 3,
    }
}

/// `LDC dst, c[buf][offset + imm]`: a constant-buffer load (NAK's `OpLdc`, the vector
/// datapath's forms; `ULDC` is slice 6d's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLdc {
    /// The first destination register.
    pub dst: Gpr,
    /// The buffer and the immediate byte offset.
    pub cbuf: CBufRef,
    /// A register added to the offset; `RZ` for none.
    pub offset: Gpr,
    /// The indexing mode; a bindless buffer has only [`LdcMode::Indexed`] (NAK:
    /// `assert!(self.mode == LdcMode::Indexed)`).
    pub mode: LdcMode,
    /// The type.
    pub mem_type: MemType,
}

impl Named for OpLdc {
    const MNEMONIC: &'static str = "ldc";
}

impl Display for LdcMode {
    /// NAK's `LdcMode` printer (guesses).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Indexed => "",
            Self::IndexedLinear => ".il",
            Self::IndexedSegmented => ".is",
            Self::IndexedSegmentedLinear => ".isl",
        })
    }
}

impl Display for OpLdc {
    /// NAK's `test_ldc`: `ldc.u8 r4, c[0x5][0x248];`, `ldc r4, cx[ur2][r6+0x248];`. A zero
    /// offset after a register is left out, as NAK's printer does (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ldc{}{} {}, {}[",
            self.mode, self.mem_type, self.dst, self.cbuf.buf
        )?;
        if self.offset.is_zero() {
            write!(f, "{:#x}", self.cbuf.offset)?;
        } else {
            write!(f, "{}", self.offset)?;
            if self.cbuf.offset != 0 {
                write!(f, "+{:#x}", self.cbuf.offset)?;
            }
        }
        f.write_str("]")
    }
}

fn cbuf_offset(reader: &mut Reader) -> u16 {
    // Cannot truncate: the field is sixteen bits wide.
    u16::try_from(reader.field(CBUF_OFFSET)).unwrap_or(u16::MAX)
}

fn decode_ldc(reader: &mut Reader) -> Result<Op, DecodeError> {
    let dst = reader.gpr(DST);
    let offset = reader.gpr(LDC_OFFSET);
    let cbuf = CBufRef {
        offset: cbuf_offset(reader),
        buf: CBuf::Binding(reader.byte(CBUF_INDEX)),
    };
    let mem_type = read_coded(reader, MEM_TYPE, MemType::from_bits)?;
    Ok(Op::Ldc(OpLdc {
        dst,
        cbuf,
        offset,
        // Cannot fail: every two-bit value names a mode.
        mode: LdcMode::from_bits(reader.field(LDC_MODE)).unwrap_or(LdcMode::Indexed),
        mem_type,
    }))
}

fn decode_ldc_bindless(reader: &mut Reader) -> Result<Op, DecodeError> {
    let dst = reader.gpr(DST);
    let handle = reader.uniform_reg(LDC_HANDLE);
    // A fixed bit of the form: the decoder has checked that it is set.
    reader.flag(UNIFORM_BIT);
    let cbuf = CBufRef {
        offset: cbuf_offset(reader),
        buf: CBuf::BindlessUniform(handle),
    };
    let offset = reader.gpr(LDC_BINDLESS_OFFSET);
    let mem_type = read_coded(reader, MEM_TYPE, MemType::from_bits)?;
    Ok(Op::Ldc(OpLdc {
        dst,
        cbuf,
        offset,
        mode: LdcMode::Indexed,
        mem_type,
    }))
}

// ---------------------------------------------------------------------------------------------
// Atomics.

/// `RED` (NAK `OpAtom` without a destination: `set_opcode(0x98e)`).
pub(crate) const RED_OPCODE: u16 = 0x98e;
/// `ATOMG` without a uniform register (`set_opcode(0x3a8)`).
pub(crate) const ATOMG_OPCODE: u16 = 0x3a8;
/// `ATOMG` (`set_opcode(0x9a8)`).
pub(crate) const ATOMG_UNIFORM_OPCODE: u16 = 0x9a8;
/// `ATOMG.CAS` (`set_opcode(0x3a9)`).
pub(crate) const ATOMG_CAS_OPCODE: u16 = 0x3a9;
/// `ATOMS` without a uniform register (`set_opcode(0x38c)`).
pub(crate) const ATOMS_OPCODE: u16 = 0x38c;
/// `ATOMS` (`set_opcode(0x98c)`).
pub(crate) const ATOMS_UNIFORM_OPCODE: u16 = 0x98c;
/// `ATOMS.CAS` (`set_opcode(0x38d)`).
pub(crate) const ATOMS_CAS_OPCODE: u16 = 0x38d;
/// The atomic operation of `ATOMG` and `ATOMS` (NAK: `set_atom_op(87..91, ..)`).
pub(crate) const ATOM_OP: Field = Field::new(87, 4);
/// `RED`'s operation, a bit narrower (`set_atom_op(87..90, ..)`; PR #4588: `Take(87, 3)`).
pub(crate) const RED_OP: Field = Field::new(87, 3);
/// The atomic type (NAK below SM 9.0: `set_atom_type`, `set_field(73..76, ..)`).
pub(crate) const ATOM_TYPE: Field = Field::new(73, 3);
/// The comparison register of the compare-and-swap forms (`set_reg_src(32..40, &self.cmpr)`).
pub(crate) const CAS_CMPR: Field = Field::new(32, Gpr::BITS);
/// The data register of the compare-and-swap forms (`set_reg_src(64..72, &self.data)`).
pub(crate) const CAS_DATA: Field = Field::new(64, Gpr::BITS);

coded! {
    /// The type of an atomic (NAK's `AtomType` and its encoding below SM 9.0).
    pub enum AtomType {
        /// 0: unsigned 32-bit.
        U32 = 0,
        /// 1: signed 32-bit.
        I32 = 1,
        /// 2: unsigned 64-bit.
        U64 = 2,
        /// 3: 32-bit float.
        F32 = 3,
        /// 4: two 16-bit floats.
        F16v2 = 4,
        /// 5: signed 64-bit.
        I64 = 5,
        /// 6: 64-bit float.
        F64 = 6,
    }
}

/// What an atomic does (NAK's `AtomOp`). The compare-and-swap has its own opcodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomOp {
    /// 0: add.
    Add,
    /// 1: minimum.
    Min,
    /// 2: maximum.
    Max,
    /// 3: increment, wrapping at the data.
    Inc,
    /// 4: decrement, wrapping at the data.
    Dec,
    /// 5: bitwise and.
    And,
    /// 6: bitwise or.
    Or,
    /// 7: bitwise exclusive or.
    Xor,
    /// 8: exchange.
    Exch,
    /// Compare and swap: the data replaces the value where it equals the compared value (NAK's
    /// `CmpExch`). It has opcodes of its own, not a value of the operation field.
    CmpExch(AtomCmpSrc),
}

/// Where a compare-and-swap finds the value it compares with (NAK's `AtomCmpSrc`, with the
/// separate source's register carried here rather than in the op, so that only a
/// compare-and-swap has one). Shared by the memory atomics and the surface atomics
/// (`op::texture`'s `SUATOM.CAS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomCmpSrc {
    /// A register of its own (`ATOMG.CAS`, `ATOMS.CAS`: [`CAS_CMPR`]); not encodable for
    /// surfaces.
    Separate {
        /// The comparison value.
        cmpr: Gpr,
    },
    /// Packed with the data, the compared value first: the only source `SUATOM.CAS` takes
    /// (NAK: `assert!(cmp_src == AtomCmpSrc::Packed)`), and one the memory atomics' forms do
    /// not encode.
    Packed,
}

impl AtomOp {
    /// Every operation the operation field encodes, in encoding order.
    pub const ENCODED: [Self; 9] = [
        Self::Add,
        Self::Min,
        Self::Max,
        Self::Inc,
        Self::Dec,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Exch,
    ];

    /// The operation field's encoding (NAK's `set_atom_op`); `None` for the compare-and-swap,
    /// which has no operation field.
    pub const fn bits(self) -> Option<u64> {
        match self {
            Self::Add => Some(0),
            Self::Min => Some(1),
            Self::Max => Some(2),
            Self::Inc => Some(3),
            Self::Dec => Some(4),
            Self::And => Some(5),
            Self::Or => Some(6),
            Self::Xor => Some(7),
            Self::Exch => Some(8),
            Self::CmpExch(_) => None,
        }
    }

    /// The operation an encoding names, if any.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::Add),
            1 => Some(Self::Min),
            2 => Some(Self::Max),
            3 => Some(Self::Inc),
            4 => Some(Self::Dec),
            5 => Some(Self::And),
            6 => Some(Self::Or),
            7 => Some(Self::Xor),
            8 => Some(Self::Exch),
            _ => None,
        }
    }
}

/// Where an atomic's old value goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomDst {
    /// `ATOMG`, `ATOMS`: the register that receives the old value; `RZ` discards it.
    Reg(Gpr),
    /// `RED`: no destination (NAK's `Dst::None`). `ignored` is what the destination field
    /// holds: NAK writes `RZ` (`set_dst(&Dst::None)`), PR #4588's translator does not read it.
    Reduction {
        /// The destination field's value.
        ignored: Gpr,
    },
}

/// `ATOM [addr], data`: an atomic (NAK's `OpAtom`): `ATOMG`, `RED` or `ATOMS` by its space and
/// destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpAtom {
    /// The destination.
    pub dst: AtomDst,
    /// The address.
    pub addr: MemAddr,
    /// The data register; a 64-bit type reads a pair.
    pub data: Gpr,
    /// The operation.
    pub op: AtomOp,
    /// The type.
    pub atom_type: AtomType,
    /// The space: global or shared (NAK: "Atomics do not support local").
    pub space: MemSpace,
    /// `ATOMG`'s predicate destination ([`PRED_DST`]; NAK writes `PT`); `PT` in the other forms.
    pub pred_dst: Pred,
}

impl Named for OpAtom {
    const MNEMONIC: &'static str = "atom";
}

impl Display for AtomOp {
    /// NAK's `AtomOp` printer, which `test_ld_st_atom` confirms but for `.cas` (a guess for the
    /// compare-and-swap with either compare source, which NAK prints `.cmpexch` and
    /// `.cmpexch.packed`).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => ".add",
            Self::Min => ".min",
            Self::Max => ".max",
            Self::Inc => ".inc",
            Self::Dec => ".dec",
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
            Self::Exch => ".exch",
            Self::CmpExch(_) => ".cas",
        })
    }
}

impl Display for AtomType {
    /// NAK's `test_ld_st_atom`: nothing for `U32`, `.s32`, `.64`, `.f32.ftz.rn`, `.f16x2.rn`,
    /// `.s64`, `.f64.rn`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::U32 => "",
            Self::I32 => ".s32",
            Self::U64 => ".64",
            Self::F32 => ".f32.ftz.rn",
            Self::F16v2 => ".f16x2.rn",
            Self::I64 => ".s64",
            Self::F64 => ".f64.rn",
        })
    }
}

impl Display for OpAtom {
    /// NAK's `test_ld_st_atom`: `atomg.e.add.ef.strong.sm pt, r0, [r4.64+ur2+0x12], r2;`,
    /// `red.e.add.ef.strong.sm [r4.64+ur2+0x12], r2;`, `atoms.add rz, [r4.x8+ur2+0x12], r2;`.
    /// The compare-and-swap's comparison register before the data is a guess (NAK's printer).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let MemSpace::Global(access) = self.space {
            f.write_str(match self.dst {
                AtomDst::Reg(_) => "atomg",
                AtomDst::Reduction { .. } => "red",
            })?;
            flag(f, self.addr.e(), ".e")?;
            write!(
                f,
                "{}{}{}{} ",
                self.op, access.eviction, self.atom_type, access.order
            )?;
            if let AtomDst::Reg(dst) = self.dst {
                write!(f, "{}, {dst}, ", self.pred_dst)?;
            }
        } else {
            // A local atomic has no encoding; it prints as a shared one would.
            write!(f, "atoms{}{} ", self.op, self.atom_type)?;
            if let AtomDst::Reg(dst) = self.dst {
                write!(f, "{dst}, ")?;
            }
        }
        write!(f, "{}, ", self.addr.text(self.space))?;
        if let AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr }) = self.op {
            write!(f, "{cmpr}, ")?;
        }
        write!(f, "{}", self.data)
    }
}

/// The kinds of atomic form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtomForm {
    /// `RED`.
    Red,
    /// `ATOMG`.
    Global,
    /// `ATOMG.CAS`.
    GlobalCas,
    /// `ATOMS`.
    Shared,
    /// `ATOMS.CAS`.
    SharedCas,
}

impl AtomForm {
    /// Whether the form is a compare-and-swap.
    const fn cas(self) -> bool {
        matches!(self, Self::GlobalCas | Self::SharedCas)
    }

    /// Whether the form is global.
    const fn global(self) -> bool {
        matches!(self, Self::Red | Self::Global | Self::GlobalCas)
    }
}

fn decode_atom(reader: &mut Reader, form: AtomForm, layout: AddrLayout) -> Result<Op, DecodeError> {
    let dst_field = reader.gpr(DST);
    let addr = read_addr(reader, layout);
    let (data, cmpr) = if form.cas() {
        (reader.gpr(CAS_DATA), Some(reader.gpr(CAS_CMPR)))
    } else {
        (reader.gpr(MEM_DATA), None)
    };
    let atom_type = read_coded(reader, ATOM_TYPE, AtomType::from_bits)?;
    let space = if form.global() {
        MemSpace::Global(global_access(reader)?)
    } else {
        MemSpace::Shared(stride(reader))
    };
    let op = match (cmpr, form) {
        (Some(cmpr), _) => AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr }),
        // Cannot fail: every three-bit value names an operation.
        (None, AtomForm::Red) => AtomOp::from_bits(reader.field(RED_OP)).unwrap_or(AtomOp::Add),
        (None, _) => read_coded(reader, ATOM_OP, AtomOp::from_bits)?,
    };
    let (dst, pred_dst) = match form {
        AtomForm::Red => (AtomDst::Reduction { ignored: dst_field }, Pred::PT),
        AtomForm::Global | AtomForm::GlobalCas => (AtomDst::Reg(dst_field), pred_dst(reader)),
        AtomForm::Shared | AtomForm::SharedCas => (AtomDst::Reg(dst_field), Pred::PT),
    };
    Ok(Op::Atom(OpAtom {
        dst,
        addr,
        data,
        op,
        atom_type,
        space,
        pred_dst,
    }))
}

/// The form of an atomic.
const fn atom_form(opcode: u16, form: AtomForm, layout: AddrLayout, decode: DecodeFn) -> Form {
    let mut fields = DST.mask() | ATOM_TYPE.mask() | layout.known();
    fields |= if form.cas() {
        CAS_DATA.mask() | CAS_CMPR.mask()
    } else {
        MEM_DATA.mask()
    };
    fields |= if form.global() {
        space_known(Space::Global)
    } else {
        space_known(Space::Shared)
    };
    fields |= match form {
        AtomForm::Red => RED_OP.mask(),
        AtomForm::Global => ATOM_OP.mask() | PRED_DST.mask(),
        AtomForm::GlobalCas => PRED_DST.mask(),
        AtomForm::Shared => ATOM_OP.mask(),
        AtomForm::SharedCas => 0,
    };
    Form::new(opcode, fields, decode)
}

// ---------------------------------------------------------------------------------------------
// ATOMS.POPC.

/// `ATOMS.POPC` (PR #4588 alone: row `0xf8c` and `Visitor::ATOMS_POPC`, "ATOMS.POPC.INC.32
/// without a returned value: every active lane adds one to the shared word").
pub(crate) const ATOMS_POPC_OPCODE: u16 = 0xf8c;
/// The operation field's value in every `ATOMS.POPC` (PR #4588: `Take(87, 4) == 10`), whose set
/// bits are those of [`ATOMS_POPC_FIXED_FIELDS`] inside [`ATOM_OP`].
#[cfg(test)]
pub(crate) const ATOMS_POPC_OP: u64 = 10;
/// The fields every `ATOMS.POPC` has all set: `RZ` as the destination (`dest == 255`), `URZ` at
/// bit 64 (`Take(64, 6) == 63`), bits 88 and 90, the set bits of the operation 10 in bits 87..91
/// (`Take(87, 4) == 10`), and bit 91 (`Take(91, 1) == 1`).
pub(crate) const ATOMS_POPC_FIXED_FIELDS: [Field; 5] = [
    DST,
    ST_UNIFORM,
    Field::bit(ATOM_OP.start + 1),
    Field::bit(ATOM_OP.start + 3),
    UNIFORM_BIT,
];
/// The bits of [`ATOMS_POPC_FIXED_FIELDS`].
pub(crate) const ATOMS_POPC_FIXED: u128 = mask_of(&ATOMS_POPC_FIXED_FIELDS);

/// `ATOMS.POPC.INC.32 [addr]`: adds one to a shared word for every active thread (PR #4588
/// alone; needs an `nvdisasm` check).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpAtomsPopc {
    /// The address register; `RZ` for none.
    pub addr: Gpr,
    /// The scale of the address register.
    pub stride: OffsetStride,
}

impl Named for OpAtomsPopc {
    /// PR #4588's row name, kept whole: `atoms` alone names the shared forms of [`OpAtom`].
    const MNEMONIC: &'static str = "atoms.popc";
}

impl Display for OpAtomsPopc {
    /// PR #4588's name for the instruction, `ATOMS.POPC.INC.32`, and the shared address (a
    /// guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "atoms.popc.inc.32 [{}{}]", self.addr, self.stride)
    }
}

fn decode_atoms_popc(reader: &mut Reader) -> Op {
    // Fixed bits of the form: the decoder has checked them.
    for field in ATOMS_POPC_FIXED_FIELDS {
        reader.field(field);
    }
    Op::AtomsPopc(OpAtomsPopc {
        addr: reader.gpr(MEM_ADDR),
        stride: stride(reader),
    })
}

// ---------------------------------------------------------------------------------------------
// CCTL, MEMBAR, ERRBAR and LDGDEPBAR.

/// `CCTL` (NAK `OpCCtl`: `set_opcode(0x98f)`).
pub(crate) const CCTL_OPCODE: u16 = 0x98f;
/// `CCTL`'s 32-bit address offset (`set_field(32..64, self.addr_offset)`, an `i32`).
pub(crate) const CCTL_OFFSET: Field = Field::new(32, 32);
/// `CCTL`'s operation (`set_field(87..91, ..)`).
pub(crate) const CCTL_OP: Field = Field::new(87, 4);
/// `MEMBAR` (NAK `OpMemBar`: `set_opcode(0x992)`).
pub(crate) const MEMBAR_OPCODE: u16 = 0x992;
/// `MEMBAR`'s `.MMIO` bit (`set_bit(72, false); // !.MMIO`).
pub(crate) const MEMBAR_MMIO: Field = Field::bit(72);
/// `MEMBAR`'s scope (`set_field(76..79, ..)`).
pub(crate) const MEMBAR_SCOPE: Field = Field::new(76, 3);
/// `MEMBAR`'s `.SC` bit (`set_bit(80, false); // .SC`).
pub(crate) const MEMBAR_SC: Field = Field::bit(80);
/// `ERRBAR` (PR #4588 alone: row `0x9ab`, no fields; `Visitor::Run`: "kept as a device memory
/// barrier").
pub(crate) const ERRBAR_OPCODE: u16 = 0x9ab;
/// `LDGDEPBAR` (PR #4588 alone: row `0x9af`, no fields; `Visitor::Run` lowers it to a workgroup
/// memory barrier).
pub(crate) const LDGDEPBAR_OPCODE: u16 = 0x9af;

coded! {
    /// A cache operation (NAK's `CCtlOp`, the values its SM 7.0 encoder writes).
    pub enum CCtlOp {
        /// 0: prefetch to L1.
        Pf1 = 0,
        /// 1: prefetch to L2.
        Pf2 = 1,
        /// 2: write back.
        Wb = 2,
        /// 3: invalidate.
        Iv = 3,
        /// 4: invalidate every line.
        IvAll = 4,
        /// 5: `RS`.
        Rs = 5,
        /// 6: `IVALLP`.
        IvAllP = 6,
        /// 7: write back every line.
        WbAll = 7,
        /// 8: `WBALLP`.
        WbAllP = 8,
    }
}

impl CCtlOp {
    /// Whether the operation applies to every line rather than to an address (NAK's
    /// `CCtlOp::is_all`).
    pub const fn is_all(self) -> bool {
        matches!(
            self,
            Self::IvAll | Self::IvAllP | Self::WbAll | Self::WbAllP
        )
    }
}

/// `CCTL [addr+offset]`: a cache operation on global memory (NAK's `OpCCtl`, which encodes
/// only the global space).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCCtl {
    /// The operation.
    pub op: CCtlOp,
    /// The address register; `RZ` for none.
    pub addr: Gpr,
    /// The byte offset ([`CCTL_OFFSET`]).
    pub offset: i32,
}

impl Named for OpCCtl {
    const MNEMONIC: &'static str = "cctl";
}

impl Display for CCtlOp {
    /// NAK's `CCtlOp` printer (lower-case names).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pf1 => "pf1",
            Self::Pf2 => "pf2",
            Self::Wb => "wb",
            Self::Iv => "iv",
            Self::IvAll => "ivall",
            Self::Rs => "rs",
            Self::IvAllP => "ivallp",
            Self::WbAll => "wball",
            Self::WbAllP => "wballp",
        })
    }
}

impl Display for OpCCtl {
    /// `cctl.iv [r2+0x10];`, the address left out for an operation on every line when it is
    /// `[rz]` (a guess; NAK's printer writes the space, `.global`, instead of the operation).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "cctl.{}", self.op)?;
        if !self.op.is_all() || !self.addr.is_zero() || self.offset != 0 {
            write!(f, " [{}", self.addr)?;
            offset_text(f, i64::from(self.offset))?;
            f.write_str("]")?;
        }
        Ok(())
    }
}

fn decode_cctl(reader: &mut Reader) -> Result<Op, DecodeError> {
    let addr = reader.gpr(MEM_ADDR);
    // Cannot fail: the field is 32 bits wide.
    let offset = i32::try_from(reader.signed(CCTL_OFFSET)).unwrap_or(0);
    let op = read_coded(reader, CCTL_OP, CCtlOp::from_bits)?;
    Ok(Op::CCtl(OpCCtl { op, addr, offset }))
}

coded! {
    /// `MEMBAR`'s scope (module docs: NAK's values and PR #4588's 5).
    pub enum MemBarScope {
        /// 0: the workgroup (NAK `MemScope::CTA`).
        Cta = 0,
        /// 1: the SM (NAK's comment `// SM => 1_u8`; NAK does not encode it).
        Sm = 1,
        /// 2: the GPU.
        Gpu = 2,
        /// 3: the system.
        System = 3,
        /// 5: a scope PR #4588 accepts and lowers to a device-wide barrier; its name is not
        /// known.
        Scope5 = 5,
    }
}

/// `MEMBAR`: orders memory accesses at a scope (NAK's `OpMemBar`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpMemBar {
    /// The scope.
    pub scope: MemBarScope,
    /// `.MMIO` (NAK writes it clear).
    pub mmio: bool,
    /// `.SC` (NAK writes it clear).
    pub sc: bool,
}

impl Named for OpMemBar {
    const MNEMONIC: &'static str = "membar";
}

impl Display for MemBarScope {
    /// NAK's `MemScope` printer, `.sm` and `.scope5` (guesses).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cta => ".cta",
            Self::Sm => ".sm",
            Self::Gpu => ".gpu",
            Self::System => ".sys",
            Self::Scope5 => ".scope5",
        })
    }
}

impl Display for OpMemBar {
    /// `membar.sc.cta;` (a guess: NAK's printer writes `.sc` always).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("membar")?;
        flag(f, self.sc, ".sc")?;
        flag(f, self.mmio, ".mmio")?;
        write!(f, "{}", self.scope)
    }
}

fn decode_membar(reader: &mut Reader) -> Result<Op, DecodeError> {
    let mmio = reader.flag(MEMBAR_MMIO);
    let scope = read_coded(reader, MEMBAR_SCOPE, MemBarScope::from_bits)?;
    Ok(Op::MemBar(OpMemBar {
        scope,
        mmio,
        sc: reader.flag(MEMBAR_SC),
    }))
}

/// `ERRBAR`: waits for errors of earlier memory accesses to be reported (PR #4588 alone; needs
/// an `nvdisasm` check).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OpErrBar {}

/// `LDGDEPBAR`: marks the end of a group of `LDGSTS` copies (PR #4588 alone; needs an
/// `nvdisasm` check).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OpLdgDepBar {}

impl Named for OpErrBar {
    const MNEMONIC: &'static str = "errbar";
}

impl Named for OpLdgDepBar {
    const MNEMONIC: &'static str = "ldgdepbar";
}

impl Display for OpErrBar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("errbar")
    }
}

impl Display for OpLdgDepBar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ldgdepbar")
    }
}

// ---------------------------------------------------------------------------------------------
// LDGSTS.

/// `LDGSTS` (PR #4588 alone: row `0xfae` and `Visitor::LDGSTS`).
pub(crate) const LDGSTS_OPCODE: u16 = 0xfae;
/// The shared address register (`shared_reg = Take(16, 8)`).
pub(crate) const LDGSTS_SHARED: Field = Field::new(16, Gpr::BITS);
/// The global address register, a pair (`global_reg = Take(24, 8)`).
pub(crate) const LDGSTS_GLOBAL: Field = Field::new(24, Gpr::BITS);
/// The global offset, a signed 12-bit byte count (`global_immediate = Take(32, 12)`,
/// sign-extended from bit 11).
pub(crate) const LDGSTS_GLOBAL_OFFSET: Field = Field::new(32, 12);
/// The shared offset, a signed 16-bit count of [`LDGSTS_SHARED_UNIT`] (`static_cast<s16>(Take(48,
/// 16)) * 16`).
pub(crate) const LDGSTS_SHARED_OFFSET: Field = Field::new(48, 16);
/// The bytes in one unit of [`LDGSTS_SHARED_OFFSET`].
pub(crate) const LDGSTS_SHARED_UNIT: i32 = 16;
/// The "memory descriptor selector" (`Ignore(64, 6)`).
pub(crate) const LDGSTS_DESCRIPTOR: Field = Field::new(64, 6);
/// The enable predicate's negation (`negate_enable = Take(70, 1)`); the register is [`COND`]
/// (`enable_index = Take(87, 3)`).
pub(crate) const LDGSTS_ENABLE_NOT: Field = Field::bit(70);
/// The fields every `LDGSTS` has all set, whose meaning no reference gives: `Take(74, 3) == 7`,
/// `Take(84, 1) == 1`, `Take(90, 2) == 3`.
pub(crate) const LDGSTS_FIXED_FIELDS: [Field; 3] =
    [Field::new(74, 3), Field::bit(84), Field::new(90, 2)];
/// The bits of [`LDGSTS_FIXED_FIELDS`].
pub(crate) const LDGSTS_FIXED: u128 = mask_of(&LDGSTS_FIXED_FIELDS);

/// `LDGSTS.E.BYPASS.128 [shared], [global]`: copies 16 bytes from global to shared memory (PR
/// #4588 alone; needs an `nvdisasm` check).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLdGsts {
    /// The shared address register.
    pub shared: Gpr,
    /// The shared offset in units of [`LDGSTS_SHARED_UNIT`] bytes.
    pub shared_offset: i16,
    /// The global address register pair.
    pub global: Gpr,
    /// The global byte offset, a signed 12-bit number.
    pub global_offset: i16,
    /// The memory descriptor selector, six bits (PR #4588 ignores it).
    pub descriptor: u8,
    /// The predicate that enables the copy; where it is false the destination is zero-filled
    /// (PR #4588).
    pub enable: PredSrc,
}

impl Named for OpLdGsts {
    const MNEMONIC: &'static str = "ldgsts";
}

impl Display for OpLdGsts {
    /// PR #4588's name for the form, `LDGSTS.E.BYPASS.128`, the two addresses, the descriptor
    /// and the enable predicate when not `PT` (guesses).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ldgsts.e.bypass.128 [{}", self.shared)?;
        offset_text(
            f,
            i64::from(self.shared_offset).saturating_mul(i64::from(LDGSTS_SHARED_UNIT)),
        )?;
        write!(f, "], [{}.64", self.global)?;
        offset_text(f, i64::from(self.global_offset))?;
        write!(f, "], {:#x}", self.descriptor)?;
        if !self.enable.is_true() {
            write!(f, ", {}", self.enable)?;
        }
        Ok(())
    }
}

fn decode_ldgsts(reader: &mut Reader) -> Op {
    // Fixed bits of the form: the decoder has checked them.
    for field in LDGSTS_FIXED_FIELDS {
        reader.field(field);
    }
    Op::LdGsts(OpLdGsts {
        shared: reader.gpr(LDGSTS_SHARED),
        // Cannot fail: the fields are 16 and 12 bits wide.
        shared_offset: i16::try_from(reader.signed(LDGSTS_SHARED_OFFSET)).unwrap_or(0),
        global: reader.gpr(LDGSTS_GLOBAL),
        global_offset: i16::try_from(reader.signed(LDGSTS_GLOBAL_OFFSET)).unwrap_or(0),
        descriptor: reader.byte(LDGSTS_DESCRIPTOR),
        enable: reader.pred_src(COND, LDGSTS_ENABLE_NOT),
    })
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// The memory group's forms.
pub(crate) const FORMS: &[Form] = &[
    ld_form(LDG_OPCODE, Space::Global, GLOBAL, |reader| {
        decode_ld(reader, Space::Global, GLOBAL)
    }),
    ld_form(LDG_UNIFORM_OPCODE, Space::Global, LDG_UNIFORM, |reader| {
        decode_ld(reader, Space::Global, LDG_UNIFORM)
    }),
    ld_form(LDL_OPCODE, Space::Local, LD_LOCAL_SHARED, |reader| {
        decode_ld(reader, Space::Local, LD_LOCAL_SHARED)
    }),
    ld_form(LDS_OPCODE, Space::Shared, LD_LOCAL_SHARED, |reader| {
        decode_ld(reader, Space::Shared, LD_LOCAL_SHARED)
    }),
    st_form(STG_OPCODE, Space::Global, GLOBAL, |reader| {
        decode_st(reader, Space::Global, GLOBAL)
    }),
    st_form(
        STG_UNIFORM_OPCODE,
        Space::Global,
        ST_GLOBAL_UNIFORM,
        |reader| decode_st(reader, Space::Global, ST_GLOBAL_UNIFORM),
    ),
    st_form(STL_OPCODE, Space::Local, AddrLayout::PLAIN, |reader| {
        decode_st(reader, Space::Local, AddrLayout::PLAIN)
    }),
    st_form(
        STL_UNIFORM_OPCODE,
        Space::Local,
        ST_UNIFORM_PLAIN,
        |reader| decode_st(reader, Space::Local, ST_UNIFORM_PLAIN),
    ),
    st_form(STS_OPCODE, Space::Shared, AddrLayout::PLAIN, |reader| {
        decode_st(reader, Space::Shared, AddrLayout::PLAIN)
    }),
    st_form(
        STS_UNIFORM_OPCODE,
        Space::Shared,
        ST_UNIFORM_PLAIN,
        |reader| decode_st(reader, Space::Shared, ST_UNIFORM_PLAIN),
    ),
    Form::new(
        LDC_OPCODE,
        mask_of(&[DST, LDC_OFFSET, CBUF_OFFSET, CBUF_INDEX, MEM_TYPE, LDC_MODE]),
        decode_ldc,
    ),
    Form::new(
        LDC_BINDLESS_OPCODE,
        mask_of(&[DST, LDC_HANDLE, CBUF_OFFSET, LDC_BINDLESS_OFFSET, MEM_TYPE]),
        decode_ldc_bindless,
    )
    .with_fixed(UNIFORM_BIT.mask()),
    atom_form(RED_OPCODE, AtomForm::Red, ST_GLOBAL_UNIFORM, |reader| {
        decode_atom(reader, AtomForm::Red, ST_GLOBAL_UNIFORM)
    }),
    atom_form(ATOMG_OPCODE, AtomForm::Global, GLOBAL, |reader| {
        decode_atom(reader, AtomForm::Global, GLOBAL)
    }),
    atom_form(
        ATOMG_UNIFORM_OPCODE,
        AtomForm::Global,
        ATOMG_UNIFORM,
        |reader| decode_atom(reader, AtomForm::Global, ATOMG_UNIFORM),
    ),
    atom_form(ATOMG_CAS_OPCODE, AtomForm::GlobalCas, GLOBAL, |reader| {
        decode_atom(reader, AtomForm::GlobalCas, GLOBAL)
    }),
    atom_form(
        ATOMS_OPCODE,
        AtomForm::Shared,
        AddrLayout::PLAIN,
        |reader| decode_atom(reader, AtomForm::Shared, AddrLayout::PLAIN),
    ),
    atom_form(
        ATOMS_UNIFORM_OPCODE,
        AtomForm::Shared,
        ST_UNIFORM_PLAIN,
        |reader| decode_atom(reader, AtomForm::Shared, ST_UNIFORM_PLAIN),
    ),
    atom_form(
        ATOMS_CAS_OPCODE,
        AtomForm::SharedCas,
        AddrLayout::PLAIN,
        |reader| decode_atom(reader, AtomForm::SharedCas, AddrLayout::PLAIN),
    ),
    Form::new(
        ATOMS_POPC_OPCODE,
        mask_of(&[MEM_ADDR, MEM_STRIDE]),
        |reader| Ok(decode_atoms_popc(reader)),
    )
    .with_fixed(ATOMS_POPC_FIXED),
    Form::new(
        CCTL_OPCODE,
        mask_of(&[MEM_ADDR, CCTL_OFFSET, CCTL_OP]),
        decode_cctl,
    ),
    Form::new(
        MEMBAR_OPCODE,
        mask_of(&[MEMBAR_MMIO, MEMBAR_SCOPE, MEMBAR_SC]),
        decode_membar,
    ),
    Form::new(ERRBAR_OPCODE, 0, |_| Ok(Op::ErrBar(OpErrBar {}))),
    Form::new(LDGDEPBAR_OPCODE, 0, |_| Ok(Op::LdgDepBar(OpLdgDepBar {}))),
    Form::new(
        LDGSTS_OPCODE,
        mask_of(&[
            LDGSTS_SHARED,
            LDGSTS_GLOBAL,
            LDGSTS_GLOBAL_OFFSET,
            LDGSTS_SHARED_OFFSET,
            LDGSTS_DESCRIPTOR,
            LDGSTS_ENABLE_NOT,
            COND,
        ]),
        |reader| Ok(decode_ldgsts(reader)),
    )
    .with_fixed(LDGSTS_FIXED),
];

/// The bit 91 the forms without a uniform register reject (module docs).
#[cfg(test)]
pub(crate) const NO_UNIFORM_BIT: u128 = UNIFORM_BIT.mask();
/// The bits `ATOMG` `0x3a8` rejects: its row is the uniform form's (module docs).
#[cfg(test)]
pub(crate) const ATOMG_DROPPED: u128 = mask_of(&[ST_UNIFORM, ATOMG_REG_64, UNIFORM_BIT]);
/// The bits `LDGSTS` rejects (module docs).
#[cfg(test)]
pub(crate) const LDGSTS_DROPPED: u128 = mask_of(&[
    Field::new(44, 4),
    Field::bit(71),
    Field::new(72, 2),
    Field::new(77, 7),
    Field::new(85, 2),
]);
/// The bits `ATOMS.POPC` rejects (module docs): the data and offset, the type, and the clear
/// bits of the operation 10.
#[cfg(test)]
pub(crate) const ATOMS_POPC_DROPPED: u128 =
    Field::new(32, 32).mask() | ATOM_TYPE.mask() | (ATOM_OP.mask() & !ATOMS_POPC_FIXED);

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x381, [0xffff_ffff, 0xffff_ff00, 0x087f_ff00, 0xffff_fe00])
        .dropping(NO_UNIFORM_BIT),
    ReferenceRow::new(0x981, [0xffff_ffff, 0xffff_ff3f, 0x0c7f_ff0f, 0xffff_fe00]),
    ReferenceRow::new(0x983, [0xffff_ffff, 0xffff_ff3f, 0x0870_0e00, 0xffff_fe00]),
    ReferenceRow::new(0x984, [0xffff_ffff, 0xffff_ff3f, 0x0880_ce00, 0xffff_fe00]),
    ReferenceRow::new(0x386, [0xff00_ffff, 0xffff_ffff, 0x0871_ef00, 0xffff_fe00])
        .dropping(NO_UNIFORM_BIT),
    ReferenceRow::new(0x986, [0xff00_ffff, 0xffff_ffff, 0x0c71_ff3f, 0xffff_fe00]),
    ReferenceRow::new(0x387, [0xff00_ffff, 0xffff_ffff, 0x0870_0e00, 0xffff_fe00])
        .dropping(NO_UNIFORM_BIT),
    ReferenceRow::new(0x987, [0xff00_ffff, 0xffff_ffff, 0x0870_0e3f, 0xffff_fe00]),
    ReferenceRow::new(0x388, [0xff00_ffff, 0xffff_ffff, 0x0800_ce00, 0xffff_fe00])
        .dropping(NO_UNIFORM_BIT),
    ReferenceRow::new(0x988, [0xff00_ffff, 0xffff_ffff, 0x0800_ce3f, 0xffff_fe00]),
    ReferenceRow::new(0xb82, [0xffff_ffff, 0x07ff_ffc0, 0x0800_ce00, 0xffff_fe00])
        .dropping(NO_UNIFORM_BIT),
    ReferenceRow::new(0x582, [0x3fff_ffff, 0x003f_ffc0, 0x0800_0eff, 0xffff_fe00])
        .fixing(UNIFORM_BIT.mask()),
    ReferenceRow::new(0x98e, [0xffff_ffff, 0xffff_ffff, 0x0ff1_ef3f, 0xffff_fe00]),
    ReferenceRow::new(0x3a8, [0xffff_ffff, 0xffff_ffff, 0x0fff_ef7f, 0xffff_fe00])
        .dropping(ATOMG_DROPPED),
    ReferenceRow::new(0x9a8, [0xffff_ffff, 0xffff_ffff, 0x0fff_ef7f, 0xffff_fe00]),
    ReferenceRow::new(0x3a9, [0xffff_ffff, 0xffff_ffff, 0x007f_efff, 0xffff_fe00]),
    ReferenceRow::new(0x38c, [0xffff_ffff, 0xffff_ffff, 0x0780_ce00, 0xffff_fe00]),
    ReferenceRow::new(0x98c, [0xffff_ffff, 0xffff_ffff, 0x0f80_ce3f, 0xffff_fe00]),
    ReferenceRow::new(0x38d, [0xffff_ffff, 0xffff_ffff, 0x0000_ceff, 0xffff_fe00]),
    ReferenceRow::new(0xf8c, [0xffff_ffff, 0xffff_ffff, 0x0f80_ce3f, 0xffff_fe00])
        .dropping(ATOMS_POPC_DROPPED)
        .fixing(ATOMS_POPC_FIXED),
    ReferenceRow::new(0x98f, [0xff00_ffff, 0xffff_ffff, 0x0780_0000, 0xffff_fe00]),
    ReferenceRow::new(0x992, [0x0000_ffff, 0x0000_0000, 0x0001_7100, 0xffff_fe00]),
    ReferenceRow::new(0x9ab, [0x0000_ffff, 0x0000_0000, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0x9af, [0x0000_ffff, 0x0000_0000, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0xfae, [0xffff_ffff, 0xffff_ffff, 0x0fff_ffff, 0xffff_fe00])
        .dropping(LDGSTS_DROPPED)
        .fixing(LDGSTS_FIXED),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::all()
}

shells! {}
