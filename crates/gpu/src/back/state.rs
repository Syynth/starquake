//! The back thread's view of engine state: GPU protocol slice G10
//! (`docs/specs/gpu-protocol.md`, "Mirrors and dirty flags", with what "3D: from state to a
//! draw" and "Compute: QMD 3.0" will ask of it).
//!
//! The back keeps, per channel and class, a [`Mirror`] of the front's register file, written only
//! from the ring's `Write` and `Run` entries ([`Mirror::apply`]), never by the back itself
//! (owner's decision 1, "The GPU front thread owns the register file"), and a dirty bit per
//! **state group** ([`Group`]). A write marks its groups; deriving a draw (slice G18) clears
//! them ([`Mirror::clear`], [`Mirror::take_dirty`]). Typed views read fields out of the mirror
//! with bounds and enum checks ([`Mirror::three_d`]: `three_d.viewport(i)?.scale_x()`).
//!
//! - `group` owns [`GroupKind`], [`Group`], [`GroupSet`] and the array lengths ([`VIEWPORTS`],
//!   [`COLOR_TARGETS`], ...), shared by the group tables and the views.
//! - `table` owns the group tables ([`GROUPS_3D`], [`GROUPS_COMPUTE`]) and the per-method lookup
//!   built from them once per class ([`GroupMap`]).
//! - `mirror` owns [`Mirror`], [`StateError`], [`ConstantBufferBinding`], [`RunPositions`] and
//!   [`Reg`] (one element of a method, read through the generated tables).
//! - `views` owns the typed views: [`ThreeD`] with one view per group ([`Viewport`],
//!   [`ColorTarget`], [`DepthStencil`], ...) and [`Compute`].
//!
//! # Tables and generation
//!
//! The generated class tables ([`crate::classes`], decision 6) give methods, fields and named
//! values but no group map and no array lengths. The group map is a hand-written table keyed on
//! the generated constants (the issue's second option; `cargo xtask gen-classes` is unchanged),
//! and the array lengths are Mesa's (`METHOD_ARRAY_SIZES` in
//! `mesa-nouveau/src/nouveau/headers/class_parser.py` @ 52238762, MIT), the same the front's
//! state table uses. The accessors read through the generated tables: a getter names a generated
//! [`crate::classes::Method`] and [`crate::field::Field`], and [`Reg`] takes the element's
//! address from the method's strides, the bits from the field, and the valid values of a named
//! field from its generated list. No bit position, stride or enum value is written by hand here.
//!
//! **A departure from the approved spec, for the owner** (`docs/specs/gpu-protocol.md`, "Mirrors
//! and dirty flags"). The spec says "A generated table maps each method to its groups" and that
//! the typed accessors "are generated from the same tables (decision 6)". Here neither is
//! generated: the group table is written by hand (keyed on the generated method constants), and
//! the getters are hand-listed macro lines that name generated `Method` and `Field` constants.
//! Nothing `cargo xtask gen-classes` emits says which methods form a group or which fields a
//! getter returns. Owner question: accept a hand-written table and getter list keyed on the
//! generated constants, or extend `cargo xtask gen-classes` to emit both. Until then the tests
//! stand in for generation: `every_register_a_view_reads_marks_that_views_group` checks every
//! register a getter reads against the group table, and `eden_tracked_registers_mark_a_group`
//! checks the table against Eden's.
//!
//! # A write
//!
//! [`Mirror::run`] (and [`Mirror::write`], one word) stores each word in the register, so the
//! register keeps the last; records a constant buffer binding for each word written to
//! `BIND_GROUP_CONSTANT_BUFFER(j)` (the selected buffer, as Eden-NX2's `ProcessCBBind` does;
//! the register alone could not say what was bound once the selector moves on); and then marks
//! the method's groups, or, for a method in no group, counts its words
//! ([`Mirror::ungrouped_words`]). A method's groups are marked whatever the value, as Eden's
//! `ProcessDirtyRegisters` does: the front has already dropped the same-value state writes.
//! A method outside the class's register file is [`StateError::OutOfRange`] and changes nothing.
//!
//! The mirror never applies a side-effect register: the front computes it and sends it as a
//! `Write` after the run (`regs` module docs, "Runs"). Where a run's words go is derived from the
//! side-effect register's value before the run, through the front's own rule
//! ([`crate::regs::SideEffectRule::apply`]): [`Mirror::run_positions`], called before the
//! following `Write`s are applied. For a run of constant buffer words these are the byte offsets
//! each word is written at, across chunks of [`crate::ring::MAX_RUN_WORDS`] and through the
//! offset field's wrap (`a_multi_chunk_run_that_wraps_the_offset_lands_where_the_front_put_it`).
//!
//! # Starting state
//!
//! A new mirror holds the class's starting values, [`crate::regs::ClassRules::initial_values`],
//! the one place both the front's register file (and its shadow file) and the mirror start from,
//! so the front never drops a same-value write the back has not seen
//! (`a_new_mirror_starts_from_the_same_values_as_the_front_file`). They are all zero today:
//! Eden's non-zero 3D register defaults (`InitializeRegisterDefaults`) are not seeded yet, and
//! seeding them there changes both sides at once. Every group the class's table names starts
//! dirty, as Eden's 3D engine starts with every dirty flag set (`dirty.flags.flip()` in
//! `Maxwell3D`'s constructor, `eden-nx2/src/video_core/engines/maxwell_3d.cpp`), so the first
//! draw derives everything. No constant buffer is bound.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - The dirty-flag model: `eden/src/video_core/dirty_flags.{h,cpp}` and
//!   `eden/src/video_core/renderer_vulkan/vk_state_tracker.cpp` @ cf654052 (GPL-3.0-or-later and
//!   GPL-2.0-or-later), used as a model for which registers belong together, not ported.
//! - Methods, fields and values: the generated tables of `classes/clc797.h` and `clc7c0.h`
//!   (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, Copyright (c) 2020, NVIDIA
//!   CORPORATION, MIT).
//! - The bind, the halves of the vertex stream and attribute arrays and the native vertex
//!   stream sizes: `eden-nx2/src/video_core/engines/maxwell_3d.{h,cpp}` and
//!   `eden-nx2/src/video_core/dirty_flags.cpp` @ 50fa86bb (GPL-3.0-or-later).
//!
//! # Divergences and guesses
//!
//! Each is pinned by the test named, in `state/tests.rs`.
//!
//! - **Sixteen bind groups, not five.** Eden-NX2 binds constant buffers for five stages
//!   (`MaxShaderStage`, `ProcessCBBind(0..4)`); Mesa gives `BIND_GROUP_CONSTANT_BUFFER` sixteen
//!   elements. The owner's rule for references that disagree (gpu-protocol decision 10: where one
//!   answer contains the other, the larger) keeps sixteen, so a bind to group 5 to 15 is recorded
//!   here and ignored there. `bind_groups_are_told_apart`.
//! - **Thirty-two constant buffer slots, not eighteen.** Eden-NX2 keeps 18 slots per stage
//!   (`MaxConstBuffers`) and drops a bind to a slot above them ("slot field is 5 bits; the state
//!   array holds 18"); here every value of the 5-bit `SHADER_SLOT` field has a slot, by the same
//!   rule. `a_bind_records_the_selected_buffer_in_its_group_and_slot` (slot 31).
//! - **Thirty-two vertex stream sizes, not sixteen.** Mesa gives `SET_VERTEX_STREAM_SIZE_A` and
//!   `_B` sixteen elements; Eden-NX2 in native mode reads a size for each of its 32 streams at
//!   method `0x600 / 4 + index * 2` (`NativeVertexStreamSize` in `maxwell_3d.h`, used by
//!   `buffer_cache.h`'s `UpdateVertexBuffer` and by `maxwell_3d.cpp`) and marks each pair dirty
//!   (`FillBlock(tables, 0x600 / sizeof(u32) + i * 2, 2, VertexBuffer0 + i, ...)` for `i` below
//!   32 in `eden-nx2/src/video_core/dirty_flags.cpp`; `eden-pr4588` has the same). By decision
//!   10's "the larger" rule the sizes are one array of 32 ([`VERTEX_STREAM_SIZES`]), and
//!   [`VertexStream::size`] reads element `j` for stream `j`. The front's `STATE_3D` still gives
//!   these methods Mesa's 16, which only makes writes to sizes 16 to 31 always-sent triggers
//!   there. `groups_by_method_number_from_the_header_text` (methods `0x1a0` and `0x1bf`),
//!   `every_vertex_stream_method_marks_the_stream_of_its_half` and
//!   `stream_attribute_and_binding_accessors_are_bounded_at_their_exact_edge` (streams 31 and 32).
//! - **The bind masks the selector to the header's fields.** Eden-NX2's `ProcessCBBind` takes
//!   the whole `const_buffer.size` word and the whole `address_high` word
//!   (`ConstantBuffer::Address` in `maxwell_3d.h`); here the size is the selector's `SIZE` field
//!   (16:0) and the upper address its `ADDRESS_UPPER` field (7:0), as `clc797.h` defines them,
//!   so bits above those fields are dropped. Decision 10 lets the header win.
//!   `a_bind_records_the_selected_buffer_in_its_group_and_slot`.
//! - **Groups are coarser than Eden's Vulkan tracker** in places: one [`GroupKind::Rasterizer`]
//!   and one [`GroupKind::DepthStencil`] where Eden keeps a flag per dynamic state (cull mode,
//!   depth compare, stencil reference, ...), and per-target blend groups where Eden has one
//!   `Blending` flag. Marking more than Eden costs a re-derivation. No change is missed for
//!   exactly these registers: every register Eden's dirty-flag tables mark that `clc797.h`
//!   defines (`eden_tracked_registers_mark_a_group`), and every register a typed view's getter
//!   reads (`every_register_a_view_reads_marks_that_views_group`). A register in neither set
//!   marks nothing and is counted ([`Mirror::ungrouped_words`]), so a later getter for one needs
//!   a table line too. `groups_by_method_number_from_the_header_text`.
//! - **Eden marks two blocks `clc797.h` does not define**, and they mark nothing here: the
//!   legacy vertex stream limits (`vertex_stream_limits`, byte 0x1F00, two words per stream) and
//!   the index buffer's limit address (`index_buffer.limit_addr_high` and `_low`, bytes 0x17D0
//!   and 0x17D4). Ampere takes sizes instead (`SET_VERTEX_STREAM_SIZE_A/B`,
//!   `SET_INDEX_BUFFER_SIZE_A/B`). Eden's padding words inside a block (the fourth word of each
//!   scissor, words 10 to 15 of each render target, ...) are no method here either.
//!   `eden_registers_clc797_does_not_define_mark_nothing`.
//! - **The spec's `SET_PROGRAM_REGION` is not in `clc797.h`**: Ampere's pipelines take a 64-bit
//!   program address each (`SET_PIPELINE_PROGRAM_ADDRESS_A` and `_B`), which
//!   [`Pipeline::program_address`] reads. `index_buffer_pools_and_pipeline_getters`.
//!
//! Not here: the root tables for constant buffer banks 24 to 31, deriving anything from the
//! groups (G18), routing ring entries to mirrors by channel and class (the executor, G19), and
//! register defaults (above).

mod group;
mod mirror;
mod table;
mod views;

#[cfg(test)]
mod tests;

pub use group::{
    BIND_GROUPS, BLEND_TARGETS, COLOR_TARGETS, CONSTANT_BUFFER_SLOTS, GROUP_COUNT, Group,
    GroupKind, GroupSet, PIPELINES, SCISSORS, VERTEX_ATTRIBUTES, VERTEX_ATTRIBUTES_PER_HALF,
    VERTEX_STREAM_SIZES, VERTEX_STREAMS, VERTEX_STREAMS_PER_HALF, VIEWPORTS,
};
pub use mirror::{ConstantBufferBinding, Mirror, Reg, RunPositions, StateError};
pub use table::{GROUPS_3D, GROUPS_COMPUTE, GroupEntry, GroupMap, Marks, group_table};
pub use views::{
    BindGroup, Blend, BlendPerTarget, ColorTarget, Compute, CtSelect, DepthStencil, IndexBuffer,
    Pipeline, Rasterizer, Scissor, TexPools, ThreeD, VertexAttribute, VertexStream, Viewport, Zt,
    arrays,
};
