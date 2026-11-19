//! Which methods start a draw, and with what counts: a port of Eden-NX2's `DrawManager`
//! (`eden-nx2/src/video_core/engines/draw_manager.cpp` @ 50fa86bb, GPL-3.0-or-later) for the
//! methods this slice handles.
//!
//! [`DrawCalls::method`] takes one word written to a 3D method, after the mirror has stored it,
//! and returns the [`DrawCall`]s it starts. The draw registers it reads (the vertex array's start
//! and count, the index buffer's first and count, the global base vertex and instance) come from
//! the mirror at the moment of the draw, as Eden reads `maxwell3d.regs` in `DrawEnd`.
//!
//! # The methods
//!
//! Exactly these methods change the state here; every other method is ignored
//! ([`DRAW_METHODS`] lists them, and `tests::only_the_listed_methods_are_seen` pins the set):
//!
//! - `BEGIN` (Eden's `draw.begin`, `DrawBegin`): `INSTANCE_ID` `FIRST` draws the deferred
//!   instances ([`DrawCalls::flush`]) and starts a general draw; `SUBSEQUENT` counts one more
//!   instance and defers; `UNCHANGED` changes neither. The topology is `OP`.
//! - `END` (`draw.end`, `DrawEnd`): in a general draw, one draw of one instance from the vertex
//!   array or, when `DRAW_INDEX_BUFFER` was written since the last draw, from the index buffer; in
//!   an instanced draw nothing (the instances wait for [`DrawCalls::flush`]).
//! - `DRAW_INDEX_BUFFER` (`index_buffer.count`): the next `END` draws indexed.
//! - `DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST` and `_SUBSEQUENT` (`DrawArrayInstanced`): one
//!   draw of one instance, its first instance the instance counter.
//! - `DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST` and `_SUBSEQUENT`, and the `16` and `8`
//!   forms (`DrawIndexSmall`): one indexed draw of one instance.
//! - `DRAW_INDEX_BUFFER_BEGIN_END_B` and `DRAW_VERTEX_ARRAY_BEGIN_END_B`: one draw, with `_A`'s
//!   first and `_B`'s count, the topology and instance count of `SET_DRAW_CONTROL_A` and `_B`.
//!   No reference handles these methods (Eden-NX2 and PR #4588 have no case for them); that
//!   writing `_B` starts the draw, and what it takes from `SET_DRAW_CONTROL_A/B`, follows only
//!   the header's names and is a **guess**, confirmed by a trace of a title using them.
//!
//! Not handled, and reported as [`Started::Unhandled`] when written: the inline index forms
//! (`DRAW_INLINE_INDEX`, `DRAW_INLINE_INDEX2X16`, `DRAW_INLINE_INDEX4X8`), `DRAW_ZERO_INDEX` and
//! `DRAW_AUTO`. Indirect draws reach the 3D class through macros, not here.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `draw/tests.rs`.
//!
//! - **The small index forms take their index size from the method.** Eden's `DrawIndexSmall`
//!   keeps `SET_INDEX_BUFFER_E`'s size whatever the method; the header names the size in the
//!   method (`DRAW_INDEX_BUFFER32_...`, `16`, `8`), and decision 10 lets the header settle a
//!   field's meaning. `the_small_index_forms_take_their_size_from_the_method`.
//! - **The small index forms count instances.** Eden's `index_buffer*_subsequent` case counts an
//!   instance and then draws with the global base instance, so every instance draws as instance
//!   0. Here they count instances as the vertex array forms do (first instance the counter), as
//!   the header's `INSTANCE_SUBSEQUENT` says. `the_small_index_forms_count_instances`.
//! - **The instanced forms leave the begin/end state alone.** Eden's `DrawArrayInstanced` also
//!   sets the general draw mode to instanced, so a later `DrawDeferred` draws the begin/end
//!   registers again with the instanced form's count. Here the instanced forms use their own
//!   counter and never defer. `an_instanced_form_does_not_defer_a_begin_end_draw`.
//! - **Topology override legacy values are refused.** Eden's `UpdateTopology` casts a legacy
//!   `SET_PRIMITIVE_TOPOLOGY` value (`LEGACY_*`, 0x1001 and up) to a topology that does not
//!   exist; here the draw reports [`super::DrawError::Topology`]. That is in `fixed`.

use crate::back::state::{Mirror, StateError};
use crate::classes::{Method, clc797};

/// The methods whose words [`DrawCalls::method`] looks at, beside [`UNHANDLED_METHODS`].
pub const DRAW_METHODS: [&Method; 13] = [
    &clc797::BEGIN,
    &clc797::END,
    &clc797::DRAW_INDEX_BUFFER,
    &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST,
    &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT,
    &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST,
    &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT,
    &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST,
    &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT,
    &clc797::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST,
    &clc797::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT,
    &clc797::DRAW_INDEX_BUFFER_BEGIN_END_B,
    &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_B,
];

/// The draw methods this slice does not handle (module docs).
pub const UNHANDLED_METHODS: [&Method; 5] = [
    &clc797::DRAW_INLINE_INDEX,
    &clc797::DRAW_INLINE_INDEX2X16,
    &clc797::DRAW_INLINE_INDEX4X8,
    &clc797::DRAW_ZERO_INDEX,
    &clc797::DRAW_AUTO,
];

/// What one word to a 3D method started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Started {
    /// No draw.
    Nothing,
    /// A draw.
    Draw(DrawCall),
    /// A draw method this slice does not handle (module docs), by its name in the class header.
    Unhandled(&'static str),
}

/// The size of one index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexSize {
    /// One byte.
    U8,
    /// Two bytes.
    U16,
    /// Four bytes.
    U32,
}

impl IndexSize {
    /// The size in bytes.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

/// What a draw reads its vertices from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// Consecutive vertices.
    Vertices {
        /// The first vertex.
        first: u32,
        /// How many.
        count: u32,
    },
    /// Indices from the index buffer.
    Indexed {
        /// The first index.
        first: u32,
        /// How many.
        count: u32,
        /// Their size: `None` for `SET_INDEX_BUFFER_E`'s, else the method's.
        size: Option<IndexSize>,
    },
}

/// One draw a method started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DrawCall {
    /// The topology as the begin method wrote it (`BEGIN_OP`'s values; `SET_PRIMITIVE_TOPOLOGY`
    /// may override it, `fixed` decides).
    pub topology: u32,
    /// Vertices or indices.
    pub source: Source,
    /// Instances.
    pub instance_count: u32,
    /// The first instance.
    pub base_instance: u32,
    /// Added to each index, the register's 32 bits (`SET_GLOBAL_BASE_VERTEX_INDEX`).
    pub base_vertex: u32,
}

/// The begin/end mode (Eden's `DrawMode`, without its inline-index mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    General,
    Instance,
}

/// The draw state machine of one 3D mirror (module docs).
#[derive(Clone, Debug, Default)]
pub struct DrawCalls {
    mode: Mode,
    /// Instances counted since `BEGIN` with `INSTANCE_ID` `FIRST` (Eden's `instance_count`).
    instance_count: u32,
    /// `DRAW_INDEX_BUFFER` was written since the last draw (Eden's `draw_indexed`).
    draw_indexed: bool,
    /// The topology of the last begin.
    topology: u32,
    /// The instance counter of the instanced begin-end forms.
    instanced_forms: u32,
}

impl DrawCalls {
    /// A machine with nothing pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// The draw a word `value` written to `method` starts, read with the mirror's registers
    /// (`mirror` has already stored the word); [`Started::Nothing`] for a method that starts
    /// none. A `BEGIN` with `INSTANCE_ID` `FIRST` starts the deferred draw, if any.
    ///
    /// # Errors
    ///
    /// [`StateError`] when a register the draw reads is outside the mirror's file (a mirror of
    /// another class); nothing changes then.
    pub fn method(
        &mut self,
        mirror: &Mirror,
        method: u32,
        value: u32,
    ) -> Result<Started, StateError> {
        if let Some(unhandled) = UNHANDLED_METHODS.iter().find(|m| m.method() == method) {
            return Ok(Started::Unhandled(unhandled.name));
        }
        let call = match method {
            m if m == clc797::BEGIN.method() => self.begin(mirror, value)?,
            m if m == clc797::END.method() => self.end(mirror, 1, false)?,
            m if m == clc797::DRAW_INDEX_BUFFER.method() => {
                self.draw_indexed = true;
                None
            }
            m if m == clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST.method() => {
                Some(self.instanced_array(value, false))
            }
            m if m == clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT.method() => {
                Some(self.instanced_array(value, true))
            }
            m => match small_index(m) {
                Some((size, subsequent)) => {
                    Some(self.instanced_small(mirror, value, size, subsequent)?)
                }
                None => Self::begin_end_b(mirror, m, value)?,
            },
        };
        Ok(call.map_or(Started::Nothing, Started::Draw))
    }

    /// Draws the instances an instanced begin/end deferred, if any (Eden's `DrawDeferred`, which
    /// Eden calls at a `BEGIN` with `INSTANCE_ID` `FIRST` and after each macro).
    ///
    /// # Errors
    ///
    /// As [`DrawCalls::method`].
    pub fn flush(&mut self, mirror: &Mirror) -> Result<Option<DrawCall>, StateError> {
        if self.mode != Mode::Instance || self.instance_count == 0 {
            return Ok(None);
        }
        let count = self.instance_count.saturating_add(1);
        let call = self.end(mirror, count, true)?;
        self.instance_count = 0;
        Ok(call)
    }

    /// `BEGIN` (Eden's `DrawBegin`); returns the deferred draw a `FIRST` flushes.
    fn begin(&mut self, mirror: &Mirror, value: u32) -> Result<Option<DrawCall>, StateError> {
        let instance_id = clc797::BEGIN_INSTANCE_ID.get(value);
        let mut deferred = None;
        if instance_id == clc797::BEGIN_INSTANCE_ID_FIRST {
            deferred = self.flush(mirror)?;
            self.instance_count = 0;
            self.mode = Mode::General;
        } else if instance_id == clc797::BEGIN_INSTANCE_ID_SUBSEQUENT {
            self.instance_count = self.instance_count.saturating_add(1);
            self.mode = Mode::Instance;
        }
        self.topology = clc797::BEGIN_OP.get(value);
        Ok(deferred)
    }

    /// `END` (Eden's `DrawEnd`).
    fn end(
        &mut self,
        mirror: &Mirror,
        instance_count: u32,
        force: bool,
    ) -> Result<Option<DrawCall>, StateError> {
        if self.mode == Mode::Instance && !force {
            return Ok(None);
        }
        let source = if self.draw_indexed {
            Source::Indexed {
                first: field(
                    mirror,
                    &clc797::SET_INDEX_BUFFER_F,
                    clc797::SET_INDEX_BUFFER_F_FIRST,
                )?,
                count: field(
                    mirror,
                    &clc797::DRAW_INDEX_BUFFER,
                    clc797::DRAW_INDEX_BUFFER_COUNT,
                )?,
                size: None,
            }
        } else {
            Source::Vertices {
                first: field(
                    mirror,
                    &clc797::SET_VERTEX_ARRAY_START,
                    clc797::SET_VERTEX_ARRAY_START_V,
                )?,
                count: field(
                    mirror,
                    &clc797::DRAW_VERTEX_ARRAY,
                    clc797::DRAW_VERTEX_ARRAY_COUNT,
                )?,
            }
        };
        let call = DrawCall {
            topology: self.topology,
            source,
            instance_count,
            base_instance: base_instance(mirror)?,
            base_vertex: base_vertex(mirror)?,
        };
        self.draw_indexed = false;
        Ok(Some(call))
    }

    /// The next value of the instanced forms' counter: 0 for a first, one more for a
    /// subsequent (Eden's `DrawArrayInstanced`: `base_instance = instance_count - 1`).
    fn next_instance(&mut self, subsequent: bool) -> u32 {
        self.instanced_forms = if subsequent {
            self.instanced_forms.saturating_add(1)
        } else {
            0
        };
        self.instanced_forms
    }

    /// `DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_*`.
    fn instanced_array(&mut self, value: u32, subsequent: bool) -> DrawCall {
        let base_instance = self.next_instance(subsequent);
        DrawCall {
            topology: clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_TOPOLOGY.get(value),
            source: Source::Vertices {
                first: clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_START_INDEX.get(value),
                count: clc797::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST_COUNT.get(value),
            },
            instance_count: 1,
            base_instance,
            // Eden's `DrawArrayInstanced` leaves `base_index` as the last draw left it; a
            // non-indexed draw adds no base vertex, so 0 here changes nothing drawn.
            base_vertex: 0,
        }
    }

    /// `DRAW_INDEX_BUFFER{32,16,8}_BEGIN_END_INSTANCE_*` (Eden's `DrawIndexSmall`).
    fn instanced_small(
        &mut self,
        mirror: &Mirror,
        value: u32,
        size: IndexSize,
        subsequent: bool,
    ) -> Result<DrawCall, StateError> {
        let base_vertex = base_vertex(mirror)?;
        let global_instance = base_instance(mirror)?;
        let counter = self.next_instance(subsequent);
        Ok(DrawCall {
            topology: clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_TOPOLOGY.get(value),
            source: Source::Indexed {
                first: clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_FIRST.get(value),
                count: clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST_COUNT.get(value),
                size: Some(size),
            },
            instance_count: 1,
            base_instance: global_instance.saturating_add(counter),
            base_vertex,
        })
    }

    /// `DRAW_INDEX_BUFFER_BEGIN_END_B` and `DRAW_VERTEX_ARRAY_BEGIN_END_B` (a guess; module
    /// docs); `None` for any other method.
    fn begin_end_b(
        mirror: &Mirror,
        method: u32,
        value: u32,
    ) -> Result<Option<DrawCall>, StateError> {
        let indexed = if method == clc797::DRAW_INDEX_BUFFER_BEGIN_END_B.method() {
            true
        } else if method == clc797::DRAW_VERTEX_ARRAY_BEGIN_END_B.method() {
            false
        } else {
            return Ok(None);
        };
        let control = mirror.reg(&clc797::SET_DRAW_CONTROL_A, &[])?.word();
        let instance_count = field(
            mirror,
            &clc797::SET_DRAW_CONTROL_B,
            clc797::SET_DRAW_CONTROL_B_INSTANCE_COUNT,
        )?;
        let source = if indexed {
            Source::Indexed {
                first: field(
                    mirror,
                    &clc797::DRAW_INDEX_BUFFER_BEGIN_END_A,
                    clc797::DRAW_INDEX_BUFFER_BEGIN_END_A_FIRST,
                )?,
                count: clc797::DRAW_INDEX_BUFFER_BEGIN_END_B_COUNT.get(value),
                size: None,
            }
        } else {
            Source::Vertices {
                first: field(
                    mirror,
                    &clc797::DRAW_VERTEX_ARRAY_BEGIN_END_A,
                    clc797::DRAW_VERTEX_ARRAY_BEGIN_END_A_START,
                )?,
                count: clc797::DRAW_VERTEX_ARRAY_BEGIN_END_B_COUNT.get(value),
            }
        };
        let ignore_vertex =
            clc797::SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_VERTEX_INDEX.is_set(control);
        let ignore_instance =
            clc797::SET_DRAW_CONTROL_A_IGNORE_GLOBAL_BASE_INSTANCE_INDEX.is_set(control);
        Ok(Some(DrawCall {
            topology: clc797::SET_DRAW_CONTROL_A_TOPOLOGY.get(control),
            source,
            instance_count,
            base_instance: if ignore_instance {
                0
            } else {
                base_instance(mirror)?
            },
            base_vertex: if ignore_vertex {
                0
            } else {
                base_vertex(mirror)?
            },
        }))
    }
}

/// The size and first-or-subsequent of a small index form, or `None`.
fn small_index(method: u32) -> Option<(IndexSize, bool)> {
    [
        (
            &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST,
            IndexSize::U32,
            false,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT,
            IndexSize::U32,
            true,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST,
            IndexSize::U16,
            false,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT,
            IndexSize::U16,
            true,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST,
            IndexSize::U8,
            false,
        ),
        (
            &clc797::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT,
            IndexSize::U8,
            true,
        ),
    ]
    .into_iter()
    .find(|(m, _, _)| m.method() == method)
    .map(|(_, size, subsequent)| (size, subsequent))
}

/// A field of a method that is not an array.
fn field(
    mirror: &Mirror,
    method: &'static Method,
    field: crate::field::Field,
) -> Result<u32, StateError> {
    mirror.reg(method, &[])?.field(field)
}

/// `SET_GLOBAL_BASE_VERTEX_INDEX`.
fn base_vertex(mirror: &Mirror) -> Result<u32, StateError> {
    field(
        mirror,
        &clc797::SET_GLOBAL_BASE_VERTEX_INDEX,
        clc797::SET_GLOBAL_BASE_VERTEX_INDEX_V,
    )
}

/// `SET_GLOBAL_BASE_INSTANCE_INDEX`.
fn base_instance(mirror: &Mirror) -> Result<u32, StateError> {
    field(
        mirror,
        &clc797::SET_GLOBAL_BASE_INSTANCE_INDEX,
        clc797::SET_GLOBAL_BASE_INSTANCE_INDEX_V,
    )
}
