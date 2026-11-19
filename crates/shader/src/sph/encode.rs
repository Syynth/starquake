//! The test encoder: builds a header's words from a [`ProgramHeader`], with the same bit
//! positions the parser reads (the private `layout` module). For this crate's tests and
//! downstream crates' dev builds; exported behind the `test-support` feature.
//!
//! Bits the parser ignores (module docs of [`crate::sph`], "Bits not read") are written as
//! zero, so `ProgramHeader::from_words(&encode(header)?)` gives `header` back for every header
//! whose fields fit their widths.

use thiserror::Error;

use super::layout::{
    BARYCENTRIC_CELL, BARYCENTRIC_W_CELL, CLIP_DISTANCES, COLOR_EXT, DOES_FP64, DOES_GLOBAL_STORE,
    DOES_LOAD_OR_STORE, FOG_COORDINATE, GENERIC_EXT, GEOMETRY_PASSTHROUGH, HeaderField, IMAP,
    INSTANCE_ID, ISBE_SPACE_SHARING, KILLS_PIXELS, LOCAL_MEMORY_CRS_SIZE, LOCAL_MEMORY_HIGH_SIZE,
    LOCAL_MEMORY_LOW_SIZE, MAX_OUTPUT_VERTEX_COUNT, MRT_ENABLE, OMAP, OMAP_DEPTH, OMAP_SAMPLE_MASK,
    OUTPUT_TOPOLOGY, OUTPUT_TOPOLOGY_LINESTRIP, OUTPUT_TOPOLOGY_NONE, OUTPUT_TOPOLOGY_POINTLIST,
    OUTPUT_TOPOLOGY_TRIANGLESTRIP, PER_PATCH_ATTRIBUTE_COUNT, POINT_SIZE, POINT_SPRITE_S,
    POINT_SPRITE_T, POSITION, PRIMITIVE_ID, PS_COLOR, PS_GENERIC, PS_SYSTEM_VALUES_C,
    PS_SYSTEM_VALUES_D, PS_TARGETS, PS_TEXTURE, READS_INTERLOCK_TICKET, RT_ARRAY_INDEX,
    SASS_VERSION, SHADER_TYPE, SHADER_TYPE_GEOMETRY, SHADER_TYPE_PIXEL, SHADER_TYPE_TESSELLATION,
    SHADER_TYPE_TESSELLATION_INIT, SHADER_TYPE_VERTEX, SPH_TYPE, SPH_TYPE_PS, SPH_TYPE_VTG,
    STORE_REQ_END, STORE_REQ_START, STREAM_OUT_MASK, SYSTEM_VALUES_A, SYSTEM_VALUES_A_EXT,
    SYSTEM_VALUES_B_EXT, SYSTEM_VALUES_C_EXT, SYSTEM_VALUES_D_EXT, TESSELLATION_POINT_U,
    TESSELLATION_POINT_V, TEXTURE_EXT, THREADS_PER_INPUT_PRIMITIVE, USES_INNER_COVERAGE, VERSION,
    VERTEX_ID, VIEWPORT_INDEX, VIEWPORT_MASK_CELL, VIEWPORT_MASK2_CELL, VMAP, VPRS_TABLE_INDEX,
    VTG_COLOR, VTG_GENERIC, VTG_SYSTEM_VALUES_C, VTG_SYSTEM_VALUES_D, VTG_TEXTURE,
};
use super::{
    Cell, Common, Components, HEADER_BYTES, HEADER_WORDS, Io, OutputTopology, PixelIo,
    ProgramHeader, SystemValuesAb, SystemValuesC, SystemValuesD, VtgMap, VtgStage, cell, entry,
};

/// Why a header cannot be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// A field's value needs more bits than the field has.
    #[error("the value {value:#x} does not fit the {width}-bit field at bit {start}")]
    TooWide {
        /// The field's first bit.
        start: u32,
        /// The field's width.
        width: u32,
        /// The value given.
        value: u32,
    },
}

/// The header's 32 words.
pub fn encode(header: &ProgramHeader) -> Result<[u32; HEADER_WORDS], EncodeError> {
    let mut write = Writer {
        words: [0; HEADER_WORDS],
    };
    write.field(VERSION, ProgramHeader::VERSION)?;
    let (sph_type, shader_type) = match &header.io {
        Io::Vtg(io) => (
            SPH_TYPE_VTG,
            match io.stage {
                VtgStage::Vertex => SHADER_TYPE_VERTEX,
                VtgStage::TessellationInit => SHADER_TYPE_TESSELLATION_INIT,
                VtgStage::Tessellation => SHADER_TYPE_TESSELLATION,
                VtgStage::Geometry => SHADER_TYPE_GEOMETRY,
            },
        ),
        Io::Pixel(_) => (SPH_TYPE_PS, SHADER_TYPE_PIXEL),
    };
    write.field(SPH_TYPE, sph_type)?;
    write.field(SHADER_TYPE, shader_type)?;
    write.common(&header.common)?;
    match &header.io {
        Io::Vtg(io) => {
            write.vtg_map(IMAP, &io.input)?;
            write.vtg_map(OMAP, &io.output)?;
            write.vtg_map(VMAP, &io.vmap)?;
        }
        Io::Pixel(io) => write.pixel_io(io)?,
    }
    Ok(write.words)
}

/// The header's 128 bytes, as they lie in front of a graphics program.
pub fn encode_bytes(header: &ProgramHeader) -> Result<[u8; HEADER_BYTES], EncodeError> {
    let words = encode(header)?;
    let mut bytes = [0; HEADER_BYTES];
    let (chunks, _) = bytes.as_chunks_mut::<4>();
    for (chunk, word) in chunks.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    Ok(bytes)
}

/// The header's words, written field by field.
struct Writer {
    words: [u32; HEADER_WORDS],
}

impl Writer {
    fn field(&mut self, field: HeaderField, value: u32) -> Result<(), EncodeError> {
        let too_wide = EncodeError::TooWide {
            start: field.start,
            width: field.width,
            value,
        };
        if value & !field.ones() != 0 {
            return Err(too_wide);
        }
        let index = usize::try_from(field.start / 32).unwrap_or(usize::MAX);
        let shift = field.start % 32;
        let Some(word) = self.words.get_mut(index) else {
            return Err(too_wide);
        };
        *word |= value.checked_shl(shift).unwrap_or(0);
        if shift != 0 {
            let high = value.checked_shr(32 - shift).unwrap_or(0);
            if high != 0 {
                let Some(next) = self.words.get_mut(index.saturating_add(1)) else {
                    return Err(too_wide);
                };
                *next |= high;
            }
        }
        Ok(())
    }

    fn flag(&mut self, field: HeaderField, value: bool) -> Result<(), EncodeError> {
        self.field(field, u32::from(value))
    }

    fn cell<T: Cell>(&mut self, start: u32, index: u32, value: T) -> Result<(), EncodeError> {
        self.field(cell::<T>(start, index), value.to_bits())
    }

    fn components<T: Cell>(&mut self, start: u32, value: Components<T>) -> Result<(), EncodeError> {
        self.cell(start, 0, value.x)?;
        self.cell(start, 1, value.y)?;
        self.cell(start, 2, value.z)?;
        self.cell(start, 3, value.w)
    }

    fn run<T: Cell>(&mut self, start: u32, values: &[Components<T>]) -> Result<(), EncodeError> {
        for (index, value) in (0_u32..).zip(values) {
            self.components(entry::<T>(start, index), *value)?;
        }
        Ok(())
    }

    fn common(&mut self, common: &Common) -> Result<(), EncodeError> {
        self.flag(MRT_ENABLE, common.mrt_enable)?;
        self.flag(KILLS_PIXELS, common.kills_pixels)?;
        self.flag(DOES_GLOBAL_STORE, common.does_global_store)?;
        self.field(SASS_VERSION, u32::from(common.sass_version))?;
        self.flag(GEOMETRY_PASSTHROUGH, common.geometry_passthrough)?;
        self.flag(ISBE_SPACE_SHARING, common.isbe_space_sharing)?;
        self.flag(DOES_LOAD_OR_STORE, common.does_load_or_store)?;
        self.flag(DOES_FP64, common.does_fp64)?;
        self.field(STREAM_OUT_MASK, u32::from(common.stream_out_mask))?;
        self.field(LOCAL_MEMORY_LOW_SIZE, common.local_memory_low_size)?;
        self.field(
            PER_PATCH_ATTRIBUTE_COUNT,
            u32::from(common.per_patch_attribute_count),
        )?;
        self.field(LOCAL_MEMORY_HIGH_SIZE, common.local_memory_high_size)?;
        self.field(
            THREADS_PER_INPUT_PRIMITIVE,
            u32::from(common.threads_per_input_primitive),
        )?;
        self.field(LOCAL_MEMORY_CRS_SIZE, common.local_memory_crs_size)?;
        self.field(
            OUTPUT_TOPOLOGY,
            match common.output_topology {
                None => OUTPUT_TOPOLOGY_NONE,
                Some(OutputTopology::PointList) => OUTPUT_TOPOLOGY_POINTLIST,
                Some(OutputTopology::LineStrip) => OUTPUT_TOPOLOGY_LINESTRIP,
                Some(OutputTopology::TriangleStrip) => OUTPUT_TOPOLOGY_TRIANGLESTRIP,
            },
        )?;
        self.field(
            MAX_OUTPUT_VERTEX_COUNT,
            u32::from(common.max_output_vertex_count),
        )?;
        self.field(STORE_REQ_START, u32::from(common.store_req_start))?;
        self.field(STORE_REQ_END, u32::from(common.store_req_end))
    }

    fn system_values_ab(&mut self, base: u32, values: &SystemValuesAb) -> Result<(), EncodeError> {
        self.field(SYSTEM_VALUES_A.at(base), values.system_values_a)?;
        self.flag(VPRS_TABLE_INDEX.at(base), values.vprs_table_index)?;
        self.flag(PRIMITIVE_ID.at(base), values.primitive_id)?;
        self.flag(RT_ARRAY_INDEX.at(base), values.rt_array_index)?;
        self.flag(VIEWPORT_INDEX.at(base), values.viewport_index)?;
        self.flag(POINT_SIZE.at(base), values.point_size)?;
        self.components(base.saturating_add(POSITION), values.position)
    }

    fn system_values_c(&mut self, start: u32, values: SystemValuesC) -> Result<(), EncodeError> {
        self.field(CLIP_DISTANCES.at(start), u32::from(values.clip_distances))?;
        self.flag(POINT_SPRITE_S.at(start), values.point_sprite_s)?;
        self.flag(POINT_SPRITE_T.at(start), values.point_sprite_t)?;
        self.flag(FOG_COORDINATE.at(start), values.fog_coordinate)?;
        self.flag(TESSELLATION_POINT_U.at(start), values.tessellation_point_u)?;
        self.flag(TESSELLATION_POINT_V.at(start), values.tessellation_point_v)?;
        self.flag(INSTANCE_ID.at(start), values.instance_id)?;
        self.flag(VERTEX_ID.at(start), values.vertex_id)
    }

    fn system_values_d<T: Cell>(
        &mut self,
        start: u32,
        values: &SystemValuesD<T>,
    ) -> Result<(), EncodeError> {
        self.cell(start, VIEWPORT_MASK_CELL, values.viewport_mask)?;
        self.cell(start, VIEWPORT_MASK2_CELL, values.viewport_mask2)?;
        for (index, value) in (BARYCENTRIC_CELL..).zip(values.barycentric) {
            self.cell(start, index, value)?;
        }
        for (index, value) in (BARYCENTRIC_W_CELL..).zip(values.barycentric_w) {
            self.cell(start, index, value)?;
        }
        Ok(())
    }

    fn vtg_map(&mut self, base: u32, map: &VtgMap) -> Result<(), EncodeError> {
        self.system_values_ab(base, &map.system_values_ab)?;
        self.run(base.saturating_add(VTG_GENERIC), &map.generic)?;
        self.run(base.saturating_add(VTG_COLOR), &map.color)?;
        self.system_values_c(
            base.saturating_add(VTG_SYSTEM_VALUES_C),
            map.system_values_c,
        )?;
        self.run(base.saturating_add(VTG_TEXTURE), &map.texture)?;
        self.system_values_d(
            base.saturating_add(VTG_SYSTEM_VALUES_D),
            &map.system_values_d,
        )
    }

    fn pixel_io(&mut self, io: &PixelIo) -> Result<(), EncodeError> {
        self.system_values_ab(IMAP, &io.system_values_ab)?;
        self.run(PS_GENERIC, &io.generic)?;
        self.run(PS_COLOR, &io.color)?;
        self.system_values_c(PS_SYSTEM_VALUES_C, io.system_values_c)?;
        self.run(PS_TEXTURE, &io.texture)?;
        self.system_values_d(PS_SYSTEM_VALUES_D, &io.system_values_d)?;
        self.run(PS_TARGETS, &io.targets)?;
        self.flag(OMAP_SAMPLE_MASK, io.writes_sample_mask)?;
        self.flag(OMAP_DEPTH, io.writes_depth)?;
        self.flag(READS_INTERLOCK_TICKET, io.reads_interlock_ticket)?;
        self.flag(USES_INNER_COVERAGE, io.uses_inner_coverage)?;
        self.field(SYSTEM_VALUES_A_EXT, io.system_values_a_ext)?;
        self.field(SYSTEM_VALUES_B_EXT, u32::from(io.system_values_b_ext))?;
        self.run(GENERIC_EXT, &io.generic_ext)?;
        self.run(COLOR_EXT, &io.color_ext)?;
        self.field(SYSTEM_VALUES_C_EXT, u32::from(io.system_values_c_ext))?;
        self.run(TEXTURE_EXT, &io.texture_ext)?;
        self.system_values_d(SYSTEM_VALUES_D_EXT, &io.system_values_d_ext)
    }
}
