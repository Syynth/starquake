//! Render capture v2: golden bytes for every record kind, every reading check and its order,
//! every bound at its edge, a generated round trip, the capturing backend and replay.

mod bounds;
mod golden;
mod read;
mod replay;
mod roundtrip;
mod wrap;

use super::{
    CaptureBuffer, CaptureError, CaptureLimits, CaptureReader, CaptureSink, CaptureWriter, Record,
    Start, WaitSetting,
};
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use std::sync::Arc;

/// A device with room for what the tests make.
pub(super) fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 4,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

/// A `Start` for the tests' device.
pub(super) fn start() -> Start {
    Start {
        build: "test".to_owned(),
        backend: "recording".to_owned(),
        caps: caps(),
        wait: WaitSetting::Wait,
        memory_limit: 1 << 20,
    }
}

/// A writer with `limits` into a fresh buffer, and the buffer.
pub(super) fn writer_with(limits: CaptureLimits) -> (CaptureWriter, Arc<CaptureBuffer>) {
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let writer =
        CaptureWriter::with_limits(sink, start(), u64::MAX, limits).expect("the start fits");
    (writer, buffer)
}

/// A writer with the shipped bounds.
pub(super) fn writer() -> (CaptureWriter, Arc<CaptureBuffer>) {
    writer_with(CaptureLimits::SHIPPED)
}

/// The header and `Start` the tests' writer writes.
pub(super) fn prefix() -> Vec<u8> {
    writer().1.bytes()
}

/// Every record of `bytes`, read with `limits`, or the first error.
pub(super) fn read_with(bytes: &[u8], limits: CaptureLimits) -> Result<Vec<Record>, CaptureError> {
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::with_limits(bytes, len, limits)?;
    let mut records = Vec::new();
    while let Some(record) = reader.next_record()? {
        records.push(record);
    }
    Ok(records)
}

/// Every record of `bytes`, with the shipped bounds.
pub(super) fn read(bytes: &[u8]) -> Result<Vec<Record>, CaptureError> {
    read_with(bytes, CaptureLimits::SHIPPED)
}

/// A record framed by hand: its tag, a one-byte length (below 128), and its body.
pub(super) fn framed(tag: u8, body: &[u8]) -> Vec<u8> {
    let len = u8::try_from(body.len()).expect("short body");
    assert!(len < 0x80, "a one-byte length");
    let mut out = vec![tag, len];
    out.extend_from_slice(body);
    out
}

/// The `End` record.
pub(super) const END: [u8; 2] = [0, 0];

/// `prefix()`, then `records`, then `End`.
pub(super) fn file(records: &[&[u8]]) -> Vec<u8> {
    let mut out = prefix();
    for record in records {
        out.extend_from_slice(record);
    }
    out.extend_from_slice(&END);
    out
}
