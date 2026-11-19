//! What the adapter reports: verification 1 of docs/specs/render-backend.md ("What wgpu reports
//! on this Mac"), which decides `Caps` and the Mac's `EmulationFlags`.
//!
//! The report names the adapter and lists, as the adapter offers them (before the backend picks
//! the features it asks the device for), the facts the verification asks for: ASTC (LDR, HDR,
//! 3D), the vertex attribute limit, `D32_FLOAT_S8_UINT`, clip distances, dual-source blending,
//! subgroups, 16-bit and 64-bit shader types, binding arrays with non-uniform indexing and mesh
//! shaders; and the limits `Caps` takes.
//!
//! It also records what decides whether the device meets "subgroup size 32"
//! (`crate::pipeline`, "Requirements"): the backend and name (the Apple GPU family), the
//! subgroup size range wgpu reports, and the verdict. The verdict here is read from the
//! adapter's features; the backend decides from the device's, which keep `SUBGROUP` whenever
//! the adapter offers it (`crate::caps::device_features`), so the two agree
//! (`pipeline::tests::a_subgroup_pipeline_follows_the_device` checks it on the host's adapter).

use crate::pipeline::{SubgroupSupport, apple_gpu_family};
use std::fmt;

/// One adapter's answers to verification 1, as plain lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterReport {
    lines: Vec<(&'static str, String)>,
}

fn yes_no(has: bool) -> String {
    if has { "yes" } else { "no" }.to_owned()
}

impl AdapterReport {
    /// The report for an adapter described by `info` that offers `features` and `limits`.
    pub fn new(info: &wgpu::AdapterInfo, features: wgpu::Features, limits: &wgpu::Limits) -> Self {
        use wgpu::Features as F;
        let has = |f: F| yes_no(features.contains(f));
        let lines = vec![
            ("adapter", info.name.clone()),
            ("backend", format!("{:?}", info.backend)),
            ("device type", format!("{:?}", info.device_type)),
            (
                "driver",
                format!("{} {}", info.driver, info.driver_info)
                    .trim()
                    .to_owned(),
            ),
            ("ASTC (LDR)", has(F::TEXTURE_COMPRESSION_ASTC)),
            ("ASTC (HDR)", has(F::TEXTURE_COMPRESSION_ASTC_HDR)),
            ("ASTC (3D)", has(F::TEXTURE_COMPRESSION_ASTC_SLICED_3D)),
            ("BC", has(F::TEXTURE_COMPRESSION_BC)),
            ("ETC2 and EAC", has(F::TEXTURE_COMPRESSION_ETC2)),
            ("D32 float + S8", has(F::DEPTH32FLOAT_STENCIL8)),
            (
                "16-bit normalised formats",
                has(F::TEXTURE_FORMAT_16BIT_NORM),
            ),
            ("clip distances", has(F::CLIP_DISTANCES)),
            ("dual-source blending", has(F::DUAL_SOURCE_BLENDING)),
            ("subgroups", has(F::SUBGROUP)),
            (
                "subgroup size range",
                format!("{} to {}", info.subgroup_min_size, info.subgroup_max_size),
            ),
            ("Apple GPU family", yes_no(apple_gpu_family(info))),
            (
                "subgroups of 32",
                SubgroupSupport::of(info, features).to_string(),
            ),
            ("f16 in shaders", has(F::SHADER_F16)),
            ("f64 in shaders", has(F::SHADER_F64)),
            ("i64 in shaders", has(F::SHADER_INT64)),
            ("texture binding arrays", has(F::TEXTURE_BINDING_ARRAY)),
            (
                "non-uniform indexing of binding arrays",
                has(F::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
            ),
            (
                "mesh shaders (experimental)",
                has(F::EXPERIMENTAL_MESH_SHADER),
            ),
            ("depth clip control", has(F::DEPTH_CLIP_CONTROL)),
            ("clamp-to-border", has(F::ADDRESS_MODE_CLAMP_TO_BORDER)),
            ("polygon mode line", has(F::POLYGON_MODE_LINE)),
            ("polygon mode point", has(F::POLYGON_MODE_POINT)),
            ("pipeline cache", has(F::PIPELINE_CACHE)),
            (
                "max vertex attributes",
                limits.max_vertex_attributes.to_string(),
            ),
            ("max vertex buffers", limits.max_vertex_buffers.to_string()),
            (
                "max colour attachments",
                limits.max_color_attachments.to_string(),
            ),
            (
                "max texture size 1D",
                limits.max_texture_dimension_1d.to_string(),
            ),
            (
                "max texture size 2D",
                limits.max_texture_dimension_2d.to_string(),
            ),
            (
                "max texture size 3D",
                limits.max_texture_dimension_3d.to_string(),
            ),
            (
                "max texture layers",
                limits.max_texture_array_layers.to_string(),
            ),
            ("max buffer size", limits.max_buffer_size.to_string()),
            ("max bind groups", limits.max_bind_groups.to_string()),
            (
                "max uniform buffers per stage",
                limits.max_uniform_buffers_per_shader_stage.to_string(),
            ),
            (
                "max storage buffers per stage",
                limits.max_storage_buffers_per_shader_stage.to_string(),
            ),
            (
                "max sampled textures per stage",
                limits.max_sampled_textures_per_shader_stage.to_string(),
            ),
            (
                "max storage textures per stage",
                limits.max_storage_textures_per_shader_stage.to_string(),
            ),
            (
                "max samplers per stage",
                limits.max_samplers_per_shader_stage.to_string(),
            ),
        ];
        Self { lines }
    }

    /// The report's lines: what each is, and the answer.
    pub fn lines(&self) -> &[(&'static str, String)] {
        &self.lines
    }
}

impl fmt::Display for AdapterReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (what, answer) in &self.lines {
            writeln!(f, "{what}: {answer}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::AdapterReport;

    #[test]
    fn the_report_reads_as_plain_lines() {
        let info = wgpu::AdapterInfo {
            name: "Test GPU".into(),
            vendor: 1,
            device: 2,
            device_type: wgpu::DeviceType::IntegratedGpu,
            device_pci_bus_id: String::new(),
            driver: "drv".into(),
            driver_info: "1.0".into(),
            backend: wgpu::Backend::Metal,
            subgroup_min_size: 4,
            subgroup_max_size: 64,
            transient_saves_memory: None,
            limit_bucket: None,
        };
        let limits = wgpu::Limits {
            max_vertex_attributes: 31,
            ..wgpu::Limits::default()
        };
        let report = AdapterReport::new(
            &info,
            wgpu::Features::TEXTURE_COMPRESSION_ASTC | wgpu::Features::SHADER_F16,
            &limits,
        );
        let text = report.to_string();
        for line in [
            "adapter: Test GPU\n",
            "backend: Metal\n",
            "device type: IntegratedGpu\n",
            "driver: drv 1.0\n",
            "ASTC (LDR): yes\n",
            "ASTC (HDR): no\n",
            "f16 in shaders: yes\n",
            "f64 in shaders: no\n",
            "max vertex attributes: 31\n",
            "subgroup size range: 4 to 64\n",
            "Apple GPU family: no\n",
            "subgroups of 32: no (the device has no subgroups)\n",
        ] {
            assert!(text.contains(line), "{line:?} in {text}");
        }
        assert_eq!(report.lines().len(), text.lines().count());
    }

    /// The facts that decide "subgroup size 32" are recorded, with the verdict: the backend and
    /// name that make the Apple family, the reported range (min before max), and the subgroup
    /// feature.
    #[test]
    fn the_report_records_what_decides_the_subgroup_size() {
        let info = |backend, name: &str, min, max| wgpu::AdapterInfo {
            name: name.into(),
            subgroup_min_size: min,
            subgroup_max_size: max,
            ..wgpu::AdapterInfo::new(wgpu::DeviceType::IntegratedGpu, backend)
        };
        let s = wgpu::Features::SUBGROUP;
        let limits = wgpu::Limits::default();
        for (info, features, lines) in [
            (
                info(wgpu::Backend::Metal, "Apple M2", 4, 64),
                s,
                [
                    "backend: Metal\n",
                    "subgroup size range: 4 to 64\n",
                    "Apple GPU family: yes\n",
                    "subgroups of 32: yes (Apple GPU family, 32-wide SIMD groups)\n",
                ],
            ),
            (
                info(wgpu::Backend::Metal, "AMD Radeon Pro 5500M", 8, 16),
                s,
                [
                    "backend: Metal\n",
                    "subgroup size range: 8 to 16\n",
                    "Apple GPU family: no\n",
                    "subgroups of 32: no (the reported range is 8 to 16)\n",
                ],
            ),
            (
                info(wgpu::Backend::Vulkan, "Apple M2", 32, 32),
                s,
                [
                    "backend: Vulkan\n",
                    "subgroup size range: 32 to 32\n",
                    "Apple GPU family: no\n",
                    "subgroups of 32: yes (the reported range is 32 to 32)\n",
                ],
            ),
        ] {
            let text = AdapterReport::new(&info, features, &limits).to_string();
            for line in lines {
                assert!(text.contains(line), "{line:?} in {text}");
            }
        }
    }
}
