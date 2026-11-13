//! The helper shaders checked against naga, with no GPU: every source parses and validates, and
//! the parsed module agrees with the tables and constants a backend uses the shader by.
//!
//! Nothing here runs a shader (parent module docs, "What the tests check, and what they cannot").

use super::{EntryPoint, HostShader, clear, present};
use crate::resource::ViewDimension;
use crate::shader::{ResourceBinding, ResourceKind, SampleKind, ShaderStage};
use naga::front::wgsl::ParseError;
use naga::valid::{
    Capabilities, EntryPointError, ModuleInfo, ValidationError, ValidationFlags, Validator,
    VaryingError,
};
use naga::{Module, TypeInner, WithSpan};
use std::path::{Path, PathBuf};

/// Why naga refused a source.
#[derive(Debug)]
enum Refused {
    /// The WGSL front end could not parse it.
    Parse(Box<ParseError>),
    /// It parsed, and the validator refused the module.
    Validate(Box<WithSpan<ValidationError>>),
}

/// Parses `source` and validates it the way every helper shader must pass: every validation
/// flag on, and no optional capability, so a shader that passes needs nothing a device may lack.
fn check(source: &str) -> Result<(Module, ModuleInfo), Refused> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| Refused::Parse(Box::new(e)))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|e| Refused::Validate(Box::new(e)))?;
    Ok((module, info))
}

/// A helper shader's checked module, or a failure showing naga's diagnostic.
fn checked(shader: HostShader) -> (Module, ModuleInfo) {
    let source = shader.source();
    match check(source) {
        Ok(checked) => checked,
        Err(Refused::Parse(e)) => unreachable!(
            "{} does not parse:\n{}",
            shader.file_name(),
            e.emit_to_string(source)
        ),
        Err(Refused::Validate(e)) => unreachable!(
            "{} does not validate:\n{}",
            shader.file_name(),
            e.emit_to_string(source)
        ),
    }
}

#[test]
fn every_host_shader_parses_and_validates() {
    for &shader in HostShader::ALL {
        checked(shader);
    }
}

#[test]
fn a_source_that_does_not_parse_is_refused() {
    let refused = check("@vertex fn vs_main( -> @builtin(position) vec4<f32> {").unwrap_err();
    assert!(matches!(refused, Refused::Parse(_)), "{refused:?}");
}

/// The validator is on, not only the parser: this source parses, and only the validator's
/// check of bindings refuses it (two resources of one entry point at the same binding).
#[test]
fn a_source_the_validator_refuses_is_refused() {
    let source = "
        @group(0) @binding(0) var a: texture_2d<f32>;
        @group(0) @binding(0) var b: texture_2d<f32>;
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return textureLoad(a, vec2<i32>(0), 0) + textureLoad(b, vec2<i32>(0), 0);
        }";
    let refused = check(source).unwrap_err();
    let Refused::Validate(error) = &refused else {
        unreachable!("{refused:?}");
    };
    assert!(
        matches!(
            error.as_inner(),
            ValidationError::EntryPoint {
                source: EntryPointError::BindingCollision(_),
                ..
            }
        ),
        "{error:?}"
    );
}

/// No optional capability is granted: a source that needs one is refused, naming it.
#[test]
fn a_source_needing_an_optional_capability_is_refused() {
    let source = "
        @fragment fn fs_main(@builtin(sample_index) i: u32) -> @location(0) vec4<f32> {
            return vec4<f32>(f32(i));
        }";
    let refused = check(source).unwrap_err();
    let Refused::Validate(error) = &refused else {
        unreachable!("{refused:?}");
    };
    assert!(
        matches!(
            error.as_inner(),
            ValidationError::EntryPoint {
                source: EntryPointError::Argument(
                    0,
                    VaryingError::UnsupportedCapability(Capabilities::MULTISAMPLED_SHADING)
                ),
                ..
            }
        ),
        "{error:?}"
    );
}

/// Every `.wgsl` file under `dir`, as paths relative to `root` with `/` separators.
fn wgsl_files(root: &Path, dir: &Path, found: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| unreachable!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            wgsl_files(root, &path, found);
        } else if path.extension().is_some_and(|e| e == "wgsl") {
            let relative = path.strip_prefix(root).expect("a path under the root");
            let parts: Vec<_> = relative
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            found.push(parts.join("/"));
        }
    }
}

/// A shader cannot be added without being validated: a WGSL file anywhere in the crate's
/// sources must be a listed helper shader, and the tests above and below run over the list.
#[test]
fn every_wgsl_file_is_a_listed_host_shader() {
    let src: PathBuf = [env!("CARGO_MANIFEST_DIR"), "src"].iter().collect();
    let mut on_disk = Vec::new();
    wgsl_files(&src, &src, &mut on_disk);
    on_disk.sort();
    let mut listed: Vec<String> = HostShader::ALL
        .iter()
        .map(|s| format!("host_shaders/{}", s.file_name()))
        .collect();
    listed.sort();
    assert_eq!(on_disk, listed);
}

fn stage(stage: naga::ShaderStage) -> ShaderStage {
    match stage {
        naga::ShaderStage::Vertex => ShaderStage::Vertex,
        naga::ShaderStage::Fragment => ShaderStage::Fragment,
        naga::ShaderStage::Compute => ShaderStage::Compute,
        other => unreachable!("a helper shader has a {other:?} entry point"),
    }
}

#[test]
fn entry_points_match_the_sources() {
    for &shader in HostShader::ALL {
        let (module, _) = checked(shader);
        let in_source: Vec<EntryPoint> = module
            .entry_points
            .iter()
            .map(|e| EntryPoint {
                // The tables hold `&'static str`s; leaking a test's few names gives the same.
                name: e.name.clone().leak(),
                stage: stage(e.stage),
            })
            .collect();
        assert_eq!(in_source, shader.entry_points(), "{}", shader.file_name());
    }
}

fn view_dimension(dim: naga::ImageDimension, arrayed: bool) -> ViewDimension {
    match (dim, arrayed) {
        (naga::ImageDimension::D1, false) => ViewDimension::D1,
        (naga::ImageDimension::D1, true) => ViewDimension::D1Array,
        (naga::ImageDimension::D2, false) => ViewDimension::D2,
        (naga::ImageDimension::D2, true) => ViewDimension::D2Array,
        (naga::ImageDimension::Cube, false) => ViewDimension::Cube,
        (naga::ImageDimension::Cube, true) => ViewDimension::CubeArray,
        (naga::ImageDimension::D3, false) => ViewDimension::D3,
        (naga::ImageDimension::D3, true) => unreachable!("an arrayed 3D image"),
    }
}

/// Whether any entry point samples `image` with a sampler that is not a comparison sampler,
/// which is what makes its binding a filterable float texture.
fn sampled_with_plain_sampler(
    module: &Module,
    info: &ModuleInfo,
    image: naga::Handle<naga::GlobalVariable>,
) -> bool {
    (0..module.entry_points.len()).any(|index| {
        info.get_entry_point(index).sampling_set.iter().any(|key| {
            let sampler = &module.global_variables[key.sampler];
            key.image == image
                && matches!(
                    module.types[sampler.ty].inner,
                    TypeInner::Sampler { comparison: false }
                )
        })
    })
}

/// The resources a checked module binds, in declaration order, as this crate describes them.
fn resources(module: &Module, info: &ModuleInfo) -> Vec<ResourceBinding> {
    let mut found = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        let kind = match (global.space, &module.types[global.ty].inner) {
            (naga::AddressSpace::Uniform, _) => ResourceKind::UniformBuffer,
            (naga::AddressSpace::Storage { access }, _) => ResourceKind::StorageBuffer {
                writable: access.contains(naga::StorageAccess::STORE),
            },
            (naga::AddressSpace::Handle, TypeInner::Sampler { comparison }) => {
                ResourceKind::Sampler {
                    comparison: *comparison,
                }
            }
            (
                naga::AddressSpace::Handle,
                TypeInner::Image {
                    dim,
                    arrayed,
                    class: naga::ImageClass::Sampled { kind, multi },
                },
            ) => ResourceKind::SampledTexture {
                dimension: view_dimension(*dim, *arrayed),
                sample: match kind {
                    naga::ScalarKind::Float => SampleKind::Float {
                        filterable: sampled_with_plain_sampler(module, info, handle),
                    },
                    naga::ScalarKind::Sint => SampleKind::Sint,
                    naga::ScalarKind::Uint => SampleKind::Uint,
                    other => unreachable!("a texture sampled as {other:?}"),
                },
                multisampled: *multi,
            },
            (
                naga::AddressSpace::Handle,
                TypeInner::Image {
                    dim,
                    arrayed,
                    class: naga::ImageClass::Depth { multi },
                },
            ) => ResourceKind::SampledTexture {
                dimension: view_dimension(*dim, *arrayed),
                sample: SampleKind::Depth,
                multisampled: *multi,
            },
            other => unreachable!("no helper shader binds {other:?} yet: teach this test the kind"),
        };
        found.push(ResourceBinding {
            group: binding.group,
            binding: binding.binding,
            kind,
        });
    }
    found
}

#[test]
fn resources_match_the_sources() {
    for &shader in HostShader::ALL {
        let (module, info) = checked(shader);
        assert_eq!(
            resources(&module, &info),
            shader.resources(),
            "{}",
            shader.file_name()
        );
    }
}

/// The members of the struct bound as a uniform buffer at `group` and `binding`, and its size.
fn uniform_block(module: &Module, group: u32, binding: u32) -> (&[naga::StructMember], u32) {
    let global = module
        .global_variables
        .iter()
        .map(|(_, global)| global)
        .find(|global| {
            global
                .binding
                .as_ref()
                .is_some_and(|b| b.group == group && b.binding == binding)
        })
        .expect("a variable at the binding");
    assert_eq!(global.space, naga::AddressSpace::Uniform);
    match &module.types[global.ty].inner {
        TypeInner::Struct { members, span } => (members, *span),
        other => unreachable!("the block is {other:?}"),
    }
}

const F32: naga::Scalar = naga::Scalar::F32;
const VEC2_F32: TypeInner = TypeInner::Vector {
    size: naga::VectorSize::Bi,
    scalar: F32,
};

#[test]
fn present_parameter_block_layout() {
    let (module, _) = checked(HostShader::Present);
    let (members, size) = uniform_block(&module, present::GROUP, present::PARAMS_BINDING);
    assert_eq!(size, present::PARAMS_SIZE);
    let [matrix, vertices] = members else {
        unreachable!("{members:?}");
    };

    assert_eq!(matrix.offset, present::MATRIX_OFFSET);
    assert_eq!(
        module.types[matrix.ty].inner,
        TypeInner::Matrix {
            columns: naga::VectorSize::Quad,
            rows: naga::VectorSize::Quad,
            scalar: F32,
        }
    );

    assert_eq!(vertices.offset, present::VERTICES_OFFSET);
    let TypeInner::Array { base, size, stride } = module.types[vertices.ty].inner else {
        unreachable!("{:?}", module.types[vertices.ty]);
    };
    assert_eq!(stride, present::VERTEX_STRIDE);
    let count = core::num::NonZeroU32::new(present::VERTEX_COUNT).expect("a vertex count");
    assert_eq!(size, naga::ArraySize::Constant(count));
    // The block ends with its last vertex: nothing follows the array.
    assert_eq!(
        present::VERTICES_OFFSET + present::VERTEX_COUNT * present::VERTEX_STRIDE,
        present::PARAMS_SIZE
    );

    let TypeInner::Struct { members, span } = &module.types[base].inner else {
        unreachable!("{:?}", module.types[base]);
    };
    assert_eq!(*span, present::VERTEX_STRIDE);
    let [position, tex_coord] = members.as_slice() else {
        unreachable!("{members:?}");
    };
    assert_eq!(position.offset, present::VERTEX_POSITION_OFFSET);
    assert_eq!(module.types[position.ty].inner, VEC2_F32);
    assert_eq!(tex_coord.offset, present::VERTEX_TEX_COORD_OFFSET);
    assert_eq!(module.types[tex_coord.ty].inner, VEC2_F32);
    // The two members are told apart by name as well as by offset, so swapping them in the
    // source (which keeps both offsets and types) fails here.
    assert_eq!(position.name.as_deref(), Some("position"));
    assert_eq!(tex_coord.name.as_deref(), Some("tex_coord"));
}

#[test]
fn clear_parameter_block_layout() {
    let (module, _) = checked(HostShader::Clear);
    let (members, size) = uniform_block(&module, clear::GROUP, clear::PARAMS_BINDING);
    assert_eq!(size, clear::PARAMS_SIZE);
    let [value] = members else {
        unreachable!("{members:?}");
    };
    assert_eq!(value.offset, clear::VALUE_OFFSET);
    assert_eq!(
        module.types[value.ty].inner,
        TypeInner::Vector {
            size: naga::VectorSize::Quad,
            scalar: naga::Scalar::U32,
        }
    );
}

/// What a fragment entry point returns.
#[derive(Debug, PartialEq, Eq)]
enum FragmentOutput {
    /// Four components of a kind, at a colour location.
    Color {
        location: u32,
        kind: naga::ScalarKind,
    },
    /// One `f32`, as the fragment's depth.
    Depth,
}

fn fragment_output(module: &Module, name: &str) -> FragmentOutput {
    let entry = module
        .entry_points
        .iter()
        .find(|e| e.name == name && e.stage == naga::ShaderStage::Fragment)
        .unwrap_or_else(|| unreachable!("no fragment entry point {name}"));
    let result = entry.function.result.as_ref().expect("a result");
    match (&result.binding, &module.types[result.ty].inner) {
        (
            Some(naga::Binding::Location { location, .. }),
            TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: naga::Scalar { kind, width: 4 },
            },
        ) => FragmentOutput::Color {
            location: *location,
            kind: *kind,
        },
        (Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth)), TypeInner::Scalar(F32)) => {
            FragmentOutput::Depth
        }
        other => unreachable!("{name} returns {other:?}"),
    }
}

#[test]
fn fragment_entry_points_write_their_stated_types() {
    let color = |kind| FragmentOutput::Color { location: 0, kind };
    let stated = [
        (
            HostShader::Present,
            present::FRAGMENT,
            color(naga::ScalarKind::Float),
        ),
        (
            HostShader::Clear,
            clear::FRAGMENT_COLOR_FLOAT,
            color(naga::ScalarKind::Float),
        ),
        (
            HostShader::Clear,
            clear::FRAGMENT_COLOR_SINT,
            color(naga::ScalarKind::Sint),
        ),
        (
            HostShader::Clear,
            clear::FRAGMENT_COLOR_UINT,
            color(naga::ScalarKind::Uint),
        ),
        (
            HostShader::Clear,
            clear::FRAGMENT_DEPTH,
            FragmentOutput::Depth,
        ),
    ];
    for (shader, name, output) in &stated {
        let (module, _) = checked(*shader);
        assert_eq!(fragment_output(&module, name), *output, "{name}");
    }
    // Every fragment entry point of every helper shader is in the table above.
    let fragment_entries = HostShader::ALL
        .iter()
        .flat_map(|s| s.entry_points())
        .filter(|e| e.stage == ShaderStage::Fragment)
        .count();
    assert_eq!(stated.len(), fragment_entries);
}

/// A present reads the view's base mip level whatever the scale: the present blit's one sample
/// names level 0 rather than leaving the level to the coordinate's derivatives.
#[test]
fn the_present_blit_samples_the_base_mip_level() {
    let (module, _) = checked(HostShader::Present);
    let function = &module
        .entry_points
        .iter()
        .find(|e| e.name == present::FRAGMENT)
        .expect("the fragment entry point")
        .function;
    let levels: Vec<_> = function
        .expressions
        .iter()
        .filter_map(|(_, expression)| match expression {
            naga::Expression::ImageSample { level, .. } => Some(*level),
            _ => None,
        })
        .collect();
    let [naga::SampleLevel::Exact(level)] = levels.as_slice() else {
        unreachable!("{levels:?}");
    };
    assert_eq!(
        function.expressions[*level],
        naga::Expression::Literal(naga::Literal::F32(0.0))
    );
}

/// The vertex stages take their positions from the vertex index alone, so the draws need no
/// vertex buffer.
#[test]
fn vertex_entry_points_read_only_the_vertex_index() {
    let mut vertex_entries = 0;
    for &shader in HostShader::ALL {
        let (module, _) = checked(shader);
        for entry in &module.entry_points {
            if entry.stage != naga::ShaderStage::Vertex {
                continue;
            }
            vertex_entries += 1;
            let bindings: Vec<_> = entry
                .function
                .arguments
                .iter()
                .map(|a| a.binding.clone())
                .collect();
            assert_eq!(
                bindings,
                [Some(naga::Binding::BuiltIn(naga::BuiltIn::VertexIndex))],
                "{} {}",
                shader.file_name(),
                entry.name
            );
        }
    }
    assert_eq!(vertex_entries, HostShader::ALL.len());
}
