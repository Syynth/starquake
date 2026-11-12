//! Golden bytes for a serialised list covering every command and every nested variant.

use super::{
    BindingResource, BlitDesc, BufferSlice, ClearDesc, ClearValue, Command, CommandList, CopyDesc,
    CopyRegion, DispatchDesc, DispatchParams, DrawDesc, DrawParams, DynamicValues,
    IndexBufferBinding, IndexFormat, Marker, PresentDesc, PresentTransform, Rect, RenderTargets,
    StagingSlice, TextureRegion, Viewport,
};
use crate::command::Binding;
use crate::id::{BufferId, PipelineId, ReadbackId, SamplerId, TextureId, ViewId};
use crate::pipeline::DepthBias;
use crate::resource::{Extent3d, FilterMode, Origin3d};
use std::collections::BTreeSet;

/// Hands out distinct non-zero values, so swapping any two fields changes the bytes.
struct Gen(u32);

impl Gen {
    fn u32(&mut self) -> u32 {
        self.0 += 1;
        self.0
    }
    fn u64(&mut self) -> u64 {
        // Above 32 bits, so a u64 field written as a u32 would show.
        u64::from(self.u32()) << 32 | u64::from(self.u32())
    }
    fn i32(&mut self) -> i32 {
        -i32::try_from(self.u32()).expect("small")
    }
    fn f32(&mut self) -> f32 {
        f32::from(u16::try_from(self.u32()).expect("small")) + 0.5
    }
    fn f4(&mut self) -> [f32; 4] {
        [self.f32(), self.f32(), self.f32(), self.f32()]
    }
    fn buffer(&mut self) -> BufferId {
        BufferId::new(self.u32(), self.u32())
    }
    fn view(&mut self) -> ViewId {
        ViewId::new(self.u32(), self.u32())
    }
    fn rect(&mut self) -> Rect {
        Rect {
            x: self.u32(),
            y: self.u32(),
            width: self.u32(),
            height: self.u32(),
        }
    }
    fn slice(&mut self) -> BufferSlice {
        BufferSlice {
            buffer: self.buffer(),
            offset: self.u64(),
            size: self.u64(),
        }
    }
    fn region(&mut self) -> TextureRegion {
        TextureRegion {
            texture: TextureId::new(self.u32(), self.u32()),
            mip: self.u32(),
            origin: Origin3d {
                x: self.u32(),
                y: self.u32(),
                z: self.u32(),
            },
            extent: Extent3d {
                width: self.u32(),
                height: self.u32(),
                depth_or_layers: self.u32(),
            },
        }
    }
    fn bindings(&mut self) -> Vec<Binding> {
        vec![
            Binding {
                group: self.u32(),
                binding: self.u32(),
                resource: BindingResource::Buffer(self.slice()),
            },
            Binding {
                group: self.u32(),
                binding: self.u32(),
                resource: BindingResource::View(self.view()),
            },
            Binding {
                group: self.u32(),
                binding: self.u32(),
                resource: BindingResource::Sampler(SamplerId::new(self.u32(), self.u32())),
            },
        ]
    }
    fn dynamic(&mut self) -> DynamicValues {
        DynamicValues {
            viewports: vec![Viewport {
                x: self.f32(),
                y: self.f32(),
                width: self.f32(),
                height: self.f32(),
                min_depth: self.f32(),
                max_depth: self.f32(),
            }],
            scissors: vec![self.rect()],
            blend_constant: self.f4(),
            stencil_reference: self.u32(),
            stencil_read_mask: self.u32(),
            stencil_write_mask: self.u32(),
            depth_bias: DepthBias {
                constant: self.f32(),
                slope_scale: self.f32(),
                clamp: self.f32(),
            },
            line_width: self.f32(),
        }
    }
}

/// A list with every command, and every variant of every enum inside one.
fn every_command() -> CommandList {
    let mut g = Gen(0);
    let mut list = CommandList::new();
    push_transfers(&mut g, &mut list);
    push_drawing(&mut g, &mut list);
    push_outputs(&mut g, &mut list);
    list
}

/// Uploads, copies, blits and clears.
fn push_transfers(g: &mut Gen, list: &mut CommandList) {
    let staged = StagingSlice::from_bytes(vec![0xA1, 0xB2, 0xC3]);
    list.push(Command::Upload {
        from: staged.clone(),
        to: CopyRegion::Buffer(g.slice()),
    });
    list.push(Command::Upload {
        from: staged,
        to: CopyRegion::Texture(g.region()),
    });
    list.push(Command::Copy(CopyDesc {
        src: CopyRegion::Buffer(g.slice()),
        dst: CopyRegion::Texture(g.region()),
    }));
    list.push(Command::Copy(CopyDesc {
        src: CopyRegion::Texture(g.region()),
        dst: CopyRegion::Buffer(g.slice()),
    }));
    for filter in [FilterMode::Linear, FilterMode::Nearest] {
        list.push(Command::Blit(BlitDesc {
            src: g.view(),
            src_rect: g.rect(),
            dst: g.view(),
            dst_rect: g.rect(),
            filter,
        }));
    }
    let clears = [
        ClearValue::Float(g.f4()),
        ClearValue::Sint([g.i32(), g.i32(), g.i32(), g.i32()]),
        ClearValue::Uint([g.u32(), g.u32(), g.u32(), g.u32()]),
        ClearValue::Depth(g.f32()),
        ClearValue::Stencil(g.u32()),
        ClearValue::DepthStencil {
            depth: g.f32(),
            stencil: g.u32(),
        },
    ];
    for (i, value) in clears.into_iter().enumerate() {
        let rect = (i != 0).then(|| g.rect());
        list.push(Command::Clear(ClearDesc {
            target: g.view(),
            value,
            rect,
        }));
    }
}

/// Render targets, draws and dispatches.
fn push_drawing(g: &mut Gen, list: &mut CommandList) {
    let mut colors = [None; crate::caps::GUEST_COLOR_TARGETS];
    colors[1] = Some(g.view());
    colors[6] = Some(g.view());
    list.push(Command::SetRenderTargets(RenderTargets {
        colors,
        depth_stencil: Some(g.view()),
    }));
    let params = [
        DrawParams::Direct {
            vertex_count: g.u32(),
            instance_count: g.u32(),
            first_vertex: g.u32(),
            first_instance: g.u32(),
        },
        DrawParams::Indexed {
            index_count: g.u32(),
            instance_count: g.u32(),
            first_index: g.u32(),
            base_vertex: g.i32(),
            first_instance: g.u32(),
        },
        DrawParams::Indirect {
            buffer: g.buffer(),
            offset: g.u64(),
            draw_count: g.u32(),
            stride: g.u32(),
            indexed: true,
        },
    ];
    let formats = [IndexFormat::U16, IndexFormat::U8, IndexFormat::U32];
    for (i, (params, format)) in params.into_iter().zip(formats).enumerate() {
        let index_buffer = (i != 0).then(|| IndexBufferBinding {
            slice: g.slice(),
            format,
        });
        // The first draw has no index buffer, so U16 is placed in a later list entry too.
        list.push(Command::Draw(DrawDesc {
            pipeline: PipelineId::new(g.u32(), g.u32()),
            bindings: g.bindings(),
            vertex_buffers: vec![g.slice(), g.slice()],
            index_buffer,
            dynamic: g.dynamic(),
            params,
        }));
    }
    list.push(Command::Draw(DrawDesc {
        pipeline: PipelineId::new(g.u32(), g.u32()),
        bindings: Vec::new(),
        vertex_buffers: Vec::new(),
        index_buffer: Some(IndexBufferBinding {
            slice: g.slice(),
            format: IndexFormat::U16,
        }),
        dynamic: g.dynamic(),
        params: DrawParams::Indexed {
            index_count: g.u32(),
            instance_count: g.u32(),
            first_index: g.u32(),
            base_vertex: g.i32(),
            first_instance: g.u32(),
        },
    }));
    let dispatches = [
        DispatchParams::Direct {
            x: g.u32(),
            y: g.u32(),
            z: g.u32(),
        },
        DispatchParams::Indirect {
            buffer: g.buffer(),
            offset: g.u64(),
        },
    ];
    for params in dispatches {
        list.push(Command::Dispatch(DispatchDesc {
            pipeline: PipelineId::new(g.u32(), g.u32()),
            bindings: g.bindings(),
            params,
        }));
    }
}

/// Readbacks, presents and a marker.
fn push_outputs(g: &mut Gen, list: &mut CommandList) {
    list.push(Command::Readback {
        from: CopyRegion::Buffer(g.slice()),
        id: ReadbackId::new(g.u32(), g.u32()),
    });
    list.push(Command::Readback {
        from: CopyRegion::Texture(g.region()),
        id: ReadbackId::new(g.u32(), g.u32()),
    });
    for transform in [
        PresentTransform::Rotate90,
        PresentTransform::Identity,
        PresentTransform::FlipHorizontal,
        PresentTransform::FlipVertical,
        PresentTransform::Rotate180,
        PresentTransform::Rotate270,
    ] {
        list.push(Command::Present(PresentDesc {
            view: g.view(),
            src_rect: g.rect(),
            transform,
            frame: g.u64(),
            dump: transform != PresentTransform::Identity,
        }));
    }
    list.push(Command::Marker(Marker {
        submit: g.u64(),
        draw: g.u64(),
    }));
}

/// Every variant name of every enum that appears in a list, collected by exhaustive matches, so
/// a new variant fails to compile here until the golden list covers it.
fn variants(list: &CommandList) -> BTreeSet<&'static str> {
    let mut seen = BTreeSet::new();
    let region = |seen: &mut BTreeSet<&'static str>, r: &CopyRegion| {
        seen.insert(match r {
            CopyRegion::Buffer(_) => "CopyRegion::Buffer",
            CopyRegion::Texture(_) => "CopyRegion::Texture",
        });
    };
    let binding = |seen: &mut BTreeSet<&'static str>, b: &Binding| {
        seen.insert(match b.resource {
            BindingResource::Buffer(_) => "BindingResource::Buffer",
            BindingResource::View(_) => "BindingResource::View",
            BindingResource::Sampler(_) => "BindingResource::Sampler",
        });
    };
    for command in &list.commands {
        match command {
            Command::Upload { to, .. } => {
                seen.insert("Upload");
                region(&mut seen, to);
            }
            Command::Copy(c) => {
                seen.insert("Copy");
                region(&mut seen, &c.src);
                region(&mut seen, &c.dst);
            }
            Command::Blit(b) => {
                seen.insert("Blit");
                seen.insert(match b.filter {
                    FilterMode::Nearest => "FilterMode::Nearest",
                    FilterMode::Linear => "FilterMode::Linear",
                });
            }
            Command::Clear(c) => {
                seen.insert("Clear");
                seen.insert(match c.value {
                    ClearValue::Float(_) => "ClearValue::Float",
                    ClearValue::Sint(_) => "ClearValue::Sint",
                    ClearValue::Uint(_) => "ClearValue::Uint",
                    ClearValue::Depth(_) => "ClearValue::Depth",
                    ClearValue::Stencil(_) => "ClearValue::Stencil",
                    ClearValue::DepthStencil { .. } => "ClearValue::DepthStencil",
                });
            }
            Command::SetRenderTargets(_) => {
                seen.insert("SetRenderTargets");
            }
            Command::Draw(d) => {
                seen.insert("Draw");
                d.bindings.iter().for_each(|b| binding(&mut seen, b));
                if let Some(ib) = d.index_buffer {
                    seen.insert(match ib.format {
                        IndexFormat::U8 => "IndexFormat::U8",
                        IndexFormat::U16 => "IndexFormat::U16",
                        IndexFormat::U32 => "IndexFormat::U32",
                    });
                }
                seen.insert(match d.params {
                    DrawParams::Direct { .. } => "DrawParams::Direct",
                    DrawParams::Indexed { .. } => "DrawParams::Indexed",
                    DrawParams::Indirect { .. } => "DrawParams::Indirect",
                });
            }
            Command::Dispatch(d) => {
                seen.insert("Dispatch");
                d.bindings.iter().for_each(|b| binding(&mut seen, b));
                seen.insert(match d.params {
                    DispatchParams::Direct { .. } => "DispatchParams::Direct",
                    DispatchParams::Indirect { .. } => "DispatchParams::Indirect",
                });
            }
            Command::Readback { from, .. } => {
                seen.insert("Readback");
                region(&mut seen, from);
            }
            Command::Present(p) => {
                seen.insert("Present");
                seen.insert(match p.transform {
                    PresentTransform::Identity => "PresentTransform::Identity",
                    PresentTransform::FlipHorizontal => "PresentTransform::FlipHorizontal",
                    PresentTransform::FlipVertical => "PresentTransform::FlipVertical",
                    PresentTransform::Rotate90 => "PresentTransform::Rotate90",
                    PresentTransform::Rotate180 => "PresentTransform::Rotate180",
                    PresentTransform::Rotate270 => "PresentTransform::Rotate270",
                });
            }
            Command::Marker(_) => {
                seen.insert("Marker");
            }
        }
    }
    seen
}

#[test]
fn golden_list_covers_every_variant() {
    let expected: BTreeSet<&str> = [
        "Upload",
        "Copy",
        "Blit",
        "Clear",
        "SetRenderTargets",
        "Draw",
        "Dispatch",
        "Readback",
        "Present",
        "Marker",
        "CopyRegion::Buffer",
        "CopyRegion::Texture",
        "FilterMode::Nearest",
        "FilterMode::Linear",
        "ClearValue::Float",
        "ClearValue::Sint",
        "ClearValue::Uint",
        "ClearValue::Depth",
        "ClearValue::Stencil",
        "ClearValue::DepthStencil",
        "BindingResource::Buffer",
        "BindingResource::View",
        "BindingResource::Sampler",
        "IndexFormat::U8",
        "IndexFormat::U16",
        "IndexFormat::U32",
        "DrawParams::Direct",
        "DrawParams::Indexed",
        "DrawParams::Indirect",
        "DispatchParams::Direct",
        "DispatchParams::Indirect",
        "PresentTransform::Identity",
        "PresentTransform::FlipHorizontal",
        "PresentTransform::FlipVertical",
        "PresentTransform::Rotate90",
        "PresentTransform::Rotate180",
        "PresentTransform::Rotate270",
    ]
    .into_iter()
    .collect();
    assert_eq!(variants(&every_command()), expected);
}

/// The postcard encoding of [`every_command`], in hex. Changing it changes the capture layout:
/// slice R11's capture header then needs a new version.
const GOLDEN: &str = concat!(
    "1c0003a1b2c3000102848080803086808080500003a1b2c3010708090a0b0c0d0e0f0100101193808080a002",
    "95808080c00201161718191a1b1c1d1e01011f2021222324252627002829ab808080a005ad808080c005022e",
    "2f3031323334353637383901023a3b3c3d3e3f404142434445000356570000008d4200008f42000091420000",
    "934200035c5d0193019501970199010158595a5b036263024e4f5051015e5f6061036869030000a542016465",
    "6667036e6f0453016a6b6c6d037475050000a94255017071727304000176770000000001787900017a7b058b",
    "018c01038d018e01008f01900192818080901294818080b01295019601019701980199019a01029b019c0102",
    "9d019e01a0818080f013a28180809014a301a401a6818080d014a8818080f01400010080294300802a430080",
    "2b4300802c4300802d4300802e4301af01b001b101b20100803343008034430080354300803643b701b801b9",
    "0100803a4300803b4300803c4300803d43007c7d7e7f05c401c50103c601c70100c801c901cb818080a019cd",
    "818080c019ce01cf0101d001d101d201d30102d401d50102d601d701d9818080801bdb818080a01bdc01dd01",
    "df818080e01be1818080801c01be01bf01c18180808018c3818080a018000100806243008063430080644300",
    "806543008066430080674301e801e901ea01eb0100806c4300806d4300806e4300806f43f001f101f2010080",
    "7343008074430080754300807643018001810182018502840105fd01fe0103ff018002008102820284828080",
    "b02086828080d020870288020189028a028b028c02028d028e02028f02900292828080902294828080b02295",
    "02960298828080f0229a828080902301f701f801fa818080901ffc818080b01f020100c08d4300408e4300c0",
    "8e4300408f4300c08f430040904301a102a202a302a40200c092430040934300c0934300409443a902aa02ab",
    "020040964300c096430040974300c09743028501860188818080f01089018a010105b002b102000001b202b3",
    "02b5828080c026b7828080e026010100409c4300c09c4300409d4300c09d4300409e4300c09e4301be02bf02",
    "c002c1020040a14300c0a1430040a24300c0a243c602c702c80200c0a4430040a54300c0a5430040a64301cd",
    "02ce02cf029f05d10206d902da0203db02dc0200dd02de02e0828080f02be2828080902ce302e40201e502e6",
    "02e702e80202e902ea0200d202d302d40206eb02ec0203ed02ee0200ef02f002f2828080902ef4828080b02e",
    "f502f60201f702f802f902fa0202fb02fc0201d502d602d8828080f02a0700fd02fe0280838080f02f828380",
    "809030830384030701850386038703880389038a038b038c038d038e038f0308900391039203930394039503",
    "0397838080e0320108980399039a039b039c039d03009f838080e0330008a003a103a203a303a403a50301a7",
    "838080e0340108a803a903aa03ab03ac03ad0302af838080e0350108b003b103b203b303b403b50304b78380",
    "80e0360108b803b903ba03bb03bc03bd0305bf838080e0370109c18380808038c3838080a038",
);

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").expect("writing to a string");
        s
    })
}

#[test]
fn golden_bytes() {
    let bytes = postcard::to_allocvec(&every_command()).expect("serialise");
    assert_eq!(hex(&bytes), GOLDEN);
}

#[test]
fn golden_bytes_decode_to_the_list() {
    let bytes: Vec<u8> = (0..GOLDEN.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&GOLDEN[i..i + 2], 16).expect("hex"))
        .collect();
    let list: CommandList = postcard::from_bytes(&bytes).expect("deserialise");
    assert_eq!(list, every_command());
}

#[test]
fn truncated_bytes_are_an_error() {
    let bytes = postcard::to_allocvec(&every_command()).expect("serialise");
    for len in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            postcard::from_bytes::<CommandList>(&bytes[..len]).is_err(),
            "{len} bytes"
        );
    }
}

#[test]
fn staging_slice_accessors() {
    let mut s = StagingSlice::from_bytes(vec![0; 4]);
    assert_eq!(s.len(), 4);
    assert!(!s.is_empty());
    s.bytes_mut()[2] = 9;
    assert_eq!(s.bytes(), &[0, 0, 9, 0]);
    assert!(StagingSlice::from_bytes(Vec::new()).is_empty());
}
