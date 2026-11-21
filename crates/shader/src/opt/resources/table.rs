//! The resource table the texture pass produces: where each texture handle is found, and the
//! four kinds of descriptor Eden keeps (`eden/src/shader_recompiler/shader_info.h` @ cf654052:
//! `TextureBufferDescriptor`, `ImageBufferDescriptor`, `TextureDescriptor`, `ImageDescriptor`),
//! merged as Eden's `Descriptors::Add` merges them (`ir_opt/texture_pass.cpp`).

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::hash::Hash;

use crate::exchange::{SampledType, TextureFormat};
use crate::ir::{ImageFormat, TextureType};

/// One constant-buffer word a texture handle is built from: `(word << shift_left) & mask`
/// (Eden's `ConstBufferAddr` primary or secondary fields, with the mask "applied after the
/// shifts", `environment.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandleWord {
    bank: u8,
    offset: u32,
    shift_left: u32,
    mask: u32,
}

impl HandleWord {
    /// A whole word, unshifted and unmasked. `bank` is below
    /// [`CBUF_BANKS`](crate::translate::float::CBUF_BANKS) and `offset` a multiple of four,
    /// which the tracer checks; `shift_left` is below 32.
    pub(super) const fn new(bank: u8, offset: u32) -> Self {
        Self {
            bank,
            offset,
            shift_left: 0,
            mask: u32::MAX,
        }
    }

    /// This word shifted left by `shift` (below 32); the mask moves with it (Eden: "A mask
    /// applied before the shift moves with the value").
    pub(super) const fn shifted(self, shift: u32) -> Self {
        Self {
            shift_left: shift,
            mask: self.mask.wrapping_shl(shift),
            ..self
        }
    }

    /// This word with `mask` applied as well.
    pub(super) const fn masked(self, mask: u32) -> Self {
        Self {
            mask: self.mask & mask,
            ..self
        }
    }

    /// The constant-buffer bank.
    pub const fn bank(self) -> u8 {
        self.bank
    }

    /// The word's byte offset in the bank, a multiple of four.
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// How far the word is shifted left, below 32.
    pub const fn shift_left(self) -> u32 {
        self.shift_left
    }

    /// The bits kept after the shift.
    pub const fn mask(self) -> u32 {
        self.mask
    }

    /// The handle bits this word gives when it holds `word`.
    pub const fn apply(self, word: u32) -> u32 {
        word.wrapping_shl(self.shift_left) & self.mask
    }
}

impl Display for HandleWord {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "cbuf {} at {:#x}", self.bank, self.offset)?;
        if self.shift_left != 0 {
            write!(f, " << {}", self.shift_left)?;
        }
        if self.mask != u32::MAX {
            write!(f, " & {:#x}", self.mask)?;
        }
        Ok(())
    }
}

/// A constant-buffer word that moves a handle array's index for the whole draw: the array's
/// byte offset is `word << shift` (Eden's `has_index_cbuf`, `index_cbuf_index`,
/// `index_cbuf_offset`, `index_shift`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexWord {
    bank: u8,
    offset: u32,
    shift: u32,
}

impl IndexWord {
    /// The index word at `offset` (a multiple of four) of `bank` (below 32), shifted by
    /// `shift` (below 32).
    pub(super) const fn new(bank: u8, offset: u32, shift: u32) -> Self {
        Self {
            bank,
            offset,
            shift,
        }
    }

    /// The constant-buffer bank.
    pub const fn bank(self) -> u8 {
        self.bank
    }

    /// The word's byte offset in the bank, a multiple of four.
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// How far the word is shifted left to give the array's byte offset.
    pub const fn shift(self) -> u32 {
        self.shift
    }
}

/// Where a texture handle is found: one word or two OR-ed together, and for an array of
/// handles indexed at run time, how many there are and how far apart (Eden's
/// `ConstBufferAddr`, less the dynamic offset, which stays in the IR).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandleSource {
    primary: HandleWord,
    secondary: Option<HandleWord>,
    count: u32,
    size_shift: u32,
    index: Option<IndexWord>,
}

impl HandleSource {
    /// A single handle in `primary`.
    pub(super) const fn single(primary: HandleWord) -> Self {
        Self {
            primary,
            secondary: None,
            count: 1,
            size_shift: super::DESCRIPTOR_SIZE_SHIFT,
            index: None,
        }
    }

    /// An array of `count` (at least 1) handles starting at `primary`, `1 << size_shift`
    /// bytes apart, its index possibly moved by `index`. An array of one handle is spaced
    /// [`DESCRIPTOR_SIZE_SHIFT`](super::DESCRIPTOR_SIZE_SHIFT), as Eden's descriptors are
    /// (`size_shift = cbuf.count > 1 ? DynamicDescriptorSizeShift(..) :
    /// DESCRIPTOR_SIZE_SHIFT`).
    pub(super) const fn array(
        primary: HandleWord,
        count: u32,
        size_shift: u32,
        index: Option<IndexWord>,
    ) -> Self {
        Self {
            primary,
            secondary: None,
            count,
            size_shift: if count > 1 {
                size_shift
            } else {
                super::DESCRIPTOR_SIZE_SHIFT
            },
            index,
        }
    }

    /// This source with `primary` replaced (a shift or a mask applied to it).
    pub(super) const fn with_primary(self, primary: HandleWord) -> Self {
        Self { primary, ..self }
    }

    /// This source with `secondary` OR-ed into the handle.
    pub(super) const fn with_secondary(self, secondary: HandleWord) -> Self {
        Self {
            secondary: Some(secondary),
            ..self
        }
    }

    /// This source with the count and index a descriptor of its kind keeps.
    pub(super) const fn with_count(self, count: u32, index: Option<IndexWord>) -> Self {
        Self {
            count,
            index,
            ..self
        }
    }

    /// The first word.
    pub const fn primary(&self) -> HandleWord {
        self.primary
    }

    /// The word OR-ed into the first, if any.
    pub const fn secondary(&self) -> Option<HandleWord> {
        self.secondary
    }

    /// How many handles the array holds; 1 for a single handle.
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The handles of an array are `1 << size_shift` bytes apart.
    pub const fn size_shift(&self) -> u32 {
        self.size_shift
    }

    /// The word that moves the array's index for the whole draw, if any.
    pub const fn index(&self) -> Option<IndexWord> {
        self.index
    }

    /// The words the handle is read from: the first, then the second.
    pub fn words(&self) -> impl Iterator<Item = HandleWord> {
        std::iter::once(self.primary).chain(self.secondary)
    }

    /// The handle (of the array's first element), given the words: `None` when a word is
    /// missing (Eden's `GetTextureHandle`).
    pub fn handle(&self, mut word: impl FnMut(u8, u32) -> Option<u32>) -> Option<u32> {
        let mut handle = 0;
        for part in self.words() {
            handle |= part.apply(word(part.bank, part.offset)?);
        }
        Some(handle)
    }
}

impl Display for HandleSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.primary)?;
        if let Some(secondary) = self.secondary {
            write!(f, " | {secondary}")?;
        }
        if self.count > 1 {
            write!(
                f,
                ", {} handles {} bytes apart",
                self.count,
                1_u64 << self.size_shift
            )?;
        }
        if let Some(index) = self.index {
            write!(
                f,
                ", index from cbuf {} at {:#x} << {}",
                index.bank, index.offset, index.shift
            )?;
        }
        Ok(())
    }
}

/// A sampled texel buffer (Eden's `TextureBufferDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureBufferDescriptor {
    /// Where the handle is found.
    pub source: HandleSource,
    /// The component type, from the `TextureKind` answer (Eden keeps none for a buffer).
    pub component: SampledType,
}

/// A storage texel buffer (Eden's `ImageBufferDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageBufferDescriptor {
    /// Where the handle is found.
    pub source: HandleSource,
    /// The format the instruction names.
    pub image_format: ImageFormat,
    /// The header's format bits, from the `TextureKind` answer to the image-usage question.
    pub header_format: TextureFormat,
    /// The component type, from the `TextureKind` answer.
    pub component: SampledType,
    /// Some use writes it.
    pub is_written: bool,
    /// Some use reads it.
    pub is_read: bool,
}

/// A sampled texture (Eden's `TextureDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    /// Where the handle is found.
    pub source: HandleSource,
    /// The dimension, from the instruction.
    pub ty: TextureType,
    /// A depth comparison, from the instruction.
    pub is_depth: bool,
    /// Some use fetches it with a sample index, from the instruction.
    pub is_multisample: bool,
    /// The component type, from the `TextureKind` answer.
    pub component: SampledType,
}

/// A storage image (Eden's `ImageDescriptor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDescriptor {
    /// Where the handle is found.
    pub source: HandleSource,
    /// The dimension, from the instruction.
    pub ty: TextureType,
    /// The format the instruction names.
    pub image_format: ImageFormat,
    /// The header's format bits, from the `TextureKind` answer to the image-usage question.
    pub header_format: TextureFormat,
    /// The component type, from the `TextureKind` answer.
    pub component: SampledType,
    /// Some use writes it.
    pub is_written: bool,
    /// Some use reads it.
    pub is_read: bool,
}

/// The shader's texture resources: each texture operation's `descriptor_index` is an index
/// into the list of its kind.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceTable {
    texture_buffers: Vec<TextureBufferDescriptor>,
    image_buffers: Vec<ImageBufferDescriptor>,
    textures: Vec<TextureDescriptor>,
    images: Vec<ImageDescriptor>,
}

impl ResourceTable {
    /// The sampled texel buffers.
    pub fn texture_buffers(&self) -> &[TextureBufferDescriptor] {
        &self.texture_buffers
    }

    /// The storage texel buffers.
    pub fn image_buffers(&self) -> &[ImageBufferDescriptor] {
        &self.image_buffers
    }

    /// The sampled textures.
    pub fn textures(&self) -> &[TextureDescriptor] {
        &self.textures
    }

    /// The storage images.
    pub fn images(&self) -> &[ImageDescriptor] {
        &self.images
    }

    /// Whether the table has no descriptor.
    pub fn is_empty(&self) -> bool {
        self.texture_buffers.is_empty()
            && self.image_buffers.is_empty()
            && self.textures.is_empty()
            && self.images.is_empty()
    }
}

/// The index of the entry `key` names in `keys`, or of `desc` pushed onto `list` (Eden's
/// `Descriptors::Add`, which scans the list; looked up here, so adding stays constant time
/// however many descriptors a hostile program makes), and whether it merged.
fn add<K: Eq + Hash, T>(
    keys: &mut HashMap<K, usize>,
    list: &mut Vec<T>,
    key: K,
    desc: T,
) -> (usize, bool) {
    if let Some(&index) = keys.get(&key) {
        return (index, true);
    }
    let index = list.len();
    list.push(desc);
    keys.insert(key, index);
    (index, false)
}

/// Builds a [`ResourceTable`], merging descriptors as Eden's `Descriptors::Add` does: each
/// kind's merge key is the fields Eden's predicate compares, and a merged entry takes the
/// flags Eden ORs in. The maps are only looked up, never iterated, so their order cannot reach
/// the table.
#[derive(Default)]
pub(super) struct TableBuilder {
    table: ResourceTable,
    texture_buffers: HashMap<HandleSource, usize>,
    image_buffers: HashMap<(ImageFormat, HandleSource), usize>,
    textures: HashMap<(TextureType, bool, HandleSource), usize>,
    images: HashMap<(TextureType, ImageFormat, HandleSource), usize>,
}

impl TableBuilder {
    /// The table built.
    pub(super) fn finish(self) -> ResourceTable {
        self.table
    }

    /// Adds `desc` or finds the entry it merges into: Eden merges texel buffers with the same
    /// source.
    pub(super) fn add_texture_buffer(&mut self, desc: TextureBufferDescriptor) -> usize {
        let list = &mut self.table.texture_buffers;
        add(&mut self.texture_buffers, list, desc.source, desc).0
    }

    /// Adds `desc` or merges it into an entry with the same format and source, whose read and
    /// written flags take `desc`'s too.
    pub(super) fn add_image_buffer(&mut self, desc: ImageBufferDescriptor) -> usize {
        let list = &mut self.table.image_buffers;
        let key = (desc.image_format, desc.source);
        let (index, merged) = add(&mut self.image_buffers, list, key, desc);
        if merged && let Some(entry) = list.get_mut(index) {
            entry.is_written |= desc.is_written;
            entry.is_read |= desc.is_read;
        }
        index
    }

    /// Adds `desc` or merges it into an entry with the same dimension, depth flag and source,
    /// which takes `desc`'s multisample flag too.
    pub(super) fn add_texture(&mut self, desc: TextureDescriptor) -> usize {
        let list = &mut self.table.textures;
        let key = (desc.ty, desc.is_depth, desc.source);
        let (index, merged) = add(&mut self.textures, list, key, desc);
        if merged && let Some(entry) = list.get_mut(index) {
            entry.is_multisample |= desc.is_multisample;
        }
        index
    }

    /// Adds `desc` or merges it into an entry with the same dimension, format and source,
    /// whose read and written flags take `desc`'s too.
    pub(super) fn add_image(&mut self, desc: ImageDescriptor) -> usize {
        let list = &mut self.table.images;
        let key = (desc.ty, desc.image_format, desc.source);
        let (index, merged) = add(&mut self.images, list, key, desc);
        if merged && let Some(entry) = list.get_mut(index) {
            entry.is_written |= desc.is_written;
            entry.is_read |= desc.is_read;
        }
        index
    }
}

/// A component type in words.
const fn component_name(component: SampledType) -> &'static str {
    match component {
        SampledType::Float => "float",
        SampledType::SignedInt => "signed",
        SampledType::UnsignedInt => "unsigned",
    }
}

/// `read`, `written`, both or neither, in words.
const fn access_name(is_read: bool, is_written: bool) -> &'static str {
    match (is_read, is_written) {
        (true, true) => "read and written",
        (true, false) => "read",
        (false, true) => "written",
        (false, false) => "unused",
    }
}

impl Display for ResourceTable {
    /// One line per descriptor, kind by kind, in index order.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (index, d) in self.texture_buffers.iter().enumerate() {
            writeln!(
                f,
                "texture buffer {index}: {}, from {}",
                component_name(d.component),
                d.source
            )?;
        }
        for (index, d) in self.image_buffers.iter().enumerate() {
            writeln!(
                f,
                "image buffer {index}: {}, {}, header format {:#x}, {}, from {}",
                d.image_format.name(),
                component_name(d.component),
                d.header_format.bits(),
                access_name(d.is_read, d.is_written),
                d.source
            )?;
        }
        for (index, d) in self.textures.iter().enumerate() {
            write!(
                f,
                "texture {index}: {}, {}",
                d.ty.name(),
                component_name(d.component)
            )?;
            if d.is_depth {
                f.write_str(", depth")?;
            }
            if d.is_multisample {
                f.write_str(", multisampled")?;
            }
            writeln!(f, ", from {}", d.source)?;
        }
        for (index, d) in self.images.iter().enumerate() {
            writeln!(
                f,
                "image {index}: {}, {}, {}, header format {:#x}, {}, from {}",
                d.ty.name(),
                d.image_format.name(),
                component_name(d.component),
                d.header_format.bits(),
                access_name(d.is_read, d.is_written),
                d.source
            )?;
        }
        Ok(())
    }
}
