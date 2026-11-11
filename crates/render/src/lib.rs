//! Host rendering backends behind one trait.
//!
//! Vulkan (Linux, and anywhere else) and Metal (macOS). Owns pipelines, resources, sparse
//! memory, GOB deswizzling and presentation. Metal samples ASTC natively; Vulkan on AMD needs
//! transcoding.
