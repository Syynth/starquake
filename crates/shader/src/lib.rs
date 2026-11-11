//! SM86 shader translation.
//!
//! Decodes SASS into an IR, recovers control flow, and lowers to host shaders through `naga`
//! (SPIR-V for Vulkan, MSL for Metal). Instruction encodings come from Mesa NAK (MIT); each
//! instruction gets fixtures checked against `nvdisasm`.
