//! Building the bytes a service lays a shared memory out with, before it writes them into the
//! memory in one write: shared by [`crate::hid::shared_memory`] and
//! [`crate::time::shared_memory`].

/// A zeroed image of `size` bytes. Every caller's size is a constant of a few pages, which fits
/// any host's `usize`.
pub(crate) fn blank(size: u64) -> Vec<u8> {
    vec![0; usize::try_from(size).unwrap_or(usize::MAX)]
}

/// Copies `bytes` into `image` at `offset`. Every caller writes constant fields of a constant
/// layout that its golden-bytes test checks, so a write that did not lie inside the image would
/// be a mistake in that layout; it is skipped and logged, naming `memory`, rather than allowed
/// to panic.
pub(crate) fn put(image: &mut [u8], offset: u64, bytes: &[u8], memory: &str) {
    let target = usize::try_from(offset).ok().and_then(|start| {
        let end = start.checked_add(bytes.len())?;
        image.get_mut(start..end)
    });
    if let Some(target) = target {
        target.copy_from_slice(bytes);
    } else {
        tracing::error!(offset, memory, "a shared memory field lies past its end");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_is_written_only_where_it_fits_whole() {
        let mut image = blank(4);
        put(&mut image, 2, &[7, 8], "test");
        assert_eq!(image, [0, 0, 7, 8]);
        // One byte past the end, and an offset past the end: nothing is written.
        put(&mut image, 3, &[9, 9], "test");
        put(&mut image, 5, &[9], "test");
        put(&mut image, u64::MAX, &[9], "test");
        assert_eq!(image, [0, 0, 7, 8]);
    }
}
