//! The loader parses what the testkit's NRO builder writes.
//!
//! The two crates were written separately from the same reference structs, and the testkit may
//! not depend on the loader, so each restates the layout. These tests are the cross-check: every
//! field the builder is given comes back out of [`Nro::parse`], for generated inputs, and the
//! bytes the parsed segments name are the bytes the builder was handed. (The offsets the loader
//! keeps private are compared with the builder's in `nro.rs`'s unit tests.)

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_loader::nro::{HEADER_SIZE, MAGIC, MOD0_MAGIC, MOD0_SIZE, Nro, NroSegment};
use starquake_testkit::a64;
use starquake_testkit::nro::{self as built, NroBuilder};
use starquake_testkit::programs;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x4E52_4F5F_5854_4B54;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 128,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// The bytes of `file` a parsed segment names.
fn segment_bytes<'a>(file: &'a [u8], segment: &NroSegment) -> &'a [u8] {
    let start = usize::try_from(segment.offset).unwrap();
    &file[start..start + usize::try_from(segment.size).unwrap()]
}

/// Whether `segment` is a whole number of pages that starts with `content` and is zero after it.
fn holds(file: &[u8], segment: &NroSegment, content: &[u8]) -> bool {
    let bytes = segment_bytes(file, segment);
    bytes.len().is_multiple_of(built::PAGE_SIZE)
        && bytes.starts_with(content)
        && bytes[content.len()..].iter().all(|byte| *byte == 0)
}

#[test]
fn the_public_constants_agree() {
    assert_eq!(MAGIC, built::MAGIC);
    assert_eq!(HEADER_SIZE, built::HEADER_SIZE);
    assert_eq!(MOD0_MAGIC, built::MOD0_MAGIC);
    assert_eq!(MOD0_SIZE, built::MOD0_SIZE);
}

proptest! {
    #![proptest_config(config())]

    /// Every input of the builder is read back by the loader from where the builder put it.
    #[test]
    fn the_loader_reads_back_what_the_builder_was_given(
        code in proptest::collection::vec(any::<u32>(), 1..2000),
        rodata in proptest::collection::vec(any::<u8>(), 0..0x2800),
        data in proptest::collection::vec(any::<u8>(), 0..0x2800),
        bss_size in any::<u16>(),
        build_id in proptest::array::uniform32(any::<u8>()),
        homebrew in any::<bool>(),
    ) {
        let builder = NroBuilder::new(&code)
            .rodata(&rodata)
            .data(&data)
            .bss_size(u32::from(bss_size))
            .build_id(build_id)
            .homebrew(homebrew);
        let file = builder.build().unwrap();
        let nro = Nro::parse(&file).unwrap();

        prop_assert_eq!(usize::try_from(nro.size).unwrap(), file.len());
        let offsets = builder.segment_offsets().unwrap();
        prop_assert_eq!(nro.segments().map(|segment| segment.offset), offsets);
        // The segments tile the image in order, with nothing between or after them.
        prop_assert_eq!(nro.text.offset, 0);
        prop_assert_eq!(nro.text.offset + nro.text.size, nro.rodata.offset);
        prop_assert_eq!(nro.rodata.offset + nro.rodata.size, nro.data.offset);
        prop_assert_eq!(nro.data.offset + nro.data.size, nro.size);

        // The text segment holds the headers and then the code.
        let text = segment_bytes(&file, &nro.text);
        prop_assert!(text.len().is_multiple_of(built::PAGE_SIZE));
        let code_bytes = a64::to_bytes(&code);
        prop_assert_eq!(
            &text[built::CODE_OFFSET..built::CODE_OFFSET + code_bytes.len()],
            &code_bytes[..]
        );
        prop_assert!(holds(&file, &nro.rodata, &rodata), "rodata");
        prop_assert!(holds(&file, &nro.data, &data), "data");

        prop_assert_eq!(nro.bss_size, u32::from(bss_size));
        prop_assert_eq!(nro.build_id, build_id);
        prop_assert_eq!(nro.is_homebrew, homebrew);
        prop_assert_eq!(usize::try_from(nro.mod0_offset).unwrap(), built::MOD0_OFFSET);
        let mod0 = nro.mod0.expect("the builder writes a MOD0 header");
        prop_assert_eq!(mod0.bss_size(), Some(u32::from(bss_size)));
        // MOD0 offsets are relative to the MOD0 header: `.bss` starts where the image ends.
        prop_assert_eq!(mod0.bss_start_offset + nro.mod0_offset, nro.size);
        prop_assert_eq!(nro.assets, None);
    }
}

#[test]
fn the_loader_accepts_every_guest_program() {
    let files = [
        programs::hello().unwrap(),
        programs::counter(19_200_000).unwrap().nro,
        programs::crash().unwrap().nro,
        programs::endless_loop().unwrap(),
    ];
    for file in files {
        let nro = Nro::parse(&file).unwrap();
        assert_eq!(usize::try_from(nro.size).unwrap(), file.len());
        assert!(nro.is_homebrew);
        assert!(nro.mod0.is_some());
        // The first word is the entry branch to the code, which is inside the text segment.
        let entry = u32::from_le_bytes(file[..4].try_into().unwrap());
        assert_eq!(
            entry,
            a64::b(i64::try_from(built::CODE_OFFSET).unwrap()).unwrap()
        );
        assert!(built::CODE_OFFSET < usize::try_from(nro.text.size).unwrap());
    }
}
