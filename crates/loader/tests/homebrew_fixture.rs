//! The `tests/homebrew/hello` fixture: a real libnx program built with devkitPro
//! (`tests/homebrew/README.md`).
//!
//! Two jobs:
//!
//! - **The manifest check** (test strategy spec, `docs/specs/testing.md`, "Homebrew fixtures"):
//!   every source and binary in the folder must hash to what `manifest.toml` records, and every
//!   file must be listed, so a source can't change without a rebuild. This is what the gate runs;
//!   it needs no toolchain. The `tests/homebrew/startup` fixture's manifest is checked the same
//!   way.
//! - **The loader against real files**: the NSO, the NRO and the NPDM parse, and what they say
//!   matches `hello.json` (the NPDM description `npdmtool` was given), the ELF build id the
//!   manifest records, and each other (the NRO and the NSO come from one ELF). These are the
//!   checks that confirmed, or corrected, the facts the loader used to mark as guesses.

/// Everything is in a `cfg(test)` module so the helpers may unwrap like the tests do.
#[cfg(test)]
mod homebrew {

    use std::collections::BTreeSet;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use starquake_loader::npdm::capabilities::{
        DebugFlags, EDEN_CAPABILITY_CORE_LIMIT, KernelCapabilities, KernelVersion, ThreadInfo,
    };
    use starquake_loader::npdm::service_access::ServiceAccess;
    use starquake_loader::npdm::{AcidLayout, AddressSpaceType, Npdm, PoolPartition};
    use starquake_loader::nro::{self, Nro, NroSegment};
    use starquake_loader::nso::{
        FLAG_COMPRESSED_SHIFT, FLAG_HASH_CHECK_SHIFT, Nso, NsoError, NsoSegment, SEGMENT_COUNT,
    };

    /// The manifest's file name.
    const MANIFEST: &str = "manifest.toml";
    /// Tables of the manifest whose entries are `"file" = "sha256"`.
    const HASHED_TABLES: [&str; 2] = ["sources", "binaries"];
    /// Keys the `[toolchain]` table must have, as the fixture's Makefile writes them.
    const TOOLCHAIN_KEYS: [&str; 4] = ["gcc", "devkitA64", "libnx", "switch-tools"];
    /// Bytes of a GNU build id made with `--build-id=sha1` (`switch.specs` in libnx).
    const SHA1_SIZE: usize = 20;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/homebrew/hello")
    }

    fn read(name: &str) -> Vec<u8> {
        let path = fixture_dir().join(name);
        fs::read(&path)
            .map_err(|error| format!("reading {}: {error}", path.display()))
            .unwrap()
    }

    fn manifest() -> toml::Table {
        let text = String::from_utf8(read(MANIFEST)).unwrap();
        text.parse().unwrap()
    }

    fn description() -> Value {
        serde_json::from_slice(&read("hello.json")).unwrap()
    }

    // ----- The manifest check -------------------------------------------------------------------

    /// Why a fixture folder does not match its manifest.
    #[derive(Debug, PartialEq, Eq)]
    enum ManifestError {
        /// The manifest is missing, unreadable or not TOML.
        Unreadable,
        /// A table or key is missing or has the wrong type.
        Malformed(String),
        /// A listed file is not in the folder.
        Missing(String),
        /// A file's SHA-256 is not the one recorded.
        HashMismatch(String),
        /// A file in the folder is not listed.
        Unlisted(String),
    }

    /// Checks the fixture folder `dir` against its manifest. Directories (`build/`) and dot files
    /// (`.DS_Store`) are not fixture files and are skipped.
    fn check_manifest(dir: &Path) -> Result<(), ManifestError> {
        let text = fs::read_to_string(dir.join(MANIFEST)).map_err(|_| ManifestError::Unreadable)?;
        let manifest: toml::Table = text.parse().map_err(|_| ManifestError::Unreadable)?;

        let toolchain = manifest
            .get("toolchain")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| ManifestError::Malformed("toolchain".into()))?;
        for key in TOOLCHAIN_KEYS {
            if toolchain
                .get(key)
                .and_then(toml::Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err(ManifestError::Malformed(format!("toolchain.{key}")));
            }
        }

        let mut listed = BTreeSet::from([MANIFEST.to_owned()]);
        for table_name in HASHED_TABLES {
            let table = manifest
                .get(table_name)
                .and_then(toml::Value::as_table)
                .ok_or_else(|| ManifestError::Malformed(table_name.into()))?;
            for (file, hash) in table {
                let hash = hash
                    .as_str()
                    .ok_or_else(|| ManifestError::Malformed(format!("{table_name}.{file}")))?;
                let bytes =
                    fs::read(dir.join(file)).map_err(|_| ManifestError::Missing(file.clone()))?;
                if hex(&Sha256::digest(&bytes)) != hash {
                    return Err(ManifestError::HashMismatch(file.clone()));
                }
                listed.insert(file.clone());
            }
        }

        let mut files: Vec<String> = fs::read_dir(dir)
            .map_err(|_| ManifestError::Unreadable)?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| !name.starts_with('.'))
            .collect();
        files.sort();
        match files.into_iter().find(|name| !listed.contains(name)) {
            Some(name) => Err(ManifestError::Unlisted(name)),
            None => Ok(()),
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").unwrap();
            out
        })
    }

    #[test]
    fn the_fixture_matches_its_manifest() {
        assert_eq!(
            check_manifest(&fixture_dir()),
            Ok(()),
            "tests/homebrew/hello no longer matches its manifest; rebuild it with `make` in that \
             folder and commit the result (tests/homebrew/README.md)"
        );
    }

    /// The second fixture (`tests/homebrew/startup`, libnx's own start-up) has a manifest of the
    /// same form, checked the same way.
    #[test]
    fn the_startup_fixture_matches_its_manifest() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/homebrew/startup");
        assert_eq!(
            check_manifest(&dir),
            Ok(()),
            "tests/homebrew/startup no longer matches its manifest; rebuild it with `make` in \
             that folder and commit the result (tests/homebrew/README.md)"
        );
    }

    /// A copy of the fixture folder in a folder unique to one test, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("starquake-homebrew-{}-{test}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            for entry in fs::read_dir(fixture_dir()).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_file() {
                    fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
                }
            }
            Self(dir)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Every file the manifest hashes, from the real manifest.
    fn hashed_files() -> Vec<String> {
        let manifest = manifest();
        HASHED_TABLES
            .iter()
            .flat_map(|table| manifest[*table].as_table().unwrap().keys().cloned())
            .collect()
    }

    #[test]
    fn an_edited_source_or_binary_fails_the_manifest_check() {
        let files = hashed_files();
        assert!(files.iter().any(|file| file == "hello.c"), "{files:?}");
        for file in files {
            let scratch = Scratch::new(&format!("edit-{file}"));
            assert_eq!(check_manifest(&scratch.0), Ok(()));
            let mut bytes = fs::read(scratch.path(&file)).unwrap();
            bytes.push(b'\n');
            fs::write(scratch.path(&file), bytes).unwrap();
            assert_eq!(
                check_manifest(&scratch.0),
                Err(ManifestError::HashMismatch(file))
            );
        }
    }

    #[test]
    fn a_missing_or_unlisted_file_fails_the_manifest_check() {
        let scratch = Scratch::new("missing");
        fs::remove_file(scratch.path("hello.nso")).unwrap();
        assert_eq!(
            check_manifest(&scratch.0),
            Err(ManifestError::Missing("hello.nso".into()))
        );

        let scratch = Scratch::new("unlisted");
        fs::write(scratch.path("extra.c"), b"int x;\n").unwrap();
        assert_eq!(
            check_manifest(&scratch.0),
            Err(ManifestError::Unlisted("extra.c".into()))
        );
        // A build folder and dot files are not fixture files.
        fs::remove_file(scratch.path("extra.c")).unwrap();
        fs::create_dir(scratch.path("build")).unwrap();
        fs::write(scratch.path("build/hello.elf"), b"elf").unwrap();
        fs::write(scratch.path(".DS_Store"), b"finder").unwrap();
        assert_eq!(check_manifest(&scratch.0), Ok(()));
    }

    /// Deleting the key outright, not only emptying its value (the test below).
    #[test]
    fn a_manifest_with_a_toolchain_key_deleted_fails_the_check() {
        let text = String::from_utf8(read(MANIFEST)).unwrap();
        for key in TOOLCHAIN_KEYS {
            let scratch = Scratch::new(&format!("toolchain-deleted-{key}"));
            let prefix = format!("{key} = ");
            let kept: Vec<&str> = text
                .lines()
                .filter(|line| !line.starts_with(&prefix))
                .collect();
            assert_eq!(kept.len() + 1, text.lines().count(), "no `{key}` line");
            fs::write(scratch.path(MANIFEST), kept.join("\n")).unwrap();
            assert_eq!(
                check_manifest(&scratch.0),
                Err(ManifestError::Malformed(format!("toolchain.{key}")))
            );
        }
    }

    #[test]
    fn a_manifest_without_a_toolchain_version_fails_the_check() {
        let text = String::from_utf8(read(MANIFEST)).unwrap();
        for key in TOOLCHAIN_KEYS {
            let scratch = Scratch::new(&format!("toolchain-{key}"));
            let prefix = format!("{key} = ");
            let emptied: Vec<String> = text
                .lines()
                .map(|line| {
                    if line.starts_with(&prefix) {
                        format!("{prefix}\"\"")
                    } else {
                        line.to_owned()
                    }
                })
                .collect();
            assert_ne!(emptied.join("\n"), text.trim_end(), "no `{key}` line");
            fs::write(scratch.path(MANIFEST), emptied.join("\n")).unwrap();
            assert_eq!(
                check_manifest(&scratch.0),
                Err(ManifestError::Malformed(format!("toolchain.{key}")))
            );
        }
        let scratch = Scratch::new("no-binaries");
        let binaries = text.find("[binaries]").unwrap();
        fs::write(scratch.path(MANIFEST), &text[..binaries]).unwrap();
        assert_eq!(
            check_manifest(&scratch.0),
            Err(ManifestError::Malformed("binaries".into()))
        );
        fs::remove_file(scratch.path(MANIFEST)).unwrap();
        assert_eq!(check_manifest(&scratch.0), Err(ManifestError::Unreadable));
    }

    // ----- The loader against the real files ----------------------------------------------------

    /// The ELF's GNU build id, as the manifest records it, padded to the 0x20-byte field.
    fn elf_build_id() -> [u8; 0x20] {
        let text = manifest()["elf"]["build-id"].as_str().unwrap().to_owned();
        assert_eq!(text.len(), SHA1_SIZE * 2, "{text}");
        let mut id = [0u8; 0x20];
        for (index, byte) in id.iter_mut().take(SHA1_SIZE).enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap();
        }
        id
    }

    /// The string literals `hello.c` passes to `svcOutputDebugString`: every `static const char`
    /// initialiser (none of them has an escape).
    fn program_strings() -> Vec<Vec<u8>> {
        let source = String::from_utf8(read("hello.c")).unwrap();
        let strings: Vec<Vec<u8>> = source
            .lines()
            .filter(|line| line.starts_with("static const char"))
            .map(|line| {
                let start = line.find('"').unwrap() + 1;
                let end = line.rfind('"').unwrap();
                line.as_bytes()[start..end].to_vec()
            })
            .collect();
        assert_eq!(strings.len(), 4, "hello.c's strings changed shape");
        strings
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    /// The module name `--nx-module-name` (libnx's `switch.specs`) wrote: the ELF's file name in the
    /// fixture's Makefile, without its extension.
    const MODULE_NAME: &[u8] = b"hello";

    #[test]
    fn the_nso_parses_with_every_segment_compressed_and_hash_checked() {
        let nso = Nso::parse(&read("hello.nso")).unwrap();
        for segment in 0..SEGMENT_COUNT {
            let shift = u32::try_from(segment).unwrap();
            assert_eq!(nso.flags >> (FLAG_COMPRESSED_SHIFT + shift) & 1, 1);
            assert_eq!(nso.flags >> (FLAG_HASH_CHECK_SHIFT + shift) & 1, 1);
        }
        assert_eq!(nso.build_id, elf_build_id());
        for text in program_strings() {
            assert!(contains(&nso.rodata.data, &text), "{text:?}");
        }
    }

    #[test]
    fn the_nso_hashes_cover_the_decompressed_segments() {
        // `elf2nso` sets the hash-check bits, and parsing verified each SHA-256 over the decompressed
        // segment. Breaking the stored hash of one segment must fail exactly that segment, which
        // shows the check ran (and over the decompressed bytes: the digest is found in the header).
        let file = read("hello.nso");
        let nso = Nso::parse(&file).unwrap();
        for (index, segment) in nso.segments().into_iter().enumerate() {
            let digest = Sha256::digest(&segment.data);
            let at = file
                .windows(digest.len())
                .position(|window| window == digest.as_slice())
                .ok_or(format!("segment {index}'s digest is not in the file"))
                .unwrap();
            let mut broken = file.clone();
            broken[at] ^= 1;
            assert!(matches!(
                Nso::parse(&broken),
                Err(NsoError::HashMismatch { segment }) if segment == index
            ));
        }
    }

    #[test]
    fn the_nso_module_name_is_at_the_start_of_rodata_not_in_the_segment_words() {
        let nso = Nso::parse(&read("hello.nso")).unwrap();
        // `elf2nso` writes 1 to both words SwitchBrew calls the module name offset and size; Eden
        // calls them alignments. The name is where the linker put it, at the start of rodata.
        assert_eq!((nso.module_name_offset, nso.module_name_size), (1, 1));
        assert_eq!(nso.module_path(), Some(MODULE_NAME));
    }

    #[test]
    fn the_nro_carries_the_same_build_id_and_segments_as_the_nso() {
        let file = read("hello.nro");
        let nro = Nro::parse(&file).unwrap();
        let nso = Nso::parse(&read("hello.nso")).unwrap();
        assert!(nro.is_homebrew);
        // Two tools, one ELF: both put its GNU build id at header offset 0x40.
        assert_eq!(nro.build_id, elf_build_id());

        let bytes = |segment: NroSegment| {
            let start = usize::try_from(segment.offset).unwrap();
            &file[start..start + usize::try_from(segment.size).unwrap()]
        };
        let pairs: [(NroSegment, &NsoSegment); 3] = [
            (nro.text, &nso.text),
            (nro.rodata, &nso.rodata),
            (nro.data, &nso.data),
        ];
        for (index, (nro_segment, nso_segment)) in pairs.into_iter().enumerate() {
            assert_eq!(
                nro_segment.offset, nso_segment.memory_offset,
                "segment {index}"
            );
            let nro_bytes = bytes(nro_segment);
            // The NRO header sits in the first bytes of its text; `elf2nso` leaves them as the ELF
            // had them. Everything after it is the same code.
            let skip = if index == 0 { nro::HEADER_SIZE } else { 0 };
            assert!(nro_bytes.len() >= nso_segment.data.len(), "segment {index}");
            assert_eq!(
                nro_bytes[skip..nso_segment.data.len()],
                nso_segment.data[skip..],
                "segment {index}"
            );
        }
        assert_eq!(nso.module_path(), Some(MODULE_NAME));
        for text in program_strings() {
            assert!(contains(bytes(nro.rodata), &text), "{text:?}");
        }

        // `MOD0`'s words are offsets from the `MOD0` header itself (`.word __bss_start__ - __nx_mod0`
        // in libnx/nx/source/runtime/switch_crt0.s @ b156b477). The NSO's bss starts where its data
        // segment's file bytes end, which can be a few alignment bytes before `.bss` proper; both
        // end at the same address.
        let mod0 = nro.mod0.unwrap();
        let data_end = nso.data.memory_offset + u32::try_from(nso.data.data.len()).unwrap();
        assert_eq!(
            nro.mod0_offset + mod0.bss_end_offset,
            data_end + nso.bss_size
        );
        assert!(nro.mod0_offset + mod0.bss_start_offset >= data_end);

        // The Makefile gives `elf2nro` a NACP and nothing else.
        let assets = nro.assets.unwrap();
        assert_eq!(assets.base, nro.size);
        let nacp = assets.file_range(&assets.nacp).unwrap();
        assert!(contains(&file[nacp], b"starquake hello"));
        assert_eq!(assets.file_range(&assets.icon), None);
        assert_eq!(assets.file_range(&assets.romfs), None);
    }

    /// A `"0x..."` string or a number from `hello.json`.
    fn number(value: &Value) -> u64 {
        match value {
            Value::String(text) => u64::from_str_radix(text.trim_start_matches("0x"), 16).unwrap(),
            other => other.as_u64().unwrap(),
        }
    }

    /// The `kernel_capabilities` entry of `kind` in `hello.json`.
    fn capability<'a>(description: &'a Value, kind: &str) -> &'a Value {
        &description["kernel_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["type"] == kind)
            .unwrap()["value"]
    }

    /// The capabilities `hello.json` describes, built from its fields: the sorted SVC ids, and every
    /// other field (with an empty SVC mask, which the caller fills from the parsed file once it has
    /// compared the ids).
    fn described_capabilities(description: &Value) -> (Vec<u32>, KernelCapabilities) {
        let flags = capability(description, "kernel_flags");
        let narrow = |value: &Value| u8::try_from(number(value)).unwrap();
        let mut svc_ids: Vec<u32> = capability(description, "syscalls")
            .as_object()
            .unwrap()
            .values()
            .map(|id| u32::try_from(number(id)).unwrap())
            .collect();
        svc_ids.sort_unstable();
        let debug = capability(description, "debug_flags");
        // npdmtool's value is the descriptor's version field: the major version above a 4-bit minor
        // (`KernelVersion` in eden/src/core/hle/kernel/k_capabilities.h @ cf654052).
        let version = number(capability(description, "min_kernel_version"));

        let capabilities = KernelCapabilities {
            // npdmtool's "highest" priority is the numerically largest, its "lowest" the smallest.
            thread_info: Some(ThreadInfo {
                priorities: narrow(&flags["lowest_thread_priority"])
                    ..=narrow(&flags["highest_thread_priority"]),
                cores: narrow(&flags["lowest_cpu_id"])..=narrow(&flags["highest_cpu_id"]),
            }),
            program_type: Some(narrow(capability(description, "application_type"))),
            kernel_version: Some(KernelVersion {
                major: u16::try_from(version >> 4).unwrap(),
                minor: u8::try_from(version & 0xF).unwrap(),
            }),
            handle_table_size: Some(
                u16::try_from(number(capability(description, "handle_table_size"))).unwrap(),
            ),
            debug_flags: Some(DebugFlags {
                allow_debug: debug["allow_debug"].as_bool().unwrap(),
                force_debug_prod: debug["force_debug_prod"].as_bool().unwrap(),
                force_debug: debug["force_debug"].as_bool().unwrap(),
            }),
            ..KernelCapabilities::default()
        };
        (svc_ids, capabilities)
    }

    /// The service entries `hello.json` lists, as the parser reports them: a trailing `*` becomes
    /// the wildcard flag.
    fn described_services(description: &Value) -> Vec<ServiceAccess> {
        let entries = |key: &str, server: bool| {
            description[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| {
                    let name = name.as_str().unwrap();
                    ServiceAccess {
                        name: name.trim_end_matches('*').to_owned(),
                        server,
                        wildcard: name.ends_with('*'),
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut services = entries("service_access", false);
        services.extend(entries("service_host", true));
        services
    }

    #[test]
    fn the_npdm_says_what_its_description_says() {
        let npdm = Npdm::parse(&read("main.npdm")).unwrap();
        let description = description();

        assert_eq!(npdm.name, description["name"].as_str().unwrap());
        assert_eq!(npdm.is_64_bit, description["is_64_bit"].as_bool().unwrap());
        assert_eq!(number(&description["address_space_type"]), 3);
        assert_eq!(npdm.address_space, AddressSpaceType::Is39Bit);
        assert_eq!(
            u64::from(npdm.main_thread_priority),
            number(&description["main_thread_priority"])
        );
        assert_eq!(
            u64::from(npdm.main_thread_core),
            number(&description["default_cpu_id"])
        );
        assert_eq!(
            u64::from(npdm.main_thread_stack_size),
            number(&description["main_thread_stack_size"])
        );
        assert_eq!(npdm.aci.title_id, number(&description["title_id"]));

        // npdmtool writes the Switch 1 ACID layout, a production ACID for `is_retail`.
        let acid = &npdm.acid;
        assert_eq!(acid.layout, AcidLayout::Switch1);
        assert_eq!(acid.production, description["is_retail"].as_bool().unwrap());
        assert_eq!(number(&description["pool_partition"]), 0);
        assert_eq!(acid.pool_partition, PoolPartition::Application);
        assert_eq!(
            acid.title_id_min,
            number(&description["title_id_range_min"])
        );
        assert_eq!(
            acid.title_id_max,
            number(&description["title_id_range_max"])
        );
        let permissions = number(&description["filesystem_access"]["permissions"]);
        assert_eq!(acid.file_access.permissions, permissions);
        assert_eq!(npdm.aci.file_access.permissions, permissions);
    }

    #[test]
    fn the_acid_access_control_is_read_from_the_start_of_the_acid_block() {
        // The ACID's SAC and KAC offsets are added to the start of the ACID block. Read from any
        // other base, this file's lists would not decode to the names and capability words
        // `hello.json` describes (npdmtool writes the same lists to the ACID and the ACI0).
        let npdm = Npdm::parse(&read("main.npdm")).unwrap();
        let description = description();
        let services = described_services(&description);
        let (svc_ids, capabilities) = described_capabilities(&description);
        assert_eq!(npdm.acid.services, services);
        assert_eq!(npdm.aci.services, services);
        for parsed in [
            &npdm.acid.kernel_capabilities,
            &npdm.aci.kernel_capabilities,
        ] {
            assert_eq!(parsed.svc_mask.iter().collect::<Vec<_>>(), svc_ids);
            let expected = KernelCapabilities {
                svc_mask: parsed.svc_mask,
                ..capabilities.clone()
            };
            assert_eq!(*parsed, expected);
            assert_eq!(parsed.check_policy(EDEN_CAPABILITY_CORE_LIMIT), Ok(()));
        }
    }

    #[test]
    fn the_service_wildcard_is_a_trailing_star() {
        // `hello.json` lists `set:*`; npdmtool stores the `*` as the name's last byte.
        let npdm = Npdm::parse(&read("main.npdm")).unwrap();
        let wildcard = npdm
            .acid
            .services
            .iter()
            .find(|entry| entry.wildcard)
            .unwrap();
        assert_eq!(wildcard.name, "set:");
        let exact = npdm
            .aci
            .services
            .iter()
            .find(|entry| !entry.wildcard)
            .unwrap();
        assert!(wildcard.covers(exact));
    }

    #[test]
    fn the_aci0_is_inside_its_acid() {
        let npdm = Npdm::parse(&read("main.npdm")).unwrap();
        assert_eq!(npdm.validate_against_acid(), Ok(()));
    }
}
