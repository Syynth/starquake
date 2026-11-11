//! Service access control (SAC): the list of services a title may open or host.
//!
//! The encoding is a run of entries, each a control byte followed by a name: bit 7 of the control
//! byte is the "server" flag (the title hosts the service rather than connecting to it) and the low
//! three bits hold the name length minus one. This is the loop in
//! `eden-nx2/tools/nx2/npdm_info.py` (`ctl & 0x80`, `(ctl & 7) + 1`).
//!
//! A trailing `*` in a name makes it a wildcard (a prefix match); it is reported as a separate
//! flag, with the `*` removed from the name.
//!
//! # Sourcing: a guess that stays
//!
//! The pinned references never handle `*` (Eden does not check service access at all, and
//! `npdm_info.py` prints names verbatim). The SwitchBrew NPDM page says the name "supports the
//! wildcard `*` character" (see "Documentation sources" in `docs/references.md`), but it does not
//! say that the `*` must be last or that the match is a prefix match. So **the trailing-only,
//! prefix-match reading is a guess**, kept because it is the simplest reading of "supports the
//! wildcard"; no fixture built from homebrew exercises it. A `*` anywhere else in a name is kept
//! as a literal character. If a real ACID disagrees, [`ServiceAccess::covers`] and the `wildcard`
//! flag are the two places to fix.

/// One service access entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceAccess {
    /// The service name, without any trailing `*`.
    pub name: String,
    /// True when the title hosts the service (control byte bit 7).
    pub server: bool,
    /// True when the name ended in `*` (a prefix match). A guess; see the module docs.
    pub wildcard: bool,
}

impl ServiceAccess {
    /// True if `self`, read as an entry of the signed `ACID` list, permits `request`, an entry of
    /// the `ACI0` list: the same role (`server`), and either the same name or, when `self` is a
    /// wildcard, a request whose name starts with `self`'s prefix.
    ///
    /// A wildcard request is permitted only by a wildcard grant whose prefix it extends, so an
    /// `ACI0` cannot widen an exact grant.
    ///
    /// Neither Eden nor any pinned reference validates `ACI0` against `ACID`, so the matching is
    /// a guess, chosen as the least permissive reading consistent with the wildcard being a
    /// prefix match (see the module docs).
    pub fn covers(&self, request: &Self) -> bool {
        if self.server != request.server {
            return false;
        }
        if self.wildcard {
            request.name.starts_with(&self.name)
        } else {
            !request.wildcard && request.name == self.name
        }
    }
}

/// Why a service access block could not be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ServiceAccessError {
    /// An entry's name runs past the end of the block.
    #[error("service access entry at {offset:#x} runs past the end of the block")]
    Truncated {
        /// Offset of the entry's control byte within the block.
        offset: usize,
    },
    /// A name is not valid UTF-8.
    #[error("service access entry at {offset:#x} has a name that is not valid UTF-8")]
    InvalidName {
        /// Offset of the entry's control byte within the block.
        offset: usize,
    },
}

/// Server flag in the control byte.
const SERVER_FLAG: u8 = 0x80;
/// Mask of the name-length field (length minus one) in the control byte.
const LENGTH_MASK: u8 = 0x07;

/// Parses a whole service access block.
pub fn parse(bytes: &[u8]) -> Result<Vec<ServiceAccess>, ServiceAccessError> {
    let mut services = Vec::new();
    let mut offset = 0usize;
    while let Some(&control) = bytes.get(offset) {
        let len = usize::from(control & LENGTH_MASK) + 1;
        let start = offset + 1;
        let end = start + len;
        let raw = bytes
            .get(start..end)
            .ok_or(ServiceAccessError::Truncated { offset })?;
        let text =
            std::str::from_utf8(raw).map_err(|_| ServiceAccessError::InvalidName { offset })?;
        let (name, wildcard) = match text.strip_suffix('*') {
            Some(prefix) => (prefix, true),
            None => (text, false),
        };
        services.push(ServiceAccess {
            name: name.to_owned(),
            server: control & SERVER_FLAG != 0,
            wildcard,
        });
        offset = end;
    }
    Ok(services)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_5143;

    /// Encodes one entry the way the file does.
    fn entry(name: &str, server: bool) -> Vec<u8> {
        let mut out =
            vec![u8::try_from(name.len() - 1).unwrap() | if server { SERVER_FLAG } else { 0 }];
        out.extend_from_slice(name.as_bytes());
        out
    }

    #[test]
    fn decodes_names_server_flag_and_wildcards() {
        let mut block = entry("fs:ro", false);
        block.extend(entry("sm:m", true));
        block.extend(entry("ldr:*", false));
        // Raw bytes: control 0x04 (length 5), then the name; the server entry has bit 7 set.
        assert_eq!(&block[..6], b"\x04fs:ro");
        assert_eq!(block[6], 0x83);
        let services = parse(&block).unwrap();
        assert_eq!(
            services,
            [
                ServiceAccess {
                    name: "fs:ro".into(),
                    server: false,
                    wildcard: false
                },
                ServiceAccess {
                    name: "sm:m".into(),
                    server: true,
                    wildcard: false
                },
                ServiceAccess {
                    name: "ldr:".into(),
                    server: false,
                    wildcard: true
                },
            ]
        );
    }

    fn service(name: &str, server: bool, wildcard: bool) -> ServiceAccess {
        ServiceAccess {
            name: name.into(),
            server,
            wildcard,
        }
    }

    #[test]
    fn a_grant_covers_an_exact_request_with_the_same_role() {
        let grant = service("fs:ro", false, false);
        assert!(grant.covers(&service("fs:ro", false, false)));
        // Different name, different role, and a wildcard request against an exact grant.
        assert!(!grant.covers(&service("fs:rw", false, false)));
        assert!(!grant.covers(&service("fs:ro", true, false)));
        assert!(!grant.covers(&service("fs:", false, true)));
        // A prefix of the grant's name is not covered either.
        assert!(!grant.covers(&service("fs:r", false, false)));
    }

    #[test]
    fn a_wildcard_grant_covers_names_with_its_prefix() {
        let grant = service("ldr:", false, true);
        assert!(grant.covers(&service("ldr:ro", false, false)));
        assert!(grant.covers(&service("ldr:", false, false)));
        assert!(grant.covers(&service("ldr:", false, true)));
        assert!(grant.covers(&service("ldr:r", false, true)));
        assert!(!grant.covers(&service("ld", false, false)));
        assert!(!grant.covers(&service("ld", false, true)));
        assert!(!grant.covers(&service("sm:", false, false)));
        assert!(!grant.covers(&service("ldr:ro", true, false)));
    }

    #[test]
    fn empty_block_has_no_services() {
        assert_eq!(parse(&[]).unwrap(), []);
    }

    #[test]
    fn truncated_and_invalid_entries_are_errors() {
        let mut block = entry("fs:ro", false);
        block.pop();
        assert_eq!(
            parse(&block),
            Err(ServiceAccessError::Truncated { offset: 0 })
        );

        let mut block = entry("sm:", false);
        block.extend_from_slice(&[0x01, 0xFF, 0xFE]);
        assert_eq!(
            parse(&block),
            Err(ServiceAccessError::InvalidName { offset: 4 })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = parse(&bytes);
        }
    }
}
