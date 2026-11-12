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
//! # Sourcing
//!
//! Eden does not check service access at all, and `npdm_info.py` prints names verbatim. The
//! wildcard form comes from Atmosphère's sm (`AccessControlEntry` in
//! `atmosphere/stratosphere/sm/source/impl/sm_service_manager.cpp` @ 4e3f3703, fact only): an
//! entry is a wildcard when the last byte of its name is `*` (`IsWildcard`:
//! `m_entry[GetServiceNameSize()] == '*'`), and a wildcard grant matches the names that start
//! with the bytes before the `*`. A `*` anywhere else in a name is an ordinary byte there and
//! here. `tests/homebrew/hello/main.npdm`, written by switch-tools' `npdmtool` from a description
//! listing `set:*`, stores the `*` as the name's last byte (`crates/loader/tests/homebrew_fixture.rs`).
//! How a grant covers a request, and the two places this differs from sm, is in
//! [`ServiceAccess::covers`].

/// One service access entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceAccess {
    /// The service name, without any trailing `*`.
    pub name: String,
    /// True when the title hosts the service (control byte bit 7).
    pub server: bool,
    /// True when the name ended in `*` (a prefix match; see the module docs).
    pub wildcard: bool,
}

impl ServiceAccess {
    /// True if `self`, read as an entry of the signed `ACID` list, permits `request`, an entry of
    /// the `ACI0` list.
    ///
    /// Mirrors one iteration of `ValidateAccessControl` in
    /// `atmosphere/stratosphere/sm/source/impl/sm_service_manager.cpp` @ 4e3f3703 (fact only):
    /// the roles must match (`IsHost() == is_host`); when both entries are wildcards or neither
    /// is, the names must be equal; when only the grant is a wildcard, the request's name must
    /// start with the grant's prefix (`memcmp` over `GetServiceNameSize() - 1` bytes).
    ///
    /// Two divergences from sm:
    ///
    /// 1. Pinned by `a_wildcard_request_is_not_covered_by_an_exact_grant`: when only the request
    ///    is a wildcard, sm's check leaves `is_valid` true and accepts it against any exact grant
    ///    of the same role. That would let an `ACI0` widen an exact grant to every name, so this
    ///    refuses it.
    /// 2. Pinned by `divergence_names_are_not_compared_zero_padded`: sm compares names as 8-byte
    ///    values padded with zero bytes (`ServiceName::Encode` in
    ///    `atmosphere/libraries/libstratosphere/include/stratosphere/sm/sm_types.hpp`), so an
    ///    entry whose name ends in NUL bytes equals the same name without them (an `ACI0` entry
    ///    `ab\0` of length 3 equals an `ACID` entry `ab`, and a wildcard grant `a\0*` covers the
    ///    name `a`). Here the names are compared as the strings stored, with their NUL bytes, and
    ///    differ. Every pair this accepts, sm accepts too: the divergence is never looser.
    pub fn covers(&self, request: &Self) -> bool {
        if self.server != request.server {
            return false;
        }
        match (self.wildcard, request.wildcard) {
            (true, false) => request.name.starts_with(&self.name),
            (false, true) => false,
            (true, true) | (false, false) => request.name == self.name,
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

/// Test-side encoder shared by the `npdm` tests.
#[cfg(test)]
pub(crate) mod encode {
    use super::*;

    /// Encodes one entry the way the file does: control byte (server flag, name length minus
    /// one), then the name.
    pub(crate) fn entry(name: &str, server: bool) -> Vec<u8> {
        let length_field = u8::try_from(name.len() - 1).unwrap();
        assert_eq!(
            length_field & !LENGTH_MASK,
            0,
            "name too long for one entry"
        );
        let mut out = vec![length_field | if server { SERVER_FLAG } else { 0 }];
        out.extend_from_slice(name.as_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_5143;

    use super::encode::entry;

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
        assert!(!grant.covers(&service("ld", false, false)));
        assert!(!grant.covers(&service("ld", false, true)));
        assert!(!grant.covers(&service("sm:", false, false)));
        assert!(!grant.covers(&service("ldr:ro", true, false)));
    }

    /// sm's rule when both entries are wildcards: the names must be equal, as for two exact
    /// entries. A longer wildcard is not covered by a shorter one.
    #[test]
    fn a_wildcard_request_needs_the_same_wildcard_grant() {
        let grant = service("ldr:", false, true);
        assert!(grant.covers(&service("ldr:", false, true)));
        assert!(!grant.covers(&service("ldr:r", false, true)));
        assert!(!grant.covers(&service("ldr", false, true)));
        assert!(!grant.covers(&service("ldr:", true, true)));
    }

    /// Divergence (module docs): sm accepts a wildcard request against an exact grant of the
    /// same role, whatever the names; this refuses it.
    #[test]
    fn a_wildcard_request_is_not_covered_by_an_exact_grant() {
        let grant = service("fs:ro", false, false);
        for request in [
            service("fs:", false, true),
            service("fs:ro", false, true),
            service("zz", false, true),
        ] {
            assert!(!grant.covers(&request), "{request:?}");
        }
    }

    /// Divergence (see [`ServiceAccess::covers`]): sm compares zero-padded 8-byte names, so
    /// trailing NUL bytes do not matter to it; here they do.
    #[test]
    fn divergence_names_are_not_compared_zero_padded() {
        // An `ACI0` entry `ab\0` against an `ACID` entry `ab`, both exact.
        let grant = service("ab", false, false);
        let request = service("ab\0", false, false);
        assert!(!grant.covers(&request));
        assert!(!request.covers(&grant));
        // A wildcard grant whose prefix ends in NUL against the name without it.
        let grant = service("a\0", false, true);
        assert!(!grant.covers(&service("a", false, false)));
        // The same names parse as distinct entries, NUL kept.
        let parsed = parse(&[0x02, b'a', b'b', 0]).unwrap();
        assert_eq!(parsed, [service("ab\0", false, false)]);
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
