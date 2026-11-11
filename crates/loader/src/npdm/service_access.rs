//! Service access control (SAC): the list of services a title may open or host.
//!
//! The encoding is a run of entries, each a control byte followed by a name: bit 7 of the control
//! byte is the "server" flag (the title hosts the service rather than connecting to it) and the low
//! three bits hold the name length minus one. This is the loop in
//! `eden-nx2/tools/nx2/npdm_info.py` (`ctl & 0x80`, `(ctl & 7) + 1`).
//!
//! A trailing `*` in a name makes it a wildcard (a prefix match). That is from the SwitchBrew NPDM
//! page and is not in the pinned references; it is reported as a separate flag, with the `*`
//! removed from the name.

/// One service access entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceAccess {
    /// The service name, without any trailing `*`.
    pub name: String,
    /// True when the title hosts the service (control byte bit 7).
    pub server: bool,
    /// True when the name ended in `*` (a prefix match). SwitchBrew; not in the pinned references.
    pub wildcard: bool,
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

    /// Encodes one entry the way the file does.
    fn entry(name: &str, server: bool) -> Vec<u8> {
        let mut out = vec![u8::try_from(name.len() - 1).unwrap() | if server { 0x80 } else { 0 }];
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
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = parse(&bytes);
        }
    }
}
