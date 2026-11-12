//! What [`crate::ReplyBuilder`] and [`crate::RequestBuilder`] share: the parts both collect, the
//! special header and handle lists, the header fill-in, and the checked conversion of counts.
//!
//! Every count that becomes a header word goes through [`to_u32`], so an impossible count is an
//! [`IpcError::FieldOutOfRange`] naming the field in both builders; nothing saturates.

use crate::error::{IpcError, Result};
use crate::hipc::{CommandHeader, SpecialHeader, high32, low32};

/// The pieces every message can carry, in the order the caller added them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Parts {
    pub(crate) copy_handles: Vec<u32>,
    pub(crate) move_handles: Vec<u32>,
    pub(crate) data: Vec<u8>,
    pub(crate) object_ids: Vec<u32>,
}

/// The out-of-range error for `value` in the field `field`.
pub(crate) fn out_of_range(field: &'static str, value: usize) -> IpcError {
    IpcError::FieldOutOfRange {
        field,
        value: u64::try_from(value).unwrap_or(u64::MAX),
    }
}

/// A count as a header word, or the out-of-range error naming `field`.
pub(crate) fn to_u32(field: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| out_of_range(field, value))
}

/// Appends `words` to `out` as little-endian bytes: how both builders lay words into a byte
/// section (the object ids of a reply, the headers and ids of a request).
pub(crate) fn extend_words(out: &mut Vec<u8>, words: &[u32]) {
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// Appends the special header, the process id words (if `pid` is set) and the handle lists, when
/// there is anything to put there. Returns whether a special header was written, which the
/// command header must announce.
pub(crate) fn push_special_header(
    words: &mut Vec<u32>,
    pid: Option<u64>,
    copy_handles: &[u32],
    move_handles: &[u32],
) -> Result<bool> {
    if pid.is_none() && copy_handles.is_empty() && move_handles.is_empty() {
        return Ok(false);
    }
    let special = SpecialHeader {
        send_pid: pid.is_some(),
        num_copy_handles: to_u32("copy handle count", copy_handles.len())?,
        num_move_handles: to_u32("move handle count", move_handles.len())?,
    };
    words.push(special.encode()?);
    if let Some(pid) = pid {
        words.extend([low32(pid), high32(pid)]);
    }
    words.extend_from_slice(copy_handles);
    words.extend_from_slice(move_handles);
    Ok(true)
}

/// Writes the encoded `header` over the first [`CommandHeader::WORDS`] words, which the builder
/// reserved before it knew the raw data size.
pub(crate) fn fill_header(words: &mut [u32], header: &CommandHeader) -> Result<()> {
    let encoded = header.encode()?;
    for (slot, word) in words.iter_mut().zip(encoded) {
        *slot = word;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipc::{CommandType, MAX_HANDLES, ReceiveList};

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn counts_beyond_a_word_name_their_field() {
        let word_limit = u64::from(u32::MAX);
        assert_eq!(
            to_u32("raw data words", usize::try_from(word_limit).unwrap()),
            Ok(u32::MAX)
        );
        assert_eq!(
            to_u32("raw data words", usize::try_from(word_limit + 1).unwrap()),
            Err(IpcError::FieldOutOfRange {
                field: "raw data words",
                value: word_limit + 1,
            })
        );
    }

    #[test]
    fn words_are_appended_as_little_endian_bytes_in_order() {
        let mut out = vec![0xEE];
        extend_words(&mut out, &[0x0403_0201, 0x0807_0605]);
        extend_words(&mut out, &[]);
        assert_eq!(out, [0xEE, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn no_special_header_without_a_pid_or_handles() {
        let mut words = vec![0; CommandHeader::WORDS];
        assert_eq!(push_special_header(&mut words, None, &[], &[]), Ok(false));
        assert_eq!(words.len(), CommandHeader::WORDS);
    }

    #[test]
    fn special_header_lists_follow_in_order() {
        let mut words = vec![0; CommandHeader::WORDS];
        let pid = 0x0000_0007_0000_0009_u64;
        assert_eq!(
            push_special_header(&mut words, Some(pid), &[0xC1, 0xC2], &[0xD1]),
            Ok(true)
        );
        let special = SpecialHeader {
            send_pid: true,
            num_copy_handles: 2,
            num_move_handles: 1,
        };
        assert_eq!(
            words[CommandHeader::WORDS..],
            [special.encode().unwrap(), 9, 7, 0xC1, 0xC2, 0xD1]
        );
    }

    #[test]
    fn handle_counts_beyond_their_fields_are_errors() {
        let mut words = vec![0; CommandHeader::WORDS];
        let handles = vec![0; usize::try_from(MAX_HANDLES).unwrap() + 1];
        let over = u64::from(MAX_HANDLES) + 1;
        assert_eq!(
            push_special_header(&mut words, None, &handles, &[]),
            Err(IpcError::FieldOutOfRange {
                field: "copy handle count",
                value: over,
            })
        );
        assert_eq!(
            push_special_header(&mut words, None, &[], &handles),
            Err(IpcError::FieldOutOfRange {
                field: "move handle count",
                value: over,
            })
        );
    }

    #[test]
    fn the_header_is_written_over_the_reserved_words_only() {
        let header = CommandHeader {
            command_type: CommandType::Request,
            num_x: 0,
            num_a: 0,
            num_b: 0,
            num_w: 0,
            raw_data_words: 3,
            receive_list: ReceiveList::Disabled,
            has_special_header: false,
        };
        let mut words = vec![0, 0, 0xAB, 0xCD];
        fill_header(&mut words, &header).unwrap();
        let [w0, w1] = header.encode().unwrap();
        assert_eq!(words, [w0, w1, 0xAB, 0xCD]);
    }
}
