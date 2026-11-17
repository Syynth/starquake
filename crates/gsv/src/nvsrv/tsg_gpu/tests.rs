//! nvhost-tsg-gpu's timeslice and bindings, driven through the nvsrv state and the server.
//!
//! Arguments are laid out by hand from PR's decode
//! (`eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_context.h` @ e5fb3b97,
//! `DecodeNativeTsgBinding`: an `s32` fd; for the extended form a `u32` subcontext id and 16 zero
//! bytes), the numbers as PR compares them (this module's constants are PR's literals).

use starquake_services::framework::UnknownPolicy;

use super::super::tests::{Direct, Rig, STALE, le, ok};
use super::*;
use crate::nvsrv::ioctl::Ioctl;
use crate::nvsrv::{IoctlVariant, NvsrvState, NvsrvStateError};

/// A direct state with a tsg fd (1) and two channel fds (2, 3).
fn opened() -> (Direct, i32, [u32; 2]) {
    let mut direct = Direct::new();
    let tsg = direct.open(Device::NvhostTsgGpu);
    let a = direct.open(Device::NvhostGpu);
    let b = direct.open(Device::NvhostGpu);
    let fds = [a, b].map(|fd| u32::try_from(fd).unwrap());
    (direct, tsg, fds)
}

/// A binding's argument: the channel fd.
fn binding(channel: u32) -> Vec<u8> {
    le(&[channel])
}

/// An extended binding's argument: the channel fd, the subcontext id, four zero words.
fn binding_ex(channel: u32, subcontext: u32) -> Vec<u8> {
    le(&[channel, subcontext, 0, 0, 0, 0])
}

/// The channels `tsg`'s group holds, with their subcontexts.
fn bound(direct: &Direct, tsg: i32) -> Vec<(u32, u32)> {
    let tsg = u32::try_from(tsg).unwrap();
    direct
        .state
        .tsg_gpu()
        .group(tsg)
        .map_or_else(Vec::new, |group| {
            group.channels.iter().map(|(&fd, &sub)| (fd, sub)).collect()
        })
}

#[test]
fn the_numbers_are_pr_s() {
    for (raw, command, size) in [
        (SET_TIMESLICE, 0x8, TIMESLICE_BYTES),
        (BIND_CHANNEL, 0x1, BINDING_BYTES),
        (UNBIND_CHANNEL, 0x2, BINDING_BYTES),
        (BIND_CHANNEL_EX, 0xB, BINDING_EX_BYTES),
    ] {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(ioctl.group(), 0x54, "{raw:#x}");
        assert_eq!(ioctl.command(), command, "{raw:#x}");
        assert_eq!(usize::try_from(ioctl.size()).unwrap(), size, "{raw:#x}");
        assert!(ioctl.is_in() && ioctl.is_out(), "{raw:#x}");
    }
}

// ----- The timeslice -------------------------------------------------------------------------

/// A timeslice is kept and echoed; 0, or either buffer not exactly four bytes, is
/// `BadParameter` with nothing kept or written; the edges at four bytes are accepted.
#[test]
fn a_timeslice_is_kept_and_echoed() {
    let cases: [(Vec<u8>, usize, NvResult); 6] = [
        (le(&[2000]), 4, NvResult::Success),
        (le(&[0]), 4, NvResult::BadParameter),
        (le(&[2000])[..3].to_vec(), 4, NvResult::BadParameter),
        ([le(&[2000]), vec![0]].concat(), 4, NvResult::BadParameter),
        (le(&[2000]), 3, NvResult::BadParameter),
        (le(&[2000]), 5, NvResult::BadParameter),
    ];
    for (input, out_len, expected) in cases {
        let (mut direct, tsg, _) = opened();
        let answer = direct.call(
            UnknownPolicy::Strict,
            IoctlVariant::Ioctl,
            tsg,
            SET_TIMESLICE,
            &input,
            (out_len, 0),
        );
        assert_eq!(answer.result, expected, "{input:?} {out_len}");
        let group = direct.state.tsg_gpu().group(1).cloned();
        if expected == NvResult::Success {
            assert_eq!(answer.output, input);
            assert_eq!(group.unwrap().timeslice_us, 2000);
        } else {
            assert_eq!(answer.output, Vec::<u8>::new());
            assert_eq!(group, None, "{input:?} {out_len}");
        }
    }
}

// ----- Bindings ---------------------------------------------------------------------------------

/// A channel binds and comes back as sent; the extended form keeps its subcontext; an unbind
/// takes it out, after which it binds again.
#[test]
fn a_channel_binds_and_unbinds() {
    let (mut direct, tsg, [a, b]) = opened();
    let answer = direct.ioctl(tsg, BIND_CHANNEL, &binding(a));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::Success, binding(a))
    );
    let answer = direct.ioctl(tsg, BIND_CHANNEL_EX, &binding_ex(b, 0x2A));
    assert_eq!(answer.result, NvResult::Success);
    assert_eq!(answer.output, binding_ex(b, 0x2A));
    assert_eq!(bound(&direct, tsg), [(a, 0), (b, 0x2A)]);
    let answer = direct.ioctl(tsg, UNBIND_CHANNEL, &binding(a));
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::Success, binding(a))
    );
    assert_eq!(bound(&direct, tsg), [(b, 0x2A)]);
    assert_eq!(
        direct.ioctl(tsg, BIND_CHANNEL, &binding(a)).result,
        NvResult::Success
    );
}

/// The bind checks, each alone: already bound here (`Busy`), not a channel (an nvmap fd, an fd
/// not open, a negative one: `BadParameter`), bound in another group (`Busy`). Nothing changes.
#[test]
fn each_bind_check_refuses() {
    let (mut direct, tsg, [a, b]) = opened();
    let nvmap = u32::try_from(direct.open(Device::Nvmap)).unwrap();
    let other = direct.open(Device::NvhostTsgGpu);
    direct.ioctl(tsg, BIND_CHANNEL, &binding(a));
    direct.ioctl(other, BIND_CHANNEL, &binding(b));
    let cases = [
        (tsg, binding(a), NvResult::Busy),
        (tsg, binding(nvmap), NvResult::BadParameter),
        (tsg, binding(0x99), NvResult::BadParameter),
        (tsg, binding(u32::MAX), NvResult::BadParameter),
        (tsg, binding(b), NvResult::Busy),
        (other, binding(a), NvResult::Busy),
    ];
    for (group, input, expected) in cases {
        let answer = direct.ioctl(group, BIND_CHANNEL, &input);
        assert_eq!(answer.result, expected, "{group} {input:?}");
        assert_eq!(answer.output, Vec::<u8>::new());
    }
    assert_eq!(bound(&direct, tsg), [(a, 0)]);
    assert_eq!(bound(&direct, other), [(b, 0)]);
}

/// A group holds [`MAX_CHANNELS`]: the last binds, one more is `InsufficientMemory`. At the limit,
/// a channel already in the group is `Busy` (checked first), and a non-channel fd is
/// `InsufficientMemory` (the limit is checked before the fd).
#[test]
fn a_group_holds_at_most_max_channels() {
    let mut direct = Direct::new();
    let tsg = direct.open(Device::NvhostTsgGpu);
    let channels: Vec<u32> = (0..=MAX_CHANNELS)
        .map(|_| u32::try_from(direct.open(Device::NvhostGpu)).unwrap())
        .collect();
    let nvmap = u32::try_from(direct.open(Device::Nvmap)).unwrap();
    for &channel in &channels[..MAX_CHANNELS] {
        assert_eq!(
            direct.ioctl(tsg, BIND_CHANNEL, &binding(channel)).result,
            NvResult::Success
        );
    }
    let last = channels[MAX_CHANNELS];
    let cases = [
        (last, NvResult::InsufficientMemory),
        (channels[0], NvResult::Busy),
        (nvmap, NvResult::InsufficientMemory),
    ];
    for (channel, expected) in cases {
        let answer = direct.ioctl(tsg, BIND_CHANNEL, &binding(channel));
        assert_eq!(answer.result, expected, "{channel}");
    }
    assert_eq!(bound(&direct, tsg).len(), MAX_CHANNELS);
}

/// The decode: the exact size for each form (a byte short and a byte long refused), a
/// subcontext at [`SUBCONTEXT_COUNT`] refused and one below accepted, a non-zero byte anywhere in
/// the last 16 refused, a negative fd refused, and out buffer 0 of another size than the in
/// buffer refused. Decoding comes before the group's own checks: a bad decode of a channel that
/// is bound already is `BadParameter`, not `Busy`.
#[test]
fn bindings_decode_as_pr_s() {
    let (mut direct, tsg, [a, b]) = opened();
    direct.ioctl(tsg, BIND_CHANNEL, &binding(a));
    let short_ex = binding_ex(b, 1)[..23].to_vec();
    let long_ex = [binding_ex(b, 1), vec![0]].concat();
    let mut cases: Vec<(u32, Vec<u8>, usize)> = vec![
        (BIND_CHANNEL, binding(b)[..3].to_vec(), 3),
        (BIND_CHANNEL, [binding(b), vec![0]].concat(), 5),
        (BIND_CHANNEL_EX, short_ex, 23),
        (BIND_CHANNEL_EX, long_ex, 25),
        (BIND_CHANNEL_EX, binding_ex(b, SUBCONTEXT_COUNT), 24),
        (BIND_CHANNEL_EX, binding_ex(a, SUBCONTEXT_COUNT), 24),
        (BIND_CHANNEL, binding((-1_i32).cast_unsigned()), 4),
        (BIND_CHANNEL, binding(b), 3),
        (BIND_CHANNEL, binding(b), 5),
        (BIND_CHANNEL, binding(a), 5),
        (BIND_CHANNEL_EX, binding_ex(b, 1), 23),
        (UNBIND_CHANNEL, binding(a)[..3].to_vec(), 3),
        (UNBIND_CHANNEL, binding(a), 5),
        (UNBIND_CHANNEL, binding((-1_i32).cast_unsigned()), 4),
    ];
    for at in BINDING_EX_ZEROS..BINDING_EX_BYTES {
        let mut input = binding_ex(b, 1);
        input[at] = 1;
        cases.push((BIND_CHANNEL_EX, input, 24));
    }
    for (raw, input, out_len) in cases {
        let answer = direct.call(
            UnknownPolicy::Strict,
            IoctlVariant::Ioctl,
            tsg,
            raw,
            &input,
            (out_len, 0),
        );
        assert_eq!(
            answer.result,
            NvResult::BadParameter,
            "{raw:#x} {input:?} {out_len}"
        );
    }
    assert_eq!(bound(&direct, tsg), [(a, 0)]);
    let top = SUBCONTEXT_COUNT - 1;
    assert_eq!(
        direct
            .ioctl(tsg, BIND_CHANNEL_EX, &binding_ex(b, top))
            .result,
        NvResult::Success
    );
    assert_eq!(bound(&direct, tsg), [(a, 0), (b, top)]);
}

/// Unbinding a channel the group does not hold is `BadParameter`: never bound, bound in another
/// group, or a group with no state yet.
#[test]
fn an_unbind_needs_the_channel_in_this_group() {
    let (mut direct, tsg, [a, b]) = opened();
    let other = direct.open(Device::NvhostTsgGpu);
    let refused = |direct: &mut Direct, group, channel| {
        direct
            .ioctl(group, UNBIND_CHANNEL, &binding(channel))
            .result
    };
    assert_eq!(refused(&mut direct, tsg, a), NvResult::BadParameter);
    direct.ioctl(other, BIND_CHANNEL, &binding(b));
    assert_eq!(refused(&mut direct, tsg, b), NvResult::BadParameter);
    assert_eq!(bound(&direct, other), [(b, 0)]);
}

/// Closing the group's fd unbinds its channels, which then bind elsewhere; closing a channel's
/// fd keeps its binding, which an unbind still finds.
#[test]
fn closing_a_group_frees_its_channels_and_closing_a_channel_does_not() {
    let (mut direct, tsg, [a, b]) = opened();
    let other = direct.open(Device::NvhostTsgGpu);
    direct.ioctl(tsg, BIND_CHANNEL, &binding(a));
    direct.ioctl(tsg, BIND_CHANNEL, &binding(b));
    assert_eq!(direct.close(i32::try_from(b).unwrap()), NvResult::Success);
    assert_eq!(bound(&direct, tsg), [(a, 0), (b, 0)]);
    assert_eq!(
        direct.ioctl(tsg, UNBIND_CHANNEL, &binding(b)).result,
        NvResult::Success
    );
    assert_eq!(direct.close(tsg), NvResult::Success);
    assert_eq!(direct.state.tsg_gpu().group(1), None);
    assert_eq!(
        direct.ioctl(other, BIND_CHANNEL, &binding(a)).result,
        NvResult::Success
    );
}

/// Other numbers, and every number on `Ioctl2` and `Ioctl3`, go to the policy.
#[test]
fn other_ioctls_go_to_the_policy() {
    let (mut direct, tsg, [a, _]) = opened();
    let cases = [
        (IoctlVariant::Ioctl, 0xC004_5403),
        (IoctlVariant::Ioctl, 0xC008_5401),
        (IoctlVariant::Ioctl, 0x4004_5401),
        (IoctlVariant::Ioctl2, BIND_CHANNEL),
        (IoctlVariant::Ioctl3, BIND_CHANNEL),
        (IoctlVariant::Ioctl3, SET_TIMESLICE),
    ];
    for (variant, raw) in cases {
        let answer = direct.call(
            UnknownPolicy::Strict,
            variant,
            tsg,
            raw,
            &binding(a),
            (4, 4),
        );
        assert_eq!(
            answer.result,
            NvResult::NotImplemented,
            "{variant:?} {raw:#x}"
        );
    }
    assert_eq!(direct.state.tsg_gpu().group(1), None);
}

/// Through the server: a refused binding writes nothing to out buffer 0, an accepted one the
/// argument as sent.
#[test]
fn a_refused_binding_writes_nothing() {
    let mut rig = Rig::new();
    let tsg = rig.open_device(Device::NvhostTsgGpu);
    let channel = u32::try_from(rig.open_device(Device::NvhostGpu)).unwrap();
    let (reply, out) = rig.ioctl(tsg, BIND_CHANNEL, &binding(0x55), 4);
    assert_eq!(
        (reply, out),
        (ok(&[NvResult::BadParameter.raw()]), vec![STALE; 4])
    );
    let (reply, out) = rig.ioctl(tsg, BIND_CHANNEL, &binding(channel), 4);
    assert_eq!((reply, out), (ok(&[0]), binding(channel)));
}

// ----- Saved state -------------------------------------------------------------------------------

fn group(timeslice_us: u32, channels: &[(u32, u32)]) -> Group {
    Group {
        timeslice_us,
        channels: channels.iter().copied().collect(),
    }
}

/// Each saved state the ioctls cannot reach is refused, first fault first; the reachable edges
/// are accepted, also through `postcard`.
#[test]
fn a_saved_state_the_ioctls_cannot_reach_is_refused() {
    let full: Vec<(u32, u32)> = (10..10 + u32::try_from(MAX_CHANNELS).unwrap())
        .map(|fd| (fd, 0))
        .collect();
    let over: Vec<(u32, u32)> = (10..11 + u32::try_from(MAX_CHANNELS).unwrap())
        .map(|fd| (fd, 0))
        .collect();
    let refused = [
        (
            vec![(1, group(0, &over))],
            TsgStateError::TooManyChannels {
                fd: 1,
                count: MAX_CHANNELS + 1,
            },
        ),
        (
            vec![(1, group(0, &[(5, SUBCONTEXT_COUNT)]))],
            TsgStateError::Subcontext {
                channel: 5,
                subcontext: SUBCONTEXT_COUNT,
            },
        ),
        (
            vec![(1, group(0, &[(5, 0)])), (2, group(0, &[(5, 1)]))],
            TsgStateError::TwoGroups { channel: 5 },
        ),
        // Two faults: the count before a subcontext, a subcontext before a second group.
        (
            vec![
                (1, group(0, &[(6, SUBCONTEXT_COUNT)])),
                (2, group(0, &over)),
            ],
            TsgStateError::TooManyChannels {
                fd: 2,
                count: MAX_CHANNELS + 1,
            },
        ),
        (
            vec![
                (1, group(0, &[(5, 0)])),
                (2, group(0, &[(5, SUBCONTEXT_COUNT)])),
            ],
            TsgStateError::Subcontext {
                channel: 5,
                subcontext: SUBCONTEXT_COUNT,
            },
        ),
    ];
    for (groups, error) in refused {
        let saved = SavedTsgGpu {
            groups: groups.into_iter().collect(),
        };
        assert_eq!(TsgGpu::try_from(saved), Err(error), "{error}");
    }
    let accepted = TsgGpu {
        groups: BTreeMap::from([
            (1, group(0x77, &full)),
            (2, group(0, &[(5, SUBCONTEXT_COUNT - 1)])),
        ]),
    };
    let bytes = postcard::to_allocvec(&accepted).unwrap();
    assert_eq!(postcard::from_bytes::<TsgGpu>(&bytes).unwrap(), accepted);
    let two = TsgGpu {
        groups: BTreeMap::from([(1, group(0, &[(5, 0)])), (2, group(0, &[(5, 0)]))]),
    };
    let bytes = postcard::to_allocvec(&two).unwrap();
    assert!(postcard::from_bytes::<TsgGpu>(&bytes).is_err());
}

/// The nvsrv state checks a group's fd and its channels' fds against its files: a group on an fd
/// that is not an open tsg-gpu, a channel fd never handed out, and a channel fd open as anything
/// but an nvhost-gpu, are refused.
#[test]
fn a_saved_group_must_fit_the_files() {
    let (mut direct, tsg, [a, _]) = opened();
    direct.ioctl(tsg, BIND_CHANNEL_EX, &binding_ex(a, 3));
    let state = direct.state;
    let tsg = u32::try_from(tsg).unwrap();
    let check = |state: &NvsrvState| NvsrvState::try_from(super::super::tests::saved(state));
    assert_eq!(check(&state).as_ref(), Ok(&state));
    let mut not_a_group = state.clone();
    not_a_group.files.insert(tsg, Device::Nvmap);
    assert_eq!(
        check(&not_a_group),
        Err(NvsrvStateError::NotATimesliceGroup { fd: tsg })
    );
    let mut closed = state.clone();
    closed.files.remove(&tsg);
    assert_eq!(
        check(&closed),
        Err(NvsrvStateError::NotATimesliceGroup { fd: tsg })
    );
    // The channel's fd is issued while its file is closed; past the next fd it is not.
    let mut channel_closed = state.clone();
    channel_closed.files.remove(&a);
    assert_eq!(check(&channel_closed).as_ref(), Ok(&channel_closed));
    channel_closed.next_fd = a;
    channel_closed.files.retain(|&fd, _| fd < a);
    assert_eq!(
        check(&channel_closed),
        Err(NvsrvStateError::ChannelNotIssued { fd: a })
    );
    // A bound channel's fd that is still open must be an nvhost-gpu: a bind takes no other
    // device, and fds are never reused. An open nvmap fd, and the group's own fd, are refused.
    let mut nvmap = Direct::new();
    let map = u32::try_from(nvmap.open(Device::Nvmap)).unwrap();
    let group_tsg = nvmap.open(Device::NvhostTsgGpu);
    let channel = u32::try_from(nvmap.open(Device::NvhostGpu)).unwrap();
    nvmap.ioctl(group_tsg, BIND_CHANNEL_EX, &binding_ex(channel, 2));
    let mut on_nvmap = nvmap.state;
    let group_tsg = u32::try_from(group_tsg).unwrap();
    let mut held = on_nvmap.tsg_gpu.groups.remove(&group_tsg).unwrap();
    let sub = held.channels.remove(&channel).unwrap();
    held.channels.insert(map, sub);
    on_nvmap.tsg_gpu.groups.insert(group_tsg, held);
    assert_eq!(
        check(&on_nvmap),
        Err(NvsrvStateError::NotAChannel { fd: map })
    );
    let mut on_itself = state.clone();
    let mut held = on_itself.tsg_gpu.groups.remove(&tsg).unwrap();
    let sub = held.channels.remove(&a).unwrap();
    held.channels.insert(tsg, sub);
    on_itself.tsg_gpu.groups.insert(tsg, held);
    assert_eq!(
        check(&on_itself),
        Err(NvsrvStateError::NotAChannel { fd: tsg })
    );
    // Closed, the nvmap fd may stay bound, as a closed channel's does.
    let mut nvmap_closed = on_nvmap.clone();
    nvmap_closed.files.remove(&map);
    assert_eq!(check(&nvmap_closed).as_ref(), Ok(&nvmap_closed));
    // Both faults at once: never handed out is reported before open as another device.
    let mut both = on_nvmap.clone();
    let mut held = both.tsg_gpu.groups.remove(&group_tsg).unwrap();
    held.channels.insert(both.next_fd, 0);
    both.tsg_gpu.groups.insert(group_tsg, held);
    let unissued = both.next_fd;
    assert_eq!(
        check(&both),
        Err(NvsrvStateError::ChannelNotIssued { fd: unissued })
    );
    assert_eq!(
        NvsrvStateError::NotAChannel { fd: 6 }.to_string(),
        "channel fd 6 is bound to a timeslice group but is open as another device"
    );
    assert_eq!(
        NvsrvStateError::ChannelNotIssued { fd: 7 }.to_string(),
        "channel fd 7 is bound to a timeslice group but was never handed out"
    );
    assert_eq!(
        NvsrvStateError::NotATimesliceGroup { fd: 4 }.to_string(),
        "fd 4 has a timeslice group but is not an open nvhost-tsg-gpu"
    );
    let said = [
        (
            TsgStateError::TooManyChannels { fd: 1, count: 65 },
            "timeslice group fd 1 holds 65 channels, more than can be",
        ),
        (
            TsgStateError::Subcontext {
                channel: 5,
                subcontext: 64,
            },
            "channel fd 5 is bound with subcontext 64, which does not exist",
        ),
        (
            TsgStateError::TwoGroups { channel: 5 },
            "channel fd 5 is bound in two timeslice groups",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
}
