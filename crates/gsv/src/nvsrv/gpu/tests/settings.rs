//! nvhost-gpu's settings (module docs, "The settings"): `ChannelSetTimeout`, `SetErrorNotifier`,
//! `SetPreemptionModes`, `SetClientData` and `GetClientData`, over the bench and through the
//! server.
//!
//! The numbers are written as libnx writes them (`libnx/nx/source/nvidia/ioctl/nvchannel.c` @
//! b156b477: `nvioctlChannel_SetTimeout`, `_NV_IOW(0x48, 0x03, ..)`;
//! `nvioctlChannel_SetErrorNotifier`, `_NV_IOWR(0x48, 0x0C, ..)`; `nvioctlChannel_SetUserData`,
//! `_NV_IOW(0x47, 0x14, ..)`), PR's `0xc008481f`, and the structs laid out by hand from PR's
//! `nvhost_gpu.h` @ e5fb3b97 (`IoctlChannelSetTimeout`, `IoctlSetErrorNotifier`,
//! `IoctlPreemptionModes`, `IoctlClientData`).

use super::*;

/// libnx `_NV_IOW(group, nr, size)`: only the "in" bit.
const fn iow(group: u32, nr: u32, size: u32) -> u32 {
    0x4000_0000 | (size << 16) | (group << 8) | nr
}

/// `_NV_IOW(0x48, 0x03, ..)`, 4 bytes.
const SET_TIMEOUT: u32 = iow(0x48, 0x03, 4);
/// `_NV_IOWR(0x48, 0x0C, ..)`, 24 bytes.
const SET_ERROR_NOTIFIER: u32 = iowr(0x48, 0x0C, 24);
/// `_NV_IOW(0x47, 0x14, ..)`, 8 bytes.
const SET_CLIENT_DATA: u32 = iow(0x47, 0x14, 8);
/// `GetClientData` as `_NV_IOWR(0x47, 0x15, ..)`, 8 bytes: no title or libnx call is known to
/// send it, so the number is built the way the others are (a guess at the direction bits; PR
/// matches the group and the command only).
const GET_CLIENT_DATA: u32 = iowr(0x47, 0x15, 8);

/// The numbers the title of issue #300 sends, and the constants they decode to, against the
/// references' values written out.
#[test]
fn the_settings_numbers_decode_to_their_commands() {
    use crate::nvsrv::ioctl::Ioctl;
    assert_eq!(
        [SET_TIMEOUT, SET_ERROR_NOTIFIER, SET_CLIENT_DATA],
        [0x4004_4803, 0xC018_480C, 0x4008_4714]
    );
    assert_eq!(SET_PREEMPTION_MODES, 0xC008_481F);
    assert_eq!(CLIENT_GROUP, 0x47);
    assert_eq!(
        [
            command::CHANNEL_SET_TIMEOUT,
            command::SET_ERROR_NOTIFIER,
            command::SET_PREEMPTION_MODES,
            client_command::SET_CLIENT_DATA,
            client_command::GET_CLIENT_DATA,
        ],
        [0x3, 0xC, 0x1F, 0x14, 0x15]
    );
    assert_eq!(
        [
            preemption::KEEP,
            preemption::DEFAULT,
            preemption::GRAPHICS_MAX,
            preemption::COMPUTE_MAX,
        ],
        [0, 1, 1, 2]
    );
    assert_eq!(
        [
            layout::TIMEOUT_BYTES,
            layout::TIMEOUT,
            layout::ERROR_NOTIFIER_BYTES,
            layout::NOTIFIER_OFFSET,
            layout::NOTIFIER_SIZE,
            layout::NOTIFIER_MEM,
            layout::PREEMPTION_BYTES,
            layout::PREEMPTION_GRAPHICS,
            layout::PREEMPTION_COMPUTE,
            layout::CLIENT_DATA_BYTES,
            layout::CLIENT_DATA,
        ],
        [4, 0, 24, 0, 8, 16, 8, 0, 4, 8, 0]
    );
    let cases = [
        (SET_TIMEOUT, GROUP, command::CHANNEL_SET_TIMEOUT, 4),
        (SET_ERROR_NOTIFIER, GROUP, command::SET_ERROR_NOTIFIER, 24),
        (
            SET_PREEMPTION_MODES,
            GROUP,
            command::SET_PREEMPTION_MODES,
            8,
        ),
        (
            SET_CLIENT_DATA,
            CLIENT_GROUP,
            client_command::SET_CLIENT_DATA,
            8,
        ),
        (
            GET_CLIENT_DATA,
            CLIENT_GROUP,
            client_command::GET_CLIENT_DATA,
            8,
        ),
    ];
    for (raw, group, cmd, size) in cases {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(
            (ioctl.group(), ioctl.command(), ioctl.size()),
            (group, cmd, size),
            "{raw:#x}"
        );
    }
}

// ----- ChannelSetTimeout -------------------------------------------------------------------------

/// `Success`, the struct back as sent, nothing kept: no channel is made for the fd.
#[test]
fn channel_set_timeout_keeps_nothing() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    let input = le(&[0x0102_0304]);
    let answer = bench.ioctl(fd, SET_TIMEOUT, &input);
    assert_eq!((answer.result, answer.output), (NvResult::Success, input));
    assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
    // On a channel with settings, they stay as they were.
    let (fd, _, _, _) = bench.channel();
    let before = bench.state.clone();
    let answer = bench.ioctl(fd, SET_TIMEOUT, &le(&[7]));
    assert_eq!(answer.result, NvResult::Success);
    assert_eq!(bench.state, before);
    assert_eq!(bench.fake.calls().len(), 1, "only the channel's creation");
}

// ----- SetErrorNotifier --------------------------------------------------------------------------

/// The notifier's offset, size and handle are kept, each read at its own width (values that
/// differ in every byte); the struct comes back as sent; a second call replaces the first; a
/// short struct is read as far as it goes, the rest zero.
#[test]
fn the_error_notifier_is_kept() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    let other = bench.open(Device::NvhostGpu);
    let input = le(&[
        0x0506_0708,
        0x0102_0304,
        0x1516_1718,
        0x1112_1314,
        0x2122_2324,
        0x3132_3334,
    ]);
    let answer = bench.ioctl(fd, SET_ERROR_NOTIFIER, &input);
    assert_eq!((answer.result, answer.output), (NvResult::Success, input));
    let kept = |bench: &Bench, fd: i32| {
        bench
            .state
            .gpu()
            .channel(fd.cast_unsigned())
            .and_then(|channel| channel.error_notifier)
    };
    assert_eq!(
        kept(&bench, fd),
        Some(ErrorNotifier {
            offset: 0x0102_0304_0506_0708,
            size: 0x1112_1314_1516_1718,
            mem: 0x2122_2324,
        })
    );
    // The other fd is untouched, and the GPU front was told nothing.
    assert_eq!(kept(&bench, other), None);
    assert_eq!(bench.fake.calls(), []);
    // A struct cut after the size: the handle is read as zero.
    let short = le(&[0x41, 0, 0x42, 0]);
    let answer = bench.ioctl(fd, SET_ERROR_NOTIFIER, &short);
    assert_eq!(answer.result, NvResult::Success);
    let mut back = short.clone();
    back.resize(layout::ERROR_NOTIFIER_BYTES, 0);
    assert_eq!(answer.output, back);
    assert_eq!(
        kept(&bench, fd),
        Some(ErrorNotifier {
            offset: 0x41,
            size: 0x42,
            mem: 0,
        })
    );
}

// ----- SetClientData and GetClientData -----------------------------------------------------------

/// `GetClientData` gives back what `SetClientData` kept on the same fd, 0 before any and on
/// another fd; reading makes no channel; a short struct is read as far as it goes.
#[test]
fn client_data_round_trips_per_fd() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    let other = bench.open(Device::NvhostGpu);
    let stale = le(&[0xAAAA_AAAA, 0xBBBB_BBBB]);
    let answer = bench.ioctl(fd, GET_CLIENT_DATA, &stale);
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::Success, vec![0; 8])
    );
    assert_eq!(bench.state.gpu().channel(fd.cast_unsigned()), None);
    let data = le(&[0x0506_0708, 0x0102_0304]);
    let answer = bench.ioctl(fd, SET_CLIENT_DATA, &data);
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::Success, data.clone())
    );
    let channel = bench.state.gpu().channel(fd.cast_unsigned()).unwrap();
    assert_eq!(channel.client_data, 0x0102_0304_0506_0708);
    assert_eq!(bench.ioctl(fd, GET_CLIENT_DATA, &stale).output, data);
    // Another fd's data is its own.
    assert_eq!(
        bench.ioctl(other, GET_CLIENT_DATA, &stale).output,
        vec![0; 8]
    );
    // Three bytes: the rest of the u64 is zero.
    bench.ioctl(fd, SET_CLIENT_DATA, &[0x11, 0x22, 0x33]);
    let output = bench.ioctl(fd, GET_CLIENT_DATA, &[]).output;
    assert_eq!(output, 0x33_2211_u64.to_le_bytes());
    assert_eq!(bench.fake.calls(), []);
}

// ----- SetPreemptionModes ------------------------------------------------------------------------

/// The modes as the struct carries them.
fn modes(graphics: u32, compute: u32) -> Vec<u8> {
    le(&[graphics, compute])
}

/// The checks in order: no GPU channel is `InvalidState` (even with modes past their limits,
/// and with a syncpoint but no GPU channel); then a mode past its limit is `NotSupported`, each
/// exactly one past, while each limit itself is taken. Modes of zero keep what was set. The
/// struct comes back as sent whatever the result, and a refusal changes nothing.
#[test]
fn preemption_modes_check_the_channel_then_the_modes() {
    let mut bench = Bench::new();
    let fresh = bench.open(Device::NvhostGpu);
    let unbound = bench.open(Device::NvhostGpu);
    bench.alloc(unbound);
    let over = modes(preemption::GRAPHICS_MAX + 1, preemption::COMPUTE_MAX + 1);
    for fd in [fresh, unbound] {
        for input in [modes(1, 2), over.clone()] {
            let before = bench.state.clone();
            let answer = bench.ioctl(fd, SET_PREEMPTION_MODES, &input);
            assert_eq!(
                (answer.result, answer.output),
                (NvResult::InvalidState, input),
                "{fd}"
            );
            assert_eq!(bench.state, before);
        }
    }
    let (fd, _, _, _) = bench.channel();
    let preemption = |bench: &Bench| {
        bench
            .state
            .gpu()
            .channel(fd.cast_unsigned())
            .unwrap()
            .preemption
    };
    assert_eq!(preemption(&bench), Preemption::default());
    for input in [
        modes(preemption::GRAPHICS_MAX + 1, preemption::DEFAULT),
        modes(preemption::DEFAULT, preemption::COMPUTE_MAX + 1),
    ] {
        let answer = bench.ioctl(fd, SET_PREEMPTION_MODES, &input);
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::NotSupported, input)
        );
        assert_eq!(preemption(&bench), Preemption::default());
    }
    let input = modes(preemption::GRAPHICS_MAX, preemption::COMPUTE_MAX);
    let answer = bench.ioctl(fd, SET_PREEMPTION_MODES, &input);
    assert_eq!((answer.result, answer.output), (NvResult::Success, input));
    let set = Preemption {
        graphics: preemption::GRAPHICS_MAX,
        compute: preemption::COMPUTE_MAX,
    };
    assert_eq!(preemption(&bench), set);
    // Zero keeps the mode: both, then each alone.
    let keep = modes(preemption::KEEP, preemption::KEEP);
    assert_eq!(
        bench.ioctl(fd, SET_PREEMPTION_MODES, &keep).result,
        NvResult::Success
    );
    assert_eq!(preemption(&bench), set);
    let compute_only = modes(preemption::KEEP, 1);
    assert_eq!(
        bench.ioctl(fd, SET_PREEMPTION_MODES, &compute_only).result,
        NvResult::Success
    );
    assert_eq!(
        preemption(&bench),
        Preemption {
            graphics: preemption::GRAPHICS_MAX,
            compute: 1,
        }
    );
    let graphics_only = modes(1, preemption::KEEP);
    assert_eq!(
        bench.ioctl(fd, SET_PREEMPTION_MODES, &graphics_only).result,
        NvResult::Success
    );
    assert_eq!(preemption(&bench).compute, 1);
    // The modes are kept, not given to the GPU.
    assert_eq!(bench.fake.calls().len(), 1, "only the channel's creation");
}

/// Command 0x1F with any number but PR's, and every setting on `Ioctl2` and `Ioctl3`, or in the
/// client group with another command, go to the policy; nothing is kept.
#[test]
fn other_forms_of_the_settings_go_to_the_policy() {
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    let before = bench.state.clone();
    let input = le(&[1, 2, 3, 4, 5, 6]);
    let cases = [
        (IoctlVariant::Ioctl, iowr(0x48, 0x1F, 16)),
        (IoctlVariant::Ioctl, iow(0x48, 0x1F, 8)),
        (IoctlVariant::Ioctl, iowr(0x47, 0x16, 8)),
        (IoctlVariant::Ioctl, iowr(0x49, 0x14, 8)),
        (IoctlVariant::Ioctl2, SET_PREEMPTION_MODES),
        (IoctlVariant::Ioctl3, SET_PREEMPTION_MODES),
        (IoctlVariant::Ioctl2, SET_CLIENT_DATA),
        (IoctlVariant::Ioctl3, GET_CLIENT_DATA),
        (IoctlVariant::Ioctl2, SET_ERROR_NOTIFIER),
        (IoctlVariant::Ioctl3, SET_TIMEOUT),
    ];
    for (variant, raw) in cases {
        let (answer, _) = bench.call(variant, fd, raw, &input, &[]);
        assert_eq!(
            answer.result,
            NvResult::NotImplemented,
            "{variant:?} {raw:#x}"
        );
    }
    assert_eq!(bench.state, before);
}

/// A server with no channel front leaves the settings to the policy too, as every nvhost-gpu
/// ioctl (module docs, `super::super`, "What each device does").
#[test]
fn without_a_channel_front_the_settings_go_to_the_policy() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostGpu);
    let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
    for (raw, len) in [
        (SET_TIMEOUT, 4),
        (SET_ERROR_NOTIFIER, 24),
        (SET_PREEMPTION_MODES, 8),
        (SET_CLIENT_DATA, 8),
        (GET_CLIENT_DATA, 8),
    ] {
        let (reply, _) = rig.ioctl(fd, raw, &vec![0x5A; len], len);
        assert_eq!(reply, not_implemented, "{raw:#x}");
    }
    assert_eq!(rig.state().gpu().channel(fd.cast_unsigned()), None);
}

// ----- Through the server, and saved -------------------------------------------------------------

/// Through the server: an `_NV_IOW` number writes no out buffer; an `_NV_IOWR` one writes the
/// struct; the reply is the result word.
#[test]
fn through_the_server_only_iowr_numbers_write_back() {
    let mut live = Live::new();
    let fd = live.channel();
    let (reply, out) = live.rig.ioctl(fd, SET_TIMEOUT, &le(&[0x10]), 4);
    assert_eq!((reply, out), (ok(&[0]), vec![STALE; 4]));
    let (reply, out) = live.rig.ioctl(fd, SET_CLIENT_DATA, &le(&[0x10, 0x20]), 8);
    assert_eq!((reply, out), (ok(&[0]), vec![STALE; 8]));
    let (reply, out) = live.rig.ioctl(fd, GET_CLIENT_DATA, &[0; 8], 8);
    assert_eq!((reply, out), (ok(&[0]), le(&[0x10, 0x20])));
    let input = modes(1, 2);
    let (reply, out) = live.rig.ioctl(fd, SET_PREEMPTION_MODES, &input, 8);
    assert_eq!((reply, out), (ok(&[0]), input));
    let notifier = le(&[1, 2, 3, 4, 5, 6]);
    let (reply, out) = live.rig.ioctl(fd, SET_ERROR_NOTIFIER, &notifier, 24);
    assert_eq!((reply, out), (ok(&[0]), notifier));
}

/// The settings are saved and restored with the channel: after a restore into a new server,
/// `GetClientData` reads what was set, and the modes and notifier are as they were.
#[test]
fn the_settings_survive_a_restore() {
    let mut live = Live::new();
    let fd = live.channel();
    let data = le(&[0x5152_5354, 0x6162_6364]);
    live.rig.ioctl(fd, SET_CLIENT_DATA, &data, 8);
    live.rig
        .ioctl(fd, SET_ERROR_NOTIFIER, &le(&[7, 0, 8, 0, 9, 0]), 24);
    let (reply, _) = live.rig.ioctl(fd, SET_PREEMPTION_MODES, &modes(1, 2), 8);
    assert_eq!(reply, ok(&[0]));
    let state = live.rig.state().clone();
    let saved = live.rig.server.save().unwrap();

    let family = NvsrvFamily::new(Arc::clone(&live.rig.shared), as_front(&live.rig.front))
        .with_channel_front(channel_front(&live.fake));
    live.rig.server = nvsrv_server(
        Arc::clone(&live.rig.shared),
        as_front(&live.rig.front),
        UnknownPolicy::Strict,
    )
    .with_family(family)
    .with_service_id(live.rig.server.service_id());
    live.rig.server.restore(&saved).unwrap();
    assert_eq!(live.rig.state(), &state);
    let (reply, out) = live.rig.ioctl(fd, GET_CLIENT_DATA, &[0; 8], 8);
    assert_eq!((reply, out), (ok(&[0]), data));
    let channel = live.rig.state().gpu().channel(fd.cast_unsigned()).unwrap();
    assert_eq!(
        channel.error_notifier,
        Some(ErrorNotifier {
            offset: 7,
            size: 8,
            mem: 9,
        })
    );
    assert_eq!(
        channel.preemption,
        Preemption {
            graphics: 1,
            compute: 2,
        }
    );
}

/// A fresh channel's settings are the references' defaults: client data 0 (`user_data{}`), no
/// notifier, both modes 1 (`graphics_preemption_mode{1}`, `compute_preemption_mode{1}`).
#[test]
fn a_fresh_channel_s_settings_are_the_defaults() {
    let channel = Channel::default();
    assert_eq!(channel.client_data, 0);
    assert_eq!(channel.error_notifier, None);
    assert_eq!(
        channel.preemption,
        Preemption {
            graphics: 1,
            compute: 1,
        }
    );
}

/// Saved preemption modes `SetPreemptionModes` cannot set are refused: zero, past a limit, or
/// other than the default on a channel with no GPU channel; each limit on a made channel, and
/// the defaults on any channel, restore. A bad class is reported before bad modes.
#[test]
fn a_saved_preemption_mode_no_ioctl_sets_is_refused() {
    let made = Some(Made {
        channel: 3,
        space: AsId(1),
    });
    let with = |made: Option<Made>, graphics: u32, compute: u32| Channel {
        syncpoint: made.map(|_| FIRST_ALLOCATABLE),
        made,
        preemption: Preemption { graphics, compute },
        ..Channel::default()
    };
    let refused = [
        (made, preemption::KEEP, preemption::DEFAULT),
        (made, preemption::DEFAULT, preemption::KEEP),
        (made, preemption::GRAPHICS_MAX + 1, preemption::DEFAULT),
        (made, preemption::DEFAULT, preemption::COMPUTE_MAX + 1),
        (None, preemption::DEFAULT, preemption::COMPUTE_MAX),
    ];
    for (made, graphics, compute) in refused {
        let saved = saved_channel(with(made, graphics, compute));
        assert_eq!(
            NvhostGpu::try_from(saved),
            Err(GpuStateError::Preemption {
                fd: 7,
                graphics,
                compute
            }),
            "{made:?} {graphics} {compute}"
        );
    }
    for (made, graphics, compute) in [
        (made, preemption::GRAPHICS_MAX, preemption::COMPUTE_MAX),
        (made, preemption::DEFAULT, preemption::DEFAULT),
        (None, preemption::DEFAULT, preemption::DEFAULT),
    ] {
        let saved = saved_channel(with(made, graphics, compute));
        assert!(NvhostGpu::try_from(saved).is_ok(), "{graphics} {compute}");
    }
    // A syncpoint but no GPU channel: only the default restores, so the check reads whether the
    // GPU channel is made, not whether the syncpoint is set.
    let syncpoint_only = Channel {
        syncpoint: Some(FIRST_ALLOCATABLE),
        ..with(None, preemption::DEFAULT, preemption::COMPUTE_MAX)
    };
    assert_eq!(
        NvhostGpu::try_from(saved_channel(syncpoint_only)),
        Err(GpuStateError::Preemption {
            fd: 7,
            graphics: preemption::DEFAULT,
            compute: preemption::COMPUTE_MAX,
        })
    );
    // Bad modes on fd 7, and fds 7 and 9 sharing a syncpoint: the modes are reported.
    let shared = SavedNvhostGpu {
        channels: BTreeMap::from([
            (7, with(made, preemption::KEEP, preemption::DEFAULT)),
            (
                9,
                Channel {
                    syncpoint: Some(FIRST_ALLOCATABLE),
                    ..Channel::default()
                },
            ),
        ]),
    };
    assert_eq!(
        NvhostGpu::try_from(shared),
        Err(GpuStateError::Preemption {
            fd: 7,
            graphics: preemption::KEEP,
            compute: preemption::DEFAULT,
        })
    );
    // Both faults: the class is reported.
    let mut both = with(made, preemption::KEEP, preemption::KEEP);
    both.classes = vec![0x1234];
    assert_eq!(
        NvhostGpu::try_from(saved_channel(both)),
        Err(GpuStateError::Class {
            fd: 7,
            class: 0x1234
        })
    );
    assert_eq!(
        GpuStateError::Preemption {
            fd: 7,
            graphics: 0,
            compute: 3
        }
        .to_string(),
        "channel fd 7 has graphics preemption mode 0 and compute mode 3, which it cannot have"
    );
}
