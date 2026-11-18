//! nvhost-gpu's `ZCullBind` (module docs, "The settings"), over the bench and through the server.
//!
//! The number is the one the title of issue #306 sends, `0xc010480b`, which is libnx's
//! `_NV_IOWR(0x48, 0x0B, ..)` for a 16-byte struct (`libnx/nx/source/nvidia/ioctl/nvchannel.c`
//! @ b156b477, `nvioctlChannel_ZCullBind`); the struct is laid out by hand from PR's
//! `nvhost_gpu.h` @ e5fb3b97 (`IoctlZCullBind`: `u64_le gpu_va`, `u32_le mode`, one padding
//! word; `static_assert(sizeof(IoctlZCullBind) == 16)`).

use super::*;

/// `_NV_IOWR(0x48, 0x0B, ..)`, 16 bytes.
const ZCULL_BIND: u32 = iowr(0x48, 0x0B, 16);

/// A struct whose every byte differs: the address `0x0102_0304_0506_0708`, the mode
/// `0x1112_1314`, the padding word `0x2122_2324`.
fn distinct() -> Vec<u8> {
    le(&[0x0506_0708, 0x0102_0304, 0x1112_1314, 0x2122_2324])
}

/// What fd `fd`'s channel keeps, if it has a channel.
fn kept(bench: &Bench, fd: i32) -> Option<ZcullBind> {
    bench
        .state
        .gpu()
        .channel(fd.cast_unsigned())
        .and_then(|channel| channel.zcull)
}

/// The number, the command and the layout, against the references' values written out.
#[test]
fn the_zcull_bind_number_decodes_to_its_command() {
    use crate::nvsrv::ioctl::Ioctl;
    assert_eq!(ZCULL_BIND, 0xC010_480B);
    assert_eq!(command::ZCULL_BIND, 0xB);
    assert_eq!(
        [
            layout::ZCULL_BIND_BYTES,
            layout::ZCULL_GPU_VA,
            layout::ZCULL_MODE,
        ],
        [16, 0, 8]
    );
    let ioctl = Ioctl::from_raw(ZCULL_BIND);
    assert_eq!(
        (ioctl.group(), ioctl.command(), ioctl.size()),
        (GROUP, command::ZCULL_BIND, 16)
    );
}

/// `Success`, the struct back as sent (the padding word too), the address and the mode kept,
/// each read at its own width; the other fd is untouched and the GPU front is told nothing; a
/// second call replaces the first. A mode outside the four PR's comment names is kept as well:
/// no reference checks it.
#[test]
fn zcull_bind_is_kept() {
    let mut bench = Bench::new();
    let fd = bench.open(Device::NvhostGpu);
    let other = bench.open(Device::NvhostGpu);
    let input = distinct();
    let answer = bench.ioctl(fd, ZCULL_BIND, &input);
    assert_eq!((answer.result, answer.output), (NvResult::Success, input));
    assert_eq!(
        kept(&bench, fd),
        Some(ZcullBind {
            gpu_va: 0x0102_0304_0506_0708,
            mode: 0x1112_1314,
        })
    );
    assert_eq!(kept(&bench, other), None);
    assert_eq!(bench.state.gpu().channel(other.cast_unsigned()), None);
    assert_eq!(bench.fake.calls(), []);
    // A second bind replaces the first, whatever its mode.
    let again = le(&[0x40, 0, u32::MAX, 0]);
    let answer = bench.ioctl(fd, ZCULL_BIND, &again);
    assert_eq!((answer.result, answer.output), (NvResult::Success, again));
    assert_eq!(
        kept(&bench, fd),
        Some(ZcullBind {
            gpu_va: 0x40,
            mode: u32::MAX,
        })
    );
}

/// On a channel that has its GPU channel, the bind is kept beside the rest, which stays as it
/// was, and the GPU front is still told nothing.
#[test]
fn zcull_bind_on_a_made_channel_changes_only_the_bind() {
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    let before = bench
        .state
        .gpu()
        .channel(fd.cast_unsigned())
        .cloned()
        .unwrap();
    let calls = bench.fake.calls();
    let answer = bench.ioctl(fd, ZCULL_BIND, &distinct());
    assert_eq!(answer.result, NvResult::Success);
    let after = bench
        .state
        .gpu()
        .channel(fd.cast_unsigned())
        .cloned()
        .unwrap();
    assert_eq!(
        after,
        Channel {
            zcull: Some(ZcullBind {
                gpu_va: 0x0102_0304_0506_0708,
                mode: 0x1112_1314,
            }),
            ..before
        }
    );
    assert_eq!(bench.fake.calls(), calls);
}

/// `WrapFixed` reads a struct as far as the input goes, the rest zero, and ignores bytes past
/// it; nothing is refused for its size. Each edge of the struct and of each field: the whole
/// struct and one byte more are read alike; one byte short loses the padding's top byte only;
/// the mode cut by one byte keeps its low three; the address alone leaves the mode zero, and
/// the address cut by one byte loses its top byte.
#[test]
fn a_short_or_long_struct_is_read_as_far_as_it_goes() {
    let full = distinct();
    let mut long = full.clone();
    long.push(0x99);
    let cases: [(&[u8], u64, u32); 6] = [
        (&full, 0x0102_0304_0506_0708, 0x1112_1314),
        (&long, 0x0102_0304_0506_0708, 0x1112_1314),
        (&full[..15], 0x0102_0304_0506_0708, 0x1112_1314),
        (&full[..11], 0x0102_0304_0506_0708, 0x0012_1314),
        (&full[..8], 0x0102_0304_0506_0708, 0),
        (&full[..7], 0x0002_0304_0506_0708, 0),
    ];
    for (input, gpu_va, mode) in cases {
        let mut bench = Bench::new();
        let fd = bench.open(Device::NvhostGpu);
        let answer = bench.ioctl(fd, ZCULL_BIND, input);
        let mut back = input.to_vec();
        back.resize(layout::ZCULL_BIND_BYTES, 0);
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::Success, back),
            "{} bytes",
            input.len()
        );
        assert_eq!(
            kept(&bench, fd),
            Some(ZcullBind { gpu_va, mode }),
            "{} bytes",
            input.len()
        );
    }
}

/// PR matches `'H'` 0xB on the group and the command alone, so a number with other direction
/// bits or another size is served too; `Ioctl2`, `Ioctl3` and group `'G'` 0xB go to the
/// policy, and keep nothing.
#[test]
fn other_forms_of_zcull_bind() {
    for raw in [
        0x4010_480B, // `_NV_IOW`
        iowr(0x48, 0x0B, 8),
        iowr(0x48, 0x0B, 24),
    ] {
        let mut bench = Bench::new();
        let fd = bench.open(Device::NvhostGpu);
        let answer = bench.ioctl(fd, raw, &distinct());
        assert_eq!(answer.result, NvResult::Success, "{raw:#x}");
        assert!(kept(&bench, fd).is_some(), "{raw:#x}");
    }
    let mut bench = Bench::new();
    let (fd, _, _, _) = bench.channel();
    let before = bench.state.clone();
    for (variant, raw) in [
        (IoctlVariant::Ioctl2, ZCULL_BIND),
        (IoctlVariant::Ioctl3, ZCULL_BIND),
        (IoctlVariant::Ioctl, iowr(0x47, 0x0B, 16)),
    ] {
        let (answer, _) = bench.call(variant, fd, raw, &distinct(), &[]);
        assert_eq!(
            answer.result,
            NvResult::NotImplemented,
            "{variant:?} {raw:#x}"
        );
    }
    assert_eq!(bench.state, before);
}

/// A server with no channel front leaves `ZCullBind` to the policy, as every nvhost-gpu ioctl.
#[test]
fn without_a_channel_front_zcull_bind_goes_to_the_policy() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostGpu);
    let (reply, out) = rig.ioctl(fd, ZCULL_BIND, &distinct(), 16);
    assert_eq!(
        (reply, out),
        (ok(&[NvResult::NotImplemented.raw()]), vec![STALE; 16])
    );
    assert_eq!(rig.state().gpu().channel(fd.cast_unsigned()), None);
}

/// Through the server: the result word, and the struct written back whole.
#[test]
fn through_the_server_zcull_bind_writes_the_struct_back() {
    let mut live = Live::new();
    let fd = live.channel();
    let (reply, out) = live.rig.ioctl(fd, ZCULL_BIND, &distinct(), 16);
    assert_eq!((reply, out), (ok(&[0]), distinct()));
}

/// The bind is saved and restored with the channel: after a restore into a new server the
/// channel holds it, and a new bind on the restored channel replaces it.
#[test]
fn the_zcull_bind_survives_a_restore() {
    let mut live = Live::new();
    let fd = live.channel();
    let (reply, _) = live.rig.ioctl(fd, ZCULL_BIND, &distinct(), 16);
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
    let zcull = |live: &Live| {
        live.rig
            .state()
            .gpu()
            .channel(fd.cast_unsigned())
            .and_then(|channel| channel.zcull)
    };
    assert_eq!(
        zcull(&live),
        Some(ZcullBind {
            gpu_va: 0x0102_0304_0506_0708,
            mode: 0x1112_1314,
        })
    );
    let (reply, _) = live.rig.ioctl(fd, ZCULL_BIND, &le(&[0x80, 0, 2, 0]), 16);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(
        zcull(&live),
        Some(ZcullBind {
            gpu_va: 0x80,
            mode: 2,
        })
    );
}

/// Any saved bind restores, on a channel with or without its GPU channel: `ZCullBind` keeps any
/// address and mode, on any channel.
#[test]
fn any_saved_zcull_bind_restores() {
    for made in [
        None,
        Some(Made {
            channel: 3,
            space: AsId(1),
        }),
    ] {
        for zcull in [
            ZcullBind { gpu_va: 0, mode: 0 },
            ZcullBind {
                gpu_va: u64::MAX,
                mode: u32::MAX,
            },
        ] {
            let channel = Channel {
                syncpoint: made.map(|_| FIRST_ALLOCATABLE),
                made,
                zcull: Some(zcull),
                ..Channel::default()
            };
            let restored = NvhostGpu::try_from(saved_channel(channel.clone()));
            assert_eq!(
                restored.map(|gpu| gpu.channel(7).cloned()),
                Ok(Some(channel))
            );
        }
    }
}

/// A fresh channel has no bind (the references' `zcull_params{}`, which nothing reads, is all
/// zero).
#[test]
fn a_fresh_channel_has_no_zcull_bind() {
    assert_eq!(Channel::default().zcull, None);
}
