//! `starquake gpu-replay` end to end: a capture the GPU crate's builder made, written to a file,
//! replayed through the command line, with its output and exit code.

/// In a `cfg(test)` module so the helpers may unwrap like the tests do.
#[cfg(test)]
mod replay {
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;

    use starquake::execute;
    use starquake_core::exit_code;
    use starquake_gpu::capture::builder::{BuilderConfig, CaptureBuilder, PendingShaders};
    use starquake_gpu::classes::clc76f;
    use starquake_gpu::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
    use starquake_gpu::interface::{AsId, GpfifoEntry, Submission, SyncpointId};
    use starquake_gpu::sema::{self, HostSemaphore, PayloadSize, ReleaseForm};
    use starquake_memory::{PAGE_SIZE, PageRange};

    const CPU_BASE: u64 = 0x8000_0000;
    const GPU_BASE: u64 = 0x12_3400_0000;
    const SEM: u64 = GPU_BASE + PAGE_SIZE + 0x20;
    const SPACE: AsId = AsId(7);

    /// A folder for one test, unique to it and to the test process, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "starquake-app-gpu-replay-{}-{test}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A capture of one release of 0x77 at `SEM`, as a file in `scratch`.
    fn capture_file(scratch: &Scratch) -> PathBuf {
        let mut builder = CaptureBuilder::new(
            BuilderConfig::synthetic("app test", CPU_BASE, 4 * 4096),
            Box::new(PendingShaders),
        )
        .unwrap();
        builder
            .address_space(
                SPACE,
                PageRange::new(GPU_BASE, 4 * PAGE_SIZE).unwrap(),
                CPU_BASE,
            )
            .unwrap();
        let channel = builder
            .channel(SPACE, SyncpointId::new(2).unwrap())
            .unwrap();
        let op = sema::encode_semaphored(&HostSemaphore::Release {
            form: ReleaseForm {
                size: PayloadSize::Four,
                timestamp: false,
            },
            wfi: false,
        })
        .unwrap();
        let mut pb = PushBuffer::new();
        pb.methods(
            MethodMode::Incrementing,
            0,
            u16::try_from(clc76f::SEMAPHOREA.method()).unwrap(),
            &[
                u32::try_from(SEM >> 32).unwrap(),
                u32::try_from(SEM & 0xffff_ffff).unwrap(),
                0x77,
                op,
            ],
        )
        .unwrap();
        let words = pb.into_words();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        builder.cpu_write(SPACE, GPU_BASE, &bytes).unwrap();
        let entry = GpEntry {
            address: GPU_BASE,
            length: u32::try_from(words.len()).unwrap(),
            sync: Sync::Proceed,
            ..GpEntry::default()
        };
        builder
            .submit(
                channel,
                Submission {
                    entries: vec![GpfifoEntry(entry.to_raw().unwrap())],
                    wait: None,
                    increments: 0,
                    wfi: false,
                },
            )
            .unwrap();
        builder.run().unwrap();
        let file = builder.finish().unwrap().to_bytes().unwrap();
        let path = scratch.0.join("capture.sqgc");
        fs::write(&path, file).unwrap();
        path
    }

    fn run(args: &[OsString]) -> (u8, String, String) {
        let mut command = vec![OsString::from("starquake"), OsString::from("gpu-replay")];
        command.extend_from_slice(args);
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = execute(&command, &mut stdout, &mut stderr);
        (
            code,
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        )
    }

    #[test]
    fn a_capture_replays_with_exit_code_0_and_prints_its_writes() {
        let scratch = Scratch::new("writes");
        let path = capture_file(&scratch);
        let (code, stdout, stderr) = run(&[path.into(), "--dump".into(), "writes".into()]);
        assert_eq!(code, exit_code::STOP_CONDITION);
        assert_eq!(stderr, "");
        assert_eq!(
            stdout,
            format!(
                "replayed the whole capture: 1 ring entries, 0 presents; the ring and the render \
                 stream match the capture\nback space 7 wrote 4 bytes at {SEM:#x}: 77 00 00 00\n"
            )
        );
    }

    #[test]
    fn an_unreadable_or_malformed_capture_is_an_emulator_error() {
        let scratch = Scratch::new("errors");
        let missing = scratch.0.join("missing.sqgc");
        let (code, stdout, stderr) = run(&[missing.clone().into()]);
        assert_eq!(code, exit_code::EMULATOR_ERROR);
        assert_eq!(stdout, "");
        assert!(stderr.starts_with(&format!(
            "starquake: the capture {} could not be read: ",
            missing.display()
        )));

        let junk = scratch.0.join("junk.sqgc");
        fs::write(&junk, b"SQGC\x09\x00\x00\x00").unwrap();
        let (code, _, stderr) = run(&[junk.clone().into()]);
        assert_eq!(code, exit_code::EMULATOR_ERROR);
        assert_eq!(
            stderr,
            format!(
                "starquake: the capture {} is malformed: the capture is format version 9; this \
                 build reads version 1 only\n",
                junk.display()
            )
        );
    }

    #[test]
    fn a_command_line_gpu_replay_cannot_parse_is_a_usage_error() {
        let (code, stdout, stderr) = run(&["a.sqgc".into(), "--until".into(), "frames=3".into()]);
        assert_eq!(code, exit_code::EMULATOR_ERROR);
        assert_eq!(stdout, "");
        assert!(stderr.starts_with(
            "starquake: `--until frames=3` is not a stop condition of gpu-replay: write \
             `present=<n>`, n a whole number from 1\n\nusage: "
        ));
        let (code, stdout, _) = run(&["--help".into()]);
        assert_eq!(code, exit_code::STOP_CONDITION);
        assert!(stdout.contains("starquake gpu-replay <capture>"));
    }
}
