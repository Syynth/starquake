//! `starquake run`'s `log.txt`, end to end: a guest asks `sm:` for a service nothing provides,
//! sends it a command, and crashes; the emulator's log, the summary and the crash report each
//! name both the service and the command (the owner's ruling of 2026-11-17 on issue #207).
//!
//! # Why this file holds exactly one test
//!
//! The test installs the emulator's log as the subscriber of every thread of the process
//! ([`starquake::install_host_log`]), as the binary does, which a process can do once. Another
//! test in the same process would log into it, and `tracing` caches which statements are
//! enabled when one is first reached (see `crates/services/tests/unknown_command_logs.rs`), so
//! the test has its process to itself: do not add a second `#[test]` here.
//!
//! It needs a VM: a Mac with Apple Silicon. It skips itself only when the `CI` environment
//! variable is set (and on hosts that cannot run guests at all).

use std::ffi::OsString;
use std::path::Path;

use starquake_core::artifacts::{CRASH_JSON_FILE, CRASH_TEXT_FILE, LOG_FILE, SUMMARY_FILE};
use starquake_core::exit_code;
use starquake_core::host_log::HostLogError;
use starquake_ipc::{BUFFER_SIZE, CommandType, ReplyBuilder, RequestBuilder, RequestLayout};
use starquake_services::sm::{self, ServiceName};
use starquake_testkit::programs::{self, SendOn, Step};

mod common;

/// The service the guest asks for, which nothing provides: a made-up name, so that no service
/// a later change registers can take it (`lm`, the name used first, is now registered).
const UNKNOWN: &str = "absent";
/// The command it sends the service.
const COMMAND: u32 = 5;

/// Helpers, in a `cfg(test)` module so they may unwrap like the test does.
#[cfg(test)]
mod support {
    use super::*;

    /// `request` as the bytes of a message buffer.
    fn message(request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        buffer
    }

    /// The 32-bit word of a reply that holds the one handle it moves, found in a reply built the
    /// way the services build them.
    fn moved_handle_word() -> u32 {
        const MARKER: u32 = 0x5EA1_ED00;
        let mut reply = vec![0; BUFFER_SIZE];
        ReplyBuilder::cmif(0)
            .move_handle(MARKER)
            .build(&mut reply)
            .unwrap();
        let word = reply
            .as_chunks::<4>()
            .0
            .iter()
            .position(|word| *word == MARKER.to_le_bytes())
            .unwrap();
        u32::try_from(word).unwrap()
    }

    /// The guest: `sm:`'s `Initialize`, `GetService(UNKNOWN)`, [`COMMAND`] on the session it gave,
    /// then a crash.
    pub(crate) fn guest() -> Vec<u8> {
        let libnx =
            |id| RequestBuilder::cmif(CommandType::Request, id).layout(RequestLayout::Libnx);
        let initialize = message(
            &libnx(sm::command::INITIALIZE)
                .send_pid(0)
                .data(&0_u64.to_le_bytes()),
        );
        let name = ServiceName::new(UNKNOWN).unwrap().to_field();
        let get_service = message(&libnx(sm::command::GET_SERVICE).data(&name));
        let command = message(&libnx(COMMAND).data(&[0xEE; 8]));
        let port = format!("{}\0", sm::PORT_NAME);
        programs::requests(
            port.as_bytes(),
            &[
                Step {
                    message: &initialize,
                    on: SendOn::Port,
                },
                Step {
                    message: &get_service,
                    on: SendOn::Port,
                },
                Step {
                    message: &command,
                    on: SendOn::Reply {
                        word: moved_handle_word(),
                    },
                },
            ],
        )
        .unwrap()
    }

    pub(crate) fn text(folder: &Path, name: &str) -> String {
        std::fs::read_to_string(folder.join(name)).unwrap()
    }
}

use support::{guest, text};

#[test]
fn the_log_the_summary_and_the_crash_report_name_an_unknown_service_and_its_command() {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) || std::env::var_os("CI").is_some()
    {
        return;
    }
    starquake::install_host_log().unwrap();
    assert_eq!(
        starquake::install_host_log(),
        Err(HostLogError::AlreadyInstalled)
    );
    let folder = Path::new(env!("CARGO_TARGET_TMPDIR")).join("run_log");
    if folder.exists() {
        std::fs::remove_dir_all(&folder).unwrap();
    }
    std::fs::create_dir_all(&folder).unwrap();
    let program = folder.join("unknown.nro");
    std::fs::write(&program, guest()).unwrap();
    let artifacts = folder.join("artifacts");
    let mut line: Vec<OsString> = vec![
        "starquake".into(),
        "run".into(),
        program.into(),
        "--until=timeout=30s".into(),
        "--artifacts".into(),
        artifacts.clone().into(),
    ];
    line.extend(common::pinned_date());
    line.extend(common::recording_backend());
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let code = starquake::execute(&line, &mut stdout, &mut stderr);
    let stdout = String::from_utf8(stdout).unwrap();
    assert_eq!(code, exit_code::GUEST_CRASHED, "{stdout}");

    // The log: `sm:`'s warning names the service, the framework's the service and the command.
    let service = sm::PORT_NAME;
    assert_eq!(
        text(&artifacts, LOG_FILE),
        format!(
            "WARN starquake_services::sm: a service was asked for that nothing provides \
             name={UNKNOWN} policy=AutoStub\n\
             WARN starquake_services::framework::server: unknown command auto-stubbed: replying \
             success with a zeroed payload service={service} interface={UNKNOWN} protocol=Cmif \
             command={COMMAND}\n"
        )
    );

    // The summary.
    let summary: serde_json::Value = serde_json::from_str(&text(&artifacts, SUMMARY_FILE)).unwrap();
    assert_eq!(summary["unknown_services"], serde_json::json!([UNKNOWN]));
    assert_eq!(
        summary["stubbed_commands"],
        serde_json::json!([{
            "service": service,
            "interface": UNKNOWN,
            "protocol": "cmif",
            "command": COMMAND,
        }])
    );
    assert_eq!(
        summary["service_commands"][format!("{service}:{UNKNOWN}::{COMMAND}")],
        1
    );
    assert_eq!(summary["host_log_lines"], 2);
    assert_eq!(summary["host_log_lines_dropped"], 0);

    // The crash report.
    let crash: serde_json::Value =
        serde_json::from_str(&text(&artifacts, CRASH_JSON_FILE)).unwrap();
    assert_eq!(crash["unknown_services"], serde_json::json!([UNKNOWN]));
    assert_eq!(crash["stubbed_commands"], summary["stubbed_commands"]);
    let stubbed = &crash["last_stubbed"];
    assert_eq!(stubbed.as_array().map(Vec::len), Some(1), "{stubbed}");
    assert_eq!(stubbed[0]["service"], service);
    assert_eq!(stubbed[0]["interface"], UNKNOWN);
    assert_eq!(stubbed[0]["protocol"], "cmif");
    assert_eq!(stubbed[0]["command"], COMMAND);
    let crash_text = text(&artifacts, CRASH_TEXT_FILE);
    let tail = format!(
        "last auto-stubbed requests, oldest first:\n  #{} thread {}: {UNKNOWN} (on {service}) \
         cmif command {COMMAND}\n",
        stubbed[0]["seq"], stubbed[0]["thread"]
    );
    assert!(crash_text.ends_with(&tail), "{crash_text}");
}
