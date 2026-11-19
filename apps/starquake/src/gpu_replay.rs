//! `starquake gpu-replay <capture> [--until present=<n>] [--dump ring|render|writes]`: the
//! command line of the GPU replay (observability spec, "Headless runs";
//! `starquake_core::gpu_replay` does the work).
//!
//! The capture is the one file argument. `--until present=<n>` stops after the `n`-th present
//! (`n` a whole number from 1), and `--dump` names what to print after the summary line. Each
//! flag takes its value as the next argument or after an `=`, and may be given once. Exit codes
//! follow the headless contract: 0 the replay reached its stop condition, 2 anything else (a
//! command line it cannot parse, a capture it cannot read, a malformed capture, a divergence).

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;

use starquake_core::exit_code;
use starquake_core::gpu_replay::{Dump, GpuReplayArgs, gpu_replay};

/// Why a `gpu-replay` command line could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuReplayUsage {
    /// No capture was given.
    #[error("`gpu-replay` needs a capture file")]
    MissingCapture,
    /// A second capture was given.
    #[error("`gpu-replay` takes one capture file; `{0}` is a second one")]
    ExtraCapture(String),
    /// A flag `gpu-replay` does not have.
    #[error("`gpu-replay` has no flag `{0}`")]
    UnknownFlag(String),
    /// A flag that needs a value was the last argument.
    #[error("`{0}` needs a value")]
    MissingValue(&'static str),
    /// A flag was given twice.
    #[error("`{0}` was given twice")]
    Repeated(&'static str),
    /// `--until` with something other than `present=<n>`, `n` from 1.
    #[error(
        "`--until {0}` is not a stop condition of gpu-replay: write `present=<n>`, n a whole \
         number from 1"
    )]
    BadUntil(String),
    /// `--dump` with something other than the three dumps.
    #[error("`--dump {0}` is not a dump: write `ring`, `render` or `writes`")]
    BadDump(String),
}

/// Parses `present=<n>`.
fn parse_until(condition: &str) -> Result<u64, GpuReplayUsage> {
    condition
        .strip_prefix("present=")
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|digits| digits.parse::<u64>().ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| GpuReplayUsage::BadUntil(condition.to_owned()))
}

/// Parses the arguments after `gpu-replay`; `None` for `--help`.
pub fn parse(rest: &[OsString]) -> Result<Option<GpuReplayArgs>, GpuReplayUsage> {
    let (mut capture, mut until_present, mut dump) = (None, None, None);
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        let shown = arg.to_string_lossy().into_owned();
        let (flag, inline) = match shown.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
            _ => (shown.as_str(), None),
        };
        let mut value = |name: &'static str| match inline {
            Some(value) => Ok(value.to_owned()),
            None => rest
                .next()
                .map(|value| value.to_string_lossy().into_owned())
                .ok_or(GpuReplayUsage::MissingValue(name)),
        };
        match flag {
            "--help" | "-h" => return Ok(None),
            "--until" => {
                let n = parse_until(&value("--until")?)?;
                if until_present.replace(n).is_some() {
                    return Err(GpuReplayUsage::Repeated("--until"));
                }
            }
            "--dump" => {
                let name = value("--dump")?;
                let named = Dump::from_name(&name).ok_or(GpuReplayUsage::BadDump(name))?;
                if dump.replace(named).is_some() {
                    return Err(GpuReplayUsage::Repeated("--dump"));
                }
            }
            flag if flag.starts_with('-') && flag.len() > 1 => {
                return Err(GpuReplayUsage::UnknownFlag(flag.to_owned()));
            }
            _ => {
                if capture.replace(PathBuf::from(arg)).is_some() {
                    return Err(GpuReplayUsage::ExtraCapture(shown));
                }
            }
        }
    }
    Ok(Some(GpuReplayArgs {
        capture: capture.ok_or(GpuReplayUsage::MissingCapture)?,
        until_present,
        dump,
    }))
}

/// Replays the capture, prints what it gives to `stdout` or why it stopped to `stderr`, and
/// returns the exit code.
pub fn run(args: &GpuReplayArgs, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match gpu_replay(args) {
        Ok(text) => {
            let _ = write!(stdout, "{text}");
            exit_code::STOP_CONDITION
        }
        Err(error) => {
            let _ = writeln!(stderr, "starquake: {error}");
            error.exit_code()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    #[test]
    fn a_capture_and_both_flags_in_either_form_and_any_order() {
        let expected = GpuReplayArgs {
            capture: PathBuf::from("frame.sqgc"),
            until_present: Some(12),
            dump: Some(Dump::Render),
        };
        for words in [
            &["frame.sqgc", "--until", "present=12", "--dump", "render"][..],
            &["--dump=render", "frame.sqgc", "--until=present=12"],
            &["--until", "present=12", "--dump", "render", "frame.sqgc"],
        ] {
            assert_eq!(parse(&args(words)), Ok(Some(expected.clone())), "{words:?}");
        }
        assert_eq!(
            parse(&args(&["a.sqgc"])),
            Ok(Some(GpuReplayArgs {
                capture: PathBuf::from("a.sqgc"),
                until_present: None,
                dump: None,
            }))
        );
        for (word, dump) in [
            ("ring", Dump::Ring),
            ("render", Dump::Render),
            ("writes", Dump::Writes),
        ] {
            assert_eq!(
                parse(&args(&["a", "--dump", word])).unwrap().unwrap().dump,
                Some(dump)
            );
        }
        assert_eq!(parse(&args(&["a", "--help"])), Ok(None));
        assert_eq!(parse(&args(&["-h"])), Ok(None));
    }

    #[test]
    fn a_present_count_is_a_whole_number_from_one() {
        assert_eq!(
            parse(&args(&["a", "--until", "present=1"])).unwrap().unwrap().until_present,
            Some(1)
        );
        assert_eq!(
            parse(&args(&["a", "--until", "present=18446744073709551615"]))
                .unwrap()
                .unwrap()
                .until_present,
            Some(u64::MAX)
        );
        for bad in [
            "present=0",
            "present=",
            "present=-1",
            "present=+1",
            "present=1x",
            "present=18446744073709551616",
            "frames=1",
            "timeout=1s",
        ] {
            assert_eq!(
                parse(&args(&["a", "--until", bad])),
                Err(GpuReplayUsage::BadUntil(bad.to_owned())),
                "{bad}"
            );
        }
        assert_eq!(
            GpuReplayUsage::BadUntil("present=0".to_owned()).to_string(),
            "`--until present=0` is not a stop condition of gpu-replay: write `present=<n>`, n a \
             whole number from 1"
        );
    }

    #[test]
    fn each_malformed_command_line_has_its_own_error() {
        let cases: [(&[&str], GpuReplayUsage, &str); 7] = [
            (
                &[],
                GpuReplayUsage::MissingCapture,
                "`gpu-replay` needs a capture file",
            ),
            (
                &["a", "b"],
                GpuReplayUsage::ExtraCapture("b".to_owned()),
                "`gpu-replay` takes one capture file; `b` is a second one",
            ),
            (
                &["a", "--png", "x"],
                GpuReplayUsage::UnknownFlag("--png".to_owned()),
                "`gpu-replay` has no flag `--png`",
            ),
            (
                &["a", "--dump"],
                GpuReplayUsage::MissingValue("--dump"),
                "`--dump` needs a value",
            ),
            (
                &["a", "--until"],
                GpuReplayUsage::MissingValue("--until"),
                "`--until` needs a value",
            ),
            (
                &["a", "--dump", "ring", "--dump", "writes"],
                GpuReplayUsage::Repeated("--dump"),
                "`--dump` was given twice",
            ),
            (
                &["a", "--dump", "frames"],
                GpuReplayUsage::BadDump("frames".to_owned()),
                "`--dump frames` is not a dump: write `ring`, `render` or `writes`",
            ),
        ];
        for (words, error, text) in cases {
            assert_eq!(parse(&args(words)), Err(error.clone()), "{words:?}");
            assert_eq!(error.to_string(), text);
        }
        assert_eq!(
            parse(&args(&["a", "--until", "present=1", "--until=present=2"])),
            Err(GpuReplayUsage::Repeated("--until"))
        );
    }
}
