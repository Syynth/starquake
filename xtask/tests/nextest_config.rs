//! Checks `.config/nextest.toml`, which the gate's test step (`cargo nextest run`) reads: a test
//! that hangs must end as a failure, not stall the gate.
//!
//! What the settings mean is nextest's own documentation ("Slow tests and timeouts"):
//! `slow-timeout = { period, terminate-after }` marks a test slow after `period` and ends it
//! after `terminate-after` periods; without `terminate-after`, or without `slow-timeout` at all,
//! a test may run for ever. That nextest does end a hung test with these settings was seen once
//! by hand, with the test this file's limit was added for
//! (`a_piece_of_no_bytes_is_a_fault_not_an_endless_walk` in `crates/core`, run against the loop
//! it fixes): `TIMEOUT [ 120.006s]`.

/// Helpers live in a `cfg(test)` module so clippy's allow-unwrap-in-tests setting covers them.
#[cfg(test)]
mod support {
    use std::path::Path;
    use std::time::Duration;

    /// The longest a hung test may hold up the gate. The slowest tests of the workspace take a
    /// few seconds; the gate as a whole takes about a minute.
    pub const LONGEST_ALLOWED: Duration = Duration::from_mins(5);
    /// The shortest limit that leaves a healthy test room on a loaded machine: the tests that
    /// run guest programs allow themselves 30 seconds.
    pub const SHORTEST_ALLOWED: Duration = Duration::from_mins(1);

    /// The repository's nextest configuration.
    pub fn config() -> toml::Table {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.config/nextest.toml");
        std::fs::read_to_string(path).unwrap().parse().unwrap()
    }

    /// How long a test may run under `profile` before nextest ends it, or `None` if it never
    /// does. `period` is a whole number of seconds (`"30s"`), which is all this repository
    /// writes.
    pub fn kill_after(config: &toml::Table, profile: &str) -> Option<Duration> {
        let slow = config
            .get("profile")?
            .get(profile)?
            .get("slow-timeout")?
            .as_table()?;
        let period = slow.get("period")?.as_str()?.strip_suffix('s')?;
        let periods = slow.get("terminate-after")?.as_integer()?;
        let seconds = period.parse::<u64>().ok()?;
        Some(Duration::from_secs(
            seconds.checked_mul(u64::try_from(periods).ok()?)?,
        ))
    }
}

use support::{LONGEST_ALLOWED, SHORTEST_ALLOWED, config, kill_after};

#[test]
fn a_test_that_hangs_is_ended_within_minutes() {
    // The gate runs the default profile (`xtask/src/main.rs`, `steps`: no `--profile`).
    let limit = kill_after(&config(), "default").expect("a limit after which a test is ended");
    assert!(limit <= LONGEST_ALLOWED, "{limit:?}");
    assert!(limit >= SHORTEST_ALLOWED, "{limit:?}");
}

#[test]
fn a_configuration_without_a_limit_has_none() {
    // The reading above is not satisfied by a file that only warns: each of these lacks what
    // ends a test.
    for text in [
        "",
        "[profile.default]\nretries = 0\n",
        "[profile.default]\nslow-timeout = \"30s\"\n",
        "[profile.default]\nslow-timeout = { period = \"30s\" }\n",
        "[profile.ci]\nslow-timeout = { period = \"30s\", terminate-after = 4 }\n",
        "[profile.default]\nslow-timeout = { period = \"30s\", terminate-after = -1 }\n",
    ] {
        let config: toml::Table = text.parse().unwrap();
        assert_eq!(kill_after(&config, "default"), None, "{text:?}");
    }
    let limited: toml::Table =
        "[profile.default]\nslow-timeout = { period = \"30s\", terminate-after = 4 }\n"
            .parse()
            .unwrap();
    assert_eq!(
        kill_after(&limited, "default"),
        Some(std::time::Duration::from_mins(2))
    );
}

#[test]
fn a_flaky_test_is_not_retried() {
    // `CLAUDE.md`, "Tests": never hide a flaky test behind retries.
    let config = config();
    let retries = config["profile"]["default"]
        .get("retries")
        .and_then(toml::Value::as_integer);
    assert_eq!(retries, Some(0));
}
