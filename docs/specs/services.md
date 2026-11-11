# Service framework

Status: **proposal, awaiting owner review** (issue #10). Builds on the execution model and the
kernel spec, both also proposals.

`crates/services` holds the HLE system services: Rust code that answers the IPC requests a game
would otherwise send to system processes. Eden has 74 service families. This spec defines the
framework they are written against, so that porting one service is a self-contained task with
its own tests.

## How a request flows

1. A guest thread calls `SendSyncRequest` on a session handle.
2. The kernel copies the message out of the thread's TLS block, parses it (`starquake-ipc`),
   resolves handles, and calls the session's handler on the same core thread, without holding
   the kernel lock.
3. The framework handles protocol-level requests itself (control commands, domain bookkeeping),
   finds the command, and calls the service's method.
4. The method writes a reply and returns `Done`, or keeps a reply token and returns `Deferred`
   (execution model: handlers never block).
5. The kernel writes the reply into the thread's TLS block, inserts any returned handles, and
   resumes or parks the thread.

## Writing a service

A service is a struct per session, a command table, and one method per command:

```rust
pub struct SetSys { settings: Arc<Settings> }

commands! {
    impl SetSys {
        3 => get_firmware_version,
        4 => get_firmware_version2,
    }
}

impl SetSys {
    fn get_firmware_version(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply<'_>) -> Outcome {
        let mut out = req.out_buffer(cx, 0)?;
        out.write(&self.settings.firmware_version().to_bytes())?;
        reply.ok()
    }
}
```

- **`commands!`** is a small declarative macro (`macro_rules!`, no procedural macros). It
  generates the dispatch `match` and a table of command ids and names used for tracing and
  auto-stubbing. TIPC commands get their own block, since ids overlap.
- **`Request`** exposes the command id, the raw payload through an explicit little-endian
  reader (`read_u32`, `read_u64`, `align(8)`, `read_bytes`), buffers by index in Horizon's
  conventions ("in" is an A or X descriptor, "out" is a B or C descriptor), the caller's process
  id when sent, and handles already resolved to kernel objects.
- **`Reply`** takes a result code, a raw payload through a matching writer, handles to copy or
  move, and sub-interfaces (`reply.push_session(Box<dyn Session>)`). Whether a sub-interface
  becomes a new session handle or a domain object id is the framework's business, not the
  service's.
- **`Context`** gives the service what it may touch: guest memory (copy in and out only), a
  narrow kernel interface (create and signal events, shared memory), the clock, and `defer()`,
  which returns a reply token.
- Payload layouts are read and written field by field with explicit alignment, citing the
  reference. No transmuting structs: guest data is untrusted, and `unsafe` is denied here.

## What the framework owns

- **Control requests:** convert to domain, copy from domain, clone session, query pointer
  buffer size.
- **Domains:** a per-session table of objects, dispatch by object id, close-object requests,
  and input and output object ids.
- **Unknown commands.** Two policies, selectable per run and per service:
  - *Strict:* log and return an error. The default in tests.
  - *Auto-stub:* log once per command and return success with a zeroed payload. The default when
    bringing up a title, as in Eden-NX2. The payload is zeroed, never an echo of the request: an
    echoed reply caused a crash in one title (row 96).
- **Tracing.** Every request produces one trace line: guest thread, service and session, command
  id and name, payload sizes, result, and whether it was stubbed or deferred. This is the golden
  trace format from the test strategy, so a trace from a real title is also the list of commands
  still to implement.
- **Deferred replies.** `cx.defer()` hands out a token. Completing it from any thread (GPU,
  timer, I/O worker) writes the reply and wakes the caller. Dropping a token without completing
  it is a bug: it replies with an error and logs.

## Concurrency

- Requests on one session are serialised by the kernel's request queue (kernel spec, "IPC"):
  a second sender waits as a guest thread until the first request has been answered, as on a
  console. A service method therefore takes `&mut self`, and no core thread ever waits on a host
  lock for a session.
- Different sessions run at the same time on different cores. State shared between sessions
  (settings, the file system, input state) is passed in at construction as `Arc<...>` and must be
  thread-safe.
- Methods never block: no sleeping, no blocking I/O, no waiting on another guest thread. Work
  that takes time goes to a worker and completes a deferred reply.

## The service manager

`sm:` is a named port. It maps service names to servers. Today a server is always a factory
that creates a Rust session object; the registry is written so that a guest process registering
a service (deferred, see the kernel spec) is one more kind of entry, and a title can't tell the
difference:

- `Initialize`, `GetService` (returns a moved session handle), and the TIPC forms of both.
- A name with no factory is logged. Under auto-stub it gets a generic session that stubs every
  command, so a title's service list can be discovered in one run; under strict it is an error.
- Access control from the NPDM (which names the title may open) is logged at first, like
  kernel capabilities.

## Real system modules and guest-hosted services

Deferred, not excluded (decision log, 2026-11-11). What this framework does so they stay additive:

- A service is reached only through its session's server end. Replacing a Rust service with a
  real module, or the other way round, is a change in the service manager's registry, not in any
  caller.
- Services talk to each other through the same registry and sessions a title uses, never by
  calling each other's Rust types directly, so a Rust service can depend on one that is later
  provided by a real module.
- Auto-stubbing and tracing sit at the session boundary, so they work the same for either kind
  of server.

## Milestone 1 services

Milestone 1's homebrew program overrides libnx's default start-up so it needs only:

- `sm:`
- `set:sys` (firmware version)

A default libnx start-up also opens `apm`, `appletOE`, `hid`, `time` and `fsp-srv`. Those come
next, as stubs backed by traces from the hello-world run, and lead into the milestone 2 list in
the scoping doc ("Services (HLE)").

## Layout and porting recipe

```text
crates/services/src/
  framework/        commands!, Request, Reply, Context, domains, control, auto-stub, trace
  sm.rs
  set.rs            one file (or folder) per service family
  ...
```

One service family is one issue:

1. Source: the Eden folder (`eden/src/core/hle/service/<family>`), plus Eden-NX2 and PR #4588
   where Switch 2 changed it.
2. Port the command table and the behaviour titles depend on; leave the rest to auto-stub, and
   say which commands were left out.
3. Tests: for each ported command, a request built with the IPC request builder and the expected
   reply bytes and state changes. Layouts are cited; guesses are marked.

## Port plan

| # | Slice | Depends on | Test |
|---|---|---|---|
| 1 | Framework core: `Session`, `commands!`, `Request`/`Reply` readers and writers, dispatch, strict and auto-stub | IPC crate, the kernel's `SessionHandler` interface | A toy service driven through raw messages; unknown-command behaviour under both policies |
| 2 | Control requests and domains | 1 | Convert to domain, object ids in and out, close object, clone |
| 3 | Deferred replies and the trace sink | 1, kernel sessions | Fake kernel; a completed token wakes the caller; a dropped token errors |
| 4 | `sm:` with the generic stub session | 1–2 | `GetService` round trip; unknown name under both policies; TIPC |
| 5 | `set:sys` (firmware version) | 1 | Reply bytes against the documented layout |

Slices 1 and 2 need only the IPC crate and a fake kernel, so they can start as soon as this spec
is approved.

## Decisions asked of the owner

1. **A declarative `commands!` macro, no procedural macros.** Recommended: cheap to compile and
   easy for agents to read and extend.
2. **Explicit readers and writers for payloads** rather than a zero-copy struct crate.
   Recommended while layouts are still being verified; revisit if it becomes repetitive.
3. **Auto-stub on by default for title bring-up, strict in tests.**
4. **One request at a time per session, queued by the kernel; shared state is `Arc` and
   thread-safe.**
5. **Milestone 1's homebrew overrides libnx start-up** so the first end-to-end run needs two
   services, not seven.
