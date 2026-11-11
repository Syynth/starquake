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

A **server** is one service family's state plus the sessions opened on it, as on a console, where
a system module is one program serving all its sessions.

- **One request at a time per server, by default.** Requests from all of a server's sessions go
  through one queue (kernel spec, "IPC"), and only one handler of that server runs at any moment.
  A service's own state therefore needs no thread-safety work, and a method takes `&mut self`.
  This is what most real system modules do: a single loop over all their sessions.
- **One request in flight per session.** A session with an unanswered request (including a
  deferred one) accepts no second request; a second sender on it waits as a guest thread.
- **A deferred request doesn't hold the server.** Once a handler returns `Deferred`, the server
  takes the next request from another session, as a real module does.
- **Opting in to parallelism.** A server known to be multi-threaded on a console (the file
  system) can declare a number of workers. Its shared state must then be thread-safe, and the
  declaration is reviewed like an `unsafe` block: with a reason.
- **Different servers run at the same time** on different cores. State shared *between* servers
  (settings, the clock) is passed in at construction as `Arc<...>` and must be thread-safe.
- **Nobody waits on a host lock.** A sender that finds its server busy waits as a guest thread;
  the core that finishes the running handler starts the next queued request.
- **Methods never block:** no sleeping, no blocking I/O, no waiting on another guest thread.
  Work that takes time goes to a worker and completes a deferred reply.

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
| 6 | `fsp-srv`, a first subset: open a file system, open a file, read, get size, close | 1–3 | Requests with buffers and sub-interfaces, with and without a domain; a read that defers |

Slices 1 and 2 need only the IPC crate and a fake kernel, so they can start as soon as this spec
is approved.

### The first three services are a trial of the handler style

`sm:`, `set:sys` and the `fsp-srv` subset are written in the style above: a `commands!` table and
explicit reads and writes. Between them they cover plain payloads, buffers, handles,
sub-interfaces, domains and a deferred reply. Before a fourth service is ported, judge the
result:

- How much of each method is unpacking and packing, compared with behaviour?
- How many layout mistakes (wrong offset, missing padding) did review catch?
- Could someone check a method against its reference at a glance?

If the methods are long and error-prone, switch to a typed style, where a method's signature
declares its inputs and outputs and the framework unpacks the message (the direction Eden has
been migrating in), while only three services need rewriting. Struct-mapped payloads can be
added alongside explicit reads at any time and don't need this trial.

## Decisions asked of the owner

1. **A declarative `commands!` macro and hand-written unpacking, as a trial.** The first three
   services use it; the style is confirmed or replaced by a typed style before a fourth is
   ported. (Revised after owner review: the first draft fixed the style up front.)
2. **Explicit readers and writers for payloads** rather than a zero-copy struct crate.
   Recommended while layouts are still being verified; revisit if it becomes repetitive.
3. **Auto-stub on by default for title bring-up, strict in tests.**
4. **One request at a time per server by default**, as most real system modules behave, with
   opt-in workers for servers that are multi-threaded on a console. (Revised after owner review:
   the first draft serialised per session only, which put thread-safety on every service.)
5. **Milestone 1's homebrew overrides libnx start-up** so the first end-to-end run needs two
   services, not seven.
