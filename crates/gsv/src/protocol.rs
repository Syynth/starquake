//! What is known of the graphics server's root protocol, and how it is known: the command
//! table with each fact's mark and source (`docs/specs/graphics-server.md`, "How the protocol is
//! known" and "The root"; port slice 1).
//!
//! Ported from (GPL-3.0-or-later), cited below by these short forms:
//!
//! - **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp`
//!   (`IGraphicsSupervisor`) and `graphics_supervisor_protocol.h` @ e5fb3b97. Its comments
//!   cite addresses in the system module, so some of its readings are the server's own.
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`INx2GraphicsSystemSession`), and its blocker log `eden-nx2/NX2_NOTES.md` @ 50fa86bb,
//!   whose rows "row N" names.
//!
//! # Marks
//!
//! Each fact carries one of the spec's four [`Mark`]s. The marks of the table rows are the
//! spec's ("The root", the table of commands by group), row for row.
//!
//! # Where the references disagree
//!
//! The spec's house rule (decision 8) picks one reading; each is open question 1 of the spec's
//! "What to verify before building" until a trace settles it:
//!
//! - **1001 to 1003:** NX2 reads them as a second video decoder, PR as the video encoder
//!   (`InitializeVideoEncoder`, `OpenVideoEncoderContext`, `FinalizeVideoEncoder`). They are
//!   named as the encoder (PR cites the module); the reply shapes are the same either way
//!   (1002 returns a session, which the codec start-up needs: row 105, observed).
//! - **Copied or moved session handles:** NX2 copies every session it returns
//!   (`ReturnRawSession`: `PushCopyObjects`), PR moves them (`AddMoveObject`). Copy is used,
//!   since titles ran on it ([`HANDLE_KIND`]). The same goes for 204's two events (NX2
//!   `AttachSyncptMemory`: `PushCopyObjects`; PR `BindSyncpointMemory`: `AddMoveObject`).
//! - **1121:** PR's comment reads the system module as replying one copied event; PR's code
//!   returns a session (`GetNativeEvent1121`: `CreateGraphicsSession(.., INativeSession1121)`).
//!   The table follows the code, which is what PR ran; NX2 has no 1121.
//!
//! The reply shapes are PR's (`IPC::ResponseBuilder` sizes and what each handler pushes after
//! its result word); where NX2 has the command, its reply has the same data words.
//!
//! Commands 904 and 905 fall inside the spec's "901–906" but are in neither reference's table,
//! so they are not here; NX2's `Unknown5` (5) is not in the spec's table either. All three are
//! answered by the server's unknown-command policy.

/// How a protocol fact is known (`docs/specs/graphics-server.md`, "How the protocol is known").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// A reference answered this way and a title got further: a row of NX2's log says so.
    Observed,
    /// A reference read it from the SDK or the system module, with no row showing a title
    /// depending on it.
    Read,
    /// A reference marks it guessed, or neither says where it came from.
    Guess,
    /// The references disagree; the spec's house rule (decision 8) picked one reading.
    Conflict,
}

impl Mark {
    /// The mark as the spec writes it.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::Read => "read",
            Self::Guess => "guess",
            Self::Conflict => "conflict",
        }
    }
}

/// A handle a root reply carries before its data, always copied ([`HANDLE_KIND`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handed {
    /// A session (the root's stubs hand out a placeholder until the family's slice lands).
    Session,
    /// An event.
    Event,
}

/// Which list of a reply the root puts the handles it returns in: copy, by the house rule (see
/// "Where the references disagree" in the [module docs](self)). Mark: [`Mark::Conflict`].
pub const HANDLE_KIND: Mark = Mark::Conflict;

/// What a root reply holds after its result word: its handles, in order, then `words` data
/// words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplyShape {
    /// The handles, in the order they are listed.
    pub handles: &'static [Handed],
    /// The data words after the result word.
    pub words: u32,
}

/// One root command: its id, name, mark, where it is known from, and its reply's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootCommand {
    /// The TIPC command id.
    pub id: u32,
    /// PR's name for it.
    pub name: &'static str,
    /// How it is known.
    pub mark: Mark,
    /// The reference rows and functions it is known from.
    pub source: &'static str,
    /// What its reply holds.
    pub reply: ReplyShape,
}

/// The root's TIPC command ids, each with its source.
pub mod command {
    /// PR `{4, &IGraphicsSupervisor::Initialize, "Initialize"}`; NX2 `{4, .., "Boot (Switch 2,
    /// guessed)"}`.
    pub const INITIALIZE: u32 = 4;
    /// PR `{11, &IGraphicsSupervisor::BindProcess, "BindProcess"}`.
    pub const BIND_PROCESS: u32 = 11;
    /// PR `{101, &IGraphicsSupervisor::CompleteBootstrap, "CompleteBootstrap"}`; NX2
    /// `"Unknown101"`.
    pub const COMPLETE_BOOTSTRAP: u32 = 101;
    /// PR `{126, &IGraphicsSupervisor::QueryMemoryMode, "QueryMemoryMode"}`.
    pub const QUERY_MEMORY_MODE: u32 = 126;
    /// PR `{115, &IGraphicsSupervisor::ConfigureNativeMemory, "ConfigureNativeMemory"}`.
    pub const CONFIGURE_NATIVE_MEMORY: u32 = 115;
    /// PR and NX2 `CreateNativeMemory`.
    pub const CREATE_NATIVE_MEMORY: u32 = 111;
    /// PR and NX2 `DestroyNativeMemory`.
    pub const DESTROY_NATIVE_MEMORY: u32 = 112;
    /// PR and NX2 `BindNativeMemory`.
    pub const BIND_NATIVE_MEMORY: u32 = 114;
    /// PR and NX2 `BindNativeGpuMemory`.
    pub const BIND_NATIVE_GPU_MEMORY: u32 = 423;
    /// PR and NX2 `CreateNativeImage`.
    pub const CREATE_NATIVE_IMAGE: u32 = 429;
    /// PR and NX2 `DestroyNativeImage`.
    pub const DESTROY_NATIVE_IMAGE: u32 = 430;
    /// PR and NX2 `CalculateNativeImageSize`.
    pub const CALCULATE_NATIVE_IMAGE_SIZE: u32 = 435;
    /// PR `CalculateImageRequirements`; NX2 `CalculateRequirements`.
    pub const CALCULATE_IMAGE_REQUIREMENTS: u32 = 1510;
    /// PR and NX2 `DestroyNativeWindowAliasImagePool`.
    pub const DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL: u32 = 314;
    /// PR and NX2 `CreateNativeWindowAliasImagePool`.
    pub const CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL: u32 = 315;
    /// PR `MapNativeGpuImage`.
    pub const MAP_NATIVE_GPU_IMAGE: u32 = 426;
    /// PR `UnmapNativeGpuImage`.
    pub const UNMAP_NATIVE_GPU_IMAGE: u32 = 427;
    /// PR `QueryNativeGpuImage`.
    pub const QUERY_NATIVE_GPU_IMAGE: u32 = 428;
    /// PR `InitializeSyncpointContext`.
    pub const INITIALIZE_SYNCPOINT_CONTEXT: u32 = 201;
    /// PR `GetSyncpointSession`; NX2 `OpenSyncptSession (guessed)`.
    pub const GET_SYNCPOINT_SESSION: u32 = 202;
    /// PR `BindSyncpointMemory`; NX2 `AttachSyncptMemory (guessed)`.
    pub const BIND_SYNCPOINT_MEMORY: u32 = 204;
    /// PR `InitializeDisplayContext`.
    pub const INITIALIZE_DISPLAY_CONTEXT: u32 = 301;
    /// PR `FinalizeDisplayContext`.
    pub const FINALIZE_DISPLAY_CONTEXT: u32 = 303;
    /// PR `OpenDisplaySession`; NX2 `OpenViSession (guessed)`.
    pub const OPEN_DISPLAY_SESSION: u32 = 305;
    /// PR `InitializeGpuContext`.
    pub const INITIALIZE_GPU_CONTEXT: u32 = 401;
    /// PR `ActivateGpuContext`.
    pub const ACTIVATE_GPU_CONTEXT: u32 = 402;
    /// PR `RegisterGpuAddressSpace`.
    pub const REGISTER_GPU_ADDRESS_SPACE: u32 = 434;
    /// PR `OpenGpuSession`; NX2 `OpenGpuSession (guessed)`.
    pub const OPEN_GPU_SESSION: u32 = 497;
    /// PR `GetNvidiaDriver`; NX2 `OpenNvsrvSession (guessed)`.
    pub const GET_NVIDIA_DRIVER: u32 = 1402;
    /// PR `InitializeVideoDecoder`; NX2 `VideoInitialize (guessed)`.
    pub const INITIALIZE_VIDEO_DECODER: u32 = 901;
    /// PR `OpenVideoDecoderContext902`; NX2 `OpenVideoDecoderSession (guessed)`.
    pub const OPEN_VIDEO_DECODER_CONTEXT_902: u32 = 902;
    /// PR `FinalizeVideoDecoder`.
    pub const FINALIZE_VIDEO_DECODER: u32 = 903;
    /// PR `OpenVideoDecoderContext`; NX2 `OpenVideoDecoderSessionWithVersion`.
    pub const OPEN_VIDEO_DECODER_CONTEXT: u32 = 906;
    /// PR `CreateVideoImagePool`; NX2 `CreateDecoderImage`.
    pub const CREATE_VIDEO_IMAGE_POOL: u32 = 911;
    /// PR `DestroyVideoImagePool`.
    pub const DESTROY_VIDEO_IMAGE_POOL: u32 = 912;
    /// PR `QueryVideoImage`; NX2 `GetCpuUserImageDescriptor`.
    pub const QUERY_VIDEO_IMAGE: u32 = 961;
    /// PR `CalculateVideoImageRequirements`; NX2 `CalculateDecoderImagePoolBuffer`.
    pub const CALCULATE_VIDEO_IMAGE_REQUIREMENTS: u32 = 991;
    /// PR `InitializeVideoEncoder`; NX2 `VideoInitialize2 (guessed)`. Conflict (module docs).
    pub const INITIALIZE_VIDEO_ENCODER: u32 = 1001;
    /// PR `OpenVideoEncoderContext`; NX2 `OpenVideoDecoderSession2 (guessed)`. Conflict.
    pub const OPEN_VIDEO_ENCODER_CONTEXT: u32 = 1002;
    /// PR `FinalizeVideoEncoder`. Conflict.
    pub const FINALIZE_VIDEO_ENCODER: u32 = 1003;
    /// PR `InitializeFrameContext`.
    pub const INITIALIZE_FRAME_CONTEXT: u32 = 1101;
    /// PR `ActivateFrameContext`.
    pub const ACTIVATE_FRAME_CONTEXT: u32 = 1102;
    /// PR `SetGraphicsClass1111`.
    pub const SET_GRAPHICS_CLASS_1111: u32 = 1111;
    /// PR `GetNativeEvent1121`.
    pub const GET_NATIVE_EVENT_1121: u32 = 1121;
    /// PR `SequenceCommand1201`.
    pub const SEQUENCE_COMMAND_1201: u32 = 1201;
    /// PR `SequenceCommand1202`.
    pub const SEQUENCE_COMMAND_1202: u32 = 1202;
    /// PR `SequenceCommand1203`.
    pub const SEQUENCE_COMMAND_1203: u32 = 1203;
    /// PR `CreateChatImagePool`.
    pub const CREATE_CHAT_IMAGE_POOL: u32 = 1211;
    /// PR `CalculateRequiredBufferMemory`.
    pub const CALCULATE_REQUIRED_BUFFER_MEMORY: u32 = 1291;
    /// PR `InitializeImageContext1501`.
    pub const INITIALIZE_IMAGE_CONTEXT_1501: u32 = 1501;
    /// PR `InitializeImageContext1502`.
    pub const INITIALIZE_IMAGE_CONTEXT_1502: u32 = 1502;
    /// PR `CreateCameraContext`.
    pub const CREATE_CAMERA_CONTEXT: u32 = 601;
    /// PR `OpenCameraSession`.
    pub const OPEN_CAMERA_SESSION: u32 = 602;
}

/// The bootstrap's request and replies (PR `graphics_supervisor_protocol.h`, `DecodeBootstrap`,
/// `IsLocalBootstrap`, `NativeGpuProfileForBootstrap`; PR `QueryMemoryMode`,
/// `ConfigureNativeMemory`).
pub mod bootstrap {
    /// How many `u64` parameters `Initialize` carries after its flags: `DecodeBootstrap` reads
    /// `std::array<u64, 4> parameters` from words 8 to 15.
    pub const PARAMETERS: usize = 4;

    /// The flags of a local bootstrap: `IsLocalBootstrap` accepts exactly these three, with
    /// every parameter zero. Any other request asks for remote memory.
    pub const LOCAL_FLAGS: [u64; 3] = [
        0x8000_0000_0000_0005,
        0x8000_0000_0000_0006,
        0x8000_0000_0000_0007,
    ];

    /// The GPU profile `NativeGpuProfileForBootstrap` gives flags 5 (`case
    /// 0x8000000000000005ULL: return 0x807`).
    pub const GPU_PROFILE_FLAGS_5: u32 = 0x807;
    /// The GPU profile it gives flags 6 and 7 (`return 0x808`).
    pub const GPU_PROFILE_FLAGS_6_AND_7: u32 = 0x808;

    /// The GPU profile a bootstrap with `flags` asks for: `NativeGpuProfileForBootstrap`, whose
    /// `default: return 0` is `None` here. PR also overrides one title's profile by its program
    /// id; that is a title workaround, not ported.
    #[must_use]
    pub const fn gpu_profile(flags: u64) -> Option<u32> {
        match flags {
            0x8000_0000_0000_0005 => Some(GPU_PROFILE_FLAGS_5),
            0x8000_0000_0000_0006 | 0x8000_0000_0000_0007 => Some(GPU_PROFILE_FLAGS_6_AND_7),
            _ => None,
        }
    }

    /// Whether a bootstrap with `flags` and `parameters` is local: `IsLocalBootstrap`.
    #[must_use]
    pub fn is_local(flags: u64, parameters: &[u64; PARAMETERS]) -> bool {
        LOCAL_FLAGS.contains(&flags) && parameters.iter().all(|&value| value == 0)
    }

    /// What `QueryMemoryMode` (126) answers after its result: `rb.Push<u32>(0); // The local
    /// backend has no remote-memory mode.`
    pub const NO_REMOTE_MEMORY_MODE: u32 = 0;

    /// What `ConfigureNativeMemory` (115) answers after its result: `rb.Push<u64>(0)`, "Clear
    /// the optional Gpuk-memory mode".
    pub const NO_NATIVE_MEMORY_MODE: u64 = 0;
}

use command::{
    ACTIVATE_FRAME_CONTEXT, ACTIVATE_GPU_CONTEXT, BIND_NATIVE_GPU_MEMORY, BIND_NATIVE_MEMORY,
    BIND_PROCESS, BIND_SYNCPOINT_MEMORY, CALCULATE_IMAGE_REQUIREMENTS, CALCULATE_NATIVE_IMAGE_SIZE,
    CALCULATE_REQUIRED_BUFFER_MEMORY, CALCULATE_VIDEO_IMAGE_REQUIREMENTS, COMPLETE_BOOTSTRAP,
    CONFIGURE_NATIVE_MEMORY, CREATE_CAMERA_CONTEXT, CREATE_CHAT_IMAGE_POOL, CREATE_NATIVE_IMAGE,
    CREATE_NATIVE_MEMORY, CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, CREATE_VIDEO_IMAGE_POOL,
    DESTROY_NATIVE_IMAGE, DESTROY_NATIVE_MEMORY, DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL,
    DESTROY_VIDEO_IMAGE_POOL, FINALIZE_DISPLAY_CONTEXT, FINALIZE_VIDEO_DECODER,
    FINALIZE_VIDEO_ENCODER, GET_NATIVE_EVENT_1121, GET_NVIDIA_DRIVER, GET_SYNCPOINT_SESSION,
    INITIALIZE, INITIALIZE_DISPLAY_CONTEXT, INITIALIZE_FRAME_CONTEXT, INITIALIZE_GPU_CONTEXT,
    INITIALIZE_IMAGE_CONTEXT_1501, INITIALIZE_IMAGE_CONTEXT_1502, INITIALIZE_SYNCPOINT_CONTEXT,
    INITIALIZE_VIDEO_DECODER, INITIALIZE_VIDEO_ENCODER, MAP_NATIVE_GPU_IMAGE, OPEN_CAMERA_SESSION,
    OPEN_DISPLAY_SESSION, OPEN_GPU_SESSION, OPEN_VIDEO_DECODER_CONTEXT,
    OPEN_VIDEO_DECODER_CONTEXT_902, OPEN_VIDEO_ENCODER_CONTEXT, QUERY_MEMORY_MODE,
    QUERY_NATIVE_GPU_IMAGE, QUERY_VIDEO_IMAGE, REGISTER_GPU_ADDRESS_SPACE, SEQUENCE_COMMAND_1201,
    SEQUENCE_COMMAND_1202, SEQUENCE_COMMAND_1203, SET_GRAPHICS_CLASS_1111, UNMAP_NATIVE_GPU_IMAGE,
};

/// No handles.
const NONE: &[Handed] = &[];
/// One session.
const SESSION: &[Handed] = &[Handed::Session];

/// A row with PR's name, a mark, a source and a shape.
const fn row(
    id: u32,
    name: &'static str,
    mark: Mark,
    source: &'static str,
    handles: &'static [Handed],
    words: u32,
) -> RootCommand {
    RootCommand {
        id,
        name,
        mark,
        source,
        reply: ReplyShape { handles, words },
    }
}

use Mark::{Conflict, Observed, Read};

/// Every root command, grouped as the spec's table groups them. Data word counts are what PR's
/// handler pushes after the result (a `u64` is two words).
#[rustfmt::skip]
pub const ROOT_COMMANDS: &[RootCommand] = &[
    // Bootstrap.
    row(INITIALIZE, "Initialize", Read, "PR Initialize, DecodeBootstrap, IsLocalBootstrap; NX2 stub", NONE, 0),
    row(BIND_PROCESS, "BindProcess", Read, "PR BindProcess; not in NX2's table", NONE, 0),
    row(COMPLETE_BOOTSTRAP, "CompleteBootstrap", Read, "PR CompleteBootstrap; NX2 stub", NONE, 0),
    // Memory modes: PR pushes a u32 0, and a u64 0.
    row(QUERY_MEMORY_MODE, "QueryMemoryMode", Read, "PR QueryMemoryMode", NONE, 1),
    row(CONFIGURE_NATIVE_MEMORY, "ConfigureNativeMemory", Read, "PR ConfigureNativeMemory", NONE, 2),
    // Syncpoints.
    row(INITIALIZE_SYNCPOINT_CONTEXT, "InitializeSyncpointContext", Read, "PR InitializeSyncpointContext", NONE, 0),
    row(GET_SYNCPOINT_SESSION, "GetSyncpointSession", Observed, "row 19; PR GetSyncpointSession; NX2 OpenSyncptSession", SESSION, 0),
    row(BIND_SYNCPOINT_MEMORY, "BindSyncpointMemory", Observed, "rows 19, 28; PR BindSyncpointMemory; NX2 AttachSyncptMemory", &[Handed::Event, Handed::Event], 0),
    // Display.
    row(INITIALIZE_DISPLAY_CONTEXT, "InitializeDisplayContext", Read, "PR InitializeDisplayContext", NONE, 0),
    row(FINALIZE_DISPLAY_CONTEXT, "FinalizeDisplayContext", Read, "PR FinalizeDisplayContext", NONE, 0),
    row(OPEN_DISPLAY_SESSION, "OpenDisplaySession", Observed, "row 21; PR OpenDisplaySession, IsSupportedDisplaySession; NX2 OpenViSession", SESSION, 0),
    // GPU.
    row(INITIALIZE_GPU_CONTEXT, "InitializeGpuContext", Read, "PR InitializeGpuContext", NONE, 0),
    row(ACTIVATE_GPU_CONTEXT, "ActivateGpuContext", Read, "PR InitializeGpuContext (402)", NONE, 0),
    row(REGISTER_GPU_ADDRESS_SPACE, "RegisterGpuAddressSpace", Read, "PR RegisterGpuAddressSpace", NONE, 0),
    row(OPEN_GPU_SESSION, "OpenGpuSession", Observed, "row 20; PR OpenGpuSession; NX2 OpenGpuSession", SESSION, 0),
    // nvsrv.
    row(GET_NVIDIA_DRIVER, "GetNvidiaDriver", Observed, "row 23; PR GetNvidiaDriver; NX2 OpenNvsrvSession", SESSION, 0),
    // Native memory: an id (u32) from 111.
    row(CREATE_NATIVE_MEMORY, "CreateNativeMemory", Observed, "row 47; PR and NX2 CreateNativeMemory", NONE, 1),
    row(DESTROY_NATIVE_MEMORY, "DestroyNativeMemory", Observed, "row 47; PR and NX2 DestroyNativeMemory", NONE, 0),
    row(BIND_NATIVE_MEMORY, "BindNativeMemory", Observed, "row 47; PR and NX2 BindNativeMemory", NONE, 0),
    row(BIND_NATIVE_GPU_MEMORY, "BindNativeGpuMemory", Observed, "rows 47, 48; PR and NX2 BindNativeGpuMemory", NONE, 0),
    // Native images: an id (u32) from 429, a u64 size from 435, a u64 size and a u64 alignment
    // from 1510.
    row(CREATE_NATIVE_IMAGE, "CreateNativeImage", Observed, "rows 33, 47; PR and NX2 CreateNativeImage", NONE, 1),
    row(DESTROY_NATIVE_IMAGE, "DestroyNativeImage", Observed, "rows 33, 47; PR and NX2 DestroyNativeImage", NONE, 0),
    row(CALCULATE_NATIVE_IMAGE_SIZE, "CalculateNativeImageSize", Observed, "rows 33, 47; PR CalculateImageRequirements (435); NX2 CalculateNativeImageSize", NONE, 2),
    row(CALCULATE_IMAGE_REQUIREMENTS, "CalculateImageRequirements", Observed, "rows 33, 47; PR CalculateImageRequirements; NX2 CalculateRequirements", NONE, 4),
    row(DESTROY_NATIVE_WINDOW_ALIAS_IMAGE_POOL, "DestroyNativeWindowAliasImagePool", Observed, "rows 33, 47; PR and NX2 ConfigureNativeWindowImagePool (314)", NONE, 0),
    row(CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL, "CreateNativeWindowAliasImagePool", Observed, "rows 33, 47; PR and NX2 ConfigureNativeWindowImagePool (315)", NONE, 0),
    // Native images for the GPU: 426 and 428 push a u64 address, a u64 last address, sixteen
    // u64 descriptor words and a u32 zero.
    row(MAP_NATIVE_GPU_IMAGE, "MapNativeGpuImage", Read, "PR MapVideoGpuImage (426)", NONE, 2 + 2 + 16 * 2 + 1),
    row(UNMAP_NATIVE_GPU_IMAGE, "UnmapNativeGpuImage", Read, "PR UnmapVideoGpuImage (427)", NONE, 0),
    row(QUERY_NATIVE_GPU_IMAGE, "QueryNativeGpuImage", Read, "PR QueryVideoGpuImage (428)", NONE, 2 + 2 + 16 * 2 + 1),
    // Video: a first image id (u32) from 911, sixteen u64 descriptor words from 961, a u64 size
    // and a u64 alignment from 991.
    row(INITIALIZE_VIDEO_DECODER, "InitializeVideoDecoder", Observed, "rows 96, 105, 146, 148; PR InitializeVideoService (901); NX2 stub", NONE, 0),
    row(OPEN_VIDEO_DECODER_CONTEXT_902, "OpenVideoDecoderContext902", Observed, "row 105; PR OpenVideoDecoderContext (902); NX2 OpenVideoSession", SESSION, 0),
    row(FINALIZE_VIDEO_DECODER, "FinalizeVideoDecoder", Observed, "rows 96, 105, 146, 148; PR FinalizeVideoService (903)", NONE, 0),
    row(OPEN_VIDEO_DECODER_CONTEXT, "OpenVideoDecoderContext", Observed, "row 146; PR OpenVideoDecoderContext, DecodeVideoContext; NX2 OpenVideoSession", SESSION, 0),
    row(CREATE_VIDEO_IMAGE_POOL, "CreateVideoImagePool", Observed, "row 148; PR CreateVideoImagePool; NX2 CreateDecoderImage", NONE, 1),
    row(DESTROY_VIDEO_IMAGE_POOL, "DestroyVideoImagePool", Observed, "rows 96, 105, 146, 148; PR DestroyVideoImagePool", NONE, 0),
    row(QUERY_VIDEO_IMAGE, "QueryVideoImage", Observed, "row 148; PR QueryVideoImage; NX2 GetCpuImageDescriptor", NONE, 16 * 2),
    row(CALCULATE_VIDEO_IMAGE_REQUIREMENTS, "CalculateVideoImageRequirements", Observed, "row 96; PR CalculateVideoImageRequirements; NX2 CalculateDecoderImagePoolBuffer", NONE, 4),
    // Encoder or second decoder: conflict (module docs).
    row(INITIALIZE_VIDEO_ENCODER, "InitializeVideoEncoder", Conflict, "PR InitializeVideoService (1001); NX2 stub VideoInitialize2; verification 1", NONE, 0),
    row(OPEN_VIDEO_ENCODER_CONTEXT, "OpenVideoEncoderContext", Conflict, "row 105; PR OpenVideoEncoderContext; NX2 OpenVideoSession (1002); verification 1", SESSION, 0),
    row(FINALIZE_VIDEO_ENCODER, "FinalizeVideoEncoder", Conflict, "PR FinalizeVideoService (1003); not in NX2's table; verification 1", NONE, 0),
    // Frame and image contexts, event supervisor, chat image pools, camera: PR only. A pool id
    // (u32) from 1211; a u64 size and a u64 alignment from 1291.
    row(INITIALIZE_FRAME_CONTEXT, "InitializeFrameContext", Read, "PR InitializeFrameContext", NONE, 0),
    row(ACTIVATE_FRAME_CONTEXT, "ActivateFrameContext", Read, "PR InitializeFrameContext (1102)", NONE, 0),
    row(SET_GRAPHICS_CLASS_1111, "SetGraphicsClass1111", Read, "PR SetGraphicsClass1111, IsSupportedCommand1111", NONE, 0),
    row(GET_NATIVE_EVENT_1121, "GetNativeEvent1121", Read, "PR GetNativeEvent1121 (code, not comment: module docs)", SESSION, 0),
    row(SEQUENCE_COMMAND_1201, "SequenceCommand1201", Read, "PR SequenceCommand, IsSupportedSequenceCommand", NONE, 0),
    row(SEQUENCE_COMMAND_1202, "SequenceCommand1202", Read, "PR SequenceCommand (1202 returns an IEventSupervisor)", SESSION, 0),
    row(SEQUENCE_COMMAND_1203, "SequenceCommand1203", Read, "PR SequenceCommand", NONE, 0),
    row(CREATE_CHAT_IMAGE_POOL, "CreateChatImagePool", Read, "PR CreateChatImagePool", NONE, 1),
    row(CALCULATE_REQUIRED_BUFFER_MEMORY, "CalculateRequiredBufferMemory", Read, "PR CalculateRequiredBufferMemory", NONE, 4),
    row(INITIALIZE_IMAGE_CONTEXT_1501, "InitializeImageContext1501", Read, "PR InitializeImageContext", NONE, 0),
    row(INITIALIZE_IMAGE_CONTEXT_1502, "InitializeImageContext1502", Read, "PR InitializeImageContext", NONE, 0),
    row(CREATE_CAMERA_CONTEXT, "CreateCameraContext", Read, "PR CreateCameraContext (601)", NONE, 0),
    row(OPEN_CAMERA_SESSION, "OpenCameraSession", Read, "PR CreateCameraContext (602)", SESSION, 0),
];

/// The table's row for a root command, if it has one.
#[must_use]
pub fn root_command(id: u32) -> Option<&'static RootCommand> {
    ROOT_COMMANDS.iter().find(|command| command.id == id)
}
