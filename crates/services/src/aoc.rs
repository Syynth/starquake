//! `aoc:u`, the add-on content manager, on a console with no add-on content installed.
//!
//! Ported from `eden/src/core/hle/service/aoc/addon_content_manager.cpp` and `.h`
//! (`IAddOnContentManager`, `LoopProcess`) and `purchase_event_manager.cpp` and `.h`
//! (`IPurchaseEventManager`) @ cf654052 (GPL-3.0-or-later). Eden-NX2 @ 50fa86bb and Eden PR
//! #4588 @ e5fb3b97 have the same four files, and the Switch 2 scoping document
//! (`docs/switch2-eden-diff-scoping.md`) has no `aoc` row, so Switch 2 changes nothing here.
//! Payload layouts are Eden's CMIF serialisation of each method's arguments
//! (`eden/src/core/hle/service/cmif_serialization.h`, `ReadInArgument`: in data in the order
//! the method lists it, each at its own alignment; a `ClientProcessId` takes a `u64` there, and
//! its value is the process id the request sent, `ctx.GetPID()`, not the payload's word).
//!
//! # Commands
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `aoc:u` ([`IAddOnContentManager`]) | 2 | `CountAddOnContent` | client process id (not read) | `u32` count: 0 |
//! | | 3 | `ListAddOnContent` | `u32` offset, `u32` count, client process id (not read); out buffer 0 (map alias, a B descriptor) | `u32` count: 0, and nothing in the buffer; [`APPLICATION_RECORD_NOT_FOUND`] when the offset is not 0 |
//! | | 8 | `GetAddOnContentListChangedEvent` | nothing | a copied handle: the list-changed event |
//! | | 10 | `GetAddOnContentListChangedEventWithProcessId` | client process id (not read) | the same event |
//! | | 100 | `CreateEcPurchasedEventManager` | nothing | an [`IPurchaseEventManager`] |
//! | | 101 | `CreatePermanentEcPurchasedEventManager` | nothing | an [`IPurchaseEventManager`] |
//! | [`IPurchaseEventManager`] | 2 | `GetPurchasedEvent` | nothing | a copied handle: the manager's purchased event |
//! | | 3 | `PopPurchasedProductInfo` | nothing | [`NO_PURCHASED_PRODUCT_INFO_AVAILABLE`] |
//! | | 4 | `PopPurchasedProductInfoWithUid` | nothing | [`NO_PURCHASED_PRODUCT_INFO_AVAILABLE`] |
//!
//! # No add-on content
//!
//! Eden lists the add-on content its content provider has for the caller's title
//! (`AccumulateAOCTitleIDs`, then a filter by base title id). No add-on content is installed
//! here, so the list is empty, and the queries answer as Eden's do for an empty list:
//! `CountAddOnContent` counts none, and `ListAddOnContent` returns none, after Eden's check of
//! the offset against the list's length (`R_UNLESS(out.size() >= offset,
//! AM::ResultApplicationRecordNotFound)`), which an empty list passes only at offset 0. The
//! count asked for does not matter: Eden returns `min(out.size() - offset, count)`, which is 0.
//!
//! # The events
//!
//! The list-changed event is one per server: Eden's `IAddOnContentManager` is one object behind
//! every `aoc:u` session (`RegisterNamedService("aoc:u", std::make_shared<..>)`), which makes
//! the event in its constructor, so commands 8 and 10 hand out the same event on every session.
//! Here it is the server's family state ([`AddOnContent`]), made the first time a command asks
//! for it. Each [`IPurchaseEventManager`] has an event of its own, as Eden makes one in each
//! manager's constructor, made here the first time `GetPurchasedEvent` asks for it. Nothing
//! signals either: Eden never does. A kept event is checked against the kernel before its
//! handle is copied ([`crate::events::unsignalled_event`]), so a restored id that is not an
//! event this server made is refused with the kernel's error.
//!
//! # Left out
//!
//! Every other command goes to the server's unknown-command policy:
//!
//! - Those Eden lists without a handler: 0, 1, 4, 6, 9, 13, 110, 200, 300, 301, 302.
//! - Those whose Eden handler replies success and nothing else, as the auto-stub policy does:
//!   `PrepareAddOnContent` (7), `NotifyMountAddOnContent` (11), `NotifyUnmountAddOnContent`
//!   (12), `CheckAddOnContentMountStatus` (50), and [`IPurchaseEventManager`]'s
//!   `SetDefaultDeliveryTarget` (0) and `SetDeliveryTarget` (1). Issue #278 ports only the
//!   commands the title reaches, and no title has been seen to send these. Under the auto-stub
//!   policy, which a run uses, the answer is already Eden's (success and nothing the caller
//!   reads), and the run's list of auto-stubbed commands names any a title sends, which is the
//!   sign to port it. Under the strict policy, the tests' default, each is answered
//!   [`UNKNOWN_COMMAND_ID`](crate::framework::result::UNKNOWN_COMMAND_ID) (`aoc/tests.rs`:
//!   `commands_that_are_left_out_go_to_the_unknown_command_policy`).
//! - `GetAddOnContentBaseId` (5), which answers with the caller's title id
//!   (`GetDLCBaseTitleId` from the title's control data, or `GetAOCBaseTitleID`), which no
//!   service here knows yet. Under the auto-stub policy it answers 0 where Eden answers that id.
//!
//! # Where this differs from Eden
//!
//! - **A payload too short for `ListAddOnContent`'s offset and count** is the framework's
//!   payload error ([`crate::framework::Fault::PayloadTruncated`]), where Eden reads whatever
//!   the message buffer holds past the payload (`aoc/tests.rs`:
//!   `a_list_request_too_short_for_its_offset_and_count_is_refused`).

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use crate::am::AM_MODULE;
use crate::commands;
use crate::events::unsignalled_event;
use crate::framework::result::code;
use crate::framework::{Context, Interfaces, Outcome, Reply, Request, Server, UnknownPolicy};

#[cfg(test)]
mod tests;

/// The name `aoc:u` is registered under
/// (`eden/src/core/hle/service/aoc/addon_content_manager.cpp`, `LoopProcess`:
/// `RegisterNamedService("aoc:u", ..)`).
pub const U_NAME: &str = "aoc:u";

/// `ErrorModule::NIMShop` (`eden/src/core/hle/result.h` @ cf654052: `NIMShop = 164`).
pub const NIM_SHOP_MODULE: u32 = 164;

/// `AM::ResultApplicationRecordNotFound` (`eden/src/core/hle/service/am/am_results.h` @
/// cf654052: `constexpr Result ResultApplicationRecordNotFound{ErrorModule::AM, 37}`), which
/// `ListAddOnContent` answers for an offset past the list (module docs, "No add-on content").
pub const APPLICATION_RECORD_NOT_FOUND: ResultCode = code(AM_MODULE, 37);

/// `ResultNoPurchasedProductInfoAvailable`
/// (`eden/src/core/hle/service/aoc/purchase_event_manager.cpp` @ cf654052: `constexpr Result
/// ResultNoPurchasedProductInfoAvailable{ErrorModule::NIMShop, 400}`), which both of
/// `IPurchaseEventManager`'s pop commands answer.
pub const NO_PURCHASED_PRODUCT_INFO_AVAILABLE: ResultCode = code(NIM_SHOP_MODULE, 400);

/// The `aoc:u` server's family state: the list-changed event, once a command has returned it
/// (module docs, "The events").
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddOnContent {
    list_changed_event: Option<ObjectId>,
}

impl AddOnContent {
    /// The list-changed event, if a command has returned it yet. A new server's has none.
    #[must_use]
    pub const fn list_changed_event(&self) -> Option<ObjectId> {
        self.list_changed_event
    }
}

/// The root interface of an `aoc:u` session (`IAddOnContentManager`). Its state is the server's
/// ([`AddOnContent`]), so it keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IAddOnContentManager;

/// A purchase event manager (`IPurchaseEventManager`): what commands 100 and 101 hand out.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IPurchaseEventManager {
    purchased_event: Option<ObjectId>,
}

impl IPurchaseEventManager {
    /// The manager's purchased event, if `GetPurchasedEvent` has returned it yet. A new
    /// manager's has none.
    #[must_use]
    pub const fn purchased_event(&self) -> Option<ObjectId> {
        self.purchased_event
    }
}

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `aoc:u`'s commands (`IAddOnContentManager::IAddOnContentManager`).
    pub mod manager {
        /// `CountAddOnContent` (`{2, D<&IAddOnContentManager::CountAddOnContent>, ..}`).
        pub const COUNT_ADD_ON_CONTENT: u32 = 2;
        /// `ListAddOnContent` (`{3, D<&IAddOnContentManager::ListAddOnContent>, ..}`).
        pub const LIST_ADD_ON_CONTENT: u32 = 3;
        /// `GetAddOnContentListChangedEvent` (`{8,
        /// D<&IAddOnContentManager::GetAddOnContentListChangedEvent>, ..}`).
        pub const GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT: u32 = 8;
        /// `GetAddOnContentListChangedEventWithProcessId` (`{10,
        /// D<&IAddOnContentManager::GetAddOnContentListChangedEventWithProcessId>, ..}`).
        pub const GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID: u32 = 10;
        /// `CreateEcPurchasedEventManager` (`{100,
        /// D<&IAddOnContentManager::CreateEcPurchasedEventManager>, ..}`).
        pub const CREATE_EC_PURCHASED_EVENT_MANAGER: u32 = 100;
        /// `CreatePermanentEcPurchasedEventManager` (`{101,
        /// D<&IAddOnContentManager::CreatePermanentEcPurchasedEventManager>, ..}`).
        pub const CREATE_PERMANENT_EC_PURCHASED_EVENT_MANAGER: u32 = 101;
    }

    /// `IPurchaseEventManager`'s commands (`IPurchaseEventManager::IPurchaseEventManager`).
    pub mod purchase {
        /// `GetPurchasedEvent` (`{2, D<&IPurchaseEventManager::GetPurchasedEvent>, ..}`).
        pub const GET_PURCHASED_EVENT: u32 = 2;
        /// `PopPurchasedProductInfo` (`{3, D<&IPurchaseEventManager::PopPurchasedProductInfo>,
        /// ..}`).
        pub const POP_PURCHASED_PRODUCT_INFO: u32 = 3;
        /// `PopPurchasedProductInfoWithUid` (`{4,
        /// D<&IPurchaseEventManager::PopPurchasedProductInfoWithUid>, ..}`).
        pub const POP_PURCHASED_PRODUCT_INFO_WITH_UID: u32 = 4;
    }
}

commands! {
    impl IAddOnContentManager {
        command::manager::COUNT_ADD_ON_CONTENT => count_add_on_content,
        command::manager::LIST_ADD_ON_CONTENT => list_add_on_content [Out(MapAlias)],
        command::manager::GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT => list_changed_event,
        command::manager::GET_ADD_ON_CONTENT_LIST_CHANGED_EVENT_WITH_PROCESS_ID => list_changed_event,
        command::manager::CREATE_EC_PURCHASED_EVENT_MANAGER => create_purchased_event_manager,
        command::manager::CREATE_PERMANENT_EC_PURCHASED_EVENT_MANAGER => create_purchased_event_manager,
    }
}

commands! {
    impl IPurchaseEventManager {
        command::purchase::GET_PURCHASED_EVENT => get_purchased_event,
        command::purchase::POP_PURCHASED_PRODUCT_INFO => pop_purchased_product_info,
        command::purchase::POP_PURCHASED_PRODUCT_INFO_WITH_UID => pop_purchased_product_info,
    }
}

impl IAddOnContentManager {
    /// `CountAddOnContent`. out: `u32` count, 0 (module docs, "No add-on content").
    fn count_add_on_content(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(0);
        reply.ok()
    }

    /// `ListAddOnContent`. in: `u32` offset, `u32` count. out: `u32` count, 0, or
    /// [`APPLICATION_RECORD_NOT_FOUND`] for an offset that is not 0 (module docs, "No add-on
    /// content").
    fn list_add_on_content(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let mut payload = req.payload();
        let offset = payload.read_u32()?;
        let _count = payload.read_u32()?;
        // The list is empty: Eden's `R_UNLESS(out.size() >= offset, ..)` with a size of 0.
        if offset > 0 {
            return Err(APPLICATION_RECORD_NOT_FOUND.into());
        }
        reply.payload().write_u32(0);
        reply.ok()
    }

    /// `GetAddOnContentListChangedEvent` and `GetAddOnContentListChangedEventWithProcessId`.
    /// out: a copied handle to the server's list-changed event, made the first time.
    fn list_changed_event(
        _: &mut Self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let kept = cx.family::<AddOnContent>()?.list_changed_event;
        let event = unsignalled_event(cx, kept)?;
        cx.family::<AddOnContent>()?.list_changed_event = Some(event);
        reply.copy_handle(event);
        reply.ok()
    }

    /// `CreateEcPurchasedEventManager` and `CreatePermanentEcPurchasedEventManager`. out: a new
    /// [`IPurchaseEventManager`].
    fn create_purchased_event_manager(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(IPurchaseEventManager::default()));
        reply.ok()
    }
}

impl IPurchaseEventManager {
    /// `GetPurchasedEvent`. out: a copied handle to the manager's purchased event, made the
    /// first time.
    fn get_purchased_event(
        &mut self,
        cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let event = unsignalled_event(cx, self.purchased_event)?;
        self.purchased_event = Some(event);
        reply.copy_handle(event);
        reply.ok()
    }

    /// `PopPurchasedProductInfo` and `PopPurchasedProductInfoWithUid`: always
    /// [`NO_PURCHASED_PRODUCT_INFO_AVAILABLE`].
    fn pop_purchased_product_info(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        _reply: &mut Reply,
    ) -> Outcome {
        Err(NO_PURCHASED_PRODUCT_INFO_AVAILABLE.into())
    }
}

/// The interfaces an `aoc:u` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new()
        .with::<IAddOnContentManager>()
        .with::<IPurchaseEventManager>()
}

/// The `aoc:u` server, with no add-on content installed.
#[must_use]
pub fn u_server(policy: UnknownPolicy) -> Server {
    Server::new(U_NAME, policy)
        .with_root(|| Box::new(IAddOnContentManager))
        .with_interfaces(interfaces())
        .with_family(AddOnContent::default())
}
