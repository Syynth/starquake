use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::page_set::PageSet;
use super::*;

/// The page range of `count` pages starting at page number `first`.
fn pages(first: u64, count: u64) -> PageRange {
    PageRange::new(first * PAGE_SIZE, count * PAGE_SIZE).unwrap()
}

fn drain(watch: &WriteWatch) -> Vec<PageRange> {
    let mut out = Vec::new();
    watch.take_dirty(&mut out);
    out
}

fn ranges(set: &mut PageSet) -> Vec<PageRange> {
    let mut out = Vec::new();
    set.drain_into(&mut out);
    out
}

// --- PageRange ---

#[test]
fn page_range_validates_in_order() {
    assert_eq!(
        PageRange::new(PAGE_SIZE + 1, PAGE_SIZE - 1),
        Err(MapError::MisalignedAddress {
            address: PAGE_SIZE + 1
        })
    );
    assert_eq!(
        PageRange::new(PAGE_SIZE, PAGE_SIZE - 1),
        Err(MapError::MisalignedSize {
            size: PAGE_SIZE - 1
        })
    );
    assert_eq!(PageRange::new(PAGE_SIZE, 0), Err(MapError::Empty));
    // The last page of the 64-bit space ends at 2^64, which is not representable.
    let last = u64::MAX - (PAGE_SIZE - 1);
    assert_eq!(
        PageRange::new(last, PAGE_SIZE),
        Err(MapError::VirtualRange {
            va: last,
            size: PAGE_SIZE
        })
    );
    // One page below it ends exactly at the last page's start.
    let below = last - PAGE_SIZE;
    let range = PageRange::new(below, PAGE_SIZE).unwrap();
    assert_eq!(
        (range.start(), range.size(), range.end()),
        (below, PAGE_SIZE, last)
    );
}

#[test]
fn covering_rounds_out_to_whole_pages_at_the_exact_edges() {
    // One byte at a page's start and at its last byte: one page.
    assert_eq!(PageRange::covering(PAGE_SIZE, 1), Ok(pages(1, 1)));
    assert_eq!(PageRange::covering(2 * PAGE_SIZE - 1, 1), Ok(pages(1, 1)));
    // A whole page exactly, and one byte more.
    assert_eq!(PageRange::covering(PAGE_SIZE, PAGE_SIZE), Ok(pages(1, 1)));
    assert_eq!(
        PageRange::covering(PAGE_SIZE, PAGE_SIZE + 1),
        Ok(pages(1, 2))
    );
    // Two bytes straddling a page boundary.
    assert_eq!(PageRange::covering(2 * PAGE_SIZE - 1, 2), Ok(pages(1, 2)));
    assert_eq!(PageRange::covering(PAGE_SIZE, 0), Err(MapError::Empty));
}

#[test]
fn covering_refuses_ranges_whose_end_is_not_representable() {
    let last = u64::MAX - (PAGE_SIZE - 1);
    // Ends at the last byte below the last page: fine.
    assert_eq!(
        PageRange::covering(last - 8, 8),
        Ok(PageRange::new(last - PAGE_SIZE, PAGE_SIZE).unwrap())
    );
    // Touches the last page: its end is 2^64.
    assert_eq!(
        PageRange::covering(last - 8, 9),
        Err(MapError::VirtualRange {
            va: last - 8,
            size: 9
        })
    );
    // The byte count itself overflows.
    assert_eq!(
        PageRange::covering(u64::MAX, 2),
        Err(MapError::VirtualRange {
            va: u64::MAX,
            size: 2
        })
    );
}

#[test]
fn contains_is_exact_at_both_ends() {
    let range = pages(4, 2);
    assert!(!range.contains(4 * PAGE_SIZE - 1));
    assert!(range.contains(4 * PAGE_SIZE));
    assert!(range.contains(6 * PAGE_SIZE - 1));
    assert!(!range.contains(6 * PAGE_SIZE));
}

// --- PageSet ---

#[test]
fn page_set_coalesces_overlapping_and_touching_ranges() {
    let mut set = PageSet::default();
    set.insert(pages(10, 2)); // 10..12
    set.insert(pages(14, 2)); // 14..16, a gap of one page
    assert_eq!(ranges(&mut set), [pages(10, 2), pages(14, 2)]);

    set.insert(pages(10, 2));
    set.insert(pages(14, 2));
    // A neighbour starting exactly at the end bound joins it.
    set.insert(pages(12, 1));
    assert_eq!(ranges(&mut set), [pages(10, 3), pages(14, 2)]);

    set.insert(pages(10, 2));
    set.insert(pages(14, 2));
    // A neighbour ending exactly at the start bound joins it.
    set.insert(pages(13, 1));
    assert_eq!(ranges(&mut set), [pages(10, 2), pages(13, 3)]);

    set.insert(pages(10, 2));
    set.insert(pages(14, 2));
    set.insert(pages(20, 1));
    // One range swallowing two and overlapping a third's start.
    set.insert(pages(9, 12));
    assert_eq!(ranges(&mut set), [pages(9, 12)]);

    // A range inside another changes nothing.
    set.insert(pages(10, 5));
    set.insert(pages(11, 1));
    assert_eq!(ranges(&mut set), [pages(10, 5)]);
}

#[test]
fn page_set_removes_a_page_and_leaves_its_neighbours() {
    for (removed, expected) in [
        (10, vec![pages(11, 2)]),
        (11, vec![pages(10, 1), pages(12, 1)]),
        (12, vec![pages(10, 2)]),
    ] {
        let mut set = PageSet::default();
        set.insert(pages(10, 3));
        set.insert(pages(14, 1));
        assert!(set.remove_page(removed * PAGE_SIZE), "page {removed}");
        let mut expected = expected;
        expected.push(pages(14, 1));
        assert_eq!(ranges(&mut set), expected, "page {removed}");
    }
    // The pages just outside, and the gap, are not in the set; nothing changes.
    let mut set = PageSet::default();
    set.insert(pages(10, 3));
    set.insert(pages(14, 1));
    for outside in [9, 13, 15] {
        assert!(!set.remove_page(outside * PAGE_SIZE), "page {outside}");
    }
    assert!(set.contains(10 * PAGE_SIZE) && set.contains(13 * PAGE_SIZE - 1));
    assert!(!set.contains(13 * PAGE_SIZE));
    assert_eq!(ranges(&mut set), [pages(10, 3), pages(14, 1)]);
}

#[test]
fn page_set_handles_the_highest_representable_page() {
    let last = u64::MAX - (PAGE_SIZE - 1);
    let top = PageRange::new(last - 2 * PAGE_SIZE, 2 * PAGE_SIZE).unwrap();
    let mut set = PageSet::default();
    set.insert(top);
    assert!(set.remove_page(last - PAGE_SIZE));
    assert!(!set.remove_page(last));
    assert_eq!(
        ranges(&mut set),
        [PageRange::new(last - 2 * PAGE_SIZE, PAGE_SIZE).unwrap()]
    );
}

// --- WriteWatch ---

#[test]
fn a_request_is_armed_only_after_the_kernel_arms_it() {
    let watch = WriteWatch::new();
    assert!(!watch.has_requests());
    let ticket = watch.request(pages(4, 2));
    assert_eq!(ticket.get(), 0);
    assert!(watch.has_requests());
    assert!(!watch.armed(ticket));

    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    assert_eq!(taken.len(), 1);
    assert_eq!((taken[0].ticket(), taken[0].range()), (ticket, pages(4, 2)));
    assert!(!watch.has_requests());
    // Taken is not armed: the kernel has not protected and flushed yet.
    assert!(!watch.armed(ticket));
    // Nor are its pages watched yet.
    assert!(!watch.record_fault(4 * PAGE_SIZE));

    watch.arm(&taken).unwrap();
    assert!(watch.armed(ticket));
    // A ticket never issued is not armed.
    assert!(!watch.armed(WatchTicket(1)));
}

#[test]
fn a_watched_page_faults_once_and_is_drained_once() {
    let watch = WriteWatch::new();
    watch.request(pages(4, 2));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();

    assert_eq!(drain(&watch), []);
    assert!(watch.record_fault(5 * PAGE_SIZE + 0x123));
    // Write access is restored: a second store to the page is not the watch's fault.
    assert!(!watch.record_fault(5 * PAGE_SIZE));
    assert_eq!(drain(&watch), [pages(5, 1)]);
    assert_eq!(drain(&watch), []);
    // The other page is still watched.
    assert!(watch.record_fault(4 * PAGE_SIZE));
    assert_eq!(drain(&watch), [pages(4, 1)]);
}

#[test]
fn faults_are_exact_at_the_watched_range_edges() {
    let watch = WriteWatch::new();
    watch.request(pages(4, 2)); // 4..6
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    // The byte below the start and the byte at the end bound are outside.
    assert!(!watch.record_fault(4 * PAGE_SIZE - 1));
    assert!(!watch.record_fault(6 * PAGE_SIZE));
    assert_eq!(drain(&watch), []);
    // The first and last bytes are inside.
    assert!(watch.record_fault(4 * PAGE_SIZE));
    assert!(watch.record_fault(6 * PAGE_SIZE - 1));
    assert_eq!(drain(&watch), [pages(4, 2)]);
}

#[test]
fn the_dirty_log_coalesces_in_address_order() {
    let watch = WriteWatch::new();
    watch.request(pages(1, 3));
    watch.request(pages(8, 1));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    for page in [8, 3, 1, 2] {
        assert!(watch.record_fault(page * PAGE_SIZE), "page {page}");
    }
    assert_eq!(drain(&watch), [pages(1, 3), pages(8, 1)]);
}

#[test]
fn take_dirty_appends_without_clearing() {
    let watch = WriteWatch::new();
    watch.request(pages(2, 1));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    assert!(watch.record_fault(2 * PAGE_SIZE));
    let mut out = vec![pages(100, 1)];
    watch.take_dirty(&mut out);
    assert_eq!(out, [pages(100, 1), pages(2, 1)]);
}

#[test]
fn armed_waits_for_every_earlier_request() {
    let watch = WriteWatch::new();
    let first = watch.request(pages(1, 1));
    let mut early = Vec::new();
    watch.take_requests(&mut early);
    let second = watch.request(pages(2, 1));
    let third = watch.request(pages(3, 1));
    let mut late = Vec::new();
    watch.take_requests(&mut late);
    assert_eq!(late.len(), 2);

    // The later batch armed first: its pages are watched, but its tickets read as unarmed.
    watch.arm(&late).unwrap();
    assert!(!watch.armed(first) && !watch.armed(second) && !watch.armed(third));
    assert!(watch.record_fault(3 * PAGE_SIZE));

    watch.arm(&early).unwrap();
    assert!(watch.armed(first) && watch.armed(second) && watch.armed(third));
}

#[test]
fn a_waiting_request_holds_back_later_ones() {
    let watch = WriteWatch::new();
    let first = watch.request(pages(1, 1));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    assert!(watch.armed(first));
    // A new request waiting does not unarm the earlier one.
    let second = watch.request(pages(2, 1));
    assert!(watch.armed(first));
    assert!(!watch.armed(second));
}

#[test]
fn arm_is_all_or_nothing() {
    let watch = WriteWatch::new();
    watch.request(pages(1, 1));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    let valid = taken[0];
    // A request never taken: the next ticket, issued but still waiting.
    let waiting_ticket = watch.request(pages(2, 1));
    let mut waiting = Vec::new();
    let stray = WatchRequest {
        ticket: waiting_ticket,
        range: pages(2, 1),
    };

    // Both faults present: the valid request first, the stray second. Nothing is armed.
    assert_eq!(
        watch.arm(&[valid, stray]),
        Err(WatchError::NotTaken { ticket: 1 })
    );
    assert!(!watch.armed(valid.ticket()));
    assert!(!watch.record_fault(PAGE_SIZE));

    // A taken ticket with a different range is refused too.
    let forged = WatchRequest {
        ticket: valid.ticket(),
        range: pages(1, 2),
    };
    assert_eq!(
        watch.arm(&[forged]),
        Err(WatchError::NotTaken { ticket: 0 })
    );

    watch.arm(&[valid]).unwrap();
    assert!(watch.armed(valid.ticket()));
    // Arming it again is refused: it is no longer waiting to be armed.
    assert_eq!(watch.arm(&[valid]), Err(WatchError::NotTaken { ticket: 0 }));
    watch.take_requests(&mut waiting);
    assert_eq!(waiting, [stray]);
}

#[test]
fn arm_tolerates_a_request_listed_twice() {
    let watch = WriteWatch::new();
    let ticket = watch.request(pages(1, 2));
    let mut taken = Vec::new();
    watch.take_requests(&mut taken);
    watch.arm(&[taken[0], taken[0]]).unwrap();
    assert!(watch.armed(ticket));
    assert!(watch.record_fault(PAGE_SIZE));
    assert!(watch.record_fault(2 * PAGE_SIZE));
    assert_eq!(drain(&watch), [pages(1, 2)]);
}

#[test]
fn rewatching_a_dirtied_page_takes_a_new_request() {
    let watch = WriteWatch::new();
    let mut taken = Vec::new();
    watch.request(pages(1, 1));
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    assert!(watch.record_fault(PAGE_SIZE));
    assert!(!watch.record_fault(PAGE_SIZE));

    taken.clear();
    watch.request(pages(1, 1));
    watch.take_requests(&mut taken);
    watch.arm(&taken).unwrap();
    assert!(watch.record_fault(PAGE_SIZE));
    // Both faults were on one page: the log holds it once.
    assert_eq!(drain(&watch), [pages(1, 1)]);
}

#[test]
fn watch_error_is_plain_words() {
    assert_eq!(
        WatchError::NotTaken { ticket: 7 }.to_string(),
        "watch request 7 is not waiting to be armed"
    );
}

#[test]
fn concurrent_requests_and_kernel_entries_arm_everything() {
    const REQUESTS: u64 = 2_000;
    let watch = Arc::new(WriteWatch::new());
    let done = Arc::new(AtomicBool::new(false));
    let kernel = {
        let watch = Arc::clone(&watch);
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let mut taken = Vec::new();
            let mut armed = 0u64;
            // Runs while the requester does, then once more: a take that starts after `done` was
            // seen finds every request. No flag of the watch decides when it stops, so a broken
            // watch fails the counts below instead of hanging the test.
            loop {
                let finished = done.load(Ordering::Acquire);
                taken.clear();
                watch.take_requests(&mut taken);
                watch.arm(&taken).unwrap();
                armed += u64::try_from(taken.len()).unwrap();
                if finished {
                    break armed;
                }
                std::thread::yield_now();
            }
        })
    };
    let mut last = None;
    for i in 0..REQUESTS {
        last = Some(watch.request(pages(i * 2, 1)));
    }
    done.store(true, Ordering::Release);
    assert_eq!(kernel.join().unwrap(), REQUESTS);
    assert!(watch.armed(last.unwrap()));
    assert!(!watch.has_requests());
    for i in 0..REQUESTS {
        assert!(watch.record_fault(i * 2 * PAGE_SIZE), "page {}", i * 2);
    }
    let dirty = drain(&watch);
    assert_eq!(dirty.len(), usize::try_from(REQUESTS).unwrap());
}

// --- FakeWatchKernel ---

#[test]
fn fake_kernel_arms_on_entry_and_faults_on_cpu_writes() {
    let watch = Arc::new(WriteWatch::new());
    let mut kernel = FakeWatchKernel::new(Arc::clone(&watch));
    assert!(Arc::ptr_eq(kernel.watch(), &watch));
    let ticket = watch.request(pages(4, 3)); // 4..7

    // Before the kernel entry nothing is protected: a store does not fault.
    assert_eq!(kernel.cpu_write(4 * PAGE_SIZE, 3 * PAGE_SIZE), Ok(0));
    assert!(!watch.armed(ticket));
    assert_eq!(kernel.enter(), Ok(1));
    assert!(watch.armed(ticket));
    // A second entry has nothing to arm.
    assert_eq!(kernel.enter(), Ok(0));

    // A store from the last byte below the range into its first page faults one page.
    assert_eq!(kernel.cpu_write(4 * PAGE_SIZE - 1, 2), Ok(1));
    // A store across pages 5 and 6 and the page past the end faults two.
    assert_eq!(kernel.cpu_write(6 * PAGE_SIZE - 1, PAGE_SIZE + 2), Ok(2));
    assert_eq!(kernel.cpu_write(4 * PAGE_SIZE, 3 * PAGE_SIZE), Ok(0));
    assert_eq!(drain(&watch), [pages(4, 3)]);
}

#[test]
fn fake_kernel_refuses_a_store_past_the_address_space() {
    let kernel = FakeWatchKernel::new(Arc::new(WriteWatch::new()));
    assert_eq!(
        kernel.cpu_write(u64::MAX, 1),
        Err(MapError::VirtualRange {
            va: u64::MAX,
            size: 1
        })
    );
    assert_eq!(kernel.cpu_write(PAGE_SIZE, 0), Err(MapError::Empty));
}
