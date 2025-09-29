//! Optimized Keyboard Input Handling
//!
//! This module provides lock-free keyboard input processing
//! to minimize interrupt handler latency.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free ring buffer for keyboard scancodes
const SCANCODE_BUFFER_SIZE: usize = 256;
static SCANCODE_BUFFER: [AtomicUsize; SCANCODE_BUFFER_SIZE] = {
    const INIT: AtomicUsize = AtomicUsize::new(0);
    [INIT; SCANCODE_BUFFER_SIZE]
};

static BUFFER_HEAD: AtomicUsize = AtomicUsize::new(0);
static BUFFER_TAIL: AtomicUsize = AtomicUsize::new(0);

/// Queue a scancode from interrupt handler (lock-free)
#[inline(always)]
pub fn queue_scancode(scancode: u8) {
    let head = BUFFER_HEAD.load(Ordering::Relaxed);
    let next_head = (head + 1) % SCANCODE_BUFFER_SIZE;

    // Check if buffer is full
    if next_head != BUFFER_TAIL.load(Ordering::Acquire) {
        SCANCODE_BUFFER[head].store(scancode as usize, Ordering::Relaxed);
        BUFFER_HEAD.store(next_head, Ordering::Release);
    }
    // If buffer is full, drop the scancode
}

/// Process queued scancodes (called from non-interrupt context)
pub fn process_scancodes() {
    while let Some(scancode) = dequeue_scancode() {
        // Process the scancode
        process_single_scancode(scancode as u8);
    }
}

/// Dequeue a scancode (lock-free)
fn dequeue_scancode() -> Option<usize> {
    let tail = BUFFER_TAIL.load(Ordering::Relaxed);
    if tail != BUFFER_HEAD.load(Ordering::Acquire) {
        let scancode = SCANCODE_BUFFER[tail].load(Ordering::Relaxed);
        let next_tail = (tail + 1) % SCANCODE_BUFFER_SIZE;
        BUFFER_TAIL.store(next_tail, Ordering::Release);
        Some(scancode)
    } else {
        None
    }
}

/// Process a single scancode
fn process_single_scancode(scancode: u8) {
    // Handle keyboard scancode processing here
    // This is deferred from interrupt context
    crate::ipc::send_keyboard_event(scancode as u32);
}