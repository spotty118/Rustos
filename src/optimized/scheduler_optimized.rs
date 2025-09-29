//! Optimized Scheduler with Minimal Latency
//!
//! This module provides high-performance scheduler operations for
//! time-critical interrupt handlers and system calls.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Fast scheduler state for interrupt handlers
static SHOULD_PREEMPT: AtomicBool = AtomicBool::new(false);
static SCHEDULER_TICKS: AtomicU64 = AtomicU64::new(0);

/// Ultra-fast timer tick handler for scheduler
#[inline(always)]
pub fn fast_timer_tick() {
    let ticks = SCHEDULER_TICKS.fetch_add(1, Ordering::Relaxed);

    // Simple time slice check - every 20ms (20 ticks @ 1ms)
    if ticks % 20 == 0 {
        SHOULD_PREEMPT.store(true, Ordering::Relaxed);
    }
}

/// Check if current process should be preempted
#[inline(always)]
pub fn should_preempt_current() -> bool {
    SHOULD_PREEMPT.load(Ordering::Relaxed)
}

/// Reset preemption flag (called after context switch)
#[inline(always)]
pub fn clear_preempt_flag() {
    SHOULD_PREEMPT.store(false, Ordering::Relaxed);
}