//! Time Management Module for RustOS
//!
//! This module provides timing functionality including uptime tracking,
//! timer interrupts, and system time management.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;

/// System uptime in milliseconds
static UPTIME_MS: AtomicU64 = AtomicU64::new(0);

/// System tick counter
static SYSTEM_TICKS: AtomicU64 = AtomicU64::new(0);

/// Timer interrupt frequency (Hz)
pub const TIMER_FREQUENCY: u64 = 1000; // 1 kHz = 1ms per tick

/// Timer statistics
#[derive(Debug, Clone)]
pub struct TimerStats {
    pub total_ticks: u64,
    pub uptime_ms: u64,
    pub timer_frequency: u64,
    pub interrupts_per_second: f64,
}

lazy_static! {
    /// Timer configuration and state
    static ref TIMER_STATE: Mutex<TimerState> = Mutex::new(TimerState::new());
}

/// Internal timer state
struct TimerState {
    initialized: bool,
    last_tick_time: u64,
    tick_count: u64,
}

impl TimerState {
    fn new() -> Self {
        Self {
            initialized: false,
            last_tick_time: 0,
            tick_count: 0,
        }
    }
}

/// Initialize the timer system
pub fn init() -> Result<(), &'static str> {
    let mut state = TIMER_STATE.lock();
    
    if state.initialized {
        return Ok(());
    }

    // Reset counters
    UPTIME_MS.store(0, Ordering::SeqCst);
    SYSTEM_TICKS.store(0, Ordering::SeqCst);
    
    state.initialized = true;
    state.tick_count = 0;
    state.last_tick_time = 0;

    Ok(())
}

/// Called by timer interrupt handler
pub fn timer_interrupt_handler() {
    // Increment system ticks
    let ticks = SYSTEM_TICKS.fetch_add(1, Ordering::SeqCst) + 1;
    
    // Update uptime (assuming 1ms per tick)
    let uptime = ticks * (1000 / TIMER_FREQUENCY);
    UPTIME_MS.store(uptime, Ordering::SeqCst);
    
    // Update timer state
    let mut state = TIMER_STATE.lock();
    state.tick_count = ticks;
    state.last_tick_time = uptime;
}

/// Get system uptime in milliseconds
pub fn uptime_ms() -> u64 {
    UPTIME_MS.load(Ordering::SeqCst)
}

/// Get system uptime in seconds
pub fn uptime_seconds() -> u64 {
    uptime_ms() / 1000
}

/// Get total system ticks
pub fn system_ticks() -> u64 {
    SYSTEM_TICKS.load(Ordering::SeqCst)
}

/// Get timer statistics
pub fn get_timer_stats() -> TimerStats {
    let ticks = SYSTEM_TICKS.load(Ordering::SeqCst);
    let uptime = UPTIME_MS.load(Ordering::SeqCst);
    
    let interrupts_per_second = if uptime > 0 {
        (ticks as f64 * 1000.0) / uptime as f64
    } else {
        0.0
    };

    TimerStats {
        total_ticks: ticks,
        uptime_ms: uptime,
        timer_frequency: TIMER_FREQUENCY,
        interrupts_per_second,
    }
}

/// Sleep for the specified number of milliseconds (busy wait)
pub fn sleep_ms(ms: u64) {
    let start = uptime_ms();
    while uptime_ms() - start < ms {
        core::hint::spin_loop();
    }
}

/// Sleep for the specified number of microseconds (busy wait)
pub fn sleep_us(us: u64) {
    // Simple busy wait - not accurate but functional
    for _ in 0..(us * 1000) {
        core::hint::spin_loop();
    }
}

/// Get current timestamp in milliseconds since boot
pub fn current_time_ms() -> u64 {
    uptime_ms()
}

/// Time measurement utility for benchmarking
pub struct Timer {
    start_time: u64,
}

impl Timer {
    /// Create a new timer and start measuring
    pub fn new() -> Self {
        Self {
            start_time: uptime_ms(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        uptime_ms() - self.start_time
    }

    /// Get elapsed time in microseconds (estimated)
    pub fn elapsed_us(&self) -> u64 {
        self.elapsed_ms() * 1000
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        self.start_time = uptime_ms();
    }
}

/// Default timer implementation
impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Time-based random number generator state
static TIME_RNG_STATE: AtomicU64 = AtomicU64::new(1);

/// Simple time-based pseudo-random number generator
pub fn time_based_random() -> u64 {
    let current_time = uptime_ms();
    let ticks = system_ticks();
    
    let mut state = TIME_RNG_STATE.load(Ordering::SeqCst);
    state = state.wrapping_mul(1103515245).wrapping_add(12345);
    state ^= current_time;
    state ^= ticks << 32;
    
    TIME_RNG_STATE.store(state, Ordering::SeqCst);
    state
}

/// Initialize RTC (Real Time Clock) if available
pub fn init_rtc() -> Result<(), &'static str> {
    // For now, just return success
    // In a real implementation, this would configure the RTC hardware
    Ok(())
}

/// Performance counter for measuring execution time
pub struct PerfCounter {
    start_ticks: u64,
    name: &'static str,
}

impl PerfCounter {
    /// Create a new performance counter
    pub fn new(name: &'static str) -> Self {
        Self {
            start_ticks: system_ticks(),
            name,
        }
    }

    /// Finish measurement and return elapsed ticks
    pub fn finish(self) -> u64 {
        system_ticks() - self.start_ticks
    }

    /// Get current elapsed ticks without finishing
    pub fn elapsed_ticks(&self) -> u64 {
        system_ticks() - self.start_ticks
    }
}

impl Drop for PerfCounter {
    fn drop(&mut self) {
        let elapsed = system_ticks() - self.start_ticks;
        // In a real implementation, we might log this or store it somewhere
        // For now, we just calculate it
        let _ = elapsed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_creation() {
        let timer = Timer::new();
        assert!(timer.start_time <= uptime_ms());
    }

    #[test]
    fn test_perf_counter() {
        let counter = PerfCounter::new("test");
        let start_ticks = counter.start_ticks;
        assert!(counter.elapsed_ticks() >= 0);
        let elapsed = counter.finish();
        assert!(elapsed >= 0);
    }

    #[test]
    fn test_timer_stats() {
        let stats = get_timer_stats();
        assert_eq!(stats.timer_frequency, TIMER_FREQUENCY);
    }
}