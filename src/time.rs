//! Production time and timer subsystem for RustOS
//!
//! Provides real timer functionality using x86_64 hardware timers

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::instructions::port::Port;

/// PIT (Programmable Interval Timer) frequency
const PIT_FREQUENCY: u32 = 1193182;
/// Target timer frequency in Hz
const TIMER_FREQUENCY: u32 = 100;
/// PIT divisor for desired frequency
const PIT_DIVISOR: u16 = (PIT_FREQUENCY / TIMER_FREQUENCY) as u16;

/// Global tick counter
static TICKS: AtomicU64 = AtomicU64::new(0);
/// TSC frequency in Hz (calibrated at boot)
static TSC_FREQUENCY: AtomicU64 = AtomicU64::new(0);
/// Boot TSC value
static BOOT_TSC: AtomicU64 = AtomicU64::new(0);

/// Initialize the timer subsystem
pub fn init() -> Result<(), &'static str> {
    // Configure PIT channel 0
    unsafe {
        let mut cmd = Port::<u8>::new(0x43);
        let mut data = Port::<u8>::new(0x40);
        
        // Channel 0, lobyte/hibyte, rate generator
        cmd.write(0x36);
        
        // Write frequency divisor
        data.write((PIT_DIVISOR & 0xFF) as u8);
        data.write((PIT_DIVISOR >> 8) as u8);
    }
    
    // Calibrate TSC
    calibrate_tsc();
    
    // Record boot TSC
    BOOT_TSC.store(read_tsc(), Ordering::Relaxed);
    
    Ok(())
}

/// Timer interrupt handler - called by interrupt system
pub fn timer_tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
}

/// Get system uptime in milliseconds
pub fn uptime_ms() -> u64 {
    let ticks = TICKS.load(Ordering::Relaxed);
    (ticks * 1000) / TIMER_FREQUENCY as u64
}

/// Get system uptime in microseconds
pub fn uptime_us() -> u64 {
    if TSC_FREQUENCY.load(Ordering::Relaxed) > 0 {
        // Use high-precision TSC if calibrated
        let current_tsc = read_tsc();
        let boot_tsc = BOOT_TSC.load(Ordering::Relaxed);
        let tsc_freq = TSC_FREQUENCY.load(Ordering::Relaxed);
        
        if tsc_freq > 0 && current_tsc > boot_tsc {
            ((current_tsc - boot_tsc) * 1_000_000) / tsc_freq
        } else {
            // Fallback to tick-based timing
            uptime_ms() * 1000
        }
    } else {
        uptime_ms() * 1000
    }
}

/// Read the Time Stamp Counter
pub fn read_tsc() -> u64 {
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
}

/// Calibrate TSC frequency using PIT
fn calibrate_tsc() {
    // Wait for next timer tick
    let start_ticks = TICKS.load(Ordering::Relaxed);
    while TICKS.load(Ordering::Relaxed) == start_ticks {
        core::hint::spin_loop();
    }

    // Record start TSC
    let start_tsc = read_tsc();
    let start = TICKS.load(Ordering::Relaxed);

    // Wait for 10 ticks (100ms at 100Hz)
    while TICKS.load(Ordering::Relaxed) < start + 10 {
        core::hint::spin_loop();
    }

    // Record end TSC
    let end_tsc = read_tsc();

    // Calculate frequency
    let tsc_delta = end_tsc - start_tsc;
    let time_ms = (10 * 1000) / TIMER_FREQUENCY as u64;
    let freq = (tsc_delta * 1000) / time_ms;

    TSC_FREQUENCY.store(freq, Ordering::Relaxed);
}

/// Sleep for specified milliseconds (busy wait)
pub fn sleep_ms(ms: u64) {
    let start = uptime_ms();
    while uptime_ms() < start + ms {
        core::hint::spin_loop();
    }
}

/// High-resolution timer for performance measurement
pub struct Timer {
    start_tsc: u64,
}

impl Timer {
    /// Create a new timer starting now
    pub fn new() -> Self {
        Self {
            start_tsc: read_tsc(),
        }
    }
    
    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        let freq = TSC_FREQUENCY.load(Ordering::Relaxed);
        if freq > 0 {
            let delta = read_tsc() - self.start_tsc;
            (delta * 1_000_000) / freq
        } else {
            0
        }
    }
    
    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_us() / 1000
    }
}
