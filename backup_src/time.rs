//! Timer and Clock Management System for RustOS
//!
//! This module provides:
//! - System timer initialization and management
//! - High-resolution timing functions
//! - Clock synchronization and calibration
//! - Timer interrupt handling
//! - Real-time clock (RTC) support
//! - Time-based scheduling support

use x86_64::instructions::port::Port;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;

/// System tick frequency (1000 Hz = 1ms resolution)
pub const TIMER_FREQUENCY: u32 = 1000;

/// PIT (Programmable Interval Timer) constants
const PIT_FREQUENCY: u32 = 1193182;
const PIT_COMMAND_PORT: u16 = 0x43;
const PIT_CHANNEL_0_PORT: u16 = 0x40;

/// RTC (Real-Time Clock) ports
const RTC_COMMAND_PORT: u16 = 0x70;
const RTC_DATA_PORT: u16 = 0x71;

/// HPET (High Precision Event Timer) constants
const HPET_BASE_ADDRESS: u64 = 0xFED00000;

/// System uptime in milliseconds
static SYSTEM_UPTIME: AtomicU64 = AtomicU64::new(0);

/// CPU cycles per millisecond (calibrated at startup)
static CPU_CYCLES_PER_MS: AtomicU64 = AtomicU64::new(0);

/// Timer calibration state
static TIMER_CALIBRATED: AtomicU64 = AtomicU64::new(0);

/// Timer interrupt handler statistics
#[derive(Debug, Default)]
pub struct TimerStats {
    pub total_ticks: u64,
    pub missed_ticks: u64,
    pub average_jitter: u64,
    pub max_jitter: u64,
    pub timer_drift: i64,
}

lazy_static! {
    static ref TIMER_STATS: Mutex<TimerStats> = Mutex::new(TimerStats::default());
}

/// Time structure for representing system time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SystemTime {
    pub milliseconds: u64,
}

impl SystemTime {
    /// Create a new SystemTime from milliseconds
    pub fn from_millis(ms: u64) -> Self {
        SystemTime { milliseconds: ms }
    }

    /// Get the current system time
    pub fn now() -> Self {
        SystemTime {
            milliseconds: SYSTEM_UPTIME.load(Ordering::Acquire),
        }
    }

    /// Convert to microseconds
    pub fn as_micros(&self) -> u64 {
        self.milliseconds * 1000
    }

    /// Convert to nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.milliseconds * 1_000_000
    }

    /// Calculate duration since another time
    pub fn duration_since(&self, other: SystemTime) -> Option<Duration> {
        if self.milliseconds >= other.milliseconds {
            Some(Duration::from_millis(self.milliseconds - other.milliseconds))
        } else {
            None
        }
    }

    /// Add a duration to this time
    pub fn add_duration(&self, duration: Duration) -> SystemTime {
        SystemTime {
            milliseconds: self.milliseconds + duration.as_millis(),
        }
    }
}

/// Duration structure for representing time intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    pub milliseconds: u64,
}

impl Duration {
    /// Create a duration from milliseconds
    pub fn from_millis(ms: u64) -> Self {
        Duration { milliseconds: ms }
    }

    /// Create a duration from microseconds
    pub fn from_micros(us: u64) -> Self {
        Duration { milliseconds: us / 1000 }
    }

    /// Create a duration from nanoseconds
    pub fn from_nanos(ns: u64) -> Self {
        Duration { milliseconds: ns / 1_000_000 }
    }

    /// Create a duration from seconds
    pub fn from_secs(s: u64) -> Self {
        Duration { milliseconds: s * 1000 }
    }

    /// Get duration as milliseconds
    pub fn as_millis(&self) -> u64 {
        self.milliseconds
    }

    /// Get duration as microseconds
    pub fn as_micros(&self) -> u64 {
        self.milliseconds * 1000
    }

    /// Get duration as nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.milliseconds * 1_000_000
    }

    /// Check if duration is zero
    pub fn is_zero(&self) -> bool {
        self.milliseconds == 0
    }
}

/// Timer manager for handling multiple timer sources
pub struct TimerManager {
    pit_initialized: bool,
    rtc_initialized: bool,
    hpet_available: bool,
    tsc_frequency: u64,
}

impl TimerManager {
    pub fn new() -> Self {
        TimerManager {
            pit_initialized: false,
            rtc_initialized: false,
            hpet_available: false,
            tsc_frequency: 0,
        }
    }

    /// Initialize the timer system
    pub fn init(&mut self) -> Result<(), &'static str> {
        crate::println!("[TIME] Initializing timer subsystem...");

        // Initialize PIT (Programmable Interval Timer)
        self.init_pit()?;

        // Try to initialize HPET if available
        if self.detect_hpet() {
            self.init_hpet()?;
            crate::println!("[TIME] HPET (High Precision Event Timer) available");
        } else {
            crate::println!("[TIME] HPET not available, using PIT fallback");
        }

        // Initialize RTC for wall clock time
        self.init_rtc()?;

        // Calibrate TSC (Time Stamp Counter)
        self.calibrate_tsc()?;

        crate::println!("[TIME] Timer system initialized successfully");
        Ok(())
    }

    /// Initialize Programmable Interval Timer
    fn init_pit(&mut self) -> Result<(), &'static str> {
        let divisor = PIT_FREQUENCY / TIMER_FREQUENCY;
        if divisor > 0xFFFF {
            return Err("Timer frequency too low for PIT");
        }

        unsafe {
            // Configure PIT channel 0 for square wave mode
            let mut command_port = Port::new(PIT_COMMAND_PORT);
            command_port.write(0x36u8); // Channel 0, LSB/MSB, square wave, binary

            // Set the frequency divisor
            let mut data_port = Port::new(PIT_CHANNEL_0_PORT);
            data_port.write((divisor & 0xFF) as u8); // LSB
            data_port.write((divisor >> 8) as u8);   // MSB
        }

        self.pit_initialized = true;
        crate::println!("[TIME] PIT initialized at {} Hz", TIMER_FREQUENCY);
        Ok(())
    }

    /// Detect HPET availability
    fn detect_hpet(&self) -> bool {
        // In a real implementation, this would check ACPI tables
        // For now, we'll assume HPET is not available
        false
    }

    /// Initialize High Precision Event Timer
    fn init_hpet(&mut self) -> Result<(), &'static str> {
        // HPET initialization would require memory mapping and ACPI parsing
        // This is a placeholder for future implementation
        self.hpet_available = true;
        Ok(())
    }

    /// Initialize Real-Time Clock
    fn init_rtc(&mut self) -> Result<(), &'static str> {
        unsafe {
            let mut command_port = Port::new(RTC_COMMAND_PORT);
            let mut data_port = Port::new(RTC_DATA_PORT);

            // Read RTC status to ensure it's working
            command_port.write(0x0A); // Status Register A
            let status = data_port.read();

            if status & 0x80 != 0 {
                return Err("RTC update in progress");
            }
        }

        self.rtc_initialized = true;
        crate::println!("[TIME] RTC initialized");
        Ok(())
    }

    /// Calibrate Time Stamp Counter
    fn calibrate_tsc(&mut self) -> Result<(), &'static str> {
        crate::println!("[TIME] Calibrating TSC...");

        let start_tsc = crate::arch::get_cpu_cycles();
        let start_time = self.get_pit_time();

        // Wait approximately 10ms for calibration
        self.busy_wait_ms(10);

        let end_tsc = crate::arch::get_cpu_cycles();
        let end_time = self.get_pit_time();

        let cycles_elapsed = end_tsc - start_tsc;
        let time_elapsed = end_time - start_time;

        if time_elapsed == 0 {
            return Err("TSC calibration failed: no time elapsed");
        }

        self.tsc_frequency = cycles_elapsed / time_elapsed;
        CPU_CYCLES_PER_MS.store(self.tsc_frequency, Ordering::Release);
        TIMER_CALIBRATED.store(1, Ordering::Release);

        crate::println!("[TIME] TSC calibrated: {} cycles/ms", self.tsc_frequency);
        Ok(())
    }

    /// Get current PIT time (for calibration)
    fn get_pit_time(&self) -> u64 {
        SYSTEM_UPTIME.load(Ordering::Acquire)
    }

    /// Busy wait for specified milliseconds (for calibration only)
    fn busy_wait_ms(&self, ms: u64) {
        let start = SYSTEM_UPTIME.load(Ordering::Acquire);
        while SYSTEM_UPTIME.load(Ordering::Acquire) - start < ms {
            core::hint::spin_loop();
        }
    }

    /// Get high-resolution timestamp using TSC
    pub fn get_high_res_time(&self) -> u64 {
        if TIMER_CALIBRATED.load(Ordering::Acquire) == 0 {
            // Fallback to system uptime if TSC not calibrated
            return SYSTEM_UPTIME.load(Ordering::Acquire) * 1000; // Convert to microseconds
        }

        let cycles = crate::arch::get_cpu_cycles();
        let cycles_per_us = CPU_CYCLES_PER_MS.load(Ordering::Acquire) / 1000;

        if cycles_per_us > 0 {
            cycles / cycles_per_us
        } else {
            SYSTEM_UPTIME.load(Ordering::Acquire) * 1000
        }
    }

    /// Get timer statistics
    pub fn get_stats(&self) -> TimerStats {
        TIMER_STATS.lock().clone()
    }
}

lazy_static! {
    static ref TIMER_MANAGER: Mutex<TimerManager> = Mutex::new(TimerManager::new());
}

/// Timer interrupt handler - called by interrupt system
pub fn timer_interrupt_handler() {
    let current_uptime = SYSTEM_UPTIME.fetch_add(1, Ordering::AcqRel);

    // Update timer statistics
    {
        let mut stats = TIMER_STATS.lock();
        stats.total_ticks += 1;

        // Calculate jitter (simplified - would need more sophisticated measurement)
        let expected_time = stats.total_ticks * (1000 / TIMER_FREQUENCY as u64);
        let actual_time = current_uptime + 1;
        let jitter = if actual_time > expected_time {
            actual_time - expected_time
        } else {
            expected_time - actual_time
        };

        if jitter > stats.max_jitter {
            stats.max_jitter = jitter;
        }

        // Update average jitter (simple moving average)
        stats.average_jitter = (stats.average_jitter * 7 + jitter) / 8;
    }

    // Notify scheduler about timer tick
    crate::task::scheduler_tick();
}

/// Initialize the timer system
pub fn init() -> Result<(), &'static str> {
    TIMER_MANAGER.lock().init()
}

/// Get current system uptime
pub fn uptime() -> SystemTime {
    SystemTime::now()
}

/// Get current uptime in milliseconds
pub fn uptime_ms() -> u64 {
    SYSTEM_UPTIME.load(Ordering::Acquire)
}

/// Get current uptime in microseconds
pub fn uptime_us() -> u64 {
    TIMER_MANAGER.lock().get_high_res_time()
}

/// Sleep for the specified duration
pub async fn sleep(duration: Duration) {
    let start = SystemTime::now();
    let target = start.add_duration(duration);

    while SystemTime::now() < target {
        crate::task::yield_now().await;
    }
}

/// Sleep for the specified number of milliseconds
pub async fn sleep_ms(ms: u64) {
    sleep(Duration::from_millis(ms)).await;
}

/// Sleep for the specified number of microseconds
pub async fn sleep_us(us: u64) {
    sleep(Duration::from_micros(us)).await;
}

/// Measure execution time of a closure
pub fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = SystemTime::now();
    let result = f();
    let end = SystemTime::now();

    let duration = end.duration_since(start).unwrap_or(Duration::from_millis(0));
    (result, duration)
}

/// Get high-resolution timestamp in microseconds
pub fn timestamp_us() -> u64 {
    TIMER_MANAGER.lock().get_high_res_time()
}

/// Get timer system statistics
pub fn get_timer_stats() -> TimerStats {
    TIMER_MANAGER.lock().get_stats()
}

/// Convert CPU cycles to microseconds (if calibrated)
pub fn cycles_to_us(cycles: u64) -> u64 {
    let cycles_per_ms = CPU_CYCLES_PER_MS.load(Ordering::Acquire);
    if cycles_per_ms > 0 {
        cycles / (cycles_per_ms / 1000)
    } else {
        0
    }
}

/// Convert microseconds to CPU cycles (if calibrated)
pub fn us_to_cycles(us: u64) -> u64 {
    let cycles_per_ms = CPU_CYCLES_PER_MS.load(Ordering::Acquire);
    if cycles_per_ms > 0 {
        us * (cycles_per_ms / 1000)
    } else {
        0
    }
}

/// Busy wait for specified microseconds (use sparingly)
pub fn busy_wait_us(us: u64) {
    let start_cycles = crate::arch::get_cpu_cycles();
    let target_cycles = us_to_cycles(us);

    if target_cycles > 0 {
        while crate::arch::get_cpu_cycles() - start_cycles < target_cycles {
            core::hint::spin_loop();
        }
    } else {
        // Fallback to simple loop if TSC not calibrated
        for _ in 0..(us * 100) {
            core::hint::spin_loop();
        }
    }
}

/// Busy wait for specified milliseconds (use sparingly)
pub fn busy_wait_ms(ms: u64) {
    busy_wait_us(ms * 1000);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_duration_creation() {
        let d1 = Duration::from_millis(1000);
        let d2 = Duration::from_secs(1);
        assert_eq!(d1, d2);
    }

    #[test_case]
    fn test_duration_conversions() {
        let duration = Duration::from_millis(5000);
        assert_eq!(duration.as_micros(), 5_000_000);
        assert_eq!(duration.as_nanos(), 5_000_000_000);
    }

    #[test_case]
    fn test_system_time() {
        let time1 = SystemTime::from_millis(1000);
        let time2 = SystemTime::from_millis(2000);

        let duration = time2.duration_since(time1);
        assert!(duration.is_some());
        assert_eq!(duration.unwrap().as_millis(), 1000);
    }

    #[test_case]
    fn test_timer_manager_creation() {
        let manager = TimerManager::new();
        assert!(!manager.pit_initialized);
        assert!(!manager.rtc_initialized);
        assert!(!manager.hpet_available);
    }
}
