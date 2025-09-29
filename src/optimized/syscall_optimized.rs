//! Optimized System Call Implementation for RustOS
//!
//! This module provides high-performance system call handling with:
//! - Fast syscall dispatch using jump tables
//! - Minimal overhead parameter validation
//! - Cache-friendly syscall statistics
//! - Optimized user/kernel boundary crossing

use core::sync::atomic::{AtomicU64, Ordering};
use crate::syscall::{SyscallResult, SyscallError, SyscallContext};
use crate::scheduler::Pid;
use crate::process::ipc::Signal;
use x86_64::instructions::segmentation::Segment;

/// Cache-aligned syscall statistics for better performance
#[repr(align(64))]
pub struct OptimizedSyscallStats {
    pub total_calls: AtomicU64,
    pub successful_calls: AtomicU64,
    pub failed_calls: AtomicU64,
    pub avg_latency_ns: AtomicU64,
    // Per-syscall counters for hot syscalls
    pub read_calls: AtomicU64,
    pub write_calls: AtomicU64,
    pub open_calls: AtomicU64,
    pub close_calls: AtomicU64,
    pub yield_calls: AtomicU64,
    pub getpid_calls: AtomicU64,
    _padding: [u64; 2], // Pad to cache line
}

static OPTIMIZED_SYSCALL_STATS: OptimizedSyscallStats = OptimizedSyscallStats {
    total_calls: AtomicU64::new(0),
    successful_calls: AtomicU64::new(0),
    failed_calls: AtomicU64::new(0),
    avg_latency_ns: AtomicU64::new(0),
    read_calls: AtomicU64::new(0),
    write_calls: AtomicU64::new(0),
    open_calls: AtomicU64::new(0),
    close_calls: AtomicU64::new(0),
    yield_calls: AtomicU64::new(0),
    getpid_calls: AtomicU64::new(0),
    _padding: [0; 2],
};

/// Fast syscall dispatch table - avoids match statements for better performance
type SyscallHandler = fn(&SyscallContext) -> SyscallResult;

static SYSCALL_TABLE: [Option<SyscallHandler>; 64] = {
    let mut table = [None; 64];
    table[0] = Some(optimized_sys_exit as SyscallHandler);         // Exit
    table[1] = Some(optimized_sys_fork as SyscallHandler);         // Fork
    table[4] = Some(optimized_sys_getpid as SyscallHandler);       // GetPid
    table[6] = Some(optimized_sys_kill as SyscallHandler);         // Kill
    table[10] = Some(optimized_sys_open as SyscallHandler);        // Open
    table[11] = Some(optimized_sys_close as SyscallHandler);       // Close
    table[12] = Some(optimized_sys_read as SyscallHandler);        // Read
    table[13] = Some(optimized_sys_write as SyscallHandler);       // Write
    table[22] = Some(optimized_sys_brk as SyscallHandler);         // Brk
    table[40] = Some(optimized_sys_sleep as SyscallHandler);       // Sleep
    table[41] = Some(optimized_sys_gettime as SyscallHandler);     // GetTime
    table[44] = Some(optimized_sys_yield);       // Yield
    table
};

/// Optimized syscall dispatcher with minimal overhead
pub fn optimized_dispatch_syscall(context: &SyscallContext) -> SyscallResult {
    // Record start time for latency measurement
    let start_time = crate::performance_monitor::read_tsc();

    // Update stats (relaxed ordering for performance)
    OPTIMIZED_SYSCALL_STATS.total_calls.fetch_add(1, Ordering::Relaxed);

    // Fast table lookup instead of match statement
    let syscall_num = context.syscall_num as u64;
    let result = if syscall_num < 64 {
        if let Some(handler) = SYSCALL_TABLE[syscall_num as usize] {
            handler(context)
        } else {
            Err(SyscallError::NotSupported)
        }
    } else {
        Err(SyscallError::InvalidSyscall)
    };

    // Update stats based on result
    match result {
        Ok(_) => {
            OPTIMIZED_SYSCALL_STATS.successful_calls.fetch_add(1, Ordering::Relaxed);
        }
        Err(_) => {
            OPTIMIZED_SYSCALL_STATS.failed_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Update latency measurement
    let end_time = crate::performance_monitor::read_tsc();
    let latency_cycles = end_time - start_time;
    // Convert cycles to nanoseconds (approximate)
    let latency_ns = latency_cycles / 3; // Assuming 3GHz CPU

    // Simple exponential moving average for latency
    let current_avg = OPTIMIZED_SYSCALL_STATS.avg_latency_ns.load(Ordering::Relaxed);
    let new_avg = (current_avg * 7 + latency_ns) / 8; // 7/8 weight to old average
    OPTIMIZED_SYSCALL_STATS.avg_latency_ns.store(new_avg, Ordering::Relaxed);

    result
}

/// Fast parameter validation for common patterns
#[inline(always)]
fn validate_user_buffer(ptr: u64, len: u64) -> Result<(), SyscallError> {
    // Fast checks without function calls
    if ptr == 0 && len > 0 {
        return Err(SyscallError::InvalidArgument);
    }
    if ptr >= 0x8000_0000_0000 {
        return Err(SyscallError::InvalidArgument);
    }
    if ptr.saturating_add(len) < ptr {
        return Err(SyscallError::InvalidArgument);
    }
    Ok(())
}

// Optimized syscall implementations

fn optimized_sys_exit(_context: &SyscallContext) -> SyscallResult {
    // Fast exit - minimal processing
    crate::process::terminate_current_process();
    Ok(0)
}

fn optimized_sys_fork(_context: &SyscallContext) -> SyscallResult {
    // Fork not yet implemented
    Err(SyscallError::NotSupported)
}

fn optimized_sys_getpid(_context: &SyscallContext) -> SyscallResult {
    // Fastest possible getpid - no validation needed
    OPTIMIZED_SYSCALL_STATS.getpid_calls.fetch_add(1, Ordering::Relaxed);
    Ok(crate::process::current_pid() as u64)
}

fn optimized_sys_kill(context: &SyscallContext) -> SyscallResult {
    let target_pid = context.args[0] as Pid;
    let signal_num = context.args[1] as i32;

    // Convert signal number to Signal enum
    let signal = match signal_num {
        1 => Signal::SIGHUP,
        2 => Signal::SIGINT,
        3 => Signal::SIGQUIT,
        4 => Signal::SIGILL,
        5 => Signal::SIGTRAP,
        6 => Signal::SIGABRT,
        7 => Signal::SIGBUS,
        8 => Signal::SIGFPE,
        9 => Signal::SIGKILL,
        10 => Signal::SIGUSR1,
        11 => Signal::SIGSEGV,
        12 => Signal::SIGUSR2,
        13 => Signal::SIGPIPE,
        14 => Signal::SIGALRM,
        15 => Signal::SIGTERM,
        17 => Signal::SIGCHLD,
        18 => Signal::SIGCONT,
        19 => Signal::SIGSTOP,
        20 => Signal::SIGTSTP,
        _ => return Err(SyscallError::InvalidArgument),
    };

    // Basic validation
    if target_pid == 0 {
        return Err(SyscallError::InvalidArgument);
    }

    // Send signal
    crate::process::send_signal(target_pid, signal, crate::process::current_pid())
        .map(|_| 0)
        .map_err(|_| SyscallError::NotFound)
}

fn optimized_sys_open(context: &SyscallContext) -> SyscallResult {
    OPTIMIZED_SYSCALL_STATS.open_calls.fetch_add(1, Ordering::Relaxed);

    let pathname = context.args[0];
    let flags = context.args[1] as u32;

    // Fast validation
    validate_user_buffer(pathname, 4096)?;

    // For now, return a dummy file descriptor
    Ok(3)
}

fn optimized_sys_close(context: &SyscallContext) -> SyscallResult {
    OPTIMIZED_SYSCALL_STATS.close_calls.fetch_add(1, Ordering::Relaxed);

    let fd = context.args[0] as i32;

    // Fast validation
    if fd < 0 || fd <= 2 {
        return Err(SyscallError::BadFileDescriptor);
    }

    // Close file descriptor
    Ok(0)
}

fn optimized_sys_read(context: &SyscallContext) -> SyscallResult {
    OPTIMIZED_SYSCALL_STATS.read_calls.fetch_add(1, Ordering::Relaxed);

    let fd = context.args[0] as i32;
    let buf = context.args[1];
    let count = context.args[2];

    // Fast validation
    if fd < 0 {
        return Err(SyscallError::BadFileDescriptor);
    }
    validate_user_buffer(buf, count)?;

    // Handle stdin specially
    if fd == 0 {
        return Ok(0); // No data available
    }

    // For other descriptors, return empty read for now
    Ok(0)
}

fn optimized_sys_write(context: &SyscallContext) -> SyscallResult {
    OPTIMIZED_SYSCALL_STATS.write_calls.fetch_add(1, Ordering::Relaxed);

    let fd = context.args[0] as i32;
    let buf = context.args[1];
    let count = context.args[2];

    // Fast validation
    if fd < 0 {
        return Err(SyscallError::BadFileDescriptor);
    }
    validate_user_buffer(buf, count)?;

    // Handle stdout/stderr
    if fd == 1 || fd == 2 {
        // TODO: Fast console write
        return Ok(count);
    }

    // For other descriptors
    Ok(count)
}

fn optimized_sys_brk(context: &SyscallContext) -> SyscallResult {
    let addr = context.args[0];

    // Fast heap management
    crate::memory::adjust_heap(addr as usize)
        .map(|new_addr| new_addr as u64)
        .map_err(|_| SyscallError::OutOfMemory)
}

fn optimized_sys_sleep(context: &SyscallContext) -> SyscallResult {
    let microseconds = context.args[0];

    // Fast sleep
    if microseconds > 0 {
        let milliseconds = microseconds / 1000;
        crate::time::sleep_ms(milliseconds);
    }

    Ok(0)
}

fn optimized_sys_gettime(_context: &SyscallContext) -> SyscallResult {
    // Fast time read
    Ok(crate::time::uptime_us())
}

fn optimized_sys_yield(_context: &SyscallContext) -> SyscallResult {
    OPTIMIZED_SYSCALL_STATS.yield_calls.fetch_add(1, Ordering::Relaxed);

    // Fast yield - minimal scheduler interaction
    crate::scheduler::yield_cpu();
    Ok(0)
}

/// Get optimized syscall statistics
pub fn get_optimized_syscall_stats() -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    (
        OPTIMIZED_SYSCALL_STATS.total_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.successful_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.failed_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.avg_latency_ns.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.read_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.write_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.open_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.close_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.yield_calls.load(Ordering::Relaxed),
        OPTIMIZED_SYSCALL_STATS.getpid_calls.load(Ordering::Relaxed),
    )
}

/// Reset optimized syscall statistics
pub fn reset_optimized_syscall_stats() {
    OPTIMIZED_SYSCALL_STATS.total_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.successful_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.failed_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.avg_latency_ns.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.read_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.write_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.open_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.close_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.yield_calls.store(0, Ordering::Relaxed);
    OPTIMIZED_SYSCALL_STATS.getpid_calls.store(0, Ordering::Relaxed);
}

/// Ultra-fast syscall entry point with inline assembly
pub unsafe extern "C" fn optimized_syscall_entry() {
    // This would be the actual syscall entry point
    // Using SYSCALL/SYSRET instructions for fastest possible transition
    core::arch::asm!(
        // Save user state
        "swapgs",
        "mov gs:0, rsp",    // Load kernel stack
        "push rcx",         // Save user RIP
        "push r11",         // Save user RFLAGS

        // Call dispatcher
        "call {dispatcher}",

        // Restore user state
        "pop r11",          // Restore user RFLAGS
        "pop rcx",          // Restore user RIP
        "swapgs",
        "sysretq",

        dispatcher = sym optimized_syscall_dispatcher_asm,
        options(noreturn)
    );
}

/// Assembly dispatcher for maximum performance
unsafe extern "C" fn optimized_syscall_dispatcher_asm() {
    // This would handle the actual syscall dispatch in assembly
    // For now, we'll use a placeholder
}

/// Install optimized syscall handling
pub fn install_optimized_syscalls() {
    // Configure SYSCALL/SYSRET MSRs for fastest user/kernel transitions
    unsafe {
        // Enable SYSCALL/SYSRET
        let efer = x86_64::registers::model_specific::Efer::read();
        x86_64::registers::model_specific::Efer::write(
            efer | x86_64::registers::model_specific::EferFlags::SYSTEM_CALL_EXTENSIONS
        );

        // Set STAR register (CS/SS selectors)
        use x86_64::structures::gdt::SegmentSelector;
        x86_64::registers::model_specific::Star::write(
            x86_64::registers::segmentation::CS::get_reg(), // Kernel CS
            SegmentSelector(0x28), // Kernel SS (typical value)
            SegmentSelector(0x33), // User CS base
            SegmentSelector(0x2B), // User SS
        ).unwrap();

        // Set LSTAR register (syscall entry point)
        x86_64::registers::model_specific::LStar::write(
            x86_64::VirtAddr::new(optimized_syscall_entry as u64)
        );

        // Set SFMASK register (flags to clear)
        x86_64::registers::model_specific::SFMask::write(
            x86_64::registers::rflags::RFlags::INTERRUPT_FLAG
        );
    }
}