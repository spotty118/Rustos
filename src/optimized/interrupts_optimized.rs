//! Optimized Interrupt Handlers for Critical Path Performance
//!
//! This module provides highly optimized interrupt handlers with:
//! - Minimal overhead for critical interrupts
//! - Inline assembly for fastest possible EOI
//! - Cache-friendly data structures
//! - Lock-free statistics updates

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::structures::idt::InterruptStackFrame;
use crate::interrupts::InterruptIndex;

/// Cache-aligned interrupt statistics for better performance
#[repr(align(64))] // Cache line alignment
pub struct OptimizedInterruptStats {
    pub timer_count: AtomicU64,
    pub keyboard_count: AtomicU64,
    pub serial_count: AtomicU64,
    pub exception_count: AtomicU64,
    pub page_fault_count: AtomicU64,
    pub spurious_count: AtomicU64,
    // Padding to fill cache line
    _padding: [u64; 2],
}

/// Global optimized statistics
static OPTIMIZED_STATS: OptimizedInterruptStats = OptimizedInterruptStats {
    timer_count: AtomicU64::new(0),
    keyboard_count: AtomicU64::new(0),
    serial_count: AtomicU64::new(0),
    exception_count: AtomicU64::new(0),
    page_fault_count: AtomicU64::new(0),
    spurious_count: AtomicU64::new(0),
    _padding: [0; 2],
};

/// Ultra-fast timer interrupt handler with minimal overhead
extern "x86-interrupt" fn optimized_timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Critical path: update stats with relaxed ordering for speed
    OPTIMIZED_STATS.timer_count.fetch_add(1, Ordering::Relaxed);

    unsafe {
        // Fastest possible EOI - inline assembly with no function calls
        if crate::apic::is_apic_available() {
            // Direct APIC EOI write
            core::arch::asm!(
                "mov dword ptr [0xfee000b0], 0",  // Write 0 to APIC EOI register
                options(preserves_flags, nostack)
            );
        } else {
            // Fast PIC EOI
            core::arch::asm!(
                "mov al, 0x20",
                "out 0x20, al",      // Send EOI to master PIC
                options(preserves_flags, nostack)
            );
        }

        // Notify scheduler with minimal overhead
        // This will be handled by a separate optimized scheduler tick
        crate::scheduler_optimized::fast_timer_tick();
    }
}

/// Optimized keyboard interrupt handler with reduced latency
extern "x86-interrupt" fn optimized_keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Fast scancode read with inline assembly
    let scancode: u8;
    unsafe {
        core::arch::asm!(
            "in al, 0x60",
            out("al") scancode,
            options(preserves_flags, nostack)
        );
    }

    // Update stats atomically
    OPTIMIZED_STATS.keyboard_count.fetch_add(1, Ordering::Relaxed);

    // Queue keyboard event to lock-free ring buffer for processing
    crate::keyboard_optimized::queue_scancode(scancode);

    unsafe {
        // Fast EOI
        if crate::apic::is_apic_available() {
            core::arch::asm!(
                "mov dword ptr [0xfee000b0], 0",
                options(preserves_flags, nostack)
            );
        } else {
            core::arch::asm!(
                "mov al, 0x20",
                "out 0x20, al",
                options(preserves_flags, nostack)
            );
        }
    }
}

/// High-performance page fault handler with fast path optimization
extern "x86-interrupt" fn optimized_page_fault_handler(
    _stack_frame: InterruptStackFrame,
    error_code: x86_64::structures::idt::PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;

    let fault_address = Cr2::read();

    // Update stats first
    OPTIMIZED_STATS.page_fault_count.fetch_add(1, Ordering::Relaxed);
    OPTIMIZED_STATS.exception_count.fetch_add(1, Ordering::Relaxed);

    // Fast path for common page faults
    if !error_code.contains(x86_64::structures::idt::PageFaultErrorCode::PROTECTION_VIOLATION) {
        // Try to handle page fault quickly
        if let Some(result) = crate::memory::try_fast_page_fault_handler(fault_address, error_code) {
            if result.is_ok() {
                return; // Fast path succeeded
            }
        }
    }

    // Slow path - full page fault handling
    crate::memory::handle_page_fault(fault_address, error_code.bits());
}

/// Lock-free interrupt statistics getter
pub fn get_optimized_stats() -> (u64, u64, u64, u64, u64, u64) {
    (
        OPTIMIZED_STATS.timer_count.load(Ordering::Relaxed),
        OPTIMIZED_STATS.keyboard_count.load(Ordering::Relaxed),
        OPTIMIZED_STATS.serial_count.load(Ordering::Relaxed),
        OPTIMIZED_STATS.exception_count.load(Ordering::Relaxed),
        OPTIMIZED_STATS.page_fault_count.load(Ordering::Relaxed),
        OPTIMIZED_STATS.spurious_count.load(Ordering::Relaxed),
    )
}

/// Reset optimized statistics
pub fn reset_optimized_stats() {
    OPTIMIZED_STATS.timer_count.store(0, Ordering::Relaxed);
    OPTIMIZED_STATS.keyboard_count.store(0, Ordering::Relaxed);
    OPTIMIZED_STATS.serial_count.store(0, Ordering::Relaxed);
    OPTIMIZED_STATS.exception_count.store(0, Ordering::Relaxed);
    OPTIMIZED_STATS.page_fault_count.store(0, Ordering::Relaxed);
    OPTIMIZED_STATS.spurious_count.store(0, Ordering::Relaxed);
}

/// Install optimized interrupt handlers
pub fn install_optimized_handlers() {
    use lazy_static::lazy_static;
    use x86_64::structures::idt::InterruptDescriptorTable;

    lazy_static! {
        static ref OPTIMIZED_IDT: InterruptDescriptorTable = {
            let mut idt = InterruptDescriptorTable::new();

            // Install optimized handlers
            idt[InterruptIndex::Timer.as_usize()].set_handler_fn(optimized_timer_interrupt_handler);
            idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(optimized_keyboard_interrupt_handler);
            idt.page_fault.set_handler_fn(optimized_page_fault_handler);

            idt
        };
    }

    OPTIMIZED_IDT.load();
}

/// Inline assembly optimized interrupt disable/enable
#[inline(always)]
pub fn optimized_without_interrupts<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let rflags: u64;
    unsafe {
        // Save flags and disable interrupts in one operation
        core::arch::asm!(
            "pushfq",
            "pop {rflags}",
            "cli",
            rflags = out(reg) rflags,
            options(preserves_flags)
        );
    }

    let result = f();

    // Restore interrupts if they were enabled
    if (rflags & 0x200) != 0 {
        unsafe {
            core::arch::asm!("sti", options(preserves_flags));
        }
    }

    result
}

/// Fast context switch detection for interrupt handlers
#[inline(always)]
pub fn should_preempt() -> bool {
    // Check if scheduler should preempt current task
    // This uses optimized scheduler state check
    crate::scheduler_optimized::should_preempt_current()
}