//! Process Management Integration
//!
//! This module provides integration between the process management system
//! and other kernel subsystems like memory management and interrupts.

use super::{Pid, get_process_manager};
use crate::println;

/// Process management integration with timer interrupts
pub struct TimerIntegration {
    /// Time slice counter
    time_slice_counter: u32,
    /// Scheduling frequency (ticks per schedule)
    schedule_frequency: u32,
}

impl TimerIntegration {
    /// Create new timer integration
    pub const fn new() -> Self {
        Self {
            time_slice_counter: 0,
            schedule_frequency: 10, // Schedule every 10 timer ticks
        }
    }

    /// Handle timer interrupt for process scheduling
    pub fn handle_timer_interrupt(&mut self) -> Result<Option<Pid>, &'static str> {
        // Increment system time
        super::tick_system_time();

        // Update scheduler tick
        let process_manager = get_process_manager();
        {
            let mut scheduler = process_manager.scheduler.lock();
            scheduler.tick();
        }

        self.time_slice_counter += 1;

        // Check if we should perform scheduling
        if self.time_slice_counter >= self.schedule_frequency {
            self.time_slice_counter = 0;

            // Trigger process scheduling
            process_manager.schedule()
        } else {
            Ok(None)
        }
    }

    /// Set scheduling frequency
    pub fn set_schedule_frequency(&mut self, frequency: u32) {
        self.schedule_frequency = frequency.max(1);
    }

    /// Get current time slice counter
    pub fn get_time_slice_counter(&self) -> u32 {
        self.time_slice_counter
    }
}

/// Process management integration with memory management
pub struct MemoryIntegration;

impl MemoryIntegration {
    /// Handle page fault for process
    pub fn handle_page_fault(pid: Pid, fault_address: u64, error_code: u64) -> Result<(), &'static str> {
        let process_manager = get_process_manager();

        // Get process information
        let process = process_manager.get_process(pid)
            .ok_or("Process not found")?;

        // Check if fault address is within process memory space
        if fault_address >= process.memory.vm_start &&
           fault_address < process.memory.vm_start + process.memory.vm_size {

            // Handle different types of page faults
            if (error_code & 0x1) == 0 {
                // Page not present - allocate page
                Self::allocate_page_for_process(pid, fault_address)
            } else if (error_code & 0x2) != 0 {
                // Write to read-only page
                Self::handle_cow_page(pid, fault_address)
            } else {
                Err("Invalid page fault")
            }
        } else {
            // Segmentation fault - terminate process
            process_manager.terminate_process(pid, -11) // SIGSEGV
        }
    }

    /// Allocate a new page for process
    fn allocate_page_for_process(pid: Pid, fault_address: u64) -> Result<(), &'static str> {
        // In a real implementation, this would:
        // 1. Allocate a physical page
        // 2. Map it to the virtual address
        // 3. Update page tables
        // 4. Set appropriate permissions

        // For now, just indicate success
        println!("Allocated page for PID {} at address 0x{:x}", pid, fault_address);
        Ok(())
    }

    /// Handle copy-on-write page fault
    fn handle_cow_page(pid: Pid, fault_address: u64) -> Result<(), &'static str> {
        // In a real implementation, this would:
        // 1. Check if page is marked as COW
        // 2. Allocate new physical page
        // 3. Copy original page content
        // 4. Update page table with new mapping
        // 5. Mark page as writable

        println!("Handled COW page for PID {} at address 0x{:x}", pid, fault_address);
        Ok(())
    }

    /// Set up memory space for new process
    pub fn setup_process_memory(pid: Pid, size: u64) -> Result<u64, &'static str> {
        // In a real implementation, this would:
        // 1. Allocate virtual address space
        // 2. Create page tables
        // 3. Set up heap and stack regions
        // 4. Map essential pages (code, data)

        // For now, return a mock virtual address
        let base_address = 0x400000 + (pid as u64 * 0x100000); // 1MB per process
        println!("Set up memory space for PID {} at 0x{:x} (size: {} bytes)", pid, base_address, size);
        Ok(base_address)
    }

    /// Clean up memory space for terminated process
    pub fn cleanup_process_memory(pid: Pid) -> Result<(), &'static str> {
        // In a real implementation, this would:
        // 1. Free all allocated pages
        // 2. Remove page table entries
        // 3. Free page directory
        // 4. Return memory to allocator

        println!("Cleaned up memory space for PID {}", pid);
        Ok(())
    }
}

/// Process management integration with interrupt handling
pub struct InterruptIntegration;

impl InterruptIntegration {
    /// Handle system call interrupt
    pub fn handle_syscall_interrupt(
        syscall_number: u64,
        args: &[u64],
    ) -> Result<u64, &'static str> {
        let process_manager = get_process_manager();
        process_manager.handle_syscall(syscall_number, args)
    }

    /// Handle keyboard interrupt for process input
    pub fn handle_keyboard_interrupt(scancode: u8) -> Result<(), &'static str> {
        // In a real implementation, this would:
        // 1. Convert scancode to character
        // 2. Find processes waiting for input
        // 3. Deliver input to appropriate process
        // 4. Wake up blocked processes

        println!("Keyboard input: scancode 0x{:02x}", scancode);
        Ok(())
    }

    /// Handle signal delivery to process
    pub fn deliver_signal(pid: Pid, signal: u32) -> Result<(), &'static str> {
        let process_manager = get_process_manager();

        // Check if process exists
        let _process = process_manager.get_process(pid)
            .ok_or("Process not found")?;

        // In a real implementation, this would:
        // 1. Check if process has signal handler
        // 2. If not, apply default action
        // 3. If yes, set up signal delivery
        // 4. Modify process context to call handler

        match signal {
            9 => { // SIGKILL
                process_manager.terminate_process(pid, -9)?;
                println!("Terminated process {} with SIGKILL", pid);
            }
            15 => { // SIGTERM
                process_manager.terminate_process(pid, -15)?;
                println!("Terminated process {} with SIGTERM", pid);
            }
            _ => {
                println!("Delivered signal {} to process {}", signal, pid);
            }
        }

        Ok(())
    }
}

/// Central integration manager
pub struct ProcessIntegration {
    timer_integration: TimerIntegration,
}

impl ProcessIntegration {
    /// Create new process integration manager
    pub const fn new() -> Self {
        Self {
            timer_integration: TimerIntegration::new(),
        }
    }

    /// Initialize integration with other kernel systems
    pub fn init(&mut self) -> Result<(), &'static str> {
        // Initialize synchronization system
        super::sync::init()?;

        println!("Process integration initialized");
        Ok(())
    }

    /// Handle timer interrupt
    pub fn handle_timer(&mut self) -> Result<Option<Pid>, &'static str> {
        self.timer_integration.handle_timer_interrupt()
    }

    /// Handle page fault
    pub fn handle_page_fault(&self, pid: Pid, fault_address: u64, error_code: u64) -> Result<(), &'static str> {
        MemoryIntegration::handle_page_fault(pid, fault_address, error_code)
    }

    /// Handle system call
    pub fn handle_syscall(&self, syscall_number: u64, args: &[u64]) -> Result<u64, &'static str> {
        InterruptIntegration::handle_syscall_interrupt(syscall_number, args)
    }

    /// Handle keyboard input
    pub fn handle_keyboard(&self, scancode: u8) -> Result<(), &'static str> {
        InterruptIntegration::handle_keyboard_interrupt(scancode)
    }

    /// Deliver signal to process
    pub fn deliver_signal(&self, pid: Pid, signal: u32) -> Result<(), &'static str> {
        InterruptIntegration::deliver_signal(pid, signal)
    }

    /// Set timer scheduling frequency
    pub fn set_schedule_frequency(&mut self, frequency: u32) {
        self.timer_integration.set_schedule_frequency(frequency);
    }

    /// Get integration statistics
    pub fn get_stats(&self) -> IntegrationStats {
        IntegrationStats {
            time_slice_counter: self.timer_integration.get_time_slice_counter(),
            schedule_frequency: self.timer_integration.schedule_frequency,
            sync_stats: super::sync::get_sync_manager().get_stats(),
        }
    }
}

/// Integration statistics
#[derive(Debug)]
pub struct IntegrationStats {
    pub time_slice_counter: u32,
    pub schedule_frequency: u32,
    pub sync_stats: super::sync::SyncStats,
}

/// Global process integration manager
static mut PROCESS_INTEGRATION: ProcessIntegration = ProcessIntegration::new();

/// Get the global process integration manager
pub fn get_integration_manager() -> &'static mut ProcessIntegration {
    unsafe { &mut PROCESS_INTEGRATION }
}

/// Initialize process integration
pub fn init() -> Result<(), &'static str> {
    let integration = get_integration_manager();
    integration.init()
}

/// Timer interrupt handler (to be called from interrupt handler)
pub fn timer_interrupt_handler() -> Option<Pid> {
    let integration = get_integration_manager();
    integration.handle_timer().unwrap_or(None)
}

/// Page fault handler (to be called from interrupt handler)
pub fn page_fault_handler(fault_address: u64, error_code: u64) -> Result<(), &'static str> {
    let process_manager = get_process_manager();
    let current_pid = process_manager.current_process();

    let integration = get_integration_manager();
    integration.handle_page_fault(current_pid, fault_address, error_code)
}

/// System call handler (to be called from interrupt handler)
pub fn syscall_interrupt_handler(syscall_number: u64, args: &[u64]) -> Result<u64, &'static str> {
    let integration = get_integration_manager();
    integration.handle_syscall(syscall_number, args)
}

/// Keyboard interrupt handler (to be called from interrupt handler)
pub fn keyboard_interrupt_handler(scancode: u8) -> Result<(), &'static str> {
    let integration = get_integration_manager();
    integration.handle_keyboard(scancode)
}
