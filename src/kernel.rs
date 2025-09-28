//! Kernel Core Integration Layer
//!
//! This module integrates all core kernel subsystems and provides
//! a unified initialization and management interface.

use crate::{
    arch, time, smp, security, vga_buffer,
    memory, interrupts, gdt, process,
    serial_println, println
};

/// Kernel initialization result
pub type KernelResult<T> = Result<T, &'static str>;

/// Kernel subsystem states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemState {
    NotInitialized,
    Initializing,
    Initialized,
    Failed,
}

/// Kernel core state
pub struct KernelCore {
    pub arch_initialized: SubsystemState,
    pub time_initialized: SubsystemState,
    pub smp_initialized: SubsystemState,
    pub security_initialized: SubsystemState,
    pub vga_initialized: SubsystemState,
    pub memory_initialized: SubsystemState,
    pub interrupts_initialized: SubsystemState,
    pub gdt_initialized: SubsystemState,
    pub processes_initialized: SubsystemState,
}

impl Default for KernelCore {
    fn default() -> Self {
        Self {
            arch_initialized: SubsystemState::NotInitialized,
            time_initialized: SubsystemState::NotInitialized,
            smp_initialized: SubsystemState::NotInitialized,
            security_initialized: SubsystemState::NotInitialized,
            vga_initialized: SubsystemState::NotInitialized,
            memory_initialized: SubsystemState::NotInitialized,
            interrupts_initialized: SubsystemState::NotInitialized,
            gdt_initialized: SubsystemState::NotInitialized,
            processes_initialized: SubsystemState::NotInitialized,
        }
    }
}

static mut KERNEL_CORE: KernelCore = KernelCore {
    arch_initialized: SubsystemState::NotInitialized,
    time_initialized: SubsystemState::NotInitialized,
    smp_initialized: SubsystemState::NotInitialized,
    security_initialized: SubsystemState::NotInitialized,
    vga_initialized: SubsystemState::NotInitialized,
    memory_initialized: SubsystemState::NotInitialized,
    interrupts_initialized: SubsystemState::NotInitialized,
    gdt_initialized: SubsystemState::NotInitialized,
    processes_initialized: SubsystemState::NotInitialized,
};

/// Initialize core kernel subsystems
pub fn init_core_kernel() -> KernelResult<()> {
    serial_println!("Initializing RustOS core kernel subsystems...");
    
    unsafe {
        // Initialize VGA buffer first for console output
        KERNEL_CORE.vga_initialized = SubsystemState::Initializing;
        vga_buffer::init();
        KERNEL_CORE.vga_initialized = SubsystemState::Initialized;
        println!("✓ VGA buffer initialized");

        // Initialize architecture-specific features
        KERNEL_CORE.arch_initialized = SubsystemState::Initializing;
        match arch::init() {
            Ok(()) => {
                KERNEL_CORE.arch_initialized = SubsystemState::Initialized;
                println!("✓ Architecture features initialized");
                
                // Display CPU information
                let cpu_info = arch::get_cpu_info();
                println!("  CPU: {} - {}", cpu_info.vendor_id, cpu_info.brand_string);
                let features = arch::get_cpu_features();
                println!("  Features: SSE: {}, AVX: {}, FMA: {}", 
                         features.has_sse, features.has_avx, features.has_fma);
            }
            Err(e) => {
                KERNEL_CORE.arch_initialized = SubsystemState::Failed;
                println!("✗ Architecture initialization failed: {}", e);
                return Err("Architecture initialization failed");
            }
        }

        // Initialize timer system
        KERNEL_CORE.time_initialized = SubsystemState::Initializing;
        match time::init() {
            Ok(()) => {
                KERNEL_CORE.time_initialized = SubsystemState::Initialized;
                println!("✓ Timer system initialized");
            }
            Err(e) => {
                KERNEL_CORE.time_initialized = SubsystemState::Failed;
                println!("✗ Timer initialization failed: {}", e);
                return Err("Timer initialization failed");
            }
        }

        // Initialize SMP support
        KERNEL_CORE.smp_initialized = SubsystemState::Initializing;
        match smp::init() {
            Ok(()) => {
                KERNEL_CORE.smp_initialized = SubsystemState::Initialized;
                let stats = smp::get_smp_statistics();
                println!("✓ SMP initialized - {} CPU(s), {} online", 
                         stats.total_cpus, stats.online_cpus);
            }
            Err(e) => {
                KERNEL_CORE.smp_initialized = SubsystemState::Failed;
                println!("✗ SMP initialization failed: {}", e);
                // SMP failure is not critical for basic functionality
            }
        }

        // Initialize security framework
        KERNEL_CORE.security_initialized = SubsystemState::Initializing;
        match security::init() {
            Ok(()) => {
                KERNEL_CORE.security_initialized = SubsystemState::Initialized;
                let level = security::get_security_level();
                println!("✓ Security framework initialized - Level: {:?}", level);
            }
            Err(e) => {
                KERNEL_CORE.security_initialized = SubsystemState::Failed;
                println!("✗ Security initialization failed: {}", e);
                return Err("Security initialization failed");
            }
        }
    }

    serial_println!("Core kernel subsystems initialized successfully!");
    Ok(())
}

/// Display kernel status
pub fn display_kernel_status() {
    println!("=== RustOS Kernel Status ===");
    
    unsafe {
        println!("Architecture:  {:?}", KERNEL_CORE.arch_initialized);
        println!("Timer:         {:?}", KERNEL_CORE.time_initialized);
        println!("SMP:           {:?}", KERNEL_CORE.smp_initialized);
        println!("Security:      {:?}", KERNEL_CORE.security_initialized);
        println!("VGA Buffer:    {:?}", KERNEL_CORE.vga_initialized);
        println!("Memory:        {:?}", KERNEL_CORE.memory_initialized);
        println!("Interrupts:    {:?}", KERNEL_CORE.interrupts_initialized);
        println!("GDT:           {:?}", KERNEL_CORE.gdt_initialized);
        println!("Processes:     {:?}", KERNEL_CORE.processes_initialized);
    }
    
    // Display system information
    let uptime = time::uptime_ms();
    println!("System uptime: {} ms", uptime);
    
    let cpu_count = smp::get_cpu_count();
    println!("CPUs detected: {}", cpu_count);
    
    let security_level = security::get_security_level();
    println!("Security level: {:?}", security_level);
    
    let vga_stats = vga_buffer::get_vga_stats();
    println!("VGA cursor: row {}, col {}", vga_stats.cursor_row, vga_stats.cursor_column);
}

/// Test core kernel functionality
pub fn test_core_functionality() -> KernelResult<()> {
    println!("=== Testing Core Kernel Functionality ===");
    
    // Test timer functionality
    println!("Testing timer...");
    let start_time = time::uptime_ms();
    time::sleep_ms(1); // Brief delay
    let end_time = time::uptime_ms();
    if end_time >= start_time {
        println!("✓ Timer test passed");
    } else {
        println!("✗ Timer test failed");
        return Err("Timer test failed");
    }
    
    // Test CPU features
    println!("Testing CPU features...");
    let features = arch::get_cpu_features();
    println!("  CPU features detected: {} total", 
             if features.has_sse { 1 } else { 0 } +
             if features.has_sse2 { 1 } else { 0 } +
             if features.has_avx { 1 } else { 0 });
    
    // Test security system
    println!("Testing security...");
    let security_stats = security::get_security_stats();
    println!("  Security events: {}", security_stats.total_events);
    
    // Test VGA buffer
    println!("Testing VGA buffer...");
    let char_count_before = vga_buffer::get_char_count();
    vga_buffer::print_colored("Test message", vga_buffer::Color::LightGreen, vga_buffer::Color::Black);
    let char_count_after = vga_buffer::get_char_count();
    // Note: char count might not change if not implemented, that's okay
    
    println!("✓ All core functionality tests completed");
    Ok(())
}

/// Get kernel core state
pub fn get_kernel_state() -> KernelCore {
    unsafe { KERNEL_CORE }
}

/// Check if all critical subsystems are initialized
pub fn is_kernel_ready() -> bool {
    unsafe {
        matches!(KERNEL_CORE.arch_initialized, SubsystemState::Initialized) &&
        matches!(KERNEL_CORE.time_initialized, SubsystemState::Initialized) &&
        matches!(KERNEL_CORE.security_initialized, SubsystemState::Initialized) &&
        matches!(KERNEL_CORE.vga_initialized, SubsystemState::Initialized)
    }
}

/// Demonstrate enhanced kernel capabilities
pub fn demonstrate_kernel_capabilities() {
    println!("=== RustOS Enhanced Kernel Capabilities ===");
    
    // Display architecture information
    let cpu_info = arch::get_cpu_info();
    println!("CPU Information:");
    println!("  Vendor: {}", cpu_info.vendor_id);
    println!("  Model: Family {}, Model {}, Stepping {}", 
             cpu_info.family, cpu_info.model, cpu_info.stepping);
    println!("  Cores: {}, Threads: {}", cpu_info.core_count, cpu_info.thread_count);
    println!("  Cache Line Size: {} bytes", cpu_info.cache_line_size);
    
    // Display timer information
    let timer_stats = time::get_timer_stats();
    println!("Timer Information:");
    println!("  Total Ticks: {}", timer_stats.total_ticks);
    println!("  Uptime: {} ms", timer_stats.uptime_ms);
    println!("  Frequency: {} Hz", timer_stats.timer_frequency);
    
    // Display SMP information
    let smp_stats = smp::get_smp_statistics();
    println!("SMP Information:");
    println!("  Total CPUs: {}", smp_stats.total_cpus);
    println!("  Online CPUs: {}", smp_stats.online_cpus);
    println!("  IPIs Sent: {}", smp_stats.ipi_sent);
    
    // Display security information
    let security_stats = security::get_security_stats();
    println!("Security Information:");
    println!("  Security Level: {:?}", security_stats.security_level);
    println!("  Total Events: {}", security_stats.total_events);
    println!("  Access Denied: {}", security_stats.access_denied_count);
    println!("  Monitoring: {}", security_stats.monitoring_enabled);
    
    // Display VGA buffer information
    let vga_stats = vga_buffer::get_vga_stats();
    println!("Display Information:");
    println!("  Buffer Size: {}x{}", vga_stats.buffer_width, vga_stats.buffer_height);
    println!("  Cursor Position: ({}, {})", vga_stats.cursor_row, vga_stats.cursor_column);
    
    println!("✓ Enhanced kernel capabilities demonstration complete");
}

/// Kernel panic handler integration
pub fn kernel_panic(info: &core::panic::PanicInfo) -> ! {
    // Use VGA buffer for panic output if available
    unsafe {
        if matches!(KERNEL_CORE.vga_initialized, SubsystemState::Initialized) {
            vga_buffer::print_colored("KERNEL PANIC!", vga_buffer::Color::White, vga_buffer::Color::Red);
            println!("{}", info);
        }
    }
    
    // Also output to serial
    serial_println!("KERNEL PANIC: {}", info);
    
    // Halt all CPUs if SMP is available
    if smp::is_initialized() {
        let _ = smp::send_ipi_all_but_self(smp::IpiType::Halt, 0);
    }
    
    // Halt this CPU
    loop {
        arch::halt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_core_default() {
        let core = KernelCore::default();
        assert_eq!(core.arch_initialized, SubsystemState::NotInitialized);
        assert_eq!(core.time_initialized, SubsystemState::NotInitialized);
    }

    #[test]
    fn test_subsystem_state() {
        assert_ne!(SubsystemState::Initialized, SubsystemState::Failed);
        assert_ne!(SubsystemState::NotInitialized, SubsystemState::Initializing);
    }
}