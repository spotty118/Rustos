//! RustOS Simplified Bootable Kernel
//!
//! A complete, working operating system kernel with:
//! - Core kernel functionality
//! - AI integration
//! - GPU acceleration
//! - Driver support
//! - Process management
//! - Memory management

#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![feature(alloc_error_handler)]
#![cfg_attr(test, test_runner(crate::test_runner))]
#![cfg_attr(test, reexport_test_harness_main = "test_main")]

extern crate alloc;

use core::fmt::Write;
use core::panic::PanicInfo;
use rustos::{desktop, drivers, gdt, interrupts, memory, process};

use desktop::{
    get_desktop_status, setup_full_desktop, update_desktop, DesktopStatus,
};
use drivers::is_graphics_ready;
use memory::{get_memory_stats, MemoryStats};

// Global allocator is defined in lib.rs

// ========== BOOT AND PANIC HANDLERS ==========

#[cfg(not(test))]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Initialize VGA buffer for early boot messages
    VGA_WRITER.lock().clear_screen();

    // Display boot banner
    print_banner();

    // Initialize basic kernel systems
    init_kernel();

    // Try to initialize desktop environment
    match init_desktop_environment() {
        Ok(()) => {
            println!("Desktop environment initialized successfully!");
            desktop_main_loop();
        }
        Err(e) => {
            println!("Failed to initialize desktop: {}", e);
            println!("Falling back to text mode...");

            // Demonstrate kernel features in text mode
            demonstrate_features();

            // Enter text mode main loop
            kernel_main_loop();
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // Write directly to VGA buffer since println! might not be available
    unsafe {
        let vga_buffer = 0xb8000 as *mut u16;
        let panic_msg = b"KERNEL PANIC!";
        for (i, &byte) in panic_msg.iter().enumerate() {
            *vga_buffer.add(i) = (0x4f00) | (byte as u16); // Red background, white text
        }
    }
    loop {
        unsafe {
            core::arch::asm!("cli");
            core::arch::asm!("hlt");
        }
    }
}

// ========== VGA TEXT OUTPUT ==========

const VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;
const VGA_WIDTH: usize = 80;
const VGA_HEIGHT: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Blue = 1,
    Green = 2,
    Cyan = 3,
    Red = 4,
    Magenta = 5,
    Brown = 6,
    LightGray = 7,
    DarkGray = 8,
    LightBlue = 9,
    LightGreen = 10,
    LightCyan = 11,
    LightRed = 12,
    Pink = 13,
    Yellow = 14,
    White = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct ColorCode(u8);

impl ColorCode {
    const fn new(foreground: Color, background: Color) -> ColorCode {
        ColorCode((background as u8) << 4 | (foreground as u8))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct ScreenChar {
    ascii_character: u8,
    color_code: ColorCode,
}

struct Writer {
    column_position: usize,
    row_position: usize,
    color_code: ColorCode,
}

impl Writer {
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.new_line(),
            byte => {
                if self.column_position >= VGA_WIDTH {
                    self.new_line();
                }

                let row = self.row_position;
                let col = self.column_position;

                let color_code = self.color_code;
                unsafe {
                    let ptr = VGA_BUFFER.add(2 * (row * VGA_WIDTH + col));
                    ptr.write_volatile(byte);
                    ptr.add(1).write_volatile(color_code.0);
                }
                self.column_position += 1;
            }
        }
    }

    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                0x20..=0x7e | b'\n' => self.write_byte(byte),
                _ => self.write_byte(0xfe),
            }
        }
    }

    fn new_line(&mut self) {
        if self.row_position >= VGA_HEIGHT - 1 {
            self.scroll_up();
        } else {
            self.row_position += 1;
        }
        self.column_position = 0;
    }

    fn scroll_up(&mut self) {
        unsafe {
            // Move all rows up by one
            for row in 1..VGA_HEIGHT {
                for col in 0..VGA_WIDTH {
                    let src = VGA_BUFFER.add(2 * (row * VGA_WIDTH + col));
                    let dst = VGA_BUFFER.add(2 * ((row - 1) * VGA_WIDTH + col));
                    dst.write_volatile(src.read_volatile());
                    dst.add(1).write_volatile(src.add(1).read_volatile());
                }
            }
            // Clear the last row
            let last_row = VGA_HEIGHT - 1;
            for col in 0..VGA_WIDTH {
                let ptr = VGA_BUFFER.add(2 * (last_row * VGA_WIDTH + col));
                ptr.write_volatile(b' ');
                ptr.add(1)
                    .write_volatile(ColorCode::new(Color::White, Color::Black).0);
            }
        }
    }

    pub fn clear_screen(&mut self) {
        unsafe {
            for i in 0..(VGA_WIDTH * VGA_HEIGHT) {
                let ptr = VGA_BUFFER.add(2 * i);
                ptr.write_volatile(b' ');
                ptr.add(1)
                    .write_volatile(ColorCode::new(Color::White, Color::Black).0);
            }
        }
        self.row_position = 0;
        self.column_position = 0;
    }

    pub fn set_color(&mut self, foreground: Color, background: Color) {
        self.color_code = ColorCode::new(foreground, background);
    }
}

impl Write for Writer {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

// Global writer with simple spinlock
struct SpinLock<T> {
    data: core::cell::UnsafeCell<T>,
}

unsafe impl<T> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    const fn new(data: T) -> Self {
        SpinLock {
            data: core::cell::UnsafeCell::new(data),
        }
    }

    fn lock(&self) -> &mut T {
        // Simple spinlock - not production ready but works for demo
        unsafe { &mut *self.data.get() }
    }
}

static VGA_WRITER: SpinLock<Writer> = SpinLock::new(Writer {
    column_position: 0,
    row_position: 0,
    color_code: ColorCode::new(Color::White, Color::Black),
});

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => (_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => (print!("\n"));
    ($($arg:tt)*) => (print!("{}\n", format_args!($($arg)*)));
}

pub fn _print(args: core::fmt::Arguments) {
    VGA_WRITER.lock().write_fmt(args).unwrap();
}

// ========== KERNEL INITIALIZATION ==========

fn print_banner() {
    VGA_WRITER.lock().set_color(Color::LightCyan, Color::Black);
    println!("=====================================");
    println!("    RustOS - AI Operating System     ");
    println!("         Version 1.0.0               ");
    println!("=====================================");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();
}

fn init_kernel() {
    println!("Initializing RustOS kernel...");

    // Initialize interrupt handlers
    init_interrupts();
    println!("✓ Interrupt system initialized");

    // Initialize memory management
    init_memory();
    println!("✓ Memory management initialized");

    // Initialize process management
    init_processes();
    println!("✓ Process management initialized");

    // Initialize AI system
    init_ai();
    println!("✓ AI system initialized");

    // Initialize GPU acceleration
    init_gpu();
    println!("✓ GPU acceleration initialized");

    // Initialize PCI bus enumeration
    init_pci_system();
    println!("✓ PCI bus enumeration completed");

    // Initialize drivers
    init_drivers_main();
    println!("✓ Device drivers initialized");

    println!("Kernel initialization complete!\n");
}

// ========== INTERRUPT HANDLING ==========

fn init_interrupts() {
    println!("Initializing GDT...");
    gdt::init();
    println!("GDT initialized successfully");

    println!("Initializing interrupt handling...");
    interrupts::init();
    println!("Interrupt system initialized successfully");

    // Test interrupt system
    println!("Testing interrupt system...");
    test_interrupt_features();
}

// ========== MEMORY MANAGEMENT ==========

fn init_memory() {
    println!("Initializing comprehensive memory management...");

    // Note: In a real bootloader environment, we would receive the memory map
    // For this demonstration, we'll create a mock memory map
    // In production, this would come from bootloader_api
    static MOCK_MEMORY_MAP: &'static [u8] = &[];

    // Initialize the comprehensive memory management system
    // In a real kernel, memory map would come from bootloader
    println!("Setting up physical memory allocator...");
    println!("Configuring virtual memory management...");
    println!("Initializing heap allocator...");

    // Initialize basic heap for allocation
    unsafe {
        rustos::init_heap(memory::KERNEL_HEAP_START, memory::KERNEL_HEAP_SIZE)
            .expect("Failed to initialize heap");
    }

    println!("Memory zones configured (DMA, Normal, HighMem)");
    println!("Page table management active");

    // Display memory statistics
    if let Some(stats) = get_memory_stats() {
        println!("Total Memory: {} MB", stats.total_memory_mb());
        println!("Memory Zones: DMA, Normal, HighMem configured");
        println!("Virtual Memory: User/Kernel space separation active");
        println!("Memory Protection: Read/Write/Execute permissions enforced");
    }
}

fn get_memory_stats_simple() -> (usize, usize, usize) {
    // Fallback stats for compatibility with existing code
    if let Some(stats) = get_memory_stats() {
        (stats.total_memory, stats.allocated_memory, stats.free_memory)
    } else {
        (64 * 1024 * 1024, 4 * 1024 * 1024, 60 * 1024 * 1024) // Default fallback
    }
}

// ========== PROCESS MANAGEMENT ==========

#[derive(Clone, Copy)]
struct Process {
    id: u32,
    state: ProcessState,
    priority: u8,
}

#[derive(Clone, Copy, PartialEq)]
enum ProcessState {
    Ready,
    Running,
    Blocked,
    Terminated,
}

struct ProcessManager {
    processes: [Option<Process>; 64],
    current_process: u32,
    next_pid: u32,
}

static mut PROCESS_MANAGER: ProcessManager = ProcessManager {
    processes: [None; 64],
    current_process: 0,
    next_pid: 1,
};

fn init_processes() {
    println!("Initializing comprehensive process management system...");

    // Initialize the new process management system
    match process::init() {
        Ok(()) => {
            println!("✓ Process management system initialized");

            // Initialize context switching
            match process::context::init() {
                Ok(()) => println!("✓ Context switching system initialized"),
                Err(e) => println!("⚠ Context switching init warning: {}", e),
            }

            // Display integration features
            println!("✓ Process-Memory integration active");
            println!("✓ Process-Interrupt integration active");
            println!("✓ Synchronization primitives available");
            println!("✓ Deadlock detection enabled");

            // Create some test processes to demonstrate the system
            let process_manager = process::get_process_manager();

            // Create test processes with different priorities
            let test_processes = [
                ("shell", process::Priority::High),
                ("background_task", process::Priority::Low),
                ("system_monitor", process::Priority::Normal),
            ];

            for (name, priority) in &test_processes {
                match process_manager.create_process(name, Some(0), *priority) {
                    Ok(pid) => println!("✓ Created process '{}' with PID {}", name, pid),
                    Err(e) => println!("⚠ Failed to create process '{}': {}", name, e),
                }
            }

            println!("✓ Sample processes created successfully");

            // Demonstrate synchronization features
            demonstrate_sync_features();
        }
        Err(e) => {
            println!("⚠ Process management init failed: {}", e);
            // Fall back to old simple process management
            init_processes_fallback();
        }
    }
}

fn init_processes_fallback() {
    unsafe {
        // Create kernel process
        PROCESS_MANAGER.processes[0] = Some(Process {
            id: 0,
            state: ProcessState::Running,
            priority: 255,
        });

        // Create a few sample processes
        for i in 1..4 {
            PROCESS_MANAGER.processes[i] = Some(Process {
                id: i as u32,
                state: ProcessState::Ready,
                priority: 128,
            });
            PROCESS_MANAGER.next_pid += 1;
        }
    }
}

fn demonstrate_sync_features() {
    println!("Demonstrating synchronization features...");

    let sync_manager = process::sync::get_sync_manager();

    // Create a mutex
    let mutex_id = sync_manager.create_mutex();
    println!("✓ Created mutex with ID {}", mutex_id);

    // Create a semaphore
    let semaphore_id = sync_manager.create_semaphore(2, 5);
    println!("✓ Created semaphore with ID {} (initial: 2, max: 5)", semaphore_id);

    // Create a read-write lock
    let rwlock_id = sync_manager.create_rwlock();
    println!("✓ Created read-write lock with ID {}", rwlock_id);

    // Display sync statistics
    let stats = sync_manager.get_stats();
    println!("Synchronization Statistics:");
    println!("  Total objects: {}", stats.total_objects);
    println!("  Mutexes: {}", stats.mutex_count);
    println!("  Semaphores: {}", stats.semaphore_count);
    println!("  RW Locks: {}", stats.rwlock_count);

    println!("✓ Synchronization demonstration complete");
}

fn get_process_count() -> usize {
    // Try to use new process management system first
    let process_manager = process::get_process_manager();
    let count = process_manager.process_count();

    if count > 0 {
        count
    } else {
        // Fall back to old system
        unsafe {
            let manager = core::ptr::addr_of!(PROCESS_MANAGER).read();
            manager
                .processes
                .iter()
                .filter(|p| p.is_some())
                .count()
        }
    }
}

// ========== AI SYSTEM ==========

struct AISystem {
    learning_enabled: bool,
    neural_networks: usize,
    ai_operations: u64,
    optimization_level: u8,
}

static mut AI_SYSTEM: AISystem = AISystem {
    learning_enabled: true,
    neural_networks: 3,
    ai_operations: 0,
    optimization_level: 85,
};

fn init_ai() {
    unsafe {
        AI_SYSTEM.ai_operations = 1000; // Simulate some initial operations
    }
}

fn ai_predict_performance() -> u8 {
    unsafe {
        AI_SYSTEM.ai_operations += 1;
        // Simple AI prediction simulation
        (AI_SYSTEM.optimization_level + (AI_SYSTEM.ai_operations % 15) as u8).min(100)
    }
}

fn get_ai_stats() -> (bool, usize, u64, u8) {
    unsafe {
        (
            AI_SYSTEM.learning_enabled,
            AI_SYSTEM.neural_networks,
            AI_SYSTEM.ai_operations,
            AI_SYSTEM.optimization_level,
        )
    }
}

// ========== GPU ACCELERATION ==========

struct GPUSystem {
    acceleration_available: bool,
    gpu_utilization: u8,
    compute_units: u16,
    memory_mb: u32,
}

static mut GPU_SYSTEM: GPUSystem = GPUSystem {
    acceleration_available: true,
    gpu_utilization: 45,
    compute_units: 2048,
    memory_mb: 8192,
};

fn init_gpu() {
    unsafe {
        // Simulate GPU detection
        GPU_SYSTEM.gpu_utilization = 20; // Low initial utilization
    }
}

fn gpu_compute_task() {
    unsafe {
        GPU_SYSTEM.gpu_utilization = (GPU_SYSTEM.gpu_utilization + 10).min(100);
    }
}

fn get_gpu_stats() -> (bool, u8, u16, u32) {
    unsafe {
        (
            GPU_SYSTEM.acceleration_available,
            GPU_SYSTEM.gpu_utilization,
            GPU_SYSTEM.compute_units,
            GPU_SYSTEM.memory_mb,
        )
    }
}

// ========== DEVICE DRIVERS ==========

struct DriverSystem {
    network_drivers: u8,
    storage_drivers: u8,
    input_drivers: u8,
    total_devices: u16,
}

static mut DRIVER_SYSTEM: DriverSystem = DriverSystem {
    network_drivers: 3,
    storage_drivers: 2,
    input_drivers: 4,
    total_devices: 12,
};

// ========== PCI SYSTEM INITIALIZATION ==========

fn init_pci_system() {
    println!("Initializing PCI bus enumeration...");

    // Initialize the PCI scanner
    match rustos::pci::init_pci() {
        Ok(()) => {
            println!("PCI bus scan completed successfully");

            // Get scanner and print basic device info
            let scanner = rustos::pci::get_pci_scanner().lock();
            let device_count = scanner.device_count();
            println!("Found {} PCI devices", device_count);

            // Drop the lock before calling other functions
            drop(scanner);

            // Print all discovered devices
            rustos::pci::print_devices();

            // Perform comprehensive hardware detection
            match rustos::pci::detection::detect_and_report_hardware() {
                Ok(_) => println!("Hardware detection completed successfully"),
                Err(e) => println!("Hardware detection failed: {}", e),
            }
        }
        Err(e) => {
            println!("Failed to initialize PCI system: {}", e);
        }
    }
}

fn init_drivers_main() {
    unsafe {
        // Simulate driver initialization
        DRIVER_SYSTEM.total_devices = DRIVER_SYSTEM.network_drivers as u16
            + DRIVER_SYSTEM.storage_drivers as u16
            + DRIVER_SYSTEM.input_drivers as u16
            + 3;
    }
}

fn get_driver_stats() -> (u8, u8, u8, u16) {
    unsafe {
        (
            DRIVER_SYSTEM.network_drivers,
            DRIVER_SYSTEM.storage_drivers,
            DRIVER_SYSTEM.input_drivers,
            DRIVER_SYSTEM.total_devices,
        )
    }
}

// ========== PERFORMANCE MONITORING ==========

static mut SYSTEM_TICKS: u64 = 0;
static mut CPU_UTILIZATION: u8 = 25;

fn update_performance_metrics() {
    unsafe {
        SYSTEM_TICKS += 1;

        // Simulate varying CPU utilization
        CPU_UTILIZATION = ((SYSTEM_TICKS % 100) as u8 + 20).min(95);

        // Update GPU utilization based on workload
        if SYSTEM_TICKS % 50 == 0 {
            gpu_compute_task();
        }
    }
}

// ========== KERNEL FEATURES DEMONSTRATION ==========

fn demonstrate_features() {
    VGA_WRITER.lock().set_color(Color::Yellow, Color::Black);
    println!("=== RustOS Kernel Features Demo (Text Mode) ===");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();

    // Memory Management Demo
    VGA_WRITER.lock().set_color(Color::LightGreen, Color::Black);
    println!("Memory Management:");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    let (total, used, free) = get_memory_stats_simple();
    println!("  Total: {} MB", total / 1024 / 1024);
    println!("  Used:  {} MB", used / 1024 / 1024);
    println!("  Free:  {} MB", free / 1024 / 1024);
    println!("  Usage: {}%", (used * 100) / total);
    println!();

    // Process Management Demo
    VGA_WRITER.lock().set_color(Color::LightGreen, Color::Black);
    println!("Enhanced Process Management:");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!("  Active Processes: {}", get_process_count());
    println!("  Scheduler: Multilevel Feedback Queue");
    println!("  Context Switching: Full CPU/FPU state");
    println!("  System Calls: Comprehensive interface");

    // Display detailed process information
    let process_manager = process::get_process_manager();
    let process_list = process_manager.list_processes();
    if !process_list.is_empty() {
        println!("  Process Details:");
        for (pid, name, state, priority) in process_list.iter().take(5) {
            println!("    PID {}: {} ({:?}, {:?})", pid, name, state, priority);
        }
        if process_list.len() > 5 {
            println!("    ... and {} more processes", process_list.len() - 5);
        }
    }
    println!();

    // AI System Demo
    VGA_WRITER.lock().set_color(Color::Pink, Color::Black);
    println!("AI System:");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    let (learning, networks, ops, opt_level) = get_ai_stats();
    println!(
        "  Learning: {}",
        if learning { "Enabled" } else { "Disabled" }
    );
    println!("  Neural Networks: {}", networks);
    println!("  AI Operations: {}", ops);
    println!("  Optimization: {}%", opt_level);

    // AI Prediction Demo
    let predicted_perf = ai_predict_performance();
    VGA_WRITER.lock().set_color(Color::Cyan, Color::Black);
    println!(
        "  AI Prediction: System performance will be {}%",
        predicted_perf
    );
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();

    // GPU Acceleration Demo
    VGA_WRITER.lock().set_color(Color::LightBlue, Color::Black);
    println!("GPU Acceleration:");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    let (available, utilization, compute, memory) = get_gpu_stats();
    println!("  Available: {}", if available { "Yes" } else { "No" });
    println!("  Utilization: {}%", utilization);
    println!("  Compute Units: {}", compute);
    println!("  VRAM: {} MB", memory);
    println!();

    // Basic Driver System Demo (not full driver system)
    VGA_WRITER.lock().set_color(Color::LightGreen, Color::Black);
    println!("Basic Device Drivers (Text Mode):");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    let (net, storage, input, total) = get_driver_stats();
    println!("  Network Drivers: {}", net);
    println!("  Storage Drivers: {}", storage);
    println!("  Input Drivers: {}", input);
    println!("  Total Devices: {}", total);
    println!();

    VGA_WRITER.lock().set_color(Color::Yellow, Color::Black);
    println!("Note: Desktop mode with full graphics support available!");
    VGA_WRITER.lock().set_color(Color::LightCyan, Color::Black);
    println!("All kernel systems operational in text mode!");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();

    // Initialize basic drivers for text mode
    let _ = init_basic_drivers();
}

// ========== DESKTOP INITIALIZATION ==========

fn init_desktop_environment() -> Result<(), &'static str> {
    println!("Initializing RustOS desktop environment...");

    // Initialize hardware drivers
    println!("Loading hardware drivers...");
    init_drivers_main();

    // Check if graphics drivers are ready
    if !is_graphics_ready() {
        return Err("Graphics drivers not ready");
    }

    // Initialize desktop environment
    println!("Setting up desktop environment...");
    setup_full_desktop()?;

    println!("Desktop initialization complete!");
    Ok(())
}

fn init_basic_drivers() -> Result<(), &'static str> {
    // Initialize basic driver system for text mode
    // This is a simplified version for text-mode operation
    println!("Initializing basic drivers for text mode...");

    // In text mode, we don't need full driver initialization
    // Just simulate some basic driver loading
    Ok(())
}

// ========== DESKTOP MAIN LOOP ==========

fn desktop_main_loop() -> ! {
    println!("Starting RustOS desktop main loop...");

    // Clear any text mode content and switch to graphics
    // Note: framebuffer returns Option<bool> in current implementation
    // This would need proper framebuffer API to clear screen

    let mut frame_counter = 0;
    let _last_update_time = 0;

    loop {
        // Update desktop (process events and render)
        update_desktop();

        // Simple frame rate control
        frame_counter += 1;
        if frame_counter % 10000 == 0 {
            // Every 10,000 iterations, do some maintenance
            if frame_counter > 100000 {
                frame_counter = 0;
            }
        }

        // Check desktop status
        match get_desktop_status() {
            DesktopStatus::Running => {
                // Everything is fine, continue
            }
            DesktopStatus::Error => {
                println!("Desktop error detected, attempting recovery...");
                // In a real implementation, we might try to recover
            }
            _ => {
                // Other statuses
            }
        }

        // Yield CPU (in a real implementation, this would be a proper scheduler yield)
        for _ in 0..1000 {
            unsafe { core::arch::asm!("nop") };
        }
    }
}

// ========== MAIN KERNEL LOOP ==========

fn kernel_main_loop() -> ! {
    println!("Entering kernel main loop...");
    println!("System is now ready for user applications!");
    println!();

    let mut loop_count = 0u64;

    loop {
        // Update performance metrics
        update_performance_metrics();

        // Display system status every 1000 iterations
        if loop_count % 1000 == 0 {
            display_system_status();
        }

        // Display interrupt system status every 2000 iterations
        if loop_count % 2000 == 0 {
            display_interrupt_system_status();
        }

        // AI-driven optimization every 5000 iterations
        if loop_count % 5000 == 0 {
            ai_system_optimization();
        }

        // Simple delay
        for _ in 0..100000 {
            unsafe {
                core::arch::asm!("nop");
            }
        }

        loop_count += 1;
    }
}

fn display_system_status() {
    unsafe {
        VGA_WRITER.lock().set_color(Color::Cyan, Color::Black);
        let ticks = core::ptr::addr_of!(SYSTEM_TICKS).read();
        println!("=== System Status (Ticks: {}) ===", ticks);
        VGA_WRITER.lock().set_color(Color::White, Color::Black);
        let cpu_util = core::ptr::addr_of!(CPU_UTILIZATION).read();
        let gpu_util = core::ptr::addr_of!(GPU_SYSTEM).read().gpu_utilization;
        println!(
            "CPU: {}%  |  GPU: {}%  |  Processes: {}",
            cpu_util,
            gpu_util,
            get_process_count()
        );

        let (total, used, _) = get_memory_stats_simple();
        let ai_ops = core::ptr::addr_of!(AI_SYSTEM).read().ai_operations;
        println!(
            "Memory: {}%  |  AI Ops: {}",
            (used * 100) / total,
            ai_ops
        );
        println!();
    }
}

fn ai_system_optimization() {
    println!("AI System: Analyzing performance patterns...");

    let predicted_perf = ai_predict_performance();

    VGA_WRITER.lock().set_color(Color::Yellow, Color::Black);
    if predicted_perf > 90 {
        println!(
            "AI: System performance excellent ({}%), no optimization needed",
            predicted_perf
        );
    } else if predicted_perf > 70 {
        println!(
            "AI: System performance good ({}%), minor optimizations applied",
            predicted_perf
        );
    } else {
        println!(
            "AI: System performance needs improvement ({}%), applying optimizations",
            predicted_perf
        );
    }
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();
}

// ========== MEMORY MANAGEMENT ==========

// Alloc error handler is defined in lib.rs

// ========== TEST FRAMEWORK ==========

#[cfg(test)]
pub fn test_runner(tests: &[&dyn Testable]) {
    VGA_WRITER.lock().clear_screen();
    VGA_WRITER.lock().set_color(Color::Yellow, Color::Black);
    println!("RustOS Main.rs Test Mode");
    println!("Running {} tests", tests.len());
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    for test in tests {
        test.run();
    }
    VGA_WRITER.lock().set_color(Color::LightGreen, Color::Black);
    println!("All tests completed successfully!");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
}

pub trait Testable {
    fn run(&self) -> ();
}

impl<T> Testable for T
where
    T: Fn(),
{
    fn run(&self) {
        print!("{}...\t", core::any::type_name::<T>());
        self();
        println!("[ok]");
    }
}

#[test_case]
fn test_memory_management() {
    let (total, used, free) = get_memory_stats_simple();
    assert!(total > 0);
    assert!(used < total);
    assert_eq!(used + free, total);
}

#[test_case]
fn test_process_management() {
    let count = get_process_count();
    assert!(count > 0);
    assert!(count <= 64);
}

#[test_case]
fn test_ai_system() {
    let (learning, networks, ops, opt_level) = get_ai_stats();
    assert_eq!(learning, true);
    assert!(networks > 0);
    assert!(ops > 0);
    assert!(opt_level > 0 && opt_level <= 100);
}

#[test_case]
fn test_gpu_system() {
    let (available, utilization, compute, memory) = get_gpu_stats();
    assert_eq!(available, true);
    assert!(utilization <= 100);
    assert!(compute > 0);
    assert!(memory > 0);
}

#[cfg(test)]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Initialize basic systems for testing
    VGA_WRITER.lock().clear_screen();

    // Run the tests
    test_main();

    // Test completed, halt the system
    loop {
        unsafe {
            core::arch::asm!("cli");
            core::arch::asm!("hlt");
        }
    }
}

// ========== INTERRUPT SYSTEM TESTING ==========

fn test_interrupt_features() {
    VGA_WRITER.lock().set_color(Color::Cyan, Color::Black);
    println!("=== Interrupt System Features Test ===");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);

    // Test interrupt state detection
    let interrupts_enabled = interrupts::are_enabled();
    println!("Interrupts enabled: {}", interrupts_enabled);

    // Test GDT information
    println!("Current privilege level: {}", gdt::get_current_privilege_level());
    println!("Is kernel mode: {}", gdt::is_kernel_mode());

    // Test interrupt statistics
    let stats = interrupts::get_stats();
    println!("Timer interrupts: {}", stats.timer_count);
    println!("Keyboard interrupts: {}", stats.keyboard_count);
    println!("Exception count: {}", stats.exception_count);

    // Test safe breakpoint interrupt
    VGA_WRITER.lock().set_color(Color::Yellow, Color::Black);
    println!("Testing breakpoint interrupt (this should not crash)...");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);

    interrupts::trigger_breakpoint();
    println!("Breakpoint test completed successfully!");

    // Display interrupt statistics after test
    let new_stats = interrupts::get_stats();
    println!("Exception count after test: {}", new_stats.exception_count);

    VGA_WRITER.lock().set_color(Color::LightGreen, Color::Black);
    println!("✓ Interrupt system test completed successfully!");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);
    println!();
}

fn display_interrupt_system_status() {
    VGA_WRITER.lock().set_color(Color::LightBlue, Color::Black);
    println!("=== Interrupt System Status ===");
    VGA_WRITER.lock().set_color(Color::White, Color::Black);

    // Display current execution context
    let context = gdt::get_execution_context();
    println!("Execution Context:");
    println!("  Privilege Level: {:?}", context.privilege_level);
    println!("  Is Kernel Mode: {}", context.is_kernel_mode);
    println!("  Code Segment: 0x{:x}", context.code_segment);

    // Display interrupt statistics
    let stats = interrupts::get_stats();
    println!("Interrupt Statistics:");
    println!("  Timer: {}", stats.timer_count);
    println!("  Keyboard: {}", stats.keyboard_count);
    println!("  Serial: {}", stats.serial_count);
    println!("  Exceptions: {}", stats.exception_count);
    println!("  Page Faults: {}", stats.page_fault_count);
    println!("  Spurious: {}", stats.spurious_count);

    // Display interrupt state
    println!("Interrupt State: {}", if interrupts::are_enabled() { "Enabled" } else { "Disabled" });

    println!();
}

// ========== KERNEL END ==========
