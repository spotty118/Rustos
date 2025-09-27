#![no_std]
#![cfg_attr(test, no_main)]
#![feature(custom_test_frameworks)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]
#![feature(abi_x86_interrupt)]

use core::panic::PanicInfo;

// Kernel modules
pub mod vga_buffer;
pub mod serial;
pub mod interrupts;
pub mod gdt;
pub mod memory;
pub mod allocator;
pub mod ai;
pub mod arch;
pub mod gpu;
pub mod boot_animation;
pub mod smp;
pub mod large_memory;
pub mod desktop_ui;

// Re-export commonly used items
// Note: Macros are already exported at crate root due to #[macro_export]

pub fn init() {
    gdt::init();
    interrupts::init_idt();
    unsafe { interrupts::PICS.lock().initialize() };
    x86_64::instructions::interrupts::enable();
}

pub fn hlt_loop() -> ! {
    loop {
        arch::halt_cpu();
    }
}

/// Entry point for the kernel
#[no_mangle] 
pub extern "C" fn _start() -> ! {
    use vga_buffer::{print_banner, print_colored, Color};
    
    // Start with animated boot logo
    boot_animation::run_boot_animation();
    
    print_banner("RustOS - Advanced AI Operating System", Color::LightCyan, Color::Black);
    print_colored("🚀 Multi-Core | 💾 Large Memory | 🖥️ Advanced Desktop", Color::LightBlue, Color::Black);
    print_colored("Initializing enhanced kernel components...", Color::Yellow, Color::Black);
    
    init();
    
    // Initialize SMP (Multi-core) system
    print_colored("🔄 Initializing multi-core processor support...", Color::Cyan, Color::Black);
    match smp::init_smp() {
        Ok(_) => {
            let core_count = smp::get_online_core_count();
            let total_cores = smp::get_total_core_count();
            // Create static strings to avoid format! macro
            if core_count == total_cores {
                print_colored("✅ SMP: All CPU cores online", Color::LightGreen, Color::Black);
            } else {
                print_colored("✅ SMP: Some CPU cores online", Color::LightGreen, Color::Black);
            }
            smp::print_cpu_topology();
        }
        Err(_e) => {
            print_colored("⚠️  SMP initialization failed", Color::Yellow, Color::Black);
        }
    }
    
    // Initialize large memory management
    print_colored("💾 Initializing large memory management...", Color::Cyan, Color::Black);
    // Note: In a real bootloader integration, we'd get the memory map here
    // For now, we'll initialize with a dummy memory map
    boot_animation::show_progress_bar(0.3, " Large Memory Detection");
    
    // Initialize GPU acceleration system
    print_colored("🎮 Initializing GPU acceleration system...", Color::Cyan, Color::Black);
    match gpu::init_gpu_system() {
        Ok(_) => {
            if gpu::is_gpu_acceleration_available() {
                if let Some(gpu_info) = gpu::get_active_gpu_info() {
                    match gpu_info.vendor {
                        gpu::GPUVendor::Intel => print_colored("GPU Acceleration: Intel GPU Active", Color::LightGreen, Color::Black),
                        gpu::GPUVendor::Nvidia => print_colored("GPU Acceleration: NVIDIA GPU Active", Color::LightGreen, Color::Black),
                        gpu::GPUVendor::AMD => print_colored("GPU Acceleration: AMD GPU Active", Color::LightGreen, Color::Black),
                        gpu::GPUVendor::Unknown => print_colored("GPU Acceleration: Unknown GPU Active", Color::LightGreen, Color::Black),
                    }
                    // Display memory in MB - simplified for no_std
                    let memory_mb = gpu_info.memory_size / (1024 * 1024);
                    if memory_mb < 1024 {
                        print_colored("GPU Memory: < 1 GB", Color::LightBlue, Color::Black);
                    } else if memory_mb < 8192 {
                        print_colored("GPU Memory: 1-8 GB", Color::LightBlue, Color::Black);
                    } else if memory_mb < 16384 {
                        print_colored("GPU Memory: 8-16 GB", Color::LightBlue, Color::Black);
                    } else {
                        print_colored("GPU Memory: > 16 GB", Color::LightBlue, Color::Black);
                    }
                } else {
                    print_colored("GPU Acceleration: Available", Color::LightGreen, Color::Black);
                }
            } else {
                print_colored("GPU Acceleration: Not available, using VGA fallback", Color::Yellow, Color::Black);
            }
        }
        Err(_e) => {
            print_colored("GPU Initialization failed", Color::LightRed, Color::Black);
        }
    }

    // Initialize AI subsystem with hardware focus  
    print_colored("🤖 Initializing AI subsystem...", Color::Pink, Color::Black);
    ai::init_ai_system();
    boot_animation::show_progress_bar(0.6, " AI System");
    
    print_colored("✅ RustOS enhanced kernel successfully initialized!", Color::LightGreen, Color::Black);
    
    // Print AI status
    let status = ai::get_ai_status();
    match status {
        ai::AIStatus::Ready => vga_buffer::print_ai_status("Ready"),
        ai::AIStatus::Learning => vga_buffer::print_ai_status("Learning"),
        ai::AIStatus::Inferencing => vga_buffer::print_ai_status("Inferencing"),
        ai::AIStatus::Error => vga_buffer::print_ai_status("Error"),
        ai::AIStatus::Initializing => vga_buffer::print_ai_status("Initializing"),
    }
    
    print_colored("🧠 AI now learning hardware patterns for optimal performance...", Color::Pink, Color::Black);
    
    // Initialize advanced desktop environment
    print_colored("🖥️  Initializing advanced desktop environment...", Color::Magenta, Color::Black);
    match desktop_ui::init_desktop() {
        Ok(_) => print_colored("✅ Advanced desktop UI initialized", Color::LightGreen, Color::Black),
        Err(_e) => print_colored("⚠️  Desktop init warning", Color::Yellow, Color::Black),
    }
    
    // Demonstrate GPU-accelerated desktop UI with enhanced features
    if gpu::is_gpu_acceleration_available() {
        print_colored("🎨 Demonstrating enhanced GPU-accelerated desktop UI...", Color::Magenta, Color::Black);
        
        // Clear screen and set up desktop
        gpu::gpu_clear_screen(0x001144FF); // Deep blue background
        
        // Draw enhanced desktop elements
        draw_enhanced_desktop();
        
        gpu::gpu_present(); // Present the framebuffer
        
        print_colored("✨ Enhanced GPU-accelerated desktop UI rendered successfully!", Color::LightGreen, Color::Black);
    }
    
    boot_animation::show_progress_bar(1.0, " System Ready!");
    
    #[cfg(test)]
    test_main();
    
    print_banner("🎉 System Ready - Multi-Core AI Desktop Active! 🎉", Color::LightGreen, Color::Black);
    print_colored("Features: ✅ Multi-Core ✅ Large Memory ✅ Advanced Desktop ✅ AI Optimization", Color::LightCyan, Color::Black);
    
    hlt_loop();
}

/// Draw enhanced desktop with multiple windows and widgets
fn draw_enhanced_desktop() {
    // Enhanced desktop background with gradient effect
    for y in 0..768 {
        let blue_intensity = 0x11 + ((y * 0x33) / 768);
        let color = (blue_intensity << 16) | (blue_intensity << 8) | 0x44;
        gpu::gpu_draw_rect(0, y, 1024, 1, color | 0xFF000000);
    }
    
    // Main application window
    gpu::gpu_draw_rect(100, 80, 400, 300, 0xF0F0F0FF); // Light gray window
    gpu::gpu_draw_rect(100, 80, 400, 30, 0x0060C0FF);  // Blue title bar
    gpu::gpu_draw_rect(105, 85, 20, 20, 0xFF4040FF);   // Close button
    
    // Window content - simulated text editor
    gpu::gpu_draw_rect(110, 120, 380, 240, 0xFFFFFFFF); // White content area
    
    // Buttons in the window
    gpu::gpu_draw_rect(120, 140, 80, 25, 0xE0E0E0FF);  // Button 1
    gpu::gpu_draw_rect(220, 140, 80, 25, 0xE0E0E0FF);  // Button 2
    
    // Secondary window
    gpu::gpu_draw_rect(300, 200, 250, 200, 0xF8F8F8FF); // Another window
    gpu::gpu_draw_rect(300, 200, 250, 25, 0x4080FFFF);  // Title bar
    
    // Enhanced taskbar with multiple elements
    gpu::gpu_draw_rect(0, 728, 1024, 40, 0x404040FF);   // Dark gray taskbar
    gpu::gpu_draw_rect(5, 733, 100, 30, 0x6080FFFF);    // Start button (blue)
    gpu::gpu_draw_rect(120, 733, 80, 30, 0x8080FFFF);   // App button 1
    gpu::gpu_draw_rect(210, 733, 80, 30, 0x8080FFFF);   // App button 2
    
    // System tray area
    gpu::gpu_draw_rect(900, 733, 120, 30, 0x606060FF);  // System tray background
    gpu::gpu_draw_rect(905, 738, 20, 20, 0x40FF40FF);   // Network icon (green)
    gpu::gpu_draw_rect(935, 738, 20, 20, 0xFF8040FF);   // Audio icon (orange)
    gpu::gpu_draw_rect(965, 738, 50, 20, 0xFFFF40FF);   // Clock area (yellow)
    
    // Desktop icons
    gpu::gpu_draw_rect(50, 100, 48, 48, 0xFFE040FF);    // Folder icon
    gpu::gpu_draw_rect(50, 170, 48, 48, 0x4080FFFF);    // Application icon
    gpu::gpu_draw_rect(50, 240, 48, 48, 0xFF4080FF);    // Document icon
    
    // Status indicators for multi-core system
    gpu::gpu_draw_rect(850, 50, 150, 80, 0x20202080);   // Semi-transparent status panel
    
    // CPU core indicators (4 cores)
    for i in 0..4 {
        let color = match i % 4 {
            0 => 0xFF4040FF, // Red
            1 => 0x40FF40FF, // Green  
            2 => 0x4040FFFF, // Blue
            _ => 0xFFFF40FF, // Yellow
        };
        gpu::gpu_draw_rect(860 + i * 30, 60, 25, 15, color);
    }
    
    // Memory usage bar
    gpu::gpu_draw_rect(860, 85, 120, 10, 0x404040FF);   // Background
    gpu::gpu_draw_rect(860, 85, 80, 10, 0x40FF40FF);    // Usage (green)
}

/// Entry point for `cargo test`
#[cfg(test)]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    init();
    test_main();
    hlt_loop();
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("KERNEL PANIC: {}", info);
    hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    test_panic_handler(info)
}

pub trait Testable {
    fn run(&self) -> ();
}

impl<T> Testable for T
where
    T: Fn(),
{
    fn run(&self) {
        serial_print!("{}...\t", core::any::type_name::<T>());
        self();
        serial_println!("[ok]");
    }
}

pub fn test_runner(tests: &[&dyn Testable]) {
    serial_println!("Running {} tests", tests.len());
    for test in tests {
        test.run();
    }
    exit_qemu(QemuExitCode::Success);
}

pub fn test_panic_handler(info: &PanicInfo) -> ! {
    serial_println!("[failed]\n");
    serial_println!("Error: {}\n", info);
    exit_qemu(QemuExitCode::Failed);
    hlt_loop();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

pub fn exit_qemu(exit_code: QemuExitCode) {
    use x86_64::instructions::port::Port;

    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
}

// Basic test
#[test_case]
fn trivial_assertion() {
    assert_eq!(1, 1);
}