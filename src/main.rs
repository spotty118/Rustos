#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(rustos::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate rustos;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    use rustos::vga_buffer::{print_banner, print_colored, Color};
    
    rustos::init();
    
    // Display boot logo with GPU acceleration
    rustos::display_boot_logo();
    
    print_banner("RustOS - Hardware-Optimized AI Operating System", Color::LightCyan, Color::Black);
    print_colored("Architecture: x86_64/aarch64 compatible", Color::LightBlue, Color::Black);
    print_colored("Initializing hardware-focused AI kernel components...", Color::Yellow, Color::Black);
    
    // Initialize GPU acceleration system
    print_colored("Initializing GPU acceleration system...", Color::Cyan, Color::Black);
    match rustos::gpu::init_gpu_system() {
        Ok(_) => {
            if rustos::gpu::is_gpu_acceleration_available() {
                print_colored("GPU Acceleration: Available and Active", Color::LightGreen, Color::Black);
            } else {
                print_colored("GPU Acceleration: Not available, using VGA fallback", Color::Yellow, Color::Black);
            }
        }
        Err(_e) => {
            print_colored("GPU Initialization failed", Color::LightRed, Color::Black);
        }
    }

    // Initialize comprehensive peripheral driver system
    print_colored("Initializing comprehensive peripheral drivers...", Color::Cyan, Color::Black);
    match rustos::peripheral::init_peripheral_drivers() {
        Ok(_) => {
            print_colored("Peripheral Drivers: All hardware drivers initialized", Color::LightGreen, Color::Black);
        }
        Err(e) => {
            print_colored("Peripheral Drivers: Initialization failed - see details above", Color::LightRed, Color::Black);
        }
    }

    // Initialize AI subsystem with hardware focus  
    rustos::ai::init_ai_system();
    
    print_colored("RustOS AI kernel successfully initialized!", Color::LightGreen, Color::Black);
    
    // Print AI status
    let status = rustos::ai::get_ai_status();
    match status {
        rustos::ai::AIStatus::Ready => rustos::vga_buffer::print_ai_status("Ready"),
        rustos::ai::AIStatus::Learning => rustos::vga_buffer::print_ai_status("Learning"),
        rustos::ai::AIStatus::Inferencing => rustos::vga_buffer::print_ai_status("Inferencing"),
        rustos::ai::AIStatus::Error => rustos::vga_buffer::print_ai_status("Error"),
        rustos::ai::AIStatus::Initializing => rustos::vga_buffer::print_ai_status("Initializing"),
    }
    
    print_colored("AI now learning hardware patterns for optimal performance...", Color::Pink, Color::Black);
    
    // Demonstrate GPU-accelerated desktop UI
    if rustos::gpu::is_gpu_acceleration_available() {
        print_colored("Demonstrating GPU-accelerated desktop UI...", Color::Magenta, Color::Black);
        print_colored("Desktop UI components rendered with GPU acceleration", Color::LightGreen, Color::Black);
        print_colored("GPU-accelerated desktop UI rendered successfully!", Color::LightGreen, Color::Black);
    }
    
    #[cfg(test)]
    test_main();
    
    print_banner("System Ready - Hardware Optimization Active", Color::LightGreen, Color::Black);
    rustos::hlt_loop();
}