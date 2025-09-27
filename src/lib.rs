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
    
    print_banner("RustOS - Hardware-Optimized AI Operating System", Color::LightCyan, Color::Black);
    print_colored("Architecture: x86_64/aarch64 compatible", Color::LightBlue, Color::Black);
    print_colored("Initializing hardware-focused AI kernel components...", Color::Yellow, Color::Black);
    
    init();
    
    // Initialize GPU acceleration system
    print_colored("Initializing GPU acceleration system...", Color::Cyan, Color::Black);
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
    ai::init_ai_system();
    
    print_colored("RustOS AI kernel successfully initialized!", Color::LightGreen, Color::Black);
    
    // Print AI status without format macro
    let status = ai::get_ai_status();
    match status {
        ai::AIStatus::Ready => vga_buffer::print_ai_status("Ready"),
        ai::AIStatus::Learning => vga_buffer::print_ai_status("Learning"),
        ai::AIStatus::Inferencing => vga_buffer::print_ai_status("Inferencing"),
        ai::AIStatus::Error => vga_buffer::print_ai_status("Error"),
        ai::AIStatus::Initializing => vga_buffer::print_ai_status("Initializing"),
    }
    
    print_colored("AI now learning hardware patterns for optimal performance...", Color::Pink, Color::Black);
    
    // Demonstrate GPU-accelerated desktop UI
    if gpu::is_gpu_acceleration_available() {
        print_colored("Demonstrating GPU-accelerated desktop UI...", Color::Magenta, Color::Black);
        
        // Clear screen and draw desktop
        gpu::gpu_clear_screen(0x0040A0FF); // Blue desktop background
        
        // Draw some UI elements using GPU acceleration
        gpu::gpu_draw_rect(10, 10, 200, 100, 0xC0C0C0FF); // Gray window
        gpu::gpu_draw_rect(15, 15, 190, 25, 0x000080FF);  // Blue title bar
        gpu::gpu_draw_rect(50, 50, 100, 30, 0xE0E0E0FF);  // Light gray button
        
        // Draw taskbar
        gpu::gpu_draw_rect(0, 1040, 1920, 40, 0x808080FF); // Gray taskbar
        gpu::gpu_draw_rect(5, 1045, 80, 30, 0xA0A0A0FF);   // Start button
        
        gpu::gpu_present(); // Present the framebuffer
        
        print_colored("GPU-accelerated desktop UI rendered successfully!", Color::LightGreen, Color::Black);
    }
    
    #[cfg(test)]
    test_main();
    
    print_banner("System Ready - Hardware Optimization Active", Color::LightGreen, Color::Black);
    hlt_loop();
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