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
pub mod package_manager;

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

/// Display boot logo with GPU acceleration if available
pub fn display_boot_logo() {
    use vga_buffer::{print_colored, print_banner, Color};
    
    // Clear screen and display boot logo
    vga_buffer::clear_screen();
    
    print_banner("RustOS - Hardware-Optimized AI Operating System", Color::LightCyan, Color::Black);
    print_colored("Version 0.1.0 - GPU Accelerated Desktop", Color::White, Color::Black);
    print_colored("", Color::White, Color::Black); // Empty line
    
    // Try to initialize GPU-accelerated desktop
    if gpu::is_gpu_acceleration_available() {
        print_colored("Initializing GPU-accelerated desktop UI...", Color::LightGreen, Color::Black);
        
        // Initialize framebuffer and draw desktop
        match gpu::init_desktop_ui() {
            Ok(_) => {
                print_colored("✓ Desktop UI initialized successfully", Color::LightGreen, Color::Black);
                
                // Take a screenshot of the boot desktop
                match gpu::take_screenshot("boot_desktop.bmp") {
                    Ok(_) => print_colored("✓ Boot screenshot saved", Color::LightBlue, Color::Black),
                    Err(_) => print_colored("⚠ Screenshot failed", Color::Yellow, Color::Black),
                }
            },
            Err(_) => print_colored("✗ Desktop UI failed", Color::LightRed, Color::Black),
        }
    } else {
        print_colored("⚠ No GPU acceleration available, using VGA fallback", Color::Yellow, Color::Black);
    }
    
    print_colored("", Color::White, Color::Black); // Empty line
    
    // Initialize package manager integration
    match package_manager::init_package_manager() {
        Ok(_) => print_colored("Package Manager Integration: Ready", Color::Magenta, Color::Black),
        Err(_) => print_colored("Package Manager Integration: Failed", Color::LightRed, Color::Black),
    }
    
    print_colored("AI Learning Systems: Initialized", Color::Pink, Color::Black);
    print_colored("", Color::White, Color::Black); // Empty line
    
    // Demonstrate package manager integration
    package_manager::demonstrate_package_operations();
}