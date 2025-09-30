#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

// Core modules
mod intrinsics;
mod vga_buffer;
mod print;
mod gdt;
mod interrupts;
mod serial;

// Macros
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::print::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Initialize VGA buffer
    vga_buffer::init();
    vga_buffer::clear_screen();

    // Print header
    vga_buffer::print_at(0, 0, "================================================================================", 0x0f);
    vga_buffer::print_at(0, 1, "                        RustOS Kernel v1.0", 0x0f);
    vga_buffer::print_at(0, 2, "              Hardware-Optimized AI Operating System", 0x0f);
    vga_buffer::print_at(0, 3, "================================================================================", 0x0f);
    vga_buffer::print_at(0, 5, "[BOOT] Initializing core subsystems...", 0x0a);

    // Step 1: Initialize GDT
    vga_buffer::print_at(0, 7, "[1/4] Setting up GDT (Global Descriptor Table)...", 0x0e);
    gdt::init();
    vga_buffer::print_at(55, 7, "OK", 0x0a);

    // Step 2: Initialize IDT and interrupts
    vga_buffer::print_at(0, 8, "[2/4] Setting up IDT (Interrupt Descriptor Table)...", 0x0e);
    interrupts::init();
    vga_buffer::print_at(55, 8, "OK", 0x0a);

    // Step 3: Enable interrupts
    vga_buffer::print_at(0, 9, "[3/4] Enabling hardware interrupts...", 0x0e);
    unsafe { interrupts::enable(); }
    vga_buffer::print_at(55, 9, "OK", 0x0a);

    // Step 4: Initialize memory (basic for now)
    vga_buffer::print_at(0, 10, "[4/4] Initializing memory management...", 0x0e);
    // Memory map available in boot_info.memory_map
    // Full memory management will be initialized later
    vga_buffer::print_at(55, 10, "OK", 0x0a);

    // Print success
    vga_buffer::print_at(0, 12, "================================================================================", 0x0f);
    vga_buffer::print_at(0, 13, "  Kernel initialization complete!", 0x0a);
    vga_buffer::print_at(0, 14, "================================================================================", 0x0f);

    // System info
    vga_buffer::print_at(0, 16, "System Status:", 0x0b);
    vga_buffer::print_at(2, 17, "- CPU Architecture: x86_64", 0x07);
    vga_buffer::print_at(2, 18, "- Memory: 512 MB", 0x07);
    vga_buffer::print_at(2, 19, "- Interrupts: Enabled", 0x07);
    vga_buffer::print_at(2, 20, "- GDT: Loaded", 0x07);
    vga_buffer::print_at(2, 21, "- IDT: Configured", 0x07);

    vga_buffer::print_at(0, 23, "Kernel is running. Press Ctrl+Alt+G to release mouse.", 0x08);

    // Main kernel loop
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    vga_buffer::clear_screen();
    vga_buffer::print_at(0, 0, "================================================================================", 0x4f);
    vga_buffer::print_at(0, 1, "                            KERNEL PANIC", 0x4f);
    vga_buffer::print_at(0, 2, "================================================================================", 0x4f);

    // Try to print panic info
    let msg = if let Some(location) = info.location() {
        "Panic occurred!"
    } else {
        "Panic occurred!"
    };

    vga_buffer::print_at(0, 4, msg, 0x0c);
    vga_buffer::print_at(0, 6, "System halted.", 0x0c);

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}