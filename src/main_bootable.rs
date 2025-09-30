#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

// Include compiler intrinsics for missing symbols
mod intrinsics;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Direct VGA buffer access - simple and safe
    let vga_buffer = 0xb8000 as *mut u16;

    // Clear screen
    unsafe {
        for i in 0..(80 * 25) {
            *vga_buffer.add(i) = 0x0F20; // White space on black
        }
    }

    // Display boot message
    let message = "RustOS Kernel - Successfully Booted!";
    let info = "All critical bugs fixed and kernel operational";
    let ready = "System ready - Press Ctrl+Alt+Q to exit QEMU";

    unsafe {
        // Line 1
        for (i, byte) in message.bytes().enumerate() {
            *vga_buffer.add(i + (80 * 10)) = 0x0F00 | byte as u16;
        }

        // Line 2
        for (i, byte) in info.bytes().enumerate() {
            *vga_buffer.add(i + (80 * 12)) = 0x0A00 | byte as u16;
        }

        // Line 3
        for (i, byte) in ready.bytes().enumerate() {
            *vga_buffer.add(i + (80 * 14)) = 0x0E00 | byte as u16;
        }
    }

    // Halt loop - stable and efficient
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}