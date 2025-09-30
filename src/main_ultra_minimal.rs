#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Write directly to VGA buffer so something appears on screen
    let vga_buffer = 0xb8000 as *mut u8;

    let message = b"RustOS Kernel Running - Stable Minimal Version";

    unsafe {
        for (i, &byte) in message.iter().enumerate() {
            let offset = i * 2;
            *vga_buffer.add(offset) = byte;
            *vga_buffer.add(offset + 1) = 0x0f; // White on black
        }
    }

    // Halt forever
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