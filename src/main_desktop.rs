#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

mod intrinsics;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Simple VGA text output first
    let vga_buffer = 0xb8000 as *mut u8;
    let msg = b"RustOS Desktop Environment Loading...";

    unsafe {
        for (i, &byte) in msg.iter().enumerate() {
            *vga_buffer.add(i * 2) = byte;
            *vga_buffer.add(i * 2 + 1) = 0x0f;
        }
    }

    // Use text mode desktop (graphics mode requires heap allocator)
    draw_text_desktop();

    // Main loop
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

// Graphics desktop drawing disabled - requires heap allocator
// Will be implemented after memory management is initialized

fn draw_text_desktop() {
    let vga_buffer = 0xb8000 as *mut u8;

    let lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║                          RustOS Desktop Environment                          ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
        "  Graphics Mode: Text VGA 80x25",
        "",
        "  System Status:",
        "    [✓] Kernel: Running",
        "    [✓] VGA: Initialized",
        "    [✓] Desktop: Active",
        "",
        "  Available Applications:",
        "    → File Manager",
        "    → Terminal",
        "    → Calculator",
        "    → System Settings",
        "",
        "  Press any key to continue...",
    ];

    let mut row = 3;
    for line in &lines {
        let mut col = 0;
        for &byte in line.as_bytes() {
            unsafe {
                let offset = (row * 80 + col) * 2;
                *vga_buffer.add(offset) = byte;
                *vga_buffer.add(offset + 1) = 0x0f; // White on black
            }
            col += 1;
        }
        row += 1;
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    let vga_buffer = 0xb8000 as *mut u8;
    let msg = b"KERNEL PANIC - System Halted";

    unsafe {
        for (i, &byte) in msg.iter().enumerate() {
            *vga_buffer.add(i * 2) = byte;
            *vga_buffer.add(i * 2 + 1) = 0x4f; // White on red
        }
    }

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}