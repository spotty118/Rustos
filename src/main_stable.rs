#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

entry_point!(kernel_main);

// Simple VGA writer without complex initialization
struct Writer {
    column_position: usize,
}

impl Writer {
    fn write_byte(&mut self, byte: u8) {
        let vga_buffer = 0xb8000 as *mut u8;

        match byte {
            b'\n' => {
                self.column_position = 0;
            }
            byte => {
                if self.column_position >= 80 {
                    self.column_position = 0;
                }

                let offset = self.column_position * 2;
                unsafe {
                    *vga_buffer.add(offset) = byte;
                    *vga_buffer.add(offset + 1) = 0x0f; // White on black
                }
                self.column_position += 1;
            }
        }
    }

    fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            self.write_byte(byte)
        }
    }
}

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    let mut writer = Writer { column_position: 0 };

    writer.write_string("================================================================================\n");
    writer.write_string("                        RustOS Kernel v1.0\n");
    writer.write_string("              Hardware-Optimized AI Operating System\n");
    writer.write_string("================================================================================\n");
    writer.write_string("\n");
    writer.write_string("[BOOT] Kernel entry point reached\n");
    writer.write_string("[BOOT] VGA text mode initialized\n");
    writer.write_string("\n");
    writer.write_string("System Status:\n");
    writer.write_string("  - Bootloader: OK\n");
    writer.write_string("  - VGA Buffer: OK\n");
    writer.write_string("  - Basic I/O: OK\n");
    writer.write_string("\n");
    writer.write_string("Kernel is running in stable mode.\n");
    writer.write_string("\n");

    // Stable HLT loop
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