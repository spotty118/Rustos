#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

// Include compiler intrinsics for missing symbols
mod intrinsics;

// Include VGA buffer module for better output
mod vga_buffer;
// Include print module for print! and println! macros
mod print;

// Print macros
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

    // Clear screen and show boot message
    println!("================================================================================");
    println!("                        RustOS Kernel v1.0");
    println!("              Hardware-Optimized AI Operating System");
    println!("================================================================================");
    println!();
    println!("[BOOT] Kernel entry point reached");
    println!("[BOOT] VGA text mode initialized");
    println!();
    println!("System Status:");
    println!("  - Bootloader: OK");
    println!("  - VGA Buffer: OK");
    println!("  - Basic I/O: OK");
    println!();
    println!("================================================================================");
    println!("Ready for subsystem initialization.");
    println!("Kernel is running in stable mode.");
    println!("================================================================================");
    println!();

    // Show kernel information
    println!("Kernel Features:");
    println!("  [x] VGA Text Output");
    println!("  [ ] GDT (Global Descriptor Table)");
    println!("  [ ] IDT (Interrupt Descriptor Table)");
    println!("  [ ] Memory Management");
    println!("  [ ] Process Scheduling");
    println!("  [ ] Network Stack");
    println!("  [ ] GPU Acceleration");
    println!("  [ ] Desktop Environment");
    println!();
    println!("Press Ctrl+Alt+G to release mouse from QEMU");
    println!();

    // Stable HLT loop
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!();
    println!("================================================================================");
    println!("                            KERNEL PANIC");
    println!("================================================================================");
    println!("{}", info);
    println!("================================================================================");

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}