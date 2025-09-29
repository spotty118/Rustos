//! RustOS Minimal Bootable Kernel
//!
//! Simple, working kernel that boots reliably

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{entry_point, BootInfo};

// Only include the minimal module we need
mod vga_buffer;

use vga_buffer::VGA_WRITER;

// Kernel entry point
entry_point!(kernel_entry);

#[cfg(not(test))]
fn kernel_entry(_boot_info: &'static BootInfo) -> ! {
    // Simple, stable kernel
    {
        let mut writer = VGA_WRITER.lock();
        writer.clear_screen();
        writer.write_string("🚀 RustOS Minimal Kernel\n");
        writer.write_string("✅ Bootloader working!\n");
        writer.write_string("✅ VGA output working!\n");
        writer.write_string("🎉 SUCCESS: Kernel booted!\n");
        writer.write_string("\nKernel is stable and running.\n");
        writer.write_string("System ready for use.\n");
    }

    // Stable halt loop
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

// Enhanced panic handler
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Try to write panic to VGA if possible
    if let Ok(mut writer) = VGA_WRITER.try_lock() {
        writer.write_string("\n🚨 KERNEL PANIC!\n");
        if let Some(location) = info.location() {
            let _ = core::fmt::write(&mut writer, format_args!(
                "at {}:{}:{}\n",
                location.file(),
                location.line(),
                location.column()
            ));
        }
        if let Some(payload) = info.payload().downcast_ref::<&str>() {
            let _ = core::fmt::write(&mut writer, format_args!("Reason: {}\n", payload));
        }
        writer.write_string("System halted.\n");
    }

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

// Minimal test framework
#[cfg(test)]
fn test_runner(tests: &[&dyn Fn()]) {
    for test in tests {
        test();
    }
}

#[cfg(test)]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}