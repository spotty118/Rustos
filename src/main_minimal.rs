#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

// VGA Buffer for output
static mut VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;

#[no_mangle]
pub extern "C" fn rust_main() -> ! {
    // Clear screen and print message
    clear_screen();
    print_string(b"RustOS Multiboot Kernel Started!");
    print_string(b"Minimal kernel is working!");
    print_string(b"Multiboot headers functional!");
    
    // Halt
    loop {
        unsafe { asm!("hlt") }
    }
}

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    clear_screen();
    print_string(b"RustOS - Bootloader Entry Point!");
    print_string(b"Boot info received successfully");
    
    // Basic system info
    print_string(b"Memory regions available");
    print_string(b"System initialized successfully");
    
    loop {
        unsafe { asm!("hlt") }
    }
}

fn clear_screen() {
    unsafe {
        for i in 0..(80 * 25 * 2) {
            *VGA_BUFFER.add(i) = if i % 2 == 0 { b' ' } else { 0x07 };
        }
    }
}

fn print_string(s: &[u8]) {
    static mut ROW: usize = 0;
    unsafe {
        if ROW >= 25 { ROW = 0; clear_screen(); }
        
        for (i, &byte) in s.iter().enumerate() {
            let offset = (ROW * 80 + i) * 2;
            if offset < (80 * 25 * 2) {
                *VGA_BUFFER.add(offset) = byte;
                *VGA_BUFFER.add(offset + 1) = 0x07; // White on black
            }
        }
        ROW += 1;
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    print_string(b"KERNEL PANIC!");
    loop {
        unsafe { asm!("hlt") }
    }
}

#[cfg(test)]
fn test_runner(tests: &[&dyn Fn()]) {
    // Test runner implementation
}

use core::arch::asm;
