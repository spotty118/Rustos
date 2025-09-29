#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};

// Include compiler intrinsics for missing symbols
mod intrinsics;

// VGA buffer for output
const VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;
static mut CURSOR: usize = 0;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    clear_screen();
    println("RustOS - Multiboot Kernel Started!");
    println("======================================");
    println("Boot information received successfully");
    println("System ready for operation");
    
    // Main kernel loop
    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}

#[no_mangle]
pub extern "C" fn rust_main() -> ! {
    clear_screen();
    println("RustOS - Direct Multiboot Entry");
    println("================================");
    println("Kernel booted via multiboot");
    println("System operational");
    
    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}

fn clear_screen() {
    unsafe {
        for i in 0..(80 * 25 * 2) {
            *VGA_BUFFER.add(i) = if i % 2 == 0 { b' ' } else { 0x07 };
        }
        CURSOR = 0;
    }
}

fn println(s: &str) {
    unsafe {
        if CURSOR >= 80 * 25 {
            clear_screen();
        }
        
        let row = CURSOR / 80;
        let col = 0;
        
        for (i, byte) in s.bytes().enumerate() {
            if col + i < 80 && row < 25 {
                let pos = ((row * 80) + col + i) * 2;
                *VGA_BUFFER.add(pos) = byte;
                *VGA_BUFFER.add(pos + 1) = 0x0F; // White on black
            }
        }
        
        CURSOR = (row + 1) * 80;
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    println("KERNEL PANIC!");
    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}
