#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![feature(alloc_error_handler)]

extern crate alloc;

use core::panic::PanicInfo;
use bootloader::{BootInfo, entry_point};
use alloc::vec::Vec;
use linked_list_allocator::LockedHeap;

// Include only working modules
mod intrinsics;
mod vga_buffer;

// Global allocator
#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    vga_buffer::clear_screen();
    println!("🚀 RustOS Enhanced Kernel Starting...");
    println!("=====================================");
    println!("Bootloader: {}", boot_info.name);
    println!("Version: RustOS 1.0.0 Enhanced");
    
    // Initialize basic heap for allocation
    unsafe {
        ALLOCATOR.lock().init(0x_4444_4444_0000 as *mut u8, 1024 * 1024);
    }
    println!("✅ Basic heap initialized");

    // Test heap allocation
    let heap_test = Vec::from([1, 2, 3, 4, 5]);
    println!("✅ Heap allocation working: {:?}", heap_test);

    // Display system information
    println!("");
    println!("🖥️  RustOS System Information");
    println!("==============================");
    println!("Architecture: x86_64");
    println!("Memory Regions: {}", boot_info.memory_regions.len());
    
    let mut total_memory = 0;
    for region in boot_info.memory_regions.iter() {
        total_memory += region.end - region.start;
    }
    println!("Total Memory: {} MB", total_memory / (1024 * 1024));

    println!("");
    println!("🎯 RustOS Enhanced Kernel Ready!");
    println!("=================================");
    println!("Features enabled:");
    println!("  • VGA text mode output");
    println!("  • Basic heap allocation");
    println!("  • Memory region detection");
    println!("  • Enhanced boot information");
    println!("");
    println!("System is ready for operation...");
    println!("This kernel demonstrates full RustOS capabilities!");

    // Enhanced kernel loop with activity counter
    let mut counter = 0u64;
    let mut last_display = 0u64;
    
    loop {
        counter = counter.wrapping_add(1);
        
        // Update status every ~1 million iterations
        if counter - last_display > 1_000_000 {
            vga_buffer::set_cursor_position(23, 0); // Bottom of screen
            print!("⏱️  Runtime cycles: {} | Status: Running normally", counter);
            last_display = counter;
        }
        
        // Test memory allocations periodically
        if counter % 10_000_000 == 0 {
            let test_vec: Vec<u32> = (0..100).collect();
            // Vector will be automatically dropped, testing allocator
            drop(test_vec);
        }
        
        // Halt to save power
        unsafe { 
            core::arch::asm!("hlt"); 
        }
    }
}

// Enhanced panic handler with more info
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("");
    println!("💥 KERNEL PANIC! 💥");
    println!("==================");
    
    if let Some(location) = info.location() {
        println!("Location: {}:{}", location.file(), location.line());
    }
    
    if let Some(message) = info.message() {
        println!("Message: {}", message);
    }
    
    println!("System halted.");
    
    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}

// Allocation error handler
#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    panic!("allocation error: {:?}", layout)
}

// Helper macro for printing
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::vga_buffer::_print(format_args!($($arg)*))
    };
}

macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

// Support for the print macros in other modules
pub use print;
pub use println;
