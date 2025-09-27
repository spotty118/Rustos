#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(rustos::test_runner)]
#![reexport_test_harness_main = "test_main"]

use core::panic::PanicInfo;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    rustos::println!("Welcome to RustOS - An AI-Powered Operating System!");
    rustos::println!("Initializing AI kernel components...");
    
    rustos::init();
    
    // Initialize AI subsystem
    rustos::ai::init_ai_system();
    
    rustos::println!("RustOS AI kernel successfully initialized!");
    rustos::println!("AI inference engine status: {}", rustos::ai::get_ai_status());
    
    #[cfg(test)]
    test_main();
    
    rustos::println!("RustOS kernel is running...");
    rustos::hlt_loop();
}

/// This function is called on panic.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    rustos::println!("KERNEL PANIC: {}", info);
    rustos::hlt_loop();
}