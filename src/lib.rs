//! RustOS Library Crate
//!
//! This library exposes the core functionality of RustOS for testing and external use.

#![no_std]
#![feature(custom_test_frameworks)]
#![feature(alloc_error_handler)]
#![feature(abi_x86_interrupt)]
#![cfg_attr(test, test_runner(crate::test_runner))]
#![cfg_attr(test, reexport_test_harness_main = "test_main")]

extern crate alloc;

use linked_list_allocator::LockedHeap;

// Re-export alloc types and traits for modules to use
pub use alloc::boxed::Box;
pub use alloc::collections::BTreeMap;
pub use alloc::string::String;
pub use alloc::string::ToString;
pub use alloc::vec::Vec;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Initialize the heap allocator with the given memory range
pub fn init_heap(heap_start: usize, heap_size: usize) -> Result<(), &'static str> {
    unsafe {
        ALLOCATOR.lock().init(heap_start, heap_size);
    }
    Ok(())
}

// Alloc error handler for no_std environment
#[alloc_error_handler]
fn alloc_error_handler(layout: core::alloc::Layout) -> ! {
    panic!("allocation error: {:?}", layout)
}

// Re-export all modules
pub mod desktop;
pub mod drivers;
pub mod graphics;
pub mod network;

// Core kernel systems
pub mod memory;
pub mod interrupts;
pub mod gdt;
pub mod process;

// PCI bus enumeration and management
pub mod pci;

// AI module - create a basic inference engine for the tests
pub mod ai {
    pub mod inference_engine {
        use alloc::vec::Vec;

        #[derive(Debug)]
        pub struct InferenceEngine {
            rules: Vec<InferenceRule>,
            initialized: bool,
        }

        #[derive(Debug, Clone)]
        pub struct InferenceRule {
            pattern: [f32; 8],
            confidence: f32,
            rule_id: u32,
        }

        impl InferenceEngine {
            pub fn new() -> Self {
                Self {
                    rules: Vec::new(),
                    initialized: false,
                }
            }

            pub fn initialize(&mut self) -> Result<(), &'static str> {
                // Add some basic rules for testing
                self.rules.push(InferenceRule::new(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    0.9,
                    1,
                ));
                self.rules.push(InferenceRule::new(
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                    0.8,
                    2,
                ));
                self.initialized = true;
                Ok(())
            }

            pub fn get_rules_count(&self) -> usize {
                self.rules.len()
            }

            pub fn infer(&self, input: &[f32; 8]) -> Result<f32, &'static str> {
                if !self.initialized {
                    return Err("Engine not initialized");
                }

                let mut max_confidence = 0.0f32;
                for rule in &self.rules {
                    let similarity = rule.matches(input);
                    let weighted_confidence = similarity * rule.confidence;
                    if weighted_confidence > max_confidence {
                        max_confidence = weighted_confidence;
                    }
                }

                Ok(max_confidence)
            }
        }

        impl InferenceRule {
            pub fn new(pattern: [f32; 8], confidence: f32, rule_id: u32) -> Self {
                Self {
                    pattern,
                    confidence,
                    rule_id,
                }
            }

            pub fn matches(&self, input: &[f32; 8]) -> f32 {
                // Calculate cosine similarity
                let mut dot_product = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;

                for i in 0..8 {
                    dot_product += self.pattern[i] * input[i];
                    norm_a += self.pattern[i] * self.pattern[i];
                    norm_b += input[i] * input[i];
                }

                if norm_a == 0.0 || norm_b == 0.0 {
                    return 0.0;
                }

                // Simple sqrt approximation for no_std
                let sqrt_norm_a = Self::sqrt_approx(norm_a);
                let sqrt_norm_b = Self::sqrt_approx(norm_b);
                dot_product / (sqrt_norm_a * sqrt_norm_b)
            }

            // Simple square root approximation using Newton's method
            fn sqrt_approx(x: f32) -> f32 {
                if x <= 0.0 {
                    return 0.0;
                }
                let mut guess = x / 2.0;
                for _ in 0..10 {
                    // 10 iterations should be enough for reasonable precision
                    guess = (guess + x / guess) / 2.0;
                }
                guess
            }
        }
    }
}

// Test runner function - available for integration tests
pub fn test_runner(tests: &[&dyn Fn()]) {
    serial_println!("RustOS Lib.rs Test Mode");
    serial_println!("Running {} tests", tests.len());
    for test in tests {
        test();
    }
    serial_println!("All library tests completed!");
    exit_qemu(QemuExitCode::Success);
}

// Serial output for testing
use lazy_static::lazy_static;
use spin::Mutex;
use uart_16550::SerialPort;

lazy_static! {
    pub static ref SERIAL1: Mutex<SerialPort> = {
        let mut serial_port = unsafe { SerialPort::new(0x3F8) };
        serial_port.init();
        Mutex::new(serial_port)
    };
}

#[doc(hidden)]
pub fn _print(args: core::fmt::Arguments) {
    use core::fmt::Write;
    SERIAL1
        .lock()
        .write_fmt(args)
        .expect("Printing to serial failed");
}

// Re-export alloc macros for modules
pub use alloc::{format, vec};

// Note: Tests in this codebase use #[cfg(feature = "disabled-tests")] // #[test] attribute with custom test framework

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::_print(format_args!($($arg)*));
    };
}

#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($fmt:expr) => ($crate::serial_print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::serial_print!(concat!($fmt, "\n"), $($arg)*));
}

#[macro_export]
macro_rules! println {
    () => ($crate::serial_println!());
    ($fmt:expr) => ($crate::serial_println!($fmt));
    ($fmt:expr, $($arg:tt)*) => ($crate::serial_println!($fmt, $($arg)*));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

pub fn exit_qemu(exit_code: QemuExitCode) {
    use x86_64::instructions::port::Port;

    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
}

// Panic handler for tests
#[cfg(test)]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial_println!("[failed]\n");
    serial_println!("Error: {}\n", info);
    exit_qemu(QemuExitCode::Failed);
    loop {}
}

#[cfg(test)]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    test_main();
    loop {}
}
