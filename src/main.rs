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
// Include basic memory management
mod memory_basic;
// Include visual boot display
mod boot_display;
// Include keyboard input handler
mod keyboard;
// Include desktop environment
mod simple_desktop;
// Include graphics system
mod graphics;
// Include advanced desktop environment  
mod desktop;

// VGA_WRITER is now used via macros in print module

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

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize VGA buffer
    vga_buffer::init();

    // Show brief boot sequence
    boot_display::show_boot_logo();
    boot_display::boot_delay();

    // Quick initialization sequence
    boot_display::show_boot_progress(1, 4, "Initializing Hardware");
    boot_display::boot_delay();

    // Initialize basic memory management
    let physical_memory_offset = x86_64::VirtAddr::new(0);
    let _memory_stats = match memory_basic::init_memory(
        &boot_info.memory_map,
        physical_memory_offset,
    ) {
        Ok(stats) => {
            boot_display::show_boot_progress(2, 4, "Memory Management Ready");
            stats
        }
        Err(_) => {
            boot_display::show_boot_progress(2, 4, "Memory Management Basic");
            memory_basic::MemoryStats {
                total_memory: 512 * 1024 * 1024,
                usable_memory: 256 * 1024 * 1024,
                memory_regions: 5,
            }
        }
    };

    boot_display::show_boot_progress(3, 4, "Starting Keyboard System");
    keyboard::init();
    boot_display::boot_delay();

    boot_display::show_boot_progress(4, 4, "Launching Desktop Environment");
    boot_display::boot_delay();

    println!("🚀 RustOS Desktop Selection");
    println!("Current kernel can boot to either:");
    println!("1. Simple Text Desktop (MS-DOS style) - Old Implementation");  
    println!("2. Modern Graphics Desktop (Current style) - New Implementation");
    println!();
    
    // For demonstration, let's try to initialize graphics but fall back gracefully
    let graphics_initialized = {
        println!("Attempting to initialize modern graphics desktop...");
        
        // Use a safe memory area that won't cause crashes
        let graphics_buffer_addr = 0xC0000; // Safe area in upper memory
        let width = 640;   
        let height = 480;
        
        println!("Setting up {}x{} framebuffer at 0x{:x}", width, height, graphics_buffer_addr);
        
        // Create framebuffer info for our graphics system  
        let fb_info = graphics::FramebufferInfo::new(
            width,
            height,
            graphics::PixelFormat::RGBA8888, // 32-bit color
            graphics_buffer_addr,
            false, // No GPU acceleration for now
        );
        
        // Try to initialize graphics
        match graphics::init(fb_info, false) {
            Ok(()) => {
                println!("✅ Graphics system initialized successfully!");
                println!("🖥️  Modern Desktop Environment Ready");
                true
            }
            Err(e) => {
                println!("❌ Failed to initialize graphics: {}", e);
                println!("⬇️  Falling back to simple desktop");
                false
            }
        }
    };

    println!();
    if graphics_initialized {
        println!("🎨 Launching MODERN DESKTOP ENVIRONMENT");
        println!("   Features:");
        println!("   • Modern gradient backgrounds");
        println!("   • Overlapping windows with shadows");
        println!("   • Glass-effect taskbar and dock");
        println!("   • Contemporary color scheme");
        println!("   • Hardware-accelerated graphics");
        println!();
        
        // Initialize modern desktop environment
        match desktop::setup_full_desktop() {
            Ok(()) => {
                println!("✅ Modern desktop initialized successfully!");
                modern_desktop_main_loop()
            }
            Err(e) => {
                println!("❌ Desktop initialization failed: {}", e);
                println!("⬇️  Falling back to simple desktop");
                simple_desktop::init_desktop();
                desktop_main_loop()
            }
        }
    } else {
        println!("📺 Launching SIMPLE TEXT DESKTOP (MS-DOS Style)");
        println!("   Features:");
        println!("   • Text-based interface");
        println!("   • 80x25 character display");
        println!("   • Basic window simulation");
        println!("   • VGA text mode");
        println!();
        simple_desktop::init_desktop();
        desktop_main_loop()
    }
}

/// Main desktop loop that handles keyboard input and desktop updates
fn desktop_main_loop() -> ! {
    let mut update_counter: u64 = 0;

    loop {
        // Process keyboard events and forward to desktop
        while let Some(key_event) = keyboard::get_key_event() {
            match key_event {
                keyboard::KeyEvent::CharacterPress(c) => {
                    simple_desktop::with_desktop(|desktop| {
                        desktop.handle_key(c as u8);
                    });
                }
                keyboard::KeyEvent::SpecialPress(special_key) => {
                    // Map special keys to desktop key codes
                    let key_code = match special_key {
                        keyboard::SpecialKey::Escape => 27, // ESC
                        keyboard::SpecialKey::Enter => 13,  // Enter
                        keyboard::SpecialKey::Backspace => 8, // Backspace
                        keyboard::SpecialKey::Tab => 9,     // Tab
                        keyboard::SpecialKey::F1 => 112,   // F1
                        keyboard::SpecialKey::F2 => 113,   // F2
                        keyboard::SpecialKey::F3 => 114,   // F3
                        keyboard::SpecialKey::F4 => 115,   // F4
                        keyboard::SpecialKey::F5 => 116,   // F5
                        _ => continue, // Ignore other special keys for now
                    };

                    simple_desktop::with_desktop(|desktop| {
                        desktop.handle_key(key_code);
                    });
                }
                _ => {
                    // Ignore key releases for now
                }
            }
        }

        // Update desktop periodically (for clock and animations)
        if update_counter.is_multiple_of(1_000_000) {
            simple_desktop::with_desktop(|desktop| {
                desktop.update();
            });
        }

        update_counter += 1;

        // Halt CPU until next interrupt to save power
        unsafe { core::arch::asm!("hlt"); }
    }
}

/// Modern desktop loop that handles graphics-based desktop
fn modern_desktop_main_loop() -> ! {
    let mut update_counter: u64 = 0;
    let mut _frame_counter: usize = 0;

    loop {
        // Process keyboard events and forward to desktop
        while let Some(key_event) = keyboard::get_key_event() {
            match key_event {
                keyboard::KeyEvent::CharacterPress(c) => {
                    desktop::handle_key_down(c as u8);
                }
                keyboard::KeyEvent::SpecialPress(special_key) => {
                    // Map special keys to desktop key codes
                    let key_code = match special_key {
                        keyboard::SpecialKey::Escape => 27, // ESC
                        keyboard::SpecialKey::Enter => 13,  // Enter
                        keyboard::SpecialKey::Backspace => 8, // Backspace
                        keyboard::SpecialKey::Tab => 9,     // Tab
                        keyboard::SpecialKey::F1 => 112,   // F1
                        keyboard::SpecialKey::F2 => 113,   // F2
                        keyboard::SpecialKey::F3 => 114,   // F3
                        keyboard::SpecialKey::F4 => 115,   // F4
                        keyboard::SpecialKey::F5 => 116,   // F5
                        _ => continue, // Ignore other special keys for now
                    };
                    
                    desktop::handle_key_down(key_code);
                }
                _ => {
                    // Ignore key releases for now
                }
            }
        }

        // Update desktop periodically
        if update_counter.is_multiple_of(100_000) {
            desktop::update_desktop();
        }

        // Render desktop periodically
        if update_counter.is_multiple_of(200_000) {
            desktop::render_desktop();
            _frame_counter += 1;
        }

        // Check if desktop needs redraw
        if desktop::desktop_needs_redraw() {
            desktop::render_desktop();
            _frame_counter += 1;
        }

        update_counter += 1;

        // Halt CPU until next interrupt to save power
        unsafe { core::arch::asm!("hlt"); }
    }
}

#[no_mangle]
pub extern "C" fn rust_main() -> ! {
    // Alternative entry point - use a static dummy boot info
    // Note: This won't have proper memory map, so memory init will be limited
    static DUMMY_BOOT_INFO: BootInfo = unsafe { core::mem::zeroed() };
    kernel_main(&DUMMY_BOOT_INFO)
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Enhanced panic handler using print macros
    println!();
    println!("🚨 KERNEL PANIC!");
    if let Some(location) = info.location() {
        println!("at {}:{}:{}", location.file(), location.line(), location.column());
    }
    // Note: PanicInfo::payload() is deprecated and doesn't provide useful information
    // The panic message is typically provided via the formatting arguments which are 
    // not directly accessible from PanicInfo in no_std environments
    println!("System halted.");

    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}
