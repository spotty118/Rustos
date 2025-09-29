//! Boot Display Module for RustOS
//!
//! Provides visual boot logo and enhanced display using VGA text mode

use crate::vga_buffer::{Color, VGA_WRITER};
use crate::{print, println};

/// Display boot logo with ASCII art
pub fn show_boot_logo() {
    // Set colors for the logo
    set_color_temp(Color::LightCyan, Color::Black);

    println!();
    println!("    ██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗");
    println!("    ██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝");
    println!("    ██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗");
    println!("    ██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║");
    println!("    ██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║");
    println!("    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝");
    println!();

    // Reset to default colors
    set_color_temp(Color::White, Color::Black);

    // Add subtitle with different color
    set_color_temp(Color::Yellow, Color::Black);
    print_centered("Advanced Rust Operating System");
    set_color_temp(Color::LightGray, Color::Black);
    print_centered("Version 1.0.0 - Enhanced Edition");
    println!();
}

/// Display boot progress bar
pub fn show_boot_progress(step: usize, total: usize, message: &str) {
    let progress = (step * 50) / total;
    let percentage = (step * 100) / total;

    set_color_temp(Color::LightBlue, Color::Black);
    print!("{}  [", message);

    // Draw progress bar
    set_color_temp(Color::LightGreen, Color::Green);
    for i in 0..progress {
        if i < progress {
            print!("█");
        }
    }

    set_color_temp(Color::DarkGray, Color::Black);
    for _ in progress..50 {
        print!("░");
    }

    set_color_temp(Color::LightBlue, Color::Black);
    println!("] {}%", percentage);

    // Reset colors
    set_color_temp(Color::White, Color::Black);
}

/// Show system information panel
pub fn show_system_info() {
    println!();
    draw_box("System Information", 60);

    set_color_temp(Color::LightCyan, Color::Black);
    println!("  ◆ Architecture: x86_64");
    println!("  ◆ Kernel Type: Microkernel");
    println!("  ◆ Memory Model: 64-bit Linear");
    println!("  ◆ Boot Method: Multiboot2");
    println!("  ◆ Graphics: VGA Text Mode");

    set_color_temp(Color::White, Color::Black);
    draw_line(60);
}

/// Show memory information
pub fn show_memory_info(total_mb: usize, usable_mb: usize, regions: usize) {
    println!();
    draw_box("Memory Configuration", 60);

    set_color_temp(Color::LightGreen, Color::Black);
    println!("  ◇ Total Memory:    {} MB", total_mb);
    println!("  ◇ Usable Memory:   {} MB", usable_mb);
    println!("  ◇ Memory Regions:  {}", regions);
    println!("  ◇ Heap Reserved:   100 MB");

    let usage_percent = if total_mb > 0 { (usable_mb * 100) / total_mb } else { 0 };
    println!("  ◇ Memory Usage:    {}%", usage_percent);

    set_color_temp(Color::White, Color::Black);
    draw_line(60);
}

/// Show kernel services status
pub fn show_services_status() {
    println!();
    draw_box("Kernel Services", 60);

    show_service_status("VGA Text Buffer", true);
    show_service_status("Print Subsystem", true);
    show_service_status("Memory Manager", true);
    show_service_status("Interrupt Handler", false);
    show_service_status("Process Scheduler", false);
    show_service_status("Network Stack", false);

    draw_line(60);
}

/// Show desktop environment startup
pub fn show_desktop_startup() {
    println!();
    set_color_temp(Color::Pink, Color::Black);
    print_centered("┌─────────────────────────────────────┐");
    print_centered("│        Starting Desktop...         │");
    print_centered("└─────────────────────────────────────┘");
    println!();

    set_color_temp(Color::White, Color::Black);

    // Show desktop features
    println!("  Desktop Features:");
    set_color_temp(Color::LightCyan, Color::Black);
    println!("    • Window Management System");
    println!("    • Hardware Accelerated Graphics");
    println!("    • Multi-tasking Environment");
    println!("    • File System Integration");
    println!("    • Network Connectivity");

    set_color_temp(Color::White, Color::Black);
    println!();
}

/// Helper function to show service status
fn show_service_status(service: &str, active: bool) {
    print!("  ► {:<20} ", service);

    if active {
        set_color_temp(Color::LightGreen, Color::Black);
        println!("[ACTIVE]");
    } else {
        set_color_temp(Color::Red, Color::Black);
        println!("[INACTIVE]");
    }

    set_color_temp(Color::White, Color::Black);
}

/// Print text centered on screen
fn print_centered(text: &str) {
    let width = 80; // VGA text mode width
    let padding = (width - text.len()) / 2;

    for _ in 0..padding {
        print!(" ");
    }
    println!("{}", text);
}

/// Draw a decorative box with title
fn draw_box(title: &str, width: usize) {
    // Top border
    set_color_temp(Color::LightBlue, Color::Black);
    print!("  ╔");
    for _ in 0..(width-4) {
        print!("═");
    }
    println!("╗");

    // Title line
    let title_padding = ((width - 4) - title.len()) / 2;
    print!("  ║");
    for _ in 0..title_padding {
        print!(" ");
    }
    set_color_temp(Color::Yellow, Color::Black);
    print!("{}", title);
    set_color_temp(Color::LightBlue, Color::Black);
    for _ in 0..title_padding {
        print!(" ");
    }
    if title.len() % 2 == 1 {
        print!(" "); // Extra space for odd titles
    }
    println!("║");

    // Separator
    print!("  ╠");
    for _ in 0..(width-4) {
        print!("═");
    }
    println!("╣");

    set_color_temp(Color::White, Color::Black);
}

/// Draw bottom line for box
fn draw_line(width: usize) {
    set_color_temp(Color::LightBlue, Color::Black);
    print!("  ╚");
    for _ in 0..(width-4) {
        print!("═");
    }
    println!("╝");
    set_color_temp(Color::White, Color::Black);
}

/// Temporarily set VGA colors (helper function)
fn set_color_temp(foreground: Color, background: Color) {
    let mut writer = VGA_WRITER.lock();
    writer.set_color(foreground, background);
}

/// Show welcome message
pub fn show_welcome_message() {
    println!();
    set_color_temp(Color::LightGreen, Color::Black);
    print_centered("┌───────────────────────────────────────────────────────────────┐");
    print_centered("│                    Welcome to RustOS!                        │");
    print_centered("│                                                               │");
    print_centered("│           Your secure, fast, and reliable OS                 │");
    print_centered("│                  Built with Rust 🦀                         │");
    print_centered("└───────────────────────────────────────────────────────────────┘");
    set_color_temp(Color::White, Color::Black);
    println!();
}

/// Add some delay for visual effect
pub fn boot_delay() {
    // Simple delay loop
    for _ in 0..10_000_000 {
        unsafe {
            core::arch::asm!("nop");
        }
    }
}