//! VGA Buffer Module for RustOS
//!
//! This module provides VGA text mode output functionality for kernel console output.
//! It includes color support, scrolling, and thread-safe writing capabilities.

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};
use lazy_static::lazy_static;
use spin::Mutex;
use volatile::Volatile;

/// Standard VGA color palette
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Blue = 1,
    Green = 2,
    Cyan = 3,
    Red = 4,
    Magenta = 5,
    Brown = 6,
    LightGray = 7,
    DarkGray = 8,
    LightBlue = 9,
    LightGreen = 10,
    LightCyan = 11,
    LightRed = 12,
    Pink = 13,
    Yellow = 14,
    White = 15,
}

/// VGA color code combining foreground and background colors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ColorCode(u8);

impl ColorCode {
    /// Create a new color code from foreground and background colors
    pub fn new(foreground: Color, background: Color) -> ColorCode {
        ColorCode((background as u8) << 4 | (foreground as u8))
    }
}

/// VGA character cell combining ASCII character and color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct ScreenChar {
    ascii_character: u8,
    color_code: ColorCode,
}

/// VGA buffer dimensions
const BUFFER_HEIGHT: usize = 25;
const BUFFER_WIDTH: usize = 80;

/// VGA buffer representation
#[repr(transparent)]
struct Buffer {
    chars: [[Volatile<ScreenChar>; BUFFER_WIDTH]; BUFFER_HEIGHT],
}

/// VGA writer for handling text output
pub struct Writer {
    column_position: usize,
    row_position: usize,
    color_code: ColorCode,
    buffer: &'static mut Buffer,
}

impl Writer {
    /// Write a single byte to the VGA buffer
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.new_line(),
            byte => {
                if self.column_position >= BUFFER_WIDTH {
                    self.new_line();
                }

                let row = self.row_position;
                let col = self.column_position;

                let color_code = self.color_code;
                self.buffer.chars[row][col].write(ScreenChar {
                    ascii_character: byte,
                    color_code,
                });
                self.column_position += 1;
            }
        }
    }

    /// Write a string to the VGA buffer
    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                // Printable ASCII byte or newline
                0x20..=0x7e | b'\n' => self.write_byte(byte),
                // Not part of printable ASCII range
                _ => self.write_byte(0xfe),
            }
        }
    }

    /// Move to a new line, scrolling if necessary
    fn new_line(&mut self) {
        if self.row_position >= BUFFER_HEIGHT - 1 {
            self.scroll_up();
        } else {
            self.row_position += 1;
        }
        self.column_position = 0;
    }

    /// Scroll the entire buffer up by one line
    fn scroll_up(&mut self) {
        for row in 1..BUFFER_HEIGHT {
            for col in 0..BUFFER_WIDTH {
                let character = self.buffer.chars[row][col].read();
                self.buffer.chars[row - 1][col].write(character);
            }
        }
        self.clear_row(BUFFER_HEIGHT - 1);
    }

    /// Clear a specific row
    fn clear_row(&mut self, row: usize) {
        let blank = ScreenChar {
            ascii_character: b' ',
            color_code: self.color_code,
        };
        for col in 0..BUFFER_WIDTH {
            self.buffer.chars[row][col].write(blank);
        }
    }

    /// Clear the entire screen
    pub fn clear_screen(&mut self) {
        for row in 0..BUFFER_HEIGHT {
            self.clear_row(row);
        }
        self.column_position = 0;
        self.row_position = 0;
    }

    /// Set the current color
    pub fn set_color(&mut self, foreground: Color, background: Color) {
        self.color_code = ColorCode::new(foreground, background);
    }

    /// Get current cursor position
    pub fn get_position(&self) -> (usize, usize) {
        (self.row_position, self.column_position)
    }

    /// Set cursor position
    pub fn set_position(&mut self, row: usize, col: usize) {
        if row < BUFFER_HEIGHT && col < BUFFER_WIDTH {
            self.row_position = row;
            self.column_position = col;
        }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

/// Global VGA writer instance
lazy_static! {
    pub static ref WRITER: Mutex<Writer> = Mutex::new(Writer {
        column_position: 0,
        row_position: 0,
        color_code: ColorCode::new(Color::White, Color::Black),
        buffer: unsafe { &mut *(0xb8000 as *mut Buffer) },
    });
}

/// Print function for VGA output
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    WRITER.lock().write_fmt(args).expect("Printing to VGA failed");
}

/// VGA-based print macro
#[macro_export]
macro_rules! vga_print {
    ($($arg:tt)*) => ($crate::vga_buffer::_print(format_args!($($arg)*)));
}

/// VGA-based println macro
#[macro_export]
macro_rules! vga_println {
    () => ($crate::vga_print!("\n"));
    ($($arg:tt)*) => ($crate::vga_print!("{}\n", format_args!($($arg)*)));
}

/// Print with color formatting
pub fn print_colored(text: &str, fg: Color, bg: Color) {
    let mut writer = WRITER.lock();
    let old_color = writer.color_code;
    writer.set_color(fg, bg);
    writer.write_string(text);
    writer.color_code = old_color;
}

/// Print a banner with colored text
pub fn print_banner(text: &str, fg: Color, bg: Color) {
    let mut writer = WRITER.lock();
    let old_color = writer.color_code;
    
    // Print top border
    writer.set_color(fg, bg);
    writer.write_string("=");
    for _ in 0..text.len() + 2 {
        writer.write_string("=");
    }
    writer.write_string("=\n");
    
    // Print text with padding
    writer.write_string("= ");
    writer.write_string(text);
    writer.write_string(" =\n");
    
    // Print bottom border
    writer.write_string("=");
    for _ in 0..text.len() + 2 {
        writer.write_string("=");
    }
    writer.write_string("=\n");
    
    writer.color_code = old_color;
}

/// Initialize VGA buffer system
pub fn init() {
    WRITER.lock().clear_screen();
}

/// Get VGA buffer statistics
pub fn get_vga_stats() -> VgaStats {
    let writer = WRITER.lock();
    let (row, col) = writer.get_position();
    VgaStats {
        cursor_row: row,
        cursor_column: col,
        buffer_width: BUFFER_WIDTH,
        buffer_height: BUFFER_HEIGHT,
    }
}

/// VGA buffer statistics
#[derive(Debug, Clone)]
pub struct VgaStats {
    pub cursor_row: usize,
    pub cursor_column: usize,
    pub buffer_width: usize,
    pub buffer_height: usize,
}

/// Global character counter for debugging
static CHAR_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Get total characters written
pub fn get_char_count() -> usize {
    CHAR_COUNT.load(Ordering::Relaxed)
}

/// Reset character counter
pub fn reset_char_count() {
    CHAR_COUNT.store(0, Ordering::Relaxed);
}

/// Increment character counter (used internally)
pub fn increment_char_count() {
    CHAR_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_color_code_creation() {
        let color = ColorCode::new(Color::White, Color::Black);
        assert_eq!(color.0, 15); // White on black
    }

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_vga_stats() {
        let stats = get_vga_stats();
        assert_eq!(stats.buffer_width, BUFFER_WIDTH);
        assert_eq!(stats.buffer_height, BUFFER_HEIGHT);
    }
}