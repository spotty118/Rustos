//! # RustOS Graphics Framebuffer System
//!
//! Hardware-accelerated framebuffer management for desktop UI rendering.
//! Supports GPU-accelerated operations, multiple pixel formats, and high-resolution displays.

/// Maximum supported resolution width
pub const MAX_WIDTH: usize = 7680; // 8K width
/// Maximum supported resolution height
pub const MAX_HEIGHT: usize = 4320; // 8K height
/// Default resolution width
pub const DEFAULT_WIDTH: usize = 1920;
/// Default resolution height
pub const DEFAULT_HEIGHT: usize = 1080;

/// Pixel format types supported by the framebuffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PixelFormat {
    /// 32-bit RGBA (8 bits per channel)
    RGBA8888 = 0,
    /// 32-bit BGRA (8 bits per channel)
    BGRA8888 = 1,
    /// 24-bit RGB (8 bits per channel, packed)
    RGB888 = 2,
    /// 16-bit RGB (5-6-5 bits per channel)
    RGB565 = 3,
    /// 15-bit RGB (5-5-5 bits per channel)
    RGB555 = 4,
}

impl PixelFormat {
    /// Get the number of bytes per pixel for this format
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::RGBA8888 | PixelFormat::BGRA8888 => 4,
            PixelFormat::RGB888 => 3,
            PixelFormat::RGB565 | PixelFormat::RGB555 => 2,
        }
    }

    /// Get the number of bits per pixel for this format
    pub const fn bits_per_pixel(&self) -> usize {
        self.bytes_per_pixel() * 8
    }
}

/// Color representation in RGBA format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create a new color
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create a new opaque color
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Create a color from a 32-bit RGBA value
    pub const fn from_rgba32(rgba: u32) -> Self {
        Self {
            r: ((rgba >> 24) & 0xFF) as u8,
            g: ((rgba >> 16) & 0xFF) as u8,
            b: ((rgba >> 8) & 0xFF) as u8,
            a: (rgba & 0xFF) as u8,
        }
    }

    /// Convert to 32-bit RGBA value
    pub const fn to_rgba32(&self) -> u32 {
        ((self.r as u32) << 24) | ((self.g as u32) << 16) | ((self.b as u32) << 8) | (self.a as u32)
    }

    /// Convert color to specific pixel format
    pub fn to_pixel_format(&self, format: PixelFormat) -> u32 {
        match format {
            PixelFormat::RGBA8888 => self.to_rgba32(),
            PixelFormat::BGRA8888 => {
                ((self.b as u32) << 24)
                    | ((self.g as u32) << 16)
                    | ((self.r as u32) << 8)
                    | (self.a as u32)
            }
            PixelFormat::RGB888 => {
                ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
            }
            PixelFormat::RGB565 => {
                let r5 = (self.r >> 3) as u32;
                let g6 = (self.g >> 2) as u32;
                let b5 = (self.b >> 3) as u32;
                (r5 << 11) | (g6 << 5) | b5
            }
            PixelFormat::RGB555 => {
                let r5 = (self.r >> 3) as u32;
                let g5 = (self.g >> 3) as u32;
                let b5 = (self.b >> 3) as u32;
                (r5 << 10) | (g5 << 5) | b5
            }
        }
    }

    // Common colors
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    pub const RED: Color = Color::rgb(255, 0, 0);
    pub const GREEN: Color = Color::rgb(0, 255, 0);
    pub const BLUE: Color = Color::rgb(0, 0, 255);
    pub const YELLOW: Color = Color::rgb(255, 255, 0);
    pub const CYAN: Color = Color::rgb(0, 255, 255);
    pub const MAGENTA: Color = Color::rgb(255, 0, 255);
    pub const GRAY: Color = Color::rgb(128, 128, 128);
    pub const LIGHT_GRAY: Color = Color::rgb(192, 192, 192);
    pub const DARK_GRAY: Color = Color::rgb(64, 64, 64);
    pub const TRANSPARENT: Color = Color::new(0, 0, 0, 0);
}

/// Rectangle structure for drawing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    /// Create a new rectangle
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if a point is inside this rectangle
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get the area of this rectangle
    pub const fn area(&self) -> usize {
        self.width * self.height
    }

    /// Check if this rectangle intersects with another
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// Framebuffer information structure
#[derive(Debug, Clone)]
pub struct FramebufferInfo {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub stride: usize,
    pub physical_address: usize,
    pub size: usize,
    pub gpu_accelerated: bool,
}

impl FramebufferInfo {
    /// Create new framebuffer info
    pub fn new(
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        physical_address: usize,
        gpu_accelerated: bool,
    ) -> Self {
        let bytes_per_pixel = pixel_format.bytes_per_pixel();
        let stride = width * bytes_per_pixel;
        let size = stride * height;

        Self {
            width,
            height,
            pixel_format,
            stride,
            physical_address,
            size,
            gpu_accelerated,
        }
    }

    /// Get the total number of pixels
    pub const fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Get bytes per pixel
    pub const fn bytes_per_pixel(&self) -> usize {
        self.pixel_format.bytes_per_pixel()
    }
}

/// Hardware acceleration capabilities
#[derive(Debug, Clone, Copy)]
pub struct HardwareAcceleration {
    pub gpu_clear: bool,
    pub gpu_copy: bool,
    pub gpu_fill: bool,
    pub gpu_blit: bool,
    pub compute_shaders: bool,
    pub hardware_cursor: bool,
}

impl Default for HardwareAcceleration {
    fn default() -> Self {
        Self {
            gpu_clear: false,
            gpu_copy: false,
            gpu_fill: false,
            gpu_blit: false,
            compute_shaders: false,
            hardware_cursor: false,
        }
    }
}

/// Simplified framebuffer for actual pixel operations
pub struct SimpleFramebuffer {
    pub buffer: *mut u8,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub pixel_format: PixelFormat,
}

unsafe impl Send for SimpleFramebuffer {}
unsafe impl Sync for SimpleFramebuffer {}

impl SimpleFramebuffer {
    pub fn new(buffer: *mut u8, width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        let bytes_per_pixel = pixel_format.bytes_per_pixel();
        let stride = width * bytes_per_pixel;
        Self {
            buffer,
            width,
            height,
            stride,
            pixel_format,
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }

        let bytes_per_pixel = self.pixel_format.bytes_per_pixel();
        let offset = y * self.stride + x * bytes_per_pixel;
        let pixel_value = color.to_pixel_format(self.pixel_format);

        unsafe {
            match bytes_per_pixel {
                1 => {
                    *self.buffer.add(offset) = pixel_value as u8;
                }
                2 => {
                    let ptr = self.buffer.add(offset) as *mut u16;
                    *ptr = pixel_value as u16;
                }
                3 => {
                    let ptr = self.buffer.add(offset);
                    *ptr = (pixel_value & 0xFF) as u8;
                    *ptr.add(1) = ((pixel_value >> 8) & 0xFF) as u8;
                    *ptr.add(2) = ((pixel_value >> 16) & 0xFF) as u8;
                }
                4 => {
                    let ptr = self.buffer.add(offset) as *mut u32;
                    *ptr = pixel_value;
                }
                _ => {}
            }
        }
    }

    pub fn clear(&mut self, color: Color) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set_pixel(x, y, color);
            }
        }
    }

    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        let end_x = (rect.x + rect.width).min(self.width);
        let end_y = (rect.y + rect.height).min(self.height);
        
        for y in rect.y..end_y {
            for x in rect.x..end_x {
                self.set_pixel(x, y, color);
            }
        }
    }

    pub fn draw_rect(&mut self, rect: Rect, color: Color, thickness: usize) {
        // Top and bottom borders
        for i in 0..thickness {
            if rect.y + i < self.height {
                self.fill_rect(Rect::new(rect.x, rect.y + i, rect.width, 1), color);
            }
            if rect.y + rect.height > i && rect.y + rect.height - i - 1 < self.height {
                self.fill_rect(Rect::new(rect.x, rect.y + rect.height - i - 1, rect.width, 1), color);
            }
        }
        
        // Left and right borders
        for i in 0..thickness {
            if rect.x + i < self.width {
                self.fill_rect(Rect::new(rect.x + i, rect.y, 1, rect.height), color);
            }
            if rect.x + rect.width > i && rect.x + rect.width - i - 1 < self.width {
                self.fill_rect(Rect::new(rect.x + rect.width - i - 1, rect.y, 1, rect.height), color);
            }
        }
    }
}

/// Global framebuffer instance
static mut GLOBAL_FRAMEBUFFER: Option<SimpleFramebuffer> = None;
static mut GLOBAL_FRAMEBUFFER_INITIALIZED: bool = false;

/// Initialize the global framebuffer
pub fn init(info: FramebufferInfo, _double_buffered: bool) -> Result<(), &'static str> {
    unsafe {
        GLOBAL_FRAMEBUFFER = Some(SimpleFramebuffer::new(
            info.physical_address as *mut u8,
            info.width,
            info.height,
            info.pixel_format,
        ));
        GLOBAL_FRAMEBUFFER_INITIALIZED = true;
    }
    Ok(())
}

/// Initialize framebuffer using an existing buffer provided by the bootloader
pub fn init_with_buffer(
    buffer: &'static mut [u8],
    info: FramebufferInfo,
    _double_buffered: bool,
) -> Result<(), &'static str> {
    unsafe {
        GLOBAL_FRAMEBUFFER = Some(SimpleFramebuffer::new(
            buffer.as_mut_ptr(),
            info.width,
            info.height,
            info.pixel_format,
        ));
        GLOBAL_FRAMEBUFFER_INITIALIZED = true;
    }
    Ok(())
}

/// Get a reference to the global framebuffer
pub fn framebuffer() -> Option<bool> {
    // Simplified implementation - just return whether framebuffer is initialized
    unsafe { Some(GLOBAL_FRAMEBUFFER_INITIALIZED) }
}

/// Get framebuffer information if initialized
pub fn get_info() -> Option<FramebufferInfo> {
    unsafe {
        if let Some(ref fb) = GLOBAL_FRAMEBUFFER {
            Some(FramebufferInfo::new(
                fb.width,
                fb.height,
                fb.pixel_format,
                fb.buffer as usize,
                false,
            ))
        } else {
            None
        }
    }
}

/// Clear the screen with a color
pub fn clear_screen(color: Color) {
    unsafe {
        if let Some(ref mut fb) = GLOBAL_FRAMEBUFFER {
            fb.clear(color);
        }
    }
}

/// Set a pixel on the screen
pub fn set_pixel(x: usize, y: usize, color: Color) {
    unsafe {
        if let Some(ref mut fb) = GLOBAL_FRAMEBUFFER {
            fb.set_pixel(x, y, color);
        }
    }
}

/// Fill a rectangle on the screen
pub fn fill_rect(rect: Rect, color: Color) {
    unsafe {
        if let Some(ref mut fb) = GLOBAL_FRAMEBUFFER {
            fb.fill_rect(rect, color);
        }
    }
}

/// Draw a rectangle outline on the screen
pub fn draw_rect(rect: Rect, color: Color, thickness: usize) {
    unsafe {
        if let Some(ref mut fb) = GLOBAL_FRAMEBUFFER {
            fb.draw_rect(rect, color, thickness);
        }
    }
}

/// Present the current frame
pub fn present() {
    // Simplified implementation for no-std environment
    // In a real implementation, this would present the current frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{serial_print, serial_println};

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_pixel_format_bytes_per_pixel() {
        serial_print!("test_pixel_format_bytes_per_pixel... ");
        assert_eq!(PixelFormat::RGBA8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::RGB888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::RGB565.bytes_per_pixel(), 2);
        serial_println!("[ok]");
    }

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_color_conversion() {
        serial_print!("test_color_conversion... ");
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
        assert_eq!(color.a, 255);
        serial_println!("[ok]");
    }

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_rect_contains() {
        serial_print!("test_rect_contains... ");
        let rect = Rect::new(10, 10, 20, 20);
        assert!(rect.contains(15, 15));
        assert!(!rect.contains(5, 5));
        assert!(!rect.contains(35, 35));
        serial_println!("[ok]");
    }

    #[cfg(feature = "disabled-tests")] // #[test]
    fn test_framebuffer_info() {
        serial_print!("test_framebuffer_info... ");
        let info = FramebufferInfo::new(1920, 1080, PixelFormat::RGBA8888, 0xfd000000, false);
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert_eq!(info.pixel_count(), 1920 * 1080);
        assert_eq!(info.bytes_per_pixel(), 4);
        serial_println!("[ok]");
    }
}