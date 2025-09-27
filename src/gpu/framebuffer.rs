/// GPU-accelerated framebuffer for desktop UI rendering
/// Provides hardware-accelerated drawing operations

use crate::gpu::GPUCapabilities;

/// Pixel format for framebuffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    RGB888,   // 24-bit RGB
    RGBA8888, // 32-bit RGBA
    BGR888,   // 24-bit BGR (common on Intel)
    BGRA8888, // 32-bit BGRA (common on Windows)
}

impl PixelFormat {
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::RGB888 | PixelFormat::BGR888 => 3,
            PixelFormat::RGBA8888 | PixelFormat::BGRA8888 => 4,
        }
    }
}

/// Color representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
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
    
    /// Convert color to 32-bit integer representation
    pub fn to_u32(&self, format: PixelFormat) -> u32 {
        match format {
            PixelFormat::RGBA8888 => {
                ((self.r as u32) << 24) | ((self.g as u32) << 16) | ((self.b as u32) << 8) | (self.a as u32)
            }
            PixelFormat::BGRA8888 => {
                ((self.b as u32) << 24) | ((self.g as u32) << 16) | ((self.r as u32) << 8) | (self.a as u32)
            }
            PixelFormat::RGB888 => {
                ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
            }
            PixelFormat::BGR888 => {
                ((self.b as u32) << 16) | ((self.g as u32) << 8) | (self.r as u32)
            }
        }
    }
}

/// Rectangle structure for drawing operations
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }
    
    pub fn contains_point(&self, x: u32, y: u32) -> bool {
        x >= self.x && 
        x < self.x + self.width && 
        y >= self.y && 
        y < self.y + self.height
    }
    
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width &&
        self.x + self.width > other.x &&
        self.y < other.y + other.height &&
        self.y + self.height > other.y
    }
}

/// GPU-accelerated framebuffer
pub struct Framebuffer {
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    buffer_address: usize,  // Changed to usize for thread safety
    buffer_size: usize,
    pitch: usize,  // bytes per row
    is_hardware_accelerated: bool,
}

// Manual Send and Sync implementation since we know this is safe in kernel context
unsafe impl Send for Framebuffer {}
unsafe impl Sync for Framebuffer {}

impl Framebuffer {
    /// Create new framebuffer for the given GPU
    pub fn new(gpu: &GPUCapabilities) -> Result<Self, &'static str> {
        // For demonstration, we'll use a common resolution
        let width = 1920;
        let height = 1080;
        let pixel_format = match gpu.vendor {
            crate::gpu::GPUVendor::Intel => PixelFormat::BGRA8888,  // Intel prefers BGR
            crate::gpu::GPUVendor::Nvidia => PixelFormat::RGBA8888, // NVIDIA prefers RGB  
            crate::gpu::GPUVendor::AMD => PixelFormat::RGBA8888,    // AMD prefers RGB
            crate::gpu::GPUVendor::Unknown => PixelFormat::RGBA8888,
        };
        
        let bytes_per_pixel = pixel_format.bytes_per_pixel();
        let pitch = width as usize * bytes_per_pixel;
        let buffer_size = pitch * height as usize;
        
        // In a real implementation, this would map GPU memory
        // For now, we'll simulate with address 0 (not actually used)
        let buffer_address = 0;
        
        crate::println!("[Framebuffer] Created {}x{} framebuffer ({:?}, {} bytes)", 
                       width, height, pixel_format, buffer_size);
        
        Ok(Self {
            width,
            height,
            pixel_format,
            buffer_address,
            buffer_size,
            pitch,
            is_hardware_accelerated: gpu.supports_2d_accel,
        })
    }
    
    pub fn width(&self) -> u32 {
        self.width
    }
    
    pub fn height(&self) -> u32 {
        self.height
    }
    
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
    
    pub fn is_hardware_accelerated(&self) -> bool {
        self.is_hardware_accelerated
    }
    
    /// Clear entire framebuffer with specified color
    pub fn clear(&mut self, color: u32) {
        if self.is_hardware_accelerated {
            self.hw_clear(color);
        } else {
            self.sw_clear(color);
        }
    }
    
    /// Draw a filled rectangle
    pub fn draw_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: u32) {
        let rect = Rect::new(x, y, width, height);
        
        if self.is_hardware_accelerated {
            self.hw_draw_rect(&rect, color);
        } else {
            self.sw_draw_rect(&rect, color);
        }
    }
    
    /// Draw a filled rectangle with Color struct
    pub fn draw_rect_color(&mut self, rect: &Rect, color: Color) {
        let color_u32 = color.to_u32(self.pixel_format);
        self.draw_rect(rect.x, rect.y, rect.width, rect.height, color_u32);
    }
    
    /// Draw a line between two points
    pub fn draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: u32) {
        if self.is_hardware_accelerated {
            self.hw_draw_line(x1, y1, x2, y2, color);
        } else {
            self.sw_draw_line(x1, y1, x2, y2, color);
        }
    }
    
    /// Set a single pixel
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        
        if self.is_hardware_accelerated {
            self.hw_set_pixel(x, y, color);
        } else {
            self.sw_set_pixel(x, y, color);
        }
    }
    
    /// Present/flush framebuffer to display
    pub fn present(&mut self) {
        if self.is_hardware_accelerated {
            self.hw_present();
        } else {
            self.sw_present();
        }
    }
    
    /// Hardware-accelerated clear operation
    fn hw_clear(&mut self, color: u32) {
        crate::println!("[Framebuffer] HW Clear: 0x{:08X}", color);
        // In a real implementation, this would use GPU 2D acceleration
        // For now, we'll just log the operation
    }
    
    /// Software fallback clear operation
    fn sw_clear(&mut self, color: u32) {
        crate::println!("[Framebuffer] SW Clear: 0x{:08X}", color);
        // Software implementation would fill buffer manually
    }
    
    /// Hardware-accelerated rectangle drawing
    fn hw_draw_rect(&mut self, rect: &Rect, color: u32) {
        crate::println!("[Framebuffer] HW Rect: ({},{}) {}x{} = 0x{:08X}", 
                       rect.x, rect.y, rect.width, rect.height, color);
        // GPU 2D engine would handle this
    }
    
    /// Software fallback rectangle drawing
    fn sw_draw_rect(&mut self, rect: &Rect, color: u32) {
        crate::println!("[Framebuffer] SW Rect: ({},{}) {}x{} = 0x{:08X}", 
                       rect.x, rect.y, rect.width, rect.height, color);
        // Manual pixel-by-pixel drawing
    }
    
    /// Hardware-accelerated line drawing
    fn hw_draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: u32) {
        crate::println!("[Framebuffer] HW Line: ({},{}) -> ({},{}) = 0x{:08X}", 
                       x1, y1, x2, y2, color);
        // GPU line drawing acceleration
    }
    
    /// Software fallback line drawing (Bresenham's algorithm)
    fn sw_draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: u32) {
        crate::println!("[Framebuffer] SW Line: ({},{}) -> ({},{}) = 0x{:08X}", 
                       x1, y1, x2, y2, color);
        // Bresenham line algorithm implementation
    }
    
    /// Hardware-accelerated pixel setting
    fn hw_set_pixel(&mut self, _x: u32, _y: u32, _color: u32) {
        // GPU pixel operation
    }
    
    /// Software fallback pixel setting
    fn sw_set_pixel(&mut self, _x: u32, _y: u32, _color: u32) {
        // Direct memory write to buffer
    }
    
    /// Hardware-accelerated present/flush
    fn hw_present(&mut self) {
        crate::println!("[Framebuffer] HW Present - GPU scanout");
        // Trigger GPU display scanout
    }
    
    /// Software fallback present
    fn sw_present(&mut self) {
        crate::println!("[Framebuffer] SW Present - VGA fallback");
        // Copy to VGA buffer or trigger display update
    }
}

/// Desktop UI drawing utilities
pub struct DesktopUI;

impl DesktopUI {
    /// Draw a simple window frame
    pub fn draw_window(fb: &mut Framebuffer, rect: &Rect, title: &str) {
        let title_bar_height = 30;
        let border_width = 2;
        
        // Window background
        fb.draw_rect_color(rect, Color::LIGHT_GRAY);
        
        // Title bar
        let title_rect = Rect::new(
            rect.x, 
            rect.y, 
            rect.width, 
            title_bar_height
        );
        fb.draw_rect_color(&title_rect, Color::BLUE);
        
        // Window border
        // Top
        fb.draw_rect_color(&Rect::new(rect.x, rect.y, rect.width, border_width), Color::DARK_GRAY);
        // Bottom
        fb.draw_rect_color(&Rect::new(rect.x, rect.y + rect.height - border_width, rect.width, border_width), Color::DARK_GRAY);
        // Left
        fb.draw_rect_color(&Rect::new(rect.x, rect.y, border_width, rect.height), Color::DARK_GRAY);
        // Right
        fb.draw_rect_color(&Rect::new(rect.x + rect.width - border_width, rect.y, border_width, rect.height), Color::DARK_GRAY);
        
        crate::println!("[DesktopUI] Drew window '{}' at ({},{}) {}x{}", 
                       title, rect.x, rect.y, rect.width, rect.height);
    }
    
    /// Draw a simple button
    pub fn draw_button(fb: &mut Framebuffer, rect: &Rect, text: &str, pressed: bool) {
        let color = if pressed { Color::GRAY } else { Color::LIGHT_GRAY };
        let border_color = if pressed { Color::DARK_GRAY } else { Color::WHITE };
        
        // Button background
        fb.draw_rect_color(rect, color);
        
        // Button border (simple 1px)
        // Top and left (highlight)
        fb.draw_rect_color(&Rect::new(rect.x, rect.y, rect.width, 1), border_color);
        fb.draw_rect_color(&Rect::new(rect.x, rect.y, 1, rect.height), border_color);
        
        // Bottom and right (shadow)
        let shadow_color = if pressed { Color::WHITE } else { Color::DARK_GRAY };
        fb.draw_rect_color(&Rect::new(rect.x, rect.y + rect.height - 1, rect.width, 1), shadow_color);
        fb.draw_rect_color(&Rect::new(rect.x + rect.width - 1, rect.y, 1, rect.height), shadow_color);
        
        crate::println!("[DesktopUI] Drew button '{}' at ({},{}) {}x{} (pressed: {})", 
                       text, rect.x, rect.y, rect.width, rect.height, pressed);
    }
    
    /// Draw a desktop with taskbar
    pub fn draw_desktop(fb: &mut Framebuffer) {
        // Desktop background
        fb.clear(Color::BLUE.to_u32(fb.pixel_format())); // Classic blue desktop
        
        // Taskbar at bottom
        let taskbar_height = 40;
        let taskbar_rect = Rect::new(
            0, 
            fb.height() - taskbar_height, 
            fb.width(), 
            taskbar_height
        );
        fb.draw_rect_color(&taskbar_rect, Color::GRAY);
        
        // Start button
        let start_button = Rect::new(5, fb.height() - taskbar_height + 5, 80, 30);
        Self::draw_button(fb, &start_button, "Start", false);
        
        crate::println!("[DesktopUI] Drew desktop with taskbar");
    }
}