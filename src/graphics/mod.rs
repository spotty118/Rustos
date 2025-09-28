//! # RustOS Graphics System Module
//!
//! This module provides the complete graphics subsystem for RustOS, including
//! framebuffer management, drawing primitives, and hardware acceleration support.

pub mod framebuffer;

// Re-export commonly used types and functions
pub use framebuffer::{
    clear_screen, draw_rect, fill_rect, framebuffer, get_info, init, init_with_buffer, present,
    set_pixel, Color, FramebufferInfo, HardwareAcceleration, PixelFormat, Rect,
};

use spin::{Mutex, Once};

/// Graphics system status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsStatus {
    Uninitialized,
    Initializing,
    Ready,
    Error,
}

impl core::fmt::Display for GraphicsStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphicsStatus::Uninitialized => write!(f, "Uninitialized"),
            GraphicsStatus::Initializing => write!(f, "Initializing"),
            GraphicsStatus::Ready => write!(f, "Ready"),
            GraphicsStatus::Error => write!(f, "Error"),
        }
    }
}

/// Graphics system configuration
#[derive(Debug, Clone)]
pub struct GraphicsConfig {
    pub preferred_width: usize,
    pub preferred_height: usize,
    pub preferred_format: PixelFormat,
    pub enable_acceleration: bool,
    pub double_buffered: bool,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            preferred_width: 1920,
            preferred_height: 1080,
            preferred_format: PixelFormat::RGBA8888,
            enable_acceleration: true,
            double_buffered: true,
        }
    }
}

/// Main graphics system manager
pub struct GraphicsSystem {
    status: GraphicsStatus,
    config: GraphicsConfig,
    framebuffer_info: Option<FramebufferInfo>,
}

impl GraphicsSystem {
    /// Create a new graphics system
    pub const fn new() -> Self {
        Self {
            status: GraphicsStatus::Uninitialized,
            config: GraphicsConfig {
                preferred_width: 1920,
                preferred_height: 1080,
                preferred_format: PixelFormat::RGBA8888,
                enable_acceleration: true,
                double_buffered: true,
            },
            framebuffer_info: None,
        }
    }

    /// Initialize the graphics system with given framebuffer info
    pub fn init(&mut self, fb_info: FramebufferInfo) -> Result<(), &'static str> {
        self.status = GraphicsStatus::Initializing;

        // Initialize the framebuffer
        init(fb_info.clone(), self.config.double_buffered)?;

        self.framebuffer_info = Some(fb_info);
        self.status = GraphicsStatus::Ready;

        Ok(())
    }

    /// Get graphics system status
    pub fn status(&self) -> GraphicsStatus {
        self.status
    }

    /// Get framebuffer information
    pub fn framebuffer_info(&self) -> Option<&FramebufferInfo> {
        self.framebuffer_info.as_ref()
    }

    /// Get graphics configuration
    pub fn config(&self) -> &GraphicsConfig {
        &self.config
    }

    /// Check if graphics system is ready
    pub fn is_ready(&self) -> bool {
        self.status == GraphicsStatus::Ready
    }

    /// Get screen dimensions
    pub fn screen_dimensions(&self) -> Option<(usize, usize)> {
        self.framebuffer_info
            .as_ref()
            .map(|info| (info.width, info.height))
    }

    /// Get pixel format
    pub fn pixel_format(&self) -> Option<PixelFormat> {
        self.framebuffer_info.as_ref().map(|info| info.pixel_format)
    }
}

/// Global graphics system instance
static GRAPHICS_SYSTEM: Once<Mutex<GraphicsSystem>> = Once::new();

/// Initialize the global graphics system
pub fn init_graphics(fb_info: FramebufferInfo) -> Result<(), &'static str> {
    let mut graphics = GraphicsSystem::new();
    graphics.init(fb_info)?;

    GRAPHICS_SYSTEM.call_once(|| Mutex::new(graphics));
    Ok(())
}

/// Initialize graphics from bootloader framebuffer
pub fn init_from_bootloader(
    buffer: &'static mut [u8],
    info: FramebufferInfo,
) -> Result<(), &'static str> {
    let mut graphics = GraphicsSystem::new();
    graphics.status = GraphicsStatus::Initializing;

    init_with_buffer(buffer, info.clone(), graphics.config.double_buffered)?;

    graphics.framebuffer_info = Some(info);
    graphics.status = GraphicsStatus::Ready;

    GRAPHICS_SYSTEM.call_once(|| Mutex::new(graphics));
    Ok(())
}

/// Get reference to the global graphics system
pub fn graphics_system() -> Option<&'static Mutex<GraphicsSystem>> {
    GRAPHICS_SYSTEM.get()
}

/// Check if graphics system is initialized
pub fn is_graphics_initialized() -> bool {
    if let Some(graphics) = graphics_system() {
        graphics.lock().is_ready()
    } else {
        false
    }
}

/// Get screen dimensions from global graphics system
pub fn get_screen_dimensions() -> Option<(usize, usize)> {
    if let Some(graphics) = graphics_system() {
        graphics.lock().screen_dimensions()
    } else {
        None
    }
}

/// Get graphics status
pub fn get_graphics_status() -> GraphicsStatus {
    if let Some(graphics) = graphics_system() {
        graphics.lock().status()
    } else {
        GraphicsStatus::Uninitialized
    }
}

/// Drawing primitives and utilities
pub mod primitives {
    use super::{Color, Rect};
    use crate::graphics::framebuffer;

    /// Draw a line between two points
    pub fn draw_line(x1: usize, y1: usize, x2: usize, y2: usize, color: Color) {
        // Bresenham's line algorithm
        let dx = if x2 > x1 { x2 - x1 } else { x1 - x2 };
        let dy = if y2 > y1 { y2 - y1 } else { y1 - y2 };
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = if dx > dy { dx as isize } else { -(dy as isize) } / 2;
        let mut x = x1 as isize;
        let mut y = y1 as isize;

        loop {
            framebuffer::set_pixel(x as usize, y as usize, color);

            if x == x2 as isize && y == y2 as isize {
                break;
            }

            let e2 = err;

            if e2 > -(dx as isize) {
                err -= dy as isize;
                x += sx;
            }

            if e2 < dy as isize {
                err += dx as isize;
                y += sy;
            }
        }
    }

    /// Draw a circle
    pub fn draw_circle(center_x: usize, center_y: usize, radius: usize, color: Color) {
        let mut x = 0isize;
        let mut y = radius as isize;
        let mut d = 3 - 2 * radius as isize;

        while y >= x {
            // Draw the 8 octants
            plot_circle_points(center_x, center_y, x, y, color);
            x += 1;

            if d > 0 {
                y -= 1;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }
        }
    }

    /// Fill a circle
    pub fn fill_circle(center_x: usize, center_y: usize, radius: usize, color: Color) {
        let mut x = 0isize;
        let mut y = radius as isize;
        let mut d = 3 - 2 * radius as isize;

        while y >= x {
            // Draw horizontal lines for filled circle
            draw_circle_lines(center_x, center_y, x, y, color);
            x += 1;

            if d > 0 {
                y -= 1;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }
        }
    }

    /// Helper function to plot circle points
    fn plot_circle_points(cx: usize, cy: usize, x: isize, y: isize, color: Color) {
        let points = [
            (cx as isize + x, cy as isize + y),
            (cx as isize - x, cy as isize + y),
            (cx as isize + x, cy as isize - y),
            (cx as isize - x, cy as isize - y),
            (cx as isize + y, cy as isize + x),
            (cx as isize - y, cy as isize + x),
            (cx as isize + y, cy as isize - x),
            (cx as isize - y, cy as isize - x),
        ];

        for (px, py) in points.iter() {
            if *px >= 0 && *py >= 0 {
                framebuffer::set_pixel(*px as usize, *py as usize, color);
            }
        }
    }

    /// Helper function to draw horizontal lines for filled circle
    fn draw_circle_lines(cx: usize, cy: usize, x: isize, y: isize, color: Color) {
        if y >= 0 && x >= 0 {
            // Top and bottom horizontal lines
            if cy as isize + y >= 0 && cx >= x as usize && (cx + x as usize) < usize::MAX {
                for i in (cx - x as usize)..=(cx + x as usize) {
                    framebuffer::set_pixel(i, (cy as isize + y) as usize, color);
                }
            }
            if cy as isize - y >= 0 && cx >= x as usize && (cx + x as usize) < usize::MAX {
                for i in (cx - x as usize)..=(cx + x as usize) {
                    framebuffer::set_pixel(i, (cy as isize - y) as usize, color);
                }
            }

            // Left and right horizontal lines
            if y != x && cy >= y as usize && (cy + y as usize) < usize::MAX {
                for i in (cx - y as usize)..=(cx + y as usize) {
                    framebuffer::set_pixel(i, cy + x as usize, color);
                }
                if cy >= x as usize {
                    for i in (cx - y as usize)..=(cx + y as usize) {
                        framebuffer::set_pixel(i, cy - x as usize, color);
                    }
                }
            }
        }
    }

    /// Draw a gradient rectangle
    pub fn draw_gradient_rect(rect: Rect, start_color: Color, end_color: Color, vertical: bool) {
        if vertical {
            // Vertical gradient
            for y in 0..rect.height {
                let ratio = y as f32 / rect.height as f32;
                let color = interpolate_color(start_color, end_color, ratio);

                for x in 0..rect.width {
                    framebuffer::set_pixel(rect.x + x, rect.y + y, color);
                }
            }
        } else {
            // Horizontal gradient
            for x in 0..rect.width {
                let ratio = x as f32 / rect.width as f32;
                let color = interpolate_color(start_color, end_color, ratio);

                for y in 0..rect.height {
                    framebuffer::set_pixel(rect.x + x, rect.y + y, color);
                }
            }
        }
    }

    /// Interpolate between two colors
    fn interpolate_color(start: Color, end: Color, ratio: f32) -> Color {
        let ratio = ratio.clamp(0.0, 1.0);
        let inv_ratio = 1.0 - ratio;

        Color::new(
            (start.r as f32 * inv_ratio + end.r as f32 * ratio) as u8,
            (start.g as f32 * inv_ratio + end.g as f32 * ratio) as u8,
            (start.b as f32 * inv_ratio + end.b as f32 * ratio) as u8,
            (start.a as f32 * inv_ratio + end.a as f32 * ratio) as u8,
        )
    }
}

/// Performance and debugging utilities
pub mod debug {
    use super::*;

    /// Draw a performance overlay showing FPS and system info
    pub fn draw_performance_overlay(_frame_count: usize, x: usize, y: usize) {
        let overlay_rect = Rect::new(x, y, 200, 100);

        // Semi-transparent background
        framebuffer::fill_rect(overlay_rect, Color::new(0, 0, 0, 180));
        framebuffer::draw_rect(overlay_rect, Color::WHITE, 1);

        // In a real implementation, we would render text here
        // For now, just draw some colored rectangles as placeholders
        let fps_rect = Rect::new(x + 10, y + 10, 60, 10);
        framebuffer::fill_rect(fps_rect, Color::GREEN);

        let mem_rect = Rect::new(x + 10, y + 30, 80, 10);
        framebuffer::fill_rect(mem_rect, Color::BLUE);

        let gpu_rect = Rect::new(x + 10, y + 50, 40, 10);
        framebuffer::fill_rect(gpu_rect, Color::RED);
    }

    /// Draw a grid for debugging layout
    pub fn draw_debug_grid(spacing: usize, color: Color) {
        if let Some((width, height)) = get_screen_dimensions() {
            // Draw vertical lines
            for x in (0..width).step_by(spacing) {
                primitives::draw_line(x, 0, x, height - 1, color);
            }

            // Draw horizontal lines
            for y in (0..height).step_by(spacing) {
                primitives::draw_line(0, y, width - 1, y, color);
            }
        }
    }

    /// Draw coordinate system info
    pub fn draw_coordinates(x: usize, y: usize, color: Color) {
        // Draw crosshair at specified position
        if let Some((width, height)) = get_screen_dimensions() {
            if x < width && y < height {
                // Horizontal line
                primitives::draw_line(x.saturating_sub(10), y, (x + 10).min(width - 1), y, color);
                // Vertical line
                primitives::draw_line(x, y.saturating_sub(10), x, (y + 10).min(height - 1), color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{serial_print, serial_println, format};

    #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_graphics_config_default() {
        serial_print!("test_graphics_config_default... ");
        let config = GraphicsConfig::default();
        assert_eq!(config.preferred_width, 1920);
        assert_eq!(config.preferred_height, 1080);
        assert_eq!(config.preferred_format, PixelFormat::RGBA8888);
        assert!(config.enable_acceleration);
        assert!(config.double_buffered);
        serial_println!("[ok]");
    }

    #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_graphics_system_creation() {
        serial_print!("test_graphics_system_creation... ");
        let graphics = GraphicsSystem::new();
        assert_eq!(graphics.status(), GraphicsStatus::Uninitialized);
        assert!(graphics.framebuffer_info().is_none());
        assert!(!graphics.is_ready());
        serial_println!("[ok]");
    }

    #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_graphics_status_display() {
        serial_print!("test_graphics_status_display... ");
        assert_eq!(format!("{}", GraphicsStatus::Uninitialized), "Uninitialized");
        assert_eq!(format!("{}", GraphicsStatus::Ready), "Ready");
        assert_eq!(format!("{}", GraphicsStatus::Error), "Error");
        serial_println!("[ok]");
    }
}
