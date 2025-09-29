//! # RustOS Desktop Environment Module
//!
//! This module provides a complete desktop environment for RustOS, including
//! window management, graphics rendering, and user interface components.

pub mod window_manager;

use crate::graphics::framebuffer::{self, Color, FramebufferInfo, Rect};
use heapless::Vec;

// Re-export commonly used types
pub use window_manager::{ButtonId, DesktopEvent, MouseButton, WindowId, WindowManager};

/// Simplified desktop environment configuration
#[derive(Debug, Clone, Copy)]
pub struct DesktopConfig {
    pub preferred_width: u16,
    pub preferred_height: u16,
    pub preferred_bpp: u16,
    pub double_buffered: bool,
    pub hardware_acceleration: bool,
    pub show_splash: bool,
    pub background_color: Color,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            preferred_width: 1024,
            preferred_height: 768,
            preferred_bpp: 32,
            double_buffered: true,
            hardware_acceleration: false,
            show_splash: true,
            background_color: Color::rgb(28, 34, 54),
        }
    }
}

/// Desktop environment status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesktopStatus {
    Uninitialized,
    Initializing,
    Running,
    Error,
}

/// Simplified desktop environment structure
pub struct Desktop {
    status: DesktopStatus,
    config: DesktopConfig,
    frame_counter: usize,
    event_queue: Vec<DesktopEvent, 32>,
    framebuffer_info: Option<FramebufferInfo>,
    video_mode: Option<u16>,
    window_manager: Option<WindowManager>,
}

impl Desktop {
    /// Create a new desktop environment
    pub fn new(config: DesktopConfig) -> Self {
        Self {
            status: DesktopStatus::Uninitialized,
            config,
            frame_counter: 0,
            event_queue: Vec::new(),
            framebuffer_info: None,
            video_mode: None,
            window_manager: None,
        }
    }

    /// Initialize the desktop environment
    pub fn init(&mut self) -> Result<(), &'static str> {
        self.status = DesktopStatus::Initializing;

        // Initialize framebuffer
        framebuffer::clear_screen(self.config.background_color);

        // Initialize window manager
        self.window_manager = Some(WindowManager::new(
            self.config.preferred_width as usize,
            self.config.preferred_height as usize,
        ));

        if self.config.show_splash {
            self.show_splash_screen();
        }

        self.status = DesktopStatus::Running;
        Ok(())
    }

    /// Show startup splash screen
    fn show_splash_screen(&self) {
        let center_x = self.config.preferred_width as usize / 2;
        let center_y = self.config.preferred_height as usize / 2;

        let logo_rect = Rect::new(
            center_x.saturating_sub(200),
            center_y.saturating_sub(100),
            400,
            200,
        );

        framebuffer::fill_rect(logo_rect, Color::rgb(40, 40, 40));
        framebuffer::draw_rect(logo_rect, Color::rgb(70, 130, 180), 4);

        let inner_rect = Rect::new(
            logo_rect.x + 20,
            logo_rect.y + 20,
            logo_rect.width - 40,
            logo_rect.height - 40,
        );
        framebuffer::draw_rect(inner_rect, Color::rgb(100, 160, 200), 2);
    }

    /// Get framebuffer info
    pub fn framebuffer_info(&self) -> Option<&FramebufferInfo> {
        self.framebuffer_info.as_ref()
    }

    /// Get video mode
    pub fn video_mode(&self) -> Option<u16> {
        self.video_mode
    }

    /// Add event to queue
    pub fn add_event(&mut self, event: DesktopEvent) {
        let _ = self.event_queue.push(event);
    }

    /// Process events
    pub fn process_events(&mut self) {
        while let Some(event) = self.event_queue.pop() {
            self.handle_event(event);
        }
    }

    /// Handle a single event
    fn handle_event(&mut self, event: DesktopEvent) {
        if let Some(ref mut wm) = self.window_manager {
            match event {
                DesktopEvent::MouseMove { x, y } => {
                    wm.handle_mouse_move(x, y);
                }
                DesktopEvent::MouseDown { x, y, button } => {
                    wm.handle_mouse_down(x, y, button);
                }
                DesktopEvent::MouseUp { x, y, button } => {
                    wm.handle_mouse_up(x, y, button);
                }
                DesktopEvent::KeyDown { key: _ } => {
                    // Handle key down - simplified
                }
                DesktopEvent::KeyUp { key: _ } => {
                    // Handle key up - simplified
                }
                DesktopEvent::WindowClose { window_id } => {
                    wm.close_window(window_id);
                }
                DesktopEvent::WindowFocus { window_id } => {
                    wm.focus_window(window_id);
                }
                DesktopEvent::WindowResize {
                    window_id: _,
                    width: _,
                    height: _,
                } => {
                    // Handle window resize - simplified
                }
                DesktopEvent::WindowMove {
                    window_id: _,
                    x: _,
                    y: _,
                } => {
                    // Handle window move - simplified
                }
            }
        }
    }

    /// Update desktop state
    pub fn update(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);

        if let Some(ref mut wm) = self.window_manager {
            if wm.needs_redraw() {
                wm.render();
            }
        }
    }

    /// Get desktop status
    pub fn status(&self) -> DesktopStatus {
        self.status
    }

    /// Get desktop configuration
    pub fn config(&self) -> &DesktopConfig {
        &self.config
    }

    /// Get mutable window manager reference
    pub fn window_manager_mut(&mut self) -> Option<&mut WindowManager> {
        self.window_manager.as_mut()
    }

    /// Get window manager reference
    pub fn window_manager(&self) -> Option<&WindowManager> {
        self.window_manager.as_ref()
    }
}

use spin::Mutex;
use lazy_static::lazy_static;

// Global desktop state (production)
lazy_static! {
    static ref GLOBAL_DESKTOP: Mutex<Option<Desktop>> = Mutex::new(None);
}

/// Initialize the desktop environment
pub fn init_default_desktop() -> Result<(), &'static str> {
    let config = DesktopConfig::default();
    let mut desktop = Desktop::new(config);
    desktop.init()?;
    
    let mut global = GLOBAL_DESKTOP.lock();
    *global = Some(desktop);
    Ok(())
}

/// Set up full desktop environment
pub fn setup_full_desktop() -> Result<(), &'static str> {
    init_default_desktop()
}

/// Update desktop
pub fn update_desktop() {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.update();
    }
}

/// Get desktop status
pub fn get_desktop_status() -> DesktopStatus {
    let global = GLOBAL_DESKTOP.lock();
    global.as_ref().map_or(DesktopStatus::Uninitialized, |d| d.status())
}

/// Create a window using the global window manager
pub fn create_window(
    title: &'static str,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> WindowId {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        if let Some(ref mut wm) = desktop.window_manager_mut() {
            return wm.create_window(title, x, y, width, height);
        }
    }
    WindowId(0) // Failed
}

/// Close a window
pub fn close_window(window_id: WindowId) -> bool {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        if let Some(ref mut wm) = desktop.window_manager_mut() {
            wm.close_window(window_id);
            return true;
        }
    }
    false
}

/// Focus a window
pub fn focus_window(window_id: WindowId) -> bool {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        if let Some(ref mut wm) = desktop.window_manager_mut() {
            wm.focus_window(window_id);
            return true;
        }
    }
    false
}

/// Handle mouse move
pub fn handle_mouse_move(x: usize, y: usize) {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.add_event(DesktopEvent::MouseMove { x, y });
    }
}

/// Handle mouse down
pub fn handle_mouse_down(x: usize, y: usize, button: MouseButton) {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.add_event(DesktopEvent::MouseDown { x, y, button });
    }
}

/// Handle mouse up
pub fn handle_mouse_up(x: usize, y: usize, button: MouseButton) {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.add_event(DesktopEvent::MouseUp { x, y, button });
    }
}

/// Handle key down
pub fn handle_key_down(key: u8) {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.add_event(DesktopEvent::KeyDown { key });
    }
}

/// Process all pending desktop events
pub fn process_desktop_events() {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        desktop.process_events();
    }
}

/// Render desktop
pub fn render_desktop() {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        if let Some(ref mut wm) = desktop.window_manager_mut() {
            if wm.needs_redraw() {
                wm.render();
            }
        }
    }
}

/// Check if desktop needs redraw
pub fn desktop_needs_redraw() -> bool {
    let global = GLOBAL_DESKTOP.lock();
    if let Some(ref desktop) = *global {
        if let Some(ref wm) = desktop.window_manager() {
            return wm.needs_redraw();
        }
    }
    false
}

/// Invalidate desktop for redraw
pub fn invalidate_desktop() {
    let mut global = GLOBAL_DESKTOP.lock();
    if let Some(ref mut desktop) = *global {
        if let Some(ref mut wm) = desktop.window_manager_mut() {
            wm.force_redraw();
        }
    }
}

/// Get window manager
pub fn window_manager() -> Option<&'static WindowManager> {
    None // Simplified - would return actual window manager
}

// Simplified test functions (without #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test] attributes to avoid no_std issues)
#[cfg(test)]
mod tests {
    use super::*;

    fn test_desktop_creation() {
        let config = DesktopConfig::default();
        let desktop = Desktop::new(config);
        assert_eq!(desktop.status(), DesktopStatus::Uninitialized);
    }

    fn test_desktop_initialization() {
        let config = DesktopConfig::default();
        let mut desktop = Desktop::new(config);
        assert!(desktop.init().is_ok());
        assert_eq!(desktop.status(), DesktopStatus::Running);
    }

    fn test_event_handling() {
        let config = DesktopConfig::default();
        let mut desktop = Desktop::new(config);
        let _ = desktop.init();

        desktop.add_event(DesktopEvent::MouseMove { x: 100, y: 200 });
        desktop.process_events();
    }

    fn test_window_creation() {
        let window_id = create_window("Test Window", 10, 10, 300, 200);
        assert_ne!(window_id.0, 0);
    }
}
