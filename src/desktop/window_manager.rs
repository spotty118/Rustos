//! # RustOS Desktop Window Manager
//!
//! A comprehensive desktop environment with window management, UI components,
//! and event handling for the RustOS kernel.

use crate::graphics::framebuffer::{Color, Rect};
// use core::cmp::max; // Currently unused
use heapless::Vec;

/// Maximum number of windows that can be managed simultaneously
pub const MAX_WINDOWS: usize = 64;

/// Default window title bar height
pub const TITLE_BAR_HEIGHT: usize = 24;

/// Default window border width
pub const BORDER_WIDTH: usize = 2;

/// Minimum window size
pub const MIN_WINDOW_WIDTH: usize = 200;
pub const MIN_WINDOW_HEIGHT: usize = 150;

/// Desktop colors
pub mod colors {
    use crate::graphics::framebuffer::Color;

    pub const DESKTOP_BACKGROUND: Color = Color::rgb(64, 128, 128);
    pub const WINDOW_BACKGROUND: Color = Color::rgb(240, 240, 240);
    pub const TITLE_BAR_ACTIVE: Color = Color::rgb(70, 130, 180);
    pub const TITLE_BAR_INACTIVE: Color = Color::rgb(128, 128, 128);
    pub const BORDER_ACTIVE: Color = Color::rgb(70, 130, 180);
    pub const BORDER_INACTIVE: Color = Color::rgb(128, 128, 128);
    pub const TEXT_COLOR: Color = Color::rgb(0, 0, 0);
    pub const TEXT_COLOR_WHITE: Color = Color::rgb(255, 255, 255);
    pub const BUTTON_BACKGROUND: Color = Color::rgb(225, 225, 225);
    pub const BUTTON_HOVER: Color = Color::rgb(200, 200, 200);
    pub const BUTTON_PRESSED: Color = Color::rgb(180, 180, 180);
}

/// Window state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowState {
    Normal,
    Maximized,
    Minimized,
    Closed,
}

/// Mouse button enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Event types for the desktop environment
#[derive(Debug, Clone)]
pub enum DesktopEvent {
    MouseMove {
        x: usize,
        y: usize,
    },
    MouseDown {
        x: usize,
        y: usize,
        button: MouseButton,
    },
    MouseUp {
        x: usize,
        y: usize,
        button: MouseButton,
    },
    KeyDown {
        key: u8,
    },
    KeyUp {
        key: u8,
    },
    WindowClose {
        window_id: WindowId,
    },
    WindowFocus {
        window_id: WindowId,
    },
    WindowResize {
        window_id: WindowId,
        width: usize,
        height: usize,
    },
    WindowMove {
        window_id: WindowId,
        x: usize,
        y: usize,
    },
}

/// Unique identifier for windows
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub usize);

/// Unique identifier for buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ButtonId(pub usize);

/// Window structure
#[derive(Debug, Clone)]
pub struct Window {
    pub id: WindowId,
    pub title: &'static str,
    pub rect: Rect,
    pub client_area: Rect,
    pub state: WindowState,
    pub focused: bool,
    pub resizable: bool,
    pub movable: bool,
    pub visible: bool,
    pub has_title_bar: bool,
    pub has_border: bool,
    pub background_color: Color,
    pub border_color: Color,
    pub title_bar_color: Color,
    pub z_order: usize,
}

/// Button structure
#[derive(Debug, Clone)]
pub struct Button {
    pub id: ButtonId,
    pub rect: Rect,
    pub text: &'static str,
    pub background_color: Color,
    pub text_color: Color,
    pub pressed: bool,
    pub hovered: bool,
    pub enabled: bool,
    pub visible: bool,
}

/// Cursor structure
#[derive(Debug, Clone)]
pub struct Cursor {
    pub x: usize,
    pub y: usize,
    pub visible: bool,
    pub color: Color,
}

impl WindowId {
    pub const INVALID: WindowId = WindowId(usize::MAX);
}

impl ButtonId {
    pub const INVALID: ButtonId = ButtonId(usize::MAX);
}

impl Window {
    /// Create a new window
    pub fn new(
        id: WindowId,
        title: &'static str,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let rect = Rect::new(x, y, width, height);
        let client_area = Rect::new(
            x + BORDER_WIDTH,
            y + TITLE_BAR_HEIGHT + BORDER_WIDTH,
            width.saturating_sub(2 * BORDER_WIDTH),
            height.saturating_sub(TITLE_BAR_HEIGHT + 2 * BORDER_WIDTH),
        );

        Self {
            id,
            title,
            rect,
            client_area,
            state: WindowState::Normal,
            focused: false,
            resizable: true,
            movable: true,
            visible: true,
            has_title_bar: true,
            has_border: true,
            background_color: colors::WINDOW_BACKGROUND,
            border_color: colors::BORDER_INACTIVE,
            title_bar_color: colors::TITLE_BAR_INACTIVE,
            z_order: 0,
        }
    }
}

impl Button {
    /// Create a new button
    pub fn new(id: ButtonId, rect: Rect, text: &'static str) -> Self {
        Self {
            id,
            rect,
            text,
            background_color: colors::BUTTON_BACKGROUND,
            text_color: colors::TEXT_COLOR,
            pressed: false,
            hovered: false,
            enabled: true,
            visible: true,
        }
    }
}

impl Cursor {
    /// Create a new cursor
    pub fn new() -> Self {
        Self {
            x: 0,
            y: 0,
            visible: true,
            color: Color::rgb(255, 255, 255),
        }
    }
}

/// Main desktop window manager
pub struct WindowManager {
    windows: Vec<Window, MAX_WINDOWS>,
    buttons: Vec<Button, 32>,
    window_count: usize,
    next_window_id: usize,
    next_button_id: usize,
    focused_window: Option<WindowId>,
    desktop_rect: Rect,
    needs_redraw: bool,
    cursor: Cursor,
    dragging_window: Option<WindowId>,
    drag_offset: (usize, usize),
}

impl WindowManager {
    /// Create a new window manager
    pub fn new(screen_width: usize, screen_height: usize) -> Self {
        Self {
            windows: Vec::new(),
            buttons: Vec::new(),
            window_count: 0,
            next_window_id: 1,
            next_button_id: 1,
            focused_window: None,
            desktop_rect: Rect::new(0, 0, screen_width, screen_height),
            needs_redraw: true,
            cursor: Cursor::new(),
            dragging_window: None,
            drag_offset: (0, 0),
        }
    }

    /// Create a new window
    pub fn create_window(
        &mut self,
        title: &'static str,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> WindowId {
        let window_id = WindowId(self.next_window_id);
        self.next_window_id += 1;

        let mut window = Window::new(window_id, title, x, y, width, height);
        window.focused = true;
        window.z_order = self.window_count;
        window.border_color = colors::BORDER_ACTIVE;
        window.title_bar_color = colors::TITLE_BAR_ACTIVE;

        // Unfocus other windows
        for w in &mut self.windows {
            w.focused = false;
            w.border_color = colors::BORDER_INACTIVE;
            w.title_bar_color = colors::TITLE_BAR_INACTIVE;
        }

        let _ = self.windows.push(window);
        self.window_count += 1;
        self.focused_window = Some(window_id);
        self.needs_redraw = true;
        window_id
    }

    /// Get window by ID
    pub fn get_window(&self, window_id: WindowId) -> Option<&Window> {
        self.windows.iter().find(|w| w.id == window_id)
    }

    /// Get mutable window by ID
    pub fn get_window_mut(&mut self, window_id: WindowId) -> Option<&mut Window> {
        self.windows.iter_mut().find(|w| w.id == window_id)
    }

    /// Focus a window
    pub fn focus_window(&mut self, window_id: WindowId) -> bool {
        let mut found = false;
        for window in &mut self.windows {
            if window.id == window_id {
                window.focused = true;
                window.border_color = colors::BORDER_ACTIVE;
                window.title_bar_color = colors::TITLE_BAR_ACTIVE;
                found = true;
            } else {
                window.focused = false;
                window.border_color = colors::BORDER_INACTIVE;
                window.title_bar_color = colors::TITLE_BAR_INACTIVE;
            }
        }
        if found {
            self.focused_window = Some(window_id);
            self.needs_redraw = true;
        }
        found
    }

    /// Get window at point
    pub fn window_at_point(&self, x: usize, y: usize) -> Option<WindowId> {
        self.windows
            .iter()
            .filter(|w| w.visible && w.rect.contains(x, y))
            .max_by_key(|w| w.z_order)
            .map(|w| w.id)
    }

    /// Close a window
    pub fn close_window(&mut self, window_id: WindowId) -> bool {
        if let Some(pos) = self.windows.iter().position(|w| w.id == window_id) {
            self.windows.swap_remove(pos);
            self.window_count = self.window_count.saturating_sub(1);

            if self.focused_window == Some(window_id) {
                self.focused_window = self.windows.last().map(|w| w.id);
            }

            self.needs_redraw = true;
            true
        } else {
            false
        }
    }

    /// Create a button
    pub fn create_button(&mut self, rect: Rect, text: &'static str) -> ButtonId {
        let button_id = ButtonId(self.next_button_id);
        self.next_button_id += 1;

        let button = Button::new(button_id, rect, text);
        let _ = self.buttons.push(button);
        self.needs_redraw = true;
        button_id
    }

    /// Handle mouse move
    pub fn handle_mouse_move(&mut self, x: usize, y: usize) {
        self.cursor.x = x;
        self.cursor.y = y;

        if let Some(window_id) = self.dragging_window {
            let drag_offset = self.drag_offset;
            if let Some(window) = self.get_window_mut(window_id) {
                let new_x = x.saturating_sub(drag_offset.0);
                let new_y = y.saturating_sub(drag_offset.1);
                window.rect.x = new_x;
                window.rect.y = new_y;
                window.client_area.x = new_x + BORDER_WIDTH;
                window.client_area.y = new_y + TITLE_BAR_HEIGHT + BORDER_WIDTH;
            }
            self.needs_redraw = true;
        }

        // Update button hover states
        for button in &mut self.buttons {
            button.hovered = button.rect.contains(x, y);
        }
    }

    /// Handle mouse down
    pub fn handle_mouse_down(&mut self, x: usize, y: usize, _button: MouseButton) -> bool {
        if let Some(window_id) = self.window_at_point(x, y) {
            self.focus_window(window_id);

            let window_rect = if let Some(window) = self.get_window(window_id) {
                Some((window.rect.x, window.rect.y, window.rect.width))
            } else {
                None
            };

            if let Some((win_x, win_y, win_width)) = window_rect {
                let title_rect = Rect::new(win_x, win_y, win_width, TITLE_BAR_HEIGHT);

                if title_rect.contains(x, y) {
                    self.dragging_window = Some(window_id);
                    self.drag_offset = (x.saturating_sub(win_x), y.saturating_sub(win_y));
                    return true;
                }
            }
        }

        // Check buttons
        for button in &mut self.buttons {
            if button.rect.contains(x, y) && button.enabled && button.visible {
                button.pressed = true;
                self.needs_redraw = true;
                return true;
            }
        }

        false
    }

    /// Handle mouse up
    pub fn handle_mouse_up(&mut self, _x: usize, _y: usize, _button: MouseButton) {
        self.dragging_window = None;

        for button in &mut self.buttons {
            if button.pressed {
                button.pressed = false;
                self.needs_redraw = true;
            }
        }
    }

    /// Render the desktop
    pub fn render(&mut self) {
        if !self.needs_redraw {
            return;
        }

        // Clear desktop background
        crate::graphics::framebuffer::fill_rect(self.desktop_rect, colors::DESKTOP_BACKGROUND);

        // Render windows (back to front)
        let mut sorted_windows: Vec<&Window, MAX_WINDOWS> = Vec::new();
        for window in &self.windows {
            if window.visible {
                let _ = sorted_windows.push(window);
            }
        }

        // Simple sort by z_order
        for i in 0..sorted_windows.len() {
            for j in i + 1..sorted_windows.len() {
                if sorted_windows[i].z_order > sorted_windows[j].z_order {
                    sorted_windows.swap(i, j);
                }
            }
        }

        for window in &sorted_windows {
            self.render_window(window);
        }

        // Render buttons
        for button in &self.buttons {
            if button.visible {
                self.render_button(button);
            }
        }

        // Render cursor
        if self.cursor.visible {
            self.render_cursor();
        }

        self.needs_redraw = false;
    }

    /// Render a single window
    fn render_window(&self, window: &Window) {
        // Render border
        if window.has_border {
            crate::graphics::framebuffer::draw_rect(window.rect, window.border_color, BORDER_WIDTH);
        }

        // Render title bar
        if window.has_title_bar {
            let title_rect = Rect::new(
                window.rect.x + BORDER_WIDTH,
                window.rect.y + BORDER_WIDTH,
                window.rect.width.saturating_sub(2 * BORDER_WIDTH),
                TITLE_BAR_HEIGHT,
            );
            crate::graphics::framebuffer::fill_rect(title_rect, window.title_bar_color);

            // Render close button
            let close_button_size = 16;
            let close_x = window.rect.x + window.rect.width - close_button_size - 4;
            let close_y = window.rect.y + 4;
            let close_rect = Rect::new(close_x, close_y, close_button_size, close_button_size);

            crate::graphics::framebuffer::fill_rect(close_rect, colors::BUTTON_BACKGROUND);
            crate::graphics::framebuffer::draw_rect(close_rect, colors::BORDER_INACTIVE, 1);
        }

        // Render window content area
        crate::graphics::framebuffer::fill_rect(window.client_area, window.background_color);
    }

    /// Render a button
    fn render_button(&self, button: &Button) {
        let bg_color = if button.pressed {
            colors::BUTTON_PRESSED
        } else if button.hovered {
            colors::BUTTON_HOVER
        } else {
            button.background_color
        };

        crate::graphics::framebuffer::fill_rect(button.rect, bg_color);
        crate::graphics::framebuffer::draw_rect(button.rect, colors::BORDER_INACTIVE, 1);
    }

    /// Render cursor
    fn render_cursor(&self) {
        // Simple cursor - just a few pixels
        for dy in 0..10 {
            for dx in 0..2 {
                if self.cursor.x + dx < self.desktop_rect.width
                    && self.cursor.y + dy < self.desktop_rect.height
                {
                    crate::graphics::framebuffer::set_pixel(
                        self.cursor.x + dx,
                        self.cursor.y + dy,
                        self.cursor.color,
                    );
                }
            }
        }
    }

    /// Get focused window
    pub fn get_focused_window(&self) -> Option<WindowId> {
        self.focused_window
    }

    /// Get window count
    pub fn get_window_count(&self) -> usize {
        self.windows.len()
    }

    /// Set cursor position
    pub fn set_cursor_position(&mut self, x: usize, y: usize) {
        self.cursor.x = x.min(self.desktop_rect.width.saturating_sub(1));
        self.cursor.y = y.min(self.desktop_rect.height.saturating_sub(1));
    }

    /// Show/hide cursor
    pub fn set_cursor_visible(&mut self, visible: bool) {
        self.cursor.visible = visible;
        self.needs_redraw = true;
    }

    /// Get desktop rect
    pub fn get_desktop_rect(&self) -> Rect {
        self.desktop_rect
    }

    /// Check if redraw is needed
    pub fn needs_redraw(&self) -> bool {
        self.needs_redraw
    }

    /// Force redraw
    pub fn force_redraw(&mut self) {
        self.needs_redraw = true;
    }
}

/// Desktop event handling result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventResult {
    Handled,
    NotHandled,
    WindowClosed(WindowId),
}

// Test functions (without test attributes to avoid no_std issues)
#[cfg(test)]
mod tests {
    use super::*;

    fn test_window_creation() {
        let mut wm = WindowManager::new(1920, 1080);
        let window_id = wm.create_window("Test Window", 100, 100, 400, 300);

        assert_ne!(window_id, WindowId::INVALID);
        assert_eq!(wm.window_count, 1);
    }

    fn test_window_focus() {
        let mut wm = WindowManager::new(1920, 1080);
        let window1 = wm.create_window("Window 1", 100, 100, 400, 300);
        let window2 = wm.create_window("Window 2", 200, 200, 400, 300);

        assert_eq!(wm.get_focused_window(), Some(window2));

        assert!(wm.focus_window(window1));
        assert_eq!(wm.get_focused_window(), Some(window1));
    }

    fn test_window_close() {
        let mut wm = WindowManager::new(1920, 1080);
        let window1 = wm.create_window("Window 1", 100, 100, 400, 300);
        let window2 = wm.create_window("Window 2", 200, 200, 400, 300);

        assert_eq!(wm.window_count, 2);
        assert!(wm.close_window(window1));
        assert_eq!(wm.window_count, 1);
        assert!(wm.get_window(window2).is_some());
        assert!(wm.get_window(window1).is_none());
    }

    fn test_window_at_point() {
        let mut wm = WindowManager::new(1920, 1080);
        let window1 = wm.create_window("Window 1", 100, 100, 200, 200);
        let window2 = wm.create_window("Window 2", 150, 150, 200, 200);

        assert_eq!(wm.window_at_point(120, 120), Some(window1));
        assert_eq!(wm.window_at_point(175, 175), Some(window2));
        assert_eq!(wm.window_at_point(50, 50), None);
    }

    fn test_button_creation() {
        let mut wm = WindowManager::new(1920, 1080);
        let button_rect = Rect::new(10, 10, 100, 30);
        let button_id = wm.create_button(button_rect, "Test Button");

        assert_ne!(button_id, ButtonId::INVALID);
        assert_eq!(wm.buttons.len(), 1);
    }
}
