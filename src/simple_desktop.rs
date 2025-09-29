//! Simple Desktop Environment for RustOS
//!
//! A functional text-mode desktop with windows, taskbar, and interactive elements

use crate::vga_buffer::{Color, VGA_WRITER};
use crate::print;
use heapless::String;
use lazy_static::lazy_static;
use spin::Mutex;

const SCREEN_WIDTH: usize = 80;
const SCREEN_HEIGHT: usize = 25;
const TASKBAR_HEIGHT: usize = 1;
const DESKTOP_HEIGHT: usize = SCREEN_HEIGHT - TASKBAR_HEIGHT;

/// Desktop state
pub struct Desktop {
    current_time: usize,
    active_window: Option<usize>,
    windows: [Option<Window>; 5],
    menu_open: bool,
}

/// Window structure
#[derive(Clone)]
pub struct Window {
    id: usize,
    title: String<32>, // Fixed-size string for no_std
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    content: WindowContent,
    minimized: bool,
}

/// Window content types
#[derive(Clone)]
pub enum WindowContent {
    Terminal,
    FileManager,
    Calculator,
    TextEditor,
    SystemInfo,
}

impl Desktop {
    pub fn new() -> Self {
        Self {
            current_time: 0,
            active_window: None,
            windows: [None, None, None, None, None],
            menu_open: false,
        }
    }

    /// Initialize and show the desktop
    pub fn init(&mut self) {
        self.clear_screen();
        self.draw_wallpaper();
        self.draw_taskbar();
        self.create_default_windows();
        self.refresh_display();
    }

    /// Clear the entire screen
    fn clear_screen(&self) {
        let mut writer = VGA_WRITER.lock();
        writer.clear_screen();
    }

    /// Draw desktop wallpaper/background
    fn draw_wallpaper(&self) {
        self.set_cursor(0, 0);
        self.set_color(Color::Blue, Color::Black);

        // Draw a simple pattern background
        for y in 0..DESKTOP_HEIGHT {
            for x in 0..SCREEN_WIDTH {
                self.set_cursor(x, y);
                if (x + y) % 4 == 0 {
                    print!("·");
                } else {
                    print!(" ");
                }
            }
        }
    }

    /// Draw the taskbar at the bottom
    fn draw_taskbar(&self) {
        let y = SCREEN_HEIGHT - 1;

        // Taskbar background
        self.set_color(Color::White, Color::DarkGray);
        self.set_cursor(0, y);
        for _ in 0..SCREEN_WIDTH {
            print!(" ");
        }

        // Start menu button
        self.set_cursor(0, y);
        self.set_color(Color::Black, Color::LightGray);
        print!(" RustOS ");

        // Window buttons
        let mut x = 8;
        for (i, window) in self.windows.iter().enumerate() {
            if let Some(win) = window {
                if !win.minimized {
                    self.set_cursor(x, y);
                    if Some(i) == self.active_window {
                        self.set_color(Color::Yellow, Color::Blue);
                    } else {
                        self.set_color(Color::Black, Color::LightGray);
                    }
                    print!(" {} ", win.title);
                    x += win.title.len() + 2;
                }
            }
        }

        // Clock
        self.set_cursor(SCREEN_WIDTH - 8, y);
        self.set_color(Color::Black, Color::LightGray);
        print!(" {:02}:{:02} ", (self.current_time / 60) % 24, self.current_time % 60);
    }

    /// Create some default windows
    fn create_default_windows(&mut self) {
        // Terminal window
        self.windows[0] = Some(Window {
            id: 0,
            title: String::from("Terminal"),
            x: 2,
            y: 2,
            width: 35,
            height: 12,
            content: WindowContent::Terminal,
            minimized: false,
        });

        // File Manager
        self.windows[1] = Some(Window {
            id: 1,
            title: String::from("Files"),
            x: 40,
            y: 3,
            width: 35,
            height: 15,
            content: WindowContent::FileManager,
            minimized: false,
        });

        // System Info (minimized by default)
        self.windows[2] = Some(Window {
            id: 2,
            title: String::from("System"),
            x: 10,
            y: 8,
            width: 30,
            height: 10,
            content: WindowContent::SystemInfo,
            minimized: true,
        });

        self.active_window = Some(0);
    }

    /// Draw all windows
    fn draw_windows(&self) {
        for (i, window) in self.windows.iter().enumerate() {
            if let Some(win) = window {
                if !win.minimized {
                    self.draw_window(win, Some(i) == self.active_window);
                }
            }
        }
    }

    /// Draw a single window
    fn draw_window(&self, window: &Window, is_active: bool) {
        let title_color = if is_active { Color::Yellow } else { Color::White };
        let border_color = if is_active { Color::LightBlue } else { Color::LightGray };

        // Draw window border
        self.set_color(Color::Black, border_color);

        // Top border with title
        self.set_cursor(window.x, window.y);
        print!("┌");
        for _ in 0..(window.width - 2) {
            print!("─");
        }
        print!("┐");

        // Title bar
        self.set_cursor(window.x + 1, window.y);
        self.set_color(title_color, border_color);
        print!(" {} ", window.title);

        // Close button
        self.set_cursor(window.x + window.width - 3, window.y);
        self.set_color(Color::Red, border_color);
        print!("×");

        // Side borders and content area
        for row in 1..(window.height - 1) {
            self.set_cursor(window.x, window.y + row);
            self.set_color(Color::Black, border_color);
            print!("│");

            // Content area
            self.set_cursor(window.x + 1, window.y + row);
            self.set_color(Color::White, Color::Black);
            for _ in 0..(window.width - 2) {
                print!(" ");
            }

            self.set_cursor(window.x + window.width - 1, window.y + row);
            self.set_color(Color::Black, border_color);
            print!("│");
        }

        // Bottom border
        self.set_cursor(window.x, window.y + window.height - 1);
        self.set_color(Color::Black, border_color);
        print!("└");
        for _ in 0..(window.width - 2) {
            print!("─");
        }
        print!("┘");

        // Draw window content
        self.draw_window_content(window);
    }

    /// Draw content inside a window
    fn draw_window_content(&self, window: &Window) {
        match window.content {
            WindowContent::Terminal => {
                self.set_cursor(window.x + 2, window.y + 2);
                self.set_color(Color::LightGreen, Color::Black);
                print!("RustOS Terminal v1.0");

                self.set_cursor(window.x + 2, window.y + 3);
                self.set_color(Color::White, Color::Black);
                print!("$ ls -la");

                self.set_cursor(window.x + 2, window.y + 4);
                print!("total 42");
                self.set_cursor(window.x + 2, window.y + 5);
                print!("drwxr-xr-x  bin/");
                self.set_cursor(window.x + 2, window.y + 6);
                print!("drwxr-xr-x  etc/");
                self.set_cursor(window.x + 2, window.y + 7);
                print!("drwxr-xr-x  home/");
                self.set_cursor(window.x + 2, window.y + 8);
                print!("drwxr-xr-x  usr/");

                self.set_cursor(window.x + 2, window.y + 10);
                self.set_color(Color::LightGreen, Color::Black);
                print!("$ _");
            }
            WindowContent::FileManager => {
                self.set_cursor(window.x + 2, window.y + 2);
                self.set_color(Color::LightCyan, Color::Black);
                print!("📁 File Manager");

                self.set_cursor(window.x + 2, window.y + 4);
                self.set_color(Color::Yellow, Color::Black);
                print!("📁 Documents");
                self.set_cursor(window.x + 2, window.y + 5);
                print!("📁 Downloads");
                self.set_cursor(window.x + 2, window.y + 6);
                print!("📁 Pictures");
                self.set_cursor(window.x + 2, window.y + 7);
                print!("📁 Programs");

                self.set_cursor(window.x + 2, window.y + 9);
                self.set_color(Color::White, Color::Black);
                print!("📄 readme.txt");
                self.set_cursor(window.x + 2, window.y + 10);
                print!("📄 config.toml");
                self.set_cursor(window.x + 2, window.y + 11);
                print!("⚙️  kernel.bin");
            }
            WindowContent::SystemInfo => {
                self.set_cursor(window.x + 2, window.y + 2);
                self.set_color(Color::LightCyan, Color::Black);
                print!("System Information");

                self.set_cursor(window.x + 2, window.y + 4);
                self.set_color(Color::White, Color::Black);
                print!("OS: RustOS v1.0");
                self.set_cursor(window.x + 2, window.y + 5);
                print!("Arch: x86_64");
                self.set_cursor(window.x + 2, window.y + 6);
                print!("RAM: 512 MB");
                self.set_cursor(window.x + 2, window.y + 7);
                print!("CPU: Virtual x86_64");
            }
            _ => {
                self.set_cursor(window.x + 2, window.y + 2);
                self.set_color(Color::White, Color::Black);
                print!("Window Content");
            }
        }
    }

    /// Show start menu
    fn draw_start_menu(&self) {
        let menu_x = 1;
        let menu_y = DESKTOP_HEIGHT - 8;
        let menu_width = 20;
        let menu_height = 7;

        // Menu background
        self.set_color(Color::Black, Color::LightGray);
        for row in 0..menu_height {
            self.set_cursor(menu_x, menu_y + row);
            for _ in 0..menu_width {
                print!(" ");
            }
        }

        // Menu border
        self.set_color(Color::Black, Color::White);
        self.set_cursor(menu_x, menu_y);
        print!("┌");
        for _ in 0..(menu_width - 2) {
            print!("─");
        }
        print!("┐");

        // Menu items
        self.set_cursor(menu_x + 1, menu_y + 1);
        self.set_color(Color::Black, Color::LightGray);
        print!(" ⚙️  System Info");

        self.set_cursor(menu_x + 1, menu_y + 2);
        print!(" 🖥️  Terminal");

        self.set_cursor(menu_x + 1, menu_y + 3);
        print!(" 📁 File Manager");

        self.set_cursor(menu_x + 1, menu_y + 4);
        print!(" 🧮 Calculator");

        self.set_cursor(menu_x + 1, menu_y + 5);
        print!(" ⚡ Shutdown");

        // Bottom border
        self.set_cursor(menu_x, menu_y + menu_height - 1);
        self.set_color(Color::Black, Color::White);
        print!("└");
        for _ in 0..(menu_width - 2) {
            print!("─");
        }
        print!("┘");
    }

    /// Handle keyboard input
    pub fn handle_key(&mut self, key: u8) {
        match key {
            b'm' => {
                // Toggle start menu
                self.menu_open = !self.menu_open;
                self.refresh_display();
            }
            b'1'..=b'5' => {
                // Switch to window
                let window_id = (key - b'1') as usize;
                if self.windows[window_id].is_some() {
                    self.active_window = Some(window_id);
                    if let Some(ref mut window) = self.windows[window_id] {
                        window.minimized = false;
                    }
                    self.refresh_display();
                }
            }
            b'h' => {
                // Hide/minimize active window
                if let Some(active_id) = self.active_window {
                    if let Some(ref mut window) = self.windows[active_id] {
                        window.minimized = true;
                    }
                    self.active_window = None;
                    self.refresh_display();
                }
            }
            b'n' => {
                // Create new terminal window
                for i in 0..self.windows.len() {
                    if self.windows[i].is_none() {
                        // Create a simple title without format macro
                        let mut title = String::new();
                        let _ = title.push_str("Terminal ");
                        // Simple number conversion for terminal numbering
                        let num_char = match i + 1 {
                            1 => '1',
                            2 => '2',
                            3 => '3',
                            4 => '4',
                            5 => '5',
                            _ => '?',
                        };
                        let _ = title.push(num_char);

                        self.windows[i] = Some(Window {
                            id: i,
                            title,
                            x: 5 + i * 3,
                            y: 3 + i * 2,
                            width: 30,
                            height: 10,
                            content: WindowContent::Terminal,
                            minimized: false,
                        });
                        self.active_window = Some(i);
                        self.refresh_display();
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    /// Update desktop (called periodically)
    pub fn update(&mut self) {
        self.current_time += 1;
        if self.current_time % 100 == 0 {
            self.draw_taskbar(); // Update clock
        }
    }

    /// Refresh the entire display
    pub fn refresh_display(&self) {
        self.clear_screen();
        self.draw_wallpaper();
        self.draw_windows();
        self.draw_taskbar();

        if self.menu_open {
            self.draw_start_menu();
        }

        // Show help
        self.show_help();
    }

    /// Show help information
    fn show_help(&self) {
        self.set_cursor(2, 0);
        self.set_color(Color::Yellow, Color::Blue);
        print!(" RustOS Desktop - Keys: M=Menu, 1-5=Windows, H=Hide, N=New Terminal ");
    }

    /// Helper functions
    fn set_cursor(&self, x: usize, y: usize) {
        let mut writer = VGA_WRITER.lock();
        writer.set_cursor_position(y, x);
    }

    fn set_color(&self, foreground: Color, background: Color) {
        let mut writer = VGA_WRITER.lock();
        writer.set_color(foreground, background);
    }
}

/// Global desktop instance
lazy_static! {
    static ref DESKTOP: Mutex<Option<Desktop>> = Mutex::new(None);
}

/// Initialize the desktop
pub fn init_desktop() {
    let mut desktop_lock = DESKTOP.lock();
    let mut desktop = Desktop::new();
    desktop.init();
    *desktop_lock = Some(desktop);
}

/// Get desktop reference safely
pub fn with_desktop<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut Desktop) -> R,
{
    let mut desktop_lock = DESKTOP.lock();
    if let Some(ref mut desktop) = *desktop_lock {
        Some(f(desktop))
    } else {
        None
    }
}

/// Main desktop loop
pub fn run_desktop() -> ! {
    init_desktop();

    loop {
        if let Some(desktop) = get_desktop() {
            desktop.update();

            // Simulate some keyboard input for demo
            // In a real OS, this would read from keyboard interrupt
            // For now, just update the display periodically
        }

        // Halt CPU until next interrupt
        unsafe { core::arch::asm!("hlt"); }
    }
}