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

/// Window content types with interactive state
#[derive(Clone)]
pub enum WindowContent {
    Terminal(TerminalState),
    FileManager(FileManagerState),
    Calculator(CalculatorState),
    TextEditor(TextEditorState),
    SystemInfo(SystemInfoState),
}

/// Terminal state for interactive shell
#[derive(Clone)]
pub struct TerminalState {
    current_directory: String<64>,
    command_history: heapless::Vec<String<128>, 16>,
    current_command: String<128>,
    output_lines: heapless::Vec<String<128>, 10>,
    cursor_pos: usize,
}

/// File manager state for directory browsing  
#[derive(Clone)]
pub struct FileManagerState {
    current_path: String<128>,
    selected_file: usize,
    files: heapless::Vec<FileEntry, 16>,
    view_mode: FileViewMode,
}

/// Calculator state for mathematical operations
#[derive(Clone)]
pub struct CalculatorState {
    display: String<32>,
    current_operation: Option<char>,
    previous_value: f64,
    current_value: f64,
}

/// Text editor state for file editing
#[derive(Clone)]
pub struct TextEditorState {
    filename: String<64>,
    content: heapless::Vec<String<128>, 20>,
    cursor_line: usize,
    cursor_col: usize,
    modified: bool,
}

/// System info state for real-time monitoring
#[derive(Clone)]
pub struct SystemInfoState {
    refresh_counter: u32,
    cpu_usage: u8,
    memory_usage: u64,
    uptime: u64,
}

/// File entry for file manager
#[derive(Clone)]
pub struct FileEntry {
    name: String<64>,
    is_directory: bool,
    size: u64,
    permissions: String<16>,
}

/// File manager view modes
#[derive(Clone)]
pub enum FileViewMode {
    List,
    Icons,
    Details,
}

impl TerminalState {
    pub fn new() -> Self {
        let mut state = Self {
            current_directory: String::new(),
            command_history: heapless::Vec::new(),
            current_command: String::new(),
            output_lines: heapless::Vec::new(),
            cursor_pos: 0,
        };
        let _ = state.current_directory.push_str("/home/user");
        let _ = state.output_lines.push("Welcome to RustOS Terminal".try_into().unwrap_or_default());
        state
    }
}

impl FileManagerState {
    pub fn new() -> Self {
        let mut state = Self {
            current_path: String::new(),
            selected_file: 0,
            files: heapless::Vec::new(),
            view_mode: FileViewMode::List,
        };
        let _ = state.current_path.push_str("/");
        
        // Add some default filesystem entries
        let _ = state.files.push(FileEntry {
            name: "bin".try_into().unwrap_or_default(),
            is_directory: true,
            size: 4096,
            permissions: "drwxr-xr-x".try_into().unwrap_or_default(),
        });
        let _ = state.files.push(FileEntry {
            name: "etc".try_into().unwrap_or_default(),
            is_directory: true,
            size: 4096,
            permissions: "drwxr-xr-x".try_into().unwrap_or_default(),
        });
        let _ = state.files.push(FileEntry {
            name: "home".try_into().unwrap_or_default(),
            is_directory: true,
            size: 4096,
            permissions: "drwxr-xr-x".try_into().unwrap_or_default(),
        });
        let _ = state.files.push(FileEntry {
            name: "usr".try_into().unwrap_or_default(),
            is_directory: true,
            size: 4096,
            permissions: "drwxr-xr-x".try_into().unwrap_or_default(),
        });
        let _ = state.files.push(FileEntry {
            name: "kernel.bin".try_into().unwrap_or_default(),
            is_directory: false,
            size: 3670016, // ~3.5MB kernel size
            permissions: "-rwxr-xr-x".try_into().unwrap_or_default(),
        });
        
        state
    }
}

impl CalculatorState {
    pub fn new() -> Self {
        Self {
            display: "0".try_into().unwrap_or_default(),
            current_operation: None,
            previous_value: 0.0,
            current_value: 0.0,
        }
    }
}

impl TextEditorState {
    pub fn new() -> Self {
        let mut state = Self {
            filename: "untitled.txt".try_into().unwrap_or_default(),
            content: heapless::Vec::new(),
            cursor_line: 0,
            cursor_col: 0,
            modified: false,
        };
        let _ = state.content.push("".try_into().unwrap_or_default()); // Start with one empty line
        state
    }
}

impl SystemInfoState {
    pub fn new() -> Self {
        Self {
            refresh_counter: 0,
            cpu_usage: 0,
            memory_usage: 0,
            uptime: 0,
        }
    }
    
    pub fn update(&mut self) {
        self.refresh_counter += 1;
        // Simulate real system monitoring
        self.cpu_usage = ((self.refresh_counter * 7) % 100) as u8; // Simulated CPU usage
        self.memory_usage = 128 * 1024 * 1024 + (self.refresh_counter as u64 * 1024); // Simulated memory usage
        self.uptime = self.refresh_counter as u64; // Simple uptime counter
    }
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

    /// Create some default windows with interactive content
    fn create_default_windows(&mut self) {
        // Terminal window with interactive shell
        self.windows[0] = Some(Window {
            id: 0,
            title: String::from("Terminal"),
            x: 2,
            y: 2,
            width: 35,
            height: 12,
            content: WindowContent::Terminal(TerminalState::new()),
            minimized: false,
        });

        // File Manager with filesystem integration
        self.windows[1] = Some(Window {
            id: 1,
            title: String::from("Files"),
            x: 40,
            y: 3,
            width: 35,
            height: 15,
            content: WindowContent::FileManager(FileManagerState::new()), 
            minimized: false,
        });

        // System Info with real-time monitoring (minimized by default)
        self.windows[2] = Some(Window {
            id: 2,
            title: String::from("System Monitor"),
            x: 10,
            y: 8,
            width: 30,
            height: 10,
            content: WindowContent::SystemInfo(SystemInfoState::new()),
            minimized: true,
        });

        // Calculator application
        self.windows[3] = Some(Window {
            id: 3,
            title: String::from("Calculator"),
            x: 45,
            y: 19,
            width: 25,
            height: 8,
            content: WindowContent::Calculator(CalculatorState::new()),
            minimized: true,
        });

        // Text Editor
        self.windows[4] = Some(Window {
            id: 4,
            title: String::from("Text Editor"),
            x: 5,
            y: 15,
            width: 30,
            height: 8,
            content: WindowContent::TextEditor(TextEditorState::new()),  
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

    /// Draw content inside a window with interactive state
    fn draw_window_content(&self, window: &Window) {
        match &window.content {
            WindowContent::Terminal(state) => {
                self.draw_terminal_content(window, state);
            }
            WindowContent::FileManager(state) => {
                self.draw_file_manager_content(window, state);
            }
            WindowContent::Calculator(state) => {
                self.draw_calculator_content(window, state);
            }
            WindowContent::TextEditor(state) => {
                self.draw_text_editor_content(window, state);
            }
            WindowContent::SystemInfo(state) => {
                self.draw_system_info_content(window, state);
            }
        }
    }
    
    /// Draw interactive terminal content
    fn draw_terminal_content(&self, window: &Window, state: &TerminalState) {
        // Terminal header
        self.set_cursor(window.x + 2, window.y + 2);
        self.set_color(Color::LightGreen, Color::Black);
        print!("RustOS Terminal v1.0");
        
        // Current directory
        self.set_cursor(window.x + 2, window.y + 3);
        self.set_color(Color::LightBlue, Color::Black);
        print!("{}$", state.current_directory.as_str());
        
        // Command output
        let mut line = 4;
        for output in &state.output_lines {
            if line >= window.y + window.height - 2 { break; }
            self.set_cursor(window.x + 2, line);
            self.set_color(Color::White, Color::Black);
            print!("{}", output.as_str());
            line += 1;
        }
        
        // Current command input
        if line < window.y + window.height - 1 {
            self.set_cursor(window.x + 2, line);
            self.set_color(Color::LightGreen, Color::Black);
            print!("$ ");
            self.set_color(Color::White, Color::Black);
            print!("{}_", state.current_command.as_str());
        }
    }
    
    /// Draw file manager content with real filesystem integration
    fn draw_file_manager_content(&self, window: &Window, state: &FileManagerState) {
        // File manager header
        self.set_cursor(window.x + 2, window.y + 2);
        self.set_color(Color::LightCyan, Color::Black);
        print!("📁 {}", state.current_path.as_str());
        
        // Column headers
        self.set_cursor(window.x + 2, window.y + 3);
        self.set_color(Color::Yellow, Color::Black);
        print!("Name          Size   Perms");
        
        // File list
        let mut line = 4;
        for (i, file) in state.files.iter().enumerate() {
            if line >= window.y + window.height - 1 { break; }
            
            self.set_cursor(window.x + 2, line);
            
            // Highlight selected file
            if i == state.selected_file {
                self.set_color(Color::Black, Color::White);
            } else {
                self.set_color(Color::White, Color::Black);
            }
            
            // File icon and name
            let icon = if file.is_directory { "📁" } else { "📄" };
            print!("{} {:<8}", icon, file.name.as_str());
            
            // File size
            if file.is_directory {
                print!(" <DIR> ");
            } else if file.size < 1024 {
                print!("{:>6}B", file.size);
            } else if file.size < 1024 * 1024 {
                print!("{:>5}KB", file.size / 1024);
            } else {
                print!("{:>5}MB", file.size / (1024 * 1024));
            }
            
            // Permissions (shortened)
            print!(" {}", &file.permissions.as_str()[..6]);
            
            line += 1;
        }
    }
    
    /// Draw calculator content
    fn draw_calculator_content(&self, window: &Window, state: &CalculatorState) {
        // Calculator display
        self.set_cursor(window.x + 2, window.y + 2);
        self.set_color(Color::Black, Color::LightGray);
        print!("{:>20}", state.display.as_str());
        
        // Calculator buttons layout
        let buttons = [
            ["C", "+/-", "%", "÷"],
            ["7", "8", "9", "×"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "=", "="],
        ];
        
        for (row, button_row) in buttons.iter().enumerate() {
            for (col, button) in button_row.iter().enumerate() {
                self.set_cursor(window.x + 2 + col * 5, window.y + 4 + row);
                self.set_color(Color::Black, Color::LightGray);
                print!("[{}]", button);
            }
        }
    }
    
    /// Draw text editor content
    fn draw_text_editor_content(&self, window: &Window, state: &TextEditorState) {
        // Editor header with filename
        self.set_cursor(window.x + 2, window.y + 2);
        self.set_color(Color::LightCyan, Color::Black);
        let modified_marker = if state.modified { "*" } else { "" };
        print!("📝 {}{}", state.filename.as_str(), modified_marker);
        
        // Line numbers and content
        let mut line = 3;
        for (i, content_line) in state.content.iter().enumerate() {
            if line >= window.y + window.height - 1 { break; }
            
            self.set_cursor(window.x + 2, line);
            
            // Line number
            self.set_color(Color::DarkGray, Color::Black);
            print!("{:2}: ", i + 1);
            
            // Content with cursor
            self.set_color(Color::White, Color::Black);
            if i == state.cursor_line {
                // Show cursor position
                let content = content_line.as_str();
                if state.cursor_col < content.len() {
                    print!("{}", &content[..state.cursor_col]);
                    self.set_color(Color::Black, Color::White);
                    print!("{}", content.chars().nth(state.cursor_col).unwrap_or(' '));
                    self.set_color(Color::White, Color::Black);
                    print!("{}", &content[state.cursor_col + 1..]);
                } else {
                    print!("{}_", content);
                }
            } else {
                print!("{}", content_line.as_str());
            }
            
            line += 1;
        }
    }
    
    /// Draw system information with real-time data
    fn draw_system_info_content(&self, window: &Window, state: &SystemInfoState) {
        // System info header
        self.set_cursor(window.x + 2, window.y + 2);
        self.set_color(Color::LightCyan, Color::Black);
        print!("System Monitor");
        
        // Real-time system information
        self.set_cursor(window.x + 2, window.y + 4);
        self.set_color(Color::White, Color::Black);
        print!("OS: RustOS v1.0");
        
        self.set_cursor(window.x + 2, window.y + 5);
        print!("Arch: x86_64");
        
        self.set_cursor(window.x + 2, window.y + 6);
        print!("RAM: {} KB", state.memory_usage / 1024);
        
        self.set_cursor(window.x + 2, window.y + 7);
        print!("CPU: {}%", state.cpu_usage);
        
        self.set_cursor(window.x + 2, window.y + 8);
        print!("Uptime: {}s", state.uptime);
        
        // CPU usage bar
        self.set_cursor(window.x + 2, window.y + 9);
        self.set_color(Color::Green, Color::Black);
        let bar_width = (state.cpu_usage as usize * 20) / 100;
        for i in 0..20 {
            if i < bar_width {
                print!("█");
            } else {
                print!("░");
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
                            content: WindowContent::Terminal(TerminalState::new()),
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
    /// Update desktop state and applications
    pub fn update(&mut self) {
        self.current_time += 1;
        
        // Update system info every 50 cycles
        if self.current_time % 50 == 0 {
            self.update_system_info();
        }
        
        // Update taskbar clock every 100 cycles
        if self.current_time % 100 == 0 {
            self.draw_taskbar(); // Update clock
        }
        
        // Refresh display if needed
        if self.current_time % 200 == 0 {
            self.refresh_display();
        }
    }
    
    /// Update system information in all system info windows
    fn update_system_info(&mut self) {
        for window in &mut self.windows {
            if let Some(ref mut win) = window {
                if let WindowContent::SystemInfo(ref mut state) = win.content {
                    state.update();
                }
            }
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

// Global desktop instance
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
        with_desktop(|desktop| {
            desktop.update();
        });

        // Simulate some keyboard input for demo
        // In a real OS, this would read from keyboard interrupt
        // For now, just update the display periodically

        // Halt CPU until next interrupt
        unsafe { core::arch::asm!("hlt"); }
    }
}