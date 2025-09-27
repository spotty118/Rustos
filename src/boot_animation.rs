/// Animated boot logo and splash screen system
/// Provides animated graphics during kernel initialization

use crate::vga_buffer::{Color, print_colored, print_banner};
use crate::gpu;

/// Animation frame for boot logo
pub struct AnimationFrame {
    pub frame_data: &'static str,
    pub duration_ms: u32,
    pub colors: (Color, Color), // (foreground, background)
}

/// Boot animation sequence
pub struct BootAnimation {
    frames: &'static [AnimationFrame],
    current_frame: usize,
    frame_timer: u32,
}

impl BootAnimation {
    pub fn new() -> Self {
        Self {
            frames: BOOT_FRAMES,
            current_frame: 0,
            frame_timer: 0,
        }
    }

    /// Update animation and return true if frame changed
    pub fn update(&mut self, delta_ms: u32) -> bool {
        self.frame_timer += delta_ms;
        
        if self.current_frame < self.frames.len() {
            let current = &self.frames[self.current_frame];
            if self.frame_timer >= current.duration_ms {
                self.frame_timer = 0;
                self.current_frame += 1;
                return true;
            }
        }
        false
    }

    /// Render current frame
    pub fn render(&self) {
        if self.current_frame < self.frames.len() {
            let frame = &self.frames[self.current_frame];
            
            // Clear screen if GPU is available
            if gpu::is_gpu_acceleration_available() {
                gpu::gpu_clear_screen(0x001122FF); // Dark blue background
            }
            
            // Render frame content
            print_colored(frame.frame_data, frame.colors.0, frame.colors.1);
        }
    }

    /// Check if animation is complete
    pub fn is_complete(&self) -> bool {
        self.current_frame >= self.frames.len()
    }

    /// Get current frame progress (0.0 to 1.0)
    pub fn get_progress(&self) -> f32 {
        if self.frames.is_empty() {
            return 1.0;
        }
        self.current_frame as f32 / self.frames.len() as f32
    }
}

/// ASCII art boot logo frames
static BOOT_FRAMES: &[AnimationFrame] = &[
    AnimationFrame {
        frame_data: "
██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝
██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗
██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║
██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝
        ",
        duration_ms: 800,
        colors: (Color::White, Color::Black),
    },
    AnimationFrame {
        frame_data: "
██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝
██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗
██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║
██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝

             🚀 AI-Enhanced Operating System 🤖
        ",
        duration_ms: 1000,
        colors: (Color::LightCyan, Color::Black),
    },
    AnimationFrame {
        frame_data: "
██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝
██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗
██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║
██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝

             🚀 AI-Enhanced Operating System 🤖
                 🖥️  GPU Accelerated Desktop 🖥️
        ",
        duration_ms: 1000,
        colors: (Color::LightGreen, Color::Black),
    },
    AnimationFrame {
        frame_data: "
██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝
██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗
██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║
██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝

             🚀 AI-Enhanced Operating System 🤖
                 🖥️  GPU Accelerated Desktop 🖥️
                 ⚡ Multi-Core Processing ⚡
        ",
        duration_ms: 1000,
        colors: (Color::Yellow, Color::Black),
    },
    AnimationFrame {
        frame_data: "
██████╗ ██╗   ██╗███████╗████████╗ ██████╗ ███████╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔═══██╗██╔════╝
██████╔╝██║   ██║███████╗   ██║   ██║   ██║███████╗
██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║╚════██║
██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚══════╝

             🚀 AI-Enhanced Operating System 🤖
                 🖥️  GPU Accelerated Desktop 🖥️
                 ⚡ Multi-Core Processing ⚡
                 💾 Large Memory & Storage 💾

                    Initializing System...
        ",
        duration_ms: 1500,
        colors: (Color::Pink, Color::Black),
    },
];

/// Simple timing function for animation delays
pub fn delay_ms(ms: u32) {
    // Simple busy-wait delay - in a real OS you'd use proper timers
    for _ in 0..(ms * 1000) {
        unsafe {
            core::arch::asm!("nop");
        }
    }
}

/// Run the full boot animation sequence
pub fn run_boot_animation() {
    let mut animation = BootAnimation::new();
    
    // Clear screen
    if gpu::is_gpu_acceleration_available() {
        gpu::gpu_clear_screen(0x000000FF); // Black background
    }
    
    // Run animation frames
    while !animation.is_complete() {
        if animation.update(100) { // 100ms updates
            animation.render();
        }
        delay_ms(100);
    }
    
    // Final boot complete message
    print_banner("🎉 Boot Animation Complete - System Ready! 🎉", Color::LightGreen, Color::Black);
}

/// Simple progress bar for loading sequences
pub fn show_progress_bar(progress: f32, label: &str) {
    let bar_width = 40;
    let filled = (progress * bar_width as f32) as usize;
    let empty = bar_width - filled;
    
    let mut bar = heapless::String::<64>::new();
    bar.push('[').ok();
    
    for _ in 0..filled {
        bar.push('█').ok();
    }
    for _ in 0..empty {
        bar.push('░').ok();
    }
    
    bar.push(']').ok();
    
    print_colored(&bar, Color::LightBlue, Color::Black);
    print_colored(label, Color::White, Color::Black);
}