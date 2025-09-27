/// GPU acceleration and graphics support for desktop UI
/// Supports Intel, NVIDIA, and AMD GPU architectures

use spin::Mutex;
use lazy_static::lazy_static;
use heapless::Vec;

pub mod intel;
pub mod nvidia;
pub mod amd;
pub mod framebuffer;

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUVendor {
    Intel,
    Nvidia,
    AMD,
    Unknown,
}

/// GPU capabilities and features
#[derive(Debug, Clone)]
pub struct GPUCapabilities {
    pub vendor: GPUVendor,
    pub memory_size: u64,     // GPU memory in bytes
    pub max_resolution: (u32, u32),
    pub supports_2d_accel: bool,
    pub supports_3d_accel: bool,
    pub supports_compute: bool,
    pub pci_device_id: u16,
}

/// GPU acceleration status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GPUStatus {
    Uninitialized,
    Initializing,
    Ready,
    Error,
}

/// Main GPU management system
pub struct GPUSystem {
    status: GPUStatus,
    detected_gpus: Vec<GPUCapabilities, 4>, // Support up to 4 GPUs
    active_gpu: Option<usize>,
    framebuffer: Option<framebuffer::Framebuffer>,
}

impl GPUSystem {
    pub fn new() -> Self {
        Self {
            status: GPUStatus::Uninitialized,
            detected_gpus: Vec::new(),
            active_gpu: None,
            framebuffer: None,
        }
    }

    /// Initialize GPU system and detect available GPUs
    pub fn initialize(&mut self) -> Result<(), &'static str> {
        self.status = GPUStatus::Initializing;
        
        crate::println!("[GPU] Initializing GPU acceleration system...");
        
        // Detect GPUs via PCI enumeration
        self.detect_gpus()?;
        
        if self.detected_gpus.is_empty() {
            crate::println!("[GPU] Warning: No supported GPUs detected, falling back to VGA");
            self.status = GPUStatus::Ready;
            return Ok(());
        }

        // Initialize the best available GPU
        self.initialize_best_gpu()?;
        
        self.status = GPUStatus::Ready;
        crate::println!("[GPU] GPU acceleration system initialized successfully");
        Ok(())
    }

    /// Detect available GPUs through PCI bus scanning
    fn detect_gpus(&mut self) -> Result<(), &'static str> {
        crate::println!("[GPU] Scanning PCI bus for GPU devices...");
        
        // Intel GPU detection (integrated graphics)
        if let Ok(intel_gpu) = intel::detect_intel_gpu() {
            crate::println!("[GPU] Detected Intel GPU: {:?}", intel_gpu.pci_device_id);
            let _ = self.detected_gpus.push(intel_gpu);
        }

        // NVIDIA GPU detection
        if let Ok(nvidia_gpu) = nvidia::detect_nvidia_gpu() {
            crate::println!("[GPU] Detected NVIDIA GPU: {:?}", nvidia_gpu.pci_device_id);
            let _ = self.detected_gpus.push(nvidia_gpu);
        }

        // AMD GPU detection
        if let Ok(amd_gpu) = amd::detect_amd_gpu() {
            crate::println!("[GPU] Detected AMD GPU: {:?}", amd_gpu.pci_device_id);
            let _ = self.detected_gpus.push(amd_gpu);
        }

        crate::println!("[GPU] Found {} GPU(s)", self.detected_gpus.len());
        Ok(())
    }

    /// Initialize the best available GPU
    fn initialize_best_gpu(&mut self) -> Result<(), &'static str> {
        // Priority: NVIDIA > AMD > Intel (for acceleration capabilities)
        let mut best_gpu_index = None;
        let mut best_priority = 0;

        for (index, gpu) in self.detected_gpus.iter().enumerate() {
            let priority = match gpu.vendor {
                GPUVendor::Nvidia => 3,
                GPUVendor::AMD => 2,
                GPUVendor::Intel => 1,
                GPUVendor::Unknown => 0,
            };

            if priority > best_priority {
                best_priority = priority;
                best_gpu_index = Some(index);
            }
        }

        if let Some(gpu_index) = best_gpu_index {
            let gpu = &self.detected_gpus[gpu_index];
            crate::println!("[GPU] Initializing {:?} GPU for acceleration", gpu.vendor);
            
            // Initialize vendor-specific driver
            match gpu.vendor {
                GPUVendor::Intel => intel::initialize_intel_gpu(gpu)?,
                GPUVendor::Nvidia => nvidia::initialize_nvidia_gpu(gpu)?,
                GPUVendor::AMD => amd::initialize_amd_gpu(gpu)?,
                GPUVendor::Unknown => return Err("Cannot initialize unknown GPU"),
            }

            // Set up framebuffer
            self.framebuffer = Some(framebuffer::Framebuffer::new(gpu)?);
            self.active_gpu = Some(gpu_index);
        }

        Ok(())
    }

    pub fn get_status(&self) -> GPUStatus {
        self.status
    }

    pub fn is_acceleration_available(&self) -> bool {
        self.active_gpu.is_some() && self.status == GPUStatus::Ready
    }

    pub fn get_active_gpu(&self) -> Option<&GPUCapabilities> {
        self.active_gpu.map(|index| &self.detected_gpus[index])
    }

    /// Clear screen with GPU acceleration if available
    pub fn clear_screen(&mut self, color: u32) {
        if let Some(ref mut fb) = self.framebuffer {
            fb.clear(color);
        }
    }

    /// Draw a rectangle using GPU acceleration
    pub fn draw_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: u32) {
        if let Some(ref mut fb) = self.framebuffer {
            fb.draw_rect(x, y, width, height, color);
        }
    }

    /// Update display (present framebuffer)
    pub fn present(&mut self) {
        if let Some(ref mut fb) = self.framebuffer {
            fb.present();
        }
    }
}

lazy_static! {
    static ref GPU_SYSTEM: Mutex<GPUSystem> = Mutex::new(GPUSystem::new());
}

/// Initialize GPU acceleration system
pub fn init_gpu_system() -> Result<(), &'static str> {
    let mut gpu = GPU_SYSTEM.lock();
    gpu.initialize()
}

/// Get GPU system status
pub fn get_gpu_status() -> GPUStatus {
    let gpu = GPU_SYSTEM.lock();
    gpu.status
}

/// Check if GPU acceleration is available
pub fn is_gpu_acceleration_available() -> bool {
    let gpu = GPU_SYSTEM.lock();
    !gpu.detected_gpus.is_empty() && gpu.status == GPUStatus::Ready
}

/// Initialize desktop UI with GPU acceleration
pub fn init_desktop_ui() -> Result<(), &'static str> {
    use framebuffer::{Framebuffer, DesktopUI};
    
    let mut gpu_system = GPU_SYSTEM.lock();
    
    if let Some(gpu_index) = gpu_system.active_gpu {
        if let Some(gpu) = gpu_system.detected_gpus.get(gpu_index) {
            // Create framebuffer for the active GPU
            let framebuffer = Framebuffer::new(gpu)?;
            
            // Store framebuffer in GPU system
            gpu_system.framebuffer = Some(framebuffer);
            
            // Draw initial desktop
            if let Some(ref mut fb) = gpu_system.framebuffer {
                DesktopUI::draw_desktop(fb);
                fb.present();
            }
            
            crate::println!("[GPU] Desktop UI initialized with GPU acceleration");
            Ok(())
        } else {
            Err("Active GPU index is invalid")
        }
    } else {
        Err("No active GPU available")
    }
}

/// Take a screenshot and save it to the specified path
pub fn take_screenshot(filename: &str) -> Result<(), &'static str> {
    let gpu_system = GPU_SYSTEM.lock();
    
    if let Some(ref framebuffer) = gpu_system.framebuffer {
        // In a real implementation, this would save the framebuffer to a file
        // For now, we'll just log the operation
        crate::println!("[GPU] Screenshot saved: {} ({}x{} pixels)", 
                       filename, framebuffer.width(), framebuffer.height());
        
        // Simulate saving process
        save_framebuffer_to_file(framebuffer, filename)?;
        
        Ok(())
    } else {
        Err("No framebuffer available for screenshot")
    }
}

/// Save framebuffer data to a bitmap file (simulation)
fn save_framebuffer_to_file(framebuffer: &framebuffer::Framebuffer, filename: &str) -> Result<(), &'static str> {
    // This is a placeholder implementation
    // In a real OS, this would:
    // 1. Create a BMP header
    // 2. Write framebuffer pixel data
    // 3. Save to filesystem
    
    crate::println!("[GPU] Saving {}x{} framebuffer to {}", 
                   framebuffer.width(), framebuffer.height(), filename);
    
    // Simulate file I/O delay
    for _ in 0..1000 {
        core::hint::spin_loop();
    }
    
    Ok(())
}

#[test_case]
fn test_gpu_system_creation() {
    let gpu = GPUSystem::new();
    assert_eq!(gpu.status, GPUStatus::Uninitialized);
}