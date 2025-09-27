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
    GPU_SYSTEM.lock().get_status()
}

/// Check if GPU acceleration is available
pub fn is_gpu_acceleration_available() -> bool {
    GPU_SYSTEM.lock().is_acceleration_available()
}

/// Get information about active GPU
pub fn get_active_gpu_info() -> Option<GPUCapabilities> {
    GPU_SYSTEM.lock().get_active_gpu().cloned()
}

/// Clear screen using GPU acceleration if available
pub fn gpu_clear_screen(color: u32) {
    GPU_SYSTEM.lock().clear_screen(color);
}

/// Draw rectangle using GPU acceleration
pub fn gpu_draw_rect(x: u32, y: u32, width: u32, height: u32, color: u32) {
    GPU_SYSTEM.lock().draw_rect(x, y, width, height, color);
}

/// Present/update display
pub fn gpu_present() {
    GPU_SYSTEM.lock().present();
}

#[test_case]
fn test_gpu_system_creation() {
    let gpu = GPUSystem::new();
    assert_eq!(gpu.get_status(), GPUStatus::Uninitialized);
}