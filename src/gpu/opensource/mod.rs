/// Opensource GPU driver integration framework
/// Provides compatibility layer for Linux DRM/KMS and Mesa drivers
/// 
/// This module enables RustOS to interface with mature opensource GPU drivers:
/// - Nouveau (NVIDIA opensource driver)
/// - AMDGPU (AMD opensource driver) 
/// - i915 (Intel opensource driver)
/// - RadeonSI (AMD Radeon driver)
/// - Panfrost (ARM Mali driver)

use crate::gpu::{GPUCapabilities, GPUVendor, PCIDevice};
use heapless::Vec;

pub mod drm_compat;
pub mod mesa_compat;
pub mod nouveau;
pub mod amdgpu;
pub mod i915;

/// Opensource driver information
#[derive(Debug, Clone)]
pub struct OpensourceDriver {
    pub name: &'static str,
    pub version: &'static str,
    pub supported_vendors: Vec<GPUVendor, 4>,
    pub supported_device_ids: Vec<u16, 32>,
    pub drm_driver_name: &'static str,
    pub mesa_driver_name: Option<&'static str>,
    pub features: DriverFeatures,
}

/// Driver feature flags
#[derive(Debug, Clone, Copy)]
pub struct DriverFeatures {
    pub kernel_modesetting: bool,     // KMS support
    pub direct_rendering: bool,       // DRI support  
    pub gpu_scheduler: bool,          // GPU command scheduling
    pub memory_management: bool,      // GPU memory management
    pub power_management: bool,       // Runtime PM
    pub display_output: bool,         // Display connector support
    pub hardware_cursor: bool,        // Hardware cursor support
    pub video_decode: bool,          // Hardware video decoding
    pub video_encode: bool,          // Hardware video encoding
    pub compute_shaders: bool,       // Compute/GPGPU support
}

/// Opensource driver registry
pub struct OpensourceDriverRegistry {
    drivers: Vec<OpensourceDriver, 8>,
    active_drivers: Vec<usize, 4>, // Indices of active drivers
}

impl OpensourceDriverRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            drivers: Vec::new(),
            active_drivers: Vec::new(),
        };
        
        // Register all supported opensource drivers
        registry.register_default_drivers();
        registry
    }
    
    /// Register default opensource GPU drivers
    fn register_default_drivers(&mut self) {
        // Register Nouveau (NVIDIA opensource)
        if let Err(_) = self.drivers.push(nouveau::create_nouveau_driver()) {
            crate::println!("[OPENSOURCE] Warning: Could not register Nouveau driver");
        }
        
        // Register AMDGPU (AMD opensource)
        if let Err(_) = self.drivers.push(amdgpu::create_amdgpu_driver()) {
            crate::println!("[OPENSOURCE] Warning: Could not register AMDGPU driver");
        }
        
        // Register i915 (Intel opensource)
        if let Err(_) = self.drivers.push(i915::create_i915_driver()) {
            crate::println!("[OPENSOURCE] Warning: Could not register i915 driver");
        }
    }
    
    /// Find compatible opensource driver for a PCI device
    pub fn find_driver_for_device(&self, pci_device: &PCIDevice) -> Option<&OpensourceDriver> {
        let vendor = match pci_device.vendor_id {
            0x8086 => GPUVendor::Intel,
            0x10DE => GPUVendor::Nvidia,
            0x1002 => GPUVendor::AMD,
            _ => GPUVendor::Unknown,
        };
        
        for driver in &self.drivers {
            // Check if driver supports this vendor
            if driver.supported_vendors.contains(&vendor) {
                // Check if driver supports this specific device ID
                if driver.supported_device_ids.contains(&pci_device.device_id) {
                    return Some(driver);
                }
            }
        }
        
        None
    }
    
    /// Initialize opensource driver for a GPU
    pub fn initialize_driver(&mut self, gpu: &GPUCapabilities, pci_device: &PCIDevice) -> Result<(), &'static str> {
        if let Some(driver) = self.find_driver_for_device(pci_device) {
            crate::println!("[OPENSOURCE] Initializing {} driver for {:?} GPU", 
                           driver.name, gpu.vendor);
            
            // Initialize DRM compatibility layer first
            drm_compat::initialize_drm_subsystem()?;
            
            // Initialize the specific driver
            match driver.drm_driver_name {
                "nouveau" => nouveau::initialize_nouveau_driver(gpu, pci_device)?,
                "amdgpu" => amdgpu::initialize_amdgpu_driver(gpu, pci_device)?,
                "i915" => i915::initialize_i915_driver(gpu, pci_device)?,
                _ => return Err("Unsupported opensource driver"),
            }
            
            // Initialize Mesa compatibility if supported
            if let Some(mesa_driver) = driver.mesa_driver_name {
                mesa_compat::initialize_mesa_driver(mesa_driver, gpu)?;
            }
            
            crate::println!("[OPENSOURCE] {} driver initialized successfully", driver.name);
            Ok(())
        } else {
            Err("No compatible opensource driver found")
        }
    }
    
    /// Get list of all registered drivers
    pub fn get_registered_drivers(&self) -> &Vec<OpensourceDriver, 8> {
        &self.drivers
    }
    
    /// Check if opensource driver is available for a vendor
    pub fn has_driver_for_vendor(&self, vendor: GPUVendor) -> bool {
        self.drivers.iter().any(|driver| driver.supported_vendors.contains(&vendor))
    }
}

/// Initialize the opensource driver subsystem
pub fn init_opensource_drivers() -> Result<OpensourceDriverRegistry, &'static str> {
    crate::println!("[OPENSOURCE] Initializing opensource GPU driver subsystem...");
    
    let registry = OpensourceDriverRegistry::new();
    
    crate::println!("[OPENSOURCE] Registered {} opensource drivers", registry.drivers.len());
    
    // Print available drivers
    for driver in &registry.drivers {
        crate::println!("[OPENSOURCE] - {} v{} ({})", 
                       driver.name, driver.version, driver.drm_driver_name);
    }
    
    Ok(registry)
}

/// Create fallback driver capabilities for unknown devices
pub fn create_fallback_capabilities(pci_device: &PCIDevice) -> GPUCapabilities {
    crate::println!("[OPENSOURCE] Creating fallback capabilities for device 0x{:04X}:0x{:04X}", 
                   pci_device.vendor_id, pci_device.device_id);
    
    let vendor = match pci_device.vendor_id {
        0x8086 => GPUVendor::Intel,
        0x10DE => GPUVendor::Nvidia,
        0x1002 => GPUVendor::AMD,
        _ => GPUVendor::Unknown,
    };
    
    GPUCapabilities {
        vendor,
        memory_size: 128 * 1024 * 1024, // Default 128MB
        max_resolution: (1920, 1080),   // Default 1080p
        supports_2d_accel: true,         // Basic 2D assumed
        supports_3d_accel: false,        // Conservative default
        supports_compute: false,         // Conservative default
        pci_device_id: pci_device.device_id,
    }
}

#[test_case]
fn test_opensource_driver_registry() {
    let registry = OpensourceDriverRegistry::new();
    assert!(registry.drivers.len() > 0);
    assert!(registry.has_driver_for_vendor(GPUVendor::Intel));
    assert!(registry.has_driver_for_vendor(GPUVendor::Nvidia));
    assert!(registry.has_driver_for_vendor(GPUVendor::AMD));
}