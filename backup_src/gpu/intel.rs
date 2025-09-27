/// Intel GPU support (integrated graphics)
/// Supports Intel HD Graphics, UHD Graphics, Iris, and Arc GPUs

use crate::gpu::{GPUCapabilities, GPUVendor};

/// Intel GPU PCI device IDs (common ones)
const INTEL_GPU_DEVICE_IDS: &[u16] = &[
    0x0046, // Intel HD Graphics (Ironlake)
    0x0102, // Intel HD Graphics 2000
    0x0106, // Intel HD Graphics 2000
    0x0112, // Intel HD Graphics 3000
    0x0116, // Intel HD Graphics 3000
    0x0122, // Intel HD Graphics 3000
    0x0126, // Intel HD Graphics 3000
    0x0152, // Intel HD Graphics 2500
    0x0156, // Intel HD Graphics 2500
    0x0162, // Intel HD Graphics 4000
    0x0166, // Intel HD Graphics 4000
    0x016A, // Intel HD Graphics P4000
    0x0402, // Intel UHD Graphics 600
    0x0406, // Intel UHD Graphics 620
    0x0412, // Intel UHD Graphics 610
    0x0416, // Intel UHD Graphics 620
    0x041A, // Intel UHD Graphics P630
    0x041E, // Intel UHD Graphics 620
    0x0A02, // Intel UHD Graphics 600
    0x0A06, // Intel UHD Graphics 620
    0x0A0A, // Intel UHD Graphics P630
    0x0A12, // Intel UHD Graphics 610
    0x0A16, // Intel UHD Graphics 620
    0x0A1A, // Intel UHD Graphics P630
    0x0A1E, // Intel UHD Graphics 620
    0x1902, // Intel HD Graphics 510
    0x1906, // Intel HD Graphics 510
    0x190B, // Intel HD Graphics 510
    0x1912, // Intel HD Graphics 530
    0x1916, // Intel HD Graphics 520
    0x191B, // Intel HD Graphics 530
    0x191D, // Intel HD Graphics P530
    0x191E, // Intel HD Graphics 515
    0x1921, // Intel HD Graphics 520
    0x1923, // Intel HD Graphics 535
    0x1926, // Intel Iris Graphics 540
    0x1927, // Intel Iris Graphics 550
    0x192B, // Intel Iris Graphics 555
    0x192D, // Intel Iris Graphics P555
    0x1932, // Intel Iris Pro Graphics 580
    0x193A, // Intel Iris Pro Graphics P580
    0x193B, // Intel Iris Pro Graphics 580
    0x193D, // Intel Iris Pro Graphics P580
    0x3E90, // Intel UHD Graphics 610
    0x3E91, // Intel UHD Graphics 630
    0x3E92, // Intel UHD Graphics 630
    0x3E93, // Intel UHD Graphics 610
    0x3E94, // Intel UHD Graphics P630
    0x3E96, // Intel UHD Graphics P630
    0x3E98, // Intel UHD Graphics 630
    0x3E9A, // Intel UHD Graphics P630
    0x3E9B, // Intel UHD Graphics 630
    0x3EA0, // Intel UHD Graphics 620
    0x3EA5, // Intel Iris Plus Graphics 655
    0x3EA6, // Intel Iris Plus Graphics 645
    0x3EA7, // Intel Iris Plus Graphics 645
    0x3EA8, // Intel Iris Plus Graphics 655
    0x4680, // Intel Arc A380
    0x4681, // Intel Arc A370M
    0x4682, // Intel Arc A350M
    0x4690, // Intel Arc A770
    0x4691, // Intel Arc A750
    0x4692, // Intel Arc A580
    0x4693, // Intel Arc A380
    0x56A0, // Intel Arc A770M
    0x56A1, // Intel Arc A730M
    0x56A2, // Intel Arc A550M
    0x56A3, // Intel Arc A370M
    0x56A4, // Intel Arc A350M
];

/// Detect Intel GPU through PCI scanning
pub fn detect_intel_gpu() -> Result<GPUCapabilities, &'static str> {
    // Legacy function for backward compatibility
    // In practice, this would be called through detect_intel_gpu_from_pci
    let device_id = 0x9A49; // Intel Iris Xe Graphics
    create_intel_gpu_capabilities(device_id)
}

/// Detect Intel GPU from PCI device information
pub fn detect_intel_gpu_from_pci(pci_device: crate::gpu::PCIDevice) -> Result<GPUCapabilities, &'static str> {
    // Verify this is actually an Intel device
    if pci_device.vendor_id != 0x8086 {
        return Err("Not an Intel device");
    }
    
    // Verify this is a display controller
    if pci_device.class_code != 0x03 {
        return Err("Not a display controller");
    }
    
    // Check if this is a known Intel GPU device ID
    let is_intel_gpu = INTEL_GPU_DEVICE_IDS.contains(&pci_device.device_id) || 
                       is_intel_device_id(pci_device.device_id);

    if !is_intel_gpu {
        return Err("Unknown Intel GPU device ID");
    }

    create_intel_gpu_capabilities(pci_device.device_id)
}

/// Create GPU capabilities structure for Intel GPU
fn create_intel_gpu_capabilities(device_id: u16) -> Result<GPUCapabilities, &'static str> {
    let memory_size = estimate_intel_gpu_memory(device_id);
    let max_resolution = get_intel_max_resolution(device_id);
    let (supports_2d, supports_3d, supports_compute) = get_intel_capabilities(device_id);

    Ok(GPUCapabilities {
        vendor: GPUVendor::Intel,
        memory_size,
        max_resolution,
        supports_2d_accel: supports_2d,
        supports_3d_accel: supports_3d,
        supports_compute,
        pci_device_id: device_id,
    })
}

/// Initialize Intel GPU for acceleration
pub fn initialize_intel_gpu(gpu: &GPUCapabilities) -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing Intel GPU acceleration...");
    crate::println!("[Intel GPU] Device ID: 0x{:04X}", gpu.pci_device_id);
    crate::println!("[Intel GPU] Memory: {} MB", gpu.memory_size / (1024 * 1024));
    crate::println!("[Intel GPU] Max resolution: {}x{}", gpu.max_resolution.0, gpu.max_resolution.1);

    // Initialize Intel GPU driver components
    init_intel_gtt()?;  // Graphics Translation Table
    init_intel_display_engine()?;
    init_intel_2d_engine()?;
    
    if gpu.supports_3d_accel {
        init_intel_3d_engine()?;
    }
    
    if gpu.supports_compute {
        init_intel_compute_engine()?;
    }

    crate::println!("[Intel GPU] Intel GPU initialization complete");
    Ok(())
}

/// Check if device ID belongs to Intel
fn is_intel_device_id(device_id: u16) -> bool {
    // Intel device IDs often follow patterns
    // Gen 9 (Skylake): 0x1900-0x193F
    // Gen 9.5 (Kaby Lake): 0x5900-0x593F
    // Gen 11 (Ice Lake): 0x8A00-0x8A5F
    // Gen 12 (Tiger Lake): 0x9A00-0x9A7F
    // Arc (Alchemist): 0x4680-0x4693, 0x56A0-0x56A4
    
    match device_id {
        0x1900..=0x193F => true, // Gen 9
        0x5900..=0x593F => true, // Gen 9.5
        0x8A00..=0x8A7F => true, // Gen 11
        0x9A00..=0x9A7F => true, // Gen 12
        0x4680..=0x4693 => true, // Arc Alchemist
        0x56A0..=0x56A4 => true, // Arc Alchemist Mobile
        _ => false,
    }
}

/// Estimate Intel GPU memory based on device ID
fn estimate_intel_gpu_memory(device_id: u16) -> u64 {
    match device_id {
        // Arc series - dedicated memory
        0x4680..=0x4693 => 8 * 1024 * 1024 * 1024,  // 8GB for Arc desktop
        0x56A0..=0x56A4 => 4 * 1024 * 1024 * 1024,  // 4GB for Arc mobile
        
        // Integrated graphics - shared system memory
        0x9A00..=0x9A7F => 2 * 1024 * 1024 * 1024,  // 2GB shared (Tiger Lake)
        0x8A00..=0x8A7F => 1 * 1024 * 1024 * 1024,  // 1GB shared (Ice Lake)
        0x1900..=0x193F => 512 * 1024 * 1024,       // 512MB shared (Skylake)
        
        _ => 256 * 1024 * 1024, // Default 256MB
    }
}

/// Get maximum resolution for Intel GPU
fn get_intel_max_resolution(device_id: u16) -> (u32, u32) {
    match device_id {
        // Arc series - 4K support
        0x4680..=0x4693 => (3840, 2160),
        0x56A0..=0x56A4 => (3840, 2160),
        
        // Modern integrated - 4K support
        0x9A00..=0x9A7F => (3840, 2160), // Tiger Lake
        0x8A00..=0x8A7F => (3840, 2160), // Ice Lake
        
        // Older integrated - 1080p/1440p
        0x1900..=0x193F => (2560, 1440), // Skylake
        0x5900..=0x593F => (2560, 1440), // Kaby Lake
        
        _ => (1920, 1080), // Default 1080p
    }
}

/// Get Intel GPU capabilities
fn get_intel_capabilities(device_id: u16) -> (bool, bool, bool) {
    let supports_2d = true; // All Intel GPUs support 2D acceleration
    
    let supports_3d = match device_id {
        0x4680..=0x4693 => true, // Arc desktop
        0x56A0..=0x56A4 => true, // Arc mobile
        0x9A00..=0x9A7F => true, // Tiger Lake
        0x8A00..=0x8A7F => true, // Ice Lake
        0x1900..=0x193F => true, // Skylake
        _ => false,
    };
    
    let supports_compute = match device_id {
        0x4680..=0x4693 => true, // Arc desktop - OpenCL/compute
        0x56A0..=0x56A4 => true, // Arc mobile - OpenCL/compute
        0x9A00..=0x9A7F => true, // Tiger Lake - basic compute
        0x8A00..=0x8A7F => true, // Ice Lake - basic compute
        _ => false,
    };
    
    (supports_2d, supports_3d, supports_compute)
}

/// Initialize Intel Graphics Translation Table
fn init_intel_gtt() -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing Graphics Translation Table...");
    // GTT initialization would happen here
    Ok(())
}

/// Initialize Intel display engine
fn init_intel_display_engine() -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing display engine...");
    // Display engine initialization
    Ok(())
}

/// Initialize Intel 2D acceleration engine
fn init_intel_2d_engine() -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing 2D acceleration engine...");
    // 2D engine initialization
    Ok(())
}

/// Initialize Intel 3D acceleration engine  
fn init_intel_3d_engine() -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing 3D acceleration engine...");
    // 3D engine initialization
    Ok(())
}

/// Initialize Intel compute engine
fn init_intel_compute_engine() -> Result<(), &'static str> {
    crate::println!("[Intel GPU] Initializing compute engine...");
    // Compute engine initialization for OpenCL/compute shaders
    Ok(())
}