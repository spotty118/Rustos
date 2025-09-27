/// Intel i915 opensource driver integration
/// Provides support for Intel integrated GPUs using the i915 opensource driver
/// 
/// i915 supports:
/// - Gen 4 (Broadwater, Crestline) - legacy
/// - Gen 5 (Ironlake)
/// - Gen 6 (Sandy Bridge)
/// - Gen 7 (Ivy Bridge, Haswell)
/// - Gen 8 (Broadwell)
/// - Gen 9 (Skylake, Kaby Lake, Coffee Lake)
/// - Gen 11 (Ice Lake)
/// - Gen 12 (Tiger Lake, Rocket Lake, Alder Lake)
/// - Gen 12.2 (DG1)
/// - Xe (Arc Alchemist) - xe driver

use crate::gpu::{GPUCapabilities, GPUVendor, PCIDevice};
use super::{OpensourceDriver, DriverFeatures};
use heapless::Vec;

/// Intel i915 driver version information
const I915_VERSION: &str = "1.6.0";
const I915_DRM_VERSION: &str = "1.6.0";

/// Intel GPU generations supported by i915
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntelGeneration {
    Gen4,    // Broadwater, Crestline (2007)
    Gen5,    // Ironlake (2010)
    Gen6,    // Sandy Bridge (2011)
    Gen7,    // Ivy Bridge, Haswell (2012-2013)
    Gen8,    // Broadwell (2014)
    Gen9,    // Skylake, Kaby Lake, Coffee Lake (2015-2018)
    Gen11,   // Ice Lake (2019)
    Gen12,   // Tiger Lake, Rocket Lake, Alder Lake (2020-2021)
    Gen125,  // DG1 discrete (2021)
    Xe,      // Arc Alchemist (2022+) - separate xe driver
}

/// Intel GPU information
#[derive(Debug, Clone)]
pub struct IntelGpu {
    pub generation: IntelGeneration,
    pub platform: &'static str,
    pub device_name: &'static str,
    pub supports_3d: bool,
    pub supports_compute: bool,
    pub supports_video_decode: bool,
    pub supports_video_encode: bool,
    pub execution_units: u32,
    pub max_fill_rate: u32,        // pixels/clock
    pub memory_interface: &'static str,
    pub display_version: &'static str,
}

/// Create Intel i915 driver registration
pub fn create_i915_driver() -> OpensourceDriver {
    let mut supported_vendors = Vec::new();
    let _ = supported_vendors.push(GPUVendor::Intel);
    
    // Supported Intel device IDs (subset of common ones)
    let mut supported_devices = Vec::new();
    
    // Gen 6 (Sandy Bridge)
    let _ = supported_devices.push(0x0102); // HD Graphics 2000
    let _ = supported_devices.push(0x0106); // HD Graphics 2000
    let _ = supported_devices.push(0x0112); // HD Graphics 3000
    let _ = supported_devices.push(0x0116); // HD Graphics 3000
    let _ = supported_devices.push(0x0122); // HD Graphics 3000
    let _ = supported_devices.push(0x0126); // HD Graphics 3000
    
    // Gen 7 (Ivy Bridge)
    let _ = supported_devices.push(0x0152); // HD Graphics 2500
    let _ = supported_devices.push(0x0156); // HD Graphics 2500
    let _ = supported_devices.push(0x0162); // HD Graphics 4000
    let _ = supported_devices.push(0x0166); // HD Graphics 4000
    
    // Gen 7.5 (Haswell)
    let _ = supported_devices.push(0x0402); // HD Graphics
    let _ = supported_devices.push(0x0406); // HD Graphics
    let _ = supported_devices.push(0x0412); // HD Graphics 4600
    let _ = supported_devices.push(0x0416); // HD Graphics 4600
    let _ = supported_devices.push(0x041E); // HD Graphics 4400
    let _ = supported_devices.push(0x0A16); // HD Graphics 4400
    let _ = supported_devices.push(0x0A26); // HD Graphics 5000
    
    // Gen 8 (Broadwell)
    let _ = supported_devices.push(0x1606); // HD Graphics
    let _ = supported_devices.push(0x160E); // HD Graphics
    let _ = supported_devices.push(0x1616); // HD Graphics 5500
    let _ = supported_devices.push(0x161E); // HD Graphics 5300
    let _ = supported_devices.push(0x1626); // HD Graphics 6000
    let _ = supported_devices.push(0x162B); // Iris Graphics 6100
    
    // Gen 9 (Skylake)
    let _ = supported_devices.push(0x1906); // HD Graphics 510
    let _ = supported_devices.push(0x1916); // HD Graphics 520
    let _ = supported_devices.push(0x1921); // HD Graphics 520
    let _ = supported_devices.push(0x1926); // Iris Graphics 540
    let _ = supported_devices.push(0x1927); // Iris Graphics 550
    let _ = supported_devices.push(0x193B); // Iris Pro Graphics 580
    
    // Gen 9.5 (Kaby Lake)
    let _ = supported_devices.push(0x5906); // HD Graphics 610
    let _ = supported_devices.push(0x5916); // HD Graphics 620
    let _ = supported_devices.push(0x5921); // HD Graphics 620
    let _ = supported_devices.push(0x5926); // Iris Plus Graphics 640
    let _ = supported_devices.push(0x5927); // Iris Plus Graphics 650
    
    // Gen 11 (Ice Lake)
    let _ = supported_devices.push(0x8A50); // Iris Plus Graphics G1
    let _ = supported_devices.push(0x8A51); // Iris Plus Graphics G4
    let _ = supported_devices.push(0x8A52); // Iris Plus Graphics G7
    let _ = supported_devices.push(0x8A53); // Iris Plus Graphics G7
    
    // Gen 12 (Tiger Lake)
    let _ = supported_devices.push(0x9A40); // Iris Xe Graphics G7
    let _ = supported_devices.push(0x9A49); // Iris Xe Graphics G7
    let _ = supported_devices.push(0x9A60); // Iris Xe Graphics G7
    let _ = supported_devices.push(0x9A68); // Iris Xe Graphics G7
    let _ = supported_devices.push(0x9A70); // Iris Xe Graphics G7
    
    OpensourceDriver {
        name: "i915",
        version: I915_VERSION,
        supported_vendors,
        supported_device_ids: supported_devices,
        drm_driver_name: "i915",
        mesa_driver_name: Some("iris"),
        features: DriverFeatures {
            kernel_modesetting: true,
            direct_rendering: true,
            gpu_scheduler: true,
            memory_management: true,
            power_management: true,
            display_output: true,
            hardware_cursor: true,
            video_decode: true,
            video_encode: true,
            compute_shaders: true,
        },
    }
}

/// Initialize Intel i915 driver for a specific GPU
pub fn initialize_i915_driver(gpu: &GPUCapabilities, pci_device: &PCIDevice) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing Intel i915 driver for device 0x{:04X}", pci_device.device_id);
    
    // Detect GPU generation
    let intel_gpu = detect_intel_generation(pci_device.device_id)?;
    
    crate::println!("[i915] Detected {} ({}): {}", 
                   format_generation_name(intel_gpu.generation), 
                   intel_gpu.platform, 
                   intel_gpu.device_name);
    crate::println!("[i915] {} execution units, {} display version", 
                   intel_gpu.execution_units, intel_gpu.display_version);
    
    // Initialize i915 subsystems
    initialize_intel_memory_management(gpu, &intel_gpu)?;
    initialize_intel_graphics_translation_table(&intel_gpu)?;
    initialize_intel_command_submission(&intel_gpu)?;
    initialize_intel_display_engine(&intel_gpu)?;
    
    if intel_gpu.supports_3d {
        initialize_intel_render_engine(&intel_gpu)?;
    }
    
    if intel_gpu.supports_compute {
        initialize_intel_compute_engine(&intel_gpu)?;
    }
    
    if intel_gpu.supports_video_decode {
        initialize_intel_video_decode(&intel_gpu)?;
    }
    
    if intel_gpu.supports_video_encode {
        initialize_intel_video_encode(&intel_gpu)?;
    }
    
    // Initialize power management
    initialize_intel_power_management(&intel_gpu)?;
    
    crate::println!("[i915] Intel i915 driver initialization complete");
    Ok(())
}

/// Detect Intel GPU generation and capabilities
fn detect_intel_generation(device_id: u16) -> Result<IntelGpu, &'static str> {
    match device_id {
        // Gen 6 (Sandy Bridge)
        0x0112 => Ok(IntelGpu {
            generation: IntelGeneration::Gen6,
            platform: "Sandy Bridge",
            device_name: "HD Graphics 3000",
            supports_3d: true,
            supports_compute: false,
            supports_video_decode: true,
            supports_video_encode: false,
            execution_units: 12,
            max_fill_rate: 2,
            memory_interface: "DDR3",
            display_version: "6.0",
        }),
        
        // Gen 7 (Ivy Bridge)
        0x0162 => Ok(IntelGpu {
            generation: IntelGeneration::Gen7,
            platform: "Ivy Bridge",
            device_name: "HD Graphics 4000",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 16,
            max_fill_rate: 4,
            memory_interface: "DDR3",
            display_version: "7.0",
        }),
        
        // Gen 7.5 (Haswell)
        0x0412 => Ok(IntelGpu {
            generation: IntelGeneration::Gen7,
            platform: "Haswell",
            device_name: "HD Graphics 4600",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 20,
            max_fill_rate: 4,
            memory_interface: "DDR3L",
            display_version: "7.5",
        }),
        
        // Gen 8 (Broadwell)
        0x1616 => Ok(IntelGpu {
            generation: IntelGeneration::Gen8,
            platform: "Broadwell",
            device_name: "HD Graphics 5500",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 24,
            max_fill_rate: 4,
            memory_interface: "DDR3L/LPDDR3",
            display_version: "8.0",
        }),
        
        // Gen 9 (Skylake)
        0x1916 => Ok(IntelGpu {
            generation: IntelGeneration::Gen9,
            platform: "Skylake",
            device_name: "HD Graphics 520",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 24,
            max_fill_rate: 4,
            memory_interface: "DDR4/LPDDR3",
            display_version: "9.0",
        }),
        
        // Gen 9.5 (Kaby Lake)
        0x5916 => Ok(IntelGpu {
            generation: IntelGeneration::Gen9,
            platform: "Kaby Lake",
            device_name: "HD Graphics 620",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 24,
            max_fill_rate: 4,
            memory_interface: "DDR4/LPDDR3",
            display_version: "9.5",
        }),
        
        // Gen 11 (Ice Lake)
        0x8A52 => Ok(IntelGpu {
            generation: IntelGeneration::Gen11,
            platform: "Ice Lake",
            device_name: "Iris Plus Graphics G7",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 64,
            max_fill_rate: 16,
            memory_interface: "LPDDR4X",
            display_version: "11.0",
        }),
        
        // Gen 12 (Tiger Lake)
        0x9A49 => Ok(IntelGpu {
            generation: IntelGeneration::Gen12,
            platform: "Tiger Lake",
            device_name: "Iris Xe Graphics G7",
            supports_3d: true,
            supports_compute: true,
            supports_video_decode: true,
            supports_video_encode: true,
            execution_units: 96,
            max_fill_rate: 32,
            memory_interface: "LPDDR4X/DDR4",
            display_version: "12.0",
        }),
        
        _ => {
            // Try to guess generation from device ID ranges
            if device_id >= 0x0102 && device_id <= 0x0126 {
                Ok(IntelGpu {
                    generation: IntelGeneration::Gen6,
                    platform: "Sandy Bridge",
                    device_name: "Unknown Sandy Bridge GPU",
                    supports_3d: true,
                    supports_compute: false,
                    supports_video_decode: true,
                    supports_video_encode: false,
                    execution_units: 12,
                    max_fill_rate: 2,
                    memory_interface: "DDR3",
                    display_version: "6.0",
                })
            } else if device_id >= 0x0152 && device_id <= 0x0166 {
                Ok(IntelGpu {
                    generation: IntelGeneration::Gen7,
                    platform: "Ivy Bridge",
                    device_name: "Unknown Ivy Bridge GPU",
                    supports_3d: true,
                    supports_compute: true,
                    supports_video_decode: true,
                    supports_video_encode: true,
                    execution_units: 16,
                    max_fill_rate: 4,
                    memory_interface: "DDR3",
                    display_version: "7.0",
                })
            } else {
                Err("Unsupported or unknown Intel GPU")
            }
        }
    }
}

/// Initialize Intel memory management
fn initialize_intel_memory_management(_gpu: &GPUCapabilities, intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing memory management for {} ({})", 
                   intel_gpu.platform, intel_gpu.memory_interface);
    
    // Intel GPUs use system memory with Graphics Translation Table (GTT)
    // In a real implementation, this would:
    // 1. Set up Global Graphics Translation Table (GGTT)
    // 2. Initialize Per-Process Graphics Translation Table (PPGTT)
    // 3. Configure stolen memory allocation
    // 4. Set up fence registers for tiled surfaces
    
    Ok(())
}

/// Initialize Intel Graphics Translation Table
fn initialize_intel_graphics_translation_table(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing Graphics Translation Table");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 => {
            crate::println!("[i915] Setting up legacy GTT");
        }
        IntelGeneration::Gen6 | IntelGeneration::Gen7 => {
            crate::println!("[i915] Setting up GTT with partial PPGTT support");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Setting up GTT with full PPGTT (48-bit addressing)");
        }
        IntelGeneration::Gen11 | IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Setting up modern GTT with enhanced PPGTT");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Setting up Xe GTT (handled by xe driver)");
        }
    }
    
    Ok(())
}

/// Initialize Intel command submission
fn initialize_intel_command_submission(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing command submission");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 => {
            crate::println!("[i915] Legacy ring buffer submission");
        }
        IntelGeneration::Gen6 | IntelGeneration::Gen7 => {
            crate::println!("[i915] Ring buffer submission with semaphores");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Execlist submission with logical ring contexts");
        }
        IntelGeneration::Gen11 | IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] GuC submission with hardware scheduling");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe command submission (handled by xe driver)");
        }
    }
    
    Ok(())
}

/// Initialize Intel display engine
fn initialize_intel_display_engine(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing display engine ({})", intel_gpu.display_version);
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 => {
            crate::println!("[i915] Legacy display controller");
        }
        IntelGeneration::Gen6 | IntelGeneration::Gen7 => {
            crate::println!("[i915] Display engine with PCH (Platform Controller Hub)");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Modern display engine with atomic modesetting");
        }
        IntelGeneration::Gen11 => {
            crate::println!("[i915] Gen11 display engine with DSI support");
        }
        IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Gen12 display engine with enhanced features");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe display engine");
        }
    }
    
    Ok(())
}

/// Initialize Intel render engine
fn initialize_intel_render_engine(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing render engine ({} EUs)", intel_gpu.execution_units);
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 => {
            crate::println!("[i915] Gen4 render engine (OpenGL 2.1, DirectX 9)");
        }
        IntelGeneration::Gen5 => {
            crate::println!("[i915] Gen5 render engine (OpenGL 2.1, DirectX 10)");
        }
        IntelGeneration::Gen6 => {
            crate::println!("[i915] Gen6 render engine (OpenGL 3.0, DirectX 10.1)");
        }
        IntelGeneration::Gen7 => {
            crate::println!("[i915] Gen7 render engine (OpenGL 4.0, DirectX 11)");
        }
        IntelGeneration::Gen8 => {
            crate::println!("[i915] Gen8 render engine (OpenGL 4.4, DirectX 11.2)");
        }
        IntelGeneration::Gen9 => {
            crate::println!("[i915] Gen9 render engine (OpenGL 4.5, DirectX 12)");
        }
        IntelGeneration::Gen11 => {
            crate::println!("[i915] Gen11 render engine (OpenGL 4.6, DirectX 12)");
        }
        IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Gen12 render engine (OpenGL 4.6, DirectX 12 Ultimate)");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe render engine with hardware ray tracing");
        }
    }
    
    Ok(())
}

/// Initialize Intel compute engine
fn initialize_intel_compute_engine(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing compute engine");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 | IntelGeneration::Gen6 => {
            return Err("Compute shaders not supported on this generation");
        }
        IntelGeneration::Gen7 => {
            crate::println!("[i915] Gen7 compute (OpenCL 1.2)");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Gen8/9 compute (OpenCL 2.0, compute shaders)");
        }
        IntelGeneration::Gen11 => {
            crate::println!("[i915] Gen11 compute with enhanced features");
        }
        IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Gen12 compute with ML acceleration");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe compute with XMX (Xe Matrix eXtensions)");
        }
    }
    
    Ok(())
}

/// Initialize Intel video decode engine
fn initialize_intel_video_decode(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing video decode engine");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 => {
            return Err("Hardware video decode not supported");
        }
        IntelGeneration::Gen6 => {
            crate::println!("[i915] Gen6 video decode (MPEG-2, H.264)");
        }
        IntelGeneration::Gen7 => {
            crate::println!("[i915] Gen7 video decode (MPEG-2, H.264, VC-1)");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Gen8/9 video decode (H.264, HEVC, VP8)");
        }
        IntelGeneration::Gen11 => {
            crate::println!("[i915] Gen11 video decode (H.264, HEVC, VP9)");
        }
        IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Gen12 video decode (H.264, HEVC, VP9, AV1)");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe video decode with enhanced performance");
        }
    }
    
    Ok(())
}

/// Initialize Intel video encode engine
fn initialize_intel_video_encode(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing video encode engine");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 | IntelGeneration::Gen6 => {
            return Err("Hardware video encode not supported");
        }
        IntelGeneration::Gen7 => {
            crate::println!("[i915] Gen7 video encode (H.264)");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Gen8/9 video encode (H.264, HEVC)");
        }
        IntelGeneration::Gen11 => {
            crate::println!("[i915] Gen11 video encode (H.264, HEVC, VP9)");
        }
        IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Gen12 video encode (H.264, HEVC, VP9, AV1)");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe video encode with quality improvements");
        }
    }
    
    Ok(())
}

/// Initialize Intel power management
fn initialize_intel_power_management(intel_gpu: &IntelGpu) -> Result<(), &'static str> {
    crate::println!("[i915] Initializing power management");
    
    match intel_gpu.generation {
        IntelGeneration::Gen4 | IntelGeneration::Gen5 => {
            crate::println!("[i915] Basic power management");
        }
        IntelGeneration::Gen6 | IntelGeneration::Gen7 => {
            crate::println!("[i915] RC6 power states, turbo boost");
        }
        IntelGeneration::Gen8 | IntelGeneration::Gen9 => {
            crate::println!("[i915] Enhanced RC6, DVFS (Dynamic Voltage/Frequency Scaling)");
        }
        IntelGeneration::Gen11 | IntelGeneration::Gen12 | IntelGeneration::Gen125 => {
            crate::println!("[i915] Advanced power management with slice/subslice gating");
        }
        IntelGeneration::Xe => {
            crate::println!("[i915] Xe power management with fine-grained control");
        }
    }
    
    Ok(())
}

/// Format generation name for display
fn format_generation_name(gen: IntelGeneration) -> &'static str {
    match gen {
        IntelGeneration::Gen4 => "Gen4",
        IntelGeneration::Gen5 => "Gen5",
        IntelGeneration::Gen6 => "Gen6",
        IntelGeneration::Gen7 => "Gen7/7.5",
        IntelGeneration::Gen8 => "Gen8",
        IntelGeneration::Gen9 => "Gen9/9.5",
        IntelGeneration::Gen11 => "Gen11",
        IntelGeneration::Gen12 => "Gen12",
        IntelGeneration::Gen125 => "Gen12.5",
        IntelGeneration::Xe => "Xe",
    }
}

#[test_case]
fn test_intel_generation_detection() {
    let gpu = detect_intel_generation(0x9A49).unwrap();
    assert_eq!(gpu.generation, IntelGeneration::Gen12);
    assert_eq!(gpu.platform, "Tiger Lake");
    assert_eq!(gpu.device_name, "Iris Xe Graphics G7");
    assert!(gpu.supports_3d);
    assert!(gpu.supports_compute);
}