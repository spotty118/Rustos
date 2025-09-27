/// Peripheral device driver framework for RustOS
/// Provides a unified interface for all peripheral devices
///
/// This module provides:
/// - PCI device enumeration and detection
/// - Device driver registration and matching
/// - Hardware abstraction layer for various peripherals
/// - Power management and device lifecycle

use heapless::Vec;
use core::fmt;

pub mod pci;
pub mod network;
pub mod storage;
pub mod audio;
pub mod input;
pub mod usb;

/// Peripheral device types supported by the driver framework
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    NetworkInterface,
    StorageController,
    AudioDevice,
    InputDevice,
    USBController,
    GraphicsAdapter,
    BridgeDevice,
    Unknown,
}

/// Peripheral device vendor identification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceVendor {
    Intel,
    AMD,
    Nvidia,
    Realtek,
    Broadcom,
    QualcommAtheros,
    Marvell,
    VIA,
    SiS,
    Creative,
    Logitech,
    Microsoft,
    Apple,
    Unknown(u16),
}

impl DeviceVendor {
    pub fn from_pci_id(vendor_id: u16) -> Self {
        match vendor_id {
            0x8086 => DeviceVendor::Intel,
            0x1002 => DeviceVendor::AMD,
            0x10DE => DeviceVendor::Nvidia,
            0x10EC => DeviceVendor::Realtek,
            0x14E4 => DeviceVendor::Broadcom,
            0x168C => DeviceVendor::QualcommAtheros,
            0x11AB => DeviceVendor::Marvell,
            0x1106 => DeviceVendor::VIA,
            0x1039 => DeviceVendor::SiS,
            0x1102 => DeviceVendor::Creative,
            _ => DeviceVendor::Unknown(vendor_id),
        }
    }
}

impl fmt::Display for DeviceVendor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeviceVendor::Intel => write!(f, "Intel"),
            DeviceVendor::AMD => write!(f, "AMD"),
            DeviceVendor::Nvidia => write!(f, "NVIDIA"),
            DeviceVendor::Realtek => write!(f, "Realtek"),
            DeviceVendor::Broadcom => write!(f, "Broadcom"),
            DeviceVendor::QualcommAtheros => write!(f, "Qualcomm Atheros"),
            DeviceVendor::Marvell => write!(f, "Marvell"),
            DeviceVendor::VIA => write!(f, "VIA"),
            DeviceVendor::SiS => write!(f, "SiS"),
            DeviceVendor::Creative => write!(f, "Creative"),
            DeviceVendor::Logitech => write!(f, "Logitech"),
            DeviceVendor::Microsoft => write!(f, "Microsoft"),
            DeviceVendor::Apple => write!(f, "Apple"),
            DeviceVendor::Unknown(id) => write!(f, "Unknown (0x{:04X})", id),
        }
    }
}

/// Generic peripheral device information
#[derive(Debug, Clone)]
pub struct PeripheralDevice {
    pub vendor: DeviceVendor,
    pub device_id: u16,
    pub device_type: DeviceType,
    pub class_code: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub base_addresses: Vec<u32, 6>,
    pub irq_line: u8,
    pub driver_name: Option<&'static str>,
}

/// Driver capabilities and features
#[derive(Debug, Clone, Copy)]
pub struct DriverCapabilities {
    pub supports_power_management: bool,
    pub supports_msi: bool, // Message Signaled Interrupts
    pub supports_dma: bool,
    pub requires_firmware: bool,
    pub hot_pluggable: bool,
}

/// Generic peripheral driver interface
pub trait PeripheralDriver {
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn supported_devices(&self) -> &[u16]; // Device IDs
    fn supported_vendor(&self) -> DeviceVendor;
    fn capabilities(&self) -> DriverCapabilities;

    /// Initialize the driver for a specific device
    fn initialize(&mut self, device: &PeripheralDevice) -> Result<(), &'static str>;

    /// Reset the device
    fn reset(&mut self) -> Result<(), &'static str>;

    /// Put device into low power state
    fn suspend(&mut self) -> Result<(), &'static str>;

    /// Resume device from low power state
    fn resume(&mut self) -> Result<(), &'static str>;

    /// Handle device interrupt
    fn handle_interrupt(&mut self) -> Result<(), &'static str>;
}

/// Simple registry using static allocation (no Box needed)
pub struct PeripheralDriverRegistry {
    network_drivers: Vec<network::NetworkDriverWrapper, 8>,
    storage_drivers: Vec<storage::StorageDriverWrapper, 8>,
    audio_drivers: Vec<audio::AudioDriverWrapper, 4>,
    input_drivers: Vec<input::InputDriverWrapper, 4>,
    usb_drivers: Vec<usb::USBDriverWrapper, 4>,
    detected_devices: Vec<PeripheralDevice, 64>,
}

impl PeripheralDriverRegistry {
    pub fn new() -> Self {
        Self {
            network_drivers: Vec::new(),
            storage_drivers: Vec::new(),
            audio_drivers: Vec::new(),
            input_drivers: Vec::new(),
            usb_drivers: Vec::new(),
            detected_devices: Vec::new(),
        }
    }

    /// Enumerate and detect all PCI devices
    pub fn enumerate_devices(&mut self) -> Result<(), &'static str> {
        crate::println!("[PERIPHERAL] Enumerating PCI devices...");

        // Scan PCI bus for devices
        let pci_devices = pci::scan_pci_bus()?;

        for pci_dev in pci_devices {
            let device = PeripheralDevice {
                vendor: DeviceVendor::from_pci_id(pci_dev.vendor_id),
                device_id: pci_dev.device_id,
                device_type: classify_device(pci_dev.class_code, pci_dev.subclass),
                class_code: pci_dev.class_code,
                subclass: pci_dev.subclass,
                prog_if: pci_dev.prog_if,
                bus: pci_dev.bus,
                device: pci_dev.device,
                function: pci_dev.function,
                base_addresses: pci_dev.base_addresses,
                irq_line: pci_dev.irq_line,
                driver_name: None,
            };

            if let Err(_) = self.detected_devices.push(device) {
                crate::println!("[PERIPHERAL] Warning: Device registry full");
                break;
            }
        }

        crate::println!("[PERIPHERAL] Detected {} devices", self.detected_devices.len());
        Ok(())
    }

    /// Match detected devices with available drivers
    pub fn match_drivers(&mut self) -> Result<(), &'static str> {
        for device in &mut self.detected_devices {
            let mut matched = false;

            // Check network drivers
            for net_driver in &self.network_drivers {
                if net_driver.supported_vendor() == device.vendor &&
                   net_driver.supported_devices().contains(&device.device_id) {
                    device.driver_name = Some(net_driver.name());
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Check storage drivers
                for storage_driver in &self.storage_drivers {
                    if storage_driver.supported_vendor() == device.vendor &&
                       storage_driver.supported_devices().contains(&device.device_id) {
                        device.driver_name = Some(storage_driver.name());
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Check audio drivers
                for audio_driver in &self.audio_drivers {
                    if audio_driver.supported_vendor() == device.vendor &&
                       audio_driver.supported_devices().contains(&device.device_id) {
                        device.driver_name = Some(audio_driver.name());
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Check input drivers
                for input_driver in &self.input_drivers {
                    if input_driver.supported_vendor() == device.vendor &&
                       input_driver.supported_devices().contains(&device.device_id) {
                        device.driver_name = Some(input_driver.name());
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Check USB drivers
                for usb_driver in &self.usb_drivers {
                    if usb_driver.supported_vendor() == device.vendor &&
                       usb_driver.supported_devices().contains(&device.device_id) {
                        device.driver_name = Some(usb_driver.name());
                        matched = true;
                        break;
                    }
                }
            }

            if matched {
                crate::println!("[PERIPHERAL] Matched {} driver for {} device 0x{:04X}",
                               device.driver_name.unwrap(), device.vendor, device.device_id);
            }
        }
        Ok(())
    }

    /// Initialize all matched drivers
    pub fn initialize_drivers(&mut self) -> Result<(), &'static str> {
        let mut initialized_count = 0;

        for device in &self.detected_devices {
            if device.driver_name.is_some() {
                crate::println!("[PERIPHERAL] Initializing {} for {} device 0x{:04X}",
                               device.driver_name.unwrap(), device.vendor, device.device_id);

                // Initialize based on device type
                match device.device_type {
                    DeviceType::NetworkInterface => {
                        for net_driver in &mut self.network_drivers {
                            if Some(net_driver.name()) == device.driver_name {
                                match net_driver.initialize(device) {
                                    Ok(_) => {
                                        crate::println!("[PERIPHERAL] {} driver initialized successfully", net_driver.name());
                                        initialized_count += 1;
                                    }
                                    Err(e) => {
                                        crate::println!("[PERIPHERAL] Failed to initialize {} driver: {}", net_driver.name(), e);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    DeviceType::StorageController => {
                        for storage_driver in &mut self.storage_drivers {
                            if Some(storage_driver.name()) == device.driver_name {
                                match storage_driver.initialize(device) {
                                    Ok(_) => {
                                        crate::println!("[PERIPHERAL] {} driver initialized successfully", storage_driver.name());
                                        initialized_count += 1;
                                    }
                                    Err(e) => {
                                        crate::println!("[PERIPHERAL] Failed to initialize {} driver: {}", storage_driver.name(), e);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    DeviceType::AudioDevice => {
                        for audio_driver in &mut self.audio_drivers {
                            if Some(audio_driver.name()) == device.driver_name {
                                match audio_driver.initialize(device) {
                                    Ok(_) => {
                                        crate::println!("[PERIPHERAL] {} driver initialized successfully", audio_driver.name());
                                        initialized_count += 1;
                                    }
                                    Err(e) => {
                                        crate::println!("[PERIPHERAL] Failed to initialize {} driver: {}", audio_driver.name(), e);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    DeviceType::InputDevice => {
                        for input_driver in &mut self.input_drivers {
                            if Some(input_driver.name()) == device.driver_name {
                                match input_driver.initialize(device) {
                                    Ok(_) => {
                                        crate::println!("[PERIPHERAL] {} driver initialized successfully", input_driver.name());
                                        initialized_count += 1;
                                    }
                                    Err(e) => {
                                        crate::println!("[PERIPHERAL] Failed to initialize {} driver: {}", input_driver.name(), e);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    DeviceType::USBController => {
                        for usb_driver in &mut self.usb_drivers {
                            if Some(usb_driver.name()) == device.driver_name {
                                match usb_driver.initialize(device) {
                                    Ok(_) => {
                                        crate::println!("[PERIPHERAL] {} driver initialized successfully", usb_driver.name());
                                        initialized_count += 1;
                                    }
                                    Err(e) => {
                                        crate::println!("[PERIPHERAL] Failed to initialize {} driver: {}", usb_driver.name(), e);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    _ => {
                        // Skip other device types for now
                    }
                }
            }
        }

        crate::println!("[PERIPHERAL] Initialized {} drivers", initialized_count);
        Ok(())
    }

    /// Get list of detected devices
    pub fn get_detected_devices(&self) -> &Vec<PeripheralDevice, 64> {
        &self.detected_devices
    }

    /// Get device count by type
    pub fn count_devices_by_type(&self, device_type: DeviceType) -> usize {
        self.detected_devices.iter()
            .filter(|d| d.device_type == device_type)
            .count()
    }
}

/// Classify device based on PCI class codes
fn classify_device(class_code: u8, subclass: u8) -> DeviceType {
    match class_code {
        0x01 => DeviceType::StorageController,   // Mass storage controller
        0x02 => DeviceType::NetworkInterface,    // Network controller
        0x03 => DeviceType::GraphicsAdapter,     // Display controller
        0x04 => DeviceType::AudioDevice,         // Multimedia controller
        0x06 => DeviceType::BridgeDevice,        // Bridge device
        0x09 => match subclass {
            0x00 => DeviceType::InputDevice,      // Keyboard controller
            0x01 => DeviceType::InputDevice,      // Digitizer pen
            0x02 => DeviceType::InputDevice,      // Mouse controller
            0x03 => DeviceType::InputDevice,      // Scanner controller
            0x04 => DeviceType::InputDevice,      // Gameport controller
            _ => DeviceType::InputDevice,         // Other input device
        },
        0x0C => match subclass {
            0x03 => DeviceType::USBController,    // USB controller
            _ => DeviceType::Unknown,
        },
        _ => DeviceType::Unknown,
    }
}

/// Initialize the peripheral driver subsystem
pub fn init_peripheral_drivers() -> Result<PeripheralDriverRegistry, &'static str> {
    crate::println!("[PERIPHERAL] Initializing peripheral driver subsystem...");

    let mut registry = PeripheralDriverRegistry::new();

    // Register network drivers
    network::register_network_drivers(&mut registry.network_drivers)?;

    // Register storage drivers
    storage::register_storage_drivers(&mut registry.storage_drivers)?;

    // Register audio drivers
    audio::register_audio_drivers(&mut registry.audio_drivers)?;

    // Register input drivers
    input::register_input_drivers(&mut registry.input_drivers)?;

    // Register USB drivers
    usb::register_usb_drivers(&mut registry.usb_drivers)?;

    // Enumerate devices
    registry.enumerate_devices()?;

    // Match drivers to devices
    registry.match_drivers()?;

    // Initialize matched drivers
    registry.initialize_drivers()?;

    crate::println!("[PERIPHERAL] Peripheral driver subsystem initialized");
    crate::println!("[PERIPHERAL] Summary:");
    crate::println!("[PERIPHERAL] - Network devices: {}",
                   registry.count_devices_by_type(DeviceType::NetworkInterface));
    crate::println!("[PERIPHERAL] - Storage devices: {}",
                   registry.count_devices_by_type(DeviceType::StorageController));
    crate::println!("[PERIPHERAL] - Audio devices: {}",
                   registry.count_devices_by_type(DeviceType::AudioDevice));
    crate::println!("[PERIPHERAL] - Input devices: {}",
                   registry.count_devices_by_type(DeviceType::InputDevice));
    crate::println!("[PERIPHERAL] - USB controllers: {}",
                   registry.count_devices_by_type(DeviceType::USBController));

    Ok(registry)
}

/// Global peripheral driver registry
use lazy_static::lazy_static;
use spin::Mutex;

lazy_static! {
    static ref GLOBAL_REGISTRY: Mutex<Option<PeripheralDriverRegistry>> = Mutex::new(None);
}

/// Initialize global peripheral registry
pub fn init() -> Result<(), &'static str> {
    let registry = init_peripheral_drivers()?;
    *GLOBAL_REGISTRY.lock() = Some(registry);
    Ok(())
}

/// Get detected devices from global registry
pub fn get_detected_devices() -> Result<Vec<PeripheralDevice, 64>, &'static str> {
    let registry = GLOBAL_REGISTRY.lock();
    match registry.as_ref() {
        Some(reg) => Ok(reg.get_detected_devices().clone()),
        None => Err("Peripheral registry not initialized"),
    }
}

#[test_case]
fn test_device_vendor_classification() {
    assert_eq!(DeviceVendor::from_pci_id(0x8086), DeviceVendor::Intel);
    assert_eq!(DeviceVendor::from_pci_id(0x1002), DeviceVendor::AMD);
    assert_eq!(DeviceVendor::from_pci_id(0x10DE), DeviceVendor::Nvidia);

    if let DeviceVendor::Unknown(id) = DeviceVendor::from_pci_id(0xFFFF) {
        assert_eq!(id, 0xFFFF);
    } else {
        panic!("Expected Unknown variant");
    }
}

#[test_case]
fn test_device_classification() {
    assert_eq!(classify_device(0x01, 0x00), DeviceType::StorageController);
    assert_eq!(classify_device(0x02, 0x00), DeviceType::NetworkInterface);
    assert_eq!(classify_device(0x03, 0x00), DeviceType::GraphicsAdapter);
    assert_eq!(classify_device(0x04, 0x00), DeviceType::AudioDevice);
    assert_eq!(classify_device(0x09, 0x00), DeviceType::InputDevice);
    assert_eq!(classify_device(0x0C, 0x03), DeviceType::USBController);
}
