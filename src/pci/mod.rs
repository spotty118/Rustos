//! PCI Bus Enumeration and Management System
//!
//! This module provides comprehensive PCI device discovery, configuration,
//! and management functionality for RustOS.

pub mod config;
pub mod database;
pub mod detection;

use alloc::vec::Vec;
use core::fmt;
use x86_64::instructions::port::{PortReadOnly, PortWriteOnly};

/// PCI Configuration Address Port (0xCF8)
const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
/// PCI Configuration Data Port (0xCFC)
const PCI_CONFIG_DATA: u16 = 0xCFC;

/// Maximum number of PCI buses
pub const MAX_BUS: u8 = 255;
/// Maximum number of devices per bus
pub const MAX_DEVICE: u8 = 32;
/// Maximum number of functions per device
pub const MAX_FUNCTION: u8 = 8;

/// PCI Device Class Codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PciClass {
    Unclassified = 0x00,
    MassStorage = 0x01,
    Network = 0x02,
    Display = 0x03,
    Multimedia = 0x04,
    Memory = 0x05,
    Bridge = 0x06,
    Communication = 0x07,
    SystemPeripheral = 0x08,
    InputDevice = 0x09,
    DockingStation = 0x0A,
    Processor = 0x0B,
    SerialBus = 0x0C,
    Wireless = 0x0D,
    IntelligentIO = 0x0E,
    Satellite = 0x0F,
    Encryption = 0x10,
    SignalProcessing = 0x11,
    ProcessingAccelerator = 0x12,
    NonEssentialInstrumentation = 0x13,
    Reserved = 0xFF,
}

impl From<u8> for PciClass {
    fn from(value: u8) -> Self {
        match value {
            0x00 => PciClass::Unclassified,
            0x01 => PciClass::MassStorage,
            0x02 => PciClass::Network,
            0x03 => PciClass::Display,
            0x04 => PciClass::Multimedia,
            0x05 => PciClass::Memory,
            0x06 => PciClass::Bridge,
            0x07 => PciClass::Communication,
            0x08 => PciClass::SystemPeripheral,
            0x09 => PciClass::InputDevice,
            0x0A => PciClass::DockingStation,
            0x0B => PciClass::Processor,
            0x0C => PciClass::SerialBus,
            0x0D => PciClass::Wireless,
            0x0E => PciClass::IntelligentIO,
            0x0F => PciClass::Satellite,
            0x10 => PciClass::Encryption,
            0x11 => PciClass::SignalProcessing,
            0x12 => PciClass::ProcessingAccelerator,
            0x13 => PciClass::NonEssentialInstrumentation,
            _ => PciClass::Reserved,
        }
    }
}

impl fmt::Display for PciClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            PciClass::Unclassified => "Unclassified",
            PciClass::MassStorage => "Mass Storage",
            PciClass::Network => "Network",
            PciClass::Display => "Display",
            PciClass::Multimedia => "Multimedia",
            PciClass::Memory => "Memory",
            PciClass::Bridge => "Bridge",
            PciClass::Communication => "Communication",
            PciClass::SystemPeripheral => "System Peripheral",
            PciClass::InputDevice => "Input Device",
            PciClass::DockingStation => "Docking Station",
            PciClass::Processor => "Processor",
            PciClass::SerialBus => "Serial Bus",
            PciClass::Wireless => "Wireless",
            PciClass::IntelligentIO => "Intelligent I/O",
            PciClass::Satellite => "Satellite",
            PciClass::Encryption => "Encryption",
            PciClass::SignalProcessing => "Signal Processing",
            PciClass::ProcessingAccelerator => "Processing Accelerator",
            PciClass::NonEssentialInstrumentation => "Non-Essential Instrumentation",
            PciClass::Reserved => "Reserved",
        };
        write!(f, "{}", name)
    }
}

/// PCI Device Capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciCapabilities {
    pub msi: bool,
    pub msi_x: bool,
    pub power_management: bool,
    pub pci_express: bool,
    pub hot_plug: bool,
    pub vendor_specific: bool,
}

impl Default for PciCapabilities {
    fn default() -> Self {
        Self {
            msi: false,
            msi_x: false,
            power_management: false,
            pci_express: false,
            hot_plug: false,
            vendor_specific: false,
        }
    }
}

/// PCI Device Information
#[derive(Debug, Clone)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: PciClass,
    pub subclass: u8,
    pub prog_if: u8,
    pub revision_id: u8,
    pub header_type: u8,
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    pub capabilities: PciCapabilities,
    pub bars: [u32; 6],
}

impl PciDevice {
    /// Create a new PCI device with default values
    pub fn new(bus: u8, device: u8, function: u8) -> Self {
        Self {
            bus,
            device,
            function,
            vendor_id: 0,
            device_id: 0,
            class_code: PciClass::Unclassified,
            subclass: 0,
            prog_if: 0,
            revision_id: 0,
            header_type: 0,
            subsystem_vendor_id: 0,
            subsystem_id: 0,
            capabilities: PciCapabilities::default(),
            bars: [0; 6],
        }
    }

    /// Get the device location as a formatted string
    pub fn location(&self) -> alloc::string::String {
        alloc::format!("{:02x}:{:02x}.{}", self.bus, self.device, self.function)
    }

    /// Check if this is a multifunction device
    pub fn is_multifunction(&self) -> bool {
        (self.header_type & 0x80) != 0
    }

    /// Get the base header type (without multifunction bit)
    pub fn base_header_type(&self) -> u8 {
        self.header_type & 0x7F
    }
}

impl fmt::Display for PciDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PCI Device {}:{:04x}:{:04x} - {} (Class: {:02x}:{:02x})",
            self.location(),
            self.vendor_id,
            self.device_id,
            self.class_code,
            self.class_code as u8,
            self.subclass
        )
    }
}

/// PCI Bus Scanner
pub struct PciBusScanner {
    devices: Vec<PciDevice>,
    initialized: bool,
}

impl PciBusScanner {
    /// Create a new PCI bus scanner
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            initialized: false,
        }
    }

    /// Initialize and scan the PCI bus
    pub fn initialize(&mut self) -> Result<(), &'static str> {
        if self.initialized {
            return Ok(());
        }

        self.scan_all_buses()?;
        self.initialized = true;
        Ok(())
    }

    /// Scan all PCI buses for devices
    fn scan_all_buses(&mut self) -> Result<(), &'static str> {
        self.devices.clear();

        // Start with bus 0
        self.scan_bus(0)?;

        // Check if there are multiple host controllers
        let host_bridge = self.read_config_word(0, 0, 0, 0x0E);
        if (host_bridge & 0x80) == 0 {
            // Single host controller, scan all buses
            for bus in 1..=MAX_BUS {
                self.scan_bus(bus)?;
            }
        } else {
            // Multiple host controllers, scan based on function 0 of device 0
            for function in 0..MAX_FUNCTION {
                let vendor_id = self.read_config_word(0, 0, function, 0x00);
                if vendor_id != 0xFFFF {
                    self.scan_bus(function)?;
                }
            }
        }

        Ok(())
    }

    /// Scan a specific PCI bus
    fn scan_bus(&mut self, bus: u8) -> Result<(), &'static str> {
        for device in 0..MAX_DEVICE {
            self.scan_device(bus, device)?;
        }
        Ok(())
    }

    /// Scan a specific device on a bus
    fn scan_device(&mut self, bus: u8, device: u8) -> Result<(), &'static str> {
        let vendor_id = self.read_config_word(bus, device, 0, 0x00);
        if vendor_id == 0xFFFF {
            return Ok(()); // Device doesn't exist
        }

        // Scan function 0
        self.scan_function(bus, device, 0)?;

        // Check if this is a multifunction device
        let header_type = self.read_config_byte(bus, device, 0, 0x0E);
        if (header_type & 0x80) != 0 {
            // Multifunction device, scan all functions
            for function in 1..MAX_FUNCTION {
                let func_vendor_id = self.read_config_word(bus, device, function, 0x00);
                if func_vendor_id != 0xFFFF {
                    self.scan_function(bus, device, function)?;
                }
            }
        }

        Ok(())
    }

    /// Scan a specific function and add it to the device list
    fn scan_function(&mut self, bus: u8, device: u8, function: u8) -> Result<(), &'static str> {
        let vendor_id = self.read_config_word(bus, device, function, 0x00);
        if vendor_id == 0xFFFF {
            return Ok(());
        }

        let mut pci_device = PciDevice::new(bus, device, function);

        // Read basic device information
        pci_device.vendor_id = vendor_id;
        pci_device.device_id = self.read_config_word(bus, device, function, 0x02);
        pci_device.revision_id = self.read_config_byte(bus, device, function, 0x08);

        let class_info = self.read_config_dword(bus, device, function, 0x08);
        pci_device.prog_if = ((class_info >> 8) & 0xFF) as u8;
        pci_device.subclass = ((class_info >> 16) & 0xFF) as u8;
        pci_device.class_code = PciClass::from(((class_info >> 24) & 0xFF) as u8);

        pci_device.header_type = self.read_config_byte(bus, device, function, 0x0E);

        // Read subsystem information for header type 0
        if pci_device.base_header_type() == 0 {
            pci_device.subsystem_vendor_id = self.read_config_word(bus, device, function, 0x2C);
            pci_device.subsystem_id = self.read_config_word(bus, device, function, 0x2E);
        }

        // Read BARs
        for i in 0..6 {
            let bar_offset = 0x10 + (i * 4) as u8;
            pci_device.bars[i] = self.read_config_dword(bus, device, function, bar_offset);
        }

        // Scan capabilities
        pci_device.capabilities = self.scan_capabilities(bus, device, function)?;

        self.devices.push(pci_device);
        Ok(())
    }

    /// Scan device capabilities
    fn scan_capabilities(&self, bus: u8, device: u8, function: u8) -> Result<PciCapabilities, &'static str> {
        let mut capabilities = PciCapabilities::default();

        // Check if device has capabilities
        let status = self.read_config_word(bus, device, function, 0x06);
        if (status & 0x10) == 0 {
            return Ok(capabilities); // No capabilities
        }

        let mut cap_ptr = self.read_config_byte(bus, device, function, 0x34) & 0xFC;

        while cap_ptr != 0 && cap_ptr != 0xFF {
            let cap_id = self.read_config_byte(bus, device, function, cap_ptr);

            match cap_id {
                0x01 => capabilities.power_management = true,
                0x05 => capabilities.msi = true,
                0x10 => capabilities.pci_express = true,
                0x11 => capabilities.msi_x = true,
                0x0C => capabilities.hot_plug = true,
                0x09 => capabilities.vendor_specific = true,
                _ => {}
            }

            cap_ptr = self.read_config_byte(bus, device, function, cap_ptr + 1) & 0xFC;
        }

        Ok(capabilities)
    }

    /// Read a 32-bit configuration space register
    pub fn read_config_dword(&self, bus: u8, device: u8, function: u8, offset: u8) -> u32 {
        let address = self.make_config_address(bus, device, function, offset);

        unsafe {
            let mut addr_port: PortWriteOnly<u32> = PortWriteOnly::new(PCI_CONFIG_ADDRESS);
            let mut data_port: PortReadOnly<u32> = PortReadOnly::new(PCI_CONFIG_DATA);

            addr_port.write(address);
            data_port.read()
        }
    }

    /// Read a 16-bit configuration space register
    pub fn read_config_word(&self, bus: u8, device: u8, function: u8, offset: u8) -> u16 {
        let dword = self.read_config_dword(bus, device, function, offset & 0xFC);
        let shift = (offset & 0x02) * 8;
        ((dword >> shift) & 0xFFFF) as u16
    }

    /// Read an 8-bit configuration space register
    pub fn read_config_byte(&self, bus: u8, device: u8, function: u8, offset: u8) -> u8 {
        let dword = self.read_config_dword(bus, device, function, offset & 0xFC);
        let shift = (offset & 0x03) * 8;
        ((dword >> shift) & 0xFF) as u8
    }

    /// Write a 32-bit configuration space register
    pub fn write_config_dword(&self, bus: u8, device: u8, function: u8, offset: u8, value: u32) {
        let address = self.make_config_address(bus, device, function, offset);

        unsafe {
            let mut addr_port: PortWriteOnly<u32> = PortWriteOnly::new(PCI_CONFIG_ADDRESS);
            let mut data_port: PortWriteOnly<u32> = PortWriteOnly::new(PCI_CONFIG_DATA);

            addr_port.write(address);
            data_port.write(value);
        }
    }

    /// Write a 16-bit configuration space register
    pub fn write_config_word(&self, bus: u8, device: u8, function: u8, offset: u8, value: u16) {
        let dword = self.read_config_dword(bus, device, function, offset & 0xFC);
        let shift = (offset & 0x02) * 8;
        let mask = !(0xFFFF << shift);
        let new_value = (dword & mask) | ((value as u32) << shift);
        self.write_config_dword(bus, device, function, offset & 0xFC, new_value);
    }

    /// Write an 8-bit configuration space register
    pub fn write_config_byte(&self, bus: u8, device: u8, function: u8, offset: u8, value: u8) {
        let dword = self.read_config_dword(bus, device, function, offset & 0xFC);
        let shift = (offset & 0x03) * 8;
        let mask = !(0xFF << shift);
        let new_value = (dword & mask) | ((value as u32) << shift);
        self.write_config_dword(bus, device, function, offset & 0xFC, new_value);
    }

    /// Create a PCI configuration address
    fn make_config_address(&self, bus: u8, device: u8, function: u8, offset: u8) -> u32 {
        let enable_bit = 1u32 << 31;
        let bus_bits = (bus as u32) << 16;
        let device_bits = (device as u32 & 0x1F) << 11;
        let function_bits = (function as u32 & 0x07) << 8;
        let offset_bits = (offset as u32) & 0xFC;

        enable_bit | bus_bits | device_bits | function_bits | offset_bits
    }

    /// Get all discovered devices
    pub fn get_devices(&self) -> &Vec<PciDevice> {
        &self.devices
    }

    /// Get devices by class
    pub fn get_devices_by_class(&self, class: PciClass) -> Vec<&PciDevice> {
        self.devices.iter().filter(|d| d.class_code == class).collect()
    }

    /// Get devices by vendor ID
    pub fn get_devices_by_vendor(&self, vendor_id: u16) -> Vec<&PciDevice> {
        self.devices.iter().filter(|d| d.vendor_id == vendor_id).collect()
    }

    /// Find a specific device by vendor and device ID
    pub fn find_device(&self, vendor_id: u16, device_id: u16) -> Option<&PciDevice> {
        self.devices.iter().find(|d| d.vendor_id == vendor_id && d.device_id == device_id)
    }

    /// Get total number of discovered devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Check if the scanner has been initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Global PCI bus scanner instance
use spin::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref PCI_SCANNER: Mutex<PciBusScanner> = Mutex::new(PciBusScanner::new());
}

/// Initialize the global PCI scanner
pub fn init_pci() -> Result<(), &'static str> {
    PCI_SCANNER.lock().initialize()
}

/// Get the global PCI scanner
pub fn get_pci_scanner() -> &'static Mutex<PciBusScanner> {
    &PCI_SCANNER
}

/// Get all PCI devices
pub fn get_all_devices() -> Vec<PciDevice> {
    PCI_SCANNER.lock().get_devices().clone()
}

/// Get devices by class
pub fn get_devices_by_class(class: PciClass) -> Vec<PciDevice> {
    PCI_SCANNER.lock().get_devices_by_class(class).into_iter().cloned().collect()
}

/// Print all discovered PCI devices
pub fn print_devices() {
    let scanner = PCI_SCANNER.lock();
    let devices = scanner.get_devices();

    if devices.is_empty() {
        crate::println!("No PCI devices found.");
        return;
    }

    crate::println!("Discovered {} PCI devices:", devices.len());
    crate::println!("Bus:Dev.Fn Vendor:Device Class           Description");
    crate::println!("---------- ----------- --------------- -----------");

    for device in devices {
        let vendor_name = database::get_vendor_name(device.vendor_id);
        let device_name = database::get_device_name(device.vendor_id, device.device_id);

        crate::println!(
            "{:02x}:{:02x}.{} {:04x}:{:04x} {:15} {} {}",
            device.bus,
            device.device,
            device.function,
            device.vendor_id,
            device.device_id,
            device.class_code,
            vendor_name,
            device_name
        );
    }
}