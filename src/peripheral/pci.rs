/// PCI device enumeration and management
/// 
/// This module provides real PCI bus scanning and device detection
/// implementing the standard PCI configuration space access methods

use heapless::Vec;

/// PCI configuration space addresses
pub const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
pub const PCI_CONFIG_DATA: u16 = 0xCFC;

/// PCI device information structure
#[derive(Debug, Clone)]
pub struct PCIDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub revision_id: u8,
    pub header_type: u8,
    pub base_addresses: Vec<u32, 6>,
    pub irq_line: u8,
    pub irq_pin: u8,
}

impl PCIDevice {
    pub fn new() -> Self {
        Self {
            bus: 0,
            device: 0,
            function: 0,
            vendor_id: 0,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            prog_if: 0,
            revision_id: 0,
            header_type: 0,
            base_addresses: Vec::new(),
            irq_line: 0,
            irq_pin: 0,
        }
    }
}

/// Read 32-bit value from PCI configuration space
fn pci_config_read_dword(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address = 0x80000000u32
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        // Write address to CONFIG_ADDRESS
        x86_64::instructions::port::Port::new(PCI_CONFIG_ADDRESS).write(address);
        // Read data from CONFIG_DATA
        x86_64::instructions::port::Port::new(PCI_CONFIG_DATA).read()
    }
}

/// Read 16-bit value from PCI configuration space
fn pci_config_read_word(bus: u8, device: u8, function: u8, offset: u8) -> u16 {
    let dword = pci_config_read_dword(bus, device, function, offset & 0xFC);
    if (offset & 0x02) != 0 {
        (dword >> 16) as u16
    } else {
        dword as u16
    }
}

/// Read 8-bit value from PCI configuration space
fn pci_config_read_byte(bus: u8, device: u8, function: u8, offset: u8) -> u8 {
    let dword = pci_config_read_dword(bus, device, function, offset & 0xFC);
    let shift = (offset & 0x03) * 8;
    (dword >> shift) as u8
}

/// Write 32-bit value to PCI configuration space
fn pci_config_write_dword(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let address = 0x80000000u32
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        x86_64::instructions::port::Port::new(PCI_CONFIG_ADDRESS).write(address);
        x86_64::instructions::port::Port::new(PCI_CONFIG_DATA).write(value);
    }
}

/// Check if a PCI device exists at the given location
fn pci_device_exists(bus: u8, device: u8, function: u8) -> bool {
    let vendor_id = pci_config_read_word(bus, device, function, 0x00);
    vendor_id != 0xFFFF
}

/// Read full PCI device information
fn read_pci_device(bus: u8, device: u8, function: u8) -> Result<PCIDevice, &'static str> {
    if !pci_device_exists(bus, device, function) {
        return Err("Device does not exist");
    }

    let mut pci_device = PCIDevice::new();
    pci_device.bus = bus;
    pci_device.device = device;
    pci_device.function = function;

    // Read basic device identification
    pci_device.vendor_id = pci_config_read_word(bus, device, function, 0x00);
    pci_device.device_id = pci_config_read_word(bus, device, function, 0x02);
    
    // Read class information
    let class_info = pci_config_read_dword(bus, device, function, 0x08);
    pci_device.revision_id = (class_info & 0xFF) as u8;
    pci_device.prog_if = ((class_info >> 8) & 0xFF) as u8;
    pci_device.subclass = ((class_info >> 16) & 0xFF) as u8;
    pci_device.class_code = ((class_info >> 24) & 0xFF) as u8;

    // Read header type
    pci_device.header_type = pci_config_read_byte(bus, device, function, 0x0E);

    // Read Base Address Registers (BARs)
    for i in 0..6 {
        let bar_offset = 0x10 + (i * 4);
        let bar_value = pci_config_read_dword(bus, device, function, bar_offset);
        if let Err(_) = pci_device.base_addresses.push(bar_value) {
            break; // Vec is full
        }
    }

    // Read interrupt information
    pci_device.irq_line = pci_config_read_byte(bus, device, function, 0x3C);
    pci_device.irq_pin = pci_config_read_byte(bus, device, function, 0x3D);

    Ok(pci_device)
}

/// Check if device is multifunction
fn is_multifunction_device(bus: u8, device: u8) -> bool {
    if !pci_device_exists(bus, device, 0) {
        return false;
    }
    let header_type = pci_config_read_byte(bus, device, 0, 0x0E);
    (header_type & 0x80) != 0
}

/// Scan entire PCI bus for devices
pub fn scan_pci_bus() -> Result<Vec<PCIDevice, 64>, &'static str> {
    let mut devices = Vec::new();
    
    // Scan all possible bus/device/function combinations
    for bus in 0..=255u8 {
        for device in 0..32u8 {
            // Check function 0 first
            if pci_device_exists(bus, device, 0) {
                if let Ok(pci_dev) = read_pci_device(bus, device, 0) {
                    crate::println!("[PCI] Found device {:04X}:{:04X} at {}:{}.{} (Class: {:02X}.{:02X}.{:02X})",
                                   pci_dev.vendor_id, pci_dev.device_id,
                                   bus, device, 0,
                                   pci_dev.class_code, pci_dev.subclass, pci_dev.prog_if);
                    
                    if let Err(_) = devices.push(pci_dev) {
                        crate::println!("[PCI] Warning: Device list full, stopping scan");
                        return Ok(devices);
                    }
                }

                // Check additional functions if this is a multifunction device
                if is_multifunction_device(bus, device) {
                    for function in 1..8u8 {
                        if pci_device_exists(bus, device, function) {
                            if let Ok(pci_dev) = read_pci_device(bus, device, function) {
                                crate::println!("[PCI] Found device {:04X}:{:04X} at {}:{}.{} (Class: {:02X}.{:02X}.{:02X})",
                                               pci_dev.vendor_id, pci_dev.device_id,
                                               bus, device, function,
                                               pci_dev.class_code, pci_dev.subclass, pci_dev.prog_if);
                                
                                if let Err(_) = devices.push(pci_dev) {
                                    crate::println!("[PCI] Warning: Device list full, stopping scan");
                                    return Ok(devices);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Early termination for buses with no devices (optimization)
        if bus > 0 && devices.is_empty() {
            break;
        }
    }
    
    crate::println!("[PCI] Scan complete, found {} devices", devices.len());
    Ok(devices)
}

/// Enable bus mastering for a PCI device
pub fn enable_bus_mastering(device: &PCIDevice) -> Result<(), &'static str> {
    let command_reg = pci_config_read_word(device.bus, device.device, device.function, 0x04);
    let new_command = command_reg | 0x04; // Set bus master bit
    
    let address = 0x80000000u32
        | ((device.bus as u32) << 16)
        | ((device.device as u32) << 11)
        | ((device.function as u32) << 8)
        | 0x04;

    unsafe {
        x86_64::instructions::port::Port::new(PCI_CONFIG_ADDRESS).write(address);
        x86_64::instructions::port::Port::new(PCI_CONFIG_DATA).write(new_command as u32);
    }
    
    crate::println!("[PCI] Enabled bus mastering for device {:04X}:{:04X}",
                   device.vendor_id, device.device_id);
    Ok(())
}

/// Enable memory space access for a PCI device
pub fn enable_memory_space(device: &PCIDevice) -> Result<(), &'static str> {
    let command_reg = pci_config_read_word(device.bus, device.device, device.function, 0x04);
    let new_command = command_reg | 0x02; // Set memory space enable bit
    
    let address = 0x80000000u32
        | ((device.bus as u32) << 16)
        | ((device.device as u32) << 11)
        | ((device.function as u32) << 8)
        | 0x04;

    unsafe {
        x86_64::instructions::port::Port::new(PCI_CONFIG_ADDRESS).write(address);
        x86_64::instructions::port::Port::new(PCI_CONFIG_DATA).write(new_command as u32);
    }
    
    crate::println!("[PCI] Enabled memory space for device {:04X}:{:04X}",
                   device.vendor_id, device.device_id);
    Ok(())
}

/// Enable I/O space access for a PCI device
pub fn enable_io_space(device: &PCIDevice) -> Result<(), &'static str> {
    let command_reg = pci_config_read_word(device.bus, device.device, device.function, 0x04);
    let new_command = command_reg | 0x01; // Set I/O space enable bit
    
    let address = 0x80000000u32
        | ((device.bus as u32) << 16)
        | ((device.device as u32) << 11)
        | ((device.function as u32) << 8)
        | 0x04;

    unsafe {
        x86_64::instructions::port::Port::new(PCI_CONFIG_ADDRESS).write(address);
        x86_64::instructions::port::Port::new(PCI_CONFIG_DATA).write(new_command as u32);
    }
    
    crate::println!("[PCI] Enabled I/O space for device {:04X}:{:04X}",
                   device.vendor_id, device.device_id);
    Ok(())
}

/// Get BAR size by writing all 1s and reading back
pub fn get_bar_size(device: &PCIDevice, bar_index: usize) -> u32 {
    if bar_index >= device.base_addresses.len() {
        return 0;
    }
    
    let bar_offset = 0x10 + (bar_index * 4) as u8;
    let original_value = device.base_addresses[bar_index];
    
    // Write all 1s to the BAR
    pci_config_write_dword(device.bus, device.device, device.function, bar_offset, 0xFFFFFFFF);
    
    // Read back the value
    let size_mask = pci_config_read_dword(device.bus, device.device, device.function, bar_offset);
    
    // Restore original value
    pci_config_write_dword(device.bus, device.device, device.function, bar_offset, original_value);
    
    // Calculate size
    if size_mask == 0 {
        return 0;
    }
    
    // For memory BARs, mask out the lower bits
    let size = if (original_value & 0x01) == 0 {
        // Memory BAR
        !(size_mask & 0xFFFFFFF0) + 1
    } else {
        // I/O BAR
        !(size_mask & 0xFFFFFFFC) + 1
    };
    
    size
}

#[test_case]
fn test_pci_address_calculation() {
    // Test PCI address calculation for bus 0, device 0, function 0, offset 0
    let expected = 0x80000000u32;
    let address = 0x80000000u32
        | ((0u32) << 16)  // bus
        | ((0u32) << 11)  // device
        | ((0u32) << 8)   // function
        | (0u32 & 0xFC);  // offset
    
    assert_eq!(address, expected);
}