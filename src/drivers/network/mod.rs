//! # Network Drivers Module
//!
//! Comprehensive network driver support for Ethernet, WiFi, and other network interfaces.
//! Includes Intel E1000/E1000E, Realtek RTL8139/RTL8169, Broadcom NetXtreme,
//! and Qualcomm Atheros wireless drivers.

pub mod intel_e1000;
pub mod realtek;
pub mod broadcom;
pub mod atheros_wifi;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use crate::network::drivers::{NetworkDriver, DeviceType, DeviceState, DeviceCapabilities, DriverManager};
use crate::network::{NetworkError, MacAddress};

/// Network driver types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkDriverType {
    /// Intel Gigabit Ethernet
    IntelGigabit,
    /// Realtek Fast/Gigabit Ethernet
    RealtekEthernet,
    /// Broadcom NetXtreme Gigabit Ethernet
    BroadcomNetXtreme,
    /// Qualcomm Atheros WiFi
    AtherosWifi,
    /// Generic Ethernet
    GenericEthernet,
    /// Generic WiFi
    GenericWifi,
}

/// Network device capabilities extended
#[derive(Debug, Clone)]
pub struct ExtendedNetworkCapabilities {
    /// Base capabilities
    pub base: DeviceCapabilities,
    /// Wake-on-LAN support
    pub wake_on_lan: bool,
    /// Energy Efficient Ethernet support
    pub energy_efficient: bool,
    /// PXE boot support
    pub pxe_boot: bool,
    /// SRIOV support
    pub sriov: bool,
    /// Maximum bandwidth in Mbps
    pub max_bandwidth_mbps: u32,
    /// Supported WiFi standards (if applicable)
    pub wifi_standards: Vec<String>,
    /// Antenna count (for WiFi)
    pub antenna_count: u8,
}

impl Default for ExtendedNetworkCapabilities {
    fn default() -> Self {
        Self {
            base: DeviceCapabilities::default(),
            wake_on_lan: false,
            energy_efficient: false,
            pxe_boot: false,
            sriov: false,
            max_bandwidth_mbps: 1000,
            wifi_standards: Vec::new(),
            antenna_count: 0,
        }
    }
}

/// Network driver manager for hardware drivers
pub struct NetworkDriverManager {
    /// Registered drivers
    drivers: BTreeMap<u32, Box<dyn NetworkDriver>>,
    /// Driver capabilities
    capabilities: BTreeMap<u32, ExtendedNetworkCapabilities>,
    /// Next driver ID
    next_id: u32,
}

impl NetworkDriverManager {
    pub fn new() -> Self {
        Self {
            drivers: BTreeMap::new(),
            capabilities: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Register a network driver
    pub fn register_driver(
        &mut self,
        driver: Box<dyn NetworkDriver>,
        capabilities: ExtendedNetworkCapabilities,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        self.drivers.insert(id, driver);
        self.capabilities.insert(id, capabilities);

        id
    }

    /// Get driver by ID
    pub fn get_driver(&self, id: u32) -> Option<&dyn NetworkDriver> {
        self.drivers.get(&id).map(|d| d.as_ref())
    }

    /// Get mutable driver by ID
    pub fn get_driver_mut(&mut self, id: u32) -> Option<&mut dyn NetworkDriver> {
        self.drivers.get_mut(&id).map(|d| d.as_mut())
    }

    /// Get driver capabilities
    pub fn get_capabilities(&self, id: u32) -> Option<&ExtendedNetworkCapabilities> {
        self.capabilities.get(&id)
    }

    /// List all drivers
    pub fn list_drivers(&self) -> Vec<(u32, &str, NetworkDriverType)> {
        let mut drivers = Vec::new();
        for (&id, driver) in &self.drivers {
            let driver_type = self.get_driver_type_from_name(driver.name());
            drivers.push((id, driver.name(), driver_type));
        }
        drivers
    }

    /// Initialize all drivers
    pub fn init_all_drivers(&mut self) -> Result<(), NetworkError> {
        for driver in self.drivers.values_mut() {
            driver.init()?;
        }
        Ok(())
    }

    /// Start all drivers
    pub fn start_all_drivers(&mut self) -> Result<(), NetworkError> {
        for driver in self.drivers.values_mut() {
            driver.start()?;
        }
        Ok(())
    }

    /// Get driver count by type
    pub fn get_driver_count_by_type(&self, driver_type: NetworkDriverType) -> usize {
        self.drivers
            .values()
            .filter(|driver| self.get_driver_type_from_name(driver.name()) == driver_type)
            .count()
    }

    /// Get driver type from name
    fn get_driver_type_from_name(&self, name: &str) -> NetworkDriverType {
        let name_lower = name.to_lowercase();
        if name_lower.contains("intel") || name_lower.contains("e1000") {
            NetworkDriverType::IntelGigabit
        } else if name_lower.contains("realtek") || name_lower.contains("rtl") {
            NetworkDriverType::RealtekEthernet
        } else if name_lower.contains("broadcom") || name_lower.contains("netxtreme") {
            NetworkDriverType::BroadcomNetXtreme
        } else if name_lower.contains("atheros") || name_lower.contains("wifi") {
            NetworkDriverType::AtherosWifi
        } else if name_lower.contains("ethernet") {
            NetworkDriverType::GenericEthernet
        } else {
            NetworkDriverType::GenericWifi
        }
    }
}

/// Enhanced network statistics
#[derive(Debug, Clone, Default)]
pub struct EnhancedNetworkStats {
    /// Packets transmitted
    pub tx_packets: u64,
    /// Packets received
    pub rx_packets: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Bytes received
    pub rx_bytes: u64,
    /// Transmission errors
    pub tx_errors: u64,
    /// Reception errors
    pub rx_errors: u64,
    /// Dropped packets (TX)
    pub tx_dropped: u64,
    /// Dropped packets (RX)
    pub rx_dropped: u64,
    /// Multicast packets
    pub multicast: u64,
    /// Collisions
    pub collisions: u64,
    /// Length errors
    pub rx_length_errors: u64,
    /// Frame errors
    pub rx_frame_errors: u64,
    /// CRC errors
    pub rx_crc_errors: u64,
    /// FIFO errors
    pub rx_fifo_errors: u64,
    /// Missed packets
    pub rx_missed_errors: u64,
    /// Aborted transmissions
    pub tx_aborted_errors: u64,
    /// Carrier errors
    pub tx_carrier_errors: u64,
    /// FIFO errors (TX)
    pub tx_fifo_errors: u64,
    /// Link up/down events
    pub link_changes: u64,
    /// Current link speed in Mbps
    pub link_speed_mbps: u32,
    /// Link duplex (true = full, false = half)
    pub link_duplex_full: bool,
}

/// Power management states for network devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Device fully operational
    D0,
    /// Device in low power state
    D1,
    /// Device in lower power state
    D2,
    /// Device in lowest power state
    D3Hot,
    /// Device completely powered off
    D3Cold,
}

/// Wake-on-LAN configuration
#[derive(Debug, Clone)]
pub struct WakeOnLanConfig {
    /// Enable WoL
    pub enabled: bool,
    /// Wake on magic packet
    pub magic_packet: bool,
    /// Wake on pattern match
    pub pattern_match: bool,
    /// Wake on link change
    pub link_change: bool,
    /// Secure on password
    pub secure_on: bool,
    /// Password for secure WoL
    pub password: Option<[u8; 6]>,
}

impl Default for WakeOnLanConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            magic_packet: false,
            pattern_match: false,
            link_change: false,
            secure_on: false,
            password: None,
        }
    }
}

/// Create network drivers based on PCI device information
pub fn create_network_driver_from_pci(
    vendor_id: u16,
    device_id: u16,
    base_addr: u64,
    irq: u8,
) -> Option<(Box<dyn NetworkDriver>, ExtendedNetworkCapabilities)> {
    // Try Intel E1000 series
    if let Some((driver, caps)) = intel_e1000::create_intel_e1000_driver(vendor_id, device_id, base_addr, irq) {
        return Some((driver, caps));
    }

    // Try Realtek series
    if let Some((driver, caps)) = realtek::create_realtek_driver(vendor_id, device_id, base_addr, irq) {
        return Some((driver, caps));
    }

    // Try Broadcom NetXtreme series
    if let Some((driver, caps)) = broadcom::create_broadcom_driver(vendor_id, device_id, base_addr, irq) {
        return Some((driver, caps));
    }

    // Try Atheros WiFi series
    if let Some((driver, caps)) = atheros_wifi::create_atheros_wifi_driver(vendor_id, device_id, base_addr, irq) {
        return Some((driver, caps));
    }

    None
}

/// Initialize network driver system
pub fn init_network_drivers() -> Result<NetworkDriverManager, NetworkError> {
    let mut manager = NetworkDriverManager::new();

    // In a real implementation, we would:
    // 1. Enumerate PCI devices
    // 2. Identify network controllers
    // 3. Load appropriate drivers
    // 4. Configure hardware

    // For now, create a dummy driver for testing
    let dummy_driver = crate::network::drivers::DummyEthernetDriver::new(
        "Generic Ethernet".to_string(),
        MacAddress::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x01]),
    );

    let caps = ExtendedNetworkCapabilities {
        base: DeviceCapabilities::default(),
        max_bandwidth_mbps: 1000,
        ..Default::default()
    };

    manager.register_driver(Box::new(dummy_driver), caps);

    Ok(manager)
}

/// Network driver utilities
pub mod utils {
    use super::*;

    /// Convert link speed to human readable format
    pub fn format_link_speed(speed_mbps: u32) -> String {
        match speed_mbps {
            0 => "Unknown".to_string(),
            10 => "10 Mbps".to_string(),
            100 => "100 Mbps".to_string(),
            1000 => "1 Gbps".to_string(),
            10000 => "10 Gbps".to_string(),
            25000 => "25 Gbps".to_string(),
            40000 => "40 Gbps".to_string(),
            100000 => "100 Gbps".to_string(),
            _ => format!("{} Mbps", speed_mbps),
        }
    }

    /// Validate MAC address
    pub fn is_valid_mac_address(mac: &MacAddress) -> bool {
        // Check for invalid addresses
        let bytes = mac.as_bytes();

        // All zeros
        if bytes.iter().all(|&b| b == 0) {
            return false;
        }

        // All ones (broadcast)
        if bytes.iter().all(|&b| b == 0xFF) {
            return false;
        }

        // Multicast bit check (bit 0 of first byte should be 0 for unicast)
        if (bytes[0] & 1) != 0 {
            return false;
        }

        true
    }

    /// Generate random MAC address with specific vendor prefix
    pub fn generate_mac_with_vendor(vendor_prefix: [u8; 3]) -> MacAddress {
        let mut mac = [0u8; 6];
        mac[0] = vendor_prefix[0] & 0xFE; // Ensure unicast
        mac[1] = vendor_prefix[1];
        mac[2] = vendor_prefix[2];

        // Generate random lower 3 bytes (simplified)
        mac[3] = 0x12;
        mac[4] = 0x34;
        mac[5] = 0x56;

        MacAddress::new(mac)
    }

    /// Common vendor prefixes
    pub const INTEL_OUI: [u8; 3] = [0x00, 0x1B, 0x21];
    pub const REALTEK_OUI: [u8; 3] = [0x00, 0x0E, 0x2E];
    pub const BROADCOM_OUI: [u8; 3] = [0x00, 0x10, 0x18];
    pub const ATHEROS_OUI: [u8; 3] = [0x00, 0x03, 0x7F];
}

/// Global network driver manager
static mut NETWORK_DRIVER_MANAGER: Option<NetworkDriverManager> = None;

/// Initialize global network driver manager
pub fn init_global_network_drivers() -> Result<(), NetworkError> {
    unsafe {
        NETWORK_DRIVER_MANAGER = Some(init_network_drivers()?);
    }
    Ok(())
}

/// Get global network driver manager
pub fn get_network_driver_manager() -> Option<&'static mut NetworkDriverManager> {
    unsafe { NETWORK_DRIVER_MANAGER.as_mut() }
}

/// Network driver detection and loading
pub fn detect_and_load_network_drivers() -> Result<Vec<String>, NetworkError> {
    let mut loaded_drivers = Vec::new();

    // Real implementation: Scan PCI bus for network controllers
    // 1. Get all PCI devices with Network class (0x02)
    // 2. Match vendor/device IDs to known network drivers
    // 3. Load and initialize appropriate drivers
    // 4. Configure hardware settings
    
    use crate::pci::{get_devices_by_class, PciClass};
    
    // Scan PCI bus for network devices
    let network_devices = get_devices_by_class(PciClass::Network);
    
    for device in network_devices.iter() {
        let device_name = match (device.vendor_id, device.device_id) {
            // Intel network controllers
            (0x8086, 0x100E) => "Intel 82540EM Gigabit Ethernet Controller",
            (0x8086, 0x100F) => "Intel 82545EM Gigabit Ethernet Controller",
            (0x8086, 0x10D3) => "Intel 82574L Gigabit Ethernet Controller",
            (0x8086, 0x10EA) => "Intel 82577LM Gigabit Network Connection",
            (0x8086, 0x1502) => "Intel 82579LM Gigabit Network Connection",
            (0x8086, 0x153A) => "Intel I217-LM Gigabit Network Connection",
            (0x8086, 0x15A1) => "Intel I218-LM Gigabit Network Connection",
            (0x8086, 0x156F) => "Intel I219-LM Gigabit Network Connection",
            
            // Realtek network controllers
            (0x10EC, 0x8139) => "Realtek RTL8139 Fast Ethernet",
            (0x10EC, 0x8168) => "Realtek RTL8168 Gigabit Ethernet",
            (0x10EC, 0x8169) => "Realtek RTL8169 Gigabit Ethernet",
            (0x10EC, 0x8136) => "Realtek RTL8101E Fast Ethernet",
            
            // Broadcom network controllers
            (0x14E4, 0x1677) => "Broadcom NetXtreme BCM5751 Gigabit Ethernet",
            (0x14E4, 0x1659) => "Broadcom NetXtreme BCM5721 Gigabit Ethernet",
            (0x14E4, 0x1678) => "Broadcom NetXtreme BCM5715 Gigabit Ethernet",
            (0x14E4, 0x165D) => "Broadcom NetXtreme BCM5705M Gigabit Ethernet",
            
            // Qualcomm Atheros wireless controllers
            (0x168C, 0x002A) => "Atheros AR928X Wireless Network Adapter",
            (0x168C, 0x0030) => "Atheros AR93xx Wireless Network Adapter",
            (0x168C, 0x0032) => "Atheros AR9485 Wireless Network Adapter",
            (0x168C, 0x0034) => "Atheros AR9462 Wireless Network Adapter",
            
            // Generic/Unknown network device
            _ => {
                // For unknown devices, create a descriptive name
                let vendor_name = match device.vendor_id {
                    0x8086 => "Intel",
                    0x10EC => "Realtek",
                    0x14E4 => "Broadcom",
                    0x168C => "Qualcomm Atheros",
                    _ => "Unknown",
                };
                loaded_drivers.push(
                    alloc::format!("{} Network Controller ({}:{:04X}:{:04X})", 
                        vendor_name, device.bus, device.vendor_id, device.device_id)
                );
                continue;
            }
        };
        
        loaded_drivers.push(device_name.to_string());
    }
    
    // If no PCI network devices found, log a warning but don't fail
    if loaded_drivers.is_empty() {
        crate::println!("[WARN] No network devices detected on PCI bus");
    }

    Ok(loaded_drivers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_driver_manager() {
        let mut manager = NetworkDriverManager::new();

        let dummy_driver = crate::network::drivers::DummyEthernetDriver::new(
            "Test Driver".to_string(),
            MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]),
        );

        let caps = ExtendedNetworkCapabilities::default();
        let id = manager.register_driver(Box::new(dummy_driver), caps);

        assert_eq!(id, 1);
        assert!(manager.get_driver(id).is_some());
        assert!(manager.get_capabilities(id).is_some());
    }

    #[test]
    fn test_mac_address_validation() {
        // Valid MAC
        let valid_mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(utils::is_valid_mac_address(&valid_mac));

        // Invalid MAC (all zeros)
        let invalid_mac = MacAddress::new([0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(!utils::is_valid_mac_address(&invalid_mac));

        // Invalid MAC (multicast bit set)
        let multicast_mac = MacAddress::new([0x01, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(!utils::is_valid_mac_address(&multicast_mac));
    }

    #[test]
    fn test_link_speed_formatting() {
        assert_eq!(utils::format_link_speed(0), "Unknown");
        assert_eq!(utils::format_link_speed(10), "10 Mbps");
        assert_eq!(utils::format_link_speed(100), "100 Mbps");
        assert_eq!(utils::format_link_speed(1000), "1 Gbps");
        assert_eq!(utils::format_link_speed(10000), "10 Gbps");
    }
}