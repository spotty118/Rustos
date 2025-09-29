//! # Network Drivers Interface
//!
//! This module provides a hardware abstraction layer for network devices,
//! supporting Ethernet, wireless, and other network interfaces with unified
//! driver management and device registration.

// Removed unused imports
use alloc::{boxed::Box, string::String, vec::Vec, collections::BTreeMap};
// use alloc::string::ToString; // Unused
use core::fmt;
use spin::RwLock;

use crate::network::{MacAddress, NetworkError, NetworkStats};

/// Network device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Ethernet device
    Ethernet,
    /// Wireless 802.11 device
    Wireless,
    /// Loopback device
    Loopback,
    /// Point-to-Point Protocol device
    Ppp,
    /// Virtual Ethernet device
    Virtual,
    /// Tunnel device
    Tunnel,
    /// Bridge device
    Bridge,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Ethernet => write!(f, "Ethernet"),
            DeviceType::Wireless => write!(f, "Wireless"),
            DeviceType::Loopback => write!(f, "Loopback"),
            DeviceType::Ppp => write!(f, "PPP"),
            DeviceType::Virtual => write!(f, "Virtual"),
            DeviceType::Tunnel => write!(f, "Tunnel"),
            DeviceType::Bridge => write!(f, "Bridge"),
        }
    }
}

/// Network device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Maximum transmission unit
    pub mtu: u16,
    /// Supports hardware checksum offload
    pub hw_checksum: bool,
    /// Supports scatter-gather DMA
    pub scatter_gather: bool,
    /// Supports VLAN tagging
    pub vlan_support: bool,
    /// Supports jumbo frames
    pub jumbo_frames: bool,
    /// Supports multicast filtering
    pub multicast_filter: bool,
    /// Number of transmit queues
    pub tx_queues: u8,
    /// Number of receive queues
    pub rx_queues: u8,
    /// Maximum packet size
    pub max_packet_size: u32,
    /// Link speed in Mbps (0 if variable/unknown)
    pub link_speed: u32,
    /// Supports full duplex
    pub full_duplex: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            mtu: 1500,
            hw_checksum: false,
            scatter_gather: false,
            vlan_support: false,
            jumbo_frames: false,
            multicast_filter: false,
            tx_queues: 1,
            rx_queues: 1,
            max_packet_size: 1518,
            link_speed: 0,
            full_duplex: true,
        }
    }
}

/// Network device state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// Device is down/inactive
    Down,
    /// Device is up and operational
    Up,
    /// Device is in testing mode
    Testing,
    /// Device has encountered an error
    Error,
    /// Device is being reset
    Resetting,
}

impl fmt::Display for DeviceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceState::Down => write!(f, "DOWN"),
            DeviceState::Up => write!(f, "UP"),
            DeviceState::Testing => write!(f, "TESTING"),
            DeviceState::Error => write!(f, "ERROR"),
            DeviceState::Resetting => write!(f, "RESETTING"),
        }
    }
}

/// Network driver interface
pub trait NetworkDriver: Send + Sync + core::fmt::Debug {
    /// Get driver name
    fn name(&self) -> &str;

    /// Get device type
    fn device_type(&self) -> DeviceType;

    /// Get device MAC address
    fn mac_address(&self) -> MacAddress;

    /// Get device capabilities
    fn capabilities(&self) -> DeviceCapabilities;

    /// Get current device state
    fn state(&self) -> DeviceState;

    /// Initialize the device
    fn init(&mut self) -> Result<(), NetworkError>;

    /// Start the device (bring up)
    fn start(&mut self) -> Result<(), NetworkError>;

    /// Stop the device (bring down)
    fn stop(&mut self) -> Result<(), NetworkError>;

    /// Reset the device
    fn reset(&mut self) -> Result<(), NetworkError>;

    /// Send a packet
    fn send_packet(&mut self, data: &[u8]) -> Result<(), NetworkError>;

    /// Receive a packet (non-blocking)
    fn receive_packet(&mut self) -> Option<Vec<u8>>;

    /// Check if link is up
    fn is_link_up(&self) -> bool;

    /// Set promiscuous mode
    fn set_promiscuous(&mut self, enabled: bool) -> Result<(), NetworkError>;

    /// Add multicast address
    fn add_multicast(&mut self, addr: MacAddress) -> Result<(), NetworkError>;

    /// Remove multicast address
    fn remove_multicast(&mut self, addr: MacAddress) -> Result<(), NetworkError>;

    /// Get device statistics
    fn get_stats(&self) -> NetworkStats;

    /// Set MTU
    fn set_mtu(&mut self, mtu: u16) -> Result<(), NetworkError>;

    /// Get current MTU
    fn get_mtu(&self) -> u16;

    /// Handle interrupt (if applicable)
    fn handle_interrupt(&mut self) -> Result<(), NetworkError>;
}

/// Driver configuration
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// Driver name
    pub name: String,
    /// Device MAC address (if configurable)
    pub mac_address: Option<MacAddress>,
    /// MTU setting
    pub mtu: u16,
    /// Enable promiscuous mode
    pub promiscuous: bool,
    /// Multicast addresses
    pub multicast_addresses: Vec<MacAddress>,
    /// Driver-specific parameters
    pub parameters: BTreeMap<String, String>,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            name: "unknown".into(),
            mac_address: None,
            mtu: 1500,
            promiscuous: false,
            multicast_addresses: Vec::new(),
            parameters: BTreeMap::new(),
        }
    }
}

/// Loopback driver implementation
#[derive(Debug)]
pub struct LoopbackDriver {
    name: String,
    mac_address: MacAddress,
    state: DeviceState,
    mtu: u16,
    stats: NetworkStats,
    packet_queue: Vec<Vec<u8>>,
}

impl LoopbackDriver {
    pub fn new(name: String) -> Self {
        Self {
            name,
            mac_address: MacAddress::ZERO,
            state: DeviceState::Down,
            mtu: 65535, // Max MTU for loopback (u16 max)
            stats: NetworkStats::default(),
            packet_queue: Vec::new(),
        }
    }
}

impl NetworkDriver for LoopbackDriver {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Loopback
    }

    fn mac_address(&self) -> MacAddress {
        self.mac_address
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            mtu: self.mtu,
            max_packet_size: self.mtu as u32,
            ..Default::default()
        }
    }

    fn state(&self) -> DeviceState {
        self.state
    }

    fn init(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Down;
        Ok(())
    }

    fn start(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Up;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Down;
        self.packet_queue.clear();
        Ok(())
    }

    fn reset(&mut self) -> Result<(), NetworkError> {
        self.packet_queue.clear();
        self.stats = NetworkStats::default();
        Ok(())
    }

    fn send_packet(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        if self.state != DeviceState::Up {
            return Err(NetworkError::InterfaceDown);
        }

        // Loopback: immediately queue packet for reception
        self.packet_queue.push(data.to_vec());
        self.stats.packets_sent += 1;
        self.stats.bytes_sent += data.len() as u64;

        Ok(())
    }

    fn receive_packet(&mut self) -> Option<Vec<u8>> {
        if self.state != DeviceState::Up {
            return None;
        }

        if let Some(packet) = self.packet_queue.pop() {
            self.stats.packets_received += 1;
            self.stats.bytes_received += packet.len() as u64;
            Some(packet)
        } else {
            None
        }
    }

    fn is_link_up(&self) -> bool {
        self.state == DeviceState::Up
    }

    fn set_promiscuous(&mut self, _enabled: bool) -> Result<(), NetworkError> {
        // Loopback doesn't need promiscuous mode
        Ok(())
    }

    fn add_multicast(&mut self, _addr: MacAddress) -> Result<(), NetworkError> {
        // Loopback accepts all packets
        Ok(())
    }

    fn remove_multicast(&mut self, _addr: MacAddress) -> Result<(), NetworkError> {
        // Loopback accepts all packets
        Ok(())
    }

    fn get_stats(&self) -> NetworkStats {
        self.stats.clone()
    }

    fn set_mtu(&mut self, mtu: u16) -> Result<(), NetworkError> {
        if mtu < 68 {
            return Err(NetworkError::InvalidPacket);
        }
        self.mtu = mtu;
        Ok(())
    }

    fn get_mtu(&self) -> u16 {
        self.mtu
    }

    fn handle_interrupt(&mut self) -> Result<(), NetworkError> {
        // Loopback doesn't have interrupts
        Ok(())
    }
}

/// Dummy Ethernet driver for testing
#[derive(Debug)]
pub struct DummyEthernetDriver {
    name: String,
    mac_address: MacAddress,
    state: DeviceState,
    capabilities: DeviceCapabilities,
    stats: NetworkStats,
    promiscuous: bool,
    multicast_addresses: Vec<MacAddress>,
}

impl DummyEthernetDriver {
    pub fn new(name: String, mac_address: MacAddress) -> Self {
        Self {
            name,
            mac_address,
            state: DeviceState::Down,
            capabilities: DeviceCapabilities::default(),
            stats: NetworkStats::default(),
            promiscuous: false,
            multicast_addresses: Vec::new(),
        }
    }
}

impl NetworkDriver for DummyEthernetDriver {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Ethernet
    }

    fn mac_address(&self) -> MacAddress {
        self.mac_address
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn state(&self) -> DeviceState {
        self.state
    }

    fn init(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Down;
        Ok(())
    }

    fn start(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Up;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Down;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), NetworkError> {
        self.stats = NetworkStats::default();
        self.promiscuous = false;
        self.multicast_addresses.clear();
        Ok(())
    }

    fn send_packet(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        if self.state != DeviceState::Up {
            return Err(NetworkError::InterfaceDown);
        }

        if data.len() > self.capabilities.max_packet_size as usize {
            return Err(NetworkError::BufferTooSmall);
        }

        // Real packet transmission via hardware interface
        self.transmit_packet_hardware(data)?;
        self.stats.packets_sent += 1;
        self.stats.bytes_sent += data.len() as u64;

        Ok(())
    }

    fn receive_packet(&mut self) -> Option<Vec<u8>> {
        // Check hardware receive buffers for packets
        self.check_hardware_rx_queue()
    }

    fn is_link_up(&self) -> bool {
        self.state == DeviceState::Up
    }

    fn set_promiscuous(&mut self, enabled: bool) -> Result<(), NetworkError> {
        self.promiscuous = enabled;
        Ok(())
    }

    fn add_multicast(&mut self, addr: MacAddress) -> Result<(), NetworkError> {
        if !self.multicast_addresses.contains(&addr) {
            self.multicast_addresses.push(addr);
        }
        Ok(())
    }

    fn remove_multicast(&mut self, addr: MacAddress) -> Result<(), NetworkError> {
        self.multicast_addresses.retain(|&a| a != addr);
        Ok(())
    }

    fn get_stats(&self) -> NetworkStats {
        self.stats.clone()
    }

    fn set_mtu(&mut self, mtu: u16) -> Result<(), NetworkError> {
        if mtu < 68 || mtu > 9000 {
            return Err(NetworkError::InvalidPacket);
        }
        self.capabilities.mtu = mtu;
        Ok(())
    }

    fn get_mtu(&self) -> u16 {
        self.capabilities.mtu
    }

    fn handle_interrupt(&mut self) -> Result<(), NetworkError> {
        // Real interrupt handling for network hardware
        self.process_hardware_interrupts()
    }
    
    /// Transmit packet via hardware interface
    fn transmit_packet_hardware(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        // In real implementation, would program network controller TX descriptors
        // and trigger packet transmission via hardware registers
        
        // Validate packet size
        if data.len() > self.capabilities.max_packet_size as usize {
            return Err(NetworkError::BufferTooSmall);
        }
        
        // Would typically:
        // 1. Get available TX descriptor
        // 2. Copy packet data to DMA buffer
        // 3. Program descriptor with buffer address and length
        // 4. Ring doorbell register to start transmission
        
        Ok(())
    }
    
    /// Check hardware RX queue for received packets
    fn check_hardware_rx_queue(&mut self) -> Option<Vec<u8>> {
        // In real implementation, would check RX descriptors for completed packets
        // and return packet data from DMA buffers
        
        // Would typically:
        // 1. Check RX descriptor status
        // 2. If packet available, copy from DMA buffer
        // 3. Update descriptor status for reuse
        // 4. Return packet data
        
        None // No packets available in simulation environment
    }
    
    /// Process hardware interrupts
    fn process_hardware_interrupts(&mut self) -> Result<(), NetworkError> {
        // Read interrupt status register and handle different interrupt types
        // Would typically handle: RX packet received, TX complete, link status change, errors
        
        // In real hardware:
        // let status = read_mmio_register(self.base_addr + INT_STATUS_REG);
        // Handle different interrupt bits accordingly
        
        Ok(())
    }
}

/// Network device descriptor
#[derive(Debug)]
pub struct NetworkDevice {
    /// Device ID
    pub id: u32,
    /// Device driver
    pub driver: Box<dyn NetworkDriver>,
    /// Device configuration
    pub config: DriverConfig,
    /// Device registration timestamp
    pub registered_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

impl NetworkDevice {
    pub fn new(
        id: u32,
        driver: Box<dyn NetworkDriver>,
        config: DriverConfig,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            driver,
            config,
            registered_at: timestamp,
            last_activity: timestamp,
        }
    }

    /// Update last activity
    pub fn update_activity(&mut self, timestamp: u64) {
        self.last_activity = timestamp;
    }

    /// Get device info
    pub fn get_info(&self) -> DeviceInfo {
        DeviceInfo {
            id: self.id,
            name: self.driver.name().into(),
            device_type: self.driver.device_type(),
            mac_address: self.driver.mac_address(),
            state: self.driver.state(),
            capabilities: self.driver.capabilities(),
            stats: self.driver.get_stats(),
            registered_at: self.registered_at,
            last_activity: self.last_activity,
        }
    }
}

/// Device information structure
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub device_type: DeviceType,
    pub mac_address: MacAddress,
    pub state: DeviceState,
    pub capabilities: DeviceCapabilities,
    pub stats: NetworkStats,
    pub registered_at: u64,
    pub last_activity: u64,
}

/// Driver manager for handling network devices
#[derive(Debug)]
pub struct DriverManager {
    /// Registered devices
    devices: BTreeMap<u32, NetworkDevice>,
    /// Next device ID
    next_id: u32,
    /// Manager statistics
    stats: DriverManagerStats,
}

impl DriverManager {
    pub fn new() -> Self {
        Self {
            devices: BTreeMap::new(),
            next_id: 1,
            stats: DriverManagerStats::default(),
        }
    }

    /// Register a network device
    pub fn register_device(
        &mut self,
        driver: Box<dyn NetworkDriver>,
        config: DriverConfig,
        timestamp: u64,
    ) -> Result<u32, NetworkError> {
        let id = self.next_id;
        self.next_id += 1;

        let device = NetworkDevice::new(id, driver, config, timestamp);
        self.devices.insert(id, device);

        self.stats.devices_registered += 1;

        Ok(id)
    }

    /// Unregister a network device
    pub fn unregister_device(&mut self, id: u32) -> Result<(), NetworkError> {
        if self.devices.remove(&id).is_some() {
            self.stats.devices_unregistered += 1;
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get device by ID
    pub fn get_device(&self, id: u32) -> Option<&NetworkDevice> {
        self.devices.get(&id)
    }

    /// Get mutable device by ID
    pub fn get_device_mut(&mut self, id: u32) -> Option<&mut NetworkDevice> {
        self.devices.get_mut(&id)
    }

    /// Get device by name
    pub fn get_device_by_name(&self, name: &str) -> Option<&NetworkDevice> {
        self.devices
            .values()
            .find(|device| device.driver.name() == name)
    }

    /// Get all device IDs
    pub fn get_device_ids(&self) -> Vec<u32> {
        self.devices.keys().copied().collect()
    }

    /// Get all device information
    pub fn get_all_device_info(&self) -> Vec<DeviceInfo> {
        self.devices
            .values()
            .map(|device| device.get_info())
            .collect()
    }

    /// Start all devices
    pub fn start_all_devices(&mut self) -> Result<(), NetworkError> {
        for device in self.devices.values_mut() {
            device.driver.start()?;
        }
        Ok(())
    }

    /// Stop all devices
    pub fn stop_all_devices(&mut self) -> Result<(), NetworkError> {
        for device in self.devices.values_mut() {
            device.driver.stop()?;
        }
        Ok(())
    }

    /// Send packet through device
    pub fn send_packet(&mut self, device_id: u32, data: &[u8]) -> Result<(), NetworkError> {
        let device = self
            .devices
            .get_mut(&device_id)
            .ok_or(NetworkError::InvalidAddress)?;

        device.driver.send_packet(data)?;
        self.stats.packets_sent += 1;

        Ok(())
    }

    /// Receive packet from device
    pub fn receive_packet(&mut self, device_id: u32) -> Option<Vec<u8>> {
        let device = self.devices.get_mut(&device_id)?;

        if let Some(packet) = device.driver.receive_packet() {
            self.stats.packets_received += 1;
            Some(packet)
        } else {
            None
        }
    }

    /// Process all device interrupts
    pub fn process_interrupts(&mut self) -> Result<(), NetworkError> {
        for device in self.devices.values_mut() {
            device.driver.handle_interrupt()?;
        }
        Ok(())
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> &DriverManagerStats {
        &self.stats
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Update device activity
    pub fn update_device_activity(&mut self, device_id: u32, timestamp: u64) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.update_activity(timestamp);
        }
    }
}

impl Default for DriverManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Driver manager statistics
#[derive(Debug, Default, Clone)]
pub struct DriverManagerStats {
    pub devices_registered: u64,
    pub devices_unregistered: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
}

/// Global driver manager instance
static DRIVER_MANAGER: RwLock<Option<DriverManager>> = RwLock::new(None);

/// Initialize global driver manager
pub fn init_driver_manager() {
    *DRIVER_MANAGER.write() = Some(DriverManager::new());
}

/// Get reference to global driver manager
pub fn with_driver_manager<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut DriverManager) -> R,
{
    DRIVER_MANAGER.write().as_mut().map(f)
}

/// Register loopback device
pub fn register_loopback_device() -> Result<u32, NetworkError> {
    let driver = Box::new(LoopbackDriver::new("lo".into()));
    let config = DriverConfig {
        name: "lo".into(),
        ..Default::default()
    };

    with_driver_manager(|manager| {
        manager.register_device(driver, config, 0) // TODO: proper timestamp
    })
    .ok_or(NetworkError::HardwareError)?
}

/// Register dummy ethernet device for testing
pub fn register_dummy_ethernet(name: String, mac: MacAddress) -> Result<u32, NetworkError> {
    let driver = Box::new(DummyEthernetDriver::new(name.clone(), mac));
    let config = DriverConfig {
        name,
        mac_address: Some(mac),
        ..Default::default()
    };

    with_driver_manager(|manager| {
        manager.register_device(driver, config, 0) // TODO: proper timestamp
    })
    .ok_or(NetworkError::HardwareError)?
}

/// High-level device management functions
pub fn get_device_list() -> Vec<DeviceInfo> {
    with_driver_manager(|manager| manager.get_all_device_info()).unwrap_or_default()
}

pub fn start_device(device_id: u32) -> Result<(), NetworkError> {
    with_driver_manager(|manager| {
        if let Some(device) = manager.get_device_mut(device_id) {
            device.driver.start()
        } else {
            Err(NetworkError::InvalidAddress)
        }
    })
    .ok_or(NetworkError::HardwareError)?
}

pub fn stop_device(device_id: u32) -> Result<(), NetworkError> {
    with_driver_manager(|manager| {
        if let Some(device) = manager.get_device_mut(device_id) {
            device.driver.stop()
        } else {
            Err(NetworkError::InvalidAddress)
        }
    })
    .ok_or(NetworkError::HardwareError)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_loopback_driver() {
        let mut driver = LoopbackDriver::new("lo".into());

        assert_eq!(driver.device_type(), DeviceType::Loopback);
        assert_eq!(driver.state(), DeviceState::Down);

        driver.init().unwrap();
        driver.start().unwrap();
        assert_eq!(driver.state(), DeviceState::Up);

        let test_data = b"Hello, loopback!";
        driver.send_packet(test_data).unwrap();

        let received = driver.receive_packet().unwrap();
        assert_eq!(received, test_data);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_dummy_ethernet_driver() {
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let mut driver = DummyEthernetDriver::new("eth0".into(), mac);

        assert_eq!(driver.device_type(), DeviceType::Ethernet);
        assert_eq!(driver.mac_address(), mac);

        driver.init().unwrap();
        driver.start().unwrap();
        assert!(driver.is_link_up());

        let test_data = b"Ethernet packet";
        assert!(driver.send_packet(test_data).is_ok());

        let stats = driver.get_stats();
        assert_eq!(stats.packets_sent, 1);
        assert_eq!(stats.bytes_sent, test_data.len() as u64);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_driver_manager() {
        let mut manager = DriverManager::new();

        let mac = MacAddress::new([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        let driver = Box::new(DummyEthernetDriver::new("test".into(), mac));
        let config = DriverConfig::default();

        let device_id = manager.register_device(driver, config, 1000).unwrap();
        assert_eq!(device_id, 1);
        assert_eq!(manager.device_count(), 1);

        let device_info = manager.get_device(device_id).unwrap().get_info();
        assert_eq!(device_info.name, "test");
        assert_eq!(device_info.device_type, DeviceType::Ethernet);

        manager.unregister_device(device_id).unwrap();
        assert_eq!(manager.device_count(), 0);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_device_capabilities() {
        let caps = DeviceCapabilities::default();
        assert_eq!(caps.mtu, 1500);
        assert!(!caps.hw_checksum);
        assert_eq!(caps.tx_queues, 1);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_device_state_display() {
        assert_eq!(format!("{}", DeviceState::Up), "UP");
        assert_eq!(format!("{}", DeviceState::Down), "DOWN");
        assert_eq!(format!("{}", DeviceState::Error), "ERROR");
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Ethernet), "Ethernet");
        assert_eq!(format!("{}", DeviceType::Wireless), "Wireless");
        assert_eq!(format!("{}", DeviceType::Loopback), "Loopback");
    }
}
