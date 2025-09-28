//! Network device abstraction layer
//!
//! This module provides an abstraction layer for network devices,
//! supporting various types of network interfaces including Ethernet,
//! wireless, loopback, and virtual devices.

use super::{NetworkAddress, NetworkResult, NetworkError, PacketBuffer, NetworkInterface, InterfaceFlags, InterfaceStats};
use alloc::{vec::Vec, string::String, boxed::Box};
use spin::{RwLock, Mutex};
use lazy_static::lazy_static;

/// Network device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Ethernet device
    Ethernet,
    /// Wireless device
    Wireless,
    /// Loopback device
    Loopback,
    /// Virtual Ethernet (veth)
    VirtualEthernet,
    /// Tunnel device
    Tunnel,
    /// Bridge device
    Bridge,
}

/// Network device capabilities
#[derive(Debug, Clone, Copy)]
pub struct DeviceCapabilities {
    /// Maximum transmission unit
    pub max_mtu: u16,
    /// Supports hardware checksumming
    pub hw_checksum: bool,
    /// Supports scatter-gather I/O
    pub scatter_gather: bool,
    /// Supports TCP segmentation offload
    pub tso: bool,
    /// Supports receive side scaling
    pub rss: bool,
    /// Supports VLAN tagging
    pub vlan: bool,
    /// Supports jumbo frames
    pub jumbo_frames: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_mtu: 1500,
            hw_checksum: false,
            scatter_gather: false,
            tso: false,
            rss: false,
            vlan: false,
            jumbo_frames: false,
        }
    }
}

/// Network device trait
pub trait NetworkDevice: Send + Sync {
    /// Get device name
    fn name(&self) -> &str;
    
    /// Get device type
    fn device_type(&self) -> DeviceType;
    
    /// Get MAC address
    fn mac_address(&self) -> NetworkAddress;
    
    /// Get device capabilities
    fn capabilities(&self) -> DeviceCapabilities;
    
    /// Get current MTU
    fn mtu(&self) -> u16;
    
    /// Set MTU
    fn set_mtu(&mut self, mtu: u16) -> NetworkResult<()>;
    
    /// Check if device is up
    fn is_up(&self) -> bool;
    
    /// Bring device up
    fn up(&mut self) -> NetworkResult<()>;
    
    /// Bring device down
    fn down(&mut self) -> NetworkResult<()>;
    
    /// Send packet
    fn send(&mut self, packet: PacketBuffer) -> NetworkResult<()>;
    
    /// Receive packet (non-blocking)
    fn recv(&mut self) -> NetworkResult<Option<PacketBuffer>>;
    
    /// Get device statistics
    fn stats(&self) -> InterfaceStats;
    
    /// Reset device statistics
    fn reset_stats(&mut self);
    
    /// Get device-specific information
    fn device_info(&self) -> DeviceInfo;
}

/// Device-specific information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Driver name
    pub driver: String,
    /// Driver version
    pub version: String,
    /// Firmware version
    pub firmware: Option<String>,
    /// Bus information
    pub bus_info: Option<String>,
    /// Supported link modes
    pub link_modes: Vec<LinkMode>,
    /// Current link speed (Mbps)
    pub link_speed: Option<u32>,
    /// Link duplex mode
    pub duplex: Option<DuplexMode>,
}

/// Network link modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMode {
    /// 10 Mbps half duplex
    Mode10BaseT_Half,
    /// 10 Mbps full duplex
    Mode10BaseT_Full,
    /// 100 Mbps half duplex
    Mode100BaseT_Half,
    /// 100 Mbps full duplex
    Mode100BaseT_Full,
    /// 1000 Mbps full duplex
    Mode1000BaseT_Full,
    /// 10 Gbps full duplex
    Mode10GBaseT_Full,
}

/// Duplex modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplexMode {
    Half,
    Full,
}

/// Loopback network device
pub struct LoopbackDevice {
    name: String,
    mac_address: NetworkAddress,
    mtu: u16,
    up: bool,
    stats: InterfaceStats,
    recv_queue: Vec<PacketBuffer>,
}

impl LoopbackDevice {
    pub fn new() -> Self {
        Self {
            name: "lo".to_string(),
            mac_address: NetworkAddress::Mac([0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            mtu: 65536,
            up: false,
            stats: InterfaceStats::default(),
            recv_queue: Vec::new(),
        }
    }
}

impl NetworkDevice for LoopbackDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Loopback
    }

    fn mac_address(&self) -> NetworkAddress {
        self.mac_address
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            max_mtu: 65536,
            hw_checksum: true, // Loopback doesn't need real checksums
            scatter_gather: true,
            tso: true,
            rss: false,
            vlan: false,
            jumbo_frames: true,
        }
    }

    fn mtu(&self) -> u16 {
        self.mtu
    }

    fn set_mtu(&mut self, mtu: u16) -> NetworkResult<()> {
        if mtu > 65536 {
            return Err(NetworkError::InvalidArgument);
        }
        self.mtu = mtu;
        Ok(())
    }

    fn is_up(&self) -> bool {
        self.up
    }

    fn up(&mut self) -> NetworkResult<()> {
        self.up = true;
        println!("Loopback device {} is up", self.name);
        Ok(())
    }

    fn down(&mut self) -> NetworkResult<()> {
        self.up = false;
        self.recv_queue.clear();
        println!("Loopback device {} is down", self.name);
        Ok(())
    }

    fn send(&mut self, packet: PacketBuffer) -> NetworkResult<()> {
        if !self.up {
            return Err(NetworkError::NetworkUnreachable);
        }

        // Loopback: send packet back to receive queue
        self.recv_queue.push(packet.clone());
        
        self.stats.tx_packets += 1;
        self.stats.tx_bytes += packet.length as u64;
        
        Ok(())
    }

    fn recv(&mut self) -> NetworkResult<Option<PacketBuffer>> {
        if !self.up {
            return Ok(None);
        }

        if let Some(packet) = self.recv_queue.pop() {
            self.stats.rx_packets += 1;
            self.stats.rx_bytes += packet.length as u64;
            Ok(Some(packet))
        } else {
            Ok(None)
        }
    }

    fn stats(&self) -> InterfaceStats {
        self.stats.clone()
    }

    fn reset_stats(&mut self) {
        self.stats = InterfaceStats::default();
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            driver: "loopback".to_string(),
            version: "1.0.0".to_string(),
            firmware: None,
            bus_info: None,
            link_modes: vec![],
            link_speed: None,
            duplex: None,
        }
    }
}

/// Virtual Ethernet device
pub struct VirtualEthernetDevice {
    name: String,
    mac_address: NetworkAddress,
    mtu: u16,
    up: bool,
    stats: InterfaceStats,
    peer: Option<String>,
    recv_queue: Vec<PacketBuffer>,
}

impl VirtualEthernetDevice {
    pub fn new(name: String, mac_address: NetworkAddress) -> Self {
        Self {
            name,
            mac_address,
            mtu: 1500,
            up: false,
            stats: InterfaceStats::default(),
            peer: None,
            recv_queue: Vec::new(),
        }
    }

    pub fn set_peer(&mut self, peer_name: String) {
        self.peer = Some(peer_name);
    }
}

impl NetworkDevice for VirtualEthernetDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::VirtualEthernet
    }

    fn mac_address(&self) -> NetworkAddress {
        self.mac_address
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            max_mtu: 65536,
            hw_checksum: false,
            scatter_gather: true,
            tso: false,
            rss: false,
            vlan: true,
            jumbo_frames: true,
        }
    }

    fn mtu(&self) -> u16 {
        self.mtu
    }

    fn set_mtu(&mut self, mtu: u16) -> NetworkResult<()> {
        if mtu > 65536 {
            return Err(NetworkError::InvalidArgument);
        }
        self.mtu = mtu;
        Ok(())
    }

    fn is_up(&self) -> bool {
        self.up
    }

    fn up(&mut self) -> NetworkResult<()> {
        self.up = true;
        println!("Virtual Ethernet device {} is up", self.name);
        Ok(())
    }

    fn down(&mut self) -> NetworkResult<()> {
        self.up = false;
        self.recv_queue.clear();
        println!("Virtual Ethernet device {} is down", self.name);
        Ok(())
    }

    fn send(&mut self, packet: PacketBuffer) -> NetworkResult<()> {
        if !self.up {
            return Err(NetworkError::NetworkUnreachable);
        }

        // TODO: Send to peer device
        println!("Sending packet through veth {}", self.name);
        
        self.stats.tx_packets += 1;
        self.stats.tx_bytes += packet.length as u64;
        
        Ok(())
    }

    fn recv(&mut self) -> NetworkResult<Option<PacketBuffer>> {
        if !self.up {
            return Ok(None);
        }

        if let Some(packet) = self.recv_queue.pop() {
            self.stats.rx_packets += 1;
            self.stats.rx_bytes += packet.length as u64;
            Ok(Some(packet))
        } else {
            Ok(None)
        }
    }

    fn stats(&self) -> InterfaceStats {
        self.stats.clone()
    }

    fn reset_stats(&mut self) {
        self.stats = InterfaceStats::default();
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            driver: "veth".to_string(),
            version: "1.0.0".to_string(),
            firmware: None,
            bus_info: None,
            link_modes: vec![
                LinkMode::Mode10BaseT_Full,
                LinkMode::Mode100BaseT_Full,
                LinkMode::Mode1000BaseT_Full,
                LinkMode::Mode10GBaseT_Full,
            ],
            link_speed: Some(10000), // 10 Gbps
            duplex: Some(DuplexMode::Full),
        }
    }
}

/// Network device manager
pub struct DeviceManager {
    devices: RwLock<Vec<Box<dyn NetworkDevice>>>,
    device_lookup: RwLock<alloc::collections::BTreeMap<String, usize>>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(Vec::new()),
            device_lookup: RwLock::new(alloc::collections::BTreeMap::new()),
        }
    }

    /// Register a network device
    pub fn register_device(&self, device: Box<dyn NetworkDevice>) -> NetworkResult<()> {
        let device_name = device.name().to_string();
        
        let mut devices = self.devices.write();
        let mut lookup = self.device_lookup.write();
        
        if lookup.contains_key(&device_name) {
            return Err(NetworkError::AddressInUse);
        }

        let index = devices.len();
        devices.push(device);
        lookup.insert(device_name.clone(), index);
        
        println!("Registered network device: {}", device_name);
        Ok(())
    }

    /// Unregister a network device
    pub fn unregister_device(&self, name: &str) -> NetworkResult<()> {
        let mut devices = self.devices.write();
        let mut lookup = self.device_lookup.write();
        
        if let Some(&index) = lookup.get(name) {
            devices.remove(index);
            lookup.remove(name);
            
            // Update indices in lookup table
            for (_, idx) in lookup.iter_mut() {
                if *idx > index {
                    *idx -= 1;
                }
            }
            
            println!("Unregistered network device: {}", name);
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get device by name
    pub fn get_device(&self, name: &str) -> Option<usize> {
        let lookup = self.device_lookup.read();
        lookup.get(name).copied()
    }

    /// List all devices
    pub fn list_devices(&self) -> Vec<String> {
        let lookup = self.device_lookup.read();
        lookup.keys().cloned().collect()
    }

    /// Get device information
    pub fn get_device_info(&self, name: &str) -> Option<DeviceInfo> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(name) {
            drop(lookup);
            let devices = self.devices.read();
            if let Some(device) = devices.get(index) {
                Some(device.device_info())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Send packet through device
    pub fn send_packet(&self, device_name: &str, packet: PacketBuffer) -> NetworkResult<()> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(device_name) {
            drop(lookup);
            let mut devices = self.devices.write();
            if let Some(device) = devices.get_mut(index) {
                device.send(packet)
            } else {
                Err(NetworkError::InvalidAddress)
            }
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Receive packet from device
    pub fn recv_packet(&self, device_name: &str) -> NetworkResult<Option<PacketBuffer>> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(device_name) {
            drop(lookup);
            let mut devices = self.devices.write();
            if let Some(device) = devices.get_mut(index) {
                device.recv()
            } else {
                Err(NetworkError::InvalidAddress)
            }
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Set device state
    pub fn set_device_state(&self, device_name: &str, up: bool) -> NetworkResult<()> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(device_name) {
            drop(lookup);
            let mut devices = self.devices.write();
            if let Some(device) = devices.get_mut(index) {
                if up {
                    device.up()
                } else {
                    device.down()
                }
            } else {
                Err(NetworkError::InvalidAddress)
            }
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get device statistics
    pub fn get_device_stats(&self, device_name: &str) -> Option<InterfaceStats> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(device_name) {
            drop(lookup);
            let devices = self.devices.read();
            if let Some(device) = devices.get(index) {
                Some(device.stats())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Create network interface from device
    pub fn create_interface(&self, device_name: &str) -> Option<NetworkInterface> {
        let lookup = self.device_lookup.read();
        if let Some(&index) = lookup.get(device_name) {
            drop(lookup);
            let devices = self.devices.read();
            if let Some(device) = devices.get(index) {
                let mut flags = InterfaceFlags::default();
                flags.up = device.is_up();
                flags.loopback = device.device_type() == DeviceType::Loopback;
                
                Some(NetworkInterface {
                    name: device.name().to_string(),
                    mac_address: device.mac_address(),
                    ip_addresses: Vec::new(),
                    mtu: device.mtu(),
                    flags,
                    stats: device.stats(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Poll all devices for incoming packets
    pub fn poll_devices(&self) -> Vec<(String, PacketBuffer)> {
        let mut packets = Vec::new();
        let mut devices = self.devices.write();
        let lookup = self.device_lookup.read();
        
        for (name, &index) in lookup.iter() {
            if let Some(device) = devices.get_mut(index) {
                if let Ok(Some(packet)) = device.recv() {
                    packets.push((name.clone(), packet));
                }
            }
        }
        
        packets
    }
}

lazy_static! {
    static ref DEVICE_MANAGER: DeviceManager = DeviceManager::new();
}

/// Initialize network device subsystem
pub fn init() -> NetworkResult<()> {
    // Register loopback device
    let loopback = Box::new(LoopbackDevice::new());
    DEVICE_MANAGER.register_device(loopback)?;
    
    // Bring loopback up
    DEVICE_MANAGER.set_device_state("lo", true)?;
    
    println!("✓ Network device subsystem initialized");
    Ok(())
}

/// Get the global device manager
pub fn device_manager() -> &'static DeviceManager {
    &DEVICE_MANAGER
}

/// Create virtual ethernet pair
pub fn create_veth_pair(name1: &str, name2: &str) -> NetworkResult<()> {
    // Generate MAC addresses
    let mac1 = NetworkAddress::Mac([0x02, 0x00, 0x00, 0x00, 0x00, 0x01]);
    let mac2 = NetworkAddress::Mac([0x02, 0x00, 0x00, 0x00, 0x00, 0x02]);
    
    // Create devices
    let mut veth1 = VirtualEthernetDevice::new(name1.to_string(), mac1);
    let mut veth2 = VirtualEthernetDevice::new(name2.to_string(), mac2);
    
    // Set peers
    veth1.set_peer(name2.to_string());
    veth2.set_peer(name1.to_string());
    
    // Register devices
    DEVICE_MANAGER.register_device(Box::new(veth1))?;
    DEVICE_MANAGER.register_device(Box::new(veth2))?;
    
    println!("Created virtual ethernet pair: {} <-> {}", name1, name2);
    Ok(())
}

/// Network device statistics
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub name: String,
    pub device_type: DeviceType,
    pub mac_address: NetworkAddress,
    pub mtu: u16,
    pub is_up: bool,
    pub stats: InterfaceStats,
}

/// Get statistics for all devices
pub fn get_all_device_stats() -> Vec<DeviceStats> {
    let device_names = DEVICE_MANAGER.list_devices();
    let mut stats = Vec::new();
    
    for name in device_names {
        if let Some(interface) = DEVICE_MANAGER.create_interface(&name) {
            stats.push(DeviceStats {
                name: interface.name,
                device_type: DeviceType::Ethernet, // Default, would need device type in interface
                mac_address: interface.mac_address,
                mtu: interface.mtu,
                is_up: interface.flags.up,
                stats: interface.stats,
            });
        }
    }
    
    stats
}
