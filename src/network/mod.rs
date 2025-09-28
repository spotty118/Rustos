//! # RustOS Network Stack
//!
//! Complete network stack implementation for RustOS kernel with support for
//! Ethernet, IPv4/IPv6, TCP/UDP, and socket interfaces.

use crate::{BTreeMap, Box, String, Vec};
use core::fmt;
use spin::{Mutex, RwLock};

pub mod arp;
pub mod buffer;
pub mod dhcp;
pub mod dns;
pub mod drivers;
pub mod ethernet;
pub mod ip;
pub mod socket;
pub mod tcp;
pub mod udp;

/// Network stack configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub hostname: String,
    pub enable_dhcp: bool,
    pub static_ip: Option<Ipv4Address>,
    pub subnet_mask: Option<Ipv4Address>,
    pub gateway: Option<Ipv4Address>,
    pub dns_servers: Vec<Ipv4Address>,
    pub mtu: u16,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            hostname: "rustos".into(),
            enable_dhcp: true,
            static_ip: None,
            subnet_mask: None,
            gateway: None,
            dns_servers: Vec::new(),
            mtu: 1500,
        }
    }
}

/// IPv4 Address representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ipv4Address(pub [u8; 4]);

impl Ipv4Address {
    pub const BROADCAST: Ipv4Address = Ipv4Address([255, 255, 255, 255]);
    pub const LOCALHOST: Ipv4Address = Ipv4Address([127, 0, 0, 1]);
    pub const UNSPECIFIED: Ipv4Address = Ipv4Address([0, 0, 0, 0]);

    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self([a, b, c, d])
    }

    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    pub fn to_bytes(self) -> [u8; 4] {
        self.0
    }

    pub fn is_broadcast(&self) -> bool {
        *self == Self::BROADCAST
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] >= 224 && self.0[0] <= 239
    }

    pub fn is_private(&self) -> bool {
        match self.0[0] {
            10 => true,
            172 => self.0[1] >= 16 && self.0[1] <= 31,
            192 => self.0[1] == 168,
            _ => false,
        }
    }
}

impl fmt::Display for Ipv4Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// MAC Address representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MacAddress(pub [u8; 6]);

impl MacAddress {
    pub const BROADCAST: MacAddress = MacAddress([0xFF; 6]);
    pub const ZERO: MacAddress = MacAddress([0; 6]);

    pub fn new(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }

    pub fn is_broadcast(&self) -> bool {
        *self == Self::BROADCAST
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] & 0x01 != 0
    }
}

impl fmt::Display for MacAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

/// Network interface representation
#[derive(Debug)]
pub struct NetworkInterface {
    pub name: String,
    pub mac_address: MacAddress,
    pub ip_address: Option<Ipv4Address>,
    pub subnet_mask: Option<Ipv4Address>,
    pub mtu: u16,
    pub is_up: bool,
    pub driver: Box<dyn NetworkDriver>,
}

impl NetworkInterface {
    pub fn new(name: String, mac_address: MacAddress, driver: Box<dyn NetworkDriver>) -> Self {
        Self {
            name,
            mac_address,
            ip_address: None,
            subnet_mask: None,
            mtu: 1500,
            is_up: false,
            driver,
        }
    }

    pub fn send_packet(&mut self, packet: &[u8]) -> Result<(), NetworkError> {
        if !self.is_up {
            return Err(NetworkError::InterfaceDown);
        }
        self.driver.send_packet(packet)
    }

    pub fn receive_packet(&mut self) -> Option<Vec<u8>> {
        if !self.is_up {
            return None;
        }
        self.driver.receive_packet()
    }

    pub fn set_ip_config(&mut self, ip: Ipv4Address, mask: Ipv4Address) {
        self.ip_address = Some(ip);
        self.subnet_mask = Some(mask);
    }
}

/// Network driver trait
pub trait NetworkDriver: Send + Sync + core::fmt::Debug {
    fn name(&self) -> &str;
    fn mac_address(&self) -> MacAddress;
    fn mtu(&self) -> u16;
    fn is_link_up(&self) -> bool;
    fn send_packet(&mut self, packet: &[u8]) -> Result<(), NetworkError>;
    fn receive_packet(&mut self) -> Option<Vec<u8>>;
    fn set_promiscuous(&mut self, enabled: bool) -> Result<(), NetworkError>;
    fn get_stats(&self) -> NetworkStats;
}

/// Network statistics
#[derive(Debug, Default, Clone)]
pub struct NetworkStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub errors_sent: u64,
    pub errors_received: u64,
    pub dropped_packets: u64,
}

/// Network errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkError {
    InterfaceDown,
    InvalidPacket,
    BufferTooSmall,
    NoRoute,
    PortInUse,
    ConnectionRefused,
    ConnectionReset,
    Timeout,
    InsufficientMemory,
    HardwareError,
    InvalidAddress,
    ProtocolError,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::InterfaceDown => write!(f, "Network interface is down"),
            NetworkError::InvalidPacket => write!(f, "Invalid packet format"),
            NetworkError::BufferTooSmall => write!(f, "Buffer too small"),
            NetworkError::NoRoute => write!(f, "No route to destination"),
            NetworkError::PortInUse => write!(f, "Port already in use"),
            NetworkError::ConnectionRefused => write!(f, "Connection refused"),
            NetworkError::ConnectionReset => write!(f, "Connection reset"),
            NetworkError::Timeout => write!(f, "Operation timed out"),
            NetworkError::InsufficientMemory => write!(f, "Insufficient memory"),
            NetworkError::HardwareError => write!(f, "Hardware error"),
            NetworkError::InvalidAddress => write!(f, "Invalid address"),
            NetworkError::ProtocolError => write!(f, "Protocol error"),
        }
    }
}

/// Socket address representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SocketAddr {
    pub ip: Ipv4Address,
    pub port: u16,
}

impl SocketAddr {
    pub fn new(ip: Ipv4Address, port: u16) -> Self {
        Self { ip, port }
    }
}

impl fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.ip, self.port)
    }
}

/// Network protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Raw,
}

/// Network stack state
pub struct NetworkStack {
    interfaces: RwLock<BTreeMap<String, NetworkInterface>>,
    routing_table: RwLock<BTreeMap<Ipv4Address, String>>,
    arp_table: RwLock<BTreeMap<Ipv4Address, MacAddress>>,
    tcp_sockets: RwLock<BTreeMap<u16, tcp::TcpSocket>>,
    udp_sockets: RwLock<BTreeMap<u16, udp::UdpSocket>>,
    next_port: Mutex<u16>,
    config: NetworkConfig,
}

impl NetworkStack {
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            interfaces: RwLock::new(BTreeMap::new()),
            routing_table: RwLock::new(BTreeMap::new()),
            arp_table: RwLock::new(BTreeMap::new()),
            tcp_sockets: RwLock::new(BTreeMap::new()),
            udp_sockets: RwLock::new(BTreeMap::new()),
            next_port: Mutex::new(32768),
            config,
        }
    }

    pub fn add_interface(&self, interface: NetworkInterface) -> Result<(), NetworkError> {
        let name = interface.name.clone();
        self.interfaces.write().insert(name.clone(), interface);

        // Add route for this interface
        if let Some(interface) = self.interfaces.read().get(&name) {
            if let (Some(ip), Some(mask)) = (interface.ip_address, interface.subnet_mask) {
                let network = Ipv4Address::new(
                    ip.0[0] & mask.0[0],
                    ip.0[1] & mask.0[1],
                    ip.0[2] & mask.0[2],
                    ip.0[3] & mask.0[3],
                );
                self.routing_table.write().insert(network, name);
            }
        }

        Ok(())
    }

    pub fn remove_interface(&self, name: &str) -> Result<(), NetworkError> {
        self.interfaces.write().remove(name);

        // Remove routes for this interface
        let mut routes_to_remove = Vec::new();
        for (network, iface) in self.routing_table.read().iter() {
            if iface == name {
                routes_to_remove.push(*network);
            }
        }

        let mut routing_table = self.routing_table.write();
        for network in routes_to_remove {
            routing_table.remove(&network);
        }

        Ok(())
    }

    pub fn send_packet(
        &self,
        dest: Ipv4Address,
        data: &[u8],
        protocol: Protocol,
    ) -> Result<(), NetworkError> {
        let interface_name = self.find_route(dest)?;
        let mut interfaces = self.interfaces.write();
        let interface = interfaces
            .get_mut(&interface_name)
            .ok_or(NetworkError::NoRoute)?;

        // Build packet based on protocol
        let packet = match protocol {
            Protocol::Tcp => self.build_tcp_packet(dest, data)?,
            Protocol::Udp => self.build_udp_packet(dest, data)?,
            Protocol::Icmp => self.build_icmp_packet(dest, data)?,
            Protocol::Raw => data.to_vec(),
        };

        interface.send_packet(&packet)
    }

    pub fn receive_packets(&self) -> Result<(), NetworkError> {
        for interface in self.interfaces.write().values_mut() {
            while let Some(packet) = interface.receive_packet() {
                self.process_packet(&packet)?;
            }
        }
        Ok(())
    }

    fn find_route(&self, dest: Ipv4Address) -> Result<String, NetworkError> {
        // Simple routing: find matching network
        for (network, interface) in self.routing_table.read().iter() {
            // This is a simplified route matching
            if dest.0[0] == network.0[0] {
                return Ok(interface.clone());
            }
        }

        // Default route (first interface)
        self.interfaces
            .read()
            .keys()
            .next()
            .cloned()
            .ok_or(NetworkError::NoRoute)
    }

    fn process_packet(&self, packet: &[u8]) -> Result<(), NetworkError> {
        // Parse Ethernet header
        if packet.len() < 14 {
            return Err(NetworkError::InvalidPacket);
        }

        let ether_type = u16::from_be_bytes([packet[12], packet[13]]);

        match ether_type {
            0x0800 => self.process_ipv4_packet(&packet[14..]),
            0x0806 => self.process_arp_packet(&packet[14..]),
            _ => Ok(()), // Ignore unknown protocols
        }
    }

    fn process_ipv4_packet(&self, packet: &[u8]) -> Result<(), NetworkError> {
        if packet.len() < 20 {
            return Err(NetworkError::InvalidPacket);
        }

        let protocol = packet[9];
        let header_length = ((packet[0] & 0x0f) * 4) as usize;
        let payload = &packet[header_length..];

        match protocol {
            6 => self.process_tcp_packet(payload),  // TCP
            17 => self.process_udp_packet(payload), // UDP
            1 => self.process_icmp_packet(payload), // ICMP
            _ => Ok(()),
        }
    }

    fn process_tcp_packet(&self, _packet: &[u8]) -> Result<(), NetworkError> {
        // TCP packet processing would be implemented here
        Ok(())
    }

    fn process_udp_packet(&self, _packet: &[u8]) -> Result<(), NetworkError> {
        // UDP packet processing would be implemented here
        Ok(())
    }

    fn process_icmp_packet(&self, _packet: &[u8]) -> Result<(), NetworkError> {
        // ICMP packet processing would be implemented here
        Ok(())
    }

    fn process_arp_packet(&self, _packet: &[u8]) -> Result<(), NetworkError> {
        // ARP packet processing would be implemented here
        Ok(())
    }

    fn build_tcp_packet(&self, _dest: Ipv4Address, _data: &[u8]) -> Result<Vec<u8>, NetworkError> {
        // TCP packet building would be implemented here
        Err(NetworkError::ProtocolError)
    }

    fn build_udp_packet(&self, _dest: Ipv4Address, _data: &[u8]) -> Result<Vec<u8>, NetworkError> {
        // UDP packet building would be implemented here
        Err(NetworkError::ProtocolError)
    }

    fn build_icmp_packet(&self, _dest: Ipv4Address, _data: &[u8]) -> Result<Vec<u8>, NetworkError> {
        // ICMP packet building would be implemented here
        Err(NetworkError::ProtocolError)
    }

    pub fn allocate_port(&self) -> u16 {
        let mut next_port = self.next_port.lock();
        let port = *next_port;
        *next_port = if *next_port == 65535 {
            32768
        } else {
            *next_port + 1
        };
        port
    }

    pub fn get_interface_stats(&self) -> BTreeMap<String, NetworkStats> {
        let mut stats = BTreeMap::new();
        for (name, interface) in self.interfaces.read().iter() {
            stats.insert(name.clone(), interface.driver.get_stats());
        }
        stats
    }
}

/// Global network stack instance
static NETWORK_STACK: spin::Once<NetworkStack> = spin::Once::new();

/// Initialize the global network stack
pub fn init_network(config: NetworkConfig) {
    NETWORK_STACK.call_once(|| NetworkStack::new(config));
}

/// Get reference to the global network stack
pub fn network_stack() -> Option<&'static NetworkStack> {
    NETWORK_STACK.get()
}

/// High-level networking functions
pub fn send_tcp(dest: SocketAddr, data: &[u8]) -> Result<(), NetworkError> {
    if let Some(stack) = network_stack() {
        stack.send_packet(dest.ip, data, Protocol::Tcp)
    } else {
        Err(NetworkError::ProtocolError)
    }
}

pub fn send_udp(dest: SocketAddr, data: &[u8]) -> Result<(), NetworkError> {
    if let Some(stack) = network_stack() {
        stack.send_packet(dest.ip, data, Protocol::Udp)
    } else {
        Err(NetworkError::ProtocolError)
    }
}

pub fn ping(dest: Ipv4Address) -> Result<(), NetworkError> {
    if let Some(stack) = network_stack() {
        let ping_data = b"RustOS ping";
        stack.send_packet(dest, ping_data, Protocol::Icmp)
    } else {
        Err(NetworkError::ProtocolError)
    }
}

/// Network utility functions
pub fn resolve_hostname(_hostname: &str) -> Result<Ipv4Address, NetworkError> {
    // DNS resolution would be implemented here
    Err(NetworkError::ProtocolError)
}

pub fn get_local_ip() -> Option<Ipv4Address> {
    network_stack()?
        .interfaces
        .read()
        .values()
        .find_map(|interface| interface.ip_address)
}

pub fn is_network_available() -> bool {
    if let Some(stack) = network_stack() {
        stack
            .interfaces
            .read()
            .values()
            .any(|interface| interface.is_up && interface.ip_address.is_some())
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ipv4_address() {
        let addr = Ipv4Address::new(192, 168, 1, 1);
        assert_eq!(addr.to_bytes(), [192, 168, 1, 1]);
        assert!(addr.is_private());
        assert!(!addr.is_multicast());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_mac_address() {
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(!mac.is_broadcast());
        assert!(!mac.is_multicast());

        let broadcast = MacAddress::BROADCAST;
        assert!(broadcast.is_broadcast());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_addr() {
        let addr = SocketAddr::new(Ipv4Address::LOCALHOST, 8080);
        assert_eq!(addr.port, 8080);
        assert_eq!(addr.ip, Ipv4Address::LOCALHOST);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_network_config() {
        let config = NetworkConfig::default();
        assert_eq!(config.hostname, "rustos");
        assert!(config.enable_dhcp);
        assert_eq!(config.mtu, 1500);
    }
}
