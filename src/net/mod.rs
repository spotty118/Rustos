//! Network stack implementation for RustOS
//!
//! This module provides a complete TCP/IP networking stack including:
//! - Ethernet frame handling
//! - IP packet processing (IPv4/IPv6)
//! - TCP connection management
//! - UDP datagram handling
//! - Socket interface
//! - Network device abstraction

pub mod ethernet;
pub mod ip;
pub mod tcp;
pub mod udp;
pub mod socket;
pub mod device;

use alloc::{vec::Vec, vec, collections::BTreeMap, string::String, format};
use spin::{RwLock, Mutex};
use lazy_static::lazy_static;
use core::fmt;
use crate::println;

/// Network address types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkAddress {
    /// IPv4 address
    IPv4([u8; 4]),
    /// IPv6 address
    IPv6([u8; 16]),
    /// MAC address
    Mac([u8; 6]),
}

impl NetworkAddress {
    /// Create IPv4 address from octets
    pub fn ipv4(a: u8, b: u8, c: u8, d: u8) -> Self {
        NetworkAddress::IPv4([a, b, c, d])
    }

    /// Create MAC address from bytes
    pub fn mac(bytes: [u8; 6]) -> Self {
        NetworkAddress::Mac(bytes)
    }

    /// Check if address is broadcast
    pub fn is_broadcast(&self) -> bool {
        match self {
            NetworkAddress::IPv4([255, 255, 255, 255]) => true,
            NetworkAddress::Mac([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]) => true,
            _ => false,
        }
    }

    /// Check if address is multicast
    pub fn is_multicast(&self) -> bool {
        match self {
            NetworkAddress::IPv4([a, _, _, _]) => (*a & 0xf0) == 0xe0,
            NetworkAddress::IPv6([a, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _]) => (*a & 0xff) == 0xff,
            NetworkAddress::Mac([a, _, _, _, _, _]) => (*a & 0x01) != 0,
        }
    }
}

impl fmt::Display for NetworkAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkAddress::IPv4([a, b, c, d]) => write!(f, "{}.{}.{}.{}", a, b, c, d),
            NetworkAddress::IPv6(bytes) => {
                write!(f, "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
                    bytes[0], bytes[1], bytes[2], bytes[3],
                    bytes[4], bytes[5], bytes[6], bytes[7],
                    bytes[8], bytes[9], bytes[10], bytes[11],
                    bytes[12], bytes[13], bytes[14], bytes[15])
            }
            NetworkAddress::Mac([a, b, c, d, e, f]) => {
                write!(f, "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", a, b, c, d, e, f)
            }
        }
    }
}

/// Network protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Protocol {
    /// Internet Control Message Protocol
    ICMP = 1,
    /// Transmission Control Protocol
    TCP = 6,
    /// User Datagram Protocol
    UDP = 17,
    /// IPv6 in IPv4
    IPv6inIPv4 = 41,
    /// Generic Routing Encapsulation
    GRE = 47,
    /// IPv6 Internet Control Message Protocol
    ICMPv6 = 58,
}

/// Network packet buffer
#[derive(Debug, Clone)]
pub struct PacketBuffer {
    /// Raw packet data
    pub data: Vec<u8>,
    /// Current position in buffer
    pub position: usize,
    /// Packet length
    pub length: usize,
}

impl PacketBuffer {
    /// Create a new packet buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity],
            position: 0,
            length: 0,
        }
    }

    /// Create packet buffer from existing data
    pub fn from_data(data: Vec<u8>) -> Self {
        let length = data.len();
        Self {
            data,
            position: 0,
            length,
        }
    }

    /// Get remaining bytes in buffer
    pub fn remaining(&self) -> usize {
        self.length.saturating_sub(self.position)
    }

    /// Read bytes from buffer
    pub fn read(&mut self, count: usize) -> Option<&[u8]> {
        if self.position + count <= self.length {
            let start = self.position;
            self.position += count;
            Some(&self.data[start..self.position])
        } else {
            None
        }
    }

    /// Write bytes to buffer
    pub fn write(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        if self.position + data.len() > self.data.len() {
            return Err(NetworkError::BufferOverflow);
        }

        self.data[self.position..self.position + data.len()].copy_from_slice(data);
        self.position += data.len();
        self.length = self.length.max(self.position);
        Ok(())
    }

    /// Reset buffer position
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get slice of current data
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.length]
    }
}

/// Network error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkError {
    /// Invalid packet format
    InvalidPacket,
    /// Buffer overflow
    BufferOverflow,
    /// Network unreachable
    NetworkUnreachable,
    /// Host unreachable
    HostUnreachable,
    /// Port unreachable
    PortUnreachable,
    /// Connection refused
    ConnectionRefused,
    /// Connection timeout
    Timeout,
    /// Connection reset
    ConnectionReset,
    /// Invalid address
    InvalidAddress,
    /// Operation not supported
    NotSupported,
    /// Resource busy
    Busy,
    /// No route to host
    NoRoute,
    /// Address already in use
    AddressInUse,
    /// Permission denied
    PermissionDenied,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkError::InvalidPacket => write!(f, "Invalid packet format"),
            NetworkError::BufferOverflow => write!(f, "Buffer overflow"),
            NetworkError::NetworkUnreachable => write!(f, "Network unreachable"),
            NetworkError::HostUnreachable => write!(f, "Host unreachable"),
            NetworkError::PortUnreachable => write!(f, "Port unreachable"),
            NetworkError::ConnectionRefused => write!(f, "Connection refused"),
            NetworkError::Timeout => write!(f, "Connection timeout"),
            NetworkError::ConnectionReset => write!(f, "Connection reset"),
            NetworkError::InvalidAddress => write!(f, "Invalid address"),
            NetworkError::NotSupported => write!(f, "Operation not supported"),
            NetworkError::Busy => write!(f, "Resource busy"),
            NetworkError::NoRoute => write!(f, "No route to host"),
            NetworkError::AddressInUse => write!(f, "Address already in use"),
            NetworkError::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

/// Network result type
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Network interface configuration
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// MAC address
    pub mac_address: NetworkAddress,
    /// IP addresses assigned to this interface
    pub ip_addresses: Vec<NetworkAddress>,
    /// Maximum transmission unit
    pub mtu: u16,
    /// Interface flags
    pub flags: InterfaceFlags,
    /// Interface statistics
    pub stats: InterfaceStats,
}

/// Interface flags
#[derive(Debug, Clone, Copy)]
pub struct InterfaceFlags {
    /// Interface is up
    pub up: bool,
    /// Interface supports broadcast
    pub broadcast: bool,
    /// Interface is loopback
    pub loopback: bool,
    /// Interface supports multicast
    pub multicast: bool,
    /// Interface is point-to-point
    pub point_to_point: bool,
}

impl Default for InterfaceFlags {
    fn default() -> Self {
        Self {
            up: false,
            broadcast: true,
            loopback: false,
            multicast: true,
            point_to_point: false,
        }
    }
}

/// Interface statistics
#[derive(Debug, Clone, Default)]
pub struct InterfaceStats {
    /// Packets received
    pub rx_packets: u64,
    /// Bytes received
    pub rx_bytes: u64,
    /// Receive errors
    pub rx_errors: u64,
    /// Packets transmitted
    pub tx_packets: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Transmit errors
    pub tx_errors: u64,
    /// Packets dropped
    pub dropped: u64,
}

/// Routing table entry
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Destination network
    pub destination: NetworkAddress,
    /// Network mask
    pub netmask: NetworkAddress,
    /// Gateway address
    pub gateway: Option<NetworkAddress>,
    /// Output interface
    pub interface: String,
    /// Route metric
    pub metric: u32,
}

/// Network stack manager
pub struct NetworkStack {
    /// Network interfaces
    interfaces: RwLock<BTreeMap<String, NetworkInterface>>,
    /// Routing table
    routing_table: RwLock<Vec<RouteEntry>>,
    /// ARP table (IP -> MAC mapping)
    arp_table: RwLock<BTreeMap<NetworkAddress, NetworkAddress>>,
    /// Socket registry
    sockets: RwLock<BTreeMap<u32, socket::Socket>>,
    /// Next socket ID
    next_socket_id: Mutex<u32>,
}

impl NetworkStack {
    /// Create a new network stack
    pub fn new() -> Self {
        Self {
            interfaces: RwLock::new(BTreeMap::new()),
            routing_table: RwLock::new(Vec::new()),
            arp_table: RwLock::new(BTreeMap::new()),
            sockets: RwLock::new(BTreeMap::new()),
            next_socket_id: Mutex::new(1),
        }
    }

    /// Add a network interface
    pub fn add_interface(&self, interface: NetworkInterface) -> NetworkResult<()> {
        let mut interfaces = self.interfaces.write();
        
        if interfaces.contains_key(&interface.name) {
            return Err(NetworkError::AddressInUse);
        }

        println!("Added network interface: {} ({})", interface.name, interface.mac_address);
        interfaces.insert(interface.name.clone(), interface);
        Ok(())
    }

    /// Remove a network interface
    pub fn remove_interface(&self, name: &str) -> NetworkResult<()> {
        let mut interfaces = self.interfaces.write();
        
        if interfaces.remove(name).is_some() {
            println!("Removed network interface: {}", name);
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get network interface by name
    pub fn get_interface(&self, name: &str) -> Option<NetworkInterface> {
        let interfaces = self.interfaces.read();
        interfaces.get(name).cloned()
    }

    /// List all network interfaces
    pub fn list_interfaces(&self) -> Vec<NetworkInterface> {
        let interfaces = self.interfaces.read();
        interfaces.values().cloned().collect()
    }

    /// Set interface up/down
    pub fn set_interface_state(&self, name: &str, up: bool) -> NetworkResult<()> {
        let mut interfaces = self.interfaces.write();
        
        if let Some(interface) = interfaces.get_mut(name) {
            interface.flags.up = up;
            println!("Interface {} is now {}", name, if up { "up" } else { "down" });
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Add IP address to interface
    pub fn add_ip_address(&self, interface_name: &str, address: NetworkAddress) -> NetworkResult<()> {
        let mut interfaces = self.interfaces.write();
        
        if let Some(interface) = interfaces.get_mut(interface_name) {
            if !interface.ip_addresses.contains(&address) {
                interface.ip_addresses.push(address);
                println!("Added IP address {} to interface {}", address, interface_name);
            }
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Add route to routing table
    pub fn add_route(&self, route: RouteEntry) -> NetworkResult<()> {
        let mut routing_table = self.routing_table.write();
        routing_table.push(route.clone());
        
        println!("Added route: {} via {} ({})", 
            route.destination, 
            route.gateway.map(|g| g.to_string()).unwrap_or_else(|| "direct".to_string()),
            route.interface);
        
        Ok(())
    }

    /// Find route for destination address
    pub fn find_route(&self, destination: &NetworkAddress) -> Option<RouteEntry> {
        let routing_table = self.routing_table.read();
        
        // Simple routing - find first matching route
        // In a real implementation, this would do longest prefix matching
        for route in routing_table.iter() {
            if self.address_matches_route(destination, &route.destination, &route.netmask) {
                return Some(route.clone());
            }
        }
        
        None
    }

    /// Check if address matches route with netmask
    fn address_matches_route(&self, addr: &NetworkAddress, dest: &NetworkAddress, mask: &NetworkAddress) -> bool {
        match (addr, dest, mask) {
            (NetworkAddress::IPv4(a), NetworkAddress::IPv4(d), NetworkAddress::IPv4(m)) => {
                for i in 0..4 {
                    if (a[i] & m[i]) != (d[i] & m[i]) {
                        return false;
                    }
                }
                true
            }
            _ => false, // Simplified for now
        }
    }

    /// Update ARP table
    pub fn update_arp(&self, ip: NetworkAddress, mac: NetworkAddress) {
        let mut arp_table = self.arp_table.write();
        arp_table.insert(ip, mac);
    }

    /// Lookup MAC address for IP
    pub fn lookup_arp(&self, ip: &NetworkAddress) -> Option<NetworkAddress> {
        let arp_table = self.arp_table.read();
        arp_table.get(ip).copied()
    }

    /// Create a new socket
    pub fn create_socket(&self, socket_type: socket::SocketType, protocol: Protocol) -> NetworkResult<u32> {
        let socket_id = {
            let mut next_id = self.next_socket_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let socket = socket::Socket::new(socket_id, socket_type, protocol);
        
        let mut sockets = self.sockets.write();
        sockets.insert(socket_id, socket);
        
        Ok(socket_id)
    }

    /// Close a socket
    pub fn close_socket(&self, socket_id: u32) -> NetworkResult<()> {
        let mut sockets = self.sockets.write();
        
        if sockets.remove(&socket_id).is_some() {
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get socket by ID
    pub fn get_socket(&self, socket_id: u32) -> Option<socket::Socket> {
        let sockets = self.sockets.read();
        sockets.get(&socket_id).cloned()
    }

    /// Process incoming packet
    pub fn process_packet(&self, interface_name: &str, packet: PacketBuffer) -> NetworkResult<()> {
        // Update interface statistics
        {
            let mut interfaces = self.interfaces.write();
            if let Some(interface) = interfaces.get_mut(interface_name) {
                interface.stats.rx_packets += 1;
                interface.stats.rx_bytes += packet.length as u64;
            }
        }

        // Process Ethernet frame
        ethernet::process_frame(self, packet)
    }

    /// Send packet through interface
    pub fn send_packet(&self, interface_name: &str, packet: PacketBuffer) -> NetworkResult<()> {
        let interfaces = self.interfaces.read();
        let interface = interfaces.get(interface_name).ok_or(NetworkError::InvalidAddress)?;
        
        if !interface.flags.up {
            return Err(NetworkError::NetworkUnreachable);
        }

        // Update interface statistics
        drop(interfaces);
        {
            let mut interfaces = self.interfaces.write();
            if let Some(interface) = interfaces.get_mut(interface_name) {
                interface.stats.tx_packets += 1;
                interface.stats.tx_bytes += packet.length as u64;
            }
        }

        // TODO: Send to network device
        println!("Sending {} bytes on interface {}", packet.length, interface_name);
        Ok(())
    }

    /// Get network statistics
    pub fn get_stats(&self) -> NetworkStats {
        let interfaces = self.interfaces.read();
        let sockets = self.sockets.read();
        let routing_table = self.routing_table.read();
        let arp_table = self.arp_table.read();

        let mut total_rx_packets = 0;
        let mut total_rx_bytes = 0;
        let mut total_tx_packets = 0;
        let mut total_tx_bytes = 0;

        for interface in interfaces.values() {
            total_rx_packets += interface.stats.rx_packets;
            total_rx_bytes += interface.stats.rx_bytes;
            total_tx_packets += interface.stats.tx_packets;
            total_tx_bytes += interface.stats.tx_bytes;
        }

        NetworkStats {
            interfaces: interfaces.len(),
            sockets: sockets.len(),
            routes: routing_table.len(),
            arp_entries: arp_table.len(),
            total_rx_packets,
            total_rx_bytes,
            total_tx_packets,
            total_tx_bytes,
        }
    }
}

/// Network stack statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub interfaces: usize,
    pub sockets: usize,
    pub routes: usize,
    pub arp_entries: usize,
    pub total_rx_packets: u64,
    pub total_rx_bytes: u64,
    pub total_tx_packets: u64,
    pub total_tx_bytes: u64,
}

lazy_static! {
    static ref NETWORK_STACK: NetworkStack = NetworkStack::new();
}

/// Initialize the network stack
pub fn init() -> NetworkResult<()> {
    // Create loopback interface
    let loopback = NetworkInterface {
        name: "lo".to_string(),
        mac_address: NetworkAddress::mac([0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        ip_addresses: vec![NetworkAddress::ipv4(127, 0, 0, 1)],
        mtu: 65536,
        flags: InterfaceFlags {
            up: true,
            broadcast: false,
            loopback: true,
            multicast: false,
            point_to_point: false,
        },
        stats: InterfaceStats::default(),
    };

    NETWORK_STACK.add_interface(loopback)?;
    NETWORK_STACK.set_interface_state("lo", true)?;

    // Add loopback route
    let loopback_route = RouteEntry {
        destination: NetworkAddress::ipv4(127, 0, 0, 0),
        netmask: NetworkAddress::ipv4(255, 0, 0, 0),
        gateway: None,
        interface: "lo".to_string(),
        metric: 0,
    };
    NETWORK_STACK.add_route(loopback_route)?;

    println!("✓ Network stack initialized with loopback interface");
    Ok(())
}

/// Get the global network stack
pub fn network_stack() -> &'static NetworkStack {
    &NETWORK_STACK
}
