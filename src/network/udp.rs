//! # User Datagram Protocol (UDP) Implementation
//!
//! This module provides UDP functionality for the RustOS network stack,
//! including packet handling, socket management, and port binding.

use crate::{BTreeMap, Vec};
// use core::fmt;

use crate::network::{Ipv4Address, NetworkError, SocketAddr};

/// UDP header size (8 bytes)
pub const UDP_HEADER_SIZE: usize = 8;

/// UDP maximum packet size (64KB - IP header - UDP header)
pub const UDP_MAX_PACKET_SIZE: usize = 65507;

/// UDP minimum port number for dynamic allocation
pub const UDP_DYNAMIC_PORT_MIN: u16 = 32768;

/// UDP maximum port number
pub const UDP_DYNAMIC_PORT_MAX: u16 = 65535;

/// Well-known UDP ports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum UdpPort {
    /// Domain Name System
    Dns = 53,
    /// Dynamic Host Configuration Protocol (Client)
    DhcpClient = 68,
    /// Dynamic Host Configuration Protocol (Server)
    DhcpServer = 67,
    /// Network Time Protocol
    Ntp = 123,
    /// Simple Network Management Protocol
    Snmp = 161,
    /// Trivial File Transfer Protocol
    Tftp = 69,
    /// Bootstrap Protocol (same as DHCP server)
    Bootp,
    /// Network File System
    Nfs = 2049,
    /// Syslog
    Syslog = 514,
}

impl From<UdpPort> for u16 {
    fn from(port: UdpPort) -> Self {
        port as u16
    }
}

/// UDP header structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpHeader {
    /// Source port number
    pub source_port: u16,
    /// Destination port number
    pub destination_port: u16,
    /// Length of UDP header and data
    pub length: u16,
    /// Checksum (optional in IPv4, required in IPv6)
    pub checksum: u16,
}

impl UdpHeader {
    /// Create a new UDP header
    pub fn new(source_port: u16, destination_port: u16, data_length: u16) -> Self {
        Self {
            source_port,
            destination_port,
            length: UDP_HEADER_SIZE as u16 + data_length,
            checksum: 0, // Will be calculated later
        }
    }

    /// Parse UDP header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        if data.len() < UDP_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let source_port = u16::from_be_bytes([data[0], data[1]]);
        let destination_port = u16::from_be_bytes([data[2], data[3]]);
        let length = u16::from_be_bytes([data[4], data[5]]);
        let checksum = u16::from_be_bytes([data[6], data[7]]);

        // Validate length
        if length < UDP_HEADER_SIZE as u16 {
            return Err(NetworkError::InvalidPacket);
        }

        Ok(Self {
            source_port,
            destination_port,
            length,
            checksum,
        })
    }

    /// Convert header to bytes
    pub fn to_bytes(&self) -> [u8; UDP_HEADER_SIZE] {
        let mut bytes = [0u8; UDP_HEADER_SIZE];

        bytes[0..2].copy_from_slice(&self.source_port.to_be_bytes());
        bytes[2..4].copy_from_slice(&self.destination_port.to_be_bytes());
        bytes[4..6].copy_from_slice(&self.length.to_be_bytes());
        bytes[6..8].copy_from_slice(&self.checksum.to_be_bytes());

        bytes
    }

    /// Get payload length
    pub fn payload_length(&self) -> u16 {
        self.length - UDP_HEADER_SIZE as u16
    }

    /// Validate header
    pub fn validate(&self) -> Result<(), NetworkError> {
        if self.length < UDP_HEADER_SIZE as u16 {
            return Err(NetworkError::InvalidPacket);
        }

        if self.length as usize > UDP_MAX_PACKET_SIZE {
            return Err(NetworkError::BufferTooSmall);
        }

        Ok(())
    }
}

/// UDP packet structure
#[derive(Debug, Clone)]
pub struct UdpPacket {
    /// UDP header
    pub header: UdpHeader,
    /// Packet payload
    pub payload: Vec<u8>,
}

impl UdpPacket {
    /// Create a new UDP packet
    pub fn new(source_port: u16, destination_port: u16, payload: Vec<u8>) -> Self {
        let header = UdpHeader::new(source_port, destination_port, payload.len() as u16);
        Self { header, payload }
    }

    /// Parse UDP packet from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        let header = UdpHeader::from_bytes(data)?;

        if data.len() < header.length as usize {
            return Err(NetworkError::InvalidPacket);
        }

        let payload = data[UDP_HEADER_SIZE..header.length as usize].to_vec();

        let packet = Self { header, payload };
        packet.validate()?;

        Ok(packet)
    }

    /// Convert packet to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.header.length as usize);
        bytes.extend_from_slice(&self.header.to_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Validate packet
    pub fn validate(&self) -> Result<(), NetworkError> {
        self.header.validate()?;

        if self.payload.len() != self.header.payload_length() as usize {
            return Err(NetworkError::InvalidPacket);
        }

        Ok(())
    }

    /// Calculate and set UDP checksum
    pub fn calculate_checksum(&mut self, source_ip: Ipv4Address, destination_ip: Ipv4Address) {
        self.header.checksum = 0;
        let checksum =
            Self::compute_checksum(&self.header, &self.payload, source_ip, destination_ip);
        self.header.checksum = checksum;
    }

    /// Compute UDP checksum including pseudo-header
    pub fn compute_checksum(
        header: &UdpHeader,
        payload: &[u8],
        source_ip: Ipv4Address,
        destination_ip: Ipv4Address,
    ) -> u16 {
        let mut sum: u32 = 0;

        // IPv4 pseudo-header
        let source_bytes = source_ip.to_bytes();
        let dest_bytes = destination_ip.to_bytes();

        // Source IP
        sum += u16::from_be_bytes([source_bytes[0], source_bytes[1]]) as u32;
        sum += u16::from_be_bytes([source_bytes[2], source_bytes[3]]) as u32;

        // Destination IP
        sum += u16::from_be_bytes([dest_bytes[0], dest_bytes[1]]) as u32;
        sum += u16::from_be_bytes([dest_bytes[2], dest_bytes[3]]) as u32;

        // Protocol (UDP = 17)
        sum += 17;

        // UDP length
        sum += header.length as u32;

        // UDP header (with checksum = 0)
        let mut header_copy = *header;
        header_copy.checksum = 0;
        let header_bytes = header_copy.to_bytes();

        for chunk in header_bytes.chunks_exact(2) {
            sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
        }

        // UDP payload
        for chunk in payload.chunks_exact(2) {
            sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
        }

        // Handle odd byte in payload
        if payload.len() % 2 == 1 {
            sum += (payload[payload.len() - 1] as u32) << 8;
        }

        // Add carry bits
        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        // One's complement
        let checksum = !sum as u16;

        // UDP checksum of 0 is represented as 0xFFFF
        if checksum == 0 {
            0xFFFF
        } else {
            checksum
        }
    }

    /// Verify UDP checksum
    pub fn verify_checksum(&self, source_ip: Ipv4Address, destination_ip: Ipv4Address) -> bool {
        // If checksum is 0, verification is skipped (optional for IPv4)
        if self.header.checksum == 0 {
            return true;
        }

        let computed =
            Self::compute_checksum(&self.header, &self.payload, source_ip, destination_ip);
        computed == self.header.checksum
    }

    /// Get packet size
    pub fn size(&self) -> usize {
        self.header.length as usize
    }
}

/// UDP socket state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdpSocketState {
    /// Socket is closed
    Closed,
    /// Socket is bound to a local port
    Bound,
    /// Socket is connected to a remote address
    Connected,
}

/// UDP socket structure
#[derive(Debug)]
pub struct UdpSocket {
    /// Socket state
    pub state: UdpSocketState,
    /// Local address binding
    pub local_addr: Option<SocketAddr>,
    /// Remote address (for connected sockets)
    pub remote_addr: Option<SocketAddr>,
    /// Receive buffer
    pub recv_buffer: Vec<UdpPacket>,
    /// Maximum receive buffer size
    pub recv_buffer_size: usize,
    /// Socket options
    pub broadcast: bool,
    /// Reuse address option
    pub reuse_addr: bool,
}

impl UdpSocket {
    /// Create a new UDP socket
    pub fn new() -> Self {
        Self {
            state: UdpSocketState::Closed,
            local_addr: None,
            remote_addr: None,
            recv_buffer: Vec::new(),
            recv_buffer_size: 64, // Default buffer size
            broadcast: false,
            reuse_addr: false,
        }
    }

    /// Bind socket to local address
    pub fn bind(&mut self, addr: SocketAddr) -> Result<(), NetworkError> {
        if self.state != UdpSocketState::Closed {
            return Err(NetworkError::InvalidAddress);
        }

        self.local_addr = Some(addr);
        self.state = UdpSocketState::Bound;
        Ok(())
    }

    /// Connect socket to remote address
    pub fn connect(&mut self, addr: SocketAddr) -> Result<(), NetworkError> {
        if self.state == UdpSocketState::Closed {
            return Err(NetworkError::InvalidAddress);
        }

        self.remote_addr = Some(addr);
        self.state = UdpSocketState::Connected;
        Ok(())
    }

    /// Send data to specified address
    pub fn send_to(&self, data: &[u8], addr: SocketAddr) -> Result<UdpPacket, NetworkError> {
        if self.state == UdpSocketState::Closed {
            return Err(NetworkError::InvalidAddress);
        }

        let local_port = self.local_addr.ok_or(NetworkError::InvalidAddress)?.port;
        let packet = UdpPacket::new(local_port, addr.port, data.to_vec());

        Ok(packet)
    }

    /// Send data to connected address
    pub fn send(&self, data: &[u8]) -> Result<UdpPacket, NetworkError> {
        let remote_addr = self.remote_addr.ok_or(NetworkError::InvalidAddress)?;
        self.send_to(data, remote_addr)
    }

    /// Receive data from buffer
    pub fn recv(&mut self) -> Option<(Vec<u8>, SocketAddr)> {
        if let Some(packet) = self.recv_buffer.pop() {
            let addr = SocketAddr::new(Ipv4Address::LOCALHOST, packet.header.source_port); // Simplified
            Some((packet.payload, addr))
        } else {
            None
        }
    }

    /// Add packet to receive buffer
    pub fn add_to_recv_buffer(&mut self, packet: UdpPacket) -> Result<(), NetworkError> {
        if self.recv_buffer.len() >= self.recv_buffer_size {
            return Err(NetworkError::InsufficientMemory);
        }

        self.recv_buffer.push(packet);
        Ok(())
    }

    /// Check if socket can accept packet
    pub fn can_accept_packet(&self, packet: &UdpPacket) -> bool {
        match self.local_addr {
            Some(local_addr) => {
                if packet.header.destination_port != local_addr.port {
                    return false;
                }

                // If connected, check source
                if let Some(remote_addr) = self.remote_addr {
                    if packet.header.source_port != remote_addr.port {
                        return false;
                    }
                }

                true
            }
            None => false,
        }
    }

    /// Close socket
    pub fn close(&mut self) {
        self.state = UdpSocketState::Closed;
        self.local_addr = None;
        self.remote_addr = None;
        self.recv_buffer.clear();
    }

    /// Set broadcast option
    pub fn set_broadcast(&mut self, enabled: bool) {
        self.broadcast = enabled;
    }

    /// Set reuse address option
    pub fn set_reuse_addr(&mut self, enabled: bool) {
        self.reuse_addr = enabled;
    }

    /// Get local address
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    /// Get remote address
    pub fn remote_addr(&self) -> Option<SocketAddr> {
        self.remote_addr
    }
}

impl Default for UdpSocket {
    fn default() -> Self {
        Self::new()
    }
}

/// UDP port manager
#[derive(Debug)]
pub struct UdpPortManager {
    /// Currently bound ports
    bound_ports: BTreeMap<u16, bool>, // port -> is_reusable
    /// Next dynamic port to try
    next_dynamic_port: u16,
}

impl UdpPortManager {
    pub fn new() -> Self {
        Self {
            bound_ports: BTreeMap::new(),
            next_dynamic_port: UDP_DYNAMIC_PORT_MIN,
        }
    }

    /// Allocate a dynamic port
    pub fn allocate_port(&mut self) -> Result<u16, NetworkError> {
        let start_port = self.next_dynamic_port;

        loop {
            if !self.bound_ports.contains_key(&self.next_dynamic_port) {
                let port = self.next_dynamic_port;
                self.bound_ports.insert(port, false);
                self.next_dynamic_port = if self.next_dynamic_port == UDP_DYNAMIC_PORT_MAX {
                    UDP_DYNAMIC_PORT_MIN
                } else {
                    self.next_dynamic_port + 1
                };
                return Ok(port);
            }

            self.next_dynamic_port = if self.next_dynamic_port == UDP_DYNAMIC_PORT_MAX {
                UDP_DYNAMIC_PORT_MIN
            } else {
                self.next_dynamic_port + 1
            };

            // If we've wrapped around, no ports available
            if self.next_dynamic_port == start_port {
                return Err(NetworkError::PortInUse);
            }
        }
    }

    /// Bind to specific port
    pub fn bind_port(&mut self, port: u16, reusable: bool) -> Result<(), NetworkError> {
        match self.bound_ports.get(&port) {
            Some(is_reusable) => {
                if *is_reusable && reusable {
                    Ok(()) // Allow reuse
                } else {
                    Err(NetworkError::PortInUse)
                }
            }
            None => {
                self.bound_ports.insert(port, reusable);
                Ok(())
            }
        }
    }

    /// Release port
    pub fn release_port(&mut self, port: u16) {
        self.bound_ports.remove(&port);
    }

    /// Check if port is available
    pub fn is_port_available(&self, port: u16) -> bool {
        !self.bound_ports.contains_key(&port)
    }

    /// Get all bound ports
    pub fn bound_ports(&self) -> impl Iterator<Item = &u16> {
        self.bound_ports.keys()
    }
}

impl Default for UdpPortManager {
    fn default() -> Self {
        Self::new()
    }
}

/// UDP processor for managing UDP operations
#[derive(Debug)]
pub struct UdpProcessor {
    /// UDP sockets indexed by local port
    sockets: BTreeMap<u16, UdpSocket>,
    /// Port manager
    port_manager: UdpPortManager,
}

impl UdpProcessor {
    pub fn new() -> Self {
        Self {
            sockets: BTreeMap::new(),
            port_manager: UdpPortManager::new(),
        }
    }

    /// Create a new socket
    pub fn create_socket(&mut self) -> Result<u16, NetworkError> {
        let port = self.port_manager.allocate_port()?;
        let mut socket = UdpSocket::new();
        socket.bind(SocketAddr::new(Ipv4Address::UNSPECIFIED, port))?;

        self.sockets.insert(port, socket);
        Ok(port)
    }

    /// Bind socket to address
    pub fn bind_socket(&mut self, socket_id: u16, addr: SocketAddr) -> Result<(), NetworkError> {
        // Release old port if different
        if let Some(socket) = self.sockets.get(&socket_id) {
            if let Some(old_addr) = socket.local_addr {
                if old_addr.port != addr.port {
                    self.port_manager.release_port(old_addr.port);
                }
            }
        }

        // Bind new port
        self.port_manager.bind_port(addr.port, false)?;

        // Update socket
        let socket = self
            .sockets
            .get_mut(&socket_id)
            .ok_or(NetworkError::InvalidAddress)?;
        socket.bind(addr)?;

        Ok(())
    }

    /// Send UDP packet
    pub fn send_packet(
        &self,
        socket_id: u16,
        data: &[u8],
        dest_addr: SocketAddr,
    ) -> Result<UdpPacket, NetworkError> {
        let socket = self
            .sockets
            .get(&socket_id)
            .ok_or(NetworkError::InvalidAddress)?;
        socket.send_to(data, dest_addr)
    }

    /// Process incoming UDP packet
    pub fn process_packet(&mut self, packet: UdpPacket) -> Result<(), NetworkError> {
        let dest_port = packet.header.destination_port;

        if let Some(socket) = self.sockets.get_mut(&dest_port) {
            if socket.can_accept_packet(&packet) {
                socket.add_to_recv_buffer(packet)?;
            }
        }

        Ok(())
    }

    /// Receive data from socket
    pub fn receive_from_socket(&mut self, socket_id: u16) -> Option<(Vec<u8>, SocketAddr)> {
        self.sockets.get_mut(&socket_id)?.recv()
    }

    /// Close socket
    pub fn close_socket(&mut self, socket_id: u16) -> Result<(), NetworkError> {
        if let Some(mut socket) = self.sockets.remove(&socket_id) {
            if let Some(addr) = socket.local_addr {
                self.port_manager.release_port(addr.port);
            }
            socket.close();
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Get socket count
    pub fn socket_count(&self) -> usize {
        self.sockets.len()
    }

    /// Get socket info
    pub fn get_socket_info(
        &self,
        socket_id: u16,
    ) -> Option<(UdpSocketState, Option<SocketAddr>, Option<SocketAddr>)> {
        self.sockets
            .get(&socket_id)
            .map(|socket| (socket.state, socket.local_addr, socket.remote_addr))
    }
}

impl Default for UdpProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level UDP functions
pub fn create_udp_packet(source_port: u16, dest_port: u16, payload: &[u8]) -> UdpPacket {
    UdpPacket::new(source_port, dest_port, payload.to_vec())
}

pub fn parse_udp_packet(data: &[u8]) -> Result<UdpPacket, NetworkError> {
    UdpPacket::from_bytes(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_header_creation() {
        let header = UdpHeader::new(1234, 5678, 100);

        assert_eq!(header.source_port, 1234);
        assert_eq!(header.destination_port, 5678);
        assert_eq!(header.length, 108); // 8 + 100
        assert_eq!(header.checksum, 0);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_header_serialization() {
        let header = UdpHeader::new(80, 8080, 50);
        let bytes = header.to_bytes();
        let parsed_header = UdpHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed_header.source_port, header.source_port);
        assert_eq!(parsed_header.destination_port, header.destination_port);
        assert_eq!(parsed_header.length, header.length);
        assert_eq!(parsed_header.checksum, header.checksum);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_packet_creation() {
        let payload = b"Hello, UDP!".to_vec();
        let packet = UdpPacket::new(1234, 5678, payload.clone());

        assert_eq!(packet.header.source_port, 1234);
        assert_eq!(packet.header.destination_port, 5678);
        assert_eq!(packet.header.length, 8 + payload.len() as u16);
        assert_eq!(packet.payload, payload);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_packet_serialization() {
        let payload = b"Test data".to_vec();
        let packet = UdpPacket::new(2000, 3000, payload.clone());

        let bytes = packet.to_bytes();
        let parsed_packet = UdpPacket::from_bytes(&bytes).unwrap();

        assert_eq!(parsed_packet.header.source_port, packet.header.source_port);
        assert_eq!(
            parsed_packet.header.destination_port,
            packet.header.destination_port
        );
        assert_eq!(parsed_packet.payload, packet.payload);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_checksum() {
        let mut packet = UdpPacket::new(53, 12345, b"DNS query".to_vec());
        let source_ip = Ipv4Address::new(192, 168, 1, 100);
        let dest_ip = Ipv4Address::new(8, 8, 8, 8);

        packet.calculate_checksum(source_ip, dest_ip);
        assert!(packet.verify_checksum(source_ip, dest_ip));

        // Test with corrupted checksum
        packet.header.checksum ^= 0x1234;
        assert!(!packet.verify_checksum(source_ip, dest_ip));
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_socket() {
        let mut socket = UdpSocket::new();
        let addr = SocketAddr::new(Ipv4Address::LOCALHOST, 8080);

        assert_eq!(socket.state, UdpSocketState::Closed);

        socket.bind(addr).unwrap();
        assert_eq!(socket.state, UdpSocketState::Bound);
        assert_eq!(socket.local_addr(), Some(addr));

        let remote_addr = SocketAddr::new(Ipv4Address::new(192, 168, 1, 1), 9090);
        socket.connect(remote_addr).unwrap();
        assert_eq!(socket.state, UdpSocketState::Connected);
        assert_eq!(socket.remote_addr(), Some(remote_addr));
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_port_manager() {
        let mut manager = UdpPortManager::new();

        // Test port allocation
        let port1 = manager.allocate_port().unwrap();
        let port2 = manager.allocate_port().unwrap();
        assert_ne!(port1, port2);

        // Test specific port binding
        assert!(manager.bind_port(8080, false).is_ok());
        assert!(manager.bind_port(8080, false).is_err()); // Already in use
        assert!(manager.bind_port(8080, true).is_err()); // Still in use (not reusable)

        // Test port release
        manager.release_port(8080);
        assert!(manager.bind_port(8080, false).is_ok());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_udp_processor() {
        let mut processor = UdpProcessor::new();

        // Create socket
        let socket_id = processor.create_socket().unwrap();

        // Send packet
        let dest_addr = SocketAddr::new(Ipv4Address::new(192, 168, 1, 1), 80);
        let packet = processor
            .send_packet(socket_id, b"Hello", dest_addr)
            .unwrap();

        assert_eq!(packet.payload, b"Hello");
        assert_eq!(packet.header.destination_port, 80);

        // Close socket
        assert!(processor.close_socket(socket_id).is_ok());
        assert_eq!(processor.socket_count(), 0);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_well_known_ports() {
        assert_eq!(u16::from(UdpPort::Dns), 53);
        assert_eq!(u16::from(UdpPort::DhcpServer), 67);
        assert_eq!(u16::from(UdpPort::DhcpClient), 68);
        assert_eq!(u16::from(UdpPort::Ntp), 123);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_packet_validation() {
        // Test invalid header length
        let mut header = UdpHeader::new(1234, 5678, 100);
        header.length = 5; // Too small
        assert!(header.validate().is_err());

        // Test oversized packet
        header.length = (UDP_MAX_PACKET_SIZE + 1) as u16;
        assert!(header.validate().is_err());
    }
}
