//! # Internet Protocol Version 4 (IPv4) Implementation
//!
//! This module provides IPv4 packet handling, routing, fragmentation,
//! and related network layer functionality for the RustOS network stack.

use crate::{vec, BTreeMap, String, Vec};
// use core::mem::size_of;

use crate::network::{Ipv4Address, NetworkError, Protocol};

/// IPv4 header minimum size (20 bytes without options)
pub const IPV4_HEADER_MIN_SIZE: usize = 20;

/// IPv4 header maximum size (60 bytes with maximum options)
pub const IPV4_HEADER_MAX_SIZE: usize = 60;

/// IPv4 maximum packet size
pub const IPV4_MAX_PACKET_SIZE: usize = 65535;

/// IPv4 minimum MTU
pub const IPV4_MIN_MTU: usize = 576;

/// IPv4 default TTL
pub const IPV4_DEFAULT_TTL: u8 = 64;

/// IPv4 version number
pub const IPV4_VERSION: u8 = 4;

/// IPv4 protocol numbers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IpProtocol {
    /// Internet Control Message Protocol
    Icmp = 1,
    /// Internet Group Management Protocol
    Igmp = 2,
    /// Transmission Control Protocol
    Tcp = 6,
    /// User Datagram Protocol
    Udp = 17,
    /// Internet Protocol version 6 (IPv6-in-IPv4)
    Ipv6 = 41,
    /// Generic Routing Encapsulation
    Gre = 47,
    /// Encapsulating Security Protocol
    Esp = 50,
    /// Authentication Header
    Ah = 51,
    /// Open Shortest Path First
    Ospf = 89,
}

impl From<u8> for IpProtocol {
    fn from(value: u8) -> Self {
        match value {
            1 => IpProtocol::Icmp,
            2 => IpProtocol::Igmp,
            6 => IpProtocol::Tcp,
            17 => IpProtocol::Udp,
            41 => IpProtocol::Ipv6,
            47 => IpProtocol::Gre,
            50 => IpProtocol::Esp,
            51 => IpProtocol::Ah,
            89 => IpProtocol::Ospf,
            _ => IpProtocol::Icmp, // Default fallback
        }
    }
}

impl From<IpProtocol> for u8 {
    fn from(protocol: IpProtocol) -> Self {
        protocol as u8
    }
}

impl From<Protocol> for IpProtocol {
    fn from(protocol: Protocol) -> Self {
        match protocol {
            Protocol::Tcp => IpProtocol::Tcp,
            Protocol::Udp => IpProtocol::Udp,
            Protocol::Icmp => IpProtocol::Icmp,
            Protocol::Raw => IpProtocol::Icmp, // Default for raw
        }
    }
}

/// IPv4 Type of Service (ToS) field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeOfService {
    /// Precedence (3 bits)
    pub precedence: u8,
    /// Delay flag
    pub low_delay: bool,
    /// Throughput flag
    pub high_throughput: bool,
    /// Reliability flag
    pub high_reliability: bool,
    /// Cost flag
    pub low_cost: bool,
}

impl TypeOfService {
    pub fn new() -> Self {
        Self {
            precedence: 0,
            low_delay: false,
            high_throughput: false,
            high_reliability: false,
            low_cost: false,
        }
    }

    pub fn from_byte(byte: u8) -> Self {
        Self {
            precedence: (byte >> 5) & 0x07,
            low_delay: (byte & 0x10) != 0,
            high_throughput: (byte & 0x08) != 0,
            high_reliability: (byte & 0x04) != 0,
            low_cost: (byte & 0x02) != 0,
        }
    }

    pub fn to_byte(&self) -> u8 {
        (self.precedence << 5)
            | if self.low_delay { 0x10 } else { 0 }
            | if self.high_throughput { 0x08 } else { 0 }
            | if self.high_reliability { 0x04 } else { 0 }
            | if self.low_cost { 0x02 } else { 0 }
    }
}

impl Default for TypeOfService {
    fn default() -> Self {
        Self::new()
    }
}

/// IPv4 flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpFlags {
    /// Don't Fragment flag
    pub dont_fragment: bool,
    /// More Fragments flag
    pub more_fragments: bool,
}

impl IpFlags {
    pub fn new() -> Self {
        Self {
            dont_fragment: false,
            more_fragments: false,
        }
    }

    pub fn from_u16(value: u16) -> Self {
        Self {
            dont_fragment: (value & 0x4000) != 0,
            more_fragments: (value & 0x2000) != 0,
        }
    }

    pub fn to_u16(&self) -> u16 {
        (if self.dont_fragment { 0x4000 } else { 0 })
            | (if self.more_fragments { 0x2000 } else { 0 })
    }
}

impl Default for IpFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// IPv4 header structure
#[derive(Debug, Clone)]
pub struct Ipv4Header {
    /// Version (4 bits) - should be 4
    pub version: u8,
    /// Internet Header Length (4 bits) - in 32-bit words
    pub ihl: u8,
    /// Type of Service
    pub tos: TypeOfService,
    /// Total Length (16 bits) - in bytes
    pub total_length: u16,
    /// Identification (16 bits)
    pub identification: u16,
    /// Flags (3 bits)
    pub flags: IpFlags,
    /// Fragment Offset (13 bits) - in 8-byte units
    pub fragment_offset: u16,
    /// Time to Live (8 bits)
    pub ttl: u8,
    /// Protocol (8 bits)
    pub protocol: IpProtocol,
    /// Header Checksum (16 bits)
    pub checksum: u16,
    /// Source Address (32 bits)
    pub source: Ipv4Address,
    /// Destination Address (32 bits)
    pub destination: Ipv4Address,
    /// Options (variable length)
    pub options: Vec<u8>,
}

impl Ipv4Header {
    /// Create a new IPv4 header
    pub fn new(
        source: Ipv4Address,
        destination: Ipv4Address,
        protocol: IpProtocol,
        payload_length: u16,
    ) -> Self {
        Self {
            version: IPV4_VERSION,
            ihl: 5, // 20 bytes without options
            tos: TypeOfService::default(),
            total_length: IPV4_HEADER_MIN_SIZE as u16 + payload_length,
            identification: 0,
            flags: IpFlags::default(),
            fragment_offset: 0,
            ttl: IPV4_DEFAULT_TTL,
            protocol,
            checksum: 0, // Will be calculated later
            source,
            destination,
            options: Vec::new(),
        }
    }

    /// Parse IPv4 header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        if data.len() < IPV4_HEADER_MIN_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let version_ihl = data[0];
        let version = (version_ihl >> 4) & 0x0F;
        let ihl = version_ihl & 0x0F;

        if version != IPV4_VERSION {
            return Err(NetworkError::ProtocolError);
        }

        if ihl < 5 {
            return Err(NetworkError::InvalidPacket);
        }

        let header_length = (ihl as usize) * 4;
        if data.len() < header_length {
            return Err(NetworkError::InvalidPacket);
        }

        let tos = TypeOfService::from_byte(data[1]);
        let total_length = u16::from_be_bytes([data[2], data[3]]);
        let identification = u16::from_be_bytes([data[4], data[5]]);

        let flags_and_offset = u16::from_be_bytes([data[6], data[7]]);
        let flags = IpFlags::from_u16(flags_and_offset);
        let fragment_offset = flags_and_offset & 0x1FFF;

        let ttl = data[8];
        let protocol = IpProtocol::from(data[9]);
        let checksum = u16::from_be_bytes([data[10], data[11]]);

        let source = Ipv4Address::from_bytes([data[12], data[13], data[14], data[15]]);
        let destination = Ipv4Address::from_bytes([data[16], data[17], data[18], data[19]]);

        let options = if header_length > IPV4_HEADER_MIN_SIZE {
            data[IPV4_HEADER_MIN_SIZE..header_length].to_vec()
        } else {
            Vec::new()
        };

        Ok(Self {
            version,
            ihl,
            tos,
            total_length,
            identification,
            flags,
            fragment_offset,
            ttl,
            protocol,
            checksum,
            source,
            destination,
            options,
        })
    }

    /// Convert header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_length = IPV4_HEADER_MIN_SIZE + self.options.len();
        let mut bytes = Vec::with_capacity(header_length);

        // Version and IHL
        bytes.push((self.version << 4) | self.ihl);

        // Type of Service
        bytes.push(self.tos.to_byte());

        // Total Length
        bytes.extend_from_slice(&self.total_length.to_be_bytes());

        // Identification
        bytes.extend_from_slice(&self.identification.to_be_bytes());

        // Flags and Fragment Offset
        let flags_and_offset = self.flags.to_u16() | (self.fragment_offset & 0x1FFF);
        bytes.extend_from_slice(&flags_and_offset.to_be_bytes());

        // TTL
        bytes.push(self.ttl);

        // Protocol
        bytes.push(self.protocol.into());

        // Checksum
        bytes.extend_from_slice(&self.checksum.to_be_bytes());

        // Source Address
        bytes.extend_from_slice(&self.source.to_bytes());

        // Destination Address
        bytes.extend_from_slice(&self.destination.to_bytes());

        // Options
        bytes.extend_from_slice(&self.options);

        // Pad to 32-bit boundary if needed
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }

        bytes
    }

    /// Calculate and set the header checksum
    pub fn calculate_checksum(&mut self) {
        self.checksum = 0;
        let header_bytes = self.to_bytes();
        self.checksum = Self::compute_checksum(&header_bytes);
    }

    /// Compute IPv4 header checksum
    pub fn compute_checksum(data: &[u8]) -> u16 {
        let mut sum: u32 = 0;

        // Sum all 16-bit words
        for chunk in data.chunks_exact(2) {
            sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
        }

        // Handle odd byte if present
        if data.len() % 2 == 1 {
            sum += (data[data.len() - 1] as u32) << 8;
        }

        // Add carry bits
        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        // One's complement
        !sum as u16
    }

    /// Verify header checksum
    pub fn verify_checksum(&self) -> bool {
        let header_bytes = self.to_bytes();
        Self::compute_checksum(&header_bytes) == 0
    }

    /// Get header length in bytes
    pub fn header_length(&self) -> usize {
        (self.ihl as usize) * 4
    }

    /// Get payload length
    pub fn payload_length(&self) -> usize {
        self.total_length as usize - self.header_length()
    }

    /// Check if packet is fragmented
    pub fn is_fragmented(&self) -> bool {
        self.flags.more_fragments || self.fragment_offset != 0
    }

    /// Validate header
    pub fn validate(&self) -> Result<(), NetworkError> {
        // Check version
        if self.version != IPV4_VERSION {
            return Err(NetworkError::ProtocolError);
        }

        // Check IHL
        if self.ihl < 5 {
            return Err(NetworkError::InvalidPacket);
        }

        // Check total length
        if self.total_length < self.header_length() as u16 {
            return Err(NetworkError::InvalidPacket);
        }

        // Check TTL
        if self.ttl == 0 {
            return Err(NetworkError::Timeout);
        }

        // Verify checksum
        if !self.verify_checksum() {
            return Err(NetworkError::InvalidPacket);
        }

        Ok(())
    }
}

/// IPv4 packet structure
#[derive(Debug, Clone)]
pub struct Ipv4Packet {
    /// IPv4 header
    pub header: Ipv4Header,
    /// Packet payload
    pub payload: Vec<u8>,
}

impl Ipv4Packet {
    /// Create a new IPv4 packet
    pub fn new(
        source: Ipv4Address,
        destination: Ipv4Address,
        protocol: IpProtocol,
        payload: Vec<u8>,
    ) -> Self {
        let mut header = Ipv4Header::new(source, destination, protocol, payload.len() as u16);
        header.calculate_checksum();

        Self { header, payload }
    }

    /// Parse IPv4 packet from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        let header = Ipv4Header::from_bytes(data)?;
        let header_length = header.header_length();

        if data.len() < header_length {
            return Err(NetworkError::InvalidPacket);
        }

        let payload_length = header.payload_length();
        if data.len() < header_length + payload_length {
            return Err(NetworkError::InvalidPacket);
        }

        let payload = data[header_length..header_length + payload_length].to_vec();

        Ok(Self { header, payload })
    }

    /// Convert packet to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.header.to_bytes();
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Validate packet
    pub fn validate(&self) -> Result<(), NetworkError> {
        self.header.validate()?;

        // Check payload length consistency
        if self.payload.len() != self.header.payload_length() {
            return Err(NetworkError::InvalidPacket);
        }

        Ok(())
    }

    /// Decrement TTL
    pub fn decrement_ttl(&mut self) -> Result<(), NetworkError> {
        if self.header.ttl == 0 {
            return Err(NetworkError::Timeout);
        }

        self.header.ttl -= 1;
        self.header.calculate_checksum();

        if self.header.ttl == 0 {
            return Err(NetworkError::Timeout);
        }

        Ok(())
    }

    /// Fragment packet if needed
    pub fn fragment(&self, mtu: usize) -> Result<Vec<Ipv4Packet>, NetworkError> {
        if self.to_bytes().len() <= mtu {
            return Ok(vec![self.clone()]);
        }

        if self.header.flags.dont_fragment {
            return Err(NetworkError::BufferTooSmall);
        }

        let mut fragments = Vec::new();
        let header_size = self.header.header_length();
        let max_payload = ((mtu - header_size) / 8) * 8; // Must be multiple of 8

        if max_payload == 0 {
            return Err(NetworkError::BufferTooSmall);
        }

        let total_payload = self.payload.len();
        let mut offset = 0;

        while offset < total_payload {
            let fragment_size = core::cmp::min(max_payload, total_payload - offset);
            let is_last_fragment = offset + fragment_size >= total_payload;

            let mut fragment_header = self.header.clone();
            fragment_header.fragment_offset = (offset / 8) as u16;
            fragment_header.flags.more_fragments = !is_last_fragment;
            fragment_header.total_length = (header_size + fragment_size) as u16;

            let fragment_payload = self.payload[offset..offset + fragment_size].to_vec();
            fragment_header.calculate_checksum();

            let fragment = Ipv4Packet {
                header: fragment_header,
                payload: fragment_payload,
            };

            fragments.push(fragment);
            offset += fragment_size;
        }

        Ok(fragments)
    }
}

/// IPv4 routing table entry
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Destination network
    pub network: Ipv4Address,
    /// Network mask
    pub netmask: Ipv4Address,
    /// Next hop gateway
    pub gateway: Option<Ipv4Address>,
    /// Output interface
    pub interface: String,
    /// Route metric
    pub metric: u32,
}

impl RouteEntry {
    pub fn new(
        network: Ipv4Address,
        netmask: Ipv4Address,
        gateway: Option<Ipv4Address>,
        interface: String,
        metric: u32,
    ) -> Self {
        Self {
            network,
            netmask,
            gateway,
            interface,
            metric,
        }
    }

    /// Check if destination matches this route
    pub fn matches(&self, destination: Ipv4Address) -> bool {
        let dest_bytes = destination.to_bytes();
        let net_bytes = self.network.to_bytes();
        let mask_bytes = self.netmask.to_bytes();

        for i in 0..4 {
            if (dest_bytes[i] & mask_bytes[i]) != (net_bytes[i] & mask_bytes[i]) {
                return false;
            }
        }

        true
    }

    /// Get prefix length (CIDR notation)
    pub fn prefix_length(&self) -> u8 {
        let mask_bytes = self.netmask.to_bytes();
        let mut prefix_len = 0u8;

        for byte in mask_bytes.iter() {
            if *byte == 0xFF {
                prefix_len += 8;
            } else {
                prefix_len += byte.leading_ones() as u8;
                break;
            }
        }

        prefix_len
    }
}

/// IPv4 routing table
#[derive(Debug)]
pub struct RoutingTable {
    routes: Vec<RouteEntry>,
}

impl RoutingTable {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a route
    pub fn add_route(&mut self, route: RouteEntry) {
        self.routes.push(route);
        // Sort by prefix length (longest prefix first)
        self.routes
            .sort_by(|a, b| b.prefix_length().cmp(&a.prefix_length()));
    }

    /// Remove routes for a specific interface
    pub fn remove_interface_routes(&mut self, interface: &str) {
        self.routes.retain(|route| route.interface != interface);
    }

    /// Find best route for destination
    pub fn lookup(&self, destination: Ipv4Address) -> Option<&RouteEntry> {
        self.routes.iter().find(|route| route.matches(destination))
    }

    /// Get all routes
    pub fn routes(&self) -> &[RouteEntry] {
        &self.routes
    }

    /// Clear all routes
    pub fn clear(&mut self) {
        self.routes.clear();
    }

    /// Add default route
    pub fn add_default_route(&mut self, gateway: Ipv4Address, interface: String, metric: u32) {
        let route = RouteEntry::new(
            Ipv4Address::UNSPECIFIED,
            Ipv4Address::UNSPECIFIED,
            Some(gateway),
            interface,
            metric,
        );
        self.add_route(route);
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Fragment reassembly buffer entry
#[derive(Debug, Clone)]
struct FragmentEntry {
    fragments: BTreeMap<u16, Vec<u8>>, // fragment_offset -> data
    total_length: Option<usize>,
    timestamp: u64,
    header: Ipv4Header,
}

/// Fragment reassembly manager
#[derive(Debug)]
pub struct FragmentReassembler {
    fragments: BTreeMap<(Ipv4Address, Ipv4Address, u16), FragmentEntry>, // (src, dst, id) -> fragments
    timeout_seconds: u64,
}

impl FragmentReassembler {
    pub fn new(timeout_seconds: u64) -> Self {
        Self {
            fragments: BTreeMap::new(),
            timeout_seconds,
        }
    }

    /// Add fragment and try to reassemble
    pub fn add_fragment(
        &mut self,
        packet: Ipv4Packet,
        timestamp: u64,
    ) -> Result<Option<Ipv4Packet>, NetworkError> {
        let key = (
            packet.header.source,
            packet.header.destination,
            packet.header.identification,
        );

        let fragment_offset = packet.header.fragment_offset;
        let more_fragments = packet.header.flags.more_fragments;

        // Get or create fragment entry
        let entry = self.fragments.entry(key).or_insert_with(|| FragmentEntry {
            fragments: BTreeMap::new(),
            total_length: None,
            timestamp,
            header: packet.header.clone(),
        });

        // Update timestamp
        entry.timestamp = timestamp;

        // Add fragment
        entry.fragments.insert(fragment_offset, packet.payload);

        // If this is the last fragment, we know the total length
        if !more_fragments {
            entry.total_length =
                Some((fragment_offset as usize * 8) + entry.fragments[&fragment_offset].len());
        }

        // Try to reassemble if we have the total length
        if let Some(total_length) = entry.total_length {
            let can_reassemble = {
                let mut current_offset = 0;
                let mut fragments_vec: Vec<_> = entry.fragments.iter().collect();
                fragments_vec.sort_by_key(|(offset, _)| *offset);

                for (&offset, data) in &fragments_vec {
                    if offset as usize * 8 != current_offset {
                        break;
                    }
                    current_offset += data.len();
                }
                current_offset == total_length
            };

            if can_reassemble {
                // Extract the entry from the map to avoid borrowing conflicts
                let entry = self.fragments.remove(&key).unwrap();
                let reassembled = self.reassemble_packet_owned(entry, total_length)?;
                return Ok(Some(reassembled));
            }
        }

        Ok(None)
    }

    fn can_reassemble(&self, entry: &FragmentEntry, total_length: usize) -> bool {
        let mut current_offset = 0;

        for (&offset, data) in &entry.fragments {
            if offset as usize * 8 != current_offset {
                return false; // Gap in fragments
            }
            current_offset += data.len();
        }

        current_offset == total_length
    }

    fn reassemble_packet_owned(
        &self,
        entry: FragmentEntry,
        total_length: usize,
    ) -> Result<Ipv4Packet, NetworkError> {
        let mut payload = Vec::with_capacity(total_length);

        for data in entry.fragments.values() {
            payload.extend_from_slice(data);
        }

        let mut header = entry.header.clone();
        header.flags.more_fragments = false;
        header.fragment_offset = 0;
        header.total_length = (header.header_length() + payload.len()) as u16;
        header.calculate_checksum();

        Ok(Ipv4Packet { header, payload })
    }

    /// Clean up expired fragments
    pub fn cleanup_expired(&mut self, current_time: u64) {
        self.fragments
            .retain(|_, entry| current_time <= entry.timestamp + self.timeout_seconds);
    }
}

/// IPv4 processor for handling IPv4 operations
pub struct Ipv4Processor {
    routing_table: RoutingTable,
    reassembler: FragmentReassembler,
    next_identification: u16,
}

impl Ipv4Processor {
    pub fn new() -> Self {
        Self {
            routing_table: RoutingTable::new(),
            reassembler: FragmentReassembler::new(30), // 30 second timeout
            next_identification: 1,
        }
    }

    /// Create IPv4 packet
    pub fn create_packet(
        &mut self,
        source: Ipv4Address,
        destination: Ipv4Address,
        protocol: IpProtocol,
        payload: Vec<u8>,
    ) -> Ipv4Packet {
        let mut packet = Ipv4Packet::new(source, destination, protocol, payload);
        packet.header.identification = self.next_identification;
        self.next_identification = self.next_identification.wrapping_add(1);
        packet
    }

    /// Process incoming IPv4 packet
    pub fn process_packet(
        &mut self,
        data: &[u8],
        timestamp: u64,
    ) -> Result<Option<Ipv4Packet>, NetworkError> {
        let packet = Ipv4Packet::from_bytes(data)?;
        packet.validate()?;

        // Handle fragmentation
        if packet.header.is_fragmented() {
            return self.reassembler.add_fragment(packet, timestamp);
        }

        Ok(Some(packet))
    }

    /// Find route for destination
    pub fn find_route(&self, destination: Ipv4Address) -> Option<&RouteEntry> {
        self.routing_table.lookup(destination)
    }

    /// Add route
    pub fn add_route(&mut self, route: RouteEntry) {
        self.routing_table.add_route(route);
    }

    /// Fragment packet if needed
    pub fn fragment_packet(
        &self,
        packet: &Ipv4Packet,
        mtu: usize,
    ) -> Result<Vec<Ipv4Packet>, NetworkError> {
        packet.fragment(mtu)
    }

    /// Cleanup expired fragments
    pub fn cleanup(&mut self, timestamp: u64) {
        self.reassembler.cleanup_expired(timestamp);
    }
}

impl Default for Ipv4Processor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ipv4_header_creation() {
        let source = Ipv4Address::new(192, 168, 1, 100);
        let dest = Ipv4Address::new(192, 168, 1, 1);
        let header = Ipv4Header::new(source, dest, IpProtocol::Tcp, 1024);

        assert_eq!(header.version, 4);
        assert_eq!(header.source, source);
        assert_eq!(header.destination, dest);
        assert_eq!(header.protocol, IpProtocol::Tcp);
        assert_eq!(header.total_length, 20 + 1024);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ipv4_header_serialization() {
        let source = Ipv4Address::new(10, 0, 0, 1);
        let dest = Ipv4Address::new(10, 0, 0, 2);
        let mut header = Ipv4Header::new(source, dest, IpProtocol::Udp, 100);
        header.calculate_checksum();

        let bytes = header.to_bytes();
        let parsed_header = Ipv4Header::from_bytes(&bytes).unwrap();

        assert_eq!(parsed_header.source, source);
        assert_eq!(parsed_header.destination, dest);
        assert_eq!(parsed_header.protocol, IpProtocol::Udp);
        assert!(parsed_header.verify_checksum());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_routing_table() {
        let mut table = RoutingTable::new();

        // Add default route
        table.add_default_route(Ipv4Address::new(192, 168, 1, 1), "eth0".to_string(), 0);

        // Add specific network route
        let route = RouteEntry::new(
            Ipv4Address::new(192, 168, 1, 0),
            Ipv4Address::new(255, 255, 255, 0),
            None,
            "eth0".to_string(),
            0,
        );
        table.add_route(route);

        // Test lookups
        let dest1 = Ipv4Address::new(192, 168, 1, 50);
        let route1 = table.lookup(dest1).unwrap();
        assert_eq!(route1.interface, "eth0");

        let dest2 = Ipv4Address::new(10, 0, 0, 1);
        let route2 = table.lookup(dest2).unwrap();
        assert_eq!(route2.network, Ipv4Address::UNSPECIFIED); // Default route
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_packet_fragmentation() {
        let source = Ipv4Address::new(192, 168, 1, 1);
        let dest = Ipv4Address::new(192, 168, 1, 2);
        let large_payload = vec![0u8; 2000]; // Larger than typical MTU

        let packet = Ipv4Packet::new(source, dest, IpProtocol::Tcp, large_payload);

        let mtu = 1500;
        let fragments = packet.fragment(mtu).unwrap();

        assert!(fragments.len() > 1);

        // Check that all fragments have the same identification
        let id = fragments[0].header.identification;
        for fragment in &fragments {
            assert_eq!(fragment.header.identification, id);
        }

        // Check that only the last fragment has more_fragments = false
        for (i, fragment) in fragments.iter().enumerate() {
            if i < fragments.len() - 1 {
                assert!(fragment.header.flags.more_fragments);
            } else {
                assert!(!fragment.header.flags.more_fragments);
            }
        }
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_checksum_calculation() {
        let source = Ipv4Address::new(192, 168, 1, 1);
        let dest = Ipv4Address::new(192, 168, 1, 2);
        let mut header = Ipv4Header::new(source, dest, IpProtocol::Icmp, 0);

        header.calculate_checksum();
        assert!(header.verify_checksum());

        // Corrupt a byte and verify checksum fails
        let mut bytes = header.to_bytes();
        bytes[0] ^= 0xFF;
        let corrupted_header = Ipv4Header::from_bytes(&bytes).unwrap();
        assert!(!corrupted_header.verify_checksum());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_route_matching() {
        let route = RouteEntry::new(
            Ipv4Address::new(192, 168, 0, 0),
            Ipv4Address::new(255, 255, 0, 0),
            None,
            "eth0".to_string(),
            0,
        );

        assert!(route.matches(Ipv4Address::new(192, 168, 1, 1)));
        assert!(route.matches(Ipv4Address::new(192, 168, 255, 254)));
        assert!(!route.matches(Ipv4Address::new(192, 169, 1, 1)));
        assert!(!route.matches(Ipv4Address::new(10, 0, 0, 1)));
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_protocol_conversions() {
        assert_eq!(u8::from(IpProtocol::Tcp), 6);
        assert_eq!(u8::from(IpProtocol::Udp), 17);
        assert_eq!(u8::from(IpProtocol::Icmp), 1);

        assert_eq!(IpProtocol::from(6u8), IpProtocol::Tcp);
        assert_eq!(IpProtocol::from(17u8), IpProtocol::Udp);
        assert_eq!(IpProtocol::from(1u8), IpProtocol::Icmp);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_tos_flags() {
        let mut tos = TypeOfService::new();
        tos.low_delay = true;
        tos.high_throughput = true;
        tos.precedence = 5;

        let byte = tos.to_byte();
        let parsed_tos = TypeOfService::from_byte(byte);

        assert_eq!(parsed_tos.low_delay, true);
        assert_eq!(parsed_tos.high_throughput, true);
        assert_eq!(parsed_tos.precedence, 5);
        assert_eq!(parsed_tos.high_reliability, false);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_fragment_flags() {
        let mut flags = IpFlags::new();
        flags.dont_fragment = true;
        flags.more_fragments = false;

        let value = flags.to_u16();
        let parsed_flags = IpFlags::from_u16(value);

        assert_eq!(parsed_flags.dont_fragment, true);
        assert_eq!(parsed_flags.more_fragments, false);
    }
}
