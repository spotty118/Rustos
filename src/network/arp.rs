//! # Address Resolution Protocol (ARP) Implementation
//!
//! This module provides ARP functionality for mapping IPv4 addresses to MAC addresses
//! in the RustOS network stack, including packet handling, table management, and caching.

// use crate::{format, vec}; // Unused
use crate::{BTreeMap, Vec};
// use alloc::string::ToString; // Unused
// use core::time::Duration; // Not available in no_std

use crate::network::{Ipv4Address, MacAddress, NetworkError};

/// ARP packet minimum size
pub const ARP_PACKET_SIZE: usize = 28;

/// ARP hardware types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ArpHardwareType {
    /// Ethernet (10Mb)
    Ethernet = 1,
    /// Experimental Ethernet (3Mb)
    ExpEthernet = 2,
    /// Amateur Radio AX.25
    AmateurRadio = 3,
    /// Proteon ProNET Token Ring
    TokenRing = 4,
    /// Chaos
    Chaos = 5,
    /// IEEE 802 Networks
    Ieee802 = 6,
    /// ARCNET
    Arcnet = 7,
}

impl From<u16> for ArpHardwareType {
    fn from(value: u16) -> Self {
        match value {
            1 => ArpHardwareType::Ethernet,
            2 => ArpHardwareType::ExpEthernet,
            3 => ArpHardwareType::AmateurRadio,
            4 => ArpHardwareType::TokenRing,
            5 => ArpHardwareType::Chaos,
            6 => ArpHardwareType::Ieee802,
            7 => ArpHardwareType::Arcnet,
            _ => ArpHardwareType::Ethernet, // Default to Ethernet
        }
    }
}

impl From<ArpHardwareType> for u16 {
    fn from(hw_type: ArpHardwareType) -> Self {
        hw_type as u16
    }
}

/// ARP protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ArpProtocolType {
    /// Internet Protocol version 4
    Ipv4 = 0x0800,
    /// Internet Protocol version 6
    Ipv6 = 0x86DD,
}

impl From<u16> for ArpProtocolType {
    fn from(value: u16) -> Self {
        match value {
            0x0800 => ArpProtocolType::Ipv4,
            0x86DD => ArpProtocolType::Ipv6,
            _ => ArpProtocolType::Ipv4, // Default to IPv4
        }
    }
}

impl From<ArpProtocolType> for u16 {
    fn from(proto_type: ArpProtocolType) -> Self {
        proto_type as u16
    }
}

/// ARP operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ArpOperation {
    /// ARP request
    Request = 1,
    /// ARP response
    Response = 2,
    /// RARP request
    RarpRequest = 3,
    /// RARP response
    RarpResponse = 4,
}

impl From<u16> for ArpOperation {
    fn from(value: u16) -> Self {
        match value {
            1 => ArpOperation::Request,
            2 => ArpOperation::Response,
            3 => ArpOperation::RarpRequest,
            4 => ArpOperation::RarpResponse,
            _ => ArpOperation::Request, // Default to request
        }
    }
}

impl From<ArpOperation> for u16 {
    fn from(operation: ArpOperation) -> Self {
        operation as u16
    }
}

/// ARP packet structure for Ethernet and IPv4
#[derive(Debug, Clone)]
pub struct ArpPacket {
    /// Hardware type
    pub hardware_type: ArpHardwareType,
    /// Protocol type
    pub protocol_type: ArpProtocolType,
    /// Hardware address length
    pub hardware_length: u8,
    /// Protocol address length
    pub protocol_length: u8,
    /// Operation
    pub operation: ArpOperation,
    /// Sender hardware address
    pub sender_hardware_addr: MacAddress,
    /// Sender protocol address
    pub sender_protocol_addr: Ipv4Address,
    /// Target hardware address
    pub target_hardware_addr: MacAddress,
    /// Target protocol address
    pub target_protocol_addr: Ipv4Address,
}

impl ArpPacket {
    /// Create a new ARP request packet
    pub fn new_request(
        sender_mac: MacAddress,
        sender_ip: Ipv4Address,
        target_ip: Ipv4Address,
    ) -> Self {
        Self {
            hardware_type: ArpHardwareType::Ethernet,
            protocol_type: ArpProtocolType::Ipv4,
            hardware_length: 6,
            protocol_length: 4,
            operation: ArpOperation::Request,
            sender_hardware_addr: sender_mac,
            sender_protocol_addr: sender_ip,
            target_hardware_addr: MacAddress::ZERO,
            target_protocol_addr: target_ip,
        }
    }

    /// Create a new ARP response packet
    pub fn new_response(
        sender_mac: MacAddress,
        sender_ip: Ipv4Address,
        target_mac: MacAddress,
        target_ip: Ipv4Address,
    ) -> Self {
        Self {
            hardware_type: ArpHardwareType::Ethernet,
            protocol_type: ArpProtocolType::Ipv4,
            hardware_length: 6,
            protocol_length: 4,
            operation: ArpOperation::Response,
            sender_hardware_addr: sender_mac,
            sender_protocol_addr: sender_ip,
            target_hardware_addr: target_mac,
            target_protocol_addr: target_ip,
        }
    }

    /// Parse ARP packet from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        if data.len() < ARP_PACKET_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let hardware_type = ArpHardwareType::from(u16::from_be_bytes([data[0], data[1]]));
        let protocol_type = ArpProtocolType::from(u16::from_be_bytes([data[2], data[3]]));
        let hardware_length = data[4];
        let protocol_length = data[5];
        let operation = ArpOperation::from(u16::from_be_bytes([data[6], data[7]]));

        // Validate lengths for Ethernet/IPv4
        if hardware_length != 6 || protocol_length != 4 {
            return Err(NetworkError::ProtocolError);
        }

        let sender_hardware_addr =
            MacAddress::new([data[8], data[9], data[10], data[11], data[12], data[13]]);

        let sender_protocol_addr = Ipv4Address::new(data[14], data[15], data[16], data[17]);

        let target_hardware_addr =
            MacAddress::new([data[18], data[19], data[20], data[21], data[22], data[23]]);

        let target_protocol_addr = Ipv4Address::new(data[24], data[25], data[26], data[27]);

        Ok(Self {
            hardware_type,
            protocol_type,
            hardware_length,
            protocol_length,
            operation,
            sender_hardware_addr,
            sender_protocol_addr,
            target_hardware_addr,
            target_protocol_addr,
        })
    }

    /// Convert ARP packet to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(ARP_PACKET_SIZE);

        // Hardware type
        bytes.extend_from_slice(&u16::to_be_bytes(self.hardware_type.into()));

        // Protocol type
        bytes.extend_from_slice(&u16::to_be_bytes(self.protocol_type.into()));

        // Hardware and protocol lengths
        bytes.push(self.hardware_length);
        bytes.push(self.protocol_length);

        // Operation
        bytes.extend_from_slice(&u16::to_be_bytes(self.operation.into()));

        // Sender hardware address
        bytes.extend_from_slice(&self.sender_hardware_addr.0);

        // Sender protocol address
        bytes.extend_from_slice(&self.sender_protocol_addr.0);

        // Target hardware address
        bytes.extend_from_slice(&self.target_hardware_addr.0);

        // Target protocol address
        bytes.extend_from_slice(&self.target_protocol_addr.0);

        bytes
    }

    /// Validate ARP packet
    pub fn validate(&self) -> Result<(), NetworkError> {
        // Check hardware and protocol types
        if self.hardware_type != ArpHardwareType::Ethernet {
            return Err(NetworkError::ProtocolError);
        }

        if self.protocol_type != ArpProtocolType::Ipv4 {
            return Err(NetworkError::ProtocolError);
        }

        // Check address lengths
        if self.hardware_length != 6 || self.protocol_length != 4 {
            return Err(NetworkError::ProtocolError);
        }

        // Validate operation
        match self.operation {
            ArpOperation::Request | ArpOperation::Response => {}
            _ => return Err(NetworkError::ProtocolError),
        }

        // Check for valid addresses
        if self.sender_hardware_addr == MacAddress::ZERO && self.operation == ArpOperation::Response
        {
            return Err(NetworkError::InvalidAddress);
        }

        Ok(())
    }

    /// Check if this is a request for our IP
    pub fn is_request_for(&self, our_ip: Ipv4Address) -> bool {
        self.operation == ArpOperation::Request && self.target_protocol_addr == our_ip
    }

    /// Check if this is a response we're interested in
    pub fn is_response_from(&self, target_ip: Ipv4Address) -> bool {
        self.operation == ArpOperation::Response && self.sender_protocol_addr == target_ip
    }
}

/// ARP table entry with timeout
#[derive(Debug, Clone)]
pub struct ArpEntry {
    /// MAC address
    pub mac_address: MacAddress,
    /// Entry creation timestamp (simplified)
    pub timestamp: u64,
    /// Entry is static (never expires)
    pub is_static: bool,
}

impl ArpEntry {
    /// Create a new ARP entry
    pub fn new(mac_address: MacAddress, timestamp: u64, is_static: bool) -> Self {
        Self {
            mac_address,
            timestamp,
            is_static,
        }
    }

    /// Check if entry has expired (simplified timeout check)
    pub fn has_expired(&self, current_time: u64, timeout_seconds: u64) -> bool {
        if self.is_static {
            return false;
        }
        current_time > self.timestamp + timeout_seconds
    }
}

/// ARP table for caching IP to MAC mappings
#[derive(Debug)]
pub struct ArpTable {
    /// Table entries
    entries: BTreeMap<Ipv4Address, ArpEntry>,
    /// Default timeout in seconds
    default_timeout: u64,
    /// Maximum table size
    max_size: usize,
}

impl ArpTable {
    /// Create a new ARP table
    pub fn new(default_timeout: u64, max_size: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            default_timeout,
            max_size,
        }
    }

    /// Add or update an entry
    pub fn insert(&mut self, ip: Ipv4Address, mac: MacAddress, timestamp: u64, is_static: bool) {
        // Remove old entries if table is full
        if self.entries.len() >= self.max_size && !self.entries.contains_key(&ip) {
            self.cleanup_expired_entries(timestamp);

            // If still full, remove oldest non-static entry
            if self.entries.len() >= self.max_size {
                if let Some((oldest_ip, _)) = self
                    .entries
                    .iter()
                    .filter(|(_, entry)| !entry.is_static)
                    .min_by_key(|(_, entry)| entry.timestamp)
                    .map(|(ip, entry)| (*ip, entry.clone()))
                {
                    self.entries.remove(&oldest_ip);
                }
            }
        }

        let entry = ArpEntry::new(mac, timestamp, is_static);
        self.entries.insert(ip, entry);
    }

    /// Lookup MAC address for IP
    pub fn lookup(&self, ip: Ipv4Address, current_time: u64) -> Option<MacAddress> {
        if let Some(entry) = self.entries.get(&ip) {
            if !entry.has_expired(current_time, self.default_timeout) {
                return Some(entry.mac_address);
            }
        }
        None
    }

    /// Remove an entry
    pub fn remove(&mut self, ip: Ipv4Address) -> Option<ArpEntry> {
        self.entries.remove(&ip)
    }

    /// Clean up expired entries
    pub fn cleanup_expired_entries(&mut self, current_time: u64) {
        let expired_ips: Vec<Ipv4Address> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.has_expired(current_time, self.default_timeout))
            .map(|(ip, _)| *ip)
            .collect();

        for ip in expired_ips {
            self.entries.remove(&ip);
        }
    }

    /// Get all entries
    pub fn entries(&self) -> impl Iterator<Item = (&Ipv4Address, &ArpEntry)> {
        self.entries.iter()
    }

    /// Get table size
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Clear non-static entries
    pub fn clear_dynamic(&mut self) {
        self.entries.retain(|_, entry| entry.is_static);
    }
}

/// ARP processor for handling ARP operations
pub struct ArpProcessor {
    /// ARP table
    table: ArpTable,
    /// Our MAC address
    our_mac: MacAddress,
    /// Our IP address
    our_ip: Option<Ipv4Address>,
}

impl ArpProcessor {
    /// Create a new ARP processor
    pub fn new(our_mac: MacAddress, table_size: usize, timeout_seconds: u64) -> Self {
        Self {
            table: ArpTable::new(timeout_seconds, table_size),
            our_mac,
            our_ip: None,
        }
    }

    /// Set our IP address
    pub fn set_our_ip(&mut self, ip: Ipv4Address) {
        self.our_ip = Some(ip);
        // Add ourselves to the table as a static entry
        self.table.insert(ip, self.our_mac, 0, true);
    }

    /// Process incoming ARP packet
    pub fn process_packet(
        &mut self,
        packet: &ArpPacket,
        timestamp: u64,
    ) -> Result<Option<ArpPacket>, NetworkError> {
        // Validate packet
        packet.validate()?;

        // Update ARP table with sender information
        self.table.insert(
            packet.sender_protocol_addr,
            packet.sender_hardware_addr,
            timestamp,
            false,
        );

        match packet.operation {
            ArpOperation::Request => {
                if let Some(our_ip) = self.our_ip {
                    if packet.is_request_for(our_ip) {
                        // Create ARP response
                        let response = ArpPacket::new_response(
                            self.our_mac,
                            our_ip,
                            packet.sender_hardware_addr,
                            packet.sender_protocol_addr,
                        );
                        return Ok(Some(response));
                    }
                }
            }
            ArpOperation::Response => {
                // Response is already processed by updating the table
                // No action needed
            }
            _ => {
                // Unsupported operation
                return Err(NetworkError::ProtocolError);
            }
        }

        Ok(None)
    }

    /// Create ARP request for IP address
    pub fn create_request(&self, target_ip: Ipv4Address) -> Result<ArpPacket, NetworkError> {
        if let Some(our_ip) = self.our_ip {
            Ok(ArpPacket::new_request(self.our_mac, our_ip, target_ip))
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Resolve IP to MAC address
    pub fn resolve_ip(&mut self, ip: Ipv4Address, timestamp: u64) -> Option<MacAddress> {
        self.table.cleanup_expired_entries(timestamp);
        self.table.lookup(ip, timestamp)
    }

    /// Add static ARP entry
    pub fn add_static_entry(&mut self, ip: Ipv4Address, mac: MacAddress, timestamp: u64) {
        self.table.insert(ip, mac, timestamp, true);
    }

    /// Remove ARP entry
    pub fn remove_entry(&mut self, ip: Ipv4Address) -> Option<ArpEntry> {
        self.table.remove(ip)
    }

    /// Get ARP table statistics
    pub fn get_stats(&self) -> ArpStats {
        ArpStats {
            total_entries: self.table.size(),
            static_entries: self
                .table
                .entries()
                .filter(|(_, entry)| entry.is_static)
                .count(),
            dynamic_entries: self
                .table
                .entries()
                .filter(|(_, entry)| !entry.is_static)
                .count(),
        }
    }

    /// Clear ARP table
    pub fn clear_table(&mut self) {
        self.table.clear_dynamic();
    }

    /// Get all ARP entries
    pub fn get_entries(&self) -> Vec<(Ipv4Address, MacAddress, bool)> {
        self.table
            .entries()
            .map(|(ip, entry)| (*ip, entry.mac_address, entry.is_static))
            .collect()
    }
}

/// ARP statistics
#[derive(Debug, Clone, Default)]
pub struct ArpStats {
    pub total_entries: usize,
    pub static_entries: usize,
    pub dynamic_entries: usize,
}

/// High-level ARP functions
pub fn create_arp_request_packet(
    our_mac: MacAddress,
    our_ip: Ipv4Address,
    target_ip: Ipv4Address,
) -> Vec<u8> {
    let packet = ArpPacket::new_request(our_mac, our_ip, target_ip);
    packet.to_bytes()
}

pub fn parse_arp_packet(data: &[u8]) -> Result<ArpPacket, NetworkError> {
    ArpPacket::from_bytes(data)
}

pub fn is_arp_request_for_ip(data: &[u8], our_ip: Ipv4Address) -> Result<bool, NetworkError> {
    let packet = ArpPacket::from_bytes(data)?;
    Ok(packet.is_request_for(our_ip))
}

pub fn create_arp_response_packet(
    our_mac: MacAddress,
    our_ip: Ipv4Address,
    target_mac: MacAddress,
    target_ip: Ipv4Address,
) -> Vec<u8> {
    let packet = ArpPacket::new_response(our_mac, our_ip, target_mac, target_ip);
    packet.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_arp_packet_creation() {
        let sender_mac = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let sender_ip = Ipv4Address::new(192, 168, 1, 100);
        let target_ip = Ipv4Address::new(192, 168, 1, 1);

        let packet = ArpPacket::new_request(sender_mac, sender_ip, target_ip);

        assert_eq!(packet.operation, ArpOperation::Request);
        assert_eq!(packet.sender_hardware_addr, sender_mac);
        assert_eq!(packet.sender_protocol_addr, sender_ip);
        assert_eq!(packet.target_protocol_addr, target_ip);
        assert_eq!(packet.target_hardware_addr, MacAddress::ZERO);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_arp_packet_serialization() {
        let sender_mac = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let sender_ip = Ipv4Address::new(192, 168, 1, 100);
        let target_ip = Ipv4Address::new(192, 168, 1, 1);

        let packet = ArpPacket::new_request(sender_mac, sender_ip, target_ip);
        let bytes = packet.to_bytes();
        let parsed_packet = ArpPacket::from_bytes(&bytes).unwrap();

        assert_eq!(parsed_packet.operation, packet.operation);
        assert_eq!(
            parsed_packet.sender_hardware_addr,
            packet.sender_hardware_addr
        );
        assert_eq!(
            parsed_packet.sender_protocol_addr,
            packet.sender_protocol_addr
        );
        assert_eq!(
            parsed_packet.target_protocol_addr,
            packet.target_protocol_addr
        );
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_arp_table() {
        let mut table = ArpTable::new(300, 100);
        let ip = Ipv4Address::new(192, 168, 1, 1);
        let mac = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);

        // Insert entry
        table.insert(ip, mac, 1000, false);
        assert_eq!(table.lookup(ip, 1100), Some(mac));

        // Test expiration
        assert_eq!(table.lookup(ip, 2000), None);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_arp_processor() {
        let our_mac = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let our_ip = Ipv4Address::new(192, 168, 1, 100);
        let mut processor = ArpProcessor::new(our_mac, 100, 300);
        processor.set_our_ip(our_ip);

        let sender_mac = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let sender_ip = Ipv4Address::new(192, 168, 1, 50);

        let request = ArpPacket::new_request(sender_mac, sender_ip, our_ip);
        let response = processor.process_packet(&request, 1000).unwrap();

        assert!(response.is_some());
        let response = response.unwrap();
        assert_eq!(response.operation, ArpOperation::Response);
        assert_eq!(response.sender_hardware_addr, our_mac);
        assert_eq!(response.target_hardware_addr, sender_mac);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_arp_operations() {
        assert_eq!(u16::from(ArpOperation::Request), 1);
        assert_eq!(u16::from(ArpOperation::Response), 2);
        assert_eq!(ArpOperation::from(1u16), ArpOperation::Request);
        assert_eq!(ArpOperation::from(2u16), ArpOperation::Response);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_hardware_types() {
        assert_eq!(u16::from(ArpHardwareType::Ethernet), 1);
        assert_eq!(ArpHardwareType::from(1u16), ArpHardwareType::Ethernet);
    }
}
