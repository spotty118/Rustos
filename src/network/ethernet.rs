//! # Ethernet Protocol Implementation
//!
//! This module provides Ethernet (IEEE 802.3) frame handling capabilities
//! for the RustOS network stack, including frame parsing, construction,
//! and validation.

use crate::Vec;
// use core::mem::size_of;

use crate::network::{MacAddress, NetworkError};

/// Ethernet frame minimum size (64 bytes including FCS)
pub const ETHERNET_MIN_FRAME_SIZE: usize = 64;

/// Ethernet frame maximum size (1518 bytes including FCS)
pub const ETHERNET_MAX_FRAME_SIZE: usize = 1518;

/// Ethernet header size (14 bytes)
pub const ETHERNET_HEADER_SIZE: usize = 14;

/// Frame Check Sequence size (4 bytes)
pub const ETHERNET_FCS_SIZE: usize = 4;

/// Maximum Transmission Unit for Ethernet
pub const ETHERNET_MTU: usize = 1500;

/// Common EtherType values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum EtherType {
    /// Internet Protocol version 4 (IPv4)
    Ipv4 = 0x0800,
    /// Address Resolution Protocol (ARP)
    Arp = 0x0806,
    /// Internet Protocol version 6 (IPv6)
    Ipv6 = 0x86DD,
    /// IEEE 802.1Q VLAN-tagged frame
    Vlan = 0x8100,
    /// Point-to-Point Protocol over Ethernet
    Pppoe = 0x8864,
    /// Link Layer Discovery Protocol
    Lldp = 0x88CC,
    /// Wake-on-LAN
    Wol = 0x0842,
}

impl From<u16> for EtherType {
    fn from(value: u16) -> Self {
        match value {
            0x0800 => EtherType::Ipv4,
            0x0806 => EtherType::Arp,
            0x86DD => EtherType::Ipv6,
            0x8100 => EtherType::Vlan,
            0x8864 => EtherType::Pppoe,
            0x88CC => EtherType::Lldp,
            0x0842 => EtherType::Wol,
            _ => EtherType::Ipv4, // Default fallback
        }
    }
}

impl From<EtherType> for u16 {
    fn from(ether_type: EtherType) -> Self {
        ether_type as u16
    }
}

/// Ethernet frame header structure
#[derive(Debug, Clone)]
pub struct EthernetHeader {
    /// Destination MAC address
    pub destination: MacAddress,
    /// Source MAC address
    pub source: MacAddress,
    /// EtherType or length field
    pub ether_type: EtherType,
}

impl EthernetHeader {
    /// Create a new Ethernet header
    pub fn new(destination: MacAddress, source: MacAddress, ether_type: EtherType) -> Self {
        Self {
            destination,
            source,
            ether_type,
        }
    }

    /// Parse Ethernet header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        if data.len() < ETHERNET_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let destination = MacAddress::new([data[0], data[1], data[2], data[3], data[4], data[5]]);

        let source = MacAddress::new([data[6], data[7], data[8], data[9], data[10], data[11]]);

        let ether_type_raw = u16::from_be_bytes([data[12], data[13]]);
        let ether_type = EtherType::from(ether_type_raw);

        Ok(Self {
            destination,
            source,
            ether_type,
        })
    }

    /// Convert header to bytes
    pub fn to_bytes(&self) -> [u8; ETHERNET_HEADER_SIZE] {
        let mut bytes = [0u8; ETHERNET_HEADER_SIZE];

        // Destination MAC address
        bytes[0..6].copy_from_slice(&self.destination.0);

        // Source MAC address
        bytes[6..12].copy_from_slice(&self.source.0);

        // EtherType
        let ether_type_bytes = u16::to_be_bytes(self.ether_type.into());
        bytes[12] = ether_type_bytes[0];
        bytes[13] = ether_type_bytes[1];

        bytes
    }

    /// Check if this frame is addressed to us
    pub fn is_for_address(&self, our_mac: MacAddress) -> bool {
        self.destination == our_mac
            || self.destination.is_broadcast()
            || (self.destination.is_multicast() && self.is_multicast_we_care_about())
    }

    /// Check if this is a multicast we care about (simplified check)
    fn is_multicast_we_care_about(&self) -> bool {
        // For now, accept all multicast frames
        // In a real implementation, we'd maintain a multicast filter
        true
    }
}

/// Complete Ethernet frame structure
#[derive(Debug, Clone)]
pub struct EthernetFrame {
    /// Ethernet header
    pub header: EthernetHeader,
    /// Frame payload
    pub payload: Vec<u8>,
}

impl EthernetFrame {
    /// Create a new Ethernet frame
    pub fn new(
        destination: MacAddress,
        source: MacAddress,
        ether_type: EtherType,
        payload: Vec<u8>,
    ) -> Result<Self, NetworkError> {
        // Check payload size limits
        if payload.len() > ETHERNET_MTU {
            return Err(NetworkError::BufferTooSmall);
        }

        let header = EthernetHeader::new(destination, source, ether_type);

        Ok(Self { header, payload })
    }

    /// Parse Ethernet frame from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, NetworkError> {
        if data.len() < ETHERNET_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let header = EthernetHeader::from_bytes(&data[..ETHERNET_HEADER_SIZE])?;
        let payload = data[ETHERNET_HEADER_SIZE..].to_vec();

        Ok(Self { header, payload })
    }

    /// Convert frame to bytes (without FCS)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(ETHERNET_HEADER_SIZE + self.payload.len());

        // Add header
        bytes.extend_from_slice(&self.header.to_bytes());

        // Add payload
        bytes.extend_from_slice(&self.payload);

        // Pad frame if too small
        let min_payload_size = ETHERNET_MIN_FRAME_SIZE - ETHERNET_HEADER_SIZE - ETHERNET_FCS_SIZE;
        if bytes.len() - ETHERNET_HEADER_SIZE < min_payload_size {
            let padding_needed = min_payload_size - (bytes.len() - ETHERNET_HEADER_SIZE);
            bytes.resize(bytes.len() + padding_needed, 0);
        }

        bytes
    }

    /// Get the total frame size including header
    pub fn size(&self) -> usize {
        ETHERNET_HEADER_SIZE + self.payload.len()
    }

    /// Check if frame size is valid
    pub fn is_valid_size(&self) -> bool {
        let size = self.size();
        size >= (ETHERNET_MIN_FRAME_SIZE - ETHERNET_FCS_SIZE)
            && size <= (ETHERNET_MAX_FRAME_SIZE - ETHERNET_FCS_SIZE)
    }

    /// Validate frame integrity
    pub fn validate(&self) -> Result<(), NetworkError> {
        // Check frame size
        if !self.is_valid_size() {
            return Err(NetworkError::InvalidPacket);
        }

        // Check for valid MAC addresses (not all zeros for source)
        if self.header.source == MacAddress::ZERO {
            return Err(NetworkError::InvalidAddress);
        }

        // Check payload size based on EtherType
        match self.header.ether_type {
            EtherType::Ipv4 => {
                if self.payload.len() < 20 {
                    return Err(NetworkError::InvalidPacket);
                }
            }
            EtherType::Ipv6 => {
                if self.payload.len() < 40 {
                    return Err(NetworkError::InvalidPacket);
                }
            }
            EtherType::Arp => {
                if self.payload.len() < 28 {
                    return Err(NetworkError::InvalidPacket);
                }
            }
            _ => {} // Other protocols have variable sizes
        }

        Ok(())
    }

    /// Create an ARP frame
    pub fn create_arp_frame(
        source_mac: MacAddress,
        destination_mac: MacAddress,
        payload: Vec<u8>,
    ) -> Result<Self, NetworkError> {
        Self::new(destination_mac, source_mac, EtherType::Arp, payload)
    }

    /// Create an IPv4 frame
    pub fn create_ipv4_frame(
        source_mac: MacAddress,
        destination_mac: MacAddress,
        payload: Vec<u8>,
    ) -> Result<Self, NetworkError> {
        Self::new(destination_mac, source_mac, EtherType::Ipv4, payload)
    }

    /// Create an IPv6 frame
    pub fn create_ipv6_frame(
        source_mac: MacAddress,
        destination_mac: MacAddress,
        payload: Vec<u8>,
    ) -> Result<Self, NetworkError> {
        Self::new(destination_mac, source_mac, EtherType::Ipv6, payload)
    }

    /// Create a broadcast frame
    pub fn create_broadcast_frame(
        source_mac: MacAddress,
        ether_type: EtherType,
        payload: Vec<u8>,
    ) -> Result<Self, NetworkError> {
        Self::new(MacAddress::BROADCAST, source_mac, ether_type, payload)
    }
}

/// Ethernet frame processing utilities
pub struct EthernetProcessor;

impl EthernetProcessor {
    /// Process incoming Ethernet frame
    pub fn process_incoming_frame(
        data: &[u8],
        our_mac: MacAddress,
    ) -> Result<EthernetFrame, NetworkError> {
        let frame = EthernetFrame::from_bytes(data)?;

        // Validate frame
        frame.validate()?;

        // Check if frame is addressed to us
        if !frame.header.is_for_address(our_mac) {
            return Err(NetworkError::InvalidAddress);
        }

        Ok(frame)
    }

    /// Prepare outgoing Ethernet frame
    pub fn prepare_outgoing_frame(
        destination_mac: MacAddress,
        source_mac: MacAddress,
        ether_type: EtherType,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, NetworkError> {
        let frame = EthernetFrame::new(destination_mac, source_mac, ether_type, payload)?;
        frame.validate()?;
        Ok(frame.to_bytes())
    }

    /// Calculate simple checksum (for testing purposes)
    pub fn calculate_checksum(data: &[u8]) -> u32 {
        let mut checksum: u32 = 0;
        for byte in data {
            checksum = checksum.wrapping_add(*byte as u32);
        }
        checksum
    }

    /// Verify frame integrity with basic checks
    pub fn verify_frame_integrity(data: &[u8]) -> bool {
        if data.len() < ETHERNET_MIN_FRAME_SIZE - ETHERNET_FCS_SIZE {
            return false;
        }

        if data.len() > ETHERNET_MAX_FRAME_SIZE - ETHERNET_FCS_SIZE {
            return false;
        }

        // Basic header validation
        if data.len() < ETHERNET_HEADER_SIZE {
            return false;
        }

        // Check for valid source MAC (not all zeros)
        let source_mac = &data[6..12];
        if source_mac.iter().all(|&b| b == 0) {
            return false;
        }

        true
    }

    /// Extract EtherType from raw frame data
    pub fn extract_ether_type(data: &[u8]) -> Result<EtherType, NetworkError> {
        if data.len() < ETHERNET_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let ether_type_raw = u16::from_be_bytes([data[12], data[13]]);
        Ok(EtherType::from(ether_type_raw))
    }

    /// Extract payload from raw frame data
    pub fn extract_payload(data: &[u8]) -> Result<&[u8], NetworkError> {
        if data.len() < ETHERNET_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        Ok(&data[ETHERNET_HEADER_SIZE..])
    }
}

/// Ethernet statistics
#[derive(Debug, Default, Clone)]
pub struct EthernetStats {
    pub frames_sent: u64,
    pub frames_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub broadcast_frames_sent: u64,
    pub broadcast_frames_received: u64,
    pub multicast_frames_sent: u64,
    pub multicast_frames_received: u64,
    pub invalid_frames: u64,
    pub oversized_frames: u64,
    pub undersized_frames: u64,
}

impl EthernetStats {
    /// Update stats for sent frame
    pub fn update_sent(&mut self, frame: &EthernetFrame) {
        self.frames_sent += 1;
        self.bytes_sent += frame.size() as u64;

        if frame.header.destination.is_broadcast() {
            self.broadcast_frames_sent += 1;
        } else if frame.header.destination.is_multicast() {
            self.multicast_frames_sent += 1;
        }
    }

    /// Update stats for received frame
    pub fn update_received(&mut self, frame: &EthernetFrame, valid: bool) {
        if valid {
            self.frames_received += 1;
            self.bytes_received += frame.size() as u64;

            if frame.header.destination.is_broadcast() {
                self.broadcast_frames_received += 1;
            } else if frame.header.destination.is_multicast() {
                self.multicast_frames_received += 1;
            }
        } else {
            self.invalid_frames += 1;
        }

        if frame.size() > ETHERNET_MAX_FRAME_SIZE {
            self.oversized_frames += 1;
        } else if frame.size() < ETHERNET_MIN_FRAME_SIZE {
            self.undersized_frames += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ethernet_header_creation() {
        let dest = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let header = EthernetHeader::new(dest, src, EtherType::Ipv4);

        assert_eq!(header.destination, dest);
        assert_eq!(header.source, src);
        assert_eq!(header.ether_type, EtherType::Ipv4);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ethernet_header_serialization() {
        let dest = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let header = EthernetHeader::new(dest, src, EtherType::Ipv4);

        let bytes = header.to_bytes();
        let parsed_header = EthernetHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed_header.destination, dest);
        assert_eq!(parsed_header.source, src);
        assert_eq!(parsed_header.ether_type, EtherType::Ipv4);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ethernet_frame_creation() {
        let dest = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let payload = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello"

        let frame = EthernetFrame::new(dest, src, EtherType::Ipv4, payload.clone()).unwrap();

        assert_eq!(frame.header.destination, dest);
        assert_eq!(frame.header.source, src);
        assert_eq!(frame.header.ether_type, EtherType::Ipv4);
        assert_eq!(frame.payload, payload);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_frame_validation() {
        let dest = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let payload = vec![0; 46]; // Minimum payload size

        let frame = EthernetFrame::new(dest, src, EtherType::Ipv4, payload).unwrap();
        assert!(frame.validate().is_ok());
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_broadcast_frame() {
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let payload = vec![0; 46];

        let frame = EthernetFrame::create_broadcast_frame(src, EtherType::Arp, payload).unwrap();
        assert_eq!(frame.header.destination, MacAddress::BROADCAST);
        assert_eq!(frame.header.ether_type, EtherType::Arp);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_ether_type_conversion() {
        assert_eq!(u16::from(EtherType::Ipv4), 0x0800);
        assert_eq!(u16::from(EtherType::Arp), 0x0806);
        assert_eq!(u16::from(EtherType::Ipv6), 0x86DD);

        assert_eq!(EtherType::from(0x0800u16), EtherType::Ipv4);
        assert_eq!(EtherType::from(0x0806u16), EtherType::Arp);
        assert_eq!(EtherType::from(0x86DDu16), EtherType::Ipv6);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[test]
    fn test_frame_processor() {
        let dest = MacAddress::new([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
        let src = MacAddress::new([0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]);
        let payload = vec![0; 46];

        let frame_bytes =
            EthernetProcessor::prepare_outgoing_frame(dest, src, EtherType::Ipv4, payload).unwrap();

        assert!(EthernetProcessor::verify_frame_integrity(&frame_bytes));
        assert_eq!(
            EthernetProcessor::extract_ether_type(&frame_bytes).unwrap(),
            EtherType::Ipv4
        );
    }
}
