//! IP packet processing (IPv4 and IPv6)
//!
//! This module handles Internet Protocol packet parsing, routing, and forwarding.

use super::{NetworkAddress, NetworkResult, NetworkError, PacketBuffer, NetworkStack, Protocol};
use alloc::vec::Vec;

/// IPv4 header minimum size
pub const IPV4_HEADER_MIN_SIZE: usize = 20;

/// IPv6 header size
pub const IPV6_HEADER_SIZE: usize = 40;

/// IPv4 header
#[derive(Debug, Clone)]
pub struct IPv4Header {
    /// Version (4 bits) and Header Length (4 bits)
    pub version_ihl: u8,
    /// Type of Service
    pub tos: u8,
    /// Total Length
    pub total_length: u16,
    /// Identification
    pub identification: u16,
    /// Flags (3 bits) and Fragment Offset (13 bits)
    pub flags_fragment: u16,
    /// Time to Live
    pub ttl: u8,
    /// Protocol
    pub protocol: u8,
    /// Header Checksum
    pub checksum: u16,
    /// Source Address
    pub source: NetworkAddress,
    /// Destination Address
    pub destination: NetworkAddress,
    /// Options (variable length)
    pub options: Vec<u8>,
}

impl IPv4Header {
    /// Parse IPv4 header from packet buffer
    pub fn parse(buffer: &mut PacketBuffer) -> NetworkResult<Self> {
        if buffer.remaining() < IPV4_HEADER_MIN_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let version_ihl = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];
        let version = (version_ihl >> 4) & 0x0f;
        let ihl = version_ihl & 0x0f;

        if version != 4 {
            return Err(NetworkError::InvalidPacket);
        }

        let header_length = (ihl as usize) * 4;
        if header_length < IPV4_HEADER_MIN_SIZE || buffer.remaining() + 1 < header_length {
            return Err(NetworkError::InvalidPacket);
        }

        let tos = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];
        
        let total_length_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let total_length = u16::from_be_bytes([total_length_bytes[0], total_length_bytes[1]]);

        let id_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let identification = u16::from_be_bytes([id_bytes[0], id_bytes[1]]);

        let flags_frag_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let flags_fragment = u16::from_be_bytes([flags_frag_bytes[0], flags_frag_bytes[1]]);

        let ttl = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];
        let protocol = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];

        let checksum_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let checksum = u16::from_be_bytes([checksum_bytes[0], checksum_bytes[1]]);

        let src_bytes = buffer.read(4).ok_or(NetworkError::InvalidPacket)?;
        let mut src_addr = [0u8; 4];
        src_addr.copy_from_slice(src_bytes);
        let source = NetworkAddress::IPv4(src_addr);

        let dst_bytes = buffer.read(4).ok_or(NetworkError::InvalidPacket)?;
        let mut dst_addr = [0u8; 4];
        dst_addr.copy_from_slice(dst_bytes);
        let destination = NetworkAddress::IPv4(dst_addr);

        // Read options if present
        let options_length = header_length - IPV4_HEADER_MIN_SIZE;
        let options = if options_length > 0 {
            let options_bytes = buffer.read(options_length).ok_or(NetworkError::InvalidPacket)?;
            options_bytes.to_vec()
        } else {
            Vec::new()
        };

        Ok(IPv4Header {
            version_ihl,
            tos,
            total_length,
            identification,
            flags_fragment,
            ttl,
            protocol,
            checksum,
            source,
            destination,
            options,
        })
    }

    /// Calculate header checksum
    pub fn calculate_checksum(&self) -> u16 {
        let mut sum = 0u32;
        
        // Add all 16-bit words in header (except checksum field)
        sum += (self.version_ihl as u32) << 8 | (self.tos as u32);
        sum += self.total_length as u32;
        sum += self.identification as u32;
        sum += self.flags_fragment as u32;
        sum += (self.ttl as u32) << 8 | (self.protocol as u32);
        // Skip checksum field
        
        if let NetworkAddress::IPv4(src) = self.source {
            sum += ((src[0] as u32) << 8) | (src[1] as u32);
            sum += ((src[2] as u32) << 8) | (src[3] as u32);
        }
        
        if let NetworkAddress::IPv4(dst) = self.destination {
            sum += ((dst[0] as u32) << 8) | (dst[1] as u32);
            sum += ((dst[2] as u32) << 8) | (dst[3] as u32);
        }

        // Add options
        for chunk in self.options.chunks(2) {
            if chunk.len() == 2 {
                sum += ((chunk[0] as u32) << 8) | (chunk[1] as u32);
            } else {
                sum += (chunk[0] as u32) << 8;
            }
        }

        // Fold 32-bit sum to 16 bits
        while (sum >> 16) != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        // One's complement
        !sum as u16
    }

    /// Check if packet is fragmented
    pub fn is_fragmented(&self) -> bool {
        let more_fragments = (self.flags_fragment & 0x2000) != 0;
        let fragment_offset = self.flags_fragment & 0x1FFF;
        more_fragments || fragment_offset != 0
    }
}

/// IPv6 header
#[derive(Debug, Clone)]
pub struct IPv6Header {
    /// Version (4 bits), Traffic Class (8 bits), Flow Label (20 bits)
    pub version_tc_fl: u32,
    /// Payload Length
    pub payload_length: u16,
    /// Next Header
    pub next_header: u8,
    /// Hop Limit
    pub hop_limit: u8,
    /// Source Address
    pub source: NetworkAddress,
    /// Destination Address
    pub destination: NetworkAddress,
}

impl IPv6Header {
    /// Parse IPv6 header from packet buffer
    pub fn parse(buffer: &mut PacketBuffer) -> NetworkResult<Self> {
        if buffer.remaining() < IPV6_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let vtf_bytes = buffer.read(4).ok_or(NetworkError::InvalidPacket)?;
        let version_tc_fl = u32::from_be_bytes([vtf_bytes[0], vtf_bytes[1], vtf_bytes[2], vtf_bytes[3]]);
        
        let version = (version_tc_fl >> 28) & 0x0f;
        if version != 6 {
            return Err(NetworkError::InvalidPacket);
        }

        let pl_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let payload_length = u16::from_be_bytes([pl_bytes[0], pl_bytes[1]]);

        let next_header = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];
        let hop_limit = buffer.read(1).ok_or(NetworkError::InvalidPacket)?[0];

        let src_bytes = buffer.read(16).ok_or(NetworkError::InvalidPacket)?;
        let mut src_addr = [0u8; 16];
        src_addr.copy_from_slice(src_bytes);
        let source = NetworkAddress::IPv6(src_addr);

        let dst_bytes = buffer.read(16).ok_or(NetworkError::InvalidPacket)?;
        let mut dst_addr = [0u8; 16];
        dst_addr.copy_from_slice(dst_bytes);
        let destination = NetworkAddress::IPv6(dst_addr);

        Ok(IPv6Header {
            version_tc_fl,
            payload_length,
            next_header,
            hop_limit,
            source,
            destination,
        })
    }
}

/// Process IPv4 packet
pub fn process_ipv4_packet(network_stack: &NetworkStack, mut packet: PacketBuffer) -> NetworkResult<()> {
    let header = IPv4Header::parse(&mut packet)?;
    
    println!("IPv4 packet: {} -> {} (proto: {}, len: {})",
        header.source, header.destination, header.protocol, header.total_length);

    // Verify checksum
    let calculated_checksum = header.calculate_checksum();
    if calculated_checksum != header.checksum {
        println!(
            "IPv4 checksum mismatch: calculated 0x{:04x}, header 0x{:04x}",
            calculated_checksum,
            header.checksum
        );
        return Err(NetworkError::InvalidPacket);
    }

    // Check if packet is for us
    if !is_packet_for_us(&header.destination) {
        // Forward packet if we're a router
        return forward_ipv4_packet(network_stack, header, packet);
    }

    // Handle fragmentation
    if header.is_fragmented() {
        println!("Fragmented packets not yet supported");
        return Ok(());
    }

    // Process based on protocol
    let protocol = Protocol::from(header.protocol);
    match protocol {
        Protocol::ICMP => {
            process_icmp_packet(network_stack, header, packet)
        }
        Protocol::TCP => {
            super::tcp::process_packet(network_stack, header.source, header.destination, packet)
        }
        Protocol::UDP => {
            super::udp::process_packet(network_stack, header.source, header.destination, packet)
        }
        _ => {
            println!("Unsupported IP protocol: {}", header.protocol);
            Ok(())
        }
    }
}

/// Process IPv6 packet
pub fn process_ipv6_packet(network_stack: &NetworkStack, mut packet: PacketBuffer) -> NetworkResult<()> {
    let header = IPv6Header::parse(&mut packet)?;
    
    println!("IPv6 packet: {} -> {} (next: {}, len: {})",
        header.source, header.destination, header.next_header, header.payload_length);

    // Check if packet is for us
    if !is_packet_for_us(&header.destination) {
        // Forward packet if we're a router
        return forward_ipv6_packet(network_stack, header, packet);
    }

    // Process based on next header
    match header.next_header {
        58 => { // ICMPv6
            process_icmpv6_packet(network_stack, header, packet)
        }
        6 => { // TCP
            super::tcp::process_packet(network_stack, header.source, header.destination, packet)
        }
        17 => { // UDP
            super::udp::process_packet(network_stack, header.source, header.destination, packet)
        }
        _ => {
            println!("Unsupported IPv6 next header: {}", header.next_header);
            Ok(())
        }
    }
}

/// Check if packet is destined for us
fn is_packet_for_us(destination: &NetworkAddress) -> bool {
    // Check against all interface addresses
    let interfaces = super::network_stack().list_interfaces();
    for interface in interfaces {
        if interface.ip_addresses.contains(destination) {
            return true;
        }
    }

    // Check for broadcast/multicast
    match destination {
        NetworkAddress::IPv4([255, 255, 255, 255]) => true, // Broadcast
        NetworkAddress::IPv4([a, _, _, _]) if (*a & 0xf0) == 0xe0 => true, // Multicast
        NetworkAddress::IPv6([0xff, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _]) => true, // Multicast
        _ => false,
    }
}

/// Forward IPv4 packet
fn forward_ipv4_packet(
    network_stack: &NetworkStack,
    mut header: IPv4Header,
    packet: PacketBuffer,
) -> NetworkResult<()> {
    // Decrement TTL
    if header.ttl <= 1 {
        // Send ICMP Time Exceeded
        println!("TTL expired, dropping packet");
        return Ok(());
    }
    header.ttl -= 1;

    // Find route to destination
    if let Some(route) = network_stack.find_route(&header.destination) {
        println!("Forwarding packet to {} via {}", header.destination, route.interface);
        // TODO: Implement actual packet forwarding
        Ok(())
    } else {
        println!("No route to {}", header.destination);
        // Send ICMP Destination Unreachable
        Ok(())
    }
}

/// Forward IPv6 packet
fn forward_ipv6_packet(
    network_stack: &NetworkStack,
    mut header: IPv6Header,
    packet: PacketBuffer,
) -> NetworkResult<()> {
    // Decrement hop limit
    if header.hop_limit <= 1 {
        // Send ICMPv6 Time Exceeded
        println!("Hop limit expired, dropping packet");
        return Ok(());
    }
    header.hop_limit -= 1;

    // Find route to destination
    if let Some(route) = network_stack.find_route(&header.destination) {
        println!("Forwarding IPv6 packet to {} via {}", header.destination, route.interface);
        // TODO: Implement actual packet forwarding
        Ok(())
    } else {
        println!("No route to {}", header.destination);
        // Send ICMPv6 Destination Unreachable
        Ok(())
    }
}

/// Process ICMP packet
fn process_icmp_packet(
    _network_stack: &NetworkStack,
    ip_header: IPv4Header,
    mut packet: PacketBuffer,
) -> NetworkResult<()> {
    if packet.remaining() < 8 {
        return Err(NetworkError::InvalidPacket);
    }

    let icmp_type = packet.read(1).ok_or(NetworkError::InvalidPacket)?[0];
    let icmp_code = packet.read(1).ok_or(NetworkError::InvalidPacket)?[0];
    let checksum_bytes = packet.read(2).ok_or(NetworkError::InvalidPacket)?;
    let _checksum = u16::from_be_bytes([checksum_bytes[0], checksum_bytes[1]]);
    let rest_bytes = packet.read(4).ok_or(NetworkError::InvalidPacket)?;

    println!("ICMP packet: type={}, code={} from {}", icmp_type, icmp_code, ip_header.source);

    match icmp_type {
        8 => {
            // Echo Request (ping)
            println!("ICMP Echo Request from {}", ip_header.source);
            // TODO: Send Echo Reply
        }
        0 => {
            // Echo Reply
            println!("ICMP Echo Reply from {}", ip_header.source);
        }
        3 => {
            // Destination Unreachable
            println!("ICMP Destination Unreachable from {} (code: {})", ip_header.source, icmp_code);
        }
        11 => {
            // Time Exceeded
            println!("ICMP Time Exceeded from {} (code: {})", ip_header.source, icmp_code);
        }
        _ => {
            println!("Unknown ICMP type: {} from {}", icmp_type, ip_header.source);
        }
    }

    Ok(())
}

/// Process ICMPv6 packet
fn process_icmpv6_packet(
    _network_stack: &NetworkStack,
    ip_header: IPv6Header,
    mut packet: PacketBuffer,
) -> NetworkResult<()> {
    if packet.remaining() < 4 {
        return Err(NetworkError::InvalidPacket);
    }

    let icmp_type = packet.read(1).ok_or(NetworkError::InvalidPacket)?[0];
    let icmp_code = packet.read(1).ok_or(NetworkError::InvalidPacket)?[0];
    let checksum_bytes = packet.read(2).ok_or(NetworkError::InvalidPacket)?;
    let _checksum = u16::from_be_bytes([checksum_bytes[0], checksum_bytes[1]]);

    println!("ICMPv6 packet: type={}, code={} from {}", icmp_type, icmp_code, ip_header.source);

    match icmp_type {
        128 => {
            // Echo Request
            println!("ICMPv6 Echo Request from {}", ip_header.source);
            // TODO: Send Echo Reply
        }
        129 => {
            // Echo Reply
            println!("ICMPv6 Echo Reply from {}", ip_header.source);
        }
        1 => {
            // Destination Unreachable
            println!("ICMPv6 Destination Unreachable from {} (code: {})", ip_header.source, icmp_code);
        }
        3 => {
            // Time Exceeded
            println!("ICMPv6 Time Exceeded from {} (code: {})", ip_header.source, icmp_code);
        }
        135 => {
            // Neighbor Solicitation
            println!("ICMPv6 Neighbor Solicitation from {}", ip_header.source);
        }
        136 => {
            // Neighbor Advertisement
            println!("ICMPv6 Neighbor Advertisement from {}", ip_header.source);
        }
        _ => {
            println!("Unknown ICMPv6 type: {} from {}", icmp_type, ip_header.source);
        }
    }

    Ok(())
}

impl From<u8> for Protocol {
    fn from(value: u8) -> Self {
        match value {
            1 => Protocol::ICMP,
            6 => Protocol::TCP,
            17 => Protocol::UDP,
            41 => Protocol::IPv6inIPv4,
            47 => Protocol::GRE,
            58 => Protocol::ICMPv6,
            _ => Protocol::TCP, // Default fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_ipv4_packet_accepts_valid_checksum() {
        let network_stack = NetworkStack::new();

        let mut packet_bytes = vec![
            0x45, 0x00, 0x00, 0x1c, // version/IHL, TOS, total length (28)
            0x00, 0x00, // identification
            0x00, 0x00, // flags/fragment offset
            0x40, 0x01, // TTL, protocol (ICMP)
            0x00, 0x00, // checksum placeholder
            0xC0, 0x00, 0x02, 0x01, // source IP 192.0.2.1
            0xFF, 0xFF, 0xFF, 0xFF, // destination IP broadcast
        ];

        let known_checksum = 0xB8E0;
        packet_bytes[10] = (known_checksum >> 8) as u8;
        packet_bytes[11] = (known_checksum & 0xFF) as u8;

        // Minimal 8-byte ICMP payload to exercise the success path
        packet_bytes.extend_from_slice(&[0u8; 8]);

        let packet = PacketBuffer::from_data(packet_bytes);
        assert!(process_ipv4_packet(&network_stack, packet).is_ok());
    }
}
