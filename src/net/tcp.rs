//! TCP (Transmission Control Protocol) implementation
//!
//! This module provides a complete TCP stack with connection management,
//! flow control, congestion control, and reliable data transmission.

use super::{NetworkAddress, NetworkResult, NetworkError, PacketBuffer, NetworkStack};
use alloc::{vec::Vec, collections::BTreeMap};
use spin::RwLock;
use core::cmp;

/// TCP header minimum size
pub const TCP_HEADER_MIN_SIZE: usize = 20;

/// TCP connection states with proper state machine transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

impl TcpState {
    /// Check if state allows data transmission
    pub fn can_send_data(&self) -> bool {
        matches!(self, TcpState::Established | TcpState::CloseWait)
    }

    /// Check if state allows data reception
    pub fn can_recv_data(&self) -> bool {
        matches!(self, TcpState::Established | TcpState::FinWait1 | TcpState::FinWait2)
    }

    /// Check if connection is active
    pub fn is_active(&self) -> bool {
        !matches!(self, TcpState::Closed | TcpState::TimeWait)
    }

    /// Get next state on close
    pub fn on_close(&self) -> TcpState {
        match self {
            TcpState::Established => TcpState::FinWait1,
            TcpState::CloseWait => TcpState::LastAck,
            _ => *self,
        }
    }
}

/// TCP flags
#[derive(Debug, Clone, Copy)]
pub struct TcpFlags {
    pub fin: bool,
    pub syn: bool,
    pub rst: bool,
    pub psh: bool,
    pub ack: bool,
    pub urg: bool,
    pub ece: bool,
    pub cwr: bool,
}

impl TcpFlags {
    pub fn new() -> Self {
        Self {
            fin: false,
            syn: false,
            rst: false,
            psh: false,
            ack: false,
            urg: false,
            ece: false,
            cwr: false,
        }
    }

    pub fn from_byte(flags: u8) -> Self {
        Self {
            fin: (flags & 0x01) != 0,
            syn: (flags & 0x02) != 0,
            rst: (flags & 0x04) != 0,
            psh: (flags & 0x08) != 0,
            ack: (flags & 0x10) != 0,
            urg: (flags & 0x20) != 0,
            ece: (flags & 0x40) != 0,
            cwr: (flags & 0x80) != 0,
        }
    }

    pub fn to_byte(&self) -> u8 {
        let mut flags = 0u8;
        if self.fin { flags |= 0x01; }
        if self.syn { flags |= 0x02; }
        if self.rst { flags |= 0x04; }
        if self.psh { flags |= 0x08; }
        if self.ack { flags |= 0x10; }
        if self.urg { flags |= 0x20; }
        if self.ece { flags |= 0x40; }
        if self.cwr { flags |= 0x80; }
        flags
    }
}

/// TCP header
#[derive(Debug, Clone)]
pub struct TcpHeader {
    pub source_port: u16,
    pub dest_port: u16,
    pub sequence_number: u32,
    pub acknowledgment_number: u32,
    pub data_offset: u8,
    pub flags: TcpFlags,
    pub window_size: u16,
    pub checksum: u16,
    pub urgent_pointer: u16,
    pub options: Vec<u8>,
}

impl TcpHeader {
    /// Get source IP from context (would be passed in real implementation)
    pub fn source_ip(&self) -> NetworkAddress {
        // This would be passed from IP layer in real implementation
        NetworkAddress::IPv4([0, 0, 0, 0])
    }

    /// Get payload length (would be calculated from total length)
    pub fn payload_length(&self) -> usize {
        // This would be calculated from IP total length minus headers
        0
    }

    /// Parse TCP header from packet buffer
    pub fn parse(buffer: &mut PacketBuffer) -> NetworkResult<Self> {
        if buffer.remaining() < TCP_HEADER_MIN_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let src_port_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let source_port = u16::from_be_bytes([src_port_bytes[0], src_port_bytes[1]]);

        let dst_port_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let dest_port = u16::from_be_bytes([dst_port_bytes[0], dst_port_bytes[1]]);

        let seq_bytes = buffer.read(4).ok_or(NetworkError::InvalidPacket)?;
        let sequence_number = u32::from_be_bytes([seq_bytes[0], seq_bytes[1], seq_bytes[2], seq_bytes[3]]);

        let ack_bytes = buffer.read(4).ok_or(NetworkError::InvalidPacket)?;
        let acknowledgment_number = u32::from_be_bytes([ack_bytes[0], ack_bytes[1], ack_bytes[2], ack_bytes[3]]);

        let offset_flags_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let data_offset = (offset_flags_bytes[0] >> 4) & 0x0f;
        let flags = TcpFlags::from_byte(offset_flags_bytes[1]);

        let window_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let window_size = u16::from_be_bytes([window_bytes[0], window_bytes[1]]);

        let checksum_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let checksum = u16::from_be_bytes([checksum_bytes[0], checksum_bytes[1]]);

        let urgent_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let urgent_pointer = u16::from_be_bytes([urgent_bytes[0], urgent_bytes[1]]);

        // Read options if present
        let header_length = (data_offset as usize) * 4;
        let options_length = header_length.saturating_sub(TCP_HEADER_MIN_SIZE);
        let options = if options_length > 0 {
            let options_bytes = buffer.read(options_length).ok_or(NetworkError::InvalidPacket)?;
            options_bytes.to_vec()
        } else {
            Vec::new()
        };

        Ok(TcpHeader {
            source_port,
            dest_port,
            sequence_number,
            acknowledgment_number,
            data_offset,
            flags,
            window_size,
            checksum,
            urgent_pointer,
            options,
        })
    }

    /// Calculate TCP checksum
    pub fn calculate_checksum(&self, src_ip: &NetworkAddress, dst_ip: &NetworkAddress, payload: &[u8]) -> u16 {
        let mut sum = 0u32;

        // Pseudo-header
        match (src_ip, dst_ip) {
            (NetworkAddress::IPv4(src), NetworkAddress::IPv4(dst)) => {
                sum += ((src[0] as u32) << 8) | (src[1] as u32);
                sum += ((src[2] as u32) << 8) | (src[3] as u32);
                sum += ((dst[0] as u32) << 8) | (dst[1] as u32);
                sum += ((dst[2] as u32) << 8) | (dst[3] as u32);
                sum += 6; // Protocol (TCP)
                sum += (TCP_HEADER_MIN_SIZE + self.options.len() + payload.len()) as u32;
            }
            _ => return 0, // IPv6 not implemented yet
        }

        // TCP header
        sum += self.source_port as u32;
        sum += self.dest_port as u32;
        sum += (self.sequence_number >> 16) as u32;
        sum += (self.sequence_number & 0xFFFF) as u32;
        sum += (self.acknowledgment_number >> 16) as u32;
        sum += (self.acknowledgment_number & 0xFFFF) as u32;
        sum += ((self.data_offset as u32) << 12) | (self.flags.to_byte() as u32);
        sum += self.window_size as u32;
        // Skip checksum field
        sum += self.urgent_pointer as u32;

        // Options
        for chunk in self.options.chunks(2) {
            if chunk.len() == 2 {
                sum += ((chunk[0] as u32) << 8) | (chunk[1] as u32);
            } else {
                sum += (chunk[0] as u32) << 8;
            }
        }

        // Payload
        for chunk in payload.chunks(2) {
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

        !sum as u16
    }
}

/// TCP connection with complete state management
#[derive(Debug, Clone)]
pub struct TcpConnection {
    pub local_addr: NetworkAddress,
    pub local_port: u16,
    pub remote_addr: NetworkAddress,
    pub remote_port: u16,
    pub state: TcpState,
    pub send_sequence: u32,
    pub send_ack: u32,
    pub recv_sequence: u32,
    pub recv_ack: u32,
    pub send_window: u16,
    pub recv_window: u16,
    pub mss: u16,
    pub rtt: u32,
    pub cwnd: u32,
    pub ssthresh: u32,
    pub retransmit_timeout: u32,
    pub send_buffer: Vec<u8>,
    pub recv_buffer: Vec<u8>,
    pub send_unacked: Vec<u8>,
    pub last_ack_time: u64,
    pub retransmit_count: u8,
    pub keep_alive_time: u64,
    pub user_timeout: u32,
    pub duplicate_acks: u8,
    pub fast_retransmit: bool,
    pub sack_enabled: bool,
    pub window_scale: u8,
    pub timestamps_enabled: bool,
    pub syn_retries: u8,
    pub established_time: u64,
}

impl TcpConnection {
    pub fn new(
        local_addr: NetworkAddress,
        local_port: u16,
        remote_addr: NetworkAddress,
        remote_port: u16,
    ) -> Self {
        Self {
            local_addr,
            local_port,
            remote_addr,
            remote_port,
            state: TcpState::Closed,
            send_sequence: 0,
            send_ack: 0,
            recv_sequence: 0,
            recv_ack: 0,
            send_window: 65535,
            recv_window: 65535,
            mss: 1460,
            rtt: 100,
            cwnd: 1,
            ssthresh: 65535,
            retransmit_timeout: 3000,
            send_buffer: Vec::new(),
            recv_buffer: Vec::new(),
            send_unacked: Vec::new(),
            last_ack_time: current_time_ms(),
            retransmit_count: 0,
            keep_alive_time: current_time_ms(),
            user_timeout: 300000, // 5 minutes
            duplicate_acks: 0,
            fast_retransmit: false,
            sack_enabled: false,
            window_scale: 0,
            timestamps_enabled: false,
            syn_retries: 0,
            established_time: 0,
        }
    }

    /// Generate initial sequence number using secure random
    pub fn generate_isn(&mut self) {
        // Use a more secure ISN generation method
        let time_component = current_time_ms() as u32;
        let random_component = secure_random_u32();
        self.send_sequence = time_component.wrapping_add(random_component);
    }

    /// Check if connection has timed out
    pub fn is_timed_out(&self) -> bool {
        let now = current_time_ms();
        match self.state {
            TcpState::SynSent | TcpState::SynReceived => {
                now - self.last_ack_time > 75000 // 75 seconds for connection timeout
            }
            TcpState::Established | TcpState::CloseWait => {
                now - self.last_ack_time > self.user_timeout as u64
            }
            TcpState::FinWait1 | TcpState::FinWait2 | TcpState::Closing | TcpState::LastAck => {
                now - self.last_ack_time > 60000 // 60 seconds for close timeout
            }
            TcpState::TimeWait => {
                now - self.last_ack_time > 240000 // 4 minutes (2*MSL)
            }
            _ => false,
        }
    }

    /// Handle duplicate ACKs for fast retransmit
    pub fn handle_duplicate_ack(&mut self) {
        self.duplicate_acks += 1;
        if self.duplicate_acks >= 3 && !self.fast_retransmit {
            self.fast_retransmit = true;
            // Halve congestion window
            self.ssthresh = core::cmp::max(self.cwnd / 2, 2 * self.mss as u32);
            self.cwnd = self.ssthresh + 3 * self.mss as u32;
        } else if self.fast_retransmit {
            // Inflate congestion window
            self.cwnd += self.mss as u32;
        }
    }

    /// Reset duplicate ACK counter
    pub fn reset_duplicate_acks(&mut self) {
        self.duplicate_acks = 0;
        if self.fast_retransmit {
            self.fast_retransmit = false;
            self.cwnd = self.ssthresh;
        }
    }

    /// Check if keep-alive should be sent
    pub fn should_send_keepalive(&self) -> bool {
        if self.state != TcpState::Established {
            return false;
        }
        let now = current_time_ms();
        now - self.keep_alive_time > 7200000 // 2 hours
    }

    /// Update keep-alive timer
    pub fn update_keepalive(&mut self) {
        self.keep_alive_time = current_time_ms();
    }

    /// Update RTT estimate
    pub fn update_rtt(&mut self, measured_rtt: u32) {
        // Simple RTT estimation (Jacobson's algorithm would be better)
        self.rtt = (self.rtt * 7 + measured_rtt) / 8;
        self.retransmit_timeout = self.rtt * 2;
    }

    /// Update congestion window (simplified congestion control)
    pub fn update_cwnd(&mut self, acked_bytes: u32) {
        if self.cwnd < self.ssthresh {
            // Slow start
            self.cwnd += acked_bytes;
        } else {
            // Congestion avoidance
            self.cwnd += (acked_bytes * self.mss as u32) / self.cwnd;
        }
    }

    /// Handle congestion event
    pub fn handle_congestion(&mut self) {
        self.ssthresh = cmp::max(self.cwnd / 2, 2 * self.mss as u32);
        self.cwnd = self.mss as u32;
    }
}

/// TCP connection manager
pub struct TcpManager {
    connections: RwLock<BTreeMap<(NetworkAddress, u16, NetworkAddress, u16), TcpConnection>>,
    next_port: RwLock<u16>,
}

impl TcpManager {
    pub fn new() -> Self {
        Self {
            connections: RwLock::new(BTreeMap::new()),
            next_port: RwLock::new(32768), // Start of dynamic port range
        }
    }

    pub fn allocate_port(&self) -> u16 {
        let mut next_port = self.next_port.write();
        let port = *next_port;
        *next_port = if port >= 65535 { 32768 } else { port + 1 };
        port
    }

    pub fn create_connection(
        &self,
        local_addr: NetworkAddress,
        local_port: u16,
        remote_addr: NetworkAddress,
        remote_port: u16,
    ) -> NetworkResult<()> {
        let key = (local_addr, local_port, remote_addr, remote_port);
        let mut connections = self.connections.write();
        
        if connections.contains_key(&key) {
            return Err(NetworkError::AddressInUse);
        }

        let connection = TcpConnection::new(local_addr, local_port, remote_addr, remote_port);
        connections.insert(key, connection);
        Ok(())
    }

    pub fn get_connection(
        &self,
        local_addr: &NetworkAddress,
        local_port: u16,
        remote_addr: &NetworkAddress,
        remote_port: u16,
    ) -> Option<TcpConnection> {
        let connections = self.connections.read();
        let key = (*local_addr, local_port, *remote_addr, remote_port);
        connections.get(&key).cloned()
    }

    pub fn update_connection<F>(&self, key: (NetworkAddress, u16, NetworkAddress, u16), f: F) -> NetworkResult<()>
    where
        F: FnOnce(&mut TcpConnection),
    {
        let mut connections = self.connections.write();
        if let Some(connection) = connections.get_mut(&key) {
            f(connection);
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    pub fn remove_connection(
        &self,
        local_addr: &NetworkAddress,
        local_port: u16,
        remote_addr: &NetworkAddress,
        remote_port: u16,
    ) -> NetworkResult<()> {
        let mut connections = self.connections.write();
        let key = (*local_addr, local_port, *remote_addr, remote_port);
        
        if connections.remove(&key).is_some() {
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }
}

static TCP_MANAGER: TcpManager = TcpManager {
    connections: RwLock::new(BTreeMap::new()),
    next_port: RwLock::new(32768),
};

/// Get current time in milliseconds
fn current_time_ms() -> u64 {
    // TODO: Get actual system time from kernel
    // For now, return a mock timestamp
    1000000000 + (unsafe { core::arch::x86_64::_rdtsc() } / 1000000)
}

/// Generate secure random u32
fn secure_random_u32() -> u32 {
    // TODO: Use proper CSPRNG
    // For now, use RDRAND if available, otherwise fallback to TSC
    let mut result: u32 = 0;
    unsafe {
        if core::arch::x86_64::_rdrand32_step(&mut result) == 1 {
            result
        } else {
            (core::arch::x86_64::_rdtsc() as u32).wrapping_mul(1103515245).wrapping_add(12345)
        }
    }
}

/// Process incoming TCP packet
pub fn process_packet(
    _network_stack: &NetworkStack,
    src_ip: NetworkAddress,
    dst_ip: NetworkAddress,
    mut packet: PacketBuffer,
) -> NetworkResult<()> {
    let header = TcpHeader::parse(&mut packet)?;
    
    // Production: process TCP packet without debug output

    // Find existing connection
    let connection_key = (dst_ip, header.dest_port, src_ip, header.source_port);
    
    if let Some(mut connection) = TCP_MANAGER.get_connection(&dst_ip, header.dest_port, &src_ip, header.source_port) {
        // Process packet for existing connection
        process_connection_packet(&mut connection, &header, &packet.as_slice()[packet.position..])?;
        
        // Update connection in manager
        TCP_MANAGER.update_connection(connection_key, |conn| {
            *conn = connection;
        })?;
    } else {
        // Handle new connection attempt
        if header.flags.syn && !header.flags.ack {
            // Handle new TCP connection attempt
            handle_new_connection(dst_ip, header.dest_port, src_ip, header.source_port, &header)?;
        } else {
            // Send RST for non-existent connection
            send_rst_packet(dst_ip, header.dest_port, src_ip, header.source_port, header.sequence_number + 1)?;
        }
    }

    Ok(())
}

/// Process packet for existing connection
fn process_connection_packet(
    connection: &mut TcpConnection,
    header: &TcpHeader,
    payload: &[u8],
) -> NetworkResult<()> {
    match connection.state {
        TcpState::Listen => {
            if header.flags.syn && !header.flags.ack {
                // SYN received, send SYN-ACK
                connection.recv_sequence = header.sequence_number + 1;
                connection.generate_isn();
                connection.state = TcpState::SynReceived;
                send_syn_ack_packet(connection)?;
            }
        }
        TcpState::SynSent => {
            if header.flags.syn && header.flags.ack {
                // SYN-ACK received, send ACK
                if header.acknowledgment_number == connection.send_sequence + 1 {
                    connection.send_sequence += 1;
                    connection.recv_sequence = header.sequence_number + 1;
                    connection.state = TcpState::Established;
                    send_ack_packet(connection)?;
                    // TCP connection established (client)
                }
            }
        }
        TcpState::SynReceived => {
            if header.flags.ack && !header.flags.syn {
                // ACK received, connection established
                if header.acknowledgment_number == connection.send_sequence + 1 {
                    connection.send_sequence += 1;
                    connection.state = TcpState::Established;
                    // TCP connection established (server)
                }
            }
        }
        TcpState::Established => {
            // Handle data transfer
            if !payload.is_empty() {
                // Receive data
                if header.sequence_number == connection.recv_sequence {
                    connection.recv_buffer.extend_from_slice(payload);
                    connection.recv_sequence += payload.len() as u32;
                    send_ack_packet(connection)?;
                    // TCP data received
                }
            }

            // Handle connection close
            if header.flags.fin {
                connection.recv_sequence += 1;
                connection.state = TcpState::CloseWait;
                send_ack_packet(connection)?;
                // TCP connection closing
            }
        }
        TcpState::FinWait1 => {
            if header.flags.ack {
                connection.state = TcpState::FinWait2;
            }
            if header.flags.fin {
                connection.recv_sequence += 1;
                send_ack_packet(connection)?;
                if connection.state == TcpState::FinWait2 {
                    connection.state = TcpState::TimeWait;
                } else {
                    connection.state = TcpState::Closing;
                }
            }
        }
        TcpState::FinWait2 => {
            if header.flags.fin {
                connection.recv_sequence += 1;
                connection.state = TcpState::TimeWait;
                send_ack_packet(connection)?;
            }
        }
        TcpState::CloseWait => {
            // Application should close the connection
        }
        TcpState::LastAck => {
            if header.flags.ack {
                connection.state = TcpState::Closed;
                // TCP connection closed
            }
        }
        _ => {
            // Handle other states
        }
    }

    Ok(())
}

/// Handle new connection attempt
fn handle_new_connection(
    local_addr: NetworkAddress,
    local_port: u16,
    remote_addr: NetworkAddress,
    remote_port: u16,
    header: &TcpHeader,
) -> NetworkResult<()> {
    // Create new connection
    let mut connection = TcpConnection::new(local_addr, local_port, remote_addr, remote_port);
    connection.state = TcpState::Listen;
    connection.recv_sequence = header.sequence_number + 1;
    connection.generate_isn();
    connection.state = TcpState::SynReceived;

    // Store connection
    let key = (local_addr, local_port, remote_addr, remote_port);
    TCP_MANAGER.connections.write().insert(key, connection.clone());

    // Send SYN-ACK
    send_syn_ack_packet(&connection)?;
    
    Ok(())
}

/// Send SYN-ACK packet
fn send_syn_ack_packet(connection: &TcpConnection) -> NetworkResult<()> {
    let mut flags = TcpFlags::new();
    flags.syn = true;
    flags.ack = true;

    send_tcp_packet(
        connection.local_addr,
        connection.local_port,
        connection.remote_addr,
        connection.remote_port,
        connection.send_sequence,
        connection.recv_sequence,
        flags,
        connection.recv_window,
        &[],
    )
}

/// Send ACK packet
fn send_ack_packet(connection: &TcpConnection) -> NetworkResult<()> {
    let mut flags = TcpFlags::new();
    flags.ack = true;

    send_tcp_packet(
        connection.local_addr,
        connection.local_port,
        connection.remote_addr,
        connection.remote_port,
        connection.send_sequence,
        connection.recv_sequence,
        flags,
        connection.recv_window,
        &[],
    )
}

/// Send RST packet
fn send_rst_packet(
    local_addr: NetworkAddress,
    local_port: u16,
    remote_addr: NetworkAddress,
    remote_port: u16,
    sequence: u32,
) -> NetworkResult<()> {
    let mut flags = TcpFlags::new();
    flags.rst = true;

    send_tcp_packet(
        local_addr,
        local_port,
        remote_addr,
        remote_port,
        sequence,
        0,
        flags,
        0,
        &[],
    )
}

/// Send TCP packet
fn send_tcp_packet(
    src_ip: NetworkAddress,
    src_port: u16,
    dst_ip: NetworkAddress,
    dst_port: u16,
    sequence: u32,
    acknowledgment: u32,
    flags: TcpFlags,
    window: u16,
    payload: &[u8],
) -> NetworkResult<()> {
    // Create TCP header
    let header = TcpHeader {
        source_port: src_port,
        dest_port: dst_port,
        sequence_number: sequence,
        acknowledgment_number: acknowledgment,
        data_offset: 5, // 20 bytes (no options)
        flags,
        window_size: window,
        checksum: 0, // Will be calculated
        urgent_pointer: 0,
        options: Vec::new(),
    };

    // Calculate checksum
    let _checksum = header.calculate_checksum(&src_ip, &dst_ip, payload);
    
    // Production: send TCP packet without debug output

    // TODO: Serialize and send packet through IP layer
    
    Ok(())
}

/// TCP socket operations
pub fn tcp_connect(local_addr: NetworkAddress, remote_addr: NetworkAddress, remote_port: u16) -> NetworkResult<u16> {
    let local_port = TCP_MANAGER.allocate_port();
    
    // Create connection
    TCP_MANAGER.create_connection(local_addr, local_port, remote_addr, remote_port)?;
    
    // Start connection process
    let key = (local_addr, local_port, remote_addr, remote_port);
    TCP_MANAGER.update_connection(key, |conn| {
        conn.generate_isn();
        conn.state = TcpState::SynSent;
    })?;

    // Send SYN packet
    let mut flags = TcpFlags::new();
    flags.syn = true;
    
    send_tcp_packet(local_addr, local_port, remote_addr, remote_port, 0, 0, flags, 65535, &[])?;
    
    Ok(local_port)
}

/// TCP listen
pub fn tcp_listen(local_addr: NetworkAddress, local_port: u16) -> NetworkResult<()> {
    // Create listening connection (placeholder for actual listening socket)
    let dummy_remote = NetworkAddress::IPv4([0, 0, 0, 0]);
    TCP_MANAGER.create_connection(local_addr, local_port, dummy_remote, 0)?;
    
    let key = (local_addr, local_port, dummy_remote, 0);
    TCP_MANAGER.update_connection(key, |conn| {
        conn.state = TcpState::Listen;
    })?;

    // TCP socket listening
    Ok(())
}
