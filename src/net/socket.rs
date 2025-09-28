//! Socket interface implementation
//!
//! This module provides the socket abstraction layer for network communication,
//! supporting TCP, UDP, and raw sockets with a POSIX-like interface.

use super::{NetworkAddress, Protocol, NetworkError, NetworkResult, PacketBuffer};
use alloc::{vec::Vec, collections::VecDeque};
use core::fmt;

/// Socket types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketType {
    /// Stream socket (TCP)
    Stream,
    /// Datagram socket (UDP)
    Datagram,
    /// Raw socket
    Raw,
}

/// Socket states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketState {
    /// Socket is closed
    Closed,
    /// Socket is listening for connections
    Listening,
    /// Socket is connecting
    Connecting,
    /// Socket is connected
    Connected,
    /// Socket is closing
    Closing,
}

/// Socket address
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocketAddress {
    /// IP address
    pub address: NetworkAddress,
    /// Port number
    pub port: u16,
}

impl SocketAddress {
    /// Create a new socket address
    pub fn new(address: NetworkAddress, port: u16) -> Self {
        Self { address, port }
    }

    /// Create IPv4 socket address
    pub fn ipv4(a: u8, b: u8, c: u8, d: u8, port: u16) -> Self {
        Self {
            address: NetworkAddress::ipv4(a, b, c, d),
            port,
        }
    }
}

impl fmt::Display for SocketAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.address, self.port)
    }
}

/// Socket options
#[derive(Debug, Clone)]
pub struct SocketOptions {
    /// Reuse address
    pub reuse_addr: bool,
    /// Reuse port
    pub reuse_port: bool,
    /// Keep alive
    pub keep_alive: bool,
    /// No delay (disable Nagle's algorithm)
    pub no_delay: bool,
    /// Receive buffer size
    pub recv_buffer_size: usize,
    /// Send buffer size
    pub send_buffer_size: usize,
    /// Receive timeout (milliseconds)
    pub recv_timeout: Option<u32>,
    /// Send timeout (milliseconds)
    pub send_timeout: Option<u32>,
}

impl Default for SocketOptions {
    fn default() -> Self {
        Self {
            reuse_addr: false,
            reuse_port: false,
            keep_alive: false,
            no_delay: false,
            recv_buffer_size: 8192,
            send_buffer_size: 8192,
            recv_timeout: None,
            send_timeout: None,
        }
    }
}

/// Socket statistics
#[derive(Debug, Clone, Default)]
pub struct SocketStats {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Send errors
    pub send_errors: u64,
    /// Receive errors
    pub recv_errors: u64,
}

/// Socket implementation
#[derive(Debug, Clone)]
pub struct Socket {
    /// Socket ID
    pub id: u32,
    /// Socket type
    pub socket_type: SocketType,
    /// Protocol
    pub protocol: Protocol,
    /// Current state
    pub state: SocketState,
    /// Local address
    pub local_address: Option<SocketAddress>,
    /// Remote address
    pub remote_address: Option<SocketAddress>,
    /// Socket options
    pub options: SocketOptions,
    /// Receive buffer
    pub recv_buffer: VecDeque<u8>,
    /// Send buffer
    pub send_buffer: VecDeque<u8>,
    /// Pending connections (for listening sockets)
    pub pending_connections: VecDeque<u32>,
    /// Socket statistics
    pub stats: SocketStats,
}

impl Socket {
    /// Create a new socket
    pub fn new(id: u32, socket_type: SocketType, protocol: Protocol) -> Self {
        Self {
            id,
            socket_type,
            protocol,
            state: SocketState::Closed,
            local_address: None,
            remote_address: None,
            options: SocketOptions::default(),
            recv_buffer: VecDeque::new(),
            send_buffer: VecDeque::new(),
            pending_connections: VecDeque::new(),
            stats: SocketStats::default(),
        }
    }

    /// Bind socket to local address
    pub fn bind(&mut self, address: SocketAddress) -> NetworkResult<()> {
        if self.state != SocketState::Closed {
            return Err(NetworkError::InvalidAddress);
        }

        // Check if address is already in use
        // TODO: Implement proper address checking
        
        self.local_address = Some(address);
        println!("Socket {} bound to {}", self.id, address);
        Ok(())
    }

    /// Listen for incoming connections (TCP only)
    pub fn listen(&mut self, backlog: u32) -> NetworkResult<()> {
        if self.socket_type != SocketType::Stream {
            return Err(NetworkError::NotSupported);
        }

        if self.local_address.is_none() {
            return Err(NetworkError::InvalidAddress);
        }

        self.state = SocketState::Listening;
        self.pending_connections.clear();
        
        println!("Socket {} listening with backlog {}", self.id, backlog);
        Ok(())
    }

    /// Connect to remote address
    pub fn connect(&mut self, address: SocketAddress) -> NetworkResult<()> {
        if self.state != SocketState::Closed {
            return Err(NetworkError::InvalidAddress);
        }

        self.remote_address = Some(address);
        self.state = SocketState::Connecting;

        // TODO: Implement actual connection logic
        match self.socket_type {
            SocketType::Stream => {
                // TCP connection
                println!("Socket {} connecting to {} (TCP)", self.id, address);
                // Simulate successful connection
                self.state = SocketState::Connected;
            }
            SocketType::Datagram => {
                // UDP "connection" (just sets default destination)
                println!("Socket {} connected to {} (UDP)", self.id, address);
                self.state = SocketState::Connected;
            }
            SocketType::Raw => {
                return Err(NetworkError::NotSupported);
            }
        }

        Ok(())
    }

    /// Accept incoming connection (TCP only)
    pub fn accept(&mut self) -> NetworkResult<Option<u32>> {
        if self.socket_type != SocketType::Stream || self.state != SocketState::Listening {
            return Err(NetworkError::NotSupported);
        }

        if let Some(connection_id) = self.pending_connections.pop_front() {
            Ok(Some(connection_id))
        } else {
            Ok(None)
        }
    }

    /// Send data through socket
    pub fn send(&mut self, data: &[u8]) -> NetworkResult<usize> {
        if self.state != SocketState::Connected {
            return Err(NetworkError::ConnectionRefused);
        }

        // Check send buffer space
        if self.send_buffer.len() + data.len() > self.options.send_buffer_size {
            return Err(NetworkError::BufferOverflow);
        }

        // Add data to send buffer
        self.send_buffer.extend(data.iter());
        
        // TODO: Implement actual packet transmission
        let bytes_sent = data.len();
        self.stats.bytes_sent += bytes_sent as u64;
        self.stats.packets_sent += 1;

        println!("Socket {} sent {} bytes", self.id, bytes_sent);
        Ok(bytes_sent)
    }

    /// Receive data from socket
    pub fn recv(&mut self, buffer: &mut [u8]) -> NetworkResult<usize> {
        if self.state != SocketState::Connected && self.state != SocketState::Listening {
            return Err(NetworkError::ConnectionRefused);
        }

        let bytes_to_read = core::cmp::min(buffer.len(), self.recv_buffer.len());
        
        if bytes_to_read == 0 {
            return Ok(0); // No data available
        }

        // Copy data from receive buffer
        for i in 0..bytes_to_read {
            buffer[i] = self.recv_buffer.pop_front().unwrap();
        }

        self.stats.bytes_received += bytes_to_read as u64;
        self.stats.packets_received += 1;

        println!("Socket {} received {} bytes", self.id, bytes_to_read);
        Ok(bytes_to_read)
    }

    /// Send data to specific address (UDP only)
    pub fn send_to(&mut self, data: &[u8], address: SocketAddress) -> NetworkResult<usize> {
        if self.socket_type != SocketType::Datagram {
            return Err(NetworkError::NotSupported);
        }

        // TODO: Implement UDP packet transmission
        let bytes_sent = data.len();
        self.stats.bytes_sent += bytes_sent as u64;
        self.stats.packets_sent += 1;

        println!("Socket {} sent {} bytes to {}", self.id, bytes_sent, address);
        Ok(bytes_sent)
    }

    /// Receive data from any address (UDP only)
    pub fn recv_from(&mut self, buffer: &mut [u8]) -> NetworkResult<(usize, SocketAddress)> {
        if self.socket_type != SocketType::Datagram {
            return Err(NetworkError::NotSupported);
        }

        // TODO: Implement UDP packet reception with source address
        let bytes_received = core::cmp::min(buffer.len(), self.recv_buffer.len());
        
        if bytes_received == 0 {
            return Err(NetworkError::Timeout); // No data available
        }

        // Copy data from receive buffer
        for i in 0..bytes_received {
            buffer[i] = self.recv_buffer.pop_front().unwrap();
        }

        // Return dummy source address for now
        let source = SocketAddress::ipv4(192, 168, 1, 100, 12345);
        
        self.stats.bytes_received += bytes_received as u64;
        self.stats.packets_received += 1;

        println!("Socket {} received {} bytes from {}", self.id, bytes_received, source);
        Ok((bytes_received, source))
    }

    /// Close the socket
    pub fn close(&mut self) -> NetworkResult<()> {
        match self.state {
            SocketState::Closed => return Ok(()),
            SocketState::Connected => {
                self.state = SocketState::Closing;
                // TODO: Implement proper connection teardown
                self.state = SocketState::Closed;
            }
            _ => {
                self.state = SocketState::Closed;
            }
        }

        self.recv_buffer.clear();
        self.send_buffer.clear();
        self.pending_connections.clear();

        println!("Socket {} closed", self.id);
        Ok(())
    }

    /// Set socket option
    pub fn set_option(&mut self, option: SocketOption) -> NetworkResult<()> {
        match option {
            SocketOption::ReuseAddr(value) => self.options.reuse_addr = value,
            SocketOption::ReusePort(value) => self.options.reuse_port = value,
            SocketOption::KeepAlive(value) => self.options.keep_alive = value,
            SocketOption::NoDelay(value) => self.options.no_delay = value,
            SocketOption::RecvBufferSize(size) => {
                self.options.recv_buffer_size = size;
                // Resize buffer if needed
                if self.recv_buffer.len() > size {
                    self.recv_buffer.truncate(size);
                }
            }
            SocketOption::SendBufferSize(size) => {
                self.options.send_buffer_size = size;
                // Resize buffer if needed
                if self.send_buffer.len() > size {
                    self.send_buffer.truncate(size);
                }
            }
            SocketOption::RecvTimeout(timeout) => self.options.recv_timeout = timeout,
            SocketOption::SendTimeout(timeout) => self.options.send_timeout = timeout,
        }

        Ok(())
    }

    /// Get socket option
    pub fn get_option(&self, option_type: SocketOptionType) -> NetworkResult<SocketOption> {
        let option = match option_type {
            SocketOptionType::ReuseAddr => SocketOption::ReuseAddr(self.options.reuse_addr),
            SocketOptionType::ReusePort => SocketOption::ReusePort(self.options.reuse_port),
            SocketOptionType::KeepAlive => SocketOption::KeepAlive(self.options.keep_alive),
            SocketOptionType::NoDelay => SocketOption::NoDelay(self.options.no_delay),
            SocketOptionType::RecvBufferSize => SocketOption::RecvBufferSize(self.options.recv_buffer_size),
            SocketOptionType::SendBufferSize => SocketOption::SendBufferSize(self.options.send_buffer_size),
            SocketOptionType::RecvTimeout => SocketOption::RecvTimeout(self.options.recv_timeout),
            SocketOptionType::SendTimeout => SocketOption::SendTimeout(self.options.send_timeout),
        };

        Ok(option)
    }

    /// Check if socket has data available for reading
    pub fn has_data(&self) -> bool {
        !self.recv_buffer.is_empty()
    }

    /// Check if socket can accept more data for sending
    pub fn can_send(&self) -> bool {
        self.send_buffer.len() < self.options.send_buffer_size
    }

    /// Get number of bytes available for reading
    pub fn available_bytes(&self) -> usize {
        self.recv_buffer.len()
    }

    /// Add data to receive buffer (used by network stack)
    pub fn add_received_data(&mut self, data: &[u8]) -> NetworkResult<()> {
        if self.recv_buffer.len() + data.len() > self.options.recv_buffer_size {
            return Err(NetworkError::BufferOverflow);
        }

        self.recv_buffer.extend(data.iter());
        Ok(())
    }
}

/// Socket option types
#[derive(Debug, Clone, Copy)]
pub enum SocketOptionType {
    ReuseAddr,
    ReusePort,
    KeepAlive,
    NoDelay,
    RecvBufferSize,
    SendBufferSize,
    RecvTimeout,
    SendTimeout,
}

/// Socket option values
#[derive(Debug, Clone)]
pub enum SocketOption {
    ReuseAddr(bool),
    ReusePort(bool),
    KeepAlive(bool),
    NoDelay(bool),
    RecvBufferSize(usize),
    SendBufferSize(usize),
    RecvTimeout(Option<u32>),
    SendTimeout(Option<u32>),
}

/// Socket domain (address family)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketDomain {
    /// Internet Protocol version 4
    IPv4,
    /// Internet Protocol version 6
    IPv6,
    /// Unix domain sockets
    Unix,
}

/// High-level socket operations
pub struct SocketManager;

impl SocketManager {
    /// Create a TCP socket
    pub fn tcp_socket() -> NetworkResult<u32> {
        super::network_stack().create_socket(SocketType::Stream, Protocol::TCP)
    }

    /// Create a UDP socket
    pub fn udp_socket() -> NetworkResult<u32> {
        super::network_stack().create_socket(SocketType::Datagram, Protocol::UDP)
    }

    /// Create a raw socket
    pub fn raw_socket(protocol: Protocol) -> NetworkResult<u32> {
        super::network_stack().create_socket(SocketType::Raw, protocol)
    }

    /// Close a socket
    pub fn close_socket(socket_id: u32) -> NetworkResult<()> {
        super::network_stack().close_socket(socket_id)
    }
}
