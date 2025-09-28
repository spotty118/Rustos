//! Inter-Process Communication (IPC) Module for RustOS
//!
//! This module provides IPC mechanisms including pipes, message queues,
//! shared memory, and semaphores for process communication.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use alloc::{vec::Vec, string::String, collections::BTreeMap, boxed::Box, format, vec};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use crate::{println, security};

/// IPC object types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcType {
    /// Anonymous pipe
    Pipe,
    /// Named pipe (FIFO)
    NamedPipe,
    /// Message queue
    MessageQueue,
    /// Shared memory segment
    SharedMemory,
    /// Semaphore
    Semaphore,
    /// Mutex
    Mutex,
}

/// IPC object ID type
pub type IpcId = u32;

/// Process ID type
pub type ProcessId = u32;

/// IPC permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpcPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub owner: ProcessId,
    pub group: u32,
}

impl Default for IpcPermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
            owner: 0,
            group: 0,
        }
    }
}

/// Message structure for message queues
#[derive(Debug, Clone)]
pub struct Message {
    pub sender: ProcessId,
    pub recipient: ProcessId,
    pub msg_type: u32,
    pub data: Vec<u8>,
    pub timestamp: u64,
    pub priority: u8,
}

/// Pipe structure
#[derive(Debug)]
pub struct Pipe {
    pub id: IpcId,
    pub buffer: Vec<u8>,
    pub capacity: usize,
    pub read_pos: usize,
    pub write_pos: usize,
    pub permissions: IpcPermissions,
    pub readers: Vec<ProcessId>,
    pub writers: Vec<ProcessId>,
}

impl Pipe {
    fn new(id: IpcId, capacity: usize) -> Self {
        Self {
            id,
            buffer: vec![0; capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
            permissions: IpcPermissions::default(),
            readers: Vec::new(),
            writers: Vec::new(),
        }
    }

    fn available_read(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    fn available_write(&self) -> usize {
        self.capacity - self.available_read() - 1
    }

    fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.available_read();
        let to_read = buf.len().min(available);
        
        for i in 0..to_read {
            buf[i] = self.buffer[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
        
        to_read
    }

    fn write(&mut self, data: &[u8]) -> usize {
        let available = self.available_write();
        let to_write = data.len().min(available);
        
        for i in 0..to_write {
            self.buffer[self.write_pos] = data[i];
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        
        to_write
    }
}

/// Message queue structure
#[derive(Debug)]
pub struct MessageQueue {
    pub id: IpcId,
    pub messages: Vec<Message>,
    pub max_messages: usize,
    pub max_message_size: usize,
    pub permissions: IpcPermissions,
}

impl MessageQueue {
    fn new(id: IpcId, max_messages: usize, max_message_size: usize) -> Self {
        Self {
            id,
            messages: Vec::new(),
            max_messages,
            max_message_size,
            permissions: IpcPermissions::default(),
        }
    }

    fn send_message(&mut self, message: Message) -> Result<(), &'static str> {
        if self.messages.len() >= self.max_messages {
            return Err("Message queue full");
        }
        
        if message.data.len() > self.max_message_size {
            return Err("Message too large");
        }
        
        self.messages.push(message);
        Ok(())
    }

    fn receive_message(&mut self, process_id: ProcessId) -> Option<Message> {
        // Find message for this process (simple FIFO for now)
        for i in 0..self.messages.len() {
            if self.messages[i].recipient == process_id || self.messages[i].recipient == 0 {
                return Some(self.messages.remove(i));
            }
        }
        None
    }
}

/// Shared memory segment
#[derive(Debug)]
pub struct SharedMemory {
    pub id: IpcId,
    pub data: Vec<u8>,
    pub size: usize,
    pub permissions: IpcPermissions,
    pub attached_processes: Vec<ProcessId>,
}

impl SharedMemory {
    fn new(id: IpcId, size: usize) -> Self {
        Self {
            id,
            data: vec![0; size],
            size,
            permissions: IpcPermissions::default(),
            attached_processes: Vec::new(),
        }
    }

    fn attach(&mut self, process_id: ProcessId) -> Result<(), &'static str> {
        if !self.attached_processes.contains(&process_id) {
            self.attached_processes.push(process_id);
        }
        Ok(())
    }

    fn detach(&mut self, process_id: ProcessId) -> Result<(), &'static str> {
        self.attached_processes.retain(|&id| id != process_id);
        Ok(())
    }
}

/// Semaphore structure
#[derive(Debug)]
pub struct Semaphore {
    pub id: IpcId,
    pub value: i32,
    pub max_value: i32,
    pub waiting_processes: Vec<ProcessId>,
    pub permissions: IpcPermissions,
}

impl Semaphore {
    fn new(id: IpcId, initial_value: i32, max_value: i32) -> Self {
        Self {
            id,
            value: initial_value,
            max_value,
            waiting_processes: Vec::new(),
            permissions: IpcPermissions::default(),
        }
    }

    fn wait(&mut self, process_id: ProcessId) -> Result<(), &'static str> {
        if self.value > 0 {
            self.value -= 1;
            Ok(())
        } else {
            if !self.waiting_processes.contains(&process_id) {
                self.waiting_processes.push(process_id);
            }
            Err("Process blocked on semaphore")
        }
    }

    fn signal(&mut self) -> Option<ProcessId> {
        if self.value < self.max_value {
            self.value += 1;
            
            // Wake up a waiting process
            if !self.waiting_processes.is_empty() {
                Some(self.waiting_processes.remove(0))
            } else {
                None
            }
        } else {
            None
        }
    }
}

lazy_static! {
    /// IPC object registry
    static ref IPC_OBJECTS: RwLock<BTreeMap<IpcId, Box<dyn IpcObject + Send + Sync>>> = RwLock::new(BTreeMap::new());
    
    /// Pipes registry
    static ref PIPES: Mutex<BTreeMap<IpcId, Pipe>> = Mutex::new(BTreeMap::new());
    
    /// Message queues registry
    static ref MESSAGE_QUEUES: Mutex<BTreeMap<IpcId, MessageQueue>> = Mutex::new(BTreeMap::new());
    
    /// Shared memory registry
    static ref SHARED_MEMORY: Mutex<BTreeMap<IpcId, SharedMemory>> = Mutex::new(BTreeMap::new());
    
    /// Semaphores registry
    static ref SEMAPHORES: Mutex<BTreeMap<IpcId, Semaphore>> = Mutex::new(BTreeMap::new());
}

/// Next IPC ID
static NEXT_IPC_ID: AtomicU32 = AtomicU32::new(1);

/// IPC statistics
static IPC_OPERATIONS: AtomicUsize = AtomicUsize::new(0);
static MESSAGES_SENT: AtomicUsize = AtomicUsize::new(0);
static MESSAGES_RECEIVED: AtomicUsize = AtomicUsize::new(0);

/// IPC object trait
pub trait IpcObject {
    fn get_id(&self) -> IpcId;
    fn get_type(&self) -> IpcType;
    fn get_permissions(&self) -> IpcPermissions;
    fn set_permissions(&mut self, permissions: IpcPermissions);
}

/// Initialize IPC subsystem
pub fn init() -> Result<(), &'static str> {
    println!("IPC: Initializing Inter-Process Communication subsystem");
    
    // Reset statistics
    IPC_OPERATIONS.store(0, Ordering::SeqCst);
    MESSAGES_SENT.store(0, Ordering::SeqCst);
    MESSAGES_RECEIVED.store(0, Ordering::SeqCst);
    
    println!("✓ IPC subsystem initialized");
    Ok(())
}

/// Create a new pipe
pub fn create_pipe(capacity: usize) -> Result<IpcId, &'static str> {
    let id = NEXT_IPC_ID.fetch_add(1, Ordering::SeqCst);
    let pipe = Pipe::new(id, capacity);
    
    PIPES.lock().insert(id, pipe);
    IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
    
    Ok(id)
}

/// Read from pipe
pub fn pipe_read(pipe_id: IpcId, process_id: ProcessId, buf: &mut [u8]) -> Result<usize, &'static str> {
    let mut pipes = PIPES.lock();
    
    if let Some(pipe) = pipes.get_mut(&pipe_id) {
        // Check permissions
        if !pipe.readers.contains(&process_id) && pipe.permissions.owner != process_id {
            security::log_security_event(security::SecurityEvent::PermissionDenied,
                                        &format!("Process {} denied pipe read access", process_id));
            return Err("Permission denied");
        }
        
        let bytes_read = pipe.read(buf);
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(bytes_read)
    } else {
        Err("Pipe not found")
    }
}

/// Write to pipe
pub fn pipe_write(pipe_id: IpcId, process_id: ProcessId, data: &[u8]) -> Result<usize, &'static str> {
    let mut pipes = PIPES.lock();
    
    if let Some(pipe) = pipes.get_mut(&pipe_id) {
        // Check permissions
        if !pipe.writers.contains(&process_id) && pipe.permissions.owner != process_id {
            security::log_security_event(security::SecurityEvent::PermissionDenied,
                                        &format!("Process {} denied pipe write access", process_id));
            return Err("Permission denied");
        }
        
        let bytes_written = pipe.write(data);
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(bytes_written)
    } else {
        Err("Pipe not found")
    }
}

/// Create message queue
pub fn create_message_queue(max_messages: usize, max_message_size: usize) -> Result<IpcId, &'static str> {
    let id = NEXT_IPC_ID.fetch_add(1, Ordering::SeqCst);
    let mq = MessageQueue::new(id, max_messages, max_message_size);
    
    MESSAGE_QUEUES.lock().insert(id, mq);
    IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
    
    Ok(id)
}

/// Send message to queue
pub fn send_message(queue_id: IpcId, message: Message) -> Result<(), &'static str> {
    let mut queues = MESSAGE_QUEUES.lock();
    
    if let Some(queue) = queues.get_mut(&queue_id) {
        queue.send_message(message)?;
        MESSAGES_SENT.fetch_add(1, Ordering::SeqCst);
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    } else {
        Err("Message queue not found")
    }
}

/// Receive message from queue
pub fn receive_message(queue_id: IpcId, process_id: ProcessId) -> Result<Option<Message>, &'static str> {
    let mut queues = MESSAGE_QUEUES.lock();
    
    if let Some(queue) = queues.get_mut(&queue_id) {
        let message = queue.receive_message(process_id);
        if message.is_some() {
            MESSAGES_RECEIVED.fetch_add(1, Ordering::SeqCst);
        }
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(message)
    } else {
        Err("Message queue not found")
    }
}

/// Create shared memory segment
pub fn create_shared_memory(size: usize) -> Result<IpcId, &'static str> {
    let id = NEXT_IPC_ID.fetch_add(1, Ordering::SeqCst);
    let shm = SharedMemory::new(id, size);
    
    SHARED_MEMORY.lock().insert(id, shm);
    IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
    
    Ok(id)
}

/// Attach to shared memory
pub fn attach_shared_memory(shm_id: IpcId, process_id: ProcessId) -> Result<(), &'static str> {
    let mut shm_segments = SHARED_MEMORY.lock();
    
    if let Some(shm) = shm_segments.get_mut(&shm_id) {
        shm.attach(process_id)?;
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    } else {
        Err("Shared memory segment not found")
    }
}

/// Create semaphore
pub fn create_semaphore(initial_value: i32, max_value: i32) -> Result<IpcId, &'static str> {
    let id = NEXT_IPC_ID.fetch_add(1, Ordering::SeqCst);
    let sem = Semaphore::new(id, initial_value, max_value);
    
    SEMAPHORES.lock().insert(id, sem);
    IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
    
    Ok(id)
}

/// Wait on semaphore
pub fn semaphore_wait(sem_id: IpcId, process_id: ProcessId) -> Result<(), &'static str> {
    let mut semaphores = SEMAPHORES.lock();
    
    if let Some(sem) = semaphores.get_mut(&sem_id) {
        sem.wait(process_id)?;
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    } else {
        Err("Semaphore not found")
    }
}

/// Signal semaphore
pub fn semaphore_signal(sem_id: IpcId) -> Result<Option<ProcessId>, &'static str> {
    let mut semaphores = SEMAPHORES.lock();
    
    if let Some(sem) = semaphores.get_mut(&sem_id) {
        let woken_process = sem.signal();
        IPC_OPERATIONS.fetch_add(1, Ordering::SeqCst);
        Ok(woken_process)
    } else {
        Err("Semaphore not found")
    }
}

/// IPC statistics
#[derive(Debug, Clone)]
pub struct IpcStatistics {
    pub total_operations: usize,
    pub messages_sent: usize,
    pub messages_received: usize,
    pub active_pipes: usize,
    pub active_message_queues: usize,
    pub active_shared_memory: usize,
    pub active_semaphores: usize,
}

/// Get IPC statistics
pub fn get_ipc_statistics() -> IpcStatistics {
    IpcStatistics {
        total_operations: IPC_OPERATIONS.load(Ordering::SeqCst),
        messages_sent: MESSAGES_SENT.load(Ordering::SeqCst),
        messages_received: MESSAGES_RECEIVED.load(Ordering::SeqCst),
        active_pipes: PIPES.lock().len(),
        active_message_queues: MESSAGE_QUEUES.lock().len(),
        active_shared_memory: SHARED_MEMORY.lock().len(),
        active_semaphores: SEMAPHORES.lock().len(),
    }
}

/// Clean up IPC objects for a process
pub fn cleanup_process_ipc(process_id: ProcessId) -> Result<(), &'static str> {
    // Clean up pipes
    let mut pipes = PIPES.lock();
    pipes.retain(|_, pipe| {
        pipe.readers.retain(|&id| id != process_id);
        pipe.writers.retain(|&id| id != process_id);
        !pipe.readers.is_empty() || !pipe.writers.is_empty() || pipe.permissions.owner != process_id
    });
    
    // Clean up shared memory
    let mut shm_segments = SHARED_MEMORY.lock();
    for shm in shm_segments.values_mut() {
        let _ = shm.detach(process_id);
    }
    
    // Clean up semaphores
    let mut semaphores = SEMAPHORES.lock();
    for sem in semaphores.values_mut() {
        sem.waiting_processes.retain(|&id| id != process_id);
    }
    
    Ok(())
}

/// Demonstrate IPC functionality
pub fn demonstrate_ipc() -> Result<(), &'static str> {
    println!("=== IPC Functionality Demonstration ===");
    
    // Create and test pipe
    let pipe_id = create_pipe(1024)?;
    println!("✓ Created pipe with ID {}", pipe_id);
    
    let test_data = b"Hello, IPC!";
    let bytes_written = pipe_write(pipe_id, 0, test_data)?;
    println!("✓ Wrote {} bytes to pipe", bytes_written);
    
    let mut read_buf = [0u8; 32];
    let bytes_read = pipe_read(pipe_id, 0, &mut read_buf)?;
    println!("✓ Read {} bytes from pipe", bytes_read);
    
    // Create and test message queue
    let mq_id = create_message_queue(10, 256)?;
    println!("✓ Created message queue with ID {}", mq_id);
    
    let message = Message {
        sender: 0,
        recipient: 1,
        msg_type: 1,
        data: b"Test message".to_vec(),
        timestamp: crate::time::uptime_ms(),
        priority: 5,
    };
    
    send_message(mq_id, message)?;
    println!("✓ Sent message to queue");
    
    // Create semaphore
    let sem_id = create_semaphore(1, 1)?;
    println!("✓ Created semaphore with ID {}", sem_id);
    
    semaphore_wait(sem_id, 0)?;
    println!("✓ Acquired semaphore");
    
    let _ = semaphore_signal(sem_id)?;
    println!("✓ Released semaphore");
    
    // Create shared memory
    let shm_id = create_shared_memory(4096)?;
    println!("✓ Created shared memory segment with ID {}", shm_id);
    
    attach_shared_memory(shm_id, 0)?;
    println!("✓ Attached to shared memory");
    
    // Display statistics
    let stats = get_ipc_statistics();
    println!("IPC Statistics:");
    println!("  Total Operations: {}", stats.total_operations);
    println!("  Messages Sent: {}", stats.messages_sent);
    println!("  Active Pipes: {}", stats.active_pipes);
    println!("  Active Message Queues: {}", stats.active_message_queues);
    println!("  Active Shared Memory: {}", stats.active_shared_memory);
    println!("  Active Semaphores: {}", stats.active_semaphores);
    
    println!("✓ IPC demonstration completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_creation() {
        let pipe = Pipe::new(1, 1024);
        assert_eq!(pipe.id, 1);
        assert_eq!(pipe.capacity, 1024);
        assert_eq!(pipe.available_read(), 0);
        assert_eq!(pipe.available_write(), 1023);
    }

    #[test]
    fn test_message_queue_creation() {
        let mq = MessageQueue::new(1, 10, 256);
        assert_eq!(mq.id, 1);
        assert_eq!(mq.max_messages, 10);
        assert_eq!(mq.max_message_size, 256);
    }

    #[test]
    fn test_semaphore_creation() {
        let sem = Semaphore::new(1, 5, 10);
        assert_eq!(sem.id, 1);
        assert_eq!(sem.value, 5);
        assert_eq!(sem.max_value, 10);
    }
}