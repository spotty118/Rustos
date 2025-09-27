//! Inter-Process Communication (IPC) System for RustOS
//!
//! This module provides comprehensive IPC mechanisms including:
//! - Pipes (anonymous and named/FIFO)
//! - Message queues with priority support
//! - Shared memory segments
//! - Signals for process notification
//! - Semaphores for synchronization
//! - Mutexes for mutual exclusion
//! - Event notifications
//! - Mailboxes for structured communication

use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    string::{String, ToString},
    vec::Vec,
};
use core::{
    fmt,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use x86_64::VirtAddr;

use crate::{
    process::{ProcessId, get_process_manager},
    memory::{MemoryRegionType, MemoryProtection, allocate_memory},
    time,
};

/// IPC object identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IpcId(u64);

impl IpcId {
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        IpcId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for IpcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IPC:{}", self.0)
    }
}

/// IPC error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcError {
    /// Object not found
    NotFound,
    /// Permission denied
    PermissionDenied,
    /// Object already exists
    AlreadyExists,
    /// Invalid argument
    InvalidArgument,
    /// Operation would block
    WouldBlock,
    /// Resource busy
    Busy,
    /// Out of memory
    OutOfMemory,
    /// Broken pipe (no readers/writers)
    BrokenPipe,
    /// Message too large
    MessageTooLarge,
    /// Queue is full
    QueueFull,
    /// Queue is empty
    QueueEmpty,
    /// Deadlock detected
    Deadlock,
    /// Timeout occurred
    Timeout,
    /// Object destroyed
    ObjectDestroyed,
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcError::NotFound => write!(f, "IPC object not found"),
            IpcError::PermissionDenied => write!(f, "Permission denied"),
            IpcError::AlreadyExists => write!(f, "IPC object already exists"),
            IpcError::InvalidArgument => write!(f, "Invalid argument"),
            IpcError::WouldBlock => write!(f, "Operation would block"),
            IpcError::Busy => write!(f, "Resource busy"),
            IpcError::OutOfMemory => write!(f, "Out of memory"),
            IpcError::BrokenPipe => write!(f, "Broken pipe"),
            IpcError::MessageTooLarge => write!(f, "Message too large"),
            IpcError::QueueFull => write!(f, "Queue is full"),
            IpcError::QueueEmpty => write!(f, "Queue is empty"),
            IpcError::Deadlock => write!(f, "Deadlock detected"),
            IpcError::Timeout => write!(f, "Timeout occurred"),
            IpcError::ObjectDestroyed => write!(f, "Object destroyed"),
        }
    }
}

/// IPC permission flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpcPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub owner_only: bool,
}

impl IpcPermissions {
    pub const READ_WRITE: Self = IpcPermissions {
        read: true,
        write: true,
        execute: false,
        owner_only: false,
    };

    pub const READ_ONLY: Self = IpcPermissions {
        read: true,
        write: false,
        execute: false,
        owner_only: false,
    };

    pub const OWNER_ONLY: Self = IpcPermissions {
        read: true,
        write: true,
        execute: false,
        owner_only: true,
    };
}

/// IPC object types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcObjectType {
    Pipe,
    MessageQueue,
    SharedMemory,
    Semaphore,
    Mutex,
    Event,
    Mailbox,
}

/// Base IPC object metadata
#[derive(Debug, Clone)]
pub struct IpcObjectMetadata {
    pub id: IpcId,
    pub object_type: IpcObjectType,
    pub owner: ProcessId,
    pub permissions: IpcPermissions,
    pub created_at: u64,
    pub last_accessed: u64,
    pub reference_count: usize,
    pub name: Option<String>,
}

impl IpcObjectMetadata {
    pub fn new(object_type: IpcObjectType, owner: ProcessId, permissions: IpcPermissions) -> Self {
        let now = time::get_ticks();
        Self {
            id: IpcId::new(),
            object_type,
            owner,
            permissions,
            created_at: now,
            last_accessed: now,
            reference_count: 1,
            name: None,
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn can_access(&self, process: ProcessId, write_access: bool) -> bool {
        if self.permissions.owner_only && process != self.owner {
            return false;
        }

        if write_access && !self.permissions.write {
            return false;
        }

        if !write_access && !self.permissions.read {
            return false;
        }

        true
    }
}

// ========== PIPES ==========

const PIPE_BUFFER_SIZE: usize = 4096;

/// Pipe implementation
#[derive(Debug)]
pub struct Pipe {
    pub metadata: IpcObjectMetadata,
    buffer: Mutex<VecDeque<u8>>,
    readers: AtomicUsize,
    writers: AtomicUsize,
    waiting_readers: Mutex<Vec<ProcessId>>,
    waiting_writers: Mutex<Vec<ProcessId>>,
}

impl Pipe {
    pub fn new(owner: ProcessId) -> Self {
        Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::Pipe, owner, IpcPermissions::READ_WRITE),
            buffer: Mutex::new(VecDeque::with_capacity(PIPE_BUFFER_SIZE)),
            readers: AtomicUsize::new(0),
            writers: AtomicUsize::new(0),
            waiting_readers: Mutex::new(Vec::new()),
            waiting_writers: Mutex::new(Vec::new()),
        }
    }

    pub fn read(&self, process: ProcessId, buffer: &mut [u8]) -> Result<usize, IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut pipe_buffer = self.buffer.lock();

        if pipe_buffer.is_empty() {
            if self.writers.load(Ordering::Relaxed) == 0 {
                return Err(IpcError::BrokenPipe);
            }

            // Block the reader
            self.waiting_readers.lock().push(process);
            return Err(IpcError::WouldBlock);
        }

        let bytes_to_read = core::cmp::min(buffer.len(), pipe_buffer.len());
        for i in 0..bytes_to_read {
            buffer[i] = pipe_buffer.pop_front().unwrap();
        }

        // Wake up waiting writers
        let waiting_writers = self.waiting_writers.lock();
        for &writer_pid in waiting_writers.iter() {
            // Signal writer process
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(writer_pid);
            }
        }

        Ok(bytes_to_read)
    }

    pub fn write(&self, process: ProcessId, data: &[u8]) -> Result<usize, IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        if data.is_empty() {
            return Ok(0);
        }

        if self.readers.load(Ordering::Relaxed) == 0 {
            return Err(IpcError::BrokenPipe);
        }

        let mut pipe_buffer = self.buffer.lock();

        if pipe_buffer.len() + data.len() > PIPE_BUFFER_SIZE {
            // Block the writer
            self.waiting_writers.lock().push(process);
            return Err(IpcError::WouldBlock);
        }

        for &byte in data {
            pipe_buffer.push_back(byte);
        }

        // Wake up waiting readers
        let waiting_readers = self.waiting_readers.lock();
        for &reader_pid in waiting_readers.iter() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(reader_pid);
            }
        }

        Ok(data.len())
    }

    pub fn add_reader(&self) {
        self.readers.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove_reader(&self) {
        self.readers.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn add_writer(&self) {
        self.writers.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove_writer(&self) {
        self.writers.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn is_readable(&self) -> bool {
        !self.buffer.lock().is_empty() || self.writers.load(Ordering::Relaxed) == 0
    }

    pub fn is_writable(&self) -> bool {
        self.buffer.lock().len() < PIPE_BUFFER_SIZE && self.readers.load(Ordering::Relaxed) > 0
    }
}

// ========== MESSAGE QUEUES ==========

const MAX_MESSAGE_SIZE: usize = 8192;
const MAX_QUEUE_SIZE: usize = 100;

/// Message with priority
#[derive(Debug, Clone)]
pub struct Message {
    pub data: Vec<u8>,
    pub priority: u32,
    pub sender: ProcessId,
    pub timestamp: u64,
    pub message_type: u32,
}

impl Message {
    pub fn new(data: Vec<u8>, priority: u32, sender: ProcessId, message_type: u32) -> Self {
        Self {
            data,
            priority,
            sender,
            timestamp: time::get_ticks(),
            message_type,
        }
    }
}

/// Message queue implementation with priority support
#[derive(Debug)]
pub struct MessageQueue {
    pub metadata: IpcObjectMetadata,
    messages: Mutex<Vec<Message>>,
    max_messages: usize,
    max_message_size: usize,
    waiting_receivers: Mutex<Vec<ProcessId>>,
    waiting_senders: Mutex<Vec<ProcessId>>,
}

impl MessageQueue {
    pub fn new(owner: ProcessId, max_messages: usize, max_message_size: usize) -> Self {
        Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::MessageQueue, owner, IpcPermissions::READ_WRITE),
            messages: Mutex::new(Vec::with_capacity(max_messages)),
            max_messages,
            max_message_size,
            waiting_receivers: Mutex::new(Vec::new()),
            waiting_senders: Mutex::new(Vec::new()),
        }
    }

    pub fn send(&self, process: ProcessId, data: Vec<u8>, priority: u32, message_type: u32) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        if data.len() > self.max_message_size {
            return Err(IpcError::MessageTooLarge);
        }

        let mut messages = self.messages.lock();

        if messages.len() >= self.max_messages {
            self.waiting_senders.lock().push(process);
            return Err(IpcError::QueueFull);
        }

        let message = Message::new(data, priority, process, message_type);

        // Insert message in priority order (higher priority first)
        let insert_pos = messages
            .iter()
            .position(|m| m.priority < priority)
            .unwrap_or(messages.len());
        messages.insert(insert_pos, message);

        // Wake up waiting receivers
        let waiting_receivers = self.waiting_receivers.lock();
        for &receiver_pid in waiting_receivers.iter() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(receiver_pid);
            }
        }

        Ok(())
    }

    pub fn receive(&self, process: ProcessId) -> Result<Message, IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut messages = self.messages.lock();

        if messages.is_empty() {
            self.waiting_receivers.lock().push(process);
            return Err(IpcError::QueueEmpty);
        }

        let message = messages.remove(0); // Remove highest priority message

        // Wake up waiting senders
        let waiting_senders = self.waiting_senders.lock();
        for &sender_pid in waiting_senders.iter() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(sender_pid);
            }
        }

        Ok(message)
    }

    pub fn peek(&self, process: ProcessId) -> Result<Message, IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let messages = self.messages.lock();

        if messages.is_empty() {
            return Err(IpcError::QueueEmpty);
        }

        Ok(messages[0].clone())
    }

    pub fn len(&self) -> usize {
        self.messages.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.lock().is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.messages.lock().len() >= self.max_messages
    }
}

// ========== SHARED MEMORY ==========

/// Shared memory segment
#[derive(Debug)]
pub struct SharedMemory {
    pub metadata: IpcObjectMetadata,
    pub virtual_address: VirtAddr,
    pub size: usize,
    attached_processes: Mutex<Vec<ProcessId>>,
}

impl SharedMemory {
    pub fn new(owner: ProcessId, size: usize, permissions: IpcPermissions) -> Result<Self, IpcError> {
        // Align size to page boundary
        let aligned_size = (size + 4095) & !4095;

        // Allocate memory
        let memory_protection = MemoryProtection {
            readable: permissions.read,
            writable: permissions.write,
            executable: permissions.execute,
            user_accessible: true,
        };

        let virtual_address = allocate_memory(aligned_size, MemoryRegionType::SharedMemory, memory_protection)
            .map_err(|_| IpcError::OutOfMemory)?;

        Ok(Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::SharedMemory, owner, permissions),
            virtual_address,
            size: aligned_size,
            attached_processes: Mutex::new(Vec::new()),
        })
    }

    pub fn attach(&self, process: ProcessId) -> Result<VirtAddr, IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut attached = self.attached_processes.lock();

        if attached.contains(&process) {
            return Ok(self.virtual_address);
        }

        attached.push(process);

        // In a real implementation, this would map the shared memory
        // into the process's address space
        Ok(self.virtual_address)
    }

    pub fn detach(&self, process: ProcessId) -> Result<(), IpcError> {
        let mut attached = self.attached_processes.lock();
        attached.retain(|&pid| pid != process);

        // Unmap from process address space in real implementation
        Ok(())
    }

    pub fn attached_count(&self) -> usize {
        self.attached_processes.lock().len()
    }

    pub fn is_attached(&self, process: ProcessId) -> bool {
        self.attached_processes.lock().contains(&process)
    }
}

// ========== SEMAPHORES ==========

/// Semaphore for synchronization
#[derive(Debug)]
pub struct Semaphore {
    pub metadata: IpcObjectMetadata,
    count: Mutex<i32>,
    max_count: i32,
    waiting_processes: Mutex<VecDeque<ProcessId>>,
}

impl Semaphore {
    pub fn new(owner: ProcessId, initial_count: i32, max_count: i32) -> Self {
        Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::Semaphore, owner, IpcPermissions::READ_WRITE),
            count: Mutex::new(initial_count),
            max_count,
            waiting_processes: Mutex::new(VecDeque::new()),
        }
    }

    pub fn wait(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut count = self.count.lock();

        if *count > 0 {
            *count -= 1;
            Ok(())
        } else {
            self.waiting_processes.lock().push_back(process);
            Err(IpcError::WouldBlock)
        }
    }

    pub fn try_wait(&self, process: ProcessId) -> Result<bool, IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut count = self.count.lock();

        if *count > 0 {
            *count -= 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn post(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        let mut count = self.count.lock();

        if *count >= self.max_count {
            return Err(IpcError::InvalidArgument);
        }

        *count += 1;

        // Wake up a waiting process
        if let Some(waiting_pid) = self.waiting_processes.lock().pop_front() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(waiting_pid);
            }
        }

        Ok(())
    }

    pub fn get_count(&self) -> i32 {
        *self.count.lock()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting_processes.lock().len()
    }
}

// ========== MUTEXES ==========

/// Mutex for mutual exclusion
#[derive(Debug)]
pub struct IpcMutex {
    pub metadata: IpcObjectMetadata,
    locked: Mutex<bool>,
    owner: Mutex<Option<ProcessId>>,
    waiting_processes: Mutex<VecDeque<ProcessId>>,
}

impl IpcMutex {
    pub fn new(owner: ProcessId) -> Self {
        Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::Mutex, owner, IpcPermissions::READ_WRITE),
            locked: Mutex::new(false),
            owner: Mutex::new(None),
            waiting_processes: Mutex::new(VecDeque::new()),
        }
    }

    pub fn lock(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        let mut locked = self.locked.lock();

        if !*locked {
            *locked = true;
            *self.owner.lock() = Some(process);
            Ok(())
        } else {
            // Check for deadlock (process trying to lock mutex it already owns)
            if let Some(current_owner) = *self.owner.lock() {
                if current_owner == process {
                    return Err(IpcError::Deadlock);
                }
            }

            self.waiting_processes.lock().push_back(process);
            Err(IpcError::WouldBlock)
        }
    }

    pub fn try_lock(&self, process: ProcessId) -> Result<bool, IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        let mut locked = self.locked.lock();

        if !*locked {
            *locked = true;
            *self.owner.lock() = Some(process);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn unlock(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        let mut locked = self.locked.lock();
        let mut owner = self.owner.lock();

        if !*locked {
            return Err(IpcError::InvalidArgument);
        }

        if let Some(current_owner) = *owner {
            if current_owner != process {
                return Err(IpcError::PermissionDenied);
            }
        }

        *locked = false;
        *owner = None;

        // Wake up next waiting process
        if let Some(waiting_pid) = self.waiting_processes.lock().pop_front() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(waiting_pid);
            }
        }

        Ok(())
    }

    pub fn is_locked(&self) -> bool {
        *self.locked.lock()
    }

    pub fn get_owner(&self) -> Option<ProcessId> {
        *self.owner.lock()
    }
}

// ========== EVENTS ==========

/// Event object for notifications
#[derive(Debug)]
pub struct Event {
    pub metadata: IpcObjectMetadata,
    signaled: Mutex<bool>,
    auto_reset: bool,
    waiting_processes: Mutex<Vec<ProcessId>>,
}

impl Event {
    pub fn new(owner: ProcessId, auto_reset: bool, initially_signaled: bool) -> Self {
        Self {
            metadata: IpcObjectMetadata::new(IpcObjectType::Event, owner, IpcPermissions::READ_WRITE),
            signaled: Mutex::new(initially_signaled),
            auto_reset,
            waiting_processes: Mutex::new(Vec::new()),
        }
    }

    pub fn wait(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, false) {
            return Err(IpcError::PermissionDenied);
        }

        let mut signaled = self.signaled.lock();

        if *signaled {
            if self.auto_reset {
                *signaled = false;
            }
            Ok(())
        } else {
            self.waiting_processes.lock().push(process);
            Err(IpcError::WouldBlock)
        }
    }

    pub fn signal(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        *self.signaled.lock() = true;

        // Wake up waiting processes
        let mut waiting = self.waiting_processes.lock();
        for &waiting_pid in waiting.iter() {
            if let Some(pm) = get_process_manager() {
                let _ = pm.unblock_process(waiting_pid);
            }
        }

        if !waiting.is_empty() && self.auto_reset {
            *self.signaled.lock() = false;
        }

        waiting.clear();

        Ok(())
    }

    pub fn reset(&self, process: ProcessId) -> Result<(), IpcError> {
        if !self.metadata.can_access(process, true) {
            return Err(IpcError::PermissionDenied);
        }

        *self.signaled.lock() = false;
        Ok(())
    }

    pub fn is_signaled(&self) -> bool {
        *self.signaled.lock()
    }
}

// ========== IPC MANAGER ==========

/// Global IPC object storage and management
#[derive(Debug)]
pub enum IpcObject {
    Pipe(Box<Pipe>),
    MessageQueue(Box<MessageQueue>),
    SharedMemory(Box<SharedMemory>),
    Semaphore(Box<Semaphore>),
    Mutex(Box<IpcMutex>),
    Event(Box<Event>),
}

impl IpcObject {
    pub fn metadata(&self) -> &IpcObjectMetadata {
        match self {
            IpcObject::Pipe(pipe) => &pipe.metadata,
            IpcObject::MessageQueue(mq) => &mq.metadata,
            IpcObject::SharedMemory(shm) => &shm.metadata,
            IpcObject::Semaphore(sem) => &sem.metadata,
            IpcObject::Mutex(mutex) => &mutex.metadata,
            IpcObject::Event(event) => &event.metadata,
        }
    }

    pub fn object_type(&self) -> IpcObjectType {
        self.metadata().object_type
    }
}

/// IPC system manager
#[derive(Debug)]
pub struct IpcManager {
    objects: RwLock<BTreeMap<IpcId, IpcObject>>,
    named_objects: RwLock<BTreeMap<String, IpcId>>,
    process_objects: RwLock<BTreeMap<ProcessId, Vec<IpcId>>>,
    stats: Mutex<IpcStats>,
}

#[derive(Debug, Default, Clone)]
pub struct IpcStats {
    pub total_objects: usize,
    pub pipes: usize,
    pub message_queues: usize,
    pub shared_memory_segments: usize,
    pub semaphores: usize,
    pub mutexes: usize,
    pub events: usize,
    pub total_messages_sent: u64,
    pub total_bytes_transferred: u64,
}

impl IpcManager {
    pub fn new() -> Self {
        Self {
            objects: RwLock::new(BTreeMap::new()),
            named_objects: RwLock::new(BTreeMap::new()),
            process_objects: RwLock::new(BTreeMap::new()),
            stats: Mutex::new(IpcStats::default()),
        }
    }

    /// Create a new pipe
    pub fn create_pipe(&self, owner: ProcessId) -> Result<IpcId, IpcError> {
        let pipe = Box::new(Pipe::new(owner));
        let id = pipe.metadata.id;

        self.objects.write().insert(id, IpcObject::Pipe(pipe));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.pipes += 1;

        Ok(id)
    }

    /// Create a new message queue
    pub fn create_message_queue(&self, owner: ProcessId, max_messages: usize, max_message_size: usize) -> Result<IpcId, IpcError> {
        let mq = Box::new(MessageQueue::new(owner, max_messages, max_message_size));
        let id = mq.metadata.id;

        self.objects.write().insert(id, IpcObject::MessageQueue(mq));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.message_queues += 1;

        Ok(id)
    }

    /// Create shared memory segment
    pub fn create_shared_memory(&self, owner: ProcessId, size: usize, permissions: IpcPermissions) -> Result<IpcId, IpcError> {
        let shm = Box::new(SharedMemory::new(owner, size, permissions)?);
        let id = shm.metadata.id;

        self.objects.write().insert(id, IpcObject::SharedMemory(shm));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.shared_memory_segments += 1;

        Ok(id)
    }

    /// Create a semaphore
    pub fn create_semaphore(&self, owner: ProcessId, initial_count: i32, max_count: i32) -> Result<IpcId, IpcError> {
        let sem = Box::new(Semaphore::new(owner, initial_count, max_count));
        let id = sem.metadata.id;

        self.objects.write().insert(id, IpcObject::Semaphore(sem));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.semaphores += 1;

        Ok(id)
    }

    /// Create a mutex
    pub fn create_mutex(&self, owner: ProcessId) -> Result<IpcId, IpcError> {
        let mutex = Box::new(IpcMutex::new(owner));
        let id = mutex.metadata.id;

        self.objects.write().insert(id, IpcObject::Mutex(mutex));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.mutexes += 1;

        Ok(id)
    }

    /// Create an event
    pub fn create_event(&self, owner: ProcessId, auto_reset: bool, initially_signaled: bool) -> Result<IpcId, IpcError> {
        let event = Box::new(Event::new(owner, auto_reset, initially_signaled));
        let id = event.metadata.id;

        self.objects.write().insert(id, IpcObject::Event(event));
        self.process_objects.write().entry(owner).or_insert_with(Vec::new).push(id);

        let mut stats = self.stats.lock();
        stats.total_objects += 1;
        stats.events += 1;

        Ok(id)
    }

    /// Destroy an IPC object
    pub fn destroy_object(&self, id: IpcId, process: ProcessId) -> Result<(), IpcError> {
        let mut objects = self.objects.write();

        if let Some(object) = objects.get(&id) {
            if !object.metadata().can_access(process, true) {
                return Err(IpcError::PermissionDenied);
            }
        } else {
            return Err(IpcError::NotFound);
        }

        objects.remove(&id);

        // Remove from process objects
        if let Some(process_objs) = self.process_objects.write().get_mut(&process) {
            process_objs.retain(|&obj_id| obj_id != id);
        }

        // Remove from named objects if it has a name
        let mut named_objects = self.named_objects.write();
        let keys_to_remove: Vec<String> = named_objects
            .iter()
            .filter(|(_, &obj_id)| obj_id == id)
            .map(|(name, _)| name.clone())
            .collect();

        for key in keys_to_remove {
            named_objects.remove(&key);
        }

        let mut stats = self.stats.lock();
        stats.total_objects = stats.total_objects.saturating_sub(1);

        Ok(())
    }

    /// Get object by ID
    pub fn get_object(&self, id: IpcId) -> Option<IpcObjectMetadata> {
        self.objects.read().get(&id).map(|obj| obj.metadata().clone())
    }

    /// Find object by name
    pub fn find_named_object(&self, name: &str) -> Option<IpcId> {
        self.named_objects.read().get(name).copied()
    }

    /// Set object name
    pub fn set_object_name(&self, id: IpcId, name: String, process: ProcessId) -> Result<(), IpcError> {
        let objects = self.objects.read();
        if let Some(object) = objects.get(&id) {
            if !object.metadata().can_access(process, true) {
                return Err(IpcError::PermissionDenied);
            }
        } else {
            return Err(IpcError::NotFound);
        }

        if self.named_objects.read().contains_key(&name) {
            return Err(IpcError::AlreadyExists);
        }

        self.named_objects.write().insert(name, id);
        Ok(())
    }

    /// List objects owned by a process
    pub fn list_process_objects(&self, process: ProcessId) -> Vec<IpcId> {
        self.process_objects.read()
            .get(&process)
            .map(|objs| objs.clone())
            .unwrap_or_default()
    }

    /// Get IPC statistics
    pub fn get_stats(&self) -> IpcStats {
        self.stats.lock().clone()
    }

    /// Cleanup orphaned objects
    pub fn cleanup_orphaned_objects(&self) {
        let objects = self.objects.read();
        let process_manager = get_process_manager();

        if let Some(pm) = process_manager {
            let active_processes: Vec<ProcessId> = pm.list_processes()
                .into_iter()
                .map(|(pid, _, _, _)| pid)
                .collect();

            // Find objects with dead owners
            let orphaned_objects: Vec<IpcId> = objects
                .values()
                .filter(|obj| !active_processes.contains(&obj.metadata().owner))
                .map(|obj| obj.metadata().id)
                .collect();

            drop(objects); // Release read lock

            // Clean up orphaned objects
            for id in orphaned_objects {
                let _ = self.destroy_object(id, ProcessId::kernel());
            }
        }
    }
}

// Global IPC manager
lazy_static! {
    static ref IPC_MANAGER: RwLock<Option<IpcManager>> = RwLock::new(None);
}

/// Initialize the IPC system
pub fn init() {
    let manager = IpcManager::new();
    *IPC_MANAGER.write() = Some(manager);

    crate::println!("Inter-Process Communication system initialized");
    crate::println!("IPC Objects: Pipes, Message Queues, Shared Memory, Semaphores, Mutexes, Events");
}

/// Get global IPC manager
pub fn get_ipc_manager() -> Option<&'static IpcManager> {
    unsafe {
        IPC_MANAGER.read().as_ref().map(|mgr| core::mem::transmute(mgr))
    }
}

// High-level IPC functions
pub fn create_pipe(owner: ProcessId) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_pipe(owner)
}

pub fn create_message_queue(owner: ProcessId, max_messages: usize) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_message_queue(owner, max_messages, MAX_MESSAGE_SIZE)
}

pub fn create_shared_memory(owner: ProcessId, size: usize) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_shared_memory(owner, size, IpcPermissions::READ_WRITE)
}

pub fn create_semaphore(owner: ProcessId, initial_count: i32) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_semaphore(owner, initial_count, i32::MAX)
}

pub fn create_mutex(owner: ProcessId) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_mutex(owner)
}

pub fn create_event(owner: ProcessId, auto_reset: bool) -> Result<IpcId, IpcError> {
    get_ipc_manager()
        .ok_or(IpcError::NotFound)?
        .create_event(owner, auto_reset, false)
}

/// Demonstrate IPC functionality
pub fn demonstrate_ipc() {
    crate::println!("=== Inter-Process Communication Demonstration ===");

    if let Some(manager) = get_ipc_manager() {
        let owner = ProcessId::kernel();

        // Create various IPC objects
        let pipe_id = manager.create_pipe(owner).expect("Failed to create pipe");
        let mq_id = manager.create_message_queue(owner, 10, 1024).expect("Failed to create message queue");
        let shm_id = manager.create_shared_memory(owner, 4096, IpcPermissions::READ_WRITE).expect("Failed to create shared memory");
        let sem_id = manager.create_semaphore(owner, 1, 10).expect("Failed to create semaphore");
        let mutex_id = manager.create_mutex(owner).expect("Failed to create mutex");
        let event_id = manager.create_event(owner, true, false).expect("Failed to create event");

        crate::println!("Created IPC objects:");
        crate::println!("  Pipe: {}", pipe_id);
        crate::println!("  Message Queue: {}", mq_id);
        crate::println!("  Shared Memory: {}", shm_id);
        crate::println!("  Semaphore: {}", sem_id);
        crate::println!("  Mutex: {}", mutex_id);
        crate::println!("  Event: {}", event_id);

        // Show statistics
        let stats = manager.get_stats();
        crate::println!("IPC Statistics:");
        crate::println!("  Total objects: {}", stats.total_objects);
        crate::println!("  Pipes: {}", stats.pipes);
        crate::println!("  Message queues: {}", stats.message_queues);
        crate::println!("  Shared memory segments: {}", stats.shared_memory_segments);
        crate::println!("  Semaphores: {}", stats.semaphores);
        crate::println!("  Mutexes: {}", stats.mutexes);
        crate::println!("  Events: {}", stats.events);
    }

    crate::println!("IPC demonstration complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_ipc_id_generation() {
        let id1 = IpcId::new();
        let id2 = IpcId::new();
        assert_ne!(id1, id2);
        assert!(id1.as_u64() > 0);
        assert!(id2.as_u64() > 0);
    }

    #[test_case]
    fn test_pipe_creation() {
        let owner = ProcessId::kernel();
        let pipe = Pipe::new(owner);
        assert_eq!(pipe.metadata.object_type, IpcObjectType::Pipe);
        assert_eq!(pipe.metadata.owner, owner);
    }

    #[test_case]
    fn test_message_queue_creation() {
        let owner = ProcessId::kernel();
        let mq = MessageQueue::new(owner, 10, 1024);
        assert_eq!(mq.metadata.object_type, IpcObjectType::MessageQueue);
        assert_eq!(mq.max_messages, 10);
        assert_eq!(mq.max_message_size, 1024);
    }

    #[test_case]
    fn test_ipc_manager_creation() {
        let manager = IpcManager::new();
        let stats = manager.get_stats();
        assert_eq!(stats.total_objects, 0);
    }
}
