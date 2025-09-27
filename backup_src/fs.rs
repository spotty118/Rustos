//! Virtual File System (VFS) for RustOS
//!
//! This module provides:
//! - Virtual file system abstraction layer
//! - Basic file operations (open, close, read, write)
//! - Directory management
//! - File metadata and attributes
//! - Mount point management
//! - Integration with storage devices

use alloc::vec::Vec;
use alloc::string::String;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::fmt;
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum path length in the file system
pub const MAX_PATH_LENGTH: usize = 4096;

/// Maximum filename length
pub const MAX_FILENAME_LENGTH: usize = 255;

/// File descriptor type
pub type FileDescriptor = u32;

/// Block size for file system operations
pub const BLOCK_SIZE: usize = 4096;

/// File system error types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileSystemError {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    InvalidPath,
    InvalidDescriptor,
    ReadOnlyFileSystem,
    OutOfSpace,
    DirectoryNotEmpty,
    NotADirectory,
    NotAFile,
    TooManyFiles,
    NameTooLong,
    InvalidOperation,
}

impl fmt::Display for FileSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileSystemError::NotFound => write!(f, "File or directory not found"),
            FileSystemError::PermissionDenied => write!(f, "Permission denied"),
            FileSystemError::AlreadyExists => write!(f, "File or directory already exists"),
            FileSystemError::InvalidPath => write!(f, "Invalid path"),
            FileSystemError::InvalidDescriptor => write!(f, "Invalid file descriptor"),
            FileSystemError::ReadOnlyFileSystem => write!(f, "Read-only file system"),
            FileSystemError::OutOfSpace => write!(f, "No space left on device"),
            FileSystemError::DirectoryNotEmpty => write!(f, "Directory not empty"),
            FileSystemError::NotADirectory => write!(f, "Not a directory"),
            FileSystemError::NotAFile => write!(f, "Not a file"),
            FileSystemError::TooManyFiles => write!(f, "Too many open files"),
            FileSystemError::NameTooLong => write!(f, "Filename too long"),
            FileSystemError::InvalidOperation => write!(f, "Invalid operation"),
        }
    }
}

/// File type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileType {
    Regular,
    Directory,
    SymbolicLink,
    BlockDevice,
    CharacterDevice,
    Pipe,
    Socket,
}

/// File permissions and attributes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilePermissions {
    pub owner_read: bool,
    pub owner_write: bool,
    pub owner_execute: bool,
    pub group_read: bool,
    pub group_write: bool,
    pub group_execute: bool,
    pub other_read: bool,
    pub other_write: bool,
    pub other_execute: bool,
}

impl FilePermissions {
    pub fn new_file() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: false,
            group_read: true,
            group_write: false,
            group_execute: false,
            other_read: true,
            other_write: false,
            other_execute: false,
        }
    }

    pub fn new_directory() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: true,
            group_read: true,
            group_write: false,
            group_execute: true,
            other_read: true,
            other_write: false,
            other_execute: true,
        }
    }

    pub fn to_octal(&self) -> u16 {
        let mut octal = 0;
        if self.owner_read { octal |= 0o400; }
        if self.owner_write { octal |= 0o200; }
        if self.owner_execute { octal |= 0o100; }
        if self.group_read { octal |= 0o040; }
        if self.group_write { octal |= 0o020; }
        if self.group_execute { octal |= 0o010; }
        if self.other_read { octal |= 0o004; }
        if self.other_write { octal |= 0o002; }
        if self.other_execute { octal |= 0o001; }
        octal
    }
}

/// File metadata structure
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub file_type: FileType,
    pub permissions: FilePermissions,
    pub size: u64,
    pub created_time: u64,
    pub modified_time: u64,
    pub accessed_time: u64,
    pub owner_id: u32,
    pub group_id: u32,
    pub device_id: u32,
    pub inode: u64,
    pub link_count: u32,
}

impl FileMetadata {
    pub fn new_file() -> Self {
        let now = crate::time::uptime_ms();
        Self {
            file_type: FileType::Regular,
            permissions: FilePermissions::new_file(),
            size: 0,
            created_time: now,
            modified_time: now,
            accessed_time: now,
            owner_id: 0,
            group_id: 0,
            device_id: 0,
            inode: 0,
            link_count: 1,
        }
    }

    pub fn new_directory() -> Self {
        let now = crate::time::uptime_ms();
        Self {
            file_type: FileType::Directory,
            permissions: FilePermissions::new_directory(),
            size: BLOCK_SIZE as u64,
            created_time: now,
            modified_time: now,
            accessed_time: now,
            owner_id: 0,
            group_id: 0,
            device_id: 0,
            inode: 0,
            link_count: 2, // . and parent reference
        }
    }
}

/// Directory entry structure
#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    pub name: String,
    pub inode: u64,
    pub file_type: FileType,
}

/// Virtual file node (inode)
#[derive(Debug, Clone)]
pub struct VNode {
    pub metadata: FileMetadata,
    pub data: VNodeData,
}

/// Virtual file node data
#[derive(Debug, Clone)]
pub enum VNodeData {
    File(Vec<u8>),
    Directory(Vec<DirectoryEntry>),
    SymbolicLink(String),
    Device { major: u32, minor: u32 },
}

impl VNode {
    pub fn new_file(name: &str) -> Self {
        let mut metadata = FileMetadata::new_file();
        metadata.inode = generate_inode();

        Self {
            metadata,
            data: VNodeData::File(Vec::new()),
        }
    }

    pub fn new_directory(name: &str) -> Self {
        let mut metadata = FileMetadata::new_directory();
        metadata.inode = generate_inode();

        let mut entries = Vec::new();
        entries.push(DirectoryEntry {
            name: ".".to_string(),
            inode: metadata.inode,
            file_type: FileType::Directory,
        });

        Self {
            metadata,
            data: VNodeData::Directory(entries),
        }
    }

    pub fn is_file(&self) -> bool {
        matches!(self.data, VNodeData::File(_))
    }

    pub fn is_directory(&self) -> bool {
        matches!(self.data, VNodeData::Directory(_))
    }

    pub fn get_size(&self) -> u64 {
        match &self.data {
            VNodeData::File(data) => data.len() as u64,
            VNodeData::Directory(entries) => entries.len() as u64 * 32, // Estimate
            VNodeData::SymbolicLink(target) => target.len() as u64,
            VNodeData::Device { .. } => 0,
        }
    }
}

/// File handle for open files
#[derive(Debug)]
pub struct FileHandle {
    pub inode: u64,
    pub position: u64,
    pub flags: OpenFlags,
    pub ref_count: u32,
}

/// File open flags
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenFlags {
    pub read: bool,
    pub write: bool,
    pub append: bool,
    pub create: bool,
    pub truncate: bool,
    pub exclusive: bool,
}

impl OpenFlags {
    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            append: false,
            create: false,
            truncate: false,
            exclusive: false,
        }
    }

    pub fn write_only() -> Self {
        Self {
            read: false,
            write: true,
            append: false,
            create: false,
            truncate: false,
            exclusive: false,
        }
    }

    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            append: false,
            create: false,
            truncate: false,
            exclusive: false,
        }
    }
}

/// Mount point information
#[derive(Debug, Clone)]
pub struct MountPoint {
    pub path: String,
    pub fs_type: String,
    pub device: String,
    pub read_only: bool,
}

/// Virtual File System structure
pub struct VirtualFileSystem {
    root_inode: u64,
    inodes: BTreeMap<u64, VNode>,
    file_handles: BTreeMap<FileDescriptor, FileHandle>,
    mount_points: Vec<MountPoint>,
    next_fd: FileDescriptor,
    next_inode: u64,
}

impl VirtualFileSystem {
    pub fn new() -> Self {
        let mut vfs = Self {
            root_inode: 1,
            inodes: BTreeMap::new(),
            file_handles: BTreeMap::new(),
            mount_points: Vec::new(),
            next_fd: 1,
            next_inode: 2,
        };

        // Create root directory
        let mut root = VNode::new_directory("/");
        root.metadata.inode = 1;
        vfs.inodes.insert(1, root);

        // Create standard directories
        vfs.create_directory("/dev").ok();
        vfs.create_directory("/proc").ok();
        vfs.create_directory("/sys").ok();
        vfs.create_directory("/tmp").ok();
        vfs.create_directory("/var").ok();
        vfs.create_directory("/usr").ok();
        vfs.create_directory("/home").ok();

        // Create device files
        vfs.create_device_file("/dev/null", 1, 3).ok();
        vfs.create_device_file("/dev/zero", 1, 5).ok();
        vfs.create_device_file("/dev/random", 1, 8).ok();

        vfs
    }

    /// Open a file and return a file descriptor
    pub fn open(&mut self, path: &str, flags: OpenFlags) -> Result<FileDescriptor, FileSystemError> {
        let inode = if flags.create {
            match self.resolve_path(path) {
                Ok(inode) => {
                    if flags.exclusive {
                        return Err(FileSystemError::AlreadyExists);
                    }
                    inode
                }
                Err(FileSystemError::NotFound) => {
                    self.create_file(path)?
                }
                Err(e) => return Err(e),
            }
        } else {
            self.resolve_path(path)?
        };

        let node = self.inodes.get(&inode).ok_or(FileSystemError::NotFound)?;

        // Check permissions
        if flags.read && !node.metadata.permissions.owner_read {
            return Err(FileSystemError::PermissionDenied);
        }
        if flags.write && !node.metadata.permissions.owner_write {
            return Err(FileSystemError::PermissionDenied);
        }

        // Truncate file if requested
        if flags.truncate && flags.write {
            self.truncate_file(inode)?;
        }

        let fd = self.next_fd;
        self.next_fd += 1;

        let handle = FileHandle {
            inode,
            position: if flags.append { node.get_size() } else { 0 },
            flags,
            ref_count: 1,
        };

        self.file_handles.insert(fd, handle);
        Ok(fd)
    }

    /// Close a file descriptor
    pub fn close(&mut self, fd: FileDescriptor) -> Result<(), FileSystemError> {
        let handle = self.file_handles.remove(&fd)
            .ok_or(FileSystemError::InvalidDescriptor)?;

        // Update access time
        if let Some(node) = self.inodes.get_mut(&handle.inode) {
            node.metadata.accessed_time = crate::time::uptime_ms();
        }

        Ok(())
    }

    /// Read from a file descriptor
    pub fn read(&mut self, fd: FileDescriptor, buffer: &mut [u8]) -> Result<usize, FileSystemError> {
        let handle = self.file_handles.get_mut(&fd)
            .ok_or(FileSystemError::InvalidDescriptor)?;

        if !handle.flags.read {
            return Err(FileSystemError::PermissionDenied);
        }

        let node = self.inodes.get(&handle.inode)
            .ok_or(FileSystemError::NotFound)?;

        let bytes_read = match &node.data {
            VNodeData::File(data) => {
                let start = handle.position as usize;
                let end = (start + buffer.len()).min(data.len());
                let bytes_to_read = end.saturating_sub(start);

                if bytes_to_read > 0 {
                    buffer[..bytes_to_read].copy_from_slice(&data[start..end]);
                    handle.position += bytes_to_read as u64;
                }
                bytes_to_read
            }
            VNodeData::Device { major: 1, minor: 3 } => {
                // /dev/null - always returns 0 bytes
                0
            }
            VNodeData::Device { major: 1, minor: 5 } => {
                // /dev/zero - fills buffer with zeros
                buffer.fill(0);
                buffer.len()
            }
            VNodeData::Device { major: 1, minor: 8 } => {
                // /dev/random - fills buffer with pseudo-random data
                self.fill_random(buffer);
                buffer.len()
            }
            _ => return Err(FileSystemError::NotAFile),
        };

        Ok(bytes_read)
    }

    /// Write to a file descriptor
    pub fn write(&mut self, fd: FileDescriptor, buffer: &[u8]) -> Result<usize, FileSystemError> {
        let handle = self.file_handles.get_mut(&fd)
            .ok_or(FileSystemError::InvalidDescriptor)?;

        if !handle.flags.write {
            return Err(FileSystemError::PermissionDenied);
        }

        let inode = handle.inode;
        let position = handle.position;

        let node = self.inodes.get_mut(&inode)
            .ok_or(FileSystemError::NotFound)?;

        let bytes_written = match &mut node.data {
            VNodeData::File(data) => {
                let start = position as usize;

                // Extend file if necessary
                if start + buffer.len() > data.len() {
                    data.resize(start + buffer.len(), 0);
                }

                data[start..start + buffer.len()].copy_from_slice(buffer);
                handle.position += buffer.len() as u64;
                node.metadata.modified_time = crate::time::uptime_ms();
                node.metadata.size = data.len() as u64;
                buffer.len()
            }
            VNodeData::Device { major: 1, minor: 3 } => {
                // /dev/null - discards all data
                buffer.len()
            }
            _ => return Err(FileSystemError::NotAFile),
        };

        Ok(bytes_written)
    }

    /// Seek to a position in a file
    pub fn seek(&mut self, fd: FileDescriptor, offset: i64, whence: SeekFrom) -> Result<u64, FileSystemError> {
        let handle = self.file_handles.get_mut(&fd)
            .ok_or(FileSystemError::InvalidDescriptor)?;

        let node = self.inodes.get(&handle.inode)
            .ok_or(FileSystemError::NotFound)?;

        let file_size = node.get_size();

        let new_position = match whence {
            SeekFrom::Start => offset.max(0) as u64,
            SeekFrom::Current => {
                let current = handle.position as i64;
                (current + offset).max(0) as u64
            }
            SeekFrom::End => {
                let end = file_size as i64;
                (end + offset).max(0) as u64
            }
        };

        handle.position = new_position;
        Ok(new_position)
    }

    /// Get file metadata
    pub fn stat(&self, path: &str) -> Result<FileMetadata, FileSystemError> {
        let inode = self.resolve_path(path)?;
        let node = self.inodes.get(&inode).ok_or(FileSystemError::NotFound)?;
        Ok(node.metadata.clone())
    }

    /// Create a new directory
    pub fn create_directory(&mut self, path: &str) -> Result<(), FileSystemError> {
        if self.resolve_path(path).is_ok() {
            return Err(FileSystemError::AlreadyExists);
        }

        let (parent_path, name) = split_path(path)?;
        let parent_inode = self.resolve_path(&parent_path)?;

        let parent = self.inodes.get_mut(&parent_inode)
            .ok_or(FileSystemError::NotFound)?;

        if !parent.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        let new_inode = self.next_inode;
        self.next_inode += 1;

        let mut new_dir = VNode::new_directory(name);
        new_dir.metadata.inode = new_inode;

        // Add parent reference
        if let VNodeData::Directory(entries) = &mut new_dir.data {
            entries.push(DirectoryEntry {
                name: "..".to_string(),
                inode: parent_inode,
                file_type: FileType::Directory,
            });
        }

        // Add to parent directory
        if let VNodeData::Directory(entries) = &mut parent.data {
            entries.push(DirectoryEntry {
                name: name.to_string(),
                inode: new_inode,
                file_type: FileType::Directory,
            });
            parent.metadata.modified_time = crate::time::uptime_ms();
        }

        self.inodes.insert(new_inode, new_dir);
        Ok(())
    }

    /// Remove a directory
    pub fn remove_directory(&mut self, path: &str) -> Result<(), FileSystemError> {
        let inode = self.resolve_path(path)?;
        let node = self.inodes.get(&inode).ok_or(FileSystemError::NotFound)?;

        if !node.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        // Check if directory is empty (only . and .. entries)
        if let VNodeData::Directory(entries) = &node.data {
            if entries.len() > 2 {
                return Err(FileSystemError::DirectoryNotEmpty);
            }
        }

        // Remove from parent directory
        let (parent_path, name) = split_path(path)?;
        let parent_inode = self.resolve_path(&parent_path)?;
        let parent = self.inodes.get_mut(&parent_inode)
            .ok_or(FileSystemError::NotFound)?;

        if let VNodeData::Directory(entries) = &mut parent.data {
            entries.retain(|entry| entry.name != name);
            parent.metadata.modified_time = crate::time::uptime_ms();
        }

        self.inodes.remove(&inode);
        Ok(())
    }

    /// List directory contents
    pub fn list_directory(&self, path: &str) -> Result<Vec<DirectoryEntry>, FileSystemError> {
        let inode = self.resolve_path(path)?;
        let node = self.inodes.get(&inode).ok_or(FileSystemError::NotFound)?;

        match &node.data {
            VNodeData::Directory(entries) => Ok(entries.clone()),
            _ => Err(FileSystemError::NotADirectory),
        }
    }

    /// Create a regular file
    fn create_file(&mut self, path: &str) -> Result<u64, FileSystemError> {
        let (parent_path, name) = split_path(path)?;
        let parent_inode = self.resolve_path(&parent_path)?;

        let parent = self.inodes.get_mut(&parent_inode)
            .ok_or(FileSystemError::NotFound)?;

        if !parent.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        let new_inode = self.next_inode;
        self.next_inode += 1;

        let mut new_file = VNode::new_file(name);
        new_file.metadata.inode = new_inode;

        // Add to parent directory
        if let VNodeData::Directory(entries) = &mut parent.data {
            entries.push(DirectoryEntry {
                name: name.to_string(),
                inode: new_inode,
                file_type: FileType::Regular,
            });
            parent.metadata.modified_time = crate::time::uptime_ms();
        }

        self.inodes.insert(new_inode, new_file);
        Ok(new_inode)
    }

    /// Create a device file
    fn create_device_file(&mut self, path: &str, major: u32, minor: u32) -> Result<(), FileSystemError> {
        let (parent_path, name) = split_path(path)?;
        let parent_inode = self.resolve_path(&parent_path)?;

        let parent = self.inodes.get_mut(&parent_inode)
            .ok_or(FileSystemError::NotFound)?;

        if !parent.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        let new_inode = self.next_inode;
        self.next_inode += 1;

        let mut metadata = FileMetadata::new_file();
        metadata.file_type = FileType::CharacterDevice;
        metadata.inode = new_inode;

        let device_node = VNode {
            metadata,
            data: VNodeData::Device { major, minor },
        };

        // Add to parent directory
        if let VNodeData::Directory(entries) = &mut parent.data {
            entries.push(DirectoryEntry {
                name: name.to_string(),
                inode: new_inode,
                file_type: FileType::CharacterDevice,
            });
            parent.metadata.modified_time = crate::time::uptime_ms();
        }

        self.inodes.insert(new_inode, device_node);
        Ok(())
    }

    /// Truncate a file to zero length
    fn truncate_file(&mut self, inode: u64) -> Result<(), FileSystemError> {
        let node = self.inodes.get_mut(&inode)
            .ok_or(FileSystemError::NotFound)?;

        match &mut node.data {
            VNodeData::File(data) => {
                data.clear();
                node.metadata.size = 0;
                node.metadata.modified_time = crate::time::uptime_ms();
                Ok(())
            }
            _ => Err(FileSystemError::NotAFile),
        }
    }

    /// Resolve a path to an inode
    fn resolve_path(&self, path: &str) -> Result<u64, FileSystemError> {
        if path.len() > MAX_PATH_LENGTH {
            return Err(FileSystemError::InvalidPath);
        }

        let path = normalize_path(path);
        let components = path.split('/').filter(|s| !s.is_empty());

        let mut current_inode = self.root_inode;

        for component in components {
            if component.len() > MAX_FILENAME_LENGTH {
                return Err(FileSystemError::NameTooLong);
            }

            let node = self.inodes.get(&current_inode)
                .ok_or(FileSystemError::NotFound)?;

            match &node.data {
                VNodeData::Directory(entries) => {
                    let entry = entries.iter()
                        .find(|e| e.name == component)
                        .ok_or(FileSystemError::NotFound)?;
                    current_inode = entry.inode;
                }
                _ => return Err(FileSystemError::NotADirectory),
            }
        }

        Ok(current_inode)
    }

    /// Fill buffer with pseudo-random data
    fn fill_random(&self, buffer: &mut [u8]) {
        let mut seed = crate::time::uptime_us() as u32;
        for byte in buffer.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (seed >> 16) as u8;
        }
    }
}

/// Seek position enumeration
#[derive(Debug, Clone, Copy)]
pub enum SeekFrom {
    Start,
    Current,
    End,
}

/// Split a path into parent directory and filename
fn split_path(path: &str) -> Result<(String, &str), FileSystemError> {
    let path = path.trim_end_matches('/');

    if path.is_empty() || path == "/" {
        return Err(FileSystemError::InvalidPath);
    }

    if let Some(pos) = path.rfind('/') {
        let parent = if pos == 0 { "/" } else { &path[..pos] };
        let name = &path[pos + 1..];

        if name.is_empty() {
            return Err(FileSystemError::InvalidPath);
        }

        Ok((parent.to_string(), name))
    } else {
        Ok(("/".to_string(), path))
    }
}

/// Normalize a path by resolving . and .. components
fn normalize_path(path: &str) -> String {
    let mut components = Vec::new();
    let absolute = path.starts_with('/');

    for component in path.split('/') {
        match component {
            "" | "." => continue,
            ".." => {
                if absolute || !components.is_empty() {
                    components.pop();
                }
            }
            comp => components.push(comp),
        }
    }

    if absolute {
        format!("/{}", components.join("/"))
    } else if components.is_empty() {
        ".".to_string()
    } else {
        components.join("/")
    }
}

/// Generate a unique inode number
fn generate_inode() -> u64 {
    static INODE_COUNTER: core::sync::atomic::AtomicU64 =
        core::sync::atomic::AtomicU64::new(1000);
    INODE_COUNTER.fetch_add(1, core::sync::atomic::Ordering::SeqCst)
}

/// Global virtual file system instance
lazy_static! {
    pub static ref VFS: Mutex<VirtualFileSystem> = Mutex::new(VirtualFileSystem::new());
}

/// Initialize the file system
pub fn init() {
    crate::println!("[FS] Virtual File System initialized");
    crate::println!("[FS] Root file system mounted at /");
    crate::println!("[FS] Standard directories created");
    crate::println!("[FS] Device files initialized");
}

/// Open a file
pub fn open(path: &str, flags: OpenFlags) -> Result<FileDescriptor, FileSystemError> {
    VFS.lock().open(path, flags)
}

/// Close a file
pub fn close(fd: FileDescriptor) -> Result<(), FileSystemError> {
    VFS.lock().close(fd)
}

/// Read from a file
pub fn read(fd: FileDescriptor, buffer: &mut [u8]) -> Result<usize, FileSystemError> {
    VFS.lock().read(fd, buffer)
}

/// Write to a file
pub fn write(fd: FileDescriptor, buffer: &[u8]) -> Result<usize, FileSystemError> {
    VFS.lock().write(fd, buffer)
}

/// Seek in a file
pub fn seek(fd: FileDescriptor, offset: i64, whence: SeekFrom) -> Result<u64, FileSystemError> {
    VFS.lock().seek(fd, offset, whence)
}

/// Get file statistics
pub fn stat(path: &str) -> Result<FileMetadata, FileSystemError> {
    VFS.lock().stat(path)
}

/// Create a directory
pub fn mkdir(path: &str) -> Result<(), FileSystemError> {
    VFS.lock().create_directory(path)
}

/// Remove a directory
pub fn rmdir(path: &str) -> Result<(), FileSystemError> {
    VFS.lock().remove_directory(path)
}

/// List directory contents
pub fn readdir(path: &str) -> Result<Vec<DirectoryEntry>, FileSystemError> {
    VFS.lock().list_directory(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_path_normalization() {
        assert_eq!(normalize_path("/a/b/../c"), "/a/c");
        assert_eq!(normalize_path("/a/./b"), "/a/b");
        assert_eq!(normalize_path("a/../b"), "b");
    }

    #[test_case]
    fn test_path_splitting() {
        let (parent, name) = split_path("/home/user/file.txt").unwrap();
        assert_eq!(parent, "/home/user");
        assert_eq!(name, "file.txt");
    }

    #[test_case]
    fn test_file_permissions() {
        let perms = FilePermissions::new_file();
        assert!(perms.owner_read);
        assert!(perms.owner_write);
        assert!(!perms.owner_execute);
    }

    #[test_case]
    fn test_vfs_creation() {
        let vfs = VirtualFileSystem::new();
        assert_eq!(vfs.root_inode, 1);
        assert!(vfs.inodes.contains_key(&1));
    }
}
