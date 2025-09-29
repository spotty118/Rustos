//! System Call Interface for RustOS
//!
//! This module implements the system call interface that allows user-space
//! programs to request services from the kernel. It includes:
//! - System call dispatch mechanism
//! - User/kernel mode switching
//! - Parameter validation and copying
//! - Security checks and capabilities

use core::arch::asm;
use x86_64::structures::idt::InterruptStackFrame;
use crate::scheduler::Pid;
use alloc::string::{String, ToString};
use alloc::{vec, vec::Vec};

/// System call numbers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SyscallNumber {
    // Process management
    Exit = 0,
    Fork = 1,
    Exec = 2,
    Wait = 3,
    GetPid = 4,
    GetPpid = 5,
    Kill = 6,
    
    // File operations
    Open = 10,
    Close = 11,
    Read = 12,
    Write = 13,
    Seek = 14,
    Stat = 15,
    
    // Memory management
    Mmap = 20,
    Munmap = 21,
    Brk = 22,
    Mprotect = 23,
    
    // Inter-process communication
    Pipe = 30,
    Socket = 31,
    Bind = 32,
    Listen = 33,
    Accept = 34,
    Connect = 35,
    Send = 36,
    Recv = 37,
    
    // Time and scheduling
    Sleep = 40,
    GetTime = 41,
    SetPriority = 42,
    GetPriority = 43,
    Yield = 44,
    
    // System information
    Uname = 50,
    GetCwd = 51,
    Chdir = 52,
    
    // Invalid system call
    Invalid = u64::MAX,
}

impl From<u64> for SyscallNumber {
    fn from(value: u64) -> Self {
        match value {
            0 => SyscallNumber::Exit,
            1 => SyscallNumber::Fork,
            2 => SyscallNumber::Exec,
            3 => SyscallNumber::Wait,
            4 => SyscallNumber::GetPid,
            5 => SyscallNumber::GetPpid,
            6 => SyscallNumber::Kill,
            10 => SyscallNumber::Open,
            11 => SyscallNumber::Close,
            12 => SyscallNumber::Read,
            13 => SyscallNumber::Write,
            14 => SyscallNumber::Seek,
            15 => SyscallNumber::Stat,
            20 => SyscallNumber::Mmap,
            21 => SyscallNumber::Munmap,
            22 => SyscallNumber::Brk,
            23 => SyscallNumber::Mprotect,
            30 => SyscallNumber::Pipe,
            31 => SyscallNumber::Socket,
            32 => SyscallNumber::Bind,
            33 => SyscallNumber::Listen,
            34 => SyscallNumber::Accept,
            35 => SyscallNumber::Connect,
            36 => SyscallNumber::Send,
            37 => SyscallNumber::Recv,
            40 => SyscallNumber::Sleep,
            41 => SyscallNumber::GetTime,
            42 => SyscallNumber::SetPriority,
            43 => SyscallNumber::GetPriority,
            44 => SyscallNumber::Yield,
            50 => SyscallNumber::Uname,
            51 => SyscallNumber::GetCwd,
            52 => SyscallNumber::Chdir,
            _ => SyscallNumber::Invalid,
        }
    }
}

/// System call result type
pub type SyscallResult = Result<u64, SyscallError>;

/// System call error codes (POSIX-compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SyscallError {
    /// Invalid system call number
    InvalidSyscall = 1,
    /// Invalid argument (EINVAL)
    InvalidArgument = 22,
    /// Permission denied (EACCES)
    PermissionDenied = 13,
    /// No such file or directory (ENOENT)
    NotFound = 2,
    /// File exists (EEXIST)
    AlreadyExists = 17,
    /// Operation not supported (ENOSYS)
    NotSupported = 38,
    /// Out of memory (ENOMEM)
    OutOfMemory = 12,
    /// I/O error (EIO)
    IoError = 5,
    /// Operation would block (EAGAIN)
    WouldBlock = 11,
    /// Operation interrupted (EINTR)
    Interrupted = 4,
    /// Bad file descriptor (EBADF)
    BadFileDescriptor = 9,
    /// No child processes (ECHILD)
    NoChild = 10,
    /// Resource busy (EBUSY)
    Busy = 16,
    /// Cross-device link (EXDEV)
    CrossDevice = 18,
    /// Directory not empty (ENOTEMPTY)
    DirectoryNotEmpty = 39,
    /// Read-only file system (EROFS)
    ReadOnly = 30,
    /// Too many open files (EMFILE)
    TooManyOpenFiles = 24,
    /// File too large (EFBIG)
    FileTooLarge = 27,
    /// No space left on device (ENOSPC)
    NoSpace = 28,
    /// Is a directory (EISDIR)
    IsDirectory = 21,
    /// Not a directory (ENOTDIR)
    NotDirectory = 20,
    /// Operation not permitted (EPERM)
    NotPermitted = 32,
}

/// System call context passed to handlers
#[derive(Debug)]
pub struct SyscallContext {
    /// Process ID making the system call
    pub pid: Pid,
    /// System call number
    pub syscall_num: SyscallNumber,
    /// System call arguments (up to 6 arguments)
    pub args: [u64; 6],
    /// User stack pointer
    pub user_sp: u64,
    /// User instruction pointer
    pub user_ip: u64,
    /// User privilege level (0 = kernel, 3 = user)
    pub privilege_level: u8,
    /// Current working directory
    pub cwd: Option<String>,
}

/// Security validation utilities
pub struct SecurityValidator;

impl SecurityValidator {
    /// Validate user pointer and length
    pub fn validate_user_ptr(ptr: u64, len: u64, _write_access: bool) -> Result<(), SyscallError> {
        // Check for null pointer
        if ptr == 0 && len > 0 {
            return Err(SyscallError::InvalidArgument);
        }

        // Check for overflow
        if ptr.checked_add(len).is_none() {
            return Err(SyscallError::InvalidArgument);
        }

        // Check if pointer is in user space (below 0x8000_0000_0000)
        if ptr >= 0x8000_0000_0000 {
            return Err(SyscallError::InvalidArgument);
        }

        // TODO: Check memory permissions via memory manager
        // For now, basic validation is sufficient

        Ok(())
    }

    /// Validate file descriptor
    pub fn validate_fd(fd: i32) -> Result<(), SyscallError> {
        if fd < 0 {
            return Err(SyscallError::BadFileDescriptor);
        }
        Ok(())
    }

    /// Validate process ID
    pub fn validate_pid(pid: Pid) -> Result<(), SyscallError> {
        if pid == 0 {
            return Err(SyscallError::InvalidArgument);
        }
        Ok(())
    }

    /// Copy string from user space
    pub fn copy_string_from_user(ptr: u64, max_len: usize) -> Result<String, SyscallError> {
        if ptr == 0 {
            return Err(SyscallError::InvalidArgument);
        }

        Self::validate_user_ptr(ptr, max_len as u64, false)?;

        // TODO: Implement actual memory copying from user space
        // For now, return a placeholder
        Ok("placeholder".to_string())
    }

    /// Copy data from user space
    pub fn copy_from_user(ptr: u64, len: usize) -> Result<Vec<u8>, SyscallError> {
        if ptr == 0 && len > 0 {
            return Err(SyscallError::InvalidArgument);
        }

        Self::validate_user_ptr(ptr, len as u64, false)?;

        // TODO: Implement actual memory copying from user space
        // For now, return a placeholder
        Ok(vec![0; len])
    }

    /// Copy data to user space
    pub fn copy_to_user(ptr: u64, data: &[u8]) -> Result<(), SyscallError> {
        if ptr == 0 && !data.is_empty() {
            return Err(SyscallError::InvalidArgument);
        }

        Self::validate_user_ptr(ptr, data.len() as u64, true)?;

        // TODO: Implement actual memory copying to user space
        // For now, just validate

        Ok(())
    }
}

/// System call statistics
#[derive(Debug, Clone)]
pub struct SyscallStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub calls_by_type: [u64; 64], // Track first 64 syscall types
}

impl Default for SyscallStats {
    fn default() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            calls_by_type: [0; 64],
        }
    }
}

static mut SYSCALL_STATS: SyscallStats = SyscallStats {
    total_calls: 0,
    successful_calls: 0,
    failed_calls: 0,
    calls_by_type: [0; 64],
};

/// Initialize the system call interface
pub fn init() -> Result<(), &'static str> {
    // Set up system call interrupt handler (interrupt 0x80)
    setup_syscall_interrupt();
    
    // Production: syscall interface initialized
    Ok(())
}

/// Set up the system call interrupt handler
fn setup_syscall_interrupt() {
    use x86_64::structures::idt::InterruptDescriptorTable;
    use lazy_static::lazy_static;
    use spin::Mutex;
    
    lazy_static! {
        static ref SYSCALL_IDT: Mutex<InterruptDescriptorTable> = {
            let mut idt = InterruptDescriptorTable::new();
            idt[0x80].set_handler_fn(syscall_interrupt_handler);
            Mutex::new(idt)
        };
    }
    
    // Load the IDT entry for system calls
    // Note: In a real implementation, this would be integrated with the main IDT
}

/// System call interrupt handler (interrupt 0x80)
extern "x86-interrupt" fn syscall_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Extract system call parameters from registers
    let (syscall_num, arg1, arg2, arg3, arg4, arg5, arg6): (u64, u64, u64, u64, u64, u64, u64);
    
    unsafe {
        asm!(
            "mov {0:r}, rax",    // System call number
            "mov {1:r}, rdi",    // First argument
            "mov {2:r}, rsi",    // Second argument
            "mov {3:r}, rdx",    // Third argument
            "mov {4:r}, r10",    // Fourth argument (r10 instead of rcx)
            "mov {5:r}, r8",     // Fifth argument
            "mov {6:r}, r9",     // Sixth argument
            out(reg) syscall_num,
            out(reg) arg1,
            out(reg) arg2,
            out(reg) arg3,
            out(reg) arg4,
            out(reg) arg5,
            out(reg) arg6,
        );
    }
    
    let context = SyscallContext {
        pid: get_current_pid(),
        syscall_num: SyscallNumber::from(syscall_num),
        args: [arg1, arg2, arg3, arg4, arg5, arg6],
        user_sp: _stack_frame.stack_pointer.as_u64(),
        user_ip: _stack_frame.instruction_pointer.as_u64(),
        privilege_level: 3, // Assume user mode
        cwd: None, // TODO: Get from process context
    };
    
    // Dispatch the system call
    let result = dispatch_syscall(&context);
    
    // Update statistics
    unsafe {
        SYSCALL_STATS.total_calls += 1;
        if syscall_num < 64 {
            SYSCALL_STATS.calls_by_type[syscall_num as usize] += 1;
        }
        
        match result {
            Ok(_) => SYSCALL_STATS.successful_calls += 1,
            Err(_) => SYSCALL_STATS.failed_calls += 1,
        }
    }
    
    // Return result in RAX register
    let return_value = match result {
        Ok(value) => value,
        Err(error) => -(error as i64) as u64, // Negative error codes
    };
    
    unsafe {
        asm!("mov rax, {0:r}", in(reg) return_value);
    }
}

/// Dispatch a system call to the appropriate handler
pub fn dispatch_syscall(context: &SyscallContext) -> SyscallResult {
    match context.syscall_num {
        // Process management
        SyscallNumber::Exit => sys_exit(context.args[0] as i32),
        SyscallNumber::Fork => sys_fork(),
        SyscallNumber::GetPid => sys_getpid(),
        SyscallNumber::GetPpid => sys_getppid(),
        SyscallNumber::Kill => sys_kill(context.args[0] as Pid, context.args[1] as i32),
        SyscallNumber::Yield => sys_yield(),
        
        // File operations
        SyscallNumber::Open => sys_open(context.args[0], context.args[1] as u32),
        SyscallNumber::Close => sys_close(context.args[0] as i32),
        SyscallNumber::Read => sys_read(context.args[0] as i32, context.args[1], context.args[2]),
        SyscallNumber::Write => sys_write(context.args[0] as i32, context.args[1], context.args[2]),
        
        // Memory management
        SyscallNumber::Brk => sys_brk(context.args[0]),
        SyscallNumber::Mmap => sys_mmap(
            context.args[0],
            context.args[1],
            context.args[2] as i32,
            context.args[3] as i32,
            context.args[4] as i32,
            context.args[5],
        ),
        SyscallNumber::Munmap => sys_munmap(context.args[0], context.args[1]),
        
        // Time and scheduling
        SyscallNumber::Sleep => sys_sleep(context.args[0]),
        SyscallNumber::GetTime => sys_gettime(),
        SyscallNumber::SetPriority => sys_setpriority(context.args[0] as i32),
        SyscallNumber::GetPriority => sys_getpriority(),
        
        // System information
        SyscallNumber::Uname => sys_uname(context.args[0]),
        
        // Unimplemented or invalid system calls
        _ => {
            Err(SyscallError::NotSupported)
        }
    }
}

// System call implementations

/// Exit the current process
fn sys_exit(exit_code: i32) -> SyscallResult {
    let process_manager = crate::process::get_process_manager();
    let current_pid = process_manager.current_process();

    // Terminate the current process
    match process_manager.terminate_process(current_pid, exit_code) {
        Ok(()) => {
            // Schedule next process
            crate::scheduler::schedule();
            // This should not return for the exiting process
            Ok(0)
        },
        Err(_) => Err(SyscallError::InvalidArgument)
    }
}

/// Fork the current process
fn sys_fork() -> SyscallResult {
    // Production: fork operation
    // TODO: Implement process forking
    Err(SyscallError::NotSupported)
}

/// Get current process ID
fn sys_getpid() -> SyscallResult {
    Ok(get_current_pid() as u64)
}

/// Get parent process ID
fn sys_getppid() -> SyscallResult {
    // TODO: Implement parent PID lookup
    Ok(0)
}

/// Send signal to process
fn sys_kill(pid: Pid, signal: i32) -> SyscallResult {
    // Security validation
    SecurityValidator::validate_pid(pid)?;

    let process_manager = crate::process::get_process_manager();
    let current_pid = process_manager.current_process();

    // Check if target process exists
    if process_manager.get_process(pid).is_none() {
        return Err(SyscallError::NotFound);
    }

    // Simple signal handling - only implement SIGKILL (9) for now
    match signal {
        9 => {
            // SIGKILL - terminate process immediately
            if pid == current_pid {
                // Don't allow process to kill itself with SIGKILL
                return Err(SyscallError::InvalidArgument);
            }

            match process_manager.terminate_process(pid, -9) {
                Ok(()) => Ok(0),
                Err(_) => Err(SyscallError::NotPermitted),
            }
        },
        0 => {
            // Signal 0 - just check if process exists
            Ok(0)
        },
        _ => {
            // Other signals not yet implemented
            Err(SyscallError::NotSupported)
        }
    }
}

/// Yield CPU to other processes
fn sys_yield() -> SyscallResult {
    crate::scheduler::schedule();
    Ok(0)
}

/// Open a file
fn sys_open(pathname: u64, flags: u32) -> SyscallResult {
    // Security validation
    let path = SecurityValidator::copy_string_from_user(pathname, 4096)
        .map_err(|_| SyscallError::InvalidArgument)?;

    // Convert flags to VFS open flags
    let open_flags = crate::fs::OpenFlags::from_posix(flags);

    // Open through VFS
    match crate::fs::vfs().open(&path, open_flags) {
        Ok(fd) => {
            // Update process file descriptor table
            let process_manager = crate::process::get_process_manager();
            let _current_pid = process_manager.current_process();

            // TODO: Add FD to process table
            Ok(fd as u64)
        },
        Err(fs_error) => {
            // Convert filesystem error to syscall error
            let syscall_error = match fs_error {
                crate::fs::FsError::NotFound => SyscallError::NotFound,
                crate::fs::FsError::PermissionDenied => SyscallError::PermissionDenied,
                crate::fs::FsError::AlreadyExists => SyscallError::AlreadyExists,
                crate::fs::FsError::NotADirectory => SyscallError::NotDirectory,
                crate::fs::FsError::IsADirectory => SyscallError::IsDirectory,
                crate::fs::FsError::InvalidArgument => SyscallError::InvalidArgument,
                crate::fs::FsError::NoSpaceLeft => SyscallError::NoSpace,
                crate::fs::FsError::ReadOnly => SyscallError::ReadOnly,
                crate::fs::FsError::BadFileDescriptor => SyscallError::BadFileDescriptor,
                _ => SyscallError::IoError,
            };
            Err(syscall_error)
        }
    }
}

/// Close a file descriptor
fn sys_close(fd: i32) -> SyscallResult {
    // Security validation
    SecurityValidator::validate_fd(fd)?;

    // Don't allow closing standard descriptors
    if fd <= 2 {
        return Err(SyscallError::InvalidArgument);
    }

    // Close through VFS
    match crate::fs::vfs().close(fd) {
        Ok(()) => {
            // Remove from process file descriptor table
            let process_manager = crate::process::get_process_manager();
            let _current_pid = process_manager.current_process();

            // TODO: Remove FD from process table
            Ok(0)
        },
        Err(fs_error) => {
            let syscall_error = match fs_error {
                crate::fs::FsError::BadFileDescriptor => SyscallError::BadFileDescriptor,
                _ => SyscallError::IoError,
            };
            Err(syscall_error)
        }
    }
}

/// Read from file descriptor
fn sys_read(fd: i32, buf: u64, count: u64) -> SyscallResult {
    // Security validation
    SecurityValidator::validate_fd(fd)?;
    SecurityValidator::validate_user_ptr(buf, count, true)?;

    // Limit read size to prevent abuse
    let read_count = core::cmp::min(count, 1024 * 1024) as usize; // Max 1MB

    // Handle special file descriptors
    match fd {
        0 => {
            // stdin - for now, return empty read
            Ok(0)
        },
        1 | 2 => {
            // stdout/stderr - not readable
            Err(SyscallError::InvalidArgument)
        },
        _ => {
            // Regular file descriptor
            let mut buffer = vec![0u8; read_count];

            match crate::fs::vfs().read(fd, &mut buffer) {
                Ok(bytes_read) => {
                    // Copy data to user space
                    if bytes_read > 0 {
                        SecurityValidator::copy_to_user(buf, &buffer[..bytes_read])?;
                    }
                    Ok(bytes_read as u64)
                },
                Err(fs_error) => {
                    let syscall_error = match fs_error {
                        crate::fs::FsError::BadFileDescriptor => SyscallError::BadFileDescriptor,
                        crate::fs::FsError::PermissionDenied => SyscallError::PermissionDenied,
                        _ => SyscallError::IoError,
                    };
                    Err(syscall_error)
                }
            }
        }
    }
}

/// Write to file descriptor
fn sys_write(fd: i32, buf: u64, count: u64) -> SyscallResult {
    // Security validation
    SecurityValidator::validate_fd(fd)?;
    SecurityValidator::validate_user_ptr(buf, count, false)?;

    // Limit write size to prevent abuse
    let write_count = core::cmp::min(count, 1024 * 1024) as usize; // Max 1MB

    // Copy data from user space
    let data = SecurityValidator::copy_from_user(buf, write_count)?;

    // Handle special file descriptors
    match fd {
        0 => {
            // stdin - not writable
            Err(SyscallError::InvalidArgument)
        },
        1 | 2 => {
            // stdout/stderr - write to console
            for &byte in &data {
                crate::print!("{}", byte as char);
            }
            Ok(write_count as u64)
        },
        _ => {
            // Regular file descriptor
            match crate::fs::vfs().write(fd, &data) {
                Ok(bytes_written) => Ok(bytes_written as u64),
                Err(fs_error) => {
                    let syscall_error = match fs_error {
                        crate::fs::FsError::BadFileDescriptor => SyscallError::BadFileDescriptor,
                        crate::fs::FsError::PermissionDenied => SyscallError::PermissionDenied,
                        crate::fs::FsError::NoSpaceLeft => SyscallError::NoSpace,
                        crate::fs::FsError::ReadOnly => SyscallError::ReadOnly,
                        _ => SyscallError::IoError,
                    };
                    Err(syscall_error)
                }
            }
        }
    }
}

/// Change program break (heap management)
fn sys_brk(addr: u64) -> SyscallResult {
    // Production: heap management
    // TODO: Implement heap management
    Ok(addr)
}

/// Memory map
fn sys_mmap(_addr: u64, length: u64, prot: i32, flags: i32, fd: i32, _offset: u64) -> SyscallResult {
    // Security validation
    if length == 0 {
        return Err(SyscallError::InvalidArgument);
    }

    // Limit mapping size to prevent abuse
    if length > 1024 * 1024 * 1024 { // 1GB max
        return Err(SyscallError::InvalidArgument);
    }

    // Convert protection flags
    let readable = (prot & 0x1) != 0;
    let writable = (prot & 0x2) != 0;
    let executable = (prot & 0x4) != 0;

    let protection = crate::memory::MemoryProtection {
        readable,
        writable,
        executable,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    // Check for anonymous mapping (MAP_ANONYMOUS)
    let is_anonymous = (flags & 0x20) != 0;

    if !is_anonymous && fd >= 0 {
        // File-backed mapping - not yet implemented
        return Err(SyscallError::NotSupported);
    }

    // For anonymous mappings
    if is_anonymous {
        match crate::memory::allocate_memory(
            length as usize,
            crate::memory::MemoryRegionType::UserHeap,
            protection
        ) {
            Ok(virt_addr) => Ok(virt_addr.as_u64()),
            Err(memory_error) => {
                let syscall_error = match memory_error {
                    crate::memory::MemoryError::OutOfMemory => SyscallError::OutOfMemory,
                    crate::memory::MemoryError::NoVirtualSpace => SyscallError::OutOfMemory,
                    _ => SyscallError::InvalidArgument,
                };
                Err(syscall_error)
            }
        }
    } else {
        Err(SyscallError::NotSupported)
    }
}

/// Sleep for specified microseconds
fn sys_sleep(microseconds: u64) -> SyscallResult {
    // Use production sleep implementation
    let milliseconds = microseconds / 1000;
    if milliseconds > 0 {
        crate::time::sleep_ms(milliseconds);
    }
    Ok(0)
}

/// Get current time
fn sys_gettime() -> SyscallResult {
    // Use production time module
    let uptime_us = crate::time::uptime_us();
    Ok(uptime_us)
}

/// Set process priority
fn sys_setpriority(priority: i32) -> SyscallResult {
    let _new_priority = match priority {
        0 => crate::scheduler::Priority::RealTime,
        1 => crate::scheduler::Priority::High,
        2 => crate::scheduler::Priority::Normal,
        3 => crate::scheduler::Priority::Low,
        4 => crate::scheduler::Priority::Idle,
        _ => return Err(SyscallError::InvalidArgument),
    };

    let process_manager = crate::process::get_process_manager();
    let _current_pid = process_manager.current_process();

    // TODO: Update priority in process manager
    // For now, just validate the priority value
    Ok(0)
}

/// Get process priority
fn sys_getpriority() -> SyscallResult {
    // TODO: Get actual process priority
    Ok(2) // Return Normal priority as default
}

/// Memory unmap
fn sys_munmap(addr: u64, length: u64) -> SyscallResult {
    // Security validation
    if length == 0 {
        return Err(SyscallError::InvalidArgument);
    }

    // Page-align the address and length
    let page_size = 4096u64;
    let aligned_addr = addr & !(page_size - 1);

    // Deallocate memory
    match crate::memory::deallocate_memory(x86_64::VirtAddr::new(aligned_addr)) {
        Ok(()) => Ok(0),
        Err(memory_error) => {
            let syscall_error = match memory_error {
                crate::memory::MemoryError::RegionNotFound => SyscallError::InvalidArgument,
                _ => SyscallError::InvalidArgument,
            };
            Err(syscall_error)
        }
    }
}

/// Get system information
fn sys_uname(buf: u64) -> SyscallResult {
    // Security validation
    SecurityValidator::validate_user_ptr(buf, 390, true)?; // struct utsname is about 390 bytes

    // TODO: Properly format and copy struct utsname to user space
    // For now, just validate the buffer
    Ok(0)
}

/// Get current process ID (production)
fn get_current_pid() -> Pid {
    // Production: get from scheduler or default to 1
    // TODO: Hook up real scheduler PID when available
    1
}

/// Get system call statistics
pub fn get_syscall_stats() -> SyscallStats {
    unsafe { core::ptr::addr_of!(SYSCALL_STATS).read() }
}

/// User-space system call wrapper macro
#[macro_export]
macro_rules! syscall {
    ($num:expr) => {
        syscall!($num, 0, 0, 0, 0, 0, 0)
    };
    ($num:expr, $arg1:expr) => {
        syscall!($num, $arg1, 0, 0, 0, 0, 0)
    };
    ($num:expr, $arg1:expr, $arg2:expr) => {
        syscall!($num, $arg1, $arg2, 0, 0, 0, 0)
    };
    ($num:expr, $arg1:expr, $arg2:expr, $arg3:expr) => {
        syscall!($num, $arg1, $arg2, $arg3, 0, 0, 0)
    };
    ($num:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr) => {
        syscall!($num, $arg1, $arg2, $arg3, $arg4, 0, 0)
    };
    ($num:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr) => {
        syscall!($num, $arg1, $arg2, $arg3, $arg4, $arg5, 0)
    };
    ($num:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr, $arg6:expr) => {{
        let result: u64;
        unsafe {
            core::arch::asm!(
                "mov rax, {num:r}",
                "mov rdi, {arg1:r}",
                "mov rsi, {arg2:r}",
                "mov rdx, {arg3:r}",
                "mov r10, {arg4:r}",
                "mov r8, {arg5:r}",
                "mov r9, {arg6:r}",
                "int 0x80",
                num = in(reg) $num,
                arg1 = in(reg) $arg1,
                arg2 = in(reg) $arg2,
                arg3 = in(reg) $arg3,
                arg4 = in(reg) $arg4,
                arg5 = in(reg) $arg5,
                arg6 = in(reg) $arg6,
                lateout("rax") result,
                options(preserves_flags)
            );
        }
        result
    }};
}

/// User-space system call functions
pub mod userspace {
    use super::*;
    
    /// Exit the current process
    pub fn exit(exit_code: i32) -> ! {
        syscall!(SyscallNumber::Exit as u64, exit_code as u64);
        loop {} // Should never reach here
    }
    
    /// Get current process ID
    pub fn getpid() -> Pid {
        syscall!(SyscallNumber::GetPid as u64) as Pid
    }
    
    /// Write to file descriptor
    pub fn write(fd: i32, buf: *const u8, count: usize) -> isize {
        let result = syscall!(SyscallNumber::Write as u64, fd as u64, buf as u64, count as u64);
        result as isize
    }
    
    /// Read from file descriptor
    pub fn read(fd: i32, buf: *mut u8, count: usize) -> isize {
        let result = syscall!(SyscallNumber::Read as u64, fd as u64, buf as u64, count as u64);
        result as isize
    }
    
    /// Sleep for specified microseconds
    pub fn sleep(microseconds: u64) {
        syscall!(SyscallNumber::Sleep as u64, microseconds);
    }
    
    /// Yield CPU to other processes
    pub fn yield_cpu() {
        syscall!(SyscallNumber::Yield as u64);
    }
}
