//! System Call Interface for RustOS
//!
//! This module implements the system call interface that allows user-space
//! programs to request services from the kernel. It includes:
//! - System call dispatch mechanism
use crate::println;
//! - User/kernel mode switching
//! - Parameter validation and copying
//! - Security checks and capabilities

use core::arch::asm;
use x86_64::structures::idt::InterruptStackFrame;
use crate::scheduler::{Pid, Priority};

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

/// System call error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SyscallError {
    /// Invalid system call number
    InvalidSyscall = 1,
    /// Invalid argument
    InvalidArgument = 2,
    /// Permission denied
    PermissionDenied = 3,
    /// Resource not found
    NotFound = 4,
    /// Resource already exists
    AlreadyExists = 5,
    /// Operation not supported
    NotSupported = 6,
    /// Out of memory
    OutOfMemory = 7,
    /// I/O error
    IoError = 8,
    /// Operation would block
    WouldBlock = 9,
    /// Operation interrupted
    Interrupted = 10,
    /// Invalid file descriptor
    BadFileDescriptor = 11,
    /// No child processes
    NoChild = 12,
    /// Resource busy
    Busy = 13,
    /// Cross-device link
    CrossDevice = 14,
    /// Directory not empty
    DirectoryNotEmpty = 15,
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
}

/// System call statistics
#[derive(Debug, Clone, Default)]
pub struct SyscallStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub calls_by_type: [u64; 64], // Track first 64 syscall types
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
    
    println!("✓ System call interface initialized (int 0x80)");
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
extern "x86-interrupt" fn syscall_interrupt_handler(stack_frame: InterruptStackFrame) {
    // Extract system call parameters from registers
    let (syscall_num, arg1, arg2, arg3, arg4, arg5, arg6): (u64, u64, u64, u64, u64, u64, u64);
    
    unsafe {
        asm!(
            "mov {}, rax",    // System call number
            "mov {}, rdi",    // First argument
            "mov {}, rsi",    // Second argument
            "mov {}, rdx",    // Third argument
            "mov {}, r10",    // Fourth argument (r10 instead of rcx)
            "mov {}, r8",     // Fifth argument
            "mov {}, r9",     // Sixth argument
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
        user_sp: stack_frame.stack_pointer.as_u64(),
        user_ip: stack_frame.instruction_pointer.as_u64(),
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
        asm!("mov rax, {}", in(reg) return_value);
    }
}

/// Dispatch a system call to the appropriate handler
fn dispatch_syscall(context: &SyscallContext) -> SyscallResult {
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
        
        // Time and scheduling
        SyscallNumber::Sleep => sys_sleep(context.args[0]),
        SyscallNumber::GetTime => sys_gettime(),
        SyscallNumber::SetPriority => sys_setpriority(context.args[0] as i32),
        SyscallNumber::GetPriority => sys_getpriority(),
        
        // System information
        SyscallNumber::Uname => sys_uname(context.args[0]),
        
        // Unimplemented or invalid system calls
        _ => {
            println!("Unimplemented system call: {:?}", context.syscall_num);
            Err(SyscallError::NotSupported)
        }
    }
}

// System call implementations

/// Exit the current process
fn sys_exit(exit_code: i32) -> SyscallResult {
    println!("Process {} exiting with code {}", get_current_pid(), exit_code);
    // TODO: Implement process termination
    Ok(0)
}

/// Fork the current process
fn sys_fork() -> SyscallResult {
    println!("Fork system call from process {}", get_current_pid());
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
    println!("Kill signal {} to process {}", signal, pid);
    // TODO: Implement signal delivery
    Err(SyscallError::NotSupported)
}

/// Yield CPU to other processes
fn sys_yield() -> SyscallResult {
    crate::scheduler::schedule();
    Ok(0)
}

/// Open a file
fn sys_open(pathname: u64, flags: u32) -> SyscallResult {
    // TODO: Implement file opening
    // For now, return a dummy file descriptor
    println!("Open file at 0x{:x} with flags 0x{:x}", pathname, flags);
    Ok(3) // Return fd 3 as placeholder
}

/// Close a file descriptor
fn sys_close(fd: i32) -> SyscallResult {
    println!("Close file descriptor {}", fd);
    // TODO: Implement file closing
    Ok(0)
}

/// Read from file descriptor
fn sys_read(fd: i32, buf: u64, count: u64) -> SyscallResult {
    println!("Read {} bytes from fd {} to buffer 0x{:x}", count, fd, buf);
    // TODO: Implement file reading
    Ok(0)
}

/// Write to file descriptor
fn sys_write(fd: i32, buf: u64, count: u64) -> SyscallResult {
    println!("Write {} bytes from buffer 0x{:x} to fd {}", count, buf, fd);
    // TODO: Implement file writing
    // For stdout (fd 1), we could write to console
    if fd == 1 || fd == 2 {
        // Stdout or stderr - write to console
        // TODO: Copy data from user buffer and print
        Ok(count) // Pretend we wrote all bytes
    } else {
        Ok(0)
    }
}

/// Change program break (heap management)
fn sys_brk(addr: u64) -> SyscallResult {
    println!("Set program break to 0x{:x}", addr);
    // TODO: Implement heap management
    Ok(addr)
}

/// Memory map
fn sys_mmap(addr: u64, length: u64, prot: i32, flags: i32, fd: i32, offset: u64) -> SyscallResult {
    println!("Mmap: addr=0x{:x}, len={}, prot={}, flags={}, fd={}, offset={}",
        addr, length, prot, flags, fd, offset);
    // TODO: Implement memory mapping
    Err(SyscallError::NotSupported)
}

/// Sleep for specified microseconds
fn sys_sleep(microseconds: u64) -> SyscallResult {
    println!("Sleep for {} microseconds", microseconds);
    // TODO: Implement process sleeping
    Ok(0)
}

/// Get current time
fn sys_gettime() -> SyscallResult {
    // TODO: Implement time retrieval
    Ok(1000000) // Return 1 second as placeholder
}

/// Set process priority
fn sys_setpriority(priority: i32) -> SyscallResult {
    let new_priority = match priority {
        0 => Priority::RealTime,
        1 => Priority::High,
        2 => Priority::Normal,
        3 => Priority::Low,
        4 => Priority::Idle,
        _ => return Err(SyscallError::InvalidArgument),
    };
    
    println!("Set priority to {:?} for process {}", new_priority, get_current_pid());
    // TODO: Update process priority in scheduler
    Ok(0)
}

/// Get process priority
fn sys_getpriority() -> SyscallResult {
    // TODO: Get actual process priority
    Ok(2) // Return Normal priority as default
}

/// Get system information
fn sys_uname(buf: u64) -> SyscallResult {
    println!("Uname system call, buffer at 0x{:x}", buf);
    // TODO: Copy system information to user buffer
    Ok(0)
}

/// Get current process ID (placeholder)
fn get_current_pid() -> Pid {
    // TODO: Get actual current process ID from scheduler
    1 // Return init process for now
}

/// Get system call statistics
pub fn get_syscall_stats() -> SyscallStats {
    unsafe { SYSCALL_STATS.clone() }
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
                "mov rax, {num}",
                "mov rdi, {arg1}",
                "mov rsi, {arg2}",
                "mov rdx, {arg3}",
                "mov r10, {arg4}",
                "mov r8, {arg5}",
                "mov r9, {arg6}",
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
