//! System Calls Interface and Dispatcher
//!
//! This module implements the system call interface for RustOS, providing
//! a standardized way for processes to request kernel services.

use super::{Pid, ProcessManager};

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
    Sleep = 6,

    // File I/O
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
    Sbrk = 23,

    // Process communication
    Pipe = 30,
    Signal = 31,
    Kill = 32,

    // System information
    Uname = 40,
    GetTime = 41,
    SetTime = 42,

    // Process control
    SetPriority = 50,
    GetPriority = 51,
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
            6 => SyscallNumber::Sleep,
            10 => SyscallNumber::Open,
            11 => SyscallNumber::Close,
            12 => SyscallNumber::Read,
            13 => SyscallNumber::Write,
            14 => SyscallNumber::Seek,
            15 => SyscallNumber::Stat,
            20 => SyscallNumber::Mmap,
            21 => SyscallNumber::Munmap,
            22 => SyscallNumber::Brk,
            23 => SyscallNumber::Sbrk,
            30 => SyscallNumber::Pipe,
            31 => SyscallNumber::Signal,
            32 => SyscallNumber::Kill,
            40 => SyscallNumber::Uname,
            41 => SyscallNumber::GetTime,
            42 => SyscallNumber::SetTime,
            50 => SyscallNumber::SetPriority,
            51 => SyscallNumber::GetPriority,
            _ => SyscallNumber::Exit, // Default to exit for unknown syscalls
        }
    }
}

/// System call return values
#[derive(Debug, Clone, Copy)]
pub enum SyscallResult {
    Success(u64),
    Error(SyscallError),
}

impl SyscallResult {
    pub fn to_u64(self) -> u64 {
        match self {
            SyscallResult::Success(val) => val,
            SyscallResult::Error(err) => err as u64,
        }
    }
}

/// System call error codes
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum SyscallError {
    InvalidSyscall = 0xFFFFFFFFFFFFFFFF,
    InvalidArgument = 0xFFFFFFFFFFFFFFFE,
    PermissionDenied = 0xFFFFFFFFFFFFFFFD,
    ProcessNotFound = 0xFFFFFFFFFFFFFFFC,
    OutOfMemory = 0xFFFFFFFFFFFFFFFB,
    InvalidFileDescriptor = 0xFFFFFFFFFFFFFFFA,
    FileNotFound = 0xFFFFFFFFFFFFFFF9,
    ResourceBusy = 0xFFFFFFFFFFFFFFF8,
    OperationNotSupported = 0xFFFFFFFFFFFFFFF7,
}

/// File open flags
#[derive(Debug, Clone, Copy)]
pub struct OpenFlags {
    pub read: bool,
    pub write: bool,
    pub create: bool,
    pub truncate: bool,
    pub append: bool,
}

impl From<u64> for OpenFlags {
    fn from(flags: u64) -> Self {
        Self {
            read: (flags & 0x01) != 0,
            write: (flags & 0x02) != 0,
            create: (flags & 0x04) != 0,
            truncate: (flags & 0x08) != 0,
            append: (flags & 0x10) != 0,
        }
    }
}

/// System call dispatcher
pub struct SyscallDispatcher {
    /// System call statistics
    syscall_count: [u64; 64],
    /// Total system calls handled
    total_syscalls: u64,
}

impl SyscallDispatcher {
    /// Create a new system call dispatcher
    pub const fn new() -> Self {
        Self {
            syscall_count: [0; 64],
            total_syscalls: 0,
        }
    }

    /// Dispatch a system call
    pub fn dispatch(&mut self, syscall_number: u64, args: &[u64], process_manager: &ProcessManager) -> Result<u64, &'static str> {
        self.total_syscalls += 1;

        let syscall = SyscallNumber::from(syscall_number);

        // Update statistics
        if (syscall_number as usize) < self.syscall_count.len() {
            self.syscall_count[syscall_number as usize] += 1;
        }

        let current_pid = process_manager.current_process();

        let result = match syscall {
            SyscallNumber::Exit => self.sys_exit(args, process_manager, current_pid),
            SyscallNumber::Fork => self.sys_fork(args, process_manager, current_pid),
            SyscallNumber::Exec => self.sys_exec(args, process_manager, current_pid),
            SyscallNumber::Wait => self.sys_wait(args, process_manager, current_pid),
            SyscallNumber::GetPid => self.sys_getpid(process_manager, current_pid),
            SyscallNumber::GetPpid => self.sys_getppid(process_manager, current_pid),
            SyscallNumber::Sleep => self.sys_sleep(args, process_manager, current_pid),
            SyscallNumber::Open => self.sys_open(args, process_manager, current_pid),
            SyscallNumber::Close => self.sys_close(args, process_manager, current_pid),
            SyscallNumber::Read => self.sys_read(args, process_manager, current_pid),
            SyscallNumber::Write => self.sys_write(args, process_manager, current_pid),
            SyscallNumber::Seek => self.sys_seek(args, process_manager, current_pid),
            SyscallNumber::Stat => self.sys_stat(args, process_manager, current_pid),
            SyscallNumber::Mmap => self.sys_mmap(args, process_manager, current_pid),
            SyscallNumber::Munmap => self.sys_munmap(args, process_manager, current_pid),
            SyscallNumber::Brk => self.sys_brk(args, process_manager, current_pid),
            SyscallNumber::Sbrk => self.sys_sbrk(args, process_manager, current_pid),
            SyscallNumber::Pipe => self.sys_pipe(args, process_manager, current_pid),
            SyscallNumber::Signal => self.sys_signal(args, process_manager, current_pid),
            SyscallNumber::Kill => self.sys_kill(args, process_manager, current_pid),
            SyscallNumber::Uname => self.sys_uname(args, process_manager, current_pid),
            SyscallNumber::GetTime => self.sys_gettime(process_manager),
            SyscallNumber::SetTime => self.sys_settime(args, process_manager, current_pid),
            SyscallNumber::SetPriority => self.sys_setpriority(args, process_manager, current_pid),
            SyscallNumber::GetPriority => self.sys_getpriority(args, process_manager, current_pid),
        };

        match result {
            SyscallResult::Success(val) => Ok(val),
            SyscallResult::Error(_) => Err("System call failed"),
        }
    }

    // Process management system calls

    /// sys_exit - Terminate the calling process
    fn sys_exit(&self, args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        let exit_status = args.get(0).copied().unwrap_or(0) as i32;

        match process_manager.terminate_process(current_pid, exit_status) {
            Ok(()) => SyscallResult::Success(0),
            Err(_) => SyscallResult::Error(SyscallError::ProcessNotFound),
        }
    }

    /// sys_fork - Create a new process with copy-on-write memory
    fn sys_fork(&self, _args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        use crate::process::integration::get_integration_manager;

        let _parent_process = match process_manager.get_process(current_pid) {
            Some(pcb) => pcb,
            None => return SyscallResult::Error(SyscallError::ProcessNotFound),
        };

        // Use production fork implementation with copy-on-write
        let integration_manager = get_integration_manager();
        match integration_manager.fork_process(current_pid) {
            Ok(child_pid) => {
                // Return 0 to child process, child PID to parent
                // For now, we return the child PID (parent perspective)
                SyscallResult::Success(child_pid as u64)
            }
            Err(_) => SyscallResult::Error(SyscallError::OutOfMemory),
        }
    }

    /// sys_exec - Execute a new program
    fn sys_exec(&self, args: &[u64], _process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        use crate::process::integration::get_integration_manager;

        let program_path_ptr = args.get(0).copied().unwrap_or(0);
        if program_path_ptr == 0 {
            return SyscallResult::Error(SyscallError::InvalidArgument);
        }

        // In a real implementation, we would:
        // 1. Validate the program path pointer
        // 2. Load the program from filesystem
        // 3. Parse ELF headers
        // 4. Set up new memory space

        // For now, simulate with a basic implementation
        let program_path = "mock_program";
        let mock_program_data = &[0x7f, 0x45, 0x4c, 0x46]; // ELF magic

        let integration_manager = get_integration_manager();
        match integration_manager.exec_process(current_pid, program_path, mock_program_data) {
            Ok(()) => SyscallResult::Success(0),
            Err(_) => SyscallResult::Error(SyscallError::OperationNotSupported),
        }
    }

    /// sys_wait - Wait for child process to terminate
    fn sys_wait(&self, _args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        // TODO: Implement process waiting
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_getpid - Get process ID
    fn sys_getpid(&self, _process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        SyscallResult::Success(current_pid as u64)
    }

    /// sys_getppid - Get parent process ID
    fn sys_getppid(&self, process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        match process_manager.get_process(current_pid) {
            Some(pcb) => match pcb.parent_pid {
                Some(ppid) => SyscallResult::Success(ppid as u64),
                None => SyscallResult::Success(0), // No parent (probably kernel process)
            },
            None => SyscallResult::Error(SyscallError::ProcessNotFound),
        }
    }

    /// sys_sleep - Sleep for specified time
    fn sys_sleep(&self, args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        let _sleep_time = args.get(0).copied().unwrap_or(0);

        // Block the process temporarily
        match process_manager.block_process(current_pid) {
            Ok(()) => {
                // TODO: Set up timer to unblock process after sleep_time
                SyscallResult::Success(0)
            },
            Err(_) => SyscallResult::Error(SyscallError::ProcessNotFound),
        }
    }

    // File I/O system calls

    /// sys_open - Open a file
    fn sys_open(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _path_ptr = args.get(0).copied().unwrap_or(0);
        let _flags = args.get(1).copied().unwrap_or(0);

        // TODO: Implement file system integration
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_close - Close a file descriptor
    fn sys_close(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _fd = args.get(0).copied().unwrap_or(0) as u32;

        // TODO: Implement file descriptor closing
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_read - Read from a file descriptor
    fn sys_read(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _fd = args.get(0).copied().unwrap_or(0) as u32;
        let _buffer_ptr = args.get(1).copied().unwrap_or(0);
        let _count = args.get(2).copied().unwrap_or(0);

        // TODO: Implement file reading
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_write - Write to a file descriptor
    fn sys_write(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let fd = args.get(0).copied().unwrap_or(0) as u32;
        let _buffer_ptr = args.get(1).copied().unwrap_or(0);
        let count = args.get(2).copied().unwrap_or(0);

        // Handle standard output for now
        if fd == 1 || fd == 2 {
            // TODO: Write to console/terminal
            SyscallResult::Success(count) // Pretend we wrote all bytes
        } else {
            SyscallResult::Error(SyscallError::InvalidFileDescriptor)
        }
    }

    /// sys_seek - Seek in a file
    fn sys_seek(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _fd = args.get(0).copied().unwrap_or(0) as u32;
        let _offset = args.get(1).copied().unwrap_or(0) as i64;
        let _whence = args.get(2).copied().unwrap_or(0) as u32;

        // TODO: Implement file seeking
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_stat - Get file status
    fn sys_stat(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _path_ptr = args.get(0).copied().unwrap_or(0);
        let _stat_buf_ptr = args.get(1).copied().unwrap_or(0);

        // TODO: Implement file stat
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    // Memory management system calls

    /// sys_mmap - Map memory using production memory manager
    fn sys_mmap(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        use crate::memory::{allocate_memory, MemoryRegionType, MemoryProtection};

        let _addr = args.get(0).copied().unwrap_or(0);
        let length = args.get(1).copied().unwrap_or(0);
        let prot = args.get(2).copied().unwrap_or(0);
        let _flags = args.get(3).copied().unwrap_or(0);
        let fd = args.get(4).copied().unwrap_or(0) as i32;
        let _offset = args.get(5).copied().unwrap_or(0);

        if length == 0 {
            return SyscallResult::Error(SyscallError::InvalidArgument);
        }

        // Parse protection flags
        let readable = (prot & 0x1) != 0;
        let writable = (prot & 0x2) != 0;
        let executable = (prot & 0x4) != 0;

        let protection = MemoryProtection {
            readable,
            writable,
            executable,
            user_accessible: true,
            cache_disabled: false,
            write_through: false,
            copy_on_write: false,
            guard_page: false,
        };

        // Determine memory region type
        let region_type = if fd == -1 {
            // Anonymous mapping
            if executable {
                MemoryRegionType::UserCode
            } else {
                MemoryRegionType::UserData
            }
        } else {
            // File mapping (not implemented)
            return SyscallResult::Error(SyscallError::OperationNotSupported);
        };

        // Allocate memory
        match allocate_memory(length as usize, region_type, protection) {
            Ok(virt_addr) => SyscallResult::Success(virt_addr.as_u64()),
            Err(_) => SyscallResult::Error(SyscallError::OutOfMemory),
        }
    }

    /// sys_munmap - Unmap memory using production memory manager
    fn sys_munmap(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        use crate::memory::deallocate_memory;
        use x86_64::VirtAddr;

        let addr = args.get(0).copied().unwrap_or(0);
        let _length = args.get(1).copied().unwrap_or(0);

        if addr == 0 {
            return SyscallResult::Error(SyscallError::InvalidArgument);
        }

        let virt_addr = VirtAddr::new(addr);
        match deallocate_memory(virt_addr) {
            Ok(()) => SyscallResult::Success(0),
            Err(_) => SyscallResult::Error(SyscallError::InvalidArgument),
        }
    }

    /// sys_brk - Change data segment size using production memory manager
    fn sys_brk(&self, args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        use crate::memory::{get_memory_manager, MemoryRegionType, MemoryProtection, PAGE_SIZE};

        let new_brk = args.get(0).copied().unwrap_or(0);

        // Get current process
        let process = match process_manager.get_process(current_pid) {
            Some(pcb) => pcb,
            None => return SyscallResult::Error(SyscallError::ProcessNotFound),
        };

        let memory_manager = match get_memory_manager() {
            Some(mm) => mm,
            None => return SyscallResult::Error(SyscallError::OperationNotSupported),
        };

        let current_heap_end = process.memory.heap_start + process.memory.heap_size;

        if new_brk == 0 {
            // Return current break
            return SyscallResult::Success(current_heap_end);
        }

        if new_brk > current_heap_end {
            // Expand heap
            let expansion_size = new_brk - current_heap_end;
            let aligned_size = ((expansion_size + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64) * PAGE_SIZE as u64;

            match memory_manager.allocate_region(
                aligned_size as usize,
                MemoryRegionType::UserHeap,
                MemoryProtection::USER_DATA
            ) {
                Ok(_) => SyscallResult::Success(new_brk),
                Err(_) => SyscallResult::Error(SyscallError::OutOfMemory),
            }
        } else if new_brk < current_heap_end {
            // Shrink heap (simplified implementation)
            SyscallResult::Success(new_brk)
        } else {
            // No change
            SyscallResult::Success(current_heap_end)
        }
    }

    /// sys_sbrk - Change data segment size incrementally
    fn sys_sbrk(&self, args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        let increment = args.get(0).copied().unwrap_or(0) as i64;

        // Get current process
        let process = match process_manager.get_process(current_pid) {
            Some(pcb) => pcb,
            None => return SyscallResult::Error(SyscallError::ProcessNotFound),
        };

        let current_brk = process.memory.heap_start + process.memory.heap_size;
        let new_brk = if increment >= 0 {
            current_brk + increment as u64
        } else {
            current_brk.saturating_sub((-increment) as u64)
        };

        // Use brk implementation
        match self.sys_brk(&[new_brk], process_manager, current_pid) {
            SyscallResult::Success(_) => SyscallResult::Success(current_brk),
            SyscallResult::Error(e) => SyscallResult::Error(e),
        }
    }

    // Inter-process communication

    /// sys_pipe - Create a pipe
    fn sys_pipe(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let pipefd_ptr = args.get(0).copied().unwrap_or(0);
        
        if pipefd_ptr == 0 {
            return SyscallResult::Error(SyscallError::InvalidArgument);
        }

        // Use production IPC pipe creation
        match crate::ipc::create_pipe(4096) { // 4KB pipe buffer
            Ok(pipe_id) => {
                // In real implementation, would write pipe FDs to user memory
                // Return pipe ID for now
                SyscallResult::Success(pipe_id as u64)
            }
            Err(_) => SyscallResult::Error(SyscallError::OperationNotSupported)
        }
    }

    /// sys_signal - Send signal to process
    fn sys_signal(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _signal = args.get(0).copied().unwrap_or(0) as u32;
        let _handler = args.get(1).copied().unwrap_or(0);

        // TODO: Implement signal handling
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_kill - Send signal to process
    fn sys_kill(&self, args: &[u64], process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let target_pid = args.get(0).copied().unwrap_or(0) as Pid;
        let signal = args.get(1).copied().unwrap_or(0) as u32;

        // Simple implementation: signal 9 (SIGKILL) terminates process
        if signal == 9 {
            match process_manager.terminate_process(target_pid, -1) {
                Ok(()) => SyscallResult::Success(0),
                Err(_) => SyscallResult::Error(SyscallError::ProcessNotFound),
            }
        } else {
            // TODO: Implement other signals
            SyscallResult::Error(SyscallError::OperationNotSupported)
        }
    }

    // System information

    /// sys_uname - Get system information
    fn sys_uname(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _buf_ptr = args.get(0).copied().unwrap_or(0);

        // TODO: Fill in system information structure
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_gettime - Get current time
    fn sys_gettime(&self, _process_manager: &ProcessManager) -> SyscallResult {
        let current_time = super::get_system_time();
        SyscallResult::Success(current_time)
    }

    /// sys_settime - Set system time
    fn sys_settime(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _new_time = args.get(0).copied().unwrap_or(0);

        // TODO: Implement time setting (requires privileges)
        SyscallResult::Error(SyscallError::PermissionDenied)
    }

    // Process control

    /// sys_setpriority - Set process priority
    fn sys_setpriority(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _target_pid = args.get(0).copied().unwrap_or(0) as Pid;
        let _priority = args.get(1).copied().unwrap_or(0) as u8;

        // TODO: Implement priority setting
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_getpriority - Get process priority
    fn sys_getpriority(&self, args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        let target_pid = args.get(0).copied().unwrap_or(current_pid as u64) as Pid;

        match process_manager.get_process(target_pid) {
            Some(pcb) => SyscallResult::Success(pcb.priority as u64),
            None => SyscallResult::Error(SyscallError::ProcessNotFound),
        }
    }

    /// Get system call statistics
    pub fn get_stats(&self) -> (u64, &[u64; 64]) {
        (self.total_syscalls, &self.syscall_count)
    }
}

/// System call handler entry point (called from assembly)
#[no_mangle]
pub extern "C" fn syscall_handler(
    syscall_number: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
    arg6: u64,
) -> u64 {
    let args = [arg1, arg2, arg3, arg4, arg5, arg6];
    let process_manager = super::get_process_manager();

    match process_manager.handle_syscall(syscall_number, &args) {
        Ok(result) => result,
        Err(_) => SyscallError::InvalidSyscall as u64,
    }
}