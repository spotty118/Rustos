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

    /// sys_fork - Create a new process
    fn sys_fork(&self, _args: &[u64], process_manager: &ProcessManager, current_pid: Pid) -> SyscallResult {
        let parent_process = match process_manager.get_process(current_pid) {
            Some(pcb) => pcb,
            None => return SyscallResult::Error(SyscallError::ProcessNotFound),
        };

        let child_name = "fork_child";
        match process_manager.create_process(child_name, Some(current_pid), parent_process.priority) {
            Ok(child_pid) => SyscallResult::Success(child_pid as u64),
            Err(_) => SyscallResult::Error(SyscallError::OutOfMemory),
        }
    }

    /// sys_exec - Execute a new program
    fn sys_exec(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _program_path = args.get(0).copied().unwrap_or(0);
        // TODO: Implement program loading and execution
        SyscallResult::Error(SyscallError::OperationNotSupported)
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

    /// sys_mmap - Map memory
    fn sys_mmap(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _addr = args.get(0).copied().unwrap_or(0);
        let _length = args.get(1).copied().unwrap_or(0);
        let _prot = args.get(2).copied().unwrap_or(0);
        let _flags = args.get(3).copied().unwrap_or(0);
        let _fd = args.get(4).copied().unwrap_or(0) as i32;
        let _offset = args.get(5).copied().unwrap_or(0);

        // TODO: Implement memory mapping
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_munmap - Unmap memory
    fn sys_munmap(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _addr = args.get(0).copied().unwrap_or(0);
        let _length = args.get(1).copied().unwrap_or(0);

        // TODO: Implement memory unmapping
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_brk - Change data segment size
    fn sys_brk(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _addr = args.get(0).copied().unwrap_or(0);

        // TODO: Implement heap management
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    /// sys_sbrk - Change data segment size incrementally
    fn sys_sbrk(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _increment = args.get(0).copied().unwrap_or(0) as i64;

        // TODO: Implement heap management
        SyscallResult::Error(SyscallError::OperationNotSupported)
    }

    // Inter-process communication

    /// sys_pipe - Create a pipe
    fn sys_pipe(&self, args: &[u64], _process_manager: &ProcessManager, _current_pid: Pid) -> SyscallResult {
        let _pipefd_ptr = args.get(0).copied().unwrap_or(0);

        // TODO: Implement pipe creation
        SyscallResult::Error(SyscallError::OperationNotSupported)
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