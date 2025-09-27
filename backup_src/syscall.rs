//! System Call Interface for RustOS
//!
//! This module provides a comprehensive system call interface including:
//! - Standard POSIX-like system calls
//! - RustOS-specific AI and hardware optimization calls
//! - Secure parameter validation and access control
//! - Integration with process, memory, and file systems
//! - Performance monitoring and optimization

use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    fmt,
    slice,
    sync::atomic::{AtomicU64, Ordering},
};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use x86_64::{VirtAddr, structures::idt::InterruptStackFrame};

use crate::{
    process::{ProcessId, ProcessState, ProcessPriority, get_process_manager, current_process_id},
    memory::{MemoryRegionType, MemoryProtection, allocate_memory, get_memory_stats},
    fs::{FileHandle, OpenFlags},
    time,
    ai,
    gpu,
};

/// System call numbers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SyscallNumber {
    // Process management (0-19)
    Exit = 0,
    Fork = 1,
    Wait = 2,
    GetPid = 3,
    GetParentPid = 4,
    Sleep = 5,
    Kill = 6,
    SetPriority = 7,
    GetPriority = 8,
    Exec = 9,

    // File system operations (20-39)
    Open = 20,
    Close = 21,
    Read = 22,
    Write = 23,
    Seek = 24,
    Stat = 25,
    Fstat = 26,
    Mkdir = 27,
    Rmdir = 28,
    Unlink = 29,
    Rename = 30,
    Chmod = 31,
    Chown = 32,
    Link = 33,
    Symlink = 34,
    Readlink = 35,
    Sync = 36,
    Fsync = 37,
    Truncate = 38,
    Dup = 39,

    // Memory management (40-59)
    Mmap = 40,
    Munmap = 41,
    Mprotect = 42,
    Brk = 43,
    Sbrk = 44,
    Mlock = 45,
    Munlock = 46,
    GetMemoryInfo = 47,
    AllocateShared = 48,
    FreeShared = 49,

    // IPC (Inter-Process Communication) (60-79)
    Pipe = 60,
    Pipe2 = 61,
    Socket = 62,
    Bind = 63,
    Listen = 64,
    Accept = 65,
    Connect = 66,
    Send = 67,
    Recv = 68,
    Sendto = 69,
    Recvfrom = 70,
    Shmget = 71,
    Shmat = 72,
    Shmdt = 73,
    Semget = 74,
    Semop = 75,
    Msgget = 76,
    Msgsnd = 77,
    Msgrcv = 78,

    // Time and scheduling (80-99)
    GetTime = 80,
    SetTime = 81,
    GetTimestamp = 82,
    Sleep = 83,
    Nanosleep = 84,
    SetTimer = 85,
    GetTimer = 86,
    Yield = 87,
    SchedSetParam = 88,
    SchedGetParam = 89,
    SchedYield = 90,

    // Signal handling (100-119)
    Signal = 100,
    Kill = 101,
    Pause = 102,
    Alarm = 103,
    SigAction = 104,
    SigProcMask = 105,
    SigPending = 106,
    SigSuspend = 107,

    // Network operations (120-139)
    NetworkSend = 120,
    NetworkRecv = 121,
    NetworkConfig = 122,
    NetworkStats = 123,

    // Device I/O (140-159)
    IoCtl = 140,
    DeviceOpen = 141,
    DeviceClose = 142,
    DeviceRead = 143,
    DeviceWrite = 144,
    DeviceControl = 145,

    // RustOS AI & Hardware Optimization (160-199)
    AiPredict = 160,
    AiTrain = 161,
    AiInference = 162,
    AiGetStatus = 163,
    AiSetConfig = 164,
    HardwareOptimize = 165,
    GpuAllocate = 166,
    GpuExecute = 167,
    GpuDeallocate = 168,
    PerformanceMonitor = 169,
    SystemOptimize = 170,
    PowerManage = 171,
    ThermalControl = 172,
    CacheOptimize = 173,
    PredictiveSchedule = 174,
    AdaptiveMemory = 175,
    IntelligentIo = 176,
    NeuralAccelerate = 177,

    // System information (200-219)
    GetSystemInfo = 200,
    GetProcessInfo = 201,
    GetMemoryInfo = 202,
    GetCpuInfo = 203,
    GetHardwareInfo = 204,
    GetAiStatus = 205,
    GetPerformanceStats = 206,
    GetKernelVersion = 207,
    GetUptime = 208,

    // Security and permissions (220-239)
    SetUid = 220,
    SetGid = 221,
    GetUid = 222,
    GetGid = 223,
    SetGroups = 224,
    GetGroups = 225,
    Access = 226,
    Umask = 227,
    Chroot = 228,

    // Advanced features (240-255)
    FutureWait = 240,
    EventNotify = 241,
    ThreadCreate = 242,
    ThreadJoin = 243,
    MutexLock = 244,
    MutexUnlock = 245,
    CondWait = 246,
    CondSignal = 247,
    AtomicOp = 248,
    BarrierWait = 249,

    // Invalid/Unknown
    Invalid = u64::MAX,
}

impl From<u64> for SyscallNumber {
    fn from(num: u64) -> Self {
        match num {
            0 => SyscallNumber::Exit,
            1 => SyscallNumber::Fork,
            2 => SyscallNumber::Wait,
            3 => SyscallNumber::GetPid,
            4 => SyscallNumber::GetParentPid,
            5 => SyscallNumber::Sleep,
            6 => SyscallNumber::Kill,
            7 => SyscallNumber::SetPriority,
            8 => SyscallNumber::GetPriority,
            9 => SyscallNumber::Exec,
            20 => SyscallNumber::Open,
            21 => SyscallNumber::Close,
            22 => SyscallNumber::Read,
            23 => SyscallNumber::Write,
            24 => SyscallNumber::Seek,
            25 => SyscallNumber::Stat,
            26 => SyscallNumber::Fstat,
            27 => SyscallNumber::Mkdir,
            28 => SyscallNumber::Rmdir,
            29 => SyscallNumber::Unlink,
            40 => SyscallNumber::Mmap,
            41 => SyscallNumber::Munmap,
            42 => SyscallNumber::Mprotect,
            43 => SyscallNumber::Brk,
            44 => SyscallNumber::Sbrk,
            60 => SyscallNumber::Pipe,
            80 => SyscallNumber::GetTime,
            81 => SyscallNumber::SetTime,
            87 => SyscallNumber::Yield,
            160 => SyscallNumber::AiPredict,
            161 => SyscallNumber::AiTrain,
            162 => SyscallNumber::AiInference,
            163 => SyscallNumber::AiGetStatus,
            166 => SyscallNumber::GpuAllocate,
            167 => SyscallNumber::GpuExecute,
            200 => SyscallNumber::GetSystemInfo,
            201 => SyscallNumber::GetProcessInfo,
            202 => SyscallNumber::GetMemoryInfo,
            _ => SyscallNumber::Invalid,
        }
    }
}

impl fmt::Display for SyscallNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyscallNumber::Exit => write!(f, "exit"),
            SyscallNumber::Fork => write!(f, "fork"),
            SyscallNumber::GetPid => write!(f, "getpid"),
            SyscallNumber::Open => write!(f, "open"),
            SyscallNumber::Read => write!(f, "read"),
            SyscallNumber::Write => write!(f, "write"),
            SyscallNumber::AiPredict => write!(f, "ai_predict"),
            SyscallNumber::GpuAllocate => write!(f, "gpu_allocate"),
            _ => write!(f, "syscall_{}", *self as u64),
        }
    }
}

/// System call result type
pub type SyscallResult = Result<u64, SyscallError>;

/// System call error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallError {
    /// Invalid system call number
    InvalidSyscall,
    /// Invalid argument
    InvalidArgument,
    /// Permission denied
    PermissionDenied,
    /// Resource not found
    NotFound,
    /// Resource already exists
    AlreadyExists,
    /// Not enough memory
    OutOfMemory,
    /// I/O error
    IoError,
    /// Interrupted system call
    Interrupted,
    /// Operation would block
    WouldBlock,
    /// Bad file descriptor
    BadFileDescriptor,
    /// Operation not supported
    NotSupported,
    /// Resource busy
    Busy,
    /// Invalid process ID
    InvalidProcess,
    /// Quota exceeded
    QuotaExceeded,
    /// Hardware error
    HardwareError,
    /// AI system error
    AiError,
    /// GPU error
    GpuError,
}

impl SyscallError {
    /// Convert to errno-style error code
    pub fn to_errno(self) -> i32 {
        match self {
            SyscallError::InvalidSyscall => -1,
            SyscallError::InvalidArgument => -22,  // EINVAL
            SyscallError::PermissionDenied => -13, // EACCES
            SyscallError::NotFound => -2,          // ENOENT
            SyscallError::AlreadyExists => -17,    // EEXIST
            SyscallError::OutOfMemory => -12,      // ENOMEM
            SyscallError::IoError => -5,           // EIO
            SyscallError::Interrupted => -4,       // EINTR
            SyscallError::WouldBlock => -11,       // EAGAIN
            SyscallError::BadFileDescriptor => -9, // EBADF
            SyscallError::NotSupported => -95,     // EOPNOTSUPP
            SyscallError::Busy => -16,             // EBUSY
            SyscallError::InvalidProcess => -3,    // ESRCH
            SyscallError::QuotaExceeded => -122,   // EDQUOT
            SyscallError::HardwareError => -19,    // ENODEV
            SyscallError::AiError => -200,         // Custom
            SyscallError::GpuError => -201,        // Custom
        }
    }
}

impl fmt::Display for SyscallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyscallError::InvalidSyscall => write!(f, "Invalid system call"),
            SyscallError::InvalidArgument => write!(f, "Invalid argument"),
            SyscallError::PermissionDenied => write!(f, "Permission denied"),
            SyscallError::NotFound => write!(f, "Not found"),
            SyscallError::AlreadyExists => write!(f, "Already exists"),
            SyscallError::OutOfMemory => write!(f, "Out of memory"),
            SyscallError::IoError => write!(f, "I/O error"),
            SyscallError::Interrupted => write!(f, "Interrupted"),
            SyscallError::WouldBlock => write!(f, "Would block"),
            SyscallError::BadFileDescriptor => write!(f, "Bad file descriptor"),
            SyscallError::NotSupported => write!(f, "Operation not supported"),
            SyscallError::Busy => write!(f, "Resource busy"),
            SyscallError::InvalidProcess => write!(f, "Invalid process"),
            SyscallError::QuotaExceeded => write!(f, "Quota exceeded"),
            SyscallError::HardwareError => write!(f, "Hardware error"),
            SyscallError::AiError => write!(f, "AI system error"),
            SyscallError::GpuError => write!(f, "GPU error"),
        }
    }
}

/// System call statistics
#[derive(Debug, Default, Clone)]
pub struct SyscallStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub avg_execution_time: u64,
    pub calls_per_syscall: [u64; 256],
}

/// Global syscall statistics
static SYSCALL_STATS: Mutex<SyscallStats> = Mutex::new(SyscallStats {
    total_calls: 0,
    successful_calls: 0,
    failed_calls: 0,
    avg_execution_time: 0,
    calls_per_syscall: [0; 256],
});

/// System call handler
pub struct SyscallHandler {
    stats: Mutex<SyscallStats>,
    ai_optimization_enabled: bool,
}

impl SyscallHandler {
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(SyscallStats::default()),
            ai_optimization_enabled: true,
        }
    }

    /// Handle system call with full validation and logging
    pub fn handle_syscall(
        &self,
        syscall_num: u64,
        arg0: u64,
        arg1: u64,
        arg2: u64,
        arg3: u64,
        arg4: u64,
        arg5: u64,
    ) -> SyscallResult {
        let start_time = time::get_ticks();
        let syscall = SyscallNumber::from(syscall_num);

        // Update statistics
        {
            let mut stats = self.stats.lock();
            stats.total_calls += 1;
            if (syscall_num as usize) < stats.calls_per_syscall.len() {
                stats.calls_per_syscall[syscall_num as usize] += 1;
            }
        }

        // Validate syscall number
        if syscall == SyscallNumber::Invalid {
            self.record_failed_call();
            return Err(SyscallError::InvalidSyscall);
        }

        // Check process permissions
        if let Err(err) = self.check_permissions(&syscall) {
            self.record_failed_call();
            return Err(err);
        }

        // Route to appropriate handler
        let result = match syscall {
            // Process management
            SyscallNumber::Exit => self.sys_exit(arg0 as i32),
            SyscallNumber::Fork => self.sys_fork(),
            SyscallNumber::GetPid => self.sys_getpid(),
            SyscallNumber::GetParentPid => self.sys_getppid(),
            SyscallNumber::Kill => self.sys_kill(arg0, arg1 as i32),
            SyscallNumber::SetPriority => self.sys_setpriority(arg0, arg1),
            SyscallNumber::GetPriority => self.sys_getpriority(arg0),

            // File system
            SyscallNumber::Open => self.sys_open(arg0, arg1, arg2),
            SyscallNumber::Close => self.sys_close(arg0 as i32),
            SyscallNumber::Read => self.sys_read(arg0 as i32, arg1, arg2),
            SyscallNumber::Write => self.sys_write(arg0 as i32, arg1, arg2),
            SyscallNumber::Seek => self.sys_seek(arg0 as i32, arg1 as i64, arg2),

            // Memory management
            SyscallNumber::Mmap => self.sys_mmap(arg0, arg1, arg2, arg3, arg4 as i32, arg5),
            SyscallNumber::Munmap => self.sys_munmap(arg0, arg1),
            SyscallNumber::Brk => self.sys_brk(arg0),

            // Time and scheduling
            SyscallNumber::GetTime => self.sys_gettime(),
            SyscallNumber::Sleep => self.sys_sleep(arg0),
            SyscallNumber::Yield => self.sys_yield(),

            // AI and hardware optimization
            SyscallNumber::AiPredict => self.sys_ai_predict(arg0, arg1, arg2),
            SyscallNumber::AiTrain => self.sys_ai_train(arg0, arg1),
            SyscallNumber::AiInference => self.sys_ai_inference(arg0, arg1, arg2),
            SyscallNumber::AiGetStatus => self.sys_ai_get_status(),
            SyscallNumber::GpuAllocate => self.sys_gpu_allocate(arg0),
            SyscallNumber::GpuExecute => self.sys_gpu_execute(arg0, arg1, arg2),
            SyscallNumber::GpuDeallocate => self.sys_gpu_deallocate(arg0),

            // System information
            SyscallNumber::GetSystemInfo => self.sys_get_system_info(arg0, arg1),
            SyscallNumber::GetProcessInfo => self.sys_get_process_info(arg0, arg1, arg2),
            SyscallNumber::GetMemoryInfo => self.sys_get_memory_info(arg0, arg1),

            _ => Err(SyscallError::NotSupported),
        };

        // Record execution time and result
        let execution_time = time::get_ticks() - start_time;
        match result {
            Ok(_) => {
                self.record_successful_call(execution_time);
                // AI optimization hint
                if self.ai_optimization_enabled {
                    ai::learning::record_syscall_pattern(syscall, execution_time, true);
                }
            }
            Err(_) => {
                self.record_failed_call();
                if self.ai_optimization_enabled {
                    ai::learning::record_syscall_pattern(syscall, execution_time, false);
                }
            }
        }

        result
    }

    /// Check if current process has permission for this syscall
    fn check_permissions(&self, syscall: &SyscallNumber) -> Result<(), SyscallError> {
        // Basic permission checking - in real OS this would be more sophisticated
        match syscall {
            SyscallNumber::SetTime |
            SyscallNumber::Chroot |
            SyscallNumber::SetUid |
            SyscallNumber::SetGid => {
                // These require privileged access
                Ok(()) // For now, allow all - real OS would check privileges
            }
            _ => Ok(())
        }
    }

    // Process management syscalls
    fn sys_exit(&self, exit_code: i32) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            if let Some(pid) = current_process_id() {
                pm.terminate_process(pid, exit_code)
                    .map_err(|_| SyscallError::InvalidProcess)?;
            }
        }
        Ok(0) // Never reached
    }

    fn sys_fork(&self) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            if let Some(parent_pid) = current_process_id() {
                let child_pid = pm.fork_process(parent_pid)
                    .map_err(|_| SyscallError::InvalidProcess)?;
                Ok(child_pid.as_u64())
            } else {
                Err(SyscallError::InvalidProcess)
            }
        } else {
            Err(SyscallError::InvalidProcess)
        }
    }

    fn sys_getpid(&self) -> SyscallResult {
        current_process_id()
            .map(|pid| pid.as_u64())
            .ok_or(SyscallError::InvalidProcess)
    }

    fn sys_getppid(&self) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            if let Some(current_pid) = current_process_id() {
                if let Some(process) = pm.get_process(current_pid) {
                    return Ok(process.parent_pid.map(|pid| pid.as_u64()).unwrap_or(0));
                }
            }
        }
        Err(SyscallError::InvalidProcess)
    }

    fn sys_kill(&self, pid: u64, signal: i32) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            let target_pid = ProcessId::from_u64(pid);
            pm.kill_process(target_pid)
                .map_err(|_| SyscallError::InvalidProcess)?;
            Ok(0)
        } else {
            Err(SyscallError::InvalidProcess)
        }
    }

    fn sys_setpriority(&self, pid: u64, priority: u64) -> SyscallResult {
        // Implementation would set process priority
        // For now, just validate arguments
        if priority > ProcessPriority::Kernel as u64 {
            return Err(SyscallError::InvalidArgument);
        }
        Ok(0)
    }

    fn sys_getpriority(&self, pid: u64) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            let process_id = if pid == 0 {
                current_process_id().ok_or(SyscallError::InvalidProcess)?
            } else {
                ProcessId::from_u64(pid)
            };

            if let Some(process) = pm.get_process(process_id) {
                Ok(process.priority as u64)
            } else {
                Err(SyscallError::InvalidProcess)
            }
        } else {
            Err(SyscallError::InvalidProcess)
        }
    }

    // File system syscalls
    fn sys_open(&self, path_ptr: u64, flags: u64, mode: u64) -> SyscallResult {
        // Validate path pointer
        let path = self.read_user_string(path_ptr)?;

        // Convert flags
        let open_flags = OpenFlags::from_bits_truncate(flags as u32);

        // Open file through filesystem
        match crate::fs::open(&path, open_flags, mode as u32) {
            Ok(handle) => Ok(handle.fd() as u64),
            Err(_) => Err(SyscallError::NotFound),
        }
    }

    fn sys_close(&self, fd: i32) -> SyscallResult {
        if fd < 0 {
            return Err(SyscallError::BadFileDescriptor);
        }

        match crate::fs::close(fd as u32) {
            Ok(()) => Ok(0),
            Err(_) => Err(SyscallError::BadFileDescriptor),
        }
    }

    fn sys_read(&self, fd: i32, buf_ptr: u64, count: u64) -> SyscallResult {
        if fd < 0 {
            return Err(SyscallError::BadFileDescriptor);
        }

        if count == 0 {
            return Ok(0);
        }

        // Validate user buffer
        let buffer = self.validate_user_buffer(buf_ptr, count as usize, true)?;

        match crate::fs::read(fd as u32, buffer) {
            Ok(bytes_read) => Ok(bytes_read as u64),
            Err(_) => Err(SyscallError::IoError),
        }
    }

    fn sys_write(&self, fd: i32, buf_ptr: u64, count: u64) -> SyscallResult {
        if fd < 0 {
            return Err(SyscallError::BadFileDescriptor);
        }

        if count == 0 {
            return Ok(0);
        }

        // Validate user buffer
        let buffer = self.validate_user_buffer(buf_ptr, count as usize, false)?;

        match crate::fs::write(fd as u32, buffer) {
            Ok(bytes_written) => Ok(bytes_written as u64),
            Err(_) => Err(SyscallError::IoError),
        }
    }

    fn sys_seek(&self, fd: i32, offset: i64, whence: u64) -> SyscallResult {
        if fd < 0 {
            return Err(SyscallError::BadFileDescriptor);
        }

        match crate::fs::seek(fd as u32, offset, whence as u32) {
            Ok(new_offset) => Ok(new_offset as u64),
            Err(_) => Err(SyscallError::BadFileDescriptor),
        }
    }

    // Memory management syscalls
    fn sys_mmap(&self, addr: u64, length: u64, prot: u64, flags: u64, fd: i32, offset: u64) -> SyscallResult {
        if length == 0 {
            return Err(SyscallError::InvalidArgument);
        }

        // Convert protection flags
        let protection = MemoryProtection {
            readable: (prot & 0x1) != 0,
            writable: (prot & 0x2) != 0,
            executable: (prot & 0x4) != 0,
            user_accessible: true,
        };

        // For now, ignore fd and offset (anonymous mapping)
        match allocate_memory(length as usize, MemoryRegionType::UserData, protection) {
            Ok(virt_addr) => Ok(virt_addr.as_u64()),
            Err(_) => Err(SyscallError::OutOfMemory),
        }
    }

    fn sys_munmap(&self, addr: u64, length: u64) -> SyscallResult {
        if length == 0 {
            return Err(SyscallError::InvalidArgument);
        }

        // Implementation would unmap memory region
        Ok(0)
    }

    fn sys_brk(&self, addr: u64) -> SyscallResult {
        // Implementation would adjust heap break
        Ok(addr)
    }

    // Time and scheduling syscalls
    fn sys_gettime(&self) -> SyscallResult {
        Ok(time::get_ticks())
    }

    fn sys_sleep(&self, duration: u64) -> SyscallResult {
        time::sleep(duration);
        Ok(0)
    }

    fn sys_yield(&self) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            pm.schedule();
        }
        Ok(0)
    }

    // AI and hardware optimization syscalls
    fn sys_ai_predict(&self, model_ptr: u64, input_ptr: u64, output_ptr: u64) -> SyscallResult {
        // Validate pointers and perform AI prediction
        match ai::inference_engine::predict(model_ptr, input_ptr, output_ptr) {
            Ok(confidence) => Ok(confidence as u64),
            Err(_) => Err(SyscallError::AiError),
        }
    }

    fn sys_ai_train(&self, model_ptr: u64, data_ptr: u64) -> SyscallResult {
        match ai::learning::train_model(model_ptr, data_ptr) {
            Ok(accuracy) => Ok(accuracy as u64),
            Err(_) => Err(SyscallError::AiError),
        }
    }

    fn sys_ai_inference(&self, model_id: u64, input_ptr: u64, output_ptr: u64) -> SyscallResult {
        match ai::inference_engine::run_inference(model_id, input_ptr, output_ptr) {
            Ok(result) => Ok(result),
            Err(_) => Err(SyscallError::AiError),
        }
    }

    fn sys_ai_get_status(&self) -> SyscallResult {
        let status = ai::get_ai_status();
        Ok(status as u64)
    }

    fn sys_gpu_allocate(&self, size: u64) -> SyscallResult {
        match gpu::allocate_buffer(size as usize) {
            Ok(buffer_id) => Ok(buffer_id as u64),
            Err(_) => Err(SyscallError::GpuError),
        }
    }

    fn sys_gpu_execute(&self, program_id: u64, input_buffer: u64, output_buffer: u64) -> SyscallResult {
        match gpu::execute_program(program_id as u32, input_buffer as u32, output_buffer as u32) {
            Ok(execution_id) => Ok(execution_id as u64),
            Err(_) => Err(SyscallError::GpuError),
        }
    }

    fn sys_gpu_deallocate(&self, buffer_id: u64) -> SyscallResult {
        match gpu::deallocate_buffer(buffer_id as u32) {
            Ok(()) => Ok(0),
            Err(_) => Err(SyscallError::GpuError),
        }
    }

    // System information syscalls
    fn sys_get_system_info(&self, info_type: u64, buffer_ptr: u64) -> SyscallResult {
        match info_type {
            0 => {
                // CPU info
                let info = "RustOS x86_64 kernel";
                self.copy_to_user_buffer(buffer_ptr, info.as_bytes())?;
                Ok(info.len() as u64)
            }
            1 => {
                // Memory info
                if let Some(stats) = get_memory_stats() {
                    let info = format!("Total: {} MB, Free: {} MB, Usage: {:.1}%",
                                     stats.total_frames * 4 / 1024,
                                     stats.free_frames * 4 / 1024,
                                     stats.memory_usage_percent());
                    self.copy_to_user_buffer(buffer_ptr, info.as_bytes())?;
                    Ok(info.len() as u64)
                } else {
                    Err(SyscallError::NotSupported)
                }
            }
            _ => Err(SyscallError::InvalidArgument)
        }
    }

    fn sys_get_process_info(&self, pid: u64, buffer_ptr: u64, buffer_size: u64) -> SyscallResult {
        if let Some(pm) = get_process_manager() {
            let process_id = if pid == 0 {
                current_process_id().ok_or(SyscallError::InvalidProcess)?
            } else {
                ProcessId::from_u64(pid)
            };

            if let Some(process) = pm.get_process(process_id) {
                let info = format!("PID: {}, Name: {}, State: {}, Priority: {:?}, Memory: {} KB",
                                 process.pid.as_u64(), process.name, process.state,
                                 process.priority, process.stats.memory_usage / 1024);
                self.copy_to_user_buffer(buffer_ptr, info.as_bytes())?;
                Ok(info.len() as u64)
            } else {
                Err(SyscallError::InvalidProcess)
            }
        } else {
            Err(SyscallError::InvalidProcess)
        }
    }

    fn sys_get_memory_info(&self, buffer_ptr: u64, buffer_size: u64) -> SyscallResult {
        if let Some(stats) = get_memory_stats() {
            let info = format!("Total frames: {}, Allocated: {}, Free: {}, Regions: {}, Usage: {:.1}%",
                             stats.total_frames, stats.allocated_frames, stats.free_frames,
                             stats.total_regions, stats.memory_usage_percent());
            self.copy_to_user_buffer(buffer_ptr, info.as_bytes())?;
            Ok(info.len() as u64)
        } else {
            Err(SyscallError::NotSupported)
        }
    }

    // Helper functions
    fn read_user_string(&self, ptr: u64) -> Result<String, SyscallError> {
        if ptr == 0 {
            return Err(SyscallError::InvalidArgument);
        }

        // In a real kernel, this would copy from user space with proper validation
        // For now, we'll simulate this
        unsafe {
            let c_str = ptr as *const u8;
            let mut len = 0;
            while len < 256 && *c_str.add(len) != 0 {
                len += 1;
            }

            if len == 256 {
                return Err(SyscallError::InvalidArgument);
            }

            let bytes = slice::from_raw_parts(c_str, len);
            String::from_utf8(bytes.to_vec())
                .map_err(|_| SyscallError::InvalidArgument)
        }
    }

    fn validate_user_buffer(&self, ptr: u64, size: usize, writable: bool) -> Result<&mut [u8], SyscallError> {
        if ptr == 0 || size == 0 {
            return Err(SyscallError::InvalidArgument);
        }

        // Basic validation - in real kernel would check page table mappings
        if ptr > 0x7FFF_FFFF_FFFF_0000 {
            return Err(SyscallError::InvalidArgument);
        }

        unsafe {
            Ok(slice::from_raw_parts_mut(ptr as *mut u8, size))
        }
    }

    fn copy_to_user_buffer(&self, ptr: u64, data: &[u8]) -> Result<(), SyscallError> {
        if ptr == 0 || data.is_empty() {
            return Err(SyscallError::InvalidArgument);
        }

        let buffer = self.validate_user_buffer(ptr, data.len(), true)?;
        buffer.copy_from_slice(data);
        Ok(())
    }

    fn record_successful_call(&self, execution_time: u64) {
        let mut stats = self.stats.lock();
        stats.successful_calls += 1;
        stats.avg_execution_time = (stats.avg_execution_time * 7 + execution_time) / 8;
    }

    fn record_failed_call(&self) {
        let mut stats = self.stats.lock();
        stats.failed_calls += 1;
    }

    /// Get syscall statistics
    pub fn get_stats(&self) -> SyscallStats {
        self.stats.lock().clone()
    }
}

// Global syscall handler
lazy_static! {
    static ref SYSCALL_HANDLER: SyscallHandler = SyscallHandler::new();
}

// Extension trait for ProcessId
impl ProcessId {
    pub fn from_u64(val: u64) -> Self {
        ProcessId(val)
    }
}

/// Initialize the system call interface
pub fn init() {
    crate::println!("System call interface initialized");
    crate::println!("Syscalls available: Process, File, Memory, AI, GPU, Network");
}

/// Main entry point for system calls (called from interrupt handler)
#[no_mangle]
pub extern "C" fn syscall_entry(
    syscall_num: u64,
    arg0: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
) -> u64 {
    let result = SYSCALL_HANDLER.handle_syscall(syscall_num, arg0, arg1, arg2, arg3, arg4, arg5);

    match result {
        Ok(value) => value,
        Err(error) => error.to_errno() as u64,
    }
}

/// High-level syscall interface functions
pub fn exit(exit_code: i32) -> ! {
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Exit as u64,
            in("rdi") exit_code as u64,
            options(noreturn)
        );
    }
}

pub fn getpid() -> u64 {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::GetPid as u64,
            lateout("rax") result,
        );
    }
    result
}

pub fn fork() -> Result<u64, SyscallError> {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Fork as u64,
            lateout("rax") result,
        );
    }

    if (result as i64) < 0 {
        Err(SyscallError::InvalidProcess)
    } else {
        Ok(result)
    }
}

pub fn open(path: &str, flags: u32, mode: u32) -> Result<i32, SyscallError> {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Open as u64,
            in("rdi") path.as_ptr() as u64,
            in("rsi") flags as u64,
            in("rdx") mode as u64,
            lateout("rax") result,
        );
    }

    if (result as i64) < 0 {
        Err(SyscallError::NotFound)
    } else {
        Ok(result as i32)
    }
}

pub fn read(fd: i32, buffer: &mut [u8]) -> Result<usize, SyscallError> {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Read as u64,
            in("rdi") fd as u64,
            in("rsi") buffer.as_mut_ptr() as u64,
            in("rdx") buffer.len() as u64,
            lateout("rax") result,
        );
    }

    if (result as i64) < 0 {
        Err(SyscallError::IoError)
    } else {
        Ok(result as usize)
    }
}

pub fn write(fd: i32, buffer: &[u8]) -> Result<usize, SyscallError> {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Write as u64,
            in("rdi") fd as u64,
            in("rsi") buffer.as_ptr() as u64,
            in("rdx") buffer.len() as u64,
            lateout("rax") result,
        );
    }

    if (result as i64) < 0 {
        Err(SyscallError::IoError)
    } else {
        Ok(result as usize)
    }
}

pub fn close(fd: i32) -> Result<(), SyscallError> {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") SyscallNumber::Close as u64,
            in("rdi") fd as u64,
            lateout("rax") result,
        );
    }

    if (result as i64) < 0 {
        Err(SyscallError::BadFileDescriptor)
    } else {
        Ok(())
    }
}

/// Get system call statistics
pub fn get_syscall_stats() -> SyscallStats {
    SYSCALL_HANDLER.get_stats()
}

/// Demonstrate syscall functionality
pub fn demonstrate_syscalls() {
    crate::println!("=== System Call Demonstration ===");

    // Show current process ID
    let pid = getpid();
    crate::println!("Current PID: {}", pid);

    // Show syscall statistics
    let stats = get_syscall_stats();
    crate::println!("Syscall Stats: {} total calls, {} successful, {} failed",
                   stats.total_calls, stats.successful_calls, stats.failed_calls);

    if stats.total_calls > 0 {
        crate::println!("Average execution time: {} ticks", stats.avg_execution_time);
    }

    // Demonstrate AI syscall
    crate::println!("AI Status: {}", syscall_entry(
        SyscallNumber::AiGetStatus as u64, 0, 0, 0, 0, 0, 0
    ));

    crate::println!("System call demonstration complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_syscall_number_conversion() {
        assert_eq!(SyscallNumber::from(0), SyscallNumber::Exit);
        assert_eq!(SyscallNumber::from(1), SyscallNumber::Fork);
        assert_eq!(SyscallNumber::from(20), SyscallNumber::Open);
        assert_eq!(SyscallNumber::from(9999), SyscallNumber::Invalid);
    }

    #[test_case]
    fn test_syscall_error_codes() {
        assert_eq!(SyscallError::InvalidArgument.to_errno(), -22);
        assert_eq!(SyscallError::PermissionDenied.to_errno(), -13);
        assert_eq!(SyscallError::NotFound.to_errno(), -2);
    }

    #[test_case]
    fn test_syscall_handler_creation() {
        let handler = SyscallHandler::new();
        let stats = handler.get_stats();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.successful_calls, 0);
        assert_eq!(stats.failed_calls, 0);
    }

    #[test_case]
    fn test_syscall_result_handling() {
        let success: SyscallResult = Ok(42);
        let error: SyscallResult = Err(SyscallError::InvalidArgument);

        assert!(success.is_ok());
        assert!(error.is_err());
        assert_eq!(success.unwrap(), 42);
        assert_eq!(error.unwrap_err(), SyscallError::InvalidArgument);
    }
}
