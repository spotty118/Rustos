//! Linux file operation APIs
//!
//! This module implements Linux-compatible file operations including
//! stat, access, dup, link operations, and directory handling.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use super::types::*;
use super::{LinuxResult, LinuxError};

/// Operation counter for statistics
static FILE_OPS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Initialize file operations subsystem
pub fn init_file_operations() {
    // Initialize file operation tracking
    FILE_OPS_COUNT.store(0, Ordering::Relaxed);
}

/// Get number of file operations performed
pub fn get_operation_count() -> u64 {
    FILE_OPS_COUNT.load(Ordering::Relaxed)
}

/// Increment operation counter
fn inc_ops() {
    FILE_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// fstat - get file status by file descriptor
pub fn fstat(fd: Fd, statbuf: *mut Stat) -> LinuxResult<i32> {
    inc_ops();

    if statbuf.is_null() {
        return Err(LinuxError::EFAULT);
    }

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Get actual file status from VFS
    // For now, return stub data
    unsafe {
        *statbuf = Stat::new();
        (*statbuf).st_mode = mode::S_IFREG | 0o644;
        (*statbuf).st_size = 0;
        (*statbuf).st_nlink = 1;
    }

    Ok(0)
}

/// lstat - get file status (don't follow symlinks)
pub fn lstat(path: *const u8, statbuf: *mut Stat) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() || statbuf.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement actual lstat via VFS
    // For now, call fstat-like logic
    unsafe {
        *statbuf = Stat::new();
        (*statbuf).st_mode = mode::S_IFREG | 0o644;
    }

    Ok(0)
}

/// access - check file accessibility
pub fn access(path: *const u8, mode: i32) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Check actual file permissions via VFS
    // For now, assume all files are accessible
    match mode {
        access::F_OK | access::R_OK | access::W_OK | access::X_OK => Ok(0),
        _ => Err(LinuxError::EINVAL),
    }
}

/// faccessat - check file accessibility relative to directory fd
pub fn faccessat(dirfd: Fd, path: *const u8, mode: i32, flags: i32) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Handle relative paths and flags
    access(path, mode)
}

/// dup - duplicate file descriptor
pub fn dup(oldfd: Fd) -> LinuxResult<Fd> {
    inc_ops();

    if oldfd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement actual FD duplication
    // For now, return a stub new FD
    Ok(oldfd + 1000)
}

/// dup2 - duplicate file descriptor to specific FD number
pub fn dup2(oldfd: Fd, newfd: Fd) -> LinuxResult<Fd> {
    inc_ops();

    if oldfd < 0 || newfd < 0 {
        return Err(LinuxError::EBADF);
    }

    if oldfd == newfd {
        return Ok(newfd);
    }

    // TODO: Close newfd if open, then duplicate oldfd to newfd
    Ok(newfd)
}

/// dup3 - duplicate file descriptor with flags
pub fn dup3(oldfd: Fd, newfd: Fd, flags: i32) -> LinuxResult<Fd> {
    inc_ops();

    if oldfd < 0 || newfd < 0 || oldfd == newfd {
        return Err(LinuxError::EINVAL);
    }

    // TODO: Handle O_CLOEXEC flag
    dup2(oldfd, newfd)
}

/// link - create hard link
pub fn link(oldpath: *const u8, newpath: *const u8) -> LinuxResult<i32> {
    inc_ops();

    if oldpath.is_null() || newpath.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement hard link creation via VFS
    Err(LinuxError::ENOSYS)
}

/// symlink - create symbolic link
pub fn symlink(target: *const u8, linkpath: *const u8) -> LinuxResult<i32> {
    inc_ops();

    if target.is_null() || linkpath.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement symbolic link creation via VFS
    Err(LinuxError::ENOSYS)
}

/// readlink - read symbolic link
pub fn readlink(path: *const u8, buf: *mut u8, bufsiz: usize) -> LinuxResult<isize> {
    inc_ops();

    if path.is_null() || buf.is_null() {
        return Err(LinuxError::EFAULT);
    }

    if bufsiz == 0 {
        return Err(LinuxError::EINVAL);
    }

    // TODO: Implement symbolic link reading via VFS
    Err(LinuxError::ENOSYS)
}

/// rename - rename file or directory
pub fn rename(oldpath: *const u8, newpath: *const u8) -> LinuxResult<i32> {
    inc_ops();

    if oldpath.is_null() || newpath.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement file renaming via VFS
    Err(LinuxError::ENOSYS)
}

/// renameat - rename file relative to directory fds
pub fn renameat(
    olddirfd: Fd,
    oldpath: *const u8,
    newdirfd: Fd,
    newpath: *const u8,
) -> LinuxResult<i32> {
    inc_ops();

    if oldpath.is_null() || newpath.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Handle relative paths
    rename(oldpath, newpath)
}

/// chmod - change file permissions
pub fn chmod(path: *const u8, mode: Mode) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement permission changes via VFS
    Ok(0)
}

/// fchmod - change file permissions by fd
pub fn fchmod(fd: Fd, mode: Mode) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement permission changes via VFS
    Ok(0)
}

/// fchmodat - change file permissions relative to directory fd
pub fn fchmodat(dirfd: Fd, path: *const u8, mode: Mode, flags: i32) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Handle relative paths and flags
    chmod(path, mode)
}

/// chown - change file owner and group
pub fn chown(path: *const u8, owner: Uid, group: Gid) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement ownership changes via VFS
    Ok(0)
}

/// fchown - change file owner and group by fd
pub fn fchown(fd: Fd, owner: Uid, group: Gid) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement ownership changes via VFS
    Ok(0)
}

/// lchown - change file owner and group (don't follow symlinks)
pub fn lchown(path: *const u8, owner: Uid, group: Gid) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement ownership changes via VFS (no symlink follow)
    Ok(0)
}

/// truncate - truncate file to specified length
pub fn truncate(path: *const u8, length: Off) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    if length < 0 {
        return Err(LinuxError::EINVAL);
    }

    // TODO: Implement file truncation via VFS
    Err(LinuxError::ENOSYS)
}

/// ftruncate - truncate file to specified length by fd
pub fn ftruncate(fd: Fd, length: Off) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    if length < 0 {
        return Err(LinuxError::EINVAL);
    }

    // TODO: Implement file truncation via VFS
    Err(LinuxError::ENOSYS)
}

/// fsync - synchronize file to storage
pub fn fsync(fd: Fd) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement file synchronization
    Ok(0)
}

/// fdatasync - synchronize file data to storage
pub fn fdatasync(fd: Fd) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement data synchronization (metadata not required)
    Ok(0)
}

/// getdents - read directory entries
pub fn getdents(fd: Fd, dirp: *mut Dirent, count: usize) -> LinuxResult<isize> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    if dirp.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement directory reading via VFS
    Ok(0)
}

/// chdir - change current working directory
pub fn chdir(path: *const u8) -> LinuxResult<i32> {
    inc_ops();

    if path.is_null() {
        return Err(LinuxError::EFAULT);
    }

    // TODO: Implement directory change
    Err(LinuxError::ENOSYS)
}

/// fchdir - change current working directory by fd
pub fn fchdir(fd: Fd) -> LinuxResult<i32> {
    inc_ops();

    if fd < 0 {
        return Err(LinuxError::EBADF);
    }

    // TODO: Implement directory change by fd
    Err(LinuxError::ENOSYS)
}

/// getcwd - get current working directory
pub fn getcwd(buf: *mut u8, size: usize) -> LinuxResult<*mut u8> {
    inc_ops();

    if buf.is_null() {
        return Err(LinuxError::EFAULT);
    }

    if size == 0 {
        return Err(LinuxError::EINVAL);
    }

    // TODO: Implement getcwd
    // For now, return root directory
    unsafe {
        *buf = b'/';
        if size > 1 {
            *buf.add(1) = 0;
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dup_operations() {
        let oldfd = 3;
        let newfd = dup(oldfd).unwrap();
        assert!(newfd != oldfd);

        let specific_fd = 10;
        let result = dup2(oldfd, specific_fd).unwrap();
        assert_eq!(result, specific_fd);
    }

    #[test]
    fn test_access_modes() {
        let path = b"/test\0".as_ptr();
        assert!(access(path, access::F_OK).is_ok());
        assert!(access(path, access::R_OK).is_ok());
    }
}
