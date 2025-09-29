//! Compiler intrinsics for bare-metal environment
//! 
//! Provides missing symbols that the compiler expects

use core::ffi::c_void;

/// Memory copy implementation
#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    let dest_bytes = dest as *mut u8;
    let src_bytes = src as *const u8;
    
    for i in 0..n {
        *dest_bytes.add(i) = *src_bytes.add(i);
    }
    
    dest
}

/// Memory set implementation
#[no_mangle]
pub unsafe extern "C" fn memset(s: *mut c_void, c: i32, n: usize) -> *mut c_void {
    let bytes = s as *mut u8;
    let byte_val = c as u8;
    
    for i in 0..n {
        *bytes.add(i) = byte_val;
    }
    
    s
}

/// Memory compare implementation
#[no_mangle]
pub unsafe extern "C" fn memcmp(s1: *const c_void, s2: *const c_void, n: usize) -> i32 {
    let bytes1 = s1 as *const u8;
    let bytes2 = s2 as *const u8;
    
    for i in 0..n {
        let b1 = *bytes1.add(i);
        let b2 = *bytes2.add(i);
        
        if b1 < b2 {
            return -1;
        } else if b1 > b2 {
            return 1;
        }
    }
    
    0
}

/// Memory move implementation (handles overlapping regions)
#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    let dest_bytes = dest as *mut u8;
    let src_bytes = src as *const u8;
    
    if (dest_bytes as usize) < (src_bytes as usize) {
        // Copy forward
        for i in 0..n {
            *dest_bytes.add(i) = *src_bytes.add(i);
        }
    } else {
        // Copy backward to handle overlap
        for i in (0..n).rev() {
            *dest_bytes.add(i) = *src_bytes.add(i);
        }
    }
    
    dest
}
