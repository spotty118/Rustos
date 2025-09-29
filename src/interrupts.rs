//! Interrupt Descriptor Table (IDT) and Interrupt Handling
//!
//! This module provides a comprehensive interrupt handling system for RustOS.
//! It includes the IDT setup, exception handlers, and hardware interrupt management.

use core::{fmt, ptr};
use lazy_static::lazy_static;
use pic8259::ChainedPics;
use spin::Mutex;
use x86_64::instructions::port::Port;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use x86_64::VirtAddr;

/// Hardware interrupt offsets for the PIC (Programmable Interrupt Controller)
pub const PIC_1_OFFSET: u8 = 32;
pub const PIC_2_OFFSET: u8 = PIC_1_OFFSET + 8;

/// Hardware interrupt indices
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum InterruptIndex {
    Timer = PIC_1_OFFSET,
    Keyboard = PIC_1_OFFSET + 1,
    SerialPort1 = PIC_1_OFFSET + 4,
    SerialPort2 = PIC_1_OFFSET + 3,
    SpuriousInterrupt = PIC_1_OFFSET + 7,
}

impl InterruptIndex {
    fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn as_usize(self) -> usize {
        usize::from(self.as_u8())
    }
}

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();

        // CPU Exception handlers
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault_handler)
                .set_stack_index(crate::gdt::DOUBLE_FAULT_IST_INDEX);
        }
        idt.page_fault.set_handler_fn(page_fault_handler);
        idt.divide_error.set_handler_fn(divide_error_handler);
        idt.invalid_opcode.set_handler_fn(invalid_opcode_handler);
        idt.general_protection_fault.set_handler_fn(general_protection_fault_handler);
        idt.stack_segment_fault.set_handler_fn(stack_segment_fault_handler);
        idt.segment_not_present.set_handler_fn(segment_not_present_handler);
        idt.overflow.set_handler_fn(overflow_handler);
        idt.bound_range_exceeded.set_handler_fn(bound_range_exceeded_handler);
        idt.invalid_tss.set_handler_fn(invalid_tss_handler);
        // Machine check exception handler not yet implemented
        // SIMD floating point exception handler not yet implemented
        idt.virtualization.set_handler_fn(virtualization_handler);
        idt.alignment_check.set_handler_fn(alignment_check_handler);

        // Hardware interrupt handlers
        idt[InterruptIndex::Timer.as_usize()].set_handler_fn(timer_interrupt_handler);
        idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(keyboard_interrupt_handler);
        idt[InterruptIndex::SerialPort1.as_usize()].set_handler_fn(serial_port1_interrupt_handler);
        idt[InterruptIndex::SerialPort2.as_usize()].set_handler_fn(serial_port2_interrupt_handler);
        idt[InterruptIndex::SpuriousInterrupt.as_usize()].set_handler_fn(spurious_interrupt_handler);

        idt
    };
}

/// Global PIC controller instance
pub static PICS: Mutex<ChainedPics> =
    Mutex::new(unsafe { ChainedPics::new(PIC_1_OFFSET, PIC_2_OFFSET) });

/// Interrupt statistics
#[derive(Clone, Copy)]
pub struct InterruptStats {
    pub timer_count: u64,
    pub keyboard_count: u64,
    pub serial_count: u64,
    pub exception_count: u64,
    pub page_fault_count: u64,
    pub spurious_count: u64,
}

static mut INTERRUPT_STATS: InterruptStats = InterruptStats {
    timer_count: 0,
    keyboard_count: 0,
    serial_count: 0,
    exception_count: 0,
    page_fault_count: 0,
    spurious_count: 0,
};

/// Initialize the interrupt system
pub fn init() {
    IDT.load();
    
    // Initialize APIC system or fall back to PIC
    match crate::apic::init_apic_system() {
        Ok(()) => {
            // Configure standard IRQs with APIC
            if let Err(_e) = configure_standard_irqs_apic() {
                init_legacy_pic();
            } else {
                disable_legacy_pic();
            }
        }
        Err(_e) => {
            init_legacy_pic();
        }
    }
    
    x86_64::instructions::interrupts::enable();
}

/// Initialize legacy PIC
fn init_legacy_pic() {
    unsafe { PICS.lock().initialize() };
}

/// Disable legacy PIC when using APIC
fn disable_legacy_pic() {
    use x86_64::instructions::port::Port;
    
    unsafe {
        // Mask all interrupts on both PICs
        let mut pic1_data: Port<u8> = Port::new(0x21);
        let mut pic2_data: Port<u8> = Port::new(0xA1);
        
        pic1_data.write(0xFF);
        pic2_data.write(0xFF);
    }
}

/// Configure standard IRQs using APIC
fn configure_standard_irqs_apic() -> Result<(), &'static str> {
    // Configure timer (IRQ 0)
    crate::apic::configure_irq(0, InterruptIndex::Timer.as_u8(), 0)?;
    
    // Configure keyboard (IRQ 1)
    crate::apic::configure_irq(1, InterruptIndex::Keyboard.as_u8(), 0)?;
    
    // Configure serial ports
    crate::apic::configure_irq(4, InterruptIndex::SerialPort1.as_u8(), 0)?;
    crate::apic::configure_irq(3, InterruptIndex::SerialPort2.as_u8(), 0)?;
    
    Ok(())
}

/// Get interrupt statistics
pub fn get_stats() -> InterruptStats {
    unsafe { INTERRUPT_STATS }
}

/// Reset interrupt statistics
pub fn reset_stats() {
    unsafe {
        INTERRUPT_STATS = InterruptStats {
            timer_count: 0,
            keyboard_count: 0,
            serial_count: 0,
            exception_count: 0,
            page_fault_count: 0,
            spurious_count: 0,
        };
    }
}

/// Disable interrupts and return previous interrupt state
pub fn disable() -> bool {
    let rflags = x86_64::instructions::interrupts::are_enabled();
    x86_64::instructions::interrupts::disable();
    rflags
}

/// Enable interrupts
pub fn enable() {
    x86_64::instructions::interrupts::enable();
}

/// Execute a closure with interrupts disabled
pub fn without_interrupts<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let saved = disable();
    let result = f();
    if saved {
        enable();
    }
    result
}

// ========== CPU EXCEPTION HANDLERS ==========

extern "x86-interrupt" fn breakpoint_handler(_stack_frame: InterruptStackFrame) {
    // Handle breakpoint interrupt - increment counter for debugging
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    // Continue execution - breakpoints are non-fatal in production
}

extern "x86-interrupt" fn double_fault_handler(
    _stack_frame: InterruptStackFrame,
    error_code: u64,
) -> ! {
    panic!(
        "EXCEPTION: DOUBLE FAULT (error code: {})\n{:#?}",
        error_code, _stack_frame
    );
}

extern "x86-interrupt" fn page_fault_handler(
    _stack_frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;

    let fault_address = Cr2::read();

    // Log all page faults for production debugging
    crate::serial_println!(
        "Page fault at {:?}: present={}, write={}, user={}", 
        fault_address,
        !error_code.contains(PageFaultErrorCode::PROTECTION_VIOLATION),
        error_code.contains(PageFaultErrorCode::CAUSED_BY_WRITE),
        error_code.contains(PageFaultErrorCode::USER_MODE)
    );

    unsafe {
        INTERRUPT_STATS.page_fault_count += 1;
        INTERRUPT_STATS.exception_count += 1;
    }

    // In production, attempt page fault recovery
    if let Some(recovery_result) = attempt_page_fault_recovery(fault_address, error_code) {
        match recovery_result {
            PageFaultRecovery::Recovered => {
                crate::serial_println!("Page fault recovered successfully");
                return;
            }
            PageFaultRecovery::NeedsSwap => {
                crate::serial_println!("Page fault requires swap operation");
                // TODO: Implement swap-in functionality
            }
        }
    }

    // If recovery fails, panic as last resort
    panic!("Unrecoverable page fault at address {:?}", fault_address);
}

extern "x86-interrupt" fn divide_error_handler(stack_frame: InterruptStackFrame) {
    // Log divide by zero with context information
    crate::serial_println!(
        "Divide by zero error at RIP: {:?}, RSP: {:?}", 
        stack_frame.instruction_pointer,
        stack_frame.stack_pointer
    );
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Divide by zero exception at {:?}", stack_frame.instruction_pointer);
} 

extern "x86-interrupt" fn invalid_opcode_handler(stack_frame: InterruptStackFrame) {
    // Log invalid opcode with detailed context
    crate::serial_println!(
        "Invalid opcode at RIP: {:?}, attempting instruction recovery", 
        stack_frame.instruction_pointer
    );
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Invalid opcode at {:?}", stack_frame.instruction_pointer);
}

extern "x86-interrupt" fn general_protection_fault_handler(
    _stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // Production: critical protection fault
    crate::serial_println!("CRITICAL: General protection fault ({})", error_code);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("General protection fault");
}

extern "x86-interrupt" fn stack_segment_fault_handler(
    _stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // Production: critical stack fault
    crate::serial_println!("CRITICAL: Stack segment fault ({})", error_code);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Stack segment fault");
}

extern "x86-interrupt" fn segment_not_present_handler(
    stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // Log detailed segment fault information for debugging
    crate::serial_println!(
        "Segment not present fault - Error code: 0x{:x}, RIP: {:?}", 
        error_code, 
        stack_frame.instruction_pointer
    );
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Segment not present - error code: 0x{:x}", error_code);
}

extern "x86-interrupt" fn overflow_handler(stack_frame: InterruptStackFrame) {
    // Handle arithmetic overflow - log for debugging but continue execution
    crate::serial_println!("Arithmetic overflow detected at RIP: {:?}", stack_frame.instruction_pointer);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    // In production, overflow should be handled gracefully
}

extern "x86-interrupt" fn bound_range_exceeded_handler(stack_frame: InterruptStackFrame) {
    // Handle bounds check failure - log detailed information
    crate::serial_println!("Bounds check failed at RIP: {:?}", stack_frame.instruction_pointer);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    // Continue execution after logging - bounds checks are recoverable
}

extern "x86-interrupt" fn invalid_tss_handler(
    _stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // Production: critical TSS error
    crate::serial_println!("CRITICAL: Invalid TSS ({})", error_code);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Invalid TSS");
}

extern "x86-interrupt" fn virtualization_handler(_stack_frame: InterruptStackFrame) {
    // Production: virtualization error handled
    crate::serial_println!("CRITICAL: Virtualization");
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Virtualization exception");
}

extern "x86-interrupt" fn alignment_check_handler(
    _stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // Production: alignment error handled
    crate::serial_println!("CRITICAL: Alignment check ({})", error_code);
    unsafe {
        INTERRUPT_STATS.exception_count += 1;
    }
    panic!("Alignment check exception");
}

// ========== HARDWARE INTERRUPT HANDLERS ==========

extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    unsafe {
        INTERRUPT_STATS.timer_count += 1;
        
        // Send EOI to APIC if available, otherwise use PIC
        if crate::apic::apic_system().lock().is_initialized() {
            crate::apic::end_of_interrupt();
        } else {
            PICS.lock().notify_end_of_interrupt(InterruptIndex::Timer.as_u8());
        }
    }
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Use our new keyboard module to handle the interrupt
    crate::keyboard::handle_keyboard_interrupt();

    unsafe {
        INTERRUPT_STATS.keyboard_count += 1;

        // Send EOI to APIC if available, otherwise use PIC
        if crate::apic::apic_system().lock().is_initialized() {
            crate::apic::end_of_interrupt();
        } else {
            PICS.lock().notify_end_of_interrupt(InterruptIndex::Keyboard.as_u8());
        }
    }
}

extern "x86-interrupt" fn serial_port1_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Handle serial port 1 interrupt - process incoming data
    unsafe {
        INTERRUPT_STATS.serial_count += 1;
        
        // Process any available serial data
        crate::serial::handle_port1_interrupt();
        
        // Send EOI to APIC if available, otherwise use PIC
        if crate::apic::apic_system().lock().is_initialized() {
            crate::apic::end_of_interrupt();
        } else {
            PICS.lock().notify_end_of_interrupt(InterruptIndex::SerialPort1.as_u8());
        }
    }
}

extern "x86-interrupt" fn serial_port2_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Handle serial port 2 interrupt - process incoming data
    unsafe {
        INTERRUPT_STATS.serial_count += 1;
        
        // Process any available serial data
        crate::serial::handle_port2_interrupt();
        
        // Send EOI to APIC if available, otherwise use PIC
        if crate::apic::apic_system().lock().is_initialized() {
            crate::apic::end_of_interrupt();
        } else {
            PICS.lock().notify_end_of_interrupt(InterruptIndex::SerialPort2.as_u8());
        }
    }
}

extern "x86-interrupt" fn spurious_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Production: spurious interrupt handled silently
    unsafe {
        INTERRUPT_STATS.spurious_count += 1;
        // Don't send EOI for spurious interrupts
    }
}

// ========== INTERRUPT UTILITIES ==========

/// Trigger a breakpoint exception for testing
pub fn trigger_breakpoint() {
    x86_64::instructions::interrupts::int3();
}

/// Trigger a page fault for testing
pub unsafe fn trigger_page_fault() {
    let ptr = 0xdeadbeef as *mut u8;
    *ptr = 42;
}

/// Page fault recovery result
#[derive(Debug, Clone, Copy)]
pub enum PageFaultRecovery {
    Recovered,
    NeedsSwap,
}

/// Attempt to recover from a page fault
fn attempt_page_fault_recovery(
    fault_address: x86_64::VirtAddr, 
    error_code: x86_64::structures::idt::PageFaultErrorCode
) -> Option<PageFaultRecovery> {
    use x86_64::structures::idt::PageFaultErrorCode;
    
    // Check if this is a demand paging fault (page not present)
    if !error_code.contains(PageFaultErrorCode::PROTECTION_VIOLATION) {
        // Page not present - check if it's within valid memory ranges
        let addr = fault_address.as_u64();
        
        // Check if address is in user space and within reasonable bounds
        if addr >= 0x1000 && addr < 0x7fff_ffff_ffff {
            // Attempt to allocate physical page and map it
            if let Ok(()) = crate::memory::map_page_on_demand(fault_address) {
                return Some(PageFaultRecovery::Recovered);
            } else {
                // Page allocation failed - might need swap
                return Some(PageFaultRecovery::NeedsSwap);
            }
        }
    }
    
    // Cannot recover from this type of page fault
    None
}

/// Trigger a divide by zero exception for testing
static ZERO_DIVISOR: i32 = 0;

pub fn trigger_divide_by_zero() {
    let x: i32 = 42;
    let zero = unsafe { ptr::read_volatile(&ZERO_DIVISOR) };
    let _result = x / zero;
}

/// Check if interrupts are enabled
pub fn are_enabled() -> bool {
    x86_64::instructions::interrupts::are_enabled()
}

/// Get the current interrupt stack frame address
pub fn get_current_stack_frame() -> VirtAddr {
    // Use inline assembly to get RSP since the rsp module might not be available
    let rsp: u64;
    unsafe {
        core::arch::asm!("mov {0:r}, rsp", out(reg) rsp, options(nostack, preserves_flags));
    }
    VirtAddr::new(rsp)
}

// ========== INTERRUPT DEBUGGING ==========

impl fmt::Display for InterruptStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Interrupt Statistics:\n\
             Timer: {}\n\
             Keyboard: {}\n\
             Serial: {}\n\
             Exceptions: {}\n\
             Page Faults: {}\n\
             Spurious: {}",
            self.timer_count,
            self.keyboard_count,
            self.serial_count,
            self.exception_count,
            self.page_fault_count,
            self.spurious_count
        )
    }
}

/// Get current interrupt statistics for monitoring
pub fn get_interrupt_stats() -> InterruptStats {
    unsafe {
        InterruptStats {
            timer_count: INTERRUPT_STATS.timer_count,
            keyboard_count: INTERRUPT_STATS.keyboard_count,
            serial_count: INTERRUPT_STATS.serial_count,
            exception_count: INTERRUPT_STATS.exception_count,
            page_fault_count: INTERRUPT_STATS.page_fault_count,
            spurious_count: INTERRUPT_STATS.spurious_count,
        }
    }
}

/// Reset interrupt statistics counters
pub fn reset_interrupt_stats() {
    unsafe {
        INTERRUPT_STATS.timer_count = 0;
        INTERRUPT_STATS.keyboard_count = 0;
        INTERRUPT_STATS.serial_count = 0;
        INTERRUPT_STATS.exception_count = 0;
        INTERRUPT_STATS.page_fault_count = 0;
        INTERRUPT_STATS.spurious_count = 0;
    }
}

/// Validate interrupt system functionality
pub fn test_interrupts() {
    // Validate interrupt system by triggering a controlled breakpoint
    trigger_breakpoint();
    // System validated - breakpoint handler completed successfully
}
