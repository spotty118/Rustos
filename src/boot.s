// Multiboot header for RustOS
// This file provides the multiboot headers required for GRUB/QEMU to boot RustOS

.set ALIGN,    1<<0             // align loaded modules on page boundaries
.set MEMINFO,  1<<1             // provide memory map  
.set FLAGS,    ALIGN | MEMINFO  // this is the Multiboot 'flag' field
.set MAGIC,    0x1BADB002       // 'magic number' lets bootloader find the header
.set CHECKSUM, -(MAGIC + FLAGS) // checksum of above, to prove we are multiboot

// Declare a multiboot header that marks the program as a kernel
.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

// Reserve a stack for the initial thread.
.section .bss
.align 16
stack_bottom:
.skip 16384 // 16 KiB
stack_top:

// The kernel entry point.
.section .text
.global _start
.type _start, @function
_start:
    movl $stack_top, %esp
    
    // Transfer control to the main kernel.
    call rust_main
    
    // Hang if rust_main unexpectedly returns.
    cli
1:  hlt
    jmp 1b
.size _start, . - _start
