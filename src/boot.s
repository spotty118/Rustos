# Multiboot header
.set ALIGN,    1<<0             # align loaded modules on page boundaries
.set MEMINFO,  1<<1             # provide memory map
.set FLAGS,    ALIGN | MEMINFO  # multiboot 'flag' field
.set MAGIC,    0x1BADB002       # magic number lets bootloader find the header
.set CHECKSUM, -(MAGIC + FLAGS) # checksum required to prove we are multiboot

# Multiboot header section
.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

# Stack section
.section .bss
.align 16
stack_bottom:
.skip 16384 # 16 KiB
stack_top:

# Entry point
.section .text
.global _start
.type _start, @function
_start:
    mov $stack_top, %esp
    call rust_main
    cli
1:  hlt
    jmp 1b
.size _start, . - _start
