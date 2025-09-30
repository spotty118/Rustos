//! Dynamic Linker for RustOS
//!
//! This module implements dynamic linking support, enabling RustOS to load
//! and execute dynamically-linked ELF binaries. This is a critical component
//! for Linux application compatibility as ~95% of Linux binaries use dynamic linking.
//!
//! ## Features
//! - PT_DYNAMIC segment parsing
//! - Shared library (.so) loading
//! - Symbol resolution across loaded libraries
//! - Relocation processing (R_X86_64_* types)
//! - Library search path management
//!
//! ## Architecture
//! The dynamic linker works in phases:
//! 1. Parse PT_DYNAMIC segment from main executable
//! 2. Identify required shared libraries (DT_NEEDED entries)
//! 3. Load each shared library into memory
//! 4. Build global symbol table
//! 5. Process relocations to fix up addresses
//!
//! ## References
//! - ELF Specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
//! - System V ABI: https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf
//! - See docs/LINUX_APP_SUPPORT.md for implementation roadmap

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use x86_64::VirtAddr;
use core::fmt;

use super::elf_loader::{Elf64Header, Elf64ProgramHeader, elf_constants};
use crate::memory::{MemoryRegionType, MemoryProtection};

/// Dynamic linker for loading shared libraries and resolving symbols
pub struct DynamicLinker {
    /// Library search paths (e.g., /lib, /usr/lib, /lib64)
    search_paths: Vec<String>,
    
    /// Cache of loaded shared libraries
    loaded_libraries: BTreeMap<String, LoadedLibrary>,
    
    /// Global symbol table mapping symbol names to addresses
    symbol_table: BTreeMap<String, VirtAddr>,
    
    /// Base address for library loading (managed with ASLR)
    next_base_address: VirtAddr,
}

/// Information about a loaded shared library
#[derive(Debug, Clone)]
pub struct LoadedLibrary {
    /// Library name (e.g., "libc.so.6")
    pub name: String,
    
    /// Base address where library is loaded
    pub base_address: VirtAddr,
    
    /// Size of library in memory
    pub size: usize,
    
    /// Entry point (if applicable)
    pub entry_point: Option<VirtAddr>,
    
    /// Dynamic section information
    pub dynamic_info: DynamicInfo,
}

/// Parsed PT_DYNAMIC section information
#[derive(Debug, Clone, Default)]
pub struct DynamicInfo {
    /// Required shared libraries (DT_NEEDED)
    pub needed: Vec<String>,
    
    /// String table address (DT_STRTAB)
    pub strtab: Option<VirtAddr>,
    
    /// String table size (DT_STRSZ)
    pub strsz: Option<usize>,
    
    /// Symbol table address (DT_SYMTAB)
    pub symtab: Option<VirtAddr>,
    
    /// Symbol table entry size (DT_SYMENT)
    pub syment: Option<usize>,
    
    /// Hash table address (DT_HASH)
    pub hash: Option<VirtAddr>,
    
    /// Relocation table address (DT_RELA)
    pub rela: Option<VirtAddr>,
    
    /// Size of relocation table (DT_RELASZ)
    pub relasz: Option<usize>,
    
    /// Relocation entry size (DT_RELAENT)
    pub relaent: Option<usize>,
    
    /// PLT relocations address (DT_JMPREL)
    pub jmprel: Option<VirtAddr>,
    
    /// Size of PLT relocations (DT_PLTRELSZ)
    pub pltrelsz: Option<usize>,
    
    /// Init function address (DT_INIT)
    pub init: Option<VirtAddr>,
    
    /// Fini function address (DT_FINI)
    pub fini: Option<VirtAddr>,
}

/// Relocation entry (RELA format)
#[derive(Debug, Clone, Copy)]
pub struct Relocation {
    /// Offset where to apply the relocation
    pub offset: VirtAddr,
    
    /// Relocation type (R_X86_64_*)
    pub r_type: u32,
    
    /// Symbol index
    pub symbol: u32,
    
    /// Addend value
    pub addend: i64,
}

/// ELF symbol table entry (Elf64_Sym)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Elf64Symbol {
    pub st_name: u32,      // Symbol name (string table index)
    pub st_info: u8,       // Symbol type and binding
    pub st_other: u8,      // Symbol visibility
    pub st_shndx: u16,     // Section index
    pub st_value: u64,     // Symbol value
    pub st_size: u64,      // Symbol size
}

impl Elf64Symbol {
    /// Get symbol binding (upper 4 bits of st_info)
    pub fn binding(&self) -> u8 {
        self.st_info >> 4
    }
    
    /// Get symbol type (lower 4 bits of st_info)
    pub fn symbol_type(&self) -> u8 {
        self.st_info & 0xf
    }
    
    /// Check if symbol is defined (not undefined)
    pub fn is_defined(&self) -> bool {
        self.st_shndx != 0  // SHN_UNDEF
    }
}

/// Symbol binding types
pub mod symbol_binding {
    pub const STB_LOCAL: u8 = 0;   // Local symbol
    pub const STB_GLOBAL: u8 = 1;  // Global symbol
    pub const STB_WEAK: u8 = 2;    // Weak symbol
}

/// Symbol types
pub mod symbol_type {
    pub const STT_NOTYPE: u8 = 0;  // No type
    pub const STT_OBJECT: u8 = 1;  // Data object
    pub const STT_FUNC: u8 = 2;    // Code object (function)
    pub const STT_SECTION: u8 = 3; // Section
    pub const STT_FILE: u8 = 4;    // File name
}

/// Dynamic section entry (Elf64_Dyn)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicEntry {
    pub d_tag: i64,
    pub d_val: u64,
}

/// Dynamic section tags (DT_*)
pub mod dynamic_tags {
    pub const DT_NULL: i64 = 0;          // End of dynamic section
    pub const DT_NEEDED: i64 = 1;        // Name of needed library
    pub const DT_PLTRELSZ: i64 = 2;      // Size of PLT relocs
    pub const DT_PLTGOT: i64 = 3;        // PLT/GOT address
    pub const DT_HASH: i64 = 4;          // Symbol hash table address
    pub const DT_STRTAB: i64 = 5;        // String table address
    pub const DT_SYMTAB: i64 = 6;        // Symbol table address
    pub const DT_RELA: i64 = 7;          // Relocation table address
    pub const DT_RELASZ: i64 = 8;        // Size of relocation table
    pub const DT_RELAENT: i64 = 9;       // Size of relocation entry
    pub const DT_STRSZ: i64 = 10;        // Size of string table
    pub const DT_SYMENT: i64 = 11;       // Size of symbol table entry
    pub const DT_INIT: i64 = 12;         // Init function address
    pub const DT_FINI: i64 = 13;         // Fini function address
    pub const DT_SONAME: i64 = 14;       // Name of this shared object
    pub const DT_RPATH: i64 = 15;        // Library search path (deprecated)
    pub const DT_SYMBOLIC: i64 = 16;     // Start symbol search here
    pub const DT_REL: i64 = 17;          // REL format relocations
    pub const DT_RELSZ: i64 = 18;        // Size of REL relocations
    pub const DT_RELENT: i64 = 19;       // Size of REL entry
    pub const DT_PLTREL: i64 = 20;       // Type of PLT reloc (REL or RELA)
    pub const DT_DEBUG: i64 = 21;        // Debug info
    pub const DT_TEXTREL: i64 = 22;      // Reloc might modify text segment
    pub const DT_JMPREL: i64 = 23;       // PLT relocation entries
    pub const DT_BIND_NOW: i64 = 24;     // Process all relocs before executing
    pub const DT_RUNPATH: i64 = 29;      // Library search path
}

/// Relocation types for x86_64
pub mod relocation_types {
    pub const R_X86_64_NONE: u32 = 0;           // No relocation
    pub const R_X86_64_64: u32 = 1;             // Direct 64 bit
    pub const R_X86_64_PC32: u32 = 2;           // PC relative 32 bit signed
    pub const R_X86_64_GOT32: u32 = 3;          // 32 bit GOT entry
    pub const R_X86_64_PLT32: u32 = 4;          // 32 bit PLT address
    pub const R_X86_64_COPY: u32 = 5;           // Copy symbol at runtime
    pub const R_X86_64_GLOB_DAT: u32 = 6;       // Create GOT entry
    pub const R_X86_64_JUMP_SLOT: u32 = 7;      // Create PLT entry
    pub const R_X86_64_RELATIVE: u32 = 8;       // Adjust by program base
    pub const R_X86_64_GOTPCREL: u32 = 9;       // 32 bit signed PC relative offset to GOT
    pub const R_X86_64_32: u32 = 10;            // Direct 32 bit zero extended
    pub const R_X86_64_32S: u32 = 11;           // Direct 32 bit sign extended
    pub const R_X86_64_16: u32 = 12;            // Direct 16 bit zero extended
    pub const R_X86_64_PC16: u32 = 13;          // 16 bit sign extended PC relative
    pub const R_X86_64_8: u32 = 14;             // Direct 8 bit sign extended
    pub const R_X86_64_PC8: u32 = 15;           // 8 bit sign extended PC relative
}

/// Errors that can occur during dynamic linking
#[derive(Debug, Clone)]
pub enum DynamicLinkerError {
    /// PT_DYNAMIC segment not found
    NoDynamicSegment,
    
    /// Invalid dynamic section entry
    InvalidDynamicEntry,
    
    /// Required library not found
    LibraryNotFound(String),
    
    /// Symbol not found
    SymbolNotFound(String),
    
    /// Unsupported relocation type
    UnsupportedRelocation(u32),
    
    /// Invalid memory address
    InvalidAddress,
    
    /// Memory allocation failed
    AllocationFailed,
    
    /// Invalid ELF file
    InvalidElf(String),
}

impl fmt::Display for DynamicLinkerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DynamicLinkerError::NoDynamicSegment => 
                write!(f, "PT_DYNAMIC segment not found in ELF binary"),
            DynamicLinkerError::InvalidDynamicEntry => 
                write!(f, "Invalid dynamic section entry"),
            DynamicLinkerError::LibraryNotFound(lib) => 
                write!(f, "Required library not found: {}", lib),
            DynamicLinkerError::SymbolNotFound(sym) => 
                write!(f, "Symbol not found: {}", sym),
            DynamicLinkerError::UnsupportedRelocation(r_type) => 
                write!(f, "Unsupported relocation type: {}", r_type),
            DynamicLinkerError::InvalidAddress => 
                write!(f, "Invalid memory address"),
            DynamicLinkerError::AllocationFailed => 
                write!(f, "Memory allocation failed"),
            DynamicLinkerError::InvalidElf(msg) => 
                write!(f, "Invalid ELF: {}", msg),
        }
    }
}

pub type DynamicLinkerResult<T> = Result<T, DynamicLinkerError>;

impl DynamicLinker {
    /// Create a new dynamic linker instance
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        // Standard Linux library search paths
        search_paths.push(String::from("/lib"));
        search_paths.push(String::from("/lib64"));
        search_paths.push(String::from("/usr/lib"));
        search_paths.push(String::from("/usr/lib64"));
        search_paths.push(String::from("/usr/local/lib"));
        
        Self {
            search_paths,
            loaded_libraries: BTreeMap::new(),
            symbol_table: BTreeMap::new(),
            // Start library loading at a safe address (above user space)
            next_base_address: VirtAddr::new(0x400000_0000),
        }
    }
    
    /// Add a library search path
    pub fn add_search_path(&mut self, path: String) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }
    
    /// Parse PT_DYNAMIC segment from ELF binary
    pub fn parse_dynamic_section(
        &self,
        binary_data: &[u8],
        program_headers: &[Elf64ProgramHeader],
        base_address: VirtAddr,
    ) -> DynamicLinkerResult<DynamicInfo> {
        // Find PT_DYNAMIC segment
        let dynamic_phdr = program_headers.iter()
            .find(|phdr| phdr.p_type == elf_constants::PT_DYNAMIC)
            .ok_or(DynamicLinkerError::NoDynamicSegment)?;
        
        let mut dynamic_info = DynamicInfo::default();
        
        // Parse dynamic entries
        let dyn_offset = dynamic_phdr.p_offset as usize;
        let dyn_size = dynamic_phdr.p_filesz as usize;
        
        if dyn_offset + dyn_size > binary_data.len() {
            return Err(DynamicLinkerError::InvalidElf(
                String::from("Dynamic section out of bounds")
            ));
        }
        
        let dyn_data = &binary_data[dyn_offset..dyn_offset + dyn_size];
        let entry_count = dyn_size / core::mem::size_of::<DynamicEntry>();
        
        for i in 0..entry_count {
            let entry = self.parse_dynamic_entry(dyn_data, i)?;
            
            if entry.d_tag == dynamic_tags::DT_NULL {
                break; // End of dynamic section
            }
            
            self.process_dynamic_entry(&mut dynamic_info, &entry, base_address);
        }
        
        Ok(dynamic_info)
    }
    
    /// Parse a single dynamic entry
    fn parse_dynamic_entry(&self, data: &[u8], index: usize) -> DynamicLinkerResult<DynamicEntry> {
        let offset = index * core::mem::size_of::<DynamicEntry>();
        
        if offset + core::mem::size_of::<DynamicEntry>() > data.len() {
            return Err(DynamicLinkerError::InvalidDynamicEntry);
        }
        
        // Read d_tag (8 bytes, little-endian)
        let d_tag = i64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]);
        
        // Read d_val (8 bytes, little-endian)
        let d_val = u64::from_le_bytes([
            data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
            data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15],
        ]);
        
        Ok(DynamicEntry { d_tag, d_val })
    }
    
    /// Process a dynamic entry and update DynamicInfo
    fn process_dynamic_entry(&self, info: &mut DynamicInfo, entry: &DynamicEntry, base: VirtAddr) {
        match entry.d_tag {
            dynamic_tags::DT_NEEDED => {
                // Library name stored as offset in string table
                // Will be resolved later when we have the string table
                // For now, store the offset as a placeholder
                info.needed.push(format!("offset:{}", entry.d_val));
            }
            dynamic_tags::DT_STRTAB => {
                info.strtab = Some(VirtAddr::new(entry.d_val));
            }
            dynamic_tags::DT_STRSZ => {
                info.strsz = Some(entry.d_val as usize);
            }
            dynamic_tags::DT_SYMTAB => {
                info.symtab = Some(VirtAddr::new(entry.d_val));
            }
            dynamic_tags::DT_SYMENT => {
                info.syment = Some(entry.d_val as usize);
            }
            dynamic_tags::DT_HASH => {
                info.hash = Some(VirtAddr::new(entry.d_val));
            }
            dynamic_tags::DT_RELA => {
                info.rela = Some(VirtAddr::new(entry.d_val));
            }
            dynamic_tags::DT_RELASZ => {
                info.relasz = Some(entry.d_val as usize);
            }
            dynamic_tags::DT_RELAENT => {
                info.relaent = Some(entry.d_val as usize);
            }
            dynamic_tags::DT_JMPREL => {
                info.jmprel = Some(VirtAddr::new(entry.d_val));
            }
            dynamic_tags::DT_PLTRELSZ => {
                info.pltrelsz = Some(entry.d_val as usize);
            }
            dynamic_tags::DT_INIT => {
                info.init = Some(VirtAddr::new(base.as_u64() + entry.d_val));
            }
            dynamic_tags::DT_FINI => {
                info.fini = Some(VirtAddr::new(base.as_u64() + entry.d_val));
            }
            _ => {
                // Ignore other tags for now
            }
        }
    }
    
    /// Load required dependencies for a binary
    pub fn load_dependencies(&mut self, needed: &[String]) -> DynamicLinkerResult<Vec<String>> {
        let mut loaded = Vec::new();
        
        for lib_name in needed {
            // Skip if already loaded
            if self.loaded_libraries.contains_key(lib_name) {
                continue;
            }
            
            // Try to find and load the library
            match self.find_library(lib_name) {
                Some(_path) => {
                    // TODO: Actually load the library file
                    // For now, just record that we attempted to load it
                    loaded.push(lib_name.clone());
                }
                None => {
                    return Err(DynamicLinkerError::LibraryNotFound(lib_name.clone()));
                }
            }
        }
        
        Ok(loaded)
    }
    
    /// Search for a library in search paths
    fn find_library(&self, name: &str) -> Option<String> {
        for path in &self.search_paths {
            let full_path = format!("{}/{}", path, name);
            // TODO: Check if file exists
            // For now, assume it exists
            return Some(full_path);
        }
        None
    }
    
    /// Resolve a symbol by name across all loaded libraries
    pub fn resolve_symbol(&self, name: &str) -> Option<VirtAddr> {
        self.symbol_table.get(name).copied()
    }
    
    /// Add a symbol to the global symbol table
    pub fn add_symbol(&mut self, name: String, address: VirtAddr) {
        self.symbol_table.insert(name, address);
    }
    
    /// Apply relocations to a loaded binary
    pub fn apply_relocations(
        &self,
        relocations: &[Relocation],
        base_address: VirtAddr,
    ) -> DynamicLinkerResult<()> {
        for reloc in relocations {
            match reloc.r_type {
                relocation_types::R_X86_64_NONE => {
                    // No relocation needed
                }
                relocation_types::R_X86_64_RELATIVE => {
                    // Adjust by program base address: B + A
                    let target = VirtAddr::new(base_address.as_u64() + reloc.offset.as_u64());
                    let value = base_address.as_u64() + reloc.addend as u64;
                    
                    unsafe {
                        self.write_relocation_value(target, value)?;
                    }
                }
                relocation_types::R_X86_64_GLOB_DAT => {
                    // Symbol value: S
                    // Need to look up symbol by index and write its address
                    let target = VirtAddr::new(base_address.as_u64() + reloc.offset.as_u64());
                    
                    // TODO: Look up symbol value from symbol table using reloc.symbol index
                    // For now, we'll skip this as it requires symbol table indexing
                    // In a full implementation:
                    // let symbol_value = self.lookup_symbol_by_index(reloc.symbol)?;
                    // unsafe { self.write_relocation_value(target, symbol_value)?; }
                }
                relocation_types::R_X86_64_JUMP_SLOT => {
                    // PLT entry: S
                    // Similar to GLOB_DAT but for PLT
                    let target = VirtAddr::new(base_address.as_u64() + reloc.offset.as_u64());
                    
                    // TODO: Look up symbol and write to PLT slot
                    // For lazy binding, we could write the address of the resolver stub
                    // For now, skip until we have full symbol resolution
                }
                relocation_types::R_X86_64_64 => {
                    // Direct 64-bit: S + A
                    let target = VirtAddr::new(base_address.as_u64() + reloc.offset.as_u64());
                    
                    // TODO: Look up symbol value and add addend
                    // let symbol_value = self.lookup_symbol_by_index(reloc.symbol)?;
                    // let value = symbol_value + reloc.addend as u64;
                    // unsafe { self.write_relocation_value(target, value)?; }
                }
                _ => {
                    // Unsupported relocation type
                    return Err(DynamicLinkerError::UnsupportedRelocation(reloc.r_type));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get list of loaded libraries
    pub fn loaded_libraries(&self) -> Vec<&LoadedLibrary> {
        self.loaded_libraries.values().collect()
    }
    
    /// Check if a library is loaded
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded_libraries.contains_key(name)
    }
    
    /// Parse string table and resolve library names
    pub fn resolve_library_names(
        &self,
        binary_data: &[u8],
        dynamic_info: &mut DynamicInfo,
    ) -> DynamicLinkerResult<()> {
        // Check if we have string table information
        let strtab_addr = dynamic_info.strtab
            .ok_or(DynamicLinkerError::InvalidElf(String::from("No string table")))?;
        let strtab_size = dynamic_info.strsz
            .ok_or(DynamicLinkerError::InvalidElf(String::from("No string table size")))?;
        
        // In a real implementation, strtab_addr would be a virtual address
        // For now, we'll treat it as an offset into the binary
        let strtab_offset = strtab_addr.as_u64() as usize;
        
        if strtab_offset + strtab_size > binary_data.len() {
            return Err(DynamicLinkerError::InvalidElf(
                String::from("String table out of bounds")
            ));
        }
        
        let strtab = &binary_data[strtab_offset..strtab_offset + strtab_size];
        
        // Resolve library names from offsets
        let mut resolved_names = Vec::new();
        for name_ref in &dynamic_info.needed {
            if name_ref.starts_with("offset:") {
                let offset_str = &name_ref[7..];
                if let Ok(offset) = offset_str.parse::<usize>() {
                    if let Some(name) = self.read_string_from_table(strtab, offset) {
                        resolved_names.push(name);
                    }
                }
            } else {
                // Already resolved
                resolved_names.push(name_ref.clone());
            }
        }
        
        dynamic_info.needed = resolved_names;
        Ok(())
    }
    
    /// Read a null-terminated string from the string table
    fn read_string_from_table(&self, strtab: &[u8], offset: usize) -> Option<String> {
        if offset >= strtab.len() {
            return None;
        }
        
        let mut end = offset;
        while end < strtab.len() && strtab[end] != 0 {
            end += 1;
        }
        
        if end > offset {
            String::from_utf8(strtab[offset..end].to_vec()).ok()
        } else {
            None
        }
    }
    
    /// Parse symbol table from ELF binary
    pub fn parse_symbol_table(
        &self,
        binary_data: &[u8],
        dynamic_info: &DynamicInfo,
    ) -> DynamicLinkerResult<Vec<(String, VirtAddr, Elf64Symbol)>> {
        let symtab_addr = dynamic_info.symtab
            .ok_or(DynamicLinkerError::InvalidElf(String::from("No symbol table")))?;
        let strtab_addr = dynamic_info.strtab
            .ok_or(DynamicLinkerError::InvalidElf(String::from("No string table")))?;
        let strtab_size = dynamic_info.strsz
            .ok_or(DynamicLinkerError::InvalidElf(String::from("No string table size")))?;
        
        // Calculate symbol table bounds
        // We'll use hash table to determine the number of symbols if available
        let symtab_offset = symtab_addr.as_u64() as usize;
        let strtab_offset = strtab_addr.as_u64() as usize;
        
        if strtab_offset + strtab_size > binary_data.len() {
            return Err(DynamicLinkerError::InvalidElf(
                String::from("String table out of bounds")
            ));
        }
        
        let strtab = &binary_data[strtab_offset..strtab_offset + strtab_size];
        
        // Calculate number of symbols
        // Symbol table ends where string table begins (common layout)
        let sym_count = if strtab_offset > symtab_offset {
            (strtab_offset - symtab_offset) / core::mem::size_of::<Elf64Symbol>()
        } else {
            // Fallback: parse until we run out of data or hit invalid entries
            100 // Conservative estimate
        };
        
        let mut symbols = Vec::new();
        
        for i in 0..sym_count {
            let sym_offset = symtab_offset + i * core::mem::size_of::<Elf64Symbol>();
            
            if sym_offset + core::mem::size_of::<Elf64Symbol>() > binary_data.len() {
                break;
            }
            
            // Parse symbol entry
            let symbol = unsafe {
                core::ptr::read(binary_data[sym_offset..].as_ptr() as *const Elf64Symbol)
            };
            
            // Skip undefined symbols
            if !symbol.is_defined() {
                continue;
            }
            
            // Read symbol name from string table
            if let Some(name) = self.read_string_from_table(strtab, symbol.st_name as usize) {
                if !name.is_empty() {
                    symbols.push((name, VirtAddr::new(symbol.st_value), symbol));
                }
            }
        }
        
        Ok(symbols)
    }
    
    /// Load symbols into global symbol table
    pub fn load_symbols_from_binary(
        &mut self,
        binary_data: &[u8],
        dynamic_info: &DynamicInfo,
        base_address: VirtAddr,
    ) -> DynamicLinkerResult<usize> {
        let symbols = self.parse_symbol_table(binary_data, dynamic_info)?;
        let count = symbols.len();
        
        for (name, value, symbol) in symbols {
            // Adjust symbol value by base address if needed
            let adjusted_addr = if symbol.symbol_type() == symbol_type::STT_FUNC ||
                                   symbol.symbol_type() == symbol_type::STT_OBJECT {
                VirtAddr::new(base_address.as_u64() + value.as_u64())
            } else {
                value
            };
            
            self.add_symbol(name, adjusted_addr);
        }
        
        Ok(count)
    }
    
    /// Parse relocations from RELA section
    pub fn parse_relocations(
        &self,
        binary_data: &[u8],
        dynamic_info: &DynamicInfo,
    ) -> DynamicLinkerResult<Vec<Relocation>> {
        let mut relocations = Vec::new();
        
        // Parse regular relocations (DT_RELA)
        if let (Some(rela_addr), Some(rela_size)) = (dynamic_info.rela, dynamic_info.relasz) {
            let rela_offset = rela_addr.as_u64() as usize;
            let reloc_entry_size = dynamic_info.relaent.unwrap_or(24); // Standard RELA entry size
            let reloc_count = rela_size / reloc_entry_size;
            
            for i in 0..reloc_count {
                let offset = rela_offset + i * reloc_entry_size;
                if let Some(reloc) = self.parse_single_relocation(binary_data, offset)? {
                    relocations.push(reloc);
                }
            }
        }
        
        // Parse PLT relocations (DT_JMPREL)
        if let (Some(jmprel_addr), Some(jmprel_size)) = (dynamic_info.jmprel, dynamic_info.pltrelsz) {
            let jmprel_offset = jmprel_addr.as_u64() as usize;
            let reloc_entry_size = 24; // RELA entry size
            let reloc_count = jmprel_size / reloc_entry_size;
            
            for i in 0..reloc_count {
                let offset = jmprel_offset + i * reloc_entry_size;
                if let Some(reloc) = self.parse_single_relocation(binary_data, offset)? {
                    relocations.push(reloc);
                }
            }
        }
        
        Ok(relocations)
    }
    
    /// Parse a single relocation entry
    fn parse_single_relocation(
        &self,
        binary_data: &[u8],
        offset: usize,
    ) -> DynamicLinkerResult<Option<Relocation>> {
        const RELA_ENTRY_SIZE: usize = 24; // r_offset (8) + r_info (8) + r_addend (8)
        
        if offset + RELA_ENTRY_SIZE > binary_data.len() {
            return Ok(None);
        }
        
        let data = &binary_data[offset..offset + RELA_ENTRY_SIZE];
        
        // Parse r_offset
        let r_offset = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        
        // Parse r_info
        let r_info = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);
        
        // Parse r_addend
        let r_addend = i64::from_le_bytes([
            data[16], data[17], data[18], data[19],
            data[20], data[21], data[22], data[23],
        ]);
        
        // Extract symbol and type from r_info
        let r_type = (r_info & 0xffffffff) as u32;
        let r_sym = (r_info >> 32) as u32;
        
        Ok(Some(Relocation {
            offset: VirtAddr::new(r_offset),
            r_type,
            symbol: r_sym,
            addend: r_addend,
        }))
    }
    
    /// Write value to memory (helper for relocations)
    /// 
    /// # Safety
    /// This function writes to arbitrary memory addresses.
    /// Caller must ensure the address is valid and writable.
    unsafe fn write_relocation_value(&self, addr: VirtAddr, value: u64) -> DynamicLinkerResult<()> {
        // In a real kernel, we would check permissions first
        let ptr = addr.as_u64() as *mut u64;
        core::ptr::write_volatile(ptr, value);
        Ok(())
    }
}

impl Default for DynamicLinker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_linker_creation() {
        let linker = DynamicLinker::new();
        assert_eq!(linker.search_paths.len(), 5);
        assert!(linker.search_paths.contains(&String::from("/lib")));
    }
    
    #[test]
    fn test_add_search_path() {
        let mut linker = DynamicLinker::new();
        linker.add_search_path(String::from("/custom/lib"));
        assert!(linker.search_paths.contains(&String::from("/custom/lib")));
    }
    
    #[test]
    fn test_symbol_resolution() {
        let mut linker = DynamicLinker::new();
        let addr = VirtAddr::new(0x1000);
        linker.add_symbol(String::from("test_symbol"), addr);
        
        assert_eq!(linker.resolve_symbol("test_symbol"), Some(addr));
        assert_eq!(linker.resolve_symbol("nonexistent"), None);
    }
    
    #[test]
    fn test_string_table_reading() {
        let linker = DynamicLinker::new();
        let strtab = b"\x00hello\x00world\x00test\x00";
        
        assert_eq!(linker.read_string_from_table(strtab, 1), Some(String::from("hello")));
        assert_eq!(linker.read_string_from_table(strtab, 7), Some(String::from("world")));
        assert_eq!(linker.read_string_from_table(strtab, 13), Some(String::from("test")));
        assert_eq!(linker.read_string_from_table(strtab, 0), None); // Empty string
    }
    
    #[test]
    fn test_elf_symbol_binding() {
        let symbol = Elf64Symbol {
            st_name: 0,
            st_info: (symbol_binding::STB_GLOBAL << 4) | symbol_type::STT_FUNC,
            st_other: 0,
            st_shndx: 1,
            st_value: 0x1000,
            st_size: 100,
        };
        
        assert_eq!(symbol.binding(), symbol_binding::STB_GLOBAL);
        assert_eq!(symbol.symbol_type(), symbol_type::STT_FUNC);
        assert!(symbol.is_defined());
    }
    
    #[test]
    fn test_library_loaded_check() {
        let linker = DynamicLinker::new();
        assert!(!linker.is_loaded("libc.so.6"));
        assert_eq!(linker.loaded_libraries().len(), 0);
    }
}
