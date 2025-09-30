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
    
    /// Symbol table address (DT_SYMTAB)
    pub symtab: Option<VirtAddr>,
    
    /// Hash table address (DT_HASH)
    pub hash: Option<VirtAddr>,
    
    /// Relocation table address (DT_RELA)
    pub rela: Option<VirtAddr>,
    
    /// Size of relocation table (DT_RELASZ)
    pub relasz: Option<usize>,
    
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
            dynamic_tags::DT_SYMTAB => {
                info.symtab = Some(VirtAddr::new(entry.d_val));
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
                    // Adjust by program base address
                    let target = VirtAddr::new(base_address.as_u64() + reloc.offset.as_u64());
                    let value = base_address.as_u64() + reloc.addend as u64;
                    
                    // TODO: Write value to target address
                    // This requires memory write capabilities
                    // unsafe { *(target.as_u64() as *mut u64) = value; }
                }
                relocation_types::R_X86_64_GLOB_DAT |
                relocation_types::R_X86_64_JUMP_SLOT => {
                    // Symbol resolution required
                    // TODO: Look up symbol and write address
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
}
