//! Compression format support for package extraction
//!
//! This module provides decompression utilities for common package formats.

pub mod gzip;
pub mod tar;

pub use gzip::GzipDecoder;
pub use tar::TarArchive;

use alloc::vec::Vec;
use crate::package::{PackageResult, PackageError};

/// Compression format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// XZ/LZMA compression
    Xz,
    /// Zstandard compression
    Zstd,
    /// Bzip2 compression
    Bzip2,
}

impl CompressionFormat {
    /// Detect compression format from magic bytes
    pub fn detect(data: &[u8]) -> Self {
        if data.len() < 4 {
            return CompressionFormat::None;
        }

        // Gzip: 1f 8b
        if data[0] == 0x1f && data[1] == 0x8b {
            return CompressionFormat::Gzip;
        }

        // XZ: fd 37 7a 58 5a 00
        if data.len() >= 6 && data[0] == 0xfd && data[1] == 0x37
            && data[2] == 0x7a && data[3] == 0x58 && data[4] == 0x5a && data[5] == 0x00 {
            return CompressionFormat::Xz;
        }

        // Zstd: 28 b5 2f fd
        if data[0] == 0x28 && data[1] == 0xb5 && data[2] == 0x2f && data[3] == 0xfd {
            return CompressionFormat::Zstd;
        }

        // Bzip2: 42 5a 68
        if data[0] == 0x42 && data[1] == 0x5a && data[2] == 0x68 {
            return CompressionFormat::Bzip2;
        }

        CompressionFormat::None
    }
}

/// Decompress data based on detected format
pub fn decompress(data: &[u8]) -> PackageResult<Vec<u8>> {
    let format = CompressionFormat::detect(data);

    match format {
        CompressionFormat::Gzip => GzipDecoder::decode(data),
        CompressionFormat::Xz => Err(PackageError::NotImplemented(
            "XZ decompression not yet implemented. Consider using libxz port.".into()
        )),
        CompressionFormat::Zstd => Err(PackageError::NotImplemented(
            "Zstd decompression not yet implemented. Consider using zstd-rs.".into()
        )),
        CompressionFormat::Bzip2 => Err(PackageError::NotImplemented(
            "Bzip2 decompression not yet implemented. Consider using bzip2-rs.".into()
        )),
        CompressionFormat::None => Ok(data.to_vec()),
    }
}
