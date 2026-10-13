//! Unreal Engine 3 packages (`.pcc`) as LE stores them.
//!
//! Layout: a header, then the name, import and export tables, a dependency
//! table, and each export's data. LE compresses everything after the
//! header: the header lists chunks, each chunk starts with its own header
//! and a table of blocks, and each block is a separate Oodle Leviathan
//! stream. Decompressing puts every chunk at its uncompressed offset, so
//! the offsets in the header and tables refer to the decompressed bytes.

use std::collections::BTreeMap;
use std::path::Path;

use crate::reader::Reader;
use crate::{Error, Game, Result};

/// The package tag, also the magic of each compressed chunk.
pub const TAG: u32 = 0x9E2A_83C1;
/// `EPackageFlags::Cooked`.
pub const COOKED_FLAG: u32 = 0x0000_0008;
/// `EPackageFlags::Compressed`.
pub const COMPRESSED_FLAG: u32 = 0x0200_0000;
/// The compression type LE writes (confirmed on every compressed package
/// of a real install).
pub const OODLE_LEVIATHAN: u32 = 0x400;

/// Which game an `(engine, licensee)` version pair belongs to.
pub fn detect_game(unreal: u16, licensee: u16) -> Option<Game> {
    match (unreal, licensee) {
        (684, 171) => Some(Game::Le1),
        (684, 168) => Some(Game::Le2),
        (685, 205) => Some(Game::Le3),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    OodleLeviathan,
    Other(u32),
}

/// Where one compressed chunk sits in the file and in the decompressed
/// package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub uncompressed_offset: usize,
    pub uncompressed_size: usize,
    pub compressed_offset: usize,
    pub compressed_size: usize,
}

#[derive(Debug, Clone)]
pub struct Header {
    pub game: Game,
    pub unreal_version: u16,
    pub licensee_version: u16,
    pub flags: u32,
    pub name_count: usize,
    pub name_offset: usize,
    pub export_count: usize,
    pub export_offset: usize,
    pub import_count: usize,
    pub import_offset: usize,
    pub compression: Compression,
    pub chunks: Vec<Chunk>,
}

impl Header {
    /// Parse the header from the start of a package file. Only the header
    /// itself needs to be present, so a prefix of a large file works.
    pub fn parse(data: &[u8]) -> Result<Header> {
        let mut r = Reader::new(data);
        let tag = r.u32()?;
        if tag != TAG {
            return Err(Error::Format {
                offset: 0,
                message: format!("not an Unreal package (tag {tag:#010x})"),
            });
        }
        let unreal_version = r.u16()?;
        let licensee_version = r.u16()?;
        let game = detect_game(unreal_version, licensee_version).ok_or_else(|| {
            Error::Unsupported(format!(
                "package version {unreal_version}/{licensee_version} is not a Legendary Edition game"
            ))
        })?;
        let _full_header_size = r.i32()?;
        let _folder_name = r.fstring()?;
        let flags = r.u32()?;
        if game == Game::Le3 && flags & COOKED_FLAG != 0 {
            let _package_type = r.i32()?;
        }
        let name_count = r.len("name count")?;
        let name_offset = r.len("name offset")?;
        let export_count = r.len("export count")?;
        let export_offset = r.len("export offset")?;
        let import_count = r.len("import count")?;
        let import_offset = r.len("import offset")?;
        let _depends_offset = r.i32()?;
        // Import/export guids offset and three counts that are always 0.
        r.skip(16)?;
        r.skip(16)?; // package guid
        let generations = r.len("generation count")?;
        r.skip(generations.saturating_mul(12))?;
        r.skip(8)?; // engine and cooked content versions
        r.skip(8)?; // build and branch
        let compression = match r.u32()? {
            0 => Compression::None,
            OODLE_LEVIATHAN => Compression::OodleLeviathan,
            other => Compression::Other(other),
        };
        let chunk_count = r.len("chunk count")?;
        let mut chunks = Vec::with_capacity(chunk_count.min(r.remaining() / 16));
        for _ in 0..chunk_count {
            chunks.push(Chunk {
                uncompressed_offset: r.len("chunk offset")?,
                uncompressed_size: r.len("chunk size")?,
                compressed_offset: r.len("chunk offset")?,
                compressed_size: r.len("chunk size")?,
            });
        }
        Ok(Header {
            game,
            unreal_version,
            licensee_version,
            flags,
            name_count,
            name_offset,
            export_count,
            export_offset,
            import_count,
            import_offset,
            compression,
            chunks,
        })
    }

    pub fn is_compressed(&self) -> bool {
        self.compression != Compression::None && !self.chunks.is_empty()
    }
}

/// A name reference: an index into the name table plus an instance
/// number (`Foo` with number 3 displays as `Foo_2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FName {
    pub index: i32,
    pub number: i32,
}

impl FName {
    fn read(r: &mut Reader) -> Result<FName> {
        Ok(FName {
            index: r.i32()?,
            number: r.i32()?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub package: FName,
    pub class: FName,
    /// Object reference to the import's outer.
    pub outer: i32,
    pub name: FName,
}

#[derive(Debug, Clone)]
pub struct Export {
    /// Object reference to the class (0 means the export is a class).
    pub class: i32,
    pub superclass: i32,
    pub outer: i32,
    pub name: FName,
    pub archetype: i32,
    pub object_flags: u64,
    pub data_size: usize,
    pub data_offset: usize,
}

/// A package with its tables read and its data decompressed.
pub struct Package {
    pub header: Header,
    /// The decompressed package.
    pub data: Vec<u8>,
    pub names: Vec<String>,
    pub imports: Vec<Import>,
    pub exports: Vec<Export>,
}

impl std::fmt::Debug for Package {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Package")
            .field("header", &self.header)
            .field("names", &self.names.len())
            .field("imports", &self.imports.len())
            .field("exports", &self.exports.len())
            .finish()
    }
}

impl Package {
    pub fn read(path: &Path) -> std::result::Result<Package, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Package::parse(bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(file: Vec<u8>) -> Result<Package> {
        let header = Header::parse(&file)?;
        let data = if header.is_compressed() {
            decompress(&file, &header)?
        } else {
            file
        };

        let mut r = Reader::at(&data, header.name_offset);
        let mut names = Vec::with_capacity(header.name_count.min(r.remaining() / 5));
        for _ in 0..header.name_count {
            names.push(r.fstring()?);
        }

        let mut r = Reader::at(&data, header.import_offset);
        let mut imports = Vec::with_capacity(header.import_count.min(r.remaining() / 28));
        for _ in 0..header.import_count {
            imports.push(Import {
                package: FName::read(&mut r)?,
                class: FName::read(&mut r)?,
                outer: r.i32()?,
                name: FName::read(&mut r)?,
            });
        }

        let mut r = Reader::at(&data, header.export_offset);
        let mut exports = Vec::with_capacity(header.export_count.min(r.remaining() / 68));
        for _ in 0..header.export_count {
            let export = Export {
                class: r.i32()?,
                superclass: r.i32()?,
                outer: r.i32()?,
                name: FName::read(&mut r)?,
                archetype: r.i32()?,
                object_flags: r.u64()?,
                data_size: r.len("export size")?,
                data_offset: r.len("export offset")?,
            };
            // No component map: engine versions past 543 dropped it.
            let _export_flags = r.u32()?;
            let net_objects = r.len("net object count")?;
            r.skip(net_objects.saturating_mul(4))?;
            r.skip(16)?; // package guid
            let _package_flags = r.u32()?;
            exports.push(export);
        }

        let package = Package {
            header,
            data,
            names,
            imports,
            exports,
        };
        for (i, e) in package.exports.iter().enumerate() {
            if e.data_offset.saturating_add(e.data_size) > package.data.len() {
                return Err(Error::Format {
                    offset: e.data_offset,
                    message: format!("export {} data runs past the end of the package", i + 1),
                });
            }
        }
        Ok(package)
    }

    /// A name as the game displays it.
    pub fn name(&self, name: FName) -> String {
        let base = usize::try_from(name.index)
            .ok()
            .and_then(|i| self.names.get(i))
            .map_or("<bad name>", String::as_str);
        if name.number > 0 {
            format!("{base}_{}", name.number - 1)
        } else {
            base.to_string()
        }
    }

    /// Whether a name reference spells `s`, ignoring its instance number.
    pub fn name_is(&self, name: FName, s: &str) -> bool {
        usize::try_from(name.index)
            .ok()
            .and_then(|i| self.names.get(i))
            .is_some_and(|n| n.eq_ignore_ascii_case(s))
    }

    /// The object name behind a reference (positive: export, negative:
    /// import, 0: none).
    pub fn object_name(&self, reference: i32) -> Option<String> {
        if reference > 0 {
            let e = self.exports.get(reference as usize - 1)?;
            Some(self.name(e.name))
        } else if reference < 0 {
            let i = self.imports.get(reference.unsigned_abs() as usize - 1)?;
            Some(self.name(i.name))
        } else {
            None
        }
    }

    /// The outer chain of a reference, dotted (`BioD_Nor.Tali_Dlg`).
    pub fn full_path(&self, reference: i32) -> String {
        let mut parts = Vec::new();
        let mut current = reference;
        // Bounded in case a damaged package has an outer loop.
        while current != 0 && parts.len() < 64 {
            let Some(name) = self.object_name(current) else {
                break;
            };
            parts.push(name);
            current = if current > 0 {
                self.exports[current as usize - 1].outer
            } else {
                self.imports[current.unsigned_abs() as usize - 1].outer
            };
        }
        parts.reverse();
        parts.join(".")
    }

    /// The class name of the export at `index` (0-based).
    pub fn class_name(&self, index: usize) -> String {
        match self.exports[index].class {
            0 => "Class".to_string(),
            c => self.object_name(c).unwrap_or_else(|| "<bad class>".into()),
        }
    }

    /// The serialized data of the export at `index` (0-based).
    pub fn export_data(&self, index: usize) -> &[u8] {
        let e = &self.exports[index];
        &self.data[e.data_offset..e.data_offset + e.data_size]
    }

    /// Indices of the exports whose class is `class`.
    pub fn exports_of_class<'a>(&'a self, class: &'a str) -> impl Iterator<Item = usize> + 'a {
        (0..self.exports.len()).filter(move |&i| self.class_name(i).eq_ignore_ascii_case(class))
    }

    /// How many exports there are of each class.
    pub fn class_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for i in 0..self.exports.len() {
            *counts.entry(self.class_name(i)).or_default() += 1;
        }
        counts
    }
}

/// Decompress a package: copy the header, then each chunk's blocks to their
/// uncompressed offsets.
pub fn decompress(file: &[u8], header: &Header) -> Result<Vec<u8>> {
    if let Compression::Other(kind) = header.compression {
        return Err(Error::Unsupported(format!(
            "compression type {kind:#x} (only Oodle Leviathan is supported)"
        )));
    }
    let end = header
        .chunks
        .iter()
        .map(|c| c.uncompressed_offset + c.uncompressed_size)
        .max()
        .unwrap_or(0);
    let first = header
        .chunks
        .iter()
        .map(|c| c.uncompressed_offset)
        .min()
        .unwrap_or(0);
    let mut out = vec![0u8; end];
    let keep = first.min(file.len());
    out[..keep].copy_from_slice(&file[..keep]);

    for (n, chunk) in header.chunks.iter().enumerate() {
        let mut r = Reader::at(file, chunk.compressed_offset);
        let magic = r.u32()?;
        if magic != TAG {
            return Err(r.error(format!("chunk {n} has magic {magic:#010x}")));
        }
        let block_size = r.len("block size")?;
        let _compressed_size = r.i32()?;
        let uncompressed_size = r.len("chunk size")?;
        if block_size == 0 || uncompressed_size != chunk.uncompressed_size {
            return Err(r.error(format!(
                "chunk {n} header disagrees with the chunk table (block size {block_size}, size {uncompressed_size} vs {})",
                chunk.uncompressed_size
            )));
        }
        let block_count = uncompressed_size.div_ceil(block_size);
        let mut blocks = Vec::with_capacity(block_count.min(r.remaining() / 8));
        for _ in 0..block_count {
            blocks.push((r.len("block size")?, r.len("block size")?));
        }
        let mut at = chunk.uncompressed_offset;
        for (b, (compressed, uncompressed)) in blocks.into_iter().enumerate() {
            let input = r.bytes(compressed)?;
            let output = out.get_mut(at..at + uncompressed).ok_or_else(|| {
                Error::Oodle(format!(
                    "chunk {n} block {b} runs past the chunk table's end"
                ))
            })?;
            let written = oozextract::Extractor::new()
                .read_from_slice(input, output)
                .map_err(|e| Error::Oodle(format!("chunk {n} block {b}: {e}")))?;
            if written != uncompressed {
                return Err(Error::Oodle(format!(
                    "chunk {n} block {b}: got {written} bytes, expected {uncompressed}"
                )));
            }
            at += uncompressed;
        }
        if at != chunk.uncompressed_offset + chunk.uncompressed_size {
            return Err(Error::Oodle(format!(
                "chunk {n}: blocks hold {} bytes, chunk says {}",
                at - chunk.uncompressed_offset,
                chunk.uncompressed_size
            )));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::PackageBuilder;

    fn sample(game: Game) -> PackageBuilder {
        let mut p = PackageBuilder::new(game);
        let core = p.import("Core", "Package", "Core", 0);
        let class = p.import("Core", "Class", "BioConversation", core);
        let level = p.export(0, "BioD_Nor_Tali", 0, 0, vec![]);
        // Repeated bytes so some blocks take the memset path.
        p.export(class, "tali_dlg", 0, level, vec![7; 300]);
        p.export(
            class,
            "tali_dlg",
            2,
            level,
            (0..=255u8).cycle().take(500).collect(),
        );
        p
    }

    #[test]
    fn reads_uncompressed_tables() {
        for game in [Game::Le1, Game::Le2, Game::Le3] {
            let pkg = Package::parse(sample(game).build()).unwrap();
            assert_eq!(pkg.header.game, game);
            assert!(!pkg.header.is_compressed());
            assert_eq!(pkg.exports.len(), 3);
            assert_eq!(pkg.class_name(1), "BioConversation");
            assert_eq!(pkg.full_path(3), "BioD_Nor_Tali.tali_dlg_1");
            assert_eq!(pkg.full_path(-2), "Core.BioConversation");
            assert_eq!(pkg.export_data(1), &[7u8; 300][..]);
            assert_eq!(pkg.exports_of_class("bioconversation").count(), 2);
        }
    }

    #[test]
    fn compressed_matches_uncompressed() {
        for game in [Game::Le1, Game::Le2, Game::Le3] {
            let builder = sample(game);
            let plain = Package::parse(builder.build()).unwrap();
            let packed = Package::parse(builder.build_compressed(64)).unwrap();
            assert!(packed.header.is_compressed());
            assert_eq!(packed.header.chunks.len(), 2);
            assert_eq!(packed.names, plain.names);
            for i in 0..plain.exports.len() {
                assert_eq!(packed.export_data(i), plain.export_data(i));
                assert_eq!(packed.class_name(i), plain.class_name(i));
            }
            let start = plain.header.name_offset;
            assert_eq!(packed.data[start..], plain.data[start..]);
        }
    }

    #[test]
    fn reports_damaged_blocks() {
        let mut bytes = sample(Game::Le2).build_compressed(64);
        let header = Header::parse(&bytes).unwrap();
        // Corrupt the first block's Oodle header.
        let c = header.chunks[0];
        let blocks = c.uncompressed_size.div_ceil(64);
        bytes[c.compressed_offset + 16 + blocks * 8] = 0xFF;
        let err = Package::parse(bytes).unwrap_err();
        assert!(matches!(err, Error::Oodle(_)), "{err}");
    }

    #[test]
    fn rejects_other_versions() {
        let mut bytes = sample(Game::Le1).build();
        bytes[4..6].copy_from_slice(&491u16.to_le_bytes());
        assert!(matches!(Header::parse(&bytes), Err(Error::Unsupported(_))));
        assert!(Header::parse(b"not a package").is_err());
    }
}
