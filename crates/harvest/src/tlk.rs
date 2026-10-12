//! Subtitle tables ("talk files"): string id to text.
//!
//! Both layouts store UTF-16 text Huffman-coded, bits read least
//! significant first within each byte, each string ending at a coded NUL.
//!
//! - LE2/LE3 ship standalone `.tlk` files ([`TalkFile::parse`]): a header,
//!   `(string id, bit offset)` entries (male first, then female variants),
//!   the tree as `(left, right)` i32 pairs where a negative id `-1 - c`
//!   is a leaf for code unit `c`, then the bit data.
//! - LE1 keeps them as the binary part of `BioTlkFile` exports inside
//!   packages ([`parse_le1`]): `(id, flags, index)` entries, a tree of
//!   tagged nodes, then one length-prefixed block of bits per string.

use std::collections::BTreeMap;

use crate::reader::Reader;
use crate::{Error, Result};

/// "Tlk\0" read as a little-endian u32.
pub const MAGIC: u32 = 0x006B_6C54;

/// Which variant of a line: some lines have a different text (and
/// recording) depending on Shepard's gender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Gender {
    Male,
    Female,
}

impl Gender {
    /// The suffix used in corpus line ids and stream names.
    pub fn suffix(self) -> &'static str {
        match self {
            Gender::Male => "m",
            Gender::Female => "f",
        }
    }
}

/// A standalone LE2/LE3 `.tlk` file.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TalkFile {
    /// The main table.
    pub male: BTreeMap<i32, String>,
    /// Strings that differ when Shepard is female.
    pub female: BTreeMap<i32, String>,
}

impl TalkFile {
    pub fn parse(data: &[u8]) -> Result<TalkFile> {
        let mut r = Reader::new(data);
        let magic = r.u32()?;
        if magic != MAGIC {
            return Err(Error::Format {
                offset: 0,
                message: format!("not a .tlk file (magic {magic:#010x})"),
            });
        }
        let _version = r.i32()?;
        let _min_version = r.i32()?;
        let male_count = r.len("male entry count")?;
        let female_count = r.len("female entry count")?;
        let node_count = r.len("tree node count")?;
        let data_len = r.len("data length")?;

        let mut entries = Vec::with_capacity((male_count + female_count).min(r.remaining() / 8));
        for _ in 0..male_count + female_count {
            entries.push((r.i32()?, r.i32()?));
        }
        let mut tree = Vec::with_capacity(node_count.min(r.remaining() / 8));
        for _ in 0..node_count {
            tree.push((r.i32()?, r.i32()?));
        }
        let bits = r.bytes(data_len)?;

        let mut decoded: BTreeMap<i32, String> = BTreeMap::new();
        let mut tlk = TalkFile::default();
        for (i, &(id, offset)) in entries.iter().enumerate() {
            // A negative offset is an entry with no text.
            let Ok(bit) = usize::try_from(offset) else {
                continue;
            };
            let text = match decoded.get(&offset) {
                Some(text) => text.clone(),
                None => {
                    let text = decode_pairs(&tree, bits, bit).map_err(|message| Error::Format {
                        offset: 0,
                        message: format!("string {id} at bit {bit}: {message}"),
                    })?;
                    decoded.insert(offset, text.clone());
                    text
                }
            };
            let table = if i < male_count {
                &mut tlk.male
            } else {
                &mut tlk.female
            };
            table.insert(id, text);
        }
        Ok(tlk)
    }

    /// The text for `id` as heard by a Shepard of `gender`, and which
    /// variant it came from: female lookups fall back to the main table.
    pub fn lookup(&self, id: i32, gender: Gender) -> Option<(&str, Gender)> {
        if gender == Gender::Female {
            if let Some(text) = self.female.get(&id) {
                return Some((text, Gender::Female));
            }
        }
        self.male.get(&id).map(|t| (t.as_str(), Gender::Male))
    }
}

fn bit(bits: &[u8], i: usize) -> Option<bool> {
    bits.get(i / 8).map(|b| (b >> (i % 8)) & 1 == 1)
}

/// Decode one string from the LE2/LE3 tree, starting at bit `start`.
fn decode_pairs(
    tree: &[(i32, i32)],
    bits: &[u8],
    start: usize,
) -> std::result::Result<String, String> {
    if tree.is_empty() {
        return Err("empty Huffman tree".into());
    }
    let mut units = Vec::new();
    let mut node = 0usize;
    let mut i = start;
    loop {
        let b = bit(bits, i).ok_or("data ends before the string's NUL")?;
        i += 1;
        let (left, right) = tree[node];
        let next = if b { right } else { left };
        if next < 0 {
            let unit = u16::try_from(-1 - i64::from(next))
                .map_err(|_| format!("leaf {next} is not a UTF-16 code unit"))?;
            if unit == 0 {
                return Ok(String::from_utf16_lossy(&units));
            }
            units.push(unit);
            node = 0;
        } else {
            node = next as usize;
            if node >= tree.len() {
                return Err(format!("node {node} outside the tree"));
            }
        }
    }
}

/// One LE1 tree node.
#[derive(Debug, Clone, Copy)]
enum Node {
    Leaf(u16),
    Branch { left: i16, right: i16 },
}

/// The strings in the binary part of an LE1 `BioTlkFile` export. Each
/// export holds a single variant; the package names them (`tlk`, `tlk_M`,
/// `GlobalTlk_tlk`, ...). **Verify** on real files which name is which
/// gender.
pub fn parse_le1(binary: &[u8]) -> Result<BTreeMap<i32, String>> {
    let mut r = Reader::new(binary);
    let entry_count = r.len("entry count")?;
    let mut entries = Vec::with_capacity(entry_count.min(r.remaining() / 12));
    for _ in 0..entry_count {
        // (string id, flags, index of the string in the data blocks).
        entries.push((r.i32()?, r.i32()?, r.i32()?));
    }
    let node_count = r.len("tree node count")?;
    let mut nodes = Vec::with_capacity(node_count.min(r.remaining() / 3));
    for _ in 0..node_count {
        let leaf = r.u8()? != 0;
        nodes.push(if leaf {
            Node::Leaf(r.u16()?)
        } else {
            // Stored right first.
            let right = r.i16()?;
            let left = r.i16()?;
            Node::Branch { left, right }
        });
    }
    let _string_count = r.i32()?;

    // Each string: an i32 we don't need, the i32 byte length of its bits,
    // then the bits.
    let mut strings = Vec::new();
    while r.remaining() >= 8 {
        r.skip(4)?;
        let start = r.pos;
        let size = r.len("string size")?;
        let bits = r.bytes(size)?;
        let text = decode_nodes(&nodes, bits).map_err(|message| Error::Format {
            offset: start,
            message: format!("string block {}: {message}", strings.len()),
        })?;
        strings.push(text);
    }

    let mut out = BTreeMap::new();
    for (id, flags, index) in entries {
        // Flag 1 marks an entry that has text.
        if flags != 1 {
            continue;
        }
        let text = usize::try_from(index)
            .ok()
            .and_then(|i| strings.get(i))
            .ok_or_else(|| Error::Format {
                offset: 0,
                message: format!(
                    "string {id} points at block {index}, only {} exist",
                    strings.len()
                ),
            })?;
        out.insert(id, text.clone());
    }
    Ok(out)
}

/// Every `BioTlkFile` export in an LE1 package: its full path and strings.
pub fn le1_tables(package: &crate::Package) -> Result<Vec<(String, BTreeMap<i32, String>)>> {
    let mut out = Vec::new();
    for i in package.exports_of_class("BioTlkFile") {
        let (_, start) = crate::props::read(package, i)?;
        let strings = parse_le1(&package.export_data(i)[start..])?;
        out.push((package.full_path(i as i32 + 1), strings));
    }
    Ok(out)
}

fn decode_nodes(nodes: &[Node], bits: &[u8]) -> std::result::Result<String, String> {
    let root = match nodes.first() {
        Some(Node::Branch { .. }) => 0usize,
        Some(Node::Leaf(_)) => return Err("Huffman tree is a single leaf".into()),
        None => return Err("empty Huffman tree".into()),
    };
    let mut units = Vec::new();
    let mut node = root;
    let mut i = 0;
    loop {
        let b = bit(bits, i).ok_or("data ends before the string's NUL")?;
        i += 1;
        let Node::Branch { left, right } = nodes[node] else {
            unreachable!("only branches are visited")
        };
        let next = usize::try_from(if b { right } else { left })
            .ok()
            .filter(|&n| n < nodes.len())
            .ok_or("branch points outside the tree")?;
        match nodes[next] {
            Node::Leaf(0) => return Ok(String::from_utf16_lossy(&units)),
            Node::Leaf(unit) => {
                units.push(unit);
                node = root;
            }
            Node::Branch { .. } => node = next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    #[test]
    fn reads_both_tables() {
        let bytes = fixture::tlk(
            &[(100, "Keelah se'lai."), (101, "Hello, Shepard."), (102, "")],
            &[(101, "Hello, Shepard!")],
        );
        let tlk = TalkFile::parse(&bytes).unwrap();
        assert_eq!(tlk.male[&100], "Keelah se'lai.");
        assert_eq!(tlk.male[&102], "");
        assert_eq!(
            tlk.lookup(101, Gender::Female),
            Some(("Hello, Shepard!", Gender::Female))
        );
        assert_eq!(
            tlk.lookup(100, Gender::Female),
            Some(("Keelah se'lai.", Gender::Male))
        );
        assert_eq!(tlk.lookup(999, Gender::Male), None);
    }

    #[test]
    fn non_ascii_text_survives() {
        let bytes = fixture::tlk(&[(7, "Rannoch — home. ✓")], &[]);
        assert_eq!(
            TalkFile::parse(&bytes).unwrap().male[&7],
            "Rannoch — home. ✓"
        );
    }

    #[test]
    fn rejects_other_files() {
        assert!(TalkFile::parse(b"RIFF\0\0\0\0").is_err());
        let mut bytes = fixture::tlk(&[(1, "abc")], &[]);
        bytes.truncate(bytes.len() - 1);
        assert!(TalkFile::parse(&bytes).is_err());
    }

    #[test]
    fn reads_le1_blob() {
        let blob = fixture::le1_tlk(&[(5, "I'm fine."), (6, "Shepard?"), (9, "")]);
        let strings = parse_le1(&blob).unwrap();
        assert_eq!(strings[&5], "I'm fine.");
        assert_eq!(strings[&6], "Shepard?");
        assert_eq!(strings[&9], "");
    }

    #[test]
    fn finds_le1_tables_in_a_package() {
        let mut p = fixture::PackageBuilder::new(crate::Game::Le1);
        let class = p.import("Engine", "Class", "BioTlkFile", 0);
        let set = p.export(0, "GlobalTlk", 0, 0, vec![]);
        let blob = fixture::le1_tlk(&[(1, "Keelah.")]);
        let data = p.object_data(&[], &blob);
        p.export(class, "GlobalTlk_tlk", 0, set, data);
        let blob = fixture::le1_tlk(&[(1, "Keelah!")]);
        let data = p.object_data(&[], &blob);
        p.export(class, "GlobalTlk_tlk_M", 0, set, data);
        let pkg = crate::Package::parse(p.build_compressed(0x40000)).unwrap();
        let tables = le1_tables(&pkg).unwrap();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].0, "GlobalTlk.GlobalTlk_tlk");
        assert_eq!(tables[0].1[&1], "Keelah.");
        assert_eq!(tables[1].0, "GlobalTlk.GlobalTlk_tlk_M");
        assert_eq!(tables[1].1[&1], "Keelah!");
    }
}
