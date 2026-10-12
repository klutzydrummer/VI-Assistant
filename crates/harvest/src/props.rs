//! UE3 tagged properties at the start of an export's data.
//!
//! An object's data is a 4-byte net index, then property tags until one
//! named `None`, then the class's own binary data. Each tag is: name, type
//! name, value size, array index, a type-specific extra (struct name for
//! structs, enum name for bytes, the value itself for bools), then the
//! value. This module only walks the tags; reading values needs a schema
//! per class and comes with the conversation reader.
//!
//! **Verify** on real exports: the 4-byte prefix, 1-byte bools and enum
//! names on byte properties (all expected for engine versions 684/685).

use crate::package::{FName, Package};
use crate::reader::Reader;
use crate::{Error, Result};

/// `EObjectFlags::HasStack`: the export starts with script state, which
/// this reader doesn't handle (dialogue classes shouldn't have it).
pub const HAS_STACK: u64 = 0x0200_0000_0000_0000;

#[derive(Debug, Clone)]
pub struct Tag {
    pub name: String,
    pub kind: String,
    pub array_index: i32,
    /// Struct or enum name, when the type carries one.
    pub type_name: Option<String>,
    /// Bool properties keep their value in the tag.
    pub bool_value: Option<bool>,
    /// Where the value starts, relative to the export's data.
    pub value_offset: usize,
    pub value_size: usize,
}

/// The property tags of export `index` (0-based) and where its binary
/// data starts, relative to the export's data.
pub fn read(package: &Package, index: usize) -> Result<(Vec<Tag>, usize)> {
    if package.exports[index].object_flags & HAS_STACK != 0 {
        return Err(Error::Unsupported(format!(
            "export {} has a script stack",
            index + 1
        )));
    }
    let data = package.export_data(index);
    let mut r = Reader::new(data);
    r.skip(4)?; // net index
    let name = |r: &mut Reader| -> Result<(FName, String)> {
        let start = r.pos;
        let n = FName {
            index: r.i32()?,
            number: r.i32()?,
        };
        if usize::try_from(n.index).map_or(true, |i| i >= package.names.len()) {
            return Err(Error::Format {
                offset: start,
                message: format!("export {}: name index {} out of range", index + 1, n.index),
            });
        }
        Ok((n, package.name(n)))
    };
    let mut tags = Vec::new();
    loop {
        let (n, prop) = name(&mut r)?;
        if package.name_is(n, "None") {
            return Ok((tags, r.pos));
        }
        let (_, kind) = name(&mut r)?;
        let value_size = r.len("property size")?;
        let array_index = r.i32()?;
        let mut tag = Tag {
            name: prop,
            kind,
            array_index,
            type_name: None,
            bool_value: None,
            value_offset: 0,
            value_size,
        };
        match tag.kind.as_str() {
            "StructProperty" | "ByteProperty" => tag.type_name = Some(name(&mut r)?.1),
            "BoolProperty" => tag.bool_value = Some(r.u8()? != 0),
            _ => {}
        }
        tag.value_offset = r.pos;
        r.skip(value_size)?;
        tags.push(tag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{PackageBuilder, Prop};
    use crate::Game;

    #[test]
    fn walks_tags_to_the_binary() {
        let mut p = PackageBuilder::new(Game::Le3);
        let class = p.import("Core", "Class", "BioConversation", 0);
        let data = p.object_data(
            &[
                Prop::Int("m_nResRefID", 42),
                Prop::Bool("m_bZoomIn", true),
                Prop::Byte("m_eType", "EConvType", "CONV_Normal"),
                Prop::Struct("m_Loc", "Vector", vec![0; 12]),
                Prop::Array("m_EntryList", vec![0, 0, 0, 0]),
            ],
            b"BINARY",
        );
        p.export(class, "conv", 0, 0, data);
        let pkg = crate::Package::parse(p.build()).unwrap();
        let (tags, start) = read(&pkg, 0).unwrap();
        let names: Vec<_> = tags.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "m_nResRefID",
                "m_bZoomIn",
                "m_eType",
                "m_Loc",
                "m_EntryList"
            ]
        );
        assert_eq!(tags[1].bool_value, Some(true));
        assert_eq!(tags[2].type_name.as_deref(), Some("EConvType"));
        assert_eq!(tags[3].value_size, 12);
        let data = pkg.export_data(0);
        let v = tags[0].value_offset;
        assert_eq!(i32::from_le_bytes(data[v..v + 4].try_into().unwrap()), 42);
        assert_eq!(&data[start..], b"BINARY");
    }
}
