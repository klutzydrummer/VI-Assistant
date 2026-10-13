//! LE2/LE3 dialogue audio: `WwiseStream` exports and the `.afc` files
//! they point into.
//!
//! Each spoken line has a `WwiseStream` export whose name carries the
//! speaker tag, the conversation, the TLK string id and Shepard's gender
//! for that recording. The two games spell it differently (checked on the
//! real install):
//!
//! - LE2: `en_us_hench_tali_nortla_debrief_d_00223055_m_wav`, with the
//!   conversation also in the `BankName` property, which is how the tag
//!   and the conversation are told apart.
//! - LE3: `en-us,hench_tali,citprs_tali_talk1_d,00604744_m_wav`.
//!
//! The export's `Filename` property names the `.afc` file (without the
//! extension) and its binary data ends with a 16-byte bulk data header:
//! flags, element count, size on disk, offset. LE2 has 8 more bytes in
//! front of it, LE3 doesn't. The bytes at that offset are a complete RIFF
//! `.wem` (Wwise Vorbis, codec `0xFFFF`), 44.1 kHz mono on the files
//! checked so far.

use crate::package::{FName, Package};
use crate::reader::Reader;
use crate::tlk::Gender;
use crate::{Error, Result};

/// Bulk data flag: the payload lives in a separate file (the `.afc`).
pub const STORE_IN_SEPARATE_FILE: u32 = 0x1;

/// What a stream's name says about the line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamName {
    pub speaker_tag: String,
    pub conversation: String,
    pub string_id: i32,
    pub gender: Gender,
}

/// Parse a dialogue stream name. `bank` is the stream's `BankName`, which
/// LE2 names need to split the speaker tag from the conversation; LE3
/// names don't use it.
pub fn parse_name(name: &str, bank: Option<&str>) -> Option<StreamName> {
    let rest = name.strip_suffix("_wav")?;
    let (rest, gender) = if let Some(r) = rest.strip_suffix("_m") {
        (r, Gender::Male)
    } else {
        (rest.strip_suffix("_f")?, Gender::Female)
    };
    if let Some(rest) = rest.strip_prefix("en-us,") {
        let mut parts = rest.split(',');
        let (tag, conversation, id) = (parts.next()?, parts.next()?, parts.next()?);
        if parts.next().is_some() || tag.is_empty() || conversation.is_empty() {
            return None;
        }
        return Some(StreamName {
            speaker_tag: tag.to_string(),
            conversation: conversation.to_string(),
            string_id: parse_id(id)?,
            gender,
        });
    }
    let rest = rest.strip_prefix("en_us_")?;
    let (rest, id) = rest.rsplit_once('_')?;
    let string_id = parse_id(id)?;
    let bank = bank.filter(|b| !b.is_empty())?;
    let tag = rest.strip_suffix(bank)?.strip_suffix('_')?;
    if tag.is_empty() {
        return None;
    }
    Some(StreamName {
        speaker_tag: tag.to_string(),
        conversation: bank.to_string(),
        string_id,
        gender,
    })
}

fn parse_id(s: &str) -> Option<i32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Where a stream's audio is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stream {
    /// The export's own name.
    pub name: String,
    /// The `.afc` file's name without the extension.
    pub filename: String,
    pub bank: Option<String>,
    pub offset: usize,
    pub size: usize,
}

/// Read export `index` (0-based) as a `WwiseStream`. Streams whose audio
/// isn't in an `.afc` (no `Filename`, or a payload stored inline) are
/// `None`.
pub fn read_stream(package: &Package, index: usize) -> Result<Option<Stream>> {
    let (tags, start) = crate::props::read(package, index)?;
    let data = package.export_data(index);
    let name_value = |prop: &str| -> Result<Option<String>> {
        let Some(tag) = tags.iter().find(|t| t.name == prop) else {
            return Ok(None);
        };
        let mut r = Reader::at(data, tag.value_offset);
        let name = FName {
            index: r.i32()?,
            number: r.i32()?,
        };
        Ok(Some(package.name(name)))
    };
    let Some(filename) = name_value("Filename")? else {
        return Ok(None);
    };
    let bank = name_value("BankName")?;
    let binary = &data[start..];
    if binary.len() < 16 {
        return Err(Error::Format {
            offset: start,
            message: format!(
                "export {}: WwiseStream binary is {} bytes, expected at least 16",
                index + 1,
                binary.len()
            ),
        });
    }
    let mut r = Reader::new(&binary[binary.len() - 16..]);
    let flags = r.u32()?;
    let count = r.len("element count")?;
    let size = r.len("size on disk")?;
    let offset = r.len("offset")?;
    if flags & STORE_IN_SEPARATE_FILE == 0 {
        return Ok(None);
    }
    if count != size {
        return Err(Error::Format {
            offset: start,
            message: format!(
                "export {}: compressed stream ({count} elements, {size} bytes on disk)",
                index + 1
            ),
        });
    }
    Ok(Some(Stream {
        name: package.name(package.exports[index].name),
        filename,
        bank,
        offset,
        size,
    }))
}

/// The sample rate in a RIFF `.wem` header, if it is one.
pub fn wem_sample_rate(wem: &[u8]) -> Option<u32> {
    if wem.len() < 28 || &wem[0..4] != b"RIFF" || &wem[8..16] != b"WAVEfmt " {
        return None;
    }
    Some(u32::from_le_bytes(wem[24..28].try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{PackageBuilder, Prop};
    use crate::Game;

    #[test]
    fn parses_le2_names() {
        let n = parse_name(
            "en_us_hench_tali_nortla_debrief_d_00223055_m_wav",
            Some("nortla_debrief_d"),
        )
        .unwrap();
        assert_eq!(n.speaker_tag, "hench_tali");
        assert_eq!(n.conversation, "nortla_debrief_d");
        assert_eq!(n.string_id, 223055);
        assert_eq!(n.gender, Gender::Male);
        // Without the bank the tag can't be found.
        assert_eq!(
            parse_name("en_us_hench_tali_x_d_00000001_f_wav", None),
            None
        );
        assert_eq!(
            parse_name("en_us_hench_tali_x_d_0000001a_f_wav", Some("x_d")),
            None
        );
    }

    #[test]
    fn parses_le3_names() {
        let n = parse_name("en-us,hench_tali,citprs_tali_talk1_d,00604744_f_wav", None).unwrap();
        assert_eq!(n.speaker_tag, "hench_tali");
        assert_eq!(n.conversation, "citprs_tali_talk1_d");
        assert_eq!(n.string_id, 604744);
        assert_eq!(n.gender, Gender::Female);
        assert_eq!(parse_name("en-us,hench_tali,00604744_f_wav", None), None);
        assert_eq!(parse_name("en-us,a,b,1_x_wav", None), None);
    }

    #[test]
    fn reads_stream_location() {
        for (game, prefix) in [
            (Game::Le2, vec![1u8, 0, 0, 0, 1, 0, 0, 0]),
            (Game::Le3, vec![]),
        ] {
            let mut p = PackageBuilder::new(game);
            let class = p.import("Engine", "Class", "WwiseStream", 0);
            let mut binary = prefix;
            for v in [1u32, 300, 300, 4096] {
                binary.extend(v.to_le_bytes());
            }
            let data = p.object_data(
                &[
                    Prop::Name("Filename", "conv_d_S_Int"),
                    Prop::Name("BankName", "conv_d"),
                ],
                &binary,
            );
            p.export(class, "en_us_hench_tali_conv_d_00000007_m_wav", 0, 0, data);
            let pkg = Package::parse(p.build()).unwrap();
            let s = read_stream(&pkg, 0).unwrap().unwrap();
            assert_eq!(s.filename, "conv_d_S_Int");
            assert_eq!(s.bank.as_deref(), Some("conv_d"));
            assert_eq!((s.offset, s.size), (4096, 300));
        }
    }

    #[test]
    fn reads_wem_rate() {
        let mut wem = b"RIFF\0\0\0\0WAVEfmt \0\0\0\0".to_vec();
        wem.extend([0xFF, 0xFF, 1, 0]);
        wem.extend(44100u32.to_le_bytes());
        assert_eq!(wem_sample_rate(&wem), Some(44100));
        assert_eq!(wem_sample_rate(b"OggS"), None);
    }
}
