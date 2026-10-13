//! `BioConversation` exports: the dialogue graph.
//!
//! A conversation is two lists of nodes plus entry points:
//! - `m_EntryList`: lines NPCs speak. Each has `srText` (TLK string id),
//!   `nSpeakerIndex`, `nListenerIndex` and `ReplyListNew`, structs whose
//!   `nIndex` points into the reply list (with a paraphrase string id for
//!   the dialogue wheel).
//! - `m_ReplyList`: Shepard's lines. Each has `srText`, `ReplyType` and
//!   `EntryList`, plain indices into the entry list.
//! - `m_StartingList`: entry indices the conversation can start at.
//!
//! Speaker indices: -1 is the conversation's owner, -2 is Shepard,
//! anything else indexes the speaker list, which is `m_SpeakerList`
//! (structs with `sSpeakerTag`) in LE1/LE2 and `m_aSpeakerList` (names) in
//! LE3. LE3 also has `m_aStageDirections`: the voice director's note
//! (`sText`) for a line (`srStrRef`).
//!
//! Nodes and their structs are tagged properties all the way down; only
//! which arrays hold structs and which hold plain values is hardcoded here.
//! Checked on the real install in all three games.

use crate::package::{FName, Package};
use crate::props::{self, Tag};
use crate::reader::Reader;
use crate::{Error, Result};

pub const OWNER: i32 = -1;
pub const SHEPARD: i32 = -2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Entry,
    Reply,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Entry => "entry",
            Kind::Reply => "reply",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: Kind,
    pub index: usize,
    /// TLK string id, or -1 for a node with no line.
    pub string_id: i32,
    /// [`SHEPARD`] for replies.
    pub speaker_index: i32,
    pub listener_index: i32,
    /// Indices into the other list: replies for an entry, entries for a
    /// reply.
    pub links: Vec<usize>,
    /// For an entry, the dialogue-wheel paraphrase string id of each link.
    pub paraphrases: Vec<i32>,
    /// `ReplyType` of a reply (`REPLY_STANDARD`, `REPLY_AUTOCONTINUE`, ...).
    pub reply_type: Option<String>,
    pub non_text_line: bool,
    pub conditional_func: i32,
    pub conditional_param: i32,
    pub state_transition: i32,
    pub state_transition_param: i32,
    pub export_id: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    /// The export's own name (`nortla_debrief_d_dlg`).
    pub name: String,
    /// Speaker tags; a node's speaker index points here.
    pub speakers: Vec<String>,
    pub starting: Vec<usize>,
    pub entries: Vec<Node>,
    pub replies: Vec<Node>,
    /// `(string id, note)` from LE3 stage directions.
    pub stage_directions: Vec<(i32, String)>,
}

impl Conversation {
    /// The speaker tag behind an index, if the speaker list names it.
    pub fn speaker_tag(&self, index: i32) -> Option<&str> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.speakers.get(i))
            .map(String::as_str)
    }

    pub fn node(&self, kind: Kind, index: usize) -> Option<&Node> {
        match kind {
            Kind::Entry => self.entries.get(index),
            Kind::Reply => self.replies.get(index),
        }
    }

    /// Nodes this one leads to.
    pub fn successors(&self, node: &Node) -> Vec<(Kind, usize)> {
        let kind = match node.kind {
            Kind::Entry => Kind::Reply,
            Kind::Reply => Kind::Entry,
        };
        node.links
            .iter()
            .filter(|&&i| self.node(kind, i).is_some())
            .map(|&i| (kind, i))
            .collect()
    }

    /// Nodes that lead to `(kind, index)`.
    pub fn predecessors(&self, kind: Kind, index: usize) -> Vec<(Kind, usize)> {
        let from = match kind {
            Kind::Entry => &self.replies,
            Kind::Reply => &self.entries,
        };
        from.iter()
            .filter(|n| n.links.contains(&index))
            .map(|n| (n.kind, n.index))
            .collect()
    }

    /// The stage direction for a string id.
    pub fn stage_direction(&self, string_id: i32) -> Option<&str> {
        self.stage_directions
            .iter()
            .find(|(id, _)| *id == string_id)
            .map(|(_, t)| t.as_str())
    }
}

/// Read export `index` (0-based) as a conversation.
pub fn read(package: &Package, index: usize) -> Result<Conversation> {
    let (tags, _) = props::read(package, index)?;
    let data = package.export_data(index);
    let v = Values { package, data };
    let at = |e: Error| match e {
        Error::Format { offset, message } => Error::Format {
            offset,
            message: format!("conversation export {}: {message}", index + 1),
        },
        other => other,
    };

    let mut entries = Vec::new();
    if let Some(t) = find(&tags, "m_EntryList") {
        for (i, node) in v.structs(t).map_err(at)?.into_iter().enumerate() {
            entries.push(v.node(&node, Kind::Entry, i).map_err(at)?);
        }
    }
    let mut replies = Vec::new();
    if let Some(t) = find(&tags, "m_ReplyList") {
        for (i, node) in v.structs(t).map_err(at)?.into_iter().enumerate() {
            replies.push(v.node(&node, Kind::Reply, i).map_err(at)?);
        }
    }
    let starting = match find(&tags, "m_StartingList") {
        Some(t) => v.indices(t).map_err(at)?,
        None => Vec::new(),
    };
    let mut speakers = Vec::new();
    if let Some(t) = find(&tags, "m_SpeakerList") {
        for s in v.structs(t).map_err(at)? {
            speakers.push(match find(&s, "sSpeakerTag") {
                Some(tag) => v.name(tag).map_err(at)?,
                None => String::new(),
            });
        }
    } else if let Some(t) = find(&tags, "m_aSpeakerList") {
        speakers = v.names(t).map_err(at)?;
    }
    let mut stage_directions = Vec::new();
    if let Some(t) = find(&tags, "m_aStageDirections") {
        for s in v.structs(t).map_err(at)? {
            let id = find(&s, "srStrRef")
                .map_or(Ok(-1), |t| v.int(t))
                .map_err(at)?;
            let text = find(&s, "sText")
                .map_or(Ok(String::new()), |t| v.string(t))
                .map_err(at)?;
            stage_directions.push((id, text.trim().to_string()));
        }
    }
    Ok(Conversation {
        name: package.name(package.exports[index].name),
        speakers,
        starting,
        entries,
        replies,
        stage_directions,
    })
}

fn find<'a>(tags: &'a [Tag], name: &str) -> Option<&'a Tag> {
    tags.iter().find(|t| t.name == name)
}

/// Reads typed values out of one export's data.
struct Values<'a> {
    package: &'a Package,
    data: &'a [u8],
}

impl Values<'_> {
    fn reader(&self, t: &Tag) -> Reader<'_> {
        Reader::at(&self.data[..t.value_offset + t.value_size], t.value_offset)
    }

    fn int(&self, t: &Tag) -> Result<i32> {
        self.reader(t).i32()
    }

    fn name(&self, t: &Tag) -> Result<String> {
        let mut r = self.reader(t);
        let n = FName {
            index: r.i32()?,
            number: r.i32()?,
        };
        Ok(self.package.name(n))
    }

    /// A byte property's enum value name, if it has one.
    fn enum_value(&self, t: &Tag) -> Result<Option<String>> {
        if t.value_size == 8 {
            self.name(t).map(Some)
        } else {
            Ok(None)
        }
    }

    fn string(&self, t: &Tag) -> Result<String> {
        self.reader(t).fstring()
    }

    fn count(&self, r: &mut Reader) -> Result<usize> {
        r.len("array count")
    }

    fn ints(&self, t: &Tag) -> Result<Vec<i32>> {
        let mut r = self.reader(t);
        let n = self.count(&mut r)?;
        (0..n).map(|_| r.i32()).collect()
    }

    fn indices(&self, t: &Tag) -> Result<Vec<usize>> {
        Ok(self
            .ints(t)?
            .into_iter()
            .filter_map(|i| usize::try_from(i).ok())
            .collect())
    }

    fn names(&self, t: &Tag) -> Result<Vec<String>> {
        let mut r = self.reader(t);
        let n = self.count(&mut r)?;
        (0..n)
            .map(|_| {
                Ok(self.package.name(FName {
                    index: r.i32()?,
                    number: r.i32()?,
                }))
            })
            .collect()
    }

    /// An array of structs, each a list of tags.
    fn structs(&self, t: &Tag) -> Result<Vec<Vec<Tag>>> {
        let end = t.value_offset + t.value_size;
        let mut r = self.reader(t);
        let n = self.count(&mut r)?;
        let mut at = r.pos;
        let mut out = Vec::with_capacity(n.min(t.value_size / 8));
        for _ in 0..n {
            let (tags, next) = props::read_at(self.package, &self.data[..end], at)?;
            out.push(tags);
            at = next;
        }
        if at != end {
            return Err(Error::Format {
                offset: at,
                message: format!("{}: {} bytes left after {n} structs", t.name, end - at),
            });
        }
        Ok(out)
    }

    fn node(&self, tags: &[Tag], kind: Kind, index: usize) -> Result<Node> {
        let int = |name: &str, default: i32| -> Result<i32> {
            find(tags, name).map_or(Ok(default), |t| self.int(t))
        };
        let mut links = Vec::new();
        let mut paraphrases = Vec::new();
        match kind {
            Kind::Entry => {
                if let Some(t) = find(tags, "ReplyListNew") {
                    for r in self.structs(t)? {
                        let i = find(&r, "nIndex").map_or(Ok(-1), |t| self.int(t))?;
                        if let Ok(i) = usize::try_from(i) {
                            links.push(i);
                            paraphrases
                                .push(find(&r, "srParaphrase").map_or(Ok(-1), |t| self.int(t))?);
                        }
                    }
                }
            }
            Kind::Reply => {
                if let Some(t) = find(tags, "EntryList") {
                    links = self.indices(t)?;
                }
            }
        }
        let reply_type = match find(tags, "ReplyType") {
            Some(t) => self.enum_value(t)?,
            None => None,
        };
        Ok(Node {
            kind,
            index,
            string_id: int("srText", -1)?,
            speaker_index: match kind {
                Kind::Entry => int("nSpeakerIndex", OWNER)?,
                Kind::Reply => SHEPARD,
            },
            listener_index: int("nListenerIndex", -1)?,
            links,
            paraphrases,
            reply_type,
            non_text_line: find(tags, "bNonTextLine").and_then(|t| t.bool_value) == Some(true),
            conditional_func: int("nConditionalFunc", -1)?,
            conditional_param: int("nConditionalParam", -1)?,
            state_transition: int("nStateTransition", -1)?,
            state_transition_param: int("nStateTransitionParam", -1)?,
            export_id: int("nExportID", -1)?,
        })
    }
}

/// Every conversation in a package, by export name.
pub fn all(package: &Package) -> Vec<(String, Result<Conversation>)> {
    package
        .exports_of_class("BioConversation")
        .map(|i| (package.full_path(i as i32 + 1), read(package, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{PackageBuilder, Prop};
    use crate::Game;

    fn array(count: i32, body: Vec<u8>) -> Vec<u8> {
        let mut v = count.to_le_bytes().to_vec();
        v.extend(body);
        v
    }

    fn ints(values: &[i32]) -> Vec<u8> {
        array(
            values.len() as i32,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// Tags for one struct, ending in `None`.
    fn tags(p: &mut PackageBuilder, props: &[Prop]) -> Vec<u8> {
        // object_data writes a 4-byte net index first; drop it.
        p.object_data(props, &[])[4..].to_vec()
    }

    fn sample(game: Game) -> Vec<u8> {
        let mut p = PackageBuilder::new(game);
        let class = p.import("Engine", "Class", "BioConversation", 0);
        let reply_link = tags(
            &mut p,
            &[Prop::Int("nIndex", 0), Prop::StrRef("srParaphrase", 900)],
        );
        let entry0 = tags(
            &mut p,
            &[
                Prop::Array("ReplyListNew", array(1, reply_link)),
                Prop::Int("nSpeakerIndex", 0),
                Prop::Int("nListenerIndex", -2),
                Prop::StrRef("srText", 100),
            ],
        );
        let entry1 = tags(
            &mut p,
            &[Prop::Int("nSpeakerIndex", -1), Prop::StrRef("srText", 102)],
        );
        let reply0 = tags(
            &mut p,
            &[
                Prop::Array("EntryList", ints(&[1])),
                Prop::Byte("ReplyType", "EReplyTypes", "REPLY_STANDARD"),
                Prop::StrRef("srText", 101),
            ],
        );
        let mut entries = entry0;
        entries.extend(entry1);
        let mut props = vec![
            Prop::Array("m_StartingList", ints(&[0])),
            Prop::Array("m_EntryList", array(2, entries)),
            Prop::Array("m_ReplyList", array(1, reply0)),
        ];
        let speakers;
        let directions;
        if game == Game::Le3 {
            let mut names = 1i32.to_le_bytes().to_vec();
            let n = p.name("hench_tali");
            names.extend(n.to_le_bytes());
            names.extend(0i32.to_le_bytes());
            speakers = names;
            props.push(Prop::Array("m_aSpeakerList", speakers));
            let d = tags(
                &mut p,
                &[
                    Prop::Str("sText", "Quietly."),
                    Prop::StrRef("srStrRef", 100),
                ],
            );
            directions = array(1, d);
            props.push(Prop::Array("m_aStageDirections", directions));
        } else {
            let s = tags(&mut p, &[Prop::Name("sSpeakerTag", "hench_tali")]);
            speakers = array(1, s);
            props.push(Prop::Array("m_SpeakerList", speakers));
        }
        let data = p.object_data(&props, &[]);
        p.export(class, "tali_d_dlg", 0, 0, data);
        p.build()
    }

    #[test]
    fn reads_the_graph() {
        for game in [Game::Le1, Game::Le2, Game::Le3] {
            let pkg = Package::parse(sample(game)).unwrap();
            let c = read(&pkg, 0).unwrap();
            assert_eq!(c.name, "tali_d_dlg");
            assert_eq!(c.speakers, ["hench_tali"]);
            assert_eq!(c.starting, [0]);
            assert_eq!(c.entries.len(), 2);
            let e = &c.entries[0];
            assert_eq!(
                (e.string_id, e.speaker_index, e.listener_index),
                (100, 0, -2)
            );
            assert_eq!(
                (e.links.clone(), e.paraphrases.clone()),
                (vec![0], vec![900])
            );
            assert_eq!(c.entries[1].speaker_index, OWNER);
            let r = &c.replies[0];
            assert_eq!((r.string_id, r.speaker_index), (101, SHEPARD));
            assert_eq!(r.reply_type.as_deref(), Some("REPLY_STANDARD"));
            assert_eq!(c.successors(e), [(Kind::Reply, 0)]);
            assert_eq!(c.successors(r), [(Kind::Entry, 1)]);
            assert_eq!(c.predecessors(Kind::Entry, 1), [(Kind::Reply, 0)]);
            assert_eq!(c.speaker_tag(0), Some("hench_tali"));
            if game == Game::Le3 {
                assert_eq!(c.stage_direction(100), Some("Quietly."));
            } else {
                assert!(c.stage_directions.is_empty());
            }
        }
    }
}
