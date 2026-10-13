//! Writers for synthetic game files, for tests and for trying the readers
//! without the game. Everything they produce is invented: no game bytes are
//! needed or ever committed.
//!
//! The package writer can't produce real Oodle Leviathan data (there is no
//! open encoder), so "compressed" fixtures use Oodle's stored and memset
//! block forms, tagged as Leviathan. That exercises our chunk and block
//! handling and the `oozextract` call; real Leviathan streams are checked
//! on the real install.

use std::collections::{BTreeMap, BinaryHeap};

use crate::package::{COMPRESSED_FLAG, COOKED_FLAG, OODLE_LEVIATHAN, TAG};
use crate::Game;

// ---------------------------------------------------------------- TLK

/// A prefix code over UTF-16 code units, built from the strings' frequencies.
struct Huffman {
    /// Internal nodes; children are `Ok(node)` or `Err(code unit)`.
    nodes: Vec<(Child, Child)>,
    codes: BTreeMap<u16, Vec<bool>>,
}

type Child = Result<usize, u16>;

impl Huffman {
    fn build<'a>(texts: impl Iterator<Item = &'a str>) -> Huffman {
        let mut freq: BTreeMap<u16, u64> = BTreeMap::new();
        for t in texts {
            for u in t.encode_utf16().chain([0]) {
                *freq.entry(u).or_default() += 1;
            }
        }
        // A tree needs two leaves.
        if freq.len() < 2 {
            freq.entry(u16::from(b' ')).or_insert(1);
            freq.entry(0).or_insert(1);
        }
        // Min-heap on (frequency, insertion order) for a deterministic tree.
        let mut heap = BinaryHeap::new();
        let mut order = 0u64;
        for (&u, &f) in &freq {
            heap.push(std::cmp::Reverse((f, order, Err::<usize, u16>(u))));
            order += 1;
        }
        let mut built: Vec<(Child, Child)> = Vec::new();
        while heap.len() > 1 {
            let std::cmp::Reverse((fa, _, a)) = heap.pop().unwrap();
            let std::cmp::Reverse((fb, _, b)) = heap.pop().unwrap();
            built.push((a, b));
            heap.push(std::cmp::Reverse((fa + fb, order, Ok(built.len() - 1))));
            order += 1;
        }
        // Renumber so the root is node 0, breadth first.
        let root = built.len() - 1;
        let mut index = BTreeMap::new();
        let mut queue = std::collections::VecDeque::from([root]);
        let mut bfs = Vec::new();
        while let Some(n) = queue.pop_front() {
            index.insert(n, bfs.len());
            bfs.push(n);
            queue.extend([built[n].0, built[n].1].into_iter().flatten());
        }
        let remap = |c: Child| c.map(|n| index[&n]);
        let nodes: Vec<_> = bfs
            .iter()
            .map(|&n| (remap(built[n].0), remap(built[n].1)))
            .collect();
        let mut codes = BTreeMap::new();
        let mut stack = vec![(0usize, Vec::new())];
        while let Some((n, path)) = stack.pop() {
            for (bit, child) in [(false, nodes[n].0), (true, nodes[n].1)] {
                let mut p: Vec<bool> = path.clone();
                p.push(bit);
                match child {
                    Ok(c) => stack.push((c, p)),
                    Err(u) => {
                        codes.insert(u, p);
                    }
                }
            }
        }
        Huffman { nodes, codes }
    }

    fn encode(&self, text: &str, bits: &mut Vec<bool>) {
        for u in text.encode_utf16().chain([0]) {
            bits.extend(&self.codes[&u]);
        }
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// A standalone LE2/LE3 `.tlk` file with the given main and female tables.
pub fn tlk(male: &[(i32, &str)], female: &[(i32, &str)]) -> Vec<u8> {
    let all = male.iter().chain(female);
    let huffman = Huffman::build(all.clone().map(|&(_, t)| t));
    let mut bits = Vec::new();
    let mut entries = Vec::new();
    for &(id, text) in all {
        entries.push((id, bits.len() as i32));
        huffman.encode(text, &mut bits);
    }
    let data = pack_bits(&bits);

    let mut out = Vec::new();
    for v in [
        crate::tlk::MAGIC as i32,
        3,
        2,
        male.len() as i32,
        female.len() as i32,
        huffman.nodes.len() as i32,
        data.len() as i32,
    ] {
        put_i32(&mut out, v);
    }
    for (id, offset) in entries {
        put_i32(&mut out, id);
        put_i32(&mut out, offset);
    }
    let leaf = |c: Child| match c {
        Ok(n) => n as i32,
        Err(u) => -1 - i32::from(u),
    };
    for &(l, r) in &huffman.nodes {
        put_i32(&mut out, leaf(l));
        put_i32(&mut out, leaf(r));
    }
    out.extend(data);
    out
}

/// The binary part of an LE1 `BioTlkFile` export.
pub fn le1_tlk(strings: &[(i32, &str)]) -> Vec<u8> {
    let huffman = Huffman::build(strings.iter().map(|&(_, t)| t));
    // LE1 stores leaves as nodes too: internal nodes first, then leaves.
    let internal = huffman.nodes.len();
    let mut leaves: Vec<u16> = Vec::new();
    let mut id_of = |c: Child| -> i16 {
        match c {
            Ok(n) => n as i16,
            Err(u) => {
                let i = leaves.iter().position(|&l| l == u).unwrap_or_else(|| {
                    leaves.push(u);
                    leaves.len() - 1
                });
                (internal + i) as i16
            }
        }
    };
    let branches: Vec<(i16, i16)> = huffman
        .nodes
        .iter()
        .map(|&(l, r)| (id_of(l), id_of(r)))
        .collect();

    let mut out = Vec::new();
    put_i32(&mut out, strings.len() as i32);
    for (i, &(id, _)) in strings.iter().enumerate() {
        put_i32(&mut out, id);
        put_i32(&mut out, 1);
        put_i32(&mut out, i as i32);
    }
    put_i32(&mut out, (branches.len() + leaves.len()) as i32);
    for (l, r) in branches {
        out.push(0);
        out.extend(r.to_le_bytes());
        out.extend(l.to_le_bytes());
    }
    for u in leaves {
        out.push(1);
        out.extend(u.to_le_bytes());
    }
    put_i32(&mut out, strings.len() as i32);
    for &(_, text) in strings {
        let mut bits = Vec::new();
        huffman.encode(text, &mut bits);
        let bytes = pack_bits(&bits);
        put_i32(&mut out, text.encode_utf16().count() as i32 + 1);
        put_i32(&mut out, bytes.len() as i32);
        out.extend(bytes);
    }
    out
}

// ---------------------------------------------------------------- packages

/// A tagged property, for building export data.
pub enum Prop<'a> {
    Int(&'a str, i32),
    Bool(&'a str, bool),
    /// An enum value: property name, enum name, value name.
    Byte(&'a str, &'a str, &'a str),
    /// A struct with its raw serialized bytes.
    Struct(&'a str, &'a str, Vec<u8>),
    /// An array with its raw serialized bytes (count included).
    Array(&'a str, Vec<u8>),
    /// A name value.
    Name(&'a str, &'a str),
    /// A TLK string id.
    StrRef(&'a str, i32),
    /// A string, stored as UTF-16.
    Str(&'a str, &'a str),
}

struct FixtureExport {
    class: i32,
    outer: i32,
    name: (i32, i32),
    data: Vec<u8>,
}

/// A synthetic LE package, written in the layout `Package` reads.
pub struct PackageBuilder {
    pub game: Game,
    names: Vec<String>,
    imports: Vec<[i32; 7]>,
    exports: Vec<FixtureExport>,
}

impl PackageBuilder {
    pub fn new(game: Game) -> Self {
        let mut p = PackageBuilder {
            game,
            names: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
        };
        p.name("None");
        p
    }

    /// The name table index of `name`, adding it if new.
    pub fn name(&mut self, name: &str) -> i32 {
        match self.names.iter().position(|n| n == name) {
            Some(i) => i as i32,
            None => {
                self.names.push(name.to_string());
                self.names.len() as i32 - 1
            }
        }
    }

    /// Add an import; returns its object reference (negative).
    pub fn import(&mut self, package: &str, class: &str, name: &str, outer: i32) -> i32 {
        let (p, c, n) = (self.name(package), self.name(class), self.name(name));
        self.imports.push([p, 0, c, 0, outer, n, 0]);
        -(self.imports.len() as i32)
    }

    /// Add an export of class `class` (an object reference); returns its
    /// object reference (positive). `number` is the name's instance number.
    pub fn export(
        &mut self,
        class: i32,
        name: &str,
        number: i32,
        outer: i32,
        data: Vec<u8>,
    ) -> i32 {
        let name = (self.name(name), number);
        self.exports.push(FixtureExport {
            class,
            outer,
            name,
            data,
        });
        self.exports.len() as i32
    }

    /// Export data for an object: the net index, `props`, the "None" that
    /// ends them, then `binary`.
    pub fn object_data(&mut self, props: &[Prop], binary: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        put_i32(&mut out, -1);
        for p in props {
            let (name, ty, extra, value): (&str, &str, Vec<u8>, Vec<u8>) = match p {
                Prop::Int(n, v) => (n, "IntProperty", vec![], v.to_le_bytes().to_vec()),
                Prop::Bool(n, v) => (n, "BoolProperty", vec![u8::from(*v)], vec![]),
                Prop::Byte(n, e, v) => {
                    let mut extra = Vec::new();
                    self.put_name(&mut extra, e);
                    let mut value = Vec::new();
                    self.put_name(&mut value, v);
                    (n, "ByteProperty", extra, value)
                }
                Prop::Struct(n, s, raw) => {
                    let mut extra = Vec::new();
                    self.put_name(&mut extra, s);
                    (n, "StructProperty", extra, raw.clone())
                }
                Prop::Array(n, raw) => (n, "ArrayProperty", vec![], raw.clone()),
                Prop::Name(n, v) => {
                    let mut value = Vec::new();
                    self.put_name(&mut value, v);
                    (n, "NameProperty", vec![], value)
                }
                Prop::StrRef(n, v) => (n, "StringRefProperty", vec![], v.to_le_bytes().to_vec()),
                Prop::Str(n, v) => {
                    let units: Vec<u16> = v.encode_utf16().chain([0]).collect();
                    let mut value = (-(units.len() as i32)).to_le_bytes().to_vec();
                    value.extend(units.iter().flat_map(|u| u.to_le_bytes()));
                    (n, "StrProperty", vec![], value)
                }
            };
            self.put_name(&mut out, name);
            self.put_name(&mut out, ty);
            put_i32(&mut out, value.len() as i32);
            put_i32(&mut out, 0);
            out.extend(extra);
            out.extend(value);
        }
        self.put_name(&mut out, "None");
        out.extend(binary);
        out
    }

    fn put_name(&mut self, out: &mut Vec<u8>, name: &str) {
        let i = self.name(name);
        put_i32(out, i);
        put_i32(out, 0);
    }

    /// The package as stored uncompressed.
    pub fn build(&self) -> Vec<u8> {
        let (header, body) = self.layout(&[]);
        let mut out = header;
        out.extend(body);
        out
    }

    /// The package compressed the way LE stores it: one chunk for the
    /// tables and one for the export data, each cut into blocks of at most
    /// `block_size` bytes. Blocks of a single repeated byte use Oodle's
    /// memset form; the rest are stored.
    pub fn build_compressed(&self, block_size: usize) -> Vec<u8> {
        let (plain_header, body) = self.layout(&[]);
        let start = plain_header.len();
        let tables_end = self.tables_end(start);
        let ranges = [(start, tables_end), (tables_end, start + body.len())];
        let ranges: Vec<_> = ranges.into_iter().filter(|(a, b)| b > a).collect();
        let full = [plain_header.clone(), body].concat();

        // The header grows by the chunk table; lay it out once to measure.
        let placeholder: Vec<[i32; 4]> = vec![[0; 4]; ranges.len()];
        let header_len = self.header(&placeholder, start, true).len();
        let mut chunks = Vec::new();
        let mut payload = Vec::new();
        for &(a, b) in &ranges {
            let blocks: Vec<&[u8]> = full[a..b].chunks(block_size).collect();
            let mut chunk = Vec::new();
            put_i32(&mut chunk, TAG as i32);
            put_i32(&mut chunk, block_size as i32);
            let encoded: Vec<Vec<u8>> = blocks.iter().map(|b| oodle_store(b)).collect();
            put_i32(
                &mut chunk,
                encoded.iter().map(Vec::len).sum::<usize>() as i32,
            );
            put_i32(&mut chunk, (b - a) as i32);
            for (raw, enc) in blocks.iter().zip(&encoded) {
                put_i32(&mut chunk, enc.len() as i32);
                put_i32(&mut chunk, raw.len() as i32);
            }
            for enc in &encoded {
                chunk.extend(enc);
            }
            chunks.push([
                a as i32,
                (b - a) as i32,
                (header_len + payload.len()) as i32,
                chunk.len() as i32,
            ]);
            payload.extend(chunk);
        }
        let mut out = self.header(&chunks, start, true);
        assert_eq!(out.len(), header_len);
        out.extend(payload);
        out
    }

    fn tables_end(&self, start: usize) -> usize {
        let names: usize = self.names.iter().map(|n| 4 + n.len() + 1).sum();
        let exports: usize = self.exports.len() * 68;
        start + names + self.imports.len() * 28 + exports + self.exports.len() * 4
    }

    /// The uncompressed header and everything after it.
    fn layout(&self, chunks: &[[i32; 4]]) -> (Vec<u8>, Vec<u8>) {
        let header_len = self.header(chunks, 0, false).len();
        (
            self.header(chunks, header_len, false),
            self.body(header_len),
        )
    }

    fn body(&self, start: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for n in &self.names {
            put_i32(&mut out, n.len() as i32 + 1);
            out.extend(n.as_bytes());
            out.push(0);
        }
        for i in &self.imports {
            for v in i {
                put_i32(&mut out, *v);
            }
        }
        let mut data_offset = self.tables_end(start);
        for e in &self.exports {
            for v in [e.class, 0, e.outer, e.name.0, e.name.1, 0] {
                put_i32(&mut out, v);
            }
            out.extend(0u64.to_le_bytes()); // object flags
            put_i32(&mut out, e.data.len() as i32);
            put_i32(&mut out, data_offset as i32);
            put_i32(&mut out, 0); // export flags
            put_i32(&mut out, 0); // generation net object count
            out.extend([0u8; 16]); // guid
            put_i32(&mut out, 0); // package flags
            data_offset += e.data.len();
        }
        out.extend(vec![0u8; self.exports.len() * 4]); // dependency table
        for e in &self.exports {
            out.extend(&e.data);
        }
        out
    }

    fn header(&self, chunks: &[[i32; 4]], start: usize, compressed: bool) -> Vec<u8> {
        let (unreal, licensee) = match self.game {
            Game::Le1 => (684u16, 171u16),
            Game::Le2 => (684, 168),
            Game::Le3 => (685, 205),
        };
        let names: usize = self.names.iter().map(|n| 4 + n.len() + 1).sum();
        let name_offset = start;
        let import_offset = name_offset + names;
        let export_offset = import_offset + self.imports.len() * 28;
        let depends_offset = export_offset + self.exports.len() * 68;
        let full_header = depends_offset + self.exports.len() * 4;

        let mut out = Vec::new();
        out.extend(TAG.to_le_bytes());
        out.extend(unreal.to_le_bytes());
        out.extend(licensee.to_le_bytes());
        put_i32(&mut out, full_header as i32);
        if self.game == Game::Le3 {
            put_i32(&mut out, -5);
            for u in "None\0".encode_utf16() {
                out.extend(u.to_le_bytes());
            }
        } else {
            put_i32(&mut out, 5);
            out.extend(b"None\0");
        }
        let flags = COOKED_FLAG | if compressed { COMPRESSED_FLAG } else { 0 };
        out.extend(flags.to_le_bytes());
        if self.game == Game::Le3 {
            put_i32(&mut out, 0);
        }
        for v in [
            self.names.len(),
            name_offset,
            self.exports.len(),
            export_offset,
            self.imports.len(),
            import_offset,
            depends_offset,
            full_header,
            0,
            0,
            0,
        ] {
            put_i32(&mut out, v as i32);
        }
        out.extend([0x42u8; 16]); // guid
        put_i32(&mut out, 1);
        put_i32(&mut out, self.exports.len() as i32);
        put_i32(&mut out, self.names.len() as i32);
        put_i32(&mut out, 0);
        put_i32(&mut out, 6383);
        put_i32(
            &mut out,
            if self.game == Game::Le3 {
                196715
            } else {
                65643
            },
        );
        put_i32(&mut out, 0); // build
        put_i32(&mut out, 0); // branch
        if compressed {
            put_i32(&mut out, OODLE_LEVIATHAN as i32);
        } else {
            put_i32(&mut out, 0);
        }
        put_i32(&mut out, chunks.len() as i32);
        for c in chunks {
            for v in c {
                put_i32(&mut out, *v);
            }
        }
        put_i32(&mut out, 0); // package source
        put_i32(&mut out, 0); // additional packages to cook
        out
    }
}

/// An Oodle stream holding `raw` without compression (or as a memset when
/// every byte is the same), tagged as Leviathan.
fn oodle_store(raw: &[u8]) -> Vec<u8> {
    const LEVIATHAN: u8 = 0x0C;
    let mut out = Vec::new();
    if raw.len() <= 0x40000 && !raw.is_empty() && raw.iter().all(|&b| b == raw[0]) {
        // Header: restart, compressed. Quantum: special (0x3FFFF) of kind 1.
        out.extend([0x8C, LEVIATHAN, 0x07, 0xFF, 0xFF, raw[0]]);
        return out;
    }
    for (i, piece) in raw.chunks(0x40000).enumerate() {
        // Header: uncompressed, plus restart on the first block.
        out.push(if i == 0 { 0xCC } else { 0x4C });
        out.push(LEVIATHAN);
        out.extend(piece);
    }
    out
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend(v.to_le_bytes());
}
