//! `vi harvest ...`: look inside the game files while the harvester is
//! being built (milestones 1 to 3 of `docs/design/harvester-plan.md`).
//! Everything here only reads the install.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use harvest::package::{Header, Package};
use harvest::tlk::{self, TalkFile};

/// Package headers are a few KB; this covers the chunk table of any LE
/// package with room to spare.
const HEADER_PREFIX: u64 = 1 << 20;

fn fail(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {e}");
    ExitCode::FAILURE
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn read_header(path: &Path) -> Result<Header, String> {
    let mut prefix = Vec::new();
    std::fs::File::open(path)
        .and_then(|f| f.take(HEADER_PREFIX).read_to_end(&mut prefix))
        .map_err(|e| e.to_string())?;
    Header::parse(&prefix).map_err(|e| e.to_string())
}

/// Milestone 1 (and 2 with `open`): list each game's files and check that
/// every package header parses; `open` also decompresses every package and
/// reads its tables.
pub fn scan(root: &Path, open: bool) -> ExitCode {
    let games = match harvest::install::scan(root) {
        Ok(g) => g,
        Err(e) => return fail(format!("{}: {e}", root.display())),
    };
    if games.is_empty() {
        return fail(format!(
            "no games under {} (expected Game/ME1, Game/ME2 or Game/ME3 with a BioGame folder)",
            root.display()
        ));
    }
    let mut failures = 0;
    for g in &games {
        let f = &g.files;
        let size = |list: &[harvest::install::FileInfo]| mib(list.iter().map(|x| x.size).sum());
        println!("{} at {}", g.game, g.biogame.display());
        println!("  packages {:>6}  {}", f.packages.len(), size(&f.packages));
        println!("  tlk      {:>6}  {}", f.tlks.len(), size(&f.tlks));
        println!("  afc      {:>6}  {}", f.afcs.len(), size(&f.afcs));

        let mut versions: BTreeMap<String, usize> = BTreeMap::new();
        let mut errors = Vec::new();
        for p in &f.packages {
            match read_header(&p.path) {
                Ok(h) => {
                    let mut key = format!(
                        "{} {}/{} {:?}",
                        h.game, h.unreal_version, h.licensee_version, h.compression
                    );
                    if h.game != g.game {
                        key.push_str(" (folder says ");
                        key.push_str(g.game.id());
                        key.push(')');
                    }
                    *versions.entry(key).or_default() += 1;
                    if open {
                        if let Err(e) = Package::read(&p.path) {
                            errors.push(e);
                        }
                    }
                }
                Err(e) => errors.push(format!("{}: {e}", p.path.display())),
            }
        }
        for (key, n) in &versions {
            println!("  headers  {n:>6}  {key}");
        }
        for tlk in &f.tlks {
            if let Err(e) = std::fs::read(&tlk.path)
                .map_err(|e| e.to_string())
                .and_then(|b| TalkFile::parse(&b).map_err(|e| e.to_string()))
            {
                errors.push(format!("{}: {e}", tlk.path.display()));
            }
        }
        for e in errors.iter().take(20) {
            println!("  error: {e}");
        }
        if errors.len() > 20 {
            println!("  ... {} more errors", errors.len() - 20);
        }
        failures += errors.len();
    }
    if failures == 0 {
        println!("all files read");
        ExitCode::SUCCESS
    } else {
        println!("{failures} files failed");
        ExitCode::FAILURE
    }
}

/// Milestone 2: one package's header, tables and class counts.
pub fn package(path: &Path) -> ExitCode {
    let pkg = match Package::read(path) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let h = &pkg.header;
    println!(
        "{} version {}/{}, flags {:#010x}, compression {:?} in {} chunks",
        h.game,
        h.unreal_version,
        h.licensee_version,
        h.flags,
        h.compression,
        h.chunks.len()
    );
    println!(
        "{} names, {} imports, {} exports, {} decompressed",
        pkg.names.len(),
        pkg.imports.len(),
        pkg.exports.len(),
        mib(pkg.data.len() as u64)
    );
    let mut counts: Vec<_> = pkg.class_counts().into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (class, n) in counts {
        println!("{n:>7}  {class}");
    }
    ExitCode::SUCCESS
}

/// Milestone 3: dump a subtitle table as `id <tab> variant <tab> text`,
/// from an LE2/LE3 `.tlk` file or the `BioTlkFile` exports of an LE1
/// package. `id` limits the output to one string.
pub fn tlk(path: &Path, id: Option<&str>) -> ExitCode {
    let id = match id.map(str::parse::<i32>) {
        Some(Ok(id)) => Some(id),
        Some(Err(_)) => return fail("--id needs a number"),
        None => None,
    };
    let wanted = |i: i32| id.is_none_or(|w| w == i);
    let one_line = |t: &str| t.replace('\n', "\\n");
    let is_package = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pcc"));
    let mut found = 0;
    if is_package {
        let pkg = match Package::read(path) {
            Ok(p) => p,
            Err(e) => return fail(e),
        };
        let tables = match tlk::le1_tables(&pkg) {
            Ok(t) => t,
            Err(e) => return fail(format!("{}: {e}", path.display())),
        };
        if tables.is_empty() {
            return fail(format!("{}: no BioTlkFile exports", path.display()));
        }
        for (table, strings) in tables {
            for (i, text) in strings.iter().filter(|(i, _)| wanted(**i)) {
                println!("{i}\t{table}\t{}", one_line(text));
                found += 1;
            }
        }
    } else {
        let parsed = std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|b| TalkFile::parse(&b).map_err(|e| e.to_string()));
        let tlk = match parsed {
            Ok(t) => t,
            Err(e) => return fail(format!("{}: {e}", path.display())),
        };
        for (variant, table) in [("m", &tlk.male), ("f", &tlk.female)] {
            for (i, text) in table.iter().filter(|(i, _)| wanted(**i)) {
                println!("{i}\t{variant}\t{}", one_line(text));
                found += 1;
            }
        }
    }
    if found == 0 && id.is_some() {
        return fail("no such string id");
    }
    ExitCode::SUCCESS
}

/// `vi extract`: the harvest stage. Writes into `out`, or `$VI_OUT` when
/// run by the pipeline.
pub fn extract(character: &Path, root: &Path, out: Option<&str>) -> ExitCode {
    let Some(out) = out
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("VI_OUT").map(std::path::PathBuf::from))
    else {
        return fail("no output folder: pass --out or set VI_OUT");
    };
    let who = match load_character(character) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    let reports = match harvest::extract::run(root, &who, &out, &|m| eprintln!("{m}")) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let mut errors = 0;
    for (game, r) in &reports {
        println!(
            "{game}: {} lines from {} streams in {} packages ({} duplicates merged, {} without subtitle, {} missing .afc, {} placeholders skipped, {} conversations, {} lines without a node, {} errors)",
            r.lines, r.streams, r.packages, r.merged_duplicates, r.no_subtitle, r.missing_afc, r.placeholders.len(), r.conversations, r.without_node, r.errors.len()
        );
        for (code, n) in &r.unknown_levels {
            println!("  no scene entry for level {code} ({n} lines)");
        }
        for e in r.errors.iter().take(10) {
            println!("  error: {e}");
        }
        errors += r.errors.len();
    }
    if errors == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn load_character(character: &Path) -> Result<harvest::extract::Character, String> {
    let sources = character.join("sources.toml");
    let text =
        std::fs::read_to_string(&sources).map_err(|e| format!("{}: {e}", sources.display()))?;
    let id = character
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_string))
        .unwrap_or_default();
    harvest::extract::Character::from_sources(&id, &text)
}

/// `vi corpus tag`: recompute `scene_tags` in an existing harvest folder
/// from the records' level, DLC and conversation, without touching audio.
pub fn retag(character: &Path, dir: &Path) -> ExitCode {
    let who = match load_character(character) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    match harvest::extract::retag(dir, &who) {
        Ok((changed, unknown)) => {
            println!("{changed} lines got new scene tags");
            for (code, n) in &unknown {
                println!("  no scene entry for level {code} ({n} lines)");
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}
