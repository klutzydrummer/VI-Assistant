//! Scene tags: where a line is spoken and in what situation, from the
//! level code, DLC folder and conversation name every record carries.
//!
//! The mapping is data, in `scenes.toml` beside this file, so it can be
//! reviewed and extended without touching code. Character-specific tags
//! (whose own story a scene is) come from the pack's `sources.toml`, as
//! [`Rule`]s.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::Game;

/// The game-wide mapping.
#[derive(Debug, Default)]
pub struct Table {
    /// Lowercase level code -> tags, per game.
    levels: BTreeMap<Game, BTreeMap<String, Vec<String>>>,
    /// Lowercase DLC folder -> tags, per game.
    dlc: BTreeMap<Game, BTreeMap<String, Vec<String>>>,
    /// Lowercase name part -> tags.
    keywords: BTreeMap<String, Vec<String>>,
    /// Lowercase glob -> tags.
    patterns: Vec<(String, Vec<String>)>,
    /// Lowercase globs of conversations heard anywhere: no level, DLC or
    /// character tags.
    global: Vec<String>,
}

/// A character's own tag: `tag` applies to lines matching any of
/// `matches`, each `level:<code>` or `conversation:<glob>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub tag: String,
    pub matches: Vec<String>,
}

/// What a line's scene is made of.
#[derive(Debug, Clone, Copy)]
pub struct Scene<'a> {
    pub game: Game,
    /// `scene.prefix`, like `BioD_QuaTlL` or `BIOA_NOR10`.
    pub prefix: &'a str,
    /// `scene.dlc`, the DLC folder.
    pub dlc: Option<&'a str>,
    pub conversation: &'a str,
}

impl Table {
    /// The table shipped with the harvester.
    pub fn builtin() -> &'static Table {
        static TABLE: OnceLock<Table> = OnceLock::new();
        TABLE.get_or_init(|| {
            Table::parse(include_str!("scenes.toml")).expect("scenes.toml is valid")
        })
    }

    pub fn parse(text: &str) -> Result<Table, String> {
        let root: toml::Table = text.parse().map_err(|e| format!("scenes.toml: {e}"))?;
        let mut table = Table::default();
        let tags_of = |v: &toml::Value, what: &str| -> Result<Vec<String>, String> {
            let list = v
                .as_array()
                .ok_or_else(|| format!("scenes.toml: {what} must be a list of tags"))?;
            list.iter()
                .map(|t| {
                    let t = t
                        .as_str()
                        .ok_or_else(|| format!("scenes.toml: {what} has a tag that isn't text"))?;
                    if valid_tag(t) {
                        Ok(t.to_string())
                    } else {
                        Err(format!(
                            "scenes.toml: {what}: tag {t:?} must be lowercase a-z, 0-9 and _"
                        ))
                    }
                })
                .collect()
        };
        let section = |v: Option<&toml::Value>,
                       what: &str|
         -> Result<BTreeMap<String, Vec<String>>, String> {
            let mut out = BTreeMap::new();
            if let Some(v) = v {
                let t = v
                    .as_table()
                    .ok_or_else(|| format!("scenes.toml: [{what}] must be a table"))?;
                for (k, v) in t {
                    out.insert(k.to_ascii_lowercase(), tags_of(v, &format!("{what}.{k}"))?);
                }
            }
            Ok(out)
        };
        for game in [Game::Le1, Game::Le2, Game::Le3] {
            let g = root.get(game.id());
            table.levels.insert(
                game,
                section(
                    g.and_then(|g| g.get("levels")),
                    &format!("{}.levels", game.id()),
                )?,
            );
            table.dlc.insert(
                game,
                section(g.and_then(|g| g.get("dlc")), &format!("{}.dlc", game.id()))?,
            );
        }
        table.keywords = section(root.get("conversation_keywords"), "conversation_keywords")?;
        table.patterns = section(root.get("conversation_patterns"), "conversation_patterns")?
            .into_iter()
            .collect();
        if let Some(g) = root.get("global_conversations") {
            let list = g
                .get("patterns")
                .and_then(|p| p.as_array())
                .ok_or("scenes.toml: [global_conversations] needs a patterns list")?;
            for p in list {
                let p = p
                    .as_str()
                    .ok_or("scenes.toml: global_conversations patterns must be text")?;
                table.global.push(p.to_ascii_lowercase());
            }
        }
        Ok(table)
    }

    /// The tags for one line, sorted, plus whether its level code was
    /// known.
    pub fn tags(&self, scene: Scene, rules: &[Rule]) -> (Vec<String>, bool) {
        let mut out = BTreeSet::new();
        let conversation = scene.conversation.to_ascii_lowercase();
        let global = self.global.iter().any(|g| glob(g, &conversation));
        self.situation(&conversation, &mut out);
        if global {
            return (out.into_iter().collect(), true);
        }
        let code = level_code(scene.prefix);
        let levels = &self.levels[&scene.game];
        let known = match levels.get(&code).or_else(|| levels.get(&area(&code))) {
            Some(tags) => {
                out.extend(tags.iter().cloned());
                true
            }
            None => false,
        };
        if let Some(dlc) = scene.dlc {
            match self.dlc[&scene.game].get(&dlc.to_ascii_lowercase()) {
                Some(tags) => out.extend(tags.iter().cloned()),
                None => {
                    out.insert("dlc".to_string());
                }
            }
        }
        for rule in rules {
            if rule
                .matches
                .iter()
                .any(|m| matches(m, &code, &conversation))
            {
                out.insert(rule.tag.clone());
            }
        }
        (out.into_iter().collect(), known)
    }

    /// Keyword and pattern tags from a lowercase conversation name.
    fn situation(&self, conversation: &str, out: &mut BTreeSet<String>) {
        for part in name_parts(conversation) {
            if let Some(tags) = self.keywords.get(part) {
                out.extend(tags.iter().cloned());
            }
        }
        for (pattern, tags) in &self.patterns {
            if glob(pattern, conversation) {
                out.extend(tags.iter().cloned());
            }
        }
    }
}

/// Whether `t` is a valid tag: lowercase letters, digits and `_`.
pub fn valid_tag(t: &str) -> bool {
    !t.is_empty()
        && t.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// `BioD_QuaTlL` -> `quatll`, `BIOA_NOR10` -> `nor10`, `QuaTlL` -> `quatll`.
fn level_code(prefix: &str) -> String {
    prefix
        .split_once('_')
        .map_or(prefix, |(_, code)| code)
        .to_ascii_lowercase()
}

/// LE1 codes are an area and a number: `nor10` -> `nor`, `war40v` -> `war`.
fn area(code: &str) -> String {
    code.chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect()
}

/// The parts of a conversation name between `_` and digits:
/// `citapt_001mr_tali_a` -> citapt, mr, tali, a.
fn name_parts(name: &str) -> impl Iterator<Item = &str> {
    name.split(|c: char| c == '_' || c.is_ascii_digit())
        .filter(|p| !p.is_empty())
}

/// `*` matches any run of characters; everything else literally.
fn glob(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !text.starts_with(first) || !text[first.len()..].ends_with(last) {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for middle in &parts[1..parts.len() - 1] {
        match rest.find(middle) {
            Some(i) => rest = &rest[i + middle.len()..],
            None => return false,
        }
    }
    true
}

fn matches(rule: &str, code: &str, conversation: &str) -> bool {
    if let Some(level) = rule.strip_prefix("level:") {
        level.eq_ignore_ascii_case(code)
    } else if let Some(pattern) = rule.strip_prefix("conversation:") {
        glob(&pattern.to_ascii_lowercase(), conversation)
    } else {
        false
    }
}

/// Parse a character's rules from a `scene_tags` table in sources.toml:
/// `tag = ["level:QuaTlL", "conversation:nortl_*"]`.
pub fn parse_rules(table: &toml::Table, what: &str) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    for (tag, v) in table {
        if !valid_tag(tag) {
            return Err(format!(
                "{what}: tag {tag:?} must be lowercase a-z, 0-9 and _"
            ));
        }
        let list = v
            .as_array()
            .ok_or_else(|| format!("{what}.{tag} must be a list"))?;
        let mut matches = Vec::new();
        for m in list {
            let m = m
                .as_str()
                .ok_or_else(|| format!("{what}.{tag}: entries must be text"))?;
            if !(m.starts_with("level:") || m.starts_with("conversation:")) {
                return Err(format!(
                    "{what}.{tag}: {m:?} must start with level: or conversation:"
                ));
            }
            matches.push(m.to_string());
        }
        rules.push(Rule {
            tag: tag.clone(),
            matches,
        });
    }
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(game: Game, prefix: &str, dlc: Option<&str>, conversation: &str) -> Vec<String> {
        let scene = Scene {
            game,
            prefix,
            dlc,
            conversation,
        };
        Table::builtin().tags(scene, &[]).0
    }

    #[test]
    fn builtin_table_parses() {
        let t = Table::builtin();
        assert!(t.levels[&Game::Le2].contains_key("quatll"));
        assert!(t.keywords.contains_key("banter"));
    }

    #[test]
    fn levels_by_code_and_le1_area() {
        assert_eq!(
            tags(Game::Le2, "BioD_QuaTlL", None, "quatll_commando_d"),
            ["loyalty", "migrant_fleet"]
        );
        assert_eq!(
            tags(Game::Le1, "BIOA_NOR10", None, "cha00_quarian_tali"),
            ["normandy"]
        );
        // Area entry for codes without their own.
        assert_eq!(
            tags(Game::Le1, "BIOA_WAR40V", None, "war40_radiochatter"),
            ["feros", "main_mission"]
        );
        // A full code beats its area.
        assert_eq!(
            tags(Game::Le1, "BIOA_UNC17", None, "sp114_triggers"),
            [
                "antibaar",
                "side_mission",
                "squad_comment",
                "uncharted_world"
            ]
        );
        // Package type and case don't matter.
        assert_eq!(
            tags(Game::Le3, "BioA_Gth001", None, "x"),
            tags(Game::Le3, "Biod_GTH001", None, "x")
        );
        // CitSam isn't the Citadel.
        assert_eq!(
            tags(Game::Le3, "BioD_CitSam", None, "citsam_falere_m"),
            ["ardat_yakshi_monastery", "side_mission"]
        );
    }

    #[test]
    fn dlc_and_situation() {
        assert_eq!(
            tags(
                Game::Le3,
                "BioD_CitApt",
                Some("DLC_EXP_Pack003"),
                "citpar_bb3_a"
            ),
            ["apartment", "citadel", "citadel_dlc", "citadel_party"]
        );
        assert_eq!(
            tags(
                Game::Le3,
                "BioD_Nor",
                Some("DLC_HEN_PR"),
                "norhen_proth_mr_cermir_a"
            ),
            ["from_ashes_dlc", "mission_reaction", "normandy"]
        );
        // A DLC folder the table doesn't know still says it's DLC.
        assert!(tags(Game::Le2, "BioD_Zzz", Some("DLC_New"), "x").contains(&"dlc".to_string()));
        assert_eq!(
            tags(Game::Le2, "BioD_Nor", None, "nortl_culmination_h"),
            ["normandy", "romance"]
        );
        // Barks play everywhere: the package they sit in isn't a place.
        assert_eq!(
            tags(Game::Le2, "BioD_ProFre", None, "ss_global_hench_tali"),
            ["combat"]
        );
    }

    #[test]
    fn patterns_not_substrings() {
        assert!(tags(Game::Le1, "BIOA_NOR10", None, "nor20_feros_done").contains(&"debrief".into()));
        assert!(
            !tags(Game::Le1, "BIOA_LOS30", None, "los30_security_system_done")
                .contains(&"debrief".into())
        );
        assert!(
            tags(Game::Le3, "BioA_Cat002", None, "cat002_shuttle_ride_m")
                .contains(&"shuttle".into())
        );
        assert!(
            !tags(Game::Le2, "BioD_QuaTlL", None, "quatll_shuttle_guard_d")
                .contains(&"shuttle".into())
        );
        // `rom` is a whole part, not inside another word.
        assert!(!tags(Game::Le2, "BioD_Nor", None, "nor_promenade_d").contains(&"romance".into()));
    }

    #[test]
    fn unknown_levels_are_reported() {
        let scene = Scene {
            game: Game::Le1,
            prefix: "BIOA_ZZZ00",
            dlc: None,
            conversation: "zzz00_banter_00",
        };
        let (t, known) = Table::builtin().tags(scene, &[]);
        assert!(!known);
        assert_eq!(t, ["squad_banter"]);
        // Global conversations aren't placed, so they don't count as unknown.
        let scene = Scene {
            conversation: "combat_banter_00",
            ..scene
        };
        assert_eq!(
            Table::builtin().tags(scene, &[]),
            (vec!["combat".into(), "squad_banter".into()], true)
        );
    }

    #[test]
    fn character_rules() {
        let rules: toml::Table = "personal_arc = [\"level:QuaTlL\", \"conversation:nortl_*\"]"
            .parse()
            .unwrap();
        let rules = parse_rules(&rules, "le2.scene_tags").unwrap();
        let t = |prefix, conv| {
            Table::builtin()
                .tags(
                    Scene {
                        game: Game::Le2,
                        prefix,
                        dlc: None,
                        conversation: conv,
                    },
                    &rules,
                )
                .0
        };
        assert!(t("BioD_QuaTlL", "quatll_boarding_d").contains(&"personal_arc".into()));
        assert!(t("BioD_Nor", "nortl_relationship_00_h").contains(&"personal_arc".into()));
        assert!(!t("BioD_Nor", "norgr_x").contains(&"personal_arc".into()));
        let bad: toml::Table = "x = [\"quatll\"]".parse().unwrap();
        assert!(parse_rules(&bad, "le2.scene_tags").is_err());
    }

    #[test]
    fn globs() {
        assert!(glob("nor20_*_done", "nor20_feros_done"));
        assert!(!glob("nor20_*_done", "nor20_feros_done_x"));
        assert!(glob("*_shuttle_m", "cit004_shuttle_m"));
        assert!(glob("a*b*c", "aXbYc"));
        assert!(!glob("a*b*c", "aXcYb"));
        assert!(glob("exact", "exact"));
    }
}
