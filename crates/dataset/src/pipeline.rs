//! Pipeline files: which stages exist, what each runs and depends on.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Error, IoContext};

/// Prefix for `files` entries relative to the corpus folder.
pub const CORPUS_PREFIX: &str = "corpus:";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineFile {
    corpus: PathBuf,
    #[serde(default)]
    stage: BTreeMap<String, Stage>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub command: Vec<String>,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub settings: toml::Table,
    #[serde(default = "one")]
    pub version: u32,
    /// Runs only when named (`vi dataset run <pipeline> <stage>`), never as
    /// part of a whole-pipeline run: for experiments that cost GPU time.
    #[serde(default)]
    pub manual: bool,
}

fn one() -> u32 {
    1
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Folder holding the pipeline file; relative paths resolve against it.
    pub dir: PathBuf,
    pub corpus: PathBuf,
    pub stages: BTreeMap<String, Stage>,
}

impl Pipeline {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(path).at(path)?;
        let file: PipelineFile = toml::from_str(&text).map_err(|e| Error::Invalid {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let pipeline = Pipeline {
            corpus: dir.join(&file.corpus),
            dir,
            stages: file.stage,
        };
        let problems = pipeline.problems();
        if !problems.is_empty() {
            return Err(Error::Invalid {
                path: path.to_path_buf(),
                message: problems.join("; "),
            });
        }
        Ok(pipeline)
    }

    fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, stage) in &self.stages {
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b))
            {
                out.push(format!(
                    "stage name {name:?} must be lowercase letters, digits, _ or -"
                ));
            }
            if stage.command.is_empty() {
                out.push(format!("stage {name} has an empty command"));
            }
            for input in &stage.inputs {
                match self.stages.get(input) {
                    None => out.push(format!("stage {name} needs unknown stage {input}")),
                    Some(i) if i.manual && !stage.manual => out.push(format!(
                        "stage {name} needs manual stage {input}, so it must be manual too"
                    )),
                    Some(_) => {}
                }
            }
        }
        if out.is_empty() {
            if let Err(cycle) = self.order(None) {
                out.push(cycle);
            }
        }
        out
    }

    /// Stages in run order, dependencies first. With a target, only that
    /// stage and what it needs; without one, every stage but manual ones.
    pub fn order(&self, target: Option<&str>) -> Result<Vec<String>, String> {
        let roots: Vec<&str> = match target {
            Some(t) if !self.stages.contains_key(t) => return Err(format!("no stage named {t}")),
            Some(t) => vec![t],
            None => self
                .stages
                .iter()
                .filter(|(_, s)| !s.manual)
                .map(|(n, _)| n.as_str())
                .collect(),
        };
        self.order_from(roots)
    }

    /// Every stage in run order, manual ones included (for status).
    pub fn order_all(&self) -> Result<Vec<String>, String> {
        self.order_from(self.stages.keys().map(String::as_str).collect())
    }

    fn order_from(&self, roots: Vec<&str>) -> Result<Vec<String>, String> {
        let mut done = BTreeSet::new();
        let mut order = Vec::new();
        for root in roots {
            self.visit(root, &mut Vec::new(), &mut done, &mut order)?;
        }
        Ok(order)
    }

    fn visit<'a>(
        &'a self,
        name: &'a str,
        path: &mut Vec<&'a str>,
        done: &mut BTreeSet<&'a str>,
        order: &mut Vec<String>,
    ) -> Result<(), String> {
        if done.contains(name) {
            return Ok(());
        }
        if path.contains(&name) {
            path.push(name);
            return Err(format!("stages form a cycle: {}", path.join(" -> ")));
        }
        path.push(name);
        for input in &self.stages[name].inputs {
            self.visit(input, path, done, order)?;
        }
        path.pop();
        done.insert(name);
        order.push(name.to_string());
        Ok(())
    }

    /// Where a `files` entry points.
    pub fn resolve_file(&self, entry: &str) -> PathBuf {
        match entry.strip_prefix(CORPUS_PREFIX) {
            Some(rest) => self.corpus.join(rest),
            None => self.dir.join(entry),
        }
    }

    /// Stages that list `name` as an input.
    pub fn dependents(&self, name: &str) -> Vec<&str> {
        self.stages
            .iter()
            .filter(|(_, s)| s.inputs.iter().any(|i| i == name))
            .map(|(n, _)| n.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> Result<Pipeline, Error> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.toml");
        fs::write(&path, text).unwrap();
        Pipeline::load(&path)
    }

    #[test]
    fn orders_dependencies_first() {
        let p = load(
            r#"
            corpus = "c"
            [stage.label]
            command = ["x"]
            inputs = ["clean"]
            [stage.clean]
            command = ["x"]
            inputs = ["harvest"]
            [stage.harvest]
            command = ["x"]
            [stage.other]
            command = ["x"]
            "#,
        )
        .unwrap();
        assert_eq!(
            p.order(Some("label")).unwrap(),
            ["harvest", "clean", "label"]
        );
        let all = p.order(None).unwrap();
        assert_eq!(all.len(), 4);
        assert!(all.iter().position(|s| s == "clean") < all.iter().position(|s| s == "label"));
    }

    #[test]
    fn manual_stages_run_only_when_named() {
        let p = load(
            r#"
            corpus = "c"
            [stage.harvest]
            command = ["x"]
            [stage.steer]
            command = ["x"]
            inputs = ["harvest"]
            manual = true
            "#,
        )
        .unwrap();
        assert_eq!(p.order(None).unwrap(), ["harvest"]);
        assert_eq!(p.order(Some("steer")).unwrap(), ["harvest", "steer"]);
        assert_eq!(p.order_all().unwrap(), ["harvest", "steer"]);
        let needs_manual = load(
            "corpus = \"c\"\n[stage.a]\ncommand = [\"x\"]\nmanual = true\n[stage.b]\ncommand = [\"x\"]\ninputs = [\"a\"]\n",
        );
        assert!(needs_manual
            .unwrap_err()
            .to_string()
            .contains("must be manual too"));
    }

    #[test]
    fn rejects_cycles_and_unknown_inputs() {
        let cycle = load(
            r#"
            corpus = "c"
            [stage.a]
            command = ["x"]
            inputs = ["b"]
            [stage.b]
            command = ["x"]
            inputs = ["a"]
            "#,
        );
        assert!(cycle.unwrap_err().to_string().contains("cycle"));
        let unknown = load("corpus = \"c\"\n[stage.a]\ncommand = [\"x\"]\ninputs = [\"zz\"]\n");
        assert!(unknown
            .unwrap_err()
            .to_string()
            .contains("unknown stage zz"));
    }
}
