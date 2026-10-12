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
                if !self.stages.contains_key(input) {
                    out.push(format!("stage {name} needs unknown stage {input}"));
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
    /// stage and what it needs.
    pub fn order(&self, target: Option<&str>) -> Result<Vec<String>, String> {
        let roots: Vec<&str> = match target {
            Some(t) if !self.stages.contains_key(t) => return Err(format!("no stage named {t}")),
            Some(t) => vec![t],
            None => self.stages.keys().map(String::as_str).collect(),
        };
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
