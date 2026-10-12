//! The dataset pipeline's file formats and its cached stage runner.
//!
//! A corpus is a folder of stage outputs. Each stage is an external command
//! (the extractor, a Python GPU job) that reads its inputs' folders and
//! writes its own. The runner keys every run by the hashes of everything
//! that went into it, so rerunning the pipeline only redoes stages whose
//! inputs changed. See `docs/design/corpus-format.md`.

pub mod corpus;
pub mod hash;
pub mod pipeline;
pub mod runner;
pub mod synth;
pub mod wav;

pub use pipeline::Pipeline;
pub use runner::Runner;

use std::path::PathBuf;

/// Version of the corpus format and run manifests.
pub const FORMAT: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid {path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("stage {stage} failed: {message}")]
    Stage { stage: String, message: String },
}

pub(crate) trait IoContext<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T, Error>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T, Error> {
        self.map_err(|source| Error::Io {
            path: path.into(),
            source,
        })
    }
}
