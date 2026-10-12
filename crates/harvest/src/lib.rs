//! Readers for Mass Effect Legendary Edition game files, the groundwork of
//! the dataset harvester (stage 1 in `docs/design/control-vector-datasets.md`).
//!
//! - [`package`]: Unreal Engine 3 `.pcc` packages, including the Oodle
//!   Leviathan decompression LE uses (via the pure-Rust `oozextract`).
//! - [`tlk`]: subtitle tables, both the standalone `.tlk` files of LE2/LE3
//!   and the `BioTlkFile` exports inside LE1 packages.
//! - [`install`]: an inventory of an install folder.
//!
//! The formats were learned from LegendaryExplorerCore's source, used as
//! documentation only (it is GPL-3; no code is copied). Nothing here has been
//! run against real game files yet: everything is tested on synthetic files
//! built by [`fixture`], and the places that most need checking on a real
//! install are marked "verify".

pub mod fixture;
pub mod install;
pub mod package;
pub mod props;
mod reader;
pub mod tlk;

pub use package::Package;

/// A file that doesn't parse as the format it should be.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("at byte {offset}: {message}")]
    Format { offset: usize, message: String },
    #[error("{0}")]
    Unsupported(String),
    #[error("oodle: {0}")]
    Oodle(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Which game a file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Game {
    Le1,
    Le2,
    Le3,
}

impl Game {
    /// The id used in corpus line ids (`le2.s348112_f`).
    pub fn id(self) -> &'static str {
        match self {
            Game::Le1 => "le1",
            Game::Le2 => "le2",
            Game::Le3 => "le3",
        }
    }
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}
