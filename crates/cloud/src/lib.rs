//! The rented GPU worker: `vi cloud up`, `down`, `status` and the idle timeout.
//!
//! The worker is stateless (see `docs/design/local-cloud-split.md`), so all this
//! crate keeps at home is which instance is ours, how to reach it and when it was
//! last used. Two guards protect the budget: the instance runs a watchdog that
//! destroys itself after `idle_minutes` without jobs or ssh sessions (or after
//! `max_hours` regardless), and `vi cloud reap` does the same from home.

pub mod config;
pub mod onstart;
pub mod state;
pub mod vast;
pub mod worker;

pub use config::Config;
pub use state::{StateDir, Tracked};
pub use vast::{Endpoint, HttpVast, Instance, Offer, Vast};
pub use worker::{Cloud, Host, SystemHost};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {message}")]
    Config {
        path: std::path::PathBuf,
        message: String,
    },
    #[error("Vast.ai API: {0}")]
    Api(String),
    #[error("{0}")]
    Worker(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io_err(path: &std::path::Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_owned(),
        source,
    }
}
