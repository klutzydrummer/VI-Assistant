//! `vi dataset ...` and `vi corpus ...`: the dataset pipeline's commands.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use dataset::runner::{Outcome, Status};
use dataset::{corpus, synth, Pipeline, Runner};

const DEFAULT_SYNTH_LINES: usize = 40;

fn runner(pipeline: &Path) -> Result<Runner, dataset::Error> {
    let vi = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("vi"));
    Runner::new(Pipeline::load(pipeline)?, vi)
}

fn fail(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {e}");
    ExitCode::FAILURE
}

pub fn run(pipeline: &Path, stage: Option<&str>, force: bool) -> ExitCode {
    let mut runner = match runner(pipeline) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    runner.echo = true;
    match runner.run(stage, force, |msg| eprintln!("{msg}")) {
        Ok(results) => {
            let ran = results.iter().filter(|r| r.outcome == Outcome::Ran).count();
            println!("{ran} ran, {} cached", results.len() - ran);
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

pub fn status(pipeline: &Path) -> ExitCode {
    let mut runner = match runner(pipeline) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    match runner.status() {
        Ok(stages) => {
            for (name, status) in stages {
                let line = match status {
                    Status::Cached { key } => format!("cached   {}", &key[..12]),
                    Status::Stale { key } => format!("stale    {}", &key[..12]),
                    Status::Waiting { on } => format!("waiting  on {on}"),
                };
                println!("{name:<12} {line}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

pub fn gc(pipeline: &Path) -> ExitCode {
    let mut runner = match runner(pipeline) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    match runner.gc() {
        Ok(r) => {
            println!(
                "removed {} old runs and {} objects, freed {:.1} MB",
                r.runs_removed,
                r.objects_removed,
                r.bytes_freed as f64 / 1e6
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

pub fn check(dir: &Path) -> ExitCode {
    match corpus::check(dir) {
        Ok(report) => {
            for e in &report.errors {
                println!("error: {e}");
            }
            println!(
                "{} lines, {:.1} minutes of audio, {} problems",
                report.lines,
                report.seconds / 60.0,
                report.errors.len()
            );
            if report.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => fail(e),
    }
}

/// With no folder, writes into `$VI_OUT`, so it works as a pipeline stage.
pub fn synth(dir: Option<&str>, lines: Option<&str>) -> ExitCode {
    let dir = match dir
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("VI_OUT").map(PathBuf::from))
    {
        Some(d) => d,
        None => return fail("give a folder, or run as a pipeline stage"),
    };
    let count = match lines.map(str::parse).unwrap_or(Ok(DEFAULT_SYNTH_LINES)) {
        Ok(n) => n,
        Err(_) => return fail("--lines takes a number"),
    };
    match synth::write(&dir, "tali", count) {
        Ok(lines) => {
            println!("wrote {} synthetic lines to {}", lines.len(), dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}
