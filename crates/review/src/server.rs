//! A tiny HTTP server for the review page, bound to localhost only.
//!
//! Game audio never leaves this machine, so the page is served from here
//! and nothing is fetched from anywhere else.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Response, Server};

use crate::corpus::{Corpus, Review, Verdict};
use crate::queue;

const PAGE: &str = include_str!("page.html");

/// About an hour of the reviewer's time.
pub const BUDGET_SECONDS: f32 = 3600.0;

/// What a request gets back, kept separate from the HTTP library so the
/// routes can be tested directly.
#[derive(Debug)]
pub enum Reply {
    Page,
    Json(Value),
    Audio(PathBuf),
    NotFound,
    Bad(String),
}

pub fn serve(mut corpus: Corpus, addr: &str) -> Result<(), String> {
    let server = Server::http(addr).map_err(|e| format!("could not listen on {addr}: {e}"))?;
    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let mut body = String::new();
        if method == Method::Post {
            if let Err(e) = request.as_reader().read_to_string(&mut body) {
                eprintln!("could not read request: {e}");
                continue;
            }
        }
        let reply = handle(&mut corpus, method == Method::Post, &url, &body);
        let sent = match reply {
            Reply::Page => request.respond(
                Response::from_string(PAGE).with_header(header("text/html; charset=utf-8")),
            ),
            Reply::Json(v) => request.respond(
                Response::from_string(v.to_string()).with_header(header("application/json")),
            ),
            Reply::Audio(path) => match std::fs::File::open(&path) {
                Ok(file) => request.respond(
                    Response::from_file(file)
                        .with_header(header(audio_type(&path)))
                        .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap()),
                ),
                Err(e) => request.respond(
                    Response::from_string(format!("{}: {e}", path.display())).with_status_code(404),
                ),
            },
            Reply::NotFound => {
                request.respond(Response::from_string("not found").with_status_code(404))
            }
            Reply::Bad(msg) => request.respond(Response::from_string(msg).with_status_code(400)),
        };
        if let Err(e) = sent {
            eprintln!("could not respond to {url}: {e}");
        }
    }
    Ok(())
}

pub fn handle(corpus: &mut Corpus, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    match (post, path) {
        (false, "/") => Reply::Page,
        (false, "/api/session") => Reply::Json(session(corpus)),
        (false, p) if p.starts_with("/api/clip/") => {
            clip(corpus, &decode(&p["/api/clip/".len()..]))
        }
        (false, p) if p.starts_with("/audio/") => {
            let id = decode(&p["/audio/".len()..]);
            match corpus.line(&id).and_then(|l| corpus.audio_path(l)) {
                Some(path) => Reply::Audio(path),
                None => Reply::NotFound,
            }
        }
        (true, "/api/review") => match record(corpus, body) {
            Ok(()) => Reply::Json(session(corpus)),
            Err(msg) => Reply::Bad(msg),
        },
        _ => Reply::NotFound,
    }
}

#[derive(Serialize)]
struct Correction {
    /// Mean of reviewer minus model; positive means the model scores low.
    bias: f32,
    /// Mean absolute nudge.
    size: f32,
    count: usize,
}

fn session(corpus: &Corpus) -> Value {
    let latest = corpus.latest();
    let kept = latest
        .values()
        .filter(|r| r.verdict == Verdict::Keep)
        .count();
    let flagged = latest
        .values()
        .filter(|r| r.verdict == Verdict::Flag)
        .count();
    let seconds: f32 = corpus.reviews.iter().map(|r| r.seconds).sum();

    let mut sums: BTreeMap<&str, (f32, f32, usize)> = BTreeMap::new();
    for r in latest.values().filter(|r| r.verdict == Verdict::Keep) {
        for (axis, model) in &r.model {
            if let Some(mine) = r.scores.get(axis) {
                let e = sums.entry(axis).or_default();
                e.0 += mine - model;
                e.1 += (mine - model).abs();
                e.2 += 1;
            }
        }
    }
    let corrections: BTreeMap<&str, Correction> = sums
        .into_iter()
        .map(|(axis, (sum, abs, n))| {
            let n_f = n as f32;
            (
                axis,
                Correction {
                    bias: sum / n_f,
                    size: abs / n_f,
                    count: n,
                },
            )
        })
        .collect();

    json!({
        "axes": corpus.axes,
        "dropped": corpus.dropped,
        "queue": queue::build(corpus),
        "labelled": corpus.labels.len(),
        "kept": kept,
        "flagged": flagged,
        "seconds": seconds,
        "budget_seconds": BUDGET_SECONDS,
        "corrections": corrections,
        "positive_from": queue::POSITIVE_FROM,
    })
}

fn clip(corpus: &Corpus, id: &str) -> Reply {
    let Some(line) = corpus.line(id) else {
        return Reply::NotFound;
    };
    let label = corpus.labels.get(id);
    let review = corpus.latest().get(id).map(|r| (*r).clone());
    let why = label.and_then(queue::rank);
    Reply::Json(json!({
        "line": line,
        "clip": corpus.clip(id),
        "label": label,
        "review": review,
        "why": why,
    }))
}

fn record(corpus: &mut Corpus, body: &str) -> Result<(), String> {
    let mut review: Review = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let label = corpus
        .labels
        .get(&review.id)
        .ok_or_else(|| format!("no labels for {}", review.id))?;
    review.model = label
        .scores
        .iter()
        .map(|(axis, s)| (axis.clone(), s.value))
        .collect();
    if review.verdict == Verdict::Keep {
        // Any axis not sent keeps the model's score.
        for (axis, value) in &review.model {
            review.scores.entry(axis.clone()).or_insert(*value);
        }
        review
            .scores
            .retain(|axis, _| label.scores.contains_key(axis));
        for v in review.scores.values_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    } else {
        review.scores.clear();
    }
    if review.verdict != Verdict::Flag {
        review.flag = None;
    }
    review.seconds = review.seconds.clamp(0.0, 300.0);
    review.at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    corpus.record(review).map_err(|e| e.to_string())
}

fn header(content_type: &str) -> Header {
    Header::from_bytes("Content-Type", content_type).unwrap()
}

fn audio_type(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("mp3") => "audio/mpeg",
        _ => "audio/wav",
    }
}

/// Percent-decoding for ids in paths.
fn decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(b) = s
                .get(i + 1..i + 3)
                .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
