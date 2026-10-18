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
    Page(&'static str),
    Json(Value),
    Audio(PathBuf),
    NotFound,
    Bad(String),
}

pub fn serve(mut corpus: Corpus, addr: &str) -> Result<(), String> {
    serve_with(addr, |post, url, body| handle(&mut corpus, post, url, body))
}

/// Serve any page's routes on `addr`.
pub fn serve_with(
    addr: &str,
    mut handle: impl FnMut(bool, &str, &str) -> Reply,
) -> Result<(), String> {
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
        let reply = handle(method == Method::Post, &url, &body);
        let sent = match reply {
            Reply::Page(page) => request.respond(
                Response::from_string(page).with_header(header("text/html; charset=utf-8")),
            ),
            Reply::Json(v) => request.respond(
                Response::from_string(v.to_string()).with_header(header("application/json")),
            ),
            Reply::Audio(path) => match std::fs::read(&path) {
                Ok(bytes) => {
                    // Phone browsers (Safari above all) only play media
                    // served in byte ranges.
                    let range = request
                        .headers()
                        .iter()
                        .find(|h| h.field.equiv("Range"))
                        .map(|h| h.value.as_str().to_string());
                    let len = bytes.len();
                    let base = |r: Response<std::io::Cursor<Vec<u8>>>| {
                        r.with_header(header(audio_type(&path)))
                            .with_header(Header::from_bytes("Accept-Ranges", "bytes").unwrap())
                            .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap())
                    };
                    match range.as_deref().map(|r| byte_range(r, len)) {
                        None => request.respond(base(Response::from_data(bytes))),
                        Some(Some((start, end))) => request.respond(
                            base(Response::from_data(bytes[start..=end].to_vec()))
                                .with_status_code(206)
                                .with_header(
                                    Header::from_bytes(
                                        "Content-Range",
                                        format!("bytes {start}-{end}/{len}"),
                                    )
                                    .unwrap(),
                                ),
                        ),
                        Some(None) => request.respond(
                            Response::from_string("range not satisfiable")
                                .with_status_code(416)
                                .with_header(
                                    Header::from_bytes("Content-Range", format!("bytes */{len}"))
                                        .unwrap(),
                                ),
                        ),
                    }
                }
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
        (false, "/") => Reply::Page(PAGE),
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

/// The inclusive byte span a `Range: bytes=...` header asks for in a file of
/// `len` bytes, or None if it can't be served. Only one span is honoured.
fn byte_range(header: &str, len: usize) -> Option<(usize, usize)> {
    let spec = header.trim().strip_prefix("bytes=")?;
    let spec = spec.split(',').next()?.trim();
    let (a, b) = spec.split_once('-')?;
    let (start, end) = match (a.trim(), b.trim()) {
        ("", suffix) => {
            let n: usize = suffix.parse().ok()?;
            if n == 0 {
                return None;
            }
            (len.saturating_sub(n), len.checked_sub(1)?)
        }
        (a, "") => (a.parse().ok()?, len.checked_sub(1)?),
        (a, b) => (
            a.parse().ok()?,
            b.parse::<usize>().ok()?.min(len.checked_sub(1)?),
        ),
    };
    (start <= end && end < len).then_some((start, end))
}

/// Percent-decoding for ids in paths.
pub(crate) fn decode(s: &str) -> String {
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

#[cfg(test)]
mod range_tests {
    use super::byte_range;

    #[test]
    fn byte_ranges() {
        assert_eq!(byte_range("bytes=0-", 10), Some((0, 9)));
        assert_eq!(byte_range("bytes=0-1", 10), Some((0, 1)));
        assert_eq!(byte_range("bytes=2-99", 10), Some((2, 9)));
        assert_eq!(byte_range("bytes=-4", 10), Some((6, 9)));
        assert_eq!(byte_range("bytes=-40", 10), Some((0, 9)));
        assert_eq!(byte_range("bytes=3-5, 7-8", 10), Some((3, 5)));
        assert_eq!(byte_range("bytes=10-", 10), None);
        assert_eq!(byte_range("bytes=5-2", 10), None);
        assert_eq!(byte_range("bytes=-0", 10), None);
        assert_eq!(byte_range("bytes=0-", 0), None);
        assert_eq!(byte_range("items=0-1", 10), None);
    }
}
