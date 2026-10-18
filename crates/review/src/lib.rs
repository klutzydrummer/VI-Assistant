//! The review stage of the dataset pipeline: a local page that plays each
//! clip with its label scores, most uncertain first, and records the
//! reviewer's corrections. See `docs/design/review-page.md`.
//!
//! `purity` is the corpus pipeline's listening step: does each clip fit its
//! cluster, then name, merge, split or reject it. `gold` is where Brandon
//! labels a sample of her lines from their text, for the text-model bake-off.

pub mod corpus;
pub mod fixture;
pub mod gold;
pub mod purity;
pub mod queue;
pub mod server;

pub use corpus::Corpus;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use serde_json::Value;

    use super::corpus::{AxisScore, Label, Review, Verdict};
    use super::server::{handle, Reply};
    use super::*;

    fn demo(count: usize) -> (tempfile::TempDir, Corpus) {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), count).unwrap();
        let corpus = Corpus::load(dir.path()).unwrap();
        (dir, corpus)
    }

    fn label(value: f32, confidence: f32, signals: &[f32]) -> Label {
        let signals = signals
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("s{i}"), Some(*v)))
            .chain([("unscored".to_string(), None)])
            .collect();
        Label {
            id: "x".into(),
            scores: BTreeMap::from([(
                "joy".into(),
                AxisScore {
                    value,
                    confidence,
                    signals,
                },
            )]),
            tags: vec![],
        }
    }

    fn json(reply: Reply) -> Value {
        match reply {
            Reply::Json(v) => v,
            other => panic!("expected json, got {other:?}"),
        }
    }

    #[test]
    fn fixture_loads_with_pack_axes() {
        let (_dir, corpus) = demo(50);
        assert_eq!(corpus.dropped, 2, "clean drops every 25th clip");
        assert_eq!(corpus.lines.len(), 48);
        assert_eq!(corpus.labels.len(), 50);
        let axes: Vec<&str> = corpus.axes.iter().map(|a| a.id.as_str()).collect();
        let want: Vec<&str> = fixture::AXES.iter().map(|a| a.0).collect();
        assert_eq!(axes, want, "pack order from axes.json");
        assert!(corpus
            .lines
            .iter()
            .all(|l| corpus.audio_path(l).unwrap().exists()));
    }

    #[test]
    fn unsure_likely_positives_rank_first() {
        let unsure_positive = queue::rank(&label(0.7, 0.3, &[])).unwrap();
        let unsure_neutral = queue::rank(&label(0.05, 0.3, &[])).unwrap();
        let sure_positive = queue::rank(&label(0.7, 0.95, &[])).unwrap();
        assert!(unsure_positive.priority > unsure_neutral.priority);
        assert!(unsure_neutral.priority > sure_positive.priority);
    }

    #[test]
    fn disagreeing_signals_count_as_uncertain() {
        let agree = queue::rank(&label(0.5, 0.9, &[0.5, 0.5])).unwrap();
        let disagree = queue::rank(&label(0.5, 0.9, &[0.1, 0.9])).unwrap();
        assert!(disagree.priority > agree.priority);
        assert!(disagree.reason.contains("disagree"));
    }

    #[test]
    fn queue_is_sorted_and_skips_go_last() {
        let (_dir, mut corpus) = demo(20);
        let q = queue::build(&corpus);
        assert_eq!(q.len(), 20, "none dropped below 25");
        assert!(q.windows(2).all(|w| w[0].priority >= w[1].priority));

        let first = q[0].id.clone();
        corpus
            .record(Review {
                id: first.clone(),
                verdict: Verdict::Skip,
                scores: BTreeMap::new(),
                model: BTreeMap::new(),
                flag: None,
                seconds: 1.0,
                at: 0,
            })
            .unwrap();
        let q = queue::build(&corpus);
        assert_eq!(q.len(), 20);
        assert_eq!(q.last().unwrap().id, first);
    }

    #[test]
    fn calibration_sample_comes_first_in_hash_order() {
        let (_dir, corpus) = demo(50);
        let q = queue::build_with(&corpus, 10);
        let labelled = corpus
            .lines
            .iter()
            .filter(|l| corpus.labels.contains_key(&l.id))
            .map(|l| l.id.as_str());
        let expected = queue::calibration(labelled, 10);
        let first: Vec<&str> = q[..10].iter().map(|i| i.id.as_str()).collect();
        assert_eq!(first, expected);
        assert!(q[..10].iter().all(|i| i.reason.starts_with("calibration")));
        assert!(q[10..].iter().all(|i| !i.reason.starts_with("calibration")));
        assert!(q.windows(2).all(|w| w[0].priority >= w[1].priority));
    }

    #[test]
    fn calibration_is_sha256_order() {
        // sha256("b") = 3e23..., sha256("a") = ca97..., sha256("c") = 2e7d...
        assert_eq!(queue::calibration(["a", "b", "c"], 2), ["c", "b"]);
    }

    #[test]
    fn reviews_persist_and_latest_wins() {
        let (dir, mut corpus) = demo(10);
        let id = corpus.lines[0].id.clone();
        let body = |verdict: &str| {
            format!(r#"{{"id":"{id}","verdict":"{verdict}","scores":{{"joy":1.4}},"seconds":7}}"#)
        };
        json(handle(&mut corpus, true, "/api/review", &body("keep")));
        json(handle(&mut corpus, true, "/api/review", &body("flag")));

        let reloaded = Corpus::load(dir.path()).unwrap();
        assert_eq!(reloaded.reviews.len(), 2);
        let first = &reloaded.reviews[0];
        assert_eq!(first.scores["joy"], 1.0, "scores are clamped");
        assert_eq!(
            first.scores.len(),
            fixture::AXES.len(),
            "unsent axes keep the model score"
        );
        assert_eq!(first.model.len(), fixture::AXES.len());
        assert_eq!(reloaded.latest()[id.as_str()].verdict, Verdict::Flag);
        assert_eq!(queue::build(&reloaded).len(), 9);
    }

    #[test]
    fn session_reports_progress_and_corrections() {
        let (_dir, mut corpus) = demo(10);
        let id = corpus.lines[0].id.clone();
        let model = corpus.labels[&id].scores["joy"].value;
        let nudged = (model + 0.2).min(1.0);
        let body =
            format!(r#"{{"id":"{id}","verdict":"keep","scores":{{"joy":{nudged}}},"seconds":12}}"#);
        let s = json(handle(&mut corpus, true, "/api/review", &body));
        assert_eq!(s["kept"], 1);
        assert_eq!(s["seconds"], 12.0);
        assert_eq!(s["queue"].as_array().unwrap().len(), 9);
        let bias = s["corrections"]["joy"]["bias"].as_f64().unwrap() as f32;
        assert!((bias - (nudged - model)).abs() < 1e-4);
    }

    #[test]
    fn routes() {
        let (_dir, mut corpus) = demo(3);
        let id = corpus.lines[0].id.clone();
        assert!(matches!(
            handle(&mut corpus, false, "/", ""),
            Reply::Page(_)
        ));
        let clip = json(handle(&mut corpus, false, &format!("/api/clip/{id}"), ""));
        assert_eq!(clip["line"]["id"], id.as_str());
        assert!(clip["why"]["reason"].is_string());
        assert!(matches!(
            handle(&mut corpus, false, &format!("/audio/{id}"), ""),
            Reply::Audio(_)
        ));
        assert!(matches!(
            handle(&mut corpus, false, "/audio/nope", ""),
            Reply::NotFound
        ));
        assert!(matches!(
            handle(
                &mut corpus,
                true,
                "/api/review",
                r#"{"id":"nope","verdict":"keep"}"#
            ),
            Reply::Bad(_)
        ));
    }

    #[test]
    fn audio_paths_stay_inside_the_stage() {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), 2).unwrap();
        let clips = dir.path().join("stages/clean/synthetic/out/clips.jsonl");
        let text = fs::read_to_string(&clips).unwrap();
        let mut rows = text.lines();
        let (a, b) = (rows.next().unwrap(), rows.next().unwrap());
        let a = a.replacen("\"audio/", "\"../../../../", 1);
        let b = b.replacen("\"audio/", "\"/etc/", 1);
        fs::write(&clips, format!("{a}\n{b}\n")).unwrap();

        let mut corpus = Corpus::load(dir.path()).unwrap();
        assert!(corpus.lines.iter().all(|l| corpus.audio_path(l).is_none()));
        let id = corpus.lines[0].id.clone();
        assert!(matches!(
            handle(&mut corpus, false, &format!("/audio/{id}"), ""),
            Reply::NotFound
        ));
    }

    #[test]
    fn without_clean_plays_harvest_audio() {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), 30).unwrap();
        fs::remove_dir_all(dir.path().join("stages/clean")).unwrap();
        let corpus = Corpus::load(dir.path()).unwrap();
        assert_eq!(corpus.lines.len(), 30);
        let path = corpus.audio_path(&corpus.lines[0]).unwrap();
        assert!(path.starts_with(dir.path().join("stages/harvest")));
        assert!(path.exists());
    }

    #[test]
    fn missing_stage_is_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), 3).unwrap();
        fs::remove_file(dir.path().join("stages/label/current")).unwrap();
        let err = Corpus::load(dir.path()).err().unwrap();
        assert!(err.to_string().contains("label stage has not run"));
    }
}
