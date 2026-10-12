//! The turn pipeline: director, prompt assembly, and streaming a reply
//! sentence by sentence to the voice. Models plug in through the
//! [`pipeline::Dialogue`] and [`pipeline::Voice`] traits; [`stub`] has
//! placeholders for both.

pub mod director;
pub mod pipeline;
pub mod prompt;
pub mod stub;
pub mod text;

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Mutex;

    use charpack::Pack;

    use crate::director::{Delivery, RulesDirector};
    use crate::pipeline::{Role, Session, Voice};
    use crate::stub::ScriptedDialogue;

    #[derive(Default)]
    struct RecordingVoice(Mutex<Vec<(String, Delivery)>>);

    impl Voice for RecordingVoice {
        async fn speak(&self, sentence: &str, delivery: &Delivery) -> Result<(), String> {
            self.0
                .lock()
                .unwrap()
                .push((sentence.to_string(), delivery.clone()));
            Ok(())
        }
    }

    fn tali() -> Session {
        let pack = Pack::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../characters/tali"))
            .unwrap();
        let machine = pack.machine().unwrap().unwrap();
        Session::new(pack, machine)
    }

    fn block<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn a_drinking_turn_flows_from_cue_to_voice() {
        let mut session = tali();
        let director = RulesDirector::new(session.machine.spec());
        let dialogue = ScriptedDialogue::new([
            "Ooh, turian brandy? You know me so well. That's... that is very strong.",
            "I'm fiiine. Totally fine!",
        ]);
        let voice = RecordingVoice::default();

        let first =
            block(session.turn("Here, have a drink. Cheers!", &director, &dialogue, &voice))
                .unwrap();
        assert_eq!(first.events, vec!["drink"]);
        assert_eq!(session.snapshot.state, "tipsy");
        assert_eq!(
            first
                .sentences
                .iter()
                .map(|(s, _)| s.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Ooh, turian brandy?",
                "You know me so well.",
                "That's... that is very strong."
            ]
        );
        let drunk_voice = first.sentences[0].1.vectors["drunk"];
        assert!(drunk_voice > 0.0);

        block(session.turn("Another round!", &director, &dialogue, &voice)).unwrap();
        assert_eq!(session.snapshot.state, "drunk");
        let spoken = voice.0.lock().unwrap();
        assert!(spoken.last().unwrap().1.vectors["drunk"] > drunk_voice);
        assert_eq!(session.history.len(), 4);
        assert_eq!(session.history[3].role, Role::Character);
    }

    #[test]
    fn mood_reaches_the_prompt_as_plain_words() {
        let mut session = tali();
        let director = RulesDirector::new(session.machine.spec());
        let dialogue = ScriptedDialogue::new(["Hey. Talk to me."]);
        let voice = RecordingVoice::default();
        block(session.turn(
            "I'm so stressed and exhausted.",
            &director,
            &dialogue,
            &voice,
        ))
        .unwrap();
        let prompt =
            crate::prompt::system_prompt(&session.pack, &session.machine, &session.snapshot);
        assert!(prompt.contains("You feel a little worried."), "{prompt}");
        assert!(prompt.contains("Raala (Raala vas Paus)"));
        assert!(!prompt.contains("[worried]"));
    }

    #[test]
    fn a_failed_turn_changes_nothing() {
        struct Broken;
        impl crate::pipeline::Dialogue for Broken {
            async fn stream(
                &self,
                _req: &crate::pipeline::DialogueRequest,
                _out: tokio::sync::mpsc::Sender<String>,
            ) -> Result<(), String> {
                Err("worker went away".into())
            }
        }
        let mut session = tali();
        let director = RulesDirector::new(session.machine.spec());
        let before = session.snapshot.clone();
        let err = block(session.turn(
            "Have a drink!",
            &director,
            &Broken,
            &RecordingVoice::default(),
        ));
        assert_eq!(err, Err("worker went away".into()));
        assert_eq!(session.snapshot, before);
        assert!(session.history.is_empty());
    }
}
