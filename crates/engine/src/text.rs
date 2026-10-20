//! Turning streamed model text into clean, speakable sentences.

/// Collects streamed text and hands back each sentence once it's complete,
/// so the voice can start on the first sentence while the rest streams.
#[derive(Debug, Default)]
pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    /// Add a chunk and return any sentences it completed.
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        while let Some(end) = sentence_end(&self.buf) {
            let rest = self.buf.split_off(end);
            let sentence = std::mem::replace(&mut self.buf, rest);
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_string());
            }
        }
        out
    }

    /// Whatever is left once the stream ends.
    pub fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        let rest = rest.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    }
}

/// Byte index just past the first complete sentence in `s`, if any.
///
/// A sentence ends at `.`, `!` or `?` (runs like `?!` or `...` included)
/// followed by whitespace and then the start of a new sentence: an
/// uppercase letter, a digit or an opening quote. Requiring what follows
/// means "I'm fine... totally fine" stays one sentence, and a sentence at
/// the very end of the buffer waits until more text or `finish` arrives.
fn sentence_end(s: &str) -> Option<usize> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        if matches!(chars[i].1, '.' | '!' | '?' | '…') {
            let mut j = i;
            while j + 1 < chars.len()
                && matches!(chars[j + 1].1, '.' | '!' | '?' | '…' | '"' | '\'' | ')')
            {
                j += 1;
            }
            let mut k = j + 1;
            let mut saw_space = false;
            while k < chars.len() && chars[k].1.is_whitespace() {
                saw_space = true;
                k += 1;
            }
            if saw_space && k < chars.len() {
                let next = chars[k].1;
                if (next.is_uppercase() || next.is_ascii_digit() || next == '"')
                    && !is_abbreviation(s, chars[i].0)
                {
                    return Some(chars[k].0);
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    None
}

fn is_abbreviation(s: &str, dot: usize) -> bool {
    let word = s[..dot]
        .rsplit(|c: char| c.is_whitespace())
        .next()
        .unwrap_or("");
    matches!(
        word,
        "Mr" | "Mrs" | "Ms" | "Dr" | "St" | "vs" | "e.g" | "i.e"
    )
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum ThinkState {
    /// Nothing but whitespace yet; a `<think>` may still be coming.
    #[default]
    Start,
    Thinking,
    /// Just past `</think>`, dropping the whitespace before the reply.
    AfterThink,
    Reply,
}

/// Separates a reasoning model's `<think>…</think>` block from what the
/// character says, as the text streams. Only a block at the very start
/// counts; tags split across chunks are handled. A block that never closes
/// (out of tokens) is all thinking and leaves no reply.
#[derive(Debug, Default)]
pub struct ThinkSplitter {
    state: ThinkState,
    buf: String,
    thinking: String,
    thought: bool,
}

impl ThinkSplitter {
    /// Add a chunk and return the part of it that's reply, if any.
    pub fn push(&mut self, chunk: &str) -> String {
        self.buf.push_str(chunk);
        let mut out = String::new();
        loop {
            match self.state {
                ThinkState::Start => {
                    let text = self.buf.trim_start();
                    if let Some(rest) = text.strip_prefix(THINK_OPEN) {
                        self.buf = rest.to_string();
                        self.state = ThinkState::Thinking;
                        self.thought = true;
                    } else if THINK_OPEN.starts_with(text) {
                        return out;
                    } else {
                        self.state = ThinkState::Reply;
                    }
                }
                ThinkState::Thinking => match self.buf.find(THINK_CLOSE) {
                    Some(end) => {
                        self.thinking.push_str(&self.buf[..end]);
                        self.buf.drain(..end + THINK_CLOSE.len());
                        self.state = ThinkState::AfterThink;
                    }
                    None => {
                        // Keep what could be the start of a split `</think>`.
                        let keep = (1..THINK_CLOSE.len())
                            .rev()
                            .find(|&n| self.buf.ends_with(&THINK_CLOSE[..n]))
                            .unwrap_or(0);
                        let cut = self.buf.len() - keep;
                        self.thinking.push_str(&self.buf[..cut]);
                        self.buf.drain(..cut);
                        return out;
                    }
                },
                ThinkState::AfterThink => {
                    let text = self.buf.trim_start();
                    if text.is_empty() {
                        self.buf.clear();
                        return out;
                    }
                    self.buf = text.to_string();
                    self.state = ThinkState::Reply;
                }
                ThinkState::Reply => {
                    out.push_str(&self.buf);
                    self.buf.clear();
                    return out;
                }
            }
        }
    }

    /// Whatever reply is left once the stream ends.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            ThinkState::Thinking => {
                self.thinking.push_str(&rest);
                String::new()
            }
            _ => rest,
        }
    }

    /// The thinking so far, trimmed; `None` if the model didn't think.
    pub fn thinking(&self) -> Option<String> {
        self.thought.then(|| self.thinking.trim().to_string())
    }
}

/// Strip what a voice can't say: markdown, emoji, bracketed asides and
/// stray whitespace. Words are never changed.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            _ if depth > 0 => {}
            '*' | '_' | '#' | '`' | '~' | '|' | '<' | '>' => {}
            c if is_emoji(c) => {}
            c if c.is_whitespace() => out.push(' '),
            c => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F | 0x200D | 0x2300..=0x23FF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(chunks: &[&str]) -> Vec<String> {
        let mut s = SentenceSplitter::default();
        let mut out: Vec<String> = chunks.iter().flat_map(|c| s.push(c)).collect();
        out.extend(s.finish());
        out
    }

    #[test]
    fn splits_streamed_text_into_sentences() {
        assert_eq!(
            split_all(&[
                "Good morn",
                "ing! You're up ",
                "early. Or did",
                " you not sleep?"
            ]),
            vec!["Good morning!", "You're up early.", "Or did you not sleep?"]
        );
    }

    #[test]
    fn keeps_trailing_ellipses_and_abbreviations_together() {
        assert_eq!(
            split_all(&["I'm fine... totally fine. Ask Dr. Chakwas. Keelah!"]),
            vec!["I'm fine... totally fine.", "Ask Dr. Chakwas.", "Keelah!"]
        );
    }

    #[test]
    fn waits_for_more_text_before_ending_a_sentence() {
        let mut s = SentenceSplitter::default();
        assert!(s.push("That's very strong.").is_empty());
        assert_eq!(s.push(" Mmm."), vec!["That's very strong."]);
        assert_eq!(s.finish().as_deref(), Some("Mmm."));
    }

    fn split_think(chunks: &[&str]) -> (Option<String>, String) {
        let mut s = ThinkSplitter::default();
        let mut reply: String = chunks.iter().map(|c| s.push(c)).collect();
        reply.push_str(&s.finish());
        (s.thinking(), reply)
    }

    #[test]
    fn keeps_thinking_out_of_the_reply() {
        assert_eq!(
            split_think(&["<think>\nShe'd be glad.\n</think>\n\nKeelah!"]),
            (Some("She'd be glad.".into()), "Keelah!".into())
        );
        assert_eq!(split_think(&["Keelah!"]), (None, "Keelah!".into()));
        assert_eq!(
            split_think(&["<think>\nThinking Process: 1. Greet."]),
            (Some("Thinking Process: 1. Greet.".into()), String::new())
        );
        assert_eq!(
            split_think(&["<think>\n\n</think>\n\nHi."]),
            (Some(String::new()), "Hi.".into())
        );
    }

    #[test]
    fn finds_think_tags_split_across_chunks() {
        assert_eq!(
            split_think(&[
                "\n<thi",
                "nk>Hmm, a ",
                "drink. </th",
                "ink>",
                "\n",
                "Oh. ",
                "Thanks."
            ]),
            (Some("Hmm, a drink.".into()), "Oh. Thanks.".into())
        );
        assert_eq!(split_think(&["<th", "e end"]), (None, "<the end".into()));
        assert_eq!(split_think(&["<"]), (None, "<".into()));
    }

    #[test]
    fn only_a_leading_think_block_counts() {
        assert_eq!(
            split_think(&["I said <think> once."]),
            (None, "I said <think> once.".into())
        );
    }

    #[test]
    fn normalize_strips_unspeakables_only() {
        assert_eq!(
            normalize("[happy] **Ha!** Got it 🎉 — the *drive*   core\nworks."),
            "Ha! Got it — the drive core works."
        );
    }
}
