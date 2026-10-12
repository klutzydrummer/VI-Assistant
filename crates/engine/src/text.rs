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

    #[test]
    fn normalize_strips_unspeakables_only() {
        assert_eq!(
            normalize("[happy] **Ha!** Got it 🎉 — the *drive*   core\nworks."),
            "Ha! Got it — the drive core works."
        );
    }
}
