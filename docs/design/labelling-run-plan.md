# Open-vocabulary labelling run: the plan

Status: proposal, 2026-10-18, for Brandon to approve. Nothing has been sent to the API.

Goal: one paid pass of Claude over all 3,085 corpus lines that we never need to repeat.
Each line gets free emotion words, consistent scores for the moods we already care about,
and the degree knobs (energy, intoxication, physical state). The output has to feed three
things: a merged emotion vocabulary for Brandon to prune, a per-mood ranking of lines for
his listening review, and in the end the per-mood control vectors for the TTS and the chat model.

Budget: $20 of API credit, pilot included. Only text goes out (lines, conversation, scene
data, voice measurements written as words). Audio never leaves home.

## 1. The core design choice: free words plus a fixed checklist

A purely open list has a known flaw for our use: the model names what is most salient and
silently drops the rest, and it drops different things on similar lines. If "worried" is
missing on a line, we can't tell "not worried" from "not mentioned". That makes per-mood
ranking unreliable, which is exactly what the listening review needs.

A purely fixed list has the flaw Brandon hit while labelling: regret, loneliness and the
like have nowhere to go.

So each line gets both, in this order (the schema fixes the order, so the free list is
written before the model sees the checklist in its own output):

1. **emotions** (free): every feeling present, in the model's own everyday words, each with
   a strength, what it rests on, and a short reason. This finds what we didn't ask for.
2. **checklist** (fixed): a 0 to 3 score for every mood on a fixed list, on every line.
   This gives a complete, rankable score per mood and scores Brandon's gold labels like for like.

The checklist is the 21 moods in `characters/tali/moods.toml` (Brandon's gold vocabulary)
plus seven that the gold list had no room for: afraid, determined, regretful, lonely,
guilty, hopeful, and irritated (split from angry, since irritation and anger are different
registers for a friend). The prompt tells the model to choose free words first on their
own merits and only reuse a checklist word when it really is the best word.

## 2. What goes in for each line

Kept from the current PC prompt (`corpus/tali/analysis/annotate/digest.py`):
- `<where>`: game, scene tags, conversation, who she is speaking to.
- `<before>`: the lines just before hers, including Shepard's dialogue choice.
- `<line>`: her line, with the director's note when there is one. Notes starting `TEMP:`
  are flagged as placeholder notes that may not match the final scene.
- `<replies>`: what follows, with alternatives marked "only one of these plays".
- `<voice>`: length, pace, pitch, pitch movement, loudness and loudness variation, in words,
  compared with her lines **from the same game** (LE1 is mastered darker, so a cross-game
  baseline would make every LE1 line look "low and quiet"). Check digest.py does this.

Changes:
- **Add a short character brief** (cached, about 300 words): where she is in her arc in each
  game (pilgrimage, exile trial and Haestrom, the Rannoch war), the suit and filter, her
  speech habits (rambling when nervous, "bosh'tet", "keelah"). The model knows the games, but
  the brief pins the facts it should rely on.
- **Drop the speech-emotion classifier readings** (emotion2vec, EmoNet) entirely. They were
  wrong on two of the three rendered examples (disgusted 89% on a quiet, subdued line about the
  Collectors, amusement on a combat shout). Showing a wrong reading and then saying
  "don't trust it" still anchors the model. The plain measurements stay; they are honest.
- **Few-shot examples are tested, not assumed.** Brandon's labelled lines use the old
  20-word list, so as examples they teach that narrow vocabulary, which is the opposite of
  what this run is for. The base prompt uses no examples; one pilot arm adds them.
  Instead, the base prompt anchors the scales in words (below), which is what the examples
  were mainly for.

## 3. What comes out (structured outputs, enforced by the API)

Per line:

| Field | Type | Meaning |
|---|---|---|
| `emotions` | list, strongest first | `{word, strength, basis, why}` |
| `emotions[].word` | text | one everyday word or a two-word phrase, lowercase, no intensity words ("sad", not "very sad") |
| `emotions[].strength` | 1, 2 or 3 | 1 faint undertone, 2 clearly present, 3 the main feeling |
| `emotions[].basis` | words, situation, delivery | what it mainly rests on; tells us which moods are heard in her voice (TTS vectors) and which live only in the text (chat-only) |
| `emotions[].why` | text, at most 12 words | for Brandon's review and the vocabulary merge |
| `checklist` | object, all 28 keys required | each 0 to 3, same scale as strength |
| `doing` | text, at most 8 words | what she does with the line: teasing Garrus, reporting a position |
| `energy` | 1 to 5 | 1 subdued, 2 calm, 3 normal talk, 4 animated or raised, 5 urgent or shouting |
| `intoxication` | 0 to 3 | 0 sober, 1 tipsy, 2 drunk, 3 very drunk |
| `body` | list from: tired, sick, injured, out_of_breath, hungover | physical state; empty when none |
| `delivery_matches_words` | true, false | false for sarcasm, deadpan, forced cheer, held-back tears |
| `confidence` | 1 to 3 | 1 when the context is too thin to judge (a bark with no scene) |

Notes on the choices:
- **Energy is 1 to 5, not calm/normal/urgent.** Control vectors are degrees, and five
  steps can always be binned back to three (1-2 calm, 3 normal, 4-5 urgent). The prompt
  says energy means intensity of delivery, not only danger: Brandon rated a drunk party line to Javik
  as urgent, so loud excitement counts as high energy.
- **Hungover moves out of intoxication into `body`.** Hungover is not a level of
  drunkenness, and the old axis is what caused the "drunk sober" confusion. `body` also
  catches her ME2 illness lines and tired lines, which the knob plan said it might want later.
- **No free `notes` field.** It was the most expensive field per useful bit.
- Integer fields use enums, so the API rejects anything off the scale.

## 4. The prompt, in outline

System prompt (identical for every request, cached):
1. Purpose: these labels pick which of her real recordings teach a voice model and a chat
   model each feeling, and her owner checks the picks by ear. So mark what is in this line at
   this moment, not what is typical for her or for the situation.
2. The character brief.
3. What each input section is, in the order they appear.
4. Each output field, defined once, using the same names as the schema.
5. The checklist moods, each with a one-line definition (from moods.toml, plus the seven new ones).
6. Rules that keep the run consistent: rate faint feelings rather than leaving them out;
   an empty emotion list only for a genuinely neutral report; when the line could belong to
   two branches, judge the most likely one and lower confidence.

User message: the line block, then one closing instruction. Long input first, question last.

## 5. Model, effort and request shape

Prices (Claude API, batch is half price): Opus 5.5 $2 in / $10 out per million tokens
batched; Sonnet 5.5 $1 / $5. Thinking can't be turned off on either; `effort` controls it.

Estimate per line, assuming no cache hits (batch caching is best effort): about 2,000 input
tokens, about 300 output tokens of JSON, plus 300 to 1,000 thinking tokens. Output is most
of the cost.

| Shape | Opus 5.5 | Sonnet 5.5 |
|---|---|---|
| One line per request | about $30 to $50 | about $15 to $25 |
| One conversation per request (up to 8 of her lines) | about $15 to $20 | about $8 to $10 |

So **Opus on every line separately probably doesn't fit in $20**. Two shapes might, and the
pilot measures real token counts instead of guessing:
- **Grouped by conversation**: her lines from one conversation go in one request, each with
  its own before/after window and its own answer. The brief, instructions and thinking are
  shared, and the model sees the scene once, which may make labels within a scene more
  consistent. The risk is that labels blur across lines in a long scene, which the pilot checks.
- **Sonnet**, one line per request.

## 6. The pilot (about $3 to $4, hard cap $5)

The same 40 of Brandon's gold lines in every arm, sent through the batch API.

| Arm | Model | Effort | Shape | Prompt |
|---|---|---|---|---|
| A | Opus 5.5 | low | single | base |
| A2 | Opus 5.5 | low | single | base, run again (how much the model disagrees with itself) |
| M | Opus 5.5 | medium | single | base |
| S | Sonnet 5.5 | low | single | base |
| G | Opus 5.5 | low | grouped | base; the gold line's conversation-mates come along and are reused in the full run if G wins |
| X | Opus 5.5 | low | single | base plus Brandon's labelled examples |

Free words are grouped into synonym families before any comparison (a first quick pass of
the merge in section 8), so "remorse" in one arm and "regret" in another count as the same.
Open-vocabulary benchmarks such as OV-MER do this for the same reason.

What gets measured:
1. **Gold recall**: of the moods Brandon marked, how many the checklist scores 1 or more.
   This is the fair half of the comparison. A mood the model adds that he didn't mark is
   **not** counted as wrong, because he picked from 20 words and often only the top few.
2. **Extras Brandon agrees with**: every free word and checklist score of 2 or more outside his
   gold label, pooled across arms, deduplicated per line, shuffled, with the arm hidden.
   He marks agree or disagree while the clip plays. Roughly 250 clicks, about 20 minutes.
   This is the measure of what the open list adds, and it is his call, not the model's.
3. **Energy** within one step of his rating (after binning), and **intoxication** exact.
4. **Over-reading neutral lines**: on lines Brandon marked informational, how often an arm
   adds emotions at strength 2 or more. A GoEmotions study found GPT-4 tends to read feeling
   into neutral sentences.
5. **Self-consistency** (A against A2): the noise floor. Differences between arms smaller
   than this mean nothing.
6. **Measured cost per line** for each arm, from the API's usage numbers.

Decision rule, fixed before any result is seen:
- Only arms whose measured full-run cost fits the remaining credit with a 15% margin qualify.
- Among those, the best on gold recall and extras-agreed wins. If a cheaper arm is within
  5 points on recall and 10 points on extras-agreed, the cheaper one wins.
- X (examples) is kept only if it beats A by more than the A-A2 noise floor on extras-agreed.
- If no arm qualifies, I come back with the options (more credit, Opus for conversations
  and Sonnet for combat barks, or fewer lines) rather than picking one.

## 7. The full run, done so it never needs repeating

- Brandon sees the measured projection and okays this specific run first.
- Sent in chunks of about 500 lines. After the first chunk: parse failures, refusals,
  empty emotion lists and cost per line are checked; anything more than 20% over the
  projection stops the run.
- Every request and raw response is saved with the model, the schema version and the usage,
  so any later analysis is free. Failed or refused lines are retried one by one; a line that
  still fails is listed, not guessed.
- Song lines (`singing`) are included and flagged, since energy still matters there.

## 8. After the run (pennies, or free)

1. **Merge the vocabulary.** The unique emotion words (likely a few hundred) are grouped by a
   local text embedder, then one cheap Claude call over the word list alone (no game text)
   proposes families, such as remorse and regret becoming one entry. Brandon prunes it.
2. **Rank lines per mood** from the checklist score plus the free words mapped through the
   merged vocabulary, weighted by strength and confidence. Words with `basis: delivery`
   or `situation` say which moods are voice-borne and which are chat-only.
3. These rankings feed the existing route: nominate, Brandon confirms by ear, then vectors.
   Model output is never stored as his label.

## 9. What Brandon is asked to decide

1. Approve this plan. The PC session then changes digest.py and annotate.py to match and
   renders three prompts again for him to read. That step is free.
2. Approve the pilot run, capped at $5, once he has read those renders.
