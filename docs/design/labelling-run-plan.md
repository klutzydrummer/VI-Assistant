# Claude labelling run: the plan (v3)

Status: proposal v3, 2026-10-18, for Brandon to approve. Nothing has been sent to the API.
Checked against `labelling-run-requirements.md` and `/mnt/project-files/design/what-brandon-wants.md`.

**What changed in v3 (Brandon, 23:55):** Claude is the main source of mood labels. Brandon's
150 gold labels, and the old 20-word list they were picked from, are removed from everything:
the prompt, the scoring and the model choice. His ears are an optional spot check only. The
$20 is Claude credit alone, separate from Vast credit.

Goal: label all 3,085 corpus lines once, well enough to build the mood vocabulary, per-mood
rankings, and per-mood control vectors (voice and chat), plus the energy, drunk and hungover
knobs. Only text goes out: lines, conversation, scene data, and voice measurements in words.

## 1. The shape: Claude finds the moods, then Claude scores every line on them

There are two ways to label. Each covers the other's weakness, so the plan uses both in order.

**Open listing** (no list given): Claude names whatever emotions it sees. This is best for
finding moods, including ones nobody thought of. It's weak for building vectors, for three
reasons:
- **Omissions:** the model names what stands out and drops the rest, differently on similar
  lines. So a missing word doesn't mean the mood is absent, and nothing can be ranked.
- **Synonyms:** "remorse" and "regret" split one mood into two thin ones.
- **No zeros:** a vector needs lines that clearly *lack* the mood, as well as lines that have it.

**Scoring against a list**: every line gets a 0 to 3 for every mood. This gives the complete,
rankable scores and real zeros that the vectors need. But it only covers moods on the list.

So:
1. **Pass 1, discover (open listing):** Claude lists the emotions in every line, freely.
2. **Vocabulary:** Claude merges its own words into mood families. The list comes from Claude
   reading her lines, not from Brandon's list.
3. **Pass 2, score:** Claude scores every line 0 to 3 on every family, plus the knobs, plus free
   words for anything still missing.

Pass 1 and pass 2 are two independent views of each line. Where they agree, a line is a
trusted positive or a trusted zero, with no listening needed.

**One exception to "Claude's list only":** the friend-and-assistant moods Brandon asked for
(`design/must-have-moods.md`, e.g. reassuring, grateful, teasing) must be covered. After pass 1,
any that Claude's vocabulary lacks are added to pass 2's list, so they get complete scores too.
This list isn't the gold labels, and pass 1 never sees it.

## 2. What goes in (both passes)

Per line, built by `digest.py` on Brandon's PC:
- `<where>`: game, scene tags, conversation, listener.
- `<before>`: up to three lines before hers, including Shepard's choice.
- `<line>`: her line plus the director's note; `TEMP:` notes are flagged as placeholders.
- `<replies>`: what follows, with alternatives marked "only one of these plays".
- `<voice>`: pace, pitch, pitch movement, loudness and loudness variation in words, compared
  with her lines from the same game.

System prompt, frozen and cached for an hour:
1. Purpose: these labels pick her recorded lines that teach a voice and chat model how she
   sounds and talks in each feeling. Judge this line at this moment, not what's typical for her.
2. A character brief of about 300 words: her arc in each game, the suit, her speech habits.
3. What each input section is.
4. Each output field, defined once with one name.

Left out: Brandon's gold labels and old word list, any examples, and the speech-emotion
classifier readings (wrong for her filtered voice). The code adds a `context_thin` flag
(no lines before, no scene) instead of asking the model.

## 3. What comes out

Structured outputs; the API enforces the schema.

**Pass 1 (open):**

| Field | Type | Meaning |
|---|---|---|
| `reading` | text, about 15 words, first | what is going on for her in this moment |
| `emotions` | list of `{word, strength 1 to 3}`, strongest first | every feeling present, in everyday words; empty only for a plain neutral report |
| `doing` | text, at most 8 words | what she does with the line |

**Pass 2 (score):**

| Field | Type | Meaning |
|---|---|---|
| `reading` | text, about 15 words, first | as above |
| `scores` | object, one key per family, each 0 to 3 | 0 not there, 1 faint undertone, 2 clearly present, 3 a main feeling |
| `other_feelings` | list of `{word, strength}` | anything no family covers; often empty |
| `stated` | stated, implied, none | whether her strongest feeling is said outright |
| `opposite_meaning` | true, false | sarcasm, irony, forced cheer |
| `energy` | 1 to 5 | subdued, calm, normal, animated, urgent or shouting; covers loud excitement, not only danger |
| `intoxication` | 0 to 3 | sober, tipsy, drunk, very drunk |
| `body` | list from tired, sick, injured, out_of_breath, hungover | physical state |

Integer fields are enums, so values stay on the scale.

## 4. Models and cost

One line per request through the batch API (half price). No grouping: labels smear across a
scene and one failure loses several lines.

Batch prices per million tokens: Sonnet 5.5 $1 in, $5 out; Opus 5.5 $2 in, $10 out. Sonnet can
run with thinking off; Opus always thinks a little, even at effort low. Prices are re-checked
the day of the run.

| Step | Default | Estimate |
|---|---|---|
| Pilot | 60 lines × 3 configurations × 2 samples | about $1 |
| Pass 1, all lines, open | Sonnet, thinking off (Opus if the pilot says so) | about $3 to $4 (Opus: about $10 to $12) |
| Vocabulary | one call over the word list alone | about $0.10 |
| Pass 2, all lines, about 40 to 60 families | Sonnet, thinking off | about $6 to $9 |
| Reserve | a second pass-2 sample on unclear lines | the rest |

The total is about $11 to $15 on Sonnet. Opus for pass 1 brings it to about $18 to $22, which
fits only if the pilot measures lower costs than estimated. These estimates assume a cached
1,000 to 1,600-token prefix, a 320-token line, and 150 (pass 1) or 350 (pass 2) output tokens.

## 5. Pilot (about $1), choosing the model without gold

P0 (free): the PC renders three prompts of each pass for Brandon to read, and counts real
input tokens with the free `count_tokens` call.

P1: 60 random lines, pass-1 style, through Opus (effort low), Sonnet (thinking off) and Sonnet
(effort low), **each run twice**. Words are grouped into families before comparing. It measures:
1. **Self-agreement:** how much each configuration agrees with its own second run. This is
   the noise floor.
2. **Agreement with Opus:** how close each Sonnet setup comes to Opus, the strongest model.
3. **Cost per line**, from the API's usage numbers, plus parse failures and refusals.
4. Optional for Brandon, by reading only: on 20 lines he picks which of two unlabelled lists
   fits the line better. About 20 clicks.

Decision rule, fixed before any result:
- Sonnet thinking off is the default.
- Opus (or Sonnet with thinking) replaces it only if Sonnet's agreement with Opus is clearly
  below Opus's agreement with itself (by more than 0.1 on family overlap), **and** the full-run
  projection still fits $20 with a $1.50 reserve. If Brandon does the optional picks, Opus also
  has to win most of them.
- If nothing fits, stop and bring Brandon the options.

Bugs get fixed in the pilot and only there. Then both prompts are frozen (hash recorded).

## 6. The runs, done so they never need repeating

- Brandon okays pass 1 after the pilot, and pass 2 after seeing the vocabulary. Each is its
  own go line.
- Lines are shuffled across chunks. The first chunk is 300 lines, and the run stops if: cost per
  line is more than 20% over projection; parse failures, truncations or refusals top 1%; or the
  cache isn't being read (then re-project).
- A running spend counter from the usage numbers stops everything at $20 minus the reserve.
- Every request and raw response is stored with the model id, prompt hash and usage. Failures
  are retried once on the same model, and anything still failing is listed. No silent switch
  to another model.

## 7. Vocabulary (between the passes)

- Pass 1 words are lowercased and lemmatised, then grouped with a local text embedder.
- One small Claude call over the word list alone (no game text) names the families and drops
  near-duplicates. Families seen in fewer than about 5 lines are kept as free words, not scored.
- The coverage check adds any must-have mood the families lack.
- Brandon may skim the list. It isn't a gate.

## 8. After pass 2 (free)

1. **Score matrix:** lines × families, 0 to 3.
2. **Two-view agreement:** a line is a trusted positive for a mood when pass 2 scores it 3, or
   2 with pass 1 having named it. It's a trusted zero when pass 2 scores 0 and pass 1 didn't
   name it. Disagreements are flagged; the reserve buys a second pass-2 sample on those.
3. **Rankings per mood:** by score, preferring lines where the mood stands out, spread across
   games and scenes so one event (the LE3 party) doesn't fill a mood.
4. **Sets for vectors:** trusted positives against trusted zeros, matched on game, measured
   energy and length. Measured energy is factored out, so a mood vector doesn't absorb the
   energy knob. Thin moods are flagged for synthetic expansion later.
5. **Brandon's spot check (optional):** reading about 5 top lines per mood, playing the clip
   only if he wants. It can remove lines; it never gates a mood.

Open, not decided here: what checks that a mood sounds consistent in her voice before it gets
a voice vector (open question 1 on the one-page goals). This plan only produces the labels.

## 9. What Brandon is asked to decide

1. Approve v3. The PC rebuilds `digest.py` and `annotate.py` and renders example prompts (free).
2. The pilot (about $1), then pass 1, then pass 2: each on its own go line.
