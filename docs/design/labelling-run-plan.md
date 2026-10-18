# Open-vocabulary labelling run: the plan (v2)

Status: proposal v2, 2026-10-18, for Brandon to approve. Nothing has been sent to the API.
v2 adopts an independent critic's review (`/mnt/project-files/annotate/critic-review.md`,
with sources). The main changes from v1 are at the end.

Goal: one paid pass of Claude over all 3,085 corpus lines that we never need to repeat. The
output feeds a merged emotion vocabulary for Brandon to prune, per-mood rankings for his
listening review, and in the end the per-mood control vectors for the TTS and the chat model.

Budget: $20 of API credit, everything included. Only text goes out (lines, conversation,
scene data, voice measurements written as words). Audio never leaves home.

## 1. What each line gets

A **complete checklist**: a 0 to 3 score for every mood on a fixed list, on every line.
Control vectors need clean positives *and* trustworthy zeros, and an open word list can't
give zeros: a missing word could mean "not there" or "not mentioned". Every mood that might
become a vector must have a complete score, or adding it later means paying again.

**Other feelings**: free words for anything the checklist misses or names too broadly. These
catch what we didn't ask for. They seed later moods without a rerun (section 7).

The checklist is built in two steps:
1. The 20 moods in `characters/tali/moods.toml`, plus seven the old list had no room for:
   afraid, determined, regretful, lonely, guilty, hopeful, irritated. That's 27.
2. A $0.40 probe on 30 non-gold lines (section 5) collects free words. About 10 of the most
   useful (candidates: homesick or longing, resigned, defiant, frustrated, nervous, hurt,
   protective, impressed, disgusted, playful) are added, **after Brandon okays the list**.
   Then the list is frozen at about 37 moods.

The checklist has boundaries for the overlapping pairs (angry vs irritated, worried vs
afraid, flustered vs embarrassed, sad vs lonely vs regretful). `informational` is redefined
as "plain delivery, no particular feeling", and `tired` lives only in `body`.

A gold mapping table is written before any paid call. For example, gold angry ← the higher of
angry and irritated.

## 2. What goes in

Per line, as in the current PC prompt (`corpus/tali/analysis/annotate/digest.py`):
- `<where>`: game, scene tags, conversation, listener.
- `<before>`: up to three lines before hers, including Shepard's choice.
- `<line>`: her line plus the director's note. `TEMP:` notes are flagged as placeholders.
- `<replies>`: what follows, with alternatives marked "only one of these plays".
- `<voice>`: pace, pitch, pitch movement, loudness and loudness variation in words, against
  her lines **from the same game** (LE1 is mastered darker).

System prompt, frozen and cached for an hour, about 1,600 tokens:
1. Purpose: these labels pick her real recordings that teach a voice and chat model each
   feeling, and Brandon checks the picks by ear. Mark what is in this line now, not what is
   typical for her.
2. A character brief of about 300 words: her arc per game, the suit, her speech habits.
3. What each input section is.
4. Each output field, defined once.
5. The checklist moods with one-line definitions and the boundaries.
6. Scale: 0 = not there, or too faint to notice; 1 = a faint undertone; 2 = clearly present;
   3 = a main feeling (two moods can both be 3).
7. Rules: score every mood; a 0 is a real claim and should be used freely; other feelings
   only for what the checklist misses; when a line could belong to two branches, judge the
   most likely one.

Left out: the speech-emotion classifier readings (wrong on two of three rendered examples,
and anchoring) and few-shot examples from the gold lines (they teach the old list and carry
known majority and recency biases). The code computes a `context_thin` flag (no lines before,
no scene) instead of asking the model.

## 3. What comes out (structured outputs; all keys required)

| Field | Type | Meaning |
|---|---|---|
| `reading` | text, about 15 words, first | what is going on for her in this moment; a short read before scoring |
| `doing` | text, at most 8 words | what she does with the line: teasing Garrus, reporting a position |
| `checklist` | object, about 37 keys, each 0 to 3 | as above |
| `other_feelings` | list of `{word, strength 1 to 3}` | only what the checklist misses; often empty |
| `stated` | stated, implied, none | whether her strongest feeling is said outright or only implied |
| `opposite_meaning` | true, false | sarcasm, irony, forced cheer |
| `energy` | 1 to 5 | 1 subdued, 2 calm, 3 normal talk, 4 animated or raised, 5 urgent or shouting; covers loud excitement, not only danger |
| `intoxication` | 0 to 3 | sober, tipsy, drunk, very drunk |
| `body` | list from tired, sick, injured, out_of_breath, hungover | physical state |

Dropped since v1: the per-emotion reasons and `basis` (the model can't hear her, so "rests on
delivery" was guesswork), `confidence` (verbalised confidence is poorly calibrated; agreement
between two samples is used instead), and `delivery_matches_words` (replaced by
`opposite_meaning`). About 260 output tokens per line. Enum integers keep values on the scale.

## 4. Models and shape

One line per request through the batch API (half price), keyed by line id. Grouping several
lines per request is dropped: labels smear across a scene, the schema can't enforce how many
answers come back, one failure loses several lines, and with caching it saves little.

Configurations the pilot compares:
- **S0**: Sonnet 5.5, thinking off (`between_tools`), effort low. About $5.5 for all lines with
  the cache, $10 without.
- **S1**: Sonnet 5.5, thinking on, effort low. About $10 to $15.
- **O1**: Opus 5.5, effort low (thinking can't be turned off on Opus). About $15 to $20 with
  the cache, $24 to $29 without. It fits only if caching works and its thinking stays short.
- **H0**: Haiku 5.5, thinking off. About $0.50. Only as a cheap second opinion.

These figures assume a 1,600-token cached prefix, a 320-token line and 260 output tokens.
Prices are re-checked the day of the run.

**Two samples per line where money allows**, the second with the checklist in reverse
order, so order effects average out. Scores are averaged, and disagreement between the two
samples marks uncertain lines.

## 5. Pilot

**P0 (free):** the PC session renders three prompts for Brandon and counts real input tokens
(`count_tokens` is free). The gold mapping table and the decision rule below are committed
before any paid call.

**P1, cost and vocabulary probe (about $0.40):** 30 random non-gold lines through S0, S1, O1
and H0. It measures real tokens and thinking per configuration, parse failures and refusals,
and collects free words for the extra checklist moods. Bugs get fixed here and only here.
Then Brandon okays the extra moods and **the prompt is frozen** (hash recorded).

**P2, quality (about $3):** all 150 gold lines through every configuration whose measured
projection fits, plus S0 again in reverse order. The winner's labels for those 150 lines
are kept as their final labels, so nothing is bought twice.

Measures:
1. **Recall at k**: per line, do the model's top k moods (k = the number Brandon marked),
   mapped through the gold table, contain his? Scattering 1s can't win this.
2. **Paired bootstrap** over lines for each arm against S0: a 95% interval on the
   difference. With 150 lines, a real gap of about 7 points shows; smaller ones don't.
3. **Agreed-extra rate**: per arm, 25 random cases where it scored a mood 3 and Brandon
   didn't mark it. Pooled and shuffled, arm hidden; he answers "is this feeling in the line,
   in words or voice?" while the clip plays. About 100 clicks.
4. **Inflation**: how many moods per line score 2 or more, and how many lines have no zeros
   among the original 20.
5. **Self-agreement**: S0 against S0 reversed. This shows the order effect and the noise.
6. **Energy** against measured loudness and pace, and within one step of his rating;
   **intoxication** exact.

**Decision rule (fixed before results):**
1. An arm qualifies only if its measured cost for all lines, plus 20%, fits the remaining
   credit minus a $1.50 reserve.
2. The default is S0.
3. S1 or O1 replaces it only if all of these hold: its recall at k beats S0 with the
   bootstrap interval clear of zero; its agreed-extra rate is no more than 10 points lower;
   and its inflation is at most 1.3 times S0's. If several qualify, the cheapest wins.
4. Second sample: if S0 agrees with itself poorly (median agreement across moods below 0.6),
   a full second sample is the first use of the remaining money. Otherwise the second sample
   covers only lines with some mood at 2 or more. H0 may serve as the second sample if its
   recall at k is within 5 points of S0's.
5. If nothing qualifies, stop and bring Brandon the options.

Rare moods can't be judged on 150 random lines. They are judged by how often Brandon
confirms the nominated lines in his listening review, which is recorded as the final test.

## 6. The full run, done so it never needs repeating

- Brandon sees the measured projection and okays this specific run.
- Lines are shuffled across chunks, so games and scenes don't line up with time.
- The first chunk is 300 lines. It stops if: cost per line is more than 20% over the
  projection; parse failures or truncations top 1%; refusals top 1%; the cache isn't being
  read (then re-project); or any mood's share of 3s drifts beyond half or double its pilot value.
- A running spend counter from the usage numbers stops the run at the budget minus reserve.
- Every request and raw response is stored with the model id, schema version, prompt hash,
  checklist order and usage. Failures are retried once on the same model; anything still
  failing is listed. No silent switch to another model.

## 7. After the run (free or pennies)

1. **Vocabulary:** checklist moods plus other-feeling words, embedded locally and clustered.
   One small call over the word list alone (no game text) proposes families. Brandon prunes.
2. **Score matrix:** lines × moods, averaged over samples. A mood that exists only in free
   words is grown from the lines Brandon confirms by nearest-neighbour search on the text
   embeddings we already have, with no paid rerun.
3. **Rankings per mood:** by score, preferring lines where the mood stands out (fewer other
   3s), keeping some implied lines, and spread across games and scenes so the LE3 party
   doesn't fill amused and teasing.
4. **Sets for vectors:** positives at 3 (or 2.5 averaged), confirmed by ear. Negatives at 0
   on that mood, matched on game, measured energy and length, ideally from the same session.
   A neutral baseline from lines with every mood at 1 or below. When vectors are fitted
   together, measured energy is factored out so a mood vector doesn't absorb the energy knob.

## 8. Budget

P1 $0.40, P2 about $3, the full run on S0 $5.5 to $10 (less the 150 lines already done),
a second sample on all or part of the lines $3 to $6, and the vocabulary call $0.10. That is
about $12 to $19, within $20 with a reserve. Opus wins only if it earns it under the rule and
the probe shows it fits.

## 9. What Brandon is asked to decide

1. Approve this plan. The PC session then rebuilds digest.py and annotate.py to match and
   renders three prompts (P0, free).
2. Approve P1 ($0.40) after reading the renders, and later the extra moods it suggests.
3. Approve P2 (about $3), and finally the full run once its measured cost is known.

## Changes from v1, and why

- The pilot moves from 40 gold lines to all 150 (40 lines gives about ±9 points, too noisy
  for a 5-point rule). It uses recall at k and a bootstrap instead of "any score of 1".
- The checklist grows from a probe before it's frozen, so new moods don't need a second run.
- Sonnet without thinking becomes the default; v1 wrongly assumed thinking can't be turned
  off. Haiku is added as a cheap second opinion.
- The grouped arm, the few-shot arm and the Opus-medium arm are dropped.
- Two samples in different orders replace the model's own confidence score.
- Labels are shaped for control vectors: real zeros, matched negatives, energy factored out.
- Fixed: moods.toml has 20 moods, not 21; `tired` was listed twice.
