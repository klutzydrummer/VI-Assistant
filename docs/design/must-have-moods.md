# Must-have moods for Tali as friend and assistant (draft for Brandon to cut)

Status: draft, 2026-10-17. Brandon's calls so far: one flat list, a line may carry several
labels, vectors fitted together (so a mix adds up as expected), plus two degree knobs
(energy, drunk). Sources: the everyday-conversation sets EmpatheticDialogues (32 emotions) and
GoEmotions (27), the gaps we measured in her corpus, and the six classes her clusters found.

Cut this to a core of about 12 to 15. Mark each line **keep**, **cut** or **merge into X**.

## Already found by her clusters (free)

| Mood | Why it matters | Note |
|---|---|---|
| warm | the default friendly register | class c8 |
| sad | sympathy, bad news, her own grief | class c5, unstable (J 0.58) |
| anger / irritation | pushback, annoyance at a bug or a person | class c3 |
| surprise | reacting to news, "wait, what?" | class c10, small (58 clips) |
| calm focus | explaining, working through a problem | class c4; may really be low energy |
| urgency | warnings, time pressure | class c2; may really be high energy |

## Needed, not found yet

| Mood | Why a friend or assistant needs it | Evidence in her lines |
|---|---|---|
| amused | laughing along, enjoying a joke | EmoNet amusement lines exist |
| teasing | playful jabs, banter | known in ME2/ME3 squad talk |
| affection / tender | closeness, romance, comfort | romance scenes |
| grateful | thanking you | "thank" in 74 lines |
| apologetic | owning a mistake | "sorry" in 41 lines |
| flustered / embarrassed | caught off guard, rambling | "um/uh/I mean" in 283 lines |
| curious | asking, wanting to know more | 448 questions |
| reassuring | "it's okay, we've got this" | no model scores it directly |
| worried / concerned | checking on you, unease | EmoNet concern lines exist |
| excited | good news, tech she loves | her tech enthusiasm |
| proud | pride in you or in her people | Quarian loyalty lines |
| relieved | after danger passes | no clear signal yet |
| doubtful / confused | "are you sure?" | question lines |
| sarcastic / deadpan | dry humour | hardest to detect; text vs voice mismatch |
| tired | late night, after a long day | weary c7 lines were mixed |
| informational (neutral) | plain answers with no colour | most assistant replies |

## Knobs (degrees, not moods)

| Knob | Range | Evidence |
|---|---|---|
| energy | calm to urgent | urgency and calm focus may fold into this |
| drunk | sober to tipsy to drunk | Checked 2026-10-17: 132 lines tagged citadel_party, not all drunk; clearly drunk in citpar_tali3, citpar_tali_a, citpar_bb3 ("Omni-tooooooooooo!"), plus the drinking-alone toast to Miranda (norhen_tali_mr_cermir_d). Estimate 40 to 80 truly drunk lines, so likely thin. EIV's intoxication score agrees on the top clips but is too weak to find them alone (+0.43 SD), so scene tags and your listening do the finding. Risks: all ME3, nearly one recording session, and party lines overlap the warm class (c8 is 3.6x enriched for the party), so drunk may come out partly "happy at a party". Slurred speech is distinctive, so the split-half test may still pass. The same knob could later take tired or sick if supported. |

## What happens to each after you cut

Each kept mood gets one of four build routes, decided by her data, not by us:
- **voice and chat**: enough lines, your listening confirms them, and they sound consistent.
- **chat only**: she says it but has no consistent sound; the voice borrows the nearest class or a mix.
- **thin**: real but rare (under about 30 lines); built as a blend or at lower strength, and flagged.
- **missing**: she never does it; persona prompt only, no vector.

Counting candidate lines per mood is cheap CPU work at home and comes next, after your cut.

## First counts from her lines (2026-10-17, CPU only)

From subtitle keywords, scene tags, the audio emotion model's top 10%, the discovered classes and
text-neighbour expansion; "strong" = two or more signals agree. Floors for subtle moods (keywords
miss phrasings) with some false hits. Tiers: 100+ voice-sized, 30 to 99 thin, under 30 rare.
Details: corpus/tali/analysis/mood_counts/mood_counts.json on Brandon's machine.

| Mood | Strong lines (LE1/LE2/LE3) | Tier |
|---|---|---|
| angry | 206 (13/78/115) | voice-sized |
| sad | 205 (28/82/95) | voice-sized |
| informational | 420 | voice-sized |
| curious / asking | 286 asking, 57 curious | voice-sized / thin |
| urgent (warning) | 289 | voice-sized |
| amused | 96 (5/1/90), mostly the Citadel party | thin |
| warm / affectionate | 94, overlaps the warm class | thin |
| teasing | 94 (5/1/88), mostly the party | thin |
| grateful | 81 | thin |
| surprised / doubtful | 67 | thin |
| apologetic | 37 | thin |
| drunk | 39, all LE3 party and the Miranda toast | thin |
| flustered | 28 | rare by keywords |
| sick | 25, mostly LE2 | rare |
| reassuring | 22 | rare by keywords |
| worried | 19 | rare by keywords |
| excited | 9 | rare by keywords |
| sarcastic | 8 | rare by keywords |
| tired | 8 | rare |
| relieved, proud | 6 each | rare by keywords |
| energy | about 1,170 per third | plenty |

Session risk: amused, teasing and drunk come mostly from one LE3 event, so a vector built from
them may partly mean "party in LE3".
