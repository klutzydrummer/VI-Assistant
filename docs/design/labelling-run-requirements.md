# Labelling run: Brandon's requirements

Every version of `labelling-run-plan.md`, and every reviewer's suggestion, is checked against
this list before it's shown to him. Each item records what he asked for and how plan v2
meets it. When he states a new concern, it's added here first.

| # | Requirement (Brandon, 2026-10-18) | How plan v2 meets it |
|---|---|---|
| 1 | One paid run, done right, with results he's satisfied with | Free renders, then a $0.40 probe, then the pilot on gold; prompt frozen before the pilot; chunked full run with stop rules; all raw responses stored |
| 2 | Don't over-constrain the output, and don't under-constrain it into useless results | Complete checklist (rankable, real zeros) plus free `other_feelings` for anything missed; the checklist grows from the probe, with his ok, before it's frozen |
| 3 | Gold labels were limited by the old list; that limit must be respected | Gold scores only the old 20 moods; a new mood can never count against the model; gold `informational` is excluded (redefined); new moods are judged by his blind review and listening |
| 4 | Unconstrained emotions, plus energy, drunk, and hungover as an option | `other_feelings` free words; energy 1-5; intoxication 0-3; hungover in `body` |
| 5 | Each line shown with all available data digested into context | Scene, listener, lines before and after, director's note, branch alternatives, prosody in words, character brief. Not the whole conversation (its branches contradict), and not the classifier readings (wrong for her) |
| 6 | Clear prompt, no confusing or contradictory wording ("drunk sober") | Each field defined once with one name; no examples that contradict the instructions; no "don't trust this" inputs |
| 7 | Claude API; pilot Opus 5.5 against a cheaper, widely trusted model; not Fable | Opus 5.5 (O1) against Sonnet 5.5 (S0, S1), plus Haiku 5.5 as a cheap second opinion |
| 8 | Only text goes out; audio stays home | Prosody is sent as words; no audio, and no game text in the repo |
| 9 | $20 budget; no paid run without his ok for that run | Each of P1, P2 and the full run waits for his own go line; spend counter and reserve |
| 10 | Plan ahead; verify the whole setup before spending, never fix-and-rerun | P0 renders and free token counts first; bugs fixed only in the $0.40 probe |
| 11 | Output feeds the vocabulary, per-mood rankings for his ear, and control vectors for TTS and LLM | Section 7 of the plan: vocabulary merge, rankings spread across games, matched negatives, energy factored out |
| 12 | Lines for him to copy go in code blocks | Every go line is sent in a code block |
