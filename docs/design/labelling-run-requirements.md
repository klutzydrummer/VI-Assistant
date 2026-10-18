# Labelling run: Brandon's requirements

Every version of `labelling-run-plan.md`, and every reviewer's suggestion, is checked against
this list before it's shown to him. When he states a new concern, it's added here first.
Goals for the whole project: `/mnt/project-files/design/what-brandon-wants.md`.

| # | Requirement (Brandon) | How plan v3 meets it |
|---|---|---|
| 1 | One paid labelling effort, done right, with results he's satisfied with (10-18) | Free renders, a $1 pilot, then pass 1 and pass 2, each run once; prompts frozen after the pilot; chunked runs with stop rules; all raw responses stored |
| 2 | Don't over-constrain the output, and don't under-constrain it into useless results (10-18) | Pass 1 is fully open; pass 2 scores every line on the moods pass 1 found, plus free words for anything left |
| 3 | The gold labels and their old list must not affect the model at all (10-18 23:55; replaces "treat gold as a floor") | Not in the prompts, not in the scoring, not in the model choice |
| 4 | Claude is the main source of mood labels (10-18 23:55) | Both passes are Claude; the vocabulary is Claude's own families |
| 5 | Unconstrained emotions, plus energy, drunk, and hungover as an option (10-18) | Pass 1 free words; pass 2 energy 1-5, intoxication 0-3, hungover in `body` |
| 6 | Each line shown with all available data digested into context (10-18) | Scene, listener, lines before and after, director's note, branch alternatives, prosody in words, character brief. Not the whole conversation (its branches contradict), and not the classifier readings (wrong for her) |
| 7 | Clear prompt, no confusing or contradictory wording ("drunk sober") (10-18) | Each field defined once with one name; no examples; no "don't trust this" inputs |
| 8 | Claude API; pilot Opus 5.5 against a cheaper, widely trusted model; not Fable (10-18) | The pilot compares Opus 5.5 with Sonnet 5.5, with thinking off and on |
| 9 | Only text goes out; audio stays home (10-18) | Prosody is sent as words; no audio, and no game text in the repo |
| 10 | $20 Claude credit, separate from Vast credit; no paid run without his ok for that run (10-18) | The pilot, pass 1 and pass 2 each wait for his own go line; a spend counter stops at $20 minus the reserve |
| 11 | Plan ahead; verify the whole setup before spending, never fix-and-rerun (10-18) | Free renders and token counts first; bugs fixed only in the pilot |
| 12 | Reduce reliance on his ears; context-enriched text should be enough (10-18 21:57) | Trusted labels come from the two passes agreeing; his only checks are optional and reading-based |
| 13 | Cover the moods that matter for friend and assistant chat (10-17) | Any must-have mood missing from Claude's vocabulary is added to pass 2 |
| 14 | Output feeds the vocabulary, per-mood rankings, and control vectors for voice and chat (10-13, 10-17) | Plan sections 7 and 8: families, rankings spread across games, trusted positives against matched zeros, energy factored out |
| 15 | Lines for him to copy go in code blocks (10-17) | Every go line is sent in a code block |
