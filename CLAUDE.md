# VI Assistant — project rules

These rules are automatic for every agent working on this project. They take precedence over any
older project doc that describes a conflicting workflow; treat such docs as stale and do not follow
them, though updating them stays allowed when a future task explicitly scopes it.

## Spending and rentals
- Never start a paid run, rental, or GPU job without the user's explicit approval for that specific run
  and budget. Approval for one run does not extend to the next.
- Before renting, plan the environment and runtime, and run a small smoke test first. Stay within
  the approved cap and scope.
- Never auto-replace a failed rental, and never touch another project's instances or processes.
- Report failures and spend honestly — do not soften or hide either.

## Ephemeral workers
- Home initiates the connection: it opens a reverse SSH tunnel and a restricted rsync receiver. The
  worker must push a random token successfully before inputs or training start.
- Outputs and checkpoints are pushed home. The worker never gets home's private keys, agents,
  credentials, or any long-lived identity.
- Listeners bind to loopback only. The receiver must accept only the exact output and checkpoint
  directories, be write-only, never delete, stay hidden, allow at most one connection, and munge symlinks.
- Stop the receiver and the tunnel afterwards.
- To diagnose a worker, use the installed Vast.ai CLI; do not read its credential, config, or state
  files.

## Resources at home
- Measure RAM and VRAM before running a home job. The home RX 480 has 8 GB VRAM.
- Run heavy jobs under `nice -n 19`.
- Use `ionice -c3` only for game-disk reads, and `ionice -c2 -n7` for system-disk work.
- No GPU jobs in CI. Use local, disposable integration tests instead.

## Data
- Learn from raw data, as unsupervised as possible. Never use legacy hand-labelled mood data for
  filtering, weighting, prompting, scoring, or selecting data or models.
- Do not remove clips merely because they appeared in an old labelled set; keep all suitable audio
  in the unsupervised pool.

## Voice and copyright
- Copyrighted game audio and text move only directly between home and the rented worker — never
  through third-party dataset hosts or into the public repo.
- Every voice-producing run creates a small, clearly labelled listening set, and human listening —
  not an automatic score — decides voice quality.

## Communication
- Write in plain English. For a genuine fork, state the options and your recommendation.
- Confirm odd approvals before acting. Report failed, skipped, or uncertain steps faithfully.
