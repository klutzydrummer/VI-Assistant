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

## Tooling
These commands are checked to work at home. Use them as written; do not hunt for other tools or
install anything globally. Run them from the repo root.
- **Rust.** `cargo`, `rustc`, `rustfmt` and `clippy` are not on the home PATH. Wrap each command in
  `nix-shell`, and set `CARGO_TARGET_DIR` outside the repo:

      export CARGO_TARGET_DIR=~/.cache/mm-tmp/<task>/target
      nice -n 19 nix-shell -p cargo rustc rustfmt clippy gcc pkg-config openssl cmake clang llvmPackages.libclang --run 'export LIBCLANG_PATH=$(dirname $(find $(nix-build --no-out-link "<nixpkgs>" -A llvmPackages.libclang.lib) -name "libclang.so*" | head -1)); cargo fmt --all --check; cargo clippy --workspace --all-targets -- -D warnings; cargo test --workspace'

  A cold clippy run takes about 5 minutes. For a Vulkan build, use the recipe in the README.
- **Python tests.** System `python3` has no numpy, scipy, scikit-learn or soundfile (the list in
  `jobs/requirements-home.txt`), so the suite shows false import failures without this:

      nice -n 19 nix-shell -p 'python3.withPackages (p: [p.numpy p.scipy p.scikit-learn p.soundfile])' --run 'python3 -m unittest discover -s jobs'

  `python3 jobs/run_tests.py` is the faster parallel version, run inside the same shell.
- **Not needed:** ffmpeg, sox and shellcheck. A pydub warning about ffmpeg is harmless, and the
  worker installs sox itself.
- **Missing tool?** Tell your parent agent or middle manager, who files a host-change request.
  Do not work around it.
- **Sandbox guest image.** It has `cargo`, `rustc` and `python3` but no `nix`, `rustfmt`, `clippy` or
  numpy. Run fmt, clippy and the Python suite at home until the host change lands.
