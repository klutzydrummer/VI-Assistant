# Background watcher (phone pings)

`scripts/vi-bg-watch.py` lets a paid run go on without anyone watching. A timer runs it every
five minutes. Each run of the script is one pass: it looks at the run, sends a phone push through
ntfy when something needs you, and rewrites a small status file. It uses only the Python standard
library and costs nothing while no run is active.

## What buzzes your phone

| Event | When |
|---|---|
| `needs-approval` | an agent is waiting for a permission answer (`paseo permit ls --json` is not empty) |
| `listening-set-ready` | a new `<platter>/step-N/clips.toml` appeared (one push per step) |
| `run-failed` | the run ended `failed` or `interrupted`, or says `running` with no live process |
| `run-finished` | the run ended `done` |

Each event is sent once. A push holds only the run id, the event, the step and dollars. No corpus
text, and the topic is never printed or logged. An `interrupted` run (for example after
`paid_run.py stop`) also counts as `run-failed`, so you hear that the worker was released.

**Quiet hours, 22:00 to 08:00 local.** Every push is held, failures included. Held events go to
`pending.jsonl`; the first pass at or after 08:00 sends one digest push and clears it. If a send
fails (no network, bad topic file), the event stays unsent and the next pass tries again.

## Telling it which run to watch

Write the run id where the timer can read it, and clear it when the run is over:

```sh
mkdir -p ~/.cache/vi-bg && echo r32 > ~/.cache/vi-bg/run-id    # start watching
rm ~/.cache/vi-bg/run-id                                       # stop watching
```

With no run id the pass exits at once and does no network. Pass `--platter DIR` (the folder with
the `step-N` listening sets) in the service line below, or the listening events stay off.

## Files (in `~/.cache/vi-bg/`)

- `status.json`: run state, spend against the cap, ETA, what is waiting on you, last pass time.
- `state.json`: events already handled. `pending.jsonl`: events held for the morning digest.
- `outbox.jsonl`: only with `--dry-run`, the pushes that would have been sent.

## Trying it without sending anything

```sh
python3 -I scripts/vi-bg-watch.py --run-id r32 --platter ~/platter --dry-run
cat ~/.cache/vi-bg/outbox.jsonl
```

Use `--now 2026-11-26T23:30:00` to test quiet hours, and `--status-cmd` / `--agents-cmd` to feed
fake JSON. The tests are `jobs/test_bg_watch.py`.

## NixOS home-manager timer and service

This is not applied for you; add it to the home-manager config (nothing under `/etc/nixos` was
edited). Change the repo path and platter to match your checkout.

```nix
systemd.user.services.vi-bg-watch = {
  Unit.Description = "VI background watcher (one pass)";
  Service = {
    Type = "oneshot";
    Nice = 19;
    WorkingDirectory = "/home/brandon/Projects/VI-Assistant";
    ExecStart = "${pkgs.python3}/bin/python3 -I scripts/vi-bg-watch.py --platter /home/brandon/platter";
    # `paseo` must be on PATH for the approval check.
    Environment = "PATH=/run/current-system/sw/bin:/home/brandon/.nix-profile/bin";
  };
};
systemd.user.timers.vi-bg-watch = {
  Unit.Description = "VI background watcher, every 5 minutes";
  Timer = { OnBootSec = "2min"; OnUnitActiveSec = "5min"; };
  Install.WantedBy = [ "timers.target" ];
};
```

## Not yet verified

- The fields of a non-empty `paseo permit ls --json` list. The script accepts `requestId`, `id`
  or `agentId`; check against a real pending request.
- That the ntfy phone app is subscribed to the topic, and that a real push arrives.
