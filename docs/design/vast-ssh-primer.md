# Vast.ai SSH and CLI primer

Research date: 2026-10-31. Covers routine account checks and diagnosing a rented instance that is "running" but accepts no SSH. Not a replacement for `vi cloud up`; see `docs/cloud-worker.md`.

## Identity and balance

`vastai show user` shows auth status and remaining credit without printing key material. Avoid `vastai show api-key` and `vastai show api-keys` for routine checks — they print secrets.

## Keys before you rent

- `vastai create ssh-key ~/.ssh/id_ed25519.pub` registers your public key on the account. Do this **before** provisioning.
- Account keys are applied only to **new** instances. They do not attach to instances already running. Adding a key later does not reach a machine that is up; changing a VM's keys requires recreating the instance.
- Per-instance commands `vastai attach ssh ID PUBLIC_KEY` and `vastai detach ssh ID KEY_ID` exist, but they are documented commands to know about, not something to run without approval — prefer recreating the instance.

## Inspecting instances

- `vastai show ssh-keys` — keys registered on the account.
- `vastai show instances --status running` — running instances; add `--raw` for JSON, or `--cols id,status,gpu,dph` to trim the table.
- `vastai show instance ID` — one instance's detail.
- `vastai ssh-url ID` — prints the connection info (host, port, user, command) rather than opening a session.
- `vastai scp-url ID` — the same, for scp/sftp.
- `vastai logs ID --tail 100` — the last 100 lines of the instance log.

## "Running" does not mean sshd is ready

An instance can report running while SSH still refuses connections. Work through these, cheapest first:

1. Confirm your key is registered: `vastai show ssh-keys`.
2. Confirm your private key permissions: `chmod 600 ~/.ssh/id_ed25519`, and that the agent holds it: `ssh-add -l`.
3. Try the direct connection with verbose output: `ssh -vv -p PORT root@IP`. The verbose log shows whether the key is offered and accepted.
4. Check whether you are reaching the host directly or through a proxy/relay; the proxy path can behave differently. `ssh-url` tells you which.
5. Remember the flag case: `ssh` uses lowercase `-p PORT`, while `scp`/`sftp` use uppercase `-P PORT`. Each tool requires its correct case.

## How this fits the automatic flow

Verify the account key before renting. After boot, home uses the direct connection shown by `ssh-url` to reach the worker. Then home creates the reverse SSH tunnel and the restricted, temporary rsync receiver, and the worker must push a random token through that tunnel that home reads back exactly before any training starts. This primer only covers getting SSH working; it does not change that return design.

## Checklist for the recent failure

One instance reported running for 15 minutes without accepting SSH and was returned. No evidence yet points to a specific cause — key injection, networking, or slow startup are all still open. On the next attempt:

- [ ] Key registered before rental (`vastai show ssh-keys`).
- [ ] Private key is `chmod 600` and in the agent (`ssh-add -l`).
- [ ] `ssh -vv -p PORT root@IP` captured for the failure.
- [ ] `vastai logs ID --tail 100` captured.
- [ ] Direct vs proxy path noted.
- [ ] Waited the configured `boot_minutes` (15) before giving up.

## Open questions

- No supported way found yet to list the keys attached to an **existing** instance.
- No evidence of a separate machine-owner "host SSH key" concept in the fetched docs.

## Sources

1. <https://raw.githubusercontent.com/vast-ai/vast-cli/c26d8a8d/vastai/SKILL.md>
2. <https://docs.vast.ai/guides/reference/troubleshooting>
3. <https://docs.vast.ai/guides/instances/connect/ssh>
4. <https://cloud.vast.ai/manage-keys/> (linked by the SSH guides, not independently fetched)
