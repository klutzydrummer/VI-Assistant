"""Every named time and retry limit of the paid run. Values only: env overrides stay at their
call sites."""


PROGRESS_SECONDS = 30.0
STOP_WAIT_SECONDS = 600.0


# ---- retries, timeouts and recovery bounds (every wait goes through nap(), so tests never wait) ----
OUTER_RETRY_SECONDS = 20          # F-NET-1: one more try of `offers` / `status` after this long
RELEASE_BACKOFF = (10, 20, 40, 80)  # F-REL-1: waits between `down --force` tries before exit 9
QUERY_TIMEOUT_SECONDS = 180       # F-CLK-1: offers / status
DOWN_TIMEOUT_SECONDS = 300        # a hung `down` would bill forever
DOWN_VERIFY_TIMEOUT_SECONDS = 600  # r32: `down --verify` = DELETE retries + up to 120 s of listing
VERIFY_EPOCH_TIMEOUT_SECONDS = 300  # r32 WP7: the debug pass's home verify-epoch
PREFLIGHT_TIMEOUT_SECONDS = 90 * 60


FATAL_QUIET_SECONDS = 180
UP_EXTRA_SECONDS = 5 * 60         # `up` may take boot_minutes plus this
DEFAULT_BOOT_MINUTES = 15
INPLACE_MAX_RESTARTS = 2          # F-RES-1 / F-TRN-2: in-place train restarts on the same worker
INPLACE_MAX_SECONDS = 45 * 60     # ... and the window, from the first failure, they must fit in
REATTACH_MAX = 3                  # F-PUSH-5: stage died but the trainer lives; reattach this often
REATTACH_WAIT_SECONDS = 15


HF_TRIES = 3
HF_TIMEOUT_SECONDS = 60
HF_BACKOFF = (5, 15)


# r32 WP2: the home dead-man timer (L2) that `vi cloud up` arms at the hard cap. While this run
# is alive and still judging the rental (VALID or SUSPECT), it moves the timer to now + the grace,
# at most once per heartbeat; `vi cloud` never moves it past the hard cap. A crashed or frozen run
# stops moving it, so the instance is destroyed (and verified) at most this long after.
DEADMAN_HEARTBEAT_SECONDS = 60
DEADMAN_STALL_GRACE_SECONDS = 30 * 60
DEADMAN_EXTEND_TIMEOUT_SECONDS = 30


# A1: host-side evidence from the Vast CLI before a failed worker is destroyed. Each call is cut
# off after this long; VI_VASTAI_BIN overrides the binary (tests use a stub). The CLI reads its
# own key; this script never reads or prints the key or the CLI's config/state files.
VAST_EVIDENCE_SECONDS = 30.0
# r21: a host drop needs two Vast status reads this far apart that both say not running.
HOST_DROP_RECHECK_SECONDS = 60


DIAG_TIMEOUT_SECONDS = 25.0


DEBUG_PROBE_SECONDS = 180
# The worker stage is cut off this long before the run's hours end, so `down` still fits.
DEBUG_RELEASE_MARGIN_SECONDS = 10 * 60
# r21: on-worker.sh gives up on an unreachable worker after this many minutes in the debug pass.
DEBUG_LOST_MINUTES = 5


PUSH_PROBE_MAX_MINUTES = 5  # VI_PUSH_PROBE_SECONDS' default in jobs/push-probe.sh, 300 s


REALPATH_TIMEOUT = {False: 900, True: 1800}
