"""Stub-driven shell tests for `voicetune-fresh.sh preflight`: steps 0a/0b (tokenise, replay) and
the kill/resume rehearsal. The script is copied beside stub trainer/replay/tokenise programs, so
no GPU, torch or omnivoice is needed. The stubs log every call to $CALLS."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent

TRAIN_STUB = r'''
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--output-dir"); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--save-steps", type=int, default=500); ap.add_argument("--resume-from-checkpoint")
ap.add_argument("--verify-checkpoint", nargs="+")
a, _ = ap.parse_known_args()
open(os.environ["CALLS"], "a").write("train_ft " + " ".join(sys.argv[1:]) + "\n")

def write_ck(step, manifest=True):
    d = os.path.join(a.output_dir, f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    data = b"weights-%d" % step
    open(os.path.join(d, "model.safetensors"), "wb").write(data)
    if manifest:
        open(os.path.join(d, "SHA256SUMS"), "w").write(f"{hashlib.sha256(data).hexdigest()}  model.safetensors\n")

if a.verify_checkpoint:
    bad = 0
    for d in a.verify_checkpoint:
        sums = os.path.join(d, "SHA256SUMS")
        if not os.path.isfile(sums):
            print(f"verify: {d}: no SHA256SUMS", file=sys.stderr); bad = 1
    sys.exit(bad)

os.makedirs(a.output_dir, exist_ok=True)
start = 0
if a.resume_from_checkpoint:
    start = int(a.resume_from_checkpoint.rsplit("-", 1)[1])
delay = 0.0 if os.environ.get("STUB_FAST") or a.resume_from_checkpoint else 0.15
for step in range(start + 1, a.steps + 1):
    with open(os.path.join(a.output_dir, "style_metrics.jsonl"), "a") as f:
        f.write(json.dumps({"micro_step": step, "attempt": 0}) + "\n")
    time.sleep(delay)
    if step % a.save_steps == 0 or step == a.steps:
        corrupt = a.resume_from_checkpoint and os.environ.get("STUB_RESUME_NO_MANIFEST") and step == 20
        write_ck(step, manifest=not corrupt)
'''

TOKENISE_STUB = r'''
import argparse, os, sys
ap = argparse.ArgumentParser()
for k in ("input_jsonl", "tar_output_pattern", "jsonl_output_pattern", "tokenizer_path",
          "samples_per_shard"):
    ap.add_argument("--" + k)
ap.add_argument("--skip_errors", action="store_true")
a = ap.parse_args()
open(os.environ["CALLS"], "a").write(f"tokenise {a.input_jsonl} tokenizer={a.tokenizer_path}\n")
base = os.path.dirname(os.path.dirname(a.tar_output_pattern))
os.makedirs(os.path.join(base, "audios"), exist_ok=True)
open(os.path.join(base, "data.lst"), "w").write("shard-000000.tar shard-000000.jsonl 1 1.0\n")
rows = sum(1 for l in open(a.input_jsonl) if l.strip())
ok, bad, skipped = os.environ.get("STUB_TOK_OK"), os.environ.get("STUB_TOK_FAILED", "0"), os.environ.get("STUB_TOK_SKIPPED", "0")
print(f"Processing Complete - Successful: {ok if ok is not None else rows}, Failed: {bad}, Filtered/Skipped: {skipped}")
'''

REPLAY_STUB = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir"); ap.add_argument("--base-model"); ap.add_argument("--voices")
ap.add_argument("--texts"); ap.add_argument("--device")
a = ap.parse_args()
open(os.environ["CALLS"], "a").write(
    f"replay_public device={a.device} voices={a.voices} texts={a.texts} base={a.base_model}\n")
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "replay_public.jsonl"), "w") as f:
    f.write(json.dumps({"id": "replay-0000", "audio_path": "x.wav", "text": "t",
                        "language_id": "en", "replay": True}) + "\n")
'''

HUB_STUB = r'''
import os
def snapshot_download(repo, revision=None, local_dir=None):
    open(os.environ["CALLS"], "a").write(
        f"snapshot_download {repo} {revision} offline={os.environ.get('HF_HUB_OFFLINE')} "
        f"token={os.environ.get('HF_TOKEN')}\n")
    os.makedirs(os.path.join(local_dir, "audio_tokenizer"), exist_ok=True)
    open(os.path.join(local_dir, "config.json"), "w").write("{}")
    # Sparse files at the pinned sizes (fetch_base_model.SIZE_TABLE / TOTAL_BYTES): no disk used.
    for rel, size in (("model.safetensors", 2450344112),
                      ("audio_tokenizer/model.safetensors", 805665628), ("pad.bin", 11460530)):
        with open(os.path.join(local_dir, rel), "wb") as f:
            f.truncate(size)
    return local_dir
'''


TORCH_STUB = r'''
import os
class _Cuda:
    @staticmethod
    def synchronize():
        pass
    @staticmethod
    def is_bf16_supported():
        return os.environ.get("STUB_BF16", "1") == "1"
cuda = _Cuda()
class _T:
    def cuda(self):
        if os.environ.get("STUB_CUDA_FAIL"):
            raise RuntimeError(os.environ["STUB_CUDA_FAIL"])
        return self
def zeros(n):
    return _T()
'''


class PreflightHarness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self.root = Path(self._tmp.name)
        jobs = root / "jobs"
        (jobs / "omnivoice-ft").mkdir(parents=True)
        shutil.copy(JOBS / "voicetune-fresh.sh", jobs / "voicetune-fresh.sh")
        shutil.copy(JOBS / "requirements-omnivoice-worker.txt", jobs / "requirements-omnivoice-worker.txt")
        shutil.copy(JOBS / "omnivoice-ft" / "fetch_base_model.py", jobs / "omnivoice-ft" / "fetch_base_model.py")
        self._exe(jobs / "worker-return.sh", "exit 0")
        (jobs / "omnivoice-ft" / "train_ft.py").write_text(TRAIN_STUB)
        (jobs / "omnivoice-ft" / "replay_public.py").write_text(REPLAY_STUB)
        (jobs / "omnivoice-ft" / "infer_listening.py").write_text(
            "import os, sys\nopen(os.environ['CALLS'], 'a').write('infer_listening\\n')\n")
        pylib = root / "pylib" / "omnivoice" / "scripts"
        pylib.mkdir(parents=True)
        (root / "pylib" / "omnivoice" / "__init__.py").write_text("")
        (pylib / "__init__.py").write_text("")
        (pylib / "extract_audio_tokens.py").write_text(TOKENISE_STUB)
        (root / "pylib" / "huggingface_hub.py").write_text(HUB_STUB)
        (root / "pylib" / "torch.py").write_text(TORCH_STUB)
        bindir = root / "bin"
        bindir.mkdir()
        self._exe(bindir / "python", 'exec python3 "$@"')
        self._exe(bindir / "accelerate", "shift 3\nexec \"${OVFT_PYTHON:-python3}\" \"$@\"")
        self._exe(bindir / "nvidia-smi", 'echo "24576, 0"')
        self.data = root / "data"
        (self.data / "voices").mkdir(parents=True)
        (self.data / "audio_tokenizer").mkdir()
        (self.data / "audio_tokenizer" / "PLACEHOLDER").write_text("x")
        for name in ("train_manifest.jsonl", "dev_manifest.jsonl", "replay_texts.jsonl",
                     "style_manifest.jsonl", "data_config.json"):
            (self.data / name).write_text('{"id": "a", "text": "t"}\n')
        self.base = root / "base"
        (self.base / "audio_tokenizer").mkdir(parents=True)
        self.work = root / "work"
        (self.work / "out").mkdir(parents=True)
        self.calls = root / "calls.log"
        self.calls.touch()

    @staticmethod
    def _exe(path, body):
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(0o755)

    def env(self, **extra):
        env = dict(os.environ, CALLS=str(self.calls), OVFT_PYTHON=str(self.root / "bin" / "python"),
                   PATH=f"{self.root / 'bin'}:{os.environ['PATH']}",
                   PYTHONPATH=str(self.root / "pylib"), VI_OUT=str(self.work / "out"),
                   VI_RETURN_PORT="1", VI_INPUT_OMNIVOICE_DATA=str(self.data),
                   HOME=str(self.root / "home"),
                   VI_BASE_MODEL=str(self.base), VI_MIN_DISK_GB="0", VI_PUSH_SECONDS="0",
                   VI_REHEARSE_TIMEOUT="30")
        env.pop("VI_INPUT_VOICETUNE_DATA", None)
        env.update(extra)
        env = {k: v for k, v in env.items() if v is not None}
        return env

    def preflight(self, **extra):
        return subprocess.run(["bash", str(self.root / "jobs" / "voicetune-fresh.sh"), "preflight"],
                              env=self.env(**extra), capture_output=True, text=True, timeout=120)

    def log(self):
        return self.calls.read_text().splitlines()


class PrepareDataTest(PreflightHarness):
    def test_tokenise_and_replay_run_before_the_probe(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        log = self.log()
        kinds = [l.split()[0] for l in log]
        first_train = kinds.index("train_ft")
        self.assertEqual(kinds[:first_train],
                         ["tokenise", "tokenise", "replay_public", "tokenise"])
        self.assertIn(f"tokenise {self.data}/train_manifest.jsonl", log[0])
        self.assertIn(f"tokenise {self.data}/dev_manifest.jsonl", log[1])
        self.assertIn("device=cuda", log[2])
        self.assertIn(f"voices={self.data}/voices", log[2])
        self.assertIn(f"texts={self.data}/replay_texts.jsonl", log[2])
        self.assertIn(f"base={self.base}", log[2])
        self.assertIn(f"tokenise {self.data}/replay/replay_public.jsonl", log[3])
        for name in ("train", "dev", "replay"):
            self.assertTrue((self.data / "shards" / name / "data.lst").is_file(), name)

    def test_placeholder_tokenizer_falls_back_to_the_base_snapshot(self):
        self.preflight()
        for line in (l for l in self.log() if l.startswith("tokenise")):
            self.assertIn(f"tokenizer={self.base}/audio_tokenizer", line)

    def test_a_bundled_tokenizer_wins_over_the_base_snapshot(self):
        (self.data / "audio_tokenizer" / "config.json").write_text("{}")
        self.preflight()
        for line in (l for l in self.log() if l.startswith("tokenise")):
            self.assertIn(f"tokenizer={self.data}/audio_tokenizer", line)

    def test_env_tokenizer_wins_over_everything(self):
        self.preflight(VI_AUDIO_TOKENIZER="/somewhere/tok")
        for line in (l for l in self.log() if l.startswith("tokenise")):
            self.assertIn("tokenizer=/somewhere/tok", line)

    def test_no_tokenizer_anywhere_fails_before_any_gpu_work(self):
        (self.base / "audio_tokenizer").rmdir()
        r = self.preflight()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no audio tokenizer", r.stderr)
        self.assertEqual(self.log(), [])

    def test_existing_shards_are_kept_on_a_rerun(self):
        self.preflight()
        before = len([l for l in self.log() if l.split()[0] in ("tokenise", "replay_public")])
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        after = len([l for l in self.log() if l.split()[0] in ("tokenise", "replay_public")])
        self.assertEqual(before, after)
        self.assertIn("already built", r.stdout)

    def test_a_replay_run_that_writes_no_rows_fails_preflight(self):
        (self.root / "jobs" / "omnivoice-ft" / "replay_public.py").write_text(
            "import os\nopen(os.environ['CALLS'], 'a').write('replay_public\\n')\n")
        r = self.preflight()
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("train_ft", [l.split()[0] for l in self.log()])


class InputNameTest(PreflightHarness):
    def test_the_omnivoice_data_input_name_is_the_one_on_worker_exports(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_older_voicetune_data_name_still_works(self):
        r = self.preflight(VI_INPUT_OMNIVOICE_DATA=None, VI_INPUT_VOICETUNE_DATA=str(self.data))
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_no_data_input_fails_with_a_clear_message(self):
        r = self.preflight(VI_INPUT_OMNIVOICE_DATA=None)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("VI_INPUT_OMNIVOICE_DATA", r.stderr)
        self.assertEqual(self.log(), [])


class BaseModelFetchTest(PreflightHarness):
    def test_unset_base_model_is_downloaded_once_and_passed_on(self):
        r = self.preflight(VI_BASE_MODEL=None, HF_TOKEN="secret", HF_HUB_OFFLINE="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        dl = [l for l in self.log() if l.startswith("snapshot_download")]
        self.assertEqual(len(dl), 1, dl)
        self.assertEqual(dl[0], "snapshot_download k2-fsa/OmniVoice "
                         "c5fdb5ccb189668d56333f77ba2629f4cd7535f4 offline=None token=None")
        snap = self.root / "home" / ".cache" / "vi" / "base-model" / "k2-fsa--OmniVoice-c5fdb5ccb189"
        self.assertTrue((snap / "config.json").is_file())
        replay = next(l for l in self.log() if l.startswith("replay_public"))
        self.assertIn(f"base={snap}", replay)
        self.assertIn(f"--init-from {snap}", next(l for l in self.log() if l.startswith("train_ft")))
        self.preflight(VI_BASE_MODEL=None)  # a rerun reuses the folder
        self.assertEqual(len([l for l in self.log() if l.startswith("snapshot_download")]), 1)

    def test_a_given_base_model_is_not_downloaded(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([l for l in self.log() if l.startswith("snapshot_download")], [])


class VramGateTest(PreflightHarness):
    def gate(self, mib, settings_gb=None, env_gb=None):
        self._exe(self.root / "bin" / "nvidia-smi", f'echo "{mib}, 0"')
        extra = {}
        if settings_gb is not None:
            sj = self.root / "settings.json"
            sj.write_text(json.dumps({"min_vram_gb": settings_gb}))
            extra["VI_SETTINGS"] = str(sj)
        if env_gb is not None:
            extra["VI_MIN_VRAM_GB"] = str(env_gb)
        return self.preflight(**extra)

    def test_a_24_gb_card_passes_the_23_5_floor_from_the_settings(self):
        r = self.gate(24467, settings_gb=23.5)  # reads 23.9 GiB
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("VRAM 23.9 GiB", r.stdout)

    def test_the_settings_floor_is_enforced(self):
        r = self.gate(24467, settings_gb=24)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("VRAM 23.9 GiB < 24.0 GiB", r.stderr)

    def test_the_env_floor_overrides_the_settings(self):
        r = self.gate(24467, settings_gb=24, env_gb=23.5)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_default_floor_is_23_5(self):
        self.assertEqual(self.gate(24467).returncode, 0)
        r = self.gate(20480)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("< 23.5 GiB", r.stderr)


class FloorsInConfigTest(unittest.TestCase):
    def test_every_stage_uses_the_2_5_push_floor_and_omnivoice_the_23_5_vram_floor(self):
        import tomllib
        chars = JOBS.parent / "characters" / "tali"
        for name, stages in (("voicetune.toml", ("voicetune_preflight", "voicetune_train")),
                             ("omnivoice-ft.toml", ("omnivoice_preflight", "omnivoice_train"))):
            cfg = tomllib.loads((chars / name).read_text())
            for stage in stages:
                st = cfg["stage"][stage]["settings"]
                self.assertEqual(st["min_push_mb_s"], 2.5, (name, stage))
                if name == "omnivoice-ft.toml":
                    self.assertEqual(st["min_vram_gb"], 23.5, (name, stage))

    def test_omnivoice_stage_input_matches_the_scripts_variable(self):
        import tomllib
        cfg = tomllib.loads((JOBS.parent / "characters" / "tali" / "omnivoice-ft.toml").read_text())
        script = (JOBS / "voicetune-fresh.sh").read_text()
        for stage in ("omnivoice_preflight", "omnivoice_train"):
            for inp in cfg["stage"][stage]["inputs"]:
                self.assertIn("VI_INPUT_" + inp.upper(), script)


class RehearsalTest(PreflightHarness):
    def trainer_calls(self):
        return [l for l in self.log() if l.startswith("train_ft")]

    def test_kill_at_about_step_15_then_resume_from_checkpoint_10(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.trainer_calls()
        # probe, killed 20-step run, resumed run, then the verify call
        self.assertEqual(len(calls), 4, calls)
        self.assertIn("--steps 5 --save-steps 5", calls[0])
        self.assertIn("--steps 20 --save-steps 10", calls[1])
        self.assertNotIn("--resume-from-checkpoint", calls[1])
        self.assertIn("--resume-from-checkpoint", calls[2])
        self.assertIn("checkpoint-10", calls[2].split("--resume-from-checkpoint")[1].split()[0])
        self.assertIn("--verify-checkpoint", calls[3])
        self.assertIn("SIGKILL at micro-step", r.stdout)
        report = json.loads((self.work / "out" / "preflight.json").read_text())
        kill = report["kill_resume"]
        self.assertEqual(kill["resumed_from"], "checkpoint-10")
        self.assertTrue(10 < kill["killed_at_micro_step"] < 20, kill)

    def test_the_killed_run_never_reaches_checkpoint_20_by_itself(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        rr = next((self.work / "voicetune-fresh").glob("rehearse-run-*"))
        # micro-steps 1..~13 were logged once by the killed run; steps after the resume point
        # (11..20) appear again only because the resume re-ran them
        steps = [json.loads(l)["micro_step"] for l in (rr / "style_metrics.jsonl").read_text().splitlines()]
        self.assertEqual(steps.count(1), 1)
        self.assertGreater(steps.count(11), 1)  # logged by the killed run and again by the resume
        self.assertTrue((rr / "checkpoint-20" / "SHA256SUMS").is_file())

    def test_a_run_that_finishes_before_the_kill_fails_the_rehearsal(self):
        r = self.preflight(STUB_FAST="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no kill was proven", r.stderr)
        self.assertNotIn("preflight: ok", r.stdout)

    def test_a_resumed_checkpoint_without_a_manifest_fails_verification(self):
        r = self.preflight(STUB_RESUME_NO_MANIFEST="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("failed checkpoint verification", r.stderr)
        self.assertNotIn("preflight: ok", r.stdout)

    def test_a_trainer_that_never_writes_checkpoint_10_times_out(self):
        (self.root / "jobs" / "omnivoice-ft" / "train_ft.py").write_text(
            TRAIN_STUB.replace("save_steps == 0", "save_steps == -1").replace(
                "if step % a.save_steps == 0 or step == a.steps:",
                "if a.steps == 5 and step == 5:"))
        r = self.preflight(VI_REHEARSE_TIMEOUT="2")
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue("no checkpoint-10" in r.stderr or "exited before checkpoint-10" in r.stderr, r.stderr)


if __name__ == "__main__":
    unittest.main()


class GuardTests(PreflightHarness):
    """WP1 guards: pip retry, CUDA/bf16 gates, tokenise count guard (failure-modes.md)."""

    def _fake_python3(self, pip_fails):
        """A python3 on PATH whose `-m venv DIR` writes a shim venv python. The shim's
        `-m pip install` fails `pip_fails` times (counted in pip.count), then succeeds;
        it logs whether the final (renamed) venv folder exists at each try."""
        real = shutil.which("python3")
        self.pip_log = self.root / "pip.log"
        self.pip_count = self.root / "pip.count"
        venv_py = self.root / "venv_python.sh"
        venv_py.write_text(f"""#!/usr/bin/env bash
if [ "$1 $2" = "-m pip" ]; then
  d=$(dirname "$(dirname "$0")"); final=${{d%.tmp-*}}
  n=$(cat {self.pip_count} 2>/dev/null || echo 0); n=$((n + 1)); echo $n >{self.pip_count}
  echo "pip try $n args: $* final_exists=$([ "$d" != "$final" ] && [ -d "$final" ] && echo yes || echo no)" >>{self.pip_log}
  [ "$n" -gt {pip_fails} ]
  exit $?
fi
exec {real} "$@"
""")
        self._exe(self.root / "bin" / "python3", f"""if [ "$1 $2" = "-m venv" ]; then
  for a in "$@"; do d=$a; done
  mkdir -p "$d/bin"; cp {venv_py} "$d/bin/python"; cp {self.root}/bin/accelerate "$d/bin/accelerate"; chmod +x "$d/bin/python" "$d/bin/accelerate"; exit 0
fi
exec {real} "$@"
""".rstrip("\n"))

    def venvs(self):
        return sorted(p.name for p in (self.root / "home" / ".cache" / "vi").glob("venv-*"))

    def test_voicetune_pip_retry(self):
        self._fake_python3(pip_fails=2)
        r = self.preflight(OVFT_PYTHON=None, VI_PIP_BACKOFF="0 0")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = self.pip_log.read_text().splitlines()
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("--retries 10 --timeout 60", lines[0])
        self.assertTrue(all("final_exists=no" in l for l in lines), lines)  # renamed only after success
        names = self.venvs()
        self.assertEqual(len(names), 1, names)
        self.assertNotIn("tmp", names[0])

    def test_voicetune_pip_retry_final_failure_leaves_no_venv(self):
        self._fake_python3(pip_fails=99)
        r = self.preflight(OVFT_PYTHON=None, VI_PIP_BACKOFF="0 0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("worker-preflight: pip install failed", r.stderr)
        self.assertEqual(len(self.pip_log.read_text().splitlines()), 3)
        self.assertEqual(self.venvs(), [])

    def test_voicetune_cuda_init_retry_text(self):
        r = self.preflight(STUB_CUDA_FAIL="CUDA error: no kernel image")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("worker-preflight: cuda init failed: RuntimeError: CUDA error: no kernel image",
                      r.stderr)
        self.assertNotIn("train_ft", [l.split()[0] for l in self.log() if l])

    def test_voicetune_bf16_gate_text(self):
        r = self.preflight(STUB_BF16="0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("worker-preflight: bf16 unsupported", r.stderr)
        self.assertEqual(self.log(), [])

    def test_preflight_logs_library_versions(self):
        r = self.preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("versions: ", r.stdout + r.stderr)

    def test_voicetune_tokenise_count_guard(self):
        (self.data / "train_manifest.jsonl").write_text(
            "".join('{"id": "c%d", "text": "t"}\n' % i for i in range(1961)))
        r = self.preflight(STUB_TOK_OK="1950", STUB_TOK_FAILED="11")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("voicetune-fresh: tokenise train: 1950 of 1961 rows tokenised "
                      "(11 failed, 0 skipped); refusing to train on a silently shrunk set", r.stderr)
        self.assertFalse((self.data / "shards" / "train" / "data.lst").exists())
        self.assertNotIn("train_ft", [l.split()[0] for l in self.log() if l])

    def test_tokenise_guard_allows_a_complete_set_and_a_missing_summary_fails(self):
        (self.data / "train_manifest.jsonl").write_text(
            "".join('{"id": "c%d", "text": "t"}\n' % i for i in range(300)))
        r = self.preflight(STUB_TOK_OK="300")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_tokenise_guard_refuses_n_below_rows_even_with_zero_failed(self):
        (self.data / "train_manifest.jsonl").write_text(
            "".join('{"id": "c%d", "text": "t"}\n' % i for i in range(300)))
        r = self.preflight(STUB_TOK_OK="299")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("299 of 300 rows tokenised", r.stderr)


class FetchBaseModelGuardTest(unittest.TestCase):
    """fetch_base_model: retries, pinned sizes, atomic rename (jobs/omnivoice-ft)."""

    REV = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

    def setUp(self):
        import importlib.util, sys
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        spec = importlib.util.spec_from_file_location(
            "fetch_base_model_under_test", JOBS / "omnivoice-ft" / "fetch_base_model.py")
        self.fb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fb)
        self.hub = type(sys)("huggingface_hub")
        sys.modules["huggingface_hub"] = self.hub
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)
        os.environ["VI_FETCH_BACKOFF"] = "0 0 0"
        self.addCleanup(os.environ.pop, "VI_FETCH_BACKOFF", None)

    @staticmethod
    def write_files(folder, truncate=False):
        folder = Path(folder)
        (folder / "audio_tokenizer").mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}")
        for rel, size in (("model.safetensors", 2450344112 - (1 if truncate else 0)),
                          ("audio_tokenizer/model.safetensors", 805665628), ("pad.bin", 11460530)):
            with open(folder / rel, "wb") as f:
                f.truncate(size)

    def run_main(self):
        import contextlib, io
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.fb.main(["k2-fsa/OmniVoice", self.REV, str(self.root / "cache")])
        return rc, out.getvalue(), err.getvalue()

    def test_fetch_base_model_retry_and_size(self):
        calls = []

        def snap(repo, revision=None, local_dir=None):
            calls.append(local_dir)
            if len(calls) <= 2:
                raise TimeoutError("read timed out")
            self.write_files(local_dir)
        self.hub.snapshot_download = snap
        rc, out, err = self.run_main()
        dest = self.root / "cache" / "k2-fsa--OmniVoice-c5fdb5ccb189"
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c.endswith(".part") for c in calls))
        self.assertEqual(out.strip(), str(dest))
        self.assertFalse(dest.with_name(dest.name + ".part").exists())  # renamed atomically
        # A passing dest is reused without a download.
        calls.clear()
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(calls, [])

    def test_truncated_safetensors_is_refused(self):
        def snap(repo, revision=None, local_dir=None):
            self.write_files(local_dir, truncate=True)
        self.hub.snapshot_download = snap
        rc, out, err = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("worker-preflight: base model download failed", err)
        self.assertEqual(out, "")
        self.assertFalse((self.root / "cache" / "k2-fsa--OmniVoice-c5fdb5ccb189").exists())

    def test_a_truncated_existing_dest_is_not_reused(self):
        dest = self.root / "cache" / "k2-fsa--OmniVoice-c5fdb5ccb189"
        self.write_files(dest, truncate=True)
        calls = []

        def snap(repo, revision=None, local_dir=None):
            calls.append(local_dir)
            self.write_files(local_dir)
        self.hub.snapshot_download = snap
        rc, out, err = self.run_main()
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.fb.size_problems(dest, self.REV), [])
