"""
A one-at-a-time background worker, so the browse page can ingest an upload.

Builds run as a **subprocess** (`rag.py build`), not inside the server. Three
reasons: the embedding model would otherwise hold its memory inside the process
answering queries; a crash in the pipeline would take the MCP endpoint down with
it; and a build that wedges can be killed without restarting the server. The cost
is a model load per job, which against a job measured in minutes is noise.

One job at a time, always. Two concurrent builds would race on corpus.json and
contend for the same GPU, and the loser's registry writes would be silently
overwritten -- the kind of corruption that only shows up as missing documents
weeks later.
"""
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

ROOT = corpus.ROOT
LOCK = ROOT / "data" / ".build.lock"
LOG = ROOT / "data" / "jobs.log"
HISTORY = 20

_lock = threading.Lock()
_current = None          # the running job, or None
_history = deque(maxlen=HISTORY)
_queue = deque()


# ------------------------------------------------------------------ locking

def build_running():
    """
    Is a build already under way -- including one started from a terminal?

    The lock file holds a pid and a start time. A pid that no longer exists means
    a previous build died without cleaning up, and the lock is ignored rather
    than blocking every future build until someone deletes it by hand.
    """
    if not LOCK.exists():
        return None
    try:
        # utf-8-sig: a lock written by PowerShell carries a BOM
        info = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    pid = info.get("pid")
    if pid and not _pid_alive(pid):
        LOCK.unlink(missing_ok=True)
        return None
    return info


def _pid_alive(pid):
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=15).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def take_lock(pid, what):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(
        {"pid": pid, "what": what,
         "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
        encoding="utf-8")


def release_lock():
    LOCK.unlink(missing_ok=True)


# --------------------------------------------------------------------- jobs

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def submit(files, label=""):
    """Queue a build for freshly uploaded files. Returns the job record."""
    job = {
        "id": f"job-{int(time.time() * 1000):x}",
        "files": [Path(f).name for f in files],
        "label": label,
        "state": "queued",
        "stage": "",
        "detail": "",
        "queued_at": _now(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
    }
    with _lock:
        _queue.append(job)
        _history.appendleft(job)
        busy = _current is not None
    if not busy:
        threading.Thread(target=_drain, daemon=True).start()
    return job


def _drain():
    global _current
    while True:
        with _lock:
            if not _queue:
                _current = None
                return
            job = _queue.popleft()
            _current = job
        try:
            _run(job)
        except Exception as exc:
            job.update(state="failed", detail=f"{type(exc).__name__}: {exc}",
                       finished_at=_now())


STAGES = {"[1/5]": "scanning", "[2/5]": "extracting", "[3/5]": "cleaning",
          "[4/5]": "chunking", "[5/5]": "embedding"}


def _run(job):
    held = build_running()
    if held:
        job.update(state="failed", finished_at=_now(),
                   detail=f"another build is already running (pid {held.get('pid')}); "
                          "the upload is saved and will be picked up by the next build")
        return

    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    cmd = [str(python), str(ROOT / "rag.py"), "build", "--keep-going"]
    env = {**os.environ, "RAG_NO_REEXEC": "1", "PYTHONUNBUFFERED": "1"}

    job.update(state="running", started_at=_now(), stage="starting")
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with LOG.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== {job['id']} {job['queued_at']} {job['files']}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        take_lock(proc.pid, f"upload {job['id']}")
        try:
            for line in proc.stdout:
                log.write(line)
                stripped = line.strip()
                for marker, name in STAGES.items():
                    if stripped.startswith(marker):
                        job["stage"] = name
                # the ingest prints "  key: done/total" as it embeds
                if ":" in stripped and "/" in stripped and len(stripped) < 120:
                    job["detail"] = stripped[:110]
            proc.wait()
        finally:
            release_lock()

    job.update(state="done" if proc.returncode == 0 else "failed",
               returncode=proc.returncode, finished_at=_now(), stage="")
    if proc.returncode != 0:
        job["detail"] = f"build exited {proc.returncode} -- see data/jobs.log"


def status():
    with _lock:
        running = dict(_current) if _current else None
        recent = [dict(j) for j in _history]
    held = build_running()
    return {
        "running": running,
        "queued": len(_queue),
        "recent": recent,
        # a build started from a terminal is not ours, but the page should say so
        # rather than letting someone queue an upload that is doomed to refuse
        "external_build": held if held and (not running or held.get("what", "").find(running["id"]) < 0) else None,
    }
