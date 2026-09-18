#!/usr/bin/env python3
"""
One entry point for the whole project. Works on Windows, macOS and Linux.

    python rag.py up                 # install, build the index, serve it -- one command
    python rag.py setup              # just the virtualenv and dependencies
    python rag.py build              # PDFs in original/ -> vector index (incremental)
    python rag.py search "why is far-end crosstalk zero in a homogeneous medium?"
    python rag.py chat               # RAG chatbot in the terminal
    python rag.py serve              # MCP server for Claude Desktop (stdio)
    python rag.py serve --tunnel cloudflare   # public HTTPS URL for Claude's connector
    python rag.py status             # what is registered, built and indexed
    python rag.py doctor             # end-to-end self test

Run it with any Python 3.10+. Every subcommand except `setup` re-executes itself
inside .venv, so there is no activate step and no way to run half the pipeline
against the wrong interpreter.
"""
import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SCRIPTS = ROOT / "scripts"
ENV_FILE = ROOT / ".env"
WINDOWS = platform.system() == "Windows"
VENV_PYTHON = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")

TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
}


# ---------------------------------------------------------------- environment

def load_env():
    """Read .env into os.environ without adding a dependency. Real env wins."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def in_venv():
    try:
        return Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except OSError:
        return False


def reexec_in_venv(argv):
    """Hand the command over to the venv interpreter, once."""
    if in_venv() or os.environ.get("RAG_NO_REEXEC"):
        return
    if not VENV_PYTHON.exists():
        sys.exit("No .venv yet. Run:  python rag.py setup")
    os.environ["RAG_NO_REEXEC"] = "1"
    raise SystemExit(subprocess.call([str(VENV_PYTHON), str(ROOT / "rag.py"), *argv]))


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.call([str(c) for c in cmd], **kw)


def need(cmd, why):
    if run(cmd) != 0:
        sys.exit(f"failed: {why}")


# --------------------------------------------------------------------- setup

def detect_accelerator():
    """cu124 if an NVIDIA driver is present, else cpu. Torch wheels differ by 2 GB."""
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=20)
            return "cu124"
        except Exception:
            pass
    return "cpu"


def cmd_setup(args):
    if not VENV_PYTHON.exists():
        need([sys.executable, "-m", "venv", str(VENV)], "creating .venv")
    pip = [str(VENV_PYTHON), "-m", "pip"]
    need([*pip, "install", "--upgrade", "pip", "--quiet"], "upgrading pip")

    flavour = args.torch or detect_accelerator()
    print(f"installing torch ({flavour})")
    need([*pip, "install", "torch", "--index-url", TORCH_INDEX[flavour]],
         "installing torch")
    need([*pip, "install", "-r", str(ROOT / "requirements.txt")],
         "installing requirements")

    if not ENV_FILE.exists():
        shutil.copyfile(ROOT / ".env.example", ENV_FILE)
        print(f"wrote {ENV_FILE.name} -- set MCP_TEAM_PASSWORD before serving publicly")
    print("\nsetup complete. Next:  python rag.py build")
    return 0


# --------------------------------------------------------------------- build

def cmd_build(args):
    load_env()
    sys.path.insert(0, str(SCRIPTS))
    import chunk_corpus
    import clean_corpus
    import corpus
    import extract_pdf
    import ingest_qdrant

    if not corpus.pdfs() and not list(corpus.TEXT_DIR.glob("*.txt")):
        sys.exit(f"no PDFs in {corpus.PDF_DIR.name}/ -- drop some in and run this again")

    print("[1/4] extracting text from PDFs")
    _, report, added = extract_pdf.run(args.only, args.engine, args.force)
    for key in added:
        print(f"  + registered new document: {key}")
    for key, status, detail in report:
        if status in {"MISSING", "NO TEXT LAYER"}:
            print(f"  ! {key}: {status} -- {detail}")
        else:
            print(f"  {key}: {status} ({detail})")
    if any(s == "NO TEXT LAYER" for _, s, _ in report) and not args.keep_going:
        sys.exit("\nA PDF has no text layer. OCR it first (e.g. `ocrmypdf in.pdf out.pdf`),\n"
                 "or re-run with --keep-going to index the rest.")

    print("[2/4] cleaning")
    _, rows, skipped = clean_corpus.run(args.only, args.force)
    if skipped:
        print(f"  unchanged: {', '.join(skipped)}")

    print("[3/4] chunking")
    _, stats, skipped = chunk_corpus.run(args.only, args.force)
    if skipped:
        print(f"  unchanged: {', '.join(skipped)}")
    for key, (secs, n) in stats.items():
        print(f"  {key}: {secs} sections, {n} chunks")

    if args.no_ingest:
        print("\nstopping before ingest (--no-ingest)")
        return 0

    print("[4/4] embedding and indexing")
    collection, loaded, skipped, points = ingest_qdrant.run(
        args.only, args.force, args.recreate, args.batch)
    if skipped:
        print(f"  unchanged: {', '.join(skipped)}")
    for key, n in loaded.items():
        print(f"  {key}: {n} points")

    where = os.environ.get("QDRANT_URL") or corpus.QDRANT_PATH
    print(f"\nindex '{collection}': {points} points at {where}")
    return 0


def cmd_remove(args):
    """Deleting the PDF is not enough: its points stay in the index until told."""
    load_env()
    sys.path.insert(0, str(SCRIPTS))
    import corpus
    import ingest_qdrant
    import search_qdrant as sq

    cfg = corpus.load()
    doc = cfg["documents"].get(args.key)
    if doc is None:
        sys.exit(f"no document {args.key!r} in corpus.json "
                 f"(have: {', '.join(cfg['documents']) or 'none'})")

    client = sq.get_client()
    try:
        ingest_qdrant.drop_document(client, sq.COLLECTION, args.key)
        manifest = ingest_qdrant.manifest_name(sq.COLLECTION)
        if client.collection_exists(manifest):
            client.delete(collection_name=manifest, points_selector=[doc["slot"]])
    finally:
        client.close()

    paths = corpus.doc_paths(args.key, doc)
    for name in ("clean", "chunks"):
        paths[name].unlink(missing_ok=True)
    if args.purge:
        for name in ("text", "pdf"):
            if paths[name]:
                paths[name].unlink(missing_ok=True)

    del cfg["documents"][args.key]
    corpus.save(cfg)
    print(f"removed {args.key} from the index and corpus.json"
          + (" (and deleted its PDF and extracted text)" if args.purge else ""))
    if not args.purge:
        print(f"its PDF is still in {corpus.PDF_DIR.name}/ -- the next build re-adds it")
    return 0


def cmd_status(args):
    load_env()
    sys.path.insert(0, str(SCRIPTS))
    import corpus

    cfg = corpus.load()
    unregistered = [p.name for p in corpus.pdfs()
                    if p.name not in {d.get("pdf") for d in cfg["documents"].values()}]

    print(f"{'document':<14} {'pdf':>5} {'text':>6} {'clean':>6} {'chunks':>8}")
    for key, doc in cfg["documents"].items():
        p = corpus.doc_paths(key, doc)
        n = doc.get("chunks", 0) if p["chunks"].exists() else 0
        print(f"{key:<14} {'yes' if p['pdf'] and p['pdf'].exists() else 'no':>5} "
              f"{'yes' if p['text'].exists() else 'no':>6} "
              f"{'yes' if p['clean'].exists() else 'no':>6} {n:>8}")
    for name in unregistered:
        print(f"  (not yet registered: {name} -- run `python rag.py build`)")

    try:
        import search_qdrant as sq
        rows = sq.list_documents()
        print(f"\nindexed in '{sq.COLLECTION}':")
        for r in rows:
            print(f"  {r['book']:<14} {r['chunks']:>6}  {r['title'][:60]}")
    except Exception as exc:
        print(f"\nindex not reachable: {exc.__class__.__name__}: {exc}")
    return 0


# ----------------------------------------------------------------- retrieval

def cmd_search(args):
    load_env()
    return run([VENV_PYTHON, SCRIPTS / "search_qdrant.py", args.query,
                "-k", str(args.k)]
               + (["--book", args.book] if args.book else [])
               + ([] if args.rerank else ["--no-rerank"])
               + (["--full"] if args.full else []))


def cmd_chat(args):
    load_env()
    return run([VENV_PYTHON, SCRIPTS / "rag_chat.py", *(["-k", str(args.k)])]
               + (["--book", args.book] if args.book else [])
               + ([args.question] if args.question else []))


# --------------------------------------------------------------------- serve

# A tunnel has two identities that are easy to conflate: the name it is run under
# and the hostname the public reaches it on. For ngrok a reserved domain is both.
# For a named Cloudflare tunnel they differ -- `cloudflared tunnel run si-rag` may
# serve rag.example.edu -- so --name and --domain are separate flags.
TUNNELS = {
    # binary, argv builder, regex that finds a quick tunnel's URL in its output
    "cloudflare": (
        "cloudflared",
        # --url on a named tunnel overrides its ingress rules, so the tunnel does
        # not need a config.yml. Without it, a named tunnel with no config accepts
        # connections and forwards them nowhere: every request simply times out.
        lambda port, name: (
            ["cloudflared", "tunnel", "run", "--url", f"http://localhost:{port}", name]
            if name else
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]),
        re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com"),
    ),
    "ngrok": (
        "ngrok",
        lambda port, name: ["ngrok", "http", str(port), "--log", "stdout"]
        + (["--domain", name] if name else []),
        re.compile(r"https://[a-z0-9.-]+\.(?:ngrok-free\.(?:app|dev)|ngrok\.io|ngrok\.app)"),
    ),
}


def port_in_use(host, port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host if host != "0.0.0.0" else "", int(port)))
            return False
        except OSError:
            return True


def port_holders(port):
    """PID and image name of whatever holds the port, so the message is actionable."""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             timeout=15).stdout
        pids = {l.split()[-1] for l in out.splitlines()
                if f":{port} " in l and "LISTENING" in l}
        named = []
        for pid in pids:
            task = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                                  capture_output=True, text=True, timeout=15).stdout
            name = task.splitlines()[1].split('","')[0].strip('"') if "\n" in task else "?"
            named.append(f"pid {pid} ({name})")
        return named or ["unknown"]
    except Exception:
        return ["unknown"]


def drain(proc, log_path=None):
    """
    Keep reading the tunnel's output for the life of the process.

    Not optional. The tunnel logs steadily, and a pipe nobody reads fills after
    ~64 KB, at which point the tunnel blocks on write and stops forwarding
    traffic -- a tunnel that works for a while and then silently stops reaching
    the origin, which looks exactly like the server having crashed.
    """
    def pump():
        handle = open(log_path, "a", encoding="utf-8", errors="replace") if log_path else None
        try:
            for line in proc.stdout:
                if handle:
                    handle.write(line)
                    handle.flush()
        except Exception:
            pass
        finally:
            if handle:
                handle.close()

    threading.Thread(target=pump, daemon=True).start()


def start_tunnel(kind, port, name, domain, timeout=40, log_path=None):
    """
    Start the tunnel first: the server needs its public hostname before it binds,
    to allow that Host header through the SDK's DNS-rebinding protection.
    """
    binary, argv, url_re = TUNNELS[kind]
    if not shutil.which(binary):
        sys.exit(f"{binary} is not on PATH. Install it, or pass --url if you terminate "
                 "HTTPS yourself.")
    if kind == "cloudflare" and name:
        # Two connectors registered for one named tunnel make Cloudflare balance
        # between them; a stale one with no origin behind it then fails roughly
        # half of all requests with error 1033.
        stale = running_tunnels(binary)
        if stale:
            print(f"! {binary} is already running (pid {', '.join(stale)}). Stop it first, "
                  "or it will take half the traffic and answer 1033.")

    proc = subprocess.Popen(argv(port, name or (domain if kind == "ngrok" else None)),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    # a known hostname needs no scraping; only a quick tunnel's random one does
    if domain:
        drain(proc, log_path)
        return proc, f"https://{domain}"
    if name and kind == "cloudflare":
        proc.terminate()
        sys.exit("a named Cloudflare tunnel does not announce its hostname -- pass "
                 "--domain <hostname> (or --url) so the server knows its public name")
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        m = url_re.search(line)
        if m:
            drain(proc, log_path)   # keep draining, for the same reason
            return proc, m.group(0)
    proc.terminate()
    sys.exit(f"{binary} did not report a public URL within {timeout}s")


def running_tunnels(binary):
    """PIDs of an already-running tunnel binary, so a second one is not started blind."""
    try:
        if WINDOWS:
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {binary}.exe", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=15).stdout
            return [l.split('","')[1] for l in out.splitlines() if l.startswith(f'"{binary}')]
        out = subprocess.run(["pgrep", "-x", binary], capture_output=True,
                             text=True, timeout=15).stdout
        return out.split()
    except Exception:
        return []


def cmd_serve(args):
    load_env()
    env = dict(os.environ)
    env.setdefault("RAG_RERANK", "0" if args.no_rerank else "1")

    if args.stdio:
        env["MCP_TRANSPORT"] = "stdio"
        return run([VENV_PYTHON, SCRIPTS / "mcp_server.py"], env=env)

    env["MCP_TRANSPORT"] = "streamable-http"
    env["MCP_HOST"] = args.host
    env["MCP_PORT"] = str(args.port)

    log_path = Path(args.log).resolve() if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check the port before starting a tunnel. Otherwise the tunnel comes up, the
    # server loses the bind race, and the console still says "tunnel up" while the
    # only evidence of the real failure is an Errno 10048 buried in the log.
    if port_in_use(args.host, args.port):
        sys.exit(f"port {args.port} is already in use -- another server is running.\n"
                 f"Stop it first, or pass --port <n> to run a second one.\n"
                 + (f"  holder: {', '.join(port_holders(args.port))}\n" if WINDOWS else ""))

    tunnel = None
    if args.url:
        public = args.url if args.url.startswith("http") else f"https://{args.url}"
    elif args.tunnel:
        tunnel, public = start_tunnel(args.tunnel, args.port, args.name, args.domain,
                                      log_path=log_path)
        print(f"tunnel up: {public}")
    else:
        public = f"http://localhost:{args.port}"

    env["MCP_PUBLIC_URL"] = public
    if not env.get("MCP_TEAM_PASSWORD") and not env.get("MCP_AUTH_TOKEN"):
        if args.tunnel or args.url:
            sys.exit("refusing to expose an unauthenticated server.\n"
                     "Set MCP_TEAM_PASSWORD in .env (Claude's connector UI needs this "
                     "OAuth mode), or MCP_AUTH_TOKEN for scripted clients.")
        print("! no MCP_TEAM_PASSWORD set -- local, unauthenticated")

    print(f"\nAdd this in Claude as a custom connector:\n    {public}/mcp\n")
    if log_path:
        print(f"logging to {log_path}\n")
    try:
        if log_path:
            with log_path.open("a", encoding="utf-8", errors="replace") as fh:
                return run([VENV_PYTHON, SCRIPTS / "mcp_server.py"], env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
        return run([VENV_PYTHON, SCRIPTS / "mcp_server.py"], env=env)
    except KeyboardInterrupt:
        return 0
    finally:
        if tunnel and tunnel.poll() is None:
            tunnel.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGTERM)
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()


# ------------------------------------------------------------------ autostart

TASK_NAME = "SignalIntegrityRAG"


def cmd_autostart(args):
    """
    Keep the connector up without a terminal window babysitting it.

    A connector is only reachable while the server runs, and a stable URL is no
    use if the process behind it dies with the shell that started it. On Windows
    this registers a logon task; elsewhere it prints the systemd unit to install,
    since a user service there needs a file, not a command.
    """
    serve = [str(VENV_PYTHON.with_name("pythonw.exe") if WINDOWS else VENV_PYTHON),
             str(ROOT / "rag.py"), "serve", "--log", str(ROOT / "data" / "serve.log")]
    for flag in ("tunnel", "name", "domain", "url"):
        value = getattr(args, flag, None)
        if value:
            serve += [f"--{flag}", value]

    if not WINDOWS:
        unit = (
            "[Unit]\nDescription=Signal-integrity RAG MCP server\nAfter=network-online.target\n\n"
            f"[Service]\nWorkingDirectory={ROOT}\nExecStart={' '.join(serve)}\n"
            "Restart=always\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n")
        print(f"Save this as ~/.config/systemd/user/{TASK_NAME}.service, then:\n"
              f"  systemctl --user daemon-reload && systemctl --user enable --now {TASK_NAME}\n")
        print(unit)
        return 0

    # The Startup folder, not a scheduled task: `schtasks /SC ONLOGON` needs an
    # elevated shell, and needing admin to run your own retrieval server is a poor
    # trade for what this is.
    startup = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
    script = startup / f"{TASK_NAME}.cmd"

    if args.remove:
        script.unlink(missing_ok=True)
        print(f"removed {script}")
        return 0

    if args.status:
        if not script.exists():
            print("autostart is not installed")
            return 1
        print(f"{script}:\n")
        print(script.read_text(encoding="utf-8"))
        return 0

    if not (args.tunnel or args.url):
        sys.exit("autostart needs the public URL it should serve on, e.g.\n"
                 "  python rag.py autostart --tunnel cloudflare --name si-rag "
                 "--domain rag.example.org")

    quoted = " ".join(f'"{part}"' if " " in part else part for part in serve)
    startup.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "@echo off\r\n"
        f"cd /d \"{ROOT}\"\r\n"
        # pythonw keeps it windowless; start lets the logon sequence carry on
        f"start \"\" {quoted}\r\n",
        encoding="utf-8")

    print(f"\ninstalled {script}")
    print("it starts at every logon, with no window.")
    print(f"  start now : {script}")
    print(f"  stop      : taskkill /F /IM pythonw.exe /IM cloudflared.exe")
    print(f"  remove    : python rag.py autostart --remove")
    print(f"  logs      : {ROOT / 'data' / 'serve.log'}")
    return 0


# -------------------------------------------------------------------- doctor

def cmd_doctor(args):
    load_env()
    return run([VENV_PYTHON, SCRIPTS / "doctor.py"]
               + (["--url", args.url] if args.url else [])
               + (["--token", args.token] if args.token else []))


def cmd_up(args):
    if not VENV_PYTHON.exists():
        rc = cmd_setup(argparse.Namespace(torch=None))
        if rc:
            return rc
        os.environ.pop("RAG_NO_REEXEC", None)
        return subprocess.call([str(VENV_PYTHON), str(ROOT / "rag.py"), "up"])
    rc = cmd_build(argparse.Namespace(
        only=None, engine="pymupdf", force=False, recreate=False,
        batch=8, no_ingest=False, keep_going=False))
    if rc:
        return rc
    return cmd_serve(argparse.Namespace(
        stdio=False, host="127.0.0.1", port=8000, tunnel=None, name=None,
        domain=None, url=None, no_rerank=False, log=None))


# ----------------------------------------------------------------------- cli

def build_parser():
    ap = argparse.ArgumentParser(
        prog="rag", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="create .venv and install dependencies")
    p.add_argument("--torch", choices=sorted(TORCH_INDEX),
                   help="torch build to install (default: cu124 if an NVIDIA GPU is present)")
    p.set_defaults(func=cmd_setup, venv=False)

    p = sub.add_parser("build", help="PDFs in original/ -> vector index (incremental)")
    p.add_argument("--only", nargs="*", help="document keys to rebuild")
    p.add_argument("--engine", choices=["pymupdf", "pdftotext"], default="pymupdf")
    p.add_argument("--force", action="store_true", help="redo every stage")
    p.add_argument("--recreate", action="store_true", help="drop the collection first")
    p.add_argument("--batch", type=int, default=8, help="embedding batch size")
    p.add_argument("--no-ingest", action="store_true", help="stop after chunking")
    p.add_argument("--keep-going", action="store_true",
                   help="index the rest even if a PDF needs OCR")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("remove", help="drop a document from the index and the registry")
    p.add_argument("key", help="document key, as shown by `status`")
    p.add_argument("--purge", action="store_true",
                   help="also delete its PDF and extracted text")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("status", help="what is registered, built and indexed")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="one hybrid search from the terminal")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--book", default=None)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("chat", help="RAG chatbot (needs ANTHROPIC_API_KEY)")
    p.add_argument("question", nargs="?")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--book", default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="run the MCP server")
    p.add_argument("--stdio", action="store_true",
                   help="stdio transport for Claude Desktop (default is HTTP)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tunnel", choices=sorted(TUNNELS),
                   help="expose it on a public HTTPS URL")
    p.add_argument("--name", help="named tunnel to run (cloudflared tunnel name)")
    p.add_argument("--domain", help="public hostname it serves on (ngrok reserved domain, "
                                    "or the named tunnel's hostname)")
    p.add_argument("--url", help="public URL you already terminate yourself")
    p.add_argument("--no-rerank", action="store_true", help="faster, worse results")
    p.add_argument("--log", help="append server and tunnel output to this file")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("autostart",
                       help="keep the server running at logon, without a terminal")
    p.add_argument("--tunnel", choices=sorted(TUNNELS))
    p.add_argument("--name", help="named tunnel to run")
    p.add_argument("--domain", help="public hostname it serves on")
    p.add_argument("--url", help="public URL you already terminate yourself")
    p.add_argument("--status", action="store_true", help="show the registered task")
    p.add_argument("--remove", action="store_true", help="unregister it")
    p.set_defaults(func=cmd_autostart)

    p = sub.add_parser("doctor", help="end-to-end self test")
    p.add_argument("--url", help="check a deployed endpoint instead of a local stdio server")
    p.add_argument("--token", help="bearer token for --url")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("up", help="setup + build + serve, in one command")
    p.set_defaults(func=cmd_up, venv=False)

    return ap


def main():
    args = build_parser().parse_args()
    if getattr(args, "venv", True):
        reexec_in_venv(sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
