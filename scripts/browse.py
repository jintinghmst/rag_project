"""
A browsable view of the corpus, served by the MCP server itself.

Routes mounted under /browse on the same origin as /mcp, so the tunnel and the
certificate already in place cover it and there is no second thing to deploy:

    /browse                 the page (or a passphrase prompt)
    /browse/login           POST, exchanges MCP_TEAM_PASSWORD for a cookie
    /browse/api/corpus      every registered document, as metadata
    /browse/api/search      hybrid retrieval, returning cited passages

Access is the same shared passphrase the MCP connector uses, held in a signed
cookie. That matters more here than for the connector: the corpus is verbatim
text from copyrighted textbooks and paywalled journals, and a browsable index of
it on a public hostname is exactly the thing not to leave open. The listing is
metadata only; passages appear only in response to a search, which is what the
MCP tools already expose.
"""
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

COOKIE = "si_browse"
MAX_AGE = 14 * 24 * 3600
SEARCH_TIMEOUT_NOTE = "first search loads the models and can take a minute"


# ----------------------------------------------------------------------- auth

def _secret():
    return (os.environ.get("MCP_TEAM_PASSWORD")
            or os.environ.get("MCP_AUTH_TOKEN") or "").encode()


def issue(secret):
    expiry = str(int(time.time() + MAX_AGE))
    sig = hmac.new(secret, expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def valid(token, secret):
    try:
        expiry, sig = token.split(".", 1)
        want = hmac.new(secret, expiry.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, want) and int(expiry) > time.time()
    except Exception:
        return False


def authorised(request):
    secret = _secret()
    if not secret:
        return True   # no passphrase configured: a local, unexposed server
    return valid(request.cookies.get(COOKIE, ""), secret)


# ----------------------------------------------------------------------- data

_cache = {"at": 0, "rows": None}


def corpus_rows(list_documents):
    """
    One row per document: what is registered, marked with what is searchable.

    The registry and the index disagree whenever a build is part-done -- 9,145
    registered against 4,031 indexed, right now -- and hiding that would make the
    page lie about what a search can actually reach.
    """
    if _cache["rows"] is not None and time.time() - _cache["at"] < 60:
        return _cache["rows"]

    indexed = {}
    try:
        for row in list_documents():
            indexed[row["book"]] = row.get("chunks", 0)
    except Exception:
        pass

    rows = []
    for key, doc in corpus.load()["documents"].items():
        pages = doc.get("pages")
        offset = doc.get("page_offset")
        if pages and offset is not None:
            span = f"{pages[0] + offset}-{pages[1] + offset}"
        elif pages:
            span = f"pdf {pages[0]}-{pages[1]}"
        else:
            span = ""
        source = doc.get("source") or ""
        venue, _, year = source.rpartition(" / ")
        rows.append({
            "key": key,
            "title": doc.get("title", key),
            "venue": venue or ("Textbook" if not source else source),
            "year": year if year.isdigit() else "",
            "kind": doc.get("kind", "book"),
            "pages": span,
            "chunks": indexed.get(key, 0),
        })
    rows.sort(key=lambda r: (r["venue"], r["year"], r["title"]))
    _cache.update(at=time.time(), rows=rows)
    return rows


# ----------------------------------------------------------------------- page

LOGIN_HTML = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corpus</title><style>
:root{color-scheme:light dark;--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--acc:#3d5a80}
@media(prefers-color-scheme:dark){:root{--bg:#17171a;--fg:#e9e9e6;--mut:#9a9a94;--line:#2e2e33;--acc:#8fb0d4}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;
background:var(--bg);color:var(--fg);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}
form{width:100%;max-width:340px}h1{font-size:1.15rem;margin:0 0 4px}
p{color:var(--mut);font-size:.88rem;margin:0 0 20px}
input{width:100%;padding:11px 13px;font:inherit;border:1px solid var(--line);border-radius:8px;
background:var(--bg);color:var(--fg)}input:focus{outline:2px solid var(--acc);outline-offset:1px}
button{width:100%;margin-top:12px;padding:11px;font:inherit;font-weight:600;border:0;border-radius:8px;
background:var(--acc);color:#fff;cursor:pointer}
.err{color:#c0392b;font-size:.85rem;margin-top:10px}
</style></head><body><form method="post" action="/browse/login">
<h1>Signal-integrity corpus</h1><p>Enter the team passphrase to browse.</p>
<input name="pw" type="password" autocomplete="current-password" autofocus required
       aria-label="Team passphrase"><button>View corpus</button>__ERROR__
</form></body></html>"""

PAGE_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corpus</title><style>
:root{color-scheme:light dark;
 --bg:#fbfbfa;--panel:#fff;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--acc:#3d5a80;--chip:#eef1f5}
@media(prefers-color-scheme:dark){:root{
 --bg:#17171a;--panel:#1e1e22;--fg:#e9e9e6;--mut:#9a9a94;--line:#2e2e33;--acc:#8fb0d4;--chip:#26262c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
 padding:14px 16px 10px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.05rem;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:.82rem;margin:0 0 12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
input,select{padding:9px 11px;font:inherit;border:1px solid var(--line);border-radius:8px;
 background:var(--panel);color:var(--fg)}
input:focus,select:focus{outline:2px solid var(--acc);outline-offset:1px}
#q{flex:1 1 260px;min-width:0}
button{padding:9px 14px;font:inherit;border:1px solid var(--line);border-radius:8px;
 background:var(--panel);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:transparent;font-weight:600}
main{max-width:1080px;margin:0 auto;padding:16px}
.stats{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:.82rem;margin-bottom:12px}
.stats b{color:var(--fg)}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th{text-align:left;font-weight:600;color:var(--mut);font-size:.76rem;text-transform:uppercase;
 letter-spacing:.04em;padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:hover td{background:var(--chip)}
.t{font-weight:500}
.chip{display:inline-block;padding:1px 7px;border-radius:999px;background:var(--chip);
 color:var(--mut);font-size:.74rem;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.no{color:var(--mut);font-style:italic}
.more{margin:16px auto;display:block}
.hit{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.cite{font-size:.78rem;color:var(--mut);margin-bottom:6px}
.hit p{margin:0;font-size:.88rem;white-space:pre-wrap}
.empty{color:var(--mut);padding:28px 0;text-align:center}
@media(max-width:620px){.hide-s{display:none}}
</style></head><body>
<header><div class="wrap">
  <h1>Signal-integrity corpus</h1>
  <p class="sub" id="sub">loading…</p>
  <div class="controls">
    <input id="q" placeholder="Filter by title, venue, year…" aria-label="Filter">
    <select id="venue" aria-label="Venue"><option value="">All venues</option></select>
    <select id="kind" aria-label="Type">
      <option value="">All types</option><option value="book">Textbooks</option>
      <option value="article">Papers</option><option value="issue">Unsplit issues</option>
    </select>
    <button id="go" class="primary" title="Search the text of the corpus">Search text</button>
  </div>
</div></header>
<main>
  <div class="stats" id="stats"></div>
  <div id="results"></div>
  <table id="table"><thead><tr>
    <th data-k="title">Title</th>
    <th data-k="venue" class="hide-s">Venue</th>
    <th data-k="year">Year</th>
    <th data-k="pages" class="hide-s">Pages</th>
    <th data-k="chunks" class="num">Chunks</th>
  </tr></thead><tbody id="rows"></tbody></table>
  <button class="more" id="more" hidden>Show more</button>
</main>
<script>
const $ = s => document.querySelector(s);
let ALL = [], VIEW = [], shown = 0, sortK = 'venue', sortAsc = true;
const PAGE = 150;

fetch('/browse/api/corpus').then(r => r.json()).then(d => {
  ALL = d.documents;
  const venues = [...new Set(ALL.map(r => r.venue))].sort();
  for (const v of venues) $('#venue').add(new Option(v, v));
  $('#sub').textContent =
    `${d.total.toLocaleString()} documents · ${d.indexed.toLocaleString()} searchable · ` +
    `${d.chunks.toLocaleString()} chunks`;
  apply();
}).catch(e => { $('#sub').textContent = 'could not load the corpus: ' + e; });

function apply() {
  const q = $('#q').value.toLowerCase().trim();
  const v = $('#venue').value, k = $('#kind').value;
  VIEW = ALL.filter(r =>
    (!v || r.venue === v) && (!k || r.kind === k) &&
    (!q || (r.title + ' ' + r.venue + ' ' + r.year).toLowerCase().includes(q)));
  VIEW.sort((a, b) => {
    const x = a[sortK], y = b[sortK];
    const c = typeof x === 'number' ? x - y : String(x).localeCompare(String(y));
    return sortAsc ? c : -c;
  });
  shown = 0; $('#rows').innerHTML = ''; $('#results').innerHTML = '';
  const n = VIEW.filter(r => r.chunks > 0).length;
  $('#stats').innerHTML =
    `<span><b>${VIEW.length.toLocaleString()}</b> shown</span>` +
    `<span><b>${n.toLocaleString()}</b> searchable</span>` +
    `<span><b>${VIEW.reduce((s, r) => s + r.chunks, 0).toLocaleString()}</b> chunks</span>`;
  draw();
}

function draw() {
  const frag = document.createDocumentFragment();
  for (const r of VIEW.slice(shown, shown + PAGE)) {
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td class="t"></td><td class="hide-s"><span class="chip"></span></td>` +
      `<td>${r.year || ''}</td><td class="hide-s">${r.pages || ''}</td>` +
      `<td class="num">${r.chunks ? r.chunks.toLocaleString() : '<span class="no">not indexed</span>'}</td>`;
    tr.children[0].textContent = r.title;            // textContent: titles are data
    tr.children[1].firstChild.textContent = r.venue;
    frag.appendChild(tr);
  }
  $('#rows').appendChild(frag);
  shown += PAGE;
  $('#more').hidden = shown >= VIEW.length;
  if (!VIEW.length) $('#rows').innerHTML =
    '<tr><td colspan="5" class="empty">Nothing matches that filter.</td></tr>';
}

$('#more').onclick = draw;
for (const el of ['#q', '#venue', '#kind']) $(el).oninput = apply;
document.querySelectorAll('th').forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  sortAsc = sortK === k ? !sortAsc : true; sortK = k; apply();
});

async function search() {
  const q = $('#q').value.trim();
  if (!q) { $('#q').focus(); return; }
  $('#results').innerHTML = '<div class="empty">searching… the first one loads the models</div>';
  try {
    const p = new URLSearchParams({q, venue: $('#venue').value, kind: $('#kind').value});
    const d = await (await fetch('/browse/api/search?' + p)).json();
    if (d.error) throw new Error(d.error);
    if (!d.hits.length) { $('#results').innerHTML = '<div class="empty">No passages matched.</div>'; return; }
    $('#results').innerHTML = '';
    for (const h of d.hits) {
      const div = document.createElement('div');
      div.className = 'hit';
      const c = document.createElement('div'); c.className = 'cite'; c.textContent = h.cite;
      const p = document.createElement('p'); p.textContent = h.text;
      div.append(c, p); $('#results').appendChild(div);
    }
  } catch (e) { $('#results').innerHTML = '<div class="empty">search failed: ' + e.message + '</div>'; }
}
$('#go').onclick = search;
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
</script></body></html>"""


# --------------------------------------------------------------------- routes

def register(server, search, cite, list_documents, quiet):
    """Mount the browse routes on the MCP server."""
    from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

    secure = os.environ.get("MCP_PUBLIC_URL", "").startswith("https")

    def gate(request):
        return None if authorised(request) else HTMLResponse(
            LOGIN_HTML.replace("__ERROR__", ""), status_code=401)

    @server.custom_route("/browse", methods=["GET"])
    async def browse(request):
        if not authorised(request):
            return HTMLResponse(LOGIN_HTML.replace("__ERROR__", ""), status_code=401)
        return HTMLResponse(PAGE_HTML)

    @server.custom_route("/browse/login", methods=["POST"])
    async def browse_login(request):
        form = await request.form()
        secret = _secret()
        if not secret or not hmac.compare_digest(form.get("pw", ""), secret.decode()):
            return HTMLResponse(
                LOGIN_HTML.replace("__ERROR__",
                                   '<div class="err">Wrong passphrase.</div>'),
                status_code=401)
        response = RedirectResponse("/browse", status_code=303)
        response.set_cookie(COOKIE, issue(secret), max_age=MAX_AGE, httponly=True,
                            samesite="lax", secure=secure, path="/browse")
        return response

    @server.custom_route("/browse/api/corpus", methods=["GET"])
    async def api_corpus(request):
        denied = gate(request)
        if denied:
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        rows = corpus_rows(list_documents)
        return JSONResponse({
            "documents": rows,
            "total": len(rows),
            "indexed": sum(1 for r in rows if r["chunks"]),
            "chunks": sum(r["chunks"] for r in rows),
        })

    @server.custom_route("/browse/api/search", methods=["GET"])
    async def api_search(request):
        if not authorised(request):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return JSONResponse({"hits": []})
        venue = request.query_params.get("venue") or None
        kind = request.query_params.get("kind") or None
        try:
            import anyio

            def work():
                with quiet():
                    return search(q, k=8, candidates=40, source=venue, kind=kind)

            # the embedding and rerank models block; keeping them off the event
            # loop lets the page stay responsive and other requests proceed
            hits = await anyio.to_thread.run_sync(work)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
        return JSONResponse({"hits": [
            {"cite": cite(h), "text": h["text"][:1800], "book": h["book"],
             "score": round(float(h.get("rerank_score", h.get("score", 0))), 4)}
            for h in hits]})
