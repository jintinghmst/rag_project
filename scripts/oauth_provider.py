"""
A minimal OAuth 2.1 authorization server for the MCP connector.

Claude's connector UI will not accept a static bearer token: it discovers the
server's auth metadata and performs Dynamic Client Registration, so the server
has to be a real authorization server. This is the smallest one that is still
honest about who gets in.

The gate is a single team passphrase (MCP_TEAM_PASSWORD), entered on a login page
during the authorize step. Everyone who knows it can connect; there is no
per-user identity beyond a label they type. That is the same trust model as the
shared bearer token, wrapped in the flow Claude requires.

State lives in memory. Restarting the server forces everyone to reconnect once,
which is acceptable for a team-sized deployment and keeps the moving parts few.
For per-user identity or audit trails, put a real IdP in front instead.
"""
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

AUTH_CODE_TTL = 300          # seconds; the code is exchanged immediately
ACCESS_TOKEN_TTL = 60 * 60 * 24 * 30
PENDING_TTL = 600            # how long a login page stays valid


@dataclass
class _Pending:
    """An authorize request parked while the user proves they know the passphrase."""
    client_id: str
    params: AuthorizationParams
    created_at: float = field(default_factory=time.time)


class TeamPasswordProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    def __init__(self, password: str, resource: str):
        if not password:
            raise ValueError("MCP_TEAM_PASSWORD must be set to use OAuth mode")
        self._password = password
        self._resource = resource
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending: dict[str, _Pending] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._subject_of: dict[str, str] = {}

    # ---- client registration -------------------------------------------------

    async def get_client(self, client_id: str):
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    # ---- authorization -------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Park the request and send the user to our login page."""
        self._sweep()
        key = secrets.token_urlsafe(24)
        self._pending[key] = _Pending(client_id=client.client_id, params=params)
        return f"{self._resource.rstrip('/')}/login?rk={key}"

    def complete_login(self, key: str, password: str, who: str) -> str | None:
        """
        Called by the login route. Returns the client's redirect URL with a fresh
        authorization code, or None if the passphrase was wrong or expired.
        """
        pending = self._pending.get(key)
        if pending is None:
            return None
        if not secrets.compare_digest(password or "", self._password):
            return None
        del self._pending[key]

        params = pending.params
        code = secrets.token_urlsafe(32)
        subject = (who or "team member").strip()[:64] or "team member"
        self._codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or ["read"],
            expires_at=time.time() + AUTH_CODE_TTL,
            client_id=pending.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject=subject,
        )
        sep = "&" if "?" in str(params.redirect_uri) else "?"
        url = f"{params.redirect_uri}{sep}code={code}"
        if params.state:
            url += f"&state={params.state}"
        return url

    def pending_exists(self, key: str) -> bool:
        self._sweep()
        return key in self._pending

    # ---- codes and tokens ----------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ):
        rec = self._codes.get(authorization_code)
        if rec is None or rec.client_id != client.client_id:
            return None
        if rec.expires_at < time.time():
            self._codes.pop(authorization_code, None)
            return None
        return rec

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # single use
        self._codes.pop(authorization_code.code, None)

        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        now = time.time()
        subject = authorization_code.subject or "team member"

        self._access[access] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(now + ACCESS_TOKEN_TTL),
            resource=authorization_code.resource or self._resource,
            subject=subject,
        )
        self._refresh[refresh] = RefreshToken(
            token=refresh,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=None,
            resource=authorization_code.resource or self._resource,
            subject=subject,
        )
        self._subject_of[access] = subject
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(authorization_code.scopes),
            refresh_token=refresh,
        )

    async def load_access_token(self, token: str):
        rec = self._access.get(token)
        if rec is None:
            return None
        if rec.expires_at and rec.expires_at < time.time():
            self._access.pop(token, None)
            return None
        return rec

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ):
        rec = self._refresh.get(refresh_token)
        if rec is None or rec.client_id != client.client_id:
            return None
        return rec

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self._refresh.pop(refresh_token.token, None)
        access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)
        now = time.time()
        granted = scopes or refresh_token.scopes
        subject = refresh_token.subject or "team member"

        self._access[access] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=granted,
            expires_at=int(now + ACCESS_TOKEN_TTL),
            resource=refresh_token.resource or self._resource,
            subject=subject,
        )
        self._refresh[new_refresh] = RefreshToken(
            token=new_refresh,
            client_id=client.client_id,
            scopes=granted,
            expires_at=None,
            resource=refresh_token.resource or self._resource,
            subject=subject,
        )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(granted),
            refresh_token=new_refresh,
        )

    async def revoke_token(self, token: Any) -> None:
        tok = getattr(token, "token", token)
        self._access.pop(tok, None)
        self._refresh.pop(tok, None)

    async def exchange_identity_assertion(self, client, params):  # not supported
        raise NotImplementedError

    # ---- housekeeping --------------------------------------------------------

    def _sweep(self):
        cutoff = time.time() - PENDING_TTL
        for k in [k for k, v in self._pending.items() if v.created_at < cutoff]:
            del self._pending[k]


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect to the signal-integrity library</title>
<style>
  :root {{ color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#d5d5d5; --accent:#2f6f4f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#eee; --bg:#141414; --muted:#9a9a9a; --line:#333; --accent:#7fc4a0; }}
  }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         background:var(--bg); color:var(--fg);
         font:16px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; }}
  .card {{ width:min(420px,calc(100vw - 32px)); padding:32px;
           border:1px solid var(--line); border-radius:12px; }}
  h1 {{ font-size:1.15rem; margin:0 0 4px; }}
  p {{ color:var(--muted); font-size:.9rem; margin:0 0 20px; }}
  label {{ display:block; font-size:.85rem; margin:14px 0 6px; }}
  input {{ width:100%; padding:10px 12px; font:inherit; box-sizing:border-box;
           border:1px solid var(--line); border-radius:8px;
           background:var(--bg); color:var(--fg); }}
  button {{ width:100%; margin-top:20px; padding:11px; font:inherit; font-weight:600;
            border:0; border-radius:8px; background:var(--accent); color:#fff;
            cursor:pointer; }}
  .err {{ margin-top:16px; padding:10px 12px; border-radius:8px; font-size:.85rem;
          background:rgba(190,60,60,.12); color:#c0392b; }}
  @media (prefers-color-scheme: dark) {{ .err {{ color:#ff9b8f; }} }}
</style></head>
<body><form class="card" method="post" action="/login">
  <h1>Signal-integrity library</h1>
  <p>Connect Claude to the shared textbook index.</p>
  <input type="hidden" name="rk" value="{rk}">
  <label for="who">Your name <span style="color:var(--muted)">(for the access log)</span></label>
  <input id="who" name="who" autocomplete="name" placeholder="e.g. Alex">
  <label for="pw">Team passphrase</label>
  <input id="pw" name="pw" type="password" autocomplete="current-password" required autofocus>
  <button type="submit">Connect</button>
  {error}
</form></body></html>"""
