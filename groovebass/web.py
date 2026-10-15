"""API e páginas do GrooveBass (montado em /groovebass pelo asgi do Kicksnare).

Login: mesmos usuários e senhas do Kicksnare (tabela de usuários dele); a sessão é própria, com
cookie no caminho /groovebass. Pedidos de conta feitos aqui aparecem para aprovação no admin do
Kicksnare. Cada usuário vê as próprias músicas e as que outros compartilharam (só leitura).
"""
import contextvars
import hashlib
import json
import re
import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kicksnare import auth as authlib
from kicksnare import db as ksdb
from kicksnare import eta as etalib
from kicksnare import sections as kssections
from kicksnare.ingest import file_song_id
from kicksnare.separate import encode_mp3

from . import db, pipeline
from .config import BASE, DATA_DIR, MAX_ACTIVE, MAX_QUEUE_TOTAL, MAX_SONGS, MAX_UPLOAD, MIN_FREE_GB, UPLOADS

STATIC = Path(__file__).parent / "static"
SESSION_COOKIE = "gb_session"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
YT_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")
MAX_PART_LABEL = 40


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC / name).read_text(encoding="utf-8").replace("__BASE__", BASE))


# ---------- sessão ----------
_user_var: contextvars.ContextVar = contextvars.ContextVar("gb_user", default=None)


class LoginRedirect(Exception):
    pass


def _rel_path(request: Request) -> str:
    path = request.url.path
    if BASE and path.startswith(BASE):
        path = path[len(BASE):]
    return path or "/"


def _client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


def _is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"


PUBLIC_PATHS = {"/login", "/favicon.ico"}


async def auth_dep(request: Request) -> None:
    path = _rel_path(request)
    tok = request.cookies.get(SESSION_COOKIE)
    user = ksdb.user_for_session(authlib.token_hash(tok)) if tok else None
    _user_var.set(user)
    if user or path in PUBLIC_PATHS or path.startswith("/api/auth/") or path.startswith("/static/"):
        return
    if path.startswith("/api/"):
        raise HTTPException(401, "faça login")
    raise LoginRedirect()


def _require_user() -> dict:
    user = _user_var.get()
    if not user:
        raise HTTPException(401, "faça login")
    return user


app = FastAPI(title="GrooveBass", dependencies=[Depends(auth_dep)], docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(LoginRedirect)
async def _login_redirect(request: Request, exc: LoginRedirect) -> RedirectResponse:
    return RedirectResponse(f"{BASE}/login", status_code=303)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/login", include_in_schema=False)
def login_page() -> HTMLResponse:
    return _page("login.html")


@app.get("/songs/{song_id}", include_in_schema=False)
def song_page(song_id: str) -> HTMLResponse:
    _song(song_id)
    return _page("song.html")


# ---------- contas (usuários do Kicksnare) ----------
class Credentials(BaseModel):
    username: str
    password: str


class Signup(Credentials):
    first_name: str = ""
    last_name: str = ""
    email: str = ""


@app.post("/api/auth/login")
def login(body: Credentials, request: Request, response: Response) -> dict:
    username = authlib.normalize_username(body.username)
    key = f"gb|{_client_ip(request)}|{username}"
    if authlib.login_failures.blocked(key):
        raise HTTPException(429, "muitas tentativas; aguarde alguns minutos")
    user = ksdb.get_user_by_name(username)
    if not user or not authlib.verify_password(body.password, user["pw_hash"]):
        if not user:
            authlib.burn_time(body.password)     # mesmo custo: não revela se o usuário existe
        authlib.login_failures.hit(key)
        raise HTTPException(401, "usuário ou senha incorretos")
    authlib.login_failures.clear(key)
    # o estado da conta só é revelado a quem sabe a senha
    if user.get("status", "approved") == "pending":
        raise HTTPException(403, "Sua conta está aguardando a aprovação do administrador.")
    if user.get("status") in ("rejected", "blocked"):
        raise HTTPException(403, "Esta conta não está autorizada a acessar.")
    tok, th = authlib.new_token()
    ksdb.create_session(user["id"], th, authlib.SESSION_DAYS)
    response.set_cookie(SESSION_COOKIE, tok, max_age=authlib.SESSION_DAYS * 86400, httponly=True,
                        samesite="lax", secure=_is_https(request), path=BASE or "/")
    return {"username": user["username"]}


@app.post("/api/auth/register")
def register(body: Signup, request: Request) -> dict:
    ip = _client_ip(request)
    if authlib.signups.blocked(ip):
        raise HTTPException(429, "muitos cadastros deste endereço; tente mais tarde")
    username = authlib.normalize_username(body.username)
    first, last, email = body.first_name.strip(), body.last_name.strip(), body.email.strip().lower()
    err = (authlib.validate_profile(first, last, email) or authlib.validate_username(username)
           or authlib.validate_password(body.password))
    if err:
        raise HTTPException(400, err)
    if ksdb.email_in_use(email):
        raise HTTPException(409, "este e-mail já tem uma conta")
    uid = ksdb.create_user(username, authlib.hash_password(body.password), status="pending",
                           first_name=first, last_name=last, email=email)
    if uid is None:
        raise HTTPException(409, "este usuário já existe")
    authlib.signups.hit(ip)
    return {"username": username, "pending": True,
            "message": "Cadastro enviado. Você poderá entrar assim que o administrador aprovar "
                       "(a mesma conta vale para o Kicksnare)."}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict:
    tok = request.cookies.get(SESSION_COOKIE)
    if tok:
        ksdb.delete_session(authlib.token_hash(tok))
    response.delete_cookie(SESSION_COOKIE, path=BASE or "/")
    return {"ok": True}


@app.get("/api/auth/me")
def me() -> dict:
    u = _require_user()
    return {"username": u["username"], "is_admin": bool(u.get("is_admin"))}


# ---------- músicas ----------
def _song(song_id: str, write: bool = False) -> dict:
    user = _require_user()
    if not ID_RE.match(song_id):
        raise HTTPException(404, "música não encontrada")
    song = db.get(song_id)
    if not song:
        raise HTTPException(404, "música não encontrada")
    can_edit = song.get("owner_id") == user["id"] or bool(user.get("is_admin"))
    if not can_edit:
        if not (song.get("shared") and song["status"] == "done"):
            raise HTTPException(404, "música não encontrada")
        if write:
            raise HTTPException(403, "só quem enviou a música pode alterá-la")
    return {**_fresh(song), "can_edit": can_edit}


def _fresh(song: dict) -> dict:
    f = DATA_DIR / song["id"] / "stats.json"
    if song.get("status") == "done" and f.exists():
        try:
            song = {**song, "stats": json.loads(f.read_text())}
        except (OSError, ValueError):
            pass
    return song


def _ready(song_id: str, write: bool = False) -> Path:
    song = _song(song_id, write)
    if song["status"] != "done":
        raise HTTPException(409, "a música ainda não foi processada")
    return DATA_DIR / song_id


# Biblioteca compartilhada com o Kicksnare: o que entra aqui entra lá (mesmo id e dono); nome, banda e
# compartilhamento acompanham. Apagar é independente em cada sistema.
def _mirror_enqueue(song_id: str, source: str, title: str | None, owner_id: int) -> None:
    if not ksdb.get(song_id):
        ksdb.enqueue(song_id, source, title=title, owner_id=owner_id)


def _mirror_update(song_id: str, **fields) -> None:
    if ksdb.get(song_id):
        ksdb.update(song_id, **fields)


def _scoped_id(user: dict, raw: str) -> str:
    return f"u{user['id']}-{raw}"


def _check_quota(user: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DATA_DIR).free < MIN_FREE_GB * 1024 ** 3:
        raise HTTPException(507, "servidor sem espaço disponível no momento")
    if user.get("is_admin"):
        return
    if db.active_total() >= MAX_QUEUE_TOTAL:
        raise HTTPException(503, "a fila de processamento está cheia; tente novamente em alguns minutos")
    total, active = db.song_counts(user["id"])
    if active >= MAX_ACTIVE:
        raise HTTPException(429, f"você já tem {active} músicas em processamento; aguarde terminarem")
    if total >= MAX_SONGS:
        raise HTTPException(409, f"limite de {MAX_SONGS} músicas; apague alguma para enviar outra")


@app.get("/api/songs")
def list_songs(mine: bool = False) -> list[dict]:
    user = _require_user()
    admin = bool(user.get("is_admin"))
    owner = user["id"] if (mine or not admin) else None
    songs = [{**_fresh(s), "can_edit": admin or s.get("owner_id") == user["id"]}
             for s in db.list_songs(owner, with_shared=not mine)]
    if any(s["status"] in ("queued", "running") for s in songs):
        etalib.annotate(songs, db.active_songs(), db.history())
    return songs


class NewSong(BaseModel):
    url: str


@app.post("/api/songs")
def add_song(body: NewSong) -> dict:
    user = _require_user()
    url = body.url.strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "informe uma URL http(s)")
    m = YT_RE.search(url)
    raw = m.group(1) if m else "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]
    song_id = _scoped_id(user, raw)
    existing = db.get(song_id)
    if not (existing and existing.get("owner_id") == user["id"] and existing["status"] in ("queued", "running", "done")):
        _check_quota(user)
    song = db.enqueue(song_id, url, owner_id=user["id"])
    _mirror_enqueue(song_id, url, None, user["id"])
    return song


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    user = _require_user()
    _check_quota(user)
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(file.filename or "audio").name)[:120]
    folder = UPLOADS / str(user["id"])
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "arquivo grande demais (máx. 300 MB)")
            out.write(chunk)
    song_id = _scoped_id(user, file_song_id(dest))
    song = db.enqueue(song_id, str(dest), title=Path(name).stem, owner_id=user["id"])
    _mirror_enqueue(song_id, str(dest), Path(name).stem, user["id"])
    return song


@app.get("/api/songs/{song_id}")
def get_song(song_id: str) -> dict:
    song = _song(song_id)
    if song["status"] in ("queued", "running"):
        etalib.annotate([song], db.active_songs(), db.history())
    owner = ksdb.get_user(song["owner_id"]) if song.get("owner_id") else None
    return {**song, "owner": owner["username"] if owner else None}


class Rename(BaseModel):
    title: str | None = None
    artist: str | None = None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", text or "")).strip()


@app.patch("/api/songs/{song_id}")
def rename_song(song_id: str, body: Rename) -> dict:
    song = _song(song_id, write=True)
    fields: dict = {}
    if body.title is not None:
        title = _clean(body.title)
        if not title or len(title) > 120:
            raise HTTPException(400, "o nome deve ter de 1 a 120 caracteres")
        fields["title"] = title
    if body.artist is not None:
        artist = _clean(body.artist)
        if len(artist) > 120:
            raise HTTPException(400, "nome da banda longo demais (máximo de 120 caracteres)")
        fields["artist"] = artist or None
    if fields:
        db.update(song_id, **fields)
        _mirror_update(song_id, **fields)
        if fields.get("title") and song["status"] == "done":
            meta = DATA_DIR / song_id / "meta.json"
            try:
                info = json.loads(meta.read_text()) if meta.exists() else {}
                info["title"] = fields["title"]
                meta.write_text(json.dumps(info))
                pipeline.write_scores(DATA_DIR / song_id)
            except (OSError, ValueError, KeyError):
                pass
    return _song(song_id)


@app.delete("/api/songs/{song_id}")
def delete_song(song_id: str) -> dict:
    song = _song(song_id, write=True)
    if song["status"] == "running":
        raise HTTPException(409, "está em processamento; aguarde terminar")
    db.delete_song(song_id)
    shutil.rmtree(DATA_DIR / song_id, ignore_errors=True)
    src = Path(song["source"])
    try:
        if src.is_file() and UPLOADS in src.resolve().parents:
            src.unlink()
    except OSError:
        pass
    return {"deleted": song_id}


@app.post("/api/songs/{song_id}/retry")
def retry(song_id: str) -> dict:
    song = _song(song_id, write=True)
    if song["status"] == "running":
        raise HTTPException(409, "já está em processamento")
    db.update(song_id, status="queued", message="Na fila", error=None, stage=None)
    return db.get(song_id)


@app.post("/api/songs/{song_id}/reprocess")
def reprocess(song_id: str) -> dict:
    """Refaz a transcrição com o algoritmo atual (o baixo já separado e as partes são mantidos)."""
    song = _song(song_id, write=True)
    if song["status"] in ("queued", "running"):
        raise HTTPException(409, "já está na fila ou em processamento")
    db.update(song_id, status="queued", message="Na fila para reprocessar", error=None, stage=None, progress=0.0)
    return db.get(song_id)


class Share(BaseModel):
    shared: bool


@app.post("/api/songs/{song_id}/share")
def share_song(song_id: str, body: Share) -> dict:
    _song(song_id, write=True)
    db.update(song_id, shared=1 if body.shared else 0)
    _mirror_update(song_id, shared=1 if body.shared else 0)
    return get_song(song_id)


# ---------- arquivos ----------
def _download_name(song_id: str, suffix: str, ext: str) -> str:
    title = (_song(song_id).get("title") or song_id)
    title = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title).strip()[:80] or song_id
    return f"{title} - {suffix}.{ext}"


@app.get("/api/songs/{song_id}/score.musicxml")
def score(song_id: str, strings: int = 4, download: bool = False) -> FileResponse:
    if strings not in (4, 5):
        raise HTTPException(400, "cordas: 4 ou 5")
    f = _ready(song_id) / f"score{strings}.musicxml"
    if not f.exists():
        raise HTTPException(404, "partitura indisponível")
    if download:
        return FileResponse(f, media_type="application/vnd.recordare.musicxml+xml",
                            filename=_download_name(song_id, f"baixo {strings} cordas", "musicxml"))
    return FileResponse(f, media_type="application/xml")


TRACKS = {"mix": ("audio.wav", "audio.mp3"), "bass": ("bass.wav", "bass.mp3"), "no_bass": ("no_bass.wav", "no_bass.mp3")}


@app.get("/api/songs/{song_id}/track/{track}.mp3")
def track_audio(song_id: str, track: str, download: bool = False) -> FileResponse:
    if track not in TRACKS:
        raise HTTPException(404, "faixa inexistente")
    song_dir = _ready(song_id)
    wav, mp3 = (song_dir / n for n in TRACKS[track])
    if not mp3.exists():
        if not wav.exists():
            raise HTTPException(404, "faixa indisponível")
        encode_mp3(wav, mp3)
    if download:
        label = {"mix": "original", "bass": "so o baixo", "no_bass": "sem o baixo"}[track]
        return FileResponse(mp3, media_type="audio/mpeg", filename=_download_name(song_id, label, "mp3"))
    return FileResponse(mp3, media_type="audio/mpeg")


@app.get("/api/songs/{song_id}/timing")
def timing(song_id: str) -> dict:
    song_dir = _ready(song_id)
    beats = json.loads((song_dir / "beats.json").read_text())["beats"]
    bars = json.loads((song_dir / "bars.json").read_text())
    return {"beats": beats, "bars": bars}


@app.get("/api/songs/{song_id}/chords")
def chords(song_id: str) -> dict:
    """Cifra por compasso, para a vista de cifra: [{"bar", "chords": [{"slot", "label"}]}] + tom e partes."""
    song_dir = _ready(song_id)
    harm = json.loads((song_dir / "harmony.json").read_text())
    stats = json.loads((song_dir / "stats.json").read_text())
    bars = json.loads((song_dir / "bars.json").read_text())
    by_bar: dict = {}
    for c in harm["chords"]:
        by_bar.setdefault(c["bar"], []).append({"slot": c["slot"], "label": c["label"]})
    return {"key": stats.get("key"), "subdiv": stats.get("subdiv", 4),
            "bars": [{"bar": b["n"], "beats": b["beats"], "chords": by_bar.get(b["n"], [])} for b in bars],
            "sections": pipeline.current_sections(song_dir)}


# ---------- partes ----------
def _sections_state(song_dir: Path) -> dict:
    data = kssections.load(song_dir)
    return {"sections": kssections.current(song_dir), "auto": data.get("auto") or [],
            "edited": bool(data.get("manual"))}


@app.get("/api/songs/{song_id}/sections")
def get_sections(song_id: str) -> dict:
    return _sections_state(_ready(song_id))


class PartItem(BaseModel):
    bar: int
    label: str


class PartsBody(BaseModel):
    sections: list[PartItem]


@app.put("/api/songs/{song_id}/sections")
def put_sections(song_id: str, body: PartsBody) -> dict:
    song_dir = _ready(song_id, write=True)
    total = len(json.loads((song_dir / "bars.json").read_text()))
    items: dict = {}
    for p in body.sections:
        label = _clean(p.label)
        if not label:
            raise HTTPException(400, "toda parte precisa de um nome")
        if len(label) > MAX_PART_LABEL:
            raise HTTPException(400, f"nome de parte longo demais (máximo de {MAX_PART_LABEL} caracteres)")
        if not 1 <= p.bar <= total:
            raise HTTPException(400, f"compasso {p.bar} fora da música (1 a {total})")
        items[p.bar] = label
    kssections.save_manual(song_dir, [{"bar": b, "label": items[b]} for b in sorted(items)] or None)
    pipeline.write_scores(song_dir)
    return _sections_state(song_dir)


@app.delete("/api/songs/{song_id}/sections/manual")
def reset_sections(song_id: str) -> dict:
    song_dir = _ready(song_id, write=True)
    kssections.save_manual(song_dir, None)
    pipeline.write_scores(song_dir)
    return _sections_state(song_dir)
