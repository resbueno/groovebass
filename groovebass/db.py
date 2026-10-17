"""Fila e músicas do GrooveBass em SQLite próprio. Os usuários e as sessões são os do Kicksnare
(mesmo login): ficam no banco dele e são lidos por kicksnare.db. Como no Kicksnare, há uma fila por
qualidade da separação (padrão e HiFi), atendidas pelos mesmos workers dele."""
import json
import sqlite3
import time
from contextlib import contextmanager

from kicksnare import db as ksdb
from kicksnare.quality import STANDARD, normalize, sql_filter as _in

from .config import DATA_DIR

DB_PATH = DATA_DIR / "groovebass.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    artist        TEXT,
    source        TEXT NOT NULL,          -- URL ou caminho do upload
    status        TEXT NOT NULL,          -- queued | running | done | error
    stage         TEXT,                   -- ingest | separate | transcribe | finalize
    message       TEXT,
    error         TEXT,
    created       REAL NOT NULL,
    started       REAL,
    finished      REAL,
    stats         TEXT,
    owner_id      INTEGER,
    shared        INTEGER NOT NULL DEFAULT 0,
    duration      REAL,
    stage_started REAL,
    timings       TEXT,
    progress      REAL,
    quality       TEXT NOT NULL DEFAULT 'standard'   -- standard | hifi (modelo do Demucs)
)
"""


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    ksdb.wal(con)                   # web e dois workers abrem o banco ao mesmo tempo
    con.execute(SCHEMA)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(songs)")}
    if "quality" not in cols:       # bancos anteriores às duas qualidades: tudo era padrão
        ksdb.add_column(con, "songs", "quality", "TEXT NOT NULL DEFAULT 'standard'")
    try:
        yield con
    finally:
        con.close()


def _row(r) -> dict:
    d = dict(r)
    d["stats"] = json.loads(d["stats"]) if d.get("stats") else None
    d["timings"] = json.loads(d["timings"]) if d.get("timings") else {}
    return d


def _with_owner(rows: list[dict]) -> list[dict]:
    """Nome do dono (`owner`), lido da tabela de usuários do Kicksnare."""
    names = {}
    for r in rows:
        oid = r.get("owner_id")
        if oid and oid not in names:
            u = ksdb.get_user(oid)
            names[oid] = u["username"] if u else None
        r["owner"] = names.get(oid)
    return rows


def enqueue(song_id: str, source: str, title: str | None = None, owner_id: int | None = None,
            quality: str = STANDARD) -> dict:
    """Cria a música na fila. Se já existe e está pronta/em andamento, não duplica."""
    with connect() as con:
        cur = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        if cur and cur["status"] in ("queued", "running", "done"):
            return _row(cur)
        con.execute(
            "INSERT OR REPLACE INTO songs(id,title,source,status,stage,message,created,owner_id,quality) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (song_id, title, source, "queued", None, "Na fila", time.time(), owner_id, normalize(quality)),
        )
        return _row(con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone())


def _fair_queue(con, qualities=None) -> list:
    """Na fila, alternando entre os donos (como no Kicksnare): vai à frente quem está há mais tempo
    sem ter uma música iniciada; entre as do mesmo dono vale a ordem de chegada."""
    where, args = _in(qualities)
    queued = con.execute(f"SELECT * FROM songs WHERE status='queued' AND {where} ORDER BY created", args).fetchall()
    last = {r["owner_id"]: r["last"] for r in con.execute(
        "SELECT owner_id, MAX(started) AS last FROM songs WHERE started IS NOT NULL GROUP BY owner_id")}
    pending: dict = {}
    for r in queued:
        pending.setdefault(r["owner_id"], []).append(r)
    order, now = [], time.time()
    while pending:
        owner = min(pending, key=lambda o: (last.get(o) or 0.0, pending[o][0]["created"]))
        order.append(pending[owner].pop(0))
        if not pending[owner]:
            del pending[owner]
        last[owner] = now + len(order)
    return order


def oldest_queued(qualities=None) -> float | None:
    """Quando chegou a música que espera há mais tempo (o worker usa para alternar com o Kicksnare)."""
    where, args = _in(qualities)
    with connect() as con:
        return con.execute(f"SELECT MIN(created) FROM songs WHERE status='queued' AND {where}", args).fetchone()[0]


def claim_next(qualities=None) -> dict | None:
    """Reserva a próxima das qualidades deste worker (transação: dois workers nunca pegam a mesma)."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        queue = _fair_queue(con, qualities)
        r = queue[0] if queue else None
        if not r:
            con.execute("COMMIT")
            return None
        con.execute("UPDATE songs SET status='running', started=?, message='Iniciando' WHERE id=?",
                    (time.time(), r["id"]))
        con.execute("COMMIT")
        return _row(r)


def update(song_id: str, **fields) -> None:
    for k in ("stats", "timings"):
        if k in fields and fields[k] is not None:
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE songs SET {cols} WHERE id=?", (*fields.values(), song_id))


def get(song_id: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        return _row(r) if r else None


def list_songs(owner_id: int | None = None, with_shared: bool = False) -> list[dict]:
    """Do dono pedido (e, com with_shared, as prontas que outros compartilharam); owner_id=None = todas."""
    with connect() as con:
        if owner_id is None:
            rows = con.execute("SELECT * FROM songs ORDER BY created DESC")
        elif with_shared:
            rows = con.execute("SELECT * FROM songs WHERE owner_id=? OR (shared=1 AND status='done') "
                               "ORDER BY created DESC", (owner_id,))
        else:
            rows = con.execute("SELECT * FROM songs WHERE owner_id=? ORDER BY created DESC", (owner_id,))
        return _with_owner([_row(r) for r in rows])


def song_counts(owner_id: int) -> tuple[int, int]:
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM songs WHERE owner_id=?", (owner_id,)).fetchone()[0]
        active = con.execute("SELECT COUNT(*) FROM songs WHERE owner_id=? AND status IN ('queued','running')",
                             (owner_id,)).fetchone()[0]
        return total, active


def active_total() -> int:
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM songs WHERE status IN ('queued','running')").fetchone()[0]


def active_songs(qualities=None) -> list[dict]:
    """Fila inteira das qualidades pedidas, na ordem em que o worker as pega (base da estimativa)."""
    where, args = _in(qualities)
    with connect() as con:
        running = con.execute(f"SELECT * FROM songs WHERE status='running' AND {where} ORDER BY started", args).fetchall()
        return [_row(r) for r in [*running, *_fair_queue(con, qualities)]]


def history(limit: int = 15, quality: str | None = None) -> list[dict]:
    where, args = _in(quality)
    with connect() as con:
        rows = con.execute("SELECT duration, timings, quality FROM songs WHERE status='done' AND duration > 0 "
                           f"AND timings IS NOT NULL AND {where} ORDER BY finished DESC LIMIT ?", (*args, limit))
        return [_row(r) for r in rows]


def delete_song(song_id: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM songs WHERE id=?", (song_id,))


def requeue_interrupted(qualities=None) -> int:
    """Só as das qualidades deste worker: as do outro worker podem estar rodando agora."""
    where, args = _in(qualities)
    with connect() as con:
        return con.execute("UPDATE songs SET status='queued', message='Reenfileirado após reinício' "
                           f"WHERE status='running' AND {where}", args).rowcount
