"""Fila do GrooveBass. Não tem processo próprio: o worker do Kicksnare chama `run_next()` quando a
música que espera há mais tempo é do GrooveBass, então nunca rodam dois Demucs ao mesmo tempo."""
import json
import re
import time
import traceback

from . import db, pipeline
from .config import DATA_DIR


def _set(song_id: str, stage: str, message: str) -> None:
    row = db.get(song_id) or {}
    timings = dict(row.get("timings") or {})
    prev, since = row.get("stage"), row.get("stage_started")
    if prev and since:
        timings[prev] = round(time.time() - since, 1)
    db.update(song_id, stage=stage, message=message, stage_started=time.time(), progress=0.0, timings=timings)


def _raw_id(song_id: str) -> str:
    """'u3-AT-l-3SRgB0' -> 'AT-l-3SRgB0' (o mesmo vídeo no Kicksnare tem o mesmo final)."""
    return re.sub(r"^u\d+-", "", song_id)


def process(song: dict) -> None:
    from kicksnare.ingest import ingest
    from kicksnare.worker import _duration

    sid, source = song["id"], song["source"]
    song_dir = DATA_DIR / sid
    _set(sid, "ingest", "Preparando o áudio")
    reused = pipeline.reuse_from_kicksnare(song_dir, _raw_id(sid))
    if reused:
        print(f"[gb {sid}] reaproveitado do Kicksnare: {reused}", flush=True)
    if not (song_dir / "audio.wav").exists():
        db.update(sid, message="Baixando áudio" if source.startswith("http") else "Preparando áudio")
        ingest(source, song_id=sid, data_dir=DATA_DIR)
    meta = song_dir / "meta.json"
    row = db.get(sid) or {}
    if meta.exists() and not row.get("title"):
        db.update(sid, title=json.loads(meta.read_text()).get("title"))
    db.update(sid, duration=_duration(song_dir / "audio.wav"))

    if not (song_dir / "bass.wav").exists():
        _set(sid, "separate", "Isolando o baixo (alguns minutos)")
    sep = pipeline.separate(song_dir, on_progress=lambda p: db.update(sid, progress=round(p, 3)))
    print(f"[gb {sid}] separação {sep}", flush=True)

    _set(sid, "transcribe", "Transcrevendo o baixo e a cifra")
    stats = pipeline.transcribe(song_dir)

    _set(sid, "finalize", "Gerando áudio para o player")
    pipeline.finalize(song_dir)
    row = db.get(sid) or {}
    timings = dict(row.get("timings") or {})
    if row.get("stage_started"):
        timings["finalize"] = round(time.time() - row["stage_started"], 1)
    db.update(sid, status="done", stage=None, message="Pronto", stats=stats, finished=time.time(), error=None,
              timings=timings, progress=1.0)


def requeue() -> int:
    return db.requeue_interrupted()


def oldest_queued() -> float | None:
    return db.oldest_queued()


def run_next() -> bool:
    """Processa a próxima da fila. False se a fila estava vazia."""
    song = db.claim_next()
    if not song:
        return False
    print(f"[gb {song['id']}] iniciando", flush=True)
    try:
        process(song)
        print(f"[gb {song['id']}] pronto", flush=True)
    except Exception as e:  # noqa: BLE001 - qualquer falha vira erro visível na interface
        traceback.print_exc()
        db.update(song["id"], status="error", message="Falhou", error=str(e)[:2000], finished=time.time())
    return True
