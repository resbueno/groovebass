"""Fila do GrooveBass. Não tem processo próprio: o worker do Kicksnare chama `run_next()` quando a
música que espera há mais tempo é do GrooveBass. Com dois workers (padrão e HiFi), cada um atende só a
sua qualidade aqui também: rodam no máximo dois Demucs, um por qualidade."""
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

    from kicksnare import quality as qualitylib

    sid, source = song["id"], song["source"]
    quality = qualitylib.normalize(song.get("quality"))
    song_dir = DATA_DIR / sid
    _set(sid, "ingest", "Preparando o áudio")
    reused = pipeline.reuse_from_kicksnare(song_dir, _raw_id(sid), quality)
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
    _identify(sid, meta)

    from kicksnare.separate import needs_separation

    if not (song_dir / "bass.wav").exists() or needs_separation(song_dir, quality):
        hifi = quality == qualitylib.HIFI
        _set(sid, "separate", "Isolando o baixo em HiFi (mais demorado)" if hifi else "Isolando o baixo (alguns minutos)")
    sep = pipeline.separate(song_dir, on_progress=lambda p: db.update(sid, progress=round(p, 3)), quality=quality)
    print(f"[gb {sid}] separação {sep}", flush=True)

    _set(sid, "transcribe", "Transcrevendo o baixo e a cifra")
    stats = pipeline.transcribe(song_dir)

    _set(sid, "finalize", "Gerando áudio para o player")
    pipeline.finalize(song_dir)
    row = db.get(sid) or {}
    timings = dict(row.get("timings") or {})
    if row.get("stage_started"):
        timings["finalize"] = round(time.time() - row["stage_started"], 1)
    from kicksnare.separate import separation_quality

    db.update(sid, status="done", stage=None, message="Pronto", stats=stats, finished=time.time(), error=None,
              timings=timings, progress=1.0, quality=separation_quality(song_dir, quality))
    _copy_identity(sid)


def _copy_identity(sid: str) -> bool:
    """Banda e nome identificados no Kicksnare (Groq) valem aqui, se a banda ainda está vazia."""
    from kicksnare import db as ksdb

    ks, row = ksdb.get(sid), db.get(sid) or {}
    if ks and ks.get("artist") and not row.get("artist"):
        db.update(sid, artist=ks["artist"], **({"title": ks["title"]} if ks.get("title") else {}))
        return True
    return False


def _identify(sid: str, meta) -> None:
    """Banda e nome da música pela mesma IA do Kicksnare (Groq), uma vez por música. Se o Kicksnare já
    identificou, copia de lá sem chamar a IA; se identificar aqui primeiro, vale lá também (onde a banda
    estiver vazia). O que o usuário editou prevalece. Falha aqui não interrompe o processamento."""
    from kicksnare import db as ksdb
    from kicksnare import identify

    try:
        if _copy_identity(sid):
            return
        info = json.loads(meta.read_text()) if meta.exists() else {}
        row = db.get(sid) or {}
        if row.get("artist") or not identify.enabled():
            return
        raw = info.get("source_title") or info.get("title") or row.get("title")
        if not raw:
            return
        found = identify.identify(raw, info.get("uploader"), info.get("artist"), info.get("track"))
        fields = {}
        if found["title"] and row.get("title") in (None, raw, info.get("title")):
            fields["title"] = info["title"] = found["title"]
        if found["artist"]:
            fields["artist"] = found["artist"]
        if fields:
            db.update(sid, **fields)
            ks = ksdb.get(sid)
            if ks and not ks.get("artist"):
                ksdb.update(sid, **fields)
        info["identified"] = True
        meta.write_text(json.dumps(info))
        print(f"[gb {sid}] identificado: {raw!r} -> {found}", flush=True)
    except Exception as e:  # noqa: BLE001 - identificação é só conveniência
        print(f"[gb {sid}] identificação falhou: {e}", flush=True)


def requeue(qualities=None) -> int:
    return db.requeue_interrupted(qualities)


def oldest_queued(qualities=None) -> float | None:
    return db.oldest_queued(qualities)


def run_next(qualities=None) -> bool:
    """Processa a próxima da fila (das qualidades deste worker). False se a fila estava vazia."""
    song = db.claim_next(qualities)
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
