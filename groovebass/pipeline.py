"""Processamento de uma música: áudio -> baixo separado -> notas -> compassos -> tom, cifra e partes
-> tablatura de 4 e de 5 cordas -> score4.musicxml e score5.musicxml.

Arquivos na pasta da música (DATA_DIR/<id>):
  audio.wav, bass.wav, no_bass.wav, drums.wav, no_drums.wav   áudio e faixas
  beats.json, bars.json                                       tempos e compassos (formato do Kicksnare)
  notes.json        compassos com as notas quantizadas
  harmony.json      tom e acordes (com compasso/slot)
  sections.json     partes {"auto": [...], "manual": [...] | null}
  score4.musicxml, score5.musicxml
"""
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

from kicksnare import sections as kssections
from kicksnare.ingest import DATA_DIR as KS_DATA
from kicksnare.separate import encode_mp3, needs_separation, separate_drums

from . import harmony, notes as notelib, rhythm, score, tab
from .config import DATA_DIR

REUSE_FILES = ("audio.wav", "meta.json", "bass.wav", "drums.wav", "no_drums.wav", "separation.json",
               "beats.json", "bars.json")


# ---------- reaproveitamento do Kicksnare ----------
def kicksnare_twin(song_id: str, raw_id: str) -> Path | None:
    """Pasta da mesma música no Kicksnare: a do mesmo dono (mesmo id) primeiro, depois a de qualquer
    usuário que já tenha o baixo separado."""
    cands = [p for p in KS_DATA.glob(f"u*-{raw_id}") if (p / "audio.wav").exists()]
    cands.sort(key=lambda p: (p.name == song_id, (p / "bass.wav").exists(), (p / "bars.json").exists()), reverse=True)
    return cands[0] if cands else None


def reuse_from_kicksnare(song_dir: Path, raw_id: str) -> dict:
    twin = kicksnare_twin(song_dir.name, raw_id)
    if not twin:
        return {}
    song_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REUSE_FILES:
        src = twin / name
        if src.exists() and not (song_dir / name).exists():
            shutil.copyfile(src, song_dir / name)
            copied.append(name)
    # partes: se os compassos são os mesmos, valem as do Kicksnare (inclusive as corrigidas lá)
    if (twin / "bars.json").exists() and "bars.json" in copied:
        current = kssections.current(twin)
        if current:
            (song_dir / "sections.json").write_text(json.dumps({"auto": current, "manual": None}, ensure_ascii=False))
            copied.append("sections")
    return {"from": twin.name, "copied": copied}


# ---------- etapas ----------
def separate(song_dir: Path, on_progress=None) -> dict:
    audio = song_dir / "audio.wav"
    if not (song_dir / "bass.wav").exists():
        # separação antiga (só bateria) ou nenhuma: refaz com as 4 faixas
        for name in ("drums.wav", "separation.json"):
            (song_dir / name).unlink(missing_ok=True)
    info = {"cached": True}
    if needs_separation(song_dir):
        _, info = separate_drums(audio, on_progress=on_progress)
    _no_bass(song_dir)
    return info


def _no_bass(song_dir: Path) -> None:
    """Música sem o baixo = mix - baixo (as faixas do Demucs somam o mix)."""
    import soundfile as sf

    out = song_dir / "no_bass.wav"
    if out.exists():
        return
    mix, sr = sf.read(str(song_dir / "audio.wav"), dtype="float32")
    bass, _ = sf.read(str(song_dir / "bass.wav"), dtype="float32")
    n = min(len(mix), len(bass))
    sf.write(str(out), np.clip(mix[:n] - bass[:n], -1.0, 1.0), sr, subtype="PCM_16")


def transcribe(song_dir: Path) -> dict:
    if not ((song_dir / "beats.json").exists() and (song_dir / "bars.json").exists()):
        rhythm.compute_grid(song_dir)
    beats, bars, bpb = rhythm.load_grid(song_dir)
    found, info = notelib.transcribe(song_dir / "bass.wav")
    measures, subdiv = rhythm.quantize(found, beats, bars)

    # nota do baixo em cada tempo (para a cifra): a que soa no tempo ou começa na 1ª metade dele
    bass_pc = []
    for i in range(len(beats) - 1):
        t0, t1 = beats[i], beats[i + 1]
        mid = t0 + (t1 - t0) / 2
        pc = None
        for n in found:
            if n.start <= t0 + 0.03 < n.end or t0 <= n.start < mid:
                pc = n.midi % 12
                break
            if n.start > mid:
                break
        bass_pc.append(pc)
    key, chords = harmony.analyze(song_dir / "no_bass.wav", beats, bass_pc)

    starts = rhythm.bar_beat_index(beats, bars)
    placed = []
    for c in chords:
        k = int(np.searchsorted(starts, c["beat"], side="right") - 1)
        if k < 0 or k >= len(bars):
            continue
        placed.append({**c, "bar": k + 1, "slot": (c["beat"] - starts[k]) * subdiv})

    (song_dir / "notes.json").write_text(json.dumps({"subdiv": subdiv, "measures": measures}))
    (song_dir / "harmony.json").write_text(json.dumps({"key": key, "chords": placed}, ensure_ascii=False))

    if not kssections.load(song_dir).get("auto"):
        _write_ks_measures(song_dir, measures, bars)
        auto = kssections.detect(song_dir)
        (song_dir / "sections.json").write_text(json.dumps({"auto": auto, "manual": None}, ensure_ascii=False))

    bpm = 60.0 / statistics.median(np.diff(beats)) if len(beats) > 1 else 120.0
    names = harmony.names_for(key["fifths"])
    stats = {"bars": len(bars), "bpm_median": round(bpm, 1), "subdiv": subdiv, "bpb": bpb,
             "key": names[key["tonic"]] + ("m" if key["mode"] == "minor" else ""), "key_fifths": key["fifths"],
             "chords": len(placed), **info}
    (song_dir / "stats.json").write_text(json.dumps(stats))
    write_scores(song_dir)
    return stats


def _write_ks_measures(song_dir: Path, measures: list[dict], bars: list[dict]) -> None:
    """measures.json no formato do Kicksnare, para o detector de partes dele agrupar compassos pelo
    desenho do baixo (classe de altura + posição) em vez do groove da bateria."""
    rows = []
    for m, b in zip(measures, bars):
        slots: dict = {}
        for n in m["notes"]:
            if not n["tie_in"]:
                slots.setdefault(str(n["slot"]), []).append(100 + n["midi"] % 12)
        rows.append({"beats": m["beats"], "subdiv": m["subdiv"], "start": b["start"], "slots": slots})
    (song_dir / "measures.json").write_text(json.dumps(rows))


def current_sections(song_dir: Path) -> list[dict]:
    return kssections.current(song_dir)


def write_scores(song_dir: Path) -> None:
    """(Re)grava score4 e score5 a partir de notes.json, harmony.json e sections.json."""
    data = json.loads((song_dir / "notes.json").read_text())
    measures = data["measures"]
    harm = json.loads((song_dir / "harmony.json").read_text())
    stats = json.loads((song_dir / "stats.json").read_text())
    meta = song_dir / "meta.json"
    title = (json.loads(meta.read_text()).get("title") if meta.exists() else None) or song_dir.name
    labels = {s["bar"] - 1: s["label"] for s in current_sections(song_dir)}
    chords_by_bar: dict = {}
    for c in harm["chords"]:
        chords_by_bar.setdefault(c["bar"] - 1, []).append((c["slot"], c))

    # sequência de notas (cabeças, sem as continuações ligadas) para escolher corda/casa
    heads = []
    for k, m in enumerate(measures):
        for i, n in enumerate(sorted(m["notes"], key=lambda n: n["slot"])):
            heads.append((k, i, n))
    beats, bars, _ = rhythm.load_grid(song_dir)
    starts = rhythm.bar_beat_index(beats, bars)
    sd = data["subdiv"]

    def t(k, slot):
        q = starts[k] + slot / sd
        i = min(int(q), len(beats) - 2)
        return beats[i] + (q - i) * (beats[i + 1] - beats[i])

    for n_strings in (4, 5):
        seq, owners = [], []
        for k, i, n in heads:
            if n["tie_in"]:
                continue
            seq.append((t(k, n["slot"]), t(k, n["slot"] + n["len"]), n["midi"]))
            owners.append((k, i))
        chosen = tab.assign(seq, n_strings)
        tabs = [[None] * len(m["notes"]) for m in measures]
        last = None
        for k, i, n in heads:                    # continuação ligada repete a corda/casa da nota
            if n["tie_in"] and last is not None:
                tabs[k][i] = last
                continue
            last = chosen[owners.index((k, i))] if (k, i) in owners else None
            tabs[k][i] = last
        score.write(song_dir / f"score{n_strings}.musicxml", measures, tabs, chords_by_bar, labels,
                    harm["key"]["fifths"], stats["bpm_median"], title, n_strings)


def finalize(song_dir: Path) -> None:
    for wav, mp3 in (("audio.wav", "audio.mp3"), ("bass.wav", "bass.mp3"), ("no_bass.wav", "no_bass.mp3")):
        if not (song_dir / mp3).exists():
            encode_mp3(song_dir / wav, song_dir / mp3)
