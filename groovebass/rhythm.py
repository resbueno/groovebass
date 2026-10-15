"""Tempos, compassos e quantização das notas do baixo.

Os tempos e o início de cada compasso vêm do mesmo método do Kicksnare (beat_this + backbeat da
bateria separada). Se a música já passou pelo Kicksnare, beats.json e bars.json são copiados de lá e
as duas partituras ficam com os mesmos compassos.
"""
import json
from pathlib import Path

import numpy as np

from kicksnare.transcribe import (
    Hit, bar_phase, bar_starts_from_phase, beats_per_bar, choose_subdiv, clean_beats, detect_beats,
    detect_hits, onset_offset,
)


def compute_grid(song_dir: Path) -> None:
    """Escreve beats.json e bars.json no formato do Kicksnare."""
    hits, _ = detect_hits(song_dir / "drums.wav")
    beats, downbeats = detect_beats(song_dir / "audio.wav")
    beats, _ = clean_beats(beats)
    bpb = beats_per_bar(beats, downbeats)
    offset = onset_offset(hits, beats)
    grid_hits = [Hit(h.time - offset, h.pitch) for h in hits]
    starts = bar_starts_from_phase(bar_phase(grid_hits, beats, downbeats, bpb), bpb)
    bars = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else min(s + bpb, len(beats) - 1)
        if e - s >= 1:
            bars.append({"n": len(bars) + 1, "start": round(beats[s], 4), "beats": e - s})
    (song_dir / "beats.json").write_text(json.dumps({"beats": beats, "downbeats": downbeats, "beats_per_bar": bpb}))
    (song_dir / "bars.json").write_text(json.dumps(bars))


def load_grid(song_dir: Path) -> tuple[list[float], list[dict], int]:
    beats = json.loads((song_dir / "beats.json").read_text())
    bars = json.loads((song_dir / "bars.json").read_text())
    return beats["beats"], bars, beats.get("beats_per_bar") or 4


def bar_beat_index(beats: list[float], bars: list[dict]) -> list[int]:
    """Índice (em `beats`) do tempo 1 de cada compasso."""
    arr = np.asarray(beats)
    return [int(np.argmin(np.abs(arr - b["start"]))) for b in bars]


def time_to_q(t: float, beats: list[float]) -> float:
    """Posição em tempos (fracionária) de um instante, interpolando entre os beats reais."""
    if t <= beats[0]:
        step = beats[1] - beats[0]
        return (t - beats[0]) / step
    if t >= beats[-1]:
        step = beats[-1] - beats[-2]
        return len(beats) - 1 + (t - beats[-1]) / step
    i = int(np.searchsorted(beats, t, side="right") - 1)
    return i + (t - beats[i]) / (beats[i + 1] - beats[i])


def quantize(notes: list, beats: list[float], bars: list[dict]) -> tuple[list[dict], int]:
    """Notas -> compassos. Cada compasso: {"beats", "subdiv", "notes": [{"slot", "len", "midi", "tie_in",
    "tie_out"}]}, slots contados a partir do início do compasso. Notas que atravessam a barra de
    compasso são divididas com ligadura."""
    subdiv = choose_subdiv([Hit(n.start, 0) for n in notes], beats) if notes else 4
    starts = bar_beat_index(beats, bars)
    first_slot = starts[0] * subdiv
    end_slot = (starts[-1] + bars[-1]["beats"]) * subdiv

    # 1) cada nota vira (slot inicial, slot final) na grade global
    raw = []
    for n in notes:
        a = int(round(time_to_q(n.start, beats) * subdiv))
        b = int(round(time_to_q(n.end, beats) * subdiv))
        if a < first_slot or a >= end_slot:
            continue
        raw.append([a, max(b, a + 1), n.midi, n.db])
    raw.sort(key=lambda r: (r[0], -r[3]))
    clean = []
    for r in raw:                                # duas notas no mesmo slot: fica a mais forte
        if clean and clean[-1][0] == r[0]:
            continue
        clean.append(r)
    for i in range(len(clean) - 1):              # monofônico
        clean[i][1] = min(clean[i][1], clean[i + 1][0])

    # 2) distribui pelos compassos, dividindo na barra
    measures = [{"beats": b["beats"], "subdiv": subdiv, "notes": []} for b in bars]
    bar_slot = [s * subdiv for s in starts]
    for a, b, midi, _ in clean:
        k = int(np.searchsorted(bar_slot, a, side="right") - 1)
        tie_in = False
        while a < b and k < len(measures):
            m_end = bar_slot[k] + measures[k]["beats"] * subdiv
            piece_end = min(b, m_end)
            measures[k]["notes"].append({"slot": a - bar_slot[k], "len": piece_end - a, "midi": midi,
                                         "tie_in": tie_in, "tie_out": piece_end < b})
            tie_in = True
            a, k = piece_end, k + 1
    return measures, subdiv
