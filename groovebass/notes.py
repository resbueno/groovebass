"""bass.wav -> notas (início, fim, altura MIDI, intensidade).

O baixo toca quase sempre uma nota por vez, então basta acompanhar a altura ao longo do tempo:
- altura a cada 10 ms com o torchcrepe (decodificação Viterbi, sem saltos espúrios);
- início das notas pelo onset do próprio stem (notas repetidas na mesma altura só se separam aí)
  e pelas mudanças de altura que se sustentam;
- a altura de cada nota é a mediana do começo dela (o fim pode ter slide ou soltura);
- erros de oitava (o mais comum do pitch tracking em baixo) são corrigidos pelos vizinhos.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# O modelo "tiny" do CREPE (o único rápido o bastante sem GPU) erra abaixo de ~60 Hz, justamente
# onde fica o fundamental do baixo. Truque: o áudio é reamostrado a 8 kHz e entregue ao CREPE como
# se fosse 16 kHz -> tudo soa uma oitava acima (Mi1 41 Hz vira 82 Hz) e a altura é dividida por 2.
# Medido com senoides: Si0 30,9 -> 31,0 Hz; Mi1 41,2 -> 40,9; Lá1 55 -> 55,2 (periodicidade 0,74-0,92).
SR = 8000           # taxa real do áudio analisado (o CREPE recebe como 16 kHz)
CREPE_SR = 16000
OCTAVE_UP = CREPE_SR / SR
HOP = 80            # 10 ms reais
FMIN, FMAX = 29.0, 420.0          # Si0 do 5 cordas a ~Lá#4 (reais)
LOW, HIGH = 21, 72                # faixa MIDI aceita (A0..C5)
VOICED = 0.30                     # periodicidade mínima para considerar que há nota
MIN_FRAMES = 4                    # 40 ms


@dataclass
class Note:
    start: float
    end: float
    midi: int
    db: float          # intensidade no ataque (dB relativo ao pico da faixa)


def _load(wav: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def _pitch(x: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torchcrepe

    audio = torch.from_numpy(x)[None]
    f0, per = torchcrepe.predict(audio, CREPE_SR, HOP, FMIN * OCTAVE_UP, FMAX * OCTAVE_UP, model=model,
                                 decoder=torchcrepe.decode.viterbi, return_periodicity=True, batch_size=1024,
                                 device="cpu", pad=True)
    per = torchcrepe.filter.median(per, 3)
    return f0[0].numpy() / OCTAVE_UP, per[0].numpy()


def _onsets(x: np.ndarray) -> np.ndarray:
    import librosa

    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP, n_fft=512, fmax=2000, n_mels=48, lag=1, max_size=3)
    return librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True,
                                      units="frames", delta=0.08, wait=6)


def transcribe(wav: Path, model: str = "tiny") -> tuple[list[Note], dict]:
    """`model`: "tiny" (~1/30 do custo; padrão, cabe no servidor sem GPU) ou "full"."""
    import librosa

    x = _load(wav)
    f0, per = _pitch(x, model)
    n = len(f0)
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=HOP, center=True)[0][:n]
    if len(rms) < n:
        rms = np.pad(rms, (0, n - len(rms)))
    db = 20 * np.log10(rms + 1e-9)
    peak = np.percentile(db, 99)
    db -= peak
    voiced = (per >= VOICED) & (db > -38) & (f0 > 0)
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0), np.nan)
    rounded = np.round(midi)

    # pontos de corte: onsets + mudança de altura sustentada (>= 4 quadros) + início de trecho com nota
    cuts = set(int(o) for o in _onsets(x) if o < n)
    for i in range(1, n):
        if voiced[i] and not voiced[i - 1]:
            cuts.add(i)
    run_val, run_start = None, 0
    for i in range(n):
        v = rounded[i] if voiced[i] else None
        if v != run_val:
            if run_val is not None and v is not None and i - run_start >= 1:
                ahead = rounded[i:i + MIN_FRAMES]
                if len(ahead) == MIN_FRAMES and np.all(ahead == v):
                    cuts.add(i)
            run_val, run_start = v, i
    cuts = sorted(cuts)
    dedup = []
    for c in cuts:                               # dois cortes a menos de 50 ms viram um
        if dedup and c - dedup[-1] < 5:
            continue
        dedup.append(c)
    edges = dedup + [n]

    notes: list[Note] = []
    for a, b in zip(edges, edges[1:]):
        seg = np.flatnonzero(voiced[a:b])
        if len(seg) < MIN_FRAMES:
            continue
        first = a + seg[0]
        # fim: até o primeiro buraco > 30 ms sem nota
        last = first
        for i in range(first, b):
            if voiced[i]:
                last = i
            elif i - last > 3:
                break
        if last - first + 1 < MIN_FRAMES:
            continue
        span = midi[first:last + 1]
        head = span[: max(MIN_FRAMES, int(0.6 * len(span)))]
        p = int(np.round(np.nanmedian(head)))
        if not LOW <= p <= HIGH:
            continue
        notes.append(Note(start=first * HOP / SR, end=(last + 1) * HOP / SR, midi=p,
                          db=float(np.max(db[first:first + 5]))))

    notes = _fix_octaves(notes)
    # notas fracas e curtas são vazamento/ruído do stem
    notes = [m for m in notes if not (m.db < -28 and m.end - m.start < 0.09)]
    for i in range(len(notes) - 1):                # monofônico: uma nota termina quando a próxima começa
        notes[i].end = min(notes[i].end, notes[i + 1].start)
    info = {"notes": len(notes), "voiced_ratio": round(float(voiced.mean()), 3), "crepe_model": model}
    return notes, info


def _fix_octaves(notes: list[Note]) -> list[Note]:
    """Nota a ~uma oitava dos dois vizinhos (que estão perto entre si) é erro de oitava do tracker."""
    for i in range(1, len(notes) - 1):
        prev, cur, nxt = notes[i - 1].midi, notes[i].midi, notes[i + 1].midi
        if abs(prev - nxt) > 5:
            continue
        ref = (prev + nxt) / 2
        for shift in (-12, 12):
            if abs(cur + shift - ref) <= 5 and abs(cur - ref) >= 9:
                notes[i].midi = cur + shift
                break
    return notes
