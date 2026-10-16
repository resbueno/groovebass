"""bass.wav -> notas (início, fim, altura MIDI, intensidade).

O baixo toca quase sempre uma nota por vez, então basta acompanhar a altura ao longo do tempo:
- altura a cada 10 ms com o torchcrepe (decodificação Viterbi, sem saltos espúrios);
- início das notas pelo onset do próprio stem (notas repetidas na mesma altura só se separam aí)
  e pelas mudanças de altura que se sustentam;
- a altura de cada nota é a mediana do começo dela (o fim pode ter slide ou soltura);
- quadros com som forte em que o CREPE não reconhece altura (hammer-on, nota abafada) recuperam a
  nota pelo pico do espectro, na oitava das vizinhas;
- notas repetidas na mesma altura se separam também pela queda e retomada do volume e pelo vale de
  periodicidade do ataque;
- a afinação da gravação (fita fora de velocidade) é medida e descontada antes de arredondar;
- depois de montadas, as notas são conferidas no espectro: a nota que entrou no ataque (outra corda
  pode continuar soando), a oitava pelos harmônicos ímpares e a oitava das notas em ligado.

Medido contra tablaturas do Songsterr (nota certa no tempo certo):
  No More Tears (álbum, comp. 5-66)        32% -> 46% (53% contando a troca de fase do riff)
  Another One Bites the Dust (inteira)      51% -> 91%
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
# Calibrados contra a partitura de No More Tears (riff de 11 notas por compasso: colcheias na corda
# solta de Dó#2 e o par Si2-Dó#3 em hammer-on por cima dela): ataques achados 45% -> 90%.
ONSET_DELTA = 0.03                # sensibilidade dos onsets (librosa) no stem do baixo
REATTACK_DB = 2.0                 # nova palhetada na mesma nota: o volume cai e volta a subir isso
SPEC_NFFT = 2048                  # espectro para recuperar notas sem periodicidade (256 ms, ~3,9 Hz/faixa)
RESCUE_DB = -22.0                 # som forte o bastante para procurar a nota no espectro
PER_DIP = 0.15                    # vale de periodicidade que marca nova palhetada (0 desliga)
OCT_WIN, OCT_NFFT = 0.18, 8192   # espectro após o ataque para decidir a oitava (180 ms, ~1 Hz/faixa)
OCTAVE_MODE = "spectral"          # "spectral" (_fix_octaves) ou "median" (_fix_high_outliers, o antigo)
OCT_UP, OCT_DOWN = 0.5, 0.8       # razão de ímpares (ver _fix_octaves) para subir / descer uma oitava
LEGATO_DB = 5.0                   # clique abaixo disso = nota em ligado (ver _legato_octaves; 0 desliga)
LEGATO_GAP = 0.03                 # silêncio máximo entre a nota anterior e a ligada (s)
ATTACK_MARGIN = 3.0               # nota nova pelo espectro do ataque: quanto precisa superar a do CREPE (0 desliga)
ATTACK_WEIGHTS = (1.0, 0.8, 0.6, 0.5, 0.4)   # peso dos harmônicos 1-5 na soma harmônica
TUNE_PER = 0.7                    # quadros confiáveis do CREPE que medem a afinação da gravação
SURE = 0.8                        # CREPE seguro (Note.conf acima disso): o espectro não troca a nota nem sobe a oitava


@dataclass
class Note:
    start: float
    end: float
    midi: int
    db: float          # intensidade no ataque (dB relativo ao pico da faixa)
    conf: float = 1.0  # periodicidade mediana do CREPE de 60 a 160 ms (baixa: outra nota soando junto)


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
                                      units="frames", delta=ONSET_DELTA, wait=6)


def _reattacks(db: np.ndarray, voiced: np.ndarray) -> list[int]:
    """Nova palhetada sem mudar de altura (colcheias repetidas na corda solta): o volume tem um vale
    e volta a subir pelo menos REATTACK_DB dentro de 70 ms, depois de ter caído desde o pico anterior."""
    out = []
    n = len(db)
    for i in range(3, n - 8):
        if not voiced[i] or db[i] > db[i - 1] or db[i] > db[i + 1]:
            continue
        rise = db[i + 1:i + 8].max() - db[i]
        drop = db[max(0, i - 10):i].max() - db[i]
        if rise >= REATTACK_DB and drop >= REATTACK_DB * 0.5:
            out.append(i)
    return out


def _per_dips(per: np.ndarray, voiced: np.ndarray) -> list[int]:
    """Nova palhetada na mesma nota pela periodicidade: o ruído do ataque derruba o CREPE por 20-40 ms
    (0,88 -> 0,5) sem chegar ao limiar de nota (baixo comprimido, em que o volume quase não muda;
    medido em No More Tears). Vale com queda de PER_DIP dos dois lados em até 60 ms."""
    out = []
    n = len(per)
    for i in range(6, n - 6):
        if per[i] > per[i - 1] or per[i] > per[i + 1] or not voiced[max(0, i - 6):i + 7].any():
            continue
        if per[i - 6:i].max() - per[i] >= PER_DIP and per[i + 1:i + 7].max() - per[i] >= PER_DIP:
            out.append(i)
    return out


def _rescue(spec: np.ndarray, freqs: np.ndarray, db: np.ndarray, voiced: np.ndarray,
            midi: np.ndarray, a4: float = 440.0) -> int:
    """Hammer-ons e notas abafadas: o CREPE dá periodicidade ~0 mesmo com o baixo soando forte (medido
    em No More Tears: o Si1 da casa 10, 3 das 11 notas de cada compasso do riff). Nesses quadros a nota
    sai do pico do espectro (50-400 Hz), na oitava mais próxima das notas que o CREPE achou em volta
    (o pico costuma ser o 2º harmônico). Altera `voiced`/`midi` no lugar; devolve quantos quadros."""
    band = (freqs >= 50) & (freqs <= 400)
    bf = freqs[band]
    idx = np.flatnonzero((~voiced) & (db > RESCUE_DB))
    ok = np.flatnonzero(voiced)
    if not len(idx) or not len(ok):
        return 0
    done = 0
    for i in idx:
        col = spec[band, min(i, spec.shape[1] - 1)]
        if col.max() <= 0:
            continue
        peak = 69 + 12 * np.log2(bf[int(np.argmax(col))] / a4)
        lo, hi = np.searchsorted(ok, i - 100), np.searchsorted(ok, i + 100)
        if hi <= lo:
            continue
        ref = float(np.nanmedian(midi[ok[lo:hi]]))
        cand = [peak - 24, peak - 12, peak]
        best = min((c for c in cand if LOW <= round(c) <= HIGH), key=lambda c: abs(c - ref), default=None)
        if best is None:
            continue
        midi[i] = best
        voiced[i] = True
        done += 1
    return done


def _tuning(f0: np.ndarray, per: np.ndarray) -> float:
    """Desvio da gravação em relação ao Lá 440, em semitons (-0,5..0,5): média circular da distância de
    cada quadro confiável ao semitom mais próximo. Fita fora de velocidade é comum em discos antigos
    (Another One Bites the Dust: +0,49, as notas caíam meio a meio no semitom de cima)."""
    ok = (per > TUNE_PER) & (f0 > 0)
    if ok.sum() < 200:
        return 0.0
    m = 69 + 12 * np.log2(f0[ok] / 440.0)
    return float(np.angle(np.mean(np.exp(2j * np.pi * m))) / (2 * np.pi))


def transcribe(wav: Path, model: str = "tiny", cache: Path | None = None) -> tuple[list[Note], dict]:
    """`model`: "tiny" (~1/30 do custo; padrão, cabe no servidor sem GPU) ou "full". `cache`: .npz com a
    saída do CREPE (o passo caro), reaproveitada ao reprocessar."""
    import librosa

    x = _load(wav)
    if cache and cache.exists():
        z = np.load(cache)
        f0, per = z["f0"], z["per"]
    else:
        f0, per = _pitch(x, model)
        if cache:
            np.savez_compressed(cache, f0=f0, per=per)
    n = len(f0)
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=HOP, center=True)[0][:n]
    if len(rms) < n:
        rms = np.pad(rms, (0, n - len(rms)))
    db = 20 * np.log10(rms + 1e-9)
    peak = np.percentile(db, 99)
    db -= peak
    voiced = (per >= VOICED) & (db > -38) & (f0 > 0)
    tune = _tuning(f0, per)
    a4 = 440.0 * 2 ** (tune / 12)                  # Lá de referência desta gravação
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / a4), np.nan)
    spec = np.abs(librosa.stft(x, n_fft=SPEC_NFFT, hop_length=HOP, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=SPEC_NFFT)
    rescued = _rescue(spec, freqs, db, voiced, midi, a4)
    rounded = np.round(midi)

    # pontos de corte: onsets + mudança de altura sustentada (>= 4 quadros) + início de trecho com nota
    cuts = set(int(o) for o in _onsets(x) if o < n)
    cuts.update(_reattacks(db, voiced))
    if PER_DIP:
        cuts.update(_per_dips(per, voiced))
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
        body = per[first + 6:first + 16] if last - first >= 10 else per[first:last + 1]   # depois do ataque
        notes.append(Note(start=first * HOP / SR, end=(last + 1) * HOP / SR, midi=p,
                          db=float(np.max(db[first:first + 5])), conf=float(np.median(body))))

    # notas fracas e curtas são vazamento/ruído do stem
    notes = [m for m in notes if not (m.db < -28 and m.end - m.start < 0.09)]
    attacks = _attack_pitch(notes, x, a4) if ATTACK_MARGIN else 0
    octaves = _fix_octaves(notes, x, a4) if OCTAVE_MODE == "spectral" else 0
    if OCTAVE_MODE == "median":
        notes = _fix_high_outliers(notes)
    legato = _legato_octaves(notes, x) if LEGATO_DB else 0
    for i in range(len(notes) - 1):                # monofônico: uma nota termina quando a próxima começa
        notes[i].end = min(notes[i].end, notes[i + 1].start)
    info = {"notes": len(notes), "voiced_ratio": round(float(voiced.mean()), 3), "crepe_model": model,
            "tuning_cents": round(100 * tune), "rescued_frames": rescued, "attack_fixes": attacks,
            "octave_fixes": octaves, "legato_fixes": legato}
    return notes, info


def _odd_ratio(spec: np.ndarray, freqs: np.ndarray, midi: int, a4: float = 440.0) -> float:
    """Harmônicos ímpares de `midi` (1º, 3º, 5º) sobre os três primeiros da oitava de cima. Os pares
    de `midi` coincidem com os da nota uma oitava acima; só os ímpares provam que a de baixo soa."""
    def amp(f):
        lo, hi = np.searchsorted(freqs, f * 2 ** (-0.4 / 12)), np.searchsorted(freqs, f * 2 ** (0.4 / 12))
        return float(spec[lo:hi + 1].max()) if lo < len(spec) else 0.0
    f0 = a4 * 2 ** ((midi - 69) / 12)
    odd = sum(amp(f0 * h) for h in (1, 3, 5))
    up = sum(amp(2 * f0 * h) for h in (1, 2, 3))
    return odd / (up + 1e-9)


def _attack_pitch(notes: list[Note], x: np.ndarray, a4: float = 440.0) -> int:
    """Nota tocada sobre outra que continua soando (corda solta de pedal): o CREPE segue a que já
    soava. O que entrou no ataque é o espectro de depois menos o de antes (OCT_WIN cada); a nota cuja
    soma harmônica explica essa diferença ATTACK_MARGIN vezes melhor que a do CREPE fica no lugar.
    Só vale com o CREPE inseguro (Note.conf < SURE). Medido em No More Tears (Si2 sobre Dó#2 soando):
    nota certa 66% -> 72% no intro. Devolve quantas mudaram."""
    w = int(OCT_WIN * SR)
    win = np.hanning(w)
    freqs = np.fft.rfftfreq(OCT_NFFT, 1 / SR)
    cand = np.arange(LOW, min(HIGH, 60) + 1)      # até Dó4: acima disso a soma harmônica sai do espectro útil
    f0s = a4 * 2 ** ((cand - 69) / 12)
    weights = np.array(ATTACK_WEIGHTS)
    harm = np.arange(1, len(weights) + 1)
    lo = np.searchsorted(freqs, np.outer(f0s, harm) * 2 ** (-0.4 / 12))
    hi = np.minimum(np.searchsorted(freqs, np.outer(f0s, harm) * 2 ** (0.4 / 12)) + 1, len(freqs))

    def spec(a):
        seg = x[a:a + w]
        return np.abs(np.fft.rfft(np.pad(seg, (0, w - len(seg))) * win, OCT_NFFT))

    changed = 0
    for m in notes:
        a = int((m.start + 0.015) * SR)
        b = a - w - int(0.035 * SR)                # janela de antes termina 20 ms antes do início
        if b < 0 or len(x) - a < w // 2:
            continue
        added = np.maximum(0.0, spec(a) - spec(b))
        peaks = np.array([[added[l:h].max() if h > l else 0.0 for l, h in zip(lr, hr)] for lr, hr in zip(lo, hi)])
        score = peaks @ weights
        best = int(cand[np.argmax(score)])
        same = score[cand % 12 == m.midi % 12].max()
        if m.conf < SURE and best % 12 != m.midi % 12 and score.max() >= ATTACK_MARGIN * same:
            m.midi = best
            changed += 1
    return changed


def _fix_octaves(notes: list[Note], x: np.ndarray, a4: float = 440.0) -> int:
    """Oitava de cada nota pelo espectro logo após o ataque (OCT_WIN). Sobe se os ímpares da oitava
    atual quase não existem (Si2 tocado sobre a corda solta de Dó#2 soando: o CREPE se perde e o
    resgate punha Si1); desce enquanto os ímpares da oitava de baixo são fortes (CREPE no 2º ou 4º
    harmônico). Medido em No More Tears: razão < 0,4 em todo Si2 e > 0,64 em todo Dó#2. Devolve
    quantas notas mudaram."""
    w = int(OCT_WIN * SR)
    win = np.hanning(w)
    freqs = np.fft.rfftfreq(OCT_NFFT, 1 / SR)
    changed = 0
    for m in notes:
        a = int((m.start + 0.015) * SR)
        seg = x[a:a + w]
        if len(seg) < w // 2:
            continue
        spec = np.abs(np.fft.rfft(np.pad(seg, (0, w - len(seg))) * win, OCT_NFFT))
        p = m.midi
        if m.conf < SURE and p + 12 <= HIGH and _odd_ratio(spec, freqs, p, a4) < OCT_UP:
            p += 12
        else:
            for _ in range(2):
                if p - 12 >= LOW and _odd_ratio(spec, freqs, p - 12, a4) >= OCT_DOWN:
                    p -= 12
                else:
                    break
        changed += p != m.midi
        m.midi = p
    return changed


def _legato_octaves(notes: list[Note], x: np.ndarray) -> int:
    """Hammer-on, pull-off e slide (nota colada na anterior, de outro nome e sem o clique da palhetada
    acima de 700 Hz) ficam na mesma região do braço: vale a oitava mais próxima da nota anterior.
    Resolve a nota de cima tocada sobre uma corda solta que continua soando (No More Tears: Si2 -> Dó#3
    em ligado sobre Dó#2; o espectro mostra as duas e o CREPE fica com a de baixo). Medido lá: o clique
    sobe ~3 dB no hammer-on e 11-13 dB na palhetada. Devolve quantas notas mudaram."""
    import scipy.signal as ss

    hp = ss.sosfilt(ss.butter(4, 700, "hp", fs=SR, output="sos"), x)
    env = 20 * np.log10(np.sqrt(np.convolve(hp ** 2, np.ones(40) / 40, "same")) + 1e-7)   # 5 ms
    changed = 0
    for prev, m in zip(notes, notes[1:]):
        if m.midi % 12 == prev.midi % 12 or m.start - prev.end > LEGATO_GAP:
            continue
        a = int(m.start * SR)
        before = env[max(0, a - int(0.08 * SR)):max(1, a - int(0.02 * SR))]
        if not len(before) or env[max(0, a - int(0.01 * SR)):a + int(0.04 * SR)].max() - np.median(before) >= LEGATO_DB:
            continue
        best = min((p for p in range(m.midi % 12, HIGH + 1, 12) if p >= LOW), key=lambda p: abs(p - prev.midi))
        if abs(best - prev.midi) <= 7 and best != m.midi:
            m.midi = best
            changed += 1
    return changed


def _fix_high_outliers(notes: list[Note], window: float = 2.0) -> list[Note]:
    """Nota uma oitava ou mais acima da mediana das vizinhas (±2 s) é o CREPE pegando o 4º harmônico
    (Réb1 lido como Réb3): desce uma oitava. Linhas que alternam oitavas (Dó2-Dó3) não mudam, porque a
    mediana fica entre elas."""
    starts = np.array([m.start for m in notes])
    pitches = np.array([m.midi for m in notes])
    for i, m in enumerate(notes):
        lo, hi = np.searchsorted(starts, m.start - window), np.searchsorted(starts, m.start + window)
        if hi - lo < 5:
            continue
        med = float(np.median(pitches[lo:hi]))
        if m.midi - med >= 11.5 and m.midi - 12 >= LOW:
            m.midi -= 12
    return notes
