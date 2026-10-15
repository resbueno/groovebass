"""Tom da música e cifra (acordes) por tempo.

- Cromagrama de cada tempo na música sem o baixo (só a harmonia de guitarra, teclado e voz).
- A nota do baixo naquele tempo entra como fundamental provável ou como baixo invertido (G/B).
- Cada tempo pontua os acordes candidatos (tríades, tétrades comuns e sus4); uma Viterbi com
  custo de troca evita que a cifra mude a cada tempo por causa de uma nota de passagem.
- O tom vem do perfil de Krumhansl sobre o cromagrama da música inteira e decide a grafia
  (F# ou Gb, A# ou Bb) das notas e das cifras.
"""
import numpy as np

from kicksnare.sections import _bar_audio

SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# cifra no padrão brasileiro: C, Cm, C7, Cm7, C7M, C°, Csus4
KINDS = {
    "": ((0, 4, 7), "major"),
    "m": ((0, 3, 7), "minor"),
    "7": ((0, 4, 7, 10), "dominant"),
    "m7": ((0, 3, 7, 10), "minor-seventh"),
    "7M": ((0, 4, 7, 11), "major-seventh"),
    "°": ((0, 3, 6), "diminished"),
    "sus4": ((0, 5, 7), "suspended-fourth"),
}
SIMPLICITY = {"": 0.0, "m": 0.0, "7": 0.03, "m7": 0.03, "7M": 0.04, "°": 0.06, "sus4": 0.06}
SWITCH = 0.22          # custo de trocar de acorde entre dois tempos

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# armadura (quintas) de cada tônica maior; menor = relativa maior (tônica + 3)
FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}


def _template(root: int, intervals: tuple) -> np.ndarray:
    t = np.zeros(12)
    weights = (1.0, 0.8, 0.8, 0.6)
    for w, iv in zip(weights, intervals):
        t[(root + iv) % 12] = w
    return t / np.linalg.norm(t)


CANDIDATES = [(r, k) for r in range(12) for k in KINDS]
TEMPLATES = np.stack([_template(r, KINDS[k][0]) for r, k in CANDIDATES])


def estimate_key(chroma: np.ndarray, weights: np.ndarray) -> dict:
    total = (chroma * weights[:, None]).sum(axis=0)
    best = None
    for tonic in range(12):
        for mode, prof in (("major", MAJOR), ("minor", MINOR)):
            r = np.corrcoef(total, np.roll(prof, tonic))[0, 1]
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    _, tonic, mode = best
    fifths = FIFTHS[tonic if mode == "major" else (tonic + 3) % 12]
    return {"tonic": tonic, "mode": mode, "fifths": fifths}


def names_for(fifths: int) -> list[str]:
    return FLATS if fifths < 0 else SHARPS


def chord_label(root: int, kind: str, bass: int | None, fifths: int) -> str:
    names = names_for(fifths)
    s = names[root] + kind
    if bass is not None and bass != root:
        s += "/" + names[bass]
    return s


def analyze(no_bass_wav, beats: list[float], bass_pc_per_beat: list) -> tuple[dict, list[dict]]:
    """(tom, acordes). Acordes: [{"beat": índice do tempo, "root", "kind", "bass", "label"}] só nas trocas;
    root=None = sem acorde (N.C.)."""
    starts = beats[:-1]
    chroma, db = _bar_audio(no_bass_wav, starts, beats[-1])
    n = len(starts)
    weights = np.clip(db - db.max() + 40, 0, None)
    key = estimate_key(chroma, weights)

    # pontuação de cada candidato em cada tempo
    score = chroma @ TEMPLATES.T                                  # (n, candidatos)
    score -= np.array([SIMPLICITY[k] for _, k in CANDIDATES])[None, :]
    for i, bpc in enumerate(bass_pc_per_beat[:n]):
        if bpc is None:
            continue
        for j, (r, k) in enumerate(CANDIDATES):
            if bpc == r:
                score[i, j] += 0.15
            elif any((r + iv) % 12 == bpc for iv in KINDS[k][0]):
                score[i, j] += 0.04
    silent = db < db.max() - 35

    # Viterbi
    m = len(CANDIDATES)
    acc = score[0].copy()
    back = np.zeros((n, m), dtype=int)
    for i in range(1, n):
        best_prev = int(np.argmax(acc))
        stay = acc
        move = acc[best_prev] - SWITCH
        choose_move = move > stay
        back[i] = np.where(choose_move, best_prev, np.arange(m))
        acc = np.where(choose_move, move, stay) + score[i]
    path = [int(np.argmax(acc))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path.reverse()

    # só (fundamental, tipo) trocam o acorde; a inversão (C/G) vem do baixo no 1º tempo do acorde,
    # senão cada nota de passagem do baixo viraria uma cifra nova
    chords, prev = [], None
    for i, j in enumerate(path):
        cur = (None, None) if silent[i] else CANDIDATES[j]
        if cur == prev:
            continue
        r, k = cur
        inv = None
        if r is not None:
            bpc = bass_pc_per_beat[i] if i < len(bass_pc_per_beat) else None
            if bpc is not None and bpc != r and any((r + iv) % 12 == bpc for iv in KINDS[k][0]):
                inv = bpc
        label = "N.C." if r is None else chord_label(r, k, inv, key["fifths"])
        chords.append({"beat": i, "root": r, "kind": k, "bass": inv, "label": label})
        prev = cur
    return key, chords
