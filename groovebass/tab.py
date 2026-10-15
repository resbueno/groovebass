"""Corda e casa de cada nota (tablatura) para baixo de 4 ou de 5 cordas.

Uma mesma nota pode ser tocada em várias cordas; a escolha minimiza o deslocamento da mão ao
longo da música (Viterbi): pular de casa custa mais que trocar de corda, casas altas custam um
pouco mais e corda solta é barata. Nota abaixo da corda mais grave sobe uma oitava (marcada).
"""
import numpy as np

# cordas numeradas como no MusicXML: 1 = a mais aguda
TUNINGS = {
    4: [43, 38, 33, 28],          # G2 D2 A1 E1
    5: [43, 38, 33, 28, 23],      # G2 D2 A1 E1 B0
}
TUNING_NAMES = {43: ("G", 2), 38: ("D", 2), 33: ("A", 1), 28: ("E", 1), 23: ("B", 0)}
MAX_FRET = 21


def _options(midi: int, strings: list[int]) -> tuple[list[tuple[int, int]], int]:
    """[(corda, casa)] possíveis e o deslocamento de oitava aplicado (0 se coube)."""
    for shift in (0, 12, -12, 24):
        opts = [(i + 1, midi + shift - open_) for i, open_ in enumerate(strings)
                if 0 <= midi + shift - open_ <= MAX_FRET]
        if opts:
            return opts, shift
    return [(len(strings), 0)], 0


def _cost(prev: tuple[int, int], cur: tuple[int, int], gap: float) -> float:
    (ps, pf), (cs, cf) = prev, cur
    if cf == 0 or pf == 0:                       # corda solta não exige posição da mão
        move = 0.3
    else:
        move = abs(cf - pf)
        move = move * 0.6 if move > 4 else move * 0.35   # fora da "caixa" de 4 casas custa mais
    move *= 1.0 / (1.0 + min(gap, 2.0))          # com tempo sobrando, mudar de posição pesa menos
    return move + 0.25 * abs(cs - ps)


def _static(opt: tuple[int, int]) -> float:
    _, f = opt
    return max(0, f - 12) * 0.25 + (0.05 * f if f > 7 else 0.0)


def assign(notes: list[tuple[float, float, int]], n_strings: int) -> list[dict]:
    """notes: [(início s, fim s, midi)] em ordem. Devolve [{"string", "fret", "shift"}] na mesma ordem."""
    strings = TUNINGS[n_strings]
    if not notes:
        return []
    opts, shifts = zip(*[_options(m, strings) for _, _, m in notes])
    # começo: prefere a região grave do braço (até a 5ª casa)
    acc = np.array([_static(o) + 0.04 * o[1] for o in opts[0]])
    backs = []
    for i in range(1, len(notes)):
        gap = max(0.0, notes[i][0] - notes[i - 1][1])
        cur = opts[i]
        trans = np.array([[acc[a] + _cost(opts[i - 1][a], c, gap) for a in range(len(opts[i - 1]))] for c in cur])
        best = trans.argmin(axis=1)
        acc = trans[np.arange(len(cur)), best] + np.array([_static(c) for c in cur])
        backs.append(best)
    k = int(np.argmin(acc))
    path = [k]
    for best in reversed(backs):
        k = int(best[k])
        path.append(k)
    path.reverse()
    return [{"string": opts[i][k][0], "fret": opts[i][k][1], "shift": shifts[i]} for i, k in enumerate(path)]
