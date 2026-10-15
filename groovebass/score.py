"""MusicXML do baixo: pauta (clave de fá) + tablatura (4 ou 5 linhas) na mesma parte, cifras acima
da pauta e partes da música (Intro, Verso...) como sinais de ensaio.

Escrito à mão (sem music21) para controlar a pauta de tablatura (<staff-details>, <string>/<fret>),
as cifras (<harmony>) e as quiálteras. O baixo é escrito uma oitava acima do som real, como é
convenção na clave de fá para baixo elétrico.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from .harmony import KINDS, names_for
from .tab import TUNING_NAMES, TUNINGS

DIV = 12                                   # divisões por semínima (serve para 16 avos e tercinas)
# comprimento em slots -> (tipo, pontos, é quiáltera)
TYPES = {
    4: {1: ("16th", 0, False), 2: ("eighth", 0, False), 3: ("eighth", 1, False), 4: ("quarter", 0, False),
        8: ("half", 0, False), 12: ("half", 1, False), 16: ("whole", 0, False)},
    3: {1: ("eighth", 0, True), 2: ("quarter", 0, True), 3: ("quarter", 0, False), 6: ("half", 0, False),
        9: ("half", 1, False), 12: ("whole", 0, False)},
}


def _split(pos: int, length: int, sd: int, total: int) -> list[int]:
    """Divide uma duração em figuras legíveis: nada atravessa o tempo fora do tempo; no tempo,
    semínima, mínima (em tempo ímpar), mínima pontuada ou semibreve (do começo do compasso)."""
    out = []
    while length > 0:
        if pos % sd:
            L = min(length, sd - pos % sd)
        else:
            full = min(length, total - pos) // sd
            if full == 0:
                L = length
            else:
                beat = pos // sd
                c = 1
                for cand in (4, 3, 2):
                    if cand <= full and (cand == 2 and beat % 2 == 0 or cand in (3, 4) and pos == 0):
                        c = cand
                        break
                L = c * sd
        out.append(L)
        pos += L
        length -= L
    return out


def _spell(midi: int, fifths: int) -> tuple[str, int, int]:
    name = names_for(fifths)[midi % 12]
    alter = 1 if name.endswith("#") else -1 if name.endswith("b") else 0
    return name[0], alter, midi // 12 - 1


def _pitch_xml(midi: int, fifths: int) -> str:
    step, alter, octave = _spell(midi + 12, fifths)        # escrito uma oitava acima
    return (f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "")
            + f"<octave>{octave}</octave></pitch>")


def _harmony_xml(ch: dict, fifths: int) -> str:
    if ch["root"] is None:                       # sem acorde (silêncio): nada na partitura
        return ""
    names = names_for(fifths)

    def part(tag, pc):
        nm = names[pc]
        alter = 1 if nm.endswith("#") else -1 if nm.endswith("b") else 0
        return (f"<{tag}-step>{nm[0]}</{tag}-step>" + (f"<{tag}-alter>{alter}</{tag}-alter>" if alter else ""))

    kind_text, kind_val = ch["kind"], KINDS[ch["kind"]][1]
    xml = f'<harmony print-frame="no"><root>{part("root", ch["root"])}</root><kind text="{escape(kind_text)}">{kind_val}</kind>'
    if ch.get("bass") is not None:
        xml += f"<bass>{part('bass', ch['bass'])}</bass>"
    return xml + "<staff>1</staff></harmony>"


def _note_xml(*, midi, dur_slots, sd, staff, voice, fifths, tie_stop, tie_start, tuplet, tab=None) -> str:
    typ, dots, triplet = TYPES[sd][dur_slots]
    parts = ["<note>"]
    if midi is None:
        parts.append("<rest/>")
    else:
        parts.append(_pitch_xml(midi, fifths))
    parts.append(f"<duration>{dur_slots * DIV // sd}</duration>")
    if tie_stop:
        parts.append('<tie type="stop"/>')
    if tie_start:
        parts.append('<tie type="start"/>')
    parts.append(f"<voice>{voice}</voice><type>{typ}</type>" + "<dot/>" * dots)
    if triplet:
        parts.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>")
    parts.append(f"<staff>{staff}</staff>")
    nots = []
    if tie_stop:
        nots.append('<tied type="stop"/>')
    if tie_start:
        nots.append('<tied type="start"/>')
    if tuplet:
        nots.append(f'<tuplet type="{tuplet}" bracket="yes"/>')
    if tab is not None and midi is not None:
        nots.append(f"<technical><string>{tab['string']}</string><fret>{tab['fret']}</fret></technical>")
    if nots:
        parts.append("<notations>" + "".join(nots) + "</notations>")
    parts.append("</note>")
    xml = "".join(parts)
    if midi is None and staff == 2:              # pausas não aparecem na tablatura
        xml = xml.replace("<note>", '<note print-object="no">', 1)
    return xml


def _events(measure: dict) -> list[dict]:
    """Notas e pausas do compasso, já divididas em figuras: [{"pos", "len", "midi", "tie_stop",
    "tie_start", "idx"}] (idx = índice da nota no compasso, None para pausa)."""
    sd, total = measure["subdiv"], measure["beats"] * measure["subdiv"]
    out, pos = [], 0
    for idx, n in enumerate(sorted(measure["notes"], key=lambda n: n["slot"])):
        if n["slot"] > pos:
            for L in _split(pos, n["slot"] - pos, sd, total):
                out.append({"pos": pos, "len": L, "midi": None, "tie_stop": False, "tie_start": False, "idx": None})
                pos += L
        pieces = _split(n["slot"], n["len"], sd, total)
        p = n["slot"]
        for j, L in enumerate(pieces):
            out.append({"pos": p, "len": L, "midi": n["midi"], "idx": idx,
                        "tie_stop": n["tie_in"] if j == 0 else True,
                        "tie_start": n["tie_out"] if j == len(pieces) - 1 else True})
            p += L
        pos = p
    if pos < total:
        for L in _split(pos, total - pos, sd, total):
            out.append({"pos": pos, "len": L, "midi": None, "tie_stop": False, "tie_start": False, "idx": None})
            pos += L
    # colchete de tercina: início e fim de cada tempo que tem figuras de tercina
    if sd == 3:
        by_beat: dict = {}
        for e in out:
            if TYPES[3][e["len"]][2]:
                by_beat.setdefault(e["pos"] // 3, []).append(e)
        for group in by_beat.values():
            group[0]["tuplet"] = "start"
            group[-1]["tuplet"] = "stop" if len(group) > 1 else group[-1].get("tuplet")
    return out


def write(out: Path, measures: list[dict], tabs: list[list[dict]], chords_by_bar: dict, sections: dict,
          fifths: int, bpm: float, title: str, n_strings: int) -> Path:
    """`tabs[k][i]`: corda/casa da i-ésima nota do compasso k (ordenadas por slot).
    `chords_by_bar`: {k: [(slot, acorde)]}; `sections`: {k: "Refrão"}."""
    tuning = list(reversed(TUNINGS[n_strings]))   # linha 1 da pauta = corda mais grave
    staff_details = "".join(
        f'<staff-tuning line="{i + 1}"><tuning-step>{TUNING_NAMES[m][0]}</tuning-step>'
        f"<tuning-octave>{TUNING_NAMES[m][1]}</tuning-octave></staff-tuning>"
        for i, m in enumerate(tuning))
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">',
           f"<work><work-title>{escape(title)}</work-title></work>",
           '<part-list><score-part id="P1"><part-name>Baixo</part-name></score-part></part-list>',
           '<part id="P1">']
    cur_beats = None
    for k, m in enumerate(measures):
        sd, total = m["subdiv"], m["beats"] * m["subdiv"]
        xml.append(f'<measure number="{k + 1}">')
        attrs = []
        if k == 0:
            attrs.append(f"<divisions>{DIV}</divisions><key><fifths>{fifths}</fifths></key>")
        if m["beats"] != cur_beats:
            attrs.append(f"<time><beats>{m['beats']}</beats><beat-type>4</beat-type></time>")
            cur_beats = m["beats"]
        if k == 0:
            attrs.append('<staves>2</staves><clef number="1"><sign>F</sign><line>4</line></clef>'
                         '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                         f'<staff-details number="2"><staff-lines>{n_strings}</staff-lines>{staff_details}</staff-details>')
        if attrs:
            xml.append("<attributes>" + "".join(attrs) + "</attributes>")
        if k == 0:
            xml.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                       f'<per-minute>{round(bpm)}</per-minute></metronome></direction-type><staff>1</staff>'
                       f'<sound tempo="{round(bpm)}"/></direction>')
        if k in sections:
            xml.append(f'<direction placement="above"><direction-type><rehearsal>{escape(sections[k])}</rehearsal>'
                       "</direction-type><staff>1</staff></direction>")

        events = _events(m)
        notes_sorted = sorted(m["notes"], key=lambda n: n["slot"])
        pending = sorted(chords_by_bar.get(k, []), key=lambda c: c[0])
        for staff in (1, 2):
            if staff == 2:
                xml.append(f"<backup><duration>{total * DIV // sd}</duration></backup>")
            ci = 0
            for e in events:
                if staff == 1:
                    # cifra na figura em que a troca acontece (várias trocas na mesma figura: vale a última)
                    hit = None
                    while ci < len(pending) and pending[ci][0] < e["pos"] + e["len"]:
                        hit = pending[ci][1]
                        ci += 1
                    if hit is not None:
                        xml.append(_harmony_xml(hit, fifths))
                tab = tabs[k][e["idx"]] if (staff == 2 and e["idx"] is not None) else None
                xml.append(_note_xml(midi=e["midi"], dur_slots=e["len"], sd=sd, staff=staff,
                                     voice=1 if staff == 1 else 5, fifths=fifths,
                                     tie_stop=e["tie_stop"], tie_start=e["tie_start"],
                                     tuplet=e.get("tuplet"), tab=tab))
        xml.append("</measure>")
    xml.append("</part></score-partwise>")
    out.write_text("\n".join(xml), encoding="utf-8")
    return out
