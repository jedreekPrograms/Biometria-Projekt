"""Składa raport PDF z rzeczywistych plików wynikowych."""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as PDFImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "wyniki"
OUTPUT = ROOT / "Raport-wykrywanie-mowy-syntetycznej.pdf"
INK = colors.HexColor("#19293D")
GREY = colors.HexColor("#475A6E")
TEAL = colors.HexColor("#12685F")
BLUE = colors.HexColor("#244F91")
PALE = colors.HexColor("#EDF3F8")
GREEN = colors.HexColor("#E7F4F0")
LINE = colors.HexColor("#D5DEE8")
WHITE = colors.white
MARGIN = 44
WIDTH = A4[0] - 2 * MARGIN - 12


def setup_fonts():
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in (
        ("DejaVu", "DejaVuSans.ttf"), ("DejaVuBold", "DejaVuSans-Bold.ttf"),
        ("DejaVuOblique", "DejaVuSans-Oblique.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVuBold", italic="DejaVuOblique",
        boldItalic="DejaVuBold",
    )


def style(name, size, leading, *, bold=False, color=INK, after=0):
    return ParagraphStyle(
        name, fontName="DejaVuBold" if bold else "DejaVu", fontSize=size,
        leading=leading, textColor=color, spaceAfter=after, alignment=TA_LEFT,
    )


STYLES = {
    "tag": style("tag", 9, 13, bold=True, color=TEAL, after=10),
    "cover": style("cover", 27, 33, bold=True, after=12),
    "heading": style("heading", 24, 30, bold=True, after=12),
    "intro": style("intro", 10.7, 16, color=GREY, after=12),
    "body": style("body", 9.4, 14.2, after=8),
    "sub": style("sub", 12.5, 17, bold=True, after=5),
    "caption": style("caption", 8.1, 11.8, color=GREY, after=10),
    "cell": style("cell", 8.6, 12.0),
    "headcell": style("headcell", 8.25, 11.4, bold=True, color=WHITE),
    "kpi": style("kpi", 22, 27, bold=True, color=TEAL, after=5),
    "kpilabel": style("kpilabel", 8.65, 12.7, color=INK),
}


def P(text, kind="body"):
    return Paragraph(text, STYLES[kind])


def optional_link(text, url):
    return f'<link href="{url}" color="#244F91"><u>{text}</u></link>'


def table(rows, widths, highlight=None):
    items = [
        [P(str(value), "headcell" if idx == 0 else "cell") for value in row]
        for idx, row in enumerate(rows)
    ]
    flow = Table(items, [WIDTH * part for part in widths], hAlign="LEFT", repeatRows=1)
    grid = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
    ]
    if highlight is not None:
        grid.append(("BACKGROUND", (0, highlight), (-1, highlight), GREEN))
    flow.setStyle(TableStyle(grid))
    return flow


def cards(values):
    names = [[P(value, "kpi"), P(label, "kpilabel")] for value, label in values]
    layout = Table([names], [WIDTH / 3] * 3, hAlign="LEFT")
    layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), GREEN),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LINEAFTER", (0, 0), (1, 0), 3, WHITE),
    ]))
    return layout


def plot(filename, max_height):
    path = RESULT / "wykresy" / filename
    with Image.open(path) as raster:
        w, h = raster.size
    ratio = min(WIDTH / w, max_height / h)
    return PDFImage(str(path), w * ratio, h * ratio, hAlign="CENTER")


def decorate(canvas, doc):
    w, h = A4
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 5, w, 5, fill=1, stroke=0)
    canvas.setFont("DejaVuBold", 7.3)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN + 6, h - 32, "SYSTEMY IDENTYFIKACYJNE I BIOMETRYCZNE")
    canvas.drawRightString(w - MARGIN - 6, h - 32, "ASVSPOOF 2019  /  LOGICAL ACCESS")
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN + 6, 43, w - MARGIN - 6, 43)
    canvas.setFont("DejaVu", 7.6)
    canvas.drawString(MARGIN + 6, 29, "Miniprojekt  ·  Wykrywanie syntetycznej mowy")
    canvas.drawRightString(w - MARGIN - 6, 29, str(doc.page))


def n(value):
    return f"{int(value):,}".replace(",", "&nbsp;")


def pct(value, decimals=1):
    return f"{value * 100:.{decimals}f}%".replace(".", ",")


def pp(value, signed=False):
    modifier = "+" if signed else ""
    return f"{value * 100:{modifier}.1f}".replace(".", ",") + " p.p."


def decimal(value, count=2):
    return f"{value:.{count}f}".replace(".", ",")


def select(frame, subset, variant):
    return frame.loc[frame["subset"].eq(subset) & frame["variant"].eq(variant)].iloc[0]


def section(story, tag, heading, intro=None):
    story.extend([P(tag, "tag"), P(heading, "heading")])
    if intro:
        story.append(P(intro, "intro"))


def stats_for_audit(audit, current):
    rows = [[
        "Część", "Z protokołu", "Krótkie<br/>&lt;2 s", "Inne<br/>odrzucone",
        "Naturalne<br/>przyjęte", "Syntezy<br/>przyjęte", "Mówcy<br/>po filtracji",
    ]]
    for split, label in (("train", "Trening"), ("dev", "Walidacja"), ("eval", "Test")):
        raw = audit[audit["split"].eq(split)]
        kept = current[current["split"].eq(split)]
        short = raw["status"].eq("too_short").sum()
        rows.append([
            label, n(len(raw)), n(short), n(len(raw) - short - len(kept)),
            n(kept["label"].eq(0).sum()), n(kept["label"].eq(1).sum()),
            n(kept["speaker_id"].nunique()),
        ])
    return rows


def build():
    setup_fonts()
    audit = pd.read_csv(RESULT / "kontrola_danych.csv.gz")
    current = pd.read_csv(RESULT / "cechy.csv.gz")
    results = pd.read_csv(RESULT / "metryki.csv")
    errors = pd.read_csv(RESULT / "bledy.csv")
    gen = pd.read_csv(RESULT / "generatory.csv")
    intervals = json.loads((RESULT / "przedzialy.json").read_text(encoding="utf-8"))
    main = results[results["seed"].eq(42)]
    e_all = select(main, "eval_all", "E")
    a_all = select(main, "eval_all", "A")
    c_all = select(main, "eval_all", "C")
    e_new = select(main, "eval_unseen", "E")
    a_new = select(main, "eval_unseen", "A")
    c_new = select(main, "eval_unseen", "C")
    delta = intervals["eval_all"]["balanced_accuracy_delta"]
    delta_new = intervals["eval_unseen"]["balanced_accuracy_delta"]
    hypothesis = (
        "Wynik wspiera hipotezę w protokole tego projektu."
        if intervals["eval_all"]["ci_low"] > 0 else
        "Wynik przemawia przeciw hipotezie w protokole tego projektu."
        if intervals["eval_all"]["ci_high"] < 0 else
        "Przedział dla różnicy obejmuje zero, więc wynik nie rozstrzyga hipotezy."
    )
    story = [
        P("RAPORT Z EKSPERYMENTU", "tag"),
        P("Porównanie metod<br/>wykrywania syntetycznej mowy", "cover"),
        P("Piotr Korpus, Jędrzej Stefanowski i Szymon Wojtasik<br/>25 listopada 2026", "intro"),
        cards([
            (pct(a_all.balanced_accuracy), "Balanced accuracy<br/>A · cechy czasowe"),
            (pct(e_all.balanced_accuracy), "Balanced accuracy<br/>E · wszystkie cechy"),
            (pp(delta_new, True), "Zmiana E względem A<br/>na nowych algorytmach"),
        ]),
        Spacer(1, 15),
        P("Wynik w skrócie", "sub"),
        P(
            f"Na pełnym teście różnica E względem A wyniosła <b>{pp(delta, True)}</b> w balanced "
            f"accuracy. Na 11 nowych algorytmach różnica wyniosła <b>{pp(delta_new, True)}</b>. "
            f"{hypothesis} EER dla A wyniósł <b>{pct(a_all.eer)}</b>, a dla E <b>{pct(e_all.eer)}</b>."
        ),
        P("Zbiór i filtracja", "sub"),
        P(
            f"Wczytano <b>{n(len(audit))}</b> nagrań opisanych w oficjalnych protokołach LA. "
            f"Do porównania po filtracji weszło <b>{n(len(current))}</b>. Do klasy syntetycznej "
            f"zaliczono syntezę TTS i konwersję głosu."
        ),
        table(stats_for_audit(audit, current), [.16, .13, .13, .14, .15, .15, .14]),
        Spacer(1, 12),
        plot("dlugosci.png", 148),
        Spacer(1, 4),
        P(
            "Rys. 1. Rozkład długości przed filtracją. Linia przerywana oznacza przyjęty próg 2 s. "
            "Oś kończy się na 10 s, a mianowniki obejmują wszystkie nagrania danej klasy i części.",
            "caption",
        ),
        PageBreak(),
    ]

    section(story, "METODYKA", "Pięć zestawów cech")
    raw_parts = audit.groupby("split")["speaker_id"].nunique()
    counts = audit["status"].value_counts()
    no_speech = counts.get("no_speech", 0)
    invalid = counts.get("invalid_or_silent", 0)
    duplicates = counts.get("duplicate_within_split", 0)
    cross = counts.get("duplicate_cross_split_or_label", 0)
    f0_missing = current["F0_mean"].isna().sum()
    story.extend([
        P("Podziały, jakość i kontrola przecieku", "sub"),
        P(
            f"Oficjalne części train, dev i eval pozostały rozłączne po identyfikatorach nagrań oraz "
            f"mówców. Protokoły zawierają odpowiednio {n(raw_parts['train'])}, {n(raw_parts['dev'])} "
            f"i {n(raw_parts['eval'])} różnych identyfikatorów mówców obu klas. Kopie PCM identyfikowano "
            f"sumą kontrolną zapisaną w nagłówku FLAC. Usunięto {n(duplicates)} powtórzeń w obrębie części "
            f"oraz {n(cross)} próbek z powtórzeniem między częściami lub konfliktem etykiety."
        ),
        P(
            f"Nagranie konwertowano do mono 16 kHz bez wyrównywania głośności. Odrzucono "
            f"{n(counts.get('too_short', 0))} nagrań &lt;2 s, {n(invalid)} niemal cichych lub "
            f"niepoprawnych oraz {n(no_speech)} bez segmentu mowy. W {n(f0_missing)} przyjętych "
            f"nagraniach brakowało F0. Jeden wiersz tabeli cech opisuje jedno przyjęte nagranie."
        ),
        P("Ekstrakcja", "sub"),
        table([
            ["Grupa", "Opis pomiaru"],
            ["Czasowe · 5", "Silero VAD v6.2.2, rozdzielczość 32 ms, progi 0,50 i 0,35, "
             "min. segment 250 ms i rozdzielająca cisza 100 ms, bez marginesu. SpeechAV i SpeechSD "
             "opisują długość odcinków mowy, SpeechProp ich udział w całym nagraniu. "
             "MiRate i MaRate liczą pauzy na minutę między pierwszym i ostatnim odcinkiem mowy "
             "(mikropauza 0,10–&lt;0,50 s, makropauza ≥0,50 s)."],
            ["Prozodyczne · 5", "Autokorelacja w Praat/Parselmouth (75–500 Hz, krok 10 ms), "
             "średnia F0, SD i rozstęp P95–P05 dla ramek dźwięcznych. "
             "Średnia i SD energii RMS dla ramek 25 ms co 10 ms."],
            ["Akustyczne · 30", "Ramki 25 ms co 10 ms, okno Hanna, FFT 512. "
             "40 filtrów mel i 13 MFCC (0–12), centroid widmowy i płaskość widma. "
             "Każdą cechę podsumowano średnią i SD po ramkach."],
        ], [.25, .75]),
        Spacer(1, 13),
        P("Trening i ocena", "sub"),
        P(
            "A to cechy czasowe. B to prozodia. C to akustyka. D łączy czasowe i prozodyczne, "
            "a E obejmuje wszystkie trzy grupy. Dla każdego wariantu uczono Random Forest "
            "ze 120 drzewami, min. 5 próbkami w liściu, wagami zbalansowanymi i domyślnym "
            "losowaniem √liczby cech. Braki uzupełniano medianą z treningu. Drzewa pracowały "
            "na naturalnej skali cech. Identyfikatory i długość nagrania zapisano jako metadane."
        ),
        P(
            "Próg dla każdego modelu wybrano na dev przez maksymalizację balanced accuracy "
            "i zamrożono przed oceną na eval. EER wyznaczono osobno z krzywej ROC przez "
            "interpolację w punkcie równych odsetków błędów. Pomiar powtórzono dla ziaren 42, 23 i 101."
        ),
        PageBreak(),
    ])

    section(
        story, "WYNIKI", "Pauzy, prozodia i widmo",
        "Wszystkie warianty oceniono na tych samych nagraniach. Synteza i konwersja głosu są klasą dodatnią.",
    )
    rows = [["Wariant", "BA", "Precision", "Recall", "F1", "EER", "Próg dev"]]
    groups = ["czasowe", "prozodyczne", "akustyczne", "czasowe + proz.", "pełny"]
    for variant, group in zip("ABCDE", groups, strict=True):
        v = select(main, "eval_all", variant)
        rows.append([
            f"<b>{variant}</b> · {group}", pct(v.balanced_accuracy), pct(v.precision),
            pct(v.recall), pct(v.f1), pct(v.eer), decimal(v.threshold, 3),
        ])
    story.extend([
        P(f"Pełny test eval · {n(e_all.n)} nagrań · ziarno 42", "sub"),
        table(rows, [.28, .11, .13, .12, .11, .11, .14], highlight=5),
        Spacer(1, 13),
        plot("metryki.png", 170),
        Spacer(1, 4),
        P(
            "Rys. 2. Balanced accuracy (BA) to średnia czułości dla obu klas. Niższy EER "
            "oznacza lepszą separację klas. Dev służył do wyboru progu i daje punkt odniesienia "
            "dla generatorów z treningu. Precision i F1 zależą od przewagi syntez w zbiorze.", "caption",
        ),
        P("Różnica E względem A oraz stabilność", "sub"),
    ])
    interval_rows = [["Test", "Zmiana BA<br/>dla ziarna 42", "95% PU<br/>bootstrap po mówcach", "Zakres zmiany BA<br/>dla trzech ziaren"]]
    for group, label in (("eval_all", "Pełny test"), ("eval_unseen", "Nowe algorytmy"),
                         ("eval_reference", "Algorytmy referencyjne")):
        info = intervals[group]
        by_seed = results[results["subset"].eq(group)].pivot(
            index="seed", columns="variant", values="balanced_accuracy"
        )
        gains = by_seed["E"] - by_seed["A"]
        interval_rows.append([
            label, pp(info["balanced_accuracy_delta"], True),
            f"od {pp(info['ci_low'], True)} do {pp(info['ci_high'], True)}",
            f"{pp(gains.min(), True)} do {pp(gains.max(), True)}",
        ])
    story.extend([
        table(interval_rows, [.25, .19, .28, .28]),
        Spacer(1, 9),
        P(
            f"W pełnym teście C osiągnął BA {pct(c_all.balanced_accuracy)} oraz EER "
            f"{pct(c_all.eer)}. Różnica E względem C wyniosła {pp(e_all.balanced_accuracy - c_all.balanced_accuracy, True)} "
            f"w BA. Przedziały dla E–A uzyskano z 2 000 powtórzeń losowania tych samych mówców "
            "dla obu modeli. Opisują niepewność na teście przy ustalonych modelach i progach."
        ),
        PageBreak(),
    ])

    section(
        story, "GENERALIZACJA", "Wyniki dla nowych algorytmów",
        f"Podzbiór obejmuje {n(e_new.n_spoof)} syntez z 11 nowych algorytmów i "
        f"{n(e_new.n_real)} nagrań naturalnych. Progi pochodzą z dev.",
    )
    rows = [["Wariant", "BA", "Precision", "Recall", "F1", "EER"]]
    for variant in "ABCDE":
        v = select(main, "eval_unseen", variant)
        rows.append([
            variant, pct(v.balanced_accuracy), pct(v.precision), pct(v.recall),
            pct(v.f1), pct(v.eer),
        ])
    story.extend([
        table(rows, [.18, .15, .18, .17, .15, .17], highlight=5),
        Spacer(1, 11),
        P(
            f"Trening i walidacja zawierały A01–A06. Podzbiór nowych algorytmów zawiera A07–A15 "
            f"i A17–A18. A16 oraz A19 tworzą podzbiór referencyjny, ponieważ powtarzają odpowiednio "
            f"algorytmy A04 i A06, wyuczone na innych danych. Na nowych algorytmach E miał EER "
            f"<b>{pct(e_new.eer)}</b>, A <b>{pct(a_new.eer)}</b>, a C <b>{pct(c_new.eer)}</b>."
        ),
        plot("generatory.png", 299),
        Spacer(1, 3),
        P(
            "Rys. 3. Odsetek błędnie przepuszczonych syntez dla poszczególnych generatorów. "
            "Gwiazdka oznacza system referencyjny.", "caption",
        ),
    ])
    worst = gen.loc[gen["variant"].eq("E") & ~gen["reference"]].nlargest(3, "miss_rate")
    names = ", ".join(f"{r.generator_id} ({pct(r.miss_rate)})" for r in worst.itertuples())
    a17, a18 = (gen.loc[gen["variant"].eq("E") & gen["generator_id"].eq(key)].iloc[0]
                for key in ("A17", "A18"))
    story.extend([
        P(f"Największy odsetek przeoczeń wariantu E wśród nowych algorytmów miały {names}. "
          f"A17 i A18 to systemy konwersji modyfikujące istniejącą wypowiedź. Dla A17 EER "
          f"wyniósł {pct(a17.eer)}, a dla A18 {pct(a18.eer)}, co wskazuje również na słabą separację wyników."),
        PageBreak(),
    ])

    section(story, "ANALIZA BŁĘDÓW", "Jakie nagrania sprawiają trudność?")
    story.extend([
        plot("macierze.png", 155),
        Spacer(1, 4),
        P("Rys. 4. Macierze pomyłek na nagraniach naturalnych i nowych algorytmach. "
          "Procenty liczono w obrębie klasy rzeczywistej.", "caption"),
        P(
            f"W porównaniu z A model E zmienił odsetek fałszywych alarmów z "
            f"{pct(a_new.false_alarm_rate)} do {pct(e_new.false_alarm_rate)}, "
            f"a odsetek przeoczonych syntez z {pct(a_new.miss_rate)} do {pct(e_new.miss_rate)}. "
            "Różnice opisują decyzje przy progach wybranych na dev."
        ),
    ])
    error_rows = [["Warunek w teście nowych algorytmów", "Nagrania", "BA A", "BA E"]]
    length_gains = {}
    for key, label in [
        ("short_2_to_4", "Długość od 2 do &lt;4 s"), ("long_4_plus", "Długość co najmniej 4 s"),
        ("no_internal_pause", "Bez wewnętrznej pauzy ≥100 ms"), ("at_least_one_pause", "Co najmniej jedna pauza ≥100 ms"),
    ]:
        sample = errors[errors["condition"].eq(key)].set_index("variant")
        length_gains[key] = sample.loc["E", "balanced_accuracy"] - sample.loc["A", "balanced_accuracy"]
        error_rows.append([
            label, n(sample.loc["A", "n"]), pct(sample.loc["A", "balanced_accuracy"]),
            pct(sample.loc["E", "balanced_accuracy"]),
        ])
    pause_absence = (
        current.loc[current["split"].eq("eval"), ["n_micro", "n_macro"]].sum(axis=1).eq(0).mean()
    )
    story.extend([
        table(error_rows, [.53, .17, .15, .15]),
        Spacer(1, 8),
        P(
            f"W pełnym eval {pct(pause_absence)} analizowanych nagrań nie miało ani jednej "
            "wykrytej pauzy wewnętrznej ≥100 ms. Wariant A może wtedy korzystać z długości i udziału "
            f"wykrytej mowy. Różnica BA między E i A wyniosła {pp(length_gains['short_2_to_4'], True)} "
            f"dla krótszych nagrań i {pp(length_gains['long_4_plus'], True)} od 4 s. W tych podgrupach "
            "różnią się także proporcje generatorów."
        ),
        P("Granice wniosku", "sub"),
        P(
            "Wyniki dotyczą warunków ASVspoof 2019 LA po filtracji. Błędy VAD zmieniają pomiar pauz, "
            "a błędy estymacji F0 i poziom sygnału wpływają na prozodię. Metryki nie określają "
            "skuteczności na polskiej mowie, nowych kodekach ani innych korpusach. "
            "W projekcie pominięto opcjonalny pomiar tempa przez Whisper i zbiór FoR."
        ),
        P("Źródła i odtwarzalność", "sub"),
        P(
            "W paczce projektu są skrypty, pięć modeli, tabele, predykcje i wykresy. Dane pochodzą z "
            + optional_link("ASVspoof 2019, University of Edinburgh CSTR", "https://doi.org/10.7488/ds/2555")
            + " ("
            + optional_link("ODC Attribution", "https://opendatacommons.org/licenses/by/1-0/")
            + "). Podział systemów opisali "
            + optional_link("Wang i in.", "https://arxiv.org/html/1911.01601")
            + " Definicje cech czasowych na podstawie "
            + optional_link("Kulangareth i in.", "https://doi.org/10.2196/56245")
            + " Silero VAD udostępnia "
            + optional_link("projekt Silero", "https://github.com/snakers4/silero-vad/releases/tag/v6.2.2")
            + ".",
            "caption",
        ),
    ])

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    doc = SimpleDocTemplate(
        str(target), pagesize=A4, topMargin=56, bottomMargin=52,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title="Porównanie metod wykrywania syntetycznej mowy",
        author="Piotr Korpus, Jędrzej Stefanowski, Szymon Wojtasik",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


if __name__ == "__main__":
    build()
