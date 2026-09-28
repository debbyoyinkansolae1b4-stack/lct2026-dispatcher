"""Оценка точности геокодирования.

Замечание из параллельной реализации, и оно верное: precision=street нельзя считать
подтверждённым домом, а fallback_level=0 говорит лишь о том, что сработал первый вариант
запроса, а не о том, что найден нужный дом.

Здесь считаем настоящее качество: сверяем номер дома из запроса с номером дома, который
вернул геокодер. Уровни:
  house  — номер дома совпал, точка на здании;
  street — дом не подтверждён, координата где-то на улице (ошибка до километра);
  wrong  — геокодер вернул другой дом (это хуже улицы: выглядит точно, а точка чужая);
  city   — не нашлось ни улицы, ни дома.
"""

import re

from src.data.addresses import normalize


def _parse_house(text: str) -> tuple[str, str, str]:
    """«128к5» → ('128', '5', ''); «2/1 к4» → ('2/1', '4', ''); «57 с14» → ('57', '', '14').

    Сравнивать подстрокой нельзя: «14» входит в «57с14», и чужой дом сойдёт за свой —
    ровно так один адрес на Авиамоторной уехал к соседнему дому.
    """
    t = str(text).lower().replace(" ", "")
    t = re.sub(r"корп(ус)?\.?", "к", t)
    t = re.sub(r"стр(оение)?\.?", "с", t)
    m = re.match(r"^([0-9]+(?:/[0-9]+)?[а-я]?)", t)
    main = m.group(1) if m else ""
    korp = (re.search(r"к([0-9]+[а-я]?)", t) or [None, ""])[1]
    stro = (re.search(r"с([0-9]+[а-я]?)", t) or [None, ""])[1]
    return main, korp or "", stro or ""


_STREET_WORDS = (r"улица|проспект|переулок|проезд|набережная|бульвар|шоссе|площадь|"
                 r"аллея|тупик|линия|квартал|микрорайон|ул|пр-кт|пер|наб|б-р|ш|пл")


def _norm_street(name: str) -> frozenset:
    """Название улицы → множество значимых слов.

    Сравнивать строкой нельзя: «1-й Советский проезд» и «Советский 1-й» — одна улица,
    порядок слов у геокодеров разный. А вот «Бирюлёвская» и «Салтыковская» — разные,
    и именно на этом офис участка уехал за несколько километров от настоящего.
    """
    s = str(name).lower().replace("ё", "е")
    s = re.sub(r"\([^)]*\)", " ", s)                      # «(дублёр)» и подобное
    s = re.sub(rf"\b({_STREET_WORDS})\b\.?", " ", s)      # тип улицы
    s = re.sub(r"(\d+)[-\s]*(го|й|я|ый|ой|е)\b", r"\1", s)  # «8-го» → «8», «1-й» → «1»
    words = [w for w in re.split(r"[^а-я0-9]+", s) if w]
    return frozenset(words)


def classify(raw_address: str, entry: dict) -> str:
    if not entry or entry.get("lat") is None:
        return "none"
    # Проверенная руками точка выигрывает у автоматической сверки: в OSM корпус и строение
    # часто путают, и формальное несовпадение номера здесь не ошибка, а известное
    # расхождение разметки. Такие записи несут поле note с объяснением.
    if entry.get("source") == "manual":
        return entry.get("quality", "house")
    parsed = normalize(raw_address)
    want = parsed["house"].lower().replace(" ", "")
    matched = str(entry.get("matched", "")).lower()
    precision = (entry.get("precision") or "").lower()

    # Сначала улица: совпадение номера дома ничего не значит, если улица другая.
    # Ровно так офис участка уехал с Бирюлёвской на Салтыковскую — дом «1с1» совпал.
    want_street = _norm_street(parsed["street"])
    got_street = _norm_street(entry.get("street", "")) if "street" in entry else frozenset()
    if want_street and got_street and not (want_street & got_street):
        return "wrong"

    if not want:
        return "street" if precision in ("street", "residential") else "house"

    # Приоритет — отдельному полю от геокодера. Разбор строки matched оставлен
    # только для старых записей кэша: у Photon номер в конце, у Nominatim в начале,
    # и одна регулярка на оба формата уже один раз подвела.
    if "housenumber" in entry:
        got_house = entry["housenumber"].strip()
    else:
        m = re.match(r"^\s*([0-9][^,]*)", matched)
        got_house = m.group(1).strip() if m else ""

    if got_house:
        w_main, w_k, w_s = _parse_house(want)
        g_main, g_k, g_s = _parse_house(got_house)
        if w_main and w_main == g_main and (not w_k or w_k == g_k) and (not w_s or w_s == g_s):
            return "house"
        return "wrong"
    if precision in ("house", "building"):
        return "house"          # номер не попал в строку, но объект — здание
    return "street" if precision in ("street", "residential") else "city"


def audit(cache: dict) -> dict:
    from collections import Counter
    marks = {addr: classify(addr, e) for addr, e in cache.items()}
    return {"counts": Counter(marks.values()), "marks": marks}
