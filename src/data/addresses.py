"""Нормализация адресов из выгрузки под геокодер.

В данных адреса разноформатные: «Город Москва, пр-кт.Волгоградский, д. 128 к 5, кв. 1»,
«г.Город Москва, наб.Семеновская, д. 2/1», «Москва Бирюлевская ул. д. 44»,
«Домодедово, проезд.Советский 1-й, д. 1А». Nominatim такое в лоб не берёт.
"""

import re

# Сокращения типов улиц → полные слова
STREET_TYPES = {
    "ул": "улица", "пр-кт": "проспект", "пр-т": "проспект", "просп": "проспект",
    "пер": "переулок", "наб": "набережная", "б-р": "бульвар", "бул": "бульвар",
    "ш": "шоссе", "проезд": "проезд", "пл": "площадь", "туп": "тупик",
    "аллея": "аллея", "линия": "линия", "кв-л": "квартал", "мкр": "микрорайон",
}

CITY_FIX = [
    (r"^\s*МО\s*,\s*", ""),                    # «МО, г. Кашира …» — область подразумевается
    (r"^\s*Московская\s+область\s*,\s*", ""),
    (r"^\s*г\.?\s*Город\s+Москва", "Москва"),
    (r"^\s*Город\s+Москва", "Москва"),
    (r"^\s*г\.?\s*Москва", "Москва"),
]


def _clean_city(s: str) -> str:
    for pat, repl in CITY_FIX:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _house(raw: str) -> str:
    """«д. 128 к 5» → «128к5», «д 83с 4» → «83с4», «д. 2/1» → «2/1»."""
    h = re.sub(r"^\s*д\.?\s*", "", raw.strip(), flags=re.IGNORECASE)
    h = re.sub(r"\s*к\.?\s*(\d)", r"к\1", h, flags=re.IGNORECASE)   # корпус
    h = re.sub(r"\s*с\.?\s*(\d)", r"с\1", h, flags=re.IGNORECASE)   # строение
    h = re.sub(r"\s*стр\.?\s*(\d)", r"с\1", h, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", h)


def normalize(raw: str) -> dict:
    """Разбирает адрес на город / тип улицы / название / дом. Возвращает и запросы к геокодеру."""
    s = " ".join(str(raw).split())
    s = re.sub(r",?\s*кв\.?\s*\d+\S*", "", s, flags=re.IGNORECASE)   # квартиру выкидываем
    s = _clean_city(s)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    city = parts[0] if parts else "Москва"
    city = re.sub(r"^г\.?\s+", "", city)        # «г. Кашира Центральная ул.» → город + улица
    rest = parts[1:]

    # Дом — последний кусок, начинающийся с «д» или с цифры
    house = ""
    if rest and re.match(r"^(д\.?\s*)?\d", rest[-1], flags=re.IGNORECASE):
        house = _house(rest.pop())

    street_raw = ", ".join(rest) if rest else ""

    # Адрес без запятых: «Москва Бирюлевская ул. д. 44»
    if not street_raw and " " in city:
        tokens = city.split()
        city = tokens[0]
        tail = " ".join(tokens[1:])
        m = re.search(r"\bд\.?\s*(\S+)", tail, flags=re.IGNORECASE)
        if m:
            house = _house(m.group(0))
            tail = tail[:m.start()].strip()
        street_raw = tail

    stype, sname = "", street_raw
    m = re.match(r"^([А-Яа-яЁё\-]+)\.\s*(.+)$", street_raw)      # «ул.Грайвороновская»
    if m and m.group(1).lower().rstrip(".") in STREET_TYPES:
        stype = STREET_TYPES[m.group(1).lower().rstrip(".")]
        sname = m.group(2).strip()
    else:
        m = re.match(r"^([А-Яа-яЁё\-]+)\s+(.+)$", street_raw)     # «ул Юных Ленинцев»
        if m and m.group(1).lower().rstrip(".") in STREET_TYPES:
            stype = STREET_TYPES[m.group(1).lower().rstrip(".")]
            sname = m.group(2).strip()
        else:
            m = re.match(r"^(.+?)\s+([А-Яа-яЁё\-]+)\.?$", street_raw)   # «Бирюлевская ул.»
            if m and m.group(2).lower().rstrip(".") in STREET_TYPES:
                stype = STREET_TYPES[m.group(2).lower().rstrip(".")]
                sname = m.group(1).strip()

    sname = re.sub(r"^(пр-зд|проезд)\.?\s*", "", sname, flags=re.IGNORECASE).strip()

    # Два порядка слов: Nominatim принимает оба, но по-разному промахивается
    queries = []
    if sname and stype:
        queries.append(f"{city}, {stype} {sname}, {house}".rstrip(", "))
        queries.append(f"{city}, {sname} {stype}, {house}".rstrip(", "))
    elif sname:
        queries.append(f"{city}, {sname}, {house}".rstrip(", "))
    if sname and stype and house:
        queries.append(f"{city}, {stype} {sname}")      # без дома — хотя бы улица

    # Дом без корпуса и строения: «9Бс1» → «9Б». В OSM корпус часто отдельным объектом
    # не заведён, и по полному номеру геокодер уходит на центр улицы.
    if house and re.search(r"[кс]\d", house):
        base_house = re.split(r"[кс]\d", house)[0]
        if base_house and sname:
            queries.append(f"{city}, {stype} {sname}, {base_house}".replace("  ", " ").strip(", "))

    # ё и е в названиях улиц пишут как придётся: Бирюлевская против Бирюлёвской
    swapped = []
    for q in queries:
        for a, b in (("ё", "е"), ("е", "ё")):
            alt = q.replace(a, b)
            if alt != q:
                swapped.append(alt)
    queries.extend(swapped)

    return {
        "city": city, "street_type": stype, "street": sname, "house": house,
        "queries": [q for q in dict.fromkeys(queries) if q],
    }
