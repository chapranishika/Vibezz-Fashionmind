"""
Lightweight India festive calendar for the trend feed.

`current_festive(today=None)` -> dict | None
    Returns the festival that is "in season" — within 3 days past or 40 days
    ahead — with shopping keywords and a short line, so the UI can always show
    a festive edit when one is near.
"""
from __future__ import annotations

from datetime import date, timedelta

# (name, month, day, keywords, blurb)
_FESTIVALS = [
    ("Makar Sankranti / Pongal", 1, 14, ["cotton saree", "bandhani dupatta", "yellow kurta"],
     "Bright cottons, bandhani and sunny yellows for the harvest festival."),
    ("Republic Day", 1, 26, ["tricolour", "handloom saree", "khadi kurta"],
     "Handloom and khadi in tricolour tones."),
    ("Holi", 3, 14, ["white kurta", "cotton co-ord", "festival dress"],
     "White and pastel cottons that take colour well."),
    ("Eid", 3, 30, ["anarkali", "sharara set", "silk kurta", "bandhgala"],
     "Anarkalis, sharara sets and fine silk kurtas for Eid."),
    ("Baisakhi", 4, 13, ["phulkari dupatta", "patiala salwar", "kurta set"],
     "Phulkari, patiala salwars and festive kurta sets."),
    ("Raksha Bandhan", 8, 9, ["kurta set", "silk saree", "ethnic co-ord"],
     "Easy silk sarees and kurta sets for Rakhi."),
    ("Janmashtami", 8, 16, ["lehenga", "dhoti kurta", "peacock print"],
     "Lehengas, dhoti-kurtas and peacock motifs."),
    ("Ganesh Chaturthi", 8, 27, ["nauvari saree", "kurta pyjama", "brocade blouse"],
     "Nauvari drapes, brocade blouses and crisp kurta-pyjamas."),
    ("Onam", 9, 5, ["kasavu saree", "off-white kurta", "gold border"],
     "Kasavu off-white and gold-border sets for Onam."),
    ("Navratri", 10, 3, ["chaniya choli", "mirror-work lehenga", "bandhani kurta", "oxidised jewellery"],
     "Chaniya cholis, mirror-work and oxidised jewellery for nine nights of garba."),
    ("Durga Puja", 10, 9, ["red-border saree", "silk saree", "dhakai jamdani"],
     "Red-and-white bordered sarees and jamdani for Pujo."),
    ("Karva Chauth", 10, 20, ["red lehenga", "zari saree", "kundan set"],
     "Reds, zari sarees and kundan sets."),
    ("Dussehra", 10, 22, ["silk kurta", "bandhgala", "anarkali"],
     "Silk kurtas, bandhgalas and anarkalis."),
    ("Diwali", 11, 8, ["sequin lehenga", "organza saree", "sherwani", "indo-western", "potli bag", "juttis"],
     "Sequins, organza, indo-western sets and everything that catches diya light."),
    ("Bhai Dooj", 11, 10, ["kurta set", "cotton saree", "nehru jacket"],
     "Understated kurta sets and cotton sarees."),
    ("Christmas", 12, 25, ["party dress", "sequin top", "velvet blazer", "red dress"],
     "Party dresses, sequins and velvet in festive reds and greens."),
    ("New Year's Eve", 12, 31, ["metallic dress", "sequin skirt", "going-out top"],
     "Metallics and sequins for the countdown."),
]


def _occurrences(f, today: date):
    """Yield this festival's date in the previous, current and next year."""
    _, m, d, *_ = f
    for y in (today.year - 1, today.year, today.year + 1):
        try:
            yield date(y, m, d)
        except ValueError:
            pass


def current_festive(today: date | None = None, past_days: int = 3, ahead_days: int = 40):
    today = today or date.today()
    best = None
    for f in _FESTIVALS:
        name, _, _, keywords, blurb = f
        for occ in _occurrences(f, today):
            delta = (occ - today).days
            if -past_days <= delta <= ahead_days:
                if best is None or abs(delta) < abs(best["days_away"]):
                    best = {"name": name, "date": occ.isoformat(), "days_away": delta,
                            "keywords": keywords, "blurb": blurb}
    return best


if __name__ == "__main__":
    import json
    print(json.dumps(current_festive(), indent=2))
