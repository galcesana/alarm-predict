"""
Tel Aviv alert zone definitions and Gush Dan region mapping.

Tel Aviv is divided into 4 alert zones by Pikud HaOref.
The broader Gush Dan region includes surrounding cities that
are often warned together during large-scale missile attacks.
"""

from dataclasses import dataclass


@dataclass
class AlertZone:
    """Represents a single alert zone."""
    name_he: str
    name_en: str
    lat: float
    lng: float


# ── Tel Aviv's 4 alert zones ────────────────────────────────────────────────
# Centroids are approximate based on neighborhood boundaries
TEL_AVIV_ZONES = [
    AlertZone(
        name_he="תל אביב - עבר הירקון",
        name_en="Tel Aviv - North (Ever HaYarkon)",
        lat=32.1100,
        lng=34.7900,
    ),
    AlertZone(
        name_he="תל אביב - מרכז העיר",
        name_en="Tel Aviv - City Center",
        lat=32.0750,
        lng=34.7750,
    ),
    AlertZone(
        name_he="תל אביב - מזרח",
        name_en="Tel Aviv - East",
        lat=32.0700,
        lng=34.7950,
    ),
    AlertZone(
        name_he="תל אביב - דרום העיר ויפו",
        name_en="Tel Aviv - South & Jaffa",
        lat=32.0500,
        lng=34.7600,
    ),
]

# Hebrew names as they appear in the oref API data[] field
TEL_AVIV_ZONE_NAMES = {z.name_he for z in TEL_AVIV_ZONES}

# ── Gush Dan / Central Israel — cities that are often warned together ────────
# When a missile is aimed at the Tel Aviv area, these surrounding cities
# typically appear in the same warning. Their presence (or absence) in a
# warning is a strong feature for predicting whether TLV will be alarmed.
GUSH_DAN_CITIES = {
    # Tel Aviv zones
    "תל אביב - עבר הירקון",
    "תל אביב - מרכז העיר",
    "תל אביב - מזרח",
    "תל אביב - דרום העיר ויפו",
    # Surrounding cities
    "רמת גן",
    "גבעתיים",
    "בני ברק",
    "חולון",
    "בת ים",
    "ראשון לציון",
    "פתח תקווה",
    "הרצליה",
    "רעננה",
    "כפר סבא",
    "הוד השרון",
    "רמת השרון",
    "גבעת שמואל",
    "קריית אונו",
    "יהוד-מונוסון",
    "אור יהודה",
    "אזור",
    "לוד",
    "רמלה",
    "נתניה",
    "ראש העין",
    "אלעד",
    "מודיעין-מכבים-רעות",
    "שוהם",
    "גני תקווה",
}

# Cities near Tel Aviv with approximate coordinates — for distance features
GUSH_DAN_COORDS = {
    "רמת גן": (32.0700, 34.8243),
    "גבעתיים": (32.0714, 34.8117),
    "בני ברק": (32.0838, 34.8353),
    "חולון": (32.0116, 34.7872),
    "בת ים": (32.0171, 34.7516),
    "ראשון לציון": (31.9714, 34.7893),
    "פתח תקווה": (32.0841, 34.8878),
    "הרצליה": (32.1629, 34.7910),
    "רעננה": (32.1836, 34.8706),
    "כפר סבא": (32.1750, 34.9068),
    "רמת השרון": (32.1465, 34.8394),
    "גבעת שמואל": (32.0756, 34.8516),
    "קריית אונו": (32.0633, 34.8554),
    "יהוד-מונוסון": (32.0330, 34.8880),
    "אור יהודה": (32.0273, 34.8566),
    "נתניה": (32.3215, 34.8532),
}

# Approximate center of Tel Aviv (for distance calculations)
TEL_AVIV_CENTER = (32.0750, 34.7800)


def is_tel_aviv_zone(city_name: str) -> bool:
    """Check if a city name is one of Tel Aviv's alert zones."""
    return city_name in TEL_AVIV_ZONE_NAMES


def warning_includes_tel_aviv_region(warned_cities: list[str]) -> bool:
    """
    Check if a warning includes any city in the Gush Dan / TLV region.
    This indicates the warning is relevant for Tel Aviv prediction.
    """
    return bool(set(warned_cities) & GUSH_DAN_CITIES)


def count_gush_dan_cities(warned_cities: list[str]) -> int:
    """Count how many Gush Dan cities are in the warning."""
    return len(set(warned_cities) & GUSH_DAN_CITIES)


def count_tel_aviv_zones_in_warning(warned_cities: list[str]) -> int:
    """Count how many of the 4 TLV zones are in the warning."""
    return len(set(warned_cities) & TEL_AVIV_ZONE_NAMES)
