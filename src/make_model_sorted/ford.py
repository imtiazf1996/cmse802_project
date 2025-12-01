# src/make_model_sorted/ford.py
# Normalization map for Ford models
# Keys = simplified first token (output of simplify_model_name)
# Values = canonical Ford model names

FORD_MODEL_MAP = {
    # ---------- F-Series ----------
    "f": "f-series",
    "f1oo": "f100",
    "f100": "f100",
    "f120": "f120",
    "f1250": "f250",     # typo
    "f140": "f150",      # old f-series often mis-labeled
    "f150": "f150",
    "f15o": "f150",
    "f150xl": "f150",
    "f150xlt": "f150",

    "f2": "f250",
    "f25": "f250",
    "f250": "f250",
    "f250f2": "f250",
    "f250hd": "f250",
    "f250sd": "f250",
    "f250xlt": "f250",

    "f350": "f350",
    "f350sd": "f350",
    "f350xlt": "f350",
    "f35o": "f350",

    "f360": "f350",   # mis-typed
    "f45": "f450",
    "f450": "f450",

    "f550": "f550",
    "f550sd": "f550",
    "550sd": "f550",
    "550": "f550",

    # ---------- Econoline / E-Series ----------
    "e": "econoline",
    "e150": "e150",
    "e250": "e250",
    "e350": "e350",
    "e35o": "e350",
    "e350sd": "e350",
    "e450": "e450",

    "econoline": "econoline",
    "econline": "econoline",
    "ecnoline": "econoline",
    "econoine": "econoline",
    "econo": "econoline",
    "econoline450": "e450",

    # ---------- Escape ----------
    "escape": "escape",
    "eacape": "escape",
    "ecape": "escape",
    "esape": "escape",
    "excape": "escape",

    # ---------- EcoSport ----------
    "ecosport": "ecosport",
    "echosport": "ecosport",

    # ---------- Edge ----------
    "edge": "edge",
    "egde": "edge",
    "edgetitanium": "edge",

    # ---------- Expedition ----------
    "expedition": "expedition",
    "expadition": "expedition",
    "expedtition": "expedition",
    "expedetion": "expedition",
    "expediton": "expedition",
    "exption": "expedition",

    # ---------- Explorer ----------
    "explorer": "explorer",
    "explor": "explorer",
    "explorerxlt": "explorer",
    "exploror": "explorer",
    "explorerer": "explorer",
    "exployer": "explorer",
    "explrer": "explorer",
    "exolorer": "explorer",

    # ---------- Excursion ----------
    "excursion": "excursion",
    "excusrion": "excursion",

    # ---------- C-Max ----------
    "cmax": "c-max",

    # ---------- Contour ----------
    "contour": "contour",

    # ---------- Crown Victoria ----------
    "crown": "crown victoria",
    "criwn": "crown victoria",
    "crow": "crown victoria",

    # ---------- Aerostar ----------
    "aerostar": "aerostar",

    # ---------- Transit & Transit Connect ----------
    "transit": "transit",
    "connect": "transit connect",

    # ---------- Misc. vans ----------
    "clubwagon": "club wagon",
    "club": "club wagon",
    "cargo": "cargo van",

    # ---------- Aspire ----------
    "aspire": "aspire",

    # ---------- Bronco ----------
    "bronco": "bronco",
    "bonco": "bronco",
    "beonco": "bronco",

    # ---------- Heavy trucks ----------
    "cf8000": "cf8000",
    "coe7000": "coe7000",

    # ---------- Rare / obscure ----------
    "custom": "custom",
    # --- more F-series / heavy trucks ---
    "f450sd": "f450",
    "f50": "f150",          # almost certainly F-150
    "f550": "f550",
    "f5500sd": "f550",
    "f550hd": "f550",

    "f59": "f59",

    "f600": "f600",
    "f650": "f650",
    "f650xl": "f650",
    "f650xlt": "f650",
    "f660": "f650",

    "f700": "f700",
    "f7000": "f7000",
    "f700f": "f7000",       # typo variant

    "f750": "f750",
    "f800": "f800",
    "f8oo": "f800",         # 8→oo typo

    # --- small cars / hatchbacks ---
    "festiva": "festiva",
    "fiesta": "fiesta",

    # --- Five Hundred ---
    "fivehundred": "five hundred",

    # --- Flex / Freestar / Freestyle ---
    "flex": "flex",
    "freestar": "freestar",
    "freestyle": "freestyle",

    # --- Focus ---
    "focus": "focus",
    "focust": "focus",
    "fo": "focus",          # very likely focus shorthand

    # --- Fox-body Mustang ---
    "foxbody": "mustang",

    # --- Fusion (and typos) ---
    "fusion": "fusion",
    "fuion": "fusion",
    "fuison": "fusion",
    "fussion": "fusion",

    # --- FX4 off-road trim (keep separate label) ---
    "fx4": "fx4",

    # --- GT / GT350 (Mustang performance models) ---
    "gt": "mustang gt",
    "gt350": "mustang gt350",
    # ---------- Lightning / L-series / LCF ----------
    "lightning": "lightning",
    "lighting": "lightning",       # typo

    "l8000": "l8000",
    "lnt8000": "l8000",
    "lt9513": "lt9513",
    "lcf": "lcf",

    # ---------- Lariat (keep as its own label) ----------
    "lariat": "lariat",

    # ---------- Taurus ----------
    "taurus": "taurus",
    "taurusx": "taurus",           # if present
    "tarsus": "taurus",
    "tarus": "taurus",
    "taurux": "taurus",
    "tuarus": "taurus",
    "turus": "taurus",
    "tuarus": "taurus",
    "turaus": "taurus",

    # ---------- Thunderbird / T-Bird ----------
    "thunderbird": "thunderbird",
    "tbird": "thunderbird",

    # ---------- Tempo ----------
    "tempo": "tempo",

    # ---------- Mustang & variants ----------
    "mustang": "mustang",
    "mustan": "mustang",
    "mustanggt": "mustang gt",
    "shelby": "mustang shelby",
    "sho": "taurus sho",          # usually Taurus SHO

    # ---------- Ranger / Raptor ----------
    "ranger": "ranger",
    "ranger2002": "ranger",
    "rnger": "ranger",
    "raptor": "raptor",

    # ---------- Probe / Pinto ----------
    "probe": "probe",
    "pinto": "pinto",

    # ---------- Windstar ----------
    "windstar": "windstar",
    "winstar": "windstar",

    # ---------- ZX2 ----------
    "zx2": "zx2",

    # ---------- Sport Trac ----------
    "sporttrac": "sport trac",
    "sporttrax": "sport trac",

    # ---------- Mariner (Mercury, but often under Ford) ----------
    "mariner": "mariner",
}
