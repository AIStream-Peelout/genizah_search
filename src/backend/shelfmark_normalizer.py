"""
Shelfmark Normalization for Cairo Genizah Documents

Provides canonical ID generation, variant generation, and bidirectional matching.
"""

import re
from typing import Dict, List, Optional, Set


class ShelfmarkNormalizer:
    """Normalize shelfmarks to canonical IDs and generate search variants"""

    COLLECTION_PREFIXES = {
        'T-S': 'T_S',
        'TS': 'T_S',
        'T S': 'T_S',
        'MS-TS-': 'T_S_',
        'T-S AS': 'T_S_AS',
        'TS AS': 'T_S_AS',
        'MS-TS-AS-': 'T_S_AS_',
        'T-S NS': 'T_S_NS',
        'TS NS': 'T_S_NS',
        'MS-TS-NS-': 'T_S_NS_',
        'T-S K': 'T_S_K',
        'MS-TS-K-': 'T_S_K_',
        'T-S Ar': 'T_S_Ar',
        'MS-TS-AR-': 'T_S_Ar_',
        'ENA': 'ENA',
        'ENA NS': 'ENA_NS',
        'L-G': 'L_G',
        'LG': 'L_G',
        'MS-L-G-': 'L_G_',
        'Mosseri': 'Mosseri',
        'Moss': 'Mosseri',
        'Moss.': 'Mosseri',
        'MS-MOSSERI-': 'Mosseri_',
        'CUL Or': 'CUL_Or',
        'MS-OR-': 'CUL_Or_',
        'Or': 'CUL_Or',       # bare "Or" prefix implies CUL Or (common in literature)
        'CUL Add': 'CUL_Add',
        'MS-ADD-': 'CUL_Add_',
        'Gaster A': 'Gaster_A',
        'Gaster B': 'Gaster_B',
        'Gaster P': 'Gaster_P',
        'Gaster L': 'Gaster_L',
        'Gaster C': 'Gaster_C',
        'Bodl': 'Bodl',
        'Bodl.': 'Bodl',
        'MS. Heb.': 'MS_Heb',
        'AIU': 'AIU',
        'BL Or': 'BL_Or',
        'BL Add': 'BL_Add',
        'Halper': 'Halper',
        'CAJS': 'CAJS',
    }

    INSTITUTION_PREFIXES = [
        'Cambridge',
        'Cambridge CUL',
        'Cambridge University Library',
        'JTS',
        'Jewish Theological Seminary',
        'Penn',
        'University of Pennsylvania',
        'Oxford',
        'Bodleian',
        'Paris',
    ]

    @staticmethod
    def to_canonical_id(shelfmark: str) -> str:
        if not shelfmark:
            return ""

        canonical = shelfmark.strip()

        for inst_prefix in ShelfmarkNormalizer.INSTITUTION_PREFIXES:
            pattern = rf'^{re.escape(inst_prefix)}\s*[:\s,]\s*'
            canonical = re.sub(pattern, '', canonical, flags=re.IGNORECASE)

        # Strip standalone "CUL:" / "CUL," prefix but NOT when followed by "Or" or "Add"
        # (those are collection names, not institution qualifiers)
        canonical = re.sub(r'^CUL\s*[:,]\s*(?!Or|Add)', '', canonical, flags=re.IGNORECASE)

        # "Box" is an obsolete physical-container term still used in older literature.
        # "CUL Or. 1080 Box 1.3" == "CUL Or.1080 1.3" — strip it everywhere.
        canonical = re.sub(r'\s+Box\s+', ' ', canonical, flags=re.IGNORECASE)

        canonical = canonical.replace("MS-TS-AS-", "T_S_AS_")
        canonical = canonical.replace("MS-TS-NS-", "T_S_NS_")
        canonical = canonical.replace("MS-TS-K-", "T_S_K_")
        canonical = canonical.replace("MS-TS-AR-", "T_S_Ar_")
        canonical = canonical.replace("MS-TS-", "T_S_")
        canonical = canonical.replace("MS-MOSSERI-", "Mosseri_")
        canonical = canonical.replace("MS-L-G-", "L_G_")
        canonical = canonical.replace("MS-OR-", "CUL_Or_")
        canonical = canonical.replace("MS-ADD-", "CUL_Add_")

        for variant, standard in sorted(ShelfmarkNormalizer.COLLECTION_PREFIXES.items(),
                                        key=lambda x: len(x[0]), reverse=True):
            if canonical.upper().startswith(variant.upper()):
                rest = canonical[len(variant):]
                canonical = standard + rest
                break

        canonical = canonical.replace(' ', '_')
        canonical = canonical.replace('.', '_')
        canonical = canonical.replace('-', '_')
        canonical = canonical.replace('/', '_')
        canonical = canonical.replace(',', '_')
        canonical = canonical.replace(':', '_')

        parts = canonical.split('_')
        normalized_parts = []
        for part in parts:
            if part.isdigit():
                normalized_parts.append(str(int(part)))
            else:
                normalized_parts.append(part)
        canonical = '_'.join(normalized_parts)

        while '__' in canonical:
            canonical = canonical.replace('__', '_')

        canonical = canonical.strip('_')

        return canonical

    @staticmethod
    def are_equivalent(shelfmark1: str, shelfmark2: str) -> bool:
        id1 = ShelfmarkNormalizer.to_canonical_id(shelfmark1)
        id2 = ShelfmarkNormalizer.to_canonical_id(shelfmark2)
        return id1 == id2

    INSTITUTION_MAPPING: Dict[str, Dict[str, Optional[str]]] = {
        "T-S": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": None},
        "T-S AS": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Additional Series"},
        "T-S NS": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "New Series"},
        "T-S Ar": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Arabic"},
        "T-S K": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Miscellaneous"},
        "CUL Add": {"institution": "Cambridge University Library", "collection": "Additional Manuscripts", "subcollection": None},
        "CUL Or": {"institution": "Cambridge University Library", "collection": "Oriental Manuscripts", "subcollection": None},
        "Mosseri": {"institution": "Cambridge University Library", "collection": "Mosseri", "subcollection": None},
        "L-G": {"institution": "Cambridge University Library / Bodleian Library Oxford", "collection": "Lewis-Gibson", "subcollection": None},
        "ENA": {"institution": "Jewish Theological Seminary", "collection": "Elkan Nathan Adler", "subcollection": "Main series"},
        "ENA NS": {"institution": "Jewish Theological Seminary", "collection": "Elkan Nathan Adler", "subcollection": "New Series"},
        "Gaster A": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series A"},
        "Gaster B": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series B"},
        "Gaster P": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series P"},
        "MS. Heb.": {"institution": "Bodleian Library, Oxford", "collection": "Hebrew Manuscripts", "subcollection": None},
        "Halper": {"institution": "University of Pennsylvania, Katz Center", "collection": "Dropsie College", "subcollection": None},
        "AIU": {"institution": "Alliance Israélite Universelle, Paris", "collection": "Cairo Genizah", "subcollection": None},
        "BL Or": {"institution": "British Library, London", "collection": "Oriental Manuscripts", "subcollection": None},
        "BL Add": {"institution": "British Library, London", "collection": "Additional Manuscripts", "subcollection": None},
    }


# Compiled regex for detecting shelf mark candidates in free text.
# Ordered from most-specific (longest prefix) to least-specific.
SHELFMARK_PATTERN = re.compile(
    r'\b('
    # Taylor-Schechter subseries (must come before plain T-S)
    r'T-S\s+AS\s+\d+[\w.]*'
    r'|T-S\s+NS\s+\d+[\w.]*'
    r'|T-S\s+K\s+\d+[\w.]*'
    r'|T-S\s+Ar\s+\d+[\w.]*'
    r'|TS\s+AS\s+\d+[\w.]*'
    r'|TS\s+NS\s+\d+[\w.]*'
    # Plain T-S (e.g. T-S 16.375, T-S 8J22.22, T-S A12.10)
    r'|T-S\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*'
    r'|T-S\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    r'|TS\s+[A-Za-z]\d+[\w.]*(?:[.\-]\d+)*'
    r'|TS\s+\d+[A-Za-z]*\d*(?:[.\-]\d+)*'
    # ENA
    r'|ENA\s+NS\s+\d+[\w.]*'
    r'|ENA\s+\d+[\w.]*'
    # Lewis-Gibson
    r'|L-G\s+[A-Za-z0-9]+[\w.]*'
    # Mosseri
    r'|Mosseri\s+[A-Za-z0-9]+[\w.]*'
    r'|Moss\.\s+[A-Za-z0-9]+[\w.]*'
    # CUL Or / Add — with or without CUL prefix; Box is stripped at normalization
    r'|CUL\s+Or\.?\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?\d+(?:\.\d+)?)?'
    r'|CUL\s+Add\s+\d+[\w.]*'
    # Bare "Or" prefix (CUL implied) — must have period or number immediately after
    r'|Or\.\s*\d+(?:\.\d+)?(?:\s+(?:Box\s+)?\d+(?:\.\d+)?)?'
    # Manchester sub-series (most specific first)
    r'|Manchester\s*[:\-]?\s*Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+'
    r'|Manchester\s*[:\-]?\s*Gaster\s+Printed\s+Series\s+\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Gaster\s+\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*Glass\s+\d+[\w.]*'
    r'|Manchester\s*[:\-]?\s*[ABCGLP]\s+\d+[\w.]*'
    # Standalone Rylands Genizah (when Manchester prefix is absent)
    r'|Rylands\s+Genizah\s+(?:Fragment\s+|Frag\.\s*)?\d+'
    # Gaster
    r'|Gaster\s+[A-Z]\s+\d+[\w.]*'
    # BL
    r'|BL\s+Or\s+\d+[\w.]*'
    r'|BL\s+Add\s+\d+[\w.]*'
    # Halper
    r'|Halper\s+\d+'
    # AIU
    r'|AIU\s+[IVX]+\.[A-Z]\.\d+'
    r')',
    re.IGNORECASE
)


def detect_shelfmarks(text: str) -> List[str]:
    """Return deduplicated list of shelf mark candidates found in text."""
    if not text:
        return []
    seen: Set[str] = set()
    results: List[str] = []
    for match in SHELFMARK_PATTERN.finditer(text):
        sm = match.group(1).strip()
        if sm.lower() not in seen:
            seen.add(sm.lower())
            results.append(sm)
    return results
