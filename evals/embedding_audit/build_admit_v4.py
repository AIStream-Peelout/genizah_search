"""Write the frozen ``semantic_v4`` admission list (``docs/handoffs/v10_embedder/eligibility_admit_v4.json``).

Eligibility v4 admits a ``short`` record (under ``MIN_CHARS`` characters) when its content names a festival or a
topic (plan B3). The patterns are ``build_subjects.FESTIVAL_TERMS`` (case-insensitive) and
``build_subjects.TOPIC_RULES_V2`` (case-sensitive), imported when this script runs and written out verbatim, so the
embedded-text contract never depends on the live label code: the file is frozen, sha-pinned
(``semantic_text.ADMIT_SHA256``, same constant in the HDA copy) and copied byte for byte to historical-document-analysis
``src/embeddings/data/eligibility_admit_v4.json``. ``TOPIC_RULES_LITURGY`` ("שבועות" is also the tractate on oaths)
is left out on purpose.

The output is deterministic (sorted keys, no timestamp): regenerating from unchanged patterns gives the same bytes and
sha. Run it again after every ``FESTIVAL_TERMS`` / ``TOPIC_RULES_V2`` change, then update ``ADMIT_SHA256`` in both
``semantic_text.py`` copies and regenerate the golden fixture (``test_semantic_v4.py --write-golden``)::

    python3 build_admit_v4.py            # write + report the sha and whether it matches the pin
    python3 build_admit_v4.py --check    # exit 1 when the file on disk is not what the label code gives today
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Pattern

from semantic_text import ADMIT_PATH, ADMIT_SHA256, admit_pattern, text_sha256

SOURCE = ("genizah_search evals/embedding_audit/build_subjects.py: FESTIVAL_TERMS (re.I) + TOPIC_RULES_V2 "
          "(case-sensitive), verbatim; TOPIC_RULES_LITURGY excluded")
# strings every generation must admit / reject (a guard against an empty or broken pattern set, not a label test)
MUST_ADMIT = ("Passover Haggadah", "הגדה של פסח", "Piyyut for Sukkot", "Hanukkah", "Shabbat")
MUST_REJECT = ("Letter", "Piyyut", "Accounts.", "Bible: Texts")


def pattern_flags(compiled: Pattern[str]) -> str:
    """The flag letters of a compiled label pattern this file can carry (only ``i``).

    :param compiled: A ``build_subjects`` compiled pattern.
    :returns: ``"i"`` or ``""``.
    :rtype: str
    :raises ValueError: For any flag other than IGNORECASE (and the implicit UNICODE).
    """
    extra = compiled.flags & ~(re.IGNORECASE | re.UNICODE)
    if extra:
        raise ValueError(f"pattern {compiled.pattern[:60]!r} uses flags {extra}; the admission file carries only 'i'")
    return "i" if compiled.flags & re.IGNORECASE else ""


def block_flags(patterns: Dict[str, Pattern[str]]) -> str:
    """The single flag string shared by every pattern of one block.

    :param patterns: Topic -> compiled pattern.
    :returns: Flag letters.
    :rtype: str
    :raises ValueError: When the patterns of the block disagree.
    """
    flags = {pattern_flags(rx) for rx in patterns.values()}
    if len(flags) != 1:
        raise ValueError(f"mixed flags in one block: {sorted(flags)}")
    return flags.pop()


def build_admission() -> dict:
    """The admission document from the live label code.

    :returns: ``{version, text_version, source, flags, festival_terms, topic_rules}``.
    :rtype: dict
    """
    import build_subjects as bs  # imported here: the label code may be mid-edit when this module is imported

    return {
        "version": 1,
        "text_version": "semantic_v4",
        "source": SOURCE,
        "flags": {"festival_terms": block_flags(bs.FESTIVAL_RE), "topic_rules": block_flags(bs.TOPIC_RE)},
        "festival_terms": {t: bs.FESTIVAL_TERMS[t] for t in sorted(bs.FESTIVAL_TERMS)},
        "topic_rules": {t: bs.TOPIC_RULES_V2[t] for t in sorted(bs.TOPIC_RULES_V2)},
    }


def render(doc: dict) -> str:
    """Deterministic file content.

    :param doc: Admission document.
    :returns: JSON text with a trailing newline.
    :rtype: str
    """
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def check_patterns(path: Path) -> None:
    """Compile a written admission file through ``semantic_text.admit_pattern`` and run the guard strings.

    :param path: Admission file.
    :raises AssertionError: When a guard string is not admitted / rejected as expected.
    """
    admit_pattern.cache_clear()
    rx = admit_pattern(path)
    bad = [s for s in MUST_ADMIT if not rx.search(s)] + [f"!{s}" for s in MUST_REJECT if rx.search(s)]
    assert not bad, f"admission guard failed for {bad}"


def main() -> None:
    """Write (or check) the admission file and report its sha against the pin."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ADMIT_PATH))
    parser.add_argument("--check", action="store_true", help="compare the file on disk; write nothing")
    args = parser.parse_args()
    content = render(build_admission())
    sha = text_sha256(content)
    out = Path(args.out)
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        same = current == content
        print(json.dumps({"file": str(out), "matches_label_code": same, "generated_sha256": sha,
                          "file_sha256": text_sha256(current) if current else None, "pinned": ADMIT_SHA256}, indent=1))
        sys.exit(0 if same else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    check_patterns(out)
    doc = json.loads(content)
    print(json.dumps({"file": str(out), "sha256": sha, "pinned_sha256": ADMIT_SHA256, "matches_pin": sha == ADMIT_SHA256,
                      "festival_terms": len(doc["festival_terms"]), "topic_rules": len(doc["topic_rules"])}, indent=1))
    if sha != ADMIT_SHA256:
        print(f"\nNOT PINNED. To freeze this file: set ADMIT_SHA256 = \"{sha}\" in genizah_search "
              "evals/embedding_audit/semantic_text.py and HDA src/embeddings/semantic_text.py, copy the file to HDA "
              "src/embeddings/data/, then run `python3 test_semantic_v4.py --write-golden` and `python3 "
              "test_semantic_v4.py`.", file=sys.stderr)


if __name__ == "__main__":
    main()
