"""Torch-free tests of the ``semantic_v4`` text function (``semantic_text.py``), and the golden-fixture generator.

The golden fixture ``tests_data/semantic_v4_golden.jsonl`` holds about 200 stratified real records, one JSON object per
line::

    {doc_id, stratum, source, v8_text, v8_text_sha256, v3_sha256, v4_text, v4_sha256, eligible, reason}

``v8_text`` is the production text (``GenizahDocument.create_text_representation()``), so the same file pins the HDA
copy: its ``tests/data/semantic_v4_golden.jsonl`` is this file, and the HDA test recomputes ``semantic_v4_text`` /
``semantic_v4_ineligible_reason`` from ``v8_text`` and compares ``v4_sha256`` and ``reason``. ``semantic_v4_golden.meta.json``
records the inputs and the pins (boilerplate / admission sha256, v4-block sha256) the fixture was generated with.

Run::

    python3 test_semantic_v4.py                       # all tests (stdlib unittest; no torch, no model, no ES)
    python3 test_semantic_v4.py --write-golden \\
        --corpus $AR/corpus_v1.jsonl --corpus $V4/corpus_v1.jsonl   # regenerate the fixture (after a pin change)
"""

import argparse
import hashlib
import heapq
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import semantic_text as st

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "tests_data" / "semantic_v4_golden.jsonl"
GOLDEN_META = HERE / "tests_data" / "semantic_v4_golden.meta.json"

# The HDA tests' merged_record() production text (tests/test_text_representation.py V8_TEXT) and its v4 text
# (text-pipeline audit §2.4): both repos must agree on it.
HDA_V8_TEXT = (
    "Document ID: Cambridge_CUL_T_S_13J8_14\n"
    "Shelf Mark: T-S 13J8.14\n"
    "Description: Letter from Nahray b. Nissim to his partner about the flax trade. "
    "Joins T-S 13J8.14 and ENA 2558.5 (PGPID 12345).\n"
    "Alternate titles: Business letter\n"
    "Language: Judaeo-Arabic\n"
    "Document Type: letter\n"
    "Date: standard_date: 1060\n"
    "Transcription: [Editor: KTIV Academic transcription (Frag. 001r)] בשם רחמנא כתאבי אליך\n"
    "Related People: Nahray b. Nissim\n"
    "Related Places: Fustat"
)
HDA_V4_TEXT = (
    "Description: Letter from Nahray b. Nissim to his partner about the flax trade. Joins [shelfmark] and [shelfmark] "
    "([shelfmark]).; Business letter\nLanguage: Judaeo-Arabic\nDate: standard_date: 1060\n"
    "Transcription: בשם רחמנא כתאבי אליך\nRelated People: Nahray b. Nissim\nRelated Places: Fustat"
)
# admission patterns for the rule tests (independent of the live label code)
TEST_ADMIT = {"version": 1, "text_version": "semantic_v4", "source": "test",
              "flags": {"festival_terms": "i", "topic_rules": ""},
              "festival_terms": {"pesach": r"\bpassover\b|(?<![א-ת])[בהול]?פסח(?![א-ת])"},
              "topic_rules": {"shabbat": r"\bShabbat\b"}}


def write_test_admit(directory: Path) -> Path:
    """Write :data:`TEST_ADMIT` to a file.

    :param directory: Target directory.
    :returns: The file path.
    :rtype: Path
    """
    path = directory / "admit.json"
    path.write_text(json.dumps(TEST_ADMIT, ensure_ascii=False), encoding="utf-8")
    return path


class LayoutTests(unittest.TestCase):
    """The v4 layout over the v3 text."""

    def test_hda_merged_record(self) -> None:
        """The HDA test record gives the agreed v4 text."""
        v3 = st.semantic_text(HDA_V8_TEXT)
        self.assertEqual(st.semantic_v4_text(v3), HDA_V4_TEXT)

    def test_fold_field_lines(self) -> None:
        """Continuation lines join their field line with one space; blank lines go; a leading non-label stays."""
        text = "Description: first part\n  second part  \n\nRelated People: A\nB"
        self.assertEqual(st.fold_field_lines(text), "Description: first part second part\nRelated People: A B")
        self.assertEqual(st.fold_field_lines("loose line\nDescription: x"), "loose line\nDescription: x")
        self.assertIsInstance(st.fold_field_lines("Description: x"), str)

    def test_subjects_and_document_type_dropped(self) -> None:
        """Subjects and Document Type lines (with continuations) leave; other lines keep their order."""
        v3 = ("Description: Letter about flax\nSubjects: trade; India\nmore subjects\nLanguage: Judaeo-Arabic\n"
              "Document Type: letter\nDate: standard_date: 1060")
        self.assertEqual(st.semantic_v4_text(v3),
                         "Description: Letter about flax\nLanguage: Judaeo-Arabic\nDate: standard_date: 1060")

    def test_alt_titles_fold_into_description(self) -> None:
        """Parts already present as whole words (normalised) are left out; the rest follow in order."""
        v3 = ("Description: Talmud Bavli: Sukkah 29a\n"
              "Alternate titles: talmud bavli – sukkah 29a; Sukkah 29a–b; Shofar; SHOFAR\nLanguage: Hebrew")
        self.assertEqual(st.semantic_v4_text(v3),
                         "Description: Talmud Bavli: Sukkah 29a; Sukkah 29a–b; Shofar\nLanguage: Hebrew")

    def test_dedupe_is_whole_word(self) -> None:
        """A part that is only a substring of a word stays."""
        v3 = "Description: Piyyutim for Sukkot\nAlternate titles: Piyyut"
        self.assertEqual(st.semantic_v4_text(v3), "Description: Piyyutim for Sukkot; Piyyut")

    def test_dedupe_never_spans_two_titles(self) -> None:
        """A part made of the end of one title and the start of the next is not a duplicate."""
        v3 = "Description: Talmud Bavli\nAlternate titles: Sukkah 29a; Bavli Sukkah"
        self.assertEqual(st.semantic_v4_text(v3), "Description: Talmud Bavli; Sukkah 29a; Bavli Sukkah")

    def test_alt_titles_without_description(self) -> None:
        """Without a description the parts become the Description line, in the alternate titles' place."""
        v3 = "Alternate titles: Hoshanot; הושענות\nLanguage: Hebrew"
        self.assertEqual(st.semantic_v4_text(v3), "Description: Hoshanot; הושענות\nLanguage: Hebrew")
        self.assertEqual(st.semantic_v4_text("Alternate titles: – ; ..\nLanguage: Hebrew"), "Language: Hebrew")

    def test_cap_keeps_whole_parts(self) -> None:
        """Appending stops before the part that passes MAX_ALT_CHARS; the first part is kept whole."""
        parts = [c * 300 for c in "abcde"]
        kept, dup, capped = st.merge_title_parts("desc", parts)
        self.assertEqual((len(kept), dup, capped), (3, 0, 2))
        kept, _, capped = st.merge_title_parts("", ["x" * 1500, "y"])
        self.assertEqual((kept, capped), (["x" * 1500], 1))

    def test_frames_switch_is_the_rendering(self) -> None:
        """Frames rendered at the end of the Alternate titles line are folded like any part; the no-frames rendering
        gives the no-frames text, and the variants have their own version names."""
        with_frames = ("Document ID: X_1\nDescription: פיוט לסוכות\nAlternate titles: Yotzer for Sukkot; "
                       "[Piyyut] (Hoshanot): \"למען אמתך\"; 64 b – 65 a\nLanguage: Hebrew")
        no_frames = "Document ID: X_1\nDescription: פיוט לסוכות\nAlternate titles: Yotzer for Sukkot\nLanguage: Hebrew"
        self.assertEqual(st.semantic_v4_text(st.semantic_text(with_frames)),
                         "Description: פיוט לסוכות; Yotzer for Sukkot; [Piyyut] (Hoshanot): \"למען אמתך\"; 64 b – 65 a"
                         "\nLanguage: Hebrew")
        self.assertEqual(st.semantic_v4_text(st.semantic_text(no_frames)),
                         "Description: פיוט לסוכות; Yotzer for Sukkot\nLanguage: Hebrew")
        self.assertEqual(st.production_text({"text": with_frames, "text_no_frames": no_frames}, False), no_frames)
        self.assertEqual(st.production_text({"text": no_frames}, False), no_frames)
        self.assertEqual((st.v4_text_version(True), st.v4_text_version(False)), ("semantic_v4", "semantic_v4_nf"))

    def test_masking_and_editor_stripping_unchanged(self) -> None:
        """v4 is a layout over v3: identifiers stay masked exactly as v3 masked them, editor credits stay out."""
        v8 = ("Document ID: Cambridge_CUL_T_S_12_34\nShelf Mark: T-S 12.34\n"
              "Description: Letter; T-S 12.34 joins T-S NS 123.45 and ENA 2558.5 (PGPID 777).\n"
              "Alternate titles: Join with T-S 12.34\nLanguage: Judaeo-Arabic\n"
              "Transcription: Editor: Gil, Moshe Editor: Goitein, S. D. [Editor: KTIV Academic] שלום")
        v3 = st.semantic_text(v8)
        self.assertNotIn("12.34", v3)
        self.assertNotIn("PGPID", v3)
        self.assertNotIn("Editor", v3)
        self.assertEqual(st.semantic_v4_text(v3),
                         "Description: Letter; [shelfmark] joins [shelfmark] and [shelfmark] ([shelfmark]).; "
                         "Join with [shelfmark]\nLanguage: Judaeo-Arabic\nTranscription: שלום")

    def test_no_dropped_label_survives(self) -> None:
        """No Alternate titles / Subjects / Document Type label is left in a v4 text."""
        v3 = ("Alternate titles: A\nDescription: B\nAlternate titles: C\nSubjects: s\nDocument Type: t\n"
              "Translation: x")
        out = st.semantic_v4_text(v3)
        self.assertEqual(out, "Description: B; A; C\nTranslation: x")


class EligibilityTests(unittest.TestCase):
    """Eligibility v4: the v3 rule on the v3 text without Subjects lines, plus the admission list."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen labels and a test admission pattern."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.boiler = st.load_boilerplate_labels(st.BOILERPLATE_PATH)
        cls.admit = st.admit_pattern(write_test_admit(Path(cls.tmp.name)))

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary admission file."""
        cls.tmp.cleanup()

    def reason(self, v3: str) -> str:
        """v4 reason with the test inputs.

        :param v3: v3 text.
        :returns: Reason.
        :rtype: str
        """
        return st.semantic_v4_ineligible_reason(v3, self.boiler, self.admit)

    def test_basic_reasons(self) -> None:
        """empty / short / boilerplate_only / eligible."""
        self.assertEqual(self.reason(""), "empty")
        self.assertEqual(self.reason("Description: A small note"), "short")
        self.assertEqual(self.reason("Description: Piyyut\nLanguage: Hebrew\nDate: standard_date: 1100"),
                         "boilerplate_only")
        self.assertEqual(self.reason("Description: Letter of a merchant about pepper prices in Aden"), "eligible")

    def test_subjects_never_count(self) -> None:
        """A Subjects line neither lifts a text past MIN_CHARS nor counts as content."""
        self.assertEqual(self.reason("Description: Piyyut\nSubjects: Liturgy; Piyyut; Cairo Genizah fragments"),
                         "short")
        self.assertEqual(self.reason("Description: Piyyut\nLanguage: Hebrew\nDate: standard_date: 1100\n"
                                     "Subjects: Something specific here"), "boilerplate_only")
        self.assertEqual(self.reason("Subjects: Liturgy"), "empty")

    def test_admission(self) -> None:
        """A short text naming a festival / topic is admitted, case rules as flagged; metadata never admits."""
        self.assertEqual(self.reason("Description: הגדה של פסח (קטע)"), "eligible")
        self.assertEqual(self.reason("Description: Passover poem"), "eligible")
        self.assertEqual(self.reason("Description: Shabbat poem"), "eligible")
        self.assertEqual(self.reason("Description: shabbat poem"), "short")  # topic rules are case-sensitive
        self.assertEqual(self.reason("Description: Piyyut\nLanguage: Passover"), "short")  # metadata line
        self.assertEqual(self.reason("Alternate titles: Piyyut; לפסח"), "eligible")
        self.assertEqual(self.reason("Alternate titles: Piyyut; Bible: Texts"), "short")

    def test_alt_parts_compared_without_brackets(self) -> None:
        """A bracketed frame that restates a boilerplate label is not content."""
        self.assertIn("Bible: Texts", self.boiler)
        self.assertEqual(self.reason("Description: Bible: Texts\nAlternate titles: [Bible]: Texts\nLanguage: Hebrew"),
                         "boilerplate_only")

    def test_v4_admits_a_superset_of_v3(self) -> None:
        """On texts without Subjects lines every v3-eligible text is v4-eligible (golden fixture texts)."""
        for row in load_golden():
            v3 = st.semantic_text(row["v8_text"])
            if st.ineligible_reason(v3, self.boiler) == "eligible" and st.SUBJECTS_PREFIX not in v3:
                self.assertEqual(self.reason(v3), "eligible", row["doc_id"])


class FrozenInputTests(unittest.TestCase):
    """The frozen data files match their pins."""

    def test_boilerplate_pin(self) -> None:
        """The 64-label file is the v9 one."""
        self.assertEqual(st.file_sha256(st.BOILERPLATE_PATH), st.BOILERPLATE_SHA256)
        self.assertEqual(len(st.frozen_v4_labels()), 64)

    def test_admission_pin(self) -> None:
        """The admission file is the pinned one and compiles."""
        self.assertEqual(st.file_sha256(st.ADMIT_PATH), st.ADMIT_SHA256,
                         "eligibility_admit_v4.json differs from ADMIT_SHA256: regenerate deliberately "
                         "(build_admit_v4.py), update the pin in both copies, then --write-golden")
        self.assertTrue(st.frozen_admit_pattern().search("Passover"))

    def test_block_markers(self) -> None:
        """The byte-identical block is delimited."""
        block = st.v4_block_source(Path(st.__file__))
        self.assertTrue(block.startswith(st.V4_BLOCK_BEGIN))
        self.assertTrue(block.rstrip().endswith(st.V4_BLOCK_END))


class HashTests(unittest.TestCase):
    """Hash columns and the corpus writer."""

    def test_digests(self) -> None:
        """sha256 / md5 of UTF-8 bytes."""
        self.assertEqual(st.text_sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(st.text_md5("abc"), "900150983cd24fb0d6963f7d28e17f72")
        self.assertEqual(st.text_sha256("פסח"), hashlib.sha256("פסח".encode("utf-8")).hexdigest())

    def test_corpus_writer_and_v3_guard(self) -> None:
        """main() writes the contract columns; a planted v3 removal on an unchanged text fails the run."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus = tmp_path / "corpus_v1.jsonl"
            rows = [{"doc_id": "A_1", "text": "Document ID: A_1\nDescription: Letter of a merchant about pepper in Aden"},
                    {"doc_id": "B_2", "text": "Document ID: B_2\nDescription: הגדה של פסח (קטע)"},
                    {"doc_id": "C_3", "text": "Document ID: C_3\nDescription: Piyyut\nLanguage: Hebrew"}]
            corpus.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            admit = write_test_admit(tmp_path)
            out = tmp_path / "corpus_semantic.jsonl"
            base = [sys.executable, "-B", str(HERE / "semantic_text.py"), "--corpus", str(corpus), "--out", str(out),
                    "--text-version", "semantic_v4", "--admit-file", str(admit), "--allow-unpinned"]
            done = subprocess.run(base, capture_output=True, text=True, cwd=HERE, check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            got = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            keys = {"doc_id", "text", "text_v3", "text_version", "eligible", "reason", "text_hash", "text_sha256",
                    "series", "topics", "frames", "framed", "sukkot_gold"}
            for row in got:
                self.assertEqual(set(row), keys)
                self.assertEqual(row["text_hash"], st.text_md5(row["text"]))
                self.assertEqual(row["text_sha256"], st.text_sha256(row["text"]))
                self.assertEqual(row["text_version"], "semantic_v4")
            self.assertEqual([r["reason"] for r in got], ["eligible", "eligible", "short"])
            v3 = tmp_path / "v3.jsonl"  # C_3 planted as v3-eligible on the same text: a text-function removal
            v3.write_text("".join(json.dumps({"doc_id": r["doc_id"], "text": r["text_v3"], "eligible": True}) + "\n"
                                  for r in got), encoding="utf-8")
            done = subprocess.run(base + ["--v3-corpus", str(v3), "--accept-source-removals"], capture_output=True,
                                  text=True, cwd=HERE, check=False)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("C_3", done.stderr)


class GoldenTests(unittest.TestCase):
    """The shared golden fixture (also the HDA copy's)."""

    def test_golden_fixture(self) -> None:
        """Every fixture row reproduces: v3 sha, v4 text and sha, reason, eligible."""
        meta = json.loads(GOLDEN_META.read_text(encoding="utf-8"))
        self.assertEqual(meta["admit_sha256"], st.ADMIT_SHA256, "fixture made with another admission file")
        self.assertEqual(meta["boilerplate_sha256"], st.BOILERPLATE_SHA256)
        self.assertEqual(meta["fixture_sha256"], st.file_sha256(GOLDEN))
        rows = load_golden()
        self.assertGreaterEqual(len(rows), 150)
        for row in rows:
            v4, reason, v3 = st.semantic_v4_record(row["v8_text"])
            self.assertEqual(st.text_sha256(row["v8_text"]), row["v8_text_sha256"], row["doc_id"])
            self.assertEqual(st.text_sha256(v3), row["v3_sha256"], row["doc_id"])
            self.assertEqual(v4, row["v4_text"], row["doc_id"])
            self.assertEqual(st.text_sha256(v4), row["v4_sha256"], row["doc_id"])
            self.assertEqual(reason, row["reason"], row["doc_id"])
            self.assertEqual(reason == "eligible", row["eligible"], row["doc_id"])


def load_golden() -> List[dict]:
    """Rows of the golden fixture.

    :returns: Fixture rows.
    :rtype: List[dict]
    """
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------------------------------
# golden fixture generator
# ---------------------------------------------------------------------------------------------------------------------
Candidate = Dict[str, object]
# stratum -> (how many rows, predicate over a candidate)
STRATA: Dict[str, Tuple[int, Callable[[Candidate], bool]]] = {
    "ktiv_alt_titles": (14, lambda c: c["n_alt"] > 0 and c["group"] == "ktiv"),
    "alt_dedupe": (12, lambda c: c["n_dup"] > 0),
    "alt_capped": (8, lambda c: c["n_cap"] > 0),
    "alt_no_description": (6, lambda c: c["n_alt"] > 0 and "\nDescription:" not in f"\n{c['v3']}"),
    "multiline": (14, lambda c: c["n_folded"] > 0),
    "document_type_line": (10, lambda c: "\nDocument Type:" in f"\n{c['v3']}"),
    "editor_credit": (8, lambda c: bool(st.EDITOR_RE.search(c["v8"]) or st.PGP_EDITOR_RE.search(c["v8"]))),
    "masked_ids": (12, lambda c: st.MASK in c["v3"]),
    "boilerplate_only": (10, lambda c: c["reason"] == "boilerplate_only"),
    "short": (10, lambda c: c["reason"] == "short"),
    "short_admitted": (14, lambda c: c["reason_v3"] == "short" and c["reason"] == "eligible"),
    "empty": (4, lambda c: c["reason"] == "empty"),
    "long_transcription": (8, lambda c: "\nTranscription:" in f"\n{c['v4']}" and len(c["v4"]) > 1200),
    "metadata_lifted": (6, lambda c: c["reason"] == "eligible" and len(st.description_value(c["v3"])) < 27
                        and "\n" in c["v3"]),
    "frames": (14, lambda c: c["variant"] == "frames"),
    "frames_off": (10, lambda c: c["variant"] == "frames_off"),
    "subjects_line": (10, lambda c: st.SUBJECTS_PREFIX in c["v3"]),
    "ktiv_language_line": (8, lambda c: c["group"] == "ktiv" and "\nLanguage:" in f"\n{c['v3']}"),
    "random": (20, lambda c: True),
}


def corpus_candidates(path: Path, groups: Dict[str, str], boiler: frozenset) -> Iterable[Candidate]:
    """Every row (and its no-frames rendering) of a corpus as a fixture candidate.

    :param path: ``build_corpus.py`` output.
    :param groups: doc_id -> source group (from corpora that carry ``sources``).
    :param boiler: Frozen boilerplate labels.
    :yields: Candidates with the fields the strata test.
    """
    for row in st.iter_corpus(path):
        variants = [("frames" if st.TEXT_NO_FRAMES_FIELD in row else "plain", row["text"])]
        if st.TEXT_NO_FRAMES_FIELD in row:
            variants.append(("frames_off", row[st.TEXT_NO_FRAMES_FIELD]))
        for variant, v8 in variants:
            v4, reason, v3 = st.semantic_v4_record(v8)
            n_alt, _, n_dup, n_cap = st.title_merge_counts(v3)
            yield {"doc_id": row["doc_id"], "source": path.name, "variant": variant, "v8": v8, "v3": v3, "v4": v4,
                   "reason": reason, "reason_v3": st.ineligible_reason(st.v4_measured_text(v3), boiler),
                   "group": groups.get(row["doc_id"], ""), "n_alt": n_alt, "n_dup": n_dup, "n_cap": n_cap,
                   "n_folded": len(v3.split("\n")) - len(st.fold_field_lines(v3).split("\n"))}


def rank_key(stratum: str, cand: Candidate) -> int:
    """Deterministic pseudo-random rank of a candidate within a stratum.

    :param stratum: Stratum name.
    :param cand: Candidate.
    :returns: Integer key (smaller is picked first).
    :rtype: int
    """
    digest = hashlib.md5(f"{stratum}|{cand['source']}|{cand['variant']}|{cand['doc_id']}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def select_strata(candidates: Iterable[Candidate]) -> Dict[str, List[Candidate]]:
    """Keep, per stratum, the candidates with the smallest :func:`rank_key` (with spares for de-duplication).

    :param candidates: All candidates.
    :returns: stratum -> candidates in rank order.
    :rtype: Dict[str, List[Candidate]]
    """
    heaps: Dict[str, List[Tuple[int, int, Candidate]]] = {name: [] for name in STRATA}
    for serial, cand in enumerate(candidates):
        for name, (k, pred) in STRATA.items():
            if pred(cand):
                item = (-rank_key(name, cand), serial, cand)
                if len(heaps[name]) < 3 * k:
                    heapq.heappush(heaps[name], item)
                elif item > heaps[name][0]:
                    heapq.heapreplace(heaps[name], item)
    return {name: [c for _, _, c in sorted(heap, key=lambda it: (-it[0], it[1]))] for name, heap in heaps.items()}


def fixture_rows(picked: Dict[str, List[Candidate]]) -> List[dict]:
    """Fixture rows: per stratum the first ``k`` candidates whose (doc_id, variant) is not taken yet.

    :param picked: :func:`select_strata` output.
    :returns: Rows in stratum order.
    :rtype: List[dict]
    """
    taken, out = set(), []
    for name, (k, _) in STRATA.items():
        n = 0
        for cand in picked[name]:
            ident = (cand["doc_id"], cand["variant"])
            if ident in taken or n >= k:
                continue
            taken.add(ident)
            n += 1
            out.append({"doc_id": cand["doc_id"], "stratum": name, "source": cand["source"], "v8_text": cand["v8"],
                        "v8_text_sha256": st.text_sha256(cand["v8"]), "v3_sha256": st.text_sha256(cand["v3"]),
                        "v4_text": cand["v4"], "v4_sha256": st.text_sha256(cand["v4"]),
                        "eligible": cand["reason"] == "eligible", "reason": cand["reason"]})
    return out


def write_golden(corpora: List[Path]) -> None:
    """Regenerate the golden fixture and its meta file from the given corpora (the frozen inputs must be pinned).

    :param corpora: ``build_corpus.py`` outputs (e.g. the v9-era corpus and a v10 corpus with frames).
    """
    st.frozen_v4_labels()
    st.frozen_admit_pattern()  # refuses an unpinned admission file
    boiler = st.frozen_v4_labels()
    groups: Dict[str, str] = {}
    for path in corpora:
        for row in st.iter_corpus(path):
            if row.get("sources"):
                groups[row["doc_id"]] = "+".join(row["sources"])
    rows = fixture_rows(select_strata(c for path in corpora for c in corpus_candidates(path, groups, boiler)))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "evals/embedding_audit/test_semantic_v4.py --write-golden",
        "inputs": [{"path": str(p), "sha256": st.file_sha256(p)} for p in corpora],
        "boilerplate_sha256": st.BOILERPLATE_SHA256, "admit_sha256": st.ADMIT_SHA256,
        "v4_block_sha256": st.text_sha256(st.v4_block_source(Path(st.__file__))),
        "fixture_sha256": st.file_sha256(GOLDEN), "rows": len(rows),
        "strata": dict(Counter(r["stratum"] for r in rows)), "reasons": dict(Counter(r["reason"] for r in rows)),
    }
    GOLDEN_META.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1, ensure_ascii=False))


def main(argv: Optional[List[str]] = None) -> None:
    """Run the tests, or regenerate the fixture with ``--write-golden``.

    :param argv: Arguments (default ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--corpus", action="append", default=[], help="build_corpus.py output (repeatable)")
    args, rest = parser.parse_known_args(argv)
    if args.write_golden:
        if not args.corpus:
            parser.error("--write-golden needs at least one --corpus")
        write_golden([Path(p) for p in args.corpus])
        return
    unittest.main(argv=[sys.argv[0], *rest])


if __name__ == "__main__":
    main()
