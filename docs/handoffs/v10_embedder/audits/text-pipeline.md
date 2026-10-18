# semantic_v4 audit: document text and index pipeline (2026-10-09)

**Method.** I streamed today's merged file (`merged_shelfmarks.jsonl`, 2026-10-08 00:05, 78,113 records) through HDA's own `GenizahDocument.from_merged_format()` and `create_text_representation()`. No torch, models, ES or docker were used, and nothing in either repo or on the NAS was modified.

There are two scans:
- **HEAD**: HDA commit `069af54` document model plus the genizah_search reference `semantic_text.py`. This is the text v9 was trained on.
- **WC**: a snapshot of HDA's uncommitted working copy taken at 13:02:36 EDT.

Token counts use the fine-tune's local `tokenizer.json` with truncation and padding switched off; counts include EOS. Scripts and outputs are in `SCR/audit_textpipe/` (`scan_today.py`, `analyze*.py`, `semantic_v4_proposed.py`, `run_v4.py`, `today_{head,wc}.jsonl`).

The HEAD scan reproduces the built `genizah_merged_v9` exactly: 63,652 records with a vector and 14,461 without, matching HDA `docs/handoffs/v9_index_switch_mbp.md:13-16`.

## Urgent: someone else's HDA working copy is changing the "semantic_v3" text in place

Between 12:58 and 13:06 EDT, while this audit ran, the HDA working copy changed under me:
- `genizah_document.py` now adds KTIV scholarly `frame` and `title` values to `alt_titles`.
- It adds a new `subjects` field, written into the v8 text as a `Subjects:` line.
- `semantic_text.content_lines` was changed: `Subjects:` is skipped and `_unbracketed` alternate-title parts are compared.
- `KTIV_GENERIC_WORK_TITLES` gained "leqakh tov" / "לקח טוב".

The version string stays `semantic_v3`. `src/datasets/indexing/update_catalogue_enrichment.py` re-embeds changed documents into v9 with the v9 model (`v9_index_switch_mbp.md:18-30`).

Impact at the 13:02 snapshot:
- **Changed texts:** 28,321 of 78,113 records have a different text (28,311 gained `Subjects:`, 8,919 changed alternate-title line, 1,179 new alternate-title line).
- **Parity with the training corpus:** against the Oct-5 training corpus, only 48,332 of 73,156 shared ids keep an identical text, versus 71,635 (97.9%) with HEAD.
- **No check catches it:**
  - The writer guard only compares `text_representation_version`.
  - The verifier's self-consistency check (`verify_reembedded_indexes.py:315`) re-derives text with the same new code, so it passes.
  - The document canary only fails if that one record changed.
- **Inaccurate claim in the handoff:** it says "a Subjects: line never makes a record eligible on its own". In the WC code, 349 records are eligible only because the `Subjects:` line lifts the text past 40 characters.

The user should hear about this before more v9 vectors are rewritten. Whatever of it is kept belongs in semantic_v4 and must be in the training corpus. Recommendations for each piece are below.

---

## 1. Where the text comes from

**The chain.** `merged["description"]` is chosen in `merge_shelfmarks.build_merged_record` (HEAD :1186-1191, WC :1214-1219) as the first non-empty of:
1. PGP `documents[].description`
2. KTIV `basic_catalog.title` (the NLI Hebrew heading)
3. Bodleian TEI, folio-verified only (`bodleian_images.tei_description` :267: msItem titles joined with "; ", else the first note)
4. FJP[0] `description` (the FGP "Running Title. Domain")

Then:
- `GenizahDocument.clean_description` (HEAD :317-326) removes every "Description" substring. This alters 82 descriptions.
- `create_text_representation` (HEAD :1375-1529, WC :1508) writes the text.
- The training corpus is `semantic_text(v8)`: the reference `semantic_text.py:136-154` in genizah_search, and HDA `src/embeddings/semantic_text.py` (its regexes are fingerprint-pinned).

**Line order** (HEAD):
1. `Document ID:`
2. `Shelf Mark:` (only when it differs from the doc id)
3. `Description:` (:1409)
4. `Alternate titles:` (:1414)
5. `Subjects:` (WC only, :1553)
6. `Title:` / `Summary:` (never for merged records: no full_metadata key contains "title" or "summary")
7. `Language:` (:1432, only from PGP `languages_primary` or FJP `language`, `from_merged_format` :1635)
8. `Document Type:` (:1437, an English keyword guess on the description, `infer_document_type` :901)
9. `Date:` (`key: value` pairs)
10. `Transcription:` (≤1,000 characters + "...")
11. `Translation:` (≤500 characters + "...")
12. `Related People:`, `Related Places:`
13. `Physical:` / `Additional Info:` (never for merged records)

**What each line is made of:**
- **`Description:`**, among all records: PGP 34,375; FJP 25,714; KTIV title 15,175; TEI 178; none 2,671. Among HEAD-eligible records: PGP 31,346; FJP 19,780; KTIV title 12,372.
- **`Alternate titles:`**, at HEAD, comes only from KTIV `basic_catalog.varying_form_of_title`, with "Title in English:" stripped (`_ktiv_alt_titles` :76). It is on 12,691 of 63,652 eligible texts (19.9%), all KTIV-sourced; 9,982 of those have a Hebrew KTIV-title description.

**KTIV material that is not in the HEAD text:**
- **Scholarly entries:**
  - frames, domains and `general_title` serve only as training labels (`build_subjects.ktiv_fields` :612-627);
  - `general_title` is still unused in WC;
  - frames are added in WC.
- **KTIV language:** 13,900 records show "Unknown" although KTIV gives one.
- **`full_catalog.notes`:** 18,233 records. These are mostly provenance boilerplate ("הנתונים מבוססים על מידע מ…": 2,194 + 1,033 + 987 …), and only 211 name a festival.

**Asymmetry by source** (HEAD, eligible records):

| | Language line | Document Type line | Alternate titles line |
|---|---|---|---|
| KTIV-only (n=10,146) | 0% | 0% | 80% |
| FJP-only (n=18,763) | 86% | 73% | 0% |
| With PGP (n=32,594) | 91% | 63% | 9% |

So the layout alone identifies a KTIV record.

**Masking and eligibility.**
- `semantic_text` drops the ID lines and `[Editor: …]` / PGP editor credits, then drops empty field lines.
- It masks the record's own shelf mark (taken from the `Shelf Mark:` line) plus `PRE_ID_RE`, `PROD_SHELFMARK_RE` and `EXTRA_ID_RE` matches as `[shelfmark]`.
- `ineligible_reason` (ref :248-260) returns, in order:
  - `empty`;
  - `short`, when `len(whole text) < 40`; labels and metadata lines count toward the 40;
  - `boilerplate_only`, when no "content line" exists. `Language`, `Document Type` and `Date` lines never count. A `Description` counts only if its whole value is not one of the 64 frozen labels. An `Alternate titles` line counts only if some `;` part is not a label.
- The reference recomputes the labels (description values and alternate-title parts shared by ≥50 records, ref :195-212, :282). HDA loads the frozen list (`data/boilerplate_descriptions_v3.json`, sha `d57204f7…`).
- Re-deriving the labels on today's file would give 75 labels (13 added, 2 dropped). The additions include specific works ("אברהם אבן עזרא, פירוש אבן עזרא למקרא", "הפטרות"), so keep the frozen list.

**Other defects in the chain:**
- **Dropped transcriptions:** 65 records lose 74 FJP string transcription items (`validate_transcriptions` :350 logs a warning and drops them).
- **Multi-line descriptions:** 1,756 descriptions contain newlines, so 1,828 records have unlabelled continuation lines. `description_value` and `text_field` read only the first line, so the festival scan of descriptions misses the rest (inferred from `build_subjects.py:169-179`).
- **Subjects in the corpus build:** `build_corpus.py:32-34,150` imports HDA's working copy, so a corpus rebuild today would silently take in the WC `Subjects:` and frame lines.

**Recommendation (1).** Freeze what the next text is made from:
- The HDA document-model changes for v4 must be committed before the corpus rebuild.
- `build_corpus.py` should record the HDA commit and refuse a dirty tree without `--allow-dirty`.
- **Fix in `from_merged_format`:**
  - convert FJP string transcriptions to `TranscriptionSection(name="FJP", lines={...})`;
  - fill the KTIV language (code in §4).
- **Optional (owner's call):** add KTIV scholarly `general_title` to `alt_titles`. It adds new text to 6,902 records, and 71 of them name a festival only there.
- **Do not add** KTIV notes (provenance boilerplate and legacy-store names) or scholarly descriptions.

## 2. Fix 4: the field-label shortcut

**Options considered:**

| Option | Known-item lookup | Information kept | KTIV and non-KTIV look alike | Effect on parsers |
|---|---|---|---|---|
| **A. Fold alternate titles into the `Description:` line** (recommended) | kept: the alternate-title text stays in the vector, and the ES `alt_titles` BM25 field is unchanged | all kept; only parts already present (normalised, whole words) are dropped: 864 at HEAD | yes: 0 texts keep the `Alternate titles:` label | small: code still parses `Description:` |
| B. Unlabelled title block | same | same | yes | large: `content_lines`, `text_field`, `label_part` would all change |
| C. Keep labels, add label dropout only in training | same | same | no: the layout still marks KTIV records | none, but the asymmetry stays in the index |
| D. Rename to a second `Description:` line | same | same | partly | parsers read only the first such line, so festival scans miss it |

**Two lines to remove along with the label:**
- **`Document Type:`.** It is derived from the description, so dropping it loses nothing. It is present on 0% of KTIV-only records and 73% of FJP-only records. It is on 34,280 eligible texts, with 70 mid-word misfires such as "verse" inside "reverse".
- **`Subjects:` (WC).** Its values are the training labels:
  - domain-subject names would appear literally in 99% of (record, domain) pairs, up from 38%;
  - PGP-tag names in 90%, up from 62%.
  - `build_training_mix_v3.subject()` (:1044-1054) skips pairs whose names appear literally in the record, so most domain and PGP training pairs would disappear (inferred).
  - It is also a new label carried only by KTIV/PGP records: 0 of 18,763 FJP-only records have it.
  - Keep `subjects` as an ES field for BM25 and facets.

**Proposed layout:** `semantic_v4 = layout(semantic_v3(v8))`. The layout works on lines:
1. fold continuation lines into the line before them;
2. drop the `Subjects:` and `Document Type:` lines;
3. fold the alternate-title parts into `Description:`, de-duplicating by normalised key and capping the appended alternate titles at 1,000 characters (whole parts; affects 112 HEAD / 244 WC records).

Eligibility is decided on the v3 text (§4), so the layout never removes a record.

Results on HEAD texts:
- Eligibility moves from 63,652 to 63,779 records, with 0 removals.
- On WC-snapshot texts it moves to 64,136, again with 0 removals versus v9.

**Code to add, identical in both copies** (HDA `src/embeddings/semantic_text.py`, after `is_semantic_eligible`; genizah_search `evals/embedding_audit/semantic_text.py`, after `ineligible_reason` :260):

```python
# --- semantic_v4: one line per field, catalogue titles under one label (layout over the frozen v3 text) ---
DESC_PREFIX = "Description:"
SUBJECTS_PREFIX = "Subjects:"   # HDA WC already defines it
FIELD_PREFIXES = ("Document ID:", "Shelf Mark:", "Description:", "Alternate titles:", "Subjects:", "Title:",
                  "Summary:", "Language:", "Document Type:", "Date:", "Transcription:", "Translation:",
                  "Related People:", "Related Places:", "Physical:", "Additional Info:")
V4_DROP_PREFIXES = ("Subjects:", "Document Type:")
TITLE_SEP = "; "
MAX_ALT_CHARS = 1000
TITLE_KEY_RE = re.compile(r"[\W_]+")
ADMIT_PATH = Path(__file__).resolve().parent / "data" / "eligibility_admit_v4.json"  # ref: docs/handoffs/v10_embedder/

def text_sha256(text: str) -> str:
    """sha256 hex of the exact UTF-8 bytes handed to the embedder.

    :param text: Embedded text.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fold_field_lines(text: str) -> List[str]:
    """Lines of a text, each line that starts no field label appended (one space) to the line before it.

    :param text: semantic_v3 text.
    :returns: One line per field.
    :rtype: List[str]
    """
    out: List[str] = []
    for ln in text.split("\n"):
        if out and not ln.startswith(FIELD_PREFIXES):
            out[-1] = f"{out[-1]} {ln.strip()}"
        else:
            out.append(ln)
    return out

def title_key(text: str) -> str:
    """Comparison key: case-folded, punctuation and brackets as single spaces.

    :param text: A title or description.
    :returns: Key.
    :rtype: str
    """
    return " ".join(TITLE_KEY_RE.sub(" ", text.casefold()).split())

def merge_title_lines(lines: List[str]) -> List[str]:
    """Fold the ``Alternate titles:`` parts into the ``Description:`` line.

    A part whose key already occurs as whole words is left out; appended parts stop at MAX_ALT_CHARS.

    :param lines: Output of fold_field_lines.
    :returns: Lines with no Alternate titles line.
    :rtype: List[str]
    """
    alt_i = next((i for i, ln in enumerate(lines) if ln.startswith(ALT_PREFIX)), None)
    if alt_i is None:
        return lines
    desc_i = next((i for i, ln in enumerate(lines) if ln.startswith(DESC_PREFIX)), None)
    desc = lines[desc_i][len(DESC_PREFIX):].strip() if desc_i is not None else ""
    seen, kept, added = f" {title_key(desc)} ", [], 0
    for part in (p.strip() for p in lines[alt_i][len(ALT_PREFIX):].split(";")):
        key = title_key(part)
        if not key or f" {key} " in seen:
            continue
        if kept and added + len(part) > MAX_ALT_CHARS:
            break
        kept.append(part)
        added += len(part) + len(TITLE_SEP)
        seen += f"{key} "
    value = TITLE_SEP.join(([desc] if desc else []) + kept)
    out = list(lines)
    if desc_i is not None:
        out[desc_i] = f"{DESC_PREFIX} {value}"
        del out[alt_i]
    elif value:
        out[alt_i] = f"{DESC_PREFIX} {value}"
    else:
        del out[alt_i]
    return out

def semantic_v4_text(v3_text: str) -> str:
    """semantic_v4 text of a record from its semantic_v3 text.

    :param v3_text: Output of semantic_text().
    :returns: Embedded text (may be empty).
    :rtype: str
    """
    lines = [ln for ln in fold_field_lines(v3_text) if not ln.startswith(V4_DROP_PREFIXES)]
    return "\n".join(merge_title_lines(lines)).strip()
```

**HDA `src/embeddings/text_representation.py` edits:**
- Add `TEXT_VERSION_SEMANTIC_V4 = "semantic_v4"` and include it in `TEXT_VERSIONS`.
- Keep `DEFAULT_TEXT_VERSION` at `semantic_v3`. Flip it in the same commit that pins the v4-trained model in `FINETUNED_MODEL_*`.
- In `text_for_version` (:82), fragments become `semantic_v4_text(semantic_text(v8))`.
- `embedding_text` (:135) handles v4 itself:
  ```python
  v3 = semantic_text(v8)
  return EmbeddingText(semantic_v4_text(v3), semantic_v4_ineligible_reason(v3) == "eligible")
  ```
- Add a new `embed_text_for_version(v8_text, version, kind) -> EmbeddingText` that `embedding_text` delegates to.
- `is_text_eligible(text, "semantic_v4")` raises `ValueError`, because merged lines cannot be checked part-wise.
- Callers to migrate: `describe_from_reads.py:1177-1185`, `update_generated_descriptions.py:~71`.
- The existing `GUARDED_META_KEYS` (`elastic_index_genizah.py:50`) then refuses to write v4 documents into v9.

**genizah_search reference `main()` (:273-332):**
- Add `--text-version {semantic_v3,semantic_v4}` and `--boilerplate-file`. For v4, default to the frozen `docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json`; never recompute.
- Write `corpus_semantic.jsonl` (v4) with these fields:
  - `text` (v4)
  - `text_v3`
  - `eligible`
  - `reason`
  - `text_sha256`
  - `text_hash` (= `text_sha256`, which keeps `row_segments` and the eval's twin detection working)
  - plus the existing fields.
- `build_subjects.py:1169-1171` and `build_eval_v3.py:256` must read `text_v3`: they need the `Document Type` and `Alternate titles` lines.

**Pinning the parity between copies:**
1. Fingerprint `TITLE_KEY_RE` like the existing regexes, and assert `FIELD_PREFIXES`, `V4_DROP_PREFIXES`, `TITLE_SEP`, `MAX_ALT_CHARS` and `MIN_CHARS`.
2. Assert the sha256 of `eligibility_admit_v4.json` and of the boilerplate file in both repos.
3. **Golden fixture:** about 200 stratified real records in both repos (`tests/data/semantic_v4_golden.jsonl` and `evals/embedding_audit/test_semantic_v4.py`), each as `{v8_text, v4_sha256, reason}`. Cover:
   - KTIV-only with alternate titles;
   - bracketed frames;
   - multi-line PGP descriptions;
   - admitted short records;
   - boilerplate-only;
   - empty.

   This is the check that would have caught the 28k-record drift; regex fingerprints alone did not. Add the same kind of fixture for v3.
4. Update the HDA test expectation (`tests/test_text_representation.py`). For `merged_record()`, `semantic_v4` must be exactly:
   `"Description: Letter from Nahray b. Nissim to his partner about the flax trade. Joins [shelfmark] and [shelfmark] ([shelfmark]).; Business letter\nLanguage: Judaeo-Arabic\nDate: standard_date: 1060\nTranscription: בשם רחמנא כתאבי אליך\nRelated People: Nahray b. Nissim\nRelated Places: Fustat"`
   The `.; ` join is cosmetic; `TITLE_SEP` is the separator alternate titles already used.

## 3. A per-document text hash

**Name.** Use HDA's existing name `embedded_text_sha256` rather than inventing `semantic_text_hash`. It is already defined in WC `update_catalogue_enrichment.py:114`, computed as `sha256(text.encode("utf-8"))` at :177-184. Move that function into `text_representation.py` and import it everywhere.

**Indexer** (`elastic_index_genizah.py`):
- Add `"embedded_text_sha256": {"type": "keyword"}` to `TEXT_VERSION_MAPPING` (:44).
- Have `_get_versioned_embeddings` (:567) return its items.
- In `process_documents`, beside :288, set `es_doc["embedded_text_sha256"] = items[i].text_sha256`. Do this for every document, ineligible ones included, so the updater can detect changed text on vector-less documents.
- Optional: store the text itself as `semantic_text: {"type": "text", "index": false}`. That is about 26 million characters for 64k documents.
- Embed once per unique hash (a cache keyed on the hash); §5 gives the duplicate counts.

**`_meta`** (`qwen_text_embedding.build_index_meta` :265):
- `text_hash: "sha256(utf-8 of the embedded text)"`
- `document_canary.text_sha256`
- `text_function: {version, golden_fixture_sha256}`
- `source: {merged_records, merged_sha256}`
- At the end of the build, set `counts: {documents, eligible, distinct_text_sha256}` through `put_mapping` (the writer guard ignores those keys).
- Add `text_hash` to `REQUIRED_META_KEYS` in `verify_reembedded_indexes.py` (:66).

**Verifier:** in `check_self_consistency_genizah` (:315), compare the hash of the recomputed text with the stored `embedded_text_sha256` before re-embedding. That reports text drift (merged file or code) separately from model drift.

**Cross-repo check before a build:** for every `doc_id`, the training corpus `text_sha256` must equal the index `embedded_text_sha256`. Run it as one read-only scroll by the build session.

**What the site needs** (prod-mbp `src/backend/search_service.py`, `search` :845-905 and `search_hybrid` :1703+):
- Nothing is required for startup.
- It can collapse duplicate catalogue texts in its vector results, either with `collapse: {"field": "embedded_text_sha256"}` (should work with `script_score` because it is a normal query; verify on the cluster) or by de-duplicating in Python after fetching about 3× the page size. Show "N fragments share this catalogue text".
- It should already filter `semantic_eligible: true` (v9 handoff :71).

## 4. Eligibility

**Why Cambridge_CUL_Or_1081_2_33 is ineligible.**
- Its HEAD text is `"Description: הגדה של פסח (קטע)"`, 30 characters, so it is `short`.
- The 40-character rule counts labels and metadata lines. KTIV gives the language "Hebrew" but `from_merged_format` ignores it (:1635); a `Language: Hebrew` line would make the text 47 characters and eligible.
- Its scholarly title ("הגדה של פסח"), `general_title` ("Passover Haggada; piyyut.") and notes are not in the text. In WC it becomes eligible through the scholarly title added as an alternate title.

**The rule is inconsistent across sources.**
- 4,282 eligible texts have under 27 characters of content and pass only because of their `Language` / `Date` lines.
- Of 1,504 short records with real content and no `Language:` line, 787 are KTIV-only.

**Counts among HEAD-ineligible records (14,461):**
- **Festival named** in description or alternate titles: 62, all `short`. By festival: sukkot 22, simhat_torah 15, yom_kippur 9, pesach / rosh_hashanah / hanukkah / shavuot / fast_days 4 each (they overlap). 4 are in a canon context.
- **topic_v2 topic named:** 188 in total, of which 127 are short with non-boilerplate content.
- **Subject vocabulary named** (≥4 characters): domain 5,028; pgp 2,263; work 383; sefaria 110; topic 107. These are mostly genre names.
- **Festival named only in KTIV extras:** 49.
- In WC: 16 festival-naming records and 97 topic-naming records remain ineligible.

**Proposal: the v4 rule.** Keep the v3 rule and the frozen 64 labels, with three changes:
1. `MIN_CHARS` is measured without `Subjects:` lines.
2. `Subjects:` never counts as content; alternate-title parts are compared to the labels without square brackets.
3. **Admission:** a `short` text whose content matches the frozen festival-term or topic patterns is eligible. The patterns are `build_subjects` `FESTIVAL_TERMS` (case-insensitive, :129-146) and `TOPIC_RULES_V2` (case-sensitive, :63-98), expanded into `data/eligibility_admit_v4.json` (9 + 19 patterns; draft sha `5d35ef16…` in scratch).

```python
@lru_cache(maxsize=None)
def admit_pattern(path: Path = ADMIT_PATH) -> Pattern[str]:
    """Frozen festival / topic patterns that admit a short record.

    :param path: eligibility_admit_v4.json.
    :returns: Compiled pattern.
    :rtype: Pattern[str]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return re.compile("|".join([f"(?i:{p})" for p in data["festival_terms"]] + [f"(?:{p})" for p in data["topic_rules"]]))

def content_lines_v4(v3_text: str, boiler: AbstractSet[str]) -> List[str]:
    """v3 content lines; Subjects never counts, alternate-title parts compared without brackets.

    :param v3_text: semantic_v3 text.
    :param boiler: Frozen labels.
    :returns: Content lines.
    :rtype: List[str]
    """
    out = []
    for ln in v3_text.split("\n"):
        if not ln.strip() or ln.startswith(META_PREFIXES + (SUBJECTS_PREFIX,)):
            continue
        if ln.startswith(DESC_PREFIX) and ln[len(DESC_PREFIX):].strip() in boiler:
            continue
        if ln.startswith(ALT_PREFIX) and {p.replace("[", "").replace("]", "").strip() for p in alt_title_parts(ln)} <= boiler:
            continue
        out.append(ln)
    return out

def semantic_v4_ineligible_reason(v3_text: str, labels: Optional[Sequence[str]] = None) -> str:
    """empty / short / boilerplate_only / eligible, decided on the semantic_v3 text.

    :param v3_text: semantic_v3 text.
    :param labels: Boilerplate labels (None: the frozen v3 list).
    :returns: Reason.
    :rtype: str
    """
    boiler = <frozen v3 labels> if labels is None else frozenset(labels)
    if not v3_text:
        return "empty"
    measured = "\n".join(ln for ln in v3_text.split("\n") if not ln.startswith(SUBJECTS_PREFIX))
    content = content_lines_v4(v3_text, boiler)
    if len(measured) < MIN_CHARS:
        value = " ; ".join(ln.split(":", 1)[1].strip() if ln.startswith(FIELD_PREFIXES) else ln for ln in content)
        return "eligible" if value and admit_pattern().search(value) else "short"
    return "eligible" if content else "boilerplate_only"
```

**Effect:**

| Applied to | Change | Eligible after |
|---|---|---|
| HEAD texts | +127 (62 festival, 65 other topics), 0 removed | 63,779 |
| WC-snapshot texts | +484 versus v9, 0 removed | 64,136 |
| HEAD texts + KTIV language fill | +707 through the 40-character rule, plus 30 admitted | 64,389 |

The vocabulary-only admissions were rejected: those 46 are genre labels ("Midrash", "הלכה", "Zohar literature"). Content-length rules were also rejected: at 12+ characters of content they add 1,109 and remove 521; at 20+ they remove 2,273.

**KTIV language fill.** Change HDA `genizah_document.from_merged_format`:

```python
language = (pgp0.get("languages_primary") or fjp0.get("language")
            or ", ".join(p.replace("Judeo-", "Judaeo-")
                         for p in str((ktiv.get("basic_catalog") or {}).get("language") or "").split())
            or "Unknown")
```

This affects 13,900 records. It also changes the ES `language`, `main_language` and `script_type` fields, which the site uses for filters, so it has to land in this same round.

## 5. Duplicate texts

Among the 63,652 HEAD-eligible records there are 55,833 unique texts. 2,869 groups have more than one record, holding 10,688 records, so 7,819 vectors (12.3%) are redundant.

| Group size | 2 | 3–5 | 6–20 | 21–100 | >100 |
|---|---|---|---|---|---|
| Groups | 1,785 | 703 | 335 | 45 | 1 (125 × T-S C6 Ibn Ezra) |

**What the groups are:**
- **Multi-part source records:** 679 groups share one PGP document (multi-fragment documents) and 123 share one FJP range record ("T-S 12.801 - T-S 12.812"), 2,051 records in all.
- **Separate catalogue entries with identical generic text:** 2,067 groups, 8,637 records. Examples: Manchester A "Book of Genesis. Bible: Texts…" ×47; "פיוט;שירת חול / Piyyut; פיוט" ×60.
- 1,768 groups share an id prefix; 1,907 contain only the description plus metadata; only 7 contain a transcription.
- v4 barely changes this: 2,890 groups, 7,869 redundant.

**Recommendation (5):**
- Keep one vector per fragment; do not make duplicates ineligible, and do not put identifiers back into the text.
- Embed once per `embedded_text_sha256` (HDA `elastic_index_genizah._get_versioned_embeddings`).
- Collapse by hash on the site and in RAG context assembly (§3).
- Training and eval de-duplicate by `text_hash` (= sha256 in the v4 corpus; `row_segments` :221 and `eval_v3` twins :470-476 already do this).

## 6. Token window

Training truncates at 384 tokens (`TRAIN_MAX_SEQ`, `train_embedder_v3.py:81`). Production embeds up to 8,192.

| Text | p50 | p90 | p95 | p99 | max | >384 | >512 | >768 | >1024 | >2048 |
|---|---|---|---|---|---|---|---|---|---|---|
| HEAD v3, eligible (n=63,652) | 53 | 374 | 538 | 853 | 1,938 | 6,159 (9.7%) | 3,516 | 1,141 | 187 | 0 |
| v4 with cap, HEAD | 49 | 370 | 533 | 848 | 1,933 | 6,026 | 3,448 | 1,094 | 160 | 0 |
| v4 with cap, WC | | | | | 1,933 | 6,262 | 3,540 | 1,104 | 160 | 0 |
| WC uncapped | | | | | 4,821 | | | | | 3 |

- The WC uncapped maximum comes from an 11k-character frame list on `Cambridge_CUL_Or_1081_2_76`.
- Measured characters per token: 3.0 overall and 2.45 for Hebrew-heavy texts, so a characters/3.2 proxy undercounts Hebrew by about 25%.
- Of the texts over 384 tokens, 5,426 of 6,159 contain a `Transcription:` line. The median position where the transcription starts is character 377, so training never saw the transcription tails that production embeds.
- **Tokenizer trap:** the fine-tune's saved `tokenizer.json` has `truncation.max_length = 384` and BatchLongest padding baked in.
  - sentence-transformers and transformers reset this on every call (`transformers/tokenization_utils_fast.py` ~470-490), so the indexer is fine.
  - Any raw `tokenizers` or TEI consumer would silently truncate at 384.

**Recommendation (6):**
- Set `TRAIN_MAX_SEQ = 1024` (:81), which covers 99.75% of v4 texts. Set `mine_negatives max_seq = 1024` (:538). The cost should stay modest because the median is about 50 tokens and padding is per mini-batch (inferred).
- Keep 8,192 for eval and production.
- Before saving the model (`train` :698-700), call `model.tokenizer.backend_tokenizer.no_truncation()` and `.no_padding()`, and record the training window in `training_info.json` and the model card.
- Keep the v8 caps (transcription 1,000 characters, translation 500) and the new `MAX_ALT_CHARS = 1000`.

## Change list by repo

**HDA** (coordinate with the person editing it):
- `src/embeddings/semantic_text.py`: the v4 block (§2, §4) and `data/eligibility_admit_v4.json`.
- `src/embeddings/text_representation.py`: `semantic_v4`, `embed_text_for_version`, `text_sha256`, `EmbeddingText.text_sha256`.
- `src/datasets/document_models/genizah_document.py`:
  - KTIV language fill;
  - convert FJP string transcriptions;
  - keep the frames in `alt_titles`;
  - optional: `general_title`.
- `src/datasets/indexing/elastic_index_genizah.py`: `embedded_text_sha256` mapping and write, plus the embed-once cache.
- `src/embeddings/qwen_text_embedding.py` (`build_index_meta`): the hash and text-function keys.
- `src/datasets/indexing/verify_reembedded_indexes.py`: hash comparison, `REQUIRED_META_KEYS`.
- `src/datasets/indexing/index_merged_genizah.py`: `DEFAULT_INDEX_NAME` → v10, final `_meta` counts.
- Callers to migrate: `describe_from_reads.py`, `update_generated_descriptions.py`, `update_catalogue_enrichment.py`.
- Tests: `test_semantic_text.py`, `test_text_representation.py`, golden fixtures.
- Stop labelling the enriched text `semantic_v3`.

**genizah_search:**
- `evals/embedding_audit/semantic_text.py`: the same v4 block, frozen-label loader, v4 `main()`.
- `build_corpus.py`: HDA commit pin.
- `build_subjects.py` and `build_eval_v3.py`: read `text_v3`.
- `colab/train_embedder_v3.py`: `TRAIN_MAX_SEQ`, `mine_negatives` window, tokenizer save; `DATA_FILES` (:85) must point at the v4 corpus.
- New `test_semantic_v4.py` with the shared fixture.
- Data files: `docs/handoffs/v10_embedder/eligibility_admit_v4.json`.
- Site (prod-mbp): optional `embedded_text_sha256` collapse in `search_service.search` / `search_hybrid`.