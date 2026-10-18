**Festival-pattern audit across all subjects (old corpus, three SCR mixes)**

Fixes 1-3 close the festival case only in `topic:*`. The same bug pattern is in three other places, and fix 1 has introduced an inverse bug. All numbers below are from `new/mix.jsonl` (fixes 1-3) unless marked otherwise. They come from the old `corpus_semantic.jsonl` (73,543 records, 60,493 eligible), which is what the SCR mixes were built from. "Genre" is a heuristic classifier I wrote over the Alternate titles head, the Description head, Document Type and domain ids (`audit.py`), so read the genre shares as approximate.

**1. The literal-name drop is not the main driver; the labels are.**
- For a topic, the drop skips a record only when every name is literal. That fired 4 times for topics (`all_names_literal:topic` = 4), and 0 of the strong carriers of every flagged topic have all their names literal.
- The genre mix of the pairs tracks the genre mix of the carriers (legacy purim: carriers 44% tractate / 33% scripture, pairs 47% / 37%).
- So the cause is that a subject's strong labels come from one genre or source (KTIV frames, or Sefaria refs). Records that do name the subject but are unlabelled are then valid negatives.
- They are also the likeliest mined negatives. v9 mined a negative for all 11,758 subject rows from base-model ranks 4-13 (`train_embedder_v3.py:538`, `training_info.json`). That the negatives are literal-name records is inferred, since no model was run.

**2. Ranked at-risk subjects (new mix)**

| # | Subject (group) | Pairs | Where the pairs go | Records naming it | Named but not a carrier | Verdict |
|---|---|---|---|---|---|---|
| 1 | `domain:*` (110 genre subjects) | 4,182 (33% of subject pairs) | 100% KTIV positives | e.g. Bible/מקרא: 11,600 records, 17% KTIV | Bible 10,229 (9,212 scripture); piyyut 2,084; liturgy 1,880; letters 1,639 | Flag: single-source labels. This is the label-level cause of fix 4 (the field-label shortcut). 95% of domain rows have ≥10 literal-name negative candidates |
| 2 | `sefaria:*` with bare (non-law) names | 535 | 98% canon (523) | Amidah: 156 (148 liturgy); synagogues: 542 (446 documentary); condolences: 48 (44 letters) | almost all | Flag. Canon by construction: 1,415 of 1,428 sefaria-labelled records are canon. Festival residue is still here: `sefaria:purim` "Purim"/"פורים" → Esther / Ta'anit (9 pairs), `sefaria:passover` "חג פסח" (5, all scripture), `sefaria:tisha-bav` (8 of 9 tractate), `sefaria:fasting` (7 of 7), `sefaria:yom-kippur` (7 of 7 canon) |
| 3 | `topic:marriage` | 65 | 68% canon (Ketubbot, Qiddushin, Ishut) | 2,066, 92% documentary | 1,997 (888 carry `pgp:marriage` etc., which is still a different id; 868 have no subject) | Flag. "wedding": 4 of 5 pairs tractate, while 118 of 128 records naming it are documentary |
| 4 | `topic:divorce` | 35 | 74% canon (Gittin, Mishneh Torah) | 662, 91% documentary | 638 (284 with no subject) | Flag. "bill of divorce": 8 of 9 pairs go to Talmud Gittin; 326 records say it, 316 documentary, 320 unlabelled |
| 5 | `topic:rosh_hodesh` | 23 | 65% Mishneh Torah Qiddush ha-Hodesh (calendar law, a different concept) | 39 (16 liturgy, 16 documentary) | 33 | Flag: it is an occasion, not a law |
| 6 | Festival law names (inverse bug from fix 1) | ~22 | "laws of the New Year" 10 of 11 → liturgy (legacy: 5 tractate / 5 liturgy / 2 law code); in `mix_label`, "הלכות פסח" 6 of 7 and "laws of Pesach" 3 of 4 → Haggadot | – | – | Flag: law names lost their law carriers |
| 7 | `topic:shabbat` | 93 | 82% canon | 912, 53% canon | 567 (212 tractate, 178 documentary) | Partial. The tractate mapping is legitimate; the main risk is unlabelled-named negatives (50 of 93 rows have ≥10). Do not make it weak |
| 8 | `topic:mourning` | 13 | 62% Mo'ed Qatan (the whole tractate) | 201, 82% documentary | 197 | Low volume (`pgp:mourning` dominates the string) |
| 9 | `topic:tefillin_mezuzah` | 26 | 81% law code | 133: 51 law code, 39 actual tefillin/mezuzah parchments | 85 (69 with no subject) | Moderate. Law mapping is fine; the parchments are never carriers |
| 10 | Ambiguous Hebrew anchors | ~120 | ספר 27, מס 20, כשר 17 (`mix_label`), אבל 15 ("but"), נשים 12, מזון 11, לוח 10, עד 8 ("until") | – | – | Noise: e.g. 43 of 64 נשים hits are Seder Nashim, 48 of 55 מזון hits are Birkat ha-Mazon |
| 11 | `pgp:medicine`, `pgp:fasting`, `pgp:bible` | 145 / 26 / 30 | documentary | medicine: 612 named are medical texts | – | Low risk |

**Confirmed OK:**
- **`pgp:*`:** 4,902 pairs: 84% documentary, 1.6% canon, 8.4% KTIV.
  - For 95 of the 108 PGP subjects with ≥20 naming records, the pairs' genre is the dominant genre of the records that name the subject (median share 0.895).
  - 60% of PGP carriers name the subject, and 2,960 of 4,902 positives state it in another form. So the name is paired with the population that actually names it, which is why PGP worked in v9.
  - Root cause 3 still applies: 81 of 154 `pgp:marriage` rows have ≥10 literal-name negative candidates.
- **`work:*`:** the top genre equals the work family in every case. This is correct.
- **`topic:niddah`:** 61% of pairs are tractate and 68% of the records naming it are tractate. Legitimate.
- **`topic:kashrut`:** law-name subject; the mapping is fine apart from the "כשר" anchor.
- **`topic:fast_days`:** fixed by fix 1 (legacy 72% tractate → new 67% liturgy); the residue is in the Sefaria subjects.

**Why the current eval cannot see this:** seen-subject AP scores agreement with these same labels. v9's `topic:pesach` name AP was 0.58 while users got Pesahim.

**3. Rule changes, all of which change the index**
- **R1 – `build_subjects.py`:**
  - In `frame_subjects` (L889-895), route every topic matched on a `CANON_WORKS` frame into `canon`, not only festivals.
  - In `main` (L1162-1182), keep festival canon labels weak as now. Keep non-festival canon labels strong.
  - Write a new per-record field `"canon": [ids]` covering those topics plus every `sefaria:*` id.
  - Add `rosh_hodesh` to `FESTIVAL_TOPICS` (L126) with `FESTIVAL_TERMS["rosh_hodesh"] = r"rosh [hḥc]h?odesh|new moon|{HB}ראש חודש{HE}"`.
  - Generalise `FESTIVAL_NAME_DROP` (L156) to drop names matching `^(mishnah|talmud|tosefta|megillat|scroll of|tractate|משנה|תלמוד|מגילת|מסכת)\b` from topic and sefaria vocabularies (this removes e.g. "Mishnah Shabbat", "Scroll of Esther").
- **R2 – `build_training_mix_v3.subject()` (L999-1063):** a name-form gate.
  - Define:
    ```python
    LAW_NAME_RE = re.compile(r"^(?:the )?(?:laws?|rules|prohibitions?) (?:of|concerning|for)\b|\blaws?$|^hil[ck]h?[oa][ts]\b|^hilchos\b|^dinei\b|^(?:הלכות|הל'|דיני|דין|איסורי)\s", re.I)
    ```
  - Key carriers by `(s, cls)`. For `topic:`/`sefaria:` ids: a canon carrier (including weak canon festival carriers) is `"law"` and takes only `LAW_NAME_RE` names; a non-canon carrier is `"bare"` and takes only the other names. Other kinds stay `"all"`.
  - Give each key its own `ceil(k*sqrt(n))` quota.
  - Read `canon` in `load_subjects` (L493).
  - Effect: about 707 bare→canon pairs removed (5.6%) and the festival law names get their law carriers back. Sefaria names are still taught through t2t (8,576 rows in v9).
- **R3 – mention guard (generalises root cause 3 and the label-level side of fix 4):**
  - In `build_subjects`, run a second pass that matches the kept names (Hebrew prefixes tolerated, stoplist from R4 applied) plus `FESTIVAL_RE` against `Description:` and `Alternate titles:` lines. Write `"mentions"` for subjects the record does not already carry.
    - That is 111,505 ids, mean 1.84 per record, 99.7% of records ≤10.
  - In `load_subjects`, add them as `M:<id>`.
  - `emit` puts `anchor_mention_id = "M:<anchor_subject>"` on subject rows and adds the `M:` versions to `anchor_topic_ids`.
  - In the trainer: `BROAD_PREFIXES = ("domain:", "M:")` (L77), and `keep=[row.get("anchor_subject"), row.get("anchor_mention_id")]` (L222). A mention then masks a record only for rows that name that subject, never through the positive.
  - In `build_eval_v3`, put mentions into `subject_relevance[id]["unjudged"]` for every subject (generalising `concept_mentions`, L624). Without this, seen AP will drop when the fix works.
- **R4 – stoplist:** add `{"אבל","עד","כשר","מס","ספר","נשים","מזון","לוח","עבד","ברכה","שמע","Get","Praise","Fast"}` as an anchor and mention stoplist in `subject_name_table` (L584).
- **R5 – eval:** extend `holiday_queries` (L237) into a non-circular name-query set, scored over the catalogue-line mentions plus strong carriers. Queries: bill of divorce, ketubba, Shabbat, Bible, letters, piyyut, Amidah, condolences, synagogue, Rosh Hodesh.
- **Decision for you:** whether FJP/PGP `Document Type` (bible, letter, liturgical) should become *strong* domain labels instead of only R3's mask. Strong would directly attack fix 4 but enlarges the domain quotas, so gate it on the R5 eval.

Expect the numbers to move with today's merged file (+4,570 records; inferred). The structural findings will not: Sefaria labels are canon by construction, and domain and work labels are KTIV-only by construction.

Scripts and outputs are in `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/6e892054-8bdc-4cf3-9c1c-de16a1b5474a/scratchpad/audit_subjects/`:
- `audit.py`
- `audit_result.json`
- `doc_info.json`
- `sources.json`
- `show.py`
- `bare_canon.py`
- `alias.py`
- `exposure.py`
- `mentions_dist.py`