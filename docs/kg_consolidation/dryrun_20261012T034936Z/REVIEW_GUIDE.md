# Scholar consolidation dry run, 2026-10-12 03:49:36Z: review guide

This is a dry run of `scripts/kg/consolidate_scholars.py` against production Neo4j
(`bolt://localhost:7681`). It used one read session and one read transaction, and wrote nothing
to the graph. The counts were 77,215 nodes and 186,231 relationships before the run and the same
after it. The name rules are in `scripts/kg/person_names.py`. The edge merge rules are in
`../census_2026-10-12/edge_merge_policy.md`.

## Headline

| | |
|---|---|
| Scholars read | 3,375 |
| MERGE groups (automatic) | 149 groups, 304 members, **155 nodes removed** |
| RENAMES (automatic, singletons) | 358 |
| Loser edges moved to the survivor | 612 |
| Loser edges merged into an existing survivor edge | 13 (12 collisions) |
| Self-loops dropped | 0 |
| Review rows (nothing changes) | 833 rows, 853 distinct Scholars |
| **Expected after `--apply`** | **77,060 nodes / 186,218 relationships** (Scholars 3,375 to 3,220) |

## Files

| File | What it holds |
|---|---|
| `summary.json` | Counts by category, plus `plan_sha256`. |
| `plan.json` | The full plan: every operation with before-state, edge moves and merges, and survivor updates. `--apply --plan <this dir>` plans again from the live graph and refuses to run if the new plan differs from this file. |
| `merges.tsv` | One row per MERGE: `key` (identity key), `canonical` (final name), `category` (`initials_only` or `full_name`), `survivor_eid`/`survivor_name` (the node that keeps the edges; this is bookkeeping only, because the name always comes from the standard), `members`, `member_eids`, and counts of edges moved, merged, collisions and self-loops. |
| `renames.tsv` | `eid`, `old_name`, `new_name` for each singleton whose name is not yet in standard form. |
| `edge_ops.tsv` | One row per loser edge operation: `op` (`move`, `merge`, `drop_self_loop`), type, direction seen from the survivor, the other endpoint (elementId, label, business key), source and target edge ids, the resulting properties (JSON) and notes. |
| `review.tsv` | Groups and names that were left alone, with `kind`, `proposed` name (empty when the standard form is known to be unreliable) and `reasons`. |

## Merge categories (overlapping flags, so the counts add up to more than 149)

- **Comma flip**: 109 groups have a member written "Surname, Given". In 90 of them, the flip
  is the only difference, e.g. `Hathaway, Jane` + `Jane Hathaway`.
- **FJP single initial**: 19 groups have a member written "Surname I" (FJP bibliography),
  e.g. `Mann J` + `J. Mann`. All 19 are also `initials_only`.
- **Case / accent / punctuation** (members' standard forms differ): 15 Latin groups, e.g.
  `Tova Beʾeri`/`Tova Beeri`, `Ben-Zion`/`Benzion Schereschewsky`, `Tamer el-Leithy`/`El-Leithy`,
  and 7 Hebrew groups (see next item). The display form is picked by `choose_display`: no `?`,
  then more diacritics, then initials in house style, then a form already stored, then the
  highest degree.
- **Hebrew**: 7 groups, all Hebrew initials written with a geresh, a period or bare, e.g.
  `ש' אסף` + `ש. אסף`. A geresh beats a period, which beats a bare initial.
- **Initial dot style only**: 6 groups, e.g. `Colin F Baker` + `Colin F. Baker`.
- **`initials_only`**: 41 groups (34 Latin and 7 Hebrew) whose final name is initials plus a
  surname, e.g. `F. Rosenthal` + `Rosenthal F`. Every member has the same initials and surname,
  but nothing proves they are one person. **Check these first.**
- **'?'-cleaned**: 1 group. `Samuel Poznanski` + `Poznanski, Samuel` merge, and
  `Samuel Poznan?ski` goes to review as `lost_accent`.

## Rename categories (358 in total; each rename is in exactly one bucket)

- FJP single initial, `Marx A` to `A. Marx`: 130. All of them come from `biblio`.
- Comma flip: 164. This is 123 plain flips (`Chapira, Bernard` to `Bernard Chapira`), 34 with
  initials (`Wansbrough, J.` to `J. Wansbrough`), 6 that also restyle initials and 1 all-caps
  name (`LARREA PALACÍN, A. DE` to `A. de Larrea Palacín`).
- Initials restyled only, dots and spaces: 53 (`S.A. Kaufman` to `S. A. Kaufman`,
  `Estara J Arrant` to `Estara J. Arrant`).
- Hebrew nikud removed: 9.
- Unicode composition only: 1 (`Muḥammad b. Mūsā al-Khūwārizmı̄`, no visible change).
- All-caps to title case only: 1 (`USULA ÇAKUTİNOLU KOKEMU`, which looks like OCR junk; see
  "Check first").

## Edges

There are 612 moves: WROTE 596, COLLABORATED_WITH 7, AFFILIATED_WITH 4, STUDIED 2, and
MENTIONS_PERSON, RELATED_TO and TRAVELED_TO 1 each. There are 12 collisions, where a loser
edge has the same identity as an edge the survivor already has: WROTE 9, COLLABORATED_WITH 2
(symmetric, so the direction is ignored) and MENTIONS_PERSON 1. Together they merge 13 loser
edges (Sharīf Kanāʿana merges 2 into 1). Properties merge as the policy says: `data_sources`
and `source_books` are unioned, `page_uuids`/`book_uuids`/`evidence_alt` lists are added, and
`confidence` takes the maximum. `evidence_page` keeps the primary edge's value, so the other
value survives only in the change log. No NULL-vs-list `data_sources` conflict occurred.

## Review kinds (833 rows)

| kind | rows | meaning |
|---|---|---|
| `review_only` | 707 | A flagged name that the standard would leave as it is: single token (542), title such as Dr./Prof./הרב (71), `?` (44), phrase (19), several authors (14), mixed scripts (12), parentheses (10), and others. |
| `rename` | 100 | A rename was withheld: several FJP trailing initials (52, e.g. `Haelst J V`), no capital letter or junk (22), Arabic name with a comma (8), French `M.` (3, e.g. `Wiet, M. G.`), `Gil, Palestine`, `Josep Pellicer I Bru`, and others. |
| `merge` | 12 | An exact group was withheld: single-token pairs (`al-Dhahabi`/`al-Dhahabī`), `?` members (`Peter Scha?fer`/`Peter Schäfer`), multi-initial FJP (`Udovitch A L`). |
| `near_exact` | 7 | Names that differ only by spaces or an inner geresh: `M. Ben-Sasson`/`M. Ben Sasson`, `ש.ד גויטיין`, `ריצ'לר`/`ריצלר`. Never merged automatically. |
| `clash` | 3 | The Scholar name is also a Person name: `Asher`, `Assaf`, `H. Yalon`. |
| `empty_key` | 3 | All `?`, so the name is unrecoverable. |
| `lost_accent` | 1 | `Samuel Poznan?ski`, left out of its group's merge. |

## Named cases

- **Jane Hathaway**: merge (`Hathaway, Jane` into `Jane Hathaway`). 1 edge moves, 28 edges in total.
- **Oded Zinger**: merge. 14 edges move, 176 in total. Hebrew `עודד זינגר` stays separate (a different script).
- **Marina Rustow**: merge. 60 edges move from `Rustow, Marina` into `Marina Rustow` (54), 114 in total.
- **Bernard Chapira**: rename `Chapira, Bernard` to `Bernard Chapira`. `Chapira` (single token),
  `M. Chapira` and `Bernhard Schapira` stay as they are (for later review).
- **Goitein**: `Goitein, S. D.` (106) + `S.D. Goitein` (2) merge into `S. D. Goitein`.
  `Goitein, Shelomo Dov Fritz` is renamed to `Shelomo Dov Fritz Goitein`. `Shelomo Dov Goitein`
  (134) is unchanged and stays separate from `S. D. Goitein`, because initials and a full name
  are never merged automatically. In Hebrew, `ש' ד' גויטיין` + `ש ד גויטיין` merge.
  `ש.ד גויטיין` is a `near_exact` review row. `גויטיין` (single token) and `שלמה דב גויטיין`
  are unchanged. The two `A Mediterranean Society … Goitein/Goitien` junk names are review rows.
- **Estara J Arrant**: rename to `Estara J. Arrant` (42 edges), because `INITIAL_STYLE = "dotted"`.
  Set it to `"bare"` to keep `Estara J Arrant`. That also changes all 53 "initials restyled"
  renames and all initials written elsewhere.

## Check first (most doubtful automatic operations)

1. `USULA ÇAKUTİNOLU KOKEMU` to `Usula Çakutinolu Kokemu`: OCR junk (a TRANSCRIBED figure
   caption). It should be junk review, not a rename.
2. The 41 `initials_only` merges, especially those with thin evidence: `F. Rosenthal`/`Rosenthal F`,
   `J. Mann`/`Mann J` (7 works), `A. Dietrich`, `H. Knopf`, `R. Brody`, `J. Blau`.
3. Display picks: `‘Alī Bahjat` (begins with a quotation mark used as an ʿayn),
   `Á. Sáenz-Badillos` (accented initial chosen over `A.`), and `Michel De Certeau` /
   `Johannes Den Heijer` (a capitalised particle is kept as written).
4. Arabic comma names that the "Arabic name with a comma" rule did not catch, because `al-` is on
   the surname side: `al-'Asyūtī, Šams al-Din Muḥammad`, `al-Šayyāl, Gamāl al-Din`,
   `al-Zirikli, Kayr al-Din`. The flipped results read naturally.
5. FJP renames whose "surname" is also a common given name: `Paul L`, `Allan N`, `Herman M`,
   `Frank D`, `Maher M`, `Rabie H`, `Lev E`, `Amar Z`. All come from `biblio`, where
   "Surname I" is consistent.
