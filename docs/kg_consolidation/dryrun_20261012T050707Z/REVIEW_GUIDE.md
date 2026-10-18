# Scholar consolidation dry run, 2026-10-12 05:07Z: review guide

This is a dry run of `scripts/kg/consolidate_scholars.py` against production Neo4j
(`bolt://localhost:7681`), made with the "form most likely to appear" name standard. It used one
read session (`default_access_mode=READ`) and one read transaction (`execute_read`, MATCH/RETURN
queries only), and wrote nothing to the graph. The counts were **77,215 nodes and 186,231
relationships** (3,375 Scholars) before the run and the same after it. The name rules are in
`scripts/kg/person_names.py`. The edge merge rules are in `../census_2026-10-12/edge_merge_policy.md`.

`plan.json` is format 2 (it records member degrees), and `plan_sha256` is
`efbd66188942d5e413498ec73fefac2bfd1d524b1b35d738d2ccea443468715a`. That is the same as the
earlier `dryrun_20261012T050408Z` run, so the plan reproduces. **Apply from this directory** and
keep its change logs, because they are the only way to revert.

## What the standard does now (changes since `dryrun_20261012T034936Z`)

- **A merged group's display name is the form most likely to appear.** The form whose members
  carry the most edges wins. A form with `?` (a lost accent) never wins. When the edge counts are
  tied, the old tie-breaks apply in this order: more diacritics, then house-style initials, then a
  form already stored, then lexicographic order. The name is always a form that some member
  already has.
- **Particles are lower case after the first word**: de, den, der, van, von, del, du
  (`Michel de Certeau`, `N. R. M. de Lange`, `Johannes den Heijer`). The last word (the surname)
  and `al-` glued to an all-caps word are never changed. Ben/Bar, ben, b., ibn, di, della, le, la
  and el- stay as written.
- **No title-casing.** An all-caps name goes to review.
- **Initials-only merges go to review.** Same initials and same surname do not prove the same
  person (`INCLUDE_INITIALS_ONLY_MERGES = False`).
- **Staged apply.** `--only "Name A,Name B"` applies only the named operations. Interacting
  operations must be applied together.

## Headline

| | |
|---|---|
| Scholars read | 3,375 |
| MERGE groups (automatic) | 108 groups, 219 members, **111 nodes removed** (0 initials-only, 0 Hebrew) |
| RENAMES (automatic, singletons) | 358 |
| Loser edges moved to the survivor | 524 |
| Loser edges merged into an existing survivor edge | 9 (8 collisions) |
| Self-loops dropped | 0 |
| Review rows (nothing changes) | 876 rows, 938 distinct Scholars |
| **Expected after a full `--apply`** | **77,104 nodes / 186,222 relationships** (Scholars 3,375 to 3,264) |
| Expected after stage 1 only (below) | 77,212 nodes / 186,231 relationships (Scholars 3,372) |

A move deletes one edge and creates one, so the relationship count only drops by the 9 merged
edges.

## Files

| File | What it holds |
|---|---|
| `summary.json` | Counts by category, plus `plan_sha256`. |
| `plan.json` | The full plan: every operation with its before-state, edge moves and merges, survivor updates and member degrees. `--apply --plan <this dir>` plans again from the live graph and refuses any selected operation that the new plan does not reproduce. |
| `merges.tsv` | One row per MERGE: `key`, `canonical`, `category` (all `full_name`), `survivor_eid`/`survivor_name` (the node that keeps the edges; bookkeeping only), `members`, `member_eids`, and counts of edges moved, merged, collisions and self-loops. |
| `renames.tsv` | `eid`, `old_name` and `new_name` for each singleton not yet in standard form. |
| `edge_ops.tsv` | One row per loser edge operation (`move`/`merge`): type, direction seen from the survivor, the other endpoint, source and target edge ids, resulting properties (JSON) and notes. |
| `review.tsv` | Groups and names left alone, with `kind`, `proposed` name (empty when the standard form is known to be unreliable) and `reasons`. |

## Merge categories (108; each group is in exactly one bucket)

- **Comma flip only, 90 groups.** For example, `Hathaway, Jane` + `Jane Hathaway`, or
  `Rustow, Marina` + `Marina Rustow`.
- **Comma flip plus an accent, case or particle difference, 7 groups.** `Israël Lévi`,
  `José Martínez Delgado`, `Naïm Vanthieghem`, `Simḥa Assaf`, `Tamer el-Leithy`,
  `Tijana Krstić` and `Yeḥezkel David`.
- **Initial dot only, 6 groups.** `Colin F. Baker`, `Craig A. Perry`, `Gene W. Heck`,
  `Julia L. Galliker`, `Michael G. Wechsler` and `Rebecca J. W. Jefferson`.
- **Accent, apostrophe or hyphen only, 5 groups.** `‘Alī Bahjat`, `Ben-Zion Schereschewsky`,
  `Dror Ze'evi`, `Tova Beeri` and `Yehoshua Frenkel`.

### How the display name was chosen where member forms differ (12 groups)

The figures in parentheses are edges.

| Canonical | Members | Decided by |
|---|---|---|
| Naïm Vanthieghem | `Vanthieghem, Naïm` (17), `Naim Vanthieghem` (2) | frequency |
| Tijana Krstić | `Tijana Krstić` (12), `Krstić, Tijana` (1), `Tijana Krstic` (1) | frequency |
| Tamer el-Leithy | `el-Leithy, Tamer` (7) + `Tamer el-Leithy` (1) = 8, against `Tamer El-Leithy` (5) | frequency |
| Simḥa Assaf | `Assaf, Simḥa` (2) + `Simḥa Assaf` (1) = 3, against `Simha Assaf` (2) | frequency |
| Tova Beeri | `Tova Beeri` (3), `Tova Beʾeri` (1) | frequency (the old rule picked Beʾeri) |
| Yehoshua Frenkel | `Yehoshua Frenkel` (2), `Yehoshuʿa Frenkel` (1) | frequency (the old rule picked Yehoshuʿa) |
| Yeḥezkel David | `Yeḥezkel David` (2), `David, Yehezkel` (1) | frequency |
| Israël Lévi | `Lévi, Israël` (1), `Israel Lévi` (1) | tie, so more diacritics |
| José Martínez Delgado | `Delgado, José Martínez` (1), `Jose Martinez Delgado` (1) | tie, so more diacritics |
| ‘Alī Bahjat | `‘Alī Bahjat` (1), `'Ali Bahjat` (1) | tie, so more diacritics |
| Ben-Zion Schereschewsky | `Ben-Zion …` (1), `Benzion …` (1) | tie, so lexicographic |
| Dror Ze'evi | `Dror Ze'evi` (1), `Dror Ze’evi` (1) | tie, so lexicographic (ASCII apostrophe) |

Every member's old name goes into the survivor's `aliases`. Only frequency decides the name;
which node survives is bookkeeping.

## Rename categories (358; each rename is in exactly one bucket)

- **FJP "Surname I" to "I. Surname": 130.** These are 116 one-word surnames (`Marx A` to
  `A. Marx`) and 14 multi-word surnames (`Diez Macho A` to `A. Diez Macho`, `del Valle C` to
  `C. del Valle`). All come from `biblio`.
- **Comma flip: 163.** These are 121 plain flips (`Chapira, Bernard` to `Bernard Chapira`), 33
  with initials kept (`Goldstein, Bernard R.` to `Bernard R. Goldstein`), 7 that also restyle
  initials (`Cowan, J.M.` to `J. M. Cowan`) and 2 that also lower a particle
  (`De Certeau, Michel` to `Michel de Certeau`, `Den Heijer, Johannes` to `Johannes den Heijer`).
- **Initials restyled only, dots and spaces: 53.** For example, `A.M. Habermann` to
  `A. M. Habermann`, and `Estara J Arrant` to `Estara J. Arrant`.
- **Particle lowered only: 2.** `N. R. M. De Lange` to `N. R. M. de Lange`, and
  `Wout Jacques Van Bekkum` to `Wout Jacques van Bekkum`.
- **Hebrew nikud removed: 9.**
- **Unicode composition only: 1.** `Muḥammad b. Mūsā al-Khūwārizmı̄` has no visible change.

The two all-caps renames of the earlier run (`USULA ÇAKUTİNOLU KOKEMU` and
`LARREA PALACÍN, A. DE`) are now review rows.

## Edges

There are 524 moves: WROTE 511, COLLABORATED_WITH 8, AFFILIATED_WITH 3, STUDIED 1 and
TRAVELED_TO 1. There are 8 collisions, where a loser edge has the same identity as an edge the
survivor already has. They merge 9 loser edges: WROTE 6 (Alexander Scheiber, Donald S. Richards,
Geoffrey Khan, Nehemia Allony, Shaul Shaked, Solomon Schechter) and COLLABORATED_WITH 2, which is
symmetric so direction is ignored (Philipp Speiser–Christina Strasse, and Sharīf Kanāʿana–Bassām
al-Kaʿbī, where 2 edges merge into 1). Properties merge as the policy says. `data_sources` and
`source_books` are unioned, `page_uuids`/`book_uuids`/`evidence_alt` lists are added, and
`confidence` takes the maximum. No NULL-vs-list `data_sources` conflict occurred.

## Review (876 rows; nothing in review changes)

| kind | rows | meaning |
|---|---|---|
| `review_only` | 733 | A flagged name that the standard would leave as it is: single token (553), title such as Dr./Prof./הרב (71), `?` (44), no capital letter (22), phrase (19), several authors (14), mixed scripts (12), parentheses (10), **all caps (4)** and others. |
| `rename` | 76 | A rename was withheld: several FJP trailing initials (52, e.g. `Haelst J V`), Arabic name with a comma (8), single token (4), **all caps (3)**, French `M.` (3), `Gil, Palestine`, `Josep Pellicer I Bru` and others. |
| `merge` | 53 | An exact group was withheld: **initials only (41)**, single-token pairs (7, e.g. `al-Dahabi`/`al-Dahabī`), `?` members (3, e.g. `Peter Scha?fer`/`Peter Schäfer`), multi-initial FJP (2). |
| `near_exact` | 7 | Names that differ only by spaces or an inner geresh: `M. Ben-Sasson`/`M. Ben Sasson`, `ש.ד גויטיין`, `ריצ'לר`/`ריצלר`. Never merged automatically. |
| `clash` | 3 | The Scholar name is also a Person name: `Asher`, `Assaf`, `H. Yalon`. |
| `empty_key` | 3 | All `?`, so the name is unrecoverable. |
| `lost_accent` | 1 | `Samuel Poznan?ski`, left out of the `Samuel Poznanski` merge. |

**Initials-only groups now in review: 41 (34 Latin, 7 Hebrew).** Examples are `S. D. Goitein`
(`Goitein, S. D.` + `S.D. Goitein`), `F. Rosenthal`/`Rosenthal F`, `J. Mann`/`Mann J`,
`A. Neubauer` (3 members), `J. Blau`, `R. Brody`, `S. Assaf`, `A. Saenz-Badillos` (3 members), and
the Hebrew `ש. אסף`/`ש' אסף`, `ח. שירמן`/`ח שירמן` and `ש ד גויטיין`/`ש' ד' גויטיין`. To merge them
later, either approve them one by one or set `INCLUDE_INITIALS_ONLY_MERGES = True` and run a new
dry run.

**All-caps names now in review: 7.** The rename-kind rows are `LARREA PALACÍN, A. DE` (the
proposal `A. DE LARREA PALACÍN` keeps the caps), `J.M.` and `I`. The review-only rows are `ARCE`,
`E. AHSTOR`, `E. ROMERO` and `USULA ÇAKUTİNOLU KOKEMU` (OCR junk).

## Named cases

- **Jane Hathaway**: merge. 1 edge moves from `Hathaway, Jane` into `Jane Hathaway` (27), 28 edges in total.
- **Oded Zinger**: merge. 14 edges move from `Zinger, Oded` into `Oded Zinger` (162), 176 in total.
  The Hebrew `עודד זינגר` (31 edges) stays a separate node, because it is a different script and
  outside this work.
- **Marina Rustow**: merge. 60 edges move from `Rustow, Marina` into `Marina Rustow` (54), 114 in total.
- **Bernard Chapira**: rename `Chapira, Bernard` to `Bernard Chapira`. `Chapira` (single token,
  review), `M. Chapira` (already standard) and `Bernhard Schapira` are unchanged.
- **Estara J Arrant**: rename to `Estara J. Arrant` (42 edges), because `INITIAL_STYLE = "dotted"`.
  The graph has only this one form. Set `INITIAL_STYLE = "bare"` to keep `Estara J Arrant`. That
  changes every initial the standard writes, including the 53 restylings and the 130 FJP renames
  (`Marx A` would become `A Marx`), so it needs a new dry run.
- **Michel de Certeau**: rename `De Certeau, Michel` to `Michel de Certeau` (lower-case particle).
- **Israel / Israël Lévi**: merge into `Israël Lévi`. It is a 1-to-1 tie, broken by diacritics.
  The survivor is `Lévi, Israël` (pgp), and `Israel Lévi` (enriched) becomes an alias with 1 edge
  moved. `Israel Levin` is a different person and is untouched.
- **Tamer el-Leithy**: a 3-way merge into `Tamer el-Leithy`. By frequency the lower-case form has
  8 edges against 5 for `Tamer El-Leithy`. 12 edges move. `Professors Tamer El-Leithy` is a
  title-prefix review row. This operation interacts with `Tijana Krstić` (a COLLABORATED_WITH
  edge between their members), so the two must be applied together.
- **Sáenz-Badillos**: no change. `Saenz-Badillos A` (2), `A. Sáenz-Badillos` (1) and
  `Á. Sáenz-Badillos` (1) form an initials-only review group. The frequency rule proposes the
  accentless `A. Saenz-Badillos`, because the FJP form has the most edges, so do not accept that
  proposal blindly. `A?ngel Sa?enz-Badillos` is a `?` review row.
- **ʿAlī Bahjat**: merge `‘Alī Bahjat` + `'Ali Bahjat` into `‘Alī Bahjat`. It is a 1-to-1 tie,
  broken by diacritics. The ʿayn is written with U+2018, a left quotation mark, as stored.
  `Aly Bey Bahgat` (3) and `Bahgat` (1) have different keys and stay separate.

## Staged apply

Stage 1 is the four key people: 3 merges and 1 rename, 75 edges moved and 3 nodes removed. The
selection matches exactly those four operations and splits no group of interacting operations:

```
python scripts/kg/consolidate_scholars.py --apply --plan docs/kg_consolidation/dryrun_20261012T050707Z \
  --only "Jane Hathaway,Oded Zinger,Marina Rustow,Bernard Chapira"
```

Pass canonical names to `--only`, because it splits on commas, so a member name such as
`Chapira, Bernard` cannot be given. The only multi-operation group in the plan is
`Tamer el-Leithy` + `Tijana Krstić`. The rest can then go in one run, or in batches with `--limit N`.
Operations that are already applied are detected and skipped. **Nothing may be applied to
production without Isaac's approval.**

## Check first

1. **Ties with no frequency signal (1 edge against 1).** `Israël Lévi`: English-language
   literature mostly writes `Israel Lévi`, while French writes Israël. `‘Alī Bahjat` uses a
   quotation mark as the ʿayn. `José Martínez Delgado` and `Ben-Zion Schereschewsky` read
   correctly.
2. **Frequency over diacritics.** `Tova Beeri` was chosen over `Tova Beʾeri`, and
   `Yehoshua Frenkel` over `Yehoshuʿa Frenkel`. This follows "most likely to appear". The
   transliterated forms survive as aliases.
3. **Review proposals biased by FJP's ASCII spelling** (no change now, but do not accept them
   blindly later): `A. Saenz-Badillos`, `M. Beit-Arie`/`Beit-Arie` (the scholar writes
   Beit-Arié), and `al-Dahabi`.
4. **Arabic comma names** that the "Arabic name with a comma" rule did not catch, because `al-` is
   on the surname side: `al-'Asyūtī, Šams al-Din Muḥammad`, `al-Šayyāl, Gamāl al-Din` and
   `al-Zirikli, Kayr al-Din`. The flipped results read naturally.
5. **FJP renames whose "surname" is also a common given name**: `Paul L`, `Allan N`, `Herman M`,
   `Frank D`, `Maher M`, `Rabie H`, `Lev E` and `Amar Z`. All come from `biblio`, where
   "Surname I" is consistent.
