# Knowledge-graph consolidation plan: scholars, edges, ES links

Branch `kg-scholar-consolidation` (off `chatbot-fixes`). Based on a read-only census of the production graph
on this MBP (`bolt://localhost:7681`, Neo4j 2026.04.0 Community, no APOC), run 2026-10-11 local time
(2026-10-12 UTC), and fact-checked against the census outputs and both repos. Census artifacts:
[`census_2026-10-12/`](census_2026-10-12/). Nothing has been written to Neo4j or Elasticsearch.

## 1. Why

"What does Jane Hathaway's book discuss?" got "no indexed text from the work itself is available",
although all 501 pages of her book are in `bibliography_text_only_0.7`:

1. `find_scholars` (`src/backend/neo4j_service.py:173-223`) sorts the plausible candidates (score ≥ 0.55) by
   WROTE-edge count (lines 213-220), so it picks the PGP node `Hathaway, Jane` (1 WROTE) over `Jane Hathaway`
   (0 WROTE, 25 STUDIED). Separately, its `LIMIT 50` has no `ORDER BY` (a latent bug).
2. `_person_name_variants` (`src/backend/search_bibliography.py:149-181`) treats the last token as the
   surname, so `Hathaway, Jane` → `H. Jane` / `Jane, H.`; `search_by_author` matches 0 pages.
3. `graph_scholar` (`lms_agentic_search.py:3203-3271`, unchanged by chatbot-fixes) uses `candidates[0]` only,
   with no fallback, and synthesis rule 8 then says "no indexed text".

Same for Oded Zinger (0 of 520 pages), Marina Rustow (0 of 135) and Bernard Chapira (0 of 8). The given-first
form matches every page.

## 2. Census (differences from the 2026-10-11 handoff marked ≠)

| Item | Value |
|---|---|
| Graph | 77,215 nodes, 186,231 relationships, 36 relationship types |
| Uniqueness constraints | ≠ 7, not 3: Scholar.name, Person.name, BookArticle.article_id, BiblicalPerson.name, Fragment.canonical_shelfmark, Institution.name, Place.name |
| Scholar nodes | 3,375; properties `name`, `data_sources` (9 shapes), `source_books` (1,770), `pipeline_version`; no alias property |
| Name formats | Latin "Given Surname" 1,583; Hebrew script 815; "Surname, Given" 293 (pgp 174, enriched 110, pgp+enriched 7, biblio 1); "Marx A" style 187; one-word Latin 484; 51 contain "?" (lost accent) |
| Degree | 0: none; 1: 1,990; 2+: 1,385. 697 Scholars have no WROTE |
| **Exact duplicate clusters** | ≠ depends on one rule detail. Deleting punctuation (so `Ben-Zion` = `Benzion`, `ד"צ` = `ד.צ.`): **147 clusters / 302 nodes / 155 surplus**. Turning punctuation into spaces: 144 / 296 / 152 (142 / 148 if "?" is not deleted). Handoff: 182 / 195 (not reproducible). Of the 147: **138 have no flags** (144 surplus); 4 involve a "?" name, 8 match only because punctuation or spacing is ignored, 3 contain likely medieval figures, 1 (X137) is a Hebrew book-title junk pair. Hathaway, Zinger and Rustow are exact. **Goitein's exact pair is only `Goitein, S. D.` + `S.D. Goitein` (X085)**; `Shelomo Dov Goitein` is not in it (review group R0039). **Chapira has no "Bernard Chapira" twin** |
| Review candidates | 481 groups (35 high, 399 medium, 47 low confidence), 765 pairs: initials vs full name 251, one-word surname 201, Latin/Hebrew 151, middle name 66, … 17 look-alike groups (16 different people, e.g. Mark R. Cohen vs Martin Cohen). 123 likely medieval/rabbinic figures stored as Scholar; 3 names exist as both Scholar and Person (Asher, Assaf, H. Yalon) |
| Junk | 333 flagged. 105 hard junk drawn from (overlapping) title/description 88, placeholder 20 ("Scholar", "Author", "[anonymous]"), several-people-in-one-name 15, empty after normalisation 3. The 51 "?" names are flagged separately (3 of them are hard junk) |
| Hanging nodes resolved | exact merges only: 140 of 1,990 degree-1 nodes; with high+medium review groups: 642 |
| Merge cost (147 set) | highest-degree node survives: 412 edges rewired; node already named "Given Surname" survives: 724 (583 of those carry `pgp`). For the 144 set: 409 vs 691. 13 collisions after the exact merges (10 WROTE, 2 COLLABORATED_WITH, 1 MENTIONS_PERSON; ≠ not 21/19), 0 self-loops, 1 new COLLABORATED_WITH reverse pair. Exact + high/medium review merges: 38 collision groups (33 WROTE), 4 would-be self-loops |
| Parallel edges today | only 7 (PGP role edges Person→Fragment, each with a different `pgpid`: separate facts) |
| Symmetric pairs stored both ways | 404: COLLABORATED_WITH 197, RELATED_TO 176, MARRIED_TO 29, JOINED_WITH 2 |
| Other edge oddities | 4 self-loops on Person (Judah ha-Levi ×2, Hai Gaon, Saadia Gaon: probably different people merged); 284 inverse pairs `MENTIONED_IN` (pgp_website) alongside `MENTIONS_PERSON`; 4 Fragment→Place pairs with both ORIGINATED_FROM and ORIGINATED_IN |
| BookArticle | 8,460 nodes, no book_uuid/doi/ES id. ≠ 26 placeholders titled with a book-folder name (25, e.g. `zinger_princet`, plus the typo `ottomoman_era`): their WROTE edges are junk, their CITED_IN edges belong to the real book. Duplicate BookArticles inflate works for 334 scholars (up to 1,269 surplus WROTE); merging them by title would create 87–1,302 parallel WROTE and 133–1,078 parallel REFERENCES (title alone is unsafe: many nodes are titled with a journal name) |
| **WROTE.book_uuid** | ≠ **extraction provenance, not authorship**: of 1,291 Scholar-WROTE edges with a book_uuid, only 28 point to a book the scholar wrote. Junk "enriched" WROTE edges also sit on real books (Hathaway's 7570663f117234ac: Mouton, Ahmad al-Murri, two medieval Person nodes) |
| Edge provenance defects | 27 page_uuids missing from 0.7/0.8 (137 edges); 10 edges point to another book's page; book_uuid `3b77eafc…` (172 edges) matches no ES book; 82 merged edges hold several `source_books` but one `book_uuid` |
| Bibliography books | 61 book_uuids in 0.7 (0.8: same books, 7 more DOIs). Matched to BookArticle: ≠ 21 one / 18 several / 22 none (strict); 25 / 24 / 12 (inclusive). "In the Kingdom of Ishmael" is one node for 13 ES books. Conflicts: Aviv barley (ES: Nadia Vidro; graph: Sacha Stern), Gil J. Stein on "A Cosmopolitan City", Schirmann near-miss d8e1885f3307974f, Ginzei Kedem (a journal, 82 container nodes), edited volumes (From Cairo to Manchester, Interconnected Traditions) |
| Fragment.es_doc_id | 92.9% coverage; 592 stale ids (595 nodes) vs v8, all `Oxford_Bodleian…` but ≠ 57 are really NLI 49 / BNF 7 / Harvard 1. Proposed v8 id is free for 257 nodes (254 ids; 7 proposals collide among themselves) and already on another Fragment node for 335 nodes (334 ids); 3 have no proposal. 1,593 ids are already shared by 3,211 nodes. 249 graph ids valid in v8 are absent in v9/v10 (246 long-form British Library ids, 226 with a unique v10 short form) |
| Backup | none exists; store 655 MB (`/data`), idle since 2026-09-18; host disk 94% full (58.9 GiB free); NAS not mounted |
| `doi`, `has_local_copy` | ≠ exist on no node, yet the backend reads them (`neo4j_service.py` ~264, 400-401, 546-547): UnknownPropertyKey warnings on every `get_scholar_rag_evidence` call |

## 3. Plan, in order

Every write tool is **dry-run by default**, writes a change log keyed on business keys (names, article_id,
canonical_shelfmark — elementIds change after a restore) and can revert from that log.

### P0. Backup (before any write)

1. **Logical export** (no downtime): every node and relationship through the Python driver in read
   transactions, ~9 s, ~15 MB gz, to the git-ignored `backups/neo4j/<UTC>/` (prototype:
   `census_2026-10-12/kg_export_prototype.py`; restore outline: `kg_restore_outline.py`). It is
   read-committed, not a snapshot: pause any historical-document-analysis writer on bolt 7681 while it runs; it
   does not carry users/roles. Counts are checked before and after.
2. **Offline `neo4j-admin database dump`** at a quiet time Isaac picks (runbook:
   `census_2026-10-12/backup_options.json`): `touch` the MAINTENANCE file; `docker stop -t 120
   genizah_search-neo4j-1`; dump `neo4j` and `system` with a throwaway `neo4j:2026.04.0` container (the
   `--user 7474:7474` flag is untested); `docker start`; verify counts and that all indexes are ONLINE; check one
   knowledge-graph endpoint on `:8000` (the backend driver's reconnect is untested); remove MAINTENANCE. About 1
   minute of graph downtime, budget 5 (chat graph tools and map layers fail meanwhile; search does not).
3. **Restore rehearsal** into a scratch volume on `127.0.0.1:7691`, removed afterwards.
4. Copy both off this 94%-full disk (NAS) once it is mounted.

### P1. Backend fixes (no graph writes; fixes the four scholars now)

A patch was prototyped in a scratch copy and replayed read-only against production: Hathaway 501 pages, Zinger
520, Rustow 135, Chapira 8; "S.D. Goitein" still resolves to `Shelomo Dov Goitein`, `Estara J Arrant` to
itself; backend suite 925 passed with the same 15 failures and 3 errors as HEAD (environment-related).

- `person_names.display_person_name`: "Surname, Given" → "Given Surname"; "Marx A" → "A. Marx" (the
  prototype gives "A Marx"; add the period for single-letter initials).
- `_person_name_variants`: flip first, keep the original as an extra variant.
- `find_scholars`: count **distinct** works, `ORDER BY` before `LIMIT`, rank exact-identity matches (same
  name tokens, initials allowed, any order) first, then works.
- `graph_scholar`: when the first candidate yields 0 pages, retry the other exact-identity candidates.
- `search_by_author`: optional `book_uuids` filter, used only once P5 has written `BookArticle.book_uuids`;
  ignore WROTE edges tagged only `enriched` for that path.
- New (not in the prototype): read `doi` / `has_local_copy` via `properties(b)[…]` to end the warnings.
- Known side effects to accept or tune: `Moshe Gil` resolves to `Gil, Moshe` (16 works) instead of `Moshe Gil
  / משה גיל` (35) until those nodes are merged (same 442 pages); the bare-surname variant of a flipped
  "Surname, I." node admits any author with that surname (0 such pages today).
- Optional in the same PR, bugs found on the way: `get_institution_detail` matches a nonexistent `MENTIONS`;
  `get_scholar_detail` (~562) and `get_place_detail` (~774) match `MENTIONED_IN` between BookArticle and Place,
  but `MENTIONED_IN` only runs Person→Fragment; `get_place_detail` reads `r.certainty` (the property is
  `certain`); the relationship query in `get_scholar_rag_evidence` (~427-436) returns both directions of
  symmetric pairs.

Tested on the dev backend on `:8010` (the separate dev container used for the v10 beta), then deployed with
Isaac's OK (backend rebuild).

### P2. Canonical scholar name (one rule, in code, with tests)

New `src/backend/person_names.py` (does not exist yet): `display_person_name` (Given Surname) and
`identity_key` (flip "Surname, Given", NFKD minus combining marks, casefold, **delete** punctuation and
apostrophe-like marks without inserting a space, so `Ben-Zion` = `Benzion` and `ד"צ` = `ד.צ.`, collapse
whitespace). Names containing "?" match by deleting it but always go to review. The census clusters are
regenerated from this function. **Never** key on surname + first initial (`scholar_normalizer.name_key` in
historical-document-analysis does; it would merge Mark R. Cohen and Martin Cohen).

### P3. Scholar consolidation (`scripts/kg/consolidate_scholars.py`)

- **Auto-merge** only unflagged exact clusters (~138). Flagged clusters ("?" names, punctuation/spacing-only
  matches, medieval members) go to review; X137 goes to the junk sheet.
- **Survivor**: keep the highest-degree node (412 rewires instead of 724) and rename it to the canonical
  "Given Surname" form; delete the merged nodes in the same transaction (`Scholar.name` is unique).
- **Eval names** `Shelomo Dov Goitein` and `Estara J Arrant` stay canonical. Goitein is a review merge: the
  exact pair X085 alone would leave `S. D. Goitein` as a separate canonical node, so X085 is applied together
  with review group R0039 (all Goitein forms, Latin and Hebrew) once Isaac approves it.
- **Aliases**: new `aliases: LIST<STRING>` on the survivor holding every merged spelling (Latin and Hebrew).
- **Edges**: rewire incoming and outgoing edges in Python (no APOC), MERGE into an existing edge of the same
  identity using the edge policy (P4); keep `data_sources` (map queries filter `'pgp' IN r.data_sources`).
- **Review sheet** (CSV Isaac approves row by row): the 481 review groups, flagged exact clusters, the 333 junk
  candidates, Latin/Hebrew pairs (merge with Hebrew names as aliases), the 123 medieval figures stored as
  Scholar (relabel to Person?), the 3 Scholar/Person name clashes, and Chapira (`Chapira, Bernard` →
  `Bernard Chapira`; candidates `Chapira`, `Bernhard Schapira`, `M. Chapira`, `Bernard Shapira`). Approved rows
  are applied by the same tool.

### P4. Edge consolidation (graph-wide)

Policy (`census_2026-10-12/edge_merge_policy.md`): a deterministic **primary** edge (has `page_uuid` >
confidence high > already on the survivor > lowest elementId) keeps its `book_uuid`, `page_uuid`, `evidence`;
the others' values go to lists `book_uuids`, `page_uuids`, `evidence_alt`; `source` keeps the first value;
union `data_sources` and `source_books`; `confidence` max; `uncertain` AND; `certain`/`deceased`/`has_*` OR;
`pgpid`, `role` and REFERENCES `pages` are identity and never merged; never merge an edge whose `data_sources`
is NULL (the map's PGP tier) with one that has a list.

- **Symmetric pairs** (404): keep one direction per pair (chosen by the endpoints' unique keys), merge the
  other into it.
- **Parallel PGP role edges** (7): keep (different `pgpid`).
- **Self-loops** (4), **inverse pairs** (284), **ORIGINATED_FROM/IN** (4): review sheet.
- Merges created by P3 and P5 go through the same function, so no parallel edges are ever created.

### P5. Books ↔ ES links (`scripts/kg/link_books.py`)

- `BookArticle.book_uuids: LIST<STRING>` and `doi` from the bibliography index (per book, after confirming
  its pages are in the live index).
- Match ES books to nodes by title (+ parallel titles, DOI in url) confirmed by author surname; edited volumes,
  Hebrew-author twins and the conflicts listed in §2 go to review.
- **Never** derive authorship from `WROTE.book_uuid` (it is provenance); do not trust the provenance defects in
  §2 when linking.
- Indexed books with no node: create the node from the book's metadata (title, DOI, book_uuids) with WROTE
  from its canonical ES authors (through P2/P3 names).
- The 26 folder-name placeholder nodes: review sheet (merge into the real book, keep CITED_IN, drop junk
  WROTE).
- **Junk "enriched" WROTE on real books**: review sheet of WROTE edges tagged only `enriched` whose scholar is
  not among the book's metadata authors (e.g. Mouton and Ahmad al-Murri on Hathaway's book); they feed
  `/book-info` `authors[0]` today.
- **Duplicate BookArticles** (334 scholars' works inflated): out of scope for the first pass; needs an identity
  rule stronger than title (title + year + ≥5 words, or DOI) and its own dry run, because merging creates
  parallel WROTE/REFERENCES.

### P6. Fragment ES ids (separate, later)

Re-point the 257 nodes whose proposed v8 id is free (check the 7 colliding proposals first); fix `HELD_AT` /
institution for the 57 non-Bodleian items mislabelled `Oxford_Bodleian…`; the 335 nodes whose v8 id is already
on another Fragment node need a Fragment merge (a separate task). Attach ids when an index goes live, not when
it is built.

### P7. Importers (historical-document-analysis)

A fix made only in the graph is undone by the next rebuild. Separate handoff:
[`docs/handoffs/2026-10-12_kg_indexing_historical_document_analysis.md`](../handoffs/2026-10-12_kg_indexing_historical_document_analysis.md).

## 4. Decisions needed from Isaac

| # | Question | Recommendation |
|---|---|---|
| D1 | Take the logical export now (to `backups/neo4j/`)? | Yes |
| D2 | When can Neo4j stop for ~1 minute for the offline dump, and may I stop/start `genizah_search-neo4j-1` then? Restore rehearsal on `:7691`? | Pick a quiet time; yes to the rehearsal |
| D3 | Ship P1 (backend fixes) first, ahead of any graph write? | Yes: it fixes the four scholars without touching the graph |
| D4 | Survivor = highest-degree node renamed (412 rewires) vs the node already named "Given Surname" (724)? | Highest-degree, renamed |
| D5 | Keep `Shelomo Dov Goitein` and `Estara J Arrant` canonical (middle initials kept), and apply the Goitein merge as a review group? | Yes |
| D6 | Link property: `book_uuids` list (+ `doi`)? historical-document-analysis's `BOOK_LINK_INDEXING_CONTRACT.md` mentions a single `es_book_id` (an ES doc_id stem) only as an example | `book_uuids` list (one node can be 13 books); add it to the contract's Fields table |
| D7 | Symmetric pairs → one direction with evidence kept in lists? | Yes |
| D8 | Inverse pairs, self-loops, ORIGINATED_FROM/IN, placeholder books, junk enriched WROTE, Hebrew-author book twins, medieval figures as Scholar | Review sheet |
| D9 | Fragment merges for the 335 stale nodes, and duplicate-BookArticle merges, in this effort? | No, separate tasks |
| D10 | Mount the NAS for off-machine backups; set up a nightly logical export (no downtime)? | Yes |

## 5. Rules

- Read-only on Neo4j and Elasticsearch until Isaac approves a dry run; then one approved step at a time.
- No container restarts or stops without Isaac's OK (the offline dump is the only planned one).
- Never print `.env` contents or credentials.
- Test commands keep `LLM_STUDIO_URL=http://127.0.0.1:9`.
- Branch + PR; never commit to master.
