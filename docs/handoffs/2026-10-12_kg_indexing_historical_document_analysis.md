# Handoff: make knowledge-graph indexing produce canonical scholars, merged edges and ES links (historical-document-analysis)

Written 2026-10-12 (census run 2026-10-11 local time) for a fresh Claude Code session in **historical-document-analysis** (the repo that builds
the Neo4j knowledge graph and the Elasticsearch indices). On the MBP it is checked out at
`/Users/isaac1/Documents/historical-document-analysis` (master `417fb64`); Isaac's main copy is on the Studio.

## Paste this to start

> Read `docs/handoffs/2026-10-12_kg_indexing_historical_document_analysis.md` from the genizah_search repo
> (branch `kg-scholar-consolidation`; `git fetch` it first), then this repo's CLAUDE.md. genizah_search is consolidating the
> production knowledge graph (duplicate Scholar nodes, symmetric edges stored twice, no ES ids on book nodes).
> Make every importer here write the graph the consolidated way, so the next rebuild does not undo it. Work on a
> branch, open a PR, never commit to master. Read-only against production Neo4j and Elasticsearch; do not run any
> import, re-index or migration against production without Isaac's explicit OK. Post a short plan first.

## 1. Why

genizah_search's plan is [`docs/kg_consolidation/PLAN.md`](../kg_consolidation/PLAN.md) (census:
`docs/kg_consolidation/census_2026-10-12/`). In short:

- Scholar names are stored as raw text under a `Scholar.name` uniqueness constraint, so every spelling became
  its own node. There are 147 exact duplicate clusters (e.g. `Hathaway, Jane` / `Jane Hathaway`) and 481 more
  that need review. `Shelomo Dov Goitein` exists as at least 9 nodes across scripts and forms.
- The chatbot picks the wrong node and finds 0 pages for Jane Hathaway, Oded Zinger, Marina Rustow and Bernard
  Chapira.
- 404 symmetric relationships (COLLABORATED_WITH, RELATED_TO, MARRIED_TO, JOINED_WITH) are stored in both
  directions.
- `BookArticle` nodes carry no link to the bibliography index (no `book_uuid`, no `doi`); `doi` and
  `has_local_copy` exist on no node, because `enrich_kg_identifiers.py` and `mark_local_sources.py` were never
  applied to production.
- 26 `BookArticle` nodes are named after a book folder (25 such as `zinger_princet`, `jewish_proselytism`, plus
  the typo `ottomoman_era`): the enriched pipeline created a placeholder for the source book and attached the
  people the book mentions as its "authors" (junk WROTE edges). Junk "enriched" WROTE edges also sit on real
  books (Mouton and Ahmad al-Murri on Jane Hathaway's book).
- `WROTE.book_uuid` records which book a relation was **extracted from**, not authorship: of 1,291 Scholar-WROTE
  edges with a book_uuid, only 28 point to a book that scholar wrote.

genizah_search fixes the live graph once (dry-run first, with revert logs). This repo must make the importers
produce the same shape from now on.

## 2. What to change

### 2.1 One canonical scholar name

- Canonical display form: **"Given Surname"** (`Hathaway, Jane` → `Jane Hathaway`, `Marx A` → `A. Marx`).
  Middle names and initials are kept (`Estara J Arrant`, `Shelomo Dov Goitein` are pinned: genizah_search's evals
  require them exactly).
- **Identity key** for matching (the only key allowed for merging): flip "Surname, Given", NFKD minus combining
  marks, casefold, delete punctuation and apostrophe-like marks without inserting a space (so `Ben-Zion` =
  `Benzion`, `ד"צ` = `ד.צ.`), collapse whitespace. genizah_search implements
  it as `src/backend/person_names.py` (`display_person_name`, `identity_key`); copy the same rules (or vendor the
  module) and share its test cases so both repos agree. (genizah_search's module is planned, step P2 of its
  plan; it does not exist yet. Coordinate so both sides land the same rules.) Single-letter initials get a period
  (`Marx A` → `A. Marx`).
- **Never merge on surname + first initial.** `scholar_normalizer.name_key`
  (`src/datasets/document_models/scholar_normalizer.py:112`) maps "Mark R. Cohen" and "Martin Cohen" both to
  `cohen|m`; the census found 16 look-alike groups that are different people. `ScholarRegistry` must not use it
  to resolve to an existing node; keep it for suggestions only (review sheet).
- Every importer that writes a Scholar name goes through one function: resolve by identity key to an existing
  canonical node, else create the node with the canonical display form, and append the raw spelling to
  `aliases` (a `LIST<STRING>`; genizah_search adds it during consolidation). Hebrew-script spellings of a known
  scholar are aliases of the Latin node.

Name-writing call sites (fact-checked against `417fb64`; re-check line numbers):

| File | Lines | Today |
|---|---|---|
| `src/datasets/indexing/neo4j/knowlege_graph_poc.py` | 466-520, Scholar MERGE at 514 (`import_sources`) | `MERGE (s:Scholar {name: author_name})` with raw ';'-split PGP `sources.csv` authors in "Surname, Given" form, tagged `pgp` (`backfill_data_sources.py:11`). **This is where the 181 pgp "Surname, Given" Scholar nodes come from** (`Hathaway, Jane`, `Goitein, S. D.`, …). Person MERGEs are at 357, 383, 403, 547 |
| `src/datasets/indexing/neo4j/academic_kg_import.py` | 254, 259-288, 436, 696-702, 940-985, 1216-1232 | names go through `_normalise_entity` → `ScholarRegistry.resolve` (the surname + first-initial `name_key`) or `PersonNormalizer.normalize`: unsafe-key merges, not raw names |
| `src/datasets/indexing/neo4j/enhanced_kg_import.py` | 458, 527, 568, 600, 713, 719 | raw names |
| `src/datasets/indexing/neo4j/mark_local_sources.py` | 286-288 | `MERGE (sc:Scholar {name: $author})` with the raw name and no `data_sources` |
| `src/datasets/indexing/neo4j/biblio_import.py` | 76-86 (`_canonical_author`), 323-338 | `ScholarRegistry.resolve(raw) or display_form(raw)`; already tagged `biblio` before commit 1efa670, so it explains at most 1 comma node |
| `src/datasets/document_models/scholar_normalizer.py` | 81 (`display_form`), 112 (`name_key`) | the existing given-first formatter and the unsafe key |
| `pgp_person_relations_import.py` | — | not on the MBP checkout; find it on the Studio (it writes the symmetric relation pairs) |

### 2.2 Edges: MERGE, never duplicate

- Write every relationship with `MERGE` on its identity: type + endpoints, plus `pgpid` / `role` / REFERENCES
  `pages` where those are part of the fact (two PGP documents on one fragment are two facts).
- On a repeat, merge properties with genizah_search's policy
  (`docs/kg_consolidation/census_2026-10-12/edge_merge_policy.md`): a deterministic **primary** edge (has
  `page_uuid` > confidence high > already present > lowest id) keeps its `book_uuid` / `page_uuid` /
  `evidence`; the other values go to `book_uuids` / `page_uuids` / `evidence_alt`; `source` keeps the first
  value; union `data_sources` and `source_books`; `confidence` max; `uncertain` AND; `certain` / `deceased` /
  `has_*` OR. Use the same rule, or importer re-runs will diverge from the consolidated graph.
- **Symmetric types** (COLLABORATED_WITH, RELATED_TO, MARRIED_TO, JOINED_WITH): store one direction only, chosen
  deterministically (e.g. by the endpoints' unique keys), and MERGE regardless of the direction the extractor
  emitted.
- Never mix an edge whose `data_sources` is NULL with one that has a list (the site's map treats NULL as the PGP
  tier); set `data_sources` on every edge an importer writes (`mark_local_sources.py` writes none today).
- Do not create self-loops; log and skip them.

### 2.3 Books ↔ Elasticsearch at indexing time

- When the bibliography indexer (`src/datasets/indexing/bibliography/index_all_bibliography.py`) indexes a book,
  find its `BookArticle` node (DOI first, then title + author identity key) and set:
  - `book_uuids: LIST<STRING>` — append this book's `book_uuid` (`corpus_ids.book_uuid`, uuid5 of the normalised
    book key); one node can stand for several ES books ("In the Kingdom of Ishmael" is one node for 13).
  - `doi` — bare DOI per `docs/BOOK_LINK_INDEXING_CONTRACT.md`.
  - Update the contract: its Fields table has no ES link today (a single `es_book_id`, the ES doc_id stem, is
    only mentioned as an example under "Durable fix"); add the `book_uuids` list (decision D6 in genizah_search's
    plan, pending Isaac).
- If no node exists, create it from the book's own metadata (title, DOI, `book_uuids`) and add WROTE edges from
  its **metadata authors** (canonical names via 2.1). `biblio_import` (bibliography CSV) and
  `knowlege_graph_poc.import_sources` (PGP) already write author WROTE for their own sources; what is missing is
  linking the ES-indexed academic-literature books to the authors in their `*_metadata.json`.
- Do not create placeholder `BookArticle` nodes named after the book folder for the source of an extraction; the
  source is recorded on the edges (`book_uuid`, `page_uuid`, `source_books`), never as a WROTE target.
- Never write WROTE from people *mentioned in* a book (the Mouton / Ahmad al-Murri edges on Hathaway's book).
- Link only after the pages are confirmed in the live index (ids must exist in the served index).

### 2.4 Fragment ES ids

- `Fragment.es_doc_id` must match the **live** fragments index. Attach or refresh it when an index goes live,
  not when it is built: 249 graph ids valid in v8 are absent from v9/v10 (246 long-form British Library ids, 226
  with a unique v10 short form), and 592 Bodleian-format ids are stale against v8 (57 of them belong to
  NLI/BNF/Harvard items mislabelled `Oxford_Bodleian…`, whose `HELD_AT` is wrong too).
- Do not create a second Fragment node for an id that already exists (1,593 ids are shared by 3,211 nodes today).

### 2.5 Graph safety in the import scripts

- Dry-run mode by default, printing the nodes/edges it would create, merge or rename.
- A change log keyed on business keys (names, `article_id`, `canonical_shelfmark`; elementIds change on restore).
- Refuse to run against production without a recent backup (genizah_search's logical export or a
  `neo4j-admin` dump; see its `census_2026-10-12/backup_options.json`; `backup_neo4j_to_gcp.py` exists here but
  no backup has been verified).
- Prefer idempotent re-runs: running an importer twice must not change the graph the second time.

## 3. Tests to add

- Name: `Hathaway, Jane` / `Jane Hathaway` / `HATHAWAY, JANE` → one identity; `Mark R. Cohen` ≠ `Martin
  Cohen`; `Peter Schäfer` = `Peter Schafer`; Hebrew spellings attach as aliases; pinned names unchanged.
- Edges: importing the same relationship twice (and reversed, for symmetric types) leaves one edge with merged
  `data_sources`; two PGP documents on one fragment stay two edges.
- Books: indexing a book sets `book_uuids`/`doi` on its node, appends (not replaces) for a second ES book of the
  same node, creates a node with metadata-author WROTE when none exists, and never creates a folder-name
  placeholder.
- Idempotence: a second run is a no-op.

## 4. Rules

- Branch + PR; never commit to master.
- Read-only against production Neo4j (`bolt://<MBP>:7681`) and Elasticsearch; ask Isaac before any import,
  re-index or migration that touches production stores.
- Never print `.env` contents or credentials.
- Coordinate with genizah_search's consolidation (branch `kg-scholar-consolidation`): the importers should not
  run against production between its dry run and its apply step.
