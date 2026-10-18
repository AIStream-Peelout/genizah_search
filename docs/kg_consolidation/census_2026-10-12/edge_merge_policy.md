# Proposed edge-consolidation / property-merge policy (DRAFT, needs approval)

Read-only census of production Neo4j at bolt://localhost:7681, 2026-10-11 (77,215 nodes, 186,231
relationships, 36 relationship types). Nothing has been written. The numbers come from
`parallel_edges_by_type.json`, `merge_rewire_counts.json`, `works_count.json` and
`parallel_group_props.json` in this folder.

## 1. What there is to consolidate

| Bucket | Count | What the census shows |
|---|---|---|
| Same type, same direction, same endpoints (true parallels) | 7 groups / 7 surplus | All are PGP role edges (SCRIBE_OF 5, RECIPIENT_OF 1, SENDER_OF 1). Each pair has a **different `pgpid`**, i.e. two PGP documents on one fragment, one "certain" and one "(uncertain)". They are **not duplicates**. |
| Symmetric types stored in both directions | 404 pairs | COLLABORATED_WITH 197 (Scholar<->Scholar 168), RELATED_TO 176, MARRIED_TO 29, JOINED_WITH 2 |
| Self-loops already present | 4 | All on Person: Judah ha-Levi (COLLABORATED_WITH + RELATED_TO), Hai Gaon, Saadia Gaon (RELATED_TO). They are extraction artefacts. |
| Inverse-type pairs | 284 | Person-[MENTIONED_IN]->Fragment (pgp_website) alongside Fragment-[MENTIONS_PERSON]->Person (merged/enriched). CITED_IN vs REFERENCES: 0 |
| Same-direction synonyms | 4 | ORIGINATED_FROM + ORIGINATED_IN on the same Fragment->Place pair |
| Created by merging the Scholar EXACT clusters | 13 groups / 13 surplus | WROTE 10, COLLABORATED_WITH 2, MENTIONS_PERSON 1. The merge creates 0 self-loops and 1 new COLLABORATED_WITH reverse pair. |

No relationship carries a timestamp property (0 keys like created/updated/date/time). The only
version marker is `pipeline_version` (always "v3", on 12,533 relationships).

## 2. Identity keys: when two edges count as "the same edge"

Collapse two edges only when they match on the **identity key** for their type. If anything outside
that key differs, merge the properties using the table in section 3.

| Type family | Identity key |
|---|---|
| PGP role edges (SCRIBE_OF, SENDER_OF, RECIPIENT_OF, WITNESS_OF, PARTY_TO, AUTHOR_OF, AUTHORITY_OF, ...) | (type, person, fragment, **pgpid**) |
| MENTIONS_PERSON | (type, fragment, person, **role** when present) |
| REFERENCES | (type, book, fragment, **normalised `pages`**) |
| Symmetric (COLLABORATED_WITH, RELATED_TO, MARRIED_TO, JOINED_WITH) | (type, unordered {a, b}) |
| Everything else (WROTE, STUDIED, AFFILIATED_WITH, TRANSCRIBED, LIVED_IN, TRAVELED_TO, HELD_AT, HAS_TAG, WRITTEN_IN, MENTIONS_PLACE, ...) | (type, start, end) |

Under these keys:
- The 7 existing parallels are not collapsed. Recommendation: keep them.
- If the reviewer still wants them collapsed, merge them as follows: `pgpids` = list union, `uncertain` = AND (one certain edge wins), `relation` taken from the certain edge, `deceased` = OR.

## 3. Per-property merge rules

The **primary edge** of a group is chosen deterministically:
1. it has `page_uuid` (it is linked to Elasticsearch);
2. otherwise, `confidence` = "high";
3. otherwise, it was already on the surviving node;
4. otherwise, it has the lowest elementId.

Neo4j list properties must hold one primitive type and no nulls, and cannot hold maps. Every
"collect" rule below therefore writes a list of strings or a list of ints.

| Property (type, where present) | Rule | Evidence from the census |
|---|---|---|
| `data_sources` (list<str>, 85,057 rels) | **Union**, keeping order: primary first, then new values. | Same in all 404 reverse pairs. Differs in 6/10 WROTE merge collisions (`['pgp']` + `['enriched']`), and the union keeps `'pgp'`. Map queries filter on `r.data_sources IS NULL OR 'pgp' IN r.data_sources`, so the union must never drop `'pgp'`/`'pgp_website'`. |
| `data_sources` = NULL (HAS_TAG, JOINED_WITH, WRITTEN_IN, WRITTEN_AT, SENT_TO, 3,952 MENTIONS_PLACE, 1,242 ORIGINATED_FROM) | **Do not auto-merge** NULL with a non-NULL list; put the group on the review sheet. | NULL counts as the PGP tier in map queries. Turning it into `['enriched']` would hide the edge from the PGP map. 0 current candidate groups mix NULL with a list. |
| `source_books` (list<str>, 12,545) | **Union**, keeping order. | This is the existing pipeline's own behaviour: in 90 relationships with several books, `source == source_books[0]`. |
| `book_uuid` (str, 12,576) | **Keep the primary's value**, plus a new **`book_uuids`** list holding the union of all non-null values. | Differs in 7/10 WROTE collisions (6 because the PGP edge has none; S.W. Baron has 2 different uuids), in 10/197 COLLABORATED_WITH reverse pairs and in 7/176 RELATED_TO reverse pairs. The backend reads no relationship `book_uuid` today, so adding the list breaks nothing. |
| `page_uuid` (str, 9,812) | **Keep the primary's value**, plus a new **`page_uuids`** union list. | Differs in 37/197 COLLABORATED_WITH, 46/176 RELATED_TO and 7/10 WROTE groups. This is the ES page id, so never drop it. |
| `evidence` (str, 12,533) | **Keep the primary's**, plus an **`evidence_alt`** list holding the other distinct texts. | Differs in 69/197 COLLABORATED_WITH, 98/176 RELATED_TO, 16/29 MARRIED_TO and 9/10 WROTE groups. |
| `evidence_page` (int, 9,761) | **Keep the primary's**, so it stays with its own evidence/page_uuid. The other values go to the revert log only. | A list cannot stay aligned when some members are null. |
| `source` (str, 22,263) | **Keep the primary's** (first-write-wins, as the pipeline does today). | Differs in 7/10 WROTE groups (PGP edge has none). |
| `confidence` (str 'high'/'medium', 12,533) | **Max** (high > medium). | Differs in 22 COLLABORATED_WITH, 4 RELATED_TO and 2 MARRIED_TO reverse pairs. |
| `pipeline_version` (str, all "v3") | **Max**; log it if the values differ. | Never differs today. |
| `uncertain` (bool; PGP role edges, MENTIONS_PERSON) | **AND**: the merged edge is certain if any member is certain. | Applies only within an identity key. |
| `certain` (bool, MENTIONS_PLACE 261) | **OR**. | No parallels today. |
| `deceased` (bool, PGP) | **OR**. | |
| `relation` (str, PGP) / `role` (str, MENTIONS_PERSON) | Part of the identity key, so it is not merged. If the reviewer forces a merge, keep the certain edge's value and put the others in a `roles` list. | |
| `pgpid` (int, PGP) | Part of the identity key, so it is never dropped. If forced, use a `pgpids` union list. | |
| REFERENCES `has_transcription` / `has_translation` / `has_discussion` (bool) | **OR**. | No REFERENCES parallels today. Merging duplicate BookArticle nodes would create 123 to 978 REFERENCES collisions (`bookarticle_merge_collisions.json`). |
| REFERENCES `pages` | Part of the identity key: different pages are different citations, so they are kept separate. | `pages` differs in 95/123 of the would-be collisions. |
| REFERENCES `transcription_extent` / `translation_extent` | **Max** by completeness: Full > Partial > Unknown. | |
| REFERENCES `doc_relation` | **Union** of the `' ; '`-separated tokens (the existing format, e.g. "Translation ; Edition"). | |
| REFERENCES `raw_citation` / `location` / `notes` | **Keep the primary's**, plus `*_alt` lists for the other distinct values. | |
| HELD_AT `sub_collection` (str) | **Keep the primary's**, plus a `sub_collections` list. | No parallels today. |
| WRITTEN_IN `primary` (bool) | **OR**. | No parallels today. |
| No properties (HAS_TAG, JOINED_WITH) | Plain dedupe. | The 2 JOINED_WITH reverse pairs have no properties. |

Every deleted or rewritten edge goes to the per-change revert log as JSONL with: elementId, type,
start/end node key (Scholar/Person `name`, Fragment `canonical_shelfmark`, BookArticle
`article_id`) and the full property map. Nothing is lost even where only the primary scalar is kept.

## 4. Symmetric types: pick one direction

- Keep one edge per unordered pair. Its start node is the endpoint with the smaller
  (label, unique key: name / canonical_shelfmark). That choice is stable across dump and restore,
  unlike elementId.
- Merge the properties with section 3.
- Surplus removed: COLLABORATED_WITH 197, RELATED_TO 176, MARRIED_TO 29, JOINED_WITH 2, total 404.
- The backend reads COLLABORATED_WITH without direction (`neo4j_service.py:429`, `LIMIT 30`, no
  DISTINCT), so today each of the 168 Scholar<->Scholar pairs shows up twice in a scholar profile.
- `get_cross_institution_joins` reads JOINED_WITH with direction. After the change each pair is
  counted once, which only affects 2 pairs.

## 5. Self-loops

- Never create a self-loop. If a rewire would turn an edge into a loop, drop it and log it.
- The exact Scholar merge creates **0** self-loops.
- No Scholar<->Scholar edge links two nodes that share a surname and first initial, so judgement
  merges of that kind would also create none. All 29 edges of that shape are between Person nodes
  (fathers/sons, different gaonim), which confirms that surname+initial must never be used to merge.
- The 4 existing self-loops go on the review sheet with a proposed delete. They suggest two
  historical people were conflated into one node (e.g. "Judah ha-Levi").

## 6. Inverse-type and synonym pairs: keep them

- MENTIONED_IN (pgp_website) and MENTIONS_PERSON (merged/enriched) come from different sources and
  serve different tiers. Do not merge them (284 pairs).
- ORIGINATED_FROM + ORIGINATED_IN on the same pair (4) go on the review sheet.
- Different-type edges between the same nodes (3,772 node pairs, e.g. SENDER_OF + MENTIONS_PERSON)
  are separate facts and are out of scope.

## 7. Rewiring for the Scholar EXACT merge (no APOC)

The exact rule:
1. flip a name of the form "Surname, Given" to "Given Surname";
2. apply NFKD and drop combining marks;
3. casefold;
4. delete apostrophe-like marks (including ʿ, ʾ, geresh, gershayim);
5. delete the mojibake `?`;
6. turn other punctuation into spaces;
7. collapse whitespace.

This gives **144 clusters / 296 nodes / 152 surplus**.
- Breakdown: 107 need the comma flip, 33 are case/accent/punctuation only, and 4 match only after
  deleting `?`.
- Without the `?` deletion the rule gives 142 / 148.

For each cluster:
1. Pick the survivor.
2. For every edge on a non-survivor, MATCH the existing (type, survivor, other) edge.
   - If one exists, merge the properties (section 3) and delete the old edge.
   - Otherwise, CREATE a copy on the survivor with all properties and delete the old edge.
3. Delete the empty node.
4. Rename the survivor to the "Given Surname" form only **after** the other member has been
   deleted, because of the `scholar_name` uniqueness constraint.

Rewire volume:
- Survivor = the member with the highest degree: **409 edges** (386 outgoing: WROTE 366, ...;
  23 incoming: COLLABORATED_WITH 19, MENTIONS_PERSON 2, RELATED_TO 2).
- Survivor = the member already named "Given Surname": **691 edges** (683 outgoing, 8 incoming).
- In total the clusters touch 2,234 edges (1,472 outgoing WROTE; 136 incoming).
- Either way the result has the same **13 collisions** (WROTE 10, COLLABORATED_WITH 2,
  MENTIONS_PERSON 1), and these must go through the MERGE path.

## 8. How find_scholars ranks: which parallels inflate it

- `find_scholars` (`src/backend/neo4j_service.py:190`) uses
  `size([(s)-[:WROTE]->(b) | b]) AS works_count`. That counts **WROTE relationships**: no DISTINCT
  and no label filter.
- Today there are 0 parallel WROTE, so edge count equals the number of distinct BookArticle nodes for
  all 2,678 scholars with works.
- If the exact-cluster merge just CREATEs edges, 10 survivors get +1 each.
- Parallels do not explain the "Hathaway, Jane" pick. That node wins only because it has 1 WROTE
  edge and "Jane Hathaway" has 0.
- A bigger inflation comes from WROTE edges to duplicate BookArticle nodes: 334 scholars have works
  sharing a normalised title, with up to 1,269 surplus. This is an upper bound: many of these
  "duplicates" are journal-name titles or PGP pseudo-works such as "Unpublished editions".
- Recommendation: also change `works_count` to `COUNT { MATCH (s)-[:WROTE]->(b:BookArticle) RETURN DISTINCT b }`.
  That is a backend change and needs separate approval.
