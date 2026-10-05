# Handoff: `source_links` on every merged record (2026-10-05)

**For:** `historical-document-analysis` (the merge pipeline, `/Users/isaac1/Documents/historical-document-analysis`,
branch `master`). The serving repo (`terraform_gcp_search`) will only *render* the field.

**Why.** cairogenizah.ai is getting crawlable per-fragment pages (`/fragment/<doc_id>`, SEO plan step 2). Each page
and the existing document modal should carry "View on Cambridge Digital Library", "View on Princeton Geniza Project",
"View on KTIV" … links to the holding institution's own page for that fragment: attribution, and outbound links to
the institutions are a trust signal for search engines. Deriving the links at render time in two places drifts, so
they become an indexed field computed once at merge time, like everything else on the record.

**Non-negotiables.**
- Never emit a link to the Friedberg Genizah Project (hosts `genizah.org`, `jewishmanuscripts.org`, `fgp.genizah.org`).
  The public site must not name or link FJP. Assert this at the end of the rule (the preview script does).
- Do **not** verify links by fetching institutional sites in bulk. NLI, CUDL and others have blocked us before for bot
  traffic. The rules below are deterministic from fields already on disk; verification was done on 16 requests total
  (`recon_requests.log`). Spot-check a handful by hand in a browser, nothing more.

## What the recon found (read-only, v8 = 73,543 records)

Most links need no guessing. PGP's published export (`/Users/isaac1/Documents/pgp-metadata/data/fragments.csv`,
already loaded by `load_pgp()` in `merge_shelfmarks.py` and kept under `sources.pgp.fragment`) has `url` and
`iiif_url` columns that the pipeline currently ignores. They hold the holding-institution links for CUDL, Luna
Manchester, the Bodleian catalogue, Penn Colenda/OPenn, ONB, BL and NLI. Joined to v8 through the pipeline's own
`ShelfmarkNormalizer` + `institution_token`, 36,071 of 36,246 PGP records match.

For Cambridge records without a PGP url, a shelfmark→CUDL-id rule (`cudl_id_rule.py`) reproduces 19,690 of the
19,701 shelfmark↔id pairs in the old 2025 CUDL scrape
(`multimodal-document-analysis/src/datasets/raw_data/cairo_genizah/cambridge_university/*.json`), and 9,836 of 9,873
PGP pairs.

| Holding institution | Records | Link source | Linked | Notes |
|---|---|---|---|---|
| Cambridge (T-S, Or., Add., Mosseri, L-G) | 41,785 | PGP `url`/`iiif_url` (9,920) + CUDL rule (31,666) | 41,586 (99.5 %) | `https://cudl.lib.cam.ac.uk/view/<CUDL-ID>`. Fallback `https://cudl.lib.cam.ac.uk/search?keyword=<shelfmark>` |
| Princeton Geniza Project | 34,375 | `tei_metadata.pgpids` | 34,375 | `https://geniza.princeton.edu/en/documents/<pgpid>/`, one per pgpid |
| NLI / KTIV (any holder) | 12,207 at merge time (11,330 have a sysnum in v8) | `ktiv.sys_num` | 11,330 | **URL shape unverified**, see below |
| Manchester JRL | 10,426 | PGP luna url (2,076) + Luna search template from `A|B|C|L|P|G|AF|Ar. <n>` | 9,475 | Luna links are *search-result* pages, mark them `kind: "search"`. Gaster not covered |
| JTS (ENA …) | 10,856 | none deterministic (images live on Princeton's figgy) | 0 direct; 7,491 via PGP, 1,740 via KTIV | 1,767 with no link at all |
| Bodleian | 1,789 | `bodleian_catalogue_url` (1,156), PGP `genizah.bodleian…` (same set), one iiif uuid (55 recs, Arab. c 56) | 1,211 | rewrite `genizah.bodleian.ox.ac.uk` → `hebrew.bodleian.ox.ac.uk` (it 301s there) |
| Penn CAJS | 415 | PGP colenda/openn url (147) + `Halper N` → `https://openn.library.upenn.edu/Data/0002/html/h<N>.html` | 310 | h298 verified; other Halper numbers not checked |
| BL / ONB / NLI-held / Utah / KB | small | PGP `url` as given | 123 | skip `drive.google`, `kestenbaum` |
| AIU, Budapest, Strasbourg, Freer, HUC, Vienna | ~4,100 | none | 0 | most of the FJP-only remainder |

Totals: 67,295 records get ≥1 link (98,410 links); 52,705 get a holding-institution link; 6,248 get none, of which
5,201 exist only in FJP (AIU/BnF 2,026, JTS 928, Manchester 743, Budapest 344, Vienna 195, Bodleian 186, Cambridge
174, Strasbourg 126, NLI 104 …).

## Field schema

```json
"source_links": [
  {"source": "cudl",       "label": "View on Cambridge Digital Library", "url": "https://cudl.lib.cam.ac.uk/view/MS-TS-00008-J-00041-00003", "kind": "item"},
  {"source": "ktiv",       "label": "View on KTIV (National Library of Israel)", "url": "…", "kind": "item"},
  {"source": "pgp",        "label": "View on Princeton Geniza Project", "url": "https://geniza.princeton.edu/en/documents/3867/", "kind": "item"}
]
```

- `source` ∈ `cudl | bodleian | bodleian_digital | manchester | penn | nli_other | ktiv | pgp | other` (the label map can
  live in the pipeline; `app.py:354` in the serving repo has a host→label map to keep consistent with).
- `kind` ∈ `item | catalogue | search` (Luna = `search`, Bodleian volume catalogue = `catalogue`).
- Order: holding institution first, then KTIV, then PGP (one entry per pgpid).
- ES mapping: `{"source_links": {"type": "object", "enabled": false}}` (stored, not indexed; nothing searches it).
  v8's mapping has no top-level `dynamic`, so a partial update without this would auto-map it as text+keyword.

## Where it goes in the pipeline

`build_merged_record(cid, pgp, fjp, ktiv, …)` in `src/datasets/merging/merge_shelfmarks.py` already has everything
in scope: `pgp_frag` (= the `fragments.csv` row with `url`, `iiif_url`), `pgp_docs` (pgpids), `ktiv["sys_num"]`,
the display shelfmark, and the Bodleian catalogue url. Add a `source_links.py` module next to `institution_tokens.py`
with one pure function `build_source_links(display_shelfmark, holder_token, pgp_frag, pgpids, ktiv_sys_num,
bodleian_catalogue_url) -> list[dict]`, unit-tested against `sample_links.jsonl` (32 rows, 4 per source) and the
CUDL pairs from the scrape, and call it from `build_merged_record`. `preview_source_links.py` is the recon script that
produced the numbers above (it reads a full ES dump, so it is a reference, not pipeline code); `cudl_id_rule.py` is
the CUDL rule to port verbatim.

Rules, in order:

```
pu, pi = pgp_frag.url, pgp_frag.iiif_url
cudl_id = regex(r'cudl\.lib\.cam\.ac\.uk/+(?:view|iiif)/(MS-[A-Za-z0-9-]+)', pu + ' ' + pi)      # strip "/1" page suffix
          or (cudl_from_shelfmark(display_shelfmark) if holder is Cambridge else None)        # NOT from canonical_id
bodleian = bodleian_catalogue_url or pu.replace('genizah.bodleian.ox.ac.uk', 'hebrew.bodleian.ox.ac.uk') if '/catalog' in pu
bodleian_digital = 'https://digital.bodleian.ox.ac.uk/objects/<uuid>/' from iiif.bodleian.ox.ac.uk/iiif/manifest/<uuid> in pi
manchester = pu if luna/digitalcollections.manchester in pu else Luna search template from shelfmark (A|B|C|L|P|G|AF|Ar. <n>)
penn = pu if colenda/openn in pu else OPenn h<N> for "Halper N"
other = pu for bl.uk, onb.ac.at, nli.org.il, utah, kb.dk; never drive.google or kestenbaum
ktiv = itempage(sys_num)                      # shape below
pgp = one link per pgpid
drop any url whose host is/ends with genizah.org or jewishmanuscripts.org (genizah.bodleian.ox.ac.uk is fine)
```

CUDL rule details (`cudl_id_rule.py`): strip the "Cambridge CUL:" / "Cambridge University Library, Cambridge, England
Ms." / "Cambridge Lewis-Gibson:" prefixes; series prefix T-S→TS, Or.→OR, Add.→ADD, Moss./Mosseri→MOSSERI, L-G→LG;
return None for joins/ranges ("+", "–", ";", " and ", trailing words such as "Cambridge Lewis", "minute fragments");
tokens `\d+|[A-Za-z]+`; digits zfill(5); NS/AS/Ar/Misc → NS/AS/AR/MISC; `Ka` → KA; other alpha of ≤2 chars → one
letter each (8Ja2 → 00008-J-A-00002, NS 320.64a → …-00064-A); keep the "(1)" in T-S Ar.18(1).1 →
MS-TS-AR-00018-00001-00001; Mosseri: first token is the roman numeral uppercased; L-G: Ar/Bib/Lit/Talm/Misc/Glass →
ARABIC/BIBLE/LITURGY/TALMUD/MISC/GLASS and the roman volume → int zfill(5). Known exceptions: T-S Ar.30.184.N → …-P-000NN
(19 rows, PGP has them); third-level sub-items (8J4.3.1, Misc.28.79.1) link the parent.

## Open item: the KTIV URL shape

Two candidates; both returned Cloudflare 403 to curl *and* to a browser from this network (we appear to be blocked
at nli.org.il right now, see below). Someone must open one in a browser on another network and confirm which
resolves to the item page for sys_num `990051232320205171` (T-S 8J41.4):

1. `https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage?vid=MANUSCRIPTS&docId=PNX_MANUSCRIPTS990051232320205171` (the form PGP uses in its own `url` column, so probably right)
2. `https://www.nli.org.il/en/manuscripts/NNL_ALEPH990051232320205171/NLI`

Do not script this.

## Backfilling v8 (optional, so the site work does not wait for v9)

If v9 is not imminent, the same function can backfill the live index without a reindex: run `preview_source_links.py`
logic over the ES dump, `PUT genizah_merged_v8/_mapping` with the object mapping above, then `_bulk` partial updates
(`{"update": {"_id": …}} {"doc": {"source_links": […]}}`) in batches of 1,000 against `localhost:9200` on the MBP.
73k updates take a few minutes. Tell the serving side when it's done; the backend reads the field straight through.

## Serving-side work (terraform_gcp_search, separate task, after the field exists)

- `DocumentMetadata.source_links` + `_extract_metadata` mapping in `src/backend/search_service.py`, filtered through
  `_BLOCKED_BIBLIOGRAPHY_URL_HOSTS` as a second line of defence.
- Modal: a "View at the source" row under the shelfmark; fragment pages: the same row + JSON-LD `sameAs`.
- `DocumentMetadata.original_url` is never populated (the "View Original Source" button in `DocumentModel.jsx` never
  renders); replace it with `source_links`.

## Also found, needs a backend fix regardless (FJP leak in the public API)

`joins_data` is passed through raw (`search_service.py:718`). 52 records carry `joins_data.metadata.pageUrl` on
`fgp.genizah.org`, and `joins_data.source` holds strings such as "Yaacov Sussmann, Head of FGP …". The UI does not
render them, but both are in the public JSON of `/document/<id>`. Strip `metadata.pageUrl` and `source` from
`joins_data` in the backend (serving repo), and drop them at merge time too.

## Data quirks worth knowing

- `canonical_id` collapses "(n)": T-S F2(1).60 and F2(2).60 both become `Cambridge_CUL_T_S_F2_60` (258 Cambridge
  shelfmarks contain parentheses). The CUDL rule must therefore work from the display shelfmark, not the id.
- Three shelfmark styles coexist (PGP, "Cambridge CUL: …", KTIV "Cambridge University Library, Cambridge, England
  Ms. …") plus broken FJP fragments ("… Cambridge Lewis" tails, "Gibson: L-G …", "T- S AS", "TS Misc.", box-level
  "minute fragments").
- Wrong holders: "New York JTS: T-S AS 62.379" (6), 'Reinach' for Frankfurt, 'Cambridge' for Jerusalem NLI 577.5/16,
  "Paris Institut" canonicalised as `Paris_BNF`. `institution` is free text; group by the canonical_id prefix.
- `source_collection` has only two values (cambridge/princeton) and means pipeline origin, not holder.
- `www.nli.org.il/en/books/NNL_ALEPH…` URLs in `bibliography` (2,355) are book records, not fragments.

## Files in this folder

- `cudl_id_rule.py` — shelfmark → CUDL id (port verbatim, add tests).
- `preview_source_links.py` — the full recon rule set over an ES dump; reference for the pipeline function.
- `sample_links.jsonl` — 32 proposed records, 4 per link source, for unit tests and hand spot-checks.
- `recon_requests.log` — every external request the recon made (16, sequential, 5 s apart).
