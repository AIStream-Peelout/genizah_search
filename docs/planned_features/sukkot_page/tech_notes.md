# /sukkot: technical notes for the page and its own visualization

Task C of the Sukkot page handoff. Written 2026-09-25 (Sukkot 5787 begins at sundown today). Read-only work: code read
from `origin/prod-mbp` in the Studio clone, 21 public requests (17 API/site calls + 1 extra bundle fetch + 3 image
HEADs; log in `sukkot_page/api/_calls.log`), no repo edits. Content, meaning the fragments, card wording and scholarship,
is in the research drafts (`research/drafts/part1_sukkot_report.md`, `part2_e2e_test.md` §"The /sukkot page") and
in the Task A/B seed files. This file covers how to build the page.

---

## 0. Key decisions (TL;DR)

1. **Start from the MBP's checked-out tree, not `origin/prod-mbp`.** The live bundle already references
   `genizah_merged_v8` (the YK `SOURCE_INDEX` and the `KG_ES_INDEX` default), but `origin/prod-mbp` still says
   `genizah_merged_v7`. The MBP has unpushed or uncommitted frontend changes. Run `git status` / `git log` there first,
   and never rebuild from the remote branch, because that would roll the site back.
2. **Zero API calls on page load.** All page data goes in one precomputed JSON, imported by a **lazy-loaded** route
   component (`React.lazy`) so that it lands in its own hashed chunk. The only live calls are on click: the record modal
   (`/document`) and `/read`.
3. **Two layers of grouping.** Every fragment gets one **theme**, a genre bucket that drives the colours (7 of them),
   and zero or more **stories**, the six curated narrative clusters: hoshanot, the Mount of Olives assembly, the
   four species, the synagogue sukkah, Simhat Torah and the triennial cycle, and the 921/2 calendar dispute. Both are
   curated by hand. Embeddings position points on the exploratory map and never assign themes.
4. **Lead visual = a "festival days" strip**: a preparations slot, then 15 to 23 Tishri, with fragments attached to the
   day they concern. This
   is the one view in which undated liturgy has a natural place. Secondary views: a 2D semantic map of the Sukkot
   fragments (UMAP, computed offline), a timeline of the dated items that says plainly how many are undated, and a
   small places map. No new npm dependencies: hand-written SVG React components. Plotly (already in the bundle) is an
   acceptable fallback for the scatter only.
5. **Embeddings: yes, available publicly, 1024-d.** `GET /shelfmark/{doc_id}/documents?include_embeddings=true`
   returns the stored vector for a doc_id (tested on 3 ids: dim 1024, L2 norm 1.000, about 25 KB each). On the MBP,
   prefer a read-only ES `ids` query, or the public endpoint at a 0.3 s pace (~250 calls). Do not use the live
   `/visualization-explorer/calculate` endpoint (see §2.3).
6. **Oxford: store two ids per card.** Use `doc_id` (the v8 catalogue id, e.g. `Oxford_Bodleian_Bodl_MS_heb_b_3_5`)
   for the record, and `ai_read_doc_id` (the pre-v8 FJP-style id, e.g. `Oxford_Bodleian_MS_heb_b_3_5`) for `/read`.
   The image index differs between the two records as well (§1.6).
7. **Per-route link preview via nginx `sub_filter`, as `/yom-kippur` already does (verified live).** Add a
   `location ~ ^/sukkot(/.*)?$` block and an `og-sukkot.jpg`.
8. **The JSON is public.** Everything in it ships to every visitor. Leave out Friedberg/FJP/FJMS names, hidden AI text,
   Claude-derived leads and internal notes; put those in a build report that is not shipped.

---

## 1. Code state: what is where

### 1.1 Branches (Studio clone; remote refs as of this session)

| ref | head | frontend relevant |
|---|---|---|
| `origin/prod-mbp` (deployed build source) | `e765768` 2026-09-22 "Add bibliography work lookup and UI" | YK page, `/read` mobile fixes, per-route OG, About, BibliographyDetail |
| `origin/yk-holiday` | `a3aaed7` 2026-09-22 | its only commit not on prod-mbp touches backend/evals/docs, **no frontend**. `d12af99` "YK holiday frags" (YK page + `/read` mobile) is already in prod-mbp via merge `e5359f8` |
| `origin/master` = `origin/sukk` = local `sukk` | `fd04a38` 2026-09-14 | **no** YomKippur.jsx, no `/read` viewer changes, no OG work. Do not build the page from master |
| **live bundle** (`https://cairogenizah.ai/static/js/main.f5edd9bb.js`, 7.2 MB, one chunk for all routes) | newer than `origin/prod-mbp` | contains `wu="genizah_merged_v8"` (YK SOURCE_INDEX) and `REACT_APP_KG_ES_INDEX\|\|"genizah_merged_v8"`; `origin/prod-mbp` has v7 in both places |

So YomKippur.jsx, the `/yom-kippur` + `/yk` routes and the `/read` mobile fixes (collapsed beta banner under 600 px,
open on the first confirmed line at page width, two-pointer pinch, no nested scroll trap under 900 px, row numbers
hidden at low zoom) are on prod-mbp **and live**: the bundle contains "Pinch to zoom, drag to pan." and the
one-line banner. They are not only on yk-holiday. The handoff note is
`docs/planned_features/2026-09-17_yom_kippur_viewer_handoff.md`; yk-holiday's copy has 42 more lines on the catalogue
metadata gap and the v7 data-side changes.

### 1.2 Router and page wiring

- Routes are in `src/frontend/src/react_app.jsx`, in `AppContent()` (around l.1654 on prod-mbp): `/`, `/explorer`, `/chat`,
  `/faq`, `/about`, `/read`, `/yom-kippur`, `/yk → <Navigate to="/yom-kippur">`, `/map`. All components are imported
  eagerly at the top of the file, and nothing uses `React.lazy`.
- `DocumentModal` is mounted once at App level. `MapView` receives `onOpenEsDocument={handleOpenEsDocument}`, which
  fetches `/document/{id}?index_name=KG_ES_INDEX` and opens the full record modal (images, transcription, bibliography
  and the bibliography "work" popup). **Reuse this for Sukkot cards**, e.g.
  `<Route path="/sukkot/*" element={<Suspense …><Sukkot onOpenEsDocument={handleOpenEsDocument}/></Suspense>}/>`.
  That gives every card a "catalogue record" button without permalinks.
- There is no record permalink route, so the modal is the record view.

### 1.3 How /yom-kippur loads data (the pattern to avoid)

`YomKippur.jsx` hard-codes 13 doc_ids and on mount runs `Promise.all` over `/ai-transcriptions/{id}` (13 calls). Each
`FragmentCard` then fetches `/document/{id}?index_name=SOURCE_INDEX` (13 more): 26 calls for static content. It sorts by
agreed share, uses the full-resolution `item.image_url` as the thumbnail, and links to
`/read?doc=…&image=…&index=SOURCE_INDEX`. `SOURCE_INDEX` is a constant that has had to be edited v6 → v7 → v8,
each time with a rebuild. Copy text comes from `read/caveat.js` (`AGREEMENT_CAVEAT`, `READERS_DESCRIPTION`); reuse those
for the machine-read caveat.

### 1.4 CSS approach

The YK page uses a plain CSS file with a class prefix (`YomKippur.css`, `.yk-*`, max-width 860 px, a card grid of 140 px
thumb + body, and one custom property `--agreed-bg`). The app shell uses inline `<style>` blocks in react_app.jsx.
Tailwind is in devDependencies but not wired in (index.css is plain). Recommendation: `src/frontend/src/sukkot/Sukkot.css`
with an `.sk-` prefix, the theme colours as CSS custom properties on `.sk-page`, and the same 16 px mobile gutter as YK.

### 1.5 Where static data can live, and the build

- Recommended: `src/frontend/src/sukkot/data/sukkot_5787.json`, imported by the lazy page component. CRA puts it
  in the page's own hashed chunk, which nginx caches as immutable (`location /static/`) while `index.html` stays
  no-cache. It does not add to the 7.2 MB main bundle and needs no cache-busting.
- Alternative: `src/frontend/public/data/sukkot_5787.v1.json`, fetched at runtime. nginx serves it through
  `try_files $uri` and gzips `application/json`. Put a version in the filename. Use this if the owner also wants the
  dataset to be openly downloadable.
- Thumbnails: `src/frontend/public/sukkot/thumbs/<doc_id>.jpg`, about 320 px wide at JPEG q≈70 (20 to 30 KB each).
  Do not reuse the full-size images. Measured by HEAD: a KTIV image is 2,758,074 bytes, a Bodleian one 1,473,178, an
  FJP one 441,888 (`storage.googleapis.com`, `cache-control: public, max-age=3600`, no CORS headers, so `<img>` works
  but a `fetch()` of files in that bucket would not).
- Build: `src/frontend/Dockerfile`, `node:18-alpine`, `npm install --omit=dev` **without a lockfile**,
  `DISABLE_ESLINT_PLUGIN=true`, `npm run build`, then `nginx:alpine` with `src/frontend/nginx.conf`. Compose
  (`docker-compose.yml` + `docker-compose.mbp.yml`) builds `frontend` with `REACT_APP_API_URL=https://api.cairogenizah.ai`
  and the chat/CARTO keys as build args. **A frontend rebuild or recreate on the MBP replaces the prod site and needs the
  owner's go-ahead.** For previews, build a scratch copy in a `node:22-alpine` container, which also needs asking first
  (there is no Node on the host). Because the page makes no API calls on load, a scratch build served by any static server
  shows it completely. Only the modal and `/read` clicks hit the prod API.
- Adding npm packages is risky: without a lockfile, each build resolves versions afresh (the Dockerfile comment on the
  eslint-plugin-jest breakage is the precedent). The page needs no new package.

### 1.6 The Oxford id problem, verified live

| call | result |
|---|---|
| `/document/Oxford_Bodleian_Bodl_MS_heb_b_3_5` (v8 default, via /shelfmark endpoint) | 200; shelfmark "Bodl. MS heb. b 3/5", **3 images**: `BODLEIAN/…/MS_HEB_b_3_5a.jpg`, `…b.jpg`, `images/9318_image.jpg` |
| `/ai-transcriptions/Oxford_Bodleian_Bodl_MS_heb_b_3_5` | `available: false` |
| `/ai-transcriptions/Oxford_Bodleian_MS_heb_b_3_5` (old FJP-style id) | `available: true`, image_index 0 = `images/9318_image.jpg`, 6 of 18 agreed, `qwen3-vl-8b-heb-v21b-step1200` |
| `/document/Oxford_Bodleian_MS_heb_b_3_5` (v8) | **404** |
| `/document/Oxford_Bodleian_MS_heb_b_3_5?index_name=genizah_merged_v7` | 200 (v7 still exists: `/indices` lists v8 default 132,393 docs, v7 128,019) |
| `/document/Oxford_Bodleian_Bodl_MS_heb_c_28_28`, `…_Bodl_MS_heb_d_76_30` | 200, 200; old KTIV-style `Oxford_Bodleian_heb_d_76_30` 404 |

Consequences for the page:
- The id rule holds on 3 of 3 checks (`Oxford_Bodleian_MS_heb_X` or `Oxford_Bodleian_heb_X` → `Oxford_Bodleian_Bodl_MS_heb_X`).
  The builder should still confirm every mapped id with a `/document` 200 (or an ES `ids` hit) and use the v8 build's
  merge map where one exists on the MBP.
- The AI read's `image_index` refers to the **old** record's image list: image 0 in the AI index is image 2 of the
  merged v8 record. Store `machine_read.image_url` and never map AI image indices onto v8 image lists.
- `/read` works with the old id. The viewer's `/document` call is best-effort, so without `index` the header shows no
  catalogue metadata, and with `index=genizah_merged_v7` it does, until v7 is deleted. Link
  `/read?doc=<ai_read_doc_id>&image=<n>` and add `&index=genizah_merged_v7` only while v7 exists (the builder can check).
- Seed list correction: part2's seed #20 says "use `Oxford_Bodleian_MS_heb_b_3_5`, not the Bodl duplicate". Under v8 that
  is reversed for the catalogue. Use `Oxford_Bodleian_Bodl_MS_heb_b_3_5` as `doc_id` and the old id only as `ai_read_doc_id`.
  Other Oxford items in the Sukkot material: c 28/28, d 36/14, e 39/1, a 2/24, d 76/30, f 26/3 (Ben Meir), f 33/24.
- Shelfmark labels in v8 are still inconsistent: "Bodl. MS heb. b 3/5" against "The Bodleian Libraries, University of
  Oxford, Oxford, England Ms. heb. d. 76.30", and the new KTIV Cambridge record `Cambridge_CUL_T_S_10H4_5` reads
  "Cambridge University Library, Cambridge, England Ms. T-S 10 H 4.5". The builder must normalise them (§3.6).
- Durable fix (the owner's call, a data write): re-key Oxford reads in a `genizah_ai_transcriptions_v3`, or add an
  id alias map to the backend's AI service.

---

## 2. Visualization building blocks already on the site

### 2.1 Frontend
- `VisualizationExplorer.jsx` (2,394 lines, route `/explorer`) calls `/indices`, `/visualization-explorer`, `/calculate`,
  `/similarities`, `/project-query`, `/full-index` and `/document`. It uses Plotly `scatter`, switching to `scattergl`
  above a point threshold. It shows a public "Debug Information" block and an index dropdown listing internal indices.
  Project-query draws a star without a nearest-documents list.
- `TSNEVisualization.jsx` (662 lines): the search page's PCA/t-SNE toggle, `react-plotly.js` scatter.
- `MapView.jsx` (1,641 lines): `react-leaflet` with CARTO raster tiles (`REACT_APP_CARTO_API_KEY`, baked at build) and
  `/map/*` KG endpoints.
- `package.json` on prod-mbp: react 18, react-router-dom 6, react-scripts 5.0.1, `plotly.js-basic-dist`, `react-plotly.js`,
  `mirador` 4, `leaflet` + `react-leaflet` 4. **No d3, no recharts, no visx.** `import Plot from 'react-plotly.js'` pulls
  in the full `plotly.js` peer: the live bundle contains `scattergl`, which the basic dist lacks. Plotly, Mirador and
  Leaflet are therefore all in the single 7.2 MB main.js already.

### 2.2 API endpoints (live `/openapi.json`, byte-identical to the 2026-09-22 copy)

| endpoint | what it does | use for /sukkot? |
|---|---|---|
| `POST /visualization-explorer` | random sample (10 to 10,000, or `load_full_index`) of docs with embeddings; `compute_method` tsne/umap server-side; returns a `sample_id` (server cache, 1 h TTL) | no (random, not a doc list) |
| `POST /visualization-explorer/calculate` | PCA/t-SNE/UMAP on posted `embeddings` or a `sample_id` | **no**: it runs on the prod backend CPU and **overwrites the process-global `_fitted_models[method]`** that `/project-query` uses for every visitor (`visualization_service.py`) |
| `POST /visualization-explorer/similarities` | cosine matrix for ≤50 indices of a cached sample | no (sample-bound) |
| `GET /visualization-explorer/full-index` | precomputed t-SNE + UMAP for the whole index: `full_index_visualization_genizah_merged_v8.json` exists (Last-Modified 2026-09-24 15:24 GMT; the body was not downloaded) | optional "Sukkot within the whole Genizah" background (§3.3) |
| `POST /visualization-explorer/project-query` | embeds a query and projects it into the last fitted space | no |
| `GET /shelfmark/{shelfmark}/documents?include_embeddings=true` (default **true**) | ES `should` including `term doc_id`, so **a doc_id works as the path segment**; returns `metadata` + `embedding` | **yes**: the embedding fetch for a doc list |
| `POST /search`, `/search-hybrid`, `/search-shelfmark` with `include_embeddings` | vectors for search hits (`num_results` ≤ 20 or 50) | no (query-driven, not id-driven) |
| `GET /document/{id}` | full metadata, **no embedding** | builder validation + modal |
| `GET /ai-transcriptions/{id}` | per-image read summary (`n_agreed`, `n_lines`, `image_url`, model) | builder only (store the counts in the JSON) |

### 2.3 Embedding test (3 doc_ids, the budget cap)

`GET /shelfmark/{id}/documents?include_embeddings=true`: count 1 each, dim **1024**, norm **1.0000**, 24–29 KB per
response, ~0.5 s. Stored contract: `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614…`, document mode = raw text, L2-normalised.

| pair | cosine |
|---|---|
| T-S NS 159.124 (Saadya hoshanot) vs T-S Misc.35.11 (1029 Mount of Olives letter) | 0.415 |
| T-S NS 159.124 vs Bodl. MS heb. b 3/5 (Damietta 1157, broke the festival) | 0.408 |
| T-S Misc.35.11 vs Bodl. MS heb. b 3/5 | 0.564 |

For reference, unrelated document pairs have a median cosine of about 0.45 under this contract (embedding-contract note,
2026-07-24). Two Sukkot items of different genres are therefore no closer than random pairs, while the two documentary
items sit together. **The vector encodes the catalogue text of the record, so genre and language dominate over
festival.** T-S NS 159.124's public description is "כתאב פי וג'וב אלצלוה; מקרא ותרגומים; פיוט; תפילה וברכות", which has
no Sukkot word at all. That is the case for about 41% of the 410 catalogued Sukkot items, whose Sukkot identification
sits in no public field. A generalist embedder cannot cluster them by festival. Curated themes come first, and the
embedding map is an exploratory layout only (§3.3).

**Getting vectors for ~250 ids on the MBP** (pick one and ask the owner):
- ES read-only against the local serving ES: one `search` with `{"ids":{"values":[…]}}`, `_source: ["doc_id","embedding_vector"]`,
  size 300 (~7 MB). This needs the local ES credentials, so it is the owner's call.
- Public API: ~250 × `/shelfmark/{id}/documents?include_embeddings=true`, sequential with 0.3 s sleeps (~3 min, ~7 MB),
  cached to disk. Send a browser-like User-Agent: Cloudflare returns 1010 to Python-urllib's default.
- Do not unpickle `data/visualization/embeddings_cache_genizah_merged_v8.pkl` on the MBP, if it exists: that is about
  132k × 1024 Python floats, several GB of RAM on the prod box.

---

## 3. Proposed design: "Sukkot in the Cairo Genizah" (`/sukkot`)

### 3.1 Page structure

```
/sukkot                      hero + festival-days strip + story tiles + "explore" tabs (Map | Timeline | Places | All fragments)
/sukkot/:story               story page: narrative intro (verified scholarship, inline refs) + its cards (+ contested callout)
/sukkot?f=<doc_id>           deep link that opens one card (the side panel on desktop, a bottom sheet on phones)
```

Short link `/sk` → `/sukkot` if wanted (same `<Navigate>` pattern as `/yk`). Keep it a **permanent** festival page:
put "Sukkot 5787" in the data, not in the route.

**Card** (one component for every view): thumbnail · normalised shelfmark · `card_line` + source badge (KTIV · PGP ·
OPenn · edition · published study · "catalogue note") · date (with "computed" marked where it applies) · place chips ·
text status (exactly one of "Human transcription (PGP / edition)" · "Machine reading (beta): N of M lines agreed by a
second machine reader" (only if N > 0) · "No transcription yet") · buttons: **Catalogue record** (`onOpenEsDocument(doc_id)`),
**Read the machine text on the manuscript** (only if N > 0; `/read?doc=<ai_read_doc_id>&image=<n>`) · one bibliography
line where the record carries scholarship · "Image courtesy of <holding library>". Hebrew and Judaeo-Arabic go in
`<span dir="rtl" lang="he">`.

### 3.2 Views

**(e) Festival-days strip: the lead visual (curated, no computation).** x = before the festival (11–14 Tishri,
preparations) · 15 · 16 · 17–20 (hol ha-moʿed) · 21 Hoshana Rabbah / "Day of the Willow" · 22 Shemini Atzeret · 23 Simhat
Torah (diaspora; one day with 22 in the Land of Israel). Chips are placed by the day a fragment concerns. Examples:
preparations (T-S 8J28.3, the Sukkot quires asked back the night after Yom Kippur; T-S NS J221 recto, palm-branch
transport on 13 Tishri 1219; T-S 10J12.20, a letter from the eve of Sukkot); Musaf for days 6–7 (T-S AS 109.184); hoshanot
for days 6 and 7 (JRL B 2562) and "for the seventh day" (T-S 8J17.24); Halper 251 (21–23 Tishri); Shemini Atzeret prayers
(ENA 2114.1, T-S B13.16); the 1029 ban (day 7); Damietta 1157 (day 2); the Alexandria bread riot "on the second day"
(T-S 12.305); the 1219 accounts ("יום ערבה" entry, day 7); the 1064 death on Simhat Torah night; the 1844 accounts dated
Hoshana Rabbah (JRL C 127). The **921 proclamation is drawn as one contested bracket spanning day 2
and day 7**, labelled "Stern 2019: second day · Laufer 2025 and earlier scholars: Hoshana Rabbah". This view gives
undated liturgy a place without implying a year.

**(a) Semantic map (exploratory).** 2D points for about 100 to 250 Sukkot fragments, filled by **theme** colour (Okabe–Ito
palette, shape as a second channel for colour-blind users). A story selection outlines its members and dims the rest.
Hover or focus shows the shelfmark and card line; click or Enter opens the card. Caption, required: "Position reflects
similarity of catalogue wording as seen by a general-purpose text model, not a scholarly classification; colours are the
curated themes." Optional toggle **"In the whole Genizah"**: take each fragment's coordinates from the existing
`full_index_visualization_genizah_merged_v8.json` (UMAP over all 132k records; on the MBP host at
`<repo>/data/visualization/`, bind-mounted into the backend as `/app/src/backend/data`) and draw a fixed random 3 to 5k
grey sample underneath. That view shows honestly that Sukkot material is scattered through the collection by genre.

**(b) Timeline, dated items only.** Two panels: 870–1300 (dense) and 1300–1900 (sparse). A lane per precision (day ·
year · circa/range), with an **undated bar** that states the count, e.g. "M of N fragments on this page carry a date;
liturgy and law are almost never dated". Dates come from human sources (PGP, colophon, edition). CE conversions marked
`computed` are ours (Julian before 1582) and are labelled as such. Known conflicts are shown, not hidden: T-S NS J221
verso (PGP 29 Sep vs computed 2 Oct 1219); Freer F 43 (editors 8 Oct 1043 vs computed 8 Oct 1042; no image on the site);
T-S 13J8.10 (catalogue "5304 (= 1544 CE)"; Tishri 5304 falls in autumn 1543); Laufer's 27 Sep 921 against the computed
26 Sep. Palaeographic date ranges could form a separate hatched lane only if the builder has them from KTIV scrape data;
`/document` does not expose them.

**(c) Places map: static SVG, no tiles.** Use an equirectangular projection over lon 8–48 E, lat 10–42 N, with a
land path pre-projected by the builder from Natural Earth 1:50m (public domain; downloading it needs owner OK) and
shipped in the JSON or as `public/sukkot/basemap.svg` (about 30–80 KB). Jerusalem and the Mount of Olives share one marker
with a sub-list, or an inset. Each place lists its fragments, with a role (written at · event at · mentions) and a
source. Leaflet + CARTO reuse is possible (it is already bundled and the key is baked in), but it loads tiles on phones
for about a dozen dots. Approximate coordinates, to be checked against `/map/places` or Wikidata before publishing:

| place | lat, lon (approx.) | anchor fragments (human source) |
|---|---|---|
| Fustat | 30.006, 31.232 | synagogue sukkah (T-S K25.190, Ar.18(1).155, 18J4.12), accounts (NS J221, Misc.8.61), most letters |
| Jerusalem · Mount of Olives | 31.777, 35.235 · 31.778, 35.244 | T-S Misc.35.11, 13J19.16, Or.1080 J105/J45 (Gil eds.), 10J9.25, NS 320.42 |
| Ramla | 31.927, 34.867 | Or.1080 J45 (pilgrims at Natan's majlis, Gil); Karaites of Ramla (13J19.16) |
| Tyre · Haifa | 33.271, 35.196 · 32.815, 34.989 | Megillat Evyatar (T-S 10K7.1): assemblies re-staged at Tyre 1081 and Haifa 1082 |
| Damietta ("Isle of Caphtor") | 31.417, 31.814 | Bodl. MS heb. b 3/5 (1157, PGP) |
| Alexandria | 31.200, 29.919 | T-S 12.305 (Day of ʿArava; bread riot); ENA 2739.8 + NS 19.10 (1200) |
| Aden (+ Dahlak, inferred) | 12.785, 45.019 | T-S 12.355, T-S Misc.28.256 (1140; no image), T-S 12.392 (Dahlak/Aden is an FOTM inference) |
| Qayrawan | 35.678, 10.096 | Yevr. III B 904 (c. 1020; **no image on the site**) |
| Salonika | 40.640, 22.944 | ENA NS 5.29 (c. 1578). **The Sukkot link rests only on the public machine read, and that read shows CJK characters ("冒险").** Leave it out of v1, or show it labelled as machine reading once the artefact is fixed |
| Damsis, al-Mahalla, Dammuh | look up | T-S 10J12.20 (PGP); Bodl. c 28/28 (the Sukkot greeting is only in a hidden read, so it cannot be shown); T-S 16.222 (plantation "prob. near Dammuh", Gil) |

**(d) Story pages (the six narrative clusters).** A 150–300-word intro drawn only from `research/verify_scholarship.md`
(C1–C18, refute-first confirmed) and `research/ext_scholarship.md`, plus a list of cards. A contested point uses a
`<Contested>` callout that sets out both positions with citations.

| story | anchors (all human-sourced) | verified basis | contested or caveat |
|---|---|---|---|
| Hoshanot and the Day of the Willow | T-S NS 159.124, H18.17, JRL B 2562, Misc.22.186, NS 159.146 (Saadya's Siddur per KTIV/catalogue); T-S 24.68 (10th-c. ketubba, hoshanot on verso); T-S 8J17.24; Halper 251; L-G Lit. I.33; T-S A17.5 margin | C2, C16, new fact 3 ("יום ערבה" in K2.8, Halper 251, NS 320.42, NS J221) | how many Saadya copies these represent is an open question |
| The assembly on the Mount of Olives | T-S Misc.35.11, 13J19.16, Mosseri VII.142, Or.1080 J105 + J45 (Gil's editions), 10J9.25, NS 320.42, 10K7.1 | C5; Laufer 2025 after Frenkel and Ben-Sasson; Rustow on the Karaite ban | Freer F 43 date (C-new 2); the J105/J45 hidden reads must not be shown (Gil's editions exist) |
| Palm branches, myrtle and citrons | T-S 16.222, NS J221, Misc.8.61, 8J27.12, AS 153.287, 12.546 (1735, "it seems … etrogim", PGP), Halper 258, AIU IV.B.46 | C12, C13 | T-S 16.222's Sukkot purpose is PGP's inference ("apparently"); PGP's 29 Sep 1219 is wrong |
| The sukkah in the synagogue courtyard | T-S K25.190, Ar.18(1).155, 10J14.30, 18J4.12, 18J2.1 | C10 (Goitein, Med. Soc. IV p. 77) | Khan's T-S Ar.38.117 (a miẓalla in a house courtyard) is an open question, not a refutation |
| Simhat Torah and the triennial cycle | Seder Fustat A (T-S H12.11 + joins; H12.11 **not in our catalogue**), NS J221 verso ("אלחזן ר ידותון"), 10J5.10 + 10J11.13 (1064), Halper 251 | C1, C18 | Yaari's thesis is known only from a secondary report; Elizur's counterexample; the 1171 colophon's weekday (computed Monday) |
| The calendar dispute of 921/2 | T-S NS 194.92 + ENA 2555.1 (BCC 3), Bodl. f 26/3 + T-S 8K7, ENA 2556.2, T-S K2.8 (al-ʿAqulī), NS 262.17 + AS 155.112 (Karaite account) | C7, C8 | **Stern 2019 (second day, 16 Tishri = Fri 21 Sep 921) against Laufer 2025 and Epstein, Malter, Bornstein and Fleischer (Hoshana Rabbah; Laufer "27 Sep 921", computed 26 Sep).** Gil is cited on both sides (Stern counts him for the second day; Be-malkhut Yishmael p. 218 says "בהושענה רבה"), so do not assign him |

**Themes (colour; one per fragment).** Map the `category`/`subcategory` of `research/hidden_catalogue.json` and the
KTIV queue `tier_reason` labels onto seven buckets:

| theme | from hidden_catalogue subcategory | from KTIV queue label |
|---|---|---|
| Hoshanot | hoshanot | hoshanot/Hoshana Rabbah liturgy |
| Festival prayer and piyyut | sukkot liturgy/piyyut, shemini atzeret | Sukkot prayer (Amidah/Musaf/Kiddush/order), Sukkot piyyut, Shemini Atzeret liturgy |
| Simhat Torah | simhat torah | Simhat Torah liturgy |
| Bible, haftarot and targum | haftarot for sukkot, bible (other), targum | Sukkot haftarot/Torah reading |
| Law and learning | talmud/mishnah sukkah, rif/halakhot gedolot, mishneh torah, responsa, other halakha, talmud commentary | Mishnah/Talmud Sukkah, Sukkot halakha, homily/exegesis/treatise, Rif, Mishneh Torah |
| Calendar | calendar | calendar listing Sukkot |
| Letters and documents | letter, legal, other (accounts) | documentary text mentioning Sukkot |

Karaite material (3 to 5 items) gets a tag, not a colour. "Incidental" and "false-positive" rows never appear.

### 3.3 Generating the JSON (run once; re-run for updates)

Proposed `scripts/sukkot_page/build_sukkot_data.py`, following CLAUDE.md: Sphinx docstrings, type hints, small testable
functions, few try/except blocks, and a unit test for the validators.

1. **Curation file (the source of truth for all public text):** `scripts/sukkot_page/sukkot_curation.yaml`, one entry per
   fragment: `doc_id`, `theme`, `stories[]`, `festival_day`, `card_line`, `card_line_source` (public label + internal
   ref), `bibliography_line`, `date{display, year_ce, precision, source, computed}`, `places[{id, role, source}]`,
   `image_index_for_thumb`, `machine_read_note`, `include: v1|v2|hold`. Seed it from part2's 20-fragment seed and
   reserve list, part1 Appendix A (90 items), the Task A/B seed, `hidden_catalogue.json` (`matched_text` = catalogue
   wording; `relevance` core or mention, with images: 345 rows), and the KTIV queue rows now in v8 (e.g.
   `Cambridge_CUL_T_S_10H4_5` is live, but its public description is only "פיוט", so its card line must come from the
   KTIV title). **A human reads every card_line before publication.**
2. **Resolve ids** against the default index: `/document` 200 for each id, or one read-only ES `ids` query. Apply the
   Oxford rule (§1.6) and fail the build on any 404.
3. **Record facts:** normalised shelfmark, image URLs, holding institution (for the courtesy line), whether a human
   transcription or translation exists, and `bibliography_entries` (source-tagged since v7) for the one-line citation.
4. **Machine reads:** `/ai-transcriptions/{ai_read_doc_id}` (the old id for Oxford). Keep the image with the most
   agreed lines, as YK's `bestImage` does. Store `n_agreed`, `n_lines`, `image_index`, `image_url` and `model`. **Drop
   the read link when `n_agreed == 0`.** Never include hidden-read text. Attach curated caveats, e.g. T-S 13J8.10 "an
   agreed line gives the year 5314; the catalogue gives 5304", AIU IV.B.46 "נפראו for נפרצו", T-S 18J4.1 Arabic-script
   characters on an agreed line.
5. **Embeddings → layout:** fetch the vectors (§2.3), cache them to disk, then run UMAP (`metric=cosine`, `n_neighbors≈12`,
   `min_dist≈0.15`, `random_state=42`) with PCA as a fallback, scaled to [-1, 1]. Use the repo `.venv`: `umap-learn` and
   `scikit-learn` are backend requirements. Ask before creating any container. Optionally add the global coordinates
   from `full_index_visualization_genizah_merged_v8.json` (a one-time JSON read; ~132k entries).
6. **Diagnostics (report only, not shipped):** 10-NN theme purity and silhouette of the curated themes in embedding
   space; for each fragment, the share of its 10 nearest neighbours **in the whole index** that are Sukkot items; and
   the items whose neighbours all belong to other themes. This makes the "embedder blurs holidays" point with numbers,
   and it flags cards a human might want to re-check.
7. **Thumbnails:** download the chosen image once and resize it to 320 px wide, written to
   `public/sukkot/thumbs/`. The builder only reads from the bucket, but it is a file download, so ask the owner first.
8. **Basemap:** pre-project the Natural Earth land polygons clipped to the bbox into one SVG path string.
9. **Validate (fail the build on any hit):** no `Friedberg|FJMS|FJP|Genazim` in public strings; no "checked"/"confirmed"
   wording when `n_agreed == 0`; every `card_line` has a human `card_line_source`; every `date.year_ce` has a `source`
   and a `computed` flag; no text taken from Claude image readings (the curation file marks provenance); every theme and
   story id resolves; JSON ≤ 300 KB raw.
10. **Write** `src/frontend/src/sukkot/data/sukkot_5787.json` plus `scripts/sukkot_page/out/build_report.json`, and
    (optionally) the raw embedding cache in `scripts/sukkot_page/out/` (gitignored).

API budget if the public route is used: about 3 calls per fragment (`/document`, `/ai-transcriptions`, embedding),
roughly 750 calls at 0.3 s for 250 fragments, run once and cached.

### 3.4 JSON shape (public)

```json
{
  "schema": "sukkot-page/1",
  "edition": "Sukkot 5787 (from sundown 25 Sep to 4 Oct 2026)",
  "generated_at": "…",
  "catalogue_index": "genizah_merged_v8",
  "ai_index": "genizah_ai_transcriptions_v2",
  "layout": {"method": "umap", "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "n": 187, "params": {"n_neighbors": 12, "min_dist": 0.15, "metric": "cosine", "random_state": 42}},
  "themes":  [{"id": "hoshanot", "label": "Hoshanot", "color_var": "--sk-t1", "order": 1}],
  "stories": [{"id": "calendar-921", "title": "The calendar dispute of 921/2", "intro": ["para…"], "refs": ["stern2019", "laufer2025"], "contested": [{"claim": "Day of the proclamation", "positions": [{"view": "Second day of Sukkot (16 Tishri)", "refs": ["stern2019"]}, {"view": "Hoshana Rabbah", "refs": ["laufer2025"]}]}]}],
  "fragments": [{
    "doc_id": "Oxford_Bodleian_Bodl_MS_heb_b_3_5",
    "ai_read_doc_id": "Oxford_Bodleian_MS_heb_b_3_5",
    "shelfmark": "Bodl. MS heb. b 3/5",
    "holding": "Bodleian Libraries, University of Oxford",
    "theme": "documents", "stories": [], "festival_day": 16, "tags": [],
    "card_line": "Damietta 1157: testimony about a man who arrived on the eve of Sukkot and sailed on the second day, breaking the festival",
    "card_line_source": "Princeton Geniza Project",
    "bibliography_line": "…",
    "date": {"display": "20 Tishri 1469 Sel. (26 Sep 1157)", "year_ce": 1157, "precision": "day", "source": "PGP", "computed": false},
    "places": [{"id": "damietta", "role": "written"}],
    "thumb": "/sukkot/thumbs/Oxford_Bodleian_Bodl_MS_heb_b_3_5.jpg",
    "text_status": "machine",
    "machine_read": {"n_agreed": 6, "n_lines": 18, "image_index": 0, "href": "/read?doc=Oxford_Bodleian_MS_heb_b_3_5&image=0", "note": null},
    "xy": [0.12, -0.41], "xy_global": null
  }],
  "places": [{"id": "damietta", "name": "Damietta", "alt": ["Isle of Caphtor"], "lat": 31.417, "lon": 31.814}],
  "refs": [{"id": "stern2019", "citation": "Stern, …"}],
  "basemap": {"viewBox": "0 0 800 640", "land": "M…"}
}
```

At about 250 fragments this is ~200 KB raw and ~50 KB gzipped. No vectors are shipped.

### 3.5 Libraries

Use no new dependencies. React + SVG for the strip, scatter, timeline and places map; each view is under ~200 lines
and handles fewer than 300 marks. This gives full control over touch targets (≥ 32 px hit areas on phones), keyboard
focus, RTL labels, `prefers-reduced-motion` and a single stylesheet. Plotly is already in the bundle (full plotly.js via
the react-plotly.js peer), so it adds no bytes and is fine for the semantic map if time is short, but hover-driven
tooltips are poor on touch and the modebar needs hiding. Do not use Plotly `scattergeo`: it fetches its topojson from
`cdn.plot.ly` at runtime. Leaflet is bundled but unnecessary here. Every visual needs a list equivalent: the "All
fragments" tab with theme and story filters.

### 3.6 Shelfmark style (one convention)

Normalise display labels to the forms used in parts 1 and 2: `T-S NS 159.124`, `T-S Misc.35.11`, `T-S AS 109.184`,
`T-S 10H4.5` (KTIV writes "T-S 10 H 4.5"), `Or.1080 J105`, `L-G Lit. I.158`, `ENA 2808.59`, `Halper 251`, `JRL B 2562`,
`AIU IV.A.157`, `Bodl. MS heb. b 3/5`, `Mosseri VII.142`, `Yevr. III B 904`. Keep the institution in a separate
`holding` field for the courtesy line.

---

## 4. Lessons from /yom-kippur, turned into rules

| YK problem (evidence) | /sukkot rule |
|---|---|
| Subtitle says every fragment was "checked by a second reader"; 6 of 13 documents have 0 agreed lines, 2 more have 1 | No page-level claim of checking. Per card: "N of M lines agreed by a second machine reader" only when N > 0; otherwise no read link and no claim |
| 15 zero-agreed YK images force-surfaced 2026-09-17 are still public after the holiday (the reset in the handoff is pending) | Never force-surface for Sukkot. The builder reads only reads that are surfaced by rule. The YK reset remains the owner's decision (data write) |
| Cards have no content description, rite, date, catalogue link or bibliography | Every card has a human-sourced `card_line` + source badge, a date where known, a catalogue-record button (modal) and one bibliography line where one exists |
| 26 parallel API calls on load for static content | 0 calls on load; one lazy chunk carries the data; calls only on click |
| Full-resolution images as thumbnails (KTIV JPEGs ~2.8 MB each) | 320 px thumbs in `public/sukkot/thumbs/` |
| `SOURCE_INDEX` hard-coded, edited v6 → v7 → v8 with a rebuild each time | Index names live in the JSON; `/read` links carry no index, or v7 only for Oxford while v7 exists; the builder validates ids against the current default |
| Shelfmark labels inconsistent ("Cambridge University Library, Cambridge, England Ms. L-G Bib. VI 29" vs "Cambridge CUL: Or.1080 7.5") | One normalised label per card (§3.6) |
| Not linked from anywhere (header, tour, FAQ, chat) | Link from the header (or a festival banner on `/` during Sukkot), an FAQ entry, a tour step and a chat example prompt. Permanent page |
| Link previews: index.html is static | **Solved for YK on prod-mbp** (live check: `curl -A facebookexternalhit/1.1 https://cairogenizah.ai/yk` returns `og:title` "Yom Kippur in the Cairo Genizah" and `og:image` `/og-yom-kippur.jpg`). Copy the nginx `sub_filter` block for `^/sukkot(/.*)?$`, make `og-sukkot.jpg` with the PIL snippet in `docs/site_metadata.md` (`og([...], [...], OUT/"og-sukkot.jpg", "סוכות")`), then re-scrape in the Facebook debugger |
| `/read` on mobile | Fixed and live (see §1.1). Still test at 375×812: the page, the card deep link `?f=`, and the jump into `/read` for AIU IV.B.46 and Bodl. b 3/5 |
| "One-time page" | Keep it permanent; put the edition in the JSON; update annually |
| YK intro dates the day as "21–22 September 2026", but 10 Tishri 5787 = Mon 21 Sep, from sundown Sun 20 Sep (computed with `scratchpad/hebcal_check2.py`, Reingold–Dershowitz) | Compute festival dates with a tested calendar function. Sukkot 5787: 15 Tishri = Sat 26 Sep (from sundown Fri 25 Sep); Hoshana Rabbah Fri 2 Oct; Shemini Atzeret Sat 3 Oct; Simhat Torah (diaspora) Sun 4 Oct |
| AI index keyed by pre-v8 Oxford ids (new) | `ai_read_doc_id` plus the image caveat (§1.6) |
| Machine text errors on agreed lines (T-S 13J8.10 year; AIU IV.B.46 "נפראו"; T-S 18J4.1 Arabic script; ENA NS 5.29 CJK) | A `machine_read.note` on the card, or hold the item |

Also avoid the /explorer patterns: no "Debug Information" block, no index dropdown, and no calls to the shared-state
`/calculate` or `/project-query`.

---

## 5. Suggested order of work on the MBP (Sukkot has started, so ship in two steps)

1. `git status` and `git log` in the MBP clone. Reconcile the unpushed v8 changes with the owner before branching (e.g.
   `sukkot-page` off the MBP's current HEAD).
2. **v1 (hol ha-moʿed):** curation YAML for the 20 seed cards plus the reserves that pass a human check, the six story
   intros, the festival-days strip, the All-fragments list, OG tags and header/FAQ links. The builder runs without
   embeddings. Scratch build and check at 375×812, then **the owner approves the frontend rebuild** (prod deploy).
3. **v2 (before Simhat Torah, or for next year):** 100–250 fragments, the embedding map (+ global toggle), timeline,
   places SVG and the diagnostics report.
4. Out of scope for the page, but it blocks clean Oxford links: re-keying the AI index to v8 ids (data write, owner's call).

## 6. Open questions for the owner

- May the builder use the local ES read-only with credentials on the MBP, or should it stay on the public API (~750
  paced calls)?
- Where should the data live: a lazy chunk (recommended) or `public/data/` as an open dataset?
- Is Leaflet with CARTO acceptable for the places map, or is a static SVG preferred (recommended)?
- Should ENA NS 5.29 (Salonika) and other items whose Sukkot link is only in machine text appear in v1 with labels, or be
  held?
- `/sukkot` or `/succot`, and is a short link (`/sk`) wanted for WhatsApp Status as with `/yk`?

## Appendix: files

- This note: `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/7af50c33-1405-430f-b603-54ef506ab90e/scratchpad/sukkot_page/tech_notes.md`
- API evidence: `…/scratchpad/sukkot_page/api/` (`_calls.log`, `_status.log`, `openapi.json`, `indices.json`,
  `emb_*.json` [3 vectors], `ai_status_*id_b35.json`, `doc_v7_oldid_b35.json`, `doc_v8_*.json`, `fullindex_headers.txt`,
  `yk_crawler.html`, `asset-manifest.json`)
- Code read (origin/prod-mbp): `src/frontend/src/react_app.jsx` (Routes about l.1654; `handleOpenEsDocument`; `KG_ES_INDEX`),
  `src/frontend/src/YomKippur.jsx` + `.css`, `src/frontend/src/read/ReadFragment.jsx` (params and fetches about l.446–531) + `caveat.js`,
  `src/frontend/{Dockerfile,nginx.conf,package.json,public/index.html}`, `docs/site_metadata.md`,
  `docs/planned_features/2026-09-17_yom_kippur_viewer_handoff.md`, `src/backend/app.py` (`/document` l.430,
  `/ai-transcriptions` l.513, visualization endpoints l.864–1180, `/shelfmark/{shelfmark}/documents` l.1248),
  `src/backend/search_service.py` (`get_shelfmark_documents` l.2437), `src/backend/visualization_service.py`
  (`_fitted_models`), `src/backend/compute_full_visualization.py`, `docker-compose.yml` (`./data/visualization` mount), `docker-compose.mbp.yml`.
