# Handoff: /read viewer on mobile + one-time Yom Kippur page (2026-09-17)

**Why now.** Yom Kippur starts the evening of 21 Sep 2026. Thirteen liturgical Yom Kippur fragments (25 images) now have
AI reads live in `genizah_ai_transcriptions_v2`, and the owner will share one of them on WhatsApp Status. Most visitors
will arrive on a phone at
`https://cairogenizah.ai/read?doc=Cambridge_Lewis_Gibson_L_G_Bib_VI_29&image=0&index=genizah_merged_v6`.
That page works on mobile but is not Status-ready (findings below). Everything here is frontend/backend work in this repo;
the reads themselves are produced in `historical-document-analysis`.

## State you inherit

- `src/backend/ai_transcriptions.py`: **uncommitted** one-line change adding `"lines-v3-20260917"` to
  `ACCEPTED_RULE_VERSIONS`. Rule v3 (a horizontal gate on Kraken fragment assignment) fixes boxes that spanned the
  gutter on two-page openings. **Deploy the backend with this before any v3 record is loaded** — the response model
  rejects unknown rule versions and the read page would error. The v3 records are ready on the Studio
  (`ai_reads/ai_reads_v21b_yom_kippur_5787_v3.jsonl`); reload them with `scripts/load_ai_transcriptions.py --apply`
  after the deploy.
- Live data: 25 Yom Kippur records loaded with `AI_READ_MIN_AGREED_LINES=0 AI_READ_MIN_AGREED_SHARE=0`; 10 surfaced by
  rule; the other 15 (zero confirmed rows) were force-surfaced with an ES `_update_by_query` setting `surfaced=true`
  (filter: doc_id in the YK list, `ai_read.vlm_model = qwen3-vl-8b-heb-v21b-step1200`). They render in the viewer's
  existing "single reader, caution" state. **After the holiday** set `surfaced=false` on those 15 (same query) or just
  reload the file — the loader recomputes the flag. Lewis-Gibson VI.29 image 0 was re-read to include its left page
  (36 of 40 rows confirmed) and re-uploaded; image 1 is 26 of 40.
- Showcase fragment: `Cambridge_Lewis_Gibson_L_G_Bib_VI_29` (Yom Kippur selihot / viddui). Other strong pages:
  `Cambridge_CUL_Or_1080_7_5` (13/16 on image 1), `Cambridge_CUL_Or_1080_15_21` (7/12 on image 1),
  `Cambridge_CUL_Or_1080_1_55` (13/49), `Cambridge_CUL_T_S_NS_200_51` (7/29, 6/29).

## Mobile findings on /read (375×812, no console errors) — `src/frontend/src/read/ReadFragment.jsx` + `.css`

1. **Banner above the fold.** `BetaBanner` is a full-screen paragraph on a phone; the manuscript is below it.
   → On viewports < 600 px collapse it to one line ("Beta: machine reading, not checked by a person.") with a
   "read more" toggle. Keep the full text on desktop.
2. **Initial zoom is useless on phones.** `ImageStage.fit()` fits the whole image into a 480 px stage → a two-page
   opening renders at 3 % with forty boxes as hatching. → On small viewports fit the *width* of the first confirmed
   line's page (or ~1.6× the viewport width) and centre on the first line (reuse the `focusRequest` path with
   `zoom: true`).
3. **No pinch-zoom.** `.read-viewport { touch-action: none }` disables browser zoom and the pointer handlers track one
   pointer only, so the only zoom is the + button. → Track two active pointers in `onPointerDown/Move/Up` and call
   `zoomAt(scale, midpoint)`; change the hint to "Pinch to zoom, drag to pan" when `(pointer: coarse)`.
4. **Nested scroll trap.** `.read-text { max-height: calc(100vh - 220px); overflow: auto }` captures vertical swipes
   inside the text list; the page appears stuck. → In the `@media (max-width: 900px)` block drop the `max-height`
   so the page scrolls naturally (the stage can stay fixed-height).
5. Small: the row numbers on the overlay overlap at low zoom (hide `showNumbers` below ~15 % zoom on mobile); the
   confirmed-count badge and image picker should sit above the banner (they already do) — keep them visible.

## One-time Yom Kippur page (suggested)

A single curated route, e.g. `/yom-kippur` (short enough for a Status link; add `/yk` → `/yom-kippur` if a redirect is
cheap in the frontend router or at Cloudflare). Content: a two-line intro ("Fragments of Yom Kippur liturgy from the
Cairo Genizah, read by machine and checked by a second reader"), then cards for the 13 documents ordered by confirmed
share (VI.29 first), each with the thumbnail, shelfmark, catalogue title, "N of M lines confirmed", and a link to the
/read page for its best image. Reuse the search-result card and the existing `/ai-transcriptions/{doc_id}` status
endpoint for the counts. Mark it clearly as a one-time page (a date in the intro) and keep the beta caveat as one line
with a link to the full wording. Doc ids: Philadelphia_CAJS_Halper_204, Philadelphia_CAJS_Halper_233,
Philadelphia_CAJS_Halper_234, Cambridge_CUL_T_S_NS_200_5, Cambridge_CUL_T_S_NS_200_11, Cambridge_CUL_T_S_NS_200_51,
Cambridge_CUL_Or_1080_15_5, Cambridge_CUL_Or_1080_15_21, Cambridge_CUL_Or_1080_7_5, Cambridge_CUL_Or_1080_1_55,
Cambridge_CUL_T_S_24_10, Manchester_JRL_A_217, Cambridge_Lewis_Gibson_L_G_Bib_VI_29.

## Verification

Check at 375×812 after each change: the manuscript visible without scrolling, pinch works, the page swipes freely,
VI.29 image 0 shows 36/40 with green boxes on both pages. No FJMS/Friedberg naming anywhere on the new page
(credit NLI-KTIV and the Princeton Geniza Project only; images courtesy of the holding libraries).

## Status (2026-09-17, later)

Implemented, not yet built or deployed (no Node on the host; the frontend build is a Docker step that needs approval):

- `src/frontend/src/read/ReadFragment.jsx` + `.css`: banner collapses to one line under 600 px with a read-more toggle;
  phones open on the first confirmed line at the width of its page (`focusRequest.zoom === 'page'`, two-page openings
  detected by aspect ratio > 1.15); two-pointer pinch zoom in the pointer handlers; hint says "Pinch to zoom" on
  coarse pointers; resize only refits when the stage width changes (address-bar show/hide no longer resets the zoom);
  row numbers hidden below 15 % zoom on phones; `.read-text` no longer scrolls inside the page under 900 px.
- `src/frontend/src/YomKippur.jsx` + `.css`, routed at `/yom-kippur` with `/yk` redirecting to it. The 13 doc ids are a
  constant in the component; counts come from `/ai-transcriptions/{doc_id}`, shelfmarks from
  `/document/{doc_id}?index_name=genizah_merged_v6` (titles are null for these records, so cards show the shelfmark).
- Backend: the `lines-v3-20260917` rule-version change is still uncommitted; deploy it before loading v3 records.
- Not done: per-route Open Graph tags for the WhatsApp preview (index.html is static; would need SSR or a Cloudflare
  rule), and the after-holiday `surfaced=false` reset.
