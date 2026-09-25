# Handoff: "Exploring Sukkot through the Cairo Genizah" page (2026-09-25)

Written on the Mac Studio for a Claude Code session on the **MacBook Pro (the production host)**. Files in this folder:

| file | what it is |
|---|---|
| `2026-09-25_sukkot_page_handoff.md` | this brief |
| `sukkot_seed.json` | 125 verified fragments against `genizah_merged_v8`, themes with scholarship intros, genre colours, clusters, references, exclusions |
| `SEED_SUMMARY.md` | counts per theme and readiness, image problems, the 12–15 hero fragments |
| `tech_notes.md` | code state, visualisation design, builder-script plan, JSON shape, lessons from /yom-kippur, open questions |

Background research (private artifact, same Claude account): https://claude.ai/artifact/QwdWJQjZpCRCW6TsL7R5qK. Use `Artifact` `action: "read"` if you need a source for a claim.

## Goal and timing

Build a **permanent** festival page at `/sukkot`, plus an `/sk` redirect if cheap. It is richer than `/yom-kippur`. A visitor should be able to explore Sukkot through Genizah fragments grouped into clear themes:
- hoshanot and the Day of the Willow;
- festival prayers;
- Shemini Atzeret and the prayer for rain;
- Simhat Torah and the reading cycle;
- piyyut;
- haftarot and Targum;
- homilies;
- the law of the sukkah and the four species;
- the calendar;
- the Mount of Olives assembly;
- palm branches, myrtle and citrons;
- the synagogue sukkah;
- documents dated by the festival;
- letters and travellers;
- Rabbanites and Karaites.

The page also gets its own visualisation.

**Dates (5787):**
- Sukkot began at sundown Fri 25 Sep 2026.
- Hol ha-mo'ed runs 28 Sep – 1 Oct.
- Hoshana Rabbah is Fri 2 Oct.
- Shemini Atzeret is Sat 3 Oct.
- Simhat Torah (diaspora) is Sun 4 Oct.

Aim for **v1 during hol ha-mo'ed** (cards, themes and the festival-days strip) and **v2 later** (semantic map, timeline, places).

## Read first: repo state on this machine

- **The live site is ahead of `origin/prod-mbp`.** The live bundle points at `genizah_merged_v8`; `origin/prod-mbp` (e765768) still says v7. Run `git status` and `git log` here first. **Never rebuild from the remote branch**: that would roll the site back.
- **Branching:**
  - Create a feature branch from this checkout's current HEAD, e.g. `feature/sukkot-page`.
  - Push the unpushed prod-mbp commits as they are.
  - Open a PR when done. The owner wants branches merged, not lost.
- **Getting these files:** they arrive on branch `docs/sukkot-page-handoff` of genizah_search. Take just the folder, without switching branches:
  ```
  git fetch origin docs/sukkot-page-handoff
  git checkout origin/docs/sukkot-page-handoff -- docs/planned_features/sukkot_page
  ```
- **Deploys:** a frontend rebuild and container recreate is a production deploy. Ask the owner before building. Never touch the backend image, cloudflared, or LM Studio for this page.
- **Code already on prod-mbp and live:**
  - `/yom-kippur` (`src/frontend/src/YomKippur.jsx` + `.css`, routed in `react_app.jsx`) and the `/read` mobile fixes.
  - nginx Open Graph rewriting for `/yk`: copy it for `/sukkot`, with an `og-sukkot.jpg` made as described in `docs/site_metadata.md`.
- **Libraries:** Plotly (via react-plotly.js), Leaflet and Mirador. No d3 and no recharts. The Dockerfile installs without a lockfile, so **don't add npm packages**. Use plain SVG components for the visuals.

## Data: `sukkot_seed.json`

### What is in it
- **125 fragments**, each with:
  - `doc_id` (v8), `old_ids`, `shelfmark`, `institution`;
  - `theme`/`subtheme`;
  - `card_line` from a human source, with `card_source_public`;
  - dates and places **only where a human source gives them**;
  - `images` (`ok_url` = a verified loading image; `first_listed_url` and its HTTP status);
  - transcription and translation flags, `bibliography`, a hedged `scholarship_note`;
  - `ai_read` (with the id it lives under);
  - `readiness` / `usable_now`.
- **16 `themes`**, each with a title, an intro drawn only from verified scholarship, hedges, `contested` positions and reference keys.
- **7 `genres`** (colour buckets), **16 `clusters`** and **35 `refs`**.
- **Excluded lists:** `excluded_machine_only` (7 items whose Sukkot link rests only on machine text) and `not_on_site` (CUL Or.1080 3.54, the 1171 Simhat Torah colophon, which is not in v8).

### How to use it
- **73 cards are usable now.** Use `images.ok_url`, and **never assume `image_urls[0]` loads** (see the image problems below). Show a card only if `usable_now` is true, and re-check each `ok_url` at build time.
- **Card text.** Use `card_line` exactly. Credit it with `card_source_public`: "KTIV", "PGP", "OPenn", "catalogue note", "catalogue record" or a named scholar. Internally some rows say `card_source: "FJP"`; publicly that is only "catalogue note". **Do not name the Friedberg project anywhere on the page.**
- **Ids and AI reads.** Oxford ids changed in v8 (PGP style, e.g. `Oxford_Bodleian_Bodl_MS_heb_b_3_5`). The AI-transcription index is still keyed by the **old** ids, and its `image_index` follows the old image order.
  - Cards therefore carry two ids: `doc_id` for the catalogue record, and `ai_read.id_used` + the read's own `image_url` for `/read`.
  - Link `/read` only where the read has agreed lines. Label it "machine reading, N of M lines confirmed by a second reader".
  - Never link or describe a read with 0 agreed lines. This rule applies to Or.1080 15.21 image 0, which is still force-surfaced from Yom Kippur.
- **Contested scholarship is shown as contested:**
  - the 921 proclamation: Stern 2019 says the second day of Sukkot; Laufer 2025 says Hoshana Rabbah;
  - Yaari vs Elizur on Palestinian Simhat Torah poetry;
  - the occasions of CAJS Genizah 259 and Bodl. MS heb. e 39/94.

  Keep every hedge in the theme intros ("probably", "thus far").
- **Don't reuse these source dates as given:**
  - PGP's 29 Sep 1219 for T-S NS J221 part (b): it is 2 Oct.
  - "1544" for Tishri 5304 on T-S 13J8.10: it is 1543.
  - Gottheil–Worrell's 1043 for Freer F 43: computed 1042.
  - Laufer's 27 Sep 921: 26 Sep.

  `SEED_SUMMARY.md` lists the corrected values.
- **Leave out of v1:** Salonika (ENA NS 5.29: its Sukkot link is only in a machine read that contains CJK characters), and every `needs_human_check` row.

## Image problems (tell the owner; don't work around them silently)

1. **The 23 September KTIV scrape images were never uploaded to GCS.** In v8 they return 404: 100 of 100 checked, about 543 records site-wide. This is a **live-site problem**, not only a problem for this page.
   - On records that gained a KTIV twin, v8 lists the dead KTIV images **first**, e.g. Paris AIU IV.A.157 and T-S K6.80.
   - The fix is `python -m src.datasets.merging.upload_ktiv_images` (dry run first) in historical-document-analysis on **the Mac Studio**, where the zips are. It is a production data write, so it is the owner's call.
   - After the upload, re-HEAD the seed's `first_listed_url`s. About 37 more cards become usable.
2. **Bodleian paths.** Two Bodleian image paths under new ids 404: heb. f 48/3 and heb. d 74/20. Check after the next upload.
3. **Records with no images:** Berlin A1.1, JTS R1892, Freer F 43, Mosseri VII.142, T-S Misc.28.256, Yevr. III B 904, RNL Evr. II A 145/16. Show these as text cards or leave them out.
4. **Images in doubt; don't feature them:**
   - T-S B14.67: the recto file may be B14.66.
   - Or.1080 15.1: its two image files are byte-identical.
   - Or.1080 J45: image 0 may belong to J44.

## Design

The full proposal is in `tech_notes.md`.

- **No API calls on page load.** Ship one prebuilt JSON in a lazily loaded chunk. Call `/document` and `/read` only on click; reuse MapView's `handleOpenEsDocument` + `DocumentModal` pattern for a "catalogue record" button.
- **Build the page data with a script** (e.g. `scripts/sukkot_page/build_sukkot_data.py`).
  - Inputs: the seed plus read-only public API calls.
  - The script must fail the build if the output contains:
    - "Friedberg", "FJMS" or "FJP" in any public field;
    - any "checked" claim for 0-agreed reads;
    - hidden-read text;
    - a card without a human source.
- **Thumbnails at 320 px**, made offline. KTIV images are about 2.8 MB each.
- **Lead visual: a "festival days" strip.** It runs from a preparations slot through 15 to 23 Tishri: first day, hol ha-mo'ed, Hoshana Rabbah / Day of the Willow, Shemini Atzeret, Simhat Torah.
  - Liturgy and practice cards sit on their day. It is the one view where undated liturgy has a natural place.
  - The 921 proclamation is a single contested bracket spanning day 2 (Stern) and Hoshana Rabbah (Laufer).
- **Theme sections.** Each has an intro from `themes[*].intro` with its references (the page itself is the scholarship-to-manuscript bridge), then its cards.
- **Own visualisation (v2), with plain SVG:**
  - **(a) An exploratory semantic map** of the Sukkot fragments, coloured by genre bucket; clicking a point opens its card.
    - Precompute it offline. Get vectors with `GET /shelfmark/{doc_id}/documents?include_embeddings=true` (1024-d, unit length), then reduce with PCA/UMAP.
    - **Do not call `/visualization-explorer/calculate`.** It runs on the prod backend and overwrites a model that `/project-query` shares with every visitor.
    - Genre dominates festival in these embeddings: hoshanot vs the Mount of Olives letter score 0.415, while random pairs average about 0.45. So **curated themes are the primary grouping**, and the map is labelled "exploratory".
    - Optional: position the Sukkot points within the existing whole-collection map `data/visualization/full_index_visualization_genizah_merged_v8.json` on this machine.
  - **(b) A timeline** of the dated items. State plainly how many are undated.
  - **(c) A static SVG map of places:** Fustat, Jerusalem / Mount of Olives, Ramla, Damietta, Alexandria, Aden, Qayrawan, Tyre/Haifa.
- **Credits:** NLI-KTIV and the Princeton Geniza Project; images courtesy of the holding libraries; OPenn items are public domain. Add one plain line saying machine readings are not checked by a person, and link to the full wording.
- **Discoverability:** link the page from the header, the FAQ, the first-visit tour and a chat example prompt ("What do Genizah fragments tell us about Hoshana Rabbah?"). Add Open Graph tags through nginx, as `/yk` does.

## Lessons from /yom-kippur (don't repeat them)

- It claimed "checked by a second reader" while 6 of 13 fragments had 0 confirmed lines.
- It force-surfaced reads with 0 agreed lines, and never reset them after the holiday.
- Its cards had no content description.
- Nothing linked to the page.
- It made 26 parallel API calls on load.
- Its intro dates are one day late: it says "21–22 September", but Yom Kippur 5787 ran from sundown Sun 20 Sep to Mon 21 Sep. Fix that copy while you're in there.

## Verify before asking to deploy

- Do the builder checks above.
- Every card image returns HTTP 200.
- At 375×812 and on desktop:
  - the page shows content without scrolling past a banner;
  - no console errors;
  - the festival strip and themes work;
  - cards open the right catalogue record, and `/read` links (where present) open the right image.
- A crawler fetch of `/sukkot` returns the Open Graph title and image.
- Then show the owner screenshots and ask before the frontend rebuild.
