# SEO plan for cairogenizah.ai (2026-09-28)

## Where we stand (measured)

| Check | Result |
|---|---|
| What a crawler gets without JavaScript | 130 characters: "You need to enable JavaScript" and the loading splash |
| `robots.txt`, `sitemap.xml` | Neither exists; both return the app shell as `text/html` |
| Page titles | One title for `/`, `/about`, `/faq`, `/map`; only `/sukkot` and `/yom-kippur` have their own |
| Document pages | None. `/document/<id>` returns the app shell; records only exist inside a modal |
| Internal links to records | None that a crawler can follow (buttons open a modal) |
| JavaScript bundle | 7.2 MB (2.4 MB gzipped), one chunk for every route |
| About page | Live with 17 "[Placeholder]" blocks, indexable |
| Off-site | Indexed under the old title "Cairo Genizah Search"; press coverage exists (Jewish Exponent, Pittsburgh Jewish Chronicle); GitHub repo ranks |

The site is a single-page app: every URL serves the same `index.html`, and all content arrives through API calls
after JavaScript runs. Google can render JavaScript, but slowly and not for the modal-only records; Bing and most
others mostly index what the HTML contains. Right now that is one sentence. The 70,000 catalogue records, which
carry exactly the terms scholars search for (shelfmarks like "T-S 8J5.14", names, places, titles of works), are
invisible to every search engine.

## Priorities

### 1. Foundations (about a day, no architecture change)

- `public/robots.txt` allowing everything except `/read?`, `/api`, and pointing at the sitemap.
- `public/sitemap.xml` for the static routes, and a sitemap index that also lists the generated record sitemaps (step 2).
- Per-route `<title>` and `meta description` for `/`, `/about`, `/faq`, `/map`, `/explorer`, `/chat` with the same nginx
  `sub_filter` mechanism `/sukkot` uses. Home title: "Cairo Genizah AI: search 70,000 medieval manuscript fragments".
- Real static content in `index.html` for non-JS crawlers: a paragraph on what the site is and links to `/about`, `/faq`,
  `/sukkot`, `/map`, inside the loading container (hidden once the app mounts) instead of "enable JavaScript".
- Google Search Console and Bing Webmaster Tools: verify the domain, submit the sitemap, request re-indexing of `/`
  (the stale "Cairo Genizah Search" title), watch the coverage report.
- Fill the About page. Placeholder text is currently indexable and looks abandoned.

### 2. Crawlable record pages (the big lever; about a week)

Give every catalogue record a real URL with real HTML: `/fragment/<doc_id>`.

- A backend endpoint renders a lightweight server-side HTML page (Jinja template) from the Elasticsearch record:
  normalised shelfmark in the title, description as meta description, canonical URL, the image, institution and
  collection, date, human transcription and translation when present, bibliography with the work links, a
  "Machine reading (beta)" line when a surfaced read exists, and JSON-LD (`schema.org/Manuscript` with
  `ImageObject`, `holdingArchive`, `citation`). A prominent "Open in Cairo Genizah AI" link opens the app on that record.
- nginx routes `/fragment/` to the backend. The React app also handles `/fragment/<id>` by opening the modal, so
  humans get the app and crawlers get the HTML (progressive enhancement, no prerender service).
- Same pattern for `/work/<slug>` (the bibliography work card, 8k works) and later for KG places and people.
- Generate record sitemaps from the index: 73k URLs across `sitemap-fragments-N.xml` files of 50k max, with `lastmod`
  from `indexed_at`; rebuild on each index bump (a script next to `build_sukkot_data.py`).
- Make existing pages link to permalinks: the modal gets a "Permalink" button, the Sukkot and Yom Kippur cards link
  their shelfmarks to `/fragment/<id>`, search results include a plain link.
- Check reuse terms for the description text: PGP descriptions and KTIV titles are being republished on a public
  page. PGP's data is openly licensed; confirm for KTIV before shipping.

### 3. Content that earns queries (ongoing)

- Curated topic pages built like `/sukkot` (prebuilt JSON, no API on load): ketubbot, the India trade, Maimonides
  autographs, Karaites, Fustat, letters of merchants, festivals. Each is a durable landing page for a term people
  search, with crawlable links into record pages.
- FAQ answers as real content: what the Genizah is, how to read a shelfmark, what "machine reading" means here.
- A dataset description page with `schema.org/Dataset` markup so the site appears in Google Dataset Search, which
  researchers use.

### 4. Speed and rendering (a few days)

- Split the 7.2 MB bundle: load Mirador, Plotly and Leaflet only on the routes that use them (`React.lazy`, as
  `/sukkot` already does). Target under 500 KB gzipped on first load; this is the Core Web Vitals lever.
- Preconnect to `api.cairogenizah.ai` and `storage.googleapis.com`; set width/height on images to avoid layout shift.
- Keep `index.html` no-cache and assets immutable (already done).

### 5. Off-site

- Ask the Princeton Geniza Project, NLI/KTIV, and the Cambridge Genizah Research Unit for a link from their resource
  pages; add the site to the "External links" of the Wikipedia article on the Cairo Geniza; get listed in university
  library research guides for Jewish studies.
- Publish the festival pages and topic pages where scholars are (H-Judaic, the Genizah mailing lists, social media);
  each share is a link.
- Keep the press coverage linked: the two 2026 articles already point here.

## Measuring

Search Console impressions and clicks by query, monthly; coverage (indexed record pages) after step 2; Core Web
Vitals from the Search Console report after step 4. GA4 is already installed through Google Tag Manager.

## Order

1 → 2 → 4 → 3 → 5. Step 1 is cheap and immediate; step 2 is what changes findability; step 4 protects rankings once
pages are indexed; 3 and 5 compound over time.
