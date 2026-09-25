# Sukkot page: merged seed summary

`sukkot_seed.json` combines the Task A core seed (96 items) and the Task B new-record picks (40 items). It contains 125 fragments, deduplicated by v8 `doc_id` against `genizah_merged_v8`. Every card line comes from a human source.

Also in the JSON:
- 16 themes, each with an intro drawn only from verified scholarship;
- 7 genre colour buckets;
- 16 human-sourced clusters for the page's own visualisation;
- 35 references;
- the excluded rows, with the reason each was left out.

`generated_at` is set to the placeholder "2026-09-25". No repo, `.env` or credentialed store was touched.

## What went in, what stayed out

| step | rows |
|---|---|
| Task A core seed | 96 |
| Task B picks | 40 |
| in both seeds (merged into one row each; Task B's fuller card wording used) | 3 (T-S H7.45, T-S H12.11, Bodl. MS heb. e 39/94) |
| excluded: Sukkot link exists only in machine text | 7 |
| excluded: not in v8 | 1 (Or.1080 3.54, the 1171 "Simhat Torah was completed" colophon) |
| **fragments in the seed** | **125** (85 core, 37 new, 3 in both) |

The seven machine-only exclusions are listed in `excluded_machine_only` with their reasons:
- T-S AS 145.52
- T-S 12.67
- T-S 18J4.1
- Bodl. MS heb. c 28/28
- Bodl. MS heb. d 36/14
- L-G Ar. II.64
- Or.1080 15.2

Bodl. MS heb. e 39/94 was excluded in Task A but comes back in from Task B. Its catalogue cards (Neubauer–Cowley and Schocken) name it a yotser for Sukkot, attributed to Yosef ibn Avitur, so it now has a human source.

Where the card line comes from:

| source | cards |
|---|---|
| KTIV | 46 |
| PGP | 34 |
| site catalogue record | 19 |
| catalogue note | 12 |
| published scholarship | 12 |
| OPenn | 2 |

"Catalogue note" is the value stored as `FJP` in the internal field `card_source`. The public label is `card_source_public`, which reads "catalogue note" and never names the project.

## Counts by theme and readiness

`usable_now` means a card can go on the page today. It requires three things:
- a human-sourced card line;
- a stored image URL that loads (`images.ok_url`);
- no human check or image-assignment doubt blocking it.

| theme (id) | total | ready | needs image fix | needs human check | needs description fix | usable now |
|---|---|---|---|---|---|---|
| Hoshanot and the Day of the Willow (`hoshanot_hoshana_rabbah`) | 17 | 10 | 6 | 0 | 1 | 10 |
| Festival prayers (`festival_prayers`) | 9 | 4 | 4 | 1 | 0 | 4 |
| Poems for Sukkot (`qerovot_piyyut`) | 11 | 4 | 6 | 1 | 0 | 5 |
| Shemini Atzeret and the prayer for rain (`shemini_atzeret_geshem`) | 9 | 3 | 6 | 0 | 0 | 3 |
| Simhat Torah and the reading cycle (`simhat_torah`) | 6 | 2 | 4 | 0 | 0 | 2 |
| Readings: haftarot, Targum and Psalms (`haftarot_targum_bible`) | 6 | 4 | 1 | 0 | 1 | 5 |
| Sermons for the festival (`homilies`) | 3 | 1 | 2 | 0 | 0 | 2 |
| The law of the sukkah and the four species (`halakha`) | 20 | 12 | 8 | 0 | 0 | 12 |
| Later customs and Kabbalah (`later_customs_kabbalah`) | 3 | 1 | 2 | 0 | 0 | 1 |
| Fixing the festival: the calendar (`calendar`) | 7 | 5 | 1 | 1 | 0 | 5 |
| The assembly on the Mount of Olives (`mount_of_olives`) | 10 | 6 | 2 | 2 | 0 | 6 |
| Palm branches, myrtle and citrons (`four_species`) | 6 | 6 | 0 | 0 | 0 | 6 |
| The sukkah in the synagogue courtyard (`communal_sukkah`) | 3 | 3 | 0 | 0 | 0 | 3 |
| Dated by the festival (`dated_by_festival`) | 8 | 4 | 4 | 0 | 0 | 4 |
| Letters, greetings and travellers (`greetings_letters`) | 3 | 2 | 1 | 0 | 0 | 2 |
| Rabbanites and Karaites (`karaites`) | 4 | 3 | 1 | 0 | 0 | 3 |
| **total** | **125** | **70** | **48** | **5** | **2** | **73** |

Rows where `usable_now` differs from `ready`:
- **Usable now although not marked ready:**
  - T-S K6.80 and AIU IV.A.157 have a working older image stored in `ok_url`.
  - JRL B 3078 is usable because the card carries the verse reference that the record lacks.
- **Not usable now:** T-S NS J370. Its Sukkot words are only in PGP's transcription, which our record does not carry, so it should be re-checked before it is quoted.

Each theme has a genre colour bucket (7 in total), and Karaite items also carry a `karaite` tag. `themes[*]` holds, for each theme:
- the title and a one-paragraph `intro`;
- `hedges`;
- `contested` positions;
- `verified_claims` (C-numbers from `verify_scholarship.md`);
- `refs` keys into the top-level `refs`.

Contested points are shown as contested:
- the day of the 921 proclamation: Stern 2019, the second day of Sukkot, against Laufer 2025, Hoshana Rabbah;
- Yaari against Elizur on Simhat Torah poetry;
- the occasion of CAJS Genizah 259 and of Bodl. MS heb. e 39/94.

## Image availability

| state | rows |
|---|---|
| first listed image loads | 79 |
| a working image exists but is not first in v8 (T-S K6.80, AIU IV.A.157; both at v8 index 2 behind two dead KTIV URLs) | 2 |
| every listed image checked returns 404 | 37 |
| no images on the record | 7 |

**The upload caveat.** The 37 all-404 rows are 34 Task B picks plus T-S H7.45, T-S H12.11 and T-S Ar.19.25. All of them point at KTIV folders from the 23 September scrape that were never uploaded to the bucket. Task B found the zips on local disk under `historical-document-analysis/src/datasets/raw_data/cairo_genizah/ktiv/ktiv_PNX_MANUSCRIPTS<sys_num>-1_images.zip`, and the file names match the URLs in v8. Uploading them to `gs://cairo-genizah-es-json/KTIV/<sys_num>/` would turn most `needs_image_fix` rows into ready ones. That upload is a production data write and needs the owner's approval. Until it happens, the page should use `images.ok_url` and never assume `image_urls[0]` loads.

**Other image problems:**
- No images on the record: Berlin A1.1, JTS MS R1892, Freer F 43, Mosseri VII.142, T-S Misc.28.256, Yevr. III B 904, RNL Evr. II A 145/16.
- Image assignment in doubt: `usable_now` is false for the first two of these; the rest are flagged in their notes.
  - T-S B14.67: the recto file may show T-S B14.66.
  - Or.1080 15.1: its two image files are byte-identical.
  - T-S 24.68: the single image needs its side confirmed before it is used as the card image.
  - Or.1080 J45: image 0 may belong to J44.
  - T-S 8J17.24: the verso is not imaged.
- Duplicated images after the KTIV merge: AIU IV.A.85, T-S A32.41, T-S K6.80, AIU IV.A.157, T-S NS 320.42 and Bodl. MS heb. b 3/5. `images.n_listed` counts URLs, not leaves.

## Machine-read availability

- **Rows with a public machine reading that has at least one agreed line: 8.** All eight carry `ai_read.linkable = true` and the label "Machine reading, not a human transcription: N of M lines agreed by two machine readers".
  - Or.1080 15.21: 7 of 12
  - Bodl. MS heb. e 39/94: 20 of 26
  - AIU IV.A.157: 17 of 19
  - AIU IV.B.46: 24 of 31
  - Halper 262: 8 of 16
  - ENA 2555.1: 8 of 28
  - Bodl. MS heb. b 3/5: 6 of 18
  - T-S 13J8.10: 9 of 14
- The other 117 rows have no public reading.
- **Old-id caveat.** `genizah_ai_transcriptions_v2` is still keyed by pre-v8 ids.
  - Bodl. MS heb. b 3/5 and e 39/94 have reads only under `Oxford_Bodleian_MS_heb_b_3_5` and `Oxford_Bodleian_MS_heb_e_39_94`. Those ids return 404 on GET /document for v8, and the v8 ids return `available=false` for reads.
  - Each card therefore stores `doc_id` (for the record) and `ai_read.ai_read_doc_id` (for `/read`).
- **Image-order caveat.** On AIU IV.A.157 and Bodl. MS heb. b 3/5, the image a read belongs to is image 3 in the v8 record. Build the `/read` link from `ai_read.image_url`, not from a v8 image index.
- **Zero-agreed read still public.** Or.1080 15.21 image 0 has 0 of 12 lines agreed. Link only image 1, and never call the zero-agreed read checked.
- **Text caveats, stored in `ai_read.caveat`:**
  - AIU IV.B.46 has "נפראו" for נפרצו on 10 agreed lines.
  - T-S 13J8.10 has the year 5314 on an agreed line, against the catalogue's 5304.
  - The Bodl. MS heb. e 39/94 read must not be used as card text.

## Spot check (15 random rows, live API)

The sample was `random.Random(20260925)`. It used 17 API calls and 17 image HEADs, sequentially with the research User-Agent. The log is in `spot/_calls.log`, and the verdicts are in `spot/spot_verdicts.json` and on each row as `spot_check`.

**Id resolution:** all 15 ids resolve on v8 (GET /document returns 200 and echoes the same doc_id).

**Images:**
- Stored images (`ok_url`): 9 of 9 return 200. These are the 8 ready rows in the sample plus T-S K6.80, whose image is at v8 index 2.
- New KTIV rows (T-S H7.2, T-S NS 71.64, T-S NS 276.107a, T-S AS 81.90): the 8 listed images all return 404, as expected from the missing upload.
- Mosseri VII.142 and Yevr. III B 904 have no images.

**Card lines:**
- 14 of 15 match their source: the v8 record, the local merged-catalogue record that v8 is built from, or the verified scholarship.
- Fixed: ENA 2555.1. The card called it "the continuation of" the Day of Assembly passage. Stern's Table 3 and the v8 description show that ENA 2555.1 belongs to a different copy (BCC2), whose verso also preserves that passage. The card line has been rewritten.
- Mixed sources, now recorded in `card_source_note`:
  - T-S NS 71.64: the first clause comes from PGP and the Arabic name from KTIV.
  - Halper 258: the description comes from OPenn and the quoted stanza from a catalogue note.
- Yevr. III B 904's Sukkot quotation comes from the PGP 6181 page, which the scholarship check verified. Our v8 record does not carry it, so the card has to.

**Machine read:** ENA 2555.1 re-fetched as 8 of 28 lines agreed (v21b), matching the seed.

**Dropped:** none.

## Other corrections made while merging

- **Homilies cluster.** Task B's "second-day homilies" cluster said AIU IV.A.157 and T-S NS 258.87 share the heading "חג הסוכות תאני יום". No human source supports that:
  - on IV.A.157 the heading appears only in the machine read;
  - on NS 258.87, KTIV says only "from a homily for the second day of Sukkot".

  The cluster's basis and the theme intro now say this, and neither heading is used as a link between the two.
- **Bibliography line dropped.** ÖNB H 184 carried "Genazim team from catalogs", which is not a citation and names a project.
- **Internal fields.** `card_source`, `card_source_text`, `scholarship_note`, `notes` and `bibliography` may still name the catalogue-note project in internal wording. The fields safe to publish are:
  - `card_line`
  - `card_source_public`
  - `holding_credit`
  - `themes[*].title`, `intro`, `hedges` and `contested`
  - `clusters[*].title`
  - `refs`
- **Validation.** It passes for the fields above: no forbidden names, every card source is human, no zero-agreed read is linkable, and every theme and cluster member resolves.

## Opening fragments (14 usable today)

Each of these has a working image, a human-sourced card line and a scholarly anchor, and together they cover the page's main stories.

| # | shelfmark | doc_id | theme | why it opens the page | caveat |
|---|---|---|---|---|---|
| 1 | T-S NS 194.92 | `Cambridge_CUL_T_S_NS_194_92` | calendar | Ben Meir's son proclaims the calendar in Jerusalem on the "Day of Assembly", 921 | **Contested:** Stern (second day, Fri 21 Sep 921) vs Laufer (Hoshana Rabbah; computed Wed 26 Sep) |
| 2 | T-S Misc.35.11 | `Cambridge_CUL_T_S_Misc_35_11` | mount_of_olives | Hoshana Rabbah 1029 on the Mount of Olives: "they ascended ... according to their custom"; ban on the Karaites | Has a transcription and translation |
| 3 | T-S NS 320.42 | `Cambridge_CUL_T_S_NS_320_42` | mount_of_olives | c. 1045: sermon on the Mount "on the day of ʿArava"; few pilgrims came | Duplicated image URLs |
| 4 | T-S K2.8 | `Cambridge_CUL_T_S_K2_8` | calendar | al-ʿĀqūlī's 247-year calendar: "(day of the) Willow: Wednesday" (Vidro 2017) | The 1296/7 copy date comes from PGP/CUDL |
| 5 | T-S 24.68 | `Cambridge_CUL_T_S_24_68` | hoshanot | A 10th-c. Palestinian ketubba reused for hoshanot in seven alphabetic stanzas (Friedman) | Confirm which side the one image shows |
| 6 | T-S NS 159.124 | `Cambridge_CUL_T_S_NS_159_124` | hoshanot | Saadya Gaon's Siddur: the hoshana for Hoshana Rabbah (KTIV, after Uri Ehrlich) | Public description is subject headings only |
| 7 | Halper 251 | `Philadelphia_CAJS_Halper_251` | hoshanot | A pocket booklet (5 × 12.5 cm) for the "Day of the Willow", Shemini Atzeret and Simhat Torah; 18 images, public domain | Untranscribed |
| 8 | T-S F10.64 | `Cambridge_CUL_T_S_F10_64` | halakha | The only manuscript witness to Yerushalmi Sukkah besides Leiden, in the margins of Halakhot Gedolot (Sharlo 2023) | Untranscribed |
| 9 | Halper 258 | `Philadelphia_CAJS_Halper_258` | halakha | The laws of Sukkot in rhyme, in the form of Ibn Gabirol's Azharot: the etrog in the left hand | Credit the stanza as a catalogue note |
| 10 | T-S NS J221 | `Cambridge_CUL_T_S_NS_J221` | four_species | 1219 accounts: carrying palm branches, Day of the Willow expenses, and pay for "the hazzan R. Yedutun" (the scribe of Seder Fustat A) | Do not reuse PGP's 29 Sep; computed 2 Oct 1219 |
| 11 | T-S 16.222 | `Cambridge_CUL_T_S_16_222` | four_species | Lease c. 1150: the tenant cuts 1,000 palm branches and carries them to the Nile (Gil, doc. 50) | The Sukkot purpose is PGP's inference |
| 12 | T-S K25.190 | `Cambridge_CUL_T_S_K25_190` | communal_sukkah | Building the booth of the Palestinians' synagogue; Goitein's "synagogue courtyard only" observation | |
| 13 | Bodl. MS heb. b 3/5 | `Oxford_Bodleian_Bodl_MS_heb_b_3_5` | dated_by_festival | Damietta 1157: sailing "on Sunday, the second day of Sukkot ... violating the holiday" | Machine read only under `Oxford_Bodleian_MS_heb_b_3_5` (6 of 18); image-order mismatch |
| 14 | T-S 16.31 | `Cambridge_CUL_T_S_16_31` | qerovot_piyyut | Ben Yijū copies Judah ha-Levi's ofan for Sukkot on a piece of cloth (India Book I) | The record does not mention Sukkot; the card does |

Three more belong in the opening once the KTIV upload lands. All three are `needs_image_fix` today, with no live image.
- **T-S H12.11, Seder Fustat A.** The Babylonians' Simhat Torah in the Palestinians' synagogue; that the two congregations celebrated it jointly is an inference.
- **T-S H7.45.** An Arabic deed of 874 reused for a Shemini Atzeret piyyut; one new PGP identification that "requires further study".
- **T-S 20.182.** A 10th-c. Sukkot piyyut scroll written over a Christian Palestinian Aramaic Bible.

## Clusters for the visualisation

There are 16 clusters, each with its `basis`, member doc_ids, `outside_seed` shelfmarks and a `caveat`:
- Saadya's Siddur
- one Kallir hoshana in three witnesses (Palestinian vs Babylonian vocalisation)
- the "Day of the Willow"
- the Mount of Olives assemblies of 1029–1094
- the 921/2 calendar dispute
- Seder Fustat A and its cantor
- Simhat Torah texts
- Sahlan's qedushta in two witnesses
- second-day homilies
- the four-species economy
- the synagogue sukkah
- reused writing material
- Palestinian vocalisation
- Sukkot on the India route
- Rabbanites and Karaites
- the law of the festival from the Mishnah to Kabbalah

## Files

All files are in `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/7af50c33-1405-430f-b603-54ef506ab90e/scratchpad/sukkot_page/`:
- `sukkot_seed.json`: the merged seed, about 330 KB. The reserve lists, the 41 queue rows not in v8 and the full 1,018-row queue status stay in `new_records_seed.json`.
- `SEED_SUMMARY.md`: this file.
- `merge_seed.py` and `themes_def.py`: rebuild with `python3 merge_seed.py`. This runs offline and validates on build.
- `spot_check.py`, `spot_sample.json`, and `spot/` (`spot_results.json`, `spot_verdicts.json`, `ai_ENA_2555_1.json`, `_calls.log`).
