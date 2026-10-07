# Synthetic query writing spec (v2) — Cairo Genizah search

You are writing **search queries that real users of a Cairo Genizah search engine would type**, for which a given
catalogue record is a correct answer. The queries train and evaluate a retrieval embedder whose weakness is
**understanding what a query is about** when it doesn't name the topic. Example: "laws of lulav" should find
Talmud Bavli Sukkah fragments, and "Kol Nidre" should find Yom Kippur liturgy.

## Input
One JSON object per line:
- `doc_id`
- `pool` (train/eval)
- `stratum`
- `topics` (catalogue-derived topic labels, may be empty)
- `text`: the record exactly as the search engine sees it. Its labelled lines are Document ID, Shelf Mark,
  Description, Alternate titles, Language, Document Type, Date, Transcription (≤1000 chars), Translation,
  Related People/Places.

## Output
One JSON object per input record, same order, written as JSONL:
```
{"doc_id": "...", "queries": [
  {"type": "implicit", "lang": "en", "text": "...", "grade": 2, "basis": "..."},
  {"type": "topical",  "lang": "en", "text": "...", "grade": 2, "basis": "..."},
  {"type": "hebrew",   "lang": "he", "text": "...", "grade": 2, "basis": "..."},
  {"type": "specific", "lang": "en", "text": "...", "grade": 2, "basis": "..."}
], "skip_reason": null}
```

## The four queries

1. **`implicit`** (English): the concept bridge. A natural query that describes the **practice, content or
   question** without naming the work, the tractate, the festival/topic, or a translation of any of them. Typical
   length is 3–9 words. This is the hardest query and the most important one.
   - Bavli Sukkah 29b–30a (stolen lulav) → "is a stolen palm branch valid for the ritual"
   - Hoshanot piyyut → "processional poems recited while circling with willows"
   - Mishneh Torah, Shehitah → "rules for checking a slaughtered animal's lungs"
   - Bavli Shabbat 90a (carrying) → "carrying small creatures out of the house on the seventh day"
   - Haggadah → "the four questions and the story of the Exodus at the table"
2. **`topical`** (English): how most users actually search. A natural research query that **may** name the
   festival, practice, genre or topic in plain English: "Sabbath laws on carrying", "Hoshana Rabbah liturgy",
   "marriage contract dowry list", "appeal for help paying the poll tax". It must still not be a copy of the
   record's title line, its shelfmark or its incipit.
3. **`hebrew`**: what a Hebrew-reading scholar would type: Hebrew terms or a short Hebrew phrase. Use standard
   scholarly Hebrew or the text's own key terms, but **not** a verbatim incipit of more than 3 words. For
   Judeo-Arabic or Arabic documentary records, a Hebrew-script term is fine ("כתובה", "שטר מכר", "מכתב בקשה
   לעזרה"). If no sensible Hebrew query exists, write a transliterated query instead ("hilkhot sukkah") and set
   `lang` to "translit".
4. **`specific`** (English): a precise query that uses concrete facts **present in the record**: a person,
   place, date, commodity or document type. Example: "Shelomo b. Yehuda letter Damascus 1029". It may reuse names,
   but not the shelfmark or Document ID. If a record has no people, places or dates, use the work and its section
   ("Maimonides Hilkhot Shabbat chapters 28–29").

## Rules
- **No invented facts.** Every query must be answerable from the record plus **general Jewish knowledge**.
  General knowledge is allowed and encouraged for the concept bridge: tractate Sukkah ↔ festival of Sukkot ↔ lulav
  and etrog; Yoma ↔ Yom Kippur; Pesahim ↔ Passover; Megillah ↔ Purim; Hullin ↔ ritual slaughter; Ketubbot ↔
  marriage. Don't guess a passage's content beyond what the frame/description/transcription says.
- `grade`: 2 = the record is squarely what the query asks for; 1 = clearly relevant but partial (e.g. a fragment
  that only touches the topic). Prefer queries you can grade 2.
- `basis`: ≤12 words saying which part of the record justifies the query (e.g. "frame: Talmud Bavli Sukkah 29b";
  "description: letter about poll tax").
- Vary phrasing across records. Don't start every query with "fragment of" or "Genizah". Real users rarely type
  "Cairo Genizah".
- **Skip threshold.** If the record names no specific work, book, tractate, festival, person, place, document
  type or readable content (e.g. only "Talmudic discussion", "Unidentified domain", "Bible: Texts" with no book),
  return `"queries": []` and a short `skip_reason`. A record that names a specific book ("Book of Genesis") is enough.
- **Records containing several texts:** grade 2 when one identified portion is squarely what the query asks for.
  **One-line mentions** (e.g. an account with one line about myrtles) are grade 1 for a topic query, and grade 2
  only when the query asks about that specific item.
- **Inference from content:** if the transcription clearly shows the topic even though the catalogue does not say
  so (e.g. a Hanukkah piyyut catalogued only as "Liturgy"), you may write queries for it. Say so in `basis`
  ("inferred from transcription: בחנוכתה").
- **Contradicting labels:** follow the most specific identification (a named piyyut or tractate beats a generic
  genre or "Document Type").
- For `pool: eval` records, take extra care. These queries become the human-validated evaluation set.
- Output must be valid JSONL, one line per input record, with nothing else in the file.
