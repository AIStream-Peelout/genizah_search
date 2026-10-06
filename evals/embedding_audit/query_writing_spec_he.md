# Synthetic query writing spec — Hebrew & transliteration round (v1)

Same task as `query_writing_spec.md`: write search queries that real users of a Cairo Genizah search engine would
type, for which the given catalogue record is a correct answer. **This round adds the queries users write in
Hebrew and in Latin-letter transliteration.** Users ask in both languages, and the retriever must understand
Hebrew questions as well as English ones.

## Input
JSONL records with `doc_id`, `pool`, `stratum`, `topics`, `text`, the same format as before.

## Output
One JSON object per input record, same order, JSONL:
```
{"doc_id": "...", "queries": [
  {"type": "he_implicit", "lang": "he", "text": "...", "grade": 2, "basis": "..."},
  {"type": "he_topical",  "lang": "he", "text": "...", "grade": 2, "basis": "..."},
  {"type": "translit",    "lang": "translit", "text": "...", "grade": 2, "basis": "..."}
], "skip_reason": null}
```

## The three queries
1. **`he_implicit`**: a natural Hebrew question or search phrase that describes the **practice, content or
   question** without naming the work, tractate, festival or topic (the Hebrew counterpart of `implicit`).
   Write it the way a Hebrew-reading user or scholar would type it, often as a full question. Typical length is
   3–10 words.
   - Bavli Sukkah 29b–30a → "האם לולב גזול כשר לצאת בו ידי חובה"
   - Hoshanot → "פיוטים שאומרים בהקפות עם הערבות"
   - Mishneh Torah, Shehitah → "דיני בדיקת הריאה בבהמה שחוטה"
   - Haggadah → "ארבע הקושיות וסיפור יציאת מצרים בליל הסדר"
   - Ketubbah (documentary) → "רשימת נדוניה ומוהר בשטר נישואין"
2. **`he_topical`**: a natural Hebrew search query that **may** name the festival, practice, genre or topic:
   "פיוטים להושענא רבה", "הלכות שבת בגניזה", "מכתבי בקשה לעזרה בתשלום הג'זיה". It must not copy the
   record's Hebrew title line or an incipit of more than 3 words. Vary the phrasing: some questions
   ("מה יש בגניזה על …"), some noun phrases.
3. **`translit`**: a Latin-letter query that uses Hebrew terms the way English-speaking Jewish users type them.
   Mix the conventions across records:
   - Ashkenazi: "hilchos sukkos", "hoshanos", "kesubah", "shechita treifos"
   - Sephardi/modern: "hilchot sukkot", "ketubah", "hoshanot"
   - academic: "hilkhot sukkah", "ketubba", "qedushta"

   It may name the topic. Keep it short (2–6 words).

## Rules (as in the main spec)
- **No invented facts.** Use the record plus general Jewish knowledge only.
- `grade` 2 = squarely relevant; 1 = partial or uncertain.
- `basis` ≤ 12 words.
- **Skip threshold.** Skip records with no specific work, book, tractate, festival, person, place, document type or
  readable content. Return `"queries": []` and a `skip_reason`.
- For Judeo-Arabic/Arabic documentary records, write Hebrew-script queries in Hebrew (e.g. "מכתב מסוחר על משלוח
  פשתן"), not Arabic.
- For `pool: eval` records, take extra care. These become the human-validated evaluation set.
- Output valid JSONL, one line per input record, nothing else.
