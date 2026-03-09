# Cairo Genizah AI Assistant — Claude Code Brief

> **Scope of this brief:** Four concrete fixes to make the assistant functional today. No infrastructure changes. No new services. Prompt fixes, control flow fixes, and display fixes only.

---

## What the Assistant Is (and Isn't)

The assistant is a **scholarly reading companion**. Its job is:

1. Retrieve chunks from the secondary source bibliography index (Elasticsearch)
2. Synthesize those chunks into a cited scholarly response — author, work, page number, direct quotes
3. Identify shelf-marks mentioned *within* those secondary source texts and render them as hyperlinks to fragment catalog pages
4. Append a short list of possibly related catalog entries at the bottom — clickable, no LLM commentary

**It does not search or interpret primary source fragments.** The user has direct access to keyword/semantic/hybrid search for that. The LLM adds no reliable value interpreting fragments with thin metadata or uncertain transcriptions — the hallucination risk is too high. Keep the LLM strictly in the secondary source lane.

---

## Current Broken Behavior (Regression Test Case)

```
User: Can you tell me about Ketubahs in the Cairo Genizah

🧠 Thought Process
The query asks for general scholarly information about ketubbot...

I cannot provide this answer. Referenced shelf marks not in sources:
T-S 12.647, T-S 8.133, T-S 16.198, ms and

Let me search again.
[nothing happens — user is left with a raw error]
```

All four fixes below address aspects of this failure. Use this query as the regression test after each fix.

---

## Fix 1 — Implement the Retry Loop (Priority 1)

**Problem:** Verification failure emits "Let me search again" as a literal text string. No retry actually occurs.

**Fix:** Find the verification failure handler. Replace the text emission with an actual second synthesis call. Pass the original retrieved chunks + the list of flagged invalid items with an instruction to rewrite without them.

Retry prompt addition (append to synthesis prompt on retry):
```
The following items were flagged as unverifiable and must be excluded from your response.
Do not reference them in any form:
{invalid_items}

Rewrite the response using only the remaining retrieved sources.
```

Cap at 2 retry attempts. If both fail, return the graceful fallback message (see Fix 1b).

**Fix 1b — Graceful fallback message:**
```
I wasn't able to construct a fully verified response for this query. 
Please try rephrasing or narrowing your question, or use the search 
panel directly to explore relevant sources.
```

This message replaces the raw error string in all failure cases. The raw error string must never reach the user under any circumstances.

---

## Fix 2 — Fix Verification False Positives (Priority 2)

**Problem:** The verification step checks whether cited shelf-marks exist in the fragment index. They don't — they come from secondary source text, not from a fragment retrieval. This causes valid responses to be rejected.

**Fix:** Change the verification rule. A shelf-mark is valid if it appears in the text of any retrieved secondary source chunk. That's it. Remove the fragment index check entirely for now.

Concretely, the verification check should be:
- ✅ Shelf-mark appears in retrieved bibliography chunk text → valid
- ✅ Page citation matches a retrieved chunk → valid  
- ❌ Shelf-mark not found anywhere in retrieved chunks → flag as invalid

Do not check whether shelf-marks exist in the fragment database. That is a separate resolution step that isn't built yet.

---

## Fix 3 — Rewrite the Synthesis Prompt (Priority 3)

**Problem:** The synthesis LLM is treating primary source shelf-marks as documents to analyze rather than references to cite. It's also not prioritizing direct quotation from secondary sources.

**Replace or augment the existing synthesis prompt with these instructions:**

```
You are a scholarly research assistant specializing in Cairo Genizah studies.

Your inputs are chunks retrieved from academic secondary sources (books and articles 
about the Genizah). Your job is to synthesize these sources into a coherent scholarly 
response with precise citations.

Rules:
1. Lead with what scholars have written. Quote directly where it strengthens the response.
   Format quotes as: Author (Year), p. X: "quote text"
2. Every factual claim must cite a specific retrieved source with page number.
   Do not draw on background knowledge — only the retrieved chunks.
3. When a shelf-mark appears in a retrieved source, include it exactly as written.
   Treat it as a reference to be cited, not a document to analyze or describe.
   Do not add any information about what the fragment contains beyond what the 
   source text says.
4. Do not invent shelf-marks, page numbers, or citations. If the retrieved sources 
   don't cover an aspect of the query, say so explicitly rather than filling the gap.
5. If retrieved sources are sparse, return what you have with honest attribution 
   rather than padding with general knowledge.
```

---

## Fix 4 — Hide Thought Process Block on Error (Priority 4)

**Problem:** When the pipeline fails, users see the thought process block followed by a raw error. This looks broken and confusing.

**Fix:** Gate the thought process block display on pipeline success. If the pipeline ends in any error state — verification failure, retry exhaustion, exception — suppress the thought process block entirely. The user should only ever see either a clean response or the graceful fallback message from Fix 1b.

This is a display/frontend fix. The thought process can still be logged server-side for debugging.

---

## Regression Test Sequence

After all four fixes, test in order:

1. `"Can you tell me about Ketubahs in the Cairo Genizah"` — known failure case, should now return a clean synthesized response with cited secondary sources
2. A query where sources are genuinely sparse — should return an honest "sources are limited" response, not a hallucinated one
3. A query with a well-known fragment like `T-S 13J20.4` — shelf-mark should appear as a link if it's in the retrieved secondary source text, not as LLM commentary about the fragment

---

## What Not to Touch

- Elasticsearch schema, mappings, or indexing pipeline
- Embedding model or embedding microservice
- The user-facing search panel (semantic/keyword/hybrid) — this is separate from the assistant
- Any fragment interpretation or primary source synthesis logic — the scope of the assistant is secondary sources only for now

---

## Repo Structure Note

- `genizah-search` — web app: frontend, backend, embedding microservice, Elasticsearch. **This is where the fixes live.**
- `historical-document-analysis` — bibliography and index building pipelines. **Do not touch for these fixes.**

---

## Glossary

| Term | Meaning |
|------|---------|
| Shelf-mark | Archival call number for a fragment, e.g. `T-S 12.647` |
| T-S | Taylor-Schechter — Cambridge collection prefix |
| ENA | Elkan Nathan Adler — JTS collection prefix |
| Ketubah (pl. ketubboth) | Jewish marriage contract — major Genizah document type |
| Goitein | S.D. Goitein — preeminent Genizah scholar, *A Mediterranean Society* (6 vols.) |
| FGP | Friedberg Genizah Project — main digitization and bibliography source |
| RAG | Retrieval-Augmented Generation — LLM responses grounded in retrieved documents |

---

*Last updated: March 2026. Maintained by Isaac.*