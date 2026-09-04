# Design precepts for the Genizah chat

Principles that define what this system is for. They outrank convenience and
should be treated as requirements when designing retrieval, synthesis, or UI
changes — a change that improves a generic RAG metric while weakening these is
the wrong change.

## 1. Primary↔secondary linkage is the product

This is not an everyday RAG pipeline over a book corpus. Its distinguishing
value is that it **connects scholarship to the manuscripts that scholarship was
built on**, in one interface, one click away.

The user experience we are building toward:

- "Goitein wrote about India traders — based on *which* fragments? Show me."
- "Friedman traced the evolution of the ketubba — which ketubbot did he work
  from? Let me open them."
- "Someone traced the development of bentching through zemirot — pull up the
  fragments behind that argument."

Consequences for design:

- **A retrieved page is not the unit of linkage; the work is.** Most secondary
  pages do not themselves print a shelf mark, but the article or book they
  belong to was written about a specific fragment or set of fragments. Surface
  the manuscripts behind the *work*, not only the marks printed on the
  retrieved page. (Some pages are genuinely general surveys; that is the
  minority case, not the design center.)
- **Every manuscript reference that can be resolved must be clickable.** A
  shelf mark rendered as plain text where a document exists is a failure of the
  core promise. A shelf mark linked to the *wrong* document is worse — link
  only on canonical equivalence (see `shelfmarks_equivalent`).
- **Say so when we cannot link.** Cited manuscripts absent from the collection
  are listed plainly and recorded in the missing-fragments worklist
  (`GET /missing-fragments`) so the corpus can grow toward the scholarship.
- **Two linkage paths, use both.** The knowledge graph
  (`(:Scholar)-[:WROTE]->(:BookArticle)-[:REFERENCES]->(:Fragment)`, fragments
  carrying `es_doc_id`) and the bibliography index (`shelf_marks_mentioned`
  across all pages of a work). Neither alone has full coverage.
- **Metadata quality is load-bearing.** Publication references extracted as
  shelf marks ("DJD II, no. 20") break the promise by offering links to things
  that are not manuscripts of this collection. Filter defensively at read time;
  fix at index time.

## 2. Grounded or silent

An answer states what the retrieved sources support, with page-level citation,
or it states plainly that the corpus does not cover the question. It never
pads with adjacent material that happens to have been retrieved, and never
substitutes general knowledge for corpus evidence. An unrelated page is
silence, not a partial answer.

## 3. Verification serves the reader, not the pipeline

Verification exists to make answers trustworthy, not to suppress them.
Contradicted or fabricated content is removed; merely-unverified claims stay
visible and are flagged for the reader with the verifier's reasoning, so the
user keeps both the substance and the means to judge it.

## 4. Retrieval speaks the corpus's language

Scholarship transliterates inconsistently and cites in several conventions.
The system, not the user, is responsible for bridging spellings
(qinot/kinnot/kinot), holiday names (Tisha B'Av / Ninth of Av / תשעה באב), and
shelf-mark dialects (`T-S 12.388` / `T_S_12_388` / institution-prefixed forms).

The same bridge runs outward: the answer speaks the user's language. A Hebrew
question receives a Hebrew answer even when the retrieved scholarship is
English (and vice versa), while quotations keep their source script and shelf
marks stay exactly as written.

## 5. A follow-up is a question about the conversation's subject

"Can you give some samples of what the verses actually state?" carries none of
its topic; searched literally it retrieves whatever mentions verses. The system
carries context forward, so retrieval, relevance checks, and synthesis all work
from the message restated as a standalone question ("the verses of the
Kol-Nidré piyyut in Rylands Genizah Fragment 1"), and the reader sees that
restatement. Two guardrails keep this honest:

- **The restatement may only use what the conversation said.** A small model
  happily invents specifics or copies the previous answer's citations into the
  question; a rewrite that introduces a shelf mark, quotes answer text, or
  drifts from the user's own terms is discarded and the literal message is
  searched instead. Wrong context is worse than no context.
- **A new topic is not a follow-up.** "Tell me about ketubbot" after a Goitein
  exchange is a fresh question and must pass through untouched — otherwise the
  previous subject pollutes every later turn.

The earlier assistant turn is context for resolving references, never a
source: every claim in the new answer still comes from newly retrieved
evidence.
