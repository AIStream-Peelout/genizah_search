import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


def build_direct_work_link(record: dict[str, Any] | None) -> str | None:
    """Build the best direct link to a work from its BookArticle record.

    Priority: DOI, then ISBN (a real WorldCat item page), then a stored
    landing page. Catalog *searches* are deliberately not produced here — a
    search is a fallback the caller may add, not a direct link.

    :param record: BookArticle record from :meth:`Neo4jService.find_book_article`.
    :returns: An absolute URL, or ``None`` when the record has no identifier.
    :rtype: str | None
    """
    if not record:
        return None
    doi = str(record.get("doi") or "").strip().rstrip(".,;")
    if doi.lower().startswith("10."):
        return f"https://doi.org/{doi}"
    isbn = re.sub(r"[^0-9Xx]", "", str(record.get("isbn") or ""))
    if len(isbn) in (10, 13):
        return f"https://search.worldcat.org/isbn/{isbn.upper()}"
    url = str(record.get("url") or "").strip().rstrip(".,;")
    return url or None


class Neo4jService:
    """Manages an async Neo4j driver and provides async query execution helpers."""

    SCHOLAR_NAME_ALIASES = {
        "s d goitein": "Shelomo Dov Goitein",
        "sd goitein": "Shelomo Dov Goitein",
        "goitein s d": "Shelomo Dov Goitein",
        "s d fritz goitein": "Shelomo Dov Goitein",
    }

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Open the driver connection. Safe to call multiple times."""
        if self._driver is not None:
            return
        auth = (self._user, self._password) if self._password else None
        try:
            self._driver = AsyncGraphDatabase.driver(self._uri, auth=auth)
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s (db: %s)", self._uri, self._database)
        except (ServiceUnavailable, AuthError) as exc:
            logger.error("Failed to connect to Neo4j: %s", exc)
            self._driver = None
            raise

    async def close(self) -> None:
        """Close the driver connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    async def run_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return a list of record dicts."""
        if self._driver is None:
            await self.connect()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, parameters or {})
            return await result.data()

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """Normalize a graph entity name for conservative fuzzy comparison.

        :param name: The entity name to normalize.
        :returns: A lowercase, whitespace-normalized representation.
        :rtype: str
        """
        normalized = re.sub(r"[^\w\s]", " ", name.lower(), flags=re.UNICODE)
        return " ".join(normalized.split())

    @classmethod
    def normalize_scholar_query(cls, name: str) -> str:
        """Expand a conservative set of common scholar-name abbreviations.

        This is a read-time compatibility layer for user queries, not an
        attempt to merge or rewrite graph nodes. Only unambiguous, explicitly
        curated aliases belong here.

        :param name: Scholar name supplied by the planner or user.
        :returns: Canonical query name when a curated alias exists.
        :rtype: str
        """
        normalized = cls._normalize_entity_name(name)
        return cls.SCHOLAR_NAME_ALIASES.get(normalized, name)

    async def resolve_scholars_in_text(self, text: str, limit: int = 5) -> list[dict[str, Any]]:
        """Resolve scholar names that occur in natural-language query text.

        This deliberately scans the comparatively small Scholar label instead
        of allowing an LLM to invent a name for a Cypher query. Candidate names
        must have all meaningful name tokens present in the user's text; Python
        then ranks them so longer, more specific matches win.

        :param text: Natural-language user query that may contain a scholar name.
        :param limit: Maximum number of ranked candidates to return.
        :returns: Candidate scholars with canonical names and match scores.
        :rtype: list[dict[str, Any]]
        """
        if not text.strip() or limit < 1:
            return []

        cypher = """
        MATCH (s:Scholar)
        WITH s,
             [token IN split(toLower(s.name), ' ')
              WHERE size(replace(replace(token, '.', ''), ',', '')) > 2 |
                    replace(replace(token, '.', ''), ',', '')] AS name_tokens
        WHERE size(name_tokens) > 0
          AND all(token IN name_tokens WHERE toLower($text) CONTAINS token)
        RETURN s.name AS name, s.data_sources AS data_sources
        LIMIT 50
        """
        rows = await self.run_query(cypher, {"text": text})
        normalized_text = self._normalize_entity_name(text)
        ranked: list[dict[str, Any]] = []

        for row in rows:
            name = row.get("name")
            if not name:
                continue
            normalized_name = self._normalize_entity_name(name)
            name_tokens = [token for token in normalized_name.split() if len(token) > 2]
            if not name_tokens:
                continue

            token_coverage = sum(token in normalized_text for token in name_tokens) / len(name_tokens)
            compact_query = " ".join(
                token for token in normalized_text.split() if token in set(name_tokens)
            )
            sequence_score = SequenceMatcher(None, normalized_name, compact_query).ratio()
            specificity = min(len(normalized_name) / max(len(normalized_text), 1), 1.0)
            score = round((0.65 * token_coverage) + (0.25 * sequence_score) + (0.10 * specificity), 4)
            ranked.append({
                "name": name,
                "data_sources": row.get("data_sources") or [],
                "match_score": score,
            })

        ranked.sort(key=lambda candidate: (-candidate["match_score"], -len(candidate["name"])))
        return ranked[:limit]

    async def find_scholars(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find canonical Scholar nodes for a supplied person name.

        :param name: Scholar name or name variant supplied by a router/user.
        :param limit: Maximum number of ranked candidates to return.
        :returns: Candidate scholars with canonical names and match scores.
        :rtype: list[dict[str, Any]]
        """
        normalized_name = self._normalize_entity_name(self.normalize_scholar_query(name))
        tokens = [token for token in normalized_name.split() if len(token) > 1]
        if not tokens or limit < 1:
            return []

        cypher = """
        MATCH (s:Scholar)
        WHERE all(token IN $tokens WHERE toLower(s.name) CONTAINS token)
        RETURN s.name AS name, s.data_sources AS data_sources,
               size([(s)-[:WROTE]->(b) | b]) AS works_count
        LIMIT 50
        """
        rows = await self.run_query(cypher, {"tokens": tokens})
        ranked: list[dict[str, Any]] = []
        for row in rows:
            candidate_name = row.get("name")
            if not candidate_name:
                continue
            candidate_normalized = self._normalize_entity_name(candidate_name)
            score = SequenceMatcher(None, normalized_name, candidate_normalized).ratio()
            if set(tokens).issubset(set(candidate_normalized.split())):
                score = min(1.0, score + 0.1)
            ranked.append({
                "name": candidate_name,
                "data_sources": row.get("data_sources") or [],
                "match_score": round(score, 4),
                "works_count": int(row.get("works_count") or 0),
            })

        # Among plausible name matches, connectivity decides: the graph holds
        # duplicate and skeletal scholar nodes ("S.D. Goitein" with 0 works
        # beside "Shelomo Dov Goitein" with 106), and a profile built from an
        # empty node is worthless even when its name string matches best.
        plausible = [c for c in ranked if c["match_score"] >= 0.55]
        if plausible:
            plausible.sort(key=lambda c: (-c["works_count"], -c["match_score"], len(c["name"])))
            rest = sorted(
                (c for c in ranked if c["match_score"] < 0.55),
                key=lambda c: (-c["match_score"], len(c["name"])),
            )
            return (plausible + rest)[:limit]
        ranked.sort(key=lambda candidate: (-candidate["match_score"], len(candidate["name"])))
        return ranked[:limit]

    async def find_book_article(self, title: str) -> dict[str, Any] | None:
        """Find the best BookArticle record for a work title, merging duplicates.

        The graph contains duplicate BookArticle nodes with complementary
        metadata (one may carry the publisher, another the authors), so fields
        from near-identical titles are coalesced into one record.

        :param title: Work title as cited in an answer or bibliography entry.
        :returns: Merged publication metadata, or ``None`` when no plausible
            match exists.
        :rtype: dict[str, Any] | None
        """
        # Index titles and graph titles diverge in three regular ways, and a
        # single all-tokens-must-match probe fails on each: bilingual titles
        # ("Ginzei Kedem / גנזי קדם" vs the graph's Hebrew-only node),
        # differing subtitles ("…: The Kettubba texts" vs "…: A Cairo Geniza
        # Study"), and transliteration variance (Kedem/Qedem). Probe each
        # language side separately and require only a majority of tokens,
        # leaning on similarity ranking and the acceptance floor for precision.
        probes = [part.strip() for part in str(title).split(" / ") if part.strip()]
        if str(title).strip() not in probes:
            probes.insert(0, str(title).strip())

        token_sets: list[list[str]] = []
        for probe in probes[:3]:
            tokens = [t for t in self._normalize_entity_name(probe).split() if len(t) > 3][:5]
            if tokens:
                token_sets.append(tokens)
        if not token_sets:
            return None

        cypher = """
        UNWIND range(0, size($token_sets) - 1) AS set_index
        WITH $token_sets[set_index] AS tokens, $min_matches[set_index] AS needed
        MATCH (b:BookArticle)
        WHERE size([t IN tokens WHERE toLower(b.title) CONTAINS t]) >= needed
        WITH DISTINCT b
        OPTIONAL MATCH (s:Scholar)-[:WROTE]->(b)
        RETURN b.title AS title, b.year AS year, b.journal AS journal,
               b.publisher AS publisher, b.doi AS doi, b.url AS url,
               b.isbn AS isbn, b.volume AS volume, b.pages AS pages,
               b.citation AS citation, b.article_id AS article_id,
               collect(DISTINCT s.name) AS authors
        LIMIT 60
        """
        rows = await self.run_query(cypher, {
            "token_sets": token_sets,
            # Majority of tokens, at least one; short probes lean on similarity.
            "min_matches": [max(1, (len(t) + 1) // 2) for t in token_sets],
        })

        normalized_probes = [self._normalize_entity_name(p) for p in probes[:3]]
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            candidate_title = row.get("title")
            if not candidate_title:
                continue
            normalized_candidate = self._normalize_entity_name(candidate_title)
            # Score against whichever language side matches best.
            similarity = max(
                SequenceMatcher(None, probe, normalized_candidate).ratio()
                for probe in normalized_probes
            )
            scored.append((similarity, row))
        if not scored:
            return None
        normalized_query = normalized_probes[0]
        scored.sort(key=lambda item: -item[0])
        best_similarity, best = scored[0]
        if best_similarity < 0.45:
            return None

        merged: dict[str, Any] = dict(best)
        authors: list[str] = list(best.get("authors") or [])
        for similarity, row in scored[1:]:
            if similarity < best_similarity - 0.05:
                break
            for field in (
                "year", "journal", "publisher", "doi", "url", "isbn",
                "volume", "pages", "citation",
            ):
                if merged.get(field) in (None, "") and row.get(field) not in (None, ""):
                    merged[field] = row[field]
            for author in row.get("authors") or []:
                if author not in authors:
                    authors.append(author)
        merged["authors"] = authors
        if merged.get("year") is not None:
            try:
                merged["year"] = int(float(merged["year"]))
            except (TypeError, ValueError):
                pass
        merged["match_score"] = round(best_similarity, 4)
        return merged

    async def get_fragments_for_works(
        self,
        titles: list[str],
        per_work_limit: int = 12,
    ) -> dict[str, list[dict[str, Any]]]:
        """Find the manuscripts each scholarly work is built on.

        A retrieved secondary-source page often prints no shelf mark, but the
        work it belongs to was written about specific fragments. Those
        REFERENCES edges are what let a reader jump from an argument to the
        manuscripts behind it.

        :param titles: Work titles as they appear in the bibliography index.
        :param per_work_limit: Maximum fragments to return per work.
        :returns: Mapping of the supplied title to its referenced fragments.
        :rtype: dict[str, list[dict[str, Any]]]
        """
        if not titles:
            return {}
        cypher = """
        UNWIND $titles AS wanted
        MATCH (b:BookArticle)
        WHERE toLower(b.title) CONTAINS toLower(wanted)
           OR toLower(wanted) CONTAINS toLower(b.title)
        MATCH (b)-[:REFERENCES]->(f:Fragment)
        WHERE f.es_doc_id IS NOT NULL
        WITH wanted, f
        RETURN wanted AS title,
               collect(DISTINCT {
                   es_doc_id: f.es_doc_id,
                   canonical_shelfmark: f.canonical_shelfmark
               })[..$per_work_limit] AS fragments
        """
        # Match on a distinctive leading portion: index titles carry subtitles
        # and editor suffixes that graph titles may lack.
        probes = [" ".join(title.split()[:6]) for title in titles if title]
        rows = await self.run_query(
            cypher, {"titles": probes, "per_work_limit": per_work_limit}
        )
        by_probe = {row["title"]: row["fragments"] for row in rows}
        return {
            title: by_probe.get(" ".join(title.split()[:6]), [])
            for title in titles
            if title
        }

    async def get_scholar_rag_evidence(self, name: str) -> dict[str, Any] | None:
        """Return a bounded, provenance-preserving scholar graph neighborhood.

        The result is designed for RAG context rather than map display. It keeps
        graph facts structured, limits fragment samples, and reports aggregate
        counts separately so synthesis does not confuse a sample with the full
        neighborhood.

        :param name: Canonical Scholar.name value returned by a resolver.
        :returns: Structured scholar, work, fragment, and relationship evidence.
        :rtype: dict[str, Any] | None
        """
        profile_rows = await self.run_query(
            """
            MATCH (s:Scholar {name: $name})
            RETURN s.name AS name,
                   s.data_sources AS data_sources,
                   s.source_books AS source_books,
                   s.position AS position
            """,
            {"name": name},
        )
        if not profile_rows:
            return None

        work_rows = await self.run_query(
            """
            MATCH (s:Scholar {name: $name})-[:WROTE]->(b:BookArticle)
            OPTIONAL MATCH (b)-[:REFERENCES]->(f:Fragment)
            RETURN b.article_id AS article_id,
                   b.title AS title,
                   b.year AS year,
                   b.journal AS journal,
                   b.publisher AS publisher,
                   b.doi AS doi,
                   b.has_local_copy AS has_local_copy,
                   b.data_sources AS data_sources,
                   count(DISTINCT f) AS referenced_fragment_count,
                   collect(DISTINCT CASE WHEN f IS NULL THEN null ELSE {
                       shelfmark: f.canonical_shelfmark,
                       es_doc_id: f.es_doc_id,
                       description: f.description
                   } END)[..8] AS referenced_fragment_samples
            ORDER BY referenced_fragment_count DESC, b.year DESC, b.title
            """,
            {"name": name},
        )

        studied_rows = await self.run_query(
            """
            MATCH (s:Scholar {name: $name})-[:STUDIED]->(f:Fragment)
            RETURN count(DISTINCT f) AS studied_fragment_count,
                   collect(DISTINCT {
                       shelfmark: f.canonical_shelfmark,
                       es_doc_id: f.es_doc_id,
                       description: f.description
                   })[..15] AS studied_fragment_samples
            """,
            {"name": name},
        )

        relationship_rows = await self.run_query(
            """
            MATCH (s:Scholar {name: $name})-[r:COLLABORATED_WITH|AFFILIATED_WITH]-(n)
            RETURN type(r) AS relationship,
                   labels(n) AS labels,
                   coalesce(n.name, n.title) AS name,
                   n.data_sources AS data_sources
            ORDER BY relationship, name
            LIMIT 30
            """,
            {"name": name},
        )

        profile = profile_rows[0]
        studied = studied_rows[0] if studied_rows else {
            "studied_fragment_count": 0,
            "studied_fragment_samples": [],
        }
        for work in work_rows:
            work["referenced_fragment_samples"] = [
                fragment for fragment in (work.get("referenced_fragment_samples") or []) if fragment
            ]

        return {
            "evidence_type": "neo4j_scholar_neighborhood",
            "scholar": profile,
            "works": work_rows,
            "studied_fragment_count": studied.get("studied_fragment_count") or 0,
            "studied_fragment_samples": studied.get("studied_fragment_samples") or [],
            "relationships": relationship_rows,
        }

    # ------------------------------------------------------------------
    # Map queries
    # ------------------------------------------------------------------

    async def get_map_places(self, include_academic: bool = False) -> list[dict[str, Any]]:
        """
        Geocoded Place nodes filtered by data provenance.
        When include_academic=False, only places with at least one PGP-sourced
        (or null data_sources) connection are returned.
        Counts reflect only the matching source tier.
        """
        cypher = """
        MATCH (pl:Place)
        WHERE pl.lat IS NOT NULL AND pl.lng IS NOT NULL
          AND coalesce(pl.geocode_suspect, false) = false

        OPTIONAL MATCH (f:Fragment)-[rf:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT]->(pl)
        WHERE $include_academic OR rf.data_sources IS NULL OR 'pgp' IN rf.data_sources

        OPTIONAL MATCH (p:Person)-[rp:LIVED_IN|TRAVELED_TO|ORIGINATED_FROM]->(pl)
        WHERE $include_academic OR rp.data_sources IS NULL OR 'pgp' IN rp.data_sources

        WITH pl,
             count(DISTINCT f) AS fragment_count,
             count(DISTINCT p) AS person_count

        WHERE fragment_count > 0 OR person_count > 0

        RETURN
            pl.name           AS name,
            pl.lat            AS lat,
            pl.lng            AS lng,
            pl.country        AS country,
            pl.region         AS region,
            pl.name_variants  AS name_variants,
            fragment_count,
            person_count
        ORDER BY fragment_count DESC
        """
        return await self.run_query(cypher, {"include_academic": include_academic})

    async def get_map_connections(self, min_connections: int = 2, include_academic: bool = False) -> list[dict[str, Any]]:
        """
        Place-to-place connections via shared fragments.
        When include_academic=False only PGP-sourced relationships are included.
        """
        cypher = """
        MATCH (pl1:Place)<-[r1:ORIGINATED_FROM]-(f:Fragment)-[r2:MENTIONS_PLACE|SENT_TO]->(pl2:Place)
        WHERE pl1 <> pl2
          AND pl1.lat IS NOT NULL AND pl1.lng IS NOT NULL
          AND pl2.lat IS NOT NULL AND pl2.lng IS NOT NULL
          AND ($include_academic OR r1.data_sources IS NULL OR 'pgp' IN r1.data_sources)
          AND ($include_academic OR r2.data_sources IS NULL OR 'pgp' IN r2.data_sources)
        RETURN
            pl1.name AS source,
            pl1.lat  AS source_lat,
            pl1.lng  AS source_lng,
            pl2.name AS target,
            pl2.lat  AS target_lat,
            pl2.lng  AS target_lng,
            count(DISTINCT f) AS connections,
            collect(DISTINCT {
                shelfmark:   f.canonical_shelfmark,
                es_doc_id:   f.es_doc_id,
                description: f.description,
                period:      f.period,
                type:        f.type,
                relations:   [type(r2)],
                certainty:   r2.certainty
            })[..5] AS sample_fragments
        ORDER BY connections DESC
        """
        rows = await self.run_query(cypher, {"include_academic": include_academic})
        return [r for r in rows if r["connections"] >= min_connections]

    async def get_scholar_detail(self, name: str) -> dict[str, Any] | None:
        """Full detail for a scholar — publications, fragments referenced, places covered."""
        cypher = """
        MATCH (s:Scholar {name: $name})

        OPTIONAL MATCH (s)-[:WROTE]->(b:BookArticle)
        WITH s, collect(DISTINCT {
            article_id:     b.article_id,
            title:          b.title,
            year:           b.year,
            journal:        b.journal,
            publisher:      b.publisher,
            doi:            b.doi,
            has_local_copy: b.has_local_copy
        }) AS books

        OPTIONAL MATCH (s)-[:WROTE]->(b2:BookArticle)-[:REFERENCES]->(f:Fragment)
        WITH s, books,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:   f.canonical_shelfmark,
                es_doc_id:   f.es_doc_id,
                description: f.description,
                date_range:  f.date_range,
                relations:   ['REFERENCED']
            })[..8] AS fragments

        OPTIONAL MATCH (s)-[:WROTE]->(b3:BookArticle)
        OPTIONAL MATCH (b3)-[:MENTIONED_IN]-(pl:Place)

        RETURN
            s.name           AS name,
            s.data_sources   AS data_sources,
            books,
            fragment_count,
            fragments,
            collect(DISTINCT pl.name)[..10] AS places
        """
        rows = await self.run_query(cypher, {"name": name})
        return rows[0] if rows else None

    async def get_institution_detail(self, name: str) -> dict[str, Any] | None:
        """Full detail for an institution — fragments held, scholars, languages."""
        cypher = """
        MATCH (i:Institution {name: $name})

        OPTIONAL MATCH (f:Fragment)-[:HELD_AT]->(i)
        WITH i, f
        WITH i,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:    f.canonical_shelfmark,
                es_doc_id:    f.es_doc_id,
                description:  f.description,
                date_range:   f.date_range,
                data_sources: f.data_sources,
                relations:    ['HELD_AT']
            })[..10] AS fragments

        OPTIONAL MATCH (f2:Fragment)-[:HELD_AT]->(i)
        OPTIONAL MATCH (f2)-[:MENTIONS]->(p:Person)
        WITH i, fragment_count, fragments, collect(DISTINCT p.name)[..15] AS people

        OPTIONAL MATCH (f3:Fragment)-[:HELD_AT]->(i)
        OPTIONAL MATCH (f3)<-[:REFERENCES]-(b:BookArticle)<-[:WROTE]-(s:Scholar)
        WITH i, fragment_count, fragments, people, collect(DISTINCT s.name)[..10] AS scholars

        OPTIONAL MATCH (f4:Fragment)-[:HELD_AT]->(i)
        OPTIONAL MATCH (f4)-[:WRITTEN_IN]->(l:Language)

        RETURN
            i.name                   AS name,
            i.lat                    AS lat,
            i.lng                    AS lng,
            coalesce(i.country, '')  AS country,
            fragment_count,
            fragments,
            people,
            scholars,
            collect(DISTINCT l.name)[..8] AS languages
        """
        rows = await self.run_query(cypher, {"name": name})
        return rows[0] if rows else None

    async def get_person_detail(self, name: str, include_academic: bool = False) -> dict[str, Any] | None:
        """Full detail for a person — places, fragments that mention them, books."""
        cypher = """
        MATCH (p:Person {name: $name})

        OPTIONAL MATCH (p)-[r:LIVED_IN|TRAVELED_TO|ORIGINATED_FROM]->(pl:Place)
        WHERE ($include_academic OR r.data_sources IS NULL OR 'pgp' IN r.data_sources)
        WITH p, collect(DISTINCT {
            place:    pl.name,
            country:  pl.country,
            relation: type(r)
        }) AS places

        OPTIONAL MATCH (f:Fragment)-[rm:MENTIONS_PERSON]->(p)
        WHERE ($include_academic OR rm.data_sources IS NULL OR 'pgp' IN rm.data_sources)
        WITH p, places,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:         f.canonical_shelfmark,
                es_doc_id:         f.es_doc_id,
                description:       f.description,
                period:            f.period,
                type:              f.type,
                has_transcription: f.has_transcription,
                pgpid:             f.pgpid,
                data_sources:      f.data_sources,
                relations:         [{type: 'MENTIONS_PERSON', certainty: coalesce(rm.certainty, 'definite')}]
            })[..10] AS fragments

        OPTIONAL MATCH (f2:Fragment)-[:MENTIONS_PERSON]->(p)
        OPTIONAL MATCH (f2)<-[:REFERENCES]-(b:BookArticle)

        RETURN
            p.name           AS name,
            p.role           AS role,
            p.description    AS description,
            p.home_base      AS home_base,
            p.gender         AS gender,
            p.data_sources   AS data_sources,
            places,
            fragment_count,
            fragments,
            collect(DISTINCT b.title)[..5] AS books
        """
        rows = await self.run_query(cypher, {"name": name, "include_academic": include_academic})
        return rows[0] if rows else None

    async def get_cross_institution_joins(self) -> list[dict[str, Any]]:
        """Fragment pairs linked by JOINED_WITH that are held at different institutions."""
        cypher = """
        MATCH (f1:Fragment)-[:JOINED_WITH]->(f2:Fragment)
        MATCH (f1)-[:HELD_AT]->(i1:Institution)
        MATCH (f2)-[:HELD_AT]->(i2:Institution)
        WHERE i1 <> i2
          AND i1.lat IS NOT NULL AND i1.lng IS NOT NULL
          AND i2.lat IS NOT NULL AND i2.lng IS NOT NULL
        RETURN
            i1.name AS source,
            i1.lat  AS source_lat,
            i1.lng  AS source_lng,
            i2.name AS target,
            i2.lat  AS target_lat,
            i2.lng  AS target_lng,
            count(DISTINCT f1)  AS join_count,
            collect(DISTINCT {
                shelfmark1:   f1.canonical_shelfmark,
                shelfmark2:   f2.canonical_shelfmark,
                es_doc_id1:   f1.es_doc_id,
                es_doc_id2:   f2.es_doc_id,
                description:  f1.description,
                date_range:   f1.date_range
            })[..5] AS sample_joins
        ORDER BY join_count DESC
        """
        return await self.run_query(cypher)

    async def get_map_institutions(self) -> list[dict[str, Any]]:
        """Institution nodes that have been geocoded."""
        cypher = """
        MATCH (i:Institution)
        WHERE i.lat IS NOT NULL AND i.lng IS NOT NULL
        OPTIONAL MATCH (f:Fragment)-[:HELD_AT]->(i)
        RETURN
            i.name            AS name,
            i.lat             AS lat,
            i.lng             AS lng,
            coalesce(i.country, '') AS country,
            coalesce(i.region, '')  AS region,
            count(DISTINCT f) AS fragment_count
        ORDER BY fragment_count DESC
        """
        return await self.run_query(cypher)

    async def get_person_journeys(self, include_academic: bool = False) -> list[dict[str, Any]]:
        """
        All Person → Place connections via LIVED_IN and TRAVELED_TO.
        Returns flat rows; the frontend groups them into per-person arcs.
        """
        cypher = """
        MATCH (p:Person)-[r:LIVED_IN|TRAVELED_TO|ORIGINATED_FROM]->(pl:Place)
        WHERE pl.lat IS NOT NULL AND pl.lng IS NOT NULL
          AND ($include_academic OR r.data_sources IS NULL OR 'pgp' IN r.data_sources)
        RETURN
            p.name        AS person,
            p.role        AS role,
            type(r)       AS relation_type,
            pl.name       AS place,
            pl.lat        AS lat,
            pl.lng        AS lng
        ORDER BY p.name
        """
        return await self.run_query(cypher, {"include_academic": include_academic})

    async def get_place_detail(self, name: str, include_academic: bool = False) -> dict[str, Any] | None:
        """
        Full detail for a single place.
        Each fragment appears once with all its relation types and certainty values.
        When include_academic=False only PGP-sourced relationships are returned.
        """
        cypher = """
        MATCH (pl:Place {name: $name})

        OPTIONAL MATCH (f:Fragment)-[r:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT|SENT_TO]->(pl)
        WHERE ($include_academic OR r.data_sources IS NULL OR 'pgp' IN r.data_sources)
        WITH pl, f,
             collect({type: type(r), certainty: coalesce(r.certainty, 'definite')}) AS rel_data

        WITH pl,
             collect(DISTINCT CASE WHEN f IS NOT NULL THEN {
                 shelfmark:         f.canonical_shelfmark,
                 es_doc_id:         f.es_doc_id,
                 description:       f.description,
                 period:            f.period,
                 type:              f.type,
                 has_transcription: f.has_transcription,
                 has_translation:   f.has_translation,
                 pgpid:             f.pgpid,
                 url:               f.url,
                 data_sources:      f.data_sources,
                 relations:         rel_data
             } END)[..15] AS fragments

        OPTIONAL MATCH (p:Person)-[rp:LIVED_IN|TRAVELED_TO|ORIGINATED_FROM]->(pl)
        WHERE ($include_academic OR rp.data_sources IS NULL OR 'pgp' IN rp.data_sources)
        WITH pl, fragments,
             collect(DISTINCT {name: p.name, role: p.role})[..20] AS people

        OPTIONAL MATCH (b:BookArticle)-[:MENTIONED_IN]->(pl)

        RETURN
            pl.name          AS name,
            pl.lat           AS lat,
            pl.lng           AS lng,
            pl.country       AS country,
            pl.region        AS region,
            pl.name_variants AS name_variants,
            fragments,
            people,
            collect(DISTINCT b.title)[..10] AS books
        """
        rows = await self.run_query(cypher, {"name": name, "include_academic": include_academic})
        return rows[0] if rows else None


def _build_service() -> Neo4jService:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return Neo4jService(uri=uri, user=user, password=password, database=database)


neo4j_service = _build_service()
