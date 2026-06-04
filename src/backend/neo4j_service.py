import logging
import os
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jService:
    """Manages a Neo4j driver and provides query execution helpers."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: Driver | None = None

    def connect(self) -> None:
        """Open the driver connection. Safe to call multiple times."""
        if self._driver is not None:
            return
        auth = (self._user, self._password) if self._password else None
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=auth)
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s (db: %s)", self._uri, self._database)
        except (ServiceUnavailable, AuthError) as exc:
            logger.error("Failed to connect to Neo4j: %s", exc)
            self._driver = None
            raise

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def run_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query and return a list of record dicts."""
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    # ------------------------------------------------------------------
    # Map queries
    # ------------------------------------------------------------------

    def get_map_places(self, include_academic: bool = False) -> list[dict[str, Any]]:
        """
        Geocoded Place nodes filtered by data provenance.
        When include_academic=False, only places with at least one PGP-sourced
        (or null data_sources) connection are returned — places that exist
        solely in academic-extracted data are hidden.
        Counts reflect only the matching source tier.
        """
        cypher = """
        MATCH (pl:Place)
        WHERE pl.lat IS NOT NULL AND pl.lng IS NOT NULL
          AND coalesce(pl.geocode_suspect, false) = false

        OPTIONAL MATCH (f:Fragment)-[rf:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT]->(pl)
        WHERE $include_academic OR rf.data_sources IS NULL OR 'pgp' IN rf.data_sources

        OPTIONAL MATCH (p:Person)-[rp:LIVED_IN|TRAVELED_TO]->(pl)
        WHERE $include_academic OR rp.data_sources IS NULL OR 'pgp' IN rp.data_sources

        WITH pl,
             count(DISTINCT f) AS fragment_count,
             count(DISTINCT p) AS person_count

        // Drop places with no qualifying connections for the current source filter
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
        return self.run_query(cypher, {"include_academic": include_academic})

    def get_map_connections(self, min_connections: int = 2, include_academic: bool = False) -> list[dict[str, Any]]:
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
                description: f.description,
                period:      f.period,
                type:        f.type,
                relations:   [type(r2)],
                certainty:   r2.certainty
            })[..5] AS sample_fragments
        ORDER BY connections DESC
        """
        rows = self.run_query(cypher, {"include_academic": include_academic})
        return [r for r in rows if r["connections"] >= min_connections]

    def get_scholar_detail(self, name: str) -> dict[str, Any] | None:
        """Full detail for a scholar — publications, fragments referenced, places covered."""
        cypher = """
        MATCH (s:Scholar {name: $name})

        // Books / articles they wrote
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

        // Fragments referenced across their work
        OPTIONAL MATCH (s)-[:WROTE]->(b2:BookArticle)-[:REFERENCES]->(f:Fragment)
        WITH s, books,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:   f.canonical_shelfmark,
                description: f.description,
                date_range:  f.date_range,
                relations:   ['REFERENCED']
            })[..8] AS fragments

        // Places their work covers
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
        rows = self.run_query(cypher, {"name": name})
        return rows[0] if rows else None

    def get_institution_detail(self, name: str) -> dict[str, Any] | None:
        """Full detail for an institution — fragments held, scholars, languages."""
        cypher = """
        MATCH (i:Institution {name: $name})

        // Fragment sample + count
        OPTIONAL MATCH (f:Fragment)-[:HELD_AT]->(i)
        WITH i, f
        WITH i,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:    f.canonical_shelfmark,
                description:  f.description,
                date_range:   f.date_range,
                data_sources: f.data_sources,
                relations:    ['HELD_AT']
            })[..10] AS fragments

        // People mentioned in fragments held here
        OPTIONAL MATCH (f2:Fragment)-[:HELD_AT]->(i)
        OPTIONAL MATCH (f2)-[:MENTIONS]->(p:Person)
        WITH i, fragment_count, fragments, collect(DISTINCT p.name)[..15] AS people

        // Scholars who wrote about fragments held here
        OPTIONAL MATCH (f3:Fragment)-[:HELD_AT]->(i)
        OPTIONAL MATCH (f3)<-[:REFERENCES]-(b:BookArticle)<-[:WROTE]-(s:Scholar)
        WITH i, fragment_count, fragments, people, collect(DISTINCT s.name)[..10] AS scholars

        // Languages represented in the collection here
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
        rows = self.run_query(cypher, {"name": name})
        return rows[0] if rows else None

    def get_person_detail(self, name: str, include_academic: bool = False) -> dict[str, Any] | None:
        """Full detail for a person — places, fragments that mention them, books."""
        cypher = """
        MATCH (p:Person {name: $name})

        // Places they lived and traveled (source-filtered)
        OPTIONAL MATCH (p)-[r:LIVED_IN|TRAVELED_TO]->(pl:Place)
        WHERE ($include_academic OR 'pgp' IN r.data_sources)
        WITH p, collect(DISTINCT {
            place:    pl.name,
            country:  pl.country,
            relation: type(r)
        }) AS places

        // Fragments mentioning this person (source-filtered)
        OPTIONAL MATCH (f:Fragment)-[rm:MENTIONS_PERSON]->(p)
        WHERE ($include_academic OR 'pgp' IN rm.data_sources)
        WITH p, places,
            count(DISTINCT f) AS fragment_count,
            collect(DISTINCT {
                shelfmark:         f.canonical_shelfmark,
                description:       f.description,
                period:            f.period,
                type:              f.type,
                has_transcription: f.has_transcription,
                pgpid:             f.pgpid,
                data_sources:      f.data_sources,
                relations:         [{type: 'MENTIONS_PERSON', certainty: coalesce(rm.certainty, 'definite')}]
            })[..10] AS fragments

        // Books that reference those fragments
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
        rows = self.run_query(cypher, {"name": name, "include_academic": include_academic})
        return rows[0] if rows else None

    def get_cross_institution_joins(self) -> list[dict[str, Any]]:
        """
        Fragment pairs linked by JOINED_WITH that are held at different institutions.
        Used to draw cross-institution join lines on the map.
        """
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
                description:  f1.description,
                date_range:   f1.date_range
            })[..5] AS sample_joins
        ORDER BY join_count DESC
        """
        return self.run_query(cypher)

    def get_map_institutions(self) -> list[dict[str, Any]]:
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
        return self.run_query(cypher)

    def get_person_journeys(self, include_academic: bool = False) -> list[dict[str, Any]]:
        """
        All Person → Place connections via LIVED_IN and TRAVELED_TO.
        Returns flat rows; the frontend groups them into per-person arcs.
        When include_academic=False only PGP-sourced relationships are returned.
        """
        cypher = """
        MATCH (p:Person)-[r:TRAVELED_TO|LIVED_IN]->(pl:Place)
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
        return self.run_query(cypher, {"include_academic": include_academic})

    def get_place_detail(self, name: str, include_academic: bool = False) -> dict[str, Any] | None:
        """
        Full detail for a single place.
        Each fragment appears once with all its relation types and certainty values.
        When include_academic=False only PGP-sourced relationships are returned.
        """
        cypher = """
        MATCH (pl:Place {name: $name})

        // Collect every relation + certainty this fragment has to the place
        OPTIONAL MATCH (f:Fragment)-[r:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT|SENT_TO]->(pl)
        WHERE ($include_academic OR r.data_sources IS NULL OR 'pgp' IN r.data_sources)
        WITH pl, f,
             collect({type: type(r), certainty: coalesce(r.certainty, 'definite')}) AS rel_data

        WITH pl,
             collect(DISTINCT CASE WHEN f IS NOT NULL THEN {
                 shelfmark:         f.canonical_shelfmark,
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

        // People — filter by source, include role
        OPTIONAL MATCH (p:Person)-[rp:LIVED_IN|TRAVELED_TO]->(pl)
        WHERE ($include_academic OR rp.data_sources IS NULL OR 'pgp' IN rp.data_sources)
        WITH pl, fragments,
             collect(DISTINCT {name: p.name, role: p.role})[..20] AS people

        // Books
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
        rows = self.run_query(cypher, {"name": name, "include_academic": include_academic})
        return rows[0] if rows else None


def _build_service() -> Neo4jService:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return Neo4jService(uri=uri, user=user, password=password, database=database)


neo4j_service = _build_service()
