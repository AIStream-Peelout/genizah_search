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

    def get_map_places(self) -> list[dict[str, Any]]:
        """All geocoded Place nodes with fragment and person counts."""
        cypher = """
        MATCH (pl:Place)
        WHERE pl.lat IS NOT NULL AND pl.lng IS NOT NULL
        OPTIONAL MATCH (f:Fragment)-[:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT]->(pl)
        OPTIONAL MATCH (p:Person)-[:LIVED_IN|TRAVELED_TO]->(pl)
        RETURN
            pl.name           AS name,
            pl.lat            AS lat,
            pl.lng            AS lng,
            pl.country        AS country,
            pl.region         AS region,
            pl.name_variants  AS name_variants,
            count(DISTINCT f) AS fragment_count,
            count(DISTINCT p) AS person_count
        ORDER BY fragment_count DESC
        """
        return self.run_query(cypher)

    def get_map_connections(self, min_connections: int = 2) -> list[dict[str, Any]]:
        """
        Place-to-place connections via shared fragments.
        Returns connection count plus up to 5 sample fragments so the
        frontend can show meaningful popup content on click.
        """
        cypher = """
        MATCH (pl1:Place)<-[:ORIGINATED_FROM]-(f:Fragment)-[r2:MENTIONS_PLACE|SENT_TO]->(pl2:Place)
        WHERE pl1 <> pl2
          AND pl1.lat IS NOT NULL AND pl1.lng IS NOT NULL
          AND pl2.lat IS NOT NULL AND pl2.lng IS NOT NULL
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
                date_range:  f.date_range,
                relations:   [type(r2)]
            })[..5] AS sample_fragments
        ORDER BY connections DESC
        """
        rows = self.run_query(cypher)
        return [r for r in rows if r["connections"] >= min_connections]

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

    def get_person_journeys(self) -> list[dict[str, Any]]:
        """
        All Person → Place connections via LIVED_IN and TRAVELED_TO.
        Returns flat rows; the frontend groups them into per-person arcs.
        """
        cypher = """
        MATCH (p:Person)-[r:TRAVELED_TO|LIVED_IN]->(pl:Place)
        WHERE pl.lat IS NOT NULL AND pl.lng IS NOT NULL
        RETURN
            p.name        AS person,
            type(r)       AS relation_type,
            pl.name       AS place,
            pl.lat        AS lat,
            pl.lng        AS lng
        ORDER BY p.name
        """
        return self.run_query(cypher)

    def get_place_detail(self, name: str) -> dict[str, Any] | None:
        """
        Full detail for a single place.
        Fragments are grouped so that all relationship types a fragment has
        to this place (ORIGINATED_FROM, MENTIONS_PLACE, WRITTEN_AT, SENT_TO)
        are collected into a 'relations' array rather than duplicating the row.
        """
        cypher = """
        MATCH (pl:Place {name: $name})

        // Collect every relation type this fragment has to the place in one pass
        OPTIONAL MATCH (f:Fragment)-[r:ORIGINATED_FROM|MENTIONS_PLACE|WRITTEN_AT|SENT_TO]->(pl)
        WITH pl, f, collect(DISTINCT type(r)) AS relations

        // Roll up into one row per place with fragment list
        WITH pl,
             collect(DISTINCT CASE WHEN f IS NOT NULL THEN {
                 shelfmark:    f.canonical_shelfmark,
                 description:  f.description,
                 date_range:   f.date_range,
                 data_sources: f.data_sources,
                 relations:    relations
             } END)[..15] AS fragments

        // People — separate pass to avoid cross-product
        OPTIONAL MATCH (p:Person)-[:LIVED_IN|TRAVELED_TO]->(pl)
        WITH pl, fragments, collect(DISTINCT p.name) AS people

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
        rows = self.run_query(cypher, {"name": name})
        return rows[0] if rows else None


def _build_service() -> Neo4jService:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return Neo4jService(uri=uri, user=user, password=password, database=database)


neo4j_service = _build_service()
