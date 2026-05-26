import logging
import os
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jService:
    """Manages a Neo4j driver and provides query execution helpers."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Driver | None = None

    def connect(self) -> None:
        """Open the driver connection. Safe to call multiple times."""
        if self._driver is not None:
            return
        try:
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._uri)
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
        with self._driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def get_places(self) -> list[dict[str, Any]]:
        """Return all PLACE nodes with their properties."""
        cypher = "MATCH (p:Place) RETURN p"
        rows = self.run_query(cypher)
        return [row["p"] for row in rows]


def _build_service() -> Neo4jService:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return Neo4jService(uri=uri, user=user, password=password)


neo4j_service = _build_service()
