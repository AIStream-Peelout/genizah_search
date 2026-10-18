"""An in-memory property graph that runs exactly the Cypher ``scripts/kg/consolidate_scholars.py`` sends.

Test support for ``test_kg_consolidate_scholars.py`` (not a test module). Write transactions are
atomic (an undo journal rolls a failed one back), ``Scholar.name`` is unique, a node that still has
relationships cannot be deleted, and freed ids can be handed to later CREATEs the way Neo4j reuses
them (``reuse="lifo"`` or ``"fifo"``), which is what makes change-log id bookkeeping fragile.
"""

import copy
import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from scripts.kg import consolidate_scholars as cs

#: Label -> the property that must be unique among nodes with that label.
UNIQUE = {"Scholar": "name"}

CREATE_EDGE = re.compile(r"MATCH \(a\) WHERE elementId\(a\) = \$start MATCH \(b\) WHERE elementId\(b\) = \$end "
                         r"CREATE \(a\)-\[r:`(\w+)`\]->\(b\) SET r = \$props RETURN elementId\(r\) AS eid")
CREATE_NODE = re.compile(r"CREATE \(n:((?:`\w+`:?)+)\) SET n = \$props RETURN elementId\(n\) AS eid")
DELETE_EDGES = "MATCH ()-[r]->() WHERE elementId(r) IN $eids DELETE r RETURN count(*) AS n"
SET_EDGE = "MATCH ()-[r]->() WHERE elementId(r) = $eid SET r = $props RETURN count(*) AS n"
DELETE_NODES = "MATCH (n) WHERE elementId(n) IN $eids DELETE n RETURN count(*) AS n"
UPDATE_SURVIVOR = "MATCH (s) WHERE elementId(s) = $eid AND s.name = $old SET s += $props RETURN count(*) AS n"
RESTORE_SURVIVOR = "MATCH (s) WHERE elementId(s) = $eid SET s = $props RETURN count(*) AS n"
WORKS = "MATCH (s) WHERE elementId(s) IN $eids OPTIONAL MATCH (s)-[:WROTE]->(b:BookArticle)"


class TxError(RuntimeError):
    """A write the database itself refuses (uniqueness, a node with relationships)."""


class Result:
    """Query rows, read with ``single()`` or by iteration."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        """Hold the rows.

        :param rows: The rows.
        :returns: None
        :rtype: None
        """
        self.rows = rows

    def single(self) -> Optional[Dict[str, Any]]:
        """Return the first row.

        :returns: The first row, or None.
        :rtype: Optional[Dict[str, Any]]
        """
        return self.rows[0] if self.rows else None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the rows.

        :returns: An iterator.
        :rtype: Iterator[Dict[str, Any]]
        """
        return iter(self.rows)


def stored(props: Dict[str, Any]) -> Dict[str, Any]:
    """What Neo4j stores for ``SET x = $props``: null values dropped, tuples as lists.

    :param props: Property map.
    :returns: The stored map.
    :rtype: Dict[str, Any]
    """
    return {k: list(v) if isinstance(v, tuple) else copy.deepcopy(v) for k, v in props.items() if v is not None}


def one_line(query: str) -> str:
    """Collapse a query's whitespace.

    :param query: Cypher.
    :returns: The query on one line.
    :rtype: str
    """
    return " ".join(query.split())


class FakeGraph:
    """The database: nodes and relationships keyed by elementId (``4:<store>:<n>`` / ``5:<store>:<n>``)."""

    def __init__(self, nodes: Dict[str, Dict[str, Any]], rels: Dict[str, Dict[str, Any]],
                 reuse: Optional[str] = "lifo", store: str = "db") -> None:
        """Load the graph.

        :param nodes: elementId -> ``{"labels", "props"}``.
        :param rels: elementId -> ``{"type", "start", "end", "props"}``.
        :param reuse: Hand freed ids to later CREATEs: ``"lifo"``, ``"fifo"`` or None (never).
        :param store: The store part of new elementIds.
        :returns: None
        :rtype: None
        """
        self.nodes = {e: {"labels": list(n["labels"]), "props": dict(n["props"])} for e, n in nodes.items()}
        self.rels = {e: {"type": r["type"], "start": r["start"], "end": r["end"], "props": dict(r["props"])}
                     for e, r in rels.items()}
        self.adj: Dict[str, Set[str]] = defaultdict(set)
        for eid, rel in self.rels.items():
            self.adj[rel["start"]].add(eid)
            self.adj[rel["end"]].add(eid)
        self.store = store
        self.reuse = reuse
        self.counters = {"4": 1 + max((int(e.rsplit(":", 1)[1]) for e in self.nodes), default=0),
                         "5": 1 + max((int(e.rsplit(":", 1)[1]) for e in self.rels), default=0)}
        self.free: Dict[str, List[str]] = {"4": [], "5": []}
        self.journal: Optional[List[Callable[[], None]]] = None
        self.writes = 0

    # ------------------------------------------------------------ mutations (each journals its undo)

    def _undo(self, action: Callable[[], None]) -> None:
        """Journal an undo action inside a write transaction.

        :param action: Restores the state before one mutation.
        :returns: None
        :rtype: None
        """
        if self.journal is not None:
            self.journal.append(action)

    def _new_id(self, kind: str) -> str:
        """Hand out an elementId, reusing a freed one when ``reuse`` says so.

        :param kind: ``"4"`` for a node, ``"5"`` for a relationship.
        :returns: The id.
        :rtype: str
        """
        free, counters = self.free[kind], self.counters
        before_free, before_counter = list(free), counters[kind]
        if self.reuse and free:
            eid = free.pop() if self.reuse == "lifo" else free.pop(0)
        else:
            eid = f"{kind}:{self.store}:{counters[kind]}"
            counters[kind] += 1

        def undo() -> None:
            """Give the id back.

            :returns: None
            :rtype: None
            """
            free[:] = before_free
            counters[kind] = before_counter

        self._undo(undo)
        return eid

    def _check_unique(self, eid: str) -> None:
        """Refuse a node whose unique property another node of its label already holds.

        :param eid: The node.
        :returns: None
        :rtype: None
        :raises TxError: On a duplicate.
        """
        node = self.nodes[eid]
        for label in node["labels"]:
            prop = UNIQUE.get(label)
            value = node["props"].get(prop) if prop else None
            if value is not None and any(other != eid and label in n["labels"] and n["props"].get(prop) == value
                                         for other, n in self.nodes.items()):
                raise TxError(f"{label}.{prop} {value!r} already exists")

    def _set_node(self, eid: str, props: Dict[str, Any]) -> None:
        """Replace a node's properties.

        :param eid: The node.
        :param props: The new full property map.
        :returns: None
        :rtype: None
        """
        before = self.nodes[eid]["props"]
        self.nodes[eid]["props"] = props
        self._undo(lambda: self.nodes[eid].__setitem__("props", before))
        self._check_unique(eid)

    def create_node(self, labels: Sequence[str], props: Dict[str, Any]) -> str:
        """Create a node.

        :param labels: Its labels.
        :param props: Its properties.
        :returns: The new elementId.
        :rtype: str
        """
        eid = self._new_id("4")
        self.nodes[eid] = {"labels": list(labels), "props": stored(props)}
        self._undo(lambda: self.nodes.pop(eid))
        self._check_unique(eid)
        return eid

    def delete_node(self, eid: str) -> None:
        """Delete a node without relationships.

        :param eid: The node.
        :returns: None
        :rtype: None
        :raises TxError: When it still has relationships.
        """
        if self.adj.get(eid):
            raise TxError(f"cannot delete node {eid}: it still has relationships")
        node = self.nodes.pop(eid)
        self.free["4"].append(eid)

        def undo() -> None:
            """Put the node back.

            :returns: None
            :rtype: None
            """
            self.nodes[eid] = node
            self.free["4"].remove(eid)

        self._undo(undo)

    def create_rel(self, rel_type: str, start: str, end: str, props: Dict[str, Any]) -> str:
        """Create a relationship.

        :param rel_type: Its type.
        :param start: Start node.
        :param end: End node.
        :param props: Its properties.
        :returns: The new elementId.
        :rtype: str
        """
        eid = self._new_id("5")
        self.rels[eid] = {"type": rel_type, "start": start, "end": end, "props": stored(props)}
        self.adj[start].add(eid)
        self.adj[end].add(eid)

        def undo() -> None:
            """Remove the relationship again.

            :returns: None
            :rtype: None
            """
            rel = self.rels.pop(eid)
            self.adj[rel["start"]].discard(eid)
            self.adj[rel["end"]].discard(eid)

        self._undo(undo)
        return eid

    def delete_rel(self, eid: str) -> None:
        """Delete a relationship.

        :param eid: The relationship.
        :returns: None
        :rtype: None
        """
        rel = self.rels.pop(eid)
        self.adj[rel["start"]].discard(eid)
        self.adj[rel["end"]].discard(eid)
        self.free["5"].append(eid)

        def undo() -> None:
            """Put the relationship back.

            :returns: None
            :rtype: None
            """
            self.rels[eid] = rel
            self.adj[rel["start"]].add(eid)
            self.adj[rel["end"]].add(eid)
            self.free["5"].remove(eid)

        self._undo(undo)

    def set_rel(self, eid: str, props: Dict[str, Any]) -> None:
        """Replace a relationship's properties.

        :param eid: The relationship.
        :param props: The new full property map.
        :returns: None
        :rtype: None
        """
        before = self.rels[eid]["props"]
        self.rels[eid]["props"] = stored(props)
        self._undo(lambda: self.rels[eid].__setitem__("props", before))

    # ------------------------------------------------------------ reads

    def business_key(self, eid: str) -> Any:
        """A node's business key (``cs.BUSINESS_KEYS``), else its elementId.

        :param eid: The node.
        :returns: The key.
        :rtype: Any
        """
        node = self.nodes[eid]
        label = next((label for label in cs.BUSINESS_KEYS if label in node["labels"]), None)
        return node["props"].get(cs.BUSINESS_KEYS[label]) if label else eid

    def edge_rows(self, node_eids: Sequence[str]) -> List[Dict[str, Any]]:
        """Rows of ``cs.EDGE_QUERY``: every relationship touching the nodes, once.

        :param node_eids: Node elementIds.
        :returns: The rows.
        :rtype: List[Dict[str, Any]]
        """
        rows, seen = [], set()
        for node in node_eids:
            for eid in sorted(self.adj.get(node, ())):
                if eid in seen:
                    continue
                seen.add(eid)
                rel = self.rels[eid]
                rows.append({"eid": eid, "type": rel["type"], "props": dict(rel["props"]),
                             "start": rel["start"], "start_labels": list(self.nodes[rel["start"]]["labels"]),
                             "start_key": self.business_key(rel["start"]),
                             "end": rel["end"], "end_labels": list(self.nodes[rel["end"]]["labels"]),
                             "end_key": self.business_key(rel["end"])})
        return rows

    def label_names(self, label: str) -> Set[Any]:
        """The ``name`` values of the nodes with a label.

        :param label: The label.
        :returns: The names (None left out).
        :rtype: Set[Any]
        """
        return {n["props"].get("name") for n in self.nodes.values() if label in n["labels"]} - {None}

    def works(self, node: str) -> Set[str]:
        """The BookArticles a node WROTE.

        :param node: The node.
        :returns: Their elementIds.
        :rtype: Set[str]
        """
        rels = [self.rels[eid] for eid in self.adj.get(node, ())]
        return {rel["end"] for rel in rels if rel["type"] == "WROTE" and rel["start"] == node
                and "BookArticle" in self.nodes[rel["end"]]["labels"]}

    def counted(self, count: int) -> Result:
        """A write query's ``n`` row (and one write counted).

        :param count: Rows touched.
        :returns: The result.
        :rtype: Result
        """
        self.writes += 1
        return Result([{"n": count}])

    # ------------------------------------------------------------ the query interpreter

    def run(self, query: str, **params: Any) -> Result:
        """Run one of the queries the tool sends.

        :param query: Cypher.
        :param params: Parameters.
        :returns: The rows.
        :rtype: Result
        :raises NotImplementedError: For any other query.
        """
        q = one_line(query)
        if q == one_line(cs.SCHOLAR_QUERY):
            return Result([{"eid": e, "labels": list(n["labels"]), "props": dict(n["props"]),
                            "degree": sum(2 if self.rels[r]["start"] == self.rels[r]["end"] else 1
                                          for r in self.adj.get(e, ()))}
                           for e, n in sorted(self.nodes.items()) if "Scholar" in n["labels"]])
        if q == one_line(cs.CLASH_QUERY):
            return Result([{"name": name} for name in sorted(self.label_names("Scholar") & self.label_names("Person"))])
        if q == one_line(cs.PLACE_QUERY):
            return Result([{"name": name} for name in sorted(self.label_names("Place"))])
        if q == one_line(cs.EDGE_QUERY):
            return Result(self.edge_rows(params["eids"]))
        if q == cs.STORE_QUERY:
            return Result([{"eid": eid} for eid in sorted(self.nodes)[:1]])
        if q == "MATCH (n) RETURN count(n) AS c":
            return Result([{"c": len(self.nodes)}])
        if q == "MATCH ()-[r]->() RETURN count(r) AS c":
            return Result([{"c": len(self.rels)}])
        if q.startswith(WORKS):
            return Result([{"eid": node, "works": len(self.works(node))} for node in params["eids"]
                           if node in self.nodes])
        match = CREATE_EDGE.fullmatch(q)
        if match:
            self.writes += 1
            if params["start"] not in self.nodes or params["end"] not in self.nodes:
                return Result([])
            return Result([{"eid": self.create_rel(match[1], params["start"], params["end"], params["props"])}])
        match = CREATE_NODE.fullmatch(q)
        if match:
            self.writes += 1
            return Result([{"eid": self.create_node(re.findall(r"`(\w+)`", match[1]), params["props"])}])
        if q == DELETE_EDGES:
            hit = [eid for eid in dict.fromkeys(params["eids"]) if eid in self.rels]
            for eid in hit:
                self.delete_rel(eid)
            return self.counted(len(hit))
        if q == SET_EDGE:
            if params["eid"] not in self.rels:
                return self.counted(0)
            self.set_rel(params["eid"], params["props"])
            return self.counted(1)
        if q == DELETE_NODES:
            hit = [eid for eid in dict.fromkeys(params["eids"]) if eid in self.nodes]
            for eid in hit:
                self.delete_node(eid)
            return self.counted(len(hit))
        if q == UPDATE_SURVIVOR:
            node = self.nodes.get(params["eid"])
            if node is None or node["props"].get("name") != params["old"]:
                return self.counted(0)
            merged = {key: value for key, value in node["props"].items() if key not in params["props"]}
            self._set_node(params["eid"], {**merged, **stored(params["props"])})
            return self.counted(1)
        if q == RESTORE_SURVIVOR:
            if params["eid"] not in self.nodes:
                return self.counted(0)
            self._set_node(params["eid"], stored(params["props"]))
            return self.counted(1)
        raise NotImplementedError(q[:200])


class FakeSession:
    """A neo4j-like session over a FakeGraph."""

    def __init__(self, graph: FakeGraph, read_only: bool) -> None:
        """Open the session.

        :param graph: The database.
        :param read_only: True for a READ session (a write transaction then fails).
        :returns: None
        :rtype: None
        """
        self.graph = graph
        self.read_only = read_only

    def __enter__(self) -> "FakeSession":
        """Enter the context.

        :returns: The session.
        :rtype: FakeSession
        """
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Leave the context without swallowing errors.

        :param exc: Exception info.
        :returns: False
        :rtype: bool
        """
        return False

    def execute_read(self, work: Callable[..., Any], *args: Any) -> Any:
        """Run a read transaction function, failing if it writes.

        :param work: Transaction function.
        :param args: Its extra arguments.
        :returns: Its result.
        :rtype: Any
        """
        writes = self.graph.writes
        result = work(self.graph, *args)
        assert self.graph.writes == writes, "a read transaction wrote"
        return result

    def execute_write(self, work: Callable[..., Any], *args: Any) -> Any:
        """Run a write transaction function atomically (rolled back when it raises).

        :param work: Transaction function.
        :param args: Its extra arguments.
        :returns: Its result.
        :rtype: Any
        """
        assert not self.read_only, "a write in a READ session"
        self.graph.journal = []
        try:
            return work(self.graph, *args)
        except BaseException:
            for undo in reversed(self.graph.journal):
                undo()
            raise
        finally:
            self.graph.journal = None


class FakeDriver:
    """A neo4j-like driver over a FakeGraph."""

    def __init__(self, graph: FakeGraph) -> None:
        """Hold the graph.

        :param graph: The database.
        :returns: None
        :rtype: None
        """
        self.graph = graph

    def session(self, database: str = "neo4j", default_access_mode: Optional[str] = None) -> FakeSession:
        """Open a session.

        :param database: Ignored.
        :param default_access_mode: ``"READ"`` for a read session.
        :returns: The session.
        :rtype: FakeSession
        """
        return FakeSession(self.graph, default_access_mode == cs.READ_ACCESS)


def build_graph(scholars: Sequence[Tuple[int, str]], rels: Sequence[Tuple[str, str, str, Dict[str, Any]]],
                books: Sequence[str] = ("B1", "B2", "B3"), reuse: Optional[str] = "lifo",
                store: str = "db") -> FakeGraph:
    """A graph of Scholars ``4:<store>:<n>`` and BookArticles ``4:<store>:<100+i>``.

    :param scholars: ``(n, name)`` pairs.
    :param rels: ``(type, start, end, props)``; endpoints are ``"S<n>"`` or a BookArticle article_id.
        Relationship i gets elementId ``5:<store>:<i+1>``.
    :param books: BookArticle article_ids.
    :param reuse: Id reuse mode (``FakeGraph``).
    :param store: The store part of the elementIds.
    :returns: The graph.
    :rtype: FakeGraph
    """
    nodes = {f"4:{store}:{n}": {"labels": ["Scholar"], "props": {"name": name}} for n, name in scholars}
    ref = {f"S{n}": f"4:{store}:{n}" for n, _ in scholars}
    for index, article in enumerate(books, 100):
        nodes[f"4:{store}:{index}"] = {"labels": ["BookArticle"], "props": {"article_id": article}}
        ref[article] = f"4:{store}:{index}"
    edges = {f"5:{store}:{i}": {"type": t, "start": ref[a], "end": ref[b], "props": props}
             for i, (t, a, b, props) in enumerate(rels, 1)}
    return FakeGraph(nodes, edges, reuse=reuse, store=store)


def image(graph: FakeGraph) -> Dict[str, Counter]:
    """An id-free image of the graph: nodes by labels and properties, relationships by endpoint images.

    :param graph: The graph.
    :returns: ``{"nodes", "rels"}`` multisets.
    :rtype: Dict[str, Counter]
    """
    key = {eid: json.dumps([sorted(n["labels"]), n["props"]], sort_keys=True, ensure_ascii=False)
           for eid, n in graph.nodes.items()}
    return {"nodes": Counter(key.values()),
            "rels": Counter(json.dumps([r["type"], key[r["start"]], key[r["end"]], r["props"]], sort_keys=True,
                                       ensure_ascii=False) for r in graph.rels.values())}


def edges_by_name(graph: FakeGraph) -> List[Tuple[str, Any, Any, Any]]:
    """The relationships as ``(type, start key, end key, data_sources)``, sorted.

    :param graph: The graph.
    :returns: The list.
    :rtype: List[Tuple[str, Any, Any, Any]]
    """
    return sorted((r["type"], graph.business_key(r["start"]), graph.business_key(r["end"]),
                   r["props"].get("data_sources")) for r in graph.rels.values())


def scholar_names(graph: FakeGraph) -> List[str]:
    """The Scholar names, sorted.

    :param graph: The graph.
    :returns: The names.
    :rtype: List[str]
    """
    return sorted(n["props"]["name"] for n in graph.nodes.values() if "Scholar" in n["labels"])
