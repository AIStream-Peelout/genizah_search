"""Durable storage for reader ratings of chat answers.

Each answer in the chat UI carries a thumbs-up / thumbs-down control.  A vote
is stored as one Elasticsearch document keyed by the client-generated
``message_id``, so changing a vote (or adding a comment after voting down)
overwrites the earlier document instead of accumulating duplicates.

The index is named by ``CHAT_FEEDBACK_INDEX`` (default
``genizah_chat_feedback_v1``) and is created with an explicit mapping the
first time a vote arrives.  Model names are read server-side from the running
RAG service (or the environment) rather than from the request body, so a
browser cannot mislabel which model produced an answer.  IP addresses are
never stored; the user agent is kept, truncated, only as a coarse hint about
the device an answer was read on.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from elasticsearch import BadRequestError, Elasticsearch
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Index used when ``CHAT_FEEDBACK_INDEX`` is not set.
DEFAULT_FEEDBACK_INDEX = "genizah_chat_feedback_v1"

MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 20000
MAX_COMMENT_CHARS = 1000
MAX_ID_CHARS = 128
MAX_MODEL_CHARS = 200
MAX_USER_AGENT_CHARS = 300

#: Explicit mapping: keyword for the facets the owner will aggregate on,
#: analysed text for the free-form fields, date for the timestamp.  Strict
#: dynamic mapping matches the other side indices in this project and makes a
#: field added on the client fail loudly instead of being silently indexed.
FEEDBACK_INDEX_MAPPING: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "rating": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "comment": {"type": "text"},
            "message_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "synthesis_model": {"type": "keyword"},
            "verification_model": {"type": "keyword"},
            "router_model": {"type": "keyword"},
            "es_index": {"type": "keyword"},
            "trace_id": {"type": "keyword"},
            "created_at": {"type": "date"},
            "user_agent": {"type": "keyword", "ignore_above": 512},
        },
    },
}


class ChatFeedbackRequest(BaseModel):
    """One rating of one assistant answer, as posted by the chat UI.

    ``models`` and ``synthesis_model`` are accepted for forward compatibility
    with clients that echo back what they were told, but they are **not**
    stored: the served model names are read server-side so they cannot be
    spoofed.
    """

    rating: Literal["up", "down"] = Field(
        ...,
        description="Thumbs up or thumbs down on the answer.",
    )
    question: str = Field(
        ...,
        max_length=MAX_QUESTION_CHARS,
        description="The user message the rated answer responded to.",
    )
    answer: str = Field(
        ...,
        description=(
            "The rated answer. Longer submissions are truncated server-side "
            f"to {MAX_ANSWER_CHARS} characters rather than rejected."
        ),
    )
    comment: Optional[str] = Field(
        None,
        max_length=MAX_COMMENT_CHARS,
        description="Optional free-text note, typically why a vote was down.",
    )
    message_id: Optional[str] = Field(
        None,
        max_length=MAX_ID_CHARS,
        description="Client-generated id of the rated answer; used as the ES "
                    "_id so a changed vote overwrites the earlier one.",
    )
    session_id: Optional[str] = Field(
        None,
        max_length=MAX_ID_CHARS,
        description="Random client id grouping the votes of one browser.",
    )
    models: Optional[Dict[str, str]] = Field(
        None,
        description="Client-reported model names. Accepted and ignored.",
    )
    synthesis_model: Optional[str] = Field(
        None,
        max_length=MAX_MODEL_CHARS,
        description="Client-reported synthesis model. Accepted and ignored.",
    )
    trace_id: Optional[str] = Field(
        None,
        max_length=MAX_MODEL_CHARS,
        description="Optional Weave/W&B call id echoed back from the answer.",
    )


class ChatFeedbackService:
    """Write and summarise chat answer ratings in Elasticsearch.

    :param es: Elasticsearch client to write with; the backend passes the
        client the rest of the application already uses.
    :param rag_service: Optional running RAG service whose configured model
        roles are recorded with each vote; the environment is used when it is
        ``None`` or missing an attribute.
    :param index_name: Optional index override, mainly for tests; defaults to
        ``CHAT_FEEDBACK_INDEX`` and then :data:`DEFAULT_FEEDBACK_INDEX`.
    :param serving_index: Name of the search index the answers were drawn
        from, recorded so votes can be read per served index; defaults to
        ``ELASTICSEARCH_INDEX``.
    """

    def __init__(
        self,
        es: Elasticsearch,
        rag_service: Optional[Any] = None,
        index_name: Optional[str] = None,
        serving_index: Optional[str] = None,
    ) -> None:
        self._es = es
        self._rag_service = rag_service
        self.index_name = index_name or os.getenv(
            "CHAT_FEEDBACK_INDEX", DEFAULT_FEEDBACK_INDEX
        )
        self.serving_index = serving_index or os.getenv("ELASTICSEARCH_INDEX") or None
        self._index_checked = False

    def ensure_index(self) -> bool:
        """Create the feedback index with its mapping if it does not exist.

        The existence check runs once per process after it first succeeds, so
        the common path is a single ``index`` call.

        :return: ``True`` when this call created the index, ``False`` when it
            already existed.
        :rtype: bool
        """
        if self._index_checked:
            return False
        if self._es.indices.exists(index=self.index_name):
            self._index_checked = True
            return False
        try:
            self._es.indices.create(index=self.index_name, **FEEDBACK_INDEX_MAPPING)
        except BadRequestError as exc:
            # Another worker created it between the check and the create; any
            # other bad request (a mapping problem) is a real failure.
            if exc.error != "resource_already_exists_exception":
                raise
            self._index_checked = True
            return False
        self._index_checked = True
        logger.info("Created chat feedback index %s", self.index_name)
        return True

    def _model_roles(self) -> Dict[str, Optional[str]]:
        """Read the served model names from the RAG service or environment.

        :return: ``synthesis_model`` / ``verification_model`` / ``router_model``
            as configured on this server, each possibly ``None``.
        :rtype: Dict[str, Optional[str]]
        """
        roles = {
            "synthesis_model": os.getenv("SYNTHESIS_MODEL") or None,
            "verification_model": os.getenv("VERIFICATION_MODEL") or None,
            "router_model": os.getenv("ROUTER_MODEL") or None,
        }
        if self._rag_service is not None:
            for role in roles:
                configured = getattr(self._rag_service, role, None)
                if configured:
                    roles[role] = str(configured)
        return roles

    def build_document(
        self,
        feedback: ChatFeedbackRequest,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the stored document for one rating.

        :param feedback: Validated request body.
        :param user_agent: Inbound ``User-Agent`` header, if any; truncated.
        :return: The document to index. No IP address is included.
        :rtype: Dict[str, Any]
        """
        document: Dict[str, Any] = {
            "rating": feedback.rating,
            "question": feedback.question,
            # Truncated rather than rejected: a reader should never lose a
            # vote because the answer they rated was unusually long.
            "answer": feedback.answer[:MAX_ANSWER_CHARS],
            "comment": feedback.comment,
            "message_id": feedback.message_id,
            "session_id": feedback.session_id,
            "es_index": self.serving_index,
            "trace_id": feedback.trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_agent": user_agent[:MAX_USER_AGENT_CHARS] if user_agent else None,
        }
        document.update(self._model_roles())
        return document

    def record(
        self,
        feedback: ChatFeedbackRequest,
        user_agent: Optional[str] = None,
    ) -> str:
        """Store one rating, overwriting any earlier vote on the same answer.

        :param feedback: Validated request body.
        :param user_agent: Inbound ``User-Agent`` header, if any.
        :return: The Elasticsearch document id the vote was stored under.
        :rtype: str
        """
        self.ensure_index()
        document_id = feedback.message_id or uuid.uuid4().hex
        self._es.index(
            index=self.index_name,
            id=document_id,
            document=self.build_document(feedback, user_agent),
        )
        return document_id

    def summary(self) -> Dict[str, Any]:
        """Count votes overall and per synthesis model.

        :return: ``{"index": ..., "total": ..., "up": ..., "down": ...,
            "by_synthesis_model": [{"synthesis_model": ..., "up": ...,
            "down": ..., "total": ...}, ...]}``. Zeros when the index has not
            been created yet.
        :rtype: Dict[str, Any]
        """
        empty: Dict[str, Any] = {
            "index": self.index_name,
            "total": 0,
            "up": 0,
            "down": 0,
            "by_synthesis_model": [],
        }
        if not self._es.indices.exists(index=self.index_name):
            return empty

        response = self._es.search(
            index=self.index_name,
            size=0,
            track_total_hits=True,
            aggs={
                "by_rating": {"terms": {"field": "rating", "size": 10}},
                "by_model": {
                    "terms": {"field": "synthesis_model", "size": 50},
                    "aggs": {"by_rating": {"terms": {"field": "rating", "size": 10}}},
                },
            },
        )
        overall = _rating_counts(response["aggregations"]["by_rating"]["buckets"])
        by_model = []
        for bucket in response["aggregations"]["by_model"]["buckets"]:
            counts = _rating_counts(bucket["by_rating"]["buckets"])
            by_model.append(
                {
                    "synthesis_model": bucket["key"],
                    "total": bucket["doc_count"],
                    **counts,
                }
            )
        return {
            "index": self.index_name,
            "total": response["hits"]["total"]["value"],
            **overall,
            "by_synthesis_model": by_model,
        }


def _rating_counts(buckets: Any) -> Dict[str, int]:
    """Flatten a ``rating`` terms aggregation into up/down counts.

    :param buckets: Bucket list from an Elasticsearch terms aggregation.
    :return: ``{"up": n, "down": n}`` with missing ratings counted as zero.
    :rtype: Dict[str, int]
    """
    counts = {"up": 0, "down": 0}
    for bucket in buckets:
        if bucket["key"] in counts:
            counts[bucket["key"]] = bucket["doc_count"]
    return counts
