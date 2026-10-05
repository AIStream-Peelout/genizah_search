"""The joins block served on the document page must not carry provider URLs or contributor names."""

from src.backend.search_service import ElasticsearchService


RAW_JOINS = {
    "joinedManuscripts": [
        {"shelfmark": "Paris, Mosseri: Moss. V,150 (Alt: P 151)", "index": 0, "source": "listOfJoins"},
        {"index": 1, "source": "listOfJoins"},
    ],
    "mainShelfmark": "Jerusalem, NLI: 577.2/16",
    "source": "Site User - Dr. Example Person",
    "metadata": {
        "pageUrl": "https://fgp.genizah.org/GeneralPages/Join/JoinDetails.aspx?InventoryId=1",
        "extractedAt": "2025-11-04T05:23:59.118Z",
        "extractionMethod": "DOM",
    },
}


def test_only_shelfmarks_survive() -> None:
    """Provider URL, contributor string and per-join ``source`` are dropped; shelfmarks stay."""
    public = ElasticsearchService._public_joins_data(RAW_JOINS)
    assert public == {
        "mainShelfmark": "Jerusalem, NLI: 577.2/16",
        "joinedManuscripts": [{"shelfmark": "Paris, Mosseri: Moss. V,150 (Alt: P 151)", "index": 0}],
    }
    assert "genizah.org" not in repr(public)
    assert "Example Person" not in repr(public)


def test_empty_or_malformed_joins_become_none() -> None:
    """Nothing to show means ``None`` so the page keeps its existing ``joins_data`` guard."""
    assert ElasticsearchService._public_joins_data(None) is None
    assert ElasticsearchService._public_joins_data("not a dict") is None
    assert ElasticsearchService._public_joins_data({"source": "x", "metadata": {"pageUrl": "y"}}) is None
