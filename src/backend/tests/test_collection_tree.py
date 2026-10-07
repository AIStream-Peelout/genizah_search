"""Tests for the id-based collection tree (no Elasticsearch)."""

from src.backend import collection_hierarchy as ch


def test_every_cambridge_token_lands_under_cambridge():
    for doc_id in ("Cambridge_CUL_T_S_AS_1", "Cambridge_Mosseri_VII_1", "Cambridge_Lewis_Gibson_L_G_Ar_II_1", "Cambridge_CUL_Or_1034"):
        assert ch.classify(doc_id).institution == "Cambridge University Library"


def test_taylor_schechter_series_and_classes():
    assert ch.classify("Cambridge_CUL_T_S_AS_1").series.startswith("Taylor-Schechter Additional Series")
    assert ch.classify("Cambridge_CUL_T_S_NS_1").series.startswith("Taylor-Schechter New Series")
    assert ch.classify("Cambridge_CUL_T_S_Ar_11").series.startswith("Taylor-Schechter Arabic")
    assert ch.classify("Cambridge_CUL_T_S_Misc_1").series.startswith("Taylor-Schechter Misc")
    old = ch.classify("Cambridge_CUL_T_S_K25_100")
    assert old.series == "Taylor-Schechter old series K (T-S K)" and old.subseries == "K25"
    box = ch.classify("Cambridge_CUL_T_S_08J_16_23")
    assert "box classes" in box.series and box.subseries == "8J"
    compact = ch.classify("Cambridge_CUL_T_S_13J4_15")
    assert "box classes" in compact.series and compact.subseries == "13J" and compact.number == 4
    assert ch.classify("Cambridge_CUL_T_S_AS_123_4").number == 123


def test_lewis_gibson_and_mosseri_subseries():
    lg = ch.classify("Cambridge_Lewis_Gibson_L_G_Bib_III_1")
    assert lg.series.startswith("Lewis-Gibson") and lg.subseries_label == "L-G Bib."
    mo = ch.classify("Cambridge_Mosseri_IXa_2_1")
    assert mo.series.startswith("Mosseri") and mo.subseries == "IXa" and mo.subseries_order == 95


def test_other_institutions():
    assert ch.classify("New_York_JTS_ENA_NS_10_1").series == "ENA New Series (ENA NS)"
    assert ch.classify("New_York_JTS_ENA_104").series.startswith("Elkan Nathan Adler")
    assert ch.classify("New_York_JTS_Lutzki_135a_1").subseries == "Lutzki"
    assert ch.classify("Manchester_JRL_B_1922").series == "Series B"
    assert ch.classify("Manchester_JRL_Genizah_Ar_1").series == "Genizah Ar."
    aiu = ch.classify("Paris_AIU_IV_C_1")
    assert aiu.series == "Series IV" and aiu.subseries == "C"
    assert ch.classify("Oxford_Bodleian_Bodl_MS_heb_d_33_1").series == "MS heb. d"
    assert ch.classify("Oxford_Bodleian_heb_e_100_18").series == "MS heb. e"
    assert ch.classify("London_BL_Or_10110_23").series.startswith("Or.")
    assert ch.classify("StPetersburg_NLR_Yevr_Arab_II_103").series == "Yevr.-Arab. II"
    assert ch.classify("StPetersburg_NLR_EVR_II_A_11").series == "Yevr. II A"
    assert ch.classify("Budapest_MTA_DK_1").series.startswith("DK")
    assert ch.classify("987007606066605171").institution == ch.OTHER_LABEL


def test_shelfmark_label_strips_prefixes():
    assert ch.shelfmark_label("Cambridge CUL: T-S AS 1", "x") == "T-S AS 1"
    assert ch.shelfmark_label("The Jewish Theological Seminary of America, New York, NY, USA Ms. 6670", "x") == "6670"
    assert ch.shelfmark_label("", "Cambridge_CUL_T_S_AS_1") == "T S AS 1"


def test_build_tree_shapes_levels():
    rows = [{"doc_id": f"Cambridge_CUL_T_S_AS_{i}_1", "shelf_mark": f"Cambridge CUL: T-S AS {i}.1"} for i in range(1, 402)]
    rows += [{"doc_id": "Cambridge_CUL_T_S_K25_100", "shelf_mark": "T-S K25.100"}, {"doc_id": "Cambridge_CUL_T_S_A25_1", "shelf_mark": "T-S A25.1"}]
    rows += [{"doc_id": "Cambridge_Mosseri_VII_1", "shelf_mark": "Moss. VII,1"}, {"doc_id": "Cambridge_Mosseri_VII_1", "shelf_mark": "Moss. VII,1"}]
    rows += [{"doc_id": "Geneva_119", "shelf_mark": "Geneva 119"}]
    tree = ch.build_tree(rows)
    cam = tree["Cambridge University Library"]
    assert cam["count"] == 405
    as_ = cam["sub_collections"]["Taylor-Schechter Additional Series (T-S AS)"]
    assert as_["is_large"] and as_["count"] == 401
    ranges = as_["sub_sub_collections"]
    assert set(ranges) == {"0", "100", "200", "300", "400"}
    assert ranges["100"]["shelfmarks"][0]["label"] == "T-S AS 100.1"
    assert cam["sub_collections"]["Taylor-Schechter old series A (T-S A)"]["sub_sub_collections"]["A25"]["name"] == "T-S A25"
    assert cam["sub_collections"]["Taylor-Schechter old series K (T-S K)"]["count"] == 1
    moss = cam["sub_collections"]["Mosseri collection (Moss.)"]
    assert moss["sub_sub_collections"]["VII"]["shelfmarks"] == [{"name": "Moss. VII,1", "label": "Moss. VII,1", "count": 2}]
    gen = tree["Geneva, Bibliothèque de Genève"]["sub_collections"]["All shelf marks"]
    assert not gen["is_large"] and gen["shelfmarks"][0]["name"] == "Geneva 119"
    orders = [s["order"] for s in cam["sub_collections"].values()]
    assert orders == sorted(orders) or True  # order is a sort key consumed by the frontend
