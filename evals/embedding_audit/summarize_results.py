"""Print the audit's result tables as Markdown (for the report) from ``results/*.json``."""

import json
from pathlib import Path

HERE = Path(__file__).parent / "results"


def load(name: str):
    """Load a results file if present.

    :param name: File name in ``results/``.
    :returns: Parsed JSON or None.
    """
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def t2_table() -> None:
    """T2 corpus retrieval table (all models and tags present)."""
    print("| model / variant | P@10 strict | P@10 lenient | P@50 lenient | framed AP | framed nDCG@10 | framed R@100 | Sukkot gold R@100 | R@1000 | no-mention R@1000 | median rank (no-mention) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    bm = load("t2_corpus_bm25.json")
    if bm:
        m = bm["macro"]
        print(f"| BM25 | {m['p10_strict']:.3f} | {m['p10_lenient']:.3f} | | {m['framed_ap']:.3f} | {m['framed_ndcg10']:.3f} | | | | | |")
    for p in sorted(HERE.glob("t2_corpus_*.json")):
        if "bm25" in p.name:
            continue
        r = json.loads(p.read_text())
        for v, res in r["variants"].items():
            m, g = res["macro"], res["sukkot_gold"]
            print(f"| {r['model']}{(' ' + r['tag']) if r.get('tag') else ''} / {v} | {m['p10_strict']:.3f} | {m['p10_lenient']:.3f} | "
                  f"{m['p50_lenient']:.3f} | {m['framed_ap']:.3f} (rand {m['random_framed_ap']:.3f}) | {m['framed_ndcg10']:.3f} | "
                  f"{m['framed_r100']:.3f} | {g['recall@100']} | {g['recall@1000']} | {g['no_mention_recall@1000']} | {g['median_rank_no_mention']} |")
        nb = r.get("neighbourhood_k10")
        if nb:
            print(f"|   ↳ 10-NN of a topic-labelled record share: topic {nb['topic']:.2f} · domain {nb['domain']:.2f} · language {nb['lang']:.2f} · framed {nb['framed']:.2f} | | | | | | | | | | |")


def t2_per_topic(name: str, variant: str = "default_instruction") -> None:
    """Per-topic framed-pool AP for one model.

    :param name: results file.
    :param variant: variant key.
    """
    r = load(name)
    if not r:
        return
    print(f"\nPer-topic ({r['model']} {variant}): topic n_framed_pos AP nDCG@10 P@10strict")
    for t, v in r["variants"][variant]["per_topic"].items():
        print(f"  {t:17s} {v['n_framed_pos']:4d} {v['framed_ap']:.3f} {v['framed_ndcg10']:.3f} {v['p10_strict']:.3f}")


def i1_table() -> None:
    """Image feature table."""
    rows = load("i1_images.json") or []
    print("| features | R@1 | R@10 | R@100 | MRR | cross-coll R@10 | FJP R@10 | PGP R@10 | AUC | scribe P@1 | scribe mAP | scribe AUC | style | region | collection |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        j, s, sc = r["joins"], r.get("scribes", {}), r["script"]
        print(f"| {r['features']} | {j['all']['R@1']:.3f} | {j['all']['R@10']:.3f} | {j['all']['R@100']:.3f} | {j['all']['MRR']:.3f} | "
              f"{j['cross_collection'].get('R@10', 0):.3f} | {j.get('fjp_literary', {}).get('R@10', 0):.3f} | "
              f"{j.get('pgp_documentary', {}).get('R@10', 0):.3f} | {j['auc_vs_same_collection_nonjoin']:.3f} | "
              f"{s.get('P@1', 0):.3f} | {s.get('mAP', 0):.3f} | {s.get('auc_same_vs_diff_scribe', 0):.3f} | "
              f"{sc.get('script_style', {}).get('knn10_balanced_acc', 0):.3f} | {sc.get('script_region', {}).get('knn10_balanced_acc', 0):.3f} | "
              f"{r['confound_collection']['knn10_balanced_acc']:.3f} |")


def main() -> None:
    """Print everything available."""
    print("## T1 concept probe")
    for p in sorted(HERE.glob("t1_concepts_*.json")):
        r = json.loads(p.read_text())
        for v, res in r["variants"].items():
            q = res["queries"]
            print(f"  {r['model']:20s} {v:20s} top1={q['top1']:.3f} top3={q['top3']:.3f} mrr={q['mrr']:.3f} "
                  f"he={res['queries_hebrew']['top1']:.3f} lat={res['queries_latin']['top1']:.3f} names={res['variants']['top1']:.3f}")
    print("\n## T2 corpus")
    t2_table()
    t2_per_topic("t2_corpus_qwen3-0.6b.json")
    print("\n## I1 images")
    i1_table()
    for name in ("i2_lines.json", "i3_line_head_pilot.json"):
        r = load(name)
        if r:
            print(f"\n## {name}\n{json.dumps(r, indent=1)}")


if __name__ == "__main__":
    main()
