"""Join the MiDRASH page inventory against our scrape state and index to build a master KTIV priority queue."""
import csv, re, json, collections
REPO = '/Users/isaac/Documents/GitHub/genizah_search'
OUT = REPO
HDA = '/Users/isaac/Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah'

def tf(v): return str(v).lower() == 'true'
def norm(s): return re.sub(r'[^a-z0-9]', '', (s or '').lower().split(' ms. ')[-1].split(' ms ')[-1])
def inst_of(shelf):
    s = (shelf or '').lower()
    if 'jewish theological seminary' in s: return 'New_York_JTS'
    if 'national library of russia' in s: return 'St_Petersburg_RNL'
    if 'cambridge university library' in s: return 'Cambridge_CUL'
    if 'british library' in s: return 'London_BL'
    if 'bodleian' in s: return 'Oxford_Bodleian'
    if 'alliance' in s: return 'Paris_AIU'
    if 'manchester' in s: return 'Manchester_Rylands'
    if 'hungarian' in s: return 'Budapest_MTA'
    return ''
RESTRICTED = {'New_York_JTS', 'St_Petersburg_RNL'}

midrash = {r['sys_id']: r for r in csv.DictReader(open('midrash_manuscripts.csv'))}
scraped = set(l.strip() for l in open('scraped_sysids.txt'))
meta = {r['sys_id']: r for r in csv.DictReader(open('ktiv_meta_clean.csv'))}
es_rows = list(csv.DictReader(open('es_v5_docs.csv')))
sys_of_doc = json.load(open('es_sys_of_doc.json'))
es_by_shelf = collections.defaultdict(list)
for r in es_rows: es_by_shelf[norm(r['shelf_mark'])].append(r)

# sys_id -> best ES doc (direct KTIV url first, then shelfmark join through our metadata records)
es_of_sys = {}
for r in es_rows:
    if r['_id'] in sys_of_doc: es_of_sys.setdefault(sys_of_doc[r['_id']], r)
for sid, m in meta.items():
    if sid not in es_of_sys:
        cands = es_by_shelf.get(norm(m['shelf_mark']))
        if cands: es_of_sys[sid] = max(cands, key=lambda d: (tf(d['has_images']), int(d['description_chars'] or 0)))

# prior queue + progress, keyed by canonical_id and by sys_id where the queue carried one
prior = {r['canonical_id']: r for r in csv.DictReader(open(f'{REPO}/ktiv_scrape_priority_queue.csv'))}
prior_by_sys = {}
for r in prior.values():
    m = re.search(r'(\d{15,})', r['ktiv_manifest_url'] or '')
    if m: prior_by_sys[m.group(1)] = r
progress = {}
for r in csv.DictReader(l for l in open(f'{HDA}/ktiv/ktiv_queue_progress.csv') if not l.startswith('#')):
    progress[r['canonical_id']] = r['status']
missing = {}
for r in csv.DictReader(open(f'{HDA}/merged/ktiv_images_missing.csv')):
    missing[r['sys_num']] = r

universe = set(midrash) | scraped | set(meta) | set(missing)
out = []
for sid in universe:
    md = midrash.get(sid); es = es_of_sys.get(sid); mt = meta.get(sid); ms = missing.get(sid)
    shelf = (mt or {}).get('shelf_mark') or (es or {}).get('shelf_mark') or (ms or {}).get('shelfmark_display') or ''
    cid = (es or {}).get('canonical_id') or (ms or {}).get('canonical_id') or ''
    inst = inst_of(shelf) or (es or {}).get('institution') or ''
    pq = prior.get(cid) or prior_by_sys.get(sid) or {}
    n_pages = int(md['n_pages']) if md else 0; n_chars = int(md['n_chars']) if md else 0
    has_zip = sid in scraped
    in_es = es is not None
    es_img = tf(es['has_images']) if es else False
    es_ktiv = tf(es['has_ktiv_images']) if es else False
    bib = tf(es['has_bibliography']) if es else False
    tr = tf(es['has_transcriptions']) if es else False
    desc = int(es['description_chars'] or 0) if es else 0
    rich = bib or tr or desc >= 200
    if has_zip:
        tier, why = 0, 'image zip already on disk' + ('' if es_ktiv else ' (not yet ingested into index)' if in_es else ' (no index record)')
    elif inst in RESTRICTED:
        tier, why = 5, f'known-restricted institution ({inst}); expect metadata only'
    elif in_es and not es_img and rich:
        tier, why = 1, 'searchable record with rich metadata but no image'
    elif in_es and not es_img:
        tier, why = 2, 'searchable record without image (thin metadata)'
    elif mt or ms:
        tier, why = 2, 'KTIV metadata scraped, no image, no index match'
    elif in_es and es_img and not es_ktiv:
        tier, why = 3, 'indexed with non-KTIV image only; KTIV copy would be a permitted image'
    elif md and n_chars >= 800 and n_pages >= 1:
        tier, why = 3, 'in MiDRASH inventory only; substantial text (>=800 chars)'
    elif md and n_chars >= 150:
        tier, why = 4, 'in MiDRASH inventory only; little text'
    elif md:
        tier, why = 6, 'in MiDRASH inventory only; near-empty read (likely blank/stamp/target sheet)'
    else:
        tier, why = 4, 'known from our records only; not in MiDRASH inventory'
    out.append(dict(tier=tier, tier_reason=why, sys_id=sid, shelf_mark=shelf, institution=inst, canonical_id=cid,
        in_index=in_es, index_has_images=es_img, index_has_ktiv_images=es_ktiv, has_bibliography=bib,
        has_transcriptions=tr, description_chars=desc, midrash_n_ie=(md or {}).get('n_ie', 0),
        midrash_n_pages=n_pages, midrash_n_chars=n_chars, in_midrash=md is not None, have_zip=has_zip,
        prior_queue_tier=pq.get('tier', ''), prior_status=progress.get(cid, ''),
        ktiv_manifest_url=f'https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS{sid}-1/manifest'))

# rank: tier, then scholarship signals, then text volume
out.sort(key=lambda r: (r['tier'], -int(r['has_bibliography']), -int(r['has_transcriptions']), -r['description_chars'], -r['midrash_n_chars'], -r['midrash_n_pages'], r['sys_id']))
cols = ['rank'] + list(out[0].keys())
with open(f'{OUT}/ktiv_master_priority_queue.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for i, r in enumerate(out, 1): w.writerow({'rank': i, **r})

# summary
tiers = collections.Counter(r['tier'] for r in out)
summ = {
  'universe': len(out), 'midrash_manuscripts': len(midrash), 'midrash_pages': sum(int(r['n_pages']) for r in midrash.values()),
  'zips_on_disk': len(scraped), 'zips_not_in_midrash': len(scraped - set(midrash)),
  'metadata_records': len(meta), 'index_docs': len(es_rows), 'index_docs_linked_to_sys_id': len(es_of_sys),
  'prior_queue_rows': len(prior), 'prior_queue_rows_with_sys_id': len(prior_by_sys),
  'midrash_pages_already_held': sum(int(midrash[s]['n_pages']) for s in scraped if s in midrash),
  'tiers': {t: {'count': tiers[t], 'pages': sum(r['midrash_n_pages'] for r in out if r['tier']==t), 'reason': next(r['tier_reason'] for r in out if r['tier']==t).split(' (')[0]} for t in sorted(tiers)},
  'restricted_by_inst': collections.Counter(r['institution'] for r in out if r['tier']==5).most_common(),
}
json.dump(summ, open(f'{OUT}/ktiv_master_queue_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
