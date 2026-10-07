import sys,json,re,csv,collections as C
from urllib.parse import quote_plus
sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/isaac1/Documents/historical-document-analysis')
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.merging.institution_tokens import combine, institution_token
from cudl2 import cudl_from_shelfmark
BLOCK=('genizah.org','jewishmanuscripts.org')
rows=list(csv.DictReader(open('/Users/isaac1/Documents/pgp-metadata/data/fragments.csv',encoding='utf-8-sig')))
frag={}; alias={}
for r in rows:
    tok=institution_token(f"{r.get('library_abbrev') or ''} {r.get('library') or ''}")
    core=ShelfmarkNormalizer.to_canonical_id(r['shelfmark'])
    if not core: continue
    p=combine(tok,core); alias[p]=p
    for v in [x.strip() for x in (r.get('shelfmarks_historic') or '').split(';') if x.strip()]:
        vc=ShelfmarkNormalizer.to_canonical_id(v)
        if vc: alias.setdefault(combine(tok,vc),p)
    frag.setdefault(p,r)
INST={'Cambridge':'Cambridge University Library','Manchester':'Manchester','New':'JTS','Oxford':'Bodleian','Paris':'Paris (AIU/BnF)','StPetersburg':'St Petersburg NLR/IOM','London':'British Library','Budapest':'Budapest MTA','Philadelphia':'Penn (CAJS)','Jerusalem':'NLI','Cairo':'Cairo ENL/JCC','Vienna':'Vienna ONB','Strasbourg':'Strasbourg BNU','Washington':'Freer','Cincinnati':'HUC'}
def row(cid): return INST.get(cid.split('_')[0],'Other')
LUNA=re.compile(r'^(?:Manchester:\s*|JRL\s+(?:Genizah\s+)?|The University of Manchester Library, Manchester, England Ms\.\s*)(A|B|C|L|P|G|AF|Ar\.)\s*(\d+[a-z]?)$')
HALPER=re.compile(r'(?:^|CAJS:\s*|Ms\.\s*)Halper\s+(\d+)$')
T=C.defaultdict(C.Counter); srcs=C.Counter()
out=open('final_preview.jsonl','w')
for l in open('all.jsonl'):
    s=json.loads(l); cid=s['canonical_id']; tm=s['tei_metadata']; R=row(cid); c=T[R]; c['n']+=1
    L=[]
    f=frag.get(alias.get(cid,cid)) or {}
    pu=(f.get('url') or '').strip(); pi=(f.get('iiif_url') or '').strip()
    m=re.search(r'cudl\.lib\.cam\.ac\.uk/+(?:view|iiif)/(MS-[A-Za-z0-9-]+)',pu+' '+pi)
    cud=m.group(1) if m else (cudl_from_shelfmark(s['shelf_mark']) if R=='Cambridge University Library' else None)
    if cud: L.append(('cudl','https://cudl.lib.cam.ac.uk/view/'+cud))
    bod=s.get('bodleian_catalogue_url') or (pu.replace('genizah.bodleian.ox.ac.uk','hebrew.bodleian.ox.ac.uk') if 'bodleian.ox.ac.uk/catalog' in pu else None)
    if bod: L.append(('bodleian',bod))
    m=re.search(r'iiif\.bodleian\.ox\.ac\.uk/iiif/manifest/([0-9a-f-]{36})',pi)
    if m: L.append(('bodleian_digital','https://digital.bodleian.ox.ac.uk/objects/%s/'%m.group(1)))
    if 'luna.manchester' in pu or 'digitalcollections.manchester' in pu: L.append(('manchester',pu))
    else:
        m=LUNA.match(s['shelf_mark'].strip())
        if m: L.append(('manchester','https://luna.manchester.ac.uk/luna/servlet/view/search?q=%s+LIMIT%%3AManchesterDev~95~&sort=reference_number%%2Cdate_created'%quote_plus('"%s %s"'%(m.group(1),m.group(2))).replace('%2B','+')))
    if any(h in pu for h in ('colenda.library.upenn','openn.library.upenn')): L.append(('penn',pu))
    else:
        m=HALPER.search(s['shelf_mark'].strip())
        if m: L.append(('penn','https://openn.library.upenn.edu/Data/0002/html/h%s.html'%m.group(1)))
    if pu and not any(x in pu for x in ('cudl','bodleian','manchester','upenn','drive.google','kestenbaum')): L.append(('other_inst',pu))
    sysn=None
    m=re.search(r'DOCID/(?:PNX_MANUSCRIPTS)?(\d{15,})',s.get('ktiv_iiif_manifest_url') or '')
    if m: sysn=m.group(1)
    else:
        for u in s.get('image_urls') or []:
            m=re.search(r'/KTIV/(\d{15,})/',u)
            if m: sysn=m.group(1); break
    if sysn: L.append(('ktiv','https://www.nli.org.il/en/discover/manuscripts/hebrew-manuscripts/itempage?vid=MANUSCRIPTS&docId=PNX_MANUSCRIPTS%s'%sysn))
    for p in tm.get('pgpids') or []: L.append(('pgp','https://geniza.princeton.edu/en/documents/%s/'%p))
    assert not any(b in u for _,u in L for b in BLOCK)
    kinds={k for k,_ in L}
    for k in kinds: c[k]+=1; srcs[k]+=1
    inst_kinds=kinds-{'pgp','ktiv'}
    if inst_kinds: c['holding_inst_link']+=1
    if L: c['any']+=1
    else: c['none']+=1
    sp=s['sources_present']
    if sp==['fjp']: c['fjp_only_src']+=1
    if 'fjp' in sp and not L: c['fjp_and_no_link']+=1
    out.write(json.dumps({'cid':cid,'sm':s['shelf_mark'],'links':L},ensure_ascii=False)+'\n')
tot=C.Counter()
for v in T.values(): tot.update(v)
T['TOTAL']=tot
for k,v in sorted(T.items(),key=lambda x:-x[1]['n']): print(k,dict(v))
