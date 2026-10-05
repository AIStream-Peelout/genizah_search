import re
ROMAN={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10}
WORDS={'NS':'NS','AS':'AS','AR':'AR','MISC':'MISC'}
PREFIX=re.compile(r'^(Cambridge CUL:\s*|Cambridge University Library, Cambridge, England Ms\.\s*|CUL\s+|Cambridge Lewis-Gibson:\s*|Cambridge University Library, Cambridge, England Ms\.\s*|Mosseri, Jacques, Paris, France Ms\.\s*)',re.I)
def series(sm):
    s=PREFIX.sub('',sm.strip())
    if re.match(r'T-S\b',s): return 'TS',s[3:]
    if re.match(r'Or\.?\s*\d',s): return 'OR',re.sub(r'^Or\.?','',s)
    if re.match(r'Add\.?\s*\d',s): return 'ADD',re.sub(r'^Add\.?','',s)
    if re.match(r'Moss(\.|eri)?\b',s,re.I): return 'MOSSERI',re.sub(r'^Moss(\.|eri)?','',s,flags=re.I)
    if re.match(r'L-G\b',s): return 'LG',s[3:]
    if re.match(r'^(VI|VII|VIII|IX|X|I|II|III|IV|V)[a-z]?\s',s) and 'Mosseri' in sm: return 'MOSSERI',s
    return None,None
def cudl_from_shelfmark(sm):
    ser,rest=series(sm)
    if not ser: return None
    if re.search(r'[–\-]\s*\d|\s(and|&)\s|;|,\s*\d+\s*-|\+',rest.replace('T-S','')) and ser!='MOSSERI': return None  # ranges/joins
    raw=re.findall(r'\d+|[A-Za-z]+',rest)
    out=[]
    if ser=='MOSSERI':
        if not raw: return None
        out.append(raw[0].upper()); raw=raw[1:]
        if any(not x.isdigit() and len(x)>1 for x in raw): return None
        out+= [x.zfill(5) if x.isdigit() else x.upper() for x in raw]
        return 'MS-MOSSERI-'+'-'.join(out)
    if ser=='LG':
        cat={'Ar':'ARABIC','Misc':'MISC','Glass':'GLASS','Bib':'BIBLE','Lit':'LITURGY','Talm':'TALMUD'}.get(raw[0]) if raw else None
        if not cat: return None
        for x in raw[1:]:
            out.append(str(ROMAN[x]).zfill(5) if x in ROMAN else (x.zfill(5) if x.isdigit() else None))
        if None in out: return None
        return 'MS-LG-'+cat+''.join('-'+o for o in out)
    for x in raw:
        if x.isdigit(): out.append(x.zfill(5))
        elif x.upper() in WORDS: out.append(WORDS[x.upper()])
        elif x=='Ka': out.append('KA')
        elif len(x)<=2: out+= list(x.upper())
        else: return None
    return 'MS-'+ser+'-'+'-'.join(out)
