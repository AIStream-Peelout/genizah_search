"""Stream the MiDRASH V0.8 transcription zip and emit per-page and per-manuscript inventories."""
import zipfile, re, csv, collections, io
HDR = re.compile(r'^==> (\d+)_(IE\d+)_P(\d+)_(FL\d+) <==\s*$')
pages = csv.writer(open('midrash_pages.csv', 'w', newline=''))
pages.writerow(['sys_id', 'ie', 'p_num', 'fl', 'n_lines', 'n_chars'])
per_ms = collections.defaultdict(lambda: {'ies': set(), 'pages': 0, 'lines': 0, 'chars': 0})
cur = None; n_lines = n_chars = 0; n_pages = 0
def flush():
    global n_pages
    if cur:
        pages.writerow([*cur, n_lines, n_chars]); n_pages += 1
        m = per_ms[cur[0]]; m['ies'].add(cur[1]); m['pages'] += 1; m['lines'] += n_lines; m['chars'] += n_chars
with zipfile.ZipFile('midrash_v08.zip') as z:
    with z.open('Transcriptions.txt') as raw:
        for line in io.TextIOWrapper(raw, encoding='utf-8', errors='replace'):
            m = HDR.match(line)
            if m:
                flush(); cur = (m.group(1), m.group(2), int(m.group(3)), m.group(4)); n_lines = n_chars = 0
            elif cur is not None:
                s = line.strip()
                if s: n_lines += 1; n_chars += len(s)
        flush()
w = csv.writer(open('midrash_manuscripts.csv', 'w', newline=''))
w.writerow(['sys_id', 'n_ie', 'n_pages', 'n_lines', 'n_chars'])
for sid, m in per_ms.items():
    w.writerow([sid, len(m['ies']), m['pages'], m['lines'], m['chars']])
print('pages', n_pages, 'manuscripts', len(per_ms))
