# Site metadata: link previews, favicon, manifest

Everything lives in `src/frontend/public/` and is copied verbatim into the build.

## What crawlers read

- **WhatsApp, Facebook, iMessage, Slack, LinkedIn**: Open Graph tags in `index.html`
  (`og:title`, `og:description`, `og:image` as an **absolute** URL, `og:url`).
  WhatsApp wants the image under 300 KB and the tags near the top of the page.
- **X / Twitter**: `twitter:card`, `twitter:title`, `twitter:description`, `twitter:image`.
- **Google**: `<title>`, `meta description`, the JSON-LD `WebSite` block, `link rel=canonical`.
- **Browsers / home screen**: `favicon.ico` + PNG icons, `apple-touch-icon.png` (opaque, 180 px),
  `manifest.json` with the 192/512 icons.

## Per-page previews

The React app serves one `index.html` for every route, so `src/frontend/nginx.conf`
rewrites the title, description, image and URL for `/yom-kippur` and `/yk` with
`sub_filter`. Add a `location` block the same way for any other page that needs
its own card. The search strings are attribute values only, so CRA's HTML
minifier cannot break them, but they must match `index.html` exactly: change
both together.

## Regenerating the images

Icons are a navy tile with a gold gimel; previews are 1200x630 JPEGs. Fonts are
the macOS system ones (`SFHebrew.ttf`, `Georgia`). PIL has no bidi shaping, so a
Hebrew word is reversed before drawing. Run from the repo root with Pillow:

```python
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
OUT = Path("src/frontend/public")
NAVY=(17,24,39); GOLD=(212,169,74); CREAM=(243,236,220); MUTED=(203,213,225); FAINT=(25,34,54)
heb = lambda s: ImageFont.truetype("/System/Library/Fonts/SFHebrew.ttf", s)
serif = lambda s,b=False: ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if b else "/System/Library/Fonts/Supplemental/Georgia.ttf", s)

def icon(size):
    im = Image.new("RGBA",(size,size),(0,0,0,0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0,0,size-1,size-1], radius=int(size*0.22), fill=NAVY)
    f = heb(int(size*0.68)); bb = d.textbbox((0,0),"ג",font=f)
    d.text(((size-(bb[2]-bb[0]))/2-bb[0], (size-(bb[3]-bb[1]))/2-bb[1]-size*0.04), "ג", font=f, fill=GOLD)
    d.rounded_rectangle([size*0.28,size*0.84,size*0.72,size*0.87], radius=size*0.02, fill=GOLD)
    return im
for s,name in [(512,"icon-512.png"),(192,"icon-192.png"),(32,"favicon-32x32.png"),(16,"favicon-16x16.png")]:
    icon(s).save(OUT/name)
a = Image.new("RGB",(180,180),NAVY); a.paste(icon(180),(0,0),icon(180)); a.save(OUT/"apple-touch-icon.png")
icon(64).save(OUT/"favicon.ico", sizes=[(16,16),(32,32),(48,48),(64,64)])

def og(title, subtitle, out, hebrew):
    W,H=1200,630; im=Image.new("RGB",(W,H),NAVY); d=ImageDraw.Draw(im)
    word = hebrew[::-1]; fb=heb(400); bb=d.textbbox((0,0),word,font=fb)
    d.text((W-(bb[2]-bb[0])+60-bb[0], H-(bb[3]-bb[1])-40-bb[1]), word, font=fb, fill=FAINT)
    d.rectangle([80,150,130,156], fill=GOLD); y=185
    for line in title: d.text((80,y), line, font=serif(74,True), fill=GOLD); y+=88
    y+=10
    for line in subtitle: d.text((82,y), line, font=serif(32), fill=MUTED); y+=46
    d.text((82,H-78), "cairogenizah.ai", font=serif(30,True), fill=CREAM)
    im.save(out, quality=92)
og(["Cairo Genizah AI"], ["AI search, machine transcription and maps","for 70,000+ medieval manuscript fragments."], OUT/"og-image.jpg", "גניזה")
og(["Yom Kippur in the","Cairo Genizah"], ["Fragments of Yom Kippur liturgy from a thousand","years ago, read by machine, on the manuscript."], OUT/"og-yom-kippur.jpg", "כיפור")
```

## Checking a preview

Crawlers cache aggressively. After a deploy, paste the URL into
https://developers.facebook.com/tools/debug/ and press "Scrape again" (this also
refreshes WhatsApp's cache, which shares Facebook's), or fetch as a crawler:

```bash
curl -s -A facebookexternalhit/1.1 https://cairogenizah.ai/yk | grep -oE '<meta (property|name)="(og|twitter):[^>]*>'
```
