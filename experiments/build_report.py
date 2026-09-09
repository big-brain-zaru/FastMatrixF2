"""Render docs/SCIENTIFIC-REPORT.md to a self-contained HTML page (figures embedded as data URIs)."""
import base64, os, re, markdown

DOCS = "../docs"
md_src = open(os.path.join(DOCS, "SCIENTIFIC-REPORT.md"), encoding="utf-8").read()

def embed(m):
    path = os.path.join(DOCS, m.group(2))
    b = base64.b64encode(open(path, "rb").read()).decode()
    return '![%s](data:image/png;base64,%s)' % (m.group(1), b)
md_emb = re.sub(r'!\[([^\]]*)\]\((figures/[^)]+)\)', embed, md_src)
body = markdown.markdown(md_emb, extensions=["tables", "toc", "fenced_code", "attr_list", "md_in_html"], extension_configs={"toc": {"toc_depth": "2-3"}})
# figure captions: paragraphs that are only an image followed by an italic paragraph
body = re.sub(r'<p>(<img[^>]+>)</p>\s*<p><em>(.*?)</em></p>', r'<figure>\1<figcaption>\2</figcaption></figure>', body, flags=re.S)

page = """<title>Fast Matrix Multiplication over F2</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f4f6f8; --ink:#17212b; --ink-2:#4a5866; --rule:#cfd6dd; --panel:#ffffff;
  --accent:#1d5f7a; --accent-ink:#ffffff; --unsat:#3f9a6b; --sat:#3a62c4; --budget:#c48a3a; --mark:#e8eef2;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#12181e; --ink:#e6ebf0; --ink-2:#a6b2bd; --rule:#2b3640; --panel:#1a222b;
  --accent:#6fb3cf; --accent-ink:#0d1319; --unsat:#5fbd8a; --sat:#7c9ce8; --budget:#d9a45a; --mark:#1f2a34; } }
:root[data-theme="dark"]{
  --paper:#12181e; --ink:#e6ebf0; --ink-2:#a6b2bd; --rule:#2b3640; --panel:#1a222b;
  --accent:#6fb3cf; --accent-ink:#0d1319; --unsat:#5fbd8a; --sat:#7c9ce8; --budget:#d9a45a; --mark:#1f2a34; }
html{color-scheme:light dark}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,Segoe UI,Arial,sans-serif;font-size:16px;line-height:1.55;margin:0}
.wrap{max-width:860px;margin:0 auto;padding:40px 24px 80px}
h1,h2,h3{font-family:"Newsreader",Georgia,"Times New Roman",serif;font-weight:600;line-height:1.15;text-wrap:balance;color:var(--ink)}
h1{font-size:2.5rem;margin:0 0 .3em;letter-spacing:-.01em}
h2{font-size:1.65rem;margin:2.4em 0 .6em;padding-top:.6em;border-top:1px solid var(--rule)}
h3{font-size:1.2rem;margin:1.8em 0 .4em}
h4{font-family:"IBM Plex Sans",sans-serif;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);margin:1.6em 0 .3em}
p,li{max-width:70ch}
a{color:var(--accent)}
.meta{color:var(--ink-2);font-size:.95rem;margin-bottom:2em}
blockquote{margin:1.2em 0;padding:.6em 1em;border-left:3px solid var(--accent);background:var(--mark);color:var(--ink)}
blockquote p{margin:.3em 0}
code{font-family:"IBM Plex Mono",Consolas,monospace;font-size:.88em;background:var(--mark);padding:.05em .3em;border-radius:3px}
pre{background:var(--mark);padding:.8em 1em;overflow-x:auto;border-radius:4px}
pre code{background:none;padding:0}
.tbl{overflow-x:auto;margin:1em 0}
table{border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:.45em .7em;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{font-family:"IBM Plex Sans",sans-serif;font-weight:600;font-size:.78rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2);border-bottom:2px solid var(--rule)}
td:first-child{white-space:nowrap}
figure{margin:1.6em 0;background:var(--panel);border:1px solid var(--rule);padding:12px 12px 8px;border-radius:4px}
figure img{max-width:100%;display:block;margin:0 auto}
figcaption{font-size:.85rem;color:var(--ink-2);margin-top:.6em;line-height:1.45}
.toc{background:var(--panel);border:1px solid var(--rule);padding:.8em 1.2em;border-radius:4px;font-size:.92rem;margin:1.5em 0 2.5em}
.toc ul{margin:.2em 0;padding-left:1.2em}
.toc>ul{padding-left:0;list-style:none}
.toc>ul>li{margin:.25em 0}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.72rem;padding:.05em .45em;border-radius:3px;margin-right:.3em;color:var(--accent-ink)}
.t-unsat{background:var(--unsat)} .t-sat{background:var(--sat)} .t-budget{background:var(--budget)}
hr{border:0;border-top:1px solid var(--rule);margin:2em 0}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
</style>
<div class="wrap">
__BODY__
</div>
""".replace("__BODY__", body)
# wrap tables for horizontal scrolling
page = page.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
out = os.path.join(DOCS, "SCIENTIFIC-REPORT.html")
open(out, "w", encoding="utf-8").write(page)
print("wrote", out, len(page) // 1024, "KB")
