#!/usr/bin/env python3
"""
Banya Framework — Build banya_main PDF (KR + EN).

Unlike build_pdf.py (which merges all 28 pages into one PDF),
this builds a SINGLE-PAGE PDF from banya_main.html (or en/banya.html),
keeping all cross-page links as absolute URLs pointing to
https://ubmscoin.github.io/banya/ (and /en/ for English).

Requirements: google-chrome (headless), Python 3.6+
"""
import os, re, sys, subprocess, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))
WEB_BASE = "https://ubmscoin.github.io/banya"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PDF_CSS = textwrap.dedent("""\
/* PDF overrides — white base, grayscale code, colored badges/borders preserved */
@media print {
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; color-adjust: exact !important; orphans: 3; widows: 3; }
  body { max-width: none; padding: 16px 26px; font-size: 11pt; line-height: 1.55; background: #fff !important; color: #1a1a1a !important; }
  .lang-switch, .toc-side, .toc-bar, .toc-overlay, .page-nav { display: none !important; }
  /* Tighter margins + page-break policy to avoid mid-document gaps */
  h1 { margin-top: 24px !important; margin-bottom: 12px !important; page-break-before: auto; }
  h2 { margin-top: 28px !important; margin-bottom: 12px !important; page-break-after: avoid; page-break-inside: avoid; }
  h3 { margin-top: 18px !important; margin-bottom: 10px !important; page-break-after: avoid; page-break-inside: avoid; }
  h4 { margin-top: 14px !important; margin-bottom: 8px !important; page-break-after: avoid; page-break-inside: avoid; }
  p { margin: 0 0 10px 0 !important; orphans: 3; widows: 3; }
  ul, ol { margin: 8px 0 !important; padding-left: 24px !important; }
  li { margin: 4px 0 !important; }
  hr { margin: 18px 0 !important; border-top: 1px solid #ccc !important; }
  blockquote { margin: 10px 0 !important; padding: 6px 14px !important; page-break-inside: avoid; }
  .table-wrap, table { page-break-inside: avoid; }
  details { margin: 10px 0 !important; padding: 6px 10px !important; background: #fafafa !important; border: 1px solid #e0e0e0 !important; border-radius: 4px !important; font-size: 0.78em !important; line-height: 1.5 !important; color: #555 !important; page-break-inside: avoid; }
  details > summary { font-size: 0.95em !important; color: #444 !important; font-weight: 600 !important; cursor: default !important; list-style: none !important; margin-bottom: 6px !important; }
  details > summary::-webkit-details-marker { display: none !important; }
  details blockquote { margin: 4px 0 !important; padding: 2px 12px !important; border-left: 2px solid #d0d0d0 !important; }
  details blockquote p { margin: 0 !important; font-size: 1em !important; }
  /* Force details to expand */
  details { display: block !important; }
  details > summary { display: list-item !important; color: #444 !important; font-weight: 600; margin-bottom: 8px; }
  details > *:not(summary) { display: block !important; }
  details[open], details { /* open by default in PDF */ }
  /* Quick Nav card grid — keep visible, adapt to A4 */
  .quicknav-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 8px !important; margin: 18px 0 !important; page-break-inside: avoid; }
  .quicknav-grid .qn-card { display: block !important; padding: 10px 12px !important; border: 1px solid #d0d0d0 !important; border-radius: 4px !important; background: #fafafa !important; color: #1a1a1a !important; text-decoration: none !important; line-height: 1.4 !important; font-size: 0.85em !important; }
  .quicknav-grid .qn-card strong { color: #0366d6 !important; font-size: 1em !important; display: block !important; margin-bottom: 2px !important; }
  h1 { font-size: 1.6em; color: #000 !important; border-bottom: 2px solid #ccc !important; page-break-after: avoid; }
  h2 { font-size: 1.3em; color: #000 !important; border-bottom: 1px solid #ccc !important; page-break-after: avoid; }
  h3 { font-size: 1.15em; color: #111 !important; page-break-after: avoid; }
  h4 { font-size: 1.0em; color: #222 !important; page-break-after: avoid; }
  a { color: #0366d6 !important; }
  pre { background: #f5f5f5 !important; border: 1px solid #d0d0d0 !important; color: #1a1a1a !important; page-break-inside: avoid; }
  code { background: #f5f5f5 !important; color: #1a1a1a !important; }
  pre code { background: none !important; }
  table { font-size: 0.82em; page-break-inside: avoid; display: table; width: 100%; }
  th { background: #f0f0f0 !important; color: #000 !important; }
  td { color: #1a1a1a !important; }
  th, td { border: 1px solid #bbb !important; padding: 6px 10px; }
  strong { color: #000 !important; }
  blockquote { border-left: 4px solid #ccc !important; color: #555 !important; }
  hr { border-top: 1px solid #ccc !important; }
  .math-block { background: #fafafa !important; border: 1px solid #d0d0d0 !important; page-break-inside: avoid; }
  .math-block .math-desc { color: #555 !important; }
  .math-block .math-legend { color: #777 !important; border-top: 1px solid #d0d0d0 !important; }
  .tag-solved { background: #2ea043 !important; color: #fff !important; }
  .tag-corollary { background: #e3742f !important; color: #fff !important; }
  /* AI mining box (special — keep dark on light for visibility) */
  #ai-mining { background: #fafafa !important; border: 1px solid #d29922 !important; color: #1a1a1a !important; font-size: 0.9em !important; padding: 10px 14px !important; }
  #ai-mining pre { background: #f0f0f0 !important; color: #333 !important; border: 1px solid #ccc !important; font-size: 0.72em !important; line-height: 1.4 !important; padding: 8px 10px !important; margin: 4px 0 8px 0 !important; page-break-inside: avoid !important; page-break-before: avoid !important; break-inside: avoid !important; break-before: avoid !important; }
  #ai-mining pre a { font-size: 1em !important; color: #0366d6 !important; }
  #ai-mining button { display: none !important; }
  /* Footer — readable on white, clean hierarchy */
  .doc-footer { border-top: 2px solid #d0d0d0 !important; padding-top: 18px !important; margin-top: 28px !important; color: #1a1a1a !important; page-break-inside: avoid; }
  .doc-footer .f-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
  .doc-footer .f-corp { color: #1a1a1a !important; font-size: 0.95em; }
  .doc-footer .f-corp b { color: #000 !important; font-size: 1.05em; }
  .doc-footer .f-right { color: #1a1a1a !important; text-align: right; }
  .doc-footer .f-right .f1 { color: #000 !important; font-weight: 600; }
  .doc-footer .f-right .f2 { color: #444 !important; font-size: 0.9em; margin-top: 4px; }
  .doc-footer .f-bottom { color: #333 !important; border-top: 1px solid #e0e0e0 !important; padding-top: 12px !important; }
  .doc-footer .f-copy { color: #444 !important; font-size: 0.9em; margin-bottom: 8px; }
  .doc-footer .f-license { color: #444 !important; font-size: 0.85em; line-height: 1.5; }
  .doc-footer .license-title { color: #000 !important; font-weight: 600; margin-bottom: 4px; }
  .doc-footer .license-cite { background: #f5f5f5 !important; color: #555 !important; margin-top: 8px !important; padding: 6px 10px !important; border-radius: 4px !important; border: 1px solid #e0e0e0 !important; font-size: 0.85em !important; }
}
/* Screen styles for preview — also white base */
body { background: #fff !important; color: #1a1a1a !important; }
h1, h2, h3 { color: #000 !important; }
h4 { color: #222 !important; }
a { color: #0366d6 !important; }
strong { color: #000 !important; }
pre { background: #f5f5f5 !important; border: 1px solid #d0d0d0 !important; color: #1a1a1a !important; }
code { background: #f5f5f5 !important; color: #1a1a1a !important; }
pre code { background: none !important; }
th { background: #f0f0f0 !important; color: #000 !important; }
td { color: #1a1a1a !important; }
th, td { border: 1px solid #bbb !important; }
blockquote { border-left: 4px solid #ccc !important; color: #555 !important; }
.math-block { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
.lang-switch, .toc-side, .toc-bar, .toc-overlay, .page-nav { display: none !important; }
/* Force details to expand (screen preview too) */
details > *:not(summary) { display: block !important; }
details > summary { color: #444 !important; font-weight: 600; }
/* Quick Nav card grid — screen */
.quicknav-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 10px !important; margin: 18px 0 !important; }
.quicknav-grid .qn-card { display: block !important; padding: 12px 14px !important; border: 1px solid #d0d0d0 !important; border-radius: 6px !important; background: #fafafa !important; color: #1a1a1a !important; text-decoration: none !important; line-height: 1.45 !important; font-size: 0.92em !important; }
.quicknav-grid .qn-card strong { color: #0366d6 !important; font-size: 1em !important; display: block !important; margin-bottom: 3px !important; }
/* Footer — readable on white */
.doc-footer { border-top: 2px solid #d0d0d0 !important; padding-top: 18px !important; margin-top: 28px !important; color: #1a1a1a !important; }
.doc-footer .f-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; flex-wrap: wrap; gap: 16px; }
.doc-footer .f-corp { color: #1a1a1a !important; font-size: 0.95em; }
.doc-footer .f-corp b { color: #000 !important; font-size: 1.05em; }
.doc-footer .f-right { color: #1a1a1a !important; text-align: right; }
.doc-footer .f-right .f1 { color: #000 !important; font-weight: 600; }
.doc-footer .f-right .f2 { color: #444 !important; font-size: 0.9em; margin-top: 4px; }
.doc-footer .f-bottom { color: #333 !important; border-top: 1px solid #e0e0e0 !important; padding-top: 12px !important; }
.doc-footer .f-copy { color: #444 !important; font-size: 0.9em; margin-bottom: 8px; }
.doc-footer .f-license { color: #444 !important; font-size: 0.85em; line-height: 1.5; }
.doc-footer .license-title { color: #000 !important; font-weight: 600; margin-bottom: 4px; }
.doc-footer .license-cite { background: #f5f5f5 !important; color: #555 !important; margin-top: 8px !important; padding: 6px 10px !important; border-radius: 4px !important; border: 1px solid #e0e0e0 !important; font-size: 0.85em !important; }
""")


def remap_inline_colors(html):
    """Dark-theme inline colors → white-base readable colors."""
    color_map = {
        "#c9d1d9": "#1a1a1a",
        "#e6edf3": "#000",
        "#8b949e": "#555",
        "#6e7681": "#666",
        "#0d1117": "#fff",
        "#161b22": "#f5f5f5",
        "#30363d": "#d0d0d0",
        "#21262d": "#e0e0e0",
        "#58a6ff": "#0366d6",
        "#3fb950": "#1a7f37",
    }
    for dark, light in color_map.items():
        html = html.replace(dark, light)
    return html


def rewrite_links_to_web(html, web_base, source_dir):
    """Convert relative href to absolute https://ubmscoin.github.io/banya/... URLs.

    Rules:
    - href="axiom.html"             → href="{web_base}/axiom.html"
    - href="axiom.html#ax1"         → href="{web_base}/axiom.html#ax1"
    - href="report/alpha137/x.pdf"  → href="{web_base}/report/alpha137/x.pdf"
    - href="../banya_en.pdf"        → href="{web_base}/banya_en.pdf" (parent ref normalized)
    - href="#sec1"                  → unchanged (same-page anchor)
    - href="https://..."            → unchanged (already absolute)
    - href="mailto:..."             → unchanged
    """
    def replace_href(m):
        url = m.group(1)
        # Already absolute or special protocol
        if url.startswith(("http://", "https://", "mailto:", "tel:", "javascript:")):
            return m.group(0)
        # Same-page anchor
        if url.startswith("#"):
            return m.group(0)
        # Parent reference (e.g. ../banya_en.pdf from en/banya.html → banya_en.pdf at root)
        if url.startswith("../"):
            normalized = url[3:]
            return f'href="{web_base}/{normalized}"'
        # Relative path from source_dir: source_dir is either "" (root) or "en"
        if source_dir:
            return f'href="{web_base}/{source_dir}/{url}"'
        return f'href="{web_base}/{url}"'

    return re.sub(r'href="([^"]+)"', replace_href, html)


def build_single_pdf(source_html, output_name, source_dir, label):
    """Build a single-page PDF from one HTML file with web-absolute links."""
    print(f"\n=== Building {label} PDF ===")
    src_path = os.path.join(BASE, source_html)
    print(f"Source: {src_path}")

    html = read_file(src_path)

    # Extract <head> title + meta and <body> content
    title_m = re.search(r'<title>([^<]+)</title>', html)
    title = title_m.group(1) if title_m else "Banya Framework"

    body_m = re.search(r'<body[^>]*>(.*)</body>', html, re.DOTALL)
    if not body_m:
        print(f"  ERROR: no <body> in {source_html}")
        return None
    body_content = body_m.group(1)

    # 1. Color remap
    body_content = remap_inline_colors(body_content)
    # 2. Rewrite links to absolute web URLs
    body_content = rewrite_links_to_web(body_content, WEB_BASE, source_dir)
    # 3. Force <details> to be open by default in PDF
    body_content = re.sub(r'<details(\s|>)', r'<details open\1', body_content)
    # 4. Inject click-to-live banner above Quick Nav grid
    quicknav_url = f"{WEB_BASE}/{source_dir}/banya.html#quicknav" if source_dir else f"{WEB_BASE}/banya_main.html#quicknav"
    if source_dir == "en":
        banner = (
            '<a href="' + quicknav_url + '" '
            'style="display:block;background:#fff3cd;border:2px solid #d29922;border-radius:6px;'
            'padding:12px 18px;margin:0 0 14px 0;color:#664d03;text-decoration:none;'
            'font-weight:600;text-align:center;font-size:0.95em;line-height:1.5;'
            'page-break-before:always;break-before:page">'
            'Each card below is clickable in this PDF — jump to the live page<br>'
            '<span style="font-size:0.88em;color:#0366d6;text-decoration:underline">'
            + quicknav_url + '</span></a>'
        )
    else:
        banner = (
            '<a href="' + quicknav_url + '" '
            'style="display:block;background:#fff3cd;border:2px solid #d29922;border-radius:6px;'
            'padding:12px 18px;margin:0 0 14px 0;color:#664d03;text-decoration:none;'
            'font-weight:600;text-align:center;font-size:0.95em;line-height:1.5;'
            'page-break-before:always;break-before:page">'
            '아래 카드는 PDF에서도 클릭 가능합니다 — 라이브 페이지로 이동<br>'
            '<span style="font-size:0.88em;color:#0366d6;text-decoration:underline">'
            + quicknav_url + '</span></a>'
        )
    body_content = re.sub(
        r'(<div id="quicknav"[^>]*>)',
        banner + r'\1',
        body_content,
        count=1,
    )

    css_common = read_file(os.path.join(BASE, "common.css"))

    combined = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="{'en' if source_dir == 'en' else 'ko'}">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
    <style>
    {css_common}
    {PDF_CSS}
    </style>
    </head>
    <body>
    """) + body_content + textwrap.dedent("""\
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"
            onload="renderMathInElement(document.body,{
              delimiters:[
                {left:'$$',right:'$$',display:true},
                {left:'$',right:'$',display:false}
              ],
              throwOnError:false
            })"></script>
    </body>
    </html>
    """)

    html_out = os.path.join(BASE, output_name + ".html")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"Combined HTML: {html_out}  ({len(combined):,} bytes)")

    pdf_out = os.path.join(BASE, output_name + ".pdf")
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-software-rasterizer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=30000",
        f"--print-to-pdf={pdf_out}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        f"file://{html_out}",
    ]
    print(f"Running Chrome headless...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if os.path.exists(pdf_out):
        size_mb = os.path.getsize(pdf_out) / (1024 * 1024)
        print(f"PDF: {pdf_out}  ({size_mb:.2f} MB)")
    else:
        print(f"  ERROR: PDF not produced")
        print(f"  stderr: {result.stderr[:500]}")
    return pdf_out


if __name__ == "__main__":
    # Korean
    build_single_pdf(
        source_html="banya_main.html",
        output_name="banya_main_kr",
        source_dir="",  # root level
        label="Korean (banya_main.html)",
    )
    # English
    build_single_pdf(
        source_html="en/banya.html",
        output_name="banya_main_en",
        source_dir="en",
        label="English (en/banya.html)",
    )
