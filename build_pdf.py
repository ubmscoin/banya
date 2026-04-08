#!/usr/bin/env python3
"""
Banya Framework — Merge all English HTML sections into one PDF-ready HTML,
then invoke Chrome headless to produce a single PDF.

Requirements: google-chrome (headless), Python 3.6+
"""
import os, re, sys, subprocess, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Document order (matches banya.html appendix) ──
DOC_ORDER = [
    "banya.html",
    "science_mine.html",
    "predictions.html",
    "lib.html",
    "discovery/alpha.html",
    "discovery/sin2_thetaW.html",
    "discovery/mass_hierarchy.html",
    "discovery/alpha57.html",
    "discovery/gauge.html",
    "discovery/baryogenesis.html",
    "discovery/ckm_pmns.html",
    "discovery/alpha_ladder.html",
    "discovery/alpha_structure.html",
    "discovery/lepton_ratio.html",
    "discovery/higgs_top.html",
    "discovery/M_W.html",
    "discovery/cas_internal.html",
    "discovery/coupling_relations.html",
    "discovery/cosmic_thermo.html",
    "discovery/ring_buffer.html",
    "discovery/lifetime.html",
    "discovery/quark_mass.html",
    "discovery/cosmology2.html",
    "discovery/atomic.html",
    "discovery/hadron_mass.html",
    "discovery/dimension_stack.html",
    "discovery/unification.html",
    "expend.html",
]

# ── File → first English anchor (for inter-doc link rewriting) ──
FILE_ANCHOR = {
    "banya.html":                       "en-sec1",
    "science_mine.html":                "en-title",
    "predictions.html":                 "en-title",
    "lib.html":                         "en-title",
    "discovery/alpha.html":             "en-title",
    "discovery/sin2_thetaW.html":       "en-title",
    "discovery/mass_hierarchy.html":    "en-title",
    "discovery/alpha57.html":           "en-title",
    "discovery/gauge.html":             "en-title",
    "discovery/baryogenesis.html":      "en-title",
    "discovery/ckm_pmns.html":          "en-title",
    "discovery/alpha_ladder.html":      "title",
    "discovery/alpha_structure.html":   "title",
    "discovery/lepton_ratio.html":      "title",
    "discovery/higgs_top.html":         "title",
    "discovery/M_W.html":              "title",
    "discovery/cas_internal.html":      "title",
    "discovery/coupling_relations.html":"title",
    "discovery/cosmic_thermo.html":     "title",
    "discovery/ring_buffer.html":       "title",
    "discovery/lifetime.html":          "title",
    "discovery/quark_mass.html":        "title",
    "discovery/cosmology2.html":        "title",
    "discovery/atomic.html":            "title",
    "discovery/hadron_mass.html":       "title",
    "discovery/dimension_stack.html":   "title",
    "discovery/unification.html":       "title",
    "expend.html":                      "en-title",
}

# ── Unique prefix per file (to avoid id collisions across files) ──
FILE_PREFIX = {
    "banya.html":                       "",
    "science_mine.html":                "sm-",
    "predictions.html":                 "pr-",
    "lib.html":                         "lib-",
    "discovery/alpha.html":             "alpha-",
    "discovery/sin2_thetaW.html":       "sw-",
    "discovery/mass_hierarchy.html":    "mh-",
    "discovery/alpha57.html":           "a57-",
    "discovery/gauge.html":             "ga-",
    "discovery/baryogenesis.html":      "ba-",
    "discovery/ckm_pmns.html":          "ckm-",
    "discovery/alpha_ladder.html":      "al-",
    "discovery/alpha_structure.html":   "as-",
    "discovery/lepton_ratio.html":      "lr-",
    "discovery/higgs_top.html":         "ht-",
    "discovery/M_W.html":             "mw-",
    "discovery/cas_internal.html":      "ci-",
    "discovery/coupling_relations.html":"cr-",
    "discovery/cosmic_thermo.html":     "ct-",
    "discovery/ring_buffer.html":       "rb-",
    "discovery/lifetime.html":          "lt-",
    "discovery/quark_mass.html":        "qm-",
    "discovery/cosmology2.html":        "co2-",
    "discovery/atomic.html":            "at-",
    "discovery/hadron_mass.html":       "hm-",
    "discovery/dimension_stack.html":   "ds-",
    "discovery/unification.html":       "un-",
    "expend.html":                      "exp-",
}

def read_file(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return f.read()

def extract_lang_en(html, filename):
    """Extract all <div class="lang-en" ...>...</div> blocks."""
    # Find all lang-en divs (they can be nested, so we do balanced brace matching)
    blocks = []
    pattern = r'<div\s+class="lang-en"[^>]*>'
    for m in re.finditer(pattern, html):
        start = m.start()
        # Find balanced closing </div>
        depth = 0
        i = start
        while i < len(html):
            if html[i:i+4] == '<div':
                depth += 1
            elif html[i:i+6] == '</div>':
                depth -= 1
                if depth == 0:
                    blocks.append(html[m.end():i])
                    break
            i += 1
    if not blocks:
        print(f"  WARNING: No lang-en content found in {filename}")
    return "\n".join(blocks)

def extract_en_footer(html):
    """Extract English footer if exists."""
    m = re.search(r'<div\s+class="doc-footer\s+lang-en"[^>]*>(.*?)</div>\s*(?=<div|<script|$)', html, re.DOTALL)
    if m:
        return m.group(0)
    return ""

def prefix_ids(content, prefix):
    """Add prefix to all id attributes to avoid collisions."""
    if not prefix:
        return content
    # Prefix id="..."
    content = re.sub(r'id="([^"]*)"', lambda m: f'id="{prefix}{m.group(1)}"', content)
    # Prefix href="#..." (internal anchors only)
    content = re.sub(r'href="#([^"]*)"', lambda m: f'href="#{prefix}{m.group(1)}"', content)
    return content

def rewrite_inter_doc_links(content, current_file):
    """Convert href="other.html" and href="other.html#anchor" to internal PDF anchors."""
    prefix_for_current = FILE_PREFIX.get(current_file, "")

    # Bare filenames that actually live in discovery/
    BARE_TO_DISCOVERY = {
        "alpha.html": "discovery/alpha.html",
        "sin2_thetaW.html": "discovery/sin2_thetaW.html",
        "mass_hierarchy.html": "discovery/mass_hierarchy.html",
        "alpha57.html": "discovery/alpha57.html",
        "gauge.html": "discovery/gauge.html",
        "baryogenesis.html": "discovery/baryogenesis.html",
        "ckm_pmns.html": "discovery/ckm_pmns.html",
        "alpha_ladder.html": "discovery/alpha_ladder.html",
        "alpha_structure.html": "discovery/alpha_structure.html",
        "lepton_ratio.html": "discovery/lepton_ratio.html",
        "higgs_top.html": "discovery/higgs_top.html",
        "M_W.html": "discovery/M_W.html",
        "cas_internal.html": "discovery/cas_internal.html",
        "coupling_relations.html": "discovery/coupling_relations.html",
        "cosmic_thermo.html": "discovery/cosmic_thermo.html",
        "ring_buffer.html": "discovery/ring_buffer.html",
        "lifetime.html": "discovery/lifetime.html",
        "quark_mass.html": "discovery/quark_mass.html",
        "cosmology2.html": "discovery/cosmology2.html",
        "atomic.html": "discovery/atomic.html",
        "hadron_mass.html": "discovery/hadron_mass.html",
        "dimension_stack.html": "discovery/dimension_stack.html",
        "unification.html": "discovery/unification.html",
    }

    def replace_link(m):
        full = m.group(0)
        filename = m.group(1)
        anchor = m.group(2) if m.group(2) else None

        # Resolve bare names to discovery/ paths
        if filename in BARE_TO_DISCOVERY:
            filename = BARE_TO_DISCOVERY[filename]

        if filename not in FILE_PREFIX:
            return full  # external link, keep as-is

        target_prefix = FILE_PREFIX[filename]
        if anchor:
            # Korean anchors (sec48, guide-what, d28, ch9 etc.) need "en-" prefix
            # because English sections use en-sec48, en-guide-what, en-d28 etc.
            en_anchor = anchor if anchor.startswith("en-") else f"en-{anchor}"
            return f'href="#{target_prefix}{en_anchor}"'
        else:
            # Link to file's first anchor
            first_anchor = FILE_ANCHOR.get(filename, "en-title")
            return f'href="#{target_prefix}{first_anchor}"'

    # Match href="filename.html" or href="discovery/filename.html" with optional #anchor
    content = re.sub(
        r'href="((?:discovery/)?[a-zA-Z0-9_]+\.html)(?:#([^"]*))?"',
        replace_link,
        content
    )
    return content

def remap_inline_colors(content):
    """Remap dark-theme inline style colors to white-base readable colors.
    Only touches inline style= attributes. Preserves colored badges/borders."""
    # Dark-theme light grays → dark readable on white
    color_map = {
        "#c9d1d9": "#1a1a1a",   # primary text (light gray → near black)
        "#e6edf3": "#000",      # heading/strong (white-ish → black)
        "#8b949e": "#555",      # secondary text (medium gray → darker gray)
        "#6e7681": "#666",      # tertiary/legend (light gray → readable gray)
        "#0d1117": "#fff",      # dark bg → white bg
        "#161b22": "#f5f5f5",   # code bg → light gray
        "#30363d": "#d0d0d0",   # border dark → border light
        "#21262d": "#e0e0e0",   # dark divider → light divider
        "#58a6ff": "#0366d6",   # link blue (too bright) → GitHub blue
        "#3fb950": "#1a7f37",   # green text (too bright on white) → darker green
    }
    for dark, light in color_map.items():
        content = content.replace(dark, light)
    return content

def build_combined_html():
    """Build a single combined HTML with all English content."""

    css_common = read_file("common.css")

    # PDF-specific overrides: white base, grayscale code blocks, colored badges/borders kept
    pdf_css = textwrap.dedent("""\
    /* PDF overrides — white base, grayscale code, colored badges/borders preserved */
    @media print {
      * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; color-adjust: exact !important; }
      body { max-width: none; padding: 20px 28px; font-size: 11pt; line-height: 1.6; background: #fff !important; color: #1a1a1a !important; }
      .lang-switch, .toc-side, .toc-bar, .toc-overlay, .page-nav { display: none !important; }
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
      /* === Colored elements preserved === */
      .discovery-block { background: #f0fff0 !important; border: 2px solid #2ea043 !important; page-break-inside: avoid; }
      .discovery-block h4 { color: #1a7f37 !important; }
      .discovery-block .formula { background: #fff !important; border: 1px solid #2ea043 !important; color: #000 !important; }
      .discovery-block .precision { color: #1a7f37 !important; }
      .hypothesis-block { background: #fffbf0 !important; border: 2px solid #d29922 !important; page-break-inside: avoid; }
      .hypothesis-block h4 { color: #9e6a03 !important; }
      .hypothesis-block .formula { background: #fff !important; border: 1px solid #d29922 !important; color: #000 !important; }
      .hypothesis-block .precision { color: #9e6a03 !important; }
      .prediction-block { background: #f0f6ff !important; border: 2px solid #1f6feb !important; page-break-inside: avoid; }
      .prediction-block h4 { color: #0550ae !important; }
      .prediction-block .formula { background: #fff !important; border: 1px solid #1f6feb !important; color: #000 !important; }
      .prediction-block .precision { color: #0550ae !important; }
      .tag-solved { background: #2ea043 !important; color: #fff !important; }
      .tag-discovery { background: #2ea043 !important; color: #fff !important; }
      .tag-hypothesis { background: #9e6a03 !important; color: #fff !important; }
      .tag-pending { background: #1f6feb !important; color: #fff !important; }
      .tag-partial { background: #9e6a03 !important; color: #fff !important; }
      .tag-wip { background: #888 !important; color: #fff !important; }
      .lib-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; page-break-inside: avoid; }
      .lib-card .lib-formula { background: #fff !important; color: #000 !important; }
      .lib-card .lib-precision { color: #1a7f37 !important; }
      .lib-card .lib-reuse { color: #9e6a03 !important; border-top: 1px solid #d0d0d0 !important; }
      .predict-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
      .predict-card .predict-value { color: #0550ae !important; }
      .predict-card .predict-test { color: #555 !important; }
      .step-block { background: #fafafa !important; border-left: 4px solid #2ea043 !important; }
      .step-block .step-num { color: #1a7f37 !important; }
      .flow-block { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
      .role-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
      .warn-block { background: #fffbf0 !important; border: 1px solid #d29922 !important; color: #9e6a03 !important; }
      .hier-tree .hier-root { color: #000 !important; }
      .hier-tree .hier-l1 { border-left: 2px solid #2ea043 !important; color: #1a1a1a !important; }
      .hier-tree .hier-l2 { border-left: 2px solid #bbb !important; color: #555 !important; }
      .doc-footer { display: none !important; }
      .doc-section-break { page-break-before: always; margin-top: 0; padding-top: 20px; }
    }
    /* Screen styles for preview — also white base */
    body { background: #fff !important; color: #1a1a1a !important; }
    h1, h2, h3 { color: #000 !important; }
    h4 { color: #222 !important; }
    h1 { border-bottom: 2px solid #ccc !important; }
    h2 { border-bottom: 1px solid #ccc !important; }
    a { color: #0366d6 !important; }
    strong { color: #000 !important; }
    pre { background: #f5f5f5 !important; border: 1px solid #d0d0d0 !important; color: #1a1a1a !important; }
    code { background: #f5f5f5 !important; color: #1a1a1a !important; }
    pre code { background: none !important; }
    th { background: #f0f0f0 !important; color: #000 !important; }
    td { color: #1a1a1a !important; }
    th, td { border: 1px solid #bbb !important; }
    blockquote { border-left: 4px solid #ccc !important; color: #555 !important; }
    hr { border-top: 1px solid #ccc !important; }
    .math-block { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
    .math-block .math-desc { color: #555 !important; }
    .math-block .math-legend { color: #777 !important; border-top: 1px solid #d0d0d0 !important; }
    .discovery-block { background: #f0fff0 !important; border: 2px solid #2ea043 !important; }
    .discovery-block h4 { color: #1a7f37 !important; }
    .discovery-block .formula { background: #fff !important; border: 1px solid #2ea043 !important; color: #000 !important; }
    .discovery-block .precision { color: #1a7f37 !important; }
    .hypothesis-block { background: #fffbf0 !important; border: 2px solid #d29922 !important; }
    .hypothesis-block h4 { color: #9e6a03 !important; }
    .hypothesis-block .formula { background: #fff !important; border: 1px solid #d29922 !important; color: #000 !important; }
    .hypothesis-block .precision { color: #9e6a03 !important; }
    .prediction-block { background: #f0f6ff !important; border: 2px solid #1f6feb !important; }
    .prediction-block h4 { color: #0550ae !important; }
    .prediction-block .formula { background: #fff !important; border: 1px solid #1f6feb !important; color: #000 !important; }
    .prediction-block .precision { color: #0550ae !important; }
    .tag-wip { background: #888 !important; color: #fff !important; }
    .lib-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
    .lib-card .lib-formula { background: #fff !important; color: #000 !important; }
    .lib-card .lib-precision { color: #1a7f37 !important; }
    .lib-card .lib-reuse { color: #9e6a03 !important; border-top: 1px solid #d0d0d0 !important; }
    .predict-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
    .predict-card .predict-value { color: #0550ae !important; }
    .predict-card .predict-test { color: #555 !important; }
    .step-block { background: #fafafa !important; border-left: 4px solid #2ea043 !important; }
    .step-block .step-num { color: #1a7f37 !important; }
    .flow-block { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
    .role-card { background: #fafafa !important; border: 1px solid #d0d0d0 !important; }
    .warn-block { background: #fffbf0 !important; border: 1px solid #d29922 !important; color: #9e6a03 !important; }
    .hier-tree .hier-root { color: #000 !important; }
    .hier-tree .hier-l1 { border-left: 2px solid #2ea043 !important; color: #1a1a1a !important; }
    .hier-tree .hier-l2 { border-left: 2px solid #bbb !important; color: #555 !important; }
    .lang-ko { display: none !important; }
    .lang-en { display: block !important; }
    .lang-switch, .toc-side, .toc-bar, .toc-overlay, .page-nav { display: none !important; }
    .doc-footer.lang-ko { display: none !important; }
    .doc-section-break { page-break-before: always; margin-top: 40px; padding-top: 20px; border-top: 3px solid #2ea043; }
    """)

    # Build HTML
    parts = []
    parts.append(textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Banya Framework — Comprehensive Report (English)</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
          crossorigin="anonymous">
    <style>
    {css_common}
    {pdf_css}
    </style>
    </head>
    <body>
    """))

    for idx, filename in enumerate(DOC_ORDER):
        print(f"Processing [{idx+1}/{len(DOC_ORDER)}] {filename} ...")
        # Read from en/ folder (standalone English files)
        en_filepath = os.path.join(BASE, "en", filename)
        if not os.path.exists(en_filepath):
            print(f"  SKIP: {en_filepath} not found")
            continue
        with open(en_filepath, "r", encoding="utf-8") as f:
            html = f.read()

        # Extract body content
        body_match = re.search(r'<body[^>]*>(.*)</body>', html, re.DOTALL)
        if not body_match:
            print(f"  SKIP: no <body> in {filename}")
            continue
        en_content = body_match.group(1)

        prefix = FILE_PREFIX[filename]

        # 1. Remap dark-theme inline colors to white-base readable colors
        en_content = remap_inline_colors(en_content)

        # 2. Prefix IDs to avoid collision
        en_content = prefix_ids(en_content, prefix)

        # 3. Rewrite inter-document links
        en_content = rewrite_inter_doc_links(en_content, filename)

        # Add section break (except for first doc)
        if idx > 0:
            parts.append(f'\n<div class="doc-section-break"></div>\n')

        parts.append(f'<!-- === {filename} === -->\n')
        parts.append(en_content)
        parts.append('\n')

    # KaTeX auto-render script
    parts.append(textwrap.dedent("""\
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
            crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
            crossorigin="anonymous"
            onload="renderMathInElement(document.body,{
              delimiters:[
                {left:'$$',right:'$$',display:true},
                {left:'$',right:'$',display:false}
              ],
              throwOnError:false
            })"></script>
    </body>
    </html>
    """))

    combined = "".join(parts)
    outpath = os.path.join(BASE, "banya_combined_en.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\nCombined HTML written: {outpath}")
    print(f"Size: {len(combined):,} bytes")
    return outpath

def generate_pdf(html_path):
    """Use Chrome headless to generate PDF."""
    pdf_path = html_path.replace(".html", ".pdf")

    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-software-rasterizer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=30000",   # 30s for KaTeX to render
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        f"file://{html_path}",
    ]

    print(f"\nGenerating PDF with Chrome headless...")
    print(f"Command: {' '.join(cmd[:6])} ...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024*1024)
        print(f"PDF generated: {pdf_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print(f"Chrome stderr: {result.stderr[:500]}")
        print(f"Chrome stdout: {result.stdout[:500]}")
        if os.path.exists(pdf_path):
            size_mb = os.path.getsize(pdf_path) / (1024*1024)
            print(f"PDF may have been generated: {pdf_path} ({size_mb:.1f} MB)")

    return pdf_path

if __name__ == "__main__":
    html_path = build_combined_html()
    pdf_path = generate_pdf(html_path)
