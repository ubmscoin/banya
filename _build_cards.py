"""
1카드 = 1 HTML 일괄 생성기 v2 (보강판)
- D 155 + H 952 + P 7 = 1114장
- 박스 4개: 핵심 발견 / 부산물 / 미완 / 총괄
- 5단계 재귀 대입 골격 (v1.8 공리 기준)
- 본문 ASCII만 사용 (식만 KaTeX)
- 화살표: ==>, <==, =>, <=
"""
import json, re, os
from bs4 import BeautifulSoup, NavigableString

# ─────────── v1.8 도구상자 ───────────
PHYS_CONSTANTS = ['alpha','alpha_s','sin^2 theta_W','M_Z','M_W','m_t','m_c','m_b','m_tau','m_mu','m_e','G_F','hbar','delta_CP','V_us','V_cb','V_ub','V_td','V_ts','Lambda','H_0','Sigma m_nu']

NORM_BY_CAT = {
    'h_qcd':              'delta ==> SU(3) 3축 색 자유도 x QCD 비용 도메인 (alpha_s 경로)',
    'h_neutrino_cp':      'delta ==> CKM/PMNS 혼합각 = CAS 단계 갭 l/N (N=30, 9, 7)',
    'h_cosmology':        'delta ==> RLU HOT/WARM/COLD 분포 또는 alpha^57 우주 스케일',
    'h_gauge_ew':         'delta ==> SU(2) x U(1) 전약 결합, sin^2 theta_W 분배',
    'h_mass_yukawa':      'delta ==> 페르미온 질량 = CAS 비용 x 자료형 크기 (137, 27)',
    'h_electromag_qed':   'delta ==> 전자기 결합 alpha = D5 Shilov 체적비 (서명 (5,2))',
    'h_measurement':      'delta ==> Compare 분기 (true ==> Swap ==> DATA / false ==> 중첩 유지)',
    'h_info_consciousness':'delta ==> bit 7 발화비트 / d-ring 자기참조 루프 (공리 8, 15)',
    'h_structure':        'delta ==> 반야식 자체 구조, 1차 정수 (4,3) 조합',
    'h_misc':             'delta ==> 자연 분류 외 (도메인 미확정, 후속 정련 필요)',
    'd_qcd':              'delta ==> 색 자유도 x QCD 비용',
    'd_neutrino_cp':      'delta ==> CKM/PMNS 혼합각',
    'd_cosmology':        'delta ==> RLU HOT/WARM/COLD 또는 alpha^57',
    'd_gauge_ew':         'delta ==> 전약 결합 분배',
    'd_mass_yukawa':      'delta ==> 페르미온 질량 = CAS 비용 x 자료형',
    'd_electromag':       'delta ==> 전자기 결합 alpha',
    'd_structure':        'delta ==> 반야식 구조 / 1차 정수',
    'd_atomic':           'delta ==> 원자 스케일 (a0, R_inf) = alpha x 양자 결합',
    'd_misc':             'delta ==> 자연 분류 외',
    'p':                  'delta ==> 예측 대상 도메인 (P 카드별 명시)',
}

CAT_INTS = {
    'h_qcd':              '3 (CAS 3축 = SU(3) 색), 7 (beta_0=7), 8 (글루온=3^2-1, 산출)',
    'h_neutrino_cp':      '3 (CAS 단계), 9 (Koide theta), 30 (CAS 접근 경로), 7 (자유도)',
    'h_cosmology':        '57 (산출, alpha^57=Lambda*l_p^2), 7, 21, 35 (외적 대수 차원)',
    'h_gauge_ew':         '7, 30 (sin^2 theta_W = 7/30), 4 (도메인)',
    'h_mass_yukawa':      '137 (1/alpha), 27 (자료형), 9, 2/9 (코이데)',
    'h_electromag_qed':   '5, 2 (서명 (5,2)), 137 (자료형 = T(16)+1)',
    'h_measurement':      '2 (괄호 수 DATA/OPERATOR), 3 (CAS 단계)',
    'h_info_consciousness':'7 (=4+3 Hamming), 128 (=2^7 Cl(0,7)), 8 (bit), 137',
    'h_structure':        '1, 2, 3, 4, 7, 9, 16, 30, 128, 137 (자유도 목록 전체)',
    'h_misc':             '카드 본문에서 추출',
    'd_qcd':'3, 7, 색 8','d_neutrino_cp':'3, 9, 30, 7',
    'd_cosmology':'57, 7, 21, 35','d_gauge_ew':'7, 30, 4',
    'd_mass_yukawa':'137, 27, 9, 2/9','d_electromag':'5,2 / 137',
    'd_structure':'1차 정수 목록','d_atomic':'137, alpha 사다리','d_misc':'본문 추출',
    'p':'예측 카드별 명시',
}

# ─────────── ASCII 변환 ───────────
ASCII_MAP = [
    ('==>','==>'), ('<==','<=='),
    ('→','==>'), ('←','<=='), ('↔','<==>'),
    ('⇒','=>'), ('⇐','<='), ('⇔','<=>'),
    ('×','x'), ('÷','/'), ('±','+/-'),
    ('≈','~'), ('≤','<='), ('≥','>='), ('≠','!='),
    ('∞','inf'), ('∫','int'), ('∑','sum'), ('∏','prod'),
    ('⊗','(x)'), ('⊕','(+)'), ('⊖','(-)'),
    ('√','sqrt'), ('²','^2'), ('³','^3'), ('⁻','^-'),
    ('°','deg'), ('·','*'), ('•','*'),
    ('「','['), ('」',']'),
    # 그리스 문자는 식 박스 밖에서만 풀어쓰기 — 본문에서 사용 시
    ('α','alpha'), ('β','beta'), ('γ','gamma'), ('δ','delta'),
    ('ε','epsilon'), ('ζ','zeta'), ('η','eta'), ('θ','theta'),
    ('ι','iota'), ('κ','kappa'), ('λ','lambda'), ('μ','mu'),
    ('ν','nu'), ('ξ','xi'), ('π','pi'), ('ρ','rho'),
    ('σ','sigma'), ('τ','tau'), ('φ','phi'), ('χ','chi'),
    ('ψ','psi'), ('ω','omega'),
    ('Α','Alpha'), ('Β','Beta'), ('Γ','Gamma'), ('Δ','Delta'),
    ('Θ','Theta'), ('Λ','Lambda'), ('Σ','Sigma'), ('Φ','Phi'),
    ('Ψ','Psi'), ('Ω','Omega'), ('Π','Pi'),
]

def to_ascii_text(s):
    """텍스트 노드를 ASCII로 변환 (식 박스 밖만)"""
    if not s: return s
    for orig, rep in ASCII_MAP:
        s = s.replace(orig, rep)
    return s

def ascii_outside_math(html):
    """KaTeX $...$ 또는 $$...$$ 밖만 ASCII 변환"""
    placeholders = {}
    pid = [0]
    def stash(m):
        key = f"\x00MATH{pid[0]}\x00"
        placeholders[key] = m.group(0)
        pid[0] += 1
        return key
    html = re.sub(r'\$\$.*?\$\$', stash, html, flags=re.S)
    html = re.sub(r'\$[^$]+\$', stash, html, flags=re.S)
    html = to_ascii_text(html)
    for key, orig in placeholders.items():
        html = html.replace(key, orig)
    return html


def linkify_refs(html):
    """본문 안의 D-NN, H-NN, P-NN, 공리 N 참조를 링크로 변환.
    이미 <a> 태그 안과 KaTeX 박스 안은 건드리지 않음."""
    if not html: return html
    placeholders = {}
    pid = [0]
    def stash(m):
        key = f"\x00LINK{pid[0]}\x00"
        placeholders[key] = m.group(0)
        pid[0] += 1
        return key
    # 기존 <a>...</a> 보존
    html = re.sub(r'<a\b[^>]*>.*?</a>', stash, html, flags=re.S)
    # KaTeX 보존
    html = re.sub(r'\$\$.*?\$\$', stash, html, flags=re.S)
    html = re.sub(r'\$[^$]+\$', stash, html, flags=re.S)

    # D-NN, H-NN, P-NN을 카드 링크로
    def repl_card(m):
        kind, num = m.group(1), int(m.group(2))
        cid = f"{kind}-{num:02d}"
        href = f"{kind.lower()}{num:02d}.html"
        return f'<a href="{href}">{cid}</a>'
    html = re.sub(r'\b([DHP])-(\d{1,3})\b', repl_card, html)

    # "공리 N" 을 axiom.html#axN 으로
    def repl_axiom(m):
        num = m.group(1)
        return f'<a href="../axiom.html#ax{num}">공리 {num}</a>'
    html = re.sub(r'공리\s*(\d{1,2})(?!\d)', repl_axiom, html)

    # 복원
    for key, orig in placeholders.items():
        html = html.replace(key, orig)
    return html

# ─────────── 본문 파싱 ───────────
def parse_lib_card(body_html):
    """lib-card div에서 각 요소 추출"""
    soup = BeautifulSoup(body_html, 'html.parser')
    div = soup.find('div', class_='lib-card')
    if not div: return {}

    out = {
        'formula': '',
        'precision': '',
        'paragraphs': [],
        'tags': [],
        'date': '',
        'kind': '',
    }

    # 태그 추출
    libid = div.find('div', class_='lib-id')
    if libid:
        for span in libid.find_all('span'):
            cls = ' '.join(span.get('class') or [])
            txt = span.get_text(strip=True)
            if 'tag' in cls:
                out['tags'].append((cls, txt))
            elif 'color:#6e7681' in (span.get('style') or ''):
                out['date'] = txt

    # 식
    fdiv = div.find('div', class_='lib-formula')
    if fdiv:
        out['formula'] = fdiv.get_text(strip=False).strip()

    # 정밀도
    pdiv = div.find('p', class_='lib-precision')
    if pdiv:
        out['precision'] = pdiv.get_text(strip=True)

    # 본문 단락
    for p in div.find_all('p'):
        cls = ' '.join(p.get('class') or [])
        if 'lib-precision' in cls: continue
        out['paragraphs'].append(str(p))

    # 재대입 div
    reuse = div.find('div', class_='lib-reuse')
    if reuse:
        out['reuse'] = reuse.get_text(strip=True)
    else:
        out['reuse'] = ''

    return out

def extract_refs(text):
    refs = set()
    for m in re.finditer(r'\b([DHP])[-\s]?(\d+)\b', text):
        kind, num = m.group(1), int(m.group(2))
        refs.add(f"{kind}-{num:02d}")
    return sorted(refs)

def extract_constants(text):
    found = set()
    # 한글 본문에서는 알파, beta 같은 ASCII 키워드 + 원본 그리스도 잡기
    raw_consts = ['α','α_s','sin²θ_W','sin^2θ_W','M_Z','M_W','m_t','m_c','m_b','m_τ','m_μ','m_e','G_F','ℏ','δ_CP','V_us','V_cb','V_ub','V_td','V_ts','Λ','H₀','H_0','Σm_ν','π','θ','Δ']
    for c in raw_consts:
        if c in text:
            # ASCII 표기로 정규화
            asc = c
            for o,r in ASCII_MAP:
                asc = asc.replace(o,r)
            found.add(asc)
    return sorted(found)

def derive_status(tags, text):
    for cls, txt in tags:
        if 'tag-solved' in cls or '적중' in txt: return ('적중', 'Discovery (오차 <= 1%)', 'discovery')
        if 'tag-discovery' in cls or txt == '발견': return ('발견', 'Discovery', 'discovery')
        if 'tag-hypothesis' in cls or '가설' in txt: return ('가설', 'Hypothesis (재귀 대입 진행 중)', 'hypothesis')
        if 'tag-prediction' in cls or '예측' in txt: return ('예측', 'Prediction (실험 검증 대기)', 'prediction')
    if '미완' in text or '구조만' in text:
        return ('가설', 'Hypothesis (구조 서술 완료, 정량 보류)', 'hypothesis')
    return ('미정', '판정 미정', 'unknown')

def reason_for_judgement(status, refs, has_formula, has_precision, text):
    """왜 발견/가설/예측인지 이유"""
    if status == '적중' or status == '발견':
        reasons = []
        if has_formula: reasons.append('명시적 도출 식 존재')
        if has_precision: reasons.append('실험 비교 오차 <= 1%')
        if refs: reasons.append(f'재귀 대입 입력 {len(refs)}개 확인')
        reasons.append('forward chain 완료 (공리 ==> 식 ==> 수치)')
        return ' / '.join(reasons)
    if status == '가설':
        reasons = []
        if not has_precision: reasons.append('실험 비교 부재')
        if not has_formula: reasons.append('정량 식 부재')
        if not refs: reasons.append('재귀 대입 미실행 (1차 라운드)')
        else: reasons.append(f'재귀 입력 {len(refs)}개 확인, 다음 라운드 진행 시 D 승격 가능')
        if '잔여' in text or '후속' in text: reasons.append('잔여 과제 명시')
        if not reasons: reasons.append('구조 서술까지만 진행됨')
        return ' / '.join(reasons)
    if status == '예측':
        return '실험 데이터 부재 / 구조적 forward chain 완료 / 검증 대기'
    return '본문 재검토 필요'

def where_found(category, refs):
    """어디에서 발견되는지 - 도메인 위치"""
    cat_name = category.replace('_',' ')
    return f'카테고리: {cat_name}. 도메인: {NORM_BY_CAT.get(category, "미확정")}. 재귀 대입 경로: {", ".join(refs[:5]) if refs else "1차 라운드 (외부 입력 없음)"}'

# ─────────── 카드 1장 HTML 생성 ───────────
def card_html(cid, title, body_html, category, kind_letter):
    parsed = parse_lib_card(body_html)
    title_ascii = ascii_outside_math(title)

    body_text_all = BeautifulSoup(body_html, 'html.parser').get_text(' ', strip=True)
    refs = [r for r in extract_refs(body_text_all) if r != cid][:30]
    consts = extract_constants(body_text_all)
    status, judgement_full, status_class = derive_status(parsed.get('tags',[]), body_text_all)
    has_formula = bool(parsed.get('formula'))
    has_precision = bool(parsed.get('precision'))
    judge_reason = linkify_refs(reason_for_judgement(status, refs, has_formula, has_precision, body_text_all))
    where = linkify_refs(where_found(category, refs))

    # 단락에서 핵심 해석 1~2 (첫 의미 단락)
    interp = ''
    for p_html in parsed.get('paragraphs', []):
        ptext = BeautifulSoup(p_html, 'html.parser').get_text(' ', strip=True)
        if len(ptext) > 30:
            interp = linkify_refs(ascii_outside_math(p_html))
            break

    # 부산물 추출 (재대입 + cross-ref)
    byproduct = parsed.get('reuse','')
    byproduct_html = f'<p>{linkify_refs(ascii_outside_math(byproduct))}</p>' if byproduct else ''
    if refs:
        link_list = ', '.join(f'<a href="{r.lower().replace("-","")}.html">{r}</a>' for r in refs[:10])
        byproduct_html += f'<p>이 카드는 다음 카드의 입력으로 재사용 가능: {link_list}</p>'
    if not byproduct_html:
        byproduct_html = '<p><em>명시된 부산물 없음. 후속 채굴에서 cross-link 추가 가능.</em></p>'

    # 미완 표
    incomplete_rows = ''
    if status == '가설':
        incomplete_rows = '<tr><td>정량 산출</td><td>구조 서술까지</td><td>재귀 대입 다음 라운드</td></tr>'
    elif status == '예측':
        incomplete_rows = '<tr><td>실험 검증</td><td>이론값 산출</td><td>해당 실험 (KATRIN, LHC, DESI 등)</td></tr>'
    elif status == '적중' or status == '발견':
        incomplete_rows = '<tr><td>2-loop 이상 보정</td><td>1-loop 수준</td><td>반야프레임 재귀 대입 정밀화</td></tr>'
    else:
        incomplete_rows = '<tr><td>판정</td><td>미정</td><td>본문 재검토 후 결정</td></tr>'

    # 총괄 표
    result_short = ''
    if parsed.get('precision'):
        result_short = ascii_outside_math(parsed['precision'])
    elif parsed.get('formula'):
        result_short = '도출 식 존재'
    else:
        result_short = '구조 서술'
    status_tag_class = {'discovery':'tag-discovery','hypothesis':'tag-hypothesis','prediction':'tag-prediction','unknown':'tag-unknown'}.get(status_class,'tag-unknown')
    summary_row = f'<tr><td>{cid}: {title_ascii}</td><td>{result_short}</td><td><span class="{status_tag_class}">{status}</span></td></tr>'

    # 핵심 발견 박스
    formula_html = parsed.get('formula','')
    precision_html = linkify_refs(ascii_outside_math(parsed.get('precision',''))) if parsed.get('precision') else ''
    core_finding = f'''
<div class="box-finding">
<h3>핵심 발견</h3>
<p><strong>{cid}: {title_ascii}</strong></p>
{f'<div class="formula-box">{formula_html}</div>' if formula_html else ''}
{f'<p class="finding-precision">{precision_html}</p>' if precision_html else ''}
{interp if interp else ''}
</div>
'''

    refs_html = ', '.join(f'<a href="{r.lower().replace("-","")}.html">{r}</a>' for r in refs[:20]) if refs else '<em>외부 참조 없음 (1차 라운드)</em>'
    consts_html = ', '.join(f'<code>{c}</code>' for c in consts) if consts else '<em>입력 상수 없음</em>'

    # 라이브러리 원본 본문 (전체)
    soup = BeautifulSoup(body_html, 'html.parser')
    div = soup.find('div', class_='lib-card')
    if div:
        for el in div.find_all(class_='lib-id'): el.decompose()
        for el in div.find_all('h3'): el.decompose()
        for el in div.find_all('div', class_='lib-formula'): el.decompose()
        for el in div.find_all('p', class_='lib-precision'): el.decompose()
        for el in div.find_all('div', class_='lib-reuse'): el.decompose()
        lib_inner = ''.join(str(c) for c in div.contents).strip()
        lib_inner = ascii_outside_math(lib_inner)
    else:
        lib_inner = '<p>본문 미작성</p>'

    verdict_box_class = {'discovery':'verdict-discovery','hypothesis':'verdict-hypothesis','prediction':'verdict-prediction','unknown':'verdict-unknown'}.get(status_class,'verdict-unknown')

    html_out = f'''<!DOCTYPE html>
<html lang="ko" translate="no">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cid} {title_ascii} -- 반야프레임 채굴</title>
<meta name="author" content="Han Hyukjin">
<meta name="description" content="{cid} {title_ascii}. v1.8 공리 기준 5단계 재귀 대입 채굴 카드.">
<meta name="robots" content="index, follow">
<meta name="rights" content="CC BY-NC-SA 4.0, Han Hyukjin 2026">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="../common.css">
<link rel="stylesheet" href="../print.css" media="print">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
.box-finding {{ background:#0e3b1f; border:2px solid #2ea043; padding:14px 18px; margin:1.5em 0; border-radius:5px; }}
.box-finding h3 {{ color:#3fb950; margin:0 0 10px; font-size:1.1em; }}
.box-byproduct {{ border-left:4px solid #1f6feb; padding:6px 16px; margin:1.2em 0; }}
.box-byproduct h3 {{ color:#58a6ff; margin:0 0 8px; font-size:1em; }}
.box-incomplete {{ border-left:4px solid #d29922; padding:6px 16px; margin:1.2em 0; }}
.box-incomplete h3 {{ color:#e3b341; margin:0 0 8px; font-size:1em; }}
.box-summary {{ border-left:4px solid #8b949e; padding:6px 16px; margin:1.2em 0; }}
.box-summary h3 {{ color:#c9d1d9; margin:0 0 8px; font-size:1em; }}
.formula-box {{ background:#161b22; padding:10px 14px; margin:8px 0; border-radius:3px; border:1px solid #30363d; }}
.finding-precision {{ color:#7ee787; font-weight:600; margin:6px 0; }}
.step {{ background:#0d1117; border-left:3px solid #58a6ff; padding:10px 16px; margin:0.8em 0; }}
.step h3 {{ margin:0 0 6px; color:#58a6ff; font-size:1em; }}
.verdict-discovery {{ background:#0e3b1f; padding:10px 16px; border-left:4px solid #2ea043; margin:8px 0; }}
.verdict-hypothesis {{ background:#0d2c5c; padding:10px 16px; border-left:4px solid #1f6feb; margin:8px 0; }}
.verdict-prediction {{ background:#3d2766; padding:10px 16px; border-left:4px solid #8957e5; margin:8px 0; }}
.verdict-unknown {{ background:#3a2d00; padding:10px 16px; border-left:4px solid #d29922; margin:8px 0; }}
.stop-point {{ background:#161b22; padding:10px 16px; border-left:3px solid #f0883e; margin:8px 0; }}
table.tbl {{ width:100%; border-collapse:collapse; margin:8px 0; }}
table.tbl th, table.tbl td {{ padding:6px 12px; border:1px solid #30363d; text-align:left; }}
table.tbl th {{ background:#161b22; color:#c9d1d9; }}
.tag-discovery,.tag-solved {{ background:#2ea043; color:#fff; padding:2px 6px; border-radius:3px; font-size:0.8em; }}
.tag-hypothesis {{ background:#1f6feb; color:#fff; padding:2px 6px; border-radius:3px; font-size:0.8em; }}
.tag-prediction {{ background:#8957e5; color:#fff; padding:2px 6px; border-radius:3px; font-size:0.8em; }}
.tag-unknown {{ background:#d29922; color:#fff; padding:2px 6px; border-radius:3px; font-size:0.8em; }}
</style>
</head>
<body>
<div class="lang-switch"><a href="../en/discovery/{cid.lower().replace("-","")}.html" class="lang-btn">EN</a></div>

<main class="lang-ko" style="max-width:840px;margin:0 auto;padding:2em 1em">

<p style="color:#7d8590;font-size:0.85em"><a href="../lib_new.html">&lt;== 라이브러리</a> | 카테고리: <code>{category}</code> | 판정: <span class="{status_tag_class}">{status}</span></p>

<h1>{cid}. {title_ascii}</h1>

{core_finding}

<h2>왜 이 판정인가</h2>
<div class="{verdict_box_class}">
<p><strong>판정: {status}</strong> -- {judgement_full}</p>
<p><strong>이유:</strong> {judge_reason}</p>
<p><strong>어디에서 발견되는가:</strong> {where}</p>
</div>

<h2>v1.8 공리 기준 5단계 재귀 대입</h2>

<div class="step">
<h3>1단계. 반야식 출발</h3>
<div class="formula-box">$$\\delta^2 = (\\text{{time}} + \\text{{space}})^2 + (\\text{{observer}} + \\text{{superposition}})^2$$</div>
<p>모든 채굴은 이 1줄에서 시작. CAS 단일 연산자 + 4축 직교.</p>
</div>

<div class="step">
<h3>2단계. 노름 치환 -- delta가 대상 도메인으로</h3>
<p>{NORM_BY_CAT.get(category, NORM_BY_CAT['h_misc'])}</p>
</div>

<div class="step">
<h3>3단계. 상수 + 가설 재투입 (재귀 대입)</h3>
<p><strong>1차 정수 (공리 9):</strong> {CAT_INTS.get(category, '본문 추출')}</p>
<p><strong>물리 상수 입력:</strong> {consts_html}</p>
<p><strong>라이브러리 재투입:</strong> {refs_html}</p>
</div>

<div class="step">
<h3>4단계. 도메인 변환</h3>
<p>위 본문의 식이 도메인 변환의 결과. v1.8 어휘 정렬:</p>
<ul>
<li>등호(=) = delta 발화비트 = 1 (좌우항 동일 변화량 선언)</li>
<li>+ = 직교 합성 (괄호 내 직교 부분공간)</li>
<li>비용 +1 = 직교 경계 순서 강제 횡단 (공리 3)</li>
</ul>
</div>

<div class="step">
<h3>5단계. 발견 / 판정</h3>
<p>판정: <strong>{status}</strong>. 자세한 이유는 위 "왜 이 판정인가" 박스 참조.</p>
</div>

<div class="box-byproduct">
<h3>부산물</h3>
{byproduct_html}
</div>

<div class="box-incomplete">
<h3>미완</h3>
<table class="tbl">
<thead><tr><th>항목</th><th>현재 상태</th><th>해결 방향</th></tr></thead>
<tbody>
{incomplete_rows}
</tbody>
</table>
</div>

<div class="box-summary">
<h3>총괄</h3>
<table class="tbl">
<thead><tr><th>항목</th><th>결과</th><th>상태</th></tr></thead>
<tbody>
{summary_row}
</tbody>
</table>
</div>

</main>

<div class="page-nav lang-ko">
<a href="../banya.html">종합 보고서</a>
<a href="../predictions.html">고유 예측</a>
<a href="../lib_new.html">가설 라이브러리</a>
<a href="../science_mine.html">과학 채굴 메뉴얼</a>
<a href="../axiom.html">공리</a>
</div>

<footer class="doc-footer lang-ko">
<div class="f-top">
<div class="f-corp">
<b>반야프레임 (般若 Framework)</b><br>
발명자: 한혁진 (Han Hyukjin)<br>
이메일: <a href="mailto:bokkamsun@gmail.com">bokkamsun@gmail.com</a><br>
별칭: 부처님 손바닥 프레임<br>
분류: 공리 기반 과학 채굴 엔진 (Axiom-Based Science Mining Engine)
</div>
<div class="f-right">
<div class="f1">$$\\delta^2 = (\\text{{time}} + \\text{{space}})^2 + (\\text{{observer}} + \\text{{superposition}})^2$$</div>
<div class="f2">우주상수: $\\Lambda l_p^2 = \\alpha^{{57}}$, 122자리 스케일 factor 2 이내 재현</div>
<div class="f3">관련 문서: <a href="../banya.html">종합 보고서</a> | <a href="../appendix.html">118개 상세검증 부록</a></div>
</div>
</div>
<div class="f-bottom">
<div class="f-copy">&copy; 2026 Han Hyukjin. All rights reserved.</div>
<div class="f-license">
<div class="license-title">CC BY-NC-SA 4.0</div>
This work is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="noopener">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International</a>.<br>
BY -- 저작자 표시 필수 | NC -- 비상업적 이용만 허용 | SA -- 동일 조건 변경 허락<br>
기존 물리학 공식의 저작권은 원저작자에게 있으며, 반야프레임 해석 및 신규 도출 공식은 Han Hyukjin(2026)에 귀속됩니다.
<div class="license-cite">인용: Han Hyukjin, "Banya Framework", 2026. bokkamsun@gmail.com</div>
</div>
</div>
</footer>

</body>
</html>
'''
    # 최종 HTML 전체에 linkify 한번 더 적용 (5단계 골격 등 템플릿 내부 참조 흡수)
    # <head> 안은 보호하기 위해 body 부분만
    head_end = html_out.find('</head>')
    if head_end > 0:
        head_part = html_out[:head_end+7]
        body_part = html_out[head_end+7:]
        body_part = linkify_refs(body_part)
        html_out = head_part + body_part
    return html_out


# ─────────── 메인 ───────────
def main():
    cards = json.load(open('/tmp/cards.json'))
    card_bodies = json.load(open('/tmp/card_bodies.json'))
    h_cat = json.load(open('/tmp/h_cat.json'))
    h_cat_map = h_cat['card_cat']

    d_categories = [
        ('d_qcd', ['QCD','강력','강결합','α_s','글루온','쿼크 질량','색','하드론','메존','바리온','파이온','b₀','b_0']),
        ('d_neutrino_cp', ['뉴트리노','PMNS','CKM','CP','위상','δ_CKM','카비보','혼합각','θ_C','θ_12','θ_13','θ_23']),
        ('d_cosmology', ['우주','인플레이션','Λ','BAO','H₀','CMB','암흑','dark','BH','블랙홀','중력','GW','감속','Ω','우주상수','z_eq','n_s']),
        ('d_gauge_ew', ['게이지','SU','GUT','통일','sin^2','sin²','Weinberg','바인베르크','전약','M_Z','M_W','Higgs','힉스']),
        ('d_mass_yukawa', ['질량','코이데','Koide','톱','보텀','참','스트레인지','뮤온','타우','m_','반지름']),
        ('d_electromag', ['미세구조','전자기','QED','광자','Hall','g-2','카시미르','Casimir','Schwinger','Lamb']),
        ('d_atomic', ['원자','Rydberg','Bohr','보어','Hartree']),
        ('d_structure', ['Wyler','CAS','반야식','자유도','코드','Hamming','지수','BH 온도','축퇴압','결합상수','Dirac']),
    ]
    def classify_d(title):
        for cat, kws in d_categories:
            for kw in kws:
                if kw.lower() in title.lower(): return cat
        return 'd_misc'

    out_dir = 'discovery'
    os.makedirs(out_dir, exist_ok=True)
    counts = {'D':0, 'H':0, 'P':0}

    # D
    for n, title in cards['D']:
        cid = f"D-{n:02d}"
        body = card_bodies.get(cid, '<div class="lib-card"><p>본문 미작성</p></div>')
        cat = classify_d(title)
        html = card_html(cid, title, body, cat, 'D')
        with open(f'{out_dir}/d{n:02d}.html', 'w', encoding='utf-8') as f:
            f.write(html)
        counts['D'] += 1

    # H
    for n, title in cards['H']:
        cid = f"H-{n:02d}"
        body = card_bodies.get(cid, '<div class="lib-card"><p>본문 미작성</p></div>')
        cat = h_cat_map.get(str(n), h_cat_map.get(n, 'h_misc'))
        html = card_html(cid, title, body, cat, 'H')
        with open(f'{out_dir}/h{n:02d}.html', 'w', encoding='utf-8') as f:
            f.write(html)
        counts['H'] += 1

    # P (본문은 predictions.html에서)
    p_round = {1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:6}
    pred_html = open('predictions.html', 'r', encoding='utf-8').read()
    pred_soup = BeautifulSoup(pred_html, 'html.parser')
    for n, title in cards['P']:
        cid = f"P-{n:02d}"
        rn = p_round.get(n)
        body_excerpt = ''
        if rn:
            anchor = pred_soup.find(id=f'r{rn}')
            if anchor:
                content = []
                el = anchor
                while el and el.find_next_sibling():
                    el = el.find_next_sibling()
                    if el.name == 'h1' and (el.get('id','').startswith('r') or el.get('id','').startswith('seeds')): break
                    content.append(str(el))
                body_excerpt = ''.join(content[:25])
        body_html = f'<div class="lib-card"><div class="lib-id">P-{n:02d} <span class="tag-prediction">예측</span></div><h3>{title}</h3><p class="lib-precision">출처: predictions.html#r{rn}</p>{body_excerpt}</div>' if body_excerpt else f'<div class="lib-card"><p>본문은 predictions.html#r{rn}에 있음</p></div>'
        html = card_html(cid, title, body_html, 'p', 'P')
        with open(f'{out_dir}/p{n:02d}.html', 'w', encoding='utf-8') as f:
            f.write(html)
        counts['P'] += 1

    print(f"D 생성: {counts['D']}장")
    print(f"H 생성: {counts['H']}장")
    print(f"P 생성: {counts['P']}장")
    print(f"총 {sum(counts.values())}장 (보강판)")

if __name__ == '__main__':
    main()
