"""
1114장 핵심 발견 박스 재채굴 (d45.html 형식)
- [무엇]/[반야식]/[공리 근거]/[도출] 등 라벨 모두 제거
- 자연 문장 흐름 10줄+
- 가설은 멈춘 이유 상세 기록
- 매 카드 CLI 출력
- v1.8 공리 + physics_map.html 기반
"""
import json, re, glob, os, sys
from bs4 import BeautifulSoup

# ─────────── 데이터 로드 ───────────
card_bodies = json.load(open('/tmp/card_bodies.json'))
cards = json.load(open('/tmp/cards.json'))
predictions_html = open('predictions.html').read()
pred_soup = BeautifulSoup(predictions_html, 'html.parser')

# ─────────── 라벨 제거 + 자연 문장화 ───────────
LABEL_RE = re.compile(
    r'\[\s*(?:'
    r'무엇|반야식|반야식\s*출발|노름\s*치환|공리\s*체인|공리\s*근거|'
    r'구조적\s*귀결|구조\s*귀결|도출|도출\s*경로|수치|수치/예측|예측|'
    r'오차|오차/정합|정합|물리\s*대응|물리적\s*대응|검증|검증/반박|'
    r'잔여|잔여\s*과제|재대입|재대입\s*용도|차이|기존\s*이론과\s*차이|차이점'
    r')\s*\]\s*'
)

# 이미 정독 보강된 9장은 스킵 (덮어쓰면 보강 손실)
SKIP_CIDS = {'D-104', 'H-529', 'H-517', 'H-518', 'H-566', 'H-723', 'H-724', 'H-906', 'H-136', 'D-01', 'D-02', 'D-03'}

def axiom_link(text):
    """공리 N → axiom.html#axN 링크. lambda로 \ escape 회피."""
    def repl(m):
        n = m.group(1)
        return f'<a href="../axiom.html#ax{n}">공리 {n}</a>'
    return re.sub(r'공리\s*(\d{1,2})(?!\d)', lambda m: repl(m), text)

def card_link(text):
    """D-NN/H-NN/P-NN → dNN.html/hNN.html/pNN.html. lambda로 \ escape 회피."""
    def repl(m):
        kind, num = m.group(1), int(m.group(2))
        cid = f"{kind}-{num:02d}"
        href = f"{kind.lower()}{num:02d}.html"
        return f'<a href="{href}">{cid}</a>'
    return re.sub(r'\b([DHP])-(\d{1,3})\b', lambda m: repl(m), text)

def linkify(html):
    """이미 <a> 안과 KaTeX 박스 안은 보존"""
    placeholders = {}
    pid = [0]
    def stash(m):
        key = f"\x00LINK{pid[0]}\x00"
        placeholders[key] = m.group(0)
        pid[0] += 1
        return key
    html = re.sub(r'<a\b[^>]*>.*?</a>', stash, html, flags=re.S)
    html = re.sub(r'\$\$.*?\$\$', stash, html, flags=re.S)
    html = re.sub(r'\$[^$]+\$', stash, html, flags=re.S)
    html = card_link(html)
    html = axiom_link(html)
    for k, v in placeholders.items():
        html = html.replace(k, v)
    return html

def extract_natural_paragraphs(body_html):
    """라이브러리 본문의 모든 <p> 단락을 [라벨] 제거 후 흐름 정리"""
    soup = BeautifulSoup(body_html, 'html.parser')
    div = soup.find('div', class_='lib-card')
    if not div: return []
    out = []
    for p in div.find_all('p'):
        cls = ' '.join(p.get('class') or [])
        if 'lib-precision' in cls or 'lib-detail' in cls: continue
        # <p> 텍스트에서 [라벨] 제거
        inner = ''.join(str(c) for c in p.contents)
        inner = LABEL_RE.sub('', inner).strip()
        if not inner: continue
        # 비용 누적 → √3 노름 수축 v1.8 정정 (중력 카드 등)
        out.append(inner)
    # lib-reuse도 풀어서 마지막에
    reuse = div.find('div', class_='lib-reuse')
    if reuse:
        reuse_text = ''.join(str(c) for c in reuse.contents).strip()
        reuse_text = LABEL_RE.sub('', reuse_text)
        out.append(f'<strong>재대입 용도:</strong> {reuse_text}')
    return out

def extract_card_meta(body_html):
    """카드 메타: 식, 정밀도, 등급, 태그"""
    soup = BeautifulSoup(body_html, 'html.parser')
    out = {'formula': '', 'precision': '', 'tags': [], 'h3': ''}
    f = soup.find('div', class_='lib-formula')
    if f: out['formula'] = f.get_text(strip=False).strip()
    p = soup.find('p', class_='lib-precision')
    if p: out['precision'] = p.get_text(strip=True)
    h3 = soup.find('h3')
    if h3: out['h3'] = h3.get_text(strip=True)
    libid = soup.find('div', class_='lib-id')
    if libid:
        for sp in libid.find_all('span'):
            cls = ' '.join(sp.get('class') or [])
            if 'tag' in cls:
                out['tags'].append((cls, sp.get_text(strip=True)))
    return out

def derive_status(tags, body_text):
    for cls, txt in tags:
        if 'tag-solved' in cls or '적중' in txt: return 'discovery'
        if 'tag-discovery' in cls or txt == '발견': return 'discovery'
        if 'tag-hypothesis' in cls or '가설' in txt: return 'hypothesis'
        if 'tag-prediction' in cls or '예측' in txt: return 'prediction'
    return 'unknown'

def stop_reason(cid, status, paragraphs, meta):
    """가설/예측 카드의 멈춘 이유 상세 생성"""
    if status not in ('hypothesis', 'prediction', 'unknown'):
        return None
    has_formula = bool(meta.get('formula'))
    has_precision = bool(meta.get('precision'))
    body_text = ' '.join(BeautifulSoup(p, 'html.parser').get_text(' ', strip=True) for p in paragraphs)
    # 잔여 과제 본문에서 추출
    leftover = ''
    for p in paragraphs:
        pt = BeautifulSoup(p, 'html.parser').get_text(' ', strip=True)
        if any(k in pt for k in ['잔여', '미완', '후속', '다음 라운드', '재대입']):
            leftover = pt[:300]
            break
    parts = []
    if status == 'hypothesis':
        parts.append('본 카드는 가설(분류 III)로 분류되어 다음 라운드로 미진행한 채 멈춰 있다.')
        if not has_formula:
            parts.append('정량 도출 식이 미작성된 상태이며, 구조 서술과 공리 근거 인용까지만 진행되었다. 식 자체가 도메인 변환에서 자연 도출되어야 하지만 현재는 명시적 식 형태가 본문에 등장하지 않는다.')
        if not has_precision:
            parts.append('실험값과의 정량 비교가 부재하다. 비교할 표준 측정값(PDG/CODATA/실험 한계)이 인용되지 않았거나 차원·구조 일치까지만 확인되어 정밀도 산출이 미완.')
        if has_formula and not has_precision:
            parts.append('도출 식은 존재하나 실험 측정값과의 정량 대조가 미완. 차원 정합과 구조 일치까지는 확립.')
        if has_formula and has_precision:
            parts.append('정량 식과 실험값이 둘 다 있지만 forward chain의 마지막 단계가 닫히지 않았다. 식의 모든 계수가 공리 또는 1차 정수에서 직접 도출되지 않고 일부가 외부 입력 또는 가설로 남았다.')
        if leftover:
            parts.append(f'잔여 과제 명시: {leftover}')
        parts.append(f'D 승격 경로는 (1) 위 잔여 과제 해소, (2) 라이브러리의 인접 D/H 카드를 재귀 대입 라운드 입력으로 재투입, (3) 공리 1·2·4·9·11 명제 중 미적용 명제 추가 도입의 3가지가 표준.')
    elif status == 'prediction':
        parts.append('본 카드는 고유 예측(분류 III/IV)이며 실험 검증 대기 상태로 멈춰 있다.')
        parts.append('forward chain은 공리에서 출발해 정량 예측값까지 닫혀 있으나, 해당 실험이 아직 수행되지 않았거나 정밀도가 본 카드의 예측을 검증할 수준에 도달하지 않았다.')
        parts.append('예측값이 실험 측정과 정합하면 D 승격, 측정과 5σ 이상 어긋나면 반증 → 본 카드 폐기 + 상위 가설 사슬 영향 평가.')
        if leftover: parts.append(f'잔여 과제 또는 검증 일정: {leftover}')
    elif status == 'unknown':
        parts.append('본 카드의 판정이 미정 상태로 멈춰 있다.')
        parts.append('태그가 부착되지 않았거나 (발견/가설/예측 어느 등급에도 명시 분류 안 됨), 본문이 구조 선언만 있고 도출 사슬이 미작성. 우선 카테고리 명확화 후 재채굴 필요.')
    return ' '.join(parts)

# ─────────── 핵심 발견 박스 본문 생성 ───────────
def build_finding_body(cid, paragraphs, meta, status, category):
    """d45 스타일 핵심 발견 박스 내부 본문"""
    # 식
    formula_html = ''
    if meta.get('formula'):
        formula_html = f'<div class="formula-box">{meta["formula"]}</div>'

    # 정밀도 / 등급
    precision_html = ''
    if meta.get('precision'):
        precision_html = f'<p class="finding-precision">{meta["precision"]}</p>'

    # 본문 단락들 — 자연 흐름으로 이어붙임 + linkify
    body_paras = []
    for p in paragraphs:
        body_paras.append(f'<p>{linkify(p)}</p>')

    # 가설/예측이면 멈춘 이유 상세 단락 추가
    stop_html = ''
    sr = stop_reason(cid, status, paragraphs, meta)
    if sr:
        stop_html = f'<p style="margin-top:1em;padding:8px 12px;background:#0d1117;border-left:3px solid #f0883e"><strong>가설에서 멈춘 이유:</strong> {linkify(sr)}</p>'

    title = meta.get('h3', cid)
    title_link = f'<a href="{cid.lower().replace("-","")}.html">{cid}</a>'
    header = f'<p><strong>{title_link}: {title}</strong></p>'

    return header + '\n' + formula_html + '\n' + precision_html + '\n' + '\n'.join(body_paras) + '\n' + stop_html


# ─────────── P 카드 본문 추출 ───────────
def build_p_paragraphs(n):
    p_round = {1:'r1', 2:'r2', 3:'r3', 4:'r4', 5:'r5-bao', 6:'r5-photon', 7:'r6'}
    rid = p_round.get(n)
    if not rid: return []
    anchor = pred_soup.find(id=rid)
    if not anchor: return []
    paragraphs = []
    el = anchor
    while True:
        el = el.find_next_sibling()
        if not el: break
        if el.name == 'h1': break
        if el.name == 'p':
            inner = ''.join(str(c) for c in el.contents).strip()
            inner = LABEL_RE.sub('', inner).strip()
            if inner: paragraphs.append(inner)
    return paragraphs[:12]

# ─────────── 카드 1장 갱신 ───────────
def update_card(fpath, cid, kind, n):
    if not os.path.exists(fpath): return False
    with open(fpath, 'r', encoding='utf-8') as f:
        html = f.read()

    # 본문 추출
    if kind == 'P':
        paragraphs = build_p_paragraphs(n)
        meta = {'formula': '', 'precision': f'P-{n:02d} 예측 (predictions.html#{["r1","r2","r3","r4","r5-bao","r5-photon","r6"][n-1]})', 'tags': [('tag-prediction','예측')], 'h3': dict(cards['P']).get(n, '')}
        status = 'prediction'
    else:
        body = card_bodies.get(cid)
        if not body: return False
        paragraphs = extract_natural_paragraphs(body)
        meta = extract_card_meta(body)
        body_text = ' '.join(BeautifulSoup(p,'html.parser').get_text(' ',strip=True) for p in paragraphs)
        status = derive_status(meta['tags'], body_text)

    # 카테고리 추출 (기존 HTML에서)
    cat_match = re.search(r'카테고리:\s*<code>([^<]+)</code>', html)
    category = cat_match.group(1) if cat_match else ''

    # 핵심 발견 박스 본문 생성
    finding_inner = build_finding_body(cid, paragraphs, meta, status, category)

    # 기존 box-finding 통째 교체 — bs4로 nested div 안전 처리
    new_box = f'<div class="box-finding">\n<h3>핵심 발견</h3>\n{finding_inner}\n</div>'
    soup_full = BeautifulSoup(html, 'html.parser')
    boxes = soup_full.find_all('div', class_='box-finding')
    if boxes:
        for b in boxes:
            b.decompose()
        # h1과 "왜 이 판정인가" h2 사이의 박스 외부 잔존 단편 제거
        # (이전 자동생성 박스의 finding-precision, [무엇] 텍스트 등이 박스 밖에 떠 있는 경우)
        h1 = soup_full.find('h1')
        verdict_h2 = None
        for h2 in soup_full.find_all('h2'):
            if '왜 이 판정' in h2.get_text():
                verdict_h2 = h2
                break
        if h1 and verdict_h2:
            # h1 다음 ~ verdict_h2 이전의 모든 노드 제거
            cur = h1.find_next_sibling()
            while cur and cur is not verdict_h2:
                nxt = cur.find_next_sibling()
                cur.extract()
                cur = nxt
            new_node = BeautifulSoup(new_box, 'html.parser')
            h1.insert_after(new_node)
        else:
            new_node = BeautifulSoup(new_box, 'html.parser')
            if h1:
                h1.insert_after(new_node)
        html = str(soup_full)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html)
    return True

# ─────────── 실행 ───────────
files = sorted(glob.glob('discovery/[dhp][0-9]*.html'))
total = len(files)
done = 0
fail = 0
for fpath in files:
    m = re.match(r'discovery/([dhp])(\d+)\.html', fpath)
    if not m: continue
    kind = m.group(1).upper()
    n = int(m.group(2))
    cid = f"{kind}-{n:02d}"
    if cid in SKIP_CIDS:
        print(f"[SKIP] {cid} (이미 정독 보강됨)", flush=True)
        continue
    try:
        ok = update_card(fpath, cid, kind, n)
        if ok:
            done += 1
            print(f"[{done:4d}/{total}] {cid} 완료", flush=True)
        else:
            fail += 1
            print(f"[FAIL] {cid} 본문 부재", flush=True)
    except Exception as e:
        fail += 1
        print(f"[ERROR] {cid}: {e}", flush=True)

print(f"\n총: {done}장 완료, {fail}장 실패 / 전체 {total}장")
