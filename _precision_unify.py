"""
1114장 핵심 발견 박스에 finding-precision 단락 통일
- h35.html 표준 형식: 두꺼운 녹색으로 핵심 정밀도 + 구조 의미
- 누락/빈 카드 자동 보강
- 본문에서 오차 %, 등급, 구조 분해 추출
"""
import json, re, glob, os
from bs4 import BeautifulSoup

card_bodies = json.load(open('/tmp/card_bodies.json'))
cards = json.load(open('/tmp/cards.json'))

def extract_precision_text(cid, body_html):
    """카드에서 정밀도 + 구조 핵심 문구 추출"""
    soup = BeautifulSoup(body_html, 'html.parser')
    parts = []

    # 1. lib-precision 우선
    p = soup.find('p', class_='lib-precision')
    if p:
        text = p.get_text(strip=True)
        if text: parts.append(text)

    # 2. 본문에서 "오차 X.X%" 패턴
    body_text = soup.get_text(' ', strip=True)
    err_m = re.search(r'(오차[\s:]*[<≤]?\s*[0-9]+(?:\.[0-9]+)?\s*%)', body_text)
    if err_m and err_m.group(1) not in ' '.join(parts):
        parts.append(err_m.group(1))

    # 3. 등급
    grade_m = re.search(r'([SABC]\s*급)', body_text)
    if grade_m and grade_m.group(1) not in ' '.join(parts):
        parts.append(f"등급 {grade_m.group(1)}")

    # 4. 실험값 비교 패턴
    cmp_m = re.search(r'(실험값?\s*[:=]?\s*[0-9.×^\-+\s]+\s*(?:[가-힣A-Za-z%/]+)?)', body_text)
    # 4. 그리고 도출값 패턴
    der_m = re.search(r'(도출값?\s*[:=]?\s*[0-9.×^\-+\s]+\s*(?:[가-힣A-Za-z%/]+)?)', body_text)

    # 5. "정수 정확", "자릿수 정확" 등 키워드
    for kw in ['정수 정확', '자릿수 정확', '정확 일치', '항등식', '구조적 대응', '정수 일치', '0% 오차']:
        if kw in body_text and kw not in ' '.join(parts):
            parts.append(kw)
            break

    return ' · '.join(parts) if parts else ''

def unify_card(fpath, cid):
    if not os.path.exists(fpath): return None
    body_html = card_bodies.get(cid)
    if not body_html and not cid.startswith('P-'): return None

    with open(fpath, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    box = soup.find('div', class_='box-finding')
    if not box: return None

    existing_p = box.find('p', class_='finding-precision')
    existing_text = existing_p.get_text(strip=True) if existing_p else ''

    if body_html:
        new_text = extract_precision_text(cid, body_html)
    else:
        new_text = ''

    # 기존이 더 길거나 비슷하면 유지 (이미 잘 보강된 카드 보호)
    if len(existing_text) >= max(20, len(new_text) - 10):
        return 'keep'

    # 새로 채울 텍스트가 비어있으면 스킵
    if not new_text: return None

    if existing_p:
        existing_p.clear()
        existing_p.append(new_text)
    else:
        # formula-box 다음에 삽입, 없으면 h3 다음
        new_p = soup.new_tag('p')
        new_p['class'] = 'finding-precision'
        new_p.string = new_text
        fbox = box.find('div', class_='formula-box')
        if fbox:
            fbox.insert_after(new_p)
        else:
            h3 = box.find('h3')
            if h3:
                h3.insert_after(new_p)
            else:
                box.insert(0, new_p)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    return 'updated'

# 실행
files = sorted(glob.glob('discovery/[dhp][0-9]*.html'))
updated = 0
kept = 0
none = 0
for fpath in files:
    m = re.match(r'discovery/([dhp])(\d+)\.html', fpath)
    if not m: continue
    cid = f"{m.group(1).upper()}-{int(m.group(2)):02d}"
    r = unify_card(fpath, cid)
    if r == 'updated': updated += 1
    elif r == 'keep': kept += 1
    else: none += 1

print(f"갱신: {updated}장")
print(f"기존 유지 (이미 충분): {kept}장")
print(f"추출 불가/식 부재: {none}장")
print(f"전체: {len(files)}장")
