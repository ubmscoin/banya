# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""assemble_toddler2.py — extracts toddler2 expressions from the generation-workflow journal and assembles them into expression modules

Radial results (refine kept) are grouped by seed into the expression dict of toddler2_expr.py,
and web expressions go as a flat list into the link list of toddler2_link_expr.py.
Coding symbols, emoticons, and English are filtered once more.
Run  python3 data_prep/assemble_toddler2.py

assemble_toddler2.py — 생성 워크플로 journal 에서 유딩2 표현을 뽑아 표현 모듈로 조립

방사(refine kept)는 시드별로 묶어 toddler2_expr.py 의 표현 dict 로, 그물(web expressions)은
평탄 리스트로 toddler2_link_expr.py 의 연결 리스트로 쓴다. 코딩기호 이모티콘 영어는 한 번 더 거른다.
실행: python3 data_prep/assemble_toddler2.py
"""
import os
import re
import json
import collections

JR = "/home/khan/.claude/projects/-home-khan------claude-work-banya-ai/5ecba458-6787-47db-86c2-f81e6773453a/subagents/workflows/wf_e0dfb1a9-5cc/journal.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))

# Allowed: Hangul syllables, jamo, spaces, question marks, periods, exclamation marks, commas. Anything else (English, Hanja, symbols, emoticons) causes the text to be discarded
# 허용: 한글 음절 자모 공백 물음표 마침표 느낌표 쉼표. 그 외(영어 한자 기호 이모티콘) 있으면 버린다
_OK = re.compile(r"^[가-힣ㄱ-ㅎㅏ-ㅣ0-9 \n\.\?\!,]+$")


def clean(t):
    t = t.replace("\n", " ").strip()
    return t if t and _OK.match(t) else None


def main():
    radial = collections.defaultdict(list)      # seed -> [text]
    web = []
    seen_r = set()
    seen_w = set()
    for line in open(JR, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") != "result":
            continue
        v = r.get("value") or r.get("result")
        if not isinstance(v, dict):
            continue
        if "kept" in v and isinstance(v["kept"], list):        # radial refinement results / 방사 정제 결과
            seed = str(v.get("seed", "")).strip()
            for e in v["kept"]:
                if not isinstance(e, dict):
                    continue
                c = clean(e.get("text", ""))
                if c and c not in seen_r:
                    seen_r.add(c)
                    radial[seed].append(c)
        elif "expressions" in v and isinstance(v["expressions"], list) and v["expressions"] and isinstance(v["expressions"][0], dict) and "links" in v["expressions"][0]:
            for e in v["expressions"]:            # web / 그물
                c = clean(e.get("text", ""))
                if c and c not in seen_w:
                    seen_w.add(c)
                    web.append(c)

    radial = {k: v for k, v in radial.items() if v}
    nr = sum(len(v) for v in radial.values())
    print(f"방사 {len(radial)}시드 {nr}표현, 그물 {len(web)}표현")

    with open(os.path.join(HERE, "toddler2_expr.py"), "w", encoding="utf-8") as f:
        f.write("# 유딩2 방사분화 표현. assemble_toddler2.py 산출. 시드별 묶음\n")
        f.write("표현 = {\n")
        for seed, lst in radial.items():
            key = seed.replace('"', "").replace("[", "").replace("]", "")
            f.write(f"    {json.dumps(key, ensure_ascii=False)}: [\n")
            for t in lst:
                f.write(f"        {json.dumps(t, ensure_ascii=False)},\n")
            f.write("    ],\n")
        f.write("}\n")

    with open(os.path.join(HERE, "toddler2_link_expr.py"), "w", encoding="utf-8") as f:
        f.write("# 유딩2 그물(개념 상호작용) 표현. assemble_toddler2.py 산출\n")
        f.write("연결 = [\n")
        for t in web:
            f.write(f"    {json.dumps(t, ensure_ascii=False)},\n")
        f.write("]\n")
    print("저장: data_prep/toddler2_expr.py + toddler2_link_expr.py")


if __name__ == "__main__":
    main()
