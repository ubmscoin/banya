# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_logic_corpus.py — Elementary logic (deepened statement/converse/inverse/contrapositive + transitivity + classification)

Design principle 41.14. The converse rotation, only partially generalized at the kindergarten stage, is now fully
generalized (several siblings are presented). Transitivity (A>B, B>C -> A>C). Deepened classification (one
superordinate with several subordinates). Formal logic is still done in words, but more rigorously than at the
kindergarten stage.

Run  python3 data_prep/prep_elem_logic_corpus.py   ->  data/elem_logic.npy (set the folder with BANYA_DATA_DIR)

prep_elem_logic_corpus.py — 초딩 논리 (정역이대우 심화 + 이행 + 분류)

설계원리 41.14. 유딩서 부분 일반화였던 역 회전을 완전 일반화로(형제 여럿 제시). 이행(A>B, B>C -> A>C).
분류 심화(상위 하나에 하위 여럿). 형식논리를 여전히 말로 하되 유딩보다 엄격하게 다룬다.

실행: python3 data_prep/prep_elem_logic_corpus.py   ->  data/elem_logic.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 149
N_PARA = 90000

josa = bu.josa

# (subordinate, superordinate, [several siblings]). Many siblings are needed to fully generalize that the converse and the inverse are false
# (하위, 상위, [형제 여럿]). 형제가 많아야 역/이가 틀림을 완전 일반화한다
회전쌍 = [
    ("곰", "동물", ["개미", "참새", "토끼", "사자", "고양이", "기린"]),
    ("사과", "과일", ["포도", "딸기", "바나나", "귤", "수박"]),
    ("참새", "새", ["닭", "비둘기", "오리", "제비"]),
    ("장미", "꽃", ["국화", "튤립", "해바라기", "민들레"]),
    ("기차", "탈것", ["버스", "배", "비행기", "자동차"]),
    ("당근", "채소", ["오이", "배추", "무", "감자"]),
]

# Transitivity. (large, middle, small). If A > B > C then A > C
# 이행. (큰것, 중간것, 작은것). A > B > C 이면 A > C
비교쌍 = [
    ("코끼리", "개", "개미"),
    ("산", "언덕", "돌"),
    ("어른", "아이", "baby"),
    ("고래", "물고기", "새우"),
    ("나무", "꽃", "씨앗"),
]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def para(rng):
    g = rng.randint(0, 10)

    if g >= 7:                                   # transitivity (A>B, B>C -> A>C) / 이행 (A>B, B>C -> A>C)
        a, b, c = 비교쌍[rng.randint(0, len(비교쌍))]
        return (f"{josa(a, '은', '는')} {b}보다 크고 {josa(b, '은', '는')} {c}보다 커. "
                f"그럼 {josa(a, '이', '가')} {c}보다 커? 응, {josa(a, '이', '가')} 제일 커.")

    하, 상, 형들 = 회전쌍[rng.randint(0, len(회전쌍))]
    정 = f"{josa(하, '은', '는')} {josa(상, '이야', '야')}."
    상이면 = josa(상, '이면', '면')
    상이야 = josa(상, '이야', '야')

    if g == 0 or g == 1:                         # contrapositive (flipping it keeps it true) / 대우 (뒤집으면 맞다)
        return f"{정} 뒤집으면? {상} 아니면 {하} 아니야. 맞아."

    if g == 2 or g == 3:                         # converse (the reverse is false; fully generalized with two siblings) / 역 (거꾸로는 틀리다. 형제 둘로 완전 일반화)
        형 = _pick(rng, 형들)
        형2 = _pick(rng, [x for x in 형들 if x != 형])
        return (f"{정} 거꾸로 {상이면} {josa(하, '이야', '야')}? "
                f"아니, {형}도 {형2}도 {상이야}. {상}엔 여럿 있어. 틀렸어.")

    if g == 4:                                   # inverse (the negation is also false) / 이 (반대도 틀리다)
        형 = _pick(rng, 형들)
        return (f"{정} 반대로 {하} 아니면 {상} 아니야? "
                f"아니, {josa(형, '이', '가')} 있잖아. 틀렸어.")

    if g == 5:                                   # summary (only the contrapositive holds) / 정리 (대우만 맞다)
        return f"{정} 뒤집은 건 맞아. 근데 거꾸로랑 반대는 틀려. {상}엔 형제가 있으니까."

    # g == 6: deepened classification (one superordinate with several subordinates)
    # g == 6: 분류 심화 (상위 하나에 하위 여럿)
    골라 = list(np.random.RandomState(rng.randint(0, 1 << 30)).choice([하] + 형들, size=min(4, 1 + len(형들)), replace=False))
    나열 = " ".join(골라)
    return f"{상}엔 {나열} 다 있어. 다 {상이야}. 그래도 서로 달라."


def main():
    tok = ba.AtomTokenizer()
    print(f"  초딩 논리. 정역이대우 심화 + 이행 + 분류. 회전쌍 {len(회전쌍)}개 비교쌍 {len(비교쌍)}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem_logic", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:420].tolist()))


if __name__ == "__main__":
    main()
