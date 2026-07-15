# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_logic_corpus.py — toddler logic (comparison and relations between objects)

From the toddler stage, comparison between objects becomes possible. A baby compared only against me, but a toddler knows that this is bigger than that.
Bigger, same, different; the because-causation relation, ordering, and classifying like with like. Formal logic starts from elementary school, so here it is done in words.
The expressions come from the toddler_logic region of toddler_expr.py.

Run: python3 data_prep/prep_toddler_logic_corpus.py   ->  data/toddler_logic.npy (specify the folder with BANYA_DATA_DIR)

prep_toddler_logic_corpus.py — 유딩 논리 (대상끼리 비교와 관계)

유딩부터는 대상끼리 비교가 된다. 아기는 나랑만 비교했지만 유딩은 이게 저것보다 크다를 안다.
더 크다 같다 다르다, 왜냐하면 인과, 순서 세우기, 같은 것끼리 분류. 형식 논리는 초딩부터라 여기선 말로.
표현은 toddler_expr.py 의 유딩_논리 영역에서 온다.

실행: python3 data_prep/prep_toddler_logic_corpus.py   ->  data/toddler_logic.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import toddler_expr as Y

SEED = 79
N_PARA = 90000
표현 = Y.표현["toddler_logic"]
josa = bu.josa

# Renders the proposition-converse-inverse-contrapositive cycle in toddler words, using no formal terms. The converse becomes reversed, the inverse becomes opposite, the contrapositive becomes flip it and it holds, with right, wrong, and why.
# A toddler classifies objects. The subordinate belongs to the superordinate. Only the contrapositive holds; the converse and inverse fail, because the superordinate also has siblings.
# (subordinate, superordinate, sibling)
# 정역이대우 회전을 유딩말로. 형식어 안 쓴다. 역은 거꾸로 이는 반대 대우는 뒤집으면 맞아 틀렸어 왜.
# 유딩은 대상 분류를 한다. 하위는 상위에 속한다. 대우만 맞고 역과 이는 틀리다. 상위엔 형제도 있으니까.
# (하위, 상위, 형제)
유딩회전쌍 = [
    ("곰", "동물", "개미"),
    ("참새", "새", "닭"),
    ("사과", "과일", "포도"),
    ("장미", "꽃", "국화"),
    ("개", "동물", "물고기"),
]


def 회전(rng):
    하, 상, 형 = 유딩회전쌍[rng.randint(0, len(유딩회전쌍))]
    정 = f"{josa(하, '은', '는')} {josa(상, '이야', '야')}."
    g = rng.randint(0, 5)

    if g == 0:                                 # Original statement plus contrapositive (flip it and it holds) / 정 진술 + 대우 (뒤집으면 맞아)
        return f"{정} 뒤집으면? {상} 아니면 {하} 아니야. 맞아."

    if g == 1:                                 # Converse (reversed is wrong, because there are siblings) / 역 (거꾸로는 틀려. 형제가 있으니까)
        return f"{정} 거꾸로 {josa(상, '이면', '면')} {josa(하, '이야', '야')}? 아니, {형}도 {josa(상, '이야', '야')}. 틀렸어."

    if g == 2:                                 # Inverse (the opposite is also wrong) / 이 (반대도 틀려)
        return f"{정} 반대로 {하} 아니면 {상} 아니야? 아니, {josa(형, '이', '가')} 있잖아. 틀렸어."

    if g == 3:                                 # Summary that only the contrapositive holds / 대우만 맞다 정리
        return f"{정} 뒤집은 건 맞아. 근데 거꾸로랑 반대는 틀려."

    # Why (a toddler's curiosity)
    # 왜 (유딩 궁금증)
    return f"{정} 근데 {josa(상, '이면', '면')} 다 {josa(하, '이야', '야')}? 아니. 왜? {상}엔 {하}도 {형}도 있으니까."


def para(rng):
    if rng.randint(0, 5) < 2:                  # 40 percent proposition-converse-inverse-contrapositive cycle (toddler words; contrapositive holds, converse and inverse fail) / 40퍼 정역이대우 회전 (유딩말. 대우 맞고 역 이 틀림)
        return 회전(rng)
    k = rng.randint(1, 4)                       # Sample the remaining existing comparison expressions / 나머지 기존 비교 표현 샘플
    picks = []
    for _ in range(k):
        e = 표현[rng.randint(0, len(표현))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def main():
    tok = ba.AtomTokenizer()
    print(f"  유딩 논리. 대상끼리 비교 관계 인과 순서 분류. 표현 {len(표현)}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_logic", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:220].tolist()))


if __name__ == "__main__":
    main()
