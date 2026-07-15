# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler2_corpus.py — toddler2 (fine-grained differentiation of baby and toddler seeds, plus web linking)

Toddler2 elaborates the toddler knowledge map at a much finer grain. One seed is rolled along 23 axes to produce
about a hundred sentences (radial spread), and the spread concepts are then linked back together (web).
Since this is the explosive period of self emergence, the self, social, evaluation, and emotion axes are thick.
The radial part is baked to toddler2.npy and the web part to toddler2_link.npy separately.

Run: BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_toddler2_corpus.py

prep_toddler2_corpus.py — 유딩2 (아기 유딩 시드 촘촘 분화 + 그물 연결)

유딩2 는 유딩 지식맵을 훨씬 촘촘히 세분화한 것이다. 시드 하나를 23축으로 굴려 문장 백여 개를 만들고
(방사), 퍼진 개념끼리 다시 잇는다(그물). 자아 창발 폭발기라 자아 사회 평가 감정 축이 두껍다.
방사는 toddler2.npy, 그물은 toddler2_link.npy 로 따로 굽는다.

실행: BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_toddler2_corpus.py
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import toddler2_expr as T
import toddler2_link_expr as W

SEED = 91
N_PARA = 200000        # number of radial paragraphs / 방사 문단 수
N_WEB = 60000          # number of web paragraphs / 그물 문단 수

씨앗들 = list(T.표현.keys())


def para_radial(rng):
    exprs = T.표현[씨앗들[rng.randint(0, len(씨앗들))]]   # pick one seed and bundle 1 to 3 expressions from it / 시드 하나 골라 그 안 표현 1~3개 묶음
    k = rng.randint(1, 4)
    picks = []
    for _ in range(k):
        e = exprs[rng.randint(0, len(exprs))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def para_web(rng):
    k = rng.randint(1, 3)
    picks = []
    for _ in range(k):
        e = W.연결[rng.randint(0, len(W.연결))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def main():
    tok = ba.AtomTokenizer()
    nr = sum(len(v) for v in T.표현.values())
    print(f"  유딩2 방사 {len(씨앗들)}시드 {nr}표현 / 그물 {len(W.연결)}표현. 자아창발 폭발기 촘촘 분화.")
    rng = np.random.RandomState(SEED)
    bu.bake("toddler2", para_radial, N_PARA, rng, tok)
    bu.bake("toddler2_link", para_web, N_WEB, rng, tok)
    print("표본(방사):")
    rng2 = np.random.RandomState(SEED + 1)
    print(para_radial(rng2))
    print("표본(그물):")
    print(para_web(rng2))


if __name__ == "__main__":
    main()
