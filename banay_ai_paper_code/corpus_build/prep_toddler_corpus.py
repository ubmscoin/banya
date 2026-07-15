# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_corpus.py — toddler base (self emergence + social life + intense affect + question explosion + environment observation)

A revolutionary change from the baby stage. A self has emerged, and it is strong and rough. The toddler is stubborn, extreme, and strongly possessive.
Communicating with the world is itself a reward, so everything is enjoyable. Early childhood is heaven, and the child is happiest under the mother's protection.
Questions also explode. The toddler endlessly asks what is this, why, how, and what does it taste like. Curiosity and energy are boundless.
Self, social, affect, and question material forms the main share (60 percent), mixed with object observation via seed branching (40 percent).

Run: python3 data_prep/prep_toddler_corpus.py   ->  data/toddler.npy (set the output folder with BANYA_DATA_DIR)

prep_toddler_corpus.py — 유딩 base (자아 창발 + 사회 + 진한 정서 + 질문 폭발 + 환경관찰)

아기에서 혁명적 변화다. 자아가 생겼고 강력하고 거칠다. 고집 세고 극단적이며 소유욕이 세다.
세상과 소통하는 것 자체가 보상이라 모든 게 즐겁다. 유년기는 천국이고 엄마 보호 아래 가장 행복하다.
그리고 질문이 폭발한다. 이건 뭐야 왜 어떻게 무슨 맛이야 를 끝없이 묻는다. 호기심과 에너지가 무한이다.
자아 사회 정서 질문을 주로(60퍼센트), 대상 관찰(가지치기)을 곁들여(40퍼센트) 섞는다.

실행: python3 data_prep/prep_toddler_corpus.py   ->  data/toddler.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import toddler_expr as T
import toddler_events as E
import toddler_affect as J
import toddler_questions as Q
import toddler_seed_expr as G

SEED = 73
N_PARA = 150000


def _끝(e):
    return e if e and e[-1] in ".?!" else e + "."


def build_자아사회():
    pool = []
    for f in ["자아", "친구", "선생님유치원", "자기대상상호작용", "사회감정"]:
        pool += T.표현[f]
    for v in E.정서.values():          # categories: waiting for mom, the world is big, small satisfactions / 엄마기다림 세상은크다 소소한만족
        pool += v
    for v in J.base.values():          # categories: stubborn tantrums, extreme emotion, possessiveness, heaven, mom's protection, intensity, no judgment / 고집떼쓰기 극단감정 소유욕 천국 엄마보호 강렬 무판단
        pool += v
    for v in Q.base.values():          # categories: what is this, why explosion, how, sense questions, exploration / 이게뭐야 왜폭발 어떻게 감각질문 탐험
        pool += v
    return [_끝(e) for e in pool]


자아사회 = build_자아사회()
씨앗들 = list(G.표현.keys())


def para(rng):
    if rng.randint(0, 10) < 6:                  # 60 percent self, social, affect, question / 60퍼센트 자아 사회 정서 질문
        exprs = 자아사회
        붙음 = True
    else:                                       # 40 percent object observation (seed branching) / 40퍼센트 대상 관찰 (가지치기)
        exprs = G.표현[씨앗들[rng.randint(0, len(씨앗들))]]
        붙음 = False
    k = rng.randint(1, 4)
    picks = []
    for _ in range(k):
        e = exprs[rng.randint(0, len(exprs))]
        if e not in picks:
            picks.append(e)
    if 붙음:
        return " ".join(picks)
    return ". ".join(picks) + "."


def main():
    tok = ba.AtomTokenizer()
    print(f"  유딩 base. 자아사회정서질문 {len(자아사회)}표현 + 가지치기 {len(씨앗들)}씨앗. 자아 창발, 질문 폭발.")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:320].tolist()))


if __name__ == "__main__":
    main()
