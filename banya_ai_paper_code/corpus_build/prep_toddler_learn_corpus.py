# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_learn_corpus.py — toddler learning (kindergarten knowledge + curiosity about the world)

Knowledge learned at kindergarten: colors, shapes, numbers, seasons, days of the week, and classification.
Curiosity about everything in the world and the joy of coming to know are added as well.
A toddler is intensely curious about everything in the world and greatly delighted on hearing an answer.
Sources are toddler_expr.py (toddler learning) + toddler_questions.py (curiosity about everything, joy of knowing).

Run: python3 data_prep/prep_toddler_learn_corpus.py   ->  data/toddler_learn.npy (set the output folder with BANYA_DATA_DIR)

prep_toddler_learn_corpus.py — 유딩 학습 (유치원 지식 + 세상 궁금증)

유치원에서 배우는 지식이다. 색 모양 숫자 계절 요일 분류. 그리고 세상 만물에 대한 궁금증과
알아가는 즐거움을 넣는다. 유딩은 세상 모든 것이 극도로 궁금하고, 답을 들으면 크게 즐거워한다.
소스는 toddler_expr.py(유딩_학습) + toddler_questions.py(세상만물궁금 알아서즐거움)다.

실행: python3 data_prep/prep_toddler_learn_corpus.py   ->  data/toddler_learn.npy (BANYA_DATA_DIR 로 폴더 지정)
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
import toddler_questions as Q

SEED = 83
N_PARA = 100000


def _끝(e):
    return e if e and e[-1] in ".?!" else e + "."


풀 = [_끝(e) for e in T.표현["toddler_learn"]]
for v in Q.학습.values():                        # categories: curiosity about everything, joy of knowing / 세상만물궁금 알아서즐거움
    풀 += [_끝(e) for e in v]


def para(rng):
    k = rng.randint(1, 4)
    picks = []
    for _ in range(k):
        e = 풀[rng.randint(0, len(풀))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def main():
    tok = ba.AtomTokenizer()
    print(f"  유딩 학습. 색 모양 숫자 계절 분류 + 세상 궁금증 + 알아가는 즐거움. 표현 {len(풀)}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_learn", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:260].tolist()))


if __name__ == "__main__":
    main()
