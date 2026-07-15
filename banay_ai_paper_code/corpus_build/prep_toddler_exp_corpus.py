# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_exp_corpus.py — toddler experience (kindergarten events + rough play)

Events the self goes through at kindergarten. Covers an actual kindergarten curriculum (free play, outdoor play,
the daily routine) plus family outings and special days. A toddler plays until scolded and makes a mess when eating, and such rough events are included as well.
Sources are toddler_expr.py (toddler experience) + toddler_events.py (eight experience categories) + toddler_affect.py (overplay, mess making).

Run: python3 data_prep/prep_toddler_exp_corpus.py   ->  data/toddler_exp.npy (set the output folder with BANYA_DATA_DIR)

prep_toddler_exp_corpus.py — 유딩 경험 (유치원 사건 + 거친 놀이)

유치원에서 자아가 겪는 사건이다. 실제 유치원 커리큘럼(자유놀이 바깥놀이 하루일과)과
가족 나들이 특별한 날을 담는다. 유딩은 혼날 때까지 놀고 먹으면 어지럽힌다. 그 거친 사건도 넣는다.
소스는 toddler_expr.py(유딩_경험) + toddler_events.py(경험 8범주) + toddler_affect.py(과잉놀이 어지럽힘)다.

실행: python3 data_prep/prep_toddler_exp_corpus.py   ->  data/toddler_exp.npy (BANYA_DATA_DIR 로 폴더 지정)
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

SEED = 89
N_PARA = 130000


def _끝(e):
    return e if e and e[-1] in ".?!" else e + "."


풀 = [_끝(e) for e in T.표현["toddler_exp"]]
for v in E.경험.values():                        # categories: free play, outdoor play, daily routine, home, errands, outings, special days, getting scolded / 자유놀이 바깥놀이 하루일과 집 심부름 나들이 특별한날 혼남
    풀 += [_끝(e) for e in v]
for v in J.경험.values():                        # categories: overplay, mess making / 과잉놀이 어지럽힘
    풀 += [_끝(e) for e in v]


def para(rng):
    k = rng.randint(2, 5)                        # experience consists of events, so several sentences are bundled like a narrative / 경험은 사건이라 여러 문장 묶어 서사처럼
    picks = []
    for _ in range(k):
        e = 풀[rng.randint(0, len(풀))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def main():
    tok = ba.AtomTokenizer()
    print(f"  유딩 경험. 유치원 커리큘럼 사건 + 거친 놀이. 표현 {len(풀)}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_exp", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:280].tolist()))


if __name__ == "__main__":
    main()
