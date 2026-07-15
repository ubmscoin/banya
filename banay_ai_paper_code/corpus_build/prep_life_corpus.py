# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_life_corpus.py — Life (birth, being alive, death; the very bottom layer, existence itself)

Design principle 41.26, Banya world model 1.3. The top-priority warm-up subject. The subject is I and is omitted.
Birth (information proliferation, increase), being alive (maintenance, the default), and death (ambiguous dissolution,
decrease) are laid out as a bidirectional ladder. The death extreme comes first, being alive sits in the middle,
and the birth extreme comes last. The time axis (past, now, later) is laid out here as well, since existence is
what persists over time and the two form one body. Four bridges are embedded in the example sentences:
the existence-sense bridge (I feel, it exists), the information-life bridge (I learned, I am excited; I do not
know, I am scared), my own birth, and threat-survival.

Run  python3 data_prep/prep_life_corpus.py   ->  data/life.npy (set the output folder with BANYA_DATA_DIR)

prep_life_corpus.py — 생명 (탄생 살아있음 죽음. 가장 바닥, 존재 그 자체)

설계원리 41.26, 반야월드모델 1.3. 워밍업 최우선 과목. 주어는 나이고 생략한다.
탄생(정보 증식, 늘어남) 살아있음(유지, 디폴트) 죽음(모호 소멸, 줄어듦)을 양방 사다리로 편다.
죽음 극단이 앞, 살아있음이 가운데, 탄생 극단이 뒤. 시간 축(옛날 지금 나중)도 여기서 함께 편다.
존재가 시간 위에서 유지되는 것이라 한 몸이다. 예문에 네 다리가 박혀 있다.
존재-감각 다리(느낀다 있다), 정보-생명 다리(알았다 신난다, 몰라 무섭다), 나의 탄생, 위협-생환.

실행: python3 data_prep/prep_life_corpus.py   ->  data/life.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import world_ladder as L

SEED = 61
N_PARA = 130000


def para(rng):
    r = rng.randint(0, 14)

    if r < 4:                                  # bidirectional life ladder (death front, alive middle, birth back) / 생명 양방 사다리 (죽음 앞, 살아있음 가운데, 탄생 뒤)
        return L.양방(rng, "life")
    if r < 7:                                  # life example sentences (the four bridges) / 생명 예문 (네 다리)
        return L.예문(rng, "life")
    if r < 10:                                 # bidirectional time ladder (past, now, later; raised to reinforce a below-gate axis) / 시간 양방 사다리 (옛날 지금 나중. 게이트 미달 보강 상향)
        return L.양방(rng, "시간")
    if r < 13:                                 # time example sentences / 시간 예문
        return L.예문(rng, "시간")
    return L.단일(rng, "시간")


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print("  생명. 탄생(증식) 살아있음(유지) 죽음(소멸) 양방 + 시간(옛날 지금 나중) 양방. 이름 없음")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("life", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:320].tolist()))


if __name__ == "__main__":
    main()
