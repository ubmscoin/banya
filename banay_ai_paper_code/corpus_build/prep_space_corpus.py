# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_space_corpus.py — Space (self-centered world model, the observation origin is the self)

The subject is I and is omitted. Even without defining the self, the very act of measuring everything by
distance, size, and direction from me plants an origin, and that origin becomes the self. No object names are
attached. Axes are laid out in order from my point of view (the world ladder). For distance, near comes first
and far last. Size, approach, and gain-loss are two-sided axes, so they are laid out in three patterns. The
ordering itself is information for gauging magnitude. The 1D exists/absent pair is the trigger that wakes
computation: when nothing is there, rest alone; when something is there, the 옴 utterance.

Run  python3 data_prep/prep_space_corpus.py   ->  data/space.npy (set the folder with BANYA_DATA_DIR)

prep_space_corpus.py — 공간 (나 중심 월드모델, 관찰 원점이 곧 자아)

주어는 나이고 생략한다. 나를 정의하지 않아도 모든 것을 나로부터의 거리 크기 방향으로 재는
것 자체가 원점을 찍고 그 원점이 자아가 된다. 대상 이름은 안 붙인다.
축을 나 기준으로 순서대로 편다(월드사다리). 거리는 가까운 게 앞 먼 게 뒤. 크기 다가옴 득실은
양쪽 축이라 세 패턴으로 편다. 순서 자체가 크기를 가늠하는 정보다.
1D 있다없다는 연산을 깨우는 트리거다. 없으면 혼자 휴식, 있으면 옴.

실행: python3 data_prep/prep_space_corpus.py   ->  data/space.npy (BANYA_DATA_DIR 로 폴더 지정)
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

SEED = 31
N_PARA = 130000

있다 = ["있다.", "왔다.", "뭔가 있다.", "또 있다.", "나타났다."]
없다 = ["없다. 혼자다.", "아무도 없다. 조용하다.", "텅 비었다. 편하다.", "없다. 쉰다."]
양방축 = ["크기", "다가옴", "득실", "높낮이", "높낮이", "넓이", "넓이"]   # duplicate weighting of below-gate axes (41.26 gate reinforcement) / 미달 축 중복 가중(41.26 게이트 보강)
자연축 = ["거리", "크기", "다가옴", "득실", "높낮이", "넓이", "좌우"]


def para(rng):
    r = rng.randint(0, 14)

    if r == 0:                                 # 1D trigger absent (rest alone) / 1D 트리거 없다 (혼자 휴식)
        return 없다[rng.randint(0, len(없다))]
    if r == 1:                                 # 1D trigger present (fires the 옴 utterance) / 1D 트리거 있다 (옴 발화)
        return 있다[rng.randint(0, len(있다))]
    if r < 5:                                  # distance ladder (near front, far back) / 거리 사다리 (가까운 앞 먼 뒤)
        return L.오름(rng, "거리")
    if r < 9:                                  # bidirectional ladders (size, approach, gain-loss; three patterns) / 양방 사다리 (크기 다가옴 득실. 세 패턴)
        return L.양방(rng, 양방축[rng.randint(0, len(양방축))])
    if r < 11:                                 # direction (categorical) / 방향 (범주)
        ax = ["방향", "좌우"][rng.randint(0, 2)]
        return L.예문(rng, ax) if rng.randint(0, 2) else L.단일(rng, ax)
    if r < 13:                                 # natural example sentences / 자연 예문
        return L.예문(rng, 자연축[rng.randint(0, len(자연축))])
    return L.단일(rng, 자연축[rng.randint(0, len(자연축))])


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print("  나 중심 공간. 거리 오름(가까운 앞 먼 뒤), 크기 다가옴 득실 양방(세 패턴). 대상 이름 없음")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("space", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:320].tolist()))


if __name__ == "__main__":
    main()
