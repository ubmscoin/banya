# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_sense_corpus.py — Senses (my five senses, what is felt at the observation origin)

The subject is I and is omitted. No object names are attached. Only the sensations I feel are listed.
Axes are laid out in order from my point of view (the world ladder). For brightness, loudness, pain, arousal,
and texture, the weak end comes first and the strong end last. Temperature is a two-sided axis, so it is laid
out in three patterns: my body temperature is the zero point, the cold side and the hot side each on their own,
and then the whole axis. Intensity is understood relative to me, and hotness is also understood from the
ordering differences along the temperature axis itself.

Run  python3 data_prep/prep_sense_corpus.py   ->  data/sense.npy (set the folder with BANYA_DATA_DIR)

prep_sense_corpus.py — 감각 (내 오감, 관찰 원점에서 느끼는 것)

주어는 나이고 생략한다. 대상 이름은 안 붙인다. 내가 느끼는 감각만 나열한다.
축을 나 기준으로 순서대로 편다(월드사다리). 밝기 소리 통증 각성 질감은 약한 게 앞 강한 게 뒤.
온도는 양쪽 축이라 세 패턴으로 편다. 내 체온이 0이고 차가운 쪽 뜨거운 쪽으로 각각, 그리고 전체.
나를 기준으로 강도를 이해하고 온도 자체의 나열 차에서 뜨거로도 이해한다.

실행: python3 data_prep/prep_sense_corpus.py   ->  data/sense.npy (BANYA_DATA_DIR 로 폴더 지정)
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

SEED = 43
N_PARA = 130000

한쪽감각 = ["밝기", "소리크기", "통증", "각성", "질감", "습기", "끈기", "탄성", "둔통", "속느낌", "투명"]
신설한쪽 = ["습기", "끈기", "탄성", "둔통", "속느낌", "투명"]   # focused reinforcement of axes below the 30k gate (41.26) / 3만 게이트 미달 축 집중 보강(41.26)
범주감각 = ["색", "맛", "냄새", "형태"]
전체감각 = ["밝기", "소리크기", "온도", "통증", "맛", "색", "냄새", "질감", "각성", "습기", "탄성", "둔통", "속느낌", "투명", "형태", "날씨", "소리높이"]


def para(rng):
    r = rng.randint(0, 14)

    if r < 3:                                  # one-sided sense ladder (weak front, strong back) / 한쪽 감각 사다리 (약한 앞 강한 뒤)
        return L.오름(rng, 한쪽감각[rng.randint(0, len(한쪽감각))])
    if r < 6:                                  # focus on newly added sub-axes (reinforcing below-gate axes) / 신설 하위축 집중 (게이트 미달 보강)
        ax = 신설한쪽[rng.randint(0, len(신설한쪽))]
        return L.오름(rng, ax) if rng.randint(0, 2) else L.예문(rng, ax)
    if r < 8:                                  # bidirectional temperature (cold, hot, whole; three patterns) / 온도 양방 (차 뜨 전체 세 패턴)
        return L.양방(rng, ["온도", "날씨", "소리높이"][rng.randint(0, 3)])
    if r < 11:                                 # categorical senses (color, taste, smell) / 범주 감각 (색 맛 냄새)
        ax = 범주감각[rng.randint(0, len(범주감각))]
        return L.예문(rng, ax) if rng.randint(0, 2) else L.단일(rng, ax)
    return L.예문(rng, 전체감각[rng.randint(0, len(전체감각))])


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print("  나 중심 감각. 밝기 소리 통증 각성 질감 오름(약한 앞 강한 뒤), 온도 양방(세 패턴). 이름 없음")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("sense", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:320].tolist()))


if __name__ == "__main__":
    main()
