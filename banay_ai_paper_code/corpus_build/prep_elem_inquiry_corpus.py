# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_inquiry_corpus.py — Elementary-stage inquiry (feedback meaning search + multiple resolution + question emission)

Design principles 41.2, 41.9, 41.10, 41.14. The new core of the elementary stage. It links the
unknown to the known (bridge bootstrapping), gets excited when one input resolves several things at
once (multiple resolution = dopamine proportional to count), and when pressure builds Banya asks
first (question emission). The toddler state corpus (유딩_상태: certain, ambiguous, unknown) is
assumed, and inquiry is laid on top of it.

Run: python3 data_prep/prep_elem_inquiry_corpus.py   ->  data/elem_inquiry.npy (set the folder with BANYA_DATA_DIR)

초등 단계 탐구 (되먹임 의미 탐색 + 다중 해소 + 질문 방출)

설계원리 41.2 41.9 41.10 41.14. 초등 단계의 신규 핵심. 모르는 것을 아는 것으로 잇는다(다리 부트스트랩),
한 입력이 여러 개를 같이 풀면 흥분한다(다중 해소 = 도파민 개수비례), 압력 차면 반야가 먼저 묻는다(질문 방출).
유딩_상태(확실 모호 모름)를 전제로 그 위에 탐구를 얹는다.

실행: python3 data_prep/prep_elem_inquiry_corpus.py   ->  data/elem_inquiry.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 131
N_PARA = 90000

josa = bu.josa

# Parent concepts and their children (known categories). Bridging and multiple resolution operate on top of these
# 상위 개념과 하위들(아는 카테고리). 다리/다중 해소가 이 위에서 작동한다
클러스터 = [
    ("동물", ["곰", "개미", "참새", "기린", "사자", "토끼", "강아지", "고양이"]),
    ("과일", ["사과", "포도", "딸기", "바나나", "귤", "수박"]),
    ("색", ["빨강", "노랑", "파랑", "초록", "보라"]),
    ("탈것", ["기차", "버스", "배", "비행기", "자동차"]),
    ("채소", ["당근", "오이", "배추", "무"]),
    ("맛", ["단맛", "신맛", "쓴맛", "짠맛"]),
]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def para(rng):
    상위, 하위들 = 클러스터[rng.randint(0, len(클러스터))]
    r = rng.randint(0, 10)

    상위는 = josa(상위, '은', '는')
    상위야 = josa(상위, '이야', '야')
    상위니까 = josa(상위, '이니까', '니까')
    상위구나 = josa(상위, '이구나', '구나')

    if r < 3:                                    # bridge + bootstrap (linking the unknown to a known parent) / 다리 + 부트스트랩 (모르는 걸 아는 상위로 잇기)
        미지 = _pick(rng, 하위들)
        return (f"사용자: {josa(미지, '이', '가')} 뭐야?\n"
                f"반야: {josa(미지, '은', '는')} 몰라. 근데 {상위는} 알아. {미지}도 {상위야}?\n"
                f"사용자: 응, 맞아.\n"
                f"반야: 아 {상위니까} 이제 {미지} 알겠다!")

    if r < 5:                                    # bootstrap (resolves at once when the user supplies the bridge) / 부트스트랩 (사용자가 다리 주면 바로 해소)
        미지 = _pick(rng, 하위들)
        return (f"사용자: {josa(미지, '은', '는')} {상위} 같은 거야.\n"
                f"반야: {상위} 아니까 {미지}도 알겠다!")

    if r < 8:                                    # multiple resolution (resolving several at once excites, dopamine proportional to count) / 다중 해소 (여러 개 같이 풀면 흥분 = 도파민 개수비례)
        n = rng.randint(2, min(4, len(하위들)) + 1)
        골라 = list(rng.choice(하위들, size=n, replace=False))
        나열 = " ".join(골라)
        each = " ".join(f"{w}도 {josa(상위, '이고', '고')}" for w in 골라)
        return (f"사용자: {josa(나열, '은', '는')} 다 {상위야}.\n"
                f"반야: {each} 다 {상위구나}! {n}개나 알았다! 신난다!")

    # Question emission (when pressure builds, Banya asks first)
    # 질문 방출 (압력 차면 반야가 먼저 묻는다)
    미지 = _pick(rng, 하위들)
    운 = ["나 궁금한 게 있어.", "모르는 게 많아.", "이거 알고 싶어."]
    return (f"반야: {_pick(rng, 운)} {josa(미지, '이', '가')} 뭐야?\n"
            f"사용자: {상위}의 한 가지야.\n"
            f"반야: 아 {상위구나}! 이제 {미지} 알았다!")


def main():
    tok = ba.AtomTokenizer()
    print(f"  초딩 탐구. 되먹임 다리 + 다중 해소 + 질문 방출. 클러스터 {len(클러스터)}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem_inquiry", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:420].tolist()))


if __name__ == "__main__":
    main()
