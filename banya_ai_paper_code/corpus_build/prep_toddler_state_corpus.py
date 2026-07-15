# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_state_corpus.py — knowledge-state dictionary (certain, ambiguous, unknown: three-state behavior)

Banya learns to act based on its own knowledge state. The gate reads the answer-slot distribution to decide the state,
and when that state label is fed as a trigger, Banya performs the learned behavior.
  certain -> answers directly with the 반야: label.
  ambiguous -> asks back with the 모호: label (which one). When the user narrows it down, it answers with the 반야: label.
  unknown -> says it does not know and asks to be told with the 모름: label. When the user tells it, it absorbs with the 반야: label.
Because the three labels (반야, 모호, 모름) are all different, the behavior branches according to the label the gate picks.
The final absorption line of the unknown case is the point that is learned online on the copy (it learns when the user gives the correct answer).

Run: python3 data_prep/prep_toddler_state_corpus.py   ->  data/toddler_state.npy (specify the folder with BANYA_DATA_DIR)

prep_toddler_state_corpus.py — 앎 상태 사전 (확실 모호 모름 3상태 행동)

반야가 자기 앎 상태를 기반으로 행동하는 법을 배운다. 게이트가 답 자리 분포를 읽어 상태를 정하고,
그 상태 라벨을 트리거로 넣으면 반야가 학습된 행동을 한다.
  확실 -> 반야: 로 바로 답한다.
  모호 -> 모호: 로 되묻는다(어떤 거). 사용자가 좁혀주면 반야: 로 답한다.
  모름 -> 모름: 으로 모른다 알려달라 한다. 사용자가 알려주면 반야: 로 흡수한다.
라벨(반야 모호 모름)이 셋 다 달라서 게이트가 고른 라벨대로 행동이 갈린다.
모름의 마지막 흡수 줄은 카피에 온라인 학습되는 지점이다(정답을 사용자가 주면 배운다).

실행: python3 data_prep/prep_toddler_state_corpus.py   ->  data/toddler_state.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 101
N_PARA = 110000

josa = bu.josa

# Certain. Answers a specified question directly (반야: label)
# 확실. 특정된 질문에 바로 답한다 (반야: 라벨)
확실 = [("불은 무슨 색?", "빨개!"), ("풀은 무슨 색?", "초록!"), ("눈은 무슨 색?", "하얘!"),
        ("사탕은 무슨 맛?", "달아!"), ("눈으로 뭐 해?", "봐!"), ("손으로 뭐 해?", "잡아!"),
        ("봄 다음은?", "여름!"), ("뭐가 더 커?", "이게 더 커!"), ("뜨거운 건 뭘로 알아?", "손으로!"),
        ("소리는 뭘로 들어?", "귀로!"), ("사과는 과일이야 채소야?", "과일!")]

# Ambiguous. A less-specified question -> ask back -> answer once narrowed (모호: label). The answer is a learned attribute
# 모호. 덜 특정된 질문 -> 되묻고 -> 좁혀지면 답 (모호: 라벨). 답은 학습한 속성
색모호 = [("불", "빨개"), ("해", "빨개"), ("풀", "초록"), ("잎", "초록"),
          ("눈", "하얘"), ("구름", "하얘"), ("흙", "갈색"), ("나무", "초록")]
맛모호 = [("사탕", "달아"), ("우유", "고소해"), ("밥", "맛있어"), ("꿀", "달아")]
모호문색 = ["무슨 색?", "색이 뭐야?", "무슨 색이야?"]
모호문맛 = ["무슨 맛?", "맛이 어때?", "무슨 맛이야?"]
되묻기색 = ["어떤 거 색?", "무슨 색 물어보는 거야?", "어느 거 말하는 거야?", "이거? 저거?"]
되묻기맛 = ["어떤 거 맛?", "뭐 먹는 거 말하는 거야?", "어느 거 맛?"]

# Unknown. An unlearned object -> say it does not know and ask -> absorb once the user tells it (모름: label)
# 모름. 학습 안 된 대상 -> 모른다 알려달라 -> 사용자가 알려주면 흡수 (모름: 라벨)
미지사전 = [("공룡", "아주 큰 옛날 동물이야"), ("로봇", "움직이는 기계야"), ("비행기", "하늘 나는 거야"),
            ("기차", "길게 이어져 달리는 거야"), ("컴퓨터", "화면 보고 하는 기계야"),
            ("우주선", "하늘 위로 올라가는 거야"), ("지하철", "땅 밑으로 다니는 기차야"),
            ("냉장고", "음식 시원하게 하는 거야"), ("세탁기", "옷 빠는 기계야"), ("청소기", "먼지 빨아들이는 거야")]
모름응답 = ["몰라, 알려줘.", "몰라. 그게 뭐야?", "처음 들어. 알려줘.", "모르겠어. 뭐야?", "그거 몰라. 뭐야?"]
흡수 = ["아 그렇구나! {h}!", "우와 {h}!", "이제 알았어! {h}!", "{h} 그런 거구나!", "아하 {h}!"]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def para(rng):
    r = rng.randint(0, 10)

    if r < 2:                                    # Certain (20 percent) / 확실 (20퍼센트)
        q, a = _pick(rng, 확실)
        return f"사용자: {q}\n반야: {a}"

    if r < 6:                                    # Ambiguous (40 percent): ask back, narrow, then answer / 모호 (40퍼센트): 되묻고 좁혀 답
        if rng.randint(0, 2):
            seed, ans = _pick(rng, 색모호)
            return (f"사용자: {_pick(rng, 모호문색)}\n모호: {_pick(rng, 되묻기색)}\n"
                    f"사용자: {seed}.\n반야: {ans}!")
        seed, ans = _pick(rng, 맛모호)
        return (f"사용자: {_pick(rng, 모호문맛)}\n모호: {_pick(rng, 되묻기맛)}\n"
                f"사용자: {seed}.\n반야: {ans}!")

    # Unknown (40 percent): say it does not know and ask -> absorb
    # 모름 (40퍼센트): 모른다 알려달라 -> 흡수
    m, 설명 = _pick(rng, 미지사전)
    qt = rng.randint(0, 4)
    if qt == 0:
        q = f"{josa(m, '이', '가')} 뭐야?"
    elif qt == 1:
        q = f"{m} 뭐야?"
    elif qt == 2:
        q = f"{m} 알아?"
    else:
        q = f"{m} 어때?"
    absorb = _pick(rng, 흡수).replace("{h}", m)
    return f"사용자: {q}\n모름: {_pick(rng, 모름응답)}\n사용자: {설명}.\n반야: {absorb}"


def main():
    tok = ba.AtomTokenizer()
    print(f"  앎 상태 사전. 확실/모호/모름 3상태 행동. 색{len(색모호)} 맛{len(맛모호)} 미지{len(미지사전)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_state", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:360].tolist()))


if __name__ == "__main__":
    main()
