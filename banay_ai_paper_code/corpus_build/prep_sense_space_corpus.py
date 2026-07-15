# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_sense_space_corpus.py — Sense-space mapping (distance change is sense change, relative to me)

Binds the what and the where into one body. No object names are attached. As something comes closer to me the
sensation grows stronger, and as it moves away the sensation weakens. The distance axis and the sense axes are
tied into one. Near comes first and far last; small comes first and large last. Size is tied to gain and loss
relative to me: small things are gain and not scary, large things are threats and scary.

Run  python3 data_prep/prep_sense_space_corpus.py   ->  data/sense_space.npy (set the folder with BANYA_DATA_DIR)

prep_sense_space_corpus.py — 감각 공간 매핑 (거리 변화가 감각 변화, 나 기준)

무엇과 어디를 한 몸으로. 대상 이름은 안 붙인다. 나로부터 가까워지면 감각이 강해지고
멀어지면 약해진다. 거리 축과 감각 축을 하나로 묶는다. 가까운 게 앞 먼 게 뒤, 작은 게 앞 큰 게 뒤.
크기는 나 기준 득실로 묶는다. 작은 것은 득이라 안 무섭고 큰 것은 위협이라 무섭다.

실행: python3 data_prep/prep_sense_space_corpus.py   ->  data/sense_space.npy (BANYA_DATA_DIR 로 폴더 지정)
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

SEED = 53
N_PARA = 130000

접근 = ["다가온다", "가까워진다", "점점 가까워진다", "코앞까지 온다"]
후퇴 = ["멀어진다", "점점 멀어진다", "저만치 간다"]
# Far is weak and near is strong. The weak (far) end comes first, the strong (near) end last
# 멀리는 약하고 가까이는 강하다. 앞이 약(멀) 뒤가 강(가까이)
매핑 = [("가까이 오니 뜨겁다", "멀어지니 시원하다"),
        ("가까이 오니 밝다", "멀어지니 어둑하다"),
        ("가까이 오니 시끄럽다", "멀어지니 조용하다"),
        ("가까이 오니 냄새가 진하다", "멀어지니 냄새가 옅다"),
        ("가까이 오니 눈부시다", "멀어지니 어둑하다"),
        ("가까이 오니 또렷하다", "멀어지니 흐릿하다"),
        ("가까이 오니 냄새가 코를 찌른다", "멀어지니 냄새가 은은하다")]
매핑축 = ["온도", "거리", "크기", "소리크기", "다가옴", "냄새", "밝기", "투명", "높낮이", "좌우"]


def para(rng):
    r = rng.randint(0, 10)

    if r < 4:                                  # direct mapping example sentences / 매핑성 예문 직접
        return L.예문(rng, 매핑축[rng.randint(0, len(매핑축))])

    if r < 7:                                  # distance change is sense change (far-weak front, near-strong back) / 거리 변화가 감각 변화 (멀 약함 앞, 가까이 강함 뒤)
        가까이, 멀어 = 매핑[rng.randint(0, len(매핑))]
        if rng.randint(0, 2):
            return f"멀리 있다. {접근[rng.randint(0, len(접근))]}. {가까이}."
        return f"{후퇴[rng.randint(0, len(후퇴))]}. {멀어}."

    if r == 7:                                 # direction bound to sense (turning toward where it comes from) / 방향과 감각 결합 (어디서 나는지 몸을 돌린다)
        방향결합 = ["왼쪽에서 소리가 난다. 왼쪽을 본다.", "오른쪽에서 반짝인다. 오른쪽을 본다.",
                "위에서 소리가 난다. 올려다본다.", "뒤에서 발소리가 난다. 돌아본다.",
                "구석이 어둡다. 잘 안 보인다.", "건너편은 멀다. 소리가 작게 들린다."]
        return 방향결합[rng.randint(0, len(방향결합))]

    # size as gain and loss (small front not scary, large back scary; small first and large last even within a sentence)
    # 크기가 득실 (작은 앞 안 무섭다, 큰 뒤 무섭다. 한 문장에도 작은 앞 큰 뒤)
    r2 = rng.randint(0, 3)
    if r2 == 0:
        return "작은 게 온다. 나보다 작다. 안 무서워."
    if r2 == 1:
        return "큰 게 온다. 나보다 크다. 무서워."
    return "작은 건 안 무섭다. 큰 건 무섭다."


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print("  나 중심 감각공간 매핑. 가까울수록 감각 강함, 크기는 득실. 작은 앞 큰 뒤. 이름 없음")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("sense_space", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:320].tolist()))


if __name__ == "__main__":
    main()
