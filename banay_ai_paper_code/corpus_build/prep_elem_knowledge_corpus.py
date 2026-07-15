# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_knowledge_corpus.py — Elementary knowledge (certain/ambiguous/unknown gating + elementary-level knowledge acquisition)

Design principle 41.14. The behavioral mechanism for certain, ambiguous, and unknown is the same as in the
kindergarten-state stage (that one is maintained through rehearsal); this corpus fills it with elementary-level
knowledge (seasons, numbers, nature, letters). Stage boundaries are preserved: kindergarten state carries
kindergarten knowledge, elementary knowledge carries elementary knowledge. The final absorption line of the
unknown case is the point where online learning (the hippocampus module) takes place.

Run  python3 data_prep/prep_elem_knowledge_corpus.py   ->  data/elem_knowledge.npy (set the folder with BANYA_DATA_DIR)

prep_elem_knowledge_corpus.py — 초딩 지식 (확실/모호/모름 게이팅 + 초딩 수준 지식 습득)

설계원리 41.14. 확실 모호 모름 행동 기제는 유딩_상태와 같고(그건 rehearsal 로 유지), 여기는 초딩 수준
지식(계절 숫자 자연 글자)으로 채운다. 스테이지 경계 보존: 유딩_상태=유딩 지식, 초딩_지식=초딩 지식.
모름의 마지막 흡수 줄이 온라인 학습(해마체) 되는 지점이다.

실행: python3 data_prep/prep_elem_knowledge_corpus.py   ->  data/elem_knowledge.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 167
N_PARA = 100000

josa = bu.josa

# Certain. Things an elementary schooler knows for sure -> answer right away
# 확실. 초딩이 확실히 아는 것 -> 바로 답
확실 = [("일주일은 며칠?", "칠 일!"), ("숫자 삼 다음은?", "사!"), ("물이 얼면 뭐가 돼?", "얼음!"),
        ("얼음이 녹으면?", "물!"), ("해는 어디서 떠?", "동쪽!"), ("봄 여름 다음은?", "가을!"),
        ("가을 다음은?", "겨울!"), ("낮 다음은?", "밤!"), ("하나 둘 다음은?", "셋!"),
        ("무지개는 몇 색?", "일곱 색!"), ("비 오면 하늘에 뭐 떠?", "구름!"), ("밤하늘에 뭐가 반짝여?", "별!")]

# Ambiguous. An under-specified question -> ask back -> answer once it is narrowed down
# 모호. 덜 특정된 질문 -> 되묻고 -> 좁혀지면 답
계절모호 = [("봄", "따뜻해"), ("여름", "더워"), ("가을", "선선해"), ("겨울", "추워")]
자연모호 = [("바다", "넓어"), ("산", "높아"), ("강", "흘러"), ("들", "넓어")]
모호문계절 = ["무슨 계절?", "계절이 어때?", "어느 철이야?"]
모호문자연 = ["거기 어때?", "어떤 곳이야?", "무슨 곳?"]
되묻기 = ["어떤 거 말하는 거야?", "어느 거? 봄? 여름?", "뭐 물어보는 거야?", "이거? 저거?"]

# Unknown. Things unknown at the elementary level -> say do-not-know and ask to be told -> absorb
# 모름. 초딩 미지 -> 모른다 알려달라 -> 흡수
미지사전 = [("바다", "아주 넓은 물이야"), ("산", "아주 높은 땅이야"), ("별", "밤하늘에 반짝이는 거야"),
            ("달", "밤에 뜨는 둥근 거야"), ("구름", "하늘에 뜬 물방울이야"), ("무지개", "비 온 뒤 일곱 색 띠야"),
            ("지도", "땅을 그린 그림이야"), ("나침반", "방향 알려 주는 거야"), ("글자", "말을 적은 거야"),
            ("숫자", "세는 데 쓰는 거야"), ("시계", "시간 알려 주는 거야"), ("달력", "날짜 적은 거야"),
            ("나라", "사람들이 모여 사는 큰 땅이야"), ("바람", "공기가 움직이는 거야")]
모름응답 = ["몰라, 알려줘.", "몰라. 그게 뭐야?", "처음 들어. 알려줘.", "모르겠어. 뭐야?", "그거 몰라. 뭐야?"]
흡수 = ["아 그렇구나! {h}!", "우와 {h}!", "이제 알았어! {h}!", "{h} 그런 거구나!", "아하 {h}!"]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def para(rng):
    r = rng.randint(0, 10)

    if r < 2:                                    # certain (20 percent) / 확실 (20퍼센트)
        q, a = _pick(rng, 확실)
        return f"사용자: {q}\n반야: {a}"

    if r < 6:                                    # ambiguous (40 percent): ask back, narrow down, answer / 모호 (40퍼센트): 되묻고 좁혀 답
        if rng.randint(0, 2):
            seed, ans = _pick(rng, 계절모호)
            return (f"사용자: {_pick(rng, 모호문계절)}\n모호: {_pick(rng, 되묻기)}\n"
                    f"사용자: {seed}.\n반야: {ans}!")
        seed, ans = _pick(rng, 자연모호)
        return (f"사용자: {_pick(rng, 모호문자연)}\n모호: {_pick(rng, 되묻기)}\n"
                f"사용자: {seed}.\n반야: {ans}!")

    # unknown (40 percent): say do-not-know and ask to be told -> absorb
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
    print(f"  초딩 지식. 확실/모호/모름 + 초딩 지식. 확실{len(확실)} 미지{len(미지사전)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem_knowledge", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:400].tolist()))


if __name__ == "__main__":
    main()
