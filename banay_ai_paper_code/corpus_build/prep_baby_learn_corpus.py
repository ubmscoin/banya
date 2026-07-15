# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_baby_learn_corpus.py — baby-stage learning (my body and my sensations, self-centered)

The subject is I and is omitted. A baby learns not object knowledge but its own body and its own
sensations. It sees with the eyes and grasps with the hands, eats when hungry, and hurts when it
falls. It also learns which organ each sensation comes from. Object knowledge (the sun is red,
fish live in water) is not here; it is passed on to toddler environment observation.

Run  python3 data_prep/prep_baby_learn_corpus.py   ->  data/baby_learn.npy (set the folder with BANYA_DATA_DIR)

prep_baby_learn_corpus.py — 아기 단계 학습 (내 몸과 자기감각, 나 중심)

주어는 나이고 생략한다. 아기는 대상 지식이 아니라 자기 몸과 자기감각을 배운다.
눈으로 보고 손으로 잡고, 배고프면 먹고 넘어지면 아프다. 감각이 어느 기관에서 오는지도 배운다.
대상 지식(해는 빨개, 물고기는 물에 산다)은 여기 없다. 그건 유딩 환경관찰로 넘긴다.

실행: python3 data_prep/prep_baby_learn_corpus.py   ->  data/baby_learn.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import seeds as S

SEED = 67
N_PARA = 100000

josa = bu.josa


def 로조사(w):                                 # the euro-ro rule: ro when there is no final consonant or it is rieul, otherwise euro / 으로 로 규칙. 받침 없거나 ㄹ받침이면 로, 나머지는 으로
    ch = w[-1]
    if not ("가" <= ch <= "힣"):
        return w + "로"
    jong = (ord(ch) - 0xAC00) % 28
    return w + "로" if jong == 0 or jong == 8 else w + "으로"


몸구어 = {"본다": "봐", "듣는다": "들어", "맡는다": "맡아", "먹는다": "먹어",
          "잡는다": "잡아", "걷는다": "걸어", "맛본다": "맛봐"}
욕구 = [("배고프면", "먹어"), ("졸리면", "자"), ("아프면", "울어"), ("기쁘면", "웃어"),
        ("무서우면", "숨어"), ("반가우면", "안아"), ("목마르면", "마셔")]
동작 = ["잡아", "놓아", "봐", "걸어", "먹어", "안아", "던져", "웃어", "울어"]
인과 = [("넘어지면", "아파"), ("부딪히면", "아파"), ("안으면", "따뜻해"),
        ("만지면", "느껴"), ("뛰면", "숨차"), ("맞으면", "아파")]
귀속 = [("뜨거운 건", "손으로 알아"), ("밝은 건", "눈으로 봐"), ("큰 소리는", "귀로 들어"),
        ("단 건", "혀로 알아"), ("냄새는", "코로 맡아"), ("아픈 건", "몸으로 느껴")]
내상태 = ["아파", "무서워", "졸려", "배고파", "추워", "더워"]


def para(rng, 몸부위):
    f = rng.randint(0, 6)

    if f == 0:                                 # body function (what my body does) / 몸 기능 (내 몸이 뭘 하나)
        부위 = 몸부위[rng.randint(0, len(몸부위))]
        기능 = 몸구어[S.몸씨앗[부위]]
        return f"{로조사(부위)} 뭐 해? {기능}."

    if f == 1:                                 # a need turns into an action / 욕구가 행동으로
        q, a = 욕구[rng.randint(0, len(욕구))]
        return f"{q}? {a}."

    if f == 2:                                 # my action / 내 동작
        return f"{동작[rng.randint(0, len(동작))]}."

    if f == 3:                                 # self causation (falling hurts) / 자기 인과 (넘어지면 아파)
        q, a = 인과[rng.randint(0, len(인과))]
        return f"{q}? {a}."

    if f == 4:                                 # sense attribution (hot things are known through the hand) / 감각 귀속 (뜨거운 건 손으로)
        q, a = 귀속[rng.randint(0, len(귀속))]
        return f"{q}? {a}."

    # Affirmation on top of negation (true or false of my state)
    # 부정 위 긍정 (내 상태 참거짓)
    st = 내상태[rng.randint(0, len(내상태))]
    return f"안 {st}? 아니, {st}."


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    몸부위 = [k for k in S.몸씨앗 if S.in_vocab(k, tok)]
    print(f"  나 중심 아기 학습. 몸 기능, 욕구 행동, 자기 인과, 감각 귀속. 대상 지식은 유딩으로. 몸{len(몸부위)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("baby_learn", lambda r: para(r, 몸부위), N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:280].tolist()))


if __name__ == "__main__":
    main()
