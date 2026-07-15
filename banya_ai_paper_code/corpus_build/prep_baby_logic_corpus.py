# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_baby_logic_corpus.py — Baby-stage logic (first-person judgment and inference, centered on the self)

The subject is I and is omitted. Objects are given no names. Judgment and inference run from my own
standpoint. Bigger or smaller than me, whether my sensation is true or false, what happens if I go
near: logic anchored on me. Comparison between objects (which is bigger, a bear or an ant) is not
here; that is handed over to the toddler environment-observation corpus.
Negation is laid on top of affirmation: hot is laid down first, then not hot.
Rotation through the original, converse, inverse, and contrapositive is included explicitly: with
"if I go near, it gets hot" as the original, it is rotated into the converse, inverse, and
contrapositive. That the original equals the contrapositive and the converse equals the inverse is
also stated as judgments. Negation is trained firmly as the skeleton.

Run: python3 data_prep/prep_baby_logic_corpus.py   ->  data/baby_logic.npy (set the folder with BANYA_DATA_DIR)

아기 단계 논리 (1인칭 판정과 추론, 나 중심)

주어는 나이고 생략한다. 대상 이름은 안 붙인다. 나를 기준으로 판정하고 추론한다.
나보다 크냐 작냐, 내 감각이 참이냐 거짓이냐, 가까이 가면 어떻게 되냐 같은 나 기준 논리다.
대상끼리 비교(곰이랑 개미 누가 크냐)는 여기 없다. 그것은 유아 환경관찰로 넘긴다.
부정은 긍정 위에 얹는다. 뜨거워를 먼저 깔고 안 뜨거워.
정역이대우 회전을 확실히 넣는다. 가까이 가면 뜨거워를 정으로 두고 역 이 대우로 돌린다.
정과 대우가 같고 역과 이가 같다는 것까지 판정으로 명시한다. 부정을 골격으로 확실히 학습시킨다.

실행: python3 data_prep/prep_baby_logic_corpus.py   ->  data/baby_logic.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 61
N_PARA = 100000

수말 = ["하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"]
감각짝 = [("뜨거워", "안 뜨거워"), ("차가워", "따뜻해"), ("밝아", "어두워"), ("아파", "안 아파"),
          ("무서워", "안 무서워"), ("좋아", "싫어"), ("시끄러워", "조용해"), ("달아", "써")]
크기짝 = [("나보다 커", "응, 커. 무서워"), ("나보다 작아", "응, 작아. 안 무서워")]
# First-person conditional causation. What happens if I go near. Anchored on me
# 1인칭 조건 인과. 가까이 가면 어떻게 되나. 나 기준
조건 = [("가까이 가면", "뜨거워", "멀어지면", "안 뜨거워"),
        ("만지면", "아파", "안 만지면", "안 아파"),
        ("가까이 오면", "커 보여", "멀어지면", "작아 보여"),
        ("만지면", "차가워", "손 떼면", "괜찮아")]

# Rotation pairs over the original, converse, inverse, and contrapositive. Self-centered sensory causation. Negative forms are irregular conjugations, so they are written out by hand.
# Order: P condition, not-P condition, Q, not-Q, Q condition, not-Q condition, P statement, not-P statement
# Original: if P then Q. Converse: if Q then P. Inverse: if not P then not Q. Contrapositive: if not Q then not P. The original equals the contrapositive and the converse equals the inverse.
# 정역이대우 회전쌍. 나 중심 감각 인과. 부정형은 불규칙 활용이라 손으로 명시한다.
# 순서 P조건 아닌P조건 Q 아닌Q Q조건 아닌Q조건 P서술 아닌P서술
# 정 P면 Q. 역 Q면 P. 이 아닌P면 아닌Q. 대우 아닌Q면 아닌P. 정과 대우가 같고 역과 이가 같다.
회전쌍 = [
    ("가까이 가면", "가까이 안 가면", "뜨거워", "안 뜨거워", "뜨거우면", "안 뜨거우면", "가까이 갔어", "가까이 안 갔어"),
    ("만지면", "안 만지면", "아파", "안 아파", "아프면", "안 아프면", "만졌어", "안 만졌어"),
    ("불에 대면", "불에 안 대면", "뜨거워", "안 뜨거워", "뜨거우면", "안 뜨거우면", "불에 댔어", "불에 안 댔어"),
    ("세게 부딪히면", "안 부딪히면", "아파", "안 아파", "아프면", "안 아프면", "부딪혔어", "안 부딪혔어"),
    ("소리 지르면", "안 지르면", "시끄러워", "안 시끄러워", "시끄러우면", "안 시끄러우면", "소리 질렀어", "안 질렀어"),
]

# The negation marker an amounts to absence. Negation is the absence of that state: not hurting means there is no hurting. Negation is tied to the existence axis (there is, there is not).
# (affirmative form, absence expression)
# 안 은 곧 없다이다. 부정은 그 상태의 없음이다. 안 아파는 아픈 게 없어. 부정을 존재 축(있다 없다)에 묶는다.
# (긍정형, 없음 표현)
없다짝 = [
    ("아파", "아픈 게"), ("뜨거워", "뜨거운 게"), ("차가워", "차가운 게"), ("무서워", "무서운 게"),
    ("시끄러워", "시끄러운 게"), ("가", "가는 게"), ("먹어", "먹는 게"), ("와", "오는 게"),
]


def para(rng):
    f = rng.randint(0, 15)                     # f 8 to 12 rotates the four conditional forms, f 13 to 14 ties an to absence / f 8~12 정역이대우 회전, f 13~14 안=없다

    if f == 0:                                 # is it there or not (my perception) / 있나 없나 (내 지각)
        return "있어? 응, 있어." if rng.randint(0, 2) else "없어? 응, 없어."

    if f == 1:                                 # bigger than me (self-comparison judgment) / 나보다 크냐 (자기비교 판정)
        q, a = 크기짝[rng.randint(0, len(크기짝))]
        return f"{q}? {a}."

    if f == 2:                                 # my sensation, true or false / 내 감각 참거짓
        pos, neg = 감각짝[rng.randint(0, len(감각짝))]
        if rng.randint(0, 2):
            return f"{pos}? 응, {pos}."
        return f"{pos}? 아니, {neg}."

    if f == 3:                                 # first-person conditional causation (if I go near it is hot) / 1인칭 조건 인과 (가까이 가면 뜨겁다)
        a, b, c, d = 조건[rng.randint(0, len(조건))]
        return f"{a}? {b}. {c}? {d}."

    if f == 4:                                 # counting (as I see it) / 셈 (내가 봄)
        return "하나 있어. 또 오면? 둘이야." if rng.randint(0, 2) else "둘 있어. 하나 가면? 하나야."

    if f == 5:                                 # order inference / 순서 추론
        i = rng.randint(0, len(수말) - 1)
        return f"{수말[i]} 다음은? {수말[i + 1]}."

    if f == 6:                                 # double negation (negation on top of affirmation) / 이중부정 (긍정 위 부정)
        pos, _ = 감각짝[rng.randint(0, len(감각짝))]
        return f"안 {pos}? 아니, {pos}."

    if f == 7:                                 # gain-loss decision (from my standpoint, big things are scary) / 득실 결정 (나 기준. 큰 게 무섭다)
        return ("큰 거랑 작은 거, 뭐가 무서워? 큰 거." if rng.randint(0, 2)
                else "가까운 거랑 먼 거, 뭐가 더 커 보여? 가까운 거.")

    if f == 13:                                # an equals absence (negation is the absence of that state, tied to the existence axis) / 안 = 없다 (부정은 그 상태의 없음. 존재 축에 묶기)
        pos, 것 = 없다짝[rng.randint(0, len(없다짝))]
        return f"안 {pos}? 응, {것} 없어."

    if f == 14:                                # from there-is to there-is-not, and that is an / 있다에서 없다로, 그게 안
        pos, 것 = 없다짝[rng.randint(0, len(없다짝))]
        return f"{것} 있어? 아니, 없어. 그래서 안 {pos}."

    # Rotation of the four conditional forms in baby speech instead of formal terms: converse is backwards, inverse is the opposite, contrapositive is flipped.
    # Dangerous outcomes get no, safe outcomes get okay and good, imprinting reward and punishment (tied to life and death)
    # 정역이대우 회전. 형식어 대신 아기말로. 역은 거꾸로 이는 반대 대우는 뒤집으면.
    # 위험 결과엔 안돼 안전 결과엔 괜찮아 옳지를 붙여 상벌로 각인한다(생사와 연결)
    P, notP, Q, notQ, Qc, notQc, Ppast, notPpast = 회전쌍[rng.randint(0, len(회전쌍))]
    g = f - 8

    if g == 0:                                 # contrapositive (flipped), flipping danger into safety / 대우 (뒤집으면). 위험을 안전으로 뒤집기
        return f"{P} {Q}, 안돼. 뒤집으면? {notQc} {notPpast}, 괜찮아."

    if g == 1:                                 # original-contrapositive equivalence (as-is and flipped are the same) / 정 대우 동치 (그대로랑 뒤집은 게 똑같다)
        return f"'{P} {Q}'랑 '{notQc} {notPpast}' 똑같아? 응, 똑같아. 뒤집어도 똑같아."

    if g == 2:                                 # converse (backwards, swapping front and back) / 역 (거꾸로. 앞뒤 바꾸기)
        return f"{P} {Q}. 거꾸로는? {Qc} {Ppast}."

    if g == 3:                                 # inverse (the opposite, attaching an) / 이 (반대. 안 붙이기)
        return f"{P} {Q}, 안돼. 반대는? {notP} {notQ}, 괜찮아. 옳지."

    # Converse-inverse equivalence plus rotation summary (backwards and the opposite are the same, and the flipped one is the same as the as-is one)
    # 역 이 동치 + 회전 요약 (거꾸로랑 반대가 똑같다. 뒤집은 건 그대로랑 똑같다)
    return f"'{Qc} {Ppast}'랑 '{notP} {notQ}' 똑같아? 응. 거꾸로랑 반대는 똑같아. 뒤집은 건 그대로랑 똑같아."


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print("  나 중심 아기 논리. 나보다 크냐, 내 감각 참거짓, 가까이 가면 인과. 대상끼리 비교는 유딩으로")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("baby_logic", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:280].tolist()))


if __name__ == "__main__":
    main()
