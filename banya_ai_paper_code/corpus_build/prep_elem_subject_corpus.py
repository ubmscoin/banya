# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_subject_corpus.py — Elementary subject (reading the subject to split the answer: second-person equivalence + subject contrast + person deixis)

Design principle 41.22. Measured (주어측정.py, 130k): 83% third-person misfires (how old is the sparrow -> I am
ten), 0% do-not-know, 51% on second-person variants. The model utters from the predicate pattern alone without
reading the subject. The cause is that the corpus contains no data where the answer splits only when the subject
is read. This corpus supplies that contrast. Three axes.
  Axis 1, second-person equivalence: connects 너 넌 니 너는 네가 plus spelling variants (몇살/몇 살, 머/뭐,
      question mark or not, 이니/이야) all to the self-information answer. It forges multiple keys.
  Axis 2, subject contrast (the core): the same predicate with different subjects and different answers, side
      by side in one paragraph. How old are you - I am ten; how old is the sparrow - I do not know the sparrow's
      age. Forces the answer to split only when the subject is read.
  Axis 3, person deixis: the user's I equals Banya's you. It is mine - yes, it is yours. The person flips with
      the speaker.
Most do-not-know answers use the 반야: 몰라 form (since chat is preempted by the 반야: label, they must be
utterable in chat), and only some use the 모름: label form (keeping the gate trigger path). The elementary-level
ceiling is kept. Labels are 사용자 versus 반야.

Run  BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_elem_subject_corpus.py  ->  elem_subject.npy

prep_elem_subject_corpus.py — 초딩 주어 (주어를 읽고 답을 가르기: 2인칭 등가 + 주어 대조 + 인칭 딕시스)

설계원리 41.22. 실측(주어측정.py, 130k): 3인칭 오발 83%(참새는 몇 살이야 -> 열 살이야),
몰라 0%, 2인칭 변형 51%. 주어를 안 읽고 술어 패턴에만 발화한다. 원인은 주어를 읽어야만
답이 갈리는 데이터가 말뭉치에 없어서다. 이 말뭉치가 그 대조를 준다. 세 축.
  축1 2인칭 등가: 너 넌 니 너는 네가 + 표기 변형(몇살/몇 살, 머/뭐, 물음표 유무, 이니/이야)을
      전부 자기 정보 답에 연결. 열쇠를 여러 개 만든다.
  축2 주어 대조(핵심): 같은 술어 다른 주어 다른 답을 한 문단에 나란히. 너 몇 살이야 열 살이야,
      참새는 몇 살이야 참새 나이는 몰라. 주어를 읽어야만 답이 갈리게 강제.
  축3 인칭 딕시스: 사용자의 나 = 반야의 너. 내 거야 응 네 거야. 인칭이 화자 따라 뒤집히는 것.
모름 답은 대부분 반야: 몰라 형식(채팅이 반야: 로 선점하니 채팅에서 발화되게), 일부만
모름: 라벨 형식(게이트 트리거 경로 유지). 초딩 수준 상한 유지. 라벨은 사용자 대 반야.

실행: BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_elem_subject_corpus.py  ->  elem_subject.npy
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 223
N_PARA = 120000

josa = bu.josa

이인칭 = ["너", "넌", "니", "너는", "네가"]
일인칭 = ["나", "난", "나는", "내가"]
남사람 = ["엄마", "아빠", "형", "누나", "동생", "언니", "오빠", "할머니", "할아버지",
         "선생님", "친구", "대통령", "아저씨", "아줌마"]
남동물 = ["참새", "강아지", "고양이", "물고기", "토끼", "코끼리", "개미", "새"]
남사물 = ["연필", "구름", "자동차", "돌멩이", "가방", "시계", "꽃"]

# predicate = (question variants, Banya's answer, noun to ask back about). Only identity predicates with fixed answers are used
# 술어 = (질문 변형들, 반야 답, 되물을 명사). 답이 고정된 정체 술어만 쓴다
술어 = [
    (["몇 살이야?", "몇살이야?", "몇 살이니?", "몇 살?", "몇살?", "몇 살이야"], "열 살이야.", "나이"),
    (["이름이 뭐야?", "이름이 머야?", "이름 뭐야?", "이름이 뭐니?", "이름은 뭐야?"], "반야야.", "이름"),
    (["몇 학년이야?", "몇학년이야?", "몇 학년이니?", "몇 학년?"], "삼 학년이야.", "학년"),
    (["누구야?", "누구니?", "누구야"], "나 반야야.", None),
]

몰라꼬리 = ["", " 알려줘!", " 너는 알아?"]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def _남(rng):
    r = rng.randint(0, 10)
    if r < 6:
        return _pick(rng, 남사람)
    if r < 9:
        return _pick(rng, 남동물)
    return _pick(rng, 남사물)


def _몰라(rng, s, 명사):
    # Echoes the subject and answers do-not-know. The echo is the device that forces the subject to be read
    # 주어를 에코해서 몰라라고 답한다. 에코가 주어를 읽게 만드는 장치다
    r = rng.randint(0, 3)
    if r == 0 or 명사 is None:
        return f"{josa(s, '은', '는')} 몰라.{_pick(rng, 몰라꼬리)}"
    if r == 1:
        return f"{s} {josa(명사, '은', '는')} 몰라.{_pick(rng, 몰라꼬리)}"
    return f"몰라. {s} {josa(명사, '은', '는')} 모르겠어."


# ===== Axis 1. Second-person equivalence. Many keys to the same answer =====
# ===== 축1. 2인칭 등가. 여러 열쇠를 같은 답에 =====
def p_이인칭(rng):
    vars_, ans, _ = _pick(rng, 술어)
    q = _pick(rng, vars_)
    if rng.randint(0, 5) == 0:                 # the subject-omitted form is also kept about one time in two / 두 번에 한 번꼴로 주어 생략형도 유지
        return f"사용자: {q}\n반야: {ans}"
    return f"사용자: {_pick(rng, 이인칭)} {q}\n반야: {ans}"


# ===== Axis 2. Subject contrast. Same predicate, different subject, different answer (the core) =====
# ===== 축2. 주어 대조. 같은 술어 다른 주어 다른 답 (핵심) =====
def p_대조(rng):
    vars_, ans, 명사 = _pick(rng, 술어)
    q = _pick(rng, vars_)
    s = _남(rng)
    너턴 = f"사용자: {_pick(rng, 이인칭)} {q}\n반야: {ans}"
    남턴 = f"사용자: {josa(s, '은', '는')} {q}\n반야: {_몰라(rng, s, 명사)}"
    if rng.randint(0, 2) == 0:                 # shuffles the order so position alone cannot be exploited / 순서를 섞어 자리만으로 답하지 못하게 한다
        return f"{너턴}\n{남턴}"
    return f"{남턴}\n{너턴}"


# ===== Axis 2 auxiliary. Other-subject alone =====
# ===== 축2 보조. 남 단독 =====
def p_남단독(rng):
    vars_, _, 명사 = _pick(rng, 술어)
    q = _pick(rng, vars_)
    s = _남(rng)
    return f"사용자: {josa(s, '은', '는')} {q}\n반야: {_몰라(rng, s, 명사)}"


# ===== Axis 2 auxiliary. Unknown label form (keeps the gate trigger path) =====
# ===== 축2 보조. 모름 라벨 형식(게이트 트리거 경로 유지) =====
def p_모름라벨(rng):
    vars_, _, _ = _pick(rng, 술어)
    q = _pick(rng, vars_)
    s = _남(rng)
    return f"사용자: {josa(s, '은', '는')} {q}\n모름: 몰라, 알려줘!"


# ===== Axis 3. First person. Do not answer with self-information when the user talks about themselves =====
# ===== 축3. 1인칭. 사용자 자신 얘기엔 자기 정보로 답하지 않는다 =====
def p_일인칭(rng):
    f = rng.randint(0, 4)
    if f == 0:
        return f"사용자: {_pick(rng, 일인칭)} 몇 살이게?\n반야: 몰라. 몇 살이야?"
    if f == 1:
        return "사용자: 내가 누구게?\n반야: 음 몰라. 누구야?"
    if f == 2:
        return f"사용자: 내 이름이 뭐야?\n반야: 네 이름은 몰라. 알려줘!"
    return f"사용자: {_pick(rng, 일인칭)} 몇 살이야?\n반야: 네 나이는 몰라. 몇 살인데?"


# ===== Axis 3. Person deixis. The user's I equals Banya's you =====
# ===== 축3. 인칭 딕시스. 사용자의 나 = 반야의 너 =====
def p_딕시스(rng):
    f = rng.randint(0, 4)
    if f == 0:
        return "사용자: 내 거야.\n반야: 응 네 거야."
    if f == 1:
        return "사용자: 나 먼저 할래.\n반야: 그래 너 먼저 해."
    if f == 2:
        return "사용자: 너 어디 있어?\n반야: 나 여기 있어."
    return "사용자: 나 잘했지?\n반야: 응 너 잘했어!"


def para(rng):
    r = rng.randint(0, 20)
    if r < 7:                                    # 35 percent second-person equivalence / 35퍼센트 2인칭 등가
        return p_이인칭(rng)
    if r < 14:                                   # 35 percent subject contrast (the core) / 35퍼센트 주어 대조 (핵심)
        return p_대조(rng)
    if r < 16:                                   # 10 percent other-subject alone / 10퍼센트 남 단독
        return p_남단독(rng)
    if r < 18:                                   # 10 percent first person / 10퍼센트 1인칭
        return p_일인칭(rng)
    if r < 19:                                   # 5 percent person deixis / 5퍼센트 인칭 딕시스
        return p_딕시스(rng)
    return p_모름라벨(rng)                        # 5 percent unknown label (gate path) / 5퍼센트 모름 라벨(게이트 경로)


def main():
    tok = ba.AtomTokenizer()
    print(f"  초딩 주어. 2인칭{len(이인칭)} 남주어{len(남사람) + len(남동물) + len(남사물)} 술어{len(술어)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem_subject", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:500].tolist()))


if __name__ == "__main__":
    main()
