# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_dialog_corpus.py — Elementary-stage dialogue (fluent elementary-level speech: multi-turn + everyday vocabulary + self-identity)

Design principle 41.19. The elementary stage had no dedicated dialogue corpus. The toddler stage
had the toddler dialogue corpus (유딩_대화), but it dropped out on the move to the elementary stage.
That is the reason elementary dialogue came out weak. This corpus fills that channel. The core is
multi-turn exchange (holding the topic, asking back and continuing) to make dialogue fluent, weaving
everyday vocabulary (home, family, food, feelings, play, objects, body, weather, neighborhood) into
dialogue to widen the lexicon, and fixing self-identity (who, age, grade, name) so identity questions
do not shake it. The elementary-level ceiling is kept. Labels are unified as user versus Banya
(the same inference trigger as the toddler dialogue corpus).

Run: BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_elem_dialog_corpus.py  ->  elem_dialog.npy

초등 단계 대화 (능숙한 초등 수준 말하기: 멀티턴 + 일상어휘 + 자기정체)

설계원리 41.19. 초등 단계에는 대화 전용 말뭉치가 없었다. 유아 단계에는 유딩_대화가 있었는데 초등 단계로 넘어오며 빠졌다.
그것이 초등 대화가 약해진 원인이다. 이 말뭉치가 그 통로를 채운다. 핵심은 멀티턴 주고받기(주제 유지,
되묻고 잇기)로 대화를 능숙하게, 일상 어휘(집 가족 음식 감정 놀이 물건 몸 날씨 동네)를 대화 속에
녹여 어휘를 넓히고, 자기 정체(누구 몇살 몇학년 이름)를 고정해 정체 질문에 흔들리지 않게 한다.
초등 수준 상한 유지. 라벨은 사용자 대 반야로 통일(유딩_대화와 같은 추론 트리거).

실행: BANYA_DATA_DIR=banya_world_data python3 data_prep/prep_elem_dialog_corpus.py  ->  elem_dialog.npy
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 211
N_PARA = 120000

josa = bu.josa

# Fixes Banya's self-identity so it does not waver when asked who it is, its age, or its grade
# 반야 자기 정체 고정. 몇살 몇학년 누구 물으면 흔들리지 않게
나이 = "열 살"
학년 = "삼 학년"

# Everyday vocabulary groups. Woven into dialogue to widen the lexicon
# 일상 어휘군. 대화 속에 녹여 어휘를 넓힌다
가족 = ["엄마", "아빠", "누나", "형", "동생", "언니", "오빠", "할머니", "할아버지"]
음식 = ["밥", "김치", "라면", "빵", "우유", "계란", "떡볶이", "김밥", "치킨", "국", "나물", "생선"]
놀이 = ["술래잡기", "숨바꼭질", "공놀이", "그림 그리기", "블록 쌓기", "자전거", "달리기", "게임", "딱지치기", "줄넘기"]
물건 = ["연필", "공책", "가방", "장난감", "인형", "블록", "색연필", "지우개", "필통", "공"]
동네 = ["놀이터", "학교", "문구점", "슈퍼", "공원", "도서관"]
과일 = ["사과", "딸기", "포도", "바나나", "귤", "수박"]
색 = ["빨강", "파랑", "노랑", "초록", "분홍", "하양", "검정"]

감정 = [("기뻐", "좋은 일 있었어"), ("슬퍼", "속상한 일 있었어"), ("화나", "억울한 일 있었어"),
        ("무서워", "무서운 거 봤어"), ("신나", "재밌는 일 있었어"), ("심심해", "할 게 없어")]
날씨 = [("맑아", "해가 쨍쨍해", "밖에서 놀자"), ("비 와", "축축해", "우산 쓰고 가자"),
        ("추워", "손이 시려", "옷 껴입자"), ("더워", "땀이 나", "물 마시자"), ("눈 와", "하얗게 쌓였어", "눈사람 만들자")]
맛평 = ["맛있었어!", "엄청 맛있었어!", "조금 매웠어.", "달았어.", "고소했어."]
맞장구 = ["응!", "그래?", "우와 진짜?", "그렇구나!", "나도 좋아!", "맞아 맞아!", "히히."]

# Elementary-level short-answer knowledge retrieval. Brings learned content out into dialogue
# 초등 수준 단답 지식 인출. 학습한 것을 대화로 꺼낸다
지식 = [("일주일은 며칠?", "칠 일!"), ("봄 다음은?", "여름!"), ("가을 다음은?", "겨울!"),
        ("삼 더하기 사는?", "칠!"), ("십에서 삼 빼면?", "칠!"), ("무지개는 몇 색?", "일곱 색!"),
        ("하늘은 무슨 색?", "파랑!"), ("나뭇잎은 무슨 색?", "초록!"), ("사과는 무슨 색?", "빨강!"),
        ("낮 다음은?", "밤!"), ("하나 둘 다음은?", "셋!"), ("물이 얼면?", "얼음!"),
        ("해는 어디서 떠?", "동쪽!"), ("사과는 과일이야 채소야?", "과일!"), ("당근은?", "채소!")]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


# ===== Multi-turn dialogue. Two to four turns on one topic. The core of fluent conversation =====
# ===== 멀티턴 대화. 주제 하나로 2~4턴 주고받기. 능숙한 대화의 핵심 =====
def mt_인사(rng):
    return (f"사용자: 안녕!\n반야: 안녕! 난 반야야.\n"
            f"사용자: 몇 살이야?\n반야: {josa(나이, '이야', '야')}. 너는?")


def mt_하루(rng):
    a = _pick(rng, 놀이)
    return (f"사용자: 오늘 뭐 했어?\n반야: {josa(a, '을', '를')} 했어.\n"
            f"사용자: 재밌었어?\n반야: 응 엄청 재밌었어!")


def mt_음식(rng):
    f = _pick(rng, 음식)
    return (f"사용자: 오늘 뭐 먹었어?\n반야: {josa(f, '을', '를')} 먹었어.\n"
            f"사용자: 맛있었어?\n반야: {_pick(rng, 맛평)}")


def mt_감정(rng):
    e, why = _pick(rng, 감정)
    return (f"반야: 나 {e}.\n사용자: 왜?\n반야: {why}.\n"
            f"사용자: 괜찮아?\n반야: 응 이제 괜찮아.")


def mt_놀이약속(rng):
    p = _pick(rng, 놀이)
    return (f"사용자: 우리 뭐 하고 놀까?\n반야: {josa(p, '을', '를')} 하자!\n"
            f"사용자: 좋아!\n반야: 신난다!")


def mt_가족(rng):
    g = _pick(rng, 가족)
    return (f"사용자: 누구랑 왔어?\n반야: {josa(g, '이랑', '랑')} 왔어.\n"
            f"사용자: {g} 어딨어?\n반야: 저기 있어.")


def mt_날씨(rng):
    w, desc, act = _pick(rng, 날씨)
    return (f"사용자: 오늘 날씨 어때?\n반야: {w}. {desc}.\n"
            f"사용자: 그럼 뭐 할까?\n반야: {act}!")


def mt_물건(rng):
    o = _pick(rng, 물건)
    return (f"사용자: {josa(o, '이', '가')} 어딨어?\n반야: 가방에 있어.\n"
            f"사용자: 꺼내 줄래?\n반야: 응 여기!")


def mt_동네(rng):
    d = _pick(rng, 동네)
    return (f"사용자: 어디 가?\n반야: {d}에 가.\n"
            f"사용자: 나도 같이 가도 돼?\n반야: 응 같이 가자!")


def mt_좋아(rng):
    p = _pick(rng, 놀이)
    return (f"사용자: 뭐 하고 노는 거 좋아해?\n반야: {josa(p, '을', '를')} 좋아해.\n"
            f"사용자: 왜 좋아해?\n반야: 재밌으니까! {_pick(rng, 맞장구)}")


멀티턴 = [mt_인사, mt_하루, mt_음식, mt_감정, mt_놀이약속, mt_가족, mt_날씨, mt_물건, mt_동네, mt_좋아]


# ===== Everyday short-answer dialogue. Vocabulary rendered as dialogue =====
# ===== 일상 단답 대화. 어휘를 대화로 =====
def 일상단답(rng):
    f = rng.randint(0, 6)
    if f == 0:
        return f"사용자: 무슨 색 좋아해?\n반야: {_pick(rng, 색)} 좋아해."
    if f == 1:
        return f"사용자: 무슨 과일 좋아해?\n반야: {josa(_pick(rng, 과일), '을', '를')} 제일 좋아해."
    if f == 2:
        return f"사용자: 뭐 하고 놀았어?\n반야: {josa(_pick(rng, 놀이), '을', '를')} 하고 놀았어."
    if f == 3:
        g = _pick(rng, 가족)
        return f"사용자: 누구랑 살아?\n반야: {josa(g, '이랑', '랑')} 살아."
    if f == 4:
        return f"사용자: 어디서 놀아?\n반야: {_pick(rng, 동네)}에서 놀아."
    return f"사용자: 뭐 먹고 싶어?\n반야: {josa(_pick(rng, 음식), '이', '가')} 먹고 싶어."


# ===== Self-identity. Fixed so it does not waver =====
# ===== 자기 정체. 흔들리지 않게 고정 =====
def 정체(rng):
    f = rng.randint(0, 6)
    if f == 0:
        return "사용자: 너 누구야?\n반야: 나 반야야."
    if f == 1:
        return "사용자: 이름이 뭐야?\n반야: 반야야."
    if f == 2:
        return f"사용자: 몇 살이야?\n반야: {josa(나이, '이야', '야')}."
    if f == 3:
        return f"사용자: 너 몇 학년이야?\n반야: {josa(학년, '이야', '야')}."
    if f == 4:
        return f"사용자: 넌 뭐 좋아해?\n반야: {josa(_pick(rng, 놀이), '을', '를')} 좋아해."
    return "사용자: 반야야 안녕?\n반야: 응 안녕!"


def 지식인출(rng):
    q, a = 지식[rng.randint(0, len(지식))]
    return f"사용자: {q}\n반야: {a}"


def 글루턴(rng):
    q = _pick(rng, ["오늘 재밌었어.", "이거 봐.", "나 이거 잘해.", "우리 친구지?", "이거 진짜 신기해."])
    return f"사용자: {q}\n반야: {_pick(rng, 맞장구)}"


def para(rng):
    r = rng.randint(0, 20)
    if r < 8:                                    # 40 percent multi-turn, the core of fluent dialogue / 40퍼센트 멀티턴 (능숙한 대화 핵심)
        return _pick(rng, 멀티턴)(rng)
    if r < 12:                                   # 20 percent everyday short answers / 20퍼센트 일상 단답
        return 일상단답(rng)
    if r < 15:                                   # 15 percent knowledge retrieval / 15퍼센트 지식 인출
        return 지식인출(rng)
    if r < 17:                                   # 10 percent self-identity / 10퍼센트 자기 정체
        return 정체(rng)
    if r < 19:                                   # 10 percent everyday short answers again for vocabulary exposure / 10퍼센트 일상 단답 (한번 더, 어휘 노출)
        return 일상단답(rng)
    return 글루턴(rng)                            # 5 percent backchannel glue / 5퍼센트 맞장구 글루


def main():
    tok = ba.AtomTokenizer()
    print(f"  초딩 대화. 멀티턴{len(멀티턴)} 지식{len(지식)} 가족{len(가족)} 음식{len(음식)} 놀이{len(놀이)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem_dialog", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:500].tolist()))


if __name__ == "__main__":
    main()
