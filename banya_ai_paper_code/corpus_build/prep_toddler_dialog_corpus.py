# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_dialog_corpus.py — toddler dialogue (retrieval channel for learning: short-answer retrieval + explanation retrieval)

Learning is storage and dialogue is retrieval. Without a retrieval channel, learning becomes dormant knowledge.
This corpus draws out what has been learned through the sense, space, baby, and toddler stages as question-and-answer dialogue. The answers are exactly the learned content.
There are two branches: short-answer retrieval (what color, is it hot) and explanation retrieval (tell me about the sparrow, how is the world).
Explanations are not invented either; Banya weaves them by combining learned expressions, which prevents ghost knowledge from arising.
Banya is a child. Speaker labels are unified as 사용자 (user) versus 반야 (Banya) for inference-trigger consistency. The relationship is kept in the content of Banya's lines.

Run: python3 data_prep/prep_toddler_dialog_corpus.py   ->  data/toddler_dialog.npy (set the output folder with BANYA_DATA_DIR)

prep_toddler_dialog_corpus.py — 유딩 대화 (학습 인출 통로: 단답 인출 + 설명 인출)

학습은 저장이고 대화는 인출이다. 인출 통로가 없으면 학습이 잠자는 지식이 된다.
이 말뭉치는 감각 공간 아기 유딩까지 학습한 것을 질문 대 답 대화로 꺼낸다. 답은 학습한 그 내용이다.
두 갈래다. 단답 인출(무슨 색 뜨거워)과 설명 인출(참새 얘기해줘 세상 어때).
설명도 지어내지 않고 학습한 표현들을 조합해 반야가 엮어 설명한다. 그래야 유령 지식이 안 생긴다.
반야는 아이다. 라벨은 사용자 대 반야로 통일한다(추론 트리거 일관성). 관계는 반야 대사 내용에 남긴다.

실행: python3 data_prep/prep_toddler_dialog_corpus.py   ->  data/toddler_dialog.npy (BANYA_DATA_DIR 로 폴더 지정)
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
import toddler_expr as T
import toddler_affect as J
import toddler_questions as Q
import toddler_events as E
import toddler_seed_expr as G

SEED = 97
N_PARA = 160000

josa = bu.josa


def 로조사(w):
    ch = w[-1]
    if not ("가" <= ch <= "힣"):
        return w + "로"
    jong = (ord(ch) - 0xAC00) % 28
    return w + "로" if jong == 0 or jong == 8 else w + "으로"


# ===== Short-answer retrieval templates. Questions from several angles versus answers, the learned content =====
# ===== 단답 인출 템플릿. 질문 여러 각도 대 답(학습한 내용) =====
감각인출 = [
    (["만지면 어때?", "이거 만지면 어때?", "뜨거워 차가워?", "만져 봐. 어때?"],
     ["뜨거워. 앗 뜨거.", "차가워.", "따뜻해.", "아파.", "부드러워.", "딱딱해.", "매끈해."]),
    (["무슨 색?", "이거 무슨 색?", "색이 뭐야?", "이거 무슨 색이야?"],
     ["빨강.", "노랑.", "파랑.", "하얘.", "초록.", "까매."]),
    (["밝아 어두워?", "밝아?", "어두워?"], ["밝아.", "어두워.", "눈부셔.", "깜깜해."]),
    (["무슨 소리 나?", "시끄러워 조용해?", "들려?"], ["시끄러워.", "조용해.", "큰 소리 나.", "쿵 소리 나.", "응 들려."]),
    (["무슨 맛?", "이거 무슨 맛이야?", "달아 써?", "맛있어?"], ["달아.", "써.", "시어.", "짜.", "맛있어.", "맛없어."]),
    (["냄새 어때?", "무슨 냄새야?"], ["좋은 냄새 나.", "향기로워.", "냄새나.", "구려."]),
]
공간인출 = [
    (["나보다 커 작아?", "저거 커?", "나보다 커?"], ["나보다 커. 무서워.", "나보다 작아. 안 무서워.", "나만 해."]),
    (["가까워 멀어?", "가까워?", "멀어?"], ["가까워.", "멀어.", "코앞이야.", "저 멀리 있어."]),
    (["있어 없어?", "있어?"], ["있어.", "없어."]),
    (["어디 있어?", "어느 쪽이야?"], ["앞에 있어.", "뒤에 있어.", "위에 있어.", "옆에 있어."]),
]
매핑인출 = [
    (["가까이 오면?", "가까이 가면?"], ["뜨거워져.", "더 밝아.", "더 시끄러워.", "더 커 보여."]),
    (["멀어지면?"], ["시원해져.", "어두워져.", "조용해져.", "작아 보여."]),
]
몸구어 = {"눈": "봐.", "귀": "들어.", "코": "맡아.", "입": "먹어.", "손": "잡아.", "발": "걸어.", "혀": "맛봐."}
귀속인출 = [("뜨거운 건 뭘로 알아?", "손으로 알아."), ("밝은 건 뭘로 봐?", "눈으로 봐."),
            ("소리는 뭘로 들어?", "귀로 들어."), ("냄새는 뭘로 맡아?", "코로 맡아."), ("맛은 뭘로 알아?", "혀로 맛봐.")]
상태인출 = [
    (["왜 그래?", "왜 울어?"], ["배고파.", "졸려.", "아파.", "심심해."]),
    (["어디 아파?"], ["여기 아파.", "배 아파.", "머리 아파.", "무릎 아파."]),
    (["졸려?", "배고파?"], ["응 졸려.", "아니 안 졸려.", "응 배고파.", "아니 배불러."]),
]
지식인출 = [("봄 다음은?", "여름."), ("여름 다음은?", "가을."), ("가을 다음은?", "겨울."),
            ("사과는 과일이야 채소야?", "과일!"), ("당근은 과일이야 채소야?", "채소!"),
            ("강아지는 동물이야?", "응 동물이야!"), ("나무는 동물이야 식물이야?", "식물이야!"),
            ("하늘은 무슨 색?", "파랑!"), ("풀은 무슨 색?", "초록!"), ("하나 둘 다음은?", "셋!"),
            ("동그라미는 어떻게 생겼어?", "동글동글해!"), ("네모는 어떻게 생겼어?", "각졌어!")]
논리인출 = [("뭐가 더 커?", "이게 더 커!"), ("뭐가 더 작아?", "이게 더 작아!"),
            ("같아 달라?", "둘 다 똑같아!"), ("이거랑 저거 같아?", "아니 달라!"),
            ("왜 울어?", "아파서!"), ("왜 안 먹어?", "매워서!"), ("왜 자?", "졸려서!"),
            ("뭐가 더 빨라?", "내가 더 빨라!"), ("이게 더 무거워?", "응 더 무거워!")]
자아문 = ["누가 할래?", "이거 누구 거?", "이거 할래?", "네가 할 거야?", "이거 하기 싫어?"]
감정pos문 = ["기분 어때?", "재밌어?", "오늘 좋았어?", "신나?"]
감정neg문 = ["왜 울어?", "왜 삐졌어?", "무슨 일이야?", "왜 화났어?"]
경험문 = ["오늘 뭐 했어?", "유치원에서 뭐 했어?", "뭐 하고 놀았어?", "오늘 재밌었어?"]
반야naming = ["이게 뭐야?", "저건 뭐야?", "이거 뭐야?", "이름이 뭐야?"]

# ===== Explanation retrieval. Banya explains one topic by weaving together several learned expressions =====
# ===== 설명 인출. 주제 하나로 학습한 표현들을 여러 개 엮어 반야가 설명 =====
대상설명문 = ["{t} 어때?", "{t} 얘기해줘", "{t}에 대해 말해줘", "{t} 뭐 알아?", "{t} 설명해줘", "{t} 어떤 거야?"]
경험설명문 = ["오늘 뭐 했어? 다 말해줘", "유치원 얘기해줘", "오늘 있었던 일 말해줘", "뭐 하고 놀았는지 말해줘", "오늘 하루 어땠어?"]
세상설명문 = ["세상 어때?", "하늘 얘기해줘", "세상 얘기해줘", "궁금한 거 말해줘", "밖에 뭐가 있어?"]
지식설명문 = ["아는 거 말해줘", "뭐 배웠어? 말해줘", "다 말해봐", "네가 아는 거 뭐야?"]

다턴 = [
    "사용자: 졸리지? 자자.\n반야: 싫어! 더 놀래!",
    "반야: 이거 살래!\n사용자: 안 돼.\n반야: 왜 안 돼! 사줘!",
    "반야: 엄마 어디 가?\n사용자: 금방 올게.\n반야: 가지 마! 같이 가!",
    "반야: 나 슬퍼.\n사용자: 왜?\n반야: 장난감 부러졌어.\n사용자: 괜찮아 안아줄게.\n반야: 히잉.",
    "사용자: 오늘 재밌었어?\n반야: 너무너무 신났어! 또 가고 싶어!",
    "사용자: 같이 놀자!\n반야: 응 좋아! 뭐 하고 놀까?",
    "사용자: 그만! 위험해!\n반야: 싫어! 더 할래!",
]


def _pick(rng, xs):
    return xs[rng.randint(0, len(xs))]


def _pickn(rng, xs, k):                          # pick k items without duplicates, in order / 중복 없이 k개 뽑아 순서대로
    picks = []
    tries = 0
    while len(picks) < min(k, len(xs)) and tries < k * 4:
        e = xs[rng.randint(0, len(xs))]
        if e not in picks:
            picks.append(e)
        tries += 1
    return picks


def _qa(rng, table):
    qs, ans = table[rng.randint(0, len(table))]
    return _pick(rng, qs), _pick(rng, ans)


def 설명(rng, lex):
    seeds, 자아답, 감정pos, 감정neg, 경험답, 반야반응, G씨앗, 세상답, 지식답 = lex
    typ = rng.randint(0, 4)
    if typ == 0:                                 # object explanation (seed branching: object plus senses) / 대상 설명 (가지치기: 대상 더하기 감각)
        s = _pick(rng, G씨앗)
        ans = ". ".join(_pickn(rng, G.표현[s], rng.randint(3, 7))) + "."
        return f"사용자: {_pick(rng, 대상설명문).replace('{t}', s)}\n반야: {ans}"
    if typ == 1:                                 # experience explanation (several events woven into a narrative) / 경험 설명 (사건 여러 개 엮어 서사)
        ans = " ".join(_pickn(rng, 경험답, rng.randint(3, 6)))
        return f"사용자: {_pick(rng, 경험설명문)}\n반야: {ans}"
    if typ == 2:                                 # world explanation (metaphysics: the world is big) / 세상 설명 (형이상학: 세상은 크다)
        ans = " ".join(_pickn(rng, 세상답, rng.randint(3, 5)))
        return f"사용자: {_pick(rng, 세상설명문)}\n반야: {ans}"
    # knowledge explanation (colors, shapes, numbers, seasons)
    # 지식 설명 (색 모양 숫자 계절)
    ans = " ".join(_pickn(rng, 지식답, rng.randint(3, 5)))
    return f"사용자: {_pick(rng, 지식설명문)}\n반야: {ans}"


def para(rng, lex):
    seeds, 자아답, 감정pos, 감정neg, 경험답, 반야반응 = lex[:6]
    상대 = "사용자"

    if rng.randint(0, 10) < 4:                   # 40 percent explanation retrieval (bulk) / 40퍼센트 설명 인출 (대량)
        return 설명(rng, lex)

    # 60 percent short-answer retrieval
    # 60퍼센트 단답 인출
    f = rng.randint(0, 14)
    if f == 0:
        q, a = _qa(rng, 감각인출)
        return f"{상대}: {q}\n반야: {a}"
    if f == 1:
        q, a = _qa(rng, 공간인출)
        return f"{상대}: {q}\n반야: {a}"
    if f == 2:
        q, a = _qa(rng, 매핑인출)
        return f"{상대}: {q}\n반야: {a}"
    if f == 3:
        부위 = _pick(rng, list(몸구어))
        return f"{상대}: {로조사(부위)} 뭐 해?\n반야: {몸구어[부위]}"
    if f == 4:
        q, a = 귀속인출[rng.randint(0, len(귀속인출))]
        return f"{상대}: {q}\n반야: {a}"
    if f == 5:
        q, a = _qa(rng, 상태인출)
        return f"{상대}: {q}\n반야: {a}"
    if f == 6:
        q, a = 지식인출[rng.randint(0, len(지식인출))]
        return f"{상대}: {q}\n반야: {a}"
    if f == 7:
        q, a = 논리인출[rng.randint(0, len(논리인출))]
        return f"{상대}: {q}\n반야: {a}"
    if f == 8:
        return f"{상대}: {_pick(rng, 자아문)}\n반야: {_pick(rng, 자아답)}"
    if f == 9:
        if rng.randint(0, 2):
            return f"{상대}: {_pick(rng, 감정pos문)}\n반야: {_pick(rng, 감정pos)}"
        return f"{상대}: {_pick(rng, 감정neg문)}\n반야: {_pick(rng, 감정neg)}"
    if f == 10:
        a = " ".join(_pickn(rng, 경험답, rng.randint(1, 3)))
        return f"{상대}: {_pick(rng, 경험문)}\n반야: {a}"
    if f == 11:                                  # Banya asks, learns, and reacts (question explosion) / 반야가 묻고 배우고 반응 (질문 폭발)
        seed = _pick(rng, seeds)
        return f"반야: {_pick(rng, 반야naming)}\n{상대}: {josa(seed, '이야', '야')}.\n반야: {_pick(rng, 반야반응)}"
    if f == 12:
        return _pick(rng, 다턴)
    q, a = 지식인출[rng.randint(0, len(지식인출))] if rng.randint(0, 2) else 논리인출[rng.randint(0, len(논리인출))]
    return f"{상대}: {q}\n반야: {a}"


def main():
    tok = ba.AtomTokenizer()
    seeds = [t[0] for t in S.동물씨앗 if S.in_vocab(t[0], tok)]
    seeds += [w for w in S.자연씨앗 if S.in_vocab(w, tok)]
    자아답 = J.base["고집떼쓰기"] + J.base["강한소유욕"] + J.base["무판단즉각욕구"]
    감정pos = J.base["천국적즐거움"] + J.base["엄마보호행복"] + T.표현["사회감정"]
    감정neg = J.base["극단감정"]
    경험답 = []
    for v in E.경험.values():
        경험답 += v
    반야반응 = Q.학습["알아서즐거움"]
    G씨앗 = list(G.표현.keys())
    세상답 = E.정서["세상은크다"]
    지식답 = T.표현["toddler_learn"]
    lex = (seeds, 자아답, 감정pos, 감정neg, 경험답, 반야반응, G씨앗, 세상답, 지식답)
    print(f"  유딩 대화(인출). 단답 + 설명. 씨앗{len(seeds)} 가지치기씨앗{len(G씨앗)} 세상{len(세상답)} 지식{len(지식답)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_dialog", lambda r: para(r, lex), N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:420].tolist()))


if __name__ == "__main__":
    main()
