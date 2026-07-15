# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_elem_corpus.py — Elementary-stage corpus (ages 8 to 13, civilization axis: farming, fishing, forestry)

The rung after baby senses on the developmental ladder. School life is laid on top of a world of
growing, catching, and harvesting. What a human elementary student actually experiences is filled
in by growth analogy.
  School: teacher, friend, class, homework, exam, school lunch, blackboard, book, pencil, bag
  Arithmetic: addition, subtraction, multiplication, division (0 to 100)
  Writing: reading, writing, dictation
  Time: weekdays, seasons, calendar, clock, morning and evening
  Money: allowance, price, change
  Farming, fishing, forestry: seed, field, paddy, rice plant, fish, net, angling, cow, chicken, pig, tree, forest
  Hobbies: drawing, sports, games

Language: complete paragraphs of 3 to 6 sentences, basic logic (because, so, but), sequence
(first, next, last), comparison (more, most). Recall patterns (what did we learn, what was the
number, what did we see) are also mixed in.
Friend names for recall are random syllable combinations, so they cannot be answered from
memorization and the preceding sentence must be retrieved.

Farming, fishing, and forestry vocabulary is drawn from the fish, crop, and tree categories of
banya_world nouns_list.json. School, time, and money vocabulary is specific to this stage and is
curated here.

Grammar: particles only through bu.josa/jo/jn. Adjective conjugations are stored fixed in tuples
(no slicing).

Run: python3 data_prep/prep_elem_corpus.py   ->  data/elem.npy (int32)
Note: developmental stage 3 (elementary). A stepping stone to the next stage. Do not exceed this
stage's complexity.

초등 단계 말뭉치 (8~13세, 문명축: 농경 어업 임업)

발달 사다리에서 아기 감각 다음 칸. 기르고 잡고 거두는 세계 위에 학교 생활을 얹는다.
사람 초등학생이 실제로 겪는 것을 성장 유추로 채운다.
  학교: 선생님 친구 반 숙제 시험 급식 칠판 책 연필 가방
  셈: 덧셈 뺄셈 곱셈 나눗셈 (0~100)
  글: 읽기 쓰기 받아쓰기
  시간: 요일 계절 달력 시계 아침저녁
  돈: 용돈 가격 거스름돈
  농사 고기잡이 임업: 씨앗 밭 논 벼 물고기 그물 낚시 소 닭 돼지 나무 숲
  취미: 그림 운동 게임

언어: 완결 문단 3~6문장, 기초 논리(왜냐하면 그래서 하지만), 순서(먼저 다음 마지막),
비교(더 가장). 회상 문형(뭐 배웠지, 몇이라고 했지, 무엇을 봤지)도 일부 섞는다.
회상용 친구 이름은 음절 무작위 조합이라 외워서 못 풀고 앞 문장을 되찾아야 답이 나온다.

농경 어업 임업 어휘는 banya_world nouns_list.json 의 어류 작물 나무 범주에서 끌어온다.
학교 시간 돈 어휘는 이 단계 전용이라 여기서 큐레이션한다.

문법: 조사는 전부 bu.josa/jo/jn 로만. 형용사 활용형은 튜플에 고정해 저장(슬라이스 금지).

실행: python3 data_prep/prep_elem_corpus.py   ->  data/elem.npy (int32)
주의: 발달 3단계(초등). 다음 단계의 발판. 이 단계 복잡도를 넘지 말 것.
"""
import os
import sys
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 53
N_PARA = 85000
ONTO = os.path.join(_ROOT, "banya_world", "onto")

josa = bu.josa
jo = bu.jo
jn = bu.jn
h = bu.h
SYL = list(bu.SYL_전체)

# Vocabulary specific to this stage. School, time, and money are absent from the ontology, so they are hand-picked
# 이 단계 전용 어휘. 학교 시간 돈은 온톨로지에 없어 손으로 고른다
실명 = ["철수", "영희", "민수", "지영", "준호", "수진", "동수", "미영", "보람", "다은", "은지", "성호"]
과목 = ["국어", "셈", "그림", "노래", "체육", "받아쓰기", "읽기", "쓰기"]
셈주제 = ["덧셈", "뺄셈", "곱셈", "나눗셈"]
요일 = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
계절 = ["봄", "여름", "가을", "겨울"]
학용품 = [("공책", "권"), ("연필", "자루"), ("지우개", "개"), ("색연필", "자루"), ("책", "권"), ("가방", "개")]
군것질 = ["사탕", "빵", "우유", "과자", "떡", "김밥"]
가축사실 = [("닭", "알을 낳는다"), ("소", "밭을 간다"), ("오리", "헤엄친다"),
            ("돼지", "먹이를 잘 먹는다"), ("염소", "풀을 뜯는다"), ("토끼", "당근을 좋아한다"),
            ("말", "빨리 달린다"), ("양", "털이 많다"), ("개", "집을 지킨다"), ("거위", "꽥꽥 운다")]
# Adjective comparison. (the former exceeds the latter, predicative form, attributive form). Conjugated forms are stored fixed instead of sliced
# 형용사 비교. (앞이 뒤보다 그러함, 서술형, 관형형). 활용형을 슬라이스하지 않고 고정해 둔다
비교사실 = [("소", "닭", "크다", "큰"), ("코끼리", "토끼", "크다", "큰"), ("나무", "풀", "크다", "큰"),
            ("산", "언덕", "높다", "높은"), ("기차", "소", "빠르다", "빠른"), ("말", "거북", "빠르다", "빠른"),
            ("돌", "깃털", "무겁다", "무거운"), ("바다", "연못", "넓다", "넓은"), ("코끼리", "개미", "크다", "큰"),
            ("여름", "봄", "덥다", "더운"), ("겨울", "가을", "춥다", "추운"), ("어른", "아이", "크다", "큰")]

# Sentence pool for narrating a school day. Each sentence is complete in itself, so concatenation does not break grammar
# 학교 하루 서술용 문장 풀. 각 문장이 그 자체로 완결이라 이어붙여도 문법이 깨지지 않는다
아침문 = ["아침 일찍 학교에 갔다.", "책가방을 메고 학교에 갔다.", "친구와 함께 학교에 갔다."]
수업문 = ["국어 시간에 책을 읽었다.", "셈 시간에 덧셈을 배웠다.", "받아쓰기 시험을 봤다.",
          "그림 시간에 그림을 그렸다.", "체육 시간에 달리기를 했다.", "음악 시간에 노래를 불렀다.",
          "칠판에 있는 글씨를 공책에 옮겨 적었다.", "곱셈을 새로 배웠다."]
점심문 = ["점심에는 급식을 먹었다.", "친구들과 급식을 맛있게 먹었다.", "급식으로 나온 반찬이 맛있었다."]
방과문 = ["수업이 끝나고 친구와 놀았다.", "집에 와서 숙제를 했다.", "운동장에서 공을 찼다.",
          "친구 집에 놀러 갔다."]
감상문 = ["참 즐거운 하루였다.", "오늘도 많이 배웠다.", "내일도 학교에 가고 싶다.",
          "조금 피곤했지만 뿌듯했다.", "기분이 좋았다."]
활동구 = ["세수를 한다", "밥을 먹는다", "이를 닦는다", "가방을 챙긴다", "숙제를 한다", "책을 읽는다"]


def load_lex(tok):
    def ok(s):
        try:
            ids = list(tok.encode(s))
            return len(ids) and max(ids) < tok.vocab
        except Exception:
            return False
    nl = json.load(open(os.path.join(ONTO, "nouns_list.json"), encoding="utf-8"))

    def pick(cat):
        return [w for w in nl.get(cat, []) if ok(w) and 1 <= len(w) <= 3]
    물고기 = pick("어류")
    작물 = pick("열매 버섯 작물")
    나무 = pick("나무")
    return 물고기, 작물, 나무


def name(rng):
    return bu.make_name(rng, SYL)


def 덧셈(rng):
    a = rng.randint(0, 60)
    b = rng.randint(0, 100 - a)
    c = a + b
    if rng.randint(0, 2):
        return f"{a} + {b} = {c}"
    return f"{h(a)} 더하기 {jo(h(b), '은', '는')} {h(c)}."


def 뺄셈(rng):
    a = rng.randint(0, 100)
    b = rng.randint(0, a + 1)
    c = a - b
    if rng.randint(0, 2):
        return f"{a} - {b} = {c}"
    return f"{h(a)} 빼기 {jo(h(b), '은', '는')} {h(c)}."


def 곱셈(rng):
    a = rng.randint(2, 10)
    b = rng.randint(1, 10)
    c = a * b
    if rng.randint(0, 2):
        return f"{a} * {b} = {c}"
    return f"{h(a)} 곱하기 {jo(h(b), '은', '는')} {h(c)}."


def 나눗셈(rng):
    a = rng.randint(2, 10)
    b = rng.randint(2, 10)
    c = a * b
    if rng.randint(0, 2):
        return f"{c} ÷ {b} = {a}"
    return f"{h(c)} 나누기 {jo(h(b), '은', '는')} {h(a)}."


def 셈회상(rng):
    f = rng.randint(0, 3)
    if f == 0:
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        c = a + b
        return f"{a} 더하기 {jn(b, '은', '는')} {jn(c, '이야', '야')}. 답이 몇이라고 했지? {c}."
    if f == 1:
        a = rng.randint(2, 10)
        b = rng.randint(2, 10)
        return f"곱셈을 배웠어. {a} 곱하기 {jn(b, '이', '가')} 뭐였지? {a * b}."
    t = 셈주제[rng.randint(0, len(셈주제))]
    return f"오늘 셈 시간에 {josa(t, '을', '를')} 배웠다. 무엇을 배웠다고 했지? {t}."


def 비교수(rng):
    f = rng.randint(0, 3)
    if f == 0:
        a = rng.randint(0, 100)
        b = rng.randint(0, 100)
        while b == a:
            b = rng.randint(0, 100)
        big = max(a, b)
        small = min(a, b)
        if rng.randint(0, 2):
            return f"{jn(a, '과', '와')} {b} 중 더 큰 수는? {big}."
        return f"{jn(a, '과', '와')} {b} 중 더 작은 수는? {small}."
    if f == 1:
        xs = []
        while len(xs) < 3:
            v = rng.randint(0, 100)
            if v not in xs:
                xs.append(v)
        run = ", ".join(str(x) for x in xs)
        if rng.randint(0, 2):
            return f"{run} 중 가장 큰 수는? {max(xs)}."
        return f"{run} 중 가장 작은 수는? {min(xs)}."
    nm1 = 실명[rng.randint(0, len(실명))]
    nm2 = 실명[rng.randint(0, len(실명))]
    while nm2 == nm1:
        nm2 = 실명[rng.randint(0, len(실명))]
    obj = 군것질[rng.randint(0, len(군것질))]
    a = rng.randint(1, 10)
    b = rng.randint(1, 10)
    while b == a:
        b = rng.randint(1, 10)
    big = nm1 if a > b else nm2
    return (f"{josa(nm1, '은', '는')} {obj} {a}개, {josa(nm2, '은', '는')} {b}개를 가졌다. "
            f"누가 더 많이 가졌나? {big}.")


def 비교형용사(rng):
    a, b, 서술, 관형 = 비교사실[rng.randint(0, len(비교사실))]
    if rng.randint(0, 2):
        return f"{josa(a, '과', '와')} {b} 중 더 {관형} 것은? {a}."
    return f"{josa(a, '은', '는')} {b}보다 {서술}."


def 요일form(rng):
    i = rng.randint(0, 7)
    d = 요일[i]
    f = rng.randint(0, 3)
    if f == 0:
        return f"오늘이 {d}이면 내일은? {요일[(i + 1) % 7]}."
    if f == 1:
        return f"오늘이 {d}이면 어제는? {요일[(i - 1) % 7]}."
    return f"한 주는 {요일[0]}부터 {요일[6]}까지다. 모두 며칠? 7일."


def 계절form(rng):
    f = rng.randint(0, 4)
    if f == 0:
        i = rng.randint(0, 4)
        return f"{계절[i]} 다음은? {계절[(i + 1) % 4]}."
    if f == 1:
        return "가장 더운 계절은? 여름."
    if f == 2:
        return "가장 추운 계절은? 겨울."
    m = rng.randint(1, 12)
    nxt = m % 12 + 1
    return f"{m}월 다음은? {nxt}월."


def 시계form(rng):
    hh = rng.randint(1, 13)
    if rng.randint(0, 2):
        return f"짧은바늘이 {jn(hh, '을', '를')} 가리키면 몇 시? {hh}시."
    return f"지금 몇 시야? {hh}시야."


def 돈form(rng):
    obj = 군것질[rng.randint(0, len(군것질))]
    f = rng.randint(0, 3)
    if f == 0:
        값 = rng.randint(1, 10) * 100
        낸돈 = 1000
        거스름 = 낸돈 - 값
        return (f"{낸돈}원을 내고 {값}원짜리 {josa(obj, '을', '를')} 샀다. "
                f"거스름돈은 얼마인가? {거스름}원.")
    if f == 1:
        용돈 = rng.randint(1, 5) * 500
        값 = rng.randint(1, 용돈 // 100) * 100
        남음 = 용돈 - 값
        return (f"용돈은 {용돈}원이다. {값}원짜리 {josa(obj, '을', '를')} 사면 얼마가 남나? "
                f"{남음}원.")
    a = rng.randint(1, 5) * 100
    b = rng.randint(1, 5) * 100
    return f"{a}원짜리 하나와 {b}원짜리 하나를 사면 모두 얼마? {a + b}원."


def 농사form(rng, 작물):
    c = 작물[rng.randint(0, len(작물))]
    f = rng.randint(0, 3)
    if f == 0:
        return (f"봄에 밭에 {josa(c, '을', '를')} 심었다. 여름에 물을 주고 가꾸었다. "
                f"가을에 {josa(c, '을', '를')} 거두었다.")
    if f == 1:
        return f"농부가 논에 벼를 심는다. 벼가 자라면 쌀이 된다. 가을에 벼를 거둔다."
    return f"{josa(c, '은', '는')} 밭에서 자란다. 어디에서 자라나? 밭."


def 어업form(rng, 물고기):
    fish = 물고기[rng.randint(0, len(물고기))]
    f = rng.randint(0, 3)
    if f == 0:
        return (f"어부가 바다에 나갔다. 그물로 {josa(fish, '을', '를')} 많이 잡았다. "
                f"오늘은 물고기를 많이 잡아 기뻤다.")
    if f == 1:
        return f"강에서 낚시를 했다. {josa(fish, '을', '를')} 한 마리 낚았다. 참 신기했다."
    return f"물고기는 무엇으로 잡나? 그물과 낚시."


def 축산form(rng):
    nm = 실명[rng.randint(0, len(실명))]
    a, act = 가축사실[rng.randint(0, len(가축사실))]
    if rng.randint(0, 2):
        return (f"{josa(nm, '은', '는')} {josa(a, '을', '를')} 기른다. {josa(a, '은', '는')} {act}. "
                f"아침마다 먹이를 준다.")
    return f"{josa(a, '은', '는')} 어떻게 하나? {act}."


def 임업form(rng, 나무):
    t = 나무[rng.randint(0, len(나무))]
    if rng.randint(0, 2):
        return (f"산에 {josa(t, '을', '를')} 심었다. {josa(t, '이', '가')} 무럭무럭 자랐다. "
                f"여러 나무가 모여 숲이 되었다.")
    return f"나무를 많이 심으면 무엇이 되나? 숲."


def 학교하루(rng):
    nm = 실명[rng.randint(0, len(실명))]
    s1 = 아침문[rng.randint(0, len(아침문))]
    s2 = 수업문[rng.randint(0, len(수업문))]
    s3 = 점심문[rng.randint(0, len(점심문))]
    s4 = 방과문[rng.randint(0, len(방과문))]
    s5 = 감상문[rng.randint(0, len(감상문))]
    parts = [f"오늘 {josa(nm, '은', '는')} {s1}", s2, s3, s4, s5]
    k = rng.randint(3, 6)
    return " ".join(parts[:k])


def 순서form(rng):
    acts = list(활동구)
    rng.shuffle(acts)
    a, b, c = acts[0], acts[1], acts[2]
    return f"아침에는 먼저 {a}. 다음에 {b}. 마지막으로 {c}."


def 논리form(rng):
    f = rng.randint(0, 3)
    if f == 0:
        return "비가 왔다. 그래서 우산을 썼다. 하지만 신발이 젖었다."
    if f == 1:
        return "시험에서 백 점을 받았다. 왜냐하면 열심히 공부했기 때문이다. 정말 기뻤다."
    return "친구가 넘어졌다. 그래서 얼른 일으켜 주었다. 친구가 고맙다고 했다."


def 사건form(rng):
    nm = 실명[rng.randint(0, len(실명))]
    f = rng.randint(0, 5)
    if f == 0:
        s = rng.randint(6, 11) * 10
        if s >= 90:
            return f"어제 시험을 봤다. 열심히 공부해서 {s}점을 받았다. 참 기뻤다."
        return f"어제 시험을 봤다. {s}점을 받아 조금 아쉬웠다. 다음엔 더 잘하고 싶다."
    if f == 1:
        return "그림 대회에서 상을 받았다. 왜냐하면 정성껏 그렸기 때문이다. 정말 자랑스러웠다."
    if f == 2:
        return "봄에 소풍을 갔다. 김밥을 먹고 친구들과 신나게 놀았다. 즐거운 하루였다."
    if f == 3:
        obj = 군것질[rng.randint(0, len(군것질))]
        change = rng.randint(1, 5) * 100
        return (f"엄마가 심부름을 시켰다. 가게에서 {josa(obj, '을', '를')} 사 왔다. "
                f"거스름돈 {change}원을 돌려드렸다.")
    return f"{josa(nm, '과', '와')} 다퉜다. 하지만 곧 화해했다. 다시 사이좋게 놀았다."


def 규칙form(rng):
    f = rng.randint(0, 3)
    if f == 0:
        return "약속을 어기면 어떻게 되나? 벌을 받는다."
    if f == 1:
        return "차례를 잘 지키면 어떻게 되나? 칭찬과 상을 받는다."
    return "친구와는 어떻게 지내야 하나? 사이좋게 지내야 한다."


def 글form(rng):
    f = rng.randint(0, 3)
    if f == 0:
        return "받아쓰기 시험을 봤다. 한 글자도 안 틀리고 다 맞았다. 참 뿌듯했다."
    if f == 1:
        return "책을 소리 내어 읽었다. 모르는 낱말은 선생님께 여쭤봤다. 새 낱말을 많이 배웠다."
    return "공책에 오늘 배운 것을 또박또박 썼다. 글씨가 점점 예뻐진다. 기분이 좋았다."


def 회상form(rng):
    f = rng.randint(0, 4)
    if f == 0:
        nm = name(rng)
        return f"오늘 새 친구를 만났어. 이름은 {josa(nm, '이야', '야')}. 친구 이름이 뭐라고 했지? {josa(nm, '이야', '야')}."
    if f == 1:
        nm = name(rng)
        obj = 군것질[rng.randint(0, len(군것질))]
        n = rng.randint(1, 20)
        return (f"{josa(nm, '이', '가')} {obj} {n}개를 가져왔다. "
                f"몇 개 가져왔다고 했지? {n}개.")
    if f == 2:
        t = 과목[rng.randint(0, len(과목))]
        return f"오늘 학교에서 {josa(t, '을', '를')} 배웠어. 뭐 배웠지? {t}."
    if f == 3:
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        return f"{a} 더하기 {jn(b, '은', '는')} {jn(a + b, '이야', '야')}. 답이 몇이었지? {a + b}."
    acts = list(활동구)
    rng.shuffle(acts)
    return f"먼저 {acts[0]}. 다음에 {acts[1]}. 먼저 뭐 했다고 했지? {acts[0]}."


def make_para(rng, lex):
    물고기, 작물, 나무 = lex
    fns = [
        덧셈, 뺄셈, 곱셈, 나눗셈, 셈회상,
        비교수, 비교형용사, 요일form, 계절form, 시계form,
        돈form, lambda r: 농사form(r, 작물), lambda r: 어업form(r, 물고기),
        축산form, lambda r: 임업form(r, 나무), 학교하루, 순서form,
        논리form, 사건form, 규칙form, 글form, 회상form,
    ]
    return fns[rng.randint(0, len(fns))](rng)


def wrap_대화(p, rng):
    # One quarter gets the dialogue frame. Multi-sentence paragraphs and text that is already dialogue are not wrapped
    # 4분의 1은 대화 틀. 여러 문장 문단이나 이미 대화인 것은 감싸지 않는다
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p and "사용자" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    for s in [s for s in SYL if s not in tok.stoi]:
        SYL.remove(s)
    lex = load_lex(tok)
    물고기, 작물, 나무 = lex
    print(f"  물고기{len(물고기)} 작물{len(작물)} 나무{len(나무)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("elem", lambda r: make_para(r, lex), N_PARA, rng, tok, wrap=wrap_대화)
    print("표본 디코드:")
    print(tok.decode(arr[:400].tolist()))


if __name__ == "__main__":
    main()
