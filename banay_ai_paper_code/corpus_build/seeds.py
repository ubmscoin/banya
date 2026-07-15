# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""seeds.py — seed lexicon for developmental training (shared skeleton for the sensory/spatial/logic/infant/salient-event corpora)

Design principles (fixed by the user):
  - Seeds stay narrow (a few dozen). They are not widened. Instead the angles are widened and the seeds are repeated in volume.
  - Seeds are archetypes with strong symbolism. Elephant equals huge, ant equals tiny, tiger equals fierce.
    A symbolic archetype represents the extreme of an axis and thus becomes the reference point (anchor) of that axis. Rare words such as heron are excluded.
  - Two layers: contact archetypes (things the infant actually touches: dog, water, hand, soil) plus symbolic archetypes (picture-book concept representatives: elephant, tiger).
  - Primitive verbs are the root of thought. Roots and conjugations are laid out richly.
  - Negation comes after the affirmative is made clear first (is -> is not, comes -> does not come).
  - Words are low-resolution and noisy. Audiovisual experience is realized through words, so precision is built through volume.
    Repeating the same seed from many angles in bulk -> compensates for low resolution and cancels noise -> the filters extract the axes.

Each seed entry: (word, {attributes}). Attributes are hand-curated (ontology noise excluded).
  Size: relative to me (a person). +large (threat), 0 similar, -small (prey). Extremes such as elephant +2 and ant -2 serve as anchors.
  Sound: onomatopoeia (if present). Values on sensory axes such as touch/color/temperature/taste/movement/danger.

seeds.py — 발달 학습의 씨앗 사전 (감각/공간/논리/아기/강렬사건 공유 뼈대)

설계 원칙 (사용자 확정):
  - 씨앗은 좁게(수십 개). 넓히지 않는다. 대신 각도를 넓히고 양으로 반복한다.
  - 씨앗은 상징성이 강한 원형(archetype)이다. 코끼리=거대, 개미=작음, 호랑이=사나움.
    상징 원형은 축의 극단을 대표해 축의 기준점(앵커)이 된다. 해오라기 같은 희귀어는 배제.
  - 두 겹: 접촉 원형(아이가 실제 만짐: 개 물 손 흙) + 상징 원형(그림책 개념 대표: 코끼리 호랑이).
  - 원초 동사가 사고의 뿌리. 어근+활용을 풍부하게.
  - 부정은 긍정을 먼저 명확히(있다->없다, 온다->안온다).
  - 단어는 저해상도에 노이즈가 많다. 시청각 경험을 단어로 구현하니 양으로 정밀도를 만든다.
    같은 씨앗을 여러 각도로 대량 반복 -> 저해상도 보상 + 노이즈 상쇄 -> 필터가 축을 추출.

각 씨앗 항목: (단어, {속성}). 속성은 손 큐레이션(온톨로지 노이즈 배제).
  크기: 나(사람) 기준. +큼(위협) 0비슷 -작음(먹이). 코끼리 +2, 개미 -2 처럼 극단이 앵커.
  소리: 의성어(있으면). 촉감/색/온도/맛/움직임/위험 등 감각축 값.
"""

# ===== Animal seeds (archetypes for the size/ferocity/speed/sound/movement axes) =====
# (word, size[-2~2], ferocity[0~2], speed[0~2], sound, movement)
# ===== 동물 씨앗 (크기/사나움/속도/소리/움직임 축의 원형) =====
# (단어, 크기[-2~2], 사나움[0~2], 속도[0~2], 소리, 움직임)
동물씨앗 = [
    ("코끼리", 2, 0, 0, None, "걷는다"),      # maximum anchor of the size axis / 크기축 최대 앵커
    ("고래", 2, 0, 1, None, "헤엄친다"),       # huge in the sea / 바다 거대
    ("소", 1, 0, 0, "음매", "걷는다"),
    ("말", 1, 0, 2, "히힝", "달린다"),         # fast speed / 속도 빠름
    ("호랑이", 1, 2, 2, "어흥", "달린다"),      # maximum anchor of the ferocity axis / 사나움 최대 앵커
    ("사자", 1, 2, 1, "어흥", "걷는다"),
    ("곰", 1, 2, 1, None, "걷는다"),
    ("늑대", 0, 2, 2, "아우", "달린다"),
    ("뱀", -1, 2, 1, "쉬익", "기어간다"),       # dangerous / 위험
    ("개", -1, 1, 1, "멍멍", "달린다"),         # contact archetype (friendly plus loyal) / 접촉 원형(친근+충성)
    ("고양이", -1, 1, 1, "야옹", "걷는다"),
    ("토끼", -1, 0, 2, None, "뛴다"),          # fast plus weak / 빠름+약함
    ("여우", -1, 1, 2, None, "달린다"),
    ("다람쥐", -1, 0, 2, None, "뛴다"),
    ("닭", -1, 0, 1, "꼬끼오", "걷는다"),
    ("오리", -1, 0, 1, "꽥꽥", "걷는다"),
    ("참새", -2, 0, 2, "짹짹", "난다"),
    ("새", -1, 0, 2, "짹짹", "난다"),
    ("물고기", -1, 0, 1, None, "헤엄친다"),
    ("개구리", -2, 0, 1, "개굴개굴", "뛴다"),
    ("거북", -1, 0, 0, None, "기어간다"),       # slow speed anchor / 속도 느림 앵커
    ("달팽이", -2, 0, 0, None, "기어간다"),      # slowest extreme / 느림 최소
    ("나비", -2, 0, 1, None, "난다"),
    ("벌", -2, 1, 2, "윙윙", "난다"),          # small but dangerous (it stings) / 작지만 위험(쏜다)
    ("개미", -2, 0, 1, None, "기어간다"),        # minimum anchor of the size axis (small and diligent) / 크기축 최소 앵커(작고 부지런)
    ("거미", -2, 1, 1, None, "기어간다"),
    ("지렁이", -2, 0, 0, None, "기어간다"),       # path finding equals primitive logic / 길찾기=원초 논리
    ("쥐", -2, 0, 2, "찍찍", "달린다"),
]

# ===== Nature seeds (touch/temperature/color/brightness/danger axes) =====
# Size versus threat is not meaningful here. Sensory attributes are the focus.
# (word, {color, brightness, temperature, hardness, danger, movement})
# ===== 자연물 씨앗 (촉감/온도/색/밝기/위험 축) =====
# 크기-위협은 무의미. 감각 속성 위주.
# (단어, {색, 밝기, 온도, 단단함, 위험, 움직임})
자연씨앗 = {
    "해":  {"색": "빨갛다", "밝기": "밝다", "온도": "뜨겁다", "위치": "높다"},
    "달":  {"색": "하얗다", "밝기": "밝다", "위치": "높다", "때": "밤"},
    "별":  {"밝기": "반짝인다", "크기말": "작다", "위치": "높다", "때": "밤"},
    "불":  {"색": "빨갛다", "밝기": "밝다", "온도": "뜨겁다", "위험": "위험하다", "움직임": "타오른다"},
    "물":  {"온도": "차갑다", "촉감": "젖는다", "움직임": "흐른다", "맛": "싱겁다"},
    "돌":  {"단단함": "단단하다", "무게": "무겁다"},
    "흙":  {"색": "갈색이다", "촉감": "부드럽다", "위치": "바닥"},
    "바람": {"온도": "시원하다", "움직임": "분다", "보임": "안 보인다"},
    "비":  {"온도": "차갑다", "촉감": "젖는다", "소리": "주룩주룩", "움직임": "내린다"},
    "눈":  {"색": "하얗다", "온도": "차갑다", "촉감": "차갑다", "움직임": "내린다"},
    "풀":  {"색": "초록이다", "촉감": "부드럽다", "위치": "바닥"},
    "꽃":  {"냄새": "향기롭다", "촉감": "부드럽다"},
    "나무": {"크기말": "크다", "위치": "높다", "단단함": "단단하다"},
    "잎":  {"색": "초록이다", "움직임": "떨어진다"},
    "구름": {"색": "하얗다", "위치": "높다"},
    "얼음": {"온도": "차갑다", "단단함": "단단하다", "촉감": "미끄럽다"},
}

# ===== Body seeds (sensory organs) =====
# ===== 몸 씨앗 (감각 기관) =====
몸씨앗 = {
    "눈": "본다", "귀": "듣는다", "코": "맡는다", "입": "먹는다",
    "손": "잡는다", "발": "걷는다", "혀": "맛본다",
}

# ===== Object/food seeds (contact archetypes) =====
# (word, {movement/action, touch, taste, shape})
# ===== 사물/음식 씨앗 (접촉 원형) =====
# (단어, {움직임/동작, 촉감, 맛, 형태})
사물씨앗 = {
    "공":  {"움직임": "굴러간다", "형태": "둥글다", "동작": "던진다"},
    "집":  {"크기말": "크다", "위치말": "안"},
    "문":  {"동작": "연다", "반대동작": "닫는다"},
    "밥":  {"맛": "맛있다", "동작": "먹는다"},
    "컵":  {"동작": "잡는다", "담음": "물"},
    "옷":  {"동작": "입는다", "촉감": "부드럽다"},
    "신":  {"동작": "신는다", "위치": "발"},
    "사탕": {"맛": "달다", "동작": "먹는다"},
    "우유": {"맛": "고소하다", "색": "하얗다", "동작": "마신다"},
}

# ===== Relation/position seeds =====
# ===== 관계/위치 씨앗 =====
인칭 = ["나", "너"]
위치 = [("위", "아래"), ("앞", "뒤"), ("안", "밖"), ("여기", "저기"), ("옆", "가운데")]

# ===== Primitive verbs: root plus conjugations (the skeleton of thought) =====
# Each verb: present/imperative/connective/conditional/past/negation/prohibition. The conjugations are laid out richly.
# ===== 원초 동사: 어근 + 활용 (사고의 뼈대) =====
# 각 동사: 현재/명령/연결/조건/과거/부정/금지. 활용을 풍부하게 펼친다.
동사활용 = {
    "가다": {"현재": "간다", "해라": "가", "서": "가서", "면": "가면", "과거": "갔다", "부정": "안 간다", "금지": "가지 마"},
    "오다": {"현재": "온다", "해라": "와", "서": "와서", "면": "오면", "과거": "왔다", "부정": "안 온다", "금지": "오지 마"},
    "있다": {"현재": "있다", "해라": "있어", "면": "있으면", "과거": "있었다", "부정": "없다"},
    "먹다": {"현재": "먹는다", "해라": "먹어", "서": "먹어서", "면": "먹으면", "과거": "먹었다", "부정": "안 먹는다", "금지": "먹지 마"},
    "잡다": {"현재": "잡는다", "해라": "잡아", "면": "잡으면", "과거": "잡았다", "부정": "안 잡는다", "반대": "놓는다"},
    "보다": {"현재": "본다", "해라": "봐", "서": "봐서", "면": "보면", "과거": "봤다", "부정": "안 본다"},
    "주다": {"현재": "준다", "해라": "줘", "면": "주면", "과거": "줬다", "부정": "안 준다", "반대": "받는다"},
    "자다": {"현재": "잔다", "해라": "자", "면": "자면", "과거": "잤다", "부정": "안 잔다"},
    "울다": {"현재": "운다", "해라": "울어", "면": "울면", "과거": "울었다", "부정": "안 운다", "금지": "울지 마"},
    "웃다": {"현재": "웃는다", "해라": "웃어", "면": "웃으면", "과거": "웃었다", "부정": "안 웃는다"},
    "안다": {"현재": "안는다", "해라": "안아", "면": "안으면", "과거": "안았다"},
    "던지다": {"현재": "던진다", "해라": "던져", "면": "던지면", "과거": "던졌다"},
}

# ===== Emotion/gain-loss seeds (affirmative-negative pairs, the affirmative first) =====
# ===== 감정/득실 씨앗 (긍정-부정 짝. 긍정을 먼저) =====
감정짝 = [
    ("좋다", "싫다"), ("기쁘다", "슬프다"), ("반갑다", "무섭다"),
    ("편하다", "아프다"), ("웃는다", "운다"),
]
# Root of gain-loss: what is larger than me equals threat equals fear, what is smaller equals prey equals fine (design 4.1, the single fear axis)
# 득실 근본: 나보다 큰 것=위협=무섭다, 작은 것=먹이=괜찮다 (설계 4.1 공포 1축)

# ===== Angle list (every angle from which the infant first meets the world) =====
# These angles are repeated in bulk for every seed so that volume overcomes low resolution and noise.
# ===== 각도 목록 (아기가 세상을 처음 만나는 모든 각도) =====
# 이 각도들을 씨앗마다 대량 반복해 저해상도+노이즈를 양으로 극복한다.
각도 = [
    "존재",   # is/is not (affirmative first) / 있다/없다 (긍정 먼저)
    "크기",   # larger or smaller than me (animals; the symbolic archetypes anchor the extremes) / 나보다 큰가 작은가 (동물, 상징이 극단 앵커)
    "위협",   # threat or prey (size to fear axis) / 위협이냐 먹이냐 (크기->공포 축)
    "거리",   # near/far / 가깝다/멀다
    "방향",   # front back up down / 앞뒤위아래
    "다가옴",  # comes/goes (temporal prediction) / 온다/간다 (시간 예측)
    "시각",   # color brightness shape / 색 밝기 형태
    "청각",   # sound and cries / 소리 울음
    "촉각",   # temperature hardness roughness / 온도 단단함 거칠기
    "후각",   # smell / 냄새
    "미각",   # taste / 맛
    "개수",   # one two / 하나 둘
    "감정",   # like/dislike/fear / 좋다/싫다/무섭다
    "동작",   # verbs (goes, eats) / 동사 (간다 먹는다)
    "긍정부정",  # comes versus does not come, is versus is not / 온다<->안온다, 있다<->없다
    "소유",   # gives/receives / 준다/받는다
]


def in_vocab(w, tok):
    return all(c in tok.stoi for c in w)


def 동물목록(tok):
    return [t for t in 동물씨앗 if in_vocab(t[0], tok)]


def 자연목록(tok):
    return {k: v for k, v in 자연씨앗.items() if in_vocab(k, tok)}


def 사물목록(tok):
    return {k: v for k, v in 사물씨앗.items() if in_vocab(k, tok)}
