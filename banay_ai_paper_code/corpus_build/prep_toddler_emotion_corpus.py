# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_toddler_emotion_corpus.py — emotion trigger corpus (planting toddler emotion labels)

Training material used when planting emotions derived from the life-death gauge as triggers.
An emotion label (공포: fear, 분노: anger, 기쁨: joy, 슬픔: sadness) is placed at the head of each line, followed by the behavior of that emotion.
Follows the label-planting structure of prep_toddler_state. The labels must differ from one another so they later split into distinct triggers.
The material was drawn per emotion from the extreme-emotion and heavenly-joy sets of toddler_affect.py.

Run: python3 data_prep/prep_toddler_emotion_corpus.py   ->  data/toddler_emotion.npy (set the output folder with BANYA_DATA_DIR)

prep_toddler_emotion_corpus.py — 감정 트리거 말뭉치 (유딩 감정 라벨 심기)

생사 게이지에서 파생한 감정을 트리거로 심을 때 쓰는 학습 재료다.
줄머리에 감정 라벨(공포: 분노: 기쁨: 슬픔:)을 두고 그 감정의 행동이 이어지게 한다.
prep_유딩_상태 의 라벨 심기 구조를 따른다. 라벨이 서로 달라야 나중에 트리거로 갈린다.
재료는 toddler_affect.py 의 극단감정 천국적즐거움에서 감정별로 추렸다.

실행: python3 data_prep/prep_toddler_emotion_corpus.py   ->  data/toddler_emotion.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu

SEED = 113
N_PARA = 100000

감정 = {
    "공포": {
        "상황": ["깜깜해졌어", "천둥이 쳤어", "혼자 남았어", "괴물 나올까 봐", "주사 맞으러 갔어",
                 "큰 개가 다가와", "번쩍하고 번개 쳤어"],
        "행동": ["무서워. 벌벌 떨었어.", "이불 뒤집어썼어.", "귀 막고 울었어.", "엄마 어디 갔어 무서워.",
                 "도망쳤어.", "심장이 콩닥콩닥 뛰었어. 벌벌."],
    },
    "분노": {
        "상황": ["형아가 뺏었어", "동생이 밀었어", "나만 혼났어", "장난감 안 준대", "내 거 가져갔어"],
        "행동": ["확 화났어. 주먹 꽉 쥐었어.", "발을 쿵쿵 굴렀어.", "빽 소리 질렀어.",
                 "씩씩거렸어. 얼굴 새빨개졌어.", "이 앙 물었어."],
    },
    "기쁨": {
        "상황": ["선물 받았어", "케이크 봤어", "놀이터 갔어", "사탕 생겼어", "엄마가 안아 줬어", "미끄럼틀 탔어"],
        "행동": ["팔짝팔짝 뛰었어.", "방방 신났어.", "눈이 반짝했어.", "와아 소리쳤어.",
                 "깔깔 웃었어. 빙글빙글 돌았어.", "심장이 콩콩 뛰어."],
    },
    "슬픔": {
        "상황": ["장난감 부러졌어", "사탕 뺏겼어", "엄마가 갔어", "친구가 안 놀아 줘", "혼자 남겨졌어"],
        "행동": ["엉엉 울었어. 눈물 뚝뚝.", "너무 서러워.", "으앙 하고 터졌어.", "입 삐죽 나왔어.",
                 "코가 막히게 울었어."],
    },
}


def para(rng):
    em = list(감정)[rng.randint(0, len(감정))]
    상 = 감정[em]["상황"][rng.randint(0, len(감정[em]["상황"]))]
    행 = 감정[em]["행동"][rng.randint(0, len(감정[em]["행동"]))]
    f = rng.randint(0, 3)

    if f == 0:                                 # label planting (situation -> emotion label -> behavior) / 라벨 심기 (상황 -> 감정 라벨 -> 행동)
        return f"{상}. {행}"

    if f == 1:                                 # dialogue (user situation -> emotion-labeled reply) / 대화 (사용자 상황 -> 감정 라벨 답)
        return f"사용자: {상}.\n반야: {행}"

    # asking how it feels -> emotion label
    # 기분 물어보기 -> 감정 라벨
    return f"사용자: {상}. 기분이 어때?\n반야: {행}"


def main():
    tok = ba.AtomTokenizer()
    print(f"  유딩 감정 트리거. 라벨 {list(감정)}")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("toddler_emotion", para, N_PARA, rng, tok)
    print("표본:")
    print(tok.decode(arr[:240].tolist()))


if __name__ == "__main__":
    main()
