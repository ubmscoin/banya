# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""baby_mimic.py — graded ladders of onomatopoeic and mimetic words per sensory axis (weak to strong, self-centered sensory resolution)

Solidifies into data the survey memory/아기말_표현_전수조사.md and the sensory-axis mapping survey output.
Each axis lines up mimetic words in order from weak to strong stimulus. Bidirectional axes (temperature, taste, smell, arousal) run from the negative end through neutral to the positive end.
Each item is (mimetic word, tag). Among the tags, 안 means safe, 주 means caution, 위 means danger. Seen in life-death terms, 안 lies on the life side while 주 and 위 lie on the death side.
Use 사다리(tok, ax) to filter and keep only entries that fall within the vocab.

baby_mimic.py — 감각축별 의성어 의태어 등급 사다리 (약에서 강, 나 중심 감각 해상도)

조사 memory/아기말_표현_전수조사.md 와 감각축 매핑 조사 산출물을 데이터로 굳힌 것이다.
각 축은 약한 자극에서 강한 자극 순서로 흉내말을 세운다. 양방축(온도 맛 냄새 각성)은 음끝에서 중립을 지나 양끝으로.
항목은 (흉내말, 태그). 태그 안은 안전 주는 주의 위는 위험. 생사로 보면 안은 생쪽 주와 위는 사쪽이다.
사다리(tok, ax) 로 vocab 안에 드는 것만 걸러 쓴다.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))

흉내 = {
    "온도": [("꽁꽁", "위"), ("오들오들", "주"), ("덜덜", "주"), ("서늘", "안"), ("시원", "안"),
             ("미지근", "안"), ("따끈따끈", "안"), ("따뜻", "안"), ("뜨끈뜨끈", "주"), ("후끈", "주"),
             ("화끈", "주"), ("앗뜨", "위"), ("이글이글", "위"), ("활활", "위")],
    "통증": [("콕", "주"), ("따끔", "주"), ("뜨끔", "주"), ("콕콕", "주"), ("쿡", "주"),
             ("따가움", "주"), ("쓰라려", "위"), ("욱신", "위"), ("지끈", "위")],
    "질감": [("몰랑몰랑", "안"), ("말랑말랑", "안"), ("물렁물렁", "안"), ("보들보들", "안"), ("보송보송", "안"),
             ("폭신폭신", "안"), ("포근", "안"), ("매끈매끈", "안"), ("반질반질", "안"), ("반들반들", "안"),
             ("미끌미끌", "주"), ("까칠까칠", "주"), ("까끌까끌", "주"), ("까슬까슬", "주"), ("오돌토돌", "주"),
             ("울퉁불퉁", "주"), ("뾰족뾰족", "위")],
    "밝기": [("깜깜", "주"), ("캄캄", "주"), ("컴컴", "주"), ("어둑어둑", "안"), ("깜빡깜빡", "안"),
             ("환히", "안"), ("훤히", "안"), ("반짝반짝", "안"), ("번쩍번쩍", "주"), ("번뜩", "주"),
             ("쨍", "위"), ("눈부셔", "위")],
    "소리크기": [("소곤소곤", "안"), ("속닥속닥", "안"), ("조곤조곤", "안"), ("도란도란", "안"), ("두런두런", "안"),
                 ("또각또각", "안"), ("통통", "안"), ("쿵쿵", "주"), ("쾅", "주"), ("꽝", "위"),
                 ("우당탕", "주"), ("우르르", "주"), ("시끌시끌", "주"), ("와글와글", "주")],
    "크기": [("쪼끄만", "안"), ("조그만", "안"), ("자그마", "안"), ("요만한", "안"), ("큼직", "안"),
             ("커다란", "안"), ("어마어마", "주"), ("거대", "주"), ("산더미만한", "주")],
    "다가옴": [("꾸물꾸물", "안"), ("엉금엉금", "안"), ("아장아장", "안"), ("뒤뚱뒤뚱", "안"), ("살금살금", "안"),
               ("살살", "안"), ("슬금슬금", "안"), ("데굴데굴", "주"), ("또박또박", "안"), ("성큼성큼", "안"),
               ("폴짝", "안"), ("후다닥", "주"), ("휙", "주"), ("쌩", "위"), ("슝", "주")],
    "각성": [("나른", "안"), ("노곤", "안"), ("새근새근", "안"), ("스르르", "안"), ("꾸벅꾸벅", "안"),
             ("몽글몽글", "안"), ("방긋", "안"), ("싱글벙글", "안"), ("두근두근", "안"), ("콩닥콩닥", "안"),
             ("울렁울렁", "주"), ("조마조마", "주"), ("움찔", "주"), ("흠칫", "주"), ("깜짝", "주"),
             ("화들짝", "주"), ("펄쩍", "주")],
    "맛": [("밍밍", "안"), ("고소", "안"), ("달콤", "안"), ("짭짤", "주"), ("새콤", "주"), ("매콤", "주")],
    "냄새": [("향긋", "안"), ("고소", "안"), ("솔솔", "안"), ("시큼", "주"), ("지린", "주"),
             ("구릿", "주"), ("칼칼", "위")],
}


def 사다리(tok, ax):                                # only mimetic words within the vocab (dropped if even one syllable falls outside) / vocab 안에 드는 흉내말만 (음절 하나라도 밖이면 뺀다)
    out = []
    for w, tag in 흉내[ax]:
        if all(c in tok.stoi for c in w):
            out.append((w, tag))
    return out


def 축목록(tok):                                    # only axes whose ladder keeps at least two entries / 사다리가 두 개 이상 남는 축만
    return [ax for ax in 흉내 if len(사다리(tok, ax)) >= 2]
