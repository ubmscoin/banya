# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""world_ladder.py — helper that narrates self-centered axes in order (core pattern)

User principle. I am the geometric center, origin 0. Axes are not merely enumerated; they unfold in order relative to me.
  Axes that extend in one direction only (distance, loudness, pain, brightness, arousal, texture) run once from 0 toward the large end. The small comes first and the large comes last.
  Axes that extend in both directions (temperature, size, approach, gain-loss) unfold three times.
    From 0 to the negative end, from 0 to the positive end, and from the negative end to the positive end.
    This is information that lets intensity be understood relative to me, and lets degree such as hotness also be understood from the ordering span of the axis itself.
Within a single sentence the front stays small and the back stays large. The order itself is information for gauging magnitude.

world_ladder.py — 나 중심 축을 순서대로 서술하는 헬퍼 (핵심 패턴)

사용자 원칙. 나는 기하 중심 원점 0이다. 축을 단순히 나열하지 않고 나를 기준으로 순서대로 편다.
  한쪽으로만 뻗는 축(거리 소리크기 통증 밝기 각성 질감)은 0에서 큰 쪽으로 한 번. 작은 것이 앞, 큰 것이 뒤.
  양쪽으로 뻗는 축(온도 크기 다가옴 득실)은 세 번 편다.
    0에서 음의 끝으로, 0에서 양의 끝으로, 음의 끝에서 양의 끝으로.
    나를 기준으로 강도를 이해하고, 축 자체의 나열 차에서 뜨거움으로도 이해하게 하는 정보다.
한 문장 안에서도 앞은 작고 뒤는 크게 유지한다. 순서 자체가 크기를 가늠하는 정보다.
"""
import baby_world_expr as W

축 = W.축
ONESIDE = ["거리", "소리크기", "통증", "밝기", "각성", "질감", "습기", "끈기", "탄성", "둔통", "속느낌", "투명"]
BIDIR = {"온도": ["미지근", "딱 좋"], "크기": ["나만", "비슷", "똑같"],
         "다가옴": ["멈춰", "가만", "그대로", "멈춘", "제자리"],
         "득실": ["괜찮", "그저", "그냥", "무덤덤"],
         "life": ["살아 있", "그대로", "여전"], "시간": ["지금"],
         "날씨": ["포근"], "소리높이": ["보통"], "높낮이": ["나만"], "넓이": ["알맞"]}


def 중립(ax):
    lv = 축[ax]["levels"]
    for kw in BIDIR.get(ax, []):
        for i, w in enumerate(lv):
            if kw in w:
                return i
    return len(lv) // 2


def _순서(rng, pool, k):
    pool = list(pool)
    if not pool:
        return []
    k = min(k, len(pool))
    return sorted(int(x) for x in rng.choice(pool, size=k, replace=False))


def 오름(rng, ax):                             # one-sided axis, from 0 toward the large end, small first and large last / 한쪽 축. 0에서 큰 쪽으로. 작은 앞 큰 뒤
    lv = 축[ax]["levels"]
    n = len(lv)
    k = rng.randint(3, 7)
    start = 0 if rng.randint(0, 2) else rng.randint(0, max(1, n - 3))
    idx = _순서(rng, range(start, n), k)
    return ". ".join(lv[i] for i in idx) + "."


def 양방(rng, ax):                             # two-sided axis, three patterns, starting from me at 0 / 양쪽 축. 세 패턴. 나 0 에서 시작
    lv = 축[ax]["levels"]
    n = len(lv)
    c = 중립(ax)
    p = rng.randint(0, 3)
    if p == 0:                                 # from 0 to the negative end (the side lower than me) / 0 에서 음의 끝으로 (나에서 낮은 쪽)
        idx = _순서(rng, range(0, c), 3)
        seq = [lv[c]] + [lv[i] for i in reversed(idx)]
    elif p == 1:                               # from 0 to the positive end (the side higher than me) / 0 에서 양의 끝으로 (나에서 높은 쪽)
        idx = _순서(rng, range(c + 1, n), 3)
        seq = [lv[c]] + [lv[i] for i in idx]
    else:                                      # from the negative end to the positive end (full sweep) / 음의 끝에서 양의 끝으로 (전체 훑기)
        idx = _순서(rng, range(0, n), 6)
        seq = [lv[i] for i in idx]
    return ". ".join(seq) + "."


def 단일(rng, ax):
    lv = 축[ax]["levels"]
    return lv[rng.randint(0, len(lv))] + "."


def 예문(rng, ax):
    sn = 축[ax]["sentences"]
    return sn[rng.randint(0, len(sn))]
