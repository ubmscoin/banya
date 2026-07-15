# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""corpus_parts.py — shared parts for the synthetic corpus generators

Collects in one place the particle rules, numerals, fake names, and save loop that used to be
duplicated in every generator. Fixing a particle bug once here propagates to all generators.

Caution: to guarantee byte-identical output with the existing generators, the function logic and
the order of random-number calls are kept exactly as in the originals. Changing the computation
order of a function here produces a different corpus from the same seed.

Why there are two particle functions: they branch differently when the last character is not
Hangul (digits, symbols).
  josa  a non-Hangul ending takes the b side (treated as having no final consonant)
  jo    a non-Hangul ending takes the a side (treated as having a final consonant)
Both are in active use, so they are left separate rather than unified.

The save path defaults to data/ and can be changed with the BANYA_DATA_DIR environment
variable (for verification).

corpus_parts.py — 합성 말뭉치 생성기 공용 부품

생성기마다 중복 작성돼 있던 조사 규칙, 수사, 가짜 이름, 저장 루프를 한 곳으로 모았다.
조사 버그를 여기 한 곳만 고치면 전 생성기에 반영된다.

주의: 기존 생성기와 바이트 동일 출력을 보장하려고 함수 논리와 난수 호출 순서를
원본 그대로 유지했다. 여기 함수의 계산 순서를 바꾸면 같은 씨앗에서 다른 말뭉치가 나온다.

조사 함수가 두 벌인 이유: 끝 글자가 한글이 아닐 때(숫자 기호) 갈래가 다르다.
  josa  비한글 끝이면 b 쪽 (받침 없음 취급)
  jo    비한글 끝이면 a 쪽 (받침 있음 취급)
둘 다 실사용 중이라 통일하지 않고 그대로 둔다.

저장 경로는 기본 data/ 이고, 환경변수 BANYA_DATA_DIR 로 바꿀 수 있다 (검증용).
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
import banya_atoms as ba

받침수 = {"0", "1", "3", "6", "7", "8"}                # readings yeong il sam yuk chil pal end with a final consonant; tens end in sip / 영 일 삼 육 칠 팔. 십 단위 끝은 십
십자리 = ["", "십", "이십", "삼십", "사십", "오십", "육십", "칠십", "팔십", "구십"]
일자리 = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

# Uses only common syllables. Rare syllables have less-trained embeddings, which makes them unfavorable as name material
# 흔한 음절만 쓴다. 희귀 음절은 임베딩이 덜 배워져 있어 이름 재료로 불리하다
SYL_전체 = list("가나다라마바사자차카타파하고노도로모보소조초코토포호구누두루무부수주추쿠투푸후"
                "기니디리미비시지치키티피히게네데레메베세제체케테페헤미리나라준수영진우현민연희철호")


def josa(w, a, b):                        # selects the particle by presence of a final consonant; a when present, non-Hangul ending takes b / 받침 유무로 조사 선택. a 는 받침 있을 때. 비한글 끝은 b
    ch = w[-1]
    if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0:
        return w + a
    return w + b


def jo(w, a, b):                          # particle after Hangul; a non-Hangul ending takes a / 한글 뒤 조사. 비한글 끝은 a
    ch = w[-1]
    return w + a if not ("가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 0) else w + b


def jn(n, a, b, sep=""):                  # particle after a number, judged by the reading of the last digit / 숫자 뒤 조사. 끝자리 발음 기준
    t = str(n)
    return f"{n}{sep}{a}" if t[-1] in 받침수 or (t.endswith("0") and len(t) > 1) else f"{n}{sep}{b}"


def h(n):                                 # numeral composition rule; covers all of 0 to 99, and 100 is baek / 수사 조합 규칙. 0~99 전부, 100 은 백
    if n == 0:
        return "영"
    if n == 100:
        return "백"
    return 십자리[n // 10] + 일자리[n % 10]


def syl_list(tok):                        # only syllables present in the vocabulary, order preserved / 사전에 있는 음절만. 순서 유지
    return [s for s in SYL_전체 if s in tok.stoi]


def make_name(rng, syl):                  # fake name of 2 to 3 syllables; the length is drawn first, then the syllables / 2~3음절 가짜 이름. 길이 뽑기가 먼저, 음절 뽑기가 다음
    n = rng.randint(2, 4)
    return "".join(syl[rng.randint(0, len(syl))] for _ in range(n))


def atomic_save(arr, path):               # atomic replacement that stays safe even while training holds the file as a memmap / 학습이 memmap 으로 물고 있어도 안전한 원자적 교체
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)


def bake(name, para, n, rng, tok, wrap=None, suffix="\n\n", progress=15000):
    # Standard baking loop. para(rng) yields one paragraph; wrap(p, rng) is post-processing such as a dialogue frame (called for every paragraph)
    # If rng is used inside wrap, that call order is also part of the output. Do not change the order in which conditions are evaluated
    # 표준 굽기 루프. para(rng) 가 문단 하나, wrap(p, rng) 는 대화 틀 등 후처리 (매 문단 호출)
    # wrap 안에서 rng 를 쓰면 그 호출 순서까지 생성물의 일부다. 조건 평가 순서를 바꾸지 말 것
    out = []
    본문 = []                                  # human-readable source text, also saved as txt / 사람이 읽을 원문. txt 로도 저장한다
    for i in range(n):
        p = para(rng)
        if wrap is not None:
            p = wrap(p, rng)
        out.extend(tok.encode(p + suffix))
        본문.append(p)
        if (i + 1) % progress == 0:
            print(f"  {i + 1:,}/{n:,} 문단, 누적 {len(out):,} 토큰", flush=True)
    arr = np.asarray(out, dtype=np.int32)
    assert int(arr.max()) < tok.vocab, "vocab 밖 토큰"
    d = os.environ.get("BANYA_DATA_DIR", os.path.join(_ROOT, "data"))
    p = os.path.join(d, f"{name}.npy")
    atomic_save(arr, p)
    with open(os.path.join(d, f"{name}.txt"), "w", encoding="utf-8") as f:   # human-readable copy for inspection / 검수용 사람 읽기
        f.write("\n\n".join(본문))
    print(f"저장 {p} + {name}.txt · {len(arr):,} 토큰")
    return arr
