# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_baby_corpus.py — baby base (no self; own body plus interactions within the range of action)

A baby does not yet have a reflective self and cannot distinguish itself from objects, so identity
sentences such as 나는 are not used. A baby learns centered on its own body and comes to understand
through interactions within its range of action (attachment figures mom and dad, nearby objects
within reach). Distant objects (picture-book symbols such as elephants, whales, tigers) are not
here; they belong to toddler environment observation. Expressions come from baby_expr.py (workflow
output, in-vocab filtering complete). There are seven domains:
  own body, mom-dad, nearby objects, needs, emotions, interaction, onomatopoeia.

Run  python3 data_prep/prep_baby_corpus.py   ->  data/baby.npy (set the folder with BANYA_DATA_DIR)

prep_baby_corpus.py — 아기 base (자아 없음, 자기 신체 + 행동반경 안 상호작용)

아기는 아직 반성적 자아가 없다. 대상과 자기를 구분 못 한다. 그래서 나는 같은 정체성 문장을 안 쓴다.
아기는 자기 신체를 중심으로 배우고, 행동반경 안(엄마 아빠 애착 대상, 손에 닿는 가까운 사물)에서
상호작용하며 이해한다. 먼 대상(코끼리 고래 호랑이 등 그림책 상징)은 여기 없다. 그건 유딩 환경관찰이다.
표현은 baby_expr.py (워크플로 산출, in-vocab 필터 완료)에서 온다. 영역은 일곱이다.
  자기신체 엄마아빠 가까운사물 욕구 감정 상호작용 의성어.

실행: python3 data_prep/prep_baby_corpus.py   ->  data/baby.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import baby_expr as A

SEED = 41
N_PARA = 100000

영역들 = list(A.표현.keys())


def para(rng):
    dom = 영역들[rng.randint(0, len(영역들))]
    exprs = A.표현[dom]
    k = rng.randint(1, 4)                       # one paragraph bundles 1 to 3 expressions from the same domain / 같은 영역 표현 1~3개 묶어 한 문단
    picks = []
    for _ in range(k):
        e = exprs[rng.randint(0, len(exprs))]
        if e not in picks:
            picks.append(e)
    return " ".join(picks)


def wrap_대화(p, rng):
    if rng.randint(0, 5) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    print(f"  아기 base. 자아 없음, 자기 신체 + 행동반경. 영역 {len(영역들)}개, 표현 {sum(len(v) for v in A.표현.values())}개")
    rng = np.random.RandomState(SEED)
    arr = bu.bake("baby", para, N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:280].tolist()))


if __name__ == "__main__":
    main()
