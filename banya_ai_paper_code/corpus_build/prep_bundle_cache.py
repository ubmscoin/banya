# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_bundle_cache.py — Offline baking of the bundle cache (two-tier dictionary) for dynamic embedding.

See 설계철학/동적임베딩_설계.md. v1 = the cache is fixed (runtime LRU growth is v2).
Adopted policy = take the top-K most frequently used words whole, as bundles. No BPE merging (slow).
Only word frequencies are counted.
A word = a run of Hangul syllables (broken at spaces and punctuation). Length 2 or more only
(length 1 is already in the syllable base).
Output = bundle id (placed after the syllable vocab) + syllable decomposition + surface form.
Used jointly by training baking and the model.
Saved to: banya_world_data/bundle_cache.json
Run: cd banya_ai && /home/khan/ubms-venv/bin/python data_prep/prep_bundle_cache.py [bundle count]

동적 임베딩용 묶음 캐시(2층 사전) 오프라인 굽기.

설계철학/동적임베딩_설계.md. v1 = 캐시 고정(런타임 LRU 성장은 v2).
채택 방침 = "많이 쓰는 단어" top-K 를 통째로 묶음으로. BPE 병합 안 함(느림). 단어 빈도만 센다.
단어 = 한글 음절 연속(공백 문장부호서 끊음). 길이 2 이상만(길이 1 = 음절 베이스에 이미 있음).
출력 = 묶음 id(음절 vocab 뒤) + 음절 분해 + 표기. 학습 굽기와 모델이 같이 씀.
저장: banya_world_data/bundle_cache.json
실행: cd banya_ai && /home/khan/ubms-venv/bin/python data_prep/prep_bundle_cache.py [묶음수]
"""
import os
import sys
import json
import time
from collections import Counter
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, "core")
import banya_atoms as ba

N_BUNDLE = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
# Broad enough to represent the whole curriculum, including vocabulary yet to be seen
# 커리큘럼 전체를 대표하게 넓게. 앞으로 볼 어휘
SOURCES = [
    "life", "space", "sense", "baby", "baby_learn", "toddler", "toddler_dialog", "toddler_exp",
    "elem", "elem_dialog", "elem_knowledge", "tale_body", "tale_qa", "tale_summary",
]
MAX_LEN = 8                 # maximum bundle length, prevents overly long single chunks / 묶음 최대 길이(너무 긴 덩어리 방지)


def main():
    t0 = time.time()
    tok = ba.AtomTokenizer()
    hset = set(i for i in range(tok.vocab) if "가" <= tok.itos[i] <= "힣")
    words = Counter()
    for nm in SOURCES:
        p = os.path.join("banya_world_data", nm + ".npy")
        if not os.path.exists(p):
            continue
        arr = np.asarray(np.load(p, mmap_mode="r"), dtype=np.int64)
        cur = []
        for t in arr:
            if int(t) in hset:
                cur.append(int(t))
            else:
                if 2 <= len(cur) <= MAX_LEN:
                    words[tuple(cur)] += 1
                cur = []
        if 2 <= len(cur) <= MAX_LEN:
            words[tuple(cur)] += 1
    print(f"고유 단어 {len(words):,} · 수집 {time.time()-t0:.1f}s")

    top = words.most_common(N_BUNDLE)
    cache = []
    base = tok.vocab
    for k, (syl, freq) in enumerate(top):
        s = "".join(tok.itos[i] for i in syl)
        cache.append({"id": base + k, "syl": list(syl), "str": s, "freq": int(freq)})

    lens = Counter(len(c["syl"]) for c in cache)
    print(f"묶음 {len(cache)}개 · 길이별 {{{', '.join(f'{k}:{lens[k]}' for k in sorted(lens))}}}")
    print("최고빈도 표본:", " ".join(c["str"] for c in cache[:20]))
    print("최저빈도 표본:", " ".join(c["str"] for c in cache[-12:]))

    out = {"base_vocab": tok.vocab, "n_bundle": len(cache), "sources": SOURCES, "bundles": cache}
    with open("banya_world_data/bundle_cache.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"저장 banya_world_data/bundle_cache.json · vocab {tok.vocab} -> {tok.vocab + len(cache)} · 총 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
