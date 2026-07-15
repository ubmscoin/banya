# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Vocabulary build smoke test. Builds the vocabulary from a small example corpus and checks
that bake_corpora and open_corpora operate correctly. CacheTokenizer merges the syllable
dictionary and the bundle dictionary into the vocabulary, the example txt is tokenized with
that vocabulary, then bake_corpora bakes the syllable ids into bundle ids and open_corpora
opens them again.

Run  python3 smoke_vocab.py

보캅 빌드 스모크. 작은 예시 말뭉치에서 어휘를 만들고 bake_corpora 와 open_corpora 가 동작하는지 확인한다.
CacheTokenizer 가 음절 사전과 묶음 사전을 합쳐 보캅을 만들고 예시 txt 를 그 보캅으로 토큰화한 뒤
bake_corpora 로 음절 id 를 묶음 id 로 굽고 open_corpora 로 다시 연다.

실행  python3 smoke_vocab.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import banya_core as bc


def main():
    _txt = os.path.join(_HERE, "example.txt")
    with open(_txt, encoding="utf-8") as f:
        _text = f.read()

    tok = bc.CacheTokenizer()
    print(f"[보캅] vocab_size {tok.m_vocab_size} = 음절 {tok.m_base_vocab} + 묶음 {len(tok.m_bundles)}", flush=True)

    _syllable_ids = np.asarray(tok.base.encode(_text), dtype=np.int64)
    print(f"[토큰화] 예시 말뭉치 글자 {len(_text)} -> 음절 id {len(_syllable_ids)}", flush=True)

    _sandbox = os.path.join(_HERE, "스모크작업")
    _corpus_dir = os.path.join(_sandbox, "banya_world_data")
    os.makedirs(_corpus_dir, exist_ok=True)
    np.save(os.path.join(_corpus_dir, "example.npy"), _syllable_ids)

    bc._ROOT = _sandbox
    bc.MIX_WEIGHTS = {"example": 1.0}
    bc.WARMUP_MIX = {}
    bc.MIX_SCHED = {}
    bc.DATA_DIRS = [_corpus_dir]

    print("[bake_corpora] 음절 id 를 묶음 id 로 굽는다", flush=True)
    bc.bake_corpora(tok)

    _bundled = np.load(os.path.join(_corpus_dir, "example_묶음.npy"))
    print(f"[결과] 묶음 id {len(_bundled)} (음절 대비 {len(_bundled)/max(len(_syllable_ids),1)*100:.0f}%)", flush=True)

    _avail = bc.open_corpora(["example"], tok.m_vocab_size)
    if "example" not in _avail:
        sys.exit("[!] open_corpora 가 example 을 열지 못했다")
    print(f"[open_corpora] 로드 성공. 길이 {len(_avail['example'])}", flush=True)

    _decoded = tok.decode(_bundled[:40].tolist())
    print(f"[왕복] 앞부분 복원: {_decoded!r}", flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
