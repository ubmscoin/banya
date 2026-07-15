# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""prep_sense_mimic_corpus.py — Building fine-grained sense axes with sense mimetic words (self-centered, resolution reinforcement)

Draws the per-axis weak-to-strong ladders from baby_mimic.py to make the senses finely distinguishable.
The subject is I and is omitted. No object names are attached. Only the intensity of my sensations is laid out
with mimetic words. Safe mimetic words are followed by 괜찮아 옳지 (it is fine, well done) and dangerous ones by
안돼 조심 (no, careful), tying life and death to them through reward and punishment.

Run  python3 data_prep/prep_sense_mimic_corpus.py   ->  data/sense_mimic.npy (set the folder with BANYA_DATA_DIR)

prep_sense_mimic_corpus.py — 감각 흉내말로 감각축을 촘촘히 세운다 (나 중심, 해상도 보강)

baby_mimic.py 의 축별 약에서 강 사다리를 뽑아 감각을 촘촘히 구분시킨다.
주어는 나이고 생략한다. 대상 이름은 안 붙인다. 나 기준 감각의 세기만 흉내말로 편다.
안전한 흉내말엔 괜찮아 옳지, 위험한 흉내말엔 안돼 조심을 붙여 상벌로 생사와 잇는다.

실행: python3 data_prep/prep_sense_mimic_corpus.py   ->  data/sense_mimic.npy (BANYA_DATA_DIR 로 폴더 지정)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banya_atoms as ba
import corpus_parts as bu
import baby_mimic as H

SEED = 71
N_PARA = 100000


def para(rng, lex):
    축들, 사다리들 = lex
    ax = 축들[rng.randint(0, len(축들))]
    seq = 사다리들[ax]
    n = len(seq)
    f = rng.randint(0, 5)

    if f == 0:                                 # weak-to-strong ladder (fine resolution; weak front, strong back) / 약에서 강 사다리 (해상도 촘촘히. 앞이 약 뒤가 강)
        i = rng.randint(0, max(1, n - 2))
        k = min(rng.randint(3, 7), n - i)
        return ". ".join(w for w, _ in seq[i:i + k]) + "."

    if f == 1:                                 # reward and punishment (safe gets fine and well done, danger gets no, caution gets careful) / 상벌 (안전엔 괜찮아 옳지, 위험엔 안돼, 주의엔 조심)
        w, tag = seq[rng.randint(0, n)]
        react = "괜찮아. 옳지." if tag == "안" else ("안돼." if tag == "위" else "조심.")
        return f"{w}. {react}"

    if f == 2:                                 # intensity comparison (the later one is stronger; intensity relative to me) / 세기 비교 (뒤가 더 세다. 나 기준 강도)
        i = rng.randint(0, n - 1)
        j = rng.randint(i + 1, n)
        return f"{bu.josa(seq[i][0], '이랑', '랑')} {seq[j][0]}, 뭐가 더 세? {seq[j][0]}."

    if f == 3:                                 # short-answer sense retrieval (what is this) / 감각 단답 인출 (이거 뭐야)
        w, _ = seq[rng.randint(0, n)]
        return f"이거? {w}."

    # contrast of the two extremes (the weakest one and the strongest one)
    # 약 강 양끝 대비 (제일 약한 것과 제일 센 것)
    약 = seq[0][0]
    강 = seq[n - 1][0]
    return f"제일 약한 건? {약}. 제일 센 건? {강}."


def wrap_대화(p, rng):
    if rng.randint(0, 4) == 0 and "?" in p and "\n" not in p:
        q, _, a = p.rpartition("? ")
        return f"사용자: {q}?\n반야: {a}"
    return p


def main():
    tok = ba.AtomTokenizer()
    축들 = H.축목록(tok)
    사다리들 = {ax: H.사다리(tok, ax) for ax in 축들}
    걸린 = {ax: [w for w, _ in H.흉내[ax] if not all(c in tok.stoi for c in w)] for ax in H.흉내}
    걸린 = {ax: v for ax, v in 걸린.items() if v}
    print(f"  축 {len(축들)}개, 흉내말 총 {sum(len(v) for v in 사다리들.values())}개")
    if 걸린:
        print(f"  vocab 밖으로 빠진 흉내말: {걸린}")
    lex = (축들, 사다리들)
    rng = np.random.RandomState(SEED)
    arr = bu.bake("sense_mimic", lambda r: para(r, lex), N_PARA, rng, tok, wrap=wrap_대화)
    print("표본:")
    print(tok.decode(arr[:280].tolist()))


if __name__ == "__main__":
    main()
