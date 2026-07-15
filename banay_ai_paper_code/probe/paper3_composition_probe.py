# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 3 compositional generalization probe. Measures whether the model correctly assembles stem and ending combinations never seen even once in training.
Builds a grid of combined forms from several stems and several endings that attach unchanged regardless of a final consonant, then scans every corpus the training accessed over the token stream to split seen from unseen combinations.
Assembly accuracy is measured by minimal pair discrimination. Assigning a higher probability to the correct assembly than to the wrong form made by only reversing the order of the same tokens counts as correct, and the random baseline is 50 percent.
Pairs whose reversed sequence actually occurs in the corpora (e.g. 지가 is common as a particle after nouns) carry no discriminative power and are excluded. Averaging over five frames reduces the prior probability bias of any single frame.
Because syllable transition statistics alone solve much of this task, a bigram baseline from the same corpora is run through the same discrimination and reported alongside,
and the diagnostic subset where the bigram errs but the model succeeds is reported separately. This subset is the evidence of assembly knowledge beyond surface statistics.
The perturbation has two tiers. Tier 1 order reversal is an extreme ill-formed string and thus a lower bound check, while tier 2 syllable swapping inside the ending (지만 vs 만지, 다가 vs 가다)
is the hard item set where the bigram fails because the neighbor statistics of the swapped sequence are often even more frequent (가다, 니다).
Loads the model through the shared foundation banya_core and runs the forward pass. The vocabulary uses only the syllable atom dictionary. GPU-only cupy.
Run  python3 paper3_composition_probe.py

반야 제3편 조합 일반화 프로브. 학습에서 한 번도 못 본 어근과 어미 조합을 모델이 바르게 조립하는지 실측한다.
어간 여러 개와 받침 유무와 무관하게 그대로 붙는 어미 여러 개로 결합형 격자를 만들고 학습이 접근한 말뭉치 전체를 토큰 열에서 훑어 본 조합과 안 본 조합을 가른다.
조립 정확도는 최소쌍 판별로 잰다. 같은 토큰들을 순서만 뒤집은 오답보다 바른 조립에 더 높은 확률을 주면 정답이고 무작위 기준선은 50퍼센트이다.
뒤집은 열이 말뭉치에 실재하는 쌍(예 지가 는 명사 뒤 조사로 흔함)은 판별력이 없으므로 제외한다. 프레임 다섯 개로 평균해 한 프레임의 사전확률 치우침을 줄인다.
음절 전이 통계만으로도 이 과제가 상당히 풀리므로 같은 말뭉치의 바이그램 기준선을 같은 판별로 돌려 병기하고
바이그램이 틀리는데 모델이 맞히는 진단 부분집합을 따로 보고한다. 이 부분집합이 표면 통계 너머의 조립 지식 증거이다.
교란은 두 단계다. 1단계 순서 뒤집기는 극단 비문이라 하한 점검이고 2단계 어미 내부 음절 뒤바꿈(지만 대 만지, 다가 대 가다)은
뒤바뀐 열의 이웃 통계가 오히려 흔한 경우가 많아(가다, 니다) 바이그램이 넘어지는 어려운 문항이다.
공통 토대 banya_core 로 모델을 불러 순전파한다. 어휘는 음절 원자 사전만 쓴다. GPU 전용 cupy.
실행  python3 paper3_composition_probe.py"""
import os
import sys
import math
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "bitok_elem2_170000_m.npz")
TRAIN_CORPORA = ["life", "space", "sense", "sense_space", "sense_mimic",
                 "baby", "baby_logic", "baby_learn",
                 "toddler", "toddler_logic", "toddler_learn", "toddler_exp", "toddler_dialog", "toddler_state", "toddler_emotion",
                 "toddler2", "toddler2_link",
                 "elem", "elem_knowledge", "elem_inquiry", "elem_logic", "elem_dialog", "elem_subject"]
# The dictionary corpus is not in the mixing list, but the will system used to insert its windows into training batches, so it is included in the exposure scope
# 사전 은 혼합 목록에 없지만 의지 시스템이 학습 배치에 창을 끼워 넣던 말뭉치라 노출 범위에 포함한다
STEMS = ["먹", "잡", "읽", "씻", "놀", "자", "보", "가", "오", "주", "받", "웃",
         "씹", "닦", "밀", "끌", "참", "숨", "벗", "뛰"]
ENDINGS = ["고", "지", "게", "지만", "거나", "다가", "도록", "자마자", "던", "든지", "다니", "곤"]
FRAMES = ["나는 ", "그는 ", "우리는 ", "동생이 ", "아기가 "]


# Role: counts occurrences of a short id pattern in a token stream
# Method: locates positions of the first id, then vector-checks that the remaining ids follow
# Why: the corpora hold hundreds of millions of tokens, so counting must run directly on the id stream
# 역할: 토큰 열에서 짧은 아이디 패턴의 등장 횟수를 센다
# 방법: 첫 아이디가 나오는 자리를 먼저 찾고 그 자리들에서 나머지 아이디가 이어지는지 벡터로 확인한다
# 이유: 말뭉치가 수억 토큰이라 문자열로 풀지 않고 아이디 열에서 바로 세야 하기 때문
def p_count_pattern(arr, pat):
    _n = len(pat)
    if len(arr) < _n:
        return 0
    _pos = np.flatnonzero(arr[:len(arr) - _n + 1] == pat[0])
    for k in range(1, _n):
        if len(_pos) == 0:
            return 0
        _pos = _pos[arr[_pos + k] == pat[k]]
    return int(len(_pos))


# Role: counts corpus occurrences of the combined forms and their reversals while also accumulating a bigram frequency table
# Method: opens each corpus in turn, counts the patterns, and accumulates neighboring token pairs into the frequency table
# Why: the unseen combination judgment and the baseline training must see the same data for the comparison to be fair
# 역할: 결합형과 뒤집은 열의 말뭉치 등장 횟수를 세고 바이그램 빈도표를 같이 쌓는다
# 방법: 말뭉치를 하나씩 열어 패턴을 세고 이웃 토큰 쌍을 빈도표에 누적한다
# 이유: 안 본 조합 판정과 기준선 학습이 같은 데이터를 봐야 비교가 공정하기 때문
def p_scan_corpora(tok_base, forms, vocab):
    _pats = {f: np.asarray(tok_base.encode(f), dtype=np.int64) for f in forms}
    _counts = {f: 0 for f in forms}
    _bigram = np.zeros(vocab * vocab, dtype=np.int64)
    for nm in TRAIN_CORPORA:
        _path = os.path.join("banya_world_data", nm + ".npy")
        if not os.path.exists(_path):
            print(f"  [경고] 말뭉치 없음 {nm}", flush=True)
            continue
        _arr = np.asarray(np.load(_path, mmap_mode="r"), dtype=np.int64)
        for f in forms:
            _counts[f] += p_count_pattern(_arr, _pats[f])
        _idx = _arr[:-1] * vocab + _arr[1:]
        _bigram += np.bincount(_idx, minlength=vocab * vocab)
        del _arr
    return _counts, _bigram.reshape(vocab, vocab)


# Role: measures the list of per-position log probabilities of a sentence from the model
# Method: converts the sentence to syllable ids, runs the forward pass, and records the log probability of the correct next token at each position
# Why: separating the full margin from the tail margin without the first divergent token requires per-position values
# 역할: 문장의 자리별 로그확률 목록을 모델에서 잰다
# 방법: 문장을 음절 아이디로 만들어 순전파하고 자리마다 다음 토큰 정답의 로그확률을 담는다
# 이유: 마진 전체와 첫 갈림 토큰을 뺀 꼬리 마진을 나눠 보려면 자리별 값이 필요하기 때문
def p_model_logps(m, ids):
    _X = xp.asarray(ids, dtype=xp.int64).reshape(len(ids), 1)
    _, _, _z = bc.forward(m, _X)
    _zz = bc.to_host(_z).astype(np.float64)
    _out = []
    for t in range(len(ids) - 1):
        _col = _zz[:, t] - _zz[:, t].max()
        _p = np.exp(_col)
        _p /= _p.sum()
        _out.append(float(np.log(_p[ids[t + 1]] + 1e-300)))
    return _out


# Role: measures the list of per-position log probabilities of a sentence from the bigram frequency table
# Method: records the log of the Laplace-smoothed conditional probability for each neighboring pair
# Why: how far surface transition statistics alone solve the same discrimination is the baseline
# 역할: 문장의 자리별 로그확률 목록을 바이그램 빈도표에서 잰다
# 방법: 이웃 쌍마다 라플라스 평활을 얹은 조건부확률의 로그를 담는다
# 이유: 표면 전이 통계만으로 같은 판별을 얼마나 푸는지가 기준선이기 때문
def p_bigram_logps(bigram, row_sum, vocab, ids):
    _out = []
    for t in range(len(ids) - 1):
        _p = (bigram[ids[t], ids[t + 1]] + 1.0) / (row_sum[ids[t]] + vocab)
        _out.append(float(np.log(_p)))
    return _out


# Role: measures the minimal pair margin of one combined form averaged over frames
# Method: for each frame, measures the difference in summed log probability between the correct assembly and the reversal, also measures the tail margin without the first divergent token term, then averages
# Why: the first divergent token term is the syllable prior after the frame rather than the assembly judgment, so it must be shown separately
# 역할: 한 결합형의 최소쌍 마진을 프레임 평균으로 잰다
# 방법: 프레임마다 바른 조립과 뒤집기의 로그확률 합 차이를 재고 첫 갈림 토큰 항을 뺀 꼬리 마진도 같이 재서 평균한다
# 이유: 첫 갈림 토큰 항은 조립 판단이 아니라 프레임 뒤 음절 사전확률이라 분리해 보여야 하기 때문
def p_pair_margin(logps_fn, tok_base, form, rev):
    _m_full = []
    _m_tail = []
    for fr in FRAMES:
        _ok = tok_base.encode(fr + form)
        _rv = tok_base.encode(fr + rev)
        _lp_ok = logps_fn(_ok)
        _lp_rv = logps_fn(_rv)
        j = next(i for i in range(min(len(_ok), len(_rv))) if _ok[i] != _rv[i])
        _full = sum(_lp_ok) - sum(_lp_rv)
        _tail = (sum(_lp_ok) - _lp_ok[j - 1]) - (sum(_lp_rv) - _lp_rv[j - 1])
        _m_full.append(_full)
        _m_tail.append(_tail)
    return float(np.mean(_m_full)), float(np.mean(_m_tail))


# Role: produces the accuracy, mean margin, and binomial test p-value of one group
# Method: counts positive margins as correct and computes exactly the probability of scoring at least as well by coin flipping
# Why: whether accuracy is distinguishable from chance under a small sample must be shown by the p-value
# 역할: 한 묶음의 정확도와 마진 평균과 이항검정 p값을 낸다
# 방법: 마진 양수를 정답으로 세고 동전던지기로 그 이상 맞힐 확률을 정확히 계산한다
# 이유: 표본이 작을 때 정확도가 무작위와 구별되는지는 p값으로 보여야 하기 때문
def p_group_stats(margins):
    _n = len(margins)
    _c = sum(1 for v in margins if v > 0)
    _acc = _c / max(_n, 1) * 100.0
    _pval = sum(math.comb(_n, k) for k in range(_c, _n + 1)) / (2.0 ** _n) if _n else 1.0
    return _acc, _c, _n, _pval


# Role: builds the tier 2 perturbed ending by swapping the first two syllables of the ending
# Method: two-syllable endings are reversed entirely and three-syllable endings swap only the first two
# Why: the swapped neighbor sequences are often frequent in the corpora, which makes this the hard item set where the bigram fails
# 역할: 어미의 앞 두 음절을 뒤바꾼 2단계 교란 어미를 만든다
# 방법: 두 음절 어미는 통째로 뒤집히고 세 음절 어미는 앞 둘만 바뀐다
# 이유: 뒤바뀐 이웃 연쇄가 말뭉치에서 흔한 경우가 많아 바이그램이 넘어지는 어려운 문항이 되기 때문
def p_swap_ending(e):
    if len(e) < 2:
        return None
    return e[1] + e[0] + e[2:]


def main():
    m, tok = bc.load_from(FROZEN)
    _base = tok.base
    _vocab = m.m_vocab_size
    _forms = [(s, e, s + e, e + s) for s in STEMS for e in ENDINGS]
    print(f"[제3편 조합 일반화] 얼린 모델 {os.path.basename(FROZEN)} · 어간 {len(STEMS)}개 x 어미 {len(ENDINGS)}개 = 결합형 {len(_forms)}개 · 프레임 {len(FRAMES)}개 평균", flush=True)
    print(f"학습이 접근한 말뭉치 {len(TRAIN_CORPORA)}종 전수 훑기로 본 조합과 안 본 조합을 가른다", flush=True)

    _all_pats = [f for _, _, f, _ in _forms] + [r for _, _, _, r in _forms]
    _swap_forms = {(s, e): s + p_swap_ending(e) for s in STEMS for e in ENDINGS if p_swap_ending(e)}
    _all_pats += list(set(_swap_forms.values()))
    _counts, _bigram = p_scan_corpora(_base, _all_pats, _vocab)
    _row_sum = _bigram.sum(1)

    _valid = [(s, e, f, r) for s, e, f, r in _forms if _counts[r] == 0]
    _dropped = [(f, r, _counts[r]) for _, _, f, r in _forms if _counts[r] > 0]
    print(f"\n[최소쌍 소독] 뒤집은 열이 말뭉치에 실재하는 쌍 {len(_dropped)}개 제외 (판별력 없음)", flush=True)
    print("  제외: " + " ".join(f"{f}({r} {c}회)" for f, r, c in _dropped), flush=True)

    _seen = [(s, e, f, r) for s, e, f, r in _valid if _counts[f] > 0]
    _unseen = [(s, e, f, r) for s, e, f, r in _valid if _counts[f] == 0]
    print(f"\n[말뭉치 전수 확인] 유효쌍 {len(_valid)}개 = 본 조합 {len(_seen)}개 + 안 본 조합 {len(_unseen)}개", flush=True)
    print("  안 본 조합: " + " ".join(f for _, _, f, _ in _unseen), flush=True)

    _model_fn = lambda ids: p_model_logps(m, ids)
    _bg_fn = lambda ids: p_bigram_logps(_bigram, _row_sum, _vocab, ids)
    _rows = {}
    for name, group in (("본", _seen), ("안본", _unseen)):
        _rows[name] = []
        for s, e, f, r in group:
            _mf, _mt = p_pair_margin(_model_fn, _base, f, r)
            _bf, _bt = p_pair_margin(_bg_fn, _base, f, r)
            _rows[name].append((f, _mf, _mt, _bf, _bt))

    print("\n[최소쌍 판별] 바른 조립 대 순서 뒤집기, 프레임 평균 로그확률 마진. 꼬리는 첫 갈림 토큰 항 제외", flush=True)
    for name in ("본", "안본"):
        _R = _rows[name]
        _acc, _c, _n, _pv = p_group_stats([v[1] for v in _R])
        _acc_t, _, _, _ = p_group_stats([v[2] for v in _R])
        _acc_b, _cb, _, _pvb = p_group_stats([v[3] for v in _R])
        print(f"  {name} {_n}개 · 모델 정확도 {_acc:.1f}% ({_c}/{_n}, p={_pv:.1e}) · 꼬리 {_acc_t:.1f}% · 마진 {np.mean([v[1] for v in _R]):+.2f}", flush=True)
        print(f"      바이그램 기준선 {_acc_b:.1f}% ({_cb}/{_n}, p={_pvb:.1e}) · 마진 {np.mean([v[3] for v in _R]):+.2f}", flush=True)

    print("\n[진단 부분집합] 바이그램이 틀리는데 모델이 맞히는 쌍 (표면 통계 너머의 조립 지식)", flush=True)
    for name in ("본", "안본"):
        _win = [(f, mf, bf) for f, mf, mt, bf, bt in _rows[name] if mf > 0 and bf <= 0]
        _lose = [(f, mf, bf) for f, mf, mt, bf, bt in _rows[name] if mf <= 0 and bf > 0]
        print(f"  {name}: 모델만 정답 {len(_win)}개 " + " ".join(f for f, _, _ in _win), flush=True)
        print(f"      바이그램만 정답 {len(_lose)}개 " + " ".join(f for f, _, _ in _lose), flush=True)

    _tier2 = {}
    for name, group in (("본", _seen), ("안본", _unseen)):
        _tier2[name] = []
        for s, e, f, r in group:
            _sw = _swap_forms.get((s, e))
            if _sw is None or _counts[_sw] > 0:
                continue
            _mf, _mt = p_pair_margin(_model_fn, _base, f, _sw)
            _bf, _bt = p_pair_margin(_bg_fn, _base, f, _sw)
            _tier2[name].append((f, _sw, _mf, _bf))

    print("\n[2단계 교란] 어미 내부 음절 뒤바꿈, 두 음절 이상 어미만, 뒤바뀐 형이 말뭉치 0회인 쌍만", flush=True)
    for name in ("본", "안본"):
        _R = _tier2[name]
        if not _R:
            continue
        _acc, _c, _n, _pv = p_group_stats([v[2] for v in _R])
        _acc_b, _cb, _, _pvb = p_group_stats([v[3] for v in _R])
        print(f"  {name} {_n}개 · 모델 {_acc:.1f}% ({_c}/{_n}, p={_pv:.1e}) · 마진 {np.mean([v[2] for v in _R]):+.2f}", flush=True)
        print(f"      바이그램 {_acc_b:.1f}% ({_cb}/{_n}, p={_pvb:.1e}) · 마진 {np.mean([v[3] for v in _R]):+.2f}", flush=True)
        _win = [f for f, sw, mf, bf in _R if mf > 0 and bf <= 0]
        _lose = [f for f, sw, mf, bf in _R if mf <= 0 and bf > 0]
        print(f"      모델만 정답 {len(_win)}개 " + " ".join(_win), flush=True)
        print(f"      바이그램만 정답 {len(_lose)}개 " + " ".join(_lose), flush=True)

    print("\n[실측 요약] 무작위 기준선 50%", flush=True)
    for name, label in (("본", "본 조합  "), ("안본", "안 본 조합")):
        _R = _rows[name]
        _acc, _c, _n, _pv = p_group_stats([v[1] for v in _R])
        print(f"  {label} {_n}개 모델 {_acc:.1f}% · 바이그램 {p_group_stats([v[3] for v in _R])[0]:.1f}%", flush=True)
    print("  마진 자료 (그림용, 모델 전체마진)", flush=True)
    for name in ("본", "안본"):
        print(f"  {name}: " + " ".join(f"{v[1]:.1f}" for v in _rows[name]), flush=True)


if __name__ == "__main__":
    main()
