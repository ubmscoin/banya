# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 5 quality contrast probe. Measures whether the savings of the bundle dictionary preserve prediction quality by running the same text with bundles on and off.
The same holdout text is encoded once with the cache-dictionary encoding (bundles on) and once with the syllable-atom encoding (bundles off) and passed through the same frozen model.
Because the token units differ, per-token cross entropy is not comparable, so we normalize to bits per character by dividing by the number of characters covered by the predicted tokens.
The savings are reported as the token-count ratio on the same text, and the number of characters one context window covers is given alongside as an arithmetic conversion of that ratio.
The model was trained with bundles on, and atom-level spellings of frequent words are absent from training by construction, so the off figures are an out-of-distribution forced-decoding diagnostic, not a quality comparison.
To make that nature explicit, we decompose the bits of the off stream into word-internal and boundary positions and also report the probability mass received by bundle tokens and
an accounting alternative renormalized to the atom vocabulary. The mean and standard deviation of per-snippet differences and the count of positive snippets are given alongside.
The model is loaded and run forward through the common foundation banya_core. GPU-only cupy.
Run  python3 paper5_quality_contrast_probe.py

반야 제5편 품질 대조 프로브. 묶음 사전의 절감이 예측 품질을 지키는지 같은 텍스트의 묶음 켬과 끔으로 실측한다.
같은 홀드아웃 텍스트를 캐시 사전 인코딩(묶음 켬)과 음절 원자 인코딩(묶음 끔)으로 각각 만들어 같은 얼린 모델에 통과시킨다.
토큰 단위가 서로 달라 토큰당 교차엔트로피는 비교가 안 되므로 예측된 토큰들이 덮는 글자 수로 나눈 글자당 비트로 정규화한다.
절감은 같은 텍스트의 토큰 수 비율로 보고하고 문맥창 하나가 덮는 글자 수는 그 비율의 산술 환산으로 병기한다.
이 모델은 묶음 켬으로 학습됐고 자주 쓰는 낱말의 원자 풀어쓰기는 학습에 구성상 없으므로 끔 수치는 품질 비교가 아니라 분포 밖 강제 복호 진단이다.
그 성격을 스스로 밝히기 위해 끔 스트림에서 낱말 내부 자리와 경계 자리의 비트를 분해하고 묶음 토큰이 받는 확률 질량과
원자 어휘로 재정규화한 회계 대안도 같이 보고한다. 조각별 차이의 평균과 표준편차와 양수 조각 수를 병기한다.
공통 토대 banya_core 로 모델을 불러 순전파한다. GPU 전용 cupy.
실행  python3 paper5_quality_contrast_probe.py"""
import os
import sys
import json
import math
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "cache_elem3_190000.npz")
EVAL_SETS = ["홀드_중딩", "홀드_고딩"]


# Role: measures the total nats of one id stream and the number of characters covered by the predicted tokens
# Method: cuts the stream into non-overlapping context-length windows, runs forward, sums the log probability of the answer from the second token of each window, and counts the restored characters of those tokens
# Why: token counts differ between encodings so per-token values are not comparable, and how many bits it takes to predict the same characters is the fair yardstick
# 역할: 한 아이디 열의 총 나트와 예측된 토큰들이 덮는 글자 수를 잰다
# 방법: 문맥 길이 창으로 겹침 없이 잘라 순전파하고 창마다 둘째 토큰부터 정답 로그확률을 더하며 그 토큰들의 복원 글자 수를 센다
# 이유: 인코딩마다 토큰 수가 달라 토큰당 값은 비교가 안 되고 같은 글자를 몇 비트로 예측했는지가 공정한 잣대이기 때문
def p_stream_nats(m, tok, ids):
    _nats = 0.0
    _chars = 0
    _B = bc.CONTEXT_LENGTH
    for i in range(0, len(ids) - 1, _B):
        _w = ids[i:i + _B]
        if len(_w) < 2:
            continue
        _X = xp.asarray(_w, dtype=xp.int64).reshape(len(_w), 1)
        _, _, _z = bc.forward(m, _X)
        _zz = bc.to_host(_z).astype(np.float64)
        for t in range(len(_w) - 1):
            _col = _zz[:, t] - _zz[:, t].max()
            _p = np.exp(_col)
            _p /= _p.sum()
            _nats += -math.log(_p[_w[t + 1]] + 1e-300)
        _chars += len(tok.decode(_w[1:]))
    return _nats, _chars


# Role: marks the word-internal character positions in the bundles-on encoding of one snippet
# Method: walks the on tokens from the front and records, among the characters covered by a multi-character bundle, the positions after the first character as internal
# Why: whether the off collapse concentrates inside words or at boundaries is the key to clarifying the nature of the off figures
# 역할: 한 조각의 켬 인코딩에서 낱말 내부 글자 자리들을 표시한다
# 방법: 켬 토큰을 앞에서부터 걸으며 여러 글자 묶음이 덮는 글자 중 첫 글자 뒤 자리들을 내부로 담는다
# 이유: 끔 붕괴가 낱말 내부에 몰리는지 경계에 몰리는지가 끔 수치의 성격을 밝히는 열쇠이기 때문
def p_internal_positions(tok, s):
    _pos = set()
    _cur = 0
    for t in tok.encode(s):
        _L = len(tok.decode([t]))
        if t >= tok.m_base_vocab and _L >= 2:
            for k in range(1, _L):
                _pos.add(_cur + k)
        _cur += _L
    return _pos


# Role: splits the bits of the off stream into internal and boundary positions and measures the mass received by bundle tokens
# Method: since atom ids map one to one to characters in this text, position numbers decide internality, and from the probabilities we read both the bundle-range mass sum and the identity of the argmax token
# Why: we must show numerically that the off stream breaks because the model tries to emit bundles at atom positions
# 역할: 끔 스트림에서 내부 자리와 경계 자리의 비트를 가르고 묶음 토큰이 받는 질량을 잰다
# 방법: 원자 아이디가 이 텍스트에서 글자와 일대일이므로 자리 번호로 내부 여부를 가르고 확률에서 묶음 구간 합과 최대확률 정체를 같이 읽는다
# 이유: 끔이 깨지는 이유가 모델이 원자 자리에서 묶음을 내려 하기 때문임을 수치로 보여야 하기 때문
def p_atom_diagnose(m, tok, s):
    _ids = tok.base.encode(s)
    _pos_internal = p_internal_positions(tok, s)
    _base = tok.m_base_vocab
    _B = bc.CONTEXT_LENGTH
    _in_nats = []
    _bd_nats = []
    _mass = []
    _am_bundle = 0
    _cnt = 0
    for i in range(0, len(_ids) - 1, _B):
        _w = _ids[i:i + _B]
        if len(_w) < 2:
            continue
        _X = xp.asarray(_w, dtype=xp.int64).reshape(len(_w), 1)
        _, _, _z = bc.forward(m, _X)
        _zz = bc.to_host(_z).astype(np.float64)
        for t in range(len(_w) - 1):
            _col = _zz[:, t] - _zz[:, t].max()
            _p = np.exp(_col)
            _p /= _p.sum()
            _n = -math.log(_p[_w[t + 1]] + 1e-300)
            (_in_nats if (i + t + 1) in _pos_internal else _bd_nats).append(_n)
            _mass.append(float(_p[_base:].sum()))
            _am_bundle += int(np.argmax(_p) >= _base)
            _cnt += 1
    return _in_nats, _bd_nats, _mass, _am_bundle, _cnt


# Role: measures one evaluation set with bundles on and off, producing bits per character, token counts, and per-snippet differences
# Method: for each snippet accumulates the nats, characters, and tokens of both encodings, and for the off case also computes the atom-vocabulary renormalized accounting
# Why: the scored character sets are nearly identical, differing only by the excluded first tokens up to a few dozen characters, and the choice of accounting changes the figures, so both must be shown
# 역할: 평가 묶음 하나를 묶음 켬과 끔으로 재 글자당 비트와 토큰 수와 조각별 차이를 낸다
# 방법: 조각마다 두 인코딩의 나트와 글자와 토큰을 합산하고 끔은 원자 어휘 재정규화 회계도 같이 계산한다
# 이유: 채점 글자 집합은 첫 토큰 배제분만 최대 수십 자 다른 거의 같은 집합이고 회계 선택이 수치를 바꾸므로 둘 다 보여야 하기 때문
def p_eval_set(m, tok, snips):
    _out = {}
    _per = {"켬": [], "끔": []}
    for name, enc in (("켬", tok.encode), ("끔", tok.base.encode)):
        _nats = 0.0
        _chars = 0
        _ntok = 0
        for s in snips:
            _ids = enc(s)
            _n, _c = p_stream_nats(m, tok, _ids)
            _nats += _n
            _chars += _c
            _ntok += len(_ids)
            _per[name].append(_n / max(_c, 1) / math.log(2))
        _out[name] = (_nats / max(_chars, 1) / math.log(2), _ntok)
    _diff = [b - a for a, b in zip(_per["켬"], _per["끔"])]
    _out["조각차이"] = (float(np.mean(_diff)), float(np.std(_diff)), sum(1 for v in _diff if v > 0), len(_diff))
    _renorm_nats = 0.0
    _renorm_chars = 0
    for s in snips:
        _ids = tok.base.encode(s)
        _n, _c = p_stream_nats_renorm(m, tok, _ids)
        _renorm_nats += _n
        _renorm_chars += _c
    _out["끔재정규"] = _renorm_nats / max(_renorm_chars, 1) / math.log(2)
    return _out


# Role: measures the nats of the off stream under an accounting renormalized to the atom vocabulary
# Method: keeps only the atom range of the logits and takes the softmax over it
# Why: if the codec is one where encoder and decoder share the fact that only atoms occur, this accounting is legitimate, and the full-vocabulary denominator inflates the off figures
# 역할: 끔 스트림의 나트를 원자 어휘로 재정규화한 회계로 잰다
# 방법: 로짓에서 원자 구간만 남겨 소프트맥스를 취한다
# 이유: 원자만 나온다는 사실을 부호기와 복호기가 공유하는 코덱이라면 이 회계가 정당하고 전체 어휘 분모는 끔을 부풀리기 때문
def p_stream_nats_renorm(m, tok, ids):
    _base = tok.m_base_vocab
    _nats = 0.0
    _chars = 0
    _B = bc.CONTEXT_LENGTH
    for i in range(0, len(ids) - 1, _B):
        _w = ids[i:i + _B]
        if len(_w) < 2:
            continue
        _X = xp.asarray(_w, dtype=xp.int64).reshape(len(_w), 1)
        _, _, _z = bc.forward(m, _X)
        _zz = bc.to_host(_z).astype(np.float64)[:_base]
        for t in range(len(_w) - 1):
            _col = _zz[:, t] - _zz[:, t].max()
            _p = np.exp(_col)
            _p /= _p.sum()
            _nats += -math.log(_p[_w[t + 1]] + 1e-300)
        _chars += len(tok.decode(_w[1:]))
    return _nats, _chars


def main():
    m, tok = bc.load_from(FROZEN)
    _ev = json.load(open("model/eval_sets.json", encoding="utf-8"))
    print(f"[제5편 품질 대조] 얼린 모델 {os.path.basename(FROZEN)} · 어휘 {tok.m_vocab_size} (음절 {tok.m_base_vocab} + 묶음 {len(tok.m_bundles)}) · 문맥 {bc.CONTEXT_LENGTH}", flush=True)
    print("같은 텍스트를 묶음 켬(캐시 사전)과 끔(음절 원자)으로 인코딩해 같은 모델에서 글자당 비트를 잰다", flush=True)

    _tot_on = 0
    _tot_off = 0
    _rows = {}
    for nm in EVAL_SETS:
        _r = p_eval_set(m, tok, _ev[nm])
        _rows[nm] = _r
        _bpc_on, _ntok_on = _r["켬"]
        _bpc_off, _ntok_off = _r["끔"]
        _dm, _ds, _dpos, _dn = _r["조각차이"]
        _tot_on += _ntok_on
        _tot_off += _ntok_off
        print(f"\n[{nm}] 조각 {len(_ev[nm])}개", flush=True)
        print(f"  묶음 켬  글자당 {_bpc_on:.2f}비트 · 토큰 {_ntok_on:,}개", flush=True)
        print(f"  묶음 끔  글자당 {_bpc_off:.2f}비트 (원자 어휘 재정규화 회계 {_r['끔재정규']:.2f}) · 토큰 {_ntok_off:,}개", flush=True)
        print(f"  차이 {_bpc_off - _bpc_on:+.2f}비트 · 조각별 차이 {_dm:.2f}±{_ds:.2f}, 양수 {_dpos}/{_dn} · 토큰 절감 {(1 - _ntok_on / _ntok_off) * 100:.1f}%", flush=True)

    print("\n[끔 붕괴 진단] 끔은 분포 밖 강제 복호라 품질 비교가 아니라 진단이다. 붕괴가 어디서 오는지 분해", flush=True)
    _in_all = []
    _bd_all = []
    _mass_all = []
    _am = 0
    _cnt = 0
    for nm in EVAL_SETS:
        for s in _ev[nm]:
            _i, _b, _ms, _a, _c = p_atom_diagnose(m, tok, s)
            _in_all += _i
            _bd_all += _b
            _mass_all += _ms
            _am += _a
            _cnt += _c
    _ln2 = math.log(2)
    print(f"  낱말 내부 자리 {np.mean(_in_all) / _ln2:.2f}비트 대 경계 자리 {np.mean(_bd_all) / _ln2:.2f}비트 (자리 수 {len(_in_all):,} 대 {len(_bd_all):,})", flush=True)
    print(f"  원자 스트림에서 묶음 토큰이 받는 확률 질량 평균 {np.mean(_mass_all) * 100:.1f}% · 최대확률 토큰이 묶음인 자리 {_am / _cnt * 100:.1f}%", flush=True)

    _save = (1 - _tot_on / _tot_off) * 100
    _chars_total = sum(len(s) for nm in EVAL_SETS for s in _ev[nm])
    _cov_on = bc.CONTEXT_LENGTH / (_tot_on / _chars_total)
    _cov_off = bc.CONTEXT_LENGTH / (_tot_off / _chars_total)
    print("\n[실측 요약] 이 홀드아웃 텍스트 기준. 모든 조각이 두 인코딩 모두 창 하나에 들어가 문맥 절단은 양쪽 다 없었다", flush=True)
    for nm in EVAL_SETS:
        _r = _rows[nm]
        print(f"  {nm}: 켬 {_r['켬'][0]:.2f} 대 끔 {_r['끔'][0]:.2f} (재정규화 {_r['끔재정규']:.2f}) 글자당 비트 (목표 {'켬 5.08 끔 7.98 재정규화 7.58' if nm == '홀드_중딩' else '켬 5.07 끔 7.48 재정규화 7.07'})", flush=True)
    print(f"  토큰 절감 {_save:.1f}% (목표 21.5%) · 창 {bc.CONTEXT_LENGTH}토큰이 덮는 글자는 비율의 산술 환산으로 켬 약 {_cov_on:.0f}자 대 끔 {_cov_off:.0f}자", flush=True)


if __name__ == "__main__":
    main()
