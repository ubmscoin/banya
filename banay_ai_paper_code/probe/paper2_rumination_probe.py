# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 2 rumination probe. Applies rumination only to the data-plane embeddings of a frozen elementary model and measures three things empirically.
It reproduces whether the grammar holdout is preserved while same-kind cohesion rises, and whether the loop converges to equilibrium without divergence even when left running for long.
The model is loaded and forward-passed via the common foundation banya_core, and the Paper 2 rumination rules are taken from 논문2_되새김. GPU only (cupy).
Run  python3 paper2_rumination_probe.py

반야 제2편 되새김 프로브. 얼린 초등 모델의 데이터면 임베딩에만 되새김을 걸어 세 가지를 실측한다.
동종 밀집도가 오르는 동안 문법 홀드아웃이 보존되는지, 오래 켜 두어도 발산 없이 평형에 수렴하는지를 재현한다.
공통 토대 banya_core 로 모델을 불러 순전파하고 제2편 되새김 규칙은 논문2_되새김 에서 가져다 쓴다. GPU 전용(cupy).
실행  python3 paper2_rumination_probe.py"""
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
import paper2_rumination as paper2

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "cache_elem3_190000.npz")
CORE = 0.05
NSEED = 16
ALPHA = 0.05
SPREAD_RETAIN = 0.80
FLOOR = 0.14
COHESION_STEPS = 800
COHESION_CKPT = 100
STANDING_STEPS = 2000
STANDING_CKPT = 200
STANDING_REPROBE = 400
STANDING_DECAY = 0.005
NOISE_NUMBER = 300
PROBE_SEEDS = ["곰", "엄마", "학교", "빨강", "공"]


def p_gpu_pmax_margin(z):
    _zf = z.astype(xp.float32)
    _zf = _zf - _zf.max(0, keepdims=True)
    _pe = xp.exp(_zf)
    _pe /= _pe.sum(0, keepdims=True)
    _s = xp.sort(_pe, axis=0)
    return _s[-1], _s[-1] - _s[-2]


def p_gpu_argmax(z):
    _zf = z.astype(xp.float32)
    _zf = _zf - _zf.max(0, keepdims=True)
    _pe = xp.exp(_zf)
    _pe /= _pe.sum(0, keepdims=True)
    return _pe.argmax(0)


# Role: collects the concepts the frozen model is itself confident about into a candidate pool of utterance seeds
# Method: forward-passes the training stream window by window, selects the positions with large prediction margins, and keeps only the concept tokens at those positions in the pool
# Why: rumination seeds are drawn from the model's own confidence rather than chosen by hand, serving as candidates for frequency-proportional utterance
# 역할: 얼린 모델이 스스로 확신하는 개념들을 발화 씨앗 후보 풀로 모은다
# 방법: 학습 스트림을 창 단위로 순전파해 예측 여유가 큰 자리들을 골라 그 자리의 개념 토큰만 풀에 담는다
# 이유: 되새김 씨앗을 사람이 고르지 않고 모델의 확신도에서 뽑아 빈도 비례 발화의 후보로 삼기 위해서다
def p_core_seed_pool(m, tok, stream, base, block):
    _scored = []
    _ncol = min(20, len(stream) // block)
    for c in range(_ncol):
        _col = stream[c * block:(c + 1) * block]
        _X = xp.asarray(_col, dtype=xp.int64).reshape(block, 1)
        _cache, _aD, _z = bc.forward(m, _X)
        _pmax, _margin = p_gpu_pmax_margin(_z)
        _am = bc.to_host(p_gpu_argmax(_z))
        _mg = bc.to_host(_margin)
        for t in range(block):
            _scored.append((float(_mg[t]), int(_am[t]), int(_col[t])))
    _scored.sort(key=lambda x: -x[0])
    _k = max(1, int(0.15 * len(_scored)))
    _pool = set()
    for _mgv, _amtok, _ctok in _scored[:_k]:
        for t in (_amtok, _ctok):
            if t >= base and paper2.firing_concept(tok, t):
                _pool.add(t)
    return sorted(_pool)


# Role: measures next-character prediction difficulty on evaluation text not used in training to see whether grammar stays intact
# Method: forward-passes each snippet, collects the probabilities of the correct tokens, and converts the mean information per character into bits
# Why: if this holdout is preserved even while concepts are being clustered, it means rumination does not break the ability to speak
# 역할: 학습에 쓰지 않은 평가 텍스트의 다음 글자 예측 난이도를 재 문법이 살아 있는지 본다
# 방법: 각 조각을 순전파해 정답 토큰의 확률을 모아 글자당 평균 정보량을 비트로 환산한다
# 이유: 개념을 뭉치는 동안에도 이 홀드아웃이 유지되면 되새김이 말하는 능력을 깨뜨리지 않음을 뜻한다
def p_holdout_bpc(m, tok, snips, block):
    _total_nats = 0.0
    _total_chars = 0
    for s in snips:
        _ids = tok.encode(s)[:block + 1]
        if len(_ids) < 2:
            continue
        _X = xp.asarray(_ids[:-1], dtype=xp.int64).reshape(-1, 1)
        _Y = _ids[1:]
        _cache, _aD, _z = bc.forward(m, _X)
        _zc = bc.to_host(_z).astype(np.float64)
        for t in range(_zc.shape[1]):
            _c = _zc[:, t] - _zc[:, t].max()
            _p = np.exp(_c)
            _p /= _p.sum()
            _total_nats += -math.log(_p[_Y[t]] + 1e-12)
        _total_chars += len(s)
    return _total_nats / max(_total_chars, 1) / math.log(2)


# Role: gauges the robustness of core knowledge as the mean prediction margin of the top concepts
# Method: forward-passes the probe stream, takes the gap between the top and runner-up probabilities at each position, and averages the largest few
# Why: to confirm that the confidence of core concepts does not collapse even while rumination tidies the periphery
# 역할: 상위 개념들의 예측 여유 평균으로 핵심 지식의 견고함을 잰다
# 방법: 프로브 스트림을 순전파해 자리별 확률 최대와 차순위의 차를 구하고 큰 순 상위 몇 개의 평균을 낸다
# 이유: 되새김으로 주변을 정리해도 핵심 개념의 확신이 무너지지 않는지 확인하기 위해서다
def p_core_margin(m, probeX, k):
    _cache, _aD, _z = bc.forward(m, probeX)
    _pmax, _margin = p_gpu_pmax_margin(_z)
    _mg = bc.to_host(_margin).astype(np.float64)
    return round(float(_mg[np.argsort(-_mg)[:k]].mean()), 3)


# Role: measures within-group cohesion and between-group separation of concept groups by cosine similarity
# Method: normalizes the data-plane embeddings, taking the mean cosine inside each group as cohesion and the mean cosine among group representatives as separation
# Why: to see at a glance whether rumination packs same-kind concepts tightly while keeping different kinds from mixing
# 역할: 같은 개념군 안의 밀집도와 다른 개념군 사이의 분리도를 코사인으로 잰다
# 방법: 데이터면 임베딩을 정규화해 각 군 내부 평균 코사인을 밀집도로 군 대표끼리 평균 코사인을 분리도로 낸다
# 이유: 되새김이 동종은 촘촘히 뭉치면서 이종은 섞이지 않게 유지하는지를 한눈에 보기 위해서다
def p_cohesion(m, probe_groups):
    _norm = bc.to_host(m.m_mat_w_data_axis / (xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9))
    _withins = []
    _reps = []
    for g in probe_groups:
        _V = _norm[:, g]
        _sim = _V.T @ _V
        _n = len(g)
        _withins.append((_sim.sum() - _n) / (_n * (_n - 1)))
        _reps.append(g[0])
    _R = _norm[:, _reps]
    _bsim = _R.T @ _R
    _nb = len(_reps)
    _between = (_bsim.sum() - _nb) / (_nb * (_nb - 1))
    return round(float(np.mean(_withins)), 3), round(float(_between), 3)


def p_build_probe_groups(frozen, tok):
    _groups = []
    for w in PROBE_SEEDS:
        _ids = tok.encode(w)
        if _ids:
            _g = [int(x) for x in paper2.firing_group(frozen, tok, _ids[-1])][:10]
            if len(_g) >= 3:
                _groups.append(_g)
    return _groups


# Role: reproduces rumination experiment 5.1, running 800 steps of pulling plus global renormalization and measuring same-kind cohesion and the grammar holdout
# Method: draws seeds from the frequency-confidence pool, pulls each same-kind group toward its centroid, and restores the concept-layer norms to the original every step
# Why: to verify empirically that compression and grammar preservation hold simultaneously, with the grammar holdout preserved while same-kind cohesion rises
# 역할: 되새김 5.1 실험 재현. 당김과 전역 재정규화로 800스텝 돌려 동종 밀집도와 문법 홀드아웃을 잰다
# 방법: 빈도 확신 풀에서 씨앗을 뽑아 동종군을 무게중심으로 당기고 매 스텝 개념층 노름을 원본으로 되돌린다
# 이유: 동종 밀집도가 오르는 동안 문법 홀드아웃이 보존되는 압축과 문법 보존의 동시 성립을 실측하기 위해서다
def p_experiment_cohesion(tok, base, block, run_stream, probeX, K, ev, probe_groups, norm0):
    print(f"[5.1 동종 되새김] 당김 {ALPHA} · 재정규화 켬 · 프로브군 {len(probe_groups)} · {COHESION_STEPS}스텝", flush=True)
    _frozen, _ = bc.load_from(FROZEN)
    _frC = round(p_holdout_bpc(_frozen, tok, ev["홀드_중딩"], block), 3)
    _w0, _b0 = p_cohesion(_frozen, probe_groups)
    print(f"  [F 프로즌] 동종밀집 {_w0} 이종분리 {_b0} 홀드중등 {_frC} 핵강도 {p_core_margin(_frozen, probeX, K)}", flush=True)
    _pool = p_core_seed_pool(_frozen, tok, run_stream, base, block)
    _m, _ = bc.load_from(FROZEN)
    np.random.seed(1)
    _last = (_w0, _b0, _frC)
    for step in range(COHESION_STEPS):
        for _ in range(NSEED):
            _s = int(np.random.choice(_pool))
            _g = paper2.firing_group(_m, tok, _s)
            if len(_g) < 2:
                continue
            _group_gpu = xp.asarray(sorted(set(int(x) for x in _g)), dtype=xp.int64)
            _w_data_axis_group = _m.m_mat_w_data_axis[:, _group_gpu]
            _centroid = _w_data_axis_group.mean(1, keepdims=True)
            _m.m_mat_w_data_axis[:, _group_gpu] = _w_data_axis_group + ALPHA * (_centroid - _w_data_axis_group)
        _m.m_mat_w_data_axis *= (norm0 / (xp.linalg.norm(_m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9))
        if (step + 1) % COHESION_CKPT == 0:
            _hc = round(p_holdout_bpc(_m, tok, ev["홀드_중딩"], block), 3)
            _cm = p_core_margin(_m, probeX, K)
            _wv, _bv = p_cohesion(_m, probe_groups)
            _last = (_wv, _bv, _hc)
            print(f"  step {step+1}: 동종밀집 {_wv} 이종분리 {_bv} 홀드중등 {_hc} (Δ{_hc-_frC:+.3f}) 핵강도 {_cm}", flush=True)
    print(f"  [결과] 동종밀집 {_w0} ▶ {_last[0]} · 홀드중등 {_frC} ▶ {_last[2]} · 이종분리 {_last[1]} 유지", flush=True)
    return _w0, _last[0], _frC, _last[2]


# Role: reproduces rumination experiment 5.3, leaving the loop running for 2000 steps with only pulling and original-norm decay and measuring the convergence of the noise norm
# Method: decays only the non-core periphery every step, pulls the firing groups and restores them to their original norms, and re-measures the core pool periodically
# Why: to show that even without any target value the loop settles by itself into the equilibrium where pulling and decay balance, without diverging
# 역할: 되새김 5.3 실험 재현. 당김과 원본 감쇠만으로 2000스텝 오래 켜 두고 잡음 노름의 수렴을 잰다
# 방법: 핵 아닌 주변층만 매 스텝 감쇠시키고 발화군은 당겨 원노름으로 되돌리며 핵풀을 주기로 재측정한다
# 이유: 목표값을 주지 않았는데도 당김과 감쇠가 비기는 평형에 스스로 안착해 발산하지 않음을 보이기 위해서다
def p_experiment_standing(tok, base, block, run_stream, probeX, K, ev, probe_groups, norm0):
    print(f"[5.3 상시루프] 당김 {ALPHA} · 감쇠 {STANDING_DECAY} · 재probe {STANDING_REPROBE}마다 · {STANDING_STEPS}스텝", flush=True)
    _frozen, _ = bc.load_from(FROZEN)
    _noise = [j for j in range(base, tok.m_vocab_size) if paper2.firing_concept(tok, j)][:NOISE_NUMBER]
    _noise_arr = xp.asarray(_noise, dtype=xp.int64)
    _n0_noise = float(bc.to_host(xp.linalg.norm(_frozen.m_mat_w_data_axis[:, _noise_arr], axis=0)).mean())
    _frC = round(p_holdout_bpc(_frozen, tok, ev["홀드_중딩"], block), 3)
    _w0, _b0 = p_cohesion(_frozen, probe_groups)
    print(f"  [F 프로즌] 동종밀집 {_w0} 홀드중등 {_frC} 핵강도 {p_core_margin(_frozen, probeX, K)} 잡음노름 1.00", flush=True)
    _m, _ = bc.load_from(FROZEN)
    np.random.seed(1)
    _pool = p_core_seed_pool(_m, tok, run_stream, base, block)
    _poolset = set(_pool)
    _periphery_arr = xp.asarray([j for j in range(base, tok.m_vocab_size) if j not in _poolset], dtype=xp.int64)
    _last_noise = 1.0
    for step in range(STANDING_STEPS):
        if step > 0 and step % STANDING_REPROBE == 0:
            _pool = p_core_seed_pool(_m, tok, run_stream, base, block)
            _poolset = set(_pool)
            _periphery_arr = xp.asarray([j for j in range(base, tok.m_vocab_size) if j not in _poolset], dtype=xp.int64)
        _m.m_mat_w_data_axis[:, _periphery_arr] *= (1.0 - STANDING_DECAY)
        for _ in range(NSEED):
            _s = int(np.random.choice(_pool))
            _g = paper2.firing_group(_m, tok, _s)
            if len(_g) < 2:
                continue
            _group_gpu = xp.asarray(sorted(set(int(x) for x in _g)), dtype=xp.int64)
            _w_data_axis_group = _m.m_mat_w_data_axis[:, _group_gpu]
            _centroid = _w_data_axis_group.mean(1, keepdims=True)
            _pulled = _w_data_axis_group + ALPHA * (_centroid - _w_data_axis_group)
            _m.m_mat_w_data_axis[:, _group_gpu] = _pulled * (norm0[:, _group_gpu] / (xp.linalg.norm(_pulled, axis=0, keepdims=True) + 1e-9))
        if (step + 1) % STANDING_CKPT == 0:
            _hc = round(p_holdout_bpc(_m, tok, ev["홀드_중딩"], block), 3)
            _cm = p_core_margin(_m, probeX, K)
            _wv, _bv = p_cohesion(_m, probe_groups)
            _last_noise = float(bc.to_host(xp.linalg.norm(_m.m_mat_w_data_axis[:, _noise_arr], axis=0)).mean()) / _n0_noise
            print(f"  step {step+1}: 홀드 {_hc}(Δ{_hc-_frC:+.2f}) 핵 {_cm} 동종밀집 {_wv} 잡음노름 {_last_noise:.2f} 풀 {len(_pool)}", flush=True)
    print(f"  [결과] 잡음노름 1.00 ▶ {_last_noise:.2f} 수렴 · 동종밀집 {_wv} · 핵강도 {_cm} 유지", flush=True)
    return _last_noise


def main():
    _frozen, tok = bc.load_from(FROZEN)
    base = tok.m_base_vocab
    block = bc.CONTEXT_LENGTH
    run_stream = np.load("model/stream_train.npy")
    probeX = xp.asarray(run_stream[:block * 4].reshape(4, block).T)
    K = max(1, int(CORE * probeX.shape[0] * probeX.shape[1]))
    ev = json.load(open("model/eval_sets.json", encoding="utf-8"))
    probe_groups = p_build_probe_groups(_frozen, tok)
    norm0 = xp.linalg.norm(_frozen.m_mat_w_data_axis, axis=0, keepdims=True)
    print(f"[제2편 되새김] 얼린 모델 {os.path.basename(FROZEN)} · H {bc.HIDDEN_SIZE} · vocab {tok.m_vocab_size} · 프로브군 {len(probe_groups)}", flush=True)
    _c0, _c1, _h0, _h1 = p_experiment_cohesion(tok, base, block, run_stream, probeX, K, ev, probe_groups, norm0)
    _noise = p_experiment_standing(tok, base, block, run_stream, probeX, K, ev, probe_groups, norm0)
    print("\n[실측 요약]", flush=True)
    print(f"  동종 밀집도 {_c0} -> {_c1} (목표 0.127 -> 0.167)", flush=True)
    print(f"  홀드아웃 보존 {_h0} -> {_h1} (목표 5.006 -> 5.087)", flush=True)
    print(f"  평형 잡음노름 1.0 -> {_noise:.2f} 수렴 (목표 1.0 -> 0.14)", flush=True)


if __name__ == "__main__":
    main()
