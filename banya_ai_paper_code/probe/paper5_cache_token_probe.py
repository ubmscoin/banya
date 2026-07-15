# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 5 cache tokenization probe. Measures three things on the frozen elementary-stage cache model.
The vocabulary composition of the two-layer cache dictionary, the structural cost of how many times fewer parameters the low-rank head uses compared with a dense head,
and whether the operate axis of the bundle-layer tokens, like the syllable layer, acts as consistent operators with coherent directions that are mutually orthogonal.
The model is loaded and run forward through the common foundation banya_core, and the Paper 5 cache mechanism is imported from 논문5_캐시토큰. GPU-only cupy.
Run  python3 paper5_cache_token_probe.py

반야 제5편 캐시 토큰화 프로브. 초등 단계 얼린 캐시 모델에서 세 가지를 실측한다.
2층 캐시 사전의 어휘 구성과, 로우랭크 헤드가 밀집 헤드 대비 몇 배 적은 파라미터를 쓰는지의 구조적 비용과,
묶음층 토큰의 연산면도 음절층처럼 일관 연산자로 방향이 일관되고 서로 직교하는지를 잰다.
공통 토대 banya_core 로 모델을 불러 순전파하고 제5편 캐시 메커니즘은 논문5_캐시토큰 에서 가져다 쓴다. GPU 전용 cupy.
실행  python3 paper5_cache_token_probe.py"""
import os
import sys
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper5_cache_token as paper5

FROZEN = os.path.join(_CODE, "model", "cache_elem3_190000.npz")
OP_TOP = 40
DATA_TOP = 200
DATA_CONSIST = 25
DATA_ORTHO = 20


# Role: builds a candidate pool keeping only tokens alive on at least one of the data and operate axes
# Method: selects tokens whose summed norm over the two axes exceeds a threshold, filtering out dead tokens that have not yet matured
# Why: tokens with near-zero norm have no defined direction, and including them in operator direction measurement adds noise
# 역할: 데이터면과 연산면 중 하나라도 살아있는 토큰만 남겨 후보 풀을 만든다
# 방법: 두 면 노름의 합이 문턱을 넘는 토큰만 골라 아직 안 익은 죽은 토큰을 걸러낸다
# 이유: 노름이 0 에 가까운 토큰은 방향이 정의되지 않아 연산자 방향 측정에 끼면 잡음이 되기 때문
def p_live_pool(pool, data_norm, operate_norm):
    return pool[(data_norm[pool] + operate_norm[pool]) > 1e-6]


# Role: picks operation candidate tokens and data sample tokens
# Method: takes the top tokens by operate-axis norm as operation candidates and randomly samples data tokens from the top pool by data-axis norm
# Why: operator emergence only shows when we observe in which direction strongly operating tokens bend various data
# 역할: 연산 후보 토큰과 데이터 표본 토큰을 뽑는다
# 방법: 연산면 노름 큰 순으로 상위 몇 개를 연산 후보로 삼고 데이터면 노름 큰 상위 풀에서 무작위로 데이터 표본을 뽑는다
# 이유: 연산이 강한 토큰이 여러 데이터를 어느 방향으로 꺾는지를 봐야 연산자 창발이 드러나기 때문
def p_pick_ops_data(live, data_norm, operate_norm, rng, n_data):
    _ops = live[np.argsort(-operate_norm[live])][:OP_TOP]
    _pool = live[np.argsort(-data_norm[live])][:DATA_TOP]
    _pick = rng.choice(len(_pool), size=min(n_data, len(_pool)), replace=False)
    return _ops, _pool[_pick]


# Role: collects the direction vectors by which one operation token bends multiple data tokens
# Method: places the operation token in the first position and a data token in the second, runs forward with the operate axis on and off, and takes the difference of the second-position hidden states as the bending direction
# Why: the operate axis of the preceding token shifts one position and is injected into the gate of the next token, so the on-off difference is the pure direction in which the operation bent the data
# 역할: 한 연산 토큰이 여러 데이터 토큰을 꺾는 방향 벡터들을 수집
# 방법: 앞자리에 연산 토큰 뒷자리에 데이터 토큰을 놓고 연산면 켬과 끔으로 순전파해 뒷자리 히든의 차이를 꺾음 방향으로 삼는다
# 이유: 직전 토큰의 연산면이 한 칸 이동해 다음 토큰의 게이트로 주입되므로 그 켬끔 차이가 연산이 데이터를 꺾은 순수 방향이기 때문이다
def p_bend_directions(m, op_id, data_ids):
    _batch = len(data_ids)
    _rows = np.stack([np.full(_batch, op_id, dtype=np.int64), np.asarray(data_ids, dtype=np.int64)])
    _X = xp.asarray(_rows)
    paper5.set_operate_axis(True)
    _cache_on, _on, _ = bc.forward(m, _X)
    paper5.set_operate_axis(False)
    _cache_off, _off, _ = bc.forward(m, _X)
    paper5.set_operate_axis(True)
    _v = _on.reshape(bc.HIDDEN_SIZE, 2, _batch)[:, 1, :] - _off.reshape(bc.HIDDEN_SIZE, 2, _batch)[:, 1, :]
    _vh = bc.to_host(_v).astype(np.float64)
    _norms = np.linalg.norm(_vh, axis=0)
    _keep = _norms > 1e-9
    _unit = _vh[:, _keep] / _norms[_keep]
    return _unit.T


# Role: measures directional consistency, how coherently the same operation bends different data
# Method: for each operation takes the mean pairwise absolute cosine of its per-data bending directions as its consistency and averages over all operation candidates
# Why: acting in a consistent direction on any data makes it a coherent operator, while bending each datum differently means it is not an operation
# 역할: 같은 연산이 여러 데이터를 얼마나 일관된 방향으로 꺾는지 방향 일관성을 잰다
# 방법: 연산마다 데이터별 꺾음 방향의 쌍별 절대 코사인 평균을 일관성으로 삼고 연산 후보 전체를 평균한다
# 이유: 어떤 데이터에도 일관 방향으로 작용하면 일관 연산자이고 데이터마다 제각각이면 연산이 아니기 때문이다
def p_direction_consistency(m, op_ids, data_ids):
    _consist = []
    for o in op_ids:
        _unit = p_bend_directions(m, int(o), data_ids)
        if _unit.shape[0] < 3:
            continue
        _sim = _unit @ _unit.T
        _n = _unit.shape[0]
        _pair = np.abs(_sim[np.triu_indices(_n, 1)])
        _consist.append(float(_pair.mean()))
    return float(np.mean(_consist))


# Role: measures operation orthogonality, whether different operations overlap in direction or stay orthogonal
# Method: normalizes each operation's representative bending direction, computes pairwise absolute cosines, and reports the near-orthogonal ratio and the same-direction ratio
# Why: if each operation carries its own direction, the operation kinds have separated out without any labels
# 역할: 서로 다른 연산끼리 방향이 겹치는지 직교하는지 연산 직교성을 측정
# 방법: 연산별 대표 꺾음 방향을 정규화해 쌍별 절대 코사인을 구하고 거의 직교 비율과 같은 방향 비율을 낸다
# 이유: 연산마다 고유한 방향을 가지면 연산 종류가 라벨 없이 갈라져 나온 것이기 때문
def p_orthogonality(m, op_ids, data_ids):
    _reps = []
    for o in op_ids:
        _unit = p_bend_directions(m, int(o), data_ids)
        if _unit.shape[0] >= 3:
            _reps.append(_unit.mean(0))
    _R = np.array([r / (np.linalg.norm(r) + 1e-9) for r in _reps])
    _sim = np.abs(_R @ _R.T)
    np.fill_diagonal(_sim, 0.0)
    _pair = _sim[np.triu_indices(len(_R), 1)]
    _ortho = float((_pair < 0.15).mean()) * 100.0
    _same = float((_pair > 0.5).mean()) * 100.0
    return _ortho, _same


def main():
    m, tok = bc.load_from(FROZEN)
    _V = tok.m_vocab_size
    _base = tok.m_base_vocab
    _bundles = len(tok.m_bundles)
    _step = int(m.t)
    print(f"[제5편 캐시 토큰화] 얼린 모델 {os.path.basename(FROZEN)} · step {_step:,} · H {bc.HIDDEN_SIZE} · 랭크 {bc.HEAD_RANK}", flush=True)

    print("\n[어휘 구성] 2층 캐시 사전 (음절 원자 위에 묶음 사전)", flush=True)
    print(f"   음절 원자 {_base} + 묶음 사전 {_bundles} = 어휘 {_V}", flush=True)

    _dense, _lowrank, _ratio = paper5.head_param_cost(_V, bc.HIDDEN_SIZE, bc.HEAD_RANK)
    _actual = int(m.m_mat_w_head_a.size + m.m_mat_w_head_b.size)
    print("\n[헤드 비용] 밀집 헤드 대 로우랭크 헤드 파라미터 (구조적 계산)", flush=True)
    print(f"   밀집 {_dense:,}개 (약 {_dense / 1e4:.0f}만) · 로우랭크 {_lowrank:,}개 (약 {_lowrank / 1e4:.0f}만) · {_ratio:.1f}배 적음", flush=True)
    print(f"   체크포인트 실제 로우랭크 헤드 파라미터 {_actual:,}개 (a {tuple(m.m_mat_w_head_a.shape)} + b {tuple(m.m_mat_w_head_b.shape)})", flush=True)

    _operate_norm = bc.to_host(xp.linalg.norm(m.m_mat_w_operate_axis, axis=0))
    _data_norm = bc.to_host(xp.linalg.norm(m.m_mat_w_data_axis, axis=0))
    _layers = [("음절", np.arange(_base)), ("묶음", np.arange(_base, _V))]
    _base_cos = 1.0 / np.sqrt(bc.HIDDEN_SIZE)
    _rng = np.random.RandomState(0)

    _consist = {}
    for nm, pool in _layers:
        _live = p_live_pool(pool, _data_norm, _operate_norm)
        _ops, _data = p_pick_ops_data(_live, _data_norm, _operate_norm, _rng, DATA_CONSIST)
        _consist[nm] = p_direction_consistency(m, _ops, _data)
    _ortho = {}
    for nm, pool in _layers:
        _live = p_live_pool(pool, _data_norm, _operate_norm)
        _ops, _data = p_pick_ops_data(_live, _data_norm, _operate_norm, _rng, DATA_ORTHO)
        _ortho[nm] = p_orthogonality(m, _ops, _data)

    print("\n[방향 일관성] 같은 연산이 여러 데이터를 같은 방향으로 꺾나 (절대 코사인 평균, 무작위 대비 배수)", flush=True)
    print(f"   무작위 {_base_cos:.3f}", flush=True)
    for nm, _ in _layers:
        _c = _consist[nm]
        print(f"   {nm}층 연산자 {_c:.3f} · {_c / _base_cos:.1f}배", flush=True)

    print("\n[연산 직교성] 서로 다른 연산끼리 방향이 겹치나 직교하나 (연산쌍 절대 코사인)", flush=True)
    for nm, _ in _layers:
        _o, _s = _ortho[nm]
        print(f"   {nm}층 거의 직교쌍(<0.15) {_o:.0f}% · 같은 방향쌍(>0.5) {_s:.0f}%", flush=True)

    print("\n[실측 요약]", flush=True)
    print(f"  어휘 {_V} (음절 {_base} + 묶음 {_bundles})  (목표 10724)", flush=True)
    print(f"  로우랭크 헤드 {_ratio:.1f}배 절감 (밀집 약 {_dense / 1e4:.0f}만 대 로우랭크 약 {_lowrank / 1e4:.0f}만)  (목표 7.3배, 1098만 대 150만)", flush=True)
    print(f"  묶음층 방향 일관성 {_consist['묶음']:.3f} 대 무작위 {_base_cos:.3f} = {_consist['묶음'] / _base_cos:.1f}배  (목표 8.1배)", flush=True)
    print(f"  음절층 방향 일관성 {_consist['음절'] / _base_cos:.1f}배 · 직교쌍 음절 {_ortho['음절'][0]:.0f}% 묶음 {_ortho['묶음'][0]:.0f}%  (목표 음절 7.5배, 직교 79/77%)", flush=True)


if __name__ == "__main__":
    main()
