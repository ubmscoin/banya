# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 1 small-scale contrast probe. Measures in a small setting whether the hand-built exact-transpose engine learns with the same gradients as standard backpropagation.
Two trainers run side by side with the same small architecture, the same initial weights, and the same batch sequence.
One side is the hand-built engine of banya_core (forward pass, head update, block credit chain) and the other is a reference implementation using plain layer-by-layer standard backpropagation.
The gradients of the reference implementation are first verified independently against finite differences, and then the loss trajectories, final losses, and wall times of the two trainers are compared.
If the two trajectories coincide within numerical error, this is a small-scale demonstration of the Paper 1 claim that the manually derived transpose yields the same gradients as standard backpropagation.
The reference implementation works with both numpy and cupy, so wall times on both the CPU and the same GPU are reported together. PyTorch is not used because it is absent from this environment.
Run  python3 paper1_small_contrast_probe.py

반야 제1편 소형 대조 프로브. 자작 정확한 전치 엔진이 표준 역전파와 같은 기울기로 학습되는지 소형 설정에서 실측한다.
같은 소형 아키텍처와 같은 초기 가중치와 같은 배치 열로 두 학습기를 나란히 돌린다.
한쪽은 banya_core 의 자작 엔진(순전파, 헤드 갱신, 블록 신용 사슬)이고 다른쪽은 층별 표준 역전파를 그대로 쓴 기준 구현이다.
기준 구현의 기울기는 유한차분과 먼저 대조해 독립으로 검증하고 그 다음 두 학습기의 손실 궤적과 최종 손실과 실행 시간을 비교한다.
두 궤적이 수치 오차 안에서 겹치면 수동 유도 전치가 표준 역전파와 같은 기울기라는 제1편 주장의 소형 실증이 된다.
기준 구현은 numpy 와 cupy 겸용이라 CPU 와 같은 GPU 양쪽의 실행 시간을 같이 보고한다. 파이토치는 이 환경에 없어 쓰지 않는다.
실행  python3 paper1_small_contrast_probe.py"""
import os
import sys
import time
import numpy as np
import cupy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc

bc.HIDDEN_SIZE = 256
bc.SCALE_LENGTH_MIN = 32
bc.NUMBER_BLOCK = 4
bc.FFN_DEPTH = 2
bc.NUMBER_TIME_MIX_HEAD = 4
bc.CONTEXT_LENGTH = 64
bc.USE_GOONGHAP = False
bc.USE_BITOKEN = False
bc.USE_GATE = False
bc.USE_NORM = True
bc.USE_RELATIVE_MIX = True
bc.TIME_MIX_USE_FFT = False
bc.BLOCKWISE_CAUSAL_MIX = True
bc.USE_LOWRANK = True
bc.HEAD_RANK = 64
bc.USE_WEIGHT_TIE = False
bc.HEAD_USE_FP16 = False

import paper1_engine as paper1

os.chdir(bc._ROOT)
CORPUS = "banya_world_data/baby.npy"
VOCAB = 2724
SEED = 0
STEPS = 300
WARM = 3
BATCH = 16
LR = bc.LEARNING_RATE_ADAM
CHECK_AT = [0, 49, 99, 199, 299]


# Role: dumps every trainable parameter of the hand-built engine model to numpy as the initial values of the reference implementation
# Method: copies the value and the Adam moment buffers of each parameter into a dictionary
# Why: the two trainers must start from the same point so that trajectory differences reflect gradient differences only
# 역할: 자작 엔진 모델의 학습 파라미터 전부를 numpy로 떠서 기준 구현의 초기값으로 만든다
# 방법: 파라미터마다 값과 아담 모멘트 버퍼를 복사해 사전에 담는다
# 이유: 두 학습기가 같은 자리에서 출발해야 궤적 차이가 기울기 차이만을 반영하기 때문
def p_snapshot(m):
    _P = {}
    for k in ("m_mat_w_data_axis", "m_mat_w_position", "m_mat_w_lag", "m_mat_w_filter",
              "m_vec_w_filter_bias", "m_mat_w_mix", "m_vec_w_norm_gain",
              "m_mat_w_head_a", "m_mat_w_head_b", "m_vec_w_head_bias"):
        _P[k] = bc.to_host(getattr(m, k)).copy()
    return _P


# Role: builds the parameters and Adam buffers of the reference implementation
# Method: moves the snapshot to the given array module and dtype and starts the Adam moments at zero
# Why: the engine-side Adam buffers also start at zero, so the buffers must match for the trajectories to coincide from the very first step
# 역할: 기준 구현의 파라미터와 아담 버퍼를 만든다
# 방법: 스냅숏을 지정한 배열 모듈과 자료형으로 옮기고 아담 모멘트는 0 으로 시작한다
# 이유: 엔진 쪽 아담 버퍼도 0 에서 시작하므로 버퍼까지 같아야 한 스텝째부터 궤적이 겹치기 때문
def p_ref_init(P, ap, dtype):
    _R = {"t": 0}
    for k, v in P.items():
        _R[k] = ap.asarray(v, dtype=dtype)
        _R[k + "_m"] = ap.zeros_like(_R[k])
        _R[k + "_v"] = ap.zeros_like(_R[k])
    return _R


# Role: unfolds the lag kernel into a lower-triangular Toeplitz matrix
# Method: indexes the kernel by position differences and masks the future direction to zero
# Why: the reference implementation must produce the same values as toep_build of banya_core for the forward passes to coincide
# 역할: 거리 커널을 하위삼각 토플리츠 행렬로 편다
# 방법: 위치 차이를 인덱스로 커널을 골라 담고 미래 방향은 0 으로 막는다
# 이유: banya_core 의 toep_build 와 같은 값을 기준 구현에서 만들어야 순전파가 겹치기 때문
def p_ref_toep(ap, wlag, T):
    _ti = ap.arange(T)
    _lag = _ti[:, None] - _ti[None, :]
    return wlag[..., ap.clip(_lag, 0, T - 1)] * (_lag >= 0)


# Role: forward pass of the reference implementation, producing the loss and a cache of intermediates for backpropagation
# Method: computes in order the embedding, then per block the Toeplitz time mixing with residual, two filter-ladder layers, and RMS normalization, then the low-rank head and softmax
# Why: rewriting the same equations as the banya_core forward pass using only standard array operations is what qualifies this as the reference
# 역할: 기준 구현의 순전파. 손실과 역전파에 쓸 중간값 캐시를 낸다
# 방법: 임베딩, 블록마다 토플리츠 시간혼합과 잔차와 필터 사다리 두 겹과 RMS 정규화, 로우랭크 헤드, 소프트맥스 순서로 계산한다
# 이유: banya_core 순전파와 같은 수식을 표준 배열 연산만으로 다시 쓴 것이 기준의 자격이기 때문
def p_ref_forward(ap, R, ids, y_flat, dtype):
    T, B = ids.shape
    _Mt = T * B
    _H = bc.HIDDEN_SIZE
    _NH = bc.NUMBER_TIME_MIX_HEAD
    _Hh = _H // _NH
    _x = R["m_mat_w_data_axis"][:, ids] + R["m_mat_w_position"][:, :T, None]
    _C = p_ref_toep(ap, R["m_mat_w_lag"], T).astype(dtype)
    _cache = []
    for bl in range(bc.NUMBER_BLOCK):
        _xr = _x.reshape(_NH, _Hh, T, B)
        _xs = ap.ascontiguousarray(_xr.transpose(0, 2, 1, 3)).reshape(_NH, T, _Hh * B)
        _xm = ap.ascontiguousarray((_C[bl] @ _xs).reshape(_NH, T, _Hh, B).transpose(0, 2, 1, 3)).reshape(_H, _Mt)
        _h = _x.reshape(_H, _Mt) + _xm
        _fa = []
        _pyr = []
        for i in range(bc.FFN_DEPTH):
            _fa.append(_h)
            s = bc.blk_scale(bl + i)
            _L = _H >> s
            _F = 1 << s
            _parts = [_h]
            for k in range(s):
                _qr = _parts[k].reshape(-1, 2, _Mt)
                _parts.append(_qr[:, 0] * R["m_mat_w_mix"][bl, i, k, 0] + _qr[:, 1] * R["m_mat_w_mix"][bl, i, k, 1])
            _q = _parts[s]
            _qt = ap.tile(_q, (_F, 1))
            _brep = ap.repeat(R["m_vec_w_filter_bias"][bl, i, :_F], _L)
            _y = ap.tanh(R["m_mat_w_filter"][bl, i][:, None] * _qt + _brep[:, None])
            _pyr.append((_parts, _q, _qt, _y, s, _L, _F))
            _h = _y
        _hres = _fa[0] + _h
        _r = 1.0 / ap.sqrt((_hres * _hres).mean(0, keepdims=True) + 1e-6)
        _yn = _hres * _r * R["m_vec_w_norm_gain"][bl]
        _cache.append((_xr, _xs, _fa, _pyr, _hres, _r))
        _x = _yn.reshape(_H, T, B)
    _aD = _x.reshape(_H, _Mt)
    _z = R["m_mat_w_head_a"] @ (R["m_mat_w_head_b"] @ _aD) + R["m_vec_w_head_bias"]
    _p = ap.exp(_z - _z.max(0, keepdims=True))
    _p /= _p.sum(0, keepdims=True)
    _ar = ap.arange(_Mt)
    _ce = float(-ap.mean(ap.log(_p[y_flat, _ar] + 1e-9)))
    return _ce, _p, _aD, _C, _cache


# Role: standard backpropagation of the reference implementation, producing the gradients of all parameters as a dictionary
# Method: walks the layers in reverse order from the softmax to the embedding, applying the chain rule one layer at a time
# Why: if these gradients match finite differences they earn the status of standard backpropagation and become the contrast baseline for the engine
# 역할: 기준 구현의 표준 역전파. 모든 파라미터의 기울기를 사전으로 낸다
# 방법: 소프트맥스에서 임베딩까지 층 순서를 거꾸로 걸어 연쇄법칙을 한 층씩 적용한다
# 이유: 이 기울기가 유한차분과 맞으면 표준 역전파의 자격을 얻고 엔진과의 대조 기준이 되기 때문
def p_ref_backward(ap, R, ids, y_flat, p, aD, C, cache, dtype):
    T, B = ids.shape
    _Mt = T * B
    _H = bc.HIDDEN_SIZE
    _NH = bc.NUMBER_TIME_MIX_HEAD
    _Hh = _H // _NH
    _G = {}
    _g = p.copy()
    _ar = ap.arange(_Mt)
    _g[y_flat, _ar] -= 1.0
    _g /= _Mt
    _tlr = R["m_mat_w_head_b"] @ aD
    _G["m_mat_w_head_a"] = _g @ _tlr.T
    _G["m_mat_w_head_b"] = (R["m_mat_w_head_a"].T @ _g) @ aD.T
    _G["m_vec_w_head_bias"] = _g.sum(1, keepdims=True)
    _dx = (R["m_mat_w_head_b"].T @ (R["m_mat_w_head_a"].T @ _g)).reshape(_H, T, B)
    _G["m_mat_w_lag"] = ap.zeros_like(R["m_mat_w_lag"])
    _G["m_mat_w_filter"] = ap.zeros_like(R["m_mat_w_filter"])
    _G["m_vec_w_filter_bias"] = ap.zeros_like(R["m_vec_w_filter_bias"])
    _G["m_mat_w_mix"] = ap.zeros_like(R["m_mat_w_mix"])
    _G["m_vec_w_norm_gain"] = ap.zeros_like(R["m_vec_w_norm_gain"])
    for bl in reversed(range(bc.NUMBER_BLOCK)):
        _xr, _xs, _fa, _pyr, _hres, _r = cache[bl]
        _dy = _dx.reshape(_H, _Mt)
        _gain = R["m_vec_w_norm_gain"][bl]
        _G["m_vec_w_norm_gain"][bl] = (_dy * _hres * _r).sum(1, keepdims=True)
        _s = (_dy * _gain * _hres).sum(0, keepdims=True)
        _dhres = _gain * _r * _dy - _hres * (_r ** 3) * _s / _H
        _dl = _dhres
        for i in reversed(range(bc.FFN_DEPTH)):
            _parts, _q, _qt, _y, s, _L, _F = _pyr[i]
            _du = _dl * (1.0 - _y * _y)
            _G["m_mat_w_filter"][bl, i] = (_du * _qt).sum(1)
            _G["m_vec_w_filter_bias"][bl, i, :_F] = _du.reshape(_F, _L, _Mt).sum(axis=(1, 2))
            _dq = (_du * R["m_mat_w_filter"][bl, i][:, None]).reshape(_F, _L, _Mt).sum(0)
            for k in reversed(range(s)):
                _qr = _parts[k].reshape(-1, 2, _Mt)
                _G["m_mat_w_mix"][bl, i, k, 0] = (_dq * _qr[:, 0]).sum()
                _G["m_mat_w_mix"][bl, i, k, 1] = (_dq * _qr[:, 1]).sum()
                _up = ap.empty((_dq.shape[0], 2, _Mt), dtype=dtype)
                _up[:, 0] = _dq * R["m_mat_w_mix"][bl, i, k, 0]
                _up[:, 1] = _dq * R["m_mat_w_mix"][bl, i, k, 1]
                _dq = _up.reshape(-1, _Mt)
            _dl = _dq
        _dh0 = _dl + _dhres
        _dxr4 = _dh0.reshape(_NH, _Hh, T, B)
        _ds = ap.ascontiguousarray(_dxr4.transpose(0, 2, 1, 3)).reshape(_NH, T, _Hh * B)
        _dC = ap.tril(_ds @ _xs.transpose(0, 2, 1))
        for l in range(T):
            _ii = ap.arange(l, T)
            _G["m_mat_w_lag"][bl, :, l] = _dC[:, _ii, _ii - l].sum(-1)
        _dxs = C[bl].transpose(0, 2, 1) @ _ds
        _dxmixin = ap.ascontiguousarray(_dxs.reshape(_NH, T, _Hh, B).transpose(0, 2, 1, 3)).reshape(_H, _Mt)
        _dx = (_dh0 + _dxmixin).reshape(_H, T, B)
    _G["m_mat_w_position"] = _dx.sum(2)[:, :T]
    _G["_dx_embed"] = _dx.reshape(_H, _Mt)
    return _G


# Role: updates one parameter with the same Adam formula as banya_core
# Method: accumulates the moment and variance with the same decay and descends with the same bias correction and epsilon
# Why: even a slight difference in the optimizer formula makes the trajectories diverge despite identical gradients
# 역할: banya_core 의 아담과 같은 수식으로 파라미터 하나를 갱신한다
# 방법: 모멘트와 분산을 같은 감쇠로 누적하고 같은 편향보정과 엡실론으로 내린다
# 이유: 옵티마이저 수식이 조금이라도 다르면 기울기가 같아도 궤적이 갈라지기 때문
def p_ref_adam(R, key, grad, lr, t):
    _m = R[key + "_m"]
    _v = R[key + "_v"]
    _m *= 0.9
    _m += 0.1 * grad
    _v *= 0.999
    _v += 0.001 * grad * grad
    R[key] -= lr * (_m / (1 - 0.9 ** t)) / ((_v / (1 - 0.999 ** t)) ** 0.5 + 1e-8)


# Role: one training step of the reference implementation, updating the same parameter set as the engine under the same accounting
# Method: after gathering all gradients, applies Adam with the same scaling as the engine and updates position and filter bias with the same SGD rule as the engine
# Why: the engine's update accounting (column-wise Adam for embeddings, SGD for positions, scaling for the lag kernel and gain) must also match for the trajectory contrast to hold
# 역할: 기준 구현의 한 학습 스텝. 엔진과 같은 파라미터 집합을 같은 회계로 갱신한다
# 방법: 기울기를 다 모은 뒤 아담은 엔진과 같은 배율로, 위치와 필터 치우침은 엔진과 같은 SGD 식으로 적용한다
# 이유: 엔진의 갱신 회계(임베딩 열단위 아담, 위치 SGD, 거리커널과 게인의 배율)까지 같아야 궤적 대조가 성립하기 때문
def p_ref_step(ap, R, ids, y_flat, lr, dtype):
    T, B = ids.shape
    _Mt = T * B
    _ce, _p, _aD, _C, _cache = p_ref_forward(ap, R, ids, y_flat, dtype)
    _G = p_ref_backward(ap, R, ids, y_flat, _p, _aD, _C, _cache, dtype)
    R["t"] += 1
    t = R["t"]
    p_ref_adam(R, "m_mat_w_head_a", _G["m_mat_w_head_a"], lr, t)
    p_ref_adam(R, "m_mat_w_head_b", _G["m_mat_w_head_b"], lr, t)
    p_ref_adam(R, "m_vec_w_head_bias", _G["m_vec_w_head_bias"], lr, t)
    p_ref_adam(R, "m_vec_w_norm_gain", _G["m_vec_w_norm_gain"] * _Mt, lr, t)
    p_ref_adam(R, "m_mat_w_mix", _G["m_mat_w_mix"], lr, t)
    p_ref_adam(R, "m_mat_w_filter", _G["m_mat_w_filter"], lr, t)
    p_ref_adam(R, "m_mat_w_lag", _G["m_mat_w_lag"] * _Mt, lr, t)
    for bl in range(bc.NUMBER_BLOCK):
        for i in range(bc.FFN_DEPTH):
            s = bc.blk_scale(bl + i)
            _L = bc.HIDDEN_SIZE >> s
            _F = 1 << s
            R["m_vec_w_filter_bias"][bl, i, :_F] -= lr * _G["m_vec_w_filter_bias"][bl, i, :_F] / _L
    R["m_mat_w_position"][:, :T] -= lr * _G["m_mat_w_position"]
    _dxe = _G["_dx_embed"]
    _flat = ids.reshape(-1)
    _uniq, _inv = ap.unique(_flat, return_inverse=True)
    _ge = ap.zeros((_uniq.shape[0], bc.HIDDEN_SIZE), dtype=dtype)
    if ap is np:
        np.add.at(_ge, _inv.reshape(-1), _dxe.T)
    else:
        bc.scatter_rows(_ge, _inv.reshape(-1), _dxe.T)
    _cols = _uniq
    _gcols = _ge.T
    _mb = R["m_mat_w_data_axis_m"][:, _cols]
    _vb = R["m_mat_w_data_axis_v"][:, _cols]
    _mb = 0.9 * _mb + 0.1 * _gcols
    _vb = 0.999 * _vb + 0.001 * _gcols * _gcols
    R["m_mat_w_data_axis_m"][:, _cols] = _mb
    R["m_mat_w_data_axis_v"][:, _cols] = _vb
    R["m_mat_w_data_axis"][:, _cols] -= lr * (_mb / (1 - 0.9 ** t)) / ((_vb / (1 - 0.999 ** t)) ** 0.5 + 1e-8)
    return _ce


# Role: measures the true cross entropy without a smoothing term
# Method: takes the log of the softmax probabilities without any correction constant
# Why: the gradients of both trainers are derivatives of the true unsmoothed loss, so finite differences must differentiate the same loss for the yardstick to match
# 역할: 평활 항 없는 참 교차엔트로피를 잰다
# 방법: 소프트맥스 확률에서 로그를 보정 상수 없이 취한다
# 이유: 두 학습기의 기울기는 평활 없는 참 손실의 미분이라 유한차분도 같은 손실을 미분해야 잣대가 맞기 때문
def p_exact_ce(ap, R, ids, y_flat, dtype):
    _, _p, _, _, _ = p_ref_forward(ap, R, ids, y_flat, dtype)
    _ar = ap.arange(len(y_flat))
    return float(-ap.mean(ap.log(_p[y_flat, _ar])))


# Role: verifies the backpropagation of the reference implementation independently against finite differences
# Method: for each parameter group picks live coordinates, measures the relative error between central-difference and backpropagated gradients, and also counts the evaluations per group
# Why: the reference must be correct on its own so that engine-reference agreement is not circular, and probing only dead coordinates would make a pass vacuous
# 역할: 기준 구현의 역전파를 유한차분과 대조해 독립으로 검증한다
# 방법: 파라미터 무리마다 살아있는 좌표를 골라 중심차분 기울기와 역전파 기울기의 상대 오차를 재고 종류별 평가 건수를 같이 센다
# 이유: 기준이 스스로 옳아야 엔진과 기준의 일치가 순환 논증이 아니게 되고 죽은 좌표만 찍으면 통과가 공허해지기 때문
def p_gradcheck(P, ids, y_flat):
    _R = p_ref_init(P, np, np.float64)
    _ce, _p, _aD, _C, _cache = p_ref_forward(np, _R, ids, y_flat, np.float64)
    _G = p_ref_backward(np, _R, ids, y_flat, _p, _aD, _C, _cache, np.float64)
    _rng = np.random.RandomState(7)
    _eps = 1e-5
    _T = ids.shape[0]
    _tokens = sorted(set(int(t) for t in np.asarray(ids).ravel()))
    _worst = 0.0
    _done = {}
    for key in ("m_mat_w_data_axis", "m_mat_w_position", "m_mat_w_lag", "m_mat_w_filter",
                "m_vec_w_filter_bias", "m_mat_w_mix", "m_vec_w_norm_gain",
                "m_mat_w_head_a", "m_mat_w_head_b", "m_vec_w_head_bias"):
        _arr = _R[key]
        _n_eval = 0
        _tries = 0
        while _n_eval < 3 and _tries < 60:
            _tries += 1
            _idx = tuple(_rng.randint(0, d) for d in _arr.shape)
            if key == "m_mat_w_position":
                _idx = (_idx[0], int(_rng.randint(0, _T)))
            if key == "m_mat_w_data_axis":
                _idx = (_idx[0], int(_tokens[_rng.randint(0, len(_tokens))]))
            if key == "m_mat_w_data_axis":
                _ana = p_embed_grad(np, _G, ids, _idx)
            elif key == "m_mat_w_position":
                _ana = float(_G[key][_idx[0], _idx[1]])
            else:
                _ana = float(_G[key][_idx])
            if abs(_ana) < 1e-9:
                continue
            _old = _arr[_idx]
            _arr[_idx] = _old + _eps
            _ce1 = p_exact_ce(np, _R, ids, y_flat, np.float64)
            _arr[_idx] = _old - _eps
            _ce2 = p_exact_ce(np, _R, ids, y_flat, np.float64)
            _arr[_idx] = _old
            _num = (_ce1 - _ce2) / (2 * _eps)
            _rel = abs(_num - _ana) / max(abs(_num), abs(_ana), 1e-12)
            _worst = max(_worst, _rel)
            _n_eval += 1
        _done[key] = _n_eval
    _n_total = sum(_done.values())
    _n_keys = sum(1 for v in _done.values() if v > 0)
    return _worst, _n_total, _n_keys, len(_done)


# Role: assembles the gradient of one embedding column from the backpropagation result
# Method: sums the embedding gradients at every position where that token appears
# Why: embeddings use column-wise sparse updates, so the finite-difference contrast also needs the column-sum gradient
# 역할: 임베딩 한 열의 기울기를 역전파 결과에서 모아 준다
# 방법: 해당 토큰이 등장한 자리의 임베딩 기울기를 전부 더한다
# 이유: 임베딩은 열 단위 희소 갱신이라 유한차분 대조에도 열 합 기울기가 필요하기 때문
def p_embed_grad(ap, G, ids, idx):
    _h, _tok = idx
    _dxe = G["_dx_embed"]
    _mask = (ids.reshape(-1) == _tok)
    return float(_dxe[_h][_mask].sum())


def main():
    np.random.seed(SEED)
    _corp = np.asarray(np.load(CORPUS, mmap_mode="r"), dtype=np.int64)
    _T = bc.CONTEXT_LENGTH
    _rng = np.random.RandomState(SEED + 1)
    _starts = _rng.randint(0, len(_corp) - _T - 1, size=(WARM + STEPS, BATCH))

    def p_batch(k):
        _ids = np.stack([_corp[s:s + _T] for s in _starts[k]], axis=1)
        _y = np.stack([_corp[s + 1:s + _T + 1] for s in _starts[k]], axis=1)
        return _ids, _y.reshape(-1)

    print(f"[제1편 소형 대조] H {bc.HIDDEN_SIZE} · 블록 {bc.NUMBER_BLOCK} · 헤드 {bc.NUMBER_TIME_MIX_HEAD} · 문맥 {_T} · 배치 {BATCH} · vocab {VOCAB} · {STEPS}스텝 · lr {LR}", flush=True)

    _m = bc.BanyaNoBP(VOCAB, SEED)
    _P = p_snapshot(_m)

    _gc_ids, _gc_y = p_batch(0)
    _gc_ids_small = _gc_ids[:16, :2]
    _gc_y_small = np.stack([_corp[s + 1:s + 17] for s in _starts[0][:2]], axis=1).reshape(-1)
    _worst, _n_eval, _n_keys, _n_all = p_gradcheck(_P, _gc_ids_small, _gc_y_small)
    print(f"[기준 검증] 표준 역전파 대 유한차분(float64) 살아있는 좌표 {_n_eval}개({_n_keys}/{_n_all}종) 상대오차 최대 {_worst:.2e}", flush=True)

    _losses_a = []
    for k in range(WARM):
        _ids, _y = p_batch(k)
        _losses_a.append(float(paper1.train_step(_m, cp.asarray(_ids), cp.asarray(_y), LR)))
    cp.cuda.Stream.null.synchronize()
    _t0 = time.perf_counter()
    for k in range(WARM, WARM + STEPS):
        _ids, _y = p_batch(k)
        _losses_a.append(float(paper1.train_step(_m, cp.asarray(_ids), cp.asarray(_y), LR)))
    cp.cuda.Stream.null.synchronize()
    _wall_a = time.perf_counter() - _t0

    _Rg = p_ref_init(_P, cp, cp.float32)
    _losses_b = []
    for k in range(WARM):
        _ids, _y = p_batch(k)
        _losses_b.append(p_ref_step(cp, _Rg, cp.asarray(_ids), cp.asarray(_y), LR, cp.float32))
    cp.cuda.Stream.null.synchronize()
    _t0 = time.perf_counter()
    for k in range(WARM, WARM + STEPS):
        _ids, _y = p_batch(k)
        _losses_b.append(p_ref_step(cp, _Rg, cp.asarray(_ids), cp.asarray(_y), LR, cp.float32))
    cp.cuda.Stream.null.synchronize()
    _wall_b = time.perf_counter() - _t0

    _Rc = p_ref_init(_P, np, np.float32)
    _loss_c_last = 0.0
    for k in range(WARM):
        _ids, _y = p_batch(k)
        _loss_c_last = p_ref_step(np, _Rc, _ids, _y, LR, np.float32)
    _t0 = time.perf_counter()
    for k in range(WARM, WARM + STEPS):
        _ids, _y = p_batch(k)
        _loss_c_last = p_ref_step(np, _Rc, _ids, _y, LR, np.float32)
    _wall_c = time.perf_counter() - _t0

    print("\n[손실 궤적] 워밍업 3스텝 뒤 300스텝, 엔진 A 대 기준 B(GPU)", flush=True)
    for k in CHECK_AT:
        _a = _losses_a[WARM + k]
        _b = _losses_b[WARM + k]
        print(f"  step {k + 1:>3}: A {_a:.4f} · B {_b:.4f} · |Δ| {abs(_a - _b):.2e}", flush=True)
    _d20 = max(abs(a - b) for a, b in zip(_losses_a[:WARM + 20], _losses_b[:WARM + 20]))
    _fin_a = _losses_a[-1]
    _fin_b = _losses_b[-1]
    print(f"  처음 20스텝 |Δ| 최대 {_d20:.2e} · 최종 |Δ| {abs(_fin_a - _fin_b):.2e} (상대 {abs(_fin_a - _fin_b) / _fin_a * 100:.2f}%)", flush=True)

    print("\n[실측 요약]", flush=True)
    print(f"  스텝 0 손실 A {_losses_a[0]:.4f} 대 B {_losses_b[0]:.4f} · |Δ| {abs(_losses_a[0] - _losses_b[0]):.2e}", flush=True)
    print(f"  최종 손실 A {_fin_a:.4f} 대 B {_fin_b:.4f} · numpy CPU 기준 {_loss_c_last:.4f}", flush=True)
    print(f"  실행 시간 300스텝  엔진 GPU {_wall_a:.2f}초 ({_wall_a / STEPS * 1e3:.1f}ms/스텝) · 기준 cupy GPU {_wall_b:.2f}초 ({_wall_b / STEPS * 1e3:.1f}ms/스텝) · 기준 numpy CPU {_wall_c:.2f}초 ({_wall_c / STEPS * 1e3:.1f}ms/스텝)", flush=True)
    print("  궤적 자료 (그림용, 10스텝마다)", flush=True)
    print("  A: " + " ".join(f"{v:.3f}" for v in _losses_a[WARM::10]), flush=True)
    print("  B: " + " ".join(f"{v:.3f}" for v in _losses_b[WARM::10]), flush=True)


if __name__ == "__main__":
    main()
