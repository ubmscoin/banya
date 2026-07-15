# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 1 engine probe. On a small synthetic setup with raw operations, it directly measures
that autodiff is not loaded, the normalization scale stability, the agreement against finite differences,
and the block-axis batching speed. GPU only (cupy).
Run  python3 paper1_engine_probe.py

반야 제1편 엔진 프로브. 자동미분 미적재, 정규화 스케일 안정성, 유한차분 대비 정합성,
블록축 배치 속도를 소형 합성 설정과 raw 연산으로 실측한다. GPU 전용(cupy).
실행  python3 paper1_engine_probe.py"""
import os
import sys
import math
import time
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import banya_core as bc

H = 32
N_MIX = 4
N_BLOCK = 6
BLOCK = 6
BATCH = 2
FFN_DEPTH = 2
VOCAB = 40
SPEED_H = 512
SPEED_MT = 128 * 32
SPEED_MIX = 8
SPEED_T = 128
SPEED_B = 32
SPEED_BLOCKS = 15
DATA_TYPE = "float32"


def p_configure_small():
    bc.HIDDEN_SIZE = H
    bc.NUMBER_TIME_MIX_HEAD = N_MIX
    bc.NUMBER_BLOCK = N_BLOCK
    bc.CONTEXT_LENGTH = BLOCK
    bc.FFN_DEPTH = FFN_DEPTH
    bc.SCALE_LENGTH_MIN = H
    bc.USE_NORM = True
    bc.EMBED_SPARSE = False
    bc.HEAD_USE_FP16 = False
    bc.USE_RELATIVE_MIX = True
    bc.BLOCKWISE_CAUSAL_MIX = True
    bc.USE_WEIGHT_TIE = False
    bc.USE_LOWRANK = False
    bc.USE_GOONGHAP = False
    bc.USE_BITOKEN = False
    bc.USE_GATE = True


def p_loss(m, ids, y, ar):
    _cache, _aD, _z = bc.forward(m, ids)
    _z2 = _z - _z.max(0, keepdims=True)
    _p = xp.exp(_z2)
    _p /= _p.sum(0, keepdims=True)
    return float(-xp.mean(xp.log(_p[y, ar] + 1e-9)))


# Role: measures normalization scale stability. Checks whether the output-to-input magnitude ratio stays near sqrt(H) even as the block count grows
# Method: for block counts 2, 4, 6, and 10 it builds a fresh small model each time and measures the ratio of the embedding input norm to the final output norm
# Why: to show that the RMSNorm transpose preserves the credit magnitude at each layer so the scale does not drift even in a deep network
# 역할: 정규화 스케일 안정성 실측. 블록 수를 늘려도 출력 대 입력 크기 비가 sqrt(H) 근처로 일정한지 본다
# 방법: 블록 수 2,4,6,10 마다 소형 모델을 새로 만들어 임베딩 입력 노름과 최종 출력 노름의 비를 잰다
# 이유: RMSNorm 전치가 층마다 신용 크기를 유지해 깊은 망에서도 스케일이 안 흐트러짐을 보이기 위해서다
def p_norm_scale(ids):
    print("[2] 정규화 스케일 (블록 수 무관 일정한가)")
    for nb in (2, 4, 6, 10):
        bc.NUMBER_BLOCK = nb
        _m = bc.BanyaNoBP(VOCAB, seed=0)
        _x0 = _m.m_mat_w_data_axis[:, ids] + _m.m_mat_w_position[:, :, None]
        _in_norm = float(xp.sqrt((_x0 ** 2).sum(0)).mean())
        _cache, _aD, _z = bc.forward(_m, ids)
        _out_norm = float(xp.sqrt((_aD ** 2).sum(0)).mean())
        print(f"    {nb:2d}블록  입력 L2={_in_norm:6.2f}  출력 L2={_out_norm:8.2f}  배수={_out_norm / _in_norm:6.2f}  (sqrt(H)={math.sqrt(H):.1f})")


# Role: measures agreement against finite differences. Confirms by ratio whether the manually derived exact transpose equals finite differences
# Method: over 10 samples it collects the ratio of the loss difference from tinily perturbing one embedding component to the gradient produced by the exact transpose, then takes the median
# Why: because a ratio median that clings to 1 means the gradient obtained without autodiff is numerically identical to backprop
# 역할: 유한차분 대비 정합성 실측. 수동으로 유도한 정확한 전치가 유한차분과 같은지 비율로 확인한다
# 방법: 임베딩 한 성분을 미세하게 흔든 손실 차분과 정확한 전치가 만든 기울기의 비를 표본 10개로 모아 중앙값을 낸다
# 이유: 비율 중앙값이 1에 밀착하면 자동미분 없이 얻은 기울기가 backprop 과 수치적으로 같음을 뜻하기 때문이다
def p_grad_check(ids, y, ar):
    print("[3] 유한차분 대비 정합성 (정확한 전치가 유한차분과 같은가)")
    bc.NUMBER_BLOCK = N_BLOCK
    _m = bc.BanyaNoBP(VOCAB, seed=0)
    _m2 = bc.BanyaNoBP(VOCAB, seed=0)
    _cap = {}
    _orig_scatter = bc.scatter_rows

    def p_wrap_scatter(dst, idx, vals):
        _orig_scatter(dst, idx, vals)
        _cap["delta"] = dst.copy()

    bc.scatter_rows = p_wrap_scatter
    _cache, _aD, _z = bc.forward(_m2, ids)
    _g, _ce = bc.p_softmax(_z, y, ar)
    _m2.t += 1
    _dtop, _g_data_axis_head = bc.head_delta_top(_m2, _aD, _g, BLOCK * BATCH, True)
    bc.block_credit(_m2, _cache, _dtop, bc.LEARNING_RATE_ADAM, ids, BLOCK, BATCH, _g_data_axis_head)
    bc.scatter_rows = _orig_scatter
    _delta = _cap["delta"]
    _eps = 1e-2
    _rng = np.random.RandomState(7)
    _ids_host = np.asarray(ids.get())
    _ratios = []
    for _ in range(10):
        _c = _rng.randint(0, H)
        _tok = int(_ids_host[_rng.randint(0, BLOCK), _rng.randint(0, BATCH)])
        _e = float(_m.m_mat_w_data_axis[_c, _tok])
        _m.m_mat_w_data_axis[_c, _tok] = _e + _eps
        _lp = p_loss(_m, ids, y, ar)
        _m.m_mat_w_data_axis[_c, _tok] = _e - _eps
        _lm = p_loss(_m, ids, y, ar)
        _m.m_mat_w_data_axis[_c, _tok] = _e
        _fd = (_lp - _lm) / (2 * _eps)
        _ana = -float(_delta[_tok, _c])
        _ratios.append(_fd / (_ana + 1e-20))
        print(f"    유한차분={_fd:>12.4e}  전치미분={_ana:>12.4e}  비율={_ratios[-1]:>8.3f}")
    _rr = np.array(_ratios)
    print(f"    비율 중앙값 {np.median(_rr):.4f}  평균 {_rr.mean():.4f}  표준편차 {_rr.std():.4f}")


def p_sync():
    xp.cuda.runtime.deviceSynchronize()


def p_timeit(fn, rep=50):
    fn()
    p_sync()
    _t = time.perf_counter()
    for _ in range(rep):
        fn()
    p_sync()
    return (time.perf_counter() - _t) / rep * 1000.0


# Role: measures the speed gain of block-axis batching. Times looping over the blocks sequentially versus bundling them along the block axis and running them at once
# Method: times the core operations of the circular channel transform and the relative-distance mixing under sequential repetition and single-batch execution respectively, then computes the ratio
# Why: to show that directly owning the credit path makes batch reshaping free, so the circular part becomes about 1.40x and the mixing about 5.07x faster
# 역할: 블록축 배치의 속도 이득 실측. 여러 블록을 순차로 도는 것과 블록축으로 묶어 한 번에 도는 것의 실행 시간을 잰다
# 방법: 순환 채널 변환과 상대 거리 혼합의 핵심 연산을 순차 반복과 배치 한 번으로 각각 시간을 재 배수를 낸다
# 이유: 신용 경로를 직접 소유하면 배치 재구성이 자유롭고 순환은 약 1.40배 혼합은 약 5.07배 빨라짐을 보이기 위해서다
def p_batch_speed():
    print(f"[4] 블록축 배치 속도  H={SPEED_H} Mt={SPEED_MT} 가운데블록={SPEED_BLOCKS}")
    _Hh = SPEED_H // SPEED_MIX
    _rng = np.random.RandomState(0)
    c = xp.asarray(_rng.randn(SPEED_BLOCKS, SPEED_H).astype(DATA_TYPE))
    h = xp.asarray(_rng.randn(SPEED_BLOCKS, SPEED_H, SPEED_MT).astype(DATA_TYPE))
    d = xp.asarray(_rng.randn(SPEED_BLOCKS, SPEED_H, SPEED_MT).astype(DATA_TYPE))
    cmix = xp.asarray(_rng.randn(SPEED_BLOCKS, SPEED_MIX, SPEED_T, SPEED_T).astype(DATA_TYPE))
    xr = xp.asarray(_rng.randn(SPEED_MIX, _Hh, SPEED_T, SPEED_B).astype(DATA_TYPE))
    dxr = xp.asarray(_rng.randn(SPEED_BLOCKS, SPEED_MIX, _Hh, SPEED_T, SPEED_B).astype(DATA_TYPE))

    def p_circ_fwd_seq():
        _out = []
        for i in range(SPEED_BLOCKS):
            _fc = xp.fft.rfft(c[i])[:, None]
            _out.append(xp.fft.irfft(_fc * xp.fft.rfft(h[i], axis=0), n=SPEED_H, axis=0))
        return xp.stack(_out)

    def p_circ_fwd_bat():
        _fc = xp.fft.rfft(c, axis=1)[:, :, None]
        return xp.fft.irfft(_fc * xp.fft.rfft(h, axis=1), n=SPEED_H, axis=1)

    def p_circ_fwd_transpose_seq():
        _out = []
        for i in range(SPEED_BLOCKS):
            _fc = xp.conj(xp.fft.rfft(c[i]))[:, None]
            _out.append(xp.fft.irfft(_fc * xp.fft.rfft(d[i], axis=0), n=SPEED_H, axis=0))
        return xp.stack(_out)

    def p_circ_fwd_transpose_bat():
        _fc = xp.conj(xp.fft.rfft(c, axis=1))[:, :, None]
        return xp.fft.irfft(_fc * xp.fft.rfft(d, axis=1), n=SPEED_H, axis=1)

    def p_mix_fwd_seq():
        _out = []
        for i in range(SPEED_BLOCKS):
            _out.append(xp.einsum('kts,kgsb->kgtb', cmix[i], xr))
        return xp.stack(_out)

    def p_mix_fwd_bat():
        return xp.einsum('nkts,kgsb->nkgtb', cmix, xr)

    def p_mix_transpose_seq():
        _out = []
        for i in range(SPEED_BLOCKS):
            _out.append(xp.einsum('kts,kgtb->kgsb', cmix[i], dxr[i]))
        return xp.stack(_out)

    def p_mix_transpose_bat():
        return xp.einsum('nkts,nkgtb->nkgsb', cmix, dxr)

    _rows = [
        ("순환 채널 변환 순전파", p_circ_fwd_seq, p_circ_fwd_bat),
        ("순환 채널 변환 전치", p_circ_fwd_transpose_seq, p_circ_fwd_transpose_bat),
        ("상대 거리 혼합 순전파", p_mix_fwd_seq, p_mix_fwd_bat),
        ("상대 거리 혼합 전치", p_mix_transpose_seq, p_mix_transpose_bat),
    ]
    for name, fseq, fbat in _rows:
        _ts = p_timeit(fseq)
        _tb = p_timeit(fbat)
        print(f"    {name:<18} 순차={_ts:>8.3f}ms  배치={_tb:>8.3f}ms  배치이득={_ts / _tb:>6.2f}배")


def main():
    p_configure_small()
    _ids = xp.asarray(np.random.RandomState(0).randint(0, VOCAB, (BLOCK, BATCH)))
    _y = xp.asarray(np.random.RandomState(1).randint(0, VOCAB, (BLOCK * BATCH,)))
    _ar = xp.arange(BLOCK * BATCH)
    print("[1] 자동미분 미적재: torch 모듈 적재됨?", "torch" in sys.modules)
    print()
    p_norm_scale(_ids)
    print()
    p_grad_check(_ids, _y, _ar)
    print()
    p_batch_speed()


if __name__ == "__main__":
    main()
