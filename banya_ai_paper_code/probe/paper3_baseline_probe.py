# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 3 standard baseline probe. Reproduces the measurement side of Table 6-1.
It scores the published bi-token model and the standard transformer baseline on identical windows of the bundled
elem_dialog corpus, window 128, 200 windows per seed, four seeds, and prints the next token cross entropy of each.
It then times raw training steps of both architectures at batch 32 and prints milliseconds per step.
The standard checkpoint model/banya_bp_pytorch.pt is produced by banya_bp_pytorch.py or downloaded; when it
is absent the standard rows are skipped with a notice. Speed depends on the machine; the paper values are the
Appendix B environment.
Run  python3 paper3_baseline_probe.py

반야 제3편 표준 기준선 프로브. 표 6-1의 측정 쪽을 재생한다.
동봉 elem_dialog 말뭉치의 동일한 창(창 128, 시드마다 200창, 시드 4벌)에서 발행 바이토큰 모델과 표준
트랜스포머 기준선을 나란히 채점해 다음 토큰 교차엔트로피를 각각 인쇄한다.
이어서 두 구조의 원시 학습 스텝을 배치 32로 시간 측정해 스텝당 밀리초를 인쇄한다.
표준 체크포인트 model/banya_bp_pytorch.pt 는 banya_bp_pytorch.py 로 학습하거나 내려받는다. 없으면
표준 줄만 건너뛰고 고지한다. 속도는 기계에 따라 다르며 본문 값은 부록 B 환경 기준이다.
실행  python3 paper3_baseline_probe.py"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper1_engine as paper1

os.chdir(bc._ROOT)
xp = bc.xp

FROZEN = os.path.join(_CODE, "model", "bitok_elem2_170000_m.npz")
BP_PATH = os.path.join(_CODE, "model", "banya_bp_pytorch.pt")
CE_CORPUS = "banya_world_data/elem_dialog.npy"
CE_WINDOW = 200
CE_SEEDS = [0, 1, 2, 3]
SPEED_WARM = 10
SPEED_STEPS = 30
BATCH = 32


# Role: draws the window start positions for one seed
# Method: one fixed seed random stream gives the start indices, shared by both models
# Why: the two architectures must be scored on byte-identical windows for the comparison to be fair
# 역할: 시드 하나의 창 시작 위치들을 뽑는다
# 방법: 고정 시드 난수 흐름 하나가 시작 색인을 내고 두 모델이 이를 공유한다
# 이유: 두 구조가 바이트 단위로 같은 창을 채점해야 비교가 공정하기 때문이다
def p_window_starts(corp, seed, t):
    _rng = np.random.RandomState(seed)
    return [_rng.randint(0, len(corp) - t - 1) for _ in range(CE_WINDOW)]


# Role: scores the bi-token engine model on the given windows
# Method: forward pass per window with the operation plane on, mean cross entropy of the correct token
# Why: this is the engine side of the Table 6-1 cross entropy row
# 역할: 주어진 창들에서 바이토큰 엔진 모델을 채점한다
# 방법: 연산면을 켠 채 창마다 순전파해 정답 토큰 교차엔트로피의 평균을 낸다
# 이유: 표 6-1 교차엔트로피 줄의 엔진 쪽이기 때문이다
def p_ce_engine(m, corp, starts, t):
    _ar = xp.arange(t)
    _sum = 0.0
    for i in starts:
        _seg = np.asarray(corp[i:i + t + 1])
        _X = xp.asarray(_seg[:t].reshape(-1, 1))
        _Y = xp.asarray(_seg[1:t + 1])
        _cache, _aD, _z = bc.forward(m, _X)
        _g, _ce = bc.p_softmax(_z, _Y, _ar)
        _sum += float(_ce)
    return _sum / len(starts)


# Role: times raw engine training steps at the published size
# Method: runs warmup steps then timed steps on random token batches and reports milliseconds per step
# Why: this is the engine side of the Table 6-1 speed row, and speed must be measured on the same machine as the standard side
# 역할: 발행 크기의 엔진 원시 학습 스텝을 시간 측정한다
# 방법: 무작위 토큰 배치로 워밍업 뒤 본측정 스텝을 돌려 스텝당 밀리초를 보고한다
# 이유: 표 6-1 속도 줄의 엔진 쪽이고, 속도는 표준 쪽과 같은 기계에서 측정되어야 하기 때문이다
def p_speed_engine(m, vocab, t):
    _rng = np.random.RandomState(0)
    def p_batch():
        _X = _rng.randint(0, vocab, size=(t, BATCH))
        _Y = _rng.randint(0, vocab, size=(t * BATCH,))
        return xp.asarray(_X), xp.asarray(_Y)
    for _ in range(SPEED_WARM):
        _X, _Y = p_batch()
        paper1.train_step(m, _X, _Y, 0.1)
    if hasattr(xp, "cuda"):
        xp.cuda.Stream.null.synchronize()
    _t0 = time.time()
    for _ in range(SPEED_STEPS):
        _X, _Y = p_batch()
        paper1.train_step(m, _X, _Y, 0.1)
    if hasattr(xp, "cuda"):
        xp.cuda.Stream.null.synchronize()
    return (time.time() - _t0) / SPEED_STEPS * 1000.0


def main():
    _corp = np.load(CE_CORPUS, mmap_mode="r")
    m, tok = bc.load_from(FROZEN)
    _t = bc.CONTEXT_LENGTH
    _vocab = m.m_vocab_size
    print(f"[제3편 표준 기준선] 얼린 모델 {os.path.basename(FROZEN)} · 창 {_t} · 시드 {len(CE_SEEDS)}벌 x {CE_WINDOW}창", flush=True)

    _std = None
    bp = None
    try:
        import torch
        import banya_bp_pytorch as bp
        if os.path.exists(BP_PATH):
            _std = bp.load()
        else:
            print(f"  표준 체크포인트 없음({BP_PATH}): 표준 줄은 건너뜀. banya_bp_pytorch.py 로 학습해 만들거나 내려받아라", flush=True)
    except ImportError:
        print("  torch 미설치로 표준 줄은 건너뜀 (선택 의존성)", flush=True)

    print("\n[1] 예측 교차엔트로피 (같은 창, 낮을수록 예측이 쉬움. 표 6-1 목표: 바이토큰 0.2065, 표준 0.1864)", flush=True)
    _eng_list = []
    _bp_list = []
    for seed in CE_SEEDS:
        _starts = p_window_starts(_corp, seed, _t)
        _eng = p_ce_engine(m, _corp, _starts, _t)
        _eng_list.append(_eng)
        _line = f"  시드{seed}: 바이토큰 {_eng:.4f}"
        if _std is not None:
            import torch
            import torch.nn.functional as F
            _sum = 0.0
            with torch.no_grad():
                for i in _starts:
                    _seg = np.asarray(_corp[i:i + _t + 1])
                    _X = torch.from_numpy(_seg[:_t][None].astype(np.int64)).to("cuda")
                    _Y = torch.from_numpy(_seg[1:_t + 1][None].astype(np.int64)).to("cuda")
                    _logits = _std(_X)
                    _sum += float(F.cross_entropy(_logits.reshape(-1, _logits.shape[-1]), _Y.reshape(-1)))
            _bp_ce = _sum / len(_starts)
            _bp_list.append(_bp_ce)
            _line += f"  표준 {_bp_ce:.4f}"
        print(_line, flush=True)
    _msg = f"  평균: 바이토큰 {np.mean(_eng_list):.4f} (목표 0.2065)"
    if _bp_list:
        _msg += f"  표준 {np.mean(_bp_list):.4f} (목표 0.1864)"
    print(_msg, flush=True)

    print("\n[2] 학습 속도 (무작위 배치 32, 스텝당 밀리초. 부록 B 환경 목표: 바이토큰 80.5, 표준 136.7. 기계 의존이라 다른 기계에선 비율만 대조)", flush=True)
    try:
        _ms_eng = p_speed_engine(m, _vocab, _t)
        print(f"  바이토큰 엔진 {_ms_eng:.1f} ms/step", flush=True)
    except Exception as e:
        print(f"  엔진 속도 측정 건너뜀 ({type(e).__name__})", flush=True)
    if bp is not None:
        try:
            import torch
            import torch.nn.functional as F
            _m2 = bp.TransformerLM(_vocab, bp.H, bp.N_BLOCK, bp.N_HEADS, bp.BLOCK,
                                      bp.FFN_MULT, bp.LOWRANK, bp.HEAD_R).to("cuda")
            _opt = torch.optim.AdamW(_m2.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
            _rng = np.random.RandomState(0)
            def p_step():
                _X = torch.from_numpy(_rng.randint(0, _vocab, size=(BATCH, _t)).astype(np.int64)).to("cuda")
                _Y = torch.from_numpy(_rng.randint(0, _vocab, size=(BATCH, _t)).astype(np.int64)).to("cuda")
                _logits = _m2(_X)
                _loss = F.cross_entropy(_logits.reshape(-1, _vocab), _Y.reshape(-1))
                _opt.zero_grad(set_to_none=True)
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(_m2.parameters(), 1.0)
                _opt.step()
            for _ in range(SPEED_WARM):
                p_step()
            torch.cuda.synchronize()
            _t0 = time.time()
            for _ in range(SPEED_STEPS):
                p_step()
            torch.cuda.synchronize()
            print(f"  표준 트랜스포머 {(time.time() - _t0) / SPEED_STEPS * 1000.0:.1f} ms/step", flush=True)
        except Exception as e:
            print(f"  표준 속도 측정 건너뜀 ({type(e).__name__})", flush=True)


if __name__ == "__main__":
    main()
