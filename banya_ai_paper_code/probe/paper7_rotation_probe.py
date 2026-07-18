# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 7 rotation-operator gate probe. One run prints all measurements of the paper.
First it verifies the storage-free adjoint of the rotation gate against finite differences.
Second it proves the kernel-injection design changes nothing, by running a python reimplementation of the
original sigmoid kernels and matching its training cross entropy trajectory.
Third it runs the mod 3 depth-excluded discrimination experiment: reading moves M0 M1 M2 one by one, the model must speak the
running sum mod 3 state S0 S1 S2 at every step. Only 14 of the 27 length-3 combinations are taught and 13 are
hidden, so a lookup-table solver fails on the blanks while a solver that learned the cyclic operation fills them.
Groups A(sigmoid gate, operation plane on), B(operation plane off), D(tanh gate), E(rotation gate) share the same
one-block Banya engine and differ only in the gate; group C is a one-layer standard transformer for contrast and
runs only when torch is installed, otherwise it is skipped with a notice.
Scoring: forced = step-scored while showing true states, self = feeding back its own outputs to the end.
GPU-only cupy. Numbers are seed-fixed but GPU accumulation order may wiggle the last digits.
Run  python3 paper7_rotation_probe.py

반야 제7편 회전 연산자 게이트 프로브. 한 번의 실행으로 논문의 측정 전부를 인쇄한다.
첫째, 회전 게이트의 무저장 수반을 유한차분과 대조해 검증한다.
둘째, 커널 주입 설계가 계산을 바꾸지 않음을 원본 시그모이드 커널의 파이썬 재구현으로 학습 교차엔트로피
궤적 일치로 증명한다.
셋째, mod 3 깊이 배제 판별 실험을 돌린다. 수 M0 M1 M2 를 하나씩 읽으며 지금까지 합의 mod 3 상태
S0 S1 S2 를 자리마다 말한다. 길이 3 조합 27개 중 14개만 가르치고 13개는 숨기므로 표를 외운 풀이는
빈칸에서 무너지고 순환 연산을 배운 풀이만 빈칸을 채운다.
실험군 A(시그모이드 게이트, 연산면 켬), B(연산면 차단), D(tanh 게이트), E(회전 게이트)는 같은 반야
1블록 엔진에서 게이트만 다르고, C 는 대조용 표준 트랜스포머 1층으로 torch 가 있을 때만 돌며 없으면
건너뛴다고 고지한다.
채점: 강제 = 정답 상태를 보여주며 스텝 채점, 자기 = 자기 출력을 되먹여 끝까지.
Fourth it verifies the storage free adjoint of the quaternion gate the same way.
Fifth it runs the triangle non commutative discrimination: reading moves R R2 F one by one, the model must speak
the state of a labeled triangle among six states at every step. Two splits are run. The inference trial hides part
of the rule table itself, so every group meets the same ceiling. The composition trial exposes the full rule
table and hides 13 of the 27 length 3 combinations, so only a gate owning non commutative composition fills the
blanks. Group Q(quaternion gate) joins the five groups of the mod 3 task.
넷째, 쿼터니언 게이트의 무저장 수반을 같은 방식으로 검증한다.
다섯째, 삼각형 비가환 판별을 돌린다. 이동 R R2 F 를 하나씩 읽으며 표시된 정삼각형의 상태 6종을
자리마다 말한다. 분할 두 가지를 돌린다. 유추시험은 규칙 표 자체를 일부 숨겨 전 군이 같은 천장에 닿고,
합성시험은 규칙 표 전체를 노출한 채 길이 3 조합 27개 중 13개를 숨겨 비가환 합성을 기본 연산으로 가진
게이트만 빈칸을 채운다. mod 3 과제의 다섯 군에 Q(쿼터니언 게이트)가 더해진다.
GPU 전용 cupy. 수치는 시드 고정이나 GPU 합산 순서로 마지막 자리가 흔들릴 수 있다.
실행  python3 paper7_rotation_probe.py"""
import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper1_engine as paper1
import paper7_rotation as paper7

os.chdir(bc._ROOT)
xp = bc.xp

PAD = 0
MV = [1, 2, 3]
ST = [4, 5, 6]
VOCAB = 8
T = 16
STEPS = 12000
LR = 0.2
TEST_LENS = [4, 5, 6]
SEEDS = [0, 1, 2]

_ORIG_FWD = bc._gate_fwd_bt_k
_ORIG_BWD = bc._gate_bwd_bt_k
_ORIG_GW = bc._gate_gw_bt_k
_STASH = {}


# Role: configures the engine to the one-block small setting shared by every group
# Method: overrides the banya_core globals to hidden 64, four mixing heads, one block, context 16, ladder depth 2
# Why: the depth-excluded design removes depth from everyone, so the only remaining difference between groups is the gate
# 역할: 모든 실험군이 공유하는 1블록 소형 설정으로 엔진을 맞춘다
# 방법: banya_core 전역을 히든 64, 혼합 헤드 4, 블록 1, 문맥 16, 사다리 겹 2 로 덮는다
# 이유: 깊이 배제 판별은 전원에게서 깊이를 제한하므로 실험군 간 남는 차이가 게이트뿐이기 때문이다
def p_config():
    bc.HIDDEN_SIZE = 64
    bc.NUMBER_TIME_MIX_HEAD = 4
    bc.NUMBER_BLOCK = 1
    bc.CONTEXT_LENGTH = T
    bc.FFN_DEPTH = 2
    bc.SCALE_LENGTH_MIN = 64
    bc.USE_NORM = True
    bc.EMBED_SPARSE = False
    bc.HEAD_USE_FP16 = False
    bc.USE_RELATIVE_MIX = True
    bc.BLOCKWISE_CAUSAL_MIX = True
    bc.USE_WEIGHT_TIE = False
    bc.USE_LOWRANK = False
    bc.USE_GOONGHAP = False
    bc.USE_GATE = True


# Role: restores the original bitoken gate kernels of the published engine
# Method: reassigns the three module attributes saved at import time
# Why: groups alternate inside one process and every group must start from the published kernels
# 역할: 발행 엔진의 원본 바이토큰 게이트 커널로 되돌린다
# 방법: 임포트 시점에 보관한 모듈 속성 셋을 재대입한다
# 이유: 한 프로세스에서 실험군이 번갈아 돌므로 모든 군이 발행 커널에서 출발해야 하기 때문이다
def p_restore():
    bc._gate_fwd_bt_k = _ORIG_FWD
    bc._gate_bwd_bt_k = _ORIG_BWD
    bc._gate_gw_bt_k = _ORIG_GW


# Role: installs a python reimplementation of the original sigmoid kernels for the design proof
# Method: replaces the three kernels with array-arithmetic functions of identical mathematics
# Why: if the injected python version reproduces the compiled kernel trajectory exactly, the injection mechanism itself is proven not to change any computation
# 역할: 설계 증명용으로 원본 시그모이드 커널의 파이썬 재구현을 설치한다
# 방법: 수식이 동일한 배열 연산 함수 셋으로 커널 셋을 교체한다
# 이유: 주입한 파이썬판이 컴파일 커널의 궤적을 정확히 재현하면 주입 방식 자체가 계산을 안 바꿈이 증명되기 때문이다
def p_patch_sigmoid_python():
    def p_fwd(xf, xm, eg, WG, BG, Mt, h):
        _g = 1.0 / (1.0 + xp.exp(-(WG * xf + BG + eg)))
        h[:] = xf + _g * xm

    def p_bwd(dl, fa0, xf, eg, WG, BG, Mt, ddir, dlg, dgt):
        _g = 1.0 / (1.0 + xp.exp(-(WG * xf + BG + eg)))
        _dg = dl * (fa0 - xf) * (1.0 - _g)
        ddir[:] = dl + _dg * WG
        dlg[:] = _g * dl
        dgt[:] = _dg

    def p_gw(dl, fa0, xf, eg, WG, BG, Mt, NCH, pw, pb):
        _g = 1.0 / (1.0 + xp.exp(-(WG * xf + BG + eg)))
        _dg = dl * (fa0 - xf) * (1.0 - _g)
        pw[:] = 0.0
        pb[:] = 0.0
        pw[:, 0] = (_dg * xf).sum(1)
        pb[:, 0] = _dg.sum(1)

    bc._gate_fwd_bt_k = p_fwd
    bc._gate_bwd_bt_k = p_bwd
    bc._gate_gw_bt_k = p_gw


# Role: installs the tanh gate variant for group D
# Method: same wiring as the sigmoid gate but with range minus one to one, stashing the mixed vector for the backward pass
# Why: tanh adds a sign to the valve, isolating how much of the rotation gate's gain comes from sign alone
# 역할: 실험군 D 용 tanh 게이트 변형을 설치한다
# 방법: 시그모이드 게이트와 같은 배선에 치역만 -1 에서 1 로 바꾸고, 역방향을 위해 혼합 벡터를 보관한다
# 이유: tanh 는 밸브에 부호만 더한 것이라 회전 게이트 이득 중 부호 몫이 얼마인지를 가른다
def p_patch_tanh():
    def p_fwd(xf, xm, eg, WG, BG, Mt, h):
        _g = xp.tanh(WG * xf + BG + eg)
        _STASH["xm"] = xm.copy()
        h[:] = xf + _g * xm

    def p_bwd(dl, fa0, xf, eg, WG, BG, Mt, ddir, dlg, dgt):
        _g = xp.tanh(WG * xf + BG + eg)
        _xm = _STASH["xm"]
        _dg = dl * _xm * (1.0 - _g * _g)
        ddir[:] = dl + _dg * WG
        dlg[:] = _g * dl
        dgt[:] = _dg

    def p_gw(dl, fa0, xf, eg, WG, BG, Mt, NCH, pw, pb):
        _g = xp.tanh(WG * xf + BG + eg)
        _xm = _STASH["xm"]
        _dg = dl * _xm * (1.0 - _g * _g)
        pw[:] = 0.0
        pb[:] = 0.0
        pw[:, 0] = (_dg * xf).sum(1)
        pb[:, 0] = _dg.sum(1)

    bc._gate_fwd_bt_k = p_fwd
    bc._gate_bwd_bt_k = p_bwd
    bc._gate_gw_bt_k = p_gw


# Role: builds mod 3 rows where the running state is written after every move
# Method: for each move tuple, left-pads to the fixed window and appends move token then state token in turn
# Why: writing the state externalizes chaining to the loop, so each step demands exactly one cyclic operation, which is what separates the gates
# 역할: 수 하나마다 누적 상태를 쓰는 mod 3 행을 만든다
# 방법: 이동 조합마다 고정 창으로 왼쪽을 채우고 이동 토큰과 상태 토큰을 차례로 붙인다
# 이유: 상태를 쓰면 연쇄가 루프로 외부화되어 스텝당 순환 연산 한 번만 남고, 그 한 번이 게이트들을 가르기 때문이다
def p_make_rows(lens):
    _rows = []
    _slots = []
    for k in lens:
        for moves in itertools.product([0, 1, 2], repeat=k):
            _seq = [PAD] * (T - 2 * k)
            _bpos = []
            _run = 0
            for mv in moves:
                _bpos.append(len(_seq))
                _seq.append(MV[mv])
                _run = (_run + mv) % 3
                _seq.append(ST[_run])
            _rows.append(_seq)
            _slots.append(_bpos)
    return np.asarray(_rows, dtype=np.int64), _slots


# Role: builds next-token targets that carry loss only at the state slots
# Method: fills the target with pad everywhere and puts the true state token at each move position
# Why: the other positions hold unpredictable moves whose noise gradient would bury the state signal
# 역할: 상태 자리에만 손실이 걸리는 다음 토큰 타겟을 만든다
# 방법: 타겟을 전부 pad 로 채우고 각 이동 자리에는 정답 상태 토큰을 넣는다
# 이유: 나머지 자리는 예측 불가능한 이동이라 그 잡음 기울기가 상태 신호를 묻기 때문이다
def p_targets(rows, slots):
    _Y = np.full_like(rows, PAD)
    for r in range(len(rows)):
        for bp in slots[r]:
            _Y[r, bp] = rows[r, bp + 1]
    return _Y


# Role: trains one engine model on the visible patterns and logs convergence
# Method: samples 32 rows per step with the seed-fixed stream and records the first step whose cross entropy drops under 0.01
# Why: the right primitive should not only solve the blanks but also wander less on the way, which the convergence step measures
# 역할: 보이는 패턴으로 엔진 모델 하나를 학습하고 수렴을 기록한다
# 방법: 시드 고정 흐름으로 스텝마다 32행을 뽑고 교차엔트로피가 처음 0.01 아래로 떨어진 스텝을 적는다
# 이유: 옳은 기본 연산은 빈칸만 채우는 게 아니라 가는 길에서도 덜 헤매야 하고 그것을 수렴 스텝이 재기 때문이다
def p_train_engine(seed, train_rows, train_slots):
    _m = bc.BanyaNoBP(VOCAB, seed=seed)
    _rng = np.random.RandomState(seed)
    _Y_all = p_targets(train_rows, train_slots)
    _conv = None
    for step in range(STEPS):
        _pick = _rng.randint(0, len(train_rows), 32)
        _X = train_rows[_pick].T
        _Y = _Y_all[_pick].T
        _ce = paper1.train_step(_m, xp.asarray(_X), xp.asarray(_Y.reshape(-1)), LR)
        if _conv is None and float(_ce) < 0.01:
            _conv = step + 1
    return _m, _conv


# Role: scores a model on one rung under forced or self-feedback evaluation
# Method: forced scores the final state with true states shown; self blanks the states, fills them one pass at a time from the model's own argmax, then scores the final state
# Why: self-feedback is the real loop where per-step errors compound, so it is the honest score, and forced isolates the per-step rule
# 역할: 강제 채점과 자기 되먹임 채점으로 한 칸을 채점한다
# 방법: 강제는 정답 상태를 보여주며 끝 상태를 채점하고, 자기는 상태를 비운 뒤 모델의 argmax 로 한 자리씩 채워 끝 상태를 채점한다
# 이유: 자기 되먹임은 스텝 오류가 눈덩이가 되는 진짜 루프라 정직한 점수이고, 강제는 스텝당 규칙만을 분리해 재기 때문이다
def p_eval_engine(m, rows, slots, rollout):
    _rows = rows.copy()
    if rollout:
        for j in range(len(_rows)):
            for bp in slots[j]:
                _rows[j, bp + 1] = PAD
    _final_hit = 0
    for i in range(0, len(_rows), 240):
        _chunk = _rows[i:i + 240].copy()
        _true = rows[i:i + 240]
        _sl = slots[i:i + 240]
        _passes = max(len(s) for s in _sl) if rollout else 1
        for step_i in range(_passes):
            _cache, _aD, _z = bc.forward(m, xp.asarray(_chunk.T))
            _zh = bc.to_host(_z.reshape(VOCAB, T, -1))
            if rollout:
                for j in range(len(_chunk)):
                    if step_i >= len(_sl[j]):
                        continue
                    bp = _sl[j][step_i]
                    _sc = [_zh[t, bp, j] for t in ST]
                    _chunk[j, bp + 1] = ST[int(np.argmax(_sc))]
        _cache, _aD, _z = bc.forward(m, xp.asarray(_chunk.T))
        _zh = bc.to_host(_z.reshape(VOCAB, T, -1))
        for j in range(len(_chunk)):
            _last = _sl[j][-1]
            if rollout:
                _pred_last = _chunk[j, _last + 1]
            else:
                _sc = [_zh[t, _last, j] for t in ST]
                _pred_last = ST[int(np.argmax(_sc))]
            if _pred_last == _true[j, _last + 1]:
                _final_hit += 1
    return _final_hit / len(_rows)


# Role: opens the trained operation plane and prints the circular statistics of token angle differences
# Method: takes the even rows of the operation plane as angles and prints the circular mean and concentration of pairwise differences for the state and move tokens
# Why: if the model formed a rotation representation, structure should appear in the angles themselves, beyond the accuracy numbers
# 역할: 학습된 연산면을 열어 토큰 각도차의 원형 통계를 인쇄한다
# 방법: 연산면의 짝수 행을 각도로 읽어 상태와 이동 토큰의 쌍별 차이의 원형 평균과 집중도를 인쇄한다
# 이유: 모델이 회전 표현을 형성했다면 정확도 숫자 너머 각도 자체에 구조가 보여야 하기 때문이다
def p_inspect_rotation(m):
    _eop = bc.to_host(m.m_mat_w_operate_axis).astype(np.float64)
    _names = {1: "M0", 2: "M1", 3: "M2", 4: "S0", 5: "S1", 6: "S2"}
    _pairs = [(4, 5), (5, 6), (6, 4), (1, 2), (2, 3), (3, 1)]
    _parts = []
    for a, b in _pairs:
        _d = _eop[0::2, b] - _eop[0::2, a]
        _z = np.exp(1j * _d).mean()
        _parts.append(f"{_names[a]}~{_names[b]} {np.angle(_z) * 180 / np.pi:+.0f}도(집중 {np.abs(_z):.2f})")
    print(f"    [군 구조] " + "  ".join(_parts), flush=True)


# Role: runs one experiment group over the seeds and prints one row per seed plus the group mean
# Method: restores the published kernels, applies the group's gate setup, trains, scores every rung under both scorings, and prints the seed mean row at the end
# Why: every group must pass through the identical pipeline so that the only remaining difference is the gate, and the mean row lets the text averages be checked straight off the output
# 역할: 한 실험군을 시드들로 돌리고 시드별 한 줄과 군 평균 한 줄을 인쇄한다
# 방법: 발행 커널로 복귀한 뒤 그 군의 게이트 설치를 적용하고 학습해 전 칸을 두 채점으로 채점하며, 끝에 시드 평균 줄을 인쇄한다
# 이유: 모든 군이 동일한 파이프라인을 지나야 남는 차이가 게이트뿐이고, 평균 줄이 있어야 본문의 평균 수치를 출력에서 바로 대조할 수 있기 때문이다
def p_run_group(name, setup, bitok, train_rows, train_slots, test_sets, inspect=False, report=None):
    _sums = {}
    _convs = []
    for seed in SEEDS:
        p_config()
        p_restore()
        if setup is not None:
            setup()
        bc.USE_BITOKEN = bitok
        _m, _conv = p_train_engine(seed, train_rows, train_slots)
        if _conv is not None:
            _convs.append(_conv)
        _parts = []
        for tname, (rows, slots) in test_sets.items():
            _tf = p_eval_engine(_m, rows, slots, rollout=False)
            _ro = p_eval_engine(_m, rows, slots, rollout=True)
            _acc = _sums.setdefault(tname, [0.0, 0.0])
            _acc[0] += _tf
            _acc[1] += _ro
            _parts.append(f"{tname} 강제{_tf * 100:.0f}%/자기{_ro * 100:.0f}%")
        print(f"  {name} 시드{seed} (수렴 {_conv}스텝): " + "  ".join(_parts), flush=True)
        if inspect:
            p_inspect_rotation(_m)
        if report is not None:
            report(_m)
    _avg_parts = [f"{t} 강제{v[0] / len(SEEDS) * 100:.1f}%/자기{v[1] / len(SEEDS) * 100:.1f}%" for t, v in _sums.items()]
    _conv_txt = f" (수렴 평균 {sum(_convs) / len(_convs):.0f}스텝)" if _convs else ""
    print(f"  [평균] {name}{_conv_txt}: " + "  ".join(_avg_parts), flush=True)
    p_restore()


# Role: proves the kernel-injection design changes nothing
# Method: trains seed 0 for 2000 steps twice, once on the compiled kernels and once on the python sigmoid reimplementation, and prints the cross entropy at five checkpoints with the largest gap
# Why: every variant below enters through the same injection, so the injection itself must first be shown computation-neutral
# 역할: 커널 주입 설계가 계산을 바꾸지 않음을 증명한다
# 방법: 시드 0 을 2000스텝씩 두 번, 컴파일 커널과 파이썬 시그모이드 재구현으로 학습해 다섯 지점의 교차엔트로피와 최대 차이를 인쇄한다
# 이유: 아래 모든 변형이 같은 주입으로 들어가므로 주입 자체가 계산 중립임을 먼저 보여야 하기 때문이다
def p_plumbing_proof(train_rows, train_slots):
    _Y_all = p_targets(train_rows, train_slots)
    _traces = []
    for use_python in (False, True):
        p_config()
        p_restore()
        if use_python:
            p_patch_sigmoid_python()
        bc.USE_BITOKEN = True
        _m = bc.BanyaNoBP(VOCAB, seed=0)
        _rng = np.random.RandomState(0)
        _trace = []
        for step in range(2000):
            _pick = _rng.randint(0, len(train_rows), 32)
            _X = train_rows[_pick].T
            _Y = _Y_all[_pick].T
            _ce = paper1.train_step(_m, xp.asarray(_X), xp.asarray(_Y.reshape(-1)), LR)
            if (step + 1) % 400 == 0:
                _trace.append(float(_ce))
        _traces.append(_trace)
    p_restore()
    _diff = max(abs(a - b) for a, b in zip(_traces[0], _traces[1]))
    print(f"    커널판  {' '.join(f'{v:.4f}' for v in _traces[0])}", flush=True)
    print(f"    파이썬판 {' '.join(f'{v:.4f}' for v in _traces[1])}", flush=True)
    print(f"    궤적 최대 차이 {_diff:.2e} (0 에 밀착하면 주입이 계산을 안 바꿈)", flush=True)


# Role: runs the standard one-layer transformer contrast group when torch is available
# Method: builds a one-block QKV transformer of comparable width, trains it on the identical data stream, and scores the same rungs
# Why: the depth-excluded discrimination claims depth-starved standard attention cannot fill the blanks, and that claim needs its own row
# 역할: torch 가 있을 때 표준 1층 트랜스포머 대조군을 돌린다
# 방법: 비슷한 폭의 1블록 QKV 트랜스포머를 만들어 같은 데이터 흐름으로 학습하고 같은 칸을 채점한다
# 이유: 깊이를 몰수당한 표준 어텐션은 빈칸을 못 채운다는 주장에도 제 줄이 필요하기 때문이다
def p_run_torch_group(train_rows, train_slots, test_sets):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("  C 표준1층: torch 미설치로 건너뜀 (선택 의존성)", flush=True)
        return
    _dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Block(nn.Module):
        def __init__(self, H, nh):
            super().__init__()
            self.nh = nh
            self.ln1 = nn.LayerNorm(H)
            self.ln2 = nn.LayerNorm(H)
            self.qkv = nn.Linear(H, 3 * H)
            self.proj = nn.Linear(H, H)
            self.ff = nn.Sequential(nn.Linear(H, 4 * H), nn.GELU(), nn.Linear(4 * H, H))

        def forward(self, x):
            B, TT, H = x.shape
            _h = self.ln1(x)
            q, k, v = self.qkv(_h).split(H, dim=2)
            q = q.view(B, TT, self.nh, H // self.nh).transpose(1, 2)
            k = k.view(B, TT, self.nh, H // self.nh).transpose(1, 2)
            v = v.view(B, TT, self.nh, H // self.nh).transpose(1, 2)
            _a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            _a = _a.transpose(1, 2).reshape(B, TT, H)
            x = x + self.proj(_a)
            x = x + self.ff(self.ln2(x))
            return x

    class LM(nn.Module):
        def __init__(self, H=64):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, H)
            self.pos = nn.Embedding(T, H)
            self.blk = Block(H, 4)
            self.lnf = nn.LayerNorm(H)
            self.head = nn.Linear(H, VOCAB)

        def forward(self, idx):
            _p = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(_p)[None]
            x = self.blk(x)
            return self.head(self.lnf(x))

    def p_eval_torch(m, rows, slots, rollout):
        _rows = rows.copy()
        if rollout:
            for j in range(len(_rows)):
                for bp in slots[j]:
                    _rows[j, bp + 1] = PAD
        _final_hit = 0
        with torch.no_grad():
            for i in range(0, len(_rows), 240):
                _chunk = _rows[i:i + 240].copy()
                _true = rows[i:i + 240]
                _sl = slots[i:i + 240]
                _passes = max(len(s) for s in _sl) if rollout else 1
                for step_i in range(_passes):
                    _logits = m(torch.from_numpy(_chunk).to(_dev)).cpu().numpy()
                    if rollout:
                        for j in range(len(_chunk)):
                            if step_i >= len(_sl[j]):
                                continue
                            bp = _sl[j][step_i]
                            _sc = [_logits[j, bp, t] for t in ST]
                            _chunk[j, bp + 1] = ST[int(np.argmax(_sc))]
                _logits = m(torch.from_numpy(_chunk).to(_dev)).cpu().numpy()
                for j in range(len(_chunk)):
                    _last = _sl[j][-1]
                    if rollout:
                        _pred_last = _chunk[j, _last + 1]
                    else:
                        _sc = [_logits[j, _last, t] for t in ST]
                        _pred_last = ST[int(np.argmax(_sc))]
                    if _pred_last == _true[j, _last + 1]:
                        _final_hit += 1
        return _final_hit / len(_rows)

    _Y_all = p_targets(train_rows, train_slots)
    for seed in SEEDS:
        torch.manual_seed(seed)
        _m = LM().to(_dev)
        _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        _rng = np.random.RandomState(seed)
        for step in range(STEPS):
            _pick = _rng.randint(0, len(train_rows), 32)
            _X = torch.from_numpy(train_rows[_pick]).to(_dev)
            _Y = torch.from_numpy(_Y_all[_pick]).to(_dev)
            _logits = _m(_X)
            _loss = F.cross_entropy(_logits.reshape(-1, VOCAB), _Y.reshape(-1))
            _opt.zero_grad(set_to_none=True)
            _loss.backward()
            _opt.step()
        _m.eval()
        _parts = []
        for tname, (rows, slots) in test_sets.items():
            _tf = p_eval_torch(_m, rows, slots, rollout=False)
            _ro = p_eval_torch(_m, rows, slots, rollout=True)
            _parts.append(f"{tname} 강제{_tf * 100:.0f}%/자기{_ro * 100:.0f}%")
        print(f"  C 표준1층 시드{seed}: " + "  ".join(_parts), flush=True)


TRI_MV = [1, 2, 3]
TRI_ST = [4, 5, 6, 7, 8, 9]
TRI_VOCAB = 10
TRI_NAME = ["R", "R2", "F"]
TRI_SEEDS = [0, 1, 2, 3, 4, 5]


# Role: switches the task-wide token globals between the mod 3 task and the triangle task
# Method: rebinds the module globals for move tokens, state tokens, and vocabulary size
# Why: the training and scoring functions read these globals, so one switch reuses the identical pipeline for both tasks
# 역할: 과제 전역 토큰 설정을 mod 3 과제와 삼각형 과제 사이에서 전환한다
# 방법: 이동 토큰, 상태 토큰, 어휘 크기의 모듈 전역을 다시 묶는다
# 이유: 학습과 채점 함수가 이 전역을 읽으므로 전환 하나로 두 과제가 동일한 파이프라인을 공유하기 때문이다
def p_set_task(mv, st, vocab):
    global MV, ST, VOCAB
    MV = mv
    ST = st
    VOCAB = vocab


# Role: builds the six states of the labeled triangle and the rule table of the three moves
# Method: enumerates the six permutations of three vertices in sorted order, composes each state with the rotation, the double rotation, and the flip, and indexes the results
# Why: the triangle symmetries form the smallest non commutative group, so this table is the ground truth the discrimination scores against
# 역할: 표시된 정삼각형의 상태 6종과 이동 3종의 규칙 표를 만든다
# 방법: 꼭짓점 세 개의 순열 6종을 정렬 순서로 나열하고 각 상태에 회전, 두 번 회전, 뒤집기를 합성해 결과를 색인한다
# 이유: 정삼각형의 대칭은 가장 작은 비가환 군이라 이 표가 판별이 채점하는 정답이기 때문이다
def p_tri_table():
    _perms = sorted(itertools.permutations((0, 1, 2)))
    _idx = {}
    for i, p in enumerate(_perms):
        _idx[p] = i

    def p_compose(a, b):
        return tuple(b[a[i]] for i in range(3))

    _r = (1, 2, 0)
    _f = (0, 2, 1)
    _mv = [_r, p_compose(_r, _r), _f]
    _trans = [[_idx[p_compose(_perms[s], _mv[m])] for m in range(3)] for s in range(6)]
    return _trans


# Role: returns the state index after applying a move word from the start state
# Method: walks the rule table move by move
# Why: row building, ceilings, and the self check all need the same walk
# 역할: 시작 상태에서 이동 낱말을 적용한 뒤의 상태 색인을 돌려준다
# 방법: 규칙 표를 이동 하나씩 따라간다
# 이유: 행 생성, 천장 계산, 자가검사가 전부 같은 걸음을 쓰기 때문이다
def p_tri_end(trans, word):
    _s = 0
    for mv in word:
        _s = trans[_s][mv]
    return _s


# Role: lists the rule table entries a move word passes through
# Method: yields the state and move pair before every step of the walk
# Why: split coverage and blank solvability are defined over these entries
# 역할: 이동 낱말이 지나가는 규칙 표 항목을 나열한다
# 방법: 걸음마다 직전 상태와 이동의 쌍을 내놓는다
# 이유: 분할의 노출과 빈칸의 풀이 가능성이 이 항목 위에서 정의되기 때문이다
def p_tri_pairs(trans, word):
    _s = 0
    _out = []
    for mv in word:
        _out.append((_s, mv))
        _s = trans[_s][mv]
    return _out


# Role: verifies the rule table against the defining relations of the triangle group
# Method: checks F then R equals R2 then F, R then F has order two, the two orders differ, and length 3 words reach all six states
# Why: every downstream number leans on this table, so the table itself must be proven before any training
# 역할: 규칙 표를 삼각형 군의 정의 관계와 대조해 검증한다
# 방법: F 뒤 R 이 R2 뒤 F 와 같은지, R 뒤 F 가 위수 2 인지, 두 순서가 다른지, 길이 3 낱말이 상태 6종을 모두 닿는지 검사한다
# 이유: 아래 모든 수치가 이 표에 기대므로 학습 전에 표 자체가 증명되어야 하기 때문이다
def p_tri_selfcheck(trans):
    assert p_tri_end(trans, [2, 0]) == p_tri_end(trans, [1, 2])
    assert p_tri_end(trans, [0, 2, 0, 2]) == 0
    assert p_tri_end(trans, [0, 2]) != p_tri_end(trans, [2, 0])
    _ends = set(p_tri_end(trans, w) for w in itertools.product([0, 1, 2], repeat=3))
    assert len(_ends) == 6
    print("    군 표 자가검사: F뒤R=R2뒤F 통과, RF 위수 2 통과, 비가환 통과, 길이 3 끝상태 6가지 통과", flush=True)


# Role: builds triangle rows where the state is written after every move
# Method: for each move word, left-pads to the fixed window and appends move token then state token following the rule table
# Why: the same loop protocol as the mod 3 task keeps the two discriminations comparable step for step
# 역할: 이동 하나마다 상태를 쓰는 삼각형 행을 만든다
# 방법: 이동 낱말마다 고정 창으로 왼쪽을 채우고 규칙 표를 따라 이동 토큰과 상태 토큰을 차례로 붙인다
# 이유: mod 3 과제와 같은 루프 프로토콜이어야 두 판별이 걸음 단위로 비교되기 때문이다
def p_make_rows_tri(lens, trans):
    _rows = []
    _slots = []
    _words = []
    for k in lens:
        for moves in itertools.product([0, 1, 2], repeat=k):
            _seq = [PAD] * (T - 2 * k)
            _bpos = []
            _run = 0
            for mv in moves:
                _bpos.append(len(_seq))
                _seq.append(TRI_MV[mv])
                _run = trans[_run][mv]
                _seq.append(TRI_ST[_run])
            _rows.append(_seq)
            _slots.append(_bpos)
            _words.append(moves)
    return np.asarray(_rows, dtype=np.int64), _slots, _words


# Role: finds the smallest split seed whose 14 kept words expose the full rule table
# Method: walks seeds upward, keeps the first permutation whose training words together with the length 2 words cover all 18 entries
# Why: the composition trial must expose every rule so that the blanks test pure composition and nothing else
# 역할: 남긴 14개 낱말이 규칙 표 전체를 노출하는 가장 작은 분할 시드를 찾는다
# 방법: 시드를 올려 가며 길이 2 낱말과 합쳐 18항목을 전부 덮는 첫 순열을 잡는다
# 이유: 합성시험은 규칙 전부를 노출해야 빈칸이 순수 합성만을 시험하기 때문이다
def p_tri_cover_seed(trans, words2, words3):
    for s in range(1000):
        _perm = np.random.RandomState(s).permutation(len(words3))
        _exp = set()
        for w in words2:
            _exp.update(p_tri_pairs(trans, w))
        for i in _perm[:14]:
            _exp.update(p_tri_pairs(trans, words3[i]))
        if len(_exp) == 18:
            return s, _perm
    raise RuntimeError("커버리지 분할 탐색 실패")


# Role: computes the ceiling of any order-blind solver on the blank words
# Method: counts the blanks whose end state is identical across every ordering of the same move multiset over all 27 words
# Why: a solver that ignores order answers one state per multiset, so it can only be sure on blanks whose answer does not depend on order at all
# 역할: 순서를 무시하는 풀이가 빈칸에서 닿을 수 있는 천장을 계산한다
# 방법: 낱말 27개 전체에서 같은 이동 다중집합의 모든 순서가 같은 끝상태를 내는 빈칸을 센다
# 이유: 순서를 무시하는 풀이는 다중집합마다 상태 하나만 답하므로 순서와 무관하게 답이 정해지는 빈칸만 확실히 맞힐 수 있기 때문이다
def p_tri_commutative_ceiling(trans, all_words, blank_words):
    _classes = {}
    for w in all_words:
        _classes.setdefault(tuple(sorted(w)), set()).add(p_tri_end(trans, w))
    _hit = 0
    for w in blank_words:
        if len(_classes[tuple(sorted(w))]) == 1:
            _hit += 1
    return _hit


# Role: opens a trained model's rule table and prints the entries it failed to learn
# Method: forward passes all 27 length 3 rows with true states shown and scores every state slot by its rule table entry
# Why: the accuracy number says how many blanks fall but only this table says which rule fell, which is the autopsy the paper reports
# 역할: 학습된 모델의 규칙 표를 열어 습득에 실패한 항목을 인쇄한다
# 방법: 길이 3 행 27개 전부를 정답 상태를 보여주며 순전파하고 상태 자리마다 규칙 표 항목별로 채점한다
# 이유: 정확도 수치는 빈칸이 몇 개 무너지는지만 말하고 어느 규칙이 무너졌는지는 이 표만 말하며 그것이 본문이 보고하는 부검이기 때문이다
def p_tri_transition_report(m, trans, rows3, slots3, words3):
    _hit = {}
    _tot = {}
    for i in range(0, len(rows3), 240):
        _chunk = rows3[i:i + 240]
        _cache, _aD, _z = bc.forward(m, xp.asarray(_chunk.T))
        _zh = bc.to_host(_z.reshape(VOCAB, T, -1))
        for j in range(len(_chunk)):
            _pairs = p_tri_pairs(trans, words3[i + j])
            for si, bp in enumerate(slots3[i + j]):
                _sc = [_zh[t, bp, j] for t in TRI_ST]
                _pred = TRI_ST[int(np.argmax(_sc))]
                _key = _pairs[si]
                _tot[_key] = _tot.get(_key, 0) + 1
                if _pred == _chunk[j, bp + 1]:
                    _hit[_key] = _hit.get(_key, 0) + 1
    _fails = []
    for key in sorted(_tot):
        if _hit.get(key, 0) < _tot[key]:
            _fails.append(f"P{key[0]}에 {TRI_NAME[key[1]]} ({_hit.get(key, 0)}/{_tot[key]})")
    if _fails:
        print(f"    [규칙 표] 18항목 중 오류 {len(_fails)}개: " + ", ".join(_fails), flush=True)
    else:
        print("    [규칙 표] 18항목 전부 정답 (완벽 습득)", flush=True)


# Role: runs the whole triangle discrimination, both splits, all groups
# Method: switches the task globals, builds the rule table and rows, runs the inference trial on the seed 42 random split and the composition trial on the coverage guaranteed split, then switches the globals back
# Why: the mod 3 task asks whether a gate owns one cyclic operation, and this task asks whether it owns non commutative composition, the next rung of the ladder
# 역할: 삼각형 판별 전체를 두 분할과 전 실험군으로 돌린다
# 방법: 과제 전역을 전환하고 규칙 표와 행을 만들어 시드 42 무작위 분할로 유추시험을, 커버리지 보장 분할로 합성시험을 돌린 뒤 전역을 되돌린다
# 이유: mod 3 과제는 게이트가 순환 연산 하나를 가졌는지를 묻고 이 과제는 비가환 합성을 가졌는지를 물으며 그것이 사다리의 다음 단이기 때문이다
def p_run_triangle():
    global SEEDS
    _trans = p_tri_table()
    p_set_task(TRI_MV, TRI_ST, TRI_VOCAB)
    _seeds_saved = SEEDS
    SEEDS = TRI_SEEDS
    try:
        p_tri_selfcheck(_trans)
        _rows2, _slots2, _words2 = p_make_rows_tri([2], _trans)
        _rows3, _slots3, _words3 = p_make_rows_tri([3], _trans)
        _exp2 = set()
        for w in _words2:
            _exp2.update(p_tri_pairs(_trans, w))
        _splits = []
        _perm42 = np.random.RandomState(42).permutation(len(_words3))
        _splits.append(("유추시험 (무작위 분할)", _perm42, None))
        _cs, _permc = p_tri_cover_seed(_trans, _words2, _words3)
        _splits.append((f"합성시험 (커버리지 보장 분할, 탐색 시드 {_cs})", _permc, _cs))
        for label, perm, cs in _splits:
            _keep = perm[:14]
            _hold = perm[14:]
            _exp = set(_exp2)
            for i in _keep:
                _exp.update(p_tri_pairs(_trans, _words3[i]))
            _solv = sum(1 for i in _hold if all(p in _exp for p in p_tri_pairs(_trans, _words3[i])))
            print(f"\n  [{label}] 규칙 표 노출 {len(_exp)}/18, 본 규칙만으로 풀리는 빈칸 {_solv}/{len(_hold)}", flush=True)
            _ceil = p_tri_commutative_ceiling(_trans, _words3, [_words3[i] for i in _hold])
            print(f"    가환 천장(순서 무시 풀이의 상계): 빈칸 {_ceil}/{len(_hold)}. 우연은 16.7%", flush=True)
            _train_rows = np.concatenate([_rows2, _rows3[_keep]])
            _train_slots = _slots2 + [_slots3[i] for i in _keep]
            _tests = {"본것": (_train_rows, _train_slots), "빈칸3": (_rows3[_hold], [_slots3[i] for i in _hold])}
            for k in TEST_LENS:
                _rk, _sk, _wk = p_make_rows_tri([k], _trans)
                _tests[f"길이{k}"] = (_rk, _sk)
            _report = None
            if cs is not None:
                _report = lambda m: p_tri_transition_report(m, _trans, _rows3, _slots3, _words3)
            p_run_group("A σ켬", None, True, _train_rows, _train_slots, _tests)
            p_run_group("B 차단", None, False, _train_rows, _train_slots, _tests)
            p_run_group("D tanh", p_patch_tanh, True, _train_rows, _train_slots, _tests)
            p_run_group("E 회전", paper7.install, True, _train_rows, _train_slots, _tests, report=_report)
            p_run_group("Q 쿼터니언", paper7.install_quaternion, True, _train_rows, _train_slots, _tests, report=_report)
            p_run_torch_group(_train_rows, _train_slots, _tests)
    finally:
        SEEDS = _seeds_saved
        p_set_task([1, 2, 3], [4, 5, 6], 8)


def main():
    _rows2, _slots2 = p_make_rows([2])
    _rows3, _slots3 = p_make_rows([3])
    _rng = np.random.RandomState(42)
    _perm = _rng.permutation(len(_rows3))
    _keep = _perm[:14]
    _hold = _perm[14:]
    _train_rows = np.concatenate([_rows2, _rows3[_keep]])
    _train_slots = _slots2 + [_slots3[i] for i in _keep]
    _tests = {"본것": (_train_rows, _train_slots), "빈칸3": (_rows3[_hold], [_slots3[i] for i in _hold])}
    for k in TEST_LENS:
        _tests[f"길이{k}"] = p_make_rows([k])
    print("[제7편 회전 연산자] 과제: 수 M0 M1 M2 를 읽으며 누적 합 mod 3 상태를 자리마다 말한다", flush=True)
    print(f"  학습: 길이2 전부 9개 + 길이3 절반 {len(_keep)}개. 길이3 나머지 {len(_hold)}개는 숨김(빈칸)", flush=True)
    print("  조건: A B D E = 같은 반야 1블록, 게이트만 교체. C = 표준 1층(torch 선택 의존)", flush=True)
    print("\n[1] 회전 수반 정합 (식이 유한차분과 같은가)", flush=True)
    paper7.adjoint_check()
    print("\n[2] 설계 증명 (커널 주입이 계산을 바꾸지 않는가)", flush=True)
    p_plumbing_proof(_train_rows, _train_slots)
    print("\n[3] 깊이 배제 판별 (빈칸과 안 본 길이, 강제/자기 되먹임)", flush=True)
    p_run_group("A σ켬", None, True, _train_rows, _train_slots, _tests)
    p_run_group("B 차단", None, False, _train_rows, _train_slots, _tests)
    p_run_group("D tanh", p_patch_tanh, True, _train_rows, _train_slots, _tests)
    p_run_group("E 회전", paper7.install, True, _train_rows, _train_slots, _tests, inspect=True)
    p_run_torch_group(_train_rows, _train_slots, _tests)
    print("\n[4] 쿼터니언 수반 정합 (식이 유한차분과 같은가)", flush=True)
    paper7.adjoint_check_quaternion()
    print("\n[5] 삼각형 판별 (비가환 과제. 유추시험과 합성시험, 강제/자기 되먹임)", flush=True)
    print("  과제: 이동 R R2 F 를 읽으며 표시된 정삼각형의 상태 P0 부터 P5 를 자리마다 말한다", flush=True)
    p_run_triangle()


if __name__ == "__main__":
    main()
