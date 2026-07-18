# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 6 vagueness trigger probe. Measures the circuit in which vagueness triggers reinterpretation and certainty rises with repetition until the routing changes.
Extracts the hidden at the position whose next-word prediction is most vague in a real corpus, places the immediately preceding context tokens in front, and repeats the accompanied reinterpretation.
Trial k places the vague hidden after the first k tokens of the context, all depths run as one batch, and the depth whose maximum probability rises the most is adopted each round.
A control that reinterprets the hidden alone without context runs alongside to determine whether the rise owes to the accompanying context.
Only the rise in certainty is measured and answer correctness is not. The labels and thresholds reuse those of the Paper 6 distribution routing as they are.
Loads the model through the shared foundation banya_core, runs the forward pass, and reuses the Paper 6 state thresholds from the paper6_metacog module. GPU-only cupy.
Run  python3 paper6_vague_trigger_probe.py

반야 제6편 모호함 트리거 프로브. 모호함이 재해석을 발동시키고 반복할수록 확실도가 올라 라우팅이 바뀌는 회로를 실측한다.
실코퍼스에서 다음 낱말 예측이 가장 모호한 자리의 히든을 뽑고 그 직전 문맥 토큰들을 전방에 배치해 동반 재해석을 반복한다.
시도 k 는 문맥 앞 k토큰 뒤에 모호 히든을 놓은 것이고 전 깊이를 한 배치로 돌려 최대 확률이 가장 오르는 깊이를 매회 채택한다.
문맥 없이 히든 단독으로 재해석하는 대조도 같이 돌려 상승이 문맥 동반 덕인지 가린다.
확실도 상승만 재고 정답 정오는 재지 않는다. 라벨과 문턱값은 제6편 분포 라우팅 것을 그대로 쓴다.
공통 토대 banya_core 로 모델을 불러 순전파하고 제6편 상태 문턱은 논문6_메타인지 에서 가져다 쓴다. GPU 전용 cupy.
실행  python3 paper6_vague_trigger_probe.py"""
import os
import sys
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper6_metacog as paper6

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "bitok_elem2_170000_m.npz")
CORPUS = "banya_world_data/elem_dialog.npy"
STAGE_DEPTH = 16
ROUNDS = 10
SAMPLE_WINDOW = 20
MOHO_COUNT = 4
SEED = 0


# Role: feeds embeddings directly through all blocks and returns the final hidden
# Method: sets the operate gate on the model, unrolls the lag kernels into Toeplitz form, then runs the blocks forward in order
# Why: the vague hidden has no token id, so it cannot ride the shared forward pass and must be placed directly in the embedding slot
# 역할: 임베딩을 직접 넣어 블록 전부를 통과시키고 마지막 히든을 돌려준다
# 방법: 연산 게이트를 모델에 걸어두고 거리커널을 토플리츠로 편 뒤 블록을 차례로 순전파한다
# 이유: 모호 히든은 토큰 아이디가 없어 공통 순전파에 못 태우고 임베딩 자리에 직접 놓아야 하기 때문
def p_blocks_forward(m, x, operate_gate):
    _T = x.shape[1]
    _B = x.shape[2]
    m.m_vec_operate_gate = operate_gate
    if bc.USE_RELATIVE_MIX and not bc.TIME_MIX_USE_FFT and bc.BLOCKWISE_CAUSAL_MIX:
        m.m_cmix_all = bc.toep_build(m.m_mat_w_lag)
    _Hh = bc.HIDDEN_SIZE // bc.NUMBER_TIME_MIX_HEAD
    for bl in range(bc.NUMBER_BLOCK):
        x, _ = bc.block_fwd(m, x, bl, _T, _B, _Hh)
    return x.reshape(bc.HIDDEN_SIZE, _T * _B)


# Role: computes the answer-slot logits of a single hidden
# Method: follows the same branches as the shared forward pass, choosing the low-rank head, the full head, or the tied embedding, multiplies, and adds the bias
# Why: the head structure differs per checkpoint, so following exactly the switches that load turned on yields the same logits as the main forward pass
# 역할: 히든 한 개의 답 자리 로짓을 구한다
# 방법: 공통 순전파와 같은 분기로 저계수 헤드나 통짜 헤드나 묶인 임베딩을 골라 곱하고 치우침을 더한다
# 이유: 체크포인트마다 헤드 구조가 달라 load 가 켠 스위치 그대로 따라가야 본 순전파와 같은 로짓이 나오기 때문
def p_logits(m, col):
    _c = col.reshape(bc.HIDDEN_SIZE, 1)
    if bc.USE_LOWRANK:
        _z = m.m_mat_w_head_a @ (m.m_mat_w_head_b @ _c) + m.m_vec_w_head_bias
    elif bc.USE_WEIGHT_TIE:
        _z = m.m_mat_w_data_axis.T @ _c + m.m_vec_w_head_bias
    else:
        _z = m.m_mat_w_head @ _c + m.m_vec_w_head_bias
    return bc.to_host(_z[:, 0]).astype(np.float64)


# Role: measures the state label and maximum probability of a single hidden
# Method: summarizes the logits with the Paper 6 distribution statistics and splits certain, vague, and unknown with the Paper 6 thresholds as they are
# Why: judging trigger firing and routing switches by the same yardstick as the main-text routing is what makes this code-level evidence
# 역할: 히든 한 개의 상태 라벨과 최대 확률을 잰다
# 방법: 로짓을 제6편 분포 통계로 요약하고 제6편 문턱값 그대로 확실 모호 모름을 가른다
# 이유: 트리거 발동과 라우팅 전환 판정을 본문 라우팅과 같은 잣대로 해야 코드 근거가 되기 때문
def p_state(m, col):
    _pmax, _entropy, _peaks = paper6.dist_stats(p_logits(m, col))
    if _pmax >= paper6.GATE_PROBABILITY_MAX_HIGH and _peaks <= paper6.GATE_PEAKS_HIGH:
        _label = "확실"
    elif _pmax < paper6.GATE_PROBABILITY_MAX_LOW:
        _label = "모름"
    else:
        _label = "모호"
    return _label, _pmax


# Role: extracts the hiddens at the most vague prediction positions in a real corpus together with their preceding context
# Method: runs multiple windows forward, measures the maximum probability at each position, picks vague candidates in ascending order, and stores the preceding tokens as material for the accompanied reinterpretation
# Why: the trigger input must be vagueness that arose in real data rather than in artificial sentences
# 역할: 실코퍼스에서 예측이 가장 모호한 자리의 히든과 직전 문맥을 뽑는다
# 방법: 창을 여럿 순전파해 자리마다 최대 확률을 재고 낮은 순으로 모호 후보를 고르며 직전 토큰들을 동반 재해석 재료로 같이 담는다
# 이유: 인위 문장이 아니라 실제 데이터에서 생긴 모호가 트리거의 입력이어야 하기 때문
def p_collect_moho(m, corp):
    _rng = np.random.RandomState(SEED)
    _cands = []
    for _ in range(SAMPLE_WINDOW):
        i = _rng.randint(0, len(corp) - bc.CONTEXT_LENGTH - 1)
        _w = np.asarray(corp[i:i + bc.CONTEXT_LENGTH], dtype=np.int64).reshape(-1, 1)
        _, _aD, _z = bc.forward(m, xp.asarray(_w))
        for t in range(STAGE_DEPTH, bc.CONTEXT_LENGTH):
            _zz = bc.to_host(_z[:, t]).astype(np.float64)
            _p = np.exp(_zz - _zz.max())
            _p /= _p.sum()
            _cands.append((float(_p.max()), _aD[:, t].copy(), [int(a) for a in _w[t - STAGE_DEPTH:t, 0]]))
    _cands.sort(key=lambda c: c[0])
    return _cands[:MOHO_COUNT]


# Role: reinterprets the vague hidden together with the front-placed context and returns the new hidden and maximum probability per depth
# Method: trial k places the vague hidden after the first k tokens of the context, and all depths are built as columns of one batch and forwarded at once
# Why: which depth fits best is unknown in advance, so all must be measured and the depth that rises the most adopted each round
# 역할: 전방 배치 문맥과 함께 모호 히든을 재해석해 깊이별 새 히든과 최대 확률을 돌려준다
# 방법: 시도 k 는 문맥 앞 k토큰 뒤에 모호 히든을 놓은 것이고 전 깊이를 한 배치의 열로 만들어 한 번에 순전파한다
# 이유: 어느 깊이가 잘 맞는지 미리 모르므로 전부 재보고 매회 가장 오르는 깊이를 채택해야 하기 때문
def p_context_reinterpret(m, v, inp):
    _n = len(inp)
    _T = _n + 1
    _x = xp.zeros((bc.HIDDEN_SIZE, _T, _n), dtype=bc.DATA_TYPE)
    _eg = xp.zeros((bc.HIDDEN_SIZE, _T, _n), dtype=bc.DATA_TYPE)
    _ii = xp.asarray(inp, dtype=xp.int64)
    for k in range(1, _n + 1):
        _lane = k - 1
        _x[:, :k, _lane] = m.m_mat_w_data_axis[:, _ii[:k]] + m.m_mat_w_position[:, :k]
        _eg[:, 1:k, _lane] = m.m_mat_w_operate_axis[:, _ii[:k - 1]]
        _eg[:, k, _lane] = m.m_mat_w_operate_axis[:, _ii[k - 1]]
        _x[:, k, _lane] = xp.asarray(v, dtype=bc.DATA_TYPE) + m.m_mat_w_position[:, k]
    _aD = p_blocks_forward(m, _x, _eg.reshape(bc.HIDDEN_SIZE, _T * _n)).reshape(bc.HIDDEN_SIZE, _T, _n)
    _hid = _aD[:, xp.arange(_n, dtype=xp.int64) + 1, xp.arange(_n)]
    _pms = []
    for i in range(_n):
        _label, _pmax = p_state(m, _hid[:, i])
        _pms.append(_pmax)
    return _hid, _pms


# Role: runs the control that reinterprets the vague hidden alone without context
# Method: places the single hidden in the first slot and runs the forward pass without the operate gate
# Why: whether the rise owes to the accompanying context or to mere repeated passes must be distinguished
# 역할: 문맥 없이 모호 히든 단독으로 재해석하는 대조를 돌린다
# 방법: 히든 하나를 첫 자리에 놓고 연산 게이트 없이 순전파한다
# 이유: 상승이 문맥 동반 덕인지 반복 통과 덕인지 가려야 하기 때문
def p_alone_reinterpret(m, v):
    _x = xp.zeros((bc.HIDDEN_SIZE, 1, 1), dtype=bc.DATA_TYPE)
    _x[:, 0, 0] = xp.asarray(v, dtype=bc.DATA_TYPE) + m.m_mat_w_position[:, 0]
    _aD = p_blocks_forward(m, _x, xp.zeros((bc.HIDDEN_SIZE, 1), dtype=bc.DATA_TYPE))
    _vnew = _aD[:, 0]
    _label, _pmax = p_state(m, _vnew)
    return _vnew, _pmax


def main():
    m, tok = bc.load_from(FROZEN)
    _corp = np.load(CORPUS, mmap_mode="r")
    print(f"[제6편 모호함 트리거] 얼린 모델 {os.path.basename(FROZEN)} · 코퍼스 {os.path.basename(CORPUS)} · 문맥 깊이 1~{STAGE_DEPTH} · 재해석 {ROUNDS}회", flush=True)
    print(f"문턱값은 본문 라우팅 그대로 확실 {paper6.GATE_PROBABILITY_MAX_HIGH} 모름 {paper6.GATE_PROBABILITY_MAX_LOW}. 확실도 상승만 재고 정답 정오는 재지 않는다", flush=True)

    _mohos = p_collect_moho(m, _corp)
    print(f"\n[트리거 입력] 실코퍼스에서 최대 확률 낮은 순으로 모호 히든 {len(_mohos)}개", flush=True)
    for kk, (_base, _v0, _inp) in enumerate(_mohos):
        _label, _ = p_state(m, xp.asarray(_v0))
        _txt = tok.decode(_inp).replace("\n", " ")
        print(f"  모호{kk + 1} pmax {_base:.2f} 상태 {_label} · 문맥 …{_txt[-24:]}", flush=True)

    print("\n[재해석 반복] 상태가 확실 아니면 트리거 발동. on 은 문맥 동반 재해석 off 는 단독 재해석", flush=True)
    _on_traj = []
    _off_traj = []
    _cross = []
    _cross_off = []
    for kk, (_base, _v0, _inp) in enumerate(_mohos):
        _von = xp.asarray(_v0)
        _voff = xp.asarray(_v0)
        _pon = [_base]
        _poff = [_base]
        for _ in range(ROUNDS):
            _hid, _pms = p_context_reinterpret(m, _von, _inp)
            j = int(np.argmax(_pms))
            _von = _hid[:, j].copy()
            _pon.append(_pms[j])
            _voff, _pmf = p_alone_reinterpret(m, _voff)
            _poff.append(_pmf)
        _on_traj.append(_pon)
        _off_traj.append(_poff)
        _hit = next((r for r, p in enumerate(_pon) if p >= paper6.GATE_PROBABILITY_MAX_HIGH), None)
        _cross.append(_hit)
        _cross_off.append(next((r for r, p in enumerate(_poff) if p >= paper6.GATE_PROBABILITY_MAX_HIGH), None))
        _state_end, _ = p_state(m, _von)
        _hit_txt = f"{_hit}회에 확실 문턱 도달" if _hit is not None else "문턱 미달"
        print(f"  모호{kk + 1} on  {' '.join(f'{p:.2f}' for p in _pon)}  {_hit_txt} · 끝 상태 {_state_end}", flush=True)
        print(f"  모호{kk + 1} off {' '.join(f'{p:.2f}' for p in _poff)}", flush=True)

    _early = 3
    _on_gain = float(np.mean([t[_early] - t[0] for t in _on_traj]))
    _off_gain = float(np.mean([t[_early] - t[0] for t in _off_traj]))
    _on_mean = [float(np.mean([t[r] for t in _on_traj])) for r in range(ROUNDS + 1)]
    _off_mean = [float(np.mean([t[r] for t in _off_traj])) for r in range(ROUNDS + 1)]
    _hits = [c for c in _cross if c is not None]
    _hits_off = [c for c in _cross_off if c is not None]
    print("\n[실측 요약] 대조 off 도 여러 회 돌면 결국 한 점으로 수렴해 최대 확률이 튀므로 초반 상승과 도달 속도로 비교한다", flush=True)
    print(f"  재해석 {_early}회까지 확실도 상승 문맥 동반 on {_on_gain:+.3f} (목표 +0.88) · 단독 off {_off_gain:+.3f} (목표 -0.06)", flush=True)
    print(f"  확실 문턱 {paper6.GATE_PROBABILITY_MAX_HIGH} 도달 on {len(_hits)}/{len(_mohos)}개" + (f" 평균 {np.mean(_hits):.1f}회" if _hits else "") + " (목표 4개 전부 1회 도달)" + f" · off {len(_hits_off)}/{len(_mohos)}개" + (f" 평균 {np.mean(_hits_off):.1f}회" if _hits_off else ""), flush=True)
    print(f"  평균 궤적 on  {' '.join(f'{p:.2f}' for p in _on_mean)}", flush=True)
    print(f"  평균 궤적 off {' '.join(f'{p:.2f}' for p in _off_mean)}", flush=True)


if __name__ == "__main__":
    main()
