# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Paper 7 rotation-operator gate. Reads the operation plane as Euler rotation angles.
Fractional differentiation factors into amplitude times Euler rotation, and the logic-bearing part is the rotation,
so the gate replaces the positive-valve sigmoid with a pure rotation: h = xf + R(theta) xm over channel pairs,
where theta comes from the even components of the previous token's operation plane.
The adjoint is closed form and storage free: the transpose is R(-theta), and the angle gradient is the two
dimensional cross product of the incoming credit and the rotated output, so no forward input needs to be stored.
No existing engine file is modified. install() swaps the gate kernels of banya_core at run time and restore() puts
them back; the probe proves this injection changes nothing by running a python reimplementation of the original
sigmoid kernels and matching its training trajectory exactly.

제7편 회전 연산자 게이트. 연산면을 오일러 회전각으로 읽는다.
분수 미분은 진폭 곱하기 오일러 회전으로 인수분해되고 논리를 나르는 쪽은 회전이므로,
게이트를 양수 밸브(시그모이드) 대신 순수 회전으로 둔다. 채널을 둘씩 짝지어 h = xf + R(theta) xm 이고
theta 는 앞 토큰 연산면의 짝수 성분이다.
수반은 닫힌형이고 무저장이다. 전치는 R(-theta) 이고 각도 기울기는 들어온 신용과 회전 결과의
이차원 외적이라 순전파 입력을 저장할 필요가 없다.
기존 엔진 파일은 한 줄도 바꾸지 않는다. install() 이 실행 중에 banya_core 의 게이트 커널을 바꿔 끼우고
restore() 가 되돌린다. 주입이 계산을 바꾸지 않음은 프로브가 원본 시그모이드 커널의 파이썬 재구현으로
학습 궤적 완전 일치를 확인해 증명한다."""
import numpy as np
import banya_core as bc

xp = bc.xp

_ORIGINAL = {}


# Role: rotation gate forward. Adds the context mixed vector rotated pairwise by the previous token's angles
# Method: pairs channel 2k with 2k+1, rotates each pair by theta from the even rows of the shifted operation plane, then adds to the stream
# Why: a pure rotation is unitary, carries sign (180 degrees equals negation), and composes by angle addition, which a positive valve cannot express
# 역할: 회전 게이트 순전파. 문맥 혼합 벡터를 앞 토큰의 각도로 쌍별 회전해 잔차에 더한다
# 방법: 채널 2k 와 2k+1 을 짝지어 이동된 연산면의 짝수 행에서 온 theta 로 각 쌍을 회전한 뒤 더한다
# 이유: 순수 회전은 노름을 보존하고 부호를 가지며(180도가 부정) 합성이 각도 덧셈이라 양수 밸브가 표현 못 하는 연산을 기본 연산으로 가진다
def p_gate_fwd_rotation(xf, xm, eg, WG, BG, Mt, h):
    _theta = eg[0::2, :]
    _c = xp.cos(_theta)
    _s = xp.sin(_theta)
    _u = xm[0::2, :]
    _v = xm[1::2, :]
    _out = xp.empty_like(xm)
    _out[0::2, :] = _c * _u - _s * _v
    _out[1::2, :] = _s * _u + _c * _v
    h[:] = xf + _out


# Role: rotation gate backward. Returns the stream credit, the mixing credit, and the operation plane credit
# Method: the mixing credit is the incoming credit rotated by minus theta, and the angle credit is the cross product of the incoming credit and the rotated output recovered as fa0 minus xf, written to the even rows
# Why: the transpose of a rotation is the conjugate rotation and the angle derivative closes without the forward input, so the adjoint is exact and storage free
# 역할: 회전 게이트 역방향. 잔차 신용, 혼합 신용, 연산면 신용을 돌려준다
# 방법: 혼합 신용은 들어온 신용을 -theta 로 회전한 것이고, 각도 신용은 들어온 신용과 회전 결과(fa0 - xf 로 복원)의 외적을 짝수 행에 쓴다
# 이유: 회전의 전치는 켤레 회전이고 각도 미분이 순전파 입력 없이 닫히므로 수반이 정확하고 무저장이기 때문이다
def p_gate_bwd_rotation(dl, fa0, xf, eg, WG, BG, Mt, ddir, dlg, dgt):
    _theta = eg[0::2, :]
    _c = xp.cos(_theta)
    _s = xp.sin(_theta)
    _rot_u = fa0[0::2, :] - xf[0::2, :]
    _rot_v = fa0[1::2, :] - xf[1::2, :]
    _dlu = dl[0::2, :]
    _dlv = dl[1::2, :]
    ddir[:] = dl
    dlg[0::2, :] = _c * _dlu + _s * _dlv
    dlg[1::2, :] = -_s * _dlu + _c * _dlv
    dgt[:] = 0.0
    dgt[0::2, :] = -_dlu * _rot_v + _dlv * _rot_u


# Role: gate weight gradient stub for the rotation gate
# Method: writes zeros because the rotation gate has no valve knob and no bias
# Why: the rotation is driven only by the operation plane, so the sigmoid gate parameters receive no gradient
# 역할: 회전 게이트의 게이트 무게 기울기 대용
# 방법: 회전 게이트에는 밸브 손잡이와 치우침이 없으므로 0 을 쓴다
# 이유: 회전은 연산면만이 이끌기 때문에 시그모이드 게이트 파라미터는 기울기를 받지 않는다
def p_gate_gw_rotation(dl, fa0, xf, eg, WG, BG, Mt, NCH, pw, pb):
    pw[:] = 0.0
    pb[:] = 0.0


# Role: installs the rotation gate into the engine at run time
# Method: saves the original bitoken gate kernels once and replaces the three module attributes of banya_core
# Why: swapping module attributes leaves every engine file untouched and restore() can undo it exactly
# 역할: 회전 게이트를 실행 중에 엔진에 설치한다
# 방법: 원본 바이토큰 게이트 커널 셋을 한 번 보관하고 banya_core 의 모듈 속성 셋을 바꿔 끼운다
# 이유: 모듈 속성 교체는 엔진 파일을 전혀 건드리지 않고 restore() 로 정확히 되돌릴 수 있기 때문이다
def install():
    if not _ORIGINAL:
        _ORIGINAL["fwd"] = bc._gate_fwd_bt_k
        _ORIGINAL["bwd"] = bc._gate_bwd_bt_k
        _ORIGINAL["gw"] = bc._gate_gw_bt_k
    bc._gate_fwd_bt_k = p_gate_fwd_rotation
    bc._gate_bwd_bt_k = p_gate_bwd_rotation
    bc._gate_gw_bt_k = p_gate_gw_rotation


# Role: restores the original sigmoid gate kernels
# Method: puts back the saved module attributes if install() has run
# Why: the probe alternates gate variants inside one process and must return to the published engine exactly
# 역할: 원본 시그모이드 게이트 커널을 되돌린다
# 방법: install() 이 실행됐다면 보관해 둔 모듈 속성을 되돌려 놓는다
# 이유: 프로브가 한 프로세스 안에서 게이트 변형을 번갈아 쓰므로 발행 엔진으로 정확히 복귀해야 하기 때문이다
def restore():
    if _ORIGINAL:
        bc._gate_fwd_bt_k = _ORIGINAL["fwd"]
        bc._gate_bwd_bt_k = _ORIGINAL["bwd"]
        bc._gate_gw_bt_k = _ORIGINAL["gw"]


# Role: verifies the adjoint consistency of the rotation gate with finite differences
# Method: checks the inner product identity of the transpose, then compares the analytic mixing gradient and angle gradient of a scalar loss against central differences by ratio over 10 samples each, in numpy float64
# Why: a new primitive enters the exact adjoint engine only after its adjoint is shown numerically exact, the same discipline as Paper 1
# 역할: 회전 게이트의 수반 정합을 유한차분으로 검증한다
# 방법: 전치의 내적 항등을 검사하고 스칼라 손실의 혼합 기울기와 각도 기울기를 중심차분과 비율 표본 10개씩으로 대조한다. numpy float64 로 계산한다
# 이유: 새 기본 연산은 수반이 수치적으로 정확함을 보인 뒤에만 정확한 전치 엔진에 들어간다. 제1편과 같은 규율이다
def adjoint_check():
    _rng = np.random.RandomState(11)
    _hh = 64
    _theta = _rng.randn(_hh // 2)
    _xm = _rng.randn(_hh)
    _d = _rng.randn(_hh)

    def p_rot(theta, v):
        _c = np.cos(theta)
        _s = np.sin(theta)
        _o = np.empty_like(v)
        _o[0::2] = _c * v[0::2] - _s * v[1::2]
        _o[1::2] = _s * v[0::2] + _c * v[1::2]
        return _o

    _y = p_rot(_theta, _xm)
    _back = p_rot(-_theta, _d)
    _lhs = float(np.dot(_d, _y))
    _rhs = float(np.dot(_back, _xm))
    print(f"    내적 항등 <d,Rx>={_lhs:.8f}  <R^T d,x>={_rhs:.8f}  차이 {abs(_lhs - _rhs):.2e}", flush=True)
    _w = _rng.randn(_hh)
    _dxm_ana = p_rot(-_theta, _w)
    _rot = p_rot(_theta, _xm)
    _dth_ana = -_w[0::2] * _rot[1::2] + _w[1::2] * _rot[0::2]
    _eps = 1e-6
    _ratio_x = []
    _ratio_t = []
    for k in range(10):
        i = _rng.randint(0, _hh)
        _xp1 = _xm.copy()
        _xp1[i] += _eps
        _xm1 = _xm.copy()
        _xm1[i] -= _eps
        _fd = (np.dot(_w, p_rot(_theta, _xp1)) - np.dot(_w, p_rot(_theta, _xm1))) / (2 * _eps)
        _ratio_x.append(_fd / (_dxm_ana[i] + 1e-20))
        j = _rng.randint(0, _hh // 2)
        _tp = _theta.copy()
        _tp[j] += _eps
        _tm = _theta.copy()
        _tm[j] -= _eps
        _fd_t = (np.dot(_w, p_rot(_tp, _xm)) - np.dot(_w, p_rot(_tm, _xm))) / (2 * _eps)
        _ratio_t.append(_fd_t / (_dth_ana[j] + 1e-20))
    print(f"    혼합 기울기 비율 중앙값 {np.median(_ratio_x):.6f}  각도 기울기 비율 중앙값 {np.median(_ratio_t):.6f}", flush=True)
