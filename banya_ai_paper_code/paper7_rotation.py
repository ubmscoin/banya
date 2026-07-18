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
학습 궤적 완전 일치를 확인해 증명한다.

The quaternion gate extends the same recipe one rung up the ladder. Channels are grouped in fours, the previous
token's operation plane block a becomes a unit quaternion q by adding the identity offset and normalizing,
u = (1 + a0, a1, a2, a3), q = u over norm u, and the gate left-multiplies each four channel block: h = xf + q x v.
Left multiplication by a unit quaternion rotates two overlapping planes at once, so compositions do not commute,
which the pairwise planar rotation gate cannot express. The adjoint is again closed form and storage free:
delta v = q conjugate x delta h, delta q = delta h x r conjugate x q with r = h - xf recovered, and the
normalization backward is delta u = (delta q - q (q dot delta q)) over norm u. install_quaternion() injects it
the same way and restore() puts the published kernels back.

쿼터니언 게이트는 같은 처방을 사다리 한 단 위로 확장한 것이다. 채널을 넷씩 묶고 앞 토큰 연산면 묶음 a 에
항등 오프셋을 더해 정규화한 단위 쿼터니언 q 로 각 묶음을 좌곱한다. u = (1 + a0, a1, a2, a3), q = u 나누기
u 의 노름, h = xf + q 곱 v 다. 단위 쿼터니언 좌곱은 겹치는 두 평면을 동시에 돌리므로 합성이 비가환이고,
이는 쌍별 평면 회전 게이트가 표현하지 못하는 성질이다. 수반은 역시 닫힌형에 무저장이다.
델타v = q 켤레 곱 델타h, 델타q = 델타h 곱 r 켤레 곱 q 이고 r = h - xf 로 복원하며, 정규화의 역방향은
델타u = (델타q - q (q 내적 델타q)) 나누기 u 노름이다. install_quaternion() 이 같은 방식으로 주입하고
restore() 가 발행 커널로 되돌린다."""
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


# Role: Hamilton product of two quaternion component sets
# Method: applies the Hamilton multiplication formula componentwise on arrays
# Why: every quaternion gate path repeats this product and one shared formula removes transcription mistakes
# 역할: 두 쿼터니언 성분 묶음의 해밀턴 곱을 계산한다
# 방법: 해밀턴 곱 공식을 배열 성분별로 적용한다
# 이유: 쿼터니언 게이트의 모든 경로가 이 곱을 반복하므로 공식 하나를 공유해야 옮겨 적기 실수가 사라지기 때문이다
def p_quat_mul(a0, a1, a2, a3, b0, b1, b2, b3):
    _w = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    _x = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    _y = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    _z = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
    return _w, _x, _y, _z


# Role: reads the operation plane block as a unit quaternion
# Method: adds the identity offset to the first component and divides the four components by their norm
# Why: a zero operation plane must give the identity quaternion, keeping the identity start convention, and a unit quaternion preserves norm and makes the storage free recovery exact
# 역할: 연산면 묶음을 단위 쿼터니언으로 읽는다
# 방법: 첫 성분에 항등 오프셋을 더하고 네 성분을 노름으로 나눈다
# 이유: 연산면이 0 이면 항등 쿼터니언이어야 항등 출발 규약이 유지되고, 단위 쿼터니언이어야 노름이 보존되며 무저장 복원이 정확해지기 때문이다
def p_quat_read(eg):
    _u0 = 1.0 + eg[0::4, :]
    _u1 = eg[1::4, :]
    _u2 = eg[2::4, :]
    _u3 = eg[3::4, :]
    _n = (_u0 * _u0 + _u1 * _u1 + _u2 * _u2 + _u3 * _u3) ** 0.5
    return _u0 / _n, _u1 / _n, _u2 / _n, _u3 / _n, _n


# Role: quaternion gate forward. Adds the context mixed vector left-multiplied blockwise by the previous token's quaternion
# Method: groups channels in fours, reads the shifted operation plane block as a unit quaternion, and adds the Hamilton product q times v to the stream
# Why: left multiplication by a unit quaternion rotates two overlapping planes at once, so compositions do not commute, which the pairwise planar rotation cannot express
# 역할: 쿼터니언 게이트 순전파. 문맥 혼합 벡터를 앞 토큰의 쿼터니언으로 묶음별 좌곱해 잔차에 더한다
# 방법: 채널을 넷씩 묶고 이동된 연산면 묶음을 단위 쿼터니언으로 읽어 해밀턴 곱 q 곱 v 를 더한다
# 이유: 단위 쿼터니언 좌곱은 겹치는 두 평면을 동시에 돌려 합성이 비가환이고 이는 쌍별 평면 회전이 표현하지 못하기 때문이다
def p_gate_fwd_quaternion(xf, xm, eg, WG, BG, Mt, h):
    _q0, _q1, _q2, _q3, _n = p_quat_read(eg)
    _r0, _r1, _r2, _r3 = p_quat_mul(_q0, _q1, _q2, _q3, xm[0::4, :], xm[1::4, :], xm[2::4, :], xm[3::4, :])
    h[:] = xf
    h[0::4, :] += _r0
    h[1::4, :] += _r1
    h[2::4, :] += _r2
    h[3::4, :] += _r3


# Role: quaternion gate backward. Returns the stream credit, the mixing credit, and the operation plane credit
# Method: the mixing credit is the conjugate left product, the quaternion credit is delta h times r conjugate times q with r recovered as fa0 minus xf, and the normalization backward projects out the radial part
# Why: the adjoint of a unit left multiplication is the conjugate left multiplication and the quaternion derivative closes without the forward input, so the adjoint is exact and storage free
# 역할: 쿼터니언 게이트 역방향. 잔차 신용, 혼합 신용, 연산면 신용을 돌려준다
# 방법: 혼합 신용은 켤레 좌곱이고, 쿼터니언 신용은 델타h 곱 r 켤레 곱 q 인데 r 은 fa0 빼기 xf 로 복원하며, 정규화 역방향은 반지름 방향 성분을 빼서 나눈다
# 이유: 단위 좌곱의 수반은 켤레 좌곱이고 쿼터니언 미분이 순전파 입력 없이 닫히므로 수반이 정확하고 무저장이기 때문이다
def p_gate_bwd_quaternion(dl, fa0, xf, eg, WG, BG, Mt, ddir, dlg, dgt):
    _q0, _q1, _q2, _q3, _n = p_quat_read(eg)
    _r0 = fa0[0::4, :] - xf[0::4, :]
    _r1 = fa0[1::4, :] - xf[1::4, :]
    _r2 = fa0[2::4, :] - xf[2::4, :]
    _r3 = fa0[3::4, :] - xf[3::4, :]
    _d0 = dl[0::4, :]
    _d1 = dl[1::4, :]
    _d2 = dl[2::4, :]
    _d3 = dl[3::4, :]
    ddir[:] = dl
    _m0, _m1, _m2, _m3 = p_quat_mul(_q0, -_q1, -_q2, -_q3, _d0, _d1, _d2, _d3)
    dlg[0::4, :] = _m0
    dlg[1::4, :] = _m1
    dlg[2::4, :] = _m2
    dlg[3::4, :] = _m3
    _vb0, _vb1, _vb2, _vb3 = p_quat_mul(_r0, -_r1, -_r2, -_r3, _q0, _q1, _q2, _q3)
    _dq0, _dq1, _dq2, _dq3 = p_quat_mul(_d0, _d1, _d2, _d3, _vb0, _vb1, _vb2, _vb3)
    _dot = _q0 * _dq0 + _q1 * _dq1 + _q2 * _dq2 + _q3 * _dq3
    dgt[:] = 0.0
    dgt[0::4, :] = (_dq0 - _q0 * _dot) / _n
    dgt[1::4, :] = (_dq1 - _q1 * _dot) / _n
    dgt[2::4, :] = (_dq2 - _q2 * _dot) / _n
    dgt[3::4, :] = (_dq3 - _q3 * _dot) / _n


# Role: gate weight gradient stub for the quaternion gate
# Method: writes zeros because the quaternion gate has no valve knob and no bias
# Why: the quaternion is driven only by the operation plane, the same convention as the rotation gate
# 역할: 쿼터니언 게이트의 게이트 무게 기울기 대용
# 방법: 쿼터니언 게이트에는 밸브 손잡이와 치우침이 없으므로 0 을 쓴다
# 이유: 쿼터니언은 연산면만이 이끌기 때문이다. 회전 게이트와 같은 규약이다
def p_gate_gw_quaternion(dl, fa0, xf, eg, WG, BG, Mt, NCH, pw, pb):
    pw[:] = 0.0
    pb[:] = 0.0


# Role: installs the quaternion gate into the engine at run time
# Method: saves the original bitoken gate kernels once and replaces the three module attributes of banya_core
# Why: the quaternion gate enters through the same injection as the rotation gate, so no engine file is touched
# 역할: 쿼터니언 게이트를 실행 중에 엔진에 설치한다
# 방법: 원본 바이토큰 게이트 커널 셋을 한 번 보관하고 banya_core 의 모듈 속성 셋을 바꿔 끼운다
# 이유: 쿼터니언 게이트도 회전 게이트와 같은 주입으로 들어가므로 엔진 파일을 전혀 건드리지 않기 때문이다
def install_quaternion():
    if not _ORIGINAL:
        _ORIGINAL["fwd"] = bc._gate_fwd_bt_k
        _ORIGINAL["bwd"] = bc._gate_bwd_bt_k
        _ORIGINAL["gw"] = bc._gate_gw_bt_k
    bc._gate_fwd_bt_k = p_gate_fwd_quaternion
    bc._gate_bwd_bt_k = p_gate_bwd_quaternion
    bc._gate_gw_bt_k = p_gate_gw_quaternion


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


# Role: verifies the adjoint consistency of the quaternion gate with finite differences
# Method: checks the inner product identity of the conjugate left product, compares the analytic mixing gradient and the storage free operation plane gradient of a scalar loss against central differences by ratio over 10 samples each, and prints a non commutativity measure, in numpy float64
# Why: the quaternion gate enters the exact adjoint engine under the same discipline as the rotation gate, and the printed non commutativity shows it owns the composition order that pairwise planar rotations lack
# 역할: 쿼터니언 게이트의 수반 정합을 유한차분으로 검증한다
# 방법: 켤레 좌곱의 내적 항등을 검사하고, 스칼라 손실의 혼합 기울기와 무저장 연산면 기울기를 중심차분과 비율 표본 10개씩으로 대조하며, 비가환 측정을 인쇄한다. numpy float64 로 계산한다
# 이유: 쿼터니언 게이트도 회전 게이트와 같은 규율로 정확한 전치 엔진에 들어가고, 인쇄한 비가환 값은 쌍별 평면 회전에 없는 합성 순서를 이 게이트가 가짐을 보이기 때문이다
def adjoint_check_quaternion():
    _rng = np.random.RandomState(11)
    _nb = 16

    def p_read(a):
        _u = a.copy()
        _u[0] = 1.0 + _u[0]
        _n = np.sqrt((_u * _u).sum(0))
        return _u / _n, _u, _n

    def p_mul(p, q):
        return np.stack(p_quat_mul(p[0], p[1], p[2], p[3], q[0], q[1], q[2], q[3]))

    def p_conj(p):
        _c = p.copy()
        _c[1:] = -_c[1:]
        return _c

    _a = _rng.randn(4, _nb)
    _v = _rng.randn(4, _nb)
    _d = _rng.randn(4, _nb)
    _q, _u, _n = p_read(_a)
    _r = p_mul(_q, _v)
    _lhs = float((_d * _r).sum())
    _rhs = float((p_mul(p_conj(_q), _d) * _v).sum())
    print(f"    내적 항등 <d,qv>={_lhs:.8f}  <q~d,v>={_rhs:.8f}  차이 {abs(_lhs - _rhs):.2e}", flush=True)
    _w = _rng.randn(4, _nb)
    _dv_ana = p_mul(p_conj(_q), _w)
    _vb = p_mul(p_conj(_r), _q)
    _dq = p_mul(_w, _vb)
    _dot = (_q * _dq).sum(0)
    _da_ana = (_dq - _q * _dot) / _n
    _eps = 1e-6
    _ratio_v = []
    _ratio_a = []
    for k in range(10):
        i = _rng.randint(0, 4)
        j = _rng.randint(0, _nb)
        _vp = _v.copy()
        _vp[i, j] += _eps
        _vm = _v.copy()
        _vm[i, j] -= _eps
        _fd = ((_w * p_mul(_q, _vp)).sum() - (_w * p_mul(_q, _vm)).sum()) / (2 * _eps)
        _ratio_v.append(_fd / (_dv_ana[i, j] + 1e-20))
        _ap = _a.copy()
        _ap[i, j] += _eps
        _am = _a.copy()
        _am[i, j] -= _eps
        _qp = p_read(_ap)[0]
        _qm = p_read(_am)[0]
        _fd_a = ((_w * p_mul(_qp, _v)).sum() - (_w * p_mul(_qm, _v)).sum()) / (2 * _eps)
        _ratio_a.append(_fd_a / (_da_ana[i, j] + 1e-20))
    print(f"    혼합 기울기 비율 중앙값 {np.median(_ratio_v):.6f}  연산면 기울기 비율 중앙값 {np.median(_ratio_a):.6f}", flush=True)
    _b = _rng.randn(4, _nb)
    _p = p_read(_b)[0]
    _pq = p_mul(_p, p_mul(_q, _v))
    _qp = p_mul(_q, p_mul(_p, _v))
    _comm = float(np.sqrt(((_pq - _qp) ** 2).sum(0)).mean())
    print(f"    비가환 |pqv-qpv| 평균 {_comm:.3f} (0 이면 가환. 쌍별 평면 회전은 이 값이 0 이다)", flush=True)
