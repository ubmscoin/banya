# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya research paper 1 engine. A language-model engine that learns through exact, manually derived transposes without automatic differentiation.
It binds the exact-transpose credit chain and the per-operation closed-form transposes in the shared foundation banya_core into the mechanism of this paper.

반야 연구 제1편 엔진. 자동미분 없이 수동으로 유도한 정확한 전치로 학습하는 언어모델 엔진.
공통 토대 banya_core 에 있는 정확한 전치 신용 사슬과 연산별 닫힌형 전치를 이 논문의 메커니즘으로 묶는다."""


import os
import sys
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import banya_core as bc

BanyaNoBP = bc.BanyaNoBP
forward = bc.forward
p_softmax = bc.p_softmax
head_delta_top = bc.head_delta_top
exact_chain = bc.exact_chain
block_credit = bc.block_credit

rms_fwd = bc.rms_fwd
rms_vjp = bc.rms_vjp
toep_build = bc.toep_build
toep_scatter = bc.toep_scatter
time_mix_forward = bc.time_mix_forward
time_mix_backward = bc.time_mix_backward

scatter_rows = bc.scatter_rows
LEARNING_RATE_ADAM = bc.LEARNING_RATE_ADAM


# Role: an exact-transpose training step that learns one step from a single forward pass
# Method: goes from the embedding through the blocks to the logits, sends the cross-entropy gradient back through the head transpose, then flows the transposes through the blocks in reverse order to update everything down to the embedding
# Why: multiplying only the closed-form transpose of each operation, without an autodiff graph, yields gradients numerically identical to backprop, which is the core of this engine
# 역할: 순전파 한 번으로 한 스텝을 학습하는 정확한 전치 학습 단계
# 방법: 임베딩에서 블록을 지나 로짓을 내고 교차엔트로피 기울기를 헤드 전치로 되돌린 뒤 블록을 역순으로  전치를 흘려 임베딩까지 갱신
# 이유: 자동미분 그래프 없이 각 연산의 닫힌형 전치만 곱해 backprop 과 수치적으로 같은 기울기를 얻는 것이 이 엔진의 핵심이기 때문
def train_step(m, ids, y, lr):
    T, B = ids.shape
    _Mt = T * B
    _ar = xp.arange(_Mt)
    _cache, _aD, _z = bc.forward(m, ids)
    _g, _ce = bc.p_softmax(_z, y, _ar)
    m.t += 1
    _dtop, _g_data_axis_head = bc.head_delta_top(m, _aD, _g, _Mt, True)
    bc.block_credit(m, _cache, _dtop, lr, ids, T, B, _g_data_axis_head)
    return _ce
