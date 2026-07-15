# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya study part 3, bitoken. Each token is split into two orthogonal embeddings, a data axis at the additive slot and an operate axis at the multiplicative gate slot.
The data axis is content added directly to the residual stream, and the operate axis is a transform whose value from the previous token shifts one step and is injected as the channel gate of the next token.
Without labeling which axis is data and which is operation, enforcing only the asymmetry of the additive slot versus the multiplicative slot makes the operate axis emerge as a consistent operator without labels.
Channel mixing forms, through a circulant filterbank, a scale ladder of broad brushes in shallow layers and fine pens in deep layers, and ordering is handled by the distance kernel of the causal mixing head.
The bitoken forward pass and exact-transpose learning of the shared foundation banya_core are used unchanged, and only this part's mechanism is assembled here.

반야 연구 제3편 바이토큰. 토큰을 데이터면 더하기 자리와 연산면 곱하기 게이트 자리 두 직교 임베딩으로 나눈다.
데이터면은 잔차 스트림에 그대로 더해지는 내용이고 연산면은 직전 토큰의 것이 한 칸 이동해 다음 토큰의 채널 게이트로 주입되는 변환이다.
어느 면이 데이터이고 어느 면이 연산인지 라벨하지 않고 더하기 자리 대 곱하기 자리 비대칭만 강제하면 연산면이 라벨 없이 일관 연산자로 창발한다.
채널 혼합은 순환 필터뱅크로 얕은 층 큰 붓 깊은 층 가는 펜의 스케일 사다리를 이루고 순서는 인과 혼합 헤드의 거리커널이 담당한다.
공통 토대 banya_core 의 바이토큰 순전파와 정확한 전치 학습을 그대로 가져다 쓰고 이 편의 메커니즘만 여기서 묶는다."""
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
block_credit = bc.block_credit
toep_build = bc.toep_build
causal_time_mix = bc.causal_time_mix
LEARNING_RATE_ADAM = bc.LEARNING_RATE_ADAM


# Role: an exact-transpose learning step that trains the data axis and the operate axis together with a single bitoken forward pass
# Method: adds the data axis to the residual, shifts the previous token's operate axis one step and injects it into the gate, then sends the cross-entropy gradient back through the head transpose and flows the transpose through the blocks in reverse order to update both axes together
# Why: the operate-axis credit must be extracted at the gate and returned by exact transpose to the token one step earlier for the axis to separate into an operator without labels
# 역할: 바이토큰 순전파 한 번으로 데이터면과 연산면을 함께 학습하는 정확한 전치 학습 단계
# 방법: 데이터면은 잔차에 더하고 연산면은 직전 토큰 것을 한 칸 이동해 게이트에 주입한 뒤 교차엔트로피 기울기를 헤드 전치로 되돌리고 블록을 역순으로 전치를 흘려 두 면을 함께 갱신
# 이유: 연산면 신용이 게이트에서 뽑혀 한 칸 앞 토큰으로 정확한 전치로 되돌아가야 라벨 없이 연산자로 갈라지기 때문
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


# Role: turns the operate-axis gate injection on or off to isolate the predictive contribution of the operate axis
# Method: sets the bitoken switch of the shared foundation so the forward pass chooses whether to inject the previous token's operate axis into the gate
# Why: the same context must be forwarded side by side with the operate axis on and off to measure how much the operate axis aids prediction
# 역할: 연산면 게이트 주입을 켜거나 꺼서 연산면의 예측 기여를 갈라준다
# 방법: 공통 토대의 바이토큰 스위치를 세워 순전파가 게이트에 직전 토큰의 연산면을 주입할지 말지 고른다
# 이유: 같은 문맥을 연산면 켬과 끔으로 나란히 순전파해야 연산면이 예측을 얼마나 돕는지 잴 수 있기 때문
def set_operate_axis(on):
    bc.USE_BITOKEN = bool(on)
