# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya study part 5, cache dictionary tokenization and dynamic embedding. Tokenization is arranged in two layers.
The lower layer consists of syllable atoms and the upper layer is a chunk dictionary holding frequent words whole, so the vocabulary grows to 10724 in total by stacking 8000 chunks on top of 2724 syllables.
Growing the vocabulary makes the dense output head expensive since it scales with the product of vocabulary and channels, but this part keeps the output head as a low-rank product of two small matrices, making vocabulary expansion nearly free.
Furthermore, it confirms by measurement that chunk-layer tokens also preserve the orthogonal structure of the data axis and the operate axis established in part 3.
The cache tokenizer, the low-rank head forward pass, and the exact-transpose learning of the shared foundation banya_core are used unchanged, and only this part's low-rank head cost formula is assembled here.

반야 연구 제5편 캐시 사전 토큰화와 동적 임베딩. 토큰화를 두 층으로 둔다.
아래층은 음절 원자이고 위층은 자주 쓰는 낱말을 통째로 담는 묶음 사전이라 어휘를 음절 2724 위에 묶음 8000 을 얹어 총 10724 로 키운다.
어휘를 키우면 밀집 출력 헤드가 어휘와 채널의 곱만큼 커져 비싸지지만 이 편은 출력 헤드를 두 작은 행렬의 곱인 로우랭크로 두어 어휘 확장을 거의 무비용으로 만든다.
나아가 묶음층 토큰도 제3편이 확립한 데이터면과 연산면의 직교 구조를 그대로 유지하는지 측정으로 확인한다.
공통 토대 banya_core 의 캐시 토크나이저와 로우랭크 헤드 순전파와 정확한 전치 학습을 그대로 가져다 쓰고 이 편의 로우랭크 헤드 비용식만 여기서 묶는다."""
import os
import sys
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import banya_core as bc

CacheTokenizer = bc.CacheTokenizer
BanyaNoBP = bc.BanyaNoBP
forward = bc.forward
p_softmax = bc.p_softmax
head_delta_top = bc.head_delta_top
block_credit = bc.block_credit
load_from = bc.load_from
LEARNING_RATE_ADAM = bc.LEARNING_RATE_ADAM
HEAD_RANK = bc.HEAD_RANK


# Role: computes directly from the structure how many times fewer parameters the low-rank head uses than the dense head
# Method: the dense cost is the product of vocabulary and channels, the low-rank cost is the sum of vocabulary and channels each multiplied only by the rank, and both values and their ratio are returned
# Why: with a fixed rank the cost grows only linearly in the vocabulary even as it expands, and this ratio shows that vocabulary expansion is nearly free
# 역할: 로우랭크 헤드가 밀집 헤드보다 파라미터를 몇 배 적게 쓰는지 구조에서 곧바로 계산한다
# 방법: 밀집은 어휘와 채널의 곱이고 로우랭크는 어휘와 채널을 각각 랭크로만 곱한 합이라 그 두 값과 비를 돌려준다
# 이유: 어휘를 키워도 랭크가 고정이면 비용이 어휘에 선형으로만 늘어 어휘 확장이 거의 무비용임을 이 비가 보이기 때문
def head_param_cost(vocab, hidden, rank):
    _dense = vocab * hidden
    _lowrank = (vocab + hidden) * rank
    return _dense, _lowrank, _dense / _lowrank


# Role: an exact-transpose learning step that trains the data axis, the operate axis, and the low-rank head together with a single cache dictionary forward pass
# Method: adds the data axis to the residual, shifts the previous token's operate axis one step and injects it into the gate, then produces logits through the low-rank head, returns the error through its transpose, and flows it through the blocks in reverse order to update both axes and the head together
# Why: even with the vocabulary enlarged by chunks, learning remains the exact-transpose engine of part 1, and only the head is low-rank, so its transpose updates the two small matrices together
# 역할: 캐시 사전 순전파 한 번으로 데이터면과 연산면과 로우랭크 헤드를 함께 학습하는 정확한 전치 학습 단계
# 방법: 데이터면은 잔차에 더하고 연산면은 직전 토큰 것을 한 칸 이동해 게이트에 주입한 뒤 로우랭크 헤드로 로짓을 내고 그 전치로 오차를 되돌려 블록을 역순으로 흘려 두 면과 헤드를 함께 갱신
# 이유: 어휘를 묶음으로 키워도 학습은 제1편의 정확한 전치 엔진 그대로이고 헤드만 로우랭크라 그 전치가 두 작은 행렬을 함께 갱신하기 때문
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


# Role: turns the operate-axis gate injection on or off to isolate the pure contribution of the chunk-layer and syllable-layer operate axes
# Method: sets the bitoken switch of the shared foundation so the forward pass chooses whether to inject the previous token's operate axis into the gate
# Why: the same context must be forwarded side by side with the operate axis on and off to measure the pure direction in which the chunk-layer operate axis bends the data
# 역할: 연산면 게이트 주입을 켜거나 꺼서 묶음층과 음절층 연산면의 순수 기여를 갈라준다
# 방법: 공통 토대의 바이토큰 스위치를 세워 순전파가 게이트에 직전 토큰의 연산면을 주입할지 말지 고른다
# 이유: 같은 문맥을 연산면 켬과 끔으로 나란히 순전파해야 묶음층 연산면이 데이터를 꺾는 순수 방향을 잴 수 있기 때문
def set_operate_axis(on):
    bc.USE_BITOKEN = bool(on)
