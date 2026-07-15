# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya research Paper 4 world-first development. Rather than hand-coding an ontology, it aligns the learning order with a human developmental order.
When trained with a developmental curriculum that first erects world axes such as existence, size, distance, and time before language, and lays language on top,
the filters and embeddings form an ontological axis structure on their own. Training uses Paper 1's exact-transpose engine as is,
and readout is done with a non-invasive probe that does not stop learning.
This paper's model is a pure world model without bitokens, so it turns off the operation plane and uses the atom dictionary and a full head.
It borrows the forward pass and exact-transpose learning of the common foundation banya_core as is, and bundles only this paper's developmental curriculum here.

반야 연구 제4편 월드먼저 발달. 온톨로지를 직접 짜 넣지 않고 학습 순서를 사람의 발달 순서에 맞춘다.
언어 이전에 존재 크기 거리 시간 같은 월드 축을 먼저 세우고 그 위에 언어를 얹는 발달 커리큘럼으로 학습하면
필터와 임베딩이 스스로 온톨로지적 축 구조를 형성한다. 학습은 제1편의 정확한 전치 엔진을 그대로 쓰고
판독은 학습을 멈추지 않는 비침습 프로브로 한다.
이 편의 모델은 바이토큰 없는 순수 월드 모델이라 연산면을 끄고 원자 사전과 완전 헤드를 쓴다.
공통 토대 banya_core 의 순전파와 정확한 전치 학습을 그대로 가져다 쓰고 이 편의 발달 커리큘럼만 여기서 묶는다."""
import os
import sys
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import banya_core as bc
import banya_atoms as ba

bc.USE_BITOKEN = False

BanyaNoBP = bc.BanyaNoBP
forward = bc.forward
p_softmax = bc.p_softmax
head_delta_top = bc.head_delta_top
block_credit = bc.block_credit
to_host = bc.to_host
LEARNING_RATE_ADAM = bc.LEARNING_RATE_ADAM

WARMUP_STEPS = 10000
WARMUP_SPLIT = 5000
WARMUP_MIX = bc.WARMUP_MIX
MIX_WEIGHTS = bc.MIX_WEIGHTS
MIX_SCHED = bc.MIX_SCHED
STAGE_NAME = {10000: "탄생", 30000: "아기", 70000: "유아", 110000: "유아2"}


class WorldTokenizer(ba.AtomTokenizer):
    def __init__(self):
        super().__init__()
        self.m_vocab_size = self.vocab
        self.m_id_to_string = list(self.itos)


# Role: reads a world-stage checkpoint and returns the pure world model and the atom tokenizer
# Method: reuses the common foundation's loader as is, but since this paper is a full-head model with no operation plane, it slots the atom tokenizer into the cache-tokenizer position
# Why: because a world checkpoint predates bitokens, its vocab is the atom dictionary and it has no operation-plane embedding, so it must be read with that dictionary for the logits and reconstruction to match
# 역할: 월드 단계 체크포인트를 읽어 순수 월드 모델과 원자 토크나이저를 돌려준다
# 방법: 공통 토대의 로더를 그대로 쓰되 이 편은 연산면 없는 완전 헤드 모델이라 캐시 토크나이저 자리에 원자 토크나이저를 끼운다
# 이유: 월드 체크포인트는 바이토큰 이전 구조라 vocab 이 원자 사전이고 연산면 임베딩이 없어 그 사전으로 읽어야 로짓과 복원이 맞기 때문
def load_from(path):
    _orig = bc.CacheTokenizer
    bc.CacheTokenizer = WorldTokenizer
    try:
        return bc.load_from(path)
    finally:
        bc.CacheTokenizer = _orig


# Role: an exact-transpose training step that trains the world model for one step with a single forward pass
# Method: goes from the embedding through the blocks to produce logits, sends the cross-entropy gradient back through the head transpose, then flows the transpose through the blocks in reverse to update all the way to the embedding
# Why: because the developmental curriculum only fixes the data order, while the training itself is carried out by Paper 1's exact-transpose engine without autodiff
# 역할: 순전파 한 번으로 월드 모델 한 스텝을 학습하는 정확한 전치 학습 단계
# 방법: 임베딩에서 블록을 지나 로짓을 내고 교차엔트로피 기울기를 헤드 전치로 되돌린 뒤 블록을 역순으로 전치를 흘려 임베딩까지 갱신
# 이유: 발달 커리큘럼이 데이터 순서를 정할 뿐 학습 자체는 제1편의 자동미분 없는 정확한 전치 엔진으로 이뤄지기 때문
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
