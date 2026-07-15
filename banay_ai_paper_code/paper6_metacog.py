# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya study part 6, metacognitive distribution routing. The ability to know whether it knows is extracted without a separate circuit, by having the model read its own output distribution.
We measure whether three states, certain for what is clearly known, ambiguous for what is uncertain, and unknown for what is not known at all, separate using only statistics of the output distribution.
The statistics are three, the maximum probability pmax, the normalized entropy, and the number of peaks: a high maximum probability with low entropy means a concentrated certain state, a low maximum probability with high entropy means a widely spread unknown state, and multiple peaks mean an ambiguous state with many candidates.
This separation is read directly from the already produced distribution without extra training or labels, and can serve as routing for whether to emit an answer as is, refine it, or mark it as unknown.
The forward pass of the shared foundation banya_core is used unchanged, and only this part's distribution statistics and three-state routing are assembled here.

반야 연구 제6편 메타인지 분포 라우팅. 자기가 아는지 모르는지를 아는 능력을 별도 회로 없이 모델이 자기 출력 분포를 읽어 뽑아낸다.
확실 아는 것 모호 애매한 것 모름 전혀 모르는 것 세 상태가 출력 분포의 통계량만으로 갈리는지를 잰다.
통계량은 최대 확률 pmax 와 정규화 엔트로피와 봉우리 수 세 가지이고 최대 확률이 크고 엔트로피가 작으면 한 곳에 몰린 확실 최대 확률이 작고 엔트로피가 크면 넓게 퍼진 모름 봉우리가 여럿이면 후보가 많은 모호이다.
이 갈림은 별도 학습이나 라벨 없이 이미 나온 분포에서 바로 읽히며 어떤 답을 그대로 낼지 다듬을지 모른다고 표시할지의 라우팅에 쓸 수 있다.
공통 토대 banya_core 의 순전파를 그대로 가져다 쓰고 이 편의 분포 통계와 세 상태 라우팅만 여기서 묶는다."""
import os
import sys
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import banya_core as bc

BanyaNoBP = bc.BanyaNoBP
forward = bc.forward
load_from = bc.load_from

GATE_PROBABILITY_MAX_HIGH = 0.55
GATE_PROBABILITY_MAX_LOW = 0.20
GATE_PEAK_THRESHOLD = 0.05
GATE_PEAKS_HIGH = 2


# Role: converts the answer-position logits into a probability distribution and measures the maximum probability, the normalized entropy, and the number of peaks
# Method: turns the logits into probabilities with a softmax and returns the maximum value, the entropy divided by the log of the vocabulary size, and the number of words whose probability exceeds the threshold
# Why: these three statistics alone reveal, without training or labels, whether the distribution is concentrated in one place, spread widely, or has many candidates
# 역할: 답 자리 로짓을 확률 분포로 바꾸고 최대 확률과 정규화 엔트로피와 봉우리 수를 잰다
# 방법: 로짓을 소프트맥스로 확률로 만들고 그중 최대값과 엔트로피를 어휘 로그로 나눈 값과 확률이 문턱을 넘는 낱말 수를 함께 돌려준다
# 이유: 이 세 통계량만으로 분포가 한 곳에 몰렸는지 넓게 퍼졌는지 후보가 여럿인지가 학습이나 라벨 없이 바로 드러나기 때문
def dist_stats(logits):
    _p = np.exp(logits - logits.max())
    _p /= _p.sum()
    _pmax = float(_p.max())
    _entropy = float(-np.sum(_p * np.log(_p + 1e-12)) / np.log(len(_p)))
    _peaks = int((_p > GATE_PEAK_THRESHOLD).sum())
    return _pmax, _entropy, _peaks


# Role: reads the answer-position distribution statistics and selects one of the three state labels certain, ambiguous, and unknown
# Method: truncates the prompt to the context length, runs a forward pass to obtain the answer-position distribution, then assigns certain when the maximum probability is high with few peaks, unknown when the maximum probability is low, and ambiguous in between
# Why: using only the already produced distribution, without training a separate discriminator, the model knows the certainty of its own state and can route whether to emit as is, refine, or mark as unknown
# 역할: 답 자리 분포 통계량을 읽어 확실 모호 모름 세 상태 라벨을 고른다
# 방법: 프롬프트를 문맥 길이만큼 잘라 순전파해 답 자리 분포를 얻고 최대 확률이 높고 봉우리가 적으면 확실 최대 확률이 낮으면 모름 그 사이는 모호로 경로를 배정한다
# 이유: 별도 판별기 학습 없이 이미 나온 분포만으로 자기 상태의 확실성을 알아 그대로 낼지 다듬을지 모른다고 표시할지의 라우팅에 쓸 수 있기 때문
def gate_label(model, tok, prompt):
    _seq = list(tok.encode(prompt))
    _ctx = _seq[-bc.CONTEXT_LENGTH:]
    _, _, _z = bc.forward(model, xp.asarray(_ctx, dtype=xp.int64).reshape(len(_ctx), 1))
    _logits = bc.to_host(_z[:, -1]).astype(np.float64)
    _pmax, _entropy, _peaks = dist_stats(_logits)
    if _pmax >= GATE_PROBABILITY_MAX_HIGH and _peaks <= GATE_PEAKS_HIGH:
        _label = "확실"
    elif _pmax < GATE_PROBABILITY_MAX_LOW:
        _label = "모름"
    else:
        _label = "모호"
    return _label, _pmax, _entropy, _peaks
