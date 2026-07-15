# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 6 metacognition distribution-routing probe. Measures empirically, on a frozen elementary-stage bi-token model, whether the three states certain, vague, and unknown separate on output distribution statistics alone.
It feeds question sets for the three states plus a direct-association set that probes related concepts directly, measures the maximum probability and normalized entropy of the next-word distribution at each question's answer slot, and averages them per set.
The statistics are read straight off the distribution already produced, with no extra training or labels. The model is loaded and forward-passed via the common foundation banya_core, and the Paper 6 distribution statistics and routing are taken from 논문6_메타인지. GPU only (cupy).
Run  python3 paper6_metacog_probe.py

반야 제6편 메타인지 분포 라우팅 프로브. 초등 단계 얼린 바이토큰 모델에서 확실 모호 모름 세 상태가 출력 분포 통계량만으로 갈리는지 실측한다.
세 상태 질문 묶음과 관련 개념을 직접 찌르는 직접 연상 묶음을 넣고 각 질문의 답 자리 다음 낱말 분포에서 최대 확률과 정규화 엔트로피를 재 묶음별 평균을 낸다.
별도 학습이나 라벨 없이 이미 나온 분포에서 바로 읽는다. 공통 토대 banya_core 로 모델을 불러 순전파하고 제6편 분포 통계와 라우팅은 논문6_메타인지 에서 가져다 쓴다. GPU 전용 cupy.
실행  python3 paper6_metacog_probe.py"""
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

확실 = [
    "사용자: 뜨거운 건 뭘로 알아?\n반야:",
    "사용자: 눈으로 뭐 해?\n반야:",
    "사용자: 봄 다음은?\n반야:",
    "사용자: 사과는 과일이야 채소야?\n반야:",
    "사용자: 소리는 뭘로 들어?\n반야:",
]
모호 = [
    "사용자: 무슨 색?\n반야:",
    "사용자: 이거 어때?\n반야:",
    "사용자: 몇 개야?\n반야:",
    "사용자: 맛이 어때?\n반야:",
]
모름 = [
    "사용자: 공룡이 뭐야?\n반야:",
    "사용자: 로봇 어때?\n반야:",
    "사용자: 비행기 어디 가?\n반야:",
    "사용자: 컴퓨터가 뭐야?\n반야:",
]
직접 = [
    "나보다 ",
    "뜨겁다. 앗 ",
    "봄 여름 가을 ",
    "눈으로 ",
]


# Role: measures the state label, maximum probability, and normalized entropy from the answer-slot distribution of one question
# Method: for prompts ending in the Banya marker the first answer token is a formatting space, so a space is appended to measure the content token, and the Paper 6 distribution routing yields the label and statistics together
# Why: whether the three statistics can be read straight off the already-produced distribution, with no training or labels, is the core claim of this paper
# 역할: 한 질문의 답 자리 분포에서 상태 라벨과 최대 확률과 정규화 엔트로피를 잰다
# 방법: 반야로 끝나는 프롬프트는 답 첫 토큰이 형식 공백이라 공백을 붙여 내용 토큰을 재고 제6편 분포 라우팅으로 라벨과 통계를 함께 얻는다
# 이유: 세 통계량이 학습이나 라벨 없이 이미 나온 분포에서 바로 읽히는지가 이 편의 핵심이기 때문
def p_signal(m, tok, prompt):
    if prompt.endswith("반야:"):
        prompt = prompt + " "
    _label, _pmax, _entropy, _peaks = paper6.gate_label(m, tok, prompt)
    return _label, _pmax, _entropy, _peaks


# Role: runs the questions of one state set and averages the maximum probability and entropy for that state
# Method: measures and prints the label and distribution statistics per question, then summarizes the set by mean maximum probability and mean entropy
# Why: if questions of the same state gather within one statistical band, that state separates on the distribution alone
# 역할: 한 상태 묶음의 질문들을 돌려 상태별 최대 확률과 엔트로피 평균을 낸다
# 방법: 질문마다 라벨과 분포 통계를 재 출력하고 묶음 평균을 최대 확률과 엔트로피로 요약한다
# 이유: 같은 상태 질문들이 한 통계 대역에 모이면 그 상태가 분포만으로 갈린다는 뜻이기 때문
def p_group(name, m, tok, prompts):
    _ps = []
    _es = []
    for pr in prompts:
        _label, _pmax, _entropy, _peaks = p_signal(m, tok, pr)
        _ps.append(_pmax)
        _es.append(_entropy)
        _q = pr.replace("\n반야:", "").replace("사용자: ", "").strip()
        print(f"   {_q[:20]:<20} -> {_label} · pmax {_pmax:.2f} · 엔트 {_entropy:.2f} · 봉 {_peaks}", flush=True)
    _pmax_mean = float(np.mean(_ps))
    _ent_mean = float(np.mean(_es))
    print(f"  [{name}] 묶음 평균 pmax {_pmax_mean:.2f} · 엔트 {_ent_mean:.2f}", flush=True)
    return _pmax_mean, _ent_mean


def main():
    m, tok = bc.load_from(FROZEN)
    _base = tok.base
    _step = int(m.t)
    print(f"[제6편 메타인지] 얼린 모델 {os.path.basename(FROZEN)} · vocab {m.m_vocab_size} · H {bc.HIDDEN_SIZE}", flush=True)
    print("최대 확률 pmax 는 top1 확률, 엔트는 정규화 엔트로피(0 몰림 1 확산), 봉은 확률 0.05 초과 낱말 수", flush=True)

    print("\n[확실 아는 것] 최대 확률 높고 엔트로피 낮음", flush=True)
    _certain = p_group("확실", m, _base, 확실)
    print("\n[직접 연상 관련 개념] 형식 없이 학습한 연상을 직접 찌름", flush=True)
    _direct = p_group("직접", m, _base, 직접)
    print("\n[모호 후보 여럿] 봉우리 여럿, 최대 확률 중간", flush=True)
    _vague = p_group("모호", m, _base, 모호)
    print("\n[모름 전혀 모름] 최대 확률 낮고 엔트로피 높음", flush=True)
    _unknown = p_group("모름", m, _base, 모름)

    print("\n[실측 요약] 확실은 몰리고 비확실은 확산한다. 모호와 모름은 서로 겹친다", flush=True)
    print(f"  최대 확률 pmax  확실 {_certain[0]:.2f} · 직접 {_direct[0]:.2f} · 모호 {_vague[0]:.2f} · 모름 {_unknown[0]:.2f}  (목표 1.00 · 0.58 · 0.32 · 0.35)", flush=True)
    print(f"  엔트로피        확실 {_certain[1]:.2f} · 직접 {_direct[1]:.2f} · 모호 {_vague[1]:.2f} · 모름 {_unknown[1]:.2f}  (목표 0.00 · 0.20 · 0.35 · 0.37)", flush=True)


if __name__ == "__main__":
    main()
