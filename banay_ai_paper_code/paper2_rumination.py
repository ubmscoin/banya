# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya research paper 2, rumination. Self-organizing learning that makes concept embeddings cluster on their own by pulling neighbors toward each other, with no gradients, no loss, and no external data.
It pulls the same-kind neighbors of seed concepts sampled in proportion to frequency toward their centroid, and this balances against a per-step decay that nudges everything slightly back toward the original, settling into equilibrium without target values.
The model and forward pass of the shared foundation banya_core are used as they are, and only this paper's rumination rule is composed here.

반야 연구 제2편 되새김. 기울기도 손실도 바깥 데이터도 없이 개념 임베딩을 이웃끼리 당겨 스스로 뭉치게 하는 자기조직화 학습.
빈도 비례로 뽑은 씨앗 개념의 동종 이웃을 무게중심으로 당기고 매 스텝 원본으로 조금 되돌리는 감쇠와 균형을 이뤄 목표값 없이 평형에 안착한다.
공통 토대 banya_core 의 모델과 순전파를 그대로 가져다 쓰고 이 편의 되새김 규칙만 여기서 구성한다."""
import os
import sys
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import banya_core as bc

DELTA_CHILD_NUMBER = 3
VERB_ENDING = ("어", "아", "지", "고", "게", "서", "며", "면", "나", "자", "니", "데", "도", "야", "까", "래", "걸", "든", "해", "와", "려", "러", "줘", "봐", "죠", "대")
FIRING_FLOOR = 0.14
FIRING_RETAIN = 0.8
FIRING_GROUP_MAX = 60
FIRING_BASE_PATH = os.path.join(_HERE, "model", "cache_elem3_190000.npz")


# Role: normalizes the data-plane bundle embeddings for cosine use and caches them
# Method: divides the data-plane embedding by per-column norms, stores the whole span and the bundle span separately, and builds them only once
# Why: associative neighbors are found by cosine, and renormalizing at every utterance would be wasteful, so the frozen geometry is computed once
# 역할: 데이터면 묶음 임베딩을 코사인용으로 정규화해 캐시한다
# 방법: 데이터면 임베딩을 열별 노름으로 나눠 전체와 묶음 구간으로 나눠 담고 처음 한 번만 만든다
# 이유: 연상 이웃을 코사인으로 찾는데 매 발화마다 정규화하면 낭비라 얼린 기하로 한 번만 계산한다
def delta_vec_data_norm(m, base):
    if getattr(m, "m_delta_norm_all", None) is None:
        _w_data_axis_norm = m.m_mat_w_data_axis / (xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9)
        m.m_delta_norm_all = _w_data_axis_norm
        m.m_delta_norm_bundle = _w_data_axis_norm[:, base:]
    return m.m_delta_norm_all, m.m_delta_norm_bundle


# Role: picks utterance children by embedding-neighbor association from the concept at the end of the path
# Method: measures the cosine between the end concept and every bundle, keeps bundles only in descending order, excludes concepts already produced, and selects the top few
# Why: next-token prediction continues grammar, but embedding neighbors continue concepts close in meaning, and this association is the actual utterance of rumination
# 역할: 발화 자식을 경로 끝 개념의 임베딩 이웃 연상으로 뽑는다
# 방법: 끝 개념과 모든 묶음의 코사인을 재고 큰 순서로 묶음만 골라 이미 나온 개념을 빼고 상위 몇 개를 골라 낸다
# 이유: 다음토큰은 문법을 잇지만 임베딩 이웃은 뜻이 가까운 개념을 잇는다 이 연상이 되새김의 실제 발화다
def delta_firing(m, tok, path_ids, seen):
    base = tok.m_base_vocab
    _vec_norm_all, _vec_norm_bundle = delta_vec_data_norm(m, base)
    _node = int(path_ids[-1])
    _sims = bc.to_host(_vec_norm_bundle.T @ _vec_norm_all[:, _node]).astype(np.float64)
    _order = np.argsort(-_sims)
    _kids = []
    _ksim = []
    for j in _order:
        _wid = base + int(j)
        if _wid == _node or _wid in seen:
            continue
        _kids.append(_wid)
        _ksim.append(float(_sims[j]))
        if len(_kids) >= DELTA_CHILD_NUMBER:
            break
    return _kids, _ksim


# Role: sets utterance priority as similarity to the seed multiplied by a frequency weight
# Method: multiplies the cosine between the node and the seed by a weight from the frequency map, pushing grammar fragments far from the seed to the back
# Why: a convergence-point constraint that confines utterances to the seed's topical orbit and prevents drifting off topic
# 역할: 발화 우선순위를 씨앗과의 유사도에 빈도가중을 곱해 정한다
# 방법: 노드와 씨앗의 코사인에 빈도맵 값의 가중을 곱해 씨앗에서 먼 문법 조각을 뒤로 보낸다
# 이유: 씨앗 주제 궤도 안에 발화를 가둬 주제를 벗어나 표류하는 것을 막는 수렴점 제약이다
def delta_priority(m, tok, node, seed_node):
    base = tok.m_base_vocab
    _vec_norm_all, _ = delta_vec_data_norm(m, base)
    _sim_seed = float(bc.to_host(_vec_norm_all[:, int(node)] @ _vec_norm_all[:, int(seed_node)]))
    _freq = delta_frequency_map(m, tok)[int(node)]
    return _sim_seed * (0.3 + 0.7 * _freq)


# Role: builds a per-token frequency weight between 0 and 1 and caches it
# Method: normalizes bundle frequencies with a log squash, leaves syllables at a neutral value, and uses this as the frequency term of the priority
# Why: it is the basis of frequency-proportional rumination, where more frequently used concepts fire more often and cluster more tightly
# 역할: 토큰별 빈도가중을 0에서 1 사이로 만들어 캐시한다
# 방법: 묶음은 빈도를 로그로 눌러 정규화하고 음절은 중립값으로 두어 우선순위의 빈도 항으로 쓴다
# 이유: 자주 쓰는 개념일수록 자주 발화돼 더 촘촘히 뭉치는 빈도 비례 되새김의 근거가 되기 때문
def delta_frequency_map(m, tok):
    if getattr(m, "m_delta_frequency", None) is None:
        base = tok.m_base_vocab
        _freq = np.zeros(tok.m_vocab_size, dtype=np.float64)
        for b in tok.m_bundles:
            _freq[int(b["id"])] = float(b.get("freq", 0))
        _freq_norm = np.log1p(_freq) / np.log1p(max(_freq.max(), 1.0))
        _freq_norm[:base] = 0.5
        m.m_delta_frequency = _freq_norm
    return m.m_delta_frequency


# Role: passes only content-concept noun-like tokens as utterance seeds
# Method: decodes the token into characters and returns true only for tokens at least two characters long that do not end with a verb or adjective ending
# Why: concepts such as bear, school, or mom should be ruminated, while inflected forms such as did or made are grammar and must be excluded
# 역할: 내용 개념 명사류만 발화 씨앗으로 통과
# 방법: 토큰을 글자로 풀어 두 글자 이상이고 동사 형용사 어미로 끝나지 않는 것만 참으로
# 이유: 곰 학교 엄마 같은 개념은 되새기고 했어 만들었어 같은 활용형은 문법이라 빼야 하기 때문이다
def firing_concept(tok, wid):
    _word = tok.decode([int(wid)])
    return len(_word) >= 2 and not _word.endswith(VERB_ENDING)


# Role: grows out from one seed with budget decay and collects a group of same-kind concepts close in meaning
# Method: expands neighbors from the highest-priority path first, multiplying the budget by a decay at each depth, and stops a branch when the product of budget and similarity falls below the threshold
# Why: the budget shrinks geometrically with depth, so concepts far from the seed are naturally cut off and the group forms only within the topic
# 역할: 한 씨앗에서 뜻이 가까운 동종 개념 무리를 예산 감쇠로 뻗어 수집
# 방법: 우선순위 큰 경로부터 이웃을 뽑되 깊이마다 예산에 감쇠를 곱해 예산과 유사도의 곱이 문턱 아래면 그 가지를 멈춘다
# 이유: 깊이가 깊을수록 예산이 기하로 줄어 씨앗에서 먼 개념은 자연히 잘려 주제 안에서만 무리를 구성
def firing_group(m, tok, seed_id):
    _seen = {int(seed_id)}
    _seed_node = int(seed_id)
    _frontier = [[delta_priority(m, tok, _seed_node, _seed_node), [int(seed_id)], 0]]
    _out = [int(seed_id)]
    while _frontier and len(_out) < FIRING_GROUP_MAX:
        _frontier.sort(key=lambda x: -x[0])
        _pri, _path, _depth = _frontier.pop(0)
        _kids, _ksim = delta_firing(m, tok, _path, _seen)
        _budget = FIRING_RETAIN ** _depth
        if not _kids or (_ksim and _ksim[0] * _budget < FIRING_FLOOR):
            continue
        for kk in _kids:
            _seen.add(kk)
            _out.append(kk)
            _frontier.append([delta_priority(m, tok, kk, _seed_node), _path + [kk], _depth + 1])
    return _out


# Role: a self-organizing loop that runs rumination without target values so concept embeddings cluster on their own
# Method: samples seeds in proportion to frequency, pulls each same-kind group toward its centroid, restores the original norm, and decays the whole concept layer slightly toward the original every step
# Why: pull alone would collapse everything to a point, so the decay toward the original acts as an opposing restoring force and the two forces settle into equilibrium on their own, while syllables, the head, and the operation plane are left untouched so grammar is preserved
# 역할: 목표값 없이 되새김을 돌려 개념 임베딩을 스스로 뭉치게 하는 자기조직화 루프
# 방법: 빈도 비례로 씨앗을 뽑아 동종군을 무게중심으로 당겨 뭉치고 원노름으로 되돌리며 매 스텝 개념층 전체를 원본으로 약간 감쇠
# 이유: 당기는 힘만 있으면 한 점으로 붕괴하므로 원본 감쇠가 방향 반력이 되어 두 힘이 비기는 평형에 스스로 안착하고 음절과 헤드와 연산면은 건드리지 않아 문법이 보존
def standing_firing(m, tok):
    base = tok.m_base_vocab
    OUT = "model/발화.npz"
    NSEED = 16
    ALPHA = 0.06
    DECAY = 0.012
    REPORT = 500
    SAVE_EVERY = 1000
    np.random.seed(1)
    _freq_norm = delta_frequency_map(m, tok)
    _concepts = np.array([j for j in range(base, tok.m_vocab_size) if firing_concept(tok, j) and _freq_norm[j] > 0], dtype=np.int64)
    _w = _freq_norm[_concepts].astype(np.float64)
    _w = _w / _w.sum()
    _conc_gpu = xp.asarray(_concepts, dtype=xp.int64)
    _norm0 = xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True)
    _w_data_axis0 = m.m_mat_w_data_axis.copy()
    _n0_concept = float(bc.to_host(xp.linalg.norm(m.m_mat_w_data_axis[:, _conc_gpu], axis=0)).mean())
    _top = _concepts[np.argsort(-_w)[:5]]
    _probe_groups = []
    for tk in _top:
        _g = [int(x) for x in firing_group(m, tok, int(tk))][:10]
        if len(_g) >= 3:
            _probe_groups.append(_g)

    def coh():
        _norm = bc.to_host(m.m_mat_w_data_axis / (xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9))
        _ws = []
        for g in _probe_groups:
            _V = _norm[:, g]
            _sim = _V.T @ _V
            _n = len(g)
            _ws.append((_sim.sum() - _n) / (_n * (_n - 1)))
        return round(float(np.mean(_ws)), 3) if _ws else 0.0

    print(f"[발화] 빈도비례 발화 · 당김 {ALPHA} · 감쇠 {DECAY} · 스텝당발화 {NSEED} · 개념 {len(_concepts)}개 · 저장 {OUT}", flush=True)
    print("  목표값 없음. 평형은 발화 연결성이 정함. Ctrl-C 로 멈추면 저장. 켤수록 동종밀집 촘촘", flush=True)
    step = 0
    try:
        while True:
            m.m_mat_w_data_axis[:, _conc_gpu] = m.m_mat_w_data_axis[:, _conc_gpu] * (1.0 - DECAY) + _w_data_axis0[:, _conc_gpu] * DECAY
            for _ in range(NSEED):
                _s = int(np.random.choice(_concepts, p=_w))
                _g = firing_group(m, tok, _s)
                if len(_g) < 2:
                    continue
                _group_gpu = xp.asarray(sorted(set(int(x) for x in _g)), dtype=xp.int64)
                _w_data_axis_group = m.m_mat_w_data_axis[:, _group_gpu]
                _centroid = _w_data_axis_group.mean(1, keepdims=True)
                _pulled = _w_data_axis_group + ALPHA * (_centroid - _w_data_axis_group)
                m.m_mat_w_data_axis[:, _group_gpu] = _pulled * (_norm0[:, _group_gpu] / (xp.linalg.norm(_pulled, axis=0, keepdims=True) + 1e-9))
            if (step + 1) % REPORT == 0:
                _conc_norm = float(bc.to_host(xp.linalg.norm(m.m_mat_w_data_axis[:, _conc_gpu], axis=0)).mean()) / _n0_concept
                print(f"  step {step+1}: 동종밀집 {coh()} 개념층노름 {_conc_norm:.2f}", flush=True)
            if (step + 1) % SAVE_EVERY == 0:
                bc.save_to(m, tok, OUT)
                print(f"    [저장] {OUT} step {step+1}", flush=True)
            step += 1
    except KeyboardInterrupt:
        pass
    bc.save_to(m, tok, OUT)
    print(f"\n[저장] {OUT} step {step}", flush=True)


if __name__ == "__main__":
    os.chdir(bc._ROOT)
    _m, _tok = bc.load_from(FIRING_BASE_PATH)
    standing_firing(_m, _tok)
