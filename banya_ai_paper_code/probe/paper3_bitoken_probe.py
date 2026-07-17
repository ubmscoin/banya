# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 3 bitoken probe. Measures in four ways that the operate axis emerges as consistent operators without labels in the frozen elementary-stage bitoken model.
It measures how many times the random level the operator direction consistency reaches, what percentage of distinct operations are orthogonal, how far the prediction cross entropy diverges when the operate axis is turned on and off, and whether the causal mixing heads differentiate into distinct past distances through the lag kernels.
Loads the model through the shared foundation banya_core, runs the forward pass, and reuses the Paper 3 bitoken mechanism from the paper3_bitoken module. GPU-only cupy.
Run  python3 paper3_bitoken_probe.py

반야 제3편 바이토큰 프로브. 초등 단계 얼린 바이토큰 모델에서 연산면이 라벨 없이 일관 연산자로 창발함을 네 가지로 실측한다.
연산자 방향 일관성이 무작위의 몇 배인지, 서로 다른 연산의 몇 퍼센트가 직교하는지, 연산면을 켜고 끌 때 예측 교차엔트로피가 얼마나 갈리는지, 인과 혼합 헤드가 거리커널로 서로 다른 과거 거리로 분화되는지를 잰다.
공통 토대 banya_core 로 모델을 불러 순전파하고 제3편 바이토큰 메커니즘은 논문3_바이토큰 에서 가져다 쓴다. GPU 전용 cupy.
실행  python3 paper3_bitoken_probe.py"""
import os
import sys
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper3_bitoken as paper3

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "bitok_elem2_170000_m.npz")
OP_CANDIDATE = 80
DATA_SAMPLE = 30
CE_CORPUS = "banya_world_data/elem_dialog.npy"
CE_WINDOW = 200
CE_SEED = 7
DIVERSITY_BLOCK = 11


# Role: collects the direction vectors along which one operation token bends multiple data tokens
# Method: places the operation token in the front slot and a data token in the rear slot, runs the forward pass with the operate axis on and off, and takes the rear-slot hidden difference as the bend direction
# Why: the operate axis of the immediately preceding token shifts one slot and is injected as the gate of the next token, so the on-off difference is the pure direction in which the operation bent the data
# 역할: 한 연산 토큰이 여러 데이터 토큰을 꺾는 방향 벡터들을 수집
# 방법: 앞자리에 연산 토큰 뒷자리에 데이터 토큰을 놓고 연산면 켬과 끔으로 순전파해 뒷자리 히든의 차이를 꺾음 방향으로 삼는다
# 이유: 직전 토큰의 연산면이 한 칸 이동해 다음 토큰의 게이트로 주입되므로 그 켬끔 차이가 연산이 데이터를 꺾은 순수 방향이기 때문이다
def p_bend_directions(m, op_id, data_ids):
    _batch = len(data_ids)
    _rows = np.stack([np.full(_batch, op_id, dtype=np.int64), np.asarray(data_ids, dtype=np.int64)])
    _X = xp.asarray(_rows)
    paper3.set_operate_axis(True)
    _cache_on, _on, _ = bc.forward(m, _X)
    paper3.set_operate_axis(False)
    _cache_off, _off, _ = bc.forward(m, _X)
    paper3.set_operate_axis(True)
    _v = _on.reshape(bc.HIDDEN_SIZE, 2, _batch)[:, 1, :] - _off.reshape(bc.HIDDEN_SIZE, 2, _batch)[:, 1, :]
    _vh = bc.to_host(_v).astype(np.float64)
    _norms = np.linalg.norm(_vh, axis=0)
    _keep = _norms > 1e-9
    _unit = _vh[:, _keep] / _norms[_keep]
    return _unit.T


# Role: measures direction consistency, how consistently the same operation bends different data
# Method: for each operation, takes the mean pairwise absolute cosine of the per-data bend directions as consistency, and also collects the mean bend direction of each operation as its representative
# Why: acting in a consistent direction on any data means a consistent operator, while bending each data differently means it is not an operation
# 역할: 같은 연산이 여러 데이터를 얼마나 일관된 방향으로 꺾는지 방향 일관성을 잰다
# 방법: 연산마다 데이터별 꺾음 방향의 쌍별 절대 코사인 평균을 일관성으로 삼고 연산별 평균 꺾음 방향도 대표로 수집
# 이유: 어떤 데이터에도 일관 방향으로 작용하면 일관 연산자이고 데이터마다 제각각이면 연산이 아니기 때문이다
def p_direction_consistency(m, op_ids, data_ids):
    _consist = []
    _op_repr = {}
    for o in op_ids:
        _unit = p_bend_directions(m, o, data_ids)
        if _unit.shape[0] < 3:
            continue
        _sim = _unit @ _unit.T
        _n = _unit.shape[0]
        _pair = np.abs(_sim[np.triu_indices(_n, 1)])
        _consist.append(float(_pair.mean()))
        _op_repr[o] = _unit.mean(0)
    return _consist, _op_repr


# Role: measures operation orthogonality, whether distinct operations overlap or stay orthogonal in direction
# Method: normalizes the representative bend direction of each operation, computes pairwise absolute cosines, and reports the near-orthogonal ratio and the same-direction ratio
# Why: each operation holding its own direction means the operation types separated out without labels
# 역할: 서로 다른 연산끼리 방향이 겹치는지 직교하는지 연산 직교성을 측정
# 방법: 연산별 대표 꺾음 방향을 정규화해 쌍별 절대 코사인을 구하고 거의 직교 비율과 같은 방향 비율을 보인다
# 이유: 연산마다 고유한 방향을 가지면 연산 종류가 라벨 없이 갈라져 나온 것이기 때문
def p_orthogonality(op_repr):
    _ops = list(op_repr)
    _R = np.array([op_repr[o] / (np.linalg.norm(op_repr[o]) + 1e-9) for o in _ops])
    _sim = np.abs(_R @ _R.T)
    np.fill_diagonal(_sim, 0.0)
    _pair = _sim[np.triu_indices(len(_ops), 1)]
    _ortho = float((_pair < 0.15).mean()) * 100.0
    _same = float((_pair > 0.5).mean()) * 100.0
    return float(_pair.mean()), _ortho, _same


# Role: measures prediction contribution, how far the next-token prediction cross entropy diverges with the operate axis on versus off
# Method: samples multiple real-corpus windows, runs the forward pass side by side with the operate axis on and off, and reports the mean cross entropy of the correct token for each
# Why: prediction dropping only when the operate axis is on means the axis actually contributes to prediction rather than being decoration
# 역할: 연산면을 켤 때와 끌 때 다음 토큰 예측 교차엔트로피가 얼마나 갈리는지 예측 기여를 측정
# 방법: 실코퍼스 창을 여럿 뽑아 연산면 켬과 끔으로 나란히 순전파해 정답 토큰의 교차엔트로피 평균을 각각 보여준다
# 이유: 연산면을 켜야 예측이 낮아지면 연산면이 장식이 아니라 예측에 실제로 기여함을 뜻하기 때문
def p_ce_on_off(m, corp, window, seed):
    _rng = np.random.RandomState(seed)
    _sum_on = 0.0
    _sum_off = 0.0
    _ar = xp.arange(bc.CONTEXT_LENGTH)
    for _ in range(window):
        i = _rng.randint(0, len(corp) - bc.CONTEXT_LENGTH - 1)
        _seg = np.asarray(corp[i:i + bc.CONTEXT_LENGTH + 1])
        _X = xp.asarray(_seg[:bc.CONTEXT_LENGTH].reshape(-1, 1))
        _Y = xp.asarray(_seg[1:bc.CONTEXT_LENGTH + 1])
        paper3.set_operate_axis(True)
        _cache_on, _aD_on, _z_on = bc.forward(m, _X)
        _g_on, _ce_on = bc.p_softmax(_z_on, _Y, _ar)
        paper3.set_operate_axis(False)
        _cache_off, _aD_off, _z_off = bc.forward(m, _X)
        _g_off, _ce_off = bc.p_softmax(_z_off, _Y, _ar)
        paper3.set_operate_axis(True)
        _sum_on += float(_ce_on)
        _sum_off += float(_ce_off)
    return _sum_on / window, _sum_off / window


# Role: measures the characteristic distance of a lag kernel
# Method: measures the weighted mean of distance using the kernel magnitude at each distance as the weight
# Why: this value summarizes in one number how many steps into the past that head mainly looks
# 역할: 거리커널의 특성 거리를 측정
# 방법: 거리마다 놓인 커널 크기를 무게로 삼아 거리의 가중 평균을 측정
# 이유: 이 값이 그 헤드가 과거 몇 칸을 주로 보는지를 한 숫자로 요약하기 때문
def p_char_lag(lag, kernel):
    _abs = np.abs(kernel)
    _sum = _abs.sum()
    if _sum <= 0:
        return 0.0
    return float((lag * _abs).sum() / _sum)


# Role: measures whether the mixing heads differentiate into distinct past distances through training
# Method: measures the characteristic distance of the shared initial lag kernel, then measures per-head characteristic distances in the block with the largest diversity to derive the differentiation range
# Why: whether order integration relations are divided across the mixing heads without labels is revealed by this differentiation
# 역할: 혼합 헤드들이 학습으로 서로 다른 과거 거리에 분화되는지를 실측한다
# 방법: 학습 시작 공통 거리커널의 특성 거리를 재고 다양성이 가장 큰 블록에서 헤드별 특성 거리를 재 분화 폭을 도출
# 이유: 라벨 없이도 순서 통합 관계가 혼합 헤드에 나뉘어 담기는지가 이 분화로 드러나기 때문
def p_head_distance(m):
    _W = bc.to_host(m.m_mat_w_lag).astype(np.float64)
    _T = _W.shape[-1]
    _lag = np.arange(_T)
    _C = bc.causal_time_mix(_T)
    _wl_init = np.array([_C[np.arange(l, _T), np.arange(_T - l)].mean() for l in range(_T)])
    _L_init = p_char_lag(_lag, _wl_init)
    _cl = np.array([p_char_lag(_lag, _W[DIVERSITY_BLOCK, h]) for h in range(_W.shape[1])])
    _std_per_block = np.array([np.array([p_char_lag(_lag, _W[bl, h]) for h in range(_W.shape[1])]).std() for bl in range(_W.shape[0])])
    return _L_init, _cl, _std_per_block


def main():
    m, tok = bc.load_from(FROZEN)
    _vocab = m.m_vocab_size
    _operate_norm = bc.to_host(xp.linalg.norm(m.m_mat_w_operate_axis, axis=0))
    _data_norm = bc.to_host(xp.linalg.norm(m.m_mat_w_data_axis, axis=0))
    print(f"[제3편 바이토큰] 얼린 모델 {os.path.basename(FROZEN)} · H {bc.HIDDEN_SIZE} · vocab {_vocab} · 연산면 노름 평균 {_operate_norm.mean():.3f} · 데이터면 노름 평균 {_data_norm.mean():.3f}", flush=True)

    _op_med = np.median(_operate_norm[_operate_norm > 1e-6])
    _data_med = np.median(_data_norm[_data_norm > 1e-6])
    _op_ids = [i for i in range(_vocab) if _operate_norm[i] > _op_med][:OP_CANDIDATE]
    _data_pool = [i for i in range(_vocab) if _data_norm[i] > _data_med]
    _rng = np.random.RandomState(0)
    _data_ids = [_data_pool[i] for i in _rng.choice(len(_data_pool), size=min(DATA_SAMPLE, len(_data_pool)), replace=False)]
    print(f"  연산 후보 {len(_op_ids)}개 · 데이터 표본 {len(_data_ids)}개", flush=True)

    _consist, _op_repr = p_direction_consistency(m, _op_ids, _data_ids)
    _mean_consist = float(np.mean(_consist))
    _base = 1.0 / np.sqrt(bc.HIDDEN_SIZE)
    print("\n[5.1 방향 일관성] 같은 연산이 여러 데이터를 같은 방향으로 꺾나 (절대 코사인 평균)", flush=True)
    print(f"   연산자 {_mean_consist:.3f} · 무작위 {_base:.3f} · 배수 {_mean_consist / _base:.1f}배", flush=True)

    _pair_mean, _ortho, _same = p_orthogonality(_op_repr)
    print("\n[5.1 연산 직교성] 서로 다른 연산끼리 방향이 겹치나 직교하나 (연산쌍 절대 코사인)", flush=True)
    print(f"   쌍 평균 {_pair_mean:.3f} · 거의 직교쌍(<0.15) {_ortho:.1f}% · 같은 방향쌍(>0.5) {_same:.1f}%", flush=True)

    _corp = np.load(CE_CORPUS, mmap_mode="r")
    _ce_on, _ce_off = p_ce_on_off(m, _corp, CE_WINDOW, CE_SEED)
    print(f"\n[5.1 예측 기여] 실코퍼스 {CE_WINDOW}창, 연산면 켬 vs 끔 예측 교차엔트로피", flush=True)
    print(f"   끔 {_ce_off:.4f} · 켬 {_ce_on:.4f} · 연산 기여 {_ce_off - _ce_on:+.4f}", flush=True)

    _ratio = _operate_norm / (_data_norm + _operate_norm + 1e-9)
    _edges = [0, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0]
    _bins = np.histogram(_ratio, bins=_edges)[0]
    _mid = _bins[3] / _vocab * 100
    _pole = (_bins[0] + _bins[6]) / _vocab * 100
    _trained = bc.to_host(xp.abs(m.m_mat_w_data_axis_adam_moment).max(axis=0)) > 0
    _n_trained = int(_trained.sum())
    _bins_trained = np.histogram(_ratio[_trained], bins=_edges)[0]
    _mid_trained = _bins_trained[3] / _n_trained * 100
    _pole_trained = (_bins_trained[0] + _bins_trained[6]) / _n_trained * 100
    _n_untrained = _vocab - _n_trained
    _untrained_pole = int((_ratio[~_trained] < 0.15).sum())
    print(f"\n[5.1 토큰 정체 분포] 연산 비율 = 연산면 노름 / (데이터면 노름 + 연산면 노름)", flush=True)
    print(f"   전체 어휘 {_vocab}개: 가운데(0.45~0.55) {_mid:.0f}% · 양끝(0.15 미만 또는 0.85 초과) {_pole:.0f}%", flush=True)
    print(f"   학습 토큰 {_n_trained}개(아담 모멘트가 0 아닌 열): 가운데 {_mid_trained:.0f}% · 양끝 {_pole_trained:.1f}% · 데이터쪽 극 {_bins_trained[0]}개 · 연산쪽 극 {_bins_trained[6]}개", flush=True)
    print(f"   미학습 토큰 {_n_untrained}개 중 데이터쪽 극 {_untrained_pole}개 (초기값 그대로, 연산면 0)", flush=True)
    print(f"   학습 토큰 노름 평균: 데이터면 {_data_norm[_trained].mean():.1f} · 연산면 {_operate_norm[_trained].mean():.1f}", flush=True)

    _ef_host = bc.to_host(m.m_mat_w_data_axis).astype(np.float64)
    _eop_host = bc.to_host(m.m_mat_w_operate_axis).astype(np.float64)
    _tr_ids = np.where(_trained)[0]
    _cos_in = np.abs((_ef_host[:, _tr_ids] * _eop_host[:, _tr_ids]).sum(0)
                     / (np.linalg.norm(_ef_host[:, _tr_ids], axis=0) * np.linalg.norm(_eop_host[:, _tr_ids], axis=0) + 1e-12))
    print(f"\n[5.1 토큰 내부 직교] 같은 토큰의 데이터면과 연산면이 겹치나 (절대 코사인)", flush=True)
    print(f"   학습 토큰 {len(_tr_ids)}개: 평균 {_cos_in.mean():.3f} · 최대 {_cos_in.max():.3f} · 거의 직교(0.15 미만) {(_cos_in < 0.15).mean() * 100:.0f}% (무작위 {_base:.3f})", flush=True)

    _itos_all = np.load(FROZEN, allow_pickle=True)["m_id_to_string"]
    _func_syl = ["은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "와", "과", "로", "서", "게", "다", "요", "까", "니", "고", "지", "면", "며", "자", "라", "야", "죠", "네", "든", "께"]
    _noun_syl = ["물", "불", "꽃", "새", "밥", "집", "손", "발", "귀", "코", "입", "별", "달", "해", "돌", "강", "곰", "소", "닭", "차", "옷", "길", "산", "숲", "비", "책", "밤", "꿈", "몸", "흙"]
    _stem_syl = ["먹", "놀", "씻", "뛰", "걷", "읽", "듣", "굽", "볶", "빼", "넣", "찾", "살", "죽", "크", "작", "맑", "검", "붉", "희", "늙", "씹", "밀", "끌", "업", "솟", "굴", "뻗", "휘", "쥐"]
    _cls = {}
    for _cname, _syls in (("기능", _func_syl), ("명사류", _noun_syl), ("어간", _stem_syl)):
        _ids_c = [i for i in range(_vocab) if str(_itos_all[i]) in _syls and _trained[i]]
        _cons_c, _ = p_direction_consistency(m, _ids_c, _data_ids)
        _cls[_cname] = (_ids_c, float(np.mean(_cons_c)))
    _func_ratio = _ratio[_cls["기능"][0]]
    _noun_ratio = _ratio[_cls["명사류"][0]]
    _win = float(sum((a > b) for a in _func_ratio for b in _noun_ratio)) / (len(_func_ratio) * len(_noun_ratio)) * 100
    print(f"\n[5.1 연산자 정체] 연산자 노릇이 문법 부류의 소유인가 (부류별 비교)", flush=True)
    print(f"   연산 비율 평균: 기능(조사어미) {_func_ratio.mean():.3f} · 명사류 {_noun_ratio.mean():.3f} · 쌍별 우세율 {_win:.1f}%", flush=True)
    print(f"   방향 일관성: 기능 {_cls['기능'][1]:.3f} · 명사류 {_cls['명사류'][1]:.3f} · 용언 어간 {_cls['어간'][1]:.3f} (무작위 {_base:.3f})", flush=True)

    _L_init, _cl, _std_per_block = p_head_distance(m)
    print(f"\n[5.3 혼합 헤드 순서 분화] 거리커널 특성 거리, 블록 {DIVERSITY_BLOCK}", flush=True)
    print(f"   학습 시작 공통 {_L_init:.1f}칸 · 학습 후 {_cl.min():.0f}~{_cl.max():.0f}칸 · 헤드간 표준편차 {_cl.std():.1f}칸", flush=True)
    print(f"   모든 블록 헤드간 표준편차 {_std_per_block.min():.1f}~{_std_per_block.max():.1f}칸", flush=True)

    print("\n[실측 요약]", flush=True)
    print(f"  방향 일관성 {_mean_consist:.3f} 대 무작위 {_base:.3f} = {_mean_consist / _base:.1f}배 (목표 0.316 대 0.031 = 10.1배)", flush=True)
    print(f"  직교쌍(<0.15) {_ortho:.1f}% (목표 72.5%)", flush=True)
    print(f"  ce 끔 {_ce_off:.2f} 대 켬 {_ce_on:.2f} 차이 {_ce_off - _ce_on:.2f} (목표 2.99 대 0.20 차이 2.79)", flush=True)
    print(f"  토큰 정체: 전체 가운데 {_mid:.0f}% 양끝 {_pole:.0f}% (목표 0%와 34%) · 학습 토큰만 양끝 {_pole_trained:.1f}%", flush=True)
    print(f"  토큰 내부 직교: 평균 {_cos_in.mean():.3f} 최대 {_cos_in.max():.3f} (목표 0.027과 0.121)", flush=True)
    print(f"  부류 비교: 기능 일관성 {_cls['기능'][1]:.3f} 대 명사류 {_cls['명사류'][1]:.3f} (목표 0.324 대 0.316, 부류 차이 없음)", flush=True)
    print(f"  헤드 특성거리 시작 {_L_init:.1f}칸 -> 블록{DIVERSITY_BLOCK} {_cl.min():.0f}~{_cl.max():.0f}칸 분화 (목표 12.7칸 -> 39~66칸)", flush=True)


if __name__ == "__main__":
    main()
