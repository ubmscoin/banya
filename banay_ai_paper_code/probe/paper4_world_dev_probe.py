# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya research Paper 4 world-first development probe. Reads the emerged axis structure without disturbing the frozen world model.
The axis measurement reads whether an axis has formed by comparing, against a random baseline, how much the embedding-difference directions of an axis's contrast pairs agree with each other,
and the state map feeds the grade ladder and reads whether that axis is ordered by the rate at which the next grade rises on the next-token distribution.
That ordered axes split by order while unordered categorical axes split without order is direct evidence of axis-structure emergence.

반야 연구 제4편 월드먼저 발달 프로브. 얼린 월드 모델을 건드리지 않고 창발한 축 구조를 읽는다.
축측정은 축의 대비쌍 임베딩 차분 방향이 서로 얼마나 일치하는지를 무작위 기준선과 견줘 축이 형성됐는지 읽고
상태맵은 등급 사다리를 넣어 다음 등급이 다음토큰 분포 위에 뜨는 비율로 그 축이 순서로 정렬됐는지 읽는다.
순서 있는 축은 순서로 순서 없는 범주 축은 순서 없이 갈리는 것이 축 구조 창발의 직접 증거다."""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper4_world_dev as p4

sys.path.insert(0, os.path.join(bc._ROOT, "data_prep"))
import world_ladder as L

MODEL_DIR = os.path.join(_CODE, "model")
DEFAULT_CKPT = os.path.join(MODEL_DIR, "world_toddler2_110000_m.npz")

AXIS_PAIRS = {
    "life": [("태어난다", "죽는다"), ("생긴다", "사라진다"), ("늘어난다", "줄어든다"), ("살았다", "죽었다")],
    "시간축": [("아까", "나중"), ("옛날", "지금"), ("방금", "이따가"), ("어제", "내일")],
    "날씨": [("춥다", "덥다"), ("쌀쌀하다", "무덥다")],
    "습기": [("마르다", "젖다"), ("보송보송", "축축")],
    "탄성": [("흐물흐물", "탱탱"), ("물컹", "통통")],
    "점성": [("매끄럽다", "끈적하다"), ("미끄럽다", "끈끈하다")],
    "둔통": [("개운하다", "뻐근하다"), ("가뿐하다", "저린다")],
    "속느낌": [("편하다", "메스껍다"), ("든든하다", "울렁거린다")],
    "형태": [("둥글다", "네모나다"), ("길쭉하다", "납작하다"), ("뾰족하다", "뭉툭하다")],
    "투명": [("비친다", "막혔다"), ("투명하다", "뿌옇다")],
    "높낮이": [("낮다", "높다"), ("발밑", "꼭대기")],
    "넓이": [("좁다", "넓다"), ("비좁다", "트였다")],
    "존재": [("있다", "없다"), ("있어", "없어"), ("존재", "부재"), ("나타난다", "사라진다")],
    "크기": [("크다", "작다"), ("거대", "왜소"), ("커다란", "조그만"), ("거인", "난쟁이"), ("코끼리", "개미")],
    "거리": [("가깝다", "멀다"), ("코앞", "저편"), ("근처", "먼곳"), ("바로앞", "아득히")],
    "방향": [("앞", "뒤"), ("위", "아래"), ("왼쪽", "오른쪽"), ("전방", "후방"), ("상단", "하단")],
    "시간": [("다가온다", "멀어진다"), ("접근", "이탈"), ("과거", "미래"), ("먼저", "나중")],
    "득실": [("좋다", "싫다"), ("기쁘다", "괴롭다"), ("반갑다", "무섭다"), ("달콤하다", "역겹다")],
    "각성": [("잔잔하다", "격렬하다"), ("고요하다", "요란하다"), ("차분하다", "사납다"), ("평온하다", "세차다")],
    "밝기": [("밝다", "어둡다"), ("환하다", "캄캄하다"), ("눈부시다", "침침하다")],
    "온도": [("뜨겁다", "차갑다"), ("따뜻하다", "서늘하다"), ("덥다", "춥다")],
    "맛": [("달다", "쓰다"), ("달콤하다", "씁쓸하다"), ("단맛", "쓴맛")],
    "단단함": [("단단하다", "무르다"), ("딱딱하다", "물렁하다"), ("굳다", "푹신하다")],
    "시제(참고)": [("었다", "는다"), ("갔다", "간다"), ("먹었", "먹는"), ("왔다", "온다")],
}

STATE_AXES = ["거리", "크기", "다가옴", "밝기", "색", "소리크기", "온도", "질감", "통증", "맛", "냄새", "득실", "각성", "방향"]


def p_word_vec(w_data_axis, stoi, word):
    _cols = [stoi[c] for c in word if c in stoi]
    if len(_cols) != len(word) or not _cols:
        return None
    return w_data_axis[:, _cols].mean(1)


# Role: a non-invasive axis measurement that gauges how many times the random level each sensory axis stands at in the embedding
# Method: for each axis it collects the embedding-difference directions of word pairs that differ only along that axis, computes their mutual cosine agreement, and compares it against a random baseline of arbitrary word pairs
# Why: because if word pairs of the same axis lie in the same direction in the embedding the axis is real, and if it is several times larger than random the axis is in place even without any concept being hand-coded
# 역할: 임베딩에 각 감각 축이 무작위의 몇 배로 섰는지 재는 비침습 축측정
# 방법: 축마다 그 축으로만 다른 낱말쌍의 임베딩 차분 방향을 모아 서로 코사인 일치도를 내고 아무 낱말쌍의 무작위 기준선과 견준다
# 이유: 같은 축의 낱말쌍이 임베딩에서 같은 방향으로 놓이면 그 축이 실재하고 무작위보다 몇 배 크면 개념을 안 짜 넣어도 축에 자리한 것이기 때문
def p_measure_axes(w_data_axis, stoi, seed=0):
    _rng = np.random.RandomState(seed)
    _V = w_data_axis.shape[1]
    _rdirs = []
    for _ in range(80):
        i, j = _rng.randint(0, _V, 2)
        if i == j:
            continue
        _d = w_data_axis[:, i] - w_data_axis[:, j]
        _n = np.linalg.norm(_d)
        if _n > 1e-9:
            _rdirs.append(_d / _n)
    _rcos = []
    for a in range(len(_rdirs)):
        for b in range(a + 1, len(_rdirs)):
            _rcos.append(abs(np.dot(_rdirs[a], _rdirs[b])))
    _base = float(np.mean(_rcos))
    _rows = []
    for ax, pairs in AXIS_PAIRS.items():
        _dirs = []
        for lo, hi in pairs:
            _va = p_word_vec(w_data_axis, stoi, lo)
            _vb = p_word_vec(w_data_axis, stoi, hi)
            if _va is None or _vb is None:
                continue
            _d = _va - _vb
            _n = np.linalg.norm(_d)
            if _n > 1e-9:
                _dirs.append(_d / _n)
        if len(_dirs) < 2:
            _rows.append((ax, None, len(_dirs)))
            continue
        _cos = []
        for a in range(len(_dirs)):
            for b in range(a + 1, len(_dirs)):
                _cos.append(abs(np.dot(_dirs[a], _dirs[b])))
        _rows.append((ax, float(np.mean(_cos)), len(_dirs)))
    return _rows, _base


def p_report_axes(w_data_axis, stoi, step):
    _rows, _base = p_measure_axes(w_data_axis, stoi)
    print(f"\n[축측정] 감각 축 방향 일치도 (무작위 대비 배수) · step {step:,}")
    print(f"무작위 기준선: {_base:.3f}")
    print(f"{'축':<12}{'일치도':>8}{'기준선대비':>10}{'쌍수':>6}  판정")
    _strong = 0
    for ax, val, n in _rows:
        if val is None:
            print(f"{ax:<12}{'--':>8}{'':>10}{n:>6}  단어부족")
            continue
        _ratio = val / _base if _base > 0 else 0
        _verdict = "축 강함" if _ratio > 5 else ("축 있음" if _ratio > 2 else "미형성")
        if _ratio > 5:
            _strong += 1
        print(f"{ax:<12}{val:>8.3f}{_ratio:>9.1f}x{n:>6}  {_verdict}")
    print(f"강한 축(5배 초과) {_strong}개")
    return _base


def p_dist(m, tok, prompt):
    _seq = list(tok.encode(prompt))
    _ctx = _seq[-bc.CONTEXT_LENGTH:]
    _, _, _z = bc.forward(m, bc.xp.asarray(_ctx, dtype=bc.xp.int64).reshape(len(_ctx), 1))
    _lg = bc.to_host(_z[:, -1]).astype(np.float64)
    _e = np.exp(_lg - _lg.max())
    return _e / _e.sum()


def p_axis_type(ax):
    if ax in L.ONESIDE:
        return "한쪽"
    if ax in L.BIDIR:
        return "양방"
    return "범주"


# Role: a non-invasive state-map probe that feeds each axis's grade ladder and gauges whether an order has formed
# Method: after one grade it measures, as order consistency, the rate at which the next grade's first token enters the top3 of the next-token distribution, and also checks whether top1 is that axis's vocabulary
# Why: because for an ordered axis the next grade floats up in the distribution so order consistency is high, while for an unordered categorical axis it is low, and this split is direct evidence of axis-structure emergence
# 역할: 각 축의 등급 사다리를 넣어 순서가 섰는지 재는 비침습 상태맵 프로브
# 방법: 한 등급 뒤에 다음 등급의 첫 토큰이 다음토큰 분포 top3 에 드는 비율을 순서 정합으로 재고 top1 이 그 축 어휘인지도 함께 본다
# 이유: 순서 있는 축은 다음 등급이 분포 위에 떠 순서 정합이 높고 순서 없는 범주 축은 낮아 이 갈림이 축 구조 창발의 직접 증거이기 때문
def p_state_map(m, tok):
    print(f"\n[상태맵] 순서 정합 (다음 등급이 분포 top3 에 든 비율)")
    print(f"{'축':<8}{'타입':<5}{'등급':>4}{'순서정합':>7}{'축내':>6}{'pmax':>6}   복원 예시(첫 등급 다음 top3)")
    print("-" * 78)
    _table = []
    for ax in STATE_AXES:
        _lv = L.축[ax]["levels"]
        _firsts = {}
        for w in _lv:
            _ids = list(tok.encode(w))
            if _ids:
                _firsts.setdefault(int(_ids[0]), w)
        _on_axis_set = set(_firsts)
        _hits = 0
        _on = 0
        _n = 0
        _pmaxs = []
        _example = ""
        for i in range(len(_lv) - 1):
            _p = p_dist(m, tok, _lv[i] + ". ")
            _order = np.argsort(_p)[::-1]
            _top1 = int(_order[0])
            _pmaxs.append(float(_p[_top1]))
            _nxt_ids = list(tok.encode(_lv[i + 1]))
            _nxt = int(_nxt_ids[0]) if _nxt_ids else -1
            _rank = int(np.where(_order == _nxt)[0][0]) if _nxt in _order else 999
            if _rank < 3:
                _hits += 1
            if _top1 in _on_axis_set:
                _on += 1
            _n += 1
            if i == 0:
                _toks = [tok.decode([int(t)]) for t in _order[:3]]
                _example = f"{_lv[0]}. -> " + " ".join(repr(t) for t in _toks)
        _order_score = 100 * _hits / max(1, _n)
        _on_score = 100 * _on / max(1, _n)
        _pm = float(np.mean(_pmaxs)) if _pmaxs else 0.0
        _table.append((ax, _order_score, _on_score, _pm))
        print(f"{ax:<8}{p_axis_type(ax):<5}{len(_lv):>4}{_order_score:>6.0f}%{_on_score:>5.0f}%{_pm:>6.2f}   {_example}")
    print("-" * 78)
    _table.sort(key=lambda r: -r[1])
    _order_axes = [r[0] for r in _table if r[1] >= 50]
    _flat_axes = [r[0] for r in _table if r[1] < 30]
    print(f"순서 잘 선 축(정합 50퍼센트 이상): {' '.join(_order_axes) if _order_axes else '없음'}")
    print(f"아직 흐린 축(정합 30퍼센트 미만): {' '.join(_flat_axes) if _flat_axes else '없음'}")


def main():
    _ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    _m, _tok = p4.load_from(_ckpt)
    _step = int(_m.t)
    print(f"체크포인트: {_ckpt}")
    _w_data_axis = bc.to_host(_m.m_mat_w_data_axis).astype(np.float64)
    _stoi = {s: i for i, s in enumerate(_tok.m_id_to_string)}
    p_report_axes(_w_data_axis, _stoi, _step)
    p_state_map(_m, _tok)


if __name__ == "__main__":
    main()
