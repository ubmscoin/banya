# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya Paper 2 rumination QA probe. Measures with two experiments whether rumination preserves downstream two-candidate discrimination.
Discrimination is 2AFC (two-alternative forced choice): for each of 16 elementary knowledge questions a correct and a wrong word are placed at the answer slot, an item counts as correct when the answer-slot log-probability is higher on the correct side, and chance level is 50 percent.
Experiment 1 applies the published 5.1 same-kind rumination schedule as is and measures before and after. Since this schedule barely moves the candidate words, whether the tokens actually moved is disclosed per item as well.
Experiment 2 is a stress run that forcibly injects the candidate words themselves as seeds. It directly measures whether discrimination still holds even after semantic-neighbor candidates such as hot and cold are actually made to cluster.
Accuracy carries a Wilson 95 percent confidence interval, and before-after changes are reported as directional flip counts. A prior-preference baseline with the questions removed is listed alongside.
The model is loaded and forward-passed via the common foundation banya_core, and the rumination rules and collection functions are taken from the existing Paper 2 probe. GPU only (cupy).
Run  python3 paper2_rumination_qa_probe.py

반야 제2편 되새김 질답 프로브. 되새김이 하류 두 후보 판별을 지키는지 두 실험으로 실측한다.
판별은 초등 지식 질문 16개에 정답과 오답 낱말을 놓고 답 자리 로그확률이 정답 쪽에서 높으면 정답으로 세는 2AFC(두 후보 강제선택)이고 우연 수준은 50퍼센트다.
실험1은 발표된 5.1 동종 되새김 스케줄을 그대로 걸어 전후를 잰다. 다만 이 스케줄은 후보 낱말을 거의 안 움직이므로 문항별로 토큰이 실제로 움직였는지 같이 공개한다.
실험2는 후보 낱말 자체를 씨앗으로 강제 투입하는 스트레스 판이다. 뜨거워와 차가워 같은 의미 이웃 후보가 실제로 뭉치도록 만든 뒤에도 판별이 유지되는지를 직접 잰다.
정확도에는 윌슨 95퍼센트 신뢰구간을 달고 전후는 방향별 뒤집힘 수로 보고한다. 질문을 지운 사전선호 기준선도 병기한다.
공통 토대 banya_core 로 모델을 불러 순전파하고 되새김 규칙과 수집 함수는 기존 제2편 프로브에서 가져다 쓴다. GPU 전용 cupy.
실행  python3 paper2_rumination_qa_probe.py"""
import os
import sys
import json
import numpy as np
import cupy as xp

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _CODE)
import banya_core as bc
import paper2_rumination as paper2
import paper2_rumination_probe as probe2

os.chdir(bc._ROOT)
FROZEN = os.path.join(_CODE, "model", "cache_elem3_190000.npz")
ALPHA = 0.05
NSEED = 16
STEPS = 800
CKPT = 200
NOQ_PROMPT = "반야: "
QA = [
    ("사용자: 사과는 과일이야 채소야?\n반야: ", "과일", "채소"),
    ("사용자: 불은 뜨거워 차가워?\n반야: ", "뜨거워", "차가워"),
    ("사용자: 얼음은 차가워 뜨거워?\n반야: ", "차가워", "뜨거워"),
    ("사용자: 소리는 뭘로 들어?\n반야: ", "귀로", "눈으로"),
    ("사용자: 냄새는 뭘로 맡아?\n반야: ", "코로", "귀로"),
    ("사용자: 밥은 뭘로 먹어?\n반야: ", "입으로", "코로"),
    ("사용자: 봄 다음은 무슨 계절이야?\n반야: ", "여름", "겨울"),
    ("사용자: 강아지는 어떻게 울어?\n반야: ", "멍멍", "야옹"),
    ("사용자: 고양이는 어떻게 울어?\n반야: ", "야옹", "멍멍"),
    ("사용자: 새는 뭘로 날아?\n반야: ", "날개로", "다리로"),
    ("사용자: 물고기는 어디서 살아?\n반야: ", "물에서", "하늘에서"),
    ("사용자: 비 오면 뭘 써?\n반야: ", "우산", "풍선"),
    ("사용자: 아침에 하늘에 뜨는 건 뭐야?\n반야: ", "해", "달"),
    ("사용자: 밤하늘에 뜨는 건 뭐야?\n반야: ", "달", "해"),
    ("사용자: 다리로 뭐 해?\n반야: ", "걸어", "먹어"),
    ("사용자: 겨울에 오는 하얀 건 뭐야?\n반야: ", "눈", "비"),
]


# Role: computes the Wilson 95 percent confidence interval of a proportion
# Method: applies the Wilson formula directly, which is safer than the normal approximation for small samples
# Why: with only 16 items a point estimate alone would be overconfident, so the interval must be shown alongside to be honest
# 역할: 비율의 윌슨 95퍼센트 신뢰구간을 계산한다
# 방법: 정규 근사보다 작은 표본에서 안전한 윌슨 공식을 그대로 쓴다
# 이유: 문항이 16개뿐이라 점추정만 적으면 과신이고 구간을 같이 보여야 정직하기 때문
def p_wilson(k, n):
    _z = 1.96
    _p = k / n
    _d = 1 + _z * _z / n
    _c = (_p + _z * _z / (2 * n)) / _d
    _h = _z * ((_p * (1 - _p) / n + _z * _z / (4 * n * n)) ** 0.5) / _d
    return (_c - _h) * 100, (_c + _h) * 100


# Role: measures the summed log-probability of the answer string appended to a prompt
# Method: encodes the prompt and the full sentence separately, finds their common prefix, and sums the log-probabilities at the answer token positions after it
# Why: merged encoding can split differently at the boundary, so counting from after the common prefix in the id stream is the safe approach
# 역할: 프롬프트 뒤에 붙은 답 문자열의 로그확률 합을 잰다
# 방법: 프롬프트와 전체 문장을 따로 인코딩해 공통 접두를 찾고 그 뒤 답 토큰 자리들의 로그확률을 더한다
# 이유: 묶음 인코딩이 경계에서 갈라질 수 있어 아이디 열 기준 공통 접두 뒤부터 세야 안전하기 때문
def p_answer_logp(m, tok, prompt, answer):
    _idp = tok.encode(prompt)
    _ids = tok.encode(prompt + answer)
    _cp = 0
    while _cp < min(len(_idp), len(_ids)) and _idp[_cp] == _ids[_cp]:
        _cp += 1
    _X = xp.asarray(_ids, dtype=xp.int64).reshape(len(_ids), 1)
    _, _, _z = bc.forward(m, _X)
    _zz = bc.to_host(_z).astype(np.float64)
    _logp = 0.0
    for t in range(max(_cp - 1, 0), len(_ids) - 1):
        _col = _zz[:, t] - _zz[:, t].max()
        _p = np.exp(_col)
        _p /= _p.sum()
        _logp += float(np.log(_p[_ids[t + 1]] + 1e-300))
    return _logp


# Role: measures per-item correctness, the correct-wrong margin, and the correct-answer log-probability over the QA set
# Method: for each question measures the answer-slot log-probabilities of the correct and wrong words and records their difference as the margin
# Why: accuracy alone hides margin swings of whole nats as long as the sign does not cross, so the margin distribution must be inspected together
# 역할: 질답 묶음의 문항별 정오와 정오 격차와 정답 로그확률을 잰다
# 방법: 질문마다 정답과 오답의 답 자리 로그확률을 재 차이를 정오 격차로 담는다
# 이유: 정확도만 보면 정오 격차가 나트 단위로 출렁여도 부호만 안 넘으면 안 보이므로 정오 격차 분포를 같이 봐야 하기 때문
def p_qa_eval(m, tok, use_question=True):
    _oks = []
    _margins = []
    _lps = []
    for prompt, ans, wrong in QA:
        _pr = prompt if use_question else NOQ_PROMPT
        _lp_ok = p_answer_logp(m, tok, _pr, ans)
        _lp_no = p_answer_logp(m, tok, _pr, wrong)
        _oks.append(_lp_ok > _lp_no)
        _margins.append(_lp_ok - _lp_no)
        _lps.append(_lp_ok)
    return _oks, _margins, float(np.mean(_lps))


# Role: measures how far the merged tokens appearing in each QA item have moved from the original
# Method: for every merged token in the item's full sentence measures the cosine against the original embedding and records the per-item minimum
# Why: identical before-after results on items that never moved are arithmetic rather than verification, so the number of genuinely tested items must be disclosed
# 역할: 질답 문항에 등장하는 묶음 토큰들이 원본에서 얼마나 움직였는지 잰다
# 방법: 문항 전체 문장의 묶음 토큰마다 원본 임베딩과의 코사인을 재 문항별 최솟값을 담는다
# 이유: 안 움직인 문항의 전후 동일은 검증이 아니라 산수이므로 실질 검증 문항 수를 공개해야 하기 때문
def p_item_movement(m, tok, base, norm_ref):
    _cur = bc.to_host(m.m_mat_w_data_axis / (xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9))
    _mins = []
    for prompt, ans, wrong in QA:
        _ids = set(tok.encode(prompt + ans)) | set(tok.encode(prompt + wrong))
        _bids = [t for t in _ids if t >= base]
        if not _bids:
            _mins.append(1.0)
            continue
        _cos = [float(np.dot(_cur[:, t], norm_ref[:, t])) for t in _bids]
        _mins.append(min(_cos))
    return _mins


# Role: compares two correctness lists and counts flips by direction
# Method: counts items that turned from correct to wrong and items that turned from wrong to correct separately
# Why: changes in paired before-after data should be reported by flip direction and count, not by total score
# 역할: 정오 목록 둘을 견줘 방향별 뒤집힘 수를 센다
# 방법: 맞다가 틀리게 된 문항 수와 틀리다가 맞게 된 문항 수를 따로 센다
# 이유: 전후 짝지은 자료의 변화는 총점이 아니라 뒤집힘 방향과 수로 보고해야 하기 때문
def p_flips(ok0, ok1):
    _b10 = sum(1 for a, b in zip(ok0, ok1) if a and not b)
    _b01 = sum(1 for a, b in zip(ok0, ok1) if not a and b)
    return _b10, _b01


# Role: runs one rumination round, measuring the QA set at the start and end and summarizing the before-after change
# Method: every step pulls same-kind groups drawn from the given seed pool and renormalizes globally, then at the end of the round measures correctness, margins, and movement
# Why: the published-schedule round and the forced-candidate round must be compared under the same procedure so that seed selection is their only difference
# 역할: 되새김 한 판을 돌리며 시작과 끝의 질답을 재고 전후를 요약한다
# 방법: 주어진 씨앗 풀에서 매 스텝 동종군을 당기고 전역 재정규화하며 판 끝에 정오와 정오 격차와 이동을 잰다
# 이유: 발표 스케줄 판과 후보 강제 투입 판이 같은 절차로 비교돼야 두 판의 차이가 씨앗 선택뿐이기 때문
def p_run(m, tok, base, pool, norm0, norm_ref, ev, block, label):
    _ok0, _mg0, _lp0 = p_qa_eval(m, tok)
    _h0 = probe2.p_holdout_bpc(m, tok, ev["홀드_중딩"], block)
    _a0 = sum(_ok0)
    _lo, _hi = p_wilson(_a0, len(QA))
    print(f"  step 0: 질답 {_a0}/{len(QA)} ({_a0/len(QA)*100:.1f}%, 신뢰구간 {_lo:.0f}~{_hi:.0f}%) · 정답 로그확률 {_lp0:+.2f} · 홀드중등 {_h0:.3f}", flush=True)
    np.random.seed(1)
    for step in range(STEPS):
        for _ in range(NSEED):
            _s = int(np.random.choice(pool))
            _g = paper2.firing_group(m, tok, _s)
            if len(_g) < 2:
                continue
            _group_gpu = xp.asarray(sorted(set(int(x) for x in _g)), dtype=xp.int64)
            _w_group = m.m_mat_w_data_axis[:, _group_gpu]
            _centroid = _w_group.mean(1, keepdims=True)
            m.m_mat_w_data_axis[:, _group_gpu] = _w_group + ALPHA * (_centroid - _w_group)
        m.m_mat_w_data_axis *= (norm0 / (xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True) + 1e-9))
        if (step + 1) % CKPT == 0:
            _ok, _mg, _lp = p_qa_eval(m, tok)
            _h = probe2.p_holdout_bpc(m, tok, ev["홀드_중딩"], block)
            print(f"  step {step + 1}: 질답 {sum(_ok)}/{len(QA)} · 정답 로그확률 {_lp:+.2f} · 홀드중등 {_h:.3f}", flush=True)
    _ok1, _mg1, _lp1 = p_qa_eval(m, tok)
    _b10, _b01 = p_flips(_ok0, _ok1)
    _mins = p_item_movement(m, tok, base, norm_ref)
    _moved = sum(1 for v in _mins if v < 0.999)
    _a1 = sum(_ok1)
    _lo1, _hi1 = p_wilson(_a1, len(QA))
    print(f"  [{label} 요약] 질답 {_a0}/{len(QA)} -> {_a1}/{len(QA)} ({_a1/len(QA)*100:.1f}%, 신뢰구간 {_lo1:.0f}~{_hi1:.0f}%) · 뒤집힘 정답->오답 {_b10} 오답->정답 {_b01}", flush=True)
    print(f"  [{label} 이동] 토큰이 실제로 움직인 문항 {_moved}/{len(QA)} · 문항 토큰 코사인 최솟값 {min(_mins):.3f}", flush=True)
    print(f"  [{label} 정오 격차] 전 {' '.join(f'{v:+.1f}' for v in _mg0)}", flush=True)
    print(f"  [{label} 정오 격차] 후 {' '.join(f'{v:+.1f}' for v in _mg1)}", flush=True)
    return _a0, _a1, _b10, _b01, _moved, _mins, _lp0, _lp1


def main():
    m, tok = bc.load_from(FROZEN)
    _base = tok.m_base_vocab
    _block = bc.CONTEXT_LENGTH
    _stream = np.load("model/stream_train.npy")
    _ev = json.load(open("model/eval_sets.json", encoding="utf-8"))
    _norm0 = xp.linalg.norm(m.m_mat_w_data_axis, axis=0, keepdims=True)
    _norm_ref = bc.to_host(m.m_mat_w_data_axis / (_norm0 + 1e-9))
    print(f"[제2편 되새김 질답] 얼린 모델 {os.path.basename(FROZEN)} · 2AFC {len(QA)}문항 우연 50% · 되새김 {STEPS}스텝 당김 {ALPHA}", flush=True)

    _ok_noq, _, _ = p_qa_eval(m, tok, use_question=False)
    print(f"  [사전선호 기준선] 질문을 지우고 답 낱말만 견주면 {sum(_ok_noq)}/{len(QA)} 문항이 정답 방향", flush=True)

    print(f"\n[실험1 발표 스케줄] 5.1 동종 되새김 그대로, 씨앗은 확신 풀에서", flush=True)
    _pool1 = probe2.p_core_seed_pool(m, tok, _stream, _base, _block)
    _r1 = p_run(m, tok, _base, _pool1, _norm0, _norm_ref, _ev, _block, "실험1")

    print(f"\n[실험2 후보 강제 투입] 질답 후보 낱말 자체를 씨앗으로 넣어 후보가 실제로 뭉치게 한 스트레스 판", flush=True)
    _m2, _ = bc.load_from(FROZEN)
    _cands = set()
    for prompt, ans, wrong in QA:
        for w in (ans, wrong):
            _ids = tok.encode(w)
            if len(_ids) == 1 and _ids[0] >= _base:
                _cands.add(_ids[0])
    _pool2 = sorted(_cands)
    print(f"  강제 씨앗 = 단일 묶음 후보 {len(_pool2)}개 (원자로 쪼개지는 후보는 못 움직이므로 제외)", flush=True)
    _r2 = p_run(_m2, tok, _base, _pool2, _norm0, _norm_ref, _ev, _block, "실험2")

    print("\n[실측 요약] 2AFC 16문항 우연 50%", flush=True)
    print(f"  사전선호 기준선(질문 제거) {sum(_ok_noq)}/{len(QA)}", flush=True)
    print(f"  실험1 발표 스케줄: {_r1[0]} -> {_r1[1]} · 뒤집힘 {_r1[2]}+{_r1[3]} · 움직인 문항 {_r1[4]}/{len(QA)}", flush=True)
    print(f"  실험2 후보 강제:   {_r2[0]} -> {_r2[1]} · 뒤집힘 {_r2[2]}+{_r2[3]} · 움직인 문항 {_r2[4]}/{len(QA)} · 코사인 최소 {min(_r2[5]):.3f}", flush=True)


if __name__ == "__main__":
    main()
