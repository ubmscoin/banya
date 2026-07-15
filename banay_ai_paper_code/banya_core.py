# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Banya common module. Gathers in one place the model class, the engine primitives, and the corpus build shared by the code of the six papers.

반야 공통  6개 논문 코드가 함께 쓰는 모델 클래스와 엔진 원시와 말뭉치 빌드를 한곳에 모은다."""
import os
import sys
import json
import cupy as xp
import numpy as np
from cupyx import scatter_add

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))
import banya_atoms as ba

CACHE_PATH = os.path.join(_ROOT, "banya_world_data", "bundle_cache.json")
SAVE_PATH = os.path.join(_ROOT, "model", "bitoken_cache.npz")

SEED = 0
HIDDEN_SIZE = 1024
BATCH_SIZE = 32
CONTEXT_LENGTH = 128
FFN_DEPTH = 2
NUMBER_TIME_MIX_HEAD = 16
NUMBER_BLOCK = 16
SCALE_LENGTH_MIN = 32
TIME_MIX_USE_FFT = False
USE_NORM = True
USE_RELATIVE_MIX = True
BLOCKWISE_CAUSAL_MIX = True
USE_GATE = True
USE_GOONGHAP = True
NUMBER_GOONGHAP_RANK = 32
GOONGHAP_SHARP_INIT = 2.0
NUMBER_GOONGHAP_BLOCK = 4
USE_WEIGHT_TIE = False
USE_TIE_SEPARATE_ADAM = False
USE_LOWRANK = True
HEAD_RANK = 128
HEAD_USE_FP16 = True
GOONGHAP_USE_FP16 = False
EMBED_SPARSE = True
USE_BITOKEN = True
LEARNING_RATE_ADAM = 0.002
DATA_TYPE = xp.float32
DATA_DIRS = ["banya_world_data", "data"]

WARMUP_MIX = {"life": 0.20,
              "space": 0.20,
              "sense": 0.20,
              "sense_space": 0.25,
              "sense_mimic": 0.15}

MIX_WEIGHTS = {
    "life": 0.10,
    "space": 0.08, "sense": 0.08, "sense_space": 0.08, "sense_mimic": 0.08,
    "baby": 0.40, "baby_logic": 0.30, "baby_learn": 0.30,
}

MIX_SCHED = {
    30000: {"life": 0.05, "space": 0.06, "sense": 0.06, "sense_space": 0.06, "sense_mimic": 0.06,
            "baby": 0.06, "baby_logic": 0.05, "baby_learn": 0.05,
            "toddler": 0.20, "toddler_logic": 0.06, "toddler_learn": 0.08, "toddler_exp": 0.10,
            "toddler_dialog": 0.18, "toddler_state": 0.12, "toddler_emotion": 0.08},
    70000: {"life": 0.03, "space": 0.03, "sense": 0.03, "sense_space": 0.03, "sense_mimic": 0.03,
            "baby": 0.02, "baby_logic": 0.02, "baby_learn": 0.02,
            "toddler": 0.04, "toddler_logic": 0.02, "toddler_learn": 0.02, "toddler_exp": 0.02,
            "toddler_dialog": 0.05, "toddler_state": 0.02, "toddler_emotion": 0.02,
            "toddler2": 0.35, "toddler2_link": 0.12},
    110000: {"life": 0.02, "space": 0.02, "sense": 0.02, "sense_space": 0.02, "sense_mimic": 0.02,
            "baby": 0.02, "baby_logic": 0.02, "baby_learn": 0.02,
            "toddler": 0.06, "toddler_logic": 0.04, "toddler_learn": 0.04, "toddler_exp": 0.05,
            "toddler_dialog": 0.07, "toddler_state": 0.06, "toddler_emotion": 0.04,
            "elem": 0.06, "elem_knowledge": 0.08, "elem_inquiry": 0.10, "elem_logic": 0.08, "elem_dialog": 0.18},
    130000: {"life": 0.02, "space": 0.02, "sense": 0.02, "sense_space": 0.02, "sense_mimic": 0.02,
            "baby": 0.02, "baby_logic": 0.02, "baby_learn": 0.02,
            "toddler": 0.04, "toddler_logic": 0.03, "toddler_learn": 0.03, "toddler_exp": 0.04,
            "toddler_dialog": 0.05, "toddler_state": 0.04, "toddler_emotion": 0.03,
            "toddler2": 0.06, "toddler2_link": 0.02,
            "tale_body": 0.16, "tale_qa": 0.03, "tale_summary": 0.06,
            "elem": 0.03, "elem_knowledge": 0.04, "elem_inquiry": 0.05, "elem_logic": 0.04, "elem_dialog": 0.09},
    170000: {"life": 0.02, "space": 0.02, "sense": 0.02, "sense_space": 0.02, "sense_mimic": 0.02,
            "baby": 0.02, "baby_logic": 0.02, "baby_learn": 0.02,
            "toddler": 0.06, "toddler_logic": 0.04, "toddler_learn": 0.04, "toddler_exp": 0.05,
            "toddler_dialog": 0.07, "toddler_state": 0.06, "toddler_emotion": 0.04,
            "tale_body": 0.03, "tale_qa": 0.01, "tale_summary": 0.01,
            "elem": 0.04, "elem_knowledge": 0.06, "elem_inquiry": 0.06, "elem_logic": 0.08, "elem_dialog": 0.14, "elem_subject": 0.12}}


def to_host(a):
    return xp.asnumpy(a)


def scatter_rows(dst, idx, src):
    scatter_add(dst, idx, src)


def hmm(a, b):
    return a @ b


_sm_exp_k = xp.ElementwiseKernel(
    'float16 z, float16 mx', 'float32 p',
    'p = expf((float)z - (float)mx);', 'banya_sm_exp')
_add3_k = xp.ElementwiseKernel(
    'float32 a, float32 b, float32 c', 'float32 y',
    'y = a + b + c;', 'banya_add3')
_rms_r_k = xp.ElementwiseKernel(
    'raw float32 X, int32 HH, int32 MT', 'float32 r',
    '''
    float acc = 0.0f;
    for (int c = 0; c < HH; ++c) {
        const float v = X[(long)c * MT + i];
        acc += v * v;
    }
    r = rsqrtf(acc / HH + 1e-6f);
    ''', 'banya_rms_r')
_rms_y_k = xp.ElementwiseKernel(
    'float32 x, raw float32 R, raw float32 G, int32 MT', 'float32 y',
    'y = x * R[i % MT] * G[i / MT];', 'banya_rms_y')
_rms_s_k = xp.ElementwiseKernel(
    'raw float32 DY, raw float32 X, raw float32 G, int32 HH, int32 MT', 'float32 s',
    '''
    float acc = 0.0f;
    for (int c = 0; c < HH; ++c) {
        const long o = (long)c * MT + i;
        acc += DY[o] * G[c] * X[o];
    }
    s = acc;
    ''', 'banya_rms_s')
_rms_dh_k = xp.ElementwiseKernel(
    'float32 dy, float32 x, raw float32 R, raw float32 S, raw float32 G, float32 invH, int32 MT', 'float32 dh',
    '''
    const int c = i / MT, m = i % MT;
    const float r = R[m];
    dh = r * dy * G[c] - (r * r * r * invH) * x * S[m];
    ''', 'banya_rms_dh')
_rms_dg_k = xp.ElementwiseKernel(
    'raw float32 DY, raw float32 X, raw float32 R, int32 MT, int32 NCH', 'float32 pg',
    '''
    const int c = i / NCH, ch = i % NCH;
    const long off = (long)c * MT;
    float acc = 0.0f;
    for (int m = ch; m < MT; m += NCH) {
        acc += DY[off + m] * X[off + m] * R[m];
    }
    pg = acc;
    ''', 'banya_rms_dg')
_toep_diag_k = xp.ElementwiseKernel(
    'raw float32 GC, int32 T', 'float32 gl',
    '''
    const int n = i / T, l = i % T;
    const long base = (long)n * T * T;
    float acc = 0.0f;
    for (int t = l; t < T; ++t) {
        acc += GC[base + (long)t * T + (t - l)];
    }
    gl = acc;
    ''', 'banya_toep_diag')
_flt_fwd_k = xp.ElementwiseKernel(
    'raw float32 Q, raw float32 W, raw float32 B, int32 L, int32 MT',
    'float32 y',
    '''
    const int c = i / MT, m = i % MT;
    y = tanhf(W[c] * Q[(long)(c % L) * MT + m] + B[c / L]);
    ''', 'banya_flt_fwd')
_flt_gw_k = xp.ElementwiseKernel(
    'raw float32 Q, raw float32 W, raw float32 B, raw float32 DL, int32 L, int32 MT, int32 NCH',
    'float32 pw, float32 pb',
    '''
    const int c = i / NCH, ch = i % NCH;
    const float w = W[c], b = B[c / L];
    const long qoff = (long)(c % L) * MT;
    const long doff = (long)c * MT;
    float aw = 0.0f, ab = 0.0f;
    for (int m = ch; m < MT; m += NCH) {
        const float q = Q[qoff + m];
        const float u = tanhf(w * q + b);
        const float dc = DL[doff + m] * (1.0f - u * u);
        aw += dc * q; ab += dc;
    }
    pw = aw; pb = ab;
    ''', 'banya_flt_gw')
_flt_dq_k = xp.ElementwiseKernel(
    'raw float32 Q, raw float32 W, raw float32 B, raw float32 DL, int32 L, int32 F, int32 MT',
    'float32 dq',
    '''
    const int qi = i / MT, m = i % MT;
    const float q = Q[(long)qi * MT + m];
    float acc = 0.0f;
    for (int f = 0; f < F; ++f) {
        const int c = f * L + qi;
        const float u = tanhf(W[c] * q + B[f]);
        acc += W[c] * DL[(long)c * MT + m] * (1.0f - u * u);
    }
    dq = acc;
    ''', 'banya_flt_dq')
_gate_fwd_k = xp.ElementwiseKernel(
    'float32 xf, float32 xm, raw float32 WG, raw float32 BG, int32 MT',
    'float32 h',
    '''
    const int c = i / MT;
    const float g = 1.0f / (1.0f + expf(-(WG[c] * xf + BG[c])));
    h = xf + g * xm;
    ''', 'banya_gate_fwd')
_gate_bwd_k = xp.ElementwiseKernel(
    'float32 dl, float32 fa0, float32 xf, raw float32 WG, raw float32 BG, int32 MT',
    'float32 ddir, float32 dlg',
    '''
    const int c = i / MT;
    const float g = 1.0f / (1.0f + expf(-(WG[c] * xf + BG[c])));
    const float dg = dl * (fa0 - xf) * (1.0f - g);
    ddir = dl + dg * WG[c];
    dlg = g * dl;
    ''', 'banya_gate_bwd')
_gate_gw_k = xp.ElementwiseKernel(
    'raw float32 DL, raw float32 FA, raw float32 XF, raw float32 WG, raw float32 BG, int32 MT, int32 NCH',
    'float32 pw, float32 pb',
    '''
    const int c = i / NCH, ch = i % NCH;
    const float w = WG[c], b = BG[c];
    const long off = (long)c * MT;
    float aw = 0.0f, ab = 0.0f;
    for (int m = ch; m < MT; m += NCH) {
        const float x = XF[off + m];
        const float g = 1.0f / (1.0f + expf(-(w * x + b)));
        const float dg = DL[off + m] * (FA[off + m] - x) * (1.0f - g);
        aw += dg * x; ab += dg;
    }
    pw = aw; pb = ab;
    ''', 'banya_gate_gw')
_gate_fwd_bt_k = xp.ElementwiseKernel(
    'float32 xf, float32 xm, float32 eg, raw float32 WG, raw float32 BG, int32 MT',
    'float32 h',
    '''
    const int c = i / MT;
    const float g = 1.0f / (1.0f + expf(-(WG[c] * xf + BG[c] + eg)));
    h = xf + g * xm;
    ''', 'banya_gate_fwd_bt')
_gate_bwd_bt_k = xp.ElementwiseKernel(
    'float32 dl, float32 fa0, float32 xf, float32 eg, raw float32 WG, raw float32 BG, int32 MT',
    'float32 ddir, float32 dlg, float32 dgt',
    '''
    const int c = i / MT;
    const float g = 1.0f / (1.0f + expf(-(WG[c] * xf + BG[c] + eg)));
    const float dg = dl * (fa0 - xf) * (1.0f - g);
    ddir = dl + dg * WG[c];
    dlg = g * dl;
    dgt = dg;
    ''', 'banya_gate_bwd_bt')
_gate_gw_bt_k = xp.ElementwiseKernel(
    'raw float32 DL, raw float32 FA, raw float32 XF, raw float32 EG, raw float32 WG, raw float32 BG, int32 MT, int32 NCH',
    'float32 pw, float32 pb',
    '''
    const int c = i / NCH, ch = i % NCH;
    const float w = WG[c], b = BG[c];
    const long off = (long)c * MT;
    float aw = 0.0f, ab = 0.0f;
    for (int m = ch; m < MT; m += NCH) {
        const float x = XF[off + m];
        const float g = 1.0f / (1.0f + expf(-(w * x + b + EG[off + m])));
        const float dg = DL[off + m] * (FA[off + m] - x) * (1.0f - g);
        aw += dg * x; ab += dg;
    }
    pw = aw; pb = ab;
    ''', 'banya_gate_gw_bt')


class CacheTokenizer:
    def __init__(self, cache_path=CACHE_PATH):
        self.base = ba.AtomTokenizer()
        with open(cache_path, encoding="utf-8") as f:
            _d = json.load(f)
        self.m_base_vocab = int(_d["base_vocab"])
        self.m_bundles = _d["bundles"]
        self.m_vocab_size = self.m_base_vocab + len(self.m_bundles)
        self.m_id_to_string = list(self.base.itos) + [b["str"] for b in self.m_bundles]
        self.m_string_to_id = dict(self.base.stoi)
        for b in self.m_bundles:
            self.m_string_to_id[b["str"]] = int(b["id"])
        self.m_bundle_tuple = {tuple(int(x) for x in b["syl"]): int(b["id"]) for b in self.m_bundles}
        self.m_bundle_syllable = {int(b["id"]): [int(x) for x in b["syl"]] for b in self.m_bundles}
        self.m_max_length = max((len(b["syl"]) for b in self.m_bundles), default=1)

    def bundle_ids(self, syl):
        _out = []
        i = 0
        _n = len(syl)
        _by = self.m_bundle_tuple
        while i < _n:
            _hit = None
            _hi = min(self.m_max_length, _n - i)
            for L in range(_hi, 1, -1):
                _bid = _by.get(tuple(int(x) for x in syl[i:i + L]))
                if _bid is not None:
                    _hit = (_bid, L)
                    break
            if _hit:
                _out.append(_hit[0])
                i += _hit[1]
            else:
                _out.append(int(syl[i]))
                i += 1
        return _out

    def encode(self, text):
        return self.bundle_ids(list(self.base.encode(text)))

    def decode(self, ids):
        _syl = []
        for t in ids:
            t = int(t)
            if t >= self.m_base_vocab:
                _syl.extend(self.m_bundle_syllable[t])
            else:
                _syl.append(t)
        return self.base.decode(_syl)


def bake_corpora(tok):
    _names = set(list(MIX_WEIGHTS) + list(WARMUP_MIX) + [n for kv in MIX_SCHED.values() for n in kv])
    for nm in sorted(_names):
        _src = os.path.join(_ROOT, "banya_world_data", nm + ".npy")
        if not os.path.exists(_src):
            continue
        _dst = os.path.join(_ROOT, "banya_world_data", nm + "_묶음.npy")
        if os.path.exists(_dst) and os.path.getmtime(_dst) > os.path.getmtime(CACHE_PATH):
            print(f"  [건너뜀] {nm}: 이미 최신 묶음본", flush=True)
            continue
        _a = np.asarray(np.load(_src, mmap_mode="r"), dtype=np.int64)
        _b = np.asarray(tok.bundle_ids(_a), dtype=np.int32)
        np.save(_dst, _b)
        print(f"  {nm}: 음절 {len(_a):,} -> 묶음 {len(_b):,} ({len(_b)/max(len(_a),1)*100:.0f}%) -> {nm}_묶음.npy", flush=True)


def open_corpora(names, vocab_size):
    _avail = {}
    for nm in names:
        _p = None
        for _dir in DATA_DIRS:
            _cb = os.path.join(_dir, f"{nm}_묶음.npy")
            if os.path.exists(_cb):
                _p = _cb
                break
        if _p is None:
            print(f"  [제외] {nm}: 묶음본 {nm}_묶음.npy 없음(bake 먼저 실행)", flush=True)
            continue
        _a = np.load(_p, mmap_mode="r")
        if len(_a) < CONTEXT_LENGTH + 2:
            continue
        _smp = np.asarray(_a[:min(len(_a), 2000000)])
        if int(_smp.max()) >= vocab_size:
            print(f"  [제외] {nm}: 옛 vocab_size(max {int(_smp.max())} >= {vocab_size}) 재토큰화 필요", flush=True)
            continue
        _avail[nm] = _a
    return _avail


def mix_dist(wdict, avail):
    _names = [n for n in wdict if n in avail]
    _w = np.array([wdict[n] for n in _names], dtype=np.float64)
    _w /= _w.sum()
    return _names, _w


def build_frequency(tok, corp_names):
    _freq = np.zeros(tok.m_vocab_size, dtype=np.int64)
    for nm in corp_names:
        _pth = os.path.join("banya_world_data", nm)
        if os.path.exists(_pth):
            _arr = np.asarray(np.load(_pth, mmap_mode="r"), dtype=np.int64)
            _freq += np.bincount(_arr, minlength=tok.m_vocab_size)[:tok.m_vocab_size]
    return _freq


def causal_time_mix(T):
    _n = np.arange(T)
    _C = np.cos(np.pi * (2 * _n[None, :] + 1) * _n[:, None] / (2 * T))
    _C = np.tril(_C)
    _C /= (np.abs(_C).sum(1, keepdims=True) + 1e-9)
    return _C


def blk_scale(d):
    s = 0
    while s < d and HIDDEN_SIZE % (2 << s) == 0 and (HIDDEN_SIZE >> (s + 1)) >= SCALE_LENGTH_MIN:
        s += 1
    return s


def pyr_pool(h, wm, s):
    _parts = [h]
    for k in range(s):
        _qr = _parts[k].reshape(-1, 2, h.shape[1])
        _parts.append(_qr[:, 0] * wm[k, 0] + _qr[:, 1] * wm[k, 1])
    return _parts


def pyr_fwd(W, b, h, s, wm):
    Mt = h.shape[1]
    L = h.shape[0] >> s
    _q = pyr_pool(h, wm, s)[-1]
    _y = xp.empty((h.shape[0], Mt), dtype=DATA_TYPE)
    _flt_fwd_k(_q, W, b, np.int32(L), np.int32(Mt), _y)
    return _y


def pyr_back(W, b, dl, h, s, wm):
    Mt = h.shape[1]
    F = 1 << s
    L = h.shape[0] >> s
    _parts = pyr_pool(h, wm, s)
    _q = _parts[-1]
    NCH = 256
    _pw = xp.empty((h.shape[0], NCH), dtype=DATA_TYPE)
    _pb = xp.empty((h.shape[0], NCH), dtype=DATA_TYPE)
    _flt_gw_k(_q, W, b, dl, np.int32(L), np.int32(Mt), np.int32(NCH), _pw, _pb)
    _gW = _pw.sum(1) / Mt
    _gB = _pb.sum(1).reshape(F, L).sum(1) / (L * Mt)
    _dq = xp.empty((L, Mt), dtype=DATA_TYPE)
    _flt_dq_k(_q, W, b, dl, np.int32(L), np.int32(F), np.int32(Mt), _dq)
    _gwm = xp.zeros((s, 2), dtype=DATA_TYPE)
    for k in range(s - 1, -1, -1):
        _qr = _parts[k].reshape(-1, 2, Mt)
        _gwm[k, 0] = (_dq * _qr[:, 0]).sum() / Mt
        _gwm[k, 1] = (_dq * _qr[:, 1]).sum() / Mt
        _up = xp.empty((_dq.shape[0], 2, Mt), dtype=DATA_TYPE)
        _up[:, 0] = _dq * wm[k, 0]
        _up[:, 1] = _dq * wm[k, 1]
        _dq = _up.reshape(-1, Mt)
    return _dq, _gW, _gB, _gwm


def adam(param, ms, vs, grad, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    ms *= b1
    ms += (1 - b1) * grad
    vs *= b2
    vs += (1 - b2) * grad * grad
    param -= lr * (ms / (1 - b1 ** t)) / (xp.sqrt(vs / (1 - b2 ** t)) + eps)


def adam_cols(param, ms, vs, cols, grad, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    _msc = b1 * ms[:, cols] + (1 - b1) * grad
    _vsc = b2 * vs[:, cols] + (1 - b2) * grad * grad
    ms[:, cols] = _msc
    vs[:, cols] = _vsc
    param[:, cols] = param[:, cols] - lr * (_msc / (1 - b1 ** t)) / (xp.sqrt(_vsc / (1 - b2 ** t)) + eps)


def rms_fwd(h, g):
    Mt = h.shape[1]
    _r = xp.empty((1, Mt), dtype=DATA_TYPE)
    _rms_r_k(h, np.int32(h.shape[0]), np.int32(Mt), _r)
    _y = xp.empty_like(h)
    _rms_y_k(h, _r, g, np.int32(Mt), _y)
    return _y, _r


# Role: transpose of RMSNorm; maps the error coming from above back into the input-side error and the gain gradient
# Method: gathers in one pass the inner product s of the gain-multiplied error and the input, then applies the Jacobian transpose formula elementwise with that s
# Why: the normalization layer must be retraced exactly without backpropagation so that the error reaches the embedding without distortion
# 역할: RMSNorm 의 전치. 위에서 내려온 오차를 입력쪽 오차와 게인 기울기로 되돌린다
# 방법: 게인을 곱한 오차와 입력의 내적 s 를 한 패스로 모으고 그 s 로 야코비안 전치식을 원소별로 적용
# 이유: 역전파를 쓰지 않고도 정규화층을 정확히 되짚어야 왜곡 없이 임베딩까지 오차를 전달할 수 있기 때문
def rms_vjp(dy, h, r, g):
    HH = h.shape[0]
    Mt = h.shape[1]
    _s = xp.empty((1, Mt), dtype=DATA_TYPE)
    _rms_s_k(dy, h, g, np.int32(HH), np.int32(Mt), _s)
    _dh = xp.empty_like(h)
    _rms_dh_k(dy, h, r, _s, g, np.float32(1.0 / HH), np.int32(Mt), _dh)
    NCH = 256
    _pg = xp.empty((HH, NCH), dtype=DATA_TYPE)
    _rms_dg_k(dy, h, r, np.int32(Mt), np.int32(NCH), _pg)
    return _dh, _pg.sum(1, keepdims=True)


# Role: unfolds the per-distance lag kernel into a lower-triangular Toeplitz matrix
# Method: uses the difference of two positions as the index to pick lag-kernel entries, and masks future-direction differences to zero
# Why: sharing the time mix by distance alone shrinks the parameters by a factor of the context length compared to keeping one per position pair
# 역할: 거리별 lag 커널을 하위삼각 Toeplitz 행렬로 편다
# 방법: 두 위치의 차이를 인덱스로 삼아 lag 커널을 골라 담고 미래 방향 차이는 0 으로 막는다
# 이유: 시간혼합을 거리 하나로 공유하면 위치쌍마다 따로 두는 것보다 파라미터가 문맥길이배 줄어든다
def toep_build(mat_w_lag):
    T = mat_w_lag.shape[-1]
    _ti = xp.arange(T)
    _lag = _ti[:, None] - _ti[None, :]
    return (mat_w_lag[..., xp.clip(_lag, 0, T - 1)] * (_lag >= 0)).astype(DATA_TYPE)


# Role: exact transpose of toep_build; folds the gradient unfolded as a Toeplitz matrix back into per-distance lag gradients
# Method: walks directly along each diagonal holding the same lag and adds the gradient values on it into that lag slot
# Why: the forward pass replicated each lag over many position pairs, so the transpose is exact only when those replicas are summed back into the same lag
# 역할: toep_build 의 정확한 전치. Toeplitz 로 편 기울기를 다시 거리별 lag 기울기로 접는다
# 방법: 같은 lag 이 놓인 대각선을 직접 걸어가며 그 위의 기울기 값을 lag 칸마다 더한다
# 이유: 순전파가 lag 을 여러 위치쌍에 복제했으니 전치는 그 복제들을 같은 lag 로 합쳐야 정확
def toep_scatter(gC):
    N = gC.shape[0]
    T = gC.shape[1]
    _gl = xp.empty((N, T), dtype=DATA_TYPE)
    _toep_diag_k(gC, np.int32(T), _gl)
    return _gl


# Role: computes via FFT the time-mix forward pass that mixes the context with a distance kernel
# Method: since the Toeplitz product equals a causal linear convolution, zero-pads to length 2T, multiplies in the frequency domain, and transforms back
# Why: as the context grows, the O(T logT) FFT is cheaper than the matrix product, and the values match the matrix path
# 역할: 거리 커널로 문맥을 섞는 시간혼합 순전파를 FFT 로 계산
# 방법: Toeplitz 곱이 인과 선형 합성곱과 같으므로 길이 2T 로 제로패딩해 주파수에서 곱하고 되돌린다
# 이유: 문맥이 길어지면 O(T logT) 인 FFT 가 행렬곱보다 싸고 값은 행렬 경로와 같다
def time_mix_forward(wlag, xr):
    T = xr.shape[2]
    L = 2 * T
    _kf = xp.fft.rfft(wlag, n=L, axis=-1)[:, None, :, None]
    _xf = xp.fft.rfft(xr, n=L, axis=2)
    return xp.fft.irfft(_kf * _xf, n=L, axis=2)[:, :, :T].astype(DATA_TYPE)


# Role: obtains the transpose of the time mix and the kernel gradient simultaneously via FFT
# Method: the transpose is undone by multiplying with the kernel conjugate, and the kernel gradient comes from the cross-correlation of the error and the input
# Why: even without backpropagation, the transpose and the kernel of the convolution are obtained in closed form, so the transpose is preserved
# 역할: 시간혼합의  전치와 커널 기울기를 FFT 로 동시에 구한다
# 방법: 전치는 커널 켤레를 곱해 되돌리고 커널 기울기는 오차와 입력의 교차상관으로
# 이유: 역전파 없이도 합성곱의 전치와 커널을 닫힌 형태로 얻어  전치를 유지
def time_mix_backward(wlag, dxr, xr):
    T = xr.shape[2]
    L = 2 * T
    _kf = xp.fft.rfft(wlag, n=L, axis=-1)[:, None, :, None]
    _df = xp.fft.rfft(dxr, n=L, axis=2)
    _dl = xp.fft.irfft(xp.conj(_kf) * _df, n=L, axis=2)[:, :, :T].astype(DATA_TYPE)
    _xf = xp.fft.rfft(xr, n=L, axis=2)
    _g = xp.fft.irfft((_df * xp.conj(_xf)).sum(axis=(1, 3)), n=L, axis=-1)[:, :T].astype(DATA_TYPE)
    return _dl, _g


class BanyaNoBP:
    def __init__(self, vocab_size, seed=0):
        _r = np.random.RandomState(seed)
        _rs = np.random.RandomState(seed + 1)
        self.m_mat_w_data_axis = xp.asarray(_r.randn(HIDDEN_SIZE, vocab_size) / np.sqrt(HIDDEN_SIZE), dtype=DATA_TYPE)
        self.m_mat_w_position = xp.asarray(_r.randn(HIDDEN_SIZE, CONTEXT_LENGTH) * 0.01, dtype=DATA_TYPE)
        _wp = np.empty((NUMBER_BLOCK, FFN_DEPTH, HIDDEN_SIZE), dtype=np.float32)
        for bl in range(NUMBER_BLOCK):
            for i in range(FFN_DEPTH):
                _wp[bl, i] = _rs.randn(HIDDEN_SIZE) * np.sqrt(2 ** blk_scale(bl + i))
        self.m_mat_w_filter = xp.asarray(_wp, dtype=DATA_TYPE)
        _maxs = blk_scale(NUMBER_BLOCK + FFN_DEPTH - 2)
        self.m_vec_w_filter_bias = xp.zeros((NUMBER_BLOCK, FFN_DEPTH, 1 << _maxs), dtype=DATA_TYPE)
        self.m_mat_w_mix = xp.full((NUMBER_BLOCK, FFN_DEPTH, _maxs, 2), 0.5, dtype=DATA_TYPE)
        self.m_mat_w_mix_adam_moment = xp.zeros_like(self.m_mat_w_mix)
        self.m_mat_w_mix_adam_variance = xp.zeros_like(self.m_mat_w_mix)
        if USE_LOWRANK:
            _head_std = (0.01 / HEAD_RANK) ** 0.25
            self.m_mat_w_head_a = xp.asarray(_r.randn(vocab_size, HEAD_RANK) * _head_std, dtype=DATA_TYPE)
            self.m_mat_w_head_b = xp.asarray(_r.randn(HEAD_RANK, HIDDEN_SIZE) * _head_std, dtype=DATA_TYPE)
        elif not USE_WEIGHT_TIE:
            self.m_mat_w_head = xp.asarray(_r.randn(vocab_size, HIDDEN_SIZE) * 0.1, dtype=DATA_TYPE)
        self.m_vec_w_head_bias = xp.zeros((vocab_size, 1), dtype=DATA_TYPE)
        self.m_vocab_size = vocab_size
        _base = causal_time_mix(CONTEXT_LENGTH)
        if not USE_RELATIVE_MIX:
            _cinit = np.tril(np.stack([_base + 0.02 * np.random.RandomState(seed + 10 + k).randn(CONTEXT_LENGTH, CONTEXT_LENGTH) for k in range(NUMBER_TIME_MIX_HEAD)]))
            if BLOCKWISE_CAUSAL_MIX:
                _cinit = np.broadcast_to(_cinit, (NUMBER_BLOCK,) + _cinit.shape).copy()
            self.m_mat_w_causal_mix = xp.asarray(_cinit, dtype=DATA_TYPE)
        self.m_causal_mask = xp.asarray(np.tril(np.ones((CONTEXT_LENGTH, CONTEXT_LENGTH))), dtype=DATA_TYPE)
        _zeros = lambda a: xp.zeros_like(a)
        self.m_mat_w_data_axis_adam_moment = _zeros(self.m_mat_w_data_axis)
        self.m_mat_w_data_axis_adam_variance = _zeros(self.m_mat_w_data_axis)
        if USE_WEIGHT_TIE and USE_TIE_SEPARATE_ADAM:
            self.m_mat_w_data_axis_head_adam_moment = _zeros(self.m_mat_w_data_axis)
            self.m_mat_w_data_axis_head_adam_variance = _zeros(self.m_mat_w_data_axis)
        if USE_LOWRANK:
            self.m_mat_w_head_a_adam_moment = _zeros(self.m_mat_w_head_a)
            self.m_mat_w_head_a_adam_variance = _zeros(self.m_mat_w_head_a)
            self.m_mat_w_head_b_adam_moment = _zeros(self.m_mat_w_head_b)
            self.m_mat_w_head_b_adam_variance = _zeros(self.m_mat_w_head_b)
        elif not USE_WEIGHT_TIE:
            self.m_mat_w_head_adam_moment = _zeros(self.m_mat_w_head)
            self.m_mat_w_head_adam_variance = _zeros(self.m_mat_w_head)
        self.m_vec_w_head_bias_adam_moment = _zeros(self.m_vec_w_head_bias)
        self.m_vec_w_head_bias_adam_variance = _zeros(self.m_vec_w_head_bias)
        if not USE_RELATIVE_MIX:
            self.m_mat_w_causal_mix_adam_moment = _zeros(self.m_mat_w_causal_mix)
            self.m_mat_w_causal_mix_adam_variance = _zeros(self.m_mat_w_causal_mix)
        self.m_mat_w_filter_adam_moment = _zeros(self.m_mat_w_filter)
        self.m_mat_w_filter_adam_variance = _zeros(self.m_mat_w_filter)
        if USE_RELATIVE_MIX:
            _wl = np.array([_base[np.arange(l, CONTEXT_LENGTH), np.arange(CONTEXT_LENGTH - l)].mean() for l in range(CONTEXT_LENGTH)])
            _wlarr = np.stack([_wl + 0.01 * np.random.RandomState(seed + 20 + k).randn(CONTEXT_LENGTH) for k in range(NUMBER_TIME_MIX_HEAD)])
            if BLOCKWISE_CAUSAL_MIX:
                _wlarr = np.broadcast_to(_wlarr, (NUMBER_BLOCK,) + _wlarr.shape).copy()
            self.m_mat_w_lag = xp.asarray(_wlarr, dtype=DATA_TYPE)
            self.m_mat_w_lag_adam_moment = _zeros(self.m_mat_w_lag)
            self.m_mat_w_lag_adam_variance = _zeros(self.m_mat_w_lag)
        if USE_NORM:
            self.m_vec_w_norm_gain = xp.ones((NUMBER_BLOCK, HIDDEN_SIZE, 1), dtype=DATA_TYPE)
            self.m_vec_w_norm_gain_adam_moment = _zeros(self.m_vec_w_norm_gain)
            self.m_vec_w_norm_gain_adam_variance = _zeros(self.m_vec_w_norm_gain)
        if USE_GATE:
            self.m_mat_w_gate = xp.zeros((NUMBER_BLOCK, HIDDEN_SIZE, 1), dtype=DATA_TYPE)
            self.m_vec_w_gate_bias = xp.full((NUMBER_BLOCK, HIDDEN_SIZE, 1), 2.0, dtype=DATA_TYPE)
            self.m_mat_w_gate_adam_moment = _zeros(self.m_mat_w_gate)
            self.m_mat_w_gate_adam_variance = _zeros(self.m_mat_w_gate)
            self.m_vec_w_gate_bias_adam_moment = _zeros(self.m_vec_w_gate_bias)
            self.m_vec_w_gate_bias_adam_variance = _zeros(self.m_vec_w_gate_bias)
        if USE_GOONGHAP:
            _rq = np.random.RandomState(seed + 40)
            self.m_mat_w_q = xp.asarray(_rq.randn(NUMBER_BLOCK, NUMBER_TIME_MIX_HEAD, HIDDEN_SIZE // NUMBER_TIME_MIX_HEAD, NUMBER_GOONGHAP_RANK) * 0.05, dtype=DATA_TYPE)
            self.m_mat_w_k = xp.asarray(_rq.randn(NUMBER_BLOCK, NUMBER_TIME_MIX_HEAD, HIDDEN_SIZE // NUMBER_TIME_MIX_HEAD, NUMBER_GOONGHAP_RANK) * 0.05, dtype=DATA_TYPE)
            self.m_vec_w_goonghap_gain = xp.zeros((NUMBER_BLOCK, NUMBER_TIME_MIX_HEAD), dtype=DATA_TYPE)
            self.m_mat_w_q_adam_moment = _zeros(self.m_mat_w_q)
            self.m_mat_w_q_adam_variance = _zeros(self.m_mat_w_q)
            self.m_mat_w_k_adam_moment = _zeros(self.m_mat_w_k)
            self.m_mat_w_k_adam_variance = _zeros(self.m_mat_w_k)
            self.m_vec_w_goonghap_gain_adam_moment = _zeros(self.m_vec_w_goonghap_gain)
            self.m_vec_w_goonghap_gain_adam_variance = _zeros(self.m_vec_w_goonghap_gain)
            self.m_vec_w_goonghap_sharp = xp.full((NUMBER_BLOCK, NUMBER_TIME_MIX_HEAD), GOONGHAP_SHARP_INIT, dtype=DATA_TYPE)
            self.m_vec_w_goonghap_sharp_adam_moment = _zeros(self.m_vec_w_goonghap_sharp)
            self.m_vec_w_goonghap_sharp_adam_variance = _zeros(self.m_vec_w_goonghap_sharp)
        if USE_BITOKEN:
            self.m_mat_w_operate_axis = xp.zeros((HIDDEN_SIZE, vocab_size), dtype=DATA_TYPE)
            self.m_mat_w_operate_axis_adam_moment = _zeros(self.m_mat_w_operate_axis)
            self.m_mat_w_operate_axis_adam_variance = _zeros(self.m_mat_w_operate_axis)
        self.t = 0


def block_fwd(m, x, bl, T, B, Hh):
    Mt = T * B
    _xr = x.reshape(NUMBER_TIME_MIX_HEAD, Hh, T, B)
    if USE_RELATIVE_MIX and TIME_MIX_USE_FFT:
        _wlag = m.m_mat_w_lag[bl] if BLOCKWISE_CAUSAL_MIX else m.m_mat_w_lag
        _xmix = time_mix_forward(_wlag[..., :T], _xr)
    else:
        _cmix = (m.m_cmix_all[bl] if BLOCKWISE_CAUSAL_MIX else m.m_mat_w_causal_mix)[..., :T, :T]
        _xs = xp.ascontiguousarray(_xr.transpose(0, 2, 1, 3)).reshape(NUMBER_TIME_MIX_HEAD, T, Hh * B)
        _xmix = xp.ascontiguousarray((_cmix @ _xs).reshape(NUMBER_TIME_MIX_HEAD, T, Hh, B).transpose(0, 2, 1, 3))
    if USE_GOONGHAP and bl >= NUMBER_BLOCK - NUMBER_GOONGHAP_BLOCK:
        _xq = xp.ascontiguousarray(_xr.transpose(0, 3, 2, 1))
        _qv = _xq @ m.m_mat_w_q[bl][:, None]
        _kv = _xq @ m.m_mat_w_k[bl][:, None]
        _qh = _qv / (xp.sqrt((_qv * _qv).sum(-1, keepdims=True)) + 1e-6)
        _kh = _kv / (xp.sqrt((_kv * _kv).sum(-1, keepdims=True)) + 1e-6)
        _tr = m.m_causal_mask[:T, :T]
        if GOONGHAP_USE_FP16:
            _cosf = (_qh.astype(xp.float16) @ _kh.astype(xp.float16).transpose(0, 1, 3, 2)).astype(DATA_TYPE)
        else:
            _cosf = _qh @ _kh.transpose(0, 1, 3, 2)
        _rowmax = (_cosf * _tr + (_tr - 1.0) * 1e9).max(-1, keepdims=True)
        _a = xp.exp(m.m_vec_w_goonghap_sharp[bl][:, None, None, None] * (_cosf - _rowmax) + (_tr - 1.0) * 1e9)
        _score = _a * m.m_vec_w_goonghap_gain[bl][:, None, None, None]
        if GOONGHAP_USE_FP16:
            _xmix = _xmix + (_score.astype(xp.float16) @ _xq.astype(xp.float16)).transpose(0, 3, 2, 1)
        else:
            _xmix = _xmix + (_score @ _xq).transpose(0, 3, 2, 1)
    _xmix = _xmix.reshape(HIDDEN_SIZE, Mt)
    _xf = x.reshape(HIDDEN_SIZE, Mt)
    if USE_GATE:
        _h = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
        if USE_BITOKEN:
            _gate_fwd_bt_k(_xf, _xmix, m.m_vec_operate_gate, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), _h)
        else:
            _gate_fwd_k(_xf, _xmix, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), _h)
    else:
        _h = _xf + _xmix
    _fa = []
    for i in range(FFN_DEPTH):
        _fa.append(_h)
        _h = pyr_fwd(m.m_mat_w_filter[bl, i], m.m_vec_w_filter_bias[bl, i], _h, blk_scale(bl + i), m.m_mat_w_mix[bl, i])
    if USE_NORM:
        _hres = _fa[0] + _h
        _hn, _rn = rms_fwd(_hres, m.m_vec_w_norm_gain[bl])
        return _hn.reshape(HIDDEN_SIZE, T, B), (_xr, _fa, _hres, _rn)
    return _h.reshape(HIDDEN_SIZE, T, B), (_xr, _fa)


def forward(m, ids):
    T, B = ids.shape
    Mt = T * B
    if USE_BITOKEN:
        m.m_vec_data_ids = m.m_mat_w_data_axis[:, ids]
        _vec_operate = m.m_mat_w_operate_axis[:, ids]
        _vec_operate_shift = xp.zeros_like(_vec_operate)
        _vec_operate_shift[:, 1:, :] = _vec_operate[:, :-1, :]
        m.m_vec_operate_gate = _vec_operate_shift.reshape(HIDDEN_SIZE, T * B)
        m.m_bt_ids = ids
        m.m_vec_operate_credit = xp.zeros((HIDDEN_SIZE, T * B), dtype=DATA_TYPE)
        _x = m.m_vec_data_ids + m.m_mat_w_position[:, :T, None]
    else:
        _x = m.m_mat_w_data_axis[:, ids] + m.m_mat_w_position[:, :T, None]
    Hh = HIDDEN_SIZE // NUMBER_TIME_MIX_HEAD
    _cache = [None] * NUMBER_BLOCK
    if USE_RELATIVE_MIX and not TIME_MIX_USE_FFT:
        if BLOCKWISE_CAUSAL_MIX:
            m.m_cmix_all = toep_build(m.m_mat_w_lag)
        else:
            m.m_mat_w_causal_mix = toep_build(m.m_mat_w_lag)
    elif not USE_RELATIVE_MIX and BLOCKWISE_CAUSAL_MIX:
        m.m_cmix_all = m.m_mat_w_causal_mix
    for bl in range(NUMBER_BLOCK):
        _x, _cache[bl] = block_fwd(m, _x, bl, T, B, Hh)
    _aD = _x.reshape(HIDDEN_SIZE, Mt)
    if HEAD_USE_FP16:
        _a16 = _aD.astype(xp.float16)
        if USE_LOWRANK:
            _z = hmm(m.m_mat_w_head_a.astype(xp.float16), hmm(m.m_mat_w_head_b.astype(xp.float16), _a16)) + m.m_vec_w_head_bias.astype(xp.float16)
        else:
            _z = (m.m_mat_w_data_axis.T if USE_WEIGHT_TIE else m.m_mat_w_head).astype(xp.float16) @ _a16 + m.m_vec_w_head_bias.astype(xp.float16)
    elif USE_LOWRANK:
        _z = hmm(m.m_mat_w_head_a, hmm(m.m_mat_w_head_b, _aD)) + m.m_vec_w_head_bias
    else:
        _z = (m.m_mat_w_data_axis.T if USE_WEIGHT_TIE else m.m_mat_w_head) @ _aD + m.m_vec_w_head_bias
    return _cache, _aD, _z


def p_softmax(z, y, ar):
    if z.dtype == xp.float16:
        _p = xp.empty(z.shape, dtype=DATA_TYPE)
        _sm_exp_k(z, z.max(0, keepdims=True), _p)
    else:
        z = z.astype(DATA_TYPE)
        _p = xp.exp(z - z.max(0, keepdims=True))
    _p /= _p.sum(0, keepdims=True)
    _ce = -xp.mean(xp.log(_p[y, ar] + 1e-9))
    _g = _p
    _g[y, ar] -= 1.0
    return _g, _ce


def head_delta_top(m, aD, g, Mt, do_update):
    if USE_LOWRANK:
        _tlr = hmm(m.m_mat_w_head_b, aD)
        _gt = -hmm(m.m_mat_w_head_a.T, g)
        _dtop = hmm(m.m_mat_w_head_b.T, _gt)
        if do_update:
            adam(m.m_mat_w_head_a, m.m_mat_w_head_a_adam_moment, m.m_mat_w_head_a_adam_variance, hmm(g, _tlr.T) / Mt, LEARNING_RATE_ADAM, m.t)
            adam(m.m_mat_w_head_b, m.m_mat_w_head_b_adam_moment, m.m_mat_w_head_b_adam_variance, hmm(-_gt, aD.T) / Mt, LEARNING_RATE_ADAM, m.t)
        _g_data_axis_head = None
    else:
        _dtop = (m.m_mat_w_data_axis if USE_WEIGHT_TIE else m.m_mat_w_head.T) @ (-g)
        if do_update and not USE_WEIGHT_TIE:
            adam(m.m_mat_w_head, m.m_mat_w_head_adam_moment, m.m_mat_w_head_adam_variance, g @ aD.T / Mt, LEARNING_RATE_ADAM, m.t)
        _g_data_axis_head = aD @ g.T / Mt if USE_WEIGHT_TIE else None
    if do_update:
        adam(m.m_vec_w_head_bias, m.m_vec_w_head_bias_adam_moment, m.m_vec_w_head_bias_adam_variance, g.mean(1, keepdims=True), LEARNING_RATE_ADAM, m.t)
    return _dtop, _g_data_axis_head


# Role: walks the blocks in reverse order, flowing the transpose through, and gathers the per-block gradients at once
# Method: retraces USE_NORM, the filter bank, the gate, the goonghap, and the time mix in order, accumulating each parameter's credit into buffers
# Why: nothing is read back after an update within a step, so the math equals per-item updates, and many small kernels shrink to a few large ones
# 역할: 블록을 역순으로 돌며  전치를 흘려 블록별 기울기를 한꺼번에 모은다
# 방법: USE_NORM 과 필터뱅크와 게이트와 궁합과 시간혼합을 순서대로 되짚어 각 파라미터의 책임량을 버퍼에 쌓는다
# 이유: 스텝 안에서 갱신 후 다시 읽는 곳이 없어 낱개 갱신과 수학이 같고 작은 커널이 큰 커널 몇 개로 줄어든다
def exact_chain(m, cache, blocks, dx, Mt, Hh):
    _grad_causal_acc = 0
    _grad_filter = xp.empty_like(m.m_mat_w_filter)
    _grad_filter_bias = xp.zeros_like(m.m_vec_w_filter_bias)
    _grad_mix = xp.zeros_like(m.m_mat_w_mix)
    _grad_norm_gain = xp.empty_like(m.m_vec_w_norm_gain) if USE_NORM else None
    _grad_gate = xp.empty_like(m.m_mat_w_gate) if USE_GATE else None
    _grad_gate_bias = xp.empty_like(m.m_vec_w_gate_bias) if USE_GATE else None
    _grad_q = xp.zeros_like(m.m_mat_w_q) if USE_GOONGHAP else None
    _grad_k = xp.zeros_like(m.m_mat_w_k) if USE_GOONGHAP else None
    _grad_goonghap_gain = xp.zeros_like(m.m_vec_w_goonghap_gain) if USE_GOONGHAP else None
    _grad_goonghap_sharp = xp.zeros_like(m.m_vec_w_goonghap_sharp) if USE_GOONGHAP else None
    _grad_timemix = None
    for bl in blocks:
        c = cache[bl]
        if USE_NORM:
            _xr, _fa, _hres, _rn = c
            _dhres, _dGn = rms_vjp(dx, _hres, _rn, m.m_vec_w_norm_gain[bl])
            _grad_norm_gain[bl] = _dGn
            _dl = _dhres
        else:
            _xr, _fa = c
            _dl = dx
        for i in reversed(range(FFN_DEPTH)):
            s = blk_scale(bl + i)
            _dl, _gW, _gB, _gwm = pyr_back(m.m_mat_w_filter[bl, i], m.m_vec_w_filter_bias[bl, i], _dl, _fa[i], s, m.m_mat_w_mix[bl, i])
            _grad_mix[bl, i, :s] = _gwm
            _grad_filter[bl, i] = _gW
            _grad_filter_bias[bl, i, :_gB.shape[0]] = _gB
        if USE_NORM:
            _dl = _dl + _dhres
        T = _xr.shape[2]
        B = _xr.shape[3]
        if USE_GATE:
            _xf = _xr.reshape(HIDDEN_SIZE, Mt)
            if USE_BITOKEN:
                NCH = 256
                _pw = xp.empty((HIDDEN_SIZE, NCH), dtype=DATA_TYPE)
                _pb = xp.empty((HIDDEN_SIZE, NCH), dtype=DATA_TYPE)
                _gate_gw_bt_k(_dl, _fa[0], _xf, m.m_vec_operate_gate, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), np.int32(NCH), _pw, _pb)
                _grad_gate[bl] = _pw.sum(1, keepdims=True) / Mt
                _grad_gate_bias[bl] = _pb.sum(1, keepdims=True) / Mt
                _ddir = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
                _dlg = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
                _dgt = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
                _gate_bwd_bt_k(_dl, _fa[0], _xf, m.m_vec_operate_gate, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), _ddir, _dlg, _dgt)
                m.m_vec_operate_credit = m.m_vec_operate_credit + _dgt
                _dl = _dlg
            else:
                NCH = 256
                _pw = xp.empty((HIDDEN_SIZE, NCH), dtype=DATA_TYPE)
                _pb = xp.empty((HIDDEN_SIZE, NCH), dtype=DATA_TYPE)
                _gate_gw_k(_dl, _fa[0], _xf, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), np.int32(NCH), _pw, _pb)
                _grad_gate[bl] = _pw.sum(1, keepdims=True) / Mt
                _grad_gate_bias[bl] = _pb.sum(1, keepdims=True) / Mt
                _ddir = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
                _dlg = xp.empty((HIDDEN_SIZE, Mt), dtype=DATA_TYPE)
                _gate_bwd_k(_dl, _fa[0], _xf, m.m_mat_w_gate[bl], m.m_vec_w_gate_bias[bl], np.int32(Mt), _ddir, _dlg)
                _dl = _dlg
        else:
            _ddir = _dl
        _dxr = _dl.reshape(NUMBER_TIME_MIX_HEAD, Hh, T, B)
        if USE_GOONGHAP and bl >= NUMBER_BLOCK - NUMBER_GOONGHAP_BLOCK:
            _xq = xp.ascontiguousarray(_xr.transpose(0, 3, 2, 1))
            _dxq = xp.ascontiguousarray(_dxr.transpose(0, 3, 2, 1))
            _qv = _xq @ m.m_mat_w_q[bl][:, None]
            _kv = _xq @ m.m_mat_w_k[bl][:, None]
            _nq = xp.sqrt((_qv * _qv).sum(-1, keepdims=True)) + 1e-6
            _nk = xp.sqrt((_kv * _kv).sum(-1, keepdims=True)) + 1e-6
            _qh = _qv / _nq
            _kh = _kv / _nk
            _tr = m.m_causal_mask[:T, :T]
            _cosf = _qh @ _kh.transpose(0, 1, 3, 2)
            _rowmax = (_cosf * _tr + (_tr - 1.0) * 1e9).max(-1, keepdims=True)
            _bbc = m.m_vec_w_goonghap_sharp[bl][:, None, None, None]
            _a = xp.exp(_bbc * (_cosf - _rowmax) + (_tr - 1.0) * 1e9)
            _dsc = (_dxq @ _xq.transpose(0, 1, 3, 2)) * _tr
            _gq4 = m.m_vec_w_goonghap_gain[bl][:, None, None, None]
            _grad_goonghap_gain[bl] = (_dsc * _a).sum(axis=(1, 2, 3)) / Mt
            _grad_goonghap_sharp[bl] = (_dsc * _gq4 * (_cosf - _rowmax) * _a).sum(axis=(1, 2, 3)) / Mt
            _dcos = _dsc * _gq4 * _bbc * _a
            _dqh = _dcos @ _kh
            _dkh = _dcos.transpose(0, 1, 3, 2) @ _qh
            _dqv = (_dqh - _qh * (_qh * _dqh).sum(-1, keepdims=True)) / _nq
            _dkv = (_dkh - _kh * (_kh * _dkh).sum(-1, keepdims=True)) / _nk
            _grad_q[bl] = xp.einsum('kbtg,kbtr->kgr', _xq, _dqv) / Mt
            _grad_k[bl] = xp.einsum('kbtg,kbtr->kgr', _xq, _dkv) / Mt
            _ca = (_a * _gq4).transpose(0, 1, 3, 2) @ _dxq
            _cb = _dqv @ m.m_mat_w_q[bl][:, None].transpose(0, 1, 3, 2)
            _cc = _dkv @ m.m_mat_w_k[bl][:, None].transpose(0, 1, 3, 2)
            _dxqo = xp.empty_like(_ca)
            _add3_k(_ca, _cb, _cc, _dxqo)
            _dqmix = _dxqo.transpose(0, 3, 2, 1)
        if USE_RELATIVE_MIX and TIME_MIX_USE_FFT:
            _wlag = m.m_mat_w_lag[bl] if BLOCKWISE_CAUSAL_MIX else m.m_mat_w_lag
            _dtm, _gw = time_mix_backward(_wlag, _dxr, _xr)
            if BLOCKWISE_CAUSAL_MIX:
                if _grad_timemix is None:
                    _grad_timemix = xp.empty((NUMBER_BLOCK,) + _gw.shape, dtype=DATA_TYPE)
                _grad_timemix[bl] = _gw
            else:
                _grad_causal_acc = _grad_causal_acc + _gw
            _tt = _dtm
        else:
            _cmix = m.m_cmix_all[bl] if BLOCKWISE_CAUSAL_MIX else m.m_mat_w_causal_mix
            _xs = xp.ascontiguousarray(_xr.transpose(0, 2, 1, 3)).reshape(NUMBER_TIME_MIX_HEAD, T, Hh * B)
            _ds = xp.ascontiguousarray(_dxr.transpose(0, 2, 1, 3)).reshape(NUMBER_TIME_MIX_HEAD, T, Hh * B)
            _gcmix = (_ds @ _xs.transpose(0, 2, 1)) * m.m_causal_mask
            _tt = xp.ascontiguousarray((_cmix.transpose(0, 2, 1) @ _ds).reshape(NUMBER_TIME_MIX_HEAD, T, Hh, B).transpose(0, 2, 1, 3))
            if BLOCKWISE_CAUSAL_MIX:
                if _grad_timemix is None:
                    _grad_timemix = xp.empty((NUMBER_BLOCK,) + _gcmix.shape, dtype=DATA_TYPE)
                _grad_timemix[bl] = _gcmix
            else:
                _grad_causal_acc = _grad_causal_acc + _gcmix
        if USE_GOONGHAP and bl >= NUMBER_BLOCK - NUMBER_GOONGHAP_BLOCK:
            _dxin = xp.empty_like(_tt)
            _add3_k(_ddir.reshape(NUMBER_TIME_MIX_HEAD, Hh, T, B), _tt, _dqmix, _dxin)
        else:
            _dxin = _ddir.reshape(NUMBER_TIME_MIX_HEAD, Hh, T, B) + _tt
        dx = _dxin.reshape(HIDDEN_SIZE, Mt)
    return dx, (_grad_norm_gain, _grad_mix, _grad_filter, _grad_filter_bias, _grad_timemix, _grad_causal_acc, _grad_gate, _grad_gate_bias, _grad_q, _grad_k, _grad_goonghap_gain, _grad_goonghap_sharp)


# Role: transposes through the blocks in reverse order and updates the filters, the time mix, and the embedding in one pass
# Method: gathers per-block gradients with exact_chain, then applies the Adam and SGD updates with those gradients in one batch at the end of the step
# Why: this is the body of zero backpropagation, which trains down to the embedding by flowing through transposes alone without backpropagation
# 역할: 블록 역순으로  전치해 필터와 시간혼합과 임베딩까지 한 번에 갱신
# 방법: exact_chain 으로 블록별 기울기를 모으고 그 기울기로 아담과 SGD 갱신을 스텝 끝에 일괄 적용
# 이유: 역전파를 쓰지 않고  전치만으로  흘려 임베딩까지 학습하는 역전파 0 의 본체이기 때문
def block_credit(m, cache, dl_top, lr, ids, T, B, g_data_axis_head):
    Mt = T * B
    Hh = HIDDEN_SIZE // NUMBER_TIME_MIX_HEAD
    _dx, _grads = exact_chain(m, cache, list(reversed(range(NUMBER_BLOCK))), dl_top, Mt, Hh)
    _grad_norm_gain, _grad_mix, _grad_filter, _grad_filter_bias, _grad_timemix, _grad_causal_acc, _grad_gate, _grad_gate_bias, _grad_q, _grad_k, _grad_goonghap_gain, _grad_goonghap_sharp = _grads
    if USE_NORM:
        adam(m.m_vec_w_norm_gain, m.m_vec_w_norm_gain_adam_moment, m.m_vec_w_norm_gain_adam_variance, -_grad_norm_gain, LEARNING_RATE_ADAM, m.t)
    if USE_GATE:
        adam(m.m_mat_w_gate, m.m_mat_w_gate_adam_moment, m.m_mat_w_gate_adam_variance, -_grad_gate, LEARNING_RATE_ADAM, m.t)
        adam(m.m_vec_w_gate_bias, m.m_vec_w_gate_bias_adam_moment, m.m_vec_w_gate_bias_adam_variance, -_grad_gate_bias, LEARNING_RATE_ADAM, m.t)
    if USE_GOONGHAP:
        adam(m.m_mat_w_q, m.m_mat_w_q_adam_moment, m.m_mat_w_q_adam_variance, -_grad_q, LEARNING_RATE_ADAM, m.t)
        adam(m.m_mat_w_k, m.m_mat_w_k_adam_moment, m.m_mat_w_k_adam_variance, -_grad_k, LEARNING_RATE_ADAM, m.t)
        adam(m.m_vec_w_goonghap_gain, m.m_vec_w_goonghap_gain_adam_moment, m.m_vec_w_goonghap_gain_adam_variance, -_grad_goonghap_gain, LEARNING_RATE_ADAM, m.t)
        adam(m.m_vec_w_goonghap_sharp, m.m_vec_w_goonghap_sharp_adam_moment, m.m_vec_w_goonghap_sharp_adam_variance, -_grad_goonghap_sharp, LEARNING_RATE_ADAM, m.t)
    adam(m.m_mat_w_mix, m.m_mat_w_mix_adam_moment, m.m_mat_w_mix_adam_variance, -_grad_mix, LEARNING_RATE_ADAM, m.t)
    adam(m.m_mat_w_filter, m.m_mat_w_filter_adam_moment, m.m_mat_w_filter_adam_variance, -_grad_filter, LEARNING_RATE_ADAM, m.t)
    m.m_vec_w_filter_bias += lr * _grad_filter_bias
    if BLOCKWISE_CAUSAL_MIX and _grad_timemix is not None:
        if USE_RELATIVE_MIX and TIME_MIX_USE_FFT:
            adam(m.m_mat_w_lag, m.m_mat_w_lag_adam_moment, m.m_mat_w_lag_adam_variance, -_grad_timemix, LEARNING_RATE_ADAM, m.t)
        elif USE_RELATIVE_MIX:
            _grad_lag = toep_scatter(_grad_timemix.reshape(NUMBER_BLOCK * NUMBER_TIME_MIX_HEAD, T, T)).reshape(NUMBER_BLOCK, NUMBER_TIME_MIX_HEAD, T)
            adam(m.m_mat_w_lag, m.m_mat_w_lag_adam_moment, m.m_mat_w_lag_adam_variance, -_grad_lag, LEARNING_RATE_ADAM, m.t)
        else:
            adam(m.m_mat_w_causal_mix, m.m_mat_w_causal_mix_adam_moment, m.m_mat_w_causal_mix_adam_variance, -_grad_timemix, LEARNING_RATE_ADAM, m.t)
    if not BLOCKWISE_CAUSAL_MIX:
        if USE_RELATIVE_MIX:
            adam(m.m_mat_w_lag, m.m_mat_w_lag_adam_moment, m.m_mat_w_lag_adam_variance, -toep_scatter(_grad_causal_acc), LEARNING_RATE_ADAM, m.t)
        else:
            adam(m.m_mat_w_causal_mix, m.m_mat_w_causal_mix_adam_moment, m.m_mat_w_causal_mix_adam_variance, -_grad_causal_acc, LEARNING_RATE_ADAM, m.t)
    _dx3 = _dx.reshape(HIDDEN_SIZE, T, B)
    m.m_mat_w_position += lr * _dx3.sum(2) / Mt
    if USE_BITOKEN:
        _dx_data_axis = _dx
        _operate_gate_3d = m.m_vec_operate_credit.reshape(HIDDEN_SIZE, T, B)
        _operate_axis_credit = _operate_gate_3d[:, 1:, :]
        _ids_prev = m.m_bt_ids[:-1]
        _delta_operate_axis = xp.zeros((m.m_vocab_size, HIDDEN_SIZE), dtype=DATA_TYPE)
        scatter_rows(_delta_operate_axis, _ids_prev.reshape(-1), (_operate_axis_credit.reshape(HIDDEN_SIZE, -1) / Mt).T)
        adam(m.m_mat_w_operate_axis, m.m_mat_w_operate_axis_adam_moment, m.m_mat_w_operate_axis_adam_variance, -_delta_operate_axis.T, LEARNING_RATE_ADAM, m.t)
    else:
        _dx_data_axis = _dx
    if EMBED_SPARSE and not (USE_WEIGHT_TIE and g_data_axis_head is not None):
        _flat = ids.reshape(-1)
        _uniq, _inv = xp.unique(_flat, return_inverse=True)
        _grad_embed = xp.zeros((_uniq.shape[0], HIDDEN_SIZE), dtype=DATA_TYPE)
        scatter_rows(_grad_embed, _inv.reshape(-1), (_dx_data_axis / Mt).T)
        adam_cols(m.m_mat_w_data_axis, m.m_mat_w_data_axis_adam_moment, m.m_mat_w_data_axis_adam_variance, _uniq, -_grad_embed.T, LEARNING_RATE_ADAM, m.t)
    else:
        _delta_data_axis = xp.zeros((m.m_vocab_size, HIDDEN_SIZE), dtype=DATA_TYPE)
        scatter_rows(_delta_data_axis, ids.reshape(-1), (_dx_data_axis / Mt).T)
        if USE_WEIGHT_TIE and USE_TIE_SEPARATE_ADAM:
            adam(m.m_mat_w_data_axis, m.m_mat_w_data_axis_adam_moment, m.m_mat_w_data_axis_adam_variance, -_delta_data_axis.T, LEARNING_RATE_ADAM, m.t)
            if g_data_axis_head is not None:
                adam(m.m_mat_w_data_axis, m.m_mat_w_data_axis_head_adam_moment, m.m_mat_w_data_axis_head_adam_variance, g_data_axis_head, LEARNING_RATE_ADAM, m.t)
        else:
            _grad_data_axis = (-_delta_data_axis.T + g_data_axis_head) if (USE_WEIGHT_TIE and g_data_axis_head is not None) else (-_delta_data_axis.T)
            adam(m.m_mat_w_data_axis, m.m_mat_w_data_axis_adam_moment, m.m_mat_w_data_axis_adam_variance, _grad_data_axis, LEARNING_RATE_ADAM, m.t)


def save(m, tok, step=0):
    _d = dict(m_mat_w_data_axis=to_host(m.m_mat_w_data_axis),
              m_mat_w_position=to_host(m.m_mat_w_position),
              m_vec_w_head_bias=to_host(m.m_vec_w_head_bias),
              m_id_to_string=np.array(tok.m_id_to_string),
              step=np.array(step))
    if hasattr(m, "m_mat_w_causal_mix"):
        _d["m_mat_w_causal_mix"] = to_host(m.m_mat_w_causal_mix)
    _d["m_mat_w_filter"] = to_host(m.m_mat_w_filter)
    _d["m_mat_w_filter_adam_moment"] = to_host(m.m_mat_w_filter_adam_moment)
    _d["m_mat_w_filter_adam_variance"] = to_host(m.m_mat_w_filter_adam_variance)
    _d["m_vec_w_filter_bias"] = to_host(m.m_vec_w_filter_bias)
    if USE_LOWRANK:
        _d["m_mat_w_head_a"] = to_host(m.m_mat_w_head_a)
        _d["m_mat_w_head_b"] = to_host(m.m_mat_w_head_b)
    elif not USE_WEIGHT_TIE:
        _d["m_mat_w_head"] = to_host(m.m_mat_w_head)
    if USE_NORM:
        _d["m_vec_w_norm_gain"] = to_host(m.m_vec_w_norm_gain)
    if USE_RELATIVE_MIX:
        _d["m_mat_w_lag"] = to_host(m.m_mat_w_lag)
    if USE_GATE:
        _d["m_mat_w_gate"] = to_host(m.m_mat_w_gate)
        _d["m_vec_w_gate_bias"] = to_host(m.m_vec_w_gate_bias)
    if USE_GOONGHAP:
        _d["m_mat_w_q"] = to_host(m.m_mat_w_q)
        _d["m_mat_w_k"] = to_host(m.m_mat_w_k)
        _d["m_vec_w_goonghap_gain"] = to_host(m.m_vec_w_goonghap_gain)
        _d["m_vec_w_goonghap_sharp"] = to_host(m.m_vec_w_goonghap_sharp)
    _d["m_mat_w_mix"] = to_host(m.m_mat_w_mix)
    if USE_BITOKEN:
        _d["m_mat_w_operate_axis"] = to_host(m.m_mat_w_operate_axis)
    _moment_names = ["m_mat_w_data_axis", "m_vec_w_head_bias", "m_mat_w_mix"]
    if USE_BITOKEN:
        _moment_names += ["m_mat_w_operate_axis"]
    if USE_LOWRANK:
        _moment_names += ["m_mat_w_head_a", "m_mat_w_head_b"]
    elif not USE_WEIGHT_TIE:
        _moment_names += ["m_mat_w_head"]
    if USE_NORM:
        _moment_names += ["m_vec_w_norm_gain"]
    if USE_GATE:
        _moment_names += ["m_mat_w_gate", "m_vec_w_gate_bias"]
    if USE_GOONGHAP:
        _moment_names += ["m_mat_w_q", "m_mat_w_k", "m_vec_w_goonghap_gain", "m_vec_w_goonghap_sharp"]
    _moment_names += ["m_mat_w_lag"] if USE_RELATIVE_MIX else ["m_mat_w_causal_mix"]
    if USE_WEIGHT_TIE and USE_TIE_SEPARATE_ADAM:
        _moment_names += ["m_mat_w_data_axis_head"]
    for name in _moment_names:
        _d[name + "_adam_moment"] = to_host(getattr(m, name + "_adam_moment"))
        _d[name + "_adam_variance"] = to_host(getattr(m, name + "_adam_variance"))
    _d["t"] = np.array(m.t)
    _tmp = SAVE_PATH + ".tmp"
    with open(_tmp, "wb") as f:
        np.savez(f, **_d)
    os.replace(_tmp, SAVE_PATH)


def load():
    _d = np.load(SAVE_PATH, allow_pickle=True)
    global HIDDEN_SIZE, CONTEXT_LENGTH, NUMBER_BLOCK, FFN_DEPTH, NUMBER_TIME_MIX_HEAD, USE_WEIGHT_TIE, USE_NORM, USE_RELATIVE_MIX, USE_LOWRANK, HEAD_RANK, BLOCKWISE_CAUSAL_MIX, USE_GATE, USE_GOONGHAP, NUMBER_GOONGHAP_RANK, SCALE_LENGTH_MIN
    HIDDEN_SIZE = int(_d["m_mat_w_data_axis"].shape[0])
    CONTEXT_LENGTH = int(_d["m_mat_w_position"].shape[1])
    if "m_vec_w_filter_bias" not in _d.files:
        sys.exit(f"[!] {SAVE_PATH} 는 옛 구조(전스케일 피라미드) 체크포인트다. 이 판에서는 못 읽는다. 새로 학습하라")
    NUMBER_BLOCK = int(_d["m_mat_w_filter"].shape[0])
    FFN_DEPTH = int(_d["m_mat_w_filter"].shape[1])
    SCALE_LENGTH_MIN = HIDDEN_SIZE >> int(_d["m_mat_w_mix"].shape[2])
    if int(_d["m_mat_w_mix"].shape[2]) != blk_scale(NUMBER_BLOCK + FFN_DEPTH - 2):
        sys.exit(f"[!] {SAVE_PATH} 는 겹 사다리 이전 저장본이라 m_mat_w_mix 폭이 다르다. 새로 학습하라")
    if "m_mat_w_causal_mix" in _d.files:
        BLOCKWISE_CAUSAL_MIX = (_d["m_mat_w_causal_mix"].ndim == 4)
        NUMBER_TIME_MIX_HEAD = int(_d["m_mat_w_causal_mix"].shape[1]) if BLOCKWISE_CAUSAL_MIX else int(_d["m_mat_w_causal_mix"].shape[0])
    else:
        BLOCKWISE_CAUSAL_MIX = (_d["m_mat_w_lag"].ndim == 3)
        NUMBER_TIME_MIX_HEAD = int(_d["m_mat_w_lag"].shape[1]) if BLOCKWISE_CAUSAL_MIX else int(_d["m_mat_w_lag"].shape[0])
    USE_LOWRANK = ("m_mat_w_head_a" in _d.files)
    if USE_LOWRANK:
        HEAD_RANK = int(_d["m_mat_w_head_a"].shape[1])
    USE_WEIGHT_TIE = ("m_mat_w_head" not in _d.files) and not USE_LOWRANK
    USE_NORM = ("m_vec_w_norm_gain" in _d.files)
    USE_RELATIVE_MIX = ("m_mat_w_lag" in _d.files)
    USE_GATE = ("m_mat_w_gate" in _d.files)
    USE_GOONGHAP = ("m_mat_w_q" in _d.files)
    if USE_GOONGHAP:
        NUMBER_GOONGHAP_RANK = int(_d["m_mat_w_q"].shape[3])
    tok = CacheTokenizer()
    _vocab = int(_d["m_mat_w_data_axis"].shape[1])
    if _vocab != tok.m_vocab_size:
        print(f"[!] 경고: 체크포인트 vocab_size {_vocab} != 캐시 사전 {tok.m_vocab_size}. 캐시 json 이 바뀐 듯", flush=True)
    m = BanyaNoBP(_vocab, SEED)
    m.m_mat_w_filter = xp.asarray(_d["m_mat_w_filter"], dtype=DATA_TYPE)
    if "m_mat_w_filter_adam_moment" in _d.files:
        m.m_mat_w_filter_adam_moment = xp.asarray(_d["m_mat_w_filter_adam_moment"], dtype=DATA_TYPE)
        m.m_mat_w_filter_adam_variance = xp.asarray(_d["m_mat_w_filter_adam_variance"], dtype=DATA_TYPE)
    for name in ("m_mat_w_data_axis", "m_mat_w_position", "m_mat_w_mix", "m_vec_w_filter_bias", "m_mat_w_head", "m_mat_w_head_a", "m_mat_w_head_b", "m_vec_w_head_bias", "m_mat_w_causal_mix", "m_vec_w_norm_gain", "m_mat_w_lag", "m_mat_w_gate", "m_vec_w_gate_bias", "m_mat_w_q", "m_mat_w_k", "m_vec_w_goonghap_gain", "m_vec_w_goonghap_sharp", "m_mat_w_operate_axis"):
        if name not in _d.files:
            continue
        setattr(m, name, xp.asarray(_d[name], dtype=DATA_TYPE))
    for name in ("m_mat_w_data_axis", "m_vec_w_head_bias", "m_mat_w_mix", "m_mat_w_head_a", "m_mat_w_head_b", "m_mat_w_head", "m_vec_w_norm_gain", "m_mat_w_lag", "m_mat_w_causal_mix", "m_mat_w_data_axis_head", "m_mat_w_gate", "m_vec_w_gate_bias", "m_mat_w_q", "m_mat_w_k", "m_vec_w_goonghap_gain", "m_vec_w_goonghap_sharp", "m_mat_w_operate_axis"):
        if (name + "_adam_moment") in _d.files and hasattr(m, name + "_adam_moment"):
            setattr(m, name + "_adam_moment", xp.asarray(_d[name + "_adam_moment"], dtype=DATA_TYPE))
            setattr(m, name + "_adam_variance", xp.asarray(_d[name + "_adam_variance"], dtype=DATA_TYPE))
    if "t" in _d.files:
        m.t = int(_d["t"])
    return m, tok


def load_from(path):
    global SAVE_PATH
    _orig = SAVE_PATH
    SAVE_PATH = path
    try:
        return load()
    finally:
        SAVE_PATH = _orig


def save_to(model, tok, path):
    global SAVE_PATH
    _orig = SAVE_PATH
    SAVE_PATH = path
    try:
        save(model, tok, getattr(model, "t", 0))
    finally:
        SAVE_PATH = _orig
