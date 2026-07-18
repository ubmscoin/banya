# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Standard transformer baseline trainer, the common contrast model of the series. First used in Paper 3, Table 6-1.
A standard backpropagation transformer (softmax attention, autograd, AdamW, 104M parameters) is trained on the
same world development curriculum, the same 170,000 steps, and the same batch 32 as the published bi-token model.
The architectures differ, so this is not an isomorphic comparison but the standard yardstick on the same data.
The per-step time printed in the training log is the speed row of Table 6-1, and the finished checkpoint
model/banya_bp_pytorch.pt is what probe/paper3_baseline_probe.py scores for the cross entropy row.
The three tale corpora are not bundled for copyright reasons; missing corpora are skipped automatically and the
mixture is renormalized, so a bundle-only retraining differs slightly from the published run at the last stages.
Requires an NVIDIA GPU and torch. Run from the package root:  python3 banya_bp_pytorch.py

표준 트랜스포머 기준선 학습기. 시리즈 공통 대조 모델이며 제3편 표 6-1에서 처음 사용한다. 표 6-1의 학습 쪽을 재생한다.
표준 역전파 트랜스포머(softmax 어텐션, autograd, AdamW, 파라미터 104M)를 발행 바이토큰 모델과 같은
월드 발달 커리큘럼, 같은 170,000스텝, 같은 배치 32로 학습한다. 구조가 다르므로 동형 비교가 아니라
같은 데이터 위의 표준 눈금이다.
학습 로그에 찍히는 스텝당 시간이 표 6-1의 속도 줄이고, 완성 체크포인트 model/banya_bp_pytorch.pt 를
probe/paper3_baseline_probe.py 가 채점하는 것이 교차엔트로피 줄이다.
동화 말뭉치 3종은 저작권 때문에 동봉하지 않았다. 없는 말뭉치는 자동으로 빠지고 혼합비가 재정규화되므로,
동봉 자료만으로 재학습하면 마지막 단계들이 발행 실행과 약간 다르다.
NVIDIA GPU 와 torch 가 필요하다. 패키지 루트에서 실행한다.  python3 banya_bp_pytorch.py"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
import banya_atoms as ba

H = 1024
BLOCK = 128
N_BLOCK = 8
N_HEADS = 8
FFN_MULT = 4
LOWRANK = True
HEAD_R = 128
BATCH = 32
SEED = 0
LR = 0.0004
WARMUP = 1000
WEIGHT_DECAY = 0.1
SAVE_EVERY = 3000
RESUME = False
SAVE_PATH = "model/banya_bp_pytorch.pt"

WARMUP_STEPS = 10000
WARMUP_SPLIT = 5000
TOTAL_STEPS = 170000
WARMUP_MIX0 = {"life": 0.50, "sense": 0.20, "space": 0.15, "sense_space": 0.10, "sense_mimic": 0.05}
WARMUP_MIX = {"life": 0.20, "space": 0.20, "sense": 0.20, "sense_space": 0.25, "sense_mimic": 0.15}
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
             "elem": 0.04, "elem_knowledge": 0.06, "elem_inquiry": 0.06, "elem_logic": 0.08,
             "elem_dialog": 0.14, "elem_subject": 0.12}}

if not torch.cuda.is_available():
    sys.exit("NVIDIA GPU(CUDA) 가 필요하다. 표 6-1 의 속도 비교가 GPU 실행 기준이기 때문이다")
DEVICE = "cuda"


class Block(nn.Module):
    # Role: one standard transformer block, softmax attention plus feed forward, both with residuals
    # Method: layernorm, fused qkv projection, scaled dot product attention with the causal mask, projection, then the standard 4x feed forward
    # Why: this is the unmodified standard block so that the baseline row of Table 6-1 measures the standard recipe and nothing else
    # 역할: 표준 트랜스포머 블록 하나. softmax 어텐션과 피드포워드, 둘 다 잔차
    # 방법: 레이어놈, qkv 한 번에 사영, 인과 마스크의 스케일드 닷 프로덕트 어텐션, 사영, 표준 4배 피드포워드
    # 이유: 표 6-1 기준선 줄이 표준 처방 그대로를 측정해야 하므로 아무것도 바꾸지 않은 표준 블록이어야 하기 때문이다
    def __init__(self, h, n_heads, ffn_mult):
        super().__init__()
        self.nh = n_heads
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, 3 * h)
        self.proj = nn.Linear(h, h)
        self.ff = nn.Sequential(nn.Linear(h, ffn_mult * h), nn.GELU(), nn.Linear(ffn_mult * h, h))

    def forward(self, x):
        B, T, HH = x.shape
        _h = self.ln1(x)
        q, k, v = self.qkv(_h).split(HH, dim=2)
        q = q.view(B, T, self.nh, HH // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, HH // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, HH // self.nh).transpose(1, 2)
        _a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        _a = _a.transpose(1, 2).reshape(B, T, HH)
        x = x + self.proj(_a)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    # Role: the full standard language model, embeddings, blocks, and the output head
    # Method: token plus position embeddings, a stack of standard blocks, final layernorm, and a low rank head that matches the published run
    # Why: the published baseline used the low rank head so the parameter budget stays at 104M, comparable to the bi-token model
    # 역할: 표준 언어모델 전체. 임베딩, 블록 더미, 출력 헤드
    # 방법: 토큰 더하기 위치 임베딩, 표준 블록 더미, 마지막 레이어놈, 발행 실행과 같은 로우랭크 헤드
    # 이유: 발행 기준선이 로우랭크 헤드로 파라미터를 104M 에 맞춰 바이토큰 모델과 비슷한 예산이 되게 했기 때문이다
    def __init__(self, vocab, h, n_layers, n_heads, block, ffn_mult=4, lowrank=True, head_r=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, h)
        self.pos = nn.Embedding(block, h)
        self.blocks = nn.ModuleList([Block(h, n_heads, ffn_mult) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(h)
        if lowrank:
            self.head = nn.Sequential(nn.Linear(h, head_r, bias=False), nn.Linear(head_r, vocab, bias=True))
        else:
            self.head = nn.Linear(h, vocab, bias=True)
        self.block = block
        self.cfg = dict(H=h, N_BLOCK=n_layers, N_HEADS=n_heads, BLOCK=block, FFN_MULT=ffn_mult,
                        LOWRANK=lowrank, HEAD_R=head_r, vocab=vocab)
        self.apply(self.p_init)

    def p_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        _p = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(_p)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


# Role: opens the curriculum corpora that exist in the bundle
# Method: memory maps each npy, skips missing or damaged files and files tokenized with an old vocabulary
# Why: large corpora do not fit in memory at once, and unbundled corpora must drop out without stopping the run
# 역할: 묶음에 존재하는 커리큘럼 말뭉치를 연다
# 방법: npy 마다 memmap 으로 열고, 없거나 손상됐거나 옛 어휘로 토큰화된 파일은 건너뛴다
# 이유: 큰 말뭉치는 한 번에 메모리에 안 올라가고, 미동봉 말뭉치는 실행을 멈추지 않고 빠져야 하기 때문이다
def p_open_corpora(names, vocab):
    _avail = {}
    for nm in names:
        _path = f"banya_world_data/{nm}.npy"
        if not os.path.exists(_path):
            continue
        try:
            _arr = np.load(_path, mmap_mode="r")
        except (EOFError, ValueError):
            continue
        if len(_arr) < BLOCK + 2:
            continue
        _smp = np.asarray(_arr[:min(len(_arr), 2000000)])
        if int(_smp.max()) >= vocab:
            print(f"  [제외] {nm}: 옛 어휘(max {int(_smp.max())} 이상 {vocab}) 재토큰화 필요", flush=True)
            continue
        _avail[nm] = _arr
    return _avail


# Role: turns a ratio dictionary into names and normalized weights over the available corpora
# Method: keeps only the corpora that were opened and renormalizes the weights to sum one
# Why: the mixture must stay a probability even when some corpora are not bundled
# 역할: 비율 사전을 사용 가능한 말뭉치의 이름과 정규화 무게로 바꾼다
# 방법: 열린 말뭉치만 남기고 무게 합을 1 로 재정규화한다
# 이유: 일부 말뭉치가 미동봉이어도 혼합이 확률로 유지되어야 하기 때문이다
def p_mix_dist(wdict, avail):
    _names = [n for n in wdict if n in avail]
    _w = np.array([wdict[n] for n in _names], dtype=np.float64)
    _w /= _w.sum()
    return _names, _w


# Role: gives the mixture ratios in effect at a global step
# Method: overwrites the base ratios with every schedule entry whose step has passed, in step order
# Why: the development curriculum is defined as an overwrite table, identical to the bi-token trainer
# 역할: 전역 스텝 시점에 유효한 혼합비를 돌려준다
# 방법: 기본 비율 위에 지난 전환표 항목들을 스텝순으로 덮어쓴다
# 이유: 발달 커리큘럼이 덮어쓰기 전환표로 정의되어 있고 바이토큰 학습기와 동일해야 하기 때문이다
def p_mix_at(gstep):
    _wd = dict(MIX_WEIGHTS)
    for s in sorted(MIX_SCHED):
        if gstep >= s:
            _wd.update(MIX_SCHED[s])
    return _wd


# Role: samples one training batch from the corpus mixture
# Method: picks a corpus per sequence by the mixture weights, then a random window inside that corpus
# Why: per-sequence corpus sampling is the same order the bi-token trainer uses, so the same seed sees the same curriculum stream shape
# 역할: 말뭉치 혼합에서 학습 배치 하나를 뽑는다
# 방법: 시퀀스마다 혼합 무게로 말뭉치를 고르고 그 말뭉치 안에서 무작위 창을 뽑는다
# 이유: 시퀀스별 말뭉치 추첨이 바이토큰 학습기와 같은 순서라 같은 시드면 같은 모양의 커리큘럼 흐름을 보기 때문이다
def p_get_mix_batch(avail, names, wts, rng):
    _pick = rng.choice(len(names), size=BATCH, p=wts)
    _xs = []
    _ys = []
    for c in _pick:
        _arr = avail[names[c]]
        _i = rng.randint(0, len(_arr) - BLOCK - 1)
        _seg = np.asarray(_arr[_i:_i + BLOCK + 1])
        _xs.append(_seg[:BLOCK])
        _ys.append(_seg[1:BLOCK + 1])
    _X = np.stack(_xs)
    _Y = np.stack(_ys)
    return torch.from_numpy(_X).long().to(DEVICE), torch.from_numpy(_Y).long().to(DEVICE)


# Role: saves the checkpoint atomically
# Method: writes to a temporary file and replaces the target in one step
# Why: a crash during the write must not leave a half checkpoint behind
# 역할: 체크포인트를 원자적으로 저장한다
# 방법: 임시 파일에 쓰고 한 번에 바꿔치기한다
# 이유: 쓰는 도중 중단되어도 반쪽 체크포인트가 남으면 안 되기 때문이다
def p_save(m, opt=None, step=0, rng=None):
    _tmp = SAVE_PATH + ".tmp"
    torch.save({"model": m.state_dict(),
                "opt": opt.state_dict() if opt is not None else None,
                "step": step,
                "rng": rng.get_state() if rng is not None else None,
                "cfg": m.cfg}, _tmp)
    os.replace(_tmp, SAVE_PATH)


# Role: loads the finished checkpoint for measurement
# Method: rebuilds the model from the stored configuration and loads the weights in evaluation mode
# Why: the probe scores the checkpoint without retraining
# 역할: 완성 체크포인트를 측정용으로 불러온다
# 방법: 저장된 설정으로 모델을 다시 만들고 가중치를 평가 모드로 올린다
# 이유: 프로브가 재학습 없이 체크포인트를 채점하기 때문이다
def load():
    _d = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    _c = _d["cfg"]
    _m = TransformerLM(_c["vocab"], _c["H"], _c["N_BLOCK"], _c["N_HEADS"], _c["BLOCK"], _c["FFN_MULT"],
                       _c.get("LOWRANK", True), _c.get("HEAD_R", 128)).to(DEVICE)
    _m.load_state_dict(_d["model"])
    _m.eval()
    return _m


# Role: trains the standard baseline on the world development curriculum
# Method: two warmup stages then the overwrite schedule, AdamW with learning rate warmup and gradient clipping, periodic atomic saves, per-step time in the log
# Why: same curriculum, same steps, same batch as the published bi-token model, so the only moving part of Table 6-1 is the architecture
# 역할: 표준 기준선을 월드 발달 커리큘럼으로 학습한다
# 방법: 워밍업 2단 뒤 덮어쓰기 전환표, 학습률 워밍업과 기울기 자르기의 AdamW, 주기적 원자 저장, 로그에 스텝당 시간
# 이유: 발행 바이토큰 모델과 같은 커리큘럼, 같은 스텝, 같은 배치라야 표 6-1 에서 움직이는 것이 구조뿐이기 때문이다
def train():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _tok = ba.AtomTokenizer()
    _rng = np.random.RandomState(SEED)
    _done = 0
    if RESUME and os.path.exists(SAVE_PATH):
        _ck = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        _c = _ck["cfg"]
        _m = TransformerLM(_c["vocab"], _c["H"], _c["N_BLOCK"], _c["N_HEADS"], _c["BLOCK"], _c["FFN_MULT"],
                           _c.get("LOWRANK", True), _c.get("HEAD_R", 128)).to(DEVICE)
        _m.load_state_dict(_ck["model"])
        _opt = torch.optim.AdamW(_m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
        if _ck.get("opt"):
            _opt.load_state_dict(_ck["opt"])
        _done = _ck.get("step", 0)
        if _ck.get("rng") is not None:
            _rng.set_state(_ck["rng"])
        print(f"  [이어학습] {SAVE_PATH} 누적 step {_done}부터", flush=True)
    else:
        _m = TransformerLM(_tok.vocab, H, N_BLOCK, N_HEADS, BLOCK, FFN_MULT, LOWRANK, HEAD_R).to(DEVICE)
        _opt = torch.optim.AdamW(_m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    _nparam = sum(p.numel() for p in _m.parameters())
    print(f"[표준 기준선 banya_bp_pytorch] H {H} BLOCK {BLOCK} 층 {N_BLOCK} 헤드 {N_HEADS} 파라미터 {_nparam / 1e6:.1f}M vocab {_tok.vocab}", flush=True)
    print(f"  총 {TOTAL_STEPS:,} step 배치 {BATCH} lr {LR} 누적 {_done:,}부터", flush=True)
    _all_names = set(list(MIX_WEIGHTS) + list(WARMUP_MIX0) + list(WARMUP_MIX) + [n for kv in MIX_SCHED.values() for n in kv])
    _avail = p_open_corpora(_all_names, _tok.vocab)
    if not _avail:
        sys.exit("banya_world_data 에 말뭉치가 없다. 패키지 준비 절차를 확인하라")
    _missing = sorted(_all_names - set(_avail))
    if _missing:
        print("  [미동봉 제외] " + " ".join(_missing) + " (혼합비 재정규화)", flush=True)
    _warm_names0, _warm_w0 = p_mix_dist(WARMUP_MIX0, _avail)
    _warm_names, _warm_w = p_mix_dist(WARMUP_MIX, _avail)
    _mix_names, _mix_w = p_mix_dist(p_mix_at(_done), _avail)
    _mix_next = min([s for s in MIX_SCHED if s > _done], default=None)
    _m.train()
    _t0 = time.time()
    for gstep in range(_done, TOTAL_STEPS):
        if _mix_next is not None and gstep >= _mix_next:
            _mix_names, _mix_w = p_mix_dist(p_mix_at(gstep), _avail)
            _mix_next = min([s for s in MIX_SCHED if s > gstep], default=None)
            print(f"  [혼합 전환] {gstep:,} step", flush=True)
        if gstep < WARMUP_STEPS:
            if gstep < WARMUP_SPLIT:
                _names, _wts = _warm_names0, _warm_w0
            else:
                _names, _wts = _warm_names, _warm_w
        else:
            _names, _wts = _mix_names, _mix_w
        _X, _Y = p_get_mix_batch(_avail, _names, _wts, _rng)
        _lr = LR * min(1.0, (gstep + 1) / WARMUP)
        for g in _opt.param_groups:
            g["lr"] = _lr
        _logits = _m(_X)
        _loss = F.cross_entropy(_logits.reshape(-1, _tok.vocab), _Y.reshape(-1))
        _opt.zero_grad(set_to_none=True)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_m.parameters(), 1.0)
        _opt.step()
        if (gstep + 1) % 10 == 0:
            torch.cuda.synchronize()
            _dt = time.time() - _t0
            _t0 = time.time()
            _ce = _loss.item()
            print(f"  {gstep + 1:,} / {TOTAL_STEPS:,}  ce {_ce:.4f}  ppl {math.exp(min(_ce, 20)):.2f}  {_dt / 10 * 1000:.1f} ms/step", flush=True)
        if SAVE_EVERY and (gstep + 1) % SAVE_EVERY == 0:
            p_save(_m, _opt, gstep + 1, _rng)
    p_save(_m, _opt, TOTAL_STEPS, _rng)
    print(f"학습 완료 {TOTAL_STEPS:,} step 저장 {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    train()
