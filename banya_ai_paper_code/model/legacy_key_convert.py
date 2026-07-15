# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""One-off legacy key conversion script. Converts an old-key npz into the key npz that banya_core.load reads.

Usage
  python3 legacy_key_convert.py input.npz output.npz
  python3 legacy_key_convert.py input_folder output_folder

레거시 키 변환 일회성 스크립트. 옛 키 npz 를 banya_core.load 가 읽는키 npz 로  바꾼다.


사용
  python3 legacy_key_convert.py 입력.npz 출력.npz
  python3 legacy_key_convert.py 입력폴더 출력폴더
"""
import os
import sys
import numpy as np

WEIGHT_KEY_MAP = {
    "Ef": "m_mat_w_data_axis",
    "Eop": "m_mat_w_operate_axis",
    "Ep": "m_mat_w_position",
    "bh": "m_vec_w_head_bias",
    "Wp": "m_mat_w_filter",
    "Bpf": "m_vec_w_filter_bias",
    "Wha": "m_mat_w_head_a",
    "Whb": "m_mat_w_head_b",
    "Wh": "m_mat_w_head",
    "Gn": "m_vec_w_norm_gain",
    "Wlag": "m_mat_w_lag",
    "Wg": "m_mat_w_gate",
    "bg": "m_vec_w_gate_bias",
    "Wq": "m_mat_w_q",
    "Wk": "m_mat_w_k",
    "Gq": "m_vec_w_goonghap_gain",
    "beta": "m_vec_w_goonghap_sharp",
    "Wm": "m_mat_w_mix",
    "Cmix": "m_mat_w_causal_mix",
    "itos": "m_id_to_string",
}
PASS_THROUGH = {"step", "t"}


def p_new_key(key):
    if key in PASS_THROUGH:
        return key
    if key in WEIGHT_KEY_MAP:
        return WEIGHT_KEY_MAP[key]
    if key[0] in ("m", "v") and key[1:] in WEIGHT_KEY_MAP:
        _base = WEIGHT_KEY_MAP[key[1:]]
        _suffix = "_adam_moment" if key[0] == "m" else "_adam_variance"
        return _base + _suffix
    return None


def convert_one(src, dst, outMsg):
    _d = np.load(src, allow_pickle=True)
    _out = {}
    for key in _d.files:
        _nk = p_new_key(key)
        if _nk is None:
            outMsg[0] = f"알 수 없는 옛 키 {key} (파일 {src})"
            return False
        _out[_nk] = _d[key]
    _tmp = dst + ".tmp"
    with open(_tmp, "wb") as f:
        np.savez(f, **_out)
    os.replace(_tmp, dst)
    print(f"  변환 {os.path.basename(src)}: 키 {len(_d.files)}개 -> {os.path.basename(dst)}", flush=True)
    return True


def convert(src, dst):
    _msg = [""]
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for nm in sorted(os.listdir(src)):
            if not nm.endswith(".npz"):
                continue
            _sp = os.path.join(src, nm)
            _dp = os.path.join(dst, nm)
            if not convert_one(_sp, _dp, _msg):
                sys.exit("[!] " + _msg[0])
    else:
        if not convert_one(src, dst, _msg):
            sys.exit("[!] " + _msg[0])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("사용: python3 legacy_key_convert.py 입력.npz 출력.npz  또는  입력폴더 출력폴더")
    convert(sys.argv[1], sys.argv[2])
