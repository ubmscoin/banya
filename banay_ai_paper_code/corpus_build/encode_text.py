# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""encode_text.py bakes any Korean text file into a syllable-atom corpus npy.
Serves two purposes together: regenerating the elementary corpus and building one's own corpus from external text.
Run  python3 encode_text.py elem.txt ../banya_world_data/elem.npy

encode_text.py 아무 한국어 텍스트 파일을 음절 원자 말뭉치 npy 로 굽는다.
초딩 말뭉치의 재생성과 외부 텍스트로 자기 말뭉치를 만드는 두 용도에 같이 쓴다.
실행  python3 encode_text.py elem.txt ../banya_world_data/elem.npy"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "core"))
import banya_atoms as ba


def main():
    if len(sys.argv) != 3:
        print("실행  python3 encode_text.py 입력텍스트.txt 출력말뭉치.npy", flush=True)
        return
    _text = open(sys.argv[1], encoding="utf-8").read()
    _ids = np.asarray(ba.AtomTokenizer().encode(_text), dtype=np.int32)
    np.save(sys.argv[2], _ids)
    print(f"저장 {sys.argv[2]} · 토큰 {len(_ids):,}개", flush=True)


if __name__ == "__main__":
    main()
