#!/bin/bash
# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
# Downloads the checkpoints from Zenodo into this folder and verifies their integrity
# Run  bash download.sh          (three core checkpoints)
# Run  bash download.sh bp       (adds the optional standard baseline checkpoint, 1.25GB)
# 제노도에서 체크포인트를 이 폴더로 내려받고 무결성을 검증한다
# 실행  bash download.sh         (핵심 체크포인트 3개)
# 실행  bash download.sh bp      (선택인 표준 기준선 체크포인트 1.25GB 추가)
set -e
cd "$(dirname "$0")"
BASE="https://zenodo.org/records/21383724/files"
for f in bitok_elem2_170000_m.npz cache_elem3_190000.npz world_toddler2_110000_m.npz; do
    if [ -f "$f" ]; then
        echo "이미 있음: $f"
    else
        echo "내려받는 중: $f"
        wget -q --show-progress "$BASE/$f?download=1" -O "$f"
    fi
done
sha256sum -c checksums.sha256 && echo "핵심 3개 무결성 검증 통과. 프로브 실행 준비 완료"
if [ "$1" = "bp" ]; then
    f=banya_bp_pytorch.pt
    if [ -f "$f" ]; then
        echo "이미 있음: $f"
    else
        echo "내려받는 중: $f (1.25GB, 제3편 표 6-1 표준 기준선용)"
        wget -q --show-progress "$BASE/$f?download=1" -O "$f"
    fi
    sha256sum -c checksums_bp.sha256 && echo "표준 기준선 체크포인트 검증 통과"
fi
