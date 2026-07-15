#!/bin/bash
# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
# Downloads the three checkpoints from Zenodo into this folder and verifies their integrity
# Run  bash download.sh
# 제노도에서 체크포인트 3개를 이 폴더로 내려받고 무결성을 검증한다
# 실행  bash download.sh
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
sha256sum -c checksums.sha256 && echo "무결성 검증 통과. 프로브 실행 준비 완료"
