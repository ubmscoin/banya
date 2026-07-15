#!/bin/bash
# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
# Regenerates all 23 self-built corpora required by the measurement probes into 코드/banya_world_data
# Run  bash build_all.sh
# 실측 프로브가 요구하는 자작 말뭉치 23종을 코드/banya_world_data 로 전부 재생성한다
# 실행  bash build_all.sh
set -e
cd "$(dirname "$0")"
export BANYA_DATA_DIR="$(cd .. && pwd)/banya_world_data"
mkdir -p "$BANYA_DATA_DIR"
echo "출력 폴더: $BANYA_DATA_DIR"
for g in life space sense sense_space sense_mimic baby baby_logic baby_learn toddler toddler_logic toddler_learn toddler_exp toddler_dialog toddler_state toddler_emotion toddler2 elem_knowledge elem_inquiry elem_logic elem_dialog elem_subject; do
    echo "── ${g} ──"
    python3 "prep_${g}_corpus.py"
done
echo "── elem ──"
python3 encode_text.py elem.txt "$BANYA_DATA_DIR/elem.npy"
echo "완료. toddler2_link 는 prep_toddler2_corpus.py 가 함께 굽는다."
