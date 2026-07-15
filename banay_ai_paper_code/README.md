# 반야 연구 시리즈 재현 코드

논문 6편의 모든 실측 수치를 재생하는 코드 패키지다. 폴더와 파일의 목차는 index.html 에 있다.

## 빠른 시작

1. 요구 환경: NVIDIA GPU 와 CUDA 13.x, Python 3.12, `pip install -r requirements.txt` (cupy 는 CUDA 버전에 맞는 배포판)
2. 체크포인트: 저장소에는 대용량 npz 가 없다. 제노도(https://doi.org/10.5281/zenodo.21383724)에서 받는다. `cd model && bash download.sh` 한 줄이면 되고 자세한 안내는 model/DOWNLOAD.md 에 있다
3. 말뭉치: 저장소에 npy 가 없으므로 한 번 재생성한다.

        cd corpus_build && bash build_all.sh

4. 실측 재현: probe 폴더의 프로브를 실행한다. 각 파일 머리말에 설명과 실행 줄이 있다

        cd probe && python3 paper1_engine_probe.py

## 폴더 요약

- `banya_core.py` 공통 토대(모델, 순전파, 정확한 전치 학습, 토크나이저), `paper1~6_*.py` 논문별 메커니즘
- `probe/` 실측 프로브 11종, `corpus_build/` 말뭉치 자가 제작(MANUAL.md 참고), `core/` 음절 원자 토크나이저
- `model/` 체크포인트 자리와 평가 자료, `banya_world_data/` 말뭉치 자리와 묶음 어휘 사전(bundle_cache.json)
- `smoke/` 최소 동작 확인, `data_prep/` 제4편 월드 사다리 자료

## 참고

- 체크포인트 로드 시 "vocab_size 2724 != 캐시 사전 10724" 경고는 음절 전용 체크포인트를 묶음 사전과 함께 열 때 나오는 정상 안내다
- 원저작물이 있는 말뭉치(동화, 사전)는 포함하지 않으며, 필요하면 corpus_build/encode_text.py 로 각자 텍스트를 인코딩해 쓴다

## 저자와 라이선스

- 한혁진 (Hyukjin Han) · https://orcid.org/0009-0007-4132-1707 · bokkamsun@gmail.com
- 코드는 Apache License 2.0 (LICENSE), 문서와 데이터는 Creative Commons Attribution 4.0 International
