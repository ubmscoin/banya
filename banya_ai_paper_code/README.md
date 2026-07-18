<!--
한혁진 (Hyukjin Han)
https://orcid.org/0009-0007-4132-1707
bokkamsun@gmail.com
SPDX-License-Identifier: CC-BY-4.0
-->

# Banya Research Reproduction Package

Hyukjin Han (bokkamsun@gmail.com) · [ORCID 0009-0007-4132-1707](https://orcid.org/0009-0007-4132-1707) · Code Apache-2.0 · Documents and data CC BY 4.0

## Usage

The three steps below reproduce every measurement in the seven papers. The requirements are an NVIDIA GPU (CUDA) and Python 3.12; install the Python packages with the single line below. For cupy, choose the build that matches your CUDA version (for example, cupy-cuda13x for CUDA 13).

```
pip install -r requirements.txt
```

### 1. Download the models

The three frozen checkpoints under measurement (471 MB in total) live on Zenodo rather than in this repository, because of their size:
[doi.org/10.5281/zenodo.21383724](https://doi.org/10.5281/zenodo.21383724).
The single line below downloads the three files into the model folder and verifies their integrity (sha256) automatically.

```
cd model && bash download.sh
```

To download by hand, fetch bitok_elem2_170000_m.npz, cache_elem3_190000.npz, and world_toddler2_110000_m.npz from the record page above, place them in the model folder, and verify with `sha256sum -c checksums.sha256`. The optional standard-baseline checkpoint banya_bp_pytorch.pt (1.25 GB), needed only for the standard row of Paper 3, Table 6-1, is fetched with `bash download.sh bp`. Details are in [model/DOWNLOAD.md](model/DOWNLOAD.md).

### 2. Build the world data (corpora)

The corpora are built by generators instead of being distributed as raw text. The single line below builds all 23 corpora into the banya_world_data folder.

```
cd corpus_build && bash build_all.sh
```

To build a corpus from your own text, use `python3 encode_text.py input.txt output.npy`. The build order and rationale are in [corpus_build/MANUAL.md](corpus_build/MANUAL.md).

### 3. Reproduce the measurements

The thirteen probes in the probe folder reproduce the measured figures of Papers 1 through 7. The header of each file states what it measures and carries its run line. For example, the verification of Paper 1 runs as follows.

```
cd probe && python3 paper1_engine_probe.py
```

The table of contents below reflects the completed state of the folder, with the models downloaded and the corpora built.

## Folders and files

- [LICENSE](LICENSE) — Full code license text (Apache License 2.0)
- [NOTICE](NOTICE) — Copyright holder and dual-license notice
- [README.md](README.md) — Quick-start guide: requirements, preparing checkpoints and corpora, and running the probes
- [banya_bp_pytorch.py](banya_bp_pytorch.py) — Standard transformer baseline trainer, the common contrast model of the series. Same world curriculum, steps, and batch as the published bi-token model (Paper 3, Table 6-1)
- [banya_core.py](banya_core.py) — Common foundation. The core module gathering the model class, forward pass, exact-adjoint training, cache tokenizer, and checkpoint save and load
- **[banya_world_data/](banya_world_data/)** — Home of the corpora and the vocabulary dictionary. Corpora are rebuilt with corpus_build
  - [baby.npy](banya_world_data/baby.npy) — baby corpus
  - [baby.txt](banya_world_data/baby.txt) — Generated text of the baby corpus
  - [baby_learn.npy](banya_world_data/baby_learn.npy) — baby_learn corpus
  - [baby_learn.txt](banya_world_data/baby_learn.txt) — Generated text of the baby_learn corpus
  - [baby_logic.npy](banya_world_data/baby_logic.npy) — baby_logic corpus
  - [baby_logic.txt](banya_world_data/baby_logic.txt) — Generated text of the baby_logic corpus
  - [bundle_cache.json](banya_world_data/bundle_cache.json) — Bundle vocabulary dictionary. It defines the checkpoint vocabulary; use this file as is instead of rebuilding it
  - [elem.npy](banya_world_data/elem.npy) — elem corpus
  - [elem_dialog.npy](banya_world_data/elem_dialog.npy) — elem_dialog corpus
  - [elem_dialog.txt](banya_world_data/elem_dialog.txt) — Generated text of the elem_dialog corpus
  - [elem_inquiry.npy](banya_world_data/elem_inquiry.npy) — elem_inquiry corpus
  - [elem_inquiry.txt](banya_world_data/elem_inquiry.txt) — Generated text of the elem_inquiry corpus
  - [elem_knowledge.npy](banya_world_data/elem_knowledge.npy) — elem_knowledge corpus
  - [elem_knowledge.txt](banya_world_data/elem_knowledge.txt) — Generated text of the elem_knowledge corpus
  - [elem_logic.npy](banya_world_data/elem_logic.npy) — elem_logic corpus
  - [elem_logic.txt](banya_world_data/elem_logic.txt) — Generated text of the elem_logic corpus
  - [elem_subject.npy](banya_world_data/elem_subject.npy) — elem_subject corpus
  - [elem_subject.txt](banya_world_data/elem_subject.txt) — Generated text of the elem_subject corpus
  - [life.npy](banya_world_data/life.npy) — life corpus
  - [life.txt](banya_world_data/life.txt) — Generated text of the life corpus
  - [sense.npy](banya_world_data/sense.npy) — sense corpus
  - [sense.txt](banya_world_data/sense.txt) — Generated text of the sense corpus
  - [sense_mimic.npy](banya_world_data/sense_mimic.npy) — sense_mimic corpus
  - [sense_mimic.txt](banya_world_data/sense_mimic.txt) — Generated text of the sense_mimic corpus
  - [sense_space.npy](banya_world_data/sense_space.npy) — sense_space corpus
  - [sense_space.txt](banya_world_data/sense_space.txt) — Generated text of the sense_space corpus
  - [space.npy](banya_world_data/space.npy) — space corpus
  - [space.txt](banya_world_data/space.txt) — Generated text of the space corpus
  - [toddler.npy](banya_world_data/toddler.npy) — toddler corpus
  - [toddler.txt](banya_world_data/toddler.txt) — Generated text of the toddler corpus
  - [toddler2.npy](banya_world_data/toddler2.npy) — toddler2 corpus
  - [toddler2.txt](banya_world_data/toddler2.txt) — Generated text of the toddler2 corpus
  - [toddler2_link.npy](banya_world_data/toddler2_link.npy) — toddler2_link corpus
  - [toddler2_link.txt](banya_world_data/toddler2_link.txt) — Generated text of the toddler2_link corpus
  - [toddler_dialog.npy](banya_world_data/toddler_dialog.npy) — toddler_dialog corpus
  - [toddler_dialog.txt](banya_world_data/toddler_dialog.txt) — Generated text of the toddler_dialog corpus
  - [toddler_emotion.npy](banya_world_data/toddler_emotion.npy) — toddler_emotion corpus
  - [toddler_emotion.txt](banya_world_data/toddler_emotion.txt) — Generated text of the toddler_emotion corpus
  - [toddler_exp.npy](banya_world_data/toddler_exp.npy) — toddler_exp corpus
  - [toddler_exp.txt](banya_world_data/toddler_exp.txt) — Generated text of the toddler_exp corpus
  - [toddler_learn.npy](banya_world_data/toddler_learn.npy) — toddler_learn corpus
  - [toddler_learn.txt](banya_world_data/toddler_learn.txt) — Generated text of the toddler_learn corpus
  - [toddler_logic.npy](banya_world_data/toddler_logic.npy) — toddler_logic corpus
  - [toddler_logic.txt](banya_world_data/toddler_logic.txt) — Generated text of the toddler_logic corpus
  - [toddler_state.npy](banya_world_data/toddler_state.npy) — toddler_state corpus
  - [toddler_state.txt](banya_world_data/toddler_state.txt) — Generated text of the toddler_state corpus
- **[core/](core/)** — Syllable-atom tokenizer layer
  - [banya_atoms.py](core/banya_atoms.py) — Syllable-atom tokenizer. Encodes and decodes Korean at the syllable level
  - [banya_syllables.txt](core/banya_syllables.txt) — Source list of the syllable atoms
- **[corpus_build/](corpus_build/)** — Self-build of the corpora. Generators rebuild every corpus without distributing raw text. See [MANUAL.md](corpus_build/MANUAL.md)
  - [MANUAL.md](corpus_build/MANUAL.md) — Build manual: run order and how to make corpora from external text
  - [assemble_toddler2.py](corpus_build/assemble_toddler2.py) — Assembly helper for the late-toddler corpus
  - [baby_expr.py](corpus_build/baby_expr.py) — Expression data module read by the generators
  - [baby_mimic.py](corpus_build/baby_mimic.py) — Expression data module read by the generators
  - [baby_world_expr.py](corpus_build/baby_world_expr.py) — Expression data module read by the generators
  - [build_all.sh](corpus_build/build_all.sh) — Rebuilds every self-made corpus at once: bash build_all.sh
  - [corpus_parts.py](corpus_build/corpus_parts.py) — Shared parts for the generators: encoding and saving
  - [elem.txt](corpus_build/elem.txt) — Self-made source text of the elementary corpus. encode_text.py builds elem.npy from it
  - [encode_text.py](corpus_build/encode_text.py) — General tool that encodes any Korean text into a syllable-corpus npy
  - [prep_baby_corpus.py](corpus_build/prep_baby_corpus.py) — baby corpus generator
  - [prep_baby_learn_corpus.py](corpus_build/prep_baby_learn_corpus.py) — baby_learn corpus generator
  - [prep_baby_logic_corpus.py](corpus_build/prep_baby_logic_corpus.py) — baby_logic corpus generator
  - [prep_bundle_cache.py](corpus_build/prep_bundle_cache.py) — Reference generator showing how the bundle dictionary was built
  - [prep_elem_corpus.py](corpus_build/prep_elem_corpus.py) — Old elementary-stage generator kept for reference only; it reads a roster file that no longer exists. elem.npy is built from elem.txt
  - [prep_elem_dialog_corpus.py](corpus_build/prep_elem_dialog_corpus.py) — elem_dialog corpus generator
  - [prep_elem_inquiry_corpus.py](corpus_build/prep_elem_inquiry_corpus.py) — elem_inquiry corpus generator
  - [prep_elem_knowledge_corpus.py](corpus_build/prep_elem_knowledge_corpus.py) — elem_knowledge corpus generator
  - [prep_elem_logic_corpus.py](corpus_build/prep_elem_logic_corpus.py) — elem_logic corpus generator
  - [prep_elem_subject_corpus.py](corpus_build/prep_elem_subject_corpus.py) — elem_subject corpus generator
  - [prep_life_corpus.py](corpus_build/prep_life_corpus.py) — life corpus generator
  - [prep_sense_corpus.py](corpus_build/prep_sense_corpus.py) — sense corpus generator
  - [prep_sense_mimic_corpus.py](corpus_build/prep_sense_mimic_corpus.py) — sense_mimic corpus generator
  - [prep_sense_space_corpus.py](corpus_build/prep_sense_space_corpus.py) — sense_space corpus generator
  - [prep_space_corpus.py](corpus_build/prep_space_corpus.py) — space corpus generator
  - [prep_toddler2_corpus.py](corpus_build/prep_toddler2_corpus.py) — toddler2 corpus generator
  - [prep_toddler_corpus.py](corpus_build/prep_toddler_corpus.py) — toddler corpus generator
  - [prep_toddler_dialog_corpus.py](corpus_build/prep_toddler_dialog_corpus.py) — toddler_dialog corpus generator
  - [prep_toddler_emotion_corpus.py](corpus_build/prep_toddler_emotion_corpus.py) — toddler_emotion corpus generator
  - [prep_toddler_exp_corpus.py](corpus_build/prep_toddler_exp_corpus.py) — toddler_exp corpus generator
  - [prep_toddler_learn_corpus.py](corpus_build/prep_toddler_learn_corpus.py) — toddler_learn corpus generator
  - [prep_toddler_logic_corpus.py](corpus_build/prep_toddler_logic_corpus.py) — toddler_logic corpus generator
  - [prep_toddler_state_corpus.py](corpus_build/prep_toddler_state_corpus.py) — toddler_state corpus generator
  - [seeds.py](corpus_build/seeds.py) — Seed vocabulary tables for generation
  - [toddler2_expr.py](corpus_build/toddler2_expr.py) — Expression data module read by the generators
  - [toddler2_link_expr.py](corpus_build/toddler2_link_expr.py) — Expression data module read by the generators
  - [toddler_affect.py](corpus_build/toddler_affect.py) — Expression data module read by the generators
  - [toddler_events.py](corpus_build/toddler_events.py) — Expression data module read by the generators
  - [toddler_expr.py](corpus_build/toddler_expr.py) — Expression data module read by the generators
  - [toddler_questions.py](corpus_build/toddler_questions.py) — Expression data module read by the generators
  - [toddler_seed_expr.py](corpus_build/toddler_seed_expr.py) — Expression data module read by the generators
  - [world_ladder.py](corpus_build/world_ladder.py) — World-ladder axis definition data
- **[data_prep/](data_prep/)** — World-ladder data modules used by the Paper 4 probe
  - [baby_world_expr.py](data_prep/baby_world_expr.py) — World expression data of the baby stage
  - [world_ladder.py](data_prep/world_ladder.py) — World-ladder axis definitions
- **[model/](model/)** — Frozen checkpoints and evaluation data. The large npz files come from Zenodo (doi.org/10.5281/zenodo.21383724): bash download.sh
  - [DOWNLOAD.md](model/DOWNLOAD.md) — Checkpoint download guide: the Zenodo address and verification
  - [download.sh](model/download.sh) — Downloads the checkpoints and verifies integrity: bash download.sh (core three), bash download.sh bp (adds the optional standard baseline)
  - [checksums.sha256](model/checksums.sha256) — Expected integrity values of the three core checkpoints
  - [checksums_bp.sha256](model/checksums_bp.sha256) — Expected integrity value of the optional standard-baseline checkpoint
  - [banya_bp_pytorch.pt](model/banya_bp_pytorch.pt) — Standard-baseline checkpoint (optional, 1.25 GB). Used by the Paper 3 baseline probe for the standard row of Table 6-1
  - [bitok_elem2_170000_m.npz](model/bitok_elem2_170000_m.npz) — Bi-token model checkpoint of the elementary stage. Used by the Paper 3 and Paper 6 probes
  - [cache_elem3_190000.npz](model/cache_elem3_190000.npz) — Cache-dictionary model checkpoint of the elementary stage. Used by the Paper 2 and Paper 5 probes
  - [eval_sets.json](model/eval_sets.json) — Evaluation sentence sets: the holdout snippets
  - [legacy_key_convert.py](model/legacy_key_convert.py) — One-off tool that converted old-key checkpoints to the new keys (m_ prefix). Reference only
  - [stream_train.npy](model/stream_train.npy) — Training-stream sample for collecting rumination seeds
  - [world_toddler2_110000_m.npz](model/world_toddler2_110000_m.npz) — World-model checkpoint of the late toddler stage. Used by the Paper 4 probe
- [paper1_engine.py](paper1_engine.py) — Paper 1 engine. Wraps train_step, which learns one step through the exact-adjoint credit chain
- [paper2_rumination.py](paper2_rumination.py) — Paper 2 rumination. Firing-group collection, the pull rule, and the other rumination mechanisms
- [paper3_bitoken.py](paper3_bitoken.py) — Paper 3 bi-token. Orthogonal-token mechanisms such as the operation-plane on and off switch
- [paper4_world_dev.py](paper4_world_dev.py) — Paper 4 world-first development. Developmental-curriculum stage definitions
- [paper5_cache_token.py](paper5_cache_token.py) — Paper 5 cache tokenization. Low-rank head cost accounting and related mechanisms
- [paper6_metacog.py](paper6_metacog.py) — Paper 6 metacognition. Distribution statistics and certain, vague, unknown routing
- [paper7_rotation.py](paper7_rotation.py) — Paper 7 rotation-operator gate and quaternion gate. Storage-free closed-form adjoints, run-time kernel injection without touching the engine
- **[probe/](probe/)** — Thirteen measurement probes. Each file header carries its run line; execute with python3 filename
  - [paper1_engine_probe.py](probe/paper1_engine_probe.py) — Paper 1. Verifies no-autodiff operation, adjoint correctness, and normalization scale stability
  - [paper1_small_contrast_probe.py](probe/paper1_small_contrast_probe.py) — Paper 1. Trains the engine and a standard-backpropagation reference side by side from the same initial weights and batches, measuring loss-trajectory agreement and run time
  - [paper2_rumination_probe.py](probe/paper2_rumination_probe.py) — Paper 2. Measures cohesion gain, holdout preservation, and equilibrium convergence of rumination
  - [paper2_rumination_qa_probe.py](probe/paper2_rumination_qa_probe.py) — Paper 2. Measures whether downstream 2AFC discrimination survives rumination, including a stress run that force-feeds the answer candidates as seeds
  - [paper3_bitoken_probe.py](probe/paper3_bitoken_probe.py) — Paper 3. Measures operator direction consistency, orthogonality, prediction contribution, and mixing-head differentiation
  - [paper3_baseline_probe.py](probe/paper3_baseline_probe.py) — Paper 3. Table 6-1 standard baseline: cross entropy on identical windows and per-step training time
  - [paper3_composition_probe.py](probe/paper3_composition_probe.py) — Paper 3. Measures assembly discrimination of unseen stem-ending combinations against a bigram baseline
  - [paper4_world_dev_probe.py](probe/paper4_world_dev_probe.py) — Paper 4. Measures axis-structure formation in the developmental-stage checkpoint
  - [paper5_cache_token_probe.py](probe/paper5_cache_token_probe.py) — Paper 5. Measures the two-layer vocabulary, the low-rank head cost, and the operator structure of the bundle layer
  - [paper5_quality_contrast_probe.py](probe/paper5_quality_contrast_probe.py) — Paper 5. Measures bits per character with bundles on and off, confirming that the saving keeps quality
  - [paper6_metacog_probe.py](probe/paper6_metacog_probe.py) — Paper 6. Measures the distribution statistics of the certain, vague, and unknown question sets
  - [paper6_vague_trigger_probe.py](probe/paper6_vague_trigger_probe.py) — Paper 6. Measures the circuit that turns a vague hidden state certain through context-accompanied reinterpretation
  - [paper7_rotation_probe.py](probe/paper7_rotation_probe.py) — Paper 7. Adjoint checks, injection design proof, and the mod 3 and triangle depth-excluded discrimination ablations across gate variants including the quaternion gate
- [requirements.txt](requirements.txt) — Required Python packages. For cupy, install the build matching your CUDA version
- **[smoke/](smoke/)** — Minimal smoke check
  - [example.txt](smoke/example.txt) — Example text for the smoke check
  - [smoke_vocab.py](smoke/smoke_vocab.py) — Quick check that the vocabulary and tokenizer are healthy

---

# 반야 연구 재현 패키지 목차

한혁진 (bokkamsun@gmail.com) · [ORCID 0009-0007-4132-1707](https://orcid.org/0009-0007-4132-1707) · 코드 Apache-2.0 · 문서와 데이터 CC BY 4.0

## 사용법

아래 세 단계면 논문 일곱 편의 모든 실측이 재현된다. 요구 환경은 NVIDIA GPU(CUDA)와 Python 3.12 이며, 파이썬 패키지는 다음 한 줄로 설치한다. cupy 는 설치된 CUDA 버전에 맞는 배포판을 고른다(예: CUDA 13 은 cupy-cuda13x).

```
pip install -r requirements.txt
```

### 1. 모델 내려받기

실측의 대상인 얼린 체크포인트 3개(총 471MB)는 용량 때문에 저장소가 아니라 제노도에 있다:
[doi.org/10.5281/zenodo.21383724](https://doi.org/10.5281/zenodo.21383724).
아래 한 줄이 세 파일을 model 폴더로 내려받고 무결성(sha256)까지 자동으로 검증한다.

```
cd model && bash download.sh
```

손으로 받으려면 위 기록 페이지에서 bitok_elem2_170000_m.npz, cache_elem3_190000.npz, world_toddler2_110000_m.npz 세 파일을 받아 model 폴더에 넣고 `sha256sum -c checksums.sha256` 으로 검증한다. 제3편 표 6-1의 표준 줄에만 필요한 선택 체크포인트 banya_bp_pytorch.pt(1.25GB)는 `bash download.sh bp` 로 받는다. 자세한 안내는 [model/DOWNLOAD.md](model/DOWNLOAD.md) 에 있다.

### 2. 월드데이터(말뭉치) 생성

말뭉치는 원문을 배포하지 않고 생성기로 만든다. 아래 한 줄이 banya_world_data 폴더에 말뭉치 23종을 전부 생성한다.

```
cd corpus_build && bash build_all.sh
```

자기 텍스트로 말뭉치를 만들려면 `python3 encode_text.py 입력텍스트.txt 출력말뭉치.npy` 를 사용한다. 생성 순서와 원리는 [corpus_build/MANUAL.md](corpus_build/MANUAL.md) 에 있다.

### 3. 실측 재현

probe 폴더의 프로브 13개가 논문 1편부터 7편까지의 실측 수치를 재생한다. 각 파일의 머리말에 무엇을 측정하는지와 실행 줄이 적혀 있다. 예를 들어 제1편의 검증은 다음과 같다.

```
cd probe && python3 paper1_engine_probe.py
```

아래 목차는 모델과 말뭉치가 준비된 완성 상태의 폴더 기준이다.

## 폴더와 파일

- [LICENSE](LICENSE) — 코드 라이선스 전문 (Apache License 2.0)
- [NOTICE](NOTICE) — 저작권자와 이중 라이선스 안내
- [README.md](README.md) — 빠른 시작 안내. 요구 환경, 체크포인트와 말뭉치 준비, 프로브 실행법
- [banya_bp_pytorch.py](banya_bp_pytorch.py) — 표준 트랜스포머 기준선 학습기. 시리즈 공통 대조 모델. 발행 바이토큰 모델과 같은 월드 커리큘럼, 스텝, 배치 (제3편 표 6-1)
- [banya_core.py](banya_core.py) — 공통 토대. 모델 클래스, 순전파, 정확한 전치 학습, 캐시 토크나이저, 체크포인트 저장과 로드를 한곳에 모은 핵심 모듈
- **[banya_world_data/](banya_world_data/)** — 말뭉치와 어휘 사전 자리. 말뭉치는 corpus_build 로 재생성한다
  - [baby.npy](banya_world_data/baby.npy) — baby 말뭉치
  - [baby.txt](banya_world_data/baby.txt) — baby 말뭉치의 생성 텍스트
  - [baby_learn.npy](banya_world_data/baby_learn.npy) — baby_learn 말뭉치
  - [baby_learn.txt](banya_world_data/baby_learn.txt) — baby_learn 말뭉치의 생성 텍스트
  - [baby_logic.npy](banya_world_data/baby_logic.npy) — baby_logic 말뭉치
  - [baby_logic.txt](banya_world_data/baby_logic.txt) — baby_logic 말뭉치의 생성 텍스트
  - [bundle_cache.json](banya_world_data/bundle_cache.json) — 묶음 어휘 사전. 체크포인트의 어휘 정의라 새로 만들지 말고 이것을 사용한다
  - [elem.npy](banya_world_data/elem.npy) — elem 말뭉치
  - [elem_dialog.npy](banya_world_data/elem_dialog.npy) — elem_dialog 말뭉치
  - [elem_dialog.txt](banya_world_data/elem_dialog.txt) — elem_dialog 말뭉치의 생성 텍스트
  - [elem_inquiry.npy](banya_world_data/elem_inquiry.npy) — elem_inquiry 말뭉치
  - [elem_inquiry.txt](banya_world_data/elem_inquiry.txt) — elem_inquiry 말뭉치의 생성 텍스트
  - [elem_knowledge.npy](banya_world_data/elem_knowledge.npy) — elem_knowledge 말뭉치
  - [elem_knowledge.txt](banya_world_data/elem_knowledge.txt) — elem_knowledge 말뭉치의 생성 텍스트
  - [elem_logic.npy](banya_world_data/elem_logic.npy) — elem_logic 말뭉치
  - [elem_logic.txt](banya_world_data/elem_logic.txt) — elem_logic 말뭉치의 생성 텍스트
  - [elem_subject.npy](banya_world_data/elem_subject.npy) — elem_subject 말뭉치
  - [elem_subject.txt](banya_world_data/elem_subject.txt) — elem_subject 말뭉치의 생성 텍스트
  - [life.npy](banya_world_data/life.npy) — life 말뭉치
  - [life.txt](banya_world_data/life.txt) — life 말뭉치의 생성 텍스트
  - [sense.npy](banya_world_data/sense.npy) — sense 말뭉치
  - [sense.txt](banya_world_data/sense.txt) — sense 말뭉치의 생성 텍스트
  - [sense_mimic.npy](banya_world_data/sense_mimic.npy) — sense_mimic 말뭉치
  - [sense_mimic.txt](banya_world_data/sense_mimic.txt) — sense_mimic 말뭉치의 생성 텍스트
  - [sense_space.npy](banya_world_data/sense_space.npy) — sense_space 말뭉치
  - [sense_space.txt](banya_world_data/sense_space.txt) — sense_space 말뭉치의 생성 텍스트
  - [space.npy](banya_world_data/space.npy) — space 말뭉치
  - [space.txt](banya_world_data/space.txt) — space 말뭉치의 생성 텍스트
  - [toddler.npy](banya_world_data/toddler.npy) — toddler 말뭉치
  - [toddler.txt](banya_world_data/toddler.txt) — toddler 말뭉치의 생성 텍스트
  - [toddler2.npy](banya_world_data/toddler2.npy) — toddler2 말뭉치
  - [toddler2.txt](banya_world_data/toddler2.txt) — toddler2 말뭉치의 생성 텍스트
  - [toddler2_link.npy](banya_world_data/toddler2_link.npy) — toddler2_link 말뭉치
  - [toddler2_link.txt](banya_world_data/toddler2_link.txt) — toddler2_link 말뭉치의 생성 텍스트
  - [toddler_dialog.npy](banya_world_data/toddler_dialog.npy) — toddler_dialog 말뭉치
  - [toddler_dialog.txt](banya_world_data/toddler_dialog.txt) — toddler_dialog 말뭉치의 생성 텍스트
  - [toddler_emotion.npy](banya_world_data/toddler_emotion.npy) — toddler_emotion 말뭉치
  - [toddler_emotion.txt](banya_world_data/toddler_emotion.txt) — toddler_emotion 말뭉치의 생성 텍스트
  - [toddler_exp.npy](banya_world_data/toddler_exp.npy) — toddler_exp 말뭉치
  - [toddler_exp.txt](banya_world_data/toddler_exp.txt) — toddler_exp 말뭉치의 생성 텍스트
  - [toddler_learn.npy](banya_world_data/toddler_learn.npy) — toddler_learn 말뭉치
  - [toddler_learn.txt](banya_world_data/toddler_learn.txt) — toddler_learn 말뭉치의 생성 텍스트
  - [toddler_logic.npy](banya_world_data/toddler_logic.npy) — toddler_logic 말뭉치
  - [toddler_logic.txt](banya_world_data/toddler_logic.txt) — toddler_logic 말뭉치의 생성 텍스트
  - [toddler_state.npy](banya_world_data/toddler_state.npy) — toddler_state 말뭉치
  - [toddler_state.txt](banya_world_data/toddler_state.txt) — toddler_state 말뭉치의 생성 텍스트
- **[core/](core/)** — 음절 원자 토크나이저 층
  - [banya_atoms.py](core/banya_atoms.py) — 음절 원자 토크나이저. 한국어를 음절 단위 아이디로 인코딩과 디코딩
  - [banya_syllables.txt](core/banya_syllables.txt) — 음절 원자 사전 원본
- **[corpus_build/](corpus_build/)** — 말뭉치 자가 제작. 원문 텍스트 없이 생성기로 말뭉치를 재생성한다. [MANUAL.md](corpus_build/MANUAL.md) 참고
  - [MANUAL.md](corpus_build/MANUAL.md) — 제작 매뉴얼. 실행 순서와 외부 텍스트 말뭉치 만드는 법
  - [assemble_toddler2.py](corpus_build/assemble_toddler2.py) — 유아 후반 말뭉치 조립 보조
  - [baby_expr.py](corpus_build/baby_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [baby_mimic.py](corpus_build/baby_mimic.py) — 생성기가 읽는 표현 자료 모듈
  - [baby_world_expr.py](corpus_build/baby_world_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [build_all.sh](corpus_build/build_all.sh) — 자작 말뭉치 전부를 한 번에 재생성. bash build_all.sh
  - [corpus_parts.py](corpus_build/corpus_parts.py) — 생성기 공용 부품. 인코딩과 저장
  - [elem.txt](corpus_build/elem.txt) — 초등 말뭉치의 자작 원문 텍스트. encode_text.py 로 elem.npy 를 만든다
  - [encode_text.py](corpus_build/encode_text.py) — 아무 한국어 텍스트를 음절 말뭉치 npy 로 굽는 범용 도구
  - [prep_baby_corpus.py](corpus_build/prep_baby_corpus.py) — baby 말뭉치 생성기
  - [prep_baby_learn_corpus.py](corpus_build/prep_baby_learn_corpus.py) — baby_learn 말뭉치 생성기
  - [prep_baby_logic_corpus.py](corpus_build/prep_baby_logic_corpus.py) — baby_logic 말뭉치 생성기
  - [prep_bundle_cache.py](corpus_build/prep_bundle_cache.py) — 묶음 어휘 사전이 만들어진 방법의 참고용 생성기
  - [prep_elem_corpus.py](corpus_build/prep_elem_corpus.py) — 옛 초등 생성기. 지금은 없는 명단 파일을 읽어 참고용으로만 남김. elem.npy 는 elem.txt 로 만든다
  - [prep_elem_dialog_corpus.py](corpus_build/prep_elem_dialog_corpus.py) — elem_dialog 말뭉치 생성기
  - [prep_elem_inquiry_corpus.py](corpus_build/prep_elem_inquiry_corpus.py) — elem_inquiry 말뭉치 생성기
  - [prep_elem_knowledge_corpus.py](corpus_build/prep_elem_knowledge_corpus.py) — elem_knowledge 말뭉치 생성기
  - [prep_elem_logic_corpus.py](corpus_build/prep_elem_logic_corpus.py) — elem_logic 말뭉치 생성기
  - [prep_elem_subject_corpus.py](corpus_build/prep_elem_subject_corpus.py) — elem_subject 말뭉치 생성기
  - [prep_life_corpus.py](corpus_build/prep_life_corpus.py) — life 말뭉치 생성기
  - [prep_sense_corpus.py](corpus_build/prep_sense_corpus.py) — sense 말뭉치 생성기
  - [prep_sense_mimic_corpus.py](corpus_build/prep_sense_mimic_corpus.py) — sense_mimic 말뭉치 생성기
  - [prep_sense_space_corpus.py](corpus_build/prep_sense_space_corpus.py) — sense_space 말뭉치 생성기
  - [prep_space_corpus.py](corpus_build/prep_space_corpus.py) — space 말뭉치 생성기
  - [prep_toddler2_corpus.py](corpus_build/prep_toddler2_corpus.py) — toddler2 말뭉치 생성기
  - [prep_toddler_corpus.py](corpus_build/prep_toddler_corpus.py) — toddler 말뭉치 생성기
  - [prep_toddler_dialog_corpus.py](corpus_build/prep_toddler_dialog_corpus.py) — toddler_dialog 말뭉치 생성기
  - [prep_toddler_emotion_corpus.py](corpus_build/prep_toddler_emotion_corpus.py) — toddler_emotion 말뭉치 생성기
  - [prep_toddler_exp_corpus.py](corpus_build/prep_toddler_exp_corpus.py) — toddler_exp 말뭉치 생성기
  - [prep_toddler_learn_corpus.py](corpus_build/prep_toddler_learn_corpus.py) — toddler_learn 말뭉치 생성기
  - [prep_toddler_logic_corpus.py](corpus_build/prep_toddler_logic_corpus.py) — toddler_logic 말뭉치 생성기
  - [prep_toddler_state_corpus.py](corpus_build/prep_toddler_state_corpus.py) — toddler_state 말뭉치 생성기
  - [seeds.py](corpus_build/seeds.py) — 생성 씨앗 어휘 표
  - [toddler2_expr.py](corpus_build/toddler2_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler2_link_expr.py](corpus_build/toddler2_link_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler_affect.py](corpus_build/toddler_affect.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler_events.py](corpus_build/toddler_events.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler_expr.py](corpus_build/toddler_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler_questions.py](corpus_build/toddler_questions.py) — 생성기가 읽는 표현 자료 모듈
  - [toddler_seed_expr.py](corpus_build/toddler_seed_expr.py) — 생성기가 읽는 표현 자료 모듈
  - [world_ladder.py](corpus_build/world_ladder.py) — 월드 사다리 축 정의 자료
- **[data_prep/](data_prep/)** — 논문4 프로브가 쓰는 월드 사다리 자료 모듈
  - [baby_world_expr.py](data_prep/baby_world_expr.py) — 아기 단계 월드 표현 자료
  - [world_ladder.py](data_prep/world_ladder.py) — 월드 사다리 축 정의
- **[model/](model/)** — 얼린 체크포인트와 평가 자료. 대용량 npz 는 제노도(doi.org/10.5281/zenodo.21383724)에서 받는다. bash download.sh
  - [DOWNLOAD.md](model/DOWNLOAD.md) — 체크포인트 내려받기 안내. 제노도 주소와 검증법
  - [download.sh](model/download.sh) — 체크포인트를 받고 무결성을 검증. bash download.sh (핵심 3개), bash download.sh bp (선택인 표준 기준선 추가)
  - [checksums.sha256](model/checksums.sha256) — 핵심 체크포인트 3개의 무결성 기대값
  - [checksums_bp.sha256](model/checksums_bp.sha256) — 선택인 표준 기준선 체크포인트의 무결성 기대값
  - [banya_bp_pytorch.pt](model/banya_bp_pytorch.pt) — 표준 기준선 체크포인트 (선택, 1.25GB). 논문3 기준선 프로브가 표 6-1의 표준 줄에 사용한다
  - [bitok_elem2_170000_m.npz](model/bitok_elem2_170000_m.npz) — 초등 단계 바이토큰 모델 체크포인트. 논문3 과 논문6 프로브가 사용한다
  - [cache_elem3_190000.npz](model/cache_elem3_190000.npz) — 초등 단계 캐시 사전 모델 체크포인트. 논문2 와 논문5 프로브가 사용한다
  - [eval_sets.json](model/eval_sets.json) — 평가 문장 묶음. 홀드아웃 조각들
  - [legacy_key_convert.py](model/legacy_key_convert.py) — 옛 키 이름 체크포인트를 새 키(m_ 접두)로 바꾸던 일회성 도구. 참고용
  - [stream_train.npy](model/stream_train.npy) — 되새김 씨앗 수집용 학습 스트림 표본
  - [world_toddler2_110000_m.npz](model/world_toddler2_110000_m.npz) — 월드 모델 유아 후반 단계 체크포인트. 논문4 프로브가 사용한다
- [paper1_engine.py](paper1_engine.py) — 제1편 엔진. 정확한 전치 신용 사슬로 한 스텝을 학습하는 train_step 을 묶는다
- [paper2_rumination.py](paper2_rumination.py) — 제2편 되새김. 발화군 수집과 당김 규칙 등 되새김 메커니즘
- [paper3_bitoken.py](paper3_bitoken.py) — 제3편 바이토큰. 연산면 켬끔 스위치 등 직교 토큰 메커니즘
- [paper4_world_dev.py](paper4_world_dev.py) — 제4편 월드발달. 발달 커리큘럼 단계 정의
- [paper5_cache_token.py](paper5_cache_token.py) — 제5편 캐시 토큰화. 로우랭크 헤드 비용 계산 등
- [paper6_metacog.py](paper6_metacog.py) — 제6편 메타인지. 분포 통계량과 확실 모호 모름 라우팅
- [paper7_rotation.py](paper7_rotation.py) — 제7편 회전 연산자 게이트와 쿼터니언 게이트. 무저장 닫힌형 수반, 엔진 무수정 실행 중 커널 주입
- **[probe/](probe/)** — 논문별 실측 프로브 13종. 각 파일 머리말에 실행 줄이 있고 python3 파일명 으로 실행한다
  - [paper1_engine_probe.py](probe/paper1_engine_probe.py) — 제1편. 자동미분 미사용, 전치 정합성, 정규화 스케일 안정성 검증
  - [paper1_small_contrast_probe.py](probe/paper1_small_contrast_probe.py) — 제1편. 표준 역전파 기준 구현과 같은 초기값 같은 배치로 나란히 학습해 손실 궤적 일치와 실행 시간을 측정한다
  - [paper2_rumination_probe.py](probe/paper2_rumination_probe.py) — 제2편. 되새김의 동종 밀집도 상승, 홀드아웃 보존, 평형 수렴을 측정한다
  - [paper2_rumination_qa_probe.py](probe/paper2_rumination_qa_probe.py) — 제2편. 되새김 전후 하류 2AFC 판별이 유지되는지 측정한다. 후보 강제 투입 압박 판 포함
  - [paper3_bitoken_probe.py](probe/paper3_bitoken_probe.py) — 제3편. 연산자 방향 일관성, 직교성, 예측 기여, 혼합 헤드 분화를 측정한다
  - [paper3_baseline_probe.py](probe/paper3_baseline_probe.py) — 제3편. 표 6-1 표준 기준선. 같은 창의 교차엔트로피와 스텝당 학습 시간을 측정한다
  - [paper3_composition_probe.py](probe/paper3_composition_probe.py) — 제3편. 안 본 어간 어미 조합의 조립 판별을 바이그램 기준선과 나란히 측정한다
  - [paper4_world_dev_probe.py](probe/paper4_world_dev_probe.py) — 제4편. 발달 단계 체크포인트의 축 구조 형성을 측정한다
  - [paper5_cache_token_probe.py](probe/paper5_cache_token_probe.py) — 제5편. 2층 어휘 구성, 로우랭크 헤드 비용, 묶음층 연산자 구조를 측정한다
  - [paper5_quality_contrast_probe.py](probe/paper5_quality_contrast_probe.py) — 제5편. 묶음 켬과 끔의 글자당 비트를 재 절감과 품질의 동행을 확인한다
  - [paper6_metacog_probe.py](probe/paper6_metacog_probe.py) — 제6편. 확실 모호 모름 질문 묶음의 분포 통계량을 측정한다
  - [paper6_vague_trigger_probe.py](probe/paper6_vague_trigger_probe.py) — 제6편. 모호 히든을 문맥 동반 재해석으로 확실화하는 회로를 측정한다
  - [paper7_rotation_probe.py](probe/paper7_rotation_probe.py) — 제7편. 수반 정합, 주입 설계 증명, 게이트 변형 절제의 mod 3, 삼각형 깊이 배제 판별 측정
- [requirements.txt](requirements.txt) — 필요 파이썬 패키지 목록. cupy 는 CUDA 버전에 맞는 배포판을 설치한다
- **[smoke/](smoke/)** — 최소 동작 확인
  - [example.txt](smoke/example.txt) — 검사용 예시 텍스트
  - [smoke_vocab.py](smoke/smoke_vocab.py) — 어휘와 토크나이저가 정상인지 빠르게 확인하는 검사
