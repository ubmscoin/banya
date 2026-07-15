# 체크포인트 내려받기

프로브가 읽는 얼린 체크포인트 3개는 용량 때문에 저장소에 없다. 아래 제노도 기록에서 받아 이 폴더(model)에 넣는다.

- DOI: https://doi.org/10.5281/zenodo.21383724
- 기록 페이지: https://zenodo.org/records/21383724

## 한 번에 받기

    bash download.sh

## 직접 받기

| 파일 | 크기 | 쓰는 곳 |
|---|---|---|
| bitok_elem2_170000_m.npz | 110MB | 논문3, 논문6 프로브 |
| cache_elem3_190000.npz | 283MB | 논문2, 논문5 프로브 |
| world_toddler2_110000_m.npz | 81MB | 논문4 프로브 |

내려받기 주소는 `https://zenodo.org/records/21383724/files/파일명?download=1` 형식이다.

## 무결성 검증

    sha256sum -c checksums.sha256

기대값:

    7114a211d9bae2eef35e2774766c4407811f9d19c95fc5877f11cfd4dfbfaf79  bitok_elem2_170000_m.npz
    ced90e7edbe7130ece919cbbedc5a5fe8297b8e1d3856ac692de200ff6d42553  cache_elem3_190000.npz
    4df6249bf5e07152735e12e74f5d6babd38b27d18312d244bc453470705a151f  world_toddler2_110000_m.npz
