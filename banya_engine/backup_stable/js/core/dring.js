// 반야프레임 d-ring: 8비트 링버퍼
// 공리 15 명제: 모든 공리 구조의 물리적 그릇
// 8비트 = 2니블. 니블0(도메인 4비트) + 니블1(연산자 3비트 + delta 1비트)

class DRing {
    // bit 0: observer     (니블0, 양자괄호, 진입점)
    // bit 1: superposition (니블0, 양자괄호, 중첩 인덱스)
    // bit 2: time         (니블0, 고전괄호, 시간축)
    // bit 3: space        (니블0, 고전괄호, 공간축)
    // bit 4: R_LOCK       (니블1, CAS Read 잠금)
    // bit 5: C_LOCK       (니블1, CAS Compare 잠금)
    // bit 6: S_LOCK       (니블1, CAS Swap 잠금)
    // bit 7: delta        (발화비트, FSM 밖의 전역 플래그)

    static BIT_OBSERVER      = 0;
    static BIT_SUPERPOSITION = 1;
    static BIT_TIME          = 2;
    static BIT_SPACE         = 3;
    static BIT_R_LOCK        = 4;
    static BIT_C_LOCK        = 5;
    static BIT_S_LOCK        = 6;
    static BIT_DELTA         = 7;

    // 니블 마스크
    static MASK_DOMAIN   = 0x0F;  // bit 0~3: 도메인 4비트
    static MASK_OPERATOR = 0xF0;  // bit 4~7: 연산자 3비트 + delta
    static MASK_CAS      = 0x70;  // bit 4~6: CAS FSM 3비트
    static MASK_LOCKS    = 0x70;  // R_LOCK + C_LOCK + S_LOCK

    constructor() {
        // 공리 15: 8비트 전체가 0으로 시작. delta=0이면 우항 무효
        this.bits = 0x00;
    }

    // 비트 읽기. 공리 2 명제: CAS Read는 현재 상태를 가져올 뿐
    getBit(pos) {
        return (this.bits >> pos) & 1;
    }

    // 비트 쓰기. CAS Swap을 통해서만 호출되어야 한다
    setBit(pos, value) {
        if (value) {
            this.bits |= (1 << pos);
        }
        else {
            this.bits &= ~(1 << pos);
        }
    }

    // 니블0 전체 읽기: 도메인 4비트 패턴
    // 공리 1: 4축 직교 = 4비트 독립
    getDomain() {
        return this.bits & DRing.MASK_DOMAIN;
    }

    // CAS FSM 3비트 읽기: bit 4,5,6
    // 공리 14: FSM 상태 = {000, 001, 011, 111}
    getCASState() {
        return (this.bits & DRing.MASK_CAS) >> 4;
    }

    // delta(bit 7) 읽기
    // 공리 15: FSM 밖의 전역 플래그
    getDelta() {
        return (this.bits >> DRing.BIT_DELTA) & 1;
    }

    // delta 발화 설정
    // 공리 15: delta가 스스로 켜고 스스로 끈다
    setDelta(value) {
        this.setBit(DRing.BIT_DELTA, value);
    }

    // 링 이음새: bit 7(delta) -> bit 0(observer)
    // 공리 10, 15: 소유권. delta가 observer를 소유한다
    // delta=1이면 observer 활성화 = 파이프라인 진입점
    getSeam() {
        return {
            delta: this.getDelta(),
            observer: this.getBit(DRing.BIT_OBSERVER)
        };
    }

    // TOCTOU 락 AND 연산
    // 공리 5: 락은 CAS 비트와 도메인 비트의 접점에 존재
    // 공리 1 명제: AND를 수행하는 것은 CAS의 Compare 단계
    checkLock(casBit, domainBit) {
        let _casVal = this.getBit(casBit);
        let _domVal = this.getBit(domainBit);
        return _casVal & _domVal;
    }

    // CAS 3비트 순차 점화
    // 공리 5: R_LOCK -> C_LOCK -> S_LOCK 순서 강제
    // 공리 2 명제: 3축 직교, 점화 순차는 논리 의존성
    advanceCAS(stage) {
        // stage: 0=idle, 1=Read, 2=Compare, 3=Swap
        if (stage === 1) {
            // R->C->S 중 R만 ON
            this.setBit(DRing.BIT_R_LOCK, 1);
        }
        else if (stage === 2) {
            // R+C ON (R이 이미 ON이어야 한다)
            if (!this.getBit(DRing.BIT_R_LOCK)) {
                return false;  // 공리 5 위반: R 없이 C 불가
            }
            this.setBit(DRing.BIT_C_LOCK, 1);
        }
        else if (stage === 3) {
            // R+C+S 전부 ON
            if (!this.getBit(DRing.BIT_C_LOCK)) {
                return false;  // 공리 5 위반: C 없이 S 불가
            }
            this.setBit(DRing.BIT_S_LOCK, 1);
        }
        return true;
    }

    // CAS 사이클 완료 후 동시 리셋
    // 공리 14: 111 -> 000 정방향 전진 후 동시 리셋
    resetCAS() {
        this.setBit(DRing.BIT_R_LOCK, 0);
        this.setBit(DRing.BIT_C_LOCK, 0);
        this.setBit(DRing.BIT_S_LOCK, 0);
    }

    // 전체 상태 스냅샷 (디버깅용)
    snapshot() {
        return {
            raw: this.bits,
            binary: this.bits.toString(2).padStart(8, '0'),
            domain: {
                observer:      this.getBit(DRing.BIT_OBSERVER),
                superposition: this.getBit(DRing.BIT_SUPERPOSITION),
                time:          this.getBit(DRing.BIT_TIME),
                space:         this.getBit(DRing.BIT_SPACE),
                pattern:       this.getDomain()
            },
            cas: {
                R_LOCK: this.getBit(DRing.BIT_R_LOCK),
                C_LOCK: this.getBit(DRing.BIT_C_LOCK),
                S_LOCK: this.getBit(DRing.BIT_S_LOCK),
                state:  this.getCASState()
            },
            delta: this.getDelta(),
            seam:  this.getSeam()
        };
    }

    // 전체 리셋
    reset() {
        this.bits = 0x00;
    }
}

export { DRing };
