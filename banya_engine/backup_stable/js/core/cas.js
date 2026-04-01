// 반야프레임 CAS: Compare-And-Swap 유일 연산자
// 공리 2: 우주에서 일어나는 모든 변화는 CAS 단일 연산의 반복
// 공리 2 명제: CAS는 자체 저장소가 없는 독립적 지역 연산

import { DRing } from './dring.js';

// 공리 2 명제: 자료형 11개
// CAS가 대상을 읽는 크기 단위. 입력 {3}에서 4개 연산(+, T(N)+1, 2^N, sqrt(3))으로 도출
const DATA_TYPES = [1, 2, 3, 4, 7, 8, 9, 16, 30, 128, 137];

class CAS {
    // 공리 2: Read(자유도1) -> Compare(자유도2) -> Swap(자유도4)
    // 공리 2 명제: CAS 내부 상태의 합 1+2+4 = 7

    static STAGE_IDLE    = 0;  // 000: 대기
    static STAGE_READ    = 1;  // 001: 읽기 완료
    static STAGE_COMPARE = 2;  // 011: 비교 완료
    static STAGE_SWAP    = 3;  // 111: 쓰기 완료 (CAS 성공)

    constructor(dring, observerId) {
        this.m_dring = dring;
        this.m_observerId = observerId;

        // CAS 내부 상태: 공리 2 명제에서 7개 독립 변수
        this.m_stage = CAS.STAGE_IDLE;
        this.m_readValue = null;       // Read가 가져온 현재값
        this.m_expectedValue = null;   // Compare의 기대값
        this.m_newValue = null;        // Swap이 쓸 새 값
        this.m_compareResult = false;  // true/false
        this.m_isWritable = false;     // Swap 가능 여부
        this.m_domainPattern = 0;      // 도메인 4비트 접근 패턴

        // 비용 누적: 공리 4
        this.m_costRead = 0;
        this.m_costCompare = 0;
        this.m_costSwap = 0;

        // 워크벤치 자료형: 공리 2 명제
        this.m_dataTypeSize = 7;  // 기본 자료형 크기
    }

    // 공리 2: Read 단계 - 현재 상태를 가져올 뿐
    // 공리 4: +를 넘으며 비용 누적
    // 공리 5: R_LOCK ON
    read(target, domainPattern) {
        if (this.m_stage !== CAS.STAGE_IDLE) {
            return null;  // 이전 사이클이 끝나지 않았다
        }

        // 공리 5: R_LOCK 점화
        let _advanced = this.m_dring.advanceCAS(1);
        if (!_advanced) {
            return null;
        }

        this.m_stage = CAS.STAGE_READ;
        this.m_domainPattern = domainPattern;
        this.m_readValue = target;

        // 공리 4: 읽기 비용 산출
        // 도메인 경계 넘기 + CAS 단계 경계 넘기 = 경로마다 다르다
        this.m_costRead = this.p_calcReadCost(domainPattern);

        return this.m_readValue;
    }

    // 공리 2: Compare 단계 - 기대값과 비교. 분기 발생
    // 공리 7: true면 봉괴, false면 중첩 유지
    // 공리 5: C_LOCK ON (R_LOCK 선행 필수)
    compare(expectedValue) {
        if (this.m_stage !== CAS.STAGE_READ) {
            return null;  // 공리 5 위반: Read 없이 Compare 불가
        }

        // 공리 5: C_LOCK 점화
        let _advanced = this.m_dring.advanceCAS(2);
        if (!_advanced) {
            return null;
        }

        this.m_stage = CAS.STAGE_COMPARE;
        this.m_expectedValue = expectedValue;

        // 공리 7: Compare 결과 = 봉괴 여부 결정
        // 공리 2 명제: Compare는 비교 쌍 T(N)+1을 생성
        this.m_compareResult = this.p_compareValues(this.m_readValue, expectedValue);
        this.m_isWritable = this.m_compareResult;

        // 공리 4: Compare 비용
        this.m_costCompare = 1;  // +를 넘음

        return this.m_compareResult;
    }

    // 공리 2: Swap 단계 - 조건부 쓰기
    // 공리 7: Compare true일 때만 실행 -> 봉괴 -> DATA에 점 생성
    // 공리 5: S_LOCK ON (C_LOCK 선행 필수)
    swap(newValue) {
        if (this.m_stage !== CAS.STAGE_COMPARE) {
            return null;  // 공리 5 위반: Compare 없이 Swap 불가
        }

        // 공리 5: S_LOCK 점화
        let _advanced = this.m_dring.advanceCAS(3);
        if (!_advanced) {
            return null;
        }

        this.m_stage = CAS.STAGE_SWAP;
        this.m_newValue = newValue;

        let _result = {
            success: false,
            collapsed: false,
            juim: null,
            cost: 0
        };

        if (this.m_isWritable) {
            // 공리 7: Compare true -> 봉괴 -> 쓰기 실행
            // 공리 2 명제: Swap은 3축 노름 점(juim)을 만든다
            _result.success = true;
            _result.collapsed = true;
            _result.juim = this.p_createJuim(newValue);

            // 공리 4: Swap 비용 (쓰기 비용)
            this.m_costSwap = this.p_calcWriteCost();
        }
        else {
            // 공리 7: Compare false -> 중첩 유지 -> Swap 미실행
            _result.success = false;
            _result.collapsed = false;
        }

        _result.cost = this.m_costRead + this.m_costCompare + this.m_costSwap;

        return _result;
    }

    // CAS 사이클 완료: 리셋
    // 공리 14: 111 -> 000 동시 리셋
    complete() {
        this.m_dring.resetCAS();
        this.m_stage = CAS.STAGE_IDLE;
        this.m_readValue = null;
        this.m_expectedValue = null;
        this.m_newValue = null;
        this.m_compareResult = false;
        this.m_isWritable = false;

        let _totalCost = this.m_costRead + this.m_costCompare + this.m_costSwap;
        this.m_costRead = 0;
        this.m_costCompare = 0;
        this.m_costSwap = 0;

        return _totalCost;
    }

    // 1회 전체 CAS 사이클 실행
    // 공리 2: Read -> Compare -> Swap -> Reset
    // 공리 2 명제: 원자성 - 3단계 분리 불가
    executeCycle(target, expectedValue, newValue, domainPattern) {
        // read 실행. target이 null이어도 "빈 슬롯 읽기"로 유효
        this.read(target, domainPattern);
        if (this.m_stage !== CAS.STAGE_READ) {
            return null;  // advanceCAS 실패 등 비정상
        }

        let _compareResult = this.compare(expectedValue);
        if (_compareResult === null) {
            this.p_abortCycle();
            return null;
        }

        let _swapResult = this.swap(newValue);
        if (_swapResult === null) {
            this.p_abortCycle();
            return null;
        }

        let _totalCost = this.complete();
        _swapResult.totalCycleCost = _totalCost;

        return _swapResult;
    }

    // 읽기 비용 계산
    // 공리 4: +를 넘는 경계 횡단 횟수
    // 공리 6: 총 비용 13 = 읽기 8 + 쓰기 5
    p_calcReadCost(domainPattern) {
        let _cost = 0;

        // CAS R 단계 진입: +를 넘음 = +1
        _cost += 1;

        // OPERATOR -> 고전 괄호 경계: +1
        _cost += 1;

        // 도메인 경계: time->space = +1
        _cost += 1;

        // space 내 축 접근: x, y, z 각 +1 = +3 (읽기)
        // 도메인 패턴에 따라 다르지만, 전체 접근 시 최대 +3
        let _activeBits = this.p_countActiveBits(domainPattern & 0x0C);  // time+space
        _cost += _activeBits + 1;  // 최소 1개 축은 접근

        // 총 읽기 비용 = 경로에 따라 5~8
        return Math.min(_cost, 8);  // 공리 6: 읽기 비용 최대 8
    }

    // 쓰기 비용 계산
    // 공리 6: 쓰기 비용 = 3축 쓰기(3) + 타임스탬프(1) + Swap DATA 커밋(1) = 5
    p_calcWriteCost() {
        // 3축 쓰기: x, y, z 각 +1
        let _cost = 3;

        // time 타임스탬프 쓰기: +1
        _cost += 1;

        // Swap -> DATA 커밋: +1
        _cost += 1;

        return _cost;  // 항상 5
    }

    // 점(juim) 생성
    // 공리 2 명제: CAS Swap(111)이 +를 넘어 DATA(space)에 공 하나를 퀸다
    // 점은 이산 단위, 구형, 3축 직교 등방
    p_createJuim(value) {
        return {
            value: value,
            timestamp: Date.now(),
            isDiscrete: true,  // 공리 3: DATA는 이산
            shape: 'sphere'    // 공리 2 명제: 3축 직교 등방 구형
        };
    }

    // 값 비교
    // 공리 7: Compare true = 봉괴 조건
    // 둘 다 null이면 "빈 슬롯에 최초 생성" = 봉괴 허용
    p_compareValues(current, expected) {
        if (current === null && expected === null) {
            return true;   // 빈 DATA 슬롯: 최초 점 생성 허용
        }
        if (current === null || expected === null) {
            return false;
        }
        if (typeof current === 'object' && typeof expected === 'object') {
            return JSON.stringify(current) === JSON.stringify(expected);
        }
        return current === expected;
    }

    // 활성 비트 수 세기
    p_countActiveBits(pattern) {
        let _count = 0;
        let _p = pattern;
        while (_p) {
            _count += _p & 1;
            _p >>= 1;
        }
        return _count;
    }

    // 비정상 중단
    p_abortCycle() {
        this.m_dring.resetCAS();
        this.m_stage = CAS.STAGE_IDLE;
        this.m_costRead = 0;
        this.m_costCompare = 0;
        this.m_costSwap = 0;
    }

    // 스냅샷 (디버깅용)
    snapshot() {
        return {
            observerId: this.m_observerId,
            stage: this.m_stage,
            stageName: ['IDLE', 'READ', 'COMPARE', 'SWAP'][this.m_stage],
            readValue: this.m_readValue,
            expectedValue: this.m_expectedValue,
            compareResult: this.m_compareResult,
            isWritable: this.m_isWritable,
            domainPattern: this.m_domainPattern,
            cost: {
                read: this.m_costRead,
                compare: this.m_costCompare,
                swap: this.m_costSwap,
                total: this.m_costRead + this.m_costCompare + this.m_costSwap
            }
        };
    }
}

export { CAS, DATA_TYPES };
