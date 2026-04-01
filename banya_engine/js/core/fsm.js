// 반야프레임 FSM: 유한 상태 기계
// 공리 14: CAS-ring의 상태 전이를 비트 연산자 관점에서 기술
// 상태 집합: {000, 001, 011, 111}
// 전이: 000 -> 001 -> 011 -> 111 -> 000 (정방향 전진, 역순 후퇴 아님)

class FSM {
    // 공리 14: FSM 구성요소
    // 상태 집합: {000, 001, 011, 111} = 공리 5 (TOCTOU 락)
    // 입력 알파벳: DATA의 현재값 = 공리 1 (4축 도메인)
    // 전이 함수: CAS Read->Compare->Swap = 공리 2
    // 시작 상태: 000 (대기) = 공리 5
    // 종료 상태: 111 (CAS 성공) -> 000 (초기화) = 공리 5
    // 출력: DATA 쓰기 또는 중첩 유지 = 공리 7
    // 클록: 시스템 시간 1틱 = 공리 8 (폴링)

    static STATE_IDLE    = 0b000;  // 대기: CAS-ring이 뜬다
    static STATE_READ    = 0b001;  // Read 진입: R_LOCK ON
    static STATE_COMPARE = 0b011;  // Compare 진입: R+C ON
    static STATE_SWAP    = 0b111;  // Swap 완료: R+C+S 전부 ON

    // FSM 노름: 활성 축 수에 따라 결정
    // 공리 2 명제: 3축 직교이므로 노름이 sqrt(1), sqrt(2), sqrt(3)
    static NORM = {
        0b000: 0,              // idle. 비용 0
        0b001: 1,              // sqrt(1) = 1. Read 진입
        0b011: Math.SQRT2,     // sqrt(2). Compare 시점
        0b111: Math.sqrt(3)    // sqrt(3). 워크벤치 노름. CAS 완성
    };

    constructor() {
        this.m_state = FSM.STATE_IDLE;
        this.m_cycleCount = 0;       // 완료된 CAS 사이클 수
        this.m_transitionLog = [];   // 전이 이력
    }

    // 현재 상태
    getState() {
        return this.m_state;
    }

    // 현재 노름
    // 공리 2 명제: FSM 상태에서 활성 축의 노름
    getNorm() {
        return FSM.NORM[this.m_state] || 0;
    }

    // 상태 전이 시도
    // 공리 14: 정방향 전진만 가능. 역순 후퇴 불가 (공리 2 명제: 비가역)
    // 공리 5: 락이 순서를 강제
    transition(targetState) {
        let _valid = this.p_isValidTransition(this.m_state, targetState);
        if (!_valid) {
            return false;
        }

        let _prevState = this.m_state;
        this.m_state = targetState;

        this.m_transitionLog.push({
            from: _prevState,
            to: targetState,
            tick: this.m_cycleCount,
            norm: this.getNorm()
        });

        // 공리 14: 111에 도달하면 사이클 완료
        if (targetState === FSM.STATE_SWAP) {
            this.m_cycleCount++;
        }

        return true;
    }

    // 리셋: 111 -> 000
    // 공리 14: 정방향 전진 후 동시 리셋. 역순 후퇴가 아니다
    reset() {
        if (this.m_state !== FSM.STATE_SWAP && this.m_state !== FSM.STATE_IDLE) {
            // 비정상 리셋: 미완성 사이클 (공리 2: 원자성 깨짐)
            this.m_transitionLog.push({
                from: this.m_state,
                to: FSM.STATE_IDLE,
                tick: this.m_cycleCount,
                abort: true
            });
        }
        this.m_state = FSM.STATE_IDLE;
    }

    // 전이 유효성 검사
    // 공리 2 명제: 비가역성. R->C->S 역순 불가
    // 공리 5: 락 순서 강제
    p_isValidTransition(from, to) {
        // 허용되는 전이만 나열
        if (from === FSM.STATE_IDLE && to === FSM.STATE_READ) {
            return true;   // 000 -> 001
        }
        if (from === FSM.STATE_READ && to === FSM.STATE_COMPARE) {
            return true;   // 001 -> 011
        }
        if (from === FSM.STATE_COMPARE && to === FSM.STATE_SWAP) {
            return true;   // 011 -> 111
        }
        // 111 -> 000은 reset()으로만 처리
        return false;
    }

    // 다음 상태 자동 결정
    // 공리 14: 순환 순서가 고정되어 있다
    nextState() {
        if (this.m_state === FSM.STATE_IDLE) {
            return FSM.STATE_READ;
        }
        if (this.m_state === FSM.STATE_READ) {
            return FSM.STATE_COMPARE;
        }
        if (this.m_state === FSM.STATE_COMPARE) {
            return FSM.STATE_SWAP;
        }
        // STATE_SWAP이면 다음은 리셋(IDLE)
        return FSM.STATE_IDLE;
    }

    // 자동 1단계 전진
    advance() {
        let _next = this.nextState();
        if (_next === FSM.STATE_IDLE) {
            this.reset();
            return { state: FSM.STATE_IDLE, cycleComplete: true };
        }
        let _ok = this.transition(_next);
        return { state: this.m_state, cycleComplete: false, valid: _ok };
    }

    // 스냅샷
    snapshot() {
        return {
            state: this.m_state,
            stateBinary: this.m_state.toString(2).padStart(3, '0'),
            stateName: this.p_stateName(this.m_state),
            norm: this.getNorm(),
            cycleCount: this.m_cycleCount,
            recentTransitions: this.m_transitionLog.slice(-10)
        };
    }

    p_stateName(state) {
        const NAMES = {
            [FSM.STATE_IDLE]: 'IDLE (000)',
            [FSM.STATE_READ]: 'READ (001)',
            [FSM.STATE_COMPARE]: 'COMPARE (011)',
            [FSM.STATE_SWAP]: 'SWAP (111)'
        };
        return NAMES[state] || 'UNKNOWN';
    }
}

export { FSM };
