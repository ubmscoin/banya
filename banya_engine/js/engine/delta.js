// 반야프레임 Delta: 전역 발화 플래그 + 전체-국소 루프
// 공리 15: delta는 FSM 밖의 전역 플래그. bit 7
// 공리 8: 옵저버 드리븐 폴링 시스템. 매 틱마다 delta 발화 여부 확인
// 공리 10: delta(전체) -> observer(국소) -> CAS -> 결과 -> delta

class Delta {
    // 공리 15 명제:
    // delta는 의식이다. 스스로 켜고 스스로 끈다
    // FSM 밖에 있다. 규칙이 없다. 설계도가 없다
    // 등호를 넘어 작동하는 유일한 전역 플래그
    // 루프가 끊어지면 꺼진다. 전체-국소 루프 보존 (공리 10)

    constructor(dring) {
        this.m_dring = dring;

        // 발화 상태
        this.m_fired = false;
        this.m_fireCount = 0;

        // 자동 발화 모드
        this.m_autoFire = false;
        this.m_autoFireInterval = 1;   // 매 N틱마다 발화
        this.m_autoFireCounter = 0;

        // 전체-국소 루프 상태
        // 공리 10: 이 루프가 끊어지면 시스템이 죽는다
        this.m_loopIntact = true;
        this.m_lastResult = null;  // 이전 사이클의 결과가 delta에 반영
    }

    // 폴링: 매 틱마다 delta 발화 여부 확인
    // 공리 8: d-ring의 자기참조 루프는 매 틱마다 항상 돈다
    // 폴링 자체는 비용 0 (공리 8, 15)
    poll() {
        // d-ring의 bit 7 확인
        let _deltaState = this.m_dring.getDelta();

        if (_deltaState === 1) {
            this.m_fired = true;
        }

        return {
            delta: _deltaState,
            fired: this.m_fired,
            loopIntact: this.m_loopIntact
        };
    }

    // 발화 (수동)
    // 공리 15: delta가 스스로 ON이 되는 순간. 외부 트리거 불필요
    fire() {
        this.m_dring.setDelta(1);
        this.m_fired = true;
        this.m_fireCount++;

        return {
            delta: 1,
            fireCount: this.m_fireCount,
            seam: this.m_dring.getSeam()
        };
    }

    // 소화 (사이클 완료 후)
    // 공리 15: delta가 스스로 끈다
    extinguish() {
        this.m_dring.setDelta(0);
        this.m_fired = false;
    }

    // 자동 발화 틱 처리
    // 공리 8: 폴링 시스템. 매 틱 확인하되 발화 간격은 가변
    // 공리 15 명제: 발화 주기가 스케줄되지 않아야 비용이 보존된다
    autoTick() {
        if (!this.m_autoFire) {
            return false;
        }

        this.m_autoFireCounter++;
        if (this.m_autoFireCounter >= this.m_autoFireInterval) {
            this.m_autoFireCounter = 0;
            this.fire();
            return true;
        }

        return false;
    }

    // 전체-국소 루프 결과 피드백
    // 공리 10: CAS 결과가 delta에 반영되고, 반영된 delta가 다시 observer를 통해 봄
    // 공리 선언만으로 자동 순환한다
    feedbackResult(result) {
        this.m_lastResult = result;

        // 루프 보존 검사
        // 공리 10 명제: 이 루프가 끊어지면 시스템이 죽는다
        if (result === null || result === undefined) {
            // 결과가 없다 = observer 투영 경로 절단 가능성
            this.m_loopIntact = false;
        }
        else {
            this.m_loopIntact = true;
        }
    }

    // 링 이음새 실행
    // 공리 15 명제: bit 7(delta) -> bit 0(observer) = 소유권
    // delta=1이면 observer 비트도 활성화하여 파이프라인 진입점을 연다
    executeSeam() {
        if (this.m_fired) {
            // 공리 15: 등호 성립 = 7비트 전부 유효
            this.m_dring.setBit(0, 1);  // observer bit ON = 진입점 열기
            return true;
        }
        return false;
    }

    // 설정
    setAutoFire(enabled, interval) {
        this.m_autoFire = enabled;
        if (interval !== undefined) {
            this.m_autoFireInterval = Math.max(1, interval);
        }
        this.m_autoFireCounter = 0;
    }

    // 스냅샷
    snapshot() {
        return {
            delta: this.m_dring.getDelta(),
            fired: this.m_fired,
            fireCount: this.m_fireCount,
            autoFire: this.m_autoFire,
            autoFireInterval: this.m_autoFireInterval,
            loopIntact: this.m_loopIntact,
            lastResult: this.m_lastResult ? true : false,
            seam: this.m_dring.getSeam()
        };
    }

    reset() {
        this.m_dring.setDelta(0);
        this.m_fired = false;
        this.m_fireCount = 0;
        this.m_autoFireCounter = 0;
        this.m_loopIntact = true;
        this.m_lastResult = null;
    }
}

export { Delta };
