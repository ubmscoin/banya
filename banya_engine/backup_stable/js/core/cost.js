// 반야프레임 비용 추적기
// 공리 4: +를 넘을 때마다 비용 1이 발생한다. +를 넘지 않으면 비용은 0
// 공리 6: CAS 1회 총 비용 13 = 읽기 8 + 쓰기 5
//         공 유지 비용 4, 잔존 비용 9 = RLU가 연속 회수

class CostTracker {
    // 공리 6: 비용의 전체 구조
    // 읽기 비용 8:
    //   CAS R->C->S 전이 +3 (CAS 내부 + 넘기)
    //   OPERATOR->고전 괄호 +1 (괄호 경계)
    //   time->space +1 (도메인 경계)
    //   x접근, x->y, y->z +3 (축 경계)
    // 쓰기 비용 5:
    //   time 타임스탬프 +1
    //   x, y, z 각 쓰기 +3
    //   Swap->DATA 커밋 +1
    // 합계 13
    //
    // 유지 비용 4 = 3축 쓰기(3) + 타임스탬프(1) -> DATA에 점으로 남음
    // 잔존 비용 9 = 13 - 4 = RLU 인덱스(중첩)에 기록 -> RLU가 연속 회수

    static TOTAL_PER_CYCLE = 13;
    static MAINTAIN_COST   = 4;   // DATA에 남는 유지 비용
    static RESIDUAL_COST   = 9;   // RLU가 회수할 잔존 비용

    constructor() {
        this.m_totalCost = 0;         // 누적 총 비용
        this.m_cycleCount = 0;        // CAS 사이클 수
        this.m_currentCycle = null;   // 현재 진행 중인 사이클 비용
        this.m_history = [];          // 사이클별 비용 이력
    }

    // 새 CAS 사이클 시작
    beginCycle() {
        this.m_currentCycle = {
            read: { cas: 0, bracket: 0, domain: 0, axis: 0, total: 0 },
            write: { axis: 0, timestamp: 0, commit: 0, total: 0 },
            total: 0,
            maintain: 0,
            residual: 0
        };
    }

    // 읽기 비용 기록
    // 공리 4: 경로마다 넘는 +의 수가 다르다
    addReadCost(category, amount) {
        if (!this.m_currentCycle) {
            return;
        }

        if (category === 'cas') {
            // CAS R->C->S 단계 경계: 각 +1씩 총 +3
            this.m_currentCycle.read.cas += amount;
        }
        else if (category === 'bracket') {
            // OPERATOR->고전 괄호 경계: +1
            this.m_currentCycle.read.bracket += amount;
        }
        else if (category === 'domain') {
            // time->space 도메인 경계: +1
            this.m_currentCycle.read.domain += amount;
        }
        else if (category === 'axis') {
            // x, y, z 축 접근: 각 +1
            this.m_currentCycle.read.axis += amount;
        }

        this.p_recalcCycle();
    }

    // 쓰기 비용 기록
    // 공리 4: 쓰기(취기) = 대상에 값을 기록. 대상 1건당 +1
    addWriteCost(category, amount) {
        if (!this.m_currentCycle) {
            return;
        }

        if (category === 'axis') {
            // x, y, z 쓰기: 각 +1 = +3
            this.m_currentCycle.write.axis += amount;
        }
        else if (category === 'timestamp') {
            // time 타임스탬프 쓰기: +1
            this.m_currentCycle.write.timestamp += amount;
        }
        else if (category === 'commit') {
            // Swap->DATA 커밋: +1
            this.m_currentCycle.write.commit += amount;
        }

        this.p_recalcCycle();
    }

    // 사이클 완료
    // 공리 6: 총 비용을 유지(4)와 잔존(9)으로 분할
    endCycle() {
        if (!this.m_currentCycle) {
            return null;
        }

        let _cycle = this.m_currentCycle;

        // 공리 6: 유지 비용 = 3축 쓰기 + 타임스탬프 = 4
        _cycle.maintain = Math.min(_cycle.write.axis + _cycle.write.timestamp, CostTracker.MAINTAIN_COST);

        // 공리 6: 잔존 비용 = 총 - 유지 = 9
        _cycle.residual = _cycle.total - _cycle.maintain;

        this.m_totalCost += _cycle.total;
        this.m_cycleCount++;
        this.m_history.push({ ..._cycle, cycleIndex: this.m_cycleCount });

        // 최근 100 사이클만 유지
        if (this.m_history.length > 100) {
            this.m_history.shift();
        }

        let _result = { ..._cycle };
        this.m_currentCycle = null;
        return _result;
    }

    // 재계산
    p_recalcCycle() {
        if (!this.m_currentCycle) {
            return;
        }
        let _c = this.m_currentCycle;
        _c.read.total = _c.read.cas + _c.read.bracket + _c.read.domain + _c.read.axis;
        _c.write.total = _c.write.axis + _c.write.timestamp + _c.write.commit;
        _c.total = _c.read.total + _c.write.total;
    }

    // 표준 CAS 1회 비용 기록 (전체 경로)
    // 공리 6: 읽기 8 + 쓰기 5 = 13
    recordFullCycleCost() {
        this.beginCycle();

        // 읽기 비용 8
        this.addReadCost('cas', 3);       // R, C, S 전이 각 +1
        this.addReadCost('bracket', 1);   // 괄호 경계
        this.addReadCost('domain', 1);    // 도메인 경계
        this.addReadCost('axis', 3);      // x, y, z 접근

        // 쓰기 비용 5
        this.addWriteCost('axis', 3);      // x, y, z 쓰기
        this.addWriteCost('timestamp', 1); // 타임스탬프
        this.addWriteCost('commit', 1);    // Swap->DATA

        return this.endCycle();
    }

    // 비가역 비용 누적 (5, 2)
    // 공리 4 명제: 비가역 축 5개(비용 누적 +), 비가역 부재 축 2개(비용 누적 없음)
    getIrreversibleBreakdown() {
        return {
            irreversible: {
                count: 5,
                axes: ['time', 'space', 'R_LOCK', 'C_LOCK', 'S_LOCK'],
                description: 'CAS가 개입하는 축. 비용 누적'
            },
            reversible: {
                count: 2,
                axes: ['observer', 'superposition'],
                description: 'CAS 미개입 구간. 비용 누적 없음'
            }
        };
    }

    // 스냅샷
    snapshot() {
        return {
            totalCost: this.m_totalCost,
            cycleCount: this.m_cycleCount,
            averageCostPerCycle: this.m_cycleCount > 0 ? this.m_totalCost / this.m_cycleCount : 0,
            currentCycle: this.m_currentCycle,
            recentHistory: this.m_history.slice(-5),
            constants: {
                totalPerCycle: CostTracker.TOTAL_PER_CYCLE,
                maintainCost: CostTracker.MAINTAIN_COST,
                residualCost: CostTracker.RESIDUAL_COST
            }
        };
    }

    reset() {
        this.m_totalCost = 0;
        this.m_cycleCount = 0;
        this.m_currentCycle = null;
        this.m_history = [];
    }
}

export { CostTracker };
