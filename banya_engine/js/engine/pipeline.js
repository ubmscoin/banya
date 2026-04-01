// 반야프레임 Pipeline: 파이프라인 실행 관리자
// 공리 15 명제: trigger -> filter -> update -> render -> screen
// d-ring 1회 발화의 흐름
//
// v0.2 샌드박스: 웹 자체가 샌드박스다
//   document ready = 샌드박스 시작 = 반야식의 최외각
//   모든 것은 샌드박스 속 현상이다. delta 발화도 샌드박스 안에서 일어난다
//   웹 종료 = 샌드박스 소멸

import { DRing } from '../core/dring.js';
import { FSM } from '../core/fsm.js';
import { CostTracker } from '../core/cost.js';
import { LRU } from '../core/lru.js';
import { Observer } from '../entity/observer.js';
import { ECS } from '../entity/ecs.js';
import { Delta } from './delta.js';
import { AXIOM, SYSTEM } from '../core/constants.js';

class Pipeline {
    // 파이프라인 5단계
    static STAGE_IDLE    = 'idle';
    static STAGE_TRIGGER = 'trigger';   // delta 발화 ON. 등호 성립
    static STAGE_FILTER  = 'filter';    // observer 필터링. 비용 0
    static STAGE_UPDATE  = 'update';    // CAS 실행. 비용 발생
    static STAGE_RENDER  = 'render';    // DATA에 기록 = 렌더링 완료
    static STAGE_SCREEN  = 'screen';    // 한 프레임 완성. delta 소화

    constructor(config) {
        let _cfg = config || {};

        // 샌드박스 플래그: 웹이 켜지면 true. 모든 것은 이 안에서 일어난다
        this.m_sandboxAlive = true;
        this.m_debugMode = true;

        // 핵심 컴포넌트 생성
        this.m_dring = new DRing();
        this.m_fsm = new FSM();
        this.m_costTracker = new CostTracker();
        this.m_lru = new LRU(_cfg.ringSize || 30);
        this.m_delta = new Delta(this.m_dring);
        this.m_ecs = new ECS(this.m_dring, this.m_costTracker, this.m_lru);

        // 이벤트 로그 (addObserver보다 먼저 초기화)
        this.m_log = [];
        this.m_maxLogSize = 200;

        // 파이프라인 상태
        this.m_stage = Pipeline.STAGE_IDLE;
        this.m_tick = 0;
        this.m_running = false;
        this.m_tickInterval = _cfg.tickInterval || 500;  // ms
        this.m_timerId = null;

        // 콜백
        this.m_onTick = _cfg.onTick || null;
        this.m_onLog = _cfg.onLog || null;

        // observer: 1개만 존재한다
        // 공리 10: delta(전체) -> observer(국소). observer는 1개의 포커스
        // 공리 15: 등호의 진입점은 하나
        this.m_observers = [];
        this.addObserver();
    }

    // observer 생성 (1개만)
    // 포커스 = 북극 고정 (theta=0). 큰 구가 회전하면 포커스는 북극에서 따라 회전
    addObserver(theta, phi) {
        let _defaultTheta = (theta !== undefined) ? theta : 0;  // 북극
        let _defaultPhi = (phi !== undefined) ? phi : 0;

        let _observer = new Observer(_defaultTheta, _defaultPhi);
        this.m_observers.push(_observer);
        this.m_ecs.registerObserver(_observer);

        this.p_log('system', `observer 생성 (focus r=${Observer.FOCUS_RADIUS.toFixed(4)}, HOT 5% from axiom 6,15)`);
        return _observer;
    }

    // observer 포커스 이동 (마우스 드래그)
    moveObserverFocus(theta, phi) {
        if (this.m_observers.length > 0) {
            this.m_observers[0].moveFocus(theta, phi);
        }
    }

    // 1틱 실행
    // d-ring 서브스텝 분할: 매 프레임 1단계씩 진행
    // 6단계로 나눠서 d-ring 비트 점등 순서가 눈에 보인다
    //   0: delta 발화 (bit 7)
    //   1: 도메인 활성화 (bit 0~3 동시)
    //   2: R_LOCK (bit 4)
    //   3: C_LOCK (bit 5)
    //   4: S_LOCK (bit 6) + CAS 실행 + LRU
    //   5: 리셋 (전부 OFF)
    // m_debugMode: 체크박스 체크시 6서브스텝, 아니면 1스텝으로 빠르게
    step() {
        if (this.m_debugMode) {
            this.p_stepDebug();
        }
        else {
            this.p_stepFast();
        }

        this.p_notifyTick();

        return {
            stage: this.m_stage,
            tick: this.m_tick,
            entityCount: this.m_ecs.m_balls.filter(_b => _b.m_alive).length
        };
    }

    // 빠른 모드: 1프레임 = 1틱. 서브스텝 없음. d-ring/FSM 락 안 건드림
    p_stepFast() {
        this.m_tick++;
        this.m_delta.fire();

        // CAS 실행 (예산 있을 때만)
        if (this.m_ecs.m_budget >= AXIOM.COST_TOTAL) {
            this.m_dring.setBit(DRing.BIT_OBSERVER, 1);
            this.m_dring.setBit(DRing.BIT_SUPERPOSITION, 1);
            this.m_dring.setBit(DRing.BIT_TIME, 1);
            this.m_dring.setBit(DRing.BIT_SPACE, 1);
            let _domainBits = this.m_dring.getDomain();
            this.m_ecs.executeTick(this.m_observers, _domainBits, this.m_tick);
        }

        // LRU 항상
        this.m_ecs.processLRU(this.m_tick);

        // 리셋
        this.m_delta.extinguish();
        this.m_dring.reset();
    }

    // 디버그 모드: 6서브스텝. d-ring 비트 점등 순서가 보인다
    p_stepDebug() {
        if (this.m_subStep === undefined) { this.m_subStep = 0; }
        let _phase = this.m_subStep;

        if (_phase === 0) {
            this.m_tick++;
            this.m_dring.reset();
            this.m_delta.fire();
            this.m_stage = Pipeline.STAGE_TRIGGER;
        }
        else if (_phase === 1) {
            this.m_delta.executeSeam();
            this.m_dring.setBit(DRing.BIT_SUPERPOSITION, 1);
            this.m_dring.setBit(DRing.BIT_TIME, 1);
            this.m_dring.setBit(DRing.BIT_SPACE, 1);
            this.m_stage = Pipeline.STAGE_FILTER;
        }
        else if (_phase === 2) {
            this.m_hasBudget = (this.m_ecs.m_budget >= AXIOM.COST_TOTAL);
            if (this.m_hasBudget) {
                this.m_dring.setBit(DRing.BIT_R_LOCK, 1);
                this.m_fsm.transition(FSM.STATE_READ);
                this.m_stage = Pipeline.STAGE_UPDATE;
            }
        }
        else if (_phase === 3) {
            if (this.m_hasBudget) {
                this.m_dring.setBit(DRing.BIT_C_LOCK, 1);
                this.m_fsm.transition(FSM.STATE_COMPARE);
            }
        }
        else if (_phase === 4) {
            if (this.m_hasBudget) {
                this.m_dring.setBit(DRing.BIT_S_LOCK, 1);
                this.m_fsm.transition(FSM.STATE_SWAP);
                this.m_stage = Pipeline.STAGE_RENDER;
            }
        }
        else if (_phase === 5) {
            if (this.m_hasBudget) {
                let _domainBits = this.m_dring.getDomain();
                this.m_lastResults = this.m_ecs.executeTick(this.m_observers, _domainBits, this.m_tick);
                this.m_delta.feedbackResult(this.m_lastResults && this.m_lastResults.length > 0 ? this.m_lastResults : null);
            }
            this.m_fsm.reset();
            this.m_stage = Pipeline.STAGE_SCREEN;
            this.m_ecs.processLRU(this.m_tick);
            this.m_delta.extinguish();
            this.m_dring.setBit(DRing.BIT_OBSERVER, 0);
            this.m_dring.setBit(DRing.BIT_SUPERPOSITION, 0);
            this.m_dring.setBit(DRing.BIT_TIME, 0);
            this.m_dring.setBit(DRing.BIT_SPACE, 0);
            this.m_dring.resetCAS();
            this.m_stage = Pipeline.STAGE_IDLE;
        }
        this.m_subStep = (_phase + 1) % 6;
    }

    // 수동 delta 발화
    fireDelta() {
        return this.m_delta.fire();
    }

    // 자동 실행 시작. 절대 멈추지 않는다
    play() {
        if (this.m_running) {
            return;
        }
        this.m_running = true;
        this.m_delta.setAutoFire(true, 1);
        this.m_lastStepTime = 0;
        this.m_rafId = requestAnimationFrame((t) => this.p_loop(t));
    }

    // requestAnimationFrame 루프: 틱 간격에 맞춰 step 호출
    // 렌더링은 브라우저 최적 타이밍(60fps)으로, 틱은 tickInterval 간격으로
    p_loop(timestamp) {
        if (!this.m_running) {
            return;
        }
        if (timestamp - this.m_lastStepTime >= this.m_tickInterval) {
            try {
                this.step();
            }
            catch (_err) {
                console.error('step error:', _err);
            }
            this.m_lastStepTime = timestamp;
        }
        this.m_rafId = requestAnimationFrame((t) => this.p_loop(t));
    }

    // 일시 정지
    pause() {
        this.m_running = false;
        if (this.m_rafId) {
            cancelAnimationFrame(this.m_rafId);
            this.m_rafId = null;
        }
        this.m_delta.setAutoFire(false);
    }

    // 속도 설정
    setSpeed(intervalMs) {
        this.m_tickInterval = Math.max(SYSTEM.MIN_TICK_MS, Math.min(SYSTEM.MAX_TICK_MS, intervalMs));
        if (this.m_running) {
            this.pause();
            this.play();
        }
    }

    // 도메인 비트 수동 토글
    toggleDomainBit(bitIndex) {
        if (bitIndex < 0 || bitIndex > 3) {
            return;
        }
        let _current = this.m_dring.getBit(bitIndex);
        this.m_dring.setBit(bitIndex, _current ? 0 : 1);
    }

    // 로그
    p_log(type, message) {
        let _entry = { type: type, message: message, tick: this.m_tick, time: Date.now() };
        this.m_log.push(_entry);
        if (this.m_log.length > this.m_maxLogSize) {
            this.m_log.shift();
        }
        if (this.m_onLog) {
            this.m_onLog(_entry);
        }
    }

    // 틱 알림
    p_notifyTick() {
        if (this.m_onTick) {
            this.m_onTick(this.snapshot());
        }
    }

    // 전체 스냅샷 (UI 렌더링용)
    // 수정9: 구면(OPERATOR)=전체, 시공(DATA)=포커스만
    snapshot() {
        return {
            tick: this.m_tick,
            stage: this.m_stage,
            running: this.m_running,
            sandboxAlive: this.m_sandboxAlive,
            dring: this.m_dring.snapshot(),
            fsm: this.m_fsm.snapshot(),
            delta: this.m_delta.snapshot(),
            cost: this.m_costTracker.snapshot(),
            ecs: this.m_ecs.snapshot(),
            balls: this.m_ecs.getBallSnapshots(),
            observers: this.m_observers.map(_o => _o.snapshot()),
            budget: this.m_ecs.m_budget
        };
    }

    // 리셋
    reset() {
        this.pause();
        this.m_dring.reset();
        this.m_fsm.reset();
        this.m_costTracker.reset();
        this.m_lru.reset();
        this.m_delta.reset();
        this.m_ecs.reset();

        // observer 재등록
        for (let _obs of this.m_observers) {
            this.m_ecs.registerObserver(_obs);
        }

        this.m_tick = 0;
        this.m_stage = Pipeline.STAGE_IDLE;
        this.m_log = [];
    }
}

export { Pipeline };
