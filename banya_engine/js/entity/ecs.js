// 반야프레임 ECS: 공 생성, 이동, 소멸, 예산 순환
import { CAS } from '../core/cas.js';
import { Entity } from './entity.js';
import { RLU } from '../core/rlu.js';
import { AXIOM, RLU_CONST } from '../core/constants.js';

class ECS {
    static COST_PER_CYCLE = AXIOM.COST_TOTAL;
    static TOTAL_SLOTS = 100;

    constructor(dring, costTracker, rlu) {
        this.m_dring = dring;
        this.m_costTracker = costTracker;
        this.m_rlu = rlu;
        this.m_workers = new Map();
        this.m_balls = [];
        this.m_remnants = [];
        this.m_totalCreated = 0;
        this.m_tickCount = 0;
        this.m_budget = AXIOM.BIGBANG_BUDGET;
        this.m_reserved = 0;
    }

    registerObserver(observer) {
        let _cas = new CAS(this.m_dring, observer.m_id);
        this.m_workers.set(observer.m_id, _cas);
    }

    unregisterObserver(observerId) { this.m_workers.delete(observerId); }

    executeTick(observers, domainBits, tick) {
        this.m_tickCount = tick;
        let _results = [];
        if (observers.length === 0) { return _results; }
        let _observer = observers[0];
        let _cas = this.m_workers.get(_observer.m_id);
        if (!_cas) { return _results; }
        let _filterResult = _observer.filter(1, domainBits);
        if (!_filterResult || !_filterResult.deltaActive) { return _results; }

        this.m_budget += this.m_reserved;
        this.m_reserved = 0;

        let _isBigBang = (this.m_totalCreated === 0);
        let _maxCreate = _isBigBang ? Math.floor(this.m_budget / AXIOM.COST_TOTAL) : 1;

        for (let _c = 0; _c < _maxCreate; _c++) {
            if (this.m_budget < AXIOM.COST_TOTAL) { break; }
            let _result = this.p_createBall(_cas, _observer, domainBits, tick);
            if (_result) { _results.push(_result); } else { break; }
        }
        return _results;
    }

    p_createBall(cas, observer, domainBits, tick) {
        if (this.m_budget < AXIOM.COST_TOTAL) { return null; }
        let _newValue = ((domainBits << 16) | (tick & 0xFFFF));
        let _casResult = cas.executeCycle(null, null, _newValue, domainBits);
        if (!_casResult) { return null; }

        this.m_budget -= AXIOM.COST_TOTAL;
        this.m_costTracker.recordFullCycleCost();

        let _phi = Math.random() * 2 * Math.PI;
        let _theta = 0;
        let _r = AXIOM.SPHERE_R;
        let _position = {
            x: Math.round(_r * Math.sin(_theta) * Math.cos(_phi) * 100) / 100,
            y: Math.round(_r * Math.sin(_theta) * Math.sin(_phi) * 100) / 100,
            z: Math.round(_r * Math.cos(_theta) * 100) / 100
        };

        let _entity = new Entity(observer.m_id, domainBits, _position);
        _entity.writeData(_casResult.juim, tick);
        _entity.updateShrinkRadius(AXIOM.COST_TOTAL);
        _entity.m_shrinkRadius = 1.0;  // RLU strength 시작값
        _entity.m_createdTick = tick;
        _entity.m_rluStatus = 'HOT';
        _entity.m_travelPhi = _phi;
        _entity.m_birthTheta = _theta;

        this.m_balls.push(_entity);
        observer.registerEntity(_entity);
        this.m_totalCreated++;

        this.m_rlu.admit(_entity.m_id, { residual: AXIOM.COST_RESIDUAL, position: _position });

        return { observerId: observer.m_id, domainPattern: domainBits, casResult: _casResult, entity: _entity, tick: tick, inFocus: true, action: 'create' };
    }

    processRLU(tick) {
        let _result = this.m_rlu.tick(tick);
        this.m_budget += _result.tickReclaim;
        this.p_moveBalls();
        for (let _evict of _result.evicted) { this.p_evictBall(_evict.entityId); }
        for (let [_id, _entry] of this.m_rlu.m_entries) {
            let _ball = this.m_balls.find(_b => _b.m_id === _id);
            if (_ball) {
                _ball.m_rluStatus = _entry.status;
                _ball.m_shrinkRadius = Math.max(0, _entry.strength);
            }
        }
        this.p_decayRemnants();
        return _result.evicted;
    }

    p_moveBalls() {
        for (let _ball of this.m_balls) {
            if (!_ball.m_alive) { continue; }
            let _rluEntry = this.m_rlu.getEntry(_ball.m_id);
            if (!_rluEntry) { continue; }
            let _progress = _rluEntry.age / RLU_CONST.BALL_LIFE;
            let _theta = _progress * Math.PI * 0.4 + (_ball.m_birthTheta || 0);
            _theta = Math.min(Math.PI, _theta);
            let _phi = _ball.m_travelPhi || 0;
            let _r = AXIOM.SPHERE_R;
            _ball.m_position = {
                x: Math.round(_r * Math.sin(_theta) * Math.cos(_phi) * 100) / 100,
                y: Math.round(_r * Math.sin(_theta) * Math.sin(_phi) * 100) / 100,
                z: Math.round(_r * Math.cos(_theta) * 100) / 100
            };
        }
    }

    p_evictBall(entityId) {
        let _idx = this.m_balls.findIndex(_b => _b.m_id === entityId);
        if (_idx === -1) { return; }
        let _ball = this.m_balls[_idx];
        _ball.evict();
        this.m_remnants.push(_ball);
        this.m_balls.splice(_idx, 1);
    }

    p_decayRemnants() {
        let _toRemove = [];
        for (let i = 0; i < this.m_remnants.length; i++) {
            let _entity = this.m_remnants[i];
            let _decayed = _entity.decayMass(RLU_CONST.REMNANT_LIFE);
            this.m_rlu.m_totalReclaimed += _decayed;
            this.m_budget += _decayed;
            if (_entity.m_remnantAge === undefined) { _entity.m_remnantAge = 0; }
            _entity.m_remnantAge++;
            let _totalProgress = (RLU_CONST.BALL_LIFE + _entity.m_remnantAge) / RLU_CONST.MAX_LIFE;
            let _theta = _totalProgress * Math.PI;
            _theta = Math.min(Math.PI, _theta);
            let _phi = _entity.m_travelPhi || 0;
            let _r = AXIOM.SPHERE_R;
            _entity.m_position = {
                x: Math.round(_r * Math.sin(_theta) * Math.cos(_phi) * 100) / 100,
                y: Math.round(_r * Math.sin(_theta) * Math.sin(_phi) * 100) / 100,
                z: Math.round(_r * Math.cos(_theta) * 100) / 100
            };
            if (_entity.isFullyDead()) { _toRemove.push(i); this.m_rlu.removeRemnant(_entity.m_id); }
        }
        for (let i = _toRemove.length - 1; i >= 0; i--) { this.m_remnants.splice(_toRemove[i], 1); }
    }

    getAllEntitiesIncludingRemnants() { return [...this.m_balls, ...this.m_remnants]; }
    getFocusedEntities(observers) {
        if (observers.length === 0) { return []; }
        return this.m_balls.filter(_b => _b.m_alive && observers[0].isInFocus(_b));
    }
    getAllEntities() { return [...this.m_balls]; }

    snapshot() {
        let _alive = this.m_balls.filter(_b => _b.m_alive);
        return {
            totalEntities: _alive.length, totalSlots: ECS.TOTAL_SLOTS,
            filledSlots: _alive.length, emptySlots: 0,
            remnantCount: this.m_remnants.length, workerCount: this.m_workers.size,
            tickCount: this.m_tickCount,
            budget: Math.round(this.m_budget * 100) / 100,
            reserved: Math.round(this.m_reserved * 100) / 100,
            totalCreated: this.m_totalCreated,
            entityCountByObserver: { 0: _alive.length },
            entities: _alive.map(_e => _e.snapshot()),
            remnants: this.m_remnants.map(_e => _e.snapshot()),
            rlu: this.m_rlu.snapshot()
        };
    }

    getBallSnapshots() {
        let _all = [];
        for (let _b of this.m_balls) {
            let _sph = _b.getSphericalCoords();
            _all.push({ id: _b.m_id, theta: _sph.theta, phi: _sph.phi, position: _b.m_position, filled: true, alive: _b.m_alive, entity: _b.snapshot() });
        }
        let _remLimit = this.m_remnants.length;
        for (let i = 0; i < _remLimit; i++) {
            let _r = this.m_remnants[i];
            let _sph = _r.getSphericalCoords();
            _all.push({ id: _r.m_id, theta: _sph.theta, phi: _sph.phi, position: _r.m_position, filled: false, alive: false, entity: { id: _r.m_id, rluStatus: 'REMNANT', alive: false, remnant: true, mass: _r.m_mass, shrinkRadius: _r.m_shrinkRadius } });
        }
        return _all;
    }

    reset() {
        this.m_balls = []; this.m_remnants = []; this.m_workers.clear();
        this.m_totalCreated = 0; this.m_tickCount = 0;
        this.m_budget = AXIOM.BIGBANG_BUDGET; this.m_reserved = 0;
    }
}

export { ECS };
