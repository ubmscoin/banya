// 반야프레임 RLU: 점의 수명 관리
// 공리 6: 잔존 비용을 RLU가 연속적으로 회수
// 1000단계 선형감쇠. 비용 13을 1000등분

import { AXIOM, RLU_CONST } from './constants.js';

class RLU {
    static STATUS_HOT  = 'HOT';
    static STATUS_WARM = 'WARM';
    static STATUS_COLD = 'COLD';
    static STATUS_REMNANT = 'REMNANT';

    static MAX_LIFE = RLU_CONST.MAX_LIFE;
    static BALL_LIFE = RLU_CONST.BALL_LIFE;
    static REMNANT_LIFE = RLU_CONST.REMNANT_LIFE;
    static HOT_END = RLU_CONST.HOT_END;
    static WARM_END = RLU_CONST.WARM_END;
    static DECAY_RATIO = RLU_CONST.DECAY_RATIO;
    static DECAY_THRESHOLD = RLU_CONST.DECAY_THRESHOLD;

    constructor(ringSize) {
        this.m_ringSize = ringSize || 30;
        this.m_entries = new Map();
        this.m_totalReclaimed = 0;
        this.m_remnants = new Map();
    }

    admit(entityId, costData) {
        this.m_entries.set(entityId, {
            entityId: entityId,
            status: RLU.STATUS_HOT,
            strength: 1.0,
            age: 0,
            residualCost: costData.residual || AXIOM.COST_RESIDUAL,
            reclaimedCost: 0,
            position: costData.position || { x: 0, y: 0, z: 0 }
        });
    }

    reenter(entityId, newCostData) {
        let _entry = this.m_entries.get(entityId);
        if (!_entry) { this.admit(entityId, newCostData); return; }
        _entry.status = RLU.STATUS_HOT;
        _entry.strength = 1.0;
        _entry.age = 0;
        _entry.residualCost = newCostData.residual || AXIOM.COST_RESIDUAL;
        _entry.reclaimedCost = 0;
        _entry.position = newCostData.position || _entry.position;
    }

    tick(currentTick) {
        let _evicted = [];
        let _tickReclaim = 0;

        for (let [_id, _entry] of this.m_entries) {
            _entry.age++;
            // 등비급수 감쇠: 매 틱 같은 비율을 곱한다
            _entry.strength *= RLU.DECAY_RATIO;

            let _reclaimAmount = _entry.residualCost / RLU.BALL_LIFE;
            _entry.reclaimedCost += _reclaimAmount;
            _tickReclaim += _reclaimAmount;

            if (_entry.strength <= RLU.DECAY_THRESHOLD) {
                _evicted.push({
                    entityId: _id,
                    reclaimedCost: _entry.reclaimedCost,
                    position: _entry.position
                });
            }
            // 등비급수 감쇠 경계 (문턱 4/13, 1000틱 수명):
            // t=50 (5%):  strength = 0.943 → HOT→WARM
            // t=320 (32%): strength = 0.686 → WARM→COLD
            else if (_entry.strength < 0.686) {
                _entry.status = RLU.STATUS_COLD;
            }
            else if (_entry.strength < 0.943) {
                _entry.status = RLU.STATUS_WARM;
            }
        }

        for (let _evict of _evicted) {
            this.m_entries.delete(_evict.entityId);
            this.m_totalReclaimed += _evict.reclaimedCost;
            this.m_remnants.set(_evict.entityId, {
                entityId: _evict.entityId,
                position: _evict.position,
                status: RLU.STATUS_REMNANT,
                remnantAge: 0
            });
        }

        return { evicted: _evicted, tickReclaim: _tickReclaim };
    }

    removeRemnant(entityId) { this.m_remnants.delete(entityId); }

    getDistance(entityIdA, entityIdB) {
        let _a = this.m_entries.get(entityIdA) || this.m_remnants.get(entityIdA);
        let _b = this.m_entries.get(entityIdB) || this.m_remnants.get(entityIdB);
        if (!_a || !_b) { return this.m_ringSize; }
        let _dx = _a.position.x - _b.position.x;
        let _dy = _a.position.y - _b.position.y;
        let _dz = _a.position.z - _b.position.z;
        return Math.max(1, Math.round(Math.sqrt(_dx*_dx + _dy*_dy + _dz*_dz))) % this.m_ringSize;
    }

    getOverlapRatio(entityIdA, entityIdB) {
        return 1.0 - (this.getDistance(entityIdA, entityIdB) / this.m_ringSize);
    }

    getInteractionStrength(entityIdA, entityIdB, stageCost) {
        let _l = this.getDistance(entityIdA, entityIdB);
        if (_l === 0) { _l = 1; }
        let _C = stageCost || 1;
        return (_C * (1.0 - _l / this.m_ringSize)) / (4.0 * Math.PI * _l * _l);
    }

    getEntry(entityId) { return this.m_entries.get(entityId) || null; }
    getAllEntries() { return Array.from(this.m_entries.values()); }
    getRemnants() { return Array.from(this.m_remnants.values()); }

    getStatusCounts() {
        let _c = { HOT: 0, WARM: 0, COLD: 0, REMNANT: this.m_remnants.size };
        for (let [, _e] of this.m_entries) { _c[_e.status]++; }
        return _c;
    }

    snapshot() {
        return {
            ringSize: this.m_ringSize,
            entryCount: this.m_entries.size,
            remnantCount: this.m_remnants.size,
            statusCounts: this.getStatusCounts(),
            totalReclaimed: Math.round(this.m_totalReclaimed * 100) / 100,
            constants: { maxLife: RLU.MAX_LIFE, ballLife: RLU.BALL_LIFE, remnantLife: RLU.REMNANT_LIFE, decayRatio: RLU.DECAY_RATIO },
            entries: this.getAllEntries().map(_e => ({
                entityId: _e.entityId, status: _e.status,
                strength: Math.round(_e.strength * 1000) / 1000,
                age: _e.age, residualCost: _e.residualCost,
                reclaimedCost: Math.round(_e.reclaimedCost * 100) / 100,
                position: _e.position
            }))
        };
    }

    reset() { this.m_entries.clear(); this.m_remnants.clear(); this.m_totalReclaimed = 0; }
}

export { RLU };
