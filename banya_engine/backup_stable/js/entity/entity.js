// 반야프레임 Entity: observer 투영 결과
import { AXIOM, RLU_CONST } from '../core/constants.js';

class Entity {
    static nextId = 0;
    static MAINTAIN_COST = AXIOM.COST_MAINTAIN;

    constructor(observerId, domainPattern, position) {
        this.m_id = Entity.nextId++;
        this.m_observerId = observerId;
        this.m_position = { x: position ? position.x : 0, y: position ? position.y : 0, z: position ? position.z : 0 };
        this.m_domainPattern = domainPattern;
        this.m_data = { time: 0, space: new Int32Array(3) };
        this.m_operator = { superposition: [], collapsed: false };
        this.m_ballValue = null;
        this.m_mass = 0;
        this.m_shrinkRadius = 0;
        this.m_totalCost = 0;
        this.m_rluStatus = 'HOT';
        this.m_createdTick = 0;
        this.m_lastUpdateTick = 0;
        this.m_alive = true;
        this.m_gridCol = Math.random();
        this.m_gridRow = Math.random();
    }

    writeData(juim, tick) {
        this.m_data.time = tick;
        this.m_data.space[0] = this.m_position.x;
        this.m_data.space[1] = this.m_position.y;
        this.m_data.space[2] = this.m_position.z;
        this.m_operator.collapsed = true;
        this.m_operator.superposition = [juim.value];
        this.m_ballValue = juim.value;
        this.m_lastUpdateTick = tick;
    }

    addSuperposition(state) {
        if (!this.m_operator.collapsed) { this.m_operator.superposition.push(state); }
    }

    move(newPosition, newJuim, tick) {
        let _old = { ...this.m_position };
        this.m_position = { x: newPosition.x, y: newPosition.y, z: newPosition.z };
        this.writeData(newJuim, tick);
        return { destroyed: _old, created: { ...this.m_position } };
    }

    updateShrinkRadius(cost) {
        this.m_shrinkRadius = Math.sqrt(cost);
        this.m_totalCost += cost;
        this.m_mass += cost;
    }

    getSphericalCoords() {
        let _x = this.m_position.x, _y = this.m_position.y, _z = this.m_position.z;
        let _r = Math.sqrt(_x*_x + _y*_y + _z*_z);
        let _theta = _r > 0 ? Math.acos(Math.max(-1, Math.min(1, _z / _r))) : 0;
        return { r: _r, theta: _theta, phi: Math.atan2(_y, _x) };
    }

    evict() {
        this.m_alive = false;
        this.m_ballValue = null;
        this.m_mass = Entity.MAINTAIN_COST;
        this.m_remnantInitMass = Entity.MAINTAIN_COST;
        this.m_data.time = 0;
        this.m_data.space.fill(0);
        this.m_operator.superposition = [];
        this.m_operator.collapsed = false;
        return { entityId: this.m_id, position: { ...this.m_position }, returnedSpace: this.m_shrinkRadius, mass: this.m_mass, totalCost: this.m_totalCost };
    }

    isRemnant() { return !this.m_alive && this.m_mass > 0; }

    decayMass(remnantLife) {
        if (!this.m_alive && this.m_mass > 0) {
            let _init = this.m_remnantInitMass || this.m_mass;
            let _amount = _init / (remnantLife || RLU_CONST.REMNANT_LIFE);
            let _decayed = Math.min(_amount, this.m_mass);
            this.m_mass -= _decayed;
            this.m_shrinkRadius = Math.sqrt(Math.max(0, this.m_mass));
            return _decayed;
        }
        return 0;
    }

    isFullyDead() { return !this.m_alive && this.m_mass <= 0.01; }

    snapshot() {
        return {
            id: this.m_id, observerId: this.m_observerId,
            position: { ...this.m_position }, spherical: this.getSphericalCoords(),
            domainPattern: this.m_domainPattern,
            data: { time: this.m_data.time, space: Array.from(this.m_data.space) },
            operator: { superpositionCount: this.m_operator.superposition.length, collapsed: this.m_operator.collapsed },
            ballValue: this.m_ballValue,
            mass: Math.round(this.m_mass * 100) / 100,
            shrinkRadius: Math.round(this.m_shrinkRadius * 1000) / 1000,
            totalCost: this.m_totalCost, rluStatus: this.m_rluStatus,
            alive: this.m_alive, remnant: this.isRemnant(),
            gridCol: this.m_gridCol, gridRow: this.m_gridRow
        };
    }
}

export { Entity };
