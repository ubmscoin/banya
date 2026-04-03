// 반야프레임 Observer: 구면 위 원형 포커스
// 공리 10: delta(전체)가 observer(국소)를 통해 자기 자신에 접근
// 공리 11: delta 1개가 다수의 observer를 통해 다수의 엔티티를 생성
// 공리 15: observer = 진입점(bit 0). 파이프라인의 시작
//
// v0.2: observer는 구면 위의 원형 영역(spherical cap)이다
//   구면에 포커스를 대면 그 부분만 필터링되어 시공에 표현된다
//   포커스 밖은 시공간에 없다 = 관측되지 않은 것은 렌더링되지 않는다

import { OBSERVER } from '../core/constants.js';

class Observer {
    static nextId = 0;
    static HOT_AREA_RATIO = OBSERVER.HOT_AREA_RATIO;
    static FOCUS_RADIUS = OBSERVER.FOCUS_RADIUS;

    constructor(focusTheta, focusPhi) {
        this.m_id = Observer.nextId++;
        this.m_name = `observer_${this.m_id}`;

        // 구면 위 포커스 중심 (구면좌표)
        this.m_focusTheta = (focusTheta !== undefined) ? focusTheta : Math.PI / 2;
        this.m_focusPhi = (focusPhi !== undefined) ? focusPhi : 0;

        // 포커스 반지름: 공리에서 도출. HOT 5% 구면 캡
        // 사용자 조절 불가. 공리가 결정한 값
        this.m_focusRadius = Observer.FOCUS_RADIUS;

        // 공리 12: observer = 엔티티의 그림자를 만드는 필터
        this.m_entities = new Map();

        this.m_active = true;
        this.m_filterCount = 0;
    }

    // 엔티티가 포커스 안에 있는지 판정
    // 구면 위 대원거리(great-circle distance)로 비교
    // 공리 11: observer마다 독립적으로 필터링 = 구면 위 서로 다른 영역
    isInFocus(entity) {
        let _sph = entity.getSphericalCoords();
        if (_sph.r < 0.001) {
            return true;  // 원점의 엔티티는 항상 포커스 안
        }

        // 대원거리: arccos(sin(t1)*sin(t2)*cos(p1-p2) + cos(t1)*cos(t2))
        let _cosD = Math.sin(this.m_focusTheta) * Math.sin(_sph.theta)
                   * Math.cos(this.m_focusPhi - _sph.phi)
                   + Math.cos(this.m_focusTheta) * Math.cos(_sph.theta);

        // 부동소수점 보정
        _cosD = Math.max(-1, Math.min(1, _cosD));
        let _distance = Math.acos(_cosD);

        return _distance <= this.m_focusRadius;
    }

    // delta 투영을 받아 필터링
    // 공리 10: delta -> observer -> CAS -> 결과 -> delta
    // 공리 15 명제: observer 필터링 자체는 비용 0
    filter(deltaState, domainBits) {
        if (!this.m_active) {
            return null;
        }

        this.m_filterCount++;

        return {
            observerId: this.m_id,
            domainPattern: domainBits,
            focusTheta: this.m_focusTheta,
            focusPhi: this.m_focusPhi,
            focusRadius: this.m_focusRadius,
            deltaActive: deltaState === 1
        };
    }

    // 포커스 이동 (마우스 드래그)
    moveFocus(theta, phi) {
        this.m_focusTheta = Math.max(0, Math.min(Math.PI, theta));
        this.m_focusPhi = phi % (2 * Math.PI);
        if (this.m_focusPhi < 0) {
            this.m_focusPhi += 2 * Math.PI;
        }
    }

    // 포커스 크기는 공리가 결정한다. 사용자가 바꿀 수 없다
    // 공리 6, 15: HOT 5% = arccos(0.90) = 0.451 rad

    // 엔티티 등록
    registerEntity(entity) {
        this.m_entities.set(entity.m_id, entity);
    }

    // 엔티티 제거
    removeEntity(entityId) {
        this.m_entities.delete(entityId);
    }

    // 포커스 안 엔티티만 반환
    getFocusedEntities() {
        let _focused = [];
        for (let [, _entity] of this.m_entities) {
            if (_entity.m_alive && this.isInFocus(_entity)) {
                _focused.push(_entity);
            }
        }
        return _focused;
    }

    // 전체 엔티티 (수축 겹침 계산용: 잔해 포함)
    getAllEntities() {
        return Array.from(this.m_entities.values());
    }

    // 스냅샷
    snapshot() {
        let _focused = this.getFocusedEntities();
        return {
            id: this.m_id,
            name: this.m_name,
            focus: {
                theta: Math.round(this.m_focusTheta * 1000) / 1000,
                phi: Math.round(this.m_focusPhi * 1000) / 1000,
                radius: Math.round(this.m_focusRadius * 1000) / 1000
            },
            active: this.m_active,
            entityCount: this.m_entities.size,
            focusedCount: _focused.length,
            filterCount: this.m_filterCount
        };
    }
}

export { Observer };
