// 반야프레임 UI 렌더러
import { AXIOM, COLOR, RENDER, GRID } from '../core/constants.js';

class Renderer {
    constructor() {
        this.m_elements = {};
        this.m_ctx = null;
        this.m_canvasW = 0;
        this.m_canvasH = 0;
        this.m_initialized = false;
        this.m_rotX = 0.4;
        this.m_rotY = 0.0;
        this.m_showPlayButton = true;
        this.m_debugMode = true;
    }

    init() {
        let _canvas = document.getElementById('sphere-canvas');
        if (_canvas) {
            this.m_ctx = _canvas.getContext('2d');
            this.m_canvasW = _canvas.width;
            this.m_canvasH = _canvas.height;
            _canvas.addEventListener('wheel', (e) => e.preventDefault());
            _canvas.addEventListener('click', (e) => {
                // 체크박스 영역 클릭 확인
                if (this.m_hudCheckbox) {
                    let _rect = _canvas.getBoundingClientRect();
                    let _mx = (e.clientX - _rect.left) * (_canvas.width / _rect.width);
                    let _my = (e.clientY - _rect.top) * (_canvas.height / _rect.height);
                    let _cb = this.m_hudCheckbox;
                    if (_mx >= _cb.x && _mx <= _cb.x + _cb.w && _my >= _cb.y && _my <= _cb.y + _cb.h) {
                        this.m_debugMode = !this.m_debugMode;
                        if (window.setDebug) { window.setDebug(this.m_debugMode); }
                        return;
                    }
                }
                if (this.m_showPlayButton) {
                    this.m_showPlayButton = false;
                    if (this.m_onPlayClick) { this.m_onPlayClick(); }
                }
            });
            // 마우스 드래그: 구면 회전 (모바일 터치도 대응)
            _canvas.addEventListener('mousedown', (e) => { this.m_dragging = true; this.m_dragX = e.clientX; this.m_dragY = e.clientY; });
            _canvas.addEventListener('mousemove', (e) => {
                if (!this.m_dragging) { return; }
                this.m_rotY -= (e.clientX - this.m_dragX) * 0.005;
                this.m_rotX += (e.clientY - this.m_dragY) * 0.005;
                this.m_rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.m_rotX));
                this.m_dragX = e.clientX;
                this.m_dragY = e.clientY;
                if (this.m_onDragRender) { this.m_onDragRender(); }
            });
            _canvas.addEventListener('mouseup', () => { this.m_dragging = false; });
            _canvas.addEventListener('mouseleave', () => { this.m_dragging = false; });
            // 터치 (모바일)
            _canvas.addEventListener('touchstart', (e) => { this.m_dragging = true; this.m_dragX = e.touches[0].clientX; this.m_dragY = e.touches[0].clientY; });
            _canvas.addEventListener('touchmove', (e) => {
                if (!this.m_dragging) { return; }
                this.m_rotY -= (e.touches[0].clientX - this.m_dragX) * 0.005;
                this.m_rotX += (e.touches[0].clientY - this.m_dragY) * 0.005;
                this.m_rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.m_rotX));
                this.m_dragX = e.touches[0].clientX;
                this.m_dragY = e.touches[0].clientY;
                e.preventDefault();
                if (this.m_onDragRender) { this.m_onDragRender(); }
            }, { passive: false });
            _canvas.addEventListener('touchend', () => { this.m_dragging = false; });
            document.addEventListener('keydown', (e) => this.p_onKeyDown(e));
        }

        this.m_elements = {
            dringBits: document.getElementById('dring-bits'),
            dringSeam: document.getElementById('dring-seam'),
            dringRaw: document.getElementById('dring-raw'),
            fsmStates: document.querySelectorAll('.fsm-state'),
            fsmNorm: document.getElementById('fsm-norm'),
            fsmCycles: document.getElementById('fsm-cycles'),
            pipelineStages: document.querySelectorAll('.pipeline-stage'),
            focusedList: document.getElementById('focused-list'),
            focusedSummary: document.getElementById('focused-summary'),
            costReadFill: document.getElementById('cost-read-fill'),
            costWriteFill: document.getElementById('cost-write-fill'),
            costMaintainFill: document.getElementById('cost-maintain-fill'),
            costResidualFill: document.getElementById('cost-residual-fill'),
            costReadVal: document.getElementById('cost-read-val'),
            costWriteVal: document.getElementById('cost-write-val'),
            costMaintainVal: document.getElementById('cost-maintain-val'),
            costResidualVal: document.getElementById('cost-residual-val'),
            costTotal: document.getElementById('cost-total'),
            costCycles: document.getElementById('cost-cycles'),
            budgetVal: document.getElementById('budget-val'),
            logEntries: document.getElementById('log-entries'),
            tickDisplay: document.getElementById('tick-display'),
            btnPlay: document.getElementById('btn-play')
        };
        this.m_initialized = true;
    }

    render(snapshot) {
        if (!this.m_initialized) { return; }
        this.p_renderSphere(snapshot);
        if (this.m_showPlayButton) {
            this.p_renderPlayOverlay();
        }
    }

    // 오버레이: 구면 위에 30% 투명 + 플레이 삼각형
    p_renderPlayOverlay() {
        let _ctx = this.m_ctx;
        if (!_ctx) { return; }
        let _w = this.m_canvasW, _h = this.m_canvasH;
        let _cx = _w / 2, _cy = _h / 2;

        // 반투명 오버레이 (30%)
        _ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        _ctx.fillRect(0, 0, _w, _h);

        // 플레이 버튼 원
        let _btnR = 50;
        _ctx.fillStyle = 'rgba(255,255,255,0.1)';
        _ctx.beginPath();
        _ctx.arc(_cx, _cy, _btnR, 0, Math.PI * 2);
        _ctx.fill();
        _ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        _ctx.lineWidth = 2;
        _ctx.beginPath();
        _ctx.arc(_cx, _cy, _btnR, 0, Math.PI * 2);
        _ctx.stroke();

        // 삼각형
        let _triSize = 25;
        _ctx.fillStyle = 'rgba(255,255,255,0.9)';
        _ctx.beginPath();
        _ctx.moveTo(_cx - _triSize * 0.4, _cy - _triSize);
        _ctx.lineTo(_cx - _triSize * 0.4, _cy + _triSize);
        _ctx.lineTo(_cx + _triSize * 0.8, _cy);
        _ctx.closePath();
        _ctx.fill();

        // 텍스트
        _ctx.fillStyle = '#ffffff';
        _ctx.font = 'bold 18px monospace';
        _ctx.textAlign = 'center';
        _ctx.fillText('\uBE45\uBC45', _cx, _cy + _btnR + 30);
        _ctx.font = '12px monospace';
        _ctx.fillStyle = '#ffffff';
        _ctx.fillText('\uBC18\uC57C\uC0CC\uB4DC\uBC15\uC2A4\uC18D\uC5D0\uC11C \uBE45\uBC45\uC774 \uC2DC\uC791\uB429\uB2C8\uB2E4', _cx, _cy + _btnR + 52);
        _ctx.textAlign = 'left';
    }

    p_renderSphere(snapshot) {
        let _ctx = this.m_ctx;
        if (!_ctx) { return; }
        let _w = this.m_canvasW, _h = this.m_canvasH;
        let _cx = _w / 2, _cy = _h / 2;
        let _sphereR = Math.min(_w, _h) * 0.40;

        _ctx.fillStyle = '#0a0e17';
        _ctx.fillRect(0, 0, _w, _h);

        // 그림자
        let _sg = _ctx.createRadialGradient(_cx, _cy + _sphereR * 0.95, 0, _cx, _cy + _sphereR * 0.95, _sphereR * 0.6);
        _sg.addColorStop(0, 'rgba(0,0,0,0.3)');
        _sg.addColorStop(1, 'rgba(0,0,0,0)');
        _ctx.fillStyle = _sg;
        _ctx.beginPath();
        _ctx.ellipse(_cx, _cy + _sphereR * 0.95, _sphereR * 0.6, _sphereR * 0.08, 0, 0, Math.PI * 2);
        _ctx.fill();

        // 구면 본체
        let _lx = _cx - _sphereR * 0.35, _ly = _cy - _sphereR * 0.35;
        let _bg = _ctx.createRadialGradient(_lx, _ly, _sphereR * 0.05, _cx, _cy, _sphereR);
        _bg.addColorStop(0, 'rgba(26,58,92,0.25)');
        _bg.addColorStop(0.4, 'rgba(15,38,64,0.2)');
        _bg.addColorStop(0.7, 'rgba(9,26,48,0.15)');
        _bg.addColorStop(1, 'rgba(4,13,26,0.1)');
        _ctx.fillStyle = _bg;
        _ctx.beginPath();
        _ctx.arc(_cx, _cy, _sphereR, 0, Math.PI * 2);
        _ctx.fill();

        // 림
        let _rg = _ctx.createRadialGradient(_cx, _cy, _sphereR - 2, _cx, _cy, _sphereR);
        _rg.addColorStop(0, 'rgba(30,58,95,0)');
        _rg.addColorStop(1, 'rgba(30,80,130,0.3)');
        _ctx.fillStyle = _rg;
        _ctx.beginPath();
        _ctx.arc(_cx, _cy, _sphereR, 0, Math.PI * 2);
        _ctx.fill();

        // 격자
        _ctx.lineWidth = 0.4;
        for (let _lat = -60; _lat <= 60; _lat += 30) {
            this.p_drawLatLine(_ctx, _cx, _cy, _sphereR, _lat * Math.PI / 180);
        }
        for (let _lon = 0; _lon < 360; _lon += 30) {
            this.p_drawLonLine(_ctx, _cx, _cy, _sphereR, _lon * Math.PI / 180);
        }

        // 법선 N극(빨강) S극(파랑) 고정
        let _nSurf = this.p_project(0, 0, 10, _sphereR, _cx, _cy);
        let _sSurf = this.p_project(Math.PI, 0, 10, _sphereR, _cx, _cy);
        let _nTip = this.p_project(0, 0, 10, _sphereR * 1.35, _cx, _cy);
        let _sTip = this.p_project(Math.PI, 0, 10, _sphereR * 1.35, _cx, _cy);

        if (_nSurf.depth > -0.3) {
            let _a = 0.4 + Math.max(0, _nSurf.depth) * 0.6;
            _ctx.strokeStyle = `rgba(239,68,68,${_a})`;
            _ctx.lineWidth = 3;
            _ctx.beginPath(); _ctx.moveTo(_nSurf.x, _nSurf.y); _ctx.lineTo(_nTip.x, _nTip.y); _ctx.stroke();
            _ctx.fillStyle = `rgba(239,68,68,${_a})`;
            _ctx.beginPath(); _ctx.arc(_nTip.x, _nTip.y, 5, 0, Math.PI * 2); _ctx.fill();
            _ctx.font = 'bold 11px monospace'; _ctx.fillText('N', _nTip.x + 8, _nTip.y + 4);
        }
        if (_sSurf.depth > -0.3) {
            let _a = 0.4 + Math.max(0, _sSurf.depth) * 0.6;
            _ctx.strokeStyle = `rgba(59,130,246,${_a})`;
            _ctx.lineWidth = 3;
            _ctx.beginPath(); _ctx.moveTo(_sSurf.x, _sSurf.y); _ctx.lineTo(_sTip.x, _sTip.y); _ctx.stroke();
            _ctx.fillStyle = `rgba(59,130,246,${_a})`;
            _ctx.beginPath(); _ctx.arc(_sTip.x, _sTip.y, 5, 0, Math.PI * 2); _ctx.fill();
            _ctx.font = 'bold 11px monospace'; _ctx.fillText('S', _sTip.x + 8, _sTip.y + 4);
        }

        // 공 렌더링
        let _balls = snapshot.balls || [];
        let _focusedIds = new Set((snapshot.focusedEntities || []).map(_e => _e.id));

        let _projected = [];
        for (let _b of _balls) {
            let _proj = this.p_project(_b.theta, _b.phi, 10, _sphereR, _cx, _cy);
            if (_proj.depth < -0.95) { continue; }
            _projected.push({ ball: _b, proj: _proj });
        }
        _projected.sort((a, b) => a.proj.depth - b.proj.depth);

        for (let _item of _projected) {
            let _b = _item.ball, _proj = _item.proj;
            let _br = Math.max(0.05, 0.3 + _proj.depth * 0.7);
            let _e = _b.entity;

            if (!_b.alive && _e) {
                // 잔해: 질량 감쇠에 따라 작아지다 소멸
                let _mass = _e.mass || 0;
                let _sz = Math.max(0.3, _mass * 0.5 * _br);
                _ctx.fillStyle = `rgba(140,160,255,${Math.max(0.1, 0.3 * _br * (_mass / 4))})`;
                _ctx.beginPath(); _ctx.arc(_proj.x, _proj.y, _sz, 0, Math.PI * 2); _ctx.fill();
            } else if (_e) {
                // 크기 = LRU strength (0~1) 직접 매핑
                let _strength = _e.shrinkRadius || 0;
                let _lruSize = Math.max(1, _strength * _strength * 12);

                let _r = 100, _g = 130, _b2 = 170;
                if (_e.lruStatus === 'HOT') { _r = 239; _g = 68; _b2 = 68; }
                else if (_e.lruStatus === 'WARM') { _r = 245; _g = 158; _b2 = 11; }
                else if (_e.lruStatus === 'COLD') { _r = 140; _g = 160; _b2 = 255; }
                else { _r = 34; _g = 211; _b2 = 238; }

                let _size = Math.max(1, _lruSize * _br);
                let _alpha = Math.max(0.05, 0.5 + _br * 0.5);
                _ctx.shadowColor = `rgb(${_r},${_g},${_b2})`;
                _ctx.shadowBlur = Math.max(0, _lruSize * _br * 0.8);
                _ctx.fillStyle = `rgba(${_r},${_g},${_b2},${_alpha * _br})`;
                _ctx.beginPath(); _ctx.arc(_proj.x, _proj.y, _size, 0, Math.PI * 2); _ctx.fill();
                _ctx.shadowBlur = 0;
            }
        }

        // 옵저버 원: N극 표면 고정. 흰색 원 + 흰색 라벨
        if (_nSurf.depth > -0.1) {
            let _fx = _nSurf.x, _fy = _nSurf.y;
            let _fR = _sphereR * 0.05;
            _ctx.fillStyle = 'rgba(255,255,255,0.06)';
            _ctx.beginPath(); _ctx.arc(_fx, _fy, _fR, 0, Math.PI * 2); _ctx.fill();
            _ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            _ctx.lineWidth = 2; _ctx.setLineDash([]);
            _ctx.beginPath(); _ctx.arc(_fx, _fy, _fR, 0, Math.PI * 2); _ctx.stroke();
            _ctx.fillStyle = '#ffffff';
            _ctx.font = 'bold 10px monospace';
            _ctx.fillText('\uC635\uC800\uBC84', _fx + _fR + 4, _fy + 3);
        }

        let _ecs = snapshot.ecs || {};
        _ctx.fillStyle = '#ffffff';
        _ctx.font = '10px monospace';

        // LRU 라벨
        _ctx.textAlign = 'center';
        _ctx.fillStyle = '#ffffff';
        _ctx.font = 'bold 13px monospace';
        _ctx.fillText('LRU \uBE44\uAC00\uC5ED\uC801 \uAC10\uC1E0', _cx, _cy + _sphereR + 24);
        // 반야식: 캔버스 너비에 맞게 폰트 조절. 파란색
        _ctx.fillStyle = '#3b82f6';
        let _banyaFont = Math.min(15, _w / 42);
        _ctx.font = 'bold ' + Math.round(_banyaFont) + 'px monospace';
        _ctx.fillText('\u03B4\u00B2 = (time+space)\u00B2 + (observer+superposition)\u00B2', _cx, _cy + _sphereR + 44);
        _ctx.textAlign = 'left';

        // 우측 상단 200x200: 엔티티 격자 (상호작용 세기 시각화)
        // 4x4 격자. 공이 생기면 격자가 수축한다
        // 공리 13: 수축 겹침비 = 1-l/N, 상호작용 세기 = C*(1-l/N)/(4*pi*l^2)
        // 격자: 항상 우상단. 캔버스 비례로 크기 조절
        // 벡터 격자: 우상단 고정. 좌측 HUD(약 300px)와 겹치지 않게 크기 조절
        let _hudRight = 310;  // 좌측 HUD가 차지하는 폭
        let _available = _w - _hudRight - 20;  // 격자가 쓸 수 있는 폭
        let _gridSize = Math.min(400, _available, _h * 0.45);
        _gridSize = Math.max(80, _gridSize);  // 최소 80px
        let _gridX = _w - _gridSize - 10;
        let _gridY = 30;
        this.p_renderEntityGrid(_ctx, _gridX, _gridY, _gridSize, snapshot);

        // 좌상단 HUD: 상시 표시. 체크 시에만 갱신(작동)
        this.p_renderHUD(_ctx, 10, 30, snapshot);
    }

    // 좌상단 캔버스 HUD
    // 상단 4개(d-ring,CAS,예산바,워크벤치): 체크 시에만 갱신
    // 하단 정보항목: 항상 갱신
    p_renderHUD(ctx, ox, oy, snapshot) {
        let _f = 'bold 10px monospace';       // 통일 폰트
        let _fc = '#ffffff';    // 라벨 색
        let _fv = '#ffffff';    // 값 색
        let _boxW = 28, _boxH = 20, _gap = 2;
        let _labelW = 60;
        let _barW = 8 * (_boxW + _gap);
        let _barH = 12;
        let _barX = ox + _labelW;
        let _lineH = 15;
        let _y = oy;

        // 상시 갱신. debugMode면 6서브스텝이라 점등 애니 보임
        let _dr = snapshot.dring || {};
        let _fsm = snapshot.fsm || {};

        // === D-RING ===
        ctx.fillStyle = _fc; ctx.font = _f;
        ctx.fillText('D-RING', ox, _y + 14);
        let _bits = (_dr.binary || '00000000').split('').reverse();
        let _names = ['OB','SP','T','S','R','C','S','D'];
        for (let i = 0; i < 8; i++) {
            let _on = _bits[i] === '1';
            let _bx = ox + _labelW + i * (_boxW + _gap);
            ctx.fillStyle = _on ? 'rgba(59,130,246,0.8)' : 'rgba(30,41,59,0.7)';
            ctx.fillRect(_bx, _y, _boxW, _boxH);
            ctx.strokeStyle = 'rgba(51,65,85,0.5)'; ctx.lineWidth = 0.5;
            ctx.strokeRect(_bx, _y, _boxW, _boxH);
            ctx.fillStyle = _on ? '#ffffff' : '#ffffff';
            ctx.font = _f; ctx.textAlign = 'center';
            ctx.fillText(_names[i], _bx + _boxW / 2, _y + 14);
        }
        ctx.textAlign = 'left';
        _y += _boxH + 8;

        // === CAS ===
        ctx.fillStyle = _fc; ctx.font = _f;
        ctx.fillText('CAS', ox, _y + 14);
        let _fsmS = ['000','001','011','111'], _fsmV = [0,1,3,7];
        for (let i = 0; i < 4; i++) {
            let _cur = _fsm.state === _fsmV[i];
            let _bx = ox + _labelW + i * (_boxW + _gap);
            ctx.fillStyle = _cur ? 'rgba(59,130,246,0.8)' : 'rgba(30,41,59,0.7)';
            ctx.fillRect(_bx, _y, _boxW, _boxH);
            ctx.strokeStyle = 'rgba(51,65,85,0.5)'; ctx.lineWidth = 0.5;
            ctx.strokeRect(_bx, _y, _boxW, _boxH);
            ctx.fillStyle = _cur ? '#ffffff' : '#ffffff';
            ctx.font = _f; ctx.textAlign = 'center';
            ctx.fillText(_fsmS[i], _bx + _boxW / 2, _y + 14);
        }
        ctx.textAlign = 'left';
        _y += _boxH + 8;

        // === 예산 바 ===
        let _budgetVal = Math.min(13, (snapshot.ecs ? snapshot.ecs.budget || 0 : 0));
        ctx.fillStyle = _fc; ctx.font = _f;
        ctx.fillText('\uBE44\uC6A9 \uD68C\uC218', ox, _y + 9);
        ctx.fillStyle = 'rgba(239,68,68,0.4)'; ctx.fillRect(_barX, _y, _barW, _barH);
        ctx.fillStyle = 'rgba(245,158,11,0.8)'; ctx.fillRect(_barX, _y, _barW * (_budgetVal / 13), _barH);
        ctx.strokeStyle = 'rgba(51,65,85,0.5)'; ctx.lineWidth = 0.5; ctx.strokeRect(_barX, _y, _barW, _barH);
        ctx.fillStyle = '#ffffff'; ctx.font = _f; ctx.textAlign = 'center';
        ctx.fillText(_budgetVal.toFixed(1) + '/13', _barX + _barW / 2, _y + 9);
        ctx.textAlign = 'left';
        _y += _barH + 8;

        // === 워크벤치 ===
        let _wbFired = _dr.delta === 1;
        ctx.fillStyle = _fc; ctx.font = _f; ctx.fillText('\uC6CC\uD06C\uBCA4\uCE58', ox, _y + 9);
        ctx.fillStyle = _wbFired ? 'rgba(59,130,246,0.6)' : 'rgba(30,41,59,0.5)';
        ctx.fillRect(_barX, _y, _barW, _barH);
        ctx.strokeStyle = 'rgba(51,65,85,0.5)'; ctx.lineWidth = 0.5; ctx.strokeRect(_barX, _y, _barW, _barH);
        ctx.fillStyle = _wbFired ? '#ffffff' : '#ffffff';
        ctx.font = _f; ctx.textAlign = 'center'; ctx.fillText('137bit', _barX + _barW / 2, _y + 9); ctx.textAlign = 'left';
        _y += _barH + 10;

        // === 체크박스 ===
        let _chkSize = 10;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1; ctx.strokeRect(ox, _y, _chkSize, _chkSize);
        if (this.m_debugMode) { ctx.fillStyle = 'rgba(59,130,246,0.8)'; ctx.fillRect(ox + 2, _y + 2, _chkSize - 4, _chkSize - 4); }
        ctx.fillStyle = _fc; ctx.font = _f; ctx.fillText('\uB0B4\uBD80\uC791\uB3D9 \uBCF4\uAE30 (\uB290\uB824\uC9D0)', ox + _chkSize + 5, _y + 9);
        this.m_hudCheckbox = { x: ox, y: _y, w: _chkSize + 100, h: _chkSize + 4 };
        _y += 35;

        // 우주 총 예산 시작 Y 저장 (벡터 격자와 높이 맞춤)
        this.m_infoStartY = _y;

        // === 하단 정보 (항상 갱신, 체크 무관) ===
        let _ecs = snapshot.ecs || {};
        let _lru = _ecs.lru || {};
        let _lc = _lru.statusCounts || {};
        let _lruTotal = (_lc.HOT||0) + (_lc.WARM||0) + (_lc.COLD||0) + (_lc.REMNANT||0);

        let _vx = ox + _labelW + 30;
        ctx.font = _f;

        ctx.fillStyle = _fc; ctx.fillText('\uC6B0\uC8FC \uCD1D \uC608\uC0B0', ox, _y); ctx.fillStyle = _fv; ctx.fillText('' + Math.round(_ecs.budget || 0), _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('\uACF5 \uC218', ox, _y); ctx.fillStyle = _fv; ctx.fillText((_ecs.totalEntities||0) + ' / ' + _lruTotal, _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('LRU \uC218\uBA85', ox, _y); _y += _lineH;
        ctx.fillStyle = '#ef4444'; ctx.fillText('HOT 5%', ox, _y); ctx.fillStyle = _fv; ctx.fillText('' + (_lc.HOT||0), _vx, _y); _y += _lineH;
        ctx.fillStyle = '#f59e0b'; ctx.fillText('WARM 27%', ox, _y); ctx.fillStyle = _fv; ctx.fillText('' + (_lc.WARM||0), _vx, _y); _y += _lineH;
        ctx.fillStyle = '#8ca0ff'; ctx.fillText('COLD 68%', ox, _y); ctx.fillStyle = _fv; ctx.fillText('' + ((_lc.COLD||0)+(_lc.REMNANT||0)), _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('CAS \uC0AC\uC774\uD074', ox, _y); ctx.fillStyle = _fv; ctx.fillText('' + (_ecs.tickCount||0), _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('\uBE45\uBC45 \uC608\uC0B0', ox, _y); ctx.fillStyle = _fv; ctx.fillText('13*100=1300', _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('\uACF5 1\uAC1C \uBE44\uC6A9', ox, _y); ctx.fillStyle = _fv; ctx.fillText('13', _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('HOT \uBA74\uC801', ox, _y); ctx.fillStyle = _fv; ctx.fillText('5%', _vx, _y); _y += _lineH;
        ctx.fillStyle = _fc; ctx.fillText('FOCUS \uBC18\uC9C0\uB984', ox, _y); ctx.fillStyle = _fv; ctx.fillText('0.451 rad', _vx, _y);
    }

    // 엔티티 공간 격자 (Entity Space)
    //
    // 이 격자가 표현하는 것:
    //   반야식의 고전 괄호(time+space) = DATA = 이산 공간을 2D 격자로 시각화
    //   공(엔티티)은 격자 위 고유 위치에서 태어나고, 제자리에서 감쇠하고, 증발한다
    //   이동하지 않는다. 이동은 큰 구(비가역 구면)에서 표현한다
    //   여기서는 "공간이 어떻게 찌그러지는가"만 본다
    //
    // 격자 왜곡의 의미:
    //   공리 13: 엔티티 간 상호작용 세기 = C * (1-l/N) / (4*PI*l^2) * cost/13
    //   공이 있는 곳의 격자가 수축한다 = 그 공이 주변 공간을 끌어당기고 있다
    //   HOT(빨강) = 강한 수축 (C=3, 비용 높음)
    //   WARM(주황) = 중간 수축 (C=2)
    //   COLD(파랑) = 약한 수축 (C=1)
    //   잔해(흐린보라) = 질량 비례 미약한 수축 (질량만 남았지만 여전히 공간을 왜곡)
    //   비용(cost)이 클수록 왜곡 강함 = 질량이 큰 천체가 공간을 더 많이 휜다
    //
    // 격자 8x8 = 64칸. 각 칸이 이산 공간의 한 단위
    // 공리 3: DATA는 이산. 격자의 교차점이 이산 좌표
    p_renderEntityGrid(ctx, ox, oy, size, snapshot) {
        let _gridN = 8;
        let _cellSize = size / _gridN;
        let _gridScale = size / 400;  // 격자 축소 비율. 공 크기도 이 비율로
        let _balls = (snapshot.ecs && snapshot.ecs.entities) || [];
        let _remnants = (snapshot.ecs && snapshot.ecs.remnants) || [];

        // 배경: 어두운 공간
        ctx.fillStyle = 'rgba(10, 14, 23, 0.85)';
        ctx.fillRect(ox, oy, size, size);
        ctx.strokeStyle = 'rgba(30, 60, 100, 0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, size, size);

        // 격자점 9x9. 매 프레임 0에서 시작 = 복원력 자체
        let _pts = [];
        for (let _gy = 0; _gy <= _gridN; _gy++) {
            _pts[_gy] = [];
            for (let _gx = 0; _gx <= _gridN; _gx++) {
                _pts[_gy][_gx] = {
                    baseX: ox + _gx * _cellSize,
                    baseY: oy + _gy * _cellSize,
                    dx: 0,
                    dy: 0
                };
            }
        }

        // 공의 격자 위치 결정
        // phi 각도를 cos/sin으로 분해하고, id로 반지름을 분산시켜 격자 전체에 흩뿌린다
        // 같은 id는 같은 위치. 새 id(새 공)는 새 위치. 소멸 후 재생성은 다른 곳
        let _allForGrid = [];
        let _margin = _cellSize * 0.5;
        let _usableSize = size - _margin * 2;

        for (let _ball of _balls) {
            let _sph = _ball.spherical || { phi: 0 };
            let _id = _ball.id || 0;
            let _phi = _sph.phi;
            // id마다 다른 반지름: 중심~가장자리에 분산
            let _rFrac = ((_id * 17 + 7) % 10) / 10 * 0.35 + 0.1;
            // phi+id*0.7로 각도도 분산: 같은 phi여도 다른 위치
            let _bx = ox + _margin + _usableSize * (0.5 + Math.cos(_phi + _id * 0.7) * _rFrac);
            let _by = oy + _margin + _usableSize * (0.5 + Math.sin(_phi + _id * 0.7) * _rFrac);
            _allForGrid.push({ bx: _bx, by: _by, ball: _ball, alive: true });
        }
        // 잔해는 격자에 표시하지 않는다. 큰 구에서만 표현

        // 격자 수축 계산
        // 공리 13 명제: 상호작용 세기 = C * (1-l/N) / (4*PI*l^2) * strength
        //
        // C = CAS 단계 비용 * 남은 수명 비율 (연속 감쇠: 3->0)
        // (1-l/N) = 수축 겹침비. l이 작을수록 겹침이 크다
        // 1/(4*PI*l^2) = 구면 분배. 역제곱 감쇠
        // strength = 남은 수명 비율. 1.0(생성)->0(소멸). shrinkRadius/sqrt(13)
        //   수명이 줄면 힘도 줄고, 격자 왜곡도 줄어든다 (복원)
        // 인력 규칙:
        //   공 1개(strength=1.0) = 자기 위치의 격자 꼭지점 4개를 자기 위치로 모으는 힘
        //   strength가 줄면 꼭지점이 원래 자리로 돌아감 (복원)
        //   공 2개가 겹쳐도 1점 이상 못 줄어듦 (겹칠 뿐)
        //
        // 구현: 각 격자점의 목표 변위 = 모든 공의 인력 합산 (strength 비례)
        //       현재 변위가 목표에 부드럽게 수렴

        for (let _gy = 0; _gy <= _gridN; _gy++) {
            for (let _gx = 0; _gx <= _gridN; _gx++) {
                let _pt = _pts[_gy][_gx];

                // 목표 변위: 모든 공의 인력 합산
                let _targetDx = 0;
                let _targetDy = 0;

                for (let _item of _allForGrid) {
                    let _strength = _item.ball.shrinkRadius || 0;
                    let _dx = _item.bx - _pt.baseX;
                    let _dy = _item.by - _pt.baseY;
                    let _l = Math.max(1, Math.sqrt(_dx * _dx + _dy * _dy));

                    // 이웃 1칸(d=1) + strength 1.0 = 인력 1 (기준)
                    // 거리: 연속. d = px거리 / 셀크기
                    // 인력 = strength / d^2. d=1,str=1 -> 1
                    let _dCell = _l / _cellSize;  // 연속 거리 (셀 단위)
                    let _attract = _strength / (_dCell * _dCell + 1);

                    // 방향(정규화) * 인력 * 셀크기 = 픽셀 변위
                    _targetDx += (_dx / _l) * _attract * _cellSize;
                    _targetDy += (_dy / _l) * _attract * _cellSize;
                }

                // 변위 정규화: 최대 인력 = 1셀 길이
                // 0(점만) ~ cellSize(최대 화살표)
                let _td = Math.sqrt(_targetDx * _targetDx + _targetDy * _targetDy);
                let _maxLen = _cellSize * 0.5;
                if (_td > _maxLen) {
                    _targetDx = _targetDx / _td * _maxLen;
                    _targetDy = _targetDy / _td * _maxLen;
                }
                _pt.dx = _targetDx;
                _pt.dy = _targetDy;
            }
        }

        // 벡터 필드: 길이 = 인력 합의 크기. 상쇄되면 점만. 강하면 길게
        for (let _gy = 0; _gy <= _gridN; _gy++) {
            for (let _gx = 0; _gx <= _gridN; _gx++) {
                let _pt = _pts[_gy][_gx];
                let _ox = _pt.baseX;
                let _oy = _pt.baseY;
                let _tx = _ox + _pt.dx;
                let _ty = _oy + _pt.dy;
                let _d = Math.sqrt(_pt.dx * _pt.dx + _pt.dy * _pt.dy);

                if (_d < 0.3) {
                    // 인력 합 = 0 (상쇄 또는 공 없음): 점만
                    ctx.fillStyle = 'rgba(100, 110, 130, 0.2)';
                    ctx.beginPath();
                    ctx.arc(_ox, _oy, 1, 0, Math.PI * 2);
                    ctx.fill();
                }
                else {
                    // 인력 합 > 0: 화살표. 길이 = 힘의 크기
                    let _alpha = Math.min(0.8, 0.2 + _d / (_cellSize * 0.3) * 0.4);

                    // 점
                    ctx.fillStyle = `rgba(100, 110, 130, ${_alpha})`;
                    ctx.beginPath();
                    ctx.arc(_ox, _oy, 1.5, 0, Math.PI * 2);
                    ctx.fill();

                    // 화살표
                    ctx.strokeStyle = `rgba(100, 110, 130, ${_alpha})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(_ox, _oy);
                    ctx.lineTo(_tx, _ty);
                    ctx.stroke();

                    // 화살촉
                    let _angle = Math.atan2(_pt.dy, _pt.dx);
                    let _headLen = Math.min(5, _d * 0.35);
                    ctx.beginPath();
                    ctx.moveTo(_tx, _ty);
                    ctx.lineTo(_tx - _headLen * Math.cos(_angle - 0.4), _ty - _headLen * Math.sin(_angle - 0.4));
                    ctx.moveTo(_tx, _ty);
                    ctx.lineTo(_tx - _headLen * Math.cos(_angle + 0.4), _ty - _headLen * Math.sin(_angle + 0.4));
                    ctx.stroke();
                }
            }
        }

        // 공 표시: 살아있는 공만. 제자리에서 크기와 색이 변하며 감쇠
        // CAS가 만드는 공은 항상 빨강(HOT)으로 시작 -> 주황(WARM) -> 파랑(COLD) -> 소멸
        for (let _item of _allForGrid) {
            let _b = _item.ball;
            let _r = 99, _g = 102, _b2 = 241;
            let _sz = 2;
            let _alpha = 0.3;

            if (_item.alive) {
                // 크기 = LRU strength 직접. 1.0->0
                let _strength = _b.shrinkRadius || 0;
                _sz = Math.max(1, _strength * _strength * 12 * _gridScale);
                _alpha = Math.max(0.2, _strength);

                // 색상은 LRU 상태별
                if (_b.lruStatus === 'HOT') { _r = 239; _g = 68; _b2 = 68; }
                else if (_b.lruStatus === 'WARM') { _r = 245; _g = 158; _b2 = 11; }
                else if (_b.lruStatus === 'COLD') { _r = 140; _g = 160; _b2 = 255; }
            }
            else {
                // 잔해: COLD 색상. 질량 비례 크기
                _r = 140; _g = 160; _b2 = 255;
                _sz = Math.max(0.5, (_b.mass || 0) / 4 * 3 * _gridScale);
                _alpha = Math.max(0.15, (_b.mass || 0) / 4 * 0.5);
            }

            // 글로우 효과: LRU 상태에 맞는 색으로 빛남
            ctx.shadowColor = `rgb(${_r},${_g},${_b2})`;
            ctx.shadowBlur = Math.max(0, _sz * 0.8);
            ctx.fillStyle = `rgba(${_r},${_g},${_b2},${_alpha})`;
            ctx.beginPath();
            ctx.arc(_item.bx, _item.by, _sz, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        // 하단 라벨 (구면 하단과 동일 형식)
        ctx.textAlign = 'center';
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 13px monospace';
        ctx.fillText('Entity \uACF5\uAC04 \uC0C1\uD638\uC791\uC6A9', ox + size / 2, oy + size + 20);
        ctx.textAlign = 'left';
    }

    p_drawLatLine(ctx, cx, cy, R, latRad) {
        let _pts = [];
        for (let _a = 0; _a < 360; _a += 3) {
            let _p = this.p_project(Math.PI / 2 - latRad, _a * Math.PI / 180, 10, R, cx, cy);
            if (_p.depth > -0.02) { _pts.push(_p); } else { if (_pts.length > 1) { this.p_strokePts(ctx, _pts); } _pts = []; }
        }
        if (_pts.length > 1) { this.p_strokePts(ctx, _pts); }
    }

    p_drawLonLine(ctx, cx, cy, R, lonRad) {
        let _pts = [];
        for (let _a = 0; _a < 360; _a += 3) {
            let _p = this.p_project(_a * Math.PI / 180, lonRad, 10, R, cx, cy);
            if (_p.depth > -0.02) { _pts.push(_p); } else { if (_pts.length > 1) { this.p_strokePts(ctx, _pts); } _pts = []; }
        }
        if (_pts.length > 1) { this.p_strokePts(ctx, _pts); }
    }

    p_strokePts(ctx, pts) {
        ctx.strokeStyle = 'rgba(30,60,100,0.25)';
        ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) { ctx.lineTo(pts[i].x, pts[i].y); }
        ctx.stroke();
    }

    p_project(theta, phi, r, sphereR, cx, cy) {
        let _nr = Math.min(r / 10, 1);
        let _x = _nr * Math.sin(theta) * Math.cos(phi);
        let _y = _nr * Math.sin(theta) * Math.sin(phi);
        let _z = _nr * Math.cos(theta);
        let _cRx = Math.cos(this.m_rotX), _sRx = Math.sin(this.m_rotX);
        let _cRy = Math.cos(this.m_rotY), _sRy = Math.sin(this.m_rotY);
        let _x2 = _x * _cRy - _z * _sRy;
        let _z2 = _x * _sRy + _z * _cRy;
        let _y2 = _y * _cRx - _z2 * _sRx;
        let _z3 = _y * _sRx + _z2 * _cRx;
        return { x: cx + _x2 * sphereR, y: cy - _y2 * sphereR, depth: _z3 };
    }

    // 키보드: 큰 구 회전
    p_onKeyDown(e) {
        let _step = 0.05;
        if (e.key === 'ArrowLeft') { this.m_rotY -= _step; }
        else if (e.key === 'ArrowRight') { this.m_rotY += _step; }
        else if (e.key === 'ArrowUp') { this.m_rotX = Math.max(-Math.PI / 2, this.m_rotX - _step); }
        else if (e.key === 'ArrowDown') { this.m_rotX = Math.min(Math.PI / 2, this.m_rotX + _step); }
        else { return; }
        e.preventDefault();
    }

    p_renderDRing(dring) {
        if (!this.m_elements.dringBits) { return; }
        let _names = ['ob','sp','time','spc','R','C','S','d'];
        let _bits = dring.binary.split('').reverse();
        let _html = '<div class="dring-ring"><div class="dring-nibble-label">nibble 0</div>';
        for (let i = 0; i < 4; i++) { let _on = _bits[i]==='1'; _html += `<div class="dring-bit ${_on?'on':'off'}"><span class="bit-label">${_names[i]}</span><span class="bit-value">${_bits[i]}</span></div>`; }
        _html += '<div class="dring-nibble-label">nibble 1</div>';
        for (let i = 4; i < 8; i++) { let _on = _bits[i]==='1'; _html += `<div class="dring-bit ${_on?'on':'off'}"><span class="bit-label">${_names[i]}</span><span class="bit-value">${_bits[i]}</span></div>`; }
        _html += '</div>';
        this.m_elements.dringBits.innerHTML = _html;
        if (this.m_elements.dringRaw) { this.m_elements.dringRaw.textContent = `raw: 0x${dring.raw.toString(16).padStart(2,'0')} = ${dring.binary}`; }
    }

    p_renderFSM(fsm) {
        let _map = {0:0,1:1,3:2,7:3};
        let _idx = _map[fsm.state] ?? 0;
        if (this.m_elements.fsmStates) { this.m_elements.fsmStates.forEach((_el,_i) => _el.classList.toggle('current', _i===_idx)); }
        if (this.m_elements.fsmNorm) { this.m_elements.fsmNorm.textContent = fsm.norm.toFixed(3); }
        if (this.m_elements.fsmCycles) { this.m_elements.fsmCycles.textContent = fsm.cycleCount; }
    }

    p_renderPipeline(stage) {
        let _order = ['idle','trigger','filter','update','render','screen'];
        let _idx = _order.indexOf(stage);
        if (this.m_elements.pipelineStages) { this.m_elements.pipelineStages.forEach((_el,_i) => { _el.classList.remove('current','done'); if(_i===_idx){_el.classList.add('current');}else if(_i<_idx){_el.classList.add('done');} }); }
    }

    p_renderFocused(focusedEntities, ecs) {
        if (!this.m_elements.focusedList) { return; }
        let _ents = focusedEntities || [];
        let _html = '';
        if (_ents.length === 0) { _html = '<div style="color:var(--text-dim);padding:8px">no entities in focus</div>'; }
        else { for (let _e of _ents.slice(-20)) { let _sc = _e.lruStatus||'HOT'; _html += `<div class="entity-item"><span class="entity-id">#${_e.id}</span><span class="entity-observer">obs:${_e.observerId}</span><span class="entity-status ${_sc}">${_sc}</span><span class="entity-pos">(${_e.position.x},${_e.position.y},${_e.position.z})</span><span class="entity-cost">m:${_e.mass}</span></div>`; } }
        this.m_elements.focusedList.innerHTML = _html;
        if (this.m_elements.focusedSummary) { let _l = ecs.lru||{}, _c = _l.statusCounts||{}; this.m_elements.focusedSummary.textContent = `focused:${_ents.length} total:${ecs.totalEntities} rem:${ecs.remnantCount} HOT:${_c.HOT||0} WARM:${_c.WARM||0} COLD:${_c.COLD||0}`; }
    }

    p_renderCost(cost, budget) {
        let _l = cost.recentHistory && cost.recentHistory.length > 0 ? cost.recentHistory[cost.recentHistory.length-1] : null;
        let _r=_l?_l.read.total:0, _w=_l?_l.write.total:0, _m=_l?_l.maintain:0, _re=_l?_l.residual:0;
        this.p_setBar('cost-read-fill','cost-read-val',_r,8);
        this.p_setBar('cost-write-fill','cost-write-val',_w,5);
        this.p_setBar('cost-maintain-fill','cost-maintain-val',_m,4);
        this.p_setBar('cost-residual-fill','cost-residual-val',_re,9);
        if (this.m_elements.costTotal) { this.m_elements.costTotal.textContent = cost.totalCost; }
        if (this.m_elements.costCycles) { this.m_elements.costCycles.textContent = cost.cycleCount; }
        if (this.m_elements.budgetVal) { this.m_elements.budgetVal.textContent = budget || 0; }
    }

    p_setBar(fId, vId, val, max) {
        let _f = document.getElementById(fId), _v = document.getElementById(vId);
        if (_f) { _f.style.width = `${(val/max)*100}%`; }
        if (_v) { _v.textContent = val; }
    }

    p_renderLog(log) {
        if (!this.m_elements.logEntries) { return; }
        let _html = '';
        for (let _e of log.slice(-30).reverse()) { _html += `<div class="log-entry ${_e.type}">${_e.message}</div>`; }
        this.m_elements.logEntries.innerHTML = _html;
    }

    p_renderToolbar(snapshot) {
        if (this.m_elements.tickDisplay) { this.m_elements.tickDisplay.textContent = `\uD2F1:${snapshot.tick} \uC608\uC0B0:${snapshot.budget}`; }
        if (this.m_elements.btnPlay) { this.m_elements.btnPlay.classList.toggle('active', snapshot.running); this.m_elements.btnPlay.textContent = snapshot.running ? '\uC77C\uC2DC\uC815\uC9C0' : '\uC7AC\uC0DD'; }
    }
}

export { Renderer };
