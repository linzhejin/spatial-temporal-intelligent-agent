/* ========================================================================
   珞珈智行 · 实时导航引擎（navigation.js）
   高德同款"北朝上 + 指令卡 + 语音播报"沉浸导航，纯前端无额外依赖：
   · 蓝点投影到路线折线（GCJ-02 局部平面米制）→ 剩余里程 / 贴线距离
   · 偏航判定（步行20m/骑行30m/驾车35m，连续2点确认）→ 静默 /api/route 重算
   · 转向指令三级状态机：远距预报 → 临近(≤6m)动作令 → 到达(剩余≤12m)
   · 相邻 <15m 的连续拐点自动合并；地图北朝上跟随，拖拽后退出跟随
   · Wake Lock 屏幕常亮；10s 无定位回调上抛 GPS 弱网黄条
   坐标系：入参路线坐标/动作点均为 GCJ-02；GPS 推送同时带 WGS-84（重算用）
   ======================================================================== */
(function () {
    'use strict';

    // ---------- 模式参数 ----------
    var OFFROAD_M = { walk: 20, bike: 30, drive: 35 };           // 偏航阈值
    var FAR_M = { walk: 25, bike: 50, drive: 100 };             // 拐点远距预报
    var ARRIVE_FAR_M = { walk: 40, bike: 60, drive: 80 };       // 终点预报
    var VIA_FAR_M = { walk: 25, bike: 40, drive: 60 };          // 途经点预报
    var NEAR_M = 7;          // 临近动作令距离
    var ARRIVE_M = 12;       // 到达判定
    var MERGE_GAP_M = 15;    // 相邻拐点合并窗口
    var OFFROAD_CONFIRM = 2; // 连续偏航点数确认
    var REROUTE_COOLDOWN = 30000;
    var STALE_MS = 10000;
    var MODE_SPEED_MS = { walk: 1.25, bike: 3.9, drive: 6.9 };

    // action → {中文口令, 箭头角度(北朝上顺时针), 图标类型}
    var ACTIONS = {
        depart:        { verb: '出发', angle: 0, icon: 'arrow' },
        straight:      { verb: '继续直行', angle: 0, icon: 'arrow' },
        slight_left:   { verb: '稍向左转', angle: -30, icon: 'arrow' },
        left:          { verb: '左转', angle: -90, icon: 'arrow' },
        sharp_left:    { verb: '急向左转', angle: -135, icon: 'arrow' },
        slight_right:  { verb: '稍向右转', angle: 30, icon: 'arrow' },
        right:         { verb: '右转', angle: 90, icon: 'arrow' },
        sharp_right:   { verb: '急向右转', angle: 135, icon: 'arrow' },
        uturn:         { verb: '掉头', angle: 180, icon: 'arrow' },
        via:           { verb: '到达途经点', angle: 0, icon: 'via' },
        arrive:        { verb: '到达终点', angle: 0, icon: 'arrive' },
    };

    var nav = null;
    var gpsTimer = null;

    function haversine(lat1, lng1, lat2, lng2) {
        var R = 6371000;
        var toRad = Math.PI / 180;
        var dLat = (lat2 - lat1) * toRad;
        var dLng = (lng2 - lng1) * toRad;
        var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) *
            Math.sin(dLng / 2) * Math.sin(dLng / 2);
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    function humanDist(m) {
        if (m < 15) return '马上';
        if (m < 100) return Math.round(m / 5) * 5 + ' 米';
        if (m < 1000) return Math.round(m / 10) * 10 + ' 米';
        return (m / 1000).toFixed(1) + ' 公里';
    }

    // 最短角差（a-b 归一到 -180..180]，任意实数输入安全）
    function angDiff(a, b) {
        return (((a - b) % 360) + 540) % 360 - 180;
    }

    // 所在路段的方位角（0=北, 90=东；pts 为米制平面坐标 x=东 y=北）
    function segBearingAt(geo, segIdx) {
        if (!geo || !geo.pts || segIdx == null) return null;
        var i = Math.max(0, Math.min(geo.pts.length - 2, segIdx));
        var dx = geo.pts[i + 1].x - geo.pts[i].x;
        var dy = geo.pts[i + 1].y - geo.pts[i].y;
        if (Math.sqrt(dx * dx + dy * dy) < 2) return null;  // 过短路段方向不可信
        return (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
    }

    // ---------- 路线几何 ----------
    function buildGeometry(coords) {
        if (!coords || coords.length < 2) return null;
        var lat0 = coords[0].lat;
        var mPerLng = 111320 * Math.cos(lat0 * Math.PI / 180);
        var mPerLat = 111320;
        var pts = coords.map(function (c) {
            return { x: (c.lng - coords[0].lng) * mPerLng, y: (c.lat - lat0) * mPerLat, lng: c.lng, lat: c.lat };
        });
        var cum = [0];
        for (var i = 1; i < pts.length; i++) {
            var dx = pts[i].x - pts[i - 1].x, dy = pts[i].y - pts[i - 1].y;
            cum.push(cum[i - 1] + Math.sqrt(dx * dx + dy * dy));
        }
        return { pts: pts, cum: cum, total: cum[cum.length - 1], mPerLng: mPerLng, mPerLat: mPerLat, origin: coords[0] };
    }

    // 点 → 最近折线段，返回 {progress(沿线里程), cross(贴线垂直距离), lng, lat}
    function projectOn(geo, lng, lat) {
        var px = (lng - geo.origin.lng) * geo.mPerLng;
        var py = (lat - geo.origin.lat) * geo.mPerLat;
        var best = null;
        var pts = geo.pts, cum = geo.cum;
        for (var i = 0; i < pts.length - 1; i++) {
            var ax = pts[i].x, ay = pts[i].y;
            var bx = pts[i + 1].x, by = pts[i + 1].y;
            var abx = bx - ax, aby = by - ay;
            var apx2 = px - ax, apy2 = py - ay;
            var len2 = abx * abx + aby * aby || 1e-9;
            var t = (apx2 * abx + apy2 * aby) / len2;
            if (t < 0) t = 0; else if (t > 1) t = 1;
            var jx = ax + abx * t, jy = ay + aby * t;
            var cross = Math.sqrt((px - jx) * (px - jx) + (py - jy) * (py - jy));
            var along = cum[i] + t * (cum[i + 1] - cum[i]);
            if (!best || cross < best.cross) {
                best = { progress: along, cross: cross, t: t, seg: i };
            }
        }
        if (best.progress < 0) best.progress = 0;
        if (best.progress > geo.total) best.progress = geo.total;
        best.lng = lng;
        best.lat = lat;
        return best;
    }

    // ---------- 指令列表预处理（合并相邻 <15m 拐点） ----------
    function buildManeuvers(steps) {
        var list = (steps || []).map(function (s) {
            return Object.assign({}, s, { spokeFar: false, spokeNear: false, absorbed: false });
        });
        for (var i = 0; i < list.length - 1; i++) {
            var cur = list[i], nxt = list[i + 1];
            // 当前是拐点、下一条也是动作点（拐点/途经），间距过小 → 合并掉前者
            if (cur.type === 'turn' && (nxt.type === 'turn' || nxt.type === 'via') &&
                (nxt.cumulative_m - cur.cumulative_m) < MERGE_GAP_M) {
                cur.absorbed = true;
            }
        }
        return list;
    }

    function nextManeuver(navState, progress) {
        for (var i = 0; i < navState.maneuvers.length; i++) {
            var m = navState.maneuvers[i];
            if (m.absorbed) continue;
            if (m.cumulative_m - progress > -NEAR_M) {
                // depart 与第一拐点同点：拐点进入远距预报窗口后 depart 让位，
                // 否则第一条转向指令会被 depart 永久遮蔽（同 cumulative）
                if (m.type === 'depart') {
                    for (var j = i + 1; j < navState.maneuvers.length; j++) {
                        var n = navState.maneuvers[j];
                        if (n.absorbed) continue;
                        if (n.type === 'arrive') break;
                        if ((n.cumulative_m - m.cumulative_m) <= NEAR_M &&
                            (n.cumulative_m - progress) <= (FAR_M[navState.mode] || FAR_M.walk)) {
                            return { m: n, idx: j };
                        }
                        break;
                    }
                }
                return { m: m, idx: i };
            }
        }
        return null;
    }

    // ---------- 地图箭头标记（北朝上，箭头随 heading 旋转） ----------
    var ARROW_HTML = ''
        + '<div class="nav-arrow-marker" data-heading="0">'
        + '<svg width="30" height="30" viewBox="0 0 24 24">'
        + '<path d="M12 2 L17 20 L12 16 L7 20 Z"></path></svg></div>';

    function ensureArrowMarker(navState, gcjLat, gcjLng) {
        var map = navState.map;
        if (!navState.arrow) {
            navState.arrow = L.marker([gcjLat, gcjLng], {
                icon: L.divIcon({
                    className: 'nav-arrow-icon',
                    html: ARROW_HTML,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15],
                }),
                interactive: false,
                zIndexOffset: 2000,
            }).addTo(map);
        } else {
            navState.arrow.setLatLng([gcjLat, gcjLng]);
        }
    }

    function setArrowHeading(navState, deg) {
        if (!navState.arrow) return;
        var el = navState.arrow.getElement && navState.arrow.getElement();
        if (el) {
            var inner = el.querySelector('.nav-arrow-marker');
            if (inner) {
                inner.setAttribute('data-heading', Math.round(deg));
                inner.style.transform = 'rotate(' + deg + 'deg)';
            }
        }
    }

    // ---------- Wake Lock（浏览器 Screen Wake Lock；APK 内走原生常亮桥） ----------
    function acquireWake(navState) {
        if (window.WhuWalkerScreen && typeof window.WhuWalkerScreen.setKeepScreenOn === 'function') {
            try { window.WhuWalkerScreen.setKeepScreenOn(true); } catch (e) { /* ignore */ }
        }
        if (!('wakeLock' in navigator)) return;
        navigator.wakeLock.request('screen').then(function (wl) {
            navState.wakeLock = wl;
            wl.addEventListener('release', function () { navState.wakeLock = null; });
        }).catch(function () { /* 不支持/拒绝时静默 */ });
    }
    function onVisibility() {
        if (nav && !nav.arrived && document.visibilityState === 'visible') acquireWake(nav);
    }

    // ---------- 对外状态回调 ----------
    function emitCard(navState, m, distTo, remaining, speed) {
        var info = ACTIONS[m.action] || ACTIONS.straight;
        var sub = '';
        var main;
        if (m.type === 'arrive') {
            main = '到达终点' + (navState.endName ? ' · ' + navState.endName : '');
        } else if (m.type === 'via') {
            main = m.text || '到达途经点';
        } else if (m.type === 'depart') {
            main = '沿' + (m.road_name || '道路') + '前行';
        } else if (info.verb === '继续直行') {
            main = '继续直行';
            if (m.next_road_name) sub = '进入' + m.next_road_name;
        } else {
            main = info.verb;
            if (m.next_road_name) sub = '进入' + m.next_road_name;
        }
        var distText = '';
        if (m.type !== 'arrive') distText = humanDist(Math.max(0, distTo));
        navState._cb.onCard({
            icon: info.icon,
            angle: info.angle,
            main: main,
            sub: sub,
            maneuverDist: distText,
            remainingM: Math.max(0, Math.round(remaining)),
            remainingText: humanDist(Math.max(0, remaining)),
            remainingMin: Math.max(1, Math.round(remaining / speed / 60)),
            roadName: m.road_name || '',
        });
    }

    // ---------- 偏航静默重算 ----------
    function reroute(navState, fix) {
        if (navState.rerouting) return;
        if (!navState.request || !navState.request.end) return;
        if (Date.now() - navState.lastRerouteAt < REROUTE_COOLDOWN) return;
        navState.rerouting = true;
        navState.lastRerouteAt = Date.now();
        if (navState._cb.onToast) navState._cb.onToast('已偏航，正在重新规划');

        var body = {
            task_type: 'path_planning',
            start: {
                type: 'coord',
                coordinates: { lng: fix.lng, lat: fix.lat },
                name: '我的位置',
            },
            end: navState.request.end,
            constraints: navState.request.constraints || {},
            weights: navState.request.weights || null,
            travel_mode: navState.mode,
        };
        fetch('/api/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
            .then(function (res) {
                navState.rerouting = false;
                if (!nav || navState !== nav) return;
                if (!res.ok || !res.j || !res.j.data || !(res.j.data.recommended || []).length) {
                    if (navState._cb.onToast) navState._cb.onToast('重算失败，将继续按原路线导航');
                    return;
                }
                var data = res.j.data;
                navState.request.start = body.start;
                applyRoute(navState, data);
                if (navState._cb.renderRoute) navState._cb.renderRoute(data);
                if (window.WhuWalkerVoiceOutput && !window.WhuWalkerVoiceOutput.isMuted()) {
                    window.WhuWalkerVoiceOutput.speak('已为您重新规划路线');
                }
                if (navState._cb.onToast) navState._cb.onToast('路线已更新');
            })
            .catch(function () {
                navState.rerouting = false;
                if (nav && navState === nav && navState._cb.onToast) {
                    navState._cb.onToast('网络异常，暂时无法重新规划');
                }
            });
    }

    // 用新路线数据重建几何/指令（偏航重算、导航中换交通方式/新查询共用）
    function applyRoute(navState, routeData) {
        var geo = buildGeometry(routeData.recommended || []);
        if (!geo) return;
        navState.geo = geo;
        navState.maneuvers = buildManeuvers(routeData.steps || []);
        navState.mode = (routeData.mode || navState.mode);
        navState.offCount = 0;
        navState.progress = 0;
        navState.lastProjection = null;
        if (routeData.end && routeData.end.name) navState.endName = routeData.end.name;
        // 多模态路径：保存 legs 以便跨段切换 mode（每段 mode 不同，影响速度/偏航阈值）
        if (routeData.route_kind === 'multimodal' && routeData.legs && routeData.legs.length > 1) {
            navState.legs = routeData.legs;
            navState.legIdx = 0;
        } else {
            navState.legs = null;
            navState.legIdx = 0;
        }
    }

    // ---------- GPS 推送（app.js 蓝点更新时调用） ----------
    function onFix(loc) {
        if (!nav || nav.arrived || !loc) return;
        nav.lastFixAt = Date.now();
        if (nav._cb.onGps) nav._cb.onGps(false);

        var gcjLng = loc.gcjLng, gcjLat = loc.gcjLat;
        if (gcjLng == null || gcjLat == null) return;

        var proj = projectOn(nav.geo, gcjLng, gcjLat);

        // 速度 EMA（沿线进度增量 / 时间）
        var nowMs = Date.now();
        if (nav.lastProjTs && proj.progress > nav.lastProgress - 5) {
            var dt = (nowMs - nav.lastProjTs) / 1000;
            if (dt > 0.5 && dt < 6) {
                var v = (proj.progress - nav.lastProgress) / dt;
                if (v >= 0 && v < 30) nav.speed = nav.speed * 0.6 + v * 0.4;
            }
        }
        nav.lastProjTs = nowMs;
        nav.lastProgress = proj.progress;
        nav.lastProjection = proj;

        // 航向决策：
        //   A) 沿线行进中（贴线≤30m 且在移动）→ 箭头吸附到所在路段方位角，
        //      顺路即贴路，彻底消除罗盘磁偏/噪声带来的方向偏差；
        //      与设备航向夹角>100°（逆行/掉头中）不吸附，显示真实朝向
        //   B) 其余情况用设备航向（原生融合罗盘+GPS bearing / 浏览器 coords.heading）
        //   C) 无传感器时相邻点位移≥3m 用 atan2 推算兜底
        var deviceHeading = (loc.heading != null && isFinite(loc.heading)) ? loc.heading : null;
        var targetHeading = null;
        var snapAlpha = 0.5;
        var moving = (loc.speed != null && isFinite(loc.speed)) ? loc.speed >= 0.5
                   : (nav.lastProjTs > 0 && nav.speed >= 0.5);
        var onRoute = proj.cross <= 30 && proj.progress > 2 && proj.progress < nav.geo.total - 2;
        if (onRoute && moving) {
            var segBear = segBearingAt(nav.geo, proj.seg);
            if (segBear != null &&
                (deviceHeading == null || Math.abs(angDiff(segBear, deviceHeading)) <= 100)) {
                targetHeading = segBear;
                snapAlpha = 0.55;
            }
        }
        if (targetHeading == null && deviceHeading != null) targetHeading = deviceHeading;

        if (targetHeading != null) {
            if (nav.heading == null) {
                nav.heading = targetHeading;  // 首次直接采用，避免从北慢慢转过去
            } else {
                // 角度最短路径低通：抑制抖动，同时 0.5s 一次的更新下转向跟手
                var dh = angDiff(targetHeading, nav.heading);
                nav.heading = (nav.heading + dh * snapAlpha + 360) % 360;
            }
        } else if (nav.lastFixGcj) {
            var moved = haversine(nav.lastFixGcj.lat, nav.lastFixGcj.lng, gcjLat, gcjLng);
            if (moved >= 3) {
                var h = Math.atan2(
                    gcjLng - nav.lastFixGcj.lng,
                    gcjLat - nav.lastFixGcj.lat
                ) * 180 / Math.PI;  // 0=北, 90=东
                nav.heading = (h + 360) % 360;
            }
        }
        nav.lastFixGcj = { lng: gcjLng, lat: gcjLat, ts: Date.now() };

        // 地图箭头 + 北朝上跟随
        ensureArrowMarker(nav, gcjLat, gcjLng);
        // 显示角展平（不归一）：避免 359↔0 跨越正北时 CSS 过渡绕远路整圈打转
        var disp = 0;
        if (nav.heading != null) {
            if (nav.headingDisp == null) nav.headingDisp = nav.heading;
            else nav.headingDisp += angDiff(nav.heading, nav.headingDisp);
            disp = nav.headingDisp;
        }
        setArrowHeading(nav, disp);
        if (nav.follow) {
            if (nav.map.getZoom() < 16) {
                nav.map.setView([gcjLat, gcjLng], 17, { animate: true });
            } else {
                nav.map.panTo([gcjLat, gcjLng], { animate: true, duration: 0.4 });
            }
        }

        // 偏航判定：精度差(>100m)时只提示不强判；连续 2 点确认
        var threshold = OFFROAD_M[nav.mode] || OFFROAD_M.walk;
        var acc = loc.accuracy || 0;
        if (proj.cross > threshold && proj.progress > 30 && proj.progress < nav.geo.total - 30 &&
            (acc === 0 || acc <= 100)) {
            nav.offCount += 1;
            if (nav.offCount >= OFFROAD_CONFIRM) {
                nav.offCount = 0;
                reroute(nav, loc);
            }
        } else {
            nav.offCount = 0;
        }

        updateGuidance(nav, proj.progress);
    }

    function updateGuidance(navState, progress) {
        var remaining = navState.geo.total - progress;

        // 到达
        if (remaining <= ARRIVE_M && !navState.arrived) {
            finishArrive(navState);
            return;
        }

        var upcoming = nextManeuver(navState, progress);
        if (!upcoming) return;
        var m = upcoming.m;
        var distTo = m.cumulative_m - progress;
        var info = ACTIONS[m.action] || ACTIONS.straight;
        var V = window.WhuWalkerVoiceOutput;

        // 语音状态机
        if (m.type === 'arrive') {
            if (!m.spokeFar && remaining <= ARRIVE_FAR_M[navState.mode]) {
                m.spokeFar = true;
                if (V) V.speak('即将到达终点' + (navState.endName ? '，' + navState.endName : ''));
            }
            if (!m.spokeNear && remaining <= NEAR_M) {
                m.spokeNear = true;
                if (V) V.speak(m.text || '到达终点，导航结束');
                finishArrive(navState);
                return;
            }
        } else if (m.type === 'via') {
            if (!m.spokeFar && distTo <= VIA_FAR_M[navState.mode]) {
                m.spokeFar = true;
                if (V) V.speak(m.text || '即将到达途经点');
            }
            if (!m.spokeNear && distTo <= NEAR_M) {
                m.spokeNear = true;
                // 多模态换乘：跨 leg 时切换 mode + 重置偏航计数（避免 mode 阈值突变误触）
                if (navState.legs && navState.legs.length > 1 &&
                    navState.legIdx < navState.legs.length - 1) {
                    navState.legIdx += 1;
                    var nextLeg = navState.legs[navState.legIdx];
                    if (nextLeg && nextLeg.mode && nextLeg.mode !== navState.mode) {
                        navState.mode = nextLeg.mode;
                        navState.offCount = 0;
                        var modeLabel = { walk: '步行', bike: '骑行', drive: '驾车' }[nextLeg.mode] || nextLeg.mode;
                        if (V) V.speak('已到达换乘点，切换为' + modeLabel + '模式，继续前行');
                    } else {
                        if (V) V.speak('到达途经点，继续前行');
                    }
                } else {
                    if (V) V.speak('到达途经点，继续前行');
                }
            }
        } else if (m.type === 'turn' || m.type === 'depart' || m.type === 'straight' || ACTIONS[m.action]) {
            var farAt = FAR_M[navState.mode];
            if (!m.spokeFar && distTo <= farAt) {
                m.spokeFar = true;
                if (V) {
                    if (m.action === 'straight' || m.action === 'depart') {
                        V.speak(m.text || info.verb);
                    } else {
                        V.speak(humanDist(Math.max(0, distTo)) + '后' + info.verb +
                            (m.next_road_name ? '，进入' + m.next_road_name : ''));
                    }
                }
            }
            if (!m.spokeNear && distTo <= NEAR_M && m.type !== 'depart') {
                m.spokeNear = true;
                if (V && m.action !== 'straight') V.speak(info.verb);
            }
        }

        emitCard(navState, m, distTo, remaining, navState.speed);
    }

    function finishArrive(navState) {
        navState.arrived = true;
        navState.follow = false;
        var V = window.WhuWalkerVoiceOutput;
        var arrive = navState.maneuvers.filter(function (m) { return m.type === 'arrive'; })[0];
        if (V) V.speak((arrive && arrive.text) || '到达终点，导航结束');
        releaseWake(navState);
        if (navState._cb.onArrive) navState._cb.onArrive(navState.endName || '');
        navState._cb.onCard({
            icon: 'arrive', angle: 0,
            main: '到达终点' + (navState.endName ? ' · ' + navState.endName : ''),
            sub: '', maneuverDist: '',
            remainingM: 0, remainingText: '已到达', remainingMin: 0,
            roadName: '',
        });
    }

    function releaseWake(navState) {
        if (window.WhuWalkerScreen && typeof window.WhuWalkerScreen.setKeepScreenOn === 'function') {
            try { window.WhuWalkerScreen.setKeepScreenOn(false); } catch (e) { /* ignore */ }
        }
        if (navState.wakeLock) {
            try { navState.wakeLock.release(); } catch (e) { /* ignore */ }
            navState.wakeLock = null;
        }
    }

    // GPS 心跳：10s 没收到 fix → 上抛弱网状态
    function startGpsHeartbeat(navState) {
        stopGpsHeartbeat();
        gpsTimer = setInterval(function () {
            if (!nav || navState !== nav) { stopGpsHeartbeat(); return; }
            var stale = Date.now() - navState.lastFixAt > STALE_MS;
            if (stale !== navState._stale) {
                navState._stale = stale;
                if (navState._cb.onGps) navState._cb.onGps(stale);
            }
        }, 2000);
    }
    function stopGpsHeartbeat() {
        if (gpsTimer) { clearInterval(gpsTimer); gpsTimer = null; }
    }

    // ===================== 对外 API =====================
    var Navigation = {
        /**
         * @param {Object} opts
         *   map         Leaflet 地图实例
         *   routeData   /api/route|/chat 的路径响应（含 recommended/steps/mode）
         *   request     canonical route_state 投影（{start,end,constraints,weights}，偏航重算用）
         *   onCard(d)   顶部指令卡刷新
         *   onGps(stale) GPS 弱网黄条
         *   onArrive(endName)
         *   onToast(msg)
         *   renderRoute(data) 重算后让 app.js 重绘折线（导航态不 fitBounds）
         */
        start: function (opts) {
            this.stop();
            var map = opts.map;
            var geo = buildGeometry((opts.routeData && opts.routeData.recommended) || []);
            if (!map || !geo) return false;

            var endName = '';
            if (opts.request && opts.request.end) endName = opts.request.end.name || '';
            if (!endName && opts.routeData && opts.routeData.end) endName = opts.routeData.end.name || '';

            nav = {
                map: map,
                mode: opts.routeData.mode || 'walk',
                geo: geo,
                maneuvers: buildManeuvers(opts.routeData.steps || []),
                request: opts.request || { end: opts.routeData.end || null },
                endName: endName,
                arrow: null,
                follow: true,
                heading: null,  // 首个设备航向到来前不绘制方向（避免误导性地朝北）
                speed: MODE_SPEED_MS[opts.routeData.mode || 'walk'] || MODE_SPEED_MS.walk,
                progress: 0,
                offCount: 0,
                lastProgress: 0,
                lastProjTs: 0,
                lastRerouteAt: 0,
                rerouting: false,
                arrived: false,
                wakeLock: null,
                lastFixAt: Date.now(),
                lastFixGcj: null,
                lastProjection: null,
                _stale: false,
                _cb: opts,
            };

            // 拖拽地图 → 退出跟随
            nav._onDragStart = function () {
                if (nav && nav.follow) {
                    nav.follow = false;
                    if (nav._cb.onFollowChange) nav._cb.onFollowChange(false);
                }
            };
            map.on('dragstart', nav._onDragStart);

            document.addEventListener('visibilitychange', onVisibility);
            acquireWake(nav);
            startGpsHeartbeat(nav);
            if (opts.onFollowChange) opts.onFollowChange(true);

            // 出发播报 + 首屏指令卡（语音异常不得阻断导航启动）
            var V = window.WhuWalkerVoiceOutput;
            if (V) {
                try { V.warmup(); V.speak('导航开始' + (endName ? '，前往' + endName : '')); }
                catch (e) { /* 语音引擎异常时静默 */ }
            }
            var depart = nav.maneuvers[0];
            if (depart) emitCard(nav, depart, depart.cumulative_m, geo.total, nav.speed);
            if (opts.onStateChange) opts.onStateChange(true);
            return true;
        },

        // app.js 蓝点订阅入口
        onFix: onFix,

        setFollow: function (v) {
            if (!nav) return;
            nav.follow = !!v;
            if (v && nav.lastFixGcj) {
                nav.map.panTo([nav.lastFixGcj.lat, nav.lastFixGcj.lng], { animate: true });
            }
            if (nav._cb.onFollowChange) nav._cb.onFollowChange(nav.follow);
        },

        isFollowing: function () { return !!(nav && nav.follow); },

        isActive: function () { return !!nav; },

        isArrived: function () { return !!(nav && nav.arrived); },

        // 导航进行中路线被替换（切交通方式/新查询/偏航重算统一入口）
        update: function (routeData) {
            if (!nav || !routeData || !(routeData.recommended || []).length) return;
            applyRoute(nav, routeData);
            nav.arrived = false;
        },

        stop: function () {
            if (!nav) return;
            var n = nav;
            releaseWake(n);
            document.removeEventListener('visibilitychange', onVisibility);
            try { n.map.off('dragstart', n._onDragStart); } catch (e) { /* ignore */ }
            if (n.arrow) {
                try { n.map.removeLayer(n.arrow); } catch (e) { /* ignore */ }
            }
            stopGpsHeartbeat();
            if (window.WhuWalkerVoiceOutput) window.WhuWalkerVoiceOutput.stop();
            if (n._cb.onStateChange) n._cb.onStateChange(false);
            nav = null;
        },
    };

    window.WhuWalkerNavigation = Navigation;
})();
