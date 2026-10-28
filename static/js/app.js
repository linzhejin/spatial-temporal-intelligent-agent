(function () {
    'use strict';

    var CFG = (function () {
        var fallback = {
            TIANDITU_KEY: '',
            API_BASE_URL: '',
            DEFAULT_CENTER: [114.3630, 30.5365],
            MAP_ZOOM: 16,
        };
        var w = window.WHU_WALKER_CONFIG || {};
        return {
            TIANDITU_KEY: w.TIANDITU_KEY || w.tiandituKey || fallback.TIANDITU_KEY,
            API_BASE_URL: w.API_BASE_URL || w.apiBase || fallback.API_BASE_URL,
            DEFAULT_CENTER: w.DEFAULT_CENTER || w.mapCenter || fallback.DEFAULT_CENTER,
            MAP_ZOOM: w.MAP_ZOOM != null ? w.MAP_ZOOM : (w.mapZoom != null ? w.mapZoom : fallback.MAP_ZOOM),
        };
    })();

    var API_BASE = CFG.API_BASE_URL;
    var adminCsrfToken = '';
    var MAP_CENTER = CFG.DEFAULT_CENTER;
    var MAP_ZOOM = CFG.MAP_ZOOM;
    var CONTEXT_KEY_PREFIX = 'whu_walker:context:';
    var SESSION_ID_KEY = 'whu_walker:active_session';
    var SERVER_CONVERSATION_KEY = 'whu_walker:server_conversation'; // 旧版共享键，仅用于一次性迁移
    var SERVER_CONVERSATION_PREFIX = 'whu_walker:server_conversation:';
    var SERVER_CONVERSATION_MIGRATION_KEY = 'whu_walker:server_conversation_migration_v1';
    var EXPLICIT_RESET_KEY = 'whu_walker:explicit_reset_session';
    var PREFS_KEY = 'whu_walker:preferences';
    var TRAVEL_MODE_KEY = 'whu_walker:travel_mode';  // 出行方式持久化偏好

    // 产品能力边界：原生 App、手机和平板进入导航模式；普通电脑进入规划模式。
    // PC 不做定位能力探测，避免权限弹窗、IP/WiFi 粗定位和错误蓝点。
    function isNavigationDevice() {
        if (window.WhuWalkerLocation && typeof window.WhuWalkerLocation.start === 'function') return true;
        if (navigator.userAgentData && navigator.userAgentData.mobile === true) return true;
        if (/Mobile|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) return true;
        return navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
    }

    var state = {
        map: null,
        recommendedLine: null,
        shortestLine: null,
        poiMarkers: [],
        roadConditionMarkers: [],  // 路况事件标记
        loading: false,
        sessionId: null,
        serverConversationId: null,
        activeRuns: {},
        serverConversationCreateRecord: null,
        restoredActiveRuns: [],
        pendingServerTaskId: null,
        pendingServerTaskRevision: null,
        pendingServerQuestion: null,
        pendingServerTaskTurnId: 0,
        conversationHistory: [],
        pendingIntent: null,  // 尚未补全的路线意图；成功路线统一由 routeStore 持有
        activeMode: null,
        loadingTimer: null,  // 轮播加载语定时器
        roadConditionMonitorTimer: null,
        roadConditionMonitorInFlight: false,
        requestSeq: 0,  // 请求序号：防止先发的请求后返回覆盖后发请求的结果
        conversationEpoch: 0, // 显式重置后使在途请求失效
        nextConversationTurnId: 0,
        latestRouteTurnId: 0,
        latestRouteSequence: 0,
        travelMode: 'walk',  // 出行方式：walk / bike / drive（持久化偏好，默认步行）
        routeStore: window.WHURouteState ? window.WHURouteState.createStore(null) : null,
        mapPointRouteActive: false,
        mapPointRevision: 0,
        userLocation: null,  // GPS 定位结果（WGS-84）：{lng, lat, accuracy}
        userMarker: null,    // 藍点标记
        userAccuracyCircle: null,  // 定位精度圈
        locateWatchId: null, // navigator.geolocation.watchPosition 句柄
        _hadUserLocation: false,  // 是否已收到本页实时定位（缓存位置不消耗首次居中）
        _hasCenteredOnLiveLocation: false,
        _outsideCampusLocationNoticeShown: false,
        _userInteractedWithMap: false,
        mobilePaneHeight: null,
        mobilePaneRatio: 0.55,
        latestRoute: null,       // 最近一次路径响应完整数据（含 steps，供开始导航）
        locSubscribers: [],      // 定位更新订阅者（导航引擎用），不干预蓝点渲染
        _navUnsubscribe: null,   // 导航引擎的定位订阅注销函数
        productMode: null,       // navigation（手机/App）/ planning（电脑）
    };

    function applyProductMode() {
        state.productMode = isNavigationDevice() ? 'navigation' : 'planning';
        document.body.classList.toggle('is-navigation-mode', state.productMode === 'navigation');
        document.body.classList.toggle('is-planning-mode', state.productMode === 'planning');
        var tagline = document.querySelector('.header-tagline');
        if (tagline) {
            tagline.textContent = state.productMode === 'planning'
                ? '电脑路线规划工作台'
                : '武大校园 · 出行向导';
        }
        var welcomeText = document.querySelector('#welcome-bubble .bubble-text');
        if (welcomeText && state.productMode === 'planning') {
            welcomeText.textContent = '输入起点和终点，或在地图上选点；规划结果会列出路线和注意事项。';
        }
    }

    // ========== 用户标识与行为埋点（P4：画像学习 + 产品观测） ==========
    var UID_KEY = 'whu_walker:uid';

    // 匿名用户 ID：首次访问生成并持久化，用于后端用户画像（EMA 权重先验）
    function getUid() {
        var uid = null;
        try { uid = window.localStorage.getItem(UID_KEY); } catch (e) { /* 隐私模式降级 */ }
        if (!uid) {
            uid = 'u_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
            try { window.localStorage.setItem(UID_KEY, uid); } catch (e) { /* 写不进就用本次的 */ }
        }
        return uid;
    }

    // 埋点上报：sendBeacon 优先（页面关闭也不丢），任何失败静默——绝不影响主流程
    function trackEvent(event, payload) {
        try {
            var body = JSON.stringify(Object.assign({ uid: getUid(), event: event }, payload || {}));
            var url = API_BASE + '/api/telemetry';
            if (navigator.sendBeacon) {
                navigator.sendBeacon(url, new Blob([body], { type: 'application/json' }));
            } else {
                fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: body,
                    keepalive: true,
                }).catch(function () {});
            }
        } catch (e) { /* 埋点失败静默 */ }
    }

    function trackRouteSignal(event, routeState, extra) {
        if (!routeState || !routeState.route_id || !routeState.strategy) return;
        var strategy = routeState.strategy;
        if (!strategy.weights) return;
        trackEvent(event, Object.assign({
            route_id: routeState.route_id,
            strategy: strategy.name,
            strategy_source: strategy.source,
            applied_weights: strategy.weights,
            route_kind: routeState.route_kind,
        }, extra || {}));
    }

    function trackRouteShown() {
        var current = state.routeStore ? state.routeStore.current() : null;
        trackRouteSignal('route_shown', current);
    }

    // ========== 出行方式配置（珞珈秋色：步行=樱花粉 / 骑行=松绿 / 驾车=黛蓝） ==========
    var TRAVEL_MODES = {
        walk: {
            label: '步行',
            color: '#E8929C',
            speedKmh: 4.5,   // 前端兜底估速（后端未返回 duration_min 时用）
            loadingText: '正在漫步找路…',
            loadingSub: '穿过樱花大道，慢慢走就好',
        },
        bike: {
            label: '骑行',
            color: '#7BA37B',
            speedKmh: 14,
            loadingText: '正在规划骑行路线…',
            loadingSub: '帮你避开台阶和陡坡',
        },
        drive: {
            label: '驾车',
            color: '#6A9FB5',
            speedKmh: 25,
            loadingText: '正在规划车行路线…',
            loadingSub: '优先校园车行道，省时省心',
        },
    };

    // ========== 出行方式：固定默认步行（不持久化其他模式） ==========
    function loadTravelMode() {
        state.travelMode = 'walk';  // 固定默认步行，不读取 localStorage
    }

    function persistTravelMode() {
        // 不再持久化出行方式，每次刷新默认步行
    }

    // 同步分段选择器选中态、aria、图例配色与文案、驾车模式下隐藏"平坦优先"
    function syncTravelModeUI() {
        var mode = TRAVEL_MODES[state.travelMode] ? state.travelMode : 'walk';
        state.travelMode = mode;

        document.querySelectorAll('.travel-mode-btn').forEach(function (btn) {
            var active = btn.getAttribute('data-travel-mode') === mode;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });

        var chips = document.getElementById('quick-chips');
        if (chips) chips.classList.toggle('is-drive', mode === 'drive');

        // 驾车不关心坡度：若"平坦优先"正高亮，清掉高亮，避免重算时带上矛盾偏好
        if (mode === 'drive' && state.activeMode === 'slope_avoid') {
            state.activeMode = null;
            document.querySelectorAll('.quick-chip').forEach(function (c) { c.classList.remove('active'); });
        }

        // 图例：推荐路线色块 + 文案随模式
        var tm = TRAVEL_MODES[mode];
        var legendLine = document.getElementById('legend-recommended-line');
        if (legendLine) legendLine.style.background = tm.color;
        var legendText = document.getElementById('legend-recommended-text');
        if (legendText) legendText.textContent = '推荐路线（' + tm.label + '）';

        // 模式色渗透：模式栏底色 / 说明卡左边框 / 预计用时数字色
        var modeBar = document.getElementById('mode-bar');
        if (modeBar) {
            modeBar.classList.remove('mode-walk', 'mode-bike', 'mode-drive');
            modeBar.classList.add('mode-' + mode);
        }
        var expBox = document.getElementById('explanation-box');
        if (expBox) {
            expBox.classList.remove('mode-walk', 'mode-bike', 'mode-drive');
            expBox.classList.add('mode-' + mode);
        }
        var durCard = document.getElementById('duration-card');
        if (durCard) {
            durCard.setAttribute('data-mode-color', mode);
        }
    }

    function strategyToShortcutMode(strategy) {
        return {
            shortest: 'distance_first',
            scenery: 'scenery_first',
            flat: 'slope_avoid',
        }[strategy] || null;
    }

    function syncRouteStrategyUI() {
        var current = state.routeStore ? state.routeStore.current() : null;
        var selector = document.getElementById('route-strategy-switch');
        if (!selector) return;
        var results = document.getElementById('results-section');
        var hasDisplayedRoute = !!current && !!state.latestRoute && !!results && !results.hidden;
        selector.hidden = !hasDisplayedRoute;
        selector.classList.toggle('is-drive', !!current && current.travel_mode === 'drive');
        var strategyName = current && current.strategy ? current.strategy.name : null;
        document.querySelectorAll('.route-strategy-btn').forEach(function (btn) {
            var active = btn.getAttribute('data-route-strategy') === strategyName;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        state.activeMode = strategyToShortcutMode(strategyName);
        document.querySelectorAll('.quick-chip').forEach(function (chip) {
            chip.classList.toggle('active', chip.getAttribute('data-mode') === state.activeMode);
        });
    }

    function adoptRouteState(data) {
        if (!data || !data.route_state || !state.routeStore) return;
        state.routeStore.replace(data.route_state);
        clearRoadConditionUpdateNotice();
        if (TRAVEL_MODES[data.route_state.travel_mode]) {
            state.travelMode = data.route_state.travel_mode;
            syncTravelModeUI();
        }
        syncRouteStrategyUI();
        saveContext();
    }

    // 用户手动切换出行方式：已有路线时先请求完整状态重规划，成功后才切换界面。
    function setTravelMode(mode) {
        if (!TRAVEL_MODES[mode]) mode = 'walk';
        if (mode === state.travelMode) return;
        var current = state.routeStore ? state.routeStore.current() : null;
        if (current) {
            replanCurrentRoute({ travel_mode: mode }, '切换为' + TRAVEL_MODES[mode].label);
            return;
        }
        state.travelMode = mode;
        persistTravelMode();
        syncTravelModeUI();
        if (state.map) loadAndRenderRoadConditions(state.roadConditionAdminView);
    }

    // 服务端响应里的 mode（NL 识别"骑车/开车"）以服务端为准，回写选择器状态
    function syncModeFromServer(data) {
        if (data && data.mode && TRAVEL_MODES[data.mode] && data.mode !== state.travelMode) {
            state.travelMode = data.mode;
            persistTravelMode();
            syncTravelModeUI();
            if (state.map) loadAndRenderRoadConditions(state.roadConditionAdminView);
        }
    }

    // 用服务端签发的完整路线状态重规划。请求期间不修改已提交状态，失败时保留原路线。
    async function replanCurrentRoute(change, actionText) {
        if (!state.routeStore) return false;
        var current = state.routeStore.current();
        if (!current) return false;

        var token = state.routeStore.begin(change);
        state.requestSeq += 1;
        var mySeq = state.requestSeq;
        var targetMode = change.travel_mode || current.travel_mode;
        var tm = TRAVEL_MODES[targetMode] || TRAVEL_MODES.walk;
        hideError();
        showLoading((actionText || '更新路线') + '…', '起终点和途经安排保持不变');

        try {
            var result = await apiRequest('/api/route/replan', {
                route_state: current,
                change: change,
            });
            if (mySeq !== state.requestSeq) return false;
            if (!result.recommended || result.recommended.length === 0 || !result.route_state) {
                throw new Error(tm.label + '路线暂时不可达');
            }
            if (!state.routeStore.commit(token, result.route_state)) return false;
            clearRoadConditionUpdateNotice();
            if (change.strategy) {
                trackRouteSignal('strategy_abandoned', current);
                trackRouteSignal('strategy_selected', result.route_state);
            }
            state.travelMode = result.route_state.travel_mode;
            persistTravelMode();
            syncTravelModeUI();
            syncRouteStrategyUI();
            if (state.map) loadAndRenderRoadConditions(state.roadConditionAdminView);
            renderRoute(result);
            showResults(result);
            trackRouteShown();
            if (window.WhuWalkerNavigation && window.WhuWalkerNavigation.isActive()) {
                trackRouteSignal('navigation_started', result.route_state);
            }
            return true;
        } catch (err) {
            if (mySeq !== state.requestSeq) return false;
            state.routeStore.rollback(token);
            state.travelMode = current.travel_mode;
            syncTravelModeUI();
            syncRouteStrategyUI();
            showError('没有切换路线', ((err && err.message) || '重算失败') + '，当前路线仍然保留。');
            return false;
        } finally {
            if (mySeq === state.requestSeq) {
                hideLoading();
            }
        }
    }

    // ========== 轮播加载语（有人味儿，随出行方式变化） ==========
    function getLoadingMessages(query) {
        var mode = TRAVEL_MODES[state.travelMode] ? state.travelMode : 'walk';
        var first = TRAVEL_MODES[mode];
        var calcSub = mode === 'bike' ? '帮你避开台阶和陡坡，骑车更省心'
                   : mode === 'drive' ? '优先校园车行道，避开步行小路'
                   : '帮你避开那些不好走的路';

        // 根据 query 内容智能匹配 Agent 正在做什么
        var q = (query || '').toLowerCase();
        var msgs = [{ text: first.loadingText, sub: first.loadingSub }];

        if (q.indexOf('吃') >= 0 || q.indexOf('饭') >= 0 || q.indexOf('食堂') >= 0 || q.indexOf('饿') >= 0) {
            msgs.push({ text: '正在搜食堂…', sub: '看看哪个最近' });
        } else if (q.indexOf('咖啡') >= 0 || q.indexOf('奶茶') >= 0 || q.indexOf('喝') >= 0) {
            msgs.push({ text: '正在找饮品店…', sub: '给你最近的' });
        } else if (q.indexOf('逛') >= 0 || q.indexOf('参观') >= 0 || q.indexOf('游览') >= 0 || q.indexOf('旅游') >= 0) {
            msgs.push({ text: '正在安排游览路线…', sub: '串起最好看的几个点' });
        } else if (q.indexOf('买') >= 0 || q.indexOf('超市') >= 0 || q.indexOf('快递') >= 0) {
            msgs.push({ text: '正在找顺路的店…', sub: '不绕路那种' });
        } else if (q.indexOf('天气') >= 0 || q.indexOf('下雨') >= 0 || q.indexOf('带伞') >= 0) {
            msgs.push({ text: '正在查天气…', sub: '看适不适合出门' });
        } else if (q.indexOf('修路') >= 0 || q.indexOf('封') >= 0 || q.indexOf('施工') >= 0) {
            msgs.push({ text: '正在查路况…', sub: '看看有没有管制' });
        } else if (q.indexOf('樱花') >= 0 || q.indexOf('赏') >= 0) {
            msgs.push({ text: '正在安排赏花路线…', sub: '樱花季这条线最美' });
        } else {
            msgs.push({ text: '正在查地图…', sub: '珞珈山的路我都熟' });
        }

        msgs.push({ text: '正在计算最佳路线…', sub: calcSub });
        msgs.push({ text: '快好了…', sub: '稍等一下下' });
        return msgs;
    }

    function startLoadingMessages(query) {
        var messages = getLoadingMessages(query);
        var idx = 0;
        var textEl = document.getElementById('loading-text');
        var subEl = document.getElementById('loading-subtext');
        showLoading(messages[0].text, messages[0].sub);

        state.loadingTimer = setInterval(function () {
            idx = (idx + 1) % messages.length;
            var msg = messages[idx];
            // 淡入淡出效果
            textEl.classList.add('fade');
            subEl.classList.add('fade');
            setTimeout(function () {
                textEl.textContent = msg.text;
                subEl.textContent = msg.sub;
                textEl.classList.remove('fade');
                subEl.classList.remove('fade');
            }, 200);
        }, 2200);
    }

    function stopLoadingMessages() {
        if (state.loadingTimer) {
            clearInterval(state.loadingTimer);
            state.loadingTimer = null;
        }
    }

    function generateSessionId() {
        return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function getOrCreateSessionId() {
        try {
            var existing = localStorage.getItem(SESSION_ID_KEY);
            if (existing) return existing;
            var created = generateSessionId();
            localStorage.setItem(SESSION_ID_KEY, created);
            return created;
        } catch (e) {
            return generateSessionId();
        }
    }

    function getContextKey() {
        return CONTEXT_KEY_PREFIX + (state.sessionId || 'default');
    }

    function getServerConversationKey(sessionId) {
        return SERVER_CONVERSATION_PREFIX + (sessionId || state.sessionId || 'default');
    }

    function readServerConversationId() {
        try {
            var key = getServerConversationKey();
            var conversationId = localStorage.getItem(key);
            if (conversationId) return conversationId;
            if (localStorage.getItem(SERVER_CONVERSATION_MIGRATION_KEY) === '1') return null;
            // 旧版本使用跨标签共享键；只迁移一次，之后旧标签页写回也不会污染新会话。
            conversationId = localStorage.getItem(SERVER_CONVERSATION_KEY);
            if (conversationId) localStorage.setItem(key, conversationId);
            localStorage.removeItem(SERVER_CONVERSATION_KEY);
            localStorage.setItem(SERVER_CONVERSATION_MIGRATION_KEY, '1');
            return conversationId;
        } catch (e) {
            return null;
        }
    }

    function beginExplicitRefreshSession() {
        var previousSessionId = state.sessionId;
        try {
            previousSessionId = previousSessionId || localStorage.getItem(SESSION_ID_KEY);
        } catch (e) {}
        var freshSessionId = generateSessionId();
        state.sessionId = freshSessionId;
        // 应用内刷新原地开启新会话，不触发页面初始化与历史恢复。
        // 清除旧构建遗留的一次性标记，避免之后普通 F5 误清空新会话。
        try { localStorage.removeItem(EXPLICIT_RESET_KEY); } catch (e) {}
        try { localStorage.setItem(SESSION_ID_KEY, freshSessionId); } catch (e) {}
        if (previousSessionId) {
            try { localStorage.removeItem(CONTEXT_KEY_PREFIX + previousSessionId); } catch (e) {}
            try { localStorage.removeItem(getServerConversationKey(previousSessionId)); } catch (e) {}
        }
        try { localStorage.setItem(SERVER_CONVERSATION_MIGRATION_KEY, '1'); } catch (e) {}
        try { localStorage.removeItem(SERVER_CONVERSATION_KEY); } catch (e) {}
    }

    function consumeExplicitRefreshSession() {
        var freshSessionId = null;
        try { freshSessionId = localStorage.getItem(EXPLICIT_RESET_KEY); } catch (e) {}
        if (!freshSessionId) return;
        if (!/^sess_[a-zA-Z0-9_-]+$/.test(freshSessionId)) freshSessionId = generateSessionId();
        state.sessionId = freshSessionId;
        try { localStorage.setItem(SESSION_ID_KEY, freshSessionId); } catch (e) {}
        try { localStorage.removeItem(CONTEXT_KEY_PREFIX + freshSessionId); } catch (e) {}
        try { localStorage.removeItem(getServerConversationKey(freshSessionId)); } catch (e) {}
        try { localStorage.setItem(SERVER_CONVERSATION_MIGRATION_KEY, '1'); } catch (e) {}
        try { localStorage.removeItem(SERVER_CONVERSATION_KEY); } catch (e) {}
        try { localStorage.removeItem(EXPLICIT_RESET_KEY); } catch (e) {}
    }

    function loadContext() {
        try {
            var raw = localStorage.getItem(getContextKey());
            if (raw) {
                var parsed = JSON.parse(raw);
                // 新结构 {history, routeState, pendingIntent}；
                // 旧结构 [{query,...}] 直接丢弃，避免污染
                if (parsed && Array.isArray(parsed.history)) {
                    state.conversationHistory = parsed.history;
                    state.pendingIntent = parsed.pendingIntent || parsed.routeSlot || null;
                    state.pendingServerTaskId = parsed.pendingServerTaskId || null;
                    state.pendingServerTaskRevision = Number.isInteger(parsed.pendingServerTaskRevision)
                        ? parsed.pendingServerTaskRevision : null;
                    state.pendingServerQuestion = parsed.pendingServerQuestion || null;
                    state.pendingServerTaskTurnId = Number.isInteger(parsed.pendingServerTaskTurnId)
                        ? parsed.pendingServerTaskTurnId : 0;
                    if (state.routeStore && parsed.routeState) state.routeStore.replace(parsed.routeState);
                }
            }
        } catch (e) {
            state.conversationHistory = [];
        }
    }

    function saveContext() {
        try {
            // 最近 4 轮（8 条消息）+ 服务端签发的路线状态 + 未补全意图
            localStorage.setItem(getContextKey(), JSON.stringify({
                history: state.conversationHistory.slice(-8),
                routeState: state.routeStore ? state.routeStore.current() : null,
                pendingIntent: state.pendingIntent,
                pendingServerTaskId: state.pendingServerTaskId,
                pendingServerTaskRevision: state.pendingServerTaskRevision,
                pendingServerQuestion: state.pendingServerQuestion,
                pendingServerTaskTurnId: state.pendingServerTaskTurnId,
            }));
        } catch (e) {}
    }

    async function restoreCompletedRouteFromRun(runId, expectedRoute, expectedTurnId,
                                                expectedSequence, restoreEpoch,
                                                restoreConversationId, restoreTurnId) {
        if (!runId || !expectedRoute || !expectedRoute.route_id
                || !expectedRoute.data_version || !expectedRoute.road_condition_version) return;
        function isCurrent() {
            var current = state.routeStore && state.routeStore.current();
            return restoreEpoch === state.conversationEpoch
                && restoreConversationId === state.serverConversationId
                && restoreTurnId === state.nextConversationTurnId
                && expectedTurnId === state.latestRouteTurnId
                && expectedSequence === state.latestRouteSequence
                && current && current.route_id === expectedRoute.route_id;
        }
        try {
            var run = await apiRequest('/api/runs/' + encodeURIComponent(runId)
                + '?conversation_id=' + encodeURIComponent(restoreConversationId), null, 'GET', 15000);
            if (!isCurrent()) return;
            if (!run || (run.status !== 'completed' && run.status !== 'partial'
                    && run.status !== 'needs_input')) return;
            var result = run && run.result;
            if (!result || !Array.isArray(result.recommended) || !result.recommended.length
                    || !result.route_state || result.route_state.route_id !== expectedRoute.route_id
                    || result.route_state.data_version !== expectedRoute.data_version
                    || result.route_state.road_condition_version !== expectedRoute.road_condition_version) return;
            var versions = await apiRequest(
                '/api/road-conditions?version_only=1&include_route_data_version=1',
                null, 'GET', 15000
            );
            if (!isCurrent()) return;
            if (!versions || versions.data_version !== expectedRoute.data_version) {
                showTopBanner('校园地图数据已更新，请重新规划这段行程。', 'warning');
                return;
            }
            if (versions.active_event_version !== expectedRoute.road_condition_version) {
                showTopBanner('校园路况已更新，请按最新路况重新规划。', 'warning');
                return;
            }
            syncModeFromServer(result);
            renderRoute(result);
            showResults(result);
        } catch (error) {
            if (isCurrent()) {
                console.warn('[SESSION] 上次路线暂时无法恢复:', error && error.message);
                showTopBanner('上次路线暂时无法恢复，请稍后重试或重新规划。', 'warning');
            }
        }
    }

    async function restoreServerConversation() {
        if (!state.serverConversationId) return false;
        var restoreEpoch = state.conversationEpoch;
        var restoreTurnId = state.nextConversationTurnId;
        var restoreConversationId = state.serverConversationId;
        try {
            var data = await apiRequest(
                '/api/conversations/' + encodeURIComponent(restoreConversationId), null, 'GET', 15000
            );
            // A reset or a new user turn makes this snapshot stale. Never let
            // delayed hydration resurrect the old transcript or its active runs.
            if (restoreEpoch !== state.conversationEpoch
                    || restoreTurnId !== state.nextConversationTurnId
                    || restoreConversationId !== state.serverConversationId) return false;
            var messages = Array.isArray(data.messages) ? data.messages : [];
            state.restoredActiveRuns = Array.isArray(data.active_runs) ? data.active_runs : [];
            var pendingTask = (data.tasks || []).find(function (task) { return task.status === 'needs_input'; });
            if (pendingTask) {
                state.pendingServerTaskId = pendingTask.task_id;
                state.pendingServerTaskRevision = pendingTask.revision;
                var questions = pendingTask.state && pendingTask.state.pending_questions;
                state.pendingServerQuestion = Array.isArray(questions) && questions.length
                    ? questions[questions.length - 1] : null;
            } else {
                state.pendingServerTaskId = null;
                state.pendingServerTaskRevision = null;
                state.pendingServerQuestion = null;
                state.pendingServerTaskTurnId = 0;
            }
            if (!messages.length) return true;
            var turnByTask = Object.create(null);
            var nextHistory = [];
            var nextTurn = 0;
            var latestRouteState = null;
            var latestRouteTurnId = 0;
            var latestRouteSequence = 0;
            var latestRouteRunId = null;
            messages.forEach(function (message) {
                if (!message || (message.role !== 'user' && message.role !== 'assistant')) return;
                var key = message.task_id || ('message-' + message.seq);
                if (!turnByTask[key]) turnByTask[key] = ++nextTurn;
                var item = { role: message.role, content: message.content || '', _turnId: turnByTask[key],
                    _taskId: message.task_id || null,
                    _seq: Number.isFinite(Number(message.seq)) ? Number(message.seq) : null };
                if (message.role === 'assistant' && message.metadata) {
                    item.task_type = message.metadata.task_type || null;
                    if (message.metadata.route_state
                            && (!message.metadata.status
                                || ['completed', 'partial', 'needs_input'].includes(message.metadata.status))) {
                        var routeSequence = Number(message.metadata.source_message_seq) || item._seq || 0;
                        if (!latestRouteState || routeSequence >= latestRouteSequence) {
                            latestRouteState = message.metadata.route_state;
                            latestRouteTurnId = item._turnId;
                            latestRouteSequence = routeSequence;
                            latestRouteRunId = message.metadata.run_id || null;
                        }
                    }
                }
                nextHistory.push(item);
            });
            if (pendingTask) {
                state.pendingServerTaskTurnId = turnByTask[pendingTask.task_id] || nextTurn;
            }
            state.conversationHistory = nextHistory.slice(-20);
            state.nextConversationTurnId = nextTurn;
            state.latestRouteTurnId = latestRouteTurnId;
            state.latestRouteSequence = latestRouteSequence;
            if (latestRouteState && state.routeStore) {
                try { state.routeStore.replace(latestRouteState); } catch (e) {
                    console.warn('[SESSION] 忽略无法恢复的路线状态');
                }
            }
            saveContext();
            if (latestRouteRunId) {
                restoreCompletedRouteFromRun(latestRouteRunId, latestRouteState,
                    latestRouteTurnId, latestRouteSequence, restoreEpoch,
                    restoreConversationId, state.nextConversationTurnId);
            }
            return true;
        } catch (error) {
            if (restoreEpoch !== state.conversationEpoch
                    || restoreTurnId !== state.nextConversationTurnId
                    || restoreConversationId !== state.serverConversationId) return false;
            if (error && error.code === 'conversation_not_found') {
                state.serverConversationId = null;
                try { localStorage.removeItem(getServerConversationKey()); } catch (e) {}
            } else {
                console.warn('[SESSION] 服务端会话恢复暂不可用:', error && error.message);
            }
            return false;
        }
    }

    function addConversationTurn(query, result, turnId, acceptRouteState) {
        // 对话历史：真实的用户/助手消息流（含回复内容与涉及地点，供指代消解）
        turnId = Number.isFinite(turnId) ? turnId : ++state.nextConversationTurnId;
        state.nextConversationTurnId = Math.max(state.nextConversationTurnId, turnId);
        var pair = [{ role: 'user', content: query, _turnId: turnId }];
        var replyText = result.explanation || result.reply || result.message || '';
        // 候选列表注入：让下一轮 Agent 知道用户刚看了哪些候选
        if (result.response_kind === 'candidates' && result.candidates && result.candidates.length) {
            var candNames = result.candidates.slice(0, 6).map(function (c) { return c.name; }).join('、');
            replyText = (replyText || '') + '（候选：' + candNames + '）';
        }
        pair.push({
            role: 'assistant',
            content: replyText,
            _turnId: turnId,
            task_type: result.task_type || null,
            entities: {
                start: result.start && result.start.name ? result.start.name : null,
                end: result.end && result.end.name ? result.end.name : null,
                poi: result.poi && result.poi.name ? result.poi.name : null,
                mode: result.mode || null,
            },
        });
        var insertion = state.conversationHistory.findIndex(function (item) {
            return Number(item._turnId || 0) > turnId;
        });
        if (insertion < 0) insertion = state.conversationHistory.length;
        state.conversationHistory.splice.apply(state.conversationHistory, [insertion, 0].concat(pair));
        if (state.conversationHistory.length > 20) state.conversationHistory = state.conversationHistory.slice(-20);
        // 规划槽位：路径规划轮次才更新（闲聊/候选/澄清不冲掉在途规划）
        var isPlanningTurn = result.task_type === 'path_planning'
            || (result.response_kind === 'route' && result.start && result.end);
        if (result.route_state && acceptRouteState !== false && turnId >= state.latestRouteTurnId) {
            state.latestRouteTurnId = turnId;
            state.pendingIntent = null;
        } else if (isPlanningTurn && acceptRouteState !== false && turnId >= state.latestRouteTurnId) {
            state.pendingIntent = {
                task_type: result.task_type || 'path_planning',
                start: result.start || null,
                end: result.end || null,
                constraints: result.constraints || null,
                weights: result.weights || result.applied_weights || null,
                ambiguity: result.ambiguity || null,
                mode: result.mode || state.travelMode,
            };
        }
        saveContext();
    }

    function wgs84ToGcj02(lng, lat) {
        var a = 6378245.0;
        var ee = 0.006693421622965823;

        if (lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271) {
            return [lng, lat];
        }

        var dlat = _transformLat(lng - 105.0, lat - 35.0);
        var dlng = _transformLng(lng - 105.0, lat - 35.0);
        var radlat = lat / 180.0 * Math.PI;
        var magic = Math.sin(radlat);
        magic = 1 - ee * magic * magic;
        var sqrtmagic = Math.sqrt(magic);
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * Math.PI);
        dlng = (dlng * 180.0) / (a / sqrtmagic * Math.cos(radlat) * Math.PI);
        return [lng + dlng, lat + dlat];
    }

    function gcj02ToWgs84(lng, lat) {
        var clng = lng;
        var clat = lat;
        for (var i = 0; i < 3; i++) {
            var wlng = clng;
            var wlat = clat;
            var glng_lat = wgs84ToGcj02(wlng, wlat);
            clng += (lng - glng_lat[0]);
            clat += (lat - glng_lat[1]);
        }
        return [clng, clat];
    }

    function _transformLat(x, y) {
        var ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x));
        ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0;
        ret += (20.0 * Math.sin(y * Math.PI) + 40.0 * Math.sin(y / 3.0 * Math.PI)) * 2.0 / 3.0;
        ret += (160.0 * Math.sin(y / 12.0 * Math.PI) + 320 * Math.sin(y * Math.PI / 30.0)) * 2.0 / 3.0;
        return ret;
    }

    function _transformLng(x, y) {
        var ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x));
        ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0;
        ret += (20.0 * Math.sin(x * Math.PI) + 40.0 * Math.sin(x / 3.0 * Math.PI)) * 2.0 / 3.0;
        ret += (150.0 * Math.sin(x / 12.0 * Math.PI) + 300.0 * Math.sin(x / 30.0 * Math.PI)) * 2.0 / 3.0;
        return ret;
    }

    function toGcj02Path(coords) {
        return coords.map(function (c) {
            return wgs84ToGcj02(c.lng || c[0], c.lat || c[1]);
        });
    }

    // 两点球面距离（米），用于 GPS 抖动抑制
    function _haversineMeters(lat1, lng1, lat2, lng2) {
        var rad = Math.PI / 180;
        var dLat = (lat2 - lat1) * rad, dLng = (lng2 - lng1) * rad;
        var a = Math.sin(dLat / 2) * Math.sin(dLat / 2)
            + Math.cos(lat1 * rad) * Math.cos(lat2 * rad)
            * Math.sin(dLng / 2) * Math.sin(dLng / 2);
        return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    // navigator.geolocation 返回的坐标系在国内不确定：
    //   - 标准 WGS-84（Chrome/Edge 规范行为，手机 GPS）
    //   - GCJ-02（国内 Windows Location Platform 可能用高德数据源，已偏移）
    // 无法从 API 层直接判断坐标系。利用"校园中心"作为锚点对比：
    //   把坐标当成 WGS-84 转一次 GCJ-02，看转换前后哪个离校园中心更近。
    //   转换后更近 → 原始是 WGS-84；直接更近 → 原始已是 GCJ-02，需逆转回 WGS-84。
    // 返回统一为 WGS-84，交给 renderUserLocation 做最终渲染转换。
    function _guessWgs84FromNavigator(lng, lat) {
        var center = MAP_CENTER; // GCJ-02 [lng, lat]
        var dDirect = _haversineMeters(lat, lng, center[1], center[0]);
        var gcj = wgs84ToGcj02(lng, lat);
        var dConverted = _haversineMeters(gcj[1], gcj[0], center[1], center[0]);
        if (dConverted < dDirect) {
            // 转换后更近 → 原始是 WGS-84
            return [lng, lat];
        }
        // 直接更近 → 原始是 GCJ-02，逆转回 WGS-84
        return gcj02ToWgs84(lng, lat);
    }

    // 后端所有坐标（POI、路径、路况）均为 GCJ-02；高德瓦片也是 GCJ-02，
    // 直接用 GCJ-02 坐标入图（[lat, lng]），无需转换
    function gcjToLatLng(lng, lat) {
        return [lat, lng];
    }

    // Leaflet divIcon 小工具
    function divMarker(latlng, html, size, anchor, title, extraCls) {
        var icon = L.divIcon({
            className: 'whu-div-icon' + (extraCls ? ' ' + extraCls : ''),
            html: html,
            iconSize: size,
            iconAnchor: anchor,
        });
        return L.marker(latlng, { icon: icon, title: title || '', keyboard: false });
    }

    function initMap() {
        if (typeof L === 'undefined') {
            setTimeout(initMap, 300);
            return;
        }

        if (state.map) return;

        var placeholder = document.getElementById('map-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        try {
            // 手机屏幕小，用更小的 zoom（比例尺更小）显示更大范围；电脑保持 16
            var initialZoom = MAP_ZOOM;
            if (window.innerWidth <= 767) {
                initialZoom = 14;
            }
            var center = gcjToLatLng(MAP_CENTER[0], MAP_CENTER[1]);

            state.map = L.map('map-container', {
                center: center,
                zoom: initialZoom,
                zoomControl: true,
                attributionControl: true,
                preferCanvas: false,
            });
            bindUserMapInteraction(state.map);

            // 用户拖拽地图 → 退出选点模式（如有）
            state.map.on('dragstart', function () {
                if (_pointPickMode) stopPointPick();
            });

            // 底图：高德瓦片（GCJ-02，国内秒开，中文标注，高缩放全覆盖）
            // 全站统一 GCJ-02：POI/路线/GPS 均按 GCJ-02 渲染，底图不可混入 WGS-84 源
            // （OSM/Esri/天地图 vec_w 为 WGS-84/CGCS2000，与 GCJ-02 路线串位约 500m，禁止接入）
            var baseLayers = {};
            var defaultLayer = null;

            // ===== 高德矢量街道图（GCJ-02，国内手机端秒开，无需 Key）=====
            var amap = L.tileLayer(
                'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                { subdomains: ['1', '2', '3', '4'],
                  maxZoom: 18,
                  attribution: '© <a href="https://ditu.amap.com/" target="_blank" rel="noopener">高德地图</a>' }
            );
            baseLayers['高德矢量'] = amap;

            // 高德卫星影像 + 路网注记（GCJ-02，无需 Key；与 POI/路线同坐标系，天然对齐）
            var amapSat = L.layerGroup([
                L.tileLayer(
                    'https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
                    { subdomains: ['1', '2', '3', '4'],
                      maxZoom: 18,
                      attribution: '© <a href="https://ditu.amap.com/" target="_blank" rel="noopener">高德地图</a>' }
                ),
                L.tileLayer(
                    'https://webst0{s}.is.autonavi.com/appmaptile?style=8&x={x}&y={y}&z={z}',
                    { subdomains: ['1', '2', '3', '4'], maxZoom: 18 }
                ),
            ]);
            baseLayers['卫星影像'] = amapSat;

            if (!defaultLayer) {
                // 默认用高德矢量（国内手机端秒开，GCJ-02 与 POI/路径天然对齐）
                amap.addTo(state.map);
                defaultLayer = amap;
            }

            L.control.layers(baseLayers, null, { position: 'topright', collapsed: true }).addTo(state.map);
            L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(state.map);

            addLocateControl();

            // 容器在隐藏状态下初始化时需要刷新尺寸
            setTimeout(function () { if (state.map) state.map.invalidateSize(); }, 0);

            // 加载路况事件标记
            loadAndRenderRoadConditions();
            // 加载实时天气徽章
            loadWeatherBadge();
        } catch (e) {
            console.error('地图初始化失败:', e);
            showError('地图加载失败', '无法初始化地图组件，请刷新页面重试');
        }
    }

    function setPendingServerClarification(taskId, revision, question, turnId) {
        var turn = Number.isInteger(turnId) ? turnId : 0;
        if (turn < state.pendingServerTaskTurnId) return false;
        state.pendingServerTaskId = taskId || null;
        state.pendingServerTaskRevision = Number.isInteger(revision) ? revision : null;
        state.pendingServerQuestion = question || null;
        state.pendingServerTaskTurnId = turn;
        return true;
    }

    function clearPendingServerClarification(taskId, turnId) {
        var turn = Number.isInteger(turnId) ? turnId : 0;
        if (taskId !== state.pendingServerTaskId || turn < state.pendingServerTaskTurnId) return false;
        state.pendingServerTaskId = null;
        state.pendingServerTaskRevision = null;
        state.pendingServerQuestion = null;
        state.pendingServerTaskTurnId = 0;
        return true;
    }

    // ========== 地图选点控制（起点/途经/终点） ==========
    function addLocateControl() {
        var container = L.DomUtil.create('div', 'whu-locate-group');

        // 起点、途经、终点三个按钮
        var btnDefs = [
            { key: 'start', label: '起点', title: '点击设置出发点' },
            { key: 'via',   label: '途经', title: '点击设置途经点' },
            { key: 'end',   label: '终点', title: '点击设置终点' },
        ];
        for (var i = 0; i < btnDefs.length; i++) {
            (function (def) {
                var btn = L.DomUtil.create('button', 'whu-point-btn whu-point-btn--' + def.key, container);
                btn.type = 'button';
                btn.setAttribute('aria-label', def.title);
                btn.setAttribute('aria-pressed', 'false');
                btn.title = def.title;
                btn.innerHTML = '<span class="whu-point-icon" aria-hidden="true"></span>'
                    + '<span class="whu-point-label">' + def.label + '</span>';
                L.DomEvent.disableClickPropagation(btn);
                L.DomEvent.disableScrollPropagation(btn);
                btn.addEventListener('click', function () { togglePointPickMode(def.key); });
                btn.dataset.pointKey = def.key;
                state['pointBtn_' + def.key] = btn;
            })(btnDefs[i]);
        }

        var PointCtrl = L.Control.extend({
            options: { position: 'topright' },
            onAdd: function () { return container; },
        });
        state.locateControl = new PointCtrl().addTo(state.map);
    }

    function setLocateBtnState(busy) {
        // 兼容：旧的 locateBtn 引用可能不存在了
    }

    // ====== 地图选点模式（起点/途经/终点） ======
    var _pointPickMode = null;   // 'start' | 'via' | 'end' | null
    var _pointPickHandler = null;
    // 地图上的选点标记
    state.pointMarkers = { start: null, via: [], end: null };
    state.mapPoints = { start: null, via: [], end: null };  // {lng, lat, gcjLng, gcjLat} WGS-84

    function togglePointPickMode(key) {
        if (_pointPickMode === key) {
            // 再次点击同一个按钮 → 退出选点模式
            stopPointPick();
            return;
        }
        stopPointPick();
        _pointPickMode = key;
        var labels = { start: '出发点', via: '途经点', end: '终点' };
        showTopBanner('点击地图设置' + labels[key], 'info');
        // 高亮当前按钮
        var btn = state['pointBtn_' + key];
        if (btn) {
            btn.classList.add('is-picking');
            btn.setAttribute('aria-pressed', 'true');
        }
        state.map.getContainer().style.cursor = 'crosshair';

        _pointPickHandler = function (e) {
            // leaflet 坐标即 GCJ-02（高德瓦片）
            var gcjLat = e.latlng.lat, gcjLng = e.latlng.lng;
            var wgs = gcj02ToWgs84(gcjLng, gcjLat);
            setMapPoint(key, wgs[0], wgs[1], gcjLng, gcjLat);
            stopPointPick();
            // 起终点都设置后自动规划
            if (state.mapPoints.start && state.mapPoints.end) {
                autoPlanFromMapPoints();
            }
        };
        state.map.on('click', _pointPickHandler);
    }

    function stopPointPick() {
        if (_pointPickHandler && state.map) state.map.off('click', _pointPickHandler);
        _pointPickHandler = null;
        _pointPickMode = null;
        // 清除按钮高亮
        ['start', 'via', 'end'].forEach(function (k) {
            var b = state['pointBtn_' + k];
            if (b) {
                b.classList.remove('is-picking');
                b.setAttribute('aria-pressed', 'false');
            }
        });
        if (state.map) state.map.getContainer().style.cursor = '';
    }

    function bindMapPointRemoval(marker, key, label) {
        var buttonLabel = key === 'via' ? '删除此途经点' : key === 'start' ? '删除此起点' : '删除此终点';
        marker.bindPopup('<div class="whu-point-popup"><strong>' + label + '</strong>'
            + '<button type="button" class="whu-point-delete">' + buttonLabel + '</button></div>');
        marker.on('popupopen', function (event) {
            var popupElement = event.popup && event.popup.getElement ? event.popup.getElement() : null;
            var deleteButton = popupElement && popupElement.querySelector('.whu-point-delete');
            if (!deleteButton) return;
            deleteButton.addEventListener('click', function (clickEvent) {
                clickEvent.preventDefault();
                clickEvent.stopPropagation();
                var index = key === 'via' ? state.pointMarkers.via.indexOf(marker) : undefined;
                removeMapPoint(key, index);
            }, { once: true });
        });
    }

    function removeMapPoint(key, index) {
        var label;
        var marker;
        if (key === 'via') {
            if (!Number.isInteger(index) || index < 0 || index >= state.mapPoints.via.length) return false;
            label = '途经点' + (index + 1);
            marker = state.pointMarkers.via[index];
            state.mapPoints.via.splice(index, 1);
            state.pointMarkers.via.splice(index, 1);
        } else if (key === 'start' || key === 'end') {
            if (!state.mapPoints[key] && !state.pointMarkers[key]) return false;
            label = key === 'start' ? '起点' : '终点';
            marker = state.pointMarkers[key];
            state.mapPoints[key] = null;
            state.pointMarkers[key] = null;
        } else {
            return false;
        }

        state.mapPointRevision += 1;
        if (marker && state.map) state.map.removeLayer(marker);
        if (state.map && typeof state.map.closePopup === 'function') state.map.closePopup();
        showTopBanner('已删除' + label, 'info');

        // 已按手选点生成的路线不能继续展示旧的起终点/途经关系。
        if (state.mapPointRouteActive) {
            if (state.mapPoints.start && state.mapPoints.end) {
                autoPlanFromMapPoints();
            } else {
                clearMapPointRoute();
            }
        }
        return true;
    }

    function clearMapPointRoute() {
        state.mapPointRouteActive = false;
        clearRouteResult();
        state.latestRoute = null;
        if (state.routeStore) state.routeStore.replace(null);
        syncRouteStrategyUI();
        saveContext();
    }

    function setMapPoint(key, lng, lat, gcjLng, gcjLat) {
        var pt = { lng: lng, lat: lat, gcjLng: gcjLng, gcjLat: gcjLat };
        var labels = { start: '起点', via: '途经', end: '终点' };
        var colors = { start: '#1B7F3B', via: '#E67E22', end: '#C0392B' };
        var icons = { start: '🟢', via: '🚏', end: '🔴' };

        if (key === 'via') {
            if (state.mapPoints.via.length >= 10) {
                showTopBanner('最多可设置 10 个途经点', 'warning');
                return;
            }
            // 途经点可多个
            state.mapPoints.via.push(pt);
            state.mapPointRevision += 1;
            var viaIdx = state.mapPoints.via.length;
            var m = L.marker([gcjLat, gcjLng], {
                icon: L.divIcon({
                    html: '<div style="font-size:20px;">' + icons.via + '</div>',
                    className: 'whu-div-icon',
                    iconSize: [24, 24], iconAnchor: [12, 12],
                }),
                // 手动点位只通过地图鼠标/触屏操作，避免 Leaflet 键盘焦点框留在图标上。
                keyboard: false,
            }).addTo(state.map);
            bindMapPointRemoval(m, 'via', labels.via);
            state.pointMarkers.via.push(m);
            showTopBanner('✅ 已设置途经点' + viaIdx + '（点标记可删除）', 'success');
        } else {
            // 起点或终点：替换已有标记
            state.mapPoints[key] = pt;
            state.mapPointRevision += 1;
            if (state.pointMarkers[key]) state.map.removeLayer(state.pointMarkers[key]);
            var mk = L.marker([gcjLat, gcjLng], {
                icon: L.divIcon({
                    html: '<div style="font-size:20px;">' + icons[key] + '</div>',
                    className: 'whu-div-icon',
                    iconSize: [24, 24], iconAnchor: [12, 12],
                }),
                keyboard: false,
            }).addTo(state.map);
            bindMapPointRemoval(mk, key, labels[key]);
            state.pointMarkers[key] = mk;
            showTopBanner('✅ 已设置' + labels[key] + '（点标记可删除）', 'success');
        }
    }

    function clearMapPoints() {
        state.mapPoints = { start: null, via: [], end: null };
        state.mapPointRevision += 1;
        if (state.map && state.pointMarkers.start) state.map.removeLayer(state.pointMarkers.start);
        state.pointMarkers.start = null;
        state.pointMarkers.via.forEach(function (m) { if (state.map) state.map.removeLayer(m); });
        state.pointMarkers.via = [];
        if (state.map && state.pointMarkers.end) state.map.removeLayer(state.pointMarkers.end);
        state.pointMarkers.end = null;
        state.mapPointRouteActive = false;
        if (state.map && typeof state.map.closePopup === 'function') state.map.closePopup();
    }

    function snapshotMapPoints() {
        var points = state.mapPoints || { start: null, via: [], end: null };
        function copyPoint(point) {
            return point ? { lng: point.lng, lat: point.lat } : null;
        }
        return {
            revision: state.mapPointRevision,
            start: copyPoint(points.start),
            via: points.via.map(copyPoint),
            end: copyPoint(points.end),
        };
    }

    function requestUsesMapPoints(requestBody, selectedPoints) {
        var selected = [selectedPoints.start, selectedPoints.end].filter(Boolean)
            .concat(selectedPoints.via || []);
        var requested = [requestBody.coord_start, requestBody.coord_end].filter(Boolean)
            .concat(requestBody.coord_waypoints || []);
        return selected.some(function (point) {
            return requested.some(function (coord) {
                return coord && coord.lng === point.lng && coord.lat === point.lat;
            });
        });
    }

    // 设置起终点后自动触发路径规划
    function autoPlanFromMapPoints() {
        var s = state.mapPoints.start, e = state.mapPoints.end;
        if (!s || !e) return;
        if (state.mapPointRouteActive || state.routeStore && state.routeStore.current()) clearMapPointRoute();
        var vias = state.mapPoints.via;
        var query = vias.length
            ? '从地图标记的起点途经地图标记点到终点'
            : '从地图起点到地图终点';
        state.mapPointRouteActive = true;
        handleNlSubmit(query, false, null, {
            coord_start: { lng: s.lng, lat: s.lat, name: '地图起点' },
            coord_end: { lng: e.lng, lat: e.lat, name: '地图终点' },
            coord_waypoints: vias.map(function (v) { return { lng: v.lng, lat: v.lat }; }),
        }, snapshotMapPoints());
    }

    // ===== 持续跟踪定位（核心逻辑：共享 state.locateWatchId，防重复 watch） =====
    // 统一入口：startTracking() 负责清理旧 watch + 启动新 watch（AMap优先 + 原生fallback）
    // 调用方：autoLocateSilent（页面加载自动定位）

    // 页面上可见的定位状态指示器（微信里看不到 console，靠这个排查）
    // success/warn/info 类型：成功后自动淡出隐藏；error 类型：保持显示直到用户点掉
    var _locStatusHideTimer = null;
    function _setLocStatus(text, type) {
        var el = document.getElementById('loc-status');
        if (!el) return;
        if (_locStatusHideTimer) { clearTimeout(_locStatusHideTimer); _locStatusHideTimer = null; }
        if (!text) { el.style.display = 'none'; return; }
        el.textContent = text;
        el.style.display = 'block';
        el.style.opacity = '1';
        el.className = 'loc-status loc-status--' + (type || 'info');
        console.log('[LOC_STATUS]', text);
        // 成功/信息/警告 → 3 秒后淡出隐藏
        if (type !== 'error') {
            _locStatusHideTimer = setTimeout(function () {
                el.style.transition = 'opacity 0.5s ease';
                el.style.opacity = '0';
                setTimeout(function () { el.style.display = 'none'; el.style.transition = ''; }, 500);
            }, 3000);
        }
    }

    function _clearWatch() {
        if (state.locateWatchId != null) {
            try { navigator.geolocation.clearWatch(state.locateWatchId); } catch(e) {}
            state.locateWatchId = null;
        }
        if (state._nativeLocActive) {
            try { window.WhuWalkerLocation && window.WhuWalkerLocation.stop(); } catch(e) {}
            state._nativeLocActive = false;
        }
        if (state._amapWatchListener) {
            try { state._amapGeolocation && state._amapGeolocation.clearWatch(state._amapWatchListener); } catch(e) {}
            state._amapWatchListener = null;
        }
    }

    // ===== APK 原生定位桥（LocationManager + 罗盘融合） =====
    // 国产 ROM WebView 的 navigator.geolocation 存在授权时序问题：页面加载即 watch，
    // 授权弹窗未返回时拿到 PERMISSION_DENIED，句柄死亡且授权返回后不会恢复。
    // 原生桥在"权限授予成功"回调里才启动定位，时序确定；同时回传设备航向（罗盘/GPS bearing）。
    function _startAppNativeWatch(silent) {
        state._nativeLocActive = true;
        window.__whuWalkerLocationCallback = function (d) {
            if (!d) return;
            if (d.event === 'error') {
                // 权限拒绝 / 系统定位服务关闭 → AMap WiFi 定位（服务端基站/WiFi 定位）仍可能可用
                if (!state._amapFallbackStarted) {
                    state._amapFallbackStarted = true;
                    console.warn('[TRACK] 原生桥定位失败:', d.message, '→ 降级 AMap');
                    _setLocStatus((d.message || '定位失败') + '，切换 WiFi 定位…', 'warn');
                    _startAmapWatch(silent);
                }
                return;
            }
            if (d.event !== 'fix') return;
            if (!silent) setLocateBtnState(false);
            var hd = (d.heading == null) ? null : d.heading;
            renderUserLocation(d.lng, d.lat, d.accuracy > 0 ? d.accuracy : 0, false, false, hd, d.speed || 0);
        };
        _setLocStatus('定位中…（原生 GPS）', 'info');
        try {
            window.WhuWalkerLocation.start();
            console.log('[TRACK] APP 原生定位桥已启动');
        } catch (e) {
            console.warn('[TRACK] 原生桥调用异常 → 降级网页定位:', e);
            state._nativeLocActive = false;
            _startAmapWatch(silent);
        }
    }

    function _startNativeWatch(silent, onError) {
        if (!navigator.geolocation || window.isSecureContext === false) {
            var reason = !navigator.geolocation ? '浏览器不支持定位' : '非 HTTPS 环境';
            if (!silent) showError('定位不可用', reason);
            _setLocStatus('定位不可用：' + reason, 'error');
            console.warn('[TRACK] 环境不支持原生定位:', reason);
            return false;
        }
        _clearWatch();
        _setLocStatus('定位中…（原生 GPS）', 'info');
        state.locateWatchId = navigator.geolocation.watchPosition(
            function (pos) {
                if (!silent) setLocateBtnState(false);
                var c = pos.coords;
                // 系统航向/速度（手机浏览器一般支持；静止时 heading 常为 null）
                var hd = (c.heading != null && isFinite(c.heading)) ? c.heading : null;
                var sp = (c.speed != null && isFinite(c.speed)) ? c.speed : null;
                // navigator.geolocation 坐标系不确定（国内 Windows 可能返回 GCJ-02），
                // 用校园中心锚点猜测并统一为 WGS-84
                var wgs = _guessWgs84FromNavigator(c.longitude, c.latitude);
                renderUserLocation(wgs[0], wgs[1], c.accuracy, false, false, hd, sp);
                console.log('[TRACK] 原生位置更新(猜系后 WGS-84):', wgs[1].toFixed(4), wgs[0].toFixed(4),
                    '原始:', c.latitude.toFixed(4), c.longitude.toFixed(4),
                    hd != null ? 'heading=' + Math.round(hd) : '');
            },
            function (err) {
                var msg = '定位失败：';
                if (err && err.code === 1) msg += '你拒绝了定位授权';
                else if (err && err.code === 2) msg += '获取不到位置信号';
                else if (err && err.code === 3) msg += '定位超时';
                else msg += '请稍后再试';
                if (!silent) { setLocateBtnState(false); showError('定位失败', msg); }
                console.warn('[TRACK] 原生 watch 错误:', err && err.code, err && err.message);
                // 句柄已失效，置空（否则降级判断和 _clearWatch 会拿到死句柄）
                state.locateWatchId = null;
                // 防重入：watch 的 error 可能连续触发多次，降级只做一次
                if (onError && !state._amapFallbackStarted) {
                    state._amapFallbackStarted = true;
                    _setLocStatus(msg + '，切换 WiFi 定位…', 'warn');
                    onError(err);
                } else if (!onError) {
                    _setLocStatus(msg, 'error');
                }
            },
            { enableHighAccuracy: true, timeout: 8000, maximumAge: 3000 }
        );
        _setLocStatus('已启动原生 GPS 跟踪', 'success');
        console.log('[TRACK] 原生 watch 已启动, id:', state.locateWatchId);
        return true;
    }

    function _startAmapWatch(silent) {
        _setLocStatus('切换 WiFi 定位…', 'info');
        // SDK 异步加载，最多等 4 秒——原生失败后它是唯一兜底，绝不能静默放弃
        // （旧版 SDK 未就绪时直接 return false，定位链整个死亡，蓝点消失）
        var waited = 0, interval = 150, maxWait = 4000;
        var tick = setInterval(function () {
            waited += interval;
            if (window.AMap && window.AMap.plugin) {
                clearInterval(tick);
                try {
                    AMap.plugin('AMap.Geolocation', function () {
                        try {
                            // 不要 _clearWatch()：原生 watch 报错后仍可能自愈出点，
                            // 两条流并存谁先到用谁，renderUserLocation 幂等
                            // convert: true 强制 AMap 返回 GCJ-02——PC 浏览器上 navigator.geolocation
                            // 坐标系不可信（Windows Location Platform 在国内可能返回已被偏移的 GCJ-02），
                            // 让 AMap 统一处理转换，避免下游链路二次偏移
                            var geo = new AMap.Geolocation({
                                enableHighAccuracy: true, timeout: 10000, maximumAge: 30000, convert: true,
                            });
                            state._amapGeolocation = geo;
                            state._amapWatchListener = geo.watchPosition(function (status, result) {
                                if (status === 'complete' && result && result.position) {
                                    var gcjLng = result.position.lng, gcjLat = result.position.lat;
                                    var wgs = gcj02ToWgs84(gcjLng, gcjLat);
                                    if (!silent) setLocateBtnState(false);
                                    renderUserLocation(wgs[0], wgs[1], result.accuracy || 0);
                                    console.log('[TRACK] AMap 位置更新, GCJ:', gcjLng.toFixed(4), gcjLat.toFixed(4));
                                } else {
                                    console.warn('[TRACK] AMap watch 状态:', status);
                                    if (!state.locateWatchId) {
                                        _setLocStatus('定位失败：GPS 和 WiFi 定位都不可用', 'error');
                                    }
                                }
                            });
                            _setLocStatus('已切换 AMap WiFi 定位', 'success');
                            console.log('[TRACK] AMap watch 已启动');
                        } catch (e) {
                            console.warn('[TRACK] AMap watch 异常:', e.message, '→ 重试原生');
                            _setLocStatus('WiFi 定位异常，重试 GPS…', 'warn');
                            _startNativeWatch(silent);
                        }
                    });
                } catch (e) {
                    console.warn('[TRACK] AMap.plugin 异常:', e.message, '→ 重试原生');
                    _startNativeWatch(silent);
                }
            } else if (waited >= maxWait) {
                clearInterval(tick);
                console.warn('[TRACK] AMap SDK 4 秒未加载 → 回退原生 GPS 重试');
                _setLocStatus('WiFi 定位不可用，重试 GPS…', 'warn');
                _startNativeWatch(silent);
            }
        }, interval);
        return true;
    }

    // ===== 两段式定位：先粗后精 =====
    // 高精度 GPS 首次锁定需下载卫星星历（冷启动 5-30s，热启动 1-5s），干等体验差。
    // 先用低精度网络定位（WiFi/基站 + 操作系统缓存，maximumAge 5 分钟）秒出粗点，
    // 粗点标记 provisional，GPS 精修一到即无条件接管。
    function _quickFix() {
        if (!navigator.geolocation) return;
        try {
            navigator.geolocation.getCurrentPosition(
                function (pos) {
                    // 高精度流已经出点（非 provisional）→ 粗点别来添乱
                    if (state._dispFix && !state._dispFix.provisional) return;
                    var wgs = _guessWgs84FromNavigator(pos.coords.longitude, pos.coords.latitude);
                    renderUserLocation(wgs[0], wgs[1], pos.coords.accuracy, false, true);
                    _setLocStatus('已粗定位（误差约 ' + Math.round(pos.coords.accuracy || 999) + ' 米），GPS 精修中…', 'info');
                    console.log('[TRACK] 粗定位(网络,猜系后):', wgs[1].toFixed(4), wgs[0].toFixed(4),
                        '原始:', pos.coords.latitude.toFixed(4), pos.coords.longitude.toFixed(4),
                        '±', Math.round(pos.coords.accuracy || 0), 'm');
                },
                function (err) {
                    console.log('[TRACK] 粗定位失败 code:', err && err.code, '（GPS 主流程继续）');
                },
                { enableHighAccuracy: false, timeout: 6000, maximumAge: 300000 }
            );
        } catch (e) { /* 忽略 */ }
    }

    // 对外：启动持续跟踪（原生 GPS 优先，0 等待；AMap 仅作为台式机等无 GPS 环境的兜底）
    // 原生 GPS 手机上通常 1-3 秒出首个定位点，比等 AMap SDK 加载 + 服务器 WiFi 定位快
    function startTracking(silent) {
        if (state.productMode === 'planning' || !isNavigationDevice()) {
            _clearWatch();
            console.log('[TRACK] 电脑规划模式不启动定位');
            return false;
        }
        _clearWatch();
        state._amapFallbackStarted = false;
        // APK 内走原生定位桥：无授权时序问题，且带罗盘/GPS 融合航向
        if (window.WhuWalkerLocation && typeof window.WhuWalkerLocation.start === 'function') {
            _startAppNativeWatch(silent);
            return;
        }
        _setLocStatus('定位中…（原生 GPS）', 'info');
        _quickFix();  // 并行：网络粗定位秒出占位点，GPS 随后精修
        // 原生启动失败（不支持/非HTTPS）或定位出错（超时/无信号）→ 降级 AMap WiFi 定位
        var fallbackFn = function (err) {
            // code 1 = 用户拒绝授权，AMap 高精度定位同样需要授权，降级无意义
            if (err && err.code === 1) return;
            console.warn('[TRACK] 原生定位失败 code:', err && err.code, '→ 降级 AMap SDK');
            _startAmapWatch(silent);
        };
        if (!_startNativeWatch(silent, fallbackFn)) {
            // 环境根本不支持原生定位 → 直接 AMap（其自带 WiFi/IP 定位，不依赖浏览器权限）
            console.warn('[TRACK] 原生定位不可用 → 降级 AMap SDK');
            _startAmapWatch(silent);
        }
    }

    // 首页地图展示的是武大主校区；约 3 km 范围覆盖校园边缘及相邻校门，
    // 远处的 GPS 仍用于位置状态/导航，但不把地图镜头带离校园。
    var WHU_CAMPUS_VIEW_RADIUS_M = 3000;

    function isWithinCampusView(lng, lat) {
        if (!isFinite(lng) || !isFinite(lat)) return false;
        var centerWgs = gcj02ToWgs84(MAP_CENTER[0], MAP_CENTER[1]);
        return _haversineMeters(lat, lng, centerWgs[1], centerWgs[0]) <= WHU_CAMPUS_VIEW_RADIUS_M;
    }

    function consumeFirstUserLocationFix(isCached, mapReady) {
        if (isCached || !mapReady) return false;
        var isFirstFix = !state._hadUserLocation;
        state._hadUserLocation = true;
        return isFirstFix;
    }

    function locationViewAction(lng, lat, canChooseInitialView, isCached, isManual,
        campusViewAlreadyKept, userInteractedWithMap) {
        if (isManual) return 'user';
        if (isCached || !canChooseInitialView) return 'preserve';
        if (userInteractedWithMap) return 'preserve';
        if (isWithinCampusView(lng, lat)) return 'user';
        return campusViewAlreadyKept ? 'preserve' : 'campus';
    }

    function bindUserMapInteraction(map) {
        var container = map && map.getContainer && map.getContainer();
        if (!container || !container.addEventListener || state._mapIntentContainer === container) return;
        state._mapIntentContainer = container;
        ['pointerdown', 'touchstart', 'mousedown', 'wheel', 'keydown'].forEach(function (type) {
            container.addEventListener(type, function () {
                state._userInteractedWithMap = true;
            }, { capture: true, passive: true });
        });
    }

    function clampMobileMapHeight(viewportHeight, requestedHeight) {
        var height = Math.max(1, Number(viewportHeight) || 1);
        var requested = Number(requestedHeight);
        if (!isFinite(requested)) requested = height * 0.55;
        var minimum = Math.min(180, Math.max(100, height * 0.28));
        var maximum = Math.max(minimum, Math.min(height * 0.76, height - 240));
        return Math.round(Math.max(minimum, Math.min(maximum, requested)));
    }

    function shouldFollowChatUpdates(scrollTop, clientHeight, scrollHeight, tolerance) {
        var safeTolerance = isFinite(tolerance) ? Math.max(0, tolerance) : 64;
        return (Number(scrollHeight) || 0) - (Number(scrollTop) || 0) - (Number(clientHeight) || 0)
            <= safeTolerance;
    }

    function chatIsNearLatest(chatContent) {
        return !!chatContent && shouldFollowChatUpdates(
            chatContent.scrollTop, chatContent.clientHeight, chatContent.scrollHeight, 64
        );
    }

    function scrollChatToLatest(chatContent, force) {
        if (!chatContent || (!force && !chatIsNearLatest(chatContent))) return;
        setTimeout(function () {
            chatContent.scrollTo({ top: chatContent.scrollHeight, behavior: 'smooth' });
        }, 100);
    }

    // ===== 页面加载自动定位（静默版，持续跟踪） =====
    function autoLocateSilent() {
        if (state.productMode === 'planning' || !isNavigationDevice()) {
            _clearWatch();
            console.log('[AUTO_LOC] 电脑规划模式已禁用自动定位');
            return;
        }
        if (!state.map) { console.warn('[AUTO_LOC] map 未就绪，跳过'); _setLocStatus('地图未就绪，跳过自动定位', 'warn'); return; }
        // PC 端无 GPS，缓存位置可能来自上次 IP 定位（偏差数百米到公里），
        // 直接显示会误导用户，跳过缓存只走实时定位（不达标则引导手动选点）
        var _isMobileUA = /Mobile|Android|iPhone|iPad/i.test(navigator.userAgent);
        // 先画上次的位置（24 小时内有效，标注时长）：感知秒出，GPS/网络定位在后台接管
        if (_isMobileUA) {
            try {
                var raw = localStorage.getItem('whu_last_pos');
                if (raw) {
                    var c = JSON.parse(raw);
                    var ageMin = (Date.now() - (c.ts || 0)) / 60000;
                    if (c.lng && c.lat && ageMin >= 0 && ageMin <= 24 * 60) {
                        renderUserLocation(c.lng, c.lat, c.accuracy || 0, false, true);
                        var ageTxt = ageMin < 1 ? '刚刚' : (ageMin < 60 ? Math.round(ageMin) + ' 分钟前' : Math.round(ageMin / 60) + ' 小时前');
                        _setLocStatus('先显示 ' + ageTxt + ' 的位置，正在重新定位…', 'info');
                        console.log('[AUTO_LOC] 使用缓存位置（' + Math.round(ageMin) + ' 分钟前），等待新定位接管');
                    }
                }
            } catch (e) { /* 缓存损坏忽略 */ }
        } else {
            console.log('[AUTO_LOC] PC 环境跳过缓存位置，等待实时定位或手动选点');
        }
        _setLocStatus('自动定位中…', 'info');
        startTracking(true);
    }

    function renderUserLocation(lng, lat, accuracy, isManual, isCached, heading, speed) {
        // 电脑端只接受用户在地图上的明确选点；系统、浏览器和地图商定位一律不进入状态。
        if ((state.productMode === 'planning' || !isNavigationDevice()) && !isManual) {
            console.log('[TRACK] 电脑规划模式忽略非手动位置');
            return;
        }
        // ===== PC 端精度门槛 =====
        // PC 无 GPS，定位多走 Windows 位置服务（WiFi 指纹）或纯 IP，精度差时会乱跳到
        // 校外/其他校区。手机端 GPS/网络定位稳定，不受此门槛影响。
        // 拦截条件：非手机 UA + 非手动点 + (精度 >100m 或 accuracy=0)
        //   缓存点也拦——避免上次 IP 定位留下的错位置继续误导 PC 用户
        // 拦截后：不画蓝点、不更新 state._dispFix，只弹横幅引导手动选点
        var isMobileUA = /Mobile|Android|iPhone|iPad/i.test(navigator.userAgent);
        if (!isMobileUA && !isManual) {
            var pcAcc = accuracy || 0;
            if (pcAcc === 0 || pcAcc > 100) {
                console.warn('[TRACK] PC 精度不足（±' + (pcAcc || '∞') + 'm），拒显蓝点，引导手动选点');
                _setLocStatus('电脑定位精度不足，请点右上角「起点」手动选点', 'warn');
                showLowAccuracyWarning(pcAcc);
                return;  // 不画蓝点、不更新 state
            }
        }

        // heading/speed：设备航向（°，0北90东）。独立于坐标——坐标被噪声抑制钉住时，
        // 原地转身航向仍然要透传给导航箭头
        var hasHeading = heading != null && isFinite(heading);
        if (hasHeading) heading = ((heading % 360) + 360) % 360;
        // ===== GPS 抖动抑制 =====
        // 高德 App 定位稳是因为有传感器融合（GPS+WiFi+惯导）+ 卡尔曼滤波 + 路网吸附；
        // 网页只能拿到操作系统吐出的原始定位流（±10-50m 噪声），蓝点稳不稳全靠前端自己滤：
        //   1) 隐含速度 > 25 m/s → 异常跳变（多径反射/基站切换），直接丢弃
        //   2) 新点落在旧点精度圈噪声内 → 蓝点钉住不动，只收紧精度圈
        //      连续钉住 10 次后强制校准一次，防止慢速漂移让蓝点永久滞后
        var disp = state._dispFix;
        if (!isManual && !isCached && disp) {
            var dtSec = (Date.now() - disp.ts) / 1000;
            var jump = _haversineMeters(disp.lat, disp.lng, lat, lng);
            // 缓存/手动点位只是"临时占位"（provisional），真实 GPS 一到就无条件接管——
            // 否则换了个地方打开页面，GPS 会被判"隐含速度异常"永久丢弃，蓝点卡在旧位置
            var provisional = !!disp.provisional;
            var fresh = dtSec > 0 && dtSec <= 5;  // 速度检测只对新鲜基线有意义
            var reject = false;
            if (!provisional && fresh) {
                if (jump / dtSec > 25) {
                    reject = true;  // 多径反射/基站切换造成的异常跳变
                } else if (jump <= Math.max(disp.acc || 0, accuracy || 0) * 0.7 && (disp.holds || 0) < 10) {
                    // 噪声内漂移：蓝点钉住不动，只收紧精度圈；10 次后强制校准防滞后
                    lng = disp.lng; lat = disp.lat;
                    accuracy = Math.min(accuracy || 1e9, disp.acc || 1e9);
                    disp.holds = (disp.holds || 0) + 1;
                } else {
                    disp.holds = 0;
                }
            }
            if (reject) {
                state._rejects = (state._rejects || 0) + 1;
                // 位置点丢弃，但设备航向（罗盘/GPS bearing）仍然有效：原地转身时箭头必须能动
                if (state.userLocation && hasHeading) {
                    state.userLocation.heading = heading;
                    state.userLocation.speed = speed || 0;
                    if (state.locSubscribers.length) {
                        state.locSubscribers.slice().forEach(function (fn) {
                            try { fn(state.userLocation); } catch (e) { console.warn('[LOC] 订阅者异常:', e); }
                        });
                    }
                }
                if (state._rejects < 5) {
                    console.log('[TRACK] 丢弃异常跳变点(' + state._rejects + '/5): ' + Math.round(jump) + 'm/' + dtSec.toFixed(1) + 's');
                    return;  // 蓝点保持原位不动
                }
                // 连续 5 次被拒 → 基线本身坏了，强制接受本次修复
                console.warn('[TRACK] 连续丢弃 5 次，强制接受修复');
            }
            state._rejects = 0;
        }
        state._dispFix = {
            lat: lat, lng: lng, acc: accuracy || 0, ts: Date.now(),
            holds: (disp && disp.holds) || 0,
            provisional: !!(isManual || isCached),
        };

        // GPS 返回 WGS-84；高德瓦片 GCJ-02，渲染前需转换
        var gcj = wgs84ToGcj02(lng, lat);
        var gcjLng = gcj[0], gcjLat = gcj[1];
        var prevLoc = state.userLocation;
        state.userLocation = {
            lng: lng, lat: lat,
            gcjLng: gcjLng, gcjLat: gcjLat,
            accuracy: accuracy || 0,
            manual: !!isManual,
            // 本次无航向（如 AMap/WiFi 源）时保留上一次航向，避免箭头突然朝北
            heading: hasHeading ? heading : (prevLoc ? prevLoc.heading : null),
            speed: (speed != null && isFinite(speed)) ? speed : (prevLoc ? prevLoc.speed : 0),
        };
        // 定位订阅者（实时导航引擎）：只推送，不干预蓝点逻辑
        if (state.locSubscribers.length) {
            state.locSubscribers.slice().forEach(function (fn) {
                try { fn(state.userLocation); } catch (e) { console.warn('[LOC] 订阅者异常:', e); }
            });
        }
        // 缓存 GPS 修复（非手动点位、精度可信时），供下次打开秒显
        if (!isManual && !isCached && accuracy > 0 && accuracy <= 300) {
            try {
                localStorage.setItem('whu_last_pos', JSON.stringify({
                    lng: lng, lat: lat, accuracy: Math.round(accuracy), ts: Date.now()
                }));
            } catch (e) { /* 隐私模式等场景忽略 */ }
        }
        // 缓存位置只作临时显示，不消耗首次实时定位，也不抢地图镜头。
        var isFirstFix = consumeFirstUserLocationFix(isCached, !!state.map);
        if (isFirstFix || isManual) {
            var accText = accuracy > 0 ? '（精度约 ' + Math.round(accuracy) + 'm）' : '';
            _setLocStatus('已定位' + accText, 'success');
        }
        if (isCached) {
            _setLocStatus('已显示上次位置，正在精确刷新…', 'info');
        }
        if (!state.map) return;

        // 精度警告：accuracy 过大或无数据 → 显示黄色警告 banner（缓存点位跳过，等真实修复再判）
        // 导航中不弹：它固定在顶部 12px，会与导航指令卡重叠；导航态有专属 #nav-gps-warn 黄条
        var acc = accuracy || 0;
        var inNavigation = document.body.classList.contains('nav-running');
        if (!isManual && !isCached && (acc > 500 || acc === 0) && !inNavigation) {
            showLowAccuracyWarning(acc);
        } else if (isManual || acc <= 500 || inNavigation) {
            hideLowAccuracyWarning();
        }

        // 精度圈
        if (state.userAccuracyCircle) {
            state.userAccuracyCircle.setLatLng([gcjLat, gcjLng]);
            if (acc > 0) state.userAccuracyCircle.setRadius(acc);
            state.userAccuracyCircle.setStyle({
                opacity: isManual ? 0.3 : 0.5,
                fillOpacity: isManual ? 0.08 : 0.12,
            });
        } else if (acc > 0) {
            state.userAccuracyCircle = L.circle([gcjLat, gcjLng], {
                radius: acc,
                color: isManual ? '#E67E22' : '#2B7CFF',
                weight: 1, opacity: 0.5,
                fillColor: isManual ? '#E67E22' : '#2B7CFF',
                fillOpacity: isManual ? 0.08 : 0.12,
                interactive: false,
            }).addTo(state.map);
        }

        // 蓝点标记：已有 marker → setLatLng 更新位置（不销毁重建，避免闪烁）
        // 没有 marker → 新建（首次定位 / 手动设点后被移除）
        var dotColor = isManual ? '#E67E22' : '#2B7CFF';
        var dotHtml = '<div style="position:relative;width:20px;height:20px;">'
            + '<div style="position:absolute;inset:0;border-radius:50%;background:' + dotColor + ';opacity:0.25;"></div>'
            + '<div style="position:absolute;left:5px;top:5px;width:10px;height:10px;border-radius:50%;'
            + 'background:' + dotColor + ';border:2px solid #fff;box-sizing:border-box;box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>'
            + '</div>';
        var tooltipTxt = isManual
            ? '我的位置（手动设置）· 说「从我这到樱顶」'
            : '我的位置' + (acc > 0 ? '（精度约 ' + Math.round(acc) + 'm）' : '（IP定位精度可能低）') + ' · 说「从我这到樱顶」';

        if (state.userMarker) {
            // 持续跟踪更新：只移动位置（setLatLng 不闪烁）
            state.userMarker.setLatLng([gcjLat, gcjLng]);
            // 颜色变化时（手动↔GPS）才重建 marker——这种情况极少
            if (state.userMarker._dotColor !== dotColor) {
                state.map.removeLayer(state.userMarker);
                state.userMarker = null;
            }
        }
        if (!state.userMarker) {
            if (state.userMarkerRaw) { state.map.removeLayer(state.userMarkerRaw); state.userMarkerRaw = null; }
            // 创建时不带过渡类：首次放置会从地图原点(0,0)飞入，落稳后再启用平滑
            state.userMarker = divMarker([gcjLat, gcjLng], dotHtml, [20, 20], [10, 10], '我的位置');
            state.userMarker._dotColor = dotColor;
            state.userMarker.bindTooltip(tooltipTxt, { direction: 'top', offset: [0, -12], opacity: 0.95 })
                .addTo(state.map);
            var dotEl = state.userMarker.getElement && state.userMarker.getElement();
            if (dotEl) setTimeout(function () { dotEl.classList.add('user-dot-smooth'); }, 150);
        }

        // 首次实时定位在校园附近聚焦用户；远处定位明确回到武大校园全局视野。
        var canChooseInitialView = !state._hasCenteredOnLiveLocation && !!state.map;
        var viewAction = locationViewAction(
            lng, lat, canChooseInitialView, isCached, isManual,
            state._outsideCampusLocationNoticeShown, state._userInteractedWithMap
        );
        if (viewAction === 'user') {
            state.map.setView([gcjLat, gcjLng], 17);
            state._hasCenteredOnLiveLocation = true;
        } else if (viewAction === 'campus') {
            state.map.setView(gcjToLatLng(MAP_CENTER[0], MAP_CENTER[1]), MAP_ZOOM);
            if (!state._outsideCampusLocationNoticeShown) {
                state._outsideCampusLocationNoticeShown = true;
                var outsideCampusMessage = '当前位置不在武大校园附近，已保留校园地图视野';
                _setLocStatus(outsideCampusMessage, 'warn');
                showTopBanner(outsideCampusMessage, 'info');
            }
        }
    }

    // 精度警告 banner（桌面 Chrome IP 定位精度差时弹）
    var _lowAccBanner = null;
    function showLowAccuracyWarning(acc) {
        if (_lowAccBanner) return;
        var msg = acc === 0
            ? '⚠️ 当前位置由 IP 推断，精度差。点右上角「起点」按钮在地图上手动选点。'
            : '⚠️ 定位精度约 ' + Math.round(acc) + 'm（' + (acc > 1000 ? '误差较大' : '可能不够准') + '）。点「起点」按钮手动选点。';
        _lowAccBanner = showTopBanner(msg, 'warning');
    }
    function hideLowAccuracyWarning() {
        if (_lowAccBanner) {
            _lowAccBanner.remove();
            _lowAccBanner = null;
        }
    }

    // 轻量顶部 banner（精度警告 / 手动选点提示）
    function showTopBanner(msg, type) {
        var el = document.createElement('div');
        var bg = type === 'warning' ? '#FFF3CD' : type === 'success' ? '#D4EDDA' : '#D1ECF1';
        var color = type === 'warning' ? '#856404' : type === 'success' ? '#155724' : '#0C5460';
        var border = type === 'warning' ? '#FFC107' : type === 'success' ? '#28A745' : '#17A2B8';
        el.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:1000;'
            + 'padding:8px 16px;background:' + bg + ';color:' + color + ';border:1px solid ' + border + ';'
            + 'border-radius:6px;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,0.15);'
            + 'max-width:90%;text-align:center;pointer-events:auto;cursor:pointer;';
        el.textContent = msg;
        el.addEventListener('click', function () { el.remove(); });
        document.body.appendChild(el);
        setTimeout(function () { if (el.parentNode) el.remove(); }, 10000);  // 10 秒自动消失
        return el;
    }

    function renderRoute(routeData) {
        if (!state.map) return;

        state.latestRoute = routeData;
        // 导航态：隐藏最短路径对照线、不 fitBounds（地图由导航引擎北朝上跟随）
        var navActive = window.WhuWalkerNavigation && window.WhuWalkerNavigation.isActive();

        clearMap();

        var recommended = routeData.recommended || [];
        var shortest = routeData.shortest || [];

        if (recommended.length > 0) {
            // 后端返回 GCJ-02，高德瓦片也是 GCJ-02，直接入图（[lat, lng]）
            var recPath = recommended.map(function (c) {
                return gcjToLatLng(c.lng, c.lat);
            });

            // 推荐线配色按出行方式：步行=樱花粉 / 骑行=松绿 / 驾车=黛蓝（响应 mode 优先，兜底当前选择器）
            var routeMode = (routeData && routeData.mode && TRAVEL_MODES[routeData.mode])
                ? routeData.mode : state.travelMode;
            var recColor = (TRAVEL_MODES[routeMode] || TRAVEL_MODES.walk).color;

            state.recommendedLine = L.polyline(recPath, {
                color: recColor,
                weight: navActive ? 7 : 6,
                opacity: 0.9,
                lineJoin: 'round',
                lineCap: 'round',
            }).addTo(state.map);
        }

        if (!navActive && routeData.shortest_baseline_feasible !== false && shortest.length > 0) {
            // 最短路径：灰色虚线对照
            var shortPath = shortest.map(function (c) {
                return gcjToLatLng(c.lng, c.lat);
            });

            state.shortestLine = L.polyline(shortPath, {
                color: '#B5B0AB',
                weight: 4,
                opacity: 0.7,
                dashArray: '6,8',
                lineJoin: 'round',
                lineCap: 'round',
            }).addTo(state.map);
        }

        var pois = routeData.pois || [];
        // 只标前 8 个途经点，避免大量标签遮挡路线；用小圆点，悬停显示名称
        pois.slice(0, 8).forEach(function (poi) {
            var lng = poi.lng || poi.lon;
            var lat = poi.lat;
            if (lng == null || lat == null) return;

            var dotHtml = '<div style="width:10px;height:10px;border-radius:50%;background:#D4915C;'
                + 'border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.35);"></div>';
            var marker = divMarker(gcjToLatLng(lng, lat), dotHtml, [10, 10], [5, 5], poi.name || '')
                .addTo(state.map);
            state.poiMarkers.push(marker);
        });

        var bounds = [];
        if (state.recommendedLine) bounds.push(state.recommendedLine.getBounds());
        if (state.shortestLine) bounds.push(state.shortestLine.getBounds());
        state.poiMarkers.forEach(function (m) { bounds.push(m.getLatLng()); });

        if (navActive) {
            // 导航中：新路线（偏航重算/切方式）交给引擎重建投影，视角保持北朝上跟随
            if (window.WhuWalkerNavigation) window.WhuWalkerNavigation.update(routeData);
        } else if (bounds.length > 0) {
            var b = bounds[0];
            bounds.slice(1).forEach(function (x) { b.extend(x); });
            state.map.fitBounds(b, { paddingTopLeft: [40, 90], paddingBottomRight: [40, 40], maxZoom: 17 });
        }
    }

    function clearMap() {
        if (state.recommendedLine) {
            state.map.removeLayer(state.recommendedLine);
            state.recommendedLine = null;
        }
        if (state.shortestLine) {
            state.map.removeLayer(state.shortestLine);
            state.shortestLine = null;
        }
        state.poiMarkers.forEach(function (m) { state.map.removeLayer(m); });
        state.poiMarkers = [];
    }

    // ===================== 路况事件 =====================

    // 路况类型对应的图标、颜色、线型（实线=硬封，虚线=软影响）
    var ROAD_CONDITION_STYLES = {
        closure:     { icon: '🚫', color: '#C0392B', label: '封闭', dash: null },
        construction:{ icon: '🚧', color: '#E67E22', label: '施工', dash: '10 7' },
        event:       { icon: '🎉', color: '#9B59B6', label: '活动', dash: '4 7' },
        flooding:    { icon: '🌊', color: '#2980B9', label: '积水', dash: '7 5' },
        accident:    { icon: '⚠️', color: '#F39C12', label: '事故', dash: '12 7' },
    };
    // 仅供旧版事件记录兼容；新记录必须使用 action + affected_modes。
    var CONDITION_MODE_EFFECTS = {
        closure:      { walk: 'block', bike: 'block', drive: 'block' },
        construction: { walk: 1.5,     bike: 'block', drive: 'block' },
        flooding:     { walk: 'block', bike: 'block', drive: 1.5 },
        accident:     { walk: 1.3,     bike: 1.3,     drive: 'block' },
        event:        { walk: 1.2,     bike: 1.2,     drive: 'block' },
    };
    var COND_STATUS_LABELS = { active: '生效中', scheduled: '未开始', expired: '已结束' };

    function getRoadConditionModeEffect(cond, mode) {
        if (cond && cond.action != null) {
            var affected = Array.isArray(cond.affected_modes) ? cond.affected_modes : [];
            if (!['notice', 'slowdown', 'closure'].includes(cond.action)
                    || !affected.length
                    || affected.some(function (item) { return !['walk', 'bike', 'drive'].includes(item); })) {
                return { status: 'unknown' };
            }
            if (!affected.includes(mode)) return { status: 'unaffected' };
            if (cond.action === 'closure') return { status: 'blocked' };
            if (cond.action === 'slowdown') {
                var multiplier = Number(cond.cost_multiplier);
                return Number.isFinite(multiplier) && multiplier >= 1
                    ? { status: 'slowdown', costMultiplier: multiplier }
                    : { status: 'unknown' };
            }
            if (cond.action === 'notice') return { status: 'notice' };
            return { status: 'unknown' };
        }
        if (cond && Object.prototype.hasOwnProperty.call(cond, 'blocked_modes')) {
            if (!Array.isArray(cond.blocked_modes) || !cond.blocked_modes.length
                    || cond.blocked_modes.some(function (item) { return !['walk', 'bike', 'drive'].includes(item); })) {
                return { status: 'unknown' };
            }
            return { status: cond.blocked_modes.includes(mode) ? 'blocked' : 'unaffected' };
        }
        if (cond && (Object.prototype.hasOwnProperty.call(cond, 'action')
                || Object.prototype.hasOwnProperty.call(cond, 'affected_modes'))) {
            return { status: 'unknown' };
        }
        var legacyEffect = (CONDITION_MODE_EFFECTS[(cond || {}).type] || {})[mode];
        if (legacyEffect === 'block') return { status: 'blocked' };
        if (typeof legacyEffect === 'number') {
            return { status: 'slowdown', costMultiplier: legacyEffect };
        }
        return { status: 'unknown' };
    }

    var ROAD_CONDITION_EFFECT_STYLES = {
        blocked:    { icon: '🚫', color: '#C0392B', label: '封闭', dash: null },
        slowdown:   { icon: '⚠️', color: '#D97706', label: '缓行', dash: '10 6' },
        notice:     { icon: 'ℹ️', color: '#2878A8', label: '仅提示', dash: '3 6' },
        unaffected: { icon: 'ℹ️', color: '#2878A8', label: '本方式不受影响', dash: '3 6' },
        unknown:    { icon: '⚠️', color: '#6B7280', label: '影响待核实', dash: '3 6' },
    };

    function getRoadConditionDisplayStyle(cond, mode) {
        var type = (cond || {}).type;
        var baseStyle = ROAD_CONDITION_STYLES[type]
            || { icon: '⚠️', color: '#999', label: type || '路况', dash: null };
        var hasExplicitEffect = !!cond && (
            cond.action != null
            || Object.prototype.hasOwnProperty.call(cond, 'affected_modes')
            || Object.prototype.hasOwnProperty.call(cond, 'blocked_modes')
        );
        if (!hasExplicitEffect) return baseStyle;
        var effect = getRoadConditionModeEffect(cond, mode);
        return ROAD_CONDITION_EFFECT_STYLES[effect.status] || ROAD_CONDITION_EFFECT_STYLES.unknown;
    }

    function describeRoadConditionModeEffect(cond, mode, modeName) {
        var effect = getRoadConditionModeEffect(cond, mode);
        if (effect.status === 'blocked') return '当前方式（' + modeName + '）禁止通行，路线会绕行';
        if (effect.status === 'slowdown') {
            return '当前方式（' + modeName + '）可通行，规划成本约 ×' + effect.costMultiplier;
        }
        if (effect.status === 'notice') return '当前方式（' + modeName + '）仅提示，不改变路线';
        if (effect.status === 'unaffected') return '当前方式（' + modeName + '）不受此事件影响';
        return '当前方式的路线影响未配置，请联系管理员核对';
    }

    // 清除路况标记（线段 + 胶囊）
    function clearRoadConditionMarkers() {
        state.roadConditionMarkers.forEach(function (m) {
            if (state.map) state.map.removeLayer(m);
        });
        state.roadConditionMarkers = [];
    }

    // 拉取并渲染路况。includeAll=true 为管理员视图（含未开始/已结束，灰显）
    function loadAndRenderRoadConditions(includeAll) {
        if (!state.map) return;
        var url = '/api/road-conditions' + (includeAll ? '?all=1' : '');
        apiRequest(url, null, 'GET').then(function (data) {
            clearRoadConditionMarkers();
            state.roadConditionAdminView = !!includeAll;
            var conditions = (data && data.conditions) || [];
            var hasClosure = false, hasSlow = false, hasInfo = false, hasUnknown = false;
            conditions.forEach(function (cond) {
                var active = !includeAll || !cond.status || cond.status === 'active';
                if (active) {
                    var effect = getRoadConditionModeEffect(cond, state.travelMode);
                    if (effect.status === 'blocked') hasClosure = true;
                    else if (effect.status === 'slowdown') hasSlow = true;
                    else if (effect.status === 'notice' || effect.status === 'unaffected') hasInfo = true;
                    else if (effect.status === 'unknown') hasUnknown = true;
                }
                try {
                    renderRoadCondition(cond, !!includeAll);
                } catch (err) {
                    console.warn('[路况] 单条事件渲染失败，跳过:', err, cond && cond.id);
                }
            });
            updateRoadConditionLegend(hasClosure, hasSlow, hasInfo, hasUnknown);
        }).catch(function (e) {
            console.warn('[路况] 加载失败:', e);
        });
    }

    function clearRoadConditionUpdateNotice() {
        var notice = document.getElementById('road-condition-update-notice');
        if (notice) notice.hidden = true;
    }

    function checkRouteConditionVersion() {
        if (state.roadConditionMonitorInFlight || !state.routeStore) return Promise.resolve(false);
        var current = state.routeStore.current();
        var notice = document.getElementById('road-condition-update-notice');
        if (!current || !current.road_condition_version || !notice) {
            clearRoadConditionUpdateNotice();
            return Promise.resolve(false);
        }
        state.roadConditionMonitorInFlight = true;
        var routeId = current.route_id;
        return apiRequest('/api/road-conditions?version_only=1', null, 'GET').then(function (data) {
            var latest = state.routeStore && state.routeStore.current();
            if (!latest || latest.route_id !== routeId) return;
            if (!data || !data.active_event_version
                    || data.active_event_version === latest.road_condition_version) {
                clearRoadConditionUpdateNotice();
                return;
            }
            notice.hidden = false;
            notice.textContent = '校园路况有更新。重新规划会保留起点、途经点和路线偏好。';
            var button = notice.querySelector('button[data-action="replan-road-conditions"]');
            if (!button) {
                button = document.createElement('button');
                button.type = 'button';
                button.dataset.action = 'replan-road-conditions';
                button.textContent = '按最新路况重算';
                notice.appendChild(button);
            }
            button.disabled = false;
            button.onclick = function () {
                button.disabled = true;
                return replanCurrentRoute({ refresh_road_conditions: true }, '按最新路况重新规划').finally(function () {
                    button.disabled = false;
                });
            };
        }).catch(function (error) {
            console.warn('[路况] 检查路线版本失败:', error);
        }).finally(function () {
            state.roadConditionMonitorInFlight = false;
        });
    }

    function startRoadConditionVersionMonitor() {
        if (state.roadConditionMonitorTimer) return;
        checkRouteConditionVersion();
        state.roadConditionMonitorTimer = setInterval(checkRouteConditionVersion, 30000);
    }

    // 图例只显示当前出行方式下实际出现的路线影响状态。
    function updateRoadConditionLegend(hasClosure, hasSlow, hasInfo, hasUnknown) {
        var elClosure = document.getElementById('legend-closure-item');
        var elSlow = document.getElementById('legend-slow-item');
        var elInfo = document.getElementById('legend-info-item');
        var elUnknown = document.getElementById('legend-unknown-item');
        if (elClosure) elClosure.hidden = !hasClosure;
        if (elSlow) elSlow.hidden = !hasSlow;
        if (elInfo) elInfo.hidden = !hasInfo;
        if (elUnknown) elUnknown.hidden = !hasUnknown;
    }

    function renderRoadCondition(cond, adminView) {
        var style = getRoadConditionDisplayStyle(cond, state.travelMode);
        var inactive = adminView && cond.status && cond.status !== 'active';
        var color = inactive ? '#9AA5B1' : style.color;
        var edge = cond.edge || {};
        var anchor = edge.snap || cond.coordinates || null;
        if (!anchor) return;
        var latlng = gcjToLatLng(anchor.lng, anchor.lat);
        var layers = [];

        // 沿真实路段画高亮线：当前模式禁行画实线，其余影响画虚线区分
        var modeEffect = getRoadConditionModeEffect(cond, state.travelMode);
        var hardClosed = modeEffect.status === 'blocked';
        var geom = edge.geometry_gcj || [];
        if (geom.length >= 2) {
            var lineLatLngs = geom.map(function (p) { return gcjToLatLng(p[0], p[1]); });
            var lineOpt = { color: color, weight: 6, opacity: inactive ? 0.4 : 0.9, lineCap: 'round' };
            if (inactive) {
                lineOpt.dashArray = '3 8';
            } else if (!hardClosed && style.dash) {
                lineOpt.dashArray = style.dash;
            }
            var line = L.polyline(lineLatLngs, lineOpt);
            line.addTo(state.map);
            layers.push(line);
        } else if (cond.radius_m) {
            // 兼容旧版半径圆数据
            var circle = L.circle(latlng, {
                radius: cond.radius_m, color: color, weight: 1, opacity: 0.6,
                fillColor: color, fillOpacity: 0.12,
            });
            circle.addTo(state.map);
            layers.push(circle);
        }

        // 吸附点胶囊（加大点击区域）
        var label = cond.name || style.label;
        if (inactive && cond.status) label = label + '（' + COND_STATUS_LABELS[cond.status] + '）';
        var pillHtml = '<div class="whu-map-pill" style="background:' + color + ';color:white;'
            + (inactive ? 'opacity:0.65;' : '') + '">'
            + style.icon + ' ' + escapeHTML(label) + '</div>';
        var marker = divMarker(latlng, pillHtml, [0, 0], [0, 0], label);
        marker.addTo(state.map);
        layers.push(marker);

        var onClick = function () { showConditionPopup(cond, style, color, inactive, layers); };
        layers.forEach(function (ly) { ly.on('click', onClick); });
        state.roadConditionMarkers = state.roadConditionMarkers.concat(layers);
    }

    // 路况事件信息窗
    function showConditionPopup(cond, style, color, inactive, layers) {
        try {
            var edge = cond.edge || {};
            var anchor = edge.snap || cond.coordinates;
            var fmt = function (ts) {
                var d = new Date(ts * 1000);
                var p = function (n) { return (n < 10 ? '0' : '') + n; };
                return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) +
                    ' ' + p(d.getHours()) + ':' + p(d.getMinutes());
            };
            var rows = [];
            if (edge.road_name) rows.push('<div style="font-size:12px;">📍 路段：<b>' + escapeHTML(edge.road_name) + '</b></div>');
            if (edge.chain_length_m) {
                rows.push('<div style="color:#888;font-size:12px;">影响长度：约 '
                    + Math.round(edge.chain_length_m) + ' 米（两端路口之间整段）</div>');
            }
            if (!inactive) {
                var modeName = (TRAVEL_MODES[state.travelMode] || {}).label || state.travelMode;
                var effText = '🚦 ' + escapeHTML(describeRoadConditionModeEffect(cond, state.travelMode, modeName));
                rows.push('<div style="font-size:12px;color:' + color + ';">' + effText + '</div>');
            }
            var start = cond.start_time || 0, end = cond.end_time || 0;
            if (start || end) {
                rows.push('<div style="color:#888;font-size:12px;">⏱ '
                    + (start ? fmt(start) : '即时') + ' 至 ' + (end ? fmt(end) : '长期有效') + '</div>');
            }
            if (cond.status && cond.status !== 'active') {
                rows.push('<div style="font-size:12px;color:#9AA5B1;">状态：'
                    + (COND_STATUS_LABELS[cond.status] || cond.status) + '</div>');
            }
            if (cond.description) {
                rows.push('<div style="font-size:12px;color:#555;">' + escapeHTML(cond.description) + '</div>');
            }
            var info = '<div style="padding:4px 2px;font-size:13px;line-height:1.7;min-width:180px;max-width:240px;">' +
                '<div style="font-weight:600;color:' + color + ';margin-bottom:4px;font-size:14px;">' +
                style.icon + ' ' + escapeHTML(cond.name || style.label) + '</div>' +
                rows.join('') +
                '<div class="cond-actions" style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;"></div>' +
                '</div>';

            var popup = L.popup({ offset: [0, -6], closeButton: true, autoPan: true })
                .setContent(info)
                .setLatLng(gcjToLatLng(anchor.lng, anchor.lat));
            layers.forEach(function (ly) { ly.bindPopup(popup); });
            popup.openOn(state.map);

            // 已登录管理员才注入操作按钮
            apiRequest('/api/admin/status', null, 'GET').then(function (data) {
                if (!(data && data.is_admin)) return;
                setTimeout(function () {
                    var area = popup.getElement()
                        ? popup.getElement().querySelector('.cond-actions')
                        : document.querySelector('.leaflet-popup .cond-actions');
                    if (!area) return;
                    var btnStyle = 'border:none;padding:7px 14px;border-radius:8px;font-size:13px;cursor:pointer;font-weight:600;color:white;';
                    if (!cond.end_time && !inactive) {
                        var endBtn = document.createElement('button');
                        endBtn.type = 'button';
                        endBtn.textContent = '⏹ 结束';
                        endBtn.style.cssText = btnStyle + 'background:#7F8C8D;';
                        endBtn.addEventListener('click', function () { endRoadCondition(cond.id, popup); });
                        area.appendChild(endBtn);
                    }
                    var delBtn = document.createElement('button');
                    delBtn.type = 'button';
                    delBtn.textContent = '🗑 删除';
                    delBtn.style.cssText = btnStyle + 'background:#e74c3c;';
                    delBtn.addEventListener('click', function () { deleteRoadCondition(cond.id, popup); });
                    area.appendChild(delBtn);
                }, 80);
            }).catch(function () {});
        } catch (e) {
            console.error('[路况] 信息窗错误:', e);
        }
    }

    // 结束路况（保留记录，end_time=now）
    function endRoadCondition(id, popup) {
        apiRequest('/api/road-conditions/' + id, { action: 'end' }, 'PATCH').then(function () {
            if (popup) state.map.closePopup();
            loadAndRenderRoadConditions(state.roadConditionAdminView);
        }).catch(function (err) {
            alert('结束失败：' + (err && err.message ? err.message : '请重试'));
        });
    }

    // 删除路况事件
    function deleteRoadCondition(id, popup) {
        if (!window.confirm('确定删除这条路况？删除后不可恢复（若只需让它失效，建议用「结束」保留记录）。')) return;
        apiRequest('/api/road-conditions/' + id, null, 'DELETE').then(function () {
            if (popup) state.map.closePopup();
            loadAndRenderRoadConditions(state.roadConditionAdminView);
        }).catch(function (err) {
            alert('删除失败：' + (err && err.message ? err.message : '请重试'));
        });
    }

    // 天气图标映射
    var WEATHER_ICONS = {
        '晴': '☀️', '少云': '🌤', '晴间多云': '🌤', '多云': '⛅', '阴': '☁️',
        '有风': '🌬', '风': '🌬', '霾': '😷', '雾': '🌫',
        '小雨': '🌦', '中雨': '🌧', '大雨': '🌧', '暴雨': '⛈', '阵雨': '🌦',
        '雷阵雨': '⛈', '雨': '🌧', '雪': '🌨', '小雪': '🌨', '中雪': '🌨',
        '大雪': '❄️', '暴雪': '❄️', '雨夹雪': '🌨'
    };

    // 加载实时天气徽章（真实动态路况来源之一：雨雪避坡、高温走树荫）
    function loadWeatherBadge() {
        apiRequest('/api/weather', null, 'GET').then(function (w) {
            if (!w) return;
            var badge = document.getElementById('weather-badge');
            var iconEl = document.getElementById('weather-icon');
            var textEl = document.getElementById('weather-text');
            if (!badge) return;
            var icon = WEATHER_ICONS[w.weather] || '🌡';
            var temp = w.temperature != null ? Math.round(w.temperature) : null;
            var text = w.weather || '';
            if (temp != null) text += ' ' + temp + '°';
            if (w.label) text += ' · ' + w.label;
            if (iconEl) iconEl.textContent = icon;
            if (textEl) textEl.textContent = text;
            badge.title = w.advice || ('当前武汉天气：' + w.weather + (temp != null ? '，' + temp + '°C' : ''));
            badge.hidden = false;
            if (w.slippery) badge.classList.add('weather-slippery');
            else if (w.hot) badge.classList.add('weather-hot');
        }).catch(function () { /* 静默失败 */ });
    }

    // ====== 上报路况（需管理员登录）======
    // HTML 转义（主 IIFE 内的本地工具；候选面板 IIFE 内另有同名函数，作用域独立）
    function escapeHTML(str) {
        if (str == null) return '';
        return String(str).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    var _pickHandler = null;
    var _pickMarker = null;      // 吸附后的红点
    var _pickEdgeLine = null;    // 吸附路段的高亮预览线
    var _pickBanner = null;      // 选点模式地图顶部浮层提示
    var _isAdmin = false;

    function openRoadReport() {
        // 先检查管理员登录态
        apiRequest('/api/admin/status', null, 'GET').then(function (data) {
            if (data && data.is_admin) {
                _isAdmin = true;
                showRoadReportForm();
                loadAndRenderRoadConditions(true);  // 管理员视图：含未开始/已结束
            } else {
                openAdminLogin();
            }
        }).catch(function () {
            openAdminLogin();
        });
    }

    function showRoadReportForm() {
        var section = document.getElementById('road-report-section');
        if (section) section.hidden = false;
        var hint = document.getElementById('road-report-hint');
        if (hint) hint.textContent = '';
    }

    // 彻底清理选点状态：解绑点击、移除红点与路段预览、清浮层与已选坐标。
    // 取消 / 退出管理 / 提交完成 都必须走这里（修复旧版红点残留在点击位置的 bug）。
    function clearPickState() {
        stopPickLocation();
        if (_pickMarker && state.map) { state.map.removeLayer(_pickMarker); _pickMarker = null; }
        if (_pickEdgeLine && state.map) { state.map.removeLayer(_pickEdgeLine); _pickEdgeLine = null; }
        hidePickBanner();
        state._pickedLng = null;
        state._pickedLat = null;
    }

    function closeRoadReport() {
        var section = document.getElementById('road-report-section');
        if (section) section.hidden = true;
        clearPickState();
        var form = document.getElementById('road-report-form');
        if (form) form.reset();
        syncRoadReportActionFields();
        var loc = document.getElementById('rr-location');
        if (loc) loc.value = '';
    }

    function showPickBanner(text, isError) {
        hidePickBanner();
        if (!state.map) return;
        var el = document.createElement('div');
        el.textContent = text;
        el.style.cssText = 'position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;'
            + 'background:' + (isError ? 'rgba(192,57,43,0.95)' : 'rgba(33,33,33,0.9)')
            + ';color:white;padding:9px 18px;border-radius:20px;font-size:13px;max-width:86%;text-align:center;'
            + 'box-shadow:0 2px 10px rgba(0,0,0,0.25);pointer-events:auto;cursor:pointer;';
        // 点击浮层可取消选点
        el.addEventListener('click', function () {
            stopPickLocation();
            hidePickBanner();
            var section = document.getElementById('road-report-section');
            if (section) section.hidden = false;
        });
        state.map.getContainer().appendChild(el);
        _pickBanner = el;
    }

    function hidePickBanner() {
        if (_pickBanner && _pickBanner.parentNode) _pickBanner.parentNode.removeChild(_pickBanner);
        _pickBanner = null;
    }

    // ====== 管理员登录 ======
    function openAdminLogin() {
        var section = document.getElementById('admin-login-section');
        if (section) section.hidden = false;
        var hint = document.getElementById('admin-login-hint');
        if (hint) hint.textContent = '';
        var pwd = document.getElementById('admin-password');
        if (pwd) pwd.value = '';
        setTimeout(function () { if (pwd) pwd.focus(); }, 100);
    }

    function closeAdminLogin() {
        var section = document.getElementById('admin-login-section');
        if (section) section.hidden = true;
    }

    function handleAdminLogin(e) {
        e.preventDefault();
        var pwd = document.getElementById('admin-password');
        var password = pwd ? pwd.value : '';
        var hint = document.getElementById('admin-login-hint');
        var submitBtn = document.getElementById('admin-login-submit');
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '登录中…'; }

        apiRequest('/api/admin/login', { password: password }).then(function () {
            _isAdmin = true;
            closeAdminLogin();
            showRoadReportForm();
            loadAndRenderRoadConditions(true);  // 进入管理视图
        }).catch(function (err) {
            if (hint) hint.textContent = (err && err.message) ? err.message : '登录失败，请重试';
            if (pwd) pwd.select();
        }).finally(function () {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '登录'; }
        });
    }

    function adminLogout() {
        apiRequest('/api/admin/logout', {}).catch(function () {}).then(function () {
            _isAdmin = false;
            closeRoadReport();               // 内含 clearPickState，红点必被移除
            loadAndRenderRoadConditions(false);  // 回到普通用户视图
        });
    }

    function startPickLocation() {
        if (!state.map) {
            var hint0 = document.getElementById('road-report-hint');
            if (hint0) hint0.textContent = '地图尚未加载，请稍后再试';
            return;
        }
        // 重置上一次选择，进入地图选点模式
        if (_pickMarker) { state.map.removeLayer(_pickMarker); _pickMarker = null; }
        if (_pickEdgeLine) { state.map.removeLayer(_pickEdgeLine); _pickEdgeLine = null; }
        state._pickedLng = null;
        state._pickedLat = null;
        var loc0 = document.getElementById('rr-location');
        if (loc0) loc0.value = '';

        var section = document.getElementById('road-report-section');
        if (section) section.hidden = true;
        showPickBanner('📍 请直接点击事发路段，系统会自动吸附到道路上（点此取消）');

        if (_pickHandler) return;
        _pickHandler = function (e) {
            var gcjLng = e.latlng.lng, gcjLat = e.latlng.lat;
            showPickBanner('⏳ 正在吸附到道路…');
            var snapQuery = 'lng=' + encodeURIComponent(gcjLng)
                + '&lat=' + encodeURIComponent(gcjLat);
            var reportAction = document.getElementById('rr-action');
            var reportType = document.getElementById('rr-type');
            var reportModes = getSelectedRoadReportModes();
            if (reportAction && reportAction.value && reportModes.length && reportType) {
                snapQuery += '&type=' + encodeURIComponent(reportType.value)
                    + '&action=' + encodeURIComponent(reportAction.value)
                    + '&modes=' + encodeURIComponent(reportModes.join(','));
                if (reportAction.value === 'slowdown') {
                    var reportCost = document.getElementById('rr-cost-multiplier');
                    if (reportCost) snapQuery += '&cost_multiplier=' + encodeURIComponent(reportCost.value);
                }
            }
            apiRequest('/api/road-conditions/snap?' + snapQuery, null, 'GET')
                .then(function (data) {
                    var snap = data && data.snap;
                    if (!snap) {
                        showPickBanner('⚠️ 吸附失败，请再点一次道路', true);
                        return;
                    }
                    // 吸附成功：红点落到路上，并高亮该路段
                    if (_pickMarker) state.map.removeLayer(_pickMarker);
                    if (_pickEdgeLine) state.map.removeLayer(_pickEdgeLine);
                    var snapLatLng = gcjToLatLng(snap.snap_lng_gcj, snap.snap_lat_gcj);
                    var pickHtml = '<div style="width:16px;height:16px;background:#e74c3c;border-radius:50%;'
                        + 'border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,0.4);"></div>';
                    _pickMarker = divMarker(snapLatLng, pickHtml, [16, 16], [8, 8], '事发位置')
                        .addTo(state.map);
                    if ((snap.geometry_gcj || []).length >= 2) {
                        var edgeLatLngs = snap.geometry_gcj.map(function (p) {
                            return gcjToLatLng(p[0], p[1]);
                        });
                        _pickEdgeLine = L.polyline(edgeLatLngs, {
                            color: '#FFD400', weight: 8, opacity: 0.75, lineCap: 'round',
                        }).addTo(state.map);
                    }
                    // 提交时仍发原始点击点，服务端重新吸附（不信任前端）
                    state._pickedLng = gcjLng;
                    state._pickedLat = gcjLat;
                    stopPickLocation();
                    hidePickBanner();
                    if (section) section.hidden = false;
                    var loc = document.getElementById('rr-location');
                    if (loc) loc.value = (snap.road_name || '未命名道路')
                        + '（吸附偏移 ' + (snap.dist_m != null ? snap.dist_m : '?') + ' 米）';
                    var hint = document.getElementById('road-report-hint');
                    if (hint) {
                        hint.style.color = '#27AE60';
                        hint.textContent = '✅ 已吸附到「' + (snap.road_name || '未命名道路')
                            + '」，偏移 ' + (snap.dist_m != null ? snap.dist_m : '?') + ' 米'
                            + (snap.chain_length_m
                                ? '，关联路段约 ' + Math.round(snap.chain_length_m) + ' 米'
                                : '');
                    }
                })
                .catch(function (err) {
                    var msg = err && err.message ? err.message : '吸附失败';
                    showPickBanner('⚠️ ' + msg + '，请点在道路上重试', true);
                });
        };
        state.map.on('click', _pickHandler);
    }

    function stopPickLocation() {
        if (_pickHandler) {
            state.map.off('click', _pickHandler);
            _pickHandler = null;
        }
    }

    function handleRoadReportSubmit(e) {
        e.preventDefault();
        var type = document.getElementById('rr-type').value;
        var actionEl = document.getElementById('rr-action');
        var action = actionEl ? actionEl.value : '';
        var affectedModes = getSelectedRoadReportModes();
        var name = document.getElementById('rr-name').value.trim();
        var descEl = document.getElementById('rr-desc');
        var desc = descEl ? descEl.value.trim() : '';
        var lng = state._pickedLng;
        var lat = state._pickedLat;

        var hint = document.getElementById('road-report-hint');
        if (hint) hint.style.color = '';
        if (lng == null || lat == null) {
            if (hint) hint.textContent = '请先在地图上选择事发路段';
            return;
        }
        if (!name) {
            if (hint) hint.textContent = '请填写事件名称';
            return;
        }
        if (!action) {
            if (hint) hint.textContent = '请明确选择仅提示、缓行或封路';
            return;
        }
        if (!affectedModes.length) {
            if (hint) hint.textContent = '请至少选择一种受影响的出行方式';
            return;
        }

        var costMultiplier = null;
        if (action === 'slowdown') {
            var costEl = document.getElementById('rr-cost-multiplier');
            costMultiplier = costEl ? Number(costEl.value) : NaN;
            if (!Number.isFinite(costMultiplier) || costMultiplier < 1 || costMultiplier > 5) {
                if (hint) hint.textContent = '缓行成本倍数需在 1 到 5 之间';
                return;
            }
        }

        var startTime = document.getElementById('rr-start-time').value;
        var endTime = document.getElementById('rr-end-time').value;
        if (startTime && endTime && endTime <= startTime) {
            if (hint) hint.textContent = '结束时间必须晚于开始时间';
            return;
        }

        var submitBtn = document.getElementById('rr-submit-btn');
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = '提交中…'; }

        var payload = {
            type: type, action: action, affected_modes: affectedModes,
            name: name, lng: lng, lat: lat, description: desc,
        };
        if (action === 'slowdown') payload.cost_multiplier = costMultiplier;
        if (startTime) payload.start_time = startTime;
        if (endTime) payload.end_time = endTime;

        apiRequest('/api/road-conditions', payload).then(function () {
            var actionSummary = action === 'closure' ? '封路并绕行'
                : action === 'slowdown' ? '允许通行并增加规划成本' : '发布提示，不改变路线';
            if (hint) {
                hint.style.color = '#27AE60';
                hint.textContent = '✅ 已发布：' + actionSummary + '；仅作用于所选出行方式。';
            }
            clearPickState();
            loadAndRenderRoadConditions(true);  // 保持管理员视图刷新
            setTimeout(function () {
                closeRoadReport();
                if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '提交'; }
            }, 1200);
        }).catch(function (err) {
            if (hint) {
                hint.style.color = '#C0392B';
                hint.textContent = '提交失败：' + (err && err.message ? err.message : '请重试');
            }
            if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = '提交'; }
        });
    }

    function getSelectedRoadReportModes() {
        return ['walk', 'bike', 'drive'].filter(function (mode) {
            var input = document.getElementById('rr-mode-' + mode);
            return !!(input && input.checked);
        });
    }

    function syncRoadReportActionFields() {
        var action = document.getElementById('rr-action');
        var costWrap = document.getElementById('rr-cost-wrap');
        var costInput = document.getElementById('rr-cost-multiplier');
        var usesCost = !!(action && action.value === 'slowdown');
        if (costWrap) costWrap.hidden = !usesCost;
        if (costInput) costInput.required = usesCost;
    }

    // ===================== 实时导航（高德同款沉浸态） =====================

    function subscribeLocation(fn) {
        state.locSubscribers.push(fn);
        return function () {
            var i = state.locSubscribers.indexOf(fn);
            if (i >= 0) state.locSubscribers.splice(i, 1);
        };
    }

    var _navToastTimer = null;
    function showNavToast(msg) {
        var t = document.getElementById('nav-toast');
        if (!t) return;
        t.textContent = msg;
        t.hidden = false;
        if (_navToastTimer) clearTimeout(_navToastTimer);
        _navToastTimer = setTimeout(function () { t.hidden = true; }, 2600);
    }

    function renderNavCard(d) {
        var iconBox = document.getElementById('nav-card-icon');
        var arrow = iconBox ? iconBox.querySelector('.nav-card__arrow') : null;
        var emoji = document.getElementById('nav-card-emoji');
        if (iconBox && arrow && emoji) {
            if (d.icon === 'arrow') {
                arrow.hidden = false;
                emoji.hidden = true;
                arrow.style.transform = 'rotate(' + (d.angle || 0) + 'deg)';
            } else {
                arrow.hidden = true;
                emoji.hidden = false;
                emoji.textContent = d.icon === 'arrive' ? '🏁' : '⭐';
            }
        }
        var distEl = document.getElementById('nav-maneuver-dist');
        if (distEl) distEl.textContent = d.maneuverDist || '';
        var mainEl = document.getElementById('nav-instruction');
        if (mainEl) mainEl.textContent = d.main || '';
        var subEl = document.getElementById('nav-sub');
        if (subEl) subEl.textContent = d.sub || '';
        var rdEl = document.getElementById('nav-remain-dist');
        if (rdEl) rdEl.textContent = d.remainingText || '—';
        var tmEl = document.getElementById('nav-remain-time');
        if (tmEl) tmEl.textContent = d.remainingMin ? '约 ' + d.remainingMin + ' 分钟' : '';
        var roadEl = document.getElementById('nav-remain-road');
        if (roadEl) roadEl.textContent = d.roadName || '';
    }

    function syncMuteIcon() {
        var ic = document.getElementById('nav-mute-icon');
        if (ic && window.WhuWalkerVoiceOutput) {
            ic.textContent = WhuWalkerVoiceOutput.isMuted() ? '🔇' : '🔊';
        }
    }

    function startNavigation() {
        if (state.productMode === 'planning' || !isNavigationDevice()) {
            showTopBanner('电脑端用于路线规划，请复制手机打开链接后开始实时导航。', 'info');
            return;
        }
        var route = state.latestRoute;
        var Nav = window.WhuWalkerNavigation;
        if (!Nav || !state.map) return;
        if (!route || !(route.recommended || []).length) {
            showError('还没有路线', '先规划一条路线，再开始导航吧～');
            return;
        }
        // 导航态不显示最短路径灰线（等下引擎接管后重绘）
        if (state.shortestLine) {
            state.map.removeLayer(state.shortestLine);
            state.shortestLine = null;
        }

        var overlay = document.getElementById('nav-overlay');
        var arriveBox = document.getElementById('nav-arrive');
        if (arriveBox) arriveBox.hidden = true;
        syncMuteIcon();

        var routeState = state.routeStore ? state.routeStore.current() : null;
        var navRequest = routeState ? {
            start: routeState.start,
            end: routeState.end,
            constraints: routeState.hard_constraints || {},
            weights: routeState.strategy && routeState.strategy.weights,
        } : { end: route.end || null };
        var ok = Nav.start({
            map: state.map,
            routeData: route,
            request: navRequest,
            onCard: renderNavCard,
            onGps: function (stale) {
                var w = document.getElementById('nav-gps-warn');
                if (w) w.hidden = !stale;
            },
            onArrive: function (endName) {
                var box = document.getElementById('nav-arrive');
                var nm = document.getElementById('nav-arrive-name');
                if (nm) nm.textContent = endName || '';
                if (box) box.hidden = false;
                trackRouteSignal(
                    'navigation_completed',
                    state.routeStore ? state.routeStore.current() : null
                );
                trackEvent('nav_arrive', { mode: route.mode || 'walk' });
            },
            onToast: showNavToast,
            onFollowChange: function (following) {
                var btn = document.getElementById('nav-recenter');
                if (btn) btn.hidden = following;
            },
            onStateChange: function (active) {
                if (!active) {
                    document.body.classList.remove('nav-running');
                    if (overlay) overlay.hidden = true;
                    if (state._navUnsubscribe) { state._navUnsubscribe(); state._navUnsubscribe = null; }
                }
            },
            renderRoute: function (data) { renderRoute(data); },
        });
        if (!ok) {
            showError('导航启动失败', '路线数据不完整，重新规划一次试试～');
            return;
        }
        trackRouteSignal('navigation_started', routeState);

        document.body.classList.add('nav-running');
        if (overlay) overlay.hidden = false;
        state._navUnsubscribe = subscribeLocation(function (loc) { Nav.onFix(loc); });

        // 立即推一次已有位置（避免等下一个 GPS 回调）
        if (state.userLocation) Nav.onFix(state.userLocation);
        // 后台补定位（已有 watch 在跑则几乎立即回调）
        try { startTracking(true); } catch (e) { /* ignore */ }

        // 移动端点地图面板可见（桌面端无影响）
        var mapPanel = document.getElementById('map-panel');
        if (mapPanel && window.innerWidth <= 767 && mapPanel.scrollIntoView) {
            mapPanel.scrollIntoView({ block: 'start' });
        }
        trackEvent('nav_start', { mode: route.mode || state.travelMode });
    }

    function stopNavigation() {
        var Nav = window.WhuWalkerNavigation;
        if (Nav && Nav.isActive()) Nav.stop();  // onStateChange(false) 里收 UI
        var w = document.getElementById('nav-gps-warn');
        if (w) w.hidden = true;
        var t = document.getElementById('nav-toast');
        if (t) t.hidden = true;
        var btn = document.getElementById('nav-recenter');
        if (btn) btn.hidden = true;
    }

    // ===================== 语音输入 =====================

    function setVoiceBtnState(listening) {
        var btn = document.getElementById('voice-btn');
        if (btn) btn.classList.toggle('listening', !!listening);
    }

    function showVoiceUnsupported(reason) {
        if (reason === 'wechat') {
            showTopBanner('微信里语音用不了：点右上角「⋯」→「在浏览器打开」就能语音输入啦', 'info');
        } else if (reason === 'permission') {
            showTopBanner('麦克风权限被拒绝，可在浏览器设置里开启后重试', 'warning');
        } else {
            showTopBanner('当前浏览器不支持语音输入，换 Chrome / Edge，或安装珞珈智行 App', 'info');
        }
    }

    // 按住说话：pointerdown → press；pointerup → release；上滑超过阈值 → cancel（单向，越过即取消）
    var voiceHold = {
        active: false, pointerId: null, startY: 0,
        cancelled: false, committed: false, baseText: '',
    };
    var VOICE_CANCEL_DY = 70;

    function resetVoiceBtn() {
        var btn = document.getElementById('voice-btn');
        if (btn) btn.classList.remove('listening', 'will-cancel');
        if (btn) btn.blur();
    }

    function startVoiceHold(clientY, pointerId) {
        var VI = window.WhuWalkerVoiceInput;
        var input = document.getElementById('nl-input');
        var btn = document.getElementById('voice-btn');
        if (!VI || !input || voiceHold.active) return;

        var cap = VI.capability();
        if (!cap.ok) { showVoiceUnsupported(cap.reason); return; }

        voiceHold.active = true;
        voiceHold.pointerId = pointerId;
        voiceHold.startY = clientY;
        voiceHold.cancelled = false;
        voiceHold.committed = false;
        voiceHold.baseText = input.value.trim();

        if (btn) btn.classList.add('listening');
        VI.press({
            onStateChange: function (s) {
                setVoiceBtnState(s === 'listening');
                if (s === 'idle') {
                    voiceHold.active = false;
                    resetVoiceBtn();
                    // 取消、没说话、无匹配：还原按下前的输入框内容
                    if (!voiceHold.committed) {
                        input.value = voiceHold.baseText;
                        input.dispatchEvent(new Event('input'));
                    }
                }
            },
            onInterim: function (text) {
                input.value = voiceHold.baseText ? (voiceHold.baseText + ' ' + text) : text;
                input.dispatchEvent(new Event('input'));
            },
            onFinal: function (text) {
                voiceHold.committed = true;
                input.value = voiceHold.baseText ? (voiceHold.baseText + ' ' + text) : text;
                input.dispatchEvent(new Event('input'));
                setVoiceBtnState(false);
                var submitBtn = document.getElementById('submit-btn');
                if (submitBtn && text.trim()) submitBtn.click();
            },
            onUnsupported: function (reason) {
                setVoiceBtnState(false);
                showVoiceUnsupported(reason);
            },
            onError: function (msg) {
                setVoiceBtnState(false);
                showTopBanner('🎤 ' + msg, 'info');
            },
        });
    }

    function moveVoiceHold(clientY, pointerId) {
        if (!voiceHold.active || pointerId !== voiceHold.pointerId || voiceHold.cancelled) return;
        if (voiceHold.startY - clientY > VOICE_CANCEL_DY) {
            voiceHold.cancelled = true;
            var btn = document.getElementById('voice-btn');
            var VI = window.WhuWalkerVoiceInput;
            if (btn) { btn.classList.add('will-cancel'); btn.classList.remove('listening'); }
            if (VI) VI.setCancelHint(true);
        }
    }

    function endVoiceHold(pointerId, abort) {
        if (!voiceHold.active || (pointerId != null && pointerId !== voiceHold.pointerId)) return;
        var VI = window.WhuWalkerVoiceInput;
        voiceHold.active = false;
        resetVoiceBtn();
        if (!VI) return;
        if (abort || voiceHold.cancelled) VI.cancel();
        else VI.release();
    }

    function clearRouteResult() {
        stopNavigation();
        clearMap();
        var section = document.getElementById('results-section');
        if (section) section.hidden = true;
        document.body.classList.remove('has-route-results');
        syncRouteStrategyUI();
    }

    // 定位单个 POI：清空现有覆盖物，移动地图中心到该 POI 并高亮标记
    function focusPoiOnMap(poi) {
        if (!state.map) return;
        var lng = poi.lon != null ? poi.lon : poi.lng;
        var lat = poi.lat;
        if (lng == null || lat == null) return;

        var pillHtml = '<div class="whu-map-pill whu-map-pill-poi">' + (poi.name || 'POI') + '</div>';
        var marker = divMarker(gcjToLatLng(lng, lat), pillHtml, [0, 0], [0, 0], poi.name || '')
            .addTo(state.map);
        state.poiMarkers.push(marker);
        state.map.setView(gcjToLatLng(lng, lat), 17);
    }

    // 返回键：清空路线 + 清空对话 + 复位地图，回到初始欢迎状态
    function handleReset() {
        cancelActiveRuns();
        state.requestSeq += 1;
        state.conversationEpoch += 1;
        state.latestRouteTurnId = state.nextConversationTurnId + 1;
        state.latestRouteSequence = 0;
        stopNavigation();     // 退出实时导航
        // 1. 清空路线、路线途经标记和手动选点
        stopPointPick();
        clearMapPoints();
        clearMap();
        // 2. 复位地图视角（GCJ 中心点转 WGS-84）
        if (state.map) {
            var resetCenter = gcjToLatLng(MAP_CENTER[0], MAP_CENTER[1]);
            var resetZoom = window.innerWidth <= 767 ? 14 : MAP_ZOOM;
            state.map.setView(resetCenter, resetZoom);
            state._userInteractedWithMap = false;
        }
        // 3. 清空对话气泡（保留欢迎元素）
        var chatContent = document.getElementById('chat-content');
        if (chatContent) {
            chatContent.querySelectorAll('.chat-bubble-row').forEach(function (b) {
                b.remove();
            });
        }
        // 4. 恢复欢迎元素
        var welcomeBubble = document.getElementById('welcome-bubble');
        var shortcutCards = document.getElementById('shortcut-cards-row');
        if (welcomeBubble) welcomeBubble.style.display = '';
        if (shortcutCards) shortcutCards.style.display = '';
        // 5. 隐藏结果区
        var results = document.getElementById('results-section');
        if (results) results.hidden = true;
        document.body.classList.remove('has-route-results');
        var suggestions = document.getElementById('suggestions-area');
        if (suggestions) suggestions.hidden = true;
        // 6. 清空多轮对话上下文
        state.conversationHistory = [];
        state.pendingIntent = null;
        state.pendingServerTaskId = null;
        state.pendingServerTaskRevision = null;
        state.pendingServerQuestion = null;
        state.pendingServerTaskTurnId = 0;
        state.latestRoute = null;
        state.serverConversationId = null;
        try { localStorage.removeItem(getServerConversationKey()); } catch (e) {}
        try { localStorage.removeItem(SERVER_CONVERSATION_KEY); } catch (e) {}
        if (state.routeStore) state.routeStore.replace(null);
        saveContext();
        // 7. 清空输入框
        var nlInput = document.getElementById('nl-input');
        if (nlInput) { nlInput.value = ''; nlInput.style.height = 'auto'; }
        var charCount = document.getElementById('char-count');
        if (charCount) charCount.textContent = '0';
        // 8. 隐藏错误与加载遮罩
        hideError();
        hideLoading();
        // 9. 重置快捷 chip 高亮
        state.activeMode = null;
        syncRouteStrategyUI();
        document.querySelectorAll('.quick-chip').forEach(function (c) { c.classList.remove('active'); });
    }

    // 预计用时文案：后端 duration_min 优先（≥1 分钟"约 X 分钟"，<1 显示"约 1 分钟"）；
    // 后端未给时用前端兜底速度 walk 4.5 / bike 14 / drive 25 km/h 按推荐距离估算
    function estimateDurationText(data) {
        var min = data ? data.duration_min : null;
        if (min == null || isNaN(min) || min <= 0) {
            var recM = (data && (data.recommended_length_m || data.distance_m)) || 0;
            var mode = (data && data.mode && TRAVEL_MODES[data.mode]) ? data.mode : state.travelMode;
            var speed = (TRAVEL_MODES[mode] || TRAVEL_MODES.walk).speedKmh;
            min = recM > 0 ? (recM / 1000) / speed * 60 : null;
        }
        if (min == null || isNaN(min)) return '—';
        return '约 ' + Math.max(1, Math.round(min)) + ' 分钟';
    }

    var STRATEGY_LABELS = {
        shortest: '最短路径',
        recommended: '综合推荐',
        scenery: '风景优先',
        flat: '平坦优先',
        custom: '个性化路线',
    };

    function endpointLabel(endpoint, fallback) {
        if (!endpoint || typeof endpoint !== 'object') return fallback;
        return endpoint.name || endpoint.label || fallback;
    }

    function strategyWeightsText(strategy) {
        var weights = strategy && strategy.weights;
        if (!weights) return '使用当前路线策略';
        var d = Number(weights.distance || 0);
        var s = Number(weights.slope || 0);
        var c = Number(weights.scenery || 0);
        return '距离 ' + Math.round(d * 100) + '% · 平坦 ' + Math.round(s * 100) + '% · 风景 ' + Math.round(c * 100) + '%';
    }

    function formatStepDistance(value) {
        var metres = Number(value || 0);
        if (!isFinite(metres) || metres <= 0) return '';
        return metres >= 1000 ? (metres / 1000).toFixed(1) + ' km' : Math.round(metres) + ' m';
    }

    function renderDesktopRouteWorkbench(data) {
        if (state.productMode !== 'planning') return;
        var details = document.getElementById('desktop-route-details');
        if (details) details.open = false;
        var stepsList = document.getElementById('desktop-route-steps');
        if (!stepsList) return;
        stepsList.innerHTML = '';
        var steps = Array.isArray(data.steps) ? data.steps : [];
        var stepCount = document.getElementById('desktop-route-step-count');
        if (stepCount) stepCount.textContent = steps.length ? steps.length + ' 段 · 展开查看' : '暂无分段';
        if (!steps.length) {
            var empty = document.createElement('li');
            empty.className = 'desktop-route-step desktop-route-step--empty';
            empty.textContent = '这条路线暂时没有分段指引，可查看地图上的路线。';
            stepsList.appendChild(empty);
            return;
        }
        steps.forEach(function (step, index) {
            var li = document.createElement('li');
            li.className = 'desktop-route-step';
            var button = document.createElement('button');
            button.type = 'button';
            button.className = 'desktop-route-step__button';
            var marker = document.createElement('span');
            marker.className = 'desktop-route-step__index';
            marker.textContent = step.type === 'arrive' ? '✓' : String(index + 1);
            var body = document.createElement('span');
            body.className = 'desktop-route-step__body';
            var text = document.createElement('strong');
            text.textContent = step.text || step.instruction || '沿路线继续前行';
            var meta = document.createElement('small');
            var metaParts = [];
            var dist = formatStepDistance(step.distance_m);
            if (dist) metaParts.push('本段 ' + dist);
            if (step.road_name) metaParts.push(step.road_name);
            meta.textContent = metaParts.join(' · ') || '点击查看位置';
            body.appendChild(text);
            body.appendChild(meta);
            button.appendChild(marker);
            button.appendChild(body);
            if (step.point && isFinite(step.point.lng) && isFinite(step.point.lat)) {
                button.addEventListener('click', function () {
                    if (!state.map) return;
                    document.querySelectorAll('.desktop-route-step').forEach(function (el) {
                        el.classList.remove('is-active');
                    });
                    li.classList.add('is-active');
                    state.map.setView(gcjToLatLng(Number(step.point.lng), Number(step.point.lat)), Math.max(state.map.getZoom(), 17));
                });
            } else {
                button.disabled = true;
            }
            li.appendChild(button);
            stepsList.appendChild(li);
        });
    }

    function routePlanText() {
        var data = state.latestRoute || {};
        var current = state.routeStore ? state.routeStore.current() : null;
        var strategy = (current && current.strategy) || data.strategy || {};
        var lines = [];
        if (current) {
            lines.push(endpointLabel(current.start, '起点') + ' → ' + endpointLabel(current.end, '终点'));
        } else {
            lines.push('珞珈智行路线方案');
        }
        lines.push('方式：' + (TRAVEL_MODES[data.mode || state.travelMode] || TRAVEL_MODES.walk).label);
        lines.push('策略：' + (STRATEGY_LABELS[strategy.name] || '综合推荐') + '（' + strategyWeightsText(strategy) + '）');
        lines.push('距离：' + Number(data.recommended_length_m || data.distance_m || 0).toFixed(0) + ' m；用时：' + estimateDurationText(data));
        if (data.hard_budget_fallback) lines.push('说明：原偏好路线超过上限，已采用符合上限的最短路线。');
        if (data.explanation) lines.push('说明：' + data.explanation);
        var steps = Array.isArray(data.steps) ? data.steps : [];
        if (steps.length) {
            lines.push('分段路线：');
            steps.forEach(function (step, index) {
                var dist = formatStepDistance(step.distance_m);
                lines.push((index + 1) + '. ' + (step.text || step.instruction || '沿路线继续前行') + (dist ? '（' + dist + '）' : ''));
            });
        }
        return lines.join('\n');
    }

    function encodeRouteState(routeState) {
        if (!routeState) return '';
        try {
            return btoa(unescape(encodeURIComponent(JSON.stringify(routeState))))
                .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
        } catch (e) {
            return '';
        }
    }

    function decodeRouteState(encoded) {
        if (!encoded) return null;
        try {
            var base64 = encoded.replace(/-/g, '+').replace(/_/g, '/');
            while (base64.length % 4) base64 += '=';
            return JSON.parse(decodeURIComponent(escape(atob(base64))));
        } catch (e) {
            return null;
        }
    }

    function mobileHandoffUrl() {
        var current = state.routeStore ? state.routeStore.current() : null;
        var query = current && current.original_query ? current.original_query.trim() : '';
        if (!query && current) {
            query = '从' + endpointLabel(current.start, '起点') + '到' + endpointLabel(current.end, '终点');
        }
        if (!query) query = '查看刚才规划的校园路线';
        var strategyName = current && current.strategy ? current.strategy.name : null;
        var strategyPrompt = { shortest: '最短路径', scenery: '风景优先', flat: '平坦优先' }[strategyName];
        if (strategyPrompt && query.indexOf(strategyPrompt) < 0) query += '，' + strategyPrompt;
        var mode = (current && current.travel_mode) || state.travelMode;
        if (mode === 'bike' && !/骑|自行车/.test(query)) query += '，骑行';
        if (mode === 'drive' && !/驾车|开车/.test(query)) query += '，驾车';
        var url = new URL(window.location.href);
        url.search = '';
        url.hash = '';
        url.searchParams.set('plan', query);
        url.searchParams.set('mode', mode);
        var encodedState = encodeRouteState(current);
        if (encodedState) url.searchParams.set('route_state', encodedState);
        return url.toString();
    }

    function copyText(text, button, successLabel) {
        function done() {
            if (!button) return;
            var old = button.textContent;
            button.textContent = successLabel;
            button.classList.add('is-copied');
            setTimeout(function () { button.textContent = old; button.classList.remove('is-copied'); }, 1800);
        }
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(done).catch(function () { fallback(); });
            return;
        }
        fallback();
        function fallback() {
            var area = document.createElement('textarea');
            area.value = text;
            area.style.position = 'fixed';
            area.style.opacity = '0';
            document.body.appendChild(area);
            area.select();
            try { document.execCommand('copy'); done(); }
            catch (e) { showTopBanner('复制失败，请手动复制路线说明。', 'warning'); }
            area.remove();
        }
    }

    function consumeMobileHandoff() {
        var params = new URLSearchParams(window.location.search);
        var query = (params.get('plan') || '').trim();
        var sharedState = decodeRouteState(params.get('route_state'));
        if (!query && !sharedState) return;
        var mode = params.get('mode');
        if (TRAVEL_MODES[mode]) {
            state.travelMode = mode;
            persistTravelMode();
            syncTravelModeUI();
        }
        try { localStorage.setItem('whu_welcome_seen', '1'); } catch (e) { /* ignore */ }
        params.delete('plan');
        params.delete('mode');
        params.delete('route_state');
        var clean = window.location.pathname + (params.toString() ? '?' + params.toString() : '') + window.location.hash;
        try { history.replaceState(null, '', clean); } catch (e) { /* ignore */ }
        var attempts = 0;
        function submitWhenReady() {
            attempts += 1;
            if (state.map || attempts >= 12) {
                if (sharedState && sharedState.strategy && sharedState.strategy.name) {
                    hideWelcomeElements();
                    showUserBubble(query || '打开电脑端规划的路线');
                    var bubble = showChatBubble('', '🌸 正在恢复电脑端规划的路线…');
                    showLoading('正在恢复路线…', '保留起终点、途经点和路线偏好');
                    apiRequest('/api/route/replan', {
                        route_state: sharedState,
                        change: { strategy: sharedState.strategy.name },
                    }).then(function (result) {
                        hideLoading();
                        if (!result || !result.recommended || !result.recommended.length) throw new Error('路线恢复失败');
                        renderRoute(result);
                        showResults(result);
                        trackRouteShown();
                        updateChatBubble(bubble, buildRouteSummary(result));
                    }).catch(function () {
                        hideLoading();
                        updateChatBubble(bubble, '电脑端路线状态已过期，正在按原需求重新规划。');
                        if (query) handleNlSubmit(query);
                    });
                } else if (query) {
                    handleNlSubmit(query);
                }
                return;
            }
            setTimeout(submitWhenReady, 250);
        }
        setTimeout(submitWhenReady, 400);
    }

    function showResults(data) {
        var resultChatContent = document.getElementById('chat-content');
        var followLatestMessage = chatIsNearLatest(resultChatContent);
        adoptRouteState(data);
        // 隐藏欢迎气泡
        var welcomeBubble = document.getElementById('welcome-bubble');
        var shortcutCards = document.getElementById('shortcut-cards-row');
        if (welcomeBubble) welcomeBubble.style.display = 'none';
        if (shortcutCards) shortcutCards.style.display = 'none';

        var section = document.getElementById('results-section');
        section.hidden = false;
        document.body.classList.toggle('has-route-results', state.productMode === 'planning');
        syncRouteStrategyUI();

        document.getElementById('recommended-distance').textContent =
            (data.recommended_length_m || data.distance_m || 0).toFixed(0) + ' m';

        // 路线类型差异化展示：via → 绕行比，tour → 景点数，direct → 最短/重叠率
        var routeKind = data.route_kind || 'direct';
        var shortestCard = document.querySelector('#shortest-distance').closest('.summary-card');
        var shortestLabel = shortestCard ? shortestCard.querySelector('.summary-label') : null;
        var overlapCard = document.querySelector('#overlap-rate').closest('.summary-card');
        var overlapLabel = overlapCard ? overlapCard.querySelector('.summary-label') : null;

        if (routeKind === 'tour') {
            // 游览环线：最短距离 → 景点数，重叠率 → 环线标志
            var tourInfo = data.tour || {};
            var poiCount = (tourInfo.ordered_pois || data.pois || []).length;
            document.getElementById('shortest-distance').textContent = poiCount + ' 个';
            if (shortestLabel) shortestLabel.textContent = '游览景点';
            document.getElementById('overlap-rate').textContent = tourInfo.loop ? '环线' : '单程';
            if (overlapLabel) overlapLabel.textContent = '游览方式';
        } else if (routeKind === 'itinerary') {
            var itineraryInfo = data.itinerary || {};
            var stopCount = itineraryInfo.stop_count || (itineraryInfo.stop_names || []).length;
            document.getElementById('shortest-distance').textContent = stopCount + ' 个';
            if (shortestLabel) shortestLabel.textContent = '行程地点';
            document.getElementById('overlap-rate').textContent = itineraryInfo.total_duration_min != null
                ? itineraryInfo.total_duration_min.toFixed(0) + '/' + (itineraryInfo.time_budget_min || 0).toFixed(0) + ' 分钟'
                : '预算已核验';
            if (overlapLabel) overlapLabel.textContent = '用时 / 预算';
        } else if (routeKind === 'multimodal') {
            // 多模态换乘：最短距离 → 换乘次数，重叠率 → 换乘点列表
            var legs = data.legs || [];
            var transferCount = Math.max(0, legs.length - 1);
            document.getElementById('shortest-distance').textContent = transferCount + ' 次';
            if (shortestLabel) shortestLabel.textContent = '换乘次数';
            // 换乘点 = 每个 leg 的 end_name（除最后一段）
            var transfers = legs.slice(0, -1).map(function (lg) { return lg.end_name || ''; })
                                .filter(function (n) { return n; });
            document.getElementById('overlap-rate').textContent =
                transfers.length ? transfers.join(' → ') : '—';
            if (overlapLabel) overlapLabel.textContent = '换乘点';
        } else if (routeKind === 'via') {
            // 途经路线：最短距离 → 途经点，重叠率 → 绕行比
            var viaInfo = data.via || {};
            document.getElementById('shortest-distance').textContent = viaInfo.name || '—';
            if (shortestLabel) shortestLabel.textContent = '途经点';
            var detour = data.detour_ratio;
            document.getElementById('overlap-rate').textContent =
                detour != null ? '+' + (detour * 100).toFixed(0) + '%' : '—';
            if (overlapLabel) overlapLabel.textContent = '绕行比';
        } else {
            // 直接路线：保持原有逻辑
            document.getElementById('shortest-distance').textContent =
                data.shortest_baseline_feasible === false ? '超出时限' :
                    (data.shortest_length_m || data.shortest_distance_m || 0).toFixed(0) + ' m';
            if (shortestLabel) shortestLabel.textContent =
                data.shortest_baseline_feasible === false ? '最短路参考' : '最短距离';
            document.getElementById('overlap-rate').textContent =
                ((data.overlap_rate || 0) * 100).toFixed(0) + '%';
            if (overlapLabel) overlapLabel.textContent = '重叠率';
        }

        // 预计用时：优先用后端 duration_min，缺失时按模式兜底速度估算（fail-soft）
        var durationEl = document.getElementById('estimated-duration');
        if (durationEl) durationEl.textContent = estimateDurationText(data);

        var poiList = document.getElementById('poi-items');
        poiList.innerHTML = '';
        var pois = data.pois || [];
        // 游览环线：优先用 tour.ordered_pois 显示有序编号
        if (routeKind === 'tour' && data.tour && data.tour.ordered_pois) {
            pois = data.tour.ordered_pois;
        }
        var poiDisclosure = document.getElementById('poi-disclosure');
        var poiCount = document.getElementById('poi-count');
        if (poiDisclosure) {
            poiDisclosure.hidden = !pois.length;
            poiDisclosure.open = state.productMode !== 'planning' && pois.length > 0;
        }
        if (poiCount) poiCount.textContent = pois.length ? pois.length + ' 处 · 展开查看' : '暂无地点';
        if (pois.length > 0) {
            pois.forEach(function (poi, i) {
                var li = document.createElement('li');
                if (routeKind === 'tour') {
                    li.textContent = (i + 1) + '. ' + (poi.name || poi.id || '未知');
                } else {
                    li.textContent = poi.name || poi.id || '未知';
                }
                poiList.appendChild(li);
            });
        }

        var explanation = data.explanation || '已为您规划好路线';
        document.getElementById('explanation-text').textContent = explanation;

        // 路况提示：若有生效的路况事件，在解释下方显示提醒
        var roadNoticeEl = document.getElementById('road-conditions-notice');
        if (roadNoticeEl) {
            var rcCount = data.road_conditions_applied || 0;
            if (rcCount > 0) {
                roadNoticeEl.style.display = 'block';
                roadNoticeEl.textContent = '⚠️ 已核对 ' + rcCount
                    + ' 条当前路况信息；路线是否调整取决于封路或缓行是否影响本路线。';
            } else {
                roadNoticeEl.style.display = 'none';
            }
        }

        renderDesktopRouteWorkbench(data);

        // 跟进建议
        showSuggestions(data);

        // 自动滚到结果区：让摘要卡顶部对齐可视区（结果卡片在用户气泡之前，
        // 短面板上若直接滚到底会把结果滚出可视区，手机端尤其明显）
        var chatContent = resultChatContent;
        if (chatContent && followLatestMessage) {
            setTimeout(function () {
                var contentRect = chatContent.getBoundingClientRect();
                var sectionRect = section.getBoundingClientRect();
                var target = chatContent.scrollTop + (sectionRect.top - contentRect.top) - 8;
                chatContent.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
            }, 100);
        }
    }

    function showSuggestions(data, anchor) {
        var area = document.getElementById('suggestions-area');
        var chips = document.getElementById('suggestions-chips');
        if (!area || !chips) return;

        var placement = anchor || document.getElementById('explanation-box');
        if (placement && typeof placement.after === 'function') placement.after(area);

        var suggestions = data.needs_input && Array.isArray(data.candidates) && data.candidates.length
            ? []
            : (data.suggestions && data.suggestions.length > 0)
                ? data.suggestions
                : buildSuggestions(data);
        if (suggestions.length === 0) { area.hidden = true; return; }

        chips.innerHTML = '';
        suggestions.forEach(function (s) {
            var chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = s.label;
            chip.title = s.query;
            chip.addEventListener('click', function () {
                document.getElementById('nl-input').value = s.query;
                document.getElementById('submit-btn').click();
            });
            chips.appendChild(chip);
        });
        area.hidden = false;
    }

    function buildSuggestions(data) {
        var suggestions = [];
        var constraints = data.constraints || {};
        var recLen = data.recommended_length_m || data.distance_m || 0;
        var shortLen = data.shortest_length_m || data.shortest_distance_m || 0;
        var mode = (data && data.mode && TRAVEL_MODES[data.mode]) ? data.mode : state.travelMode;

        // 偏好类建议
        if (mode === 'drive') {
            if (constraints.distance !== 'short') {
                suggestions.push({ label: '⚡ 最快到达', query: '帮我规划最快到达的路线' });
            }
        } else {
            if (constraints.slope !== 'avoid') {
                suggestions.push({ label: '🪜 走更平坦的路', query: '帮我找一条更平坦的路线' });
            }
        }
        if (constraints.scenery !== 'high') {
            suggestions.push({ label: '🌸 想看风景好的路', query: '走风景更好的路线' });
        }
        if (mode !== 'drive' && data.shortest_baseline_feasible !== false &&
                recLen > shortLen * 1.3 && constraints.distance !== 'short') {
            suggestions.push({ label: '⚡ 我要最短路径', query: '帮我规划最短路径' });
        }

        // 校园生活场景建议（基于起终点推断）
        var start = data.start && data.start.name ? data.start.name : '';
        var end = data.end && data.end.name ? data.end.name : '';
        var routeText = (start + end) || '';

        // 不重复已有的偏好建议
        var existingLabels = suggestions.map(function (s) { return s.label; });
        function addIfNew(label, query) {
            if (existingLabels.indexOf(label) === -1) {
                suggestions.push({ label: label, query: query });
                existingLabels.push(label);
            }
        }

        // 吃饭场景
        if (routeText.indexOf('食堂') === -1 && routeText.indexOf('餐') === -1) {
            addIfNew('🍜 去附近食堂', '从' + (end || '这里') + '去最近的食堂');
        }
        // 学习场景
        if (routeText.indexOf('图书馆') === -1 && routeText.indexOf('教') === -1) {
            addIfNew('📚 去图书馆', '从' + (end || '这里') + '去总图书馆');
        }
        // 赏樱/景点场景
        if (routeText.indexOf('樱') === -1) {
            addIfNew('🌸 去樱花大道', '从' + (end || '这里') + '去樱花大道');
        }
        // 校门场景
        if (routeText.indexOf('门') === -1) {
            addIfNew('🚪 去最近校门', '从' + (end || '这里') + '去最近的校门');
        }

        // 兜底
        if (suggestions.length === 0) {
            if (mode === 'drive') {
                suggestions.push({ label: '⚡ 换条更快的路线', query: '换一条更快到达的路线' });
            } else {
                suggestions.push({ label: '🪜 换条更平坦的', query: '换一条更平坦的路线' });
            }
            suggestions.push({ label: '🌸 换条风景更好的', query: '换一条风景更好的路线' });
        }
        return suggestions.slice(0, 4);
    }

    function collapseResults() {
        var section = document.getElementById('results-section');
        section.classList.toggle('collapsed');
    }

    function showLoading(text, subtext) {
        state.loading = true;
        var section = document.getElementById('loading-section');
        section.hidden = false;
        document.getElementById('loading-text').textContent = text || '正在处理…';
        document.getElementById('loading-subtext').textContent = subtext || '请稍候';
    }

    function hideLoading() {
        state.loading = false;
        stopLoadingMessages();
        document.getElementById('loading-section').hidden = true;
    }

    function showError(title, message) {
        var section = document.getElementById('error-section');
        document.getElementById('error-title').textContent = title || '唔，出错了';
        document.getElementById('error-message').textContent = message || '抱歉，出了点意外状况';
        section.hidden = false;
    }

    function hideError() {
        document.getElementById('error-section').hidden = true;
    }

    async function apiRequest(endpoint, data, method) {
        var url = API_BASE + endpoint;
        var httpMethod = method || 'POST';
        var options = {
            method: httpMethod,
            headers: { 'Content-Type': 'application/json' },
        };
        if (httpMethod !== 'GET' && adminCsrfToken) {
            options.headers['X-CSRF-Token'] = adminCsrfToken;
        }
        // GET 请求不带 body（否则部分服务器/缓存层会拒绝）
        if (httpMethod !== 'GET') {
            options.body = JSON.stringify(data);
        }

        var response;
        var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
        if (controller) options.signal = controller.signal;
        var timeoutHandle;
        try {
            var timeoutMs = arguments.length > 3 && Number(arguments[3]) > 0 ? Number(arguments[3]) : 45000;
            response = await Promise.race([
                fetch(url, options),
                new Promise(function (_, reject) {
                    timeoutHandle = setTimeout(function () {
                        if (controller) controller.abort();
                        var timeoutError = new Error('请求超时，请稍后重试');
                        timeoutError.code = 'request_timeout';
                        reject(timeoutError);
                    }, timeoutMs);
                }),
            ]);
        } catch (e) {
            if (e && e.code === 'request_timeout') throw e;
            throw new Error('网络请求失败，请检查网络连接');
        } finally {
            if (timeoutHandle) clearTimeout(timeoutHandle);
        }

        var result;
        try {
            result = await response.json();
        } catch (e) {
            throw new Error('服务器响应格式错误');
        }

        if (!response.ok) {
            var errMsg = (result && result.message) || ('请求失败 (' + response.status + ')');
            var errCode = result && result.error;
            var userMsg = mapError(errCode, errMsg);
            var err = new Error(userMsg);
            err.code = errCode;  // 带上错误码，供调用方判断是否对话式引导
            throw err;
        }

        var payload = result.data || result;
        if (endpoint === '/api/admin/status' || endpoint === '/api/admin/login') {
            adminCsrfToken = payload && payload.csrf_token ? payload.csrf_token : '';
        } else if (endpoint === '/api/admin/logout') {
            adminCsrfToken = '';
        }
        return payload;
    }

    function mapError(code, msg) {
        var errorMap = {
            'missing_query': '嗯？你还没告诉我你想去哪呢～试试输入「从珞珈门到樱顶」',
            'missing_endpoints': '需要起点和终点才能规划路线哦，在地图上选点或者打字告诉我吧',
            'missing_poi_names': '起点和终点得有个名字才行～',
            'poi_not_found': '抱歉，我没找到这个地方😅 试试换个说法？比如「教五」就是「第五教学楼」',
            'route_not_found': '这条路线走不通…可能是路网数据还不够全，试试换个目的地？',
            'network_not_initialized': '地图还没加载完，稍等一下下就好～',
            'network_load_failed': '地图数据加载失败了，刷新一下页面试试？',
            'parse_failed': '我没太理解你的意思…试试简单一点的说法，比如「从珞珈门到樱顶」',
            'parse_validation_error': '输入格式有点问题，试试更简洁的描述？',
            'route_computation_failed': '路线计算出错了，可能是网络不太好，再试一次？',
            'nearest_node_failed': '这个位置我没法定位，换个附近的地点试试？',
            'unsupported_task': '这个功能我暂时还不会，试试问路或者查景点吧～',
            'internal_error': '出了点小问题，稍等一下再试就好',
        };
        return errorMap[code] || msg || '出了点意外，再试一次吧';
    }

    function makeRequestId() {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
        return 'msg-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
    }

    async function ensureServerConversation(epoch) {
        if (!Number.isInteger(epoch)) epoch = state.conversationEpoch;
        if (!state.serverConversationId) {
            try { state.serverConversationId = localStorage.getItem(getServerConversationKey()); }
            catch (e) { state.serverConversationId = null; }
        }
        if (state.serverConversationId) return state.serverConversationId;
        var existing = state.serverConversationCreateRecord;
        if (existing && existing.epoch === epoch) return existing.promise;
        // Creation is serialized across resets: each response sets an auth cookie,
        // so an old in-flight create must finish before the new session is created.
        var previous = existing ? existing.promise.catch(function () {}) : Promise.resolve();
        var promise = previous.then(async function () {
            if (epoch !== state.conversationEpoch) return null;
            if (state.serverConversationId) return state.serverConversationId;
            var created = await apiRequest('/api/conversations', {});
            if (!created || !created.conversation_id) throw new Error('会话没有成功建立，请稍后重试。');
            if (epoch !== state.conversationEpoch) return null;
            state.serverConversationId = created.conversation_id;
            try { localStorage.setItem(getServerConversationKey(), state.serverConversationId); } catch (e) {}
            return state.serverConversationId;
        });
        var record = { epoch: epoch, promise: promise };
        state.serverConversationCreateRecord = record;
        promise.catch(function () {
            if (state.serverConversationCreateRecord === record) state.serverConversationCreateRecord = null;
        });
        return promise;
    }

    function waitForNextPoll(delayMs) {
        return new Promise(function (resolve) { setTimeout(resolve, delayMs || 900); });
    }

    async function submitQueuedMessage(query, requestBody, replyBubble, epoch) {
        var conversationId = await ensureServerConversation(epoch);
        if (epoch !== state.conversationEpoch) return null;
        var requestId = makeRequestId();
        var payload = Object.assign({ query: query, request_id: requestId }, requestBody || {});
        var accepted;
        try {
            accepted = await apiRequest('/api/conversations/' + encodeURIComponent(conversationId) + '/messages', payload);
        } catch (err) {
            if (!err || err.code !== 'conversation_not_found') throw err;
            // Do not let one stale response invalidate a newer conversation that
            // a concurrent request has already created.
            if (state.serverConversationId === conversationId) {
                state.serverConversationId = null;
                state.serverConversationCreateRecord = null;
                try { localStorage.removeItem(getServerConversationKey()); } catch (e) {}
            }
            conversationId = await ensureServerConversation(epoch);
            if (epoch !== state.conversationEpoch) return null;
            payload.request_id = makeRequestId();
            accepted = await apiRequest('/api/conversations/' + encodeURIComponent(conversationId) + '/messages', payload);
        }
        var runId = accepted && accepted.run_id;
        if (!runId) throw new Error('任务没有进入处理队列，请重试。');
        state.activeRuns[runId] = { conversationId: conversationId, epoch: epoch };
        if (epoch !== state.conversationEpoch) {
            apiRequest('/api/runs/' + encodeURIComponent(runId) + '/cancel', {
                conversation_id: conversationId,
            }).catch(function () {});
            delete state.activeRuns[runId];
            return null;
        }
        return pollRun(runId, conversationId, replyBubble, epoch);
    }

    async function pollRun(runId, conversationId, replyBubble, epoch) {
        state.activeRuns[runId] = { conversationId: conversationId, epoch: epoch };
        var afterSeq = 0;
        var lastLabel = '';
        var terminal = ['completed', 'partial', 'failed', 'needs_input', 'cancelled', 'superseded'];
        var terminalReached = false;
        var pollStartedAt = Date.now();
        var pollFailures = 0;
        try {
            while (epoch === state.conversationEpoch) {
                var run;
                try {
                    run = await apiRequest(
                        '/api/runs/' + encodeURIComponent(runId) + '?conversation_id=' + encodeURIComponent(conversationId)
                            + '&after_seq=' + afterSeq,
                        null, 'GET', 15000
                    );
                    pollFailures = 0;
                } catch (err) {
                    var transient = err && (err.code === 'request_timeout' || err.code === 'internal_error'
                        || /网络请求失败|服务器响应格式错误/.test(err.message || ''));
                    if (!transient) throw err;
                    pollFailures++;
                    updateChatBubble(replyBubble, '连接暂时中断，正在恢复任务…');
                    await waitForNextPoll(Math.min(5000, 900 * pollFailures));
                    continue;
                }
                (run.events || []).forEach(function (event) {
                    afterSeq = Math.max(afterSeq, Number(event.seq) || 0);
                    if (event.event_type === 'node_started' && event.payload && event.payload.label) {
                        lastLabel = event.payload.label;
                    }
                });
                if (lastLabel) updateChatBubble(replyBubble, '🌸 ' + lastLabel + '…');
                if (terminal.indexOf(run.status) !== -1) {
                    terminalReached = true;
                    if (run.result && (run.status === 'completed' || run.status === 'partial' || run.status === 'needs_input')) {
                        run.result._task_id = run.task_id;
                        run.result._task_revision = run.current_task_revision;
                        return run.result;
                    }
                    var message = run.error && run.error.message;
                    throw new Error(message || (run.status === 'cancelled'
                        ? '这项任务已取消。' : '任务没有完成，请稍后重试。'));
                }
                if (Date.now() - pollStartedAt >= 90000) {
                    updateChatBubble(replyBubble, '任务处理时间较长，正在继续等待…');
                }
                await waitForNextPoll();
            }
            return null;
        } finally {
            if (terminalReached || epoch !== state.conversationEpoch) delete state.activeRuns[runId];
        }
    }

    function cancelActiveRuns() {
        Object.keys(state.activeRuns).forEach(function (runId) {
            var active = state.activeRuns[runId];
            if (!active) return;
            apiRequest('/api/runs/' + encodeURIComponent(runId) + '/cancel', {
                conversation_id: active.conversationId,
            }).catch(function () {});
        });
    }

    // 可"对话式引导"的错误码：信息不完整 / 输入无法理解，用气泡友好提示而非报错弹窗
    var GUIDEABLE_ERRORS = [
        'missing_query', 'missing_endpoints', 'missing_poi_names',
        'poi_not_found', 'same_poi', 'parse_failed',
        'parse_validation_error', 'unsupported_task',
    ];
    function isGuideableError(code) {
        return GUIDEABLE_ERRORS.indexOf(code) !== -1;
    }

    // GPS 指代表达识别
    // 起点：「从我这/我这里/我的位置/当前位置…去/到/出发」；终点：「到我这(来)/来我的位置」
    var LOC_START_RE = /从\s*(我这(?:儿|里)?|我的位置|当前位置|我现在的?位置|我这边)/;
    var LOC_BARE_RE = /^(我这(?:儿|里)?|我的位置|当前位置)[^，。,.]{0,6}(去|到|出发)/;
    var LOC_END_RE = /(?:到|去)\s*(我这(?:儿|里)?(?:来)?|我的位置|当前位置)(?:来)?$/;

    function detectLocationRefs(text) {
        return {
            asStart: LOC_START_RE.test(text) || LOC_BARE_RE.test(text),
            asEnd: LOC_END_RE.test(text),
        };
    }

    function isLikelyClarificationAnswer(query) {
        if (!state.pendingServerTaskId || !Number.isInteger(state.pendingServerTaskRevision)) return false;
        var text = String(query || '').trim();
        if (!text) return false;
        // An origin plus "take me there" answers the pending origin question;
        // an added weather request must not detach it from the saved destination.
        if (/(起点|出发|从哪里|在哪儿|在哪里)/.test(state.pendingServerQuestion || '')
                && /^(?:现在)?我在|^从|^起点/.test(text)
                && /带我过去|带我去那(?:里|儿)|去那(?:里|儿)|到那(?:里|儿)/.test(text)) return true;
        // Questions and independent search intents must not be consumed as
        // route-slot answers merely because they are short.
        if (/(天气|温度|下雨|风力|能见度|活动|有哪些|介绍|是什么|什么|哪里|哪儿|怎么|为什么|谁|何时|几点|多少|帮我|想要|给我|请问|能否|是否|查询|查一下|找一下|搜索|推荐|附近|哪家|有没有|路线|规划|导航)/.test(text)) return false;
        // Clear slot values and explicit route/mode changes are safe answers.
        if (/^(从.+|到\S+|去\S+|起点\S*|出发\S*|终点\S*|步行|骑行|骑车|开车|换成\S+|改成\S+|避开\S+|不要\S+|是的|好的|对的|可以)$/.test(text)) return true;
        // A free-form reply such as "我在信息学部" is still an answer to a
        // pending place question, even when it is not an exact POI name.
        if (/(起点|出发|终点|目的地|从哪里|到哪里|地点|哪儿|哪里)/.test(state.pendingServerQuestion || '')
                && text.length <= 40 && !/[？?]/.test(text)) return true;
        // 活动澄清（例如“想打什么球？”）的按钮或自由输入也必须续接
        // 当前任务，不能因为它不是路线槽位而被当成新任务。
        if (/(什么球|哪种运动|球类|运动类型)/.test(state.pendingServerQuestion || '')
                && text.length <= 20 && !/[？?]/.test(text)
                && !/(天气|路线|导航|帮我|推荐|附近|哪里|哪儿)/.test(text)) return true;
        return false;
    }

    async function isKnownPoiClarificationAnswer(query) {
        if (!/(起点|出发|终点|目的地|从哪里|到哪里|地点)/.test(state.pendingServerQuestion || '')) return false;
        var text = String(query || '').trim();
        if (!text || text.length > 20 || /[，。,.!?！？]/.test(text)) return false;
        try {
            var data = await apiRequest('/api/pois?keyword=' + encodeURIComponent(text), null, 'GET', 5000);
            var candidates = data && Array.isArray(data.pois) ? data.pois : [];
            function normalize(value) { return String(value || '').replace(/[\s　]/g, '').toLowerCase(); }
            var expected = normalize(text);
            return candidates.some(function (poi) {
                if (normalize(poi && poi.name) === expected) return true;
                return Array.isArray(poi && poi.aliases) && poi.aliases.some(function (alias) {
                    return normalize(alias) === expected;
                });
            });
        } catch (error) {
            return false;
        }
    }

    // 附近推荐只接收近期、精度合格的真实 GPS 修复；缓存点和手动临时点不能当作当前位置。
    function hasFreshAccurateGpsFix() {
        var fix = state._dispFix;
        if (!state.userLocation || !fix || fix.provisional) return false;
        var age = Date.now() - Number(fix.ts);
        var accuracy = Number(fix.acc);
        return isFinite(age) && age >= 0 && age <= 120000
            && isFinite(accuracy) && accuracy > 0 && accuracy <= 100;
    }

    // 确保有近期定位：已有定位过期或精度不够时，重新等一次可信修复。
    function ensureUserLocation() {
        if (state.productMode === 'planning' || !isNavigationDevice()) {
            return Promise.reject(new Error('电脑规划模式不使用自动定位，请用地图右上角“起点”选点，或输入明确的出发地点。'));
        }
        if (hasFreshAccurateGpsFix()) return Promise.resolve(state.userLocation);

        return new Promise(function (resolve, reject) {
            // 启动跟踪（如果还没启动），等第一次 renderUserLocation 后 resolve
            var resolved = false;
            var origRender = renderUserLocation;
            // 临时包装 renderUserLocation，第一次被调时 resolve
            renderUserLocation = function (lng, lat, accuracy, isManual) {
                origRender(lng, lat, accuracy, isManual);
                if (!resolved && hasFreshAccurateGpsFix()) {
                    resolved = true;
                    renderUserLocation = origRender;
                    resolve(state.userLocation);
                }
            };
            // 启动跟踪
            startTracking(true);
            // 10 秒后还没 resolve → 恢复原函数并 reject
            setTimeout(function () {
                if (!resolved) {
                    resolved = true;
                    renderUserLocation = origRender;  // 恢复原函数
                    reject(new Error('定位超时，请稍后重试或在地图上手动选点。'));
                }
            }, 10000);
        });
    }

    async function handleNlSubmit(query, isAutoLocationRetry, continuation, extraContext, mapPointContext) {
        hideError();
        if (!isAutoLocationRetry) state._autoLocated = false;  // 同一请求自动补定位最多一次
        var selectedPointContext = mapPointContext || snapshotMapPoints();
        hideWelcomeElements();
        var turnId = continuation ? continuation.turnId : ++state.nextConversationTurnId;
        var conversationEpoch = continuation ? continuation.conversationEpoch : state.conversationEpoch;
        var thinkingBubble = continuation && continuation.thinkingBubble;
        if (!continuation) {
            showUserBubble(query);
            // 每个请求都有自己的结果气泡，互不覆盖。
            thinkingBubble = showChatBubble(query, '🌸 正在为你规划路线…');
        }

        try {
            // GPS 指代表达：识别「从我这到X / 到我这来」，现场补一次定位
            var requestBody = {
                travel_mode: state.travelMode,
            };
            if (extraContext) requestBody = Object.assign(requestBody, extraContext);
            var locRefs = detectLocationRefs(query);
            if (locRefs.asStart || locRefs.asEnd) {
                var loc = await ensureUserLocation();
                if (conversationEpoch !== state.conversationEpoch) return;
                var coordRef = { lng: loc.lng, lat: loc.lat, name: '我的位置' };
                if (locRefs.asStart) requestBody.coord_start = coordRef;
                if (locRefs.asEnd) requestBody.coord_end = coordRef;
            }
            // 用户手动选过起点/途经点时，将其作为后续自然语言路线请求的坐标上下文。
            // 明确说“从我这里出发”时，上面的实时定位仍优先于地图手选点。
            if (!requestBody.coord_start && selectedPointContext.start) {
                var selectedStart = selectedPointContext.start;
                requestBody.coord_start = {
                    lng: selectedStart.lng, lat: selectedStart.lat, name: '地图起点',
                };
            }
            if (!requestBody.coord_end && selectedPointContext.end) {
                var selectedEnd = selectedPointContext.end;
                requestBody.coord_end = {
                    lng: selectedEnd.lng, lat: selectedEnd.lat, name: '地图终点',
                };
            }
            if (!requestBody.coord_waypoints && selectedPointContext.via.length) {
                requestBody.coord_waypoints = selectedPointContext.via.map(function (point) {
                    return { lng: point.lng, lat: point.lat };
                });
            }
            // 自动附起点仅使用近期且精度合格的 GPS。过期、粗略或缓存定位交给后端追问。
            if (!requestBody.coord_start && hasFreshAccurateGpsFix() && !query._auto_located) {
                requestBody.coord_start = {
                    lng: state.userLocation.lng, lat: state.userLocation.lat, name: '我的位置',
                };
            }
            var requestUsesSelectedMapPoints = requestUsesMapPoints(requestBody, selectedPointContext);
            if (requestUsesSelectedMapPoints && selectedPointContext.revision !== state.mapPointRevision) {
                updateChatBubble(thinkingBubble, '地图选点刚有变化，请按当前选点重新发送路线需求。');
                return;
            }
            var clarificationAnswer = isLikelyClarificationAnswer(query);
            if (!clarificationAnswer && state.pendingServerTaskId) {
                clarificationAnswer = await isKnownPoiClarificationAnswer(query);
                if (conversationEpoch !== state.conversationEpoch) return;
            }
            if (!requestBody.continuation_task_id && clarificationAnswer) {
                requestBody.continuation_task_id = state.pendingServerTaskId;
                requestBody.base_revision = state.pendingServerTaskRevision;
            }

            var result = await submitQueuedMessage(query, requestBody, thinkingBubble, conversationEpoch);
            if (!result) return;
            if (requestUsesSelectedMapPoints && selectedPointContext.revision !== state.mapPointRevision) {
                updateChatBubble(thinkingBubble, '地图选点已更新，这条旧路线已取消显示。请按当前选点继续规划。');
                return;
            }

            if (result.response_kind === 'clarify') {
                if (setPendingServerClarification(
                    result._task_id, result._task_revision,
                    result.clarify && result.clarify.question, turnId
                )) saveContext();
            } else if (requestBody.continuation_task_id
                    && requestBody.continuation_task_id === state.pendingServerTaskId) {
                if (clearPendingServerClarification(requestBody.continuation_task_id, turnId)) saveContext();
            }

            if (conversationEpoch !== state.conversationEpoch) return;

            var taskType = result.task_type;

            if (taskType === 'path_planning' && requestUsesSelectedMapPoints) state.mapPointRouteActive = true;

            // A comparison is a read-only answer with its own deterministic payload;
            // keep it out of the generic chat branch so the measurements stay visible.
            if (result.response_kind === 'route_comparison') {
                hideWelcomeElements();
                updateChatBubble(thinkingBubble, result.reply || result.message || '路线比较结果见下方。');
                if (result.route_comparison) renderRouteComparisonCard(result.route_comparison);
                if (Array.isArray(result.suggestions) && result.suggestions.length) {
                    showSuggestions(result, thinkingBubble);
                } else {
                    var comparisonSuggestions = document.getElementById('suggestions-area');
                    if (comparisonSuggestions) comparisonSuggestions.hidden = true;
                }
                addConversationTurn(query, result, turnId);
                return;
            }

            if (result.response_kind === 'no_route') {
                hideWelcomeElements();
                updateChatBubble(thinkingBubble, result.reply || result.message || '当前条件下没有找到完整路线。');
                if (turnId >= state.latestRouteTurnId) {
                    state.latestRouteTurnId = turnId;
                    clearRouteResult();
                    state.latestRoute = null;
                    state.mapPointRouteActive = false;
                    if (state.routeStore) state.routeStore.replace(null);
                    var rejectedRoute = result.route_state;
                    state.pendingIntent = rejectedRoute && rejectedRoute.start && rejectedRoute.end ? {
                        task_type: 'path_planning', start: rejectedRoute.start,
                        end: rejectedRoute.end,
                        constraints: rejectedRoute.hard_constraints || null,
                        mode: rejectedRoute.travel_mode || state.travelMode,
                    } : null;
                    syncRouteStrategyUI();
                }
                addConversationTurn(query, result, turnId, false);
                return;
            }

            if (taskType === 'chat') {
                hideWelcomeElements();
                updateChatBubble(thinkingBubble, result.reply || result.message || '嗯…这个问题有点难，换个问法试试？');
                if (Array.isArray(result.suggestions) && result.suggestions.length) {
                    showSuggestions(result, thinkingBubble);
                } else {
                    var suggestionsArea = document.getElementById('suggestions-area');
                    if (suggestionsArea) suggestionsArea.hidden = true;
                }
                addConversationTurn(query, result, turnId);
                return;
            }

            // clarify（response_kind）：Agent 主动追问 → 渲染可点选项
            if (result.response_kind === 'clarify' && result.clarify) {
                // 层 2：如果后端追问的是起点（"从哪出发""你在哪儿"等），且没标记过自动定位 → 自动定位后重发
                var clarifyMsg = (result.clarify && result.clarify.question) || result.message || '';
                var NEED_START_RE = /起点|从哪|你在哪|在哪儿|哪里出发|告诉我起点|出发地/;
                if (NEED_START_RE.test(clarifyMsg) && !state._autoLocated
                    && state.productMode === 'navigation') {
                    try {
                        console.log('[AUTO_LOC] clarify 追问起点 → 自动定位并重发');
                        state._autoLocated = true;
                        var autoLoc = await ensureUserLocation();
                        if (conversationEpoch !== state.conversationEpoch) return;
                        if (autoLoc) state.userLocation = autoLoc;
                        // 复用本轮气泡与顺序号，自动补定位最多一次。
                        var retryContext = Object.assign({}, requestBody);
                        if (result._task_id && Number.isInteger(result._task_revision)) {
                            retryContext.continuation_task_id = result._task_id;
                            retryContext.base_revision = result._task_revision;
                        }
                        await handleNlSubmit(query, true, {
                            turnId: turnId,
                            thinkingBubble: thinkingBubble,
                            conversationEpoch: conversationEpoch,
                        }, retryContext, selectedPointContext);
                        return;  // 已重发，不再往下渲染
                    } catch (autoErr) {
                        console.warn('[AUTO_LOC] 自动定位失败，显示 clarify 让用户手动选:', autoErr.message);
                        state._autoLocated = false;  // 失败重置，下次还能再试
                        // fallthrough 渲染 clarify 选项
                    }
                }
                hideWelcomeElements();
                updateChatBubble(thinkingBubble, clarifyMsg || '能再具体一点吗？');
                renderClarifyOptions(result.clarify.options || [], result._task_id, result._task_revision);
                addConversationTurn(query, result, turnId);
                return;
            }

            // candidates（response_kind）：目标型需求 → 渲染候选卡片
            if (result.response_kind === 'candidates') {
                hideWelcomeElements();
                var cands = result.candidates || [];
                updateChatBubble(thinkingBubble, result.message || (cands.length ? '帮你找到这些地点，点一个我帮你规划路线～' : '校内没找到匹配的地点，换个说法试试？'));
                // 候选本身也是一个待完成任务：保留任务版本，点选时把
                // 稳定 poi_id 续回后端，避免按名称重新匹配到别的地点。
                if (setPendingServerClarification(
                    result._task_id, result._task_revision,
                    cands.length ? '请选择一个地点继续规划路线。' : null, turnId
                )) saveContext();
                if (cands.length) renderCandidateCards(cands, result._task_id, result._task_revision);
                addConversationTurn(query, result, turnId);
                return;
            }

            if (taskType === 'help' || taskType === 'unknown') {
                hideWelcomeElements();
                updateChatBubble(thinkingBubble, result.message || '有什么可以帮你的？');
                addConversationTurn(query, result, turnId);
                return;
            }

            if (taskType === 'poi_query') {
                hideWelcomeElements();
                var poi = result.poi;
                updateChatBubble(thinkingBubble, result.message || (poi ? poi.description : '找到相关信息了～'));
                if (poi && state.map) {
                    focusPoiOnMap(poi);
                }
                addConversationTurn(query, result, turnId);
                return;
            }

            // path_planning → 渲染路线 + 对话反馈
            if (result.recommended && result.recommended.length > 0) {
                if (turnId >= state.latestRouteTurnId) {
                    state.latestRouteTurnId = turnId;
                    syncModeFromServer(result);
                    renderRoute(result);
                    showResults(result);
                    trackRouteShown();
                    var reply = state.productMode === 'planning'
                        ? buildRouteSummary(result)
                        : appendRouteFieldTruthNotice(result.explanation || buildRouteSummary(result), result);
                    var rankedCandidates = rankedRouteCandidatesSummary(result);
                    if (rankedCandidates) reply = rankedCandidates + ' ' + reply;
                    updateChatBubble(thinkingBubble, reply);
                    if (result.needs_input && result._task_id
                            && Number.isInteger(result._task_revision)) {
                        var nextQuestion = result.clarify && result.clarify.question;
                        if (setPendingServerClarification(
                            result._task_id, result._task_revision, nextQuestion, turnId
                        )) saveContext();
                        if (nextQuestion) showChatBubble('', nextQuestion);
                        if (Array.isArray(result.candidates) && result.candidates.length) {
                            renderCandidateCards(result.candidates, result._task_id, result._task_revision);
                        } else if (result.clarify && Array.isArray(result.clarify.options)) {
                            renderClarifyOptions(result.clarify.options,
                                result._task_id, result._task_revision);
                        }
                    }
                    addConversationTurn(query, result, turnId);
                } else {
                    updateChatBubble(thinkingBubble, '这条较早提交的路线已返回；地图保留了你之后生成的路线。');
                    addConversationTurn(query, result, turnId, false);
                }
            } else {
                updateChatBubble(thinkingBubble, '唔，这条路我没能规划出来😅 试试换个目的地？比如「从珞珈门到樱顶」');
                addConversationTurn(query, result, turnId);
            }
        } catch (err) {
            if (conversationEpoch !== state.conversationEpoch) return;
            // 所有错误都走对话气泡，不弹错误窗
            hideWelcomeElements();
            var errMsg = (err && err.message) || '出了点小问题，再试一次吧～';
            updateChatBubble(thinkingBubble, errMsg);
            addConversationTurn(query, { task_type: 'error', message: errMsg }, turnId);
        } finally { /* Each task updates its own reply bubble. */ }
    }

    // 显示用户输入气泡（靠右，主题色）
    function showUserBubble(query) {
        if (!query) return;
        var chatContent = document.getElementById('chat-content');
        if (!chatContent) return;

        var bubble = document.createElement('div');
        bubble.className = 'chat-bubble-row';
        bubble.innerHTML =
            '<div class="chat-bubble chat-bubble-user">' +
                '<div class="chat-bubble-body">' +
                    '<p class="chat-bubble-text">' + escapeHtml(query) + '</p>' +
                '</div>' +
            '</div>';
        chatContent.appendChild(bubble);
        // 用户主动发起新请求时跟随到自己的新消息；后续回复仅在用户仍靠近底部时跟随。
        scrollChatToLatest(chatContent, true);
    }

    // 在对话流中显示聊天气泡，返回气泡元素供后续 updateChatBubble 替换内容
    function showChatBubble(query, reply) {
        var chatContent = document.getElementById('chat-content');
        if (!chatContent) return null;
        var followLatest = chatIsNearLatest(chatContent);

        var bubble = document.createElement('div');
        bubble.className = 'chat-bubble-row';
        bubble.innerHTML =
            '<div class="chat-bubble chat-bubble-reply">' +
                '<div class="chat-bubble-avatar">🌸</div>' +
                '<div class="chat-bubble-body">' +
                    '<p class="chat-bubble-text">' + escapeHtml(reply) + '</p>' +
                '</div>' +
            '</div>';
        chatContent.appendChild(bubble);
        scrollChatToLatest(chatContent, followLatest);

        return bubble;
    }

    function renderRouteComparisonCard(comparison) {
        var chatContent = document.getElementById('chat-content');
        if (!chatContent || !comparison) return;
        var alternatives = Array.isArray(comparison.alternatives) ? comparison.alternatives : [];
        var unavailable = Array.isArray(comparison.unavailable) ? comparison.unavailable : [];
        var modeLabels = { walk: '步行', bike: '骑行', drive: '驾车' };
        var start = String(comparison.start_name || '起点');
        var end = String(comparison.end_name || '终点');

        var row = document.createElement('div');
        row.className = 'chat-bubble-row route-comparison-row';
        var card = document.createElement('section');
        card.className = 'route-comparison-card';
        card.setAttribute('role', 'group');
        card.setAttribute('aria-label', '路线方案比较');

        var heading = document.createElement('div');
        heading.className = 'route-comparison-heading';
        var endpoints = document.createElement('strong');
        endpoints.className = 'route-comparison-endpoints';
        endpoints.textContent = start + ' → ' + end;
        var availability = document.createElement('span');
        availability.className = 'route-comparison-availability';
        availability.textContent = (modeLabels[comparison.mode] || '步行') + ' · ' + alternatives.length + ' 种可用';
        heading.appendChild(endpoints);
        heading.appendChild(availability);
        card.appendChild(heading);

        if (!alternatives.length) {
            var empty = document.createElement('p');
            empty.className = 'route-comparison-empty';
            empty.textContent = '当前条件下没有可行方案。';
            card.appendChild(empty);
        }

        alternatives.forEach(function (item) {
            if (!item || typeof item !== 'object') return;
            var option = document.createElement('div');
            option.className = 'route-comparison-option';
            var primary = document.createElement('div');
            primary.className = 'route-comparison-option-primary';
            var label = document.createElement('span');
            label.className = 'route-comparison-label';
            label.textContent = String(item.label || item.strategy || '路线方案');
            var distance = document.createElement('strong');
            distance.className = 'route-comparison-distance';
            var metres = Number(item.distance_m);
            distance.textContent = isFinite(metres)
                ? (metres >= 1000 ? (metres / 1000).toFixed(1) + ' km' : Math.round(metres) + ' m')
                : '距离未知';
            primary.appendChild(label);
            primary.appendChild(distance);
            option.appendChild(primary);

            var details = [];
            var duration = Number(item.duration_min);
            if (item.duration_min != null && isFinite(duration)) details.push('约 ' + duration.toFixed(1) + ' 分钟');
            var delta = Number(item.distance_delta_from_shortest_m);
            if (item.distance_delta_from_shortest_m != null && isFinite(delta) && delta > 0) {
                details.push('比最短多 ' + Math.round(delta) + ' m');
            }
            var slope = Number(item.slope_level_avg);
            if (item.slope_level_avg != null && isFinite(slope)) details.push('坡度 ' + slope.toFixed(1));
            var scenery = Number(item.scenery_level_avg);
            if (item.scenery_level_avg != null && isFinite(scenery)) details.push('风景 ' + scenery.toFixed(1));
            if (details.length) {
                var meta = document.createElement('p');
                meta.className = 'route-comparison-meta';
                meta.textContent = details.join(' · ');
                option.appendChild(meta);
            }
            card.appendChild(option);
        });

        unavailable.forEach(function (item) {
            if (!item || typeof item !== 'object') return;
            var missing = document.createElement('div');
            missing.className = 'route-comparison-option is-unavailable';
            var primary = document.createElement('div');
            primary.className = 'route-comparison-option-primary';
            var label = document.createElement('span');
            label.className = 'route-comparison-label';
            label.textContent = String(item.label || item.strategy || '路线方案');
            var status = document.createElement('span');
            status.className = 'route-comparison-unavailable-label';
            status.textContent = '暂不可用';
            primary.appendChild(label);
            primary.appendChild(status);
            missing.appendChild(primary);
            if (item.reason) {
                var reason = document.createElement('p');
                reason.className = 'route-comparison-meta';
                reason.textContent = String(item.reason);
                missing.appendChild(reason);
            }
            card.appendChild(missing);
        });

        var note = document.createElement('p');
        note.className = 'route-comparison-note';
        note.textContent = '时间为速度模型估算；坡度数值越低越平缓，风景数值越高代表沿线景观属性更高。';
        card.appendChild(note);
        row.appendChild(card);
        var followLatest = chatIsNearLatest(chatContent);
        chatContent.appendChild(row);
        scrollChatToLatest(chatContent, followLatest);
    }

    // 替换已有气泡内容（用于"思考中"→最终回复）
    function updateChatBubble(bubble, text) {
        if (!bubble) return;
        var chatContent = document.getElementById('chat-content');
        var followLatest = chatIsNearLatest(chatContent);
        var textEl = bubble.querySelector('.chat-bubble-text');
        if (textEl) {
            textEl.textContent = text;
        }
        scrollChatToLatest(chatContent, followLatest);
    }

    function routeOverviewEndpoint(endpoint, fallback) {
        if (typeof endpoint === 'string' && endpoint.trim()) return endpoint.trim();
        return endpointLabel(endpoint, fallback);
    }

    function routeOverviewStopCount(routeKind, current, data) {
        var itinerary = data.itinerary || (current && current.itinerary) || {};
        if (routeKind === 'itinerary') {
            return Number(itinerary.stop_count || (itinerary.stop_names || []).length || 0);
        }
        var via = (current && current.via) || data.via;
        if (Array.isArray(via)) return via.length;
        if (via && Array.isArray(via.points)) return via.points.length;
        return routeKind === 'via' && via ? 1 : 0;
    }

    function routeOverviewPoiCount(routeKind, data) {
        var pois = data.pois || [];
        if (routeKind === 'tour' && data.tour && Array.isArray(data.tour.ordered_pois)) {
            pois = data.tour.ordered_pois;
        }
        return Array.isArray(pois) ? pois.length : 0;
    }

    function conciseRouteReason(explanation) {
        var text = String(explanation || '').trim();
        if (!text) return '';
        return text
            .replace(/^(?:(?:已为你|已为您|为你|为您)?规划(?:好|好了|完成)?从.+?到(?:终点|目的地)的(?:步行|骑行|驾车)?路线)[，,。；;\s]*/g, '')
            .replace(/^(?:路线已规划|路线规划完成)[，,。；;\s]*/g, '')
            .replace(/^(?:(?:已为你|已为您|为你|为您)?规划(?:好了|好|完成)?(?:一条)?(?:步行|骑行|驾车)?路线)[，,。；;\s]*/g, '')
            .replace(/(?:步行|骑行|驾车)?\s*(?:约)?\s*\d+(?:[.,]\d+)?\s*(?:公里|km|米|m|分钟)/gi, '')
            .replace(/(?:最短路径|风景优先|平坦优先|综合推荐|个性化路线|优先较短路线)(?:策略)?/g, '')
            .replace(/按(?:默认|快捷|当前|您的|用户)?偏好[，,、]*/g, '')
            .replace(/^[，,、；;。\s]+|[，,、；;。\s]+$/g, '')
            .trim();
    }

    function routeFieldTruthNotice(data) {
        var verification = data && data.route_verification || {};
        if (verification.status !== 'pass' || verification.field_truth_verified !== false) return '';
        return '提示：本次核对的是路网和路线约束，未实地确认道路当前可通行。';
    }

    function appendRouteFieldTruthNotice(text, data) {
        var summary = String(text || '').trim();
        var notice = routeFieldTruthNotice(data);
        if (!notice || summary.indexOf(notice) !== -1) return summary;
        return summary ? summary + ' ' + notice : notice;
    }

    function rankedRouteCandidatesSummary(data) {
        var candidates = data && data.recommendation_candidates;
        if (!Array.isArray(candidates) || !candidates.length) return '';
        var ranked = candidates.slice(0, 4).filter(function (candidate) {
            return candidate && typeof candidate.name === 'string' && candidate.name.trim();
        });
        if (!ranked.length) return '';
        var labels = ranked.map(function (candidate, index) {
            var distance = candidate.distance_m;
            var estimate = typeof distance === 'number' && Number.isFinite(distance) && distance >= 0
                ? '（约 ' + Math.round(distance) + ' 米）' : '';
            return (index + 1) + '. ' + candidate.name.trim() + estimate;
        });
        return '附近地点按路网与接驳估算从近到远：' + labels.join('；')
            + '。已规划到' + ranked[0].name.trim() + '。';
    }

    // 电脑端把路线信息合并成一条可读总览，避免统计卡与助手反馈重复报数。
    function buildRouteSummary(data) {
        data = data || {};
        var current = state.routeStore ? state.routeStore.current() : null;
        var start = routeOverviewEndpoint((current && current.start) || data.start, '地图起点');
        var end = routeOverviewEndpoint((current && current.end) || data.end, '地图终点');
        var mode = (current && current.travel_mode) || data.mode || state.travelMode;
        var modeLabel = (TRAVEL_MODES[mode] || TRAVEL_MODES.walk).label;
        var routeLength = Number(data.recommended_length_m || data.distance_m || 0);
        var distanceText = routeLength >= 1000
            ? (routeLength / 1000).toFixed(1) + ' 公里'
            : (routeLength > 0 ? Math.round(routeLength) + ' 米' : '距离待估');
        var duration = estimateDurationText(data).replace(/^约\s*/, '');
        var strategy = (current && current.strategy) || data.strategy || {};
        var strategyName = typeof strategy === 'string' ? strategy : strategy.name;
        var strategyLabel = STRATEGY_LABELS[strategyName] || '综合推荐';
        var routeKind = (current && current.route_kind) || data.route_kind || 'direct';
        var tour = data.tour || (current && current.tour) || {};
        var endpointAccess = data.endpoint_access || {};
        var connectorMeters = (function sumConnectorDistances(value) {
            if (!value || typeof value !== 'object') return 0;
            if (Object.prototype.hasOwnProperty.call(value, 'snap_distance_m')) {
                var connector = Number(value.snap_distance_m);
                return Number.isFinite(connector) && connector > 0 ? connector : 0;
            }
            return Object.keys(value).reduce(function (sum, key) {
                return sum + sumConnectorDistances(value[key]);
            }, 0);
        })(endpointAccess);
        var hasConnectorEstimate = connectorMeters >= 1;
        var distancePhrase = modeLabel + (hasConnectorEstimate ? '路网段约 ' : '约 ') + distanceText;
        var connectorDisclosure = hasConnectorEstimate
            ? '端点接驳另按直线估算约 ' + Math.round(connectorMeters) + ' 米，尚未核实实际通行。'
            : '';
        var overview = routeKind === 'tour' && tour.loop !== false
            ? '游览环线：从' + start + '出发，游览后返回' + start + '。' + distancePhrase
            : '路线总览：' + start + ' → ' + end + '。' + distancePhrase;
        if (duration !== '—') overview += '，预计用时约 ' + duration;
        overview += '；采用' + strategyLabel + '策略。';
        if (data.hard_budget_fallback) overview += ' 原偏好路线超过上限，已采用符合上限的最短路线。';
        if (connectorDisclosure) overview += ' ' + connectorDisclosure;
        overview = appendRouteFieldTruthNotice(overview, data);

        var facts = [];
        var stopCount = routeOverviewStopCount(routeKind, current, data);
        var poiCount = routeOverviewPoiCount(routeKind, data);
        if (routeKind === 'tour') {
            if (poiCount) facts.push('串联 ' + poiCount + ' 处景点');
        } else if (routeKind === 'itinerary') {
            if (stopCount) facts.push('包含 ' + stopCount + ' 个行程地点');
            var budget = (data.itinerary && data.itinerary.time_budget_min)
                || (current && current.itinerary && current.itinerary.time_budget_min);
            if (budget) facts.push('时间预算 ' + Math.round(Number(budget)) + ' 分钟');
        } else if (stopCount) {
            facts.push('包含 ' + stopCount + ' 个途经点');
        }
        if (poiCount && routeKind !== 'tour' && routeKind !== 'itinerary') {
            facts.push('沿途可查看 ' + poiCount + ' 处地点');
        }
        var conditionCount = Number(data.road_conditions_applied || 0);
        if (conditionCount > 0) facts.push('已核对 ' + conditionCount + ' 条当前生效路况信息');
        if (facts.length) overview += ' ' + facts.join('；') + '。';

        var reason = conciseRouteReason(data.explanation);
        if (reason.length >= 4) overview += ' 路线特点：' + reason;
        if ((Array.isArray(data.steps) && data.steps.length) || poiCount) {
            overview += ' 分段路线和地点名单可展开查看。';
        }
        return overview;
    }

    function candidateRouteQuery(candidate) {
        var name = String(candidate && candidate.name || '').trim();
        var startName = String(candidate && candidate.recommendation_start_name || '').trim();
        return startName && startName !== '我的位置'
            ? '从' + startName + '到' + name
            : '去' + name;
    }

    // 候选 POI 卡片（response_kind=candidates）：点击时沿用文字起点或当前选定位置
    function renderCandidateCards(candidates, taskId, taskRevision) {
        var chatContent = document.getElementById('chat-content');
        if (!chatContent || !candidates.length) return;
        var followLatest = chatIsNearLatest(chatContent);

        var row = document.createElement('div');
        row.className = 'chat-bubble-row';
        var wrap = document.createElement('div');
        wrap.className = 'candidate-cards';
        row.appendChild(wrap);
        chatContent.appendChild(row);

        candidates.slice(0, 6).forEach(function (p) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'candidate-card';

            var nameEl = document.createElement('span');
            nameEl.className = 'candidate-name';
            nameEl.textContent = p.name || '';
            btn.appendChild(nameEl);

            var placeName = String(p.name || '');
            var activityLabels = Array.isArray(p.activity_labels) ? p.activity_labels : [];
            var cat = activityLabels.length
                ? activityLabels.filter(function (label) {
                    return label && !placeName.includes(String(label));
                }).join('、')
                : (p.category_label || p.subcategory_label || p.subcategory || '');
            if (cat && placeName.includes(String(cat))) cat = '';
            if (cat) {
                var catEl = document.createElement('span');
                catEl.className = 'candidate-cat';
                catEl.textContent = cat;
                btn.appendChild(catEl);
            }

            if (p.campus && !placeName.includes(String(p.campus))) {
                var campusEl = document.createElement('span');
                campusEl.className = 'candidate-campus';
                campusEl.textContent = p.campus;
                btn.appendChild(campusEl);
            }

            if (p.distance_m != null) {
                var distEl = document.createElement('span');
                distEl.className = 'candidate-dist';
                distEl.textContent = p.distance_m >= 1000
                    ? '约 ' + (p.distance_m / 1000).toFixed(1) + ' km'
                    : '约 ' + Math.round(p.distance_m) + ' m';
                btn.appendChild(distEl);
            }

            btn.addEventListener('click', function () {
                // 防连点：点击后整组卡片禁用
                wrap.querySelectorAll('.candidate-card').forEach(function (b) { b.disabled = true; });
                trackEvent('candidate_click', { name: p.name, subcategory: p.subcategory || null });
                var continuation = taskId && Number.isInteger(taskRevision)
                    ? { continuation_task_id: taskId, base_revision: taskRevision,
                        selected_poi_id: p.poi_id || p.place_id || null }
                    : { selected_poi_id: p.poi_id || p.place_id || null };
                handleNlSubmit(candidateRouteQuery(p), false, null, continuation);
            });
            wrap.appendChild(btn);
        });

        // 地图上同步标注候选点，方便用户看着地图挑
        if (state.map) {
            var bounds = [];
            candidates.slice(0, 6).forEach(function (p, i) {
                var lng = p.lon != null ? p.lon : p.lng;
                var lat = p.lat;
                if (lng == null || lat == null) return;
                var pillHtml = '<div class="whu-map-pill whu-map-pill-poi">' + (i + 1) + '. ' + (p.name || '') + '</div>';
                var marker = divMarker(gcjToLatLng(lng, lat), pillHtml, [0, 0], [0, 0], p.name || '')
                    .addTo(state.map);
                // 点地图标记 = 选这个候选（跟点卡片一样）
                marker.on('click', function () {
                    trackEvent('candidate_click_map', { name: p.name, subcategory: p.subcategory || null });
                    var continuation = taskId && Number.isInteger(taskRevision)
                        ? { continuation_task_id: taskId, base_revision: taskRevision,
                            selected_poi_id: p.poi_id || p.place_id || null }
                        : { selected_poi_id: p.poi_id || p.place_id || null };
                    handleNlSubmit(candidateRouteQuery(p), false, null, continuation);
                });
                state.poiMarkers.push(marker);
                bounds.push(marker.getLatLng());
            });
            if (bounds.length > 1) {
                state.map.fitBounds(bounds, { padding: [40, 40] });
            } else if (bounds.length === 1) {
                state.map.setView(bounds[0], 17);
            }
        }

        scrollChatToLatest(chatContent, followLatest);
    }

    // 澄清选项（response_kind=clarify）：点击选项 → 把选项文本作为新 query 提交
    function renderClarifyOptions(options, taskId, taskRevision) {
        var chatContent = document.getElementById('chat-content');
        if (!chatContent || !options.length) return;
        var followLatest = chatIsNearLatest(chatContent);

        var row = document.createElement('div');
        row.className = 'chat-bubble-row';
        var wrap = document.createElement('div');
        wrap.className = 'clarify-chips';
        row.appendChild(wrap);
        chatContent.appendChild(row);

        options.slice(0, 5).forEach(function (opt) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'clarify-chip';
            btn.textContent = opt;
            btn.addEventListener('click', function () {
                // “其他”不是一个地点。把输入焦点交还给用户，并保留
                // 当前澄清任务，避免把它提交成固定 POI 查询。
                if (/^其他(?:球类|运动)?$/.test(String(opt || '').trim())) {
                    var input = document.getElementById('nl-input');
                    if (input) {
                        input.placeholder = /球/.test(opt)
                            ? '请补充球类，例如排球、网球…'
                            : '请补充运动类型，例如游泳、瑜伽…';
                        input.focus();
                    }
                    wrap.querySelectorAll('.clarify-chip').forEach(function (b) { b.disabled = false; });
                    trackEvent('clarify_custom_input', { answer: opt });
                    return;
                }
                wrap.querySelectorAll('.clarify-chip').forEach(function (b) { b.disabled = true; });
                trackEvent('clarify_answer', { answer: opt });
                var continuation = taskId && Number.isInteger(taskRevision)
                    ? { continuation_task_id: taskId, base_revision: taskRevision }
                    : null;
                handleNlSubmit(opt, false, null, continuation);
            });
            wrap.appendChild(btn);
        });

        scrollChatToLatest(chatContent, followLatest);
    }

    function hideWelcomeElements() {
        var welcomeBubble = document.getElementById('welcome-bubble');
        var shortcutCards = document.getElementById('shortcut-cards-row');
        if (welcomeBubble) welcomeBubble.style.display = 'none';
        if (shortcutCards) shortcutCards.style.display = 'none';
    }

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async function handleShortcutMode(mode) {
        hideError();

        var modeConfig = {
            distance_first: {
                loadingText: '正在规划最短路径…',
                loadingSubtext: '优先考虑距离',
                userQuery: '帮我规划最短路径',
            },
            scenery_first: {
                loadingText: '正在规划风景路线…',
                loadingSubtext: '优先考虑景观',
                userQuery: '帮我找一条风景好的路线',
            },
            slope_avoid: {
                loadingText: '正在规划平坦路线…',
                loadingSubtext: '优先考虑坡度',
                userQuery: '帮我找一条平坦的路线',
            },
        };

        var cfg = modeConfig[mode] || {};

        var strategyName = {
            distance_first: 'shortest',
            scenery_first: 'scenery',
            slope_avoid: 'flat',
        }[mode];
        var current = state.routeStore ? state.routeStore.current() : null;
        if (current && strategyName) {
            return replanCurrentRoute({ strategy: strategyName }, cfg.loadingText || '更新路线');
        }

        state.requestSeq += 1;
        var mySeq = state.requestSeq;

        // 优先复用上一次成功路线的端点（可能含 WGS-84 坐标，避免「我的位置」这类名无法解析）
        var startEp = { name: '珞珈门' };
        var endEp = { name: '樱顶' };
        if (state.pendingIntent && state.pendingIntent.start && state.pendingIntent.end) {
            startEp = state.pendingIntent.start;
            endEp = state.pendingIntent.end;
        }
        var startName = startEp.name || '珞珈门';
        var endName = endEp.name || '樱顶';

        // 快捷模式也走对话流：用户气泡 + 思考气泡
        hideWelcomeElements();
        showUserBubble(cfg.userQuery || ('从' + startName + '到' + endName));
        var thinkingBubble = showChatBubble('', '🌸 正在为你规划路线…');

        var tm = TRAVEL_MODES[state.travelMode] || TRAVEL_MODES.walk;
        showLoading(tm.loadingText, cfg.loadingSubtext || tm.loadingSub);

        try {
            var parsePayload = {
                start: startEp,
                end: endEp,
                input_method: 'shortcut',
                travel_mode: state.travelMode,
            };
            if (mode) parsePayload.mode = mode;

            var parseResult = await apiRequest('/api/parse', parsePayload);
            if (mySeq !== state.requestSeq) return;

            var routePayload = parseResult || {};
            routePayload.travel_mode = state.travelMode;

            var routeResult = await apiRequest('/api/route', routePayload);
            if (mySeq !== state.requestSeq) return;

            if (routeResult.recommended && routeResult.recommended.length > 0) {
                renderRoute(routeResult);
                showResults(routeResult);
                trackRouteShown();
                // 快捷模式后端不返回 explanation，用兜底摘要
                updateChatBubble(thinkingBubble, buildRouteSummary(routeResult));
            } else {
                clearRouteResult();
                updateChatBubble(thinkingBubble, '唔，这条路我没能规划出来😅 试试换个目的地？');
            }
        } catch (err) {
            if (mySeq !== state.requestSeq) return;
            clearRouteResult();
            var errMsg = (err && err.message) || '出了点小问题，再试一次吧～';
            updateChatBubble(thinkingBubble, errMsg);
        } finally {
            if (mySeq === state.requestSeq) {
                hideLoading();
            }
        }
    }

    // ========== 快捷键系统 ==========

    // 给按钮加 0.35s 闪动反馈（快捷键触发后视觉确认）
    function flashButton(el) {
        if (!el) return;
        el.classList.remove('flash');
        // 强制 reflow 再加 class，确保动画重新播放
        void el.offsetWidth;
        el.classList.add('flash');
        setTimeout(function () { el.classList.remove('flash'); }, 400);
    }

    function showKbdHelp() {
        if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;
        var sec = document.getElementById('kbd-help-section');
        if (sec) sec.hidden = false;
    }
    function hideKbdHelp() {
        var sec = document.getElementById('kbd-help-section');
        if (sec) sec.hidden = true;
    }

    function refreshPage() {
        // 应用内“刷新”开启新一轮规划：原地清除会话和路线，避免页面重载恢复旧聊天。
        // 浏览器 F5 仍是普通重载，会按记忆策略恢复当前会话。
        beginExplicitRefreshSession();
        handleReset();
        state.restoredActiveRuns = [];
    }

    // 全局键盘快捷键
    function handleGlobalKeydown(e) {
        // 输入框/textarea 聚焦时：只放行 Esc，其余字母不拦截（用户在打字）
        var tag = (e.target.tagName || '').toLowerCase();
        var inInput = (tag === 'input' || tag === 'textarea' || e.target.isContentEditable);

        // Esc：关闭所有弹窗（输入框内也生效）
        if (e.key === 'Escape') {
            hideError();
            hideKbdHelp();
            return;
        }

        // 输入框内：不拦截其余按键（Enter/Shift+Enter 已有专门处理）
        if (inInput) return;

        // 数字键 1/2/3：切出行方式（输入框失焦时才触发）
        if (e.key === '1') { triggerTravelMode('walk'); e.preventDefault(); return; }
        if (e.key === '2') { triggerTravelMode('bike'); e.preventDefault(); return; }
        if (e.key === '3') { triggerTravelMode('drive'); e.preventDefault(); return; }

        // 字母快捷键：偏好 chip + 功能键
        var key = e.key.toUpperCase();
        if (key === 'F') { triggerChip('scenery_first'); e.preventDefault(); return; }
        if (key === 'S') { triggerChip('slope_avoid'); e.preventDefault(); return; }
        if (key === 'D') { triggerChip('distance_first'); e.preventDefault(); return; }
        if (key === 'R') { e.preventDefault(); refreshPage(); return; }
        if (e.key === '?' || (e.shiftKey && e.key === '/')) { var hb = document.getElementById('help-btn'); flashButton(hb); showKbdHelp(); e.preventDefault(); return; }

        // / 聚焦输入框
        if (e.key === '/') {
            var inp = document.getElementById('nl-input');
            if (inp) { inp.focus(); inp.select(); e.preventDefault(); }
            return;
        }
    }

    // 快捷键触发出行方式切换
    function triggerTravelMode(mode) {
        var btn = document.querySelector('.travel-mode-btn[data-travel-mode="' + mode + '"]');
        flashButton(btn);
        setTravelMode(mode);
    }

    // 快捷键触发偏好路线（风景/平坦/最短），不再依赖页面上的 chip 元素
    function triggerChip(chipMode, sourceEl) {
        // 驾车模式下 S（平坦优先）无效
        if (chipMode === 'slope_avoid' && state.travelMode === 'drive') return;
        if (sourceEl) flashButton(sourceEl);
        handleShortcutMode(chipMode);
    }

    function bindMobilePanelResizer() {
        var handle = document.getElementById('mobile-panel-resizer');
        if (!handle || !document.body) return;

        function isMobileViewport() {
            return window.matchMedia
                ? window.matchMedia('(max-width: 767px)').matches
                : (window.innerWidth || 0) <= 767;
        }
        function viewportHeight() {
            var visualHeight = window.visualViewport && window.visualViewport.height;
            return Math.max(1, visualHeight || document.documentElement.clientHeight || window.innerHeight || 1);
        }
        function updateAria(height, viewport) {
            var value = Math.round(height / viewport * 100);
            var minimum = clampMobileMapHeight(viewport, 0);
            var maximum = clampMobileMapHeight(viewport, viewport);
            handle.setAttribute('aria-valuemin', String(Math.round(minimum / viewport * 100)));
            handle.setAttribute('aria-valuemax', String(Math.round(maximum / viewport * 100)));
            handle.setAttribute('aria-valuenow', String(value));
            handle.setAttribute('aria-valuetext', '地图约 ' + value + '%，对话约 ' + (100 - value) + '%');
        }
        function setPaneHeight(requestedHeight) {
            if (!isMobileViewport()) return;
            var viewport = viewportHeight();
            var height = clampMobileMapHeight(viewport, requestedHeight);
            state.mobilePaneHeight = height;
            state.mobilePaneRatio = height / viewport;
            document.body.style.setProperty('--mobile-map-pane-height', height + 'px');
            updateAria(height, viewport);
            var refreshMap = function () {
                if (state.map && typeof state.map.invalidateSize === 'function') {
                    state.map.invalidateSize({ pan: false, debounceMoveend: true });
                }
            };
            if (window.requestAnimationFrame) window.requestAnimationFrame(refreshMap);
            else setTimeout(refreshMap, 0);
        }
        function syncViewport() {
            if (!isMobileViewport()) {
                state.mobilePaneHeight = null;
                document.body.style.removeProperty('--mobile-map-pane-height');
                return;
            }
            var viewport = viewportHeight();
            var requested = state.mobilePaneHeight == null
                ? viewport * 0.55
                : viewport * (state.mobilePaneRatio || 0.55);
            setPaneHeight(requested);
        }

        var activePointer = null;
        var pointerOffset = 0;
        handle.addEventListener('pointerdown', function (event) {
            if (!isMobileViewport() || (event.button != null && event.button !== 0)) return;
            activePointer = event.pointerId;
            pointerOffset = event.clientY - (state.mobilePaneHeight || viewportHeight() * 0.55);
            if (handle.setPointerCapture) {
                try { handle.setPointerCapture(event.pointerId); } catch (e) { /* browser fallback */ }
            }
            document.body.classList.add('is-resizing-mobile-panels');
            event.preventDefault();
        });
        handle.addEventListener('pointermove', function (event) {
            if (activePointer !== event.pointerId) return;
            setPaneHeight(event.clientY - pointerOffset);
            event.preventDefault();
        });
        function finishPointer(event) {
            if (activePointer !== event.pointerId) return;
            activePointer = null;
            document.body.classList.remove('is-resizing-mobile-panels');
            if (handle.releasePointerCapture) {
                try { handle.releasePointerCapture(event.pointerId); } catch (e) { /* already released */ }
            }
        }
        handle.addEventListener('pointerup', finishPointer);
        handle.addEventListener('pointercancel', finishPointer);
        handle.addEventListener('lostpointercapture', function () {
            activePointer = null;
            document.body.classList.remove('is-resizing-mobile-panels');
        });
        handle.addEventListener('keydown', function (event) {
            if (!isMobileViewport()) return;
            var current = state.mobilePaneHeight || viewportHeight() * 0.55;
            var step = event.shiftKey ? 80 : 24;
            if (event.key === 'ArrowUp') setPaneHeight(current - step);
            else if (event.key === 'ArrowDown') setPaneHeight(current + step);
            else if (event.key === 'PageUp') setPaneHeight(current - 80);
            else if (event.key === 'PageDown') setPaneHeight(current + 80);
            else if (event.key === 'Home') setPaneHeight(viewportHeight());
            else if (event.key === 'End') setPaneHeight(0);
            else return;
            event.preventDefault();
        });

        syncViewport();
        window.addEventListener('resize', syncViewport);
        if (window.visualViewport) window.visualViewport.addEventListener('resize', syncViewport);
    }

    function bindEvents() {
        bindMobilePanelResizer();
        var nlInput = document.getElementById('nl-input');
        var charCount = document.getElementById('char-count');
        var submitBtn = document.getElementById('submit-btn');
        var quickChips = document.querySelectorAll('.quick-chip');
        var errorCloseBtn = document.getElementById('error-close-btn');
        var resetBtn = document.getElementById('reset-btn');

        if (nlInput) {
            // 字数统计 + auto-resize
            nlInput.addEventListener('input', function () {
                var len = nlInput.value.length;
                if (charCount) charCount.textContent = len;
                if (len >= 200 && charCount) charCount.style.color = '#C76B7A';
                else if (charCount) charCount.style.color = '';
                // auto-resize
                nlInput.style.height = 'auto';
                nlInput.style.height = Math.min(nlInput.scrollHeight, 100) + 'px';
            });

            nlInput.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (submitBtn) submitBtn.click();
                }
            });
        }

        if (submitBtn) {
            submitBtn.addEventListener('click', function () {
                var query = nlInput.value.trim();
                if (!query) {
                    showError('嗯？还没说去哪呢', '告诉我你想从哪走到哪吧～比如「从珞珈门到樱顶」');
                    nlInput.focus();
                    return;
                }
                handleNlSubmit(query);
                // 提交后清空输入框 + 重置字数统计与高度
                nlInput.value = '';
                nlInput.style.height = 'auto';
                if (charCount) charCount.textContent = '0';
            });
        }

        // 快捷 Chip 按钮
        quickChips.forEach(function (chip) {
            chip.addEventListener('click', function () {
                quickChips.forEach(function (c) { c.classList.remove('active'); });
                chip.classList.add('active');
                var mode = chip.getAttribute('data-mode');
                state.activeMode = mode;
                handleShortcutMode(mode);
            });
        });

        // 出行方式分段选择器（步行 / 骑行 / 驾车）：只切状态，不自动发请求（有路线结果时自动重算）
        var travelModeBtns = document.querySelectorAll('.travel-mode-btn');
        travelModeBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                var travelMode = btn.getAttribute('data-travel-mode');
                setTravelMode(travelMode);
            });
        });

        // 路线偏好选择器：通过 route_state 原子重规划，保留途经点、游览点与硬约束。
        document.querySelectorAll('.route-strategy-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var strategy = btn.getAttribute('data-route-strategy');
                if (strategy === 'flat' && state.travelMode === 'drive') return;
                var shortcutMode = strategyToShortcutMode(strategy);
                if (shortcutMode) handleShortcutMode(shortcutMode);
            });
        });

        // 快捷小卡片（经典路线、赏樱路线）
        var miniCards = document.querySelectorAll('.mini-card');
        miniCards.forEach(function (card) {
            card.addEventListener('click', function () {
                var displayText = card.getAttribute('data-display-text');
                if (nlInput && displayText) {
                    nlInput.value = displayText;
                    nlInput.dispatchEvent(new Event('input'));
                }
                if (submitBtn) submitBtn.click();
            });
        });

        if (errorCloseBtn) {
            errorCloseBtn.addEventListener('click', hideError);
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', refreshPage);
        }

        // ===== 实时导航按钮 =====
        var startNavBtn = document.getElementById('start-nav-btn');
        if (startNavBtn) startNavBtn.addEventListener('click', startNavigation);

        var copyMobileLinkBtn = document.getElementById('copy-mobile-link-btn');
        if (copyMobileLinkBtn) {
            copyMobileLinkBtn.addEventListener('click', function () {
                copyText(mobileHandoffUrl(), copyMobileLinkBtn, '链接已复制');
            });
        }
        var copyRouteSummaryBtn = document.getElementById('copy-route-summary-btn');
        if (copyRouteSummaryBtn) {
            copyRouteSummaryBtn.addEventListener('click', function () {
                copyText(routePlanText(), copyRouteSummaryBtn, '路线说明已复制');
            });
        }

        function bindNavStop(id) {
            var el = document.getElementById(id);
            if (el) el.addEventListener('click', stopNavigation);
        }
        bindNavStop('nav-exit-top');
        bindNavStop('nav-stop');
        bindNavStop('nav-arrive-close');

        var navMuteBtn = document.getElementById('nav-mute');
        if (navMuteBtn) {
            navMuteBtn.addEventListener('click', function () {
                if (window.WhuWalkerVoiceOutput) {
                    var muted = WhuWalkerVoiceOutput.toggleMuted();
                    syncMuteIcon();
                    if (!muted) WhuWalkerVoiceOutput.speak('语音已开启');
                }
            });
        }
        var navRecenterBtn = document.getElementById('nav-recenter');
        if (navRecenterBtn) {
            navRecenterBtn.addEventListener('click', function () {
                if (window.WhuWalkerNavigation) WhuWalkerNavigation.setFollow(true);
            });
        }

        // ===== 语音输入（按住说话；松开发送，上滑取消） =====
        var voiceBtn = document.getElementById('voice-btn');
        if (voiceBtn) {
            voiceBtn.title = '按住说话，松开发送，上滑取消';
            voiceBtn.setAttribute('aria-label', '按住说话，松开发送，上滑取消');
            voiceBtn.addEventListener('pointerdown', function (e) {
                e.preventDefault();  // 抑制聚焦/合成 click/滚动
                try { voiceBtn.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
                startVoiceHold(e.clientY, e.pointerId);
            });
            voiceBtn.addEventListener('pointermove', function (e) {
                moveVoiceHold(e.clientY, e.pointerId);
            });
            var releaseVoice = function (e) { endVoiceHold(e.pointerId, false); };
            voiceBtn.addEventListener('pointerup', releaseVoice);
            voiceBtn.addEventListener('pointercancel', function (e) { endVoiceHold(e.pointerId, true); });
            // 兜底：捕获失效/手指滑出按钮
            voiceBtn.addEventListener('lostpointercapture', function () {
                if (voiceHold.active) endVoiceHold(voiceHold.pointerId, voiceHold.cancelled);
            });
            // 键盘可达性：空格/回车按住说话
            voiceBtn.addEventListener('keydown', function (e) {
                if ((e.key === ' ' || e.key === 'Enter') && !voiceHold.active) {
                    e.preventDefault();
                    startVoiceHold(0, -1);
                }
            });
            voiceBtn.addEventListener('keyup', function (e) {
                if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); endVoiceHold(-1, false); }
            });
            // 彻底吞掉 click，避免 pointerup 后二次触发
            voiceBtn.addEventListener('click', function (e) { e.preventDefault(); });
            voiceBtn.addEventListener('contextmenu', function (e) { e.preventDefault(); });
        }

        // 帮助按钮 → 快捷键帮助弹窗
        var helpBtn = document.getElementById('help-btn');
        if (helpBtn) {
            helpBtn.addEventListener('click', function () {
                flashButton(helpBtn);
                showKbdHelp();
            });
        }

        // 上报路况按钮
        var roadReportBtn = document.getElementById('road-report-btn');
        if (roadReportBtn) {
            roadReportBtn.addEventListener('click', function () {
                flashButton(roadReportBtn);
                openRoadReport();
            });
        }
        var rrCancel = document.getElementById('rr-cancel-btn');
        if (rrCancel) rrCancel.addEventListener('click', closeRoadReport);
        var rrLogout = document.getElementById('rr-logout-btn');
        if (rrLogout) rrLogout.addEventListener('click', adminLogout);
        var rrPick = document.getElementById('rr-pick-btn');
        if (rrPick) rrPick.addEventListener('click', startPickLocation);
        var rrAction = document.getElementById('rr-action');
        if (rrAction) rrAction.addEventListener('change', syncRoadReportActionFields);
        syncRoadReportActionFields();
        var rrForm = document.getElementById('road-report-form');
        if (rrForm) rrForm.addEventListener('submit', handleRoadReportSubmit);

        // 管理员登录弹窗
        var adminLoginCancel = document.getElementById('admin-login-cancel');
        if (adminLoginCancel) adminLoginCancel.addEventListener('click', closeAdminLogin);
        var adminLoginForm = document.getElementById('admin-login-form');
        if (adminLoginForm) adminLoginForm.addEventListener('submit', handleAdminLogin);

        // 快捷键帮助弹窗关闭按钮
        var kbdHelpClose = document.getElementById('kbd-help-close');
        if (kbdHelpClose) {
            kbdHelpClose.addEventListener('click', hideKbdHelp);
        }

        // 全局键盘快捷键
        document.addEventListener('keydown', handleGlobalKeydown);

        // 底部快捷键提示条：点击 kbd 直接执行对应动作（不依赖焦点，鼠标可用）
        document.querySelectorAll('#kbd-hint-bar kbd[data-kbd]').forEach(function (k) {
            k.addEventListener('click', function () {
                var action = k.getAttribute('data-kbd');
                var active = document.activeElement;
                if (active && active.id !== 'nl-input') active.blur();
                if (action.indexOf('mode:') === 0) {
                    triggerTravelMode(action.slice(5));
                } else if (action.indexOf('chip:') === 0) {
                    triggerChip(action.slice(5), k);
                } else if (action === 'refresh') {
                    flashButton(k);
                    refreshPage();
                } else if (action === 'focus') {
                    var inp = document.getElementById('nl-input');
                    if (inp) inp.focus();
                } else if (action === 'help') {
                    showKbdHelp();
                }
            });
        });
    }

    function showWelcomeHint() {
        var welcomeBubble = document.getElementById('welcome-bubble');
        var shortcutCards = document.getElementById('shortcut-cards-row');
        if (welcomeBubble) welcomeBubble.style.display = '';
        if (shortcutCards) shortcutCards.style.display = '';
        var results = document.getElementById('results-section');
        if (results) results.hidden = true;
        var sa = document.getElementById('suggestions-area');
        if (sa) sa.hidden = true;
    }

    function restoreRecentBubbles() {
        var history = state.conversationHistory;
        if (!history || history.length === 0) return;
        var chatContent = document.getElementById('chat-content');
        if (!chatContent) return;
        // 有历史对话：隐藏欢迎引导
        hideWelcomeElements();
        // 最近 3 轮（6 条消息）
        var recent = history.slice(-6);
        recent.forEach(function (msg) {
            if (msg.role === 'user') {
                showUserBubble(msg.content);
            } else {
                showChatBubble('', msg.content || '');
            }
        });
    }

    function recordRestoredAssistant(query, result, taskId, turnId) {
        var replyText = result.explanation || result.reply || result.message || '';
        var item = { role: 'assistant', content: replyText, _turnId: turnId, _taskId: taskId,
            task_type: result.task_type || null };
        // The worker commits its assistant message at completion time. Keep the
        // same append order as the server transcript, including parallel tasks.
        state.conversationHistory.push(item);
        if (state.conversationHistory.length > 20) state.conversationHistory = state.conversationHistory.slice(-20);
        saveContext();
    }

    function resumeRestoredRuns() {
        var pending = state.restoredActiveRuns.slice();
        state.restoredActiveRuns = [];
        pending.forEach(function (active) {
            if (!active || !active.run_id || !active.query || !state.serverConversationId) return;
            hideWelcomeElements();
            var recent = state.conversationHistory.slice(-6);
            var queryAlreadyVisible = recent.some(function (message) {
                return message.role === 'user' && message._taskId === active.task_id;
            });
            if (!queryAlreadyVisible) showUserBubble(active.query);
            var bubble = showChatBubble(active.query, '🌸 正在恢复这项任务…');
            var conversationId = state.serverConversationId;
            var epoch = state.conversationEpoch;
            var userMessage = state.conversationHistory.find(function (message) {
                return message.role === 'user' && message._taskId === active.task_id;
            });
            var turnId = userMessage ? userMessage._turnId : 0;
            var messageSequence = Number(active.message_seq) || (userMessage && userMessage._seq) || 0;
            pollRun(active.run_id, conversationId, bubble, epoch).then(function (result) {
                if (!result || epoch !== state.conversationEpoch) return;
                var isAtLeastAsNewAsTranscript = !messageSequence || !state.latestRouteSequence
                    || messageSequence >= state.latestRouteSequence;
                if (result.task_type === 'path_planning' && result.recommended && result.recommended.length
                        && isAtLeastAsNewAsTranscript && turnId >= state.latestRouteTurnId) {
                    if (turnId) state.latestRouteTurnId = turnId;
                    state.latestRouteSequence = Math.max(state.latestRouteSequence, messageSequence);
                    syncModeFromServer(result);
                    renderRoute(result);
                    showResults(result);
                    trackRouteShown();
                }
                if (result.response_kind === 'clarify') {
                    var acceptedPending = setPendingServerClarification(
                        result._task_id || active.task_id,
                        Number.isInteger(result._task_revision) ? result._task_revision : active.task_revision,
                        result.clarify && result.clarify.question, turnId
                    );
                    if (acceptedPending) {
                        saveContext();
                        renderClarifyOptions((result.clarify && result.clarify.options) || [],
                            state.pendingServerTaskId, state.pendingServerTaskRevision);
                    }
                } else if (result.response_kind === 'candidates' && result.candidates && result.candidates.length) {
                    var candidateTaskId = result._task_id || active.task_id;
                    var candidateRevision = Number.isInteger(result._task_revision)
                        ? result._task_revision : active.task_revision;
                    var acceptedCandidates = setPendingServerClarification(
                        candidateTaskId, candidateRevision,
                        '请选择一个地点继续规划路线。', turnId
                    );
                    if (acceptedCandidates) {
                        saveContext();
                        renderCandidateCards(result.candidates, candidateTaskId, candidateRevision);
                    }
                } else if (result.task_type === 'poi_query' && result.poi && state.map) {
                    focusPoiOnMap(result.poi);
                }
                var restoredReply = result.explanation || result.reply || result.message || '任务已完成。';
                if (result.task_type === 'path_planning' && result.recommended && result.recommended.length) {
                    var restoredCandidates = rankedRouteCandidatesSummary(result);
                    if (restoredCandidates) restoredReply = restoredCandidates + ' ' + restoredReply;
                }
                updateChatBubble(bubble, restoredReply);
                recordRestoredAssistant(active.query, result, active.task_id, turnId);
            }).catch(function (error) {
                if (epoch !== state.conversationEpoch) return;
                updateChatBubble(bubble, (error && error.message) || '任务暂时无法恢复，请稍后刷新查看。');
            });
        });
    }

    async function init() {
        applyProductMode();
        consumeExplicitRefreshSession();
        state.sessionId = getOrCreateSessionId();
        state.serverConversationId = readServerConversationId();
        loadContext();
        loadTravelMode();  // 读取持久化的出行方式偏好（非法值回退 walk）
        bindEvents();
        syncTravelModeUI();  // 同步选择器选中态 / 图例 / 驾车隐藏平坦 chip
        syncRouteStrategyUI();
        showWelcomeHint();
        initMap();
        startRoadConditionVersionMonitor();
        // 暴露公开函数给欢迎卡片等模块调用
        window.submitNaturalLanguageQuery = handleNlSubmit;

        // 手机/App 才启动定位；电脑只做路线规划，不探测定位能力。
        if (state.productMode === 'navigation') setTimeout(autoLocateSilent, 1500);
        consumeMobileHandoff();
        if (state.serverConversationId) await restoreServerConversation();
        restoreRecentBubbles();
        resumeRestoredRuns();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
/* ========================================================================
   珞珈智行 · 冷启动欢迎卡片（方案 1）交互逻辑
   · Append-only IIFE，命名空间：window.WelcomeColdStart
   · 不修改任何原有函数、不覆盖原有全局变量
   · 所有事件绑定使用事件委托 + once/debounce
   · Fail-soft：任何错误不影响原 app.js 核心功能
   ======================================================================== */
(function () {
    'use strict';

    // =============== 常量配置（对应 spec 所有硬编码数字，改一处全联动） ===============
    var LS_KEY_SEEN = 'whu_welcome_seen';
    var LS_KEY_FOREVER = 'whu_welcome_dont_show_forever';
    var DEBOUNCE_MS = 1500; // shortcut 连点防抖窗口（对应 T-017 14.11：1.5s 内连点只发第一次）
    var AUTO_SHOW_DELAY_MS = 150; // DOMContentLoaded 后延迟弹卡（不阻塞首屏 Leaflet 地图加载）
    var HIGHLIGHT_PULSE_CLASS = 'highlight';
    var SIDEBAR_HIGHLIGHT_CLASS = 'flash-highlight';
    var ANIM_OUT_DURATION_MS = 200; // 退场动画总时长（略长于 CSS 180ms，保险）
    var INPUT_ID = 'nl-input'; // 输入框 id（如果前几轮不是这个名字，JS 会自动降级找第一个 input[type=text]/textarea，不崩）

    // =============== 内部状态（闭包私有，不暴露） ===============
    var sessionWelcomeShown = false; // localStorage 不可用时的内存兜底（仅当前标签页）
    var lastShortcutAt = 0; // shortcut 防抖时间戳
    var animating = false; // 防止入场退场动画叠加 + 连续点 X 和 start 造成重复关

    // =============== DOM 引用（缓存一次，懒初始化） ===============
    var $overlay = null;
    var $card = null;
    var $helpBtn = null;
    var $shortcuts = null;
    var $input = null;
    var $resultArea = null;
    var $sidebar = null;
    var $dontShowCheckbox = null;

    function initDomRefs() {
        $overlay = document.getElementById('welcome-overlay');
        $card = $overlay ? $overlay.querySelector('.welcome-card') : null;
        $helpBtn = document.getElementById('help-btn');
        $shortcuts = $overlay ? $overlay.querySelector('.welcome-shortcuts') : null;
        // 输入框：先找 INPUT_ID，找不到就降级找 input[type=text]，再找不到找 textarea（fail-soft）
        $input = document.getElementById(INPUT_ID)
            || document.querySelector('input[type="text"]')
            || document.querySelector('textarea');
        // 结果区：先找 id=result-area，再找 id=route-result
        $resultArea = document.getElementById('result-area') || document.getElementById('route-result');
        // 侧边栏：先找 .poi-sidebar，再找 #poi-sidebar
        $sidebar = document.querySelector('.poi-sidebar') || document.getElementById('poi-sidebar');
        $dontShowCheckbox = document.getElementById('welcome_dont_show');
        // 只要 overlay/helpbtn/shortcuts 三大件在就继续，其他 DOM（input/sidebar/resultArea）找不到没关系，用降级分支
        return !!($overlay && $card && $helpBtn && $shortcuts);
    }

    // =============== localStorage 工具（安全封装，任何抛错自动降级，fail-soft 核心！） ===============
    function safeLsGet(key) {
        try { return window.localStorage.getItem(key); }
        catch (e) { return null; } // 隐私模式/QuotaExceededError 直接当没存
    }
    function safeLsSet(key, value) {
        try { window.localStorage.setItem(key, value); return true; }
        catch (e) { return false; } // 失败就当没写，用内存变量兜底
    }

    // =============== 公开方法：show / close / isOpen ===============
    function show(options) {
        options = options || {};
        var ignoreLs = !!options.ignoreLocalStorage;

        if (!$overlay) return; // DOM 不存在直接静默 fail-soft
        if (animating) return; // 动画中不响应，避免叠图

        // 若卡片已显示：？按钮唤回高亮脉冲（E-07：点了没反应的焦虑 → 给视觉反馈）
        if (isOpen()) {
            $overlay.classList.remove(HIGHLIGHT_PULSE_CLASS);
            // 强制 reflow 重启动画（防止浏览器把两次 class 操作合并）
            void $overlay.offsetWidth;
            $overlay.classList.add(HIGHLIGHT_PULSE_CLASS);
            setTimeout(function () {
                if ($overlay) $overlay.classList.remove(HIGHLIGHT_PULSE_CLASS);
            }, 650); // 动画 600ms + 50ms buffer
            return;
        }

        animating = true;

        // 显示（先移除 hidden 才能播动画）
        $overlay.classList.remove('hidden');
        $overlay.setAttribute('aria-hidden', 'false');

        // 触发入场动画（下一帧再加 .opening，确保 CSS transition 生效，老浏览器不会白屏）
        if (window.requestAnimationFrame) {
            window.requestAnimationFrame(function () {
                if ($overlay) $overlay.classList.add('opening');
            });
        } else {
            // 老浏览器无 rAF，直接加（fail-soft，无动画也能显示）
            setTimeout(function () { if ($overlay) $overlay.classList.add('opening'); }, 16);
        }

        // 动画结束清理（最长 400ms：卡片 280ms + stagger 300ms buffer）
        setTimeout(function () {
            animating = false;
            if ($overlay) $overlay.classList.remove('opening');
        }, 420);

        // 如果不是 ignoreLocalStorage 模式（= 非 ?按钮手动唤回），先标记内存兜底 seen，避免隐私模式闪弹
        if (!ignoreLs) {
            sessionWelcomeShown = true;
        }
    }

    function close(options) {
        options = options || {};
        var markSeen = options.markSeen !== false; // 默认 true：除了调试外都写 seen

        if (!$overlay) return;
        if (animating) return;
        if (!isOpen()) return;

        animating = true;

        // 1. 先写 localStorage（如果需要）
        if (markSeen) {
            safeLsSet(LS_KEY_SEEN, 'true');
            sessionWelcomeShown = true;
            // 如果勾选了"以后不再显示"，再写 forever 标记
            if ($dontShowCheckbox && $dontShowCheckbox.checked) {
                safeLsSet(LS_KEY_FOREVER, 'true');
            }
        }

        // 2. 触发退场动画
        $overlay.classList.add('closing');

        // 3. 动画结束切 .hidden（ANIM_OUT_DURATION_MS 200ms）
        setTimeout(function () {
            animating = false;
            if ($overlay) {
                $overlay.classList.remove('closing');
                $overlay.classList.add('hidden');
                $overlay.setAttribute('aria-hidden', 'true');
            }
        }, ANIM_OUT_DURATION_MS);
    }

    function isOpen() {
        return !!($overlay
            && !$overlay.classList.contains('hidden')
            && !$overlay.classList.contains('closing'));
    }

    // =============== Shortcut-card 数据流分发（3 种 action 走完全不同链路） ===============
    function handleShortcutClick(e) {
        var card = e.target.closest('.shortcut-card');
        if (!card || !($shortcuts && $shortcuts.contains(card))) return;

        // 防抖：DEBOUNCE_MS 1.5s 内只跑第一次（T-017 14.11）
        var now = Date.now();
        if (now - lastShortcutAt < DEBOUNCE_MS) {
            e.preventDefault();
            return;
        }
        lastShortcutAt = now;

        var action = card.getAttribute('data-action');
        var displayText = card.getAttribute('data-display-text') || '';

        // 先填输入框（模拟用户手输，占位 + 方便改 + 多轮上下文继承）
        if ($input && displayText) {
            $input.value = displayText;
        }

        // 分发：所有 action 统一走 Agent（自然语言 query → /api/chat）
        switch (action) {
            case 'path_planning':
            case 'poi_query':
            case 'recommend_poi':
                // 统一复用 handleNlSubmit（走 Agent 循环，含多轮上下文）
                close({ markSeen: true });
                setTimeout(function () {
                    if (typeof window.submitNaturalLanguageQuery === 'function') {
                        window.submitNaturalLanguageQuery(displayText);
                    } else if ($input) {
                        var ev;
                        if (typeof KeyboardEvent === 'function') {
                            ev = new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', which: 13, keyCode: 13, bubbles: true });
                        } else {
                            ev = document.createEvent('Event');
                            ev.initEvent('keydown', true, true);
                            ev.key = 'Enter'; ev.which = 13; ev.keyCode = 13;
                        }
                        $input.dispatchEvent(ev);
                    }
                    if ($resultArea && typeof $resultArea.scrollIntoView === 'function') {
                        try { $resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) { $resultArea.scrollIntoView(); }
                    }
                }, ANIM_OUT_DURATION_MS + 20);
                break;

            default:
                close({ markSeen: true });
        }
    }

    // =============== 事件绑定（全部挂在 overlay 级别，防止 DOM 变更后失效） ===============
    function bindEvents() {
        // ① 关闭按钮统一事件委托（X / mask / 开始使用 —— 所有带 data-close-welcome 的）
        $overlay.addEventListener('click', function (e) {
            var closer = e.target.closest('[data-close-welcome="true"]');
            if (closer) {
                e.preventDefault();
                close({ markSeen: true });
            }
        });
        // ①+ 关键！点卡片内部区域必须阻止冒泡到 mask，否则点哪里都关（最常见 bug，Constraint 6）
        // 但 data-close-welcome 元素（X / 开始使用）在卡片内部，必须允许冒泡到 overlay 触发关闭
        $card.addEventListener('click', function (e) {
            if (e.target.closest('[data-close-welcome="true"]')) {
                return; // 关闭按钮不阻止冒泡，让 overlay 的委托监听器处理
            }
            e.stopPropagation();
        });

        // ② 键盘 Esc 关（仅在卡片显示时有效，不影响输入框；T-017 14.13：Esc 不关 input 内容 —— 这里只关卡，不动 input.value，所以天然满足）
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && isOpen()) {
                close({ markSeen: true });
            }
        });

        // ③ ？帮助按钮唤回（强制 ignoreLocalStorage，不写任何 LS 标记）
        $helpBtn.addEventListener('click', function () {
            show({ ignoreLocalStorage: true });
        });

        // ④ shortcut-card 点击数据流分发（事件委托挂在 .welcome-shortcuts 上，后续加卡片只要加 HTML 不用改 JS）
        $shortcuts.addEventListener('click', handleShortcutClick);
    }

    // =============== 自动弹出判断（DOMContentLoaded 后调用，G-03 老用户不打扰） ===============
    function shouldAutoShow() {
        // 优先级：forever（用户明确不要）→ ls_seen（看过一次）→ session_seen（内存兜底）→ 默认弹
        if (safeLsGet(LS_KEY_FOREVER) === 'true') return false;
        if (safeLsGet(LS_KEY_SEEN) === 'true') return false;
        if (sessionWelcomeShown) return false;
        return true;
    }

    // =============== 初始化入口（fail-soft 第一关：三大件 DOM 找不到就静默 return，不影响原功能） ===============
    function init() {
        if (!initDomRefs()) {
            // DOM 没加（Task 2 没合并 / 或用户没加欢迎卡片 DOM）→ 直接 return，静默降级不报错
            return;
        }
        bindEvents();

        // 按判断结果决定是否自动弹卡
        if (shouldAutoShow()) {
            // 延迟 AUTO_SHOW_DELAY_MS 再弹：避免卡首屏 Leaflet 地图的加载（AUTO_SHOW_DELAY_MS = 150ms）
            setTimeout(function () { show({ ignoreLocalStorage: false }); }, AUTO_SHOW_DELAY_MS);
        }

        // 公开命名空间到 window（供 ?按钮调试 / 测试脚本用 / 后续多轮上下文用）
        window.WelcomeColdStart = {
            show: show,
            close: close,
            isOpen: isOpen,
            shouldAutoShow: shouldAutoShow,
            // 方便测试/调试的工具方法（清 localStorage，测试自动弹卡）
            _debugClearLs: function () {
                try {
                    window.localStorage.removeItem(LS_KEY_SEEN);
                    window.localStorage.removeItem(LS_KEY_FOREVER);
                } catch (e) { /* 忽略 */ }
                sessionWelcomeShown = false;
            }
        };
    }

    // =============== 挂载到 DOMContentLoaded 或立即执行（DOM 已就绪的情况） ===============
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // DOM 已经解析完了（比如 app.js 被异步 defer/async 加载，或者动态插入）→ 等一帧再 init 稳一点
        if (window.requestAnimationFrame) {
            window.requestAnimationFrame(init);
        } else {
            setTimeout(init, 16);
        }
    }
})();
/* ========================================================================
   珞珈智行 · T-024 快捷按钮权重优先级（前端高亮 + 清除逻辑）
   · Append-only IIFE，不修改任何原有函数
   · 高亮类名：.shortcut-active
   · 清除触发：收到 /api/parse 或 /api/chat 返回 weights != null 且 input_method != shortcut 时
   ======================================================================== */
(function () {
    'use strict';

    var SHORTCUT_ACTIVE_CLASS = 'shortcut-active';
    var SHORTCUT_BTN_SELECTOR = '.shortcut-btn';

    // =============== 公开函数（挂 window，供测试/外部调用） ===============
    function addShortcutHighlight(btn) {
        if (!btn) return;
        // 先清除所有按钮高亮（单选行为）
        clearShortcutHighlight();
        btn.classList.add(SHORTCUT_ACTIVE_CLASS);
        // 兼容原代码同时也加 .active（防止其他逻辑依赖 .active）
        btn.classList.add('active');
    }

    function clearShortcutHighlight() {
        var btns = document.querySelectorAll(SHORTCUT_BTN_SELECTOR);
        btns.forEach(function (b) {
            b.classList.remove(SHORTCUT_ACTIVE_CLASS);
            b.classList.remove('active');
        });
    }

    window.addShortcutHighlight = addShortcutHighlight;
    window.clearShortcutHighlight = clearShortcutHighlight;

    // =============== 事件增强：点击快捷按钮即高亮（捕获阶段绑定，append-only 不影响原监听） ===============
    document.addEventListener('click', function (e) {
        var btn = e.target.closest(SHORTCUT_BTN_SELECTOR);
        if (btn) {
            addShortcutHighlight(btn);
        }
    }, true);

    // =============== Fetch 猴子补丁：拦截响应后判断是否清除高亮 ===============
    var _origFetch = window.fetch;
    if (typeof _origFetch === 'function') {
        window.fetch = function (url, options) {
            var promise = _origFetch.apply(this, arguments);
            // 只对本域 API 感兴趣
            var urlStr = String(url);
            var isParseApi = urlStr.indexOf('/api/parse') !== -1;
            var isChatApi = urlStr.indexOf('/api/chat') !== -1;
            if (!isParseApi && !isChatApi) {
                return promise;
            }
            // 克隆响应体，确保原逻辑也能拿到完整数据
            return promise.then(function (response) {
                var cloned = response.clone();
                cloned.json().then(function (payload) {
                    var data = payload && (payload.data || payload);
                    if (!data) return;
                    var weights = data.weights;
                    var inputMethod = data.input_method;
                    // 核心判断：weights != null 且 input_method !== shortcut → 清除高亮
                    // 含义：NL 解析出了显式偏好（LLM 解析权重），快捷按钮预设不再是当前偏好来源
                    if (weights !== null && weights !== undefined && inputMethod !== 'shortcut') {
                        clearShortcutHighlight();
                    }
                    // input_method=shortcut 时保留高亮（说明当前就是快捷按钮生效）
                }).catch(function () {
                    // JSON 解析失败静默忽略，不影响原逻辑
                });
                return response;
            });
        };
    }
})();
/* ========================================================================
   珞珈智行 · T-022 场景 3 候选 POI 交互卡片（候选点选择闭环）
   · Append-only IIFE，命名空间：window.showCandidateCards
   · 不修改任何原有函数、不覆盖原有全局变量
   · 卡片点击 → 自动触发路线规划（复用现有提交流程）
   · Fail-soft：任何错误不影响原 app.js 核心功能
   ======================================================================== */
(function () {
    'use strict';

    var CONTAINER_ID = 'candidates-panel';
    var BRAND_PINK = '#E8929C';
    var BRAND_GOLD = '#D4915C';

    // =============== 内部状态 ===============
    var currentStartName = '';
    var $container = null;

    // =============== 工具函数 ===============
    function escapeHTML(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function renderStars(score) {
        score = Math.max(0, Math.min(5, parseInt(score, 10) || 3));
        var filled = '';
        var empty = '';
        for (var i = 0; i < score; i++) { filled += '★'; }
        for (var i = score; i < 5; i++) { empty += '☆'; }
        return '<span style="color:' + BRAND_GOLD + ';letter-spacing:1px;">' + filled + '</span>'
            + '<span style="color:#CCC;letter-spacing:1px;">' + empty + '</span>';
    }

    function typeBadgeHTML(poiType) {
        var labelMap = { scenery: '赏景', landmark: '地标', study: '学习', gate: '校门', dining: '食堂', service: '服务' };
        var colorMap = {
            scenery: '#E8929C', landmark: '#D4915C', study: '#6A9FB5',
            gate: '#8B7E74', dining: '#C79A63', service: '#7BA37B'
        };
        var label = labelMap[poiType] || poiType || 'POI';
        var bg = colorMap[poiType] || '#A8A5A2';
        return '<span style="display:inline-block;background:' + bg + ';color:#FFF;font-size:11px;'
            + 'padding:2px 8px;border-radius:10px;margin-left:6px;">' + escapeHTML(label) + '</span>';
    }

    // =============== 创建容器 DOM ===============
    function ensureContainer() {
        var el = document.getElementById(CONTAINER_ID);
        if (el) {
            el.innerHTML = '';
            return el;
        }
        el = document.createElement('div');
        el.id = CONTAINER_ID;
        el.style.cssText =
            'margin:16px 20px;padding:0;font-family:"Noto Serif SC","PingFang SC","Microsoft YaHei",serif;';

        var resultsSec = document.getElementById('results-section');
        if (resultsSec && resultsSec.parentNode) {
            resultsSec.parentNode.insertBefore(el, resultsSec);
        } else {
            var inputSec = document.querySelector('.input-section');
            if (inputSec && inputSec.parentNode) {
                inputSec.parentNode.insertBefore(el, inputSec.nextSibling);
            } else {
                var mainEl = document.querySelector('.app-main') || document.querySelector('main');
                if (mainEl) {
                    mainEl.appendChild(el);
                } else {
                    document.body.appendChild(el);
                }
            }
        }
        return el;
    }

    // =============== 候选 POI 点击处理 ===============
    function onCandidateClick(candidateName) {
        var cards = $container ? $container.querySelectorAll('.candidate-card') : [];
        cards.forEach(function (card) {
            card.style.boxShadow = '';
            card.style.transform = '';
        });

        var query = '从' + currentStartName + '到' + candidateName;
        var input = document.getElementById('nl-input')
            || document.querySelector('input[type="text"]')
            || document.querySelector('textarea');

        if (input) {
            input.value = query;
        }

        var submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            setTimeout(function () {
                submitBtn.click();
            }, 80);
        } else {
            if (input) {
                var ev;
                if (typeof KeyboardEvent === 'function') {
                    ev = new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', which: 13, keyCode: 13, bubbles: true });
                } else {
                    ev = document.createEvent('Event');
                    ev.initEvent('keydown', true, true);
                    ev.key = 'Enter'; ev.which = 13; ev.keyCode = 13;
                }
                input.dispatchEvent(ev);
            }
        }

        if ($container) {
            $container.style.transition = 'opacity 0.4s ease';
            $container.style.opacity = '0.3';
            setTimeout(function () {
                if ($container) {
                    $container.innerHTML = '';
                    $container.style.opacity = '1';
                }
            }, 1200);
        }
    }

    // =============== 渲染候选卡片 ===============
    function renderCards(candidates, startName) {
        $container = ensureContainer();
        currentStartName = startName || '';

        if (!candidates || candidates.length === 0) {
            $container.innerHTML =
                '<div style="background:#FDF0F2;border:1px dashed ' + BRAND_PINK + ';border-radius:12px;'
                + 'padding:20px 24px;text-align:center;color:#C76B7A;font-size:14px;">'
                + '🔍 未找到匹配的候选 POI，'
                + '请尝试其他关键词或类型。'
                + '</div>';
            return;
        }

        var html =
            '<div style="margin-bottom:10px;display:flex;align-items:center;gap:8px;">'
            + '<span style="font-size:13px;color:#A8A5A2;font-weight:500;">📍 候选目的地'
            + '（点击卡片自动规划路线）</span>'
            + '<span style="font-size:11px;color:#CCC;">— 从 '
            + escapeHTML(currentStartName) + ' 出发</span>'
            + '</div>'
            + '<div style="display:flex;flex-wrap:wrap;gap:10px;">';

        candidates.forEach(function (c) {
            var starsHTML = renderStars(c.scenery_score || 3);
            var badgeHTML = typeBadgeHTML(c.type || 'landmark');
            var distText = (c.distance_m != null) ? (c.distance_m >= 1000
                ? (c.distance_m / 1000).toFixed(1) + ' km'
                : Math.round(c.distance_m) + ' m') : '—';
            var desc = (c.description || '').substring(0, 60);

            html +=
                '<div class="candidate-card" data-candidate-name="' + escapeHTML(c.name) + '"'
                + ' style="flex:1 1 200px;min-width:180px;max-width:280px;'
                + 'background:#FFF;border:1.5px solid #F0EDE8;border-radius:14px;'
                + 'padding:14px 16px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.06);'
                + 'transition:all 0.2s ease;position:relative;overflow:hidden;'
                + 'box-sizing:border-box;"'
                + ' onmouseenter="this.style.boxShadow=\'0 4px 16px rgba(232,146,156,0.25)\';'
                + 'this.style.borderColor=\'' + BRAND_PINK + '\';this.style.transform=\'translateY(-2px)\';"'
                + ' onmouseleave="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,0.06)\';'
                + 'this.style.borderColor=\'#F0EDE8\';this.style.transform=\'\';">'
                + '<div style="position:absolute;left:0;top:0;bottom:0;width:4px;'
                + 'background:linear-gradient(180deg,' + BRAND_PINK + ',' + BRAND_GOLD + ');'
                + 'border-radius:4px 0 0 4px;"></div>'
                + '<div style="padding-left:8px;">'
                + '<div style="font-size:15px;font-weight:600;color:#3D322B;'
                + 'margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                + escapeHTML(c.name) + badgeHTML + '</div>'
                + '<div style="font-size:12px;margin-bottom:4px;">'
                + '<span style="color:#A8A5A2;">景色 </span>' + starsHTML
                + '<span style="color:#A8A5A2;margin-left:4px;font-size:11px;">'
                + (c.scenery_score || 3) + '/5</span>'
                + '</div>'
                + '<div style="font-size:12px;color:#8B7E74;margin-bottom:4px;">'
                + '📏 路网距离：' + distText + '</div>'
                + (desc
                    ? '<div style="font-size:11px;color:#B5B0AB;line-height:1.4;">'
                    + escapeHTML(desc) + (c.description && c.description.length > 60 ? '...' : '')
                    + '</div>'
                    : '')
                + '</div>'
                + '</div>';
        });

        html += '</div>';
        $container.innerHTML = html;

        $container.addEventListener('click', function (e) {
            var card = e.target.closest('.candidate-card');
            if (!card) return;
            var name = card.getAttribute('data-candidate-name');
            if (name) {
                onCandidateClick(name);
            }
        });
    }

    // =============== 公开命名空间 ===============
    window.showCandidateCards = function (candidates, startName) {
        try {
            renderCards(candidates, startName);
        } catch (e) {
            console.warn('showCandidateCards 渲染失败:', e);
        }
    };

    window.hideCandidateCards = function () {
        try {
            var el = document.getElementById(CONTAINER_ID);
            if (el) {
                el.innerHTML = '';
            }
        } catch (e) { /* 静默忽略 */ }
    };
})();
/* =============== PWA 安装引导（独立模块） =============== */
(function () {
    // v2：横幅改版（APK 直下载）后重置历史"已关闭"记录，让被旧版关掉的用户重新看到
    var LS_DISMISS = 'whu_walker:install_dismissed_at_v2';
    var deferredPrompt = null;
    var banner = null, btn = null, sub = null;

    // 已在安卓壳应用（WebView UA 带 WHUWalkerApp）内：无需引导安装，整个模块关闭
    if (/WHUWalkerApp\//.test(navigator.userAgent)) return;

    function isStandalone() {
        return (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches)
            || window.navigator.standalone === true;
    }
    function isIos() {
        return /iphone|ipad|ipod/i.test(navigator.userAgent)
            || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    }
    function isWeChat() {
        return /MicroMessenger/i.test(navigator.userAgent);
    }
    function recentlyDismissed() {
        try {
            var t = parseInt(localStorage.getItem(LS_DISMISS) || '0', 10);
            return t && (Date.now() - t < 30 * 24 * 3600 * 1000); // 30 天内不再打扰
        } catch (e) { return false; }
    }
    function showBanner(text) {
        if (isStandalone() || recentlyDismissed() || !banner) return;
        if (text && sub) sub.textContent = text;
        banner.hidden = false;
    }
    function hideBanner(persist) {
        if (banner) banner.hidden = true;
        if (persist) {
            try { localStorage.setItem(LS_DISMISS, String(Date.now())); } catch (e) {}
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        banner = document.getElementById('install-banner');
        btn = document.getElementById('install-btn');
        sub = document.getElementById('install-banner-sub');
        if (!banner) return;

        document.getElementById('install-dismiss').addEventListener('click', function () {
            hideBanner(true);
        });
        btn.addEventListener('click', function () {
            if (deferredPrompt) {
                // Chrome / Edge / Android 浏览器：原生安装弹窗
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then(function () {
                    deferredPrompt = null;
                    hideBanner(true);
                });
            } else if (isWeChat()) {
                // 微信内：引导跳转系统浏览器
                if (sub) sub.textContent = '点右上角「⋯」→「在浏览器打开」→ 再添加到桌面';
                btn.textContent = '知道了';
                btn.addEventListener('click', function () { hideBanner(true); }, { once: true });
            } else if (isIos()) {
                // iOS Safari 无原生弹窗：展开文字步骤
                if (sub) sub.textContent = '点底部分享图标「□↑」→ 选「添加到主屏幕」';
                btn.textContent = '知道了';
                btn.addEventListener('click', function () { hideBanner(true); }, { once: true });
            } else {
                // QQ/夸克/UC 等：无安装事件 → 按钮直接下载 APK 安装包
                window.location.href = '/app/whu-walker.apk';
                if (sub) sub.textContent = '已开始下载，完成后在通知栏点开即可安装';
                btn.textContent = '知道了';
                btn.addEventListener('click', function () { hideBanner(false); }, { once: true });
            }
        });

        // Android / Chrome / Edge：捕获系统安装事件
        window.addEventListener('beforeinstallprompt', function (e) {
            e.preventDefault();
            deferredPrompt = e;
            // 延迟一会儿，避免首屏就弹
            setTimeout(function () { showBanner(); }, 2500);
        });
        window.addEventListener('appinstalled', function () { hideBanner(true); });

        // 微信内：beforeinstallprompt 不触发，主动展示引导横幅
        if (isWeChat() && !isStandalone()) {
            setTimeout(function () {
                showBanner('微信不支持直接安装，点右上角「⋯」→ 在浏览器打开');
                if (btn) btn.textContent = '知道了';
            }, 3000);
        }

        // iOS Safari：手动引导（仅非 standalone 且首次）
        if (isIos() && !isStandalone() && !isWeChat()) {
            setTimeout(function () {
                showBanner('点底部分享图标「□↑」→「添加到主屏幕」');
            }, 3000);
        }

        // 安卓非 Chromium 浏览器（QQ/夸克/UC/华为/小米等）：无 PWA 安装能力，
        // 横幅直接提供 APK 下载入口（任何浏览器都能下载安装）
        if (!isIos() && !isWeChat() && !isStandalone()) {
            setTimeout(function () {
                if (deferredPrompt || isStandalone()) return;  // Chromium 内核已捕获安装事件
                var ua = navigator.userAgent;
                var isChromium = /Chrome\/|Edg\//.test(ua)
                    && !/QQBrowser|Quark|UCBrowser|HuaweiBrowser|HeyTapBrowser|VivoBrowser|MiuiBrowser|XiaoMi|OppoBrowser|baiduboxapp|BIDUBrowser/i.test(ua);
                if (!isChromium) {
                    var t = document.querySelector('.install-banner__title');
                    if (t) t.textContent = '下载珞珈智行 APP';
                    showBanner('安卓安装包仅 49KB · 下载后点开安装，无需应用商店');
                    if (btn) btn.textContent = '下载 APP';
                }
            }, 3500);
        }
    });
})();
