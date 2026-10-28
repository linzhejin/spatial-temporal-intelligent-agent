const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const routeState = require('../../static/js/route-state.js');

// Run the real app control flow. Only DOM/map/network boundaries are replaced;
// exposing lexical functions here avoids shipping test hooks in the application.
function harness(storage = new Map()) {
    const bubbles = [], rendered = [], candidateRenders = [], comparisonRenders = [], requests = [], userBubbles = [];
    let clears = 0, stopped = 0, locationCalls = 0, reloads = 0;
    const localStorage = {
        getItem: k => storage.get(k) ?? null,
        setItem: (k,v) => storage.set(k,v),
        removeItem: k => storage.delete(k),
    };
    const noop = () => {};
    const element = () => ({ hidden: false, style: {}, children: [], innerHTML: '', scrollTop: 0, clientHeight: 100, scrollHeight: 100,
        scrollTo: noop, classList: { add: noop, remove: noop, toggle: noop },
        setAttribute: noop,
        querySelectorAll: () => [],
        querySelector(selector) { return this.children.find(child => selector.includes(child.dataset && child.dataset.action)) || null; },
        appendChild(child) { this.children.push(child); } });
    const elements = new Map();
    const context = vm.createContext({
        console, AbortController, setTimeout, clearTimeout, setInterval, clearInterval,
        localStorage, navigator: { userAgent: 'desktop', platform: 'Win32' },
        window: { localStorage, WHURouteState: routeState, location: { reload: () => { reloads++; } } },
        document: { readyState: 'loading', body: { classList: { add: noop, remove: noop, toggle: noop } }, addEventListener: noop, querySelectorAll: () => [],
            createElement: tag => ({ tagName: tag.toUpperCase(), dataset: {}, children: [], textContent: '', disabled: false,
                addEventListener(name, fn) { this['on' + name] = fn; },
                appendChild(child) { this.children.push(child); },
                querySelectorAll: () => [] }), getElementById: id => {
            if (!elements.has(id)) elements.set(id, element()); return elements.get(id);
        } },
        boundaries: {
            noop, showUserBubble: q => userBubbles.push(q),
            showChatBubble: (q,text) => { const b = { text }; bubbles.push(b); return b; },
            updateChatBubble: (b,text) => { b.text = text; },
            clearRouteResult: () => { clears++; stopped++; }, clearMap: () => { clears++; },
            stopNavigation: () => { stopped++; },
            renderRoute: r => { rendered.push(r); context.app.state.routeStore.replace(r.route_state || null); },
            renderCandidateCards: (candidates, taskId, revision) => candidateRenders.push({ candidates, taskId, revision }),
            renderRouteComparison: comparison => comparisonRenders.push(comparison),
            ensureUserLocation: async () => {
                locationCalls++;
                const fix = { lng: 114.3, lat: 30.5, accuracy: 35 };
                context.app.state.userLocation = fix;
                context.app.state._dispFix = {
                    lng: fix.lng, lat: fix.lat, acc: fix.accuracy, ts: Date.now(), provisional: false,
                };
                return fix;
            },
            apiRequest: async (url, body) => { requests.push({ url, body }); return context.reply; },
            logRequest: (url, body) => requests.push({ url, body }),
        },
        fetch: async () => ({ ok: true, json: async () => ({ data: {} }) }),
    });
    const source = fs.readFileSync(require.resolve('../../static/js/app.js'), 'utf8');
    const marker = "    if (document.readyState === 'loading') {";
    const end = source.indexOf(marker);
    assert.ok(end > 0);
    vm.runInContext(source.slice(0,end) + `
        var realApiRequest = apiRequest;
        var realRenderRoute = renderRoute;
        var realRenderClarifyOptions = renderClarifyOptions;
        var realShowSuggestions = showSuggestions;
        [${['hideError','startLoadingMessages','hideWelcomeElements','stopLoadingMessages','hideLoading',
            'renderClarifyOptions','renderCandidateCards','renderRouteComparison','renderRouteComparisonCard','syncModeFromServer','showResults','trackRouteShown','showTopBanner',
            'applyProductMode','loadTravelMode','bindEvents','syncTravelModeUI','syncRouteStrategyUI',
            'startRoadConditionVersionMonitor',
            'showWelcomeHint','initMap','consumeMobileHandoff'].map(n => JSON.stringify(n)).join(',')}]
            .forEach(function (name) { eval(name + ' = boundaries.noop'); });
        showUserBubble = boundaries.showUserBubble;
        showChatBubble = boundaries.showChatBubble;
        updateChatBubble = boundaries.updateChatBubble;
        clearRouteResult = boundaries.clearRouteResult;
        clearMap = boundaries.clearMap;
        stopNavigation = boundaries.stopNavigation;
        renderRoute = boundaries.renderRoute;
        renderCandidateCards = boundaries.renderCandidateCards;
        renderRouteComparisonCard = boundaries.renderRouteComparison;
        ensureUserLocation = boundaries.ensureUserLocation;
        apiRequest = boundaries.apiRequest;
        var realSubmitQueuedMessage = submitQueuedMessage;
        globalThis.app = { state: state, init: init, submit: handleNlSubmit, reset: handleReset,
            candidateRouteQuery: candidateRouteQuery,
            buildRouteSummary: buildRouteSummary,
            renderRouteDirect: realRenderRoute,
            renderClarifyOptions: realRenderClarifyOptions,
            showSuggestions: realShowSuggestions,
            globalKeydown: handleGlobalKeydown,
            load: loadContext, save: saveContext, request: realApiRequest, queued: submitQueuedMessage,
            removeMapPoint: removeMapPoint,
            bindMapPointRemoval: bindMapPointRemoval,
            setMapPoint: setMapPoint,
            refreshPage: refreshPage,
            setTravelMode: setTravelMode,
            checkRoadConditionVersion: checkRouteConditionVersion,
            loadRoadConditions: loadAndRenderRoadConditions,
            submitRoadReport: handleRoadReportSubmit,
            roadConditionEffect: getRoadConditionModeEffect,
            roadConditionEffectLabel: describeRoadConditionModeEffect,
            roadConditionDisplayStyle: getRoadConditionDisplayStyle,
            setApi: function (fn) {
                submitQueuedMessage = function (query, body, bubble, epoch) {
                    var payload = Object.assign({ query: query }, body || {});
                    boundaries.logRequest('/api/chat', payload);
                    return fn('/api/chat', payload, bubble, epoch);
                };
            },
            setQueuedApi: function (fn) { apiRequest = fn; submitQueuedMessage = realSubmitQueuedMessage; },
            setRawApi: function (fn) { apiRequest = fn; },
            setFocus: function (fn) { focusPoiOnMap = fn; },
            focus: focusPoiOnMap };
    })();`, context);
    context.app.state.productMode = 'planning';
    return { ...context.app, context, bubbles, rendered, candidateRenders, comparisonRenders, requests, userBubbles, storage,
        clearCount: () => clears, stopCount: () => stopped, locationCount: () => locationCalls,
        reloadCount: () => reloads };
}
const existingRoute = () => ({ route_id: 'existing', start: { name: '星湖园' }, end: { name: '科技门' }, travel_mode: 'walk' });
const routeResult = id => ({ task_type: 'path_planning', response_kind: 'route', recommended: [[1,2],[2,3]], route_state: { ...existingRoute(), route_id: id }, explanation: id });
function deferred() { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return { promise, resolve, reject }; }

test('an over-time shortest reference is not drawn as a selectable comparison', () => {
    const h = harness();
    const lines = [];
    h.context.L = { polyline: (coords, style) => {
        const line = { coords, style, addTo: () => line,
            getBounds: () => ({ extend() {} }) };
        lines.push(line);
        return line;
    } };
    h.state.map = { fitBounds() {} };
    h.state.poiMarkers = [];
    h.renderRouteDirect({
        recommended: [{ lng: 114.3, lat: 30.5 }, { lng: 114.31, lat: 30.51 }],
        shortest: [{ lng: 114.3, lat: 30.5 }, { lng: 114.32, lat: 30.52 }],
        shortest_baseline_feasible: false, mode: 'walk', pois: [],
    });
    assert.equal(lines.length, 1);
    assert.equal(h.state.shortestLine, null);
});

test('R starts a clean conversation without reloading the page', () => {
    const h = harness();
    h.state.conversationHistory = [{ role: 'user', content: '去东湖' }];
    h.state.serverConversationId = 'server-c1';
    const removed = [];
    const startMarker = { id: 'start' }, viaMarker = { id: 'via' }, endMarker = { id: 'end' };
    h.state.map = { removeLayer: marker => removed.push(marker), getContainer: () => ({ style: {} }), setView() {} };
    h.state.mapPoints = { start: { lng: 114.3, lat: 30.5 }, via: [{ lng: 114.31, lat: 30.51 }], end: { lng: 114.32, lat: 30.52 } };
    h.state.pointMarkers = { start: startMarker, via: [viaMarker], end: endMarker };
    let prevented = false;

    h.globalKeydown({ key: 'r', target: { tagName: 'BODY' }, preventDefault() { prevented = true; } });

    assert.equal(h.reloadCount(), 0);
    assert.equal(h.clearCount(), 1);
    assert.equal(h.state.serverConversationId, null);
    assert.equal(h.state.conversationHistory.length, 0);
    assert.equal(h.storage.has('whu_walker:server_conversation'), false);
    assert.equal(h.storage.has('whu_walker:explicit_reset_session'), false);
    const activeSessionId = h.storage.get('whu_walker:active_session');
    assert.ok(activeSessionId);
    const savedContext = JSON.parse(h.storage.get('whu_walker:context:' + activeSessionId));
    assert.deepEqual(savedContext.history, []);
    assert.equal(prevented, true);
    assert.equal(h.state.mapPoints.start, null);
    assert.equal(h.state.mapPoints.via.length, 0);
    assert.equal(h.state.mapPoints.end, null);
    assert.equal(h.state.pointMarkers.start, null);
    assert.equal(h.state.pointMarkers.via.length, 0);
    assert.equal(h.state.pointMarkers.end, null);
    assert.deepEqual(removed, [startMarker, viaMarker, endMarker]);
});

test('explicit refresh clears old history but later browser reload restores only the new conversation', async () => {
    const storage = new Map([
        ['whu_walker:active_session', 'sess_previous'],
        ['whu_walker:context:sess_previous', JSON.stringify({
            history: [{ role: 'user', content: '旧聊天' }], routeState: existingRoute(),
        })],
        ['whu_walker:server_conversation', 'server-previous'],
    ]);
    const first = harness(storage);
    first.setQueuedApi(async url => {
        if (url === '/api/conversations/server-previous') return {
            conversation_id: 'server-previous',
            messages: [{ task_id: 'old-task', role: 'user', content: '旧聊天', seq: 1 }],
            tasks: [], active_runs: [],
        };
        throw new Error('unexpected request ' + url);
    });
    await first.init();
    assert.equal(first.state.conversationHistory[0].content, '旧聊天');

    first.refreshPage();
    const freshSessionId = storage.get('whu_walker:active_session');
    assert.ok(freshSessionId && freshSessionId !== 'sess_previous');
    assert.equal(storage.has('whu_walker:explicit_reset_session'), false);
    assert.equal(first.state.conversationHistory.length, 0);
    assert.equal(storage.has('whu_walker:server_conversation'), false);

    first.state.serverConversationId = 'server-new';
    storage.set('whu_walker:server_conversation:' + freshSessionId, 'server-new');
    first.state.conversationHistory = [
        { role: 'user', content: '新聊天' },
        { role: 'assistant', content: '新回复' },
    ];
    first.save();

    const afterReload = harness(storage);
    const restoreCalls = [];
    afterReload.setQueuedApi(async url => {
        restoreCalls.push(url);
        if (url === '/api/conversations/server-new') return {
            conversation_id: 'server-new',
            messages: [
                { task_id: 'new-task', role: 'user', content: '新聊天', seq: 1 },
                { task_id: 'new-task', role: 'assistant', content: '新回复', seq: 2 },
            ],
            tasks: [], active_runs: [],
        };
        throw new Error('unexpected request ' + url);
    });
    await afterReload.init();

    assert.deepEqual(Array.from(afterReload.state.conversationHistory, message => message.content), ['新聊天', '新回复']);
    assert.equal(afterReload.state.serverConversationId, 'server-new');
    assert.deepEqual(restoreCalls, ['/api/conversations/server-new']);
    assert.equal(storage.has('whu_walker:explicit_reset_session'), false);
});

test('explicit refresh removes the already-rendered chat bubbles from the current page', () => {
    const h = harness();
    const removed = [];
    const chatContent = h.context.document.getElementById('chat-content');
    chatContent.querySelectorAll = selector => {
        assert.equal(selector, '.chat-bubble-row');
        return [
            { remove() { removed.push('user'); } },
            { remove() { removed.push('assistant'); } },
        ];
    };

    h.refreshPage();

    assert.deepEqual(removed, ['user', 'assistant']);
    assert.equal(h.state.conversationHistory.length, 0);
    assert.equal(h.state.serverConversationId, null);
});

test('each selected map point can be removed independently, including a via point', () => {
    const h = harness();
    const removed = [];
    const markers = [{ id: 'via-1' }, { id: 'via-2' }, { id: 'via-3' }];
    const startMarker = { id: 'start' }, endMarker = { id: 'end' };
    h.state.map = { removeLayer: marker => removed.push(marker), closePopup() {} };
    h.state.mapPoints = {
        start: { lng: 114.3, lat: 30.5 },
        via: [{ lng: 1, lat: 1 }, { lng: 2, lat: 2 }, { lng: 3, lat: 3 }],
        end: { lng: 114.4, lat: 30.6 },
    };
    h.state.pointMarkers = { start: startMarker, via: markers.slice(), end: endMarker };

    assert.equal(h.removeMapPoint('via', 1), true);
    assert.deepEqual(Array.from(h.state.mapPoints.via, point => point.lng), [1, 3]);
    assert.deepEqual(Array.from(h.state.pointMarkers.via, marker => marker.id), ['via-1', 'via-3']);
    assert.deepEqual(removed, [markers[1]]);
    assert.equal(h.removeMapPoint('start'), true);
    assert.equal(h.removeMapPoint('end'), true);
    assert.equal(h.state.mapPoints.start, null);
    assert.equal(h.state.mapPoints.end, null);
    assert.deepEqual(removed, [markers[1], startMarker, endMarker]);
});

test('map marker popup wires its delete button to that exact selected point', () => {
    const h = harness();
    const removed = [];
    const marker = {
        bindPopup(html) { this.popupHtml = html; },
        on(eventName, handler) { this.popupOpen = handler; },
    };
    h.state.map = { removeLayer: item => removed.push(item), closePopup() {} };
    h.state.mapPoints.start = { lng: 114.3, lat: 30.5 };
    h.state.pointMarkers.start = marker;
    h.bindMapPointRemoval(marker, 'start', '起点');
    let onDelete;
    const button = { addEventListener: (_name, handler) => { onDelete = handler; } };

    assert.match(marker.popupHtml, /删除此起点/);
    marker.popupOpen({ popup: { getElement: () => ({ querySelector: () => button }) } });
    onDelete({ preventDefault() {}, stopPropagation() {} });

    assert.equal(h.state.mapPoints.start, null);
    assert.deepEqual(removed, [marker]);
});

test('manual map markers do not render tooltip label boxes', () => {
    const h = harness();
    const markers = [];
    h.state.map = { removeLayer() {}, closePopup() {} };
    h.context.L = {
        divIcon: options => options,
        marker: (position, options) => {
            const marker = {
                position,
                options,
                tooltipCalls: [],
                addTo() { return this; },
                bindTooltip(...args) { this.tooltipCalls.push(args); return this; },
                bindPopup(content) { this.popupContent = content; return this; },
                on() { return this; },
            };
            markers.push(marker);
            return marker;
        },
    };

    h.setMapPoint('start', 114.3, 30.5, 114.3, 30.5);
    h.setMapPoint('via', 114.31, 30.51, 114.31, 30.51);
    h.setMapPoint('end', 114.32, 30.52, 114.32, 30.52);

    assert.equal(markers.length, 3);
    assert.deepEqual(markers.map(marker => marker.tooltipCalls.length), [0, 0, 0]);
    assert.deepEqual(markers.map(marker => marker.options.icon.className), ['whu-div-icon', 'whu-div-icon', 'whu-div-icon']);
    assert.ok(markers.every(marker => /删除此/.test(marker.popupContent)));
    const css = fs.readFileSync(require.resolve('../../static/css/style.css'), 'utf8');
    assert.match(css, /\.leaflet-marker-icon\.whu-div-icon\s*\{[^}]*background:\s*transparent\s*!important;[^}]*border:\s*0\s*!important;/s);
});

test('deleting a via point replans with the remaining selected points', async () => {
    const h = harness();
    h.state.map = { removeLayer() {}, closePopup() {} };
    h.state.mapPoints = {
        start: { lng: 114.3, lat: 30.5 },
        via: [{ lng: 114.31, lat: 30.51 }, { lng: 114.32, lat: 30.52 }],
        end: { lng: 114.33, lat: 30.53 },
    };
    h.state.pointMarkers = { start: {}, via: [{}, {}], end: {} };
    h.state.mapPointRouteActive = true;
    let sent;
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '收到' }; });

    h.removeMapPoint('via', 0);
    await new Promise(resolve => setTimeout(resolve, 0));

    assert.deepEqual(Array.from(sent.coord_waypoints, point => [point.lng, point.lat]), [[114.32, 30.52]]);
});

test('removing a point while its route request is pending prevents the stale route from rendering', async () => {
    const h = harness();
    const pending = deferred();
    h.state.mapPoints.start = { lng: 114.31, lat: 30.51 };
    h.state.pointMarkers.start = {};
    h.state.map = { removeLayer() {}, closePopup() {} };
    h.setApi(() => pending.promise);

    const request = h.submit('去最近的食堂');
    await new Promise(resolve => setTimeout(resolve, 0));
    h.removeMapPoint('start');
    pending.resolve(routeResult('stale-manual-start'));
    await request;

    assert.equal(h.rendered.length, 0);
    assert.match(h.bubbles[0].text, /地图选点已更新/);
});

test('a natural-language route request receives manually selected start and via coordinates', async () => {
    const h = harness();
    h.state.userLocation = { lng: 114.39, lat: 30.59 };
    h.state.mapPoints.start = { lng: 114.31, lat: 30.51 };
    h.state.mapPoints.via = [{ lng: 114.32, lat: 30.52 }];
    let sent;
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '好的' }; });

    await h.submit('去最近的食堂');

    assert.deepEqual({ lng: sent.coord_start.lng, lat: sent.coord_start.lat, name: sent.coord_start.name },
        { lng: 114.31, lat: 30.51, name: '地图起点' });
    assert.deepEqual(Array.from(sent.coord_waypoints, point => [point.lng, point.lat]), [[114.32, 30.52]]);
});

test('nearby recommendation does not reuse stale, coarse, or provisional GPS as its origin', async () => {
    for (const fix of [
        { ts: Date.now() - 121000, acc: 20, provisional: false },
        { ts: Date.now(), acc: 101, provisional: false },
        { ts: Date.now(), acc: 20, provisional: true },
    ]) {
        const h = harness();
        h.state.userLocation = { lng: 114.3, lat: 30.5, accuracy: fix.acc };
        h.state._dispFix = { lat: 30.5, lng: 114.3, ...fix };
        let sent;
        h.setApi(async (_url, body) => {
            sent = body;
            return { task_type: 'chat', reply: '请先告诉我从哪里出发。' };
        });

        await h.submit('我想吃美食');

        assert.equal(sent.coord_start, undefined);
    }
});

test('nearby recommendation may use only a recent accurate GPS fix as its origin', async () => {
    const h = harness();
    h.state.userLocation = { lng: 114.3, lat: 30.5, accuracy: 35 };
    h.state._dispFix = { lat: 30.5, lng: 114.3, acc: 35, ts: Date.now(), provisional: false };
    let sent;
    h.setApi(async (_url, body) => {
        sent = body;
        return { task_type: 'chat', reply: '按附近地点为你推荐。' };
    });

    await h.submit('我想吃美食');

    assert.deepEqual(
        { lng: sent.coord_start.lng, lat: sent.coord_start.lat, name: sent.coord_start.name },
        { lng: 114.3, lat: 30.5, name: '我的位置' },
    );
});

test('clicking a candidate keeps a typed origin while GPS and map origins stay attached by state', () => {
    const h = harness();
    assert.equal(h.candidateRouteQuery({ name: '工学部清真食堂', recommendation_start_name: '工学部' }),
        '从工学部到工学部清真食堂');
    assert.equal(h.candidateRouteQuery({ name: '附近食堂', recommendation_start_name: '我的位置' }),
        '去附近食堂');
    assert.equal(h.candidateRouteQuery({ name: '附近食堂' }), '去附近食堂');
});

test('R shortcut and header button both start a new conversation', () => {
    const html = fs.readFileSync(require.resolve('../../static/index.html'), 'utf8');
    const app = fs.readFileSync(require.resolve('../../static/js/app.js'), 'utf8');

    assert.match(html, /data-kbd="refresh">R<\/kbd><span class="kbd-label">刷新<\/span>/);
    assert.match(html, /↻ 清空路线与对话<\/span><kbd>R<\/kbd>/);
    assert.match(html, /id="reset-btn"[^>]*aria-label="刷新路线与对话"[^>]*title="清空当前路线和聊天，开始新一轮规划"[^>]*>刷新<\/button>/);
    assert.match(app, /resetBtn\.addEventListener\('click', refreshPage\)/);
    assert.match(app, /function refreshPage\(\)\s*\{[\s\S]*?beginExplicitRefreshSession\(\);\s*handleReset\(\);\s*state\.restoredActiveRuns = \[\];\s*\}/);
    assert.doesNotMatch(app.match(/function refreshPage\(\)\s*\{[\s\S]*?\n    \}/)[0], /location\.reload/);
    assert.match(app, /删除此起点|删除此途经点|删除此终点/);
    assert.doesNotMatch(html, /data-kbd="reset"/);
});

test('browser reload can restore the same local session and route context', () => {
    const a = harness(); a.init(); a.state.conversationHistory = [{role:'user',content:'去科技门'}];
    a.state.routeStore.replace(existingRoute());
    a.state.mapPoints.start = { lng: 114.31, lat: 30.51 };
    a.state.mapPoints.via = [{ lng: 114.32, lat: 30.52 }];
    a.save();
    const b = harness(a.storage); b.init();
    assert.equal(b.state.sessionId, a.state.sessionId);
    assert.equal(b.state.routeStore.current().route_id, 'existing');
    assert.equal(b.state.conversationHistory[0].content, '去科技门');
    assert.equal(b.state.mapPoints.start, null);
    assert.equal(b.state.mapPoints.via.length, 0);
});
test('server transcript restores the trusted conversation after local messages are absent', async () => {
    const storage = new Map([['whu_walker:server_conversation', 'server-c1']]);
    const h = harness(storage);
    h.setQueuedApi(async (url, body, method) => {
        assert.equal(url, '/api/conversations/server-c1');
        assert.equal(method, 'GET');
        return { conversation_id: 'server-c1', messages: [
            { task_id: 't1', role: 'user', content: '去樱顶', seq: 1 },
            { task_id: 't1', role: 'assistant', content: '从珞珈门过去约十分钟。', seq: 2,
                metadata: { task_type: 'chat' } },
        ] };
    });
    await h.init();
    assert.deepEqual(Array.from(h.state.conversationHistory, m => [m.role, m.content]), [
        ['user', '去樱顶'], ['assistant', '从珞珈门过去约十分钟。'],
    ]);
    assert.equal(h.state.serverConversationId, 'server-c1');
});
for (const result of [
    {task_type:'chat',reply:'今天晴'}, {response_kind:'clarify',clarify:{question:'哪个图书馆？', options:[]}},
    {response_kind:'candidates',candidates:[],message:'查到了'}, {task_type:'help',message:'帮助'},
    {task_type:'poi_query',poi:null,message:'地点信息'}, {task_type:'path_planning', recommended:[]},
]) test('keeps displayed route and navigation for ' + JSON.stringify(result), async () => {
    const h=harness(); h.state.routeStore.replace(existingRoute()); h.context.reply=result;
    await h.submit('问题'); assert.equal(h.clearCount(),0); assert.equal(h.stopCount(),0);
    assert.equal(h.state.routeStore.current().route_id,'existing');
    assert.ok(!h.bubbles[0].text.includes('正在为你'));
});
test('no-route result clears an old route while retaining the requested endpoints', async () => {
    const h = harness();
    h.setApi(async () => h.context.reply);
    h.state.routeStore.replace(existingRoute());
    h.state.latestRoute = existingRoute();
    h.context.reply = {
        task_type: 'chat', response_kind: 'no_route',
        reply: '当前路网下没有可通行路线。',
        route_state: {
            start: { name: '梅园综合楼' }, end: { name: '武汉大学图书馆' },
            travel_mode: 'walk', hard_constraints: { avoid_steps: true },
        },
    };
    await h.submit('从梅园综合楼到武汉大学图书馆，避开台阶');
    assert.equal(h.bubbles.at(-1).text, '当前路网下没有可通行路线。');
    assert.equal(h.state.routeStore.current(), null);
    assert.equal(h.state.latestRoute, null);
    assert.ok(h.clearCount() > 0);
    assert.equal(h.state.pendingIntent.start.name, '梅园综合楼');
    assert.equal(h.state.pendingIntent.end.name, '武汉大学图书馆');
});

test('older no-route reply cannot clear a newer displayed route', async () => {
    const h = harness();
    h.setApi(async () => h.context.reply);
    h.state.routeStore.replace(existingRoute());
    h.state.latestRouteTurnId = 2;
    h.context.reply = { task_type: 'chat', response_kind: 'no_route',
        reply: '旧任务无路。', route_state: {
            start: { name: '旧起点' }, end: { name: '旧终点' }, travel_mode: 'walk',
        } };
    await h.submit('旧任务');
    assert.equal(h.state.routeStore.current().route_id, 'existing');
    assert.equal(h.clearCount(), 0);
});

test('failed request keeps route and settles its bubble', async () => {
    const h=harness(); h.state.routeStore.replace(existingRoute()); h.setApi(async () => { throw Error('断网'); });
    await h.submit('查天气'); assert.equal(h.clearCount(),0); assert.equal(h.bubbles[0].text,'断网');
});
test('desktop route feedback gives one useful overview with endpoints, strategy, distance, time and stop count', async () => {
    const h = harness();
    h.setApi(async () => h.context.reply);
    h.context.reply = {
        task_type: 'path_planning', response_kind: 'route', recommended: [[1, 2], [3, 4]],
        route_kind: 'via', mode: 'walk', recommended_length_m: 5077, duration_min: 68,
        road_conditions_applied: 2,
        route_state: {
            ...existingRoute(), route_id: 'overview-route', route_kind: 'via',
            via: { type: 'multi', points: [{ name: '途经点一' }, { name: '途经点二' }, { name: '途经点三' }] },
            strategy: { name: 'shortest', source: 'button' },
        },
        pois: [{ name: '沿途地点一' }, { name: '沿途地点二' }, { name: '沿途地点三' }, { name: '沿途地点四' }],
        explanation: '已为你规划好步行路线，约 5.1 公里、68 分钟。',
    };

    await h.submit('从星湖园到科技门，经过三个地点');

    const reply = h.bubbles.at(-1).text;
    assert.match(reply, /星湖园.*科技门/);
    assert.match(reply, /步行/);
    assert.match(reply, /5\.1 公里/);
    assert.match(reply, /68 分钟/);
    assert.match(reply, /最短路径/);
    assert.match(reply, /3 个途经点/);
    assert.match(reply, /2 条.*路况/);
    assert.match(reply, /已核对 2 条当前生效路况信息/);
    assert.doesNotMatch(reply, /路况调整路线/);
    assert.match(reply, /4 处地点/);
    assert.doesNotMatch(reply, /沿途地点一|沿途地点二/);
    assert.doesNotMatch(reply, /已为你规划好步行路线/);
});
test('desktop tour feedback identifies the loop and omits generic filler', async () => {
    const h = harness();
    h.setApi(async () => h.context.reply);
    h.context.reply = {
        task_type: 'path_planning', response_kind: 'route', recommended: [[1, 2], [3, 4]],
        route_kind: 'tour', mode: 'walk', recommended_length_m: 6100, duration_min: 81,
        tour: { loop: true, ordered_pois: [{ name: '珞珈山' }, { name: '凌波门东湖观景点' }] },
        route_state: {
            ...existingRoute(), start: { name: '珞珈山' }, end: { name: '珞珈山' },
            route_id: 'tour-loop', route_kind: 'tour',
            tour: { loop: true, pois: [{ name: '珞珈山' }, { name: '凌波门东湖观景点' }] },
            strategy: { name: 'recommended' },
        },
        explanation: '已为你规划好从珞珈山到终点的步行路线。',
    };

    await h.submit('游览珞珈山和东湖');

    const reply = h.bubbles.at(-1).text;
    assert.match(reply, /环线/);
    assert.match(reply, /珞珈山/);
    assert.match(reply, /返回/);
    assert.doesNotMatch(reply, /路线特点：已为你规划好/);
});
test('a recommended nearest route also tells the user which places were ranked', async () => {
    const h = harness();
    h.setApi(async () => h.context.reply);
    h.context.reply = {
        ...routeResult('nearest-canteen'),
        recommended_length_m: 210, duration_min: 3, mode: 'walk',
        recommendation_candidates: [
            { poi_id: 'canteen_a', name: '食堂甲', distance_m: 210 },
            { poi_id: 'canteen_b', name: '食堂乙', distance_m: 340 },
            { poi_id: 'canteen_c', name: '食堂丙', distance_m: 510 },
        ],
    };

    await h.submit('推荐附近3个食堂，并规划到最近的一个');

    const reply = h.bubbles.at(-1).text;
    assert.match(reply, /食堂甲.*210.*食堂乙.*340.*食堂丙.*510/);
    assert.match(reply, /已规划到食堂甲/);
    assert.match(reply, /路线总览/);
});
test('route overview separates road-network length from estimated unverified endpoint connectors', () => {
    const h = harness();
    const summary = h.buildRouteSummary({
        start: { name: '武汉大学信息学部图书馆' },
        end: { name: '乒乓球馆' },
        mode: 'walk', recommended_length_m: 237, duration_min: 3,
        strategy: { name: 'shortest' },
        endpoint_access: {
            start: { snap_distance_m: 20.1 },
            end: { snap_distance_m: 16.8 },
        },
    });

    assert.match(summary, /路网段.*237/);
    assert.match(summary, /接驳.*直线.*37.*未核实/);
});
test('route overview states when route checks do not include field passability verification', () => {
    const h = harness();
    const summary = h.buildRouteSummary({
        start: { name: '信息学部' }, end: { name: '卓尔体育馆' },
        mode: 'walk', recommended_length_m: 640, duration_min: 9,
        strategy: { name: 'shortest' },
        route_verification: { status: 'pass', field_truth_verified: false },
    });

    assert.match(summary, /核对的是路网和路线约束/);
    assert.match(summary, /未实地确认道路当前可通行/);
});
test('route overview does not add an unverified-field caveat when field truth is confirmed', () => {
    const h = harness();
    const summary = h.buildRouteSummary({
        start: { name: '信息学部' }, end: { name: '卓尔体育馆' },
        mode: 'walk', recommended_length_m: 640, duration_min: 9,
        strategy: { name: 'shortest' },
        route_verification: { status: 'pass', field_truth_verified: true },
    });

    assert.doesNotMatch(summary, /未实地确认道路当前可通行/);
});
test('mobile route feedback keeps the route explanation and appends the field-passability limit once', async () => {
    const h = harness();
    h.state.productMode = 'navigation';
    h.setApi(async () => ({
        ...routeResult('mobile-field-limit'),
        explanation: '路线约 800 米，预计步行 10 分钟。',
        route_verification: { status: 'pass', field_truth_verified: false },
    }));

    await h.submit('从信息学部到图书馆');

    const reply = h.bubbles.at(-1).text;
    assert.match(reply, /路线约 800 米，预计步行 10 分钟/);
    assert.match(reply, /未实地确认道路当前可通行/);
    assert.equal((reply.match(/未实地确认道路当前可通行/g) || []).length, 1);
});
test('desktop route details are compact and use collapsed disclosures for long lists', () => {
    const html = fs.readFileSync(require.resolve('../../static/index.html'), 'utf8');
    const css = fs.readFileSync(require.resolve('../../static/css/style.css'), 'utf8');

    assert.match(html, /class="route-summary navigation-only"/);
    assert.match(html, /<details class="desktop-route-details" id="desktop-route-details">/);
    assert.match(html, /<details class="poi-section" id="poi-disclosure" hidden>/);
    assert.doesNotMatch(html, /路线生成过程|desktop-route-checks/);
    assert.ok(css.includes('body.is-planning-mode .explanation-box'));
    assert.ok(/\.desktop-route-details\s+\.desktop-route-steps\s*\{[^}]*max-height:\s*18\dpx/s.test(css));
});
test('both concurrent informational responses finish even out of order', async () => {
    const h=harness(), a=deferred(), b=deferred(); let count=0;
    h.setApi(() => ++count===1 ? a.promise : b.promise);
    const pa=h.submit('天气'), pb=h.submit('学校介绍');
    b.resolve({task_type:'chat',reply:'介绍'}); await pb;
    a.resolve({task_type:'chat',reply:'晴'}); await pa;
    assert.equal(h.bubbles[0].text,'晴'); assert.equal(h.bubbles[1].text,'介绍');
    assert.deepEqual(Array.from(h.state.conversationHistory, m=>m.content), ['天气','晴','学校介绍','介绍']);
});
test('late route cannot replace a newer route but gets a terminal response', async () => {
    const h=harness(), a=deferred(), b=deferred(); let count=0; h.setApi(()=>++count===1?a.promise:b.promise);
    const pa=h.submit('路线A'), pb=h.submit('路线B');
    b.resolve(routeResult('B')); await pb; a.resolve(routeResult('A')); await pa;
    assert.equal(h.rendered.length,1); assert.equal(h.state.routeStore.current().route_id,'B');
    assert.ok(!h.bubbles[0].text.includes('正在为你'));
});
test('switching travel mode replans from the committed route without dropping via points or constraints', async () => {
    const h = harness();
    const committed = {
        route_id: 'walk-route', route_kind: 'via', original_query: '从牌坊经樱顶到图书馆',
        start: { name: '牌坊', poi_id: 'gate-main' },
        via: [{ name: '樱顶', poi_id: 'poi-cherry-top' }],
        end: { name: '图书馆', poi_id: 'library' }, travel_mode: 'walk',
        hard_constraints: { slope: 'avoid', scenery: 'high' },
        strategy: { name: 'scenery', weights: { distance: 0.15, slope: 0.1, scenery: 0.75 } },
    };
    h.state.routeStore.replace(committed);
    h.state.travelMode = 'walk';
    const response = deferred();
    let requestBody;
    h.setQueuedApi(async (url, body) => {
        assert.equal(url, '/api/route/replan');
        requestBody = body;
        return response.promise;
    });

    h.setTravelMode('bike');
    assert.deepEqual(h.state.routeStore.current(), committed);
    assert.equal(h.state.travelMode, 'walk');
    response.resolve({
        recommended: [[1, 2], [3, 4]],
        route_state: { ...committed, route_id: 'bike-route', travel_mode: 'bike' },
        mode: 'bike', recommended_length_m: 1400,
    });
    await new Promise(resolve => setTimeout(resolve, 0));

    assert.deepEqual(requestBody.route_state, committed);
    assert.equal(JSON.stringify(requestBody.change), JSON.stringify({ travel_mode: 'bike' }));
    assert.equal(h.state.routeStore.current().route_id, 'bike-route');
    assert.deepEqual(h.state.routeStore.current().via, committed.via);
    assert.deepEqual(h.state.routeStore.current().hard_constraints, committed.hard_constraints);
    assert.equal(h.state.travelMode, 'bike');
});
test('failed travel-mode replan keeps the previous mode and complete route context', async () => {
    const h = harness();
    const committed = {
        route_id: 'walk-route', route_kind: 'via',
        start: { name: '牌坊' }, via: [{ name: '樱顶' }], end: { name: '图书馆' },
        travel_mode: 'walk', hard_constraints: { slope: 'avoid' },
        strategy: { name: 'flat' },
    };
    h.state.routeStore.replace(committed);
    h.state.travelMode = 'walk';
    h.setQueuedApi(async () => { throw Error('骑行道路暂不可达'); });

    h.setTravelMode('bike');
    await new Promise(resolve => setTimeout(resolve, 0));

    assert.deepEqual(h.state.routeStore.current(), committed);
    assert.equal(h.state.travelMode, 'walk');
    assert.equal(h.bubbles.length, 0);
});
test('new weather query does not silently discard an independent route result', async () => {
    const h=harness(), a=deferred(), b=deferred(); let count=0; h.setApi(()=>++count===1?a.promise:b.promise);
    const pa=h.submit('路线A'), pb=h.submit('天气');
    b.resolve({task_type:'chat',reply:'晴'}); await pb; a.resolve(routeResult('A')); await pa;
    assert.equal(h.rendered.length,1); assert.equal(h.bubbles[1].text,'晴');
});
test('start clarification retries GPS once in the same request without duplicate bubbles', async () => {
    const h=harness(); h.state.productMode='navigation'; let count=0;
    h.setApi(async()=> { if (++count>3) throw Error('recursion'); return { response_kind:'clarify', clarify:{question:'起点在哪？',options:[]} }; });
    await h.submit('去图书馆'); await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(count,2); assert.equal(h.locationCount(),1); assert.equal(h.bubbles.length,1); assert.equal(h.userBubbles.length,1);
});
test('automatic GPS answer continues the pending server task', async () => {
    const h = harness(); h.state.productMode = 'navigation';
    const sent = [];
    h.setApi(async (_url, body) => {
        sent.push(body);
        return sent.length === 1
            ? { response_kind: 'clarify', _task_id: 'original-task', _task_revision: 1,
                clarify: { question: '起点在哪？', options: [] } }
            : routeResult('gps-continued');
    });
    await h.submit('帮我规划去樱顶，顺便查天气');
    assert.equal(sent.length, 2);
    assert.equal(sent[1].continuation_task_id, 'original-task');
    assert.equal(sent[1].base_revision, 1);
    assert.deepEqual({ lng: sent[1].coord_start.lng, lat: sent[1].coord_start.lat },
        { lng: 114.3, lat: 30.5 });
});
test('PC clarification never asks for device location', async () => {
    const h=harness(); h.context.reply={response_kind:'clarify',clarify:{question:'起点在哪？',options:[]}};
    await h.submit('去图书馆'); assert.equal(h.locationCount(),0);
});
test('clarification answer continues its server task while an unrelated weather query does not', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-task';
    h.state.pendingServerTaskRevision = 4;
    h.state.pendingServerQuestion = '你想从哪里出发？';
    let sent;
    h.setApi(async (_url, body) => { sent = body; return routeResult('resumed'); });
    await h.submit('从樱顶出发');
    assert.equal(sent.continuation_task_id, 'pending-task');
    assert.equal(sent.base_revision, 4);
    assert.equal(h.state.pendingServerTaskId, null);

    h.state.pendingServerTaskId = 'pending-task-2';
    h.state.pendingServerTaskRevision = 1;
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '今天晴' }; });
    await h.submit('查一下天气');
    assert.equal(sent.continuation_task_id, undefined);
    assert.equal(h.state.pendingServerTaskId, 'pending-task-2');
});
test('free-form location answer continues the original pending task', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 3;
    h.state.pendingServerQuestion = '你想从哪里出发？';
    let sent;
    h.setApi(async (_url, body) => { sent = body; return routeResult('continued'); });
    await h.submit('我在信息学部');
    assert.equal(sent.continuation_task_id, 'pending-route');
    assert.equal(sent.base_revision, 3);
});
test('origin plus weather and anaphoric destination continues the pending route task', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 3;
    h.state.pendingServerQuestion = '你想从哪里出发？';
    let sent;
    h.setApi(async (_url, body) => { sent = body; return routeResult('continued'); });

    await h.submit('我在武汉大学科技门，查天气后再带我过去');

    assert.equal(sent.continuation_task_id, 'pending-route');
    assert.equal(sent.base_revision, 3);
});
test('free-form ball type answer continues the pending activity task', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'sports-task';
    h.state.pendingServerTaskRevision = 3;
    h.state.pendingServerQuestion = '想打什么球？';
    let sent;
    h.setApi(async (_url, body) => { sent = body; return routeResult('continued'); });

    await h.submit('橄榄球');

    assert.equal(sent.query, '橄榄球');
    assert.equal(sent.continuation_task_id, 'sports-task');
    assert.equal(sent.base_revision, 3);
});
test('ball clarification renders all five choices and leaves other ball types for free input', () => {
    const h = harness();
    h.renderClarifyOptions(['篮球', '羽毛球', '乒乓球', '足球', '其他球类'], 'sports-task', 3);
    const chat = h.context.document.getElementById('chat-content');
    const buttons = chat.children[0].children[0].children;
    assert.deepEqual(Array.from(buttons, button => button.textContent),
        ['篮球', '羽毛球', '乒乓球', '足球', '其他球类']);

    const input = h.context.document.getElementById('nl-input');
    let focused = false;
    input.focus = () => { focused = true; };
    buttons[4].onclick();

    assert.equal(focused, true);
    assert.match(input.placeholder, /补充球类/);
});
test('follow-up suggestions render from the agent result and submit the selected query', () => {
    const h = harness();
    let submitted = 0;
    const submitButton = h.context.document.getElementById('submit-btn');
    submitButton.click = () => { submitted++; };

    h.showSuggestions({ suggestions: [{ label: '附近的食堂', query: '找附近食堂' }] });

    const area = h.context.document.getElementById('suggestions-area');
    const chips = h.context.document.getElementById('suggestions-chips');
    assert.equal(area.hidden, false);
    assert.equal(chips.children.length, 1);
    assert.equal(chips.children[0].textContent, '附近的食堂');
    assert.equal(chips.children[0].title, '找附近食堂');
    chips.children[0].onclick();
    assert.equal(h.context.document.getElementById('nl-input').value, '找附近食堂');
    assert.equal(submitted, 1);
});
test('chat follow-up suggestions are placed directly after the reply bubble', () => {
    const h = harness();
    const replyBubble = { after(node) { this.nextNode = node; } };

    h.showSuggestions({ suggestions: [{ label: '查附近道路', query: '查看附近道路管制' }] }, replyBubble);

    assert.equal(replyBubble.nextNode, h.context.document.getElementById('suggestions-area'));
});
test('chat responses render tool-provided follow-up suggestions', async () => {
    const h = harness();
    h.setApi(async () => ({
        task_type: 'chat', reply: '有几处路况信息。',
        suggestions: [{ label: '查附近道路', query: '查看附近道路管制' }],
    }));

    await h.submit('附近有什么路况？');

    const chips = h.context.document.getElementById('suggestions-chips');
    assert.equal(chips.children.length, 1);
    assert.equal(chips.children[0].textContent, '查附近道路');
    assert.equal(chips.children[0].title, '查看附近道路管制');
});
test('route comparison responses render structured data without replacing the current route', async () => {
    const h = harness();
    h.state.routeStore.replace(existingRoute());
    const comparison = {
        alternatives: [{ strategy: 'shortest', label: '最短路径', distance_m: 500 }],
        unavailable: [{ strategy: 'scenery', label: '风景优先', reason: '缺少数据' }],
    };
    h.setApi(async () => ({ task_type: 'chat', response_kind: 'route_comparison',
        reply: '比较了可用路线。', route_comparison: comparison }));

    await h.submit('比较三种走法');

    assert.equal(h.comparisonRenders.length, 1);
    assert.equal(h.comparisonRenders[0], comparison);
    assert.equal(h.state.routeStore.current().route_id, 'existing');
    assert.equal(h.clearCount(), 0);
});
test('chat responses without suggestions hide stale route suggestion chips', async () => {
    const h = harness();
    h.showSuggestions({ suggestions: [{ label: '看风景路线', query: '规划风景优先路线' }] });
    h.setApi(async () => ({ task_type: 'chat', reply: '今天天气晴。' }));

    await h.submit('今天天气怎么样？');

    assert.equal(h.context.document.getElementById('suggestions-area').hidden, true);
});
test('independent POI search does not get attached to a pending route clarification', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 2;
    h.state.pendingServerQuestion = '你想从哪里出发？';
    let sent;
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '找到附近咖啡店' }; });
    await h.submit('找一下樱顶附近咖啡');
    assert.equal(sent.continuation_task_id, undefined);
    assert.equal(h.state.pendingServerTaskId, 'pending-route');
});
test('independent route request does not consume an earlier clarification', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 2;
    let sent;
    h.setApi(async (_url, body) => { sent = body; return routeResult('new-route'); });
    await h.submit('怎么去樱顶');
    assert.equal(sent.continuation_task_id, undefined);
    assert.equal(h.state.pendingServerTaskId, 'pending-route');
});
test('late older clarification cannot replace the newer pending clarification', async () => {
    const h = harness(), older = deferred(), newer = deferred();
    let count = 0;
    h.setApi(() => ++count === 1 ? older.promise : newer.promise);
    const first = h.submit('规划第一条路线');
    const second = h.submit('规划第二条路线');
    newer.resolve({ response_kind: 'clarify', _task_id: 'new-task', _task_revision: 3,
        clarify: { question: '第二条路线的终点是哪里？', options: [] } });
    await second;
    older.resolve({ response_kind: 'clarify', _task_id: 'old-task', _task_revision: 1,
        clarify: { question: '第一条路线的终点是哪里？', options: [] } });
    await first;
    assert.equal(h.state.pendingServerTaskId, 'new-task');
    assert.equal(h.state.pendingServerTaskRevision, 3);
    assert.equal(h.state.pendingServerQuestion, '第二条路线的终点是哪里？');
});
test('a short place-name answer continues a pending route clarification', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 2;
    h.state.pendingServerQuestion = '你想从哪里出发？';
    let sent;
    h.setRawApi(async url => url.startsWith('/api/pois?keyword=') ? { pois: [{ name: '樱顶', aliases: ['樱花顶'] }] } : {});
    h.setApi(async (_url, body) => { sent = body; return routeResult('resumed'); });
    await h.submit('樱顶');
    assert.equal(sent.continuation_task_id, 'pending-route');
});
test('a short independent question does not continue a pending route clarification', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 2;
    let sent;
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '我来查活动信息。' }; });
    await h.submit('本周有哪些活动');
    assert.equal(sent.continuation_task_id, undefined);
    assert.equal(h.state.pendingServerTaskId, 'pending-route');
});
test('a known POI is not attached when the pending clarification asks for travel mode', async () => {
    const h = harness();
    h.state.pendingServerTaskId = 'pending-route';
    h.state.pendingServerTaskRevision = 2;
    h.state.pendingServerQuestion = '你想步行还是骑行？';
    let sent;
    h.setRawApi(async () => ({ pois: [{ name: '樱顶' }] }));
    h.setApi(async (_url, body) => { sent = body; return { task_type: 'chat', reply: '请说明出行方式。' }; });
    await h.submit('樱顶');
    assert.equal(sent.continuation_task_id, undefined);
});
test('reset during server conversation restore cannot restore old history or resume its runs', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const restore = deferred();
    const calls = [];
    h.setQueuedApi(async (url) => {
        calls.push(url);
        return restore.promise;
    });
    const initializing = h.init();
    h.reset();
    restore.resolve({
        conversation_id: 'server-c1',
        messages: [{ task_id: 'old-task', role: 'assistant', content: '旧回复', seq: 2 }],
        tasks: [],
        active_runs: [{ run_id: 'old-run', task_id: 'old-task', query: '旧路线', message_seq: 1 }],
    });
    await initializing;
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.state.serverConversationId, null);
    assert.equal(h.state.conversationHistory.length, 0);
    assert.equal(h.state.routeStore.current(), null);
    assert.equal(calls.some(url => url.includes('/api/runs/old-run?')), false);
});
test('a new submission during restore is not overwritten by the restored transcript', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const restore = deferred(), reply = deferred();
    h.setQueuedApi(async (url) => {
        if (url.startsWith('/api/conversations/')) return restore.promise;
        throw new Error('unexpected restore request ' + url);
    });
    h.setApi(() => reply.promise);
    const initializing = h.init();
    const submitting = h.submit('从玉兰二门去图书馆');
    restore.resolve({
        conversation_id: 'server-c1',
        messages: [{ task_id: 'old-task', role: 'assistant', content: '旧历史', seq: 2 }],
        tasks: [], active_runs: [],
    });
    await initializing;
    reply.resolve(routeResult('new-route'));
    await submitting;
    assert.ok(h.state.conversationHistory.some(message => message.content === '从玉兰二门去图书馆'));
    assert.equal(h.state.conversationHistory.some(message => message.content === '旧历史'), false);
});
test('reset invalidates an in-flight response before it can repopulate context', async () => {
    const h=harness(), a=deferred(); h.setApi(()=>a.promise); const p=h.submit('路线A');
    h.reset(); a.resolve(routeResult('A')); await p;
    assert.equal(h.rendered.length,0); assert.equal(h.state.conversationHistory.length,0);
});
test('reset cancels owned queued runs and starts a clean server conversation next time', () => {
    const h = harness(new Map());
    const calls = [];
    h.state.serverConversationId = 'server-c2';
    h.storage.set('whu_walker:server_conversation', 'server-c2');
    h.state.activeRuns.r2 = { conversationId: 'server-c2', epoch: 0 };
    h.setQueuedApi(async (url, body) => { calls.push({ url, body }); return {}; });
    h.reset();
    assert.equal(h.state.serverConversationId, null);
    assert.equal(h.storage.has('whu_walker:server_conversation'), false);
    assert.equal(calls[0].url, '/api/runs/r2/cancel');
    assert.equal(calls[0].body.conversation_id, 'server-c2');
});
test('request timeout aborts fetch and produces a specific terminal error', async () => {
    const h=harness(); let signal;
    h.context.fetch=(_url,opts)=>{signal=opts.signal; return new Promise(()=>{});};
    const result = await Promise.race([
        h.request('/api/chat',{},'POST',10).then(()=> 'unexpected', e=>e.code),
        new Promise(resolve=>setTimeout(()=>resolve('no-timeout'),100)),
    ]);
    assert.equal(result,'request_timeout'); assert.equal(signal.aborted,true);
});

test('durable submission stores the accepted query and polls independent progress to its result', async () => {
    const h = harness();
    const calls = [];
    let polls = 0;
    h.setQueuedApi(async (url, body, method) => {
        calls.push({ url, body, method });
        if (url === '/api/conversations') return { conversation_id: 'server-session-1' };
        if (url.endsWith('/messages')) return { run_id: 'run-1', status: 'queued' };
        if (url.includes('/api/runs/run-1?')) {
            polls++;
            if (polls === 1) return { status: 'running', events: [
                { seq: 1, event_type: 'node_started', payload: { label: '正在识别需求' } },
            ] };
            return { status: 'completed', events: [
                { seq: 2, event_type: 'node_completed', payload: { label: '完成' } },
            ], result: routeResult('durable') };
        }
        throw new Error('unexpected request ' + url);
    });
    const bubble = { text: 'waiting' };
    const result = await h.queued('从地图起点到地图终点', {
        travel_mode: 'bike', coord_start: { lng: 114.36, lat: 30.53, name: '地图起点' },
    }, bubble, 0);
    assert.equal(result.route_state.route_id, 'durable');
    assert.equal(bubble.text, '🌸 正在识别需求…');
    assert.equal(calls[0].url, '/api/conversations');
    assert.equal(calls[1].body.travel_mode, 'bike');
    assert.equal(calls[1].body.coord_start.lng, 114.36);
    assert.ok(calls[1].body.request_id);
    assert.equal(calls[2].method, 'GET');
    assert.equal(h.state.activeRuns['run-1'], undefined);
});

test('polling recovers from a temporary network failure without losing the accepted task', async () => {
    const h = harness();
    let polls = 0;
    h.setQueuedApi(async url => {
        if (url === '/api/conversations') return { conversation_id: 'server-session-retry' };
        if (url.endsWith('/messages')) return { run_id: 'run-retry' };
        if (url.includes('/api/runs/run-retry?')) {
            polls++;
            if (polls === 1) throw new Error('网络请求失败，请检查网络连接');
            return { status: 'completed', events: [], result: routeResult('recovered') };
        }
        throw new Error('unexpected request ' + url);
    });
    const bubble = { text: 'waiting' };
    const result = await h.queued('经樱顶去图书馆', {}, bubble, 0);
    assert.equal(result.route_state.route_id, 'recovered');
    assert.equal(polls, 2);
    assert.equal(h.state.activeRuns['run-retry'], undefined);
});

test('a still-running task continues polling after ninety seconds without requiring refresh', async () => {
    const h = harness();
    let now = 0;
    h.context.Date = class extends Date { static now() { return now; } };
    let polls = 0;
    h.setQueuedApi(async url => {
        if (url === '/api/conversations') return { conversation_id: 'server-session-slow' };
        if (url.endsWith('/messages')) return { run_id: 'run-slow' };
        if (url.includes('/api/runs/run-slow?')) {
            polls++;
            if (polls === 1) { now = 90001; return { status: 'running', events: [] }; }
            return { status: 'completed', events: [], result: routeResult('slow-result') };
        }
        throw new Error('unexpected request ' + url);
    });
    const bubble = { text: 'waiting' };
    const result = await h.queued('从珞珈门去樱顶', {}, bubble, 0);
    assert.equal(result.route_state.route_id, 'slow-result');
    assert.equal(polls, 2);
});

test('simultaneous first messages share one conversation creation', async () => {
    const h = harness();
    const create = deferred();
    const postedConversations = [];
    let createCount = 0;
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations') { createCount++; return create.promise; }
        if (url.endsWith('/messages')) {
            postedConversations.push(url.split('/')[3]);
            return { run_id: 'run-' + postedConversations.length, status: 'queued' };
        }
        if (url.includes('/api/runs/'))
            return { status: 'completed', events: [], result: { task_type: 'chat', reply: '完成' } };
        throw new Error('unexpected request ' + url);
    });
    const first = h.queued('查天气', {}, { text: '' }, 0);
    const second = h.queued('介绍学校', {}, { text: '' }, 0);
    await Promise.resolve();
    assert.equal(createCount, 1);
    assert.equal(postedConversations.length, 0);
    create.resolve({ conversation_id: 'shared-conversation' });
    await Promise.all([first, second]);
    assert.equal(postedConversations.length, 2);
    assert.deepEqual(postedConversations, ['shared-conversation', 'shared-conversation']);
    assert.equal(h.state.serverConversationId, 'shared-conversation');
});

test('reset serializes a new conversation behind the old in-flight creation', async () => {
    const h = harness();
    const oldCreate = deferred(), newCreate = deferred();
    let createCount = 0;
    const posted = [];
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations') return ++createCount === 1 ? oldCreate.promise : newCreate.promise;
        if (url.endsWith('/messages')) {
            posted.push(url.split('/')[3]);
            return { run_id: 'fresh-run', status: 'queued' };
        }
        if (url.includes('/api/runs/'))
            return { status: 'completed', events: [], result: { task_type: 'chat', reply: '完成' } };
        throw new Error('unexpected request ' + url);
    });
    const stale = h.queued('旧会话请求', {}, { text: '' }, 0);
    await Promise.resolve();
    h.reset();
    const freshEpoch = h.state.conversationEpoch;
    const fresh = h.queued('新会话请求', {}, { text: '' }, freshEpoch);
    await Promise.resolve();
    assert.equal(createCount, 1);
    oldCreate.resolve({ conversation_id: 'old-conversation' });
    await stale;
    await Promise.resolve();
    assert.equal(createCount, 2);
    newCreate.resolve({ conversation_id: 'new-conversation' });
    await fresh;
    assert.deepEqual(posted, ['new-conversation']);
    assert.equal(h.state.serverConversationId, 'new-conversation');
});

test('failed conversation creation can be retried in the same epoch', async () => {
    const h = harness();
    let createCount = 0;
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations') {
            createCount++;
            if (createCount === 1) throw new Error('temporary create failure');
            return { conversation_id: 'retry-conversation' };
        }
        if (url.endsWith('/messages')) return { run_id: 'retry-run', status: 'queued' };
        if (url.includes('/api/runs/'))
            return { status: 'completed', events: [], result: { task_type: 'chat', reply: '完成' } };
        throw new Error('unexpected request ' + url);
    });
    await assert.rejects(h.queued('首次提交', {}, { text: '' }, 0), /temporary create failure/);
    await h.queued('重试提交', {}, { text: '' }, 0);
    assert.equal(createCount, 2);
    assert.equal(h.state.serverConversationId, 'retry-conversation');
});

test('expired server conversation is replaced after a rejected message', async () => {
    const h = harness();
    let createCount = 0, messageCount = 0;
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations') return { conversation_id: 'conversation-' + (++createCount) };
        if (url.endsWith('/messages')) {
            messageCount++;
            if (messageCount === 2) {
                const error = new Error('expired'); error.code = 'conversation_not_found'; throw error;
            }
            return { run_id: 'run-' + messageCount, status: 'queued' };
        }
        if (url.includes('/api/runs/'))
            return { status: 'completed', events: [], result: { task_type: 'chat', reply: '完成' } };
        throw new Error('unexpected request ' + url);
    });
    await h.queued('先发一条', {}, { text: '' }, 0);
    await h.queued('会话失效后重试', {}, { text: '' }, 0);
    assert.equal(createCount, 2);
    assert.equal(h.state.serverConversationId, 'conversation-2');
});

test('refresh resumes polling for an accepted active run', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const calls = [];
    h.setQueuedApi(async (url, body, method) => {
        calls.push({ url, method });
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [{ task_id: 'active-task', role: 'user', content: '从珞珈门到樱顶', seq: 1 }],
            tasks: [],
            active_runs: [{ run_id: 'active-run', task_id: 'active-task', query: '从珞珈门到樱顶', status: 'running' }],
        };
        if (url.includes('/api/runs/active-run?')) return {
            status: 'completed', task_id: 'active-task', current_task_revision: 1,
            events: [], result: routeResult('restored-route'),
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.ok(calls.some(call => call.url.includes('/api/runs/active-run?')));
    assert.deepEqual(h.userBubbles, ['从珞珈门到樱顶']);
    assert.equal(h.state.routeStore.current().route_id, 'restored-route');
    assert.equal(h.bubbles.at(-1).text, 'restored-route');
});

test('browser reload redraws the latest completed route only while its data and road versions match', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const saved = { ...existingRoute(), route_id: 'route-1', data_version: 'data-v1',
        road_condition_version: 'roads-v1' };
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'route-task', role: 'user', content: '从星湖园到卓尔体育馆', seq: 1 },
                { task_id: 'route-task', role: 'assistant', content: '路线已生成', seq: 2,
                    metadata: { task_type: 'path_planning', source_message_seq: 1,
                        run_id: 'completed-run', route_state: saved } },
            ],
            tasks: [], active_runs: [],
        };
        if (url === '/api/runs/completed-run?conversation_id=server-c1') return {
            status: 'completed', result: { ...routeResult('route-1'), route_state: saved },
        };
        if (url === '/api/road-conditions?version_only=1&include_route_data_version=1') return {
            data_version: 'data-v1', active_event_version: 'roads-v1',
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.rendered.length, 1);
    assert.equal(h.rendered[0].route_state.route_id, 'route-1');
});

test('browser reload does not redraw a completed route after road conditions change', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const saved = { ...existingRoute(), route_id: 'route-1', data_version: 'data-v1',
        road_condition_version: 'roads-v1' };
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'route-task', role: 'user', content: '从星湖园到卓尔体育馆', seq: 1 },
                { task_id: 'route-task', role: 'assistant', content: '路线已生成', seq: 2,
                    metadata: { task_type: 'path_planning', source_message_seq: 1,
                        run_id: 'completed-run', route_state: saved } },
            ],
            tasks: [], active_runs: [],
        };
        if (url === '/api/runs/completed-run?conversation_id=server-c1') return {
            status: 'completed', result: { ...routeResult('route-1'), route_state: saved },
        };
        if (url === '/api/road-conditions?version_only=1&include_route_data_version=1') return {
            data_version: 'data-v1', active_event_version: 'roads-v2',
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.rendered.length, 0);
    assert.equal(h.state.routeStore.current().route_id, 'route-1');
});

test('browser reload does not redraw a route after the campus data version changes', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const saved = { ...existingRoute(), route_id: 'route-1', data_version: 'data-v1',
        road_condition_version: 'roads-v1' };
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'route-task', role: 'user', content: '旧路线', seq: 1 },
                { task_id: 'route-task', role: 'assistant', content: '路线已生成', seq: 2,
                    metadata: { task_type: 'path_planning', source_message_seq: 1,
                        run_id: 'completed-run', route_state: saved } },
            ],
            tasks: [], active_runs: [],
        };
        if (url === '/api/runs/completed-run?conversation_id=server-c1') return {
            status: 'completed', result: { ...routeResult('route-1'), route_state: saved },
        };
        if (url === '/api/road-conditions?version_only=1&include_route_data_version=1') return {
            data_version: 'data-v2', active_event_version: 'roads-v1',
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.rendered.length, 0);
});

test('reset while a saved route is loading cannot redraw the old route', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const runRequest = deferred();
    const saved = { ...existingRoute(), route_id: 'route-1', data_version: 'data-v1',
        road_condition_version: 'roads-v1' };
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'route-task', role: 'user', content: '旧路线', seq: 1 },
                { task_id: 'route-task', role: 'assistant', content: '路线已生成', seq: 2,
                    metadata: { task_type: 'path_planning', source_message_seq: 1,
                        run_id: 'completed-run', route_state: saved } },
            ],
            tasks: [], active_runs: [],
        };
        if (url === '/api/runs/completed-run?conversation_id=server-c1') return runRequest.promise;
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    h.refreshPage();
    runRequest.resolve({ status: 'completed', result: { ...routeResult('route-1'), route_state: saved } });
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.rendered.length, 0);
    assert.equal(h.state.routeStore.current(), null);
});

test('a superseded route in the transcript is not restored as the current route', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'old-task', role: 'user', content: '旧路线', seq: 1 },
                { task_id: 'old-task', role: 'assistant', content: '已取消', seq: 2,
                    metadata: { task_type: 'path_planning', status: 'superseded',
                        run_id: 'old-run', route_state: { ...existingRoute(), route_id: 'old' } } },
            ],
            tasks: [], active_runs: [],
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    assert.equal(h.state.routeStore.current(), null);
    assert.equal(h.rendered.length, 0);
});

test('a restored older run cannot replace a newer route already in the transcript', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'older-task', role: 'user', content: '路线A', seq: 1 },
                { task_id: 'newer-task', role: 'user', content: '路线B', seq: 2 },
                { task_id: 'newer-task', role: 'assistant', content: '新路线', seq: 3,
                    metadata: { task_type: 'path_planning', route_state: { ...existingRoute(), route_id: 'newer-route' } } },
            ],
            tasks: [],
            active_runs: [{ run_id: 'older-run', task_id: 'older-task', query: '路线A', status: 'running' }],
        };
        if (url.includes('/api/runs/older-run?')) return {
            status: 'completed', task_id: 'older-task', current_task_revision: 1,
            events: [], result: routeResult('older-route'),
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.state.routeStore.current().route_id, 'newer-route');
    assert.equal(h.rendered.length, 0);
});

test('an older restored run absent from the capped transcript cannot replace a newer route', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'newer-task', role: 'user', content: '路线B', seq: 48 },
                { task_id: 'newer-task', role: 'assistant', content: '新路线', seq: 49,
                    metadata: { task_type: 'path_planning', route_state: { ...existingRoute(), route_id: 'newer-route' } } },
            ],
            tasks: [],
            active_runs: [{ run_id: 'older-run', task_id: 'older-task', query: '路线A', message_seq: 2, status: 'running' }],
        };
        if (url.includes('/api/runs/older-run?')) return {
            status: 'completed', task_id: 'older-task', current_task_revision: 1,
            events: [], result: routeResult('older-route'),
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.state.routeStore.current().route_id, 'newer-route');
    assert.equal(h.rendered.length, 0);
});

test('an old restored run with a missing transcript message cannot overwrite a route requested later', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    const oldRun = deferred();
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'newer-task', role: 'user', content: '路线B', seq: 48 },
                { task_id: 'newer-task', role: 'assistant', content: '新路线', seq: 49,
                    metadata: { task_type: 'path_planning', source_message_seq: 48,
                        route_state: { ...existingRoute(), route_id: 'newer-route' } } },
            ],
            tasks: [],
            active_runs: [{ run_id: 'older-run', task_id: 'older-task', query: '路线A', message_seq: 2 }],
        };
        if (url.includes('/api/runs/older-run?')) return oldRun.promise;
        throw new Error('unexpected request ' + url);
    });
    h.setApi(async () => routeResult('fresh-route'));
    await h.init();
    await h.submit('从珞珈门到图书馆');
    oldRun.resolve({ status: 'completed', task_id: 'older-task', current_task_revision: 1,
        events: [], result: routeResult('older-route') });
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(h.state.routeStore.current().route_id, 'fresh-route');
});

test('a late old route answer does not become the latest route after refresh', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    h.setQueuedApi(async url => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'newer-task', role: 'user', content: '路线B', seq: 48 },
                { task_id: 'newer-task', role: 'assistant', content: '新路线', seq: 49,
                    metadata: { task_type: 'path_planning', source_message_seq: 48,
                        route_state: { ...existingRoute(), route_id: 'newer-route' } } },
                { task_id: 'older-task', role: 'assistant', content: '迟到的旧路线', seq: 50,
                    metadata: { task_type: 'path_planning', source_message_seq: 2,
                        route_state: { ...existingRoute(), route_id: 'older-route' } } },
            ],
            tasks: [], active_runs: [],
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    assert.equal(h.state.routeStore.current().route_id, 'newer-route');
    assert.equal(h.state.latestRouteSequence, 48);
});

test('resumed clarification continuation appends its result after the latest user answer', async () => {
    const h = harness(new Map([['whu_walker:server_conversation', 'server-c1']]));
    h.setQueuedApi(async (url) => {
        if (url === '/api/conversations/server-c1') return {
            conversation_id: 'server-c1',
            messages: [
                { task_id: 'same-task', role: 'user', content: '带我去图书馆', seq: 1 },
                { task_id: 'same-task', role: 'assistant', content: '从哪里出发？', seq: 2 },
                { task_id: 'same-task', role: 'user', content: '从樱顶出发', seq: 3 },
            ],
            tasks: [],
            active_runs: [{ run_id: 'continuation-run', task_id: 'same-task', query: '从樱顶出发', status: 'running' }],
        };
        if (url.includes('/api/runs/continuation-run?')) return {
            status: 'completed', task_id: 'same-task', current_task_revision: 2,
            events: [], result: routeResult('continued-route'),
        };
        throw new Error('unexpected request ' + url);
    });
    await h.init();
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.deepEqual(Array.from(h.state.conversationHistory, message => message.content), [
        '带我去图书馆', '从哪里出发？', '从樱顶出发', 'continued-route',
    ]);
});

test('road-condition popup follows the published action and affected travel modes', () => {
    const h = harness();
    const notice = { type: 'closure', action: 'notice', affected_modes: ['walk'] };
    const slowdown = { type: 'event', action: 'slowdown', affected_modes: ['walk'], cost_multiplier: 1.7 };
    const closure = { type: 'event', action: 'closure', affected_modes: ['bike'] };
    const incomplete = { type: 'event', action: 'closure' };
    const unsupported = { type: 'event', action: 'detour', affected_modes: ['walk'] };

    assert.equal(h.roadConditionEffect(notice, 'walk').status, 'notice');
    assert.equal(h.roadConditionEffect(notice, 'drive').status, 'unaffected');
    assert.equal(h.roadConditionEffect(slowdown, 'walk').costMultiplier, 1.7);
    assert.equal(h.roadConditionEffect(slowdown, 'drive').status, 'unaffected');
    assert.equal(h.roadConditionEffect(closure, 'bike').status, 'blocked');
    assert.equal(h.roadConditionEffect(incomplete, 'drive').status, 'unknown');
    assert.equal(h.roadConditionEffect(unsupported, 'drive').status, 'unknown');
    assert.match(h.roadConditionEffectLabel(notice, 'walk', '步行'), /仅提示，不改变路线/);
    assert.match(h.roadConditionEffectLabel(slowdown, 'walk', '步行'), /规划成本约 ×1\.7/);
    assert.match(h.roadConditionEffectLabel(closure, 'bike', '骑行'), /禁止通行，路线会绕行/);
});

test('road-condition marker style follows the effect on the current travel mode', () => {
    const h = harness();
    const info = h.roadConditionDisplayStyle(
        { type: 'closure', action: 'notice', affected_modes: ['walk'] }, 'walk');
    const closure = h.roadConditionDisplayStyle(
        { type: 'event', action: 'closure', affected_modes: ['bike'] }, 'bike');
    const unaffected = h.roadConditionDisplayStyle(
        { type: 'flooding', action: 'closure', affected_modes: ['drive'] }, 'walk');
    const unknown = h.roadConditionDisplayStyle(
        { type: 'event', action: 'closure' }, 'walk');
    const legacy = h.roadConditionDisplayStyle({ type: 'flooding' }, 'walk');

    assert.deepEqual({ icon: info.icon, color: info.color, label: info.label },
        { icon: 'ℹ️', color: '#2878A8', label: '仅提示' });
    assert.deepEqual({ icon: closure.icon, color: closure.color, label: closure.label },
        { icon: '🚫', color: '#C0392B', label: '封闭' });
    assert.equal(unaffected.label, '本方式不受影响');
    assert.equal(unaffected.color, '#2878A8');
    assert.equal(unknown.label, '影响待核实');
    assert.equal(unknown.color, '#6B7280');
    assert.equal(legacy.icon, '🌊');
    assert.equal(legacy.color, '#2980B9');
});

test('road-condition legend follows the published action and affected travel modes', async () => {
    const h = harness();
    h.state.map = {};
    h.state.travelMode = 'walk';
    h.setRawApi(async () => ({ conditions: [
        { type: 'closure', action: 'notice', affected_modes: ['walk'] },
        { type: 'event', action: 'closure' },
    ] }));

    h.loadRoadConditions(false);
    await Promise.resolve();

    assert.equal(h.context.document.getElementById('legend-closure-item').hidden, true);
    assert.equal(h.context.document.getElementById('legend-slow-item').hidden, true);
    assert.equal(h.context.document.getElementById('legend-info-item').hidden, false);
    assert.equal(h.context.document.getElementById('legend-unknown-item').hidden, false);
});

test('road report submits a deliberate action and only the selected travel modes', async () => {
    const h = harness();
    const get = id => h.context.document.getElementById(id);
    h.state._pickedLng = 114.36;
    h.state._pickedLat = 30.53;
    get('rr-type').value = 'event';
    get('rr-action').value = 'slowdown';
    get('rr-mode-walk').checked = true;
    get('rr-name').value = '步行缓行';
    get('rr-desc').value = '';
    get('rr-start-time').value = '';
    get('rr-end-time').value = '';
    get('rr-cost-multiplier').value = '1.7';
    const sent = [];
    h.setRawApi(async (url, body) => {
        sent.push({ url, body });
        throw new Error('测试中止响应，保留表单');
    });

    await h.submitRoadReport({ preventDefault() {} });

    assert.equal(sent[0].url, '/api/road-conditions');
    assert.equal(sent[0].body.action, 'slowdown');
    assert.deepEqual(Array.from(sent[0].body.affected_modes), ['walk']);
    assert.equal(sent[0].body.cost_multiplier, 1.7);
    assert.equal(Object.hasOwn(sent[0].body, 'blocked_modes'), false);
});

test('road report refuses to publish before the route effect and mode are selected', async () => {
    const h = harness();
    const get = id => h.context.document.getElementById(id);
    h.state._pickedLng = 114.36;
    h.state._pickedLat = 30.53;
    get('rr-name').value = '未选影响';
    get('rr-type').value = 'event';
    get('rr-desc').value = '';
    get('rr-start-time').value = '';
    get('rr-end-time').value = '';
    let requestCount = 0;
    h.setRawApi(async () => { requestCount += 1; return {}; });

    await h.submitRoadReport({ preventDefault() {} });

    assert.equal(requestCount, 0);
    assert.match(get('road-report-hint').textContent, /明确选择仅提示、缓行或封路/);
});

test('new road conditions prompt a context-preserving route refresh', async () => {
    const h = harness();
    const original = {
        ...existingRoute(), route_id: 'before-road-change',
        road_condition_version: 'roads-old',
        start: { name: '信息学部', lng: 114.35, lat: 30.53 },
        end: { name: '星湖园食堂', lng: 114.36, lat: 30.54 },
        waypoints: [{ name: '桂园操场', lng: 114.355, lat: 30.535 }],
        hard_constraints: { avoid_steps: true },
        strategy: { name: 'shortest', weights: [1, 0, 0] },
    };
    h.state.routeStore.replace(original);
    const notice = h.context.document.getElementById('road-condition-update-notice');
    const calls = [];
    h.setRawApi(async (url, body) => {
        calls.push({ url, body });
        if (url === '/api/road-conditions?version_only=1') {
            return { active_event_version: 'roads-new' };
        }
        if (url === '/api/route/replan') {
            return {
                recommended: [[1, 2], [2, 3]],
                route_state: { ...body.route_state, route_id: 'after-road-change', road_condition_version: 'roads-new' },
            };
        }
        throw new Error('unexpected request ' + url);
    });

    await h.checkRoadConditionVersion();
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /保留起点、途经点和路线偏好/);
    const refresh = notice.children.find(child => child.dataset.action === 'replan-road-conditions');
    assert.ok(refresh);
    assert.equal(refresh.textContent, '按最新路况重算');

    await refresh.onclick();

    const request = calls.find(call => call.url === '/api/route/replan');
    assert.equal(request.body.change.refresh_road_conditions, true);
    assert.deepEqual(request.body.route_state.waypoints, original.waypoints);
    assert.deepEqual(request.body.route_state.hard_constraints, original.hard_constraints);
    assert.equal(h.state.routeStore.current().route_id, 'after-road-change');
    assert.equal(notice.hidden, true);
});

test('unchanged road conditions do not interrupt the displayed route', async () => {
    const h = harness();
    h.state.routeStore.replace({ ...existingRoute(), road_condition_version: 'roads-stable' });
    const notice = h.context.document.getElementById('road-condition-update-notice');
    notice.hidden = false;
    h.setRawApi(async url => {
        assert.equal(url, '/api/road-conditions?version_only=1');
        return { active_event_version: 'roads-stable' };
    });

    await h.checkRoadConditionVersion();

    assert.equal(notice.hidden, true);
    assert.equal(h.state.routeStore.current().route_id, 'existing');
});

test('route with next-stage place choices keeps the same task open', async () => {
    const h = harness();
    const reply = { ...routeResult('stage-one'), needs_input: true,
        _task_id: 'task-composite', _task_revision: 3,
        clarify: { question: 'Choose a meal place', options: [] },
        candidates: [{ poi_id: 'poi_310', name: 'Canteen' }],
    };
    h.setApi(async () => reply);
    await h.submit('first place');
    assert.equal(h.rendered.length, 1);
    assert.equal(h.candidateRenders.length, 1);
    assert.equal(h.candidateRenders[0].taskId, 'task-composite');
    assert.equal(h.state.pendingServerTaskId, 'task-composite');
    assert.equal(h.state.pendingServerTaskRevision, 3);
});
