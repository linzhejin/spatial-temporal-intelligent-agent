/* ===========================================================
 * 珞珈智行 · Service Worker（T-018 §5 标准）
 * 三缓存策略（TDD §9.4 明确）:
 *   1. cache-first        → 首屏静态资源 /static/*  (HTML/CSS/JS/icon/manifest)
 *   2. stale-while-revalidate → /api/* 后端接口调用（SWR：出缓存 + 后台静默更新）
 *   3. network-only       → 第三方资源（高德 JS API / amap.com / amapw.com）
 * =========================================================== */

var CACHE_NAME = 'whu-walker-v91';
var PRECACHE_URLS = [
    '/',
    '/index.html',
    '/css/style.css?v=20261014-roadconditioneffects2',
    '/js/app.js?v=20261014-roadconditioneffects2',
    '/js/config.js',
    '/js/voice-output.js',
    '/js/voice-input.js',
    '/js/navigation.js',
    '/manifest.json',
    '/icons/icon-192.png',
    '/icons/icon-512.png',
    '/icons/icon-maskable-192.png',
    '/icons/icon-maskable-512.png',
    '/icons/apple-touch-icon.png',
];

/* ---------- install: 预缓存首屏资源 ---------- */
self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            return cache.addAll(PRECACHE_URLS);
        }).then(function () {
            /* 完整缓存首屏资源后才接管；失败时旧 worker 和旧缓存继续可用。 */
            return self.skipWaiting();
        })
    );
});

/* ---------- activate: 清理旧版本缓存 ---------- */
self.addEventListener('activate', function (event) {
    event.waitUntil(
            caches.keys().then(function (keys) {
                return Promise.all(
                    keys.filter(function (k) { return k.indexOf('whu-walker-') === 0 && k !== CACHE_NAME; })
                    .map(function (k) { return caches.delete(k); })
            );
        })
    );
    self.clients.claim();
});

/* ---------- fetch: 三策略分发 ---------- */
self.addEventListener('fetch', function (event) {
    var request = event.request;
    var url = new URL(request.url);

    /* ───────────────────────────────────────────────
     * 策略 ③ network-only：第三方高德 API / 非同源
     *   · 不缓存任何 amap.com / amapw.com 资源
     *   · 其他非同源请求也一律直走网络
     * ─────────────────────────────────────────────── */
    var isThirdParty = url.origin !== location.origin;
    var isAmapDomain = /amapw?\.com$/i.test(url.hostname) ||
                       /webapi\.amap\.com/i.test(url.href) ||
                       /restapi\.amap\.com/i.test(url.href);

    if (isThirdParty || isAmapDomain) {
        /* NETWORK-ONLY：直接 fetch，不读也不写缓存 */
        event.respondWith(fetch(request));
        return;
    }

    /* ───────────────────────────────────────────────
     * 策略 ② stale-while-revalidate：后端 /api/* 接口
     *   · 优先从缓存返回（秒级响应）
     *   · 同时后台发起新请求更新缓存（下次命中）
     *   · 缓存 miss 时直走网络，成功后写缓存
     * ─────────────────────────────────────────────── */
    var isApiCall = url.pathname.startsWith('/api/');

    if (isApiCall) {
        /* 会话、运行结果和管理台包含私有数据：永不写入离线 Cache API。 */
        var isPrivateApi = url.pathname === '/api/conversations' ||
            url.pathname.indexOf('/api/conversations/') === 0 ||
            url.pathname === '/api/runs' || url.pathname.indexOf('/api/runs/') === 0 ||
            url.pathname === '/api/admin' || url.pathname.indexOf('/api/admin/') === 0 ||
            url.pathname === '/api/manager' || url.pathname.indexOf('/api/manager/') === 0;
        if (isPrivateApi) {
            event.respondWith(fetch(request));
            return;
        }
        /* 校园 POI 主库与源路网必须使用当前发布版；非 GET 请求也不进入 Cache API。 */
        var isCampusMasterApi = request.method !== 'GET' ||
            url.pathname === '/api/pois' ||
            url.pathname === '/api/course-spatial-reference';
        if (isCampusMasterApi) {
            event.respondWith(fetch(request));
            return;
        }
        /* 路况管制 / 管理员接口：NETWORK-ONLY（强一致）。
           管理者结束/删除事件后立即刷新列表，SWR 会返回旧快照导致"删了还在"；
           管制信息时效性高，宁可离线不可用（前端已静默 catch），也不展示过期状态。 */
        var isRealtimeApi = url.pathname.indexOf('/api/road-conditions') === 0 ||
                           url.pathname.indexOf('/api/admin') === 0;
        if (isRealtimeApi) {
            event.respondWith(fetch(request));
            return;
        }
        event.respondWith(
            caches.open(CACHE_NAME).then(function (cache) {
                return cache.match(request).then(function (cached) {
                    /* 后台无论如何都重新 fetch，更新到缓存（revalidate） */
                    var fetchPromise = fetch(request).then(function (networkResponse) {
                        if (networkResponse && networkResponse.status === 200) {
                            cache.put(request, networkResponse.clone());
                        }
                        return networkResponse;
                    }).catch(function () {
                        /* 离线/网络失败：吞掉错误，返回 cached（或兜底 504） */
                        return cached || new Response(
                            JSON.stringify({ error: 'Network unavailable (offline fallback)' }),
                            { status: 504, statusText: 'Offline', headers: { 'Content-Type': 'application/json' } }
                        );
                    });
                    /* SWR 核心：有 cached → 立即返回 cached，同时后台 revalidate */
                    return cached || fetchPromise;
                });
            })
        );
        return;
    }

    /* ───────────────────────────────────────────────
     * 安装包 / 版本清单：NETWORK-ONLY
     *   · APK 以同名 URL 覆盖发版，若走 cache-first，浏览器横幅一旦下载过，
     *     以后拿到的永远是缓存里的旧包；版本清单同理必须实时
     *   · 原生 App 的更新检查走 HttpURLConnection，不经过 SW，
     *     这里保护的是浏览器/网页内的下载入口与未来可能的网页侧检测
     * ─────────────────────────────────────────────── */
    if (url.pathname.indexOf('/app/') === 0) {
        event.respondWith(fetch(request));
        return;
    }

    /* ───────────────────────────────────────────────
     * 策略 ① 静态资源
     *   · navigate（HTML 导航）：network-first —— 有网必拿最新页面，
     *     成功后写缓存；离线才回退缓存的 index.html（SPA offline shell）
     *   · 其他静态（CSS/JS/img/icon）：cache-first（发版同步更新缓存名与资源版本号）
     * ─────────────────────────────────────────────── */
    if (request.mode === 'navigate') {
        var isAppShell = url.pathname === '/' || url.pathname === '/index.html';
        event.respondWith(
            fetch(request).then(function (response) {
                if (isAppShell && response && response.status === 200) {
                    var copy = response.clone();
                    caches.open(CACHE_NAME).then(function (cache) {
                        cache.put('/index.html', copy);
                    });
                }
                return response;
            }).catch(function () {
                return isAppShell
                    ? caches.match('/index.html')
                    : new Response('', { status: 504, statusText: 'Offline' });
            })
        );
        return;
    }

    event.respondWith(
        caches.match(request).then(function (cached) {
            if (cached) {
                return cached; /* CACHE-HIT：直接返回，不必访问网络 */
            }
            /* CACHE-MISS：回源 fetch，成功后写缓存下次命中 */
            return fetch(request).then(function (response) {
                if (response && response.status === 200) {
                    var copy = response.clone();
                    caches.open(CACHE_NAME).then(function (cache) {
                        cache.put(request, copy);
                    });
                }
                return response;
            }).catch(function () {
                /* navigate：离线时兜底返回已缓存的 index.html（SPA offline shell） */
                if (request.mode === 'navigate') {
                    return caches.match('/index.html');
                }
                return new Response('', { status: 504, statusText: 'Offline' });
            });
        })
    );
});
