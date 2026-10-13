const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../../static/sw.js'), 'utf8');
const indexSource = fs.readFileSync(path.join(__dirname, '../../static/index.html'), 'utf8');

test('deployed app bundle uses a fresh and consistent service-worker cache key', () => {
  const htmlAsset = indexSource.match(/<script\s+src="([^\"]*\/js\/app\.js\?v=([^\"]+))"/);
  const workerAsset = source.match(/'([^']*\/js\/app\.js\?v=([^']+))'/);
  const htmlStylesheet = indexSource.match(/<link\s+rel="stylesheet"\s+href="([^"]*\/css\/style\.css\?v=([^"]+))"/);
  const workerStylesheet = source.match(/'([^']*\/css\/style\.css\?v=([^']+))'/);
  const cacheVersion = source.match(/var CACHE_NAME = 'whu-walker-v(\d+)'/);

  assert.ok(htmlAsset, 'index.html must version app.js');
  assert.ok(workerAsset, 'the service worker must precache versioned app.js');
  assert.ok(htmlStylesheet, 'index.html must version style.css');
  assert.ok(workerStylesheet, 'the service worker must precache versioned style.css');
  assert.ok(cacheVersion, 'the service worker must declare its cache version');
  assert.equal(htmlAsset[1], workerAsset[1]);
  assert.equal(htmlStylesheet[1], workerStylesheet[1]);
  assert.notEqual(htmlAsset[2], '20261009c');
  assert.notEqual(htmlStylesheet[2], '20261009a');
  assert.notEqual(htmlAsset[2], '20261003a');
  assert.notEqual(htmlAsset[2], '20261007a');
  assert.ok(Number(cacheVersion[1]) > 84);
});

function loadWorker(overrides = {}) {
  const listeners = {};
  const skipWaitingCalls = [];
  const self = {
    addEventListener: (name, callback) => { listeners[name] = callback; },
    skipWaiting: () => { skipWaitingCalls.push(true); return Promise.resolve(); },
    clients: { claim: () => Promise.resolve() },
  };
  vm.runInNewContext(source, {
    self,
    URL,
    Response,
    Promise,
    location: { origin: 'https://campus.example' },
    caches: { keys: async () => [], open: async () => ({ put: async () => {} }), match: async () => undefined },
    fetch: async () => new Response('ok', { status: 200 }),
    ...overrides,
  });
  listeners.skipWaitingCalls = skipWaitingCalls;
  return listeners;
}

test('service worker activation deletes the previously deployed cache version', async () => {
  const deleted = [];
  const listeners = loadWorker({
    caches: {
      keys: async () => ['whu-walker-v60', 'unrelated-cache'],
      delete: async (key) => { deleted.push(key); return true; },
    },
  });
  const event = { waitUntil(promise) { this.promise = promise; } };
  listeners.activate(event);
  await event.promise;
  assert.deepEqual(deleted, ['whu-walker-v60']);
});

test('failed shell precache keeps the installed worker active and preserves old caches', async () => {
  const listeners = loadWorker({
    caches: {
      open: async () => ({ addAll: async () => { throw new Error('network unavailable'); } }),
      keys: async () => ['whu-walker-v61'],
      delete: async () => true,
    },
  });
  const event = { waitUntil(promise) { this.promise = promise; } };
  listeners.install(event);
  await assert.rejects(event.promise, /network unavailable/);
  assert.equal(listeners.skipWaitingCalls.length, 0);
});

test('private conversation, run and manager APIs bypass the Cache API', async () => {
  const requested = [];
  const listeners = loadWorker({
    caches: {
      open: async () => { throw new Error('private API must not open a cache'); },
      match: async () => { throw new Error('private API must not read a cache'); },
    },
    fetch: async request => {
      requested.push(new URL(request.url).pathname);
      return new Response('private', { status: 200 });
    },
  });

  for (const pathname of [
    '/api/conversations', '/api/conversations/c1', '/api/runs', '/api/runs/r1',
    '/api/manager/vision-jobs', '/api/manager/vision-jobs/j1',
  ]) {
    const event = {
      request: { url: 'https://campus.example' + pathname, method: 'GET', mode: 'cors' },
      respondWith(promise) { this.promise = promise; },
    };
    listeners.fetch(event);
    assert.equal(await (await event.promise).text(), 'private');
  }
  assert.equal(requested.length, 6);
});

test('manager navigation never overwrites the offline app-shell cache', async () => {
  const writes = [];
  const listeners = loadWorker({
    caches: {
      open: async () => ({ put: async (key) => { writes.push(key); } }),
      match: async () => undefined,
    },
  });
  const event = {
    request: { url: 'https://campus.example/manager', mode: 'navigate' },
    respondWith(promise) { this.promise = promise; },
  };
  listeners.fetch(event);
  const response = await event.promise;
  assert.equal(await response.text(), 'ok');
  assert.deepEqual(writes, []);
});

test('offline app-shell requests still use the cached home page', async () => {
  const listeners = loadWorker({
    fetch: async () => { throw new Error('offline'); },
    caches: { match: async (key) => key === '/index.html' ? new Response('cached home') : undefined },
  });
  const event = {
    request: { url: 'https://campus.example/', mode: 'navigate' },
    respondWith(promise) { this.promise = promise; },
  };
  listeners.fetch(event);
  assert.equal(await (await event.promise).text(), 'cached home');
});
