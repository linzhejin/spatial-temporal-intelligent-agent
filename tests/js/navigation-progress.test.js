const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function navigation() {
  let now = 100000;
  const cards = [], paths = [];
  const map = { on() {}, off() {}, removeLayer() {}, getZoom() { return 17; }, panTo() {} };
  const marker = { addTo() { return this; }, setLatLng() {}, getElement() { return null; } };
  const context = vm.createContext({
    window: {}, navigator: {}, document: { addEventListener() {}, removeEventListener() {} },
    L: { marker() { return marker; }, divIcon(o) { return o; } },
    Date: class extends Date { static now() { return now; } },
    setInterval() { return 1; }, clearInterval() {}, console,
  });
  vm.runInContext(fs.readFileSync(require.resolve('../../static/js/navigation.js'), 'utf8'), context);
  const nav = context.window.WhuWalkerNavigation;
  nav.start({ map, routeData: { mode: 'walk', recommended: [
    { lat: 30, lng: 114 }, { lat: 30, lng: 114.001 }, { lat: 30, lng: 114.002 },
  ], steps: [{ type: 'depart', action: 'depart', cumulative_m: 0 },
    { type: 'arrive', action: 'arrive', cumulative_m: 193 }] },
  onCard(card) { cards.push(card); }, onProgress(path) { paths.push(path); } });
  return { nav, cards, paths, fix(lng) { now += 1000; nav.onFix({ gcjLat: 30, gcjLng: lng, accuracy: 5 }); } };
}

test('stationary fixes do not send the estimated time toward infinity', () => {
  const n = navigation();
  for (let i = 0; i < 60; i++) n.fix(114.0002);
  assert.ok(Number.isFinite(n.cards.at(-1).remainingMin));
  assert.ok(n.cards.at(-1).remainingMin <= 5, `remaining ${n.cards.at(-1).remainingMin} minutes`);
  n.nav.stop();
});

test('route progress starts at the projected position and advances along the remaining geometry', () => {
  const n = navigation();
  n.fix(114.0005);
  assert.equal(n.paths.length, 1);
  assert.ok(Math.abs(n.paths[0][0].lng - 114.0005) < 1e-8);
  assert.equal(n.paths[0].length, 3);
  n.fix(114.0015);
  assert.ok(Math.abs(n.paths[1][0].lng - 114.0015) < 1e-8);
  assert.equal(n.paths[1].length, 2);
  n.nav.stop();
});

test('location updates preserve the zoom level selected by the user', () => {
  let zoom = 14;
  const map = { on() {}, off() {}, removeLayer() {}, getZoom() { return zoom; },
    panTo() {}, setView(center, level) { zoom = level; } };
  const marker = { addTo() { return this; }, setLatLng() {}, getElement() { return null; } };
  const context = vm.createContext({ window: {}, navigator: {},
    document: { addEventListener() {}, removeEventListener() {} },
    L: { marker() { return marker; }, divIcon(o) { return o; } },
    setInterval() { return 1; }, clearInterval() {}, console });
  vm.runInContext(fs.readFileSync(require.resolve('../../static/js/navigation.js'), 'utf8'), context);
  const nav = context.window.WhuWalkerNavigation;
  nav.start({ map, routeData: { mode: 'walk', recommended: [{ lat: 30, lng: 114 },
    { lat: 30, lng: 114.002 }], steps: [] }, onCard() {} });
  nav.onFix({ gcjLat: 30, gcjLng: 114.0002, accuracy: 5 });
  assert.equal(zoom, 14);
  nav.stop();
});
