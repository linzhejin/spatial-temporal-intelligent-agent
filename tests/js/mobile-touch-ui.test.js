const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const app = fs.readFileSync(path.join(root, 'static/js/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'static/index.html'), 'utf8');
const css = fs.readFileSync(path.join(root, 'static/css/style.css'), 'utf8');

function cssBlock(source, selector, fromIndex = 0) {
  const start = source.indexOf(selector, fromIndex);
  assert.notEqual(start, -1, 'missing CSS selector: ' + selector);
  const end = source.indexOf('}', start);
  return source.slice(start, end + 1);
}

test('map point controls keep only centered point labels', () => {
  assert.equal(app.includes('whu-point-action'), false);
  assert.ok(cssBlock(css, '.whu-point-btn {').includes('justify-content: center'));
});

test('mobile panel separator has a larger touch target without taking extra layout space', () => {
  const mobileStart = css.indexOf('@media (max-width: 767px)');
  assert.notEqual(mobileStart, -1);
  const block = cssBlock(css, '.mobile-panel-resizer {', mobileStart);
  const heightMatch = block.match(/height:[ ]*([0-9]+)px/);
  assert.ok(heightMatch, 'resizer height is declared');
  const height = Number(heightMatch[1]);
  const marginMatch = block.match(/margin-bottom:[ ]*(-?[0-9]+)px/);
  const margin = marginMatch ? Number(marginMatch[1]) : 0;
  assert.ok(height >= 44, 'touch target is only ' + height + 'px high');
  assert.equal(height + margin, 20, 'larger hit area should preserve the current panel layout footprint');
});

test('coarse-pointer phones do not display keyboard shortcut hints', () => {
  const coarseStart = css.indexOf('@media (pointer: coarse)');
  assert.notEqual(coarseStart, -1, 'touchscreen-specific rule exists');
  assert.ok(cssBlock(css, '.kbd-hint-bar {', coarseStart).includes('display: none !important'));
  const helpStart = app.indexOf('function showKbdHelp()');
  assert.notEqual(helpStart, -1);
  assert.ok(app.slice(helpStart, app.indexOf('\n    function hideKbdHelp()', helpStart)).includes("'(pointer: coarse)'"));
});

test('phone guidance uses plain wording and removes slang', () => {
  assert.equal(html.includes('\u8bf4\u4eba\u8bdd'), false);
  assert.ok(html.includes('\u8f93\u5165\u8d77\u70b9\u3001\u7ec8\u70b9\u6216\u8def\u7ebf\u8981\u6c42'));
});

test('chat bubbles wrap long words and URLs', () => {
  assert.match(css, /\.chat-bubble-text\s*\{[^}]*overflow-wrap:\s*anywhere/);
  assert.match(css, /\.bubble-text\s*\{[^}]*overflow-wrap:\s*anywhere/);
});

test('readable secondary grey and larger header touch targets', () => {
  assert.doesNotMatch(css, /\.rr-time-hint\s*\{[^}]*#999/);
  assert.match(css, /\.rr-time-hint\s*\{[^}]*color:\s*var\(--ink-light\)/);
  assert.ok(Number(cssBlock(css, '.road-report-btn {').match(/width:[ ]*([0-9]+)px/)[1]) >= 40);
  assert.ok(Number(cssBlock(css, '.help-btn {').match(/width:[ ]*([0-9]+)px/)[1]) >= 40);
});
