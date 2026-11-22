const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  const calls = [];
  const element = () => ({ classList: { add() {}, remove() {}, toggle() {} }, style: {},
    setAttribute() {}, querySelector() { return element(); }, offsetWidth: 1, textContent: '' });
  const window = { WhuWalkerVoice: {
    press() { calls.push('press'); }, release() { calls.push('release'); }, cancel() { calls.push('cancel'); },
  } };
  const context = vm.createContext({ window, navigator: {}, document: { createElement: element,
    body: { appendChild() {} } }, console });
  vm.runInContext(fs.readFileSync(require.resolve('../../static/js/voice-input.js'), 'utf8'), context);
  const results = [];
  window.WhuWalkerVoiceInput.press({ onFinal(text) { results.push(text); } });
  return { window, calls, results };
}

test('native recognition waits for release even if a final result arrives early', () => {
  const t = setup();
  t.window.__whuWalkerVoiceCallback({ event: 'final', text: '去图书馆' });
  assert.deepEqual(t.results, []);
  t.window.WhuWalkerVoiceInput.release();
  assert.deepEqual(t.results, ['去图书馆']);
});

test('an early final result is discarded when the hold is cancelled', () => {
  const t = setup();
  t.window.__whuWalkerVoiceCallback({ event: 'final', text: '去图书馆' });
  t.window.WhuWalkerVoiceInput.cancel();
  assert.deepEqual(t.results, []);
});

test('browser final results wait for release even if recognition ends first', () => {
  let recognition;
  const element = () => ({ classList: { add() {}, remove() {}, toggle() {} }, style: {},
    setAttribute() {}, querySelector() { return element(); }, offsetWidth: 1 });
  function SpeechRecognition() { recognition = this; }
  SpeechRecognition.prototype.start = function () {};
  SpeechRecognition.prototype.stop = function () {};
  const window = { SpeechRecognition };
  const context = vm.createContext({ window, navigator: {}, document: { createElement: element,
    body: { appendChild() {} } }, console });
  vm.runInContext(fs.readFileSync(require.resolve('../../static/js/voice-input.js'), 'utf8'), context);
  const results = [];
  window.WhuWalkerVoiceInput.press({ onFinal(text) { results.push(text); } });
  recognition.onstart();
  const result = [{ transcript: '去图书馆' }];
  result.isFinal = true;
  recognition.onresult({ resultIndex: 0, results: [result] });
  recognition.onend();
  assert.deepEqual(results, []);
  window.WhuWalkerVoiceInput.release();
  assert.deepEqual(results, ['去图书馆']);
});
