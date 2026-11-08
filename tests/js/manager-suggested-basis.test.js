const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function loadSuggestedBasisLabel() {
  const source = fs.readFileSync(require.resolve('../../static/js/manager.js'), 'utf8');
  const marker = 'function appendSuggestedEdges(card, job, edges) {';
  const end = source.indexOf(marker);
  assert.ok(end > 0, 'manager.js suggested-edges boundary exists');
  const context = vm.createContext({
    window: {
      sessionStorage: {
        getItem() { return null; },
        setItem() {},
        removeItem() {},
      },
    },
    document: {
      getElementById() { return null; },
    },
  });
  vm.runInContext(source.slice(0, end) + `
    globalThis.managerBasis = {
      suggestedBasisLabel: suggestedBasisLabel,
    };
  ` + '\n})();', context);
  return context.managerBasis;
}

const helpers = loadSuggestedBasisLabel();

test('suggestedBasisLabel maps each known basis to its Chinese label', () => {
  assert.equal(helpers.suggestedBasisLabel('mask_projection'), '按影像轮廓投影排序');
  assert.equal(helpers.suggestedBasisLabel('pose_footprint'), '按拍摄范围排序');
  assert.equal(helpers.suggestedBasisLabel('radius_only'), '邻近路段参考');
});

test('suggestedBasisLabel falls back for unknown or missing basis', () => {
  assert.equal(helpers.suggestedBasisLabel(undefined), '邻近路段参考');
  assert.equal(helpers.suggestedBasisLabel(null), '邻近路段参考');
  assert.equal(helpers.suggestedBasisLabel('some-future-basis'), '邻近路段参考');
});
