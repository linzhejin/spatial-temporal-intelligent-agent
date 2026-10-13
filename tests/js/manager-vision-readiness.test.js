const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../../static/js/manager.js'), 'utf8');
const managerHtml = fs.readFileSync(path.join(__dirname, '../../static/manager.html'), 'utf8');
const managerCss = fs.readFileSync(path.join(__dirname, '../../static/css/manager.css'), 'utf8');

function managerHarness(sessionSeed = []) {
  const elements = new Map();
  const intervalCallbacks = [];
  const mapListeners = {};
  const mapLayers = [];
  const removedMapLayers = [];
  const requested = [];
  const uploadResolvers = [];
  const impactPreviewResolvers = [];
  const createdObjectUrls = [];
  const revokedObjectUrls = [];
  const formEntries = [];
  const sessionValues = new Map(sessionSeed);
  let storageFailure = false;
  let visionPostCount = 0;
  let videoUploadOffset = 0;
  let videoChunkResponseDrops = true;
  const videoChunkSizes = [];
  let visionJobs = [];
  let visionScenes = [];
  let roadEvents = [];
  const calls = [];
  let inferenceReady = false;
  let visionPublicationAlreadyExists = false;
  let optionalModelStatus = { accident: 'not_configured', flood: 'not_configured' };
  let visionStatusFailure = false;
  let sessionExpired = false;
  let deferImpactPreviews = false;
  let createdNodeCount = 0;

  function element(id) {
    if (elements.has(id)) return elements.get(id);
    const listeners = {};
    const classes = new Set();
    let textContent = '';
    const node = {
      id, hidden: false, value: '', innerHTML: '', disabled: false, currentTime: 0, duration: NaN,
      checked: false, files: [], style: {}, listeners, children: [], attributes: {},
      clickCount: 0, scrollCount: 0,
      setAttribute(name, value) { this.attributes[name] = String(value); },
      removeAttribute(name) { delete this.attributes[name]; },
      classList: {
        add: (value) => classes.add(value),
        remove: (value) => classes.delete(value),
        contains: (value) => classes.has(value) || String(node.className || '').split(/\s+/).includes(value),
        toggle: (value, force) => force ? classes.add(value) : classes.delete(value),
      },
      addEventListener: (name, callback) => { listeners[name] = callback; },
      appendChild(child) { this.children.push(child); return child; },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      click() { this.clickCount += 1; },
      scrollIntoView() { this.scrollCount += 1; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    };
    Object.defineProperty(node, 'textContent', {
      get: () => textContent,
      set(value) {
        textContent = value;
        if (value === '') this.children = [];
      },
    });
    elements.set(id, node);
    return node;
  }

  const map = {
    setView() { return this; },
    fitBounds(bounds, options) { this.lastFitBounds = { bounds, options }; return this; },
    invalidateSize() {},
    on(name, callback) { mapListeners[name] = callback; },
    removeLayer(layer) { removedMapLayers.push(layer); },
  };
  function makeLayer(kind, coordinates, options) {
    return {
      kind, coordinates, options,
      addTo() { mapLayers.push(this); return this; },
      clearLayers() {},
      setLatLng(value) { this.coordinates = value; return this; },
    };
  }
  const L = {
    map: () => map,
    tileLayer: () => makeLayer('tile'),
    layerGroup: () => makeLayer('group'),
    circleMarker: (coordinates, options) => makeLayer('circleMarker', coordinates, options),
    polyline: (coordinates, options) => makeLayer('polyline', coordinates, options),
  };
  const window = {
    WHU_WALKER_CONFIG: { mapCenter: [114.363, 30.5365], mapZoom: 15 },
    setTimeout: (fn) => fn(),
    setInterval: (fn) => { intervalCallbacks.push(fn); return intervalCallbacks.length; },
    clearInterval() {},
    alert() {}, confirm: () => true,
    sessionStorage: {
      getItem(key) { if (storageFailure) throw new Error('storage unavailable'); return sessionValues.get(key) || null; },
      setItem(key, value) { if (storageFailure) throw new Error('storage unavailable'); sessionValues.set(key, String(value)); },
      removeItem(key) { if (storageFailure) throw new Error('storage unavailable'); sessionValues.delete(key); },
    },
  };
  const document = { hidden: false, activeElement: null, getElementById: element, createElement: (tag) => {
      const node = element('created-' + tag + '-' + createdNodeCount++);
      node.tagName = tag;
      node.getContext = () => ({
        clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, fill() {}, stroke() {}, arc() {},
        drawImage() {}, getImageData() { return { data: new Uint8ClampedArray(9 * 8 * 4).fill(100) }; },
      });
      return node;
    }, createElementNS: (_namespace, tag) => {
      const node = element('created-' + tag + '-' + createdNodeCount++);
      node.tagName = tag;
      return node;
    }, createTextNode: (text) => ({ text }) };
  const URLApi = {
    createObjectURL(file) {
      const value = 'blob:preview-' + createdObjectUrls.length;
      createdObjectUrls.push({ file, value });
      return value;
    },
    revokeObjectURL(value) { revokedObjectUrls.push(value); },
  };
  const context = {
    document, URL: URLApi,
    window, L, FormData: class FormData { append(name, value) { formEntries.push([name, value]); } }, URLSearchParams, Date, Math, Promise, setTimeout,
    fetch: async (url, options = {}) => {
      requested.push(url);
      calls.push({ url, options });
      if (url === '/api/manager/vision-uploads' && options.method === 'POST') {
        const body = JSON.parse(options.body);
        return { ok: true, json: async () => ({ data: {
          upload_id: 'upload-1', offset: 0, chunk_size: body.size > 100 ? 512 : 4, size: body.size,
        } }) };
      }
      if (url.startsWith('/api/manager/vision-uploads/') && options.method === 'GET') {
        return { ok: true, json: async () => ({ data: { offset: videoUploadOffset } }) };
      }
      if (url.endsWith('/chunks') && options.method === 'PUT') {
        const offset = Number(options.headers['Upload-Offset']);
        videoChunkSizes.push(options.body.size);
        videoUploadOffset = offset + options.body.size;
        if (videoChunkResponseDrops) {
          videoChunkResponseDrops = false;
          return { ok: false, status: 503, json: async () => ({ message: '模拟网络中断' }) };
        }
        return { ok: true, json: async () => ({ data: { offset: videoUploadOffset } }) };
      }
      if (url.endsWith('/complete') && options.method === 'POST') {
        return { ok: true, json: async () => ({ data: { job: { job_id: 'job-video' } } }) };
      }
      if (url.startsWith('/api/road-conditions/snap') && deferImpactPreviews) {
        const match = url.match(/[?&]type=([^&]+)/);
        const type = match ? decodeURIComponent(match[1]) : '';
        return new Promise((resolve) => impactPreviewResolvers.push({ type, resolve }));
      }
      if (url === '/api/manager/vision-jobs' && options.method === 'POST') {
        visionPostCount += 1;
        return new Promise((resolve) => uploadResolvers.push(resolve));
      }
      if (url === '/api/road-conditions' && options.method === 'POST' && visionPublicationAlreadyExists) {
        return { ok: true, json: async () => ({ data: { already_published: true, event: { id: 'existing-event' } } }) };
      }
      if (url === '/api/manager/vision-scenes' && options.method === 'POST') {
        const body = typeof options.body === 'string' ? JSON.parse(options.body) : {};
        const scene = { ...body, scene_id: body.scene_id || 'scene-created' };
        visionScenes = visionScenes.filter((item) => item.scene_id !== scene.scene_id).concat(scene);
        return { ok: true, json: async () => ({ data: { scene } }) };
      }
      if (url.startsWith('/api/manager/vision-scenes/') && options.method === 'DELETE') {
        const sceneId = decodeURIComponent(url.split('/').pop());
        visionScenes = visionScenes.filter((item) => item.scene_id !== sceneId);
        return { ok: true, json: async () => ({ data: { deleted: true } }) };
      }
      if (url.startsWith('/api/manager/vision-jobs/') && options.method === 'DELETE') {
        const jobId = decodeURIComponent(url.split('/')[4] || '');
        visionJobs = visionJobs.filter((job) => job.job_id !== jobId);
        return { ok: true, json: async () => ({ data: { message: 'deleted' } }) };
      }
      if (url.startsWith('/api/road-conditions/') && options.method === 'DELETE') {
        const conditionId = decodeURIComponent(url.split('/').pop() || '');
        roadEvents = roadEvents.filter((event) => event.id !== conditionId);
        return { ok: true, json: async () => ({ data: { message: 'deleted' } }) };
      }
      if (url.startsWith('/api/manager/vision-status') && (visionStatusFailure || sessionExpired)) {
        return { ok: false, status: sessionExpired ? 401 : 503,
          json: async () => ({ message: sessionExpired ? '需要管理员权限' : '状态读取失败' }) };
      }
      let data = {};
      if (url === '/api/admin/status') data = { is_admin: true, csrf_token: 'csrf' };
      else if (url === '/api/manager/vision-scenes') data = { scenes: visionScenes };
      else if (url.startsWith('/api/road-conditions/snap')) data = {
        snap: { u: 1, v: 2, key: 0, road_name: '测试路', dist_m: 2,
          chain_length_m: 100, snap_lng_gcj: 114.36002, snap_lat_gcj: 30.53,
          geometry_gcj: [[114.36, 30.53], [114.361, 30.53]] },
        impact_preview: {
          road_name: '测试路', affected_road_segments: 1, affected_length_m: 100,
          active_event_count: 2, event_scope_note: '基于当前有效路况快照',
          route_sample_scope: '仅为所选路段两端的示例路线',
          effects_by_mode: {}, sample_routes: {},
        },
      };
      else if (url.startsWith('/api/manager/vision-status')) data = {
        inference_ready: inferenceReady, max_media_bytes: 1024, max_image_bytes: 1024,
        max_video_bytes: 4096, upload_chunk_bytes: 512, notice: inferenceReady ? '已就绪' : '尚未就绪',
        optional_model_status: optionalModelStatus,
      };
      else if (url.startsWith('/api/manager/vision-jobs')) data = { jobs: visionJobs };
      else if (url.startsWith('/api/road-conditions')) data = { conditions: roadEvents };
      return { ok: true, json: async () => ({ data }) };
    },
  };
  vm.runInNewContext(source, context);
  return {
    elements, intervalCallbacks, mapListeners, map, mapLayers, removedMapLayers, requested, calls, document, window,
    createdObjectUrls, revokedObjectUrls, formEntries, sessionValues, getElement: element,
    setInferenceReady: (value) => { inferenceReady = value; },
    setOptionalModelStatus: (value) => { optionalModelStatus = value; },
    setVisionJobs: (items) => { visionJobs = items; },
    setVisionScenes: (items) => { visionScenes = items; },
    setRoadEvents: (items) => { roadEvents = items; },
    setVisionPublicationAlreadyExists: (value) => { visionPublicationAlreadyExists = value; },
    setVisionStatusFailure: (value) => { visionStatusFailure = value; },
    setSessionExpired: (value) => { sessionExpired = value; },
    setStorageFailure: (value) => { storageFailure = value; },
    deferImpactPreviews: () => { deferImpactPreviews = true; },
    hasPendingImpactPreview: (type) => impactPreviewResolvers.some((item) => item.type === type),
    resolveImpactPreview: (type, roadName) => {
      const index = impactPreviewResolvers.findIndex((item) => item.type === type);
      if (index < 0) throw new Error('No pending impact preview for ' + type);
      const item = impactPreviewResolvers.splice(index, 1)[0];
      const preview = {
        road_name: roadName, affected_road_segments: 1, affected_length_m: 80,
        event_scope_note: '预览快照', route_sample_scope: '局部示例',
        effects_by_mode: {}, sample_routes: {},
      };
      item.resolve({ ok: true, json: async () => ({ data: {
        snap: { road_name: roadName, chain_length_m: 80, edges: [] },
        impact_preview: preview,
      } }) });
    },
    getVisionPostCount: () => visionPostCount,
    getVideoChunkSizes: () => videoChunkSizes.slice(),
    releaseVisionUploads: () => uploadResolvers.splice(0).forEach((resolve) => resolve({
      ok: true, json: async () => ({ data: { job: { job_id: 'job-1' } } }),
    })),
  };
}

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
}

test('manager media summary has responsive preview styling and no rough-location control', () => {
  assert.match(managerHtml, /id="selected-media"[^>]*aria-live="polite"/);
  assert.match(managerCss, /\.selected-media-preview\{[^}]*object-fit:contain/);
  assert.match(managerCss, /\.selected-media-copy strong\{[^}]*overflow-wrap:anywhere/);
  assert.match(managerHtml, /manager\.css\?v=20261013-visionroute1/);
  assert.match(managerHtml, /manager\.js\?v=20261013-visionfeedback1/);
  assert.match(managerHtml, /id="dji-subtitle-file"[^>]*accept="\.srt,text\/plain"/);
  assert.match(managerHtml, /机身 GPS 不作为地面位置或道路绑定依据/);
  assert.match(managerHtml, /id="show-dji-track"[^>]*hidden/);
  assert.match(managerCss, /\.dji-track-preview-button/);
  assert.match(managerCss, /\.vision-region-mapping/);
  assert.match(managerHtml, /value="road_surface"/);
  assert.match(managerHtml, /id="pick-anchor"/);
  assert.match(managerHtml, /id="start-region"/);
  assert.match(managerHtml, /id="observation-region-list"/);
  assert.match(managerHtml, /id="vision-scene-select"/);
  assert.match(managerHtml, /id="save-vision-scene"/);
  assert.match(managerHtml, /id="load-vision-scene"/);
  assert.doesNotMatch(managerHtml, /id="restore-vision-results"/);
});

test('saved observation scene loads only after frame check and lets manager rebind its road', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  harness.setVisionScenes([{
    scene_id: 'scene-1', name: '工学部东门机位', frame_signature: '0000000000000000',
    anchor_gcj: { lng: 114.36, lat: 30.54 }, camera_stabilized: true,
    observation_regions: [{
      id: 'lane-1', kind: 'vehicle_lane', polygon: [[0, 0], [1, 0], [1, 1]],
      road_point_gcj: { lng: 114.36, lat: 30.54 },
      road_link: { road_name: '旧道路名', edges: [[1, 2, 0]] },
    }],
  }]);
  elements.get('media-file').files = [{ type: 'image/png', name: 'drone.png', size: 50 }];
  elements.get('media-file').listeners.change();
  elements.get('observation-image').hidden = false;
  elements.get('observation-image').naturalWidth = 100;
  elements.get('vision-scene-select').value = 'scene-1';
  await elements.get('refresh-vision-scenes').listeners.click();
  await elements.get('load-vision-scene').listeners.click();

  assert.match(elements.get('vision-message').textContent, /画面初筛匹配/);
  const regionRoot = elements.get('observation-region-list');
  assert.match(descendants(regionRoot).map((node) => node.textContent).join(' '), /旧道路名/);
  const bind = descendants(regionRoot).find((node) => node.className === 'region-bind-road');
  bind.listeners.click();
  await mapListeners.click({ latlng: { lng: 114.3601, lat: 30.5401 } });
  assert.match(elements.get('vision-message').textContent, /绑定道路：测试路/);
  assert.ok(harness.calls.some((call) => call.url.includes('/api/road-conditions/snap?lng=114.3601')));
  harness.getElement('vision-scene-name').value = '工学部东门机位';
  await elements.get('save-vision-scene').listeners.click();
  const updateCall = harness.calls.filter((call) => call.url === '/api/manager/vision-scenes' && call.options.method === 'POST').pop();
  assert.equal(JSON.parse(updateCall.options.body).scene_id, 'scene-1');
  assert.equal(JSON.parse(updateCall.options.body).observation_regions[0].road_link.road_name, '测试路');
  const clear = descendants(regionRoot).find((node) => node.className === 'region-clear-road');
  assert.ok(clear, 'bound road should offer a clear action');
  clear.listeners.click();
  assert.doesNotMatch(descendants(regionRoot).map((node) => node.textContent).join(' '), /测试路/);
  const rebind = descendants(regionRoot).find((node) => node.className === 'region-bind-road');
  rebind.listeners.click();
  await mapListeners.click({ latlng: { lng: 114.3601, lat: 30.5401 } });
  await elements.get('save-vision-scene').listeners.click();
  const reboundCall = harness.calls.filter((call) => call.url === '/api/manager/vision-scenes' && call.options.method === 'POST').pop();
  assert.equal(JSON.parse(reboundCall.options.body).observation_regions[0].road_link.road_name, '测试路');
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  elements.get('captured-at').value = '2026-10-10T10:00';
  const submitting = elements.get('vision-form').listeners.submit({ preventDefault() {} });
  await flush();
  assert.ok(harness.formEntries.some(([name, value]) => name === 'scene_id' && value === 'scene-1'));
  harness.releaseVisionUploads();
  await submitting;
});

test('a changed camera view is rejected before saved observation regions are loaded', async () => {
  const harness = managerHarness();
  await flush();
  const { elements } = harness;
  harness.setVisionScenes([{
    scene_id: 'scene-different-view', name: '另一机位', frame_signature: 'ffffffffffffffff',
    anchor_gcj: { lng: 114.36, lat: 30.54 }, camera_stabilized: false,
    observation_regions: [{ id: 'lane', kind: 'vehicle_lane', polygon: [[0, 0], [1, 0], [1, 1]] }],
  }]);
  elements.get('media-file').files = [{ type: 'image/png', name: 'other.png', size: 20 }];
  elements.get('media-file').listeners.change();
  elements.get('observation-image').hidden = false;
  elements.get('observation-image').naturalWidth = 10;
  elements.get('vision-scene-select').value = 'scene-different-view';
  await elements.get('refresh-vision-scenes').listeners.click();
  await elements.get('load-vision-scene').listeners.click();
  assert.match(elements.get('vision-message').textContent, /差异较大/);
  assert.equal(elements.get('observation-region-list').textContent, '尚未圈选区域');
});

test('manager can save a named scene with the current frame signature and road-linked region', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  elements.get('media-file').files = [{ type: 'image/png', name: 'fixed-camera.png', size: 20 }];
  elements.get('media-file').listeners.change();
  elements.get('observation-image').hidden = false;
  elements.get('observation-image').naturalWidth = 100;
  elements.get('pick-anchor').listeners.click();
  mapListeners.click({ latlng: { lng: 114.36, lat: 30.54 } });
  const canvas = elements.get('observation-canvas');
  canvas.width = 100; canvas.height = 100;
  harness.getElement('observation-kind').value = 'exclude';
  elements.get('start-region').listeners.click();
  canvas.listeners.click.call(canvas, { clientX: 10, clientY: 10 });
  canvas.listeners.click.call(canvas, { clientX: 80, clientY: 10 });
  canvas.listeners.click.call(canvas, { clientX: 80, clientY: 80 });
  elements.get('finish-region').listeners.click();
  harness.getElement('vision-scene-name').value = '工学部固定机位';
  await elements.get('save-vision-scene').listeners.click();

  const save = harness.calls.find((call) => call.url === '/api/manager/vision-scenes' && call.options.method === 'POST');
  assert.ok(save);
  const body = JSON.parse(save.options.body);
  assert.equal(body.frame_signature, '0000000000000000');
  assert.equal(body.anchor_gcj.lng, 114.36);
  assert.equal(body.observation_regions.length, 1);
  assert.equal(body.observation_regions[0].kind, 'exclude');
  assert.match(elements.get('vision-message').textContent, /场景已保存/);
});

test('selecting an image immediately enables recognition without a map point', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  const selected = harness.getElement('selected-media');
  selected.hidden = true;
  const file = { type: 'image/png', name: '珞珈道路.png', size: 768 };
  harness.elements.get('media-file').files = [file];
  harness.elements.get('media-file').listeners.change();

  assert.equal(selected.hidden, false);
  assert.equal(harness.elements.get('selected-media-name').textContent, '珞珈道路.png');
  assert.match(harness.elements.get('selected-media-meta').textContent, /768 B.*图片/);
  assert.equal(harness.elements.get('selected-media-preview').src, 'blob:preview-0');
  assert.match(harness.elements.get('vision-message').textContent, /可以开始识别.*复核后选择具体道路/);
  assert.equal(harness.elements.get('submit-vision').disabled, false);
});

test('a matching DJI SRT fills capture time locally and never uses aircraft GPS as a road location', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, calls } = harness;
  const originalAnchorText = harness.getElement('selected-anchor').textContent;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_1081.mp4', size: 100 }];
  elements.get('media-file').listeners.change();
  const subtitle = {
    name: 'DJI_1081.SRT', size: 300,
    text: async () => [
      '2025-06-15 15:47:50.149 [latitude:30.5] [longitude:114.3]',
      '2025-06-15 15:47:50.182 [latitude:30.5] [longitude:114.3]',
    ].join('\n'),
  };
  elements.get('dji-subtitle-file').files = [subtitle];
  await elements.get('dji-subtitle-file').listeners.change();

  assert.equal(elements.get('captured-at').value, '2025-06-15T15:47:50');
  assert.match(elements.get('dji-subtitle-status').textContent, /读到 2 个机身 GPS 位置点/);
  assert.match(elements.get('dji-subtitle-status').textContent, /不能表示镜头拍到的地面位置/);
  assert.equal(harness.getElement('selected-anchor').textContent, originalAnchorText);
  assert.equal(calls.some((call) => call.url.includes('/api/road-conditions/snap?')), false);
  assert.equal(calls.some((call) => call.options && JSON.stringify(call.options.body || '').includes('30.5')), false);

  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_1081.SRT', size: 10, text: async () => 'unreadable subtitle',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.equal(elements.get('captured-at').value, '');
  assert.match(elements.get('dji-subtitle-status').textContent, /没有找到有效的拍摄时间/);
});

test('DJI subtitle capture provenance and browser timezone are sent with resumable video upload', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  const { elements, calls } = harness;
  elements.get('media-file').files = [{
    type: 'video/mp4', name: 'DJI_1081.mp4', size: 100, lastModified: 7,
    slice(start, end) { return { size: end - start, arrayBuffer() { return Promise.resolve(new ArrayBuffer(end - start)); } }; },
  }];
  await elements.get('media-file').listeners.change();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_1081.SRT', size: 10,
    text: async () => '2025-06-15 15:47:50.149 [latitude:1] [longitude:2]',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  await elements.get('vision-form').listeners.submit({ preventDefault() {} });

  const startRequest = calls.find((call) => call.url === '/api/manager/vision-uploads' && call.options.method === 'POST');
  const metadata = JSON.parse(startRequest.options.body);
  assert.equal(metadata.capture_time_source, 'dji_srt');
  assert.ok(metadata.capture_timezone.length > 0);
});

test('DJI subtitle flight GPS is an explicit map hint and never becomes a road location', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapLayers, calls } = harness;
  const showTrack = harness.getElement('show-dji-track');
  showTrack.hidden = true;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_flight.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_flight.SRT', size: 100,
    text: async () => [
      '2026-10-12 10:00:00.000 [latitude: 30.536000] [longitude: 114.363000]',
      '2026-10-12 10:00:01.000 [latitude: 30.537000] [longitude: 114.364000]',
    ].join('\n'),
  }];
  await elements.get('dji-subtitle-file').listeners.change();

  assert.equal(showTrack.hidden, false);
  assert.match(elements.get('dji-subtitle-status').textContent, /虚线航迹.*不能表示镜头拍到的地面位置/);
  assert.equal(mapLayers.some((layer) => layer.options && layer.options.dashArray === '6 7'), false,
    'reading a local subtitle must not move the map before an explicit action');
  assert.equal(harness.map.lastFitBounds, undefined);

  await showTrack.listeners.click();
  const track = mapLayers.find((layer) => layer.kind === 'polyline' && layer.options && layer.options.dashArray === '6 7');
  assert.ok(track, 'explicit action should draw a distinct dashed aircraft trace');
  assert.ok(Math.abs(track.coordinates[0][0] - 30.5336732) < 1e-6
    && Math.abs(track.coordinates[0][1] - 114.3685263) < 1e-6,
  'WGS-84 GPS must use the project GCJ-02 conversion before rendering');
  assert.ok(track.options.interactive === false, 'the aircraft trace must not act as a road-picking target');
  assert.ok(harness.map.lastFitBounds, 'map should center on the trace only after the explicit action');
  assert.match(elements.get('dji-subtitle-status').textContent, /机身航迹仅供查看无人机飞行位置/);
  assert.equal(calls.some((call) => call.url.startsWith('/api/road-conditions/snap')), false,
    'aircraft GPS must never be snapped to the road network');

  await showTrack.listeners.click();
  assert.ok(harness.removedMapLayers.includes(track), 'the trace should be removable');
  assert.equal(showTrack.textContent, '在地图上查看无人机航迹');
  await showTrack.listeners.click();
  const visibleTrack = mapLayers.find((layer) => layer.kind === 'polyline' && layer.options && layer.options.className === 'dji-flight-track');
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'another-flight.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  assert.equal(showTrack.hidden, true, 'changing the selected video must clear the prior flight track');
  assert.ok(harness.removedMapLayers.includes(visibleTrack));
});

test('DJI aircraft marker follows SRT timecodes while the local video is scrubbed', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapLayers, removedMapLayers } = harness;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_timed.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_timed.SRT', size: 200,
    text: async () => [
      '1', '00:00:00,000 --> 00:00:00,200',
      '[latitude: 30.536000] [longitude: 114.363000]', '',
      '2', '00:00:01,000 --> 00:00:01,200',
      '[latitude: 30.537000] [longitude: 114.364000]', '',
    ].join('\n'),
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.match(elements.get('dji-subtitle-status').textContent, /字幕含视频时间码.*播放或拖动视频/);

  await elements.get('show-dji-track').listeners.click();
  const marker = mapLayers.find((layer) => layer.options && layer.options.className === 'dji-flight-current-position');
  assert.ok(marker, 'the current video time should select the nearest timed GPS position');
  const firstPosition = [...marker.coordinates];
  elements.get('observation-video').currentTime = 1.05;
  elements.get('observation-video').listeners.timeupdate();
  assert.notDeepEqual(marker.coordinates, firstPosition, 'scrubbing must move the aircraft hint on the map');

  elements.get('observation-video').currentTime = 10;
  elements.get('observation-video').listeners.timeupdate();
  assert.ok(removedMapLayers.includes(marker), 'do not carry the last coordinate across a large subtitle gap');
});

test('DJI SRT time range outside video duration warns even when metadata loads after the subtitle', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, window } = harness;
  window.URL = {
    createObjectURL: () => 'blob:test-observation-video',
    revokeObjectURL() {},
  };
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_offset.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_offset.SRT', size: 200,
    text: async () => [
      '1', '00:00:08,000 --> 00:00:08,200',
      '[latitude: 30.536000] [longitude: 114.363000]', '',
      '2', '00:00:09,000 --> 00:00:09,200',
      '[latitude: 30.537000] [longitude: 114.364000]', '',
    ].join('\n'),
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.doesNotMatch(elements.get('dji-subtitle-status').textContent, /时间范围与视频时长不重叠/);

  elements.get('observation-video').duration = 2;
  elements.get('observation-video').listeners.loadedmetadata();
  assert.match(elements.get('dji-subtitle-status').textContent, /字幕时间范围与视频时长不重叠/);
  assert.match(elements.get('dji-subtitle-status').textContent, /请核对视频和字幕是否配套/);
  assert.equal(elements.get('show-dji-track').hidden, false, 'keep the static aircraft-track preview available');
});

test('DJI GPS-only subtitle can preview one valid position without inventing a capture time', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapLayers } = harness;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'flight-gps.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '2026-10-12T10:30:00';
  elements.get('dji-subtitle-file').files = [{
    name: 'flight-gps.SRT', size: 100,
    text: async () => [
      '[longitude: 114.363000] [latitude: 30.536000]',
      '[latitude: 30.500000] [longitude: 200.000000]',
      '[latitude: 0] [longitude: 0]',
    ].join('\n'),
  }];
  await elements.get('dji-subtitle-file').listeners.change();

  assert.equal(elements.get('captured-at').value, '2026-10-12T10:30:00');
  assert.match(elements.get('dji-subtitle-status').textContent, /字幕没有有效拍摄时间，保留已填写的时间/);
  assert.match(elements.get('dji-subtitle-status').textContent, /读到 1 个机身 GPS 位置点/);
  await elements.get('show-dji-track').listeners.click();
  assert.equal(mapLayers.filter((layer) => layer.kind === 'circleMarker' && layer.options && layer.options.interactive === false).length, 1);
  assert.equal(mapLayers.some((layer) => layer.options && layer.options.className === 'dji-flight-track'), false,
    'one GPS point should be a marker, not a fabricated line');
});

test('long DJI subtitles sample the whole flight and retain its final position', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapLayers } = harness;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'long-flight.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  const rows = Array.from({ length: 20004 }, (_item, index) =>
    `[latitude: 30.536000] [longitude: ${114.363 + index / 1000000}]`);
  rows.push('[latitude: 30.537000] [longitude: 114.400000]');
  elements.get('dji-subtitle-file').files = [{
    name: 'long-flight.SRT', size: 100,
    text: async () => '2026-10-12 10:00:00.000 ' + rows.join('\n'),
  }];
  await elements.get('dji-subtitle-file').listeners.change();

  assert.match(elements.get('dji-subtitle-status').textContent, /读到 20005 个机身 GPS 位置点/);
  assert.match(elements.get('dji-subtitle-status').textContent, /抽样预览/);
  await elements.get('show-dji-track').listeners.click();
  const track = mapLayers.find((layer) => layer.kind === 'polyline' && layer.options && layer.options.className === 'dji-flight-track');
  assert.ok(track);
  assert.ok(track.coordinates.length <= 1200);
  assert.ok(Math.abs(track.coordinates.at(-1)[0] - 30.5347041) < 1e-6);
  assert.ok(Math.abs(track.coordinates.at(-1)[1] - 114.4055430) < 1e-6);
});

test('DJI subtitle mismatch or unreadable contents leave manual capture time and upload available', async () => {
  const harness = managerHarness();
  await flush();
  const { elements } = harness;
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'flight-a.mp4', size: 100 }];
  elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '2026-10-12T09:30';
  elements.get('dji-subtitle-file').files = [{
    name: 'flight-b.SRT', size: 10, text: async () => '2025-06-15 15:47:50.149 [latitude:1] [longitude:2]',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.equal(elements.get('captured-at').value, '2026-10-12T09:30');
  assert.match(elements.get('dji-subtitle-status').textContent, /文件名与视频不一致/);
  assert.equal(elements.get('submit-vision').disabled, false);

  elements.get('dji-subtitle-file').files = [{
    name: 'flight-a.SRT', size: 10, text: async () => 'not a DJI subtitle',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.equal(elements.get('captured-at').value, '2026-10-12T09:30');
  assert.match(elements.get('dji-subtitle-status').textContent, /没有找到有效的拍摄时间/);
  assert.equal(elements.get('submit-vision').disabled, false);
});

test('manual capture time stays authoritative after DJI subtitle auto-fill', async () => {
  const harness = managerHarness();
  await flush();
  const { elements } = harness;
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_1081.mp4', size: 100 }];
  elements.get('media-file').listeners.change();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_1081.SRT', size: 10, text: async () => '2025-06-15 15:47:50.149 [latitude:1] [longitude:2]',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  elements.get('captured-at').value = '2026-10-12T08:00';
  elements.get('captured-at').listeners.input();
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_1081.SRT', size: 10, text: async () => '2025-06-15 16:00:00.149 [latitude:1] [longitude:2]',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.equal(elements.get('captured-at').value, '2026-10-12T08:00');
  assert.match(elements.get('dji-subtitle-status').textContent, /保留手动填写的拍摄时间/);
});

test('a DJI subtitle selected before its video is applied once the matching video is selected', async () => {
  const harness = managerHarness();
  await flush();
  const { elements } = harness;
  elements.get('dji-subtitle-file').files = [{
    name: 'DJI_1081.SRT', size: 10, text: async () => '2025-06-15 15:47:50.149 [latitude:x] [longitude:y]',
  }];
  await elements.get('dji-subtitle-file').listeners.change();
  assert.match(elements.get('dji-subtitle-status').textContent, /先选择视频/);
  elements.get('media-file').files = [{ type: 'video/mp4', name: 'DJI_1081.mp4', size: 100 }];
  await elements.get('media-file').listeners.change();
  assert.equal(elements.get('captured-at').value, '2025-06-15T15:47:50');
  assert.doesNotMatch(elements.get('dji-subtitle-status').textContent, /GPS 记录/);
});

test('an oversized file is acknowledged but cannot be submitted', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  harness.getElement('selected-media').hidden = true;
  harness.elements.get('media-file').files = [
    { type: 'image/jpeg', name: 'too-large.jpg', size: 2048 },
  ];
  harness.elements.get('media-file').listeners.change();

  assert.equal(harness.elements.get('selected-media').hidden, false);
  assert.match(harness.elements.get('vision-message').textContent, /超过.*1\.0 KB/);
  assert.equal(harness.elements.get('submit-vision').disabled, true);
});

test('reselecting and logging out release local image previews', async () => {
  const harness = managerHarness();
  await flush();
  const input = harness.elements.get('media-file');
  harness.getElement('selected-media').hidden = true;
  input.files = [{ type: 'image/png', name: 'first.png', size: 10 }];
  input.listeners.change();
  input.files = [{ type: 'image/jpeg', name: 'second.jpg', size: 10 }];
  input.listeners.change();
  assert.deepEqual(harness.revokedObjectUrls, ['blob:preview-0']);

  await harness.elements.get('logout').listeners.click();
  assert.deepEqual(harness.revokedObjectUrls, ['blob:preview-0', 'blob:preview-1']);
  assert.equal(harness.elements.get('selected-media').hidden, true);
});

test('manager refreshes vision readiness and updates submission availability while open', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, intervalCallbacks } = harness;
  assert.equal(intervalCallbacks.length, 1);

  elements.get('media-file').files = [{ type: 'image/png', name: 'campus.png', size: 10 }];
  elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '2026-10-09T14:00';
  assert.equal(elements.get('submit-vision').disabled, true);

  harness.setInferenceReady(true);
  intervalCallbacks[0]();
  await flush();
  assert.equal(elements.get('submit-vision').disabled, false);

  harness.setVisionStatusFailure(true);
  intervalCallbacks[0]();
  await flush();
  assert.equal(elements.get('submit-vision').disabled, true);
  assert.equal(elements.get('vision-status').classList.contains('is-ready'), false);

  harness.setVisionStatusFailure(false);
  harness.setInferenceReady(false);
  intervalCallbacks[0]();
  await flush();
  assert.equal(elements.get('submit-vision').disabled, true);
});

test('readiness polling cannot re-enable or duplicate an in-flight media upload', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, intervalCallbacks } = harness;
  harness.setInferenceReady(true);
  intervalCallbacks[0]();
  await flush();

  elements.get('media-file').files = [{ type: 'image/png', name: 'campus.png', size: 10 }];
  elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '2026-10-09T14:00';
  assert.equal(elements.get('submit-vision').disabled, false);

  const upload = elements.get('vision-form').listeners.submit({ preventDefault() {} });
  await flush();
  try {
    intervalCallbacks[0]();
    await flush();
    assert.equal(elements.get('submit-vision').disabled, true);
    await elements.get('vision-form').listeners.submit({ preventDefault() {} });
    assert.equal(harness.getVisionPostCount(), 1);
  } finally {
    harness.releaseVisionUploads();
    await upload;
  }
});

test('successful media upload clears the preview and preserves the queued confirmation', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  const input = harness.elements.get('media-file');
  harness.getElement('selected-media').hidden = true;
  input.files = [{ type: 'image/png', name: 'campus.png', size: 10 }];
  input.listeners.change();
  harness.elements.get('captured-at').value = '2026-10-09T14:00';
  const upload = harness.elements.get('vision-form').listeners.submit({ preventDefault() {} });
  await flush();
  assert.deepEqual(harness.formEntries.map(([name]) => name), [
    'media', 'camera_stabilized', 'captured_at', 'capture_time_source', 'capture_timezone', 'observation_regions',
  ]);
  assert.equal(harness.formEntries.find(([name]) => name === 'capture_time_source')[1], 'manual');
  assert.equal(typeof harness.formEntries.find(([name]) => name === 'capture_timezone')[1], 'string');
  assert.ok(harness.formEntries.find(([name]) => name === 'capture_timezone')[1].length > 0);
  harness.releaseVisionUploads();
  await upload;

  assert.equal(harness.getElement('selected-media').hidden, true);
  assert.deepEqual(harness.revokedObjectUrls, ['blob:preview-0']);
  assert.match(harness.elements.get('vision-message').textContent, /影像任务已入队/);
});

test('manager impact preview discloses its active-condition snapshot and local scope', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  elements.get('event-type').value = 'closure';
  elements.get('pick-road').listeners.click();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });

  const preview = elements.get('impact-preview');
  assert.equal(preview.hidden, false);
  const note = preview.children.find((child) => child.tagName === 'small');
  const noteText = note ? note.textContent : preview.children.at(-1).textContent;
  assert.match(noteText, /基于当前有效路况快照/);
  assert.match(noteText, /2 条当前生效路况/);
  assert.match(noteText, /所选路段两端/);
});

test('a selected road can be explicitly cleared before publishing', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  const clearButton = harness.getElement('clear-picked-road');
  const confirmButton = harness.getElement('confirm-picked-road');
  clearButton.hidden = true;
  confirmButton.hidden = true;
  elements.get('event-type').value = 'closure';
  elements.get('event-action').value = 'closure';
  elements.get('event-mode-walk').checked = true;
  elements.get('event-name').value = '测试路封闭';
  elements.get('pick-road').listeners.click();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });

  assert.equal(clearButton.hidden, false, 'clear action should appear after selecting a snapped road');
  assert.equal(confirmButton.hidden, false, 'the snapped road should require explicit confirmation');
  assert.match(elements.get('selected-road').textContent, /测试路.*吸附距离 2\.0 米/);
  assert.equal(elements.get('publish-event').disabled, true);
  const connector = harness.mapLayers.find((layer) => layer.kind === 'polyline' && layer.options && layer.options.dashArray === '4 5');
  assert.ok(connector, 'the map should show the click-to-snap distance');
  assert.equal(JSON.stringify(connector.coordinates), JSON.stringify([[30.53, 114.36], [30.53, 114.36002]]));
  assert.equal(harness.mapLayers.filter((layer) => layer.kind === 'circleMarker').length, 2);
  await elements.get('condition-form').listeners.submit({ preventDefault() {} });
  assert.equal(harness.calls.some((call) => call.url === '/api/road-conditions' && call.options.method === 'POST'), false);

  confirmButton.listeners.click();
  assert.equal(confirmButton.hidden, true);
  assert.equal(elements.get('publish-event').disabled, false);

  clearButton.listeners.click();

  assert.ok(harness.removedMapLayers.includes(connector));
  assert.equal(clearButton.hidden, true);
  assert.equal(elements.get('selected-road').textContent, '尚未选择道路');
  assert.equal(elements.get('publish-event').disabled, true);
  assert.equal(elements.get('impact-preview').hidden, true);
});

test('road event preview and publication use an action and affected travel modes', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners, calls } = harness;
  elements.get('event-type').value = 'construction';
  elements.get('event-type').listeners.change();
  elements.get('event-action').value = 'closure';
  elements.get('event-name').value = '测试路施工';
  const walkMode = elements.get('event-mode-walk');
  elements.get('event-action').listeners.change();
  elements.get('pick-road').listeners.click();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  elements.get('confirm-picked-road').listeners.click();

  assert.equal(elements.get('publish-event').disabled, true, 'at least one affected travel mode is required');

  walkMode.checked = true;
  await walkMode.listeners.change();
  assert.equal(elements.get('publish-event').disabled, false);
  const preview = calls.filter((call) => call.url.startsWith('/api/road-conditions/snap')).at(-1);
  assert.match(preview.url, /action=closure/);
  assert.match(preview.url, /modes=walk/);
  assert.doesNotMatch(preview.url, /modes=bike|modes=drive/);

  await elements.get('condition-form').listeners.submit({ preventDefault() {} });
  const posted = calls.find((call) => call.url === '/api/road-conditions' && call.options.method === 'POST');
  const body = JSON.parse(posted.options.body);
  assert.equal(body.action, 'closure');
  assert.deepEqual(body.affected_modes, ['walk']);
  assert.equal(Object.hasOwn(body, 'blocked_modes'), false);
});

test('manager reports optional model load failures accurately instead of claiming models are configured', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.setOptionalModelStatus({ accident: 'load_failed', flood: 'file_missing' });
  harness.intervalCallbacks[0]();
  await flush();

  const status = harness.elements.get('vision-status').textContent;
  assert.match(status, /事故模型权重未能加载/);
  assert.match(status, /积水模型权重文件缺失/);
});

test('manager explains optional models are intentionally disabled before release approval', async () => {
  const harness = managerHarness();
  await flush();
  harness.setInferenceReady(true);
  harness.setOptionalModelStatus({ accident: 'disabled', flood: 'disabled' });
  harness.intervalCallbacks[0]();
  await flush();

  const status = harness.elements.get('vision-status').textContent;
  assert.match(status, /事故模型尚未验收，暂不启用/);
  assert.match(status, /积水模型尚未验收，暂不启用/);
});

test('canceling route picking prevents a late snap response from selecting a road', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  const clearButton = harness.getElement('clear-picked-road');
  const selectedRoad = harness.getElement('selected-road');
  const publishButton = harness.getElement('publish-event');
  clearButton.hidden = true;
  selectedRoad.textContent = '尚未选择道路';
  publishButton.disabled = true;
  elements.get('event-type').value = 'closure';
  harness.deferImpactPreviews();
  elements.get('pick-road').listeners.click();
  const pending = mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  await flush();

  elements.get('pick-road').listeners.click();
  harness.resolveImpactPreview('closure', '迟到路段');
  await pending;

  assert.equal(clearButton.hidden, true);
  assert.equal(selectedRoad.textContent, '尚未选择道路');
  assert.equal(publishButton.disabled, true);
});

test('a snap response refreshes its impact preview when the event type changed while waiting', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  harness.deferImpactPreviews();
  elements.get('event-type').value = 'closure';
  elements.get('pick-road').listeners.click();
  const pendingSnap = mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  await flush();

  elements.get('event-type').value = 'accident';
  await elements.get('event-type').listeners.change();
  harness.resolveImpactPreview('closure', '旧事件类型预览');
  await pendingSnap;

  assert.equal(harness.hasPendingImpactPreview('accident'), true);
  harness.resolveImpactPreview('accident', '当前事件类型预览');
  await flush();
  const previewText = elements.get('impact-preview').children.map((child) => child.textContent).join(' ');
  assert.match(previewText, /当前事件类型预览/);
  assert.doesNotMatch(previewText, /旧事件类型预览/);
});

test('a late event-type preview cannot overwrite the newer selection', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  elements.get('pick-road').listeners.click();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  elements.get('confirm-picked-road').listeners.click();

  harness.deferImpactPreviews();
  elements.get('event-type').value = 'accident';
  const older = elements.get('event-type').listeners.change();
  elements.get('event-type').value = 'closure';
  const newer = elements.get('event-type').listeners.change();
  harness.resolveImpactPreview('closure', '新选择的道路');
  await newer;
  harness.resolveImpactPreview('accident', '旧选择的道路');
  await older;

  const previewText = elements.get('impact-preview').children.map((child) => child.textContent).join(' ');
  assert.match(previewText, /新选择的道路/);
  assert.doesNotMatch(previewText, /旧选择的道路/);
});

test('a pending preview safely settles after the selected road is cleared', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  elements.get('pick-road').listeners.click();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  elements.get('confirm-picked-road').listeners.click();

  harness.deferImpactPreviews();
  elements.get('event-type').value = 'closure';
  const pending = elements.get('event-type').listeners.change();
  elements.get('clear-picked-road').listeners.click();
  harness.resolveImpactPreview('closure', '已清除路段');

  await assert.doesNotReject(pending);
  assert.equal(elements.get('impact-preview').hidden, true);
});

test('videos are sent through resumable chunks without requiring a map anchor', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, calls } = harness;
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();

  elements.get('media-file').files = [{
    type: 'video/mp4', name: 'campus.mp4', size: 2050, lastModified: 7,
    slice(start, end) { return { size: end - start, arrayBuffer() { return Promise.resolve(new ArrayBuffer(end - start)); } }; },
  }];
  elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '2026-10-09T14:00';
  await elements.get('vision-form').listeners.submit({ preventDefault() {} });

  assert.deepEqual(harness.getVideoChunkSizes(), [512, 512, 512, 512, 2]);
  assert.equal(calls.some((call) => call.url === '/api/manager/vision-jobs' && call.options.method === 'POST'), false);
  const startRequest = calls.find((call) => call.url === '/api/manager/vision-uploads' && call.options.method === 'POST');
  const metadata = JSON.parse(startRequest.options.body);
  assert.equal(metadata.lng, undefined);
  assert.equal(metadata.capture_time_source, 'manual');
  assert.ok(metadata.capture_timezone.length > 0);
  assert.equal(elements.get('vision-message').textContent.includes('已入队'), true);
});

test('images without an actual capture time are not submitted', async () => {
  const harness = managerHarness();
  await flush();
  const { elements, mapListeners } = harness;
  harness.setInferenceReady(true);
  harness.intervalCallbacks[0]();
  await flush();
  elements.get('media-file').files = [{ type: 'image/png', name: 'drone.png', size: 10 }];
  elements.get('media-file').listeners.change();
  elements.get('captured-at').value = '';
  elements.get('pick-anchor').listeners.click();
  mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  await elements.get('vision-form').listeners.submit({ preventDefault() {} });
  assert.equal(harness.calls.some((call) => call.url === '/api/manager/vision-jobs'), false);
  assert.match(elements.get('vision-message').textContent, /实际拍摄时间/);
});

test('completed image jobs show vehicle counts and safely render detector boxes', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'vision-1', status: 'completed', media_kind: 'image', media_url: '/private/media',
    original_name: 'drone.png', result: {
      media: { width: 640, height: 480 },
      model: { id: 'visdrone-rtdetrv4-s', version: 'abc123' },
      metrics: { peak_vehicle_count: 2, peak_class_counts: { car: 2 }, frames_analyzed: 1 },
      candidates: [],
      preview_detections: [
        { label: 'car', confidence: 0.91, box: [10, 20, 100, 90] },
        { label: '<script>alert(1)</script>', confidence: 0.4, box: [120, 40, 200, 110] },
      ],
    },
  }]);
  const { elements, intervalCallbacks } = harness;
  intervalCallbacks[0]();
  await flush();
  const card = elements.get('vision-jobs').children[0];
  const stage = card.children.find((node) => node.className === 'vision-image-stage');
  const svg = stage.children.find((node) => node.tagName === 'svg');
  assert.equal(svg.attributes.viewBox, '0 0 640 480');
  assert.equal(svg.children.filter((node) => node.tagName === 'rect').length, 1);
  assert.equal(svg.children.find((node) => node.tagName === 'text').textContent, 'car 91%');
  assert.doesNotMatch(svg.children.map((node) => node.textContent).join(' '), /script|alert/);
  const summary = card.children.find((node) => node.className === 'vision-metrics');
  const summaryText = summary.children.map((node) => node.textContent).join(' ');
  assert.match(summaryText, /检出车辆 2/);
  assert.match(summaryText, /事故场景模型 未启用/);
  assert.match(summaryText, /路面积水分割模型 未启用/);
});

test('running video progress compares analyzed samples with the sampled-frame total', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'running-video', status: 'running', media_kind: 'video',
    original_name: 'campus.mp4', capture_time_source: 'dji_srt', capture_timezone: 'Asia/Shanghai', progress: {
      phase: 'analyzing', percent: 42, frames_analyzed: 63, total_frames: 18000,
      total_sampled_frames: 3000, analyzed_through_seconds: 126, duration_seconds: 600,
    },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const rendered = descendants(harness.elements.get('vision-jobs').children[0]);
  const progress = rendered.find((node) => node.className === 'vision-job-meta'
    && /抽样帧/.test(node.textContent || ''));
  assert.ok(progress);
  const captureMeta = rendered.find((node) => node.className === 'vision-job-meta'
    && /时间来自/.test(node.textContent || ''));
  assert.ok(captureMeta);
  assert.match(captureMeta.textContent, /DJI 字幕.*Asia\/Shanghai/);
  assert.match(progress.textContent, /63\s*\/\s*3000 个抽样帧/);
  assert.match(progress.textContent, /已分析到 126\.0 秒/);
  assert.doesNotMatch(progress.textContent, /18000/);
});

test('moving camera results ask the manager to recheck the observation region', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'moving-video', status: 'completed', media_kind: 'video', original_name: 'moving.mp4',
    observation_regions: [{ id: 'east-lane', kind: 'vehicle_lane' }],
    result: {
      metrics: {
        peak_vehicle_count: 3, motion_assessment: 'camera_motion_compensated',
        observation_region_mapping_valid_ratios: { 'east-lane': 0.63 },
      },
      safety: { observation_region_reassociation_required: true },
      candidates: [],
    },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const card = harness.elements.get('vision-jobs').children[0];
  const summary = card.children.find((node) => node.className === 'vision-metrics');
  const summaryText = descendants(summary).map((node) => node.textContent || '').join(' ');
  assert.match(summaryText, /镜头移动后仍需人工核对观察区域与道路的对应关系/);
  assert.match(summaryText, /轨迹运动补偿通过/);
  assert.match(summaryText, /east-lane/);
  assert.match(summaryText, /63\.0%/);
  assert.match(summaryText, /低于 80%/);
  assert.match(summaryText, /不会生成拥堵\/人流候选/);
  assert.match(summaryText, /地图定位精度/);
  const resubmit = descendants(summary).find((node) => node.tagName === 'button'
    && node.textContent === '重新圈选并重新分析');
  assert.ok(resubmit);
  resubmit.listeners.click();
  assert.equal(harness.elements.get('vision-upload-card').scrollCount, 1);
  assert.equal(harness.elements.get('media-file').clickCount, 1);
});

test('empty analysis distinguishes a mapping safety block from a model finding nothing', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'mapping-blocked', status: 'completed', media_kind: 'video', original_name: 'moving.mp4',
    observation_regions: [{ id: 'walkway', kind: 'pedestrian' }],
    result: {
      metrics: { observation_region_mapping_valid_ratios: { walkway: 0.41 } },
      safety: { observation_region_reassociation_required: true }, candidates: [],
    },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const rendered = descendants(harness.elements.get('vision-jobs').children[0]);
  const terminalText = rendered.map((node) => node.textContent || '').join(' ');
  assert.match(terminalText, /映射成功率低于 80%/);
  assert.match(terminalText, /不会基于相关区域生成拥堵或人群候选/);
  assert.doesNotMatch(terminalText, /模型未生成可复核的路况候选/);
});

test('short video with no candidates explains the temporal evidence limits and next step', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'short-video-empty', status: 'completed', media_kind: 'video', original_name: '1081.mp4',
    result: {
      metrics: { frames_analyzed: 10, mean_vehicle_count: 149.33 },
      video_coverage: { complete: true, duration_seconds: 2, analyzed_through_seconds: 1.8 },
      safety: {}, candidates: [],
    },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const nodes = descendants(harness.elements.get('vision-jobs').children[0]);
  const terminalText = nodes.map((node) => node.textContent || '').join(' ');
  assert.match(terminalText, /本段视频约 2 秒/);
  assert.match(terminalText, /缓行判断至少需要 15 秒连续证据/);
  assert.match(terminalText, /人群聚集判断至少需要 5 秒连续证据/);
  assert.match(terminalText, /画面车辆数只是抽样帧检测数量/);
  assert.match(terminalText, /建议上传至少 15 秒、尽量固定角度的视频/);
  assert.match(terminalText, /依据原片手工登记路况/);
});

function descendants(node) {
  return [node, ...(node.children || []).flatMap(descendants)];
}

test('completed jobs without candidates allow manual visual road registration and can be retried or deleted', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'finished-empty', status: 'completed', media_kind: 'image', original_name: '航拍.png',
    result: { metrics: { peak_vehicle_count: 3, peak_class_counts: { car: 3 } }, candidates: [] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const root = harness.elements.get('vision-jobs');
  const card = root.children[0];
  const nodes = descendants(card);
  const terminalText = nodes.map((node) => node.textContent || '').join(' ');
  const manual = nodes.find((node) => node.tagName === 'button' && node.textContent === '依据原片手工登记路况');
  const retry = nodes.find((node) => node.tagName === 'button' && node.textContent === '重新选择影像');
  const clear = nodes.find((node) => node.tagName === 'button' && node.textContent === '删除影像记录');
  assert.match(terminalText, /分析已结束/);
  assert.match(terminalText, /静态图片/);
  assert.match(terminalText, /不能判断车辆是否持续缓行或人群是否持续聚集/);
  assert.ok(manual);
  assert.ok(retry);
  assert.ok(clear);

  manual.listeners.click();
  assert.equal(harness.elements.get('vision-source-banner').hidden, false);
  assert.match(harness.elements.get('vision-source-label').textContent, /人工登记/);
  assert.equal(harness.elements.get('field-confirmation-wrap').hidden, false);
  assert.equal(harness.elements.get('field-confirmation').required, true);

  retry.listeners.click();
  assert.equal(harness.elements.get('vision-upload-card').scrollCount, 1);
  assert.equal(harness.elements.get('media-file').clickCount, 1);

  clear.listeners.click();
  await flush();
  assert.ok(harness.calls.some((call) => call.url === '/api/manager/vision-jobs/finished-empty' && call.options.method === 'DELETE'));
  assert.doesNotMatch(root.children.map((node) => node.textContent).join(' '), /航拍\.png/);
  harness.intervalCallbacks[0]();
  await flush();
  assert.doesNotMatch(root.children.map((node) => node.textContent).join(' '), /航拍\.png/);
});

test('permanent deletion does not depend on browser session storage', async () => {
  const harness = managerHarness();
  await flush();
  harness.setStorageFailure(true);
  harness.setVisionJobs([{
    job_id: 'memory-only', status: 'completed', media_kind: 'image', original_name: '临时结果.png',
    result: { metrics: {}, candidates: [] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();
  const root = harness.elements.get('vision-jobs');
  const clear = descendants(root).find((node) => node.tagName === 'button' && node.textContent === '删除影像记录');

  clear.listeners.click();
  await flush();
  harness.intervalCallbacks[0]();
  await flush();

  assert.doesNotMatch(descendants(root).map((node) => node.textContent || '').join(' '), /临时结果\.png/);
  assert.ok(harness.calls.some((call) => call.url === '/api/manager/vision-jobs/memory-only' && call.options.method === 'DELETE'));
});

test('confirmed visual evidence transfers to an explicitly verified road event', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e',
    status: 'needs_review', review_status: 'confirmed', media_kind: 'image',
    original_name: '巡查影像.png', anchor_gcj: { lng: 114.36, lat: 30.53 },
    observation_regions: [{ id: 'lane-east', kind: 'vehicle_lane', polygon: [[0.1, 0.2], [0.8, 0.2], [0.8, 0.6]] }],
    result: { candidates: [{ kind: 'vehicle_cluster_review', confidence: 0.8,
      evidence: { summary: { region_id: 'lane-east' } } }] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();
  const { elements, mapListeners, calls } = harness;
  const card = elements.get('vision-jobs').children[0];
  const transfer = descendants(card).find((node) => node.tagName === 'button' && node.textContent === '将此候选转入道路事件');
  assert.ok(transfer);
  transfer.listeners.click();

  assert.equal(elements.get('vision-source-banner').hidden, false);
  assert.match(elements.get('vision-source-label').textContent, /观察区域：lane-east（机动车道）/);
  assert.equal(elements.get('publish-event').disabled, true);
  assert.equal(elements.get('event-type').value, '');
  elements.get('event-type').value = 'event';
  elements.get('event-type').listeners.change();
  elements.get('event-action').value = 'notice';
  elements.get('event-name').value = '测试人流提示';
  elements.get('event-mode-walk').checked = true;
  elements.get('event-action').listeners.change();
  await mapListeners.click({ latlng: { lng: 114.36, lat: 30.53 } });
  assert.equal(elements.get('publish-event').disabled, true);

  elements.get('confirm-picked-road').listeners.click();
  elements.get('field-confirmation').value = '已联系现场负责人核实，该路段确有人流聚集。';
  elements.get('field-confirmation').listeners.input();
  assert.equal(elements.get('publish-event').disabled, false);
  assert.equal(elements.get('event-duration').value, '30');
  assert.equal(elements.get('event-ongoing-option').disabled, true);
  elements.get('event-duration').value = 'ongoing';
  elements.get('event-duration').listeners.change();
  assert.equal(elements.get('publish-event').disabled, true);
  assert.match(elements.get('event-time-summary').textContent, /影像来源事件必须设置结束时间/);
  await elements.get('condition-form').listeners.submit({ preventDefault() {} });
  assert.equal(calls.some((call) => call.url === '/api/road-conditions' && call.options.method === 'POST'), false);
  elements.get('event-duration').value = '30';
  elements.get('event-duration').listeners.change();
  harness.setVisionPublicationAlreadyExists(true);
  await elements.get('condition-form').listeners.submit({ preventDefault() {} });
  const posted = calls.find((call) => call.url === '/api/road-conditions' && call.options.method === 'POST');
  assert.ok(posted);
  const body = JSON.parse(posted.options.body);
  assert.equal(body.source_vision_job_id, 'a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e');
  assert.equal(body.source_vision_candidate_index, 0);
  assert.equal(body.field_confirmation, '已联系现场负责人核实，该路段确有人流聚集。');
  assert.equal(Date.parse(body.end_time) - Date.parse(body.start_time), 1800000);
  assert.match(elements.get('form-message').textContent, /已经关联到路况，没有重复新增事件/);
  assert.equal(elements.get('vision-source-banner').hidden, true);
});

test('candidate review shows congestion duration and image-space crowd evidence', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'review-evidence', status: 'needs_review', media_kind: 'video', original_name: '校园巡查.mp4',
    result: { candidates: [
      { kind: 'possible_congestion', confidence: 0.8, reason: '持续低位移', evidence: { summary: {
        region_id: 'lane-east', observed_duration_seconds: 18, mean_vehicle_count: 7.2,
        stationary_track_ratio: 0.75,
      } } },
      { kind: 'possible_crowding', confidence: 0.8, reason: '行人较多', evidence: {
        region_id: 'walkway-east', peak_person_count: 24,
        peak_people_with_nearby_peer_count: 18,
        observed_duration_seconds: 6,
        sustained_cluster_duration_seconds: 5,
        sustained_cluster_peak_size: 12,
        minimum_cluster_size: 8,
        movement_assessment: 'mostly_moving_in_image',
      } },
      { kind: 'possible_accident', confidence: 0.81, reason: '事故场景线索', evidence: {} },
    ] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const renderedText = descendants(harness.elements.get('vision-jobs'))
    .map((node) => node.textContent || '').join(' ');
  assert.match(renderedText, /连续观察 18\.0 秒/);
  assert.match(renderedText, /低位移车辆轨迹占 75%/);
  assert.match(renderedText, /峰值 24 人/);
  assert.match(renderedText, /连续观察 6\.0 秒/);
  assert.match(renderedText, /最大画面相邻组 12 人，最长连续 5\.0 秒（筛查门槛 8 人）/);
  assert.match(renderedText, /有邻近同伴的检测数 18 人/);
  assert.match(renderedText, /多数轨迹有画面位移/);
  assert.match(renderedText, /不代表地面密度或实际速度/);
  assert.match(renderedText, /人行区域人群聚集候选 · 规则筛查 · 需复核/);
  assert.doesNotMatch(renderedText, /人行区域人群聚集候选 · 置信度/);
  assert.match(renderedText, /可能存在车辆排队\/低速聚集 · 规则筛查 · 需复核/);
  assert.match(renderedText, /疑似事故场景（位置待复核） · 模型分数 81% · 需复核/);
});

test('road event deletion removes its record from the manager list after refresh', async () => {
  const harness = managerHarness();
  await flush();
  harness.setRoadEvents([{
    id: 'event-1', name: '已撤销事件', type: 'event', status: 'revoked',
    source: { kind: 'vision_job', job_id: 'a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e',
      region_id: 'lane-east', associated_edge: { road_name: '樱园西路' }, field_confirmation: '现场核查' },
    audit: [{ action: 'revoked', actor: 'web', at: 1791000000 }],
    edge: { road_name: '测试路', geometry_gcj: [[114.36, 30.53], [114.361, 30.53]] },
  }]);
  harness.elements.get('refresh-events').listeners.click();
  await flush();
  const row = harness.elements.get('event-list').children[0];
  const text = descendants(row).map((node) => node.textContent || '').join(' ');
  assert.match(text, /已撤销/);
  assert.match(text, /影像来源/);
  assert.match(text, /lane-east → 樱园西路/);
  const remove = descendants(row).find((node) => node.tagName === 'button' && node.textContent === '永久删除');
  assert.ok(remove);
  remove.listeners.click();
  await flush();
  assert.ok(harness.calls.some((call) => call.url === '/api/road-conditions/event-1' && call.options.method === 'DELETE'));
  assert.equal(harness.elements.get('event-list').children.length, 1);
  assert.match(harness.elements.get('event-list').children[0].textContent, /目前没有路况记录/);
});

test('old session-only cleared images are permanently deleted after manager login', async () => {
  const harness = managerHarness([['managerDismissedVisionJobs', '["old-hidden-job"]']]);
  await flush();
  assert.ok(harness.calls.some((call) => call.url === '/api/manager/vision-jobs/old-hidden-job' && call.options.method === 'DELETE'));
  assert.equal(harness.sessionValues.has('managerDismissedVisionJobs'), false);
});

test('expired manager session returns to login while polling', async () => {
  const harness = managerHarness();
  await flush();
  assert.equal(harness.elements.get('workspace').hidden, false);
  harness.setSessionExpired(true);
  harness.intervalCallbacks[0]();
  await flush();
  assert.equal(harness.elements.get('workspace').hidden, true);
  assert.equal(harness.elements.get('login-panel').hidden, false);
});

test('review of multiple visual candidates records the selected item and human note', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'job-multi', status: 'needs_review', media_kind: 'video', original_name: '巡查视频.mp4',
    result: { candidates: [
      { kind: 'possible_congestion', confidence: 0.72,
        evidence: { segments: [{ start_seconds: 12.5, end_seconds: 18.2 }] } },
      { kind: 'possible_accident', confidence: 0.81,
        evidence: { segments: [{ start_seconds: 43, end_seconds: 51.3 }] } },
    ] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();
  const card = harness.elements.get('vision-jobs').children[0];
  const note = descendants(card).find((node) => node.tagName === 'textarea');
  const confirms = descendants(card).filter((node) => node.tagName === 'button' && node.textContent === '确认此候选');
  assert.ok(note);
  assert.equal(confirms.length, 2);
  const evidenceButtons = descendants(card).filter((node) => node.tagName === 'button' && node.className === 'outline-button candidate-evidence-jump');
  assert.equal(evidenceButtons.length, 2);
  evidenceButtons[1].listeners.click();
  const video = descendants(card).find((node) => node.tagName === 'video');
  assert.equal(video.currentTime, 43);
  confirms[1].listeners.click();
  assert.equal(harness.calls.filter((call) => call.url.includes('/candidates/1/review')).length, 0);
  note.value = '逐帧核对发现明确的事故迹象，需现场继续确认。';
  confirms[1].listeners.click();
  await flush();
  const posted = harness.calls.find((call) => call.url.includes('/candidates/1/review'));
  assert.ok(posted);
  assert.equal(JSON.parse(posted.options.body).status, 'confirmed');
  assert.equal(JSON.parse(posted.options.body).note, note.value);
});

test('flood candidates show their limits and expose per-candidate review', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'job-flood', status: 'needs_review', media_kind: 'image', original_name: '雨后道路.png',
    result: { candidates: [{
      kind: 'possible_flooding', confidence: 0.82,
      reason: '圈选路面内出现疑似淹水像素。',
      evidence: { summary: { region_id: 'road-surface-east', max_flooded_road_area_ratio: 0.27, outline_polygons: [] } },
    }] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const card = harness.elements.get('vision-jobs').children[0];
  const cardText = descendants(card).map((node) => node.textContent || '').join(' ');
  assert.match(cardText, /疑似道路积水/);
  assert.match(cardText, /27%/);
  assert.match(cardText, /不是水深或实际淹水面积/);
  assert.match(cardText, /观察区域：road-surface-east/);
  assert.equal(descendants(card).filter((node) => node.tagName === 'button' && node.textContent === '确认此候选').length, 1);
});

test('visible-water candidates do not present the prediction as a confirmed flood', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'job-visible-water', status: 'needs_review', media_kind: 'image', original_name: '航拍.png',
    result: { candidates: [{
      kind: 'possible_flooding', confidence: 0.82,
      evidence: { summary: {
        region_id: 'road-surface-east', water_evidence_type: 'visible_water',
        max_visible_water_area_ratio: 0.27, outline_polygons: [],
      } },
    }] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const cardText = descendants(harness.elements.get('vision-jobs').children[0])
    .map((node) => node.textContent || '').join(' ');
  assert.match(cardText, /路面区域可见水体线索/);
  assert.match(cardText, /需人工核对/);
  assert.doesNotMatch(cardText, /路面区域可见水体线索 · 置信度/);
  assert.match(cardText, /可见水体像素约占 27%/);
  assert.match(cardText, /无法区分路面积水与其他可见水体/);
});

test('video candidate cards play their bounded manager-only evidence clip', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: '12345678-1234-4234-8234-123456789abc', status: 'needs_review', media_kind: 'video',
    original_name: '巡查视频.mp4',
    media_url: '/private/original',
    result: { candidates: [{
      kind: 'possible_congestion', confidence: 0.74,
      evidence: {
        segments: [{ start_seconds: 12, end_seconds: 18 }],
        clips: [{ media_url: '/api/manager/vision-jobs/12345678-1234-4234-8234-123456789abc/media?clip=0:0', start_seconds: 10, end_seconds: 20 }],
      },
    }] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();

  const card = harness.elements.get('vision-jobs').children[0];
  const videos = descendants(card).filter((node) => node.tagName === 'video');
  const evidenceVideo = videos.find((node) => node.src.endsWith('?clip=0:0'));
  assert.ok(evidenceVideo);
  assert.equal(evidenceVideo.controls, true);
  assert.equal(evidenceVideo.preload, 'metadata');
  assert.match(descendants(card).map((node) => node.textContent || '').join(' '), /证据片段/);
});

test('periodic updates preserve an unfinished vision review note and its focus', async () => {
  const harness = managerHarness();
  await flush();
  harness.setVisionJobs([{
    job_id: 'job-draft', status: 'needs_review', media_kind: 'image', original_name: '巡查图片.png',
    result: { candidates: [{ kind: 'vehicle_cluster_review', confidence: 0.7 }] },
  }]);
  harness.intervalCallbacks[0]();
  await flush();
  const root = harness.elements.get('vision-jobs');
  const note = descendants(root).find((node) => node.tagName === 'textarea');
  note.value = '刚才看到很多车辆，正在核对具体的路段';
  note.listeners.input();
  harness.document.activeElement = note;
  harness.intervalCallbacks[0]();
  await flush();
  assert.equal(descendants(root).find((node) => node.tagName === 'textarea'), note);
  assert.equal(note.value, '刚才看到很多车辆，正在核对具体的路段');

  harness.document.activeElement = null;
  harness.intervalCallbacks[0]();
  await flush();
  assert.equal(descendants(root).find((node) => node.tagName === 'textarea').value,
    '刚才看到很多车辆，正在核对具体的路段');
});
