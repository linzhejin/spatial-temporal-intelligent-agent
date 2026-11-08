const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto, createHash } = require('node:crypto');

const source = fs.readFileSync(path.join(__dirname, '../../static/js/manager.js'), 'utf8');
const poseStart = source.indexOf('  function convertPoseForUpload(');
const poseEnd = source.indexOf('  function transformLatitude(', poseStart);
const uploadStart = source.indexOf('  async function uploadVideoInChunks(');
const uploadEnd = source.indexOf('  async function refreshVisionStatus()', uploadStart);
assert.ok(poseStart >= 0 && poseEnd > poseStart && uploadStart >= 0 && uploadEnd > uploadStart);

function setup(oldPose, incomingPose, { oldDigest = true, crypto = webcrypto } = {}) {
  const file = { name: 'same.mp4', size: 100, lastModified: 1, slice: (a, b) => ({ size: b - a }) };
  let pending = { upload_id: 'previous', filename: file.name, size: file.size, last_modified: 1,
    camera_stabilized: false, captured_at: 1000, observation_regions: [], scene_id: null, anchor: null };
  const calls = [];
  let stored = oldPose;
  const context = vm.createContext({
    state: { anchor: null, uploadChunkBytes: 100 }, window: { crypto, setTimeout }, TextEncoder, Uint8Array,
    readPendingVisionUpload: () => pending,
    fileMatchesPending: (f, p) => f.name === p.filename && f.size === p.size && f.lastModified === p.last_modified,
    savePendingVisionUpload: (value) => { pending = value; }, showVisionUploadProgress() {},
    request: async (url, method, body) => {
      calls.push({ url, method, body });
      if (url === '/api/manager/vision-uploads') {
        stored = body.camera_pose ? JSON.parse(body.camera_pose) : null;
        return { upload_id: 'new', offset: 100 };
      }
      if (method === 'GET') return { size: 100, offset: 100 };
      if (url.endsWith('/complete')) return { job: { camera_pose: stored } };
      throw new Error('unexpected request ' + url);
    },
  });
  vm.runInContext(source.slice(poseStart, poseEnd) + source.slice(uploadStart, uploadEnd) +
    '\nthis.run = uploadVideoInChunks; this.serializePose = poseForUpload;', context);
  const serializedOld = context.serializePose(oldPose);
  if (oldDigest) pending.camera_pose_sha256 = createHash('sha256').update(JSON.stringify(serializedOld)).digest('hex');
  stored = serializedOld;
  return { calls, run: () => context.run(file, false, 1000,
    { capture_time_source: 'dji_srt', capture_timezone: 'Asia/Shanghai' }, [], null, incomingPose) };
}

const oldPose = { parsed: true, pointCount: 1, sampled: false,
  points: [{ lng: 114.36, lat: 30.53, rel_alt_m: 50 }] };
const changedPose = { parsed: true, pointCount: 1, sampled: false,
  points: [{ lng: 114.37, lat: 30.54, rel_alt_m: 100 }] };

test('unchanged pose resumes the existing upload', async () => {
  const env = setup(oldPose, oldPose);
  const result = await env.run();
  assert.equal(env.calls[0].url, '/api/manager/vision-uploads/previous');
  assert.equal(result.job.camera_pose.trajectory[0].rel_alt_m, 50);
});

for (const [label, previous, incoming] of [
  ['changed subtitle', oldPose, changedPose],
  ['new subtitle', null, changedPose],
  ['removed subtitle', oldPose, null],
]) {
  test(label + ' starts a new session with the current pose', async () => {
    const env = setup(previous, incoming);
    const result = await env.run();
    assert.equal(env.calls[0].url, '/api/manager/vision-uploads');
    assert.equal(result.job.camera_pose && result.job.camera_pose.trajectory[0].rel_alt_m,
      incoming && incoming.points[0].rel_alt_m);
  });
}

test('legacy pending metadata and unavailable crypto never resume old sessions', async () => {
  for (const options of [{ oldDigest: false }, { crypto: null }]) {
    const env = setup(oldPose, oldPose, options);
    await env.run();
    assert.equal(env.calls[0].url, '/api/manager/vision-uploads');
  }
});
