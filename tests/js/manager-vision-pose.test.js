const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function loadManagerPoseHelpers() {
  const source = fs.readFileSync(require.resolve('../../static/js/manager.js'), 'utf8');
  const marker = 'function transformLatitude(x, y) {';
  const end = source.indexOf(marker);
  assert.ok(end > 0, 'manager.js pose boundary exists');
  const context = vm.createContext({
    window: {
      sessionStorage: {
        getItem() { return null; },
        setItem() {},
        removeItem() {},
      },
    },
  });
  vm.runInContext(source.slice(0, end) + `
    globalThis.managerPose = {
      parseDjiSubtitle: parseDjiSubtitle,
      parseDjiPoseTrack: parseDjiPoseTrack,
      convertPoseForUpload: convertPoseForUpload,
      poseForUpload: poseForUpload,
      MAX_DJI_GPS_STORED_POINTS: MAX_DJI_GPS_STORED_POINTS,
    };
  ` + '\n})();', context);
  return context.managerPose;
}

const helpers = loadManagerPoseHelpers();

test('a full DJI pose line parses every pose field with its original sign', () => {
  const result = helpers.parseDjiPoseTrack(
    '00:00:01,000 --> 00:00:01,100\n' +
    '[latitude : 30.535117] [longitude : 114.356321] [rel_alt : 82.104 abs_alt : 109.538] ' +
    '[gb_yaw : -92.200000 gb_pitch : -79.300000 gb_roll : 0.300000] [yaw : -92.2] [focal_len : 240]');
  assert.equal(result.parsed, true);
  assert.equal(result.pointCount, 1);
  assert.equal(result.sampled, false);
  const point = result.points[0];
  assert.equal(point.seconds, 1);
  assert.equal(point.lat, 30.535117);
  assert.equal(point.lng, 114.356321);
  assert.equal(point.rel_alt_m, 82.104);
  assert.equal(point.abs_alt_m, 109.538);
  assert.equal(point.yaw_deg, -92.2);
  assert.equal(point.gb_yaw_deg, -92.2);
  assert.equal(point.gb_pitch_deg, -79.3);
  assert.equal(point.gb_roll_deg, 0.3);
  assert.equal(point.focal_len_35mm, 240);
});

test('a pose line missing gimbal and focal fields keeps the point with null fields', () => {
  const result = helpers.parseDjiPoseTrack(
    '00:00:02,000 --> 00:00:02,100\n' +
    '[latitude : 30.535] [longitude : 114.356] [rel_alt : 100 abs_alt : 130]');
  assert.equal(result.parsed, true);
  assert.equal(result.pointCount, 1);
  const point = result.points[0];
  assert.equal(point.rel_alt_m, 100);
  assert.equal(point.abs_alt_m, 130);
  assert.equal(point.yaw_deg, null);
  assert.equal(point.gb_yaw_deg, null);
  assert.equal(point.gb_pitch_deg, null);
  assert.equal(point.gb_roll_deg, null);
  assert.equal(point.focal_len_35mm, null);
});

test('a time-only subtitle keeps its capture time and reports no camera pose', () => {
  const result = helpers.parseDjiSubtitle('2025-06-15 15:47:50.149');
  assert.ok(result);
  assert.equal(result.captureTime, '2025-06-15T15:47:50');
  assert.equal(result.cameraPose.parsed, false);
  assert.equal(result.cameraPose.pointCount, 0);
  assert.equal(result.cameraPose.sampled, false);
  assert.equal(result.cameraPose.points.length, 0);
});

test('many pose points sample down to the storage cap and retain the final point', () => {
  const rows = Array.from({ length: 20002 }, (_item, index) =>
    `[latitude : 30.536] [longitude : ${(114.363 + index / 1000000).toFixed(6)}]`);
  const result = helpers.parseDjiPoseTrack(rows.join('\n'));
  assert.equal(result.pointCount, 20002);
  assert.equal(result.sampled, true);
  assert.ok(result.points.length <= helpers.MAX_DJI_GPS_STORED_POINTS);
  assert.equal(result.points.at(-1).lng, Number((114.363 + 20001 / 1000000).toFixed(6)));
});

test('invalid GPS coordinates are skipped without counting toward the pose total', () => {
  const result = helpers.parseDjiPoseTrack([
    '[latitude : 0] [longitude : 0]',
    '[latitude : 95] [longitude : 114.3]',
    '[latitude : -91] [longitude : 114.3]',
    '[latitude : 30.5] [longitude : 114.3]',
  ].join('\n'));
  assert.equal(result.parsed, true);
  assert.equal(result.pointCount, 1);
  assert.equal(result.points[0].lat, 30.5);
});

test('pose points carry the seconds from their preceding SRT timecode', () => {
  const result = helpers.parseDjiPoseTrack([
    '1',
    '00:00:05,500 --> 00:00:05,800',
    '[latitude : 30.5] [longitude : 114.3]',
    '2',
    '00:01:02,250 --> 00:01:02,500',
    '[latitude : 30.6] [longitude : 114.4]',
  ].join('\n'));
  assert.equal(result.points[0].seconds, 5.5);
  assert.equal(result.points[1].seconds, 62.25);
});

test('convertPoseForUpload emits the backend camera_pose contract', () => {
  const pose = helpers.parseDjiPoseTrack(
    '00:00:01,000 --> 00:00:01,100\n' +
    '[latitude : 30.5351] [longitude : 114.3563] [rel_alt : 82.1 abs_alt : 109.5] ' +
    '[gb_yaw : -92.2 gb_pitch : -79.3 gb_roll : 0.4] [yaw : -91.5] [focal_len : 240]');
  assert.deepEqual(JSON.parse(JSON.stringify(helpers.convertPoseForUpload(pose))), {
    crs: 'WGS84',
    parsed: true,
    point_count: 1,
    sampled: false,
    trajectory: [{
      seconds: 1,
      lng: 114.3563,
      lat: 30.5351,
      rel_alt_m: 82.1,
      abs_alt_m: 109.5,
      yaw_deg: -91.5,
      gb_yaw_deg: -92.2,
      gb_pitch_deg: -79.3,
      gb_roll_deg: 0.4,
      focal_len_35mm: 240,
    }],
  });
});

test('poseForUpload returns null for an unparsed or missing pose track', () => {
  assert.equal(helpers.poseForUpload({ parsed: false, pointCount: 0, sampled: false, points: [] }), null);
  assert.equal(helpers.poseForUpload(null), null);
});

test('poseForUpload returns the same contract as convertPoseForUpload and keeps null fields', () => {
  const pose = helpers.parseDjiPoseTrack(
    '00:00:02,000 --> 00:00:02,100\n' +
    '[latitude : 30.535] [longitude : 114.356] [rel_alt : 100 abs_alt : 130]');
  const result = helpers.poseForUpload(pose);
  assert.deepEqual(
    JSON.parse(JSON.stringify(result)),
    JSON.parse(JSON.stringify(helpers.convertPoseForUpload(pose))));
  assert.equal(result.trajectory[0].yaw_deg, null);
  assert.equal(result.trajectory[0].gb_yaw_deg, null);
  assert.equal(result.trajectory[0].gb_pitch_deg, null);
  assert.equal(result.trajectory[0].gb_roll_deg, null);
  assert.equal(result.trajectory[0].focal_len_35mm, null);
});

test('a whole DJI SRT file parses into capture time, GPS track and upload pose', () => {
  const path = require('node:path');
  const srt = fs.readFileSync(
    path.join(__dirname, '..', 'fixtures', 'dji_flight.srt'), 'utf8');
  const metadata = helpers.parseDjiSubtitle(srt);

  assert.equal(metadata.captureTime, '2026-11-06T10:12:31');
  assert.equal(metadata.gpsPointCount, 3);
  assert.equal(metadata.gpsTimedPointCount, 3);
  assert.equal(metadata.gpsPoints.length, 3);

  assert.equal(metadata.cameraPose.parsed, true);
  assert.equal(metadata.cameraPose.pointCount, 3);
  assert.equal(metadata.cameraPose.sampled, false);

  const upload = JSON.parse(JSON.stringify(helpers.poseForUpload(metadata.cameraPose)));
  assert.equal(upload.crs, 'WGS84');
  assert.equal(upload.point_count, 3);
  assert.deepEqual(upload.trajectory.map((point) => point.seconds), [0, 1, 2]);
  assert.deepEqual(upload.trajectory.map((point) => point.rel_alt_m), [82.104, 84.5, 86.87]);
  assert.deepEqual(upload.trajectory.map((point) => point.gb_pitch_deg), [-79.3, -78.9, -88.7]);
  assert.deepEqual(upload.trajectory.map((point) => point.gb_roll_deg), [0, 0.1, 0]);
  assert.deepEqual(upload.trajectory.map((point) => point.gb_yaw_deg), [-92.2, -91.5, -90.8]);
  assert.deepEqual(upload.trajectory.map((point) => point.lat),
    [30.535117, 30.535209, 30.535301]);
  assert.deepEqual(upload.trajectory.map((point) => point.lng),
    [114.356321, 114.356318, 114.356322]);
});