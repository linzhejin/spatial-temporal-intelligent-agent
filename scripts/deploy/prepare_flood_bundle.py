"""Build an immutable flood release bundle only from completed acceptance evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath

MODEL_HASH = '1ef47ccd3aef9cafee359485cfb38c65edcd6625e1552d02e23afacd98f7eb52'
MODEL_ROOT = Path('/home/ubuntu/luojia-models')
STAGE_ROOT = Path('/home/ubuntu/luojia-validation')
RESOURCE_SOURCES = {
    'config.py', 'storage/vision_repository.py', 'vision/analysis.py', 'vision/engine.py',
    'vision/model_release.py', 'vision/specialized.py', 'vision/worker.py',
    'vision/doodleverse_preprocessing.py', 'vision/assisted_image.py',
    'vision/assisted_image_child.py', 'vision/assisted_image_supervisor.py',
    'requirements-flood-runtime.txt',
}


def sha256(path, *, normalized=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if normalized else data).hexdigest()


def require_junit(path, count):
    suites = ET.parse(path).getroot().findall('testsuite')
    totals = {key: sum(int(suite.attrib.get(key, '0')) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if totals != {'tests': count, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError('An integration report did not pass completely')


def validate_resource(report):
    if (report.get('status') != 'passed' or report.get('vision_services_active') is not True
            or report.get('other_services_healthy') is not True):
        raise ValueError('Linux resource acceptance did not pass')
    cases = {str(case.get('case')): case for case in report.get('cases', [])}
    required = {'preflight', *(str(index) for index in range(10)), 'cancel', 'timeout', 'failure_retry'}
    if not required.issubset(cases):
        raise ValueError('Linux resource acceptance is incomplete')
    for case in cases.values():
        if (case.get('exit_code') != 0 or not 0 < case.get('peak_bytes', 0) <= 640 * 1024 * 1024
                or case.get('swap_max') != '0' or str(case.get('cpu_max', '')).split()[0:1] != ['75000']
                or (case.get('memory_events') or {}).get('oom_kill', 0)
                or (case.get('worker') or {}).get('status_check') != 'passed'):
            raise ValueError('Linux resource case violated a limit or failed')
    if (cases['cancel']['worker'].get('cancel_to_terminal_seconds', 5) >= 5
            or cases['timeout']['worker'].get('status') != 'failed'
            or cases['failure_retry']['worker'].get('retry_terminal_status') not in ('completed', 'needs_review')):
        raise ValueError('Cancellation, timeout or retry evidence is incomplete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--bundle-tag', default=None)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    run_tag = stage.name.removeprefix('flood-resource-')
    bundle_tag = args.bundle_tag or run_tag
    if (stage.parent != STAGE_ROOT.resolve() or not re.fullmatch(r'flood-resource-[0-9a-f]{10}', stage.name)
            or not re.fullmatch(re.escape(run_tag) + r'(?:-r[1-9][0-9]*)?', bundle_tag)
            or not re.fullmatch(r'[0-9a-f]{40}', args.commit)):
        raise ValueError('Invalid isolated stage or commit')
    evidence_dir = stage / 'release-evidence'
    resource_path = stage / 'resource_acceptance.json'
    parity_path = evidence_dir / 'parity.json'
    workflow_path = evidence_dir / 'workflow.json'
    source_path = evidence_dir / 'tested-sources.json'
    resource = json.loads(resource_path.read_text(encoding='utf-8'))
    parity = json.loads(parity_path.read_text(encoding='utf-8'))
    workflow = json.loads(workflow_path.read_text(encoding='utf-8'))
    sources = json.loads(source_path.read_text(encoding='utf-8'))
    validate_resource(resource)
    observed = parity.get('observed') or {}
    if (parity.get('status') != 'passed' or parity.get('onnx_sha256') != MODEL_HASH
            or parity.get('splits', {}).get('validation', {}).get('sample_count') != 64
            or parity.get('splits', {}).get('test', {}).get('sample_count') != 69
            or observed.get('max_area_ratio_delta_percentage_points', 1) > 0.5
            or observed.get('aggregate_binary_mask_iou', 0) < 0.99
            or observed.get('candidate_decisions_equal') is not True):
        raise ValueError('Model parity evidence is missing or outside its thresholds')
    if workflow.get('status') != 'passed' or workflow.get('model_sha256') != MODEL_HASH:
        raise ValueError('Workflow evidence does not match the reviewed model')
    required_reports = {
        'flood_browser_real_recovery_20261110.xml': 3,
        'flood_regression_recovery_20261110.xml': 264,
        'flood_browser_cache_release_helpers_20261110.xml': 10,
        'flood_normal_worker_startup_20261110.xml': 1,
        'flood_deploy_helpers_final_20261110.xml': 11,
        'flood_state_views_20261110.xml': 1,
        'flood_cache_actual_209d19f_20261110.xml': 2,
        'flood_public_smoke_e2e_20261110.xml': 1,
    }
    reports = workflow.get('junit_reports') or []
    for name, expected_count in required_reports.items():
        record = next((row for row in reports if PureWindowsPath(row.get('file', '')).name == name), None)
        report_file = evidence_dir / name
        if not record or not report_file.is_file() or sha256(report_file) != record.get('sha256'):
            raise ValueError('Workflow report does not match its recorded hash')
        require_junit(report_file, expected_count)
    js = workflow.get('javascript') or {}
    js_path = evidence_dir / PureWindowsPath(js.get('file', '')).name
    if (not js_path.is_file() or sha256(js_path) != js.get('sha256')
            or (js.get('passed'), js.get('failed'), js.get('skipped')) != (233, 0, 0)):
        raise ValueError('JavaScript workflow evidence is incomplete')
    if not RESOURCE_SOURCES.issubset(sources):
        raise ValueError('Resource-tested source manifest is incomplete')
    for name in RESOURCE_SOURCES:
        if sha256(stage / 'candidate' / name, normalized=True) != sources[name]:
            raise ValueError('Resource-tested source differs from the proposed release')
    model = stage / 'flood.onnx'
    if sha256(model) != MODEL_HASH:
        raise ValueError('Staged model hash changed')
    detector = Path('/opt/whu-walker/models/visdrone-rtdetrv4-s.onnx')
    if not detector.is_file() or sha256(detector) != sha256(stage / 'traffic.onnx'):
        raise ValueError('Production traffic detector differs from the tested detector')
    samples = json.loads((stage / 'samples.json').read_text(encoding='utf-8'))
    if (len(samples) != 10 or sum(item['ground_truth_label'] == 'positive_event' for item in samples) != 5
            or sum(item['ground_truth_label'] == 'dry_road' for item in samples) != 5
            or not samples[-1]['transformed_for_resource_test']):
        raise ValueError('Resource sample set does not match the acceptance contract')

    MODEL_ROOT.mkdir(parents=True, mode=0o700, exist_ok=True)
    destination = MODEL_ROOT / ('floodnet-v7-' + bundle_tag)
    destination.mkdir(mode=0o700, exist_ok=False)
    shutil.copy2(model, destination / 'model.onnx')
    public_sample = stage / 'samples' / samples[0]['filename']
    if sha256(public_sample) != samples[0]['sha256']:
        raise ValueError('Public smoke image differs from the resource-tested sample')
    shutil.copy2(public_sample, destination / 'public-smoke.png')
    shutil.copytree((stage / 'flood-venv').resolve(strict=True), destination / 'runtime', symlinks=True)
    runtime = destination / 'runtime/bin/python'
    version_code = ("import importlib.metadata as m,json,sys; "
                    "print(json.dumps({'prefix':sys.prefix,'versions':{n:m.version(n) "
                    "for n in ['numpy','onnxruntime','opencv-python-headless','scikit-image','scipy','Pillow','python-dotenv']}}))")
    runtime_info = json.loads(subprocess.check_output([str(runtime), '-c', version_code], text=True))
    versions = runtime_info['versions']
    expected_versions = {'numpy': '1.26.4', 'onnxruntime': '1.20.1', 'opencv-python-headless': '4.10.0.84',
                         'scikit-image': '0.24.0', 'scipy': '1.14.1', 'Pillow': '11.0.0',
                         'python-dotenv': '1.1.0'}
    if versions != expected_versions or Path(runtime_info['prefix']).resolve() != (destination / 'runtime').resolve():
        raise ValueError('Copied flood runtime differs from the tested runtime')
    outputs = destination / 'evidence'
    outputs.mkdir()
    for path in [parity_path, resource_path, workflow_path, source_path, js_path, stage / 'samples.json',
                 *(evidence_dir / name for name in required_reports)]:
        shutil.copy2(path, outputs / path.name)
    metadata = evidence_dir / 'metadata'
    if not metadata.is_dir() or not all((metadata / name).is_file() for name in (
            'readme.txt', 'classes.txt', 'floodnet_resunet_v7_modelcard.json', 'floodnet_resunet_v7.json')):
        raise ValueError('Model attribution, model card or configuration is missing')
    shutil.copytree(metadata, destination / 'metadata')
    result_evidence = {}
    for kind, filename in (('parity', 'parity.json'), ('resource', 'resource_acceptance.json'),
                           ('workflow', 'workflow.json')):
        path = outputs / filename
        result_evidence[kind] = {'status': 'passed', 'path': str(path), 'sha256': sha256(path)}

    sys.path.insert(0, str(stage / 'candidate'))
    from vision.model_release import assisted_flood_runtime_config, check_optional_model_release
    now = datetime.now(timezone.utc)
    entry = {'mode': 'assisted_review', 'approved': True, 'approved_by': 'project-owner-authorized-session',
             'approved_at': now.isoformat(), 'approval_expires_at': (now + timedelta(days=90)).isoformat(),
             'rights_status': 'approved', 'license': 'CC BY 4.0',
             'rights_evidence': 'https://zenodo.org/records/7566797', 'field_validated': False,
             'model_revision': 'doodleverse-floodnet-resunet-v7', 'model_sha256': MODEL_HASH,
             'runtime_config': assisted_flood_runtime_config(), 'acceptance': result_evidence}
    manifest = destination / 'release.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'models': {'flood': entry}}, indent=2), encoding='utf-8')
    decision = check_optional_model_release('flood', destination / 'model.onnx', entry['model_revision'],
                                            manifest, None, assisted_flood_runtime_config(),
                                            release_mode='assisted_review')
    if decision != {'approved': True, 'reason': 'approved'}:
        raise ValueError('Final release manifest was rejected')
    bundle = {'target_commit': args.commit, 'model_path': str(destination / 'model.onnx'),
              'model_sha256': MODEL_HASH, 'runtime_python': str(runtime), 'release_manifest': str(manifest),
              'evidence': result_evidence, 'tested_source_sha256': sources, 'runtime_versions': versions,
              'public_smoke_image': str(destination / 'public-smoke.png'),
              'public_smoke_sha256': samples[0]['sha256']}
    bundle_path = destination / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'bundle_path': str(bundle_path), 'bundle_sha256': sha256(bundle_path)}))


if __name__ == '__main__':
    main()
