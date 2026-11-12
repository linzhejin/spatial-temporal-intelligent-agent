"""Prepare an isolated automatic-flood bundle from existing model and test evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from scripts.deploy.flood_release import REQUIRED_TESTED_SOURCES
from scripts.deploy.prepare_flood_bundle import RESOURCE_SOURCES, require_junit, validate_resource
from vision.model_release import (assisted_flood_runtime_config, automatic_flood_runtime_config,
                                  check_automatic_flood_release, check_optional_model_release)

APP = Path('/home/ubuntu/campus-spatial-intelligence-agent')
MODEL_ROOT = Path('/home/ubuntu/luojia-models')
STAGE_ROOT = Path('/home/ubuntu/luojia-validation')
MODEL_HASH = '1ef47ccd3aef9cafee359485cfb38c65edcd6625e1552d02e23afacd98f7eb52'
REVISION = 'doodleverse-floodnet-resunet-v7'
REQUIRED_WORKFLOW_REPORTS = {'regression': 258, 'browser': 4, 'cache': 1,
                             'helpers': 11, 'public_smoke': 1, 'worker_startup': 1,
                             'bundle_helpers': 13}


def sha256(path, *, normalized=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if normalized else data).hexdigest()


def validate_workflow(workflow, evidence_dir, model_hash):
    if (workflow.get('status') != 'passed' or workflow.get('scope') != 'automatic_flood'
            or workflow.get('model_sha256') != model_hash
            or workflow.get('runtime_config') != automatic_flood_runtime_config()):
        raise ValueError('Automatic workflow evidence does not match the model or runtime')
    reports = workflow.get('reports') or []
    for name, count in REQUIRED_WORKFLOW_REPORTS.items():
        matches = [row for row in reports if row.get('name') == name]
        if len(matches) != 1:
            raise ValueError('Automatic workflow report is missing or duplicated')
        item = matches[0]
        report = evidence_dir / PureWindowsPath(item.get('file', '')).name
        if not report.is_file() or sha256(report) != item.get('sha256'):
            raise ValueError('Automatic workflow report is missing or changed')
        require_junit(report, count)
    js = workflow.get('javascript') or {}
    js_path = evidence_dir / PureWindowsPath(js.get('file', '')).name
    if (not js_path.is_file() or sha256(js_path) != js.get('sha256')
            or (js.get('tests'), js.get('passed'), js.get('failed'), js.get('skipped')) != (238, 238, 0, 0)):
        raise ValueError('Automatic workflow JavaScript evidence is incomplete')
    text = js_path.read_text(encoding='utf-8-sig')
    for label, number in (('tests', 238), ('pass', 238), ('fail', 0), ('skipped', 0)):
        if not re.search(r'(?m)^.*?\b' + label + r'\s+' + str(number) + r'\s*$', text):
            raise ValueError('Automatic workflow JavaScript output is incomplete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--base-bundle', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--bundle-tag')
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    base_bundle = args.base_bundle.resolve(strict=True)
    base = base_bundle.parent
    run_tag = stage.name.removeprefix('flood-resource-')
    bundle_tag = args.bundle_tag or run_tag
    if (stage.parent != STAGE_ROOT.resolve() or not re.fullmatch(r'flood-resource-[0-9a-f]{10}', stage.name)
            or base.parent != MODEL_ROOT.resolve() or not base.name.startswith('floodnet-v7-')
            or base_bundle.name != 'bundle.json' or not re.fullmatch(r'[0-9a-f]{40}', args.commit)
            or not re.fullmatch(re.escape(run_tag) + r'(?:-r[1-9][0-9]*)?', bundle_tag)):
        raise ValueError('Invalid isolated stage, base bundle or release commit')
    evidence_dir = stage / 'automatic-evidence'
    resource_path = stage / 'resource_acceptance.json'
    workflow_path = evidence_dir / 'workflow.json'
    sources_path = evidence_dir / 'tested-sources.json'
    resource = json.loads(resource_path.read_text(encoding='utf-8'))
    workflow = json.loads(workflow_path.read_text(encoding='utf-8'))
    sources = json.loads(sources_path.read_text(encoding='utf-8'))
    validate_resource(resource)
    if (resource.get('scope') != 'automatic_flood' or resource.get('model_sha256') != MODEL_HASH
            or resource.get('runtime_config') != automatic_flood_runtime_config()):
        raise ValueError('Automatic resource evidence does not match the model or runtime')
    validate_workflow(workflow, evidence_dir, MODEL_HASH)
    if not REQUIRED_TESTED_SOURCES.issubset(sources):
        raise ValueError('Tested source manifest is incomplete')
    for name in REQUIRED_TESTED_SOURCES:
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', name) or '..' in Path(name).parts:
            raise ValueError('Invalid tested source path')
        content = subprocess.check_output(['git', '-C', str(APP), 'show', f'{args.commit}:{name}'])
        if hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest() != sources[name]:
            raise ValueError('Release commit differs from the tested source snapshot')
    for name in RESOURCE_SOURCES:
        if sha256(stage / 'candidate' / name, normalized=True) != sources.get(name):
            raise ValueError('Resource-tested code differs from the release')
    old_bundle = json.loads(base_bundle.read_text(encoding='utf-8'))
    old_manifest = Path(old_bundle['release_manifest']).resolve(strict=True)
    old_model = Path(old_bundle['model_path']).resolve(strict=True)
    old_runtime = Path(old_bundle['runtime_python'])
    if (not all(p.is_relative_to(base) for p in (old_manifest, old_model)) or not old_runtime.is_file()
            or not old_runtime.parent.resolve().is_relative_to(base)
            or sha256(old_model) != MODEL_HASH or old_bundle.get('model_sha256') != MODEL_HASH):
        raise ValueError('Existing model bundle differs from the reviewed model')
    if check_optional_model_release('flood', old_model, REVISION, old_manifest, None,
                                    assisted_flood_runtime_config(), release_mode='assisted_review') != {
                                        'approved': True, 'reason': 'approved'}:
        raise ValueError('Existing manually reviewed model release is invalid')
    old_entry = json.loads(old_manifest.read_text(encoding='utf-8'))['models']['flood']
    samples = json.loads((stage / 'samples.json').read_text(encoding='utf-8'))
    sample = samples[4]
    if 'HURRICANE-IAN_000491' not in sample['filename']:
        raise ValueError('Expected demonstration image is missing')
    image = stage / 'samples' / sample['filename']
    if sha256(image) != sample['sha256']:
        raise ValueError('Public demonstration image has changed')

    destination = MODEL_ROOT / ('floodnet-v7-' + bundle_tag)
    destination.mkdir(mode=0o700, exist_ok=False)
    shutil.copy2(old_model, destination / 'model.onnx')
    shutil.copytree(base / 'runtime', destination / 'runtime', symlinks=True)
    shutil.copytree(base / 'metadata', destination / 'metadata')
    shutil.copy2(image, destination / 'public-smoke.png')
    outputs = destination / 'evidence'
    outputs.mkdir()
    manual_evidence = {}
    for kind in ('parity', 'resource', 'workflow'):
        item = old_entry['acceptance'][kind]
        original = Path(item['path']).resolve(strict=True)
        if not original.is_relative_to(base) or sha256(original) != item['sha256']:
            raise ValueError('Existing manual evidence is missing or changed')
        target = outputs / ('manual-' + kind + '.json')
        shutil.copy2(original, target)
        manual_evidence[kind] = {'status': 'passed', 'path': str(target), 'sha256': sha256(target)}
    auto_evidence = {}
    for kind, source in (('resource', resource_path), ('workflow', workflow_path)):
        target = outputs / ('automatic-' + kind + '.json')
        shutil.copy2(source, target)
        auto_evidence[kind] = {'status': 'passed', 'path': str(target), 'sha256': sha256(target)}
    for row in workflow['reports']:
        shutil.copy2(evidence_dir / PureWindowsPath(row['file']).name, outputs)
    shutil.copy2(evidence_dir / PureWindowsPath(workflow['javascript']['file']).name, outputs)
    shutil.copy2(sources_path, outputs / 'tested-sources.json')
    shutil.copy2(stage / 'samples.json', outputs / 'resource-samples.json')
    entry = dict(old_entry)
    entry['acceptance'] = manual_evidence
    entry['automatic_flood'] = {'enabled': True, 'runtime_config': automatic_flood_runtime_config(),
                                'acceptance': auto_evidence}
    manifest = destination / 'release.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'models': {'flood': entry}}, indent=2), encoding='utf-8')
    model = destination / 'model.onnx'
    runtime = destination / 'runtime/bin/python'
    if sha256(model) != MODEL_HASH or not runtime.is_file():
        raise ValueError('Copied model or runtime is invalid')
    if check_automatic_flood_release(model, REVISION, manifest) != {'approved': True, 'reason': 'approved'}:
        raise ValueError('Automatic release manifest was rejected')
    version_code = ("import importlib.metadata as m,json,sys; "
                    "print(json.dumps({'prefix':sys.prefix,'versions':{n:m.version(n) "
                    "for n in ['numpy','onnxruntime','opencv-python-headless','scikit-image','scipy','Pillow','python-dotenv']}}))")
    info = json.loads(subprocess.check_output([str(runtime), '-c', version_code], text=True))
    if info['versions'] != old_bundle['runtime_versions'] or Path(info['prefix']).resolve() != (destination / 'runtime').resolve():
        raise ValueError('Copied runtime differs from the accepted runtime')
    bundle = {'target_commit': args.commit, 'model_path': str(model), 'model_sha256': MODEL_HASH,
              'runtime_python': str(runtime), 'release_manifest': str(manifest), 'evidence': manual_evidence,
              'automatic_evidence': auto_evidence, 'tested_source_sha256': sources,
              'runtime_versions': info['versions'], 'public_smoke_image': str(destination / 'public-smoke.png'),
              'public_smoke_sha256': sample['sha256']}
    path = destination / 'bundle.json'
    path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'bundle_path': str(path), 'bundle_sha256': sha256(path)}))


if __name__ == '__main__':
    main()
