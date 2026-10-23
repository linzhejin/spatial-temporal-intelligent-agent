"""The Harness middleware must stop before a seventh logical model stream."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_harness_plugin_enforces_budget_and_immutable_request_controls(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise the Harness plugin")
    schemas = tmp_path / "schemas.json"
    schemas.write_text(json.dumps([{"type": "function", "function": {
        "name": "resolve_poi", "description": "Look up one location",
        "parameters": {"type": "object", "properties": {}},
    }}]), encoding="utf-8")
    script = """
import { pathToFileURL } from 'node:url'
const { apply } = await import(pathToFileURL(process.argv[1]).href)
const hooks = new Map()
const registered = []
apply({
  on(name, handler) { hooks.set(name, handler) },
  tools: { register(tool) { registered.push(tool.name) } },
})
if (registered.join(',') !== 'resolve_poi') throw Error('tool registration changed')
const pre = hooks.get('agent/pre-step')
const stream = hooks.get('llm/stream')
const configure = hooks.get('agent/request')
const seed = Object.freeze({ provider: 'deepseek-official', model: 'deepseek-flash' })
const config = await configure({}, () => seed)
if (config.temperature !== 0 || seed.temperature !== undefined) {
  throw Error('temperature config did not use a detached copy')
}
pre({ agent: { id: 'sid' }, step: 1 }, () => null)
let passed = 0
for (let index = 0; index < 6; index++) {
  const request = Object.freeze({ sessionId: 'sid' })
  stream(request, () => { passed++; return null })
}
let blocked = false
try { stream(Object.freeze({ sessionId: 'sid' }), () => { passed++; return null }) }
catch (error) { blocked = error.message === 'campus_model_request_budget_exceeded' }
if (!blocked || passed !== 6) throw Error('seventh model stream was not blocked')
pre({ agent: { id: 'sid' }, step: 1 }, () => null)
stream(Object.freeze({ sessionId: 'sid' }), () => { passed++; return null })
if (passed !== 7) throw Error('new turn did not reset limit')
console.log(JSON.stringify({ blocked, first_turn_requests: 6, second_turn_requests: 1 }))
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script,
         str(ROOT / "experiments/baselines/harness_campus_tools.mjs")],
        capture_output=True, text=True, check=True,
        env={**{name: os.environ[name] for name in
                ("SYSTEMROOT", "PATH", "TEMP", "TMP", "COMSPEC")
                if name in os.environ},
             "CAMPUS_TOOL_SCHEMA_PATH": str(schemas),
             "CAMPUS_TOOL_BRIDGE_URL": "http://127.0.0.1:11111",
             "CAMPUS_TOOL_BRIDGE_TOKEN": "test-only-token",
             "CAMPUS_MAX_MODEL_REQUESTS_PER_TURN": "6"},
    )
    assert json.loads(result.stdout)["blocked"] is True
