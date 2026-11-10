"""Exercise manager state views at mobile and desktop widths with an isolated database."""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from PIL import Image
from psycopg.types.json import Jsonb
from werkzeug.serving import make_server

from storage import database, vision_repository

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures('isolated_test_database')


@pytest.mark.skipif(os.getenv('RUN_FLOOD_STATE_BROWSER') != '1' or sync_playwright is None,
                    reason='requires isolated PostgreSQL and Playwright')
def test_flood_manager_wait_empty_failure_cancel_retry_and_refresh(monkeypatch, tmp_path):
    import config
    from app import create_app

    uploads = tmp_path / 'uploads'
    uploads.mkdir()
    image = uploads / 'state-fixture.png'
    Image.new('RGB', (80, 60), (120, 140, 160)).save(image)
    monkeypatch.setattr(config, 'ROAD_CONDITION_ADMIN_PASSWORD', 'isolated-state-browser')
    monkeypatch.setattr(config, 'ROAD_CONDITION_ADMIN_TOKEN', '')
    monkeypatch.setattr(config, 'VISION_UPLOAD_DIR', str(uploads))
    monkeypatch.setenv('DATABASE_URL', DATABASE_URL)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute('TRUNCATE manager_vision_job, manager_vision_worker')
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY='isolated-state-browser')
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={'width': 390, 'height': 844}, service_workers='block')
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/*', lambda route: route.continue_()
                       if urlsplit(route.request.url).hostname in {'127.0.0.1', 'localhost'}
                       else route.abort())
            page.goto(base + '/manager', wait_until='networkidle')
            page.locator('#password').fill('isolated-state-browser')
            page.locator('#login-form button[type=submit]').click()
            expect(page.locator('#workspace')).to_be_visible(timeout=30000)

            def views(label):
                for width in (360, 390, 430, 1365):
                    page.set_viewport_size({'width': width, 'height': 844})
                    assert page.evaluate('document.documentElement.scrollWidth') <= width
                    page.locator('#vision-jobs').screenshot(path=str(tmp_path / f'{label}-{width}.png'))

            expect(page.locator('#vision-jobs')).to_contain_text('尚无巡查任务')
            views('no-jobs')
            job = vision_repository.create_job(
                DATABASE_URL, created_by='isolated-state-test', original_name=image.name,
                media_kind='image', media_path=image.name, sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                anchor_gcj=None, purpose='trial',
                observation_regions=[{'id': 'road', 'kind': 'road_surface',
                                      'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}],
            )
            page.reload(wait_until='networkidle')
            card = page.locator('.vision-job').filter(has_text=image.name)
            expect(card.locator('.vision-job-state')).to_have_text('排队中')
            expect(card.get_by_role('button', name='取消分析')).to_be_visible()
            views('waiting')
            page.once('dialog', lambda dialog: dialog.accept())
            card.get_by_role('button', name='取消分析').click()
            expect(card.locator('.vision-job-state')).to_have_text('已取消')
            page.reload(wait_until='networkidle')
            expect(card.locator('.vision-job-state')).to_have_text('已取消')
            views('cancelled')
            page.once('dialog', lambda dialog: dialog.accept())
            card.get_by_role('button', name='从断点重试').click()
            expect(card.locator('.vision-job-state')).to_have_text('排队中')
            views('retry')
            with database.connect(DATABASE_URL) as conn:
                conn.execute("UPDATE manager_vision_job SET status='failed', error=%s WHERE job_id=%s",
                             (Jsonb({'code': 'injected_failure', 'message': '隔离验收：模拟分析失败'}), job['job_id']))
            page.reload(wait_until='networkidle')
            expect(card.locator('.vision-job-state')).to_have_text('分析失败')
            expect(card).to_contain_text('隔离验收：模拟分析失败')
            views('failure')
            page.once('dialog', lambda dialog: dialog.accept())
            card.get_by_role('button', name='从断点重试').click()
            expect(card.locator('.vision-job-state')).to_have_text('排队中')
            with database.connect(DATABASE_URL) as conn:
                conn.execute("UPDATE manager_vision_job SET status='completed', result=%s WHERE job_id=%s",
                             (Jsonb({'candidates': [], 'metrics': {}, 'safety': {},
                                     'trial_coverage': {'flood': 'checked', 'accident': 'not_configured'}}),
                              job['job_id']))
            page.reload(wait_until='networkidle')
            expect(card.locator('.vision-job-state')).to_have_text('完成')
            expect(card).to_contain_text('积水：已检查')
            assert card.locator('.candidate-list .candidate-card').count() == 0
            views('empty-result')
            assert not errors
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
