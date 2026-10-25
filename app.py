"""
珞珈智行 (WHU-Walker) — Flask 应用入口

注册 API 蓝图、配置 CORS、统一错误处理、dev/prod 模式切换。
"""
import json
import ipaddress
import logging
import mimetypes
import os
import re
import threading
import time
from flask import Flask, abort, send_from_directory, jsonify, Response, request
from flask_cors import CORS
from psycopg.conninfo import conninfo_to_dict

# 部分系统（含 Linux 生产机）的 mimetypes 表不识别 .apk，浏览器下载时会变成未知文件
mimetypes.add_type("application/vnd.android.package-archive", ".apk")

import config

logger = logging.getLogger(__name__)


def _cache_max_age(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".html",):
        return 0
    if ext in (".js", ".css", ".json", ".svg", ".png", ".ico", ".webmanifest"):
        return 3600
    if ext in (".woff2", ".woff", ".ttf", ".eot"):
        return 2592000
    return 60


def create_app() -> Flask:
    basedir = os.path.abspath(os.path.dirname(__file__))

    app = Flask(
        __name__,
        static_folder=os.path.join(basedir, "static"),
        static_url_path="",
    )

    # Production must use one stable, high-entropy key shared by every web worker.
    secret_key = os.getenv("SECRET_KEY", "")
    is_production = os.getenv("FLASK_ENV", "development") == "production"
    if is_production and (
        len(secret_key) < 32
        or secret_key == "whu-walker-dev-secret-change-me"
        or secret_key.lower() in {"secret", "changeme", "change-me", "development"}
    ):
        raise RuntimeError("Production requires a strong SECRET_KEY of at least 32 characters")
    app.config["SECRET_KEY"] = secret_key or "whu-walker-dev-secret-change-me"
    app.config["DATABASE_URL"] = os.getenv("DATABASE_URL")
    if (os.getenv("AGENT_REPLAY_EVENT_EPOCH") is not None
            or os.getenv("AGENT_REPLAY_EVENT_SHA256") is not None):
        from experiments.event_replay_guard import assert_replay_database_target

        if (is_production or os.getenv("AGENT_REPLAY_ENABLE_LOCAL") != "1"
                or os.getenv("AGENT_TOOL_TRACE_ENABLED") != "1"):
            raise ValueError("synthetic event replay requires an isolated local service")
        assert_replay_database_target(app.config["DATABASE_URL"])
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Strict"

    if os.getenv("FLASK_ENV", "development") == "production":
        app.config["SESSION_COOKIE_SECURE"] = True
        origins_raw = os.getenv("CORS_ORIGINS", "")
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        if origins:
            CORS(app, origins=origins)
        else:
            render_url = os.getenv("RENDER_EXTERNAL_URL", "https://your-app.onrender.com")
            CORS(app, origins=[render_url])
        app.config["DEBUG"] = False
        app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB
    else:
        app.config["SESSION_COOKIE_SECURE"] = False
        CORS(app)
        app.config["DEBUG"] = True
        app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB dev

    from api.routes import api_bp
    app.register_blueprint(api_bp)
    from api.conversations import conversations_bp
    app.register_blueprint(conversations_bp)
    from api.runs import runs_bp
    app.register_blueprint(runs_bp)

    @app.before_request
    def _set_manager_upload_limit():
        if request.path == "/api/manager/vision-jobs" and request.method == "POST":
            request.max_content_length = config.VISION_MAX_MEDIA_BYTES + 1024 * 1024
        elif (request.path.startswith("/api/manager/vision-uploads/")
              and request.path.endswith("/chunks") and request.method == "PUT"):
            request.max_content_length = config.VISION_CHUNK_BYTES + 1024

    if app.config["DATABASE_URL"]:
        from storage.database import initialize
        # A configured durable store is authoritative; a migration error must
        # stop startup instead of silently serving an empty in-memory session.
        initialize(app.config["DATABASE_URL"])

    @app.route("/")
    def serve_index():
        resp = send_from_directory(app.static_folder, "index.html")
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

    @app.route("/manager")
    def serve_manager():
        resp = send_from_directory(app.static_folder, "manager.html")
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

    @app.route("/health")
    @app.route("/api/health")
    def health_check():
        return jsonify({
            "status": "ok",
            "project": "珞珈智行",
            "timestamp": int(time.time()),
        })

    @app.route("/api/research/replay-readiness")
    def research_replay_readiness():
        """Opt-in local proof that research replay writes target a test database."""
        if (os.getenv("FLASK_ENV", "development") == "production"
                or os.getenv("AGENT_REPLAY_ENABLE_LOCAL") != "1"):
            abort(404)
        try:
            remote = ipaddress.ip_address(request.remote_addr or "")
        except ValueError:
            abort(404)
        if not remote.is_loopback:
            abort(404)

        database_url = app.config.get("DATABASE_URL")
        try:
            info = conninfo_to_dict(database_url or "")
            configured_name = info.get("dbname") or ""
            configured_host = info.get("host") or ""
            if (not re.search(r"(?:^|_)test(?:_|$)", configured_name, re.IGNORECASE)
                    or configured_host.casefold() not in {"localhost", "127.0.0.1", "::1"}):
                raise ValueError("database is not a local test target")

            from storage.database import connect

            with connect(database_url) as conn:
                actual = conn.execute(
                    "SELECT current_database() AS name, inet_server_addr()::text AS address"
                ).fetchone()
            server_address = ipaddress.ip_interface(actual["address"] or "").ip
            if (actual["name"].casefold() != configured_name.casefold()
                    or not server_address.is_loopback):
                raise ValueError("database target does not match the local test database")
        except Exception:
            return jsonify({"status": "blocked", "database_isolated": False}), 503

        return jsonify({
            "status": "ready", "service": "local_development",
            "database_isolated": True,
        })

    @app.route("/api/research/event-replay-readiness")
    def event_replay_readiness():
        """Certify both local stores and the frozen clock before event replay."""
        database_proof = research_replay_readiness()
        if isinstance(database_proof, tuple):
            return database_proof
        try:
            from experiments.event_replay_guard import certify_local_event_fixture
            from spatial import road_conditions

            fixture_proof = certify_local_event_fixture(
                road_conditions._CONDITIONS_FILE
            )
        except (OSError, ValueError):
            return jsonify({
                "status": "blocked", "database_isolated": True,
                "road_conditions_isolated": False,
            }), 503
        return jsonify({
            "status": "ready", "service": "local_development",
            "database_isolated": True, "road_conditions_isolated": True,
            **fixture_proof,
        })

    @app.after_request
    def _add_cache_headers(resp):
        path = request.path
        if (path == "/api/conversations" or path.startswith("/api/conversations/")
                or path.startswith("/api/runs/") or path.startswith("/api/admin/")
                or path == "/api/admin" or path.startswith("/api/manager/")
                or path.startswith("/api/research/")
                or path == "/api/manager" or path.startswith("/api/road-conditions")):
            resp.headers["Cache-Control"] = "private, no-store"
            resp.headers["Vary"] = "Cookie"
        if path.startswith("/static/") or path.startswith("/icons/") or path.startswith("/css/") or path.startswith("/js/"):
            resp.headers["Cache-Control"] = f"public, max-age={_cache_max_age(path)}"
        return resp

    if os.getenv("FLASK_ENV", "development") == "production":
        def _warmup_network():
            try:
                from spatial.network import load_or_download_network
                logger.info("[warmup] 预热路网中...")
                t0 = time.time()
                load_or_download_network()
                logger.info("[warmup] 路网预热完成，耗时 %.1fs", time.time() - t0)
            except Exception as e:
                logger.warning("[warmup] 路网预热失败 (不影响运行): %s", e)

        threading.Thread(target=_warmup_network, daemon=True).start()

    @app.route("/js/config.js")
    def serve_config_js():
        config_data = {
            "amapKey": config.AMAP_KEY or "",
            "amapSecurityCode": config.AMAP_SECURITY_CODE or "",
            "apiBase": "",
            "mapCenter": [config.MAP_CENTER["lng"], config.MAP_CENTER["lat"]],
            "mapZoom": config.MAP_ZOOM,
        }
        js = "window.WHU_WALKER_CONFIG = " + json.dumps(config_data, ensure_ascii=False) + ";"
        return Response(js, mimetype="application/javascript; charset=utf-8")

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "bad_request", "message": "请求参数不合法"}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not_found", "message": "资源不存在"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "internal_error", "message": "服务器内部错误"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    # Glitch/Cloud 等平台注入 PORT 环境变量，本地开发用 FLASK_PORT
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
