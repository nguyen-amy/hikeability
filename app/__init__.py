import os

from flask import Flask

from .gcs import build_geojson, get_client, load_latest_predictions


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAPBOX_TOKEN"] = os.environ["MAPBOX_TOKEN"]

    # Load predictions once at startup and cache in memory.
    # To refresh without restarting, hit GET /api/refresh (see routes.py).
    app.gcs_client = get_client()
    app.hikes_cache = load_latest_predictions(app.gcs_client, date="2026-04-14")
    app.geojson_cache = build_geojson(app.hikes_cache)
    print(f"[hikeability] Loaded {len(app.hikes_cache)} hikes from GCS")

    from .routes import bp
    app.register_blueprint(bp)

    return app
