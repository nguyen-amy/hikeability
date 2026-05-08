from flask import Blueprint, abort, current_app, jsonify, render_template

from .gcs import build_geojson, get_hike, load_latest_predictions

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template(
        "index.html",
        mapbox_token=current_app.config["MAPBOX_TOKEN"],
    )


@bp.route("/hike/<hike_id>")
def hike_detail(hike_id):
    hike = get_hike(hike_id, current_app.hikes_cache, current_app.gcs_client)
    if not hike:
        abort(404)
    return render_template(
        "hike.html",
        hike=hike,
        mapbox_token=current_app.config["MAPBOX_TOKEN"],
    )


@bp.route("/api/hikes.geojson")
def hikes_geojson():
    return jsonify(current_app.geojson_cache)


@bp.route("/about")
def about():
    return render_template("about.html")


@bp.route("/trails")
def trails():
    return render_template("trails.html")


@bp.route("/stats")
def stats():
    return render_template("stats.html")


@bp.route("/api/hike/<hike_id>/json")
def hike_json(hike_id):
    """Return full hike data as JSON for the side panel."""
    hike = get_hike(hike_id, current_app.hikes_cache, current_app.gcs_client)
    if not hike:
        abort(404)
    return jsonify(hike)


@bp.route("/api/debug/sample")
def debug_sample():
    """Return the first prediction dict so we can inspect its fields."""
    cache = current_app.hikes_cache
    return jsonify(cache[0] if cache else {})


@bp.route("/api/refresh", methods=["POST"])
def refresh():
    """Reload predictions from GCS without restarting the server."""
    current_app.hikes_cache = load_latest_predictions(current_app.gcs_client)
    current_app.geojson_cache = build_geojson(current_app.hikes_cache)
    return jsonify({"status": "ok", "hikes": len(current_app.hikes_cache)})
