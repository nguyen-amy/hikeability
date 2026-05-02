"""
GCS data access layer for the Flask web app.
Reads predictions and weather data; builds GeoJSON for Mapbox.
"""
from __future__ import annotations

import json

from google.cloud import storage

# Bucket / prefix constants (mirrors classification/config.py)
_BUCKET_OUTPUT  = "hikes-model-output"
_PRED_PREFIX    = "predictions"
_BUCKET_WEATHER = "weather-conditions"
_WEATHER_PREFIX = "weather-scraped"
_BUCKET_RAW     = "wta-hikes"
_RAW_PREFIX     = "output/hikes"

LABEL_COLORS = {
    "hikeable":   "#54B393",
    "modest":     "#F7A745",
    "unhikeable": "#DC4848",
}


def get_client() -> storage.Client:
    return storage.Client()


def load_latest_predictions(client: storage.Client, date: str | None = None) -> list[dict]:
    """
    Load all predictions from a specific date folder in GCS.
    If date is None, uses the most recent folder.
    Deduplicates by hike_id, keeping the last occurrence.
    """
    bucket = client.bucket(_BUCKET_OUTPUT)

    if date:
        prefix = f"{_PRED_PREFIX}/{date}/"
    else:
        blobs = bucket.list_blobs(prefix=f"{_PRED_PREFIX}/", delimiter="/")
        list(blobs)  # force iteration to populate prefixes
        date_prefixes = sorted(blobs.prefixes, reverse=True)
        if not date_prefixes:
            return []
        prefix = date_prefixes[0]

    all_preds: list[dict] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".json"):
            data = json.loads(blob.download_as_text())
            if isinstance(data, list):
                all_preds.extend(data)

    # Deduplicate — last write wins
    seen: dict[str, dict] = {}
    for p in all_preds:
        seen[p["hike_id"]] = p
    predictions = list(seen.values())

    _enrich_coordinates(client, predictions)
    return predictions


def _enrich_coordinates(client: storage.Client, predictions: list[dict]) -> None:
    """Fetch lat/lng from raw metadata.json for any prediction missing coordinates."""
    bucket = client.bucket(_BUCKET_RAW)
    for p in predictions:
        if p.get("latitude") is not None and p.get("longitude") is not None:
            continue
        blob = bucket.blob(f"{_RAW_PREFIX}/{p['hike_id']}/metadata.json")
        if not blob.exists():
            continue
        meta = json.loads(blob.download_as_text())
        p["latitude"]  = _to_float(meta.get("latitude"))
        p["longitude"] = _to_float(meta.get("longitude"))
        # Backfill other missing fields while we're here
        if not p.get("hike_name"):
            p["hike_name"] = meta.get("name", p["hike_id"])
        if not p.get("url"):
            p["url"] = meta.get("url")
        if not p.get("elevation_gain"):
            p["elevation_gain"] = meta.get("elevation_gain")
        if not p.get("highest_point"):
            p["highest_point"] = meta.get("highest_point")
        if not p.get("distance"):
            p["distance"] = meta.get("distance")
        if not p.get("rating"):
            p["rating"] = meta.get("rating")


def get_hike(hike_id: str, all_predictions: list[dict], client: storage.Client) -> dict | None:
    """
    Return a single hike's prediction dict enriched with a parsed weather summary.
    Returns None if the hike_id is not found.
    """
    hike = next((p for p in all_predictions if p["hike_id"] == hike_id), None)
    if not hike:
        return None

    hike = dict(hike)  # don't mutate the cache
    hike["weather"] = _fetch_weather_summary(client, hike_id)
    hike["label_color"] = LABEL_COLORS.get(hike.get("predicted_label", ""), "#888888")
    return hike


def build_geojson(predictions: list[dict]) -> dict:
    """Convert predictions list into a GeoJSON FeatureCollection for Mapbox."""
    features = []
    for p in predictions:
        lat = p.get("latitude")
        lng = p.get("longitude")
        if lat is None or lng is None:
            continue
        label = p.get("predicted_label", "unknown")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {
                "hike_id":        p["hike_id"],
                "name":           p.get("hike_name", p["hike_id"]),
                "label":          label,
                "color":          LABEL_COLORS.get(label, "#888888"),
                "explanation":    p.get("label_explanation", p.get("explanation", "")),
                "region":         p.get("hike_region", ""),
                "distance":       p.get("distance"),
                "elevation_gain": p.get("elevation_gain"),
                "classification_source": p.get("classification_source", ""),
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Snow showers", 95: "Thunderstorm",
}


def _fetch_weather_summary(client: storage.Client, hike_id: str) -> dict:
    """Fetch and parse weather JSON for a hike. Returns empty dict on miss."""
    bucket = client.bucket(_BUCKET_WEATHER)
    blobs = bucket.list_blobs(prefix=f"{_WEATHER_PREFIX}/", delimiter="/")
    list(blobs)
    date_prefixes = sorted(blobs.prefixes, reverse=True)
    if not date_prefixes:
        return {}

    weather_date = date_prefixes[0].rstrip("/").split("/")[-1]
    blob = bucket.blob(f"{_WEATHER_PREFIX}/{weather_date}/{hike_id}/open_meteo_24h.json")
    if not blob.exists():
        return {}

    raw = json.loads(blob.download_as_text())
    return _parse_weather(raw)


def _parse_weather(raw: dict) -> dict:
    out = {
        "temp_f": None,
        "aqi": None,
        "snow_depth_in": None,
        "wind_gusts_mph": None,
        "precip_chance_pct": None,
        "description": None,
        "fetched_at": raw.get("fetched_at_utc"),
    }
    hourly = raw.get("hourly_forecast") or []
    if hourly:
        h = hourly[0]
        out["temp_f"] = _to_float(h.get("apparent_temperature"))
        out["aqi"] = _to_float(h.get("us_aqi"))
        snow_m = _to_float(h.get("snow_depth"))
        if snow_m is not None:
            out["snow_depth_in"] = round(snow_m * 39.37, 1)

    daily = raw.get("daily_summary") or {}
    gusts_kmh = _to_float(daily.get("wind_gusts_10m_max"))
    if gusts_kmh is not None:
        out["wind_gusts_mph"] = round(gusts_kmh * 0.6214, 1)
    out["precip_chance_pct"] = _to_float(daily.get("precipitation_probability_max"))
    code = daily.get("weather_code")
    if code is not None:
        out["description"] = _WMO_CODES.get(int(float(code)), f"Code {code}")

    return out
