import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import requests
from pymorphy3 import MorphAnalyzer
from shapely.geometry import Point, mapping
from tqdm import tqdm

from flair.data import Sentence
from flair.models import SequenceTagger


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REQUEST_SLEEP_SEC = 1.0
NOMINATIM_ERROR_SLEEP_SEC = 5.0
NOMINATIM_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


class ModelsInit:
    def __init__(self) -> None:
        self._ner_model: SequenceTagger | None = None

    def init_models(self) -> None:
        ner_model_name = "Geor111y/flair-ner-addresses-extractor"
        print(f"Launching NER model {ner_model_name} with Flair SequenceTagger")
        with tqdm(total=1, desc="Loading NER model", unit="model") as progress:
            self._ner_model = SequenceTagger.load(ner_model_name)
            progress.update(1)


models_initialization = ModelsInit()


@dataclass
class GeocodeResult:
    geometry: Point | None
    location: str | None
    osm_id: str | None
    source_text: str


class Geocoder:
    """Геокодер входящих текстовых сообщений через Nominatim с NER‑парсингом."""

    def __init__(self, bbox: str | None = None) -> None:
        self.bbox = bbox
        self.morph = MorphAnalyzer()

    def normalize_street_name(self, raw: str) -> str:
        """Лемматизирует каждое слово улицы в именительный падеж."""
        tokens = raw.split()
        lemmas = []
        for tok in tokens:
            p = self.morph.parse(tok)[0]
            nom = p.inflect({"nomn"})
            lemmas.append(nom.word if nom else p.normal_form)
        return " ".join(lemmas)

    @staticmethod
    def extract_address_texts(text: str) -> List[str]:
        tagger = models_initialization._ner_model
        if tagger is None:
            raise RuntimeError("NER model is not initialized. Call init_models() first.")
        sent = Sentence(text)
        tagger.predict(sent)
        spans = sent.get_spans("ner")
        return [span.text for span in spans][:2]

    @staticmethod
    def parse_address_components(texts: List[str]) -> Dict[str, str]:
        result = {"street_name": "", "house_number": ""}
        for elem in texts:
            if result["street_name"] and result["house_number"]:
                break
            tokens = elem.split()
            str_tokens: list[str] = []
            for token in tokens:
                if token.isdigit():
                    if not result["house_number"]:
                        result["house_number"] = token
                else:
                    str_tokens.append(token)
            if str_tokens and not result["street_name"]:
                result["street_name"] = " ".join(str_tokens)
        return result

    def fetch_nominatim(
        self, query: str, bbox: str | None, limit: int
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": limit,
        }
        if bbox:
            minx, miny, maxx, maxy = map(float, bbox.split(","))
            params["viewbox"] = f"{minx},{maxy},{maxx},{miny}"
            params["bounded"] = 1

        headers = {"User-Agent": "digital-identity-geocoder/1.0"}
        last_exc: Exception | None = None
        for attempt in range(1, NOMINATIM_MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    NOMINATIM_URL, params=params, headers=headers, timeout=30
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"Nominatim error status {resp.status_code}",
                        response=resp,
                    )
                resp.raise_for_status()
                results = resp.json()
                time.sleep(NOMINATIM_REQUEST_SLEEP_SEC)
                return results
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Nominatim request failed on attempt %s/%s: %s",
                    attempt,
                    NOMINATIM_MAX_RETRIES,
                    exc,
                )
                if attempt < NOMINATIM_MAX_RETRIES:
                    time.sleep(NOMINATIM_ERROR_SLEEP_SEC)
                    continue
                break
        if last_exc:
            raise last_exc
        return []

    def process_text(self, text: str) -> GeocodeResult:
        texts = self.extract_address_texts(text)
        addr = self.parse_address_components(texts)
        street = addr.get("street_name", "").strip()
        house = addr.get("house_number", "").strip()

        if not street:
            return GeocodeResult(None, None, None, text)

        query = f"{house}, {street}" if house else street

        feats = self.fetch_nominatim(query, self.bbox, limit=1)

        if not feats:
            normalized = self.normalize_street_name(street)
            if normalized.lower() != street.lower():
                query = f"{house}, {normalized}" if house else normalized
                feats = self.fetch_nominatim(query, self.bbox, limit=1)

        if not feats:
            return GeocodeResult(None, None, None, text)

        feat = feats[0]
        point = None
        if "lon" in feat and "lat" in feat:
            point = Point(float(feat["lon"]), float(feat["lat"]))

        address = feat.get("address", {})
        name_parts = [
            address.get("road", "").strip(),
            address.get("house_number", "").strip(),
        ]
        name = ", ".join(filter(None, name_parts)) or feat.get("display_name")
        osm_id = feat.get("osm_id")

        return GeocodeResult(point, name or None, osm_id, text)


def build_geojson(results: Iterable[GeocodeResult]) -> Dict[str, Any]:
    features = []
    for res in results:
        if res.geometry is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(res.geometry),
                "properties": {
                    "location": res.location,
                    "osm_id": res.osm_id,
                    "text": res.source_text,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def save_geojson(path: str, geojson_data: Dict[str, Any]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(geojson_data, fh, ensure_ascii=False, indent=2)


def save_map(path: str, results: List[GeocodeResult]) -> None:
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("folium is required to save the map output") from exc

    points = [res.geometry for res in results if res.geometry is not None]
    if points:
        avg_lat = sum(pt.y for pt in points) / len(points)
        avg_lon = sum(pt.x for pt in points) / len(points)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    else:
        m = folium.Map(location=[0, 0], zoom_start=2)

    for res in results:
        if res.geometry is None:
            continue
        tooltip = res.location or "geocoded point"
        popup = f"{res.source_text}\n{res.location or ''}".strip()
        folium.Marker(
            location=[res.geometry.y, res.geometry.x],
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

    m.save(path)


def bbox_from_area_name(area_name: str) -> str:
    try:
        import osmnx as ox
    except ImportError as exc:
        logger.info("osmnx is unavailable, falling back to Nominatim for bbox lookup.")
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": area_name, "format": "json", "limit": 1},
                headers={"User-Agent": "digital-identity-geocoder/1.0"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as request_exc:
            raise RuntimeError(
                "Failed to resolve bbox via Nominatim. Install osmnx or check network access."
            ) from request_exc
        data = response.json()
        if not data:
            raise ValueError(f"Territory not found: {area_name}")
        bbox = data[0].get("boundingbox")
        if not bbox or len(bbox) != 4:
            raise RuntimeError(f"Unexpected bbox response for: {area_name}")
        south, north, west, east = map(float, bbox)
        return ",".join(f"{value:.6f}" for value in (west, south, east, north))

    gdf = ox.geocode_to_gdf(area_name)
    if gdf.empty:
        raise ValueError(f"Territory not found: {area_name}")
    minx, miny, maxx, maxy = gdf.total_bounds
    return ",".join(f"{value:.6f}" for value in (minx, miny, maxx, maxy))


def geocode_texts(
    texts: List[str], bbox: str | None, bbox_name: str | None = None
) -> List[GeocodeResult]:
    if not bbox and bbox_name:
        bbox = bbox_from_area_name(bbox_name)
    models_initialization.init_models()
    geocoder = Geocoder(bbox=bbox)
    results: List[GeocodeResult] = []
    for text in tqdm(texts, total=len(texts), desc="Geocoding"):
        clean = text.strip()
        if not clean:
            results.append(GeocodeResult(None, None, None, text))
            continue
        try:
            result = geocoder.process_text(clean)
        except Exception as exc:
            logger.warning(f"Failed to geocode text: {clean}. Error: {exc}")
            result = GeocodeResult(None, None, None, clean)
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract toponyms with NER, geocode them via Nominatim, and save map/GeoJSON."
    )
    parser.add_argument("input", help="Path to a text file with one message per line.")
    parser.add_argument(
        "--bbox",
        help="Bounding box as 'minx,miny,maxx,maxy' to restrict Nominatim results.",
    )
    parser.add_argument(
        "--bbox-name",
        help="Resolve bbox by territory name using OpenStreetMap (requires osmnx).",
    )
    parser.add_argument(
        "--geojson",
        default="geocoded_points.geojson",
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        default="geocoded_points.html",
        help="Output HTML map path.",
    )
    return parser.parse_args()


def load_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def main() -> None:
    args = parse_args()
    texts = load_texts(args.input)
    logger.info(f"Loaded {len(texts)} texts for geocoding")

    results = geocode_texts(texts, bbox=args.bbox, bbox_name=args.bbox_name)
    geojson_data = build_geojson(results)
    save_geojson(args.geojson, geojson_data)
    logger.info(f"Saved GeoJSON to {args.geojson}")

    save_map(args.map_path, results)
    logger.info(f"Saved map to {args.map_path}")


if __name__ == "__main__":
    main()
