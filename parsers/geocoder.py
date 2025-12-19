import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import requests
from loguru import logger
from pymorphy3 import MorphAnalyzer
from shapely.geometry import Point, mapping
from tqdm.asyncio import tqdm

from flair.data import Sentence
from flair.models import SequenceTagger


PHOTON_URL = "https://photon.komoot.io/api"


class ModelsInit:
    def __init__(self) -> None:
        self._ner_model: SequenceTagger | None = None

    async def init_models(self) -> None:
        loop = asyncio.get_event_loop()
        ner_model_name = "Geor111y/flair-ner-addresses-extractor"
        logger.info(f"Launching NER model {ner_model_name} with Flair SequenceTagger")
        self._ner_model = await loop.run_in_executor(
            None, lambda: SequenceTagger.load(ner_model_name)
        )


models_initialization = ModelsInit()


@dataclass
class GeocodeResult:
    geometry: Point | None
    location: str | None
    osm_id: str | None
    source_text: str


class Geocoder:
    """Геокодер входящих текстовых сообщений через Photon с NER‑парсингом."""

    def __init__(self, bbox: str | None = None) -> None:
        self.bbox = bbox
        self._rate_limit = asyncio.Semaphore(1)
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

    async def fetch_photon(
        self, query: str, bbox: str | None, layers: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        async with self._rate_limit:
            def _request() -> List[Dict[str, Any]]:
                params: Dict[str, Any] = {"q": query, "layer": layers, "limit": limit}
                if bbox:
                    params["bbox"] = bbox
                resp = requests.get(PHOTON_URL, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json().get("features", [])

            features = await asyncio.to_thread(_request)
            await asyncio.sleep(1)
            return features

    async def process_text(self, text: str) -> GeocodeResult:
        texts = self.extract_address_texts(text)
        addr = self.parse_address_components(texts)
        street = addr.get("street_name", "").strip()
        house = addr.get("house_number", "").strip()

        if not street:
            return GeocodeResult(None, None, None, text)

        layers = ["house"] if house else ["street"]
        query = f"{house}, {street}" if house else street

        feats = await self.fetch_photon(query, self.bbox, layers, limit=1)

        if not feats:
            normalized = self.normalize_street_name(street)
            if normalized.lower() != street.lower():
                query = f"{house}, {normalized}" if house else normalized
                feats = await self.fetch_photon(query, self.bbox, layers, limit=1)

        if not feats:
            return GeocodeResult(None, None, None, text)

        feat = feats[0]
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        point = None
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            point = Point(lon, lat)

        name = props.get("name") or ", ".join(
            filter(
                None,
                [props.get("street", "").strip(), props.get("housenumber", "").strip()],
            )
        )
        osm_id = props.get("osm_id")

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


async def geocode_texts(texts: List[str], bbox: str | None) -> List[GeocodeResult]:
    await models_initialization.init_models()
    geocoder = Geocoder(bbox=bbox)
    results: List[GeocodeResult] = []
    for text in tqdm(texts, total=len(texts), desc="Geocoding"):
        clean = text.strip()
        if not clean:
            results.append(GeocodeResult(None, None, None, text))
            continue
        try:
            result = await geocoder.process_text(clean)
        except Exception as exc:
            logger.warning(f"Failed to geocode text: {clean}. Error: {exc}")
            result = GeocodeResult(None, None, None, clean)
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract toponyms with NER, geocode them via Photon, and save map/GeoJSON."
    )
    parser.add_argument("input", help="Path to a text file with one message per line.")
    parser.add_argument(
        "--bbox",
        help="Bounding box as 'minx,miny,maxx,maxy' to restrict Photon results.",
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

    results = asyncio.run(geocode_texts(texts, bbox=args.bbox))
    geojson_data = build_geojson(results)
    save_geojson(args.geojson, geojson_data)
    logger.info(f"Saved GeoJSON to {args.geojson}")

    save_map(args.map_path, results)
    logger.info(f"Saved map to {args.map_path}")


if __name__ == "__main__":
    main()
