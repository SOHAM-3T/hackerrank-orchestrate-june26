"""CSV, cache, and image utility functions."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

try:
    import pillow_avif  # noqa: F401
except Exception:
    pillow_avif = None

from PIL import Image, ImageOps

from .schema import OUTPUT_COLUMNS


def repo_root_from_code() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_output_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def load_lookup(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in read_csv(path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_image_path(dataset_root: Path, image_ref: str) -> Path:
    ref = image_ref.strip().replace("\\", "/")
    path = Path(ref)
    if path.is_absolute():
        return path
    if ref.startswith("images/"):
        return dataset_root / ref
    return repo_root_from_code() / ref


def image_id_from_ref(image_ref: str) -> str:
    return Path(image_ref.replace("\\", "/")).stem


def detect_image_kind(path: Path) -> str:
    header = path.read_bytes()[:16]
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header[4:12] in {b"ftypavif", b"ftypavis"}:
        return "avif"
    return "unknown"


def normalize_image_for_provider(source: Path, cache_dir: Path) -> tuple[Path, str, bool, str]:
    """Return provider-safe image path, mime type, valid flag, and note."""
    if not source.exists():
        return source, "application/octet-stream", False, "missing image file"

    kind = detect_image_kind(source)
    if kind in {"jpeg", "png", "webp"}:
        mime = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[kind]
        return source, mime, True, f"source format {kind}"

    digest = sha256_file(source)[:16]
    target = cache_dir / "normalized_images" / f"{source.stem}_{digest}.jpg"
    if target.exists():
        return target, "image/jpeg", True, "cached normalized image"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            img.save(target, "JPEG", quality=92, optimize=True)
        return target, "image/jpeg", True, f"converted from {kind}"
    except Exception as exc:
        return source, mimetypes.guess_type(source.name)[0] or "application/octet-stream", False, (
            f"could not normalize {kind}: {exc}"
        )


def image_data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
