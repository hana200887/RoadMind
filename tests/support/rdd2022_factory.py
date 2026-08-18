"""Synthetic RDD2022 fixture factory for ROAD-001B tests.

Builds canonical RDD2022 trees (``<domain>/train/images``,
``<domain>/train/annotations``, optional ``<domain>/test/images`` public
test) with tiny valid JPEGs and Pascal VOC XMLs. Everything is generated in
memory and written under a caller-provided scratch directory; no real
RDD2022 data is required.
"""

from __future__ import annotations

import io
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from roadmind.data import DatasetDomain

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 48
DEFAULT_COLOR = (16, 32, 48)
DEFAULT_OBJECTS: Sequence[tuple[str, tuple[int, int, int, int]]] = (
    ("D00", (1, 1, 10, 10)),
)


def jpeg_bytes(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    *,
    mode: str = "RGB",
    color: tuple[int, int, int] = DEFAULT_COLOR,
    quality: int = 90,
    exif_orientation: int | None = None,
) -> bytes:
    """Encode a tiny solid-color JPEG in memory."""
    fill: int | tuple[int, int, int] = color[0] if mode == "L" else color
    image = Image.new(mode, (width, height), fill)
    buffer = io.BytesIO()
    if exif_orientation is None:
        image.save(buffer, format="JPEG", quality=quality)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(buffer, format="JPEG", quality=quality, exif=exif)
    return buffer.getvalue()


def jpeg_gradient_bytes(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    *,
    quality: int = 90,
) -> bytes:
    """Encode a horizontal grayscale gradient JPEG (non-trivial dHash)."""
    image = Image.new("L", (width, height))
    for x in range(width):
        level = round(x * 255 / max(width - 1, 1))
        for y in range(height):
            image.putpixel((x, y), level)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def voc_xml(
    filename: str,
    width: int,
    height: int,
    objects: Sequence[tuple[str, tuple[int, int, int, int]]] | None = None,
) -> str:
    """Render a Pascal VOC XML document as text."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "  </size>",
    ]
    for name, (xmin, ymin, xmax, ymax) in objects or ():
        parts.append("  <object>")
        parts.append(f"    <name>{name}</name>")
        parts.append("    <bndbox>")
        parts.append(f"      <xmin>{xmin}</xmin>")
        parts.append(f"      <ymin>{ymin}</ymin>")
        parts.append(f"      <xmax>{xmax}</xmax>")
        parts.append(f"      <ymax>{ymax}</ymax>")
        parts.append("    </bndbox>")
        parts.append("  </object>")
    parts.append("</annotation>")
    return "\n".join(parts)


def write_bytes(directory: Path, name: str, content: bytes) -> Path:
    """Write raw bytes under ``directory``, creating parents as needed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def write_jpeg(directory: Path, stem: str, **kwargs: object) -> Path:
    return write_bytes(directory, f"{stem}.jpg", jpeg_bytes(**kwargs))  # type: ignore[arg-type]


def write_voc(
    directory: Path,
    stem: str,
    *,
    raw: str | None = None,
    filename: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    objects: Sequence[tuple[str, tuple[int, int, int, int]]] | None = DEFAULT_OBJECTS,
) -> Path:
    content = (
        raw
        if raw is not None
        else voc_xml(
            filename if filename is not None else f"{stem}.jpg", width, height, objects
        )
    )
    return write_bytes(directory, f"{stem}.xml", content.encode("utf-8"))


def build_domain(
    dataset_root: Path,
    domain: DatasetDomain,
    *,
    image_count: int = 2,
    object_spec: Sequence[tuple[str, tuple[int, int, int, int]]]
    | None = DEFAULT_OBJECTS,
    public_test_count: int = 0,
) -> Path:
    """Build one canonical domain subset (``train/images``, ``train/annotations``)."""
    domain_dir = dataset_root / domain.value
    images_dir = domain_dir / "train" / "images"
    annotations_dir = domain_dir / "train" / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, image_count + 1):
        stem = f"{domain.value}_{index:05d}"
        write_jpeg(images_dir, stem)
        write_voc(annotations_dir, stem, objects=object_spec)
    if public_test_count:
        for index in range(1, public_test_count + 1):
            write_jpeg(
                domain_dir / "test" / "images", f"{domain.value}_test_{index:05d}"
            )
    return domain_dir


def build_rdd2022(
    dataset_root: Path,
    *,
    image_count: int = 2,
) -> Path:
    """Build the happy-path tree with all seven domains."""
    for domain in DatasetDomain:
        build_domain(dataset_root, domain, image_count=image_count)
    return dataset_root


def make_directory_link(link: Path, target: Path) -> None:
    """Create a directory link (junction on Windows, symlink elsewhere)."""
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)
