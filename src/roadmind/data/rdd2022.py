"""ROAD-001B: RDD2022 discovery, image validation, and Pascal VOC parsing.

Canonical per-domain layout (matches the RDD2022 release structure):

    <domain>/
        train/
            images/       annotated images (``*.jpg`` / ``*.JPG``)
            annotations/  Pascal VOC XML files (``*.xml``)
        test/
            images/       optional unlabeled public-test images (excluded)

Domain directories are discovered with exact, case-sensitive names and the
fixed domain-to-split mapping from ROAD-001A. A dataset root is recognized
directly or through exactly one wrapper extraction directory.

Determinism contract:

- All results are sorted by normalized raw-root-relative POSIX path
  (samples by ``sample_id``, source files and excluded public-test images by
  path, issues by ``(code, domain, paths, sample_id)``).
- Hashing is byte-level stable: raw SHA-256 over file bytes, pixel SHA-256
  over ``struct.pack(">II", width, height) + RGB pixel bytes``, dHash64 from
  a 9x8 LANCZOS grayscale resize, and a semantic annotation SHA-256 over a
  canonical serialization of sorted annotations.
- Nothing is written to disk and no raw file is modified.

This module is package-internal; its symbols are not re-exported from
``roadmind.data``.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring
from PIL import Image
from pydantic import ValidationError

from roadmind.data.models import (
    _ISSUE_SEVERITY,
    RDD2022_DOMAIN_SPLITS,
    AnnotationRecord,
    AnnotationSource,
    BoundingBoxXYXY,
    DamageClass,
    DatasetContractError,
    DatasetContractErrorCode,
    DatasetDomain,
    DatasetIssue,
    DatasetIssueCode,
    DatasetSplit,
    ImageAsset,
    IssueDetail,
    SampleRecord,
    _validate_detail_string,
    _validate_relative_path,
)

_VALID_LABELS: frozenset[str] = frozenset({"D00", "D10", "D20", "D40"})
_IMAGE_SUFFIX = ".jpg"
_XML_SUFFIX = ".xml"


@dataclass(frozen=True)
class SourceFileRecord:
    """One raw source file: normalized relative path, byte size, raw SHA-256."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DomainLayout:
    """Resolved layout of one domain subset."""

    domain: DatasetDomain
    split: DatasetSplit
    images_dir: Path | None
    annotations_dir: Path | None
    public_test_dir: Path | None


@dataclass(frozen=True)
class RDD2022Layout:
    """Canonical dataset root and the seven domain layouts."""

    source_root: Path
    domain_layouts: tuple[DomainLayout, ...]


@dataclass(frozen=True)
class RDD2022ScanResult:
    """Deterministic scan output for one RDD2022 release."""

    samples: tuple[SampleRecord, ...]
    issues: tuple[DatasetIssue, ...]
    source_files: tuple[SourceFileRecord, ...]
    excluded_public_test: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedImage:
    width: int
    height: int
    size_bytes: int
    sha256: str
    pixel_sha256: str
    dhash64: str


def discover_rdd2022(source_root: Path) -> RDD2022Layout:
    """Resolve the canonical dataset root and per-domain layouts."""
    root = Path(source_root).resolve()
    if not root.exists():
        raise DatasetContractError(
            DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND,
            "dataset root does not exist",
        )
    if not root.is_dir():
        raise DatasetContractError(
            DatasetContractErrorCode.DATASET_ROOT_NOT_DIRECTORY,
            "dataset root is not a directory",
        )
    try:
        with os.scandir(root) as entries:
            children = sorted(
                (Path(entry.path) for entry in entries),
                key=lambda child: child.name,
            )
    except OSError:
        raise DatasetContractError(
            DatasetContractErrorCode.DATASET_ROOT_UNREADABLE,
            "dataset root is not readable",
        ) from None
    dataset_root = _resolve_dataset_root(root, children)
    layouts = tuple(
        _build_domain_layout(dataset_root, domain) for domain in DatasetDomain
    )
    return RDD2022Layout(source_root=dataset_root, domain_layouts=layouts)


def scan_rdd2022(
    layout: RDD2022Layout,
    *,
    max_xml_bytes: int = 1_048_576,
    max_image_pixels: int = 100_000_000,
) -> RDD2022ScanResult:
    """Scan a discovered layout read-only and aggregate all defects."""
    samples: list[SampleRecord] = []
    issues: list[DatasetIssue] = []
    source_files: list[SourceFileRecord] = []
    excluded_public_test: list[str] = []
    for domain_layout in layout.domain_layouts:
        _scan_domain(
            domain_layout,
            layout.source_root,
            max_xml_bytes,
            max_image_pixels,
            samples,
            issues,
            source_files,
            excluded_public_test,
        )
    return RDD2022ScanResult(
        samples=tuple(sorted(samples, key=lambda sample: sample.sample_id)),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
        source_files=tuple(sorted(source_files, key=lambda record: record.path)),
        excluded_public_test=tuple(sorted(excluded_public_test)),
    )


def _resolve_dataset_root(root: Path, children: Sequence[Path]) -> Path:
    if _matches_rdd2022(root):
        return root
    candidates = [
        child for child in children if child.is_dir() and _matches_rdd2022(child)
    ]
    if not candidates:
        raise DatasetContractError(
            DatasetContractErrorCode.UNSUPPORTED_RDD2022_LAYOUT,
            "no RDD2022 layout found under the dataset root",
        )
    if len(candidates) > 1:
        raise DatasetContractError(
            DatasetContractErrorCode.DATASET_ROOT_AMBIGUOUS,
            "multiple candidate RDD2022 extraction directories found",
        )
    return candidates[0]


def _matches_rdd2022(directory: Path) -> bool:
    try:
        return any((directory / domain.value).is_dir() for domain in DatasetDomain)
    except OSError:
        return False


def _build_domain_layout(dataset_root: Path, domain: DatasetDomain) -> DomainLayout:
    split = RDD2022_DOMAIN_SPLITS[domain]
    domain_dir = dataset_root / domain.value
    if not domain_dir.is_dir():
        return DomainLayout(domain, split, None, None, None)
    images_dir = domain_dir / "train" / "images"
    annotations_dir = domain_dir / "train" / "annotations"
    public_test_dir = domain_dir / "test" / "images"
    return DomainLayout(
        domain,
        split,
        images_dir if images_dir.is_dir() else None,
        annotations_dir if annotations_dir.is_dir() else None,
        public_test_dir if public_test_dir.is_dir() else None,
    )


def _scan_domain(
    domain_layout: DomainLayout,
    dataset_root: Path,
    max_xml_bytes: int,
    max_image_pixels: int,
    samples: list[SampleRecord],
    issues: list[DatasetIssue],
    source_files: list[SourceFileRecord],
    excluded_public_test: list[str],
) -> None:
    domain = domain_layout.domain
    if domain_layout.images_dir is None and domain_layout.annotations_dir is None:
        issues.append(_make_issue(DatasetIssueCode.MISSING_REQUIRED_DOMAIN, domain))
        return

    image_files, unsafe_image_dirs = (
        _walk_files(domain_layout.images_dir, dataset_root)
        if domain_layout.images_dir is not None
        else ([], [])
    )
    xml_files, unsafe_xml_dirs = (
        _walk_files(domain_layout.annotations_dir, dataset_root)
        if domain_layout.annotations_dir is not None
        else ([], [])
    )
    image_files = [file for file in image_files if file.suffix.lower() == _IMAGE_SUFFIX]
    xml_files = [file for file in xml_files if file.suffix.lower() == _XML_SUFFIX]
    for unsafe_dir in [*unsafe_image_dirs, *unsafe_xml_dirs]:
        issues.append(_unsafe_path_issue(domain, _rel_path(unsafe_dir, dataset_root)))

    image_rels = [_rel_path(file, dataset_root) for file in image_files]
    xml_rels = [_rel_path(file, dataset_root) for file in xml_files]
    for group in _find_case_collisions(image_rels):
        issues.append(
            _make_issue(DatasetIssueCode.PATH_CASE_COLLISION, domain, paths=group)
        )
    for group in _find_case_collisions(xml_rels):
        issues.append(
            _make_issue(DatasetIssueCode.PATH_CASE_COLLISION, domain, paths=group)
        )
    collided_images = {
        path for group in _find_case_collisions(image_rels) for path in group
    }
    collided_xmls = {
        path for group in _find_case_collisions(xml_rels) for path in group
    }

    xmls_by_stem: dict[str, Path] = {}
    for file in xml_files:
        rel = _rel_path(file, dataset_root)
        if _unsafe_file_reason(file, rel, dataset_root):
            issues.append(_unsafe_path_issue(domain, rel))
            continue
        source_files.append(_record_source(file, rel))
        if rel not in collided_xmls:
            xmls_by_stem[file.stem] = file

    valid_image_stems: set[str] = set()
    for file in image_files:
        rel = _rel_path(file, dataset_root)
        if _unsafe_file_reason(file, rel, dataset_root):
            issues.append(
                _unsafe_path_issue(domain, rel, sample_id=_sample_id(domain, file.stem))
            )
            continue
        source_files.append(_record_source(file, rel))
        if rel in collided_images:
            continue
        stem = file.stem
        sample_id = _sample_id(domain, stem)
        image_info, image_issues = _validate_image(
            file, domain, sample_id, rel, max_image_pixels
        )
        issues.extend(image_issues)
        if image_info is None:
            continue
        valid_image_stems.add(stem)
        xml_file = xmls_by_stem.get(stem)
        if xml_file is None:
            issues.append(
                _make_issue(
                    DatasetIssueCode.MISSING_ANNOTATION,
                    domain,
                    sample_id=sample_id,
                    paths=(rel,),
                )
            )
            continue
        xml_rel = _rel_path(xml_file, dataset_root)
        sample, sample_issues = _build_sample(
            domain,
            stem,
            sample_id,
            rel,
            xml_file,
            xml_rel,
            image_info,
            max_xml_bytes,
        )
        issues.extend(sample_issues)
        if sample is not None:
            samples.append(sample)

    for file in xml_files:
        rel = _rel_path(file, dataset_root)
        if _unsafe_file_reason(file, rel, dataset_root):
            continue
        if rel in collided_xmls:
            continue
        if file.stem not in valid_image_stems:
            issues.append(
                _make_issue(
                    DatasetIssueCode.ORPHAN_ANNOTATION,
                    domain,
                    paths=(rel,),
                )
            )

    if domain_layout.public_test_dir is not None:
        test_files, unsafe_test_dirs = _walk_files(
            domain_layout.public_test_dir, dataset_root
        )
        for unsafe_dir in unsafe_test_dirs:
            issues.append(
                _unsafe_path_issue(domain, _rel_path(unsafe_dir, dataset_root))
            )
        for file in test_files:
            rel = _rel_path(file, dataset_root)
            if _unsafe_file_reason(file, rel, dataset_root):
                issues.append(_unsafe_path_issue(domain, rel))
                continue
            source_files.append(_record_source(file, rel))
            excluded_public_test.append(rel)
            issues.append(
                _make_issue(
                    DatasetIssueCode.EXCLUDED_PUBLIC_TEST_IMAGE,
                    domain,
                    paths=(rel,),
                )
            )


def _validate_image(
    file: Path,
    domain: DatasetDomain,
    sample_id: str,
    rel: str,
    max_image_pixels: int,
) -> tuple[_ValidatedImage | None, list[DatasetIssue]]:
    try:
        data = file.read_bytes()
    except OSError:
        return None, [
            _make_issue(
                DatasetIssueCode.CORRUPT_IMAGE,
                domain,
                sample_id=sample_id,
                paths=(rel,),
            )
        ]
    if not data:
        return None, [
            _make_issue(
                DatasetIssueCode.CORRUPT_IMAGE,
                domain,
                sample_id=sample_id,
                paths=(rel,),
            )
        ]
    raw_sha256 = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception:
        return None, [
            _make_issue(
                DatasetIssueCode.CORRUPT_IMAGE,
                domain,
                sample_id=sample_id,
                paths=(rel,),
            )
        ]
    try:
        image = Image.open(io.BytesIO(data))
        if image.format != "JPEG":
            return None, [
                _make_issue(
                    DatasetIssueCode.UNSUPPORTED_IMAGE_FORMAT,
                    domain,
                    sample_id=sample_id,
                    paths=(rel,),
                )
            ]
        orientation = _exif_orientation(image)
        if orientation not in (None, 1):
            return None, [
                _make_issue(
                    DatasetIssueCode.UNSUPPORTED_EXIF_ORIENTATION,
                    domain,
                    sample_id=sample_id,
                    paths=(rel,),
                )
            ]
        width, height = image.size
        if width * height > max_image_pixels:
            return None, [
                _make_issue(
                    DatasetIssueCode.IMAGE_TOO_LARGE,
                    domain,
                    sample_id=sample_id,
                    paths=(rel,),
                )
            ]
        image.load()
        rgb = image.convert("RGB")
        pixel_bytes = rgb.tobytes()
    except Exception:
        return None, [
            _make_issue(
                DatasetIssueCode.CORRUPT_IMAGE,
                domain,
                sample_id=sample_id,
                paths=(rel,),
            )
        ]
    pixel_sha256 = hashlib.sha256(
        struct.pack(">II", width, height) + pixel_bytes
    ).hexdigest()
    dhash64 = _compute_dhash64(rgb)
    return (
        _ValidatedImage(
            width=width,
            height=height,
            size_bytes=len(data),
            sha256=raw_sha256,
            pixel_sha256=pixel_sha256,
            dhash64=dhash64,
        ),
        [],
    )


def _exif_orientation(image: Image.Image) -> int | None:
    exif = image.getexif()
    orientation = exif.get(0x0112)
    return int(orientation) if orientation is not None else None


def _compute_dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    bits = "".join(
        "1"
        if cast(int, gray.getpixel((x, y))) > cast(int, gray.getpixel((x + 1, y)))
        else "0"
        for y in range(8)
        for x in range(8)
    )
    return format(int(bits, 2), "016x")


def _build_sample(
    domain: DatasetDomain,
    stem: str,
    sample_id: str,
    image_rel: str,
    xml_file: Path,
    xml_rel: str,
    image_info: _ValidatedImage,
    max_xml_bytes: int,
) -> tuple[SampleRecord | None, list[DatasetIssue]]:
    issues: list[DatasetIssue] = []
    try:
        data = xml_file.read_bytes()
    except OSError:
        issues.append(
            _make_issue(
                DatasetIssueCode.MALFORMED_XML,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
            )
        )
        return None, issues
    xml_sha256 = hashlib.sha256(data).hexdigest()
    root, parse_issue = _parse_xml_document(data, max_xml_bytes)
    if parse_issue is not None:
        issues.append(
            _make_issue(
                parse_issue,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
            )
        )
        return None, issues
    if root is None or root.tag != "annotation":
        issues.append(
            _make_issue(
                DatasetIssueCode.MALFORMED_XML,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
            )
        )
        return None, issues

    image_filename = image_rel.rsplit("/", 1)[-1]
    if not _xml_filename_matches(root, image_filename):
        issues.append(
            _make_issue(
                DatasetIssueCode.ANNOTATION_FILENAME_MISMATCH,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
                details=(
                    IssueDetail(
                        key="actual",
                        value=_safe_detail_value(_xml_text(root, "filename")),
                    ),
                    IssueDetail(key="expected", value=image_filename),
                ),
            )
        )
        return None, issues
    declared_size = _xml_size(root)
    if declared_size != (image_info.width, image_info.height):
        issues.append(
            _make_issue(
                DatasetIssueCode.ANNOTATION_IMAGE_SIZE_MISMATCH,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
                details=(
                    IssueDetail(key="actual_height", value=image_info.height),
                    IssueDetail(key="actual_width", value=image_info.width),
                    IssueDetail(key="expected_height", value=declared_size[1]),
                    IssueDetail(key="expected_width", value=declared_size[0]),
                ),
            )
        )
        return None, issues

    annotations, has_error = _parse_objects(
        root, domain, sample_id, xml_rel, image_info.width, image_info.height, issues
    )
    if has_error:
        return None, issues
    if not annotations:
        issues.append(
            _make_issue(
                DatasetIssueCode.EMPTY_ANNOTATION,
                domain,
                sample_id=sample_id,
                paths=(xml_rel,),
            )
        )
    sorted_annotations = tuple(sorted(annotations, key=_annotation_sort_key))
    try:
        sample = SampleRecord(
            sample_id=sample_id,
            domain=domain,
            split=RDD2022_DOMAIN_SPLITS[domain],
            image=ImageAsset(
                path=image_rel,
                sha256=image_info.sha256,
                pixel_sha256=image_info.pixel_sha256,
                dhash64=image_info.dhash64,
                width=image_info.width,
                height=image_info.height,
                size_bytes=image_info.size_bytes,
                format="JPEG",
            ),
            annotation_source=AnnotationSource(
                path=xml_rel,
                sha256=xml_sha256,
                semantic_sha256=_semantic_sha256(sorted_annotations),
            ),
            annotations=sorted_annotations,
            is_negative=not sorted_annotations,
        )
    except ValidationError:
        raise DatasetContractError(
            DatasetContractErrorCode.INTERNAL_ERROR,
            "failed to construct a validated sample",
        ) from None
    return sample, issues


def _parse_xml_document(
    data: bytes, max_xml_bytes: int
) -> tuple[Any | None, DatasetIssueCode | None]:
    if len(data) > max_xml_bytes:
        return None, DatasetIssueCode.XML_TOO_LARGE
    try:
        return fromstring(data), None
    except DefusedXmlException:
        return None, DatasetIssueCode.UNSAFE_XML
    except Exception:
        return None, DatasetIssueCode.MALFORMED_XML


def _xml_text(root: Any, tag: str) -> str:
    element = root.find(tag)
    if element is None or element.text is None:
        return ""
    return str(element.text)


def _xml_filename_matches(root: Any, image_filename: str) -> bool:
    text = _xml_text(root, "filename")
    basename = text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return basename == image_filename


def _xml_size(root: Any) -> tuple[int, int]:
    size_element = root.find("size")
    width = (
        _parse_int_text(size_element.find("width"))
        if size_element is not None
        else None
    )
    height = (
        _parse_int_text(size_element.find("height"))
        if size_element is not None
        else None
    )
    return (width if width is not None else -1, height if height is not None else -1)


def _parse_int_text(element: Any | None) -> int | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    try:
        return int(text)
    except ValueError:
        return None


def _parse_objects(
    root: Any,
    domain: DatasetDomain,
    sample_id: str,
    xml_rel: str,
    image_width: int,
    image_height: int,
    issues: list[DatasetIssue],
) -> tuple[list[AnnotationRecord], bool]:
    annotations: list[AnnotationRecord] = []
    seen_keys: set[tuple[int, int, int, int, int]] = set()
    seen_boxes: dict[tuple[int, int, int, int], DamageClass] = {}
    has_error = False
    for object_element in root.findall("object"):
        label = _xml_text(object_element, "name")
        if label not in _VALID_LABELS:
            issues.append(
                _make_issue(
                    DatasetIssueCode.INVALID_CLASS_LABEL,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                    details=(
                        IssueDetail(key="label", value=_safe_detail_value(label)),
                    ),
                )
            )
            has_error = True
            continue
        bbox = _parse_bndbox(object_element)
        if bbox is None:
            issues.append(
                _make_issue(
                    DatasetIssueCode.NON_INTEGER_BBOX,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                )
            )
            has_error = True
            continue
        x_min, y_min, x_max, y_max = bbox
        if x_min < 1 or y_min < 1 or x_max > image_width or y_max > image_height:
            issues.append(
                _make_issue(
                    DatasetIssueCode.BBOX_OUT_OF_BOUNDS,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                )
            )
            has_error = True
            continue
        if x_min > x_max or y_min > y_max:
            issues.append(
                _make_issue(
                    DatasetIssueCode.INVALID_BBOX_ORDER,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                )
            )
            has_error = True
            continue
        category = DamageClass[label]
        converted = (x_min - 1, y_min - 1, x_max, y_max)
        key = (int(category), *converted)
        if key in seen_keys:
            issues.append(
                _make_issue(
                    DatasetIssueCode.EXACT_OBJECT_DUPLICATE,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                )
            )
            continue
        seen_keys.add(key)
        previous = seen_boxes.get(converted)
        if previous is not None and previous != category:
            issues.append(
                _make_issue(
                    DatasetIssueCode.OBJECT_LABEL_CONFLICT,
                    domain,
                    sample_id=sample_id,
                    paths=(xml_rel,),
                )
            )
            has_error = True
            continue
        seen_boxes[converted] = category
        annotations.append(
            AnnotationRecord(
                category_id=category,
                source_code=category.name,
                bbox=BoundingBoxXYXY(
                    x_min=converted[0],
                    y_min=converted[1],
                    x_max=converted[2],
                    y_max=converted[3],
                ),
            )
        )
    return annotations, has_error


def _parse_bndbox(object_element: Any) -> tuple[int, int, int, int] | None:
    bbox_element = object_element.find("bndbox")
    if bbox_element is None:
        return None
    values = []
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        value = _parse_int_text(bbox_element.find(tag))
        if value is None:
            return None
        values.append(value)
    return (values[0], values[1], values[2], values[3])


def _annotation_sort_key(
    annotation: AnnotationRecord,
) -> tuple[int, int, int, int, int]:
    bbox = annotation.bbox
    return (int(annotation.category_id), bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)


def _semantic_sha256(annotations: Iterable[AnnotationRecord]) -> str:
    parts = "|".join(
        (
            f"{annotation.category_id.name},{annotation.bbox.x_min},"
            f"{annotation.bbox.y_min},{annotation.bbox.x_max},"
            f"{annotation.bbox.y_max}"
        )
        for annotation in sorted(annotations, key=_annotation_sort_key)
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def _walk_files(base: Path, dataset_root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    unsafe_dirs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in list(dirnames):
            entry = Path(dirpath) / name
            if _is_link(entry):
                if _escapes_root(entry, dataset_root):
                    unsafe_dirs.append(entry)
                dirnames.remove(name)
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files, unsafe_dirs


def _find_case_collisions(paths: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Group normalized relative paths that differ only by case."""
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(path.lower(), []).append(path)
    collisions = [tuple(sorted(names)) for names in groups.values() if len(names) > 1]
    return tuple(sorted(collisions))


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
        except OSError:
            return False
    return False


def _escapes_root(path: Path, dataset_root: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return True
    return resolved != dataset_root and dataset_root not in resolved.parents


def _unsafe_file_reason(file: Path, rel: str, dataset_root: Path) -> bool:
    return not _is_safe_rel_path(rel) or (
        _is_link(file) and _escapes_root(file, dataset_root)
    )


def _is_safe_rel_path(rel: str) -> bool:
    try:
        _validate_relative_path(rel)
    except ValueError:
        return False
    return True


def _record_source(file: Path, rel: str) -> SourceFileRecord:
    data = file.read_bytes()
    return SourceFileRecord(
        path=rel,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _rel_path(file: Path, dataset_root: Path) -> str:
    return file.relative_to(dataset_root).as_posix()


def _sample_id(domain: DatasetDomain, stem: str) -> str:
    return f"rdd2022/{domain.value}/{stem}"


def _make_issue(
    code: DatasetIssueCode,
    domain: DatasetDomain,
    *,
    sample_id: str | None = None,
    paths: tuple[str, ...] = (),
    details: tuple[IssueDetail, ...] = (),
) -> DatasetIssue:
    return DatasetIssue(
        code=code,
        severity=_ISSUE_SEVERITY[code],
        domain=domain,
        sample_id=sample_id,
        paths=paths,
        details=details,
    )


def _issue_sort_key(issue: DatasetIssue) -> tuple[str, str, tuple[str, ...], str]:
    return (
        issue.code.value,
        issue.domain.value if issue.domain is not None else "",
        issue.paths,
        issue.sample_id or "",
    )


def _unsafe_path_issue(
    domain: DatasetDomain,
    rel: str,
    *,
    sample_id: str | None = None,
) -> DatasetIssue:
    return _make_issue(
        DatasetIssueCode.UNSAFE_SOURCE_PATH,
        domain,
        sample_id=sample_id,
        details=(IssueDetail(key="path", value=_safe_detail_value(rel)),),
    )


def _safe_detail_value(text: str) -> str:
    try:
        return _validate_detail_string(text)
    except ValueError:
        return "<unsafe>"
