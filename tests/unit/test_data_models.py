"""Unit tests for the ROAD-001A typed data contracts.

These tests pin the exact public contract of ``roadmind.data``: taxonomy,
domain splits, issue vocabulary, model invariants, and package hygiene.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from roadmind import data as data_pkg
from roadmind.data import (
    RDD2022_CLASS_NAMES,
    RDD2022_DATASET_ID,
    RDD2022_DOMAIN_SPLITS,
    RDD2022_MANIFEST_SCHEMA_VERSION,
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
    IssueSeverity,
    SampleRecord,
)
from roadmind.data.models import IssueDetail

VALID_SHA256_A = "a" * 64
VALID_SHA256_B = "b" * 64
VALID_DHASH64 = "0123456789abcdef"


def make_image(
    path: str = "Japan/Japan_000001.jpg",
    width: int = 640,
    height: int = 480,
    *,
    sha256: str = VALID_SHA256_A,
    pixel_sha256: str = VALID_SHA256_B,
    dhash64: str = VALID_DHASH64,
    size_bytes: int = 12345,
    format: str = "JPEG",
) -> ImageAsset:
    return ImageAsset(
        path=path,
        sha256=sha256,
        pixel_sha256=pixel_sha256,
        dhash64=dhash64,
        width=width,
        height=height,
        size_bytes=size_bytes,
        format=format,
    )


def make_annotation_source(
    path: str = "Japan/Japan_000001.xml",
    *,
    sha256: str = VALID_SHA256_A,
    semantic_sha256: str = VALID_SHA256_B,
) -> AnnotationSource:
    return AnnotationSource(
        path=path,
        sha256=sha256,
        semantic_sha256=semantic_sha256,
    )


def make_annotation(
    category: DamageClass = DamageClass.D00,
    x_min: int = 0,
    y_min: int = 0,
    x_max: int = 10,
    y_max: int = 10,
) -> AnnotationRecord:
    return AnnotationRecord(
        category_id=category,
        source_code=category.name,
        bbox=BoundingBoxXYXY(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
    )


def make_sample(
    domain: DatasetDomain = DatasetDomain.JAPAN,
    annotations: tuple[AnnotationRecord, ...] = (
        make_annotation(),
        make_annotation(
            category=DamageClass.D40,
            x_min=20,
            y_min=20,
            x_max=30,
            y_max=30,
        ),
    ),
    *,
    split: DatasetSplit | None = None,
    sample_id: str | None = None,
    is_negative: bool | None = None,
    image: ImageAsset | None = None,
    annotation_source: AnnotationSource | None = None,
) -> SampleRecord:
    image = image if image is not None else make_image()
    annotation_source = (
        annotation_source if annotation_source is not None else make_annotation_source()
    )
    stem = Path(image.path).stem
    return SampleRecord(
        sample_id=sample_id
        if sample_id is not None
        else f"rdd2022/{domain.value}/{stem}",
        domain=domain,
        split=split if split is not None else RDD2022_DOMAIN_SPLITS[domain],
        image=image,
        annotation_source=annotation_source,
        annotations=annotations,
        is_negative=not annotations if is_negative is None else is_negative,
    )


class TestEnumsAndConstants:
    def test_damage_class_values(self) -> None:
        assert [(c.name, c.value) for c in DamageClass] == [
            ("D00", 0),
            ("D10", 1),
            ("D20", 2),
            ("D40", 3),
        ]
        assert len(DamageClass) == 4

    def test_canonical_class_names(self) -> None:
        assert dict(RDD2022_CLASS_NAMES) == {
            DamageClass.D00: "longitudinal_crack",
            DamageClass.D10: "transverse_crack",
            DamageClass.D20: "alligator_crack",
            DamageClass.D40: "pothole",
        }

    def test_class_names_mapping_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            RDD2022_CLASS_NAMES[DamageClass.D00] = "mutated"  # type: ignore[index]

    def test_domain_values(self) -> None:
        assert [(d.name, d.value) for d in DatasetDomain] == [
            ("JAPAN", "Japan"),
            ("INDIA", "India"),
            ("CZECH", "Czech"),
            ("NORWAY", "Norway"),
            ("UNITED_STATES", "United_States"),
            ("CHINA_MOTORBIKE", "China_MotorBike"),
            ("CHINA_DRONE", "China_Drone"),
        ]
        assert len(DatasetDomain) == 7

    def test_split_values(self) -> None:
        assert [(s.name, s.value) for s in DatasetSplit] == [
            ("TRAIN", "train"),
            ("VALIDATION", "validation"),
            ("TEST", "test"),
            ("OOD", "ood"),
        ]

    def test_domain_split_mapping(self) -> None:
        assert dict(RDD2022_DOMAIN_SPLITS) == {
            DatasetDomain.JAPAN: DatasetSplit.TRAIN,
            DatasetDomain.INDIA: DatasetSplit.TRAIN,
            DatasetDomain.CZECH: DatasetSplit.VALIDATION,
            DatasetDomain.NORWAY: DatasetSplit.TRAIN,
            DatasetDomain.UNITED_STATES: DatasetSplit.TRAIN,
            DatasetDomain.CHINA_MOTORBIKE: DatasetSplit.TEST,
            DatasetDomain.CHINA_DRONE: DatasetSplit.OOD,
        }

    def test_domain_split_mapping_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            RDD2022_DOMAIN_SPLITS[DatasetDomain.JAPAN] = DatasetSplit.OOD  # type: ignore[index]

    def test_dataset_id_and_schema_version(self) -> None:
        assert RDD2022_DATASET_ID == "rdd2022-roadmind-four-class-v1"
        assert RDD2022_MANIFEST_SCHEMA_VERSION == "1.0.0"

    def test_issue_severity_values(self) -> None:
        assert [(s.name, s.value) for s in IssueSeverity] == [
            ("INFO", "info"),
            ("WARNING", "warning"),
            ("ERROR", "error"),
            ("REVIEW_REQUIRED", "review_required"),
        ]

    def test_issue_code_vocabulary(self) -> None:
        expected = {
            "MISSING_REQUIRED_DOMAIN",
            "UNSAFE_SOURCE_PATH",
            "PATH_CASE_COLLISION",
            "MISSING_ANNOTATION",
            "ORPHAN_ANNOTATION",
            "MALFORMED_XML",
            "UNSAFE_XML",
            "XML_TOO_LARGE",
            "ANNOTATION_FILENAME_MISMATCH",
            "ANNOTATION_IMAGE_SIZE_MISMATCH",
            "INVALID_CLASS_LABEL",
            "NON_INTEGER_BBOX",
            "INVALID_BBOX_ORDER",
            "BBOX_OUT_OF_BOUNDS",
            "OBJECT_LABEL_CONFLICT",
            "CORRUPT_IMAGE",
            "IMAGE_TOO_LARGE",
            "UNSUPPORTED_IMAGE_FORMAT",
            "UNSUPPORTED_EXIF_ORIENTATION",
            "EXACT_DUPLICATE_CROSS_SPLIT",
            "PIXEL_DUPLICATE_CROSS_SPLIT",
            "DUPLICATE_LABEL_CONFLICT",
            "PUBLISHED_INVENTORY_MISMATCH",
            "SOURCE_FILE_ADDED",
            "SOURCE_FILE_REMOVED",
            "SOURCE_FILE_CHANGED",
            "EMPTY_ANNOTATION",
            "EXACT_OBJECT_DUPLICATE",
            "EXACT_DUPLICATE_WITHIN_SPLIT",
            "PIXEL_DUPLICATE_WITHIN_SPLIT",
            "NEAR_DUPLICATE_WITHIN_SPLIT",
            "NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE",
            "EXCLUDED_PUBLIC_TEST_IMAGE",
        }
        assert {c.name for c in DatasetIssueCode} == expected
        assert len(DatasetIssueCode) == 33

    def test_contract_error_code_vocabulary(self) -> None:
        expected = {
            "DATASET_ROOT_NOT_FOUND",
            "DATASET_ROOT_NOT_DIRECTORY",
            "DATASET_ROOT_UNREADABLE",
            "DATASET_ROOT_AMBIGUOUS",
            "UNSUPPORTED_RDD2022_LAYOUT",
            "OUTPUT_INSIDE_RAW_ROOT",
            "OUTPUT_ALREADY_EXISTS",
            "OUTPUT_PARENT_UNWRITABLE",
            "OUTPUT_COMMIT_FAILED",
            "MANIFEST_SCHEMA_INVALID",
            "UNSUPPORTED_SCHEMA_VERSION",
            "INTERNAL_ERROR",
        }
        assert {c.name for c in DatasetContractErrorCode} == expected
        assert len(DatasetContractErrorCode) == 12


class TestBoundingBox:
    def test_valid_box(self) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=10, y_max=10)
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0, 0, 10, 10)

    def test_full_image_box(self) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=640, y_max=480)
        assert box.as_normalized_xywh(640, 480) == (0.5, 0.5, 1.0, 1.0)

    def test_one_pixel_box(self) -> None:
        box = BoundingBoxXYXY(x_min=5, y_min=5, x_max=6, y_max=6)
        assert box.as_normalized_xywh(8, 8) == (0.6875, 0.6875, 0.125, 0.125)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x_min": -1, "y_min": 0, "x_max": 5, "y_max": 5},
            {"x_min": 0, "y_min": -1, "x_max": 5, "y_max": 5},
        ],
        ids=["negative-x", "negative-y"],
    )
    def test_negative_coordinates_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x_min": 2, "y_min": 0, "x_max": 2, "y_max": 5},
            {"x_min": 0, "y_min": 2, "x_max": 5, "y_max": 2},
        ],
        ids=["zero-width", "zero-height"],
    )
    def test_zero_area_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(**kwargs)

    def test_reversed_coordinates_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(x_min=5, y_min=5, x_max=2, y_max=2)

    def test_float_coordinates_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(x_min=1.5, y_min=0, x_max=5, y_max=5)

    def test_string_coordinates_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(x_min="1", y_min=0, x_max=5, y_max=5)

    def test_boolean_coordinates_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(x_min=True, y_min=0, x_max=5, y_max=5)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5, z=1)  # type: ignore[call-arg]

    def test_frozen_assignment_rejected(self) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(ValidationError):
            box.x_min = 1  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("box", "width", "height", "expected"),
        [
            (
                BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5),
                10,
                10,
                (0.25, 0.25, 0.5, 0.5),
            ),
            (
                BoundingBoxXYXY(x_min=2, y_min=2, x_max=10, y_max=10),
                16,
                16,
                (0.375, 0.375, 0.5, 0.5),
            ),
            (
                BoundingBoxXYXY(x_min=0, y_min=0, x_max=1, y_max=1),
                8,
                8,
                (0.0625, 0.0625, 0.125, 0.125),
            ),
        ],
        ids=["small", "offset", "one-pixel"],
    )
    def test_normalized_xywh_exact_values(
        self,
        box: BoundingBoxXYXY,
        width: int,
        height: int,
        expected: tuple[float, float, float, float],
    ) -> None:
        assert box.as_normalized_xywh(width, height) == expected

    @pytest.mark.parametrize(
        ("width", "height"),
        [(0, 10), (10, 0), (-5, 10), (10, -5)],
        ids=["zero-width", "zero-height", "negative-width", "negative-height"],
    )
    def test_normalized_xywh_rejects_non_positive_dimensions(
        self, width: int, height: int
    ) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(ValueError):
            box.as_normalized_xywh(width, height)

    @pytest.mark.parametrize(
        ("width", "height"),
        [(10.0, 10), (10, 10.0), ("10", 10), (10, "10")],
        ids=["float-width", "float-height", "str-width", "str-height"],
    )
    def test_normalized_xywh_rejects_non_integer_dimensions(
        self, width: object, height: object
    ) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(ValueError):
            box.as_normalized_xywh(width, height)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("width", "height"),
        [(True, 10), (10, True)],
        ids=["bool-width", "bool-height"],
    )
    def test_normalized_xywh_rejects_boolean_dimensions(
        self, width: object, height: object
    ) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(ValueError):
            box.as_normalized_xywh(width, height)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("box", "width", "height"),
        [
            (BoundingBoxXYXY(x_min=0, y_min=0, x_max=10, y_max=5), 5, 10),
            (BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=10), 10, 5),
        ],
        ids=["exceeds-width", "exceeds-height"],
    )
    def test_normalized_xywh_rejects_out_of_bounds_box(
        self, box: BoundingBoxXYXY, width: int, height: int
    ) -> None:
        with pytest.raises(ValueError):
            box.as_normalized_xywh(width, height)


class TestAnnotationRecord:
    def test_all_four_valid_source_code_pairs(self) -> None:
        for member in DamageClass:
            record = AnnotationRecord(
                category_id=member,
                source_code=member.name,
                bbox=BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5),
            )
            assert record.category_id is member
            assert record.source_code == member.name

    @pytest.mark.parametrize(
        "source_code",
        ["d00", " D00", "D00 ", "", "D30", "D10", "D20", "D40", "longitudinal_crack"],
        ids=[
            "lowercase",
            "padded-left",
            "padded-right",
            "blank",
            "unknown",
            "mismatch-d10",
            "mismatch-d20",
            "mismatch-d40",
            "canonical-name",
        ],
    )
    def test_invalid_source_codes_rejected(self, source_code: str) -> None:
        with pytest.raises(ValidationError):
            AnnotationRecord(
                category_id=DamageClass.D00,
                source_code=source_code,
                bbox=BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5),
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnnotationRecord(  # type: ignore[call-arg]
                category_id=DamageClass.D00,
                source_code="D00",
                bbox=BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5),
                extra=True,
            )

    def test_frozen_assignment_rejected(self) -> None:
        record = make_annotation()
        with pytest.raises(ValidationError):
            record.source_code = "D10"  # type: ignore[misc]


class TestPathsAndHashes:
    @pytest.mark.parametrize(
        "path",
        ["a.jpg", "a/b.jpg", "Japan/Japan_000001.jpg", "a/b/c/d.jpg", "caf\u00e9.jpg"],
        ids=["single", "nested", "domain", "deep", "nfc"],
    )
    def test_valid_paths(self, path: str) -> None:
        assert make_image(path=path).path == path

    @pytest.mark.parametrize(
        "path",
        [
            "/a/b.jpg",
            "C:/a/b.jpg",
            "c:/a/b.jpg",
            "//server/share/a.jpg",
            "a\\b.jpg",
            "C:\\a\\b.jpg",
            "./a.jpg",
            "a/./b.jpg",
            "a/../b.jpg",
            ".",
            "..",
            "a//b.jpg",
            "a/b/",
            "",
            "a\x00b.jpg",
            "cafe\u0301.jpg",
        ],
        ids=[
            "absolute",
            "drive",
            "drive-lower",
            "unc",
            "backslash",
            "windows",
            "leading-dot",
            "dot-segment",
            "dotdot-segment",
            "bare-dot",
            "bare-dotdot",
            "empty-segment",
            "trailing-slash",
            "empty",
            "nul",
            "non-nfc",
        ],
    )
    def test_invalid_paths_rejected(self, path: str) -> None:
        with pytest.raises(ValidationError):
            make_image(path=path)

    def test_valid_sha256_hashes(self) -> None:
        image = make_image(sha256=VALID_SHA256_B)
        assert image.sha256 == VALID_SHA256_B

    @pytest.mark.parametrize(
        "sha256",
        [
            "A" * 64,
            "a" * 63,
            "a" * 65,
            "g" * 64,
            "",
        ],
        ids=["uppercase", "short", "long", "non-hex", "empty"],
    )
    def test_invalid_sha256_rejected(self, sha256: str) -> None:
        with pytest.raises(ValidationError):
            make_image(sha256=sha256)

    def test_valid_dhash64(self) -> None:
        image = make_image(dhash64="fedcba9876543210")
        assert image.dhash64 == "fedcba9876543210"

    @pytest.mark.parametrize(
        "dhash64",
        [
            "A" * 16,
            "a" * 15,
            "a" * 17,
            "g" * 16,
            "",
        ],
        ids=["uppercase", "short", "long", "non-hex", "empty"],
    )
    def test_invalid_dhash64_rejected(self, dhash64: str) -> None:
        with pytest.raises(ValidationError):
            make_image(dhash64=dhash64)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("width", 0),
            ("height", 0),
            ("width", -1),
            ("height", -1),
            ("size_bytes", 0),
            ("size_bytes", -1),
        ],
        ids=[
            "width-zero",
            "height-zero",
            "width-negative",
            "height-negative",
            "size-zero",
            "size-negative",
        ],
    )
    def test_non_positive_dimensions_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            make_image(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("width", 10.0), ("height", 10.0), ("size_bytes", 10.0)],
        ids=["width-float", "height-float", "size-float"],
    )
    def test_float_dimensions_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            make_image(**{field: value})  # type: ignore[arg-type]

    def test_non_jpeg_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_image(format="PNG")  # type: ignore[call-arg]

    def test_annotation_source_valid(self) -> None:
        source = make_annotation_source()
        assert source.path == "Japan/Japan_000001.xml"

    def test_annotation_source_invalid_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_annotation_source(path="C:/x.xml")

    def test_annotation_source_invalid_hashes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_annotation_source(semantic_sha256="A" * 64)
        with pytest.raises(ValidationError):
            make_annotation_source(sha256="a" * 63)


class TestSamples:
    def test_positive_sample(self) -> None:
        sample = make_sample()
        assert sample.sample_id == "rdd2022/Japan/Japan_000001"
        assert sample.split is DatasetSplit.TRAIN
        assert len(sample.annotations) == 2
        assert sample.is_negative is False

    def test_negative_sample(self) -> None:
        sample = make_sample(annotations=())
        assert sample.is_negative is True

    def test_domain_split_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_sample(split=DatasetSplit.VALIDATION)

    @pytest.mark.parametrize(
        "sample_id",
        [
            "rdd2022/Japan",
            "rdd2022/Japan/other",
            "Japan/Japan_000001",
            "rdd2022/Japan/Japan_000001/extra",
            "rdd2022/India/Japan_000001",
            "other/Japan/Japan_000001",
        ],
        ids=[
            "missing-stem",
            "wrong-stem",
            "missing-prefix",
            "extra-segment",
            "wrong-domain",
            "wrong-prefix",
        ],
    )
    def test_malformed_sample_ids_rejected(self, sample_id: str) -> None:
        with pytest.raises(ValidationError):
            make_sample(sample_id=sample_id)

    def test_out_of_bounds_annotation_rejected(self) -> None:
        out_of_bounds = make_annotation(x_min=0, y_min=0, x_max=700, y_max=10)
        with pytest.raises(ValidationError):
            make_sample(annotations=(out_of_bounds,))

    def test_unsorted_annotations_rejected(self) -> None:
        d40 = make_annotation(
            category=DamageClass.D40, x_min=20, y_min=20, x_max=30, y_max=30
        )
        d00 = make_annotation()
        with pytest.raises(ValidationError):
            make_sample(annotations=(d40, d00))

    def test_duplicate_annotations_rejected(self) -> None:
        d00 = make_annotation()
        with pytest.raises(ValidationError):
            make_sample(annotations=(d00, d00))

    def test_same_bbox_different_class_rejected(self) -> None:
        d00 = make_annotation()
        d40 = make_annotation(category=DamageClass.D40)
        with pytest.raises(ValidationError):
            make_sample(annotations=(d00, d40))

    def test_same_class_different_bbox_accepted(self) -> None:
        first = make_annotation(x_min=0, y_min=0, x_max=10, y_max=10)
        second = make_annotation(x_min=20, y_min=20, x_max=30, y_max=30)
        sample = make_sample(annotations=(first, second))
        assert len(sample.annotations) == 2

    def test_image_path_domain_mismatch_rejected(self) -> None:
        image = make_image(path="India/Japan_000001.jpg")
        with pytest.raises(ValidationError):
            make_sample(image=image)

    def test_annotation_path_domain_mismatch_rejected(self) -> None:
        annotation_source = make_annotation_source(path="Czech/Japan_000001.xml")
        with pytest.raises(ValidationError):
            make_sample(annotation_source=annotation_source)

    def test_image_xml_stem_mismatch_rejected(self) -> None:
        annotation_source = make_annotation_source(path="Japan/Other.xml")
        with pytest.raises(ValidationError):
            make_sample(annotation_source=annotation_source)

    def test_wrong_image_extension_rejected(self) -> None:
        image = make_image(path="Japan/Japan_000001.png")
        with pytest.raises(ValidationError):
            make_sample(image=image)

    def test_wrong_xml_extension_rejected(self) -> None:
        annotation_source = make_annotation_source(path="Japan/Japan_000001.txt")
        with pytest.raises(ValidationError):
            make_sample(annotation_source=annotation_source)

    def test_uppercase_extensions_accepted(self) -> None:
        image = make_image(path="Japan/Japan_000001.JPG")
        annotation_source = make_annotation_source(path="Japan/Japan_000001.XML")
        sample = make_sample(image=image, annotation_source=annotation_source)
        assert sample.sample_id == "rdd2022/Japan/Japan_000001"

    def test_domain_plus_nested_paths_accepted(self) -> None:
        image = make_image(path="Japan/train/images/Japan_000001.jpg")
        annotation_source = make_annotation_source(
            path="Japan/train/annotations/Japan_000001.xml"
        )
        sample = make_sample(image=image, annotation_source=annotation_source)
        assert sample.sample_id == "rdd2022/Japan/Japan_000001"

    def test_is_negative_false_with_empty_annotations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_sample(annotations=(), is_negative=False)

    def test_is_negative_true_with_annotations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_sample(is_negative=True)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SampleRecord(  # type: ignore[call-arg]
                sample_id="rdd2022/Japan/x",
                domain=DatasetDomain.JAPAN,
                split=DatasetSplit.TRAIN,
                image=make_image(),
                annotation_source=make_annotation_source(),
                annotations=(),
                is_negative=True,
                extra=True,
            )

    def test_nested_immutability(self) -> None:
        sample = make_sample()
        with pytest.raises(ValidationError):
            sample.annotations[0].bbox.x_min = 9  # type: ignore[misc]
        with pytest.raises(ValidationError):
            sample.image.path = "other.jpg"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            sample.annotations = ()  # type: ignore[misc]
        with pytest.raises(TypeError):
            sample.annotations[0] = make_annotation()  # type: ignore[index]

    def test_model_json_round_trip(self) -> None:
        sample = make_sample()
        dumped = sample.model_dump_json()
        assert SampleRecord.model_validate_json(dumped) == sample


class TestIssuesAndErrors:
    ERROR_CODES: ClassVar[frozenset[DatasetIssueCode]] = frozenset(
        {
            DatasetIssueCode.MISSING_REQUIRED_DOMAIN,
            DatasetIssueCode.UNSAFE_SOURCE_PATH,
            DatasetIssueCode.PATH_CASE_COLLISION,
            DatasetIssueCode.MISSING_ANNOTATION,
            DatasetIssueCode.ORPHAN_ANNOTATION,
            DatasetIssueCode.MALFORMED_XML,
            DatasetIssueCode.UNSAFE_XML,
            DatasetIssueCode.XML_TOO_LARGE,
            DatasetIssueCode.ANNOTATION_FILENAME_MISMATCH,
            DatasetIssueCode.ANNOTATION_IMAGE_SIZE_MISMATCH,
            DatasetIssueCode.INVALID_CLASS_LABEL,
            DatasetIssueCode.NON_INTEGER_BBOX,
            DatasetIssueCode.INVALID_BBOX_ORDER,
            DatasetIssueCode.BBOX_OUT_OF_BOUNDS,
            DatasetIssueCode.OBJECT_LABEL_CONFLICT,
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.IMAGE_TOO_LARGE,
            DatasetIssueCode.UNSUPPORTED_IMAGE_FORMAT,
            DatasetIssueCode.UNSUPPORTED_EXIF_ORIENTATION,
            DatasetIssueCode.EXACT_DUPLICATE_CROSS_SPLIT,
            DatasetIssueCode.PIXEL_DUPLICATE_CROSS_SPLIT,
            DatasetIssueCode.DUPLICATE_LABEL_CONFLICT,
            DatasetIssueCode.PUBLISHED_INVENTORY_MISMATCH,
            DatasetIssueCode.SOURCE_FILE_ADDED,
            DatasetIssueCode.SOURCE_FILE_REMOVED,
            DatasetIssueCode.SOURCE_FILE_CHANGED,
        }
    )
    WARNING_CODES: ClassVar[frozenset[DatasetIssueCode]] = frozenset(
        {
            DatasetIssueCode.EMPTY_ANNOTATION,
            DatasetIssueCode.EXACT_OBJECT_DUPLICATE,
            DatasetIssueCode.EXACT_DUPLICATE_WITHIN_SPLIT,
            DatasetIssueCode.PIXEL_DUPLICATE_WITHIN_SPLIT,
            DatasetIssueCode.NEAR_DUPLICATE_WITHIN_SPLIT,
        }
    )
    REVIEW_CODES: ClassVar[frozenset[DatasetIssueCode]] = frozenset(
        {DatasetIssueCode.NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE}
    )
    INFO_CODES: ClassVar[frozenset[DatasetIssueCode]] = frozenset(
        {DatasetIssueCode.EXCLUDED_PUBLIC_TEST_IMAGE}
    )

    ISSUE_SEVERITY_POLICY: ClassVar[dict[DatasetIssueCode, IssueSeverity]] = {
        **dict.fromkeys(ERROR_CODES, IssueSeverity.ERROR),
        **dict.fromkeys(WARNING_CODES, IssueSeverity.WARNING),
        **dict.fromkeys(REVIEW_CODES, IssueSeverity.REVIEW_REQUIRED),
        **dict.fromkeys(INFO_CODES, IssueSeverity.INFO),
    }

    def test_policy_covers_every_code(self) -> None:
        assert set(self.ISSUE_SEVERITY_POLICY) == set(DatasetIssueCode)

    def test_every_issue_code_accepts_exact_severity(self) -> None:
        for code, severity in self.ISSUE_SEVERITY_POLICY.items():
            issue = DatasetIssue(code=code, severity=severity)
            assert issue.severity is severity

    @pytest.mark.parametrize(
        ("code", "severity"),
        [
            (DatasetIssueCode.CORRUPT_IMAGE, IssueSeverity.INFO),
            (DatasetIssueCode.EMPTY_ANNOTATION, IssueSeverity.ERROR),
            (
                DatasetIssueCode.NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE,
                IssueSeverity.WARNING,
            ),
            (DatasetIssueCode.EXCLUDED_PUBLIC_TEST_IMAGE, IssueSeverity.ERROR),
            (DatasetIssueCode.MALFORMED_XML, IssueSeverity.REVIEW_REQUIRED),
        ],
        ids=[
            "error-code-info",
            "warning-code-error",
            "review-code-warning",
            "info-code-error",
            "error-code-review",
        ],
    )
    def test_wrong_severity_rejected(
        self, code: DatasetIssueCode, severity: IssueSeverity
    ) -> None:
        with pytest.raises(ValidationError):
            DatasetIssue(code=code, severity=severity)

    def test_issue_paths_sorted_and_unique(self) -> None:
        with pytest.raises(ValidationError):
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_ANNOTATION,
                severity=IssueSeverity.WARNING,
                paths=("b.jpg", "a.jpg"),
            )
        with pytest.raises(ValidationError):
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_ANNOTATION,
                severity=IssueSeverity.WARNING,
                paths=("a.jpg", "a.jpg"),
            )
        issue = DatasetIssue(
            code=DatasetIssueCode.EMPTY_ANNOTATION,
            severity=IssueSeverity.WARNING,
            paths=("a.jpg", "b.jpg"),
        )
        assert issue.paths == ("a.jpg", "b.jpg")

    def test_issue_paths_must_be_canonical(self) -> None:
        with pytest.raises(ValidationError):
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_ANNOTATION,
                severity=IssueSeverity.WARNING,
                paths=("C:/absolute.jpg",),
            )

    def test_issue_details_sorted_and_unique(self) -> None:
        with pytest.raises(ValidationError):
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_ANNOTATION,
                severity=IssueSeverity.WARNING,
                details=(IssueDetail(key="b", value=1), IssueDetail(key="a", value=2)),
            )
        with pytest.raises(ValidationError):
            DatasetIssue(
                code=DatasetIssueCode.EMPTY_ANNOTATION,
                severity=IssueSeverity.WARNING,
                details=(IssueDetail(key="a", value=1), IssueDetail(key="a", value=2)),
            )

    def test_issue_detail_key_must_be_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="", value=1)

    @pytest.mark.parametrize(
        "value",
        ["text", 1, 2.5, True, False, None],
        ids=["str", "int", "float", "true", "false", "none"],
    )
    def test_issue_detail_scalar_values_accepted(
        self, value: str | int | float | bool | None
    ) -> None:
        detail = IssueDetail(key="k", value=value)
        assert detail.value == value

    @pytest.mark.parametrize(
        "value",
        [{"a": 1}, [1, 2], (1, 2)],
        ids=["dict", "list", "tuple"],
    )
    def test_issue_detail_container_values_rejected(self, value: object) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="k", value=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "neg-inf"],
    )
    def test_issue_detail_non_finite_floats_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="k", value=value)

    @pytest.mark.parametrize(
        "value",
        [
            "C:/Users/example/raw",
            "C:\\Users\\example\\raw",
            "/etc/passwd",
            "\\server\\share",
            "copy from C:/Users/example",
            "failed at /etc/passwd",
            "failed at \\\\server\\share",
            "failed at //server/share",
        ],
        ids=[
            "posix-drive",
            "windows-drive",
            "leading-slash",
            "leading-backslash",
            "embedded-drive",
            "embedded-posix",
            "embedded-unc",
            "embedded-posix-unc",
        ],
    )
    def test_issue_detail_absolute_path_strings_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="k", value=value)

    def test_issue_detail_slash_values_accepted(self) -> None:
        for value in ("ratio 1/2", "2024/05/12", "https://example.com/x", "n/a"):
            detail = IssueDetail(key="k", value=value)
            assert detail.value == value

    @pytest.mark.parametrize(
        "value",
        ["line\x00break", "line\nbreak", "tab\there"],
        ids=["nul", "newline", "tab"],
    )
    def test_issue_detail_control_character_strings_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="k", value=value)

    @pytest.mark.parametrize(
        "value",
        ["\x85", "\x9b", "\u2028", "\u2029", "line\u2028break", "line\u2029break"],
        ids=[
            "u0085-nel",
            "u009b-csi",
            "u2028-ls",
            "u2029-ps",
            "embedded-ls",
            "embedded-ps",
        ],
    )
    def test_issue_detail_unicode_separator_strings_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="k", value=value)

    def test_issue_detail_non_ascii_safe_values_accepted(self) -> None:
        for value in ("caf\u00e9", "m\u00fcller road", "\u65e5\u672c"):
            detail = IssueDetail(key="k", value=value)
            assert detail.value == value

    def test_issue_detail_key_control_characters_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueDetail(key="bad\x00key", value=1)

    def test_issue_optional_fields(self) -> None:
        issue = DatasetIssue(
            code=DatasetIssueCode.CORRUPT_IMAGE,
            severity=IssueSeverity.ERROR,
            domain=DatasetDomain.JAPAN,
            sample_id="rdd2022/Japan/x",
            paths=("a.jpg",),
            details=(IssueDetail(key="k", value=1),),
        )
        assert issue.domain is DatasetDomain.JAPAN
        assert issue.sample_id == "rdd2022/Japan/x"

    def test_contract_error_attributes(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND,
            message="missing root",
            context=(IssueDetail(key="source_root", value="raw-root"),),
        )
        assert error.code is DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND
        assert error.message == "missing root"
        assert error.context == (IssueDetail(key="source_root", value="raw-root"),)

    def test_contract_error_is_exception(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        assert isinstance(error, Exception)

    def test_contract_error_string_contains_only_code_and_message(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND,
            message="missing root",
            context=(IssueDetail(key="source_root", value="secret-value"),),
        )
        assert str(error) == "DATASET_ROOT_NOT_FOUND: missing root"
        assert "secret" not in str(error)
        assert "source_root" not in str(error)

    def test_contract_error_immutable(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR,
            message="boom",
            context=(IssueDetail(key="a", value=1),),
        )
        with pytest.raises(AttributeError):
            error.code = DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND  # type: ignore[misc]
        with pytest.raises(AttributeError):
            error.message = "mutated"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            error._context = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            error.context = ()  # type: ignore[misc]
        assert error.code is DatasetContractErrorCode.INTERNAL_ERROR
        assert error.message == "boom"
        assert error.context == (IssueDetail(key="a", value=1),)

    def test_contract_error_context_sorted_and_unique(self) -> None:
        with pytest.raises(ValueError):
            DatasetContractError(
                code=DatasetContractErrorCode.INTERNAL_ERROR,
                message="boom",
                context=(IssueDetail(key="b", value=1), IssueDetail(key="a", value=2)),
            )
        with pytest.raises(ValueError):
            DatasetContractError(
                code=DatasetContractErrorCode.INTERNAL_ERROR,
                message="boom",
                context=(IssueDetail(key="a", value=1), IssueDetail(key="a", value=2)),
            )

    def test_all_contract_error_codes_construct(self) -> None:
        for code in DatasetContractErrorCode:
            error = DatasetContractError(code=code, message="boom")
            assert error.code is code

    @pytest.mark.parametrize(
        "message",
        [
            "boom at /etc/passwd",
            "boom at C:/Users/x",
            "boom at \\\\server\\share",
            "boom\nsecond line",
            "boom\u2028second line",
        ],
        ids=["posix", "drive", "unc", "newline", "line-separator"],
    )
    def test_contract_error_message_rejects_unsafe_content(self, message: str) -> None:
        with pytest.raises(ValueError):
            DatasetContractError(
                code=DatasetContractErrorCode.INTERNAL_ERROR, message=message
            )

    def test_contract_error_message_accepts_safe_content(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR,
            message="audit failed for ratio 1/2",
        )
        assert error.message == "audit failed for ratio 1/2"

    def test_contract_error_add_note(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        error.add_note("additional context")
        assert error.__notes__ == ["additional context"]

    @pytest.mark.parametrize(
        "note",
        [
            "note\nsecond line",
            "note at /etc/passwd",
            "note at C:/Users/x",
            "note at \\\\server\\share",
            "note\u2028second line",
            "note\u2029second line",
            "\x85note",
        ],
        ids=[
            "newline",
            "posix",
            "drive",
            "unc",
            "line-separator",
            "paragraph-separator",
            "c1-control",
        ],
    )
    def test_contract_error_add_note_rejects_unsafe(self, note: str) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        with pytest.raises(ValueError):
            error.add_note(note)
        assert not hasattr(error, "__notes__")

    def test_contract_error_pickle_round_trip(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND,
            message="missing root",
            context=(IssueDetail(key="source_root", value="raw-root"),),
        )
        restored = pickle.loads(pickle.dumps(error))
        assert isinstance(restored, DatasetContractError)
        assert restored.code is DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND
        assert restored.message == "missing root"
        assert restored.context == (IssueDetail(key="source_root", value="raw-root"),)
        assert str(restored) == str(error)
        with pytest.raises(AttributeError):
            restored.code = DatasetContractErrorCode.INTERNAL_ERROR  # type: ignore[misc]

    def test_contract_error_pickle_preserves_notes(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        error.add_note("additional context")
        restored = pickle.loads(pickle.dumps(error))
        assert restored.__notes__ == ["additional context"]
        assert str(restored) == "INTERNAL_ERROR: boom"

    def test_contract_error_setstate_rejects_contract_fields(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR,
            message="boom",
            context=(IssueDetail(key="a", value=1),),
        )
        with pytest.raises(ValueError):
            error.__setstate__(
                {"code": DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND}
            )
        with pytest.raises(ValueError):
            error.__setstate__({"code": "not-a-valid-code"})
        with pytest.raises(ValueError):
            error.__setstate__({"message": "boom at /etc/passwd"})
        with pytest.raises(ValueError):
            error.__setstate__({"_context": ("not", "a", "context")})
        assert error.code is DatasetContractErrorCode.INTERNAL_ERROR
        assert error.message == "boom"
        assert error.context == (IssueDetail(key="a", value=1),)

    def test_contract_error_setstate_allows_metadata(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        error.__setstate__({"__notes__": ["note"]})
        assert error.__notes__ == ["note"]

    def test_contract_error_setstate_none(self) -> None:
        error = DatasetContractError(
            code=DatasetContractErrorCode.INTERNAL_ERROR, message="boom"
        )
        error.__setstate__(None)
        assert error.code is DatasetContractErrorCode.INTERNAL_ERROR
        assert error.message == "boom"


class TestModelCopy:
    def test_bbox_copy_rejects_update(self) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        with pytest.raises(TypeError):
            box.model_copy(update={"x_min": -1})
        with pytest.raises(TypeError):
            box.model_copy(update={"x_max": 2})

    def test_issue_detail_copy_rejects_update(self) -> None:
        detail = IssueDetail(key="k", value=1)
        with pytest.raises(TypeError):
            detail.model_copy(update={"value": float("nan")})
        with pytest.raises(TypeError):
            detail.model_copy(update={"key": "other"})

    def test_sample_copy_rejects_update(self) -> None:
        sample = make_sample()
        with pytest.raises(TypeError):
            sample.model_copy(update={"is_negative": True})

    def test_copy_without_update_is_equal(self) -> None:
        box = BoundingBoxXYXY(x_min=0, y_min=0, x_max=5, y_max=5)
        assert box.model_copy() == box
        sample = make_sample()
        assert sample.model_copy() == sample
        assert sample.model_copy(deep=True) == sample


class TestPackageBoundary:
    def test_all_approved_symbols_importable(self) -> None:
        from roadmind.data import (
            RDD2022_CLASS_NAMES,
            RDD2022_DATASET_ID,
            RDD2022_DOMAIN_SPLITS,
            RDD2022_MANIFEST_SCHEMA_VERSION,
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
            IssueSeverity,
            SampleRecord,
        )

        symbols = (
            RDD2022_DATASET_ID,
            RDD2022_MANIFEST_SCHEMA_VERSION,
            RDD2022_CLASS_NAMES,
            RDD2022_DOMAIN_SPLITS,
            DamageClass,
            DatasetDomain,
            DatasetSplit,
            IssueSeverity,
            DatasetIssueCode,
            DatasetContractErrorCode,
            BoundingBoxXYXY,
            AnnotationRecord,
            ImageAsset,
            AnnotationSource,
            SampleRecord,
            DatasetIssue,
            DatasetContractError,
        )
        assert len(symbols) == 17
        assert symbols[0] == "rdd2022-roadmind-four-class-v1"

    def test_public_export_list_is_exact(self) -> None:
        expected = {
            "RDD2022_DATASET_ID",
            "RDD2022_MANIFEST_SCHEMA_VERSION",
            "RDD2022_CLASS_NAMES",
            "RDD2022_DOMAIN_SPLITS",
            "DamageClass",
            "DatasetDomain",
            "DatasetSplit",
            "IssueSeverity",
            "DatasetIssueCode",
            "DatasetContractErrorCode",
            "BoundingBoxXYXY",
            "AnnotationRecord",
            "ImageAsset",
            "AnnotationSource",
            "SampleRecord",
            "DatasetIssue",
            "DatasetContractError",
        }
        assert set(data_pkg.__all__) == expected
        assert len(data_pkg.__all__) == 17

    def test_no_later_stage_symbols(self) -> None:
        for name in (
            "prepare_rdd2022",
            "load_dataset",
            "discover_rdd2022",
            "scan_rdd2022",
            "detect_duplicates",
        ):
            assert not hasattr(data_pkg, name)

    def test_import_is_side_effect_free(self) -> None:
        script = (
            "import sys\n"
            "import roadmind\n"
            "banned = [\n"
            "    'PIL', 'defusedxml', 'torch', 'ultralytics',\n"
            "    'fastapi', 'pandas', 'numpy',\n"
            "]\n"
            "loaded = [m for m in banned if m in sys.modules]\n"
            "if loaded:\n"
            "    print('banned modules loaded:', loaded)\n"
            "    sys.exit(3)\n"
            "import roadmind.data\n"
            "loaded = [m for m in banned if m in sys.modules]\n"
            "if loaded:\n"
            "    print('banned modules loaded after data import:', loaded)\n"
            "    sys.exit(4)\n"
        )
        env: dict[str, str] = {}
        for passthrough in ("SYSTEMROOT", "SystemRoot", "PATH", "TEMP", "TMP"):
            value = os.environ.get(passthrough)
            if value is not None:
                env[passthrough] = value
        temp_dir = tempfile.mkdtemp(prefix="roadmind-import-")
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert list(Path(temp_dir).iterdir()) == []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
