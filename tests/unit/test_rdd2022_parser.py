"""Unit tests for ROAD-001B: RDD2022 discovery, image validation, VOC parsing.

All tests use synthetic fixtures from ``tests.support.rdd2022_factory``; no
real RDD2022 data is required.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image
from support.rdd2022_factory import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    build_domain,
    build_rdd2022,
    jpeg_bytes,
    jpeg_gradient_bytes,
    make_directory_link,
    write_bytes,
    write_jpeg,
    write_voc,
)

from roadmind.data import (
    DatasetContractError,
    DatasetContractErrorCode,
    DatasetDomain,
    DatasetIssue,
    DatasetIssueCode,
    DatasetSplit,
)
from roadmind.data.rdd2022 import (
    RDD2022ScanResult,
    _find_case_collisions,
    discover_rdd2022,
    scan_rdd2022,
)

GOLDEN_SOLID_RAW = "f3e0133145392733a1ba8411b1fbad0739960eea5cac1ed6d64fe191ca6a0baf"
GOLDEN_SOLID_PIXEL = "5466675c8f9c951c5ea26ca1062343e5ec7545b86628a1c216ff75dd5354cc69"
GOLDEN_PATTERN_DHASH = "03060c3060030618"
EMPTY_SEMANTIC_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


def pattern_jpeg_bytes() -> bytes:
    image = Image.new("L", (8, 8))
    for y in range(8):
        for x in range(8):
            image.putpixel((x, y), (x * 37 + y * 53) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def scan_tree(root: Path) -> RDD2022ScanResult:
    return scan_rdd2022(discover_rdd2022(root))


def tree_fingerprint(root: Path) -> tuple[tuple[str, int, str], ...]:
    entries: list[tuple[str, int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            entries.append((rel, len(data), hashlib.sha256(data).hexdigest()))
    return tuple(sorted(entries))


class TestDiscoverRdd2022:
    def test_direct_root(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data")
        layout = discover_rdd2022(root)
        assert layout.source_root == root.resolve()
        assert len(layout.domain_layouts) == 7

    def test_single_wrapper_root(self, scratch_dir: Path) -> None:
        wrapper = scratch_dir / "data" / "RDD2022"
        build_rdd2022(wrapper)
        layout = discover_rdd2022(scratch_dir / "data")
        assert layout.source_root == wrapper.resolve()

    def test_missing_root_raises(self, scratch_dir: Path) -> None:
        with pytest.raises(DatasetContractError) as excinfo:
            discover_rdd2022(scratch_dir / "nope")
        assert excinfo.value.code is DatasetContractErrorCode.DATASET_ROOT_NOT_FOUND

    def test_root_is_file_raises(self, scratch_dir: Path) -> None:
        path = write_bytes(scratch_dir, "file.txt", b"x")
        with pytest.raises(DatasetContractError) as excinfo:
            discover_rdd2022(path)
        assert excinfo.value.code is DatasetContractErrorCode.DATASET_ROOT_NOT_DIRECTORY

    def test_unreadable_root_raises(
        self, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = build_rdd2022(scratch_dir / "data")

        def raise_permission(*args: object, **kwargs: object) -> object:
            raise PermissionError("denied")

        monkeypatch.setattr(os, "scandir", raise_permission)
        with pytest.raises(DatasetContractError) as excinfo:
            discover_rdd2022(root)
        assert excinfo.value.code is DatasetContractErrorCode.DATASET_ROOT_UNREADABLE

    def test_empty_directory_raises(self, scratch_dir: Path) -> None:
        root = scratch_dir / "data"
        root.mkdir()
        with pytest.raises(DatasetContractError) as excinfo:
            discover_rdd2022(root)
        assert excinfo.value.code is DatasetContractErrorCode.UNSUPPORTED_RDD2022_LAYOUT

    def test_ambiguous_wrappers_raise(self, scratch_dir: Path) -> None:
        for name in ("one", "two"):
            build_rdd2022(scratch_dir / "data" / name)
        with pytest.raises(DatasetContractError) as excinfo:
            discover_rdd2022(scratch_dir / "data")
        assert excinfo.value.code is DatasetContractErrorCode.DATASET_ROOT_AMBIGUOUS

    def test_domain_split_mapping(self, scratch_dir: Path) -> None:
        layout = discover_rdd2022(build_rdd2022(scratch_dir / "data"))
        expected = {
            DatasetDomain.JAPAN: DatasetSplit.TRAIN,
            DatasetDomain.INDIA: DatasetSplit.TRAIN,
            DatasetDomain.NORWAY: DatasetSplit.TRAIN,
            DatasetDomain.UNITED_STATES: DatasetSplit.TRAIN,
            DatasetDomain.CZECH: DatasetSplit.VALIDATION,
            DatasetDomain.CHINA_MOTORBIKE: DatasetSplit.TEST,
            DatasetDomain.CHINA_DRONE: DatasetSplit.OOD,
        }
        assert {d.domain: d.split for d in layout.domain_layouts} == expected
        assert [d.domain for d in layout.domain_layouts] == list(DatasetDomain)


class TestScanHappyPath:
    def test_seven_domains_samples_and_splits(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=2)
        result = scan_tree(root)
        assert len(result.samples) == 14
        expected_splits = {
            DatasetDomain.JAPAN: DatasetSplit.TRAIN,
            DatasetDomain.INDIA: DatasetSplit.TRAIN,
            DatasetDomain.NORWAY: DatasetSplit.TRAIN,
            DatasetDomain.UNITED_STATES: DatasetSplit.TRAIN,
            DatasetDomain.CZECH: DatasetSplit.VALIDATION,
            DatasetDomain.CHINA_MOTORBIKE: DatasetSplit.TEST,
            DatasetDomain.CHINA_DRONE: DatasetSplit.OOD,
        }
        for sample in result.samples:
            assert sample.split is expected_splits[sample.domain]
        assert result.issues == ()
        assert len(result.source_files) == 14 * 2

    def test_sample_fields(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        (sample,) = scan_tree(root).samples
        assert sample.sample_id == "rdd2022/Japan/Japan_00001"
        assert sample.image.path == "Japan/train/images/Japan_00001.jpg"
        assert sample.image.format == "JPEG"
        assert sample.image.size_bytes == sample.image.size_bytes
        assert len(sample.image.sha256) == 64
        assert len(sample.image.pixel_sha256) == 64
        assert len(sample.image.dhash64) == 16
        assert (
            sample.annotation_source.path == "Japan/train/annotations/Japan_00001.xml"
        )
        assert sample.is_negative is False
        assert len(sample.annotations) == 1

    def test_scan_is_deterministic(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=2)
        first = scan_tree(root)
        second = scan_tree(root)
        assert first == second

    def test_filesystem_order_independence(self, scratch_dir: Path) -> None:
        normal = build_rdd2022(scratch_dir / "normal", image_count=2)
        reversed_root = scratch_dir / "reversed"
        for domain in reversed(list(DatasetDomain)):
            for index in reversed(range(1, 3)):
                stem = f"{domain.value}_{index:05d}"
                write_voc(reversed_root / domain.value / "train" / "annotations", stem)
                write_jpeg(reversed_root / domain.value / "train" / "images", stem)
        assert scan_tree(normal) == scan_tree(reversed_root)

    def test_result_ordering(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=2)
        result = scan_tree(root)
        assert result.samples == tuple(
            sorted(result.samples, key=lambda s: s.sample_id)
        )
        assert result.source_files == tuple(
            sorted(result.source_files, key=lambda r: r.path)
        )
        assert result.excluded_public_test == ()

    def test_no_absolute_paths_in_results(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=1)
        result = scan_tree(root)
        for sample in result.samples:
            assert not Path(sample.image.path).is_absolute()
            assert not Path(sample.annotation_source.path).is_absolute()
            assert "\\" not in sample.image.path
            assert "\\" not in sample.annotation_source.path
        for record in result.source_files:
            assert not Path(record.path).is_absolute()
            assert "\\" not in record.path


class TestLabelsAndBoxes:
    def test_all_four_labels(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(
                ("D00", (1, 1, 10, 10)),
                ("D10", (12, 12, 20, 20)),
                ("D20", (22, 22, 30, 30)),
                ("D40", (32, 32, 40, 40)),
            ),
        )
        (sample,) = scan_tree(root).samples
        assert [a.category_id.name for a in sample.annotations] == [
            "D00",
            "D10",
            "D20",
            "D40",
        ]
        assert sample.annotations[0].source_code == "D00"

    @pytest.mark.parametrize(
        "label", ["d00", " D00", "D00 ", "D30", "longitudinal_crack", ""]
    )
    def test_invalid_label_rejected(self, scratch_dir: Path, label: str) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=((label, (1, 1, 10, 10)),),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.INVALID_CLASS_LABEL]

    def test_non_integer_box(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(
            annotations,
            "Japan_00001",
            raw=(
                "<annotation><filename>Japan_00001.jpg</filename>"
                "<size><width>64</width><height>48</height></size>"
                "<object><name>D00</name><bndbox>"
                "<xmin>1.5</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
                "</bndbox></object></annotation>"
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.NON_INTEGER_BBOX]

    @pytest.mark.parametrize(
        ("box", "expected_code"),
        [
            ((-2, 1, 10, 10), DatasetIssueCode.BBOX_OUT_OF_BOUNDS),
            ((0, 1, 10, 10), DatasetIssueCode.BBOX_OUT_OF_BOUNDS),
            ((1, 0, 10, 10), DatasetIssueCode.BBOX_OUT_OF_BOUNDS),
            ((20, 1, 5, 10), DatasetIssueCode.INVALID_BBOX_ORDER),
            ((1, 20, 10, 5), DatasetIssueCode.INVALID_BBOX_ORDER),
            ((1, 1, 100, 10), DatasetIssueCode.BBOX_OUT_OF_BOUNDS),
            ((1, 1, 10, 100), DatasetIssueCode.BBOX_OUT_OF_BOUNDS),
        ],
        ids=[
            "negative-x",
            "zero-x",
            "zero-y",
            "reversed-x",
            "reversed-y",
            "beyond-width",
            "beyond-height",
        ],
    )
    def test_invalid_boxes(
        self,
        scratch_dir: Path,
        box: tuple[int, int, int, int],
        expected_code: DatasetIssueCode,
    ) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(("D00", box),),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [expected_code]

    def test_one_pixel_box(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(("D00", (5, 5, 5, 5)),),
        )
        (sample,) = scan_tree(root).samples
        bbox = sample.annotations[0].bbox
        assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == (4, 4, 5, 5)

    def test_full_image_box(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(("D00", (1, 1, DEFAULT_WIDTH, DEFAULT_HEIGHT)),),
        )
        (sample,) = scan_tree(root).samples
        bbox = sample.annotations[0].bbox
        assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == (
            0,
            0,
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
        )

    def test_duplicate_object_collapsed(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(
                ("D00", (1, 1, 10, 10)),
                ("D00", (1, 1, 10, 10)),
            ),
        )
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert len(result.samples[0].annotations) == 1
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.EXACT_OBJECT_DUPLICATE
        ]

    def test_label_conflict_same_box(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.JAPAN,
            image_count=1,
            object_spec=(
                ("D00", (1, 1, 10, 10)),
                ("D40", (1, 1, 10, 10)),
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.OBJECT_LABEL_CONFLICT
        ]


class TestPairing:
    def test_missing_annotation(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        write_jpeg(images, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.MISSING_ANNOTATION]

    def test_orphan_annotation(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        annotations = root / "Japan" / "train" / "annotations"
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.ORPHAN_ANNOTATION]

    def test_negative_sample(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1, object_spec=None)
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert result.samples[0].is_negative is True
        assert [i.code for i in result.issues] == [DatasetIssueCode.EMPTY_ANNOTATION]
        assert (
            result.samples[0].annotation_source.semantic_sha256 == EMPTY_SEMANTIC_SHA256
        )

    def test_filename_mismatch(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(annotations, "Japan_00001", filename="Other.jpg")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ANNOTATION_FILENAME_MISMATCH
        ]

    def test_image_size_mismatch(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(annotations, "Japan_00001", width=32, height=24)
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ANNOTATION_IMAGE_SIZE_MISMATCH
        ]

    def test_public_test_excluded(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.CHINA_MOTORBIKE,
            image_count=1,
            public_test_count=2,
        )
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert result.excluded_public_test == (
            "China_MotorBike/test/images/China_MotorBike_test_00001.jpg",
            "China_MotorBike/test/images/China_MotorBike_test_00002.jpg",
        )
        assert all(
            i.code is DatasetIssueCode.EXCLUDED_PUBLIC_TEST_IMAGE for i in result.issues
        )
        for sample in result.samples:
            assert not sample.image.path.startswith("China_MotorBike/test/")

    def test_no_public_test_inference(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        write_jpeg(images, "Japan_00001")
        result = scan_tree(root)
        assert [i.code for i in result.issues] == [DatasetIssueCode.MISSING_ANNOTATION]

    def test_public_test_unsafe_link_excluded(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(
            root,
            DatasetDomain.CHINA_MOTORBIKE,
            image_count=1,
            public_test_count=1,
        )
        outside = scratch_dir / "outside"
        write_jpeg(outside, "evil")
        make_directory_link(
            root / "China_MotorBike" / "test" / "images" / "escape", outside
        )
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert len(result.excluded_public_test) == 1
        assert any(i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH for i in result.issues)


class TestXmlSafety:
    def test_wrong_root_tag(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_bytes(
            annotations,
            "Japan_00001.xml",
            b"<notannotation><filename>Japan_00001.jpg</filename>"
            b"<size><width>64</width><height>48</height></size></notannotation>",
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.MALFORMED_XML]

    def test_missing_filename_element(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(
            annotations,
            "Japan_00001",
            raw=(
                "<annotation><size><width>64</width><height>48</height></size>"
                "<object><name>D00</name><bndbox>"
                "<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
                "</bndbox></object></annotation>"
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ANNOTATION_FILENAME_MISMATCH
        ]

    def test_missing_size_element(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(
            annotations,
            "Japan_00001",
            raw=(
                "<annotation><filename>Japan_00001.jpg</filename>"
                "<size><width>64</width></size>"
                "<object><name>D00</name><bndbox>"
                "<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
                "</bndbox></object></annotation>"
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ANNOTATION_IMAGE_SIZE_MISMATCH
        ]

    def test_missing_bndbox(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(
            annotations,
            "Japan_00001",
            raw=(
                "<annotation><filename>Japan_00001.jpg</filename>"
                "<size><width>64</width><height>48</height></size>"
                "<object><name>D00</name></object></annotation>"
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.NON_INTEGER_BBOX]

    def test_control_char_label_unsafe_detail(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(
            annotations,
            "Japan_00001",
            raw=(
                "<annotation><filename>Japan_00001.jpg</filename>"
                "<size><width>64</width><height>48</height></size>"
                "<object><name>D00\x85</name><bndbox>"
                "<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
                "</bndbox></object></annotation>"
            ),
        )
        result = scan_tree(root)
        assert result.samples == ()
        (issue,) = result.issues
        assert issue.code is DatasetIssueCode.INVALID_CLASS_LABEL
        assert issue.details[0].value == "<unsafe>"

    def test_malformed_xml(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_bytes(annotations, "Japan_00001.xml", b"<annotation><broken")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.MALFORMED_XML]

    def test_dtd_xml_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_bytes(
            annotations,
            "Japan_00001.xml",
            b'<?xml version="1.0"?><!DOCTYPE annotation [<!ENTITY x "boom">]>'
            b"<annotation><filename>Japan_00001.jpg</filename>"
            b"<size><width>64</width><height>48</height></size>"
            b"<object><name>&x;</name><bndbox>"
            b"<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
            b"</bndbox></object></annotation>",
        )
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.UNSAFE_XML]

    def test_oversized_xml(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "Japan_00001")
        write_voc(annotations, "Japan_00001")
        result = scan_rdd2022(discover_rdd2022(root), max_xml_bytes=100)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.XML_TOO_LARGE]


class TestImageValidation:
    def test_corrupt_image(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", b"this is not a jpeg at all")
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        ]

    def test_truncated_image(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(
            images,
            "Japan_00001.jpg",
            jpeg_bytes(64, 48)[: len(jpeg_bytes(64, 48)) // 2],
        )
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        ]

    def test_zero_byte_image(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", b"")
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        ]

    def test_spoofed_image(self, scratch_dir: Path) -> None:
        png = io.BytesIO()
        Image.new("RGB", (8, 8)).save(png, format="PNG")
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", png.getvalue())
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ORPHAN_ANNOTATION,
            DatasetIssueCode.UNSUPPORTED_IMAGE_FORMAT,
        ]

    @pytest.mark.parametrize("name", ["unexpected.png", "photo.jpeg", "readme.txt"])
    def test_unsupported_image_file_inventoried_and_flagged(
        self, scratch_dir: Path, name: str
    ) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, name, jpeg_bytes())
        write_voc(annotations, Path(name).stem)
        result = scan_tree(root)
        assert result.samples == ()
        assert any(
            record.path == f"Japan/train/images/{name}"
            for record in result.source_files
        )
        assert any(
            i.code is DatasetIssueCode.UNSUPPORTED_IMAGE_FORMAT for i in result.issues
        )
        assert any(i.code is DatasetIssueCode.ORPHAN_ANNOTATION for i in result.issues)

    @pytest.mark.parametrize("name", ["notes.txt", "README"])
    def test_annotation_dir_extra_files_inventoried(
        self, scratch_dir: Path, name: str
    ) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(annotations, name, b"not an annotation")
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert any(
            record.path == f"Japan/train/annotations/{name}"
            for record in result.source_files
        )
        assert not any(
            i.code is DatasetIssueCode.ORPHAN_ANNOTATION for i in result.issues
        )

    def test_oversized_image(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        result = scan_rdd2022(discover_rdd2022(root), max_image_pixels=100)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.IMAGE_TOO_LARGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        ]

    @pytest.mark.parametrize("mode", ["L", "CMYK"])
    def test_grayscale_and_cmyk_valid(self, scratch_dir: Path, mode: str) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", jpeg_bytes(mode=mode))
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert result.issues == ()

    def test_exif_orientation_unsupported(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", jpeg_bytes(exif_orientation=3))
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.ORPHAN_ANNOTATION,
            DatasetIssueCode.UNSUPPORTED_EXIF_ORIENTATION,
        ]

    def test_exif_orientation_1_accepted(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", jpeg_bytes(exif_orientation=1))
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert len(result.samples) == 1


class TestHashes:
    def test_golden_raw_and_pixel_sha256(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        solid = jpeg_bytes(8, 8)
        write_bytes(images, "Japan_00001.jpg", solid)
        write_voc(
            annotations,
            "Japan_00001",
            width=8,
            height=8,
            objects=(("D00", (1, 1, 8, 8)),),
        )
        (sample,) = scan_tree(root).samples
        assert sample.image.sha256 == hashlib.sha256(solid).hexdigest()
        assert sample.image.sha256 == GOLDEN_SOLID_RAW
        assert sample.image.pixel_sha256 == GOLDEN_SOLID_PIXEL

    def test_golden_dhash(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", pattern_jpeg_bytes())
        write_voc(
            annotations,
            "Japan_00001",
            width=8,
            height=8,
            objects=(("D00", (1, 1, 8, 8)),),
        )
        (sample,) = scan_tree(root).samples
        assert sample.image.dhash64 == GOLDEN_PATTERN_DHASH

    def test_dhash_format(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", jpeg_gradient_bytes())
        write_voc(annotations, "Japan_00001")
        (sample,) = scan_tree(root).samples
        assert len(sample.image.dhash64) == 16
        assert all(c in "0123456789abcdef" for c in sample.image.dhash64)

    def test_pixel_hash_changes_with_content(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", jpeg_bytes(8, 8, color=(1, 2, 3)))
        write_bytes(images, "Japan_00002.jpg", jpeg_bytes(8, 8, color=(9, 9, 9)))
        write_voc(
            annotations,
            "Japan_00001",
            width=8,
            height=8,
            objects=(("D00", (1, 1, 8, 8)),),
        )
        write_voc(
            annotations,
            "Japan_00002",
            width=8,
            height=8,
            objects=(("D00", (1, 1, 8, 8)),),
        )
        samples = scan_tree(root).samples
        assert samples[0].image.pixel_sha256 != samples[1].image.pixel_sha256

    def test_semantic_sha256_deterministic_and_sensitive(
        self, scratch_dir: Path
    ) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        for domain, objects in (
            (DatasetDomain.JAPAN, (("D00", (1, 1, 10, 10)),)),
            (DatasetDomain.INDIA, (("D00", (1, 1, 10, 10)),)),
            (DatasetDomain.CZECH, (("D40", (1, 1, 10, 10)),)),
        ):
            build_domain(root, domain, image_count=1, object_spec=objects)
        by_domain = {s.domain: s for s in scan_tree(root).samples}
        assert by_domain[DatasetDomain.JAPAN].annotation_source.semantic_sha256 == (
            by_domain[DatasetDomain.INDIA].annotation_source.semantic_sha256
        )
        assert by_domain[DatasetDomain.JAPAN].annotation_source.semantic_sha256 != (
            by_domain[DatasetDomain.CZECH].annotation_source.semantic_sha256
        )


class TestPathSafety:
    def test_escaping_directory_link_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        outside = scratch_dir / "outside"
        write_jpeg(outside, "evil")
        make_directory_link(images / "escape", outside)
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.UNSAFE_SOURCE_PATH]
        assert result.source_files == ()

    @pytest.mark.skipif(
        sys.platform == "win32" and not os.environ.get("ROADMIND_TEST_SYMLINK"),
        reason="file symlinks require elevated privileges or developer mode on Windows",
    )
    def test_escaping_file_symlink_unsafe(
        self, scratch_dir: Path, file_symlink_supported: bool
    ) -> None:
        if not file_symlink_supported:
            pytest.skip("file symlinks not supported on this platform")
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        outside = scratch_dir / "outside"
        write_jpeg(outside, "evil")
        os.symlink(outside / "evil.jpg", images / "Japan_00001.jpg")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.UNSAFE_SOURCE_PATH]

    def test_non_nfc_filename_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "cafe\u0301.jpg", jpeg_bytes())
        write_voc(annotations, "cafe\u0301")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.UNSAFE_SOURCE_PATH,
            DatasetIssueCode.UNSAFE_SOURCE_PATH,
        ]

    def test_find_case_collisions(self) -> None:
        assert _find_case_collisions(()) == ()
        assert _find_case_collisions(("a/b.jpg",)) == ()
        assert _find_case_collisions(("a/b.jpg", "a/c.jpg")) == ()
        found = _find_case_collisions(("x/A.jpg", "x/a.jpg"))
        assert found == (("x/A.jpg", "x/a.jpg"),)
        found = _find_case_collisions(("x/A.jpg", "x/a.jpg", "x/b.jpg", "x/B.jpg"))
        assert found == (("x/A.jpg", "x/a.jpg"), ("x/B.jpg", "x/b.jpg"))

    @pytest.mark.skipif(sys.platform == "win32", reason="case-insensitive filesystem")
    def test_case_collision_files_excluded(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_jpeg(images, "A")
        write_jpeg(images, "a")
        write_voc(annotations, "A")
        result = scan_tree(root)
        assert result.samples == ()
        assert {i.code for i in result.issues} == {
            DatasetIssueCode.PATH_CASE_COLLISION,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        }


class TestLayoutJunctionSafety:
    def test_domain_level_junction_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        outside = scratch_dir / "outside"
        build_domain(outside, DatasetDomain.JAPAN, image_count=1)
        shutil.rmtree(root / "Japan")
        make_directory_link(root / "Japan", outside / "Japan")
        result = scan_tree(root)
        assert all(s.domain is not DatasetDomain.JAPAN for s in result.samples)
        assert any(
            i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )
        assert not any(
            i.code is DatasetIssueCode.MISSING_REQUIRED_DOMAIN for i in result.issues
        )

    def test_domain_level_junction_inside_root_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        shutil.rmtree(root / "Japan")
        make_directory_link(root / "Japan", root / "India")
        result = scan_tree(root)
        assert all(s.domain is not DatasetDomain.JAPAN for s in result.samples)
        assert any(
            i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )

    def test_train_level_junction_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        outside = scratch_dir / "outside"
        write_jpeg(outside / "images", "sneaky")
        write_voc(outside / "annotations", "sneaky")
        shutil.rmtree(root / "Japan" / "train")
        make_directory_link(root / "Japan" / "train", outside)
        result = scan_tree(root)
        assert all(s.domain is not DatasetDomain.JAPAN for s in result.samples)
        assert any(
            i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )
        assert not any(
            i.code is DatasetIssueCode.MISSING_REQUIRED_DOMAIN for i in result.issues
        )

    def test_images_level_junction_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        outside = scratch_dir / "outside"
        write_jpeg(outside, "sneaky")
        shutil.rmtree(root / "Japan" / "train" / "images")
        make_directory_link(root / "Japan" / "train" / "images", outside)
        result = scan_tree(root)
        assert all(s.domain is not DatasetDomain.JAPAN for s in result.samples)
        assert any(
            i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )
        assert any(
            i.code is DatasetIssueCode.ORPHAN_ANNOTATION
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )

    def test_test_dir_junction_unsafe(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        outside = scratch_dir / "outside"
        write_jpeg(outside / "images", "sneaky")
        shutil.rmtree(root / "Japan" / "test", ignore_errors=True)
        make_directory_link(root / "Japan" / "test", outside)
        result = scan_tree(root)
        assert len(result.samples) == 1
        assert result.excluded_public_test == ()
        assert any(
            i.code is DatasetIssueCode.UNSAFE_SOURCE_PATH
            and i.domain is DatasetDomain.JAPAN
            for i in result.issues
        )


class TestAggregation:
    def test_multiple_defects_aggregated(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", b"garbage")
        write_voc(annotations, "Japan_00001")
        write_jpeg(images, "Japan_00002")
        write_bytes(annotations, "Japan_00002.xml", b"<broken")
        write_jpeg(images, "Japan_00003")
        write_jpeg(images, "Japan_00004")
        write_voc(annotations, "Japan_00004")
        result = scan_tree(root)
        codes = [i.code for i in result.issues]
        assert DatasetIssueCode.CORRUPT_IMAGE in codes
        assert DatasetIssueCode.MALFORMED_XML in codes
        assert DatasetIssueCode.MISSING_ANNOTATION in codes
        assert len(result.samples) == 1
        assert result.samples[0].sample_id == "rdd2022/Japan/Japan_00004"

    def test_invalid_sample_never_returned(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", b"garbage")
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert {i.code for i in result.issues} == {
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        }

    def test_issues_sorted_deterministically(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        annotations = root / "Japan" / "train" / "annotations"
        write_bytes(images, "Japan_00001.jpg", b"garbage")
        write_voc(annotations, "Japan_00001")
        write_jpeg(images, "Japan_00002")
        write_voc(annotations, "Japan_00002", filename="Wrong.jpg")
        result = scan_tree(root)

        def key(issue: DatasetIssue) -> tuple[str, str, tuple[str, ...], str]:
            return (
                issue.code.value,
                issue.domain.value if issue.domain else "",
                issue.paths,
                issue.sample_id or "",
            )

        assert result.issues == tuple(sorted(result.issues, key=key))


class TestMissingDomains:
    def test_missing_domain_dir(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=1)
        import shutil

        shutil.rmtree(root / "Japan")
        result = scan_tree(root)
        assert len(result.samples) == 6
        assert all(s.domain is not DatasetDomain.JAPAN for s in result.samples)
        japan_issues = [
            i
            for i in result.issues
            if i.domain is DatasetDomain.JAPAN
            and i.code is DatasetIssueCode.MISSING_REQUIRED_DOMAIN
        ]
        assert len(japan_issues) == 1

    def test_missing_annotations_dir(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        images = root / "Japan" / "train" / "images"
        write_jpeg(images, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.MISSING_ANNOTATION]

    def test_missing_images_dir(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        annotations = root / "Japan" / "train" / "annotations"
        write_voc(annotations, "Japan_00001")
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [DatasetIssueCode.ORPHAN_ANNOTATION]

    def test_missing_both_dirs(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        import shutil

        shutil.rmtree(root / "Japan" / "train")
        (root / "Japan" / "train").mkdir(parents=True)
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.MISSING_REQUIRED_DOMAIN
        ]


class TestReadOnly:
    def test_tree_unchanged_by_scan(self, scratch_dir: Path) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=2)
        before = tree_fingerprint(root)
        scan_tree(root)
        scan_tree(root)
        assert tree_fingerprint(root) == before

    def test_unreadable_source_file_does_not_abort_scan(
        self, scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = build_rdd2022(scratch_dir / "data", image_count=0)
        build_domain(root, DatasetDomain.JAPAN, image_count=1)
        images = root / "Japan" / "train" / "images"
        target = images / "Japan_00001.jpg"
        real_read_bytes = Path.read_bytes

        def selective_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
            if os.path.normcase(str(self)) == os.path.normcase(str(target)):
                raise PermissionError("permission denied")
            return real_read_bytes(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_bytes", selective_read_bytes)
        result = scan_tree(root)
        assert result.samples == ()
        assert [i.code for i in result.issues] == [
            DatasetIssueCode.CORRUPT_IMAGE,
            DatasetIssueCode.ORPHAN_ANNOTATION,
        ]
        assert not any(
            record.path == "Japan/train/images/Japan_00001.jpg"
            for record in result.source_files
        )
