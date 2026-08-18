"""Typed data contracts for the RDD2022 dataset foundation (ROAD-001A).

This module defines the immutable, strict Pydantic models and the fixed
taxonomy, split, issue, and error vocabularies shared by every later
ROAD-001 ticket and by ROAD-002. It performs no filesystem access.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections.abc import Callable, Mapping
from enum import IntEnum, StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

RDD2022_DATASET_ID: Final[str] = "rdd2022-roadmind-four-class-v1"
RDD2022_MANIFEST_SCHEMA_VERSION: Final[str] = "1.0.0"


class DamageClass(IntEnum):
    """The fixed four-class RDD2022 damage taxonomy."""

    D00 = 0
    D10 = 1
    D20 = 2
    D40 = 3


RDD2022_CLASS_NAMES: Final[Mapping[DamageClass, str]] = MappingProxyType(
    {
        DamageClass.D00: "longitudinal_crack",
        DamageClass.D10: "transverse_crack",
        DamageClass.D20: "alligator_crack",
        DamageClass.D40: "pothole",
    }
)


class DatasetDomain(StrEnum):
    """The seven annotated RDD2022 domain subsets."""

    JAPAN = "Japan"
    INDIA = "India"
    CZECH = "Czech"
    NORWAY = "Norway"
    UNITED_STATES = "United_States"
    CHINA_MOTORBIKE = "China_MotorBike"
    CHINA_DRONE = "China_Drone"


class DatasetSplit(StrEnum):
    """The fixed dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"


RDD2022_DOMAIN_SPLITS: Final[Mapping[DatasetDomain, DatasetSplit]] = MappingProxyType(
    {
        DatasetDomain.JAPAN: DatasetSplit.TRAIN,
        DatasetDomain.INDIA: DatasetSplit.TRAIN,
        DatasetDomain.NORWAY: DatasetSplit.TRAIN,
        DatasetDomain.UNITED_STATES: DatasetSplit.TRAIN,
        DatasetDomain.CZECH: DatasetSplit.VALIDATION,
        DatasetDomain.CHINA_MOTORBIKE: DatasetSplit.TEST,
        DatasetDomain.CHINA_DRONE: DatasetSplit.OOD,
    }
)


class IssueSeverity(StrEnum):
    """Issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    REVIEW_REQUIRED = "review_required"


class DatasetIssueCode(StrEnum):
    """The complete dataset issue-code vocabulary for ROAD-001B-D."""

    # Blocking / source errors
    MISSING_REQUIRED_DOMAIN = "MISSING_REQUIRED_DOMAIN"
    UNSAFE_SOURCE_PATH = "UNSAFE_SOURCE_PATH"
    PATH_CASE_COLLISION = "PATH_CASE_COLLISION"
    MISSING_ANNOTATION = "MISSING_ANNOTATION"
    ORPHAN_ANNOTATION = "ORPHAN_ANNOTATION"
    MALFORMED_XML = "MALFORMED_XML"
    UNSAFE_XML = "UNSAFE_XML"
    XML_TOO_LARGE = "XML_TOO_LARGE"
    ANNOTATION_FILENAME_MISMATCH = "ANNOTATION_FILENAME_MISMATCH"
    ANNOTATION_IMAGE_SIZE_MISMATCH = "ANNOTATION_IMAGE_SIZE_MISMATCH"
    INVALID_CLASS_LABEL = "INVALID_CLASS_LABEL"
    NON_INTEGER_BBOX = "NON_INTEGER_BBOX"
    INVALID_BBOX_ORDER = "INVALID_BBOX_ORDER"
    BBOX_OUT_OF_BOUNDS = "BBOX_OUT_OF_BOUNDS"
    OBJECT_LABEL_CONFLICT = "OBJECT_LABEL_CONFLICT"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_IMAGE_FORMAT = "UNSUPPORTED_IMAGE_FORMAT"
    UNSUPPORTED_EXIF_ORIENTATION = "UNSUPPORTED_EXIF_ORIENTATION"
    EXACT_DUPLICATE_CROSS_SPLIT = "EXACT_DUPLICATE_CROSS_SPLIT"
    PIXEL_DUPLICATE_CROSS_SPLIT = "PIXEL_DUPLICATE_CROSS_SPLIT"
    DUPLICATE_LABEL_CONFLICT = "DUPLICATE_LABEL_CONFLICT"
    PUBLISHED_INVENTORY_MISMATCH = "PUBLISHED_INVENTORY_MISMATCH"
    SOURCE_FILE_ADDED = "SOURCE_FILE_ADDED"
    SOURCE_FILE_REMOVED = "SOURCE_FILE_REMOVED"
    SOURCE_FILE_CHANGED = "SOURCE_FILE_CHANGED"

    # Non-blocking / review findings
    EMPTY_ANNOTATION = "EMPTY_ANNOTATION"
    EXACT_OBJECT_DUPLICATE = "EXACT_OBJECT_DUPLICATE"
    EXACT_DUPLICATE_WITHIN_SPLIT = "EXACT_DUPLICATE_WITHIN_SPLIT"
    PIXEL_DUPLICATE_WITHIN_SPLIT = "PIXEL_DUPLICATE_WITHIN_SPLIT"
    NEAR_DUPLICATE_WITHIN_SPLIT = "NEAR_DUPLICATE_WITHIN_SPLIT"
    NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE = "NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE"
    EXCLUDED_PUBLIC_TEST_IMAGE = "EXCLUDED_PUBLIC_TEST_IMAGE"


class DatasetContractErrorCode(StrEnum):
    """Operational error codes mapped to exit status 2 by ROAD-001E."""

    DATASET_ROOT_NOT_FOUND = "DATASET_ROOT_NOT_FOUND"
    DATASET_ROOT_NOT_DIRECTORY = "DATASET_ROOT_NOT_DIRECTORY"
    DATASET_ROOT_UNREADABLE = "DATASET_ROOT_UNREADABLE"
    DATASET_ROOT_AMBIGUOUS = "DATASET_ROOT_AMBIGUOUS"
    UNSUPPORTED_RDD2022_LAYOUT = "UNSUPPORTED_RDD2022_LAYOUT"
    OUTPUT_INSIDE_RAW_ROOT = "OUTPUT_INSIDE_RAW_ROOT"
    OUTPUT_ALREADY_EXISTS = "OUTPUT_ALREADY_EXISTS"
    OUTPUT_PARENT_UNWRITABLE = "OUTPUT_PARENT_UNWRITABLE"
    OUTPUT_COMMIT_FAILED = "OUTPUT_COMMIT_FAILED"
    MANIFEST_SCHEMA_INVALID = "MANIFEST_SCHEMA_INVALID"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_ISSUE_SEVERITY: Final[Mapping[DatasetIssueCode, IssueSeverity]] = MappingProxyType(
    {
        DatasetIssueCode.MISSING_REQUIRED_DOMAIN: IssueSeverity.ERROR,
        DatasetIssueCode.UNSAFE_SOURCE_PATH: IssueSeverity.ERROR,
        DatasetIssueCode.PATH_CASE_COLLISION: IssueSeverity.ERROR,
        DatasetIssueCode.MISSING_ANNOTATION: IssueSeverity.ERROR,
        DatasetIssueCode.ORPHAN_ANNOTATION: IssueSeverity.ERROR,
        DatasetIssueCode.MALFORMED_XML: IssueSeverity.ERROR,
        DatasetIssueCode.UNSAFE_XML: IssueSeverity.ERROR,
        DatasetIssueCode.XML_TOO_LARGE: IssueSeverity.ERROR,
        DatasetIssueCode.ANNOTATION_FILENAME_MISMATCH: IssueSeverity.ERROR,
        DatasetIssueCode.ANNOTATION_IMAGE_SIZE_MISMATCH: IssueSeverity.ERROR,
        DatasetIssueCode.INVALID_CLASS_LABEL: IssueSeverity.ERROR,
        DatasetIssueCode.NON_INTEGER_BBOX: IssueSeverity.ERROR,
        DatasetIssueCode.INVALID_BBOX_ORDER: IssueSeverity.ERROR,
        DatasetIssueCode.BBOX_OUT_OF_BOUNDS: IssueSeverity.ERROR,
        DatasetIssueCode.OBJECT_LABEL_CONFLICT: IssueSeverity.ERROR,
        DatasetIssueCode.CORRUPT_IMAGE: IssueSeverity.ERROR,
        DatasetIssueCode.IMAGE_TOO_LARGE: IssueSeverity.ERROR,
        DatasetIssueCode.UNSUPPORTED_IMAGE_FORMAT: IssueSeverity.ERROR,
        DatasetIssueCode.UNSUPPORTED_EXIF_ORIENTATION: IssueSeverity.ERROR,
        DatasetIssueCode.EXACT_DUPLICATE_CROSS_SPLIT: IssueSeverity.ERROR,
        DatasetIssueCode.PIXEL_DUPLICATE_CROSS_SPLIT: IssueSeverity.ERROR,
        DatasetIssueCode.DUPLICATE_LABEL_CONFLICT: IssueSeverity.ERROR,
        DatasetIssueCode.PUBLISHED_INVENTORY_MISMATCH: IssueSeverity.ERROR,
        DatasetIssueCode.SOURCE_FILE_ADDED: IssueSeverity.ERROR,
        DatasetIssueCode.SOURCE_FILE_REMOVED: IssueSeverity.ERROR,
        DatasetIssueCode.SOURCE_FILE_CHANGED: IssueSeverity.ERROR,
        DatasetIssueCode.EMPTY_ANNOTATION: IssueSeverity.WARNING,
        DatasetIssueCode.EXACT_OBJECT_DUPLICATE: IssueSeverity.WARNING,
        DatasetIssueCode.EXACT_DUPLICATE_WITHIN_SPLIT: IssueSeverity.WARNING,
        DatasetIssueCode.PIXEL_DUPLICATE_WITHIN_SPLIT: IssueSeverity.WARNING,
        DatasetIssueCode.NEAR_DUPLICATE_WITHIN_SPLIT: IssueSeverity.WARNING,
        DatasetIssueCode.NEAR_DUPLICATE_CROSS_SPLIT_CANDIDATE: (
            IssueSeverity.REVIEW_REQUIRED
        ),
        DatasetIssueCode.EXCLUDED_PUBLIC_TEST_IMAGE: IssueSeverity.INFO,
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DHASH64_RE = re.compile(r"^[0-9a-f]{16}$")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_EMBEDDED_DRIVE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
_EMBEDDED_POSIX_ABSOLUTE_RE = re.compile(r"(?<![\w:/])/(?=[^\s/])")
_EMBEDDED_POSIX_UNC_RE = re.compile(r"(?<![\w:])//")
_UNSAFE_CHAR_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Zl", "Zp"})


def _validate_relative_path(value: str) -> str:
    """Validate a raw-root-relative POSIX path in Unicode NFC form."""
    if "\x00" in value:
        raise ValueError("path must not contain NUL")
    if "\\" in value:
        raise ValueError("path must use POSIX separators")
    if value.startswith("/"):
        raise ValueError("path must be relative")
    if _DRIVE_PREFIX_RE.match(value):
        raise ValueError("path must not have a drive prefix")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("path must be Unicode NFC")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("path must not contain empty, '.', or '..' segments")
    return value


def _validate_detail_string(value: str) -> str:
    """Validate a safe detail/message string.

    Rejects Unicode control characters (categories ``Cc``: C0 and C1,
    including ``U+0085`` and ``U+009B``), line/paragraph separators
    (categories ``Zl``/``Zp``: ``U+2028``/``U+2029``) for log-injection
    safety, any backslash (Windows or UNC paths), drive-prefixed paths,
    embedded POSIX absolute paths such as ``"failed at /etc/passwd"``, and
    POSIX-style UNC paths such as ``"failed at //server/share"``.
    """
    if any(unicodedata.category(char) in _UNSAFE_CHAR_CATEGORIES for char in value):
        raise ValueError(
            "strings must not contain Unicode control or line-separator characters"
        )
    if "\\" in value:
        raise ValueError("strings must not contain backslashes")
    if _EMBEDDED_DRIVE_RE.search(value):
        raise ValueError("strings must not contain absolute paths")
    if _EMBEDDED_POSIX_ABSOLUTE_RE.search(value):
        raise ValueError("strings must not contain absolute paths")
    if _EMBEDDED_POSIX_UNC_RE.search(value):
        raise ValueError("strings must not contain UNC paths")
    return value


class RoadMindModel(BaseModel):
    """Base for all contract models: strict, frozen, extra-forbid."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, object] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy a model; reject ``update`` because it bypasses validation.

        Pydantic's ``model_copy(update=...)`` constructs a new instance
        without running validators, which would allow invalid contracts
        (for example negative bbox coordinates or non-finite issue details).
        Callers must construct a new validated instance instead.
        """
        if update is not None:
            raise TypeError(
                "model_copy(update=...) bypasses validation; "
                "construct a new validated instance instead"
            )
        return super().model_copy(deep=deep)


class BoundingBoxXYXY(RoadMindModel):
    """Zero-based, half-open axis-aligned box in integer pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @model_validator(mode="after")
    def _check_geometry(self) -> BoundingBoxXYXY:
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("bbox coordinates must be non-negative")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bbox must have positive width and height")
        return self

    def as_normalized_xywh(
        self,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        """Return ``(center_x, center_y, width, height)`` normalized to [0, 1].

        Performs no clamping, rounding, or coordinate repair.
        """
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("image dimensions must be integers, not booleans")
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("image dimensions must be strict integers")
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if (
            self.x_min < 0
            or self.y_min < 0
            or self.x_max > width
            or self.y_max > height
        ):
            raise ValueError("bbox does not fit image dimensions")
        center_x = (self.x_min + self.x_max) / (2 * width)
        center_y = (self.y_min + self.y_max) / (2 * height)
        box_width = (self.x_max - self.x_min) / width
        box_height = (self.y_max - self.y_min) / height
        return (center_x, center_y, box_width, box_height)


class AnnotationRecord(RoadMindModel):
    """A single object annotation."""

    category_id: DamageClass
    source_code: str
    bbox: BoundingBoxXYXY

    @model_validator(mode="after")
    def _check_source_code(self) -> AnnotationRecord:
        if self.source_code != self.category_id.name:
            raise ValueError("source_code must exactly match the category enum name")
        return self


class ImageAsset(RoadMindModel):
    """Decoded and hashed properties of one raw image file."""

    path: str
    sha256: str
    pixel_sha256: str
    dhash64: str
    width: int
    height: int
    size_bytes: int
    format: Literal["JPEG"]

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("sha256", "pixel_sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("dhash64")
    @classmethod
    def _check_dhash64(cls, value: str) -> str:
        if not _DHASH64_RE.fullmatch(value):
            raise ValueError("dhash64 must be 16 lowercase hex characters")
        return value

    @field_validator("width", "height", "size_bytes")
    @classmethod
    def _check_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value


class AnnotationSource(RoadMindModel):
    """Identity and hashes of one raw VOC annotation file."""

    path: str
    sha256: str
    semantic_sha256: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("sha256", "semantic_sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value


class SampleRecord(RoadMindModel):
    """One fully validated annotated sample."""

    sample_id: str
    domain: DatasetDomain
    split: DatasetSplit
    image: ImageAsset
    annotation_source: AnnotationSource
    annotations: tuple[AnnotationRecord, ...] = ()
    is_negative: bool

    @model_validator(mode="after")
    def _check_invariants(self) -> SampleRecord:
        expected_split = RDD2022_DOMAIN_SPLITS[self.domain]
        if self.split != expected_split:
            raise ValueError("split must match the fixed domain-to-split mapping")
        if self.is_negative != (not self.annotations):
            raise ValueError(
                "is_negative must be true exactly when annotations is empty"
            )
        stem = PurePosixPath(self.image.path).stem
        expected_id = f"rdd2022/{self.domain.value}/{stem}"
        if self.sample_id != expected_id:
            raise ValueError("sample_id must be rdd2022/<domain>/<image-stem>")
        domain_prefix = f"{self.domain.value}/"
        if not self.image.path.startswith(domain_prefix):
            raise ValueError(
                "image path must be rooted under the sample domain directory"
            )
        if not self.annotation_source.path.startswith(domain_prefix):
            raise ValueError(
                "annotation path must be rooted under the sample domain directory"
            )
        annotation_stem = PurePosixPath(self.annotation_source.path).stem
        if stem != annotation_stem:
            raise ValueError("image and annotation must share the same filename stem")
        if not self.image.path.lower().endswith(".jpg"):
            raise ValueError("image path must end with .jpg or .JPG")
        if not self.annotation_source.path.lower().endswith(".xml"):
            raise ValueError("annotation path must end with .xml or .XML")
        for annotation in self.annotations:
            bbox = annotation.bbox
            if bbox.x_max > self.image.width or bbox.y_max > self.image.height:
                raise ValueError("annotation bbox exceeds image bounds")
        keys = [
            (
                int(annotation.category_id),
                annotation.bbox.x_min,
                annotation.bbox.y_min,
                annotation.bbox.x_max,
                annotation.bbox.y_max,
            )
            for annotation in self.annotations
        ]
        boxes = [
            (
                annotation.bbox.x_min,
                annotation.bbox.y_min,
                annotation.bbox.x_max,
                annotation.bbox.y_max,
            )
            for annotation in self.annotations
        ]
        if len(set(boxes)) != len(boxes):
            raise ValueError("duplicate annotation bbox regardless of class")
        for previous, current in itertools.pairwise(keys):
            if current < previous:
                raise ValueError(
                    "annotations must be ordered by "
                    "(category_id, x_min, y_min, x_max, y_max)"
                )
        return self


class IssueDetail(RoadMindModel):
    """A single key/value fact attached to a :class:`DatasetIssue`."""

    key: str
    value: str | int | float | bool | None

    @field_validator("key")
    @classmethod
    def _key_must_be_safe(cls, value: str) -> str:
        if not value:
            raise ValueError("issue detail key must be non-empty")
        return _validate_detail_string(value)

    @field_validator("value")
    @classmethod
    def _value_must_be_safe(
        cls, value: str | int | float | bool | None
    ) -> str | int | float | bool | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("issue detail float values must be finite")
        if isinstance(value, str):
            _validate_detail_string(value)
        return value


class DatasetIssue(RoadMindModel):
    """An aggregated source-data finding with fixed severity policy."""

    code: DatasetIssueCode
    severity: IssueSeverity
    domain: DatasetDomain | None = None
    sample_id: str | None = None
    paths: tuple[str, ...] = ()
    details: tuple[IssueDetail, ...] = ()

    @model_validator(mode="after")
    def _check_severity_policy(self) -> DatasetIssue:
        expected = _ISSUE_SEVERITY[self.code]
        if self.severity != expected:
            raise ValueError(
                f"severity {self.severity.value} is not allowed "
                f"for code {self.code.value}"
            )
        return self

    @field_validator("paths")
    @classmethod
    def _check_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_relative_path(path)
        if list(value) != sorted(value):
            raise ValueError("paths must be sorted")
        if len(set(value)) != len(value):
            raise ValueError("paths must be unique")
        return value

    @field_validator("details")
    @classmethod
    def _check_details(cls, value: tuple[IssueDetail, ...]) -> tuple[IssueDetail, ...]:
        keys = [detail.key for detail in value]
        if keys != sorted(keys):
            raise ValueError("detail keys must be sorted")
        if len(set(keys)) != len(keys):
            raise ValueError("detail keys must be unique")
        return value


_IMMUTABLE_ERROR_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"code", "message", "_context"}
)


def _rebuild_contract_error(
    code: DatasetContractErrorCode,
    message: str,
    context: tuple[IssueDetail, ...],
) -> DatasetContractError:
    """Pickle-rebuild helper that reconstructs through the validated constructor."""
    return DatasetContractError(code=code, message=message, context=context)


class DatasetContractError(Exception):
    """Operational error carrying a typed code and safe message.

    The contract fields ``code``, ``message``, and ``context`` are immutable
    after construction: they are only ever set through the validated
    constructor, and both attribute assignment and ``__setstate__`` reject
    them. Standard :class:`BaseException` behavior (for example ``add_note``
    and pickle round-trips) remains available. Dataset defects must be
    reported as :class:`DatasetIssue` objects, never raised as this
    exception.
    """

    code: DatasetContractErrorCode
    message: str
    _context: tuple[IssueDetail, ...]

    def __init__(
        self,
        code: DatasetContractErrorCode,
        message: str,
        *,
        context: tuple[IssueDetail, ...] = (),
    ) -> None:
        # BaseException.__reduce__ reconstructs from self.args, where the code
        # is stored as its string value; the constructor resolves either form.
        resolved_code = DatasetContractErrorCode(code)
        _validate_detail_string(message)
        keys = [detail.key for detail in context]
        if keys != sorted(keys):
            raise ValueError("context keys must be sorted")
        if len(set(keys)) != len(keys):
            raise ValueError("context keys must be unique")
        object.__setattr__(self, "code", resolved_code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "_context", context)
        super().__init__(resolved_code.value, message)

    def __setattr__(self, name: str, value: object) -> None:
        if name in _IMMUTABLE_ERROR_ATTRIBUTES:
            raise AttributeError(
                f"DatasetContractError attribute {name!r} is immutable"
            )
        object.__setattr__(self, name, value)

    def add_note(self, note: str) -> None:
        """Add a note after validating it as a safe string."""
        _validate_detail_string(note)
        super().add_note(note)

    def __reduce__(
        self,
    ) -> tuple[
        Callable[
            [DatasetContractErrorCode, str, tuple[IssueDetail, ...]],
            DatasetContractError,
        ],
        tuple[DatasetContractErrorCode, str, tuple[IssueDetail, ...]],
        dict[str, object],
    ]:
        # Reconstruct through the validated constructor; the state holds only
        # standard exception metadata (for example __notes__).
        metadata = dict(self.__dict__)
        for name in _IMMUTABLE_ERROR_ATTRIBUTES:
            metadata.pop(name, None)
        return (
            _rebuild_contract_error,
            (self.code, self.message, self._context),
            metadata,
        )

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        # Restore only vetted BaseException metadata. The immutable contract
        # fields are never part of pickle state and must not be injectable
        # through this public hook.
        if state is None:
            return
        if _IMMUTABLE_ERROR_ATTRIBUTES.intersection(state):
            raise ValueError(
                "state must not contain immutable contract fields "
                "(code, message, context)"
            )
        self.__dict__.update(state)

    @property
    def context(self) -> tuple[IssueDetail, ...]:
        return self._context

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"
