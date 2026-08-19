# ROAD-001B — RDD2022 Discovery, Image Validation and Pascal VOC Parsing (TDD evidence)

Status: implemented on `feat/road-001b-rdd2022-parser`, ready for review.

## Summary

Added the package-internal module `roadmind.data.rdd2022` with:

- `discover_rdd2022(source_root)` — resolves a direct dataset root or exactly
  one wrapper extraction directory and returns the seven `DomainLayout`
  entries (domain, fixed split, `train/images`, `train/annotations`, optional
  `test/images` public-test dir) in enum order.
- `scan_rdd2022(layout, *, max_xml_bytes, max_image_pixels)` — read-only scan
  producing `RDD2022ScanResult(samples, issues, source_files,
  excluded_public_test)`.

Deterministic output contract: samples sorted by `sample_id`, source files
and excluded public-test paths by normalized relative POSIX path, issues by
`(code, domain, paths, sample_id)`. No absolute paths, timestamps or machine
dependent metadata in results; nothing is written to the raw tree.

Fixed hash serializations (documented in the module docstring):

- raw SHA-256 over file bytes;
- pixel SHA-256 over `struct.pack(">II", width, height) + RGB pixel bytes`;
- dHash64 over a 9x8 LANCZOS grayscale resize (row-major left>right bits),
  lowercase hex;
- semantic annotation SHA-256 over the sorted canonical serialization of
  `(class name, x1, y1, x2, y2)` tuples (empty annotation hashes to
  `e3b0c4...`).

## Files changed (exactly the ticket scope)

Created:

- `src/roadmind/data/rdd2022.py`
- `tests/conftest.py`
- `tests/support/__init__.py`
- `tests/support/rdd2022_factory.py`
- `tests/unit/test_rdd2022_parser.py`
- `docs/testing/road-001b-rdd2022-parser.tdd.md` (this file)

Modified:

- `pyproject.toml` (runtime deps `Pillow`, `defusedxml`; mypy override for
  `defusedxml` which ships no type stubs)
- `uv.lock`

`src/roadmind/data/models.py` and `src/roadmind/data/__init__.py` were not
modified; ROAD-001A contracts are unchanged (all 200 model tests still pass).

## RED evidence

The test suite was written first (fixtures in `tests/support/rdd2022_factory.py`,
tests in `tests/unit/test_rdd2022_parser.py`) and run against the empty module:

```text
uv run --frozen pytest tests/unit/test_rdd2022_parser.py -q
E   ModuleNotFoundError: No module named 'roadmind.data.rdd2022'
1 error during collection
```

## GREEN + coverage evidence

```text
uv run --frozen pytest tests/unit/test_rdd2022_parser.py -q
# 74 passed, 2 skipped in 3.37s
# (skips are platform capability checks: file symlinks and
#  case-insensitive filesystem collision creation on Windows; both are
#  covered by the junction-escape test and the pure-function
#  _find_case_collisions test which run on every platform)

uv run --frozen pytest tests/unit/test_rdd2022_parser.py --cov=roadmind.data.rdd2022 --cov-branch --cov-report=term-missing --cov-fail-under=90 -q
# src\roadmind\data\rdd2022.py  388    23    118    10   93%
# Required test coverage of 90% reached. Total coverage: 93.19%
#
# Remaining uncovered lines are Windows-unreachable OSError branches
# (stat/read failures), POSIX-only symlink and case-collision filesystem
# paths, and the defensive INTERNAL_ERROR catch.

uv run --frozen pytest -m "not rdd2022" -q
# 274 passed, 2 skipped

uv run --frozen pytest -m "not rdd2022" --cov=roadmind.data --cov-branch -q
# TOTAL  730    23    220    10   97%   (global >= 80% required)

uv run --frozen ruff format --check .   # 11 files already formatted
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 4 source files

uv lock --check                         # Resolved 50 packages
uv run --frozen pip-audit               # No known vulnerabilities found
uv build --wheel --no-build-isolation   # wheel contains roadmind/data/rdd2022.py
git diff --check                        # clean
```

## Determinism evidence

- `test_scan_is_deterministic` — two scans of the same tree produce equal
  results.
- `test_filesystem_order_independence` — a tree built with domain dirs and
  file writes in reverse order produces byte-identical logical results.
- `test_result_ordering` — samples/source files/issues are sorted.
- `test_tree_unchanged_by_scan` — full tree fingerprint (rel path, size,
  SHA-256 of every file) is identical before and after repeated scans.
- Golden hash values hardcoded for a fixed 8x8 image and a fixed 8x8
  pseudo-random pattern (raw, pixel SHA-256, dHash64), plus the
  empty-annotation semantic SHA-256 constant.

## Review findings and fixes

| Finding | Fix |
|---|---|
| Empty `train/images` + `train/annotations` dirs looked identical to missing domain dirs, so empty domains were mis-flagged `MISSING_REQUIRED_DOMAIN` | Factory now always materializes the `train/images` + `train/annotations` structure; `MISSING_REQUIRED_DOMAIN` is only emitted when the domain dir or both required dirs are absent |
| `build_domain` passed `object_spec=None` (meaning "empty") by default, producing EMPTY_ANNOTATION on the happy path | Default `object_spec` is now one `D00` object; explicit `None` still means a negative sample |
| Unsafe file paths (non-NFC, backslash) cannot be stored in `DatasetIssue.paths`, which the model validates, so `UNSAFE_SOURCE_PATH` construction crashed | Unsafe-path issues carry `paths=()` and the sanitized offending path in an `IssueDetail` (`<unsafe>` fallback for control characters) |
| `<filename>`/`<size>` detail keys must be sorted per the ROAD-001A `IssueDetail` contract | Detail tuples are emitted in sorted key order (`actual` before `expected`) |
| Grayscale (`L`) factory images crashed `Image.new` with an RGB tuple | Factory uses a single-int fill for `L` mode |
| Truncated/corrupt/spoofed/EXIF-invalid images leave their XML without a valid image | Consistent semantics: the XML is reported `ORPHAN_ANNOTATION` alongside the image issue |

## Review round 2 (REQUEST CHANGES) and fixes

| Finding | Severity | Fix | Verification |
|---|---|---|---|
| A domain, `train`, `images`, or `test` directory that is a symlink/junction (even resolving inside the root) was accepted via `is_dir()`; a `Japan` junction to an outside dir produced a valid sample and no issue | P1 | `_build_domain_layout` now rejects layout paths that are links via `_is_link`/`_usable_layout_dir`; offending paths are recorded in `DomainLayout.unsafe_paths` and surfaced as `UNSAFE_SOURCE_PATH` issues; the linked directories are never walked or read | `test_domain_level_junction_unsafe`, `test_domain_level_junction_inside_root_unsafe`, `test_train_level_junction_unsafe`, `test_images_level_junction_unsafe`, `test_test_dir_junction_unsafe` (junction via `mklink /J`, no admin needed) |
| Non-`.jpg` files under `train/images` (`.png`, `.jpeg`, arbitrary) were dropped before inventory/validation, so ROAD-001C could not derive a complete inventory or detect format drift | P1 | All readable files under image/annotation/public-test dirs are inventoried into `source_files`; non-`.jpg` files under the images directory are flagged `UNSUPPORTED_IMAGE_FORMAT` (blocking) and excluded from pairing; non-`.xml` annotation files are inventoried without a pairing role | `test_unsupported_image_file_inventoried_and_flagged` (3 suffixes), `test_annotation_dir_extra_files_inventoried` |
| `_record_source` read `read_bytes()` unguarded before validation, so a permission change/concurrent removal aborted the whole scan | P2 | `_record_source` returns `None` on `OSError`; each per-file boundary emits the appropriate issue (`CORRUPT_IMAGE` for images/public test, `MALFORMED_XML` for annotations) and the scan continues | `test_unreadable_source_file_does_not_abort_scan` (selective `PermissionError` via monkeypatched `Path.read_bytes`) |
| Strict mypy failed: `images_dir`/`annotations_dir`/`public_test_dir` inferred as `Path` then assigned `None` in the junction branches | P1 | Candidate paths are typed `Path`; the assigned variables are declared `Path | None` up front | `mypy --strict` clean |
| New tests not formatter-compliant | P2 | `ruff format` applied | `ruff format --check .` clean |

## Re-run evidence after review round 2

```text
uv run --frozen pytest tests/unit/test_rdd2022_parser.py -q
# 85 passed, 2 skipped

uv run --frozen pytest tests/unit/test_rdd2022_parser.py --cov=roadmind.data.rdd2022 --cov-branch --cov-report=term-missing --cov-fail-under=90 -q
# src\roadmind\data\rdd2022.py  434    26    140    11   94%   (>= 90%)
# Remaining uncovered lines are Windows-unreachable OSError/symlink/case-collision
# filesystem branches and the defensive INTERNAL_ERROR catch.

uv run --frozen pytest -m "not rdd2022" -q
# 285 passed, 2 skipped

uv run --frozen ruff format --check .   # 12 files already formatted
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 4 source files

uv lock --check                         # Resolved 50 packages
uv run --frozen pip-audit               # No known vulnerabilities found
uv build --wheel --no-build-isolation   # wheel contains roadmind/data/rdd2022.py
git diff --check                        # clean
```

## Remaining notes

- Real RDD2022 has not been required or claimed; running against real data is
  ROAD-001E.
- Public-test images are excluded only when under the official `<domain>/test/images`
  layout; missing XML is never inferred as public test.
- Labels are matched exactly (`D00`/`D10`/`D20`/`D40`, case-sensitive, no
  trimming); VOC boxes are 1-based inclusive on input and converted to
  zero-based half-open for `SampleRecord`.