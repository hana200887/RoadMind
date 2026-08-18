# RoadMind

RoadMind is a deterministic, read-only preparation and audit toolchain for the
RDD2022 road-damage dataset. It converts an extracted RDD2022 release into
byte-reproducible dataset artifacts (manifest, samples, audit, checksums) that
downstream consumers load without touching raw files or re-parsing XML.

## Status

ROAD-001 (dataset foundation) is **not yet complete**.

- ROAD-001A — project scaffold and typed data contracts: **current scope**
- ROAD-001B — discovery, image validation, Pascal VOC parsing: planned
- ROAD-001C — audit and deterministic manifest generation: planned
- ROAD-001D — duplicate and leakage detection: planned
- ROAD-001E — CLI, integration, and real-dataset acceptance: planned

ROAD-001A is foundation-only: immutable typed contracts (taxonomy, domain
split, bounding boxes, samples, issues, operational errors), packaging, and
quality tooling. It contains no dataset discovery, parsing, manifests, CLI, or
training code, and makes no claim about real RDD2022 counts, integrity, or
readiness.

## Raw-data immutability

Raw RDD2022 files are strictly read-only inputs. No command in this project
downloads, extracts, copies, renames, hashes-and-writes, or modifies raw
dataset files. Generated artifacts live outside the raw root.

## Setup

```powershell
uv sync --frozen --all-extras
```

Requires Python 3.11 (pinned in `.python-version`) and uv.

## Commands

```powershell
# Unit tests (no real dataset required)
uv run --frozen pytest -m "not rdd2022" -q

# Coverage for the data contracts
uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-fail-under=90 -q

# Quality gates
uv lock --check
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy --strict src/roadmind
uv run --frozen pip-audit
uv build --wheel --no-build-isolation
```

Real-dataset tests are marked `rdd2022` and are excluded by default. They
require `ROADMIND_RDD2022_ROOT` and are introduced in ROAD-001E.

## Non-goals

No dataset download, YOLO training, VLM, RAG, Agent, video pipeline, database,
DVC, MLflow, API, or UI. No real-data artifact directory is created by
ROAD-001A.
