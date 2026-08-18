# ROAD-001A — Project scaffold and typed data contracts — TDD evidence

Status: **GREEN**

## Approach

Tests were written against the approved public contract before the production
models existed. The contract was then implemented incrementally until all
tests passed without skips, xfails, or data/network dependencies.

## RED evidence

Command:

```powershell
uv run --frozen pytest tests/unit/test_data_models.py -q
```

Result (recorded before `src/roadmind/data/models.py` existed):

```text
ImportError while importing test module 'tests/unit/test_data_models.py'.
...
tests\unit\test_data_models.py:18: in <module>
    from roadmind.data import (
E   ImportError: cannot import name 'RDD2022_CLASS_NAMES' from 'roadmind.data' (D:\RoadMind\src\roadmind\data\__init__.py)
...
1 error in 0.46s
```

The failure is caused by the missing contracts (empty `roadmind.data` package),
not by broken test setup: the test module itself was syntactically valid and
all helper builders were importable from the target package surface.

## GREEN evidence

Command:

```powershell
uv run --frozen pytest tests/unit/test_data_models.py -q
```

Result:

```text
146 passed in 0.61s
```

## Coverage evidence

Command:

```powershell
uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-report=term-missing --cov-fail-under=90 -q
```

Result:

```text
Name    Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------
TOTAL     272      0     68      0   100%

Required test coverage of 90% reached. Total coverage: 100.00%
```

Branch-aware coverage for `roadmind.data` is 100% (requirement: >= 90%).

## Quality and packaging evidence

```powershell
uv lock --check
# OK (resolved 49 packages)

uv run --frozen ruff format --check .
# OK (all files formatted)

uv run --frozen ruff check .
# All checks passed!

uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 3 source files

uv run --frozen pytest -m "not rdd2022" -q
# 146 passed in 0.75s

uv run --frozen pip-audit
# No known vulnerabilities found
# (roadmind skipped: dependency not published on PyPI, expected for a local package)

uv build --wheel --no-build-isolation
# Successfully built dist\roadmind-0.1.0-py3-none-any.whl
```

## Known gaps and limitations

- No real RDD2022 data is read, hashed, or validated in ROAD-001A. All tests
  use hand-constructed objects.
- Discovery, VOC parsing, audit, manifest, duplicate detection, and CLI
  symbols are intentionally absent; a package-boundary test asserts this.
- Import-side-effect hygiene is verified by importing `roadmind` and
  `roadmind.data` in a fresh interpreter with a scrubbed environment and
  asserting no banned modules (PIL, defusedxml, torch, ultralytics, fastapi,
  pandas, numpy) load and no files are created.
- Local environment note: pytest's default tmp dir
  (`%TEMP%\pytest-of-huyho`) is inaccessible on this machine due to an
  unrelated ACL issue, so the side-effect test creates its own temporary
  directory instead of using the `tmp_path` fixture.

## Review round (REQUEST CHANGES) and fixes

Independent review found five issues; all were fixed and re-verified.

| Finding | Severity | Fix | Verification |
|---|---|---|---|
| `.gitignore` rule `data/` matched `src/roadmind/data/` (untracked, absent from wheel) | P0 | Root-anchored raw-data rules (`/data/`, `/datasets/`); wheel contents + install/import smoke steps added to CI | `git check-ignore` clean; wheel rebuilt and imported from site-packages in a fresh venv |
| `SampleRecord` did not tie image/XML paths to the declared domain | P1 | Enforce domain-rooted image and annotation paths, identical filename stem, `.jpg`/`.xml` extensions in `_check_invariants` | 6 new tests (domain mismatch image/XML, stem mismatch, wrong extensions, nested paths accepted) |
| Same bbox with different classes was accepted (contradicts `OBJECT_LABEL_CONFLICT`) | P1 | Duplicate detection now compares bbox only, class-independent | `test_same_bbox_different_class_rejected` replaces the previous acceptance test |
| `IssueDetail` accepted absolute paths and NaN/Infinity floats (leak + silent null serialization) | P1 | Reject non-finite floats and strings with control characters or embedded absolute paths | 3 parametrized rejection tests + control-char key test; existing context tests updated |
| `DatasetContractError` attributes were reassignable | P2 | `__slots__` + `__setattr__` guard; attributes set via `object.__setattr__` | `test_contract_error_immutable` covers `code`, `message`, `_context`, `context` |

Re-run evidence after fixes:

```text
uv run --frozen pytest tests/unit/test_data_models.py -q
# 165 passed in 0.58s

uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-fail-under=90 -q
# TOTAL     312      0     88      0   100%  (requirement: >= 90%)

uv run --frozen ruff format --check .   # OK
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 3 source files

uv run --frozen pytest -m "not rdd2022" -q   # 165 passed
uv run --frozen pip-audit                     # No known vulnerabilities found
uv lock --check                               # OK
uv build --wheel --no-build-isolation         # wheel contains roadmind.data/models.py
# wheel installed into a fresh venv; import resolves from site-packages
```

`git add -n .` now lists `src/roadmind/data/__init__.py` and
`src/roadmind/data/models.py` as tracked candidates, and `git check-ignore`
no longer matches them.

## Review round 2 (REQUEST CHANGES) and fixes

| Finding | Severity | Fix | Verification |
|---|---|---|---|
| `model_copy(update=...)` bypasses all validators (could build negative bbox, NaN `IssueDetail` serialized as `null`) | P1 | `RoadMindModel.model_copy` override rejects any `update` argument; plain and deep copies still work | `TestModelCopy` (bbox, issue detail, sample rejects; copy equality) |
| Safe-string validation still accepted embedded POSIX (`"failed at /etc/passwd"`) and UNC paths; `DatasetContractError.message` unvalidated | P1 | `_validate_detail_string` now rejects any backslash, embedded drive, embedded POSIX absolute (`(?<![\w:/])/(?=[^\s/])`), and POSIX-UNC (`(?<![\w:])//`) patterns; applied to `DatasetContractError.message` (newline → log injection rejected) | New rejection cases + accept cases (`ratio 1/2`, `2024/05/12`, `https://…`, `n/a`) and unsafe-message parametrization |
| Blanket `__setattr__` ban broke `add_note` and pickle round-trip | P2 | Guard only `code`, `message`, `_context`; custom `__setstate__` restores contract fields via `object.__setattr__` and the rest via `__dict__`; constructor resolves a raw string code during unpickling | `add_note` test, pickle round-trip test, partial/None `__setstate__` tests; JPEG/XML extensions now case-insensitive (`.JPG`/`.XML` accepted) |

Re-run evidence after round 2:

```text
uv run --frozen pytest tests/unit/test_data_models.py -q
# 183 passed in 0.73s

uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-fail-under=90 -q
# TOTAL     333      0    102      0   100%  (requirement: >= 90%)

uv run --frozen ruff format --check .   # OK
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 3 source files

uv run --frozen pytest -m "not rdd2022" -q   # 183 passed
uv run --frozen pip-audit                     # No known vulnerabilities found
uv lock --check                               # OK
uv build --wheel --no-build-isolation         # wheel contains roadmind/data/models.py
# wheel re-installed into the smoke venv; import resolves from site-packages
```

## Review round 3 (final) and fixes

| Finding | Severity | Fix | Verification |
|---|---|---|---|
| `__setstate__` was a public bypass: could swap `code`/`message`/`_context` for invalid values and crash `__str__` | P1 | Custom `__reduce__` reconstructs exclusively through `_rebuild_contract_error`, which calls the fully validated constructor; pickle state holds only exception metadata (`__notes__`). `__setstate__` rejects any state containing `code`, `message`, or `_context` and only restores vetted metadata | Pickle round-trip (incl. `__notes__` preservation), 4 rejection cases (`code` member, raw invalid string, unsafe `message`, wrong-typed `_context`), metadata-only state accepted, `None` state no-op, immutability asserts unchanged |

Re-run evidence after round 3:

```text
uv run --frozen pytest tests/unit/test_data_models.py -q
# 185 passed in 0.72s

uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-fail-under=90 -q
# TOTAL     339      0    102      0   100%  (requirement: >= 90%)

uv run --frozen ruff format --check .   # OK
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 3 source files

uv run --frozen pytest -m "not rdd2022" -q   # 185 passed
uv run --frozen pip-audit                     # No known vulnerabilities found
uv lock --check                               # OK
uv build --wheel --no-build-isolation         # wheel contains roadmind/data/models.py
# wheel re-installed into the smoke venv; import + pickle round-trip from site-packages
```

## Review round 4 (hardening P2) and fixes

| Finding | Severity | Fix | Verification |
|---|---|---|---|
| Unicode C1 controls (`U+0085`, `U+009B`) and line/paragraph separators (`U+2028`, `U+2029`) still accepted in details and messages | P2 | `_validate_detail_string` now rejects characters by Unicode category `Cc` (C0 + C1 controls) and `Zl`/`Zp` (line/paragraph separators) via `unicodedata.category` | 6-case separator parametrization, existing ASCII control cases, non-ASCII safe values accepted (`café`, `müller road`, `日本`) |
| `add_note()` accepted newlines and absolute paths, leaking them into formatted tracebacks | P2 | `DatasetContractError.add_note` validates the note through `_validate_detail_string` before delegating to `BaseException.add_note` | 7-case unsafe-note parametrization (newline, POSIX/drive/UNC paths, separators, C1 control); `__notes__` never created on rejection; valid notes + pickle preservation still pass |

Re-run evidence after round 4:

```text
uv run --frozen pytest tests/unit/test_data_models.py -q
# 200 passed in 0.69s

uv run --frozen pytest tests/unit/test_data_models.py --cov=roadmind.data --cov-branch --cov-fail-under=90 -q
# TOTAL     342      0    102      0   100%  (requirement: >= 90%)

uv run --frozen ruff format --check .   # OK
uv run --frozen ruff check .            # All checks passed!
uv run --frozen mypy --strict src/roadmind
# Success: no issues found in 3 source files

uv run --frozen pytest -m "not rdd2022" -q   # 200 passed
uv run --frozen pip-audit                     # No known vulnerabilities found
uv lock --check                               # OK
uv build --wheel --no-build-isolation         # wheel contains roadmind/data/models.py
# wheel re-installed into the smoke venv; import + add_note + pickle from site-packages
```
