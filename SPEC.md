# TIF Series Analyzer — Implementation Specification

A terminal-interactive Python application that scans a directory of `.tif` images,
extracts a numeric parameter from each filename, computes a metric per image, and
plots metric-vs-parameter.

This document is a design handoff. It fixes the architecture and the decisions that
are expensive to change later. Anything not specified here is left to implementer
judgement, but the **Locked decisions** section must be honoured.

---

## 1. Locked decisions

These were settled deliberately during design. Do not silently revise them.

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Multi-page TIFF → one data point per file.** Loader has a policy setting `first` / `mean` / `error`, defaulting to `error`. | A stack must never silently collapse into a number the user didn't ask for. |
| 2 | **Metrics return `dict[str, float]`**, never a bare float. | Metrics with multiple outputs (mean + std) must not require a plumbing rewrite. |
| 3 | **Preprocessing is an ordered list of steps**, empty in v1, pass-through. | ROI crop / dark-frame subtraction are plausibly coming; the plumbing exists now. |
| 4 | **Duplicate parameter values are a hard error, raised at discovery time**, before any file I/O. Overridable via a toggle, default off. | Fast failure; user fixes the pattern without waiting on I/O. |
| 5 | **Results are cached**, keyed on `(path, mtime_ns, size, metric_key, preprocess_hash)`. | Re-running after a plot tweak shouldn't re-read large images. |
| 6 | **Plotting is interactive** (blocking matplotlib window). | See §8 for the consequences — this one has real trade-offs. |
| 7 | **Last used directory is persisted** and offered as the default on next launch. | See §9. |

---

## 2. Data flow

```
directory ──► discover ──► [FileRecord]        (filename parsing only, no I/O)
                             │
                             ├─ duplicate check (hard error)
                             ▼
                          load ──► validate pages ──► preprocess ──► metric.compute
                             │                                            │
                             ▼                                            ▼
                        RunReport ◄──────────────── [MetricResult] ── dict[str, float]
                             │
                             ├──► interactive plot
                             └──► CSV export
```

---

## 3. Module layout

| Module | Responsibility |
|--------|----------------|
| `discovery.py` | Glob the directory, apply the regex, build `FileRecord`s, detect duplicates. No image I/O. |
| `loading.py` | Open TIFFs, enforce the multi-page policy, return a numpy array. |
| `preprocess.py` | Ordered step pipeline + canonical serialization + hashing. Empty in v1. |
| `metrics.py` | Metric registry. Metrics self-register; nothing else hardcodes a metric list. |
| `analysis.py` | Orchestrates load → preprocess → compute. Generator, so the UI can show progress. Collects failures rather than aborting. |
| `cache.py` | In-memory result cache (v1). Key as per locked decision 5. |
| `plotting.py` | Builds and shows the matplotlib figure. |
| `config.py` | Persisted user config (last directory). |
| `state.py` | The `AppState` dataclass + invalidation rules. |
| `ui/` | Menu loop, key dispatch, prompts, rendering. The only module that touches stdin/stdout. |
| `app.py` | Entry point; wires everything together. |

**Hard rule:** nothing outside `ui/` prints or reads from the terminal. The analysis
layer must be importable and driveable from a script or a notebook with no UI present.

---

## 4. Data structures

```python
@dataclass(frozen=True)
class FileRecord:
    path: Path
    param_name: str      # from the regex named group, e.g. "power"
    param_value: float
    raw_match: str       # the matched substring, for error messages

@dataclass(frozen=True)
class OutputSpec:
    key: str             # "std"
    label: str           # "Standard deviation"
    unit: str | None     # "counts", or None

@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    description: str
    outputs: tuple[OutputSpec, ...]
    compute: Callable[[np.ndarray], dict[str, float]]

@dataclass
class MetricResult:
    record: FileRecord
    values: dict[str, float]
    duration_s: float
    from_cache: bool

@dataclass
class RunReport:
    results: list[MetricResult]
    failures: list[tuple[Path, Exception]]
    config_snapshot: dict     # dir, pattern, metric key, preprocess hash, timestamp
```

`outputs` is declared **statically**, so the UI can list what a metric produces and
offer a series selection *without running anything first*.

**Runtime contract:** the dict returned by `compute` must have exactly the declared
output keys. A mismatch is a bug in the metric, not a data problem — raise, don't warn.

---

## 5. Discovery and filename parsing

The pattern is a **regex with exactly one named group**. The group's name becomes the
x-axis label; its captured text is parsed as a float. This means adding a new
parameter type later costs nothing.

### Default pattern

```
medida_\d+_(?P<power>[\d.]+)mW\.tiff?$
```

Matches `medida_001_1mW.tif`, `medida_002_2mW.tif`, `medida_017_12.5mW.tiff`.

Note the sequence index is matched with a **bare `\d+`, not a named group** — only the
parameter gets a name, otherwise the app cannot tell which group is the x-axis.

Ship a small table of example patterns in the pattern prompt so users have something
to edit rather than compose from scratch.

### Pattern validation (at the prompt, before accepting)

1. Compiles as a regex.
2. Contains **exactly one** named group. Zero → "no parameter to plot against";
   two or more → "ambiguous, which is the x-axis?".

Both are instant checks. Catching them here beats a baffling zero-match result later.

### Reporting

Distinguish three counts and show all of them:

- files matching the glob (`*.tif` by default)
- of those, files matching the regex
- of those, files whose captured text parses as a float

A file that matched the glob but not the regex is **reported, never silently dropped.**
Silent dropping is the single most likely way this app misleads someone.

### Duplicate detection

Group records by `param_value` **as a parsed float**, not by the raw string —
`5.0mW`, `5.00mW` and `05mW` are the same point and string comparison would let them
through into a mangled plot.

Report **all** collisions at once, grouped by value with filenames listed underneath.
One-at-a-time whack-a-mole across 40 files is miserable.

Provide a `[!]` toggle to allow duplicates, default off, for the day replicates are
genuinely wanted.

---

## 6. Loading

Use **`tifffile`**, not PIL. PIL is quietly unreliable on 16/32-bit scientific TIFFs
and on unusual photometric interpretations.

- Read as-is; **do not normalise or rescale**. A std dev in raw counts is the point.
- Check page count. Apply the policy: `error` (default) / `first` / `mean`.
- Convert to float64 before computing, to avoid integer overflow in accumulations.
- Reject colour/multi-channel images in v1 with a clear message.

---

## 7. Metrics

Registry: `dict[str, MetricSpec]`, populated by a `@register` decorator.

v1 ships one metric:

```
key:     "std"
name:    "Standard deviation"
outputs: (OutputSpec("std", "Standard deviation", None),)
```

Use `ddof=1` (sample standard deviation) and say so in the description — the
population/sample distinction matters to anyone doing noise analysis.

Adding a metric must require touching **only** `metrics.py`.

---

## 8. Plotting — interactive

`plt.show()` on the default GUI backend, blocking. The menu loop resumes when the
user closes the window.

Requirements:

- **Detect a headless / non-GUI backend at startup.** If matplotlib resolves to `Agg`
  (common over SSH without X forwarding, and in bare containers), the plot option must
  fall back to saving a PNG and printing its path, with a one-line explanation. It must
  **not** silently do nothing — that is a confusing failure mode and it will happen.
- Sort points by `param_value` before plotting.
- Markers **and** connecting line; small datasets read badly as bare points.
- Axis labels: x from `param_name` plus unit parsed from the pattern where available
  (`mW`), y from the `OutputSpec` label and unit.
- Title carries the `config_snapshot`: directory name, metric, file count, timestamp.
  A saved figure should be self-describing.
- Series selection when the active metric declares more than one output.

---

## 9. Persisted configuration

Location: `$XDG_CONFIG_HOME/tif-analyzer/config.json`, falling back to
`~/.config/tif-analyzer/config.json`. On Windows, `%APPDATA%`. No extra dependency
needed for this.

Persisted keys: `last_directory`, `last_pattern`, `last_metric_key`, `allow_duplicates`.

Behaviour:

- Written **only after a directory is successfully validated**, so a typo never becomes
  the next session's default.
- On startup, load and validate. If `last_directory` no longer exists, fall back to CWD
  and say so once — don't crash, and don't silently pretend it worked.
- The directory prompt pre-fills with the remembered value; bare Enter accepts it.
- A missing or corrupt config file is a non-event: use defaults, carry on.

---

## 10. Application state

```python
@dataclass
class AppState:
    directory: Path
    glob: str = "*.tif"
    pattern: str = DEFAULT_PATTERN
    metric_key: str = "std"
    preprocess_steps: list = field(default_factory=list)
    allow_duplicates: bool = False
    records: list[FileRecord] | None = None
    report: RunReport | None = None
    selected_series: str | None = None
```

### Invalidation rule

Changing `directory`, `glob`, `pattern`, `metric_key`, `preprocess_steps`, or
`allow_duplicates` **clears `records` and `report`**.

Changing `selected_series` does **not** — it only affects rendering.

Implement this as **one function that every mutation goes through**, not as scattered
assignments. Stale-plot bugs come from exactly this and they are hard to spot because
the output looks plausible.

---

## 11. Terminal UI

### Input mode

Use **`readchar`** for single-key menu dispatch (no Enter required).

Sub-prompts that take typed text (directory, pattern) switch to line input; use
`readline` there for **tab-completion of paths and history**. Entering a long absolute
path without completion is unpleasant enough that users will avoid the feature.

If `readchar` is unavailable or raw mode fails (some IDE consoles), degrade to
key-plus-Enter via `input()` rather than crashing.

### Main screen

The header renders from `AppState`, so the user always sees where they are:

```
  TIF Series Analyzer

  dir:      /home/user/data/2026-08-12      (47 tif, 44 matched)
  pattern:  medida_\d+_(?P<power>[\d.]+)mW\.tiff?$
  metric:   std — Standard deviation
  results:  44 points   (2 failures)

  [d] directory        [p] pattern
  [m] metric           [g] glob filter
  [r] run analysis     [v] view plot
  [e] export CSV       [!] allow duplicates: OFF
  [q] quit
```

### State machine

```
MAIN ─┬─ [d] → PROMPT_DIR       line input, tab completion, validate, persist
      ├─ [p] → PROMPT_PATTERN   validate: compiles + exactly one named group
      ├─ [g] → PROMPT_GLOB
      ├─ [m] → MENU_METRIC
      ├─ [r] → RUNNING → REPORT (only if failures) → MAIN
      ├─ [s] → MENU_SERIES      shown only when metric has >1 output
      ├─ [v] → PLOT             blocking window; Agg fallback per §8
      ├─ [e] → PROMPT_EXPORT
      ├─ [!] → toggle duplicates
      └─ [q] → exit
```

Every sub-state has an escape key (`Esc` or `b`) back to MAIN with no changes applied.

**Disabled options stay visible with a reason** rather than vanishing —
`[r] run analysis  (no matching files)` teaches; a missing line puzzles.

### Path input

`Path(raw).expanduser().resolve()` handles relative, absolute and `~` uniformly.
Validate that it exists and is a directory **immediately**, re-prompt on failure.

### Progress

`analysis.py` is a generator, so print a single updating line:
`[ 23/44 ] medida_023_23mW.tif  (cached)`. Keep it to one line — scrolling 44 lines
buries the failure report that follows.

---

## 12. Error handling

- **Per-file failures do not abort the run.** Collect into `RunReport.failures` and
  present them together at the end, then return to MAIN with partial results usable.
- **Discovery-level problems** (no matches, duplicates) block the run and explain why.
- **Programming errors** (metric returning wrong keys) raise loudly. Do not defend
  against your own bugs by swallowing them.

---

## 13. CSV export

**Long format:**

```
filename,param_name,param_value,output_key,value
medida_001_1mW.tif,power,1.0,std,12.4471
```

Long beats wide here: it survives metrics with differing output sets, and pivots
trivially in pandas. Prepend the `config_snapshot` as `#`-commented header lines so an
exported file records how it was produced.

---

## 14. Dependencies

- `numpy`
- `tifffile`
- `matplotlib`
- `readchar` (soft dependency — degrade gracefully if absent)

Target Python 3.10+. Type-hint throughout; the dataclasses above are the contract
between layers.

---

## 15. Suggested build order

1. `discovery.py` + `state.py` + tests on synthetic filenames — no images needed yet,
   and it flushes out pattern edge cases early.
2. `loading.py` + `metrics.py` with the `std` metric.
3. `analysis.py` + `cache.py`, driven from a throwaway script.
4. `ui/` on top of the working core.
5. `plotting.py`, then `config.py`, then CSV export.

Steps 1–3 are the whole application; the UI is a client of them. Build in that order
and the "no terminal I/O outside `ui/`" rule enforces itself.

---

## 16. Deliberately out of scope for v1

Recorded so they aren't accidentally designed out:

- Preprocessing steps with actual behaviour (the pipeline exists, empty).
- Persistent on-disk cache (in-memory only; sidecar JSON is the natural extension).
- Two-parameter extraction / grouped series.
- Recursive directory scanning.
- Non-TIFF formats.
