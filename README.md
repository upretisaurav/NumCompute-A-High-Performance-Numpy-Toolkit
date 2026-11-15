# NumCompute-Stream

Streaming machine learning with NumPy, built on top of the NumCompute toolkit from assignment 2.1.

The main part is a **decision tree classifier written from scratch** with a vectorised split search and **buffered `partial_fit`**, so it can learn from data that arrives in chunks. Around it are streaming versions of the 2.1 modules: preprocessing with `partial_fit`, chunk-based statistics, metrics that accumulate over chunks, a pipeline that forwards `y`, a test-then-train loop with logging, and plots.

Only Python, NumPy and matplotlib are used.

> The group's 2.1 package `numcompute/` is unchanged. All new work is in `numcompute_stream/`.

## Install

Python 3.12+ is required.

```bash
git clone https://github.com/upretisaurav/NumCompute-A-High-Performance-Numpy-Toolkit.git
cd NumCompute-A-High-Performance-Numpy-Toolkit
git checkout individual

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(With [uv](https://docs.astral.sh/uv/) instead: `uv sync`.)

## Quick start

```python
from numcompute_stream import (
    DecisionTreeClassifier, Pipeline, SimpleImputer, StandardScaler,
    StreamTrainer, iter_chunks, load_csv,
)

X, y = load_csv("demo/data/electricity.csv")          # last column is the target

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

# Option 1: plain partial_fit loop
for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=250):
    pipe.partial_fit(X_chunk, y_chunk)

# Option 2: test-then-train with logging
trainer = StreamTrainer(pipe)
history = trainer.run(iter_chunks(X, y, chunk_size=250))
print(trainer.log_table(every=10))
```

## Running things

| What | Command (from the repo root) |
|---|---|
| All tests (2.1 + new) | `pytest` |
| Only the new tests | `pytest tests/test_*.py` |
| Demo notebook | `jupyter notebook demo/stream_demo.ipynb` then Run All |
| Split benchmark | `python -m benchmark.bench_split` |
| Predict benchmark | `python -m benchmark.bench_predict` |
| Buffer size sweep | `python -m benchmark.bench_buffer_size` |
| Imputer benchmark | `python -m benchmark.bench_imputer` |
| Re-download the dataset | `python demo/data/prepare_electricity.py` |

Benchmark results are written to `benchmark/results/` (CSV, markdown table and PNG).

## Modules

| Module | What it provides |
|---|---|
| `tree` | `DecisionTreeClassifier` (`fit`, `partial_fit`, `flush`, `predict`, `predict_proba`), `gini`, `entropy`, `best_split` (vectorised) and `best_split_loop` (for benchmarking) |
| `io` | `load_csv` (returns `X, y`), `iter_chunks`, `read_csv_chunks` (reads the file a chunk at a time) |
| `preprocessing` | `StandardScaler`, `SimpleImputer`, `OneHotEncoder` with `partial_fit`; `MinMaxScaler` |
| `stats` | `StreamingStats.update_stats`: running mean/variance/min/max, fixed-bin histogram with optional sliding window, approximate quantiles; plus the 2.1 functions |
| `metrics` | `ConfusionMatrix`, `Accuracy`, `Precision`, `Recall`, `F1` (`update`/`result`/`reset`), `RollingMetric`, `StreamingAUC`; function versions `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `mse` |
| `pipeline` | `Pipeline` with `fit(X, y)`, `partial_fit(X, y)`, `predict`, `predict_proba`, `named_steps` |
| `stream` | `StreamTrainer`: test-then-train loop with per-chunk history and `log_table()` |
| `visualise` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` |
| `utils` | input checks, `safe_divide`, `logsumexp`, `softmax`, `sigmoid`, running-moment helpers |

## Design notes

**Split search.** For each feature the column is sorted once, the one-hot labels are cumulatively summed down the sorted rows (left-child class counts for every cut at once) and the right counts are `total - left`. The gain of every cut is then one array expression. This is O(n log n) per feature instead of O(n²) for a loop over thresholds.

**Rules.**
- A row goes left when `x <= threshold`; thresholds are midpoints between two *different* neighbouring values, so equal values are never split apart.
- Ties: the first split found wins (lowest feature index, then lowest threshold). A split needs a gain above `1e-12`.
- Leaves predict the majority class; ties go to the smallest class.
- NaN: the median of each feature in the training rows is stored at every (re)train and NaN is replaced with it before splitting and predicting. An all-NaN feature gets 0. Inf is rejected.

**Buffered streaming.** `partial_fit` adds the chunk to a buffer. Once at least `buffer_size` rows are buffered, a new tree is trained on all buffered rows and the buffer is cleared. Between retrains `predict` uses the last tree. Each tree only knows the most recent `buffer_size` rows, which forgets old data but helps under concept drift. Rows left in the buffer at the end of a stream are only used if `flush()` is called. New class labels can appear in any chunk; `predict_proba` gives them probability 0 until the next retrain.

**Test-then-train.** `StreamTrainer` scores each chunk with the current model *before* training on it, so all scores are on unseen data. Chunks before the first tree exists are logged as N/A.

**Streaming statistics.** The scaler and `StreamingStats` share one helper (`utils.merge_moments`, Chan et al.) that merges the count/mean/M2 of each chunk, which avoids the cancellation of the naive `E[x²] - E[x]²` formula.

**OneHotEncoder width.** Categories are fixed after the first chunk. If new categories were added later, the output width would change and a tree trained on the old width would break. Later unseen categories are encoded as all zeros and listed in `unseen_categories_`.

## Fixes from the 2.1 feedback

| Issue in 2.1 | Fix | Test that catches it |
|---|---|---|
| No chunked input | `io.iter_chunks`, `io.read_csv_chunks` | `tests/test_stream_io.py` |
| Tiny-scale standardisation: `np.isclose(std, 0)` treats a column like `[1e-12, 2e-12, 3e-12]` as constant, so it is never scaled | tolerance relative to the mean | `test_tiny_scale_column_is_standardised` |
| F1 returned NaN when a class was never predicted, and averaging was not stated | metrics from one confusion matrix, `safe_divide`, `average="macro"`/`"binary"` | `test_f1_class_never_predicted_is_zero_not_nan` |
| Pipeline did not pass `y` to the final step | `fit(X, y)` / `partial_fit(X, y)` forward `y` | `test_fit_forwards_y_to_the_model` |
| Numerical utilities missing | `safe_divide`, `logsumexp`, `softmax`, `sigmoid`, moment helpers | `tests/test_stream_utils.py` |
| Imputer benchmark was wrong (aliased arrays so the check compared an array with itself, the loop filled every cell, timings stored at the wrong index, 5×5 input) | `benchmark/bench_imputer.py` | asserts inside the benchmark |

Note: the new metric functions take `(y_true, y_pred)`, the opposite order of the 2.1 functions.

## Benchmark results

Median of 5 runs on an Apple Silicon laptop. Each benchmark checks that both versions give the same result before timing.

**Best split search, loop vs vectorised** (`bench_split.py`)

| n_samples | n_features | loop_ms | vectorised_ms | speedup |
|---|---|---|---|---|
| 500 | 1 | 11.89 | 0.08 | 141× |
| 2000 | 1 | 71.33 | 0.25 | 288× |
| 10000 | 1 | 1127.91 | 1.28 | 881× |
| 500 | 5 | 63.91 | 0.62 | 102× |
| 2000 | 5 | 356.53 | 2.43 | 147× |
| 10000 | 5 | 6103.77 | 6.66 | 916× |

**Predict, per-row loop vs masks** (`bench_predict.py`, tree with 251 nodes)

| n_samples | loop_ms | vectorised_ms | speedup |
|---|---|---|---|
| 1000 | 1.8 | 0.5 | 3.6× |
| 10000 | 17.46 | 1.41 | 12.4× |
| 50000 | 88.69 | 5.79 | 15.3× |

**Buffer size on the Electricity stream** (`bench_buffer_size.py`, chunks of 250 rows, `max_depth=5`)

| buffer_size | final cumulative accuracy | retrains | total retrain time (s) |
|---|---|---|---|
| 100 | 0.7515 | 181 | 0.84 |
| 250 | 0.7515 | 181 | 0.97 |
| 500 | 0.7390 | 90 | 0.70 |
| 1000 | 0.7370 | 45 | 0.47 |
| 2000 | 0.7315 | 22 | 0.33 |

Smaller buffers retrain more often on recent data and follow the drift better; larger buffers cost less. With chunks of 250, `buffer_size=100` and `250` behave the same (every chunk triggers a retrain); their time difference is noise.

**Imputer, loop vs vectorised** (`bench_imputer.py`, 10 features, 10% NaN)

| n_samples | strategy | loop_ms | vectorised_ms | speedup |
|---|---|---|---|---|
| 1000 | constant | 5.29 | 0.056 | 95× |
| 10000 | constant | 54.4 | 0.553 | 98× |
| 100000 | constant | 559.02 | 9.034 | 62× |
| 1000 | mean | 11.72 | 0.064 | 183× |
| 10000 | mean | 120.77 | 0.519 | 233× |
| 100000 | mean | 1195.24 | 6.98 | 171× |

## Dataset

`demo/data/electricity.csv` is the Electricity (elec2) dataset from OpenML (dataset 151, Harries 1999): 45,312 half-hourly records of the NSW electricity market, label = price UP (1) or DOWN (0). The `date` column was dropped and 2% of the feature values were removed at random (seed 0) to exercise the imputer. See `demo/data/prepare_electricity.py`.

## Project layout

```
numcompute_stream/   new package (tree, io, preprocessing, stats, metrics, pipeline, stream, visualise, utils)
numcompute/          group package from 2.1 (unchanged)
tests/               test_*.py for numcompute_stream, *_test.py for numcompute
demo/                stream_demo.ipynb, data/, figures/
benchmark/           benchmark scripts and results/
```
