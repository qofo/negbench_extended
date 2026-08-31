# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

NegBench (CVPR 2025) — the public benchmark for "Vision-Language Models Do *Not* Understand Negation" — plus a
substantial local research extension. The upstream half (`README.md`, `benchmarks/README.md`, `synthetic_datasets/`)
reproduces MCQ/retrieval numbers. The local half (everything under `benchmarks/src/analysis/` and most
`benchmarks/src/evaluation/eval_*.py`) asks *why* CLIP fails at negation:

- **RQ1** Is negation lost in the representation, or only unreadable by the parameter-free cosine head?
- **RQ2** Is the negation signal a few shortcut dimensions or distributed over all 512?
- **RQ3** What do bilinear / cross-dimensional heads recover that a diagonal (cosine) match cannot?

`GUIDE.md` at the repo root is the technical hub: full math for every experiment (§5, ①–⑩), a per-directory
navigation table, and a "Limitations & Reviewer Response" section (§7) recording known design flaws.
Read it before touching analysis code. Per-module `GUIDE.md` files sit in each `benchmarks/src/*/` directory,
and `DATA_SCHEMA.md` documents every CSV column. These docs drift — verify claims against source.
A 2026-08-29 pass fixed the known cases (the `eval_per_object_polarity_probe.py` "empty file" entry, the
dangling `CLI_CHEATSHEET.md` links, and 9 wrong relative paths), but assume nothing has been re-checked since.

## Environment and setup

```bash
conda activate clip_negation            # python 3.9
pip install -e benchmarks/              # editable install of the vendored OpenCLIP fork
```

The editable install puts `benchmarks/src/` on `sys.path`, which has two consequences worth internalizing:

- `import open_clip` resolves to **`benchmarks/src/open_clip/`** (a modified fork), never to PyPI OpenCLIP.
  Changes there affect every entrypoint.
- `training.*`, `evaluation.*`, `analysis.*` are importable as top-level packages from anywhere.

Optional extras: `pip install -r benchmarks/requirements-pca.txt` (scikit-learn/matplotlib/seaborn — required by
almost all analysis scripts), `requirements-llava.txt`, `requirements-training.txt`.

## Running things — three coexisting invocation conventions

The codebase accreted three import styles. All three work **from the repo root** with the env active
(cwd `''` on the path supplies the `benchmarks.src.*` namespace; the editable install supplies the rest):

| Style | Used by | Canonical command |
|---|---|---|
| `from training.x` / `from evaluation.x` | upstream eval + training (`eval_negation.py`, `mcq.py`, `retrieval.py`, `training/`) | `cd benchmarks && python -m src.evaluation.eval_negation ...` |
| `from analysis.x` | `analysis/run_beaf_*.py`, `train_beaf_dual_probes.py` | `python -m analysis.run_beaf_analysis_v2 ...` |
| `from benchmarks.src.analysis.x` | all newer mechanistic `eval_*.py` (E1–E4, ablations, probes) | `python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis ...` |

If an import fails, `export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src"` satisfies all three at once
(this is what `benchmarks/scripts/*.sh` do). Every script's module docstring carries its own `Usage:` block —
trust that over guessing.

**Run mechanistic/analysis scripts from the repo root.** Default `--csv_path` values are repo-root-relative
(`benchmarks/data/images/...`), and image paths *inside* the CSVs are relative (`data/coco/images/val2014/...`),
resolved through the `data -> benchmarks/data` symlink at the root.

### Representative commands

```bash
# Upstream MCQ/retrieval eval (CLIP or LLaVA; edit the vars at the top first)
bash benchmarks/scripts/run_single_model_evaluations.sh

# 6 scoring heads, 5-fold OOF comparison (RQ1/RQ3 core result)
bash benchmarks/scripts/run_scoring_head_evaluations.sh

# 16-step text-encoder geometry pipeline
python -m analysis.run_analysis --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --output_dir logs/analysis_modular/openai_vit_b32

# E1-E4 unary mechanistic analysis on BEAF counterfactual pairs
python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \
    --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
    --output_dir logs/evaluation/03_discarded/2026-08-28_unary_mechanistic_LEAKY_CV
```

`make test` in `benchmarks/` (`python -m pytest -x -s -v tests`) runs a small regression suite covering the
shared helpers and the experiment invariants that carry meaning — the Hadamard identity, the eval-time image
transform, the fold splitters, the counterfactual pairing contract. It needs `pip install pytest` and takes
~12s; it deliberately does **not** cover the experiments end to end. That validation is still empirical:
scripts print sanity assertions (e.g. `extractor.assert_embedding_consistency`, the E2
`|Δ_empirical − Δ_analytical| < 1e-6` identity check) and dump reports for inspection.

## Architecture

```
benchmarks/src/
  open_clip/       vendored, modified OpenCLIP — the model layer everything imports
  training/        NegCLIP fine-tuning: main.py, train.py, data.py (CsvMCQDataset...), params.py (the CLI parser
                   that eval_negation.py also reuses), video_utils/
  evaluation/      scoring heads + MCQ/retrieval eval + ~20 standalone mechanistic experiments
  analysis/        representation geometry: extractor.py, metrics.py, subspace_analysis.py, reporter.py
    beaf/          counterfactual-pair framework: beaf_loader.py, beaf_stats.py, vision_mechanisms.py, probe_factory.py
  data_generation/ builds the paired-caption and BEAF AB-swap CSVs consumed by everything above
  llava/, e5v_analysis/   alternative model families evaluated with the same MCQ protocol
```

Data flows one way: `data_generation/` → CSVs in `benchmarks/data/images/` → feature extraction
(`analysis/extractor.py` for text, `analysis/beaf/vision_mechanisms.py` for vision — both do a **single forward
pass** returning every layer *and* every pipeline step) → probes/scorers → `logs/evaluation/<run>/`.

### Modules that are single sources of truth (don't re-implement)

- **`analysis/config.py`** — `PipelineStep`/`MetadataKey` enums, `set_seed()`, `to_bool()`,
  `coerce_bool_column()`, `get_layer_features()`, `DEFAULT_TUNING_GRIDS`, and the batched geometry ops. These
  were deduplicated out of four files on purpose. Parse `object_in_image` through `coerce_bool_column`, never
  by inlining a `str(x).lower() == "true"` lambda — that fork is what made experiments disagree before.
- **`analysis/paths.py`** — `resolve_image_path()`, the one rule for turning a CSV image path into a real one.
  The four E1/E2 entrypoints import it rather than each carrying a copy.
- **`analysis/beaf/probe_factory.py`** — the four PyTorch probes (`LowRankBilinear`, `FullBilinear`, `MLPVision`,
  `ElementWiseNonLinear`), `SUPPORTED_PROBES`, and `PyTorchProbeEstimator` (sklearn-API wrapper so probes drop
  into `GroupKFold`/`StratifiedKFold` loops).
- **`evaluation/scoring_heads.py`** — 9 scoring heads behind `build_scorer` (Cosine → Bilinear → MLP →
  LowRankBilinear → NonLinearBiEncoder → DualClassifierProduct). Note `eval_scoring_heads.py` only compares
  6 of them; `eval_rank_sweep.py` and `eval_vision_ablation_shortcut.py` reach the rest. Adding a head means
  adding it here **and** to the registry each of those scripts iterates.
- **`analysis/extractor.py` / `beaf/vision_mechanisms.py`** — layerwise feature extraction. Downstream scripts
  import these rather than re-running encoders.

### Conventions that carry experimental meaning

- **CV strategy is a claim, not a detail.** `StratifiedKFold(5)` for within-distribution accuracy; `GroupKFold` on
  `object_name` or `pair_id` whenever the question is generalization to unseen objects/pairs. Copying the wrong
  one silently converts a generalization result into a memorization result. §6 of `evaluation/GUIDE.md` tabulates
  which experiment uses which.
- **`--no_bias` / `--no-bias`** appears across analysis entrypoints and sets `fit_intercept=False` on probes — it
  exists to test whether results survive without an intercept absorbing class priors. Keep it plumbed through
  when adding probes.
- **Seeds**: always via `set_seed()` from `analysis/config.py`; default 42 everywhere.
- **Plots**: `matplotlib.use("Agg")` before `pyplot` import — these run headless on GPU nodes.
- Every experiment writes a triple: per-item CSV + summary JSON + figure PNGs, into
  `logs/evaluation/<experiment_name>/`. Later scripts read earlier scripts' JSON (e.g. the E2 sanity check reads
  `full_mechanistic_report.json`), so keep output filenames stable.

### Data

Four CSV schemas, fully specified in `DATA_SCHEMA.md`: **A** paired-caption 6-column (`image_path`, `object_name`,
`positive_caption`, `negative_caption`, `object_in_image`, `source_template`) — the workhorse for all
representation analysis; **B** MCQ 7-column; **C** retrieval (`filepath` + list-literal `captions`); **D** CheXpert
binary MCQ. BEAF counterfactual files are schema A with an integrity contract: consecutive row pairs share
`object_name`/`source_template` and differ only in `object_in_image`; `beaf_loader.load_and_verify_counterfactual_pairs`
enforces this — use it rather than reading the CSV directly.

`benchmarks/data/`, `benchmarks/models/`, and `logs/` are gitignored. Downloaded images/checkpoints and all
experiment output live outside version control; only code and the `results/*.csv` paper tables are tracked.

### Known-fragile results (from `GUIDE.md` §7)

- `DualClassifierProductScorer` is unconditional in the vision term — `f_V(v)` sees no text, so it emits a constant
  per image. Any new scorer must be conditional (`v^T W t`-shaped).
- Text-probe accuracies near 99.9% are suspected token-presence detection ("not" as a lexical cue);
  `eval_word_swap_probe.py` and the AB-swap experiments are the controls for it.
- BEAF inpainted counterfactuals can leak blur/brush artifacts; `audit_ab_swap_dataset.py` is the priority-0 audit
  and `eval_e1_placebo_test.py` the placebo control. Run them before believing a vision-side effect.
