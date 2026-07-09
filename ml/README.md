# ML Workspace

Offline training and experimentation workspace for the BMS project. Nothing
in this directory runs inside the production API — `backend/` only reads
finished artifacts from `ml/models/registry/`.

## Layout

```
ml/
├── data/
│   ├── raw/            # Original, untouched dataset downloads (NASA, CALCE, ...)
│   ├── processed/      # Cleaned/feature-engineered datasets ready for training
│   └── external/       # Reference data (e.g. battery spec sheets) not from the core datasets
├── notebooks/          # Exploratory analysis - promote reusable logic into src/ once stable
├── src/
│   ├── data/            # Loaders, validators, cleaning (NASA .mat parser, validation.py)
│   ├── features/        # Feature engineering, feature selection (battery_features.py)
│   ├── pipelines/       # End-to-end orchestration scripts (build_nasa_dataset.py)
│   ├── eda/             # Exploratory analysis report generation (generate_eda_report.py)
│   ├── models/           # Model definitions per task (soh/, rul/, failure/)
│   ├── training/         # Training pipelines, cross-validation, hyperparameter search
│   ├── evaluation/       # Metrics, model comparison, best-model selection
│   ├── explainability/   # SHAP / LIME wrappers shared with the backend
│   └── utils/            # Shared helpers (config loading, seeding, logging)
├── configs/             # YAML configs: dataset paths, battery metadata schema refs, model search spaces
├── reports/             # Generated EDA output (figures/, eda_summary.md) - gitignored, regenerate via the command below
└── models/
    └── registry/        # Versioned, serialized model artifacts (gitignored - see below)
```

## NASA dataset pipeline

```bash
# 1. Download and extract the NASA Battery Data Set into data/raw/nasa_raw/
#    (see configs/nasa_battery.yaml for the expected per-batch folder layout)

# 2. Build the cleaned, feature-engineered per-discharge-cycle dataset
python -m src.pipelines.build_nasa_dataset --config configs/nasa_battery.yaml --output-dir data/processed

# 3. Generate the EDA report (figures + correlation analysis + written summary)
python -m src.eda.generate_eda_report --dataset data/processed/nasa_cycle_features.parquet \
    --config configs/nasa_battery.yaml --output-dir reports
```

**Read `reports/eda_summary.md`'s "Methodological caveats" section before using this
dataset for modeling.** In short: some batteries were deliberately cycled at
multiple discharge current levels and ambient temperatures as part of NASA's
test design (not a single fixed profile), so raw capacity is confounded with
test condition for those batteries - `current_mean` and `ambient_temperature_c`
are kept as explicit features specifically so models can learn to condition on
this rather than being misled by it. The outlier detector
(`src/data/validation.py`) is regime-aware for the same reason.

## SOH model training

```bash
python -m src.training.soh.train_soh_models \
    --dataset data/processed/nasa_cycle_features.parquet \
    --registry-dir models/registry --reports-dir reports \
    --n-trials 25 --n-cv-splits 5 --n-repeats 15
```

Compares Random Forest, Extra Trees, Gradient Boosting, SVR, XGBoost,
LightGBM, and CatBoost. Each is tuned with Optuna over a `GroupKFold` (keyed
on `battery_id`) cross-validation, then **every tuned model is re-evaluated
across `n_repeats` independent random 80/20 grouped train/test splits** -
not just one - before the best is selected and promoted to the registry.

This two-phase design exists because a single grouped split over only ~34
batteries is a high-variance generalization estimate: on this project's
first run, one 80/20 split happened to concentrate several of the
deliberately multi-condition batteries (see "Methodological caveats" above)
into the test side, and every model's single-split test MAE came out
1.3-2.6x worse than its cross-validation MAE - purely from which batteries
that split happened to hold out. Read `reports/soh_training_report.md`
after training for the full single-split-vs-repeated-holdout comparison
that motivated this.

## Model registry contract

Every trained model is written to:

```
ml/models/registry/<task>/<version>/
├── model.joblib         # or model.pt for torch-based models
└── manifest.json        # algorithm, hyperparameters, metrics, dataset hash, feature schema, created_at
```

`<task>` is one of `soh`, `rul`, `failure`. The backend's model loader
(`backend/app/ml/`) resolves the latest (or explicitly pinned) version per
task by reading `manifest.json` - it never imports training code. This is
what makes "retrain without redeploying the API" and "automatic best model
selection" possible: a new version directory is just dropped in, and the
loader picks it up.

Registry contents are gitignored (large binaries don't belong in git) - use
DVC, a model artifact store, or object storage for team-shared versioning
once training is under way.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```
