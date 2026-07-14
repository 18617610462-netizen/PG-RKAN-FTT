# Public Release File List

This folder is prepared for GitHub release of the PG-RKAN-FTT model.

## Included Model Assets

- `released_model/PG-RKAN-FTT_best_model.pt`
- `released_model/PG-RKAN-FTT_preprocessor.joblib`
- `released_model/metrics/PG-RKAN-FTT_metrics_summary.csv`
- `released_model/metrics/PG-RKAN-FTT_metrics_summary.xlsx`

## Included Dataset Sample

- `data/sample_10_percent.csv`
- `data/sample_10_percent.xlsx`
- `data/sample_10_percent_summary.csv`

The sample contains 1054 rows, corresponding to 10% of the original 10540-row
tabular dataset, stratified by structure type with random seed 2026.

## Excluded From Public Release

- Full dataset
- Raw monitoring records
- Full simulation outputs
- Other checkpoints and model weights
- Intermediate experiment outputs and result folders
- Local absolute paths and private metadata
