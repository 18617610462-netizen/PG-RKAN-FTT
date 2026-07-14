# PG-RKAN-FTT for Rapid Seismic Damage Assessment

This repository provides a reproducible implementation of the FT-Transformer,
RKAN-FFN, RKAN-Attn, RKAN-FTT, and PG-RKAN-FTT models used for rapid structural
seismic damage assessment.

The repository is prepared for public release. It contains source code,
configuration files, the released PG-RKAN-FTT checkpoint, its fitted
preprocessor, and a 10% stratified sample of the tabular dataset. It does not
include the full dataset, raw monitoring records, SeismoStruct simulation
outputs, or intermediate experimental results.

## Paper

**Physics-guided RKAN-enhanced FT-Transformer for rapid structural seismic
damage assessment in earthquake early-warning applications**

If you use this code, please cite the associated paper once it is available.

## Repository Structure

```text
README.md
requirements.txt
LICENSE
train.py
evaluate.py
predict.py
models/
losses/
data_preprocess/
configs/
examples/
scripts/
figures/
data/
released_model/
```

## Installation

Create a clean Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code is written for Python 3.10+ and PyTorch 2.0+.

## Data Format

Training, validation, and test files should be provided as CSV files. The
default feature columns are defined in `configs/default.yaml`.

Required input columns:

```text
structure_type
Vs30
basement_stories
num_stories
T1
total_height
ground_floor_height
typical_floor_height
transverse_length
longitudinal_length
PGA
PGV
PGD
PGV_PGA
Ia
CAA
TBD_D5_95
ASI
HI
ERA
Sa
MIDR
```

The target column is `MIDR`. The structure type should use:

```text
frame
shear_wall
frame_shear_wall
masonry
```

Small fictional examples are provided in:

- `examples/demo_input.csv`
- `examples/demo_output.csv`

These files are only for demonstrating the input and output format. They are
not real monitoring data or simulation results.

A 10% stratified dataset sample is released for reproducibility:

- `data/sample_10_percent.csv`
- `data/sample_10_percent.xlsx`
- `data/sample_10_percent_summary.csv`

The sample was stratified by structure type with random seed 2026.

The trained PG-RKAN-FTT model assets are released in:

- `released_model/PG-RKAN-FTT_best_model.pt`
- `released_model/PG-RKAN-FTT_preprocessor.joblib`
- `released_model/metrics/PG-RKAN-FTT_metrics_summary.csv`
- `released_model/metrics/PG-RKAN-FTT_metrics_summary.xlsx`

## Training

Edit `configs/default.yaml` so that the CSV paths point to your local training,
validation, and test files:

```yaml
data:
  train_csv: data/train.csv
  val_csv: data/val.csv
  test_csv: data/test.csv
```

Then train:

```bash
python train.py --config configs/default.yaml --output_dir outputs/pg_rkan_ftt
```

The model name can be changed in the config:

```yaml
model:
  name: PG-RKAN-FTT
```

Supported model names:

```text
FT-Transformer
RKAN-FFN
RKAN-Attn
RKAN-FTT
PG-RKAN-FTT
```

## Physics-Informed Loss

The `losses/` module implements the physical information fusion loss used by
PG-RKAN-FTT:

- intensity-trend constraint
- MIDR threshold-neighborhood weighting constraint
- period-spectrum feature matching weighting constraint
- total physics-informed multi-objective loss

The structure-specific MIDR damage thresholds are stored in
`configs/default.yaml` and can be modified for another taxonomy if needed.

## Evaluation

Evaluate a trained checkpoint on a test CSV:

```bash
python evaluate.py \
  --checkpoint released_model/PG-RKAN-FTT_best_model.pt \
  --preprocessor released_model/PG-RKAN-FTT_preprocessor.joblib \
  --test_csv data/test.csv \
  --output_csv outputs/pg_rkan_ftt/test_predictions.csv
```

The script reports MIDR regression metrics and damage-state classification
metrics.

## Prediction

Run prediction on a CSV file with the input columns:

```bash
python predict.py \
  --checkpoint released_model/PG-RKAN-FTT_best_model.pt \
  --preprocessor released_model/PG-RKAN-FTT_preprocessor.joblib \
  --input_csv examples/demo_input.csv \
  --output_csv outputs/demo_predictions.csv
```

The output contains predicted MIDR and predicted damage state.

## Data Availability

The full dataset is not publicly released and is available from the
corresponding author upon reasonable request.

This public repository intentionally excludes:

- full datasets
- CESMD raw or processed monitoring data
- SeismoStruct simulation results
- checkpoints and trained weights other than the released PG-RKAN-FTT assets
- outputs and intermediate result files
- local paths, credentials, API keys, or private metadata

## Public Release Check

Before pushing to GitHub, run:

```bash
python scripts/check_public_repo.py
```

The script checks for common local absolute paths and disallowed large/private
file types.

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{PG_RKAN_FTT_SeismicDamage,
  title = {Physics-guided RKAN-enhanced FT-Transformer for rapid structural seismic damage assessment},
  author = {Author information omitted for review},
  journal = {To be updated},
  year = {2026}
}
```
