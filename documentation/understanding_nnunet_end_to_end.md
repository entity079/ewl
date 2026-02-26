# Understanding nnU-Net v2 end-to-end (coding teacher guide)

If you want to understand **the whole nnU-Net**, this is the big-picture walkthrough.

---

## 1) What nnU-Net is (in one sentence)

nnU-Net is an automated segmentation framework that:

1. inspects your dataset,
2. creates a suitable U-Net pipeline automatically,
3. preprocesses data,
4. trains models (2D/3D variants),
5. selects postprocessing,
6. and runs inference.

You provide data in nnU-Net format + labels. nnU-Net does the pipeline engineering.

---

## 2) Mental model: the full pipeline

Think of nnU-Net as 6 stages:

1. **Raw dataset preparation**
2. **Planning + fingerprint extraction**
3. **Preprocessing**
4. **Training**
5. **Validation/model selection/postprocessing selection**
6. **Inference + export**

Each stage writes artifacts that the next stage consumes.

---

## 3) Stage-by-stage with commands

## Stage A — Raw data setup

You place data in nnU-Net structure (`imagesTr`, `labelsTr`, `dataset.json`) and configure environment paths.

Key docs:
- `documentation/dataset_format.md`
- `documentation/setting_up_paths.md`

---

## Stage B — Plan + preprocess entry point

Command most users run first:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

What this does:
- verifies labels/shapes/spacings/classes,
- computes a dataset fingerprint,
- creates experiment plans,
- preprocesses and saves training-ready tensors.

Relevant code:
- CLI: `nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py`
- API: `nnunetv2/experiment_planning/plan_and_preprocess_api.py`
- Integrity checks: `nnunetv2/experiment_planning/verify_dataset_integrity.py`

---

## Stage C — Experiment planning

Planner decides things like:
- target spacing,
- patch size,
- batch size,
- architecture parameters,
- which configurations are valid (`2d`, `3d_fullres`, possibly `3d_lowres` + cascade).

Relevant code area:
- `nnunetv2/experiment_planning/experiment_planners/`
- `nnunetv2/utilities/plans_handling/plans_handler.py`

Output artifact:
- plans JSON files in `nnUNet_preprocessed/<DatasetName>/...`

---

## Stage D — Preprocessing

Preprocessing does (conceptually):
- read image/seg,
- transpose/crop/resample,
- normalization,
- foreground sampling metadata,
- save compressed preprocessed cases.

Relevant code:
- `nnunetv2/preprocessing/preprocessors/default_preprocessor.py`
- `nnunetv2/preprocessing/resampling/`
- `nnunetv2/preprocessing/normalization/`

---

## Stage E — Training

Typical command:

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0
```

What happens internally:
- trainer loads plans + dataset config,
- builds network from plans,
- builds loss (Dice + CE/BCE variants),
- runs train/val loops per epoch,
- logs metrics and checkpoints.

Main trainer code:
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

Related modules:
- losses: `nnunetv2/training/loss/`
- dataloading: `nnunetv2/training/dataloading/`
- augmentation/transforms: in trainer + batchgenerators pipelines
- logging: `nnunetv2/training/logging/`

Outputs:
- checkpoints + logs in `nnUNet_results/...`

---

## Stage F — Inference

Typical command:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f 0
```

Inference uses:
- sliding-window prediction,
- test-time mirroring,
- logits/probabilities resampling back to original geometry,
- segmentation export.

Relevant code:
- `nnunetv2/inference/predict_from_raw_data.py`
- `nnunetv2/inference/sliding_window_prediction.py`
- `nnunetv2/inference/export_prediction.py`

---

## 4) File/folder map you should know

Three environment paths are core:

- `nnUNet_raw` — original datasets in nnU-Net format
- `nnUNet_preprocessed` — fingerprints, plans, and preprocessed tensors
- `nnUNet_results` — model outputs, checkpoints, logs, postprocessing configs

If these are wrong, almost everything breaks or loads wrong artifacts.

---

## 5) How code is organized (developer map)

- `nnunetv2/experiment_planning/` → dataset analysis + planning
- `nnunetv2/preprocessing/` → preprocessing pipeline
- `nnunetv2/training/` → trainers, losses, dataloading, schedulers, logging
- `nnunetv2/inference/` → prediction and export
- `nnunetv2/evaluation/` → metrics + postprocessing selection
- `nnunetv2/postprocessing/` → connected-components/postprocessing logic

If you are studying nnU-Net, read in exactly that order.

---

## 6) What “automatic” really means

nnU-Net does **not** magically learn architecture from scratch. It uses:

- fixed robust defaults,
- rule-based heuristics derived from data fingerprints,
- empirical model/config/postprocessing selection.

So it is best seen as an expert system around U-Net training.

---

## 7) Suggested learning path (7 days)

### Day 1
- Read `readme.md`
- Read `documentation/dataset_format.md`
- Read `documentation/how_to_use_nnunet.md`

### Day 2
- Run one toy dataset with `--verify_dataset_integrity`
- Inspect outputs in `nnUNet_preprocessed`

### Day 3
- Open plans JSON and correlate with dataset properties

### Day 4
- Read `nnUNetTrainer.py` and follow one epoch flow

### Day 5
- Read losses in `training/loss/`
- Understand Dice/CE composition

### Day 6
- Run inference, read `predict_from_raw_data.py`

### Day 7
- Modify one component (for example augmentation or loss), rerun short training, compare

---

## 8) Common beginner confusions

1. **“Where is the model architecture?”**
   - Generated from plans at runtime; not one single static model file.

2. **“Why so many configs (2d, 3d_fullres, etc.)?”**
   - Different dataset properties require different receptive field/memory trade-offs.

3. **“Why does preprocessing matter so much?”**
   - Because training assumptions depend on consistent spacing, normalization, and patch sampling.

4. **“Why can two datasets with same modality behave differently?”**
   - Different shape/spacing/class statistics drive different planned pipelines.

---

## 9) If you want line-by-line training NaN debugging

Use this companion doc:
- `documentation/nan_loss_code_walkthrough.md`

That doc focuses specifically on `train_loss nan` / `val_loss nan` troubleshooting.
