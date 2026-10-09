# Kaggle run

Use a fresh Kaggle session with two T4 GPUs. Set `CODE` to the Kaggle input
folder that contains this training package.

These instructions run the included source. Candidate Roboflow v2 export
records have been recovered, but exact historical run identity, its
environment, and some earlier reported results remain unresolved; see
[experimental evidence and provenance](../../docs/EXPERIMENTS.md) before
interpreting a new run as a reproduction of historical metrics.

```python
CODE = "/kaggle/input/your-training-code"
DATA = "/kaggle/input/your-dataset"
```

Recovered source records identify these candidate datasets:

- [Roboflow `test_augment`, version 2](https://universe.roboflow.com/experiment-coijn/test_augment/dataset/2):
  2,886 train, 120 validation, and 121 test TFRecords in the local export.
- [Roboflow `Nutrition Labels no augs`, version 2](https://universe.roboflow.com/nutrition-yw3l2/nutrition-labels-no-augs/dataset/2):
  300 train, 53 validation, and 45 test TFRecords in the local export.

See the copied [export records and dataset manifest](../../docs/EXPERIMENTS.md#dataset-provenance-recovered-from-the-original-research-folder)
for their source declarations, file hashes, and remaining lineage limits.
These versions and counts identify plausible inputs; the historical metric
JSONs do not prove that these exact files were evaluated.

The earlier project documentation also points to Kaggle
[`test-augment-2`](https://www.kaggle.com/datasets/abhieversor/test-augment-2)
and [`validation-data`](https://www.kaggle.com/datasets/abhieversor/validation-data).
Historical Kaggle version IDs are not recorded, and the raw data is not
included in the repository. Current unauthenticated access to these Kaggle
pointers could not be verified. Use the recovered Roboflow export references
to identify candidate inputs, check dataset terms, and record the exact
version/checksum you actually attach.

Install the dependencies required by quantization-aware training:

```python
!pip install -q -r "$CODE/requirements-kaggle.txt"
```

This requirements file supplies QAT packages and relies on Kaggle's existing
TensorFlow environment; it is not a historical environment lockfile. Retain
the Python, TensorFlow, Keras, GPU/CUDA, and full package versions with each
new run. An installation today may differ from the original environment.

The experiment uses a 128 x 128 input, SSD feature maps `[8, 4, 2, 1]`, and
510 anchors. The teacher uses MobileNetV2 with `alpha=1.0` and 512 FPN
channels; the student uses `alpha=0.35` and 64 FPN channels. The configured
global batch size is 128 on two T4 GPUs. The existing project notes report
this hardware and batch setup; no saved historical environment log confirms
it independently. The configuration describes the included source.

The pipeline calculates its runtime training batch from the GPUs visible to
TensorFlow. A one-GPU run therefore uses a global batch of 64 by default;
reduce `BATCH_SIZE_PER_GPU` if memory requires it. `NUM_GPUS` defines the
configured `GLOBAL_BATCH_SIZE` for code paths that use that constant directly,
so set it to 1 when using those helpers independently. A one-GPU run is useful
for functional reproduction, but its hardware and batch behavior differ from
the reported two-T4 experiment and should not be compared directly with the
reported training metrics.

Set the dataset path explicitly so automatic discovery cannot select another
attached export that happens to share the expected filenames:

```python
%env SSD_DATA_DIR=$DATA
```

Run the preflight check first:

```python
!python "$CODE/pipeline.py" --step validate
```

Run the complete pipeline:

```python
!python "$CODE/pipeline.py" --step all
```

By default, outputs are written to `/kaggle/working`:

- `checkpoints/teacher/best_teacher.weights.h5`
- `checkpoints/student_kd/best_student.weights.h5`
- `checkpoints/pruned/pruned_student.keras`
- `checkpoints/qat/best_qat.weights.h5`
- `nutrition_detector.tflite`
- `test_metrics.json`

To resume a stage, use `--step train_teacher`, `--step distill`,
`--step prune`, or `--step qat`. A resumed stage requires the preceding
artifact to be present.

Each training stage is selected using validation mAP50. The current
configuration uses a patience of 32 epochs and requires an improvement of at
least the larger of `1e-5` absolute mAP50 and `1e-5` times the best mAP50
recorded so far.

Choose model variants and operating thresholds using validation data. Freeze
the chosen settings before evaluating an untouched test split. The primary
Android app's existing 0.30 confidence threshold is a demo setting; its
historical threshold-sweep evidence has not been located.

Evaluate the saved stages on the configured test TFRecord:

```python
!python "$CODE/pipeline.py" --step evaluate
```

This reports AP50/mAP50, precision, recall, F1, TP, FP, and FN for the teacher,
KD student, pruned student, and final INT8 TFLite model. The metrics are saved
to `/kaggle/working/test_metrics.json`.

## External Roboflow dataset

To evaluate a different single-class Roboflow TFRecord export while keeping
the trained artifacts in `/kaggle/working`, run:

```python
EXTERNAL_DATA = "/kaggle/input/validation-data"

!python "$CODE/evaluate_external_dataset.py" \
    --data-dir "$EXTERNAL_DATA" \
    --split test \
    --class-name "nutrition table" \
    --models final
```

Use `--models all` to evaluate the teacher, KD student, pruned student, and
INT8 TFLite model on the same external split. The evaluator treats each valid
box in the external TFRecord as foreground class 1.

For a new validation operating-point check, write to a distinct filename and
record the invocation, dataset version, model hash, and threshold alongside
the result. For example:

```python
!python "$CODE/evaluate_external_dataset.py" \
    --data-dir "$EXTERNAL_DATA" \
    --split valid \
    --models final \
    --score-threshold 0.30 \
    --nms-iou 0.45 \
    --output "/kaggle/working/new_external_valid_score_030.json"
```

This command is an example for a new validation run, not recovered historical
sweep evidence. The current evaluator names precision/recall/F1 JSON fields
`*_at_0.5` even when `--score-threshold` overrides 0.50. Those field names do
not record the threshold actually used. Keep a companion run record and
avoid overwriting the original saved metrics. Once a threshold is selected
on validation data, use the same frozen settings for a separate final test
evaluation.
