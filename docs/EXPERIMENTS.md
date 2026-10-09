# Experimental evidence and provenance

This document distinguishes saved results, the source code currently included
in the repository, and earlier reported results whose supporting outputs are
missing. Inspecting present-day source files does not prove which source
revision, dependencies, dataset version, or command produced a historical run.
No training or evaluation was rerun during this upload preparation.

See [artifact retention and hashes](ARTIFACTS.md) and the
[archive inventory](archive-manifest.json) for which original outputs remain
in the published repository and which were preserved outside it. Archival
changes file placement; it does not make archived outputs newly verified.

## Included evidence

| Artifact | What it records | Limits |
| --- | --- | --- |
| [MobileNet pipeline test metrics](../nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/test_metrics.json) | Teacher, KD student, pruned student, and TFLite metrics; each reports 121 images and 121 ground-truth objects | No dataset version/checksum, model hash, invocation, source revision, environment, or run timestamp in the JSON |
| [MobileNet external test metrics](../nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/external_test_metrics.json) | Final TFLite metrics; 45 images and 45 ground-truth objects | Only the final model is represented; no threshold sweep or external train/validation evaluation is saved here |
| [Final YOLO training arguments](../yolo26_training_results2/yolo26_combined-5/args.yaml) | Requested YOLO26s configuration and local Kaggle paths | Referenced combined dataset configuration, construction code, and exact training environment are absent |
| [Final YOLO epoch results](../yolo26_training_results2/yolo26_combined-5/results.csv) | Training losses and validation metrics through epoch 127 | Epoch validation columns are not test results or Android device measurements |
| [YOLO exported-model test CSV](../yolo26_training_results2/yolo26_combined-5/weights/test_metrics_int8.csv) | `best_int8.tflite`, `Test (Unseen)`, mAP50 0.9606 and mAP50-95 0.7445 | The CSV does not identify dataset contents, sample count, export/evaluation command, or model hash |

The MobileNet pipeline test AP50 values are teacher **0.9382**, KD student
**0.9374**, pruned **0.9293**, and TFLite INT8 **0.9237**. The external
TFLite AP50 is **0.7166**. These are values recorded in the included files,
not results independently reproduced during repository preparation.

The two Android assets were checked against their corresponding saved exports:

| Runtime asset | Size in bytes | SHA-256 | Matching export |
| --- | ---: | --- | --- |
| `app/src/main/assets/nutrition_detector.tflite` | 813,944 | `1BCC5D406A9469C9545AD9B1BF7DA6E682A09DBF4AA2F391F1460A6656B06E84` | MobileNet run `kaggle/working/nutrition_detector.tflite` |
| `NutritionScannerV5Demo/app/src/main/assets/nutrition_label_yolo26s_int8.tflite` | 10,202,421 | `E3DD5B69C48AD1309A932320F60274E6762DE062B16B46DEB4702084D98E79DF` | Final YOLO run `weights/best_int8.tflite` before archival |

Matching runtime/export hashes establish byte identity between those files.
The metric files lack hashes, so this check cannot establish that those exact
bytes were used for every historical evaluation.

## Unresolved external operating-point results

[Earlier project notes](../MODEL_AND_DEMO_NOTES.md#historical-reported-results-not-fully-verified)
reported external-test precision/recall/F1 **0.7045 / 0.6889 / 0.6966** at
score threshold 0.50. The saved external JSON instead contains fields named
`precision_at_0.5`, `recall_at_0.5`, and `f1_at_0.5` with values
**0.7742 / 0.5333 / 0.6316**. Its counts are TP 24, FP 7, FN 21; therefore
its F1 agrees with `2 * 24 / (2 * 24 + 7 + 21) = 0.6315789474`.

The disagreement is unresolved. Two external JSON copies in the original
`VCR_Lab` research folder (`code/working/` and the v5 run's `kaggle/working/`)
also contain F1 0.6316. They corroborate the stored result but do not identify
the historical invocation or explain the conflicting notes.

The original notes also reported threshold
0.30 precision/recall/F1 **0.7083 / 0.7556 / 0.7312**, claimed best F1 in a
sweep, and listed additional external train/validation and teacher/student
results. No corresponding sweep or missing external-model result files were
located in the repository, surrounding workspace, or subsequent `VCR_Lab`
audit. That audit examined loose notes/notebooks/metrics and 210 bounded text
members across 15 ZIPs. The research folder's `.git` directory was empty,
so no earlier Git history was available there. External evaluation snippets
in its run guides use a `NEW_VERSION` placeholder; they are instructions,
not executed run records.
Those numbers remain preserved as unverified historical claims in the notes.

The current [external evaluator](../nutrition_detector_model_128_07mb_v5/code/evaluate_external_dataset.py)
allows `--score-threshold`, while [evaluation.py](../nutrition_detector_model_128_07mb_v5/code/evaluation.py)
always names the precision/recall/F1 fields `*_at_0.5`, including after an
override. Defaults in current source are score threshold 0.50 and NMS IoU
0.45, but the saved JSON does not record the actual settings or invocation.
Neither matching AP50 nor a field name resolves this provenance gap. AP50
summarizes ranked detections; precision, recall, and F1 describe one operating
point and need not agree across evaluations that share an AP50 value.

The app retains confidence threshold **0.30** and NMS IoU **0.45** as existing
demo settings. Its source comment about a historical sweep is not independent
evidence of that sweep. No new operating-point performance is claimed here.
For a new study, choose model variants and thresholds on validation data,
record the selection rule, freeze settings, then evaluate once on an untouched
test set. A test split used to choose a threshold cannot also provide an
independent test estimate for that selected setting.

## Dataset provenance recovered from the original research folder

The original `VCR_Lab` folder contains two candidate Roboflow exports whose
test split counts match the saved v5 metrics. Their original export READMEs
were copied into this repository; raw TFRecords and dataset images were not.
[dataset-manifest.json](provenance/dataset-manifest.json) records source-file
hashes and record-framing counts from those local TFRecords. The count checks
did not validate protobuf payloads or CRCs; they are not a semantic annotation
audit or a model evaluation.

| Candidate export | Version and exported record | Local train / validation / test TFRecord counts | Copied source records |
| --- | --- | --- | --- |
| [Roboflow `experiment-coijn/test_augment`, version 2](https://universe.roboflow.com/experiment-coijn/test_augment/dataset/2) | Export README dated June 17, 2026, 17:38 GMT; declares 3,127 images | 2,886 / 120 / 121; total 3,127 | [Dataset README](provenance/test_augment_v2/README.dataset.txt), [export README](provenance/test_augment_v2/README.roboflow.txt), [Kaggle upload metadata](provenance/test_augment_v2/kaggle-dataset-metadata.json) |
| [Roboflow `nutrition-yw3l2/nutrition-labels-no-augs`, version 2](https://universe.roboflow.com/nutrition-yw3l2/nutrition-labels-no-augs/dataset/2) | Export README dated June 18, 2026, 16:47 GMT; declares 398 images | 300 / 53 / 45; total 398 | [Dataset README](provenance/nutrition_labels_no_augs_v2/README.dataset.txt), [export README](provenance/nutrition_labels_no_augs_v2/README.roboflow.txt) |

Both export READMEs specify auto-orientation with EXIF stripping, stretch
resize to 640 x 640, and **no export-time image augmentation**. This is a
statement from the recovered export record, not an inference from the
`test_augment` filename. Current MobileNet training adds its own online
augmentations, described below.

The saved `test.ipynb` setup cells in `VCR_Lab` explicitly request version 2
of `nutrition-labels-no-augs` in TFRecord and YOLOv8 formats, and version 2
of `test_augment` in YOLOv8 format. Their outputs concern dataset downloads;
they do not contain an evaluation or threshold sweep. The notebook contains
credential-bearing setup lines and was not copied into this repository.

The 121-record pipeline test export matches the saved metric's 121 images.
The 300/53/45-record external export matches the split sizes in the original
notes and the saved external JSON's 45 images. This identifies plausible
source exports more precisely than the earlier Kaggle pointers. It does not
prove that these exact TFRecord bytes were used for the historical metrics:
the metric JSONs contain no dataset checksums, and an executed v5 run record
linking them has not been located.

The training guide's original Kaggle pointers are
[`abhieversor/test-augment-2`](https://www.kaggle.com/datasets/abhieversor/test-augment-2)
and [`abhieversor/validation-data`](https://www.kaggle.com/datasets/abhieversor/validation-data).
The recovered upload metadata identifies the first slug, but contains no
historical Kaggle dataset version. On October 10, 2026, unauthenticated
requests to these landing pages returned HTTP 404 and metadata/API requests
returned 403. Current public metadata could not be obtained; this does not
establish whether the historical datasets were private, renamed, or removed.

The copied Roboflow dataset READMEs label both exports **CC BY 4.0**, while
the `test_augment` Kaggle upload metadata labels it **CC0-1.0**. This is an
unresolved licensing discrepancy, not evidence that attribution rights were
waived. See [third-party notices](../THIRD_PARTY_NOTICES.md). The export
records identify source projects; they do not supply complete original-image
attribution or establish ownership of every source image.

The MobileNet code expects this TFRecord layout:

```text
dataset-root/
  train/test_augment.tfrecord
  train/test_augment_label_map.pbtxt
  valid/test_augment.tfrecord
  test/test_augment.tfrecord
```

The parser treats every valid box as foreground class 1; the background class
is 0. It clips normalized box coordinates and discards non-finite or empty
boxes. This is a single-class detection pipeline, not a general evaluator for
arbitrary multiclass TFRecords. Expected filenames establish a compatible
input format, not dataset identity.

The recovered export supplies candidate training and validation record counts,
but the included metrics still do not identify the original run's train or
validation files. External train/validation score outputs remain missing.

The repository does not establish complete original image sources,
per-image split manifests, duplicate removal, whether related images were
grouped by product or capture session, or whether augmentation occurred before
or after the upstream split was created. The export READMEs' no-augmentation
statement does not establish how source images were collected or split. It
also does not establish overlap between the training and external datasets.
Do not assume the external data is independent until source-image overlap
is checked.

For YOLO, `args.yaml` references `/kaggle/working/combined/data.yaml`. That
`data.yaml`, the scripts/notebook assembling `combined`, and the exact member
datasets are missing. The available YOLO test CSV and MobileNet JSON files
do not establish identical test sets or matching evaluation implementations;
their scores should not be presented as a controlled head-to-head comparison.

Before distributing dataset-derived example images or publishing a dataset,
record original source attribution and redistribution terms. A project source
license alone does not establish permissions for datasets or pretrained
weights.

## Current MobileNet training configuration

The following settings come from the included [config.py](../nutrition_detector_model_128_07mb_v5/code/config.py)
and pipeline modules. They describe what a new run of this source requests;
they are not a verified historical run manifest.

| Setting | Current value |
| --- | --- |
| Input and classes | 128 x 128 RGB; background 0, nutrition foreground 1 |
| Teacher | MobileNetV2 alpha 1.0; 512 FPN channels |
| Student | MobileNetV2 alpha 0.35; 64 FPN channels |
| Backbone features | `block_12_project_BN`, `out_relu` |
| SSD feature maps and anchors | `[8, 4, 2, 1]`; 510 anchors; aspect ratios `[1, 2, 0.5, 3, 1/3]` plus one extra anchor per location; scales 0.2 to 0.9 |
| Anchor matching and box variances | Positive IoU 0.5; negative IoU 0.3; variances `[0.1, 0.1, 0.2, 0.2]` |
| Batch | 64 per visible replica; configured `NUM_GPUS=2` and `GLOBAL_BATCH_SIZE=128`; runtime pipeline derives replica count |
| Teacher | At most 320 epochs; base learning rate `1.5e-3` |
| KD student | At most 240 epochs; base learning rate `1.5e-3`; EMA decay 0.999 |
| Distillation | Temperature 4.0; hard/soft/feature loss weights 1.0/1.0/0.5; three feature levels |
| Structured pruning | Expansion-channel prune ratio 0.10; minimum eight expansion channels |
| Pruning fine-tune | At most 60 epochs; base learning rate `1e-4` |
| QAT | At most 40 epochs; base learning rate `5e-5`; 100 calibration images |
| Loss | Focal alpha 0.25, gamma 2.0; localization weight 1.0 |
| Warmup and optimization | Adam with warmup/cosine scheduling; 20 warmup epochs for teacher/KD/pruning, two for QAT under the current default epoch limits |
| Checkpoint selection | Validation mAP50; patience 32; improvement threshold is max(`1e-5`, best mAP50 x `1e-5`); validation metric batch 32 |
| Evaluation | Match IoU 0.5; default operating score 0.5; AP minimum score 0.001; NMS IoU 0.45; at most 100 detections per image |

Stages are teacher training, student knowledge distillation, structured
pruning/fine-tuning, then quantization-aware training and TFLite conversion.
The KD stage validates EMA student weights. Pruning prefers
`best_student.weights.h5`, with `final_student_ema.weights.h5` as a fallback
when the preferred file is absent. QAT requires `pruned_student.keras`.
The converter requests INT8 operations with UINT8 model input/output tensors.
The primary app can use its included runtime asset without retraining.

Current online training augmentations are random crop (probability 0.25),
horizontal flip (0.5), and color jitter (0.5). Crop sampling uses minimum
object coverage 0.5, aspect ratio 0.5 to 2.0, and area 0.3 to 1.0; boxes are
adjusted with the image. Jitter uses brightness delta 32, contrast/saturation
0.8 to 1.2, and hue delta 0.05. Images are resized to 128 x 128 and normalized
to [-1, 1]. Validation/evaluation adds no random online augmentation. This
does not rule out transformations already present in exported TFRecords.

The training dataset is shuffled with a 2,048-record buffer and reshuffled
each iteration; incomplete training batches are dropped. No explicit global
random seed is configured in the included MobileNet source. GPU count,
dataset order, augmentation, dependency versions, and nondeterministic
execution can affect results. A one-GPU run is useful for checking the
pipeline but uses a different default global batch than the reported two-T4
setup and should not be expected to reproduce exact historical metrics.

## Current YOLO evidence and reproduction limits

The final run's `args.yaml` requests pretrained `yolo26s.pt`, 640-pixel input,
200 epochs, patience 30, batch 64, devices `0,1`, automatic optimizer,
seed 0, deterministic mode, and cosine learning-rate scheduling. It records
NMS IoU 0.7 and validation split `val`. The epoch CSV has rows 1 through 127;
the requested 200-epoch maximum does not mean 200 epochs were completed.
The optional app uses confidence threshold 0.49 and NMS IoU 0.70.

Training arguments alone do not reproduce the benchmark. The exact
Ultralytics/package version, original training notebook/commands, combined
dataset inputs, model-export command, quantization/calibration settings, and
test-evaluation command are missing. The exported-model filename does not
establish those settings. Archived checkpoints and excess generated outputs
may help later analysis, but do not fill these missing provenance records.

## Environment and measurement limits

The existing guide reports two T4 GPUs. No saved environment lockfile or
executed v5 notebook log independently verifies the original setup.
`requirements-kaggle.txt` currently lists
`tensorflow-model-optimization>=0.8.0` and unpinned `tf_keras`; it relies on
TensorFlow and other packages already present in the Kaggle environment.
Current source uses legacy Keras compatibility. Installing those requirements
today does not reconstruct the historical training environment.

`VCR_Lab/code/recovered_report_data/` and its unexecuted report-generation
notebook are constructed summaries for a **different** experiment: 256 x 256
input, 2,728 anchors, 6,363 train / 528 validation / 387 test images, global
batch 256, and reported TensorFlow 2.20.0. The summaries cite a separate
console log and previously supplied test results; they report final INT8
AP50 0.9545 and F1 0.9449. They are not original executed v5 notebook output,
and neither those results nor that environment is attributed to the deployed
128 x 128 model. Those materials were not added to this repository as v5
evidence.

For each new run, retain the dataset versions/checksums and split manifest,
source revision, complete dependency snapshot, Python/TensorFlow/Keras and
GPU/CUDA details, random seeds, invocation and stdout, evaluation thresholds,
per-image predictions, and model SHA-256. Preserve them beside new outputs
instead of overwriting historical metric files.

The current MobileNet evaluator times the predictor call and then performs
postprocessing separately. Its `mean_inference_ms` is a mean per predictor
batch call in that execution environment; the JSON lacks hardware, delegate,
warmup, and invocation details. Values such as the saved TFLite 2.17 ms are
not Android camera-to-result latency, OCR latency, or measured device
benchmarks. The detector's small test splits, incomplete provenance, and
optional Gemini guidance do not establish medical or clinical effectiveness.
Detection AP50 also does not measure OCR accuracy, nutrient-field extraction
accuracy, translation quality, or the correctness of generated guidance.
