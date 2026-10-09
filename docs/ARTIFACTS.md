# Artifact inventory and archive policy

Both Android applications include the exact runtime detector they load. No
training is needed to build either app. MobileNet training, distillation,
pruning, QAT, evaluation, and Android deployment code remain in the repository.
Original metric JSONs and result CSVs were preserved byte-for-byte.

## Retained deployment models

| Model | Repository path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| MobileNetV2-SSD INT8 | `app/src/main/assets/nutrition_detector.tflite` | 813,944 | `1bcc5d406a9469c9545ad9b1bf7da6e682a09dbf4aa2f391f1460a6656b06e84` |
| YOLO26s benchmark export | `NutritionScannerV5Demo/app/src/main/assets/nutrition_label_yolo26s_int8.tflite` | 10,202,421 | `e3dd5b69c48ad1309a932320f60274e6762de062b16b46deb4702084d98e79df` |

The primary model matches the retained experiment export exactly. The YOLO
app model matches the original `yolo26_combined-5/weights/best_int8.tflite`
export exactly; that additional copy is archived. The older
`yolo26_combined/weights/best_int8.tflite` is a different model with the same
file size, so its hash is recorded separately in the archive manifest.
A matching export hash establishes file identity, not the export command,
checkpoint lineage, or historical training environment.

## Retained research evidence

- All MobileNet stage artifacts under
  `nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/`,
  including teacher, KD student, pruned student, QAT checkpoints, the final
  TFLite export, and the two supplied metric JSONs.
- YOLO `args.yaml`, `results.csv`, analytical plots, class-distribution plots,
  and confusion matrices from the supplied runs, including incomplete-run
  metadata. Earlier run metadata remains useful even when its weights are
  archived.
- `yolo26_training_results2/yolo26_combined-5/weights/test_metrics_int8.csv`.
  Its original location and contents are retained despite the adjacent
  checkpoint binaries being excluded.

[artifact-manifest.json](artifact-manifest.json) records hashes and sizes of
retained deployment models, MobileNet experiment artifacts, and YOLO reports.
[EXPERIMENTS.md](EXPERIMENTS.md) explains what the supplied results establish
and which run details remain unknown.

## Local archive

Preparation on 2026-10-10 moved **825 files / 618.378 MiB**
to the sibling directory
`../NutritionScanner-preparation-archive-2026-10-10/`.

The archive preserves:

- both bulk YOLO prediction directories;
- additional YOLO `.pt` checkpoints and TFLite exports;
- training and validation image montages, whose source-image redistribution
  provenance is not established.

Every moved file was hashed before moving and verified afterward. Files were
preserved rather than deleted or deduplicated in place. This includes
byte-identical `best.pt`/`last.pt` copies in the older runs.

[archive-manifest.json](archive-manifest.json) lists each original
repository-relative path, size, SHA-256, and archive reason. A second copy of
this manifest is in the local archive. That directory is a local backup;
it is neither tracked by this repository nor a published download. No public
archive link or redistribution permission is implied.

To recover an artifact on the preparation machine, take its `path` from the
manifest, append it to the sibling archive directory, verify the listed hash,
and copy it to a separate experiment workspace. Android apps already have
their required models. Re-evaluating an archived YOLO checkpoint requires
access to the local archive or a future separately published research archive.
Do not re-add all archived files to Git merely to build the applications.

The root `.gitignore` excludes future build outputs, SDK/key configuration,
Python-generated outputs, extra YOLO weights, prediction renders, and image
montages. It deliberately allows the saved MobileNet checkpoints, reports,
and the two app assets.
