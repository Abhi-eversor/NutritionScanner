# NutritionScanner project notes

## Detector used by the Android app

The app asset is the verified v5 model copied from:

`nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/nutrition_detector.tflite`

| Item | Value |
| --- | --- |
| Input | UINT8 `[1, 128, 128, 3]` |
| Output boxes | UINT8 `[1, 510, 4]` |
| Output logits | UINT8 `[1, 510, 2]` |
| Anchors | 510 over feature maps `[8, 4, 2, 1]` |
| NMS IoU | 0.45 |
| App confidence threshold | 0.30 |
| App asset SHA-256 | `1BCC5D406A9469C9545AD9B1BF7DA6E682A09DBF4AA2F391F1460A6656B06E84` |
| App asset size | 813,944 bytes (0.78 MB) |

The app currently uses confidence threshold **0.30** as its existing demo
setting. Earlier notes attributed this value to an external-test sweep with
best F1 0.7312, but the sweep output and its invocation have not been located.
That historical selection claim is unverified. Changing the app display/crop
threshold does not change the AP50 already recorded in an evaluation artifact.
For a new experiment, select an operating threshold on validation data, lock
it, then evaluate on an untouched test set.

## Saved evaluation evidence

The included [pipeline test metrics](nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/test_metrics.json)
contain 121 images and the following mAP@0.5 values:

| Teacher | Student KD | Pruned | TFLite INT8 |
| ---: | ---: | ---: | ---: |
| 0.9382 | 0.9374 | 0.9293 | **0.9237** |

The included [external test metrics](nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/kaggle/working/external_test_metrics.json)
contain only the TFLite model: 45 images, mAP@0.5 **0.7166**, and saved fields
`precision_at_0.5` **0.7742**, `recall_at_0.5` **0.5333**, and `f1_at_0.5`
**0.6316** (TP 24, FP 7, FN 21). These operating-point values disagree with
the earlier notes below. The evaluator defaults to score threshold 0.50,
but its JSON keys remain named `*_at_0.5` when a threshold override is used;
the artifact does not record the invocation or actual threshold. Its field
names alone therefore do not authenticate the historical evaluation settings.

Two matching copies of the external JSON were also found in the original
`VCR_Lab` research folder; both retain F1 0.6316. Recovered Roboflow v2 export
records contain 121 pipeline-test and 45 external-test TFRecords, matching
these reported image counts. The JSON does not identify dataset file hashes,
so this strengthens the candidate dataset association without proving the
exact historical invocation. See the copied provenance records linked below.

See [experimental evidence and provenance](docs/EXPERIMENTS.md) for the
current code configuration, unresolved evidence gaps, and reproduction limits.

## Historical reported results: not fully verified

The table below preserves results from the earlier project notes. All figures
are mAP@0.5 unless otherwise indicated. The pipeline row and external TFLite
test AP50 agree with the included JSON files. Corresponding outputs for the
other external entries have not been located in either this repository or
`VCR_Lab`, so those entries remain historical claims rather than independently
traceable results. The recovered external export does contain 300 train, 53
validation, and 45 test records, supporting the split-count references without
verifying the reported scores.

| Evaluation split | Teacher | Student KD | Pruned | v5 TFLite INT8 |
| --- | ---: | ---: | ---: | ---: |
| Pipeline test set | 0.9382 | 0.9374 | 0.9293 | **0.9237** |
| External train (300 images) | 0.8341 | 0.8024 | 0.7382 | 0.7613 |
| External validation (53 images) | 0.8500 | 0.8651 | 0.7859 | 0.8333 |
| External test (45 images) | 0.8229 | 0.7320 | 0.6710 | **0.7166** |

Earlier notes reported external-test precision 0.7045, recall 0.6889, and F1
0.6966 at threshold 0.50, then precision 0.7083, recall 0.7556, and F1 0.7312
at threshold 0.30. These values are preserved as **unverified historical
claims**. The 0.50 figures conflict with the saved JSON fields, and no sweep
artifact supports the 0.30 figures. The cause is unresolved; neither set has
been silently substituted for the other. The small external test set and
uncertain threshold-selection history do not establish clinical performance.

## Demo flow

1. Open **Profile** before scanning to save dietary preferences, allergies, and optional health flags locally on the device.
2. Capture a photo. Live detection assists framing, but the detector is deliberately run again on the exact high-resolution photo used for OCR.
3. The app crops the best panel, runs bundled on-device OCR, and translates to English locally when necessary.
4. Review and correct the parsed nutrition fields. Only confirmed values, profile flags, and optional reviewed ingredients are sent to Gemini.
5. Gemini returns a JSON-structured, non-diagnostic general nutrition recommendation. It is displayed as typed sections rather than raw model text.

Photos and OCR text are not persisted as scan history. The profile is stored
in local Android DataStore and can be cleared from the Profile screen. The
primary app's manifest enables Android backup; OS backup and device-transfer
behavior have not been verified. Profile flags are also sent to Gemini when
the user requests guidance, as described above.

## Gemini configuration

The demo calls Gemini directly from Android and reads `GEMINI_API_KEY` from
the untracked `local.properties` file. The app sends the key as the
`x-goog-api-key` request header, not in the request URL. Use
`local.properties.example` as the setup template.

Copy `local.properties.example` to `local.properties` and set a valid Google AI Studio key before using the Gemini stage.

## Build

Use Android Studio with JDK 17 and run the `app` configuration, or execute:

```powershell
.\gradlew.bat :app:assembleDebug
```

Android Studio recreates the local SDK path when needed.
